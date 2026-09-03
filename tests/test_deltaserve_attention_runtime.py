from __future__ import annotations

import unittest
from types import SimpleNamespace

try:
    import torch
except ImportError:  # pragma: no cover - portable policy tests still run without torch
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class DeltaServeAttentionRuntimeTests(unittest.TestCase):
    def test_qwen3_packed_qkv_and_gate_up_views_match_hf_shapes(self) -> None:
        from engine.deltaserve_attention_runtime import _resolve_base_tensor

        config = SimpleNamespace(
            hidden_size=4,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=2,
            intermediate_size=6,
            tie_word_embeddings=False,
        )
        imported = {
            "model.layers.0.self_attn.qkv_proj.weight": SimpleNamespace(
                tensor=torch.arange(16 * 4, dtype=torch.float32).reshape(16, 4)
            ),
            "model.layers.0.mlp.gate_up_proj.weight": SimpleNamespace(
                tensor=torch.arange(12 * 4, dtype=torch.float32).reshape(12, 4)
            ),
        }
        q = _resolve_base_tensor("model.layers.0.self_attn.q_proj.weight", imported, config)
        k = _resolve_base_tensor("model.layers.0.self_attn.k_proj.weight", imported, config)
        v = _resolve_base_tensor("model.layers.0.self_attn.v_proj.weight", imported, config)
        gate = _resolve_base_tensor("model.layers.0.mlp.gate_proj.weight", imported, config)
        up = _resolve_base_tensor("model.layers.0.mlp.up_proj.weight", imported, config)
        self.assertEqual(tuple(q.shape), (4, 4))
        self.assertEqual(tuple(k.shape), (2, 4))
        self.assertEqual(tuple(v.shape), (2, 4))
        self.assertEqual(tuple(gate.shape), (6, 4))
        self.assertEqual(tuple(up.shape), (6, 4))
        self.assertTrue(torch.equal(v, imported["model.layers.0.self_attn.qkv_proj.weight"].tensor[6:8]))
        self.assertTrue(torch.equal(up, imported["model.layers.0.mlp.gate_up_proj.weight"].tensor[6:]))

    def test_variable_length_rows_are_padded_and_loss_masked(self) -> None:
        from engine.deltaserve_attention_runtime import _make_padded_batch

        flat = torch.tensor([10, 11, 12, 20, 21], dtype=torch.long)
        batch, mask, labels = _make_padded_batch(flat, [3, 2], pad_token_id=0)
        self.assertEqual(batch.tolist(), [[10, 11, 12], [20, 21, 0]])
        self.assertEqual(mask.tolist(), [[1, 1, 1], [1, 1, 0]])
        self.assertEqual(labels.tolist(), [[10, 11, 12], [20, 21, -100]])

    def test_qkv_hook_accepts_keyword_input(self) -> None:
        from engine.deltaserve_attention_runtime import DeltaServeQwen3AttentionRuntime

        class KeywordQKV(torch.nn.Module):
            def forward(self, hidden_states=None):
                return torch.zeros(
                    (*hidden_states.shape[:-1], 4),
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )

        runtime = DeltaServeQwen3AttentionRuntime.__new__(
            DeltaServeQwen3AttentionRuntime
        )
        runtime.adapter_ready = True
        state = SimpleNamespace(
            q_size=2,
            kv_size=1,
            q_lora_a=SimpleNamespace(tensor=torch.tensor([[1.0, 0.0]])),
            q_lora_b=SimpleNamespace(tensor=torch.tensor([[1.0], [2.0]])),
            v_lora_a=SimpleNamespace(tensor=torch.tensor([[0.0, 1.0]])),
            v_lora_b=SimpleNamespace(tensor=torch.tensor([[3.0]])),
        )
        module = KeywordQKV()
        module.register_forward_hook(
            runtime._make_qkv_hook(state),
            with_kwargs=True,
        )
        output = module(hidden_states=torch.tensor([[[2.0, 4.0]]]))
        self.assertEqual(tuple(output.shape), (1, 1, 4))
        self.assertTrue(torch.equal(output[..., :2], torch.tensor([[[2.0, 4.0]]])))
        self.assertTrue(torch.equal(output[..., 2], torch.zeros((1, 1))))
        self.assertTrue(torch.equal(output[..., 3], torch.tensor([[12.0]])))

    def test_replayed_inflight_training_batch_is_not_submitted_twice(self) -> None:
        from engine.deltaserve_attention_runtime import DeltaServeQwen3AttentionRuntime

        runtime = DeltaServeQwen3AttentionRuntime.__new__(DeltaServeQwen3AttentionRuntime)
        runtime._inflight_training_req_ids = ("deltaserve-ft-0", "deltaserve-ft-1")
        self.assertTrue(
            runtime._is_inflight_replay(
                ["deltaserve-ft-0", "deltaserve-ft-1", "inference-0"],
                [0, 1],
            )
        )
        self.assertFalse(
            runtime._is_inflight_replay(
                ["deltaserve-ft-0", "deltaserve-ft-new"],
                [0, 1],
            )
        )

    def test_meta_qwen3_graph_runs_after_packed_base_binding(self) -> None:
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

        from engine.concurrent_vllm_lora import inject_lora, trainable_parameters
        from engine.deltaserve_attention_runtime import (
            _resolve_base_tensor,
            _set_parameter,
        )

        config = Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            head_dim=4,
            tie_word_embeddings=False,
        )
        with torch.device("meta"):
            model = Qwen3ForCausalLM(config)
        imported = {}
        for name, parameter in model.named_parameters():
            if ".self_attn." in name and any(
                name.endswith(f".self_attn.{projection}.weight")
                for projection in ("q_proj", "k_proj", "v_proj")
            ):
                source = name.split(".self_attn.")[0] + ".self_attn.qkv_proj.weight"
                imported[source] = SimpleNamespace(
                    tensor=torch.randn((32, 16), dtype=torch.float32)
                )
            elif name.endswith(".mlp.gate_proj.weight") or name.endswith(".mlp.up_proj.weight"):
                source = name.split(".mlp.")[0] + ".mlp.gate_up_proj.weight"
                imported[source] = SimpleNamespace(
                    tensor=torch.randn((64, 16), dtype=torch.float32)
                )
            else:
                imported[name] = SimpleNamespace(
                    tensor=torch.randn(tuple(parameter.shape), dtype=torch.float32)
                )

        model = model.to_empty(device="cpu")
        model.model.rotary_emb = Qwen3RotaryEmbedding(config, device="cpu")
        for name, parameter in list(model.named_parameters()):
            tensor = _resolve_base_tensor(name, imported, config)
            self.assertEqual(tuple(tensor.shape), tuple(parameter.shape))
            _set_parameter(model, name, tensor, requires_grad=False)
        injected = inject_lora(model, rank=2, alpha=4.0)
        model.train()
        input_ids = torch.randint(0, config.vocab_size, (1, 8))
        loss = model(input_ids=input_ids, labels=input_ids, use_cache=False).loss
        loss.backward()
        self.assertEqual(len(injected), 4)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(parameter.grad is not None for parameter in trainable_parameters(model)))


if __name__ == "__main__":
    unittest.main()
