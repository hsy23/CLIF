from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - portable policy tests still run without torch
    torch = None
    nn = None


@unittest.skipIf(torch is None, "torch is not installed")
class ConcurrentVLLMLoRATests(unittest.TestCase):
    def test_lora_injection_updates_only_adapter_parameters(self) -> None:
        from engine.concurrent_vllm_lora import inject_lora, trainable_parameters

        class TinyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_proj = nn.Linear(4, 4)
                self.v_proj = nn.Linear(4, 2)
                self.other = nn.Linear(4, 4)

            def forward(self, inputs):
                return self.q_proj(inputs).sum() + self.v_proj(inputs).sum()

        torch.manual_seed(1)
        model = TinyModel()
        model.requires_grad_(False)
        injected = inject_lora(model, rank=2, alpha=4.0)
        self.assertEqual(injected, ("q_proj", "v_proj"))
        parameters = trainable_parameters(model)
        self.assertEqual(len(parameters), 4)
        before = [parameter.detach().clone() for parameter in parameters]
        optimizer = torch.optim.AdamW(parameters, lr=0.1)
        loss = model(torch.randn(3, 4))
        loss.backward()
        optimizer.step()
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, parameters, strict=True)))
        self.assertFalse(any(parameter.requires_grad for parameter in model.other.parameters()))

    def test_overlap_counts_require_real_interval_intersection(self) -> None:
        from engine.concurrent_vllm_lora import count_request_step_overlaps

        requests = [
            {"started_wall_s": 1.0, "finished_wall_s": 2.0},
            {"started_wall_s": 4.0, "finished_wall_s": 5.0},
        ]
        steps = [
            {"started_wall_s": 1.5, "finished_wall_s": 3.0},
            {"started_wall_s": 6.0, "finished_wall_s": 7.0},
        ]
        self.assertEqual(count_request_step_overlaps(requests, steps), (1, 1))

    def test_export_uses_vllm_peft_weight_names(self) -> None:
        from safetensors import safe_open

        from engine.concurrent_vllm_lora import export_vllm_lora_adapter, inject_lora

        class TinyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_proj = nn.Linear(4, 4)

        model = TinyModel()
        model.requires_grad_(False)
        inject_lora(model, target_names=("q_proj",), rank=2, alpha=4.0)
        with TemporaryDirectory() as directory:
            metadata = export_vllm_lora_adapter(
                model,
                directory,
                base_model="local-model",
                rank=2,
                alpha=4.0,
            )
            self.assertEqual(metadata["tensors"], 2)
            with safe_open(f"{directory}/adapter_model.safetensors", framework="pt") as tensors:
                self.assertEqual(
                    set(tensors.keys()),
                    {
                        "base_model.model.q_proj.lora_A.weight",
                        "base_model.model.q_proj.lora_B.weight",
                    },
                )


if __name__ == "__main__":
    unittest.main()
