"""Live Qwen3 attention-LoRA runtime for the patched vLLM runner.

The host keeps vLLM as the inference owner.  Its Qwen3 packed ``qkv_proj``
modules receive a no-grad LoRA delta from shared adapter tensors.  A persistent
worker constructs a meta-initialized Transformers Qwen3 graph, binds every
frozen base parameter to the host's CUDA VMM allocations, and runs the real
q/v LoRA backward step against those same allocations.

This is a correctness-first G6 path.  The worker recomputes the training row
for autograd because vLLM's ``execute_model`` is inference-only; it does not
claim the final DeltaServe activation-reuse performance until that recompute
is optimized.
"""

from __future__ import annotations

import atexit
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.multiprocessing as mp
from torch import nn
from torch.nn import functional as F

from .concurrent_vllm_lora import inject_lora, trainable_parameters
from .cuda_vmm import ExportableCudaTensor, ImportedCudaTensor
from .deltaserve_vllm_runtime import TRAINING_REQUEST_PREFIX, _append_trace


TARGET_MODULES = ("q_proj", "v_proj")


def _set_parameter(root: nn.Module, name: str, tensor: torch.Tensor, *, requires_grad: bool) -> None:
    parent_name, leaf_name = name.rsplit(".", 1)
    parent = root
    for part in parent_name.split("."):
        parent = getattr(parent, part)
    if leaf_name not in parent._parameters:
        raise KeyError(f"{name} is not a parameter in the constructed Qwen3 graph")
    parent._parameters[leaf_name] = nn.Parameter(tensor, requires_grad=requires_grad)


def _get_module(root: nn.Module, name: str) -> nn.Module:
    module = root
    for part in name.split("."):
        module = getattr(module, part)
    return module


def _qwen3_head_sizes(config) -> tuple[int, int]:
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    return (
        int(config.num_attention_heads * head_dim),
        int(config.num_key_value_heads * head_dim),
    )


def _resolve_base_tensor(name: str, imported: dict[str, ImportedCudaTensor], config) -> torch.Tensor:
    """Resolve an HF parameter from vLLM's exact or packed parameter names."""

    if name in imported:
        return imported[name].tensor

    q_size, kv_size = _qwen3_head_sizes(config)
    if ".self_attn." in name and name.endswith(".weight"):
        for projection, start, width in (
            ("q_proj", 0, q_size),
            ("k_proj", q_size, kv_size),
            ("v_proj", q_size + kv_size, kv_size),
        ):
            suffix = f".self_attn.{projection}.weight"
            if name.endswith(suffix):
                source = name[: -len(suffix)] + ".self_attn.qkv_proj.weight"
                if source not in imported:
                    raise KeyError(f"missing packed vLLM parameter {source} for {name}")
                return imported[source].tensor[start : start + width]

    if ".self_attn." in name and name.endswith(".bias"):
        for projection, start, width in (
            ("q_proj", 0, q_size),
            ("k_proj", q_size, kv_size),
            ("v_proj", q_size + kv_size, kv_size),
        ):
            suffix = f".self_attn.{projection}.bias"
            if name.endswith(suffix):
                source = name[: -len(suffix)] + ".self_attn.qkv_proj.bias"
                if source not in imported:
                    raise KeyError(f"missing packed vLLM parameter {source} for {name}")
                return imported[source].tensor[start : start + width]

    for projection, start in (("gate_proj", 0), ("up_proj", int(config.intermediate_size))):
        suffix = f".mlp.{projection}.weight"
        if name.endswith(suffix):
            source = name[: -len(suffix)] + ".mlp.gate_up_proj.weight"
            if source not in imported:
                raise KeyError(f"missing packed vLLM parameter {source} for {name}")
            return imported[source].tensor[start : start + int(config.intermediate_size)]

    if name == "lm_head.weight" and getattr(config, "tie_word_embeddings", False):
        tied_name = "model.embed_tokens.weight"
        if tied_name in imported:
            return imported[tied_name].tensor

    available = ", ".join(sorted(imported)[:8])
    raise KeyError(f"cannot map HF parameter {name}; first vLLM names: {available}")


def _make_padded_batch(
    flat_input_ids: torch.Tensor,
    row_lengths: list[int],
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not row_lengths or any(length < 2 for length in row_lengths):
        raise ValueError(f"training rows must contain at least two tokens: {row_lengths}")
    if sum(row_lengths) > flat_input_ids.numel():
        raise ValueError("training row lengths exceed the shared input buffer")

    max_length = max(row_lengths)
    batch = flat_input_ids.new_full((len(row_lengths), max_length), pad_token_id)
    attention_mask = torch.zeros(
        (len(row_lengths), max_length), dtype=torch.long, device=flat_input_ids.device
    )
    labels = torch.full_like(batch, -100)
    offset = 0
    for row_index, length in enumerate(row_lengths):
        row = flat_input_ids[offset : offset + length]
        batch[row_index, :length] = row
        attention_mask[row_index, :length] = 1
        labels[row_index, :length] = row
        offset += length
    return batch, attention_mask, labels


def _attention_worker_main(
    base_specs: dict[str, dict],
    adapter_specs: dict[str, dict],
    input_ids_spec: dict,
    model_config: dict,
    command_queue,
    result_queue,
    trace_path: str,
    rank: int,
    alpha: float,
    learning_rate: float,
) -> None:
    """Map shared parameters and execute real Qwen3 q/v LoRA updates."""

    torch.cuda.set_device(0)
    imported_base: dict[str, ImportedCudaTensor] = {}
    imported_adapters: dict[str, ImportedCudaTensor] = {}
    imported_inputs = None
    try:
        imported_base = {
            name: ImportedCudaTensor.from_spec(spec) for name, spec in base_specs.items()
        }
        imported_adapters = {
            name: ImportedCudaTensor.from_spec(spec) for name, spec in adapter_specs.items()
        }
        imported_inputs = ImportedCudaTensor.from_spec(input_ids_spec)

        from transformers import Qwen3Config, Qwen3ForCausalLM
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

        config = Qwen3Config.from_dict(model_config)
        if hasattr(config, "_attn_implementation"):
            config._attn_implementation = "eager"
        config.use_cache = False
        with torch.device("meta"):
            model = Qwen3ForCausalLM(config)
        model = model.to_empty(device="cuda")
        # ``to_empty`` materializes parameters and buffers without values.  The
        # rotary inverse-frequency buffer is not a checkpoint parameter, so
        # rebuild that small buffer explicitly before the first worker forward.
        model.model.rotary_emb = Qwen3RotaryEmbedding(config, device="cuda")

        mapped_names: set[str] = set()
        for name, parameter in list(model.named_parameters()):
            tensor = _resolve_base_tensor(name, imported_base, config)
            if tuple(tensor.shape) != tuple(parameter.shape):
                raise RuntimeError(
                    f"shape mismatch for {name}: shared {tuple(tensor.shape)} vs HF {tuple(parameter.shape)}"
                )
            _set_parameter(model, name, tensor, requires_grad=False)
            mapped_names.add(name)

        injected = inject_lora(model, target_names=TARGET_MODULES, rank=rank, alpha=alpha)
        if len(injected) != int(config.num_hidden_layers) * len(TARGET_MODULES):
            raise RuntimeError(
                f"expected {int(config.num_hidden_layers) * len(TARGET_MODULES)} q/v modules, got {len(injected)}"
            )
        for module_name in injected:
            module = _get_module(model, module_name)
            for suffix, attribute in (("lora_a", "lora_a"), ("lora_b", "lora_b")):
                key = f"{module_name}.{suffix}"
                if key not in imported_adapters:
                    raise KeyError(f"missing shared adapter allocation {key}")
                setattr(
                    module,
                    attribute,
                    nn.Parameter(imported_adapters[key].tensor, requires_grad=True),
                )

        model.train()
        parameters = trainable_parameters(model)
        if len(parameters) != len(injected) * 2:
            raise RuntimeError(f"unexpected trainable parameter count: {len(parameters)}")
        optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
        _append_trace(
            trace_path,
            {
                "event": "attention_worker_initialized",
                "pid": os.getpid(),
                "device": torch.cuda.get_device_name(0),
                "base_parameter_count": len(imported_base),
                "mapped_parameter_count": len(mapped_names),
                "adapter_module_count": len(injected),
                "trainable_parameter_count": sum(parameter.numel() for parameter in parameters),
                "target_modules": list(TARGET_MODULES),
            },
        )

        pad_token_id = int(getattr(config, "pad_token_id", None) or 0)
        while True:
            command = command_queue.get()
            if command.get("op") == "close":
                break
            if command.get("op") != "backward":
                raise RuntimeError(f"unexpected attention worker command: {command}")
            job_id = int(command["job_id"])
            row_lengths = [int(value) for value in command["row_lengths"]]
            try:
                before = [parameter.detach().clone() for parameter in parameters]
                input_ids = imported_inputs.tensor
                batch, attention_mask, labels = _make_padded_batch(
                    input_ids, row_lengths, pad_token_id
                )
                optimizer.zero_grad(set_to_none=True)
                _append_trace(
                    trace_path,
                    {
                        "event": "backward_started",
                        "job_id": job_id,
                        "pid": os.getpid(),
                        "token_count": sum(row_lengths),
                        "training_row_count": len(row_lengths),
                        "target_modules": list(TARGET_MODULES),
                    },
                )
                started = time.time()
                outputs = model(
                    input_ids=batch,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False,
                )
                loss = outputs.loss.float()
                loss.backward()
                gradients_finite = all(
                    parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
                    for parameter in parameters
                )
                if not torch.isfinite(loss).item() or not gradients_finite:
                    raise FloatingPointError("non-finite attention LoRA loss or gradient")
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
                optimizer.step()
                torch.cuda.synchronize()
                finished = time.time()
                parameter_delta_l1 = sum(
                    float((parameter.detach() - old).abs().sum().cpu())
                    for parameter, old in zip(parameters, before, strict=True)
                )
                result = {
                    "event": "backward_finished",
                    "job_id": job_id,
                    "pid": os.getpid(),
                    "loss": float(loss.detach().cpu()),
                    "grad_norm": float(grad_norm.detach().cpu()),
                    "parameter_delta_l1": parameter_delta_l1,
                    "training_token_count": sum(row_lengths),
                    "training_loss_token_count": sum(max(length - 1, 0) for length in row_lengths),
                    "training_row_count": len(row_lengths),
                    "training_row_token_counts": row_lengths,
                    "elapsed_s": finished - started,
                    "base_parameter_count": len(imported_base),
                    "target_modules": list(TARGET_MODULES),
                    "gradients_finite": gradients_finite,
                    "loss_finite": bool(torch.isfinite(loss).item()),
                }
                _append_trace(trace_path, result)
                result_queue.put(result)
            except Exception as exc:
                error = {
                    "event": "backward_error",
                    "job_id": job_id,
                    "pid": os.getpid(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                _append_trace(trace_path, error)
                result_queue.put(error)
                raise
    finally:
        for allocation in (*imported_base.values(), *imported_adapters.values()):
            allocation.close()
        if imported_inputs is not None:
            imported_inputs.close()


class DeltaServeQwen3AttentionRuntime:
    """Shared-base live q/v LoRA runtime selected by the vLLM patch."""

    def __init__(self, runner) -> None:
        if runner.parallel_config.tensor_parallel_size != 1:
            raise RuntimeError("attention LoRA prototype requires tensor_parallel_size=1")
        self.runner = runner
        self.model = runner.model
        if getattr(self.model.config, "model_type", None) != "qwen3":
            raise RuntimeError("live attention runtime currently targets Qwen3 only")
        self.trace_path = os.environ["CLIF_DELTASERVE_TRACE"]
        Path(self.trace_path).parent.mkdir(parents=True, exist_ok=True)
        self.rank = int(os.environ.get("CLIF_DELTASERVE_RANK", "4"))
        self.alpha = float(os.environ.get("CLIF_DELTASERVE_ALPHA", "8"))
        self.learning_rate = float(os.environ.get("CLIF_DELTASERVE_LR", "3e-2"))
        self.max_training_tokens = int(os.environ.get("CLIF_DELTASERVE_MAX_TOKENS", "128"))
        self.max_training_steps = int(os.environ.get("CLIF_DELTASERVE_MAX_STEPS", "1"))
        self.adapter_ready = False
        self._active = None
        self._inflight_job_id = None
        self._inflight_training_req_ids = ()
        self._submitted_steps = 0
        self._job_id = 0
        self._closed = False

        self.base_allocations: dict[str, ExportableCudaTensor] = {}
        for name, parameter in self.model.named_parameters():
            allocation = ExportableCudaTensor(parameter.shape, parameter.dtype)
            allocation.tensor.copy_(parameter.detach())
            parameter.data = allocation.tensor
            self.base_allocations[name] = allocation
        torch.cuda.synchronize()

        q_size, kv_size = _qwen3_head_sizes(self.model.config)
        hidden_size = int(self.model.config.hidden_size)
        self.adapters: dict[str, SimpleNamespace] = {}
        for name, module in self.model.named_modules():
            if not name.endswith(".self_attn.qkv_proj"):
                continue
            prefix = name[: -len(".qkv_proj")]
            actual_q = int(getattr(module, "q_size", q_size))
            actual_kv = int(getattr(module, "kv_size", kv_size))
            if (actual_q, actual_kv) != (q_size, kv_size):
                raise RuntimeError(
                    f"unsupported heterogeneous Qwen3 head sizes at {name}: {actual_q}/{actual_kv}"
                )
            self.adapters[prefix] = SimpleNamespace(
                q_size=actual_q,
                kv_size=actual_kv,
                q_lora_a=ExportableCudaTensor((self.rank, hidden_size), torch.float32),
                q_lora_b=ExportableCudaTensor((actual_q, self.rank), torch.float32),
                v_lora_a=ExportableCudaTensor((self.rank, hidden_size), torch.float32),
                v_lora_b=ExportableCudaTensor((actual_kv, self.rank), torch.float32),
            )
        if len(self.adapters) != int(self.model.config.num_hidden_layers):
            raise RuntimeError(f"expected one qkv projection per layer, got {len(self.adapters)}")
        torch.manual_seed(13)
        for state in self.adapters.values():
            state.q_lora_a.tensor.normal_(mean=0.0, std=0.02)
            state.q_lora_b.tensor.zero_()
            state.v_lora_a.tensor.normal_(mean=0.0, std=0.02)
            state.v_lora_b.tensor.zero_()

        self.training_input_ids = ExportableCudaTensor(
            (self.max_training_tokens,), torch.int64
        )
        context = mp.get_context("spawn")
        self.command_queue = context.SimpleQueue()
        self.result_queue = context.SimpleQueue()
        base_specs = {
            name: allocation.export_spec()
            for name, allocation in self.base_allocations.items()
        }
        adapter_specs = {}
        for prefix, state in self.adapters.items():
            adapter_specs[f"{prefix}.q_proj.lora_a"] = state.q_lora_a.export_spec()
            adapter_specs[f"{prefix}.q_proj.lora_b"] = state.q_lora_b.export_spec()
            adapter_specs[f"{prefix}.v_proj.lora_a"] = state.v_lora_a.export_spec()
            adapter_specs[f"{prefix}.v_proj.lora_b"] = state.v_lora_b.export_spec()
        self.backward_process = context.Process(
            target=_attention_worker_main,
            args=(
                base_specs,
                adapter_specs,
                self.training_input_ids.export_spec(),
                self.model.config.to_dict(),
                self.command_queue,
                self.result_queue,
                self.trace_path,
                self.rank,
                self.alpha,
                self.learning_rate,
            ),
        )
        self.backward_process.start()
        self.hook_handles = []
        for name, module in self.model.named_modules():
            if name not in self.adapters:
                continue
            self.hook_handles.append(
                module.register_forward_hook(
                    self._make_qkv_hook(self.adapters[name]),
                    with_kwargs=True,
                )
            )
        atexit.register(self.close)
        _append_trace(
            self.trace_path,
            {
                "event": "runtime_initialized",
                "pid": os.getpid(),
                "backward_pid": self.backward_process.pid,
                "base_model_replaced_with_shared_vmm": True,
                "shared_base_allocation_count": len(self.base_allocations),
                "base_allocation_ids": [
                    allocation.allocation_id for allocation in self.base_allocations.values()
                ],
                "shared_base_vmm_bytes": sum(
                    allocation.size for allocation in self.base_allocations.values()
                ),
                "attention_adapter_layer_count": len(self.adapters),
                "attention_adapter_module_count": len(self.adapters) * 2,
                "target_modules": list(TARGET_MODULES),
                "rank": self.rank,
                "alpha": self.alpha,
                "base_parameter_names": sorted(self.base_allocations),
            },
        )

    def _make_qkv_hook(self, state: SimpleNamespace):
        def apply_delta(_module, inputs, kwargs, output):
            if not self.adapter_ready:
                return output
            if inputs:
                hidden_states = inputs[0]
            else:
                hidden_states = next(
                    (
                        kwargs[name]
                        for name in ("input_", "hidden_states", "x", "input")
                        if name in kwargs
                    ),
                    None,
                )
            if hidden_states is None:
                raise RuntimeError(
                    "DeltaServe attention hook could not find the qkv input tensor"
                )
            q_delta = F.linear(
                F.linear(hidden_states.float(), state.q_lora_a.tensor), state.q_lora_b.tensor
            )
            v_delta = F.linear(
                F.linear(hidden_states.float(), state.v_lora_a.tensor), state.v_lora_b.tensor
            )
            if isinstance(output, tuple):
                qkv, bias = output
            else:
                qkv, bias = output, None
            delta = torch.zeros(qkv.shape, device=qkv.device, dtype=torch.float32)
            delta[..., : state.q_size] = q_delta
            v_start = state.q_size + state.kv_size
            delta[..., v_start : v_start + state.kv_size] = v_delta
            qkv = qkv + delta.to(qkv.dtype)
            return (qkv, bias) if isinstance(output, tuple) else qkv

        return apply_delta

    def poll_results(self) -> None:
        reader = getattr(self.result_queue, "_reader", None)
        if reader is None or not reader.poll():
            return
        while reader.poll():
            result = self.result_queue.get()
            event = result.get("event")
            if event == "backward_error":
                self._inflight_job_id = None
                self._inflight_training_req_ids = ()
                raise RuntimeError(
                    f"DeltaServe attention worker failed for job {result.get('job_id')}: {result.get('error')}"
                )
            if event != "backward_finished":
                continue
            self.adapter_ready = result["parameter_delta_l1"] > 0
            self._inflight_job_id = None
            self._inflight_training_req_ids = ()
            _append_trace(
                self.trace_path,
                {
                    "event": "adapter_published",
                    "pid": os.getpid(),
                    "job_id": result["job_id"],
                    "parameter_delta_l1": result["parameter_delta_l1"],
                    "adapter_ready": self.adapter_ready,
                    "target_modules": list(TARGET_MODULES),
                },
            )

    def _is_inflight_replay(self, req_ids, training_indices) -> bool:
        inflight_requests = set(self._inflight_training_req_ids)
        current_requests = {req_ids[index] for index in training_indices}
        return bool(current_requests) and current_requests.issubset(inflight_requests)

    def prepare_batch(self, req_ids, scheduled_tokens, computed_tokens, input_ids) -> bool:
        self.poll_results()
        training_indices = [
            index
            for index, request_id in enumerate(req_ids)
            if request_id.startswith(TRAINING_REQUEST_PREFIX)
        ]
        if not training_indices:
            self._active = None
            return False
        if self._inflight_job_id is not None:
            if self._is_inflight_replay(req_ids, training_indices):
                self._active = None
                return False
            raise RuntimeError("a DeltaServe attention backward step is already in flight")
        if self._submitted_steps >= self.max_training_steps:
            raise RuntimeError("received more training requests than configured DeltaServe steps")
        starts = [0]
        for count in scheduled_tokens:
            starts.append(starts[-1] + int(count))
        rows = []
        for index in training_indices:
            if int(computed_tokens[index]) != 0:
                self._active = None
                return False
            rows.append({"start": starts[index], "end": starts[index + 1]})
        self._active = {
            "job_id": self._job_id,
            "req_ids": list(req_ids),
            "training_req_ids": [req_ids[index] for index in training_indices],
            "inference_req_ids": [
                request_id
                for request_id in req_ids
                if not request_id.startswith(TRAINING_REQUEST_PREFIX)
            ],
            "training_rows": rows,
            "input_ids": input_ids,
            "scheduled_tokens": [int(value) for value in scheduled_tokens],
            "computed_tokens": [int(value) for value in computed_tokens],
        }
        return True

    def after_model_forward(self) -> None:
        if self._active is None:
            return
        active = self._active
        self._active = None
        offset = 0
        row_lengths = []
        for row in active["training_rows"]:
            source = active["input_ids"][row["start"] : row["end"]].long()
            length = int(source.numel())
            next_offset = offset + length
            if next_offset > self.max_training_tokens:
                raise RuntimeError(
                    f"training token capacity exceeded: {next_offset} > {self.max_training_tokens}"
                )
            self.training_input_ids.tensor[offset:next_offset].copy_(source)
            row_lengths.append(length)
            offset = next_offset
        torch.cuda.synchronize()
        self.command_queue.put(
            {
                "op": "backward",
                "job_id": active["job_id"],
                "row_lengths": row_lengths,
            }
        )
        self._inflight_job_id = active["job_id"]
        self._inflight_training_req_ids = tuple(active["training_req_ids"])
        self._submitted_steps += 1
        self._job_id += 1
        _append_trace(
            self.trace_path,
            {
                "event": "merged_forward",
                "pid": os.getpid(),
                "job_id": active["job_id"],
                "request_ids": active["req_ids"],
                "training_request_ids": active["training_req_ids"],
                "inference_request_ids": active["inference_req_ids"],
                "all_requests_were_prefill": all(
                    value == 0 for value in active["computed_tokens"]
                ),
                "scheduled_tokens": active["scheduled_tokens"],
                "training_row_count": len(row_lengths),
                "training_row_token_counts": row_lengths,
                "training_token_count_for_loss": sum(max(length - 1, 0) for length in row_lengths),
                "shared_base_allocation_count": len(self.base_allocations),
                "target_modules": list(TARGET_MODULES),
            },
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        process = getattr(self, "backward_process", None)
        if process is not None and process.is_alive():
            self.command_queue.put({"op": "close"})
            process.join(5)
            if process.is_alive():
                process.terminate()
                process.join(5)
        for handle in getattr(self, "hook_handles", []):
            handle.remove()
        for state in getattr(self, "adapters", {}).values():
            state.q_lora_a.close()
            state.q_lora_b.close()
            state.v_lora_a.close()
            state.v_lora_b.close()
        for allocation in getattr(self, "base_allocations", {}).values():
            allocation.close()
        input_allocation = getattr(self, "training_input_ids", None)
        if input_allocation is not None:
            input_allocation.close()
