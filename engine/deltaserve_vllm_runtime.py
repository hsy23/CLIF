"""DeltaServe-principle runtime loaded inside a patched vLLM model runner.

The prototype trains a rank-r LoRA on the shared LM head.  The frozen trunk
forward for the synthetic training row is executed in the same vLLM batch as
inference prefill.  A forward hook on the final RMSNorm copies only the
training row to a CUDA VMM activation buffer.  A spawned GPU process maps that
buffer, the live vLLM LM-head weight, and the LoRA tensors through POSIX FDs,
then performs loss/backward/optimizer without loading another base model.
"""

from __future__ import annotations

import atexit
import json
import os
import time
from pathlib import Path
from types import MethodType

import torch
import torch.multiprocessing as mp
from torch.nn import functional as F

from .cuda_vmm import ExportableCudaTensor, ImportedCudaTensor


TRAINING_REQUEST_PREFIX = "deltaserve-ft-"
_RUNTIME = None


def _append_trace(path: str, event: dict) -> None:
    event = {"wall_time_s": time.time(), **event}
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")


def _backward_worker_main(specs: dict, command_queue, result_queue, trace_path: str, rank: int, alpha: float) -> None:
    torch.cuda.set_device(0)
    imported = {name: ImportedCudaTensor.from_spec(spec) for name, spec in specs.items()}
    base_weight = imported["base_weight"].tensor
    lora_a = imported["lora_a"].tensor.requires_grad_(True)
    lora_b = imported["lora_b"].tensor.requires_grad_(True)
    activations = imported["activations"].tensor
    labels = imported["labels"].tensor
    scale = alpha / rank
    optimizer = torch.optim.AdamW([lora_a, lora_b], lr=3e-2)
    reference_done = False

    while True:
        command = command_queue.get()
        if command.get("op") == "close":
            break
        if command.get("op") != "backward":
            raise RuntimeError(f"unexpected backward command: {command}")
        token_count = int(command["token_count"])
        job_id = command["job_id"]
        _append_trace(
            trace_path,
            {
                "event": "backward_started",
                "job_id": job_id,
                "pid": os.getpid(),
                "device": torch.cuda.get_device_name(0),
                "base_allocation_id": imported["base_weight"].allocation_id,
                "activation_allocation_id": imported["activations"].allocation_id,
                "token_count": token_count,
            },
        )
        before_b = lora_b.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        hidden = activations[:token_count].float()
        target = labels[:token_count]
        vocab_size = lora_b.shape[0]
        base_logits = F.linear(hidden.to(base_weight.dtype), base_weight[:vocab_size]).float().detach()
        delta = F.linear(F.linear(hidden, lora_a), lora_b) * scale
        loss = F.cross_entropy(base_logits + delta, target)

        reference = {}
        reference_b = None
        if not reference_done:
            reference_a = lora_a.detach().clone().requires_grad_(True)
            reference_b = lora_b.detach().clone().requires_grad_(True)
            reference_optimizer = torch.optim.AdamW([reference_a, reference_b], lr=3e-2)
            reference_optimizer.zero_grad(set_to_none=True)
            reference_delta = F.linear(
                F.linear(hidden, reference_a), reference_b
            ) * scale
            reference_loss = F.cross_entropy(base_logits + reference_delta, target)
            reference_loss.backward()
            torch.nn.utils.clip_grad_norm_([reference_a, reference_b], 1.0)
            reference_optimizer.step()
            reference_loss_error = abs(float(loss.detach().cpu()) - float(reference_loss.detach().cpu()))
            reference = {
                "reference_loss": float(reference_loss.detach().cpu()),
                "reference_loss_relative_error": reference_loss_error
                / max(abs(float(reference_loss.detach().cpu())), 1e-12),
            }
            reference_done = True
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_([lora_a, lora_b], 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        if reference_b is not None:
            reference_parameter_error = float(
                (lora_b.detach() - reference_b.detach()).abs().sum().cpu()
            )
            reference_parameter_scale = max(float(reference_b.detach().abs().sum().cpu()), 1e-12)
            reference["reference_parameter_delta_relative_error"] = (
                reference_parameter_error / reference_parameter_scale
            )
            reference["reference_check_passed"] = (
                reference["reference_loss_relative_error"] <= 1e-4
                and reference["reference_parameter_delta_relative_error"] <= 1e-4
            )
        result = {
            "event": "backward_finished",
            "job_id": job_id,
            "pid": os.getpid(),
            "loss": float(loss.detach().cpu()),
            "grad_norm": float(grad_norm.detach().cpu()),
            "parameter_delta_l1": float((lora_b.detach() - before_b).abs().sum().cpu()),
            "training_token_count": token_count,
            "base_allocation_id": imported["base_weight"].allocation_id,
            "activation_allocation_id": imported["activations"].allocation_id,
            **reference,
        }
        _append_trace(trace_path, result)
        result_queue.put(result)
    for allocation in imported.values():
        allocation.close()


class DeltaServeVLLMRuntime:
    def __init__(self, runner) -> None:
        if runner.parallel_config.tensor_parallel_size != 1:
            raise RuntimeError("prototype currently requires tensor_parallel_size=1")
        self.runner = runner
        self.trace_path = os.environ["CLIF_DELTASERVE_TRACE"]
        Path(self.trace_path).parent.mkdir(parents=True, exist_ok=True)
        self.rank = int(os.environ.get("CLIF_DELTASERVE_RANK", "4"))
        self.alpha = float(os.environ.get("CLIF_DELTASERVE_ALPHA", "8"))
        self.compute_forward_loss = os.environ.get("CLIF_DELTASERVE_COMPUTE_FORWARD_LOSS", "0") == "1"
        self.max_training_tokens = int(os.environ.get("CLIF_DELTASERVE_MAX_TOKENS", "128"))
        self.max_training_steps = int(os.environ.get("CLIF_DELTASERVE_MAX_STEPS", "1"))
        self.adapter_ready = False
        self._active = None
        self._submitted_steps = 0
        self._inflight_job_id = None
        self._inflight_training_req_ids = ()
        self.completed_results = []
        self._job_id = 0
        self._closed = False

        model = runner.model
        weight_parameter = model.lm_head.weight
        self.base_weight = ExportableCudaTensor(weight_parameter.shape, weight_parameter.dtype)
        self.base_weight.tensor.copy_(weight_parameter.detach())
        torch.cuda.synchronize()
        weight_parameter.data = self.base_weight.tensor
        hidden_size = weight_parameter.shape[1]
        vocab_size = int(model.config.vocab_size)
        self.lora_a = ExportableCudaTensor((self.rank, hidden_size), torch.float32)
        self.lora_b = ExportableCudaTensor((vocab_size, self.rank), torch.float32)
        self.activations = ExportableCudaTensor(
            (self.max_training_tokens, hidden_size),
            torch.float16,
        )
        self.labels = ExportableCudaTensor((self.max_training_tokens,), torch.int64)
        torch.manual_seed(13)
        self.lora_a.tensor.normal_(mean=0.0, std=0.02)
        self.lora_b.tensor.zero_()

        context = mp.get_context("spawn")
        self.command_queue = context.SimpleQueue()
        self.result_queue = context.SimpleQueue()
        specs = {
            "base_weight": self.base_weight.export_spec(),
            "lora_a": self.lora_a.export_spec(),
            "lora_b": self.lora_b.export_spec(),
            "activations": self.activations.export_spec(),
            "labels": self.labels.export_spec(),
        }
        self.backward_process = context.Process(
            target=_backward_worker_main,
            args=(specs, self.command_queue, self.result_queue, self.trace_path, self.rank, self.alpha),
        )
        self.backward_process.start()
        self._install_hook()
        self._install_logits_adapter()
        atexit.register(self.close)
        _append_trace(
            self.trace_path,
            {
                "event": "runtime_initialized",
                "pid": os.getpid(),
                "backward_pid": self.backward_process.pid,
                "base_allocation_id": self.base_weight.allocation_id,
                "base_weight_shape": list(self.base_weight.tensor.shape),
                "base_weight_data_ptr": self.base_weight.tensor.data_ptr(),
                "base_weight_vmm_bytes": self.base_weight.size,
                "base_model_replaced_with_shared_vmm": True,
                "hook_module": "model.norm",
                "compute_forward_loss": self.compute_forward_loss,
            },
        )

    def _install_hook(self) -> None:
        def capture_hook(_module, _inputs, output):
            if self._active is None:
                return
            hidden_states = output[0] if isinstance(output, tuple) else output
            activation_offset = 0
            loss_sum = 0.0
            for row in self._active["training_rows"]:
                start = row["start"]
                end = row["end"]
                training_hidden = hidden_states[start : end - 1]
                token_count = training_hidden.shape[0]
                if token_count <= 0:
                    raise RuntimeError(f"invalid training token count: {token_count}")
                target = self._active["input_ids"][start + 1 : end].long()
                next_offset = activation_offset + token_count
                if next_offset > self.max_training_tokens:
                    raise RuntimeError(
                        f"training token capacity exceeded: {next_offset} > {self.max_training_tokens}"
                    )
                self.activations.tensor[activation_offset:next_offset].copy_(
                    training_hidden.to(torch.float16)
                )
                self.labels.tensor[activation_offset:next_offset].copy_(target)
                if self.compute_forward_loss:
                    base_logits = F.linear(
                        training_hidden.to(self.base_weight.tensor.dtype),
                        self.base_weight.tensor[: self.lora_b.tensor.shape[0]],
                    ).float()
                    delta = F.linear(
                        F.linear(training_hidden.float(), self.lora_a.tensor),
                        self.lora_b.tensor,
                    ) * (self.alpha / self.rank)
                    loss_sum += float(
                        F.cross_entropy(base_logits + delta, target, reduction="sum")
                        .detach()
                        .cpu()
                    )
                row["activation_start"] = activation_offset
                row["activation_end"] = next_offset
                row["token_count"] = token_count
                activation_offset = next_offset
            self._active["forward_loss"] = (
                loss_sum / max(activation_offset, 1) if self.compute_forward_loss else None
            )
            self._active["token_count"] = activation_offset
            self._active["hook_called"] = True

        self.hook_handle = self.runner.model.model.norm.register_forward_hook(capture_hook)

    def _install_logits_adapter(self) -> None:
        model = self.runner.model
        original_compute_logits = model.compute_logits

        def compute_logits(_model, hidden_states):
            logits = original_compute_logits(hidden_states)
            if logits is None or not self.adapter_ready:
                return logits
            delta = F.linear(
                F.linear(hidden_states.float(), self.lora_a.tensor),
                self.lora_b.tensor,
            ) * (self.alpha / self.rank)
            return logits + delta.to(logits.dtype)

        model.compute_logits = MethodType(compute_logits, model)

    def poll_results(self) -> None:
        reader = getattr(self.result_queue, "_reader", None)
        if reader is None or not reader.poll():
            return
        while reader.poll():
            result = self.result_queue.get()
            if result.get("event") == "backward_finished":
                self.adapter_ready = result["parameter_delta_l1"] > 0
                self._inflight_job_id = None
                self._inflight_training_req_ids = ()
                self.completed_results.append(result)
                _append_trace(
                    self.trace_path,
                    {
                        "event": "adapter_published",
                        "pid": os.getpid(),
                        "job_id": result["job_id"],
                        "parameter_delta_l1": result["parameter_delta_l1"],
                        "adapter_ready": self.adapter_ready,
                    },
                )

    def _is_inflight_replay(self, req_ids, training_indices) -> bool:
        inflight_requests = set(self._inflight_training_req_ids)
        current_requests = {req_ids[index] for index in training_indices}
        return bool(current_requests) and current_requests.issubset(inflight_requests)

    def prepare_batch(self, req_ids, scheduled_tokens, computed_tokens, input_ids) -> bool:
        self.poll_results()
        training_indices = [
            index for index, request_id in enumerate(req_ids) if request_id.startswith(TRAINING_REQUEST_PREFIX)
        ]
        if not training_indices:
            self._active = None
            return False
        if self._inflight_job_id is not None:
            if self._is_inflight_replay(req_ids, training_indices):
                # vLLM can replay the just-prefilled training request while its
                # one-token output is being finalized.  It is not a new step;
                # let the engine finish the request without capturing twice.
                self._active = None
                return False
            raise RuntimeError("a DeltaServe backward step is already in flight")
        if self._submitted_steps >= self.max_training_steps:
            raise RuntimeError("received more training requests than configured DeltaServe steps")
        starts = [0]
        for count in scheduled_tokens:
            starts.append(starts[-1] + int(count))
        training_rows = []
        for index in training_indices:
            if int(computed_tokens[index]) != 0:
                self._active = None
                return False
            training_rows.append(
                {
                    "request_id": req_ids[index],
                    "start": starts[index],
                    "end": starts[index + 1],
                }
            )
        self._active = {
            "job_id": self._job_id,
            "req_ids": list(req_ids),
            "training_req_id": training_rows[0]["request_id"],
            "training_req_ids": [row["request_id"] for row in training_rows],
            "training_rows": training_rows,
            "inference_req_ids": [request_id for request_id in req_ids if not request_id.startswith(TRAINING_REQUEST_PREFIX)],
            "input_ids": input_ids,
            "computed_tokens": [int(value) for value in computed_tokens],
            "scheduled_tokens": [int(value) for value in scheduled_tokens],
            "hook_called": False,
        }
        return True

    def after_model_forward(self) -> None:
        if self._active is None:
            return
        active = self._active
        self._active = None
        if not active["hook_called"]:
            raise RuntimeError("DeltaServe final-residual hook did not run")
        if self._inflight_job_id is not None:
            raise RuntimeError("a DeltaServe backward step is already in flight")
        torch.cuda.synchronize()
        self.command_queue.put(
            {
                "op": "backward",
                "job_id": active["job_id"],
                "token_count": active["token_count"],
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
                "training_request_id": active["training_req_id"],
                "training_request_ids": active["training_req_ids"],
                "inference_request_ids": active["inference_req_ids"],
                "all_requests_were_prefill": all(value == 0 for value in active["computed_tokens"]),
                "scheduled_tokens": active["scheduled_tokens"],
                "training_row_count": len(active["training_rows"]),
                "training_row_token_counts": [row["token_count"] for row in active["training_rows"]],
                "training_flat_slice": [
                    active["training_rows"][0]["start"],
                    active["training_rows"][-1]["end"],
                ],
                "training_token_count_for_loss": active["token_count"],
                "hook_called": active["hook_called"],
                "forward_loss": active["forward_loss"],
                "base_allocation_id": self.base_weight.allocation_id,
                "activation_allocation_id": self.activations.allocation_id,
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
        hook_handle = getattr(self, "hook_handle", None)
        if hook_handle is not None:
            hook_handle.remove()
        for name in ("base_weight", "lora_a", "lora_b", "activations", "labels"):
            allocation = getattr(self, name, None)
            if allocation is not None:
                allocation.close()


def get_runtime(runner) -> DeltaServeVLLMRuntime:
    global _RUNTIME
    if _RUNTIME is None:
        target_modules = tuple(
            module.strip()
            for module in os.environ.get("CLIF_DELTASERVE_TARGET_MODULES", "lm_head").split(",")
            if module.strip()
        )
        if target_modules == ("q_proj", "v_proj"):
            from .deltaserve_attention_runtime import DeltaServeQwen3AttentionRuntime

            _RUNTIME = DeltaServeQwen3AttentionRuntime(runner)
        else:
            _RUNTIME = DeltaServeVLLMRuntime(runner)
    return _RUNTIME
