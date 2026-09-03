"""Principle-level single-GPU vLLM inference and LoRA training backend.

This module intentionally implements the smallest real co-execution path
before the DeltaServe activation-sharing fork exists.  vLLM owns inference in
one process.  A second process loads a frozen copy of the same base model and
updates only LoRA parameters.  The processes share a GPU through CUDA context
time-slicing; they do not share model storage or activations yet.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """A frozen linear layer plus a trainable low-rank residual."""

    def __init__(self, base: nn.Linear, *, rank: int, alpha: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

        # Keep the small trainable matrices in fp32 for stable consumer-GPU
        # optimization.  The residual is cast back to the base output dtype.
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base(inputs)
        residual = F.linear(F.linear(inputs.float(), self.lora_a), self.lora_b)
        return base_output + residual.to(base_output.dtype) * self.scale


def inject_lora(
    model: nn.Module,
    *,
    target_names: Sequence[str] = ("q_proj", "v_proj"),
    rank: int = 4,
    alpha: float = 8.0,
) -> tuple[str, ...]:
    """Replace matching linear children and return their fully-qualified names."""

    targets = set(target_names)
    replacements: list[tuple[nn.Module, str, nn.Linear, str]] = []
    for module_name, module in model.named_modules():
        for child_name, child in module.named_children():
            if child_name in targets and isinstance(child, nn.Linear):
                full_name = f"{module_name}.{child_name}" if module_name else child_name
                replacements.append((module, child_name, child, full_name))

    for parent, child_name, child, _ in replacements:
        setattr(parent, child_name, LoRALinear(child, rank=rank, alpha=alpha))

    return tuple(full_name for _, _, _, full_name in replacements)


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def intervals_overlap(first: tuple[float, float], second: tuple[float, float]) -> bool:
    return max(first[0], second[0]) < min(first[1], second[1])


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


@dataclass(frozen=True)
class TrainingWorkerConfig:
    model: str
    steps: int
    sequence_length: int
    learning_rate: float
    lora_rank: int
    lora_alpha: float
    ready_file: str
    go_file: str
    result_file: str
    adapter_output_dir: str
    start_timeout_s: float = 180.0
    step_go_dir: str | None = None
    step_done_dir: str | None = None
    batch_size: int = 1


def _atomic_write_json(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, target)


def _training_text(step: int) -> str:
    examples = (
        "高吞吐推理需要连续批处理，参数高效微调只更新低秩适配器。",
        "A serving engine schedules prefill and decode while a LoRA worker performs optimization.",
        "训推并行的第一步是证明推理请求与真实反向传播能在同一张 GPU 上共同推进。",
    )
    return (examples[step % len(examples)] + " ") * 12


def _wait_for_step_marker(directory: str, step: int, timeout_s: float) -> None:
    marker = Path(directory) / f"step-{step}"
    deadline = time.monotonic() + timeout_s
    while not marker.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"controller did not release training step {step}")
        time.sleep(0.01)


def export_vllm_lora_adapter(
    model: nn.Module,
    output_dir: str | Path,
    *,
    base_model: str,
    rank: int,
    alpha: float,
) -> dict:
    """Write the custom LoRA modules in the PEFT layout consumed by vLLM."""

    from safetensors.torch import save_file

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {}
    module_count = 0
    for module_name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        prefix = f"base_model.model.{module_name}"
        tensors[f"{prefix}.lora_A.weight"] = module.lora_a.detach().cpu().contiguous()
        tensors[f"{prefix}.lora_B.weight"] = module.lora_b.detach().cpu().contiguous()
        module_count += 1
    if not tensors:
        raise RuntimeError("no LoRA tensors were available to export")

    save_file(tensors, destination / "adapter_model.safetensors")
    config = {
        "base_model_name_or_path": base_model,
        "bias": "none",
        "inference_mode": True,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "peft_type": "LORA",
        "r": rank,
        "target_modules": ["q_proj", "v_proj"],
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
    }
    _atomic_write_json(destination / "adapter_config.json", config)
    return {
        "path": str(destination),
        "modules": module_count,
        "tensors": len(tensors),
        "bytes": (destination / "adapter_model.safetensors").stat().st_size,
    }


def run_training_worker(config: TrainingWorkerConfig) -> None:
    """Run real LoRA optimization and publish machine-readable timing evidence."""

    result: dict = {"phase": "starting", "pid": os.getpid(), "config": asdict(config)}
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the training worker")
        if config.batch_size <= 0:
            raise ValueError("training batch size must be positive")

        torch.manual_seed(7)
        torch.cuda.manual_seed_all(7)
        tokenizer = AutoTokenizer.from_pretrained(config.model, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            config.model,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        model.config.use_cache = False
        model.requires_grad_(False)
        injected = inject_lora(
            model,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
        )
        if not injected:
            raise RuntimeError("no q_proj/v_proj linear layers were found for LoRA injection")
        model.to("cuda")
        model.train()

        parameters = trainable_parameters(model)
        optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate)
        initial_parameters = [parameter.detach().float().cpu().clone() for parameter in parameters]
        ready_payload = {
            "phase": "ready",
            "pid": os.getpid(),
            "injected_modules": len(injected),
            "trainable_parameters": sum(parameter.numel() for parameter in parameters),
            "cuda_allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 2),
        }
        _atomic_write_json(config.ready_file, ready_payload)

        deadline = time.monotonic() + config.start_timeout_s
        while not Path(config.go_file).exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("controller did not release the training worker")
            time.sleep(0.01)

        training_started = time.time()
        step_records: list[dict] = []
        if (config.step_go_dir is None) != (config.step_done_dir is None):
            raise ValueError("step_go_dir and step_done_dir must be provided together")
        if config.step_done_dir is not None:
            Path(config.step_done_dir).mkdir(parents=True, exist_ok=True)
        for step in range(config.steps):
            if config.step_go_dir is not None:
                _wait_for_step_marker(config.step_go_dir, step, config.start_timeout_s)
            encoded = tokenizer(
                _training_text(step),
                return_tensors="pt",
                truncation=True,
                max_length=config.sequence_length,
            )
            sample_ids = encoded["input_ids"].squeeze(0)
            if sample_ids.shape[0] < 2:
                raise RuntimeError("training sample tokenized to fewer than two tokens")
            input_ids = sample_ids.unsqueeze(0).repeat(config.batch_size, 1).to("cuda")

            optimizer.zero_grad(set_to_none=True)
            step_started = time.time()
            outputs = model(input_ids=input_ids, labels=input_ids, use_cache=False)
            loss = outputs.loss.float()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            optimizer.step()
            torch.cuda.synchronize()
            step_finished = time.time()
            step_records.append(
                {
                    "step": step,
                    "started_wall_s": step_started,
                    "finished_wall_s": step_finished,
                    "elapsed_s": step_finished - step_started,
                    "token_count": int(input_ids.numel()),
                    "loss": float(loss.detach().cpu()),
                    "grad_norm": float(grad_norm.detach().cpu()),
                }
            )
            if config.step_done_dir is not None:
                (Path(config.step_done_dir) / f"step-{step}").touch()

        training_finished = time.time()
        parameter_delta_l1 = 0.0
        for before, after in zip(initial_parameters, parameters, strict=True):
            parameter_delta_l1 += float((after.detach().float().cpu() - before).abs().sum())
        adapter = export_vllm_lora_adapter(
            model,
            config.adapter_output_dir,
            base_model=config.model,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
        )

        result = {
            **ready_payload,
            "phase": "finished",
            "training_started_wall_s": training_started,
            "training_finished_wall_s": training_finished,
            "training_elapsed_s": training_finished - training_started,
            "training_tokens": sum(record["token_count"] for record in step_records),
            "training_tokens_per_s": sum(record["token_count"] for record in step_records)
            / max(training_finished - training_started, 1e-9),
            "parameter_delta_l1": parameter_delta_l1,
            "initial_loss": step_records[0]["loss"],
            "final_loss": step_records[-1]["loss"],
            "steps": step_records,
            "adapter": adapter,
            "peak_cuda_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 2),
        }
        _atomic_write_json(config.result_file, result)
    except Exception as exc:
        result = {
            **result,
            "phase": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _atomic_write_json(config.result_file, result)
        raise


def count_request_step_overlaps(
    requests: Iterable[dict],
    steps: Iterable[dict],
) -> tuple[int, int]:
    request_intervals = [
        (float(request["started_wall_s"]), float(request["finished_wall_s"]))
        for request in requests
    ]
    step_intervals = [
        (float(step["started_wall_s"]), float(step["finished_wall_s"]))
        for step in steps
    ]
    overlapping_requests = sum(
        any(intervals_overlap(request, step) for step in step_intervals)
        for request in request_intervals
    )
    overlapping_steps = sum(
        any(intervals_overlap(step, request) for request in request_intervals)
        for step in step_intervals
    )
    return overlapping_requests, overlapping_steps
