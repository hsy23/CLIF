"""Benchmark the shared-allocation DeltaServe vLLM prototype."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
venv_bin = str(Path(sys.prefix) / "bin")
os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--baseline-requests", type=int, default=8)
    parser.add_argument("--training-steps", type=int, default=8)
    parser.add_argument("--training-batch-size", type=int, default=1)
    parser.add_argument("--training-sequence-length", type=int, default=96)
    parser.add_argument("--mixed-warmup-steps", type=int, default=0)
    parser.add_argument("--target-modules", default="lm_head")
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def output_throughput(requests: list[dict]) -> float:
    if not requests:
        return 0.0
    elapsed = max(request["finished_wall_s"] for request in requests) - min(
        request["started_wall_s"] for request in requests
    )
    return sum(request["output_tokens"] for request in requests) / max(elapsed, 1e-9)


def read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def wait_for_event(path: Path, event_name: str, job_id: int, timeout_s: float) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for event in read_events(path):
            if event.get("event") == event_name and event.get("job_id") == job_id:
                return event
        time.sleep(0.02)
    raise TimeoutError(f"did not observe {event_name} for job {job_id} in {path}")


def drain_engine(llm) -> list:
    outputs = []
    while llm.llm_engine.has_unfinished_requests():
        for output in llm.llm_engine.step():
            if output.finished:
                outputs.append(output)
    return outputs


def inference_prompt(index: int) -> str:
    return f"Request {index}: 用一句话解释连续批处理为什么能提高大模型推理吞吐。"


def run_baseline_request(llm, sampling, index: int) -> dict:
    started = time.time()
    outputs = llm.generate([inference_prompt(index)], sampling, use_tqdm=False)
    finished = time.time()
    output = outputs[0].outputs[0]
    return {
        "index": index,
        "started_wall_s": started,
        "finished_wall_s": finished,
        "latency_s": finished - started,
        "output_tokens": len(output.token_ids),
        "text": output.text.strip(),
    }


def make_training_prompt_ids(tokenizer, sequence_length: int) -> list[int]:
    seed = (
        "高吞吐推理需要连续批处理，参数高效微调只更新低秩适配器。"
        "A shared base model lets the serving engine and the training worker reuse weights. "
    )
    token_ids: list[int] = []
    while len(token_ids) < sequence_length:
        token_ids.extend(tokenizer.encode(seed, add_special_tokens=False))
    return token_ids[:sequence_length]


def run_mixed_step(
    llm,
    sampling,
    training_rows: list[list[int]],
    step: int,
    request_step: int,
) -> dict:
    from vllm import SamplingParams

    inference_id = f"deltaserve-inference-{request_step}"
    started = time.time()
    for row_index, training_ids in enumerate(training_rows):
        llm.llm_engine.add_request(
            f"deltaserve-ft-{request_step}-{row_index}",
            {"prompt_token_ids": training_ids},
            SamplingParams(temperature=0.0, max_tokens=1),
        )
    llm.llm_engine.add_request(inference_id, inference_prompt(step), sampling)
    outputs = drain_engine(llm)
    finished = time.time()
    inference_output = next(
        (output for output in outputs if output.request_id == inference_id),
        next((output for output in outputs if not output.request_id.startswith("deltaserve-ft-")), None),
    )
    if inference_output is None:
        raise RuntimeError(f"no inference output for step {step}: {outputs}")
    generated = inference_output.outputs[0]
    return {
        "step": step,
        "started_wall_s": started,
        "finished_wall_s": finished,
        "latency_s": finished - started,
        "output_tokens": len(generated.token_ids),
        "text": generated.text.strip(),
    }


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.is_dir():
        raise FileNotFoundError(args.model)
    if args.training_steps <= 0 or args.training_batch_size <= 0 or args.training_sequence_length < 2:
        raise ValueError("training steps and batch size must be positive, sequence length must be at least 2")
    if args.mixed_warmup_steps < 0:
        raise ValueError("mixed warmup steps must be non-negative")

    args.trace.parent.mkdir(parents=True, exist_ok=True)
    args.trace.unlink(missing_ok=True)
    os.environ["CLIF_DELTASERVE_ENABLE"] = "1"
    os.environ["CLIF_DELTASERVE_TRACE"] = str(args.trace.resolve())
    os.environ["CLIF_DELTASERVE_RANK"] = "4"
    os.environ["CLIF_DELTASERVE_ALPHA"] = "8"
    os.environ["CLIF_DELTASERVE_TARGET_MODULES"] = args.target_modules
    os.environ["CLIF_DELTASERVE_MAX_TOKENS"] = str(
        args.training_batch_size * args.training_sequence_length
    )
    os.environ["CLIF_DELTASERVE_MAX_STEPS"] = str(args.training_steps + args.mixed_warmup_steps)

    import torch
    from vllm import LLM, SamplingParams

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    llm = LLM(
        model=str(model_path),
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        max_num_seqs=max(8, args.training_batch_size + 1),
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        async_scheduling=False,
        disable_log_stats=True,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    run_baseline_request(llm, sampling, -1)
    baseline = [run_baseline_request(llm, sampling, index) for index in range(args.baseline_requests)]

    tokenizer = llm.get_tokenizer()
    base_training_ids = make_training_prompt_ids(tokenizer, args.training_sequence_length)

    def training_rows(runtime_step: int) -> list[list[int]]:
        rows = []
        for row_index in range(args.training_batch_size):
            row = list(base_training_ids)
            row[0] = 1000 + runtime_step * args.training_batch_size + row_index
            rows.append(row)
        return rows

    for warmup_step in range(args.mixed_warmup_steps):
        run_mixed_step(llm, sampling, training_rows(warmup_step), -1, warmup_step)
        wait_for_event(args.trace, "backward_finished", warmup_step, 120)

    mixed: list[dict] = []
    training_started = time.time()
    for step in range(args.training_steps):
        # Prefix caching must not turn a later synthetic FT row into a decode
        # continuation; change the first token so every row is a fresh prefill.
        runtime_step = args.mixed_warmup_steps + step
        mixed.append(run_mixed_step(llm, sampling, training_rows(runtime_step), step, runtime_step))
        wait_for_event(args.trace, "backward_finished", runtime_step, 120)
    training_finished = time.time()

    post_id = "deltaserve-post-update"
    llm.llm_engine.add_request(post_id, "说明 LoRA 微调的作用。", SamplingParams(temperature=0.0, max_tokens=8))
    post_outputs = drain_engine(llm)
    wait_for_event(
        args.trace,
        "adapter_published",
        args.mixed_warmup_steps + args.training_steps - 1,
        30,
    )
    post_output = next(output for output in post_outputs if output.request_id == post_id)

    events = read_events(args.trace)
    initialized = next(event for event in events if event["event"] == "runtime_initialized")
    worker_initialized = next(
        (event for event in events if event["event"] == "attention_worker_initialized"),
        {},
    )
    measured_job_ids = set(range(args.mixed_warmup_steps, args.mixed_warmup_steps + args.training_steps))
    merged = [
        event
        for event in events
        if event["event"] == "merged_forward" and event["job_id"] in measured_job_ids
    ]
    backward = [
        event
        for event in events
        if event["event"] == "backward_finished" and event["job_id"] in measured_job_ids
    ]
    published = [
        event
        for event in events
        if event["event"] == "adapter_published" and event["job_id"] in measured_job_ids
    ]
    if len(merged) != args.training_steps or len(backward) != args.training_steps:
        raise RuntimeError(f"expected {args.training_steps} merged/backward events, got {len(merged)}/{len(backward)}")

    baseline_latencies = [request["latency_s"] for request in baseline]
    mixed_latencies = [request["latency_s"] for request in mixed]
    measured_backward_started = [
        event
        for event in events
        if event["event"] == "backward_started" and event["job_id"] in measured_job_ids
    ]
    backward_compute_s = sum(
        event["wall_time_s"] - started["wall_time_s"]
        for event, started in zip(
            backward,
            measured_backward_started,
            strict=True,
        )
    )
    summary = {
        "success": True,
        "environment": {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "model": str(model_path),
            "vllm_gpu_memory_utilization": args.gpu_memory_utilization,
            "vllm_max_model_len": args.max_model_len,
            "vllm_max_tokens": args.max_tokens,
            "flashinfer_sampler": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER"),
            "target_modules": args.target_modules,
            "async_scheduling": False,
            "enforce_eager": True,
            "gpu_memory": {
                "allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 2),
                "reserved_mib": round(torch.cuda.memory_reserved() / 2**20, 2),
                "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 2),
                "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 2),
            },
        },
        "inference": {
            "baseline_requests": len(baseline),
            "mixed_requests": len(mixed),
            "baseline_output_tokens": sum(request["output_tokens"] for request in baseline),
            "mixed_output_tokens": sum(request["output_tokens"] for request in mixed),
            "baseline_output_tokens_per_s": output_throughput(baseline),
            "mixed_output_tokens_per_s": output_throughput(mixed),
            "baseline_median_latency_s": percentile(baseline_latencies, 0.5),
            "baseline_p95_latency_s": percentile(baseline_latencies, 0.95),
            "mixed_median_latency_s": percentile(mixed_latencies, 0.5),
            "mixed_p95_latency_s": percentile(mixed_latencies, 0.95),
            "requests": baseline + mixed,
        },
            "training": {
            "steps": args.training_steps,
            "sequence_length": args.training_sequence_length,
            "training_rows_per_step": args.training_batch_size,
            "scheduled_tokens": (
                args.training_steps * args.training_batch_size * args.training_sequence_length
            ),
            "loss_tokens": sum(event["training_token_count_for_loss"] for event in merged),
            "end_to_end_elapsed_s": training_finished - training_started,
            "end_to_end_tokens_per_s": (
                args.training_steps * args.training_batch_size * args.training_sequence_length
                / max(training_finished - training_started, 1e-9)
            ),
            "backward_compute_elapsed_s": backward_compute_s,
            "backward_loss_tokens_per_s": sum(event["training_token_count_for_loss"] for event in merged)
            / max(backward_compute_s, 1e-9),
            "warmup_mixed_steps": args.mixed_warmup_steps,
            "backward_events": backward,
        },
        "delta_engine": {
            "shared_base_allocation": initialized["base_model_replaced_with_shared_vmm"],
            "base_allocation_id": initialized.get("base_allocation_id"),
            "base_allocation_ids": initialized.get("base_allocation_ids", []),
            "shared_base_allocation_count": initialized.get("shared_base_allocation_count", 1),
            "attention_adapter_layer_count": initialized.get("attention_adapter_layer_count", 0),
            "target_modules": initialized.get("target_modules", ["lm_head"]),
            "rank": initialized.get("rank", 4),
            "alpha": initialized.get("alpha", 8),
            "trainable_parameter_count": worker_initialized.get("trainable_parameter_count"),
            "backward_pid": initialized.get("backward_pid"),
            "merged_forward_steps": len(merged),
            "published_updates": len(published),
            "last_post_update_output": post_output.outputs[0].text.strip(),
        },
        "trace": str(args.trace),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
