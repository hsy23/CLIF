"""Run a long shared-execution DeltaServe stability check."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
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
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--training-steps", type=int, default=100)
    parser.add_argument("--inference-requests", type=int, default=1000)
    parser.add_argument("--inference-batch-size", type=int, default=10)
    parser.add_argument("--training-sequence-length", type=int, default=96)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


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


def drain_engine_timed(llm, request_starts: dict[str, float]) -> dict[str, dict]:
    """Collect per-request first/last token timestamps from the offline engine."""

    partial: dict[str, dict] = {}
    finished: dict[str, dict] = {}
    while llm.llm_engine.has_unfinished_requests():
        for output in llm.llm_engine.step():
            request_id = output.request_id
            if request_id not in request_starts or not output.outputs:
                continue
            now = time.time()
            generated = output.outputs[0]
            token_count = len(generated.token_ids)
            record = partial.setdefault(
                request_id,
                {
                    "request_id": request_id,
                    "started_wall_s": request_starts[request_id],
                    "first_token_wall_s": None,
                    "last_token_wall_s": None,
                    "last_token_count": 0,
                },
            )
            if token_count > record["last_token_count"]:
                record["first_token_wall_s"] = record["first_token_wall_s"] or now
                record["last_token_wall_s"] = now
                record["last_token_count"] = token_count
            if output.finished:
                first_token = record["first_token_wall_s"]
                last_token = record["last_token_wall_s"]
                finished[request_id] = {
                    "request_id": request_id,
                    "started_wall_s": record["started_wall_s"],
                    "finished_wall_s": now,
                    "latency_s": now - record["started_wall_s"],
                    "output_tokens": token_count,
                    "ttft_s": (
                        first_token - record["started_wall_s"]
                        if first_token is not None
                        else None
                    ),
                    "tpot_s": (
                        (last_token - first_token) / max(token_count - 1, 1)
                        if first_token is not None and last_token is not None
                        else None
                    ),
                }
    return finished


def inference_batch(llm, sampling, indices: list[int], prefix: str) -> list[dict]:
    request_starts: dict[str, float] = {}
    request_ids = []
    for index in indices:
        request_id = f"{prefix}-{index}"
        request_ids.append(request_id)
        request_starts[request_id] = time.time()
        llm.llm_engine.add_request(request_id, inference_prompt(index), sampling)
    records = drain_engine_timed(llm, request_starts)
    missing = [request_id for request_id in request_ids if request_id not in records]
    if missing:
        raise RuntimeError(f"missing timed inference outputs: {missing}")
    return [records[request_id] for request_id in request_ids]


def gpu_metrics() -> dict:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=2.0,
        )
        fields = [field.strip() for field in completed.stdout.strip().split(",")]
        if len(fields) == 4:
            return {
                "gpu": fields[0],
                "utilization_percent": float(fields[1]),
                "memory_used_mib": float(fields[2]),
                "memory_total_mib": float(fields[3]),
            }
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return {}


class GPUMetricsSampler:
    def __init__(self, interval_s: float = 0.25) -> None:
        self.interval_s = interval_s
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            sample = gpu_metrics()
            if sample:
                sample["wall_time_s"] = time.time()
                self.samples.append(sample)
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        self._thread.join(timeout=3.0)
        if not self.samples:
            return {"sample_count": 0}
        peak_memory = max(self.samples, key=lambda sample: sample["memory_used_mib"])
        peak_utilization = max(self.samples, key=lambda sample: sample["utilization_percent"])
        return {
            "sample_count": len(self.samples),
            "first": self.samples[0],
            "last": self.samples[-1],
            "peak_memory_used_mib": peak_memory["memory_used_mib"],
            "peak_utilization_percent": peak_utilization["utilization_percent"],
            "memory_total_mib": self.samples[-1]["memory_total_mib"],
        }


def summarize_inference(records: list[dict], elapsed_s: float) -> dict:
    def values(key: str) -> list[float]:
        return [float(record[key]) for record in records if record.get(key) is not None]

    latency = values("latency_s")
    ttft = values("ttft_s")
    tpot = values("tpot_s")
    output_tokens = sum(record["output_tokens"] for record in records)
    return {
        "requests": len(records),
        "output_tokens": output_tokens,
        "output_tokens_per_s": output_tokens / max(elapsed_s, 1e-9),
        "p50_latency_s": percentile(latency, 0.50),
        "p95_latency_s": percentile(latency, 0.95),
        "p99_latency_s": percentile(latency, 0.99),
        "p50_ttft_s": percentile(ttft, 0.50),
        "p95_ttft_s": percentile(ttft, 0.95),
        "p99_ttft_s": percentile(ttft, 0.99),
        "p50_tpot_s": percentile(tpot, 0.50),
        "p95_tpot_s": percentile(tpot, 0.95),
        "p99_tpot_s": percentile(tpot, 0.99),
        "elapsed_s": elapsed_s,
    }


def relative_change(value: float, reference: float) -> float | None:
    return value / reference - 1.0 if reference else None


def inference_prompt(index: int) -> str:
    return f"Request {index}: 用一句话解释连续批处理为什么能提高大模型推理吞吐。"


def make_training_prompt_ids(tokenizer, sequence_length: int) -> list[int]:
    seed = (
        "高吞吐推理需要连续批处理，参数高效微调只更新低秩适配器。"
        "A shared base model lets the serving engine and the training worker reuse weights. "
    )
    token_ids: list[int] = []
    while len(token_ids) < sequence_length:
        token_ids.extend(tokenizer.encode(seed, add_special_tokens=False))
    return token_ids[:sequence_length]


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.is_dir():
        raise FileNotFoundError(args.model)
    if args.training_steps <= 0 or args.inference_requests <= 0:
        raise ValueError("training steps and inference requests must be positive")
    if args.inference_batch_size <= 0 or args.max_num_seqs <= 0 or args.training_sequence_length < 2:
        raise ValueError("batch size and max sequences must be positive, sequence length must be at least 2")

    args.trace.parent.mkdir(parents=True, exist_ok=True)
    args.trace.unlink(missing_ok=True)
    os.environ["CLIF_DELTASERVE_ENABLE"] = "1"
    os.environ["CLIF_DELTASERVE_TRACE"] = str(args.trace.resolve())
    os.environ["CLIF_DELTASERVE_RANK"] = "4"
    os.environ["CLIF_DELTASERVE_ALPHA"] = "8"
    os.environ["CLIF_DELTASERVE_MAX_TOKENS"] = str(args.training_sequence_length)
    os.environ["CLIF_DELTASERVE_MAX_STEPS"] = str(args.training_steps)

    import torch
    from vllm import LLM, SamplingParams

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    gpu_sampler = GPUMetricsSampler()
    gpu_sampler.start()
    llm = LLM(
        model=str(model_path),
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        async_scheduling=False,
        disable_log_stats=True,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    tokenizer = llm.get_tokenizer()
    base_training_ids = make_training_prompt_ids(tokenizer, args.training_sequence_length)

    # Warmup is outside both measured intervals and leaves the first batch with
    # the same initialized engine state as the benchmark.
    inference_batch(llm, sampling, [-1], "deltaserve-warmup")

    # Use the same request count and batch partition for a pure-inference
    # reference.  The runtime is initialized but no synthetic FT row is
    # admitted, so this isolates the per-request effect of mixed execution.
    baseline_records: list[dict] = []
    baseline_started = time.time()
    remaining = args.inference_requests
    next_request = 0
    while remaining:
        count = min(args.inference_batch_size, remaining)
        baseline_records.extend(
            inference_batch(
                llm,
                sampling,
                list(range(next_request, next_request + count)),
                "deltaserve-stability-baseline",
            )
        )
        remaining -= count
        next_request += count
    baseline_finished = time.time()

    remaining = args.inference_requests
    next_request = 0
    request_records: list[dict] = []
    measured_started = time.time()
    for step in range(args.training_steps):
        count = min(args.inference_batch_size, remaining)
        if count == 0:
            raise RuntimeError("inference request budget was exhausted before training finished")
        training_ids = list(base_training_ids)
        training_ids[0] = 1000 + step
        training_id = f"deltaserve-ft-stability-{step}"
        started = time.time()
        llm.llm_engine.add_request(
            training_id,
            {"prompt_token_ids": training_ids},
            SamplingParams(temperature=0.0, max_tokens=1),
        )
        request_ids = []
        for offset in range(count):
            request_id = f"deltaserve-stability-inference-{next_request + offset}"
            request_ids.append(request_id)
            llm.llm_engine.add_request(request_id, inference_prompt(next_request + offset), sampling)
        request_starts = {request_id: started for request_id in request_ids}
        timed_outputs = drain_engine_timed(llm, request_starts)
        for request_id in request_ids:
            output = timed_outputs.get(request_id)
            if output is None:
                raise RuntimeError(f"missing output for {request_id}")
            request_records.append(
                {
                    "request_id": request_id,
                    "step": step,
                    **output,
                }
            )
        wait_for_event(args.trace, "backward_finished", step, 120)
        remaining -= count
        next_request += count

    measured_finished = time.time()
    if remaining != 0:
        raise RuntimeError(f"training completed with {remaining} inference requests unissued")
    post_id = "deltaserve-stability-post-update"
    llm.llm_engine.add_request(post_id, "说明 LoRA 微调的作用。", SamplingParams(temperature=0.0, max_tokens=8))
    post_outputs = drain_engine(llm)
    wait_for_event(args.trace, "adapter_published", args.training_steps - 1, 30)
    if not any(output.request_id == post_id for output in post_outputs):
        raise RuntimeError("post-update inference did not finish")

    events = read_events(args.trace)
    initialized = next(event for event in events if event["event"] == "runtime_initialized")
    merged = [event for event in events if event["event"] == "merged_forward"]
    backward = [event for event in events if event["event"] == "backward_finished"]
    published = [event for event in events if event["event"] == "adapter_published"]
    allocation_ids = {
        event.get("base_allocation_id")
        for event in merged + backward
        if event.get("base_allocation_id") is not None
    }
    worker_pids = {event.get("pid") for event in backward if event.get("pid") is not None}
    finite_values = [
        event[key]
        for event in backward
        for key in ("loss", "grad_norm", "parameter_delta_l1")
        if key in event
    ]
    reference = next((event for event in backward if "reference_check_passed" in event), {})
    output_tokens = sum(request["output_tokens"] for request in request_records)
    elapsed = max(measured_finished - measured_started, 1e-9)
    baseline_elapsed = max(baseline_finished - baseline_started, 1e-9)
    latency_values = [request["latency_s"] for request in request_records]
    baseline_summary = summarize_inference(baseline_records, baseline_elapsed)
    mixed_summary = summarize_inference(request_records, elapsed)
    nvidia_smi_observation = gpu_sampler.stop()
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
            "async_scheduling": False,
            "enforce_eager": True,
            "gpu_memory": {
                "allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 2),
                "reserved_mib": round(torch.cuda.memory_reserved() / 2**20, 2),
                "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 2),
                "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 2),
            },
            "nvidia_smi": gpu_metrics(),
            "nvidia_smi_observation": nvidia_smi_observation,
        },
        "workload": {
            "training_steps": args.training_steps,
            "inference_requests_requested": args.inference_requests,
            "inference_requests_completed": len(request_records),
            "inference_success_rate": len(request_records) / args.inference_requests,
            "inference_batch_size": args.inference_batch_size,
            "max_num_seqs": args.max_num_seqs,
            "training_sequence_length": args.training_sequence_length,
            "output_tokens": output_tokens,
            "inference_output_tokens_per_s": output_tokens / elapsed,
            "inference_median_latency_s": percentile(latency_values, 0.5),
            "inference_p95_latency_s": percentile(latency_values, 0.95),
            "inference_p99_latency_s": percentile(latency_values, 0.99),
            "measured_elapsed_s": elapsed,
        },
        "baseline": baseline_summary,
        "mixed": mixed_summary,
        "interference": {
            "throughput_retention": (
                mixed_summary["output_tokens_per_s"]
                / baseline_summary["output_tokens_per_s"]
                if baseline_summary["output_tokens_per_s"]
                else None
            ),
            "latency_p50_relative_change": relative_change(
                mixed_summary["p50_latency_s"], baseline_summary["p50_latency_s"]
            ),
            "latency_p95_relative_change": relative_change(
                mixed_summary["p95_latency_s"], baseline_summary["p95_latency_s"]
            ),
            "latency_p99_relative_change": relative_change(
                mixed_summary["p99_latency_s"], baseline_summary["p99_latency_s"]
            ),
            "ttft_p50_relative_change": relative_change(
                mixed_summary["p50_ttft_s"], baseline_summary["p50_ttft_s"]
            ),
            "ttft_p95_relative_change": relative_change(
                mixed_summary["p95_ttft_s"], baseline_summary["p95_ttft_s"]
            ),
            "tpot_p50_relative_change": relative_change(
                mixed_summary["p50_tpot_s"], baseline_summary["p50_tpot_s"]
            ),
            "tpot_p95_relative_change": relative_change(
                mixed_summary["p95_tpot_s"], baseline_summary["p95_tpot_s"]
            ),
        },
        "training": {
            "scheduled_tokens": args.training_steps * args.training_sequence_length,
            "end_to_end_tokens_per_s": args.training_steps * args.training_sequence_length / elapsed,
            "merged_forward_steps": len(merged),
            "backward_steps": len(backward),
            "published_updates": len(published),
        },
        "stability": {
            "shared_base_allocation": initialized["base_model_replaced_with_shared_vmm"],
            "base_allocation_ids": sorted(allocation_ids),
            "single_base_allocation_reused": len(allocation_ids) == 1,
            "backward_worker_pids": sorted(worker_pids),
            "single_backward_worker_reused": len(worker_pids) == 1,
            "all_numeric_values_finite": all(torch.isfinite(torch.tensor(finite_values)).tolist()),
            "merged_batches_with_inference": sum(
                bool(event.get("inference_request_ids")) for event in merged
            ),
            "reference_check": reference,
        },
        "requests": request_records,
        "baseline_requests": baseline_records,
        "trace": str(args.trace),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
