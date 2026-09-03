"""Run a real single-GPU vLLM inference + LoRA training experiment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
venv_bin = str(Path(sys.prefix) / "bin")
os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from engine.concurrent_vllm_lora import (
    TrainingWorkerConfig,
    count_request_step_overlaps,
    percentile,
    run_training_worker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Existing local Hugging Face model directory")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--baseline-requests", type=int, default=4)
    parser.add_argument(
        "--concurrent-requests",
        type=int,
        default=None,
        help="Stop after this many inference requests while the training worker is running",
    )
    parser.add_argument("--training-steps", type=int, default=8)
    parser.add_argument("--training-batch-size", type=int, default=1)
    parser.add_argument("--training-sequence-length", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--worker-ready-timeout", type=float, default=180.0)
    parser.add_argument("--experiment-timeout", type=float, default=300.0)
    parser.add_argument(
        "--mixed-mode",
        choices=("batch", "paired-step"),
        default="batch",
        help="Schedule mixed inference as one batch or pair one request with each training step",
    )
    parser.add_argument(
        "--mixed-warmup-steps",
        type=int,
        default=0,
        help="Run paired training/inference steps outside the measured mixed window",
    )
    parser.add_argument("--training-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--ready-file", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--go-file", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--result-file", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--adapter-output-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--step-go-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--step-done-dir", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def worker_main(args: argparse.Namespace) -> None:
    required = (args.ready_file, args.go_file, args.result_file, args.adapter_output_dir)
    if any(value is None for value in required):
        raise ValueError("training worker requires ready, go, and result files")
    run_training_worker(
        TrainingWorkerConfig(
            model=args.model,
            steps=args.training_steps,
            sequence_length=args.training_sequence_length,
            learning_rate=args.learning_rate,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            ready_file=args.ready_file,
            go_file=args.go_file,
            result_file=args.result_file,
            adapter_output_dir=args.adapter_output_dir,
            start_timeout_s=args.worker_ready_timeout,
            step_go_dir=args.step_go_dir,
            step_done_dir=args.step_done_dir,
            batch_size=args.training_batch_size,
        )
    )


def run_request(llm, sampling, phase: str, index: int, *, lora_request=None) -> dict:
    prompt = f"Request {index}: 用一句话解释连续批处理为什么能提高大模型推理吞吐。"
    started = time.time()
    outputs = llm.generate(
        [prompt],
        sampling,
        use_tqdm=False,
        lora_request=lora_request,
    )
    finished = time.time()
    output = outputs[0].outputs[0]
    return {
        "phase": phase,
        "index": index,
        "started_wall_s": started,
        "finished_wall_s": finished,
        "latency_s": finished - started,
        "output_tokens": len(output.token_ids),
        "text": output.text.strip(),
    }


def output_throughput(requests: list[dict]) -> float:
    if not requests:
        return 0.0
    elapsed = max(request["finished_wall_s"] for request in requests) - min(
        request["started_wall_s"] for request in requests
    )
    return sum(request["output_tokens"] for request in requests) / max(elapsed, 1e-9)


def run_concurrent_requests(llm, sampling, count: int, timeout_s: float) -> list[dict]:
    started: dict[str, float] = {}
    request_ids = []
    for index in range(count):
        request_id = f"concurrent-{index}"
        request_ids.append(request_id)
        started[request_id] = time.time()
        llm.llm_engine.add_request(
            request_id,
            f"Request {index}: 用一句话解释连续批处理为什么能提高大模型推理吞吐。",
            sampling,
        )

    completed: list[dict] = []
    deadline = time.monotonic() + timeout_s
    while llm.llm_engine.has_unfinished_requests():
        if time.monotonic() >= deadline:
            raise TimeoutError("concurrent inference exceeded its timeout")
        for output in llm.llm_engine.step():
            if not output.finished:
                continue
            finished = time.time()
            generated = output.outputs[0]
            completed.append(
                {
                    "phase": "concurrent",
                    "index": request_ids.index(output.request_id),
                    "started_wall_s": started[output.request_id],
                    "finished_wall_s": finished,
                    "latency_s": finished - started[output.request_id],
                    "output_tokens": len(generated.token_ids),
                    "text": generated.text.strip(),
                }
            )
    return sorted(completed, key=lambda request: request["index"])


def run_paired_step_requests(
    llm,
    sampling,
    count: int,
    step_go_dir: Path,
    step_done_dir: Path,
    timeout_s: float,
    step_offset: int = 0,
) -> list[dict]:
    completed: list[dict] = []
    deadline = time.monotonic() + timeout_s
    for step in range(count):
        if time.monotonic() >= deadline:
            raise TimeoutError("paired-step inference exceeded its timeout")
        runtime_step = step + step_offset
        request_id = f"paired-{runtime_step}"
        started = time.time()
        llm.llm_engine.add_request(
            request_id,
            f"Request {step}: 用一句话解释连续批处理为什么能提高大模型推理吞吐。",
            sampling,
        )
        (step_go_dir / f"step-{runtime_step}").touch()
        inference_output = None
        while llm.llm_engine.has_unfinished_requests():
            if time.monotonic() >= deadline:
                raise TimeoutError("paired-step inference exceeded its timeout")
            for output in llm.llm_engine.step():
                if output.finished and output.request_id == request_id:
                    inference_output = output.outputs[0]
        if inference_output is None:
            raise RuntimeError(f"no inference output for paired step {step}")
        finished = time.time()
        done_marker = step_done_dir / f"step-{runtime_step}"
        while not done_marker.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"training step {runtime_step} did not finish")
            time.sleep(0.01)
        completed.append(
            {
                "phase": "concurrent",
                "step": step,
                "index": step,
                "started_wall_s": started,
                "finished_wall_s": finished,
                "latency_s": finished - started,
                "output_tokens": len(inference_output.token_ids),
                "text": inference_output.text.strip(),
            }
        )
    return completed


def wait_for_ready(path: Path, process: subprocess.Popen, timeout_s: float) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        if process.poll() is not None:
            stdout, _ = process.communicate()
            raise RuntimeError(f"training worker exited before ready:\n{stdout}")
        time.sleep(0.05)
    process.terminate()
    raise TimeoutError("training worker did not become ready")


def controller_main(args: argparse.Namespace) -> None:
    import torch
    from vllm import LLM, SamplingParams

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside WSL")
    model_path = Path(args.model)
    if not model_path.is_dir():
        raise FileNotFoundError(f"use an existing local model directory: {model_path}")
    if args.mixed_warmup_steps < 0:
        raise ValueError("mixed warmup steps must be non-negative")
    if args.mixed_warmup_steps and args.mixed_mode != "paired-step":
        raise ValueError("mixed warmup steps require paired-step mode")
    if args.training_batch_size <= 0:
        raise ValueError("training batch size must be positive")
    supported_vllm_ranks = (1, 8, 16, 32, 64, 128, 256, 320, 512)
    try:
        vllm_max_lora_rank = next(rank for rank in supported_vllm_ranks if rank >= args.lora_rank)
    except StopIteration as exc:
        raise ValueError(f"LoRA rank {args.lora_rank} exceeds vLLM's supported maximum") from exc

    llm = LLM(
        model=str(model_path),
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs or max(8, args.training_batch_size + 1),
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        async_scheduling=False,
        disable_log_stats=True,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=vllm_max_lora_rank,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    run_request(llm, sampling, "warmup", 0)
    baseline = [
        run_request(llm, sampling, "baseline", index)
        for index in range(args.baseline_requests)
    ]

    run_dir = Path(tempfile.mkdtemp(prefix="vllm-lora-concurrent-"))
    ready_file = run_dir / "ready.json"
    go_file = run_dir / "go"
    result_file = run_dir / "training-result.json"
    adapter_output_dir = (
        args.output.parent / f"{args.output.stem}-adapter"
        if args.output is not None
        else run_dir / "adapter"
    )
    step_go_dir = run_dir / "step-go"
    step_done_dir = run_dir / "step-done"
    if args.mixed_mode == "paired-step":
        step_go_dir.mkdir()
        step_done_dir.mkdir()
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--training-worker",
        "--model",
        str(model_path),
        "--training-steps",
        str(args.training_steps + args.mixed_warmup_steps),
        "--training-sequence-length",
        str(args.training_sequence_length),
        "--learning-rate",
        str(args.learning_rate),
        "--lora-rank",
        str(args.lora_rank),
        "--lora-alpha",
        str(args.lora_alpha),
        "--worker-ready-timeout",
        str(args.worker_ready_timeout),
        "--ready-file",
        str(ready_file),
        "--go-file",
        str(go_file),
        "--result-file",
        str(result_file),
        "--adapter-output-dir",
        str(adapter_output_dir),
    ]
    if args.mixed_mode == "paired-step":
        command.extend(
            [
                "--step-go-dir",
                str(step_go_dir),
                "--step-done-dir",
                str(step_done_dir),
            ]
        )
    worker = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    ready = wait_for_ready(ready_file, worker, args.worker_ready_timeout)
    go_file.touch()

    concurrent_count = args.concurrent_requests
    if concurrent_count is None:
        concurrent_count = max(args.baseline_requests, args.training_steps)
    if concurrent_count <= 0:
        raise ValueError("concurrent request count must be positive")
    try:
        if args.mixed_mode == "paired-step":
            if args.mixed_warmup_steps:
                run_paired_step_requests(
                    llm,
                    sampling,
                    args.mixed_warmup_steps,
                    step_go_dir,
                    step_done_dir,
                    args.experiment_timeout,
                )
            concurrent = run_paired_step_requests(
                llm,
                sampling,
                concurrent_count,
                step_go_dir,
                step_done_dir,
                args.experiment_timeout,
                step_offset=args.mixed_warmup_steps,
            )
        else:
            concurrent = run_concurrent_requests(
                llm,
                sampling,
                concurrent_count,
                args.experiment_timeout,
            )
    except Exception:
        if worker.poll() is None:
            worker.terminate()
            worker.wait()
        raise

    worker_stdout, _ = worker.communicate()
    if not result_file.exists():
        raise RuntimeError(f"training worker produced no result:\n{worker_stdout}")
    worker_training = json.loads(result_file.read_text(encoding="utf-8"))
    if worker.returncode != 0 or worker_training.get("phase") != "finished":
        raise RuntimeError(
            f"training worker failed with code {worker.returncode}:\n"
            f"{worker_stdout}\n{json.dumps(worker_training, ensure_ascii=False, indent=2)}"
        )
    measured_steps = worker_training["steps"][args.mixed_warmup_steps :]
    if len(measured_steps) != args.training_steps:
        raise RuntimeError(
            "training worker produced an unexpected measured step count: "
            f"{len(measured_steps)} != {args.training_steps}"
        )
    training = {
        **worker_training,
        "warmup_steps": args.mixed_warmup_steps,
        "steps": measured_steps,
        "training_started_wall_s": measured_steps[0]["started_wall_s"],
        "training_finished_wall_s": measured_steps[-1]["finished_wall_s"],
        "training_elapsed_s": measured_steps[-1]["finished_wall_s"] - measured_steps[0]["started_wall_s"],
        "training_tokens": sum(step["token_count"] for step in measured_steps),
        "training_tokens_per_s": sum(step["token_count"] for step in measured_steps)
        / max(measured_steps[-1]["finished_wall_s"] - measured_steps[0]["started_wall_s"], 1e-9),
        "initial_loss": measured_steps[0]["loss"],
        "final_loss": measured_steps[-1]["loss"],
    }

    from vllm.lora.request import LoRARequest

    adapter_request = LoRARequest("concurrent-trained", 1, training["adapter"]["path"])
    adapter_first_request = run_request(
        llm,
        sampling,
        "trained_adapter_load",
        0,
        lora_request=adapter_request,
    )
    adapter_warm_request = run_request(
        llm,
        sampling,
        "trained_adapter_warm",
        1,
        lora_request=adapter_request,
    )

    overlapping_requests, overlapping_steps = count_request_step_overlaps(
        concurrent,
        training["steps"],
    )
    baseline_latencies = [request["latency_s"] for request in baseline]
    concurrent_latencies = [request["latency_s"] for request in concurrent]
    summary = {
        "success": bool(
            concurrent
            and training["parameter_delta_l1"] > 0
            and overlapping_requests > 0
            and overlapping_steps > 0
            and adapter_first_request["output_tokens"] > 0
            and adapter_warm_request["output_tokens"] > 0
        ),
        "environment": {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "model": str(model_path),
            "vllm_gpu_memory_utilization": args.gpu_memory_utilization,
            "vllm_max_lora_rank": vllm_max_lora_rank,
            "gpu_memory": {
                "allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 2),
                "reserved_mib": round(torch.cuda.memory_reserved() / 2**20, 2),
                "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 2),
                "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 2),
            },
        },
        "worker_ready": ready,
        "inference": {
            "baseline_requests": len(baseline),
            "concurrent_requests": len(concurrent),
            "concurrent_request_target": args.concurrent_requests,
            "baseline_output_tokens": sum(request["output_tokens"] for request in baseline),
            "concurrent_output_tokens": sum(request["output_tokens"] for request in concurrent),
            "baseline_output_tokens_per_s": output_throughput(baseline),
            "concurrent_output_tokens_per_s": output_throughput(concurrent),
            "baseline_median_latency_s": percentile(baseline_latencies, 0.5),
            "baseline_p95_latency_s": percentile(baseline_latencies, 0.95),
            "concurrent_median_latency_s": percentile(concurrent_latencies, 0.5),
            "concurrent_p95_latency_s": percentile(concurrent_latencies, 0.95),
            "concurrent_to_baseline_median_ratio": (
                percentile(concurrent_latencies, 0.5) / percentile(baseline_latencies, 0.5)
            ),
            "requests": baseline + concurrent,
            "mixed_mode": args.mixed_mode,
        },
        "training": training,
        "trained_adapter_validation": {
            "first_request_includes_load_and_jit": adapter_first_request,
            "warm_request": adapter_warm_request,
        },
        "overlap": {
            "requests_overlapping_training_steps": overlapping_requests,
            "training_steps_overlapping_inference": overlapping_steps,
        },
        "worker_stdout": worker_stdout,
        "temporary_run_dir": str(run_dir),
    }
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    if not summary["success"]:
        raise RuntimeError("training updated no parameters or no inference/training step overlap was observed")


def main() -> None:
    args = parse_args()
    if args.training_worker:
        worker_main(args)
    else:
        controller_main(args)


if __name__ == "__main__":
    main()
