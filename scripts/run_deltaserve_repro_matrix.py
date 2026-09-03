"""Repeat the separated and shared DeltaEngine benchmarks with one config."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--baseline-requests", type=int, default=8)
    parser.add_argument("--training-steps", type=int, default=8)
    parser.add_argument("--training-batch-size", type=int, default=1)
    parser.add_argument("--training-sequence-length", type=int, default=96)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--sampler", choices=("native", "flashinfer"), default="native")
    parser.add_argument(
        "--separated-mixed-mode",
        choices=("batch", "paired-step"),
        default="paired-step",
    )
    parser.add_argument("--mixed-warmup-steps", type=int, default=1)
    return parser.parse_args()


def run_json_script(
    command: list[str],
    output_path: Path,
    log_path: Path,
    env: dict[str, str],
    timeout_s: float,
) -> dict:
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    if completed.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"benchmark failed ({completed.returncode}): {log_path}\n{tail}")
    if not output_path.is_file():
        raise RuntimeError(f"benchmark produced no JSON result: {output_path}")
    result = json.loads(output_path.read_text(encoding="utf-8"))
    if not result.get("success"):
        raise RuntimeError(f"benchmark reported failure: {output_path}")
    return result


def metric_summary(results: list[dict], path: tuple[str, ...]) -> dict:
    values: list[float] = []
    for result in results:
        value: object = result
        for key in path:
            if not isinstance(value, dict):
                raise KeyError(".".join(path))
            value = value[key]
        values.append(float(value))
    mean = statistics.fmean(values)
    deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "values": values,
        "mean": mean,
        "stdev": deviation,
        "coefficient_of_variation": deviation / mean if mean else 0.0,
    }


def build_environment(sampler: str) -> dict[str, str]:
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    environment["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    environment["VLLM_USE_FLASHINFER_SAMPLER"] = "1" if sampler == "flashinfer" else "0"
    cuda_root = Path(sys.prefix) / "lib/python3.10/site-packages/nvidia/cu13"
    if (cuda_root / "bin/nvcc").is_file():
        environment["CUDA_HOME"] = str(cuda_root)
        environment["CUDA_PATH"] = str(cuda_root)
        environment["CUDACXX"] = str(cuda_root / "bin/nvcc")
        environment["FLASHINFER_NVCC"] = str(cuda_root / "bin/nvcc")
        environment["PATH"] = str(cuda_root / "bin") + os.pathsep + environment.get("PATH", "")
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(
            [str(cuda_root / "lib"), str(cuda_root / "lib64"), environment.get("LD_LIBRARY_PATH", "")]
        )
        environment["CPATH"] = os.pathsep.join(
            [str(cuda_root / "include/cccl"), str(cuda_root / "include"), environment.get("CPATH", "")]
        )
        environment["CPLUS_INCLUDE_PATH"] = environment["CPATH"]
    return environment


def toolchain_snapshot(environment: dict[str, str]) -> dict[str, object]:
    nvcc = shutil.which("nvcc", path=environment.get("PATH"))
    nvcc_version = "not-found"
    if nvcc:
        completed = subprocess.run(
            [nvcc, "--version"],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        nvcc_version = (completed.stdout or completed.stderr).strip()
    versions = {}
    for package in ("vllm", "torch", "flashinfer-python", "flashinfer-cubin"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return {
        "nvcc": nvcc or "not-found",
        "nvcc_version": nvcc_version,
        "cuda_home": environment.get("CUDA_HOME"),
        "cuda_path": environment.get("CUDA_PATH"),
        "cudacxx": environment.get("CUDACXX"),
        "flashinfer_nvcc": environment.get("FLASHINFER_NVCC"),
        "ld_library_path": environment.get("LD_LIBRARY_PATH", ""),
        "packages": versions,
    }


def main() -> None:
    args = parse_args()
    if (
        args.repetitions <= 0
        or args.baseline_requests <= 0
        or args.training_steps <= 0
        or args.training_batch_size <= 0
    ):
        raise ValueError("repetitions, baseline requests, training steps, and batch size must be positive")
    model = Path(args.model)
    if not model.is_dir():
        raise FileNotFoundError(model)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    environment = build_environment(args.sampler)
    separated_results: list[dict] = []
    deltaserve_results: list[dict] = []
    for repetition in range(1, args.repetitions + 1):
        separated_output = args.output_dir / f"separated-r{repetition}.json"
        separated_log = args.output_dir / f"separated-r{repetition}.log"
        separated_result = run_json_script(
                [
                    sys.executable,
                    str(ROOT / "scripts/run_vllm_lora_concurrent.py"),
                    "--model",
                    str(model),
                    "--output",
                    str(separated_output),
                    "--gpu-memory-utilization",
                    str(args.gpu_memory_utilization),
                    "--max-model-len",
                    str(args.max_model_len),
                    "--max-num-seqs",
                    str(max(8, args.training_batch_size + 1)),
                    "--max-tokens",
                    str(args.max_tokens),
                    "--baseline-requests",
                    str(args.baseline_requests),
                    "--concurrent-requests",
                    str(args.training_steps),
                    "--training-steps",
                    str(args.training_steps),
                    "--training-batch-size",
                    str(args.training_batch_size),
                    "--training-sequence-length",
                    str(args.training_sequence_length),
                    "--mixed-mode",
                    args.separated_mixed_mode,
                    "--mixed-warmup-steps",
                    str(args.mixed_warmup_steps),
                ],
                separated_output,
                separated_log,
                environment,
                args.timeout,
            )
        if separated_result["inference"]["concurrent_requests"] != args.training_steps:
            raise RuntimeError(
                "separated benchmark ended before the requested concurrent request count: "
                f"{separated_result['inference']['concurrent_requests']} != {args.training_steps}"
            )
        if separated_result["overlap"]["training_steps_overlapping_inference"] <= 0:
            raise RuntimeError("separated benchmark did not overlap training and inference")
        separated_results.append(separated_result)

        delta_output = args.output_dir / f"deltaserve-r{repetition}.json"
        delta_trace = args.output_dir / f"deltaserve-r{repetition}.jsonl"
        delta_log = args.output_dir / f"deltaserve-r{repetition}.log"
        delta_result = run_json_script(
                [
                    sys.executable,
                    str(ROOT / "scripts/run_deltaserve_vllm_benchmark.py"),
                    "--model",
                    str(model),
                    "--output",
                    str(delta_output),
                    "--trace",
                    str(delta_trace),
                    "--gpu-memory-utilization",
                    str(args.gpu_memory_utilization),
                    "--max-model-len",
                    str(args.max_model_len),
                    "--max-tokens",
                    str(args.max_tokens),
                    "--baseline-requests",
                    str(args.baseline_requests),
                    "--training-steps",
                    str(args.training_steps),
                    "--training-batch-size",
                    str(args.training_batch_size),
                    "--training-sequence-length",
                    str(args.training_sequence_length),
                    "--mixed-warmup-steps",
                    str(args.mixed_warmup_steps),
                ],
                delta_output,
                delta_log,
                environment,
                args.timeout,
            )
        if delta_result["inference"]["mixed_requests"] != args.training_steps:
            raise RuntimeError(
                "DeltaEngine benchmark produced an unexpected mixed request count: "
                f"{delta_result['inference']['mixed_requests']} != {args.training_steps}"
            )
        deltaserve_results.append(delta_result)

    summary = {
        "success": True,
        "config": {
            "model": str(model),
            "repetitions": args.repetitions,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "max_tokens": args.max_tokens,
            "baseline_requests": args.baseline_requests,
            "concurrent_requests": args.training_steps,
            "training_steps": args.training_steps,
            "training_batch_size": args.training_batch_size,
            "max_num_seqs": max(8, args.training_batch_size + 1),
            "training_sequence_length": args.training_sequence_length,
            "sampler": args.sampler,
            "separated_mixed_mode": args.separated_mixed_mode,
            "mixed_warmup_steps": args.mixed_warmup_steps,
        },
        "toolchain": toolchain_snapshot(environment),
        "separated": {
            "runs": separated_results,
            "metrics": {
                "baseline_output_tokens_per_s": metric_summary(
                    separated_results, ("inference", "baseline_output_tokens_per_s")
                ),
                "concurrent_output_tokens_per_s": metric_summary(
                    separated_results, ("inference", "concurrent_output_tokens_per_s")
                ),
                "baseline_p50_s": metric_summary(
                    separated_results, ("inference", "baseline_median_latency_s")
                ),
                "baseline_p95_s": metric_summary(
                    separated_results, ("inference", "baseline_p95_latency_s")
                ),
                "concurrent_p50_s": metric_summary(
                    separated_results, ("inference", "concurrent_median_latency_s")
                ),
                "concurrent_p95_s": metric_summary(
                    separated_results, ("inference", "concurrent_p95_latency_s")
                ),
                "training_tokens_per_s": metric_summary(
                    separated_results, ("training", "training_tokens_per_s")
                ),
            },
        },
        "deltaserve": {
            "runs": deltaserve_results,
            "metrics": {
                "baseline_output_tokens_per_s": metric_summary(
                    deltaserve_results, ("inference", "baseline_output_tokens_per_s")
                ),
                "mixed_output_tokens_per_s": metric_summary(
                    deltaserve_results, ("inference", "mixed_output_tokens_per_s")
                ),
                "baseline_p50_s": metric_summary(
                    deltaserve_results, ("inference", "baseline_median_latency_s")
                ),
                "baseline_p95_s": metric_summary(
                    deltaserve_results, ("inference", "baseline_p95_latency_s")
                ),
                "mixed_p50_s": metric_summary(
                    deltaserve_results, ("inference", "mixed_median_latency_s")
                ),
                "mixed_p95_s": metric_summary(
                    deltaserve_results, ("inference", "mixed_p95_latency_s")
                ),
                "end_to_end_training_tokens_per_s": metric_summary(
                    deltaserve_results, ("training", "end_to_end_tokens_per_s")
                ),
            },
        },
    }
    output = args.output_dir / "repro-matrix.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
