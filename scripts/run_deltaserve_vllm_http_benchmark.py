"""Run a client-level HTTP interference benchmark against the DeltaServe server."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--inference-requests", type=int)
    parser.add_argument("--training-steps", type=int, default=100)
    parser.add_argument("--training-warmup-steps", type=int, default=1)
    parser.add_argument("--training-batch-size", type=int, default=1)
    parser.add_argument("--training-sequence-length", type=int, default=96)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    return parser.parse_args()


def post_json(url: str, payload: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def get_json(url: str, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def post_stream(url: str, payload: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    started = time.perf_counter()
    first_token_s = None
    last_token_s = None
    output_tokens = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                if not raw_line.startswith(b"data: "):
                    continue
                event = json.loads(raw_line[6:].decode("utf-8"))
                if event.get("error"):
                    raise RuntimeError(event["error"])
                token_count = int(event.get("output_tokens", 0))
                if token_count > output_tokens:
                    output_tokens = token_count
                    now = time.perf_counter()
                    first_token_s = first_token_s or now
                    last_token_s = now
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    if output_tokens <= 0 or first_token_s is None or last_token_s is None:
        raise RuntimeError("stream ended without generated tokens")
    return {
        "output_tokens": output_tokens,
        "ttft_s": first_token_s - started,
        "tpot_s": (last_token_s - first_token_s) / max(output_tokens - 1, 1),
    }


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def summarize(records: list[dict]) -> dict:
    successful = [record for record in records if record["ok"]]
    latencies = [record["latency_s"] for record in successful]
    ttft = [record["ttft_s"] for record in successful if record["ttft_s"] is not None]
    tpot = [record["tpot_s"] for record in successful if record["tpot_s"] is not None]
    if successful:
        elapsed = max(record["finished_s"] for record in successful) - min(
            record["started_s"] for record in successful
        )
        output_tokens = sum(record["output_tokens"] for record in successful)
    else:
        elapsed = 0.0
        output_tokens = 0
    return {
        "requested": len(records),
        "successful": len(successful),
        "success_rate": len(successful) / len(records) if records else 0.0,
        "output_tokens": output_tokens,
        "output_tokens_per_s": output_tokens / max(elapsed, 1e-9),
        "p50_latency_s": percentile(latencies, 0.50),
        "p95_latency_s": percentile(latencies, 0.95),
        "p99_latency_s": percentile(latencies, 0.99),
        "p50_ttft_s": percentile(ttft, 0.50),
        "p95_ttft_s": percentile(ttft, 0.95),
        "p50_tpot_s": percentile(tpot, 0.50),
        "p95_tpot_s": percentile(tpot, 0.95),
        "elapsed_s": elapsed,
        "errors": [record["error"] for record in records if not record["ok"]],
    }


async def wait_health(url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            await asyncio.sleep(1.0)
    raise TimeoutError(f"HTTP server did not become ready at {url}")


async def one_inference(url: str, index: int, max_tokens: int, timeout: float, semaphore) -> dict:
    async with semaphore:
        started = time.perf_counter()
        try:
            response = await asyncio.to_thread(
                post_stream,
                url,
                {
                    "request_id": f"http-inference-{index}",
                    "prompt": f"Request {index}: 用一句话解释连续批处理为什么能提高大模型推理吞吐。",
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                },
                timeout,
            )
            error = None
            output_tokens = int(response.get("output_tokens", 0))
        except Exception as exc:  # noqa: BLE001 - preserve per-request failure evidence
            response = {}
            error = str(exc)
            output_tokens = 0
        finished = time.perf_counter()
        return {
            "index": index,
            "ok": error is None and output_tokens > 0,
            "started_s": started,
            "finished_s": finished,
            "latency_s": finished - started,
            "output_tokens": output_tokens,
            "server_elapsed_s": response.get("server_elapsed_s"),
            "ttft_s": response.get("ttft_s"),
            "tpot_s": response.get("tpot_s"),
            "error": error,
        }


async def run_inference(url: str, count: int, clients: int, max_tokens: int, timeout: float) -> list[dict]:
    semaphore = asyncio.Semaphore(clients)
    return await asyncio.gather(
        *(one_inference(url, index, max_tokens, timeout, semaphore) for index in range(count))
    )


def training_rows(step: int, batch_size: int, sequence_length: int) -> list[list[int]]:
    return [
        [1000 + step * batch_size + row_index] + [1] * (sequence_length - 1)
        for row_index in range(batch_size)
    ]


async def one_training_step(url: str, step: int, batch_size: int, sequence_length: int, timeout: float) -> dict:
    started = time.perf_counter()
    try:
        response = await asyncio.to_thread(
            post_json,
            url,
            {"step": step, "tokens": training_rows(step, batch_size, sequence_length)},
            timeout,
        )
        error = None
    except Exception as exc:  # noqa: BLE001 - preserve per-step failure evidence
        response = {}
        error = str(exc)
    finished = time.perf_counter()
    return {
        "step": step,
        "ok": error is None,
        "started_s": started,
        "finished_s": finished,
        "latency_s": finished - started,
        "scheduled_tokens": int(response.get("scheduled_tokens", 0)),
        "server_elapsed_s": response.get("server_elapsed_s"),
        "error": error,
    }


async def run_training(url: str, steps: int, batch_size: int, sequence_length: int, timeout: float, offset: int = 0) -> list[dict]:
    records = []
    for step in range(steps):
        records.append(
            await one_training_step(url, offset + step, batch_size, sequence_length, timeout)
        )
    return records


async def benchmark(args: argparse.Namespace, root: Path) -> dict:
    request_count = args.inference_requests or args.clients * args.training_steps
    if min(args.clients, request_count, args.training_steps, args.training_batch_size) <= 0:
        raise ValueError("clients, requests, training steps, and batch size must be positive")
    if args.training_warmup_steps < 0 or args.training_sequence_length < 2:
        raise ValueError("warmup steps must be non-negative and sequence length must be at least 2")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    server_log = args.server_log or args.output.with_suffix(".server.log")
    server_log.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root) + os.pathsep + environment.get("PYTHONPATH", "")
    command = [
        sys.executable,
        str(root / "scripts" / "deltaserve_vllm_http_server.py"),
        "--model",
        args.model,
        "--trace",
        str(args.trace),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.clients + args.training_batch_size + 1),
        "--max-training-steps",
        str(args.training_warmup_steps + args.training_steps),
        "--training-max-tokens",
        str(args.training_batch_size * args.training_sequence_length),
    ]
    with server_log.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT)
        base_url = f"http://{args.host}:{args.port}"
        try:
            await wait_health(f"{base_url}/health", args.request_timeout)
            inference_url = f"{base_url}/generate/stream"
            await run_inference(inference_url, args.clients, args.clients, args.max_tokens, args.request_timeout)
            baseline_records = await run_inference(
                inference_url, request_count, args.clients, args.max_tokens, args.request_timeout
            )

            warmup_inference = asyncio.create_task(
                run_inference(inference_url, args.clients, args.clients, args.max_tokens, args.request_timeout)
            )
            warmup_training = asyncio.create_task(
                run_training(
                    f"{base_url}/train-step",
                    args.training_warmup_steps,
                    args.training_batch_size,
                    args.training_sequence_length,
                    args.request_timeout,
                    offset=0,
                )
            )
            await asyncio.gather(warmup_inference, warmup_training)

            mixed_inference, training_records = await asyncio.gather(
                run_inference(inference_url, request_count, args.clients, args.max_tokens, args.request_timeout),
                run_training(
                    f"{base_url}/train-step",
                    args.training_steps,
                    args.training_batch_size,
                    args.training_sequence_length,
                    args.request_timeout,
                    offset=args.training_warmup_steps,
                ),
            )
            try:
                gpu_metrics = await asyncio.to_thread(
                    get_json, f"{base_url}/metrics", args.request_timeout
                )
            except Exception as exc:  # noqa: BLE001 - retain missing observability evidence
                gpu_metrics = {"error": str(exc)}
            events = []
            if args.trace.exists():
                events = [json.loads(line) for line in args.trace.read_text(encoding="utf-8").splitlines() if line]
            successful_steps = [record for record in training_records if record["ok"]]
            training_elapsed = max(
                (record["finished_s"] for record in successful_steps), default=0.0
            ) - min((record["started_s"] for record in successful_steps), default=0.0)
            scheduled_tokens = sum(record["scheduled_tokens"] for record in successful_steps)
            return {
                "success": len(baseline_records) == request_count
                and all(record["ok"] for record in baseline_records)
                and len(mixed_inference) == request_count
                and all(record["ok"] for record in mixed_inference)
                and len(successful_steps) == args.training_steps,
                "config": {
                    "clients": args.clients,
                    "inference_requests": request_count,
                    "training_steps": args.training_steps,
                    "training_warmup_steps": args.training_warmup_steps,
                    "training_batch_size": args.training_batch_size,
                    "training_sequence_length": args.training_sequence_length,
                    "max_tokens": args.max_tokens,
                    "max_model_len": args.max_model_len,
                    "max_num_seqs": args.clients + args.training_batch_size + 1,
                    "server_url": base_url,
                },
                "baseline": summarize(baseline_records),
                "mixed": summarize(mixed_inference),
                "training": {
                    "steps_requested": args.training_steps,
                    "steps_completed": len(successful_steps),
                    "scheduled_tokens": scheduled_tokens,
                    "scheduled_tokens_per_s": scheduled_tokens / max(training_elapsed, 1e-9),
                    "p50_step_latency_s": percentile([record["latency_s"] for record in successful_steps], 0.5),
                    "p95_step_latency_s": percentile([record["latency_s"] for record in successful_steps], 0.95),
                    "records": training_records,
                },
                "trace": str(args.trace),
                "server_log": str(server_log),
                "gpu_metrics": gpu_metrics,
                "trace_event_counts": {
                    "merged_forward": sum(event.get("event") == "merged_forward" for event in events),
                    "backward_finished": sum(event.get("event") == "backward_finished" for event in events),
                    "adapter_published": sum(event.get("event") == "adapter_published" for event in events),
                },
            }
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    result = asyncio.run(benchmark(args, root))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
