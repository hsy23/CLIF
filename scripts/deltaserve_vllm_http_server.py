"""Minimal HTTP wrapper for the DeltaServe vLLM single-replica prototype.

The server exposes normal generation plus a small training control endpoint.
Both endpoints submit requests to the same patched AsyncLLM engine, so the
HTTP benchmark can measure client-level interference without loading a second
base model.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any


def _gpu_metrics() -> dict:
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
            sample = _gpu_metrics()
            if sample:
                sample["wall_time_s"] = time.time()
                self.samples.append(sample)
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self._thread.start()

    def snapshot(self) -> dict:
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

    def stop(self) -> dict:
        self._stop.set()
        self._thread.join(timeout=3.0)
        return self.snapshot()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=33)
    parser.add_argument("--max-training-steps", type=int, required=True)
    parser.add_argument("--training-max-tokens", type=int, required=True)
    return parser.parse_args()


def _prompt(value: Any) -> str | dict[str, list[int]]:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(token, int) for token in value):
        return {"prompt_token_ids": value}
    raise ValueError("prompt must be a string or a list of integer token ids")


async def _collect(engine, request_id: str, prompt, sampling_params):
    last = None
    async for output in engine.generate(prompt, sampling_params, request_id):
        last = output
    if last is None or not last.outputs:
        raise RuntimeError(f"request {request_id} produced no output")
    return last.outputs[0]


def _trace_event_count(trace_path: Path, event_name: str) -> int:
    try:
        lines = trace_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return 0
    count = 0
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        count += event.get("event") == event_name
    return count


async def _wait_for_trace_event(
    trace_path: Path,
    event_name: str,
    previous_count: int,
    timeout_s: float,
) -> int:
    deadline = time.monotonic() + timeout_s
    while True:
        count = _trace_event_count(trace_path, event_name)
        if count > previous_count:
            return count
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"timed out waiting for DeltaServe {event_name} after {timeout_s:.1f}s"
            )
        await asyncio.sleep(0.02)


def create_app(args: argparse.Namespace):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.entrypoints.openai.api_server import build_async_engine_client_from_engine_args
    from vllm.sampling_params import SamplingParams
    from vllm.usage.usage_lib import UsageContext

    args.trace.parent.mkdir(parents=True, exist_ok=True)
    args.trace.unlink(missing_ok=True)
    gpu_sampler = GPUMetricsSampler()
    training_lock = asyncio.Lock()
    training_sync_timeout_s = float(
        os.environ.get("CLIF_DELTASERVE_TRAINING_SYNC_TIMEOUT_S", "30")
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        gpu_sampler.start()
        engine_args = AsyncEngineArgs(
            model=args.model,
            dtype="bfloat16",
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=True,
            async_scheduling=False,
            disable_log_stats=True,
        )
        try:
            async with build_async_engine_client_from_engine_args(
                engine_args,
                usage_context=UsageContext.OPENAI_API_SERVER,
            ) as engine:
                app.state.engine = engine
                app.state.gpu_sampler = gpu_sampler
                yield
        finally:
            app.state.gpu_observation = gpu_sampler.stop()

    app = FastAPI(title="DeltaServe vLLM HTTP prototype", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics() -> dict[str, Any]:
        import torch

        current = _gpu_metrics()
        return {
            "gpu": torch.cuda.get_device_name(0),
            "gpu_utilization_percent": current.get("utilization_percent"),
            "allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 2),
            "reserved_mib": round(torch.cuda.memory_reserved() / 2**20, 2),
            "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 2),
            "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 2),
            "nvidia_smi": {
                **app.state.gpu_sampler.snapshot(),
                "current": current,
            },
        }

    @app.post("/generate")
    async def generate(payload: dict[str, Any]) -> dict[str, Any]:
        request_id = str(payload.get("request_id") or f"http-inference-{uuid.uuid4().hex}")
        started = time.perf_counter()
        try:
            prompt = _prompt(payload.get("prompt"))
            max_tokens = int(payload.get("max_tokens", 24))
            temperature = float(payload.get("temperature", 0.0))
            output = await _collect(
                app.state.engine,
                request_id,
                prompt,
                SamplingParams(temperature=temperature, max_tokens=max_tokens),
            )
        except (ValueError, TypeError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "request_id": request_id,
            "text": output.text.strip(),
            "output_tokens": len(output.token_ids),
            "server_elapsed_s": time.perf_counter() - started,
        }

    @app.post("/generate/stream")
    async def generate_stream(payload: dict[str, Any]):
        request_id = str(payload.get("request_id") or f"http-inference-{uuid.uuid4().hex}")
        try:
            prompt = _prompt(payload.get("prompt"))
            max_tokens = int(payload.get("max_tokens", 24))
            temperature = float(payload.get("temperature", 0.0))
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        async def events():
            try:
                async for output in app.state.engine.generate(
                    prompt,
                    SamplingParams(temperature=temperature, max_tokens=max_tokens),
                    request_id,
                ):
                    generated = output.outputs[0]
                    yield "data: " + json.dumps(
                        {
                            "request_id": request_id,
                            "text": generated.text,
                            "output_tokens": len(generated.token_ids),
                            "finished": output.finished,
                        },
                        ensure_ascii=False,
                    ) + "\n\n"
            except Exception as exc:  # noqa: BLE001 - expose stream failure to the client
                yield "data: " + json.dumps({"error": str(exc)}, ensure_ascii=False) + "\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.post("/train-step")
    async def train_step(payload: dict[str, Any]) -> dict[str, Any]:
        rows = payload.get("tokens")
        if not isinstance(rows, list) or not rows or not all(isinstance(row, list) for row in rows):
            raise HTTPException(status_code=400, detail="tokens must be a non-empty list of token rows")
        if not all(row and all(isinstance(token, int) for token in row) for row in rows):
            raise HTTPException(status_code=400, detail="each token row must contain integer ids")
        step = int(payload.get("step", 0))
        started = time.perf_counter()
        async with training_lock:
            published_before = _trace_event_count(args.trace, "adapter_published")
            tasks = [
                asyncio.create_task(
                    _collect(
                        app.state.engine,
                        f"deltaserve-ft-http-{step}-{row_index}",
                        {"prompt_token_ids": row},
                        SamplingParams(temperature=0.0, max_tokens=1),
                    )
                )
                for row_index, row in enumerate(rows)
            ]
            try:
                outputs = await asyncio.gather(*tasks)
                await _wait_for_trace_event(
                    args.trace,
                    "adapter_published",
                    published_before,
                    training_sync_timeout_s,
                )
            except (ValueError, TypeError, RuntimeError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "step": step,
            "training_rows": len(rows),
            "scheduled_tokens": sum(len(row) for row in rows),
            "output_tokens": sum(len(output.token_ids) for output in outputs),
            "server_elapsed_s": time.perf_counter() - started,
        }

    return app


def main() -> None:
    args = parse_args()
    os.environ["CLIF_DELTASERVE_ENABLE"] = "1"
    os.environ["CLIF_DELTASERVE_TRACE"] = str(args.trace.resolve())
    os.environ["CLIF_DELTASERVE_RANK"] = "4"
    os.environ["CLIF_DELTASERVE_ALPHA"] = "8"
    os.environ["CLIF_DELTASERVE_MAX_TOKENS"] = str(args.training_max_tokens)
    os.environ["CLIF_DELTASERVE_MAX_STEPS"] = str(args.max_training_steps)
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    import uvicorn

    uvicorn.run(create_app(args), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
