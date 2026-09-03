"""Offline vLLM baseline benchmark used by the 2026-08-27 gap analysis."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

venv_bin = str(Path(sys.prefix) / "bin")
os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")

import torch
from vllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--use-cudagraph", action="store_true")
    return parser.parse_args()


def run_case(llm: LLM, sampling: SamplingParams, batch_size: int) -> dict[str, float | int]:
    prompts = ["Explain continuous batching in one short sentence."] * batch_size
    started = time.perf_counter()
    outputs = llm.generate(prompts, sampling, use_tqdm=False)
    elapsed = time.perf_counter() - started
    output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    return {
        "batch_size": batch_size,
        "elapsed_s": round(elapsed, 4),
        "output_tokens": output_tokens,
        "output_tokens_per_s": round(output_tokens / elapsed, 3),
    }


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "model": args.model,
                "execution_mode": "cudagraph" if args.use_cudagraph else "eager",
            }
        )
    )
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=1024,
        max_num_seqs=8,
        gpu_memory_utilization=0.55,
        enforce_eager=not args.use_cudagraph,
        disable_log_stats=True,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    print(json.dumps({"phase": "warmup", **run_case(llm, sampling, 1)}))
    print(json.dumps({"phase": "measured", **run_case(llm, sampling, 1)}))
    print(json.dumps({"phase": "measured", **run_case(llm, sampling, 4)}))


if __name__ == "__main__":
    main()
