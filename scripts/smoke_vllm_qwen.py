"""Run a minimal GPU smoke test for the vLLM environment.

This validates the baseline vLLM runtime only.  The DeltaServe-style execution
prototype still needs the documented vLLM fork before mixed training/inference
can run on the model worker.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# vLLM starts an EngineCore subprocess.  When this script is launched through
# ``.venv/bin/python`` rather than an activated shell, that subprocess does not
# otherwise see pip-installed build helpers such as ``ninja``.
venv_bin = str(Path(sys.prefix) / "bin")
os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")
# The cluster's system CUDA headers are incomplete for FlashInfer sampling.
# Keep the smoke test runnable with vLLM's PyTorch-native sampler by default.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
flashinfer_nvcc = Path(sys.prefix) / "lib/python3.10/site-packages/nvidia/cu13/bin/nvcc"
if flashinfer_nvcc.is_file():
    os.environ.setdefault("FLASHINFER_NVCC", str(flashinfer_nvcc))

import torch
from vllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--max-model-len", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside this WSL environment")

    print(
        {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "model": args.model,
        }
    )
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        max_num_seqs=8,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        disable_log_stats=True,
    )
    outputs = llm.generate(
        ["用一句中文说明 vLLM 的连续批处理有什么作用。"],
        SamplingParams(temperature=0.0, max_tokens=48),
    )
    print(outputs[0].outputs[0].text.strip())


if __name__ == "__main__":
    main()
