"""Probe CUDA tensor sharing across spawned processes in WSL."""

from __future__ import annotations

import json

import torch
import torch.multiprocessing as mp


def consume(tensor_queue: mp.Queue, result_queue: mp.Queue) -> None:
    tensor = tensor_queue.get(timeout=60)
    result_queue.put(
        {
            "device": str(tensor.device),
            "sum": float(tensor.sum().cpu()),
            "is_cuda": tensor.is_cuda,
        }
    )


def main() -> None:
    context = mp.get_context("spawn")
    tensor_queue = context.Queue()
    result_queue = context.Queue()
    tensor = torch.arange(8, dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()
    process = context.Process(target=consume, args=(tensor_queue, result_queue))
    process.start()
    tensor_queue.put(tensor)
    result = result_queue.get(timeout=60)
    process.join(timeout=60)
    result["child_exitcode"] = process.exitcode
    print(json.dumps(result))
    if result != {"device": "cuda:0", "sum": 28.0, "is_cuda": True, "child_exitcode": 0}:
        raise RuntimeError(f"CUDA IPC probe failed: {result}")


if __name__ == "__main__":
    main()
