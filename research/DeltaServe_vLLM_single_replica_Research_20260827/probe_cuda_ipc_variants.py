"""Probe CUDA IPC using direct spawn arguments and a SimpleQueue."""

from __future__ import annotations

import argparse
import json

import torch
import torch.multiprocessing as mp


def consume_direct(tensor: torch.Tensor, result_queue) -> None:
    torch.cuda.set_device(0)
    torch.cuda.synchronize()
    result_queue.put({"sum": float(tensor.sum().cpu()), "device": str(tensor.device)})


def consume_queue(tensor_queue, result_queue) -> None:
    torch.cuda.set_device(0)
    tensor = tensor_queue.get()
    torch.cuda.synchronize()
    result_queue.put({"sum": float(tensor.sum().cpu()), "device": str(tensor.device)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("direct", "queue"), required=True)
    args = parser.parse_args()
    context = mp.get_context("spawn")
    result_queue = context.SimpleQueue()
    tensor = torch.arange(8, dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()

    if args.mode == "direct":
        process = context.Process(target=consume_direct, args=(tensor, result_queue))
        process.start()
    else:
        tensor_queue = context.SimpleQueue()
        process = context.Process(target=consume_queue, args=(tensor_queue, result_queue))
        process.start()
        tensor_queue.put(tensor)

    result = result_queue.get()
    process.join(60)
    result["exitcode"] = process.exitcode
    result["mode"] = args.mode
    print(json.dumps(result))


if __name__ == "__main__":
    main()
