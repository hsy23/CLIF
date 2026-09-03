"""Run a one-shot DeltaServe-principle vLLM merged-forward/backward test."""

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
# The cluster's system CUDA headers are incomplete for FlashInfer sampling.
# Keep the hook prototype runnable with vLLM's PyTorch-native sampler by default.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
flashinfer_nvcc = Path(sys.prefix) / "lib/python3.10/site-packages/nvidia/cu13/bin/nvcc"
if flashinfer_nvcc.is_file():
    os.environ.setdefault("FLASHINFER_NVCC", str(flashinfer_nvcc))


def read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def wait_for_event(path: Path, event_name: str, timeout_s: float) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for event in read_events(path):
            if event.get("event") == event_name:
                return event
        time.sleep(0.05)
    raise TimeoutError(f"did not observe {event_name} in {path}")


def drain_engine(llm) -> list:
    """Mirror vLLM's offline loop without its numeric request-id sorting."""
    outputs = []
    while llm.llm_engine.has_unfinished_requests():
        for output in llm.llm_engine.step():
            if output.finished:
                outputs.append(output)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not Path(args.model).is_dir():
        raise FileNotFoundError(args.model)
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    args.trace.unlink(missing_ok=True)
    os.environ["CLIF_DELTASERVE_ENABLE"] = "1"
    os.environ["CLIF_DELTASERVE_TRACE"] = str(args.trace.resolve())
    os.environ["CLIF_DELTASERVE_RANK"] = "4"
    os.environ["CLIF_DELTASERVE_ALPHA"] = "8"
    os.environ["CLIF_DELTASERVE_COMPUTE_FORWARD_LOSS"] = "1"
    os.environ["CLIF_DELTASERVE_MAX_TOKENS"] = "128"

    import torch
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=256,
        max_num_seqs=8,
        gpu_memory_utilization=0.42,
        enforce_eager=True,
        async_scheduling=False,
        disable_log_stats=True,
    )
    params = SamplingParams(temperature=0.0, max_tokens=1)
    training_text = (
        "训推并发通过共享基础模型，把微调前向作为 synthetic prefill row 合入推理批次。"
        "独立子进程读取共享 activation 并更新 LoRA。"
    )
    inference_text = "用一句话解释连续批处理。"
    llm.llm_engine.add_request("deltaserve-ft-0", training_text, params)
    llm.llm_engine.add_request("inference-prefill-0", inference_text, params)
    merged_outputs = drain_engine(llm)
    backward = wait_for_event(args.trace, "backward_finished", 120)

    llm.llm_engine.add_request(
        "inference-after-backward-0",
        "说明 LoRA 微调的作用。",
        SamplingParams(temperature=0.0, max_tokens=8),
    )
    post_outputs = drain_engine(llm)
    published = wait_for_event(args.trace, "adapter_published", 30)
    events = read_events(args.trace)
    initialized = next(event for event in events if event["event"] == "runtime_initialized")
    merged = next(event for event in events if event["event"] == "merged_forward")
    started = next(event for event in events if event["event"] == "backward_started")
    assert merged["hook_called"] is True
    assert merged["all_requests_were_prefill"] is True
    assert merged["inference_request_ids"]
    assert initialized["base_model_replaced_with_shared_vmm"] is True
    assert initialized["base_allocation_id"] == started["base_allocation_id"] == backward["base_allocation_id"]
    assert merged["activation_allocation_id"] == started["activation_allocation_id"]
    assert initialized["pid"] != started["pid"]
    assert backward["parameter_delta_l1"] > 0
    assert published["adapter_ready"] is True
    summary = {
        "success": True,
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "model": args.model,
        },
        "invariants": {
            "shared_base_allocation": True,
            "synthetic_ft_and_inference_prefill_in_same_forward": True,
            "forward_hook_captured_training_row": True,
            "backward_ran_in_separate_gpu_process": True,
            "lora_parameters_updated": True,
            "adapter_published_to_live_vllm": True,
        },
        "runtime_pid": initialized["pid"],
        "backward_pid": started["pid"],
        "base_allocation_id": initialized["base_allocation_id"],
        "forward_loss": merged["forward_loss"],
        "backward_loss": backward["loss"],
        "parameter_delta_l1": backward["parameter_delta_l1"],
        "merged_request_ids": merged["request_ids"],
        "training_flat_slice": merged["training_flat_slice"],
        "merged_outputs": [output.request_id for output in merged_outputs],
        "post_output": post_outputs[0].outputs[0].text,
        "trace": str(args.trace),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
