"""Apply or restore the version-pinned vLLM 0.21.0 DeltaServe hook patch."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


GPU_MARKER = "CLIF_DELTASERVE_RUNTIME_PATCH"
SCHEDULER_MARKER = "CLIF_DELTASERVE_BATCH_BARRIER_PATCH"


def patch_source(source: str) -> str:
    if GPU_MARKER in source:
        return source

    first_anchor = """        if self.routed_experts_initialized:\n            capturer = get_global_experts_capturer()\n"""
    first_replacement = """        # CLIF_DELTASERVE_RUNTIME_PATCH: initialize the shared-model runtime.\n        deltaserve_runtime = None\n        if __import__(\"os\").environ.get(\"CLIF_DELTASERVE_ENABLE\") == \"1\":\n            from engine.deltaserve_vllm_runtime import get_runtime\n            deltaserve_runtime = get_runtime(self)\n            deltaserve_runtime.poll_results()\n\n        if self.routed_experts_initialized:\n            capturer = get_global_experts_capturer()\n"""

    second_anchor = """            ) = self._preprocess(\n                scheduler_output, num_tokens_padded, intermediate_tensors\n            )\n\n        # Set cudagraph mode to none if calc_kv_scales is true.\n"""
    second_replacement = """            ) = self._preprocess(\n                scheduler_output, num_tokens_padded, intermediate_tensors\n            )\n\n        deltaserve_capture = False\n        if deltaserve_runtime is not None:\n            deltaserve_capture = deltaserve_runtime.prepare_batch(\n                req_ids,\n                num_scheduled_tokens_np,\n                self.input_batch.num_computed_tokens_cpu[:num_reqs],\n                input_ids,\n            )\n            if deltaserve_capture:\n                cudagraph_mode = CUDAGraphMode.NONE\n\n        # Set cudagraph mode to none if calc_kv_scales is true.\n"""

    third_anchor = """            model_output = self._model_forward(\n                input_ids=input_ids,\n                positions=positions,\n                intermediate_tensors=intermediate_tensors,\n                inputs_embeds=inputs_embeds,\n                **model_kwargs,\n            )\n\n        with record_function_or_nullcontext(\"gpu_model_runner: postprocess\"):\n"""
    third_replacement = """            model_output = self._model_forward(\n                input_ids=input_ids,\n                positions=positions,\n                intermediate_tensors=intermediate_tensors,\n                inputs_embeds=inputs_embeds,\n                **model_kwargs,\n            )\n            if deltaserve_capture:\n                deltaserve_runtime.after_model_forward()\n\n        with record_function_or_nullcontext(\"gpu_model_runner: postprocess\"):\n"""

    for anchor, replacement in (
        (first_anchor, first_replacement),
        (second_anchor, second_replacement),
        (third_anchor, third_replacement),
    ):
        if source.count(anchor) != 1:
            raise RuntimeError(f"vLLM source anchor count is {source.count(anchor)}, expected 1")
        source = source.replace(anchor, replacement)
    return source


def patch_scheduler_source(source: str) -> str:
    if SCHEDULER_MARKER in source:
        return source
    anchor = """        token_budget = self.max_num_scheduled_tokens\n        if self._pause_state == PauseState.PAUSED_ALL:\n"""
    replacement = """        token_budget = self.max_num_scheduled_tokens\n        # CLIF_DELTASERVE_BATCH_BARRIER_PATCH: keep a lone synthetic FT row\n        # waiting for one ingress cycle so the following inference prefill can\n        # join the same scheduler/model-forward batch.\n        if (\n            __import__(\"os\").environ.get(\"CLIF_DELTASERVE_ENABLE\") == \"1\"\n            and not self.running\n            and len(self.waiting) == 1\n            and self.waiting.peek_request().request_id.startswith(\"deltaserve-ft-\")\n        ):\n            token_budget = 0\n        if self._pause_state == PauseState.PAUSED_ALL:\n"""
    if source.count(anchor) != 1:
        raise RuntimeError(
            f"vLLM scheduler anchor count is {source.count(anchor)}, expected 1"
        )
    return source.replace(anchor, replacement)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()
    import vllm

    if vllm.__version__ != "0.21.0":
        raise RuntimeError(f"patch targets vLLM 0.21.0, found {vllm.__version__}")
    vllm_root = Path(vllm.__file__).resolve().parent
    targets = (
        (vllm_root / "v1" / "worker" / "gpu_model_runner.py", patch_source),
        (vllm_root / "v1" / "core" / "sched" / "scheduler.py", patch_scheduler_source),
    )
    if args.restore:
        restored = []
        for target, _patcher in targets:
            backup = target.with_suffix(".py.clif-deltaserve-original")
            if not backup.exists():
                raise FileNotFoundError(f"no backup exists at {backup}")
            shutil.copy2(backup, target)
            restored.append(str(target))
        print({"status": "restored", "targets": restored})
        return

    results = []
    for target, patcher in targets:
        backup = target.with_suffix(".py.clif-deltaserve-original")
        if not backup.exists():
            shutil.copy2(target, backup)
        source = target.read_text(encoding="utf-8")
        patched = patcher(source)
        target.write_text(patched, encoding="utf-8")
        results.append({"target": str(target), "changed": patched != source})
    print({"status": "patched", "results": results})


if __name__ == "__main__":
    main()
