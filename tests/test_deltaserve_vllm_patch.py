import unittest

from scripts.apply_vllm_deltaserve_patch import (
    GPU_MARKER,
    SCHEDULER_MARKER,
    patch_scheduler_source,
    patch_source,
)


class DeltaServeVLLMPatchTest(unittest.TestCase):
    def test_gpu_runner_patch_is_idempotent(self):
        source = """        if self.routed_experts_initialized:
            capturer = get_global_experts_capturer()
            ) = self._preprocess(
                scheduler_output, num_tokens_padded, intermediate_tensors
            )

        # Set cudagraph mode to none if calc_kv_scales is true.
            model_output = self._model_forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
            )

        with record_function_or_nullcontext("gpu_model_runner: postprocess"):
"""
        patched = patch_source(source)
        self.assertEqual(patched.count(GPU_MARKER), 1)
        self.assertIn("deltaserve_runtime.prepare_batch", patched)
        self.assertIn("deltaserve_runtime.after_model_forward", patched)
        self.assertEqual(patch_source(patched), patched)

    def test_scheduler_barrier_patch_is_idempotent(self):
        source = """        token_budget = self.max_num_scheduled_tokens
        if self._pause_state == PauseState.PAUSED_ALL:
"""
        patched = patch_scheduler_source(source)
        self.assertEqual(patched.count(SCHEDULER_MARKER), 1)
        self.assertIn("len(self.waiting) == 1", patched)
        self.assertIn('startswith("deltaserve-ft-")', patched)
        self.assertEqual(patch_scheduler_source(patched), patched)


if __name__ == "__main__":
    unittest.main()
