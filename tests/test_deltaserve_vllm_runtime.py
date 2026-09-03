from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover - runtime tests need the GPU stack
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class DeltaServeVLLMRuntimeTests(unittest.TestCase):
    def test_replayed_inflight_training_batch_is_not_submitted_twice(self):
        from engine.deltaserve_vllm_runtime import DeltaServeVLLMRuntime

        runtime = DeltaServeVLLMRuntime.__new__(DeltaServeVLLMRuntime)
        runtime._inflight_job_id = 3
        runtime._inflight_training_req_ids = (
            "deltaserve-ft-0",
            "deltaserve-ft-1",
        )

        self.assertTrue(
            runtime._is_inflight_replay(
                ["deltaserve-ft-0", "deltaserve-ft-1", "inference-0"],
                [0, 1],
            )
        )
        self.assertTrue(
            runtime._is_inflight_replay(
                ["deltaserve-ft-0", "inference-0"],
                [0],
            )
        )
        self.assertFalse(
            runtime._is_inflight_replay(
                ["deltaserve-ft-0", "deltaserve-ft-new"],
                [0, 1],
            )
        )


if __name__ == "__main__":
    unittest.main()
