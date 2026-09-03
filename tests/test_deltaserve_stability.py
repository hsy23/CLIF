import unittest

from scripts.run_deltaserve_stability import relative_change, summarize_inference


class DeltaServeStabilityTests(unittest.TestCase):
    def test_summary_exposes_tail_latency_and_streaming_metrics(self):
        records = [
            {
                "latency_s": 1.0,
                "ttft_s": 0.2,
                "tpot_s": 0.1,
                "output_tokens": 10,
            },
            {
                "latency_s": 1.2,
                "ttft_s": 0.3,
                "tpot_s": 0.2,
                "output_tokens": 10,
            },
        ]
        summary = summarize_inference(records, elapsed_s=2.0)
        self.assertEqual(summary["output_tokens"], 20)
        self.assertEqual(summary["output_tokens_per_s"], 10.0)
        self.assertEqual(summary["p99_latency_s"], 1.2)
        self.assertEqual(summary["p95_ttft_s"], 0.3)
        self.assertAlmostEqual(relative_change(1.1, 1.0), 0.1)


if __name__ == "__main__":
    unittest.main()
