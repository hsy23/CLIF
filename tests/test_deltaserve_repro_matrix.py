import unittest

from scripts.run_deltaserve_repro_matrix import metric_summary


class ReproMatrixTests(unittest.TestCase):
    def test_metric_summary_reports_mean_variability(self):
        summary = metric_summary(
            [{"value": 10.0}, {"value": 12.0}, {"value": 14.0}],
            ("value",),
        )
        self.assertEqual([10.0, 12.0, 14.0], summary["values"])
        self.assertEqual(12.0, summary["mean"])
        self.assertAlmostEqual(2.0, summary["stdev"])
        self.assertAlmostEqual(2.0 / 12.0, summary["coefficient_of_variation"])


if __name__ == "__main__":
    unittest.main()
