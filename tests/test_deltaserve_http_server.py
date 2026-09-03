from __future__ import annotations

import unittest
from unittest import mock
from pathlib import Path

from scripts.deltaserve_vllm_http_server import _trace_event_count


class DeltaServeHTTPServerTest(unittest.TestCase):
    def test_trace_event_count_ignores_partial_lines(self):
        trace_path = Path("trace.jsonl")
        trace = (
            '{"event":"adapter_published"}\n'
            '{"event":"backward_finished"}\n'
            '{"event":"adapter_published"}\n'
            '{"event":"adapter_published"'
        )
        with mock.patch.object(Path, "read_text", return_value=trace):
            self.assertEqual(_trace_event_count(trace_path, "adapter_published"), 2)


if __name__ == "__main__":
    unittest.main()
