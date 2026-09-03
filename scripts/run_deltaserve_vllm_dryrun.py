"""Demonstrate a DeltaServe-style admission decision without a GPU or vLLM."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.deltaserve_core import (
    ActivationBuffer,
    DeltaServeAdmissionController,
    ExecutionMode,
    FineTunePool,
    FineTuneSample,
    HostBatch,
    LatencyCoefficients,
    LatencyModel,
)


def main() -> None:
    graph = LatencyCoefficients(attention=0.00001, token=0.0001, activation=0.00015, kv_cache=0.0, intercept=0.001)
    eager = LatencyCoefficients(attention=0.00002, token=0.0002, activation=0.00025, kv_cache=0.0, intercept=0.002)
    model = LatencyModel(
        forward={ExecutionMode.GRAPH: graph, ExecutionMode.EAGER: eager},
        decode={ExecutionMode.GRAPH: graph, ExecutionMode.EAGER: eager},
    )
    controller = DeltaServeAdmissionController(
        FineTunePool(
            [
                FineTuneSample("short", (1, 2, 3, 4), adapter_id=1),
                FineTuneSample("long", tuple(range(24)), adapter_id=1),
            ]
        ),
        ActivationBuffer(capacity_tokens=16),
        model,
    )
    host_batch = HostBatch(
        earliest_arrival=10.0,
        prefill_token_lengths=(12,),
        active_decode_requests=2,
        kv_cache_tokens=0,
        execution_mode=ExecutionMode.GRAPH,
    )
    decision = controller.admit(
        host_batch,
        now=10.0,
        ttft_slo_s=0.02,
        tpot_slo_s=0.02,
        host_supports_mixed_prefill_decode=True,
    )
    print(
        {
            "reason": decision.reason,
            "accepted": [sample.sample_id for sample in decision.accepted],
            "mode": decision.execution_mode.value,
            "baseline_ms": round(decision.baseline_latency_s * 1000, 3),
            "mixed_ms": round(decision.mixed_latency_s * 1000, 3),
        }
    )
    controller.begin_backward(decision)
    blocked = controller.admit(
        host_batch,
        now=10.001,
        ttft_slo_s=0.02,
        tpot_slo_s=0.02,
        host_supports_mixed_prefill_decode=True,
    )
    print({"while_backward": blocked.reason})
    controller.finish_backward(succeeded=True)


if __name__ == "__main__":
    main()
