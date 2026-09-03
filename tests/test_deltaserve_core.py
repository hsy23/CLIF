import unittest

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


def make_controller(capacity=12):
    graph = LatencyCoefficients(attention=0.0, token=0.0, activation=0.001, kv_cache=0.0, intercept=0.001)
    eager = LatencyCoefficients(attention=0.0, token=0.0, activation=0.002, kv_cache=0.0, intercept=0.002)
    model = LatencyModel(
        forward={ExecutionMode.GRAPH: graph, ExecutionMode.EAGER: eager},
        decode={ExecutionMode.GRAPH: graph, ExecutionMode.EAGER: eager},
    )
    pool = FineTunePool(
        [
            FineTuneSample("eight", tuple(range(8)), adapter_id=1),
            FineTuneSample("four", tuple(range(4)), adapter_id=1),
            FineTuneSample("sixteen", tuple(range(16)), adapter_id=1),
        ]
    )
    return DeltaServeAdmissionController(pool, ActivationBuffer(capacity), model)


def make_host_batch():
    return HostBatch(
        earliest_arrival=100.0,
        prefill_token_lengths=(10,),
        active_decode_requests=0,
        kv_cache_tokens=0,
        execution_mode=ExecutionMode.GRAPH,
    )


class DeltaServeAdmissionTests(unittest.TestCase):
    def test_shortest_first_admission_reserves_activation_capacity(self):
        controller = make_controller()
        decision = controller.admit(
            make_host_batch(),
            now=100.0,
            ttft_slo_s=0.03,
            tpot_slo_s=0.03,
            host_supports_mixed_prefill_decode=True,
        )

        self.assertEqual("admitted", decision.reason)
        self.assertEqual(["four", "eight"], [sample.sample_id for sample in decision.accepted])
        self.assertEqual(ExecutionMode.EAGER, decision.execution_mode)
        self.assertEqual(12, controller.activation_buffer.used_tokens)

    def test_eager_cost_can_reject_even_when_graph_baseline_fits(self):
        controller = make_controller()
        decision = controller.admit(
            make_host_batch(),
            now=100.0,
            ttft_slo_s=0.005,
            tpot_slo_s=0.03,
            host_supports_mixed_prefill_decode=True,
        )

        self.assertEqual("no_sample_fits_slo", decision.reason)
        self.assertEqual(ExecutionMode.GRAPH, decision.execution_mode)

    def test_backward_blocks_new_admission_and_failed_backward_requeues(self):
        controller = make_controller()
        admitted = controller.admit(
            make_host_batch(),
            now=100.0,
            ttft_slo_s=0.03,
            tpot_slo_s=0.03,
            host_supports_mixed_prefill_decode=True,
        )
        controller.begin_backward(admitted)
        blocked = controller.admit(
            make_host_batch(),
            now=100.001,
            ttft_slo_s=0.03,
            tpot_slo_s=0.03,
            host_supports_mixed_prefill_decode=True,
        )
        self.assertEqual("backward_running", blocked.reason)

        controller.finish_backward(succeeded=False)
        self.assertFalse(controller.backward_running)
        self.assertEqual(0, controller.activation_buffer.used_tokens)
        self.assertEqual(3, controller.pool.pending_count)

    def test_host_without_mixed_prefill_decode_support_rejects_decode_batch(self):
        controller = make_controller()
        decode_batch = HostBatch(
            earliest_arrival=100.0,
            prefill_token_lengths=(10,),
            active_decode_requests=1,
            kv_cache_tokens=0,
        )
        decision = controller.admit(
            decode_batch,
            now=100.0,
            ttft_slo_s=0.03,
            tpot_slo_s=0.03,
            host_supports_mixed_prefill_decode=False,
        )
        self.assertEqual("host_cannot_mix_prefill_decode", decision.reason)


if __name__ == "__main__":
    unittest.main()
