"""Small bridge from a vLLM iteration to CLIF's existing control planes."""

from __future__ import annotations

from dataclasses import asdict

from .deltaserve_core import AdmissionDecision, DeltaServeAdmissionController, HostBatch


class CLIFDeltaServeBridge:
    """Expose backend headroom without taking over CLIF scheduling or FL.

    ``Coordinator`` can continue selecting local batch sizes.  ``FLLauncher``
    can continue deciding when to launch/publish an adapter.  This bridge only
    reports whether the engine admitted a fine-tuning microbatch for a given
    vLLM scheduler iteration.
    """

    def __init__(self, controller: DeltaServeAdmissionController) -> None:
        self.controller = controller
        self.last_decision: AdmissionDecision | None = None

    def on_engine_iteration(
        self,
        batch: HostBatch,
        *,
        now: float,
        ttft_slo_s: float,
        tpot_slo_s: float,
        host_supports_mixed_prefill_decode: bool,
    ) -> AdmissionDecision:
        decision = self.controller.admit(
            batch,
            now=now,
            ttft_slo_s=ttft_slo_s,
            tpot_slo_s=tpot_slo_s,
            host_supports_mixed_prefill_decode=host_supports_mixed_prefill_decode,
        )
        self.last_decision = decision
        return decision

    def local_scheduler_signal(self) -> dict[str, object]:
        """A serializable signal for CLIF's existing local-batch controller."""

        if self.last_decision is None:
            return {"engine_ft_ready": False, "reason": "no_engine_iteration"}
        return {
            "engine_ft_ready": bool(self.last_decision.accepted),
            "engine_ft_tokens": self.last_decision.accepted_token_count,
            "engine_ft_reason": self.last_decision.reason,
            "engine_backward_running": self.controller.backward_running,
        }

    def decision_metrics(self) -> dict[str, object]:
        if self.last_decision is None:
            return {}
        metrics = asdict(self.last_decision)
        metrics["accepted"] = [sample.sample_id for sample in self.last_decision.accepted]
        metrics["execution_mode"] = self.last_decision.execution_mode.value
        return metrics
