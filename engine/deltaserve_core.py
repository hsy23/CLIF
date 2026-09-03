"""A dependency-free DeltaServe admission core.

This module deliberately owns only fine-tuning admission.  vLLM remains the
owner of request queues, KV cache, batching, and token sampling; CLIF remains
the owner of global replica selection and adapter promotion.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import RLock
from typing import Iterable, Optional


class ExecutionMode(str, Enum):
    GRAPH = "graph"
    EAGER = "eager"


@dataclass(frozen=True)
class LatencyCoefficients:
    """Coefficients for the paper's lightweight online latency model."""

    attention: float
    token: float
    activation: float
    kv_cache: float
    intercept: float = 0.0


@dataclass(frozen=True)
class HostBatch:
    """Engine-visible work scheduled for the next iteration.

    Times are in seconds and token counts are logical tokens.  ``None`` for
    ``earliest_arrival`` denotes a training-only iteration.
    """

    earliest_arrival: Optional[float]
    prefill_token_lengths: tuple[int, ...]
    active_decode_requests: int
    kv_cache_tokens: int
    execution_mode: ExecutionMode = ExecutionMode.GRAPH

    @property
    def has_prefill(self) -> bool:
        return bool(self.prefill_token_lengths)


@dataclass(frozen=True)
class FineTuneSample:
    sample_id: str
    token_ids: tuple[int, ...]
    adapter_id: int
    payload: object | None = None

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)


@dataclass(frozen=True)
class AdmissionDecision:
    accepted: tuple[FineTuneSample, ...]
    reason: str
    baseline_latency_s: float
    mixed_latency_s: float
    budget_s: float
    execution_mode: ExecutionMode

    @property
    def accepted_token_count(self) -> int:
        return sum(sample.num_tokens for sample in self.accepted)


class FineTunePool:
    """A small, thread-safe pool with shortest-sequence-first selection."""

    def __init__(self, samples: Iterable[FineTuneSample] = ()) -> None:
        self._pending: dict[str, FineTuneSample] = {}
        self._claimed: dict[str, FineTuneSample] = {}
        self._lock = RLock()
        self.add_many(samples)

    def add_many(self, samples: Iterable[FineTuneSample]) -> None:
        with self._lock:
            for sample in samples:
                if sample.num_tokens == 0:
                    raise ValueError("fine-tuning samples must contain at least one token")
                if sample.sample_id in self._pending or sample.sample_id in self._claimed:
                    raise ValueError(f"duplicate fine-tuning sample id: {sample.sample_id}")
                self._pending[sample.sample_id] = sample

    def candidates(self) -> tuple[FineTuneSample, ...]:
        with self._lock:
            return tuple(sorted(self._pending.values(), key=lambda item: (item.num_tokens, item.sample_id)))

    def claim(self, sample_ids: Iterable[str]) -> tuple[FineTuneSample, ...]:
        ids = tuple(sample_ids)
        with self._lock:
            missing = [sample_id for sample_id in ids if sample_id not in self._pending]
            if missing:
                raise KeyError(f"fine-tuning samples are no longer pending: {missing}")
            claimed = tuple(self._pending.pop(sample_id) for sample_id in ids)
            self._claimed.update({sample.sample_id: sample for sample in claimed})
            return claimed

    def complete(self, sample_ids: Iterable[str]) -> None:
        with self._lock:
            for sample_id in sample_ids:
                self._claimed.pop(sample_id, None)

    def requeue(self, sample_ids: Iterable[str]) -> None:
        with self._lock:
            for sample_id in sample_ids:
                sample = self._claimed.pop(sample_id, None)
                if sample is not None:
                    self._pending[sample_id] = sample

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)


class ActivationBuffer:
    """Tracks token-equivalent activation capacity without retaining tensors."""

    def __init__(self, capacity_tokens: int) -> None:
        if capacity_tokens <= 0:
            raise ValueError("activation buffer capacity must be positive")
        self.capacity_tokens = capacity_tokens
        self._reservations: dict[str, int] = {}
        self._lock = RLock()

    @property
    def used_tokens(self) -> int:
        with self._lock:
            return sum(self._reservations.values())

    @property
    def available_tokens(self) -> int:
        return self.capacity_tokens - self.used_tokens

    def reserve(self, samples: Iterable[FineTuneSample]) -> bool:
        samples = tuple(samples)
        token_count = sum(sample.num_tokens for sample in samples)
        with self._lock:
            if any(sample.sample_id in self._reservations for sample in samples):
                raise ValueError("activation is already reserved for one or more samples")
            if token_count > self.capacity_tokens - sum(self._reservations.values()):
                return False
            self._reservations.update({sample.sample_id: sample.num_tokens for sample in samples})
            return True

    def release(self, sample_ids: Iterable[str]) -> None:
        with self._lock:
            for sample_id in sample_ids:
                self._reservations.pop(sample_id, None)


class LatencyModel:
    """Online-model form used by DeltaServe for forward/decode admission."""

    def __init__(
        self,
        forward: dict[ExecutionMode, LatencyCoefficients],
        decode: dict[ExecutionMode, LatencyCoefficients],
    ) -> None:
        missing_forward = set(ExecutionMode) - set(forward)
        missing_decode = set(ExecutionMode) - set(decode)
        if missing_forward or missing_decode:
            raise ValueError("latency coefficients must be supplied for graph and eager modes")
        self.forward = forward
        self.decode = decode

    @staticmethod
    def _predict(
        coefficients: LatencyCoefficients,
        *,
        prefill_lengths: tuple[int, ...],
        active_decode_requests: int,
        kv_cache_tokens: int,
        fine_tune_tokens: int,
    ) -> float:
        attention_work = sum((length + active_decode_requests) ** 2 for length in prefill_lengths)
        token_work = sum(prefill_lengths) + active_decode_requests
        return (
            coefficients.attention * attention_work
            + coefficients.token * token_work
            + coefficients.activation * fine_tune_tokens
            + coefficients.kv_cache * kv_cache_tokens
            + coefficients.intercept
        )

    def predict_forward(
        self,
        batch: HostBatch,
        fine_tune_tokens: int,
        mode: ExecutionMode,
    ) -> float:
        return self._predict(
            self.forward[mode],
            prefill_lengths=batch.prefill_token_lengths,
            active_decode_requests=batch.active_decode_requests,
            kv_cache_tokens=batch.kv_cache_tokens,
            fine_tune_tokens=fine_tune_tokens,
        )

    def predict_decode(self, batch: HostBatch, mode: ExecutionMode) -> float:
        return self._predict(
            self.decode[mode],
            prefill_lengths=(),
            active_decode_requests=batch.active_decode_requests,
            kv_cache_tokens=batch.kv_cache_tokens,
            fine_tune_tokens=0,
        )


class DeltaServeAdmissionController:
    """Admits at most one in-flight fine-tuning forward/backward batch.

    This captures the paper's conservative policy: a live backward pass blocks
    new fine-tuning forward work, and any mixed forward is costed in eager mode
    because activation hooks invalidate vLLM CUDA graph capture.
    """

    def __init__(self, pool: FineTunePool, activation_buffer: ActivationBuffer, latency_model: LatencyModel) -> None:
        self.pool = pool
        self.activation_buffer = activation_buffer
        self.latency_model = latency_model
        self._backward_sample_ids: tuple[str, ...] = ()
        self._lock = RLock()

    @property
    def backward_running(self) -> bool:
        with self._lock:
            return bool(self._backward_sample_ids)

    def admit(
        self,
        batch: HostBatch,
        *,
        now: float,
        ttft_slo_s: float,
        tpot_slo_s: float,
        host_supports_mixed_prefill_decode: bool,
        allow_training_only: bool = True,
    ) -> AdmissionDecision:
        with self._lock:
            if self.backward_running:
                return self._reject("backward_running", batch)
            if not batch.has_prefill and batch.earliest_arrival is not None:
                return self._reject("no_prefill_in_host_batch", batch)
            if batch.earliest_arrival is None and not allow_training_only:
                return self._reject("training_only_disabled", batch)
            if batch.active_decode_requests and not host_supports_mixed_prefill_decode:
                return self._reject("host_cannot_mix_prefill_decode", batch)

            budget_s = self._latency_budget(
                batch,
                now=now,
                ttft_slo_s=ttft_slo_s,
                tpot_slo_s=tpot_slo_s,
                host_supports_mixed_prefill_decode=host_supports_mixed_prefill_decode,
            )
            baseline = self.latency_model.predict_forward(batch, 0, batch.execution_mode)
            if baseline > budget_s:
                return AdmissionDecision((), "host_batch_already_exceeds_slo", baseline, baseline, budget_s, batch.execution_mode)

            accepted: list[FineTuneSample] = []
            accepted_tokens = 0
            for sample in self.pool.candidates():
                candidate_tokens = accepted_tokens + sample.num_tokens
                if candidate_tokens > self.activation_buffer.available_tokens:
                    break
                candidate_latency = self.latency_model.predict_forward(batch, candidate_tokens, ExecutionMode.EAGER)
                if candidate_latency > budget_s:
                    break
                accepted.append(sample)
                accepted_tokens = candidate_tokens

            if not accepted:
                reason = "activation_buffer_full" if self.activation_buffer.available_tokens == 0 else "no_sample_fits_slo"
                return AdmissionDecision((), reason, baseline, baseline, budget_s, batch.execution_mode)

            claimed = self.pool.claim(sample.sample_id for sample in accepted)
            if not self.activation_buffer.reserve(claimed):
                self.pool.requeue(sample.sample_id for sample in claimed)
                return self._reject("activation_buffer_raced", batch, budget_s=budget_s, baseline=baseline)

            mixed_latency = self.latency_model.predict_forward(batch, accepted_tokens, ExecutionMode.EAGER)
            return AdmissionDecision(tuple(claimed), "admitted", baseline, mixed_latency, budget_s, ExecutionMode.EAGER)

    def begin_backward(self, decision: AdmissionDecision) -> None:
        sample_ids = tuple(sample.sample_id for sample in decision.accepted)
        if not sample_ids:
            raise ValueError("cannot start backward without an admitted fine-tuning batch")
        with self._lock:
            if self._backward_sample_ids:
                raise RuntimeError("a fine-tuning backward pass is already running")
            self._backward_sample_ids = sample_ids

    def finish_backward(self, *, succeeded: bool) -> None:
        with self._lock:
            sample_ids = self._backward_sample_ids
            if not sample_ids:
                raise RuntimeError("no fine-tuning backward pass is running")
            if succeeded:
                self.pool.complete(sample_ids)
            else:
                self.pool.requeue(sample_ids)
            self.activation_buffer.release(sample_ids)
            self._backward_sample_ids = ()

    def _reject(
        self,
        reason: str,
        batch: HostBatch,
        *,
        budget_s: float = 0.0,
        baseline: float | None = None,
    ) -> AdmissionDecision:
        baseline = self.latency_model.predict_forward(batch, 0, batch.execution_mode) if baseline is None else baseline
        return AdmissionDecision((), reason, baseline, baseline, budget_s, batch.execution_mode)

    def _latency_budget(
        self,
        batch: HostBatch,
        *,
        now: float,
        ttft_slo_s: float,
        tpot_slo_s: float,
        host_supports_mixed_prefill_decode: bool,
    ) -> float:
        ttft_budget = float("inf")
        if batch.earliest_arrival is not None:
            ttft_budget = batch.earliest_arrival + ttft_slo_s - now

        tpot_budget = float("inf")
        if batch.active_decode_requests:
            deferred_decode = 0.0
            if not host_supports_mixed_prefill_decode:
                deferred_decode = self.latency_model.predict_decode(batch, batch.execution_mode)
            tpot_budget = tpot_slo_s - deferred_decode
        return min(ttft_budget, tpot_budget)
