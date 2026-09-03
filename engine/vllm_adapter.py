"""The narrow vLLM boundary required by the DeltaServe prototype.

Vanilla vLLM is intentionally not treated as a training runtime.  This module
can construct synthetic one-step prefill requests, but dispatching them with
activation capture requires the version-pinned fork points documented below.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import time
from typing import Any

from .deltaserve_core import FineTuneSample


VLLM_TARGET_VERSION = "0.21.0"


class VLLMIntegrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class VLLMTrainingRequest:
    """Opaque request plus metadata retained by the CLIF fine-tuning manager."""

    request: Any
    sample: FineTuneSample
    request_id: str


@dataclass(frozen=True)
class VLLMForkPoints:
    """Exact vLLM 0.21.0 locations a full prototype must patch."""

    scheduler: str = "vllm/v1/core/sched/scheduler.py::Scheduler.schedule"
    engine_core: str = "vllm/v1/engine/core.py::EngineCore.step"
    scheduler_output: str = "vllm/v1/core/sched/output.py::SchedulerOutput"
    model_runner: str = "vllm/v1/worker/gpu/model_runner.py::GPUModelRunner.execute_model"
    worker: str = "vllm/v1/worker/gpu_worker.py::GPUWorker.execute_model"


def validate_vllm_version() -> str:
    """Fail early instead of silently applying 0.21.0 assumptions to another vLLM."""

    try:
        from vllm.version import __version__
    except ImportError as exc:
        raise VLLMIntegrationError(
            "vLLM is not installed. Install requirements-vllm.txt on the CUDA host."
        ) from exc
    if __version__ != VLLM_TARGET_VERSION:
        raise VLLMIntegrationError(
            f"DeltaServe prototype targets vLLM {VLLM_TARGET_VERSION}, found {__version__}."
        )
    return __version__


def make_synthetic_prefill_request(sample: FineTuneSample, *, lora_request: Any) -> VLLMTrainingRequest:
    """Build the paper's one-step prefill-only request for a patched vLLM host.

    The fork must mark this request as internal, skip user-visible streaming,
    force eager execution for a mixed batch, and pass its activations to the
    backward subprocess.  Calling this against unmodified ``vllm serve`` is
    unsupported because its model runner executes inside ``inference_mode``.
    """

    validate_vllm_version()
    try:
        from vllm.sampling_params import SamplingParams
        from vllm.v1.request import Request
    except ImportError as exc:  # pragma: no cover - version validation above is clearer in practice
        raise VLLMIntegrationError("vLLM 0.21.0 request APIs are unavailable") from exc

    request_id = f"deltaserve-ft-{sample.sample_id}"
    request = Request(
        request_id=request_id,
        prompt_token_ids=list(sample.token_ids),
        sampling_params=SamplingParams(temperature=0.0, max_tokens=1),
        pooling_params=None,
        arrival_time=time(),
        lora_request=lora_request,
        priority=-1,
    )
    # Request is a regular Python object in vLLM 0.21.0.  The patched scheduler
    # turns this marker into SchedulerOutput.deltaserve_training_request_ids.
    request.deltaserve_training_sample_id = sample.sample_id
    request.deltaserve_adapter_id = sample.adapter_id
    return VLLMTrainingRequest(request=request, sample=sample, request_id=request_id)


def require_patched_engine(engine_core: Any) -> None:
    """Check the single capability that vanilla vLLM deliberately lacks."""

    if not hasattr(engine_core, "register_deltaserve_hooks"):
        points = VLLMForkPoints()
        raise VLLMIntegrationError(
            "A patched vLLM engine is required. Add scheduler admission at "
            f"{points.scheduler}, propagate internal request ids through {points.scheduler_output}, "
            f"and capture activations from {points.model_runner}."
        )
