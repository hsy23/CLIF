# WSL / RTX 5070 Ti validation log

Date: 2026-08-27 (Asia/Shanghai)

This file records the commands and material outputs used by the accompanying research report. The model path was the already-present Hugging Face snapshot at `/home/shaoyuan/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca`; no model weights were downloaded during these tests.

## Environment

```text
WSL distribution: Ubuntu-24.04
GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
GPU memory reported by PyTorch: 11.94 GiB
torch: 2.11.0+cu130
vLLM: 0.21.0
torch CUDA runtime: 13.0
torch.cuda.is_available(): True
```

vLLM also emitted `Using 'pin_memory=False' as WSL is detected. This may slow down the performance.`

## Policy-core tests

Command:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Result:

```text
test_backward_blocks_new_admission_and_failed_backward_requeues ... ok
test_eager_cost_can_reject_even_when_graph_baseline_fits ... ok
test_host_without_mixed_prefill_decode_support_rejects_decode_batch ... ok
test_shortest_first_admission_reserves_activation_capacity ... ok
Ran 4 tests in 0.000s
OK
```

Dry-run output:

```text
{'reason': 'admitted', 'accepted': ['short'], 'mode': 'eager', 'baseline_ms': 4.36, 'mixed_ms': 9.72}
{'while_backward': 'backward_running'}
```

These tests execute only the dependency-free admission model. They do not execute a training forward pass or backward pass inside vLLM.

## vLLM integration probe

The installed `vllm.v1.request.Request` signature requires both `sampling_params` and `pooling_params` positional parameters. Calling the current adapter produced:

```text
TypeError: Request.__init__() missing 1 required positional argument: 'pooling_params'
```

Directly constructing the same request with `pooling_params=None` succeeded and retained the dynamic DeltaServe marker:

```text
{'constructed_with_pooling_none': 'direct-probe', 'dynamic_marker': 'probe'}
```

The unmodified vLLM 0.21.0 runtime reported:

```text
has_register_deltaserve_hooks False
GPUModelRunner.execute_model first line: @torch.inference_mode()
```

Therefore the adapter bug is small, but vanilla vLLM still lacks the scheduler, activation-capture, and retirement hooks required for a real mixed training request.

## CLIF environment integration probe

The WSL virtual environment contains the vLLM-focused requirements only. Importing the current CLIF entry point failed before execution:

```text
peft False
sklearn False
ModuleNotFoundError: No module named 'pandas'
```

Static search also found no import or construction of `CLIFDeltaServeBridge` from `main.py`, `run.py`, or the existing `core` package. Existing replica inference still calls `model.generate`, and existing training still calls `local_train`.

## Qwen3-0.6B inference baseline

Benchmark script: `benchmark_vllm_baseline.py`. Each measured request generated 16 tokens. The first result in each configuration was a warm-up and is not used for comparison.

| vLLM execution mode | Batch | Generated tokens | Elapsed seconds | Aggregate tokens/s |
|---|---:|---:|---:|---:|
| eager | 1 | 16 | 0.3098 | 51.641 |
| eager | 4 | 64 | 0.4669 | 137.078 |
| CUDA graph | 1 | 16 | 0.0678 | 235.824 |
| CUDA graph | 4 | 64 | 0.2021 | 316.622 |

On this one short-output microbenchmark, graph/eager throughput ratios were 4.57x at batch 1 and 2.31x at batch 4. These are local measurements, not general vLLM performance claims. They show why DeltaServe must explicitly model the eager-mode penalty when a training row disables graph replay.

## CUDA IPC and MPS probe

No `nvidia-cuda-mps-control` or `nvidia-cuda-mps-server` executable was present in the WSL environment. A `torch.multiprocessing` spawn test attempted to send a CUDA tensor to a child process. The child failed while rebuilding CUDA storage:

```text
torch.AcceleratorError: CUDA error: invalid resource handle
... torch.UntypedStorage._new_shared_cuda
```

The parent timed out and emitted:

```text
_queue.Empty
Producer process has been terminated before all shared CUDA tensors released.
```

This is a result for this exact Windows/WSL/driver/PyTorch combination. It does not establish that CUDA IPC is unsupported by WSL in general. It does mean the paper-faithful separate-process backward path cannot currently be validated on this machine without resolving the runtime/driver IPC problem and supplying MPS tooling.

## Interpretation boundary

The successful inference benchmark proves that vLLM 0.21.0 can run Qwen3-0.6B on this GPU. The successful unit tests prove that the admission-policy scaffold behaves as encoded. There was no treatment run that performed fine-tuning concurrently with inference, so no TTFT/TPOT interference or fine-tuning throughput result exists yet.
