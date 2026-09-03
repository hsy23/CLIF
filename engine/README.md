# DeltaServe-style vLLM single-replica prototype

This is an engine-backed execution prototype for CLIF, not a replacement for
CLIF's existing coordinator or federated launcher.

## What is implemented

- A single-replica, shortest-first fine-tuning admission controller following
  DeltaServe's TTFT/TPOT-slack idea.
- Separate latency coefficients for CUDA-graph and eager execution.  Any batch
  with admitted fine-tuning work is conservatively costed as eager.
- Token-equivalent activation-buffer reservations and one-backward-at-a-time
  exclusion; a failed backward returns its samples to the pool.
- A CLIF bridge that exports backend capacity as a local-scheduler signal.
- A vLLM 0.21.0 request adapter that turns a fine-tuning example into an
  internal one-step prefill request with a reserved LoRA request.
- A version-pinned vLLM 0.21.0 hook path that co-batches one synthetic
  fine-tuning row with inference prefill, captures that row at the final
  RMSNorm, and runs a real LM-head LoRA backward step in a separate GPU
  subprocess over CUDA VMM shared allocations.
- A principle-level concurrent backend in which vLLM serves inference while a
  separate process performs real LoRA forward/backward/optimizer steps on the
  same GPU.  It records parameter deltas and wall-clock overlap between
  inference requests and training steps, exports a PEFT-compatible adapter,
  and verifies that the live vLLM engine can load it for generation.
- An experimental HTTP wrapper around one patched vLLM `AsyncLLM` replica.  It
  exposes `/generate`, streaming `/generate/stream`, `/train-step`, and
  `/metrics` for client-level co-serving measurements without loading a second
  base model.
- A Transformers Qwen3 `q_proj`/`v_proj` LoRA one-step reference used to
  validate the G6 attention-target math.
- An experimental live Qwen3 attention runtime selected with
  `--target-modules q_proj,v_proj`: it maps vLLM base parameters, including
  packed `qkv_proj` and `gate_up_proj`, to CUDA VMM allocations; host-side
  hooks apply the shared q/v delta, while a meta-initialized Transformers
  worker performs real q/v autograd updates over the same base allocations.

## Hooked shared-allocation path (current)

Apply the reversible vLLM patch and run the Qwen3-0.6B integration test:

```bash
.venv/bin/python scripts/apply_vllm_deltaserve_patch.py
.venv/bin/python scripts/run_deltaserve_vllm_hooked.py \
  --model /path/to/Qwen3-0.6B/snapshot \
  --trace hooked-trace.jsonl \
  --output hooked-result.json
```

Restore the original installed vLLM sources with:

```bash
.venv/bin/python scripts/apply_vllm_deltaserve_patch.py --restore
```

The validated WSL run is recorded in
`research/DeltaServe_vLLM_single_replica_Research_20260827/hooked-result.json`.
It proves one live vLLM base model, a synthetic FT row and inference prefill in
the same model-forward batch, hook-based activation capture, and a real LoRA
backward/update in a different GPU PID.  The current adapter targets only the
LM head.  It is a functional DeltaServe-principle prototype, not yet the full
all-transformer-layer training path or an SLO-isolated performance result.

The patch also disables asynchronous scheduling for this one-shot test and
holds a lone synthetic FT request for one scheduler ingress cycle, ensuring
the following inference prefill joins the same batch.  Co-serving batches run
eager because activation hooks are not CUDA-graph captured.

## HTTP co-serving path (experimental)

On a native Linux CUDA host after applying the patch, the HTTP benchmark can
launch one `AsyncLLM` replica and run inference clients alongside training
control requests:

```bash
MODEL=/scratch/users/<id>/models/Qwen3-0.6B \
  sbatch scripts/run_deltaserve_http_slurm.sh
```

The benchmark records client latency, streamed TTFT/TPOT, output-token
throughput, success rate, training step time, GPU memory, utilization, and
DeltaServe trace event counts.  It is an experimental validation endpoint,
not the standard OpenAI server and not yet a paper-level HTTP result.

The attention-target reference is run independently of vLLM:

```bash
MODEL=/scratch/users/<id>/models/Qwen3-0.6B \
  sbatch scripts/run_qwen3_attention_reference_slurm.sh
```

This reference proves only the `q_proj`/`v_proj` LoRA update math; the live
shared-VMM attention path can be smoke-tested separately:

```bash
MODEL=/scratch/users/<id>/models/Qwen3-0.6B \
  sbatch scripts/run_deltaserve_attention_slurm.sh
```

The live path is correctness-first: because vLLM's model runner is wrapped in
`inference_mode`, the worker recomputes the training row to retain an autograd
graph.  A successful smoke proves packed q/v mapping, shared-base execution,
real gradients, live adapter visibility, and publish metadata; it is not yet
the final low-overhead DeltaServe performance path.

## Earlier duplicated-model baseline

Run the standalone co-execution experiment with an existing local model path:

```bash
.venv/bin/python scripts/run_vllm_lora_concurrent.py \
  --model /path/to/Qwen3-0.6B/snapshot \
  --output concurrent-result.json
```

This first working path duplicates the frozen base-model weights between the
vLLM and training processes.  It proves real train/infer progress on one GPU,
but it is not yet DeltaServe's shared-activation/MPS design.

The 2026-08-27 WSL/RTX 5070 Ti run is saved at
`research/DeltaServe_vLLM_single_replica_Research_20260827/concurrent-run.json`.
It completed 8 real optimizer steps, observed inference overlap for every
training step, exported 112 LoRA tensors, and loaded the trained adapter back
into the live vLLM instance.  Current limitations are explicit:

- the frozen base weights are duplicated between processes;
- CUDA contexts are time-sliced because MPS/IPC is not available in this WSL;
- inference uses eager mode for the first correctness prototype;
- the trained adapter is published after the training job, not continuously
  synchronized into in-flight inference batches.

## Deliberate boundary

Unmodified vLLM cannot run the complete paper design: `GPUModelRunner` is
wrapped in `torch.inference_mode()`, so it cannot capture the tensors needed by
the backward subprocess.  The prototype therefore requires a small vLLM 0.21.0
fork before it can execute training rows inside a mixed vLLM model batch.  The
standalone concurrent backend above is an intermediate proof: it runs real
training beside vLLM, but does not inject training rows into vLLM itself.

The exact fork points are exported by `engine.vllm_adapter.VLLMForkPoints`:

1. In `Scheduler.schedule`, accept CLIF's internal synthetic prefill requests
   alongside normal waiting requests.
2. In `SchedulerOutput` and `EngineCore.step`, carry the set of internal
   request IDs into the worker and suppress their client-visible output.
3. In `GPUModelRunner.execute_model`, force eager for a batch containing those
   IDs; use forward hooks to copy the chosen rows' activations to a CUDA-IPC
   buffer and notify the separate backward process.
4. In the backward process, honor DeltaServe's layer-boundary yielding/abort
   policy and publish a new LoRA version only through CLIF's existing launcher.

This separation is intentional: it preserves vLLM ownership of queuing, KV
cache, normal batching, sampling, and optimized inference while preserving
CLIF ownership of local batch control, global replica/round selection, and
adapter promotion.

## Verify the portable core

From the repository root:

```powershell
python -m unittest discover -s tests -v
python scripts/run_deltaserve_vllm_dryrun.py
```

The tests use only the standard library.  The dry run demonstrates that a
mixed batch is costed in eager mode and that an in-flight backward blocks a new
fine-tuning admission.

To verify that the pinned vLLM runtime can use a local GPU, run:

```bash
.venv/bin/python scripts/smoke_vllm_qwen.py
```

This is a baseline inference smoke test, not an end-to-end DeltaServe result.

## CUDA host setup

Run the actual vLLM integration on a Linux CUDA host, pinned to the source
version inspected for this prototype:

```bash
pip install -r requirements-vllm.txt
```

Apply the version-pinned patch used by the current prototype:

```bash
.venv/bin/python scripts/apply_vllm_deltaserve_patch.py
```

It patches `GPUModelRunner.execute_model` and `Scheduler.schedule` in the
installed vLLM 0.21.0 source.  The current implementation does not claim that
vanilla `vllm serve` exposes a complete DeltaServe control API; use the
experimental HTTP wrapper above when a server-layer training endpoint is
needed.

## Native Linux / Slurm

The KCL path uses a native Linux virtual environment and runs CUDA work on a
Slurm GPU node. Do not run the vLLM executable on a login node. After placing
the repository at `/users/<id>/CLIF_2027` and installing `vllm==0.21.0`, probe
the allocation with:

```bash
srun --partition=gpu --gres=gpu:1 --time=00:03:00 \
  /users/<id>/CLIF_2027/.venv/bin/python \
  /users/<id>/CLIF_2027/scripts/probe_native_linux.py
```

The end-to-end launcher is `scripts/run_deltaserve_vllm_slurm.sh`. It requires
`MODEL` to point to an existing local Hugging Face snapshot and defaults
`HF_HUB_OFFLINE=1`, so a missing model fails fast instead of downloading
weights inside a job:

```bash
MODEL=/users/<id>/models/Qwen3-0.6B \
  sbatch scripts/run_deltaserve_vllm_slurm.sh
```

The launcher exports `PYTHONPATH`, selects the target venv, enables the
DeltaServe patch, and writes the JSON result and JSONL trace under `output/`.
