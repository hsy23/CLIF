# CLIF

CLIF is a research prototype for continuous PEFT fine-tuning under online LLM inference load. 

## Core Mechanisms

CLIF organizes each replica as a stateful worker with three main serving/training states:

- `SERVING`: the replica is dedicated to inference.
- `IDLE`: the replica is lightly loaded and remains available for inference.
- `COMBINED`: the replica serves inference while participating in local PEFT fine-tuning.

The dispatcher continuously observes request pressure and replica quality, then adjusts inference batch bounds. The fine-tuning launcher admits replicas into federated PEFT rounds only when recent serving pressure indicates sufficient slack. Dual-adapter replicas isolate the actively served adapter from the shadow adapter being trained, then promote trained adapters at controlled boundaries.

## Repository Layout

- `run.py`: command-line entry point.
- `main.py`: system assembly, runtime orchestration, and metric export.
- `core/dispatcher.py`: CLIF subflow dispatcher with exploration, fitting, pressure-aware adjustment, and dispatch metrics.
- `core/coordinator.py`: joint training/inference batch-size optimizer for combined replicas.
- `core/fl_launcher.py`: federated PEFT round launcher and aggregation logic.
- `core/replica.py`: single-adapter and dual-adapter replica implementations.
- `common/`: dataset loading, evaluation, training, state tracking, request generation, and GPU monitoring.
- `baselines/`: round-robin, dLoRA-style, Shepherd-style, and dual-model baselines.
- `scripts/`: generic reproduction scripts.
- `docs/`: design notes for the anonymous artifact.

## Installation

```bash
conda create -n clif python=3.12 -y
conda activate clif
pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build that matches the target machine before running GPU experiments.

## Quick Smoke Test

The smoke test uses a fixed request stream and a small public model. Adjust GPU IDs and model name for the local environment.

```bash
bash scripts/run_smoke_fixed.sh
```

## Example Ablation

The dispatcher can be switched independently from the system baseline:

```bash
python run.py --baseline CLIF --dispatcher subflow
python run.py --baseline CLIF --dispatcher round
```

This keeps the CLIF fine-tuning and replica logic unchanged while replacing the dispatch policy.

## Output Metrics

Each run writes structured metrics under `output/`:

- `serve_metrics.xlsx`: per-served-batch latency, success, throughput, token, and quality metrics.
- `train_metrics.xlsx`: per-round local PEFT metrics.
- `train_step_metrics.xlsx`: per-update local training metrics.
- `dispatch_metrics.xlsx`: dispatcher queueing and routing events.
- `request_gen_metrics.xlsx`: generated request records.
- `state_metrics.xlsx`: replica state transitions with timestamps.
- `fl_round_metrics.xlsx`: federated round summaries.
- `gpu_monitor.xlsx`: GPU utilization and memory statistics when available.
- `summary.xlsx`: aggregate run-level metrics.

## Artifact Notes

This directory is an anonymous artifact version. It excludes generated outputs, local traces, private paths, checkpoints, cache directories, and manuscript-specific material.
