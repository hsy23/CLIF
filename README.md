# CLIF

CLIF is a research prototype for continuous PEFT fine-tuning under online LLM inference load. The system explores how a shared pool of LLM replicas can serve requests and opportunistically perform federated adapter updates without collapsing serving quality.

## Core Mechanisms

CLIF is built around a state-aware replica pool, a pressure-aware dispatcher, and a federated PEFT launcher. The key idea is to treat fine-tuning as an opportunistic background activity whose admission and resource allocation are continuously constrained by online serving pressure.

### State-Aware Replica Pool

Each replica is managed as a stateful worker:

- `SERVING`: the replica is dedicated to inference.
- `IDLE`: the replica is lightly loaded and remains available for inference.
- `COMBINED`: the replica serves inference while participating in local PEFT fine-tuning.

The state manager records every state transition in `state_metrics.xlsx`, which makes it possible to align inference latency, request success, dispatcher decisions, and federated rounds during analysis.

### Pressure-Aware Fine-Tuning Admission

Before starting a PEFT round, the fine-tuning launcher queries recent serving pressure from the dispatcher. The pressure summary combines recent request success rate, dispatch queueing delay, end-to-end service time, and request backlog. Under high pressure, candidate replicas are returned to `SERVING` and the round is postponed. Under moderate pressure, only a subset of eligible replicas can enter `COMBINED`. Under low pressure, all eligible replicas may participate.

This admission rule prevents fine-tuning from consuming serving capacity when the system is already overloaded, while still allowing local adaptation when there is service slack.

### Dispatcher Control Loops

The CLIF dispatcher uses two lightweight control loops:

- A slower fitting loop estimates the relationship between inference batch size and observed processing time, then computes per-replica batch bounds.
- A faster adjustment loop uses recent service satisfaction and replica quality to refine inference batch sizes online.

Together, these loops let the dispatcher react to both long-term capacity changes and short-term quality or load imbalance. The dispatcher is intentionally separated from the system baseline, so the same CLIF fine-tuning logic can be evaluated with either the CLIF subflow dispatcher or a simpler round-robin dispatcher.

### Joint Batch Sizing for Combined Replicas

When replicas enter `COMBINED`, CLIF jointly reasons about training and inference batch sizes. The coordinator uses observed training and inference metrics to estimate how training batch size affects fine-tuning progress and how inference batch size affects serving latency. The resulting decision keeps inference batch sizes above a configured minimum while selecting feasible training batch sizes for the current service budget.

### Dual-Adapter Execution

Dual-adapter replicas isolate the adapter being served from the adapter being trained. Inference uses the active adapter, while local PEFT updates are applied to a shadow adapter. At controlled promotion points, the trained shadow adapter is swapped into the active role. This design reduces interference between online inference and local fine-tuning updates.

### Federated PEFT Aggregation

After each round, the launcher aggregates participating replicas' adapter states and redistributes the global adapter. Replica participation is also filtered by local training effectiveness: replicas whose local loss no longer improves can be excluded from later rounds and returned to serving capacity.

## System Overview

The following figure summarizes CLIF's runtime architecture. Requests enter the proactive dispatcher, while the fine-tuning task launcher identifies eligible replicas and controls whether they can safely join PEFT rounds. The global monitor and replica state manager provide the shared feedback signals used by the dispatcher, launcher, and coordinator.

![CLIF system overview](docs/images/clif_overview.png)

The coordinator jointly controls inference serving and local fine-tuning for replicas in `COMBINED`. It uses runtime feedback to balance serving slack, local PEFT progress, and federated aggregation requirements.

![CLIF inference-training coordinator](docs/images/it_coordinator.png)

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
