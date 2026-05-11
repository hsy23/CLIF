# Architecture Overview

CLIF separates online inference control from federated fine-tuning control while allowing both to share the same replica pool.

## Runtime Flow

1. `RequestGenerator` creates either fixed-rate requests or trace-driven requests.
2. `SubflowDispatcher` assigns requests to replicas and maintains dispatch metrics.
3. Each `Replica` serves requests in batches and reports latency, success, quality, and throughput metrics.
4. `StateManagement` records replica state transitions.
5. `FederatedLearningLauncher` periodically evaluates service pressure and selects eligible replicas for PEFT rounds.
6. `Coordinator` computes batch-size decisions for replicas in `COMBINED`.
7. Replica adapters are locally trained and aggregated into the next global adapter state.

## Dispatcher

The dispatcher has two control loops. The slower loop fits a lightweight inference-time model from observed serving batches and updates per-replica batch bounds. The faster loop adjusts batch sizes using recent service satisfaction and replica quality scores. This design keeps dispatch decisions responsive without making every request depend on expensive optimization.

## Pressure Feedback

Before each federated round, the launcher queries recent service pressure from the dispatcher. The pressure summary includes recent success rate, dispatch queueing delay, end-to-end service time, and backlog. High pressure blocks fine-tuning and returns candidate replicas to `SERVING`. Moderate pressure admits only a subset of eligible replicas into `COMBINED`. Low pressure permits all eligible replicas to participate.

## Dual Adapter

The dual-adapter replica keeps an active adapter for inference and a shadow adapter for local PEFT updates. Local training modifies the shadow adapter. At controlled promotion points, the shadow adapter becomes active. This reduces interference between training updates and online inference.

## Metrics

The artifact exports structured spreadsheets rather than requiring log parsing. The most important cross-cutting table is `state_metrics.xlsx`, which makes it possible to align request latency, dispatcher decisions, and federated rounds with replica state transitions.
