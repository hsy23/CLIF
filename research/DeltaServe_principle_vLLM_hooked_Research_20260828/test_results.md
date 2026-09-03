# Verification results - 2026-08-28

## End-to-end WSL GPU run

Environment:

- Ubuntu 24.04 under WSL
- NVIDIA GeForce RTX 5070 Ti Laptop GPU
- PyTorch 2.11.0+cu130 / CUDA 13.0
- vLLM 0.21.0
- existing local Qwen3-0.6B snapshot; no model download was performed

Command:

```bash
.venv/bin/python scripts/run_deltaserve_vllm_hooked.py \
  --model /home/shaoyuan/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
  --trace research/DeltaServe_vLLM_single_replica_Research_20260827/hooked-trace.jsonl \
  --output research/DeltaServe_vLLM_single_replica_Research_20260827/hooked-result.json
```

Result: exit code 0. All asserted invariants passed:

- the live vLLM LM-head base allocation was exported once and mapped by the backward child;
- the synthetic FT row (37 scheduled tokens) and inference prefill row (7 scheduled tokens) appeared in one model-forward batch;
- the final-RMSNorm hook captured the FT flat slice `[0, 37]` and produced 36 supervised token pairs;
- runtime PID 423 and backward PID 557 were different GPU processes;
- parent forward loss and child backward loss both equalled `6.731245517730713`;
- LoRA-B changed by L1 `15712.62109375` and the adapter was published to live vLLM logits.

Authoritative artifacts:

- `../DeltaServe_vLLM_single_replica_Research_20260827/hooked-result.json`
- `../DeltaServe_vLLM_single_replica_Research_20260827/hooked-trace.jsonl`

## Unit tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Result: 9 tests passed in 1.737 seconds, including the two version-pinned vLLM patch idempotence tests.

## Research artifact validation

`validate_report.py` passed all nine checks. `verify_citations.py` found all 16 bibliography entries; its Windows process could not reach external URLs and flagged the three local artifacts as non-URL sources, so web sources were manually checked through the browser retrieval used for the report.

The HTML report was generated. The bundled HTML verifier has a known regex issue that classifies CJK characters as emoji; its CJK error is therefore a verifier false positive, not report content corruption.

The PDF contains seven pages and 15,786 extractable characters. All rendered pages were visually inspected; no clipping, overlap, missing glyphs, or broken table layout was observed.
