#!/usr/bin/env bash
#SBATCH --job-name=clif-ds-vllm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --chdir=/users/k24104674/CLIF_2027

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
ROOT="${CLIF_ROOT:-$PWD}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
MODEL="${MODEL:-}"
OUTPUT="${OUTPUT:-$ROOT/output/deltaserve-vllm-result.json}"
TRACE="${TRACE:-$ROOT/output/deltaserve-vllm-trace.jsonl}"

if [[ -z "$MODEL" ]]; then
    echo 'Set MODEL to an existing local Hugging Face snapshot.' >&2
    exit 2
fi
if [[ ! -d "$MODEL" ]]; then
    echo "MODEL is not a directory: $MODEL" >&2
    exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
    echo "Python executable not found: $PYTHON" >&2
    exit 2
fi

CUDA_ROOT="${CUDA_ROOT:-$ROOT/.venv/lib/python3.10/site-packages/nvidia/cu13}"
if [[ ! -x "$CUDA_ROOT/bin/nvcc" ]]; then
    echo "CUDA 13 nvcc not found: $CUDA_ROOT/bin/nvcc" >&2
    exit 2
fi

mkdir -p "$(dirname -- "$OUTPUT")" "$(dirname -- "$TRACE")"
export FLASHINFER_WORKSPACE_BASE="$(dirname -- "$OUTPUT")/.flashinfer-workspace"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="false"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export PATH="$CUDA_ROOT/bin:$PATH"
export CUDA_HOME="$CUDA_ROOT"
export CUDA_PATH="$CUDA_ROOT"
export CUDACXX="$CUDA_ROOT/bin/nvcc"
export FLASHINFER_NVCC="$CUDA_ROOT/bin/nvcc"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib:$CUDA_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CPATH="$CUDA_ROOT/include/cccl:$CUDA_ROOT/include${CPATH:+:$CPATH}"
export CPLUS_INCLUDE_PATH="$CUDA_ROOT/include/cccl:$CUDA_ROOT/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export CLIF_DELTASERVE_ENABLE=1

exec "$PYTHON" "$ROOT/scripts/run_deltaserve_vllm_hooked.py" \
    --model "$MODEL" \
    --trace "$TRACE" \
    --output "$OUTPUT"
