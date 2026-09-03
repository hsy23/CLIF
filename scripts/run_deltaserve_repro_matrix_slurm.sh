#!/usr/bin/env bash
#SBATCH --job-name=clif-repro-matrix
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --chdir=/users/k24104674/CLIF_2027

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
ROOT="${CLIF_ROOT:-$PWD}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
MODEL="${MODEL:-/scratch/users/k24104674/models/Qwen3-0.6B}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/output/repro-matrix-native}"
REPETITIONS="${REPETITIONS:-3}"
SAMPLER="${SAMPLER:-native}"
CUDA_ROOT="${CUDA_ROOT:-$ROOT/.venv/lib/python3.10/site-packages/nvidia/cu13}"

if [[ ! -d "$MODEL" ]]; then
    echo "MODEL is not a directory: $MODEL" >&2
    exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
    echo "Python executable not found: $PYTHON" >&2
    exit 2
fi
if [[ ! -x "$CUDA_ROOT/bin/nvcc" ]]; then
    echo "CUDA 13 nvcc not found: $CUDA_ROOT/bin/nvcc" >&2
    exit 2
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="false"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export CUDA_HOME="$CUDA_ROOT"
export CUDA_PATH="$CUDA_ROOT"
export CUDACXX="$CUDA_ROOT/bin/nvcc"
export FLASHINFER_NVCC="$CUDA_ROOT/bin/nvcc"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib:$CUDA_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CPATH="$CUDA_ROOT/include/cccl:$CUDA_ROOT/include${CPATH:+:$CPATH}"
export CPLUS_INCLUDE_PATH="$CUDA_ROOT/include/cccl:$CUDA_ROOT/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export PATH="$CUDA_ROOT/bin:$PATH"

mkdir -p "$OUTPUT_DIR"
export FLASHINFER_WORKSPACE_BASE="$OUTPUT_DIR/.flashinfer-workspace"
exec "$PYTHON" "$ROOT/scripts/run_deltaserve_repro_matrix.py" \
    --model "$MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --repetitions "$REPETITIONS" \
    --sampler "$SAMPLER" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.55}" \
    --max-model-len "${MAX_MODEL_LEN:-512}" \
    --max-tokens "${MAX_TOKENS:-24}" \
    --baseline-requests "${BASELINE_REQUESTS:-8}" \
    --training-steps "${TRAINING_STEPS:-8}" \
    --training-batch-size "${TRAINING_BATCH_SIZE:-1}" \
    --training-sequence-length "${TRAINING_SEQUENCE_LENGTH:-96}" \
    --mixed-warmup-steps "${MIXED_WARMUP_STEPS:-1}" \
    --separated-mixed-mode "${SEPARATED_MIXED_MODE:-paired-step}"
