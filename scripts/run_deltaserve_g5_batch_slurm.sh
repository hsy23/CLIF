#!/usr/bin/env bash
#SBATCH --job-name=clif-ds-g5-batch
#SBATCH --partition=interruptible_gpu
#SBATCH --constraint=l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --chdir=/users/k24104674/CLIF_2027
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

ROOT="${CLIF_ROOT:-$PWD}"
VENV="${ROOT}/.venv"
MODEL="${MODEL:-/scratch/users/k24104674/models/Qwen3-0.6B}"
OUTPUT="${OUTPUT:-${ROOT}/output/deltaserve-g5-batch-${SLURM_JOB_ID:-local}}"
TRACE="${TRACE:-${OUTPUT%.json}.jsonl}"
CUDA_ROOT="${CUDA_ROOT:-${VENV}/lib/python3.10/site-packages/nvidia/cu13}"

if [[ ! -d "$MODEL" || ! -x "$VENV/bin/python" || ! -x "$CUDA_ROOT/bin/nvcc" ]]; then
  echo "model, python, or CUDA 13 nvcc is unavailable" >&2
  exit 2
fi

mkdir -p "$(dirname -- "$OUTPUT")" "$(dirname -- "$TRACE")"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export CUDA_HOME="$CUDA_ROOT"
export CUDA_PATH="$CUDA_ROOT"
export CUDACXX="$CUDA_ROOT/bin/nvcc"
export FLASHINFER_NVCC="$CUDA_ROOT/bin/nvcc"
export PATH="$CUDA_ROOT/bin:$VENV/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib:$CUDA_ROOT/lib64:${VENV}/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:${VENV}/lib/python3.10/site-packages/nvidia/cublas/lib:${VENV}/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
export CPATH="$CUDA_ROOT/include:$CUDA_ROOT/include/cccl:${CPATH:-}"
export CPLUS_INCLUDE_PATH="$CUDA_ROOT/include:$CUDA_ROOT/include/cccl:${CPLUS_INCLUDE_PATH:-}"
export FLASHINFER_WORKSPACE_BASE="$(dirname -- "$OUTPUT")/.flashinfer-workspace"

exec "$VENV/bin/python" "$ROOT/scripts/run_deltaserve_vllm_benchmark.py" \
  --model "$MODEL" \
  --output "$OUTPUT" \
  --trace "$TRACE" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.55}" \
  --max-model-len "${MAX_MODEL_LEN:-512}" \
  --max-tokens "${MAX_TOKENS:-24}" \
  --baseline-requests "${BASELINE_REQUESTS:-8}" \
  --training-steps "${TRAINING_STEPS:-8}" \
  --training-batch-size "${TRAINING_BATCH_SIZE:-4}" \
  --training-sequence-length "${TRAINING_SEQUENCE_LENGTH:-96}" \
  --mixed-warmup-steps "${MIXED_WARMUP_STEPS:-1}"
