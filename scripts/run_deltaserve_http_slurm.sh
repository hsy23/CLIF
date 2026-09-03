#!/usr/bin/env bash
#SBATCH --job-name=clif-ds-http
#SBATCH --partition=interruptible_gpu
#SBATCH --constraint=l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --chdir=/users/k24104674/CLIF_2027
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

ROOT="${CLIF_ROOT:-$PWD}"
VENV="${ROOT}/.venv"
MODEL="${MODEL:-/scratch/users/k24104674/models/Qwen3-0.6B}"
OUTPUT="${OUTPUT:-${ROOT}/output/deltaserve-http-${SLURM_JOB_ID:-local}.json}"
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

"$VENV/bin/python" "$ROOT/scripts/apply_vllm_deltaserve_patch.py"
exec "$VENV/bin/python" "$ROOT/scripts/run_deltaserve_vllm_http_benchmark.py" \
  --model "$MODEL" \
  --output "$OUTPUT" \
  --trace "$TRACE" \
  --clients "${CLIENTS:-8}" \
  --training-steps "${TRAINING_STEPS:-100}" \
  --training-warmup-steps "${TRAINING_WARMUP_STEPS:-1}" \
  --training-batch-size "${TRAINING_BATCH_SIZE:-1}" \
  --training-sequence-length "${TRAINING_SEQUENCE_LENGTH:-96}" \
  --max-model-len "${MAX_MODEL_LEN:-512}" \
  --max-tokens "${MAX_TOKENS:-24}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.55}"
