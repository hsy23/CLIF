#!/usr/bin/env bash
#SBATCH --job-name=clif-deltaserve-stability
#SBATCH --partition=gpu
#SBATCH --constraint=l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --chdir=/users/k24104674/CLIF_2027
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
ROOT="${CLIF_ROOT:-$PWD}"
VENV="${ROOT}/.venv"
MODEL="${MODEL:-/scratch/users/k24104674/models/Qwen3-0.6B}"
OUTPUT="${OUTPUT:-${ROOT}/output/deltaserve-stability-${SLURM_JOB_ID:-local}}"

CUDA_ROOT="${VENV}/lib/python3.10/site-packages/nvidia/cu13"
export CUDA_HOME="$CUDA_ROOT"
export CUDA_PATH="$CUDA_ROOT"
export CUDACXX="$CUDA_ROOT/bin/nvcc"
export FLASHINFER_NVCC="$CUDA_ROOT/bin/nvcc"
export PATH="$CUDA_ROOT/bin:$VENV/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib:$CUDA_ROOT/lib64:${VENV}/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:${VENV}/lib/python3.10/site-packages/nvidia/cublas/lib:${VENV}/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
export CPATH="$CUDA_ROOT/include:$CUDA_ROOT/include/cccl:${CPATH:-}"
export CPLUS_INCLUDE_PATH="$CUDA_ROOT/include:$CUDA_ROOT/include/cccl:${CPLUS_INCLUDE_PATH:-}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

mkdir -p "$OUTPUT"
export FLASHINFER_WORKSPACE_BASE="$OUTPUT/.flashinfer-workspace"
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDACXX=$CUDACXX"
echo "FLASHINFER_NVCC=$FLASHINFER_NVCC"
echo "PATH_NVCC=$(command -v nvcc)"
echo "PATH_PTXAS=$(command -v ptxas)"
"$CUDACXX" --version | tail -n 1
"$CUDA_ROOT/bin/ptxas" --version | tail -n 1
exec "$VENV/bin/python" "$ROOT/scripts/run_deltaserve_stability.py" \
  --model "$MODEL" \
  --output "$OUTPUT/result.json" \
  --trace "$OUTPUT/trace.jsonl" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.55}" \
  --max-model-len "${MAX_MODEL_LEN:-512}" \
  --max-tokens "${MAX_TOKENS:-24}" \
  --max-num-seqs "${MAX_NUM_SEQS:-8}" \
  --training-steps "${TRAINING_STEPS:-100}" \
  --inference-requests "${INFERENCE_REQUESTS:-1000}" \
  --inference-batch-size "${INFERENCE_BATCH_SIZE:-10}" \
  --training-sequence-length "${TRAINING_SEQUENCE_LENGTH:-96}"
