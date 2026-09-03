#!/usr/bin/env bash
#SBATCH --job-name=clif-qwen3-attn-ref
#SBATCH --partition=interruptible_gpu
#SBATCH --constraint=l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --chdir=/users/k24104674/CLIF_2027
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

ROOT="${CLIF_ROOT:-$PWD}"
VENV="${ROOT}/.venv"
MODEL="${MODEL:-/scratch/users/k24104674/models/Qwen3-0.6B}"
OUTPUT="${OUTPUT:-${ROOT}/output/qwen3-attention-reference-${SLURM_JOB_ID:-local}.json}"
CUDA_ROOT="${CUDA_ROOT:-${VENV}/lib/python3.10/site-packages/nvidia/cu13}"

if [[ ! -d "$MODEL" || ! -x "$VENV/bin/python" || ! -x "$CUDA_ROOT/bin/nvcc" ]]; then
  echo "model, python, or CUDA 13 nvcc is unavailable" >&2
  exit 2
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_HOME="$CUDA_ROOT"
export CUDA_PATH="$CUDA_ROOT"
export CUDACXX="$CUDA_ROOT/bin/nvcc"
export PATH="$CUDA_ROOT/bin:$VENV/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib:$CUDA_ROOT/lib64:${VENV}/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:${VENV}/lib/python3.10/site-packages/nvidia/cublas/lib:${VENV}/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
export CPATH="$CUDA_ROOT/include:$CUDA_ROOT/include/cccl:${CPATH:-}"
export CPLUS_INCLUDE_PATH="$CUDA_ROOT/include:$CUDA_ROOT/include/cccl:${CPLUS_INCLUDE_PATH:-}"

exec "$VENV/bin/python" "$ROOT/scripts/run_qwen3_attention_lora_reference.py" \
  --model "$MODEL" \
  --output "$OUTPUT" \
  --sequence-length "${SEQUENCE_LENGTH:-96}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --rank "${LORA_RANK:-4}" \
  --alpha "${LORA_ALPHA:-8}" \
  --learning-rate "${LEARNING_RATE:-3e-2}"
