#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-./output}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-1B}"
REPLICA_GPUS="${REPLICA_GPUS:-[[0],[1]]}"

python run.py \
  --baseline CLIF \
  --dispatcher subflow \
  --enable_dual_adapter \
  --adapter_swap_strategy fixed \
  --adapter_swap_interval 4 \
  --model_name "${MODEL_NAME}" \
  --num_replicas 2 \
  --replica_gpus "${REPLICA_GPUS}" \
  --task_mode conv \
  --train_datasets_conv "tatsu-lab/alpaca" \
  --max_train_samples 256 \
  --train_batch_size 4 4 \
  --infer_batch_size 4 4 \
  --infer_length 128 \
  --max_new_tokens 64 \
  --generation_strategy greedy \
  --low_rank 8 \
  --initial_score 1.0 1.0 \
  --fl_rounds 2 \
  --local_steps 8 \
  --min_train_batch 2 \
  --max_train_batch 4 \
  --min_infer_batch 4 \
  --max_infer_batch 4 \
  --request_pattern fixed \
  --request_interval 1.0 \
  --concurrent_requests 2 \
  --ddl 5.0 \
  --max_requests 200 \
  --run_time 180 \
  --fl_start_time 30 \
  --idle_strategy ewma \
  --output_dir "${OUTPUT_DIR}" \
  --output_subdir "smoke_fixed"
