#!/bin/bash
set -euo pipefail

source ~/mambaforge/etc/profile.d/conda.sh
conda activate ~/sarathi-serve/env

REPO=~/sarathi-serve
OUT=$REPO/benchmark_output/plot1
MODEL="mistralai/Mistral-7B-Instruct-v0.2"
GPU=0                      # pick the GPU you want
CHUNKS=(64 128 256 384 512 768 1024 2048)

mkdir -p "$OUT"

for CHUNK in "${CHUNKS[@]}"; do
  echo -e "\n===== chunk_size=$CHUNK ====="
  ray stop -f || true                 # free any old Ray workers
  export CUDA_VISIBLE_DEVICES=$GPU    # mask to ONE card

  python $REPO/sarathi/benchmark/main.py \
    --output_dir "$OUT/chunk_$CHUNK" \
    --model_name $MODEL \
    --model_max_model_len 2049 \
    --cluster_num_replicas 1 \
    --model_tensor_parallel_degree 1 \
    --model_pipeline_parallel_degree 1 \
    \
    --request_generator_provider synthetic \
    --synthetic_request_generator_interval_provider static \
    --synthetic_request_generator_num_requests 100 \
    --synthetic_request_generator_length_provider fixed \
    --fixed_request_length_generator_prefill_tokens 2048 \
    --fixed_request_length_generator_decode_tokens 1 \
    \
    --replica_scheduler_provider sarathi \
    --replica_scheduler_max_batch_size 1 \
    --sarathi_scheduler_chunk_size $CHUNK \
    --sarathi_scheduler_fcfs true \
    --sarathi_scheduler_enable_dynamic_chunking_schedule false \
    \
    --metrics_store_keep_individual_batch_metrics true \
    --metrics_store_enable_op_level_metrics false
done
