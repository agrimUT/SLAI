#!/usr/bin/env bash
set -euo pipefail
source ~/mambaforge/etc/profile.d/conda.sh
conda activate ~/sarathi-serve/env

REPO=~/sarathi-serve
OUT=$REPO/benchmark_output/plot3
MODEL="mistralai/Mistral-7B-Instruct-v0.2"
GPU=0

DECODE_NS=(8 16 32 64 108)   # sweep points
TOK_BUDGET=2048
REPS=1                             # repetitions per N

mkdir -p "$OUT"
SUMMARY="$OUT/summary.csv"
echo "N_decode,run_id,batch_exec_time_s" > "$SUMMARY"       # overwrite each run

for N in "${DECODE_NS[@]}"; do
  NUM_REQ=$(( N + 1 ))
  TRACE="$REPO/scripts/trace_p2048_n${N}.csv"                     # pre-generated

  for r in $(seq 1 "$REPS"); do
    echo -e "\n===== N=$N  run=$r/$REPS ====="
    ray stop -f || true
    export CUDA_VISIBLE_DEVICES=$GPU
    export MASTER_PORT=$(( 10000 + RANDOM % 40000 ))
    # main.py adds its own timestamp directory under OUT_BASE
    OUT_BASE="$OUT/dec_${N}"
    python $REPO/sarathi/benchmark/main.py \
      --output_dir "$OUT_BASE" \
      --model_name "$MODEL" \
      --model_max_model_len 8192 \
      --cluster_num_replicas 1 \
      --model_tensor_parallel_degree 1 \
      --model_pipeline_parallel_degree 1 \
      --request_generator_provider synthetic \
      --synthetic_request_generator_interval_provider static \
      --synthetic_request_generator_length_provider trace \
      --trace_request_length_generator_trace_file "$TRACE" \
      --synthetic_request_generator_num_requests "$NUM_REQ" \
      --replica_scheduler_provider hold_n \
      --replica_scheduler_max_batch_size "$NUM_REQ" \
      --hold_n_scheduler_hold_n "$N" \
      --hold_n_scheduler_token_budget "$TOK_BUDGET" \
      --metrics_store_keep_individual_batch_metrics true \
      --metrics_store_enable_op_level_metrics false

    # ------------------------------------------------------------------
    # Locate the newest timestamp directory just created by main.py
    #   OUT_BASE/
    #        └── 2025-06-19_13-40-56-050386/
    #                 └── replica_0/batch_metrics.csv
    # ------------------------------------------------------------------
    RUN_DIR=$(ls -td "$OUT_BASE"/*/ | head -n 1)            # trailing slash kept
    CSV="$RUN_DIR/replica_0/batch_metrics.csv"

    # Extract batch_execution_time from the last row (11th column)
    LAST_TIME=$(tail -n 1 "$CSV" | cut -d',' -f11)
    echo "$N,$r,$LAST_TIME" >> "$SUMMARY"

    # ------------------------------------------------------------------
    #  Clean-up: delete the whole timestamped run directory
    # ------------------------------------------------------------------
    echo "   ➜ recorded $LAST_TIME s  (deleted $RUN_DIR)"
  done
done

echo -e "\nAll done.  Summary in $SUMMARY"
