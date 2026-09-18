#!/usr/bin/env bash
# Fan the production-engine Talker-only KV8 sweep across 4 GPUs (4,5,6,7).
# Each GPU owns one batch-size slot:
#   GPU4 -> bs1, GPU5 -> bs2, GPU6 -> bs4, GPU7 -> bs8
# All GPUs sweep the same 8-point rolling-KV grid with 3 repeats + 1 warmup.
set -euo pipefail
ROOT=/app/vllm-omni
OUT=$ROOT/intermediate/minicpmo45_talker_kv8_engine_20260918
mkdir -p "$OUT"
cd "$ROOT"

# Reuse the same KV grid the eager §5.4 sweep used.
KV="0,512,1024,1536,2048,2560,3072,3584"
MIN_TOKENS="0,1024"
WARMUP=1
REPEATS=3
COMMON_KV="--kv-lengths $KV --talker-min-tokens $MIN_TOKENS --warmup $WARMUP --repeats $REPEATS"

run_one() {
  local gpu=$1 bs=$2
  local json="$OUT/talker_kv8_gpu${gpu}_bs${bs}.json"
  local log="$OUT/talker_kv8_gpu${gpu}_bs${bs}.log"
  echo "==> GPU$gpu bs=$bs"
  CUDA_VISIBLE_DEVICES=$gpu \
    .venv/bin/python benchmarks/minicpmo/benchmark_talker_only_engine.py \
      --batch-sizes $bs $COMMON_KV \
      --output-json "$json" \
      --engine-log "$log" \
      > "$OUT/talker_kv8_gpu${gpu}_bs${bs}.stdout" 2>&1
}

run_one 4 1 &
run_one 5 2 &
run_one 6 4 &
run_one 7 8 &
wait
echo "all done"
