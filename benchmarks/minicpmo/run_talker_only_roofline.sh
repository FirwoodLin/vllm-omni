#!/usr/bin/env bash
# Production-engine Talker roofline sweep — extend bs past 8 to find the
# aggregate-throughput plateau. Production CUDA Graph path only.
#
# 8-GPU worker-pool dispatch: each GPU holds one (bs, budget) setting;
# when a GPU goes idle the dispatcher assigns it the next batch in the
# staircase. The same pattern worked for b_engTkv8 (GPUs 4-7 held bs 1/2/4/8).
#
# Memory budget per GPU: b_engTkv8 peak at bs=8 KV=3584 was ~64 GiB;
# roofline points run at KV=512 so peak per process is well under that.
# 8 parallel model loads fit easily into 8 × ~120 GiB available on H200.
#
# Mirrors the design of minicpmo45_thinker_roofline_g_20260917/run.sh:
#   - fix KV at a small value (KV=512) so the bs range is gated by
#     activations + cudagraph workspace, not by KV cache size (KV=512
#     at bs=128 only costs 128*512*2*8*128*2 bytes ≈ 134 MiB, vs model
#     weights + activations dominating the budget).
#   - raise --max-num-seqs and --max-num-batched-tokens together so the
#     scheduler admits the larger bs and piecewise prefill doesn't
#     artificially split the prefill chunk at the default 8192 budget.
#   - talker_min_tokens=0 to mirror the b_engTkv8 short-token branch.
set -euo pipefail
ROOT=/app/vllm-omni
OUT=$ROOT/intermediate/minicpmo45_talker_roofline_20260918
mkdir -p "$OUT"
cd "$ROOT"

WARMUP=1
REPEATS=3
NUM_GPUS=8

# Batch staircase. max_num_seqs == bs + small slack so the scheduler
# admits exactly the batch we want. max_num_batched_tokens covers
# bs*(KV+2) + safety (KV=512, so bs*514 → next power-of-two-ish budget):
#   bs=16,  KV=512  -> 16*514  = 8224   (16384)
#   bs=32,  KV=512  -> 32*514  = 16448  (32768)
#   bs=64,  KV=512  -> 64*514  = 32896  (65536)
#   bs=96,  KV=512  -> 96*514  = 49344  (98304)
#   bs=128, KV=512  -> 128*514 = 65792  (131072)
#   bs=192, KV=512  -> 192*514 = 98688  (196608)
BATCHES=(16 32 64 96 128 192)
declare -A BUDGET
BUDGET[16]=16384
BUDGET[32]=32768
BUDGET[64]=65536
BUDGET[96]=98304
BUDGET[128]=131072
BUDGET[192]=196608

# Active worker registry: gpu -> pid
declare -A WORKERS
WORKER_PIDS=()

is_gpu_idle() {
  local gpu=$1
  local mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')
  # Idle threshold: <500 MiB used (a fully loaded engine sits at ~50-65 GiB)
  if [ "$mem" -lt 500 ]; then
    return 0
  fi
  return 1
}

wait_for_idle_gpu() {
  local gpu
  while true; do
    for gpu in $(seq 0 $((NUM_GPUS-1))); do
      # Skip GPUs already holding a worker
      if [ -n "${WORKERS[$gpu]:-}" ]; then
        continue
      fi
      if is_gpu_idle "$gpu"; then
        echo "$gpu"
        return 0
      fi
    done
    sleep 5
  done
}

launch_bs() {
  local bs=$1 gpu=$2
  local json="$OUT/talker_roofline_bs${bs}.json"
  local log="$OUT/talker_roofline_bs${bs}.log"
  local stdout="$OUT/talker_roofline_bs${bs}.stdout"
  echo "[$(date +%H:%M:%S)] dispatch bs=$bs budget=${BUDGET[$bs]} -> gpu=$gpu"
  CUDA_VISIBLE_DEVICES=$gpu \
    .venv/bin/python benchmarks/minicpmo/benchmark_talker_only_engine.py \
      --batch-sizes "$bs" \
      --kv-lengths "0,512" \
      --talker-min-tokens "0" \
      --warmup $WARMUP --repeats $REPEATS \
      --max-num-seqs "$bs" \
      --max-num-batched-tokens "${BUDGET[$bs]}" \
      --output-json "$json" \
      --engine-log "$log" \
      > "$stdout" 2>&1 &
  WORKERS[$gpu]=$!
  WORKER_PIDS+=("$!")
  echo "[$(date +%H:%M:%S)]   pid=$! bs=$bs gpu=$gpu"
}

reap_finished() {
  local gpu
  for gpu in "${!WORKERS[@]}"; do
    local pid=${WORKERS[$gpu]}
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null || true
      echo "[$(date +%H:%M:%S)]   gpu=$gpu (pid=$pid) finished"
      unset WORKERS[$gpu]
    fi
  done
}

# Main dispatch loop
for bs in "${BATCHES[@]}"; do
  # Wait until at least one GPU is free AND empty (idle)
  while true; do
    reap_finished
    gpu=$(wait_for_idle_gpu)
    # wait_for_idle_gpu only returns idle GPUs (skipping those with active workers),
    # so this gpu is guaranteed free
    break
  done
  launch_bs "$bs" "$gpu"
done

echo "[$(date +%H:%M:%S)] all batches dispatched, waiting for remaining workers"
for pid in "${WORKER_PIDS[@]}"; do
  wait "$pid" 2>/dev/null || true
done
echo "all done"
