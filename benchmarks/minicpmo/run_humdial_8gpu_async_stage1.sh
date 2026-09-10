#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Run the Stage-1 async-scheduler A/B sweep.  This script intentionally calls
# the service benchmark directly so it cannot be selected accidentally by the
# existing `--topology all` baseline sweep.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
BENCHMARK="${ROOT}/benchmarks/minicpmo/benchmark_humdial_service.py"
DEPLOY_CONFIG="${SCRIPT_DIR}/configs/humdial_8gpu/thinker6_downstream_colocated_async_stage1.yaml"
MODEL="${MODEL:-/mnt/shared-storage-user/gpfs2-shared-public/huggingface/zskj-hub/models--OpenBMB--MiniCPM-o-4_5}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/nvme1n1/ml_research/linbinbin1/src-omni-modal/Humdial-Track2-Test}"
CLIENT_MODE="${CLIENT_MODE:-arrival}"
E2E_MANIFEST="${E2E_MANIFEST:-}"
FEEDBACK_CONTRACT="${FEEDBACK_CONTRACT:-L2}"
REF_AUDIO="${REF_AUDIO:-${MODEL}/assets/HT_ref_audio.wav}"
CAMPAIGN="${CAMPAIGN:-humdial_8gpu}"
TOPOLOGY="thinker6_downstream_colocated_async_stage1"
ATTEMPT="${1:-attempt_async_stage1_001}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/intermediate/humdial_minicpmo_serving}"
OUTPUT_DIR="${OUTPUT_ROOT}/${CAMPAIGN}/${TOPOLOGY}/${ATTEMPT}"

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "refusing existing attempt directory: ${OUTPUT_DIR}" >&2
  echo "choose a new attempt label; partial artifacts are kept intact" >&2
  exit 2
fi

cmd=(
  "${PYTHON}" "${BENCHMARK}"
  --model "${MODEL}" \
  --deploy-config "${DEPLOY_CONFIG}" \
  --dataset-root "${DATASET_ROOT}" \
  --client-mode "${CLIENT_MODE}" \
  --ref-audio "${REF_AUDIO}" \
  --gpus "${GPUS:-0,1,2,3,4,5,6,7}" \
  --rates "${RATES:-0.025,0.05,0.1}" \
  --duration-s "${DURATION_S:-300}" \
  --seed "${SEED:-20260901}" \
  --repeats "${REPEATS:-1}" \
  --warmup-rate "${WARMUP_RATE:-0.05}" \
  --warmup-duration-s "${WARMUP_DURATION_S:-60}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-label "humdial-${TOPOLOGY}-${ATTEMPT}" \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-8113}" \
  --omni-master-address "${OMNI_MASTER_ADDRESS:-127.0.0.1}" \
  --omni-master-port "${OMNI_MASTER_PORT:-26000}" \
  --startup-timeout-s "${STARTUP_TIMEOUT_S:-1800}" \
  --request-timeout-s "${REQUEST_TIMEOUT_S:-180}" \
  --chunk-ms "${CHUNK_MS:-200}" \
  --tail-drain-s "${TAIL_DRAIN_S:-2}" \
  --playback-initial-buffer-ms "${PLAYBACK_INITIAL_BUFFER_MS:-300}" \
  --model-unit-decision-p99-slo-ms "${DECISION_P99_SLO_MS:-1500}" \
  --telemetry-interval-s "${TELEMETRY_INTERVAL_S:-1}" \
  --idle-stability-s "${IDLE_STABILITY_S:-30}" \
  --max-preexisting-memory-mib "${MAX_PREEXISTING_MEMORY_MIB:-1024}" \
  --omni-lb-policy "${OMNI_LB_POLICY:-least-queue-length}" \
  --no-stop-on-slo-fail \
  --parallel-stage-launch
)

if [[ "${CLIENT_MODE}" == "e2e" ]]; then
  if [[ -z "${E2E_MANIFEST}" ]]; then
    echo "E2E_MANIFEST is required when CLIENT_MODE=e2e" >&2
    exit 2
  fi
  cmd+=(--e2e-manifest "${E2E_MANIFEST}" --feedback-contract "${FEEDBACK_CONTRACT}")
  if [[ "${EXPLICIT_FOLLOWUP_RESPONSE:-0}" == "1" ]]; then
    cmd+=(--explicit-followup-response)
  fi
  if [[ "${EXPLICIT_ALL_RESPONSES:-0}" == "1" ]]; then
    cmd+=(--explicit-all-responses)
  fi
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

exec "${cmd[@]}"
