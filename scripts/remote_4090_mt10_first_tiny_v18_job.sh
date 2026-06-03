#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-4090_mt10_first_v18_tiny_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-artifacts/runs/${RUN_ID}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TARGET_STAGE="${TARGET_STAGE:-mt10_first_resource}"
mkdir -p "${RUN_DIR}"/{mt10,logs}

route="${START_ROUTE:-artifacts/runs/4090_guard_chain_v9_20260528_012259/hp_ready/s1_before_mt8_low_gems_route_mid_gems_route_hp_ready_route.jsonl}"

timeout 8m "${PYTHON_BIN}" -m mota_rl.beam_decode \
  --target-stage "${TARGET_STAGE}" \
  --start-route "${route}" \
  --model-weight 0.0 \
  --beam-width 8 \
  --action-top-k 3 \
  --max-steps 35 \
  --action-bias-weight 5.0 \
  --potential-weight 0.05 \
  --fast-score-weight 0.001 \
  --distance-weight 8.0 \
  --path-step-penalty 0.0001 \
  --macro-step-penalty 0.001 \
  --success-bonus 10000 \
  --max-per-diversity-key 8 \
  --route-out "${RUN_DIR}/mt10/tiny_first_resource_route.jsonl" \
  --summary-out "${RUN_DIR}/mt10/tiny_first_resource_summary.json" \
  --trace-out "${RUN_DIR}/mt10/tiny_first_resource_trace.jsonl" \
  > "${RUN_DIR}/logs/tiny.log" 2>&1 || true

date -Is | tee "${RUN_DIR}/finished_at.txt"
