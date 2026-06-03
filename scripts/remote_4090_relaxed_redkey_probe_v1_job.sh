#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-4090_relaxed_redkey_probe_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-artifacts/runs/${RUN_ID}}"
UPSTREAM_RUN="${UPSTREAM_RUN:-artifacts/runs/4090_gem_chain_v7_20260527_210353}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TIMEOUT_MIN="${TIMEOUT_MIN:-25}"

mkdir -p "${RUN_DIR}"/{env,red_key,logs,validation}
echo "${RUN_DIR}" > artifacts/runs/latest_remote_relaxed_redkey_probe.txt

{
  echo "pwd=$(pwd)"
  echo "python=$(${PYTHON_BIN} --version 2>&1)"
  echo "upstream_run=${UPSTREAM_RUN}"
  echo "timeout_min=${TIMEOUT_MIN}"
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
} | tee "${RUN_DIR}/env/env_check.txt"

ADAPT_RED='{"hp_delta":0.12,"atk_delta":210.0,"def_delta":230.0,"yellow_key_delta":65.0,"blue_key_delta":80.0,"yellow_key_level":22.0,"blue_key_level":30.0,"guard_margin_delta":1.2,"guard_margin_level":0.015,"red_key_route_margin_delta":1.2,"red_key_route_margin_level":0.012,"last_yellow_key_spent":220.0}'

run_probe() {
  local gpu="$1"
  local tag="$2"
  local route="$3"
  [ -s "${route}" ] || return 0
  CUDA_VISIBLE_DEVICES="${gpu}" MOTA_FAST_GRAPH_STATE=1 timeout "${TIMEOUT_MIN}m" \
    "${PYTHON_BIN}" -m mota_rl.beam_decode \
      --target-stage red_key \
      --start-route "${route}" \
      --allow-negative-hp \
      --relaxed-min-hp -1800 \
      --beam-width 420 \
      --action-top-k 44 \
      --max-steps 240 \
      --model-weight 0.0 \
      --action-bias-weight 1.65 \
      --potential-weight 0.130 \
      --fast-score-weight 0.0060 \
      --distance-weight 2.0 \
      --path-step-penalty 0.0018 \
      --macro-step-penalty 0.010 \
      --revisit-penalty 2.1 \
      --success-bonus 2200 \
      --max-per-diversity-key 22 \
      --continue-after-success \
      --success-patience 36 \
      --adaptive-weights-json "${ADAPT_RED}" \
      --route-out "${RUN_DIR}/red_key/${tag}_red_key_route.jsonl" \
      --summary-out "${RUN_DIR}/red_key/${tag}_red_key_summary.json" \
      --trace-out "${RUN_DIR}/red_key/${tag}_red_key_trace.jsonl" \
      > "${RUN_DIR}/logs/${tag}.log" 2>&1 || true
}

run_probe 0 s1_before "${UPSTREAM_RUN}/mid_gems/s1_before_mt8_low_gems_route_mid_gems_route.jsonl" &
run_probe 1 s1_after "${UPSTREAM_RUN}/mid_gems/s1_after_7f_buy_low_gems_trace_best_route_mid_gems_route.jsonl" &
wait || true

for route in "${RUN_DIR}"/red_key/*_route.jsonl "${RUN_DIR}"/red_key/*_trace_best_route.jsonl; do
  [ -s "${route}" ] || continue
  safe="$(echo "${route#${RUN_DIR}/}" | tr '/.' '__')"
  PYTHONPATH=src "${PYTHON_BIN}" scripts/replay_route.py --route "${route}" > "${RUN_DIR}/validation/${safe}_replay.json" || true
done

date -Is | tee "${RUN_DIR}/finished_at.txt"
