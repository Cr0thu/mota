#!/usr/bin/env bash
set -euo pipefail

if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /opt/conda/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-humanoid}" || true
fi

export PYTHONPATH="${PYTHONPATH:-src}"
export PYTHONUNBUFFERED=1
export MOTA_SIM_CACHE_LIMIT="${MOTA_SIM_CACHE_LIMIT:-200000}"
export MOTA_GRAPH_CACHE_LIMIT="${MOTA_GRAPH_CACHE_LIMIT:-50000}"
export MOTA_ENABLE_STAGE_CACHE="${MOTA_ENABLE_STAGE_CACHE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

RUN_ID="${RUN_ID:-4090_no_hp403_alpha_broad_opt_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-artifacts/runs/${RUN_ID}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_RUN="${BASE_RUN:-artifacts/runs/4090_pure_alpha_zero_no_hp403_v2b_20260528}"
SWORD_CKPT="${SWORD_CKPT:-${BASE_RUN}/selfplay_from_planner_sword/final_model.pt}"
SWORD_ROUTE="${SWORD_ROUTE:-${BASE_RUN}/planner_sword/best_route.jsonl}"

mkdir -p "${RUN_DIR}"/{env,logs,validation,search,selfplay}
echo "${RUN_DIR}" > artifacts/runs/latest_remote_no_hp403_alpha_broad_opt_run.txt

{
  echo "pwd=$(pwd)"
  echo "run_dir=${RUN_DIR}"
  echo "hp403_usage=none"
  echo "sword_ckpt=${SWORD_CKPT}"
  echo "sword_route=${SWORD_ROUTE}"
  echo "python=$(${PYTHON_BIN} --version 2>&1)"
  echo "MOTA_SIM_CACHE_LIMIT=${MOTA_SIM_CACHE_LIMIT}"
  echo "MOTA_GRAPH_CACHE_LIMIT=${MOTA_GRAPH_CACHE_LIMIT}"
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
} | tee "${RUN_DIR}/env/env_check.txt"

if [ ! -s "${SWORD_CKPT}" ]; then
  echo "[no-hp403-broad] missing sword checkpoint: ${SWORD_CKPT}" >&2
  exit 1
fi
if [ ! -s "${SWORD_ROUTE}" ]; then
  echo "[no-hp403-broad] missing sword route: ${SWORD_ROUTE}" >&2
  exit 1
fi

validate_route() {
  local route="$1"
  local tag="$2"
  [ -s "${route}" ] || return 0
  "${PYTHON_BIN}" scripts/replay_route.py --route "${route}" > "${RUN_DIR}/validation/${tag}_replay.json" || true
  "${PYTHON_BIN}" scripts/validate_route_constraints.py "${route}" > "${RUN_DIR}/validation/${tag}_constraints.json" || true
}

alpha_variant() {
  local gpu="$1"
  local seed="$2"
  local tag="$3"
  local mix="$4"
  local c_puct="$5"
  local action_temp="$6"
  local root_noise="$7"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" scripts/train_alpha_mota_stage.py \
    --out-dir "${RUN_DIR}/selfplay/${tag}" \
    --target-stage shield \
    --init-checkpoint "${SWORD_CKPT}" \
    --start-route "${SWORD_ROUTE}" \
    --start-route-stop-stage sword \
    --episodes 42 \
    --max-macros 170 \
    --simulations 56 \
    --max-depth 52 \
    --c-puct "${c_puct}" \
    --max-state-revisits 0 \
    --selected-policy-target \
    --hp-aware-success-value \
    --seed "${seed}" \
    --device cuda \
    --d-model 96 \
    --heads 4 \
    --layers 3 \
    --batch-size 64 \
    --train-steps-per-episode 14 \
    --heuristic-prior-mix "${mix}" \
    --heuristic-temperature 0.70 \
    --policy-temperature 0.88 \
    --value-target-mode mixed \
    --mixed-final-weight 0.58 \
    --final-action-value-weight 0.82 \
    --final-action-prior-weight 0.50 \
    --root-dirichlet-alpha 0.30 \
    --root-exploration-fraction "${root_noise}" \
    --action-temperature "${action_temp}" \
    --action-top-k 9 \
    --policy-value-weighted \
    --save-every 10 \
    > "${RUN_DIR}/logs/${tag}.log" 2>&1 || true
  validate_route "${RUN_DIR}/selfplay/${tag}/best_route.jsonl" "${tag}"
  echo "${tag}_done" > "${RUN_DIR}/logs/${tag}.done"
}

go_variant() {
  local seed="$1"
  local tag="$2"
  local mode="$3"
  local min_hp="$4"
  local iterations="$5"
  local rollout="$6"
  "${PYTHON_BIN}" scripts/run_go_explore_experiment.py \
    --out-dir "${RUN_DIR}/search/${tag}" \
    --target-stage shield \
    --mode "${mode}" \
    --relaxed-min-hp "${min_hp}" \
    --iterations "${iterations}" \
    --rollout-steps "${rollout}" \
    --archive-top-k 80 \
    --candidate-top-k 14 \
    --temperature 1.20 \
    --novelty-bonus 520 \
    --revisit-penalty 0.015 \
    --trace-limit 2500 \
    --seed "${seed}" \
    > "${RUN_DIR}/logs/${tag}.log" 2>&1 || true
  validate_route "${RUN_DIR}/search/${tag}/best_route.jsonl" "${tag}"
  echo "${tag}_done" > "${RUN_DIR}/logs/${tag}.done"
}

planner_variant() {
  local seed="$1"
  local tag="$2"
  local mode="$3"
  local min_hp="$4"
  local expansions="$5"
  "${PYTHON_BIN}" scripts/run_resource_planner_experiment.py \
    --out-dir "${RUN_DIR}/search/${tag}" \
    --target-stage shield \
    --mode "${mode}" \
    --relaxed-min-hp "${min_hp}" \
    --max-expansions "${expansions}" \
    --archive-top-k 96 \
    --trace-limit 2500 \
    --seed "${seed}" \
    > "${RUN_DIR}/logs/${tag}.log" 2>&1 || true
  validate_route "${RUN_DIR}/search/${tag}/best_route.jsonl" "${tag}"
  echo "${tag}_done" > "${RUN_DIR}/logs/${tag}.done"
}

alpha_variant 0 20267201 alpha_shield_m030_t060_n030 0.30 1.65 0.60 0.30 &
alpha_variant 0 20267202 alpha_shield_m045_t080_n040 0.45 1.85 0.80 0.40 &
alpha_variant 1 20267203 alpha_shield_m060_t075_n035 0.60 1.70 0.75 0.35 &
alpha_variant 1 20267204 alpha_shield_m075_t095_n045 0.75 1.95 0.95 0.45 &
alpha_variant 2 20267205 alpha_shield_m040_t110_n050 0.40 2.10 1.10 0.50 &
alpha_variant 2 20267206 alpha_shield_m065_t070_n030 0.65 1.55 0.70 0.30 &
alpha_variant 3 20267207 alpha_shield_m050_t090_n040 0.50 1.80 0.90 0.40 &
alpha_variant 3 20267208 alpha_shield_m085_t120_n055 0.85 2.20 1.20 0.55 &

go_variant 20267301 go_strict_s1 strict -1800 16000 24 &
go_variant 20267302 go_strict_s2 strict -1800 16000 28 &
go_variant 20267303 go_relaxed_s1 relaxed -1800 22000 28 &
go_variant 20267304 go_relaxed_s2 relaxed -2400 22000 32 &
planner_variant 20267401 planner_relaxed_s1 relaxed -1800 80000 &
planner_variant 20267402 planner_relaxed_s2 relaxed -2400 80000 &

wait

{
  echo "# no-hp403 Alpha broad optimized report"
  echo
  echo "- run_dir: ${RUN_DIR}"
  echo "- hp403_usage: none"
  echo "- optimization: no state-log cloning, cached map signatures, parent-pointer BFS paths"
  echo
  echo "## Summaries"
  for f in "${RUN_DIR}"/search/*/summary.json "${RUN_DIR}"/selfplay/*/episodes.jsonl; do
    [ -e "${f}" ] || continue
    echo "### ${f}"
    if [[ "${f}" == *.json ]]; then
      cat "${f}"
    else
      tail -8 "${f}"
    fi
    echo
  done
} > "${RUN_DIR}/no_hp403_alpha_broad_opt_report.md"

date -Is | tee "${RUN_DIR}/finished_at.txt"
