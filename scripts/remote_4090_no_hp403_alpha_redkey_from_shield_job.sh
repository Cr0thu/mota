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

RUN_ID="${RUN_ID:-4090_no_hp403_alpha_redkey_from_shield_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-artifacts/runs/${RUN_ID}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SHIELD_RUN_BASE="${SHIELD_RUN_BASE:-artifacts/runs/4090_no_hp403_alpha_broad_opt3_20260528/selfplay}"

mkdir -p "${RUN_DIR}"/{env,logs,validation}
echo "${RUN_DIR}" > artifacts/runs/latest_remote_no_hp403_alpha_redkey_from_shield_run.txt

{
  echo "pwd=$(pwd)"
  echo "run_dir=${RUN_DIR}"
  echo "hp403_usage=none"
  echo "shield_run_base=${SHIELD_RUN_BASE}"
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
} | tee "${RUN_DIR}/env/env_check.txt"

validate_route() {
  local route="$1"
  local tag="$2"
  [ -s "${route}" ] || return 0
  "${PYTHON_BIN}" scripts/replay_route.py --route "${route}" > "${RUN_DIR}/validation/${tag}_replay.json" || true
  "${PYTHON_BIN}" scripts/validate_route_constraints.py "${route}" > "${RUN_DIR}/validation/${tag}_constraints.json" || true
}

run_variant() {
  local gpu="$1"
  local seed="$2"
  local source_tag="$3"
  local tag="$4"
  local mix="$5"
  local c_puct="$6"
  local action_temp="$7"
  local source_dir="${SHIELD_RUN_BASE}/${source_tag}"
  local route="${source_dir}/best_route.jsonl"
  local ckpt="${source_dir}/best_model.pt"
  if [ ! -s "${route}" ] || [ ! -s "${ckpt}" ]; then
    echo "missing shield source for ${tag}: ${source_dir}" >&2
    return 0
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" scripts/train_alpha_mota_stage.py \
    --out-dir "${RUN_DIR}/${tag}" \
    --target-stage red_key \
    --init-checkpoint "${ckpt}" \
    --start-route "${route}" \
    --start-route-stop-stage shield \
    --episodes 50 \
    --max-macros 190 \
    --simulations 64 \
    --max-depth 58 \
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
    --heuristic-temperature 0.72 \
    --policy-temperature 0.88 \
    --value-target-mode mixed \
    --mixed-final-weight 0.58 \
    --final-action-value-weight 0.82 \
    --final-action-prior-weight 0.50 \
    --root-dirichlet-alpha 0.30 \
    --root-exploration-fraction 0.38 \
    --action-temperature "${action_temp}" \
    --action-top-k 10 \
    --policy-value-weighted \
    --save-every 10 \
    > "${RUN_DIR}/logs/${tag}.log" 2>&1 || true
  validate_route "${RUN_DIR}/${tag}/best_route.jsonl" "${tag}"
  echo "${tag}_done" > "${RUN_DIR}/logs/${tag}.done"
}

run_variant 0 20267501 alpha_shield_m085_t120_n055 redkey_from_m085_mix045 0.45 1.75 0.75 &
run_variant 1 20267502 alpha_shield_m085_t120_n055 redkey_from_m085_mix070 0.70 2.05 1.00 &
run_variant 2 20267503 alpha_shield_m065_t070_n030 redkey_from_m065_mix045 0.45 1.75 0.75 &
run_variant 3 20267504 alpha_shield_m065_t070_n030 redkey_from_m065_mix080 0.80 2.15 1.10 &
wait

{
  echo "# no-hp403 red_key from shield report"
  echo
  echo "- run_dir: ${RUN_DIR}"
  echo "- hp403_usage: none"
  for f in "${RUN_DIR}"/*/episodes.jsonl; do
    [ -e "${f}" ] || continue
    echo "## ${f}"
    tail -8 "${f}"
    echo
  done
} > "${RUN_DIR}/no_hp403_redkey_from_shield_report.md"

date -Is | tee "${RUN_DIR}/finished_at.txt"
