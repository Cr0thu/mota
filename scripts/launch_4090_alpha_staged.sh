#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-4090_alpha_staged_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="artifacts/runs/${RUN_ID}"
mkdir -p "${RUN_DIR}/logs"
echo "${RUN_DIR}" > artifacts/runs/latest_remote_alpha_staged_run.txt

source /opt/conda/etc/profile.d/conda.sh >/dev/null 2>&1 || true
conda activate humanoid >/dev/null 2>&1 || true

run_job() {
  local gpu="$1"
  local seed="$2"
  local target="$3"
  local tag="$4"
  local sims="$5"
  local macros="$6"
  local episodes="$7"
  local extra_args="${8:-}"

  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
  PYTHONPATH=src python scripts/train_alpha_mota_stage.py \
    --out-dir "${RUN_DIR}/${tag}" \
    --target-stage "${target}" \
    --episodes "${episodes}" \
    --max-macros "${macros}" \
    --simulations "${sims}" \
    --max-depth 16 \
    --c-puct 1.15 \
    --max-state-revisits 1 \
    --selected-policy-target \
    --seed "${seed}" \
    --device cuda \
    --d-model 64 \
    --heads 4 \
    --layers 1 \
    --dropout 0.03 \
    --policy-temperature 1.0 \
    --heuristic-prior-mix 1.0 \
    --heuristic-temperature 0.42 \
    --lr 2e-4 \
    --batch-size 64 \
    --replay-size 6000 \
    --train-steps-per-episode 24 \
    --value-loss-coef 0.5 \
    --save-every 10 \
    ${extra_args} \
    > "${RUN_DIR}/logs/${tag}.log" 2>&1 &
}

run_job 0 20263001 sword sword_s1_honly 1 55 120
run_job 1 20263002 sword sword_s2_honly 2 60 100
run_job 2 20263003 shield shield_s1_relaxed 1 170 80 "--allow-negative-hp --relaxed-min-hp -2500"
run_job 3 20263004 shield shield_s2_relaxed 2 180 70 "--allow-negative-hp --relaxed-min-hp -2500"

wait
