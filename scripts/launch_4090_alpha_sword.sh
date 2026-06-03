#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-4090_alpha_sword_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="artifacts/runs/${RUN_ID}"
mkdir -p "${RUN_DIR}/logs"
echo "${RUN_DIR}" > artifacts/runs/latest_remote_alpha_sword_run.txt

source /opt/conda/etc/profile.d/conda.sh >/dev/null 2>&1 || true
conda activate humanoid >/dev/null 2>&1 || true

run_job() {
  local gpu="$1"
  local seed="$2"
  local tag="$3"
  local sims="$4"
  local mix="$5"
  local cpuct="$6"

  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
  PYTHONPATH=src python scripts/train_alpha_mota_stage.py \
    --out-dir "${RUN_DIR}/${tag}" \
    --target-stage sword \
    --episodes 80 \
    --max-macros 70 \
    --simulations "${sims}" \
    --max-depth 12 \
    --c-puct "${cpuct}" \
    --max-state-revisits 1 \
    --selected-policy-target \
    --seed "${seed}" \
    --device cuda \
    --d-model 64 \
    --heads 4 \
    --layers 1 \
    --dropout 0.03 \
    --policy-temperature 1.0 \
    --heuristic-prior-mix "${mix}" \
    --heuristic-temperature 0.7 \
    --lr 2e-4 \
    --batch-size 64 \
    --replay-size 4000 \
    --train-steps-per-episode 24 \
    --value-loss-coef 0.5 \
    --save-every 10 \
    > "${RUN_DIR}/logs/${tag}.log" 2>&1 &
}

run_job 0 20262901 sword_s1_honly 1 1.00 1.0
run_job 1 20262902 sword_s2_honly 2 1.00 1.1
run_job 2 20262903 sword_s4_honly 4 1.00 1.2
run_job 3 20262904 sword_s8_honly 8 1.00 1.3

wait
