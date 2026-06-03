#!/usr/bin/env bash
set -euo pipefail

if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /opt/conda/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-humanoid}" || true
fi

export MOTA_ENABLE_STAGE_CACHE="${MOTA_ENABLE_STAGE_CACHE:-1}"
export MOTA_SIM_CACHE_LIMIT="${MOTA_SIM_CACHE_LIMIT:-200000}"
export MOTA_GRAPH_CACHE_LIMIT="${MOTA_GRAPH_CACHE_LIMIT:-50000}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTHONPATH="${PYTHONPATH:-src}"
export PYTHONUNBUFFERED=1

RUN_ID="${RUN_ID:-4090_no_hp403_alpha_goexplore_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-artifacts/runs/${RUN_ID}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_RUN="${BASE_RUN:-artifacts/runs/4090_pure_alpha_zero_no_hp403_v2b_20260528}"
SWORD_CKPT="${SWORD_CKPT:-${BASE_RUN}/selfplay_from_planner_sword/final_model.pt}"
SWORD_ROUTE="${SWORD_ROUTE:-${BASE_RUN}/planner_sword/best_route.jsonl}"

mkdir -p "${RUN_DIR}"/{env,logs,validation}
echo "${RUN_DIR}" > artifacts/runs/latest_remote_no_hp403_alpha_goexplore_run.txt

{
  echo "pwd=$(pwd)"
  echo "run_dir=${RUN_DIR}"
  echo "base_run=${BASE_RUN}"
  echo "sword_ckpt=${SWORD_CKPT}"
  echo "sword_route=${SWORD_ROUTE}"
  echo "python=$(${PYTHON_BIN} --version 2>&1)"
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
  "${PYTHON_BIN}" - <<'PY'
import json, sys
payload = {"executable": sys.executable, "version": sys.version}
try:
    import torch
    payload.update({
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    })
except Exception as exc:
    payload["torch_error"] = repr(exc)
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
} | tee "${RUN_DIR}/env/env_check.txt"

if [ ! -s "${SWORD_CKPT}" ]; then
  echo "[no-hp403-alpha] missing sword checkpoint: ${SWORD_CKPT}" >&2
  exit 1
fi
if [ ! -s "${SWORD_ROUTE}" ]; then
  echo "[no-hp403-alpha] missing sword route: ${SWORD_ROUTE}" >&2
  exit 1
fi

solved_json() {
  local summary="$1"
  "${PYTHON_BIN}" - "$summary" <<'PY'
import json, sys
path = sys.argv[1]
try:
    data = json.load(open(path, encoding="utf8"))
except FileNotFoundError:
    sys.exit(1)
sys.exit(0 if bool(data.get("solved") or data.get("target_success")) else 1)
PY
}

validate_route() {
  local route="$1"
  local tag="$2"
  [ -s "${route}" ] || return 0
  "${PYTHON_BIN}" scripts/replay_route.py --route "${route}" > "${RUN_DIR}/validation/${tag}_replay.json" || true
  "${PYTHON_BIN}" scripts/validate_route_constraints.py "${route}" > "${RUN_DIR}/validation/${tag}_constraints.json" || true
}

train_and_eval_from_route() {
  local gpu="$1"
  local route="$2"
  local tag="$3"
  local extra_train_args="${4:-}"
  [ -s "${route}" ] || return 0
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" scripts/train_alpha_from_route.py \
    --route "${route}" \
    --out-dir "${RUN_DIR}/az_from_${tag}" \
    --target-stage shield \
    --device cuda \
    --d-model 96 \
    --heads 4 \
    --layers 3 \
    --batch-size 64 \
    --train-steps 1800 \
    --log-every 100 \
    ${extra_train_args} \
    > "${RUN_DIR}/logs/az_from_${tag}.log" 2>&1

  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" scripts/run_az_mcts_stage.py \
    --out-dir "${RUN_DIR}/eval_${tag}_strict_mcts" \
    --target-stage shield \
    --checkpoint "${RUN_DIR}/az_from_${tag}/best_model.pt" \
    --device cuda \
    --d-model 96 \
    --heads 4 \
    --layers 3 \
    --max-macros 155 \
    --simulations 64 \
    --max-depth 50 \
    --c-puct 1.55 \
    --max-state-revisits 0 \
    --policy-temperature 0.75 \
    --heuristic-prior-mix 0.25 \
    --heuristic-temperature 0.70 \
    --final-action-value-weight 0.75 \
    --final-action-prior-weight 0.45 \
    > "${RUN_DIR}/logs/eval_${tag}_strict_mcts.log" 2>&1 || true

  validate_route "${RUN_DIR}/eval_${tag}_strict_mcts/shield_az_route.jsonl" "eval_${tag}_strict_mcts"
}

run_go_strict_chain() {
  local gpu="$1"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" scripts/run_go_explore_experiment.py \
    --out-dir "${RUN_DIR}/go_shield_strict" \
    --target-stage shield \
    --mode strict \
    --iterations 18000 \
    --rollout-steps 24 \
    --archive-top-k 48 \
    --candidate-top-k 10 \
    --temperature 0.90 \
    --novelty-bonus 320 \
    --revisit-penalty 0.04 \
    --trace-limit 4000 \
    --seed 20267101 \
    > "${RUN_DIR}/logs/go_shield_strict.log" 2>&1 || true
  validate_route "${RUN_DIR}/go_shield_strict/best_route.jsonl" "go_shield_strict"
  if solved_json "${RUN_DIR}/go_shield_strict/summary.json"; then
    train_and_eval_from_route "${gpu}" "${RUN_DIR}/go_shield_strict/best_route.jsonl" "go_shield_strict"
  fi
  echo strict_chain_done > "${RUN_DIR}/logs/go_shield_strict.done"
}

run_go_relaxed_chain() {
  local gpu="$1"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" scripts/run_go_explore_experiment.py \
    --out-dir "${RUN_DIR}/go_shield_relaxed" \
    --target-stage shield \
    --mode relaxed \
    --relaxed-min-hp -1800 \
    --iterations 24000 \
    --rollout-steps 28 \
    --archive-top-k 64 \
    --candidate-top-k 12 \
    --temperature 1.10 \
    --novelty-bonus 420 \
    --revisit-penalty 0.025 \
    --trace-limit 5000 \
    --seed 20267102 \
    > "${RUN_DIR}/logs/go_shield_relaxed.log" 2>&1 || true
  validate_route "${RUN_DIR}/go_shield_relaxed/best_route.jsonl" "go_shield_relaxed"
  if solved_json "${RUN_DIR}/go_shield_relaxed/summary.json"; then
    train_and_eval_from_route "${gpu}" "${RUN_DIR}/go_shield_relaxed/best_route.jsonl" "go_shield_relaxed" "--allow-negative-hp --relaxed-min-hp -1800"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" scripts/train_alpha_mota_stage.py \
      --out-dir "${RUN_DIR}/selfplay_from_go_relaxed_strict" \
      --target-stage shield \
      --init-checkpoint "${RUN_DIR}/az_from_go_shield_relaxed/best_model.pt" \
      --episodes 45 \
      --max-macros 160 \
      --simulations 72 \
      --max-depth 52 \
      --c-puct 1.65 \
      --max-state-revisits 0 \
      --selected-policy-target \
      --hp-aware-success-value \
      --device cuda \
      --d-model 96 \
      --heads 4 \
      --layers 3 \
      --batch-size 64 \
      --train-steps-per-episode 16 \
      --heuristic-prior-mix 0.32 \
      --heuristic-temperature 0.72 \
      --policy-temperature 0.82 \
      --value-target-mode mixed \
      --mixed-final-weight 0.56 \
      --final-action-value-weight 0.75 \
      --final-action-prior-weight 0.45 \
      --root-dirichlet-alpha 0.25 \
      --root-exploration-fraction 0.30 \
      --action-temperature 0.58 \
      --action-top-k 8 \
      --policy-value-weighted \
      --save-every 10 \
      > "${RUN_DIR}/logs/selfplay_from_go_relaxed_strict.log" 2>&1 || true
    validate_route "${RUN_DIR}/selfplay_from_go_relaxed_strict/best_route.jsonl" "selfplay_from_go_relaxed_strict"
  fi
  echo relaxed_chain_done > "${RUN_DIR}/logs/go_shield_relaxed.done"
}

run_planner_relaxed_chain() {
  local gpu="$1"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" scripts/run_resource_planner_experiment.py \
    --out-dir "${RUN_DIR}/planner_shield_relaxed" \
    --target-stage shield \
    --mode relaxed \
    --relaxed-min-hp -1800 \
    --max-expansions 220000 \
    --archive-top-k 80 \
    --trace-limit 4000 \
    --seed 20267103 \
    > "${RUN_DIR}/logs/planner_shield_relaxed.log" 2>&1 || true
  validate_route "${RUN_DIR}/planner_shield_relaxed/best_route.jsonl" "planner_shield_relaxed"
  if solved_json "${RUN_DIR}/planner_shield_relaxed/summary.json"; then
    train_and_eval_from_route "${gpu}" "${RUN_DIR}/planner_shield_relaxed/best_route.jsonl" "planner_shield_relaxed" "--allow-negative-hp --relaxed-min-hp -1800"
  fi
  echo planner_relaxed_done > "${RUN_DIR}/logs/planner_shield_relaxed.done"
}

run_alpha_sword_init_strict() {
  local gpu="$1"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" scripts/train_alpha_mota_stage.py \
    --out-dir "${RUN_DIR}/shield_from_sword_ckpt_strict_v2" \
    --target-stage shield \
    --init-checkpoint "${SWORD_CKPT}" \
    --start-route "${SWORD_ROUTE}" \
    --start-route-stop-stage sword \
    --episodes 70 \
    --max-macros 165 \
    --simulations 80 \
    --max-depth 52 \
    --c-puct 1.75 \
    --max-state-revisits 0 \
    --selected-policy-target \
    --hp-aware-success-value \
    --device cuda \
    --d-model 96 \
    --heads 4 \
    --layers 3 \
    --batch-size 64 \
    --train-steps-per-episode 16 \
    --heuristic-prior-mix 0.40 \
    --heuristic-temperature 0.68 \
    --policy-temperature 0.90 \
    --value-target-mode mixed \
    --mixed-final-weight 0.58 \
    --final-action-value-weight 0.82 \
    --final-action-prior-weight 0.48 \
    --root-dirichlet-alpha 0.30 \
    --root-exploration-fraction 0.35 \
    --action-temperature 0.70 \
    --action-top-k 9 \
    --policy-value-weighted \
    --save-every 10 \
    > "${RUN_DIR}/logs/shield_from_sword_ckpt_strict_v2.log" 2>&1 || true
  validate_route "${RUN_DIR}/shield_from_sword_ckpt_strict_v2/best_route.jsonl" "shield_from_sword_ckpt_strict_v2"
  echo alpha_sword_init_done > "${RUN_DIR}/logs/shield_from_sword_ckpt_strict_v2.done"
}

run_go_strict_chain 0 &
run_go_relaxed_chain 1 &
run_planner_relaxed_chain 2 &
run_alpha_sword_init_strict 3 &
wait

{
  echo "# no-hp403 AlphaZero + Go-Explore status"
  echo
  echo "- run_dir: ${RUN_DIR}"
  echo "- hp403_usage: none"
  echo "- base sword checkpoint: ${SWORD_CKPT}"
  echo
  for f in "${RUN_DIR}"/*/summary.json "${RUN_DIR}"/eval_*/summary.json; do
    [ -e "${f}" ] || continue
    echo "## ${f}"
    echo '```json'
    cat "${f}"
    echo
    echo '```'
  done
  echo
  echo "## Best episode tails"
  for f in "${RUN_DIR}"/*/episodes.jsonl; do
    [ -e "${f}" ] || continue
    echo "### ${f}"
    tail -8 "${f}"
    echo
  done
} > "${RUN_DIR}/no_hp403_alpha_goexplore_report.md"

date -Is | tee "${RUN_DIR}/finished_at.txt"
