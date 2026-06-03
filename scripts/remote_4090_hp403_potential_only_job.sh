#!/usr/bin/env bash
set -euo pipefail

# Run this script on the 4090 pod from /root/mota/mota.
# It uses hp403 only to learn a potential Phi(s).  The route is not used as a
# behavior-cloning label or start-route for the AlphaZero-style search.

if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /opt/conda/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-humanoid}" || true
fi

export PYTHONPATH="${PYTHONPATH:-src}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ID="${RUN_ID:-4090_hp403_potential_only_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-artifacts/runs/${RUN_ID}}"
HP403_ROUTE="${HP403_ROUTE:-artifacts/manual_exploration_20260524/manual_success_no_shop_true_10f_trap_optimized_hp403.jsonl}"
REWARD_DIR="${RUN_DIR}/reward"
REWARD_FILE="${REWARD_DIR}/hp403_learned_reward_weights.json"

LEARN_EPOCHS="${LEARN_EPOCHS:-1800}"
LEARN_NEGATIVES="${LEARN_NEGATIVES:-24}"

RUN_ALPHA="${RUN_ALPHA:-1}"
ALPHA_TARGETS="${ALPHA_TARGETS:-shield red_key boss}"
ALPHA_EPISODES="${ALPHA_EPISODES:-20}"
ALPHA_MACROS="${ALPHA_MACROS:-190}"
ALPHA_SIMS="${ALPHA_SIMS:-80}"
ALPHA_DEPTH="${ALPHA_DEPTH:-80}"
ALPHA_TRAIN_STEPS="${ALPHA_TRAIN_STEPS:-24}"
ALPHA_BATCH_SIZE="${ALPHA_BATCH_SIZE:-64}"
ALPHA_D_MODEL="${ALPHA_D_MODEL:-128}"
ALPHA_HEADS="${ALPHA_HEADS:-4}"
ALPHA_LAYERS="${ALPHA_LAYERS:-3}"
ALPHA_EDGE_SCALE="${ALPHA_EDGE_SCALE:-2000}"
ALPHA_EDGE_CLIP="${ALPHA_EDGE_CLIP:-1.0}"
ALPHA_REWARD_STAGE_MODE="${ALPHA_REWARD_STAGE_MODE:-current}"

RUN_ARCHIVE="${RUN_ARCHIVE:-1}"
ARCHIVE_ITERATIONS="${ARCHIVE_ITERATIONS:-200000}"
ARCHIVE_SIMS="${ARCHIVE_SIMS:-8}"
ARCHIVE_DEPTH="${ARCHIVE_DEPTH:-20}"
ARCHIVE_TARGETS="${ARCHIVE_TARGETS:-red_key boss}"
ARCHIVE_HP_MODES="${ARCHIVE_HP_MODES:-strict}"
ARCHIVE_SCORE_SCHEME="${ARCHIVE_SCORE_SCHEME:-strict_stage}"
ARCHIVE_VISIT_PENALTY_SCALE="${ARCHIVE_VISIT_PENALTY_SCALE:-5000}"
ARCHIVE_FAILED_ROLLOUT_PENALTY="${ARCHIVE_FAILED_ROLLOUT_PENALTY:-250000}"
ARCHIVE_STALE_ROLLOUT_PENALTY="${ARCHIVE_STALE_ROLLOUT_PENALTY:-75000}"
ARCHIVE_STALE_MIN_SCORE_GAIN="${ARCHIVE_STALE_MIN_SCORE_GAIN:-5000}"
ARCHIVE_USE_STAGE_ACTION_FILTER="${ARCHIVE_USE_STAGE_ACTION_FILTER:-1}"

mkdir -p "${RUN_DIR}"/{env,logs,alpha,archive,validation} "${REWARD_DIR}"
echo "${RUN_DIR}" > artifacts/runs/latest_remote_hp403_potential_only_run.txt

{
  echo "pwd=$(pwd)"
  echo "run_id=${RUN_ID}"
  echo "run_dir=${RUN_DIR}"
  echo "hp403_usage=potential_only"
  echo "hp403_route=${HP403_ROUTE}"
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

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  "${PYTHON_BIN}" scripts/learn_reward_from_hp403.py \
    --route "${HP403_ROUTE}" \
    --out-dir "${REWARD_DIR}" \
    --epochs "${LEARN_EPOCHS}" \
    --negatives-per-state "${LEARN_NEGATIVES}" \
    --device cuda \
    > "${RUN_DIR}/logs/reward_learning.log" 2>&1

cp "${REWARD_FILE}" "${RUN_DIR}/hp403_learned_reward_weights.json"

run_alpha() {
  local gpu="$1"
  local target="$2"
  local tag="alpha_${target}_potential"
  CUDA_VISIBLE_DEVICES="${gpu}" MOTA_FAST_GRAPH_STATE=1 PYTHONUNBUFFERED=1 \
    "${PYTHON_BIN}" scripts/train_alpha_mota_stage.py \
      --protocol auto_reward_search \
      --out-dir "${RUN_DIR}/alpha/${tag}" \
      --target-stage "${target}" \
      --episodes "${ALPHA_EPISODES}" \
      --max-macros "${ALPHA_MACROS}" \
      --simulations "${ALPHA_SIMS}" \
      --max-depth "${ALPHA_DEPTH}" \
      --c-puct 1.7 \
      --max-state-revisits 1 \
      --selected-policy-target \
      --hp-aware-success-value \
      --device cuda \
      --d-model "${ALPHA_D_MODEL}" \
      --heads "${ALPHA_HEADS}" \
      --layers "${ALPHA_LAYERS}" \
      --batch-size "${ALPHA_BATCH_SIZE}" \
      --train-steps-per-episode "${ALPHA_TRAIN_STEPS}" \
      --reward-weights-file "${REWARD_FILE}" \
      --reward-value-scale 25000 \
      --reward-value-stage-mode "${ALPHA_REWARD_STAGE_MODE}" \
      --use-learned-potential-edge-reward \
      --mcts-edge-reward-scale "${ALPHA_EDGE_SCALE}" \
      --mcts-edge-reward-clip "${ALPHA_EDGE_CLIP}" \
      --value-target-mode mixed \
      --mixed-final-weight 0.40 \
      --value-return-gamma 0.99 \
      --policy-value-weighted \
      --root-dirichlet-alpha 0.35 \
      --root-exploration-fraction 0.20 \
      --save-every 5 \
      > "${RUN_DIR}/logs/${tag}.log" 2>&1 &
}

run_archive() {
  local target="$1"
  local hp_mode="$2"
  local tag="archive_${target}_${hp_mode}_potential"
  local hp_args=()
  if [ "${hp_mode}" = "relaxed" ]; then
    hp_args=(--allow-negative-hp --relaxed-min-hp -1600)
  fi
  local filter_args=()
  if [ "${ARCHIVE_USE_STAGE_ACTION_FILTER}" = "1" ]; then
    filter_args=(--use-stage-action-filter)
  fi
  MOTA_FAST_GRAPH_STATE=1 PYTHONUNBUFFERED=1 \
    "${PYTHON_BIN}" scripts/run_archive_mcts_experiment.py \
      --protocol potential_shaping \
      --out-dir "${RUN_DIR}/archive/${tag}" \
      --target-stage "${target}" \
      --iterations "${ARCHIVE_ITERATIONS}" \
      --rollout-steps 10 \
      --archive-top-k 16 \
      --archive-temperature 1.1 \
      --archive-sample-limit 128 \
      --score-scheme "${ARCHIVE_SCORE_SCHEME}" \
      --visit-penalty-scale "${ARCHIVE_VISIT_PENALTY_SCALE}" \
      --failed-rollout-penalty "${ARCHIVE_FAILED_ROLLOUT_PENALTY}" \
      --stale-rollout-penalty "${ARCHIVE_STALE_ROLLOUT_PENALTY}" \
      --stale-rollout-min-score-gain "${ARCHIVE_STALE_MIN_SCORE_GAIN}" \
      --simulations "${ARCHIVE_SIMS}" \
      --max-depth "${ARCHIVE_DEPTH}" \
      --c-puct 1.8 \
      --child-temperature 0.20 \
      --reward-weights-file "${REWARD_FILE}" \
      --reward-stage-mode current \
      --reward-value-scale 25000 \
      --edge-reward-scale 2000 \
      --edge-reward-clip 1.0 \
      --root-dirichlet-alpha 0.35 \
      --root-exploration-fraction 0.20 \
      --progress-interval 5000 \
      "${filter_args[@]}" \
      "${hp_args[@]}" \
      > "${RUN_DIR}/logs/${tag}.log" 2>&1 &
}

if [ "${RUN_ALPHA}" = "1" ]; then
  gpu=0
  for target in ${ALPHA_TARGETS}; do
    run_alpha "${gpu}" "${target}"
    gpu=$(( (gpu + 1) % 4 ))
  done
fi

if [ "${RUN_ARCHIVE}" = "1" ]; then
  for target in ${ARCHIVE_TARGETS}; do
    for hp_mode in ${ARCHIVE_HP_MODES}; do
      run_archive "${target}" "${hp_mode}"
    done
  done
fi

wait || true

for route in "${RUN_DIR}"/alpha/*/best_route.jsonl "${RUN_DIR}"/archive/*/best_route.jsonl; do
  [ -s "${route}" ] || continue
  safe="$(echo "${route#${RUN_DIR}/}" | tr '/.' '__')"
  "${PYTHON_BIN}" scripts/replay_route.py --route "${route}" > "${RUN_DIR}/validation/${safe}_replay.json" || true
  "${PYTHON_BIN}" scripts/validate_route_constraints.py "${route}" > "${RUN_DIR}/validation/${safe}_constraints.json" || true
done

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
run = Path("${RUN_DIR}")
rows = []
for p in sorted(run.glob("alpha/*/episodes.jsonl")):
    lines = [json.loads(line) for line in p.read_text(encoding="utf8").splitlines() if line.strip()]
    rows.append({"kind": "alpha", "path": str(p), "last": lines[-1] if lines else None})
for p in sorted(run.glob("archive/*/summary.json")):
    try:
        rows.append({"kind": "archive", "path": str(p), "data": json.loads(p.read_text(encoding="utf8"))})
    except Exception as exc:
        rows.append({"kind": "archive", "path": str(p), "error": repr(exc)})
summary = {
    "run_dir": str(run),
    "hp403_usage": "potential_only",
    "negative_hp_exploration": "${ARCHIVE_HP_MODES}" != "strict",
    "reward_file": "${REWARD_FILE}",
    "rows": rows,
}
(run / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8")
lines = ["# hp403 Potential-only Training", "", f"Run dir: `{run}`", "", "hp403 is used only to fit Phi(s); no BC/start-route is used.", ""]
for row in rows:
    if row["kind"] == "alpha":
        last = row.get("last") or {}
        final = last.get("final") or {}
        lines.append(f"- alpha `{row['path']}` target={last.get('target_stage')} success={last.get('target_success')} boss={last.get('boss_success')} final={final}")
    else:
        data = row.get("data", {})
        lines.append(f"- archive `{row['path']}` target={data.get('target_stage')} success={data.get('target_success')} boss={data.get('boss_success')} len={data.get('route_length')} final={data.get('best_state')}")
(run / "hp403_potential_only_report.md").write_text("\\n".join(lines) + "\\n", encoding="utf8")
print(json.dumps({"summary": str(run / "summary.json"), "report": str(run / "hp403_potential_only_report.md")}, ensure_ascii=False))
PY

date -Is | tee "${RUN_DIR}/finished_at.txt"
