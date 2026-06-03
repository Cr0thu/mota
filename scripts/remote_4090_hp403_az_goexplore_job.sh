#!/usr/bin/env bash
set -euo pipefail

if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /opt/conda/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-humanoid}" || true
fi

export PYTHONPATH="${PYTHONPATH:-src}"

RUN_ID="${RUN_ID:-4090_hp403_az_goexplore_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-artifacts/runs/${RUN_ID}}"
REWARD_RUN="${REWARD_RUN:-artifacts/runs/hp403_reward_learning_20260528}"
REWARD_FILE="${REWARD_FILE:-${REWARD_RUN}/hp403_learned_reward_weights.json}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WAIT_SECONDS="${WAIT_SECONDS:-3600}"
ALPHA_EPISODES="${ALPHA_EPISODES:-18}"
ALPHA_MACROS="${ALPHA_MACROS:-180}"
ALPHA_SIMS="${ALPHA_SIMS:-96}"
ALPHA_DEPTH="${ALPHA_DEPTH:-80}"
ALPHA_TARGETS="${ALPHA_TARGETS:-shield red_key boss}"
ALPHA_TRAIN_STEPS="${ALPHA_TRAIN_STEPS:-24}"
ALPHA_BATCH_SIZE="${ALPHA_BATCH_SIZE:-64}"
ALPHA_D_MODEL="${ALPHA_D_MODEL:-128}"
ALPHA_HEADS="${ALPHA_HEADS:-4}"
ALPHA_LAYERS="${ALPHA_LAYERS:-3}"
ALPHA_HEURISTIC_PRIOR_MIX="${ALPHA_HEURISTIC_PRIOR_MIX:-0.35}"
ALPHA_REWARD_VALUE_SCALE="${ALPHA_REWARD_VALUE_SCALE:-25000}"
ALPHA_VALUE_TARGET_MODE="${ALPHA_VALUE_TARGET_MODE:-final}"
ALPHA_MIXED_FINAL_WEIGHT="${ALPHA_MIXED_FINAL_WEIGHT:-0.35}"
ALPHA_FINAL_ACTION_VALUE_WEIGHT="${ALPHA_FINAL_ACTION_VALUE_WEIGHT:-0.0}"
ALPHA_FINAL_ACTION_PRIOR_WEIGHT="${ALPHA_FINAL_ACTION_PRIOR_WEIGHT:-0.0}"
GO_EXPANSIONS="${GO_EXPANSIONS:-120000}"
GO_TRACE_LIMIT="${GO_TRACE_LIMIT:-120000}"
GO_SPECS="${GO_SPECS:-strict:red_key relaxed:boss}"
GO_IMPL="${GO_IMPL:-planner}"
GO_ROLLOUT_STEPS="${GO_ROLLOUT_STEPS:-16}"
GO_CANDIDATE_TOP_K="${GO_CANDIDATE_TOP_K:-6}"
RUN_ALPHA="${RUN_ALPHA:-1}"
RUN_GO="${RUN_GO:-1}"

mkdir -p "${RUN_DIR}"/{env,alpha,go_explore,logs,validation}
echo "${RUN_DIR}" > artifacts/runs/latest_remote_hp403_az_goexplore_run.txt

{
  echo "pwd=$(pwd)"
  echo "python=$(${PYTHON_BIN} --version 2>&1)"
  echo "reward_file=${REWARD_FILE}"
  echo "wait_seconds=${WAIT_SECONDS}"
  echo "alpha_episodes=${ALPHA_EPISODES}"
  echo "alpha_macros=${ALPHA_MACROS}"
  echo "alpha_sims=${ALPHA_SIMS}"
  echo "alpha_depth=${ALPHA_DEPTH}"
  echo "alpha_targets=${ALPHA_TARGETS}"
  echo "alpha_train_steps=${ALPHA_TRAIN_STEPS}"
  echo "alpha_value_target_mode=${ALPHA_VALUE_TARGET_MODE}"
  echo "alpha_final_action_value_weight=${ALPHA_FINAL_ACTION_VALUE_WEIGHT}"
  echo "go_expansions=${GO_EXPANSIONS}"
  echo "go_specs=${GO_SPECS}"
  echo "go_impl=${GO_IMPL}"
  echo "go_rollout_steps=${GO_ROLLOUT_STEPS}"
  echo "run_alpha=${RUN_ALPHA}"
  echo "run_go=${RUN_GO}"
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

deadline=$(( $(date +%s) + WAIT_SECONDS ))
while [ ! -s "${REWARD_FILE}" ]; do
  if [ "$(date +%s)" -ge "${deadline}" ]; then
    echo "[hp403-az-go] timeout waiting for ${REWARD_FILE}" >&2
    exit 2
  fi
  sleep 20
done

cp "${REWARD_FILE}" "${RUN_DIR}/hp403_learned_reward_weights.json"

run_alpha() {
  local gpu="$1"
  local target="$2"
  local tag="$3"
  CUDA_VISIBLE_DEVICES="${gpu}" MOTA_FAST_GRAPH_STATE=1 PYTHONUNBUFFERED=1 \
    "${PYTHON_BIN}" scripts/train_alpha_mota_stage.py \
      --out-dir "${RUN_DIR}/alpha/${tag}" \
      --target-stage "${target}" \
      --episodes "${ALPHA_EPISODES}" \
      --max-macros "${ALPHA_MACROS}" \
      --simulations "${ALPHA_SIMS}" \
      --max-depth "${ALPHA_DEPTH}" \
      --c-puct 1.65 \
      --max-state-revisits 1 \
      --selected-policy-target \
      --hp-aware-success-value \
      --device cuda \
      --d-model "${ALPHA_D_MODEL}" \
      --heads "${ALPHA_HEADS}" \
      --layers "${ALPHA_LAYERS}" \
      --batch-size "${ALPHA_BATCH_SIZE}" \
      --train-steps-per-episode "${ALPHA_TRAIN_STEPS}" \
      --heuristic-prior-mix "${ALPHA_HEURISTIC_PRIOR_MIX}" \
      --heuristic-temperature 0.75 \
      --policy-temperature 1.0 \
      --reward-weights-file "${REWARD_FILE}" \
      --reward-value-scale "${ALPHA_REWARD_VALUE_SCALE}" \
      --reward-value-stage-mode current \
      --value-target-mode "${ALPHA_VALUE_TARGET_MODE}" \
      --mixed-final-weight "${ALPHA_MIXED_FINAL_WEIGHT}" \
      --final-action-value-weight "${ALPHA_FINAL_ACTION_VALUE_WEIGHT}" \
      --final-action-prior-weight "${ALPHA_FINAL_ACTION_PRIOR_WEIGHT}" \
      --policy-value-weighted \
      --save-every 6 \
      > "${RUN_DIR}/logs/${tag}.log" 2>&1 &
}

run_go() {
  local mode="$1"
  local target="$2"
  local tag="$3"
  if [ "${GO_IMPL}" = "archive" ]; then
    "${PYTHON_BIN}" scripts/run_go_explore_experiment.py \
      --out-dir "${RUN_DIR}/go_explore/${tag}" \
      --target-stage "${target}" \
      --mode "${mode}" \
      --iterations "${GO_EXPANSIONS}" \
      --rollout-steps "${GO_ROLLOUT_STEPS}" \
      --archive-top-k 12 \
      --seed 20260628 \
      --trace-limit "${GO_TRACE_LIMIT}" \
      --relaxed-min-hp -1200 \
      --candidate-top-k "${GO_CANDIDATE_TOP_K}" \
      --reward-weights-file "${REWARD_FILE}" \
      --reward-potential-weight 0.018 \
      --reward-stage-mode current \
      > "${RUN_DIR}/logs/${tag}.log" 2>&1 &
  else
    "${PYTHON_BIN}" scripts/run_resource_planner_experiment.py \
      --out-dir "${RUN_DIR}/go_explore/${tag}" \
      --target-stage "${target}" \
      --mode "${mode}" \
      --max-expansions "${GO_EXPANSIONS}" \
      --archive-top-k 12 \
      --seed 20260628 \
      --trace-limit "${GO_TRACE_LIMIT}" \
      --relaxed-min-hp -1200 \
      --reward-weights-file "${REWARD_FILE}" \
      --reward-potential-weight 0.018 \
      --reward-stage-mode current \
      > "${RUN_DIR}/logs/${tag}.log" 2>&1 &
  fi
}

if [ "${RUN_ALPHA}" = "1" ]; then
  gpu=0
  for target in ${ALPHA_TARGETS}; do
    run_alpha "${gpu}" "${target}" "alpha_${target}_hp403_reward"
    gpu=$(( (gpu + 1) % 4 ))
  done
fi

if [ "${RUN_GO}" = "1" ]; then
  for spec in ${GO_SPECS}; do
    mode="${spec%%:*}"
    target="${spec#*:}"
    run_go "${mode}" "${target}" "go_${mode}_${target}_hp403_reward"
  done
fi
wait || true

for route in "${RUN_DIR}"/alpha/*/best_route.jsonl "${RUN_DIR}"/go_explore/*/*/best_route.jsonl "${RUN_DIR}"/go_explore/*/best_route.jsonl; do
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
for p in sorted(run.glob("go_explore/**/summary.json")):
    try:
        rows.append({"kind": "go_explore", "path": str(p), "data": json.loads(p.read_text(encoding="utf8"))})
    except Exception as exc:
        rows.append({"kind": "go_explore", "path": str(p), "error": repr(exc)})
payload = {"run_dir": str(run), "reward_file": "${REWARD_FILE}", "rows": rows}
(run / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf8")
lines = ["# hp403 AlphaZero + Go-Explore", "", f"Run dir: `{run}`", f"Reward file: `${REWARD_FILE}`", ""]
for row in rows:
    if row["kind"] == "alpha":
        last = row.get("last") or {}
        lines.append(f"- alpha `{row['path']}` success={last.get('target_success')} boss={last.get('boss_success')} final={last.get('final')}")
    else:
        data = row.get("data", {})
        lines.append(f"- go `{row['path']}` solved={data.get('solved')} best={data.get('best_summary')} route_len={data.get('route_length')}")
(run / "hp403_az_goexplore_report.md").write_text("\\n".join(lines) + "\\n", encoding="utf8")
print(json.dumps({"summary": str(run / "summary.json"), "report": str(run / "hp403_az_goexplore_report.md")}, ensure_ascii=False))
PY

date -Is | tee "${RUN_DIR}/finished_at.txt"
