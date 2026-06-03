#!/usr/bin/env bash
set -euo pipefail

if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /opt/conda/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-humanoid}" || true
fi

export PYTHONPATH="${PYTHONPATH:-src}"
export PYTHONUNBUFFERED=1
export MOTA_ENABLE_STAGE_CACHE="${MOTA_ENABLE_STAGE_CACHE:-1}"
export MOTA_SIM_CACHE_LIMIT="${MOTA_SIM_CACHE_LIMIT:-200000}"
export MOTA_GRAPH_CACHE_LIMIT="${MOTA_GRAPH_CACHE_LIMIT:-50000}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

TARGET_STAGE="${TARGET_STAGE:?TARGET_STAGE is required}"
STOP_STAGE="${STOP_STAGE:?STOP_STAGE is required}"
BASE_ROUTE="${BASE_ROUTE:?BASE_ROUTE is required}"
BASE_CKPT="${BASE_CKPT:?BASE_CKPT is required}"
RUN_ID="${RUN_ID:-4090_no_hp403_${TARGET_STAGE}_focus_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-artifacts/runs/${RUN_ID}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EPISODES="${EPISODES:-24}"
MAX_MACROS="${MAX_MACROS:-170}"
MAX_DEPTH="${MAX_DEPTH:-56}"
SIM_A="${SIM_A:-44}"
SIM_B="${SIM_B:-52}"
export RUN_DIR TARGET_STAGE

if [[ "${BASE_ROUTE}" == *hp403* && "${BASE_ROUTE}" != *no_hp403* ]]; then
  echo "[stage-focus] refusing hp403 route ${BASE_ROUTE}" >&2
  exit 2
fi

mkdir -p "${RUN_DIR}"/{env,logs,validation,alpha}
echo "${RUN_DIR}" > "artifacts/runs/latest_remote_no_hp403_${TARGET_STAGE}_focus_run.txt"

{
  echo "pwd=$(pwd)"
  echo "run_dir=${RUN_DIR}"
  echo "hp403_usage=none"
  echo "target_stage=${TARGET_STAGE}"
  echo "stop_stage=${STOP_STAGE}"
  echo "base_route=${BASE_ROUTE}"
  echo "base_ckpt=${BASE_CKPT}"
  echo "episodes=${EPISODES}"
  echo "max_macros=${MAX_MACROS}"
  echo "python=$(${PYTHON_BIN} --version 2>&1)"
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
} | tee "${RUN_DIR}/env/env_check.txt"

stage_check() {
  local route="$1"
  local out="$2"
  "${PYTHON_BIN}" - "$route" "${TARGET_STAGE}" "$out" <<'PY' || true
import json
import sys
from pathlib import Path

from mota_env import MotaSimulator, load_game_data
from mota_env.rewards import boss_route_margin, remaining_stage_gem_targets, stage_complete
from mota_solver.search import state_summary

route = Path(sys.argv[1])
target = sys.argv[2]
out = Path(sys.argv[3])
sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
state = sim.reset()
ok = True
message = ""
for index, line in enumerate(route.read_text(encoding="utf8").splitlines()):
    if not line.strip():
        continue
    transition = sim.apply_macro_action(state, json.loads(line)["action"])
    if not transition.ok:
        ok = False
        message = f"failed at {index}: {transition.message}"
        break
payload = {
    "route": str(route),
    "target_stage": target,
    "strict_replay_ok": ok,
    "target_success": bool(ok and stage_complete(sim, state, target) and state.hp > 0),
    "boss_success": bool(state.flags.get("10f战胜骷髅队长")),
    "boss_margin": boss_route_margin(sim, state),
    "message": message,
    "final": state_summary(state),
    "stage_success": {
        stage: bool(ok and stage_complete(sim, state, stage) and state.hp > 0)
        for stage in ("mid_gems", "low_gems", "mt8_gems", "mt10_resources", "red_key", "boss_ready", "trap", "boss")
    },
    "remaining": {
        "mid_gems": remaining_stage_gem_targets(sim, state, "mid_gems"),
        "low_gems": remaining_stage_gem_targets(sim, state, "low_gems"),
        "mt8_gems": remaining_stage_gem_targets(sim, state, "mt8_gems"),
    },
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf8")
print(json.dumps(payload, ensure_ascii=False))
PY
}

run_one() {
  local gpu="$1"
  local tag="$2"
  local mix="$3"
  local temp="$4"
  local cpuct="$5"
  local sims="$6"
  local out_dir="${RUN_DIR}/alpha/${tag}"
  CUDA_VISIBLE_DEVICES="${gpu}" timeout 90m "${PYTHON_BIN}" scripts/train_alpha_mota_stage.py \
    --out-dir "${out_dir}" \
    --target-stage "${TARGET_STAGE}" \
    --init-checkpoint "${BASE_CKPT}" \
    --start-route "${BASE_ROUTE}" \
    --start-route-stop-stage "${STOP_STAGE}" \
    --episodes "${EPISODES}" \
    --max-macros "${MAX_MACROS}" \
    --simulations "${sims}" \
    --max-depth "${MAX_DEPTH}" \
    --c-puct "${cpuct}" \
    --max-state-revisits 0 \
    --selected-policy-target \
    --hp-aware-success-value \
    --seed "$((20272000 + gpu * 1000 + RANDOM % 997))" \
    --device cuda \
    --d-model 96 \
    --heads 4 \
    --layers 3 \
    --batch-size 64 \
    --train-steps-per-episode 12 \
    --heuristic-prior-mix "${mix}" \
    --heuristic-temperature 0.60 \
    --policy-temperature 0.84 \
    --value-target-mode mixed \
    --mixed-final-weight 0.58 \
    --final-action-value-weight 0.82 \
    --final-action-prior-weight 0.52 \
    --root-dirichlet-alpha 0.30 \
    --root-exploration-fraction 0.38 \
    --action-temperature "${temp}" \
    --action-top-k 9 \
    --policy-value-weighted \
    --save-every 6 \
    > "${RUN_DIR}/logs/${tag}.log" 2>&1 || true
  if [ -s "${out_dir}/best_route.jsonl" ]; then
    stage_check "${out_dir}/best_route.jsonl" "${RUN_DIR}/validation/${tag}_stage.json" >/dev/null || true
    "${PYTHON_BIN}" scripts/validate_route_constraints.py "${out_dir}/best_route.jsonl" > "${RUN_DIR}/validation/${tag}_constraints.json" || true
  fi
}

run_one 0 a 0.74 0.55 1.70 "${SIM_A}" &
run_one 1 b 0.86 0.75 1.90 "${SIM_B}" &
run_one 2 c 0.94 0.95 2.10 "${SIM_B}" &
run_one 3 d 0.68 0.65 1.80 "${SIM_A}" &
wait || true

"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path
run = Path(os.environ["RUN_DIR"])
target_stage = os.environ["TARGET_STAGE"]
rows = []
for path in sorted((run / "validation").glob("*_stage.json")):
    rows.append(json.loads(path.read_text(encoding="utf8")))
(run / "summary.json").write_text(json.dumps({"run_dir": str(run), "hp403_usage": "none", "target_stage": target_stage, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf8")
lines = ["# no-hp403 stage focus", "", f"Run dir: `{run}`", f"Target: `{target_stage}`", ""]
for row in rows:
    final = row.get("final", {})
    lines.append(
        f"- {Path(row['route']).parent.name}: success={row.get('target_success')} "
        f"hp={final.get('hp')} atk={final.get('atk')} def={final.get('defense') or final.get('def')} "
        f"money={final.get('money')} keys={final.get('keys')} floor={final.get('floor_id') or final.get('floor')} "
        f"boss_margin={row.get('boss_margin')} remaining={row.get('remaining')}"
    )
(run / f"{target_stage}_focus_report.md").write_text("\n".join(lines) + "\n", encoding="utf8")
print(json.dumps({"summary": str(run / "summary.json"), "report": str(run / f"{target_stage}_focus_report.md")}, ensure_ascii=False))
PY

date -Is | tee "${RUN_DIR}/finished_at.txt"
