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
export MOTA_SIM_CACHE_LIMIT="${MOTA_SIM_CACHE_LIMIT:-220000}"
export MOTA_GRAPH_CACHE_LIMIT="${MOTA_GRAPH_CACHE_LIMIT:-60000}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

RUN_ID="${RUN_ID:-4090_no_hp403_mt4red_shield_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-artifacts/runs/${RUN_ID}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_RUN="${BASE_RUN:-artifacts/runs/4090_pure_alpha_zero_no_hp403_v2b_20260528}"
SWORD_CKPT="${SWORD_CKPT:-${BASE_RUN}/selfplay_from_planner_sword/final_model.pt}"
SWORD_ROUTE="${SWORD_ROUTE:-${BASE_RUN}/planner_sword/best_route.jsonl}"
MT4_EPISODES="${MT4_EPISODES:-14}"
SHIELD_EPISODES="${SHIELD_EPISODES:-28}"
TIMEOUT_MIN="${TIMEOUT_MIN:-85}"
MAX_MT4_INPUTS="${MAX_MT4_INPUTS:-4}"

mkdir -p "${RUN_DIR}"/{env,logs,mt4_redgem,shield,inputs,validation}
echo "${RUN_DIR}" > artifacts/runs/latest_remote_no_hp403_mt4red_shield_run.txt

{
  echo "pwd=$(pwd)"
  echo "run_dir=${RUN_DIR}"
  echo "hp403_usage=none"
  echo "base_run=${BASE_RUN}"
  echo "sword_ckpt=${SWORD_CKPT}"
  echo "sword_route=${SWORD_ROUTE}"
  echo "python=$(${PYTHON_BIN} --version 2>&1)"
  echo "mt4_episodes=${MT4_EPISODES}"
  echo "shield_episodes=${SHIELD_EPISODES}"
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
} | tee "${RUN_DIR}/env/env_check.txt"

if { [[ "${SWORD_ROUTE}" == *hp403* ]] && [[ "${SWORD_ROUTE}" != *no_hp403* ]]; } || \
   { [[ "${SWORD_CKPT}" == *hp403* ]] && [[ "${SWORD_CKPT}" != *no_hp403* ]]; }; then
  echo "[mt4red-shield] refusing hp403 input: ${SWORD_ROUTE} ${SWORD_CKPT}" >&2
  exit 2
fi

stage_check() {
  local route="$1"
  local target="$2"
  local out="$3"
  "${PYTHON_BIN}" - "$route" "$target" "$out" <<'PY' || true
import json
import sys
from pathlib import Path

from mota_env import MotaSimulator, load_game_data
from mota_env.rewards import boss_route_margin, stage_complete
from mota_solver.search import state_summary

route_path = Path(sys.argv[1])
target = sys.argv[2]
out_path = Path(sys.argv[3])
sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
state = sim.reset()
ok = True
message = ""
rows = [json.loads(line) for line in route_path.read_text(encoding="utf8").splitlines() if line.strip()]
for index, row in enumerate(rows):
    transition = sim.apply_macro_action(state, row.get("action") or {})
    if not transition.ok:
        ok = False
        message = f"failed at {index}: {transition.message}"
        break
summary = {
    "route": str(route_path),
    "target_stage": target,
    "strict_replay_ok": ok,
    "target_success": bool(ok and stage_complete(sim, state, target) and state.hp > 0),
    "boss_success": bool(state.flags.get("10f战胜骷髅队长")),
    "boss_margin": boss_route_margin(sim, state),
    "macro_steps": len(rows),
    "final": state_summary(state),
    "message": message,
}
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8")
print(json.dumps(summary, ensure_ascii=False))
PY
}

route_stage_success() {
  local route="$1"
  local target="$2"
  "${PYTHON_BIN}" - "$route" "$target" <<'PY'
import json
import sys
from pathlib import Path
from mota_env import MotaSimulator, load_game_data
from mota_env.rewards import stage_complete

sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
state = sim.reset()
for line in Path(sys.argv[1]).read_text(encoding="utf8").splitlines():
    if not line.strip():
        continue
    transition = sim.apply_macro_action(state, json.loads(line)["action"])
    if not transition.ok:
        sys.exit(1)
sys.exit(0 if state.hp > 0 and stage_complete(sim, state, sys.argv[2]) else 1)
PY
}

run_alpha() {
  local gpu="$1"
  local out_dir="$2"
  local target="$3"
  local route="$4"
  local stop_stage="$5"
  local ckpt="$6"
  local episodes="$7"
  local max_macros="$8"
  local sims="$9"
  local depth="${10}"
  local mix="${11}"
  local action_temp="${12}"
  local c_puct="${13}"
  mkdir -p "${out_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" timeout "${TIMEOUT_MIN}m" "${PYTHON_BIN}" scripts/train_alpha_mota_stage.py \
    --out-dir "${out_dir}" \
    --target-stage "${target}" \
    --init-checkpoint "${ckpt}" \
    --start-route "${route}" \
    --start-route-stop-stage "${stop_stage}" \
    --episodes "${episodes}" \
    --max-macros "${max_macros}" \
    --simulations "${sims}" \
    --max-depth "${depth}" \
    --c-puct "${c_puct}" \
    --max-state-revisits 0 \
    --selected-policy-target \
    --hp-aware-success-value \
    --seed "$((20269500 + gpu * 1000 + RANDOM % 997))" \
    --device cuda \
    --d-model 96 \
    --heads 4 \
    --layers 3 \
    --batch-size 64 \
    --train-steps-per-episode 10 \
    --heuristic-prior-mix "${mix}" \
    --heuristic-temperature 0.62 \
    --policy-temperature 0.88 \
    --value-target-mode mixed \
    --mixed-final-weight 0.56 \
    --final-action-value-weight 0.82 \
    --final-action-prior-weight 0.48 \
    --root-dirichlet-alpha 0.38 \
    --root-exploration-fraction 0.46 \
    --action-temperature "${action_temp}" \
    --action-top-k 10 \
    --policy-value-weighted \
    --save-every 7 \
    > "${RUN_DIR}/logs/$(basename "${out_dir}").log" 2>&1 || true
  if [ -s "${out_dir}/best_route.jsonl" ]; then
    stage_check "${out_dir}/best_route.jsonl" "${target}" "${RUN_DIR}/validation/$(basename "${out_dir}")_${target}.json" >/dev/null || true
  fi
}

run_alpha 0 "${RUN_DIR}/mt4_redgem/a" mt4_redgem "${SWORD_ROUTE}" sword "${SWORD_CKPT}" "${MT4_EPISODES}" 48 80 34 0.78 0.70 1.65 &
run_alpha 1 "${RUN_DIR}/mt4_redgem/b" mt4_redgem "${SWORD_ROUTE}" sword "${SWORD_CKPT}" "${MT4_EPISODES}" 52 96 38 0.88 0.90 1.85 &
run_alpha 2 "${RUN_DIR}/mt4_redgem/c" mt4_redgem "${SWORD_ROUTE}" sword "${SWORD_CKPT}" "${MT4_EPISODES}" 56 112 42 0.95 1.05 2.05 &
run_alpha 3 "${RUN_DIR}/mt4_redgem/d" mt4_redgem "${SWORD_ROUTE}" sword "${SWORD_CKPT}" "${MT4_EPISODES}" 54 96 42 0.65 1.15 2.20 &
wait || true

: > "${RUN_DIR}/inputs/mt4_success.tsv"
: > "${RUN_DIR}/inputs/mt4_all.tsv"
for route in "${RUN_DIR}"/mt4_redgem/*/best_route.jsonl; do
  [ -s "${route}" ] || continue
  dir="$(dirname "${route}")"
  tag="$(basename "${dir}")"
  ckpt="${dir}/best_model.pt"
  [ -s "${ckpt}" ] || continue
  printf '%s\t%s\t%s\n' "${tag}" "${route}" "${ckpt}" >> "${RUN_DIR}/inputs/mt4_all.tsv"
  if route_stage_success "${route}" mt4_redgem; then
    printf '%s\t%s\t%s\n' "${tag}" "${route}" "${ckpt}" >> "${RUN_DIR}/inputs/mt4_success.tsv"
  fi
done

if [ -s "${RUN_DIR}/inputs/mt4_success.tsv" ]; then
  cp "${RUN_DIR}/inputs/mt4_success.tsv" "${RUN_DIR}/inputs/mt4_selected.tsv"
else
  cp "${RUN_DIR}/inputs/mt4_all.tsv" "${RUN_DIR}/inputs/mt4_selected.tsv"
fi

idx=0
while IFS=$'\t' read -r tag route ckpt; do
  [ -n "${tag}" ] || continue
  [ -s "${route}" ] || continue
  [ -s "${ckpt}" ] || continue
  idx=$((idx + 1))
  if [ "${idx}" -gt "${MAX_MT4_INPUTS}" ]; then
    break
  fi
  gpu=$(( (idx - 1) % 4 ))
  mix="$(awk "BEGIN { printf \"%.2f\", 0.60 + (${idx} % 3) * 0.13 }")"
  temp="$(awk "BEGIN { printf \"%.2f\", 0.72 + (${idx} % 4) * 0.12 }")"
  # Use the complete MT4-red route as the shield prefix.  The best MT4 route
  # often collects the adjacent MT4 potion after the red gem; truncating exactly
  # at stage completion throws away that HP and weakens the shield continuation.
  run_alpha "${gpu}" "${RUN_DIR}/shield/${tag}_shield" shield "${route}" none "${ckpt}" "${SHIELD_EPISODES}" 135 72 54 "${mix}" "${temp}" 1.90 &
done < "${RUN_DIR}/inputs/mt4_selected.tsv"
wait || true

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

run = Path("${RUN_DIR}")
rows = []
for path in sorted((run / "validation").glob("*.json")):
    try:
        rows.append({"path": str(path), **json.loads(path.read_text(encoding="utf8"))})
    except Exception as exc:
        rows.append({"path": str(path), "error": repr(exc)})
payload = {
    "run_dir": str(run),
    "hp403_usage": "none",
    "sword_route": "${SWORD_ROUTE}",
    "validation": rows,
}
(run / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf8")
lines = ["# no-hp403 MT4 RedGem -> Shield Report", "", f"Run dir: `{run}`", "", "hp403 usage: none", ""]
for row in rows:
    final = row.get("final", {}) if isinstance(row.get("final"), dict) else {}
    lines.append(
        f"- `{Path(row.get('path', '')).name}` target={row.get('target_stage')} "
        f"success={row.get('target_success')} ok={row.get('strict_replay_ok')} "
        f"hp={final.get('hp')} atk={final.get('atk')} def={final.get('defense') or final.get('def')} "
        f"money={final.get('money')} keys={final.get('keys')} floor={final.get('floor')} "
        f"boss_margin={row.get('boss_margin')} steps={row.get('macro_steps')}"
    )
(run / "mt4red_shield_report.md").write_text("\\n".join(lines) + "\\n", encoding="utf8")
print(json.dumps({"summary": str(run / "summary.json"), "report": str(run / "mt4red_shield_report.md")}, ensure_ascii=False))
PY

date -Is | tee "${RUN_DIR}/finished_at.txt"
