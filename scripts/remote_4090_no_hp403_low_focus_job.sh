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

RUN_ID="${RUN_ID:-4090_no_hp403_low_focus_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-artifacts/runs/${RUN_ID}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASE="${BASE:-artifacts/runs/4090_no_hp403_postshield_resources_20260529_v3/alpha/mid_gems}"
EPISODES="${EPISODES:-24}"

mkdir -p "${RUN_DIR}"/{env,logs,validation,alpha}
echo "${RUN_DIR}" > artifacts/runs/latest_remote_no_hp403_low_focus_run.txt

{
  echo "pwd=$(pwd)"
  echo "run_dir=${RUN_DIR}"
  echo "hp403_usage=none"
  echo "base=${BASE}"
  echo "episodes=${EPISODES}"
  echo "python=$(${PYTHON_BIN} --version 2>&1)"
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
} | tee "${RUN_DIR}/env/env_check.txt"

stage_check() {
  local route="$1"
  local target="$2"
  local out="$3"
  "${PYTHON_BIN}" - "$route" "$target" "$out" <<'PY' || true
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
for line in route.read_text(encoding="utf8").splitlines():
    if not line.strip():
        continue
    transition = sim.apply_macro_action(state, json.loads(line)["action"])
    if not transition.ok:
        ok = False
        break
payload = {
    "route": str(route),
    "target_stage": target,
    "strict_replay_ok": ok,
    "target_success": bool(ok and stage_complete(sim, state, target) and state.hp > 0),
    "mid_success": bool(ok and stage_complete(sim, state, "mid_gems") and state.hp > 0),
    "low_remaining": remaining_stage_gem_targets(sim, state, "low_gems"),
    "boss_margin": boss_route_margin(sim, state),
    "final": state_summary(state),
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf8")
print(json.dumps(payload, ensure_ascii=False))
PY
}

alpha_one() {
  local gpu="$1"
  local src="$2"
  local tag="$3"
  local mix="$4"
  local temp="$5"
  local cpuct="$6"
  local sims="$7"
  local out_dir="${RUN_DIR}/alpha/${tag}"
  local route="${BASE}/${src}/best_route.jsonl"
  local ckpt="${BASE}/${src}/best_model.pt"
  if [ ! -s "${route}" ] || [ ! -s "${ckpt}" ]; then
    echo "[low-focus] missing source ${src}" >&2
    return 0
  fi
  if [[ "${route}" == *hp403* && "${route}" != *no_hp403* ]]; then
    echo "[low-focus] refusing hp403 source ${route}" >&2
    return 0
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" timeout 80m "${PYTHON_BIN}" scripts/train_alpha_mota_stage.py \
    --out-dir "${out_dir}" \
    --target-stage low_gems \
    --init-checkpoint "${ckpt}" \
    --start-route "${route}" \
    --start-route-stop-stage mid_gems \
    --episodes "${EPISODES}" \
    --max-macros 132 \
    --simulations "${sims}" \
    --max-depth 54 \
    --c-puct "${cpuct}" \
    --max-state-revisits 0 \
    --selected-policy-target \
    --hp-aware-success-value \
    --seed "$((20270000 + gpu * 1000 + RANDOM % 997))" \
    --device cuda \
    --d-model 96 \
    --heads 4 \
    --layers 3 \
    --batch-size 64 \
    --train-steps-per-episode 12 \
    --heuristic-prior-mix "${mix}" \
    --heuristic-temperature 0.58 \
    --policy-temperature 0.84 \
    --value-target-mode mixed \
    --mixed-final-weight 0.58 \
    --final-action-value-weight 0.82 \
    --final-action-prior-weight 0.52 \
    --root-dirichlet-alpha 0.30 \
    --root-exploration-fraction 0.36 \
    --action-temperature "${temp}" \
    --action-top-k 8 \
    --policy-value-weighted \
    --save-every 6 \
    > "${RUN_DIR}/logs/${tag}.log" 2>&1 || true
  if [ -s "${out_dir}/best_route.jsonl" ]; then
    stage_check "${out_dir}/best_route.jsonl" low_gems "${RUN_DIR}/validation/${tag}_stage.json" >/dev/null || true
    "${PYTHON_BIN}" scripts/validate_route_constraints.py "${out_dir}/best_route.jsonl" > "${RUN_DIR}/validation/${tag}_constraints.json" || true
  fi
}

alpha_one 0 alpha_shield_m050_t090_n040_mid_gems m050_keyprotect_a 0.82 0.55 1.70 44 &
alpha_one 1 alpha_shield_m050_t090_n040_mid_gems m050_keyprotect_b 0.92 0.75 1.95 48 &
alpha_one 2 alpha_shield_m075_t095_n045_mid_gems m075_hp_a 0.78 0.65 1.80 44 &
alpha_one 3 alpha_shield_m075_t095_n045_mid_gems m075_hp_b 0.90 0.90 2.05 48 &
wait || true

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
run = Path("${RUN_DIR}")
rows = []
for path in sorted((run / "validation").glob("*_stage.json")):
    rows.append(json.loads(path.read_text(encoding="utf8")))
(run / "summary.json").write_text(json.dumps({"run_dir": str(run), "hp403_usage": "none", "rows": rows}, ensure_ascii=False, indent=2), encoding="utf8")
lines = ["# no-hp403 low_gems focus", "", f"Run dir: `{run}`", ""]
for row in rows:
    final = row.get("final", {})
    lines.append(
        f"- {Path(row['route']).parent.name}: success={row.get('target_success')} "
        f"low_remaining={row.get('low_remaining')} hp={final.get('hp')} atk={final.get('atk')} "
        f"def={final.get('defense') or final.get('def')} money={final.get('money')} "
        f"keys={final.get('keys')} floor={final.get('floor_id') or final.get('floor')}"
    )
(run / "low_focus_report.md").write_text("\\n".join(lines) + "\\n", encoding="utf8")
print(json.dumps({"summary": str(run / "summary.json"), "report": str(run / "low_focus_report.md")}, ensure_ascii=False))
PY

date -Is | tee "${RUN_DIR}/finished_at.txt"
