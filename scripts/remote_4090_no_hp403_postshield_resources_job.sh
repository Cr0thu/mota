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

RUN_ID="${RUN_ID:-4090_no_hp403_postshield_resources_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-artifacts/runs/${RUN_ID}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SHIELD_BASE="${SHIELD_BASE:-artifacts/runs/4090_no_hp403_alpha_broad_opt3_20260528/selfplay}"
MAX_INPUTS_PER_STAGE="${MAX_INPUTS_PER_STAGE:-4}"
ALPHA_EPISODES="${ALPHA_EPISODES:-18}"
ALPHA_TIMEOUT_MIN="${ALPHA_TIMEOUT_MIN:-70}"
GO_TIMEOUT_MIN="${GO_TIMEOUT_MIN:-45}"
ENABLE_GO_PROBES="${ENABLE_GO_PROBES:-0}"
VARIANTS_PER_INPUT="${VARIANTS_PER_INPUT:-1}"

mkdir -p "${RUN_DIR}"/{env,inputs,logs,validation,alpha,go_explore,planner}
echo "${RUN_DIR}" > artifacts/runs/latest_remote_no_hp403_postshield_resources_run.txt

{
  echo "pwd=$(pwd)"
  echo "run_dir=${RUN_DIR}"
  echo "hp403_usage=none"
  echo "shield_base=${SHIELD_BASE}"
  echo "alpha_episodes=${ALPHA_EPISODES}"
  echo "alpha_timeout_min=${ALPHA_TIMEOUT_MIN}"
  echo "go_timeout_min=${GO_TIMEOUT_MIN}"
  echo "enable_go_probes=${ENABLE_GO_PROBES}"
  echo "variants_per_input=${VARIANTS_PER_INPUT}"
  echo "python=$(${PYTHON_BIN} --version 2>&1)"
  echo "MOTA_ENABLE_STAGE_CACHE=${MOTA_ENABLE_STAGE_CACHE}"
  echo "MOTA_SIM_CACHE_LIMIT=${MOTA_SIM_CACHE_LIMIT}"
  echo "MOTA_GRAPH_CACHE_LIMIT=${MOTA_GRAPH_CACHE_LIMIT}"
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
    action = row.get("action") or {}
    actions = sim.macro_actions(state)
    selected = None
    for candidate in actions:
        if candidate.get("label") == action.get("label"):
            selected = candidate
            break
    if selected is None:
        ok = False
        message = f"illegal at {index}: {action.get('label')}"
        break
    transition = sim.apply_macro_action(state, selected)
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
    "final": state_summary(state),
    "message": message,
    "steps": len(rows),
}
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8")
print(json.dumps(summary, ensure_ascii=False))
PY
}

constraint_check() {
  local route="$1"
  local tag="$2"
  [ -s "${route}" ] || return 0
  stage_check "${route}" "${3:-boss}" "${RUN_DIR}/validation/${tag}_stage.json" >/dev/null || true
  "${PYTHON_BIN}" scripts/validate_route_constraints.py "${route}" > "${RUN_DIR}/validation/${tag}_constraints.json" || true
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

route = Path(sys.argv[1])
target = sys.argv[2]
sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
state = sim.reset()
for line in route.read_text(encoding="utf8").splitlines():
    if not line.strip():
        continue
    transition = sim.apply_macro_action(state, json.loads(line)["action"])
    if not transition.ok:
        sys.exit(1)
sys.exit(0 if stage_complete(sim, state, target) and state.hp > 0 else 1)
PY
}

is_forbidden_hp403_path() {
  local path="$1"
  [[ "${path}" == *hp403* && "${path}" != *no_hp403* ]]
}

write_seed_inputs() {
  local out="${RUN_DIR}/inputs/shield_inputs.tsv"
  : > "${out}"
  local tags=(
    "alpha_shield_m065_t070_n030"
    "alpha_shield_m050_t090_n040"
    "alpha_shield_m085_t120_n055"
    "alpha_shield_m075_t095_n045"
  )
  for tag in "${tags[@]}"; do
    local route="${SHIELD_BASE}/${tag}/best_route.jsonl"
    local ckpt="${SHIELD_BASE}/${tag}/best_model.pt"
    if [ -s "${route}" ] && [ -s "${ckpt}" ]; then
      if is_forbidden_hp403_path "${route}" || is_forbidden_hp403_path "${ckpt}"; then
        echo "[postshield] refusing hp403 input: ${route} ${ckpt}" >&2
        exit 2
      fi
      constraint_check "${route}" "seed_${tag}" shield
      if route_stage_success "${route}" shield; then
        printf '%s\t%s\t%s\n' "${tag}" "${route}" "${ckpt}" >> "${out}"
      else
        echo "[postshield] skipping non-shield seed: ${route}" >&2
      fi
    fi
  done
  if [ ! -s "${out}" ]; then
    find "${SHIELD_BASE}" -mindepth 2 -maxdepth 2 -name best_route.jsonl -print | sort | while read -r route; do
      [ -s "${route}" ] || continue
      local dir
      dir="$(dirname "${route}")"
      local ckpt="${dir}/best_model.pt"
      [ -s "${ckpt}" ] || continue
      if is_forbidden_hp403_path "${route}" || is_forbidden_hp403_path "${ckpt}"; then
        continue
      fi
      local tag
      tag="$(basename "${dir}")"
      constraint_check "${route}" "seed_${tag}" shield
      if route_stage_success "${route}" shield; then
        printf '%s\t%s\t%s\n' "${tag}" "${route}" "${ckpt}" >> "${out}"
      else
        echo "[postshield] skipping non-shield seed: ${route}" >&2
      fi
    done
  fi
  if [ ! -s "${out}" ]; then
    echo "[postshield] no no-hp403 shield routes found under ${SHIELD_BASE}" >&2
    find "${SHIELD_BASE}" -maxdepth 3 -type f \( -name best_route.jsonl -o -name best_model.pt \) -print >&2 || true
    exit 1
  fi
  sort -u "${out}" -o "${out}"
}

run_alpha_one() {
  local gpu="$1"
  local tag="$2"
  local route="$3"
  local ckpt="$4"
  local target="$5"
  local stop_stage="$6"
  local max_macros="$7"
  local sims="$8"
  local depth="$9"
  local mix="${10}"
  local action_temp="${11}"
  local out_dir="${RUN_DIR}/alpha/${target}/${tag}_${target}"
  mkdir -p "$(dirname "${out_dir}")"
  CUDA_VISIBLE_DEVICES="${gpu}" timeout "${ALPHA_TIMEOUT_MIN}m" "${PYTHON_BIN}" scripts/train_alpha_mota_stage.py \
    --out-dir "${out_dir}" \
    --target-stage "${target}" \
    --init-checkpoint "${ckpt}" \
    --start-route "${route}" \
    --start-route-stop-stage "${stop_stage}" \
    --episodes "${ALPHA_EPISODES}" \
    --max-macros "${max_macros}" \
    --simulations "${sims}" \
    --max-depth "${depth}" \
    --c-puct 1.85 \
    --max-state-revisits 0 \
    --selected-policy-target \
    --hp-aware-success-value \
    --seed "$((20268000 + gpu * 1000 + RANDOM % 997))" \
    --device cuda \
    --d-model 96 \
    --heads 4 \
    --layers 3 \
    --batch-size 64 \
    --train-steps-per-episode 10 \
    --heuristic-prior-mix "${mix}" \
    --heuristic-temperature 0.66 \
    --policy-temperature 0.88 \
    --value-target-mode mixed \
    --mixed-final-weight 0.55 \
    --final-action-value-weight 0.78 \
    --final-action-prior-weight 0.46 \
    --root-dirichlet-alpha 0.35 \
    --root-exploration-fraction 0.42 \
    --action-temperature "${action_temp}" \
    --action-top-k 10 \
    --policy-value-weighted \
    --save-every 6 \
    > "${RUN_DIR}/logs/${tag}_${target}.log" 2>&1 || true
  if [ -s "${out_dir}/best_route.jsonl" ]; then
    constraint_check "${out_dir}/best_route.jsonl" "${tag}_${target}" "${target}"
  fi
  echo "${tag}_${target}_done" > "${RUN_DIR}/logs/${tag}_${target}.done"
}

collect_alpha_outputs() {
  local target="$1"
  local out="$2"
  local solved="${out}.solved"
  local all="${out}.all"
  : > "${solved}"
  : > "${all}"
  for route in "${RUN_DIR}/alpha/${target}"/*/best_route.jsonl; do
    [ -s "${route}" ] || continue
    local dir
    dir="$(dirname "${route}")"
    local ckpt="${dir}/best_model.pt"
    [ -s "${ckpt}" ] || continue
    local tag
    tag="$(basename "${dir}")"
    printf '%s\t%s\t%s\n' "${tag}" "${route}" "${ckpt}" >> "${all}"
    if route_stage_success "${route}" "${target}"; then
      printf '%s\t%s\t%s\n' "${tag}" "${route}" "${ckpt}" >> "${solved}"
    fi
  done
  if [ -s "${solved}" ]; then
    sort -u "${solved}" > "${out}"
  elif [ -s "${all}" ]; then
    sort -u "${all}" > "${out}"
  else
    return 1
  fi
}

run_alpha_stage_from_inputs() {
  local target="$1"
  local stop_stage="$2"
  local input_file="$3"
  local max_macros="$4"
  local sims="$5"
  local depth="$6"
  local input_idx=0
  local launch_idx=0
  while IFS=$'\t' read -r tag route ckpt; do
    [ -n "${tag}" ] || continue
    [ -s "${route}" ] || continue
    [ -s "${ckpt}" ] || continue
    input_idx=$((input_idx + 1))
    if [ "${input_idx}" -gt "${MAX_INPUTS_PER_STAGE}" ]; then
      break
    fi
    local variant
    for variant in $(seq 1 "${VARIANTS_PER_INPUT}"); do
      launch_idx=$((launch_idx + 1))
      local gpu=$(( (launch_idx - 1) % 4 ))
      local mix
      local temp
      mix="$(awk "BEGIN { printf \"%.2f\", 0.39 + ((${launch_idx} + ${variant}) % 4) * 0.15 }")"
      temp="$(awk "BEGIN { printf \"%.2f\", 0.62 + ((${launch_idx} + ${variant}) % 5) * 0.11 }")"
      run_alpha_one "${gpu}" "${tag}_v${variant}" "${route}" "${ckpt}" "${target}" "${stop_stage}" "${max_macros}" "${sims}" "${depth}" "${mix}" "${temp}" &
    done
  done < "${input_file}"
  wait || true
}

run_go_probe() {
  local tag="$1"
  local route="$2"
  local target="$3"
  local stop_stage="$4"
  local mode="$5"
  local min_hp="$6"
  local out_dir="${RUN_DIR}/go_explore/${tag}_${target}_${mode}"
  timeout "${GO_TIMEOUT_MIN}m" "${PYTHON_BIN}" scripts/run_go_explore_experiment.py \
    --out-dir "${out_dir}" \
    --target-stage "${target}" \
    --start-route "${route}" \
    --start-route-stop-stage "${stop_stage}" \
    --mode "${mode}" \
    --relaxed-min-hp "${min_hp}" \
    --iterations 9000 \
    --rollout-steps 18 \
    --archive-top-k 64 \
    --candidate-top-k 12 \
    --temperature 1.05 \
    --novelty-bonus 420 \
    --revisit-penalty 0.018 \
    --trace-limit 2500 \
    --seed "$((20269000 + RANDOM % 997))" \
    > "${RUN_DIR}/logs/go_${tag}_${target}_${mode}.log" 2>&1 || true
  if [ -s "${out_dir}/best_route.jsonl" ]; then
    constraint_check "${out_dir}/best_route.jsonl" "go_${tag}_${target}_${mode}" "${target}"
  fi
}

write_seed_inputs

if [ "${ENABLE_GO_PROBES}" = "1" ]; then
  # Small Go-Explore probes keep the archive/reset idea active without flooding CPU.
  while IFS=$'\t' read -r tag route ckpt; do
    [ -s "${route}" ] || continue
    run_go_probe "${tag}" "${route}" mid_gems shield strict -1600 &
    run_go_probe "${tag}" "${route}" low_gems shield relaxed -1600 &
  done < <(head -n 2 "${RUN_DIR}/inputs/shield_inputs.tsv")
  wait || true
fi

run_alpha_stage_from_inputs mid_gems shield "${RUN_DIR}/inputs/shield_inputs.tsv" 105 36 44
collect_alpha_outputs mid_gems "${RUN_DIR}/inputs/mid_inputs.tsv" || cp "${RUN_DIR}/inputs/shield_inputs.tsv" "${RUN_DIR}/inputs/mid_inputs.tsv"

run_alpha_stage_from_inputs low_gems mid_gems "${RUN_DIR}/inputs/mid_inputs.tsv" 120 36 46
collect_alpha_outputs low_gems "${RUN_DIR}/inputs/low_inputs.tsv" || cp "${RUN_DIR}/inputs/mid_inputs.tsv" "${RUN_DIR}/inputs/low_inputs.tsv"

run_alpha_stage_from_inputs mt8_gems low_gems "${RUN_DIR}/inputs/low_inputs.tsv" 150 40 50
collect_alpha_outputs mt8_gems "${RUN_DIR}/inputs/mt8_inputs.tsv" || cp "${RUN_DIR}/inputs/low_inputs.tsv" "${RUN_DIR}/inputs/mt8_inputs.tsv"

run_alpha_stage_from_inputs mt10_resources mt8_gems "${RUN_DIR}/inputs/mt8_inputs.tsv" 165 44 52
collect_alpha_outputs mt10_resources "${RUN_DIR}/inputs/mt10_inputs.tsv" || cp "${RUN_DIR}/inputs/mt8_inputs.tsv" "${RUN_DIR}/inputs/mt10_inputs.tsv"

run_alpha_stage_from_inputs red_key mt10_resources "${RUN_DIR}/inputs/mt10_inputs.tsv" 165 44 52
collect_alpha_outputs red_key "${RUN_DIR}/inputs/red_key_inputs.tsv" || true

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

run = Path("${RUN_DIR}")
rows = []
for path in sorted((run / "validation").glob("*_stage.json")):
    try:
        data = json.loads(path.read_text(encoding="utf8"))
    except Exception as exc:
        data = {"error": repr(exc)}
    rows.append({"path": str(path), **data})
summary = {
    "run_dir": str(run),
    "hp403_usage": "none",
    "shield_base": "${SHIELD_BASE}",
    "alpha_episodes": int("${ALPHA_EPISODES}"),
    "validation": rows,
}
(run / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8")
lines = [
    "# no-hp403 Post-Shield Resources Report",
    "",
    f"Run dir: `{run}`",
    "",
    "This run uses no hp403 route. Inputs come from no-hp403 AlphaZero shield routes only.",
    "",
]
for row in rows:
    final = row.get("final") if isinstance(row.get("final"), dict) else {}
    keys = final.get("keys")
    lines.append(
        f"- `{Path(row.get('path', '')).name}` target={row.get('target_stage')} "
        f"success={row.get('target_success')} replay_ok={row.get('strict_replay_ok')} "
        f"hp={final.get('hp')} atk={final.get('atk')} def={final.get('defense') or final.get('def')} "
        f"money={final.get('money')} keys={keys} floor={final.get('floor_id') or final.get('floor')} "
        f"boss_margin={row.get('boss_margin')}"
    )
(run / "postshield_resources_report.md").write_text("\\n".join(lines) + "\\n", encoding="utf8")
print(json.dumps({"summary": str(run / "summary.json"), "report": str(run / "postshield_resources_report.md")}, ensure_ascii=False))
PY

date -Is | tee "${RUN_DIR}/finished_at.txt"
