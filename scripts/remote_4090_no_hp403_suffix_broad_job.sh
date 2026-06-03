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
export MOTA_SIM_CACHE_LIMIT="${MOTA_SIM_CACHE_LIMIT:-250000}"
export MOTA_GRAPH_CACHE_LIMIT="${MOTA_GRAPH_CACHE_LIMIT:-60000}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

RUN_ID="${RUN_ID:-4090_no_hp403_suffix_broad_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-artifacts/runs/${RUN_ID}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TIMEOUT_MIN="${TIMEOUT_MIN:-130}"

mkdir -p "${RUN_DIR}"/{logs,beam,validation,env}
echo "${RUN_DIR}" > artifacts/runs/latest_remote_no_hp403_suffix_broad_run.txt

{
  echo "pwd=$(pwd)"
  echo "run_dir=${RUN_DIR}"
  echo "hp403_usage=none"
  echo "python=$(${PYTHON_BIN} --version 2>&1)"
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
} | tee "${RUN_DIR}/env/env_check.txt"

ADAPT_RED='{"hp_delta":0.26,"atk_delta":240.0,"def_delta":260.0,"yellow_key_delta":105.0,"blue_key_delta":130.0,"yellow_key_level":42.0,"blue_key_level":52.0,"guard_margin_delta":2.0,"guard_margin_level":0.035,"red_key_route_margin_delta":2.1,"red_key_route_margin_level":0.032,"last_yellow_key_spent":420.0,"pre_mt10_key_refill_potion_penalty":120.0}'
ADAPT_BOSS='{"hp_delta":0.30,"atk_delta":230.0,"def_delta":230.0,"yellow_key_delta":75.0,"blue_key_delta":100.0,"yellow_key_level":28.0,"blue_key_level":35.0,"guard_margin_delta":1.1,"red_key_route_margin_delta":1.1,"last_yellow_key_spent":260.0}'

run_beam() {
  local gpu="$1"
  local tag="$2"
  local route="$3"
  local target="$4"
  local mode="$5"
  local steps="$6"
  local width="$7"
  local topk="$8"
  local adapt="$9"
  local extra=()
  if [ "${mode}" = relaxed ]; then
    extra=(--allow-negative-hp --relaxed-min-hp -5000)
  fi
  if [ ! -s "${route}" ]; then
    echo "[suffix-broad] missing route for ${tag}: ${route}" >&2
    return 0
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" timeout "${TIMEOUT_MIN}m" "${PYTHON_BIN}" -m mota_rl.beam_decode \
    --target-stage "${target}" \
    --start-route "${route}" \
    --model-weight 0.0 \
    --beam-width "${width}" \
    --action-top-k "${topk}" \
    --max-steps "${steps}" \
    --action-bias-weight 2.05 \
    --potential-weight 0.18 \
    --fast-score-weight 0.008 \
    --distance-weight 2.65 \
    --path-step-penalty 0.0012 \
    --macro-step-penalty 0.008 \
    --revisit-penalty 1.65 \
    --success-bonus 4200 \
    --max-per-diversity-key 42 \
    --continue-after-success \
    --success-patience 45 \
    --adaptive-weights-json "${adapt}" \
    "${extra[@]}" \
    --route-out "${RUN_DIR}/beam/${tag}_route.jsonl" \
    --summary-out "${RUN_DIR}/beam/${tag}_summary.json" \
    --trace-out "${RUN_DIR}/beam/${tag}_trace.jsonl" \
    > "${RUN_DIR}/logs/${tag}.log" 2>&1 || true
}

MT8_D="${MT8_D:-artifacts/runs/4090_no_hp403_mt8_focus_20260529_v9/alpha/d/best_route.jsonl}"
MT8_B="${MT8_B:-artifacts/runs/4090_no_hp403_mt8_focus_20260529_v9/alpha/b/best_route.jsonl}"
MT10_A="${MT10_A:-artifacts/runs/4090_no_hp403_mt10_yellow_direct_from_mt8d_20260529_v3/alpha/a/best_route.jsonl}"

run_beam 0 mt8d_redkey_strict "${MT8_D}" red_key strict 280 760 78 "${ADAPT_RED}" &
run_beam 1 mt8b_boss_strict "${MT8_B}" boss strict 380 760 80 "${ADAPT_BOSS}" &
run_beam 2 mt10a_redkey_strict "${MT10_A}" red_key strict 300 760 78 "${ADAPT_RED}" &
run_beam 3 mt10a_boss_relaxed "${MT10_A}" boss relaxed 390 820 86 "${ADAPT_BOSS}" &
wait || true

for route in "${RUN_DIR}"/beam/*_route.jsonl "${RUN_DIR}"/beam/*_trace_best_route.jsonl; do
  [ -s "${route}" ] || continue
  safe="$(basename "${route}" .jsonl)"
  "${PYTHON_BIN}" scripts/replay_route.py --route "${route}" > "${RUN_DIR}/validation/${safe}_replay.json" 2>&1 || true
  "${PYTHON_BIN}" scripts/validate_route_constraints.py "${route}" > "${RUN_DIR}/validation/${safe}_constraints.json" 2>&1 || true
done

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

run = Path("${RUN_DIR}")
rows = []
for path in sorted((run / "beam").glob("*_summary.json")):
    try:
        rows.append({"path": str(path), **json.loads(path.read_text(encoding="utf8"))})
    except Exception as exc:
        rows.append({"path": str(path), "error": repr(exc)})
payload = {"run_dir": str(run), "hp403_usage": "none", "rows": rows}
(run / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf8")
lines = ["# no-hp403 Suffix Broad Report", "", f"Run dir: `{run}`", ""]
for row in rows:
    final = row.get("final", {}) if isinstance(row.get("final"), dict) else {}
    lines.append(
        f"- `{Path(row.get('path', '')).name}` solved={row.get('solved')} "
        f"strict_success={row.get('strict_success')} hp={final.get('hp')} "
        f"atk={final.get('atk')} def={final.get('defense') or final.get('def')} "
        f"keys={final.get('keys')} floor={final.get('floor_id') or final.get('floor')} "
        f"boss_margin={final.get('boss_margin') or row.get('boss_margin')}"
    )
(run / "suffix_broad_report.md").write_text("\\n".join(lines) + "\\n", encoding="utf8")
print(json.dumps({"summary": str(run / "summary.json"), "report": str(run / "suffix_broad_report.md")}, ensure_ascii=False))
PY

date -Is | tee "${RUN_DIR}/finished_at.txt"
