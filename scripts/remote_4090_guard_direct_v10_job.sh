#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-4090_guard_direct_v10_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-artifacts/runs/${RUN_ID}}"
UPSTREAM_RUN="${UPSTREAM_RUN:-artifacts/runs/4090_gem_chain_v7_20260527_210353}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TIMEOUT_MIN="${TIMEOUT_MIN:-35}"
MAX_INPUTS_PER_PHASE="${MAX_INPUTS_PER_PHASE:-4}"

mkdir -p "${RUN_DIR}"/{env,inputs,guard_ready,red_key,mt10,boss_ready,boss,logs,validation}
echo "${RUN_DIR}" > artifacts/runs/latest_remote_guard_direct_run.txt

{
  echo "pwd=$(pwd)"
  echo "python=$(${PYTHON_BIN} --version 2>&1)"
  echo "upstream_run=${UPSTREAM_RUN}"
  echo "timeout_min=${TIMEOUT_MIN}"
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
} | tee "${RUN_DIR}/env/env_check.txt"

ADAPT_GUARD='{"hp_delta":0.34,"atk_delta":230.0,"def_delta":250.0,"yellow_key_delta":86.0,"blue_key_delta":105.0,"yellow_key_level":38.0,"blue_key_level":46.0,"guard_margin_delta":1.80,"guard_margin_level":0.035,"red_key_route_margin_delta":1.80,"red_key_route_margin_level":0.030,"last_yellow_key_spent":360.0}'
ADAPT_RED='{"hp_delta":0.18,"atk_delta":190.0,"def_delta":205.0,"yellow_key_delta":72.0,"blue_key_delta":86.0,"yellow_key_level":26.0,"blue_key_level":30.0,"guard_margin_delta":1.20,"guard_margin_level":0.018,"red_key_route_margin_delta":1.20,"red_key_route_margin_level":0.016,"last_yellow_key_spent":320.0}'
ADAPT_MT10='{"hp_delta":0.12,"atk_delta":150.0,"def_delta":170.0,"yellow_key_delta":110.0,"blue_key_delta":130.0,"yellow_key_level":56.0,"blue_key_level":60.0}'
ADAPT_BOSS='{"hp_delta":0.18,"atk_delta":190.0,"def_delta":190.0,"yellow_key_delta":45.0,"blue_key_delta":70.0,"yellow_key_level":16.0,"blue_key_level":24.0,"guard_margin_delta":0.60,"red_key_route_margin_delta":0.60}'

decode_route() {
  local gpu="$1"
  local route="$2"
  local target="$3"
  local out_dir="$4"
  local tag="$5"
  local max_steps="$6"
  local beam_width="$7"
  local top_k="$8"
  local adaptive="$9"
  CUDA_VISIBLE_DEVICES="${gpu}" MOTA_FAST_GRAPH_STATE=1 timeout "${TIMEOUT_MIN}m" \
    "${PYTHON_BIN}" -m mota_rl.beam_decode \
      --target-stage "${target}" \
      --start-route "${route}" \
      --start-route-max-steps 0 \
      --model-weight 0.0 \
      --beam-width "${beam_width}" \
      --action-top-k "${top_k}" \
      --max-steps "${max_steps}" \
      --action-bias-weight 1.85 \
      --potential-weight 0.150 \
      --fast-score-weight 0.0065 \
      --distance-weight 2.35 \
      --path-step-penalty 0.0015 \
      --macro-step-penalty 0.008 \
      --revisit-penalty 2.2 \
      --success-bonus 2600 \
      --max-per-diversity-key 24 \
      --continue-after-success \
      --success-patience 34 \
      --adaptive-weights-json "${adaptive}" \
      --route-out "${RUN_DIR}/${out_dir}/${tag}_route.jsonl" \
      --summary-out "${RUN_DIR}/${out_dir}/${tag}_summary.json" \
      --trace-out "${RUN_DIR}/${out_dir}/${tag}_trace.jsonl" \
      > "${RUN_DIR}/logs/${tag}.log" 2>&1 || true
}

collect_routes() {
  local dir="$1"
  local out="$2"
  : > "${out}"
  for route in "${RUN_DIR}/${dir}"/*_route.jsonl "${RUN_DIR}/${dir}"/*_trace_best_route.jsonl; do
    [ -s "${route}" ] || continue
    printf '%s\n' "${route}" >> "${out}"
  done
  sort -u "${out}" -o "${out}"
}

run_phase() {
  local phase="$1"
  local target="$2"
  local out_dir="$3"
  local max_steps="$4"
  local beam_width="$5"
  local top_k="$6"
  local adaptive="$7"
  local input_list="$8"
  local idx=0
  while IFS= read -r route; do
    [ -s "${route}" ] || continue
    idx=$((idx + 1))
    if [ "${idx}" -gt "${MAX_INPUTS_PER_PHASE}" ]; then
      break
    fi
    base="$(basename "${route}" .jsonl)"
    decode_route "$(( (idx - 1) % 4 ))" "${route}" "${target}" "${out_dir}" "${base}_${phase}" "${max_steps}" "${beam_width}" "${top_k}" "${adaptive}" &
  done < "${input_list}"
  wait || true
}

cat > "${RUN_DIR}/inputs/mid_routes.txt" <<EOF
${UPSTREAM_RUN}/mid_gems/s1_before_mt8_low_gems_route_mid_gems_route.jsonl
${UPSTREAM_RUN}/mid_gems/s1_after_7f_buy_low_gems_trace_best_route_mid_gems_route.jsonl
${UPSTREAM_RUN}/mid_gems/s2_after_7f_key_refill_low_gems_trace_best_route_mid_gems_route.jsonl
${UPSTREAM_RUN}/mid_gems/s3_after_6f_yellow_gain_low_gems_trace_best_route_mid_gems_route.jsonl
EOF

run_phase guard_ready guard_ready guard_ready 260 520 52 "${ADAPT_GUARD}" "${RUN_DIR}/inputs/mid_routes.txt"
collect_routes guard_ready "${RUN_DIR}/guard_routes.txt"

run_phase red_key red_key red_key 170 360 38 "${ADAPT_RED}" "${RUN_DIR}/guard_routes.txt"
collect_routes red_key "${RUN_DIR}/red_key_routes.txt"

run_phase mt10 mt10_resources mt10 180 360 38 "${ADAPT_MT10}" "${RUN_DIR}/red_key_routes.txt"
collect_routes mt10 "${RUN_DIR}/mt10_routes.txt"

run_phase boss_ready boss_ready boss_ready 150 340 36 "${ADAPT_BOSS}" "${RUN_DIR}/mt10_routes.txt"
collect_routes boss_ready "${RUN_DIR}/boss_ready_routes.txt"

run_phase boss boss boss 190 360 38 "${ADAPT_BOSS}" "${RUN_DIR}/boss_ready_routes.txt"
collect_routes boss "${RUN_DIR}/boss_routes.txt"

for route in "${RUN_DIR}"/guard_ready/*_route.jsonl "${RUN_DIR}"/red_key/*_route.jsonl "${RUN_DIR}"/mt10/*_route.jsonl "${RUN_DIR}"/boss_ready/*_route.jsonl "${RUN_DIR}"/boss/*_route.jsonl "${RUN_DIR}"/boss/*_trace_best_route.jsonl; do
  [ -s "${route}" ] || continue
  safe="$(echo "${route#${RUN_DIR}/}" | tr '/.' '__')"
  PYTHONPATH=src "${PYTHON_BIN}" scripts/replay_route.py --route "${route}" > "${RUN_DIR}/validation/${safe}_replay.json" || true
  PYTHONPATH=src "${PYTHON_BIN}" scripts/validate_route_constraints.py "${route}" > "${RUN_DIR}/validation/${safe}_constraints.json" || true
done

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
run = Path("${RUN_DIR}")
rows = []
for part in ("guard_ready", "red_key", "mt10", "boss_ready", "boss"):
    for p in sorted((run / part).glob("*_summary.json")):
        try:
            rows.append({"part": part, "path": str(p), "data": json.loads(p.read_text(encoding="utf8"))})
        except Exception as exc:
            rows.append({"part": part, "path": str(p), "error": repr(exc)})
(run / "summary.json").write_text(json.dumps({"run_dir": str(run), "upstream_run": "${UPSTREAM_RUN}", "rows": rows}, ensure_ascii=False, indent=2), encoding="utf8")
lines = ["# 4090 Guard Direct v10 Report", "", f"Run dir: `{run}`", f"Upstream: `${UPSTREAM_RUN}`", ""]
for row in rows:
    if "error" in row:
        lines.append(f"- `{row['path']}` error={row['error']}")
        continue
    data = row["data"]
    final = data.get("final", {})
    flags = final.get("target_flags", {}) if isinstance(final, dict) else {}
    lines.append(
        f"- {row['part']} `{row['path']}` solved={data.get('solved')} strict_success={data.get('strict_success')} "
        f"route={data.get('route_out')} hp={final.get('hp')} atk={final.get('atk')} def={final.get('defense') or final.get('def')} "
        f"keys={final.get('keys')} floor={final.get('floor_id') or final.get('floor')} pos=({final.get('x')},{final.get('y')}) flags={flags}"
    )
(run / "guard_direct_v10_report.md").write_text("\\n".join(lines) + "\\n", encoding="utf8")
print(json.dumps({"summary": str(run / "summary.json"), "report": str(run / "guard_direct_v10_report.md")}, ensure_ascii=False))
PY

date -Is | tee "${RUN_DIR}/finished_at.txt"
