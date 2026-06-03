#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-4090_gem_chain_v7_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-artifacts/runs/${RUN_ID}}"
STAGED_RUN="${STAGED_RUN:-artifacts/runs/4090_pure_search_rl_20260527_104212}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TIMEOUT_MIN="${TIMEOUT_MIN:-35}"
MAX_INPUTS_PER_PHASE="${MAX_INPUTS_PER_PHASE:-8}"

mkdir -p "${RUN_DIR}"/{env,low_gems,mid_gems,mt8_gems,mt10,red_key,boss,logs,validation}
echo "${RUN_DIR}" > artifacts/runs/latest_remote_gem_chain_run.txt

{
  echo "pwd=$(pwd)"
  echo "python=$(${PYTHON_BIN} --version 2>&1)"
  echo "staged_run=${STAGED_RUN}"
  echo "timeout_min=${TIMEOUT_MIN}"
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
} | tee "${RUN_DIR}/env/env_check.txt"

ADAPT_GEMS='{"atk_delta":210.0,"def_delta":230.0,"yellow_key_delta":70.0,"blue_key_delta":95.0,"yellow_key_level":24.0,"blue_key_level":40.0,"hp_delta":0.09}'
ADAPT_MT8='{"atk_delta":190.0,"def_delta":220.0,"yellow_key_delta":85.0,"blue_key_delta":125.0,"yellow_key_level":35.0,"blue_key_level":55.0,"hp_delta":0.10}'
ADAPT_MT10='{"atk_delta":145.0,"def_delta":165.0,"yellow_key_delta":105.0,"blue_key_delta":130.0,"yellow_key_level":54.0,"blue_key_level":62.0,"hp_delta":0.08}'
ADAPT_RED='{"atk_delta":170.0,"def_delta":180.0,"yellow_key_delta":72.0,"blue_key_delta":88.0,"yellow_key_level":30.0,"blue_key_level":32.0,"guard_margin_delta":0.95,"guard_margin_level":0.010,"red_key_route_margin_delta":0.95,"red_key_route_margin_level":0.008,"last_yellow_key_spent":260.0,"hp_delta":0.08}'
ADAPT_BOSS='{"atk_delta":180.0,"def_delta":180.0,"yellow_key_delta":45.0,"blue_key_delta":70.0,"yellow_key_level":16.0,"blue_key_level":24.0,"guard_margin_delta":0.55,"red_key_route_margin_delta":0.55,"hp_delta":0.12}'

decode_route() {
  local gpu="$1"
  local route="$2"
  local max_prefix="$3"
  local target="$4"
  local out_dir="$5"
  local tag="$6"
  local max_steps="$7"
  local beam_width="$8"
  local top_k="$9"
  local adaptive="${10}"
  CUDA_VISIBLE_DEVICES="${gpu}" MOTA_FAST_GRAPH_STATE=1 timeout "${TIMEOUT_MIN}m" \
    "${PYTHON_BIN}" -m mota_rl.beam_decode \
      --target-stage "${target}" \
      --start-route "${route}" \
      --start-route-max-steps "${max_prefix}" \
      --model-weight 0.0 \
      --beam-width "${beam_width}" \
      --action-top-k "${top_k}" \
      --max-steps "${max_steps}" \
      --action-bias-weight 1.55 \
      --potential-weight 0.120 \
      --fast-score-weight 0.0055 \
      --distance-weight 1.85 \
      --path-step-penalty 0.0020 \
      --macro-step-penalty 0.012 \
      --revisit-penalty 1.9 \
      --success-bonus 1800 \
      --max-per-diversity-key 18 \
      --continue-after-success \
      --success-patience 26 \
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

run_phase_from_tsv() {
  local phase="$1"
  local target="$2"
  local out_dir="$3"
  local max_steps="$4"
  local beam_width="$5"
  local top_k="$6"
  local adaptive="$7"
  local input_tsv="$8"
  local idx=0
  while IFS=$'\t' read -r tag route max_prefix; do
    [ -n "${tag}" ] || continue
    [ -s "${route}" ] || continue
    idx=$((idx + 1))
    if [ "${idx}" -gt "${MAX_INPUTS_PER_PHASE}" ]; then
      break
    fi
    decode_route "$(( (idx - 1) % 4 ))" "${route}" "${max_prefix}" "${target}" "${out_dir}" "${tag}_${phase}" "${max_steps}" "${beam_width}" "${top_k}" "${adaptive}" &
  done < "${input_tsv}"
  wait || true
}

run_phase_from_routes() {
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
    local base
    base="$(basename "${route}" .jsonl)"
    decode_route "$(( (idx - 1) % 4 ))" "${route}" 0 "${target}" "${out_dir}" "${base}_${phase}" "${max_steps}" "${beam_width}" "${top_k}" "${adaptive}" &
  done < "${input_list}"
  wait || true
}

cat > "${RUN_DIR}/prefix_candidates.tsv" <<EOF
s1_after_7f_buy	${STAGED_RUN}/staged/strict_seed_20260601.jsonl	164
s1_before_mt8	${STAGED_RUN}/staged/strict_seed_20260601.jsonl	198
s2_after_shield	${STAGED_RUN}/staged/strict_seed_20260602.jsonl	149
s2_after_7f_key_refill	${STAGED_RUN}/staged/strict_seed_20260602.jsonl	170
s3_after_6f_yellow_gain	${STAGED_RUN}/staged/strict_seed_20260603.jsonl	145
s3_after_shield	${STAGED_RUN}/staged/strict_seed_20260603.jsonl	135
EOF

run_phase_from_tsv low_gems low_gems low_gems 130 300 34 "${ADAPT_GEMS}" "${RUN_DIR}/prefix_candidates.tsv"
collect_routes low_gems "${RUN_DIR}/low_routes.txt"

run_phase_from_routes mid_gems mid_gems mid_gems 130 300 34 "${ADAPT_GEMS}" "${RUN_DIR}/low_routes.txt"
collect_routes mid_gems "${RUN_DIR}/mid_routes.txt"

run_phase_from_routes mt8_gems mt8_gems mt8_gems 180 360 36 "${ADAPT_MT8}" "${RUN_DIR}/mid_routes.txt"
collect_routes mt8_gems "${RUN_DIR}/mt8_routes.txt"

run_phase_from_routes mt10 mt10_resources mt10 210 380 38 "${ADAPT_MT10}" "${RUN_DIR}/mt8_routes.txt"
collect_routes mt10 "${RUN_DIR}/mt10_routes.txt"

run_phase_from_routes red_key red_key red_key 170 320 34 "${ADAPT_RED}" "${RUN_DIR}/mt10_routes.txt"
collect_routes red_key "${RUN_DIR}/red_key_routes.txt"

run_phase_from_routes boss boss boss 190 340 36 "${ADAPT_BOSS}" "${RUN_DIR}/red_key_routes.txt"
collect_routes boss "${RUN_DIR}/boss_routes.txt"

for route in "${RUN_DIR}"/low_gems/*_route.jsonl "${RUN_DIR}"/mid_gems/*_route.jsonl "${RUN_DIR}"/mt8_gems/*_route.jsonl "${RUN_DIR}"/mt10/*_route.jsonl "${RUN_DIR}"/red_key/*_route.jsonl "${RUN_DIR}"/boss/*_route.jsonl "${RUN_DIR}"/boss/*_trace_best_route.jsonl; do
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
for part in ("low_gems", "mid_gems", "mt8_gems", "mt10", "red_key", "boss"):
    for p in sorted((run / part).glob("*_summary.json")):
        try:
            rows.append({"part": part, "path": str(p), "data": json.loads(p.read_text(encoding="utf8"))})
        except Exception as exc:
            rows.append({"part": part, "path": str(p), "error": repr(exc)})
payload = {"run_dir": str(run), "staged_run": "${STAGED_RUN}", "rows": rows}
(run / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf8")
lines = ["# 4090 Gem Chain v7 Report", "", f"Run dir: `{run}`", ""]
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
(run / "gem_chain_v7_report.md").write_text("\\n".join(lines) + "\\n", encoding="utf8")
print(json.dumps({"summary": str(run / "summary.json"), "report": str(run / "gem_chain_v7_report.md")}, ensure_ascii=False))
PY

date -Is | tee "${RUN_DIR}/finished_at.txt"
