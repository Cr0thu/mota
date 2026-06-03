#!/usr/bin/env bash
set -euo pipefail

if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /opt/conda/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-humanoid}" || true
fi

export PYTHONPATH="${PYTHONPATH:-src}"

RUN_ID="${RUN_ID:-4090_delay_refill_v30_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-artifacts/runs/${RUN_ID}}"
STAGED_RUN="${STAGED_RUN:-artifacts/runs/4090_pure_search_rl_20260527_104212}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TIMEOUT_MIN="${TIMEOUT_MIN:-42}"
MAX_INPUTS_PER_PHASE="${MAX_INPUTS_PER_PHASE:-6}"

mkdir -p "${RUN_DIR}"/{env,inputs,low_gems,mid_gems,mt8_gems,mt10_first,guard_low_refill,guard_ready,red_key,mt10_resources,boss_ready,boss,logs,validation}
echo "${RUN_DIR}" > artifacts/runs/latest_remote_delay_refill_run.txt

{
  echo "pwd=$(pwd)"
  echo "python=$(${PYTHON_BIN} --version 2>&1)"
  echo "staged_run=${STAGED_RUN}"
  echo "timeout_min=${TIMEOUT_MIN}"
  echo "max_inputs_per_phase=${MAX_INPUTS_PER_PHASE}"
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
} | tee "${RUN_DIR}/env/env_check.txt"

REFILL_PENALTY='"pre_mt10_potion_penalty":35.0,"pre_mt10_blue_potion_penalty":70.0,"pre_mt10_low_floor_potion_penalty":55.0,"pre_mt10_key_refill_potion_penalty":260.0,"pre_mt10_potion_hp_floor":170.0,"pre_mt10_potion_low_hp_multiplier":0.20'
ADAPT_GEMS="{\"atk_delta\":220.0,\"def_delta\":245.0,\"yellow_key_delta\":72.0,\"blue_key_delta\":100.0,\"yellow_key_level\":26.0,\"blue_key_level\":42.0,\"hp_delta\":0.04,${REFILL_PENALTY}}"
ADAPT_MT8="{\"atk_delta\":205.0,\"def_delta\":240.0,\"yellow_key_delta\":88.0,\"blue_key_delta\":130.0,\"yellow_key_level\":38.0,\"blue_key_level\":60.0,\"hp_delta\":0.05,${REFILL_PENALTY}}"
ADAPT_MT10="{\"atk_delta\":160.0,\"def_delta\":190.0,\"yellow_key_delta\":130.0,\"blue_key_delta\":150.0,\"yellow_key_level\":66.0,\"blue_key_level\":76.0,\"hp_delta\":0.07,${REFILL_PENALTY}}"
ADAPT_REFILL='{"hp_delta":0.52,"atk_delta":210.0,"def_delta":235.0,"yellow_key_delta":84.0,"blue_key_delta":100.0,"yellow_key_level":34.0,"blue_key_level":42.0,"guard_margin_delta":1.65,"guard_margin_level":0.030,"red_key_route_margin_delta":1.65,"red_key_route_margin_level":0.026,"last_yellow_key_spent":340.0}'
ADAPT_RED='{"hp_delta":0.18,"atk_delta":190.0,"def_delta":205.0,"yellow_key_delta":72.0,"blue_key_delta":86.0,"yellow_key_level":26.0,"blue_key_level":30.0,"guard_margin_delta":1.20,"guard_margin_level":0.018,"red_key_route_margin_delta":1.20,"red_key_route_margin_level":0.016,"last_yellow_key_spent":320.0}'
ADAPT_BOSS='{"hp_delta":0.18,"atk_delta":190.0,"def_delta":190.0,"yellow_key_delta":45.0,"blue_key_delta":70.0,"yellow_key_level":16.0,"blue_key_level":24.0,"guard_margin_delta":0.60,"red_key_route_margin_delta":0.60}'

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
      --action-bias-weight 1.75 \
      --potential-weight 0.145 \
      --fast-score-weight 0.0065 \
      --distance-weight 2.25 \
      --path-step-penalty 0.0016 \
      --macro-step-penalty 0.010 \
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

cat > "${RUN_DIR}/inputs/prefix_candidates.tsv" <<EOF
s1_after_7f_buy	${STAGED_RUN}/staged/strict_seed_20260601.jsonl	164
s1_before_mt8	${STAGED_RUN}/staged/strict_seed_20260601.jsonl	198
s2_after_shield	${STAGED_RUN}/staged/strict_seed_20260602.jsonl	149
s2_after_7f_key_refill	${STAGED_RUN}/staged/strict_seed_20260602.jsonl	170
s3_after_6f_yellow_gain	${STAGED_RUN}/staged/strict_seed_20260603.jsonl	145
s3_after_shield	${STAGED_RUN}/staged/strict_seed_20260603.jsonl	135
EOF

run_phase_from_tsv low_gems low_gems low_gems 135 340 38 "${ADAPT_GEMS}" "${RUN_DIR}/inputs/prefix_candidates.tsv"
collect_routes low_gems "${RUN_DIR}/inputs/low_routes.txt"

run_phase_from_routes mid_gems mid_gems mid_gems 140 360 40 "${ADAPT_GEMS}" "${RUN_DIR}/inputs/low_routes.txt"
collect_routes mid_gems "${RUN_DIR}/inputs/mid_routes.txt"

run_phase_from_routes mt8_gems mt8_gems mt8_gems 190 420 42 "${ADAPT_MT8}" "${RUN_DIR}/inputs/mid_routes.txt"
collect_routes mt8_gems "${RUN_DIR}/inputs/mt8_routes.txt"

run_phase_from_routes mt10_first mt10_first_resource mt10_first 150 460 42 "${ADAPT_MT10}" "${RUN_DIR}/inputs/mt8_routes.txt"
collect_routes mt10_first "${RUN_DIR}/inputs/mt10_first_routes.txt"

run_phase_from_routes guard_low_refill guard_low_refill guard_low_refill 125 420 42 "${ADAPT_REFILL}" "${RUN_DIR}/inputs/mt10_first_routes.txt"
collect_routes guard_low_refill "${RUN_DIR}/inputs/refill_routes.txt"

run_phase_from_routes guard_ready guard_ready guard_ready 150 440 44 "${ADAPT_REFILL}" "${RUN_DIR}/inputs/refill_routes.txt"
collect_routes guard_ready "${RUN_DIR}/inputs/guard_routes.txt"

run_phase_from_routes red_key red_key red_key 180 440 44 "${ADAPT_RED}" "${RUN_DIR}/inputs/guard_routes.txt"
collect_routes red_key "${RUN_DIR}/inputs/red_key_routes.txt"

run_phase_from_routes mt10_resources mt10_resources mt10_resources 180 420 42 "${ADAPT_MT10}" "${RUN_DIR}/inputs/red_key_routes.txt"
collect_routes mt10_resources "${RUN_DIR}/inputs/mt10_resource_routes.txt"

run_phase_from_routes boss_ready boss_ready boss_ready 150 380 40 "${ADAPT_BOSS}" "${RUN_DIR}/inputs/mt10_resource_routes.txt"
collect_routes boss_ready "${RUN_DIR}/inputs/boss_ready_routes.txt"

run_phase_from_routes boss boss boss 190 420 44 "${ADAPT_BOSS}" "${RUN_DIR}/inputs/boss_ready_routes.txt"
collect_routes boss "${RUN_DIR}/inputs/boss_routes.txt"

for route in "${RUN_DIR}"/low_gems/*_route.jsonl "${RUN_DIR}"/mid_gems/*_route.jsonl "${RUN_DIR}"/mt8_gems/*_route.jsonl "${RUN_DIR}"/mt10_first/*_route.jsonl "${RUN_DIR}"/guard_low_refill/*_route.jsonl "${RUN_DIR}"/guard_ready/*_route.jsonl "${RUN_DIR}"/red_key/*_route.jsonl "${RUN_DIR}"/mt10_resources/*_route.jsonl "${RUN_DIR}"/boss_ready/*_route.jsonl "${RUN_DIR}"/boss/*_route.jsonl "${RUN_DIR}"/boss/*_trace_best_route.jsonl; do
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
for part in ("low_gems", "mid_gems", "mt8_gems", "mt10_first", "guard_low_refill", "guard_ready", "red_key", "mt10_resources", "boss_ready", "boss"):
    for p in sorted((run / part).glob("*_summary.json")):
        try:
            rows.append({"part": part, "path": str(p), "data": json.loads(p.read_text(encoding="utf8"))})
        except Exception as exc:
            rows.append({"part": part, "path": str(p), "error": repr(exc)})
payload = {"run_dir": str(run), "staged_run": "${STAGED_RUN}", "rows": rows}
(run / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf8")
lines = ["# 4090 Delayed Refill v30 Report", "", f"Run dir: `{run}`", f"Staged run: `${STAGED_RUN}`", ""]
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
        f"keys={final.get('keys')} floor={final.get('floor_id') or final.get('floor')} pos=({final.get('x')},{final.get('y')}) "
        f"mt10_progress={final.get('mt10_resource_progress')} red_margin={final.get('red_key_route_margin')} boss_margin={final.get('boss_margin')} flags={flags}"
    )
(run / "delay_refill_v30_report.md").write_text("\\n".join(lines) + "\\n", encoding="utf8")
print(json.dumps({"summary": str(run / "summary.json"), "report": str(run / "delay_refill_v30_report.md")}, ensure_ascii=False))
PY

date -Is | tee "${RUN_DIR}/finished_at.txt"
