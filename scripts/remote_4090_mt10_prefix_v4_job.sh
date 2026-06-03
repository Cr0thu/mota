#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-4090_mt10_prefix_v4_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-artifacts/runs/${RUN_ID}}"
STAGED_RUN="${STAGED_RUN:-artifacts/runs/4090_pure_search_rl_20260527_104212}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TIMEOUT_MIN="${TIMEOUT_MIN:-45}"

mkdir -p "${RUN_DIR}"/{env,mt10,red_key,boss,logs,validation}
echo "${RUN_DIR}" > artifacts/runs/latest_remote_mt10_prefix_run.txt

{
  echo "pwd=$(pwd)"
  echo "python=$(${PYTHON_BIN} --version 2>&1)"
  echo "staged_run=${STAGED_RUN}"
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
} | tee "${RUN_DIR}/env/env_check.txt"

ADAPT_MT10='{"atk_delta":115.0,"def_delta":135.0,"yellow_key_delta":80.0,"blue_key_delta":95.0,"yellow_key_level":42.0,"blue_key_level":36.0,"hp_delta":0.05}'
ADAPT_RED='{"atk_delta":145.0,"def_delta":155.0,"yellow_key_delta":70.0,"blue_key_delta":85.0,"yellow_key_level":28.0,"blue_key_level":30.0,"guard_margin_delta":0.8,"guard_margin_level":0.008,"red_key_route_margin_delta":0.8,"red_key_route_margin_level":0.006,"last_yellow_key_spent":240.0,"hp_delta":0.06}'
ADAPT_BOSS='{"atk_delta":165.0,"def_delta":165.0,"yellow_key_delta":45.0,"blue_key_delta":70.0,"yellow_key_level":16.0,"blue_key_level":24.0,"guard_margin_delta":0.55,"red_key_route_margin_delta":0.55,"hp_delta":0.10}'

decode_prefix() {
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
      --action-bias-weight 1.45 \
      --potential-weight 0.105 \
      --fast-score-weight 0.0045 \
      --distance-weight 1.65 \
      --path-step-penalty 0.0025 \
      --macro-step-penalty 0.014 \
      --revisit-penalty 1.6 \
      --success-bonus 1500 \
      --max-per-diversity-key 14 \
      --continue-after-success \
      --success-patience 24 \
      --adaptive-weights-json "${adaptive}" \
      --route-out "${RUN_DIR}/${out_dir}/${tag}_route.jsonl" \
      --summary-out "${RUN_DIR}/${out_dir}/${tag}_summary.json" \
      --trace-out "${RUN_DIR}/${out_dir}/${tag}_trace.jsonl" \
      > "${RUN_DIR}/logs/${tag}.log" 2>&1 || true
}

route1="${STAGED_RUN}/staged/strict_seed_20260601.jsonl"
route2="${STAGED_RUN}/staged/strict_seed_20260602.jsonl"
route3="${STAGED_RUN}/staged/strict_seed_20260603.jsonl"

cat > "${RUN_DIR}/prefix_candidates.tsv" <<EOF
s1_after_7f_buy	${route1}	164
s1_before_mt8	${route1}	198
s2_after_shield	${route2}	149
s2_after_7f_key_refill	${route2}	170
s3_after_6f_yellow_gain	${route3}	145
s3_after_shield	${route3}	135
EOF

idx=0
while IFS=$'\t' read -r tag route max_prefix; do
  [ -n "${tag}" ] || continue
  idx=$((idx + 1))
  decode_prefix "$(( (idx - 1) % 4 ))" "${route}" "${max_prefix}" mt10_resources mt10 "${tag}_mt10" 180 340 32 "${ADAPT_MT10}" &
done < "${RUN_DIR}/prefix_candidates.tsv"
wait || true

idx=0
for summary in "${RUN_DIR}"/mt10/*_summary.json; do
  [ -f "${summary}" ] || continue
  route="${summary%_summary.json}_route.jsonl"
  tag="$(basename "${summary%_summary.json}")"
  idx=$((idx + 1))
  decode_prefix "$(( (idx - 1) % 4 ))" "${route}" 0 red_key red_key "${tag}_red_key" 140 280 30 "${ADAPT_RED}" &
done
wait || true

idx=0
for summary in "${RUN_DIR}"/red_key/*_summary.json; do
  [ -f "${summary}" ] || continue
  route="${summary%_summary.json}_route.jsonl"
  tag="$(basename "${summary%_summary.json}")"
  idx=$((idx + 1))
  decode_prefix "$(( (idx - 1) % 4 ))" "${route}" 0 boss boss "${tag}_boss" 160 300 32 "${ADAPT_BOSS}" &
done
wait || true

for route in "${RUN_DIR}"/mt10/*_route.jsonl "${RUN_DIR}"/red_key/*_route.jsonl "${RUN_DIR}"/boss/*_route.jsonl; do
  [ -f "${route}" ] || continue
  safe="$(echo "${route#${RUN_DIR}/}" | tr '/.' '__')"
  PYTHONPATH=src "${PYTHON_BIN}" scripts/replay_route.py --route "${route}" > "${RUN_DIR}/validation/${safe}_replay.json" || true
  PYTHONPATH=src "${PYTHON_BIN}" scripts/validate_route_constraints.py "${route}" > "${RUN_DIR}/validation/${safe}_constraints.json" || true
done

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
run = Path("${RUN_DIR}")
rows = []
for part in ("mt10", "red_key", "boss"):
    for p in sorted((run / part).glob("*_summary.json")):
        try:
            rows.append({"part": part, "path": str(p), "data": json.loads(p.read_text(encoding="utf8"))})
        except Exception as exc:
            rows.append({"part": part, "path": str(p), "error": repr(exc)})
payload = {"run_dir": str(run), "staged_run": "${STAGED_RUN}", "rows": rows}
(run / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf8")
lines = ["# 4090 MT10 Prefix v4 Report", "", f"Run dir: `{run}`", ""]
for row in rows:
    if "error" in row:
        lines.append(f"- `{row['path']}` error={row['error']}")
        continue
    data = row["data"]
    final = data.get("final", {})
    flags = final.get("target_flags", {}) if isinstance(final, dict) else {}
    lines.append(
        f"- {row['part']} `{row['path']}` solved={data.get('solved')} strict_success={data.get('strict_success')} "
        f"route={data.get('route_out')} hp={final.get('hp')} atk={final.get('atk')} def={final.get('defense')} "
        f"keys={final.get('keys')} floor={final.get('floor_id') or final.get('floor')} pos=({final.get('x')},{final.get('y')}) flags={flags}"
    )
(run / "mt10_prefix_v4_report.md").write_text("\\n".join(lines) + "\\n", encoding="utf8")
print(json.dumps({"summary": str(run / "summary.json"), "report": str(run / "mt10_prefix_v4_report.md")}, ensure_ascii=False))
PY

date -Is | tee "${RUN_DIR}/finished_at.txt"
