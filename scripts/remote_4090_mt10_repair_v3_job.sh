#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-4090_mt10_repair_v3_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-artifacts/runs/${RUN_ID}}"
STAGED_RUN="${STAGED_RUN:-artifacts/runs/4090_pure_search_rl_20260527_104212}"
RANKER_RUN="${RANKER_RUN:-artifacts/runs/4090_action_ranker_v1_20260527_152511}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TIMEOUT_MIN="${TIMEOUT_MIN:-35}"

mkdir -p "${RUN_DIR}"/{env,mt10,red_key,boss,logs,validation}
echo "${RUN_DIR}" > artifacts/runs/latest_remote_mt10_repair_run.txt

{
  echo "pwd=$(pwd)"
  echo "python=$(${PYTHON_BIN} --version 2>&1)"
  echo "staged_run=${STAGED_RUN}"
  echo "ranker_run=${RANKER_RUN}"
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

model_for() {
  local needle="$1"
  local model
  model="$(find "${RANKER_RUN}/ranker" -path "*${needle}*/best_model.pt" -print 2>/dev/null | sort | head -1 || true)"
  printf '%s' "${model}"
}

MT10_MODEL="$(model_for mt10_resources)"
RED_MODEL="$(model_for red_key)"
BOSS_MODEL="$(model_for boss)"

{
  echo "mt10_model=${MT10_MODEL}"
  echo "red_key_model=${RED_MODEL}"
  echo "boss_model=${BOSS_MODEL}"
} | tee "${RUN_DIR}/models.txt"

ADAPT_MT10='{"atk_delta":105.0,"def_delta":115.0,"yellow_key_delta":35.0,"blue_key_delta":70.0,"yellow_key_level":12.0,"blue_key_level":28.0,"hp_delta":0.04}'
ADAPT_RED='{"atk_delta":135.0,"def_delta":145.0,"yellow_key_delta":45.0,"blue_key_delta":80.0,"yellow_key_level":16.0,"blue_key_level":34.0,"guard_margin_delta":0.75,"guard_margin_level":0.008,"red_key_route_margin_delta":0.75,"red_key_route_margin_level":0.006,"last_yellow_key_spent":180.0,"hp_delta":0.05}'
ADAPT_BOSS='{"atk_delta":155.0,"def_delta":150.0,"yellow_key_delta":28.0,"blue_key_delta":55.0,"yellow_key_level":8.0,"blue_key_level":22.0,"guard_margin_delta":0.45,"red_key_route_margin_delta":0.45,"hp_delta":0.08}'

run_decode() {
  local gpu="$1"
  local target="$2"
  local start_route="$3"
  local model="$4"
  local out_dir="$5"
  local tag="$6"
  local max_steps="$7"
  local beam_width="$8"
  local top_k="$9"
  local adaptive="${10}"
  local model_args=()
  if [ -n "${model}" ]; then
    model_args+=(--model-path "${model}" --model-weight 0.7 --model-prior-weight 0.04)
  else
    model_args+=(--model-weight 0.0)
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" MOTA_FAST_GRAPH_STATE=1 timeout "${TIMEOUT_MIN}m" \
    "${PYTHON_BIN}" -m mota_rl.beam_decode \
      --target-stage "${target}" \
      --start-route "${start_route}" \
      "${model_args[@]}" \
      --beam-width "${beam_width}" \
      --action-top-k "${top_k}" \
      --max-steps "${max_steps}" \
      --action-bias-weight 1.25 \
      --potential-weight 0.09 \
      --fast-score-weight 0.0035 \
      --distance-weight 1.45 \
      --path-step-penalty 0.003 \
      --macro-step-penalty 0.018 \
      --revisit-penalty 1.4 \
      --success-bonus 1100 \
      --max-per-diversity-key 12 \
      --continue-after-success \
      --success-patience 20 \
      --adaptive-weights-json "${adaptive}" \
      --route-out "${RUN_DIR}/${out_dir}/${tag}_route.jsonl" \
      --summary-out "${RUN_DIR}/${out_dir}/${tag}_summary.json" \
      --trace-out "${RUN_DIR}/${out_dir}/${tag}_trace.jsonl" \
      > "${RUN_DIR}/logs/${tag}.log" 2>&1 || true
}

STRICT_ROUTES=()
while IFS= read -r route; do
  STRICT_ROUTES+=("${route}")
done < <(find "${STAGED_RUN}/staged" -name 'strict_seed_*.jsonl' ! -name '*_dataset.jsonl' -size +0c | sort)
printf '%s\n' "${STRICT_ROUTES[@]}" > "${RUN_DIR}/strict_prefixes.txt"
if [ "${#STRICT_ROUTES[@]}" -eq 0 ]; then
  echo "[mt10-repair] no strict staged routes found under ${STAGED_RUN}/staged" >&2
  exit 1
fi

idx=0
for route in "${STRICT_ROUTES[@]}"; do
  idx=$((idx + 1))
  run_decode "$(( (idx - 1) % 4 ))" mt10_resources "${route}" "" mt10 "strict${idx}_nomodel_mt10" 90 260 26 "${ADAPT_MT10}" &
  run_decode "$(( idx % 4 ))" mt10_resources "${route}" "${MT10_MODEL}" mt10 "strict${idx}_ranker_mt10" 90 220 24 "${ADAPT_MT10}" &
done
wait || true

idx=0
for summary in "${RUN_DIR}"/mt10/*_summary.json; do
  [ -f "${summary}" ] || continue
  idx=$((idx + 1))
  route="${summary%_summary.json}_route.jsonl"
  tag="$(basename "${summary%_summary.json}")"
  run_decode "$(( (idx - 1) % 4 ))" red_key "${route}" "" red_key "${tag}_nomodel_red_key" 120 260 28 "${ADAPT_RED}" &
  run_decode "$(( idx % 4 ))" red_key "${route}" "${RED_MODEL}" red_key "${tag}_ranker_red_key" 120 220 26 "${ADAPT_RED}" &
done
wait || true

idx=0
for summary in "${RUN_DIR}"/red_key/*_summary.json; do
  [ -f "${summary}" ] || continue
  idx=$((idx + 1))
  route="${summary%_summary.json}_route.jsonl"
  tag="$(basename "${summary%_summary.json}")"
  run_decode "$(( (idx - 1) % 4 ))" boss "${route}" "" boss "${tag}_nomodel_boss" 140 260 30 "${ADAPT_BOSS}" &
  run_decode "$(( idx % 4 ))" boss "${route}" "${BOSS_MODEL}" boss "${tag}_ranker_boss" 140 220 28 "${ADAPT_BOSS}" &
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
payload = {"run_dir": str(run), "staged_run": "${STAGED_RUN}", "ranker_run": "${RANKER_RUN}", "rows": rows}
(run / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf8")
lines = ["# 4090 MT10 Repair v3 Report", "", f"Run dir: `{run}`", f"Staged run: `${STAGED_RUN}`", ""]
for row in rows:
    if "error" in row:
        lines.append(f"- `{row['path']}` error={row['error']}")
        continue
    data = row["data"]
    final = data.get("final", {})
    flags = final.get("target_flags", {}) if isinstance(final, dict) else {}
    lines.append(
        f"- {row['part']} `{row['path']}` solved={data.get('solved')} strict_success={data.get('strict_success')} "
        f"route={data.get('route_out')} final_hp={final.get('hp')} atk={final.get('atk')} def={final.get('defense')} "
        f"floor={final.get('floor_id') or final.get('floor')} pos=({final.get('x')},{final.get('y')}) "
        f"flags={flags}"
    )
(run / "mt10_repair_v3_report.md").write_text("\\n".join(lines) + "\\n", encoding="utf8")
print(json.dumps({"summary": str(run / "summary.json"), "report": str(run / "mt10_repair_v3_report.md")}, ensure_ascii=False))
PY

date -Is | tee "${RUN_DIR}/finished_at.txt"
