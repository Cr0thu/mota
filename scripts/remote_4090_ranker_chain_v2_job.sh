#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-4090_ranker_chain_v2_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-artifacts/runs/${RUN_ID}}"
SOURCE_RUN="${SOURCE_RUN:-artifacts/runs/4090_action_ranker_v1_20260527_152511}"
STAGED_RUN="${STAGED_RUN:-artifacts/runs/4090_pure_search_rl_20260527_104212}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TIMEOUT_MIN="${TIMEOUT_MIN:-45}"

mkdir -p "${RUN_DIR}"/{env,logs,chain,repair,validation}
echo "${RUN_DIR}" > artifacts/runs/latest_remote_ranker_chain_run.txt

{
  echo "pwd=$(pwd)"
  echo "python=$(${PYTHON_BIN} --version 2>&1)"
  echo "source_run=${SOURCE_RUN}"
  echo "staged_run=${STAGED_RUN}"
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
  model="$(find "${SOURCE_RUN}/ranker" -path "*${needle}*/model.pt" -print 2>/dev/null | sort | head -1 || true)"
  if [ -z "${model}" ]; then
    model="$(find "${SOURCE_RUN}/ranker" -path "*${needle}*/best_model.pt" -print 2>/dev/null | sort | head -1 || true)"
  fi
  printf '%s' "${model}"
}

SHIELD_MODEL="$(model_for shield)"
MT10_MODEL="$(model_for mt10_resources)"
RED_KEY_MODEL="$(model_for red_key)"
BOSS_MODEL="$(model_for boss)"

{
  echo "shield_model=${SHIELD_MODEL}"
  echo "mt10_resources_model=${MT10_MODEL}"
  echo "red_key_model=${RED_KEY_MODEL}"
  echo "boss_model=${BOSS_MODEL}"
} | tee "${RUN_DIR}/models.txt"

if [ -z "${SHIELD_MODEL}" ] || [ -z "${MT10_MODEL}" ] || [ -z "${RED_KEY_MODEL}" ] || [ -z "${BOSS_MODEL}" ]; then
  echo "[chain] missing one or more ranker models under ${SOURCE_RUN}" >&2
  exit 1
fi

ADAPT_GENERAL='{"atk_delta":90.0,"def_delta":90.0,"yellow_key_delta":28.0,"blue_key_delta":50.0,"yellow_key_level":8.0,"blue_key_level":18.0}'
ADAPT_RED='{"atk_delta":120.0,"def_delta":120.0,"yellow_key_delta":35.0,"blue_key_delta":60.0,"yellow_key_level":12.0,"blue_key_level":24.0,"guard_margin_delta":0.55,"guard_margin_level":0.006,"red_key_route_margin_delta":0.55,"red_key_route_margin_level":0.005,"last_yellow_key_spent":120.0}'
ADAPT_BOSS='{"atk_delta":145.0,"def_delta":135.0,"yellow_key_delta":18.0,"blue_key_delta":38.0,"yellow_key_level":5.0,"blue_key_level":12.0,"hp_delta":0.06,"guard_margin_delta":0.35,"red_key_route_margin_delta":0.35}'

decode_stage() {
  local gpu="$1"
  local target="$2"
  local model="$3"
  local start_route="$4"
  local tag="$5"
  local max_steps="$6"
  local beam_width="$7"
  local action_top_k="$8"
  local adaptive="$9"
  local args=()
  if [ -n "${start_route}" ]; then
    args+=(--start-route "${start_route}")
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" MOTA_FAST_GRAPH_STATE=1 timeout "${TIMEOUT_MIN}m" \
    "${PYTHON_BIN}" -m mota_rl.beam_decode \
      --model-path "${model}" \
      --target-stage "${target}" \
      "${args[@]}" \
      --beam-width "${beam_width}" \
      --action-top-k "${action_top_k}" \
      --max-steps "${max_steps}" \
      --model-weight 1.15 \
      --model-prior-weight 0.08 \
      --action-bias-weight 0.9 \
      --potential-weight 0.06 \
      --fast-score-weight 0.002 \
      --distance-weight 1.2 \
      --path-step-penalty 0.006 \
      --macro-step-penalty 0.035 \
      --revisit-penalty 1.2 \
      --success-bonus 700 \
      --max-per-diversity-key 8 \
      --continue-after-success \
      --success-patience 14 \
      --adaptive-weights-json "${adaptive}" \
      --route-out "${RUN_DIR}/chain/${tag}_route.jsonl" \
      --summary-out "${RUN_DIR}/chain/${tag}_summary.json" \
      --trace-out "${RUN_DIR}/chain/${tag}_trace.jsonl" \
      > "${RUN_DIR}/logs/${tag}.log" 2>&1 || true
}

decode_stage 0 shield "${SHIELD_MODEL}" "" 01_shield 150 160 16 "${ADAPT_GENERAL}"
decode_stage 1 mt10_resources "${MT10_MODEL}" "${RUN_DIR}/chain/01_shield_route.jsonl" 02_mt10_resources 230 180 18 "${ADAPT_GENERAL}"
decode_stage 2 red_key "${RED_KEY_MODEL}" "${RUN_DIR}/chain/02_mt10_resources_route.jsonl" 03_red_key 250 200 20 "${ADAPT_RED}"
decode_stage 3 boss "${BOSS_MODEL}" "${RUN_DIR}/chain/03_red_key_route.jsonl" 04_boss 280 220 22 "${ADAPT_BOSS}"

repair_one() {
  local gpu="$1"
  local route="$2"
  local tag="$3"
  CUDA_VISIBLE_DEVICES="${gpu}" MOTA_FAST_GRAPH_STATE=1 timeout "${TIMEOUT_MIN}m" \
    "${PYTHON_BIN}" -m mota_rl.beam_decode \
      --model-path "${MT10_MODEL}" \
      --target-stage mt10_resources \
      --start-route "${route}" \
      --beam-width 220 \
      --action-top-k 22 \
      --max-steps 120 \
      --model-weight 1.05 \
      --model-prior-weight 0.08 \
      --action-bias-weight 1.05 \
      --potential-weight 0.08 \
      --fast-score-weight 0.003 \
      --distance-weight 1.4 \
      --path-step-penalty 0.004 \
      --macro-step-penalty 0.025 \
      --success-bonus 900 \
      --max-per-diversity-key 10 \
      --continue-after-success \
      --success-patience 18 \
      --adaptive-weights-json "${ADAPT_GENERAL}" \
      --route-out "${RUN_DIR}/repair/${tag}_mt10_resources_route.jsonl" \
      --summary-out "${RUN_DIR}/repair/${tag}_mt10_resources_summary.json" \
      --trace-out "${RUN_DIR}/repair/${tag}_mt10_resources_trace.jsonl" \
      > "${RUN_DIR}/logs/${tag}_mt10_resources_repair.log" 2>&1 || true
}

index=0
while IFS= read -r route; do
  index=$((index + 1))
  repair_one "$(( (index - 1) % 4 ))" "${route}" "strict_seed_${index}" &
done < <(find "${STAGED_RUN}/staged" -name 'strict_seed_*.jsonl' ! -name '*_dataset.jsonl' -size +0c | sort)
wait || true

for route in "${RUN_DIR}"/chain/*_route.jsonl "${RUN_DIR}"/repair/*_route.jsonl; do
  [ -f "${route}" ] || continue
  safe="$(echo "${route#${RUN_DIR}/}" | tr '/.' '__')"
  "${PYTHON_BIN}" scripts/replay_route.py --route "${route}" > "${RUN_DIR}/validation/${safe}_replay.json" || true
  "${PYTHON_BIN}" scripts/validate_route_constraints.py "${route}" > "${RUN_DIR}/validation/${safe}_constraints.json" || true
done

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
run = Path("${RUN_DIR}")
rows = []
for p in sorted((run / "chain").glob("*_summary.json")) + sorted((run / "repair").glob("*_summary.json")):
    try:
        rows.append({"path": str(p), "data": json.loads(p.read_text(encoding="utf8"))})
    except Exception as exc:
        rows.append({"path": str(p), "error": repr(exc)})
payload = {"run_dir": str(run), "source_run": "${SOURCE_RUN}", "staged_run": "${STAGED_RUN}", "rows": rows}
(run / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf8")
lines = ["# 4090 Ranker Chain v2 Report", "", f"Run dir: `{run}`", f"Source run: `${SOURCE_RUN}`", ""]
for row in rows:
    if "error" in row:
        lines.append(f"- `{row['path']}` error={row['error']}")
        continue
    data = row["data"]
    final = data.get("final", {})
    lines.append(
        f"- `{row['path']}` solved={data.get('solved')} strict_success={data.get('strict_success')} "
        f"route={data.get('route_out')} final_hp={final.get('hp')} final_atk={final.get('atk')} "
        f"final_def={final.get('defense')} final_floor={final.get('floor_id')} final_pos=({final.get('x')},{final.get('y')})"
    )
(run / "ranker_chain_v2_report.md").write_text("\\n".join(lines) + "\\n", encoding="utf8")
print(json.dumps({"summary": str(run / "summary.json"), "report": str(run / "ranker_chain_v2_report.md")}, ensure_ascii=False))
PY

date -Is | tee "${RUN_DIR}/finished_at.txt"
