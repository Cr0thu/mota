#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-4090_mt10_first_v17_focused_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-artifacts/runs/${RUN_ID}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TIMEOUT_MIN="${TIMEOUT_MIN:-18}"
TARGET_STAGE="${TARGET_STAGE:-mt10_first_resource}"
mkdir -p "${RUN_DIR}"/{mt10,logs,env}

{
  echo "pwd=$(pwd)"
  echo "python=$(${PYTHON_BIN} --version 2>&1)"
  echo "target_stage=${TARGET_STAGE}"
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
} > "${RUN_DIR}/env/env_check.txt"

routes=(
  "artifacts/runs/4090_guard_chain_v9_20260528_012259/hp_ready/s1_before_mt8_low_gems_route_mid_gems_route_hp_ready_route.jsonl"
  "artifacts/runs/4090_guard_chain_v9_20260528_012259/hp_ready/s1_before_mt8_low_gems_route_mid_gems_route_hp_ready_trace_best_route.jsonl"
)

idx=0
for route in "${routes[@]}"; do
  [ -s "${route}" ] || continue
  idx=$((idx + 1))
  base="$(basename "${route}" .jsonl)"
  CUDA_VISIBLE_DEVICES="$((idx - 1))" MOTA_FAST_GRAPH_STATE=1 timeout "${TIMEOUT_MIN}m" \
    "${PYTHON_BIN}" -m mota_rl.beam_decode \
      --target-stage "${TARGET_STAGE}" \
      --start-route "${route}" \
      --model-weight 0.0 \
      --beam-width 96 \
      --action-top-k 8 \
      --max-steps 70 \
      --action-bias-weight 3.0 \
      --potential-weight 0.10 \
      --fast-score-weight 0.003 \
      --distance-weight 5.0 \
      --path-step-penalty 0.0005 \
      --macro-step-penalty 0.002 \
      --revisit-penalty 1.0 \
      --success-bonus 8000 \
      --max-per-diversity-key 12 \
      --route-out "${RUN_DIR}/mt10/${base}_first_resource_route.jsonl" \
      --summary-out "${RUN_DIR}/mt10/${base}_first_resource_summary.json" \
      --trace-out "${RUN_DIR}/mt10/${base}_first_resource_trace.jsonl" \
      > "${RUN_DIR}/logs/${base}.log" 2>&1 &
done
wait || true

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
run = Path("${RUN_DIR}")
rows = []
for p in sorted((run / "mt10").glob("*_summary.json")):
    try:
        rows.append({"path": str(p), "data": json.loads(p.read_text(encoding="utf8"))})
    except Exception as exc:
        rows.append({"path": str(p), "error": repr(exc)})
(run / "summary.json").write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf8")
print(json.dumps({"run_dir": str(run), "summaries": len(rows)}, ensure_ascii=False))
PY

date -Is | tee "${RUN_DIR}/finished_at.txt"
