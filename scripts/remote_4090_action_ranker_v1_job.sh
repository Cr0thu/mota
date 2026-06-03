#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-4090_action_ranker_v1_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-artifacts/runs/${RUN_ID}}"
STAGED_RUN="${STAGED_RUN:-artifacts/runs/4090_pure_search_rl_20260527_104212}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EPOCHS="${EPOCHS:-32}"
TIMEOUT_MIN="${TIMEOUT_MIN:-55}"

mkdir -p "${RUN_DIR}"/{env,logs,ranker,beam,validation}
echo "${RUN_DIR}" > artifacts/runs/latest_remote_action_ranker_run.txt

{
  echo "pwd=$(pwd)"
  echo "python=$(${PYTHON_BIN} --version 2>&1)"
  echo "staged_run=${STAGED_RUN}"
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
  ${PYTHON_BIN} - <<'PY'
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

ROUTES=()
while IFS= read -r route; do
  ROUTES+=("${route}")
done < <(find "${STAGED_RUN}/staged" \( -name 'strict_seed_*.jsonl' -o -name 'relaxed_seed_*.jsonl' \) ! -name '*_dataset.jsonl' -size +0c | sort)
printf '%s\n' "${ROUTES[@]}" > "${RUN_DIR}/routes.txt"
if [ "${#ROUTES[@]}" -eq 0 ]; then
  echo "[ranker] no routes found under ${STAGED_RUN}/staged" >&2
  exit 1
fi

run_ranker() {
  local gpu="$1"
  local target="$2"
  local seed="$3"
  local max_steps="$4"
  local tag="$5"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 MOTA_FAST_GRAPH_STATE=1 timeout "${TIMEOUT_MIN}m" \
    "${PYTHON_BIN}" -m mota_rl.train_action_ranker \
      --routes "${ROUTES[@]}" \
      --target-stage "${target}" \
      --out-dir "${RUN_DIR}/ranker/${tag}" \
      --epochs "${EPOCHS}" \
      --eval-every 4 \
      --eval-episodes 6 \
      --max-steps "${max_steps}" \
      --seed "${seed}" \
      --device cuda \
      --allow-negative-hp \
      --relaxed-min-hp -2500 \
      --hidden 160 \
      --lr 5e-4 \
      --margin 2.0 \
      --margin-coef 0.45 \
      --prior-weight 0.15 \
      > "${RUN_DIR}/logs/${tag}.log" 2>&1 &
}

run_ranker 0 shield 20267001 120 ranker_shield_seed20267001
run_ranker 1 mt10_resources 20267002 210 ranker_mt10_resources_seed20267002
run_ranker 2 red_key 20267003 240 ranker_red_key_seed20267003
run_ranker 3 boss 20267004 280 ranker_boss_seed20267004
wait || true

for model in "${RUN_DIR}"/ranker/*/best_model.pt; do
  [ -f "${model}" ] || continue
  tag="$(basename "$(dirname "${model}")")"
  target="boss"
  case "${tag}" in
    *shield*) target="shield" ;;
    *mt10_resources*) target="mt10_resources" ;;
    *red_key*) target="red_key" ;;
  esac
  "${PYTHON_BIN}" -m mota_rl.beam_decode \
    --model-path "${model}" \
    --target-stage "${target}" \
    --beam-width 80 \
    --action-top-k 10 \
    --max-steps 260 \
    --model-weight 1.4 \
    --model-prior-weight 0.10 \
    --action-bias-weight 0.8 \
    --potential-weight 0.05 \
    --continue-after-success \
    --success-patience 12 \
    --route-out "${RUN_DIR}/beam/${tag}_route.jsonl" \
    --summary-out "${RUN_DIR}/beam/${tag}_summary.json" \
    > "${RUN_DIR}/logs/${tag}_beam.log" 2>&1 || true
done

for route in "${RUN_DIR}"/beam/*_route.jsonl "${RUN_DIR}"/ranker/*/*route*.jsonl; do
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
for p in sorted((run / "ranker").glob("*/eval.json")):
    rows.append({"kind": "ranker_eval", "path": str(p), "data": json.loads(p.read_text(encoding="utf8"))})
for p in sorted((run / "beam").glob("*_summary.json")):
    rows.append({"kind": "beam", "path": str(p), "data": json.loads(p.read_text(encoding="utf8"))})
payload = {"run_dir": str(run), "rows": rows}
(run / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf8")
lines = ["# 4090 Action Ranker Report", "", f"Run dir: `{run}`", ""]
for row in rows:
    data = row["data"]
    if row["kind"] == "ranker_eval":
        lines.append(f"- `{row['path']}` target_successes={data.get('target_successes')} strict_successes={data.get('strict_successes')}")
    else:
        lines.append(f"- `{row['path']}` solved={data.get('solved')} strict_success={data.get('strict_success')} final={data.get('final')}")
(run / "action_ranker_report.md").write_text("\\n".join(lines) + "\\n", encoding="utf8")
print(json.dumps({"summary": str(run / "summary.json"), "report": str(run / "action_ranker_report.md")}, ensure_ascii=False))
PY

date -Is | tee "${RUN_DIR}/finished_at.txt"
