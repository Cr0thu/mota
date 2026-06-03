#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Run this script on the 4090 pod from /root/mota/mota.
# It is intentionally strict: no expert route, no checkpoint, no heuristic prior,
# no hand-shaped stage value, and no reward file are allowed in the trial command.

RUN_ID="${RUN_ID:-4090_no_agent_manual_sweep_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="artifacts/runs/${RUN_ID}"
TARGET_STAGE="${TARGET_STAGE:-sword}"
N_CONFIGS="${N_CONFIGS:-16}"
BASE_SEED="${BASE_SEED:-2026053100}"
CONFIG_SEED="${CONFIG_SEED:-20260531}"
TIMEOUT_MIN="${TIMEOUT_MIN:-180}"
GPUS="${GPUS:-0 1 2 3}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${RUN_DIR}"/{env,configs,jobs,logs,trials,validation}
echo "${RUN_DIR}" > artifacts/runs/latest_no_agent_manual_sweep_run.txt

{
  echo "run_id=${RUN_ID}"
  echo "run_dir=${RUN_DIR}"
  echo "target_stage=${TARGET_STAGE}"
  echo "n_configs=${N_CONFIGS}"
  echo "base_seed=${BASE_SEED}"
  echo "config_seed=${CONFIG_SEED}"
  echo "timeout_min=${TIMEOUT_MIN}"
  echo "gpus=${GPUS}"
  echo "protocol=no_agent_manual"
  echo "selection_metric=target_success_count,strict_replay_ok,final_hp,final_atk,final_def,-macro_steps"
} > "${RUN_DIR}/run_config.txt"

nvidia-smi > "${RUN_DIR}/env/nvidia-smi.txt" 2>&1 || true
"${PYTHON_BIN}" --version > "${RUN_DIR}/env/python.txt" 2>&1 || true
"${PYTHON_BIN}" - <<'PY' > "${RUN_DIR}/env/torch_cuda.json" 2>&1 || true
import json
try:
    import torch
    payload = {
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
except Exception as exc:
    payload = {"error": str(exc)}
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

export RUN_DIR TARGET_STAGE N_CONFIGS BASE_SEED CONFIG_SEED TIMEOUT_MIN PYTHON_BIN
"${PYTHON_BIN}" - <<'PY'
import json
import os
import random
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
target_stage = os.environ["TARGET_STAGE"]
n_configs = int(os.environ["N_CONFIGS"])
base_seed = int(os.environ["BASE_SEED"])
timeout_min = os.environ["TIMEOUT_MIN"]
python_bin = os.environ["PYTHON_BIN"]
rng = random.Random(int(os.environ["CONFIG_SEED"]))

space = {
    "episodes": [64, 96, 128],
    "max_macros": [80, 100, 120],
    "simulations": [64, 96, 128, 192],
    "max_depth": [80, 100, 120],
    "c_puct": [1.1, 1.3, 1.5, 1.8, 2.1],
    "root_dirichlet_alpha": [0.15, 0.3, 0.5, 0.8],
    "root_exploration_fraction": [0.15, 0.25, 0.35, 0.5],
    "action_temperature": [0.25, 0.45, 0.7, 1.0],
    "action_top_k": [0, 8, 16, 32],
    "policy_temperature": [0.8, 1.0, 1.2, 1.5],
    "d_model_heads": [(128, 4), (192, 6), (256, 8)],
    "layers": [3, 4, 5],
    "batch_size": [64, 96, 128],
    "train_steps_per_episode": [2, 4, 8],
    "lr": [3e-5, 5e-5, 1e-4, 2e-4],
}

configs = []
seen = set()
while len(configs) < n_configs:
    d_model, heads = rng.choice(space["d_model_heads"])
    cfg = {
        "trial": len(configs),
        "tag": f"trial_{len(configs):03d}",
        "seed": base_seed + len(configs),
        "target_stage": target_stage,
        "episodes": rng.choice(space["episodes"]),
        "max_macros": rng.choice(space["max_macros"]),
        "simulations": rng.choice(space["simulations"]),
        "max_depth": rng.choice(space["max_depth"]),
        "c_puct": rng.choice(space["c_puct"]),
        "root_dirichlet_alpha": rng.choice(space["root_dirichlet_alpha"]),
        "root_exploration_fraction": rng.choice(space["root_exploration_fraction"]),
        "action_temperature": rng.choice(space["action_temperature"]),
        "action_top_k": rng.choice(space["action_top_k"]),
        "policy_temperature": rng.choice(space["policy_temperature"]),
        "d_model": d_model,
        "heads": heads,
        "layers": rng.choice(space["layers"]),
        "batch_size": rng.choice(space["batch_size"]),
        "train_steps_per_episode": rng.choice(space["train_steps_per_episode"]),
        "lr": rng.choice(space["lr"]),
    }
    key = tuple(sorted(cfg.items()))
    if key in seen:
        continue
    seen.add(key)
    configs.append(cfg)

(run_dir / "configs" / "search_space.json").write_text(
    json.dumps(space, ensure_ascii=False, indent=2),
    encoding="utf8",
)
(run_dir / "configs" / "manifest.json").write_text(
    json.dumps(
        {
            "protocol": "no_agent_manual",
            "target_stage": target_stage,
            "selection_metric": [
                "target_success_count",
                "strict_replay_ok",
                "final_hp",
                "final_atk",
                "final_def",
                "-macro_steps",
            ],
            "configs": configs,
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf8",
)

for cfg in configs:
    trial_dir = run_dir / "trials" / cfg["tag"]
    log_path = run_dir / "logs" / f'{cfg["tag"]}.log'
    job_path = run_dir / "jobs" / f'{cfg["tag"]}.sh'
    trial_dir.mkdir(parents=True, exist_ok=True)
    args = [
        "scripts/train_alpha_mota_stage.py",
        "--protocol", "no_agent_manual",
        "--out-dir", str(trial_dir),
        "--target-stage", cfg["target_stage"],
        "--episodes", str(cfg["episodes"]),
        "--max-macros", str(cfg["max_macros"]),
        "--simulations", str(cfg["simulations"]),
        "--max-depth", str(cfg["max_depth"]),
        "--c-puct", str(cfg["c_puct"]),
        "--max-state-revisits", "1",
        "--selected-policy-target",
        "--device", "cuda",
        "--d-model", str(cfg["d_model"]),
        "--heads", str(cfg["heads"]),
        "--layers", str(cfg["layers"]),
        "--batch-size", str(cfg["batch_size"]),
        "--replay-size", "6000",
        "--train-steps-per-episode", str(cfg["train_steps_per_episode"]),
        "--lr", str(cfg["lr"]),
        "--root-dirichlet-alpha", str(cfg["root_dirichlet_alpha"]),
        "--root-exploration-fraction", str(cfg["root_exploration_fraction"]),
        "--action-temperature", str(cfg["action_temperature"]),
        "--action-top-k", str(cfg["action_top_k"]),
        "--heuristic-prior-mix", "0",
        "--policy-temperature", str(cfg["policy_temperature"]),
        "--seed", str(cfg["seed"]),
    ]
    quoted_args = " ".join("'" + a.replace("'", "'\\''") + "'" for a in args)
    job_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "export PYTHONPATH=src",
                f"TIMEOUT_MIN=\"${{TIMEOUT_MIN:-{timeout_min}}}\"",
                f"echo '[job] {cfg['tag']} gpu='${{GPU_ID}}' started' > '{log_path}'",
                "if command -v timeout >/dev/null 2>&1; then",
                f"  CUDA_VISIBLE_DEVICES=\"${{GPU_ID}}\" timeout \"${{TIMEOUT_MIN}}m\" {python_bin} {quoted_args} >> '{log_path}' 2>&1",
                "else",
                f"  CUDA_VISIBLE_DEVICES=\"${{GPU_ID}}\" {python_bin} {quoted_args} >> '{log_path}' 2>&1",
                "fi",
                f"if [ -f '{trial_dir}/best_route.jsonl' ]; then",
                f"  {python_bin} scripts/replay_route.py --route '{trial_dir}/best_route.jsonl' > '{trial_dir}/strict_replay.json' 2> '{trial_dir}/strict_replay.err' || true",
                f"  {python_bin} scripts/validate_route_constraints.py '{trial_dir}/best_route.jsonl' > '{trial_dir}/constraints.json' 2> '{trial_dir}/constraints.err' || true",
                "fi",
                f"echo '[job] {cfg['tag']} done' >> '{log_path}'",
                "",
            ]
        ),
        encoding="utf8",
    )
    job_path.chmod(0o755)
PY

gpu_list=()
while IFS= read -r gpu_id; do
  [ -n "${gpu_id}" ] && gpu_list+=("${gpu_id}")
done < <(tr ' ' '\n' <<<"${GPUS}" | sed '/^$/d')
if [ "${#gpu_list[@]}" -eq 0 ]; then
  echo "No GPUs configured" >&2
  exit 1
fi

active=0
job_index=0
for job in "${RUN_DIR}"/jobs/trial_*.sh; do
  gpu="${gpu_list[$((job_index % ${#gpu_list[@]}))]}"
  (
    export GPU_ID="${gpu}"
    export TIMEOUT_MIN="${TIMEOUT_MIN}"
    bash "${job}"
  ) &
  active=$((active + 1))
  job_index=$((job_index + 1))
  if [ "${active}" -ge "${#gpu_list[@]}" ]; then
    wait
    active=0
  fi
done
wait

"${PYTHON_BIN}" - <<'PY'
import json
import os
import shutil
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
target_stage = os.environ["TARGET_STAGE"]
rows = []
for trial_dir in sorted((run_dir / "trials").glob("trial_*")):
    episodes_path = trial_dir / "episodes.jsonl"
    episodes = []
    if episodes_path.exists():
        episodes = [json.loads(line) for line in episodes_path.read_text(encoding="utf8").splitlines() if line.strip()]
    successes = [row for row in episodes if row.get("target_success")]
    last = episodes[-1] if episodes else {}
    best_success = max(
        successes,
        key=lambda row: (
            float(row.get("final", {}).get("hp", -10**9)),
            float(row.get("final", {}).get("atk", 0)),
            float(row.get("final", {}).get("def", 0)),
            -int(row.get("macro_steps", 10**9)),
        ),
        default={},
    )
    replay = {}
    replay_path = trial_dir / "strict_replay.json"
    if replay_path.exists():
        try:
            replay = json.loads(replay_path.read_text(encoding="utf8"))
        except Exception:
            replay = {"parse_error": True}
    constraints = {}
    constraints_path = trial_dir / "constraints.json"
    if constraints_path.exists():
        try:
            constraints = json.loads(constraints_path.read_text(encoding="utf8"))
        except Exception:
            constraints = {"parse_error": True}
    final = (best_success or last).get("final", {})
    score = (
        len(successes) * 1_000_000
        + int(bool(replay)) * 10_000
        + int(bool(constraints.get("ok", False))) * 10_000
        + float(final.get("hp", -10**9))
        + float(final.get("atk", 0)) * 100
        + float(final.get("def", 0)) * 100
        - float((best_success or last).get("macro_steps", 10**6)) * 2
    )
    rows.append(
        {
            "trial": trial_dir.name,
            "score": score,
            "target_success_count": len(successes),
            "episodes": len(episodes),
            "best_success": best_success,
            "last": last,
            "strict_replay": replay,
            "constraints": constraints,
            "route": str(trial_dir / "best_route.jsonl") if (trial_dir / "best_route.jsonl").exists() else "",
        }
    )

rows.sort(key=lambda row: row["score"], reverse=True)
(run_dir / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf8")
if rows and rows[0]["route"]:
    src = Path(rows[0]["route"])
    if src.exists():
        shutil.copy2(src, run_dir / "best_auto_selected_route.jsonl")

lines = [
    "# No-Agent-Manual 4090 Sweep Report",
    "",
    f"Run dir: `{run_dir}`",
    f"Protocol: `no_agent_manual`",
    f"Target stage: `{target_stage}`",
    "",
    "This run used a predeclared hyperparameter search space. No expert route, checkpoint, heuristic prior, stage-action filter, hand-shaped stage value, or reward file was used.",
    "",
    "## Top Trials",
    "",
]
for row in rows[:10]:
    best = row.get("best_success") or {}
    final = best.get("final") or (row.get("last") or {}).get("final") or {}
    lines.append(
        f"- `{row['trial']}` score={row['score']:.1f} successes={row['target_success_count']}/{row['episodes']} "
        f"final={final} route=`{row['route']}`"
    )
lines.extend(["", "## Selection Metric", "", "`target_success_count, strict_replay_ok, final_hp, final_atk, final_def, -macro_steps`"])
(run_dir / "no_agent_manual_sweep_report.md").write_text("\n".join(lines) + "\n", encoding="utf8")
print(json.dumps({"run_dir": str(run_dir), "top": rows[0] if rows else None}, ensure_ascii=False, indent=2))
PY
