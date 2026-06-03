#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Run this script on the 4090 pod from /root/mota/mota.
# This is not a pure binary-reward run. It is an automatic reward-search line:
# reward weights are sampled by a fixed program and selected by a predeclared
# objective, then used as an MCTS leaf value. No expert/manual route or
# checkpoint is used.

RUN_ID="${RUN_ID:-4090_auto_reward_alpha_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="artifacts/runs/${RUN_ID}"
TARGET_STAGE="${TARGET_STAGE:-sword}"
REWARD_TRIALS="${REWARD_TRIALS:-128}"
REWARD_SEED="${REWARD_SEED:-2026053300}"
ALPHA_CONFIGS="${ALPHA_CONFIGS:-8}"
ALPHA_BASE_SEED="${ALPHA_BASE_SEED:-2026053400}"
CONFIG_SEED="${CONFIG_SEED:-20260534}"
TIMEOUT_MIN="${TIMEOUT_MIN:-180}"
GPUS="${GPUS:-0 1 2 3}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${RUN_DIR}"/{env,reward,configs,jobs,logs,trials,validation}
echo "${RUN_DIR}" > artifacts/runs/latest_auto_reward_alpha_run.txt

{
  echo "run_id=${RUN_ID}"
  echo "run_dir=${RUN_DIR}"
  echo "target_stage=${TARGET_STAGE}"
  echo "reward_trials=${REWARD_TRIALS}"
  echo "reward_seed=${REWARD_SEED}"
  echo "alpha_configs=${ALPHA_CONFIGS}"
  echo "alpha_base_seed=${ALPHA_BASE_SEED}"
  echo "config_seed=${CONFIG_SEED}"
  echo "timeout_min=${TIMEOUT_MIN}"
  echo "gpus=${GPUS}"
  echo "protocol=auto_reward_search"
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

echo "[auto-reward] tuning reward weights" | tee "${RUN_DIR}/logs/reward_tune.log"
"${PYTHON_BIN}" scripts/tune_learnable_stage_reward.py \
  --trials "${REWARD_TRIALS}" \
  --seed "${REWARD_SEED}" \
  --max-macros 160 \
  --out "${RUN_DIR}/reward/reward_trials.jsonl" \
  --best-out "${RUN_DIR}/reward/reward_best_raw.json" \
  >> "${RUN_DIR}/logs/reward_tune.log" 2>&1

export RUN_DIR TARGET_STAGE ALPHA_CONFIGS ALPHA_BASE_SEED CONFIG_SEED TIMEOUT_MIN PYTHON_BIN
"${PYTHON_BIN}" - <<'PY'
import json
import os
import random
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
raw = json.loads((run_dir / "reward" / "reward_best_raw.json").read_text(encoding="utf8"))
reward_config = {
    "source": "tune_learnable_stage_reward.py",
    "protocol": "auto_reward_search",
    "selection_objective": "fixed greedy-rollout objective in tune_learnable_stage_reward.py",
    "weights": {
        "gamma": 0.99,
        "global_weights": raw.get("weights", {}),
        "stage_weights": {},
    },
    "raw_best": {k: v for k, v in raw.items() if k != "route"},
}
(run_dir / "reward" / "reward_config.json").write_text(
    json.dumps(reward_config, ensure_ascii=False, indent=2),
    encoding="utf8",
)

target_stage = os.environ["TARGET_STAGE"]
n_configs = int(os.environ["ALPHA_CONFIGS"])
base_seed = int(os.environ["ALPHA_BASE_SEED"])
timeout_min = os.environ["TIMEOUT_MIN"]
python_bin = os.environ["PYTHON_BIN"]
rng = random.Random(int(os.environ["CONFIG_SEED"]))

space = {
    "episodes": [48, 64, 96],
    "max_macros": [80, 100, 120],
    "simulations": [48, 64, 96, 128],
    "max_depth": [80, 100, 120],
    "c_puct": [1.1, 1.3, 1.5, 1.8],
    "root_dirichlet_alpha": [0.15, 0.3, 0.5],
    "root_exploration_fraction": [0.15, 0.25, 0.35],
    "action_temperature": [0.25, 0.45, 0.7],
    "action_top_k": [0, 8, 16],
    "policy_temperature": [0.8, 1.0, 1.2],
    "reward_value_scale": [12000.0, 18000.0, 25000.0, 35000.0],
    "d_model_heads": [(128, 4), (192, 6), (256, 8)],
    "layers": [3, 4],
    "batch_size": [64, 96, 128],
    "train_steps_per_episode": [2, 4, 8],
    "lr": [3e-5, 5e-5, 1e-4],
}

configs = []
for trial in range(n_configs):
    d_model, heads = rng.choice(space["d_model_heads"])
    configs.append(
        {
            "trial": trial,
            "tag": f"trial_{trial:03d}",
            "seed": base_seed + trial,
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
            "reward_value_scale": rng.choice(space["reward_value_scale"]),
            "d_model": d_model,
            "heads": heads,
            "layers": rng.choice(space["layers"]),
            "batch_size": rng.choice(space["batch_size"]),
            "train_steps_per_episode": rng.choice(space["train_steps_per_episode"]),
            "lr": rng.choice(space["lr"]),
        }
    )

(run_dir / "configs" / "alpha_search_space.json").write_text(
    json.dumps(space, ensure_ascii=False, indent=2),
    encoding="utf8",
)
(run_dir / "configs" / "manifest.json").write_text(
    json.dumps(
        {
            "protocol": "auto_reward_search",
            "target_stage": target_stage,
            "reward_config": str(run_dir / "reward" / "reward_config.json"),
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
        "--protocol", "auto_reward_search",
        "--reward-weights-file", str(run_dir / "reward" / "reward_config.json"),
        "--reward-value-scale", str(cfg["reward_value_scale"]),
        "--reward-value-stage-mode", "target",
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
    "# Auto Reward Alpha Report",
    "",
    f"Run dir: `{run_dir}`",
    "Protocol: `auto_reward_search`",
    f"Target stage: `{target_stage}`",
    "",
    "Reward weights were selected by `scripts/tune_learnable_stage_reward.py` before Alpha training. No expert route, manual route, checkpoint, heuristic prior, or stage-action coordinate filter was used.",
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
(run_dir / "auto_reward_alpha_report.md").write_text("\n".join(lines) + "\n", encoding="utf8")
print(json.dumps({"run_dir": str(run_dir), "top": rows[0] if rows else None}, ensure_ascii=False, indent=2))
PY
