from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mota_env import MotaSimulator, SimulatorConfig, load_game_data
from mota_env.rewards import stage_complete, stage_names
from mota_solver.go_explore import GoExploreConfig, run_go_explore
from mota_solver.resource_planner import dump_resource_planner_outputs


def load_reward_weights(path: str) -> dict[str, object] | None:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf8"))


def action_signature(action: dict[str, object]) -> tuple[object, ...]:
    return (
        action.get("kind"),
        tuple(action.get("target", [])),
        action.get("floor"),
        tuple(action.get("loc", [])),
        action.get("shop"),
        action.get("label"),
    )


def find_action_index(actions: list[dict[str, object]], route_action: dict[str, object]) -> int | None:
    signature = action_signature(route_action)
    for index, action in enumerate(actions):
        if action_signature(action) == signature:
            return index
    target = tuple(route_action.get("target", []))
    label = route_action.get("label")
    for index, action in enumerate(actions):
        if tuple(action.get("target", [])) == target and action.get("label") == label:
            return index
    for index, action in enumerate(actions):
        if action.get("label") == label:
            return index
    return None


def apply_start_route(sim: MotaSimulator, args: argparse.Namespace):
    state = sim.reset()
    if not args.start_route:
        return state
    rows = [
        json.loads(line)
        for line in Path(args.start_route).read_text(encoding="utf8").splitlines()
        if line.strip()
    ]
    for prefix_index, row in enumerate(rows[: max(0, int(args.start_route_max_actions)) or None]):
        if args.start_route_stop_stage and stage_complete(sim, state, args.start_route_stop_stage):
            break
        actions = sim.macro_actions(state)
        action_index = find_action_index(actions, row.get("action") or {})
        if action_index is None:
            raise RuntimeError(
                f"start route action not legal at row {prefix_index}: "
                f"{(row.get('action') or {}).get('label')}"
            )
        transition = sim.apply_macro_action(state, actions[action_index])
        if not transition.ok or state.dead or state.done:
            raise RuntimeError(f"start route failed at row {prefix_index}: {transition.message}")
    return state


def run_one(args: argparse.Namespace, *, relaxed: bool, out_dir: Path) -> dict[str, object]:
    sim = MotaSimulator(
        load_game_data(args.data),
        SimulatorConfig(
            allow_negative_hp=relaxed,
            min_hp=args.relaxed_min_hp,
        ),
    )
    start_state = apply_start_route(sim, args)
    trace_path = out_dir / "go_explore_trace.jsonl"
    result = run_go_explore(
        sim,
        GoExploreConfig(
            target_stage=args.target_stage,
            iterations=args.iterations,
            rollout_steps=args.rollout_steps,
            archive_top_k=args.archive_top_k,
            seed=args.seed,
            allow_relaxed=relaxed,
            trace_limit=args.trace_limit,
            reward_weights=load_reward_weights(args.reward_weights_file),
            reward_potential_weight=args.reward_potential_weight,
            reward_stage_mode=args.reward_stage_mode,
            candidate_top_k=args.candidate_top_k,
            temperature=args.temperature,
            novelty_bonus=args.novelty_bonus,
            revisit_penalty=args.revisit_penalty,
            use_stage_action_filter=not args.disable_stage_action_filter,
        ),
        trace_path=trace_path,
        start_state=start_state,
    )
    outputs = dump_resource_planner_outputs(result, out_dir)
    return {
        "mode": "relaxed" if relaxed else "strict",
        "out_dir": str(out_dir),
        "trace": str(trace_path),
        "solved": result.solved,
        "iterations": result.expansions,
        "archive_cells": result.archive_cells,
        "route_length": len(result.route),
        "best_summary": result.best_summary,
        "outputs": outputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="artifacts/data/mota_first10.json")
    parser.add_argument("--out-dir", default="artifacts/runs/go_explore")
    parser.add_argument(
        "--protocol",
        choices=("manual_guided", "no_agent_manual", "auto_reward_search", "self_generated_curriculum"),
        default="manual_guided",
    )
    parser.add_argument("--target-stage", choices=stage_names(), default="shield")
    parser.add_argument("--iterations", type=int, default=2_000)
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--archive-top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--trace-limit", type=int, default=20_000)
    parser.add_argument("--relaxed-min-hp", type=int, default=-1200)
    parser.add_argument("--reward-weights-file", default="")
    parser.add_argument("--reward-potential-weight", type=float, default=0.02)
    parser.add_argument("--reward-stage-mode", choices=("current", "target"), default="current")
    parser.add_argument("--start-route", default="")
    parser.add_argument("--start-route-stop-stage", choices=("", *stage_names()), default="")
    parser.add_argument("--start-route-max-actions", type=int, default=10_000)
    parser.add_argument("--candidate-top-k", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.65)
    parser.add_argument("--novelty-bonus", type=float, default=150.0)
    parser.add_argument("--revisit-penalty", type=float, default=0.08)
    parser.add_argument("--disable-stage-action-filter", action="store_true")
    parser.add_argument(
        "--mode",
        choices=("strict", "relaxed", "both"),
        default="strict",
        help="strict forbids HP<=0; relaxed allows negative HP only for exploration structure.",
    )
    args = parser.parse_args()
    if args.protocol in {"no_agent_manual", "auto_reward_search"}:
        args.disable_stage_action_filter = True
        if args.start_route:
            raise SystemExit(f"--protocol {args.protocol} forbids --start-route")
        if args.protocol == "no_agent_manual" and args.reward_weights_file:
            raise SystemExit("--protocol no_agent_manual forbids --reward-weights-file")
        if args.protocol == "auto_reward_search" and not args.reward_weights_file:
            raise SystemExit("--protocol auto_reward_search requires --reward-weights-file")
    if args.protocol == "self_generated_curriculum":
        args.disable_stage_action_filter = True
        if not args.start_route:
            raise SystemExit("--protocol self_generated_curriculum requires --start-route")
        if args.reward_weights_file:
            raise SystemExit("--protocol self_generated_curriculum forbids --reward-weights-file")
        route_path = Path(args.start_route)
        if not route_path.exists():
            raise SystemExit(f"start route does not exist: {route_path}")
        lowered = str(route_path).lower()
        if any(token in lowered for token in ("hp403", "manual", "alpha4090_boss_success")):
            raise SystemExit(f"self_generated_curriculum rejects suspicious route path: {route_path}")

    base = Path(args.out_dir)
    base.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, object]] = []
    if args.mode in {"strict", "both"}:
        runs.append(run_one(args, relaxed=False, out_dir=base / "strict" if args.mode == "both" else base))
    if args.mode in {"relaxed", "both"}:
        runs.append(run_one(args, relaxed=True, out_dir=base / "relaxed" if args.mode == "both" else base))
    manifest = {
        "data": args.data,
        "target_stage": args.target_stage,
        "iterations": args.iterations,
        "rollout_steps": args.rollout_steps,
        "archive_top_k": args.archive_top_k,
        "seed": args.seed,
        "mode": args.mode,
        "runs": runs,
    }
    (base / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
