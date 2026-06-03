from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mota_env import MotaSimulator, load_game_data
from mota_env.rewards import (
    DEFAULT_LEARNABLE_GLOBAL_WEIGHTS,
    LearnableStageReward,
    Rewarder,
    boss_route_margin,
    current_stage_name,
    stage_complete,
)
from mota_solver.search import state_summary


TUNABLE_FACTORS = (
    "reachable_resource_value",
    "blocked_resource_pressure",
    "reachable_enemy_damage_drop",
    "reachable_enemy_cost",
    "reachable_unlock_value",
    "potion_need",
    "key_buffer",
    "mt10_resource_progress",
    "boss_margin",
    "stat_marginal_damage_boost",
)


def sample_global_weights(rng: random.Random) -> dict[str, float]:
    weights = dict(DEFAULT_LEARNABLE_GLOBAL_WEIGHTS)
    for factor in TUNABLE_FACTORS:
        weights[factor] = DEFAULT_LEARNABLE_GLOBAL_WEIGHTS.get(factor, 1.0) * rng.uniform(0.35, 3.0)
    return weights


def greedy_rollout(
    sim: MotaSimulator,
    rewarder: Rewarder,
    max_macros: int,
) -> dict[str, Any]:
    state = sim.reset()
    route: list[dict[str, Any]] = []
    visited = {sim.state_key(state)}
    solved_stages: set[str] = set()
    loop_count = 0
    deadend_count = 0

    for _ in range(max_macros):
        if state.dead or state.done:
            break
        stage = current_stage_name(sim, state)
        candidates = []
        for action in sim.macro_actions(state):
            child = state.clone()
            transition = sim.apply_macro_action(child, action)
            if not transition.ok:
                continue
            breakdown = rewarder.score(sim, state, child, action, transition)
            revisit = sim.state_key(child) in visited
            loop_penalty = 800.0 if revisit else 0.0
            if ("upFloor" in action.get("label", "") or "downFloor" in action.get("label", "")) and route:
                last_floor = route[-1].get("before", {}).get("floor")
                if child.floor_id == last_floor:
                    loop_penalty += 4000.0
            candidates.append((breakdown.total - loop_penalty, action.get("label", ""), action, child, breakdown, revisit))
        if not candidates:
            deadend_count += 1
            break
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        _score, _label, action, child, breakdown, revisit = candidates[0]
        before_summary = state_summary(state)
        if revisit:
            loop_count += 1
        if stage_complete(sim, child, stage):
            solved_stages.add(stage)
        state = child
        visited.add(sim.state_key(state))
        route.append(
            {
                "index": len(route),
                "stage": stage,
                "action": action,
                "reward": breakdown.total,
                "reward_components": breakdown.components,
                "before": before_summary,
                "after": state_summary(state),
            }
        )

    return {
        "solved": bool(state.flags.get("10f战胜骷髅队长")),
        "route_len": len(route),
        "route": route,
        "final": state_summary(state),
        "boss_margin": boss_route_margin(sim, state),
        "stages_solved": sorted(solved_stages),
        "loop_count": loop_count,
        "deadend_count": deadend_count,
    }


def objective(row: dict[str, Any]) -> float:
    final = row["final"]
    return (
        (200_000.0 if row["solved"] else 0.0)
        + len(row["stages_solved"]) * 15_000.0
        + max(-1500.0, float(row["boss_margin"])) * 15.0
        + max(0.0, float(final.get("hp", 0))) * 2.0
        + float(final.get("atk", 0)) * 600.0
        + float(final.get("def", 0)) * 550.0
        - row["loop_count"] * 3000.0
        - row["deadend_count"] * 10_000.0
        - row["route_len"] * 20.0
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="artifacts/data/mota_first10.json")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--max-macros", type=int, default=140)
    parser.add_argument("--out", default="artifacts/runs/learnable_stage_reward_trials.jsonl")
    parser.add_argument("--best-out", default="artifacts/runs/learnable_stage_reward_best.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best: dict[str, Any] | None = None
    with out_path.open("w", encoding="utf8") as handle:
        for trial in range(args.trials):
            weights = sample_global_weights(rng)
            sim = MotaSimulator(load_game_data(args.data))
            reward = LearnableStageReward(global_weights=weights)
            rewarder = Rewarder("learnable_stage_pbrs")
            rewarder.learnable_stage_reward = reward
            row = greedy_rollout(sim, rewarder, max_macros=args.max_macros)
            row["trial"] = trial
            row["weights"] = weights
            row["objective"] = objective(row)
            handle.write(json.dumps({key: value for key, value in row.items() if key != "route"}, ensure_ascii=False) + "\n")
            if best is None or row["objective"] > best["objective"]:
                best = row
            print(json.dumps({"trial": trial, "objective": row["objective"], "solved": row["solved"]}, ensure_ascii=False))

    if best is not None:
        Path(args.best_out).write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf8")
        print(json.dumps({"best_out": args.best_out, "best_objective": best["objective"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
