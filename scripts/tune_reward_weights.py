from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from mota_env import MotaSimulator, SimulatorConfig, load_game_data
from mota_env.rewards import DEFAULT_STAGE_POTENTIAL_WEIGHTS, red_key_taken, yellow_guard_margin
from mota_solver.staged import solve_staged_first10


def sample_weights(rng: random.Random) -> dict[str, float]:
    sampled: dict[str, float] = {}
    for key, default in DEFAULT_STAGE_POTENTIAL_WEIGHTS.items():
        if key in {"deadend", "key_pressure"}:
            sampled[key] = default * rng.uniform(0.7, 2.5)
        elif key.startswith("stage_") or key in {"threshold", "boss_margin"}:
            sampled[key] = default * rng.uniform(0.5, 3.0)
        else:
            sampled[key] = default * rng.uniform(0.4, 1.8)
    return sampled


def objective(row: dict[str, Any]) -> float:
    solved_bonus = 100_000.0 if row["solved"] else 0.0
    stage_bonus = 8_000.0 * sum(1 for stage in row["stages"] if stage["solved"])
    final = row["final"]
    hp_score = max(0, final["hp"]) * 1.5
    atk_def_score = final["atk"] * 120.0 + final["def"] * 110.0
    boss_margin = final.get("boss_margin", -10_000)
    boss_score = max(-2_000, boss_margin) * 8.0
    red_key_bonus = 20_000.0 if final.get("red_key_taken") else 0.0
    guard_score = max(-2_000, final.get("yellow_guard_margin", -2_000)) * 12.0
    expansion_penalty = row["expansions"] * 0.03
    return solved_bonus + stage_bonus + red_key_bonus + guard_score + hp_score + atk_def_score + boss_score - expansion_penalty


def run_trial(
    data_path: str,
    weights: dict[str, float],
    max_expansions_per_stage: int,
    keep_per_parent: int,
    frontier_size: int,
    seed: int,
    allow_negative_hp: bool,
    relaxed_min_hp: int,
) -> dict[str, Any]:
    sim = MotaSimulator(
        load_game_data(data_path),
        SimulatorConfig(allow_negative_hp=allow_negative_hp, min_hp=relaxed_min_hp),
    )
    result = solve_staged_first10(
        sim,
        max_expansions_per_stage=max_expansions_per_stage,
        keep_per_parent=keep_per_parent,
        frontier_size=frontier_size,
        seed=seed,
        stage_weights=weights,
    )
    boss = sim.damage_info(result.state, "skeletonCaptain")
    final = {
        "floor": result.state.floor_id,
        "x": result.state.x,
        "y": result.state.y,
        "hp": result.state.hp,
        "atk": result.state.atk,
        "def": result.state.defense,
        "money": result.state.money,
        "keys": {
            "yellowKey": result.state.items.get("yellowKey", 0),
            "blueKey": result.state.items.get("blueKey", 0),
            "redKey": result.state.items.get("redKey", 0),
        },
        "boss_margin": -10_000 if boss is None else result.state.hp - boss["damage"],
        "yellow_guard_margin": yellow_guard_margin(sim, result.state),
        "red_key_taken": red_key_taken(sim, result.state),
    }
    row = {
        "seed": seed,
        "weights": weights,
        "solved": result.solved,
        "expansions": result.expansions,
        "route_len": len(result.route),
        "final": final,
        "stages": [
            {
                "stage": summary.stage,
                "solved": summary.solved,
                "expansions": summary.expansions,
                "best": summary.best,
            }
            for summary in result.stage_summaries
        ],
    }
    row["objective"] = objective(row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="artifacts/data/mota_first10.json")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--max-expansions-per-stage", type=int, default=5_000)
    parser.add_argument("--keep-per-parent", type=int, default=80)
    parser.add_argument("--frontier-size", type=int, default=16)
    parser.add_argument("--allow-negative-hp", action="store_true")
    parser.add_argument("--relaxed-min-hp", type=int, default=-2000)
    parser.add_argument("--out", default="artifacts/runs/reward_weight_trials.jsonl")
    parser.add_argument("--best-out", default="artifacts/runs/reward_weight_best.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best: dict[str, Any] | None = None
    with out_path.open("w", encoding="utf8") as handle:
        for trial in range(args.trials):
            weights = sample_weights(rng)
            row = run_trial(
                args.data,
                weights,
                args.max_expansions_per_stage,
                args.keep_per_parent,
                args.frontier_size,
                seed=args.seed + trial,
                allow_negative_hp=args.allow_negative_hp,
                relaxed_min_hp=args.relaxed_min_hp,
            )
            row["trial"] = trial
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if best is None or row["objective"] > best["objective"]:
                best = row
            print(json.dumps({"trial": trial, "objective": row["objective"], "solved": row["solved"]}))

    Path(args.best_out).write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps({"best_out": args.best_out, "best_objective": best["objective"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
