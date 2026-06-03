from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mota_env import MotaSimulator, SimulatorConfig, load_game_data
from mota_env.rewards import red_key_route_margin, red_key_taken, yellow_guard_margin
from mota_rl.beam_decode import (
    BeamSearchResult,
    action_context_key,
    beam_search,
    enrich_summary,
    load_policy_model,
)
from mota_rl.train_actor_critic import load_route
from mota_solver.search import write_route_jsonl


@dataclass
class BacktrackDecision:
    rollback_step: int
    reason: str
    penalty_key: tuple[Any, ...] | None = None
    label_penalty: tuple[str, float] | None = None
    reward_weights: dict[str, float] | None = None
    source_step: int | None = None
    source_label: str = ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeated red-key beam decode with explicit rollback penalties.")
    parser.add_argument("--data", default="artifacts/data/mota_first10.json")
    parser.add_argument("--start-route", default="artifacts/expert/route_first10_staged_shield_redkey_100k_v3.jsonl")
    parser.add_argument("--model-path", default="artifacts/runs/action_ranker_red_key_v1/best_model.pt")
    parser.add_argument("--target-stage", default="red_key")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--initial-rollback", type=int, default=243)
    parser.add_argument("--min-rollback", type=int, default=160)
    parser.add_argument("--device", default="cpu", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--allow-negative-hp", action="store_true")
    parser.add_argument("--relaxed-min-hp", type=int, default=-800)
    parser.add_argument("--out-dir", default="artifacts/runs/backtrack_red_key")
    parser.add_argument("--route-dir", default="artifacts/expert/backtrack_red_key")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    route_dir = Path(args.route_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    route_dir.mkdir(parents=True, exist_ok=True)
    base_rows = load_route(args.start_route)
    model_bundle = load_policy_model(args.model_path, hidden=128, requested_device=args.device)
    action_penalties: dict[tuple[Any, ...], float] = {}
    label_penalties: dict[str, float] = {}
    adaptive_weights: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    rollback_step = min(args.initial_rollback, len(base_rows))
    best_score = -10**18
    best_route: list[dict[str, Any]] = []
    best_summary: dict[str, Any] = {}

    aggregate_path = out_dir / "trials.jsonl"
    with aggregate_path.open("w", encoding="utf8") as handle:
        for trial in range(1, args.trials + 1):
            params = trial_params(trial)
            sim = MotaSimulator(
                load_game_data(args.data),
                SimulatorConfig(
                    allow_negative_hp=args.allow_negative_hp,
                    min_hp=args.relaxed_min_hp,
                ),
            )
            route_out = route_dir / f"trial_{trial:02d}.jsonl"
            result = beam_search(
                sim=sim,
                target_stage=args.target_stage,
                model_bundle=model_bundle,
                start_route_path=args.start_route,
                start_route_max_steps=rollback_step,
                action_penalties=action_penalties,
                label_penalties=label_penalties,
                adaptive_weights=adaptive_weights,
                **params,
            )
            write_route_jsonl(result.route, route_out)
            final = enrich_summary(sim, result.node.state)
            decision = choose_backtrack(
                sim=sim,
                result=result,
                current_rollback=rollback_step,
                min_rollback=args.min_rollback,
            )
            apply_decision(decision, action_penalties, label_penalties, adaptive_weights)
            score = route_score(sim, result)
            row = {
                "trial": trial,
                "rollback_step": rollback_step,
                "params": params,
                "solved": result.solved,
                "strict_success": result.strict_success,
                "score": score,
                "final": final,
                "route_out": str(route_out),
                "expanded_nodes": result.expanded_nodes,
                "generated_nodes": result.generated_nodes,
                "decision": decision.__dict__,
                "action_penalties": len(action_penalties),
                "label_penalties": dict(label_penalties),
                "adaptive_weights": dict(adaptive_weights),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps(slim_row(row), ensure_ascii=False), flush=True)
            rows.append(row)
            if score > best_score:
                best_score = score
                best_route = result.route
                best_summary = row
                write_route_jsonl(best_route, route_dir / "best.jsonl")
                (out_dir / "best.json").write_text(
                    json.dumps(best_summary, ensure_ascii=False, indent=2),
                    encoding="utf8",
                )
            if result.strict_success:
                break
            rollback_step = next_rollback_step(
                decision=decision,
                trial=trial,
                current=rollback_step,
                min_rollback=args.min_rollback,
                route_len=len(base_rows),
            )

    summary = {
        "trials_requested": args.trials,
        "trials_completed": len(rows),
        "strict_successes": sum(int(row["strict_success"]) for row in rows),
        "solved_relaxed_or_stage": sum(int(row["solved"]) for row in rows),
        "best": best_summary,
        "aggregate_path": str(aggregate_path),
        "best_route": str(route_dir / "best.jsonl"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def trial_params(trial: int) -> dict[str, Any]:
    beam_options = [12, 16, 20, 24, 28, 32]
    topk_options = [8, 10, 12, 14, 16]
    distance_options = [1.8, 2.1, 2.4, 2.7, 3.0]
    fast_options = [0.001, 0.0015, 0.002, 0.003]
    potential_options = [0.0, 0.003, 0.005, 0.008]
    return {
        "beam_width": beam_options[(trial - 1) % len(beam_options)],
        "action_top_k": topk_options[(trial * 2) % len(topk_options)],
        "max_steps": 70 + (trial % 5) * 10,
        "model_weight": 1.0,
        "model_prior_weight": 0.0,
        "action_bias_weight": 1.1 + (trial % 4) * 0.15,
        "potential_weight": potential_options[trial % len(potential_options)],
        "fast_score_weight": fast_options[trial % len(fast_options)],
        "distance_weight": distance_options[trial % len(distance_options)],
        "path_step_penalty": 0.01,
        "macro_step_penalty": 0.06,
        "success_bonus": 500.0,
        "max_per_diversity_key": 2 + (trial % 3),
    }


def choose_backtrack(
    sim: MotaSimulator,
    result: BeamSearchResult,
    current_rollback: int,
    min_rollback: int,
) -> BacktrackDecision:
    route = result.route
    for row in route:
        label = row["action"].get("label", "")
        after = row.get("after", {})
        if "yellowGuard" in label and after.get("floor") == "MT8" and after.get("hp", 0) < 280:
            return decision_from_row(
                row,
                rollback=max(min_rollback, int(row["index"]) - 10),
                reason="hp_short_after_yellow_guard",
                label_penalty=("yellowGuard", 4.0),
                reward_weights={
                    "hp_delta": 0.20,
                    "atk_delta": 130.0,
                    "def_delta": 100.0,
                    "guard_margin_delta": 0.65,
                    "guard_margin_level": 0.015,
                    "red_key_route_margin_delta": 0.75,
                    "red_key_route_margin_level": 0.02,
                },
            )
    route_margin = red_key_route_margin(sim, result.node.state)
    if (
        result.node.state.floor_id in {"MT8", "MT9"}
        and not result.solved
        and not red_key_taken(sim, result.node.state)
        and route_margin < 0
    ):
        stage_rows = [row for row in route if int(row["index"]) >= max(0, current_rollback - 12)]
        search_rows = list(reversed(stage_rows or route))
        for row in search_rows:
            label = row["action"].get("label", "")
            if any(token in label for token in ("redGem", "blueGem", "Potion", "yellowDoor", "blueDoor", "fight")):
                return decision_from_row(
                    row,
                    rollback=max(min_rollback, int(row["index"]) - 8),
                    reason="red_key_route_margin_short",
                    label_penalty=(label, 5.0)
                    if any(token in label for token in ("yellowDoor", "blueDoor"))
                    else None,
                    reward_weights={
                        "hp_delta": 0.32,
                        "atk_delta": 210.0,
                        "def_delta": 195.0,
                        "red_key_route_margin_delta": 1.8,
                        "red_key_route_margin_level": 0.06,
                        "yellow_key_level": 12.0,
                    },
                )
    if result.node.state.items.get("yellowKey", 0) == 0 and not red_key_taken(sim, result.node.state):
        for row in reversed(route):
            label = row["action"].get("label", "")
            if "yellowDoor" in label or "blueDoor" in label:
                return decision_from_row(
                    row,
                    rollback=max(min_rollback, int(row["index"]) - 10),
                    reason="key_depleted_before_red_key",
                    label_penalty=(label, 8.0),
                    reward_weights={
                        "yellow_key_delta": 110.0,
                        "yellow_key_level": 10.0,
                        "blue_key_delta": 80.0,
                        "last_yellow_key_spent": 80.0,
                        "red_key_route_margin_delta": 0.45,
                    },
                )
    if result.node.state.floor_id == "MT10" and result.node.state.items.get("yellowKey", 0) == 0:
        for row in reversed(route):
            label = row["action"].get("label", "")
            if "yellowDoor" in label:
                return decision_from_row(
                    row,
                    rollback=max(min_rollback, int(row["index"]) - 12),
                    reason="mt10_resource_path_spent_last_key",
                    label_penalty=(label, 10.0),
                    reward_weights={
                        "yellow_key_delta": 120.0,
                        "yellow_key_level": 12.0,
                        "last_yellow_key_spent": 120.0,
                        "hp_delta": 0.06,
                        "red_key_route_margin_delta": 0.55,
                    },
                )
    if result.node.state.floor_id in {"MT8", "MT9"} and not result.solved:
        for row in reversed(route):
            label = row["action"].get("label", "")
            if "downFloor" in label or "upFloor" in label:
                return decision_from_row(
                    row,
                    rollback=max(min_rollback, int(row["index"]) - 8),
                    reason="stuck_floor_navigation",
                    label_penalty=(label, 2.0),
                    reward_weights={
                        "atk_delta": 45.0,
                        "def_delta": 40.0,
                        "yellow_key_level": 4.0,
                        "red_key_route_margin_delta": 0.30,
                    },
                )
    fallback = max(min_rollback, current_rollback - 5)
    return BacktrackDecision(
        rollback_step=fallback,
        reason="fallback_earlier_prefix",
        reward_weights={
            "hp_delta": 0.04,
            "atk_delta": 35.0,
            "def_delta": 35.0,
            "red_key_route_margin_delta": 0.20,
        },
    )


def decision_from_row(
    row: dict[str, Any],
    rollback: int,
    reason: str,
    label_penalty: tuple[str, float] | None = None,
    reward_weights: dict[str, float] | None = None,
) -> BacktrackDecision:
    return BacktrackDecision(
        rollback_step=rollback,
        reason=reason,
        penalty_key=context_key_from_summary(row["before"], row["action"]),
        label_penalty=label_penalty,
        reward_weights=reward_weights,
        source_step=int(row["index"]),
        source_label=row["action"].get("label", ""),
    )


def context_key_from_summary(summary: dict[str, Any], action: dict[str, Any]) -> tuple[Any, ...]:
    keys = summary.get("keys", {})
    flags = summary.get("flags", {})
    return (
        summary.get("floor"),
        int(summary.get("x", 0)) // 2,
        int(summary.get("y", 0)) // 2,
        summary.get("atk"),
        summary.get("def"),
        keys.get("yellowKey", 0),
        keys.get("blueKey", 0),
        keys.get("redKey", 0),
        bool(flags.get("8")),
        bool(flags.get("10f机关")),
        action.get("label", ""),
    )


def apply_decision(
    decision: BacktrackDecision,
    action_penalties: dict[tuple[Any, ...], float],
    label_penalties: dict[str, float],
    adaptive_weights: dict[str, float],
) -> None:
    if decision.penalty_key is not None:
        increment = 24.0 if decision.reason == "red_key_route_margin_short" else 14.0
        action_penalties[decision.penalty_key] = action_penalties.get(decision.penalty_key, 0.0) + increment
    if decision.label_penalty is not None:
        label, value = decision.label_penalty
        label_penalties[label] = label_penalties.get(label, 0.0) + value
    if decision.reward_weights:
        for key, value in decision.reward_weights.items():
            adaptive_weights[key] = adaptive_weights.get(key, 0.0) * 0.85 + value


def next_rollback_step(
    decision: BacktrackDecision,
    trial: int,
    current: int,
    min_rollback: int,
    route_len: int,
) -> int:
    proposed = decision.rollback_step
    if trial % 6 == 0:
        proposed = current - 8
    elif trial % 4 == 0:
        proposed = min(route_len, proposed + 5)
    return max(min_rollback, min(route_len, proposed))


def route_score(sim: MotaSimulator, result: BeamSearchResult) -> float:
    state = result.node.state
    score = 0.0
    if result.solved:
        score += 1_000_000.0
    if result.strict_success:
        score += 1_000_000.0
    score += state.hp * 3.0
    score += state.atk * 700.0 + state.defense * 500.0
    score += state.items.get("yellowKey", 0) * 1000.0
    score += state.items.get("redKey", 0) * 100_000.0
    score += yellow_guard_margin(sim, state) * 20.0
    score += red_key_route_margin(sim, state) * 32.0
    if state.floor_id == "MT8":
        score += 5000.0
    if state.floor_id == "MT10":
        score += 2500.0
    return score


def slim_row(row: dict[str, Any]) -> dict[str, Any]:
    final = row["final"]
    return {
        "trial": row["trial"],
        "rollback_step": row["rollback_step"],
        "solved": row["solved"],
        "strict_success": row["strict_success"],
        "score": row["score"],
        "final": {
            "floor": final.get("floor"),
            "x": final.get("x"),
            "y": final.get("y"),
            "hp": final.get("hp"),
            "atk": final.get("atk"),
            "def": final.get("def"),
            "keys": final.get("keys"),
            "yellow_guard_margin": final.get("yellow_guard_margin"),
            "red_key_route_margin": final.get("red_key_route_margin"),
        },
        "decision": row["decision"],
        "expanded_nodes": row["expanded_nodes"],
        "generated_nodes": row["generated_nodes"],
    }


if __name__ == "__main__":
    main()
