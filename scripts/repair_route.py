from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mota_env import MotaSimulator, load_game_data
from mota_env.rewards import mt10_resource_progress, mt10_resources_taken, red_key_route_margin, stage_complete
from mota_rl.train_actor_critic import find_matching_action, load_route
from mota_solver.search import state_summary, write_route_jsonl


PROTECTED_LABEL_TOKENS = (
    "redGem",
    "blueGem",
    "bluePotion MT10",
    "shield1",
    "sword1",
    "buy 5 yellowKey",
    "buy blueKey",
)

INSERT_LABEL_TOKENS = (
    "Potion",
    "yellowKey",
    "blueKey",
    "redKey",
    "event",
    "specialTrader",
)


def replay(sim: MotaSimulator, rows: list[dict[str, Any]]) -> tuple[Any, list[dict[str, Any]], str]:
    state = sim.reset()
    replayed: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        actions = sim.macro_actions(state)
        match = find_matching_action(actions, row["action"])
        if match is None:
            return None, replayed, f"missing action {index}: {row['action'].get('label', '')}"
        action = actions[match]
        before = state.clone()
        transition = sim.apply_macro_action(state, action)
        replayed.append(
            {
                "index": len(replayed),
                "action": action,
                "before": state_summary(before),
                "after": state_summary(state),
                "transition": transition.message,
                "reward": transition.reward,
            }
        )
        if not transition.ok or state.dead:
            return None, replayed, f"bad transition {index}: {transition.message}"
    return state, replayed, ""


def target_done(sim: MotaSimulator, state: Any, target_stage: str) -> bool:
    if state is None or state.hp <= 0:
        return False
    if target_stage == "mt10_first_resource":
        return mt10_resource_progress(sim, state) >= 1
    if target_stage == "mt10_resources":
        return mt10_resources_taken(sim, state)
    return stage_complete(sim, state, target_stage)


def quality(
    sim: MotaSimulator,
    state: Any,
    target_stage: str,
    *,
    min_yellow_keys: int = 0,
) -> float:
    if not target_done(sim, state, target_stage):
        return -1_000_000_000.0
    yellow_keys = state.items.get("yellowKey", 0)
    missing_yellow_keys = max(0, int(min_yellow_keys) - yellow_keys)
    keyed_target_bonus = min(yellow_keys, max(0, int(min_yellow_keys))) * 10_000.0
    keyed_target_penalty = missing_yellow_keys * 50_000.0
    return (
        red_key_route_margin(sim, state) * 200.0
        + state.hp
        + yellow_keys * 80.0
        + state.items.get("blueKey", 0) * 50.0
        + state.items.get("redKey", 0) * 200.0
        + keyed_target_bonus
        - keyed_target_penalty
    )


def prune_route(
    sim: MotaSimulator,
    rows: list[dict[str, Any]],
    target_stage: str,
    max_passes: int,
    min_yellow_keys: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Any]:
    state, replayed, error = replay(sim, rows)
    if error:
        raise SystemExit(error)
    removed: list[dict[str, Any]] = []
    current = rows[:]
    current_quality = quality(sim, state, target_stage, min_yellow_keys=min_yellow_keys)

    for pass_index in range(1, max_passes + 1):
        changed = False
        index = 0
        while index < len(current):
            label = current[index]["action"].get("label", "")
            if any(token in label for token in PROTECTED_LABEL_TOKENS):
                index += 1
                continue
            trial = current[:index] + current[index + 1 :]
            candidate_state, candidate_replayed, error = replay(sim, trial)
            candidate_quality = quality(
                sim,
                candidate_state,
                target_stage,
                min_yellow_keys=min_yellow_keys,
            )
            if not error and candidate_quality >= current_quality:
                removed.append(
                    {
                        "pass": pass_index,
                        "index": index,
                        "label": label,
                        "old_quality": current_quality,
                        "new_quality": candidate_quality,
                        "old_hp": state.hp,
                        "new_hp": candidate_state.hp,
                        "old_red_key_route_margin": red_key_route_margin(sim, state),
                        "new_red_key_route_margin": red_key_route_margin(sim, candidate_state),
                    }
                )
                current = trial
                state = candidate_state
                replayed = candidate_replayed
                current_quality = candidate_quality
                changed = True
                continue
            index += 1
        if not changed:
            break
    return replayed, removed, state


def try_insertions(
    sim: MotaSimulator,
    rows: list[dict[str, Any]],
    target_stage: str,
    rounds: int,
    min_yellow_keys: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Any]:
    state, replayed, error = replay(sim, rows)
    if error:
        raise SystemExit(error)
    accepted: list[dict[str, Any]] = []
    current = replayed

    for round_index in range(1, rounds + 1):
        states = [sim.reset().clone()]
        prefix_state = sim.reset()
        ok = True
        for index, row in enumerate(current):
            actions = sim.macro_actions(prefix_state)
            match = find_matching_action(actions, row["action"])
            if match is None:
                ok = False
                break
            sim.apply_macro_action(prefix_state, actions[match])
            states.append(prefix_state.clone())
        if not ok:
            break

        best: tuple[float, int, dict[str, Any], Any, list[dict[str, Any]]] | None = None
        base_quality = quality(sim, state, target_stage, min_yellow_keys=min_yellow_keys)
        for index, candidate_state in enumerate(states):
            next_label = current[index]["action"].get("label", "") if index < len(current) else ""
            for action in sim.macro_actions(candidate_state):
                label = action.get("label", "")
                if label == next_label:
                    continue
                if not any(token in label for token in INSERT_LABEL_TOKENS):
                    continue
                trial = current[:index] + [{"action": action}] + current[index:]
                trial_state, trial_replayed, error = replay(sim, trial)
                trial_quality = quality(
                    sim,
                    trial_state,
                    target_stage,
                    min_yellow_keys=min_yellow_keys,
                )
                if not error and trial_quality > base_quality:
                    candidate = (trial_quality, index, action, trial_state, trial_replayed)
                    if best is None or candidate[0] > best[0]:
                        best = candidate
        if best is None:
            break
        best_quality, index, action, state, current = best
        accepted.append(
            {
                "round": round_index,
                "insert_index": index,
                "label": action.get("label", ""),
                "quality": best_quality,
                "hp": state.hp,
                "red_key_route_margin": red_key_route_margin(sim, state),
            }
        )
    return current, accepted, state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="artifacts/data/mota_first10.json")
    parser.add_argument("--route-in", required=True)
    parser.add_argument("--route-out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--target-stage", default="mt10_resources")
    parser.add_argument("--prune-passes", type=int, default=4)
    parser.add_argument("--insert-rounds", type=int, default=2)
    parser.add_argument(
        "--min-yellow-keys",
        type=int,
        default=0,
        help="Soft route-quality target for preserving yellow keys after the target stage.",
    )
    args = parser.parse_args()

    sim = MotaSimulator(load_game_data(args.data))
    rows = load_route(args.route_in)
    pruned, removed, pruned_state = prune_route(
        sim,
        rows,
        args.target_stage,
        args.prune_passes,
        min_yellow_keys=args.min_yellow_keys,
    )
    repaired, inserted, final_state = try_insertions(
        sim,
        pruned,
        args.target_stage,
        args.insert_rounds,
        min_yellow_keys=args.min_yellow_keys,
    )

    route_out = Path(args.route_out)
    summary_out = Path(args.summary_out)
    write_route_jsonl(repaired, route_out)
    summary = {
        "route_in": args.route_in,
        "route_out": str(route_out),
        "target_stage": args.target_stage,
        "input_len": len(rows),
        "pruned_len": len(pruned),
        "final_len": len(repaired),
        "removed": removed,
        "inserted": inserted,
        "min_yellow_keys": args.min_yellow_keys,
        "pruned_final": state_summary(pruned_state),
        "final": state_summary(final_state),
        "red_key_route_margin": red_key_route_margin(sim, final_state),
        "quality": quality(
            sim,
            final_state,
            args.target_stage,
            min_yellow_keys=args.min_yellow_keys,
        ),
        "target_done": target_done(sim, final_state, args.target_stage),
    }
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
