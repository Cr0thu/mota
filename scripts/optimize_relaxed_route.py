from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mota_env import MotaSimulator, SimulatorConfig, load_game_data
from mota_rl.train_actor_critic import find_matching_action, load_route
from mota_solver.search import state_summary, write_route_jsonl


INSERT_TOKENS = (
    "Potion",
    "redGem",
    "blueGem",
    "yellowKey",
    "blueKey",
    "redKey",
    "sword",
    "shield",
)
SKIP_INSERT_TOKENS = (
    "fly",
    "shop",
    "buy ",
    "downFloor",
    "upFloor",
)


def replay(
    sim: MotaSimulator,
    rows: list[dict[str, Any]],
) -> tuple[Any | None, list[dict[str, Any]], str]:
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
                "source": row.get("source", "optimized_replay"),
            }
        )
        if not transition.ok or state.dead:
            return None, replayed, f"bad transition {index}: {transition.message}"
    return state, replayed, ""


def solved(state: Any | None) -> bool:
    return bool(state is not None and state.flags.get("10f战胜骷髅队长"))


def strict_progress(sim: MotaSimulator, rows: list[dict[str, Any]]) -> dict[str, Any]:
    state = sim.reset()
    replayed = 0
    failure = ""
    for index, row in enumerate(rows):
        actions = sim.macro_actions(state)
        match = find_matching_action(actions, row["action"])
        if match is None:
            failure = f"missing action {index}: {row['action'].get('label', '')}"
            break
        transition = sim.apply_macro_action(state, actions[match])
        if not transition.ok or state.dead:
            failure = f"bad transition {index}: {transition.message}"
            break
        replayed += 1
    return {
        "steps": replayed,
        "solved": bool(state.flags.get("10f战胜骷髅队长")),
        "state": state_summary(state),
        "failure": failure,
    }


def route_quality(
    relaxed_state: Any | None,
    strict_row: dict[str, Any],
    route_len: int,
    mode: str,
) -> float:
    if not solved(relaxed_state):
        return -1_000_000_000.0
    if mode == "relaxed_hp":
        return float(relaxed_state.hp) - route_len * 0.02
    strict_steps = int(strict_row["steps"])
    strict_hp = int(strict_row["state"].get("hp", -100_000))
    if strict_row["solved"]:
        return 1_000_000_000.0 + strict_hp - route_len * 0.02
    return strict_steps * 10_000.0 + strict_hp + float(relaxed_state.hp) * 0.2 - route_len * 0.02


def normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        result.append({"action": row["action"], "source": row.get("source", "input")})
    return result


def insert_candidate_allowed(label: str) -> bool:
    if any(token in label for token in SKIP_INSERT_TOKENS):
        return False
    return any(token in label for token in INSERT_TOKENS)


def build_prefix_states(
    sim: MotaSimulator,
    rows: list[dict[str, Any]],
) -> tuple[list[Any], bool]:
    states = [sim.reset()]
    state = sim.reset()
    for row in rows:
        actions = sim.macro_actions(state)
        match = find_matching_action(actions, row["action"])
        if match is None:
            return states, False
        transition = sim.apply_macro_action(state, actions[match])
        if not transition.ok or state.dead:
            return states, False
        states.append(state.clone())
    return states, True


def prune_once(
    sim: MotaSimulator,
    strict_sim: MotaSimulator,
    rows: list[dict[str, Any]],
    base_quality: float,
    mode: str,
) -> tuple[list[dict[str, Any]], float, dict[str, Any] | None]:
    index = 0
    while index < len(rows):
        label = str(rows[index]["action"].get("label", ""))
        trial = rows[:index] + rows[index + 1 :]
        state, replayed, error = replay(sim, trial)
        strict_row = strict_progress(strict_sim, trial)
        trial_quality = route_quality(state, strict_row, len(replayed), mode)
        if not error and trial_quality >= base_quality:
            return (
                normalize_rows(replayed),
                trial_quality,
                {
                    "kind": "remove",
                    "index": index,
                    "label": label,
                    "old_quality": base_quality,
                    "new_quality": trial_quality,
                    "final": state_summary(state),
                    "strict": strict_row,
                },
            )
        index += 1
    return rows, base_quality, None


def insertion_once(
    sim: MotaSimulator,
    strict_sim: MotaSimulator,
    rows: list[dict[str, Any]],
    base_quality: float,
    mode: str,
    max_prefixes: int,
    max_actions_per_prefix: int,
) -> tuple[list[dict[str, Any]], float, dict[str, Any] | None]:
    states, ok = build_prefix_states(sim, rows)
    if not ok:
        return rows, base_quality, None

    best: tuple[float, list[dict[str, Any]], dict[str, Any]] | None = None
    stride = max(1, len(states) // max_prefixes) if max_prefixes > 0 else 1
    candidate_indices = list(range(0, len(states), stride))
    if candidate_indices[-1] != len(states) - 1:
        candidate_indices.append(len(states) - 1)

    for index in candidate_indices:
        state = states[index]
        next_label = str(rows[index]["action"].get("label", "")) if index < len(rows) else ""
        actions = [
            action
            for action in sim.macro_actions(state)
            if insert_candidate_allowed(str(action.get("label", "")))
            and str(action.get("label", "")) != next_label
        ]
        actions.sort(
            key=lambda action: (
                "Gem" in str(action.get("label", "")),
                "Potion" in str(action.get("label", "")),
                "Key" in str(action.get("label", "")),
                -len(action.get("path", [])),
            ),
            reverse=True,
        )
        for action in actions[:max_actions_per_prefix]:
            trial = rows[:index] + [{"action": action, "source": "relaxed_optimizer_insert"}] + rows[index:]
            state_after, replayed, error = replay(sim, trial)
            strict_row = strict_progress(strict_sim, trial)
            trial_quality = route_quality(state_after, strict_row, len(replayed), mode)
            if error or trial_quality <= base_quality:
                continue
            change = {
                "kind": "insert",
                "index": index,
                "label": action.get("label", ""),
                "old_quality": base_quality,
                "new_quality": trial_quality,
                "final": state_summary(state_after),
                "strict": strict_row,
            }
            if best is None or trial_quality > best[0]:
                best = (trial_quality, normalize_rows(replayed), change)
    if best is None:
        return rows, base_quality, None
    return best[1], best[0], best[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prune/insert optimize a relaxed solved route by final HP.")
    parser.add_argument("--data", default="artifacts/data/mota_first10.json")
    parser.add_argument("--route-in", required=True)
    parser.add_argument("--route-out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--relaxed-min-hp", type=int, default=-25000)
    parser.add_argument("--prune-passes", type=int, default=6)
    parser.add_argument("--insert-rounds", type=int, default=8)
    parser.add_argument("--max-prefixes", type=int, default=160)
    parser.add_argument("--max-actions-per-prefix", type=int, default=10)
    parser.add_argument(
        "--quality-mode",
        choices=("strict_progress", "relaxed_hp"),
        default="strict_progress",
    )
    args = parser.parse_args()

    sim = MotaSimulator(
        load_game_data(args.data),
        SimulatorConfig(allow_negative_hp=True, min_hp=args.relaxed_min_hp),
    )
    strict_sim = MotaSimulator(load_game_data(args.data))
    rows = normalize_rows(load_route(args.route_in))
    state, replayed, error = replay(sim, rows)
    if error or not solved(state):
        raise SystemExit(f"Input route must relaxed-solve boss: {error or state_summary(state)}")
    rows = normalize_rows(replayed)
    base_strict = strict_progress(strict_sim, rows)
    base_quality = route_quality(state, base_strict, len(rows), args.quality_mode)

    changes: list[dict[str, Any]] = []
    for pass_index in range(1, args.prune_passes + 1):
        rows, base_quality, change = prune_once(
            sim,
            strict_sim,
            rows,
            base_quality,
            args.quality_mode,
        )
        if change is None:
            break
        change["pass"] = pass_index
        changes.append(change)

    for round_index in range(1, args.insert_rounds + 1):
        rows, base_quality, change = insertion_once(
            sim,
            strict_sim,
            rows,
            base_quality,
            args.quality_mode,
            max_prefixes=args.max_prefixes,
            max_actions_per_prefix=args.max_actions_per_prefix,
        )
        if change is None:
            break
        change["round"] = round_index
        changes.append(change)
        while True:
            rows, base_quality, prune_change = prune_once(
                sim,
                strict_sim,
                rows,
                base_quality,
                args.quality_mode,
            )
            if prune_change is None:
                break
            prune_change["round"] = round_index
            changes.append(prune_change)

    final_state, final_replayed, final_error = replay(sim, rows)
    if final_error:
        raise SystemExit(final_error)
    write_route_jsonl(final_replayed, Path(args.route_out))
    summary = {
        "route_in": args.route_in,
        "route_out": args.route_out,
        "input_len": len(load_route(args.route_in)),
        "final_len": len(final_replayed),
        "changes": changes,
        "final": state_summary(final_state),
        "strict": strict_progress(strict_sim, final_replayed),
        "solved": solved(final_state),
        "quality": route_quality(
            final_state,
            strict_progress(strict_sim, final_replayed),
            len(final_replayed),
            args.quality_mode,
        ),
        "quality_mode": args.quality_mode,
    }
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
