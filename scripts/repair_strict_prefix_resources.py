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


RESOURCE_TOKENS = (
    "Potion",
    "redGem",
    "blueGem",
    "yellowKey",
    "blueKey",
    "redKey",
    "sword",
    "shield",
)
BLOCKED_TOKENS = ("upFloor", "downFloor", "fly", "shop", "buy ")


def normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"action": row["action"], "source": row.get("source", "input")} for row in rows]


def replay(
    sim: MotaSimulator,
    rows: list[dict[str, Any]],
    stop: int | None = None,
) -> tuple[Any, list[dict[str, Any]], str]:
    state = sim.reset()
    replayed: list[dict[str, Any]] = []
    limit = len(rows) if stop is None else min(stop, len(rows))
    for index, row in enumerate(rows[:limit]):
        actions = sim.macro_actions(state)
        match = find_matching_action(actions, row["action"])
        if match is None:
            return state, replayed, f"missing action {index}: {row['action'].get('label', '')}"
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
                "source": row.get("source", "replay"),
            }
        )
        if not transition.ok or state.dead:
            return state, replayed, f"bad transition {index}: {transition.message}"
    return state, replayed, ""


def strict_status(sim: MotaSimulator, rows: list[dict[str, Any]]) -> dict[str, Any]:
    state, replayed, error = replay(sim, rows)
    return {
        "steps": len(replayed),
        "error": error,
        "solved": bool(state.flags.get("10f战胜骷髅队长")),
        "state": state_summary(state),
        "failed_label": rows[len(replayed)]["action"].get("label", "") if error and len(replayed) < len(rows) else "",
    }


def relaxed_status(sim: MotaSimulator, rows: list[dict[str, Any]]) -> dict[str, Any]:
    state, replayed, error = replay(sim, rows)
    return {
        "steps": len(replayed),
        "error": error,
        "solved": bool(state.flags.get("10f战胜骷髅队长")),
        "state": state_summary(state),
    }


def score(strict: dict[str, Any], relaxed: dict[str, Any], route_len: int) -> float:
    if not relaxed["solved"] or relaxed["error"]:
        return -1_000_000_000.0
    if strict["solved"]:
        return 1_000_000_000.0 + strict["state"]["hp"] - route_len * 0.02
    return (
        strict["steps"] * 10_000.0
        + strict["state"]["hp"]
        + relaxed["state"]["hp"] * 0.2
        - route_len * 0.02
    )


def strict_state_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    state = row["state"]
    keys = state.get("keys", {})
    return (
        row.get("failed_label", ""),
        state.get("floor"),
        state.get("x"),
        state.get("y"),
        state.get("hp"),
        state.get("atk"),
        state.get("def"),
        keys.get("yellowKey", 0),
        keys.get("blueKey", 0),
        keys.get("redKey", 0),
    )


def candidate_allowed(label: str) -> bool:
    if any(token in label for token in BLOCKED_TOKENS):
        return False
    return any(token in label for token in RESOURCE_TOKENS)


def prefix_states(
    sim: MotaSimulator,
    rows: list[dict[str, Any]],
    upto: int,
) -> list[tuple[int, Any]]:
    result: list[tuple[int, Any]] = [(0, sim.reset())]
    state = sim.reset()
    for index, row in enumerate(rows[:upto]):
        actions = sim.macro_actions(state)
        match = find_matching_action(actions, row["action"])
        if match is None:
            break
        transition = sim.apply_macro_action(state, actions[match])
        if not transition.ok or state.dead:
            break
        result.append((index + 1, state.clone()))
    return result


def one_round(
    strict_sim: MotaSimulator,
    relaxed_sim: MotaSimulator,
    rows: list[dict[str, Any]],
    window: int,
    max_actions_per_prefix: int,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    base_strict = strict_status(strict_sim, rows)
    base_relaxed = relaxed_status(relaxed_sim, rows)
    base_score = score(base_strict, base_relaxed, len(rows))
    fail_index = min(int(base_strict["steps"]), len(rows) - 1)
    start_index = max(0, fail_index - window)

    best: tuple[float, list[dict[str, Any]], dict[str, Any]] | None = None
    states = prefix_states(strict_sim, rows, fail_index)
    for index, state in states:
        if index < start_index:
            continue
        next_label = rows[index]["action"].get("label", "") if index < len(rows) else ""
        actions = [
            action
            for action in strict_sim.macro_actions(state)
            if candidate_allowed(str(action.get("label", "")))
            and str(action.get("label", "")) != str(next_label)
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
            trial = rows[:index] + [{"action": action, "source": "strict_prefix_resource_repair"}] + rows[index:]
            trial_strict = strict_status(strict_sim, trial)
            trial_relaxed = relaxed_status(relaxed_sim, trial)
            if (
                not trial_strict["solved"]
                and strict_state_signature(trial_strict) == strict_state_signature(base_strict)
            ):
                continue
            trial_score = score(trial_strict, trial_relaxed, len(trial))
            if trial_score <= base_score:
                continue
            change = {
                "kind": "insert",
                "index": index,
                "label": action.get("label", ""),
                "base_score": base_score,
                "new_score": trial_score,
                "base_strict": base_strict,
                "new_strict": trial_strict,
                "new_relaxed": trial_relaxed,
            }
            if best is None or trial_score > best[0]:
                best = (trial_score, trial, change)
    if best is None:
        return rows, None
    return normalize_rows(best[1]), best[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair a relaxed route by inserting resources before strict failure.")
    parser.add_argument("--data", default="artifacts/data/mota_first10.json")
    parser.add_argument("--route-in", required=True)
    parser.add_argument("--route-out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--relaxed-min-hp", type=int, default=-25000)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--window", type=int, default=80)
    parser.add_argument("--max-actions-per-prefix", type=int, default=12)
    args = parser.parse_args()

    data = load_game_data(args.data)
    strict_sim = MotaSimulator(data)
    relaxed_sim = MotaSimulator(data, SimulatorConfig(allow_negative_hp=True, min_hp=args.relaxed_min_hp))
    rows = normalize_rows(load_route(args.route_in))
    initial_relaxed = relaxed_status(relaxed_sim, rows)
    if not initial_relaxed["solved"]:
        raise SystemExit(f"Input route must relaxed-solve boss: {initial_relaxed}")

    changes: list[dict[str, Any]] = []
    for round_index in range(1, args.rounds + 1):
        rows, change = one_round(
            strict_sim,
            relaxed_sim,
            rows,
            window=args.window,
            max_actions_per_prefix=args.max_actions_per_prefix,
        )
        if change is None:
            break
        change["round"] = round_index
        changes.append(change)
        if change["new_strict"]["solved"]:
            break

    final_state, final_replayed, error = replay(relaxed_sim, rows)
    if error:
        raise SystemExit(error)
    write_route_jsonl(final_replayed, Path(args.route_out))
    final_strict = strict_status(strict_sim, final_replayed)
    final_relaxed = relaxed_status(relaxed_sim, final_replayed)
    summary = {
        "route_in": args.route_in,
        "route_out": args.route_out,
        "input_len": len(load_route(args.route_in)),
        "final_len": len(final_replayed),
        "initial_strict": strict_status(strict_sim, normalize_rows(load_route(args.route_in))),
        "final_strict": final_strict,
        "final_relaxed": final_relaxed,
        "changes": changes,
        "score": score(final_strict, final_relaxed, len(final_replayed)),
    }
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
