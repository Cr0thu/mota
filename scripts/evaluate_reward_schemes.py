from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from mota_env import MotaSimulator, load_game_data
from mota_env.rewards import Rewarder, reward_scheme_names
from mota_solver.search import state_summary


def replay_route(
    sim: MotaSimulator,
    route_path: Path,
    scheme: str,
) -> dict[str, Any]:
    rewarder = Rewarder(scheme)
    state = sim.reset()
    total = 0.0
    components: dict[str, float] = defaultdict(float)
    rows = [
        json.loads(line)
        for line in route_path.read_text(encoding="utf8").splitlines()
        if line.strip()
    ]
    for idx, row in enumerate(rows):
        before = state.clone()
        transition = sim.apply_macro_action(state, row["action"])
        breakdown = rewarder.score(sim, before, state, row["action"], transition)
        total += breakdown.total
        for key, value in breakdown.components.items():
            components[key] += value
        if not transition.ok or state.dead:
            return {
                "mode": "replay_route",
                "scheme": scheme,
                "failed_at": idx,
                "message": transition.message,
                "total_reward": total,
                "components": dict(sorted(components.items())),
                "final": state_summary(state),
            }
    return {
        "mode": "replay_route",
        "scheme": scheme,
        "route_len": len(rows),
        "total_reward": total,
        "components": dict(sorted(components.items())),
        "solved": bool(state.flags.get("10f战胜骷髅队长")),
        "final": state_summary(state),
    }


def greedy_episode(
    sim: MotaSimulator,
    scheme: str,
    max_macros: int,
) -> dict[str, Any]:
    rewarder = Rewarder(scheme)
    state = sim.reset()
    total = 0.0
    route: list[dict[str, Any]] = []
    components: dict[str, float] = defaultdict(float)
    visited_state_keys = {sim.state_key(state)}

    for _ in range(max_macros):
        if state.done or state.dead:
            break
        candidates = []
        for action in sim.macro_actions(state):
            child = state.clone()
            transition = sim.apply_macro_action(child, action)
            if not transition.ok or child.dead:
                continue
            breakdown = rewarder.score(sim, state, child, action, transition)
            revisit = sim.state_key(child) in visited_state_keys
            candidates.append((breakdown.total, -int(revisit), action["label"], action, child, breakdown))
        if not candidates:
            break
        candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        _, _, _, action, child, breakdown = candidates[0]
        before_summary = state_summary(state)
        state = child
        visited_state_keys.add(sim.state_key(state))
        total += breakdown.total
        for key, value in breakdown.components.items():
            components[key] += value
        route.append(
            {
                "index": len(route),
                "action": action,
                "before": before_summary,
                "after": state_summary(state),
                "reward": breakdown.total,
                "reward_components": breakdown.components,
            }
        )

    return {
        "mode": "greedy_immediate",
        "scheme": scheme,
        "route_len": len(route),
        "total_reward": total,
        "components": dict(sorted(components.items())),
        "solved": bool(state.flags.get("10f战胜骷髅队长")),
        "final": state_summary(state),
        "route": route,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="artifacts/data/mota_first10.json")
    parser.add_argument("--route", default="artifacts/expert/route_first10.jsonl")
    parser.add_argument("--out", default="artifacts/runs/reward_scheme_eval.json")
    parser.add_argument("--max-macros", type=int, default=80)
    parser.add_argument("--schemes", nargs="*", choices=reward_scheme_names(), default=reward_scheme_names())
    args = parser.parse_args()

    route_path = Path(args.route)
    output: dict[str, Any] = {"route": str(route_path), "schemes": {}}
    for scheme in args.schemes:
        sim = MotaSimulator(load_game_data(args.data))
        replay = replay_route(sim, route_path, scheme)
        sim = MotaSimulator(load_game_data(args.data))
        greedy = greedy_episode(sim, scheme, args.max_macros)
        output["schemes"][scheme] = {
            "replay_route": {key: value for key, value in replay.items() if key != "route"},
            "greedy_immediate": {key: value for key, value in greedy.items() if key != "route"},
        }
        route_out = Path(args.out).with_name(f"reward_greedy_{scheme}.jsonl")
        route_out.parent.mkdir(parents=True, exist_ok=True)
        with route_out.open("w", encoding="utf8") as handle:
            for row in greedy["route"]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
