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


def replay_rows(sim: MotaSimulator, rows: list[dict[str, Any]]) -> tuple[Any, list[dict[str, Any]]]:
    state = sim.reset()
    route: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        actions = sim.macro_actions(state)
        match = find_matching_action(actions, row["action"])
        if match is None:
            raise SystemExit(f"Cannot replay input route at step {index}: {row['action'].get('label', '')}")
        action = actions[match]
        before = state.clone()
        transition = sim.apply_macro_action(state, action)
        route.append(
            {
                "index": len(route),
                "action": action,
                "before": state_summary(before),
                "after": state_summary(state),
                "transition": transition.message,
                "reward": transition.reward,
                "source": row.get("source", "input_route"),
            }
        )
        if not transition.ok:
            raise SystemExit(f"Input route failed at step {index}: {transition.message}")
    return state, route


def main() -> None:
    parser = argparse.ArgumentParser(description="Extend a route by exact macro-action labels.")
    parser.add_argument("--data", default="artifacts/data/mota_first10.json")
    parser.add_argument("--input-route", required=True)
    parser.add_argument("--output-route", required=True)
    parser.add_argument("--labels-json", required=True, help="JSON list of exact macro-action labels.")
    parser.add_argument("--allow-negative-hp", action="store_true")
    parser.add_argument("--relaxed-min-hp", type=int, default=-15000)
    args = parser.parse_args()

    labels = json.loads(args.labels_json)
    if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
        raise SystemExit("--labels-json must be a JSON list of strings.")

    sim = MotaSimulator(
        load_game_data(args.data),
        SimulatorConfig(allow_negative_hp=args.allow_negative_hp, min_hp=args.relaxed_min_hp),
    )
    state, route = replay_rows(sim, load_route(args.input_route))
    for label in labels:
        actions = sim.macro_actions(state)
        action = next((candidate for candidate in actions if candidate.get("label") == label), None)
        if action is None:
            available = [candidate.get("label", "") for candidate in actions]
            raise SystemExit(
                f"Cannot find label {label!r} from {state.floor_id}:{state.x},{state.y}. "
                f"Available: {available}"
            )
        before = state.clone()
        transition = sim.apply_macro_action(state, action)
        route.append(
            {
                "index": len(route),
                "action": action,
                "before": state_summary(before),
                "after": state_summary(state),
                "transition": transition.message,
                "reward": transition.reward,
                "source": "forced_label_extension",
            }
        )
        if not transition.ok:
            raise SystemExit(f"Forced label failed: {label}: {transition.message}")

    write_route_jsonl(route, Path(args.output_route))
    print(json.dumps({"output_route": args.output_route, "steps": len(route), "final": state_summary(state)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
