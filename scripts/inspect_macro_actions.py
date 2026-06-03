from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mota_env import MotaSimulator, load_game_data
from mota_env.rewards import current_stage_name, stage_complete
from mota_solver.search import state_summary


def action_signature(action: dict[str, object]) -> tuple[object, ...]:
    return (
        action.get("kind"),
        tuple(action.get("target", [])),
        action.get("floor"),
        tuple(action.get("loc", [])),
        action.get("shop"),
        action.get("label"),
    )


def find_action(actions: list[dict[str, object]], route_action: dict[str, object]) -> dict[str, object] | None:
    signature = action_signature(route_action)
    for action in actions:
        if action_signature(action) == signature:
            return action
    label = route_action.get("label")
    for action in actions:
        if action.get("label") == label:
            return action
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="artifacts/data/mota_first10.json")
    parser.add_argument("--route", required=True)
    parser.add_argument("--stop-stage", default="")
    parser.add_argument("--filter", default="")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    sim = MotaSimulator(load_game_data(args.data))
    state = sim.reset()
    rows = [
        json.loads(line)
        for line in Path(args.route).read_text(encoding="utf8").splitlines()
        if line.strip()
    ]
    for index, row in enumerate(rows):
        if args.stop_stage and stage_complete(sim, state, args.stop_stage):
            break
        actions = sim.macro_actions(state)
        action = find_action(actions, row.get("action") or {})
        if action is None:
            raise SystemExit(f"route action not legal at {index}: {(row.get('action') or {}).get('label')}")
        transition = sim.apply_macro_action(state, action)
        if not transition.ok:
            raise SystemExit(f"route failed at {index}: {transition.message}")

    actions = sim.macro_actions(state)
    needle = args.filter.lower()
    shown = []
    for index, action in enumerate(actions):
        label = str(action.get("label", ""))
        if needle and needle not in label.lower():
            continue
        shown.append(
            {
                "index": index,
                "label": label,
                "kind": action.get("kind"),
                "target": action.get("target"),
                "shop": action.get("shop"),
                "path_len": len(action.get("path") or []),
            }
        )
        if len(shown) >= args.limit:
            break
    print(
        json.dumps(
            {
                "route": args.route,
                "stop_stage": args.stop_stage,
                "stage": current_stage_name(sim, state),
                "state": state_summary(state),
                "action_count": len(actions),
                "actions": shown,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
