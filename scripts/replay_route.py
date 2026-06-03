from __future__ import annotations

import argparse
import json
from pathlib import Path

from mota_env import MotaSimulator, SimulatorConfig, load_game_data
from mota_solver.search import state_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="artifacts/data/mota_first10.json")
    parser.add_argument("--route", default="artifacts/expert/route_first10.jsonl")
    parser.add_argument("--continue-after-boss", action="store_true")
    args = parser.parse_args()

    sim = MotaSimulator(
        load_game_data(args.data),
        SimulatorConfig(stop_on_boss=not args.continue_after_boss),
    )
    state = sim.reset()
    rows = [
        json.loads(line)
        for line in Path(args.route).read_text(encoding="utf8").splitlines()
        if line.strip()
    ]
    for row in rows:
        transition = sim.apply_macro_action(state, row["action"])
        if not transition.ok or state.dead:
            raise SystemExit(
                json.dumps(
                    {
                        "failed_at": row["index"],
                        "action": row["action"],
                        "message": transition.message,
                        "state": state_summary(state),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
    print(
        json.dumps(
            {
                "steps": len(rows),
                "solved": bool(state.flags.get("10f战胜骷髅队长")),
                "done": bool(state.done),
                "final": state_summary(state),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
