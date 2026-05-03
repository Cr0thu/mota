from __future__ import annotations

import argparse
import json
from pathlib import Path

from mota_env import MotaSimulator, load_game_data
from mota_solver.search import (
    heuristic_scheme_names,
    solve_first10,
    state_summary,
    write_route_jsonl,
)


DEFAULT_ROUTE = Path("artifacts/expert/route_first10.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="artifacts/data/mota_first10.json")
    parser.add_argument("--write-route", action="store_true")
    parser.add_argument("--route-out", default=str(DEFAULT_ROUTE))
    parser.add_argument("--max-expansions", type=int, default=120_000)
    parser.add_argument("--keep-per-parent", type=int, default=80)
    parser.add_argument("--heuristic", choices=heuristic_scheme_names(), default="baseline")
    parser.add_argument(
        "--write-best",
        action="store_true",
        help="Write the best legal partial route even if the boss is not defeated.",
    )
    args = parser.parse_args()

    sim = MotaSimulator(load_game_data(args.data))
    result = solve_first10(
        sim,
        max_expansions=args.max_expansions,
        keep_per_parent=args.keep_per_parent,
        heuristic_scheme=args.heuristic,
    )
    print(
        json.dumps(
            {
                "solved": result.solved,
                "heuristic": args.heuristic,
                "expansions": result.expansions,
                "route_len": len(result.route),
                "final": state_summary(result.state),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.write_route and (result.solved or args.write_best):
        write_route_jsonl(result.route, args.route_out)
        print(args.route_out)


if __name__ == "__main__":
    main()
