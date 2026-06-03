from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from mota_env import MotaSimulator, SimulatorConfig, load_game_data
from mota_env.rewards import stage_names
from mota_solver.search import write_route_jsonl
from mota_solver.staged import solve_staged_first10, write_stage_dataset_jsonl


def load_stage_weights(inline_json: str, file_path: str) -> dict[str, float] | None:
    if inline_json and file_path:
        raise SystemExit("Use either --stage-weights-json or --stage-weights-file, not both.")
    raw = ""
    if inline_json:
        raw = inline_json
    elif file_path:
        raw = Path(file_path).read_text(encoding="utf8")
    if not raw:
        return None
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise SystemExit("Stage weights must be a JSON object.")
    return {str(key): float(value) for key, value in data.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="artifacts/data/mota_first10.json")
    parser.add_argument("--max-expansions-per-stage", type=int, default=20_000)
    parser.add_argument("--keep-per-parent", type=int, default=80)
    parser.add_argument("--frontier-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--trace-out", default="")
    parser.add_argument("--trace-limit", type=int, default=50_000)
    parser.add_argument("--route-out", default="artifacts/expert/route_first10_staged.jsonl")
    parser.add_argument("--dataset-out", default="artifacts/stage_dataset/staged_route.jsonl")
    parser.add_argument("--write-route", action="store_true")
    parser.add_argument("--write-dataset", action="store_true")
    parser.add_argument(
        "--stop-stage",
        default="",
        choices=("", *stage_names()),
        help="Stop after this stage instead of continuing through the full first-10 task.",
    )
    parser.add_argument(
        "--stage-weights-json",
        default="",
        help="Inline JSON object overriding stage potential weights.",
    )
    parser.add_argument(
        "--stage-weights-file",
        default="",
        help="JSON file overriding stage potential weights.",
    )
    parser.add_argument(
        "--allow-negative-hp",
        action="store_true",
        help="Relax battle legality so search can explore routes below 0 HP.",
    )
    parser.add_argument(
        "--relaxed-min-hp",
        type=int,
        default=-2000,
        help="Lower HP bound used with --allow-negative-hp.",
    )
    args = parser.parse_args()

    stage_weights = load_stage_weights(args.stage_weights_json, args.stage_weights_file)
    sim = MotaSimulator(
        load_game_data(args.data),
        SimulatorConfig(allow_negative_hp=args.allow_negative_hp, min_hp=args.relaxed_min_hp),
    )
    result = solve_staged_first10(
        sim,
        max_expansions_per_stage=args.max_expansions_per_stage,
        keep_per_parent=args.keep_per_parent,
        frontier_size=args.frontier_size,
        seed=args.seed,
        trace_path=args.trace_out or None,
        trace_limit=args.trace_limit,
        stage_weights=stage_weights,
        stop_stage=args.stop_stage or None,
    )
    payload = {
        "solved": result.solved,
        "relaxed": args.allow_negative_hp,
        "relaxed_min_hp": args.relaxed_min_hp if args.allow_negative_hp else None,
        "stage_weights": stage_weights,
        "expansions": result.expansions,
        "route_len": len(result.route),
        "route_out": args.route_out if args.write_route else None,
        "dataset_out": args.dataset_out if args.write_dataset else None,
        "final": {
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
            "flags": {
                key: result.state.flags.get(key)
                for key in ["10f机关", "10f战胜骷髅队长", "nowWeapon", "nowShield"]
                if key in result.state.flags
            },
        },
        "stages": [
            {
                "stage": summary.stage,
                "label": summary.label,
                "solved": summary.solved,
                "expansions": summary.expansions,
                "frontier_size": summary.frontier_size,
                "best": summary.best,
            }
            for summary in result.stage_summaries
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.write_route:
        write_route_jsonl(result.route, Path(args.route_out))
        print(f"wrote route: {args.route_out}", file=sys.stderr)
    if args.write_dataset:
        sim = MotaSimulator(
            load_game_data(args.data),
            SimulatorConfig(allow_negative_hp=args.allow_negative_hp, min_hp=args.relaxed_min_hp),
        )
        write_stage_dataset_jsonl(sim, result.route, args.dataset_out, solved=result.solved)
        print(f"wrote dataset: {args.dataset_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
