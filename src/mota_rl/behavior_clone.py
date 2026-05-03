from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_route(path: str | Path) -> list[dict[str, Any]]:
    route_path = Path(path)
    rows: list[dict[str, Any]] = []
    with route_path.open(encoding="utf8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def route_features(row: dict[str, Any]) -> tuple[int, ...]:
    before = row["before"]
    keys = before["keys"]
    floor_idx = int(before["floor"][2:])
    return (
        floor_idx,
        before["hp"],
        before["atk"],
        before["def"],
        before["money"],
        keys.get("yellowKey", 0),
        keys.get("blueKey", 0),
        keys.get("redKey", 0),
    )


def write_table_policy(route: list[dict[str, Any]], out_path: str | Path) -> None:
    """Write a deterministic nearest-state table baseline when torch is unavailable."""

    table = [
        {
            "features": route_features(row),
            "label": row["action"]["label"],
            "action": row["action"],
        }
        for row in route
    ]
    payload = {
        "kind": "table_behavior_clone",
        "num_steps": len(route),
        "action_histogram": Counter(row["action"]["label"] for row in route),
        "table": table,
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=dict), encoding="utf8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", default="artifacts/expert/route_first10.jsonl")
    parser.add_argument("--out", default="artifacts/runs/bc_policy.json")
    args = parser.parse_args()

    route = load_route(args.route)
    if not route:
        raise SystemExit(f"No route rows found in {args.route}")
    write_table_policy(route, args.out)
    print(
        json.dumps(
            {
                "route": args.route,
                "out": args.out,
                "steps": len(route),
                "unique_actions": len({row["action"]["label"] for row in route}),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
