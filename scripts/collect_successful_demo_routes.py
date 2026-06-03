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
from mota_solver.search import state_summary
from validate_route_constraints import validate_route


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"non-object row in {path}")
                rows.append(row)
    return rows


def action_fingerprint(rows: list[dict[str, Any]]) -> str:
    labels = [str((row.get("action") or {}).get("label") or "") for row in rows]
    return "\n".join(labels)


def replay_route(path: Path, data: str, continue_after_boss: bool = False) -> dict[str, Any] | None:
    try:
        rows = load_jsonl(path)
    except Exception:
        return None
    if not rows or not all(isinstance(row.get("action"), dict) for row in rows[: min(3, len(rows))]):
        return None

    sim = MotaSimulator(
        load_game_data(data),
        SimulatorConfig(stop_on_boss=not continue_after_boss),
    )
    state = sim.reset()
    applied = 0
    used_rows: list[dict[str, Any]] = []
    failed: dict[str, Any] | None = None
    for index, row in enumerate(rows):
        transition = sim.apply_macro_action(state, row["action"])
        if not transition.ok or state.dead:
            failed = {
                "index": index,
                "message": transition.message,
                "action": row.get("action", {}),
                "state": state_summary(state),
            }
            break
        applied += 1
        used_rows.append(row)
        if state.flags.get("10f战胜骷髅队长") and not continue_after_boss:
            break

    solved = bool(state.flags.get("10f战胜骷髅队长"))
    return {
        "route": str(path),
        "rows": len(rows),
        "applied_steps": applied,
        "solved": solved,
        "failed": failed,
        "final": state_summary(state),
        "fingerprint": action_fingerprint(used_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts")
    parser.add_argument("--data", default="artifacts/data/mota_first10.json")
    parser.add_argument("--out", default="artifacts/demos/successful_boss_routes_20260603.json")
    parser.add_argument("--include-failed", action="store_true")
    parser.add_argument("--allow-shop", action="store_true")
    parser.add_argument("--allow-fly", action="store_true")
    parser.add_argument("--continue-after-boss", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    results: list[dict[str, Any]] = []
    failed_or_unsolved: list[dict[str, Any]] = []
    seen_fingerprints: set[str] = set()
    for path in sorted(root.rglob("*.jsonl")):
        replay = replay_route(path, args.data, continue_after_boss=args.continue_after_boss)
        if replay is None:
            continue
        if not replay["solved"] or replay["failed"] is not None:
            if args.include_failed:
                failed_or_unsolved.append(replay)
            continue
        constraints = validate_route(
            path,
            forbid_shop=not args.allow_shop,
            forbid_fly=not args.allow_fly,
            forbid_merchants=False,
        )
        if not constraints["ok"]:
            if args.include_failed:
                failed_or_unsolved.append({**replay, "constraints": constraints})
            continue
        fingerprint = str(replay.pop("fingerprint"))
        duplicate = fingerprint in seen_fingerprints
        seen_fingerprints.add(fingerprint)
        final = replay.get("final", {})
        results.append(
            {
                **replay,
                "constraints_ok": True,
                "duplicate_action_sequence": duplicate,
                "final_hp": final.get("hp"),
                "final_atk": final.get("atk"),
                "final_def": final.get("def"),
            }
        )

    results.sort(key=lambda row: (bool(row["duplicate_action_sequence"]), -int(row.get("final_hp") or -999999), int(row["applied_steps"])))
    payload = {
        "data": args.data,
        "root": args.root,
        "route_count": len(results),
        "unique_route_count": sum(1 for row in results if not row["duplicate_action_sequence"]),
        "routes": results,
    }
    if args.include_failed:
        payload["failed_or_unsolved"] = failed_or_unsolved

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
