from __future__ import annotations

import json
from pathlib import Path

from mota_env import MotaSimulator, load_game_data
from mota_solver.az_mcts import filter_stage_actions


def main() -> None:
    run = Path("artifacts/runs/latest_remote_hp403_potential_only_run.txt").read_text().strip()
    run_path = Path(run)
    names = [
        "archive_mt10_resources_v100_from_v81_shield_mt9chain_continue",
        "archive_mt10_resources_v101_from_v82_shield_mt9chain_continue",
        "archive_mt10_resources_v102_from_v81_shield_mt8_return",
        "archive_mt10_resources_v103_from_v82_shield_mt8_return",
        "archive_shield_buffer_v105_quality_prefix",
        "archive_shield_buffer_v106_quality_prefix",
    ]
    prefix_by_name = {
        "archive_mt10_resources_v100_from_v81_shield_mt9chain_continue": run_path
        / "routes"
        / "auto_prefix_shield_v81_hp67_y1_full.jsonl",
        "archive_mt10_resources_v101_from_v82_shield_mt9chain_continue": run_path
        / "routes"
        / "auto_prefix_shield_v82_hp59_y1_full.jsonl",
        "archive_mt10_resources_v102_from_v81_shield_mt8_return": run_path
        / "routes"
        / "auto_prefix_shield_v81_hp67_y1_full.jsonl",
        "archive_mt10_resources_v103_from_v82_shield_mt8_return": run_path
        / "routes"
        / "auto_prefix_shield_v82_hp59_y1_full.jsonl",
        "archive_shield_buffer_v105_quality_prefix": None,
        "archive_shield_buffer_v106_quality_prefix": None,
    }
    target_by_name = {
        "archive_shield_buffer_v105_quality_prefix": "shield_buffer",
        "archive_shield_buffer_v106_quality_prefix": "shield_buffer",
    }
    for name in names:
        print(f"==={name}===")
        route_path = run_path / "archive" / name / "best_stats_route_so_far.jsonl"
        if not route_path.exists():
            print("missing", route_path)
            continue
        suffix_rows = [json.loads(line) for line in route_path.read_text().splitlines() if line.strip()]
        prefix_path = prefix_by_name[name]
        prefix_rows = []
        if prefix_path is not None:
            prefix_rows = [
                json.loads(line) for line in prefix_path.read_text().splitlines() if line.strip()
            ]
        rows = [*prefix_rows, *suffix_rows]
        print("prefix_len", len(prefix_rows), "suffix_len", len(suffix_rows), "total_len", len(rows))
        for index, row in list(enumerate(suffix_rows, 1))[-40:]:
            action = row.get("action", {})
            after = row.get("after") or row.get("state") or {}
            keys = after.get("keys") or after.get("items")
            print(
                index,
                action.get("label") or row.get("label"),
                "=>",
                {
                    key: after.get(key)
                    for key in ["floor", "floor_id", "x", "y", "hp", "atk", "def", "defense", "money"]
                },
                keys,
            )

        sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
        state = sim.reset()
        for row in rows:
            action = row.get("action") or row
            label = action.get("label") or row.get("label")
            if not label:
                continue
            matches = [candidate for candidate in sim.macro_actions(state) if candidate.get("label") == label]
            if not matches:
                print(
                    "REPLAY_MISSING",
                    label,
                    "at",
                    state.floor_id,
                    state.x,
                    state.y,
                    state.hp,
                    state.atk,
                    state.defense,
                    state.money,
                    dict(state.items),
                )
                break
            transition = sim.apply_macro_action(state, matches[0])
            if not transition.ok:
                print("REPLAY_FAILED", label, transition.message)
                break
        print(
            "replayed",
            state.floor_id,
            state.x,
            state.y,
            "hp",
            state.hp,
            "atk",
            state.atk,
            "def",
            state.defense,
            "money",
            state.money,
            "keys",
            dict(state.items),
            "dead",
            state.dead,
        )
        raw = sim.macro_actions(state)
        target_stage = target_by_name.get(name, "mt10_resources")
        filtered = filter_stage_actions(raw, state, target_stage, sim=sim)
        print("raw")
        for action in raw:
            print("  ", action.get("label"), "path", len(action.get("path") or []))
        print("filtered")
        for action in filtered:
            next_state = state.clone()
            transition = sim.apply_macro_action(next_state, action)
            print(
                "  ",
                action.get("label"),
                "path",
                len(action.get("path") or []),
                "->",
                next_state.floor_id,
                next_state.x,
                next_state.y,
                "hp",
                next_state.hp,
                "atk",
                next_state.atk,
                "def",
                next_state.defense,
                "money",
                next_state.money,
                "keys",
                dict(next_state.items),
                "dead",
                next_state.dead,
                "ok",
                transition.ok,
                transition.message,
            )


if __name__ == "__main__":
    main()
