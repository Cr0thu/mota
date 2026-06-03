from __future__ import annotations

from mota_env import MotaSimulator, load_game_data
from mota_solver.go_explore import GoExploreConfig, run_go_explore


def test_go_explore_smoke_records_archive() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    result = run_go_explore(
        sim,
        GoExploreConfig(
            target_stage="sword",
            iterations=4,
            rollout_steps=2,
            seed=20260528,
        ),
    )
    assert result.expansions >= 1
    assert result.archive_cells >= 1
    assert result.best_summary["hp"] > 0
