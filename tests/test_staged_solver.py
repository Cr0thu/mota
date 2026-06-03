from __future__ import annotations

from mota_env import MotaSimulator, load_game_data
from mota_solver.staged import solve_staged_first10


def test_staged_solver_smoke_returns_stage_summary() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    result = solve_staged_first10(
        sim,
        max_expansions_per_stage=50,
        keep_per_parent=20,
        frontier_size=4,
        seed=123,
    )

    assert result.expansions > 0
    assert result.stage_summaries
    assert result.stage_summaries[0].stage == "sword"
    assert not result.state.dead
