from __future__ import annotations

from mota_env import MotaSimulator, load_game_data
from mota_solver.search import solve_first10


def test_solver_returns_legal_best_route_smoke() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    result = solve_first10(sim, max_expansions=200)
    state = sim.reset()
    for row in result.route:
        transition = sim.apply_macro_action(state, row["action"])
        assert transition.ok
        assert not state.dead
    assert result.expansions == 200


def test_landmark_key_potential_heuristic_smoke() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    result = solve_first10(
        sim,
        max_expansions=200,
        heuristic_scheme="landmark_key_potential",
    )
    assert result.expansions == 200
    assert not result.state.dead
