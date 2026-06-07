from __future__ import annotations

from mota_agentic import AgenticRLConfig, AgenticRLOrchestrator
from mota_env import MotaSimulator, load_game_data


def test_agentic_orchestrator_smoke_produces_legal_prefix() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    outcome = AgenticRLOrchestrator(
        sim,
        AgenticRLConfig(episodes=1, max_steps=6, seed=20260606),
    ).run()
    assert outcome.route
    assert len(outcome.route) <= 6

    replay = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = replay.reset()
    for row in outcome.route:
        transition = replay.apply_macro_action(state, row["action"])
        assert transition.ok
        assert not state.dead
        assert "agentic" in row
