from __future__ import annotations

from mota_env import MotaSimulator, load_game_data
from mota_env.rewards import Rewarder, progress_stage, reward_scheme_names


def test_reward_schemes_score_a_legal_macro_action() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    action = sim.macro_actions(state)[0]
    before = state.clone()
    transition = sim.apply_macro_action(state, action)
    assert transition.ok
    for scheme in reward_scheme_names():
        breakdown = Rewarder(scheme).score(sim, before, state, action, transition)
        assert isinstance(breakdown.total, float)
        assert breakdown.components


def test_progress_stage_detects_boss_flag() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    assert progress_stage(sim, state) == 0
    state.flags["10f战胜骷髅队长"] = True
    assert progress_stage(sim, state) == 7
