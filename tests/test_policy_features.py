from __future__ import annotations

from mota_env import MotaSimulator, load_game_data
from mota_rl.policy_features import (
    ACTION_FEATURE_DIM,
    STATE_FEATURE_DIM,
    action_feature_vector,
    state_feature_vector,
)
from mota_solver.search import state_summary


def test_actor_critic_feature_vectors_have_stable_shapes() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    action = sim.macro_actions(state)[0]

    state_vec = state_feature_vector(sim, state, "shield")
    action_vec = action_feature_vector(sim, state, action, "shield")

    assert len(state_vec) == STATE_FEATURE_DIM
    assert len(action_vec) == ACTION_FEATURE_DIM
    assert any(value != 0 for value in state_vec)
    assert any(value != 0 for value in action_vec)


def test_loop_action_penalty_discourages_immediate_stair_bounce() -> None:
    try:
        import torch
    except Exception:
        return

    from mota_rl.train_actor_critic import loop_action_penalties

    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    up_action = next(action for action in sim.macro_actions(state) if "upFloor" in action["label"])
    before = state_summary(state)
    sim.apply_macro_action(state, up_action)
    route = [{"action": up_action, "before": before, "after": state_summary(state)}]
    actions = sim.macro_actions(state)
    penalties = loop_action_penalties(
        sim=sim,
        state=state,
        actions=actions,
        route=route,
        penalty=8.0,
        window=4,
        torch=torch,
        device=torch.device("cpu"),
    )
    down_index = next(index for index, action in enumerate(actions) if "downFloor" in action["label"])
    assert float(penalties[down_index]) > 0.0
