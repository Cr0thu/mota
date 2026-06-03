from __future__ import annotations

import sys
from pathlib import Path


VISUALIZER_DIR = Path(__file__).resolve().parents[1] / "tools" / "visualizer"
if str(VISUALIZER_DIR) not in sys.path:
    sys.path.insert(0, str(VISUALIZER_DIR))

from environment import Mota  # noqa: E402
from stage_reward import SWORD_POS, stage_potential, transition_reward  # noqa: E402


def _make_env() -> Mota:
    env = Mota()
    env.build_env("10層魔塔")
    env.create_nodes()
    env.reset()
    return env


def _take(env: Mota, pos: tuple[int, int, int]):
    for action in env.get_feasible_actions():
        if env.n2p[action][:3] == pos:
            before = stage_potential(env)
            before_state = env.get_player_state().copy()
            ending = env.step(action)
            after_state = env.get_player_state().copy()
            reward = transition_reward(env, action, before_state, after_state, ending, before)
            assert ending == "continue"
            return action, reward
    available = [(env.n2p[action][:3], action.id) for action in env.get_feasible_actions()]
    raise AssertionError(f"missing action {pos}; available={available}")


def test_stage_reward_prefers_key_over_intro_npc_at_start() -> None:
    env = _make_env()
    rows = []
    for action in env.get_feasible_actions():
        before = stage_potential(env)
        before_state = env.get_player_state().copy()
        ending = env.step(action)
        after_state = env.get_player_state().copy()
        reward = transition_reward(env, action, before_state, after_state, ending, before)
        env.back_step(1)
        rows.append((action.id, reward.total))

    by_id = dict(rows)
    assert by_id["yellowKey"] > by_id["princess"]


def test_stage_advances_after_taking_sword() -> None:
    env = _make_env()
    before = stage_potential(env)
    assert before.stage == "sword"
    action = env.p2n[SWORD_POS]
    before_state = env.get_player_state().copy()
    ending = env.step(action)
    after_state = env.get_player_state().copy()
    reward = transition_reward(env, action, before_state, after_state, ending, before)
    assert action.id == "sword1"
    assert reward.before.stage == "sword"
    assert reward.after.stage == "shield"
    assert reward.total > 1000.0
