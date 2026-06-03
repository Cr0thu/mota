from __future__ import annotations

import sys
from pathlib import Path


VISUALIZER_DIR = Path(__file__).resolve().parents[1] / "tools" / "visualizer"
if str(VISUALIZER_DIR) not in sys.path:
    sys.path.insert(0, str(VISUALIZER_DIR))

from environment import Mota  # noqa: E402
from q_learning import TabularQLearningAgent  # noqa: E402
import json


def _make_env() -> Mota:
    env = Mota()
    env.build_env("10層魔塔")
    env.create_nodes()
    env.reset()
    return env


def test_q_learning_update_and_roundtrip(tmp_path: Path) -> None:
    env = _make_env()
    actions = env.get_feasible_actions()
    assert actions

    agent = TabularQLearningAgent(alpha=0.5, gamma=0.9, epsilon=0.0, prior_weight=0.0)
    state_key = agent.state_key(env)
    action = actions[0]
    action_key = agent.action_key(env, action)
    td = agent.update(state_key, action_key, reward=2.0, next_state_key=None, next_action_keys=[], done=True)

    assert td["new"] == 1.0
    assert agent.q_value(env, action, state_key) == 1.0

    path = tmp_path / "q_table.json"
    agent.save(path)
    loaded = TabularQLearningAgent.load(path)
    assert loaded.q_value(env, action, state_key) == 1.0
    assert loaded.alpha == 0.5


def test_q_learning_state_uses_exact_hp_money_and_no_mdef() -> None:
    env = _make_env()
    agent = TabularQLearningAgent()
    payload = json.loads(agent.state_key(env))

    assert payload["version"] == 2
    assert payload["hp"] == env.player.hp
    assert payload["money"] == env.player.money
    assert "hp25" not in payload
    assert "money2" not in payload
    assert "mdef" not in payload


def test_q_learning_action_choice_uses_q_and_prior() -> None:
    env = _make_env()
    actions = env.get_feasible_actions()
    assert len(actions) >= 2

    agent = TabularQLearningAgent(epsilon=0.0, prior_weight=0.5)
    state_key = agent.state_key(env)
    first_key = agent.action_key(env, actions[0])
    second_key = agent.action_key(env, actions[1])
    agent.q[state_key] = {first_key: 1.0, second_key: 0.0}

    action, index, mode = agent.choose_action(env, actions[:2], priors=[0.0, 4.0])

    assert action is actions[1]
    assert index == 1
    assert mode == "greedy"
