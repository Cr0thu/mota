from __future__ import annotations

import sys
from pathlib import Path


VISUALIZER_DIR = Path(__file__).resolve().parents[1] / "tools" / "visualizer"
if str(VISUALIZER_DIR) not in sys.path:
    sys.path.insert(0, str(VISUALIZER_DIR))

from environment import Mota  # noqa: E402


def _take(env: Mota, pos: tuple[int, int, int]):
    for action in env.get_feasible_actions():
        if env.n2p[action][:3] == pos:
            ending = env.step(action)
            assert ending == "continue"
            return action
    available = [(action.id, env.n2p[action][:3]) for action in env.get_feasible_actions()]
    raise AssertionError(f"missing action {pos}; available={available}")


def test_visualizer_macro_actions_hide_stairs_and_plain_landing_backgrounds() -> None:
    env = Mota()
    env.build_env("10層魔塔")
    env.create_nodes()
    env.reset()

    assert all(action.id not in {"upFloor", "downFloor"} for action in env.get_feasible_actions())

    # Clear the compulsory 1F path. Stairs are traversed internally, so after
    # this point the visible actions are meaningful cross-floor targets rather
    # than up/down stair nodes or plain landing backgrounds.
    for pos in [
        (0, 10, 7),  # princess
        (0, 10, 5),  # yellow key
        (0, 9, 6),  # yellow door
        (0, 1, 5),  # green slime
        (0, 1, 4),  # red slime
        (0, 1, 3),  # green slime
    ]:
        _take(env, pos)

    actions = env.get_feasible_actions()
    assert all(action.id not in {"upFloor", "downFloor"} for action in actions)
    assert all(
        action.id != "background" or env.n2p[action][:3] in env.env_data["floors"]["afterEvent"]
        for action in actions
    )
    assert [(env.n2p[action][:3], action.id) for action in actions] == [((2, 9, 5), "background")]

    # Trigger the 3F demon event, then the 2F thief event.
    _take(env, (2, 9, 5))
    assert env.n2p[env.observation[-1]][:3] == (1, 8, 3)
    assert env.flags["03"] == 1
    _take(env, (1, 7, 3))
    assert env.n2p[env.observation[-1]][:3] == (1, 7, 3)
    assert all(action.id not in {"upFloor", "downFloor"} for action in env.get_feasible_actions())


def test_feasible_action_probe_does_not_pollute_key_merchant_state() -> None:
    env = Mota()
    env.build_env("10層魔塔")
    env.create_nodes()
    env.reset()

    merchant = env.p2n[(6, 1, 6)]
    env.observation.append(merchant)
    env.player.money = 40
    env.player.items["yellowKey"] = 2

    before = (env.player.money, env.player.items["yellowKey"])
    actions = env.get_feasible_actions()
    after = (env.player.money, env.player.items["yellowKey"])

    assert merchant not in actions
    assert after == before
