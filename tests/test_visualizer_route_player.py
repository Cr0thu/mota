from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VISUALIZER_DIR = PROJECT_ROOT / "tools" / "visualizer"
if str(VISUALIZER_DIR) not in sys.path:
    sys.path.insert(0, str(VISUALIZER_DIR))

from environment import Mota  # noqa: E402
from environment import Terrain  # noqa: E402
from route_player import load_route  # noqa: E402


def _make_env() -> Mota:
    env = Mota()
    env.build_env("10層魔塔")
    env.create_nodes()
    env.reset()
    return env


def _clear(env: Mota, pos: tuple[int, int, int]) -> None:
    node = env.p2n.get(pos)
    if node is not None:
        node.activated = True


def _sync_algorithm_start(env: Mota) -> None:
    env.reset()
    for pos in [(0, 1, 3), (0, 1, 4), (0, 1, 5)]:
        _clear(env, pos)
    for pos in [(2, 7, 5), (2, 8, 5), (2, 9, 4), (2, 9, 6), (2, 10, 5), (2, 9, 5)]:
        _clear(env, pos)
    env.player.money = 4
    assert env.step(env.p2n[(1, 7, 3)]) == "continue"
    env.player.hp = 400
    env.player.atk = 10
    env.player.def_ = 10
    env.player.mdef = 0
    env.player.money = 4


def _sync_internal_route_step(env: Mota, step) -> bool:
    if not step.is_state_noop():
        return False
    pos = step.after_pos or step.target_pos
    node = env.p2n.get(pos)
    if node is None or getattr(node, "class_", None) != "terrains":
        return False
    node.activate(env.player)
    env.observation.append(node)
    return True


def _apply_route_state(env: Mota, state: dict) -> None:
    for key, attr in [("hp", "hp"), ("atk", "atk"), ("def", "def_"), ("mdef", "mdef"), ("money", "money")]:
        if key in state:
            setattr(env.player, attr, int(state[key]))
    for key, value in state.get("keys", {}).items():
        env.player.items[key] = int(value)
    if isinstance(state.get("flags"), dict):
        env.flags = dict(state["flags"])


def _sync_hidden_route_state(env: Mota, step) -> bool:
    if not step.needs_visualizer_state_sync() or not step.after:
        return False
    pos = step.after_pos or step.target_pos
    node = env.p2n.get(pos)
    if node is None:
        node = Terrain({"cls": "terrains", "id": "background", "noPass": False})
        node.activated = True
        env.n2p[node] = pos
        env.p2n[pos] = node
        for node2 in env.n2p:
            node2.links.clear()
        env.create_nodes()
    else:
        node.activated = True
    _apply_route_state(env, step.after)
    env.observation.append(node)
    return True


def _replay_visualizer_route(route_path: Path, max_ticks: int = 5000) -> Mota:
    env = _make_env()
    _sync_algorithm_start(env)
    route = load_route(route_path)

    for _ in range(max_ticks):
        if route.done:
            return env
        assert route.align_to_current(env)
        action, _actions, completes_step, matched_cursor = route.find_visualizer_action(env)
        if matched_cursor is not None and matched_cursor != route.cursor:
            assert route._can_skip_to_step(matched_cursor)
            route.cursor = matched_cursor
        step = route.current_step()
        if action is None:
            if step.is_visualizer_noop():
                route.advance()
                continue
            if _sync_hidden_route_state(env, step):
                route.advance()
                continue
            assert _sync_internal_route_step(env, step), step.label
            route.advance()
            continue
        ending = env.step(action)
        assert ending == "continue"
        if completes_step:
            route.advance()

    raise AssertionError(f"route did not finish within {max_ticks} ticks at cursor {route.cursor}")


def test_visualizer_removes_4f_shop_and_flyer_but_keeps_key_merchants() -> None:
    env = _make_env()

    assert env.player.hp == 1000
    assert env.player.atk == 100
    assert env.player.def_ == 100
    assert env.player.items["I300"] == 1
    assert (3, 1, 6) not in env.p2n
    assert all(getattr(node, "id", "") != "centerFly" for node in env.n2p)
    assert env.p2n[(5, 4, 8)].id == "redMan"
    assert env.p2n[(6, 1, 6)].id == "redMan"
    assert "skeletonSoilder" not in env.env_data["enemies"]
    assert "skeletonSoldier" in env.env_data["enemies"]


def test_route_player_replays_current_solver_route_without_action_mismatch() -> None:
    env = _replay_visualizer_route(
        PROJECT_ROOT / "artifacts/expert/route_mt10_resources_manual_refill_success_20260524.jsonl"
    )
    assert env.n2p[env.observation[-1]][:3] == (9, 11, 11)
    assert env.player.hp == 248
    assert env.player.atk == 27
    assert env.player.def_ == 27
    assert env.player.items["redKey"] == 0


def test_route_player_replays_hp403_without_state_drift() -> None:
    env = _replay_visualizer_route(
        PROJECT_ROOT
        / "artifacts/manual_exploration_20260524/manual_success_no_shop_true_10f_trap_optimized_hp403.jsonl",
        max_ticks=8000,
    )

    assert env.n2p[env.observation[-1]][:3] == (9, 1, 6)
    assert env.player.hp == 403
    assert env.player.atk == 27
    assert env.player.def_ == 27
    assert env.player.items["redKey"] == 0
    assert env.flags["10f战胜骷髅队长"] is True


def test_route_player_does_not_skip_state_changing_steps() -> None:
    route = load_route(
        PROJECT_ROOT / "artifacts/manual_exploration_20260524/manual_success_no_shop_true_10f_trap_optimized_hp403.jsonl"
    )
    route.cursor = 67

    assert not route._can_skip_to_step(92)
    assert route.steps[67].label == "open yellowDoor MT7:1,5"
