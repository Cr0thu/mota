from __future__ import annotations

from mota_env import MotaSimulator, load_game_data


def make_sim() -> MotaSimulator:
    return MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))


def test_damage_formula_and_unbreakable_enemy() -> None:
    sim = make_sim()
    state = sim.reset()
    state.hp, state.atk, state.defense = 400, 10, 10
    assert sim.damage_info(state, "greenSlime") == {"damage": 24, "turn": 4}
    assert sim.damage_info(state, "blueGuard") is None


def test_default_state_is_after_mt3_plot() -> None:
    sim = make_sim()
    state = sim.reset()
    assert (state.floor_id, state.x, state.y) == ("MT2", 3, 7)
    assert (state.hp, state.atk, state.defense) == (400, 10, 10)
    assert state.flags["03"] == 1
    assert ("MT3", 5, 9) in state.triggered_events


def test_key_door_consumes_key() -> None:
    sim = make_sim()
    state = sim.reset()
    state.floor_id, state.x, state.y = "MT1", 6, 10
    state.items["yellowKey"] = 1
    assert sim.step(state, "up").ok
    assert state.items["yellowKey"] == 0
    assert sim.tile(state, 6, 9) == 0


def test_items_and_shop_disabled_in_default_scenario() -> None:
    sim = make_sim()
    state = sim.reset()
    state.floor_id, state.x, state.y = "MT4", 6, 1
    state.money = 20
    assert not sim.shop_buy(state, "def")
    assert state.money == 20
    assert state.defense == 10
    state.floor_id = "MT3"
    sim.apply_item(state, "redGem")
    assert state.atk == 11


def test_mt10_trap_and_boss_flag() -> None:
    sim = make_sim()
    state = sim.reset()
    state.floor_id, state.x, state.y = "MT10", 6, 6
    state.hp, state.atk, state.defense = 5000, 200, 200
    assert sim.step(state, "up").ok
    assert state.flags["10f机关"] is True
    for x, y in [(5, 4), (6, 4), (7, 4), (5, 5), (7, 5), (5, 6), (6, 6), (7, 6)]:
        if sim.is_enemy_tile(sim.tile(state, x, y)):
            sim.battle(state, x, y)
    assert sim.tile(state, 6, 3) == 0
    state.x, state.y = 6, 2
    assert sim.step(state, "up").ok
    assert state.flags["10f战胜骷髅队长"] is True
    assert state.done is True


def test_simple_scenario_starts_after_thief_without_shop_or_fly() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    assert (state.floor_id, state.x, state.y) == ("MT2", 3, 7)
    assert (state.hp, state.atk, state.defense, state.money) == (400, 10, 10, 4)
    assert state.flags["03"] == 1
    assert ("MT2", 3, 7) in state.triggered_events
    assert sim.tile(state, 2, 7, "MT2") == 0
    assert sim.tile(state, 2, 11, "MT1") == 0
    assert all(sim.tile(state, x, 1, "MT4") == 0 for x in (5, 6, 7))
    labels = [action["label"] for action in sim.macro_actions(state)]
    assert not any(label.startswith("fly") or label.startswith("shop") for label in labels)
