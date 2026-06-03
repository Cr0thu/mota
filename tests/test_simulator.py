from __future__ import annotations

from mota_env import MotaSimulator, SimulatorConfig, load_game_data
from mota_env.rewards import remaining_attack_defense_gems, stage_complete


def make_sim() -> MotaSimulator:
    return MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))


def test_damage_formula_and_unbreakable_enemy() -> None:
    sim = make_sim()
    state = sim.reset()
    state.hp, state.atk, state.defense = 400, 10, 10
    assert sim.damage_info(state, "greenSlime") == {"damage": 24, "turn": 4}
    assert sim.damage_info(state, "blueGuard") is None


def test_default_state_is_after_mt3_and_mt2_story_sequence() -> None:
    sim = make_sim()
    state = sim.reset()
    assert (state.floor_id, state.x, state.y) == ("MT2", 3, 7)
    assert (state.hp, state.atk, state.defense) == (400, 10, 10)
    assert state.flags["03"] == 1
    assert ("MT2", 3, 7) in state.triggered_events
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


def test_mt6_and_mt7_merchants_sell_keys() -> None:
    sim = make_sim()
    state = sim.reset()

    state.floor_id, state.x, state.y = "MT6", 7, 4
    state.money = 50
    assert sim.block_id(sim.tile(state, 8, 4)) == "trader"
    assert sim.step(state, "right").ok
    assert state.money == 0
    assert state.items["blueKey"] == 1
    assert sim.tile(state, 8, 4, "MT6") == 0

    state = sim.reset()
    state.floor_id, state.x, state.y = "MT7", 5, 1
    state.money = 50
    assert sim.block_id(sim.tile(state, 6, 1)) == "trader"
    assert sim.step(state, "right").ok
    assert state.money == 0
    assert state.items["yellowKey"] == 5
    assert sim.tile(state, 6, 1, "MT7") == 0


def test_merchants_require_enough_money_and_are_macro_actions() -> None:
    sim = make_sim()
    state = sim.reset()
    state.floor_id, state.x, state.y = "MT6", 7, 4
    state.money = 49

    before = state.clone()
    transition = sim.step(state, "right")
    assert not transition.ok
    assert (state.floor_id, state.x, state.y, state.money) == (
        before.floor_id,
        before.x,
        before.y,
        before.money,
    )
    assert state.items["blueKey"] == before.items["blueKey"]
    assert sim.tile(state, 8, 4, "MT6") == sim.tile(before, 8, 4, "MT6")
    assert not any("buy blueKey" in action["label"] for action in sim.macro_actions(state))

    state.money = 50
    labels = [action["label"] for action in sim.macro_actions(state)]
    assert any("buy blueKey MT6:8,4" in label for label in labels)


def test_mt10_trap_and_boss_flag() -> None:
    sim = make_sim()
    state = sim.reset()
    assert sim.tile(state, 2, 3, "MT10") == 0
    assert sim.tile(state, 2, 4, "MT10") == 0
    state.floor_id, state.x, state.y = "MT10", 6, 6
    state.hp, state.atk, state.defense = 5000, 200, 200
    assert sim.step(state, "up").ok
    assert state.flags["10f机关"] is True
    for x, y in [(5, 4), (6, 4), (7, 4), (5, 5), (7, 5), (5, 6), (6, 6), (7, 6)]:
        if sim.is_enemy_tile(sim.tile(state, x, y)):
            sim.battle(state, x, y)
    assert sim.tile(state, 6, 3) == 0
    assert any(action["target"] == ["MT10", 6, 1] for action in sim.macro_actions(state))
    state.x, state.y = 6, 2
    assert sim.step(state, "up").ok
    assert state.flags["10f战胜骷髅队长"] is True
    assert state.done is True


def test_mt8_two_blue_guards_open_special_door() -> None:
    sim = make_sim()
    state = sim.reset()
    state.floor_id, state.x, state.y = "MT8", 9, 5
    state.hp, state.atk, state.defense = 5000, 300, 300
    assert sim.block_id(sim.tile(state, 10, 4, "MT8")) == "specialDoor"

    assert sim.battle(state, 9, 5)
    assert sim.block_id(sim.tile(state, 10, 4, "MT8")) == "specialDoor"
    assert state.flags["8"] == 1

    assert sim.battle(state, 11, 5)
    assert sim.tile(state, 10, 4, "MT8") == 0


def test_mt10_boss_generates_resources_and_removes_red_door() -> None:
    sim = make_sim()
    state = sim.reset()
    state.floor_id, state.x, state.y = "MT10", 6, 6
    state.hp, state.atk, state.defense = 5000, 200, 200
    sim.trigger_mt10_trap(state)
    for x, y in [(5, 4), (6, 4), (7, 4), (5, 5), (7, 5), (5, 6), (6, 6), (7, 6)]:
        if sim.is_enemy_tile(sim.tile(state, x, y)):
            sim.battle(state, x, y)

    assert sim.battle(state, 6, 1)
    assert state.flags["10f战胜骷髅队长"] is True
    assert sim.tile(state, 6, 9, "MT10") == 0
    assert sim.block_id(sim.tile(state, 1, 3, "MT10")) == "redGem"
    assert sim.block_id(sim.tile(state, 9, 3, "MT10")) == "blueGem"
    assert sim.block_id(sim.tile(state, 1, 4, "MT10")) == "bluePotion"
    assert sim.block_id(sim.tile(state, 9, 4, "MT10")) == "yellowKey"


def test_can_continue_after_mt10_boss_to_collect_generated_gems() -> None:
    sim = MotaSimulator(
        load_game_data("artifacts/data/mota_first10.json"),
        SimulatorConfig(stop_on_boss=False),
    )
    state = sim.reset()
    state.floor_id, state.x, state.y = "MT10", 6, 6
    state.hp, state.atk, state.defense = 5000, 200, 200
    sim.trigger_mt10_trap(state)
    for x, y in [(5, 4), (6, 4), (7, 4), (5, 5), (7, 5), (5, 6), (6, 6), (7, 6)]:
        if sim.is_enemy_tile(sim.tile(state, x, y)):
            sim.battle(state, x, y)

    assert sim.battle(state, 6, 1)
    state.x, state.y = 6, 1
    assert state.flags["10f战胜骷髅队长"] is True
    assert state.done is False
    assert not stage_complete(sim, state, "boss_all_gems")

    labels = [action["label"] for action in sim.macro_actions(state)]
    assert "go redGem MT10:1,3" in labels
    action = next(action for action in sim.macro_actions(state) if action["label"] == "go redGem MT10:1,3")
    before_atk = state.atk
    transition = sim.apply_macro_action(state, action)
    assert transition.ok
    assert state.atk > before_atk
    assert remaining_attack_defense_gems(sim, state) >= 5


def test_simple_scenario_starts_after_story_without_shop_or_fly() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    assert (state.floor_id, state.x, state.y) == ("MT2", 3, 7)
    assert (state.hp, state.atk, state.defense, state.money) == (400, 10, 10, 4)
    assert state.flags["03"] == 1
    assert ("MT2", 3, 7) in state.triggered_events
    assert ("MT3", 5, 9) in state.triggered_events
    assert all(sim.tile(state, x, 1, "MT1") == 0 for x in (3, 4, 5))
    assert sim.tile(state, 2, 7, "MT2") == 0
    assert all(sim.tile(state, x, y, "MT3") == 0 for x, y in [(5, 7), (5, 8), (4, 9), (6, 9), (5, 10), (5, 9)])
    assert sim.tile(state, 2, 11, "MT1") == 0
    assert all(sim.tile(state, x, 1, "MT4") == 0 for x in (5, 6, 7))
    labels = [action["label"] for action in sim.macro_actions(state)]
    assert not any(label.startswith("fly") or label.startswith("shop") for label in labels)
    assert not any("event MT3:5,9" in label for label in labels)


def test_reachable_parent_map_matches_reachable_paths() -> None:
    sim = make_sim()
    state = sim.reset()
    parents = sim.reachable_parent_map(state)
    reachable = sim.reachable_cells(state)

    assert set(parents) == set(reachable)
    for cell, path in reachable.items():
        assert sim.reachable_path(state, cell, parents) == path


def test_lazy_macro_action_paths_are_executable() -> None:
    sim = make_sim()
    state = sim.reset()
    actions = sim.macro_actions(state)

    assert actions
    for action in actions:
        candidate = state.clone()
        transition = sim.apply_macro_action(candidate, action)
        assert transition.ok, action["label"]


def test_full_mt2_thief_event_resets_hp_without_touching_stats() -> None:
    sim = MotaSimulator(
        load_game_data("artifacts/data/mota_first10.json"),
        SimulatorConfig(scenario="full"),
    )
    state = sim.reset()
    state.floor_id, state.x, state.y = "MT2", 3, 7
    state.hp, state.atk, state.defense = 777, 13, 14
    sim.trigger_event_at(state, 3, 7)
    assert (state.hp, state.atk, state.defense) == (400, 13, 14)
    assert sim.tile(state, 2, 7, "MT2") == 0
    assert sim.tile(state, 3, 7, "MT2") == 0


def test_full_mt3_demon_event_resets_stats_and_teleports_to_mt2() -> None:
    sim = MotaSimulator(
        load_game_data("artifacts/data/mota_first10.json"),
        SimulatorConfig(scenario="full"),
    )
    state = sim.reset()
    state.floor_id, state.x, state.y = "MT3", 5, 9
    state.hp, state.atk, state.defense = 777, 13, 14
    state.flags["nowWeapon"] = "sword5"
    state.flags["nowShield"] = "shield5"
    state.flags["魔法免疫"] = True
    sim.trigger_event_at(state, 5, 9)
    assert (state.floor_id, state.x, state.y) == ("MT2", 3, 8)
    assert (state.hp, state.atk, state.defense, state.mdef) == (400, 10, 10, 0)
    assert state.flags["03"] == 1
    assert state.flags["nowWeapon"] is None
    assert state.flags["nowShield"] is None
    assert state.flags["魔法免疫"] is False
    assert ("MT3", 5, 9) in state.triggered_events
    assert all(sim.tile(state, x, y, "MT3") == 0 for x, y in [(5, 7), (5, 8), (4, 9), (6, 9), (5, 10), (5, 9)])


def test_relaxed_negative_hp_battle_allows_exploration_debt() -> None:
    sim = MotaSimulator(
        load_game_data("artifacts/data/mota_first10.json"),
        SimulatorConfig(allow_negative_hp=True, min_hp=-500),
    )
    state = sim.reset()
    state.hp = 10
    state.atk = 10
    state.defense = 10
    state.floor_id, state.x, state.y = "MT1", 1, 4

    assert sim.step(state, "right").ok
    assert state.hp < 0
    assert not state.dead
