from __future__ import annotations

from mota_env import MotaSimulator, load_game_data
from mota_env.rewards import (
    LearnableStageReward,
    Rewarder,
    all_attack_defense_gems_taken,
    boss_ready,
    boss_route_margin,
    boss_route_required_damage,
    boosted_stat_stage_potential_components,
    current_stage_name,
    critical_gems_ready,
    guard_ready,
    mt10_resource_progress,
    mt10_resources_taken,
    mt10_access_ready,
    pre_shield_gems_ready,
    shield_buffer_ready,
    progress_stage,
    red_key_route_margin,
    stage_potential,
    stage_complete,
    reward_scheme_names,
    yellow_guard_margin,
)


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


def test_stage_potential_rewards_boss_attack_threshold() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.hp = 700
    state.atk = 23
    state.defense = 21
    improved = state.clone()
    improved.atk += 1

    assert stage_potential(sim, improved, stage="boss") > stage_potential(sim, state, stage="boss")


def test_stat_boost_potential_rewards_attack_and_defense() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    improved = state.clone()
    improved.atk += 1
    improved.defense += 1

    base = boosted_stat_stage_potential_components(sim, state, stage="shield")
    boosted = boosted_stat_stage_potential_components(sim, improved, stage="shield")

    assert sum(boosted.values()) > sum(base.values())


def test_learnable_stage_reward_has_graph_factors_and_scores() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    reward = LearnableStageReward()
    components = reward.potential_components(sim, state, stage="sword")

    assert "reachable_resource_value" in components
    assert "reachable_enemy_damage_drop" in components
    assert "blocked_resource_pressure" in components

    action = sim.macro_actions(state)[0]
    before = state.clone()
    transition = sim.apply_macro_action(state, action)
    breakdown = Rewarder("learnable_stage_pbrs").score(sim, before, state, action, transition)

    assert isinstance(breakdown.total, float)
    assert "env_step" in breakdown.components
    assert any(key.startswith("delta_reachable") for key in breakdown.components)


def test_post_shield_guard_stage_values_gems_more_than_shield_stage() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    state.hp = 544
    state.atk = 25
    state.defense = 25

    red_gem = state.clone()
    red_gem.atk += 1
    blue_gem = state.clone()
    blue_gem.defense += 1

    guard_atk_gain = stage_potential(sim, red_gem, stage="guard_ready") - stage_potential(
        sim, state, stage="guard_ready"
    )
    shield_atk_gain = stage_potential(sim, red_gem, stage="shield") - stage_potential(
        sim, state, stage="shield"
    )
    guard_def_gain = stage_potential(sim, blue_gem, stage="guard_ready") - stage_potential(
        sim, state, stage="guard_ready"
    )
    shield_def_gain = stage_potential(sim, blue_gem, stage="shield") - stage_potential(
        sim, state, stage="shield"
    )

    assert guard_atk_gain > shield_atk_gain
    assert guard_def_gain > shield_def_gain


def test_current_stage_starts_at_sword() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    assert current_stage_name(sim, state) == "sword"


def test_all_gems_stage_requires_mt10_gems() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    state.atk = 24
    state.defense = 22
    state.hp = 260
    state.items["yellowKey"] = 5
    state.items["blueKey"] = 1

    for floor_id, floor in state.floors.items():
        for y, row in enumerate(floor):
            for x, tile in enumerate(row):
                if floor_id != "MT10" and sim.block_id(tile) in {"redGem", "blueGem", "redPotion", "bluePotion", "blueKey"}:
                    row[x] = 0

    assert current_stage_name(sim, state) == "pre_mt10_buffer"
    assert mt10_resource_progress(sim, state) == 0
    assert not mt10_resources_taken(sim, state)
    assert not all_attack_defense_gems_taken(sim, state)

    state.floors["MT10"][6][2] = 0
    state.floors["MT10"][6][10] = 0

    assert mt10_resource_progress(sim, state) == 2
    assert all_attack_defense_gems_taken(sim, state)
    assert current_stage_name(sim, state) == "mt10_resources"

    state.floors["MT10"][11][11] = 0

    assert mt10_resources_taken(sim, state)
    assert current_stage_name(sim, state) == "guard_ready"


def test_mt10_resource_access_requires_five_yellow_keys_before_entry() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    state.atk = 26
    state.defense = 26
    state.items["blueKey"] = 1
    state.items["yellowKey"] = 4

    assert not mt10_access_ready(sim, state)

    state.items["yellowKey"] = 5

    assert mt10_access_ready(sim, state)


def test_stage_order_sword_route_gem_shield_all_gems() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()

    assert current_stage_name(sim, state) == "sword"
    state.flags["nowWeapon"] = "sword1"
    state.atk = 20
    assert current_stage_name(sim, state) == "mt4_redgem"
    state.floors["MT4"][10][7] = 0
    assert current_stage_name(sim, state) == "pre_shield_gems"
    assert not pre_shield_gems_ready(state)
    state.atk = 21
    state.defense = 11
    state.hp = 399
    assert current_stage_name(sim, state) == "pre_shield_gems"
    state.hp = 400
    assert current_stage_name(sim, state) == "shield"
    state.flags["nowShield"] = "shield1"
    state.defense = 20
    assert current_stage_name(sim, state) == "shield_buffer"
    assert not shield_buffer_ready(sim, state)
    state.items["yellowKey"] = 2
    state.hp = 300
    assert shield_buffer_ready(sim, state)
    assert current_stage_name(sim, state) == "mid_gems"


def test_current_stage_does_not_regress_to_mt8_hp_ready_after_entering_mt8() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT8"
    state.x = 5
    state.y = 11
    state.hp = 34
    state.atk = 26
    state.defense = 26
    state.items["yellowKey"] = 2
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    for floor_id, x, y, item_id in (
        ("MT5", 1, 9, "blueGem"),
        ("MT6", 4, 9, "blueGem"),
        ("MT1", 7, 3, "redGem"),
        ("MT1", 7, 4, "blueGem"),
        ("MT3", 2, 1, "blueGem"),
            ("MT3", 2, 9, "redGem"),
            ("MT8", 1, 5, "redPotion"),
            ("MT8", 4, 10, "redGem"),
            ("MT8", 5, 11, "blueGem"),
        ("MT8", 8, 10, "redPotion"),
        ("MT8", 7, 10, "blueKey"),
    ):
        assert sim.block_id(state.floors[floor_id][y][x]) in {item_id, None}
        state.floors[floor_id][y][x] = 0

    assert current_stage_name(sim, state) == "pre_mt10_buffer"

    state.floor_id = "MT2"
    state.x = 1
    state.y = 10
    assert current_stage_name(sim, state) == "pre_mt10_buffer"


def test_pre_mt10_buffer_requires_hp_key_and_stat_buffer() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    state.atk = 25
    state.defense = 26
    state.hp = 66
    state.items["yellowKey"] = 2
    state.items["blueKey"] = 1

    assert not stage_complete(sim, state, "pre_mt10_buffer")

    state.hp = 260
    assert stage_complete(sim, state, "pre_mt10_buffer")

    state.items["blueKey"] = 0
    assert not stage_complete(sim, state, "pre_mt10_buffer")

    state.floors["MT9"][11][3] = 0
    assert stage_complete(sim, state, "pre_mt10_buffer")


def test_mt8_hp_ready_requires_right_chain_hp_or_extra_yellow_key() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    state.atk = 22
    state.defense = 23
    for floor_id, x, y, item_id in (
        ("MT1", 7, 3, "redGem"),
        ("MT1", 7, 4, "blueGem"),
        ("MT3", 2, 1, "blueGem"),
        ("MT3", 2, 9, "redGem"),
    ):
        assert sim.block_id(state.floors[floor_id][y][x]) in {item_id, None}
        state.floors[floor_id][y][x] = 0

    state.hp = 273
    state.items["yellowKey"] = 2
    assert not stage_complete(sim, state, "mt8_hp_ready")

    state.items["yellowKey"] = 3
    assert stage_complete(sim, state, "mt8_hp_ready")

    state.items["yellowKey"] = 2
    state.hp = 340
    assert stage_complete(sim, state, "mt8_hp_ready")


def test_critical_gems_stage_uses_thresholds_not_full_map_clear() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    state.atk = 23
    state.defense = 21

    assert critical_gems_ready(state)
    assert not all_attack_defense_gems_taken(sim, state)


def test_guard_ready_includes_red_key_route_blockers() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.hp = 650
    state.atk = 26
    state.defense = 22

    assert yellow_guard_margin(sim, state) > 0
    assert red_key_route_margin(sim, state) < 0
    assert not guard_ready(sim, state)

    state.hp = 900
    assert guard_ready(sim, state)


def test_boss_ready_counts_mt10_trap_damage() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.items["redKey"] = 1
    state.atk = 27
    state.defense = 27
    state.hp = 358

    assert sim.damage_info(state, "skeletonCaptain") == {"damage": 304, "turn": 9}
    assert boss_route_required_damage(sim, state) == 634
    assert boss_route_margin(sim, state) == -276
    assert not boss_ready(sim, state)

    state.hp = 635
    assert boss_ready(sim, state)
