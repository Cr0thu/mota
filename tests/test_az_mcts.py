from __future__ import annotations

import json
from pathlib import Path

import pytest

from mota_env import MotaSimulator, load_game_data
from mota_solver.az_mcts import (
    AlphaMCTS,
    AlphaMCTSConfig,
    LearnedRewardValueFn,
    MCTSNode,
    filter_stage_actions,
    uniform_policy_value,
)


def test_alpha_mcts_returns_legal_macro_action() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    actions = sim.macro_actions(state)
    legal_targets = {tuple(action.get("target", [])) for action in actions}

    mcts = AlphaMCTS(
        sim,
        policy_value_fn=uniform_policy_value,
        config=AlphaMCTSConfig(target_stage="sword", num_simulations=8, max_depth=4, seed=7),
    )
    result = mcts.search(state)

    assert result.action is not None
    assert tuple(result.action.get("target", [])) in legal_targets
    assert result.visit_count > 0
    assert sum(result.policy_target) == 1.0
    assert result.child_stats


def test_alpha_mcts_single_player_backup_adds_edge_rewards() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    mcts = AlphaMCTS(
        sim,
        policy_value_fn=uniform_policy_value,
        config=AlphaMCTSConfig(target_stage="sword", discount=0.9),
    )

    root_node = MCTSNode()
    first_child = MCTSNode()
    second_child = MCTSNode()

    mcts._backpropagate(
        [
            (root_node, 0.0),
            (first_child, 0.5),
            (second_child, 0.25),
        ],
        1.0,
    )

    assert second_child.value == pytest.approx(0.25 + 0.9)
    assert first_child.value == pytest.approx(0.5 + 0.9 * (0.25 + 0.9))
    assert root_node.value == pytest.approx(first_child.value)


def test_learned_reward_leaf_value_is_bounded() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    leaf = LearnedRewardValueFn(
        {
            "global_weights": {"hp": 0.01, "atk": 1.0, "def": 1.0},
            "stage_weights": {},
            "gamma": 0.99,
        },
        scale=1000.0,
    )

    value = leaf(sim, state, "sword")

    assert -1.0 <= value <= 1.0


def test_shield_buffer_filter_continues_after_weak_shield() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT9"
    state.x = 9
    state.y = 7
    state.hp = 72
    state.atk = 20
    state.defense = 20
    state.items["yellowKey"] = 0
    state.items["blueKey"] = 0
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    state.floors["MT9"][1][8] = 0
    state.floors["MT9"][1][9] = 0
    state.floors["MT9"][5][10] = 0
    state.floors["MT9"][7][9] = 0

    labels = [
        action["label"]
        for action in filter_stage_actions(sim.macro_actions(state), state, "shield_buffer", sim=sim)
    ]

    assert labels == ["go redPotion MT9:11,1"]
    assert not any("greenSlime MT9:10,2" in label for label in labels)


def test_shield_buffer_prefers_mt4_red_gem_after_sword_before_climb() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT5"
    state.x = 11
    state.y = 11
    state.hp = 754
    state.atk = 20
    state.defense = 10
    state.items["yellowKey"] = 1
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    actions = [
        {"label": "go downFloor MT5:1,11"},
        {"label": "fight bat MT5:6,4"},
        {"label": "go upFloor MT5:1,1"},
    ]

    labels = [
        action["label"] for action in filter_stage_actions(actions, state, "shield_buffer", sim=sim)
    ]

    assert labels == ["go downFloor MT5:1,11"]


def test_shield_buffer_goes_up_after_mt4_red_gem_pocket() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    route = [
        json.loads(line)
        for line in Path(
            "artifacts/manual_exploration_20260524/"
            "manual_success_no_shop_true_10f_trap_optimized_hp403.jsonl"
        )
        .read_text(encoding="utf8")
        .splitlines()
        if line.strip()
    ]
    state = sim.reset()
    for row in route[:24]:
        transition = sim.apply_macro_action(state, row["action"])
        assert transition.ok
    red_potion = next(
        action for action in sim.macro_actions(state) if action["label"] == "go redPotion MT4:9,10"
    )
    assert sim.apply_macro_action(state, red_potion).ok

    labels = [
        action["label"]
        for action in filter_stage_actions(sim.macro_actions(state), state, "shield_buffer", sim=sim)
    ]

    assert labels == ["go upFloor MT4:1,11"]


def test_shield_buffer_refills_mt4_left_keys_after_mt5_first_key() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    route = [
        json.loads(line)
        for line in Path(
            "artifacts/manual_exploration_20260524/"
            "manual_success_no_shop_true_10f_trap_optimized_hp403.jsonl"
        )
        .read_text(encoding="utf8")
        .splitlines()
        if line.strip()
    ]
    state = sim.reset()
    for row in route[:24]:
        transition = sim.apply_macro_action(state, row["action"])
        assert transition.ok
    red_potion = next(
        action for action in sim.macro_actions(state) if action["label"] == "go redPotion MT4:9,10"
    )
    assert sim.apply_macro_action(state, red_potion).ok
    for label in ("go upFloor MT4:1,11", "fight bat MT5:6,4", "go yellowKey MT5:6,2"):
        action = next(action for action in sim.macro_actions(state) if action["label"] == label)
        transition = sim.apply_macro_action(state, action)
        assert transition.ok

    labels = [
        action["label"]
        for action in filter_stage_actions(sim.macro_actions(state), state, "shield_buffer", sim=sim)
    ]

    assert labels == ["go downFloor MT5:1,11"]

    transition = sim.apply_macro_action(state, route[27]["action"])
    assert transition.ok
    labels = [
        action["label"]
        for action in filter_stage_actions(sim.macro_actions(state), state, "shield_buffer", sim=sim)
    ]

    assert labels == ["open yellowDoor MT4:4,8"]

    door = next(
        action for action in sim.macro_actions(state) if action["label"] == "open yellowDoor MT4:4,8"
    )
    assert sim.apply_macro_action(state, door).ok
    labels = [
        action["label"]
        for action in filter_stage_actions(sim.macro_actions(state), state, "shield_buffer", sim=sim)
    ]

    assert labels == ["fight bat MT4:4,9"]

    bat = next(action for action in sim.macro_actions(state) if action["label"] == "fight bat MT4:4,9")
    assert sim.apply_macro_action(state, bat).ok
    key = next(action for action in sim.macro_actions(state) if action["label"] == "go yellowKey MT4:5,11")
    assert sim.apply_macro_action(state, key).ok
    labels = [
        action["label"]
        for action in filter_stage_actions(sim.macro_actions(state), state, "shield_buffer", sim=sim)
    ]

    assert labels == ["fight greenSlime MT4:3,10"]


def test_shield_buffer_uses_low_damage_mt6_corridor() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    route = [
        json.loads(line)
        for line in Path(
            "artifacts/manual_exploration_20260524/"
            "manual_success_no_shop_true_10f_trap_optimized_hp403.jsonl"
        )
        .read_text(encoding="utf8")
        .splitlines()
        if line.strip()
    ]
    state = sim.reset()
    for row in route[:42]:
        transition = sim.apply_macro_action(state, row["action"])
        assert transition.ok

    labels = [
        action["label"]
        for action in filter_stage_actions(sim.macro_actions(state), state, "shield_buffer", sim=sim)
    ]

    assert labels == ["open yellowDoor MT6:2,4"]
    assert not any("bluePriest" in label or "skeleton" in label for label in labels)


def test_pre_mt10_buffer_returns_to_lower_resources_after_mt7_red_gem() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    route = [
        json.loads(line)
        for line in Path(
            "artifacts/manual_exploration_20260524/"
            "manual_success_no_shop_true_10f_trap_optimized_hp403.jsonl"
        )
        .read_text(encoding="utf8")
        .splitlines()
        if line.strip()
    ]
    state = sim.reset()
    for row in route[:90]:
        transition = sim.apply_macro_action(state, row["action"])
        assert transition.ok

    # Reproduce the high-chain archive state found by v242.  Although the 8F
    # stair is reachable, many lower-floor resources are still uncollected in
    # this route family, so pre-MT10 buffering should continue delegating to the
    # lower-gems stage instead of prematurely forcing the 8F chain.
    state.atk = 23
    state.defense = 23
    state.items["blueKey"] = 1

    labels = [
        action["label"]
        for action in filter_stage_actions(sim.macro_actions(state), state, "pre_mt10_buffer", sim=sim)
    ]

    assert labels == ["go downFloor MT7:11,11"]


def test_pre_mt10_buffer_keeps_descending_to_low_resources_after_mt7_red_gem() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    route = [
        json.loads(line)
        for line in Path(
            "artifacts/manual_exploration_20260524/"
            "manual_success_no_shop_true_10f_trap_optimized_hp403.jsonl"
        )
        .read_text(encoding="utf8")
        .splitlines()
        if line.strip()
    ]
    state = sim.reset()
    for row in route[:90]:
        transition = sim.apply_macro_action(state, row["action"])
        assert transition.ok

    state.atk = 23
    state.defense = 23
    state.items["blueKey"] = 1
    for floor_id, x, y, block_id in (
        ("MT1", 7, 3, "redGem"),
        ("MT1", 7, 4, "blueGem"),
        ("MT3", 2, 9, "redGem"),
        ("MT8", 4, 10, "redGem"),
    ):
        sim.set_tile(state, x, y, block_id, floor_id)

    for expected in (
        "go downFloor MT7:11,11",
        "go downFloor MT6:1,1",
        "go downFloor MT5:1,11",
        "go downFloor MT4:11,11",
    ):
        action = next(action for action in sim.macro_actions(state) if action["label"] == expected)
        transition = sim.apply_macro_action(state, action)
        assert transition.ok

    labels = [
        action["label"]
        for action in filter_stage_actions(sim.macro_actions(state), state, "pre_mt10_buffer", sim=sim)
    ]

    assert labels == ["go downFloor MT3:1,11"]


def test_pre_mt10_buffer_uses_mt1_key_refill_before_central_door_when_keys_tight() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT1"
    state.x = 2
    state.y = 1
    state.hp = 306
    state.atk = 23
    state.defense = 23
    state.items["yellowKey"] = 1
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    actions = [
        {"label": "go upFloor MT1:1,1", "path": ["left"]},
        {"label": "open yellowDoor MT1:10,9", "path": ["right"] * 8},
        {"label": "open yellowDoor MT1:6,9", "path": ["right"] * 4},
        {"label": "open yellowDoor MT1:6,6", "path": ["right"] * 4},
        {"label": "fight skeleton MT1:2,4", "path": ["down"] * 3},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "pre_mt10_buffer", sim=sim)]

    assert labels == ["fight skeleton MT1:2,4"]


def test_pre_mt10_buffer_leaves_mt1_after_key_refill_when_center_needs_two_keys() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT1"
    state.x = 3
    state.y = 10
    state.hp = 252
    state.atk = 23
    state.defense = 23
    state.items["yellowKey"] = 1
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    state.floors["MT1"][4][2] = 0
    state.floors["MT1"][5][2] = 0
    state.floors["MT1"][6][1] = 0
    state.floors["MT1"][7][2] = 0
    state.floors["MT1"][8][2] = 0
    actions = [
        {"label": "go upFloor MT1:1,1", "path": ["up"] * 9 + ["left"] * 2},
        {"label": "open yellowDoor MT1:6,6", "path": ["right"] * 3},
        {"label": "open yellowDoor MT1:10,9", "path": ["right"] * 7},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "pre_mt10_buffer", sim=sim)]

    assert labels == ["go upFloor MT1:1,1"]


def test_pre_mt10_buffer_prioritizes_mt8_blue_key_chain_over_retreat() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT8"
    state.x = 1
    state.y = 2
    state.hp = 271
    state.atk = 22
    state.defense = 23
    state.items["yellowKey"] = 1
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"

    actions = [
        {"label": "go downFloor MT8:1,1"},
        {"label": "fight bluePriest MT8:7,5"},
        {"label": "go upFloor MT8:6,1"},
    ]

    labels = [
        action["label"] for action in filter_stage_actions(actions, state, "pre_mt10_buffer", sim=sim)
    ]

    assert labels == ["fight bluePriest MT8:7,5"]


def test_mt10_resources_underprepared_mt8_returns_to_lower_resources() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT8"
    state.x = 6
    state.y = 2
    state.hp = 352
    state.atk = 22
    state.defense = 21
    state.items["yellowKey"] = 3
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"

    actions = [
        {"label": "fight greenSlime MT8:7,2"},
        {"label": "go upFloor MT8:6,1"},
        {"label": "go downFloor MT8:1,1"},
        {"label": "open yellowDoor MT8:1,3"},
    ]

    labels = [
        action["label"] for action in filter_stage_actions(actions, state, "mt10_resources", sim=sim)
    ]

    assert labels == ["go downFloor MT8:1,1"]


def test_shield_buffer_low_key_filter_keeps_mt6_progress() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT6"
    state.x = 1
    state.y = 2
    state.hp = 566
    state.atk = 20
    state.defense = 10
    state.items["yellowKey"] = 1
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = None
    sim.set_tile(state, 7, 10, 0, "MT4")
    sim.set_tile(state, 9, 10, 0, "MT4")

    actions = [
        {"label": "go downFloor MT6:1,1"},
        {"label": "open yellowDoor MT6:2,4"},
        {"label": "fight redSlime MT6:3,6"},
    ]

    labels = [
        action["label"] for action in filter_stage_actions(actions, state, "shield_buffer", sim=sim)
    ]

    assert labels == ["open yellowDoor MT6:2,4"]

    state.items["yellowKey"] = 0
    labels = [
        action["label"]
        for action in filter_stage_actions(
            [
                {"label": "go downFloor MT6:1,1"},
                {"label": "fight redSlime MT6:3,6"},
            ],
            state,
            "shield_buffer",
            sim=sim,
        )
    ]

    assert labels == ["fight redSlime MT6:3,6"]


def test_shield_buffer_low_hp_after_shield_keeps_resource_doors() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT9"
    state.x = 9
    state.y = 7
    state.hp = 186
    state.atk = 20
    state.defense = 20
    state.items["yellowKey"] = 1
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"

    actions = [
        {"label": "go downFloor MT9:6,1"},
        {"label": "open yellowDoor MT9:9,4"},
        {"label": "open blueDoor MT9:6,3"},
        {"label": "fight greenSlime MT9:10,2"},
    ]

    labels = [
        action["label"] for action in filter_stage_actions(actions, state, "shield_buffer", sim=sim)
    ]

    assert labels == ["go downFloor MT9:6,1"]

    labels = [
        action["label"]
        for action in filter_stage_actions(
            [
                {"label": "go downFloor MT9:6,1"},
                {"label": "fight greenSlime MT9:10,2", "target": ["MT9", 10, 2]},
            ],
            state,
            "shield_buffer",
            sim=sim,
        )
    ]

    assert labels == ["go downFloor MT9:6,1"]


def test_shield_buffer_low_key_uses_mt7_merchant() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT7"
    state.x = 11
    state.y = 10
    state.hp = 186
    state.atk = 20
    state.defense = 20
    state.money = 68
    state.items["yellowKey"] = 1
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    state.floors["MT6"][9][4] = 0

    labels = [
        action["label"]
        for action in filter_stage_actions(
            [
                {"label": "go downFloor MT7:11,11"},
                {"label": "open blueDoor MT7:5,5"},
                {"label": "open yellowDoor MT7:1,7"},
            ],
            state,
            "shield_buffer",
            sim=sim,
        )
    ]

    assert labels == ["open blueDoor MT7:5,5"]

    labels = [
        action["label"]
        for action in filter_stage_actions(
            [
                {"label": "go downFloor MT7:11,11"},
                {"label": "buy 5 yellowKey MT7:6,1"},
            ],
            state,
            "shield_buffer",
            sim=sim,
        )
    ]

    assert labels == ["buy 5 yellowKey MT7:6,1"]


def test_shield_buffer_prefers_mt7_merchant_before_upstairs_when_money_ready() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT7"
    state.x = 7
    state.y = 11
    state.hp = 278
    state.atk = 20
    state.defense = 10
    state.money = 60
    state.items["yellowKey"] = 2
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = None

    labels = [
        action["label"]
        for action in filter_stage_actions(
            [
                {"label": "go upFloor MT7:1,1"},
                {"label": "open blueDoor MT7:5,5"},
            ],
            state,
            "shield_buffer",
            sim=sim,
        )
    ]

    assert labels == ["open blueDoor MT7:5,5"]


def test_shield_buffer_after_mt7_merchant_prefers_stair_corridor() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT7"
    state.x = 6
    state.y = 1
    state.hp = 258
    state.atk = 20
    state.defense = 10
    state.money = 12
    state.items["yellowKey"] = 7
    state.items["blueKey"] = 0
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = None

    labels = [
        action["label"]
        for action in filter_stage_actions(
            [
                {"label": "open yellowDoor MT7:5,7"},
                {"label": "open yellowDoor MT7:3,7"},
                {"label": "fight bluePriest MT7:4,6"},
            ],
            state,
            "shield_buffer",
            sim=sim,
        )
    ]

    assert labels == ["open yellowDoor MT7:3,7"]

    labels = [
        action["label"]
        for action in filter_stage_actions(
            [
                {"label": "open yellowDoor MT7:5,7"},
                {"label": "go upFloor MT7:1,1"},
            ],
            state,
            "shield_buffer",
            sim=sim,
        )
    ]

    assert labels == ["go upFloor MT7:1,1"]


def test_shield_buffer_mt8_collects_key_pocket_before_upstairs() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT8"
    state.x = 6
    state.y = 1
    state.hp = 134
    state.atk = 20
    state.defense = 10
    state.money = 21
    state.items["yellowKey"] = 2
    state.items["blueKey"] = 0
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = None

    labels = [
        action["label"]
        for action in filter_stage_actions(
            [
                {"label": "go upFloor MT8:6,1"},
                {"label": "open yellowDoor MT8:6,3"},
            ],
            state,
            "shield_buffer",
            sim=sim,
        )
    ]

    assert labels == ["open yellowDoor MT8:6,3"]

    labels = [
        action["label"]
        for action in filter_stage_actions(
            [
                {"label": "go upFloor MT8:6,1"},
                {"label": "go yellowKey MT8:3,4"},
            ],
            state,
            "shield_buffer",
            sim=sim,
        )
    ]

    assert labels == ["go yellowKey MT8:3,4"]


def test_shield_buffer_low_hp_on_mt6_prefers_refill_fights_over_stairs() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT6"
    state.x = 11
    state.y = 10
    state.hp = 80
    state.atk = 20
    state.defense = 20
    state.money = 28
    state.items["yellowKey"] = 3
    state.items["blueKey"] = 0
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"

    labels = [
        action["label"]
        for action in filter_stage_actions(
            [
                {"label": "go upFloor MT6:11,11"},
                {"label": "fight redSlime MT6:9,9"},
                {"label": "fight greenSlime MT6:2,11"},
                {"label": "go downFloor MT6:1,1"},
            ],
            state,
            "shield_buffer",
            sim=sim,
        )
    ]

    assert labels == ["fight redSlime MT6:9,9", "fight greenSlime MT6:2,11"]


def test_mt8_stage_filter_routes_up_before_resource_floor() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT4"
    actions = [
        {"label": "go upFloor MT4:1,11"},
        {"label": "go yellowKey MT4:3,2"},
        {"label": "open yellowDoor MT4:4,5"},
        {"label": "fight bat MT4:7,8"},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "mt8_gems")]

    assert labels == ["go yellowKey MT4:3,2"]


def test_lower_gems_filter_keeps_mt5_blue_gem_path_over_stairs() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT5"
    state.x = 1
    state.y = 2
    state.hp = 150
    state.atk = 20
    state.defense = 21
    state.items["yellowKey"] = 3
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    state.floors["MT6"][9][4] = 0

    labels = [
        action["label"]
        for action in filter_stage_actions(
            [
                {"label": "go upFloor MT5:1,1"},
                {"label": "fight bat MT5:4,6"},
                {"label": "fight bluePriest MT5:3,5"},
                {"label": "go downFloor MT5:1,11"},
            ],
            state,
            "lower_gems",
            sim=sim,
        )
    ]

    assert labels == ["fight bat MT5:4,6", "fight bluePriest MT5:3,5"]

    state.x = 3
    state.y = 5
    state.hp = 72
    labels = [
        action["label"]
        for action in filter_stage_actions(
            [
                {"label": "go yellowKey MT5:1,5"},
                {"label": "go yellowKey MT5:1,6"},
                {"label": "go upFloor MT5:1,1"},
                {"label": "go downFloor MT5:1,11"},
            ],
            state,
            "lower_gems",
            sim=sim,
        )
    ]

    assert labels == ["go yellowKey MT5:1,5", "go yellowKey MT5:1,6"]


def test_lower_gems_filter_climbs_to_last_mt7_red_gem() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT1"
    state.x = 1
    state.y = 3
    state.hp = 139
    state.atk = 25
    state.defense = 26
    state.money = 89
    state.items["yellowKey"] = 0
    state.items["blueKey"] = 0
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    for floor_id, x, y in (
        ("MT1", 7, 3),
        ("MT1", 7, 4),
        ("MT3", 2, 1),
        ("MT3", 2, 9),
        ("MT5", 1, 9),
        ("MT6", 4, 9),
        ("MT8", 4, 10),
        ("MT8", 5, 11),
        ("MT9", 1, 5),
        ("MT9", 6, 5),
    ):
        sim.set_tile(state, x, y, 0, floor_id)

    labels = [
        action["label"]
        for action in filter_stage_actions(
            [
                {"label": "fight skeleton MT1:2,4"},
                {"label": "go upFloor MT1:1,1"},
            ],
            state,
            "lower_gems",
            sim=sim,
        )
    ]

    assert labels == ["go upFloor MT1:1,1"]


def test_lower_gems_filter_prioritizes_mt7_red_gem_chain() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT7"
    state.x = 11
    state.y = 11
    state.hp = 139
    state.atk = 25
    state.defense = 26
    state.money = 89
    state.items["yellowKey"] = 0
    state.items["blueKey"] = 0
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    for floor_id, x, y in (
        ("MT1", 7, 3),
        ("MT1", 7, 4),
        ("MT3", 2, 1),
        ("MT3", 2, 9),
        ("MT5", 1, 9),
        ("MT6", 4, 9),
        ("MT8", 4, 10),
        ("MT8", 5, 11),
        ("MT9", 1, 5),
        ("MT9", 6, 5),
    ):
        sim.set_tile(state, x, y, 0, floor_id)

    labels = [
        action["label"]
        for action in filter_stage_actions(
            [
                {"label": "buy 5 yellowKey MT7:6,1"},
                {"label": "open yellowDoor MT7:3,5"},
                {"label": "fight bat MT7:3,3"},
                {"label": "go redPotion MT7:3,2"},
                {"label": "go redGem MT7:3,1"},
                {"label": "fight skeleton MT7:9,5"},
                {"label": "go downFloor MT7:11,11"},
            ],
            state,
            "lower_gems",
            sim=sim,
        )
    ]

    assert labels == [
        "buy 5 yellowKey MT7:6,1",
        "open yellowDoor MT7:3,5",
        "fight bat MT7:3,3",
        "go redPotion MT7:3,2",
        "go redGem MT7:3,1",
    ]


def test_lower_gems_filter_prioritizes_mt9_gems_before_mt7_red_gem() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT9"
    state.x = 6
    state.y = 2
    state.hp = 79
    state.atk = 24
    state.defense = 25
    state.money = 60
    state.items["yellowKey"] = 2
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    for floor_id, x, y in (
        ("MT1", 7, 3),
        ("MT1", 7, 4),
        ("MT3", 2, 1),
        ("MT3", 2, 9),
        ("MT5", 1, 9),
        ("MT6", 4, 9),
        ("MT8", 4, 10),
        ("MT8", 5, 11),
    ):
        sim.set_tile(state, x, y, 0, floor_id)

    labels = [
        action["label"]
        for action in filter_stage_actions(
            [
                {"label": "open blueDoor MT9:6,3"},
                {"label": "open yellowDoor MT9:4,1"},
                {"label": "fight greenSlime MT9:10,2"},
                {"label": "open yellowDoor MT9:9,4"},
                {"label": "go downFloor MT9:6,1"},
            ],
            state,
            "lower_gems",
            sim=sim,
        )
    ]

    assert labels == [
        "open blueDoor MT9:6,3",
        "open yellowDoor MT9:4,1",
        "fight greenSlime MT9:10,2",
        "open yellowDoor MT9:9,4",
    ]


def test_low_gems_filter_keeps_mt1_center_route_after_entry_opened() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT1"
    state.x = 6
    state.y = 6
    state.hp = 221
    state.atk = 22
    state.defense = 22
    state.items["yellowKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    sim.set_tile(state, 6, 6, 0, "MT1")
    sim.set_tile(state, 2, 1, 0, "MT3")
    sim.set_tile(state, 2, 9, 0, "MT3")

    labels = [
        action["label"]
        for action in filter_stage_actions(
            [
                {"label": "fight bat MT1:7,6"},
                {"label": "open yellowDoor MT1:4,3"},
                {"label": "open yellowDoor MT1:10,9"},
                {"label": "go upFloor MT1:1,1"},
            ],
            state,
            "low_gems",
            sim=sim,
        )
    ]

    assert labels == ["fight bat MT1:7,6"]


def test_lower_gems_filter_prefers_visible_low_floor_potion_when_hp_is_low() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT1"
    state.x = 7
    state.y = 4
    state.hp = 149
    state.atk = 23
    state.defense = 23
    state.items["yellowKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    for floor_id, x, y in (
        ("MT1", 7, 3),
        ("MT1", 7, 4),
        ("MT3", 2, 1),
        ("MT3", 2, 9),
        ("MT6", 4, 9),
    ):
        sim.set_tile(state, x, y, 0, floor_id)

    labels = [
        action["label"]
        for action in filter_stage_actions(
            [
                {"label": "go redPotion MT1:8,4"},
                {"label": "open yellowDoor MT1:6,9"},
                {"label": "go upFloor MT1:1,1"},
            ],
            state,
            "lower_gems",
            sim=sim,
        )
    ]

    assert labels == ["go redPotion MT1:8,4"]


def test_lower_gems_filter_prefers_mt8_refill_chain_before_left_bat_when_hp_is_low() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT8"
    state.x = 6
    state.y = 8
    state.hp = 32
    state.atk = 23
    state.defense = 24
    state.items["yellowKey"] = 2
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    for floor_id, x, y in (
        ("MT1", 7, 3),
        ("MT1", 7, 4),
        ("MT3", 2, 1),
        ("MT3", 2, 9),
        ("MT5", 1, 9),
        ("MT6", 4, 9),
    ):
        sim.set_tile(state, x, y, 0, floor_id)

    labels = [
        action["label"]
        for action in filter_stage_actions(
            [
                {"label": "open yellowDoor MT8:5,7"},
                {"label": "fight bat MT8:4,8"},
                {"label": "fight bluePriest MT8:8,8"},
                {"label": "go upFloor MT8:6,1"},
            ],
            state,
            "lower_gems",
            sim=sim,
        )
    ]

    assert labels == ["open yellowDoor MT8:5,7"]


def test_mt10_resources_delegates_to_lower_gems_when_mt9_gems_pending() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT8"
    state.x = 5
    state.y = 11
    state.hp = 79
    state.atk = 24
    state.defense = 25
    state.money = 60
    state.items["yellowKey"] = 2
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    for floor_id, x, y in (
        ("MT1", 7, 3),
        ("MT1", 7, 4),
        ("MT3", 2, 1),
        ("MT3", 2, 9),
        ("MT5", 1, 9),
        ("MT6", 4, 9),
        ("MT8", 4, 10),
        ("MT8", 5, 11),
    ):
        sim.set_tile(state, x, y, 0, floor_id)

    labels = [
        action["label"]
        for action in filter_stage_actions(
            [
                {"label": "open yellowDoor MT8:5,7"},
                {"label": "fight bluePriest MT8:8,8"},
                {"label": "fight greenSlime MT8:7,2"},
                {"label": "go upFloor MT8:6,1"},
                {"label": "go downFloor MT8:1,1"},
                {"label": "open yellowDoor MT8:1,3"},
            ],
            state,
            "mt10_resources",
            sim=sim,
        )
    ]

    assert labels == ["go upFloor MT8:6,1"]


def test_sword_stage_filter_removes_down_stair_and_keeps_progress_actions() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT2"
    actions = [
        {"label": "go downFloor MT2:1,1"},
        {"label": "go upFloor MT2:1,11"},
        {"label": "go event MT2:1,9"},
        {"label": "fight greenSlime MT2:3,3"},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "sword", sim=sim)]

    assert "go downFloor MT2:1,1" not in labels
    assert "go upFloor MT2:1,11" in labels
    assert "go event MT2:1,9" in labels


def test_late_stage_filter_falls_back_to_sword_before_sword_taken() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT2"
    actions = [
        {"label": "go downFloor MT2:1,1"},
        {"label": "go upFloor MT2:1,11"},
        {"label": "go event MT2:1,9"},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "boss", sim=sim)]

    assert labels == ["go upFloor MT2:1,11", "go event MT2:1,9"] or labels == [
        "go event MT2:1,9",
        "go upFloor MT2:1,11",
    ]


def test_mt8_stage_filter_keeps_only_mt8_resource_chain_on_floor() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT8"
    actions = [
        {"label": "go downFloor MT8:1,1"},
        {"label": "go redGem MT8:4,10"},
        {"label": "fight skeletonSoldier MT8:10,11"},
        {"label": "go redPotion MT7:2,2"},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "mt8_gems")]

    assert labels == ["go redGem MT8:4,10", "fight skeletonSoldier MT8:10,11"]


def test_mt8_stage_filter_blocks_stair_loop_when_only_local_interactions_remain() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT8"
    actions = [
        {"label": "go downFloor MT8:1,1"},
        {"label": "go upFloor MT8:6,1"},
        {"label": "fight greenSlime MT8:3,6"},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "mt8_gems")]

    assert labels == ["fight greenSlime MT8:3,6"]


def test_mt8_stage_filter_prefers_right_bottom_resource_door() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT8"
    actions = [
        {"label": "open yellowDoor MT8:10,7"},
        {"label": "open yellowDoor MT8:9,11"},
        {"label": "go upFloor MT8:6,1"},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "mt8_gems")]

    assert labels == ["open yellowDoor MT8:9,11"]


def test_mt8_stage_filter_keeps_right_bottom_blue_key_chain() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT8"
    actions = [
        {"label": "go upFloor MT8:6,1"},
        {"label": "fight bat MT8:7,7"},
        {"label": "fight bluePriest MT8:8,8"},
        {"label": "open yellowDoor MT8:11,9"},
        {"label": "fight skeleton MT8:11,10"},
        {"label": "go blueKey MT8:7,10"},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "mt8_gems")]

    assert labels == [
        "fight bat MT8:7,7",
        "fight bluePriest MT8:8,8",
        "open yellowDoor MT8:11,9",
        "fight skeleton MT8:11,10",
        "go blueKey MT8:7,10",
    ]


def test_mt8_stage_filter_prioritizes_blue_key_chain_before_left_potion_when_keys_tight() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT8"
    state.flags["nowWeapon"] = "sword1"
    state.items["yellowKey"] = 2
    actions = [
        {"label": "open yellowDoor MT8:1,3"},
        {"label": "go redPotion MT8:1,5"},
        {"label": "fight bat MT8:7,7"},
        {"label": "go upFloor MT8:6,1"},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "mt8_gems", sim=sim)]

    assert labels == ["fight bat MT8:7,7"]


def test_mt8_stage_filter_keeps_second_right_bottom_door_for_blue_key() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT8"
    state.flags["nowWeapon"] = "sword1"
    state.items["yellowKey"] = 1
    actions = [
        {"label": "go redPotion MT8:1,5"},
        {"label": "open yellowDoor MT8:9,11"},
        {"label": "fight skeletonSoldier MT8:10,11"},
        {"label": "go blueKey MT8:7,10"},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "mt8_gems", sim=sim)]

    assert labels == [
        "open yellowDoor MT8:9,11",
        "fight skeletonSoldier MT8:10,11",
        "go blueKey MT8:7,10",
    ]


def test_mt8_stage_filter_preserves_last_yellow_key_when_blue_key_blocked_by_hp() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT8"
    state.flags["nowWeapon"] = "sword1"
    state.items["yellowKey"] = 1
    actions = [
        {"label": "open yellowDoor MT8:10,7"},
        {"label": "open yellowDoor MT8:5,7"},
        {"label": "fight bat MT8:4,8"},
        {"label": "go upFloor MT8:6,1"},
        {"label": "go downFloor MT8:1,1"},
        {"label": "open yellowDoor MT8:1,3"},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "mt8_gems", sim=sim)]

    assert labels == ["fight bat MT8:4,8"]


def test_red_key_filter_continues_corridor_after_8f_entry_door_opened() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT8"
    state.x = 8
    state.y = 8
    state.hp = 299
    state.atk = 26
    state.defense = 26
    state.items["yellowKey"] = 0
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    state.floors["MT8"][7][10] = 0
    actions = [
        {"label": "go downFloor MT8:1,1"},
        {"label": "go upFloor MT8:6,1"},
        {"label": "go redPotion MT1:1,3"},
        {"label": "fight yellowGuard MT8:9,5"},
        {"label": "fight yellowGuard MT8:11,5"},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "red_key", sim=sim)]

    assert labels == ["fight yellowGuard MT8:9,5"]


def test_red_key_filter_prefers_8f_left_potion_before_entry_door() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT8"
    state.x = 1
    state.y = 3
    state.hp = 317
    state.atk = 26
    state.defense = 26
    state.items["yellowKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    state.floors["MT8"][3][1] = 0
    actions = [
        {"label": "fight bluePriest MT8:8,8"},
        {"label": "open yellowDoor MT8:10,7"},
        {"label": "go redPotion MT8:1,5"},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "red_key", sim=sim)]

    assert labels == ["go redPotion MT8:1,5"]


def test_red_key_filter_takes_7f_blue_potion_buffer_chain() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT7"
    state.x = 5
    state.y = 11
    state.hp = 189
    state.atk = 26
    state.defense = 26
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    actions = [
        {"label": "go upFloor MT7:1,1"},
        {"label": "go downFloor MT7:11,11"},
        {"label": "open yellowDoor MT7:7,7"},
        {"label": "fight redSlime MT7:7,9"},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "red_key", sim=sim)]

    assert labels == ["open yellowDoor MT7:7,7"]


def test_red_key_filter_takes_7f_blue_potion_after_buffer_slime() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT7"
    state.x = 7
    state.y = 9
    state.hp = 189
    state.atk = 26
    state.defense = 26
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    state.floors["MT7"][7][7] = 0
    state.floors["MT7"][9][7] = 0
    actions = [
        {"label": "go upFloor MT7:1,1"},
        {"label": "go downFloor MT7:11,11"},
        {"label": "go bluePotion MT7:7,11"},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "red_key", sim=sim)]

    assert labels == ["go bluePotion MT7:7,11"]


def test_red_key_filter_keeps_8f_green_slime_after_first_guard() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT8"
    state.x = 11
    state.y = 5
    state.hp = 35
    state.atk = 26
    state.defense = 26
    state.items["yellowKey"] = 0
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    state.floors["MT8"][7][10] = 0
    state.floors["MT8"][5][11] = 0
    actions = [
        {"label": "go downFloor MT8:1,1"},
        {"label": "go upFloor MT8:6,1"},
        {"label": "fight greenSlime MT8:7,2"},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "red_key", sim=sim)]

    assert labels == ["fight greenSlime MT8:7,2"]


def test_red_key_filter_prefers_left_guard_before_right_guard() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT8"
    state.x = 10
    state.y = 7
    state.hp = 349
    state.atk = 26
    state.defense = 26
    state.items["yellowKey"] = 0
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    state.floors["MT8"][7][10] = 0
    actions = [
        {"label": "fight yellowGuard MT8:11,5"},
        {"label": "fight yellowGuard MT8:9,5"},
        {"label": "fight greenSlime MT8:7,2"},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "red_key", sim=sim)]

    assert labels == ["fight yellowGuard MT8:9,5"]


def test_red_key_filter_prefers_left_blue_potion_after_left_guard() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT8"
    state.x = 9
    state.y = 5
    state.hp = 85
    state.atk = 26
    state.defense = 26
    state.items["yellowKey"] = 0
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    state.floors["MT8"][7][10] = 0
    state.floors["MT8"][5][9] = 0
    actions = [
        {"label": "fight yellowGuard MT8:11,5"},
        {"label": "go bluePotion MT8:9,3"},
        {"label": "fight greenSlime MT8:7,2"},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "red_key", sim=sim)]

    assert labels == ["go bluePotion MT8:9,3"]


def test_red_key_filter_uses_last_yellow_key_for_8f_entry_when_hp_is_ready() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT8"
    state.x = 8
    state.y = 8
    state.hp = 299
    state.atk = 26
    state.defense = 26
    state.items["yellowKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    actions = [
        {"label": "go downFloor MT8:1,1"},
        {"label": "go upFloor MT8:6,1"},
        {"label": "go redPotion MT1:1,3"},
        {"label": "fight greenSlime MT8:7,2"},
        {"label": "open yellowDoor MT8:10,7"},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "red_key", sim=sim)]

    assert labels == ["open yellowDoor MT8:10,7"]


def test_mt8_stage_filter_after_blue_key_prioritizes_blue_door_gems() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT8"
    state.flags["nowWeapon"] = "sword1"
    state.items["yellowKey"] = 1
    state.items["blueKey"] = 1
    state.floors["MT8"][10][7] = 0
    actions = [
        {"label": "go redPotion MT8:8,10"},
        {"label": "open blueDoor MT8:3,11"},
        {"label": "go redGem MT8:4,10"},
        {"label": "go blueGem MT8:5,11"},
        {"label": "fight skeleton MT8:6,8"},
        {"label": "open yellowDoor MT8:5,7"},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "mt8_gems", sim=sim)]

    assert labels == [
        "go redPotion MT8:8,10",
        "open blueDoor MT8:3,11",
        "go redGem MT8:4,10",
        "go blueGem MT8:5,11",
    ]


def test_pre_shield_filter_delays_7f_bottom_refill() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT7"
    actions = [
        {"label": "go bluePotion MT7:7,11"},
        {"label": "fight bat MT7:5,9"},
        {"label": "go yellowKey MT7:5,11"},
        {"label": "go upFloor MT7:1,1"},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "pre_shield_gems")]

    assert labels == ["go bluePotion MT7:7,11", "go upFloor MT7:1,1"]


def test_mt10_blue_ready_filter_routes_back_for_blue_key() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT8"
    state.items["blueKey"] = 0
    actions = [
        {"label": "go upFloor MT8:6,1"},
        {"label": "go downFloor MT8:1,1"},
        {"label": "go yellowKey MT8:5,10"},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "mt10_blue_ready")]

    assert labels == ["go downFloor MT8:1,1"]


def test_mt10_blue_ready_filter_prefers_blue_key_action() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT6"
    state.items["blueKey"] = 0
    actions = [
        {"label": "go upFloor MT6:11,11"},
        {"label": "buy blueKey MT6:1,8"},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "mt10_blue_ready")]

    assert labels == ["buy blueKey MT6:1,8"]


def test_mt10_blue_ready_filter_opens_local_mt6_path_when_merchant_blocked() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT6"
    state.items["blueKey"] = 0
    actions = [
        {"label": "go upFloor MT6:11,11"},
        {"label": "go downFloor MT6:1,1"},
        {"label": "fight redSlime MT6:11,9"},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "mt10_blue_ready")]

    assert labels == ["fight redSlime MT6:11,9"]


def test_mt10_yellow_ready_filter_takes_mt6_potion_before_climbing() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT6"
    state.hp = 32
    state.items["yellowKey"] = 1
    state.items["blueKey"] = 1
    actions = [
        {"label": "go redPotion MT6:8,3", "path": ["up"]},
        {"label": "go upFloor MT6:11,11", "path": ["right"] * 26},
        {"label": "go downFloor MT6:1,1", "path": ["left"] * 22},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "mt10_yellow_ready")]

    assert labels == ["go redPotion MT6:8,3"]


def test_mt10_resources_filter_takes_mt6_potion_after_blue_key_buy() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT6"
    state.hp = 128
    state.items["yellowKey"] = 3
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    actions = [
        {"label": "go redPotion MT6:8,3", "path": ["up"]},
        {"label": "go upFloor MT6:11,11", "path": ["right"] * 26},
        {"label": "go downFloor MT6:1,1", "path": ["left"] * 22},
    ]

    labels = [
        action["label"] for action in filter_stage_actions(actions, state, "mt10_resources", sim=sim)
    ]

    assert labels == ["go redPotion MT6:8,3"]


def test_mt10_yellow_ready_filter_routes_back_for_missing_blue_key() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT8"
    state.items["yellowKey"] = 2
    state.items["blueKey"] = 0
    actions = [
        {"label": "go upFloor MT8:6,1", "path": ["up"]},
        {"label": "go downFloor MT8:1,1", "path": ["down"]},
        {"label": "go yellowKey MT8:5,10", "path": ["right"]},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "mt10_yellow_ready")]

    assert labels == ["go downFloor MT8:1,1"]


def test_mt10_yellow_ready_filter_returns_to_7f_merchant_when_stock_is_low() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT9"
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    state.hp = 113
    state.atk = 24
    state.defense = 25
    state.money = 104
    state.items["yellowKey"] = 2
    state.items["blueKey"] = 1
    actions = [
        {"label": "fight skeleton MT9:10,10"},
        {"label": "open yellowDoor MT9:6,11"},
        {"label": "open blueDoor MT9:6,3"},
        {"label": "go downFloor MT9:6,1"},
        {"label": "open yellowDoor MT9:4,1"},
    ]

    labels = [
        action["label"]
        for action in filter_stage_actions(actions, state, "mt10_yellow_ready", sim=sim)
    ]

    assert labels == ["go downFloor MT9:6,1"]


def test_mt10_yellow_ready_filter_backtracks_when_7f_merchant_unreachable() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT7"
    state.x = 5
    state.y = 11
    state.hp = 97
    state.money = 93
    state.items["yellowKey"] = 2
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    actions = [
        {"label": "open yellowDoor MT7:3,5", "path": ["up"] * 8},
        {"label": "open yellowDoor MT7:11,5", "path": ["right"] * 6},
        {"label": "go downFloor MT7:11,11", "path": ["right"] * 6},
        {"label": "go upFloor MT7:1,1", "path": ["up"] * 12},
    ]

    labels = [
        action["label"]
        for action in filter_stage_actions(actions, state, "mt10_yellow_ready", sim=sim)
    ]

    assert labels == ["go downFloor MT7:11,11"]


def test_mt10_yellow_ready_filter_takes_7f_bottom_key_chain_before_backtracking() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT7"
    state.x = 11
    state.y = 10
    state.hp = 110
    state.money = 90
    state.items["yellowKey"] = 1
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    actions = [
        {"label": "go downFloor MT7:11,11", "path": ["down"]},
        {"label": "open yellowDoor MT7:11,5", "path": ["up"] * 5},
        {"label": "fight skeletonSoldier MT7:7,3", "path": ["left"] * 4},
        {"label": "open yellowDoor MT7:5,7", "path": ["left"] * 6},
        {"label": "go upFloor MT7:1,1", "path": ["left"] * 10},
    ]

    labels = [
        action["label"]
        for action in filter_stage_actions(actions, state, "mt10_yellow_ready", sim=sim)
    ]

    assert labels == ["open yellowDoor MT7:5,7"]


def test_mt10_yellow_ready_filter_climbs_from_7f_after_first_1f_refill() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT7"
    state.x = 11
    state.y = 10
    state.hp = 92
    state.atk = 24
    state.defense = 25
    state.money = 104
    state.items["yellowKey"] = 1
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    state.floors["MT1"][4][2] = 0
    state.floors["MT1"][5][2] = 0
    state.floors["MT1"][6][1] = 0
    state.floors["MT4"][8][8] = 0
    state.floors["MT4"][9][8] = 0
    state.floors["MT4"][10][7] = 0
    state.floors["MT4"][10][9] = 0
    state.floors["MT7"][10][5] = 0
    state.floors["MT7"][11][5] = 0
    state.floors["MT6"][3][8] = 0
    actions = [
        {"label": "go downFloor MT7:11,11", "path": ["down"]},
        {"label": "open yellowDoor MT7:11,5", "path": ["up"] * 5},
        {"label": "open yellowDoor MT7:3,5", "path": ["left"] * 8},
        {"label": "go upFloor MT7:1,1", "path": ["left"] * 10},
    ]

    labels = [
        action["label"]
        for action in filter_stage_actions(actions, state, "mt10_yellow_ready", sim=sim)
    ]

    assert labels == ["go upFloor MT7:1,1"]


def test_mt10_yellow_ready_filter_climbs_from_8f_after_first_1f_refill() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT8"
    state.x = 1
    state.y = 2
    state.hp = 92
    state.atk = 24
    state.defense = 25
    state.money = 104
    state.items["yellowKey"] = 1
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    state.floors["MT1"][4][2] = 0
    state.floors["MT1"][5][2] = 0
    state.floors["MT1"][6][1] = 0
    state.floors["MT4"][8][8] = 0
    state.floors["MT4"][9][8] = 0
    state.floors["MT4"][10][7] = 0
    state.floors["MT4"][10][9] = 0
    state.floors["MT7"][10][5] = 0
    state.floors["MT7"][11][5] = 0
    actions = [
        {"label": "go downFloor MT8:1,1", "path": ["down"]},
        {"label": "go upFloor MT8:6,1", "path": ["right"] * 5},
        {"label": "open yellowDoor MT8:10,7", "path": ["right"] * 9},
    ]

    labels = [
        action["label"]
        for action in filter_stage_actions(actions, state, "mt10_yellow_ready", sim=sim)
    ]

    assert labels == ["go upFloor MT8:6,1"]


def test_mt10_yellow_ready_filter_preserves_blue_key_and_backtracks_when_yellow_low() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT9"
    state.x = 6
    state.y = 2
    state.hp = 92
    state.atk = 24
    state.defense = 25
    state.money = 104
    state.items["yellowKey"] = 1
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    state.floors["MT1"][4][2] = 0
    state.floors["MT1"][5][2] = 0
    state.floors["MT1"][6][1] = 0
    state.floors["MT4"][8][8] = 0
    state.floors["MT4"][9][8] = 0
    state.floors["MT4"][10][7] = 0
    state.floors["MT4"][10][9] = 0
    state.floors["MT7"][10][5] = 0
    state.floors["MT7"][11][5] = 0
    actions = [
        {"label": "open blueDoor MT9:6,3", "path": ["down"]},
        {"label": "go downFloor MT9:6,1", "path": ["up"]},
        {"label": "open yellowDoor MT9:4,1", "path": ["left"] * 2},
        {"label": "fight greenSlime MT9:10,2", "path": ["right"] * 4},
        {"label": "open yellowDoor MT9:9,4", "path": ["right"] * 4},
    ]

    labels = [
        action["label"]
        for action in filter_stage_actions(actions, state, "mt10_yellow_ready", sim=sim)
    ]

    assert labels == ["go downFloor MT9:6,1"]


def test_mt10_yellow_ready_filter_returns_to_8f_when_9f_blue_key_spent() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT9"
    state.x = 11
    state.y = 11
    state.hp = 99
    state.atk = 25
    state.defense = 26
    state.money = 118
    state.items["yellowKey"] = 2
    state.items["blueKey"] = 0
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    actions = [
        {"label": "fight bluePriest MT9:11,9", "path": ["up"] * 2},
        {"label": "fight skeleton MT9:10,10", "path": ["left"]},
        {"label": "open yellowDoor MT9:6,11", "path": ["left"] * 5},
        {"label": "go downFloor MT9:6,1", "path": ["up"] * 10},
        {"label": "open yellowDoor MT9:4,1", "path": ["up"] * 10},
    ]

    labels = [
        action["label"]
        for action in filter_stage_actions(actions, state, "mt10_yellow_ready", sim=sim)
    ]

    assert labels == ["go downFloor MT9:6,1"]


def test_mt10_yellow_ready_filter_preserves_last_yellow_for_8f_blue_key_chain() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT9"
    state.x = 7
    state.y = 6
    state.hp = 92
    state.atk = 25
    state.defense = 25
    state.money = 106
    state.items["yellowKey"] = 1
    state.items["blueKey"] = 0
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    actions = [
        {"label": "open yellowDoor MT9:4,5", "path": ["left"] * 3},
        {"label": "fight bat MT9:7,10", "path": ["down"] * 4},
        {"label": "go downFloor MT9:6,1", "path": ["up"] * 5},
    ]

    labels = [
        action["label"]
        for action in filter_stage_actions(actions, state, "mt10_yellow_ready", sim=sim)
    ]

    assert labels == ["go downFloor MT9:6,1"]


def test_mt10_yellow_ready_filter_collects_9f_direct_resources_after_blue_door() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT9"
    state.x = 6
    state.y = 3
    state.hp = 92
    state.atk = 24
    state.defense = 25
    state.money = 104
    state.items["yellowKey"] = 1
    state.items["blueKey"] = 0
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    actions = [
        {"label": "go downFloor MT9:6,1", "path": ["up"]},
        {"label": "go redGem MT9:6,5", "path": ["down"] * 2},
        {"label": "go yellowKey MT9:5,4", "path": ["left"]},
        {"label": "go yellowKey MT9:7,4", "path": ["right"]},
        {"label": "open yellowDoor MT9:4,5", "path": ["left"] * 2},
    ]

    labels = [
        action["label"]
        for action in filter_stage_actions(actions, state, "mt10_yellow_ready", sim=sim)
    ]

    assert labels == [
        "go redGem MT9:6,5",
        "go yellowKey MT9:5,4",
        "go yellowKey MT9:7,4",
    ]


def test_mt10_yellow_ready_filter_continues_down_from_6f_after_7f_keys() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT6"
    state.x = 11
    state.y = 10
    state.hp = 97
    state.money = 93
    state.items["yellowKey"] = 2
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    state.floors["MT6"][3][8] = 0
    actions = [
        {"label": "go upFloor MT6:11,11", "path": ["down"]},
        {"label": "fight skeleton MT6:8,6", "path": ["left"] * 3},
        {"label": "open yellowDoor MT6:2,4", "path": ["left"] * 9},
        {"label": "go downFloor MT6:1,1", "path": ["left"] * 10},
    ]

    labels = [
        action["label"]
        for action in filter_stage_actions(actions, state, "mt10_yellow_ready", sim=sim)
    ]

    assert labels == ["go downFloor MT6:1,1"]


def test_mt10_yellow_ready_filter_continues_down_from_5f_after_7f_keys() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT5"
    state.x = 1
    state.y = 2
    state.hp = 97
    state.money = 93
    state.items["yellowKey"] = 2
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    state.floors["MT6"][3][8] = 0
    actions = [
        {"label": "go upFloor MT5:1,1", "path": ["up"]},
        {"label": "fight bluePriest MT5:3,5", "path": ["right"] * 4},
        {"label": "go downFloor MT5:1,11", "path": ["down"] * 9},
    ]

    labels = [
        action["label"]
        for action in filter_stage_actions(actions, state, "mt10_yellow_ready", sim=sim)
    ]

    assert labels == ["go downFloor MT5:1,11"]


def test_mt10_yellow_ready_filter_starts_low_floor_refill_on_1f() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT1"
    state.x = 2
    state.y = 1
    state.hp = 97
    state.money = 93
    state.items["yellowKey"] = 2
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    actions = [
        {"label": "go upFloor MT1:1,1", "path": ["left"]},
        {"label": "go king MT1:7,10", "path": ["right"] * 6},
        {"label": "fight skeleton MT1:2,4", "path": ["down"] * 3},
    ]

    labels = [
        action["label"]
        for action in filter_stage_actions(actions, state, "mt10_yellow_ready", sim=sim)
    ]

    assert labels == ["fight skeleton MT1:2,4"]


def test_mt10_yellow_ready_filter_climbs_from_2f_when_no_local_refill() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT2"
    state.x = 1
    state.y = 2
    state.hp = 63
    state.money = 99
    state.items["yellowKey"] = 2
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    state.floors["MT6"][3][8] = 0
    actions = [
        {"label": "go downFloor MT2:1,1", "path": ["up"]},
        {"label": "go upFloor MT2:1,11", "path": ["down"] * 9},
    ]

    labels = [
        action["label"]
        for action in filter_stage_actions(actions, state, "mt10_yellow_ready", sim=sim)
    ]

    assert labels == ["go upFloor MT2:1,11"]


def test_mt10_yellow_ready_filter_climbs_from_3f_when_no_local_refill() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT3"
    state.x = 2
    state.y = 11
    state.hp = 97
    state.money = 93
    state.items["yellowKey"] = 2
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    state.floors["MT6"][3][8] = 0
    actions = [
        {"label": "go downFloor MT3:1,11", "path": ["left"]},
        {"label": "open yellowDoor MT3:9,8", "path": ["right"] * 7},
        {"label": "go upFloor MT3:11,11", "path": ["right"] * 9},
    ]

    labels = [
        action["label"]
        for action in filter_stage_actions(actions, state, "mt10_yellow_ready", sim=sim)
    ]

    assert labels == ["go upFloor MT3:11,11"]


def test_mt10_yellow_ready_filter_takes_mt4_red_gem_door_before_backtracking() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT4"
    state.x = 1
    state.y = 10
    state.hp = 97
    state.money = 93
    state.items["yellowKey"] = 2
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    state.floors["MT6"][3][8] = 0
    actions = [
        {"label": "go upFloor MT4:1,11", "path": ["down"]},
        {"label": "fight redSlime MT4:6,5", "path": ["right"] * 5},
        {"label": "fight greenSlime MT4:3,10", "path": ["right"] * 2},
        {"label": "open yellowDoor MT4:8,8", "path": ["right"] * 7},
        {"label": "go downFloor MT4:11,11", "path": ["right"] * 10},
    ]

    labels = [
        action["label"]
        for action in filter_stage_actions(actions, state, "mt10_yellow_ready", sim=sim)
    ]

    assert labels == ["open yellowDoor MT4:8,8"]


def test_mt10_yellow_ready_filter_continues_mt4_red_gem_pocket_after_door() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT4"
    state.x = 8
    state.y = 8
    state.hp = 97
    state.money = 93
    state.items["yellowKey"] = 1
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    state.floors["MT4"][8][8] = 0
    state.floors["MT6"][3][8] = 0
    actions = [
        {"label": "fight bluePriest MT4:8,9", "path": ["down"]},
        {"label": "fight greenSlime MT4:3,10", "path": ["left"] * 5},
        {"label": "go downFloor MT4:11,11", "path": ["right"] * 3},
        {"label": "go upFloor MT4:1,11", "path": ["left"] * 7},
    ]

    labels = [
        action["label"]
        for action in filter_stage_actions(actions, state, "mt10_yellow_ready", sim=sim)
    ]

    assert labels == ["fight bluePriest MT4:8,9"]


def test_mt10_yellow_ready_filter_climbs_when_blue_and_yellow_buffer_are_ready() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT4"
    state.x = 1
    state.y = 10
    state.hp = 160
    state.atk = 22
    state.defense = 24
    state.items["yellowKey"] = 3
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    actions = [
        {"label": "go upFloor MT4:1,11", "path": ["down"]},
        {"label": "open yellowDoor MT4:8,8", "path": ["right"] * 8},
        {"label": "fight redSlime MT4:6,5", "path": ["right"] * 5},
    ]

    labels = [
        action["label"]
        for action in filter_stage_actions(actions, state, "mt10_yellow_ready", sim=sim)
    ]

    assert labels == ["go upFloor MT4:1,11"]


def test_mt10_yellow_ready_filter_continues_down_from_6f_when_key_stock_is_one() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT6"
    state.x = 11
    state.y = 10
    state.hp = 126
    state.atk = 24
    state.defense = 25
    state.money = 98
    state.items["yellowKey"] = 1
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    state.floors["MT6"][3][8] = 0
    state.floors["MT7"][10][5] = 0
    state.floors["MT7"][11][5] = 0
    actions = [
        {"label": "go upFloor MT6:11,11", "path": ["down"]},
        {"label": "fight skeletonSoldier MT6:9,6", "path": ["left"] * 2},
        {"label": "go downFloor MT6:1,1", "path": ["left"] * 10},
    ]

    labels = [
        action["label"]
        for action in filter_stage_actions(actions, state, "mt10_yellow_ready", sim=sim)
    ]

    assert labels == ["go downFloor MT6:1,1"]


def test_mt10_yellow_ready_filter_does_not_skip_7f_bottom_keys_from_6f() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT6"
    state.x = 11
    state.y = 10
    state.hp = 126
    state.atk = 24
    state.defense = 25
    state.money = 98
    state.items["yellowKey"] = 1
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    state.floors["MT6"][3][8] = 0
    actions = [
        {"label": "go upFloor MT6:11,11", "path": ["down"]},
        {"label": "go downFloor MT6:1,1", "path": ["left"] * 10},
    ]

    labels = [
        action["label"]
        for action in filter_stage_actions(actions, state, "mt10_yellow_ready", sim=sim)
    ]

    assert labels == ["go upFloor MT6:11,11"]


def test_mt10_yellow_ready_filter_descends_from_4f_after_red_gem_pocket() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT4"
    state.x = 1
    state.y = 10
    state.hp = 126
    state.atk = 24
    state.defense = 25
    state.money = 98
    state.items["yellowKey"] = 1
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    state.floors["MT4"][8][8] = 0
    state.floors["MT4"][9][8] = 0
    state.floors["MT4"][10][7] = 0
    state.floors["MT4"][10][9] = 0
    state.floors["MT6"][3][8] = 0
    state.floors["MT7"][10][5] = 0
    state.floors["MT7"][11][5] = 0
    actions = [
        {"label": "go upFloor MT4:1,11", "path": ["down"]},
        {"label": "fight greenSlime MT4:3,10", "path": ["right"] * 2},
        {"label": "go downFloor MT4:11,11", "path": ["right"] * 10},
    ]

    labels = [
        action["label"]
        for action in filter_stage_actions(actions, state, "mt10_yellow_ready", sim=sim)
    ]

    assert labels == ["go downFloor MT4:11,11"]


def test_mt10_yellow_ready_filter_climbs_from_4f_after_first_1f_refill() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT4"
    state.x = 11
    state.y = 10
    state.hp = 92
    state.atk = 24
    state.defense = 25
    state.money = 104
    state.items["yellowKey"] = 1
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    state.floors["MT1"][4][2] = 0
    state.floors["MT1"][5][2] = 0
    state.floors["MT1"][6][1] = 0
    state.floors["MT4"][8][8] = 0
    state.floors["MT4"][9][8] = 0
    state.floors["MT4"][10][7] = 0
    state.floors["MT4"][10][9] = 0
    state.floors["MT7"][10][5] = 0
    state.floors["MT7"][11][5] = 0
    actions = [
        {"label": "go downFloor MT4:11,11", "path": ["down"]},
        {"label": "fight greenSlime MT4:8,11", "path": ["left"] * 3},
        {"label": "go upFloor MT4:1,11", "path": ["left"] * 10},
    ]

    labels = [
        action["label"]
        for action in filter_stage_actions(actions, state, "mt10_yellow_ready", sim=sim)
    ]

    assert labels == ["go upFloor MT4:1,11"]


def test_mt10_yellow_ready_filter_climbs_from_5f_after_first_1f_refill() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT5"
    state.x = 2
    state.y = 11
    state.hp = 92
    state.atk = 24
    state.defense = 25
    state.money = 104
    state.items["yellowKey"] = 1
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    state.floors["MT1"][4][2] = 0
    state.floors["MT1"][5][2] = 0
    state.floors["MT1"][6][1] = 0
    state.floors["MT4"][8][8] = 0
    state.floors["MT4"][9][8] = 0
    state.floors["MT4"][10][7] = 0
    state.floors["MT4"][10][9] = 0
    state.floors["MT7"][10][5] = 0
    state.floors["MT7"][11][5] = 0
    state.floors["MT6"][3][8] = 0
    actions = [
        {"label": "go downFloor MT5:1,11", "path": ["down"]},
        {"label": "fight bluePriest MT5:3,5", "path": ["right"] * 4},
        {"label": "go upFloor MT5:1,1", "path": ["up"] * 10},
    ]

    labels = [
        action["label"]
        for action in filter_stage_actions(actions, state, "mt10_yellow_ready", sim=sim)
    ]

    assert labels == ["go upFloor MT5:1,1"]


def test_mt10_yellow_ready_filter_descends_from_3f_after_mt4_pocket() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT3"
    state.x = 10
    state.y = 11
    state.hp = 126
    state.atk = 24
    state.defense = 25
    state.money = 98
    state.items["yellowKey"] = 1
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    state.floors["MT4"][8][8] = 0
    state.floors["MT4"][9][8] = 0
    state.floors["MT4"][10][7] = 0
    state.floors["MT4"][10][9] = 0
    state.floors["MT7"][10][5] = 0
    state.floors["MT7"][11][5] = 0
    actions = [
        {"label": "go upFloor MT3:11,11", "path": ["right"]},
        {"label": "open yellowDoor MT3:9,8", "path": ["up"] * 3},
        {"label": "go downFloor MT3:1,11", "path": ["left"] * 9},
    ]

    labels = [
        action["label"]
        for action in filter_stage_actions(actions, state, "mt10_yellow_ready", sim=sim)
    ]

    assert labels == ["go downFloor MT3:1,11"]


def test_mt10_yellow_ready_filter_climbs_from_3f_after_first_1f_refill() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT3"
    state.x = 2
    state.y = 11
    state.hp = 92
    state.atk = 24
    state.defense = 25
    state.money = 104
    state.items["yellowKey"] = 1
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    state.floors["MT1"][4][2] = 0
    state.floors["MT1"][5][2] = 0
    state.floors["MT1"][6][1] = 0
    state.floors["MT4"][8][8] = 0
    state.floors["MT4"][9][8] = 0
    state.floors["MT4"][10][7] = 0
    state.floors["MT4"][10][9] = 0
    state.floors["MT7"][10][5] = 0
    state.floors["MT7"][11][5] = 0
    actions = [
        {"label": "go downFloor MT3:1,11", "path": ["left"]},
        {"label": "open yellowDoor MT3:9,8", "path": ["right"] * 7},
        {"label": "go upFloor MT3:11,11", "path": ["right"] * 9},
    ]

    labels = [
        action["label"]
        for action in filter_stage_actions(actions, state, "mt10_yellow_ready", sim=sim)
    ]

    assert labels == ["go upFloor MT3:11,11"]


def test_mt10_yellow_ready_filter_descends_from_2f_after_mt4_pocket() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT2"
    state.x = 1
    state.y = 10
    state.hp = 126
    state.atk = 24
    state.defense = 25
    state.money = 98
    state.items["yellowKey"] = 1
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    state.floors["MT4"][8][8] = 0
    state.floors["MT4"][9][8] = 0
    state.floors["MT4"][10][7] = 0
    state.floors["MT4"][10][9] = 0
    state.floors["MT7"][10][5] = 0
    state.floors["MT7"][11][5] = 0
    actions = [
        {"label": "go upFloor MT2:1,11", "path": ["down"]},
        {"label": "go downFloor MT2:1,1", "path": ["up"] * 9},
    ]

    labels = [
        action["label"]
        for action in filter_stage_actions(actions, state, "mt10_yellow_ready", sim=sim)
    ]

    assert labels == ["go downFloor MT2:1,1"]


def test_mt10_yellow_ready_filter_climbs_from_2f_after_first_1f_refill() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT2"
    state.x = 1
    state.y = 2
    state.hp = 92
    state.atk = 24
    state.defense = 25
    state.money = 104
    state.items["yellowKey"] = 1
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    state.floors["MT1"][4][2] = 0
    state.floors["MT1"][5][2] = 0
    state.floors["MT1"][6][1] = 0
    state.floors["MT4"][8][8] = 0
    state.floors["MT4"][9][8] = 0
    state.floors["MT4"][10][7] = 0
    state.floors["MT4"][10][9] = 0
    state.floors["MT7"][10][5] = 0
    state.floors["MT7"][11][5] = 0
    actions = [
        {"label": "go downFloor MT2:1,1", "path": ["up"]},
        {"label": "go upFloor MT2:1,11", "path": ["down"] * 9},
    ]

    labels = [
        action["label"]
        for action in filter_stage_actions(actions, state, "mt10_yellow_ready", sim=sim)
    ]

    assert labels == ["go upFloor MT2:1,11"]


def test_mt10_yellow_ready_filter_starts_1f_refill_after_low_descent() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT1"
    state.x = 2
    state.y = 1
    state.hp = 126
    state.atk = 24
    state.defense = 25
    state.money = 98
    state.items["yellowKey"] = 1
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    state.floors["MT4"][8][8] = 0
    state.floors["MT4"][9][8] = 0
    state.floors["MT4"][10][7] = 0
    state.floors["MT4"][10][9] = 0
    state.floors["MT7"][10][5] = 0
    state.floors["MT7"][11][5] = 0
    actions = [
        {"label": "go upFloor MT1:1,1", "path": ["left"]},
        {"label": "go king MT1:7,10", "path": ["right"] * 5},
        {"label": "fight skeleton MT1:2,4", "path": ["down"] * 3},
    ]

    labels = [
        action["label"]
        for action in filter_stage_actions(actions, state, "mt10_yellow_ready", sim=sim)
    ]

    assert labels == ["fight skeleton MT1:2,4"]


def test_mt10_yellow_ready_filter_harvests_9f_after_first_blue_spent() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT9"
    state.items["yellowKey"] = 2
    state.items["blueKey"] = 0
    actions = [
        {"label": "go downFloor MT9:6,1", "path": ["up"]},
        {"label": "open yellowDoor MT9:4,5", "path": ["left"] * 4},
        {"label": "fight redSlime MT9:7,6", "path": ["down"] * 4},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "mt10_yellow_ready")]

    assert labels == ["open yellowDoor MT9:4,5", "fight redSlime MT9:7,6"]


def test_mt10_yellow_ready_filter_preserves_blue_key_for_10f_stair() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT9"
    state.items["yellowKey"] = 2
    state.items["blueKey"] = 1
    actions = [
        {"label": "open blueDoor MT9:6,3", "path": ["down"]},
        {"label": "fight redSlime MT9:7,6", "path": ["down"] * 4},
        {"label": "go downFloor MT9:6,1", "path": ["up"]},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "mt10_yellow_ready")]

    assert labels == ["fight redSlime MT9:7,6"]


def test_mt10_yellow_ready_filter_prefers_9f_right_key_potion_chain() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT9"
    state.items["yellowKey"] = 2
    state.items["blueKey"] = 1
    actions = [
        {"label": "open blueDoor MT9:6,3", "path": ["up"]},
        {"label": "open yellowDoor MT9:8,11", "path": ["right"]},
        {"label": "fight redSlime MT9:7,6", "path": ["down"] * 4},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "mt10_yellow_ready")]

    assert labels == ["open yellowDoor MT9:8,11"]


def test_mt10_yellow_ready_filter_continues_9f_right_chain_after_entry_opened() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT9"
    state.x = 9
    state.y = 4
    state.hp = 123
    state.atk = 22
    state.defense = 24
    state.items["yellowKey"] = 2
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    state.floors["MT9"][4][9] = 0
    actions = [
        {"label": "fight greenSlime MT9:10,2", "path": ["right", "up", "up"]},
        {"label": "go downFloor MT9:6,1", "path": ["left"]},
        {"label": "open blueDoor MT9:6,3", "path": ["left", "down"]},
    ]

    labels = [
        action["label"]
        for action in filter_stage_actions(actions, state, "mt10_yellow_ready", sim=sim)
    ]

    assert labels == ["fight greenSlime MT9:10,2"]


def test_mt10_yellow_ready_filter_opens_9f_key_supply_after_green_slime() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT9"
    state.x = 10
    state.y = 2
    state.hp = 123
    state.atk = 22
    state.defense = 24
    state.money = 68
    state.items["yellowKey"] = 2
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    state.floors["MT9"][4][9] = 0
    state.floors["MT9"][2][10] = 0
    actions = [
        {"label": "go downFloor MT9:6,1", "path": ["left"]},
        {"label": "open yellowDoor MT9:8,4", "path": ["down"]},
        {"label": "open blueDoor MT9:6,3", "path": ["left", "down"]},
        {"label": "open yellowDoor MT9:4,1", "path": ["left"] * 5},
    ]

    labels = [
        action["label"]
        for action in filter_stage_actions(actions, state, "mt10_yellow_ready", sim=sim)
    ]

    assert labels == ["open yellowDoor MT9:8,4"]


def test_mt10_yellow_ready_filter_starts_9f_supply_with_two_yellow_keys() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT9"
    state.x = 6
    state.y = 2
    state.hp = 141
    state.atk = 23
    state.defense = 24
    state.money = 72
    state.items["yellowKey"] = 2
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    actions = [
        {"label": "open blueDoor MT9:6,3", "path": ["down"]},
        {"label": "go downFloor MT9:6,1", "path": ["up"]},
        {"label": "open yellowDoor MT9:4,1", "path": ["left"] * 2},
        {"label": "fight greenSlime MT9:10,2", "path": ["right"] * 4},
        {"label": "open yellowDoor MT9:9,4", "path": ["right", "down"]},
    ]

    labels = [
        action["label"]
        for action in filter_stage_actions(actions, state, "mt10_yellow_ready", sim=sim)
    ]

    assert labels == ["open yellowDoor MT9:9,4"]


def test_mt10_yellow_ready_filter_takes_reachable_9f_keys_before_merchant_backtrack() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT9"
    state.x = 8
    state.y = 4
    state.hp = 141
    state.atk = 23
    state.defense = 24
    state.money = 73
    state.items["yellowKey"] = 0
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    state.floors["MT9"][4][9] = 0
    state.floors["MT9"][2][10] = 0
    state.floors["MT9"][4][8] = 0
    actions = [
        {"label": "go yellowKey MT9:7,4", "path": ["left"]},
        {"label": "fight redSlime MT9:7,6", "path": ["down"]},
        {"label": "open blueDoor MT9:6,3", "path": ["left", "up"]},
        {"label": "go redGem MT9:6,5", "path": ["left", "down"]},
        {"label": "go yellowKey MT9:5,4", "path": ["left"] * 3},
        {"label": "go downFloor MT9:6,1", "path": ["up"]},
    ]

    labels = [
        action["label"]
        for action in filter_stage_actions(actions, state, "mt10_yellow_ready", sim=sim)
    ]

    assert labels == ["go yellowKey MT9:7,4", "go yellowKey MT9:5,4"]


def test_mt10_yellow_ready_filter_refills_before_low_hp_9f_soldier() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT9"
    state.x = 6
    state.y = 5
    state.hp = 123
    state.atk = 23
    state.defense = 24
    state.money = 68
    state.items["yellowKey"] = 3
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    state.floors["MT9"][4][9] = 0
    state.floors["MT9"][2][10] = 0
    state.floors["MT9"][4][8] = 0
    state.floors["MT9"][4][7] = 0
    state.floors["MT9"][4][5] = 0
    state.floors["MT9"][5][6] = 0
    actions = [
        {"label": "open blueDoor MT9:6,3", "path": ["up"]},
        {"label": "open yellowDoor MT9:4,5", "path": ["left"]},
        {"label": "fight redSlime MT9:7,6", "path": ["down"]},
        {"label": "fight skeletonSoldier MT9:11,6", "path": ["right"] * 5},
        {"label": "go downFloor MT9:6,1", "path": ["up"] * 4},
        {"label": "open yellowDoor MT9:4,1", "path": ["left"] * 4},
    ]

    labels = [
        action["label"]
        for action in filter_stage_actions(actions, state, "mt10_yellow_ready", sim=sim)
    ]

    assert labels == ["go downFloor MT9:6,1"]


def test_mt10_yellow_ready_filter_fights_9f_soldier_when_hp_covers_priest() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT9"
    state.x = 6
    state.y = 5
    state.hp = 141
    state.atk = 24
    state.defense = 24
    state.money = 73
    state.items["yellowKey"] = 2
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    state.floors["MT9"][4][9] = 0
    state.floors["MT9"][2][10] = 0
    state.floors["MT9"][4][8] = 0
    state.floors["MT9"][4][7] = 0
    state.floors["MT9"][4][5] = 0
    state.floors["MT9"][5][6] = 0
    actions = [
        {"label": "open blueDoor MT9:6,3", "path": ["up"]},
        {"label": "open yellowDoor MT9:4,5", "path": ["left"]},
        {"label": "fight redSlime MT9:7,6", "path": ["down"]},
        {"label": "fight skeletonSoldier MT9:11,6", "path": ["right"] * 5},
        {"label": "go downFloor MT9:6,1", "path": ["up"] * 4},
    ]

    labels = [
        action["label"]
        for action in filter_stage_actions(actions, state, "mt10_yellow_ready", sim=sim)
    ]

    assert labels == ["fight skeletonSoldier MT9:11,6"]


def test_mt10_resources_filter_returns_to_merchant_when_yellow_key_empty() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT9"
    state.x = 1
    state.y = 10
    state.hp = 67
    state.atk = 24
    state.defense = 24
    state.money = 96
    state.items["yellowKey"] = 0
    state.items["blueKey"] = 0
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    actions = [
        {"label": "go upFloor MT9:1,11", "path": ["down"]},
        {"label": "fight skeleton MT9:5,10", "path": ["right"] * 4},
        {"label": "go downFloor MT9:6,1", "path": ["right"] * 5},
    ]

    labels = [
        action["label"]
        for action in filter_stage_actions(actions, state, "mt10_resources", sim=sim)
    ]

    assert labels == ["go downFloor MT9:6,1"]


def test_mt10_resources_filter_buys_yellow_keys_on_7f_when_empty() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT7"
    state.x = 6
    state.y = 1
    state.hp = 67
    state.atk = 24
    state.defense = 24
    state.money = 96
    state.items["yellowKey"] = 0
    state.items["blueKey"] = 0
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    actions = [
        {"label": "go downFloor MT7:11,11", "path": ["down"]},
        {"label": "go upFloor MT7:1,1", "path": ["up"]},
        {"label": "go buy 5 yellowKey MT7:6,1", "path": []},
    ]

    labels = [
        action["label"]
        for action in filter_stage_actions(actions, state, "mt10_resources", sim=sim)
    ]

    assert labels == ["go buy 5 yellowKey MT7:6,1"]


def test_mt10_yellow_ready_filter_skips_unneeded_9f_8_11_door() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT9"
    state.x = 9
    state.y = 11
    state.hp = 31
    state.atk = 24
    state.defense = 24
    state.money = 91
    state.items["yellowKey"] = 2
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    for x, y in ((9, 4), (10, 2), (8, 4), (7, 4), (5, 4), (6, 5), (11, 6), (11, 8), (11, 9), (9, 9), (11, 11), (9, 11)):
        state.floors["MT9"][y][x] = 0
    actions = [
        {"label": "open yellowDoor MT9:8,11", "path": ["left"]},
        {"label": "fight redSlime MT9:7,6", "path": ["up"]},
        {"label": "open blueDoor MT9:6,3", "path": ["up"]},
        {"label": "open yellowDoor MT9:4,5", "path": ["left"]},
        {"label": "go downFloor MT9:6,1", "path": ["up"]},
    ]

    labels = [
        action["label"]
        for action in filter_stage_actions(actions, state, "mt10_yellow_ready", sim=sim)
    ]

    assert labels == ["fight redSlime MT9:7,6"]


def test_mt10_yellow_ready_filter_climbs_after_mt4_refill_with_two_yellow_keys() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT4"
    state.x = 9
    state.y = 10
    state.hp = 141
    state.atk = 23
    state.defense = 24
    state.money = 72
    state.items["yellowKey"] = 2
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    state.floors["MT4"][10][7] = 0
    state.floors["MT4"][10][9] = 0
    actions = [
        {"label": "fight greenSlime MT4:8,11", "path": ["down"]},
        {"label": "fight greenSlime MT4:3,10", "path": ["left"] * 6},
        {"label": "go downFloor MT4:11,11", "path": ["right"]},
        {"label": "fight redSlime MT4:6,5", "path": ["left", "up"]},
        {"label": "go upFloor MT4:1,11", "path": ["left"] * 8},
    ]

    labels = [
        action["label"]
        for action in filter_stage_actions(actions, state, "mt10_yellow_ready", sim=sim)
    ]

    assert labels == ["go upFloor MT4:1,11"]


def test_mt10_yellow_ready_filter_prefers_correct_9f_blue_door() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT9"
    state.items["yellowKey"] = 0
    state.items["blueKey"] = 1
    actions = [
        {"label": "open blueDoor MT9:6,3", "path": ["up"]},
        {"label": "open blueDoor MT9:3,11", "path": ["left"]},
        {"label": "fight bluePriest MT9:9,11", "path": ["right"]},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "mt10_yellow_ready")]

    assert labels == ["open blueDoor MT9:3,11"]


def test_mt10_yellow_ready_filter_climbs_after_correct_blue_door_opened() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT9"
    state.items["yellowKey"] = 1
    state.items["blueKey"] = 0
    actions = [
        {"label": "open yellowDoor MT9:8,11", "path": ["right"]},
        {"label": "go upFloor MT9:1,11", "path": ["left"]},
        {"label": "go downFloor MT9:6,1", "path": ["up"]},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "mt10_yellow_ready")]

    assert labels == ["go upFloor MT9:1,11"]


def test_mt10_yellow_ready_filter_takes_9f_potion_before_climbing() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT9"
    state.items["yellowKey"] = 1
    state.items["blueKey"] = 0
    actions = [
        {"label": "go redPotion MT9:2,10", "path": ["left"]},
        {"label": "go upFloor MT9:1,11", "path": ["left", "down"]},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "mt10_yellow_ready")]

    assert labels == ["go redPotion MT9:2,10"]


def test_mt10_yellow_ready_filter_recovers_8f_second_blue_key() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT8"
    state.items["yellowKey"] = 2
    state.items["blueKey"] = 0
    actions = [
        {"label": "go upFloor MT8:6,1", "path": ["up"]},
        {"label": "go downFloor MT8:1,1", "path": ["down"]},
        {"label": "fight skeletonSoldier MT8:10,11", "path": ["right"] * 4},
        {"label": "go blueKey MT8:7,10", "path": ["right"] * 2},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "mt10_yellow_ready")]

    assert labels == ["go blueKey MT8:7,10"]


def test_mt10_yellow_ready_filter_enters_8f_blue_key_chain_when_blue_spent() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT8"
    state.x = 6
    state.y = 2
    state.hp = 92
    state.atk = 25
    state.defense = 25
    state.items["yellowKey"] = 1
    state.items["blueKey"] = 0
    actions = [
        {"label": "go upFloor MT8:6,1", "path": ["up"]},
        {"label": "go downFloor MT8:1,1", "path": ["left"] * 5},
        {"label": "open yellowDoor MT8:10,7", "path": ["right"] * 4},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "mt10_yellow_ready")]

    assert labels == ["go downFloor MT8:1,1"]


def test_pre_mt10_buffer_filter_climbs_to_pending_mt9_gems() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT8"
    state.x = 5
    state.y = 11
    state.hp = 79
    state.atk = 24
    state.defense = 25
    state.money = 60
    state.items["yellowKey"] = 2
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    for floor_id, x, y in (
        ("MT5", 1, 9),
        ("MT6", 4, 9),
        ("MT1", 7, 3),
        ("MT1", 7, 4),
        ("MT3", 2, 1),
        ("MT3", 2, 9),
        ("MT8", 4, 10),
        ("MT8", 5, 11),
    ):
        state.floors[floor_id][y][x] = 0
    actions = [
        {"label": "go upFloor MT8:6,1", "path": ["up"]},
        {"label": "go downFloor MT8:1,1", "path": ["left"]},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "pre_mt10_buffer", sim=sim)]

    assert labels == ["go upFloor MT8:6,1"]


def test_pre_mt10_buffer_filter_takes_mt8_blue_key_before_mt9_blue_door() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT8"
    state.x = 6
    state.y = 2
    state.hp = 79
    state.atk = 24
    state.defense = 25
    state.items["yellowKey"] = 2
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    actions = [
        {"label": "go blueKey MT8:7,10", "path": ["down"] * 8 + ["right"]},
        {"label": "go upFloor MT8:6,1", "path": ["up"]},
        {"label": "go downFloor MT8:1,1", "path": ["left"]},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "pre_mt10_buffer", sim=sim)]

    assert labels == ["go blueKey MT8:7,10"]


def test_pre_mt10_buffer_filter_preserves_two_yellow_keys_for_high_floor_resources() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT1"
    state.x = 1
    state.y = 3
    state.hp = 271
    state.atk = 22
    state.defense = 23
    state.money = 23
    state.items["yellowKey"] = 2
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    actions = [
        {"label": "open yellowDoor MT1:6,6", "path": ["right"] * 5},
        {"label": "fight skeleton MT1:2,4", "path": ["down"]},
        {"label": "go upFloor MT1:1,1", "path": ["up"] * 2},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "pre_mt10_buffer", sim=sim)]

    assert labels == ["go upFloor MT1:1,1"]


def test_pre_mt10_buffer_filter_climbs_from_mt5_only_after_high_chain_stats_ready() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT5"
    state.x = 1
    state.y = 9
    state.hp = 234
    state.atk = 21
    state.defense = 22
    state.money = 9
    state.items["yellowKey"] = 3
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    actions = [
        {"label": "go wand MT5:4,9", "path": ["right"] * 3},
        {"label": "fight bluePriest MT5:3,5", "path": ["up"] * 4},
        {"label": "go upFloor MT5:1,1", "path": ["up"] * 8},
        {"label": "go downFloor MT5:1,11", "path": ["down"] * 2},
        {"label": "open yellowDoor MT5:10,1", "path": ["right"] * 9},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "pre_mt10_buffer", sim=sim)]

    assert labels == ["open yellowDoor MT5:10,1"]

    state.atk = 22
    state.defense = 23
    labels = [action["label"] for action in filter_stage_actions(actions, state, "pre_mt10_buffer", sim=sim)]

    assert labels == ["go upFloor MT5:1,1"]


def test_pre_mt10_buffer_filter_climbs_from_mt4_after_high_chain_stats_ready() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT4"
    state.x = 11
    state.y = 10
    state.hp = 271
    state.atk = 22
    state.defense = 23
    state.money = 23
    state.items["yellowKey"] = 2
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    actions = [
        {"label": "go downFloor MT4:11,11", "path": ["down"]},
        {"label": "fight greenSlime MT4:8,11", "path": ["left"] * 3},
        {"label": "go upFloor MT4:1,11", "path": ["left"] * 10},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "pre_mt10_buffer", sim=sim)]

    assert labels == ["go upFloor MT4:1,11"]


def test_pre_mt10_buffer_filter_does_not_force_mt6_mt7_stair_loop() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT6"
    state.x = 11
    state.y = 10
    state.hp = 370
    state.atk = 21
    state.defense = 21
    state.money = 1
    state.items["yellowKey"] = 2
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    actions = [
        {"label": "go upFloor MT6:11,11", "path": ["down"]},
        {"label": "go downFloor MT6:1,1", "path": ["left"] * 10},
        {"label": "fight skeletonSoldier MT6:9,6", "path": ["left"] * 2},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "pre_mt10_buffer", sim=sim)]

    assert labels != ["go upFloor MT6:11,11"]


def test_pre_mt10_buffer_filter_keeps_non_key_exploration_when_only_stairs_match() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT8"
    state.x = 1
    state.y = 2
    state.hp = 370
    state.atk = 21
    state.defense = 21
    state.money = 1
    state.items["yellowKey"] = 2
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    actions = [
        {"label": "open yellowDoor MT8:1,3", "path": ["down"]},
        {"label": "go downFloor MT8:1,1", "path": ["up"]},
        {"label": "go upFloor MT8:6,1", "path": ["right"] * 5},
        {"label": "fight greenSlime MT8:7,2", "path": ["right"] * 6},
        {"label": "fight bluePriest MT8:7,5", "path": ["right"] * 6 + ["down"] * 3},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "pre_mt10_buffer", sim=sim)]

    assert "go downFloor MT8:1,1" in labels
    assert "fight bluePriest MT8:7,5" not in labels
    assert "fight greenSlime MT8:7,2" not in labels
    assert "open yellowDoor MT8:1,3" not in labels

    state.atk = 22
    state.defense = 23
    labels = [action["label"] for action in filter_stage_actions(actions, state, "pre_mt10_buffer", sim=sim)]

    assert "fight bluePriest MT8:7,5" in labels


def test_pre_mt10_buffer_filter_does_not_force_7f_8f_stair_loop_for_mt8_blue_key() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT7"
    state.x = 1
    state.y = 2
    state.hp = 370
    state.atk = 21
    state.defense = 21
    state.items["yellowKey"] = 2
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    actions = [
        {"label": "go upFloor MT7:1,1", "path": ["up"]},
        {"label": "go downFloor MT7:11,11", "path": ["down"]},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "pre_mt10_buffer", sim=sim)]

    assert labels != ["go upFloor MT7:1,1"]


def test_pre_mt10_buffer_filter_takes_local_potion_when_stats_done_but_hp_low() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT9"
    state.x = 2
    state.y = 4
    state.hp = 66
    state.atk = 25
    state.defense = 26
    state.money = 66
    state.items["yellowKey"] = 2
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    for floor_id, x, y in (
        ("MT5", 1, 9),
        ("MT6", 4, 9),
        ("MT1", 7, 3),
        ("MT1", 7, 4),
        ("MT3", 2, 1),
        ("MT3", 2, 9),
        ("MT7", 3, 1),
        ("MT8", 4, 10),
        ("MT8", 5, 11),
        ("MT9", 1, 5),
        ("MT9", 6, 5),
    ):
        state.floors[floor_id][y][x] = 0
    actions = [
        {"label": "go redPotion MT9:2,10", "path": ["down"] * 6},
        {"label": "go upFloor MT9:1,11", "path": ["down"] * 7},
        {"label": "go downFloor MT9:6,1", "path": ["right"] * 4},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "pre_mt10_buffer", sim=sim)]

    assert labels == ["go redPotion MT9:2,10"]


def test_mt10_resources_filter_returns_to_refill_when_only_stairs_remain() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT10"
    state.hp = 75
    state.items["yellowKey"] = 0
    actions = [
        {"label": "go downFloor MT10:1,11", "path": ["down"]},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "mt10_resources")]

    assert labels == ["go downFloor MT10:1,11"]


def test_mt10_resources_filter_fights_left_blocker_after_left_door() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT10"
    state.x = 1
    state.y = 9
    state.hp = 67
    state.atk = 24
    state.defense = 24
    state.money = 96
    state.items["yellowKey"] = 0
    sim.set_tile(state, 1, 9, 0, "MT10")

    labels = [
        action["label"]
        for action in filter_stage_actions(sim.macro_actions(state), state, "mt10_resources", sim=sim)
    ]

    assert "go downFloor MT10:1,11" not in labels
    assert "fight skeleton MT10:1,6" in labels


def test_mt10_resources_filter_rejects_underprepared_first_mt10_entry() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT10"
    state.x = 1
    state.y = 10
    state.hp = 67
    state.atk = 24
    state.defense = 24
    state.money = 96
    state.items["yellowKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    sim.set_tile(state, 3, 11, 0, "MT9")

    labels = [
        action["label"]
        for action in filter_stage_actions(sim.macro_actions(state), state, "mt10_resources", sim=sim)
    ]

    assert labels == ["go downFloor MT10:1,11"]


def test_mt10_resources_filter_accepts_prepared_first_mt10_entry() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT10"
    state.x = 1
    state.y = 10
    state.hp = 370
    state.atk = 26
    state.defense = 26
    state.money = 66
    state.items["yellowKey"] = 2
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    sim.set_tile(state, 3, 11, 0, "MT9")

    labels = [
        action["label"]
        for action in filter_stage_actions(sim.macro_actions(state), state, "mt10_resources", sim=sim)
    ]

    assert labels == ["open yellowDoor MT10:1,9"]


def test_mt10_resources_filter_accepts_damage_safe_first_mt10_entry() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT10"
    state.x = 1
    state.y = 10
    state.hp = 291
    state.atk = 26
    state.defense = 26
    state.money = 109
    state.items["yellowKey"] = 2
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    sim.set_tile(state, 3, 11, 0, "MT9")

    labels = [
        action["label"]
        for action in filter_stage_actions(sim.macro_actions(state), state, "mt10_resources", sim=sim)
    ]

    assert labels == ["open yellowDoor MT10:1,9"]


def test_mt10_resources_filter_returns_to_9f_after_left_blue_gem_when_refill_pending() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT10"
    state.x = 1
    state.y = 10
    state.hp = 383
    state.atk = 26
    state.defense = 27
    state.money = 99
    state.items["yellowKey"] = 2
    state.items["blueKey"] = 0
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    sim.set_tile(state, 3, 11, 0, "MT9")
    sim.set_tile(state, 1, 9, 0, "MT10")
    sim.set_tile(state, 1, 6, 0, "MT10")
    sim.set_tile(state, 2, 6, 0, "MT10")

    labels = [
        action["label"]
        for action in filter_stage_actions(sim.macro_actions(state), state, "mt10_resources", sim=sim)
    ]

    assert labels == ["go downFloor MT10:1,11"]


def test_mt10_resources_filter_continues_after_left_blue_gem_after_9f_refill() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT10"
    state.x = 1
    state.y = 10
    state.hp = 383
    state.atk = 26
    state.defense = 27
    state.money = 99
    state.items["yellowKey"] = 2
    state.items["blueKey"] = 0
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    sim.set_tile(state, 3, 11, 0, "MT9")
    sim.set_tile(state, 8, 11, 0, "MT9")
    sim.set_tile(state, 9, 11, 0, "MT9")
    sim.set_tile(state, 11, 9, 0, "MT9")
    sim.set_tile(state, 1, 9, 0, "MT10")
    sim.set_tile(state, 1, 6, 0, "MT10")
    sim.set_tile(state, 2, 6, 0, "MT10")

    labels = [
        action["label"]
        for action in filter_stage_actions(sim.macro_actions(state), state, "mt10_resources", sim=sim)
    ]

    assert "go downFloor MT10:1,11" not in labels
    assert "open yellowDoor MT10:3,9" in labels


def test_mt10_resources_filter_returns_after_middle_left_priest_when_yellow_short() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT10"
    state.x = 4
    state.y = 11
    state.hp = 340
    state.atk = 26
    state.defense = 27
    state.money = 99
    state.items["yellowKey"] = 1
    state.items["blueKey"] = 0
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    for floor_id, coords in {
        "MT9": ((3, 11), (8, 11), (9, 11), (11, 9)),
        "MT10": ((1, 9), (1, 6), (2, 6), (3, 9), (4, 11)),
    }.items():
        for x, y in coords:
            sim.set_tile(state, x, y, 0, floor_id)
    actions = [
        {"label": "fight bluePriest MT10:8,11", "path": ["right"]},
        {"label": "open yellowDoor MT10:9,9", "path": ["right", "up"]},
        {"label": "go downFloor MT10:1,11", "path": ["left"]},
    ]

    labels = [
        action["label"] for action in filter_stage_actions(actions, state, "mt10_resources", sim=sim)
    ]

    assert labels == ["go downFloor MT10:1,11"]


def test_mt10_resources_filter_allows_right_side_when_yellow_buffer_sufficient() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT10"
    state.x = 4
    state.y = 11
    state.hp = 520
    state.atk = 26
    state.defense = 27
    state.money = 99
    state.items["yellowKey"] = 2
    state.items["blueKey"] = 0
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    for floor_id, coords in {
        "MT9": ((3, 11), (8, 11), (9, 11), (11, 9)),
        "MT10": ((1, 9), (1, 6), (2, 6), (3, 9), (4, 11)),
    }.items():
        for x, y in coords:
            sim.set_tile(state, x, y, 0, floor_id)
    actions = [
        {"label": "fight bluePriest MT10:8,11", "path": ["right"]},
        {"label": "open yellowDoor MT10:9,9", "path": ["right", "up"]},
        {"label": "go downFloor MT10:1,11", "path": ["left"]},
    ]

    labels = [
        action["label"] for action in filter_stage_actions(actions, state, "mt10_resources", sim=sim)
    ]

    assert "go downFloor MT10:1,11" not in labels
    assert labels == ["fight bluePriest MT10:8,11", "open yellowDoor MT10:9,9"]


def test_mt10_resources_filter_returns_for_yellow_keys_after_right_red_gem() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT10"
    state.x = 10
    state.y = 6
    state.hp = 297
    state.atk = 27
    state.defense = 27
    state.money = 109
    state.items["yellowKey"] = 0
    state.items["blueKey"] = 0
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    sim.set_tile(state, 1, 9, 0, "MT10")
    sim.set_tile(state, 1, 6, 0, "MT10")
    sim.set_tile(state, 2, 6, 0, "MT10")
    sim.set_tile(state, 3, 9, 0, "MT10")
    sim.set_tile(state, 3, 6, 0, "MT10")
    sim.set_tile(state, 2, 7, 0, "MT10")
    sim.set_tile(state, 10, 6, 0, "MT10")

    actions = [
        {"label": "fight skeletonSoldier MT10:10,7", "path": ["down"]},
        {"label": "fight skeleton MT10:11,6", "path": ["right"]},
        {"label": "go downFloor MT10:1,11", "path": ["left", "down"]},
    ]

    labels = [
        action["label"] for action in filter_stage_actions(actions, state, "mt10_resources", sim=sim)
    ]

    assert labels == ["go downFloor MT10:1,11"]


def test_mt10_resources_filter_returns_from_mt9_for_yellow_keys_after_right_red_gem() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT9"
    state.x = 2
    state.y = 10
    state.hp = 347
    state.atk = 27
    state.defense = 27
    state.money = 109
    state.items["yellowKey"] = 0
    state.items["blueKey"] = 0
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    sim.set_tile(state, 2, 6, 0, "MT10")
    sim.set_tile(state, 10, 6, 0, "MT10")

    actions = [
        {"label": "fight skeleton MT9:5,10", "path": ["right"]},
        {"label": "go downFloor MT9:6,1", "path": ["up"]},
        {"label": "go upFloor MT9:1,11", "path": ["left"]},
    ]

    labels = [
        action["label"] for action in filter_stage_actions(actions, state, "mt10_resources", sim=sim)
    ]

    assert labels == ["go downFloor MT9:6,1"]


def test_mt10_resources_filter_buys_yellow_keys_after_right_red_gem_when_merchant_reachable() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT7"
    state.hp = 315
    state.atk = 27
    state.defense = 27
    state.money = 115
    state.items["yellowKey"] = 0
    state.items["blueKey"] = 0
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    sim.set_tile(state, 2, 6, 0, "MT10")
    sim.set_tile(state, 10, 6, 0, "MT10")

    actions = [
        {"label": "go upFloor MT7:1,1", "path": ["up"]},
        {"label": "buy 5 yellowKey MT7:3,6", "path": ["right"]},
        {"label": "go downFloor MT7:11,11", "path": ["down"]},
    ]

    labels = [
        action["label"] for action in filter_stage_actions(actions, state, "mt10_resources", sim=sim)
    ]

    assert labels == ["buy 5 yellowKey MT7:3,6"]


def test_mt10_resources_filter_prioritizes_reachable_gems_before_ready() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT6"
    state.atk = 20
    state.defense = 21
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    actions = [
        {"label": "go upFloor MT6:11,11", "path": ["right"]},
        {"label": "go blueGem MT6:4,9", "path": ["left"]},
        {"label": "fight redSlime MT6:9,9", "path": ["down"]},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "mt10_resources", sim=sim)]

    assert labels == ["go blueGem MT6:4,9"]


def test_mt10_resources_filter_starts_mt9_right_chain_after_shield() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT9"
    state.hp = 80
    state.atk = 20
    state.defense = 20
    state.items["yellowKey"] = 3
    state.items["blueKey"] = 0
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    actions = [
        {"label": "open yellowDoor MT9:9,4", "path": ["up"]},
        {"label": "go downFloor MT9:6,1", "path": ["left"]},
        {"label": "open yellowDoor MT9:4,1", "path": ["right"]},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "mt10_resources", sim=sim)]

    assert labels == ["open yellowDoor MT9:9,4"]


def test_mt10_resources_filter_continues_mt9_right_chain_after_first_door() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT9"
    state.hp = 80
    state.atk = 20
    state.defense = 20
    state.items["yellowKey"] = 2
    state.items["blueKey"] = 0
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    sim.set_tile(state, 9, 4, 0, "MT9")
    actions = [
        {"label": "fight greenSlime MT9:10,2", "path": ["up"]},
        {"label": "go downFloor MT9:6,1", "path": ["left"]},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "mt10_resources", sim=sim)]

    assert labels == ["fight greenSlime MT9:10,2"]


def test_mt10_resources_filter_takes_left_blue_gem_after_blocker() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT10"
    state.x = 1
    state.y = 9
    state.hp = 67
    state.atk = 24
    state.defense = 24
    state.money = 96
    state.items["yellowKey"] = 0
    sim.set_tile(state, 1, 9, 0, "MT10")
    blocker = [
        action for action in sim.macro_actions(state) if action["label"] == "fight skeleton MT10:1,6"
    ][0]
    sim.apply_macro_action(state, blocker)

    labels = [
        action["label"]
        for action in filter_stage_actions(sim.macro_actions(state), state, "mt10_resources", sim=sim)
    ]

    assert labels == ["go blueGem MT10:2,6"]


def test_mt10_resources_filter_prefers_lower_refill_resources() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT7"
    state.hp = 120
    state.items["yellowKey"] = 0
    actions = [
        {"label": "go upFloor MT7:1,1", "path": ["up"]},
        {"label": "go downFloor MT7:11,11", "path": ["down"]},
        {"label": "fight bat MT7:5,9", "path": ["left"]},
        {"label": "go bluePotion MT7:7,11", "path": ["right"]},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "mt10_resources")]

    assert labels == ["fight bat MT7:5,9", "go bluePotion MT7:7,11"]


def test_mt10_resources_filter_opens_mt7_refill_entry_after_first_10f_gem() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT7"
    state.x = 1
    state.y = 2
    state.hp = 360
    state.atk = 26
    state.defense = 27
    state.money = 104
    state.items["yellowKey"] = 1
    state.items["blueKey"] = 0
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    sim.set_tile(state, 2, 6, 0, "MT10")
    sim.set_tile(state, 3, 9, 0, "MT10")
    sim.set_tile(state, 4, 11, 0, "MT10")
    actions = [
        {"label": "go upFloor MT7:1,1", "path": ["up"]},
        {"label": "open yellowDoor MT7:7,7", "path": ["right"]},
        {"label": "go downFloor MT7:11,11", "path": ["down"]},
    ]

    labels = [
        action["label"] for action in filter_stage_actions(actions, state, "mt10_resources", sim=sim)
    ]

    assert labels == ["open yellowDoor MT7:7,7"]


def test_mt10_resources_filter_prioritizes_mt7_key_pocket_before_potion() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT7"
    state.x = 7
    state.y = 10
    state.hp = 515
    state.atk = 26
    state.defense = 27
    state.money = 117
    state.items["yellowKey"] = 0
    state.items["blueKey"] = 0
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    for floor_id, coords in {
        "MT7": ((7, 7), (7, 9), (7, 10)),
        "MT10": ((1, 9), (1, 6), (2, 6), (3, 9), (4, 11)),
    }.items():
        for x, y in coords:
            sim.set_tile(state, x, y, 0, floor_id)
    actions = [
        {"label": "fight skeletonSoldier MT7:9,7", "path": ["right"]},
        {"label": "go bluePotion MT7:7,11", "path": ["down"]},
    ]

    labels = [
        action["label"] for action in filter_stage_actions(actions, state, "mt10_resources", sim=sim)
    ]

    assert labels == ["fight skeletonSoldier MT7:9,7"]


def test_mt10_resources_filter_takes_mt7_key_before_potion_after_soldier() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT7"
    state.x = 9
    state.y = 7
    state.hp = 500
    state.atk = 26
    state.defense = 27
    state.money = 122
    state.items["yellowKey"] = 0
    state.items["blueKey"] = 0
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    for floor_id, coords in {
        "MT7": ((7, 7), (7, 9), (7, 10), (9, 7)),
        "MT10": ((1, 9), (1, 6), (2, 6), (3, 9), (4, 11)),
    }.items():
        for x, y in coords:
            sim.set_tile(state, x, y, 0, floor_id)
    actions = [
        {"label": "go yellowKey MT7:9,11", "path": ["down"]},
        {"label": "go bluePotion MT7:7,11", "path": ["left", "down"]},
    ]

    labels = [
        action["label"] for action in filter_stage_actions(actions, state, "mt10_resources", sim=sim)
    ]

    assert labels == ["go yellowKey MT7:9,11"]


def test_mt10_resources_filter_climbs_after_6f_blue_key_with_hp_buffer() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT6"
    state.x = 8
    state.y = 3
    state.hp = 647
    state.atk = 26
    state.defense = 26
    state.money = 71
    state.items["yellowKey"] = 1
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    actions = [
        {"label": "go downFloor MT6:1,1", "path": ["left"]},
        {"label": "go upFloor MT6:11,11", "path": ["right"]},
    ]

    labels = [
        action["label"] for action in filter_stage_actions(actions, state, "mt10_resources", sim=sim)
    ]

    assert labels == ["go upFloor MT6:11,11"]


def test_mt10_resources_filter_continues_on_mt9_after_blue_refill() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT9"
    state.hp = 128
    state.items["yellowKey"] = 3
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    actions = [
        {"label": "go downFloor MT9:6,1", "path": ["down"]},
        {"label": "open yellowDoor MT9:9,4", "path": ["right"]},
        {"label": "fight greenSlime MT9:10,2", "path": ["right"]},
        {"label": "open blueDoor MT9:6,3", "path": ["down"]},
    ]

    labels = [
        action["label"] for action in filter_stage_actions(actions, state, "mt10_resources", sim=sim)
    ]

    assert "go downFloor MT9:6,1" not in labels
    assert labels == ["open yellowDoor MT9:9,4"]


def test_mt10_resources_filter_does_not_spend_mt9_11_8_before_first_10f_resource() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT9"
    state.x = 6
    state.y = 1
    state.hp = 357
    state.atk = 26
    state.defense = 26
    state.items["yellowKey"] = 3
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    sim.set_tile(state, 1, 5, 0, "MT9")
    actions = [
        {"label": "open yellowDoor MT9:11,8", "path": ["right"]},
        {"label": "open blueDoor MT9:3,11", "path": ["left"]},
        {"label": "go upFloor MT9:1,11", "path": ["left", "down"]},
    ]

    labels = [
        action["label"] for action in filter_stage_actions(actions, state, "mt10_resources", sim=sim)
    ]

    assert "open yellowDoor MT9:11,8" not in labels
    assert labels == ["open blueDoor MT9:3,11"]


def test_mt10_resources_filter_refills_before_first_mt10_stair_when_keyless() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT9"
    state.x = 1
    state.y = 10
    state.hp = 52
    state.atk = 25
    state.defense = 26
    state.items["yellowKey"] = 0
    state.items["blueKey"] = 0
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    state.floors["MT9"][11][3] = 0
    actions = [
        {"label": "go upFloor MT9:1,11", "path": ["down"]},
        {"label": "go redPotion MT9:2,10", "path": ["right"]},
        {"label": "fight skeleton MT9:5,10", "path": ["right", "right", "right"]},
        {"label": "go yellowKey MT9:5,4", "path": ["up"] * 8},
    ]

    labels = [
        action["label"] for action in filter_stage_actions(actions, state, "mt10_resources", sim=sim)
    ]

    assert labels == ["go redPotion MT9:2,10"]


def test_mt10_resources_filter_returns_to_10f_after_9f_refill_complete() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT9"
    state.x = 11
    state.y = 11
    state.hp = 598
    state.atk = 26
    state.defense = 27
    state.money = 100
    state.items["yellowKey"] = 1
    state.items["blueKey"] = 0
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    for floor_id, coords in {
        "MT9": ((8, 11), (9, 11), (11, 9)),
        "MT10": ((1, 9), (1, 6), (2, 6)),
    }.items():
        for x, y in coords:
            sim.set_tile(state, x, y, 0, floor_id)
    actions = [
        {"label": "fight skeleton MT9:5,10", "path": ["left"]},
        {"label": "go upFloor MT9:1,11", "path": ["left"]},
    ]

    labels = [
        action["label"] for action in filter_stage_actions(actions, state, "mt10_resources", sim=sim)
    ]

    assert labels == ["go upFloor MT9:1,11"]


def test_mt10_resources_filter_prefers_mt9_gems_before_skeleton_soldier() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT9"
    state.x = 9
    state.y = 4
    state.hp = 178
    state.atk = 24
    state.defense = 25
    state.items["yellowKey"] = 2
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    actions = [
        {"label": "fight skeletonSoldier MT9:11,6", "path": ["right", "down"]},
        {"label": "fight redSlime MT9:7,6", "path": ["left", "down"]},
        {"label": "go yellowKey MT9:5,4", "path": ["left"] * 4},
    ]

    labels = [
        action["label"] for action in filter_stage_actions(actions, state, "mt10_resources", sim=sim)
    ]

    assert "fight skeletonSoldier MT9:11,6" not in labels
    assert labels == ["fight redSlime MT9:7,6", "go yellowKey MT9:5,4"]


def test_mt10_resources_filter_returns_to_mt8_when_underprepared_after_mt9_left_gem() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT9"
    state.x = 2
    state.y = 4
    state.hp = 137
    state.atk = 21
    state.defense = 22
    state.money = 6
    state.items["yellowKey"] = 2
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    for x, y in (
        (9, 4),
        (10, 2),
        (8, 4),
        (6, 5),
        (5, 4),
        (4, 5),
        (3, 5),
        (1, 5),
        (2, 4),
    ):
        state.floors["MT9"][y][x] = 0
    actions = [
        {"label": "open blueDoor MT9:6,3", "path": ["right"] * 7},
        {"label": "fight bat MT9:7,10", "path": ["down"] * 11},
        {"label": "go downFloor MT9:6,1", "path": ["right"] * 4},
        {"label": "open yellowDoor MT9:4,1", "path": ["right"] * 2},
    ]

    labels = [
        action["label"] for action in filter_stage_actions(actions, state, "mt10_resources", sim=sim)
    ]

    assert labels == ["go downFloor MT9:6,1"]


def test_low_gems_filter_avoids_mt3_right_stair_loop() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT3"
    state.x = 10
    state.y = 11
    state.hp = 122
    state.atk = 21
    state.defense = 22
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    actions = [
        {"label": "go upFloor MT3:11,11", "path": ["right"]},
        {"label": "go downFloor MT3:1,11", "path": ["left"]},
        {"label": "fight bat MT3:3,5", "path": ["left", "up"]},
        {"label": "open yellowDoor MT3:9,8", "path": ["up"]},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "low_gems", sim=sim)]

    assert "go upFloor MT3:11,11" not in labels
    assert labels == ["fight bat MT3:3,5"]


def test_low_gems_filter_moves_up_only_after_mt1_done() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT2"
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    state.floors["MT1"][3][7] = 0
    state.floors["MT1"][4][7] = 0
    actions = [
        {"label": "go downFloor MT2:1,11", "path": ["down"]},
        {"label": "go upFloor MT2:11,1", "path": ["up"]},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "low_gems", sim=sim)]

    assert labels == ["go upFloor MT2:11,1"]


def test_low_gems_filter_prefers_mt1_gem_route_over_stair_loop() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT1"
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    actions = [
        {"label": "go upFloor MT1:1,1", "path": ["up"]},
        {"label": "open yellowDoor MT1:6,6", "path": ["right"]},
        {"label": "open yellowDoor MT1:10,9", "path": ["right"]},
        {"label": "open yellowDoor MT1:6,9", "path": ["right"]},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "low_gems", sim=sim)]

    assert "go upFloor MT1:1,1" not in labels
    assert labels == ["open yellowDoor MT1:6,6", "open yellowDoor MT1:6,9"]


def test_low_gems_filter_seeks_key_buffer_before_mt1_gems() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT2"
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    state.items["yellowKey"] = 1
    state.floors["MT3"][1][2] = 0
    state.floors["MT3"][9][2] = 0
    actions = [
        {"label": "go downFloor MT2:1,1", "path": ["down"]},
        {"label": "go upFloor MT2:1,11", "path": ["up"]},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "low_gems", sim=sim)]

    assert labels == ["go upFloor MT2:1,11"]


def test_low_gems_filter_uses_mt7_key_buffer() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT7"
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    state.items["yellowKey"] = 1
    state.floors["MT3"][1][2] = 0
    state.floors["MT3"][9][2] = 0
    actions = [
        {"label": "go downFloor MT7:11,11", "path": ["down"]},
        {"label": "fight bat MT7:5,9", "path": ["left"]},
        {"label": "go upFloor MT7:1,1", "path": ["up"]},
    ]

    labels = [action["label"] for action in filter_stage_actions(actions, state, "low_gems", sim=sim)]

    assert labels == ["fight bat MT7:5,9"]


def test_pre_mt10_buffer_filter_returns_to_low_refill_when_hp_is_low() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT7"
    state.x = 3
    state.y = 1
    state.hp = 79
    state.atk = 26
    state.defense = 26
    state.items["yellowKey"] = 1
    state.items["blueKey"] = 0
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    actions = [
        {"label": "fight skeletonSoldier MT7:2,6", "path": ["down"] * 6},
        {"label": "open yellowDoor MT7:5,7", "path": ["right"] * 8},
        {"label": "go downFloor MT7:11,11", "path": ["right"] * 18},
        {"label": "go upFloor MT7:1,1", "path": ["left"] * 22},
    ]

    labels = [
        action["label"]
        for action in filter_stage_actions(actions, state, "pre_mt10_buffer", sim=sim)
    ]

    assert labels == ["go downFloor MT7:11,11"]


def test_pre_mt10_buffer_filter_preserves_last_yellow_key_on_4f_low_hp() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT4"
    state.x = 1
    state.y = 10
    state.hp = 129
    state.atk = 26
    state.defense = 26
    state.items["yellowKey"] = 1
    state.items["blueKey"] = 0
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    actions = [
        {"label": "open yellowDoor MT4:4,5", "path": ["right"] * 8},
        {"label": "go downFloor MT4:11,11", "path": ["right"] * 12},
        {"label": "go upFloor MT4:1,11", "path": ["down"]},
    ]

    labels = [
        action["label"]
        for action in filter_stage_actions(actions, state, "pre_mt10_buffer", sim=sim)
    ]

    assert labels == ["go downFloor MT4:11,11"]


def test_pre_mt10_buffer_filter_preserves_last_yellow_key_on_3f_low_hp() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT3"
    state.x = 10
    state.y = 11
    state.hp = 129
    state.atk = 26
    state.defense = 26
    state.items["yellowKey"] = 1
    state.items["blueKey"] = 0
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    actions = [
        {"label": "go upFloor MT3:11,11", "path": ["right"]},
        {"label": "open yellowDoor MT3:9,8", "path": ["up"] * 3},
        {"label": "open yellowDoor MT3:9,2", "path": ["up"] * 9},
        {"label": "go downFloor MT3:1,11", "path": ["left"] * 9},
    ]

    labels = [
        action["label"]
        for action in filter_stage_actions(actions, state, "pre_mt10_buffer", sim=sim)
    ]

    assert labels == ["go downFloor MT3:1,11"]


def test_pre_mt10_buffer_filter_descends_from_2f_to_low_refill_when_hp_low() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT2"
    state.x = 1
    state.y = 10
    state.hp = 129
    state.atk = 26
    state.defense = 26
    state.items["yellowKey"] = 1
    state.items["blueKey"] = 0
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    actions = [
        {"label": "go upFloor MT2:1,11", "path": ["down"]},
        {"label": "go downFloor MT2:1,1", "path": ["up"] * 9},
    ]

    labels = [
        action["label"]
        for action in filter_stage_actions(actions, state, "pre_mt10_buffer", sim=sim)
    ]

    assert labels == ["go downFloor MT2:1,1"]


def test_pre_mt10_buffer_filter_descends_for_key_shortage_without_direct_key() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT6"
    state.x = 11
    state.y = 10
    state.hp = 317
    state.atk = 26
    state.defense = 26
    state.money = 99
    state.items["yellowKey"] = 0
    state.items["blueKey"] = 0
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    actions = [
        {"label": "go upFloor MT6:11,11", "path": ["down"]},
        {"label": "fight skeletonSoldier MT6:9,6", "path": ["left"] * 4},
        {"label": "go downFloor MT6:1,1", "path": ["left"] * 10},
    ]

    labels = [
        action["label"]
        for action in filter_stage_actions(actions, state, "pre_mt10_buffer", sim=sim)
    ]

    assert labels == ["go downFloor MT6:1,1"]


def test_pre_mt10_buffer_filter_uses_mt7_enemy_that_unlocks_keys() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT7"
    state.x = 11
    state.y = 10
    state.hp = 317
    state.atk = 26
    state.defense = 26
    state.money = 99
    state.items["yellowKey"] = 0
    state.items["blueKey"] = 0
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    for floor_id, x, y in (
        ("MT7", 3, 1),
        ("MT8", 4, 10),
        ("MT8", 5, 11),
        ("MT9", 1, 5),
        ("MT9", 6, 5),
    ):
        state.floors[floor_id][y][x] = 0
    actions = [
        {"label": "go downFloor MT7:11,11", "path": ["down"]},
        {"label": "fight skeletonSoldier MT7:9,7", "path": ["left"] * 5},
        {"label": "fight skeletonSoldier MT7:2,6", "path": ["left"] * 12},
        {"label": "go upFloor MT7:1,1", "path": ["left"] * 20},
    ]

    labels = [
        action["label"]
        for action in filter_stage_actions(actions, state, "pre_mt10_buffer", sim=sim)
    ]

    assert labels == ["fight skeletonSoldier MT7:9,7"]


def test_mt10_resources_filter_prioritizes_9f_key_chain_before_first_entry() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT9"
    state.x = 6
    state.y = 2
    state.hp = 647
    state.atk = 26
    state.defense = 26
    state.money = 71
    state.items["yellowKey"] = 1
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    actions = [
        {"label": "open yellowDoor MT9:6,11", "path": ["down"]},
        {"label": "fight redSlime MT9:7,6", "path": ["down"] * 4},
        {"label": "go yellowKey MT9:2,4", "path": ["left"] * 4},
        {"label": "go upFloor MT9:1,11", "path": ["left"] * 5},
    ]

    labels = [
        action["label"]
        for action in filter_stage_actions(actions, state, "mt10_resources", sim=sim)
    ]

    assert labels == ["go yellowKey MT9:2,4"]


def test_mt10_resources_filter_opens_mt9_blue_door_before_spurious_yellow_door() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT9"
    state.x = 7
    state.y = 10
    state.hp = 569
    state.atk = 26
    state.defense = 26
    state.money = 81
    state.items["yellowKey"] = 3
    state.items["blueKey"] = 1
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    actions = [
        {"label": "open yellowDoor MT9:6,11", "path": ["left", "down"]},
        {"label": "open blueDoor MT9:3,11", "path": ["left"] * 4},
        {"label": "go upFloor MT9:1,11", "path": ["left"] * 6},
    ]

    labels = [
        action["label"]
        for action in filter_stage_actions(actions, state, "mt10_resources", sim=sim)
    ]

    assert labels == ["open blueDoor MT9:3,11"]


def test_mt10_resources_filter_blocks_middle_left_door_with_short_yellow_buffer() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    state.floor_id = "MT10"
    state.x = 1
    state.y = 11
    state.hp = 603
    state.atk = 26
    state.defense = 27
    state.money = 82
    state.items["yellowKey"] = 1
    state.items["blueKey"] = 0
    state.flags["nowWeapon"] = "sword1"
    state.flags["nowShield"] = "shield1"
    sim.set_tile(state, 1, 9, 0, "MT10")
    sim.set_tile(state, 2, 6, 0, "MT10")
    actions = [
        {"label": "open yellowDoor MT10:3,9", "path": ["right"] * 2},
        {"label": "go downFloor MT10:1,11", "path": ["down"]},
    ]

    labels = [
        action["label"]
        for action in filter_stage_actions(actions, state, "mt10_resources", sim=sim)
    ]

    assert labels == ["go downFloor MT10:1,11"]
