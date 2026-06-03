from __future__ import annotations

import random

from mota_env import MotaSimulator, load_game_data
from mota_solver.search import SearchNode
from scripts.run_archive_mcts_experiment import ArchiveEntry, ArchiveMCTS, strict_stage_score_state


def test_archive_choose_expands_unvisited_entry_before_score_softmax() -> None:
    archive = ArchiveMCTS(top_k=4, visit_penalty_scale=1_000.0)
    high_score_old = ArchiveEntry(
        score=20_000_000.0,
        node=SearchNode(state=None),  # type: ignore[arg-type]
        cell=("old",),
        state_key=("old",),
        visits=7,
    )
    lower_score_new = ArchiveEntry(
        score=10_000_000.0,
        node=SearchNode(state=None),  # type: ignore[arg-type]
        cell=("new",),
        state_key=("new",),
        visits=0,
    )
    archive.buckets = {
        high_score_old.cell: [high_score_old],
        lower_score_new.cell: [lower_score_new],
    }

    picked = archive.choose(random.Random(0), temperature=8.0, limit=100)

    assert picked is lower_score_new
    assert lower_score_new.visits == 1


def test_archive_choose_balances_least_visited_entries_after_initial_visit() -> None:
    archive = ArchiveMCTS(top_k=4, visit_penalty_scale=1_000.0)
    high_score_frequent = ArchiveEntry(
        score=20_000_000.0,
        node=SearchNode(state=None),  # type: ignore[arg-type]
        cell=("frequent",),
        state_key=("frequent",),
        visits=9,
    )
    lower_score_rare = ArchiveEntry(
        score=10_000_000.0,
        node=SearchNode(state=None),  # type: ignore[arg-type]
        cell=("rare",),
        state_key=("rare",),
        visits=1,
    )
    archive.buckets = {
        high_score_frequent.cell: [high_score_frequent],
        lower_score_rare.cell: [lower_score_rare],
    }

    picked = archive.choose(random.Random(0), temperature=8.0, limit=100)

    assert picked is lower_score_rare
    assert lower_score_rare.visits == 2


def test_shield_buffer_score_keeps_mt4_red_gem_as_stepping_stone() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))

    sword_state = sim.reset()
    sword_state.floor_id = "MT5"
    sword_state.x = 11
    sword_state.y = 11
    sword_state.hp = 754
    sword_state.atk = 20
    sword_state.defense = 10
    sword_state.money = 14
    sword_state.items["yellowKey"] = 1
    sword_state.items["blueKey"] = 1
    sword_state.flags["nowWeapon"] = "sword1"
    sword_state.steps = 200

    red_gem_state = sword_state.clone()
    red_gem_state.floor_id = "MT4"
    red_gem_state.x = 7
    red_gem_state.y = 10
    red_gem_state.hp = 716
    red_gem_state.atk = 21
    red_gem_state.money = 19
    red_gem_state.items["yellowKey"] = 0
    red_gem_state.items["blueKey"] = 1
    red_gem_state.steps = 230
    sim.set_tile(red_gem_state, 7, 10, 0, "MT4")
    sim.set_tile(red_gem_state, 9, 10, 0, "MT4")

    assert strict_stage_score_state(
        sim, red_gem_state, "shield_buffer"
    ) > strict_stage_score_state(sim, sword_state, "shield_buffer")


def test_pre_mt10_buffer_score_handles_money_and_rewards_buffer() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    weak = sim.reset()
    weak.flags["nowWeapon"] = "sword1"
    weak.flags["nowShield"] = "shield1"
    weak.hp = 66
    weak.atk = 25
    weak.defense = 26
    weak.money = 66
    weak.items["yellowKey"] = 2
    weak.items["blueKey"] = 1

    buffered = weak.clone()
    buffered.hp = 260
    buffered.money = 80

    assert strict_stage_score_state(sim, buffered, "pre_mt10_buffer") > strict_stage_score_state(
        sim,
        weak,
        "pre_mt10_buffer",
    )


def test_pre_mt10_buffer_score_rewards_mt8_blue_key_chain_door() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    before = sim.reset()
    before.floor_id = "MT8"
    before.x = 8
    before.y = 8
    before.hp = 265
    before.atk = 21
    before.defense = 21
    before.money = 14
    before.items["yellowKey"] = 2
    before.items["blueKey"] = 1
    before.flags["nowWeapon"] = "sword1"
    before.flags["nowShield"] = "shield1"
    for x, y in ((7, 5), (7, 7), (8, 8)):
        sim.set_tile(before, x, y, 0, "MT8")

    after = before.clone()
    after.x = 11
    after.y = 9
    after.items["yellowKey"] = 1
    sim.set_tile(after, 11, 9, 0, "MT8")

    assert strict_stage_score_state(sim, after, "pre_mt10_buffer") > strict_stage_score_state(
        sim,
        before,
        "pre_mt10_buffer",
    )


def test_mt10_resources_score_prefers_left_blue_gem_over_entrance_buffer() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    entrance = sim.reset()
    entrance.floor_id = "MT10"
    entrance.x = 1
    entrance.y = 10
    entrance.hp = 389
    entrance.atk = 26
    entrance.defense = 26
    entrance.money = 87
    entrance.items["yellowKey"] = 3
    entrance.items["blueKey"] = 0
    entrance.flags["nowWeapon"] = "sword1"
    entrance.flags["nowShield"] = "shield1"
    sim.set_tile(entrance, 3, 11, 0, "MT9")

    after_gem = entrance.clone()
    after_gem.x = 2
    after_gem.y = 6
    after_gem.hp = 357
    after_gem.defense = 27
    after_gem.money = 93
    after_gem.items["yellowKey"] = 2
    sim.set_tile(after_gem, 1, 9, 0, "MT10")
    sim.set_tile(after_gem, 1, 6, 0, "MT10")
    sim.set_tile(after_gem, 2, 6, 0, "MT10")
    sim.set_tile(after_gem, 3, 11, 0, "MT9")

    assert strict_stage_score_state(sim, after_gem, "mt10_resources") > strict_stage_score_state(
        sim,
        entrance,
        "mt10_resources",
    )


def test_mt10_resources_score_prefers_opening_left_door_over_entrance() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    entrance = sim.reset()
    entrance.floor_id = "MT10"
    entrance.x = 1
    entrance.y = 10
    entrance.hp = 635
    entrance.atk = 26
    entrance.defense = 26
    entrance.money = 76
    entrance.items["yellowKey"] = 1
    entrance.items["blueKey"] = 0
    entrance.flags["nowWeapon"] = "sword1"
    entrance.flags["nowShield"] = "shield1"
    sim.set_tile(entrance, 3, 11, 0, "MT9")

    opened = entrance.clone()
    opened.x = 1
    opened.y = 9
    opened.items["yellowKey"] = 0
    sim.set_tile(opened, 1, 9, 0, "MT10")

    cleared = opened.clone()
    cleared.x = 3
    cleared.y = 6
    cleared.hp = 595
    cleared.money = 82
    sim.set_tile(cleared, 3, 6, 0, "MT10")

    assert strict_stage_score_state(sim, opened, "mt10_resources") > strict_stage_score_state(
        sim,
        entrance,
        "mt10_resources",
    )
    assert strict_stage_score_state(sim, cleared, "mt10_resources") > strict_stage_score_state(
        sim,
        opened,
        "mt10_resources",
    )


def test_mt10_resources_score_keeps_pre10_refill_bonus_after_left_blue_gem() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    before_gem = sim.reset()
    before_gem.floor_id = "MT10"
    before_gem.x = 1
    before_gem.y = 6
    before_gem.hp = 603
    before_gem.atk = 26
    before_gem.defense = 26
    before_gem.money = 82
    before_gem.items["yellowKey"] = 0
    before_gem.items["blueKey"] = 0
    before_gem.flags["nowWeapon"] = "sword1"
    before_gem.flags["nowShield"] = "shield1"
    for floor_id, coords in {
        "MT7": ((9, 11), (5, 11), (7, 11)),
        "MT9": ((3, 11),),
        "MT10": ((1, 9), (1, 6)),
    }.items():
        for x, y in coords:
            sim.set_tile(before_gem, x, y, 0, floor_id)
    sim.set_tile(before_gem, 8, 4, 0, "MT6")

    after_gem = before_gem.clone()
    after_gem.x = 2
    after_gem.y = 6
    after_gem.defense = 27
    sim.set_tile(after_gem, 2, 6, 0, "MT10")

    assert strict_stage_score_state(sim, after_gem, "mt10_resources") > strict_stage_score_state(
        sim,
        before_gem,
        "mt10_resources",
    )


def test_mt10_resources_score_prefers_finishing_9f_refill_after_left_gem() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    before_priest = sim.reset()
    before_priest.floor_id = "MT9"
    before_priest.x = 11
    before_priest.y = 11
    before_priest.hp = 535
    before_priest.atk = 26
    before_priest.defense = 27
    before_priest.money = 103
    before_priest.items["yellowKey"] = 1
    before_priest.items["blueKey"] = 0
    before_priest.flags["nowWeapon"] = "sword1"
    before_priest.flags["nowShield"] = "shield1"
    for floor_id, coords in {
        "MT7": ((9, 11), (5, 11), (7, 11)),
        "MT9": ((3, 11), (8, 11), (9, 11), (9, 9), (11, 11)),
        "MT10": ((1, 9), (1, 6), (2, 6)),
    }.items():
        for x, y in coords:
            sim.set_tile(before_priest, x, y, 0, floor_id)
    sim.set_tile(before_priest, 8, 4, 0, "MT6")

    after_priest = before_priest.clone()
    after_priest.x = 11
    after_priest.y = 9
    after_priest.hp = 520
    after_priest.money = 108
    sim.set_tile(after_priest, 11, 9, 0, "MT9")

    upstairs = after_priest.clone()
    upstairs.floor_id = "MT10"
    upstairs.x = 1
    upstairs.y = 10

    assert strict_stage_score_state(sim, after_priest, "mt10_resources") > strict_stage_score_state(
        sim,
        before_priest,
        "mt10_resources",
    )
    assert strict_stage_score_state(sim, upstairs, "mt10_resources") > strict_stage_score_state(
        sim,
        after_priest,
        "mt10_resources",
    )


def test_mt10_resources_score_prefers_low_key_refill_after_middle_left_clear() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    mt9 = sim.reset()
    mt9.floor_id = "MT9"
    mt9.x = 1
    mt9.y = 10
    mt9.hp = 475
    mt9.atk = 26
    mt9.defense = 27
    mt9.money = 119
    mt9.items["yellowKey"] = 0
    mt9.items["blueKey"] = 0
    mt9.flags["nowWeapon"] = "sword1"
    mt9.flags["nowShield"] = "shield1"
    for floor_id, coords in {
        "MT7": ((9, 11), (5, 11), (7, 11)),
        "MT9": ((3, 11), (8, 11), (9, 11), (9, 9), (11, 11), (11, 9)),
        "MT10": ((1, 9), (1, 6), (2, 6), (3, 9), (4, 11)),
    }.items():
        for x, y in coords:
            sim.set_tile(mt9, x, y, 0, floor_id)
    sim.set_tile(mt9, 8, 4, 0, "MT6")

    mt8 = mt9.clone()
    mt8.floor_id = "MT8"
    mt8.x = 1
    mt8.y = 1

    mt4 = mt9.clone()
    mt4.floor_id = "MT4"
    mt4.x = 9
    mt4.y = 2

    mt4_key = mt4.clone()
    mt4_key.items["yellowKey"] = 1
    sim.set_tile(mt4_key, 9, 2, 0, "MT4")

    assert strict_stage_score_state(sim, mt8, "mt10_resources") > strict_stage_score_state(
        sim,
        mt9,
        "mt10_resources",
    )
    assert strict_stage_score_state(sim, mt4, "mt10_resources") > strict_stage_score_state(
        sim,
        mt8,
        "mt10_resources",
    )
    assert strict_stage_score_state(sim, mt4_key, "mt10_resources") > strict_stage_score_state(
        sim,
        mt4,
        "mt10_resources",
    )


def test_mt10_resources_score_prefers_after_left_continuation_over_stopping_at_gem() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    left_gem = sim.reset()
    left_gem.floor_id = "MT10"
    left_gem.x = 2
    left_gem.y = 6
    left_gem.hp = 325
    left_gem.atk = 26
    left_gem.defense = 27
    left_gem.money = 99
    left_gem.items["yellowKey"] = 2
    left_gem.items["blueKey"] = 0
    left_gem.flags["nowWeapon"] = "sword1"
    left_gem.flags["nowShield"] = "shield1"
    for floor_id, coords in {
        "MT9": ((3, 11), (8, 11), (9, 11), (11, 9)),
        "MT10": ((1, 9), (1, 6), (2, 6)),
    }.items():
        for x, y in coords:
            sim.set_tile(left_gem, x, y, 0, floor_id)

    after_middle = left_gem.clone()
    after_middle.x = 3
    after_middle.y = 9
    after_middle.items["yellowKey"] = 1
    sim.set_tile(after_middle, 3, 9, 0, "MT10")

    after_priest = after_middle.clone()
    after_priest.x = 4
    after_priest.y = 11
    after_priest.hp = 310
    after_priest.money = 104
    sim.set_tile(after_priest, 4, 11, 0, "MT10")

    returned = after_priest.clone()
    returned.floor_id = "MT9"
    returned.x = 1
    returned.y = 10

    left_score = strict_stage_score_state(sim, left_gem, "mt10_resources")
    assert strict_stage_score_state(sim, after_middle, "mt10_resources") > left_score
    assert strict_stage_score_state(sim, after_priest, "mt10_resources") > left_score
    assert strict_stage_score_state(sim, returned, "mt10_resources") > strict_stage_score_state(
        sim,
        after_priest,
        "mt10_resources",
    )


def test_mt10_resources_score_prefers_mt7_refill_chain_progress() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    entrance = sim.reset()
    entrance.floor_id = "MT7"
    entrance.x = 1
    entrance.y = 2
    entrance.hp = 360
    entrance.atk = 26
    entrance.defense = 27
    entrance.money = 104
    entrance.items["yellowKey"] = 1
    entrance.items["blueKey"] = 0
    entrance.flags["nowWeapon"] = "sword1"
    entrance.flags["nowShield"] = "shield1"
    for floor_id, coords in {
        "MT9": ((3, 11), (8, 11), (9, 11), (11, 9)),
        "MT10": ((1, 9), (1, 6), (2, 6), (3, 9), (4, 11)),
    }.items():
        for x, y in coords:
            sim.set_tile(entrance, x, y, 0, floor_id)

    opened = entrance.clone()
    opened.x = 7
    opened.y = 7
    opened.items["yellowKey"] = 0
    sim.set_tile(opened, 7, 7, 0, "MT7")

    key_taken = opened.clone()
    key_taken.x = 9
    key_taken.y = 11
    key_taken.hp = 330
    key_taken.money = 112
    key_taken.items["yellowKey"] = 2
    sim.set_tile(key_taken, 7, 10, 0, "MT7")
    sim.set_tile(key_taken, 9, 7, 0, "MT7")
    sim.set_tile(key_taken, 9, 11, 0, "MT7")

    assert strict_stage_score_state(sim, opened, "mt10_resources") > strict_stage_score_state(
        sim,
        entrance,
        "mt10_resources",
    )
    assert strict_stage_score_state(sim, key_taken, "mt10_resources") > strict_stage_score_state(
        sim,
        opened,
        "mt10_resources",
    )


def test_mt10_resources_score_prefers_6f_blue_key_after_pre10_mt7_refill() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    mt7_done = sim.reset()
    mt7_done.floor_id = "MT7"
    mt7_done.x = 5
    mt7_done.y = 11
    mt7_done.hp = 609
    mt7_done.atk = 26
    mt7_done.defense = 26
    mt7_done.money = 117
    mt7_done.items["yellowKey"] = 2
    mt7_done.items["blueKey"] = 0
    mt7_done.flags["nowWeapon"] = "sword1"
    mt7_done.flags["nowShield"] = "shield1"
    for floor_id, coords in {
        "MT7": ((9, 7), (9, 11), (7, 7), (7, 9), (7, 10), (7, 11), (5, 7), (5, 9), (5, 11)),
    }.items():
        for x, y in coords:
            sim.set_tile(mt7_done, x, y, 0, floor_id)

    mt6 = mt7_done.clone()
    mt6.floor_id = "MT6"
    mt6.x = 11
    mt6.y = 10

    bought_blue = mt6.clone()
    bought_blue.x = 8
    bought_blue.y = 4
    bought_blue.money = 67
    bought_blue.items["blueKey"] = 1
    sim.set_tile(bought_blue, 8, 4, 0, "MT6")

    climbed = bought_blue.clone()
    climbed.floor_id = "MT8"
    climbed.x = 6
    climbed.y = 2

    mt9_blue_open = climbed.clone()
    mt9_blue_open.floor_id = "MT9"
    mt9_blue_open.x = 3
    mt9_blue_open.y = 11
    mt9_blue_open.items["blueKey"] = 0
    sim.set_tile(mt9_blue_open, 3, 11, 0, "MT9")

    assert strict_stage_score_state(sim, mt6, "mt10_resources") > strict_stage_score_state(
        sim,
        mt7_done,
        "mt10_resources",
    )
    assert strict_stage_score_state(sim, bought_blue, "mt10_resources") > strict_stage_score_state(
        sim,
        mt6,
        "mt10_resources",
    )
    assert strict_stage_score_state(sim, climbed, "mt10_resources") > strict_stage_score_state(
        sim,
        bought_blue,
        "mt10_resources",
    )
    assert strict_stage_score_state(sim, mt9_blue_open, "mt10_resources") > strict_stage_score_state(
        sim,
        climbed,
        "mt10_resources",
    )


def test_mt10_resources_score_rewards_key_return_after_right_red_gem() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    mt8 = sim.reset()
    mt8.floor_id = "MT8"
    mt8.x = 6
    mt8.y = 2
    mt8.hp = 315
    mt8.atk = 27
    mt8.defense = 27
    mt8.money = 115
    mt8.items["yellowKey"] = 0
    mt8.items["blueKey"] = 0
    mt8.flags["nowWeapon"] = "sword1"
    mt8.flags["nowShield"] = "shield1"
    sim.set_tile(mt8, 2, 6, 0, "MT10")
    sim.set_tile(mt8, 10, 6, 0, "MT10")

    mt7 = mt8.clone()
    mt7.floor_id = "MT7"
    mt7.x = 1
    mt7.y = 2

    assert strict_stage_score_state(sim, mt7, "mt10_resources") > strict_stage_score_state(
        sim,
        mt8,
        "mt10_resources",
    )


def test_mt10_resources_score_prefers_pre_entry_yellow_buffer() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    short = sim.reset()
    short.floor_id = "MT10"
    short.x = 1
    short.y = 9
    short.hp = 647
    short.atk = 26
    short.defense = 26
    short.money = 71
    short.items["yellowKey"] = 1
    short.items["blueKey"] = 1
    short.flags["nowWeapon"] = "sword1"
    short.flags["nowShield"] = "shield1"

    buffered = short.clone()
    buffered.floor_id = "MT9"
    buffered.x = 2
    buffered.y = 2
    buffered.items["yellowKey"] = 3
    sim.set_tile(buffered, 2, 4, 0, "MT9")
    sim.set_tile(buffered, 1, 3, 0, "MT9")
    sim.set_tile(buffered, 2, 2, 0, "MT9")

    assert strict_stage_score_state(sim, buffered, "mt10_resources") > strict_stage_score_state(
        sim,
        short,
        "mt10_resources",
    )
