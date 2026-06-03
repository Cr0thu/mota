from __future__ import annotations

from typing import Any

from mota_env import MotaSimulator, MotaState
from mota_env.rewards import (
    remaining_attack_defense_gems,
    stage_names,
    yellow_guard_damage,
    yellow_guard_margin,
)
from mota_solver.staged import _boss_margin, _stage_action_bias


STATE_FEATURE_DIM = 31
ACTION_FEATURE_DIM = 36


def state_feature_vector(
    sim: MotaSimulator,
    state: MotaState,
    target_stage: str,
) -> list[float]:
    summary = sim.describe_state(state)
    resources = summary.get("global_resource_summary", {})
    remaining = resources.get("remaining", {})
    mt10 = resources.get("mt10_remaining", {})
    keys = state.items
    potions = sum(
        remaining.get(item, 0)
        for item in ("redPotion", "bluePotion", "yellowPotion", "greenPotion")
    )
    values = [
        _floor(state.floor_id) / 10.0,
        state.x / 12.0,
        state.y / 12.0,
        _clip(state.hp / 2000.0, -2.0, 3.0),
        state.atk / 80.0,
        state.defense / 80.0,
        state.mdef / 100.0,
        state.money / 300.0,
        state.exp / 500.0,
        keys.get("yellowKey", 0) / 10.0,
        keys.get("blueKey", 0) / 3.0,
        keys.get("redKey", 0) / 2.0,
        float(state.flags.get("nowWeapon") == "sword1"),
        float(state.flags.get("nowShield") == "shield1"),
        float(bool(state.flags.get("10f机关"))),
        float(bool(state.flags.get("10f战胜骷髅队长"))),
        _stage_id(target_stage) / max(1, len(stage_names()) - 1),
        remaining.get("redGem", 0) / 12.0,
        remaining.get("blueGem", 0) / 12.0,
        remaining.get("yellowKey", 0) / 60.0,
        remaining.get("blueKey", 0) / 6.0,
        remaining.get("redKey", 0) / 2.0,
        potions / 30.0,
        mt10.get("redGem", 0),
        mt10.get("blueGem", 0),
        mt10.get("bluePotion", 0),
        resources.get("remaining_attack_gain", 0) / 12.0,
        resources.get("remaining_defense_gain", 0) / 12.0,
        _clip(yellow_guard_damage(sim, state) / 2000.0, 0.0, 5.0),
        _clip(yellow_guard_margin(sim, state) / 2000.0, -5.0, 5.0),
        _clip(_boss_margin(sim, state) / 2000.0, -5.0, 5.0),
    ]
    return _fixed(values, STATE_FEATURE_DIM)


def action_feature_vector(
    sim: MotaSimulator,
    state: MotaState,
    action: dict[str, Any],
    target_stage: str,
) -> list[float]:
    label = action.get("label", "")
    kind = action.get("kind", "")
    target = action.get("target") or []
    floor_id = str(target[0]) if len(target) == 3 else state.floor_id
    x = int(target[1]) if len(target) == 3 else state.x
    y = int(target[2]) if len(target) == 3 else state.y
    tile = sim.tile(state, x, y, floor_id) if floor_id in state.floors else 0
    block_id = sim.block_id(tile) or ""
    category = _block_category(sim, tile, label)
    enemy_values = _enemy_action_values(sim, state, tile, block_id)
    door_values = _door_action_values(sim, state, tile)
    prior = _clip(_stage_action_bias(action, target_stage, state, sim) / 3000.0, -2.0, 3.0)

    values = [
        float(kind == "move"),
        float(kind == "shop"),
        float(kind == "fly"),
        float(kind not in {"move", "shop", "fly"}),
        len(action.get("path", [])) / 80.0,
        _floor(floor_id) / 10.0,
        x / 12.0,
        y / 12.0,
        float(category == "item"),
        float(category == "enemy"),
        float(category == "door"),
        float(category == "stair"),
        float(category == "event"),
        float(category == "empty"),
        _item_id(block_id) / 16.0,
        _door_id(block_id) / 8.0,
        *enemy_values,
        *door_values,
        prior,
        float("redKey" in label),
        float("yellowGuard" in label),
        float("redGem" in label),
        float("blueGem" in label),
        float("Potion" in label),
        float("upFloor" in label),
        float("downFloor" in label),
        float("MT10" in label),
    ]
    return _fixed(values, ACTION_FEATURE_DIM)


def _enemy_action_values(
    sim: MotaSimulator,
    state: MotaState,
    tile: int,
    block_id: str,
) -> list[float]:
    if not block_id or not sim.is_enemy_tile(tile):
        return [0.0] * 7
    enemy = sim.data.enemys[block_id]
    info = sim.damage_info(state, block_id)
    damage = 2500.0 if info is None else float(info["damage"])
    margin = -2500.0 if info is None else float(state.hp - info["damage"])
    return [
        _clip(damage / 2000.0, 0.0, 5.0),
        _clip(margin / 2000.0, -5.0, 5.0),
        _damage_drop(sim, state, block_id, atk_delta=1) / 300.0,
        _damage_drop(sim, state, block_id, def_delta=1) / 300.0,
        int(enemy.get("hp", 0)) / 1000.0,
        int(enemy.get("atk", 0)) / 100.0,
        int(enemy.get("def", 0)) / 100.0,
    ]


def _door_action_values(sim: MotaSimulator, state: MotaState, tile: int) -> list[float]:
    if not sim.is_door_tile(tile):
        return [0.0] * 4
    keys = sim.block_info(tile).get("doorInfo", {}).get("keys", {})
    return [
        float(sim.can_open_door(state, tile)),
        keys.get("yellowKey", 0) / 4.0,
        keys.get("blueKey", 0) / 2.0,
        keys.get("redKey", 0),
    ]


def _block_category(sim: MotaSimulator, tile: int, label: str) -> str:
    if sim.is_item_tile(tile):
        return "item"
    if sim.is_enemy_tile(tile):
        return "enemy"
    if sim.is_door_tile(tile):
        return "door"
    if sim.is_stair_tile(tile):
        return "stair"
    if "event" in label:
        return "event"
    return "empty"


def _damage_drop(
    sim: MotaSimulator,
    state: MotaState,
    enemy_id: str,
    atk_delta: int = 0,
    def_delta: int = 0,
) -> float:
    current = sim.damage_info(state, enemy_id)
    if current is None:
        return 0.0
    probe = state.clone()
    probe.atk += atk_delta
    probe.defense += def_delta
    improved = sim.damage_info(probe, enemy_id)
    if improved is None:
        return 0.0
    return float(max(0, current["damage"] - improved["damage"]))


def _stage_id(stage: str) -> int:
    try:
        return list(stage_names()).index(stage)
    except ValueError:
        return 0


def _floor(floor_id: str) -> int:
    try:
        return int(str(floor_id).replace("MT", ""))
    except ValueError:
        return 0


def _item_id(item: str) -> int:
    ids = {
        "yellowKey": 1,
        "blueKey": 2,
        "redKey": 3,
        "redGem": 4,
        "blueGem": 5,
        "greenGem": 6,
        "redPotion": 7,
        "bluePotion": 8,
        "yellowPotion": 9,
        "greenPotion": 10,
        "sword1": 11,
        "shield1": 12,
    }
    return ids.get(item, 0)


def _door_id(door: str) -> int:
    ids = {
        "yellowDoor": 1,
        "blueDoor": 2,
        "redDoor": 3,
        "specialDoor": 4,
    }
    return ids.get(door, 0)


def _fixed(values: list[float], size: int) -> list[float]:
    return (values[:size] + [0.0] * size)[:size]


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
