from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .simulator import MotaSimulator, MotaState, Transition


KEY_ITEMS = ("yellowKey", "blueKey", "redKey")
STAT_ITEMS = ("redGem", "blueGem", "sword1", "shield1")
RESOURCE_ITEMS = KEY_ITEMS + STAT_ITEMS + (
    "greenGem",
    "redPotion",
    "bluePotion",
    "yellowPotion",
    "greenPotion",
)


def hero_summary(state: MotaState) -> dict[str, Any]:
    return {
        "floor": state.floor_id,
        "x": state.x,
        "y": state.y,
        "hp": state.hp,
        "atk": state.atk,
        "def": state.defense,
        "mdef": state.mdef,
        "money": state.money,
        "exp": state.exp,
        "keys": {key: state.items.get(key, 0) for key in KEY_ITEMS},
        "flags": {
            key: state.flags.get(key)
            for key in ["03", "2", "8", "10f机关", "10f战胜骷髅队长", "nowWeapon", "nowShield"]
            if key in state.flags
        },
        "steps": state.steps,
        "dead": state.dead,
        "done": state.done,
    }


def describe_state(
    sim: MotaSimulator,
    state: MotaState,
    actions: list[dict[str, Any]] | None = None,
    max_macro_actions: int = 256,
) -> dict[str, Any]:
    """Return a JSON-serializable research snapshot for search/RL traces."""

    from .rewards import (
        current_stage_name,
        floor_index,
        progress_stage,
        remaining_attack_defense_gems,
        stage_potential_components,
    )
    from .graph_state import build_graph_state

    legal_actions = actions if actions is not None else sim.macro_actions(state)
    reachable = sim.reachable_cells(state)
    return {
        "hero": hero_summary(state),
        "stage": {
            "index": progress_stage(sim, state),
            "name": current_stage_name(sim, state),
        },
        "floor_index": floor_index(sim, state),
        "reachable": {
            "cell_count": len(reachable),
            "resources": reachable_resources(sim, state, reachable),
            "openable_doors": adjacent_doors(sim, state, reachable, openable=True),
            "blocked_doors": adjacent_doors(sim, state, reachable, openable=False),
            "monsters": adjacent_monsters(sim, state, reachable),
        },
        "global_resources": global_resource_counts(sim, state),
        "global_resource_summary": global_resource_summary(sim, state),
        "graph_state": graph_state(sim, state, legal_actions, reachable, max_macro_actions),
        "full_graph_state": build_graph_state(
            sim,
            state,
            actions=legal_actions,
            max_nodes=max_macro_actions,
        ),
        "remaining_attack_defense_gems": remaining_attack_defense_gems(sim, state),
        "boss": boss_damage_snapshot(sim, state),
        "action_mask": {
            "max_macro_actions": max_macro_actions,
            "valid_count": len(legal_actions),
            "invalid_count": max(0, max_macro_actions - len(legal_actions)),
            "invalid_reason": "empty_macro_slot",
            "valid_labels": [action.get("label", "") for action in legal_actions[:max_macro_actions]],
        },
        "phi": stage_potential_components(sim, state),
    }


def reachable_resources(
    sim: MotaSimulator,
    state: MotaState,
    reachable: dict[tuple[int, int], list[str]],
) -> list[dict[str, Any]]:
    resources: list[dict[str, Any]] = []
    for (x, y), path in sorted(reachable.items()):
        tile = sim.tile(state, x, y)
        item_id = sim.block_id(tile)
        if item_id in RESOURCE_ITEMS:
            resources.append(
                {
                    "floor": state.floor_id,
                    "x": x,
                    "y": y,
                    "item": item_id,
                    "path_len": len(path),
                }
            )
    return resources


def adjacent_doors(
    sim: MotaSimulator,
    state: MotaState,
    reachable: dict[tuple[int, int], list[str]],
    openable: bool,
) -> list[dict[str, Any]]:
    doors: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for x, y in reachable:
        for nx, ny in ((x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)):
            if (nx, ny) in seen:
                continue
            tile = sim.tile(state, nx, ny)
            if not sim.is_door_tile(tile):
                continue
            can_open = sim.can_open_door(state, tile)
            if can_open != openable:
                continue
            keys = sim.block_info(tile).get("doorInfo", {}).get("keys", {})
            missing = {
                key: max(0, int(need) - state.items.get(key, 0))
                for key, need in keys.items()
                if state.items.get(key, 0) < int(need)
            }
            doors.append(
                {
                    "floor": state.floor_id,
                    "x": nx,
                    "y": ny,
                    "door": sim.block_id(tile),
                    "required_keys": keys,
                    "missing_keys": missing,
                }
            )
            seen.add((nx, ny))
    return doors


def adjacent_monsters(
    sim: MotaSimulator,
    state: MotaState,
    reachable: dict[tuple[int, int], list[str]],
) -> list[dict[str, Any]]:
    monsters: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for x, y in reachable:
        for nx, ny in ((x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)):
            if (nx, ny) in seen:
                continue
            tile = sim.tile(state, nx, ny)
            if not sim.is_enemy_tile(tile):
                continue
            enemy_id = sim.block_id(tile)
            if enemy_id is None:
                continue
            monsters.append(monster_snapshot(sim, state, nx, ny, enemy_id, reachable))
            seen.add((nx, ny))
    monsters.sort(key=lambda row: (not row["killable"], row["damage"] is None, row["damage"] or 10**9))
    return monsters


def monster_snapshot(
    sim: MotaSimulator,
    state: MotaState,
    x: int,
    y: int,
    enemy_id: str,
    reachable: dict[tuple[int, int], list[str]],
) -> dict[str, Any]:
    enemy = sim.data.enemys[enemy_id]
    info = sim.damage_info(state, enemy_id)
    damage = None if info is None else info["damage"]
    killable = damage is not None and state.hp > damage
    atk_drop = _damage_drop(sim, state, enemy_id, atk_delta=1)
    def_drop = _damage_drop(sim, state, enemy_id, def_delta=1)
    unlocks = unlocked_resources_after_battle(sim, state, x, y, reachable) if killable else []
    return {
        "floor": state.floor_id,
        "x": x,
        "y": y,
        "enemy": enemy_id,
        "enemy_hp": int(enemy.get("hp", 0)),
        "enemy_atk": int(enemy.get("atk", 0)),
        "enemy_def": int(enemy.get("def", 0)),
        "damage": damage,
        "turn": None if info is None else info["turn"],
        "killable": killable,
        "hp_margin": None if damage is None else state.hp - damage,
        "damage_drop_atk+1": atk_drop,
        "damage_drop_def+1": def_drop,
        "blocking_new_resources": unlocks,
    }


def unlocked_resources_after_battle(
    sim: MotaSimulator,
    state: MotaState,
    x: int,
    y: int,
    reachable: dict[tuple[int, int], list[str]],
) -> list[dict[str, Any]]:
    child = state.clone()
    if not sim.battle(child, x, y):
        return []
    after_reachable = sim.reachable_cells(child)
    new_resources: list[dict[str, Any]] = []
    for ix, iy in sorted(set(after_reachable) - set(reachable)):
        tile = sim.tile(child, ix, iy)
        item_id = sim.block_id(tile)
        if item_id in RESOURCE_ITEMS:
            new_resources.append({"floor": child.floor_id, "x": ix, "y": iy, "item": item_id})
    return new_resources


def boss_damage_snapshot(sim: MotaSimulator, state: MotaState) -> dict[str, Any]:
    enemy_id = "skeletonCaptain"
    enemy = sim.data.enemys.get(enemy_id, {})
    info = sim.damage_info(state, enemy_id)
    damage = None if info is None else info["damage"]
    return {
        "enemy": enemy_id,
        "enemy_hp": int(enemy.get("hp", 0)),
        "enemy_atk": int(enemy.get("atk", 0)),
        "enemy_def": int(enemy.get("def", 0)),
        "damage": damage,
        "turn": None if info is None else info["turn"],
        "killable": damage is not None and state.hp > damage,
        "hp_margin": None if damage is None else state.hp - damage,
        "break_atk_needed": max(0, int(enemy.get("def", 0)) + 1 - state.atk),
        "damage_drop_atk+1": _damage_drop(sim, state, enemy_id, atk_delta=1),
        "damage_drop_def+1": _damage_drop(sim, state, enemy_id, def_delta=1),
    }


def global_resource_counts(sim: MotaSimulator, state: MotaState) -> dict[str, int]:
    counts = {item: 0 for item in RESOURCE_ITEMS}
    for floor_id in sim.floor_order:
        if floor_id not in state.floors:
            continue
        for row in state.floors[floor_id]:
            for tile in row:
                item_id = sim.block_id(tile)
                if item_id in counts:
                    counts[item_id] += 1
    return {key: value for key, value in counts.items() if value}


def global_resource_summary(sim: MotaSimulator, state: MotaState) -> dict[str, Any]:
    remaining = global_resource_counts(sim, state)
    total = total_resource_counts(sim)
    collected = {
        item: max(0, total.get(item, 0) - remaining.get(item, 0))
        for item in sorted(set(total) | set(remaining))
    }
    by_floor: dict[str, dict[str, int]] = {}
    for floor_id in sim.floor_order:
        if floor_id not in state.floors:
            continue
        floor_counts = {item: 0 for item in RESOURCE_ITEMS}
        for row in state.floors[floor_id]:
            for tile in row:
                item_id = sim.block_id(tile)
                if item_id in floor_counts:
                    floor_counts[item_id] += 1
        floor_counts = {key: value for key, value in floor_counts.items() if value}
        if floor_counts:
            by_floor[floor_id] = floor_counts

    remaining_attack = remaining.get("redGem", 0)
    remaining_defense = remaining.get("blueGem", 0)
    return {
        "remaining": remaining,
        "total": total,
        "collected": {key: value for key, value in collected.items() if value},
        "by_floor_remaining": by_floor,
        "mt10_remaining": by_floor.get("MT10", {}),
        "remaining_attack_gain": remaining_attack,
        "remaining_defense_gain": remaining_defense,
        "total_attack_gain": total.get("redGem", 0),
        "total_defense_gain": total.get("blueGem", 0),
        "remaining_key_count": sum(remaining.get(key, 0) for key in KEY_ITEMS),
    }


def total_resource_counts(sim: MotaSimulator) -> dict[str, int]:
    cached = getattr(sim, "_total_resource_counts", None)
    if cached is not None:
        return dict(cached)
    counts = {item: 0 for item in RESOURCE_ITEMS}
    for floor_id in sim.floor_order:
        original_map = sim.data.floors[floor_id]["map"]
        for row in original_map:
            for tile in row:
                item_id = sim.block_id(tile)
                if item_id in counts:
                    counts[item_id] += 1
    counts = {key: value for key, value in counts.items() if value}
    setattr(sim, "_total_resource_counts", dict(counts))
    return counts


def graph_state(
    sim: MotaSimulator,
    state: MotaState,
    actions: list[dict[str, Any]],
    reachable: dict[tuple[int, int], list[str]],
    max_macro_actions: int,
) -> dict[str, Any]:
    """Graph-style state: hero node, reachable/interactable nodes, and macro edges."""

    nodes: list[dict[str, Any]] = [
        {
            "id": "hero",
            "kind": "hero",
            "floor": state.floor_id,
            "x": state.x,
            "y": state.y,
            "features": {
                "hp": state.hp,
                "atk": state.atk,
                "def": state.defense,
                "money": state.money,
                "yellowKey": state.items.get("yellowKey", 0),
                "blueKey": state.items.get("blueKey", 0),
                "redKey": state.items.get("redKey", 0),
            },
        }
    ]
    edges: list[dict[str, Any]] = []
    seen_nodes = {"hero"}

    def add_node(node: dict[str, Any]) -> None:
        node_id = str(node["id"])
        if node_id in seen_nodes:
            return
        seen_nodes.add(node_id)
        nodes.append(node)

    for (x, y), path in sorted(reachable.items(), key=lambda item: (len(item[1]), item[0][1], item[0][0])):
        tile = sim.tile(state, x, y)
        block_id = sim.block_id(tile) or "empty"
        if (x, y) == (state.x, state.y):
            continue
        if sim.is_item_tile(tile) or sim.is_stair_tile(tile):
            kind = "resource" if sim.is_item_tile(tile) else "stair"
            node_id = f"{state.floor_id}:{x},{y}:{block_id}"
            add_node(
                {
                    "id": node_id,
                    "kind": kind,
                    "floor": state.floor_id,
                    "x": x,
                    "y": y,
                    "block": block_id,
                    "path_len": len(path),
                    "features": item_or_stair_features(sim, state, block_id, len(path)),
                }
            )
            edges.append({"source": "hero", "target": node_id, "kind": "reachable", "cost": len(path)})

    boundary = graph_boundary_nodes(sim, state, reachable)
    for node in boundary:
        add_node(node)
        edges.append(
            {
                "source": "hero",
                "target": node["id"],
                "kind": "frontier",
                "cost": node.get("path_len", 0),
            }
        )

    for index, action in enumerate(actions[:max_macro_actions]):
        target = action.get("target")
        if not target or len(target) != 3:
            continue
        floor_id, x, y = target
        node_id = f"action:{index}:{floor_id}:{x},{y}"
        add_node(
            {
                "id": node_id,
                "kind": "macro_action",
                "floor": floor_id,
                "x": x,
                "y": y,
                "label": action.get("label", ""),
                "path_len": len(action.get("path", [])),
                "features": macro_action_features(sim, state, action),
            }
        )
        edges.append({"source": "hero", "target": node_id, "kind": "macro_action", "cost": len(action.get("path", []))})

    return {
        "nodes": nodes,
        "edges": edges,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "reachable_cell_count": len(reachable),
    }


def graph_boundary_nodes(
    sim: MotaSimulator,
    state: MotaState,
    reachable: dict[tuple[int, int], list[str]],
) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for x, y in reachable:
        path_len = len(reachable[(x, y)])
        for nx, ny in ((x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)):
            if (nx, ny) in seen:
                continue
            tile = sim.tile(state, nx, ny)
            block_id = sim.block_id(tile)
            if block_id is None:
                continue
            if sim.is_enemy_tile(tile):
                enemy_id = block_id
                info = sim.damage_info(state, enemy_id)
                damage = None if info is None else info["damage"]
                nodes.append(
                    {
                        "id": f"{state.floor_id}:{nx},{ny}:{enemy_id}",
                        "kind": "monster",
                        "floor": state.floor_id,
                        "x": nx,
                        "y": ny,
                        "block": enemy_id,
                        "path_len": path_len + 1,
                        "features": {
                            "damage": damage,
                            "strict_killable": damage is not None and state.hp > damage,
                            "relaxed_killable": sim.can_battle(state, tile),
                            "hp_margin": None if damage is None else state.hp - damage,
                            "damage_drop_atk+1": _damage_drop(sim, state, enemy_id, atk_delta=1),
                            "damage_drop_def+1": _damage_drop(sim, state, enemy_id, def_delta=1),
                        },
                    }
                )
                seen.add((nx, ny))
            elif sim.is_door_tile(tile):
                keys = sim.block_info(tile).get("doorInfo", {}).get("keys", {})
                nodes.append(
                    {
                        "id": f"{state.floor_id}:{nx},{ny}:{block_id}",
                        "kind": "door",
                        "floor": state.floor_id,
                        "x": nx,
                        "y": ny,
                        "block": block_id,
                        "path_len": path_len + 1,
                        "features": {
                            "openable": sim.can_open_door(state, tile),
                            "required_keys": keys,
                            "missing_keys": {
                                key: max(0, int(need) - state.items.get(key, 0))
                                for key, need in keys.items()
                            },
                        },
                    }
                )
                seen.add((nx, ny))
    return nodes


def item_or_stair_features(
    sim: MotaSimulator,
    state: MotaState,
    block_id: str,
    path_len: int,
) -> dict[str, Any]:
    return {
        "path_len": path_len,
        "item_value_hint": resource_value_hint(sim, state, block_id),
        "is_mt10_resource": state.floor_id == "MT10",
    }


def macro_action_features(
    sim: MotaSimulator,
    state: MotaState,
    action: dict[str, Any],
) -> dict[str, Any]:
    label = action.get("label", "")
    target = action.get("target") or []
    features: dict[str, Any] = {"path_len": len(action.get("path", []))}
    if len(target) == 3:
        _, x, y = target
        tile = sim.tile(state, int(x), int(y))
        block_id = sim.block_id(tile)
        features["block"] = block_id
        features["resource_value_hint"] = resource_value_hint(sim, state, block_id or "")
        if sim.is_enemy_tile(tile) and block_id:
            info = sim.damage_info(state, block_id)
            features["damage"] = None if info is None else info["damage"]
            features["hp_margin"] = None if info is None else state.hp - info["damage"]
            features["relaxed_killable"] = sim.can_battle(state, tile)
        if sim.is_door_tile(tile):
            features["openable"] = sim.can_open_door(state, tile)
    features["is_stair"] = "upFloor" in label or "downFloor" in label
    features["is_mt10"] = "MT10" in label
    return features


def resource_value_hint(sim: MotaSimulator, state: MotaState, item_id: str) -> float:
    if item_id == "redGem":
        return 100.0
    if item_id == "blueGem":
        return 85.0
    if item_id == "sword1":
        return 900.0
    if item_id == "shield1":
        return 900.0
    if item_id == "redKey":
        return 700.0
    if item_id == "blueKey":
        return 260.0
    if item_id == "yellowKey":
        return 120.0
    if item_id.endswith("Potion"):
        return 50.0 if state.hp < 900 else 20.0
    return 0.0


def trajectory_step_record(
    sim: MotaSimulator,
    before: MotaState,
    after: MotaState,
    action: dict[str, Any],
    transition: Transition,
    index: int,
    reward: float | None = None,
) -> dict[str, Any]:
    return {
        "index": index,
        "action": action,
        "transition": {
            "ok": transition.ok,
            "message": transition.message,
            "raw_reward": transition.reward,
            "reward": reward if reward is not None else transition.reward,
        },
        "before": hero_summary(before),
        "after": hero_summary(after),
        "before_features": describe_state(sim, before),
        "after_features": describe_state(sim, after),
    }


def write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _damage_drop(
    sim: MotaSimulator,
    state: MotaState,
    enemy_id: str,
    atk_delta: int = 0,
    def_delta: int = 0,
) -> int:
    current = sim.damage_info(state, enemy_id)
    if current is None:
        return 0
    probe = state.clone()
    probe.atk += atk_delta
    probe.defense += def_delta
    improved = sim.damage_info(probe, enemy_id)
    if improved is None:
        return 0
    return max(0, current["damage"] - improved["damage"])
