from __future__ import annotations

import heapq
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mota_env import MotaSimulator, MotaState
from mota_env.rewards import (
    floor_index,
    dynamic_pbrs_potential,
    has_first_sword,
    mt8_blue_key_taken,
    mt9_shield_taken,
    progress_stage,
    simple_potential,
)


HEURISTIC_SCHEMES = ("baseline", "landmark_key_potential", "dynamic_pbrs", "threshold_landmark")


@dataclass
class SearchResult:
    solved: bool
    state: MotaState
    route: list[dict[str, Any]]
    expansions: int


@dataclass
class SearchNode:
    state: MotaState
    parent: "SearchNode | None" = None
    step: dict[str, Any] | None = None
    depth: int = 0


def reconstruct_route(node: SearchNode) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cursor: SearchNode | None = node
    while cursor is not None and cursor.step is not None:
        rows.append(cursor.step)
        cursor = cursor.parent
    rows.reverse()
    for index, row in enumerate(rows):
        row["index"] = index
    return rows


def state_summary(state: MotaState) -> dict[str, Any]:
    return {
        "floor": state.floor_id,
        "x": state.x,
        "y": state.y,
        "hp": state.hp,
        "atk": state.atk,
        "def": state.defense,
        "mdef": state.mdef,
        "money": state.money,
        "keys": {
            "yellowKey": state.items.get("yellowKey", 0),
            "blueKey": state.items.get("blueKey", 0),
            "redKey": state.items.get("redKey", 0),
        },
        "flags": {
            key: state.flags.get(key)
            for key in ["03", "2", "8", "10f机关", "10f战胜骷髅队长"]
            if key in state.flags
        },
        "steps": state.steps,
    }


def heuristic_scheme_names() -> tuple[str, ...]:
    return HEURISTIC_SCHEMES


def heuristic(sim: MotaSimulator, state: MotaState, scheme: str = "baseline") -> float:
    if scheme == "baseline":
        return baseline_heuristic(sim, state)
    if scheme == "landmark_key_potential":
        return landmark_key_potential_heuristic(sim, state)
    if scheme == "dynamic_pbrs":
        return dynamic_pbrs_heuristic(sim, state)
    if scheme == "threshold_landmark":
        return threshold_landmark_heuristic(sim, state)
    raise ValueError(f"Unknown heuristic scheme {scheme!r}; choose one of {HEURISTIC_SCHEMES}")


def baseline_heuristic(sim: MotaSimulator, state: MotaState) -> float:
    stage = progress_stage(sim, state)
    key_score = (
        12 * state.items.get("yellowKey", 0)
        + 30 * state.items.get("blueKey", 0)
        + 50 * state.items.get("redKey", 0)
    )
    story = 20_000 if state.flags.get("03") else 0
    trap = 80 if state.flags.get("10f机关") else 0
    done = 100000 if state.flags.get("10f战胜骷髅队长") else 0
    # HP matters, but attack/defense thresholds dominate classic Magic Tower routing.
    return (
        done
        + stage * 2_500
        + story
        + trap
        + state.atk * 120
        + state.defense * 120
        + state.money * 1.2
        + key_score
        + min(state.hp, 5000) * 1.8
        - state.steps * 0.03
    )


def landmark_key_potential_heuristic(sim: MotaSimulator, state: MotaState) -> float:
    """State value tuned for the simplified first-10-floor route.

    This keeps the original resource intuition, but gives much stronger signal to
    the actual bottlenecks we observed: preserving yellow keys before the 8F blue
    key, reaching the 5F sword / 9F shield landmarks, and not treating high HP on
    a key-deadended MT7 branch as valuable.
    """

    floor_idx = floor_index(sim, state)
    stage = progress_stage(sim, state)
    yellow = state.items.get("yellowKey", 0)
    blue = state.items.get("blueKey", 0)
    red = state.items.get("redKey", 0)

    done = 1_000_000 if state.flags.get("10f战胜骷髅队长") else 0
    trap = 120_000 if state.flags.get("10f机关") else 0

    key_buffer = min(yellow, 4) * 800 + min(blue, 2) * 3_000 + min(red, 1) * 5_000
    if stage >= 3:
        key_buffer = min(yellow, 3) * 500 + min(blue, 2) * 3_000 + min(red, 1) * 5_000

    key_deadend_penalty = 0
    if stage < 3 and yellow == 0:
        key_deadend_penalty -= 5_000
    if floor_idx >= 7 and yellow == 0 and not mt8_blue_key_taken(sim, state):
        key_deadend_penalty -= 180_000
    elif floor_idx >= 7 and yellow == 1 and not mt8_blue_key_taken(sim, state):
        key_deadend_penalty -= 55_000

    landmark_bonus = stage * 75_000
    if has_first_sword(sim, state):
        landmark_bonus += 25_000
    if mt8_blue_key_taken(sim, state):
        landmark_bonus += 70_000
    if mt9_shield_taken(sim, state):
        landmark_bonus += 90_000
    landmark_bonus += collected_stat_item_bonus(sim, state)

    low_hp_penalty = 0.0
    if state.hp < 500 and stage >= 2:
        low_hp_penalty -= (500 - state.hp) * 80
    elif state.hp < 250:
        low_hp_penalty -= (250 - state.hp) * 40

    resource_score = (
        state.atk * 650
        + state.defense * 650
        + min(state.hp, 3_000) * 2.5
        + state.money * 0.5
        + key_buffer
    )
    progress_score = stage * 35_000 + simple_potential(sim, state) * 5_000
    step_penalty = state.steps * 0.2

    return (
        done
        + trap
        + landmark_bonus
        + resource_score
        + progress_score
        + key_deadend_penalty
        + low_hp_penalty
        - step_penalty
    )


def dynamic_pbrs_heuristic(sim: MotaSimulator, state: MotaState) -> float:
    return landmark_key_potential_heuristic(sim, state) + dynamic_pbrs_potential(sim, state) * 500


def threshold_landmark_heuristic(sim: MotaSimulator, state: MotaState) -> float:
    score = landmark_key_potential_heuristic(sim, state)
    if progress_stage(sim, state) >= 3:
        score += critical_boss_damage_score(sim, state)
    return score


def collected_stat_item_bonus(sim: MotaSimulator, state: MotaState) -> float:
    bonus = 0.0
    stage = progress_stage(sim, state)
    for floor_id, x, y, item_id in stat_item_positions(sim):
        if sim.block_id(sim.tile(state, x, y, floor_id)) == item_id:
            continue
        if item_id == "redGem":
            bonus += 8_000 if stage >= 3 else 0
        elif item_id == "blueGem":
            bonus += 9_000 if stage >= 3 else 0
        elif item_id == "sword1":
            bonus += 35_000
        elif item_id == "shield1":
            bonus += 45_000
    return bonus


def stat_item_positions(sim: MotaSimulator) -> list[tuple[str, int, int, str]]:
    cached = getattr(sim, "_stat_item_positions", None)
    if cached is not None:
        return cached
    positions: list[tuple[str, int, int, str]] = []
    for floor_id in sim.floor_order:
        original_map = sim.data.floors[floor_id]["map"]
        for y, row in enumerate(original_map):
            for x, tile in enumerate(row):
                item_id = sim.block_id(tile)
                if item_id in {"redGem", "blueGem", "sword1", "shield1"}:
                    positions.append((floor_id, x, y, item_id))
    setattr(sim, "_stat_item_positions", positions)
    return positions


def critical_boss_damage_score(sim: MotaSimulator, state: MotaState) -> float:
    score = 0.0
    targets = [
        ("skeletonCaptain", 320.0, 700),
        ("skeletonSoldier", 95.0, 260),
        ("skeleton", 45.0, 160),
        ("bluePriest", 35.0, 160),
    ]
    for enemy_id, damage_weight, cap in targets:
        info = sim.damage_info(state, enemy_id)
        if info is None:
            score -= 80_000
            continue
        damage = min(info["damage"], cap)
        score += max(0, cap - damage) * damage_weight
        if enemy_id == "skeletonCaptain":
            score += max(-650, min(state.hp - info["damage"], 900)) * 110

            atk_gain = damage_drop_for_stat(sim, state, enemy_id, atk_delta=1)
            def_gain = damage_drop_for_stat(sim, state, enemy_id, def_delta=1)
            score += atk_gain * 260
            score += def_gain * 160
    return score


def damage_drop_for_stat(
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


def action_bias(action: dict[str, Any]) -> float:
    label = action["label"]
    bias = 0.0
    if "redGem" in label or "blueGem" in label:
        bias += 140
    if "sword" in label or "shield" in label:
        bias += 55
    if "Key" in label or "yellowKey" in label or "blueKey" in label or "redKey" in label:
        bias += 35
    if label.startswith("fight skeletonCaptain"):
        bias += 500
    if label.startswith("shop atk") or label.startswith("shop def"):
        bias += 45
    if label.startswith("fly shop atk") or label.startswith("fly shop def"):
        bias += 180
    if label.startswith("fly shop hp"):
        bias += 60
    if label.startswith("go king"):
        bias -= 120
    return bias


def solve_first10(
    sim: MotaSimulator,
    max_expansions: int = 120_000,
    keep_per_parent: int = 80,
    heuristic_scheme: str = "baseline",
) -> SearchResult:
    start = sim.reset()
    root = SearchNode(start)
    queue: list[tuple[float, int, SearchNode]] = []
    counter = 0
    heapq.heappush(queue, (-heuristic(sim, start, heuristic_scheme), counter, root))
    seen: set[tuple[Any, ...]] = {sim.state_key(start)}
    dominance: dict[tuple[Any, ...], list[tuple[int, ...]]] = {
        sim.dominance_key(start): [sim.resource_vector(start)]
    }
    best = start
    best_node = root
    expansions = 0

    while queue and expansions < max_expansions:
        _, _, node = heapq.heappop(queue)
        state = node.state
        expansions += 1
        if heuristic(sim, state, heuristic_scheme) > heuristic(sim, best, heuristic_scheme):
            best = state
            best_node = node
        if state.flags.get("10f战胜骷髅队长"):
            return SearchResult(True, state, reconstruct_route(node), expansions)

        actions = sim.macro_actions(state)
        actions.sort(key=action_bias, reverse=True)
        for action in actions[:keep_per_parent]:
            child = state.clone()
            before = state_summary(child)
            transition = sim.apply_macro_action(child, action)
            if not transition.ok or child.dead:
                continue
            key = sim.state_key(child)
            if key in seen:
                continue
            dkey = sim.dominance_key(child)
            vec = sim.resource_vector(child)
            existing = dominance.get(dkey, [])
            if any(all(a >= b for a, b in zip(old, vec)) for old in existing):
                continue
            dominance[dkey] = [
                old for old in existing if not all(a >= b for a, b in zip(vec, old))
            ] + [vec]
            seen.add(key)
            step = {
                "index": node.depth,
                "action": action,
                "before": before,
                "after": state_summary(child),
                "reward": transition.reward,
            }
            child_node = SearchNode(child, parent=node, step=step, depth=node.depth + 1)
            counter += 1
            priority = -(heuristic(sim, child, heuristic_scheme) + action_bias(action) * 0.15)
            heapq.heappush(queue, (priority, counter, child_node))

    return SearchResult(False, best, reconstruct_route(best_node), expansions)


def write_route_jsonl(route: list[dict[str, Any]], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf8") as handle:
        for row in route:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
