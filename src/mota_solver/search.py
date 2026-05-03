from __future__ import annotations

import heapq
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mota_env import MotaSimulator, MotaState
from mota_env.rewards import (
    floor_index,
    has_first_sword,
    mt8_blue_key_taken,
    mt9_shield_taken,
    progress_stage,
    simple_potential,
)


HEURISTIC_SCHEMES = ("baseline", "landmark_key_potential")


@dataclass
class SearchResult:
    solved: bool
    state: MotaState
    route: list[dict[str, Any]]
    expansions: int


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
    raise ValueError(f"Unknown heuristic scheme {scheme!r}; choose one of {HEURISTIC_SCHEMES}")


def baseline_heuristic(sim: MotaSimulator, state: MotaState) -> float:
    floor_idx = sim.floor_order.index(state.floor_id)
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
        + floor_idx * 160
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
    if floor_idx >= 10:
        landmark_bonus += 110_000

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
    progress_score = floor_idx * 1_500 + simple_potential(sim, state) * 5_000
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


def action_bias(action: dict[str, Any]) -> float:
    label = action["label"]
    bias = 0.0
    if "go upFloor" in label or "go downFloor" in label:
        bias += 40
    if "redGem" in label or "blueGem" in label or "sword" in label or "shield" in label:
        bias += 55
    if "Key" in label or "yellowKey" in label or "blueKey" in label or "redKey" in label:
        bias += 35
    if label.startswith("fight skeletonCaptain"):
        bias += 500
    if "event MT3:5,9" in label:
        bias += 2000
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
    queue: list[tuple[float, int, MotaState, list[dict[str, Any]]]] = []
    counter = 0
    heapq.heappush(queue, (-heuristic(sim, start, heuristic_scheme), counter, start, []))
    seen: set[tuple[Any, ...]] = {sim.state_key(start)}
    dominance: dict[tuple[Any, ...], list[tuple[int, ...]]] = {
        sim.dominance_key(start): [sim.resource_vector(start)]
    }
    best = start
    best_route: list[dict[str, Any]] = []
    expansions = 0

    while queue and expansions < max_expansions:
        _, _, state, route = heapq.heappop(queue)
        expansions += 1
        if heuristic(sim, state, heuristic_scheme) > heuristic(sim, best, heuristic_scheme):
            best = state
            best_route = route
        if state.flags.get("10f战胜骷髅队长"):
            return SearchResult(True, state, route, expansions)

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
                "index": len(route),
                "action": action,
                "before": before,
                "after": state_summary(child),
                "reward": transition.reward,
            }
            counter += 1
            priority = -(heuristic(sim, child, heuristic_scheme) + action_bias(action) * 0.15)
            heapq.heappush(queue, (priority, counter, child, route + [step]))

    return SearchResult(False, best, best_route, expansions)


def write_route_jsonl(route: list[dict[str, Any]], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf8") as handle:
        for row in route:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
