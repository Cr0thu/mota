from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .simulator import MotaSimulator, MotaState, Transition


REWARD_SCHEMES = (
    "raw",
    "label_dense",
    "milestone",
    "resource_delta",
    "key_pressure",
    "potential",
)


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    components: dict[str, float]


def reward_scheme_names() -> tuple[str, ...]:
    return REWARD_SCHEMES


class Rewarder:
    """Macro-action reward variants for quick RL/reward-shaping experiments."""

    def __init__(self, scheme: str = "label_dense", gamma: float = 0.99):
        if scheme not in REWARD_SCHEMES:
            raise ValueError(f"Unknown reward scheme {scheme!r}; choose one of {REWARD_SCHEMES}")
        self.scheme = scheme
        self.gamma = gamma

    def score(
        self,
        sim: MotaSimulator,
        before: MotaState,
        after: MotaState,
        action: dict[str, Any],
        transition: Transition,
    ) -> RewardBreakdown:
        if self.scheme == "raw":
            return RewardBreakdown(transition.reward, {"raw": transition.reward})
        if self.scheme == "label_dense":
            components = {"raw": transition.reward, **label_components(action, after)}
        elif self.scheme == "milestone":
            components = milestone_components(sim, before, after)
        elif self.scheme == "resource_delta":
            components = resource_delta_components(before, after)
        elif self.scheme == "key_pressure":
            components = {
                **resource_delta_components(before, after),
                **key_pressure_components(sim, before, after, action),
            }
        elif self.scheme == "potential":
            phi_before = simple_potential(sim, before)
            phi_after = simple_potential(sim, after)
            components = {
                "raw": transition.reward,
                "potential_delta": self.gamma * phi_after - phi_before,
            }
        else:  # pragma: no cover - guarded in __init__.
            components = {}

        if after.flags.get("10f战胜骷髅队长"):
            components["boss"] = components.get("boss", 0.0) + 10.0
        if after.dead:
            components["dead"] = components.get("dead", 0.0) - 10.0
        return RewardBreakdown(sum(components.values()), components)


def label_components(action: dict[str, Any], after: MotaState) -> dict[str, float]:
    label = action.get("label", "")
    components: dict[str, float] = {"step": -0.002}
    if "upFloor" in label:
        components["up_floor"] = 0.05
    if "redGem" in label or "blueGem" in label or "sword" in label or "shield" in label:
        components["stat_item"] = 0.08
    if "Key" in label or "yellowKey" in label or "blueKey" in label:
        components["key"] = 0.03
    if label.startswith("fight skeletonCaptain"):
        components["captain_action"] = 1.0
    if after.flags.get("10f战胜骷髅队长"):
        components["boss"] = 10.0
    return components


def milestone_components(sim: MotaSimulator, before: MotaState, after: MotaState) -> dict[str, float]:
    before_stage = progress_stage(sim, before)
    after_stage = progress_stage(sim, after)
    components = {"step": -0.01}
    if after_stage > before_stage:
        components["milestone"] = 2.0 * (after_stage - before_stage)
    if floor_index(sim, after) > floor_index(sim, before):
        components["new_floor"] = 0.2 * (floor_index(sim, after) - floor_index(sim, before))
    return components


def resource_delta_components(before: MotaState, after: MotaState) -> dict[str, float]:
    hp_delta = after.hp - before.hp
    atk_delta = after.atk - before.atk
    def_delta = after.defense - before.defense
    money_delta = after.money - before.money
    yk_delta = after.items.get("yellowKey", 0) - before.items.get("yellowKey", 0)
    bk_delta = after.items.get("blueKey", 0) - before.items.get("blueKey", 0)
    rk_delta = after.items.get("redKey", 0) - before.items.get("redKey", 0)
    return {
        "step": -0.004,
        "hp_delta": hp_delta * 0.0008,
        "atk_delta": atk_delta * 0.18,
        "def_delta": def_delta * 0.18,
        "money_delta": money_delta * 0.015,
        "yellow_key_delta": yk_delta * 0.35,
        "blue_key_delta": bk_delta * 0.8,
        "red_key_delta": rk_delta * 1.2,
    }


def key_pressure_components(
    sim: MotaSimulator,
    before: MotaState,
    after: MotaState,
    action: dict[str, Any],
) -> dict[str, float]:
    label = action.get("label", "")
    components: dict[str, float] = {}
    yk_before = before.items.get("yellowKey", 0)
    yk_after = after.items.get("yellowKey", 0)
    stage = progress_stage(sim, before)
    if "yellowDoor" in label and stage < 4:
        components["early_yellow_door"] = -0.65
    if yk_before > 0 and yk_after == 0 and stage < 4:
        components["yellow_key_depleted"] = -1.2
    if yk_after >= 2 and stage < 4:
        components["yellow_key_buffer"] = 0.12
    if floor_index(sim, after) >= 7 and yk_after == 0 and not mt8_blue_key_taken(sim, after):
        components["mt7_key_deadend_risk"] = -1.8
    return components


def simple_potential(sim: MotaSimulator, state: MotaState) -> float:
    stage = progress_stage(sim, state)
    yk = state.items.get("yellowKey", 0)
    bk = state.items.get("blueKey", 0)
    hp_term = min(state.hp, 3000) / 3000.0
    key_buffer = min(yk, 4) * 0.15 + min(bk, 2) * 0.4
    stat_term = state.atk * 0.04 + state.defense * 0.04
    deadend_penalty = 0.0
    if floor_index(sim, state) >= 7 and yk == 0 and not mt8_blue_key_taken(sim, state):
        deadend_penalty = 2.0
    return stage * 2.0 + floor_index(sim, state) * 0.15 + hp_term + key_buffer + stat_term - deadend_penalty


def progress_stage(sim: MotaSimulator, state: MotaState) -> int:
    if state.flags.get("10f战胜骷髅队长"):
        return 7
    if state.flags.get("10f机关"):
        return 6
    if floor_index(sim, state) >= 10:
        return 5
    if mt9_shield_taken(sim, state):
        return 4
    if mt8_blue_key_taken(sim, state):
        return 3
    if floor_index(sim, state) >= 8:
        return 2
    if has_first_sword(sim, state):
        return 1
    return 0


def floor_index(sim: MotaSimulator, state: MotaState) -> int:
    if state.floor_id not in sim.floor_order:
        return 0
    return int(state.floor_id[2:])


def tile_id(sim: MotaSimulator, state: MotaState, floor_id: str, x: int, y: int) -> str | None:
    if floor_id not in state.floors:
        return None
    return sim.block_id(sim.tile(state, x, y, floor_id))


def has_first_sword(sim: MotaSimulator, state: MotaState) -> bool:
    return state.flags.get("nowWeapon") == "sword1" or tile_id(sim, state, "MT5", 11, 11) != "sword1"


def mt8_blue_key_taken(sim: MotaSimulator, state: MotaState) -> bool:
    return tile_id(sim, state, "MT8", 7, 10) != "blueKey"


def mt9_shield_taken(sim: MotaSimulator, state: MotaState) -> bool:
    return state.flags.get("nowShield") == "shield1" or tile_id(sim, state, "MT9", 9, 7) != "shield1"
