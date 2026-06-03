from __future__ import annotations

import os
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
    "dynamic_pbrs",
    "stage_pbrs",
    "stage_stat_pbrs",
    "learnable_stage_pbrs",
)

STAGE_ORDER = (
    "sword",
    "mt4_redgem",
    "pre_shield_gems",
    "shield",
    "shield_buffer",
    "mid_gems",
    "low_gems",
    "mt8_hp_ready",
    "mt8_gems",
    "lower_gems",
    "pre_mt10_buffer",
    "mt10_blue_ready",
    "mt10_yellow_ready",
    "mt10_resources",
    "all_gems",
    "guard_ready",
    "red_key",
    "boss_ready",
    "trap",
    "boss",
    "boss_all_gems",
)
STAGE_LABELS = {
    "sword": "拿剑",
    "mt4_redgem": "拿4F右侧红宝石",
    "pre_shield_gems": "拿剑后路上宝石",
    "shield": "拿盾",
    "shield_buffer": "拿盾后保留推进资源",
    "gems": "属性阈值准备",
    "lower_gems": "拿齐1-9F攻防宝石",
    "pre_mt10_buffer": "10F前属性/血量/钥匙缓冲",
    "mt8_gems": "拿8F攻防宝石",
    "mid_gems": "拿6F防御宝石",
    "low_gems": "拿1-3F攻防宝石",
    "mt8_hp_ready": "8F前补血准备",
    "all_gems": "拿齐前十层攻防宝石",
    "mt10_blue_ready": "10F蓝钥匙准备",
    "mt10_yellow_ready": "10F黄钥匙准备",
    "mt10_ready": "10F资源准备",
    "mt10_resources": "拿10F关键资源",
    "guard_ready": "红钥匙卫兵准备",
    "red_key": "拿红钥匙",
    "boss_ready": "凑血量清机关并打队长",
    "trap": "进入10F机关",
    "boss": "击败骷髅队长",
    "boss_all_gems": "击败队长并拿完10F顶部宝石",
    "done": "完成",
}

DEFAULT_STAGE_POTENTIAL_WEIGHTS = {
    "asset": 1.0,
    "combat": 1.0,
    "threshold": 1.0,
    "lookahead": 1.0,
    "progress": 1.0,
    "deadend": 1.0,
    "stage_sword": 1.0,
    "stage_mt4_redgem": 1.0,
    "stage_pre_shield_gems": 1.0,
    "stage_shield": 1.0,
    "stage_shield_buffer": 1.0,
    "stage_gems": 1.0,
    "stage_lower_gems": 1.0,
    "stage_pre_mt10_buffer": 1.0,
    "stage_mt8_gems": 1.0,
    "stage_mid_gems": 1.0,
    "stage_low_gems": 1.0,
    "stage_mt8_hp_ready": 1.0,
    "stage_all_gems": 1.0,
    "stage_mt10_blue_ready": 1.0,
    "stage_mt10_yellow_ready": 1.0,
    "stage_mt10_ready": 1.0,
    "stage_mt10_resources": 1.0,
    "stage_guard_ready": 1.0,
    "stage_red_key": 1.0,
    "stage_boss_ready": 1.0,
    "stage_trap": 1.0,
    "stage_boss": 1.0,
    "stage_boss_all_gems": 1.0,
    "boss_margin": 1.0,
    "key_pressure": 1.0,
    "global_resource": 1.0,
}

CRITICAL_GEM_TARGET_ATK = 23
CRITICAL_GEM_TARGET_DEF = 21
PRE_MT10_TARGET_ATK = 21
PRE_MT10_TARGET_DEF = 21
GUARD_READY_HP_BUFFER = 0
RED_KEY_ROUTE_ENEMIES = (
    ("MT8", 7, 5, "bluePriest"),
    ("MT8", 7, 7, "bat"),
    ("MT8", 6, 8, "skeleton"),
    ("MT8", 8, 8, "bluePriest"),
    ("MT8", 9, 5, "yellowGuard"),
    ("MT8", 11, 5, "yellowGuard"),
)
MT10_RESOURCE_TARGETS = (
    ("MT10", 2, 6, "blueGem"),
    ("MT10", 10, 6, "redGem"),
    ("MT10", 11, 11, "bluePotion"),
)
MT10_RESOURCE_YELLOW_KEY_TARGET = 5
GEM_STAGE_TARGETS = {
    "mt8_gems": (
        ("MT8", 1, 5, "redPotion"),
        ("MT8", 4, 10, "redGem"),
        ("MT8", 5, 11, "blueGem"),
    ),
    "mid_gems": (
        ("MT6", 4, 9, "blueGem"),
    ),
    "low_gems": (
        ("MT1", 7, 3, "redGem"),
        ("MT1", 7, 4, "blueGem"),
        ("MT3", 2, 1, "blueGem"),
        ("MT3", 2, 9, "redGem"),
    ),
}


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    components: dict[str, float]


@dataclass(frozen=True)
class LearnableStageRewardConfig:
    gamma: float = 0.99
    global_weights: dict[str, float] | None = None
    stage_weights: dict[str, dict[str, float]] | None = None


def reward_scheme_names() -> tuple[str, ...]:
    return REWARD_SCHEMES


def stage_names() -> tuple[str, ...]:
    return STAGE_ORDER


class Rewarder:
    """Macro-action reward variants for quick RL/reward-shaping experiments."""

    def __init__(self, scheme: str = "label_dense", gamma: float = 0.99):
        if scheme not in REWARD_SCHEMES:
            raise ValueError(f"Unknown reward scheme {scheme!r}; choose one of {REWARD_SCHEMES}")
        self.scheme = scheme
        self.gamma = gamma
        self.learnable_stage_reward = (
            LearnableStageReward(gamma=gamma) if scheme == "learnable_stage_pbrs" else None
        )

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
        elif self.scheme == "dynamic_pbrs":
            before_phi = dynamic_potential_components(sim, before)
            after_phi = dynamic_potential_components(sim, after)
            components = {"env_step": -0.01}
            for key in sorted(set(before_phi) | set(after_phi)):
                components[f"delta_{key}"] = (
                    self.gamma * after_phi.get(key, 0.0) - before_phi.get(key, 0.0)
                )
        elif self.scheme == "stage_pbrs":
            before_phi = stage_potential_components(sim, before)
            after_phi = stage_potential_components(sim, after)
            components = {"env_step": -0.01}
            for key in sorted(set(before_phi) | set(after_phi)):
                components[f"delta_{key}"] = (
                    self.gamma * after_phi.get(key, 0.0) - before_phi.get(key, 0.0)
                )
        elif self.scheme == "stage_stat_pbrs":
            before_phi = boosted_stat_stage_potential_components(sim, before)
            after_phi = boosted_stat_stage_potential_components(sim, after)
            components = {"env_step": -0.01}
            for key in sorted(set(before_phi) | set(after_phi)):
                components[f"delta_{key}"] = (
                    self.gamma * after_phi.get(key, 0.0) - before_phi.get(key, 0.0)
                )
        elif self.scheme == "learnable_stage_pbrs":
            assert self.learnable_stage_reward is not None
            return self.learnable_stage_reward.score(sim, before, after, action, transition)
        else:  # pragma: no cover - guarded in __init__.
            components = {}

        if after.flags.get("10f战胜骷髅队长"):
            boss_reward = 100.0 if self.scheme in {"dynamic_pbrs", "stage_pbrs", "stage_stat_pbrs", "learnable_stage_pbrs"} else 10.0
            components["boss"] = components.get("boss", 0.0) + boss_reward
        if after.dead:
            death_penalty = 100.0 if self.scheme in {"dynamic_pbrs", "stage_pbrs", "stage_stat_pbrs", "learnable_stage_pbrs"} else 10.0
            components["dead"] = components.get("dead", 0.0) - death_penalty
        return RewardBreakdown(sum(components.values()), components)


DEFAULT_LEARNABLE_GLOBAL_WEIGHTS = {
    "asset": 1.0,
    "combat": 1.0,
    "threshold": 1.0,
    "lookahead": 1.0,
    "progress": 1.0,
    "deadend": 1.0,
    "stage_sword": 1.0,
    "stage_mt4_redgem": 1.0,
    "stage_pre_shield_gems": 1.0,
    "stage_shield": 1.0,
    "stage_shield_buffer": 1.0,
    "stage_gems": 1.0,
    "stage_lower_gems": 1.0,
    "stage_pre_mt10_buffer": 1.0,
    "stage_mt8_gems": 1.0,
    "stage_mid_gems": 1.0,
    "stage_low_gems": 1.0,
    "stage_mt8_hp_ready": 1.0,
    "stage_all_gems": 1.0,
    "stage_red_key": 1.0,
    "stage_boss_ready": 1.0,
    "stage_trap": 1.0,
    "stage_boss": 1.0,
    "stat_asset_boost": 1.0,
    "stat_threshold_boost": 1.0,
    "stat_marginal_damage_boost": 1.0,
    "stat_guard_boss_boost": 1.0,
    "reachable_resource_value": 1.2,
    "blocked_resource_pressure": 0.7,
    "reachable_enemy_damage_drop": 1.0,
    "reachable_enemy_cost": 0.8,
    "reachable_unlock_value": 1.0,
    "potion_need": 0.8,
    "key_buffer": 0.9,
    "mt10_resource_progress": 1.2,
    "boss_margin": 1.2,
}

DEFAULT_LEARNABLE_STAGE_OVERRIDES = {
    "sword": {
        "stage_sword": 1.8,
        "key_buffer": 1.4,
        "reachable_enemy_cost": 1.3,
        "reachable_resource_value": 1.1,
    },
    "mt4_redgem": {
        "stage_mt4_redgem": 2.2,
        "stat_marginal_damage_boost": 1.8,
        "reachable_enemy_damage_drop": 1.6,
        "key_buffer": 1.2,
    },
    "pre_shield_gems": {
        "stage_pre_shield_gems": 1.7,
        "stat_marginal_damage_boost": 1.4,
        "reachable_enemy_damage_drop": 1.5,
    },
    "shield": {
        "stage_shield": 1.9,
        "key_buffer": 1.3,
        "reachable_enemy_cost": 1.2,
    },
    "shield_buffer": {
        "stage_shield_buffer": 2.0,
        "key_buffer": 1.8,
        "reachable_enemy_cost": 1.5,
        "potion_need": 1.2,
    },
    "mt8_gems": {
        "stage_mt8_gems": 1.9,
        "stage_gems": 1.6,
        "reachable_enemy_damage_drop": 1.8,
    },
    "mid_gems": {
        "stage_mid_gems": 1.9,
        "stage_gems": 1.6,
        "reachable_enemy_damage_drop": 1.8,
    },
    "low_gems": {
        "stage_low_gems": 1.9,
        "stage_gems": 1.6,
        "reachable_enemy_damage_drop": 1.8,
    },
    "all_gems": {
        "stage_all_gems": 2.0,
        "stage_gems": 1.9,
        "stat_marginal_damage_boost": 1.8,
        "mt10_resource_progress": 1.8,
        "reachable_enemy_cost": 1.1,
    },
    "pre_mt10_buffer": {
        "stage_pre_mt10_buffer": 2.0,
        "stage_gems": 1.5,
        "key_buffer": 2.0,
        "potion_need": 1.7,
        "reachable_enemy_cost": 1.4,
        "mt10_resource_progress": 1.4,
    },
    "red_key": {
        "stage_red_key": 2.1,
        "reachable_unlock_value": 1.7,
        "boss_margin": 1.2,
    },
    "boss_ready": {
        "stage_boss_ready": 2.0,
        "boss_margin": 2.0,
        "potion_need": 1.8,
        "mt10_resource_progress": 1.5,
    },
    "trap": {
        "stage_trap": 2.0,
        "boss_margin": 2.2,
        "reachable_enemy_cost": 1.5,
    },
    "boss": {
        "stage_boss": 2.3,
        "boss_margin": 2.4,
        "reachable_enemy_damage_drop": 1.4,
    },
}


class LearnableStageReward:
    """PBRS reward with stage-specific, externally tunable factor weights.

    This is intentionally still a linear potential over interpretable factors.
    The "learnable" part is the weight table: Optuna/random search can tune the
    per-stage weights without replacing the reward with a black-box model.
    """

    def __init__(
        self,
        gamma: float = 0.99,
        global_weights: dict[str, float] | None = None,
        stage_weights: dict[str, dict[str, float]] | None = None,
    ):
        self.gamma = float(gamma)
        self.global_weights = {**DEFAULT_LEARNABLE_GLOBAL_WEIGHTS, **(global_weights or {})}
        self.stage_weights = {
            stage: {**DEFAULT_LEARNABLE_STAGE_OVERRIDES.get(stage, {}), **weights}
            for stage, weights in (stage_weights or {}).items()
        }
        for stage, weights in DEFAULT_LEARNABLE_STAGE_OVERRIDES.items():
            self.stage_weights.setdefault(stage, dict(weights))
        self._components_cache: dict[tuple[Any, ...], dict[str, float]] = {}
        self._potential_cache: dict[tuple[Any, ...], float] = {}
        self._cache_limit = 20000

    def weights_for_stage(self, stage: str) -> dict[str, float]:
        weights = dict(self.global_weights)
        weights.update(self.stage_weights.get(stage, {}))
        return weights

    def potential_components(
        self,
        sim: MotaSimulator,
        state: MotaState,
        stage: str | None = None,
    ) -> dict[str, float]:
        active_stage = stage or current_stage_name(sim, state)
        cache_key = None
        if hasattr(sim, "fast_state_key"):
            cache_key = (active_stage, sim.fast_state_key(state))
            cached = self._components_cache.get(cache_key)
            if cached is not None:
                return cached
        components = boosted_stat_stage_potential_components(sim, state, stage=active_stage)
        components.update(self._graph_factor_components(sim, state, active_stage))
        if cache_key is not None:
            if len(self._components_cache) >= self._cache_limit:
                self._components_cache.clear()
            self._components_cache[cache_key] = components
        return components

    def potential(self, sim: MotaSimulator, state: MotaState, stage: str | None = None) -> float:
        active_stage = stage or current_stage_name(sim, state)
        cache_key = None
        if hasattr(sim, "fast_state_key"):
            cache_key = (active_stage, sim.fast_state_key(state))
            cached = self._potential_cache.get(cache_key)
            if cached is not None:
                return cached
        weights = self.weights_for_stage(active_stage)
        value = sum(
            component * weights.get(key, 1.0)
            for key, component in self.potential_components(sim, state, active_stage).items()
        )
        if cache_key is not None:
            if len(self._potential_cache) >= self._cache_limit:
                self._potential_cache.clear()
            self._potential_cache[cache_key] = value
        return value

    def score(
        self,
        sim: MotaSimulator,
        before: MotaState,
        after: MotaState,
        action: dict[str, Any],
        transition: Transition,
    ) -> RewardBreakdown:
        stage = current_stage_name(sim, before)
        weights = self.weights_for_stage(stage)
        before_phi = self.potential_components(sim, before, stage=stage)
        after_phi = self.potential_components(sim, after, stage=stage)
        components: dict[str, float] = {"env_step": -0.02}
        if not transition.ok:
            components["invalid"] = -8.0
        for key in sorted(set(before_phi) | set(after_phi)):
            weighted_before = before_phi.get(key, 0.0) * weights.get(key, 1.0)
            weighted_after = after_phi.get(key, 0.0) * weights.get(key, 1.0)
            components[f"delta_{key}"] = self.gamma * weighted_after - weighted_before
        if not stage_complete(sim, before, stage) and stage_complete(sim, after, stage):
            components["stage_complete"] = 120.0
        if after.flags.get("10f战胜骷髅队长"):
            components["boss"] = 250.0
        if after.dead:
            components["dead"] = -250.0
        if "downFloor" in action.get("label", "") or "upFloor" in action.get("label", ""):
            components["stair_neutralizer"] = 0.02
        return RewardBreakdown(sum(components.values()), components)

    def _graph_factor_components(
        self,
        sim: MotaSimulator,
        state: MotaState,
        stage: str,
    ) -> dict[str, float]:
        reachable = sim.reachable_cells(state)
        item_value_cache: dict[str, float] = {}

        def current_item_value(item_id: str) -> float:
            cached = item_value_cache.get(item_id)
            if cached is not None:
                return cached
            value = item_value(sim, state, item_id)
            item_value_cache[item_id] = value
            return value

        reachable_items = 0.0
        blocked_items = 0.0
        mt10_remaining = 0
        for floor_id in sim.floor_order:
            if floor_id not in state.floors:
                continue
            for y, row in enumerate(state.floors[floor_id]):
                for x, tile in enumerate(row):
                    item_id = sim.block_id(tile)
                    if item_id not in {
                        "yellowKey",
                        "blueKey",
                        "redKey",
                        "redGem",
                        "blueGem",
                        "redPotion",
                        "bluePotion",
                        "sword1",
                        "shield1",
                    }:
                        continue
                    value = current_item_value(item_id)
                    if floor_id == state.floor_id and (x, y) in reachable:
                        reachable_items += value
                    else:
                        blocked_items += value
                    if floor_id == "MT10" and item_id in {"redGem", "blueGem", "bluePotion"}:
                        mt10_remaining += 1

        enemy_drop = 0.0
        enemy_cost = 0.0
        unlock_value = 0.0
        for x, y, enemy_id in adjacent_enemy_targets(sim, state, reachable)[:48]:
            info = sim.damage_info(state, enemy_id)
            if info is None:
                enemy_cost += 55.0
                continue
            damage = int(info["damage"])
            enemy_drop += damage_drop_for_stats(sim, state, enemy_id, atk_delta=1) * 0.8
            enemy_drop += damage_drop_for_stats(sim, state, enemy_id, def_delta=1) * 0.7
            enemy_cost -= min(damage, 1200) * 0.035
            child = state.clone()
            if sim.battle(child, x, y):
                after_reachable = sim.reachable_cells(child)
                for ix, iy in set(after_reachable) - set(reachable):
                    item_id = sim.block_id(sim.tile(child, ix, iy))
                    if item_id:
                        unlock_value += item_value(sim, child, item_id)

        boss_margin_value = max(-1000.0, min(1500.0, boss_route_margin(sim, state))) * 0.08
        potion_need = 0.0
        if state.hp < 700 or boss_route_margin(sim, state) < 0:
            potion_need = max(0.0, 900.0 - min(state.hp, 900)) * 0.08
        yk = state.items.get("yellowKey", 0)
        bk = state.items.get("blueKey", 0)
        rk = state.items.get("redKey", 0)
        key_buffer = yk * 12.0 + bk * 28.0 + rk * 85.0
        if stage in {"sword", "pre_shield_gems", "shield"} and yk <= 0:
            key_buffer -= 40.0

        return {
            "reachable_resource_value": reachable_items,
            "blocked_resource_pressure": -blocked_items * 0.12,
            "reachable_enemy_damage_drop": enemy_drop,
            "reachable_enemy_cost": enemy_cost,
            "reachable_unlock_value": unlock_value * 0.35,
            "potion_need": potion_need,
            "key_buffer": key_buffer,
            "mt10_resource_progress": (3 - mt10_remaining) * 65.0,
            "boss_margin": boss_margin_value,
        }


def label_components(action: dict[str, Any], after: MotaState) -> dict[str, float]:
    label = action.get("label", "")
    components: dict[str, float] = {"step": -0.002}
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
    bk_before = before.items.get("blueKey", 0)
    bk_after = after.items.get("blueKey", 0)
    stage = progress_stage(sim, before)
    if "yellowDoor" in label and stage < 4:
        components["early_yellow_door"] = -0.65
    if yk_before > 0 and yk_after == 0 and stage < 4:
        components["yellow_key_depleted"] = -1.2
    if yk_after >= 2 and stage < 4:
        components["yellow_key_buffer"] = 0.12
    late_key_stage = (
        before.flags.get("nowShield") == "shield1"
        or current_stage_name(sim, before)
        in {"shield_buffer", "mid_gems", "low_gems", "all_gems", "red_key", "boss_ready", "trap", "boss"}
    )
    if late_key_stage:
        if "yellowDoor" in label and yk_after < 2:
            components["late_yellow_door_key_pressure"] = -0.9 * float(2 - yk_after)
        if yk_before >= 2 and yk_after < 2:
            components["yellow_buffer_broken"] = -0.8 * float(2 - yk_after)
        if yk_before > 0 and yk_after == 0:
            components["late_yellow_depleted"] = -1.6
        if yk_after >= 2:
            components["late_yellow_buffer"] = 0.16
        if "blueDoor MT7:5,5" in label and bk_before > 0 and bk_after == 0:
            components["mt7_merchant_blue_door"] = 0.45
        elif "blueDoor" in label and bk_before > 0 and bk_after == 0 and not mt9_mt10_blue_door_opened(sim, after):
            components["blue_key_depleted_before_mt10_access"] = -1.1
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
    return stage * 2.0 + hp_term + key_buffer + stat_term - deadend_penalty


def dynamic_pbrs_potential(sim: MotaSimulator, state: MotaState) -> float:
    return sum(dynamic_potential_components(sim, state).values())


def dynamic_potential_components(sim: MotaSimulator, state: MotaState) -> dict[str, float]:
    """Topology-aware PBRS potential for the simplified Magic Tower task.

    The components intentionally mirror the project reward design notes:
    asset reserves, deterministic combat threshold value, and one-step unlock
    lookahead. The absolute scale is moderate so the same potential can be used
    both as Gym reward shaping and as a search heuristic component.
    """

    yk = state.items.get("yellowKey", 0)
    bk = state.items.get("blueKey", 0)
    rk = state.items.get("redKey", 0)
    stage = progress_stage(sim, state)

    yellow_weight = 4.0
    if stage >= 3:
        yellow_weight = 1.5
    elif floor_index(sim, state) >= 7 and not mt8_blue_key_taken(sim, state):
        yellow_weight = 8.0

    asset = (
        min(state.hp, 3000) * 0.002
        + state.atk * 1.8
        + state.defense * 1.7
        + state.money * 0.005
        + yk * yellow_weight
        + bk * 10.0
        + rk * 30.0
    )

    combat, lookahead = combat_and_lookahead_potential(sim, state)
    threshold = critical_threshold_potential(sim, state)
    progress = progress_stage(sim, state) * 35.0

    deadend = 0.0
    if floor_index(sim, state) >= 7 and yk == 0 and not mt8_blue_key_taken(sim, state):
        deadend -= 60.0
    if state.hp < 250 and progress_stage(sim, state) >= 2:
        deadend -= (250 - state.hp) * 0.08

    return {
        "asset": asset,
        "combat": combat,
        "threshold": threshold,
        "lookahead": lookahead * 0.25,
        "progress": progress,
        "deadend": deadend,
    }


def stage_potential(
    sim: MotaSimulator,
    state: MotaState,
    stage: str | None = None,
    weights: dict[str, float] | None = None,
) -> float:
    components = stage_potential_components(sim, state, stage=stage)
    merged = {**DEFAULT_STAGE_POTENTIAL_WEIGHTS, **(weights or {})}
    return sum(value * merged.get(key, 1.0) for key, value in components.items())


def stage_potential_components(
    sim: MotaSimulator,
    state: MotaState,
    stage: str | None = None,
) -> dict[str, float]:
    """Stage-aware PBRS potential.

    The base terms are topology/combat-aware. The stage terms make the same
    physical state evaluate differently depending on whether we are currently
    trying to secure the coarse shield, red-key, trap, or boss milestones. Sword,
    gem, and guard-readiness signals remain as internal subgoals.
    """

    active_stage = stage or current_stage_name(sim, state)
    components = dynamic_potential_components(sim, state)
    floor_idx = floor_index(sim, state)
    yk = state.items.get("yellowKey", 0)
    bk = state.items.get("blueKey", 0)
    rk = state.items.get("redKey", 0)
    remaining_gems = remaining_attack_defense_gems(sim, state)
    total_gems = max(1, total_attack_defense_gems(sim))
    captain = sim.damage_info(state, "skeletonCaptain")
    captain_damage = None if captain is None else captain["damage"]
    boss_damage = boss_route_required_damage(sim, state)
    boss_margin = max(-900.0, min(1200.0, boss_route_margin(sim, state)))
    boss_damage_score = 0.0 if captain_damage is None else max(0.0, 1200.0 - min(captain_damage, 1200)) * 1.2
    boss_atk_drop = damage_drop_for_stats(sim, state, "skeletonCaptain", atk_delta=1)
    boss_def_drop = damage_drop_for_stats(sim, state, "skeletonCaptain", def_delta=1)
    guard_atk_drop = damage_drop_for_stats(sim, state, "yellowGuard", atk_delta=1)
    guard_def_drop = damage_drop_for_stats(sim, state, "yellowGuard", def_delta=1)
    collected_gems = total_gems - remaining_gems

    components["stage_sword"] = 0.0
    components["stage_mt4_redgem"] = 0.0
    components["stage_pre_shield_gems"] = 0.0
    components["stage_shield"] = 0.0
    components["stage_shield_buffer"] = 0.0
    components["stage_gems"] = 0.0
    components["stage_lower_gems"] = 0.0
    components["stage_pre_mt10_buffer"] = 0.0
    components["stage_mt8_gems"] = 0.0
    components["stage_mid_gems"] = 0.0
    components["stage_low_gems"] = 0.0
    components["stage_all_gems"] = 0.0
    components["stage_mt10_blue_ready"] = 0.0
    components["stage_mt10_yellow_ready"] = 0.0
    components["stage_mt10_ready"] = 0.0
    components["stage_mt10_resources"] = 0.0
    components["stage_guard_ready"] = 0.0
    components["stage_red_key"] = 0.0
    components["stage_boss_ready"] = 0.0
    components["stage_trap"] = 0.0
    components["stage_boss"] = 0.0

    if active_stage == "sword":
        components["stage_sword"] = (
            (120.0 if has_first_sword(sim, state) else 0.0)
            + _target_floor_score(floor_idx, 5, 8.0)
            + min(4, yk) * 10.0
            + state.hp * 0.01
        )
        components["key_pressure"] = yk * 8.0 - max(0, 2 - yk) * 18.0
    elif active_stage == "mt4_redgem":
        taken = mt4_right_red_gem_taken(sim, state)
        components["stage_mt4_redgem"] = (
            (110.0 if has_first_sword(sim, state) else 0.0)
            + (180.0 if taken else 0.0)
            + (60.0 if tile_id(sim, state, "MT4", 8, 8) != "yellowDoor" else 0.0)
            + (45.0 if tile_id(sim, state, "MT4", 8, 9) != "bluePriest" else 0.0)
            + min(max(0, state.atk - 20), 2) * 70.0
            + _target_floor_score(floor_idx, 4, 24.0)
            + min(state.hp, 900) * 0.012
        )
        components["stage_sword"] = (
            75.0 if has_first_sword(sim, state) else _target_floor_score(floor_idx, 5, 4.0)
        )
        # Spending one yellow key to open MT4:8,8 is expected here; keep a
        # small buffer term so the route still values recovering MT5 keys next.
        components["key_pressure"] = yk * 5.0 + bk * 13.0
    elif active_stage == "pre_shield_gems":
        route_gem_progress = 1.0 if pre_shield_gems_ready(state) else 0.0
        target_floor = 3 if has_first_sword(sim, state) and not pre_shield_gems_ready(state) else 5
        components["stage_pre_shield_gems"] = (
            (130.0 if has_first_sword(sim, state) else 0.0)
            + route_gem_progress * 140.0
            + min(max(0, state.atk - 20), 3) * 35.0
            + min(max(0, state.defense - 10), 3) * 32.0
            + _target_floor_score(floor_idx, target_floor, 10.0)
            + min(state.hp, 900) * 0.018
        )
        components["stage_sword"] = (
            90.0 if has_first_sword(sim, state) else _target_floor_score(floor_idx, 5, 4.0)
        )
        components["key_pressure"] = yk * 7.0 + bk * 12.0
    elif active_stage == "shield":
        components["stage_shield"] = (
            (150.0 if mt9_shield_taken(sim, state) else 0.0)
            + (45.0 if mt8_blue_key_taken(sim, state) else 0.0)
            + (80.0 if has_first_sword(sim, state) else 0.0)
            + _target_floor_score(floor_idx, 9, 22.0)
            + critical_gem_progress(state) * 7.0
            + state.atk * 3.2
            + state.defense * 3.0
            + min(state.hp, 900) * 0.008
        )
        components["stage_sword"] = (
            55.0 if has_first_sword(sim, state) else _target_floor_score(floor_idx, 5, 3.0)
        )
        components["stage_gems"] = critical_gem_progress(state) * 4.5 - remaining_gems * 0.8
        components["key_pressure"] = bk * 24.0 + yk * 5.5
    elif active_stage == "shield_buffer":
        shield_taken = mt9_shield_taken(sim, state)
        components["stage_shield"] = 150.0 if shield_taken else _target_floor_score(floor_idx, 9, 18.0)
        components["stage_shield_buffer"] = (
            (180.0 if shield_taken else 0.0)
            + (70.0 if state.atk >= 21 else 0.0)
            + min(max(0, state.defense - 10), 10) * 8.0
            + min(max(0, state.hp), 420) * 0.09
            + min(yk, 2) * 45.0
            + min(bk, 1) * 20.0
            + _target_floor_score(floor_idx, 7 if shield_taken else 9, 10.0)
        )
        components["stage_gems"] = critical_gem_progress(state) * 6.0 - remaining_gems * 0.7
        components["key_pressure"] = yk * 16.0 + bk * 12.0 - max(0, 2 - yk) * 25.0
    elif active_stage == "gems":
        collected = total_gems - remaining_gems
        pre_progress = pre_mt10_stat_progress(state)
        components["stage_gems"] = (
            pre_progress * 70.0
            + critical_gem_progress(state) * 18.0
            + (180.0 if pre_mt10_stats_ready(state) else 0.0)
            + collected * 12.0
            - remaining_gems * 2.0
        )
        components["global_resource"] = -remaining_gems * 4.0
        components["stage_boss"] = boss_atk_drop * 2.2 + boss_def_drop * 1.5
    elif active_stage in {"lower_gems", "mt8_gems", "mid_gems", "low_gems"}:
        lower_remaining = remaining_lower_attack_defense_gems(sim, state)
        lower_total = max(1, total_lower_attack_defense_gems(sim))
        lower_collected = lower_total - lower_remaining
        target_remaining = remaining_stage_gem_targets(sim, state, active_stage)
        target_total = max(1, len(GEM_STAGE_TARGETS.get(active_stage, ())))
        target_collected = target_total - target_remaining
        lower_done = lower_attack_defense_gems_taken(sim, state)
        components["stage_lower_gems"] = (
            (330.0 if lower_done else 0.0)
            + lower_collected * 38.0
            + critical_gem_progress(state) * 36.0
            + (100.0 if mt9_shield_taken(sim, state) else 0.0)
            + state.atk * 5.0
            + state.defense * 5.0
            + min(state.hp, 1400) * 0.05
            - lower_remaining * 15.0
        )
        components["stage_gems"] = (
            critical_gem_progress(state) * 46.0
            + guard_atk_drop * 2.6
            + guard_def_drop * 2.2
            + boss_atk_drop * 2.4
            + boss_def_drop * 2.0
        )
        components["global_resource"] = -lower_remaining * 16.0
        components["key_pressure"] = yk * 10.0 + bk * 18.0 - max(0, 2 - yk) * 15.0
        if active_stage in {"mt8_gems", "mid_gems", "low_gems"}:
            components[f"stage_{active_stage}"] = (
                (280.0 if stage_gem_targets_taken(sim, state, active_stage) else 0.0)
                + target_collected * 120.0
                + critical_gem_progress(state) * 22.0
                + state.atk * 4.0
                + state.defense * 4.0
                + min(state.hp, 1200) * 0.05
                - target_remaining * 42.0
            )
    elif active_stage == "all_gems":
        all_done = all_attack_defense_gems_taken(sim, state)
        components["stage_all_gems"] = (
            (420.0 if all_done else 0.0)
            + collected_gems * 34.0
            + critical_gem_progress(state) * 32.0
            + mt10_resource_progress(sim, state) * 95.0
            + (90.0 if mt9_shield_taken(sim, state) else 0.0)
            + (60.0 if mt10_access_ready(sim, state) else 0.0)
            + state.atk * 5.5
            + state.defense * 5.2
            + min(state.hp, 1400) * 0.055
            - remaining_gems * 12.0
        )
        components["stage_gems"] = (
            critical_gem_progress(state) * 48.0
            + boss_atk_drop * 3.2
            + boss_def_drop * 2.4
            + guard_atk_drop * 2.8
            + guard_def_drop * 2.4
        )
        components["global_resource"] = -remaining_gems * 14.0
        components["key_pressure"] = yk * 12.0 + bk * 20.0 - max(0, 2 - yk) * 18.0
    elif active_stage == "mt10_blue_ready":
        needed_blue = 0 if mt9_mt10_blue_door_opened(sim, state) else 1
        components["stage_mt10_blue_ready"] = (
            (220.0 if mt10_blue_ready(sim, state) else 0.0)
            + pre_mt10_stat_progress(state) * 50.0
            + _target_floor_score(floor_idx, 9, 10.0)
            + min(state.hp, 360) * 0.12
            + min(max(needed_blue, 1), bk) * 60.0
            - max(0, needed_blue - bk) * 120.0
        )
        components["stage_gems"] = pre_mt10_stat_progress(state) * 35.0
    elif active_stage in {"mt10_yellow_ready", "mt10_ready"}:
        needed_blue = 0 if mt9_mt10_blue_door_opened(sim, state) else 1
        target_yk = MT10_RESOURCE_YELLOW_KEY_TARGET
        components["stage_mt10_yellow_ready"] = (
            (260.0 if mt10_access_ready(sim, state) else 0.0)
            + pre_mt10_stat_progress(state) * 55.0
            + _target_floor_score(floor_idx, 9, 11.0)
            + min(state.hp, 360) * 0.18
            + min(target_yk, yk) * 22.0
            + min(max(needed_blue, 1), bk) * 32.0
            - max(0, 235 - state.hp) * 1.8
            - max(0, target_yk - yk) * 65.0
            - max(0, needed_blue - bk) * 80.0
        )
        components["stage_mt10_ready"] = components["stage_mt10_yellow_ready"]
        components["stage_gems"] = pre_mt10_stat_progress(state) * 45.0
    elif active_stage == "mt10_resources":
        mt10_progress = mt10_resource_progress(sim, state)
        mt10_remaining = len(MT10_RESOURCE_TARGETS) - mt10_progress
        components["stage_mt10_resources"] = (
            (260.0 if mt10_resources_taken(sim, state) else 0.0)
            + mt10_progress * 95.0
            + _target_floor_score(floor_idx, 10, 5.5)
            + (70.0 if state.floor_id == "MT10" else 0.0)
            + state.atk * 4.0
            + state.defense * 4.0
            + min(state.hp, 1200) * 0.05
            - mt10_remaining * 15.0
        )
        components["stage_gems"] = critical_gem_progress(state) * 52.0 + collected_gems * 18.0
        components["key_pressure"] = yk * 18.0 + bk * 18.0 - max(0, 2 - yk) * 28.0
    elif active_stage == "guard_ready":
        guard_damage = red_key_route_damage(sim, state)
        guard_margin = red_key_route_margin(sim, state)
        components["stage_guard_ready"] = (
            (260.0 if guard_ready(sim, state) else 0.0)
            + max(0.0, 1600.0 - min(guard_damage, 1600.0)) * 0.18
            + max(-600.0, min(guard_margin, 900.0)) * 0.12
            + state.atk * 7.5
            + state.defense * 6.8
            + min(state.hp, 1400) * 0.05
        )
        components["stage_gems"] = (
            critical_gem_progress(state) * 72.0
            + collected_gems * 18.0
            + max(0.0, 1200.0 - min(float(guard_damage), 1200.0)) * 1.2
            + guard_atk_drop * 4.2
            + guard_def_drop * 3.6
            + boss_atk_drop * 1.0
            + boss_def_drop * 0.8
            - remaining_gems * 3.2
        )
        components["key_pressure"] = state.items.get("yellowKey", 0) * 12.0 + state.items.get("blueKey", 0) * 20.0
    elif active_stage == "red_key":
        guard_damage = red_key_route_damage(sim, state)
        guard_margin = red_key_route_margin(sim, state)
        components["stage_red_key"] = (
            (180.0 if red_key_taken(sim, state) else 0.0)
            + rk * 70.0
            + (120.0 if guard_ready(sim, state) else 0.0)
            + (35.0 if floor_idx >= 8 else 0.0)
            + max(0.0, 1600.0 - min(guard_damage, 1600.0)) * 0.11
            + max(-700.0, min(guard_margin, 1000.0)) * 0.13
            + yk * 3.0
        )
        components["stage_guard_ready"] = (
            (240.0 if guard_ready(sim, state) else 0.0)
            + max(0.0, 1800.0 - min(guard_damage, 1800.0)) * 0.18
            + max(-900.0, min(guard_margin, 1200.0)) * 0.16
            + state.atk * 8.0
            + state.defense * 7.0
            + min(state.hp, 1600) * 0.055
        )
        components["stage_gems"] = (
            critical_gem_progress(state) * 84.0
            + collected_gems * 20.0
            + max(0.0, 1200.0 - min(float(guard_damage), 1200.0)) * 1.4
            + guard_atk_drop * 4.8
            + guard_def_drop * 4.0
            + boss_atk_drop * 2.3
            + boss_def_drop * 1.8
            - remaining_gems * 3.5
        )
        components["key_pressure"] = yk * 10.0 + bk * 16.0 + rk * 70.0
    elif active_stage == "boss_ready":
        components["stage_boss_ready"] = (
            (260.0 if boss_ready(sim, state) else 0.0)
            + (150.0 if red_key_taken(sim, state) else 0.0)
            + boss_damage_score
            + boss_margin * 1.0
            + state.atk * 4.0
            + state.defense * 3.8
            + min(state.hp, 1800) * 0.20
        )
        components["stage_boss"] = boss_atk_drop * 2.8 + boss_def_drop * 2.2
        components["key_pressure"] = rk * 70.0 + yk * 6.0
    elif active_stage == "trap":
        components["stage_trap"] = (
            (220.0 if state.flags.get("10f机关") else 0.0)
            + _target_floor_score(floor_idx, 10, 6.0)
            + boss_margin * 0.035
        )
    elif active_stage == "boss":
        components["stage_boss"] = (
            (500.0 if state.flags.get("10f战胜骷髅队长") else 0.0)
            + boss_damage_score
            + boss_margin * 0.12
            + boss_atk_drop * 1.5
            + boss_def_drop * 1.0
            + state.atk * 2.5
            + state.defense * 2.0
        )
    components["boss_margin"] = boss_margin * 0.04
    return components


def _target_floor_score(current_floor: int, target_floor: int, scale: float) -> float:
    """Score proximity to a stage target floor instead of raw floor height."""

    return max(0.0, float(target_floor - abs(current_floor - target_floor))) * scale


def boosted_stat_stage_potential_components(
    sim: MotaSimulator,
    state: MotaState,
    stage: str | None = None,
) -> dict[str, float]:
    """Stage PBRS with stronger attack/defense pressure.

    This keeps the existing topology-aware stage potential intact, then adds
    explicit stat terms so red/blue gems, sword, and shield generate a larger
    immediate PBRS signal. The extra terms still live inside Phi(s), so they are
    easier to compare against the base `stage_pbrs` run than hand-written
    one-off item rewards.
    """

    active_stage = stage or current_stage_name(sim, state)
    components = stage_potential_components(sim, state, stage=stage)
    guard_damage = red_key_route_damage(sim, state)
    guard_margin = red_key_route_margin(sim, state)
    route_damage = red_key_route_damage(sim, state)
    boss_damage = boss_route_required_damage(sim, state)
    boss_margin = max(-900.0, min(1200.0, boss_route_margin(sim, state)))
    key_enemies = ("skeletonCaptain", "yellowGuard", "skeletonSoldier", "bluePriest", "bat")
    marginal_atk = sum(damage_drop_for_stats(sim, state, enemy_id, atk_delta=1) for enemy_id in key_enemies)
    marginal_def = sum(damage_drop_for_stats(sim, state, enemy_id, def_delta=1) for enemy_id in key_enemies)

    components["stat_asset_boost"] = state.atk * 7.5 + state.defense * 7.0
    components["stat_threshold_boost"] = critical_gem_progress(state) * 22.0
    components["stat_marginal_damage_boost"] = marginal_atk * 0.55 + marginal_def * 0.45

    if active_stage in {
        "mt4_redgem",
        "pre_shield_gems",
        "shield",
        "shield_buffer",
        "lower_gems",
        "mt8_gems",
        "mid_gems",
        "low_gems",
        "all_gems",
        "mt10_blue_ready",
        "mt10_yellow_ready",
        "mt10_ready",
        "mt10_resources",
        "guard_ready",
        "red_key",
        "boss_ready",
        "trap",
        "boss",
    }:
        components["stat_guard_boss_boost"] = (
            max(0.0, 1800.0 - min(float(route_damage), 1800.0)) * 0.10
            + max(-900.0, min(float(guard_margin), 1200.0)) * 0.06
            + max(0.0, 1600.0 - min(float(boss_damage), 1600.0)) * 0.08
        )
    if active_stage == "mt4_redgem":
        components["stage_mt4_redgem"] = components.get("stage_mt4_redgem", 0.0) + (
            state.atk * 3.5
            + critical_gem_progress(state) * 8.0
            + marginal_atk * 0.35
        )
    elif active_stage == "pre_shield_gems":
        components["stage_pre_shield_gems"] = components.get("stage_pre_shield_gems", 0.0) + (
            state.atk * 2.5
            + state.defense * 2.0
            + critical_gem_progress(state) * 5.0
        )
    elif active_stage == "shield":
        components["stage_shield"] = components.get("stage_shield", 0.0) + (
            state.atk * 3.5
            + state.defense * 4.0
            + critical_gem_progress(state) * 8.0
        )
    elif active_stage == "shield_buffer":
        components["stage_shield_buffer"] = components.get("stage_shield_buffer", 0.0) + (
            state.atk * 4.0
            + state.defense * 4.5
            + min(max(0, state.hp), 500) * 0.04
            + min(state.items.get("yellowKey", 0), 2) * 18.0
            + critical_gem_progress(state) * 10.0
        )
    elif active_stage == "lower_gems":
        components["stage_lower_gems"] = components.get("stage_lower_gems", 0.0) + (
            state.atk * 5.0
            + state.defense * 5.0
            + critical_gem_progress(state) * 16.0
            + marginal_atk * 0.55
            + marginal_def * 0.50
            - remaining_lower_attack_defense_gems(sim, state) * 7.0
        )
    elif active_stage == "pre_mt10_buffer":
        needed_blue = 0 if mt9_mt10_blue_door_opened(sim, state) else 1
        components["stage_pre_mt10_buffer"] = components.get("stage_pre_mt10_buffer", 0.0) + (
            min(max(0, state.hp), 420) * 0.18
            + min(state.atk, 26) * 8.0
            + min(state.defense, 26) * 8.5
            + min(state.items.get("yellowKey", 0), 4) * 24.0
            + min(state.items.get("blueKey", 0), max(1, needed_blue)) * 18.0
            + mt10_resource_progress(sim, state) * 45.0
            - max(0, 25 - state.atk) * 32.0
            - max(0, 26 - state.defense) * 32.0
            - max(0, 240 - state.hp) * 0.35
            - max(0, 2 - state.items.get("yellowKey", 0)) * 35.0
            - max(0, needed_blue - state.items.get("blueKey", 0)) * 40.0
        )
        components["stage_gems"] = components.get("stage_gems", 0.0) + (
            marginal_atk * 0.35
            + marginal_def * 0.45
            - remaining_lower_attack_defense_gems(sim, state) * 3.0
        )
    elif active_stage in {"mt8_gems", "mid_gems", "low_gems"}:
        target_remaining = remaining_stage_gem_targets(sim, state, active_stage)
        components[f"stage_{active_stage}"] = components.get(f"stage_{active_stage}", 0.0) + (
            state.atk * 4.5
            + state.defense * 4.5
            + critical_gem_progress(state) * 15.0
            + marginal_atk * 0.45
            + marginal_def * 0.45
            - target_remaining * 12.0
        )
    elif active_stage == "all_gems":
        components["stage_all_gems"] = components.get("stage_all_gems", 0.0) + (
            state.atk * 6.0
            + state.defense * 6.0
            + critical_gem_progress(state) * 18.0
            + marginal_atk * 0.75
            + marginal_def * 0.65
            + mt10_resource_progress(sim, state) * 45.0
            - remaining_attack_defense_gems(sim, state) * 6.0
        )
    elif active_stage == "guard_ready":
        components["stage_guard_ready"] = components.get("stage_guard_ready", 0.0) + (
            state.atk * 7.0
            + state.defense * 7.0
            + critical_gem_progress(state) * 16.0
        )
        components["stage_gems"] = components.get("stage_gems", 0.0) + (
            marginal_atk * 0.75
            + marginal_def * 0.65
            + critical_gem_progress(state) * 18.0
        )
    elif active_stage == "mt10_resources":
        components["stage_mt10_resources"] = components.get("stage_mt10_resources", 0.0) + (
            mt10_resource_progress(sim, state) * 38.0
            + state.atk * 5.5
            + state.defense * 5.0
            + critical_gem_progress(state) * 12.0
        )
        components["stage_gems"] = components.get("stage_gems", 0.0) + (
            marginal_atk * 0.45 + marginal_def * 0.45
        )
    elif active_stage == "red_key":
        components["stage_red_key"] = components.get("stage_red_key", 0.0) + (
            state.atk * 4.5
            + state.defense * 4.5
            + critical_gem_progress(state) * 9.0
        )
        components["stage_gems"] = components.get("stage_gems", 0.0) + (
            marginal_atk * 0.65
            + marginal_def * 0.55
            + critical_gem_progress(state) * 14.0
        )
    elif active_stage == "boss_ready":
        components["stage_boss_ready"] = components.get("stage_boss_ready", 0.0) + (
            state.atk * 4.5
            + state.defense * 4.0
            + critical_gem_progress(state) * 10.0
            + max(0.0, 1600.0 - min(float(boss_damage), 1600.0)) * 0.10
            + boss_margin * 0.35
        )
    elif active_stage == "boss":
        components["stage_boss"] = components.get("stage_boss", 0.0) + (
            state.atk * 4.0
            + state.defense * 3.5
            + critical_gem_progress(state) * 7.0
        )
    return components


def current_stage_name(sim: MotaSimulator, state: MotaState) -> str:
    cache = getattr(state, "_signature_cache", None) if _stage_cache_enabled() else None
    cache_key = ("current_stage", _stage_cache_fingerprint(state))
    if cache is not None and cache_key in cache:
        return str(cache[cache_key])
    for stage in STAGE_ORDER:
        if stage == "mt8_hp_ready" and (
            floor_index(sim, state) >= 8 or stage_gem_targets_taken(sim, state, "mt8_gems")
        ):
            continue
        if stage == "lower_gems" and (
            floor_index(sim, state) >= 8 or stage_gem_targets_taken(sim, state, "mt8_gems")
        ):
            continue
        if not stage_complete(sim, state, stage):
            if cache is not None:
                cache[cache_key] = stage
            return stage
    if cache is not None:
        cache[cache_key] = "done"
    return "done"


def stage_complete(sim: MotaSimulator, state: MotaState, stage: str) -> bool:
    cache = getattr(state, "_signature_cache", None) if _stage_cache_enabled() else None
    cache_key = ("stage_complete", stage, _stage_cache_fingerprint(state))
    if cache is not None and cache_key in cache:
        return bool(cache[cache_key])
    result: bool
    if stage == "sword":
        result = has_first_sword(sim, state)
    elif stage == "mt4_redgem":
        result = mt4_right_red_gem_taken(sim, state)
    elif stage == "pre_shield_gems":
        result = pre_shield_gems_ready(state)
    elif stage == "shield":
        result = mt9_shield_taken(sim, state)
    elif stage == "shield_buffer":
        result = shield_buffer_ready(sim, state)
    elif stage == "gems":
        result = pre_mt10_stats_ready(state)
    elif stage == "lower_gems":
        result = lower_attack_defense_gems_taken(sim, state)
    elif stage == "pre_mt10_buffer":
        result = pre_mt10_buffer_ready(sim, state)
    elif stage == "mt8_hp_ready":
        result = stage_gem_targets_taken(sim, state, "low_gems") and mt8_right_chain_ready(state)
    elif stage == "mid_gems":
        result = stage_gem_targets_taken(sim, state, stage) and (
            state.hp >= 120 or (state.atk >= 26 and state.defense >= 26)
        )
    elif stage in GEM_STAGE_TARGETS:
        result = stage_gem_targets_taken(sim, state, stage)
    elif stage == "all_gems":
        result = all_attack_defense_gems_taken(sim, state)
    elif stage == "mt10_blue_ready":
        result = mt10_blue_ready(sim, state)
    elif stage == "mt10_yellow_ready":
        needed_blue = 0 if mt9_mt10_blue_door_opened(sim, state) else 1
        result = (
            mt10_blue_ready(sim, state)
            and state.items.get("yellowKey", 0) >= MT10_RESOURCE_YELLOW_KEY_TARGET
            and state.items.get("blueKey", 0) >= needed_blue
            and state.hp >= 180
        )
    elif stage == "mt10_ready":
        result = mt10_access_ready(sim, state)
    elif stage == "mt10_resources":
        result = mt10_resources_taken(sim, state)
    elif stage == "guard_ready":
        result = guard_ready(sim, state)
    elif stage == "red_key":
        result = red_key_taken(sim, state)
    elif stage == "boss_ready":
        result = boss_ready(sim, state)
    elif stage == "trap":
        result = bool(state.flags.get("10f机关"))
    elif stage == "boss":
        result = bool(state.flags.get("10f战胜骷髅队长"))
    elif stage == "boss_all_gems":
        result = bool(state.flags.get("10f战胜骷髅队长")) and all_attack_defense_gems_taken(
            sim, state
        )
    elif stage == "done":
        result = True
    else:
        raise ValueError(f"Unknown stage {stage!r}; choose one of {STAGE_ORDER}")
    if cache is not None:
        cache[cache_key] = result
    return result


def _stage_cache_fingerprint(state: MotaState) -> tuple[Any, ...]:
    return (
        state.floor_id,
        int(state.x),
        int(state.y),
        int(state.hp),
        int(state.atk),
        int(state.defense),
        int(state.mdef),
        int(state.money),
        int(state.exp),
        tuple(sorted((str(key), int(value)) for key, value in state.items.items() if value)),
        tuple(sorted((str(key), repr(value)) for key, value in state.flags.items() if value)),
        tuple(sorted(state.triggered_events)),
    )


def _stage_cache_enabled() -> bool:
    return os.environ.get("MOTA_ENABLE_STAGE_CACHE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def remaining_attack_defense_gems(sim: MotaSimulator, state: MotaState) -> int:
    count = 0
    for floor_id in sim.floor_order:
        if floor_id not in state.floors:
            continue
        for row in state.floors[floor_id]:
            for tile in row:
                if sim.block_id(tile) in {"redGem", "blueGem"}:
                    count += 1
    return count


def total_attack_defense_gems(sim: MotaSimulator) -> int:
    cached = getattr(sim, "_total_attack_defense_gems", None)
    if cached is not None:
        return int(cached)
    count = 0
    for floor_id in sim.floor_order:
        original_map = sim.data.floors[floor_id]["map"]
        for row in original_map:
            for tile in row:
                if sim.block_id(tile) in {"redGem", "blueGem"}:
                    count += 1
    setattr(sim, "_total_attack_defense_gems", count)
    return count


def remaining_lower_attack_defense_gems(sim: MotaSimulator, state: MotaState) -> int:
    count = 0
    for floor_id in sim.floor_order:
        if floor_id == "MT10" or floor_id not in state.floors:
            continue
        for row in state.floors[floor_id]:
            for tile in row:
                if sim.block_id(tile) in {"redGem", "blueGem"}:
                    count += 1
    return count


def total_lower_attack_defense_gems(sim: MotaSimulator) -> int:
    cached = getattr(sim, "_total_lower_attack_defense_gems", None)
    if cached is not None:
        return int(cached)
    count = 0
    for floor_id in sim.floor_order:
        if floor_id == "MT10":
            continue
        original_map = sim.data.floors[floor_id]["map"]
        for row in original_map:
            for tile in row:
                if sim.block_id(tile) in {"redGem", "blueGem"}:
                    count += 1
    setattr(sim, "_total_lower_attack_defense_gems", count)
    return count


def critical_gems_ready(state: MotaState) -> bool:
    return state.atk >= CRITICAL_GEM_TARGET_ATK and state.defense >= CRITICAL_GEM_TARGET_DEF


def pre_shield_gems_ready(state: MotaState) -> bool:
    if state.flags.get("nowShield") == "shield1":
        return True
    return (
        state.flags.get("nowWeapon") == "sword1"
        and state.atk >= 21
        and state.defense >= 11
        and state.hp >= 400
    )


def shield_buffer_ready(sim: MotaSimulator, state: MotaState) -> bool:
    """A continuation-safe shield milestone.

    Merely taking the 9F shield can be a bad terminal set for single-player
    AlphaZero: the search may accept HP/key-starved states that satisfy the
    local milestone but have poor continuation value.  This predicate keeps the
    same route target while requiring enough buffer to continue to gem/red-key
    stages without immediately dead-ending.
    """

    if not mt9_shield_taken(sim, state):
        return False
    if state.hp >= 280 and state.defense >= 20 and state.items.get("yellowKey", 0) >= 2:
        return True
    if state.atk >= 24 and state.defense >= 22 and state.items.get("yellowKey", 0) >= 2:
        return True
    return (
        state.hp >= 300
        and state.atk >= 21
        and state.defense >= 20
        and state.items.get("yellowKey", 0) >= 2
    )


def mt8_right_chain_ready(state: MotaState) -> bool:
    """Continuation-safe resource condition before committing to the 8F gem route.

    The 8F blue-key chain has two strict deterministic-resource variants.  With
    two yellow keys, the route must skip the left potion and survive the right
    skeleton-soldier branch directly.  With three yellow keys, it can spend one
    key on the left potion and still preserve the two yellow doors required by
    the right-bottom blue-key corridor.
    """

    yellow_keys = state.items.get("yellowKey", 0)
    if yellow_keys >= 2 and state.hp >= 340:
        return True
    if yellow_keys >= 3 and state.hp >= 273:
        return True
    return False


def mt4_right_red_gem_taken(sim: MotaSimulator, state: MotaState) -> bool:
    return mt9_shield_taken(sim, state) or tile_id(sim, state, "MT4", 7, 10) != "redGem"


def all_attack_defense_gems_taken(sim: MotaSimulator, state: MotaState) -> bool:
    return remaining_attack_defense_gems(sim, state) == 0


def lower_attack_defense_gems_taken(sim: MotaSimulator, state: MotaState) -> bool:
    return remaining_lower_attack_defense_gems(sim, state) == 0


def remaining_stage_gem_targets(sim: MotaSimulator, state: MotaState, stage: str) -> int:
    remaining = 0
    for floor_id, x, y, item_id in GEM_STAGE_TARGETS.get(stage, ()):
        if tile_id(sim, state, floor_id, x, y) == item_id:
            remaining += 1
    return remaining


def stage_gem_targets_taken(sim: MotaSimulator, state: MotaState, stage: str) -> bool:
    return remaining_stage_gem_targets(sim, state, stage) == 0


def critical_gem_progress(state: MotaState) -> float:
    atk_progress = min(CRITICAL_GEM_TARGET_ATK, state.atk) - 10
    def_progress = min(CRITICAL_GEM_TARGET_DEF, state.defense) - 10
    return max(0.0, float(atk_progress + def_progress))


def pre_mt10_stats_ready(state: MotaState) -> bool:
    return state.atk >= PRE_MT10_TARGET_ATK and state.defense >= PRE_MT10_TARGET_DEF


def pre_mt10_stat_progress(state: MotaState) -> float:
    atk_progress = min(PRE_MT10_TARGET_ATK, state.atk) - 10
    def_progress = min(PRE_MT10_TARGET_DEF, state.defense) - 10
    return max(0.0, float(atk_progress + def_progress))


def yellow_guard_damage(sim: MotaSimulator, state: MotaState) -> int:
    info = sim.damage_info(state, "yellowGuard")
    if info is None:
        return 10_000
    return info["damage"]


def yellow_guard_margin(sim: MotaSimulator, state: MotaState) -> int:
    damage = yellow_guard_damage(sim, state)
    if damage >= 10_000:
        return -10_000
    return state.hp - 2 * damage - GUARD_READY_HP_BUFFER


def red_key_route_damage(sim: MotaSimulator, state: MotaState) -> int:
    """Damage needed to clear the currently remaining direct route to 8F red key."""

    total = 0
    for floor_id, x, y, enemy_id in RED_KEY_ROUTE_ENEMIES:
        if tile_id(sim, state, floor_id, x, y) != enemy_id:
            continue
        info = sim.damage_info(state, enemy_id)
        if info is None:
            return 10_000
        total += int(info["damage"])
    return total


def red_key_route_margin(sim: MotaSimulator, state: MotaState) -> int:
    damage = red_key_route_damage(sim, state)
    if damage >= 10_000:
        return -10_000
    return state.hp - damage - GUARD_READY_HP_BUFFER


def mt10_resource_progress(sim: MotaSimulator, state: MotaState) -> int:
    progress = 0
    for floor_id, x, y, item_id in MT10_RESOURCE_TARGETS:
        if tile_id(sim, state, floor_id, x, y) != item_id:
            progress += 1
    return progress


def mt10_resources_taken(sim: MotaSimulator, state: MotaState) -> bool:
    return mt10_resource_progress(sim, state) == len(MT10_RESOURCE_TARGETS)


def mt9_mt10_blue_door_opened(sim: MotaSimulator, state: MotaState) -> bool:
    return tile_id(sim, state, "MT9", 3, 11) != "blueDoor"


def mt10_blue_ready(sim: MotaSimulator, state: MotaState) -> bool:
    if state.floor_id == "MT10" or mt10_resource_progress(sim, state) > 0:
        return True
    needed_blue = 0 if mt9_mt10_blue_door_opened(sim, state) else 1
    return pre_mt10_stats_ready(state) and state.items.get("blueKey", 0) >= needed_blue


def pre_mt10_buffer_ready(sim: MotaSimulator, state: MotaState) -> bool:
    """Continuation-safe state before committing to 10F resource doors.

    The lower-gem milestone can be satisfied by routes that spend all keys or
    arrive with tiny HP.  This predicate keeps the next archive frontier focused
    on states that still have enough deterministic resources to attempt the 10F
    resource branch.
    """

    if mt10_resource_progress(sim, state) > 0:
        return True
    needed_blue = 0 if mt9_mt10_blue_door_opened(sim, state) else 1
    return (
        state.atk >= 25
        and state.defense >= 26
        and state.hp >= 240
        and state.items.get("yellowKey", 0) >= 2
        and state.items.get("blueKey", 0) >= needed_blue
    )


def mt10_access_ready(sim: MotaSimulator, state: MotaState) -> bool:
    if mt10_resource_progress(sim, state) > 0:
        return True
    needed_blue = 0 if mt9_mt10_blue_door_opened(sim, state) else 1
    return (
        mt10_blue_ready(sim, state)
        and state.items.get("yellowKey", 0) >= MT10_RESOURCE_YELLOW_KEY_TARGET
    )


def guard_ready(sim: MotaSimulator, state: MotaState) -> bool:
    damage = red_key_route_damage(sim, state)
    return damage < 10_000 and state.hp > damage + GUARD_READY_HP_BUFFER


def red_key_taken(sim: MotaSimulator, state: MotaState) -> bool:
    red_key_collected = state.items.get("redKey", 0) > 0 or tile_id(sim, state, "MT8", 10, 2) != "redKey"
    if not red_key_collected:
        return False
    return (
        tile_id(sim, state, "MT8", 9, 3) != "bluePotion"
        and tile_id(sim, state, "MT8", 11, 3) != "redPotion"
        and tile_id(sim, state, "MT8", 9, 1) != "yellowKey"
        and tile_id(sim, state, "MT8", 11, 1) != "yellowKey"
    )


def boss_ready(sim: MotaSimulator, state: MotaState) -> bool:
    if state.items.get("redKey", 0) <= 0 and not red_key_taken(sim, state):
        return False
    required = boss_route_required_damage(sim, state)
    return required < 10_000 and state.hp > required


MT10_TRAP_GUARD_CELLS = (
    (5, 4),
    (6, 4),
    (7, 4),
    (5, 5),
    (7, 5),
    (5, 6),
    (6, 6),
    (7, 6),
)
MT10_UNTRIGGERED_TRAP_ENEMIES = (
    "skeletonSoldier",
    "skeletonSoldier",
    "skeleton",
    "skeleton",
    "skeleton",
    "skeleton",
    "skeleton",
    "skeleton",
)


def boss_route_required_damage(sim: MotaSimulator, state: MotaState) -> int:
    """Damage still required before the 10F skeleton captain flag can be set.

    The captain is unreachable until the 10F trap guards are cleared. A state
    that can survive the captain alone is therefore not a valid boss-ready
    state; it must also have enough HP for either the untriggered trap wave or
    the currently remaining trap monsters.
    """

    if state.flags.get("10f战胜骷髅队长"):
        return 0
    captain = sim.damage_info(state, "skeletonCaptain")
    if captain is None:
        return 100_000

    required = int(captain["damage"])
    if state.flags.get("10f机关"):
        for x, y in MT10_TRAP_GUARD_CELLS:
            enemy_id = sim.block_id(sim.tile(state, x, y, "MT10"))
            if enemy_id in {"skeleton", "skeletonSoldier"}:
                info = sim.damage_info(state, enemy_id)
                if info is None:
                    return 100_000
                required += int(info["damage"])
    else:
        for enemy_id in MT10_UNTRIGGERED_TRAP_ENEMIES:
            info = sim.damage_info(state, enemy_id)
            if info is None:
                return 100_000
            required += int(info["damage"])
    return required


def boss_route_margin(sim: MotaSimulator, state: MotaState) -> int:
    return state.hp - boss_route_required_damage(sim, state)


def damage_drop_for_stats(
    sim: MotaSimulator,
    state: MotaState,
    enemy_id: str,
    atk_delta: int = 0,
    def_delta: int = 0,
) -> int:
    current = sim.damage_info(state, enemy_id)
    if current is None:
        return 0
    improved = sim.damage_info_for_stats(
        enemy_id,
        atk=state.atk + atk_delta,
        defense=state.defense + def_delta,
        mdef=state.mdef,
    )
    if improved is None:
        return 0
    return max(0, current["damage"] - improved["damage"])


def combat_and_lookahead_potential(sim: MotaSimulator, state: MotaState) -> tuple[float, float]:
    reachable = sim.reachable_cells(state)
    enemy_targets = adjacent_enemy_targets(sim, state, reachable)
    combat = 0.0
    lookahead = 0.0
    base_atk, base_def = 10, 10

    for x, y, enemy_id in enemy_targets[:32]:
        current_info = sim.damage_info(state, enemy_id)
        if current_info is None:
            continue
        baseline_damage = damage_with_stats(sim, state, enemy_id, base_atk, base_def)
        if baseline_damage is None:
            baseline_damage = 500
        saved_hp = max(0, min(baseline_damage, 500) - min(current_info["damage"], 500))
        combat += saved_hp * 0.018

        one_more_atk = damage_with_stats(sim, state, enemy_id, state.atk + 1, state.defense)
        if one_more_atk is not None:
            combat += max(0, current_info["damage"] - one_more_atk) * 0.01
        one_more_def = damage_with_stats(sim, state, enemy_id, state.atk, state.defense + 1)
        if one_more_def is not None:
            combat += max(0, current_info["damage"] - one_more_def) * 0.008

        child = state.clone()
        if sim.battle(child, x, y):
            after_reachable = sim.reachable_cells(child)
            new_cells = set(after_reachable) - set(reachable)
            value = sim.data.enemys[enemy_id].get("money", 0) * 0.03
            for ix, iy in new_cells:
                tile = sim.tile(child, ix, iy)
                if sim.is_item_tile(tile):
                    value += item_value(sim, child, sim.block_id(tile) or "")
            lookahead += min(value, 30.0)

    return combat, lookahead


def critical_threshold_potential(sim: MotaSimulator, state: MotaState) -> float:
    """Reward current attack/defense when they cross deterministic damage cliffs."""
    return _critical_threshold_potential_for_stats(
        sim,
        state,
        atk=state.atk,
        defense=state.defense,
    )


def _critical_threshold_potential_for_stats(
    sim: MotaSimulator,
    state: MotaState,
    *,
    atk: int,
    defense: int,
) -> float:
    cache = getattr(sim, "_reward_scalar_cache", None)
    if cache is None:
        cache = {}
        setattr(sim, "_reward_scalar_cache", cache)
    cache_key = ("critical_threshold", int(atk), int(defense), int(state.mdef), int(state.hp))
    if cache_key in cache:
        return float(cache[cache_key])

    targets = [
        ("skeletonCaptain", 3.0),
        ("skeletonSoldier", 1.7),
        ("skeleton", 1.0),
        ("bluePriest", 0.8),
        ("bat", 0.45),
        ("redSlime", 0.25),
        ("greenSlime", 0.2),
    ]
    value = 0.0
    for enemy_id, weight in targets:
        current = sim.damage_info_for_stats(enemy_id, atk=atk, defense=defense, mdef=state.mdef)
        if current is None:
            enemy = sim.data.enemys[enemy_id]
            value -= min(30.0, max(0, int(enemy.get("def", 0)) + 1 - int(atk)) * 5.0) * weight
            continue

        damage = current["damage"]
        value += max(0, 700 - min(damage, 700)) * 0.018 * weight
        if enemy_id == "skeletonCaptain":
            value += max(0, 1200 - min(damage, 1200)) * 0.09
            value += max(-100, min(state.hp - damage, 500)) * 0.035

        one_atk = damage_with_stats(sim, state, enemy_id, int(atk) + 1, int(defense))
        if one_atk is not None:
            value += min(150, max(0, damage - one_atk)) * 0.06 * weight
        one_def = damage_with_stats(sim, state, enemy_id, int(atk), int(defense) + 1)
        if one_def is not None:
            value += min(100, max(0, damage - one_def)) * 0.035 * weight
    if len(cache) > 20000:
        cache.clear()
    cache[cache_key] = value
    return value


def stat_delta_critical_value(
    sim: MotaSimulator,
    state: MotaState,
    atk_delta: int = 0,
    def_delta: int = 0,
) -> float:
    before = critical_threshold_potential(sim, state)
    after = _critical_threshold_potential_for_stats(
        sim,
        state,
        atk=state.atk + atk_delta,
        defense=state.defense + def_delta,
    )
    return max(0.0, after - before)


def adjacent_enemy_targets(
    sim: MotaSimulator,
    state: MotaState,
    reachable: dict[tuple[int, int], list[str]],
) -> list[tuple[int, int, str]]:
    targets: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int]] = set()
    for x, y in reachable:
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = x + dx, y + dy
            if (nx, ny) in seen:
                continue
            tile = sim.tile(state, nx, ny)
            if not sim.is_enemy_tile(tile):
                continue
            enemy_id = sim.block_id(tile)
            if enemy_id:
                targets.append((nx, ny, enemy_id))
                seen.add((nx, ny))
    return targets


def damage_with_stats(
    sim: MotaSimulator,
    state: MotaState,
    enemy_id: str,
    atk: int,
    defense: int,
) -> int | None:
    info = sim.damage_info_for_stats(enemy_id, atk=atk, defense=defense, mdef=state.mdef)
    return None if info is None else info["damage"]


def item_value(sim: MotaSimulator, state: MotaState, item_id: str) -> float:
    stage = progress_stage(sim, state)
    if item_id == "yellowKey":
        return 18.0 if stage < 3 else 8.0
    if item_id == "blueKey":
        return 35.0
    if item_id == "redKey":
        return 70.0
    if item_id == "redGem":
        return 18.0 + stat_delta_critical_value(sim, state, atk_delta=1) * 1.2
    if item_id == "blueGem":
        return 18.0 + stat_delta_critical_value(sim, state, def_delta=1) * 1.2
    if item_id == "greenGem":
        return 10.0
    if item_id == "sword1":
        return 120.0
    if item_id == "shield1":
        return 130.0
    if item_id.startswith("sword") or item_id.startswith("shield"):
        return 160.0
    if item_id.endswith("Potion"):
        return 8.0
    return 1.0


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
