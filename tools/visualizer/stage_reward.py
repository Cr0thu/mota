# -*- coding: utf-8 -*-
"""Stage reward and potential functions for the visualizer macro-action env."""

from __future__ import annotations

from dataclasses import dataclass


SWORD_POS = (4, 11, 11)
SHIELD_POS = (8, 7, 9)
RED_KEY_POS = (7, 2, 10)
CAPTAIN_POS = (9, 1, 6)
MT10_RESOURCE_POSITIONS = {
    (9, 6, 2),   # blue jewel
    (9, 6, 10),  # red jewel
    (9, 11, 11), # blue potion
}
STAT_ITEM_IDS = {"redJewel", "blueJewel", "sword1", "shield1"}
JEWEL_IDS = {"redJewel", "blueJewel"}
KEY_ENEMY_IDS = ("skeletonCaptain", "yellowGuard", "skeletonSoldier", "skeleton", "bluePriest", "bat")


@dataclass(frozen=True)
class Potential:
    total: float
    stage: str
    components: dict[str, float]


@dataclass(frozen=True)
class TransitionReward:
    total: float
    before: Potential
    after: Potential
    components: dict[str, float]


def stage_name(env) -> str:
    if not has_sword(env):
        return "sword"
    if not has_shield(env):
        return "shield"
    if remaining_stat_items(env) > 0:
        return "gems"
    if not red_key_taken(env):
        return "red_key"
    if not captain_defeated(env):
        return "boss"
    return "done"


def stage_index(name: str) -> int:
    order = ("sword", "shield", "gems", "red_key", "boss", "done")
    return order.index(name) if name in order else 0


def stage_potential(env) -> Potential:
    p = env.player
    stage = stage_name(env)
    collected_stat = total_stat_items(env) - remaining_stat_items(env)
    boss_damage = enemy_damage_by_id(env, "skeletonCaptain")
    boss_damage = 9999 if boss_damage is None else boss_damage
    boss_margin = p.hp - boss_damage
    marginal_atk = marginal_damage_drop(env, atk_delta=1, def_delta=0)
    marginal_def = marginal_damage_drop(env, atk_delta=0, def_delta=1)

    components = {
        "stage_progress": stage_index(stage) * 900.0,
        "hp_asset": min(max(p.hp, 0), 5000) * 0.04,
        "stat_asset": p.atk * 16.0 + p.def_ * 17.0,
        "key_asset": (
            min(p.items.get("yellowKey", 0), 8) * 25.0
            + min(p.items.get("blueKey", 0), 3) * 70.0
            + min(p.items.get("redKey", 0), 1) * 160.0
        ),
        "stat_items_collected": collected_stat * 150.0,
        "marginal_combat": marginal_atk * 0.75 + marginal_def * 0.75,
        "boss_margin": max(-1500.0, min(float(boss_margin), 2500.0)) * 0.16,
    }

    floor = env.n2p[env.observation[-1]][0]
    if stage == "sword":
        components["target_sword"] = (
            2600.0 if has_sword(env) else _target_floor_score(floor, SWORD_POS[0]) * 125.0
        )
        components["early_safety"] = min(p.hp, 800) * 0.08
    elif stage == "shield":
        components["target_shield"] = (
            3600.0 if has_shield(env) else _target_floor_score(floor, SHIELD_POS[0]) * 220.0
        )
        components["pre_shield_stats"] = collected_stat * 150.0 + p.def_ * 35.0
    elif stage == "gems":
        components["target_gems"] = -remaining_stat_items(env) * 220.0 + collected_stat * 210.0
        components["threshold_pressure"] = critical_progress(env) * 500.0
        components["mt10_resource_probe"] = mt10_resources_taken(env) * 120.0
    elif stage == "red_key":
        components["target_red_key"] = (
            1500.0 if red_key_taken(env) else _target_floor_score(floor, RED_KEY_POS[0]) * 70.0
        )
        components["guard_readiness"] = critical_progress(env) * 450.0 + max(0.0, boss_margin) * 0.12
    elif stage == "boss":
        components["target_boss"] = (
            9000.0 if captain_defeated(env) else _target_floor_score(floor, CAPTAIN_POS[0]) * 90.0
        )
        components["boss_damage_reduction"] = max(0.0, 1800.0 - min(float(boss_damage), 1800.0)) * 0.35
        components["mt10_resources"] = mt10_resources_taken(env) * 260.0
    else:
        components["done"] = 12000.0

    reachable_bonus = reachable_target_bonus(env, stage)
    if reachable_bonus:
        components["reachable_target"] = reachable_bonus

    return Potential(total=sum(components.values()), stage=stage, components=components)


def transition_reward(
    env,
    action,
    before_player_state,
    after_player_state,
    ending: str,
    before: Potential,
    gamma: float = 0.99,
) -> TransitionReward:
    after = stage_potential(env)
    action_id = getattr(action, "id", "")
    action_class = getattr(action, "class_", "")
    action_pos = env.n2p.get(action, (None, None, None))[:3]
    components: dict[str, float] = {
        "env_step": -2.0,
        "pbrs": gamma * after.total - before.total,
    }

    before_hp = float(before_player_state[0])
    after_hp = float(after_player_state[0])
    hp_delta = after_hp - before_hp
    yk_delta = float(after_player_state[6] - before_player_state[6])
    bk_delta = float(after_player_state[7] - before_player_state[7])
    if hp_delta < 0:
        components["hp_loss"] = hp_delta * 0.045
    elif hp_delta > 0:
        components["hp_gain"] = min(hp_delta, 500.0) * 0.06

    if action_class == "items":
        if action_id == "sword1":
            components["item_sword"] = 3000.0
        elif action_id == "shield1":
            components["item_shield"] = 3600.0
        elif action_id in JEWEL_IDS:
            if before.stage == "sword":
                components["item_jewel"] = 80.0 + marginal_bonus_for_item(env, action_id) * 0.35
            elif before.stage == "shield":
                if action_id == "blueJewel":
                    components["item_jewel"] = 520.0 + marginal_bonus_for_item(env, action_id) * 1.15
                else:
                    components["item_jewel"] = 340.0 + marginal_bonus_for_item(env, action_id) * 0.90
            else:
                components["item_jewel"] = 260.0 + marginal_bonus_for_item(env, action_id)
        elif action_id == "redKey":
            components["item_red_key"] = 1500.0
        elif action_id == "blueKey":
            components["item_blue_key"] = 70.0 if before.stage in {"sword", "shield"} else 140.0
        elif action_id == "yellowKey":
            components["item_yellow_key"] = 30.0 if before.stage in {"sword", "shield"} else 60.0
        elif "Potion" in action_id or "potion" in action_id:
            components["item_potion"] = 50.0

    if action_class == "enemies":
        damage = max(0.0, before_hp - after_hp)
        components["fight_cost"] = -damage * (0.12 if before.stage == "shield" else 0.06)
        if before.stage == "sword" and damage > 60:
            components["early_fight_risk"] = -160.0
        if before.stage == "shield" and damage > 70:
            components["pre_shield_fight_risk"] = -220.0
        if before.stage != "sword" and action_id in {"yellowGuard", "skeletonCaptain"}:
            components["key_enemy"] = 220.0
        if action_pos == CAPTAIN_POS and action_id == "skeletonCaptain":
            components["captain_clear"] = 12000.0

    if action_class == "terrains" and "Door" in action_id:
        components["door_cost"] = {
            "yellowDoor": -35.0,
            "blueDoor": -85.0,
            "redDoor": -140.0,
        }.get(action_id, -20.0)
        if before.stage == "sword":
            components["early_door_caution"] = -120.0
        elif before.stage == "shield":
            if action_id == "blueDoor":
                components["shield_blue_route_unlock"] = 1050.0
            elif action_id == "yellowDoor" and action_pos[0] is not None and action_pos[0] >= 6:
                components["shield_yellow_route_unlock"] = 180.0
            else:
                components["pre_shield_door_caution"] = -60.0

    if before.stage in {"sword", "shield"}:
        if yk_delta < 0:
            components["yellow_key_spend"] = yk_delta * (45.0 if before.stage == "shield" else 90.0)
        if bk_delta < 0:
            components["blue_key_spend"] = bk_delta * (70.0 if before.stage == "shield" else 180.0)
        if after_player_state[6] <= 0 and before.stage == "sword":
            components["yellow_key_depleted_before_sword"] = -100.0

    if action_class == "endFlag" or action_id == "end":
        components["clear"] = 15000.0
    if ending in {"death", "stop"}:
        components["bad_terminal"] = -1500.0

    return TransitionReward(total=sum(components.values()), before=before, after=after, components=components)


def stage_action_priors(env, actions: list) -> list[float]:
    """Return stage-aware action priors for the visualizer action table."""

    if not actions:
        return []
    priors: list[float] = []
    for action in actions:
        before = stage_potential(env)
        before_state = env.get_player_state().copy()
        ending = "continue"
        stepped = False
        try:
            ending = env.step(action, return_reward=False)
            stepped = True
            after_state = env.get_player_state().copy()
            reward = transition_reward(env, action, before_state, after_state, ending, before)
            direct = _direct_action_prior(env, action, reward)
            score = reward.total / 90.0 + direct
        except Exception:
            score = -8.0
        finally:
            if stepped:
                try:
                    env.back_step(1)
                except Exception:
                    pass
        priors.append(float(max(-10.0, min(10.0, score))))
    return priors


def _direct_action_prior(env, action, reward: TransitionReward) -> float:
    action_id = getattr(action, "id", "")
    action_class = getattr(action, "class_", "")
    action_pos = env.n2p.get(action, (None, None, None))[:3]
    stage = reward.before.stage
    direct = 0.0

    if reward.after.stage != reward.before.stage:
        direct += 8.0
    if stage == "sword":
        if action_pos == SWORD_POS:
            direct += 10.0
        elif action_class == "items" and action_id in {"yellowKey", "blueKey", "redJewel", "blueJewel"}:
            direct += 2.0
        elif action_class == "enemies":
            direct -= 1.5
    elif stage == "shield":
        if action_pos == SHIELD_POS:
            direct += 10.0
        elif action_id == "blueJewel":
            direct += 5.5
        elif action_id == "redJewel":
            direct += 4.0
        elif action_id == "blueDoor":
            direct += 4.0
        elif action_id == "yellowDoor" and action_pos[0] is not None and action_pos[0] >= 6:
            direct += 1.0
    elif stage == "gems":
        if action_id in JEWEL_IDS:
            direct += 7.0
        elif action_id in {"sword1", "shield1"}:
            direct += 5.0
    elif stage == "red_key":
        if action_pos == RED_KEY_POS:
            direct += 10.0
        elif action_id in JEWEL_IDS:
            direct += 4.0
    elif stage == "boss":
        if action_pos == CAPTAIN_POS:
            direct += 10.0
        elif action_pos in MT10_RESOURCE_POSITIONS:
            direct += 6.0
        elif action_id in JEWEL_IDS:
            direct += 4.0

    if action_class == "terrains" and "Door" in action_id:
        direct -= 0.8
    if action_class == "enemies":
        before_hp = reward.components.get("hp_loss", 0.0)
        if before_hp < -20.0:
            direct -= 2.0
    return direct


def total_stat_items(env) -> int:
    return sum(1 for node in env.n2p if getattr(node, "id", "") in STAT_ITEM_IDS)


def remaining_stat_items(env) -> int:
    return sum(
        1
        for node in env.n2p
        if getattr(node, "id", "") in STAT_ITEM_IDS and not getattr(node, "activated", False)
    )


def has_sword(env) -> bool:
    return position_consumed(env, SWORD_POS, "sword1")


def has_shield(env) -> bool:
    return position_consumed(env, SHIELD_POS, "shield1")


def red_key_taken(env) -> bool:
    return env.player.items.get("redKey", 0) > 0 or position_consumed(env, RED_KEY_POS, "redKey")


def captain_defeated(env) -> bool:
    node = env.p2n.get(CAPTAIN_POS)
    return node is None or getattr(node, "activated", False) or getattr(node, "id", "") != "skeletonCaptain"


def mt10_resources_taken(env) -> int:
    return sum(1 for pos in MT10_RESOURCE_POSITIONS if position_consumed(env, pos))


def position_consumed(env, pos: tuple[int, int, int], expected_id: str | None = None) -> bool:
    node = env.p2n.get(pos)
    if node is None:
        return True
    if expected_id is not None and getattr(node, "id", "") != expected_id:
        return True
    return bool(getattr(node, "activated", False))


def reachable_target_bonus(env, stage: str) -> float:
    try:
        actions = env.get_feasible_actions()
    except Exception:
        return 0.0
    bonus = 0.0
    for action in actions:
        pos = env.n2p.get(action, (None, None, None))[:3]
        action_id = getattr(action, "id", "")
        if stage == "sword" and pos == SWORD_POS:
            bonus += 350.0
        elif stage == "shield" and pos == SHIELD_POS:
            bonus += 380.0
        elif stage == "red_key" and pos == RED_KEY_POS:
            bonus += 420.0
        elif stage == "boss" and pos == CAPTAIN_POS:
            bonus += 700.0
        elif stage == "gems" and action_id in STAT_ITEM_IDS:
            bonus += 90.0
    return bonus


def critical_progress(env) -> float:
    atk_progress = min(1.0, max(0.0, (env.player.atk - 10.0) / 14.0))
    def_progress = min(1.0, max(0.0, (env.player.def_ - 10.0) / 12.0))
    return (atk_progress + def_progress) / 2.0


def marginal_bonus_for_item(env, item_id: str) -> float:
    if item_id == "redJewel":
        return min(400.0, marginal_damage_drop(env, atk_delta=1, def_delta=0) * 1.3)
    if item_id == "blueJewel":
        return min(400.0, marginal_damage_drop(env, atk_delta=0, def_delta=1) * 1.3)
    return 0.0


def marginal_damage_drop(env, atk_delta: int, def_delta: int) -> float:
    total = 0.0
    for enemy_id in KEY_ENEMY_IDS:
        current = enemy_damage_by_id(env, enemy_id)
        improved = enemy_damage_by_id(env, enemy_id, atk_delta=atk_delta, def_delta=def_delta)
        if current is not None and improved is not None:
            total += max(0.0, float(current - improved))
    return total


def enemy_damage_by_id(env, enemy_id: str, atk_delta: int = 0, def_delta: int = 0) -> int | None:
    data = env.env_data.get("enemies", {}).get(enemy_id)
    if data is None:
        return None
    return enemy_damage(
        hp=data["hp"],
        atk=data["atk"],
        defense=data["def"],
        special=data.get("special", 0),
        special_kw=data,
        player_hp=env.player.hp,
        player_atk=env.player.atk + atk_delta,
        player_def=env.player.def_ + def_delta,
        player_mdef=env.player.mdef,
    )


def enemy_damage(
    hp: int,
    atk: int,
    defense: int,
    special: int,
    special_kw: dict,
    player_hp: int,
    player_atk: int,
    player_def: int,
    player_mdef: int,
) -> int | None:
    player_damage = player_atk - defense
    if player_damage <= 0:
        return None
    rounds = hp // player_damage - (hp % player_damage == 0)
    enemy_damage_per_round = max(atk - player_def, 0)
    damage = enemy_damage_per_round * rounds
    if special == 1:
        damage += enemy_damage_per_round
    elif special == 11:
        damage += player_hp // special_kw["value"]
    elif special == 22:
        damage += special_kw["damage"]
    damage -= player_mdef
    return max(0, min(int(damage), player_hp))


def _target_floor_score(current_floor: int, target_floor: int) -> float:
    return max(0.0, 10.0 - abs(float(current_floor - target_floor)))
