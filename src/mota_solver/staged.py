from __future__ import annotations

import heapq
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from mota_env import MotaSimulator, MotaState, build_graph_state
from mota_env.rewards import (
    DEFAULT_STAGE_POTENTIAL_WEIGHTS,
    STAGE_LABELS,
    STAGE_ORDER,
    GEM_STAGE_TARGETS,
    all_attack_defense_gems_taken,
    boss_ready,
    boss_route_margin,
    boss_route_required_damage,
    critical_gem_progress,
    critical_gems_ready,
    current_stage_name,
    damage_drop_for_stats,
    floor_index,
    guard_ready,
    has_first_sword,
    lower_attack_defense_gems_taken,
    mt8_blue_key_taken,
    mt10_access_ready,
    mt10_blue_ready,
    mt9_shield_taken,
    mt10_resource_progress,
    MT10_RESOURCE_YELLOW_KEY_TARGET,
    mt10_resources_taken,
    pre_shield_gems_ready,
    pre_mt10_stat_progress,
    pre_mt10_stats_ready,
    remaining_attack_defense_gems,
    red_key_route_damage,
    red_key_route_margin,
    red_key_taken,
    remaining_lower_attack_defense_gems,
    remaining_stage_gem_targets,
    stage_complete,
    stage_gem_targets_taken,
    stage_potential,
    tile_id,
    total_attack_defense_gems,
    total_lower_attack_defense_gems,
    yellow_guard_damage,
)
from mota_solver.search import SearchNode, action_bias, reconstruct_route, state_summary


_GUARD_TARGET_ATK = 26
_GUARD_TARGET_DEF = 25


@dataclass
class StageSummary:
    stage: str
    label: str
    solved: bool
    expansions: int
    frontier_size: int
    best: dict[str, Any]


@dataclass
class StagedSearchResult:
    solved: bool
    state: MotaState
    route: list[dict[str, Any]]
    expansions: int
    stage_summaries: list[StageSummary] = field(default_factory=list)


def solve_staged_first10(
    sim: MotaSimulator,
    max_expansions_per_stage: int = 20_000,
    keep_per_parent: int = 80,
    frontier_size: int = 32,
    seed: int = 20260513,
    trace_path: str | Path | None = None,
    trace_limit: int = 50_000,
    stage_weights: dict[str, float] | None = None,
    stop_stage: str | None = None,
) -> StagedSearchResult:
    """Explore the first-10-floor task stage by stage without expert routes."""

    rng = random.Random(seed)
    start = sim.reset()
    frontier = [SearchNode(start)]
    total_expansions = 0
    summaries: list[StageSummary] = []
    best_node = frontier[0]
    trace_handle: TextIO | None = None
    trace_count = 0
    if trace_path is not None:
        path = Path(trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        trace_handle = path.open("w", encoding="utf8")

    try:
        if stop_stage is not None and stop_stage not in STAGE_ORDER:
            raise ValueError(f"Unknown stop_stage {stop_stage!r}; choose one of {STAGE_ORDER}")
        stage_sequence = []
        for stage_name in STAGE_ORDER:
            stage_sequence.append(stage_name)
            if stage_name == stop_stage:
                break

        for stage_index, stage in enumerate(stage_sequence):
            stage_result = _expand_stage(
                sim=sim,
                starts=frontier,
                stage=stage,
                rng=rng,
                max_expansions=max_expansions_per_stage,
                keep_per_parent=keep_per_parent,
                frontier_size=frontier_size,
                trace_handle=trace_handle,
                trace_limit=trace_limit,
                trace_count=trace_count,
                stage_weights=stage_weights,
            )
            trace_count += stage_result["trace_count"]
            total_expansions += stage_result["expansions"]
            best_node = stage_result["best_node"]
            success_nodes = stage_result["success_nodes"]
            summaries.append(
                StageSummary(
                    stage=stage,
                    label=STAGE_LABELS[stage],
                    solved=bool(success_nodes),
                    expansions=stage_result["expansions"],
                    frontier_size=len(success_nodes),
                    best=state_summary(best_node.state),
                )
                )
            if not success_nodes:
                return StagedSearchResult(
                    solved=False,
                    state=best_node.state,
                    route=reconstruct_route(best_node),
                    expansions=total_expansions,
                    stage_summaries=summaries,
                )
            frontier_stage = stage if stage_index == len(stage_sequence) - 1 else _next_stage(stage)
            frontier = _select_frontier(
                sim,
                success_nodes,
                frontier_stage,
                frontier_size,
                stage_weights,
            )
            best_node = frontier[0]

        if stop_stage is not None:
            best_node = max(
                frontier,
                key=lambda node: _node_score(sim, node.state, stop_stage, stage_weights, 0.0),
            )
            return StagedSearchResult(
                solved=stage_complete(sim, best_node.state, stop_stage) and best_node.state.hp > 0,
                state=best_node.state,
                route=reconstruct_route(best_node),
                expansions=total_expansions,
                stage_summaries=summaries,
            )

        best_node = max(
            frontier,
            key=lambda node: _node_score(sim, node.state, "boss", stage_weights, 0.0),
        )
        return StagedSearchResult(
            solved=bool(best_node.state.flags.get("10f战胜骷髅队长")) and best_node.state.hp > 0,
            state=best_node.state,
            route=reconstruct_route(best_node),
            expansions=total_expansions,
            stage_summaries=summaries,
        )
    finally:
        if trace_handle is not None:
            trace_handle.close()


def _expand_stage(
    sim: MotaSimulator,
    starts: list[SearchNode],
    stage: str,
    rng: random.Random,
    max_expansions: int,
    keep_per_parent: int,
    frontier_size: int,
    trace_handle: TextIO | None,
    trace_limit: int,
    trace_count: int,
    stage_weights: dict[str, float] | None,
) -> dict[str, Any]:
    queue: list[tuple[float, int, SearchNode]] = []
    counter = 0
    novelty: set[tuple[Any, ...]] = set()
    seen: set[tuple[Any, ...]] = set()
    dominance: dict[tuple[Any, ...], list[tuple[int, ...]]] = {}
    for node in starts:
        counter += 1
        score = _node_score(sim, node.state, stage, stage_weights, _gumbel(rng))
        heapq.heappush(queue, (-score, counter, node))
        seen.add(sim.state_key(node.state))
        dominance.setdefault(sim.dominance_key(node.state), []).append(
            _stage_resource_vector(sim, node.state)
        )

    successes: list[SearchNode] = []
    best_node = starts[0]
    best_score = -math.inf
    expansions = 0
    local_trace_count = 0

    while queue and expansions < max_expansions:
        _, _, node = heapq.heappop(queue)
        state = node.state
        expansions += 1
        score = _node_score(sim, state, stage, stage_weights, 0.0)
        if score > best_score:
            best_score = score
            best_node = node
        if stage_complete(sim, state, stage):
            successes.append(node)
            if len(successes) >= max(frontier_size * 4, 16):
                break
            continue

        novelty_key = _novelty_key(state, stage)
        is_novel = novelty_key not in novelty
        novelty.add(novelty_key)
        actions = _filter_stage_actions(sim, state, stage, sim.macro_actions(state))
        actions.sort(key=lambda action: _stage_action_bias(action, stage, state, sim), reverse=True)
        if trace_handle is not None and trace_count + local_trace_count < trace_limit:
            _write_trace_row(trace_handle, sim, state, stage, actions, expansions)
            local_trace_count += 1

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
            vec = _stage_resource_vector(sim, child)
            existing = dominance.get(dkey, [])
            if any(all(old_v >= new_v for old_v, new_v in zip(old, vec)) for old in existing):
                continue
            dominance[dkey] = [
                old for old in existing if not all(new_v >= old_v for new_v, old_v in zip(vec, old))
            ] + [vec]
            seen.add(key)
            step = {
                "index": node.depth,
                "action": action,
                "stage": stage,
                "before": before,
                "after": state_summary(child),
                "reward": transition.reward,
            }
            child_node = SearchNode(child, parent=node, step=step, depth=node.depth + 1)
            counter += 1
            gumbel = _gumbel(rng)
            novelty_bonus = 35.0 if is_novel else 0.0
            priority = -(
                _node_score(sim, child, stage, stage_weights, gumbel)
                + _stage_action_bias(action, stage, state, sim) * 0.08
                + novelty_bonus
            )
            heapq.heappush(queue, (priority, counter, child_node))

    return {
        "success_nodes": _select_frontier(
            sim,
            successes,
            stage,
            max(frontier_size * 4, 32),
            stage_weights,
        ),
        "best_node": best_node,
        "expansions": expansions,
        "trace_count": local_trace_count,
    }


def _select_frontier(
    sim: MotaSimulator,
    nodes: list[SearchNode],
    stage: str,
    limit: int,
    stage_weights: dict[str, float] | None,
) -> list[SearchNode]:
    selected: list[SearchNode] = []
    dominance: dict[tuple[Any, ...], list[tuple[int, ...]]] = {}
    ordered = sorted(
        nodes,
        key=lambda node: _node_score(sim, node.state, stage, stage_weights, 0.0),
        reverse=True,
    )
    for node in ordered:
        dkey = sim.dominance_key(node.state)
        vec = _stage_resource_vector(sim, node.state)
        existing = dominance.get(dkey, [])
        if any(all(old_v >= new_v for old_v, new_v in zip(old, vec)) for old in existing):
            continue
        dominance[dkey] = [
            old for old in existing if not all(new_v >= old_v for new_v, old_v in zip(vec, old))
        ] + [vec]
        selected.append(node)
        if len(selected) >= limit:
            break
    return selected or ordered[:limit]


def _next_stage(stage: str) -> str:
    try:
        index = STAGE_ORDER.index(stage)
    except ValueError:
        return stage
    if index + 1 >= len(STAGE_ORDER):
        return stage
    return STAGE_ORDER[index + 1]


def _node_score(
    sim: MotaSimulator,
    state: MotaState,
    stage: str,
    stage_weights: dict[str, float] | None,
    gumbel: float,
) -> float:
    score_stage = _effective_score_stage(sim, state, stage)
    weights = {**DEFAULT_STAGE_POTENTIAL_WEIGHTS, **(stage_weights or {})}
    value = sum(
        component * weights.get(name, 1.0)
        for name, component in _fast_stage_components(sim, state, score_stage).items()
    )
    value += _boss_margin(sim, state) * 0.02
    value += int(stage_complete(sim, state, stage)) * 400.0
    value += -state.steps * 0.03
    value -= max(0, -state.hp) * 12.0
    value += gumbel * 18.0
    return value


def _effective_score_stage(sim: MotaSimulator, state: MotaState, stage: str) -> str:
    """Map coarse research milestones to their current hidden subgoal."""

    if stage in {
        "pre_shield_gems",
        "shield",
        "lower_gems",
        "mt8_gems",
        "mid_gems",
        "low_gems",
        "mt8_hp_ready",
        "mt10_blue_ready",
        "mt10_yellow_ready",
        "mt10_ready",
        "mt10_resources",
        "all_gems",
        "red_key",
        "boss_ready",
        "trap",
        "boss",
    } and not has_first_sword(sim, state):
        return "sword"
    pre_shield_stats_ready = (
        has_first_sword(sim, state)
        and state.atk >= 21
        and state.defense >= 11
    )
    if stage == "shield" and not pre_shield_stats_ready:
        return "pre_shield_gems"
    if stage in {"lower_gems", "mt8_gems", "mid_gems", "low_gems", "mt10_blue_ready", "mt10_yellow_ready", "mt10_ready", "mt10_resources", "all_gems", "red_key", "boss_ready", "trap", "boss"} and not mt9_shield_taken(sim, state):
        return "shield"
    if stage in {"low_gems", "mt8_hp_ready", "mt8_gems", "mt10_blue_ready", "mt10_yellow_ready", "mt10_ready", "mt10_resources", "all_gems", "red_key", "boss_ready", "trap", "boss"} and not stage_gem_targets_taken(sim, state, "mid_gems"):
        return "mid_gems"
    if stage in {"mt8_hp_ready", "mt8_gems", "mt10_blue_ready", "mt10_yellow_ready", "mt10_ready", "mt10_resources", "all_gems", "red_key", "boss_ready", "trap", "boss"} and not stage_gem_targets_taken(sim, state, "low_gems"):
        return "low_gems"
    if (
        stage in {"mt8_gems", "mt10_blue_ready", "mt10_yellow_ready", "mt10_ready", "mt10_resources", "all_gems", "red_key", "boss_ready", "trap", "boss"}
        and floor_index(sim, state) < 8
        and not stage_complete(sim, state, "mt8_hp_ready")
        and not stage_gem_targets_taken(sim, state, "mt8_gems")
    ):
        return "mt8_hp_ready"
    if stage in {"mt10_blue_ready", "mt10_yellow_ready", "mt10_ready", "mt10_resources", "all_gems", "red_key", "boss_ready", "trap", "boss"} and not stage_gem_targets_taken(sim, state, "mt8_gems"):
        return "mt8_gems"
    if stage in {"mt10_blue_ready", "mt10_yellow_ready", "mt10_ready", "mt10_resources", "all_gems", "red_key", "boss_ready", "trap", "boss"} and not lower_attack_defense_gems_taken(sim, state):
        return "lower_gems"
    if stage in {"mt10_yellow_ready", "mt10_ready", "mt10_resources", "all_gems", "red_key", "boss_ready", "trap", "boss"} and not mt10_blue_ready(sim, state):
        return "mt10_blue_ready"
    if stage in {"mt10_resources", "all_gems", "red_key", "boss_ready", "trap", "boss"} and not mt10_access_ready(sim, state):
        return "mt10_yellow_ready"
    if stage in {"all_gems", "red_key", "boss_ready", "trap", "boss"} and not mt10_resources_taken(sim, state):
        return "mt10_resources"
    if stage in {"red_key", "boss_ready", "trap", "boss"} and not all_attack_defense_gems_taken(sim, state):
        return "all_gems"
    if stage == "red_key" and not _guard_stats_ready(sim, state):
        return "guard_stats_ready"
    if stage == "red_key" and not _guard_route_ready(sim, state):
        return "guard_route_ready"
    if stage in {"boss_ready", "trap", "boss"} and not red_key_taken(sim, state):
        return "red_key"
    if stage in {"trap", "boss"} and not boss_ready(sim, state):
        return "boss_ready"
    if stage == "boss" and not state.flags.get("10f机关"):
        return "trap"
    return stage


def _fast_stage_components(
    sim: MotaSimulator,
    state: MotaState,
    stage: str,
) -> dict[str, float]:
    floor_idx = floor_index(sim, state)
    yk = state.items.get("yellowKey", 0)
    bk = state.items.get("blueKey", 0)
    rk = state.items.get("redKey", 0)
    boss_margin = _boss_margin(sim, state)
    boss_damage = _boss_damage(sim, state)
    boss_atk_drop = damage_drop_for_stats(sim, state, "skeletonCaptain", atk_delta=1)
    boss_def_drop = damage_drop_for_stats(sim, state, "skeletonCaptain", def_delta=1)
    guard_atk_drop = damage_drop_for_stats(sim, state, "yellowGuard", atk_delta=1)
    guard_def_drop = damage_drop_for_stats(sim, state, "yellowGuard", def_delta=1)
    remaining_gems = remaining_attack_defense_gems(sim, state)
    total_gems = max(1, total_attack_defense_gems(sim))
    remaining_lower_gems = remaining_lower_attack_defense_gems(sim, state)
    total_lower_gems = max(1, total_lower_attack_defense_gems(sim))
    components = {
        "asset": state.hp * 0.18 + state.atk * 70.0 + state.defense * 65.0 + state.money * 0.4,
        "combat": max(-600.0, boss_margin) * 0.1,
        "threshold": max(0, 1200 - min(boss_damage, 1200)) * 1.2 + boss_atk_drop * 4.0 + boss_def_drop * 3.0,
        "lookahead": 0.0,
        "progress": 0.0,
        "deadend": 0.0,
        "stage_sword": 0.0,
        "stage_pre_shield_gems": 0.0,
        "stage_shield": 0.0,
        "stage_gems": 0.0,
        "stage_lower_gems": 0.0,
        "stage_mt8_gems": 0.0,
        "stage_mid_gems": 0.0,
        "stage_low_gems": 0.0,
        "stage_mt8_hp_ready": 0.0,
        "stage_all_gems": 0.0,
        "stage_mt10_blue_ready": 0.0,
        "stage_mt10_yellow_ready": 0.0,
        "stage_mt10_ready": 0.0,
        "stage_mt10_resources": 0.0,
        "stage_guard_stats_ready": 0.0,
        "stage_guard_route_ready": 0.0,
        "stage_guard_ready": 0.0,
        "stage_red_key": 0.0,
        "stage_boss_ready": 0.0,
        "stage_trap": 0.0,
        "stage_boss": 0.0,
        "boss_margin": boss_margin * 0.35,
        "key_pressure": yk * 42.0 + bk * 130.0 + rk * 260.0,
        "global_resource": -remaining_gems * 25.0,
    }
    if state.hp < 0:
        components["deadend"] -= abs(state.hp) * 15.0
    if floor_idx >= 7 and yk == 0 and not mt8_blue_key_taken(sim, state):
        components["deadend"] -= 1500.0
    if state.hp < 180 and stage in {
        "pre_shield_gems",
        "shield",
        "lower_gems",
        "mt8_gems",
        "mid_gems",
        "low_gems",
        "mt8_hp_ready",
        "all_gems",
        "mt10_blue_ready",
        "mt10_yellow_ready",
        "mt10_ready",
        "mt10_resources",
        "gems",
        "red_key",
        "boss_ready",
        "trap",
        "boss",
    }:
        components["deadend"] -= (180 - state.hp) * 8.0
    if stage == "mt10_resources" and state.hp < 120:
        components["deadend"] -= (120 - state.hp) * 20.0
    if stage == "sword":
        components["stage_sword"] = (
            (2500.0 if has_first_sword(sim, state) else 0.0)
            + _target_floor_score(floor_idx, 5, 120.0)
        )
    elif stage == "pre_shield_gems":
        gem_progress = _gems_landmark_progress(sim, state)
        target_floor = 3 if has_first_sword(sim, state) and not pre_shield_gems_ready(state) else 5
        components["stage_pre_shield_gems"] = (
            (2800.0 if pre_shield_gems_ready(state) else 0.0)
            + (1800.0 if has_first_sword(sim, state) else 0.0)
            + min(gem_progress, 3) * 820.0
            + max(0, state.atk - 20) * 480.0
            + max(0, state.defense - 10) * 450.0
            + _target_floor_score(floor_idx, target_floor, 90.0)
            + min(state.hp, 800) * 0.9
            + boss_atk_drop * 2.0
            + boss_def_drop * 1.6
        )
        components["key_pressure"] = yk * 240.0 + bk * 260.0
    elif stage == "shield":
        progress = _shield_landmark_progress(sim, state)
        gem_progress = _gems_landmark_progress(sim, state)
        sword_bonus = 1800.0 if has_first_sword(sim, state) else 0.0
        components["stage_shield"] = (
            (3200.0 if mt9_shield_taken(sim, state) else 0.0)
            + (900.0 if mt8_blue_key_taken(sim, state) else 0.0)
            + sword_bonus
            + progress * 700.0
            + min(gem_progress, 5) * 220.0
            + critical_gem_progress(state) * 140.0
            + _target_floor_score(floor_idx, 9, 360.0)
            + state.atk * 95.0
            + state.defense * 80.0
            + min(state.hp, 900) * 0.8
            + boss_atk_drop * 2.0
            + boss_def_drop * 1.4
        )
        components["key_pressure"] = _shield_key_pressure(sim, state)
        if not has_first_sword(sim, state):
            components["deadend"] -= max(0, floor_idx - 4) * 450.0
        if floor_idx >= 7 and state.items.get("blueKey", 0) == 0 and not _mt9_blue_door_opened(sim, state):
            components["deadend"] -= 1800.0
        if state.items.get("blueKey", 0) == 0 and not mt8_blue_key_taken(sim, state):
            components["deadend"] -= 2200.0
        if mt9_shield_taken(sim, state) and state.hp < 80:
            components["deadend"] -= (80 - state.hp) * 140.0
    elif stage == "gems":
        gem_progress = _gems_landmark_progress(sim, state)
        pre_progress = pre_mt10_stat_progress(state)
        components["stage_gems"] = (
            (3600.0 if pre_mt10_stats_ready(state) else 0.0)
            + pre_progress * 1150.0
            + critical_gem_progress(state) * 180.0
            + gem_progress * 720.0
            + max(0, 1200 - min(boss_damage, 1200)) * 1.8
            + damage_drop_for_stats(sim, state, "skeletonCaptain", atk_delta=1) * 5.0
            + damage_drop_for_stats(sim, state, "skeletonCaptain", def_delta=1) * 4.0
            + min(state.hp, 700) * 0.8
            - remaining_gems * 35.0
        )
        components["key_pressure"] = (
            min(5, yk) * 900.0
            + bk * 520.0
            - max(0, 5 - yk) * 1400.0
            - max(0, 1 - bk) * 1200.0
        )
    elif stage == "mt8_hp_ready":
        hp_target = 155
        components["lookahead"] = 0.0
        target_yk = MT10_RESOURCE_YELLOW_KEY_TARGET
        components["progress"] = min(state.hp, 360) * 4.0 + min(target_yk, yk) * 420.0
        components["stage_mt8_hp_ready"] = (
            (6200.0 if stage_complete(sim, state, "mt8_hp_ready") else 0.0)
            + min(state.hp, 420) * 42.0
            + min(target_yk, yk) * 1250.0
            + bk * 720.0
            + state.atk * 260.0
            + state.defense * 260.0
            - max(0, hp_target - state.hp) * 210.0
            - max(0, target_yk - yk) * 1500.0
        )
        components["key_pressure"] = (
            min(target_yk, yk) * 1100.0
            + bk * 620.0
            - max(0, target_yk - yk) * 1300.0
        )
        if floor_idx > 4:
            components["progress"] -= (floor_idx - 4) * 520.0
        if state.hp < hp_target:
            components["deadend"] -= (hp_target - state.hp) * 120.0
    elif stage in {"mt8_gems", "mid_gems", "low_gems"}:
        target_remaining = remaining_stage_gem_targets(sim, state, stage)
        target_total = max(1, len(GEM_STAGE_TARGETS.get(stage, ())))
        target_collected = target_total - target_remaining
        target_floor = _stage_target_floor(sim, state, stage)
        components["lookahead"] = 0.0
        components["progress"] = target_collected * 900.0 - abs(floor_idx - target_floor) * 120.0
        components[f"stage_{stage}"] = (
            (9200.0 if stage_gem_targets_taken(sim, state, stage) else 0.0)
            + target_collected * 9000.0
            + _stage_gem_blocker_progress(sim, state, stage) * 9000.0
            + critical_gem_progress(state) * 1200.0
            + state.atk * 340.0
            + state.defense * 340.0
            + min(state.hp, 1600) * 2.3
            + boss_atk_drop * 8.0
            + boss_def_drop * 7.0
            - target_remaining * 9000.0
        )
        components["key_pressure"] = min(5, yk) * 650.0 + bk * 800.0 - max(0, 2 - yk) * 1800.0
        if stage == "mt8_gems" and floor_idx < 8:
            components["progress"] -= (8 - floor_idx) * 360.0
        if target_remaining > 0 and floor_idx != target_floor:
            components["deadend"] -= abs(floor_idx - target_floor) * 4500.0
        if stage == "mid_gems" and floor_idx > 6:
            components["progress"] -= (floor_idx - 6) * 360.0
        if stage == "low_gems" and floor_idx > 3:
            components["progress"] -= (floor_idx - 3) * 420.0
    elif stage == "lower_gems":
        gem_progress = _gems_landmark_progress(sim, state)
        collected_lower = total_lower_gems - remaining_lower_gems
        components["lookahead"] = 0.0
        components["progress"] = collected_lower * 520.0
        components["stage_lower_gems"] = (
            (8500.0 if lower_attack_defense_gems_taken(sim, state) else 0.0)
            + collected_lower * 1850.0
            + gem_progress * 980.0
            + critical_gem_progress(state) * 1280.0
            + max(0, 1500 - min(boss_damage, 1500)) * 2.2
            + boss_atk_drop * 8.0
            + boss_def_drop * 7.0
            + guard_atk_drop * 8.0
            + guard_def_drop * 7.0
            + state.atk * 360.0
            + state.defense * 350.0
            + min(state.hp, 1500) * 2.0
            - remaining_lower_gems * 950.0
        )
        components["key_pressure"] = (
            min(4, yk) * 620.0
            + bk * 760.0
            - max(0, 2 - yk) * 1600.0
        )
        if state.floor_id == "MT10":
            components["deadend"] -= 2200.0
        if floor_idx >= 9 and remaining_lower_gems > 0:
            components["deadend"] -= 2600.0
    elif stage == "all_gems":
        gem_progress = _gems_landmark_progress(sim, state)
        mt10_progress = mt10_resource_progress(sim, state)
        remaining_penalty = remaining_gems * 900.0
        components["lookahead"] = 0.0
        components["progress"] = mt10_progress * 420.0
        components["stage_all_gems"] = (
            (11000.0 if all_attack_defense_gems_taken(sim, state) else 0.0)
            + gem_progress * 950.0
            + (total_gems - remaining_gems) * 1500.0
            + mt10_progress * 3200.0
            + critical_gem_progress(state) * 1450.0
            + max(0, 1700 - min(boss_damage, 1700)) * 3.2
            + boss_atk_drop * 12.0
            + boss_def_drop * 10.0
            + guard_atk_drop * 9.0
            + guard_def_drop * 8.0
            + state.atk * 420.0
            + state.defense * 410.0
            + min(state.hp, 1800) * 2.2
            - remaining_penalty
        )
        components["stage_gems"] = (
            critical_gem_progress(state) * 900.0
            + max(0, 1600 - min(boss_damage, 1600)) * 2.0
            + boss_atk_drop * 7.0
            + boss_def_drop * 6.0
        )
        components["key_pressure"] = (
            min(MT10_RESOURCE_YELLOW_KEY_TARGET, yk) * 760.0
            + bk * 820.0
            - max(0, 2 - yk) * 1800.0
        )
        if mt9_shield_taken(sim, state) and not mt10_access_ready(sim, state):
            components["deadend"] -= max(0, MT10_RESOURCE_YELLOW_KEY_TARGET - yk) * 2400.0
            components["deadend"] -= max(0, 1 - bk) * 2600.0
    elif stage == "mt10_blue_ready":
        needed_blue = 0 if _mt9_mt10_blue_door_opened(sim, state) else 1
        components["lookahead"] = 0.0
        components["progress"] = _target_floor_score(floor_idx, 9, 120.0)
        components["stage_mt10_blue_ready"] = (
            (6200.0 if mt10_blue_ready(sim, state) else 0.0)
            + pre_mt10_stat_progress(state) * 1150.0
            + _target_floor_score(floor_idx, 9, 190.0)
            + min(state.hp, 420) * 12.0
            + min(max(needed_blue, 1), bk) * 1800.0
            - max(0, needed_blue - bk) * 7600.0
        )
        components["key_pressure"] = min(max(needed_blue, 1), bk) * 980.0 - max(0, needed_blue - bk) * 6200.0
    elif stage in {"mt10_yellow_ready", "mt10_ready"}:
        needed_blue = 0 if _mt9_mt10_blue_door_opened(sim, state) else 1
        target_yk = MT10_RESOURCE_YELLOW_KEY_TARGET
        blue_key_progress = _mt8_blue_key_route_progress(sim, state)
        yellow_budget_progress = _mt7_yellow_budget_progress(sim, state)
        yellow_refill_damage = _enemy_damage(sim, state, "skeletonSoldier")
        components["lookahead"] = 0.0
        components["progress"] = _target_floor_score(floor_idx, 9, 120.0)
        components["stage_mt10_ready"] = (
            (7000.0 if mt10_access_ready(sim, state) else 0.0)
            + pre_mt10_stat_progress(state) * 1150.0
            + blue_key_progress * 950.0
            + yellow_budget_progress * 900.0
            + _target_floor_score(floor_idx, 9, 220.0)
            + min(state.hp, 420) * 18.0
            + min(target_yk, yk) * 900.0
            + min(max(needed_blue, 1), bk) * 900.0
            - max(0, 235 - state.hp) * 130.0
            - max(0, target_yk - yk) * 5600.0
            - max(0, needed_blue - bk) * 6400.0
        )
        if yk < target_yk:
            components["stage_mt10_ready"] += max(0, state.hp - yellow_refill_damage) * 38.0
            components["deadend"] -= max(0, yellow_refill_damage + 8 - state.hp) * 170.0
            components["threshold"] += damage_drop_for_stats(sim, state, "skeletonSoldier", atk_delta=1) * 18.0
        components["stage_mt10_yellow_ready"] = components["stage_mt10_ready"]
        components["key_pressure"] = (
            min(target_yk, yk) * 680.0
            + min(max(needed_blue, 1), bk) * 860.0
            - max(0, target_yk - yk) * 4300.0
            - max(0, needed_blue - bk) * 5400.0
        )
    elif stage == "mt10_resources":
        mt10_progress = mt10_resource_progress(sim, state)
        mt10_remaining = 3 - mt10_progress
        target_yk = 4 if state.floor_id == "MT10" else 5
        target_bk = 0 if _mt9_mt10_blue_door_opened(sim, state) or state.floor_id == "MT10" else 1
        components["lookahead"] = 0.0
        components["progress"] = _target_floor_score(floor_idx, 10, 120.0)
        components["stage_mt10_resources"] = (
            (9500.0 if mt10_resources_taken(sim, state) else 0.0)
            + mt10_progress * 2600.0
            + _target_floor_score(floor_idx, 10, 160.0)
            + (1800.0 if state.floor_id == "MT10" else 0.0)
            + state.atk * 380.0
            + state.defense * 360.0
            + min(state.hp, 2200) * 5.0
            - mt10_remaining * 450.0
        )
        if mt10_progress == 0:
            target_hp = 270 if state.floor_id == "MT10" else 235
            components["stage_mt10_resources"] += min(state.hp, 360) * 16.0
            components["deadend"] -= max(0, target_hp - state.hp) * 95.0
        components["stage_gems"] = (
            critical_gem_progress(state) * 1120.0
            + _gems_landmark_progress(sim, state) * 620.0
            + boss_atk_drop * 4.0
            + boss_def_drop * 3.5
            - remaining_gems * 45.0
        )
        components["key_pressure"] = (
            min(target_yk, yk) * 650.0
            + min(max(target_bk, 1), bk) * 720.0
            - max(0, target_yk - yk) * 5200.0
            - max(0, target_bk - bk) * 5600.0
        )
        if floor_idx >= 9 and yk == 0 and not mt10_resources_taken(sim, state):
            components["deadend"] -= 3200.0
    elif stage == "guard_stats_ready":
        guard_damage = red_key_route_damage(sim, state)
        guard_margin = red_key_route_margin(sim, state)
        resource_progress = _guard_ready_resource_progress(sim, state)
        gem_progress = _gems_landmark_progress(sim, state)
        atk_gap = max(0, _GUARD_TARGET_ATK - state.atk)
        def_gap = max(0, _GUARD_TARGET_DEF - state.defense)
        components["lookahead"] = 0.0
        components["progress"] = 0.0
        components["stage_gems"] = (
            critical_gem_progress(state) * 1280.0
            + gem_progress * 920.0
            + guard_atk_drop * 16.0
            + guard_def_drop * 14.0
            + boss_atk_drop * 5.0
            + boss_def_drop * 4.0
            - remaining_gems * 80.0
        )
        components["stage_guard_stats_ready"] = (
            (8800.0 if _guard_stats_ready(sim, state) else 0.0)
            + resource_progress * 420.0
            + max(0, 2200 - min(guard_damage, 2200)) * 4.8
            + max(-1600, min(guard_margin, 1800)) * 0.9
            + state.atk * 520.0
            + state.defense * 520.0
            + min(state.hp, 1800) * 1.4
            - atk_gap * 5200.0
            - def_gap * 5400.0
        )
        components["key_pressure"] = yk * 130.0 + bk * 240.0 - max(0, 2 - yk) * 1200.0
        if guard_damage >= 10_000:
            components["deadend"] -= 4500.0
    elif stage == "guard_route_ready":
        guard_damage = red_key_route_damage(sim, state)
        guard_margin = red_key_route_margin(sim, state)
        resource_progress = _guard_ready_resource_progress(sim, state)
        route_progress = _red_key_landmark_progress(sim, state)
        components["lookahead"] = 0.0
        components["progress"] = _target_floor_score(floor_idx, 8, 160.0)
        components["stage_guard_route_ready"] = (
            (9000.0 if _guard_route_ready(sim, state) else 0.0)
            + (2200.0 if _guard_stats_ready(sim, state) else 0.0)
            + resource_progress * 720.0
            + route_progress * 520.0
            + max(0, 2200 - min(guard_damage, 2200)) * 2.8
            + max(-1800, min(guard_margin, 2400)) * 3.9
            + state.atk * 230.0
            + state.defense * 220.0
            + min(state.hp, 2200) * 2.6
        )
        components["stage_gems"] = (
            critical_gem_progress(state) * 900.0
            + guard_atk_drop * 9.0
            + guard_def_drop * 8.0
            - remaining_gems * 45.0
        )
        components["key_pressure"] = yk * 110.0 + bk * 220.0 - max(0, 1 - yk) * 900.0
        if guard_damage >= 10_000:
            components["deadend"] -= 4000.0
        elif guard_margin < 0:
            components["deadend"] -= min(5500.0, abs(float(guard_margin)) * 7.0)
    elif stage == "guard_ready":
        guard_damage = red_key_route_damage(sim, state)
        guard_margin = red_key_route_margin(sim, state)
        resource_progress = _guard_ready_resource_progress(sim, state)
        gem_progress = _gems_landmark_progress(sim, state)
        components["lookahead"] = 0.0
        components["progress"] = 0.0
        components["stage_gems"] = (
            critical_gem_progress(state) * 980.0
            + gem_progress * 760.0
            + max(0, 1800 - min(guard_damage, 1800)) * 5.0
            + guard_atk_drop * 12.0
            + guard_def_drop * 10.0
            + boss_atk_drop * 4.0
            + boss_def_drop * 3.0
            - remaining_gems * 60.0
        )
        components["stage_guard_ready"] = (
            (7000.0 if guard_ready(sim, state) else 0.0)
            + resource_progress * 620.0
            + max(0, 1800 - min(guard_damage, 1800)) * 2.2
            + max(-1200, min(guard_margin, 1600)) * 2.4
            + state.atk * 210.0
            + state.defense * 185.0
            + min(state.hp, 1600) * 1.4
        )
        components["key_pressure"] = state.items.get("yellowKey", 0) * 110.0 + state.items.get("blueKey", 0) * 220.0
    elif stage == "red_key":
        guard_damage = red_key_route_damage(sim, state)
        guard_margin = red_key_route_margin(sim, state)
        guard_progress = _red_key_landmark_progress(sim, state)
        resource_progress = _guard_ready_resource_progress(sim, state)
        gem_progress = _gems_landmark_progress(sim, state)
        components["lookahead"] = 0.0
        components["progress"] = 0.0
        components["stage_gems"] = (
            critical_gem_progress(state) * 1080.0
            + gem_progress * 840.0
            + max(0, 1900 - min(guard_damage, 1900)) * 5.5
            + guard_atk_drop * 13.0
            + guard_def_drop * 11.0
            + boss_atk_drop * 5.5
            + boss_def_drop * 4.5
            + (2400.0 if critical_gems_ready(state) else 0.0)
            - remaining_gems * 65.0
        )
        components["stage_guard_ready"] = (
            (7600.0 if guard_ready(sim, state) else 0.0)
            + resource_progress * 720.0
            + max(0, 1900 - min(guard_damage, 1900)) * 2.7
            + max(-1600, min(guard_margin, 2200)) * 3.1
            + state.atk * 230.0
            + state.defense * 205.0
            + min(state.hp, 1800) * 1.6
        )
        components["stage_red_key"] = (
            (5200.0 if red_key_taken(sim, state) else 0.0)
            + (2200.0 if guard_ready(sim, state) else 0.0)
            + guard_progress * 850.0
            + rk * 1200.0
            + max(0, 1900 - min(guard_damage, 1900)) * 1.8
            + max(-1600, min(guard_margin, 1800)) * 1.6
            + state.atk * 135.0
            + state.defense * 95.0
            + min(state.hp, 1400) * 1.1
        )
        components["threshold"] += (
            damage_drop_for_stats(sim, state, "yellowGuard", atk_delta=1) * 8.0
            + damage_drop_for_stats(sim, state, "yellowGuard", def_delta=1) * 6.0
        )
        components["key_pressure"] = yk * 95.0 + bk * 190.0 + rk * 900.0
        if guard_damage >= 10_000:
            components["deadend"] -= 2500.0
        if not guard_ready(sim, state) and guard_progress > 0:
            components["deadend"] -= 1400.0
    elif stage == "boss_ready":
        components["lookahead"] = 0.0
        components["progress"] = _target_floor_score(floor_idx, 10, 110.0)
        components["stage_boss_ready"] = (
            (9000.0 if boss_ready(sim, state) else 0.0)
            + (3800.0 if red_key_taken(sim, state) else 0.0)
            + max(0, 1900 - min(boss_damage, 1900)) * 3.5
            + max(-1600, min(boss_margin, 2200)) * 4.2
            + state.atk * 260.0
            + state.defense * 230.0
            + min(state.hp, 2200) * 3.0
        )
        components["stage_boss"] = (
            max(0, 1900 - min(boss_damage, 1900)) * 4.0
            + boss_atk_drop * 10.0
            + boss_def_drop * 8.0
        )
        if boss_damage >= 10_000:
            components["deadend"] -= 5000.0
        elif state.hp <= boss_damage:
            components["deadend"] -= (boss_damage - state.hp + 1) * 45.0
    elif stage == "trap":
        components["stage_trap"] = (
            (3600.0 if state.flags.get("10f机关") else 0.0)
            + _target_floor_score(floor_idx, 10, 170.0)
        )
    elif stage == "boss":
        components["stage_boss"] = (
            (8000.0 if state.flags.get("10f战胜骷髅队长") else 0.0)
            + boss_margin * 4.0
            + state.atk * 110.0
            + state.defense * 95.0
        )
    return components


def _target_floor_score(current_floor: int, target_floor: int, scale: float) -> float:
    """Reward being near a stage's target floor without making higher floors globally better."""

    return max(0.0, float(target_floor - abs(current_floor - target_floor))) * scale


def _shield_landmark_progress(sim: MotaSimulator, state: MotaState) -> int:
    progress = 0
    if floor_index(sim, state) >= 7:
        progress += 1
    if _mt8_top_yellow_doors_opened(sim, state) >= 1:
        progress += 1
    if _mt8_top_yellow_doors_opened(sim, state) >= 2:
        progress += 1
    if floor_index(sim, state) >= 9:
        progress += 1
    if _mt9_blue_door_opened(sim, state):
        progress += 1
    if _mt9_shield_fakewall_opened(sim, state):
        progress += 1
    if mt9_shield_taken(sim, state):
        progress += 2
    return progress


def _shield_key_pressure(sim: MotaSimulator, state: MotaState) -> float:
    yk = state.items.get("yellowKey", 0)
    bk = state.items.get("blueKey", 0)
    opened = _mt8_top_yellow_doors_opened(sim, state)
    needed_yellow = max(0, 2 - opened)
    needed_blue = 0 if _mt9_blue_door_opened(sim, state) else 1
    pressure = min(yk, 6) * 80.0 + bk * 500.0
    pressure -= max(0, needed_yellow - yk) * 900.0
    pressure -= max(0, needed_blue - bk) * 1600.0
    if yk > needed_yellow + 1:
        pressure += min(yk - needed_yellow, 4) * 35.0
    return pressure


def _mt8_top_yellow_doors_opened(sim: MotaSimulator, state: MotaState) -> int:
    opened = 0
    for x, y in [(3, 1), (4, 1)]:
        if tile_id(sim, state, "MT8", x, y) != "yellowDoor":
            opened += 1
    return opened


def _mt9_blue_door_opened(sim: MotaSimulator, state: MotaState) -> bool:
    return tile_id(sim, state, "MT9", 6, 3) != "blueDoor"


def _mt9_mt10_blue_door_opened(sim: MotaSimulator, state: MotaState) -> bool:
    return tile_id(sim, state, "MT9", 3, 11) != "blueDoor"


def _mt8_blue_key_route_progress(sim: MotaSimulator, state: MotaState) -> int:
    progress = 0
    if floor_index(sim, state) >= 8:
        progress += 1
    for x, y, block_id in [
        (7, 5, "bluePriest"),
        (7, 7, "bat"),
        (8, 8, "bluePriest"),
        (11, 9, "yellowDoor"),
        (11, 10, "skeleton"),
        (10, 11, "skeletonSoldier"),
        (9, 11, "yellowDoor"),
    ]:
        if tile_id(sim, state, "MT8", x, y) != block_id:
            progress += 1
    if mt8_blue_key_taken(sim, state):
        progress += 4
    return progress


def _mt7_yellow_budget_progress(sim: MotaSimulator, state: MotaState) -> int:
    progress = 0
    if state.items.get("yellowKey", 0) >= MT10_RESOURCE_YELLOW_KEY_TARGET:
        progress += MT10_RESOURCE_YELLOW_KEY_TARGET
    if floor_index(sim, state) <= 8:
        progress += 1
    if tile_id(sim, state, "MT7", 9, 7) != "skeletonSoldier":
        progress += 2
    for x, y in [(9, 10), (9, 11)]:
        if tile_id(sim, state, "MT7", x, y) != "yellowKey":
            progress += 1
    return progress


def _mt7_lower_right_refill_remaining(sim: MotaSimulator, state: MotaState) -> bool:
    """7F lower-right cache: one hard soldier unlocks a blue potion and two keys."""

    return any(
        tile_id(sim, state, "MT7", x, y) == block_id
        for x, y, block_id in (
            (9, 7, "skeletonSoldier"),
            (9, 9, "bluePotion"),
            (9, 10, "yellowKey"),
            (9, 11, "yellowKey"),
        )
    )


def _mt9_right_refill_remaining(sim: MotaSimulator, state: MotaState) -> bool:
    """9F right-bottom route is the safe HP refill before the 7F soldier route."""

    return any(
        tile_id(sim, state, "MT9", x, y) == block_id
        for x, y, block_id in (
            (7, 10, "bat"),
            (8, 11, "yellowDoor"),
            (9, 11, "bluePriest"),
            (11, 11, "redPotion"),
            (9, 9, "yellowKey"),
        )
    )


def _mt1_lower_refill_remaining(sim: MotaSimulator, state: MotaState) -> bool:
    """Late low-floor refill: costs HP but nets the extra yellow key needed for 10F."""

    return any(
        tile_id(sim, state, "MT1", x, y) == block_id
        for x, y, block_id in (
            (2, 4, "skeleton"),
            (2, 7, "skeletonSoldier"),
            (3, 10, "yellowKey"),
            (3, 11, "yellowKey"),
        )
    )


def _mt9_shield_fakewall_opened(sim: MotaSimulator, state: MotaState) -> bool:
    return tile_id(sim, state, "MT9", 10, 5) != "fakeWall"


def _red_key_landmark_progress(sim: MotaSimulator, state: MotaState) -> int:
    progress = 0
    if tile_id(sim, state, "MT8", 9, 5) != "yellowGuard":
        progress += 1
    if tile_id(sim, state, "MT8", 11, 5) != "yellowGuard":
        progress += 1
    if tile_id(sim, state, "MT8", 10, 4) != "specialDoor":
        progress += 2
    if red_key_taken(sim, state):
        progress += 3
    return progress


def _gems_landmark_progress(sim: MotaSimulator, state: MotaState) -> int:
    progress = 0
    if tile_id(sim, state, "MT7", 3, 1) != "redGem":
        progress += 1
    if tile_id(sim, state, "MT9", 6, 5) != "redGem":
        progress += 1
    if tile_id(sim, state, "MT9", 1, 5) != "blueGem":
        progress += 1
    if tile_id(sim, state, "MT8", 5, 7) != "yellowDoor":
        progress += 1
    if tile_id(sim, state, "MT8", 4, 10) != "redGem":
        progress += 2
    if tile_id(sim, state, "MT8", 5, 11) != "blueGem":
        progress += 1
    if critical_gems_ready(state):
        progress += 3
    return progress


def _floor_remaining_attack_defense_gems(sim: MotaSimulator, state: MotaState, floor_id: str) -> int:
    if floor_id == "MT10" or floor_id not in state.floors:
        return 0
    count = 0
    for row in state.floors[floor_id]:
        for tile in row:
            if sim.block_id(tile) in {"redGem", "blueGem"}:
                count += 1
    return count


def _stage_target_floor(sim: MotaSimulator, state: MotaState, stage: str) -> int:
    if stage == "mt8_gems":
        return 8
    if stage == "mid_gems":
        if tile_id(sim, state, "MT5", 1, 9) == "blueGem":
            return 5
        return 6
    if stage == "low_gems":
        if tile_id(sim, state, "MT3", 2, 1) == "blueGem" or tile_id(sim, state, "MT3", 2, 9) == "redGem":
            return 3
        if tile_id(sim, state, "MT1", 7, 3) == "redGem" or tile_id(sim, state, "MT1", 7, 4) == "blueGem":
            return 1
        return 1
    return floor_index(sim, state)


def _stage_gem_blocker_progress(sim: MotaSimulator, state: MotaState, stage: str) -> int:
    blockers = {
        "mt8_gems": (
            ("MT8", 6, 3, "yellowDoor"),
            ("MT8", 7, 5, "bluePriest"),
            ("MT8", 7, 7, "bat"),
            ("MT8", 6, 8, "skeleton"),
            ("MT8", 4, 8, "bat"),
            ("MT8", 1, 9, "yellowDoor"),
            ("MT8", 3, 11, "blueDoor"),
            ("MT8", 8, 8, "bluePriest"),
            ("MT8", 11, 9, "yellowDoor"),
            ("MT8", 11, 10, "skeleton"),
            ("MT8", 10, 11, "skeletonSoldier"),
            ("MT8", 9, 11, "yellowDoor"),
        ),
        "mid_gems": (
            ("MT8", 8, 8, "bluePriest"),
            ("MT8", 10, 7, "yellowDoor"),
            ("MT8", 11, 9, "yellowDoor"),
            ("MT8", 11, 10, "skeleton"),
            ("MT8", 8, 10, "redPotion"),
            ("MT6", 2, 9, "bat"),
            ("MT5", 4, 4, "yellowDoor"),
            ("MT5", 3, 5, "bluePriest"),
            ("MT5", 4, 6, "bat"),
            ("MT5", 2, 7, "skeleton"),
        ),
        "low_gems": (
            ("MT1", 4, 1, "redSlime"),
            ("MT1", 5, 1, "greenSlime"),
            ("MT1", 6, 6, "yellowDoor"),
            ("MT3", 1, 3, "bluePriest"),
        ),
    }.get(stage, ())
    progress = 0
    for floor_id, x, y, block_id in blockers:
        if tile_id(sim, state, floor_id, x, y) != block_id:
            progress += 1
    return progress


def _guard_ready_resource_progress(sim: MotaSimulator, state: MotaState) -> int:
    progress = 0
    for floor_id, x, y, item_id in [
        ("MT7", 3, 1, "redGem"),
        ("MT7", 3, 2, "redPotion"),
        ("MT7", 9, 3, "redPotion"),
        ("MT7", 9, 9, "bluePotion"),
        ("MT7", 7, 11, "bluePotion"),
        ("MT8", 4, 10, "redGem"),
        ("MT8", 5, 11, "blueGem"),
        ("MT8", 1, 5, "redPotion"),
        ("MT8", 8, 10, "redPotion"),
        ("MT8", 9, 3, "bluePotion"),
        ("MT8", 11, 3, "redPotion"),
        ("MT1", 7, 3, "redGem"),
        ("MT1", 7, 4, "blueGem"),
        ("MT1", 8, 4, "redPotion"),
        ("MT1", 1, 10, "redPotion"),
        ("MT1", 1, 11, "redPotion"),
        ("MT2", 3, 10, "bluePotion"),
        ("MT2", 4, 10, "bluePotion"),
        ("MT2", 3, 11, "bluePotion"),
        ("MT4", 11, 2, "bluePotion"),
        ("MT5", 1, 9, "blueGem"),
        ("MT5", 3, 9, "redPotion"),
        ("MT6", 4, 9, "blueGem"),
        ("MT3", 2, 9, "redGem"),
        ("MT4", 7, 10, "redGem"),
        ("MT10", 2, 6, "blueGem"),
        ("MT10", 10, 6, "redGem"),
        ("MT10", 11, 11, "bluePotion"),
    ]:
        if tile_id(sim, state, floor_id, x, y) != item_id:
            progress += 1
    if yellow_guard_damage(sim, state) < 10_000:
        progress += 2
    if guard_ready(sim, state):
        progress += 4
    return progress


def _guard_stats_ready(sim: MotaSimulator, state: MotaState) -> bool:
    return (
        mt9_shield_taken(sim, state)
        and state.atk >= _GUARD_TARGET_ATK
        and state.defense >= _GUARD_TARGET_DEF
        and red_key_route_damage(sim, state) < 10_000
    )


def _guard_route_ready(sim: MotaSimulator, state: MotaState) -> bool:
    return _guard_stats_ready(sim, state) and guard_ready(sim, state)


def _stage_resource_vector(sim: MotaSimulator, state: MotaState) -> tuple[int, ...]:
    return (
        *sim.resource_vector(state),
        _boss_margin(sim, state),
        -remaining_attack_defense_gems(sim, state),
        _shield_landmark_progress(sim, state),
        _gems_landmark_progress(sim, state),
        mt10_resource_progress(sim, state),
        _guard_ready_resource_progress(sim, state),
        int(_guard_stats_ready(sim, state)),
        int(_guard_route_ready(sim, state)),
        _red_key_landmark_progress(sim, state),
        int(state.flags.get("10f机关", False)),
        int(state.flags.get("10f战胜骷髅队长", False)),
    )


def _boss_margin(sim: MotaSimulator, state: MotaState) -> int:
    return boss_route_margin(sim, state)


def _boss_damage(sim: MotaSimulator, state: MotaState) -> int:
    return boss_route_required_damage(sim, state)


def _enemy_damage(sim: MotaSimulator, state: MotaState, enemy_id: str) -> int:
    info = sim.damage_info(state, enemy_id)
    if info is None:
        return 10_000
    return info["damage"]


def _novelty_key(state: MotaState, stage: str) -> tuple[Any, ...]:
    return (
        stage,
        state.floor_id,
        state.x // 2,
        state.y // 2,
        state.atk // 2,
        state.defense // 2,
        state.hp // 100,
        state.items.get("yellowKey", 0),
        state.items.get("blueKey", 0),
        state.items.get("redKey", 0),
        current_stage_name_placeholder(state),
    )


def current_stage_name_placeholder(state: MotaState) -> str:
    if state.flags.get("10f战胜骷髅队长"):
        return "done"
    if state.flags.get("10f机关"):
        return "trap"
    if state.flags.get("nowShield") == "shield1":
        return "shield"
    if state.flags.get("nowWeapon") == "sword1":
        return "sword"
    return "early"


def _stage_action_bias(
    action: dict[str, Any],
    stage: str,
    state: MotaState | None = None,
    sim: MotaSimulator | None = None,
) -> float:
    if state is not None and sim is not None:
        stage = _effective_score_stage(sim, state, stage)
    label = action.get("label", "")
    bias = action_bias(action)
    if state is not None and sim is not None:
        bias += _resource_aware_action_bias(action, stage, state, sim)
    if stage == "red_key" and state is not None and sim is not None:
        current_floor = floor_index(sim, state)
        if current_floor < 8:
            if "upFloor" in label:
                bias += 1500.0
            if "downFloor" in label:
                bias -= 900.0
        elif current_floor > 8:
            if "downFloor" in label:
                bias += 1500.0
            if "upFloor" in label:
                bias -= 900.0
        else:
            if "upFloor" in label or "downFloor" in label:
                bias -= 600.0
            if label.startswith("fight"):
                bias += 900.0
            if "yellowGuard" in label or "specialDoor" in label or "redKey" in label:
                bias += 1800.0
            if "yellowDoor" in label:
                bias += 450.0
    if stage == "guard_stats_ready" and state is not None and sim is not None:
        if "redGem" in label:
            bias += 5200.0 + _gem_margin_bias("redGem", state, sim)
        if "blueGem" in label:
            bias += 5400.0 + _gem_margin_bias("blueGem", state, sim)
        if any(token in label for token in ["Potion", "yellowKey", "blueKey"]):
            bias += 1100.0
        if label.startswith("fight"):
            target = action.get("target") or []
            if len(target) == 3:
                _, x, y = target
                enemy_id = sim.block_id(sim.tile(state, int(x), int(y)))
                if enemy_id:
                    info = sim.damage_info(state, enemy_id)
                    if info is not None:
                        bias += max(0.0, 420.0 - min(info["damage"], 420.0)) * 0.9
                        if info["damage"] <= 40:
                            bias += 420.0
            if "yellowGuard" in label:
                bias -= 9000.0
        current_floor = floor_index(sim, state)
        if current_floor > 8 and "downFloor" in label:
            bias += 900.0
        if current_floor < 8 and "upFloor" in label:
            bias += 650.0
    if stage == "guard_route_ready" and state is not None and sim is not None:
        current_floor = floor_index(sim, state)
        margin = red_key_route_margin(sim, state)
        if current_floor < 8 and "upFloor" in label:
            bias += 2200.0
        elif current_floor > 8 and "downFloor" in label:
            bias += 2200.0
        elif current_floor == 8 and ("upFloor" in label or "downFloor" in label):
            bias -= 1000.0
        if label.startswith("fight"):
            target = action.get("target") or []
            if len(target) == 3:
                _, x, y = target
                enemy_id = sim.block_id(sim.tile(state, int(x), int(y)))
                if enemy_id:
                    info = sim.damage_info(state, enemy_id)
                    if info is not None:
                        bias += max(0.0, 520.0 - min(info["damage"], 520.0)) * 0.6
        if "yellowGuard" in label:
            if margin < 0:
                bias -= 7000.0 + min(6000.0, abs(float(margin)) * 6.0)
            else:
                bias += 4200.0 + min(3500.0, float(margin) * 2.8)
        if "specialDoor" in label or "redKey" in label:
            bias += 5600.0
    if stage == "guard_ready" and state is not None and sim is not None and label.startswith("fight"):
        target = action.get("target") or []
        if len(target) == 3:
            _, x, y = target
            enemy_id = sim.block_id(sim.tile(state, int(x), int(y)))
            if enemy_id:
                info = sim.damage_info(state, enemy_id)
                if info is not None:
                    bias += max(0.0, 700.0 - min(info["damage"], 700.0)) * 0.8
                    if info["damage"] <= 40:
                        bias += 260.0
        if "yellowGuard" in label:
            margin = red_key_route_margin(sim, state)
            if margin < 0:
                bias -= 4500.0 + min(5000.0, abs(float(margin)) * 8.0)
            else:
                bias += 2600.0 + min(2500.0, float(margin) * 3.0)
    if stage == "sword" and ("sword1" in label or "MT5" in label):
        bias += 320
    if stage == "pre_shield_gems" and any(
        token in label for token in ["sword1", "redGem", "blueGem", "yellowKey", "Potion", "upFloor", "downFloor"]
    ):
        bias += 780
    if stage == "pre_shield_gems" and state is not None and sim is not None:
        current_floor = floor_index(sim, state)
        need_atk = state.atk < 21
        need_def = state.defense < 11
        if has_first_sword(sim, state) and (need_atk or need_def):
            if current_floor > 3 and "downFloor" in label:
                bias += 3600.0
            if current_floor <= 3 and "upFloor" in label:
                bias -= 2400.0
            if need_atk and "redGem" in label:
                bias += 7600.0 + _gem_margin_bias("redGem", state, sim)
            if need_def and "blueGem" in label:
                bias += 7600.0 + _gem_margin_bias("blueGem", state, sim)
            if label.startswith("fight"):
                unlock_value = _one_step_unlock_value(sim, state, action)
                if unlock_value < 1800:
                    bias -= 1600.0
    if stage == "pre_shield_gems" and ("redGem" in label or "blueGem" in label):
        bias += 2600 + _gem_margin_bias("redGem" if "redGem" in label else "blueGem", state, sim)
    if stage == "shield" and (
        "shield1" in label
        or "blueKey" in label
        or "sword1" in label
        or "redGem" in label
        or "blueGem" in label
        or "Potion" in label
        or "MT9" in label
    ):
        bias += 360
    if stage == "shield" and state is not None and sim is not None:
        current_floor = floor_index(sim, state)
        if current_floor < 8 and "upFloor" in label:
            bias += 2600.0
        elif current_floor == 8 and "upFloor MT8:6,1" in label:
            bias += 4200.0
        elif current_floor == 9 and ("downFloor" in label or "upFloor" in label):
            bias -= 1400.0
        elif current_floor > 9 and "downFloor" in label:
            bias += 1400.0
        if any(token in label for token in ("blueDoor MT9:6,3", "fakeWall MT9:10,5", "shield1 MT9:9,7")):
            bias += 6200.0
        if any(
            token in label
            for token in (
                "yellowDoor MT5:5,1",
                "yellowDoor MT5:2,3",
                "yellowDoor MT5:4,4",
                "redSlime MT5:4,1",
                "bat MT5:3,3",
                "upFloor MT5:1,1",
                "upFloor MT6:11,11",
                "blueDoor MT7:5,5",
                "bluePriest MT7:4,6",
                "redGem MT7:3,1",
                "upFloor MT7:1,1",
                "upFloor MT8:6,1",
            )
        ):
            bias += 5200.0
        if "skeletonSoldier MT9" in label:
            info = None
            target = action.get("target") or []
            if len(target) == 3:
                _, x, y = target
                enemy_id = sim.block_id(sim.tile(state, int(x), int(y), "MT9"))
                info = sim.damage_info(state, enemy_id) if enemy_id else None
            if info is not None:
                if state.hp > info["damage"] + 60:
                    bias += 3600.0
                else:
                    bias -= 5200.0
    if stage == "shield" and any(
        token in label
        for token in [
            "sword1",
            "yellowKey",
            "blueKey",
            "redGem",
            "blueGem",
            "Potion",
            "upFloor",
            "downFloor",
            "upFloor MT7:1,1",
            "yellowDoor MT5:5,1",
            "yellowDoor MT5:2,3",
            "yellowDoor MT5:4,4",
            "redSlime MT5:4,1",
            "bat MT5:3,3",
            "upFloor MT5:1,1",
            "upFloor MT6:11,11",
            "blueDoor MT7:5,5",
            "bluePriest MT7:4,6",
            "redGem MT7:3,1",
            "yellowDoor MT8:3,1",
            "yellowDoor MT8:4,1",
            "upFloor MT8:6,1",
            "blueDoor MT9:6,3",
            "fakeWall MT9:10,5",
            "shield1 MT9:9,7",
        ]
    ):
        bias += 900
    if stage == "shield" and "sword1" in label:
        bias += 1200
    if stage == "shield" and "blueDoor MT9:6,3" in label and state is not None:
        bias += 1200.0 if state.items.get("blueKey", 0) >= 2 else -8500.0
    if stage == "shield" and "skeletonSoldier MT9:1,3" in label and state is not None:
        bias += 1600.0 if state.hp > 360 else -7800.0
    if stage == "mt8_hp_ready" and state is not None and sim is not None:
        current_floor = floor_index(sim, state)
        if "Potion" in label:
            bias += 13500.0
        if "yellowKey" in label or "blueKey" in label:
            bias += 2200.0
        if state.hp < 155:
            if current_floor > 4 and "downFloor" in label:
                bias += 3800.0
            if current_floor < 4 and "upFloor" in label:
                bias += 1400.0
            if label.startswith("fight"):
                unlock_value = _one_step_unlock_value(sim, state, action)
                target = action.get("target") or []
                info = None
                if len(target) == 3:
                    floor_id, x, y = target
                    enemy_id = sim.block_id(sim.tile(state, int(x), int(y), str(floor_id)))
                    info = sim.damage_info(state, enemy_id) if enemy_id else None
                if unlock_value >= 1500 or (info is not None and info["damage"] <= 12):
                    bias += 900.0
                else:
                    bias -= 7600.0
        elif current_floor < 8 and "upFloor" in label:
            bias += 2600.0
    if stage in {"gems", "lower_gems", "all_gems"} and ("redGem" in label or "blueGem" in label):
        bias += 420
    if stage in {"gems", "lower_gems", "all_gems"} and any(
        token in label
        for token in [
            "redGem MT7:3,1",
            "blueGem MT9:1,5",
            "redGem MT9:6,5",
            "yellowDoor MT8:5,7",
            "bat MT8:4,8",
            "skeleton MT8:6,8",
            "redGem MT8:4,10",
            "blueGem MT8:5,11",
            "downFloor MT9:6,1",
            "downFloor MT8:1,1",
        ]
    ):
        bias += 760
    if stage in {"gems", "lower_gems", "all_gems"} and any(
        token in label
        for token in [
            "yellowDoor MT8:5,7",
            "bat MT8:4,8",
            "redGem MT8:4,10",
            "blueGem MT8:5,11",
        ]
    ):
        bias += 2200
    if stage in {"mt8_gems", "mid_gems", "low_gems"} and state is not None and sim is not None:
        current_floor = floor_index(sim, state)
        target_floor = _stage_target_floor(sim, state, stage)
        target_remaining = remaining_stage_gem_targets(sim, state, stage)
        if current_floor < target_floor and "upFloor" in label:
            bias += 4200.0
        if current_floor > target_floor and "downFloor" in label:
            bias += 5200.0
        if current_floor == target_floor and target_remaining > 0 and ("upFloor" in label or "downFloor" in label):
            bias -= 12000.0
        elif abs(current_floor - target_floor) <= 1 and ("upFloor" in label or "downFloor" in label):
            bias -= 1800.0
        if stage == "mt8_gems" and tile_id(sim, state, "MT8", 1, 5) == "redPotion" and state.hp < 150:
            if any(token in label for token in ("yellowDoor MT8:1,3", "redPotion MT8:1,5", "yellowKey MT8:3,4")):
                bias += 9800.0
            elif "yellowDoor MT8:6,3" in label:
                bias -= 15000.0
            elif label.startswith("fight") and any(
                token in label
                for token in (
                    "bluePriest MT8:7,5",
                    "bat MT8:7,7",
                    "skeleton MT8:6,8",
                    "bat MT8:4,8",
                )
            ):
                bias -= 6200.0
        target_tokens = {
            "mt8_gems": [
                "redGem MT8:4,10",
                "blueGem MT8:5,11",
                "redPotion MT8:1,5",
                "yellowKey MT8:3,4",
                "yellowDoor MT8:1,3",
                "yellowDoor MT8:6,3",
                "yellowDoor MT8:5,7",
                "greenSlime MT8:1,10",
                "bat MT8:2,11",
                "bat MT8:4,8",
                "skeleton MT8:6,8",
                "blueDoor MT8:3,11",
                "bluePriest MT8:8,8",
                "yellowDoor MT8:11,9",
                "skeleton MT8:11,10",
                "skeletonSoldier MT8:10,11",
                "yellowDoor MT8:9,11",
                "redPotion MT8:8,10",
                "blueKey MT8:7,10",
                "yellowKey MT8:7,11",
            ],
            "mid_gems": [
                "bluePriest MT8:8,8",
                "yellowDoor MT8:10,7",
                "yellowDoor MT8:11,9",
                "skeleton MT8:11,10",
                "redPotion MT8:8,10",
                "blueKey MT8:7,10",
                "blueGem MT5:1,9",
                "redPotion MT5:3,9",
                "blueGem MT6:4,9",
                "yellowDoor MT5",
                "bluePriest MT5:3,5",
                "bat MT5:4,6",
                "skeleton MT5:2,7",
                "bluePriest MT6:1,8",
                "bat MT6:2,9",
            ],
            "low_gems": [
                "redGem MT1:7,3",
                "blueGem MT1:7,4",
                "redPotion MT1:1,10",
                "redPotion MT1:1,11",
                "bluePotion MT2:3,10",
                "bluePotion MT2:4,10",
                "bluePotion MT2:3,11",
                "blueGem MT3:2,1",
                "redGem MT3:2,9",
                "redPotion MT3:11,1",
                "redPotion MT3:11,7",
                "bluePotion MT4:11,2",
                "yellowDoor MT1:6,6",
                "yellowDoor MT1:4,3",
                "redSlime MT1:4,1",
                "greenSlime MT1:5,1",
                "yellowDoor MT3",
                "bluePriest MT3:1,3",
            ],
        }[stage]
        if any(token in label for token in target_tokens):
            bias += 7800.0
        if stage == "mt8_gems":
            if state.floor_id == "MT6" and state.money >= 50 and state.items.get("blueKey", 0) <= 1:
                if any(
                    token in label
                    for token in (
                        "skeleton MT6:10,3",
                        "redPotion MT6:8,3",
                        "buy blueKey MT6:8,4",
                        "upFloor MT6:11,11",
                    )
                ):
                    bias += 52000.0
            if (
                floor_index(sim, state) == 8
                and not mt8_blue_key_taken(sim, state)
                and state.items.get("blueKey", 0) >= 1
                and any(token in label for token in ("blueDoor MT8:3,11", "redGem MT8:4,10", "blueGem MT8:5,11"))
            ):
                bias += 46000.0
            if "skeletonSoldier MT8:10,11" in label:
                bias += 42000.0
            if "yellowDoor MT8:9,11" in label:
                bias += 42000.0 if state.items.get("yellowKey", 0) >= 1 else -22000.0
            if any(token in label for token in ("redPotion MT8:8,10", "blueKey MT8:7,10", "yellowKey MT8:7,11")):
                bias += 36000.0
            if (
                not mt8_blue_key_taken(sim, state)
                and state.items.get("yellowKey", 0) <= 1
                and any(
                    token in label
                    for token in (
                        "yellowDoor MT8:10,7",
                        "yellowDoor MT8:5,7",
                        "yellowDoor MT8:1,9",
                    )
                )
            ):
                bias -= 26000.0
            if "blueDoor MT8:3,11" in label and state.items.get("blueKey", 0) <= 0:
                bias -= 18000.0
        if label.startswith("fight") and current_floor != target_floor:
            bias -= 5200.0
        elif label.startswith("fight") and stage == "mt8_gems" and "MT8" in label:
            bias -= 1800.0
        if "Potion" in label and state.hp < 260:
            bias += 4200.0
    if stage == "lower_gems" and state is not None and sim is not None:
        current_floor = floor_index(sim, state)
        floor_remaining = _floor_remaining_attack_defense_gems(sim, state, state.floor_id)
        if remaining_lower_attack_defense_gems(sim, state) > 0:
            if current_floor >= 9 and "downFloor" in label:
                bias += 7200.0
            if current_floor >= 9 and "upFloor" in label:
                bias -= 6200.0
            if current_floor > 6 and any(token in label for token in ["downFloor", "MT8", "MT7", "MT6", "MT5", "MT3", "MT1"]):
                bias += 2400.0
        if floor_remaining > 0 and ("upFloor" in label or "downFloor" in label):
            bias -= 7200.0
        if floor_remaining > 0 and state.floor_id in label:
            bias += 2400.0
        if "MT10" in label:
            bias -= 5000.0
        if "redGem" in label:
            bias += 2800 + _gem_margin_bias("redGem", state, sim)
        if "blueGem" in label:
            bias += 2800 + _gem_margin_bias("blueGem", state, sim)
        if any(
            token in label
            for token in [
                "redGem MT1:7,3",
                "blueGem MT1:7,4",
                "blueGem MT3:2,1",
                "redGem MT3:2,9",
                "blueGem MT5:1,9",
                "blueGem MT6:4,9",
                "redGem MT8:4,10",
                "blueGem MT8:5,11",
                "yellowDoor MT1:6,6",
                "yellowDoor MT1:4,3",
                "redSlime MT1:4,1",
                "greenSlime MT1:5,1",
                "yellowDoor MT3",
                "bluePriest MT3:1,3",
                "yellowDoor MT5",
                "skeleton MT5:2,7",
                "bluePriest MT6:1,8",
                "bat MT6:2,9",
                "yellowDoor MT8:5,7",
                "bat MT8:4,8",
                "skeleton MT8:6,8",
                "blueDoor MT8:3,11",
            ]
        ):
            bias += 5200.0
    if stage in {"all_gems", "mt10_blue_ready", "mt10_yellow_ready", "mt10_ready", "mt10_resources"} and state is not None and sim is not None:
        current_floor = floor_index(sim, state)
        if (
            stage in {"mt10_yellow_ready", "mt10_ready", "mt10_resources"}
            and "blueDoor" in label
            and "blueDoor MT9:3,11" not in label
            and not _mt9_mt10_blue_door_opened(sim, state)
            and state.items.get("blueKey", 0) <= 1
        ):
            bias -= 60000.0
        if current_floor < 9:
            if "upFloor" in label:
                if stage in {"mt10_blue_ready", "mt10_yellow_ready", "mt10_ready"} and current_floor == 8 and not mt8_blue_key_taken(sim, state):
                    bias -= 6200.0
                else:
                    bias += 1400.0
            if "downFloor" in label:
                bias -= 700.0
        if "upFloor MT9:1,11" in label:
            if state.items.get("yellowKey", 0) >= MT10_RESOURCE_YELLOW_KEY_TARGET and state.hp >= 80:
                bias += 17000.0
            else:
                bias -= 8500.0
        if "yellowDoor MT9:6,11" in label:
            if state.items.get("yellowKey", 0) >= MT10_RESOURCE_YELLOW_KEY_TARGET and state.hp >= 60:
                bias += 11000.0
            else:
                bias -= 6500.0
        if "downFloor MT9:6,1" in label and not mt10_resources_taken(sim, state):
            if stage in {"mt10_yellow_ready", "mt10_ready"} and state.items.get("yellowKey", 0) < MT10_RESOURCE_YELLOW_KEY_TARGET:
                bias += 9800.0
            elif not mt8_blue_key_taken(sim, state) and state.items.get("blueKey", 0) == 0:
                bias += 7800.0
            else:
                bias -= 9000.0
        if "yellowKey" in label and state.items.get("yellowKey", 0) < MT10_RESOURCE_YELLOW_KEY_TARGET:
            bias += 6500.0
        if "blueKey" in label and not _mt9_mt10_blue_door_opened(sim, state):
            bias += 7200.0
        if stage in {"mt10_yellow_ready", "mt10_ready"} and state.items.get("yellowKey", 0) < MT10_RESOURCE_YELLOW_KEY_TARGET:
            if (
                state.items.get("yellowKey", 0) == MT10_RESOURCE_YELLOW_KEY_TARGET - 1
                and state.hp >= 180
                and _mt1_lower_refill_remaining(sim, state)
            ):
                if "downFloor" in label and floor_index(sim, state) > 1:
                    bias += 24000.0
                if any(
                    token in label
                    for token in (
                        "skeleton MT1:2,4",
                        "yellowDoor MT1:2,5",
                        "yellowKey MT1:1,6",
                        "skeletonSoldier MT1:2,7",
                        "yellowDoor MT1:2,8",
                        "yellowKey MT1:3,10",
                        "redPotion MT1:1,11",
                    )
                ):
                    bias += 26000.0
            if any(
                token in label
                for token in [
                    "yellowDoor MT7:11,5",
                    "skeletonSoldier MT7:9,7",
                    "bluePotion MT7:9,9",
                    "yellowKey MT7:9,10",
                    "yellowKey MT7:9,11",
                    "yellowDoor MT7:5,7",
                    "bat MT7:5,9",
                    "yellowKey MT7:5,10",
                    "yellowKey MT7:5,11",
                    "redSlime MT9:7,6",
                    "bat MT9:7,10",
                    "yellowDoor MT9:8,11",
                    "bluePriest MT9:9,11",
                    "redPotion MT9:11,11",
                    "yellowKey MT9:9,9",
                    "yellowDoor MT9:4,1",
                    "skeleton MT9:3,1",
                    "yellowKey MT9:2,2",
                ]
            ):
                bias += 9600.0
            if (
                _mt9_right_refill_remaining(sim, state)
                and _mt7_lower_right_refill_remaining(sim, state)
                and state.hp <= _enemy_damage(sim, state, "skeletonSoldier") + 15
            ):
                if "upFloor MT8:6,1" in label or "upFloor MT7:1,1" in label:
                    bias += 18000.0
                if any(token in label for token in ("yellowDoor MT7:5,7", "bat MT7:5,9")):
                    bias -= 22000.0
            elif _mt7_lower_right_refill_remaining(sim, state) and state.hp > _enemy_damage(sim, state, "skeletonSoldier") + 5:
                if any(
                    token in label
                    for token in (
                        "yellowDoor MT7:11,5",
                        "skeletonSoldier MT7:9,7",
                        "bluePotion MT7:9,9",
                        "yellowKey MT7:9,10",
                        "yellowKey MT7:9,11",
                    )
                ):
                    bias += 22000.0
        if stage in {"all_gems", "mt10_blue_ready", "mt10_yellow_ready", "mt10_ready"} and any(
            token in label
            for token in [
                "bluePriest MT8:7,5",
                "bat MT8:7,7",
                "bluePriest MT8:8,8",
                "yellowDoor MT8:11,9",
                "skeleton MT8:11,10",
                "yellowDoor MT8:9,11",
                "yellowKey MT8:7,11",
                "blueKey MT8:7,10",
                "redPotion MT8:8,10",
                "yellowKey MT8:5,10",
                "redGem MT8:4,10",
                "blueGem MT8:5,11",
                "downFloor MT8:1,1",
                "downFloor MT7:11,11",
                "skeletonSoldier MT7:9,7",
                "yellowKey MT7:9,10",
                "yellowKey MT7:9,11",
            ]
        ):
            bias += 7200.0
        if "blueDoor MT9:3,11" in label:
            bias += 9500.0 if state.items.get("blueKey", 0) > 0 else -9000.0
        if "Potion" in label and state.hp < 320:
            bias += 6200.0
        if "MT10" in label:
            bias += 900.0
        if any(token in label for token in ["blueGem MT10:2,6", "redGem MT10:10,6", "bluePotion MT10:11,11"]):
            bias += 5200.0
        if any(token in label for token in ["yellowDoor MT10:1,9", "yellowDoor MT10:3,9", "yellowDoor MT10:9,9", "yellowDoor MT10:11,9"]):
            if state.items.get("yellowKey", 0) > 0:
                bias += 2200.0
            else:
                bias -= 2600.0
        if label.startswith("fight") and "MT10" in label:
            target = action.get("target") or []
            enemy_id = sim.block_id(sim.tile(state, int(target[1]), int(target[2]), str(target[0]))) if len(target) == 3 else None
            info = sim.damage_info(state, enemy_id) if enemy_id else None
            if info is not None:
                bias += max(0.0, 420.0 - min(float(info["damage"]), 420.0)) * 3.0
        elif (
            label.startswith("fight")
            and "MT8:" in label
            and not mt8_blue_key_taken(sim, state)
        ):
            bias += 1600.0
        elif label.startswith("fight") and "MT9:7,10" not in label and "MT9:7,6" not in label:
            bias -= 2600.0
        if "downFloor MT10" in label and not mt10_resources_taken(sim, state):
            bias -= 2600.0
    if stage == "all_gems" and state is not None and sim is not None:
        if "redGem" in label:
            bias += 3600 + _gem_margin_bias("redGem", state, sim)
        if "blueGem" in label:
            bias += 3600 + _gem_margin_bias("blueGem", state, sim)
        if "MT10" in label and not mt10_resources_taken(sim, state):
            bias += 1800.0
        if remaining_attack_defense_gems(sim, state) <= 2 and ("upFloor" in label or "MT10" in label):
            bias += 2200.0
    if stage == "red_key" and ("redKey" in label or "MT8:10,2" in label):
        bias += 450
    if stage == "boss_ready" and any(token in label for token in ["Potion", "redGem", "blueGem", "redKey"]):
        bias += 1800
    if stage == "boss_ready" and "Potion" in label:
        bias += 3600
    if stage == "boss_ready" and "skeletonCaptain" in label:
        info = sim.damage_info(state, "skeletonCaptain") if state is not None and sim is not None else None
        if info is not None and state is not None and state.hp > info["damage"]:
            bias += 6200.0
        else:
            bias -= 6200.0
    if stage == "guard_ready" and any(
        token in label
        for token in [
            "Potion",
            "redGem",
            "blueGem",
            "yellowKey",
            "blueKey",
            "yellowDoor",
            "blueDoor",
        ]
    ):
        bias += 1000
    if stage == "guard_ready" and "redGem" in label:
        bias += 3200 + _gem_margin_bias("redGem", state, sim)
    if stage == "guard_ready" and "blueGem" in label:
        bias += 3000 + _gem_margin_bias("blueGem", state, sim)
    if stage == "guard_ready" and "Potion" in label:
        bias += 800
    if stage == "guard_ready" and any(
        token in label
        for token in [
            "blueDoor MT7:5,5",
            "blueDoor MT8:3,11",
            "yellowDoor MT8:5,7",
            "bat MT8:4,8",
            "skeleton MT8:6,8",
            "redGem MT8:4,10",
            "blueGem MT8:5,11",
            "redPotion MT8:8,10",
            "yellowKey MT8:5,10",
            "blueKey MT8:7,10",
        ]
    ):
        bias += 1900
    if stage == "red_key" and any(
        token in label
        for token in [
            "redGem",
            "blueGem",
            "Potion",
            "yellowKey",
            "blueKey",
            "downFloor",
            "downFloor MT9:6,1",
            "upFloor MT8:6,1",
            "yellowGuard MT8:9,5",
            "yellowGuard MT8:11,5",
            "specialDoor MT8:10,4",
            "redKey MT8:10,2",
        ]
    ):
        bias += 900
    if stage == "red_key" and "redGem" in label:
        bias += 2800 + _gem_margin_bias("redGem", state, sim)
    if stage == "red_key" and "blueGem" in label:
        bias += 2600 + _gem_margin_bias("blueGem", state, sim)
    if stage == "red_key" and "Potion" in label:
        bias += 520
    if stage == "red_key" and "yellowGuard" in label:
        margin = red_key_route_margin(sim, state) if state is not None and sim is not None else 0
        if margin < 0:
            bias -= 4500.0 + min(5000.0, abs(float(margin)) * 8.0)
        else:
            bias += 2600.0 + min(2500.0, float(margin) * 3.0)
    if stage == "trap" and ("MT10" in label or "event MT10:6,5" in label):
        bias += 450
    if stage == "boss" and ("skeletonCaptain" in label or "skeletonSoldier" in label):
        bias += 520
    return bias


def _resource_aware_action_bias(
    action: dict[str, Any],
    stage: str,
    state: MotaState,
    sim: MotaSimulator,
) -> float:
    """Generic guard against local greedy actions.

    The staged code already contains many landmark-specific priors. This layer
    adds broad resource accounting so the policy does not learn bad shortcuts:
    fighting before stats, spending keys just because a door is openable, or
    clearing every visible monster.
    """

    label = action.get("label", "")
    bias = 0.0

    if "sword1" in label:
        bias += 5200.0
    if "shield1" in label:
        bias += 5200.0
    if "redGem" in label:
        bias += 3000.0 + _gem_margin_bias("redGem", state, sim)
    if "blueGem" in label:
        bias += 3000.0 + _gem_margin_bias("blueGem", state, sim)
    if "yellowKey" in label:
        bias += 700.0 if state.items.get("yellowKey", 0) < 5 else 120.0
    if "blueKey" in label:
        bias += 1400.0 if state.items.get("blueKey", 0) < 2 else 200.0
    if "redKey" in label:
        bias += 6000.0
    if "Potion" in label and state.hp < 320:
        bias += 1700.0

    critical_shield_door = stage == "shield" and any(
        token in label
        for token in (
            "blueDoor MT7:5,5",
            "blueDoor MT8:3,11",
            "blueDoor MT9:6,3",
            "fakeWall MT9:10,5",
        )
    )

    if label.startswith("open"):
        unlock_value = _one_step_unlock_value(sim, state, action)
        key_cost = _door_key_cost(label)
        if critical_shield_door:
            # The path to the 9F shield intentionally spends scarce blue keys.
            # Generic key conservation otherwise masks these doors and leaves
            # the search looping around MT7 resources.
            bias += 7600.0 + unlock_value * 0.35
        else:
            bias += unlock_value * 0.8 - key_cost
            if unlock_value <= 0:
                bias -= key_cost * 1.5
            if key_cost >= 1800 and state.items.get("blueKey", 0) <= 1:
                bias -= 1400.0
            if key_cost >= 3600 and state.items.get("redKey", 0) <= 1:
                bias -= 3000.0

    if label.startswith("fight"):
        target = action.get("target") or []
        enemy_id = None
        if len(target) == 3:
            floor_id, x, y = target
            enemy_id = sim.block_id(sim.tile(state, int(x), int(y), str(floor_id)))
        info = sim.damage_info(state, enemy_id) if enemy_id else None
        if info is None:
            return bias - 12_000.0
        damage = float(info["damage"])
        unlock_value = _one_step_unlock_value(sim, state, action)
        critical = any(token in label for token in ("yellowGuard", "skeletonCaptain", "skeletonSoldier MT9:1,3"))
        cheap = damage <= _cheap_fight_damage_limit(stage, state)

        # Damage is real cost. Gold/exp are not enough to justify most early
        # fights in first-10-floor Mota unless the monster is cheap or unlocks
        # useful resources.
        bias -= min(damage, 1200.0) * 4.0
        bias += unlock_value
        if cheap:
            bias += 1100.0
        if critical:
            bias += 4200.0
        if not has_first_sword(sim, state) and damage > 45:
            bias -= 5200.0
        if state.flags.get("nowShield") != "shield1" and damage > 180:
            bias -= 2600.0
        if remaining_attack_defense_gems(sim, state) > 0 and damage > 220 and unlock_value < 1800:
            bias -= 3200.0
        if unlock_value <= 0 and not critical and not cheap:
            bias -= 3600.0

    return bias


def _one_step_unlock_value(sim: MotaSimulator, state: MotaState, action: dict[str, Any]) -> float:
    before_targets = _visible_resource_targets(sim, state)
    child = state.clone()
    transition = sim.apply_macro_action(child, action)
    if not transition.ok or child.dead:
        return -8000.0
    after_targets = _visible_resource_targets(sim, child)
    new_targets = after_targets - before_targets
    value = 0.0
    for target in new_targets:
        label = target[-1]
        if "sword1" in label or "shield1" in label:
            value += 7600.0
        elif "redGem" in label or "blueGem" in label:
            value += 4200.0
        elif "redKey" in label:
            value += 6200.0
        elif "blueKey" in label:
            value += 2600.0
        elif "yellowKey" in label:
            value += 1500.0
        elif "Potion" in label:
            value += 900.0
        elif "event" in label:
            value += 1200.0
    return value


def _visible_resource_targets(sim: MotaSimulator, state: MotaState) -> set[tuple[Any, ...]]:
    targets: set[tuple[Any, ...]] = set()
    for action in sim.macro_actions(state):
        label = action.get("label", "")
        if any(
            token in label
            for token in ("sword1", "shield1", "redGem", "blueGem", "Potion", "yellowKey", "blueKey", "redKey", "event")
        ):
            target = action.get("target") or []
            targets.add((*target, label))
    return targets


def _door_key_cost(label: str) -> float:
    if "redDoor" in label:
        return 5200.0
    if "blueDoor" in label:
        return 2600.0
    if "yellowDoor" in label:
        return 950.0
    return 500.0


def _cheap_fight_damage_limit(stage: str, state: MotaState) -> float:
    if stage in {"sword", "pre_shield_gems"}:
        return 55.0
    if stage in {"shield", "mt8_gems", "mid_gems", "low_gems", "all_gems"}:
        return 95.0 if state.flags.get("nowShield") != "shield1" else 150.0
    if stage in {"red_key", "guard_ready", "boss_ready"}:
        return 180.0
    return 120.0


def _gem_margin_bias(
    item_id: str,
    state: MotaState | None,
    sim: MotaSimulator | None,
) -> float:
    if state is None or sim is None:
        return 0.0
    if item_id == "redGem":
        return (
            damage_drop_for_stats(sim, state, "yellowGuard", atk_delta=1) * 10.0
            + damage_drop_for_stats(sim, state, "skeletonCaptain", atk_delta=1) * 4.0
        )
    if item_id == "blueGem":
        return (
            damage_drop_for_stats(sim, state, "yellowGuard", def_delta=1) * 9.0
            + damage_drop_for_stats(sim, state, "skeletonCaptain", def_delta=1) * 3.5
        )
    return 0.0


def _filter_stage_actions(
    sim: MotaSimulator,
    state: MotaState,
    stage: str,
    actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep guard/red-key search focused on resource and gate actions.

    The first-ten task has many harmless clean-up fights. They make the search
    frontier wide while rarely advancing the red-key bottleneck. We still keep
    cheap fights because they may unblock resource corridors.
    """

    score_stage = _effective_score_stage(sim, state, stage)
    if score_stage not in {
        "shield",
        "lower_gems",
        "mt8_gems",
        "mid_gems",
        "low_gems",
        "mt8_hp_ready",
        "all_gems",
        "mt10_blue_ready",
        "mt10_yellow_ready",
        "mt10_ready",
        "mt10_resources",
        "guard_ready",
        "red_key",
        "boss_ready",
    }:
        return actions
    if (
        score_stage == "mt8_gems"
        and state.floor_id == "MT6"
        and state.money >= 50
        and state.items.get("blueKey", 0) <= 1
        and tile_id(sim, state, "MT6", 8, 4) == "trader"
    ):
        merchant_route_tokens = (
            "skeleton MT6:10,3",
            "redPotion MT6:8,3",
            "buy blueKey MT6:8,4",
            "upFloor MT6:11,11",
        )
        focused = [
            action
            for action in actions
            if any(token in action.get("label", "") for token in merchant_route_tokens)
        ]
        if focused:
            return focused
    if (
        score_stage == "mt8_gems"
        and state.hp < 150
        and tile_id(sim, state, "MT8", 1, 5) == "redPotion"
    ):
        focused = [
            action
            for action in actions
            if any(
                token in action.get("label", "")
                for token in ("yellowDoor MT8:1,3", "redPotion MT8:1,5", "yellowKey MT8:3,4")
            )
        ]
        if focused:
            return focused
    if score_stage == "mt8_gems" and floor_index(sim, state) < 8:
        focused = [
            action
            for action in actions
            if not action.get("label", "").startswith("fight")
            and any(
                token in action.get("label", "")
                for token in ("upFloor", "downFloor", "Potion", "yellowKey", "blueKey")
            )
        ]
        if focused:
            return focused
    if (
        score_stage == "mt8_gems"
        and floor_index(sim, state) == 8
        and not mt8_blue_key_taken(sim, state)
        and state.items.get("blueKey", 0) < 1
    ):
        right_route_tokens = (
            "bluePriest MT8:8,8",
            "yellowDoor MT8:11,9",
            "skeleton MT8:11,10",
            "skeletonSoldier MT8:10,11",
            "yellowDoor MT8:9,11",
            "redPotion MT8:8,10",
            "blueKey MT8:7,10",
            "yellowKey MT8:7,11",
        )
        focused = [
            action
            for action in actions
            if any(token in action.get("label", "") for token in right_route_tokens)
        ]
        if focused:
            return focused
    if (
        score_stage == "mt8_gems"
        and floor_index(sim, state) == 8
        and not mt8_blue_key_taken(sim, state)
        and state.items.get("blueKey", 0) >= 1
        and (
            tile_id(sim, state, "MT8", 4, 10) == "redGem"
            or tile_id(sim, state, "MT8", 5, 11) == "blueGem"
        )
    ):
        early_gem_tokens = (
            "yellowDoor MT8:6,3",
            "bluePriest MT8:7,5",
            "yellowKey MT8:3,4",
            "yellowDoor MT8:1,3",
            "redPotion MT8:1,5",
            "redSlime MT8:2,6",
            "greenSlime MT8:3,6",
            "redSlime MT8:4,6",
            "yellowDoor MT8:5,7",
            "bat MT8:4,8",
            "yellowDoor MT8:1,9",
            "greenSlime MT8:1,10",
            "bat MT8:2,11",
            "blueDoor MT8:3,11",
            "redGem MT8:4,10",
            "blueGem MT8:5,11",
            "yellowKey MT8:4,11",
            "yellowKey MT8:5,10",
        )
        focused = [
            action
            for action in actions
            if any(token in action.get("label", "") for token in early_gem_tokens)
        ]
        if focused:
            return focused
    if (
        score_stage == "mt8_gems"
        and floor_index(sim, state) == 8
        and mt8_blue_key_taken(sim, state)
        and (
            tile_id(sim, state, "MT8", 4, 10) == "redGem"
            or tile_id(sim, state, "MT8", 5, 11) == "blueGem"
        )
    ):
        gem_route_tokens = (
            "yellowKey MT8:7,11",
            "skeleton MT8:6,8",
            "bat MT8:4,8",
            "yellowDoor MT8:5,7",
            "yellowDoor MT8:1,9",
            "greenSlime MT8:1,10",
            "bat MT8:2,11",
            "blueDoor MT8:3,11",
            "redGem MT8:4,10",
            "blueGem MT8:5,11",
            "yellowKey MT8:4,11",
            "yellowKey MT8:5,10",
        )
        focused = [
            action
            for action in actions
            if any(token in action.get("label", "") for token in gem_route_tokens)
        ]
        if focused:
            return focused
    if (
        score_stage in {"mt10_yellow_ready", "mt10_ready", "mt10_resources"}
        and not _mt9_mt10_blue_door_opened(sim, state)
        and state.items.get("blueKey", 0) <= 1
    ):
        preserved = [
            action
            for action in actions
            if "blueDoor" not in action.get("label", "")
            or "blueDoor MT9:3,11" in action.get("label", "")
        ]
        if preserved:
            actions = preserved
    if (
        score_stage in {"mt10_yellow_ready", "mt10_ready"}
        and state.items.get("yellowKey", 0) == MT10_RESOURCE_YELLOW_KEY_TARGET - 1
        and state.hp >= 180
        and _mt1_lower_refill_remaining(sim, state)
    ):
        if state.floor_id == "MT1":
            mt1_refill_tokens = (
                "skeleton MT1:2,4",
                "yellowDoor MT1:2,5",
                "yellowKey MT1:1,6",
                "skeletonSoldier MT1:2,7",
                "yellowDoor MT1:2,8",
                "yellowKey MT1:3,10",
                "redPotion MT1:1,11",
            )
            focused = [
                action
                for action in actions
                if any(token in action.get("label", "") for token in mt1_refill_tokens)
            ]
            if focused:
                return focused
        elif floor_index(sim, state) > 1:
            focused = [
                action
                for action in actions
                if "downFloor" in action.get("label", "")
            ]
            if focused:
                return focused
    if (
        score_stage in {"mt10_yellow_ready", "mt10_ready"}
        and state.items.get("yellowKey", 0) < MT10_RESOURCE_YELLOW_KEY_TARGET
        and state.floor_id == "MT8"
        and _mt9_right_refill_remaining(sim, state)
        and _mt7_lower_right_refill_remaining(sim, state)
        and state.hp <= _enemy_damage(sim, state, "skeletonSoldier") + 15
    ):
        # Do not go down to 7F too early.  The lower-right 7F refill route
        # starts with a skeleton soldier, so the 9F red potion should be taken
        # first when HP is marginal.
        focused = [
            action
            for action in actions
            if "upFloor MT8:6,1" in action.get("label", "")
        ]
        if focused:
            return focused
    if (
        score_stage in {"mt10_yellow_ready", "mt10_ready"}
        and state.items.get("yellowKey", 0) < MT10_RESOURCE_YELLOW_KEY_TARGET
        and state.floor_id == "MT7"
    ):
        soldier_damage = _enemy_damage(sim, state, "skeletonSoldier")
        if (
            _mt7_lower_right_refill_remaining(sim, state)
            and state.hp <= soldier_damage + 15
            and _mt9_right_refill_remaining(sim, state)
        ):
            focused = [
                action
                for action in actions
                if "upFloor MT7:1,1" in action.get("label", "")
            ]
            if focused:
                return focused
        if _mt7_lower_right_refill_remaining(sim, state) and state.hp > soldier_damage + 5:
            mt7_refill_tokens = (
                "yellowDoor MT7:11,5",
                "skeletonSoldier MT7:9,7",
                "bluePotion MT7:9,9",
                "yellowKey MT7:9,10",
                "yellowKey MT7:9,11",
            )
        else:
            mt7_refill_tokens = (
                "yellowDoor MT7:5,7",
                "bat MT7:5,9",
                "yellowKey MT7:5,10",
                "yellowKey MT7:5,11",
            )
        focused = [
            action
            for action in actions
            if any(token in action.get("label", "") for token in mt7_refill_tokens)
        ]
        if focused:
            return focused
    if (
        score_stage in {"mt10_yellow_ready", "mt10_ready"}
        and state.items.get("yellowKey", 0) < MT10_RESOURCE_YELLOW_KEY_TARGET
        and state.floor_id == "MT9"
    ):
        mt9_refill_tokens = (
            "redSlime MT9:7,6",
            "bat MT9:7,10",
            "yellowDoor MT9:8,11",
            "bluePriest MT9:9,11",
            "redPotion MT9:11,11",
            "yellowKey MT9:9,9",
            "yellowDoor MT9:4,1",
            "skeleton MT9:3,1",
            "yellowKey MT9:2,2",
        )
        focused = [
            action
            for action in actions
            if any(token in action.get("label", "") for token in mt9_refill_tokens)
        ]
        if focused:
            return focused
    kept: list[dict[str, Any]] = []
    resource_tokens = (
        "redGem",
        "blueGem",
        "Potion",
        "yellowKey",
        "blueKey",
        "redKey",
        "yellowDoor",
        "blueDoor",
        "specialDoor",
        "fakeWall",
        "shield1",
        "upFloor",
        "downFloor",
    )
    critical_fight_tokens = ("yellowGuard", "skeletonCaptain")
    if score_stage == "shield":
        critical_fight_tokens = ("skeletonSoldier MT9", "bluePriest", "bat")
    for action in actions:
        label = action.get("label", "")
        if any(token in label for token in resource_tokens + critical_fight_tokens):
            kept.append(action)
            continue
        if not label.startswith("fight"):
            continue
        target = action.get("target") or []
        if len(target) != 3:
            continue
        floor_id, x, y = target
        enemy_id = sim.block_id(sim.tile(state, int(x), int(y), str(floor_id)))
        if not enemy_id:
            continue
        info = sim.damage_info(state, enemy_id)
        if score_stage == "shield":
            damage_limit = 260
        elif score_stage in {"lower_gems", "mt8_gems", "mid_gems", "low_gems", "all_gems"}:
            damage_limit = 220
        elif score_stage in {"mt10_yellow_ready", "mt10_ready", "boss_ready"}:
            damage_limit = 170
        else:
            damage_limit = 120
        if info is not None and info["damage"] <= damage_limit:
            kept.append(action)
    return kept or actions


def _gumbel(rng: random.Random) -> float:
    u = min(1.0 - 1e-9, max(1e-9, rng.random()))
    return -math.log(-math.log(u))


def _write_trace_row(
    handle: TextIO,
    sim: MotaSimulator,
    state: MotaState,
    stage: str,
    actions: list[dict[str, Any]],
    expansion: int,
) -> None:
    graph = build_graph_state(sim, state, actions=actions)
    graph_node_by_action = {
        int(node["action_index"]): idx
        for idx, node in enumerate(graph.get("nodes", []))
        if node.get("action_index") is not None
    }
    row = {
        "kind": "search_expansion",
        "stage": stage,
        "expansion": expansion,
        "state": state_summary(state),
        "features": sim.describe_state(state, actions=actions),
        "graph_state": graph,
        "graph_summary": graph.get("summary", {}),
        "candidate_actions": [
            {
                "index": index,
                "label": action.get("label", ""),
                "bias": _stage_action_bias(action, stage, state, sim),
                "graph_node_index": graph_node_by_action.get(index),
            }
            for index, action in enumerate(actions[:32])
        ],
    }
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_stage_dataset_jsonl(
    sim: MotaSimulator,
    route: list[dict[str, Any]],
    path: str | Path,
    solved: bool | None = None,
) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state = sim.reset()
    rows: list[dict[str, Any]] = []
    for index, route_row in enumerate(route):
        before = state.clone()
        stage_before = current_stage_name(sim, before)
        transition = sim.apply_macro_action(state, route_row["action"])
        record = sim.trajectory_step_record(
            before,
            state,
            route_row["action"],
            transition,
            index=index,
            reward=route_row.get("reward"),
        )
        stage_after = current_stage_name(sim, state)
        record["stage_before"] = stage_before
        record["stage_after"] = stage_after
        record["stage_complete_after"] = stage_complete(sim, state, stage_before)
        record["target"] = {
            "stage_value_after": stage_potential(sim, state, stage=stage_before),
            "boss_margin_after": _boss_margin(sim, state),
            "solved_final": None,
        }
        rows.append(record)
        if not transition.ok or state.dead:
            break
    final_solved = bool(state.flags.get("10f战胜骷髅队长")) if solved is None else solved
    with out_path.open("w", encoding="utf8") as handle:
        for row in rows:
            row["target"]["solved_final"] = final_solved
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
