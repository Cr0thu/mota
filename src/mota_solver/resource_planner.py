from __future__ import annotations

import heapq
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from mota_env import (
    MotaResourceGraphBuilder,
    MotaSimulator,
    MotaState,
    archive_cell_key,
    dominance_signature,
    resource_vector,
)
from mota_env.rewards import (
    boss_route_margin,
    current_stage_name,
    mt10_resource_progress,
    red_key_route_margin,
    stage_complete,
    stage_names,
)
from mota_env.rewards import LearnableStageReward
from mota_solver.search import SearchNode, reconstruct_route, state_summary


@dataclass(frozen=True)
class ResourcePlannerConfig:
    target_stage: str = "shield"
    max_expansions: int = 10_000
    archive_top_k: int = 8
    seed: int = 20260527
    allow_relaxed: bool = False
    trace_limit: int = 50_000
    record_graph_nodes: bool = False
    reward_weights: dict[str, Any] | None = None
    reward_potential_weight: float = 0.0
    reward_stage_mode: str = "current"


@dataclass
class ResourcePlannerResult:
    solved: bool
    state: MotaState
    route: list[dict[str, Any]]
    expansions: int
    archive_cells: int
    best_summary: dict[str, Any]
    archive_snapshot: list[dict[str, Any]] = field(default_factory=list)
    failure_cases: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ArchiveEntry:
    score: float
    node: SearchNode
    cell: tuple[Any, ...]


class DominanceTable:
    def __init__(self, *, relaxed: bool = False):
        self.relaxed = relaxed
        self._table: dict[tuple[Any, ...], list[tuple[int, ...]]] = {}

    def accepts(self, state: MotaState) -> bool:
        signature = dominance_signature(state)
        vector = resource_vector(state, hp_debt=self.relaxed)
        existing = self._table.setdefault(signature, [])
        if any(_dominates(candidate, vector) for candidate in existing):
            return False
        existing[:] = [candidate for candidate in existing if not _dominates(vector, candidate)]
        existing.append(vector)
        return True


class NoveltyTracker:
    def __init__(self):
        self.seen: set[tuple[Any, ...]] = set()

    def score(self, sim: MotaSimulator, state: MotaState) -> float:
        features = {
            ("floor", state.floor_id),
            ("atk", state.atk),
            ("def", state.defense),
            ("yk", min(9, state.items.get("yellowKey", 0))),
            ("bk", min(4, state.items.get("blueKey", 0))),
            ("rk", min(1, state.items.get("redKey", 0))),
            ("trap", bool(state.flags.get("10f机关"))),
            ("boss_margin", max(-30, min(30, boss_route_margin(sim, state) // 50))),
        }
        value = 0.0
        for feature in features:
            if feature not in self.seen:
                value += 1.0
                self.seen.add(feature)
        return value


class GoExploreArchive:
    def __init__(self, top_k: int):
        self.top_k = int(top_k)
        self.cells: dict[tuple[Any, ...], list[ArchiveEntry]] = {}

    def add(self, state: MotaState, stage: str, node: SearchNode, score: float) -> None:
        cell = archive_cell_key(state, stage)
        bucket = self.cells.setdefault(cell, [])
        bucket.append(ArchiveEntry(score=score, node=node, cell=cell))
        bucket.sort(key=lambda entry: entry.score, reverse=True)
        del bucket[self.top_k :]

    def best_entries(self) -> list[ArchiveEntry]:
        entries = [entry for bucket in self.cells.values() for entry in bucket]
        entries.sort(key=lambda entry: entry.score, reverse=True)
        return entries

    def snapshot(self, limit: int = 2_000) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for entry in self.best_entries()[: max(0, int(limit))]:
            rows.append(
                {
                    "cell": list(entry.cell),
                    "score": entry.score,
                    "depth": entry.node.depth,
                    "state": state_summary(entry.node.state),
                    "route_length": entry.node.depth,
                    "strict_alive": entry.node.state.hp > 0,
                    "solved_boss": bool(entry.node.state.flags.get("10f战胜骷髅队长")),
                }
            )
        return rows


def run_resource_planner(
    sim: MotaSimulator,
    config: ResourcePlannerConfig | None = None,
    *,
    trace_path: str | Path | None = None,
    start_state: MotaState | None = None,
) -> ResourcePlannerResult:
    config = config or ResourcePlannerConfig()
    rng = random.Random(config.seed)
    builder = MotaResourceGraphBuilder(sim) if config.record_graph_nodes else None
    start_state = start_state.clone() if start_state is not None else sim.reset()
    start_node = SearchNode(start_state)
    dominance = DominanceTable(relaxed=config.allow_relaxed)
    novelty = NoveltyTracker()
    archive = GoExploreArchive(config.archive_top_k)
    failure_cases: list[dict[str, Any]] = []
    queue: list[tuple[float, int, SearchNode]] = []
    counter = 0
    best = start_node
    reward_model = _make_reward_model(config.reward_weights)
    best_score = _state_score(sim, start_state, config.target_stage, config, reward_model)
    trace_handle: TextIO | None = None
    trace_count = 0

    if trace_path is not None:
        path = Path(trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        trace_handle = path.open("w", encoding="utf8")

    try:
        heapq.heappush(queue, (-best_score, counter, start_node))
        dominance.accepts(start_state)
        archive.add(start_state, current_stage_name(sim, start_state), start_node, best_score)

        expansions = 0
        while queue and expansions < config.max_expansions:
            _priority, _order, node = heapq.heappop(queue)
            state = node.state
            if _target_solved(sim, state, config.target_stage):
                best = node
                break
            actions = sim.macro_actions(state)
            graph = (
                builder.build(state, stage=current_stage_name(sim, state), actions=actions)
                if builder is not None
                else None
            )
            for action_index, action in enumerate(actions):
                child = state.clone()
                transition = sim.apply_macro_action(child, action)
                if not transition.ok:
                    _append_failure(
                        failure_cases,
                        kind="transition_failed",
                        state=state,
                        action=action,
                        detail=transition.message,
                    )
                    continue
                if not config.allow_relaxed and child.hp <= 0:
                    _append_failure(
                        failure_cases,
                        kind="dead_or_nonpositive_hp",
                        state=child,
                        action=action,
                        detail=f"hp={child.hp}",
                    )
                    continue
                if not dominance.accepts(child):
                    continue
                child_stage = current_stage_name(sim, child)
                novelty_score = novelty.score(sim, child)
                base_score = _state_score(sim, child, config.target_stage, config, reward_model)
                score = base_score + novelty_score * 250.0 + rng.random() * 1e-3
                step = {
                    "stage": child_stage,
                    "action": action,
                    "before": state_summary(state),
                    "after": state_summary(child),
                    "resource_graph_node": (
                        graph.action_to_node_id.get(action_index)
                        if graph is not None
                        else _action_node_id(action_index, action)
                    ),
                    "planner_score": score,
                    "novelty": novelty_score,
                }
                child_node = SearchNode(child, parent=node, step=step, depth=node.depth + 1)
                archive.add(child, child_stage, child_node, score)
                counter += 1
                heapq.heappush(queue, (-score, counter, child_node))
                if base_score > best_score:
                    best_score = base_score
                    best = child_node
                if trace_handle is not None and trace_count < config.trace_limit:
                    trace_handle.write(
                        json.dumps(
                            {
                                "expansion": expansions,
                                "depth": child_node.depth,
                                "score": score,
                                "stage": child_stage,
                                "action": action.get("label", ""),
                                "state": state_summary(child),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    trace_count += 1
                if not _target_solved(sim, child, config.target_stage) and not sim.macro_actions(child):
                    _append_failure(
                        failure_cases,
                        kind="deadend_no_actions",
                        state=child,
                        action=action,
                        detail="no executable macro actions",
                    )
            expansions += 1

        solved = _target_solved(sim, best.state, config.target_stage)
        route = reconstruct_route(best)
        archive_snapshot = archive.snapshot()
        return ResourcePlannerResult(
            solved=solved,
            state=best.state,
            route=route,
            expansions=expansions,
            archive_cells=len(archive.cells),
            best_summary=state_summary(best.state),
            archive_snapshot=archive_snapshot,
            failure_cases=failure_cases,
        )
    finally:
        if trace_handle is not None:
            trace_handle.close()


def _dominates(left: tuple[int, ...], right: tuple[int, ...]) -> bool:
    return all(a >= b for a, b in zip(left, right)) and any(a > b for a, b in zip(left, right))


def _target_solved(sim: MotaSimulator, state: MotaState, target_stage: str) -> bool:
    if target_stage == "boss":
        return bool(state.flags.get("10f战胜骷髅队长")) and state.hp > 0
    return stage_complete(sim, state, target_stage) and state.hp > 0


def _make_reward_model(payload: dict[str, Any] | None) -> LearnableStageReward | None:
    if not payload:
        return None
    data = payload.get("weights", payload)
    return LearnableStageReward(
        gamma=float(data.get("gamma", 0.99)),
        global_weights=dict(data.get("global_weights", {})),
        stage_weights={
            str(stage): {str(key): float(value) for key, value in dict(weights).items()}
            for stage, weights in dict(data.get("stage_weights", {})).items()
        },
    )


def _state_score(
    sim: MotaSimulator,
    state: MotaState,
    target_stage: str,
    config: ResourcePlannerConfig | None = None,
    reward_model: LearnableStageReward | None = None,
) -> float:
    solved_bonus = 100_000.0 if _target_solved(sim, state, target_stage) else 0.0
    floor_idx = int(state.floor_id[2:]) if state.floor_id.startswith("MT") else 0
    floor_bonus = floor_idx * _floor_weight(target_stage)
    reward_bonus = 0.0
    if config is not None and reward_model is not None and config.reward_potential_weight:
        stage = target_stage if config.reward_stage_mode == "target" else current_stage_name(sim, state)
        reward_bonus = reward_model.potential(sim, state, stage=stage) * float(config.reward_potential_weight)
    hp_weight = _hp_weight(target_stage)
    hp_buffer_bonus = float(state.hp) * hp_weight
    if target_stage in {"mt10_resources", "all_gems", "red_key", "trap", "boss", "boss_all_gems"}:
        if state.hp < 250:
            hp_buffer_bonus -= 90_000.0
        elif state.hp < 500:
            hp_buffer_bonus -= (500.0 - float(state.hp)) * 180.0
    return (
        solved_bonus
        + _stage_progress_bonus(sim, state, target_stage)
        + _target_landmark_bonus(sim, state, target_stage, floor_idx)
        + _red_key_preparation_bonus(sim, state, target_stage, floor_idx)
        + reward_bonus
        + floor_bonus
        + hp_buffer_bonus
        + float(state.atk) * 600.0
        + float(state.defense) * 650.0
        + float(state.items.get("yellowKey", 0)) * 25.0
        + float(state.items.get("blueKey", 0)) * 80.0
        + float(state.items.get("redKey", 0)) * 250.0
        + max(-2000.0, float(boss_route_margin(sim, state))) * 0.2
        - float(state.steps) * 0.08
    )


def _hp_weight(target_stage: str) -> float:
    if target_stage in {"mt10_resources", "all_gems"}:
        return 9.0
    if target_stage in {"red_key", "trap", "boss", "boss_all_gems"}:
        return 12.0
    if target_stage in {"mt10_yellow_ready", "mt10_blue_ready", "mt8_hp_ready"}:
        return 6.0
    return 1.0


def _red_key_preparation_bonus(
    sim: MotaSimulator,
    state: MotaState,
    target_stage: str,
    floor_idx: int,
) -> float:
    if target_stage not in {"red_key", "trap", "boss", "boss_all_gems"}:
        return 0.0
    margin = red_key_route_margin(sim, state)
    progress = mt10_resource_progress(sim, state)
    bonus = progress * 55_000.0
    bonus += max(-900.0, min(900.0, float(margin))) * 260.0
    red_key_entry_open = sim.block_id(sim.tile(state, 10, 7, "MT8")) != "yellowDoor"
    if progress >= 3 and not red_key_entry_open and state.items.get("yellowKey", 0) < 1:
        # Taking every 10F resource while spending the last yellow key is a
        # dead end for the 8F red-key entrance. Keep this as a planner score
        # penalty rather than a simulator rule so route validation stays pure.
        bonus -= 250_000.0
    if state.atk >= 27:
        bonus += 45_000.0
    if state.defense >= 27:
        bonus += 35_000.0
    if margin >= 0:
        bonus += 120_000.0
        bonus += max(0.0, 10.0 - abs(float(floor_idx - 8))) * 8_000.0
    elif progress < 3:
        bonus += max(0.0, 10.0 - abs(float(floor_idx - 10))) * 5_000.0
    return bonus


def _floor_weight(target_stage: str) -> float:
    if target_stage in {"sword", "pre_shield_gems", "shield"}:
        return 4_000.0
    if target_stage in {"red_key", "trap", "boss", "boss_all_gems"}:
        return 1_500.0
    return 2_500.0


def _stage_progress_bonus(sim: MotaSimulator, state: MotaState, target_stage: str) -> float:
    names = list(stage_names())
    current = current_stage_name(sim, state)
    try:
        target_rank = names.index(target_stage)
        current_rank = names.index(current)
    except ValueError:
        return 0.0
    return float(min(current_rank, target_rank + 1)) * 35_000.0


def _target_landmark_bonus(
    sim: MotaSimulator,
    state: MotaState,
    target_stage: str,
    floor_idx: int,
) -> float:
    bonus = 0.0
    if target_stage in {"sword", "pre_shield_gems", "shield"}:
        if floor_idx >= 4:
            bonus += 20_000.0
        if floor_idx >= 5:
            bonus += 40_000.0
        if stage_complete(sim, state, "sword"):
            bonus += 150_000.0
    if target_stage in {
        "shield",
        "mid_gems",
        "low_gems",
        "all_gems",
        "red_key",
        "trap",
        "boss",
        "boss_all_gems",
    }:
        if floor_idx >= 8:
            bonus += 20_000.0
        if floor_idx >= 9:
            bonus += 35_000.0
        if stage_complete(sim, state, "shield"):
            bonus += 180_000.0
    if target_stage in {"red_key", "trap", "boss", "boss_all_gems"} and stage_complete(sim, state, "red_key"):
        bonus += 200_000.0
    if target_stage in {"trap", "boss", "boss_all_gems"} and state.flags.get("10f机关"):
        bonus += 250_000.0
    if target_stage in {"boss", "boss_all_gems"} and state.flags.get("10f战胜骷髅队长"):
        bonus += 1_000_000.0
    return bonus


def _action_node_id(action_index: int, action: dict[str, Any]) -> str:
    target = action.get("target")
    if target and len(target) == 3:
        return f"{target[0]}:{int(target[1])},{int(target[2])}:action:{action.get('kind', 'unknown')}"
    floor = action.get("floor", "")
    loc = action.get("loc")
    if floor and loc and len(loc) == 2:
        return f"{floor}:{int(loc[0])},{int(loc[1])}:action:{action.get('kind', 'unknown')}"
    return f"action:{action_index}:{action.get('label', '')}"


def _append_failure(
    failure_cases: list[dict[str, Any]],
    *,
    kind: str,
    state: MotaState,
    action: dict[str, Any],
    detail: str,
    limit: int = 2_000,
) -> None:
    if len(failure_cases) >= limit:
        return
    failure_cases.append(
        {
            "kind": kind,
            "action": {
                "label": action.get("label", ""),
                "kind": action.get("kind", ""),
                "target": action.get("target"),
            },
            "state": state_summary(state),
            "detail": detail,
        }
    )


def dump_resource_planner_outputs(result: ResourcePlannerResult, out_dir: str | Path) -> dict[str, str]:
    """Write the standard artifact set consumed by later RL/search runs."""
    from mota_solver.search import write_route_jsonl

    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    outputs = {
        "summary": path / "summary.json",
        "archive_cells": path / "archive_cells.jsonl",
        "strict_candidates": path / "strict_candidates.jsonl",
        "relaxed_candidates": path / "relaxed_candidates.jsonl",
        "best_route": path / "best_route.jsonl",
        "failure_cases": path / "failure_cases.jsonl",
    }
    summary = {
        "solved": result.solved,
        "expansions": result.expansions,
        "archive_cells": result.archive_cells,
        "route_length": len(result.route),
        "best_summary": result.best_summary,
    }
    outputs["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8")
    _write_jsonl(outputs["archive_cells"], result.archive_snapshot)
    _write_jsonl(
        outputs["strict_candidates"],
        [row for row in result.archive_snapshot if row.get("strict_alive")],
    )
    _write_jsonl(
        outputs["relaxed_candidates"],
        [row for row in result.archive_snapshot if not row.get("strict_alive")],
    )
    write_route_jsonl(result.route, outputs["best_route"])
    _write_jsonl(outputs["failure_cases"], result.failure_cases)
    return {key: str(value) for key, value in outputs.items()}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
