from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from mota_env import MotaSimulator, MotaState, archive_cell_key
from mota_env.rewards import current_stage_name
from mota_solver.resource_planner import (
    ResourcePlannerConfig,
    ResourcePlannerResult,
    _make_reward_model,
    _state_score,
    _target_solved,
)
from mota_solver.az_mcts import filter_stage_actions
from mota_solver.search import SearchNode, reconstruct_route, state_summary


@dataclass(frozen=True)
class GoExploreConfig:
    target_stage: str = "shield"
    iterations: int = 2_000
    rollout_steps: int = 16
    archive_top_k: int = 8
    seed: int = 20260528
    allow_relaxed: bool = False
    trace_limit: int = 20_000
    reward_weights: dict[str, Any] | None = None
    reward_potential_weight: float = 0.02
    reward_stage_mode: str = "current"
    candidate_top_k: int = 6
    temperature: float = 0.65
    novelty_bonus: float = 150.0
    revisit_penalty: float = 0.08
    use_stage_action_filter: bool = True


@dataclass
class GoExploreEntry:
    score: float
    node: SearchNode
    cell: tuple[Any, ...]
    visits: int = 0


@dataclass
class GoExploreArchive:
    top_k: int
    cells: dict[tuple[Any, ...], list[GoExploreEntry]] = field(default_factory=dict)

    def add(self, state: MotaState, stage: str, node: SearchNode, score: float) -> bool:
        cell = archive_cell_key(state, stage)
        bucket = self.cells.setdefault(cell, [])
        bucket.append(GoExploreEntry(score=float(score), node=node, cell=cell))
        bucket.sort(key=lambda entry: entry.score, reverse=True)
        before = len(bucket)
        del bucket[max(1, int(self.top_k)) :]
        return len(bucket) != before or any(entry.node is node for entry in bucket)

    def entries(self) -> list[GoExploreEntry]:
        rows = [entry for bucket in self.cells.values() for entry in bucket]
        rows.sort(key=lambda entry: entry.score - entry.visits * 25.0, reverse=True)
        return rows

    def choose(self, rng: random.Random) -> GoExploreEntry:
        rows = self.entries()
        if not rows:
            raise RuntimeError("empty archive")
        limit = min(len(rows), 64)
        weights = []
        for rank, entry in enumerate(rows[:limit]):
            weights.append(max(1e-6, math.exp(-rank / 12.0) / (1.0 + entry.visits * 0.35)))
        total = sum(weights)
        pick = rng.random() * total
        acc = 0.0
        for entry, weight in zip(rows[:limit], weights):
            acc += weight
            if acc >= pick:
                entry.visits += 1
                return entry
        rows[0].visits += 1
        return rows[0]

    def snapshot(self, limit: int = 2_000) -> list[dict[str, Any]]:
        rows = []
        for entry in self.entries()[: max(0, int(limit))]:
            rows.append(
                {
                    "cell": list(entry.cell),
                    "score": entry.score,
                    "visits": entry.visits,
                    "depth": entry.node.depth,
                    "state": state_summary(entry.node.state),
                    "route_length": entry.node.depth,
                    "strict_alive": entry.node.state.hp > 0,
                    "solved_boss": bool(entry.node.state.flags.get("10f战胜骷髅队长")),
                }
            )
        return rows


def run_go_explore(
    sim: MotaSimulator,
    config: GoExploreConfig | None = None,
    *,
    trace_path: str | Path | None = None,
    start_state: MotaState | None = None,
) -> ResourcePlannerResult:
    config = config or GoExploreConfig()
    rng = random.Random(config.seed)
    planner_config = ResourcePlannerConfig(
        target_stage=config.target_stage,
        archive_top_k=config.archive_top_k,
        allow_relaxed=config.allow_relaxed,
        reward_weights=config.reward_weights,
        reward_potential_weight=config.reward_potential_weight,
        reward_stage_mode=config.reward_stage_mode,
    )
    reward_model = _make_reward_model(config.reward_weights)
    start = start_state.clone() if start_state is not None else sim.reset()
    start_node = SearchNode(start)
    start_score = _state_score(sim, start, config.target_stage, planner_config, reward_model)
    archive = GoExploreArchive(config.archive_top_k)
    archive.add(start, current_stage_name(sim, start), start_node, start_score)
    best = start_node
    best_score = start_score
    seen_cells: set[tuple[Any, ...]] = {archive_cell_key(start, current_stage_name(sim, start))}
    failure_cases: list[dict[str, Any]] = []
    trace_handle: TextIO | None = None
    trace_count = 0

    if trace_path is not None:
        path = Path(trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        trace_handle = path.open("w", encoding="utf8")

    iterations = 0
    try:
        for iterations in range(1, max(1, int(config.iterations)) + 1):
            entry = archive.choose(rng)
            state = entry.node.state.clone()
            parent = entry.node
            for rollout_step in range(max(1, int(config.rollout_steps))):
                if _target_solved(sim, state, config.target_stage):
                    best = parent
                    break
                actions = sim.macro_actions(state)
                if config.use_stage_action_filter:
                    actions = filter_stage_actions(actions, state, config.target_stage, sim=sim)
                candidates = []
                for action in actions:
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
                    child_stage = current_stage_name(sim, child)
                    cell = archive_cell_key(child, child_stage)
                    novelty = 1.0 if cell not in seen_cells else 0.0
                    score = _state_score(sim, child, config.target_stage, planner_config, reward_model)
                    score += novelty * float(config.novelty_bonus)
                    score -= parent.depth * float(config.revisit_penalty)
                    candidates.append((score, novelty, action, child, child_stage, cell))
                if not candidates:
                    break
                candidates.sort(key=lambda row: row[0], reverse=True)
                selected = _sample_ranked(candidates[: max(1, int(config.candidate_top_k))], rng, config.temperature)
                score, novelty, action, child, child_stage, cell = selected
                seen_cells.add(cell)
                step = {
                    "stage": child_stage,
                    "action": action,
                    "before": state_summary(state),
                    "after": state_summary(child),
                    "go_explore_score": score,
                    "novelty": novelty,
                    "archive_cell": list(cell),
                }
                child_node = SearchNode(child, parent=parent, step=step, depth=parent.depth + 1)
                archive.add(child, child_stage, child_node, score)
                if score > best_score or _target_solved(sim, child, config.target_stage):
                    best = child_node
                    best_score = score
                if trace_handle is not None and trace_count < config.trace_limit:
                    trace_handle.write(
                        json.dumps(
                            {
                                "iteration": iterations,
                                "rollout_step": rollout_step,
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
                state = child
                parent = child_node
                if _target_solved(sim, child, config.target_stage):
                    best = child_node
                    break
            if _target_solved(sim, best.state, config.target_stage):
                break
    finally:
        if trace_handle is not None:
            trace_handle.close()

    return ResourcePlannerResult(
        solved=_target_solved(sim, best.state, config.target_stage),
        state=best.state,
        route=reconstruct_route(best),
        expansions=iterations,
        archive_cells=len(archive.cells),
        best_summary=state_summary(best.state),
        archive_snapshot=archive.snapshot(),
        failure_cases=failure_cases,
    )


def _sample_ranked(
    candidates: list[tuple[float, float, dict[str, Any], MotaState, str, tuple[Any, ...]]],
    rng: random.Random,
    temperature: float,
) -> tuple[float, float, dict[str, Any], MotaState, str, tuple[Any, ...]]:
    if len(candidates) == 1 or temperature <= 1e-6:
        return candidates[0]
    best_score = max(float(row[0]) for row in candidates)
    weights = [math.exp((float(row[0]) - best_score) / max(1e-6, float(temperature) * 1000.0)) for row in candidates]
    total = sum(weights)
    pick = rng.random() * total
    acc = 0.0
    for row, weight in zip(candidates, weights):
        acc += weight
        if acc >= pick:
            return row
    return candidates[0]


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
