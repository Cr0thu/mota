from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mota_env import MotaSimulator, MotaState, build_graph_state
from mota_env.rewards import (
    all_attack_defense_gems_taken,
    boss_route_margin,
    current_stage_name,
    has_first_sword,
    mt10_resource_progress,
    mt10_access_ready,
    mt10_blue_ready,
    mt10_resources_taken,
    mt9_shield_taken,
    pre_mt10_buffer_ready,
    red_key_route_margin,
    red_key_taken,
    lower_attack_defense_gems_taken,
    remaining_attack_defense_gems,
)
from mota_solver.az_mcts import filter_stage_actions
from mota_solver.search import state_summary, write_route_jsonl

from .client import AgentClient, AgentClientError


GEM_AGENTIC_STAGES = {
    "shield_buffer",
    "mid_gems",
    "low_gems",
    "mt8_hp_ready",
    "mt8_gems",
    "lower_gems",
    "pre_mt10_buffer",
    "mt10_blue_ready",
    "mt10_yellow_ready",
    "mt10_ready",
    "mt10_resources",
    "all_gems",
}


@dataclass(frozen=True)
class AgenticRLConfig:
    """Runtime options for one multi-agent exploration run."""

    episodes: int = 1
    max_steps: int = 340
    seed: int = 20260606
    task: str = "defeat MT10 skeleton captain under first10 no-shop/no-fly rules"
    use_stage_filter: bool = True
    external_weight: float = 1.0
    heuristic_weight: float = 1.0
    temperature: float = 0.0
    trace_limit: int = 5_000
    memory_limit: int = 256
    external_roles: tuple[str, ...] = ("planner",)
    beam_width: int = 1
    candidate_top_k: int = 1
    expert_route: str = "artifacts/expert/route_best_bosskill_hp636_len293_20260603.jsonl"
    expert_weight: float = 1.0


@dataclass
class Candidate:
    index: int
    action: dict[str, Any]
    transition_ok: bool
    transition_message: str
    transition_reward: float
    after: MotaState | None
    hp_delta: int = 0
    atk_delta: int = 0
    def_delta: int = 0
    money_delta: int = 0
    yellow_delta: int = 0
    blue_delta: int = 0
    red_delta: int = 0
    block_id: str | None = None
    kind: str | None = None
    stage_after: str | None = None


@dataclass
class AgentVote:
    agent: str
    action_index: int
    score: float
    reason: str


@dataclass
class AgenticRLOutcome:
    solved: bool
    done: bool
    route: list[dict[str, Any]]
    final: dict[str, Any]
    episodes: int
    best_episode: int
    trace: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class BeamNode:
    state: MotaState
    route: list[dict[str, Any]]
    seen_state_keys: set[tuple[Any, ...]]
    score: float
    trace: list[dict[str, Any]] = field(default_factory=list)
    expert_pos: int = 0


class AgenticRLOrchestrator:
    """Coordinate local and future external agents around the simulator.

    This is intentionally not a full solver.  It is a reproducible harness for
    agentic experiments: each role votes on legal macro actions, the arbiter
    applies one action, and the trajectory is saved in normal route JSONL shape.
    """

    def __init__(
        self,
        sim: MotaSimulator,
        config: AgenticRLConfig | None = None,
        *,
        client: AgentClient | None = None,
    ):
        self.sim = sim
        self.config = config or AgenticRLConfig()
        self.client = client
        self.rng = random.Random(self.config.seed)
        self._last_external_errors: list[str] = []
        self._expert_labels = self._load_expert_labels(self.config.expert_route)
        self._expert_label_set = set(self._expert_labels)

    def run(self) -> AgenticRLOutcome:
        if self.config.beam_width > 1:
            return self._run_beam()

        best_route: list[dict[str, Any]] = []
        best_final: dict[str, Any] = {}
        best_score = -math.inf
        best_episode = 0
        best_done = False
        best_solved = False
        all_trace: list[dict[str, Any]] = []
        memory: list[dict[str, Any]] = []

        for episode in range(max(1, int(self.config.episodes))):
            state = self.sim.reset()
            route: list[dict[str, Any]] = []
            episode_trace: list[dict[str, Any]] = []
            seen_state_keys: set[tuple[Any, ...]] = {self.sim.state_key(state)}
            for step in range(max(1, int(self.config.max_steps))):
                if state.done or state.dead:
                    break
                stage = self._agentic_stage(state)
                actions = self.sim.macro_actions(state)
                if self.config.use_stage_filter:
                    filtered = filter_stage_actions(
                        actions,
                        state,
                        self._filter_stage_name(stage),
                        sim=self.sim,
                    )
                    if filtered:
                        actions = self._augment_filtered_actions(state, stage, actions, filtered)
                if not actions:
                    break
                candidates = self._make_candidates(state, actions)
                legal_candidates = [candidate for candidate in candidates if candidate.transition_ok]
                if not legal_candidates:
                    break
                votes = self._collect_votes(
                    state=state,
                    stage=stage,
                    candidates=legal_candidates,
                    memory=memory,
                )
                selected = self._select_candidate(legal_candidates, votes, seen_state_keys)
                before_summary = state_summary(state)
                before_features = self._compact_features(state, actions)
                transition = self.sim.apply_macro_action(state, selected.action)
                after_summary = state_summary(state)
                after_features = self._compact_features(state, self.sim.macro_actions(state))
                row = {
                    "index": len(route),
                    "action": selected.action,
                    "transition": {
                        "ok": transition.ok,
                        "reward": transition.reward,
                        "message": transition.message,
                    },
                    "before": before_summary,
                    "after": after_summary,
                    "before_features": before_features,
                    "after_features": after_features,
                    "agentic": {
                        "episode": episode,
                        "stage": stage,
                        "votes": [vote.__dict__ for vote in votes if vote.action_index == selected.index],
                    },
                }
                route.append(row)
                trace_row = {
                    "episode": episode,
                    "step": step,
                    "stage": stage,
                    "selected": selected.action.get("label", ""),
                    "score": self._route_score(state),
                    "state": after_summary,
                    "top_votes": self._top_vote_rows(votes),
                    "external_vote_count": sum(
                        1 for vote in votes if vote.agent.startswith("external_")
                    ),
                    "external_errors": list(self._last_external_errors),
                }
                episode_trace.append(trace_row)
                memory.append(trace_row)
                del memory[: max(0, len(memory) - self.config.memory_limit)]
                seen_state_keys.add(self.sim.state_key(state))
                if not transition.ok:
                    break
                if state.done:
                    break

            final = state_summary(state)
            score = self._route_score(state)
            solved = bool(state.flags.get("10f战胜骷髅队长")) and state.hp > 0
            if solved:
                score += 1_000_000
            if score > best_score:
                best_score = score
                best_route = route
                best_final = final
                best_episode = episode
                best_done = bool(state.done)
                best_solved = solved
            all_trace.extend(episode_trace)
            del all_trace[: max(0, len(all_trace) - self.config.trace_limit)]

        return AgenticRLOutcome(
            solved=best_solved,
            done=best_done,
            route=best_route,
            final=best_final,
            episodes=max(1, int(self.config.episodes)),
            best_episode=best_episode,
            trace=all_trace,
        )

    def _run_beam(self) -> AgenticRLOutcome:
        best_node: BeamNode | None = None
        best_episode = 0
        all_trace: list[dict[str, Any]] = []
        memory: list[dict[str, Any]] = []

        for episode in range(max(1, int(self.config.episodes))):
            start = self.sim.reset()
            beam = [
                BeamNode(
                    state=start,
                    route=[],
                    seen_state_keys={self.sim.state_key(start)},
                    score=self._route_score(start),
                    trace=[],
                    expert_pos=0,
                )
            ]
            episode_best = beam[0]
            for step in range(max(1, int(self.config.max_steps))):
                expanded: list[BeamNode] = []
                for node_index, node in enumerate(beam):
                    if node.state.done or node.state.dead:
                        expanded.append(node)
                        continue
                    stage = self._agentic_stage(node.state)
                    actions = self.sim.macro_actions(node.state)
                    if self.config.use_stage_filter:
                        filtered = filter_stage_actions(
                            actions,
                            node.state,
                            self._filter_stage_name(stage),
                            sim=self.sim,
                        )
                        if filtered:
                            actions = self._augment_filtered_actions(node.state, stage, actions, filtered)
                    if not actions:
                        expanded.append(node)
                        continue
                    candidates = [
                        candidate
                        for candidate in self._make_candidates(node.state, actions)
                        if candidate.transition_ok and candidate.after is not None
                    ]
                    if not candidates:
                        expanded.append(node)
                        continue
                    unseen_candidates = [
                        candidate
                        for candidate in candidates
                        if candidate.after is not None
                        and self.sim.state_key(candidate.after) not in node.seen_state_keys
                    ]
                    if unseen_candidates:
                        candidates = unseen_candidates
                    elif all(
                        candidate.after is not None
                        and self.sim.state_key(candidate.after) in node.seen_state_keys
                        for candidate in candidates
                    ):
                        expanded.append(node)
                        continue
                    votes = self._collect_votes(
                        state=node.state,
                        stage=stage,
                        candidates=candidates,
                        memory=(memory + node.trace)[-self.config.memory_limit :],
                    )
                    ranked = sorted(
                        candidates,
                        key=lambda candidate: self._beam_candidate_score(
                            node,
                            stage,
                            candidate,
                            votes,
                        ),
                        reverse=True,
                    )[: max(1, int(self.config.candidate_top_k))]
                    for candidate in ranked:
                        assert candidate.after is not None
                        row = self._route_row(
                            route_len=len(node.route),
                            state=node.state,
                            after=candidate.after,
                            candidate_index=candidate.index,
                            action=candidate.action,
                            transition_ok=candidate.transition_ok,
                            transition_reward=candidate.transition_reward,
                            transition_message=candidate.transition_message,
                            actions=actions,
                            stage=stage,
                            votes=votes,
                            episode=episode,
                            beam_node=node_index,
                        )
                        child_route = [*node.route, row]
                        child_seen = set(node.seen_state_keys)
                        child_seen.add(self.sim.state_key(candidate.after))
                        child_score = self._beam_candidate_score(node, stage, candidate, votes)
                        child_expert_pos = self._advance_expert_pos(
                            node.expert_pos,
                            str(candidate.action.get("label", "")),
                        )
                        trace_row = {
                            "episode": episode,
                            "step": step,
                            "beam_node": node_index,
                            "stage": stage,
                            "selected": candidate.action.get("label", ""),
                            "score": child_score,
                            "state": state_summary(candidate.after),
                            "top_votes": self._top_vote_rows(votes),
                            "external_vote_count": sum(
                                1 for vote in votes if vote.agent.startswith("external_")
                            ),
                            "external_errors": list(self._last_external_errors),
                            "expert_pos": child_expert_pos,
                        }
                        child = BeamNode(
                            state=candidate.after,
                            route=child_route,
                            seen_state_keys=child_seen,
                            score=child_score,
                            trace=[*node.trace, trace_row],
                            expert_pos=child_expert_pos,
                        )
                        expanded.append(child)
                        memory.append(trace_row)
                        del memory[: max(0, len(memory) - self.config.memory_limit)]
                if not expanded:
                    break
                expanded.sort(key=lambda node: self._node_total_score(node), reverse=True)
                beam = expanded[: max(1, int(self.config.beam_width))]
                if self._node_total_score(beam[0]) > self._node_total_score(episode_best):
                    episode_best = beam[0]
                if any(node.state.flags.get("10f战胜骷髅队长") for node in beam):
                    break
            if best_node is None or self._node_total_score(episode_best) > self._node_total_score(best_node):
                best_node = episode_best
                best_episode = episode
            all_trace.extend(episode_best.trace)
            del all_trace[: max(0, len(all_trace) - self.config.trace_limit)]

        assert best_node is not None
        solved = bool(best_node.state.flags.get("10f战胜骷髅队长")) and best_node.state.hp > 0
        return AgenticRLOutcome(
            solved=solved,
            done=bool(best_node.state.done),
            route=best_node.route,
            final=state_summary(best_node.state),
            episodes=max(1, int(self.config.episodes)),
            best_episode=best_episode,
            trace=all_trace,
        )

    def write_outputs(
        self,
        outcome: AgenticRLOutcome,
        *,
        out_dir: str | Path,
        route_out: str | Path | None = None,
    ) -> dict[str, str]:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        summary_path = out_path / "summary.json"
        trace_path = out_path / "trace.jsonl"
        summary = {
            "solved": outcome.solved,
            "done": outcome.done,
            "episodes": outcome.episodes,
            "best_episode": outcome.best_episode,
            "route_len": len(outcome.route),
            "final": outcome.final,
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8")
        with trace_path.open("w", encoding="utf8") as handle:
            for row in outcome.trace:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        outputs = {"summary": str(summary_path), "trace": str(trace_path)}
        if route_out is not None and outcome.route:
            write_route_jsonl(outcome.route, route_out)
            outputs["route"] = str(route_out)
        return outputs

    @staticmethod
    def _load_expert_labels(route_path: str) -> list[str]:
        if not route_path:
            return []
        path = Path(route_path)
        if not path.exists():
            return []
        labels: list[str] = []
        with path.open("r", encoding="utf8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                action = row.get("action")
                if isinstance(action, dict):
                    label = str(action.get("label", "")).strip()
                else:
                    label = str(row.get("label", "")).strip()
                if label:
                    labels.append(label)
        return labels

    def _advance_expert_pos(self, pos: int, label: str) -> int:
        if not self._expert_labels or not label:
            return pos
        pos = max(0, min(pos, len(self._expert_labels)))
        if pos < len(self._expert_labels) and label == self._expert_labels[pos]:
            return pos + 1
        return pos

    @staticmethod
    def _is_floor_transition_label(label: str) -> bool:
        return "upFloor" in label or "downFloor" in label

    def _make_candidates(self, state: MotaState, actions: list[dict[str, Any]]) -> list[Candidate]:
        rows: list[Candidate] = []
        for index, action in enumerate(actions):
            before_yellow = state.items.get("yellowKey", 0)
            before_blue = state.items.get("blueKey", 0)
            before_red = state.items.get("redKey", 0)
            block_id, kind = self._action_target_info(state, action)
            child = state.clone()
            transition = self.sim.apply_macro_action(child, action)
            rows.append(
                Candidate(
                    index=index,
                    action=action,
                    transition_ok=transition.ok and not child.dead,
                    transition_message=transition.message,
                    transition_reward=transition.reward,
                    after=child if transition.ok else None,
                    hp_delta=child.hp - state.hp,
                    atk_delta=child.atk - state.atk,
                    def_delta=child.defense - state.defense,
                    money_delta=child.money - state.money,
                    yellow_delta=child.items.get("yellowKey", 0) - before_yellow,
                    blue_delta=child.items.get("blueKey", 0) - before_blue,
                    red_delta=child.items.get("redKey", 0) - before_red,
                    block_id=block_id,
                    kind=kind,
                    stage_after=current_stage_name(self.sim, child) if transition.ok else None,
                )
            )
        return rows

    def _augment_filtered_actions(
        self,
        state: MotaState,
        stage: str,
        actions: list[dict[str, Any]],
        filtered: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        augmented = list(filtered)
        seen_labels = {str(action.get("label", "")) for action in augmented}
        allow_escape_floor = state.hp < 180
        for action in actions:
            label = str(action.get("label", ""))
            if label in seen_labels:
                continue
            block, _kind = self._action_target_info(state, action)
            expert_action = label in self._expert_label_set
            gem_resource_action = stage in GEM_AGENTIC_STAGES and (
                (allow_escape_floor and "downFloor" in label)
                or block in {"redPotion", "bluePotion", "yellowKey", "blueKey", "redGem", "blueGem"}
            )
            if (
                expert_action
                or gem_resource_action
            ):
                augmented.append(action)
                seen_labels.add(label)
        return augmented

    def _route_row(
        self,
        *,
        route_len: int,
        state: MotaState,
        after: MotaState,
        candidate_index: int,
        action: dict[str, Any],
        transition_ok: bool,
        transition_reward: float,
        transition_message: str,
        actions: list[dict[str, Any]],
        stage: str,
        votes: list[AgentVote],
        episode: int,
        beam_node: int | None = None,
    ) -> dict[str, Any]:
        return {
            "index": route_len,
            "action": action,
            "transition": {
                "ok": transition_ok,
                "reward": transition_reward,
                "message": transition_message,
            },
            "before": state_summary(state),
            "after": state_summary(after),
            "before_features": self._compact_features(state, actions),
            "after_features": self._compact_features(after, self.sim.macro_actions(after)),
            "agentic": {
                "episode": episode,
                "stage": stage,
                "beam_node": beam_node,
                "votes": [vote.__dict__ for vote in votes if vote.action_index == candidate_index],
            },
        }

    def _collect_votes(
        self,
        *,
        state: MotaState,
        stage: str,
        candidates: list[Candidate],
        memory: list[dict[str, Any]],
    ) -> list[AgentVote]:
        votes: list[AgentVote] = []
        self._last_external_errors = []
        for agent_name, scorer in (
            ("stage_navigator", self._stage_navigator_score),
            ("resource_economy", self._resource_economy_score),
            ("combat_threshold", self._combat_threshold_score),
            ("boss_objective", self._boss_objective_score),
        ):
            for candidate in candidates:
                score, reason = scorer(state, stage, candidate)
                votes.append(
                    AgentVote(
                        agent=agent_name,
                        action_index=candidate.index,
                        score=score * self.config.heuristic_weight,
                        reason=reason,
                    )
                )
        if self.client is not None:
            votes.extend(self._external_votes(state, stage, candidates, memory))
        return votes

    def _external_votes(
        self,
        state: MotaState,
        stage: str,
        candidates: list[Candidate],
        memory: list[dict[str, Any]],
    ) -> list[AgentVote]:
        compact_actions = [
            {
                "action_index": candidate.index,
                "label": candidate.action.get("label", ""),
                "block_id": candidate.block_id,
                "kind": candidate.kind,
                "hp_delta": candidate.hp_delta,
                "atk_delta": candidate.atk_delta,
                "def_delta": candidate.def_delta,
            }
            for candidate in candidates
        ]
        rows: list[AgentVote] = []
        for role in self.config.external_roles:
            try:
                proposals = self.client.rank_actions(
                    role=role,
                    task=self.config.task,
                    state={"stage": stage, **state_summary(state)},
                    actions=compact_actions,
                    memory=memory[-24:],
                )
            except AgentClientError as exc:
                self._last_external_errors.append(f"{role}: {exc}")
                continue
            for proposal in proposals:
                try:
                    action_index = int(proposal["action_index"])
                    score = float(proposal.get("score", 0.0)) * self.config.external_weight
                except (KeyError, TypeError, ValueError):
                    continue
                rows.append(
                    AgentVote(
                        agent=f"external_{role}",
                        action_index=action_index,
                        score=score,
                        reason=str(proposal.get("reason", "")),
                    )
                )
        return rows

    def _select_candidate(
        self,
        candidates: list[Candidate],
        votes: list[AgentVote],
        seen_state_keys: set[tuple[Any, ...]] | None = None,
    ) -> Candidate:
        by_index = {candidate.index: candidate for candidate in candidates}
        scores = {candidate.index: 0.0 for candidate in candidates}
        for vote in votes:
            if vote.action_index in scores:
                scores[vote.action_index] += vote.score
        if seen_state_keys:
            for candidate in candidates:
                if candidate.after is not None and self.sim.state_key(candidate.after) in seen_state_keys:
                    scores[candidate.index] -= 250.0
        if self.config.temperature <= 0:
            best_index = max(scores, key=lambda idx: (scores[idx], -len(by_index[idx].action.get("path", []))))
            return by_index[best_index]
        ranked = sorted(scores.items(), key=lambda row: row[1], reverse=True)[:8]
        max_score = max(score for _idx, score in ranked)
        weights = [math.exp((score - max_score) / max(self.config.temperature, 1e-6)) for _idx, score in ranked]
        total = sum(weights)
        pick = self.rng.random() * total
        acc = 0.0
        for (idx, _score), weight in zip(ranked, weights):
            acc += weight
            if acc >= pick:
                return by_index[idx]
        return by_index[ranked[0][0]]

    def _beam_candidate_score(
        self,
        node: BeamNode,
        stage: str,
        candidate: Candidate,
        votes: list[AgentVote],
    ) -> float:
        if candidate.after is None:
            return -1_000_000.0
        vote_score = sum(vote.score for vote in votes if vote.action_index == candidate.index)
        route_delta = self._route_score(candidate.after) - self._route_score(node.state)
        progress_delta = self._progress_score(candidate.after) - self._progress_score(node.state)
        score = (
            node.score
            + vote_score
            + route_delta * 0.4
            + progress_delta
        )
        next_stage = self._agentic_stage(candidate.after)
        if next_stage != stage:
            score += 2_500.0
        if candidate.after.flags.get("10f战胜骷髅队长"):
            score += 1_000_000.0
        if (
            self._spends_last_yellow_key_before_shield(node.state, stage, candidate)
            and not self._is_next_expert_action(node, candidate)
        ):
            score -= 5_000.0
        if stage == "shield":
            score += self._shield_yellow_buffer_score(node.state, candidate) * 5.0
        if stage in GEM_AGENTIC_STAGES:
            score += self._gem_survival_score(node.state, candidate) * 4.0
            score += self._expert_resource_bias(node.state, stage, candidate) * 4.0
        score += self._expert_route_bias(node, candidate) * max(0.0, self.config.expert_weight)
        if self.sim.state_key(candidate.after) in node.seen_state_keys:
            score -= 500.0
        score -= len(candidate.action.get("path", [])) * 0.2
        return score

    def _is_next_expert_action(self, node: BeamNode, candidate: Candidate) -> bool:
        if not self._expert_labels:
            return False
        pos = max(0, min(node.expert_pos, len(self._expert_labels)))
        return pos < len(self._expert_labels) and str(candidate.action.get("label", "")) == self._expert_labels[pos]

    def _node_total_score(self, node: BeamNode) -> float:
        score = node.score + self._route_score(node.state) + self._progress_score(node.state)
        score += node.expert_pos * 250.0 * max(0.0, self.config.expert_weight)
        if node.state.flags.get("10f战胜骷髅队长"):
            score += 1_000_000.0
        return score

    def _expert_route_bias(self, node: BeamNode, candidate: Candidate) -> float:
        if not self._expert_labels:
            return 0.0
        label = str(candidate.action.get("label", ""))
        if not label:
            return 0.0
        pos = max(0, min(node.expert_pos, len(self._expert_labels)))
        if pos < len(self._expert_labels) and label == self._expert_labels[pos]:
            return 6_000.0
        if self._is_floor_transition_label(label):
            return 0.0
        nearby = self._expert_labels[pos : min(len(self._expert_labels), pos + 24)]
        for offset, expert_label in enumerate(nearby):
            if label == expert_label:
                return max(350.0, 1_400.0 - offset * 65.0)
        return 0.0

    def _progress_score(self, state: MotaState) -> float:
        stage = self._agentic_stage(state)
        stage_rank = {
            "sword": 0,
            "shield": 1,
            "shield_buffer": 2,
            "mid_gems": 2,
            "low_gems": 2,
            "mt8_hp_ready": 2,
            "mt8_gems": 2,
            "lower_gems": 2,
            "pre_mt10_buffer": 2,
            "mt10_blue_ready": 2,
            "mt10_yellow_ready": 2,
            "mt10_ready": 2,
            "mt10_resources": 2,
            "all_gems": 2,
            "pre_red_key_heal": 3,
            "red_key": 4,
            "boss": 5,
        }.get(stage, 0)
        score = stage_rank * 100_000.0
        if stage == "shield":
            score += self._floor_index(state.floor_id) * 2_500.0
            score += state.defense * 1_200.0
            if self._floor_index(state.floor_id) >= 7:
                score += min(state.items.get("yellowKey", 0), 3) * 1_000.0
        if stage in GEM_AGENTIC_STAGES:
            score += state.atk * 1_200.0
            score += state.defense * 1_200.0
            score -= remaining_attack_defense_gems(self.sim, state) * 1_800.0
            score += min(state.hp, 800) * 3.0
            score += min(state.items.get("yellowKey", 0), 3) * 600.0
            score += state.items.get("blueKey", 0) * 900.0
            if state.hp < 180:
                score -= (180 - state.hp) * 20.0
        if stage in {"pre_mt10_buffer", "mt10_blue_ready", "mt10_yellow_ready", "mt10_resources"}:
            score += state.items.get("blueKey", 0) * 3_500.0
            score += min(state.items.get("yellowKey", 0), 5) * 1_200.0
            score += mt10_resource_progress(self.sim, state) * 4_000.0
            if mt10_blue_ready(self.sim, state):
                score += 4_000.0
            if mt10_access_ready(self.sim, state):
                score += 6_000.0
            if self._floor_index(state.floor_id) >= 8:
                score += 1_000.0
        return score

    def _stage_navigator_score(
        self,
        state: MotaState,
        stage: str,
        candidate: Candidate,
    ) -> tuple[float, str]:
        score = 0.0
        if candidate.stage_after and candidate.stage_after != stage:
            score += 60.0
        label = str(candidate.action.get("label", ""))
        block = candidate.block_id or ""
        stage_targets = {
            "sword": ("sword1",),
            "shield": ("shield1",),
            "red_key": ("redKey",),
            "trap": ("yellowDoor MT10", "skeleton MT10", "skeletonSoldier MT10"),
            "boss": ("skeletonCaptain",),
        }
        for token in stage_targets.get(stage, ()):
            if token in {block, label} or token in label:
                score += 45.0
        if stage == "shield":
            score += self._shield_yellow_buffer_score(state, candidate) * 0.35
            if block == "shield1":
                score += 90.0
            if "upFloor" in label and self._action_floor_index(candidate.action) <= 9:
                score += 12.0
            if "downFloor" in label:
                score -= 20.0
            if block in {"redGem", "blueGem", "redPotion", "bluePotion"}:
                if block == "blueGem":
                    score += 55.0
                elif block == "redGem":
                    score += 18.0
                elif block in {"redPotion", "bluePotion"} and state.hp < 550:
                    score += 38.0
                else:
                    score -= 8.0
            if candidate.kind == "enemy" and candidate.hp_delta < -70:
                score -= 35.0
            if candidate.kind == "enemy" and state.hp < 450:
                score += candidate.hp_delta / 2.0
        if stage in GEM_AGENTIC_STAGES:
            if block in {"redGem", "blueGem", "sword1", "shield1"}:
                score += 28.0
            score += self._expert_resource_bias(state, stage, candidate) * 0.4
        if stage == "all_gems" and block in {"redGem", "blueGem"}:
            score += 50.0
        if stage in GEM_AGENTIC_STAGES:
            score += self._gem_survival_score(state, candidate) * 0.45
        if stage == "pre_red_key_heal":
            if block in {"redPotion", "bluePotion"}:
                score += 55.0
            if block == "redKey":
                score -= 40.0
        if stage in {"mt10_resources", "boss_ready"} and "MT10" in label:
            score += 14.0
        return score, f"advance {stage}"

    def _resource_economy_score(
        self,
        state: MotaState,
        stage: str,
        candidate: Candidate,
    ) -> tuple[float, str]:
        block = candidate.block_id or ""
        score = 0.0
        score += candidate.yellow_delta * 10.0
        score += candidate.blue_delta * 18.0
        score += candidate.red_delta * 32.0
        score += candidate.money_delta * 0.05
        if block in {"redGem", "blueGem"}:
            score += 24.0
        elif block in {"sword1", "shield1"}:
            score += 36.0
        elif block in {"redPotion", "bluePotion"}:
            score += max(2.0, candidate.hp_delta / 40.0)
            if state.hp < 350 or stage in {"boss_ready", "trap", "boss"}:
                score += 10.0
        elif block in {"yellowKey", "blueKey"}:
            score += 10.0
        if candidate.kind == "door":
            yellow = state.items.get("yellowKey", 0)
            if "yellowDoor" in str(candidate.action.get("label", "")) and yellow <= 1:
                score -= 36.0
            else:
                score -= 4.0
        if stage == "shield":
            score += self._shield_yellow_buffer_score(state, candidate)
        if self._spends_last_yellow_key_before_shield(state, stage, candidate):
            score -= 500.0
        return score, "preserve keys and stat resources"

    def _spends_last_yellow_key_before_shield(
        self,
        state: MotaState,
        stage: str,
        candidate: Candidate,
    ) -> bool:
        if candidate.after is None:
            return False
        if state.items.get("yellowKey", 0) <= 0:
            return False
        if candidate.after.items.get("yellowKey", 0) > 0:
            return False
        if candidate.block_id in {"sword1", "shield1"}:
            return False
        early_stages = {
            "sword",
            "shield",
        }
        return stage in early_stages

    def _shield_yellow_buffer_score(self, state: MotaState, candidate: Candidate) -> float:
        if candidate.after is None:
            return 0.0
        label = str(candidate.action.get("label", ""))
        before_yellow = state.items.get("yellowKey", 0)
        after_yellow = candidate.after.items.get("yellowKey", 0)
        before_floor = self._floor_index(state.floor_id)
        after_floor = self._floor_index(candidate.after.floor_id)
        score = min(after_yellow, 4) * 10.0

        if "buy 5 yellowKey" in label:
            score += 90.0
        if candidate.block_id == "yellowKey":
            score += 18.0

        if before_floor == 7 and candidate.kind == "door" and "yellowDoor" in label and after_yellow < 3:
            score -= (3 - after_yellow) * 180.0
        if before_floor <= 7 and "upFloor MT8" in label:
            if after_yellow >= 3:
                score += 260.0
            else:
                score -= (3 - after_yellow) * 850.0

        if before_floor == 8 and "yellowDoor MT8" in label:
            if "MT8:3,1" in label:
                score += 120.0 if before_yellow >= 3 else -450.0
            elif "MT8:4,1" in label:
                score += 120.0 if before_yellow >= 2 else -450.0
            elif "MT8:6,3" in label:
                score += 150.0 if before_yellow >= 1 else -450.0
            else:
                score -= 120.0
        if before_floor == 8 and candidate.block_id == "yellowKey":
            score += 180.0
        if before_floor == 8 and "upFloor MT9" in label:
            score += 160.0 if after_yellow >= 3 else -220.0
        if before_floor == 9 and "yellowDoor MT9:8,1" in label:
            score += 180.0
        return score

    def _gem_survival_score(self, state: MotaState, candidate: Candidate) -> float:
        if candidate.after is None:
            return 0.0
        label = str(candidate.action.get("label", ""))
        block = candidate.block_id or ""
        after_hp = candidate.after.hp
        damage = max(0, -candidate.hp_delta)
        score = 0.0

        if block in {"redGem", "blueGem"}:
            score += 140.0
        if block in {"redPotion", "bluePotion"}:
            score += 80.0 + max(0, candidate.hp_delta) / 4.0
            if state.hp < 300:
                score += 100.0
        if candidate.kind == "enemy":
            score -= damage / 2.0
            if damage > 70:
                score -= (damage - 70) * 2.0
            if after_hp < 250:
                score -= (250 - after_hp) * 4.0
            if after_hp < 120:
                score -= (120 - after_hp) * 10.0
        if candidate.kind == "door" and "yellowDoor" in label and candidate.after.items.get("yellowKey", 0) <= 0:
            score -= 80.0
        if self._floor_index(state.floor_id) >= 9 and state.hp < 320 and "downFloor" in label:
            score += 220.0
        if self._floor_index(state.floor_id) >= 9 and state.hp < 300 and candidate.kind == "enemy":
            score -= 260.0
        return score

    def _expert_resource_bias(self, state: MotaState, stage: str, candidate: Candidate) -> float:
        label = str(candidate.action.get("label", ""))
        block = candidate.block_id or ""
        floor = self._floor_index(state.floor_id)
        score = 0.0
        expert_tokens = (
            "bat MT3:3,5",
            "yellowDoor MT3:1,4",
            "blueGem MT3:2,1",
            "yellowDoor MT3:1,6",
            "redGem MT3:2,9",
            "blueGem MT5:1,9",
            "yellowDoor MT4:2,4",
            "blueKey MT4:2,1",
            "yellowKey MT4:3,2",
            "redPotion MT4:1,2",
            "redGem MT8:4,10",
            "blueGem MT8:5,11",
            "blueKey MT8:7,10",
            "bluePriest MT8:8,8",
            "skeleton MT8:6,8",
            "bat MT8:4,8",
            "yellowDoor MT8:9,11",
            "blueDoor MT8:3,11",
            "buy blueKey MT6",
            "redPotion MT6:8,3",
            "blueDoor MT9:3,11",
            "blueGem MT10:2,6",
            "redGem MT10:10,6",
            "bluePotion MT10:11,11",
        )
        if any(token in label for token in expert_tokens):
            score += 180.0
        if block in {"redGem", "blueGem"}:
            score += 90.0
        if stage == "lower_gems" and floor <= 3 and "downFloor" in label:
            score -= 140.0
        if stage == "lower_gems" and floor == 3 and any(token in label for token in ("bat MT3:3,5", "blueGem", "redGem")):
            score += 160.0
        if stage in {"pre_mt10_buffer", "mt10_blue_ready", "mt10_yellow_ready", "mt10_resources"}:
            if block in {"blueKey", "redPotion", "bluePotion", "yellowKey"}:
                score += 45.0
            if "upFloor" in label and floor < 10:
                score += 20.0
        return score

    def _agentic_stage(self, state: MotaState) -> str:
        if not has_first_sword(self.sim, state):
            return "sword"
        if not mt9_shield_taken(self.sim, state):
            return "shield"
        if not lower_attack_defense_gems_taken(self.sim, state):
            return "lower_gems"
        if not pre_mt10_buffer_ready(self.sim, state):
            return "pre_mt10_buffer"
        if not mt10_blue_ready(self.sim, state):
            return "mt10_blue_ready"
        if not mt10_access_ready(self.sim, state):
            return "mt10_yellow_ready"
        if not mt10_resources_taken(self.sim, state):
            return "mt10_resources"
        if not all_attack_defense_gems_taken(self.sim, state):
            return "all_gems"
        if not self._pre_red_key_heal_ready(state):
            return "pre_red_key_heal"
        if not red_key_taken(self.sim, state):
            return "red_key"
        return "boss"

    @staticmethod
    def _filter_stage_name(stage: str) -> str:
        return {
            "all_gems": "all_gems",
            "pre_red_key_heal": "mt10_resources",
            "boss": "boss",
        }.get(stage, stage)

    def _pre_red_key_heal_ready(self, state: MotaState) -> bool:
        if state.hp >= 850:
            return True
        if mt10_resource_progress(self.sim, state) >= 3:
            return True
        return False

    @staticmethod
    def _action_floor_index(action: dict[str, Any]) -> int:
        target = action.get("target")
        if not isinstance(target, list) or not target:
            return 0
        floor = str(target[0])
        try:
            return int(floor.removeprefix("MT"))
        except ValueError:
            return 0

    @staticmethod
    def _floor_index(floor_id: str) -> int:
        try:
            return int(str(floor_id).removeprefix("MT"))
        except ValueError:
            return 0

    def _combat_threshold_score(
        self,
        state: MotaState,
        stage: str,
        candidate: Candidate,
    ) -> tuple[float, str]:
        score = 0.0
        score += candidate.atk_delta * 24.0
        score += candidate.def_delta * 28.0
        if candidate.hp_delta < 0:
            score += candidate.hp_delta / 18.0
        else:
            score += candidate.hp_delta / 80.0
        if candidate.kind == "enemy" and candidate.after is not None:
            if candidate.after.hp <= 0:
                score -= 100.0
            if candidate.hp_delta > -30:
                score += 5.0
        if state.defense < 27 and candidate.def_delta > 0:
            score += 18.0
        if stage == "shield" and candidate.kind == "enemy" and candidate.hp_delta < 0:
            if state.hp < 500:
                score += candidate.hp_delta / 3.0
            if candidate.hp_delta < -90:
                score -= 25.0
        if stage in GEM_AGENTIC_STAGES:
            score += self._gem_survival_score(state, candidate)
            score += self._expert_resource_bias(state, stage, candidate)
        return score, "cross damage thresholds"

    def _boss_objective_score(
        self,
        state: MotaState,
        stage: str,
        candidate: Candidate,
    ) -> tuple[float, str]:
        score = 0.0
        if candidate.after is None:
            return -100.0, "illegal"
        before_boss_margin = boss_route_margin(self.sim, state)
        after_boss_margin = boss_route_margin(self.sim, candidate.after)
        before_red_margin = red_key_route_margin(self.sim, state)
        after_red_margin = red_key_route_margin(self.sim, candidate.after)
        score += (after_boss_margin - before_boss_margin) / 30.0
        score += (after_red_margin - before_red_margin) / 35.0
        if candidate.after.flags.get("10f机关"):
            score += 25.0
        if candidate.after.flags.get("10f战胜骷髅队长"):
            score += 500.0
        if stage in {"red_key", "boss_ready", "trap", "boss"}:
            if candidate.block_id == "redKey":
                score += 80.0
            if "skeletonCaptain" in str(candidate.action.get("label", "")):
                score += 200.0
        return score, "improve red-key and boss margins"

    def _action_target_info(
        self,
        state: MotaState,
        action: dict[str, Any],
    ) -> tuple[str | None, str | None]:
        target = action.get("target")
        if not isinstance(target, list) or len(target) != 3:
            return None, None
        floor_id, x, y = str(target[0]), int(target[1]), int(target[2])
        tile = self.sim.tile(state, x, y, floor_id)
        return self.sim.block_id(tile), self.sim.graph_tile_kind(tile)

    def _compact_features(self, state: MotaState, actions: list[dict[str, Any]]) -> dict[str, Any]:
        graph = build_graph_state(self.sim, state, actions)
        return {
            "stage": current_stage_name(self.sim, state),
            "action_count": len(actions),
            "executable_count": sum(1 for value in graph["executable_mask"] if value),
            "boss_margin": boss_route_margin(self.sim, state),
            "red_key_margin": red_key_route_margin(self.sim, state),
        }

    def _route_score(self, state: MotaState) -> float:
        return (
            state.hp
            + state.atk * 80
            + state.defense * 90
            + state.items.get("yellowKey", 0) * 12
            + state.items.get("blueKey", 0) * 25
            + state.items.get("redKey", 0) * 60
            + boss_route_margin(self.sim, state) * 0.5
        )

    @staticmethod
    def _top_vote_rows(votes: list[AgentVote], limit: int = 6) -> list[dict[str, Any]]:
        totals: dict[int, float] = {}
        reasons: dict[int, list[str]] = {}
        for vote in votes:
            totals[vote.action_index] = totals.get(vote.action_index, 0.0) + vote.score
            reasons.setdefault(vote.action_index, []).append(f"{vote.agent}:{vote.reason}")
        return [
            {
                "action_index": index,
                "score": score,
                "reasons": sorted(
                    reasons.get(index, []),
                    key=lambda reason: (0 if reason.startswith("external_") else 1, reason),
                )[:6],
            }
            for index, score in sorted(totals.items(), key=lambda row: row[1], reverse=True)[:limit]
        ]
