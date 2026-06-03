from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mota_env import MotaSimulator, MotaState, SimulatorConfig, load_game_data
from mota_env.rewards import (
    all_attack_defense_gems_taken,
    boss_ready,
    floor_index,
    guard_ready,
    lower_attack_defense_gems_taken,
    MT10_RESOURCE_YELLOW_KEY_TARGET,
    mt10_access_ready,
    mt10_blue_ready,
    mt10_resource_progress,
    mt10_resources_taken,
    red_key_route_margin,
    red_key_taken,
    remaining_attack_defense_gems,
    remaining_lower_attack_defense_gems,
    stage_complete,
    stage_names,
    stage_potential,
    yellow_guard_margin,
)
from mota_rl.train_actor_critic import (
    ActionActorCritic,
    action_logits,
    find_matching_action,
    load_route,
    select_device,
)
from mota_solver.search import state_summary, write_route_jsonl
from mota_solver.staged import (
    _boss_margin,
    _effective_score_stage,
    _node_score,
    _stage_action_bias,
    _stage_resource_vector,
)


EXTRA_TARGET_STAGES = (
    "mt10_first_resource",
    "mt10_mid_open",
    "post_mid_refill",
    "mt10_right_resources",
    "mt10_blue_ready",
    "mt10_yellow_ready",
    "mt10_ready",
    "mt10_resources",
    "guard_low_refill",
    "guard_ready",
    "guard_stats_ready",
    "guard_route_ready",
)


@dataclass
class BeamNode:
    state: MotaState
    score: float
    parent: "BeamNode | None" = None
    step: dict[str, Any] | None = None
    depth: int = 0


@dataclass
class BeamSearchResult:
    solved: bool
    strict_success: bool
    node: BeamNode
    prefix_route: list[dict[str, Any]]
    route: list[dict[str, Any]]
    expanded_nodes: int
    generated_nodes: int
    best_by_depth: list[dict[str, Any]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode a staged Mota route with action-ranker logits plus dynamic PBRS potential."
    )
    parser.add_argument("--data", default="artifacts/data/mota_first10.json")
    parser.add_argument(
        "--target-stage",
        choices=tuple(dict.fromkeys(stage_names() + EXTRA_TARGET_STAGES)),
        default="red_key",
    )
    parser.add_argument("--model-path", default="")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--start-route", default="")
    parser.add_argument("--start-route-max-steps", type=int, default=0)
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--action-top-k", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument(
        "--continue-after-success",
        action="store_true",
        help="Keep searching after the first stage success so the beam can find a higher-quality success state.",
    )
    parser.add_argument(
        "--success-patience",
        type=int,
        default=0,
        help="When continuing after success, stop this many depths after the first success; 0 means no patience limit.",
    )
    parser.add_argument("--model-weight", type=float, default=1.0)
    parser.add_argument("--model-prior-weight", type=float, default=0.0)
    parser.add_argument("--action-bias-weight", type=float, default=0.5)
    parser.add_argument("--potential-weight", type=float, default=0.04)
    parser.add_argument("--fast-score-weight", type=float, default=0.0015)
    parser.add_argument("--distance-weight", type=float, default=1.0)
    parser.add_argument("--path-step-penalty", type=float, default=0.01)
    parser.add_argument("--macro-step-penalty", type=float, default=0.06)
    parser.add_argument("--revisit-penalty", type=float, default=1.0)
    parser.add_argument("--success-bonus", type=float, default=500.0)
    parser.add_argument("--max-per-diversity-key", type=int, default=4)
    parser.add_argument(
        "--enable-shop",
        action="store_true",
        help="Enable the simplified 4F shop actions for ablation experiments; default keeps the no-shop setting.",
    )
    parser.add_argument("--allow-negative-hp", action="store_true")
    parser.add_argument("--relaxed-min-hp", type=int, default=-1000)
    parser.add_argument(
        "--label-penalties-json",
        default="",
        help='JSON object of substring penalties, e.g. {"upFloor MT9:1,11": 6.0}.',
    )
    parser.add_argument(
        "--adaptive-weights-json",
        default="",
        help='JSON object for adaptive reward weights, e.g. {"atk_delta": 180.0}.',
    )
    parser.add_argument("--route-out", default="artifacts/expert/route_first10_beam_decode.jsonl")
    parser.add_argument("--summary-out", default="artifacts/runs/beam_decode_summary.json")
    parser.add_argument("--trace-out", default="")
    args = parser.parse_args()

    label_penalties = parse_float_mapping_arg(args.label_penalties_json, "--label-penalties-json")
    adaptive_weights = parse_float_mapping_arg(args.adaptive_weights_json, "--adaptive-weights-json")
    model_bundle = load_policy_model(args.model_path, args.hidden, args.device)
    sim = MotaSimulator(
        load_game_data(args.data),
        SimulatorConfig(
            enable_shop=args.enable_shop,
            allow_negative_hp=args.allow_negative_hp,
            min_hp=args.relaxed_min_hp,
            stop_on_boss=args.target_stage != "boss_all_gems",
        ),
    )
    result = beam_search(
        sim=sim,
        target_stage=args.target_stage,
        model_bundle=model_bundle,
        start_route_path=args.start_route,
        start_route_max_steps=args.start_route_max_steps,
        beam_width=args.beam_width,
        action_top_k=args.action_top_k,
        max_steps=args.max_steps,
        stop_on_success=not args.continue_after_success,
        success_patience=args.success_patience,
        model_weight=args.model_weight,
        model_prior_weight=args.model_prior_weight,
        action_bias_weight=args.action_bias_weight,
        potential_weight=args.potential_weight,
        fast_score_weight=args.fast_score_weight,
        distance_weight=args.distance_weight,
        path_step_penalty=args.path_step_penalty,
        macro_step_penalty=args.macro_step_penalty,
        revisit_penalty=args.revisit_penalty,
        success_bonus=args.success_bonus,
        max_per_diversity_key=args.max_per_diversity_key,
        label_penalties=label_penalties,
        adaptive_weights=adaptive_weights,
        trace_out=Path(args.trace_out) if args.trace_out else None,
    )
    write_route_jsonl(result.route, args.route_out)
    summary = {
        "solved": result.solved,
        "strict_success": result.strict_success,
        "target_stage": args.target_stage,
        "route_out": args.route_out,
        "summary_out": args.summary_out,
        "model_path": args.model_path,
        "prefix_steps": len(result.prefix_route),
        "continuation_steps": len(result.route) - len(result.prefix_route),
        "expanded_nodes": result.expanded_nodes,
        "generated_nodes": result.generated_nodes,
        "final": enrich_summary(sim, result.node.state),
        "best_by_depth": result.best_by_depth,
        "config": vars(args),
        "label_penalties": label_penalties,
        "adaptive_weights": adaptive_weights,
    }
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8")
    stdout_summary = dict(summary)
    stdout_summary["best_by_depth_tail"] = summary["best_by_depth"][-3:]
    stdout_summary.pop("best_by_depth", None)
    print(json.dumps(stdout_summary, ensure_ascii=False, indent=2))


def parse_float_mapping_arg(raw: str, flag_name: str) -> dict[str, float]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{flag_name} must be a JSON object: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit(f"{flag_name} must be a JSON object.")
    result: dict[str, float] = {}
    for key, value in parsed.items():
        try:
            result[str(key)] = float(value)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"{flag_name} value for {key!r} must be numeric.") from exc
    return result


def load_policy_model(model_path: str, hidden: int, requested_device: str):
    if not model_path:
        return None
    try:
        import torch
        from torch import nn
    except Exception as exc:  # pragma: no cover - depends on optional local deps.
        raise SystemExit(f"beam_decode with --model-path requires torch: {exc}") from exc

    device = select_device(torch, requested_device)
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get("config", {})
    hidden_size = int(config.get("hidden", hidden))
    model = ActionActorCritic(hidden_size, nn).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return {"model": model, "torch": torch, "device": device, "path": model_path}


def beam_search(
    sim: MotaSimulator,
    target_stage: str,
    model_bundle,
    start_route_path: str = "",
    start_route_max_steps: int = 0,
    beam_width: int = 64,
    action_top_k: int = 12,
    max_steps: int = 160,
    stop_on_success: bool = True,
    success_patience: int = 0,
    model_weight: float = 1.0,
    model_prior_weight: float = 0.0,
    action_bias_weight: float = 0.5,
    potential_weight: float = 0.04,
    fast_score_weight: float = 0.0015,
    distance_weight: float = 1.0,
    path_step_penalty: float = 0.01,
    macro_step_penalty: float = 0.06,
    revisit_penalty: float = 1.0,
    success_bonus: float = 500.0,
    max_per_diversity_key: int = 4,
    action_penalties: dict[tuple[Any, ...], float] | None = None,
    label_penalties: dict[str, float] | None = None,
    adaptive_weights: dict[str, float] | None = None,
    trace_out: Path | None = None,
) -> BeamSearchResult:
    start_state, prefix_route = prepare_start_state(
        sim,
        start_route_path,
        max_steps=start_route_max_steps,
    )
    root = BeamNode(start_state, score=0.0)
    beam = [root]
    best = root
    expanded_nodes = 0
    generated_nodes = 0
    seen_best: dict[tuple[Any, ...], float] = {beam_loop_key(sim, start_state): 0.0}
    best_by_depth: list[dict[str, Any]] = []
    first_success_depth: int | None = None
    trace_handle = None
    trace_best_route_out: Path | None = None
    if trace_out is not None:
        trace_out.parent.mkdir(parents=True, exist_ok=True)
        trace_handle = trace_out.open("w", encoding="utf8")
        trace_best_route_out = trace_out.with_name(f"{trace_out.stem}_best_route.jsonl")

    try:
        for depth in range(max_steps):
            next_candidates: list[BeamNode] = []
            for node in beam:
                if beam_stage_complete(sim, node.state, target_stage) or node.state.dead or node.state.done:
                    best = choose_better_node(sim, best, node, target_stage)
                    continue
                actions = sim.macro_actions(node.state)
                if not actions:
                    best = choose_better_node(sim, best, node, target_stage)
                    continue
                actions = restrict_actions_for_target(sim, node.state, actions, target_stage)
                scored_actions = rank_actions(
                    sim=sim,
                    state=node.state,
                    actions=actions,
                    target_stage=target_stage,
                    model_bundle=model_bundle,
                    model_prior_weight=model_prior_weight,
                    action_bias_weight=action_bias_weight,
                    action_penalties=action_penalties,
                    label_penalties=label_penalties,
                )
                expanded_nodes += 1
                for action_score in scored_actions[:action_top_k]:
                    action = action_score["action"]
                    child = node.state.clone()
                    before = child.clone()
                    transition = sim.apply_macro_action(child, action)
                    if not transition.ok or child.dead:
                        continue
                    score_parts = transition_score_parts(
                        sim=sim,
                        before=before,
                        after=child,
                        action=action,
                        target_stage=target_stage,
                        action_score=action_score,
                        model_weight=model_weight,
                        action_bias_weight=action_bias_weight,
                        potential_weight=potential_weight,
                        fast_score_weight=fast_score_weight,
                        distance_weight=distance_weight,
                        path_step_penalty=path_step_penalty,
                        macro_step_penalty=macro_step_penalty,
                        success_bonus=success_bonus,
                        adaptive_weights=adaptive_weights,
                    )
                    key = beam_loop_key(sim, child)
                    ancestor_revisits = ancestor_state_visits(sim, node, key, limit=40)
                    if ancestor_revisits > 0:
                        continue
                    revisit_cost = revisit_penalty * 160.0 * ancestor_revisits
                    score_parts["ancestor_revisit_penalty"] = -revisit_cost
                    candidate_score = node.score + score_parts["total"] - revisit_cost
                    if key in seen_best:
                        continue
                    seen_best[key] = candidate_score
                    step = {
                        "index": node.depth,
                        "action": action,
                        "before": state_summary(before),
                        "after": state_summary(child),
                        "after_key": key,
                        "transition": transition.message,
                        "reward": transition.reward,
                        "beam_score": candidate_score,
                        "score_parts": score_parts,
                    }
                    candidate = BeamNode(
                        child,
                        score=candidate_score,
                        parent=node,
                        step=step,
                        depth=node.depth + 1,
                    )
                    generated_nodes += 1
                    next_candidates.append(candidate)
                    best = choose_better_node(sim, best, candidate, target_stage)

            if not next_candidates:
                break
            beam = select_next_beam(
                sim=sim,
                nodes=next_candidates,
                target_stage=target_stage,
                beam_width=beam_width,
                max_per_diversity_key=max_per_diversity_key,
            )
            best = choose_better_node(sim, best, beam[0], target_stage)
            if first_success_depth is None and beam_stage_complete(sim, best.state, target_stage):
                first_success_depth = depth + 1
            depth_row = {
                "depth": depth + 1,
                "beam_size": len(beam),
                "best": enrich_summary(sim, best.state),
                "best_score": best.score,
                "frontier_top": [
                    {
                        "score": node.score,
                        "state": enrich_summary(sim, node.state),
                        "last_action": None if node.step is None else node.step["action"].get("label", ""),
                    }
                    for node in beam[:5]
                ],
            }
            best_by_depth.append(depth_row)
            if trace_handle is not None:
                trace_handle.write(json.dumps(depth_row, ensure_ascii=False) + "\n")
                trace_handle.flush()
                if trace_best_route_out is not None:
                    write_route_jsonl(
                        reindex_route(prefix_route + reconstruct_beam_route(best)),
                        trace_best_route_out,
                    )
            if stop_on_success and beam_stage_complete(sim, best.state, target_stage):
                break
            if (
                not stop_on_success
                and success_patience > 0
                and first_success_depth is not None
                and depth + 1 >= first_success_depth + success_patience
            ):
                break
    finally:
        if trace_handle is not None:
            trace_handle.close()

    continuation = reconstruct_beam_route(best)
    route = reindex_route(prefix_route + continuation)
    solved = beam_stage_complete(sim, best.state, target_stage)
    strict = solved and best.state.hp > 0 and not best.state.dead
    return BeamSearchResult(
        solved=solved,
        strict_success=strict,
        node=best,
        prefix_route=prefix_route,
        route=route,
        expanded_nodes=expanded_nodes,
        generated_nodes=generated_nodes,
        best_by_depth=best_by_depth,
    )


def ancestor_state_visits(
    sim: MotaSimulator,
    node: BeamNode,
    state_key: tuple[Any, ...],
    limit: int = 40,
) -> int:
    """Count recent route ancestors with the same simulator state key.

    Beam scoring can otherwise create positive up/down stair cycles: the
    resulting map/resource state is identical, but distance and potential terms
    can make the loop look slightly better.  The global seen-best table alone
    is insufficient because those loops may monotonically increase score for
    the same state.
    """

    visits = 0
    current: BeamNode | None = node
    remaining = limit
    while current is not None and remaining > 0:
        if beam_loop_key(sim, current.state) == state_key:
            visits += 1
        current = current.parent
        remaining -= 1
    return visits


def beam_loop_key(sim: MotaSimulator, state: MotaState) -> tuple[Any, ...]:
    """State key for beam de-duplication, intentionally excluding step count."""

    return (
        state.floor_id,
        state.x,
        state.y,
        state.hp,
        state.atk,
        state.defense,
        state.mdef,
        state.money,
        state.exp,
        tuple(sorted((key, value) for key, value in state.items.items() if value)),
        tuple(sorted((key, str(value)) for key, value in state.flags.items() if value)),
        tuple(sorted(state.triggered_events)),
        sim.floors_signature(state),
    )


def prepare_start_state(
    sim: MotaSimulator,
    start_route_path: str,
    max_steps: int = 0,
) -> tuple[MotaState, list[dict[str, Any]]]:
    state = sim.reset()
    if not start_route_path:
        return state, []

    prefix: list[dict[str, Any]] = []
    rows = load_route(start_route_path)
    for index, row in enumerate(rows):
        if max_steps > 0 and index >= max_steps:
            break
        actions = sim.macro_actions(state)
        match = find_matching_action(actions, row["action"])
        if match is None:
            raise SystemExit(
                f"Cannot replay start route at step {index}: {row['action'].get('label', '')}"
            )
        action = actions[match]
        before = state.clone()
        transition = sim.apply_macro_action(state, action)
        prefix.append(
            {
                "index": index,
                "action": action,
                "before": state_summary(before),
                "after": state_summary(state),
                "transition": transition.message,
                "reward": transition.reward,
                "source": "start_route",
            }
        )
        if not transition.ok:
            raise SystemExit(f"Start route failed at step {index}: {transition.message}")
        if state.dead or state.done:
            break
    return state, prefix


def rank_actions(
    sim: MotaSimulator,
    state: MotaState,
    actions: list[dict[str, Any]],
    target_stage: str,
    model_bundle,
    model_prior_weight: float,
    action_bias_weight: float,
    action_penalties: dict[tuple[Any, ...], float] | None = None,
    label_penalties: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    logits: list[float]
    if model_bundle is None:
        logits = [0.0 for _ in actions]
    else:
        torch = model_bundle["torch"]
        with torch.no_grad():
            raw_logits, _ = action_logits(
                sim=sim,
                state=state,
                actions=actions,
                model=model_bundle["model"],
                target_stage=target_stage,
                prior_weight=model_prior_weight,
                device=model_bundle["device"],
                torch=torch,
            )
        logits = [float(value) for value in raw_logits.detach().cpu().tolist()]
    max_logit = max(logits) if logits else 0.0
    ranked = []
    score_target_stage = scoring_stage_name(target_stage)
    for action, logit in zip(actions, logits):
        bias = (
            _stage_action_bias(action, score_target_stage, state, sim) / 3000.0
            + decoder_action_bias(sim, state, action, target_stage)
        )
        penalty = action_penalty_for(
            state=state,
            action=action,
            action_penalties=action_penalties,
            label_penalties=label_penalties,
        )
        ranked.append(
            {
                "action": action,
                "model_logit": logit,
                "model_logit_centered": logit - max_logit,
                "action_bias": bias,
                "action_penalty": penalty,
                "ranking_score": logit + action_bias_weight * bias - penalty,
            }
        )
    ranked.sort(key=lambda row: row["ranking_score"], reverse=True)
    return ranked


def restrict_actions_for_target(
    sim: MotaSimulator,
    state: MotaState,
    actions: list[dict[str, Any]],
    target_stage: str,
) -> list[dict[str, Any]]:
    """Apply hard subgoal constraints for narrow repair stages.

    The first 10F resource stage is a navigation repair, not a general resource
    collection stage. Once the state already has the 10F key budget, letting the
    beam consider 8F side fights reintroduces the exact failure mode this stage
    is meant to isolate.
    """

    if target_stage == "guard_low_refill":
        return restrict_guard_low_refill_actions(sim, state, actions)
    if target_stage == "mt10_mid_open":
        return restrict_mt10_mid_open_actions(sim, state, actions)
    if target_stage == "post_mid_refill":
        return restrict_post_mid_refill_actions(sim, state, actions)
    if target_stage == "mt10_right_resources":
        return restrict_mt10_right_resource_actions(sim, state, actions)
    if target_stage not in {"mt10_first_resource"}:
        return actions
    entry_ready = mt10_access_ready(sim, state) or (
        mt10_blue_ready(sim, state) and state.items.get("yellowKey", 0) >= 3
    )
    if mt10_resource_progress(sim, state) > 0 or not entry_ready:
        return actions

    def keep(patterns: tuple[str, ...]) -> list[dict[str, Any]]:
        kept = [action for action in actions if any(p in action.get("label", "") for p in patterns)]
        return kept or actions

    if state.floor_id == "MT8":
        return keep(("upFloor MT8:6,1",))
    if state.floor_id == "MT9":
        return keep((
            "redSlime MT9:7,6",
            "bat MT9:7,10",
            "yellowDoor MT9:6,11",
            "blueDoor MT9:3,11",
            "upFloor MT9:1,11",
        ))
    if state.floor_id == "MT10":
        return keep((
            "yellowDoor MT10",
            "blueDoor MT10",
            "blueGem MT10:2,6",
            "redGem MT10:10,6",
            "bluePotion MT10:11,11",
        ))
    if floor_index(sim, state) < 8:
        return keep(("upFloor",))
    return actions


def restrict_mt10_mid_open_actions(
    sim: MotaSimulator,
    state: MotaState,
    actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if mt10_mid_opened(sim, state):
        return actions
    if mt10_resource_progress(sim, state) <= 0:
        return restrict_actions_for_target(sim, state, actions, "mt10_first_resource")

    def keep(patterns: tuple[str, ...]) -> list[dict[str, Any]]:
        kept = [action for action in actions if any(p in action.get("label", "") for p in patterns)]
        return kept or actions

    current_floor = floor_index(sim, state)
    if current_floor < 10:
        return keep(("upFloor",))
    if state.floor_id == "MT10":
        return keep((
            "yellowDoor MT10:3,9",
            "bluePriest MT10:4,11",
            "downFloor MT10:1,11",
            "blueGem MT10:2,6",
        ))
    return actions


def restrict_post_mid_refill_actions(
    sim: MotaSimulator,
    state: MotaState,
    actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if post_mid_refill_ready(sim, state):
        return actions
    if not mt10_mid_opened(sim, state):
        return restrict_actions_for_target(sim, state, actions, "mt10_mid_open")

    current_floor = floor_index(sim, state)
    refill_tokens = (
        "Potion",
        "yellowKey",
        "blueKey",
        "redGem",
        "blueGem",
    )
    kept: list[dict[str, Any]] = []
    for action in actions:
        label = action.get("label", "")
        if "merchant" in label or "shop" in label:
            continue
        if "redDoor MT10" in label or "event MT10" in label or "skeletonCaptain" in label:
            continue
        if state.floor_id == "MT10" and (
            "bluePriest MT10:8,11" in label
            or "yellowDoor MT10:9,9" in label
            or "yellowDoor MT10:11,9" in label
            or "redGem MT10:10,6" in label
            or "bluePotion MT10:11,11" in label
        ):
            continue
        if any(token in label for token in ("greenSlime MT1:9,11", "king MT1:7,10")):
            continue
        if current_floor > 1 and "downFloor" in label:
            kept.append(action)
            continue
        if current_floor <= 7 and "upFloor" in label:
            kept.append(action)
            continue
        if (
            "yellowDoor" in label
            and "MT10" not in label
            and state.items.get("yellowKey", 0) > 1
        ):
            kept.append(action)
            continue
        if (
            "blueDoor" in label
            and "MT10" not in label
            and state.items.get("blueKey", 0) > 0
        ):
            kept.append(action)
            continue
        if any(token in label for token in refill_tokens):
            kept.append(action)
            continue
        if label.startswith("fight"):
            allowed_blockers = (
                "bluePriest MT4:2,5",
                "greenSlime MT4:3,10",
                "greenSlime MT4:8,11",
            )
            if not any(token in label for token in allowed_blockers):
                continue
            target = action.get("target") or []
            enemy_id = (
                sim.block_id(sim.tile(state, int(target[1]), int(target[2]), str(target[0])))
                if len(target) == 3
                else None
            )
            info = sim.damage_info(state, enemy_id) if enemy_id else None
            if info is not None and int(info["damage"]) <= 80:
                kept.append(action)
                continue
        if current_floor <= 3 and ("downFloor" in label or label.startswith("go ")):
            kept.append(action)
            continue
    return kept or actions


def restrict_mt10_right_resource_actions(
    sim: MotaSimulator,
    state: MotaState,
    actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if mt10_right_resources_ready(sim, state):
        return actions
    if not post_mid_refill_ready(sim, state):
        return restrict_actions_for_target(sim, state, actions, "post_mid_refill")

    def keep(patterns: tuple[str, ...]) -> list[dict[str, Any]]:
        kept = [action for action in actions if any(p in action.get("label", "") for p in patterns)]
        return kept or actions

    current_floor = floor_index(sim, state)
    if current_floor < 10:
        return keep(("upFloor",))
    if state.floor_id == "MT10":
        return keep((
            "bluePriest MT10:8,11",
            "yellowDoor MT10:9,9",
            "skeleton MT10:9,6",
            "redGem MT10:10,6",
            "yellowDoor MT10:11,9",
            "bluePotion MT10:11,11",
            "downFloor MT10:1,11",
        ))
    return actions


def restrict_guard_low_refill_actions(
    sim: MotaSimulator,
    state: MotaState,
    actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Narrow repair stage for the post-10F-first-resource refill pattern.

    The successful benchmark structure descends after the first 10F stat item
    and uses low-floor refills before fighting the red-key guard route.  A broad
    guard-ready beam spends most work on local 8-10F detours, so this stage keeps
    only descent, explicit resources, and cheap blockers until the margin is
    non-negative.
    """

    if guard_ready(sim, state) or red_key_route_margin(sim, state) >= 0:
        return actions
    if mt10_resource_progress(sim, state) <= 0:
        return restrict_actions_for_target(sim, state, actions, "mt10_first_resource")

    current_floor = floor_index(sim, state)

    def action_damage(action: dict[str, Any]) -> int:
        target = action.get("target") or []
        if len(target) != 3:
            return 0
        enemy_id = sim.block_id(sim.tile(state, int(target[1]), int(target[2]), str(target[0])))
        info = sim.damage_info(state, enemy_id) if enemy_id else None
        return 10_000 if info is None else int(info["damage"])

    high_value_tokens = (
        "Potion",
        "redGem",
        "blueGem",
        "yellowKey",
        "blueKey",
    )
    kept: list[dict[str, Any]] = []
    for action in actions:
        label = action.get("label", "")
        if "merchant" in label or "shop" in label:
            continue
        if "skeletonCaptain" in label or "redDoor MT10" in label or "event MT10" in label:
            continue
        if current_floor > 2 and "downFloor" in label:
            kept.append(action)
            continue
        if any(token in label for token in high_value_tokens):
            kept.append(action)
            continue
        if "yellowDoor" in label and state.items.get("yellowKey", 0) > 1 and current_floor <= 4:
            kept.append(action)
            continue
        if "blueDoor" in label and state.items.get("blueKey", 0) > 0 and current_floor <= 4:
            kept.append(action)
            continue
        if label.startswith("fight") and action_damage(action) <= 36 and current_floor <= 4:
            kept.append(action)
            continue
        if current_floor <= 2 and ("downFloor" in label or "upFloor" in label or label.startswith("go ")):
            kept.append(action)
            continue
    return kept or actions


def action_context_key(state: MotaState, action: dict[str, Any]) -> tuple[Any, ...]:
    return (
        state.floor_id,
        state.x // 2,
        state.y // 2,
        state.atk,
        state.defense,
        state.items.get("yellowKey", 0),
        state.items.get("blueKey", 0),
        state.items.get("redKey", 0),
        bool(state.flags.get("8")),
        bool(state.flags.get("10f机关")),
        action.get("label", ""),
    )


def action_penalty_for(
    state: MotaState,
    action: dict[str, Any],
    action_penalties: dict[tuple[Any, ...], float] | None = None,
    label_penalties: dict[str, float] | None = None,
) -> float:
    penalty = 0.0
    if action_penalties:
        penalty += action_penalties.get(action_context_key(state, action), 0.0)
    if label_penalties:
        label = action.get("label", "")
        for pattern, value in label_penalties.items():
            if pattern in label:
                penalty += value
    return penalty


def transition_score_parts(
    sim: MotaSimulator,
    before: MotaState,
    after: MotaState,
    action: dict[str, Any],
    target_stage: str,
    action_score: dict[str, Any],
    model_weight: float,
    action_bias_weight: float,
    potential_weight: float,
    fast_score_weight: float,
    distance_weight: float,
    path_step_penalty: float,
    macro_step_penalty: float,
    success_bonus: float,
    adaptive_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    score_target_stage = scoring_stage_name(target_stage)
    score_stage = _effective_score_stage(sim, before, score_target_stage)
    if potential_weight:
        before_phi = stage_potential(sim, before, stage=score_stage)
        after_phi = stage_potential(sim, after, stage=score_stage)
    else:
        before_phi = 0.0
        after_phi = 0.0
    if fast_score_weight:
        before_fast = _node_score(sim, before, score_target_stage, None, 0.0)
        after_fast = _node_score(sim, after, score_target_stage, None, 0.0)
    else:
        before_fast = 0.0
        after_fast = 0.0
    model = action_score["model_logit_centered"] * model_weight
    bias = action_score["action_bias"] * action_bias_weight
    potential_delta = (after_phi - before_phi) * potential_weight
    fast_delta = (after_fast - before_fast) * fast_score_weight
    distance_delta = (
        stage_distance_potential(sim, after, target_stage)
        - stage_distance_potential(sim, before, target_stage)
    ) * distance_weight
    path_penalty = len(action.get("path", [])) * path_step_penalty
    macro_penalty = macro_step_penalty
    terminal_bonus = success_bonus if beam_stage_complete(sim, after, target_stage) else 0.0
    hp_debt = -max(0, -after.hp) * 0.05
    irreversible_risk = red_key_irreversible_risk(sim, before, after, action, target_stage)
    adaptive_delta = adaptive_reward_delta(sim, before, after, target_stage, adaptive_weights)
    delayed_refill = delayed_refill_penalty(
        sim=sim,
        before=before,
        after=after,
        action=action,
        target_stage=target_stage,
        weights=adaptive_weights,
    )
    total = (
        model
        + bias
        + potential_delta
        + fast_delta
        + distance_delta
        + terminal_bonus
        + hp_debt
        + irreversible_risk
        + adaptive_delta
        + delayed_refill
        - path_penalty
        - macro_penalty
    )
    return {
        "total": total,
        "model": model,
        "action_bias": bias,
        "potential_delta": potential_delta,
        "fast_delta": fast_delta,
        "distance_delta": distance_delta,
        "success_bonus": terminal_bonus,
        "hp_debt": hp_debt,
        "irreversible_risk": irreversible_risk,
        "adaptive_delta": adaptive_delta,
        "delayed_refill_penalty": delayed_refill,
        "path_penalty": -path_penalty,
        "macro_penalty": -macro_penalty,
        "raw_phi_before": before_phi,
        "raw_phi_after": after_phi,
        "raw_fast_before": before_fast,
        "raw_fast_after": after_fast,
    }


def adaptive_reward_delta(
    sim: MotaSimulator,
    before: MotaState,
    after: MotaState,
    target_stage: str,
    weights: dict[str, float] | None,
) -> float:
    if not weights:
        return 0.0
    before_yk = before.items.get("yellowKey", 0)
    after_yk = after.items.get("yellowKey", 0)
    before_bk = before.items.get("blueKey", 0)
    after_bk = after.items.get("blueKey", 0)
    value = 0.0
    value += weights.get("hp_delta", 0.0) * (after.hp - before.hp)
    value += weights.get("atk_delta", 0.0) * (after.atk - before.atk)
    value += weights.get("def_delta", 0.0) * (after.defense - before.defense)
    value += weights.get("yellow_key_delta", 0.0) * (after_yk - before_yk)
    value += weights.get("blue_key_delta", 0.0) * (after_bk - before_bk)
    value += weights.get("yellow_key_level", 0.0) * min(after_yk, 6)
    value += weights.get("blue_key_level", 0.0) * min(after_bk, 2)
    if target_stage in {"post_mid_refill", "mt10_right_resources", "guard_low_refill", "guard_ready", "red_key"}:
        before_route_margin = red_key_route_margin(sim, before)
        after_route_margin = red_key_route_margin(sim, after)
        value += weights.get("guard_margin_delta", 0.0) * (
            yellow_guard_margin(sim, after) - yellow_guard_margin(sim, before)
        )
        value += weights.get("guard_margin_level", 0.0) * max(
            -600.0,
            min(float(yellow_guard_margin(sim, after)), 600.0),
        )
        value += weights.get("red_key_route_margin_delta", 0.0) * (
            after_route_margin - before_route_margin
        )
        value += weights.get("red_key_route_margin_level", 0.0) * max(
            -800.0,
            min(float(after_route_margin), 800.0),
        )
        if before_yk > 0 and after_yk == 0 and not red_key_taken(sim, after):
            value -= weights.get("last_yellow_key_spent", 0.0)
    return value


def delayed_refill_penalty(
    sim: MotaSimulator,
    before: MotaState,
    after: MotaState,
    action: dict[str, Any],
    target_stage: str,
    weights: dict[str, float] | None,
) -> float:
    """Penalize consuming refill resources before the first 10F resource.

    The current failure mode is not that the beam cannot reach 10F; it reaches
    10F after spending low-floor potions that are needed for the red-key guard
    route.  This term is stage-local and weight-controlled so pure experiments
    can search the strength of this preference without using a hand route as a
    label.
    """

    if not weights:
        return 0.0
    if target_stage not in {
        "pre_shield_gems",
        "lower_gems",
        "low_gems",
        "mid_gems",
        "mt8_gems",
        "all_gems",
        "mt10_blue_ready",
        "mt10_yellow_ready",
        "mt10_ready",
        "mt10_first_resource",
        "mt10_resources",
        "guard_low_refill",
        "guard_ready",
    }:
        return 0.0
    if mt10_resource_progress(sim, before) > 0:
        return 0.0

    label = action.get("label", "")
    if "Potion" not in label:
        return 0.0
    if "MT10" in label:
        return 0.0

    penalty = weights.get("pre_mt10_potion_penalty", 0.0)
    if "bluePotion" in label:
        penalty += weights.get("pre_mt10_blue_potion_penalty", 0.0)
    if any(token in label for token in ("MT1", "MT2", "MT3", "MT4")):
        penalty += weights.get("pre_mt10_low_floor_potion_penalty", 0.0)
    for token in (
        "bluePotion MT1:10,11",
        "bluePotion MT2:3,10",
        "bluePotion MT2:4,10",
        "bluePotion MT2:3,11",
        "bluePotion MT4:11,2",
    ):
        if token in label:
            penalty += weights.get("pre_mt10_key_refill_potion_penalty", 0.0)

    hp_floor = weights.get("pre_mt10_potion_hp_floor", 0.0)
    if hp_floor > 0.0 and before.hp < hp_floor:
        penalty *= weights.get("pre_mt10_potion_low_hp_multiplier", 0.25)
    return -penalty


def red_key_irreversible_risk(
    sim: MotaSimulator,
    before: MotaState,
    after: MotaState,
    action: dict[str, Any],
    target_stage: str,
) -> float:
    if target_stage != "red_key" or red_key_taken(sim, after):
        return 0.0
    label = action.get("label", "")
    risk = 0.0
    if "yellowGuard" in label and red_key_route_margin(sim, after) < 0:
        risk -= 220.0
    if before.items.get("yellowKey", 0) > 0 and after.items.get("yellowKey", 0) == 0:
        if not any(token in label for token in ["redKey", "yellowGuard", "specialDoor"]):
            risk -= 45.0
    if after.floor_id == "MT10" and after.items.get("yellowKey", 0) == 0 and not red_key_taken(sim, after):
        risk -= 60.0
    return risk


def select_next_beam(
    sim: MotaSimulator,
    nodes: list[BeamNode],
    target_stage: str,
    beam_width: int,
    max_per_diversity_key: int,
) -> list[BeamNode]:
    ordered = sorted(
        nodes,
        key=lambda node: (
            node.score
            + _node_score(sim, node.state, scoring_stage_name(target_stage), None, 0.0) * 0.0003
            + stage_distance_potential(sim, node.state, target_stage) * 0.05
        ),
        reverse=True,
    )
    selected: list[BeamNode] = []
    per_bucket: Counter[tuple[Any, ...]] = Counter()
    dominance: dict[tuple[Any, ...], list[tuple[int, ...]]] = {}
    for node in ordered:
        bucket = diversity_key(node.state)
        if max_per_diversity_key > 0 and per_bucket[bucket] >= max_per_diversity_key:
            continue
        dkey = sim.dominance_key(node.state)
        vec = _stage_resource_vector(sim, node.state)
        existing = dominance.get(dkey, [])
        if any(all(old_v >= new_v for old_v, new_v in zip(old, vec)) for old in existing):
            continue
        dominance[dkey] = [
            old for old in existing if not all(new_v >= old_v for new_v, old_v in zip(vec, old))
        ] + [vec]
        selected.append(node)
        per_bucket[bucket] += 1
        if len(selected) >= beam_width:
            break
    return selected or ordered[:beam_width]


def choose_better_node(
    sim: MotaSimulator,
    current: BeamNode,
    candidate: BeamNode,
    target_stage: str,
) -> BeamNode:
    current_success = beam_stage_complete(sim, current.state, target_stage)
    candidate_success = beam_stage_complete(sim, candidate.state, target_stage)
    if candidate_success and not current_success:
        return candidate
    if current_success and not candidate_success:
        return current
    current_value = (
        current.score
        + _node_score(sim, current.state, scoring_stage_name(target_stage), None, 0.0) * 0.0008
        + stage_distance_potential(sim, current.state, target_stage) * 0.1
    )
    candidate_value = (
        candidate.score
        + _node_score(sim, candidate.state, scoring_stage_name(target_stage), None, 0.0) * 0.0008
        + stage_distance_potential(sim, candidate.state, target_stage) * 0.1
    )
    return candidate if candidate_value > current_value else current


def tile_id_at(sim: MotaSimulator, state: MotaState, floor_id: str, x: int, y: int) -> str | None:
    return sim.block_id(sim.tile(state, x, y, floor_id))


def mt10_mid_opened(sim: MotaSimulator, state: MotaState) -> bool:
    return (
        tile_id_at(sim, state, "MT10", 3, 9) != "yellowDoor"
        and tile_id_at(sim, state, "MT10", 4, 11) != "bluePriest"
    )


def red_key_entry_opened(sim: MotaSimulator, state: MotaState) -> bool:
    return tile_id_at(sim, state, "MT8", 10, 7) != "yellowDoor"


def post_mid_refill_ready(sim: MotaSimulator, state: MotaState) -> bool:
    if not mt10_mid_opened(sim, state):
        return False
    return state.hp >= 760 and state.items.get("yellowKey", 0) >= 3


def mt10_right_resources_ready(sim: MotaSimulator, state: MotaState) -> bool:
    if tile_id_at(sim, state, "MT10", 10, 6) == "redGem":
        return False
    if tile_id_at(sim, state, "MT10", 11, 11) == "bluePotion":
        return False
    return red_key_entry_opened(sim, state) or state.items.get("yellowKey", 0) >= 1


def scoring_stage_name(target_stage: str) -> str:
    if target_stage in {"mt10_first_resource", "mt10_mid_open", "mt10_right_resources"}:
        return "mt10_resources"
    if target_stage in {"guard_low_refill", "post_mid_refill"}:
        return "guard_ready"
    return target_stage


def beam_stage_complete(sim: MotaSimulator, state: MotaState, target_stage: str) -> bool:
    if target_stage == "mt10_first_resource":
        return mt10_resource_progress(sim, state) >= 1
    if target_stage == "mt10_mid_open":
        return mt10_mid_opened(sim, state)
    if target_stage == "post_mid_refill":
        return post_mid_refill_ready(sim, state)
    if target_stage == "mt10_right_resources":
        return mt10_right_resources_ready(sim, state)
    if target_stage == "guard_low_refill":
        return guard_ready(sim, state)
    return stage_complete(sim, state, target_stage)


def stage_distance_potential(sim: MotaSimulator, state: MotaState, target_stage: str) -> float:
    """Small navigation potential for stage decoding.

    This is intentionally separate from PBRS. It only helps the decoder choose
    between beam frontiers when all routes are still incomplete.
    """

    requested_stage = target_stage
    target_stage = _effective_score_stage(sim, state, scoring_stage_name(target_stage))
    current_floor = floor_index(sim, state)
    if requested_stage == "mt10_mid_open":
        if mt10_mid_opened(sim, state):
            return 900.0 + min(state.hp, 900) * 0.12 + state.items.get("yellowKey", 0) * 25.0
        door_open = tile_id_at(sim, state, "MT10", 3, 9) != "yellowDoor"
        priest_removed = tile_id_at(sim, state, "MT10", 4, 11) != "bluePriest"
        floor_term = min(current_floor, 10) * 62.0 - abs(current_floor - 10) * 80.0
        if state.floor_id == "MT10":
            floor_term += 260.0
        progress = mt10_resource_progress(sim, state) * 160.0
        progress += 220.0 if door_open else 0.0
        progress += 360.0 if priest_removed else 0.0
        key_term = min(state.items.get("yellowKey", 0), 3) * 35.0
        hp_term = min(state.hp, 700) * 0.22
        return floor_term + progress + key_term + hp_term
    if requested_stage == "post_mid_refill":
        if post_mid_refill_ready(sim, state):
            return 1050.0 + min(state.hp, 1100) * 0.10
        if not mt10_mid_opened(sim, state):
            return stage_distance_potential(sim, state, "mt10_mid_open") - 120.0
        yellow_keys = state.items.get("yellowKey", 0)
        floor_term = -abs(current_floor - 3) * 34.0 + max(0, 10 - current_floor) * 18.0
        hp_term = min(state.hp, 1000) * 0.72
        key_term = min(yellow_keys, 3) * 170.0
        if yellow_keys >= 3:
            key_term += 120.0
        return floor_term + hp_term + key_term + state.atk * 4.0 + state.defense * 4.0
    if requested_stage == "mt10_right_resources":
        if mt10_right_resources_ready(sim, state):
            return 1200.0 + min(state.hp, 1200) * 0.08
        if not post_mid_refill_ready(sim, state):
            return stage_distance_potential(sim, state, "post_mid_refill") - 160.0
        red_taken = tile_id_at(sim, state, "MT10", 10, 6) != "redGem"
        potion_taken = tile_id_at(sim, state, "MT10", 11, 11) != "bluePotion"
        floor_term = min(current_floor, 10) * 58.0 - abs(current_floor - 10) * 76.0
        if state.floor_id == "MT10":
            floor_term += 300.0
        progress = (280.0 if red_taken else 0.0) + (260.0 if potion_taken else 0.0)
        reserve_ok = red_key_entry_opened(sim, state) or state.items.get("yellowKey", 0) >= 1
        reserve_term = 180.0 if reserve_ok else -450.0
        key_term = min(state.items.get("yellowKey", 0), 3) * 90.0
        return floor_term + progress + reserve_term + key_term + min(state.hp, 1000) * 0.22
    if target_stage == "pre_shield_gems":
        sword_bonus = 140.0 if state.flags.get("nowWeapon") == "sword1" else 0.0
        stat_bonus = max(0, state.atk - 20) * 30.0 + max(0, state.defense - 10) * 28.0
        return sword_bonus + stat_bonus + min(current_floor, 5) * 12.0
    if target_stage == "lower_gems":
        if lower_attack_defense_gems_taken(sim, state):
            return 1050.0
        remaining = remaining_lower_attack_defense_gems(sim, state)
        key_term = min(state.items.get("yellowKey", 0), 4) * 32.0 + state.items.get("blueKey", 0) * 55.0
        stat_term = state.atk * 4.0 + state.defense * 4.0 + min(state.hp, 1000) * 0.10
        return (12 - remaining) * 85.0 + key_term + stat_term - max(0, current_floor - 9) * 25.0
    if target_stage == "mt8_gems":
        floor_term = min(current_floor, 8) * 28.0 - abs(current_floor - 8) * 36.0
        if state.floor_id == "MT8":
            floor_term += 140.0 - abs(state.x - 6) * 4.0 - abs(state.y - 9) * 4.0
        key_term = min(state.items.get("yellowKey", 0), 5) * 24.0 + state.items.get("blueKey", 0) * 45.0
        stat_term = state.atk * 3.0 + state.defense * 3.0 + min(state.hp, 900) * 0.08
        return floor_term + key_term + stat_term
    if target_stage == "all_gems":
        if all_attack_defense_gems_taken(sim, state):
            return 1200.0
        remaining = remaining_attack_defense_gems(sim, state)
        mt10_progress = mt10_resource_progress(sim, state) * 180.0
        key_term = min(state.items.get("yellowKey", 0), 5) * 35.0 + min(state.items.get("blueKey", 0), 1) * 70.0
        stat_term = state.atk * 5.0 + state.defense * 5.0 + min(state.hp, 1000) * 0.12
        return (14 - remaining) * 90.0 + mt10_progress + key_term + stat_term - abs(current_floor - 10) * 12.0
    if target_stage == "red_key":
        if red_key_taken(sim, state):
            return 1000.0
        guard_margin = red_key_route_margin(sim, state)
        if guard_margin < 0:
            mt10_resource = 0.0
            for floor_id, x, y, item_id, value in [
                ("MT10", 10, 6, "redGem", 260.0),
                ("MT10", 2, 6, "blueGem", 180.0),
                ("MT10", 11, 11, "bluePotion", 90.0),
            ]:
                if sim.block_id(sim.tile(state, x, y, floor_id)) != item_id:
                    mt10_resource += value
            prep_floor = -abs(current_floor - 10) * 28.0
            resource_margin = max(-800.0, min(float(guard_margin), 260.0)) * 0.9
            stat_reserve = state.atk * 8.0 + state.defense * 6.0 + min(state.hp, 900) * 0.08
            return prep_floor + resource_margin + stat_reserve + mt10_resource
        target_floor = 8
        floor_gap = abs(current_floor - target_floor)
        distance = floor_gap * 22.0
        if state.floor_id == "MT8":
            distance += abs(state.x - 10) + abs(state.y - 2)
        progress = 0.0
        for x, y in [(9, 5), (11, 5)]:
            if sim.block_id(sim.tile(state, x, y, "MT8")) != "yellowGuard":
                progress += 110.0
        if sim.block_id(sim.tile(state, 10, 4, "MT8")) != "specialDoor":
            progress += 160.0
        return progress - distance
    if target_stage == "guard_stats_ready":
        route_margin = red_key_route_margin(sim, state)
        stat_term = (
            max(0, state.atk - 24) * 95.0
            + max(0, state.defense - 22) * 90.0
            + min(state.hp, 700) * 0.18
            + min(state.items.get("yellowKey", 0), 4) * 44.0
        )
        floor_term = -abs(current_floor - 8) * 18.0
        margin_term = max(-900.0, min(float(route_margin), 400.0)) * 0.45
        return stat_term + floor_term + margin_term
    if target_stage == "guard_route_ready":
        route_margin = red_key_route_margin(sim, state)
        floor_term = -abs(current_floor - 8) * 22.0
        progress = 0.0
        for x, y in [(9, 5), (11, 5)]:
            if sim.block_id(sim.tile(state, x, y, "MT8")) != "yellowGuard":
                progress += 130.0
        if sim.block_id(sim.tile(state, 10, 4, "MT8")) != "specialDoor":
            progress += 180.0
        margin_term = max(-900.0, min(float(route_margin), 700.0)) * 0.55
        return floor_term + progress + margin_term + min(state.hp, 900) * 0.10
    if target_stage == "mt10_blue_ready":
        if mt10_blue_ready(sim, state):
            return 650.0
        return min(current_floor, 9) * 22.0 + min(state.items.get("blueKey", 0), 1) * 120.0 + min(state.hp, 320) * 0.35
    if target_stage in {"mt10_yellow_ready", "mt10_ready"}:
        if mt10_access_ready(sim, state):
            return 760.0
        key_term = min(state.items.get("yellowKey", 0), 5) * 45.0 + min(state.items.get("blueKey", 0), 1) * 80.0
        hp_term = min(state.hp, 320) * 0.6
        return min(current_floor, 9) * 24.0 + key_term + hp_term
    if target_stage == "mt10_resources":
        if mt10_resources_taken(sim, state):
            return 900.0
        progress = mt10_resource_progress(sim, state) * 220.0
        yellow_keys = state.items.get("yellowKey", 0)
        blue_keys = state.items.get("blueKey", 0)
        access_ready = mt10_access_ready(sim, state)
        blue_ready = mt10_blue_ready(sim, state)
        first_resource_ready = blue_ready and yellow_keys >= 3
        floor_gap = abs(current_floor - 10)
        floor_term = -floor_gap * 32.0
        hp_term = min(state.hp, 900) * 0.22
        stat_term = state.atk * 4.0 + state.defense * 4.0
        if blue_ready:
            floor_term += min(current_floor, 10) * 32.0
        if access_ready or first_resource_ready:
            # Once the 10F entry budget is available, stop rewarding more
            # low-floor hoarding and aggressively prefer climbing.
            floor_term += min(current_floor, 10) * 90.0 - floor_gap * 90.0
            hp_term += min(state.hp, 380) * 0.34
        if state.floor_id == "MT10":
            floor_term += 360.0
        key_term = min(yellow_keys, 5) * 35.0 + min(blue_keys, 1) * 80.0
        if yellow_keys >= MT10_RESOURCE_YELLOW_KEY_TARGET and blue_ready:
            key_term += 220.0
        elif first_resource_ready:
            key_term += 130.0
        if mt10_resource_progress(sim, state) > 0:
            progress += mt10_resource_progress(sim, state) * 120.0
        return progress + floor_term + key_term + hp_term + stat_term
    if target_stage in {"guard_low_refill", "guard_ready"}:
        if guard_ready(sim, state):
            return 700.0
        route_margin = red_key_route_margin(sim, state)
        mt10_progress = mt10_resource_progress(sim, state)
        mt10_resource = 0.0
        for floor_id, x, y, item_id, value in [
            ("MT10", 10, 6, "redGem", 320.0),
            ("MT10", 2, 6, "blueGem", 260.0),
            ("MT10", 11, 11, "bluePotion", 240.0),
        ]:
            if sim.block_id(sim.tile(state, x, y, floor_id)) != item_id:
                mt10_resource += value
        if route_margin < 0:
            if mt10_progress > 0:
                # After the first 10F stat item, the good route should usually
                # descend and refill low-floor resources before committing to
                # the red-key guards.  Keeping the floor potential centered on
                # 10F makes the beam waste HP on local 10F fights.
                floor_term = -abs(current_floor - 1) * 34.0 + max(0, 10 - current_floor) * 24.0
            else:
                floor_term = -abs(current_floor - 10) * 34.0
                if mt10_access_ready(sim, state):
                    floor_term += min(current_floor, 10) * 42.0
            hp_term = min(state.hp, 1000) * 0.28
            stat_term = state.atk * 7.0 + state.defense * 7.0
            key_term = min(state.items.get("yellowKey", 0), 5) * 32.0 + min(state.items.get("blueKey", 0), 1) * 65.0
            margin_term = max(-1200.0, min(float(route_margin), 600.0)) * 0.85
            return floor_term + hp_term + stat_term + key_term + margin_term + mt10_resource
        return (
            min(current_floor, 9) * 12.0
            + state.atk * 4.0
            + state.defense * 4.0
            + min(state.hp, 1000) * 0.20
            + min(float(route_margin), 800.0) * 0.35
        )
    if target_stage == "trap":
        if state.flags.get("10f机关"):
            return 800.0
        return -abs(current_floor - 10) * 20.0
    if target_stage == "boss_ready":
        if boss_ready(sim, state):
            return 1000.0
        return (500.0 if red_key_taken(sim, state) else 0.0) - abs(current_floor - 10) * 18.0 + _boss_margin(sim, state) * 0.35
    if target_stage == "boss":
        if state.flags.get("10f战胜骷髅队长"):
            return 1200.0
        return -abs(current_floor - 10) * 20.0 + _boss_margin(sim, state) * 0.2
    if target_stage == "boss_all_gems":
        if stage_complete(sim, state, "boss_all_gems"):
            return 1400.0
        boss_term = 500.0 if state.flags.get("10f战胜骷髅队长") else 0.0
        remaining = remaining_attack_defense_gems(sim, state)
        gem_term = (14 - remaining) * 70.0
        mt10_term = 120.0 if state.floor_id == "MT10" else -abs(current_floor - 10) * 18.0
        return boss_term + gem_term + mt10_term + _boss_margin(sim, state) * 0.15
    return 0.0


def decoder_action_bias(
    sim: MotaSimulator,
    state: MotaState,
    action: dict[str, Any],
    target_stage: str,
) -> float:
    label = action.get("label", "")
    current_floor = floor_index(sim, state)
    delayed_mt7_penalty = delayed_mt7_refill_action_penalty(sim, state, label, target_stage)
    if target_stage in {
        "mt10_first_resource",
        "mt10_mid_open",
        "mt10_right_resources",
        "mt10_resources",
        "trap",
        "boss",
        "boss_ready",
        "boss_all_gems",
    }:
        bias = delayed_mt7_penalty
        yellow_keys = state.items.get("yellowKey", 0)
        blue_ready = mt10_blue_ready(sim, state)
        access_ready = mt10_access_ready(sim, state)
        entry_ready = access_ready or (
            target_stage == "mt10_first_resource" and blue_ready and yellow_keys >= 3
        )
        if target_stage in {"mt10_first_resource", "mt10_resources"} and entry_ready:
            if "upFloor" in label and current_floor < 10:
                bias += 75.0
            if "downFloor" in label and current_floor >= 7:
                bias -= 55.0
            if state.floor_id == "MT8" and mt10_resource_progress(sim, state) == 0:
                if "upFloor MT8:6,1" in label:
                    bias += 260.0
                elif label.startswith("fight") and "MT8" in label:
                    bias -= 120.0
                elif "yellowDoor MT8" in label or "blueDoor MT8" in label:
                    bias -= 80.0
            if "upFloor MT9:1,11" in label:
                bias += 130.0
            if "yellowDoor MT9:6,11" in label:
                bias += 95.0
            if "blueDoor MT9:3,11" in label and blue_ready:
                bias += 90.0
            if "yellowDoor MT10" in label and yellow_keys > 0:
                bias += 80.0
            if any(token in label for token in ("blueGem MT10:2,6", "redGem MT10:10,6", "bluePotion MT10:11,11")):
                bias += 150.0
            if state.floor_id == "MT9" and mt10_resource_progress(sim, state) == 0:
                if any(
                    token in label
                    for token in (
                        "bat MT9:7,10",
                        "redSlime MT9:7,6",
                        "yellowDoor MT9:6,11",
                        "blueDoor MT9:3,11",
                        "upFloor MT9:1,11",
                    )
                ):
                    bias += 90.0
                elif label.startswith("fight") and "MT9" in label:
                    bias -= 36.0
                elif "yellowDoor MT9" in label:
                    bias -= 26.0
        if "upFloor MT9:1,11" in label or "upFloor MT9" in label:
            bias += 18.0
        if "yellowDoor MT9:6,11" in label:
            bias += 24.0
        if "blueDoor MT9:3,11" in label:
            bias += 22.0
        if "downFloor MT10" in label:
            bias -= 18.0
        if "downFloor" in label and current_floor >= 8:
            bias -= 6.0
        if "upFloor" in label and current_floor < 10:
            bias += 4.0
        if "MT10" in label:
            bias += 5.0
        if target_stage == "mt10_mid_open":
            if "upFloor" in label and current_floor < 10:
                bias += 88.0
            if "downFloor" in label and current_floor >= 8 and not mt10_mid_opened(sim, state):
                bias -= 95.0
            if "yellowDoor MT10:3,9" in label:
                bias += 320.0
            if "bluePriest MT10:4,11" in label:
                bias += 360.0
            if any(token in label for token in ("bluePriest MT10:8,11", "redGem MT10:10,6", "bluePotion MT10:11,11")):
                bias -= 110.0
        if target_stage == "mt10_right_resources":
            if "upFloor" in label and current_floor < 10:
                bias += 95.0
            if "downFloor" in label and current_floor >= 8 and not mt10_right_resources_ready(sim, state):
                bias -= 95.0
            if "bluePriest MT10:8,11" in label:
                bias += 320.0
            if "yellowDoor MT10:9,9" in label:
                bias += 210.0
            if "skeleton MT10:9,6" in label:
                bias += 160.0
            if "redGem MT10:10,6" in label:
                bias += 360.0
            if "yellowDoor MT10:11,9" in label:
                if state.items.get("yellowKey", 0) <= 1 and not red_key_entry_opened(sim, state):
                    bias -= 320.0
                else:
                    bias += 190.0
            if "bluePotion MT10:11,11" in label:
                bias += 320.0
            if "yellowDoor MT10:3,9" in label or "bluePriest MT10:4,11" in label:
                bias -= 60.0
        if target_stage in {"mt10_first_resource", "mt10_resources"}:
            if any(token in label for token in ("redGem MT10", "blueGem MT10", "Potion MT10")):
                bias += 20.0
            if "yellowDoor MT10" in label or "blueDoor MT10" in label:
                bias += 12.0
            if label.startswith("fight") and "MT10" in label and "skeletonCaptain" not in label:
                bias += 8.0
            if "redDoor MT10" in label or "event MT10" in label or "skeletonCaptain" in label:
                bias -= 10.0
        elif target_stage == "trap":
            if "redDoor MT10" in label:
                bias += 16.0
            if "event MT10" in label or "specialDoor MT10" in label:
                bias += 20.0
            if "skeletonCaptain" in label:
                bias -= 8.0
        elif target_stage in {"boss", "boss_all_gems"}:
            if state.flags.get("10f机关") and "event MT10" in label:
                bias -= 24.0
            elif "event MT10" in label or "specialDoor MT10" in label:
                bias += 16.0
            if label.startswith("fight") and "MT10" in label and "skeletonCaptain" not in label:
                bias += 10.0
            if "skeletonCaptain" in label:
                bias += 28.0
            if any(token in label for token in ("redGem MT10", "blueGem MT10", "Potion MT10")):
                bias += 32.0 if target_stage == "boss_all_gems" else 7.0
        return bias

    if target_stage in {"guard_low_refill", "post_mid_refill", "guard_ready"}:
        margin = red_key_route_margin(sim, state)
        mt10_progress = mt10_resource_progress(sim, state)
        bias = delayed_mt7_penalty
        if target_stage in {"guard_low_refill", "post_mid_refill"}:
            if "downFloor" in label and current_floor > 1:
                bias += 110.0
            if "upFloor MT1" in label:
                bias += 260.0 if target_stage == "post_mid_refill" else 45.0
            if target_stage == "post_mid_refill" and any(
                token in label
                for token in (
                    "yellowKey MT2:3,4",
                    "yellowKey MT2:4,4",
                    "yellowKey MT2:3,5",
                    "bluePotion MT2:3,10",
                    "bluePotion MT2:4,10",
                    "bluePotion MT2:3,11",
                    "bluePotion MT4:11,2",
                    "yellowKey MT4:3,2",
                    "yellowKey MT4:9,2",
                )
            ):
                bias += 380.0
            if any(token in label for token in ("redPotion MT1", "bluePotion MT2", "bluePotion MT4")):
                bias += 220.0
            if target_stage == "post_mid_refill" and any(
                token in label for token in ("yellowKey MT7:9,11", "yellowKey MT7:5,11", "bluePotion MT7:7,11")
            ):
                bias += 260.0
            if any(token in label for token in ("yellowKey MT1", "yellowKey MT2")):
                bias += 70.0
            if any(token in label for token in ("skeleton MT1:2,4", "skeleton MT1:2,7")):
                bias += 135.0
            if "greenSlime MT1:9,11" in label or "king MT1:7,10" in label:
                bias += -95.0 if target_stage == "post_mid_refill" else 55.0
        if margin >= 0 and target_stage != "post_mid_refill":
            if "yellowGuard" in label:
                bias += 12.0
            return bias
        if "yellowGuard" in label:
            bias -= 26.0 + min(40.0, abs(float(margin)) / 35.0)
        if "redGem" in label:
            bias += 6.0
        if "blueGem" in label:
            bias += 6.0
        if "Potion" in label:
            bias += 8.0
        if "bluePotion" in label:
            bias += 10.0
        if any(token in label for token in ("blueGem MT10:2,6", "redGem MT10:10,6", "bluePotion MT10:11,11")):
            bias += 42.0
        if mt10_progress > 0:
            if "downFloor" in label and current_floor > 1:
                bias += 72.0
            if "upFloor" in label and current_floor < 10:
                bias -= 24.0
            if label.startswith("fight") and "MT10" in label:
                bias -= 70.0
        else:
            if "upFloor" in label and current_floor < 10:
                bias += 6.0
            if "downFloor MT10" in label and not mt10_resources_taken(sim, state):
                bias -= 18.0
        if "yellowDoor MT9:6,11" in label or "blueDoor MT9:3,11" in label or "upFloor MT9:1,11" in label:
            bias += 16.0
        if state.floor_id == "MT9" and mt10_resource_progress(sim, state) == 0:
            if any(
                token in label
                for token in (
                    "bat MT9:7,10",
                    "yellowDoor MT9:6,11",
                    "blueDoor MT9:3,11",
                    "upFloor MT9:1,11",
                )
            ):
                bias += 48.0
            elif label.startswith("fight") and "MT9" in label:
                bias -= 24.0
            elif "yellowDoor MT9" in label:
                bias -= 18.0
        if "yellowDoor MT10" in label and state.items.get("yellowKey", 0) > 0:
            bias += 12.0
        if label.startswith("fight") and "MT10" in label and "skeletonCaptain" not in label:
            target = action.get("target") or []
            enemy_id = sim.block_id(sim.tile(state, int(target[1]), int(target[2]), str(target[0]))) if len(target) == 3 else None
            info = sim.damage_info(state, enemy_id) if enemy_id else None
            if info is not None:
                bias += max(0.0, 420.0 - min(float(info["damage"]), 420.0)) * 0.12
        return bias

    if target_stage != "red_key":
        return delayed_mt7_penalty
    margin = red_key_route_margin(sim, state)
    bias = delayed_mt7_penalty
    if "yellowGuard" in label:
        if margin < 0:
            bias -= 12.0 + min(20.0, abs(float(margin)) / 50.0)
        else:
            bias += 8.0 + min(12.0, float(margin) / 50.0)
    if margin >= 0:
        return bias
    if "redGem" in label:
        bias += 3.0
    if "blueGem" in label:
        bias += 2.0
    if "Potion" in label:
        bias += 1.0
    if "yellowDoor MT9:6,11" in label:
        bias += 6.0
    if "upFloor" in label and current_floor < 10:
        bias += 2.5
    if "downFloor" in label and current_floor == 9:
        bias -= 2.5
    if "downFloor" in label and current_floor > 10:
        bias += 2.0
    return bias


def delayed_mt7_refill_action_penalty(
    sim: MotaSimulator,
    state: MotaState,
    label: str,
    target_stage: str,
) -> float:
    if mt10_mid_opened(sim, state):
        return 0.0
    if target_stage in {"post_mid_refill", "guard_low_refill", "guard_ready"}:
        return 0.0
    protected_tokens = {
        "bluePriest MT7:7,10": -150.0,
        "skeletonSoldier MT7:9,7": -150.0,
        "yellowKey MT7:9,11": -240.0,
        "yellowDoor MT7:5,7": -210.0,
        "bat MT7:5,9": -130.0,
        "yellowKey MT7:5,11": -260.0,
        "bluePotion MT7:7,11": -320.0,
    }
    penalty = 0.0
    for token, value in protected_tokens.items():
        if token in label:
            penalty += value
    return penalty


def reconstruct_beam_route(node: BeamNode) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cursor: BeamNode | None = node
    while cursor is not None and cursor.step is not None:
        rows.append(cursor.step)
        cursor = cursor.parent
    rows.reverse()
    return reindex_route(rows)


def reindex_route(route: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for index, row in enumerate(route):
        row["index"] = index
    return route


def diversity_key(state: MotaState) -> tuple[Any, ...]:
    return (
        state.floor_id,
        state.x // 3,
        state.y // 3,
        state.atk // 2,
        state.defense // 2,
        state.items.get("yellowKey", 0),
        state.items.get("blueKey", 0),
        state.items.get("redKey", 0),
        bool(state.flags.get("10f机关")),
    )


def enrich_summary(sim: MotaSimulator, state: MotaState) -> dict[str, Any]:
    row = state_summary(state)
    row["yellow_guard_margin"] = yellow_guard_margin(sim, state)
    row["red_key_route_margin"] = red_key_route_margin(sim, state)
    row["mt10_resource_progress"] = mt10_resource_progress(sim, state)
    row["boss_margin"] = _boss_margin(sim, state)
    row["target_flags"] = {
        "shield": state.flags.get("nowShield") == "shield1",
        "red_key": state.items.get("redKey", 0) > 0,
        "trap": bool(state.flags.get("10f机关")),
        "boss": bool(state.flags.get("10f战胜骷髅队长")),
    }
    return row


if __name__ == "__main__":
    main()
