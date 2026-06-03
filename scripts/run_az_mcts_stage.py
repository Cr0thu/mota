from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mota_env import MotaSimulator, SimulatorConfig, load_game_data
from mota_env.rewards import boss_route_margin, current_stage_name, reward_scheme_names, stage_complete, stage_names
from mota_solver.az_mcts import (
    AlphaMCTS,
    AlphaMCTSConfig,
    BlendedPolicyValueFn,
    HeuristicPolicyValueFn,
    LearnedRewardValueFn,
    TorchPolicyValueFn,
    uniform_policy_value,
)
from mota_solver.search import state_summary, write_route_jsonl
from mota_rl.train_actor_critic import find_matching_action, load_route


def load_policy_value_fn(args: argparse.Namespace):
    if not args.checkpoint:
        base_fn = uniform_policy_value
        if args.heuristic_prior_mix <= 0:
            return base_fn
        return BlendedPolicyValueFn(
            base_fn,
            HeuristicPolicyValueFn(args.target_stage, temperature=args.heuristic_temperature),
            secondary_mix=args.heuristic_prior_mix,
        )
    import torch

    from mota_rl.graph_policy_value_model import GraphPolicyValueConfig, GraphPolicyValueNet

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GraphPolicyValueNet(
        GraphPolicyValueConfig(
            d_model=args.d_model,
            nhead=args.heads,
            num_layers=args.layers,
            dropout=0.0,
        )
    ).to(device)
    payload = torch.load(args.checkpoint, map_location=device)
    state_dict = payload.get("model_state_dict") or payload.get("policy_state_dict") or payload
    model.load_state_dict(state_dict)
    base_fn = TorchPolicyValueFn(model, device=device, temperature=args.policy_temperature)
    if args.heuristic_prior_mix <= 0:
        return base_fn
    return BlendedPolicyValueFn(
        base_fn,
        HeuristicPolicyValueFn(args.target_stage, temperature=args.heuristic_temperature),
        secondary_mix=args.heuristic_prior_mix,
    )


def load_leaf_value_fn(args: argparse.Namespace):
    if not args.reward_weights_file:
        return None
    payload = json.loads(Path(args.reward_weights_file).read_text(encoding="utf8"))
    return LearnedRewardValueFn(
        payload,
        scale=args.reward_value_scale,
        stage_mode=args.reward_value_stage_mode,
        gamma=args.reward_gamma,
    )


def choose_non_revisit_action(
    sim: MotaSimulator,
    state,
    result,
    visit_counts: dict[tuple[Any, ...], int],
    max_revisits: int,
) -> tuple[dict[str, Any] | None, int | None, int | None]:
    candidates: list[tuple[int, dict[str, Any], int, int]] = []
    for child in result.child_stats:
        probe = state.clone()
        transition = sim.apply_macro_action(probe, child["action"])
        if not transition.ok:
            continue
        candidates.append(
            (
                visit_counts.get(sim.state_key(probe), 0),
                child["action"],
                int(child["action_index"]),
                int(child["action_node_index"]),
            )
        )
    if not candidates:
        return result.action, result.action_index, result.action_node_index
    for seen, action, action_index, node_index in candidates:
        if seen <= max_revisits:
            return action, action_index, node_index
    seen, action, action_index, node_index = min(candidates, key=lambda row: row[0])
    return action, action_index, node_index


def apply_start_route(sim: MotaSimulator, start_route: str, max_actions: int = 0):
    state = sim.reset()
    prefix: list[dict[str, Any]] = []
    if not start_route:
        return state, prefix
    rows = load_route(start_route)
    for index, row in enumerate(rows):
        if max_actions > 0 and index >= max_actions:
            break
        actions = sim.macro_actions(state)
        match = find_matching_action(actions, row.get("action") or {})
        if match is None:
            raise SystemExit(
                f"Cannot replay start route at row {index}: "
                f"{(row.get('action') or {}).get('label', '')}"
            )
        before = state.clone()
        transition = sim.apply_macro_action(state, actions[match])
        prefix.append(
            {
                "index": index,
                "stage": current_stage_name(sim, before),
                "target_stage": "prefix",
                "action": actions[match],
                "before": state_summary(before),
                "after": state_summary(state),
                "ok": transition.ok,
                "message": transition.message,
                "source": "start_route",
            }
        )
        if not transition.ok:
            raise SystemExit(f"Start route failed at row {index}: {transition.message}")
        if state.dead or state.done:
            break
    return state, prefix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="artifacts/data/mota_first10.json")
    parser.add_argument("--out-dir", default="artifacts/runs/az_mcts_stage")
    parser.add_argument("--target-stage", choices=stage_names(), default="sword")
    parser.add_argument("--max-macros", type=int, default=80)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--max-depth", type=int, default=80)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--max-state-revisits", type=int, default=1)
    parser.add_argument("--allow-negative-hp", action="store_true")
    parser.add_argument("--relaxed-min-hp", type=int, default=-2000)
    parser.add_argument(
        "--continue-after-boss",
        action="store_true",
        help="Keep the simulator active after the 10F captain so post-boss resources can be collected.",
    )
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--policy-temperature", type=float, default=1.0)
    parser.add_argument("--heuristic-prior-mix", type=float, default=0.0)
    parser.add_argument("--heuristic-temperature", type=float, default=0.8)
    parser.add_argument("--reward-weights-file", default="")
    parser.add_argument("--reward-value-scale", type=float, default=25_000.0)
    parser.add_argument("--reward-value-stage-mode", choices=("target", "current"), default="target")
    parser.add_argument("--reward-gamma", type=float, default=0.99)
    parser.add_argument("--final-action-visit-weight", type=float, default=1.0)
    parser.add_argument("--final-action-value-weight", type=float, default=0.0)
    parser.add_argument("--final-action-prior-weight", type=float, default=0.0)
    parser.add_argument("--root-dirichlet-alpha", type=float, default=0.0)
    parser.add_argument("--root-exploration-fraction", type=float, default=0.0)
    parser.add_argument(
        "--mcts-edge-reward-scheme",
        choices=("none", *reward_scheme_names()),
        default="none",
        help="Use single-player MDP backup G=r+gamma*V with this edge reward scheme.",
    )
    parser.add_argument("--mcts-edge-reward-scale", type=float, default=100.0)
    parser.add_argument("--mcts-edge-reward-clip", type=float, default=1.0)
    parser.add_argument(
        "--disable-stage-action-filter",
        action="store_true",
        help=(
            "Expand all legal macro actions. This is required when evaluating a "
            "policy trained on the full graph action space; otherwise the staged "
            "hand filter can remove actions the policy learned to select."
        ),
    )
    parser.add_argument(
        "--hp-aware-success-value",
        action="store_true",
        help="Score successful terminal states by remaining HP/keys instead of a flat success value.",
    )
    parser.add_argument("--hp-success-base", type=float, default=0.55)
    parser.add_argument("--hp-success-scale", type=float, default=1000.0)
    parser.add_argument("--start-route", default="")
    parser.add_argument("--start-route-max-actions", type=int, default=0)
    parser.add_argument(
        "--write-prefix",
        action="store_true",
        help="Include start-route prefix rows in the route output. Default writes only the decoded suffix.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf8")

    sim = MotaSimulator(
        load_game_data(args.data),
        SimulatorConfig(
            allow_negative_hp=args.allow_negative_hp,
            min_hp=args.relaxed_min_hp,
            stop_on_boss=not (args.continue_after_boss or args.target_stage == "boss_all_gems"),
        ),
    )
    state, prefix_route = apply_start_route(
        sim,
        args.start_route,
        max_actions=args.start_route_max_actions,
    )
    policy_value_fn = load_policy_value_fn(args)
    leaf_value_fn = load_leaf_value_fn(args)
    route: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    visit_counts: dict[tuple[Any, ...], int] = {sim.state_key(state): 1}

    for step in range(args.max_macros):
        if stage_complete(sim, state, args.target_stage) or state.dead or state.done:
            break
        mcts = AlphaMCTS(
            sim,
            policy_value_fn=policy_value_fn,
            leaf_value_fn=leaf_value_fn,
            config=AlphaMCTSConfig(
                target_stage=args.target_stage,
                num_simulations=args.simulations,
                c_puct=args.c_puct,
                max_depth=args.max_depth,
                seed=args.seed + step,
                final_action_visit_weight=args.final_action_visit_weight,
                final_action_value_weight=args.final_action_value_weight,
                final_action_prior_weight=args.final_action_prior_weight,
                root_dirichlet_alpha=args.root_dirichlet_alpha,
                root_exploration_fraction=args.root_exploration_fraction,
                hp_aware_success_value=args.hp_aware_success_value,
                hp_success_base=args.hp_success_base,
                hp_success_scale=args.hp_success_scale,
                use_stage_action_filter=not args.disable_stage_action_filter,
                edge_reward_scheme=args.mcts_edge_reward_scheme,
                edge_reward_scale=args.mcts_edge_reward_scale,
                edge_reward_clip=args.mcts_edge_reward_clip,
            ),
        )
        before = state.clone()
        result = mcts.search(before)
        if result.action is None:
            break
        selected_action, selected_action_index, selected_node_index = choose_non_revisit_action(
            sim,
            before,
            result,
            visit_counts,
            args.max_state_revisits,
        )
        if selected_action is None:
            break
        examples.append(
            {
                "index": step,
                "stage": current_stage_name(sim, before),
                "target_stage": args.target_stage,
                "before": state_summary(before),
                "selected_action_index": selected_action_index,
                "selected_action_node_index": selected_node_index,
                "root_value": result.root_value,
                "visit_count": result.visit_count,
                "policy_target": result.policy_target,
                "top_children": result.child_stats[:12],
            }
        )
        transition = sim.apply_macro_action(state, selected_action)
        visit_counts[sim.state_key(state)] = visit_counts.get(sim.state_key(state), 0) + 1
        route.append(
            {
                "index": step,
                "stage": current_stage_name(sim, before),
                "target_stage": args.target_stage,
                "action": selected_action,
                "mcts": {
                    "root_value": result.root_value,
                    "visit_count": result.visit_count,
                    "selected_action_index": selected_action_index,
                    "selected_action_node_index": selected_node_index,
                    "top_children": result.child_stats[:8],
                },
                "before": state_summary(before),
                "after": state_summary(state),
                "ok": transition.ok,
                "message": transition.message,
            }
        )
        if not transition.ok:
            break

    solved = stage_complete(sim, state, args.target_stage)
    output_route = prefix_route + route if args.write_prefix else route
    for index, row in enumerate(output_route):
        row["index"] = index
    write_route_jsonl(output_route, out_dir / f"{args.target_stage}_az_route.jsonl")
    with (out_dir / "az_examples.jsonl").open("w", encoding="utf8") as handle:
        for row in examples:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "target_stage": args.target_stage,
        "target_success": solved,
        "boss_success": bool(state.flags.get("10f战胜骷髅队长")),
        "boss_margin": boss_route_margin(sim, state),
        "macro_steps": len(route),
        "prefix_steps": len(prefix_route),
        "output_route_steps": len(output_route),
        "primitive_steps": state.steps,
        "final": state_summary(state),
        "route": str(out_dir / f"{args.target_stage}_az_route.jsonl"),
        "examples": str(out_dir / "az_examples.jsonl"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
