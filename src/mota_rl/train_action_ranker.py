from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mota_env import MotaSimulator, SimulatorConfig, load_game_data
from mota_env.rewards import stage_complete, stage_names
from mota_rl.policy_features import (
    ACTION_FEATURE_DIM,
    STATE_FEATURE_DIM,
    action_feature_vector,
    state_feature_vector,
)
from mota_rl.train_actor_critic import (
    ActionActorCritic,
    action_logits,
    find_matching_action,
    load_route,
    select_device,
)
from mota_solver.search import state_summary, write_route_jsonl
from mota_solver.staged import _stage_action_bias


@dataclass(frozen=True)
class RouteSample:
    route_index: int
    step: int
    state: Any
    state_features: list[float]
    action_features: list[list[float]]
    action_priors: list[float]
    target_action: dict[str, Any]
    target_index: int
    action_count: int
    target_label: str


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="artifacts/data/mota_first10.json")
    parser.add_argument("--routes", nargs="+", required=True)
    parser.add_argument("--target-stage", choices=stage_names(), default="shield")
    parser.add_argument("--out-dir", default="artifacts/runs/action_ranker")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--margin", type=float, default=2.0)
    parser.add_argument("--margin-coef", type=float, default=0.35)
    parser.add_argument("--prior-weight", type=float, default=0.0)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--load-model", default="")
    parser.add_argument("--allow-negative-hp", action="store_true")
    parser.add_argument("--relaxed-min-hp", type=int, default=-1000)
    args = parser.parse_args()

    try:
        import torch
        from torch import nn
    except Exception as exc:
        raise SystemExit(f"Action ranker requires torch: {exc}") from exc

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(torch, args.device)
    sim_config = SimulatorConfig(
        allow_negative_hp=args.allow_negative_hp,
        min_hp=args.relaxed_min_hp,
    )
    model = ActionActorCritic(args.hidden, nn).to(device)
    if args.load_model:
        checkpoint = torch.load(args.load_model, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    cross_entropy = nn.CrossEntropyLoss()
    samples = build_route_samples(
        route_paths=[Path(route) for route in args.routes],
        data_path=args.data,
        sim_config=sim_config,
        target_stage=args.target_stage,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "samples.json").write_text(
        json.dumps(
            {
                "routes": args.routes,
                "target_stage": args.target_stage,
                "sample_count": len(samples),
                "state_feature_dim": STATE_FEATURE_DIM,
                "action_feature_dim": ACTION_FEATURE_DIM,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf8",
    )

    history: list[dict[str, Any]] = []
    best_score = -1.0
    best_eval_successes = -1
    for epoch in range(args.epochs):
        random.shuffle(samples)
        losses: list[float] = []
        ce_losses: list[float] = []
        margin_losses: list[float] = []
        top1 = 0
        top3 = 0
        mean_rank_total = 0.0
        for sample in samples:
            if sample.target_index >= sample.action_count:
                continue
            state_vec = torch.tensor(
                [sample.state_features],
                dtype=torch.float32,
                device=device,
            )
            action_vecs = torch.tensor(
                [sample.action_features],
                dtype=torch.float32,
                device=device,
            )
            logits, _ = model(state_vec, action_vecs)
            logits = logits.squeeze(0)
            if args.prior_weight:
                prior_logits = torch.tensor(sample.action_priors, dtype=torch.float32, device=device)
                logits = logits + prior_logits * args.prior_weight
            target = torch.tensor([sample.target_index], dtype=torch.long, device=device)
            ce = cross_entropy(logits.unsqueeze(0), target)
            target_logit = logits[sample.target_index]
            negative_mask = torch.ones_like(logits, dtype=torch.bool)
            negative_mask[sample.target_index] = False
            if negative_mask.any():
                margin_loss = torch.relu(args.margin - target_logit + logits[negative_mask]).mean()
            else:
                margin_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
            loss = ce + args.margin_coef * margin_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            ce_losses.append(float(ce.detach().cpu()))
            margin_losses.append(float(margin_loss.detach().cpu()))
            with torch.no_grad():
                order = torch.argsort(logits, descending=True).detach().cpu().tolist()
                rank = int(order.index(sample.target_index) + 1)
                top1 += int(rank == 1)
                top3 += int(rank <= 3)
                mean_rank_total += rank
        train_accuracy = top1 / max(1, len(samples))
        run_eval = (
            args.eval_every > 0
            and (epoch == 0 or (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1)
        )
        if run_eval:
            eval_summary = evaluate_policy(
                data_path=args.data,
                sim_config=sim_config,
                model=model,
                target_stage=args.target_stage,
                max_steps=args.max_steps,
                eval_episodes=args.eval_episodes,
                prior_weight=args.prior_weight,
                device=device,
                torch=torch,
                route_out=out_dir / "eval_route_0.jsonl",
            )
            eval_target_successes: int | None = eval_summary["target_successes"]
        else:
            eval_summary = {
                "target_successes": 0,
                "strict_successes": 0,
                "episodes": args.eval_episodes,
                "rows": [],
            }
            eval_target_successes = None
        row = {
            "epoch": epoch,
            "loss": sum(losses) / max(1, len(losses)),
            "ce_loss": sum(ce_losses) / max(1, len(ce_losses)),
            "margin_loss": sum(margin_losses) / max(1, len(margin_losses)),
            "route_top1": train_accuracy,
            "route_top3": top3 / max(1, len(samples)),
            "route_mean_rank": mean_rank_total / max(1, len(samples)),
            "eval_target_successes": eval_target_successes,
            "eval_episodes": args.eval_episodes,
            "device": str(device),
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        score = eval_summary["target_successes"] + train_accuracy
        if score > best_score or (run_eval and eval_summary["target_successes"] > best_eval_successes):
            if run_eval:
                best_eval_successes = eval_summary["target_successes"]
            best_score = score
            save_model(model, out_dir / "best_model.pt", args.hidden, args.target_stage, epoch, torch)
            if run_eval:
                (out_dir / "best_eval.json").write_text(
                    json.dumps(eval_summary, ensure_ascii=False, indent=2),
                    encoding="utf8",
                )

    save_model(model, out_dir / "model.pt", args.hidden, args.target_stage, args.epochs - 1, torch)
    (out_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf8")
    final_eval = evaluate_policy(
        data_path=args.data,
        sim_config=sim_config,
        model=model,
        target_stage=args.target_stage,
        max_steps=args.max_steps,
        eval_episodes=args.eval_episodes,
        prior_weight=args.prior_weight,
        device=device,
        torch=torch,
        route_out=out_dir / "final_eval_route_0.jsonl",
    )
    (out_dir / "eval.json").write_text(json.dumps(final_eval, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps({"out_dir": str(out_dir), **final_eval}, ensure_ascii=False, indent=2))


def build_route_samples(
    route_paths: list[Path],
    data_path: str,
    sim_config: SimulatorConfig,
    target_stage: str,
) -> list[RouteSample]:
    samples: list[RouteSample] = []
    for route_index, route_path in enumerate(route_paths):
        route = load_route(route_path)
        sim = MotaSimulator(load_game_data(data_path), sim_config)
        state = sim.reset()
        for step, route_row in enumerate(route):
            if stage_complete(sim, state, target_stage):
                break
            actions = sim.macro_actions(state)
            if not actions:
                break
            target_index = find_matching_action(actions, route_row["action"])
            if target_index is None:
                raise SystemExit(
                    f"Route action missing at {route_path}:{step}: "
                    f"{route_row['action'].get('label', '')}"
                )
            samples.append(
                RouteSample(
                    route_index=route_index,
                    step=step,
                    state=state.clone(),
                    state_features=state_feature_vector(sim, state, target_stage),
                    action_features=[
                        action_feature_vector(sim, state, action, target_stage)
                        for action in actions
                    ],
                    action_priors=[
                        _stage_action_bias(action, target_stage, state, sim) / 3000.0
                        for action in actions
                    ],
                    target_action=route_row["action"],
                    target_index=target_index,
                    action_count=len(actions),
                    target_label=actions[target_index].get("label", ""),
                )
            )
            transition = sim.apply_macro_action(state, route_row["action"])
            if not transition.ok:
                break
    if not samples:
        raise SystemExit("No route samples were produced.")
    return samples


def evaluate_policy(
    data_path: str,
    sim_config: SimulatorConfig,
    model,
    target_stage: str,
    max_steps: int,
    eval_episodes: int,
    prior_weight: float,
    device,
    torch,
    route_out: Path | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    first_route: list[dict[str, Any]] = []
    for episode in range(eval_episodes):
        sim = MotaSimulator(load_game_data(data_path), sim_config)
        state = sim.reset()
        route: list[dict[str, Any]] = []
        for step in range(max_steps):
            if stage_complete(sim, state, target_stage) or state.dead or state.done:
                break
            actions = sim.macro_actions(state)
            if not actions:
                break
            with torch.no_grad():
                logits, _ = action_logits(
                    sim=sim,
                    state=state,
                    actions=actions,
                    model=model,
                    target_stage=target_stage,
                    prior_weight=prior_weight,
                    device=device,
                    torch=torch,
                )
                action_index = int(torch.argmax(logits).item())
            action = actions[action_index]
            before = state.clone()
            transition = sim.apply_macro_action(state, action)
            route.append(
                {
                    "index": step,
                    "action": action,
                    "before": state_summary(before),
                    "after": state_summary(state),
                    "transition": transition.message,
                }
            )
            if not transition.ok:
                break
        target_success = stage_complete(sim, state, target_stage)
        rows.append(
            {
                "episode": episode,
                "target_success": target_success,
                "strict_success": target_success and state.hp > 0 and not state.dead,
                "steps": len(route),
                "final": state_summary(state),
            }
        )
        if episode == 0:
            first_route = route
    if route_out is not None and first_route:
        write_route_jsonl(first_route, route_out)
    return {
        "target_successes": sum(int(row["target_success"]) for row in rows),
        "strict_successes": sum(int(row["strict_success"]) for row in rows),
        "episodes": eval_episodes,
        "rows": rows,
    }


def save_model(model, path: Path, hidden: int, target_stage: str, epoch: int, torch) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "hidden": hidden,
                "state_feature_dim": STATE_FEATURE_DIM,
                "action_feature_dim": ACTION_FEATURE_DIM,
                "target_stage": target_stage,
                "epoch": epoch,
            },
        },
        path,
    )


if __name__ == "__main__":
    main()
