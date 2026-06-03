from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mota_env import MotaSimulator, SimulatorConfig, load_game_data
from mota_env.rewards import (
    Rewarder,
    red_key_route_margin,
    reward_scheme_names,
    stage_complete,
    stage_names,
    yellow_guard_margin,
)
from mota_rl.policy_features import (
    ACTION_FEATURE_DIM,
    STATE_FEATURE_DIM,
    action_feature_vector,
    state_feature_vector,
)
from mota_solver.search import state_summary, write_route_jsonl
from mota_solver.staged import _boss_margin, _stage_action_bias


@dataclass
class EpisodeResult:
    route: list[dict[str, Any]]
    total_reward: float
    target_success: bool
    strict_success: bool
    final: dict[str, Any]
    steps: int


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="artifacts/data/mota_first10.json")
    parser.add_argument("--target-stage", choices=stage_names(), default="red_key")
    parser.add_argument("--reward-scheme", choices=reward_scheme_names(), default="stage_pbrs")
    parser.add_argument("--reward-gamma", type=float, default=0.99)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--reward-scale", type=float, default=0.01)
    parser.add_argument("--reward-clip", type=float, default=10.0)
    parser.add_argument("--target-bonus", type=float, default=25.0)
    parser.add_argument("--failure-penalty", type=float, default=0.0)
    parser.add_argument("--hp-debt-penalty", type=float, default=0.02)
    parser.add_argument("--revisit-penalty", type=float, default=1.0)
    parser.add_argument("--max-state-revisits", type=int, default=3)
    parser.add_argument("--loop-action-penalty", type=float, default=35.0)
    parser.add_argument("--stair-loop-window", type=int, default=8)
    parser.add_argument("--prior-weight", type=float, default=0.8)
    parser.add_argument("--normalize-returns", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-stochastic", action="store_true")
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--allow-negative-hp", action="store_true")
    parser.add_argument("--relaxed-min-hp", type=int, default=-1000)
    parser.add_argument("--load-model", default="")
    parser.add_argument("--warm-start-route", default="")
    parser.add_argument("--warm-start-epochs", type=int, default=0)
    parser.add_argument("--warm-start-prior-weight", type=float, default=None)
    parser.add_argument("--bc-anchor-route", default="")
    parser.add_argument("--bc-anchor-coef", type=float, default=0.0)
    parser.add_argument("--bc-anchor-steps", type=int, default=0)
    parser.add_argument("--diagnose-route", default="")
    parser.add_argument("--out-dir", default="artifacts/runs/actor_critic")
    args = parser.parse_args()

    try:
        import torch
        from torch import nn
        from torch.distributions import Categorical
    except Exception as exc:
        raise SystemExit(
            "Actor-critic training requires torch. Use the local Cr0 env or install RL deps.\n"
            f"Import error: {exc}"
        ) from exc

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(torch, args.device)
    sim_config = SimulatorConfig(
        allow_negative_hp=args.allow_negative_hp,
        min_hp=args.relaxed_min_hp,
    )
    rewarder = Rewarder(args.reward_scheme, gamma=args.reward_gamma)
    model = ActionActorCritic(args.hidden, nn).to(device)
    if args.load_model:
        checkpoint = torch.load(args.load_model, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    warm_start_rows: list[dict[str, Any]] = []
    if args.warm_start_route and args.warm_start_epochs > 0:
        warm_start_prior_weight = (
            args.prior_weight if args.warm_start_prior_weight is None else args.warm_start_prior_weight
        )
        warm_start_rows = warm_start_policy(
            route_path=args.warm_start_route,
            data_path=args.data,
            sim_config=sim_config,
            model=model,
            optimizer=optimizer,
            target_stage=args.target_stage,
            epochs=args.warm_start_epochs,
            prior_weight=warm_start_prior_weight,
            device=device,
            torch=torch,
            nn=nn,
        )
        (out_dir / "warm_start.json").write_text(
            json.dumps(warm_start_rows, ensure_ascii=False, indent=2),
            encoding="utf8",
        )
    episode_path = out_dir / "episodes.jsonl"
    bc_anchor_rows = load_route(args.bc_anchor_route) if args.bc_anchor_route else []
    cross_entropy = nn.CrossEntropyLoss()
    best_route: list[dict[str, Any]] = []
    best_score = -10**18
    history: list[dict[str, Any]] = []

    with episode_path.open("w", encoding="utf8") as handle:
        for episode in range(args.episodes):
            sim = MotaSimulator(load_game_data(args.data), sim_config)
            result, tensors = rollout(
                sim=sim,
                model=model,
                rewarder=rewarder,
                target_stage=args.target_stage,
                max_steps=args.max_steps,
                reward_scale=args.reward_scale,
                reward_clip=args.reward_clip,
                target_bonus=args.target_bonus,
                failure_penalty=args.failure_penalty,
                hp_debt_penalty=args.hp_debt_penalty,
                revisit_penalty=args.revisit_penalty,
                max_state_revisits=args.max_state_revisits,
                loop_action_penalty=args.loop_action_penalty,
                stair_loop_window=args.stair_loop_window,
                prior_weight=args.prior_weight,
                device=device,
                torch=torch,
                categorical_cls=Categorical,
                deterministic=False,
            )
            loss_row = update_policy(
                model=model,
                optimizer=optimizer,
                tensors=tensors,
                gamma=args.gamma,
                entropy_coef=args.entropy_coef,
                value_coef=args.value_coef,
                normalize_returns=args.normalize_returns,
                torch=torch,
            )
            if bc_anchor_rows and args.bc_anchor_coef > 0:
                anchor_row = bc_anchor_update(
                    route=bc_anchor_rows,
                    data_path=args.data,
                    sim_config=sim_config,
                    model=model,
                    optimizer=optimizer,
                    target_stage=args.target_stage,
                    coef=args.bc_anchor_coef,
                    max_steps=args.bc_anchor_steps,
                    prior_weight=0.0,
                    device=device,
                    torch=torch,
                    cross_entropy=cross_entropy,
                )
                loss_row = {**loss_row, **anchor_row}
            summary = {
                "episode": episode,
                "total_reward": result.total_reward,
                "target_success": result.target_success,
                "strict_success": result.strict_success,
                "steps": result.steps,
                "final": result.final,
                **loss_row,
            }
            handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
            handle.flush()
            history.append(summary)
            score = (
                result.total_reward
                + int(result.target_success) * 10_000
                + result.final.get("yellow_guard_margin", -10_000)
            )
            if score > best_score:
                best_score = score
                best_route = result.route
                write_route_jsonl(best_route, out_dir / "best_route.jsonl")
                (out_dir / "best_episode.json").write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2),
                    encoding="utf8",
                )
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": {
                            "hidden": args.hidden,
                            "state_feature_dim": STATE_FEATURE_DIM,
                            "action_feature_dim": ACTION_FEATURE_DIM,
                            "target_stage": args.target_stage,
                            "best_episode": episode,
                        },
                    },
                    out_dir / "best_model.pt",
                )
            print(json.dumps(summary, ensure_ascii=False), flush=True)

    eval_rows = []
    for index in range(args.eval_episodes):
        sim = MotaSimulator(load_game_data(args.data), sim_config)
        result, _ = rollout(
            sim=sim,
            model=model,
            rewarder=rewarder,
            target_stage=args.target_stage,
            max_steps=args.max_steps,
            reward_scale=args.reward_scale,
            reward_clip=args.reward_clip,
            target_bonus=args.target_bonus,
            failure_penalty=args.failure_penalty,
            hp_debt_penalty=args.hp_debt_penalty,
            revisit_penalty=args.revisit_penalty,
            max_state_revisits=args.max_state_revisits,
            loop_action_penalty=args.loop_action_penalty,
            stair_loop_window=args.stair_loop_window,
            prior_weight=args.prior_weight,
            device=device,
            torch=torch,
            categorical_cls=Categorical,
            deterministic=not args.eval_stochastic,
        )
        eval_rows.append(
            {
                "episode": index,
                "target_success": result.target_success,
                "strict_success": result.strict_success,
                "total_reward": result.total_reward,
                "steps": result.steps,
                "final": result.final,
            }
        )
        if index == 0 and result.route:
            write_route_jsonl(result.route, out_dir / "eval_route_0.jsonl")
    eval_summary = {
        "episodes": args.eval_episodes,
        "target_successes": sum(int(row["target_success"]) for row in eval_rows),
        "strict_successes": sum(int(row["strict_success"]) for row in eval_rows),
        "rows": eval_rows,
    }
    if args.diagnose_route:
        diagnosis = diagnose_route_policy(
            route_path=args.diagnose_route,
            data_path=args.data,
            sim_config=sim_config,
            model=model,
            target_stage=args.target_stage,
            prior_weight=args.prior_weight,
            device=device,
            torch=torch,
        )
        (out_dir / "route_diagnosis.json").write_text(
            json.dumps(diagnosis, ensure_ascii=False, indent=2),
            encoding="utf8",
        )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "hidden": args.hidden,
                "state_feature_dim": STATE_FEATURE_DIM,
                "action_feature_dim": ACTION_FEATURE_DIM,
                "target_stage": args.target_stage,
            },
        },
        out_dir / "model.pt",
    )
    (out_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf8")
    (out_dir / "eval.json").write_text(json.dumps(eval_summary, ensure_ascii=False, indent=2), encoding="utf8")
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "device": str(device),
                "eval_target_successes": eval_summary["target_successes"],
                "eval_episodes": args.eval_episodes,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


class ActionActorCritic:
    def __new__(cls, hidden: int, nn):
        class _ActionActorCritic(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.state_encoder = nn.Sequential(
                    nn.Linear(STATE_FEATURE_DIM, hidden),
                    nn.LayerNorm(hidden),
                    nn.GELU(),
                    nn.Linear(hidden, hidden),
                    nn.GELU(),
                )
                self.action_encoder = nn.Sequential(
                    nn.Linear(ACTION_FEATURE_DIM, hidden),
                    nn.LayerNorm(hidden),
                    nn.GELU(),
                    nn.Linear(hidden, hidden),
                    nn.GELU(),
                )
                self.score = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, 1))
                self.value = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

            def forward(self, state_features, action_features):
                state_embedding = self.state_encoder(state_features)
                action_embedding = self.action_encoder(action_features)
                expanded_state = state_embedding.unsqueeze(1).expand(-1, action_embedding.shape[1], -1)
                logits = self.score(torch_cat((expanded_state, action_embedding), dim=-1)).squeeze(-1)
                value = self.value(state_embedding).squeeze(-1)
                return logits, value

        return _ActionActorCritic()


def rollout(
    sim: MotaSimulator,
    model,
    rewarder: Rewarder,
    target_stage: str,
    max_steps: int,
    reward_scale: float,
    reward_clip: float,
    target_bonus: float,
    failure_penalty: float,
    hp_debt_penalty: float,
    revisit_penalty: float,
    max_state_revisits: int,
    loop_action_penalty: float,
    stair_loop_window: int,
    prior_weight: float,
    device,
    torch,
    categorical_cls,
    deterministic: bool,
) -> tuple[EpisodeResult, dict[str, list[Any]]]:
    state = sim.reset()
    route: list[dict[str, Any]] = []
    rewards: list[Any] = []
    log_probs: list[Any] = []
    values: list[Any] = []
    entropies: list[Any] = []
    total_reward = 0.0
    visit_counts: dict[tuple[Any, ...], int] = {sim.state_key(state): 1}

    for step_index in range(max_steps):
        if state.done or state.dead or stage_complete(sim, state, target_stage):
            break
        actions = sim.macro_actions(state)
        if not actions:
            break
        state_vec = torch.tensor(
            [state_feature_vector(sim, state, target_stage)],
            dtype=torch.float32,
            device=device,
        )
        action_vecs = torch.tensor(
            [[action_feature_vector(sim, state, action, target_stage) for action in actions]],
            dtype=torch.float32,
            device=device,
        )
        logits, value = action_logits(
            sim=sim,
            state=state,
            actions=actions,
            model=model,
            target_stage=target_stage,
            prior_weight=prior_weight,
            device=device,
            torch=torch,
            state_vec=state_vec,
            action_vecs=action_vecs,
        )
        logits = logits.unsqueeze(0)
        if loop_action_penalty > 0:
            penalties = loop_action_penalties(
                sim=sim,
                state=state,
                actions=actions,
                route=route,
                penalty=loop_action_penalty,
                window=stair_loop_window,
                torch=torch,
                device=device,
            )
            logits = logits - penalties.unsqueeze(0)
        dist = categorical_cls(logits=logits.squeeze(0))
        action_index = int(torch.argmax(logits, dim=-1).item()) if deterministic else int(dist.sample().item())
        action = actions[action_index]
        before = state.clone()
        transition = sim.apply_macro_action(state, action)
        breakdown = rewarder.score(sim, before, state, action, transition)
        reward = _clip(breakdown.total * reward_scale, -reward_clip, reward_clip)
        if state.hp < 0:
            reward -= min(reward_clip, abs(state.hp) * hp_debt_penalty)
        state_key = sim.state_key(state)
        previous_visits = visit_counts.get(state_key, 0)
        if previous_visits and revisit_penalty > 0:
            reward -= min(reward_clip, revisit_penalty * previous_visits)
        visit_counts[state_key] = previous_visits + 1
        if stage_complete(sim, state, target_stage):
            reward += target_bonus
        if state.dead:
            reward -= target_bonus
        total_reward += float(reward)
        route.append(
            {
                "index": step_index,
                "action": action,
                "before": state_summary(before),
                "after": state_summary(state),
                "reward": float(reward),
                "raw_reward": breakdown.total,
                "reward_components": breakdown.components,
            }
        )
        rewards.append(torch.tensor(float(reward), dtype=torch.float32, device=device))
        values.append(value.squeeze(0))
        if deterministic:
            log_probs.append(torch.tensor(0.0, dtype=torch.float32, device=device))
            entropies.append(torch.tensor(0.0, dtype=torch.float32, device=device))
        else:
            selected = torch.tensor(action_index, dtype=torch.long, device=device)
            log_probs.append(dist.log_prob(selected))
            entropies.append(dist.entropy())
        if not transition.ok:
            break
        if max_state_revisits > 0 and visit_counts[state_key] >= max_state_revisits:
            break

    target_success = stage_complete(sim, state, target_stage)
    if not target_success and failure_penalty > 0 and rewards:
        penalty = torch.tensor(float(failure_penalty), dtype=torch.float32, device=device)
        rewards[-1] = rewards[-1] - penalty
        total_reward -= float(failure_penalty)
        route[-1]["reward"] = float(route[-1]["reward"] - failure_penalty)
        route[-1].setdefault("reward_components", {})["terminal_failure"] = -float(failure_penalty)
    strict_success = target_success and state.hp > 0 and not state.dead
    final = state_summary(state)
    final["target_stage"] = target_stage
    final["yellow_guard_margin"] = yellow_guard_margin(sim, state)
    final["red_key_route_margin"] = red_key_route_margin(sim, state)
    final["boss_margin"] = _boss_margin(sim, state)
    return (
        EpisodeResult(
            route=route,
            total_reward=total_reward,
            target_success=target_success,
            strict_success=strict_success,
            final=final,
            steps=len(route),
        ),
        {
            "rewards": rewards,
            "log_probs": log_probs,
            "values": values,
            "entropies": entropies,
        },
    )


def warm_start_policy(
    route_path: str,
    data_path: str,
    sim_config: SimulatorConfig,
    model,
    optimizer,
    target_stage: str,
    epochs: int,
    prior_weight: float,
    device,
    torch,
    nn,
) -> list[dict[str, Any]]:
    route = load_route(route_path)
    rows: list[dict[str, Any]] = []
    cross_entropy = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        sim = MotaSimulator(load_game_data(data_path), sim_config)
        state = sim.reset()
        losses: list[float] = []
        matched = 0
        for route_row in route:
            if stage_complete(sim, state, target_stage):
                break
            actions = sim.macro_actions(state)
            if not actions:
                break
            target_index = find_matching_action(actions, route_row["action"])
            if target_index is None:
                break
            state_vec = torch.tensor(
                [state_feature_vector(sim, state, target_stage)],
                dtype=torch.float32,
                device=device,
            )
            action_vecs = torch.tensor(
                [[action_feature_vector(sim, state, action, target_stage) for action in actions]],
                dtype=torch.float32,
                device=device,
            )
            logits, _ = action_logits(
                sim=sim,
                state=state,
                actions=actions,
                model=model,
                target_stage=target_stage,
                prior_weight=prior_weight,
                device=device,
                torch=torch,
                state_vec=state_vec,
                action_vecs=action_vecs,
            )
            logits = logits.unsqueeze(0)
            target = torch.tensor([target_index], dtype=torch.long, device=device)
            loss = cross_entropy(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            matched += 1
            sim.apply_macro_action(state, route_row["action"])
        row = {
            "epoch": epoch,
            "matched_steps": matched,
            "loss": sum(losses) / max(1, len(losses)),
            "target_complete": stage_complete(sim, state, target_stage),
            "final": state_summary(state),
        }
        diagnosis = diagnose_route_policy(
            route_path=route_path,
            data_path=data_path,
            sim_config=sim_config,
            model=model,
            target_stage=target_stage,
            prior_weight=prior_weight,
            device=device,
            torch=torch,
            max_detail_rows=0,
        )
        row.update(
            {
                "route_top1": diagnosis["top1_accuracy"],
                "route_top3": diagnosis["top3_accuracy"],
                "route_mean_rank": diagnosis["mean_rank"],
                "first_mismatch_step": (
                    None
                    if diagnosis.get("first_mismatch") is None
                    else diagnosis["first_mismatch"]["step"]
                ),
            }
        )
        rows.append(row)
        print(json.dumps({"warm_start": row}, ensure_ascii=False), flush=True)
    return rows


def bc_anchor_update(
    route: list[dict[str, Any]],
    data_path: str,
    sim_config: SimulatorConfig,
    model,
    optimizer,
    target_stage: str,
    coef: float,
    max_steps: int,
    prior_weight: float,
    device,
    torch,
    cross_entropy,
) -> dict[str, float]:
    sim = MotaSimulator(load_game_data(data_path), sim_config)
    state = sim.reset()
    losses = []
    matched = 0
    for route_row in route:
        if max_steps > 0 and matched >= max_steps:
            break
        if stage_complete(sim, state, target_stage):
            break
        actions = sim.macro_actions(state)
        if not actions:
            break
        target_index = find_matching_action(actions, route_row["action"])
        if target_index is None:
            break
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
        target = torch.tensor([target_index], dtype=torch.long, device=device)
        losses.append(cross_entropy(logits.unsqueeze(0), target))
        matched += 1
        sim.apply_macro_action(state, route_row["action"])
    if not losses:
        return {"bc_anchor_loss": 0.0, "bc_anchor_steps": 0}
    loss = torch.stack(losses).mean() * coef
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {
        "bc_anchor_loss": float(loss.detach().cpu()),
        "bc_anchor_steps": float(matched),
    }


def loop_action_penalties(
    sim: MotaSimulator,
    state,
    actions: list[dict[str, Any]],
    route: list[dict[str, Any]],
    penalty: float,
    window: int,
    torch,
    device,
):
    """Return logit penalties for immediate stair bounces in greedy rollout.

    The learned policy is memoryless, so it can score "go up" and "go down"
    equally well and oscillate between paired stairs. This helper injects a
    small amount of rollout memory without changing the environment itself.
    """

    if penalty <= 0 or not route:
        return torch.zeros(len(actions), dtype=torch.float32, device=device)
    last = route[-1]
    last_label = last.get("action", {}).get("label", "")
    last_was_stair = _is_stair_label(last_label)
    last_before_floor = last.get("before", {}).get("floor")
    recent_positions = {
        (
            row.get("after", {}).get("floor"),
            row.get("after", {}).get("x"),
            row.get("after", {}).get("y"),
        )
        for row in route[-max(1, window):]
        if row.get("after")
    }
    recent_floors = {
        row.get("after", {}).get("floor")
        for row in route[-max(1, window):]
        if row.get("after")
    }
    values: list[float] = []
    for action in actions:
        label = action.get("label", "")
        if not _is_stair_label(label):
            values.append(0.0)
            continue
        child = state.clone()
        transition = sim.apply_macro_action(child, action)
        if not transition.ok or child.dead:
            values.append(0.0)
            continue
        child_pos = (child.floor_id, child.x, child.y)
        value = 0.0
        if last_was_stair and child.floor_id == last_before_floor:
            value += penalty
        if child_pos in recent_positions:
            value += penalty * 0.75
        elif child.floor_id in recent_floors and last_was_stair:
            value += penalty * 0.35
        values.append(value)
    return torch.tensor(values, dtype=torch.float32, device=device)


def _is_stair_label(label: str) -> bool:
    return "upFloor" in label or "downFloor" in label


def action_logits(
    sim: MotaSimulator,
    state,
    actions: list[dict[str, Any]],
    model,
    target_stage: str,
    prior_weight: float,
    device,
    torch,
    state_vec=None,
    action_vecs=None,
):
    if state_vec is None:
        state_vec = torch.tensor(
            [state_feature_vector(sim, state, target_stage)],
            dtype=torch.float32,
            device=device,
        )
    if action_vecs is None:
        action_vecs = torch.tensor(
            [[action_feature_vector(sim, state, action, target_stage) for action in actions]],
            dtype=torch.float32,
            device=device,
        )
    logits, value = model(state_vec, action_vecs)
    priors = torch.tensor(
        [[_stage_action_bias(action, target_stage, state, sim) / 3000.0 for action in actions]],
        dtype=torch.float32,
        device=device,
    )
    return (logits + priors * prior_weight).squeeze(0), value


def diagnose_route_policy(
    route_path: str | Path,
    data_path: str,
    sim_config: SimulatorConfig,
    model,
    target_stage: str,
    prior_weight: float,
    device,
    torch,
    max_detail_rows: int = 200,
) -> dict[str, Any]:
    route = load_route(route_path)
    sim = MotaSimulator(load_game_data(data_path), sim_config)
    state = sim.reset()
    details: list[dict[str, Any]] = []
    ranks: list[int] = []
    first_mismatch: dict[str, Any] | None = None
    matched = 0

    with torch.no_grad():
        for step, route_row in enumerate(route):
            if stage_complete(sim, state, target_stage):
                break
            actions = sim.macro_actions(state)
            if not actions:
                break
            target_index = find_matching_action(actions, route_row["action"])
            if target_index is None:
                mismatch = {
                    "step": step,
                    "reason": "target action missing from current macro action set",
                    "target": route_row["action"].get("label", ""),
                    "before": state_summary(state),
                    "candidate_labels": [action.get("label", "") for action in actions[:10]],
                }
                first_mismatch = first_mismatch or mismatch
                if len(details) < max_detail_rows:
                    details.append(mismatch)
                break
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
            order = torch.argsort(logits, descending=True).detach().cpu().tolist()
            rank = int(order.index(target_index) + 1)
            ranks.append(rank)
            matched += 1
            top = [
                {
                    "rank": rank_index + 1,
                    "label": actions[action_index].get("label", ""),
                    "score": float(logits[action_index].detach().cpu()),
                }
                for rank_index, action_index in enumerate(order[:5])
            ]
            detail = {
                "step": step,
                "rank": rank,
                "target": actions[target_index].get("label", ""),
                "predicted": actions[order[0]].get("label", ""),
                "top": top,
                "before": state_summary(state),
            }
            if rank != 1 and first_mismatch is None:
                first_mismatch = detail
            if len(details) < max_detail_rows:
                details.append(detail)
            transition = sim.apply_macro_action(state, route_row["action"])
            if not transition.ok:
                break

    top1 = sum(int(rank == 1) for rank in ranks)
    top3 = sum(int(rank <= 3) for rank in ranks)
    return {
        "route": str(route_path),
        "target_stage": target_stage,
        "matched_steps": matched,
        "target_complete": stage_complete(sim, state, target_stage),
        "final": state_summary(state),
        "top1": top1,
        "top3": top3,
        "top1_accuracy": top1 / max(1, len(ranks)),
        "top3_accuracy": top3 / max(1, len(ranks)),
        "mean_rank": sum(ranks) / max(1, len(ranks)),
        "first_mismatch": first_mismatch,
        "details": details,
    }


def load_route(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"No route rows found in {path}")
    return rows


def find_matching_action(actions: list[dict[str, Any]], target: dict[str, Any]) -> int | None:
    target_key = action_key(target)
    for index, action in enumerate(actions):
        if action_key(action) == target_key:
            return index
    target_label = target.get("label")
    for index, action in enumerate(actions):
        if action.get("label") == target_label:
            return index
    return None


def action_key(action: dict[str, Any]) -> tuple[Any, ...]:
    return (
        action.get("kind"),
        tuple(action.get("target") or []),
        action.get("shop"),
        action.get("label"),
    )


def update_policy(
    model,
    optimizer,
    tensors: dict[str, list[Any]],
    gamma: float,
    entropy_coef: float,
    value_coef: float,
    normalize_returns: bool,
    torch,
) -> dict[str, float]:
    if not tensors["rewards"]:
        return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    returns = []
    running = torch.tensor(0.0, dtype=torch.float32, device=tensors["rewards"][0].device)
    for reward in reversed(tensors["rewards"]):
        running = reward + gamma * running
        returns.append(running)
    returns = list(reversed(returns))
    returns_t = torch.stack(returns)
    if normalize_returns and returns_t.numel() > 1:
        returns_t = (returns_t - returns_t.mean()) / returns_t.std().clamp_min(1e-6)
    log_probs_t = torch.stack(tensors["log_probs"])
    values_t = torch.stack(tensors["values"])
    entropies_t = torch.stack(tensors["entropies"])
    advantages = returns_t - values_t.detach()
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-6)
    policy_loss = -(log_probs_t * advantages).mean()
    value_loss = (values_t - returns_t).pow(2).mean()
    entropy = entropies_t.mean()
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {
        "loss": float(loss.detach().cpu()),
        "policy_loss": float(policy_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "entropy": float(entropy.detach().cpu()),
    }


def select_device(torch, requested: str):
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def torch_cat(items, dim: int):
    import torch

    return torch.cat(items, dim=dim)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


if __name__ == "__main__":
    main()
