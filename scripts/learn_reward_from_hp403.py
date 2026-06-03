from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mota_env import MotaSimulator, SimulatorConfig, load_game_data
from mota_env.rewards import DEFAULT_LEARNABLE_GLOBAL_WEIGHTS, LearnableStageReward, Rewarder, current_stage_name
from mota_rl.train_actor_critic import find_matching_action, load_route
from mota_solver.search import state_summary, write_route_jsonl


def component_delta(
    reward: LearnableStageReward,
    sim: MotaSimulator,
    before,
    after,
    stage: str,
    gamma: float,
) -> dict[str, float]:
    before_components = reward.potential_components(sim, before, stage=stage)
    after_components = reward.potential_components(sim, after, stage=stage)
    keys = set(before_components) | set(after_components)
    return {
        key: gamma * float(after_components.get(key, 0.0)) - float(before_components.get(key, 0.0))
        for key in keys
    }


def add_dict_diff(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    keys = set(a) | set(b)
    return {key: float(a.get(key, 0.0)) - float(b.get(key, 0.0)) for key in keys}


def build_pairwise_examples(
    sim: MotaSimulator,
    route_path: Path,
    gamma: float,
    negatives_per_state: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    reward = LearnableStageReward(gamma=gamma)
    state = sim.reset()
    route_rows = load_route(route_path)
    examples: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for route_index, row in enumerate(route_rows):
        stage = current_stage_name(sim, state)
        actions = sim.macro_actions(state)
        expert_index = find_matching_action(actions, row["action"])
        if expert_index is None:
            raise SystemExit(
                f"Cannot replay expert route at step {route_index}: {row['action'].get('label', '')}"
            )

        expert_action = actions[expert_index]
        before = state.clone()
        expert_after = state.clone()
        expert_transition = sim.apply_macro_action(expert_after, expert_action)
        if not expert_transition.ok:
            raise SystemExit(f"Expert route failed at step {route_index}: {expert_transition.message}")
        expert_delta = component_delta(reward, sim, before, expert_after, stage, gamma)

        alt_rows: list[tuple[dict[str, Any], Any, dict[str, float]]] = []
        for action_index, action in enumerate(actions):
            if action_index == expert_index:
                continue
            child = before.clone()
            transition = sim.apply_macro_action(child, action)
            if not transition.ok or child.dead:
                continue
            alt_delta = component_delta(reward, sim, before, child, stage, gamma)
            alt_rows.append((action, child, alt_delta))

        if negatives_per_state > 0 and len(alt_rows) > negatives_per_state:
            alt_rows = rng.sample(alt_rows, negatives_per_state)

        for alt_action, alt_after, alt_delta in alt_rows:
            examples.append(
                {
                    "route_index": route_index,
                    "stage": stage,
                    "expert_label": expert_action.get("label", ""),
                    "alt_label": alt_action.get("label", ""),
                    "diff": add_dict_diff(expert_delta, alt_delta),
                    "before": state_summary(before),
                    "expert_after": state_summary(expert_after),
                    "alt_after": state_summary(alt_after),
                }
            )

        if not alt_rows:
            skipped.append(
                {
                    "route_index": route_index,
                    "stage": stage,
                    "expert_label": expert_action.get("label", ""),
                    "reason": "no_valid_alternative",
                }
            )

        replay_rows.append(
            {
                "index": route_index,
                "stage": stage,
                "action": expert_action,
                "before": state_summary(before),
                "after": state_summary(expert_after),
                "transition": expert_transition.message,
                "reward": expert_transition.reward,
            }
        )
        state = expert_after

    return examples, replay_rows, skipped


def train_stage_ranker(
    examples: list[dict[str, Any]],
    epochs: int,
    lr: float,
    l2: float,
    seed: int,
    device_name: str,
) -> dict[str, Any]:
    try:
        import numpy as np
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - optional dependency.
        raise SystemExit(f"Reward learning requires numpy and torch: {exc}") from exc

    if not examples:
        raise SystemExit("No pairwise examples were generated.")

    torch.manual_seed(seed)
    rng = random.Random(seed)
    feature_names = sorted({key for example in examples for key in example["diff"]})
    stage_names = sorted({str(example["stage"]) for example in examples})
    stage_to_index = {stage: index for index, stage in enumerate(stage_names)}

    raw_x = np.asarray(
        [[float(example["diff"].get(name, 0.0)) for name in feature_names] for example in examples],
        dtype=np.float32,
    )
    scale = raw_x.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale).astype(np.float32)
    x = raw_x / scale
    stage_ids = np.asarray([stage_to_index[str(example["stage"])] for example in examples], dtype=np.int64)
    indices = list(range(len(examples)))
    rng.shuffle(indices)
    split = max(1, int(len(indices) * 0.82))
    train_indices = np.asarray(indices[:split], dtype=np.int64)
    valid_indices = np.asarray(indices[split:] or indices[:split], dtype=np.int64)

    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    x_tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
    stage_tensor = torch.as_tensor(stage_ids, dtype=torch.long, device=device)
    train_tensor = torch.as_tensor(train_indices, dtype=torch.long, device=device)
    valid_tensor = torch.as_tensor(valid_indices, dtype=torch.long, device=device)

    weights = torch.nn.Parameter(torch.zeros((len(stage_names), len(feature_names)), dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam([weights], lr=lr)
    history: list[dict[str, float]] = []

    def compute_loss(index_tensor) -> tuple[Any, float]:
        logits = (x_tensor[index_tensor] * weights[stage_tensor[index_tensor]]).sum(dim=1)
        loss = F.softplus(-logits).mean() + l2 * (weights.square().mean())
        accuracy = float((logits > 0).float().mean().detach().cpu().item())
        return loss, accuracy

    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss, train_acc = compute_loss(train_tensor)
        loss.backward()
        optimizer.step()
        if epoch == 0 or epoch + 1 == epochs or (epoch + 1) % max(1, epochs // 10) == 0:
            with torch.no_grad():
                valid_loss, valid_acc = compute_loss(valid_tensor)
            history.append(
                {
                    "epoch": float(epoch + 1),
                    "train_loss": float(loss.detach().cpu().item()),
                    "train_acc": train_acc,
                    "valid_loss": float(valid_loss.detach().cpu().item()),
                    "valid_acc": valid_acc,
                }
            )

    raw_weights = (weights.detach().cpu().numpy() / scale.reshape(1, -1)).astype(float)
    stage_weights = {
        stage: {
            feature: float(raw_weights[stage_to_index[stage], feature_index])
            for feature_index, feature in enumerate(feature_names)
        }
        for stage in stage_names
    }
    return {
        "feature_names": feature_names,
        "stage_names": stage_names,
        "stage_weights": stage_weights,
        "normalization_scale": {feature: float(scale[index]) for index, feature in enumerate(feature_names)},
        "history": history,
    }


def rank_expert_actions(
    sim: MotaSimulator,
    replay_rows: list[dict[str, Any]],
    weight_payload: dict[str, Any],
    gamma: float,
) -> dict[str, Any]:
    rewarder = Rewarder("learnable_stage_pbrs", gamma=gamma)
    rewarder.learnable_stage_reward = LearnableStageReward(
        gamma=gamma,
        global_weights=weight_payload["global_weights"],
        stage_weights=weight_payload["stage_weights"],
    )
    state = sim.reset()
    ranks: list[int] = []
    top1 = 0
    top3 = 0
    rows: list[dict[str, Any]] = []

    for route_index, row in enumerate(replay_rows):
        actions = sim.macro_actions(state)
        expert_index = find_matching_action(actions, row["action"])
        if expert_index is None:
            break
        scored = []
        for action_index, action in enumerate(actions):
            child = state.clone()
            transition = sim.apply_macro_action(child, action)
            if not transition.ok or child.dead:
                continue
            score = rewarder.score(sim, state, child, action, transition).total
            scored.append((float(score), action_index, action.get("label", "")))
        scored.sort(reverse=True)
        order = [action_index for _score, action_index, _label in scored]
        rank = order.index(expert_index) + 1 if expert_index in order else len(order) + 1
        ranks.append(rank)
        top1 += int(rank == 1)
        top3 += int(rank <= 3)
        rows.append(
            {
                "route_index": route_index,
                "stage": row["stage"],
                "expert_label": row["action"].get("label", ""),
                "rank": rank,
                "top_actions": [
                    {"score": score, "label": label}
                    for score, _action_index, label in scored[:5]
                ],
            }
        )
        transition = sim.apply_macro_action(state, actions[expert_index])
        if not transition.ok:
            break

    return {
        "steps": len(ranks),
        "top1": top1 / max(1, len(ranks)),
        "top3": top3 / max(1, len(ranks)),
        "mean_rank": sum(ranks) / max(1, len(ranks)),
        "rank_histogram": dict(Counter(ranks)),
        "rows": rows,
    }


def replay_summary(sim: MotaSimulator, replay_rows: list[dict[str, Any]]) -> dict[str, Any]:
    state = sim.reset()
    for row in replay_rows:
        actions = sim.macro_actions(state)
        expert_index = find_matching_action(actions, row["action"])
        if expert_index is None:
            break
        transition = sim.apply_macro_action(state, actions[expert_index])
        if not transition.ok:
            break
    return {
        "final": state_summary(state),
        "boss_flag": bool(state.flags.get("10f战胜骷髅队长")),
        "route_len": len(replay_rows),
    }


def top_weights(stage_weights: dict[str, dict[str, float]], limit: int = 8) -> dict[str, list[dict[str, float | str]]]:
    out = {}
    for stage, weights in stage_weights.items():
        rows = sorted(weights.items(), key=lambda item: abs(item[1]), reverse=True)[:limit]
        out[stage] = [{"factor": key, "weight": float(value)} for key, value in rows]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Learn learnable_stage_pbrs reward weights from the hp403 route by pairwise action ranking.")
    parser.add_argument("--data", default="artifacts/data/mota_first10.json")
    parser.add_argument("--route", default="artifacts/manual_exploration_20260524/manual_success_no_shop_true_10f_trap_optimized_hp403.jsonl")
    parser.add_argument("--out-dir", default="artifacts/runs/hp403_reward_learning")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.035)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--negatives-per-state", type=int, default=18)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--continue-after-boss",
        action="store_true",
        help="Keep replaying route rows after the 10F captain is defeated.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    route_path = Path(args.route)
    sim = MotaSimulator(
        load_game_data(args.data),
        SimulatorConfig(enable_shop=False, stop_on_boss=not args.continue_after_boss),
    )
    examples, replay_rows, skipped = build_pairwise_examples(
        sim=sim,
        route_path=route_path,
        gamma=args.gamma,
        negatives_per_state=args.negatives_per_state,
        seed=args.seed,
    )
    learned = train_stage_ranker(
        examples=examples,
        epochs=args.epochs,
        lr=args.lr,
        l2=args.l2,
        seed=args.seed,
        device_name=args.device,
    )
    weight_payload = {
        "source": "hp403_pairwise_action_ranking",
        "route": str(route_path),
        "global_weights": dict(DEFAULT_LEARNABLE_GLOBAL_WEIGHTS),
        "stage_weights": learned["stage_weights"],
        "gamma": args.gamma,
        "training": {
            "examples": len(examples),
            "skipped": len(skipped),
            "epochs": args.epochs,
            "lr": args.lr,
            "l2": args.l2,
            "negatives_per_state": args.negatives_per_state,
            "feature_names": learned["feature_names"],
            "stage_names": learned["stage_names"],
            "normalization_scale": learned["normalization_scale"],
            "history": learned["history"],
        },
    }

    rank_eval = rank_expert_actions(
        MotaSimulator(
            load_game_data(args.data),
            SimulatorConfig(enable_shop=False, stop_on_boss=not args.continue_after_boss),
        ),
        replay_rows,
        weight_payload,
        gamma=args.gamma,
    )
    summary = {
        "route": str(route_path),
        "examples": len(examples),
        "skipped": skipped,
        "stage_counts": dict(Counter(example["stage"] for example in examples)),
        "replay": replay_summary(
            MotaSimulator(
                load_game_data(args.data),
                SimulatorConfig(enable_shop=False, stop_on_boss=not args.continue_after_boss),
            ),
            replay_rows,
        ),
        "rank_eval": {key: value for key, value in rank_eval.items() if key != "rows"},
        "top_weights": top_weights(weight_payload["stage_weights"]),
        "outputs": {
            "weights": str(out_dir / "hp403_learned_reward_weights.json"),
            "examples": str(out_dir / "pairwise_examples.jsonl"),
            "rank_eval": str(out_dir / "rank_eval.json"),
            "route_replay": str(out_dir / "hp403_replay.jsonl"),
        },
    }

    (out_dir / "hp403_learned_reward_weights.json").write_text(
        json.dumps(weight_payload, ensure_ascii=False, indent=2),
        encoding="utf8",
    )
    with (out_dir / "pairwise_examples.jsonl").open("w", encoding="utf8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
    (out_dir / "rank_eval.json").write_text(json.dumps(rank_eval, ensure_ascii=False, indent=2), encoding="utf8")
    write_route_jsonl(replay_rows, out_dir / "hp403_replay.jsonl")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8")

    report_lines = [
        "# hp403 Reward Learning",
        "",
        f"- Route: `{route_path}`",
        f"- Pairwise examples: `{len(examples)}`",
        f"- Expert top-1 under learned reward: `{rank_eval['top1']:.3f}`",
        f"- Expert top-3 under learned reward: `{rank_eval['top3']:.3f}`",
        f"- Mean rank: `{rank_eval['mean_rank']:.2f}`",
        f"- Boss flag replay: `{summary['replay']['boss_flag']}`",
        "",
        "## Top Stage Weights",
    ]
    for stage, rows in summary["top_weights"].items():
        report_lines.append(f"### {stage}")
        for row in rows:
            report_lines.append(f"- `{row['factor']}`: `{row['weight']:.6g}`")
    (out_dir / "hp403_reward_learning_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
