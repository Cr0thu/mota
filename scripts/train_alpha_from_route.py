from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mota_env import MotaSimulator, SimulatorConfig, build_graph_state, load_game_data
from mota_env.rewards import stage_complete, stage_names
from mota_solver.search import state_summary


def graph_arrays(graph: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    return {
        "node_features": np.asarray(graph["node_features"], dtype=np.float32),
        "node_type_ids": np.asarray(graph["node_type_ids"], dtype=np.int64),
        "node_mask": np.asarray(graph["node_mask"], dtype=bool),
        "executable_mask": np.asarray(graph["executable_mask"], dtype=bool),
    }


def one_hot(index: int | None, size: int):
    import numpy as np

    row = np.zeros((size,), dtype=np.float32)
    if index is not None and 0 <= int(index) < size:
        row[int(index)] = 1.0
    return row


def action_signature(action: dict[str, Any]) -> tuple[Any, ...]:
    return (
        action.get("kind"),
        tuple(action.get("target", [])),
        action.get("floor"),
        tuple(action.get("loc", [])),
        action.get("shop"),
        action.get("label"),
    )


def find_action_index(actions: list[dict[str, Any]], expert_action: dict[str, Any]) -> int | None:
    expert_sig = action_signature(expert_action)
    for index, action in enumerate(actions):
        if action_signature(action) == expert_sig:
            return index
    expert_target = tuple(expert_action.get("target", []))
    expert_label = expert_action.get("label")
    for index, action in enumerate(actions):
        if tuple(action.get("target", [])) == expert_target and action.get("label") == expert_label:
            return index
    for index, action in enumerate(actions):
        if action.get("label") == expert_label:
            return index
    return None


def load_routes_from_args(args: argparse.Namespace) -> list[str]:
    routes: list[str] = []
    for value in args.route or []:
        routes.append(value)
    for value in args.routes or []:
        routes.append(value)
    if args.routes_file:
        path = Path(args.routes_file)
        text = path.read_text(encoding="utf8")
        if path.suffix.lower() == ".json":
            payload = json.loads(text)
            if isinstance(payload, dict):
                values = payload.get("routes", [])
            else:
                values = payload
            routes.extend(str(item["route"] if isinstance(item, dict) else item) for item in values)
        else:
            routes.extend(line.strip() for line in text.splitlines() if line.strip())
    if not routes:
        routes.append("artifacts/manual_exploration_20260524/manual_success_no_shop_true_10f_trap_optimized_hp403.jsonl")

    seen: set[str] = set()
    unique_routes: list[str] = []
    for route in routes:
        normalized = str(Path(route))
        if normalized not in seen:
            seen.add(normalized)
            unique_routes.append(normalized)
    return unique_routes


def route_action_fingerprint(rows: list[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(str((row.get("action") or {}).get("label") or "") for row in rows)


def build_single_route_dataset(
    args: argparse.Namespace,
    route: str,
    *,
    route_index: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], tuple[str, ...]]:
    sim = MotaSimulator(
        load_game_data(args.data),
        SimulatorConfig(
            allow_negative_hp=args.allow_negative_hp,
            min_hp=args.relaxed_min_hp,
            stop_on_boss=args.target_stage != "boss_all_gems",
        ),
    )
    state = sim.reset()
    rows = [json.loads(line) for line in Path(route).read_text(encoding="utf8").splitlines() if line.strip()]
    samples: list[dict[str, Any]] = []
    replay_rows = 0

    for row in rows:
        if stage_complete(sim, state, args.target_stage):
            break
        actions = sim.macro_actions(state)
        expert_action = row.get("action") or {}
        action_index = find_action_index(actions, expert_action)
        if action_index is None:
            raise RuntimeError(
                f"expert action not legal at replay row {replay_rows}: {expert_action.get('label')}; "
                f"state={state_summary(state)}"
            )
        graph = build_graph_state(sim, state, actions=actions)
        node_index = graph["action_to_node_index"].get(int(action_index))
        sample = graph_arrays(graph)
        sample["policy_target"] = one_hot(node_index, int(graph["max_nodes"]))
        sample["before"] = state_summary(state)
        sample["expert_label"] = expert_action.get("label", "")
        sample["route"] = route
        sample["route_index"] = route_index
        sample["route_step"] = replay_rows
        samples.append(sample)

        transition = sim.apply_macro_action(state, actions[action_index])
        if not transition.ok:
            raise RuntimeError(f"expert replay failed at row {replay_rows}: {transition.message}")
        replay_rows += 1

    success = stage_complete(sim, state, args.target_stage)
    n = max(1, len(samples) - 1)
    for index, sample in enumerate(samples):
        progress_value = -0.2 + 1.2 * (index / n)
        sample["value_target"] = 1.0 if success and index == len(samples) - 1 else max(-1.0, min(1.0, progress_value))
    summary = {
        "route": route,
        "route_index": route_index,
        "target_stage": args.target_stage,
        "samples": len(samples),
        "replay_rows": replay_rows,
        "target_success": success,
        "final": state_summary(state),
    }
    return samples, summary, route_action_fingerprint(rows[:replay_rows])


def build_route_dataset(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    routes = load_routes_from_args(args)
    all_samples: list[dict[str, Any]] = []
    route_summaries: list[dict[str, Any]] = []
    fingerprints: set[tuple[str, ...]] = set()
    skipped_duplicates: list[dict[str, Any]] = []

    for route_index, route in enumerate(routes):
        samples, summary, fingerprint = build_single_route_dataset(args, route, route_index=route_index)
        if args.deduplicate_routes and fingerprint in fingerprints:
            summary = {**summary, "used_for_training": False, "skip_reason": "duplicate_action_sequence"}
            skipped_duplicates.append(summary)
            route_summaries.append(summary)
            continue
        fingerprints.add(fingerprint)
        summary = {**summary, "used_for_training": True}
        route_summaries.append(summary)
        all_samples.extend(samples)

    if not all_samples:
        raise RuntimeError("no route samples available after replay/deduplication")

    successful = [row for row in route_summaries if row.get("target_success")]
    summary = {
        "routes": routes,
        "route_count": len(routes),
        "used_route_count": sum(1 for row in route_summaries if row.get("used_for_training")),
        "skipped_duplicate_count": len(skipped_duplicates),
        "target_stage": args.target_stage,
        "samples": len(all_samples),
        "target_success_count": len(successful),
        "deduplicate_routes": bool(args.deduplicate_routes),
        "route_summaries": route_summaries,
    }
    return all_samples, summary


def make_batch(samples: list[dict[str, Any]], device: str):
    import numpy as np
    import torch

    return {
        "node_features": torch.as_tensor(
            np.stack([sample["node_features"] for sample in samples]),
            dtype=torch.float32,
            device=device,
        ),
        "node_type_ids": torch.as_tensor(
            np.stack([sample["node_type_ids"] for sample in samples]),
            dtype=torch.long,
            device=device,
        ),
        "node_mask": torch.as_tensor(
            np.stack([sample["node_mask"] for sample in samples]),
            dtype=torch.bool,
            device=device,
        ),
        "executable_mask": torch.as_tensor(
            np.stack([sample["executable_mask"] for sample in samples]),
            dtype=torch.bool,
            device=device,
        ),
        "policy_target": torch.as_tensor(
            np.stack([sample["policy_target"] for sample in samples]),
            dtype=torch.float32,
            device=device,
        ),
        "value_target": torch.as_tensor(
            [sample["value_target"] for sample in samples],
            dtype=torch.float32,
            device=device,
        ),
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import numpy as np
        import torch
        import torch.nn.functional as F
    except Exception as exc:
        raise SystemExit(f"train_alpha_from_route requires numpy and torch: {exc}") from exc
    from mota_rl.graph_policy_value_model import GraphPolicyValueConfig, GraphPolicyValueNet

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    samples, dataset_summary = build_route_dataset(args)
    if not samples:
        raise SystemExit("empty route dataset")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dataset_summary.json").write_text(
        json.dumps(dataset_summary, ensure_ascii=False, indent=2),
        encoding="utf8",
    )
    (out_dir / "config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf8")

    model = GraphPolicyValueNet(
        GraphPolicyValueConfig(
            d_model=args.d_model,
            nhead=args.heads,
            num_layers=args.layers,
            dropout=args.dropout,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = random.Random(args.seed)
    start = time.time()
    log_path = out_dir / "train_log.jsonl"
    best_loss = float("inf")

    with log_path.open("w", encoding="utf8") as handle:
        for step in range(max(1, args.train_steps)):
            batch_samples = rng.choices(samples, k=min(args.batch_size, max(args.batch_size, len(samples))))
            batch = make_batch(batch_samples, device)
            logits, values = model(batch["node_features"], batch["node_type_ids"], batch["node_mask"])
            executable = batch["executable_mask"]
            masked_logits = logits.masked_fill(~executable, -1.0e9)
            log_probs = F.log_softmax(masked_logits, dim=-1)
            policy_target = batch["policy_target"] * executable.float()
            target_mass = policy_target.sum(dim=-1, keepdim=True)
            valid_rows = target_mass.squeeze(-1) > 1.0e-8
            normalized_target = torch.where(
                valid_rows.unsqueeze(-1),
                policy_target / target_mass.clamp_min(1.0e-8),
                torch.zeros_like(policy_target),
            )
            policy_loss_per_sample = -(normalized_target * log_probs).sum(dim=-1)
            policy_loss = (
                policy_loss_per_sample[valid_rows].mean()
                if bool(valid_rows.any())
                else logits.sum() * 0.0
            )
            value_loss = F.mse_loss(values, batch["value_target"])
            loss = policy_loss + args.value_loss_coef * value_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            if step % args.log_every == 0 or step + 1 == args.train_steps:
                row = {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "policy_loss": float(policy_loss.detach().cpu()),
                    "value_loss": float(value_loss.detach().cpu()),
                    "elapsed_sec": round(time.time() - start, 2),
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                print(json.dumps(row, ensure_ascii=False), flush=True)
                if row["loss"] < best_loss:
                    best_loss = row["loss"]
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "args": vars(args),
                            "dataset_summary": dataset_summary,
                            "train_step": row,
                        },
                        out_dir / "best_model.pt",
                    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "dataset_summary": dataset_summary,
        },
        out_dir / "final_model.pt",
    )
    return {"out_dir": str(out_dir), "dataset": dataset_summary, "best_loss": best_loss}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="artifacts/data/mota_first10.json")
    parser.add_argument(
        "--route",
        action="append",
        default=None,
        help="Route JSONL to train from. Can be passed multiple times.",
    )
    parser.add_argument("--routes", nargs="+", default=None, help="Additional route JSONLs to train from.")
    parser.add_argument("--routes-file", default=None, help="Text or JSON manifest containing route paths.")
    parser.add_argument(
        "--no-deduplicate-routes",
        dest="deduplicate_routes",
        action="store_false",
        help="Keep exact duplicate action sequences instead of using one copy.",
    )
    parser.set_defaults(deduplicate_routes=True)
    parser.add_argument("--out-dir", default="artifacts/runs/alpha_route_warmstart")
    parser.add_argument("--target-stage", choices=stage_names(), default="boss")
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--allow-negative-hp", action="store_true")
    parser.add_argument("--relaxed-min-hp", type=int, default=-2000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=800)
    parser.add_argument("--value-loss-coef", type=float, default=0.3)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()
    print(json.dumps(train(args), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
