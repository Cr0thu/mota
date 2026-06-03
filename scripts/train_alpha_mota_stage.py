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
from mota_env.rewards import (
    GEM_STAGE_TARGETS,
    LearnableStageReward,
    MT10_RESOURCE_TARGETS,
    Rewarder,
    boss_route_margin,
    current_stage_name,
    mt10_resource_progress,
    red_key_route_margin,
    reward_scheme_names,
    stage_complete,
    stage_names,
)
from mota_solver.az_mcts import (
    AlphaMCTS,
    AlphaMCTSConfig,
    BlendedPolicyValueFn,
    HeuristicPolicyValueFn,
    LearnedRewardValueFn,
    TorchPolicyValueFn,
    filter_stage_actions,
    uniform_policy_value,
)
from mota_solver.search import state_summary, write_route_jsonl


def graph_arrays(graph: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    return {
        "node_features": np.asarray(graph["node_features"], dtype=np.float32),
        "node_type_ids": np.asarray(graph["node_type_ids"], dtype=np.int64),
        "node_mask": np.asarray(graph["node_mask"], dtype=bool),
        "executable_mask": np.asarray(graph["executable_mask"], dtype=bool),
    }


def policy_target_array(target: list[float], max_nodes: int):
    import numpy as np

    row = np.zeros((max_nodes,), dtype=np.float32)
    limit = min(len(target), max_nodes)
    if limit:
        row[:limit] = np.asarray(target[:limit], dtype=np.float32)
    total = float(row.sum())
    if total > 0:
        row /= total
    return row


def one_hot_policy_target(node_index: int | None, max_nodes: int):
    import numpy as np

    row = np.zeros((max_nodes,), dtype=np.float32)
    if node_index is not None and 0 <= int(node_index) < max_nodes:
        row[int(node_index)] = 1.0
    return row


def _softmax_scores(scores: list[float], mask: list[bool], temperature: float) -> list[float]:
    import math

    if not any(mask):
        return [0.0] * len(scores)
    temp = max(float(temperature), 1e-6)
    max_score = max(score / temp for score, valid in zip(scores, mask) if valid)
    values = [math.exp(score / temp - max_score) if valid else 0.0 for score, valid in zip(scores, mask)]
    total = sum(values)
    if total <= 1e-12:
        count = sum(1 for value in mask if value)
        return [1.0 / count if value else 0.0 for value in mask]
    return [value / total for value in values]


class GraphFeaturePolicyValueFn:
    """Feature-only AlphaZero prior over graph nodes.

    This uses graph/reward factors such as item value, unlock value, combat
    damage, and marginal ATK/DEF damage drops.  It intentionally avoids route
    demonstrations and coordinate-token action lists.
    """

    def __init__(self, target_stage: str, temperature: float = 1.0):
        self.target_stage = target_stage
        self.temperature = float(temperature)

    def __call__(self, graph: dict[str, Any]) -> tuple[list[float], float]:
        names = list(graph["feature_names"])
        mask = [bool(value) for value in graph["executable_mask"]]
        scores = [
            self._score_node(node, row, names)
            for node, row in zip(graph["nodes"], graph["node_features"])
        ]
        return _softmax_scores(scores, mask, self.temperature), self._value(graph, names)

    def _score_node(self, node: dict[str, Any], row: list[float], names: list[str]) -> float:
        kind = str(node.get("kind") or "")
        block_id = str(node.get("block_id") or "")

        def feature(name: str) -> float:
            return float(row[names.index(name)]) if name in names else 0.0

        score = 0.0
        score += 4.2 * feature("item_value_norm")
        score += 4.8 * feature("unlock_value_norm")
        score += 2.8 * feature("damage_drop_atk1_norm")
        score += 3.0 * feature("damage_drop_def1_norm")
        score -= 5.5 * feature("enemy_damage_norm")
        score -= 2.0 * feature("missing_yellow_norm")
        score -= 3.0 * feature("missing_blue_norm")
        score -= 5.0 * feature("missing_red_norm")
        score -= 0.55 * feature("path_len_norm")
        if self._is_stage_target(node):
            score += 55.0

        if kind == "enemy":
            if feature("enemy_killable") <= 0:
                score -= 20.0
            if feature("unlock_value_norm") <= 0.05:
                score -= 1.5
            score += 2.5 * feature("boss_margin_norm")
        elif kind == "door":
            score -= 0.8
            score += 2.4 * feature("door_openable")
            if self.target_stage in {"shield_buffer", "mid_gems", "low_gems", "all_gems"}:
                yellow_stock = feature("yellow_key_norm") * 10.0
                blue_stock = feature("blue_key_norm") * 5.0
                unlock = feature("unlock_value_norm")
                if feature("required_yellow_norm") > 0:
                    if yellow_stock <= 0.5:
                        score -= 8.0
                    elif yellow_stock <= 2.0 and unlock < 0.30:
                        score -= 4.0 * (2.5 - yellow_stock)
                is_mt7_merchant_blue_door = (
                    str(node.get("floor") or "") == "MT7"
                    and int(node.get("x") or -1) == 5
                    and int(node.get("y") or -1) == 5
                )
                if is_mt7_merchant_blue_door and self.target_stage == "shield_buffer":
                    score += 5.0
                elif feature("required_blue_norm") > 0 and blue_stock <= 1.0 and unlock < 0.35:
                    score -= 7.0
        elif kind == "stair":
            score += self._stair_score(node)

        if block_id == "sword1":
            score += 24.0 if self.target_stage == "sword" else 6.0
        elif block_id == "shield1":
            score += 24.0 if self.target_stage in {"shield", "shield_buffer", "pre_shield_gems"} else 8.0
        elif block_id == "redKey":
            score += 26.0 if self.target_stage in {"red_key", "boss_ready", "trap", "boss"} else 8.0
        elif block_id == "redGem":
            score += 12.0
            if self.target_stage in {"mid_gems", "low_gems", "mt8_gems", "lower_gems", "all_gems"}:
                score += 7.0
        elif block_id == "blueGem":
            score += 12.0
            if self.target_stage in {"mid_gems", "low_gems", "mt8_gems", "lower_gems", "all_gems"}:
                score += 7.0
        elif block_id in {"yellowKey", "blueKey", "redKey"}:
            score += {"yellowKey": 7.0, "blueKey": 10.0, "redKey": 18.0}.get(block_id, 6.0)
        elif block_id in {"redPotion", "bluePotion"}:
            hp_norm = feature("hp_norm")
            score += 1.5
            if hp_norm < 0.10:
                score += 12.0
            elif hp_norm < 0.20:
                score += 5.0
            if self.target_stage in {"mt10_resources", "boss_ready", "trap", "boss"} and feature("is_mt10") > 0:
                score += 8.0

        if self.target_stage in {"mt10_resources", "all_gems", "boss_ready", "trap", "boss"}:
            score += 3.0 * feature("mt10_resource_remaining_norm") * feature("is_mt10")
            score += 2.0 * feature("boss_margin_norm")
        return score

    def _is_stage_target(self, node: dict[str, Any]) -> bool:
        floor = str(node.get("floor") or "")
        try:
            x = int(node.get("x"))
            y = int(node.get("y"))
        except (TypeError, ValueError):
            return False
        block_id = str(node.get("block_id") or "")
        targets = GEM_STAGE_TARGETS.get(self.target_stage, ())
        if self.target_stage == "all_gems":
            targets = tuple(target for rows in GEM_STAGE_TARGETS.values() for target in rows)
        if self.target_stage in {"mt10_resources", "boss_all_gems"}:
            targets = tuple(targets) + tuple(MT10_RESOURCE_TARGETS)
        return any(
            floor == target_floor and x == target_x and y == target_y and block_id == target_id
            for target_floor, target_x, target_y, target_id in targets
        )

    def _stair_score(self, node: dict[str, Any]) -> float:
        floor = str(node.get("floor") or "")
        try:
            target_floor = int(floor.removeprefix("MT"))
        except ValueError:
            target_floor = 0
        stage_floor = {
            "sword": 5,
            "mt4_redgem": 4,
            "pre_shield_gems": 8,
            "shield": 9,
            "shield_buffer": 7,
            "mid_gems": 6,
            "low_gems": 3,
            "mt8_hp_ready": 8,
            "mt8_gems": 8,
            "lower_gems": 6,
            "mt10_blue_ready": 10,
            "mt10_yellow_ready": 10,
            "mt10_resources": 10,
            "all_gems": 10,
            "red_key": 8,
            "boss_ready": 10,
            "trap": 10,
            "boss": 10,
            "boss_all_gems": 10,
        }.get(self.target_stage, target_floor)
        if target_floor <= 0:
            return 0.2
        return max(-2.0, 2.0 - 0.45 * abs(target_floor - stage_floor))

    def _value(self, graph: dict[str, Any], names: list[str]) -> float:
        if not graph["node_features"]:
            return 0.0
        hero = graph["node_features"][0]

        def feature(name: str) -> float:
            return float(hero[names.index(name)]) if name in names else 0.0

        value = -0.35
        value += 0.55 * feature("hp_norm")
        value += 0.75 * feature("atk_norm")
        value += 0.75 * feature("def_norm")
        value += 0.25 * feature("yellow_key_norm")
        value += 0.35 * feature("blue_key_norm")
        value += 0.55 * feature("boss_margin_norm")
        return max(-1.0, min(1.0, value))


def action_signature(action: dict[str, Any]) -> tuple[Any, ...]:
    return (
        action.get("kind"),
        tuple(action.get("target", [])),
        action.get("floor"),
        tuple(action.get("loc", [])),
        action.get("shop"),
        action.get("label"),
    )


def find_action_index(actions: list[dict[str, Any]], route_action: dict[str, Any]) -> int | None:
    signature = action_signature(route_action)
    for index, action in enumerate(actions):
        if action_signature(action) == signature:
            return index
    target = tuple(route_action.get("target", []))
    label = route_action.get("label")
    for index, action in enumerate(actions):
        if tuple(action.get("target", [])) == target and action.get("label") == label:
            return index
    for index, action in enumerate(actions):
        if action.get("label") == label:
            return index
    return None


def apply_start_route(
    sim: MotaSimulator,
    state,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if not args.start_route:
        return []
    rows = [
        json.loads(line)
        for line in Path(args.start_route).read_text(encoding="utf8").splitlines()
        if line.strip()
    ]
    prefix: list[dict[str, Any]] = []
    raw_stop_stage = str(args.start_route_stop_stage or "").strip()
    stop_stage = "" if raw_stop_stage.lower() in {"none", "all", "full", "__none__"} else raw_stop_stage
    for row in rows[: max(0, int(args.start_route_max_actions)) or None]:
        if stop_stage and stage_complete(sim, state, stop_stage):
            break
        actions = sim.macro_actions(state)
        action_index = find_action_index(actions, row.get("action") or {})
        if action_index is None:
            raise RuntimeError(
                f"start route action not legal at prefix row {len(prefix)}: "
                f"{(row.get('action') or {}).get('label')}; state={state_summary(state)}"
            )
        before = state.clone()
        action = actions[action_index]
        transition = sim.apply_macro_action(state, action)
        prefix.append(
            {
                "index": len(prefix),
                "prefix": True,
                "stage": current_stage_name(sim, before),
                "target_stage": args.target_stage,
                "action": action,
                "before": state_summary(before),
                "after": state_summary(state),
                "ok": transition.ok,
                "message": transition.message,
            }
        )
        if not transition.ok or state.dead or state.done:
            break
    return prefix


def choose_non_revisit_action(
    sim: MotaSimulator,
    before,
    result,
    visit_counts: dict[tuple[Any, ...], int],
    max_revisits: int,
    target_stage: str,
    *,
    rng: random.Random | None = None,
    action_temperature: float = 0.0,
    action_top_k: int = 0,
    avoid_regressive_stair: bool = True,
) -> tuple[dict[str, Any] | None, int | None, int | None]:
    candidates: list[tuple[int, dict[str, Any], int, int, dict[str, Any]]] = []
    for child in result.child_stats:
        candidate = before.clone()
        transition = sim.apply_macro_action(candidate, child["action"])
        if not transition.ok:
            continue
        key = sim.state_key(candidate)
        candidates.append(
            (
                visit_counts.get(key, 0),
                child["action"],
                int(child["action_index"]),
                int(child["action_node_index"]),
                child,
            )
        )
    if not candidates:
        return result.action, result.action_index, result.action_node_index

    def is_regressive_stair(action: dict[str, Any]) -> bool:
        label = str(action.get("label") or "").lower()
        if target_stage == "shield_buffer" and before.flags.get("nowShield") == "shield1":
            return False
        effective_target = target_stage
        if target_stage in {
            "pre_shield_gems",
            "shield",
            "shield_buffer",
            "red_key",
            "boss_ready",
            "trap",
            "boss",
            "boss_all_gems",
        } and before.atk < 20:
            effective_target = "sword"
        if target_stage in {"shield", "shield_buffer"} and before.atk >= 20:
            try:
                floor_index = int(str(before.floor_id).removeprefix("MT"))
            except ValueError:
                floor_index = 1
            if floor_index <= 9:
                effective_target = "sword"
        return effective_target == "sword" and "downfloor" in label

    filtered = [
        row
        for row in candidates
        if row[0] < max_revisits and (not avoid_regressive_stair or not is_regressive_stair(row[1]))
    ]
    if not filtered:
        filtered = [row for row in candidates if row[0] < max_revisits]
    if not filtered:
        return None, None, None

    if action_temperature > 0.0 and len(filtered) > 1:
        local_rng = rng or random
        ranked = sorted(
            filtered,
            key=lambda row: (
                -int(row[4].get("visit_count", 0)),
                -float(row[4].get("value", 0.0)),
                -float(row[4].get("prior", 0.0)),
            ),
        )
        if action_top_k > 0:
            ranked = ranked[: max(1, int(action_top_k))]
        inv_temp = 1.0 / max(float(action_temperature), 1e-6)
        weights = [max(1e-6, float(row[4].get("visit_count", 0)) + 1.0) ** inv_temp for row in ranked]
        threshold = local_rng.random() * sum(weights)
        running = 0.0
        for row, weight in zip(ranked, weights):
            running += weight
            if running >= threshold:
                _, action, action_index, node_index, _ = row
                return action, action_index, node_index
        _, action, action_index, node_index, _ = ranked[-1]
        return action, action_index, node_index

    _, action, action_index, node_index, _ = filtered[0]
    return action, action_index, node_index


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


def replay_batch_samples(
    replay: list[dict[str, Any]],
    batch_size: int,
    args: argparse.Namespace,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if batch_size >= len(replay) and args.positive_replay_ratio <= 0:
        return rng.sample(replay, batch_size)
    ratio = max(0.0, min(1.0, float(args.positive_replay_ratio)))
    if ratio <= 0:
        return rng.sample(replay, batch_size)

    positives = [sample for sample in replay if float(sample.get("value_target", 0.0)) > 0.0]
    if not positives:
        return rng.sample(replay, batch_size)
    negatives = [sample for sample in replay if float(sample.get("value_target", 0.0)) <= 0.0]

    def draw(pool: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
        if count <= 0 or not pool:
            return []
        if len(pool) >= count:
            return rng.sample(pool, count)
        return [rng.choice(pool) for _ in range(count)]

    positive_count = min(batch_size, max(1, int(round(batch_size * ratio))))
    negative_count = batch_size - positive_count
    samples = draw(positives, positive_count) + draw(negatives, negative_count)
    if len(samples) < batch_size:
        samples.extend(draw(replay, batch_size - len(samples)))
    rng.shuffle(samples)
    return samples[:batch_size]


def train_step(model, optimizer, replay: list[dict[str, Any]], args: argparse.Namespace, rng: random.Random, device: str):
    import torch
    import torch.nn.functional as F

    batch_size = min(args.batch_size, len(replay))
    samples = replay_batch_samples(replay, batch_size, args, rng)
    batch = make_batch(samples, device)
    logits, values = model(batch["node_features"], batch["node_type_ids"], batch["node_mask"])
    executable = batch["executable_mask"]
    masked_logits = logits.masked_fill(~executable, -1.0e9)
    log_probs = F.log_softmax(masked_logits, dim=-1)
    # MCTS can be run with a different filtered action set than the graph stored
    # for training.  Project the visit distribution back onto currently legal
    # executable nodes so a stale/filtered target cannot create enormous losses
    # on masked logits.
    policy_target = batch["policy_target"] * executable.float()
    target_mass = policy_target.sum(dim=-1, keepdim=True)
    valid_policy_rows = target_mass.squeeze(-1) > 1.0e-8
    normalized_target = torch.where(
        valid_policy_rows.unsqueeze(-1),
        policy_target / target_mass.clamp_min(1.0e-8),
        torch.zeros_like(policy_target),
    )
    policy_loss_per_sample = -(normalized_target * log_probs).sum(dim=-1)
    if args.policy_value_weighted:
        weights = ((batch["value_target"] + 1.0) * 0.5).clamp(
            min=float(args.policy_min_weight),
            max=1.0,
        )
        weights = weights * valid_policy_rows.float()
        policy_loss = (policy_loss_per_sample * weights).sum() / weights.sum().clamp_min(1e-6)
    else:
        policy_loss = (
            policy_loss_per_sample[valid_policy_rows].mean()
            if bool(valid_policy_rows.any())
            else logits.sum() * 0.0
        )
    value_loss = F.mse_loss(values, batch["value_target"])
    loss = policy_loss + args.value_loss_coef * value_loss
    if not bool(torch.isfinite(loss)):
        optimizer.zero_grad(set_to_none=True)
        return {
            "loss": float("nan"),
            "policy_loss": float("nan"),
            "value_loss": float(value_loss.detach().cpu()) if bool(torch.isfinite(value_loss)) else float("nan"),
            "skipped": 1.0,
        }
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    return {
        "loss": float(loss.detach().cpu()),
        "policy_loss": float(policy_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "valid_policy_rows": float(valid_policy_rows.float().mean().detach().cpu()),
    }


def stage_value_target(
    sim: MotaSimulator,
    state,
    target_stage: str,
    target_success: bool,
    *,
    stage_value_mode: str = "heuristic",
    hp_aware_success_value: bool = False,
    hp_success_base: float = 0.55,
    hp_success_scale: float = 1000.0,
) -> float:
    if stage_value_mode == "binary":
        return 1.0 if target_success else -1.0
    if target_success:
        if hp_aware_success_value and target_stage in {
            "shield",
            "shield_buffer",
            "red_key",
            "boss_ready",
            "trap",
            "boss",
            "boss_all_gems",
        }:
            scale = max(1.0, float(hp_success_scale))
            key_bonus = (
                min(0.30, 0.06 * float(state.items.get("yellowKey", 0)))
                + min(0.12, 0.06 * float(state.items.get("blueKey", 0)))
            )
            if target_stage in {"shield", "shield_buffer"}:
                key_bonus += min(0.12, 0.002 * float(state.money))
            value = float(hp_success_base) + float(state.hp) / scale + key_bonus
            return max(-1.0, min(1.0, value))
        return 1.0
    if target_stage == "shield_buffer":
        yk = float(state.items.get("yellowKey", 0))
        bk = float(state.items.get("blueKey", 0))
        value = -0.78
        value += min(0.28, max(0.0, float(state.hp)) / 1100.0)
        value += 0.18 if state.flags.get("nowShield") == "shield1" else 0.0
        value += 0.12 if state.atk >= 21 else max(0.0, state.atk - 20) * 0.05
        value += 0.12 if state.defense >= 20 else max(0.0, state.defense - 10) * 0.01
        value += min(0.30, 0.12 * yk)
        value += min(0.10, 0.05 * bk)
        value += min(0.12, max(0.0, float(state.money)) / 700.0)
        if yk <= 0:
            value -= 0.28
        elif yk <= 1:
            value -= 0.12
        if state.hp < 120:
            value -= 0.16
        return max(-1.0, min(0.72, value))
    if target_stage in {"red_key", "boss_ready", "trap", "boss", "boss_all_gems"}:
        red_margin = red_key_route_margin(sim, state)
        boss_margin = boss_route_margin(sim, state)
        margin = red_margin if target_stage == "red_key" else boss_margin
        if margin <= -9000:
            value = -0.95
        else:
            value = -0.65 + max(-0.30, min(0.45, float(margin) / 1200.0))
        value += min(0.18, max(0.0, float(state.hp)) / 1800.0)
        value += min(0.12, 0.025 * float(state.items.get("yellowKey", 0)))
        value += min(0.08, 0.04 * float(state.items.get("blueKey", 0)))
        value += min(0.18, 0.06 * float(mt10_resource_progress(sim, state)))
        if state.items.get("redKey", 0) > 0:
            value += 0.35
        if state.hp <= 30:
            value -= 0.35
        if state.items.get("yellowKey", 0) <= 0:
            value -= 0.15
        return max(-1.0, min(0.65, value))
    try:
        floor_index = int(str(state.floor_id).removeprefix("MT"))
    except ValueError:
        floor_index = 1
    value = -1.0
    value += min(0.35, max(0, floor_index - 1) * 0.04)
    if state.atk >= 20:
        value += 0.45
    elif state.atk > 10:
        value += 0.12
    if state.defense > 10:
        value += min(0.18, (state.defense - 10) * 0.06)
    if state.hp > 500:
        value += 0.08
    elif state.hp <= 0:
        value -= 0.25
    if target_stage in {"shield", "shield_buffer", "red_key", "boss_ready", "trap", "boss", "boss_all_gems"}:
        if state.floor_id in {"MT8", "MT9", "MT10"}:
            value += 0.2
        if state.flags.get("nowShield") == "shield1":
            value += 0.5
    if target_stage in {"red_key", "boss_ready", "trap", "boss", "boss_all_gems"} and state.items.get("redKey", 0) > 0:
        value += 0.4
    if target_stage in {"boss", "boss_all_gems"} and state.flags.get("10f战胜骷髅队长"):
        value = 1.0
    return max(-1.0, min(1.0, value))


def normalized_step_reward(
    sim: MotaSimulator,
    before,
    after,
    action: dict[str, Any],
    transition,
    args: argparse.Namespace,
    rewarder: Rewarder | None = None,
) -> float:
    scheme = str(args.mcts_edge_reward_scheme or "none")
    if scheme == "none":
        return 0.0
    if scheme == "raw":
        reward = float(transition.reward)
    else:
        rewarder = rewarder or Rewarder(scheme, gamma=args.reward_gamma)
        reward = float(rewarder.score(sim, before, after, action, transition).total)
    scale = max(1.0e-6, float(args.mcts_edge_reward_scale))
    value = reward / scale
    clip = max(0.0, float(args.mcts_edge_reward_clip))
    if clip > 0.0:
        value = max(-clip, min(clip, value))
    return float(value)


def make_learned_potential_reward(
    payload: dict[str, Any],
    *,
    gamma: float,
    stage_mode: str,
) -> LearnableStageReward:
    weights = payload.get("weights", payload)
    return LearnableStageReward(
        gamma=float(weights.get("gamma", gamma)),
        global_weights=dict(weights.get("global_weights", {})),
        stage_weights={
            str(stage): {str(key): float(value) for key, value in dict(stage_weights).items()}
            for stage, stage_weights in dict(weights.get("stage_weights", {})).items()
        },
    )


def learned_potential_edge_delta(
    reward: LearnableStageReward,
    sim: MotaSimulator,
    before,
    after,
    *,
    target_stage: str,
    gamma: float,
    stage_mode: str,
) -> float:
    stage = current_stage_name(sim, before) if stage_mode == "current" else target_stage
    return float(gamma) * reward.potential(sim, after, stage=stage) - reward.potential(sim, before, stage=stage)


def normalized_learned_potential_reward(
    reward: LearnableStageReward,
    sim: MotaSimulator,
    before,
    after,
    args: argparse.Namespace,
) -> float:
    raw = learned_potential_edge_delta(
        reward,
        sim,
        before,
        after,
        target_stage=args.target_stage,
        gamma=args.reward_gamma,
        stage_mode=args.reward_value_stage_mode,
    )
    value = raw / max(1.0e-6, float(args.mcts_edge_reward_scale))
    clip = max(0.0, float(args.mcts_edge_reward_clip))
    if clip > 0.0:
        value = max(-clip, min(clip, value))
    return float(value)


def route_score_components(result: dict[str, Any]) -> dict[str, float]:
    final = result.get("final", {}) or {}
    keys = final.get("keys", {}) or {}
    return {
        "target_success": float(bool(result.get("target_success"))),
        "boss_success": float(bool(result.get("boss_success"))),
        "hp": float(final.get("hp", 0) or 0),
        "atk": float(final.get("atk", 0) or 0),
        "def": float(final.get("def", 0) or 0),
        "money": float(final.get("money", 0) or 0),
        "yellow": float(keys.get("yellowKey", 0) or 0),
        "blue": float(keys.get("blueKey", 0) or 0),
        "red": float(keys.get("redKey", 0) or 0),
        "macro_steps": float(result.get("macro_steps", 0) or 0),
        "boss_margin": float(result.get("boss_margin", 0) or 0),
    }


def scalar_route_score(result: dict[str, Any], *, mode: str) -> float:
    c = route_score_components(result)
    base = 1_000_000.0 * c["target_success"] + 2_000_000.0 * c["boss_success"]
    if mode == "hp":
        return (
            base
            + c["hp"]
            + 75.0 * c["atk"]
            + 75.0 * c["def"]
            + 25.0 * c["yellow"]
            + 45.0 * c["blue"]
            + 100.0 * c["red"]
            - 2.0 * c["macro_steps"]
        )
    if mode == "resource":
        return (
            base
            + 0.35 * c["hp"]
            + 55.0 * c["atk"]
            + 55.0 * c["def"]
            + 130.0 * c["yellow"]
            + 190.0 * c["blue"]
            + 320.0 * c["red"]
            + 1.0 * c["money"]
            - 1.5 * c["macro_steps"]
        )
    return (
        base
        + c["hp"]
        + 100.0 * c["atk"]
        + 100.0 * c["def"]
        + 90.0 * c["yellow"]
        + 130.0 * c["blue"]
        + 250.0 * c["red"]
        + 0.8 * c["money"]
        - 2.0 * c["macro_steps"]
    )


def run_episode(
    *,
    model,
    sim: MotaSimulator,
    args: argparse.Namespace,
    device: str,
    seed: int,
    episode_index: int = 0,
) -> dict[str, Any]:
    state = sim.reset()
    route: list[dict[str, Any]] = apply_start_route(sim, state, args)
    examples: list[dict[str, Any]] = []
    visit_counts: dict[tuple[Any, ...], int] = {sim.state_key(state): 1}
    if int(args.uniform_prior_episodes) > 0 and episode_index < int(args.uniform_prior_episodes):
        policy_value_fn = uniform_policy_value
    elif args.heuristic_prior_mix >= 0.999:
        policy_value_fn = HeuristicPolicyValueFn(args.target_stage, temperature=args.heuristic_temperature)
    else:
        policy_value_fn = TorchPolicyValueFn(model, device=device, temperature=args.policy_temperature)
    if 0 < args.heuristic_prior_mix < 0.999:
        policy_value_fn = BlendedPolicyValueFn(
            policy_value_fn,
            HeuristicPolicyValueFn(args.target_stage, temperature=args.heuristic_temperature),
            secondary_mix=args.heuristic_prior_mix,
        )
    if args.feature_prior_mix > 0:
        policy_value_fn = BlendedPolicyValueFn(
            policy_value_fn,
            GraphFeaturePolicyValueFn(args.target_stage, temperature=args.feature_prior_temperature),
            secondary_mix=args.feature_prior_mix,
        )
    leaf_value_fn = None
    learned_potential_reward = None
    learned_edge_reward_fn = None
    if args.reward_weights_file:
        payload = json.loads(Path(args.reward_weights_file).read_text(encoding="utf8"))
        leaf_value_fn = LearnedRewardValueFn(
            payload,
            scale=args.reward_value_scale,
            stage_mode=args.reward_value_stage_mode,
            gamma=args.reward_gamma,
        )
        if args.use_learned_potential_edge_reward:
            learned_potential_reward = make_learned_potential_reward(
                payload,
                gamma=args.reward_gamma,
                stage_mode=args.reward_value_stage_mode,
            )

            def learned_edge_reward_fn(sim, before, after, action, transition, target_stage):
                return learned_potential_edge_delta(
                    learned_potential_reward,
                    sim,
                    before,
                    after,
                    target_stage=target_stage,
                    gamma=args.reward_gamma,
                    stage_mode=args.reward_value_stage_mode,
                )

    step_rewarder = None
    if str(args.mcts_edge_reward_scheme or "none") not in {"none", "raw"}:
        step_rewarder = Rewarder(str(args.mcts_edge_reward_scheme), gamma=args.reward_gamma)

    for step in range(args.max_macros):
        if stage_complete(sim, state, args.target_stage) or state.dead or state.done:
            break
        actions = sim.macro_actions(state)
        if not args.disable_stage_action_filter:
            actions = filter_stage_actions(actions, state, args.target_stage, sim=sim)
        if not actions:
            break
        graph = build_graph_state(sim, state, actions=actions)
        mcts = AlphaMCTS(
            sim,
            policy_value_fn=policy_value_fn,
            leaf_value_fn=leaf_value_fn,
            edge_reward_fn=learned_edge_reward_fn,
            config=AlphaMCTSConfig(
                target_stage=args.target_stage,
                num_simulations=args.simulations,
                c_puct=args.c_puct,
                max_depth=args.max_depth,
                seed=seed + step,
                hp_aware_success_value=args.hp_aware_success_value,
                hp_success_base=args.hp_success_base,
                hp_success_scale=args.hp_success_scale,
                final_action_visit_weight=args.final_action_visit_weight,
                final_action_value_weight=args.final_action_value_weight,
                final_action_prior_weight=args.final_action_prior_weight,
                root_dirichlet_alpha=args.root_dirichlet_alpha,
                root_exploration_fraction=args.root_exploration_fraction,
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
            args.target_stage,
            rng=random.Random(seed + step * 9973),
            action_temperature=args.action_temperature,
            action_top_k=args.action_top_k,
            avoid_regressive_stair=not args.disable_stage_action_filter,
        )
        if selected_action is None:
            break
        arrays = graph_arrays(graph)
        if args.selected_policy_target:
            arrays["policy_target"] = one_hot_policy_target(selected_node_index, int(graph["max_nodes"]))
        else:
            arrays["policy_target"] = policy_target_array(result.policy_target, int(graph["max_nodes"]))
        arrays["stage"] = current_stage_name(sim, before)
        arrays["before"] = state_summary(before)
        arrays["mcts_root_value"] = float(result.root_value)
        examples.append(arrays)
        transition = sim.apply_macro_action(state, selected_action)
        if learned_potential_reward is not None:
            arrays["step_reward"] = normalized_learned_potential_reward(
                learned_potential_reward,
                sim,
                before,
                state,
                args,
            )
        else:
            arrays["step_reward"] = normalized_step_reward(
                sim,
                before,
                state,
                selected_action,
                transition,
                args,
                rewarder=step_rewarder,
            )
        visit_counts[sim.state_key(state)] = visit_counts.get(sim.state_key(state), 0) + 1
        route.append(
            {
                "index": len(route),
                "prefix": False,
                "stage": arrays["stage"],
                "target_stage": args.target_stage,
                "action": selected_action,
                "mcts": {
                    "root_value": result.root_value,
                    "visit_count": result.visit_count,
                    "selected_action_index": selected_action_index,
                    "selected_action_node_index": selected_node_index,
                    "top_children": result.child_stats[:8],
                },
                "before": arrays["before"],
                "after": state_summary(state),
                "ok": transition.ok,
                "message": transition.message,
            }
        )
        if not transition.ok:
            break

    target_success = stage_complete(sim, state, args.target_stage)
    value_target = stage_value_target(
        sim,
        state,
        args.target_stage,
        target_success,
        stage_value_mode=args.stage_value_mode,
        hp_aware_success_value=args.hp_aware_success_value,
        hp_success_base=args.hp_success_base,
        hp_success_scale=args.hp_success_scale,
    )
    final_return_targets: list[float] = []
    backed_return = float(value_target)
    return_gamma = float(args.value_return_gamma)
    for sample in reversed(examples):
        backed_return = float(sample.get("step_reward", 0.0)) + return_gamma * backed_return
        final_return_targets.append(max(-1.0, min(1.0, float(backed_return))))
    final_return_targets.reverse()
    for index, sample in enumerate(examples):
        return_target = final_return_targets[index] if index < len(final_return_targets) else value_target
        if args.value_target_mode == "final":
            sample_value = return_target
        elif args.value_target_mode == "root":
            sample_value = float(sample.get("mcts_root_value", value_target))
        else:
            root_value = float(sample.get("mcts_root_value", value_target))
            sample_value = (
                float(args.mixed_final_weight) * return_target
                + (1.0 - float(args.mixed_final_weight)) * root_value
            )
        sample["value_target"] = max(-1.0, min(1.0, float(sample_value)))
    return {
        "examples": examples,
        "route": route,
        "target_success": target_success,
        "boss_success": bool(state.flags.get("10f战胜骷髅队长")),
        "boss_margin": boss_route_margin(sim, state),
        "final": state_summary(state),
        "macro_steps": len(route),
        "primitive_steps": state.steps,
        "value_target": value_target,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="artifacts/data/mota_first10.json")
    parser.add_argument("--out-dir", default="artifacts/runs/alpha_mota_stage")
    parser.add_argument(
        "--protocol",
        choices=(
            "manual_guided",
            "no_agent_manual",
            "auto_reward_search",
            "self_generated_curriculum",
            "auto_reward_curriculum",
        ),
        default="manual_guided",
        help=(
            "manual_guided permits coordinate/token filters and route prefixes. "
            "no_agent_manual disables agent-authored stage filters and rejects route warm-starts. "
            "auto_reward_search keeps the no-route/no-checkpoint/no-filter constraints but permits a "
            "predeclared automatically generated reward-weight file. "
            "self_generated_curriculum permits only a proven self-generated start route. "
            "auto_reward_curriculum permits that route plus a generated reward-weight file."
        ),
    )
    parser.add_argument("--target-stage", choices=stage_names(), default="sword")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-macros", type=int, default=80)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--max-depth", type=int, default=80)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--max-state-revisits", type=int, default=1)
    parser.add_argument("--selected-policy-target", action="store_true")
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--allow-negative-hp", action="store_true")
    parser.add_argument("--relaxed-min-hp", type=int, default=-2000)
    parser.add_argument("--hp-aware-success-value", action="store_true")
    parser.add_argument("--hp-success-base", type=float, default=0.55)
    parser.add_argument("--hp-success-scale", type=float, default=1000.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument(
        "--uniform-prior-episodes",
        type=int,
        default=0,
        help=(
            "Use uniform MCTS priors for the first N episodes so early search is "
            "not biased by an untrained random policy/value network."
        ),
    )
    parser.add_argument(
        "--feature-prior-mix",
        type=float,
        default=0.0,
        help=(
            "Blend the neural/uniform AlphaZero prior with a graph-feature prior "
            "computed from reward factors such as item value, unlock value, and "
            "combat damage. This does not use demonstration routes."
        ),
    )
    parser.add_argument("--feature-prior-temperature", type=float, default=0.8)
    parser.add_argument("--policy-temperature", type=float, default=1.0)
    parser.add_argument("--heuristic-prior-mix", type=float, default=0.0)
    parser.add_argument("--heuristic-temperature", type=float, default=0.8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-size", type=int, default=5000)
    parser.add_argument(
        "--positive-replay-ratio",
        type=float,
        default=0.0,
        help=(
            "Sample this fraction of each training batch from successful AlphaZero "
            "search examples when available.  This is self-generated replay only; "
            "it does not load external demonstrations."
        ),
    )
    parser.add_argument("--train-steps-per-episode", type=int, default=16)
    parser.add_argument("--value-loss-coef", type=float, default=1.0)
    parser.add_argument("--policy-value-weighted", action="store_true")
    parser.add_argument("--policy-min-weight", type=float, default=0.1)
    parser.add_argument("--reward-weights-file", default="")
    parser.add_argument("--reward-value-scale", type=float, default=25_000.0)
    parser.add_argument("--reward-value-stage-mode", choices=("target", "current"), default="target")
    parser.add_argument("--reward-gamma", type=float, default=0.99)
    parser.add_argument(
        "--use-learned-potential-edge-reward",
        action="store_true",
        help=(
            "When --reward-weights-file is set, use gamma*Phi(s')-Phi(s) from "
            "that learned potential as the single-player MCTS edge reward and "
            "the value-return step reward."
        ),
    )
    parser.add_argument(
        "--value-target-mode",
        choices=("final", "root", "mixed"),
        default="final",
        help=(
            "final is the single-player AlphaZero return target G_t; "
            "root/mixed use search leaf values for dense single-player training."
        ),
    )
    parser.add_argument(
        "--value-return-gamma",
        type=float,
        default=1.0,
        help=(
            "Discount for constructing value targets G_t=r_t+gamma*G_{t+1}. "
            "Use 1.0 to recover undiscounted AlphaZero-style episode outcomes."
        ),
    )
    parser.add_argument(
        "--stage-value-mode",
        choices=("heuristic", "binary"),
        default="heuristic",
        help="heuristic uses hand-shaped stage values; binary uses only stage success/failure outcomes.",
    )
    parser.add_argument("--mixed-final-weight", type=float, default=0.35)
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
        help="Disable coordinate/token stage action filters; MCTS expands all legal macro actions.",
    )
    parser.add_argument("--action-temperature", type=float, default=0.0)
    parser.add_argument("--action-top-k", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--start-route", default="")
    parser.add_argument(
        "--start-route-provenance",
        default="",
        help="Manifest or summary proving the prefix route came from an earlier automatic run.",
    )
    parser.add_argument(
        "--start-route-stop-stage",
        choices=("", "none", "all", "full", "__none__", *stage_names()),
        default="",
    )
    parser.add_argument("--start-route-max-actions", type=int, default=10_000)
    parser.add_argument("--save-every", type=int, default=5)
    args = parser.parse_args()
    if args.protocol in {"no_agent_manual", "auto_reward_search"}:
        args.disable_stage_action_filter = True
        args.stage_value_mode = "binary"
        if args.start_route:
            raise SystemExit(f"--protocol {args.protocol} forbids --start-route")
        if args.init_checkpoint:
            raise SystemExit(f"--protocol {args.protocol} forbids --init-checkpoint")
        if args.heuristic_prior_mix != 0.0:
            raise SystemExit(f"--protocol {args.protocol} requires --heuristic-prior-mix 0")
        if args.protocol == "no_agent_manual" and args.reward_weights_file:
            raise SystemExit("--protocol no_agent_manual forbids --reward-weights-file")
        if args.protocol == "auto_reward_search" and not args.reward_weights_file:
            raise SystemExit("--protocol auto_reward_search requires --reward-weights-file")
    if args.protocol in {"self_generated_curriculum", "auto_reward_curriculum"}:
        args.disable_stage_action_filter = True
        args.stage_value_mode = "binary"
        if not args.start_route:
            raise SystemExit(f"--protocol {args.protocol} requires --start-route")
        if not args.start_route_provenance:
            raise SystemExit(f"--protocol {args.protocol} requires --start-route-provenance")
        if args.init_checkpoint:
            raise SystemExit(f"--protocol {args.protocol} forbids --init-checkpoint")
        if args.heuristic_prior_mix != 0.0:
            raise SystemExit(f"--protocol {args.protocol} requires --heuristic-prior-mix 0")
        if args.protocol == "self_generated_curriculum" and args.reward_weights_file:
            raise SystemExit("--protocol self_generated_curriculum forbids --reward-weights-file")
        if args.protocol == "auto_reward_curriculum" and not args.reward_weights_file:
            raise SystemExit("--protocol auto_reward_curriculum requires --reward-weights-file")
        route_path = Path(args.start_route)
        provenance_path = Path(args.start_route_provenance)
        if not route_path.exists():
            raise SystemExit(f"start route does not exist: {route_path}")
        if not provenance_path.exists():
            raise SystemExit(f"start route provenance does not exist: {provenance_path}")
        lowered = str(route_path).lower()
        if any(token in lowered for token in ("hp403", "manual", "alpha4090_boss_success")):
            raise SystemExit(f"{args.protocol} rejects suspicious route path: {route_path}")

    try:
        import numpy as np
        import torch
    except Exception as exc:
        raise SystemExit(f"Alpha Mota training requires numpy and torch: {exc}") from exc
    from mota_rl.graph_policy_value_model import GraphPolicyValueConfig, GraphPolicyValueNet

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf8")

    model = GraphPolicyValueNet(
        GraphPolicyValueConfig(
            d_model=args.d_model,
            nhead=args.heads,
            num_layers=args.layers,
            dropout=args.dropout,
        )
    ).to(device)
    if args.init_checkpoint:
        payload = torch.load(args.init_checkpoint, map_location=device)
        state_dict = payload.get("model_state_dict") or payload.get("policy_state_dict") or payload
        model.load_state_dict(state_dict)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sim_config = SimulatorConfig(
        allow_negative_hp=args.allow_negative_hp,
        min_hp=args.relaxed_min_hp,
        stop_on_boss=args.target_stage != "boss_all_gems",
    )
    replay: list[dict[str, Any]] = []
    rng = random.Random(args.seed)
    best_score = -10**18
    best_mode_scores = {"hp": -10**18, "resource": -10**18}
    start = time.time()
    episode_log = out_dir / "episodes.jsonl"

    with episode_log.open("w", encoding="utf8") as handle:
        for episode in range(args.episodes):
            sim = MotaSimulator(load_game_data(args.data), sim_config)
            result = run_episode(
                model=model,
                sim=sim,
                args=args,
                device=device,
                seed=args.seed + episode * 10_000,
                episode_index=episode,
            )
            replay.extend(result["examples"])
            if len(replay) > args.replay_size:
                replay = replay[-args.replay_size :]
            losses = []
            if replay:
                for _ in range(args.train_steps_per_episode):
                    losses.append(train_step(model, optimizer, replay, args, rng, device))
            finite_losses = [
                float(row["loss"])
                for row in losses
                if isinstance(row.get("loss"), (int, float)) and row["loss"] == row["loss"]
            ]
            mean_loss = sum(finite_losses) / len(finite_losses) if finite_losses else 0.0
            skipped_updates = sum(1 for row in losses if row.get("skipped"))
            valid_policy_rows = [
                float(row["valid_policy_rows"])
                for row in losses
                if isinstance(row.get("valid_policy_rows"), (int, float))
            ]
            row = {
                "episode": episode,
                "target_stage": args.target_stage,
                "target_success": result["target_success"],
                "boss_success": result["boss_success"],
                "boss_margin": result["boss_margin"],
                "macro_steps": result["macro_steps"],
                "primitive_steps": result["primitive_steps"],
                "final": result["final"],
                "value_target": result["value_target"],
                "replay_size": len(replay),
                "mean_loss": mean_loss,
                "skipped_updates": skipped_updates,
                "valid_policy_rows": (
                    sum(valid_policy_rows) / len(valid_policy_rows) if valid_policy_rows else None
                ),
                "elapsed_sec": round(time.time() - start, 1),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps(row, ensure_ascii=False), flush=True)
            final_keys = result["final"].get("keys", {}) or {}
            resource_score = 0.0
            if args.target_stage in {
                "shield",
                "shield_buffer",
                "mid_gems",
                "low_gems",
                "mt8_gems",
                "red_key",
                "boss_ready",
                "trap",
                "boss",
            }:
                resource_score = (
                    float(final_keys.get("yellowKey", 0)) * 90.0
                    + float(final_keys.get("blueKey", 0)) * 130.0
                    + float(final_keys.get("redKey", 0)) * 250.0
                    + float(result["final"].get("money", 0)) * 0.8
                )
            score = (
                int(result["target_success"]) * 1_000_000
                + float(result["final"].get("hp", 0))
                + float(result["final"].get("atk", 0)) * 100
                + float(result["final"].get("def", 0)) * 100
                + resource_score
                - result["macro_steps"] * 2
            )
            if result["route"] and score > best_score:
                best_score = score
                write_route_jsonl(result["route"], out_dir / "best_route.jsonl")
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "args": vars(args),
                        "episode": row,
                    },
                    out_dir / "best_model.pt",
                )
            for mode in ("hp", "resource"):
                mode_score = scalar_route_score(result, mode=mode)
                if result["route"] and mode_score > best_mode_scores[mode]:
                    best_mode_scores[mode] = mode_score
                    write_route_jsonl(result["route"], out_dir / f"best_route_{mode}.jsonl")
                    (out_dir / f"best_route_{mode}_summary.json").write_text(
                        json.dumps(
                            {
                                "mode": mode,
                                "score": mode_score,
                                "episode": episode,
                                "components": route_score_components(result),
                                "row": row,
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf8",
                    )
            if args.save_every > 0 and (episode + 1) % args.save_every == 0:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "args": vars(args),
                        "episode": row,
                    },
                    out_dir / f"checkpoint_ep{episode + 1}.pt",
                )
    torch.save({"model_state_dict": model.state_dict(), "args": vars(args)}, out_dir / "final_model.pt")
    print(json.dumps({"out_dir": str(out_dir), "best_score": best_score}, ensure_ascii=False))


if __name__ == "__main__":
    main()
