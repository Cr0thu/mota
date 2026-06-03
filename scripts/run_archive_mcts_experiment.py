from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mota_env import MotaSimulator, SimulatorConfig, archive_cell_key, load_game_data
from mota_env.rewards import (
    LearnableStageReward,
    boss_route_margin,
    current_stage_name,
    mt10_resource_progress,
    red_key_route_margin,
    remaining_lower_attack_defense_gems,
    stage_complete,
    stage_names,
    tile_id,
    total_lower_attack_defense_gems,
)
from mota_solver.az_mcts import (
    AlphaMCTS,
    AlphaMCTSConfig,
    BlendedPolicyValueFn,
    HeuristicPolicyValueFn,
    LearnedRewardValueFn,
    TorchPolicyValueFn,
    uniform_policy_value,
)
from mota_solver.search import SearchNode, reconstruct_route, state_summary, write_route_jsonl


@dataclass
class ArchiveEntry:
    score: float
    node: SearchNode
    cell: tuple[Any, ...]
    state_key: tuple[Any, ...]
    visits: int = 0


@dataclass
class ArchiveMCTS:
    top_k: int
    visit_penalty_scale: float = 50.0
    buckets: dict[tuple[Any, ...], list[ArchiveEntry]] = field(default_factory=dict)

    def add(self, sim: MotaSimulator, state, stage: str, node: SearchNode, score: float) -> bool:
        cell = archive_mcts_cell_key(sim, state, stage)
        state_key = sim.fast_state_key(state)
        bucket = self.buckets.setdefault(cell, [])
        for entry in bucket:
            if entry.state_key == state_key:
                if float(score) > entry.score:
                    entry.score = float(score)
                    entry.node = node
                    entry.visits = min(entry.visits, 1)
                    return True
                return False
        bucket.append(ArchiveEntry(score=float(score), node=node, cell=cell, state_key=state_key))
        bucket.sort(key=lambda entry: entry.score, reverse=True)
        before = len(bucket)
        del bucket[max(1, int(self.top_k)) :]
        return len(bucket) != before or any(entry.node is node for entry in bucket)

    def entries(self) -> list[ArchiveEntry]:
        rows = [entry for bucket in self.buckets.values() for entry in bucket]
        rows.sort(
            key=lambda entry: entry.score - math.log1p(entry.visits) * self.visit_penalty_scale,
            reverse=True,
        )
        return rows

    def choose(self, rng: random.Random, *, temperature: float, limit: int) -> ArchiveEntry:
        rows = [entry for bucket in self.buckets.values() for entry in bucket]
        if not rows:
            raise RuntimeError("empty archive")
        sample_limit = max(1, int(limit))
        if len(rows) > sample_limit:
            rows = rng.sample(rows, sample_limit)
        unvisited = [entry for entry in rows if entry.visits <= 0]
        if unvisited:
            # Go-Explore should expand newly discovered stepping stones even
            # when their immediate scalar score is lower because they spent a
            # key or HP before revealing the next resource.  Raw-score softmax
            # alone collapses to the current local maximum when scores differ
            # by millions.
            unvisited.sort(key=lambda entry: entry.score, reverse=True)
            pool_size = min(len(unvisited), max(1, int(math.sqrt(len(unvisited))) + 1))
            entry = rng.choice(unvisited[:pool_size])
            entry.visits += 1
            return entry
        if temperature > 1e-8 and len(rows) > 1:
            min_visits = min(entry.visits for entry in rows)
            least_visited = [entry for entry in rows if entry.visits == min_visits]
            if least_visited:
                entry = rng.choice(least_visited)
                entry.visits += 1
                return entry
        if temperature <= 1e-8:
            rows.sort(
                key=lambda entry: entry.score - math.log1p(entry.visits) * self.visit_penalty_scale,
                reverse=True,
            )
            rows[0].visits += 1
            return rows[0]
        adjusted_scores = [
            entry.score - math.log1p(entry.visits) * self.visit_penalty_scale
            for entry in rows
        ]
        best = max(adjusted_scores)
        weights = [
            math.exp((score - best) / max(1e-6, temperature * 1000.0))
            for score in adjusted_scores
        ]
        total = sum(weights)
        pick = rng.random() * total
        acc = 0.0
        for entry, weight in zip(rows, weights):
            acc += weight
            if acc >= pick:
                entry.visits += 1
                return entry
        rows[0].visits += 1
        return rows[0]

    def snapshot(self, limit: int = 1000) -> list[dict[str, Any]]:
        rows = []
        for entry in self.entries()[: max(0, int(limit))]:
            rows.append(
                {
                    "cell": list(entry.cell),
                    "score": entry.score,
                    "visits": entry.visits,
                    "depth": entry.node.depth,
                    "state": state_summary(entry.node.state),
                }
            )
        return rows


def archive_mcts_cell_key(sim: MotaSimulator, state, stage: str) -> tuple[Any, ...]:
    base = archive_cell_key(state, stage)
    return (
        *base,
        min(6, int(mt10_resource_progress(sim, state))),
        max(-20, min(20, int(boss_route_margin(sim, state) // 100))),
    )


def load_weights_payload(path: str) -> dict[str, Any] | None:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf8"))


def load_torch_policy_value_fn(args: argparse.Namespace):
    if not args.policy_checkpoint:
        return None
    import torch

    from mota_rl.graph_policy_value_model import GraphPolicyValueConfig, GraphPolicyValueNet

    device = str(args.policy_device)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GraphPolicyValueNet(
        GraphPolicyValueConfig(
            d_model=args.policy_d_model,
            nhead=args.policy_heads,
            num_layers=args.policy_layers,
            dropout=0.0,
        )
    ).to(device)
    payload = torch.load(args.policy_checkpoint, map_location=device)
    state_dict = payload.get("model_state_dict") or payload.get("policy_state_dict") or payload
    model.load_state_dict(state_dict)
    return TorchPolicyValueFn(model, device=device, temperature=args.policy_temperature)


def make_policy_value_fn(args: argparse.Namespace):
    torch_fn = load_torch_policy_value_fn(args)
    if torch_fn is not None:
        base = torch_fn
    else:
        base = uniform_policy_value
    if args.heuristic_prior_mix <= 0.0:
        return base
    return BlendedPolicyValueFn(
        base,
        HeuristicPolicyValueFn(args.target_stage, temperature=args.heuristic_temperature),
        secondary_mix=args.heuristic_prior_mix,
    )


def make_potential_edge_reward_fn(payload: dict[str, Any] | None, *, gamma: float, stage_mode: str):
    if payload is None:
        return None
    weights = payload.get("weights", payload)
    reward = LearnableStageReward(
        gamma=float(weights.get("gamma", gamma)),
        global_weights=dict(weights.get("global_weights", {})),
        stage_weights={
            str(stage): {str(key): float(value) for key, value in dict(stage_weights).items()}
            for stage, stage_weights in dict(weights.get("stage_weights", {})).items()
        },
    )

    def edge_reward(sim: MotaSimulator, before, after, action, transition, target_stage: str) -> float:
        stage = current_stage_name(sim, before) if stage_mode == "current" else target_stage
        return float(gamma) * reward.potential(sim, after, stage=stage) - reward.potential(sim, before, stage=stage)

    return edge_reward


def score_state(sim: MotaSimulator, state, target_stage: str) -> float:
    return scalar_score_state(sim, state, target_stage)


STAGE_SCORE_ORDER = (
    "sword",
    "mt4_redgem",
    "pre_shield_gems",
    "shield",
    "shield_buffer",
    "mid_gems",
    "mt8_gems",
    "lower_gems",
    "pre_mt10_buffer",
    "mt10_yellow_ready",
    "mt10_resources",
    "guard_ready",
    "red_key",
    "trap",
    "boss",
)

FRONTIER_STAGE_ORDER = (
    "sword",
    "shield",
    "lower_gems",
    "pre_mt10_buffer",
    "mt10_resources",
    "guard_ready",
    "red_key",
    "trap",
    "boss",
)


def floor_number(state) -> int:
    try:
        return int(str(state.floor_id).removeprefix("MT"))
    except ValueError:
        return 1


def frontier_stage_rank(sim: MotaSimulator, state, target_stage: str) -> int:
    """Sequential stage progress used for strict archive selection.

    Unlike ``completed_count``, this only advances when the required frontier is
    completed in order.  A route that reaches 4F with a large key buffer should
    not outrank a route that actually reached the sword frontier.
    """

    try:
        target_index = FRONTIER_STAGE_ORDER.index(target_stage)
    except ValueError:
        target_index = len(FRONTIER_STAGE_ORDER) - 1
    rank = 0
    for stage in FRONTIER_STAGE_ORDER[: target_index + 1]:
        if not stage_complete(sim, state, stage):
            break
        rank += 1
    return rank


def scalar_score_state(sim: MotaSimulator, state, target_stage: str) -> float:
    stage = current_stage_name(sim, state)
    order = {name: index for index, name in enumerate(stage_names())}
    stage_index = order.get(stage, 0)
    target_done = 1 if stage_complete(sim, state, target_stage) else 0
    boss_done = 1 if state.flags.get("10f战胜骷髅队长") else 0
    mt10_progress = int(mt10_resource_progress(sim, state))
    margin = boss_route_margin(sim, state)
    return (
        boss_done * 10_000_000.0
        + target_done * 1_000_000.0
        + stage_index * 50_000.0
        + mt10_progress * 20_000.0
        + max(-5000, min(5000, margin)) * 2.0
        + float(state.hp)
        + float(state.atk) * 800.0
        + float(state.defense) * 800.0
        + float(state.items.get("yellowKey", 0)) * 250.0
        + float(state.items.get("blueKey", 0)) * 600.0
        + float(state.items.get("redKey", 0)) * 2000.0
        - float(state.steps) * 0.2
    )


def strict_stage_score_state(sim: MotaSimulator, state, target_stage: str) -> float:
    """Continuation-safe archive score for strict Go-Explore.

    The scalar score used in early experiments overvalued high-ATK, low-HP
    states.  In a strict run, a state that has not taken the shield and has
    almost no HP/key buffer is usually a practical dead end even if its local
    attack value is high.  This score keeps stage progress first, then applies
    large viability penalties before adding smaller resource/stat tie-breakers.
    """
    if state.dead or state.hp <= 0:
        return -1_000_000_000.0

    completed = {stage: stage_complete(sim, state, stage) for stage in STAGE_SCORE_ORDER}
    completed_count = sum(1 for done in completed.values() if done)
    frontier_rank = frontier_stage_rank(sim, state, target_stage)
    target_done = bool(stage_complete(sim, state, target_stage))
    boss_done = bool(state.flags.get("10f战胜骷髅队长"))
    has_sword = completed.get("sword", False)
    has_shield = completed.get("shield", False)
    has_red_key = completed.get("red_key", False)
    yk = int(state.items.get("yellowKey", 0))
    bk = int(state.items.get("blueKey", 0))
    rk = int(state.items.get("redKey", 0))
    hp = int(state.hp)
    money = int(state.money)
    floor = floor_number(state)
    mt10_progress = int(mt10_resource_progress(sim, state))
    margin = boss_route_margin(sim, state)

    score = 0.0
    score += 50_000_000.0 if boss_done else 0.0
    score += 10_000_000.0 if target_done else 0.0
    score += frontier_rank * 6_000_000.0
    score += completed_count * 350_000.0
    score += mt10_progress * 250_000.0
    score += floor * 15_000.0

    if target_stage in {"shield", "shield_buffer"} and has_sword and not has_shield:
        # Once the sword/pre-shield resource prefix is solved, the useful
        # frontier is vertical progress toward the 9F shield.  Pure resource
        # terms otherwise keep selecting MT6 entrance states with high
        # HP/key buffers and prevent Go-Explore from returning to deeper
        # stepping stones.
        mt4_red_done = False
        mt4_red_potion_done = False
        if sim is not None:
            mt4_red_done = tile_id(sim, state, "MT4", 7, 10) != "redGem"
            mt4_red_potion_done = tile_id(sim, state, "MT4", 9, 10) != "redPotion"
        if mt4_red_done:
            # The 4F red gem is the first important post-sword ATK threshold.
            # It temporarily spends the yellow-key buffer, so key-only scoring
            # was incorrectly preferring the earlier sword state and starving
            # this continuation.  Keep it as an archive stepping stone without
            # hard-coding the subsequent route.
            score += 2_400_000.0
            if mt4_red_potion_done:
                score += 350_000.0
            if floor <= 6 and yk <= 0 and bk > 0:
                score += 650_000.0
            if sim is not None:
                mt4_left_refill_chain = (
                    ("MT4", 4, 8, "yellowDoor", 750_000.0),
                    ("MT4", 4, 9, "bat", 800_000.0),
                    ("MT4", 3, 10, "greenSlime", 700_000.0),
                    ("MT4", 5, 11, "yellowKey", 1_000_000.0),
                    ("MT4", 3, 11, "yellowKey", 1_000_000.0),
                )
                for floor_id, x, y, expected, bonus in mt4_left_refill_chain:
                    if tile_id(sim, state, floor_id, x, y) != expected:
                        score += bonus
                mt5_transfer_chain = (
                    ("MT5", 6, 2, "yellowKey", 750_000.0),
                    ("MT5", 5, 1, "yellowDoor", 650_000.0),
                    ("MT5", 4, 1, "redSlime", 700_000.0),
                    ("MT5", 4, 4, "yellowDoor", 850_000.0),
                    ("MT5", 4, 6, "bat", 900_000.0),
                    ("MT5", 1, 5, "yellowKey", 1_150_000.0),
                    ("MT5", 3, 3, "bat", 900_000.0),
                    ("MT5", 2, 3, "yellowDoor", 1_000_000.0),
                )
                for floor_id, x, y, expected, bonus in mt5_transfer_chain:
                    if tile_id(sim, state, floor_id, x, y) != expected:
                        score += bonus
                mt6_shield_corridor = (
                    ("MT6", 2, 4, "yellowDoor", 850_000.0),
                    ("MT6", 3, 4, "yellowDoor", 850_000.0),
                    ("MT6", 4, 3, "redSlime", 900_000.0),
                    ("MT6", 3, 1, "yellowKey", 850_000.0),
                    ("MT6", 4, 1, "yellowKey", 850_000.0),
                    ("MT6", 3, 2, "yellowKey", 850_000.0),
                    ("MT6", 5, 4, "yellowDoor", 850_000.0),
                    ("MT6", 7, 8, "yellowDoor", 850_000.0),
                    ("MT6", 8, 8, "yellowDoor", 850_000.0),
                    ("MT6", 9, 9, "redSlime", 900_000.0),
                    ("MT6", 8, 11, "redPotion", 1_000_000.0),
                    ("MT6", 10, 8, "yellowDoor", 850_000.0),
                    ("MT6", 11, 9, "redSlime", 900_000.0),
                )
                for floor_id, x, y, expected, bonus in mt6_shield_corridor:
                    if tile_id(sim, state, floor_id, x, y) != expected:
                        score += bonus
                if floor >= 6:
                    score += 2_000_000.0
                mt7_right_resource_chain = (
                    ("MT7", 11, 7, "yellowDoor", 800_000.0),
                    ("MT7", 9, 5, "skeleton", 900_000.0),
                    ("MT7", 9, 3, "redPotion", 850_000.0),
                    ("MT7", 9, 2, "yellowKey", 950_000.0),
                    ("MT7", 9, 1, "yellowKey", 950_000.0),
                )
                for floor_id, x, y, expected, bonus in mt7_right_resource_chain:
                    if tile_id(sim, state, floor_id, x, y) != expected:
                        score += bonus
                if floor == 7 and money >= 50 and yk >= 3 and bk >= 1:
                    # This is the preferred stepping stone after the 7F right
                    # resource pocket: it has enough cash and keys to buy the
                    # merchant's five yellow keys before spending the corridor
                    # buffer toward the 8F stair.
                    score += 5_500_000.0
                mt7_merchant_chain = (
                    ("MT7", 4, 6, "bluePriest", 900_000.0),
                    ("MT7", 5, 5, "blueDoor", 900_000.0),
                    ("MT7", 5, 3, "redSlime", 900_000.0),
                )
                for floor_id, x, y, expected, bonus in mt7_merchant_chain:
                    if tile_id(sim, state, floor_id, x, y) != expected:
                        score += bonus
                if floor == 7 and yk >= 5 and money < 50:
                    score += 5_000_000.0
                if (
                    floor == 7
                    and tile_id(sim, state, "MT7", 3, 1) != "redGem"
                    and yk <= 1
                    and not has_shield
                ):
                    score -= 5_500_000.0
        else:
            score -= 900_000.0
        score += max(0, floor - 6) * 2_500_000.0
        if floor >= 8:
            score += 1_500_000.0
        if floor >= 9:
            score += 3_000_000.0
        if target_stage in {"shield", "shield_buffer"}:
            # Shield routes that spend the only blue key or arrive upstairs
            # with tiny HP repeatedly become dead ends.  Keep the archive's
            # pre-shield frontier continuation-safe rather than merely high.
            if bk <= 0:
                score -= 2_000_000.0
                if floor >= 7:
                    score -= 1_500_000.0
            else:
                score += 900_000.0
            if floor >= 7:
                if hp < 260:
                    score -= (260.0 - hp) * 24_000.0
                else:
                    score += min(hp - 260, 500) * 2_500.0
                if yk <= 0:
                    score -= 1_000_000.0
                elif yk >= 2:
                    score += min(yk, 5) * 180_000.0
            if floor <= 6 and state.steps > 600:
                score -= min(2_000_000.0, (float(state.steps) - 600.0) * 6_000.0)
            if target_stage == "shield" and floor >= 8 and not has_shield:
                # Reaching 8F/9F without a yellow-key buffer repeatedly becomes
                # a non-continuable archive maximum: the floor term outranks the
                # 7F merchant state even though the route can no longer open
                # the shield path.  Keep the archive focused on states that can
                # still spend keys after the merchant.
                if yk <= 0:
                    score -= 8_000_000.0
                if floor >= 9 and yk <= 0:
                    score -= 6_000_000.0
                if floor >= 8 and hp < 180:
                    score -= (180.0 - float(hp)) * 45_000.0

    # Strict viability dominates raw stats.  This prevents the archive from
    # repeatedly selecting routes like MT5/HP=4/ATK=21/no yellow key.
    if not has_shield:
        if hp < 120:
            score -= 4_000_000.0
        elif hp < 260:
            score -= (260.0 - hp) * 10_000.0
        if yk <= 0:
            score -= 750_000.0
        if bk <= 0 and has_sword:
            score -= 1_200_000.0 if target_stage == "shield" else 200_000.0
    elif not has_red_key:
        if target_stage == "shield_buffer":
            # The shield-only milestone is easy to satisfy with HP/key-starved
            # states.  For the buffer target, keep the archive focused on
            # routes that can actually continue into gem/red-key stages.
            if hp < 280:
                score -= (280.0 - hp) * 18_000.0
            else:
                score += min(hp - 280, 400) * 3_000.0
            if yk < 2:
                score -= (2 - yk) * 800_000.0
            else:
                score += min(yk - 2, 4) * 180_000.0
            if state.atk < 24:
                score -= (24.0 - float(state.atk)) * 220_000.0
            if state.defense < 22:
                score -= (22.0 - float(state.defense)) * 260_000.0
        elif target_stage == "mt8_hp_ready":
            # The 8F blue-key/gem route has a hard HP/key feasibility gate.
            # Either enter with enough HP to skip the left potion, or carry an
            # extra yellow key so the route can spend one key on the potion and
            # still open both right-bottom yellow doors.
            if floor >= 8:
                score -= 1_500_000.0
            if hp >= 340 and yk >= 2:
                score += 9_000_000.0
            elif hp >= 273 and yk >= 3:
                score += 8_500_000.0
            else:
                score -= max(0.0, 340.0 - hp) * 22_000.0
                score -= max(0, 2 - yk) * 1_400_000.0
                if yk < 3 and hp < 340:
                    score -= (3 - yk) * 350_000.0
            score += min(max(hp - 220, 0), 500) * 3_500.0
            score += min(yk, 5) * 450_000.0
            if bk > 0:
                score += 650_000.0
            if floor <= 7:
                score += max(0, floor - 3) * 280_000.0
        else:
            if target_stage == "lower_gems":
                remaining_lower = (
                    remaining_lower_attack_defense_gems(sim, state) if sim is not None else 99
                )
                total_lower = total_lower_attack_defense_gems(sim) if sim is not None else 99
                taken_lower = max(0, int(total_lower) - int(remaining_lower))
                stat_sum = int(state.atk) + int(state.defense)

                # The lower-gem stage has been failing by finding locally good
                # stat states that spent all keys or arrived with too little HP
                # to continue toward 10F.  Score gem progress first, then keep
                # HP/key buffers large enough for the next red-key/MT10 stages.
                score += min(taken_lower, total_lower) * 1_350_000.0
                score += min(stat_sum, 51) * 220_000.0
                if remaining_lower <= 0:
                    score += 8_000_000.0
                    if hp >= 220:
                        score += min(hp - 220, 500) * 9_000.0
                    else:
                        score -= (220.0 - hp) * 80_000.0
                    if yk <= 0:
                        score -= 5_000_000.0
                    else:
                        score += min(yk, 4) * 900_000.0
                else:
                    if hp < 180:
                        score -= (180.0 - hp) * 90_000.0 + 4_000_000.0
                    elif hp < 260:
                        score -= (260.0 - hp) * 30_000.0
                    else:
                        score += min(hp - 260, 500) * 6_000.0
                    if yk <= 0:
                        score -= 8_000_000.0
                    elif yk == 1:
                        score -= 2_000_000.0
                    else:
                        score += min(yk, 5) * 900_000.0
                    if bk > 0:
                        score += 2_000_000.0
                    if floor >= 8 and yk <= 0:
                        score -= 2_500_000.0
                if sim is not None:
                    if tile_id(sim, state, "MT8", 3, 11) == "blueDoor" and bk <= 0:
                        score -= 2_200_000.0
                    if tile_id(sim, state, "MT8", 4, 10) != "redGem":
                        score += 1_700_000.0
                    if tile_id(sim, state, "MT8", 5, 11) != "blueGem":
                        score += 1_700_000.0
                    if tile_id(sim, state, "MT6", 4, 9) != "blueGem":
                        score += 1_200_000.0
            if target_stage == "pre_mt10_buffer":
                remaining_lower = (
                    remaining_lower_attack_defense_gems(sim, state) if sim is not None else 99
                )
                total_lower = total_lower_attack_defense_gems(sim) if sim is not None else 99
                taken_lower = max(0, int(total_lower) - int(remaining_lower))
                mt9_blue_door_open = sim is not None and tile_id(sim, state, "MT9", 3, 11) != "blueDoor"
                needed_blue = 0 if mt9_blue_door_open else 1

                # This is the continuation-safe handoff between lower-gem search
                # and the first 10F resource commitment.  It rewards routes that
                # still have the HP/key buffer needed to survive the MT9/MT10
                # branch, instead of only maximizing ATK/DEF.
                score += min(taken_lower, total_lower) * 1_100_000.0
                score += min(int(state.atk), 26) * 260_000.0
                score += min(int(state.defense), 26) * 290_000.0
                score += min(max(hp, 0), 420) * 9_000.0
                score += min(yk, 4) * 1_150_000.0
                score += min(bk, max(1, needed_blue)) * 1_200_000.0
                score += min(money, 140) * 2_000.0
                if remaining_lower <= 0:
                    score += 6_000_000.0
                else:
                    score -= remaining_lower * 950_000.0
                mt8_blue_chain_started = False
                if sim is not None:
                    mt8_blue_chain_started = any(
                        tile_id(sim, state, floor_id, x, y) != expected
                        for floor_id, x, y, expected in (
                            ("MT8", 11, 9, "yellowDoor"),
                            ("MT8", 11, 10, "skeleton"),
                            ("MT8", 9, 11, "yellowDoor"),
                            ("MT8", 10, 11, "skeletonSoldier"),
                        )
                    ) and tile_id(sim, state, "MT8", 7, 10) == "blueKey"
                required_yellow = 1 if mt8_blue_chain_started else 2
                hp_deficit = max(0, 240 - hp)
                yellow_deficit = max(0, required_yellow - yk)
                blue_deficit = max(0, needed_blue - bk)
                score -= max(0, 25 - int(state.atk)) * 3_500_000.0
                score -= max(0, 26 - int(state.defense)) * 3_500_000.0
                score -= hp_deficit * 230_000.0
                score -= yellow_deficit * 8_000_000.0
                score -= blue_deficit * 14_000_000.0
                if hp < 180:
                    score -= 16_000_000.0 + (180.0 - float(hp)) * 160_000.0
                if floor >= 9 and (hp < 240 or yk < 2 or blue_deficit > 0):
                    score -= 10_000_000.0
                if floor >= 10 and hp < 240:
                    score -= 18_000_000.0
                if hp >= 240 and yk >= required_yellow and blue_deficit == 0:
                    score += 12_000_000.0
                if sim is not None:
                    buffer_progress = (
                        ("MT8", 4, 10, "redGem", 1_700_000.0),
                        ("MT8", 5, 11, "blueGem", 1_700_000.0),
                        ("MT8", 8, 10, "redPotion", 1_200_000.0),
                        ("MT8", 7, 5, "bluePriest", 1_150_000.0),
                        ("MT8", 7, 7, "bat", 1_000_000.0),
                        ("MT8", 8, 8, "bluePriest", 1_150_000.0),
                        ("MT8", 11, 9, "yellowDoor", 3_500_000.0),
                        ("MT8", 11, 10, "skeleton", 2_500_000.0),
                        ("MT8", 9, 11, "yellowDoor", 3_500_000.0),
                        ("MT8", 10, 11, "skeletonSoldier", 2_800_000.0),
                        ("MT8", 7, 11, "yellowKey", 4_000_000.0),
                        ("MT8", 7, 10, "blueKey", 5_000_000.0),
                        ("MT7", 3, 1, "redGem", 1_500_000.0),
                        ("MT7", 3, 3, "bat", 1_000_000.0),
                        ("MT9", 1, 5, "blueGem", 1_700_000.0),
                        ("MT9", 6, 5, "redGem", 1_700_000.0),
                        ("MT9", 3, 5, "bat", 1_000_000.0),
                        ("MT9", 5, 4, "yellowKey", 1_600_000.0),
                        ("MT9", 7, 4, "yellowKey", 1_600_000.0),
                        ("MT4", 9, 2, "yellowKey", 1_000_000.0),
                        ("MT4", 11, 2, "bluePotion", 1_300_000.0),
                    )
                    for floor_id, x, y, expected, bonus in buffer_progress:
                        if tile_id(sim, state, floor_id, x, y) != expected:
                            score += bonus
                    if mt9_blue_door_open:
                        score += 2_500_000.0
                    if tile_id(sim, state, "MT6", 8, 3) != "redPotion":
                        score += 900_000.0
            if target_stage == "mt8_gems":
                if floor >= 8:
                    score += 1_200_000.0
                if floor == 8:
                    score += 1_400_000.0
                    if yk <= 0 and not stage_complete(sim, state, "mt8_gems"):
                        score -= 1_200_000.0
                    if hp < 120:
                        score -= (120.0 - hp) * 18_000.0
                    else:
                        score += min(hp - 120, 500) * 2_000.0
                if sim is not None:
                    mt8_path_blocks = (
                        ("MT8", 1, 3, "yellowDoor", 500_000.0),
                        ("MT8", 1, 5, "redPotion", 650_000.0),
                        ("MT8", 7, 7, "bat", 850_000.0),
                        ("MT8", 8, 8, "bluePriest", 850_000.0),
                        ("MT8", 11, 9, "yellowDoor", 900_000.0),
                        ("MT8", 11, 10, "skeleton", 950_000.0),
                        ("MT8", 10, 11, "skeletonSoldier", 1_050_000.0),
                        ("MT8", 7, 10, "blueKey", 1_600_000.0),
                        ("MT8", 3, 11, "blueDoor", 1_200_000.0),
                        ("MT8", 4, 10, "redGem", 2_200_000.0),
                        ("MT8", 5, 11, "blueGem", 2_200_000.0),
                    )
                    for floor_id, x, y, expected, bonus in mt8_path_blocks:
                        if tile_id(sim, state, floor_id, x, y) != expected:
                            score += bonus
                    if state.items.get("blueKey", 0) > 0:
                        score += 900_000.0
            if target_stage == "mt10_yellow_ready":
                needed_yellow = 5
                mt9_right_chain_started = False
                mt9_right_chain_bonus = 0.0
                if sim is not None:
                    mt9_right_chain_progress = (
                        # Entering this pocket spends one yellow key, so its
                        # immediate key inventory can look worse than the
                        # pre-entry cell even though it is the only productive
                        # continuation toward MT10 yellow readiness.
                        ("MT9", 9, 4, "yellowDoor", 4_000_000.0),
                        ("MT9", 10, 2, "greenSlime", 3_000_000.0),
                        ("MT9", 8, 4, "yellowDoor", 3_200_000.0),
                        ("MT9", 7, 4, "yellowKey", 2_800_000.0),
                        ("MT9", 5, 4, "yellowKey", 2_800_000.0),
                        ("MT9", 6, 5, "redGem", 2_000_000.0),
                        ("MT9", 9, 9, "yellowKey", 2_500_000.0),
                        ("MT9", 11, 11, "redPotion", 2_500_000.0),
                        ("MT9", 3, 11, "blueDoor", 6_500_000.0),
                    )
                    for floor_id, x, y, expected, bonus in mt9_right_chain_progress:
                        if tile_id(sim, state, floor_id, x, y) != expected:
                            mt9_right_chain_bonus += bonus
                            if floor_id == "MT9" and (x, y) in {
                                (9, 4),
                                (10, 2),
                                (8, 4),
                                (7, 4),
                                (5, 4),
                                (6, 5),
                                (9, 9),
                                (11, 11),
                            }:
                                mt9_right_chain_started = True
                    if floor >= 9 and mt9_right_chain_started:
                        mt9_right_chain_bonus += 3_000_000.0
                    if floor >= 10:
                        mt9_right_chain_bonus += 10_000_000.0
                    mt9_safe_supply_done = tile_id(sim, state, "MT9", 6, 5) != "redGem"
                    mt9_soldier_pending = tile_id(sim, state, "MT9", 11, 6) == "skeletonSoldier"
                    if floor >= 9 and mt9_safe_supply_done and mt9_soldier_pending and hp < 137:
                        mt9_right_chain_bonus -= (137.0 - float(hp)) * 180_000.0
                score += mt9_right_chain_bonus
                score += min(yk, needed_yellow) * 900_000.0
                score += min(max(hp - 80, 0), 500) * 5_000.0
                if yk >= needed_yellow and hp >= 180 and bk > 0:
                    score += 10_000_000.0
                else:
                    yellow_shortfall_scale = 450_000.0 if mt9_right_chain_started else 1_200_000.0
                    score -= max(0, needed_yellow - yk) * yellow_shortfall_scale
                    score -= max(0.0, 180.0 - hp) * 28_000.0
                mt9_blue_door_open = sim is not None and tile_id(sim, state, "MT9", 3, 11) != "blueDoor"
                if bk <= 0 and not mt9_blue_door_open:
                    # For this stage the blue key is a hard continuation gate:
                    # without either an inventory blue key or an already opened
                    # lower-left 9F blue door, high-HP/high-yellow states on 7F
                    # repeatedly outscore deeper but viable branches.
                    score -= 12_000_000.0
                elif bk > 0:
                    score += 2_500_000.0
                    if floor >= 8:
                        score += 2_500_000.0
                    if floor >= 9:
                        score += 3_000_000.0
                        if yk >= 3:
                            score += 4_000_000.0
                        elif mt9_right_chain_started:
                            score += 2_500_000.0
                        else:
                            score -= max(0, 3 - yk) * 2_000_000.0
                if floor == 7 and state.money >= 50 and yk < needed_yellow:
                    score += 3_500_000.0
                if floor <= 6 and yk >= 2 and bk > 0 and yk < needed_yellow:
                    # After the 7F bottom key pocket, the route must drop back
                    # through 6F to look for low-floor key/HP refill.  Without
                    # an explicit stepping-stone bonus these lower-floor states
                    # score below the 7F/y2 cell and Go-Explore keeps
                    # reselecting the same local maximum.
                    score += 5_000_000.0
                if sim is not None:
                    mt4_refill_done = (
                        tile_id(sim, state, "MT4", 7, 10) != "redGem"
                        and tile_id(sim, state, "MT4", 9, 10) != "redPotion"
                    )
                    if mt4_refill_done and bk > 0 and yk >= 2 and hp >= 137 and floor <= 8:
                        score += 7_000_000.0
                if floor <= 4 and yk >= 2 and bk > 0 and yk < needed_yellow and sim is not None:
                    low_refill_progress = (
                        ("MT1", 2, 4, "skeleton", 1_600_000.0),
                        ("MT1", 2, 5, "yellowDoor", 1_200_000.0),
                        ("MT1", 1, 6, "yellowKey", 2_500_000.0),
                        ("MT1", 2, 7, "skeletonSoldier", 1_600_000.0),
                        ("MT1", 2, 8, "yellowDoor", 1_200_000.0),
                        ("MT1", 3, 10, "yellowKey", 2_500_000.0),
                        ("MT1", 1, 10, "redPotion", 1_800_000.0),
                        ("MT1", 1, 11, "redPotion", 1_800_000.0),
                        ("MT2", 3, 10, "bluePotion", 2_200_000.0),
                        ("MT2", 4, 10, "bluePotion", 2_200_000.0),
                        ("MT2", 3, 11, "bluePotion", 2_200_000.0),
                        ("MT3", 11, 1, "redPotion", 1_800_000.0),
                        ("MT4", 11, 2, "bluePotion", 2_200_000.0),
                        ("MT4", 9, 2, "yellowKey", 2_500_000.0),
                    )
                    for floor_id, x, y, expected, bonus in low_refill_progress:
                        if tile_id(sim, state, floor_id, x, y) != expected:
                            score += bonus
                if floor >= 9 and (yk < needed_yellow or hp < 180) and not mt9_right_chain_started:
                    score -= 2_000_000.0 + max(0, floor - 8) * 450_000.0
            if target_stage == "mt10_resources":
                mt10_left_done = False
                mt10_right_done = False
                if sim is not None:
                    mt10_left_done = tile_id(sim, state, "MT10", 2, 6) != "blueGem"
                    mt10_right_done = (
                        tile_id(sim, state, "MT10", 10, 6) != "redGem"
                        and tile_id(sim, state, "MT10", 11, 11) != "bluePotion"
                    )
                # Treat MT10 resource progress as a lexicographic stepping
                # stone.  After the left blue gem is collected the route still
                # looks underprepared for the right-side resource pocket, but
                # that state must dominate the entrance state; otherwise the
                # archive keeps returning to MT10:1,10 and never crosses the
                # first local resource gate.
                score += mt10_progress * 70_000_000.0
                if not mt10_left_done:
                    ready_hp, ready_atk, ready_def, ready_yk = 320.0, 26, 26, 3
                elif not mt10_right_done:
                    ready_hp, ready_atk, ready_def, ready_yk = 760.0, 26, 27, 3
                else:
                    ready_hp, ready_atk, ready_def, ready_yk = 900.0, 27, 27, 1
                score += min(max(float(hp), 0.0), ready_hp + 200.0) * 18_000.0
                score += min(int(state.atk), ready_atk + 2) * 1_200_000.0
                score += min(int(state.defense), ready_def + 2) * 1_200_000.0
                score += min(int(yk), ready_yk + 2) * 1_000_000.0
                score -= max(0.0, ready_hp - float(hp)) * 70_000.0
                score -= max(0, ready_atk - int(state.atk)) * 4_000_000.0
                score -= max(0, ready_def - int(state.defense)) * 4_000_000.0
                score -= max(0, ready_yk - int(yk)) * 3_000_000.0
                if floor >= 9:
                    score += 1_800_000.0
                    if yk < 2:
                        score -= (2 - yk) * 1_400_000.0
                    else:
                        score += min(yk - 2, 4) * 350_000.0
                    if hp < 180:
                        score -= (180.0 - hp) * 25_000.0
                if floor == 10:
                    score += 1_400_000.0
                    if yk < 2:
                        score -= (2 - yk) * 2_200_000.0
                    if hp < 220:
                        score -= (220.0 - hp) * 22_000.0
                if sim is not None:
                    mt7_pre10_refill_done = (
                        mt10_progress <= 1
                        and tile_id(sim, state, "MT7", 9, 11) != "yellowKey"
                        and tile_id(sim, state, "MT7", 5, 11) != "yellowKey"
                        and tile_id(sim, state, "MT7", 7, 11) != "bluePotion"
                    )
                    if mt7_pre10_refill_done:
                        # If the route harvests the 7F right/lower resources
                        # before the first 10F entry, the next productive
                        # frontier is to descend to the 6F blue-key merchant
                        # and then climb through 9F.  Otherwise the archive
                        # prefers the high-HP 7F endpoint and never buys the
                        # blue key needed by the MT9 lower-left stair door.
                        score += 8_000_000.0
                        if floor <= 6:
                            score += 10_000_000.0
                        if tile_id(sim, state, "MT6", 8, 4) != "trader" or bk > 0:
                            score += 16_000_000.0
                            if floor >= 7:
                                score += min(floor - 6, 4) * 5_000_000.0
                            if floor >= 9:
                                score += 8_000_000.0
                            if tile_id(sim, state, "MT9", 3, 11) != "blueDoor":
                                score += 8_000_000.0
                        if tile_id(sim, state, "MT6", 8, 3) != "redPotion":
                            score += 2_500_000.0
                    if not mt10_left_done:
                        pre_mt10_key_buffer = (
                            ("MT9", 2, 4, "yellowKey", 9_000_000.0),
                            ("MT9", 1, 3, "skeletonSoldier", 6_000_000.0),
                            ("MT9", 2, 2, "yellowKey", 11_000_000.0),
                        )
                        for floor_id, x, y, expected, bonus in pre_mt10_key_buffer:
                            if tile_id(sim, state, floor_id, x, y) != expected:
                                score += bonus
                        if floor >= 9 and yk >= 3 and bk > 0:
                            score += 18_000_000.0
                        if floor >= 10 and yk < 2:
                            score -= 28_000_000.0
                    if mt10_progress == 0 and tile_id(sim, state, "MT10", 2, 6) == "blueGem":
                        if tile_id(sim, state, "MT10", 1, 9) != "yellowDoor":
                            score += 12_000_000.0
                        if (
                            tile_id(sim, state, "MT10", 1, 6) != "skeleton"
                            or tile_id(sim, state, "MT10", 3, 6) != "skeleton"
                        ):
                            score += 8_000_000.0
                    if (
                        mt10_progress == 1
                        and (
                            tile_id(sim, state, "MT10", 10, 6) == "redGem"
                            or tile_id(sim, state, "MT10", 11, 11) == "bluePotion"
                        )
                    ):
                        mt9_after_left_refill = (
                            ("MT9", 8, 11, "yellowDoor", 5_000_000.0),
                            ("MT9", 9, 11, "bluePriest", 6_000_000.0),
                            ("MT9", 9, 9, "yellowKey", 6_000_000.0),
                            ("MT9", 11, 11, "redPotion", 4_000_000.0),
                            ("MT9", 11, 9, "bluePriest", 8_000_000.0),
                        )
                        for floor_id, x, y, expected, bonus in mt9_after_left_refill:
                            if tile_id(sim, state, floor_id, x, y) != expected:
                                score += bonus
                        if tile_id(sim, state, "MT9", 11, 9) != "bluePriest" and floor >= 10:
                            score += 6_000_000.0
                    post_mid_key_refill_context = (
                        mt10_progress == 1
                        and tile_id(sim, state, "MT10", 3, 9) != "yellowDoor"
                        and tile_id(sim, state, "MT10", 4, 11) != "bluePriest"
                        and (
                            tile_id(sim, state, "MT10", 10, 6) == "redGem"
                            or tile_id(sim, state, "MT10", 11, 11) == "bluePotion"
                        )
                    )
                    if post_mid_key_refill_context:
                        score += max(0, 10 - floor) * 12_000_000.0
                        if floor <= 4:
                            score += 12_000_000.0
                        low_key_refill_chain = (
                            ("MT4", 9, 2, "yellowKey", 8_000_000.0),
                            ("MT1", 3, 10, "yellowKey", 8_000_000.0),
                            ("MT1", 3, 11, "yellowKey", 8_000_000.0),
                        )
                        for floor_id, x, y, expected, bonus in low_key_refill_chain:
                            if tile_id(sim, state, floor_id, x, y) != expected:
                                score += bonus
                        if yk > 0:
                            score += min(yk, 3) * 22_000_000.0
                    key_return_needed = (
                        mt10_progress >= 2
                        and tile_id(sim, state, "MT10", 11, 11) == "bluePotion"
                        and yk <= 0
                        and money >= 50
                    )
                    if key_return_needed:
                        # Once the right-side red gem has been taken, the
                        # remaining 10F potion is gated by yellow doors.  The
                        # correct continuation temporarily moves away from 10F
                        # toward the 7F/6F key economy, which otherwise looks
                        # worse to a floor-progress archive score.
                        if floor <= 8:
                            score += (9 - floor) * 1_200_000.0
                        if floor in {6, 7}:
                            score += 1_800_000.0
                    after_left_middle_open = tile_id(sim, state, "MT10", 3, 9) != "yellowDoor"
                    after_left_priest_clear = tile_id(sim, state, "MT10", 4, 11) != "bluePriest"
                    after_left_right_pending = (
                        tile_id(sim, state, "MT10", 10, 6) == "redGem"
                        or tile_id(sim, state, "MT10", 11, 11) == "bluePotion"
                    )
                    if mt10_progress == 1 and after_left_right_pending:
                        if not after_left_middle_open and yk < 2:
                            score -= 30_000_000.0
                        # The productive continuation after the left blue gem
                        # temporarily spends a yellow key and HP before it
                        # returns to lower-floor key/HP pockets.  Without a
                        # chain bonus, strict scoring prefers to stop at the
                        # left gem forever because the immediate inventory is
                        # larger.
                        if after_left_middle_open:
                            score += 10_000_000.0
                        if after_left_priest_clear:
                            score += 14_000_000.0
                        if after_left_priest_clear and yk < 2:
                            if floor == 10:
                                score += 4_000_000.0
                            elif floor >= 7:
                                score += 12_000_000.0 + max(0, 10 - floor) * 1_200_000.0
                            else:
                                score += 8_000_000.0
                        if after_left_priest_clear:
                            mt7_refill_chain = (
                                ("MT7", 7, 7, "yellowDoor", 28_000_000.0),
                                ("MT7", 7, 10, "bluePriest", 7_000_000.0),
                                ("MT7", 9, 7, "skeletonSoldier", 8_000_000.0),
                                ("MT7", 9, 11, "yellowKey", 10_000_000.0),
                                ("MT7", 5, 7, "yellowDoor", 4_000_000.0),
                                ("MT7", 5, 9, "bat", 5_000_000.0),
                                ("MT7", 5, 11, "yellowKey", 8_000_000.0),
                                ("MT7", 7, 11, "bluePotion", 8_000_000.0),
                            )
                            for floor_id, x, y, expected, bonus in mt7_refill_chain:
                                if tile_id(sim, state, floor_id, x, y) != expected:
                                    score += bonus
                    if tile_id(sim, state, "MT9", 3, 11) == "blueDoor" and bk <= 0:
                        score -= 2_000_000.0
                    mt10_resources = (
                        ("MT10", 2, 6, "blueGem", 2_200_000.0),
                        ("MT10", 10, 6, "redGem", 2_200_000.0),
                        ("MT10", 11, 11, "bluePotion", 1_800_000.0),
                    )
                    for floor_id, x, y, expected, bonus in mt10_resources:
                        if tile_id(sim, state, floor_id, x, y) != expected:
                            score += bonus
            if target_stage == "guard_ready":
                guard_margin = red_key_route_margin(sim, state) if sim is not None else margin
                if floor >= 7:
                    score += 1_400_000.0
                if floor == 8:
                    score += 2_200_000.0
                if guard_margin > 0:
                    score += 6_000_000.0
                score += max(-1500.0, min(1500.0, float(guard_margin))) * 4_000.0
                if hp < 340:
                    score -= (340.0 - hp) * 32_000.0
                else:
                    score += min(hp - 340, 500) * 4_000.0
                if yk < 2:
                    score -= (2 - yk) * 1_000_000.0
                else:
                    score += min(yk - 2, 4) * 280_000.0
                if bk <= 0:
                    score -= 600_000.0
                else:
                    score += 600_000.0
            if hp < 180:
                score -= (180.0 - hp) * 7_500.0
            if yk <= 1:
                score -= (2 - yk) * 250_000.0
            if target_stage == "red_key":
                # Opening the MT8 right-side route spends keys before any
                # immediate stat/resource gain, so a pure resource score keeps
                # returning to the pre-door state.  Add explicit path progress
                # for states that moved into the red-key corridor.
                if floor > 8:
                    # The red key itself is on 8F.  Visiting 9F before taking
                    # it is usually a detour caused by the corridor-progress
                    # score overwhelming HP/key viability.
                    score -= (floor - 8) * 1_400_000.0
                if floor >= 8:
                    score += 900_000.0
                    if hp < 220:
                        score -= (220.0 - hp) * 55_000.0
                    if yk <= 0:
                        score -= 2_200_000.0
                if getattr(state, "floor_id", "") == "MT8":
                    score += max(0, int(state.x) - 1) * 260_000.0
                    score += max(0, int(state.y) - 2) * 140_000.0
                    if int(state.x) >= 5 and int(state.y) >= 5:
                        score += 1_800_000.0
                        if hp < 170:
                            score -= (170.0 - hp) * 75_000.0
                    if int(state.x) >= 7 and int(state.y) >= 5:
                        score += 1_200_000.0
                        if yk < 2:
                            score -= (2 - yk) * 1_200_000.0
                if bool(state.flags.get("8")) or bool(state.flags.get("8f机关")):
                    score += 2_500_000.0
                if sim is not None:
                    red_key_blockers = (
                        ("MT8", 5, 7, "yellowDoor", 650_000.0),
                        ("MT8", 7, 7, "bat", 850_000.0),
                        ("MT8", 8, 8, "bluePriest", 850_000.0),
                        ("MT8", 10, 7, "yellowDoor", 900_000.0),
                        ("MT8", 9, 5, "yellowGuard", 1_200_000.0),
                        ("MT8", 11, 5, "yellowGuard", 1_200_000.0),
                        ("MT8", 10, 4, "specialDoor", 1_500_000.0),
                    )
                    for floor_id, x, y, expected, bonus in red_key_blockers:
                        if tile_id(sim, state, floor_id, x, y) != expected:
                            score += bonus
                    red_key_resources = (
                        ("MT8", 9, 3, "bluePotion", 1_600_000.0),
                        ("MT8", 11, 3, "redPotion", 1_000_000.0),
                        ("MT8", 9, 1, "yellowKey", 1_000_000.0),
                        ("MT8", 11, 1, "yellowKey", 1_000_000.0),
                    )
                    for floor_id, x, y, expected, bonus in red_key_resources:
                        if tile_id(sim, state, floor_id, x, y) != expected:
                            score += bonus
                    if tile_id(sim, state, "MT8", 10, 2) != "redKey":
                        score += 4_000_000.0
    else:
        if hp < 300:
            score -= (300.0 - hp) * 5_000.0

    score += min(hp, 2500) * 250.0
    score += yk * 80_000.0 + bk * 150_000.0 + rk * 300_000.0
    if has_sword:
        score += 800_000.0
    if has_shield:
        score += 1_600_000.0
    score += float(state.money) * 800.0
    score += float(state.atk) * (18_000.0 if has_shield else 8_000.0)
    score += float(state.defense) * (18_000.0 if has_shield else 7_000.0)
    score += max(-5000, min(5000, margin)) * (60.0 if has_red_key else 8.0)
    score -= float(state.steps) * 15.0
    return score


def score_state_with_scheme(sim: MotaSimulator, state, target_stage: str, scheme: str) -> float:
    if scheme == "strict_stage":
        return strict_stage_score_state(sim, state, target_stage)
    return scalar_score_state(sim, state, target_stage)


def sample_child_action(result, rng: random.Random, temperature: float) -> dict[str, Any] | None:
    if not result.child_stats:
        return result.action
    if temperature <= 1e-8:
        return result.child_stats[0]["action"]
    weights = []
    for child in result.child_stats:
        score = float(child.get("visit_count", 0)) + 0.25 * float(child.get("value", 0.0))
        weights.append(max(1e-6, math.exp(score / max(1e-6, temperature))))
    total = sum(weights)
    pick = rng.random() * total
    acc = 0.0
    for child, weight in zip(result.child_stats, weights):
        acc += weight
        if acc >= pick:
            return child["action"]
    return result.child_stats[0]["action"]


def route_label_guard(path: str, protocol: str) -> None:
    lowered = str(path).lower()
    if protocol == "pure_search_rl" and any(token in lowered for token in ("hp403", "manual", "expert")):
        raise SystemExit(f"pure_search_rl rejects expert/manual route path: {path}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    weights_payload = load_weights_payload(args.reward_weights_file)
    if args.protocol in {"pure_search_rl", "warmstart_policy"} and weights_payload is not None:
        raise SystemExit(f"{args.protocol} forbids --reward-weights-file")
    if args.protocol == "pure_search_rl" and args.policy_checkpoint:
        route_label_guard(args.policy_checkpoint, args.protocol)
    if args.protocol == "potential_shaping" and weights_payload is None:
        raise SystemExit("potential_shaping requires --reward-weights-file")

    sim = MotaSimulator(
        load_game_data(args.data),
        SimulatorConfig(
            allow_negative_hp=args.allow_negative_hp,
            min_hp=args.relaxed_min_hp,
            stop_on_boss=not args.continue_after_boss,
        ),
    )
    start = sim.reset()
    if args.start_route:
        route_label_guard(args.start_route, args.protocol)
        start = apply_route_prefix(sim, start, args.start_route, args.start_route_max_actions)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf8")
    root = SearchNode(start.clone(), depth=0)
    archive = ArchiveMCTS(
        top_k=args.archive_top_k,
        visit_penalty_scale=args.visit_penalty_scale,
    )
    root_score = score_state_with_scheme(sim, start, args.target_stage, args.score_scheme)
    archive.add(sim, start, current_stage_name(sim, start), root, root_score)
    best = root
    best_score = root_score
    trace_rows: list[dict[str, Any]] = []
    policy_value_fn = make_policy_value_fn(args)
    leaf_value_fn = (
        LearnedRewardValueFn(
            weights_payload,
            scale=args.reward_value_scale,
            stage_mode=args.reward_stage_mode,
            gamma=args.reward_gamma,
        )
        if weights_payload is not None
        else None
    )
    edge_reward_fn = make_potential_edge_reward_fn(
        weights_payload,
        gamma=args.reward_gamma,
        stage_mode=args.reward_stage_mode,
    )

    for iteration in range(1, max(1, int(args.iterations)) + 1):
        entry = archive.choose(rng, temperature=args.archive_temperature, limit=args.archive_sample_limit)
        state = entry.node.state.clone()
        parent = entry.node
        selected_entry_score = float(entry.score)
        selected_archive_cells = len(archive.buckets)
        rollout_best_score = selected_entry_score
        kept_child = False
        failed_rollout = False
        for rollout_step in range(max(1, int(args.rollout_steps))):
            if state.dead or stage_complete(sim, state, args.target_stage):
                break
            mcts = AlphaMCTS(
                sim,
                policy_value_fn=policy_value_fn,
                leaf_value_fn=leaf_value_fn,
                edge_reward_fn=edge_reward_fn,
                config=AlphaMCTSConfig(
                    target_stage=args.target_stage,
                    num_simulations=args.simulations,
                    c_puct=args.c_puct,
                    max_depth=args.max_depth,
                    seed=args.seed + iteration * 1009 + rollout_step,
                    use_stage_action_filter=args.use_stage_action_filter,
                    edge_reward_scheme=args.edge_reward_scheme,
                    edge_reward_scale=args.edge_reward_scale,
                    edge_reward_clip=args.edge_reward_clip,
                    root_dirichlet_alpha=args.root_dirichlet_alpha,
                    root_exploration_fraction=args.root_exploration_fraction,
                    final_action_visit_weight=args.final_action_visit_weight,
                    final_action_value_weight=args.final_action_value_weight,
                    final_action_prior_weight=args.final_action_prior_weight,
                    hp_aware_success_value=args.hp_aware_success_value,
                    hp_success_scale=args.hp_success_scale,
                ),
            )
            result = mcts.search(state)
            action = (
                choose_valid_child(sim, state, result, rng, args.child_temperature, args.allow_negative_hp)
                if args.validate_selected_action
                else result.action
            )
            if action is None:
                entry.score -= float(args.failed_rollout_penalty)
                failed_rollout = True
                break
            before = state.clone()
            transition = sim.apply_macro_action(state, action)
            if not transition.ok:
                entry.score -= float(args.failed_rollout_penalty)
                failed_rollout = True
                break
            if not args.allow_negative_hp and state.hp <= 0:
                entry.score -= float(args.failed_rollout_penalty)
                failed_rollout = True
                break
            child_score = score_state_with_scheme(sim, state, args.target_stage, args.score_scheme)
            rollout_best_score = max(rollout_best_score, child_score)
            step = {
                "iteration": iteration,
                "rollout_step": rollout_step,
                "stage": current_stage_name(sim, before),
                "target_stage": args.target_stage,
                "action": action,
                "mcts": {
                    "root_value": result.root_value,
                    "visit_count": result.visit_count,
                    "top_children": result.child_stats[:8],
                },
                "before": state_summary(before),
                "after": state_summary(state),
                "ok": transition.ok,
                "message": transition.message,
                "archive_score": child_score,
            }
            child = SearchNode(state.clone(), parent=parent, step=step, depth=parent.depth + 1)
            kept_child = archive.add(sim, state, current_stage_name(sim, state), child, child_score) or kept_child
            parent = child
            if child_score > best_score or stage_complete(sim, state, args.target_stage):
                best = child
                best_score = child_score
            if len(trace_rows) < args.trace_limit:
                trace_rows.append(
                    {
                        "iteration": iteration,
                        "rollout_step": rollout_step,
                        "score": child_score,
                        "stage": current_stage_name(sim, state),
                        "state": state_summary(state),
                        "action": action.get("label", ""),
                    }
                )
            if stage_complete(sim, state, args.target_stage):
                best = child
                break
        if (
            not failed_rollout
            and not stage_complete(sim, best.state, args.target_stage)
            and len(archive.buckets) <= selected_archive_cells
            and not kept_child
            and rollout_best_score <= selected_entry_score + float(args.stale_rollout_min_score_gain)
        ):
            entry.score -= float(args.stale_rollout_penalty)
        if stage_complete(sim, best.state, args.target_stage):
            break
        if args.progress_interval > 0 and iteration % args.progress_interval == 0:
            route_so_far = reconstruct_route(best)
            write_route_jsonl(route_so_far, out_dir / "best_route_so_far.jsonl")
            route_views: dict[str, dict[str, Any]] = {}
            archive_entries = archive.entries()
            def balanced_entry_key(entry: ArchiveEntry) -> tuple[float, int, int, int, int]:
                state = entry.node.state
                stat_sum = int(state.atk) + int(state.defense)
                yellow = int(state.items.get("yellowKey", 0))
                blue = int(state.items.get("blueKey", 0))
                money = int(state.money)
                hp = int(state.hp)
                if args.target_stage in {"lower_gems", "pre_mt10_buffer"}:
                    remaining_lower = remaining_lower_attack_defense_gems(sim, state)
                    total_lower = total_lower_attack_defense_gems(sim)
                    taken_lower = max(0, int(total_lower) - int(remaining_lower))
                    pre_buffer_bonus = 0.0
                    if args.target_stage == "pre_mt10_buffer":
                        needed_blue = 0 if tile_id(sim, state, "MT9", 3, 11) != "blueDoor" else 1
                        pre_buffer_bonus = (
                            min(max(hp, 0), 420) * 22_000.0
                            + min(yellow, 4) * 1_150_000.0
                            + min(blue, max(1, needed_blue)) * 1_350_000.0
                            - max(0, 240 - hp) * 220_000.0
                            - max(0, 2 - yellow) * 7_500_000.0
                            - max(0, needed_blue - blue) * 12_000_000.0
                            - (18_000_000.0 if hp < 180 else 0.0)
                        )
                    score = (
                        taken_lower * 1_500_000.0
                        + min(stat_sum, 51) * 180_000.0
                        + min(max(hp, 0), 500) * 11_000.0
                        + min(yellow, 5) * 950_000.0
                        + min(blue, 3) * 350_000.0
                        + min(money, 140) * 1_500.0
                        - max(0, 180 - hp) * 95_000.0
                        - (6_000_000.0 if yellow <= 0 else 0.0)
                        - (1_500_000.0 if yellow == 1 else 0.0)
                        + pre_buffer_bonus
                        - int(entry.node.depth) * 90.0
                    )
                else:
                    score = (
                        min(stat_sum, 49) * 100_000.0
                        + min(max(hp, 0), 360) * 120.0
                        + min(yellow, 5) * 8_000.0
                        + min(blue, 3) * 12_000.0
                        + min(money, 120) * 80.0
                        - int(entry.node.depth) * 12.0
                    )
                return (score, stat_sum, hp, yellow, blue)

            view_specs = {
                "best_hp": lambda entry: (
                    int(entry.node.state.hp),
                    int(entry.node.state.atk) + int(entry.node.state.defense),
                    floor_number(entry.node.state),
                    int(entry.score),
                ),
                "best_floor": lambda entry: (
                    floor_number(entry.node.state),
                    int(entry.node.state.hp),
                    int(entry.node.state.atk) + int(entry.node.state.defense),
                    int(entry.score),
                ),
                "best_stats": lambda entry: (
                    int(entry.node.state.atk) + int(entry.node.state.defense),
                    int(entry.node.state.hp),
                    floor_number(entry.node.state),
                    int(entry.score),
                ),
                "best_balanced": balanced_entry_key,
            }
            for view_name, key_fn in view_specs.items():
                if not archive_entries:
                    continue
                entry = max(archive_entries, key=key_fn)
                view_route = reconstruct_route(entry.node)
                view_path = out_dir / f"{view_name}_route_so_far.jsonl"
                write_route_jsonl(view_route, view_path)
                route_views[view_name] = {
                    "route": str(view_path),
                    "route_length": len(view_route),
                    "score": entry.score,
                    "visits": entry.visits,
                    "depth": entry.node.depth,
                    "state": state_summary(entry.node.state),
                }
            progress = {
                "iteration": iteration,
                "archive_cells": len(archive.buckets),
                "best_score": best_score,
                "target_success": stage_complete(sim, best.state, args.target_stage),
                "boss_success": bool(best.state.flags.get("10f战胜骷髅队长")),
                "boss_margin": boss_route_margin(sim, best.state),
                "route_length": len(route_so_far),
                "best_state": state_summary(best.state),
                "route_views": route_views,
            }
            with (out_dir / "progress.jsonl").open("a", encoding="utf8") as handle:
                handle.write(json.dumps(progress, ensure_ascii=False) + "\n")
            (out_dir / "progress_latest.json").write_text(
                json.dumps(progress, ensure_ascii=False, indent=2),
                encoding="utf8",
            )
            (out_dir / "archive_snapshot_latest.json").write_text(
                json.dumps(archive.snapshot(args.progress_archive_snapshot_limit), ensure_ascii=False, indent=2),
                encoding="utf8",
            )
            (out_dir / "route_views_latest.json").write_text(
                json.dumps(route_views, ensure_ascii=False, indent=2),
                encoding="utf8",
            )

    route = reconstruct_route(best)
    route_path = out_dir / "best_route.jsonl"
    write_route_jsonl(route, route_path)
    (out_dir / "archive_cells.json").write_text(
        json.dumps(archive.snapshot(args.archive_snapshot_limit), ensure_ascii=False, indent=2),
        encoding="utf8",
    )
    with (out_dir / "search_trace.jsonl").open("w", encoding="utf8") as handle:
        for row in trace_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "protocol": args.protocol,
        "target_stage": args.target_stage,
        "target_success": stage_complete(sim, best.state, args.target_stage),
        "boss_success": bool(best.state.flags.get("10f战胜骷髅队长")),
        "boss_margin": boss_route_margin(sim, best.state),
        "iterations": args.iterations,
        "archive_cells": len(archive.buckets),
        "route_length": len(route),
        "best_score": best_score,
        "best_state": state_summary(best.state),
        "route": str(route_path),
        "score_scheme": args.score_scheme,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def choose_valid_child(
    sim: MotaSimulator,
    state,
    result,
    rng: random.Random,
    temperature: float,
    allow_negative_hp: bool,
) -> dict[str, Any] | None:
    children = list(result.child_stats)
    if result.action is not None and not children:
        children = [{"action": result.action, "visit_count": 1, "value": result.root_value}]
    if temperature > 1e-8 and len(children) > 1:
        rng.shuffle(children)
    children.sort(key=lambda row: (float(row.get("visit_count", 0)), float(row.get("value", 0.0))), reverse=True)
    if temperature > 1e-8 and len(children) > 1:
        preferred = sample_child_action(result, rng, temperature)
        if preferred is not None:
            children.insert(0, {"action": preferred, "visit_count": math.inf, "value": math.inf})
    seen: set[tuple[Any, ...]] = set()
    for child in children:
        action = child.get("action")
        if not action:
            continue
        key = (
            action.get("kind"),
            tuple(action.get("target", [])),
            action.get("floor"),
            tuple(action.get("loc", [])),
            action.get("shop"),
            action.get("label"),
        )
        if key in seen:
            continue
        seen.add(key)
        candidate = state.clone()
        transition = sim.apply_macro_action(candidate, action)
        if not transition.ok:
            continue
        if not allow_negative_hp and candidate.hp <= 0:
            continue
        return action
    return None


def apply_route_prefix(sim: MotaSimulator, state, route_path: str, max_actions: int) -> Any:
    from mota_rl.train_actor_critic import find_matching_action, load_route

    rows = load_route(route_path)
    for index, row in enumerate(rows):
        if max_actions > 0 and index >= max_actions:
            break
        actions = sim.macro_actions(state)
        action_index = find_matching_action(actions, row.get("action") or {})
        if action_index is None:
            raise SystemExit(f"cannot replay start route at row {index}: {(row.get('action') or {}).get('label')}")
        transition = sim.apply_macro_action(state, actions[action_index])
        if not transition.ok or state.dead or state.done:
            raise SystemExit(f"start route failed at row {index}: {transition.message}")
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="artifacts/data/mota_first10.json")
    parser.add_argument("--out-dir", default="artifacts/runs/archive_mcts")
    parser.add_argument(
        "--protocol",
        choices=("pure_search_rl", "warmstart_policy", "potential_shaping"),
        default="pure_search_rl",
    )
    parser.add_argument("--target-stage", choices=stage_names(), default="boss")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--rollout-steps", type=int, default=8)
    parser.add_argument("--archive-top-k", type=int, default=8)
    parser.add_argument("--archive-temperature", type=float, default=0.8)
    parser.add_argument("--archive-sample-limit", type=int, default=96)
    parser.add_argument("--archive-snapshot-limit", type=int, default=1000)
    parser.add_argument("--progress-archive-snapshot-limit", type=int, default=80)
    parser.add_argument("--score-scheme", choices=("scalar", "strict_stage"), default="scalar")
    parser.add_argument("--visit-penalty-scale", type=float, default=50.0)
    parser.add_argument("--failed-rollout-penalty", type=float, default=25_000.0)
    parser.add_argument("--stale-rollout-penalty", type=float, default=25_000.0)
    parser.add_argument("--stale-rollout-min-score-gain", type=float, default=100.0)
    parser.add_argument("--trace-limit", type=int, default=5000)
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--simulations", type=int, default=48)
    parser.add_argument("--max-depth", type=int, default=24)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--child-temperature", type=float, default=0.0)
    parser.add_argument("--validate-selected-action", action="store_true")
    parser.add_argument("--policy-checkpoint", default="")
    parser.add_argument("--policy-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--policy-d-model", type=int, default=128)
    parser.add_argument("--policy-heads", type=int, default=4)
    parser.add_argument("--policy-layers", type=int, default=4)
    parser.add_argument("--policy-temperature", type=float, default=1.0)
    parser.add_argument("--heuristic-prior-mix", type=float, default=0.0)
    parser.add_argument("--heuristic-temperature", type=float, default=0.8)
    parser.add_argument("--reward-weights-file", default="")
    parser.add_argument("--reward-stage-mode", choices=("target", "current"), default="target")
    parser.add_argument("--reward-gamma", type=float, default=0.99)
    parser.add_argument("--reward-value-scale", type=float, default=25_000.0)
    parser.add_argument("--edge-reward-scheme", choices=("none", "raw", "label_dense", "milestone", "resource_delta", "key_pressure", "potential", "dynamic_pbrs", "stage_pbrs", "stage_stat_pbrs", "learnable_stage_pbrs"), default="none")
    parser.add_argument("--edge-reward-scale", type=float, default=2000.0)
    parser.add_argument("--edge-reward-clip", type=float, default=1.0)
    parser.add_argument("--root-dirichlet-alpha", type=float, default=0.0)
    parser.add_argument("--root-exploration-fraction", type=float, default=0.0)
    parser.add_argument("--final-action-visit-weight", type=float, default=1.0)
    parser.add_argument("--final-action-value-weight", type=float, default=0.0)
    parser.add_argument("--final-action-prior-weight", type=float, default=0.0)
    parser.add_argument("--hp-aware-success-value", action="store_true")
    parser.add_argument("--hp-success-scale", type=float, default=1000.0)
    parser.add_argument("--allow-negative-hp", action="store_true")
    parser.add_argument("--relaxed-min-hp", type=int, default=-2000)
    parser.add_argument("--continue-after-boss", action="store_true")
    parser.add_argument("--use-stage-action-filter", action="store_true")
    parser.add_argument("--start-route", default="")
    parser.add_argument("--start-route-max-actions", type=int, default=0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
