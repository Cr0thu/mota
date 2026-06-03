from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mota_env import MotaSimulator, load_game_data
from mota_env.rewards import (
    boss_ready,
    boss_route_margin,
    boss_route_required_damage,
    stage_potential,
)
from mota_rl.train_actor_critic import find_matching_action, load_route
from mota_solver.search import state_summary, write_route_jsonl


RESOURCE_TOKENS = (
    "Potion",
    "redGem",
    "blueGem",
    "yellowKey",
    "blueKey",
    "redKey",
    "sword",
    "shield",
)
STAIR_TOKENS = ("upFloor", "downFloor")
BLOCKER_TOKENS = (
    "yellowDoor",
    "blueDoor",
    "redDoor",
    "specialDoor",
    "fakeWall",
    "skeleton",
    "skeletonSoldier",
    "bluePriest",
    "bat",
    "redSlime",
    "greenSlime",
)


@dataclass
class ProbeNode:
    state: Any
    score: float
    route: list[dict[str, Any]]


def replay_route(sim: MotaSimulator, route_path: str) -> tuple[Any, list[dict[str, Any]]]:
    state = sim.reset()
    rows = load_route(route_path)
    replayed: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        actions = sim.macro_actions(state)
        match = find_matching_action(actions, row["action"])
        if match is None:
            raise SystemExit(f"Cannot replay start route at {index}: {row['action'].get('label', '')}")
        action = actions[match]
        before = state.clone()
        transition = sim.apply_macro_action(state, action)
        replayed.append(
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
        if not transition.ok or state.dead:
            raise SystemExit(f"Start route failed at {index}: {transition.message}")
    return state, replayed


def compact_key(sim: MotaSimulator, state: Any) -> tuple[Any, ...]:
    """Cheap approximate de-dup key for exploratory continuation search.

    This intentionally avoids hashing the full 10-floor map. It is not a
    proof-search key; it is a fast route-probing key used to keep the beam from
    wasting time on exact local repeats.
    """

    trap_alive = 0
    if state.flags.get("10f机关"):
        for x, y in ((5, 4), (6, 4), (7, 4), (5, 5), (7, 5), (5, 6), (6, 6), (7, 6)):
            if sim.block_id(sim.tile(state, x, y, "MT10")) in {"skeleton", "skeletonSoldier"}:
                trap_alive += 1
    return (
        state.floor_id,
        state.x,
        state.y,
        state.hp,
        state.atk,
        state.defense,
        state.money,
        state.items.get("yellowKey", 0),
        state.items.get("blueKey", 0),
        state.items.get("redKey", 0),
        bool(state.flags.get("10f机关")),
        bool(state.flags.get("10f战胜骷髅队长")),
        trap_alive,
        boss_route_required_damage(sim, state),
    )


def enemy_damage(sim: MotaSimulator, state: Any, action: dict[str, Any]) -> int:
    target = action.get("target") or []
    if len(target) != 3:
        return 0
    floor_id, x, y = target
    enemy_id = sim.block_id(sim.tile(state, int(x), int(y), str(floor_id)))
    if not enemy_id:
        return 0
    info = sim.damage_info(state, enemy_id)
    return 10_000 if info is None else int(info["damage"])


def action_allowed(sim: MotaSimulator, state: Any, action: dict[str, Any]) -> bool:
    label = action.get("label", "")
    margin = boss_route_margin(sim, state)
    if margin <= 0 and any(token in label for token in ("redDoor MT10:6,9", "event MT10:6,5", "skeletonCaptain")):
        return False
    if label.startswith("fight"):
        damage = enemy_damage(sim, state, action)
        if any(token in label for token in ("skeletonCaptain", "yellowGuard")):
            return True
        if any(token in label for token in RESOURCE_TOKENS):
            return True
        # In boss-prep search, expensive filler fights are usually route debt.
        if damage > 180 and not any(token in label for token in ("skeletonSoldier MT4:10,3", "skeleton MT4:9,5")):
            return False
    return any(token in label for token in RESOURCE_TOKENS + STAIR_TOKENS + BLOCKER_TOKENS)


def action_score(sim: MotaSimulator, state: Any, child: Any, action: dict[str, Any]) -> float:
    label = action.get("label", "")
    before_margin = boss_route_margin(sim, state)
    after_margin = boss_route_margin(sim, child)
    before_yellow = state.items.get("yellowKey", 0)
    after_yellow = child.items.get("yellowKey", 0)
    before_blue = state.items.get("blueKey", 0)
    after_blue = child.items.get("blueKey", 0)
    score = 0.0
    score += (after_margin - before_margin) * 5.0
    score += (stage_potential(sim, child, stage="boss_ready") - stage_potential(sim, state, stage="boss_ready")) * 0.2
    score += (child.hp - state.hp) * 0.25
    score += after_yellow * 45.0 + (after_yellow - before_yellow) * 130.0
    score += after_blue * 85.0 + (after_blue - before_blue) * 180.0
    score += child.items.get("redKey", 0) * 40.0
    if "bluePotion" in label:
        score += 360.0
    elif "redPotion" in label:
        score += 120.0
    if "redGem" in label or "blueGem" in label:
        score += 110.0
    if "yellowKey" in label:
        score += 160.0
    if "blueKey" in label:
        score += 220.0
    if "yellowDoor" in label:
        score -= 180.0 if before_yellow <= 2 else 80.0
    if "blueDoor" in label:
        score -= 260.0 if before_blue <= 1 else 120.0
    if label.startswith("fight"):
        score -= enemy_damage(sim, state, action) * 0.35
    if "upFloor" in label and int(state.floor_id[2:]) < 10 and after_margin > -120:
        score += 25.0
    if "downFloor" in label and after_margin < -80:
        score += 12.0
    if after_margin > 0:
        score += 600.0
    return score


def node_value(sim: MotaSimulator, node: ProbeNode) -> float:
    state = node.state
    if state.flags.get("10f战胜骷髅队长"):
        return 1_000_000.0 + state.hp
    margin = boss_route_margin(sim, state)
    return (
        node.score
        + margin * 20.0
        + min(state.hp, 1400) * 0.5
        + state.atk * 20.0
        + state.defense * 20.0
        + state.items.get("yellowKey", 0) * 140.0
        + state.items.get("blueKey", 0) * 180.0
        + state.items.get("redKey", 0) * 240.0
        + (5000.0 if boss_ready(sim, state) else 0.0)
    )


def select_beam(sim: MotaSimulator, candidates: list[ProbeNode], beam_width: int) -> list[ProbeNode]:
    ordered = sorted(candidates, key=lambda node: node_value(sim, node), reverse=True)
    selected: list[ProbeNode] = []
    buckets: Counter[tuple[Any, ...]] = Counter()
    for node in ordered:
        state = node.state
        bucket = (
            state.floor_id,
            state.x // 2,
            state.y // 2,
            state.hp // 100,
            state.items.get("yellowKey", 0),
            state.items.get("redKey", 0),
            bool(state.flags.get("10f机关")),
        )
        if buckets[bucket] >= 3:
            continue
        selected.append(node)
        buckets[bucket] += 1
        if len(selected) >= beam_width:
            break
    return selected or ordered[:beam_width]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="artifacts/data/mota_first10.json")
    parser.add_argument("--start-route", required=True)
    parser.add_argument("--route-out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--beam-width", type=int, default=48)
    parser.add_argument("--action-top-k", type=int, default=10)
    parser.add_argument("--max-depth", type=int, default=40)
    args = parser.parse_args()

    sim = MotaSimulator(load_game_data(args.data))
    start, prefix = replay_route(sim, args.start_route)
    root = ProbeNode(start, 0.0, prefix)
    beam = [root]
    best = root
    seen = {compact_key(sim, start)}
    expanded = 0
    generated = 0
    best_by_depth: list[dict[str, Any]] = []

    for depth in range(1, args.max_depth + 1):
        candidates: list[ProbeNode] = []
        for node in beam:
            actions = [action for action in sim.macro_actions(node.state) if action_allowed(sim, node.state, action)]
            scored: list[tuple[float, dict[str, Any]]] = []
            for action in actions:
                child = node.state.clone()
                transition = sim.apply_macro_action(child, action)
                if not transition.ok or child.dead:
                    continue
                score = action_score(sim, node.state, child, action)
                scored.append((score, action))
            scored.sort(key=lambda item: item[0], reverse=True)
            expanded += 1
            for _, action in scored[: args.action_top_k]:
                child = node.state.clone()
                before = child.clone()
                transition = sim.apply_macro_action(child, action)
                if not transition.ok or child.dead:
                    continue
                key = compact_key(sim, child)
                if key in seen:
                    continue
                seen.add(key)
                step = {
                    "index": len(node.route),
                    "action": action,
                    "before": state_summary(before),
                    "after": state_summary(child),
                    "transition": transition.message,
                    "reward": transition.reward,
                    "probe_score": action_score(sim, before, child, action),
                }
                candidate = ProbeNode(
                    child,
                    node.score + step["probe_score"],
                    node.route + [step],
                )
                generated += 1
                candidates.append(candidate)
                if node_value(sim, candidate) > node_value(sim, best):
                    best = candidate
        if not candidates:
            break
        beam = select_beam(sim, candidates, args.beam_width)
        best_by_depth.append(
            {
                "depth": depth,
                "best": state_summary(best.state),
                "boss_required_damage": boss_route_required_damage(sim, best.state),
                "boss_route_margin": boss_route_margin(sim, best.state),
                "boss_ready": boss_ready(sim, best.state),
                "last_action": best.route[-1]["action"].get("label", "") if best.route else "",
            }
        )
        if boss_ready(sim, best.state):
            break

    route_out = Path(args.route_out)
    summary_out = Path(args.summary_out)
    write_route_jsonl(best.route, route_out)
    summary = {
        "route_out": str(route_out),
        "start_route": args.start_route,
        "expanded": expanded,
        "generated": generated,
        "route_len": len(best.route),
        "continuation_len": len(best.route) - len(prefix),
        "boss_ready": boss_ready(sim, best.state),
        "boss_required_damage": boss_route_required_damage(sim, best.state),
        "boss_route_margin": boss_route_margin(sim, best.state),
        "final": state_summary(best.state),
        "best_by_depth": best_by_depth,
    }
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps({**summary, "best_by_depth": best_by_depth[-5:]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
