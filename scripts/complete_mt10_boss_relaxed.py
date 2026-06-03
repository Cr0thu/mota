from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mota_env import MotaSimulator, SimulatorConfig, load_game_data
from mota_rl.train_actor_critic import find_matching_action, load_route
from mota_solver.search import state_summary, write_route_jsonl


TRAP_GUARDS = {
    ("MT10", 5, 4),
    ("MT10", 6, 4),
    ("MT10", 7, 4),
    ("MT10", 5, 5),
    ("MT10", 7, 5),
    ("MT10", 5, 6),
    ("MT10", 6, 6),
    ("MT10", 7, 6),
}
BOSS = ("MT10", 6, 1)
MT10_RESOURCES = {
    ("MT10", 2, 6),  # blue gem before trap path
    ("MT10", 10, 6),  # red gem after trap/right chamber
    ("MT10", 11, 11),  # blue potion
}


@dataclass
class Node:
    state: Any
    score: float
    parent: "Node | None" = None
    step: dict[str, Any] | None = None
    depth: int = 0


def replay_route(sim: MotaSimulator, route_path: str) -> tuple[Any, list[dict[str, Any]]]:
    state = sim.reset()
    replayed: list[dict[str, Any]] = []
    for index, row in enumerate(load_route(route_path)):
        actions = sim.macro_actions(state)
        match = find_matching_action(actions, row["action"])
        if match is None:
            raise SystemExit(f"Cannot replay start route at {index}: {row['action'].get('label', '')}")
        action = actions[match]
        before = state.clone()
        transition = sim.apply_macro_action(state, action)
        replayed.append(
            {
                "index": len(replayed),
                "action": action,
                "before": state_summary(before),
                "after": state_summary(state),
                "transition": transition.message,
                "reward": transition.reward,
                "source": row.get("source", "start_route"),
            }
        )
        if not transition.ok or state.dead:
            raise SystemExit(f"Start route failed at {index}: {transition.message}")
    return state, replayed


def route_from(node: Node) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cursor: Node | None = node
    while cursor is not None and cursor.step is not None:
        rows.append(cursor.step)
        cursor = cursor.parent
    rows.reverse()
    for index, row in enumerate(rows):
        row["index"] = index
    return rows


def target_tuple(action: dict[str, Any]) -> tuple[str, int, int] | None:
    raw = action.get("target")
    if not isinstance(raw, list) or len(raw) != 3:
        return None
    return str(raw[0]), int(raw[1]), int(raw[2])


def enemy_damage(sim: MotaSimulator, state: Any, action: dict[str, Any]) -> int:
    target = target_tuple(action)
    if target is None:
        return 0
    floor_id, x, y = target
    tile = sim.tile(state, x, y, floor_id)
    if not sim.is_enemy_tile(tile):
        return 0
    enemy_id = sim.block_id(tile)
    if not enemy_id:
        return 0
    info = sim.damage_info(state, enemy_id)
    return 100_000 if info is None else int(info["damage"])


def trap_guards_left(sim: MotaSimulator, state: Any) -> int:
    left = 0
    for floor_id, x, y in TRAP_GUARDS:
        if sim.block_id(sim.tile(state, x, y, floor_id)) in {"skeleton", "skeletonSoldier"}:
            left += 1
    return left


def mt10_resources_left(sim: MotaSimulator, state: Any) -> int:
    left = 0
    for floor_id, x, y in MT10_RESOURCES:
        block_id = sim.block_id(sim.tile(state, x, y, floor_id))
        if block_id in {"redGem", "blueGem", "redPotion", "bluePotion"}:
            left += 1
    return left


def compact_key(sim: MotaSimulator, state: Any) -> tuple[Any, ...]:
    trap_bits = []
    for _, x, y in sorted(TRAP_GUARDS):
        trap_bits.append(sim.block_id(sim.tile(state, x, y, "MT10")) or "0")
    resource_bits = []
    for _, x, y in sorted(MT10_RESOURCES):
        resource_bits.append(sim.block_id(sim.tile(state, x, y, "MT10")) or "0")
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
        tuple(trap_bits),
        tuple(resource_bits),
        bool(state.flags.get("10f机关")),
        bool(state.flags.get("10f战胜骷髅队长")),
        sim.block_id(sim.tile(state, 6, 3, "MT10")) or "0",
    )


def action_allowed(
    sim: MotaSimulator,
    state: Any,
    action: dict[str, Any],
    require_resources_before_boss: bool,
) -> bool:
    label = str(action.get("label", ""))
    target = target_tuple(action)
    if require_resources_before_boss and target == BOSS and mt10_resources_left(sim, state) > 0:
        return False
    if "downFloor" in label or "upFloor" in label:
        return False
    if target is not None and target[0] != "MT10":
        return False
    if "event MT10:6,5" in label:
        return False
    if label.startswith(("go ", "fight ", "open ")):
        return True
    return "specialDoor MT10:6,3" in label


def action_score(sim: MotaSimulator, before: Any, after: Any, action: dict[str, Any]) -> float:
    label = str(action.get("label", ""))
    target = target_tuple(action)
    score = 0.0
    score += (trap_guards_left(sim, before) - trap_guards_left(sim, after)) * 900.0
    score += (mt10_resources_left(sim, before) - mt10_resources_left(sim, after)) * 260.0
    score += (after.atk - before.atk) * 500.0
    score += (after.defense - before.defense) * 420.0
    score += max(0, after.hp - before.hp) * 0.4
    score += (after.hp - before.hp) * 0.05
    score += (after.money - before.money) * 8.0
    score -= len(action.get("path", [])) * 0.8
    score -= enemy_damage(sim, before, action) * 0.22

    if target == BOSS:
        score += 5000.0
    if target in TRAP_GUARDS:
        score += 350.0
    if target in MT10_RESOURCES:
        score += 600.0
    if "specialDoor MT10:6,3" in label:
        score += 700.0
    if "skeletonCaptain" in label:
        score += 4000.0
    if after.flags.get("10f战胜骷髅队长"):
        score += 100_000.0
    return score


def node_value(sim: MotaSimulator, node: Node) -> float:
    state = node.state
    if state.flags.get("10f战胜骷髅队长"):
        return 1_000_000.0 + state.hp
    value = node.score
    value -= trap_guards_left(sim, state) * 1200.0
    value -= mt10_resources_left(sim, state) * 180.0
    value += state.atk * 80.0 + state.defense * 65.0 + min(state.hp, 1200) * 0.10
    if sim.block_id(sim.tile(state, 6, 3, "MT10")) == "0":
        value += 900.0
    return value


def select_beam(sim: MotaSimulator, candidates: list[Node], beam_width: int, bucket_limit: int) -> list[Node]:
    ordered = sorted(candidates, key=lambda node: node_value(sim, node), reverse=True)
    selected: list[Node] = []
    buckets: Counter[tuple[Any, ...]] = Counter()
    for node in ordered:
        state = node.state
        bucket = (
            state.x // 2,
            state.y // 2,
            state.atk,
            state.defense,
            trap_guards_left(sim, state),
            mt10_resources_left(sim, state),
            state.items.get("yellowKey", 0),
            state.items.get("blueKey", 0),
        )
        if buckets[bucket] >= bucket_limit:
            continue
        buckets[bucket] += 1
        selected.append(node)
        if len(selected) >= beam_width:
            break
    return selected or ordered[:beam_width]


def main() -> None:
    parser = argparse.ArgumentParser(description="Specialized relaxed completion from MT10 trap to captain.")
    parser.add_argument("--data", default="artifacts/data/mota_first10.json")
    parser.add_argument("--start-route", required=True)
    parser.add_argument("--route-out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--trace-out", default="")
    parser.add_argument("--beam-width", type=int, default=96)
    parser.add_argument("--action-top-k", type=int, default=16)
    parser.add_argument("--max-depth", type=int, default=40)
    parser.add_argument("--bucket-limit", type=int, default=4)
    parser.add_argument("--relaxed-min-hp", type=int, default=-25000)
    parser.add_argument(
        "--require-mt10-resources-before-boss",
        action="store_true",
        help="Do not fight the captain while known MT10 gem/potion resources remain.",
    )
    args = parser.parse_args()

    sim = MotaSimulator(
        load_game_data(args.data),
        SimulatorConfig(allow_negative_hp=True, min_hp=args.relaxed_min_hp),
    )
    start, prefix = replay_route(sim, args.start_route)
    if not start.flags.get("10f机关"):
        raise SystemExit("Start route must already trigger MT10 trap.")

    root = Node(start, 0.0)
    beam = [root]
    best = root
    seen = {compact_key(sim, start)}
    expanded = 0
    generated = 0
    trace_rows: list[dict[str, Any]] = []

    for depth in range(1, args.max_depth + 1):
        candidates: list[Node] = []
        for node in beam:
            if node.state.done or node.state.flags.get("10f战胜骷髅队长"):
                best = node if node_value(sim, node) > node_value(sim, best) else best
                continue
            scored: list[tuple[float, dict[str, Any]]] = []
            for action in sim.macro_actions(node.state):
                if not action_allowed(
                    sim,
                    node.state,
                    action,
                    require_resources_before_boss=args.require_mt10_resources_before_boss,
                ):
                    continue
                child = node.state.clone()
                transition = sim.apply_macro_action(child, action)
                if not transition.ok or child.dead:
                    continue
                scored.append((action_score(sim, node.state, child, action), action))
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
                    "index": len(prefix) + node.depth,
                    "action": action,
                    "before": state_summary(before),
                    "after": state_summary(child),
                    "transition": transition.message,
                    "reward": transition.reward,
                    "source": "mt10_boss_relaxed_completion",
                    "completion_score": action_score(sim, before, child, action),
                }
                candidate = Node(
                    state=child,
                    score=node.score + float(step["completion_score"]),
                    parent=node,
                    step=step,
                    depth=node.depth + 1,
                )
                generated += 1
                candidates.append(candidate)
                if node_value(sim, candidate) > node_value(sim, best):
                    best = candidate
        if not candidates:
            break
        beam = select_beam(sim, candidates, args.beam_width, args.bucket_limit)
        if node_value(sim, beam[0]) > node_value(sim, best):
            best = beam[0]
        row = {
            "depth": depth,
            "beam_size": len(beam),
            "best": state_summary(best.state),
            "trap_guards_left": trap_guards_left(sim, best.state),
            "mt10_resources_left": mt10_resources_left(sim, best.state),
            "best_score": best.score,
            "last_action": best.step["action"].get("label", "") if best.step else "",
            "frontier_top": [
                {
                    "value": node_value(sim, node),
                    "score": node.score,
                    "state": state_summary(node.state),
                    "trap_guards_left": trap_guards_left(sim, node.state),
                    "mt10_resources_left": mt10_resources_left(sim, node.state),
                    "last_action": node.step["action"].get("label", "") if node.step else "",
                }
                for node in beam[:8]
            ],
        }
        trace_rows.append(row)
        if best.state.flags.get("10f战胜骷髅队长"):
            break

    continuation = route_from(best)
    full_route = prefix + continuation
    for index, row in enumerate(full_route):
        row["index"] = index
    write_route_jsonl(full_route, Path(args.route_out))
    summary = {
        "solved": bool(best.state.flags.get("10f战胜骷髅队长")),
        "strict_success": bool(best.state.flags.get("10f战胜骷髅队长") and best.state.hp > 0),
        "start_route": args.start_route,
        "route_out": args.route_out,
        "prefix_steps": len(prefix),
        "continuation_steps": len(continuation),
        "expanded": expanded,
        "generated": generated,
        "final": state_summary(best.state),
        "trap_guards_left": trap_guards_left(sim, best.state),
        "mt10_resources_left": mt10_resources_left(sim, best.state),
        "best_by_depth": trace_rows,
    }
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8")
    if args.trace_out:
        trace_path = Path(args.trace_out)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("w", encoding="utf8") as handle:
            for row in trace_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({**summary, "best_by_depth": trace_rows[-5:]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
