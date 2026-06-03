from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

from mota_env import MotaSimulator, SimulatorConfig, load_game_data
from mota_solver.search import state_summary, write_route_jsonl


STAIR_WORDS = ("upFloor", "downFloor")


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]


def is_stair_step(row: dict[str, Any]) -> bool:
    label = str(row.get("action", {}).get("label", ""))
    return any(word in label for word in STAIR_WORDS)


def objective_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if not is_stair_step(row)]


def floor_index(floor_id: str) -> int:
    return int(floor_id[2:])


def action_after(sim: MotaSimulator, state, action: dict[str, Any]):
    clone = state.clone()
    transition = sim.apply_macro_action(clone, action)
    return clone if transition.ok and not clone.dead else None


def choose_stair_action(sim: MotaSimulator, state, target_floor: str) -> dict[str, Any] | None:
    target_idx = floor_index(target_floor)
    current_distance = abs(floor_index(state.floor_id) - target_idx)
    best: tuple[int, int, dict[str, Any]] | None = None
    for action in sim.macro_actions(state):
        label = str(action.get("label", ""))
        if not any(word in label for word in STAIR_WORDS):
            continue
        after = action_after(sim, state, action)
        if after is None:
            continue
        distance = abs(floor_index(after.floor_id) - target_idx)
        if distance >= current_distance:
            continue
        candidate = (distance, len(action.get("path", [])), action)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    return None if best is None else best[2]


def find_target_action(sim: MotaSimulator, state, target: tuple[str, int, int]) -> dict[str, Any] | None:
    for action in sim.macro_actions(state):
        action_target = action.get("target")
        if isinstance(action_target, list) and tuple(action_target) == target:
            return action
    return None


def append_action(sim: MotaSimulator, state, action: dict[str, Any], route: list[dict[str, Any]], stage: str) -> bool:
    before = state_summary(state)
    transition = sim.apply_macro_action(state, action)
    if not transition.ok or state.dead:
        return False
    route.append(
        {
            "index": len(route),
            "action": action,
            "stage": stage,
            "before": before,
            "after": state_summary(state),
            "reward": transition.reward,
        }
    )
    return True


def boss_kill_hp_from_route(route: list[dict[str, Any]]) -> int | None:
    for row in route:
        label = str(row.get("action", {}).get("label", ""))
        if "skeletonCaptain MT10:6,1" in label:
            return int(row.get("after", {}).get("hp", -10**9))
    return None


def replay_objectives(
    sim: MotaSimulator,
    objectives: list[dict[str, Any]],
    stage: str,
    hp_objective: str,
    max_auto_stairs: int = 32,
) -> tuple[bool, int, list[dict[str, Any]], dict[str, Any]]:
    state = sim.reset()
    route: list[dict[str, Any]] = []
    for objective in objectives:
        raw_target = objective["action"]["target"]
        target = (str(raw_target[0]), int(raw_target[1]), int(raw_target[2]))
        stair_count = 0
        while state.floor_id != target[0]:
            stair = choose_stair_action(sim, state, target[0])
            if stair is None or stair_count >= max_auto_stairs:
                return False, -10**9, route, state_summary(state)
            if not append_action(sim, state, stair, route, stage):
                return False, -10**9, route, state_summary(state)
            stair_count += 1
        action = find_target_action(sim, state, target)
        if action is None:
            return False, -10**9, route, state_summary(state)
        if not append_action(sim, state, action, route, stage):
            return False, -10**9, route, state_summary(state)
    solved = bool(state.flags.get("10f战胜骷髅队长"))
    if hp_objective == "boss_kill":
        hp = boss_kill_hp_from_route(route)
        return solved, (-10**9 if hp is None else hp), route, state_summary(state)
    return solved, state.hp, route, state_summary(state)


def mutate_objectives(
    objectives: list[dict[str, Any]],
    rng: random.Random,
    max_segment: int,
    bias: bool,
) -> list[dict[str, Any]]:
    candidate = list(objectives)
    n = len(candidate)
    if n < 3:
        return candidate
    op = rng.choices(
        ["move", "swap", "late_fight", "early_resource", "shuffle_window"],
        weights=[5, 2, 3 if bias else 1, 3 if bias else 1, 1],
        k=1,
    )[0]
    if op == "move":
        length = rng.randint(1, min(max_segment, n - 1))
        i = rng.randrange(0, n - length)
        segment = candidate[i : i + length]
        del candidate[i : i + length]
        j = rng.randrange(0, len(candidate) + 1)
        candidate[j:j] = segment
    elif op == "swap":
        length_a = rng.randint(1, min(max_segment, n // 2))
        length_b = rng.randint(1, min(max_segment, n // 2))
        i = rng.randrange(0, n - length_a)
        j = rng.randrange(0, n - length_b)
        if i > j:
            i, j = j, i
            length_a, length_b = length_b, length_a
        if i + length_a <= j:
            a = candidate[i : i + length_a]
            b = candidate[j : j + length_b]
            candidate[i : i + length_a] = b
            shift = len(b) - len(a)
            j += shift
            candidate[j : j + length_b] = a
    elif op == "late_fight":
        fight_indices = [
            idx
            for idx, row in enumerate(candidate)
            if str(row["action"].get("label", "")).startswith("fight")
        ]
        if fight_indices:
            i = rng.choice(fight_indices)
            row = candidate.pop(i)
            j = rng.randrange(i, len(candidate) + 1)
            candidate[j:j] = [row]
    elif op == "early_resource":
        resource_words = ("Gem", "sword", "shield", "Key")
        resource_indices = [
            idx
            for idx, row in enumerate(candidate)
            if any(word in str(row["action"].get("label", "")) for word in resource_words)
        ]
        if resource_indices:
            i = rng.choice(resource_indices)
            row = candidate.pop(i)
            j = rng.randrange(0, i + 1)
            candidate[j:j] = [row]
    else:
        width = rng.randint(3, min(max(3, max_segment * 2), n))
        i = rng.randrange(0, n - width + 1)
        window = candidate[i : i + width]
        rng.shuffle(window)
        candidate[i : i + width] = window
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="artifacts/data/mota_first10.json")
    parser.add_argument("--base-route", required=True)
    parser.add_argument("--out-dir", default="artifacts/manual_exploration_20260524")
    parser.add_argument("--prefix", default="manual_success_no_shop_true_10f_trap_local")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--max-segment", type=int, default=8)
    parser.add_argument("--save-each-improvement", action="store_true")
    parser.add_argument("--bias", action="store_true")
    parser.add_argument(
        "--hp-objective",
        choices=("final", "boss_kill"),
        default="final",
        help="Optimize final HP after the whole route, or HP immediately after killing the 10F captain.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    sim = MotaSimulator(load_game_data(args.data), SimulatorConfig(stop_on_boss=False))
    base_rows = load_rows(Path(args.base_route))
    base_objectives = objective_rows(base_rows)
    solved, hp, route, final = replay_objectives(
        sim,
        base_objectives,
        args.prefix,
        args.hp_objective,
    )
    if not solved:
        raise SystemExit(f"base objective replay failed: hp={hp} final={final}")

    best_hp = hp
    best_route = route
    best_objectives = base_objectives
    solved_count = 0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    print(
        json.dumps(
            {
                "base_solved": solved,
                "base_hp": hp,
                "base_route_len": len(route),
                "base_objectives": len(base_objectives),
                "iterations": args.iterations,
                "seed": args.seed,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for iteration in range(1, args.iterations + 1):
        parent = best_objectives if rng.random() < 0.7 else base_objectives
        candidate_objectives = mutate_objectives(parent, rng, args.max_segment, args.bias)
        solved, hp, route, final = replay_objectives(
            sim,
            candidate_objectives,
            args.prefix,
            args.hp_objective,
        )
        if solved:
            solved_count += 1
        is_better = solved and (hp > best_hp or (hp == best_hp and len(route) < len(best_route)))
        if is_better:
            best_hp = hp
            best_route = route
            best_objectives = candidate_objectives
            if args.save_each_improvement:
                write_route_jsonl(best_route, out_dir / f"{args.prefix}_hp{best_hp}.jsonl")
            print(
                json.dumps(
                    {
                        "improvement": True,
                        "iteration": iteration,
                        "hp": best_hp,
                        "route_len": len(best_route),
                        "solved_count": solved_count,
                        "elapsed_sec": round(time.time() - start, 1),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        if iteration % 1000 == 0:
            print(
                json.dumps(
                    {
                        "iteration": iteration,
                        "best_hp": best_hp,
                        "best_len": len(best_route),
                        "solved_count": solved_count,
                        "elapsed_sec": round(time.time() - start, 1),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    best_path = out_dir / f"{args.prefix}_best.jsonl"
    summary_path = out_dir / f"{args.prefix}_summary.json"
    write_route_jsonl(best_route, best_path)
    summary = {
        "base_route": args.base_route,
        "route": str(best_path),
        "solved": True,
        "final_hp": best_hp,
        "route_len": len(best_route),
        "objective_len": len(best_objectives),
        "hp_objective": args.hp_objective,
        "boss_kill_hp": boss_kill_hp_from_route(best_route),
        "iterations": args.iterations,
        "seed": args.seed,
        "solved_count": solved_count,
        "elapsed_sec": round(time.time() - start, 1),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
