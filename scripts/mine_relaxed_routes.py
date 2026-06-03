from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mota_env import MotaSimulator, SimulatorConfig, load_game_data
from mota_solver.search import state_summary, write_route_jsonl
from mota_solver.solve_staged import load_stage_weights
from mota_solver.staged import solve_staged_first10


MANUAL_ROUTE_MARKERS = (
    "manual_exploration_20260524",
    "manual_success_no_shop_true_10f_trap",
    "optimized_hp403",
)


@dataclass(frozen=True)
class MineConfig:
    index: int
    seed: int
    relaxed_min_hp: int
    variant: str
    stage_weights: dict[str, float] | None


def parse_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def route_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]


def route_has_forbidden_actions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    violations = []
    for index, row in enumerate(rows):
        action = row.get("action", {})
        kind = str(action.get("kind", ""))
        label = str(action.get("label", ""))
        if kind in {"shop", "fly_shop"} or label.startswith("shop ") or label.startswith("fly shop "):
            violations.append({"index": index, "type": "shop", "label": label, "kind": kind})
        if kind in {"fly", "fly_shop"} or label.startswith("fly "):
            violations.append({"index": index, "type": "fly", "label": label, "kind": kind})
    return violations


def analyze_replay(
    data_path: str,
    route_path: Path,
    *,
    allow_negative_hp: bool,
    relaxed_min_hp: int,
) -> dict[str, Any]:
    rows = route_rows(route_path)
    sim = MotaSimulator(
        load_game_data(data_path),
        SimulatorConfig(allow_negative_hp=allow_negative_hp, min_hp=relaxed_min_hp),
    )
    state = sim.reset()
    min_hp = state.hp
    failure: dict[str, Any] | None = None
    hp_trace_tail: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        action = row.get("action", {})
        before = state.clone()
        transition = sim.apply_macro_action(state, action)
        min_hp = min(min_hp, state.hp)
        hp_trace_tail.append(
            {
                "index": index,
                "label": action.get("label", ""),
                "hp": state.hp,
                "atk": state.atk,
                "def": state.defense,
                "floor": state.floor_id,
            }
        )
        hp_trace_tail = hp_trace_tail[-12:]
        if not transition.ok or state.dead:
            failure = {
                "index": index,
                "label": action.get("label", ""),
                "message": transition.message,
                "before": state_summary(before),
                "after": state_summary(state),
                "combat": combat_failure_context(sim, before, action),
            }
            break
    return {
        "route": str(route_path),
        "mode": "relaxed" if allow_negative_hp else "strict",
        "steps_replayed": len(rows) if failure is None else failure["index"] + 1,
        "route_len": len(rows),
        "failed": failure is not None,
        "failure": failure,
        "solved": bool(state.flags.get("10f战胜骷髅队长")) and not state.dead,
        "boss_flag": bool(state.flags.get("10f战胜骷髅队长")),
        "final": state_summary(state),
        "min_hp": min_hp,
        "hp_debt": max(0, -min_hp),
        "hp_trace_tail": hp_trace_tail,
        "violations": route_has_forbidden_actions(rows),
    }


def combat_failure_context(sim: MotaSimulator, before, action: dict[str, Any]) -> dict[str, Any]:
    label = str(action.get("label", ""))
    target = action.get("target")
    if not label.startswith("fight") or not isinstance(target, list) or len(target) < 3:
        return {}
    floor_id, x, y = str(target[0]), int(target[1]), int(target[2])
    tile = sim.tile(before, x, y, floor_id)
    enemy_id = sim.block_id(tile)
    info = sim.damage_info(before, enemy_id) if enemy_id else None
    if info is None:
        return {"enemy": enemy_id, "damage": None}
    return {
        "enemy": enemy_id,
        "damage": info["damage"],
        "turn": info.get("turn"),
        "hp_before": before.hp,
        "hp_deficit_for_strict": max(0, info["damage"] + 1 - before.hp),
        "target": [floor_id, x, y],
    }


def score_candidate(relaxed: dict[str, Any], strict: dict[str, Any]) -> float:
    final = relaxed.get("final", {})
    strict_final = strict.get("final", {})
    score = 0.0
    score += 1_000_000.0 if relaxed.get("boss_flag") else 0.0
    score += 250_000.0 if strict.get("boss_flag") else 0.0
    score += max(-6000.0, float(final.get("hp", 0))) * 40.0
    score -= float(relaxed.get("hp_debt", 0)) * 60.0
    score += float(final.get("atk", 0)) * 1800.0
    score += float(final.get("def", 0)) * 1800.0
    score += float(final.get("money", 0)) * 20.0
    score += min(float(strict.get("steps_replayed", 0)), float(relaxed.get("route_len", 0))) * 120.0
    score += float(strict_final.get("hp", 0)) * 3.0
    score -= float(relaxed.get("route_len", 0)) * 8.0
    score -= len(relaxed.get("violations", [])) * 1_000_000.0
    return score


def stage_weight_variants() -> dict[str, dict[str, float] | None]:
    return {
        "default": None,
        "stat_critical": {
            "threshold": 2.2,
            "combat": 1.7,
            "stage_pre_shield_gems": 1.8,
            "stage_mt8_gems": 1.8,
            "stage_mid_gems": 1.8,
            "stage_low_gems": 1.8,
            "boss_margin": 1.5,
        },
        "hp_debt": {
            "asset": 2.0,
            "stage_mt8_hp_ready": 2.4,
            "stage_boss_ready": 2.2,
            "stage_mt10_resources": 1.8,
            "boss_margin": 1.8,
        },
        "key_route": {
            "key_pressure": 2.2,
            "lookahead": 1.8,
            "global_resource": 1.5,
            "stage_mt10_blue_ready": 2.0,
            "stage_mt10_yellow_ready": 2.0,
        },
        "boss_finish": {
            "stage_red_key": 1.8,
            "stage_trap": 2.2,
            "stage_boss": 2.8,
            "boss_margin": 2.8,
            "combat": 1.6,
        },
    }


def build_configs(args: argparse.Namespace) -> list[MineConfig]:
    seeds = parse_ints(args.seeds)
    min_hps = parse_ints(args.relaxed_min_hps)
    variants = stage_weight_variants()
    if args.variants:
        selected = [name.strip() for name in args.variants.split(",") if name.strip()]
        variants = {name: variants[name] for name in selected}
    configs: list[MineConfig] = []
    index = 0
    for seed in seeds:
        for relaxed_min_hp in min_hps:
            for variant, weights in variants.items():
                configs.append(MineConfig(index, seed, relaxed_min_hp, variant, weights))
                index += 1
    rng = random.Random(args.shuffle_seed)
    rng.shuffle(configs)
    if args.max_configs > 0:
        configs = configs[: args.max_configs]
    return [MineConfig(i, cfg.seed, cfg.relaxed_min_hp, cfg.variant, cfg.stage_weights) for i, cfg in enumerate(configs)]


def mine_one(args_dict: dict[str, Any], config: MineConfig) -> dict[str, Any]:
    data = args_dict["data"]
    out_dir = Path(args_dict["out_dir"])
    max_expansions = int(args_dict["max_expansions_per_stage"])
    keep_per_parent = int(args_dict["keep_per_parent"])
    frontier_size = int(args_dict["frontier_size"])
    trace_limit = int(args_dict["trace_limit"])
    stop_stage = str(args_dict.get("stop_stage") or "")

    stem = f"{config.index:03d}_{config.variant}_seed{config.seed}_min{abs(config.relaxed_min_hp)}"
    route_path = out_dir / "routes" / f"{stem}.jsonl"
    summary_path = out_dir / "summaries" / f"{stem}.json"
    route_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    sim = MotaSimulator(
        load_game_data(data),
        SimulatorConfig(allow_negative_hp=True, min_hp=config.relaxed_min_hp),
    )
    result = solve_staged_first10(
        sim,
        max_expansions_per_stage=max_expansions,
        keep_per_parent=keep_per_parent,
        frontier_size=frontier_size,
        seed=config.seed,
        trace_limit=trace_limit,
        stage_weights=config.stage_weights,
        stop_stage=stop_stage or None,
    )
    write_route_jsonl(result.route, route_path)
    relaxed = analyze_replay(
        data,
        route_path,
        allow_negative_hp=True,
        relaxed_min_hp=config.relaxed_min_hp,
    )
    strict = analyze_replay(
        data,
        route_path,
        allow_negative_hp=False,
        relaxed_min_hp=0,
    )
    row = {
        "config": {
            "index": config.index,
            "seed": config.seed,
            "relaxed_min_hp": config.relaxed_min_hp,
            "variant": config.variant,
            "stage_weights": config.stage_weights,
        },
        "route": str(route_path),
        "summary": str(summary_path),
        "search": {
            "solved": result.solved,
            "expansions": result.expansions,
            "route_len": len(result.route),
            "final": state_summary(result.state),
            "stages": [
                {
                    "stage": item.stage,
                    "solved": item.solved,
                    "expansions": item.expansions,
                    "frontier_size": item.frontier_size,
                    "best": item.best,
                }
                for item in result.stage_summaries
            ],
        },
        "relaxed_replay": relaxed,
        "strict_replay": strict,
        "score": score_candidate(relaxed, strict),
        "elapsed_sec": round(time.time() - started, 2),
    }
    summary_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf8")
    return row


def analyze_existing(args: argparse.Namespace) -> list[dict[str, Any]]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, raw_path in enumerate(args.analyze_routes):
        path = Path(raw_path)
        marker_text = str(path)
        if not args.allow_manual and any(marker in marker_text for marker in MANUAL_ROUTE_MARKERS):
            raise SystemExit(f"Refusing manual route without --allow-manual: {path}")
        relaxed = analyze_replay(
            args.data,
            path,
            allow_negative_hp=True,
            relaxed_min_hp=args.analysis_relaxed_min_hp,
        )
        strict = analyze_replay(args.data, path, allow_negative_hp=False, relaxed_min_hp=0)
        row = {
            "config": {"index": index, "source": "existing"},
            "route": str(path),
            "relaxed_replay": relaxed,
            "strict_replay": strict,
            "score": score_candidate(relaxed, strict),
        }
        rows.append(row)
    return rows


def write_report(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda item: item["score"], reverse=True)
    manifest = out_dir / "manifest.jsonl"
    with manifest.open("w", encoding="utf8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    top = rows[:20]
    report = [
        "# Relaxed Route Mining Report",
        "",
        f"- candidates: `{len(rows)}`",
        f"- relaxed boss flags: `{sum(int(row['relaxed_replay'].get('boss_flag')) for row in rows)}`",
        f"- strict boss flags: `{sum(int(row['strict_replay'].get('boss_flag')) for row in rows)}`",
        "",
        "## Top Candidates",
        "",
    ]
    for rank, row in enumerate(top, 1):
        relaxed = row["relaxed_replay"]
        strict = row["strict_replay"]
        final = relaxed.get("final", {})
        failure = strict.get("failure") or {}
        combat = failure.get("combat") or {}
        cfg = row.get("config", {})
        report.append(
            " | ".join(
                [
                    f"{rank}. score `{row['score']:.1f}`",
                    f"variant `{cfg.get('variant', cfg.get('source', ''))}`",
                    f"seed `{cfg.get('seed', '')}`",
                    f"min_hp `{cfg.get('relaxed_min_hp', '')}`",
                    f"boss `{relaxed.get('boss_flag')}`",
                    f"relaxed_hp `{final.get('hp')}`",
                    f"hp_debt `{relaxed.get('hp_debt')}`",
                    f"strict_steps `{strict.get('steps_replayed')}/{strict.get('route_len')}`",
                    f"fail `{failure.get('label', '')}`",
                    f"need_hp `{combat.get('hp_deficit_for_strict', '')}`",
                    f"route `{row.get('route')}`",
                ]
            )
        )
    (out_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf8")
    if rows:
        best_path = out_dir / "best_candidate.json"
        best_path.write_text(json.dumps(rows[0], ensure_ascii=False, indent=2), encoding="utf8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="artifacts/data/mota_first10.json")
    parser.add_argument("--out-dir", default="artifacts/runs/relaxed_mining")
    parser.add_argument("--seeds", default="20261301,20261302,20261303")
    parser.add_argument("--relaxed-min-hps", default="-800,-1400,-2200")
    parser.add_argument("--variants", default="")
    parser.add_argument("--max-configs", type=int, default=0)
    parser.add_argument("--shuffle-seed", type=int, default=20260525)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-expansions-per-stage", type=int, default=15000)
    parser.add_argument("--keep-per-parent", type=int, default=80)
    parser.add_argument("--frontier-size", type=int, default=48)
    parser.add_argument("--trace-limit", type=int, default=0)
    parser.add_argument("--stop-stage", default="")
    parser.add_argument("--analyze-routes", nargs="*", default=[])
    parser.add_argument("--analysis-relaxed-min-hp", type=int, default=-3000)
    parser.add_argument("--allow-manual", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    started = time.time()
    if args.analyze_routes:
        rows = analyze_existing(args)
    else:
        configs = build_configs(args)
        args_dict = vars(args)
        rows = []
        if args.workers <= 1:
            for cfg in configs:
                rows.append(mine_one(args_dict, cfg))
                print(json.dumps({"done": len(rows), "total": len(configs), "score": rows[-1]["score"], "route": rows[-1]["route"]}, ensure_ascii=False), flush=True)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(mine_one, args_dict, cfg) for cfg in configs]
                for future in concurrent.futures.as_completed(futures):
                    row = future.result()
                    rows.append(row)
                    print(json.dumps({"done": len(rows), "total": len(configs), "score": row["score"], "route": row["route"]}, ensure_ascii=False), flush=True)
    write_report(rows, out_dir)
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "candidates": len(rows),
                "elapsed_sec": round(time.time() - started, 2),
                "report": str(out_dir / "report.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
