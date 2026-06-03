from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib.pyplot as plt
import numpy as np

from mota_env import MotaSimulator, load_game_data
from mota_env.rewards import Rewarder, boss_route_margin, current_stage_name, stage_names
from mota_solver.search import state_summary
from mota_rl.train_actor_critic import find_matching_action, load_route


OUT_DIR = PROJECT_ROOT / "report" / "iclr_mota_202606" / "figures"
DATA = PROJECT_ROOT / "artifacts" / "data" / "mota_first10.json"


ROUTES = {
    "Codex-guided hp403": PROJECT_ROOT
    / "artifacts"
    / "manual_exploration_20260524"
    / "manual_success_no_shop_true_10f_trap_optimized_hp403.jsonl",
    "Self-generated sword": PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "4090_alphazero_mdp_backup_20260601"
    / "routes"
    / "selfgen_sword_v3.jsonl",
    "Self-generated shield": PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "4090_alphazero_mdp_backup_20260601"
    / "routes"
    / "selfgen_shield_v5.jsonl",
}


def replay_route(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sim = MotaSimulator(load_game_data(str(DATA)))
    state = sim.reset()
    rewarder = Rewarder("stage_stat_pbrs", gamma=0.99)
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(load_route(str(path))):
        actions = sim.macro_actions(state)
        match = find_matching_action(actions, row.get("action") or {})
        if match is None:
            break
        action = actions[match]
        before = state.clone()
        transition = sim.apply_macro_action(state, action)
        reward = rewarder.score(sim, before, state, action, transition).total
        rows.append(
            {
                "index": index,
                "stage": current_stage_name(sim, before),
                "reward": reward,
                "hp": state.hp,
                "atk": state.atk,
                "def": state.defense,
                "boss_margin": boss_route_margin(sim, state),
                "action": action.get("label", ""),
            }
        )
        if not transition.ok or state.dead:
            break
    return rows, state_summary(state)


def plot_map_snapshot() -> None:
    sim = MotaSimulator(load_game_data(str(DATA)))
    state = sim.reset()
    floors = ["MT2", "MT5", "MT8", "MT10"]
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.0), constrained_layout=True)
    color_by_cls = {
        "terrains": 0,
        "items": 1,
        "enemys": 2,
        "npcs": 3,
        "animates": 4,
    }
    for ax, floor_id in zip(axes, floors):
        grid = state.floors[floor_id]
        values = np.zeros((len(grid), len(grid[0])), dtype=float)
        for y, row in enumerate(grid):
            for x, tile in enumerate(row):
                if tile == 0:
                    values[y, x] = 0
                elif sim.is_wall_tile(tile):
                    values[y, x] = 5
                elif sim.is_door_tile(tile):
                    values[y, x] = 4
                else:
                    values[y, x] = color_by_cls.get(sim.block_cls(tile) or "", 0)
        ax.imshow(values, cmap="viridis", interpolation="nearest")
        ax.set_title(floor_id)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("First-10-floor MOTA state abstraction")
    fig.savefig(OUT_DIR / "mota_floor_snapshots.png", dpi=220)
    plt.close(fig)


def plot_system_diagram() -> None:
    fig, ax = plt.subplots(figsize=(12.0, 4.4))
    ax.axis("off")
    boxes = [
        (0.12, 0.68, "Python simulator\nstrict MOTA rules"),
        (0.36, 0.68, "Fast macro actions\nBFS parent map\nlazy path recovery"),
        (0.60, 0.68, "Resource graph\nnode features\nexecutable mask"),
        (0.84, 0.68, "Go-Explore archive\nstepping-stone cells\nTop-K candidates"),
        (0.84, 0.24, "PUCT-MCTS\nsingle-agent backup\nG = r + gamma V"),
        (0.60, 0.24, "Learned potential\nhp403 -> Phi only\nno behavior cloning"),
        (0.36, 0.24, "Strict replay\nboss flag check\nno shop / no fly"),
    ]
    for x, y, text in boxes:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#eef4ff", edgecolor="#4676b8", linewidth=1.2),
            transform=ax.transAxes,
        )
    arrows = [
        ((0.23, 0.68), (0.29, 0.68)),
        ((0.47, 0.68), (0.53, 0.68)),
        ((0.71, 0.68), (0.77, 0.68)),
        ((0.84, 0.54), (0.84, 0.38)),
        ((0.73, 0.24), (0.68, 0.24)),
        ((0.49, 0.24), (0.43, 0.24)),
        ((0.36, 0.38), (0.36, 0.54)),
    ]
    for start, end in arrows:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(arrowstyle="->", lw=1.4, color="#34495e"),
            xycoords=ax.transAxes,
        )
    fig.savefig(OUT_DIR / "system_diagram.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_reward_and_stage_curves() -> None:
    route_rows: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for name, path in ROUTES.items():
        if path.exists():
            rows, summary = replay_route(path)
            route_rows[name] = rows
            summaries[name] = summary

    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    for name, rows in route_rows.items():
        if not rows:
            continue
        cumulative = np.cumsum([float(row["reward"]) for row in rows])
        ax.plot(cumulative, label=name, linewidth=2)
    ax.set_title("Cumulative shaped reward along replayed routes")
    ax.set_xlabel("macro step")
    ax.set_ylabel("cumulative stage-stat PBRS")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.savefig(OUT_DIR / "reward_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    order = {name: idx for idx, name in enumerate(stage_names())}
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    for name, rows in route_rows.items():
        if not rows:
            continue
        stages = [order.get(str(row["stage"]), 0) for row in rows]
        ax.step(range(len(stages)), stages, where="post", label=name, linewidth=2)
    ax.set_title("Stage progression during strict route replay")
    ax.set_xlabel("macro step")
    ax.set_ylabel("stage index")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.savefig(OUT_DIR / "stage_progression.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    with (OUT_DIR / "route_summaries.json").open("w", encoding="utf8") as handle:
        json.dump(summaries, handle, ensure_ascii=False, indent=2)


def plot_search_status() -> None:
    status = [
        ("Codex hp403", 6, 403),
        ("Self sword", 1, 0),
        ("Self shield", 2, 0),
        ("AZ mt10-resources", 4, 0),
        ("GE no-filter", 3, 162),
    ]
    labels = [row[0] for row in status]
    progress = [row[1] for row in status]
    hp = [row[2] for row in status]
    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(8.2, 3.5))
    ax1.bar(x - 0.18, progress, width=0.36, color="#4c78a8", label="stage/progress bucket")
    ax1.set_ylabel("progress bucket")
    ax1.set_ylim(0, 7)
    ax2 = ax1.twinx()
    ax2.bar(x + 0.18, hp, width=0.36, color="#f58518", label="final HP")
    ax2.set_ylabel("strict final HP")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_title("Current route families and observed progress")
    ax1.grid(True, axis="y", alpha=0.2)
    fig.legend(loc="upper right", bbox_to_anchor=(0.92, 0.9), fontsize=8)
    fig.savefig(OUT_DIR / "search_status.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_map_snapshot()
    plot_system_diagram()
    plot_reward_and_stage_curves()
    plot_search_status()


if __name__ == "__main__":
    main()
