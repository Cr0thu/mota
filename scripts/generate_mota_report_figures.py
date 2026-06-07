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
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image, ImageDraw

from mota_env import MotaSimulator, load_game_data
from mota_env.rewards import Rewarder, boss_route_margin, current_stage_name, stage_names
from mota_solver.search import state_summary
from mota_rl.train_actor_critic import find_matching_action, load_route


OUT_DIR = PROJECT_ROOT / "report" / "iclr_mota_202606" / "figures"
DATA = PROJECT_ROOT / "artifacts" / "data" / "mota_first10.json"
PROJECT_ASSET_DIR = (
    PROJECT_ROOT
    / "game"
    / "Falsh原版魔塔合集"
    / "51_2"
    / "project"
)


ROUTES = {
    "Reference trajectory": PROJECT_ROOT
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
    icons = _load_js_object(PROJECT_ASSET_DIR / "icons.js")
    tile_cache: dict[tuple[str, str], Image.Image] = {}

    def lookup_tile(cls: str | None, block_id: str | None, floor_id: str | None = None) -> Image.Image:
        if not cls or not block_id:
            return _base_ground_tile(tile_cache)
        if cls == "autotile":
            return _fallback_wall_tile(tile_cache)
        if cls == "animates" and block_id == "unbreakableWall":
            # The original editor sprite for unbreakableWall is a red X overlay.
            # In the report figure this is visually misleading.  Use floor-
            # specific visual aliases while leaving simulator semantics
            # unchanged: the Floor-2 story blockers are rendered as walls,
            # while the Floor-10 central trap blockers are rendered as ground.
            if floor_id == "MT10":
                return _base_ground_tile(tile_cache)
            if floor_id == "MT2":
                return _fallback_wall_tile(tile_cache)
            return _steel_door_tile(tile_cache)
        key = (cls, block_id)
        if key in tile_cache:
            return tile_cache[key]
        loc = icons.get(cls, {}).get(block_id)
        if loc is None:
            if cls == "animates" and block_id.endswith("Wall"):
                return _fallback_wall_tile(tile_cache)
            return _base_ground_tile(tile_cache)
        sheet = PROJECT_ASSET_DIR / "materials" / f"{cls}.png"
        img = _crop_material_tile(sheet, int(loc))
        tile_cache[key] = img
        return img

    rendered = []
    for floor_id in floors:
        grid = state.floors[floor_id]
        rendered.append(_render_floor(sim, grid, lookup_tile, label=floor_id))

    fig, axes = plt.subplots(1, len(rendered), figsize=(11.5, 3.15), constrained_layout=True)
    for ax, floor_id, image in zip(axes, floors, rendered):
        ax.imshow(image, interpolation="nearest")
        ax.set_title(floor_id, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#444444")
            spine.set_linewidth(0.8)
    fig.savefig(OUT_DIR / "mota_floor_snapshots.png", dpi=220)
    plt.close(fig)


def _load_js_object(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    body = text.split("=", 1)[1].strip()
    if body.endswith(";"):
        body = body[:-1]
    return json.loads(body)


def _crop_material_tile(sheet: Path, loc: int, tile_size: int = 32) -> Image.Image:
    # In H5Mota material sheets, icons.js locations are row indices.  Sheets
    # such as animates.png, enemys.png, and npcs.png store animation frames
    # horizontally; the static paper figure should use the first frame.
    image = Image.open(sheet).convert("RGBA")
    x = 0
    y = loc * tile_size
    return image.crop((x, y, x + tile_size, y + tile_size))


def _base_ground_tile(tile_cache: dict[tuple[str, str], Image.Image]) -> Image.Image:
    key = ("terrains", "ground")
    if key not in tile_cache:
        tile_cache[key] = Image.open(PROJECT_ASSET_DIR / "materials" / "ground.png").convert("RGBA")
    return tile_cache[key]


def _fallback_wall_tile(tile_cache: dict[tuple[str, str], Image.Image]) -> Image.Image:
    key = ("animates", "yellowWall")
    if key not in tile_cache:
        tile_cache[key] = _crop_material_tile(PROJECT_ASSET_DIR / "materials" / "animates.png", 10)
    return tile_cache[key]


def _steel_door_tile(tile_cache: dict[tuple[str, str], Image.Image]) -> Image.Image:
    key = ("animates", "steelDoor")
    if key not in tile_cache:
        tile_cache[key] = _crop_material_tile(PROJECT_ASSET_DIR / "materials" / "animates.png", 9)
    return tile_cache[key]


def _render_floor(
    sim: MotaSimulator,
    grid: list[list[int]],
    lookup_tile: Any,
    label: str,
    tile_size: int = 32,
) -> Image.Image:
    height = len(grid)
    width = len(grid[0])
    base = _base_ground_tile({})
    image = Image.new("RGBA", (width * tile_size, height * tile_size), (255, 255, 255, 255))
    for y in range(height):
        for x in range(width):
            image.paste(base, (x * tile_size, y * tile_size), base)
    for y, row in enumerate(grid):
        for x, tile in enumerate(row):
            tile = int(tile)
            if tile == 0:
                continue
            block = lookup_tile(sim.block_cls(tile), sim.block_id(tile), label)
            image.paste(block, (x * tile_size, y * tile_size), block)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width * tile_size - 1, height * tile_size - 1), outline=(45, 45, 45, 255), width=2)
    draw.rectangle((4, 4, 58, 24), fill=(0, 0, 0, 150))
    draw.text((9, 8), label, fill=(255, 255, 255, 255))
    return image


def plot_system_diagram() -> None:
    fig, ax = plt.subplots(figsize=(10.2, 3.55))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def box(
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        lines: list[str],
        *,
        face: str = "#ffffff",
        edge: str = "#222222",
        lw: float = 0.85,
    ) -> None:
        ax.add_patch(
            Rectangle(
                (x, y),
                w,
                h,
                linewidth=lw,
                edgecolor=edge,
                facecolor=face,
                transform=ax.transAxes,
                zorder=2,
            )
        )
        ax.text(
            x + 0.0125,
            y + h - 0.035,
            title,
            ha="left",
            va="top",
            fontsize=8.4,
            fontweight="bold",
            color="#111111",
            transform=ax.transAxes,
            zorder=3,
        )
        for idx, line in enumerate(lines):
            ax.text(
                x + 0.0125,
                y + h - 0.083 - idx * 0.038,
                line,
                ha="left",
                va="top",
                fontsize=6.9,
                color="#222222",
                transform=ax.transAxes,
                zorder=3,
            )

    def arrow(
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        dashed: bool = False,
        label: str | None = None,
        rad: float = 0.0,
        lw: float = 0.85,
    ) -> None:
        patch = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=lw,
            color="#222222",
            linestyle=(0, (3.2, 2.2)) if dashed else "solid",
            connectionstyle=f"arc3,rad={rad}",
            transform=ax.transAxes,
            zorder=4,
        )
        ax.add_patch(patch)
        if label:
            mx = (start[0] + end[0]) / 2
            my = (start[1] + end[1]) / 2 + 0.020
            ax.text(
                mx,
                my,
                label,
                ha="center",
                va="bottom",
                fontsize=7.2,
                color="#111111",
                bbox=dict(boxstyle="square,pad=0.12", facecolor="white", edgecolor="none"),
                transform=ax.transAxes,
                zorder=5,
            )

    ax.plot([0.055, 0.945], [0.565, 0.565], color="#b8b8b8", linewidth=0.55, transform=ax.transAxes)
    ax.text(0.055, 0.920, "Planning loop", fontsize=9.0, fontweight="bold", transform=ax.transAxes)
    ax.text(0.055, 0.508, "Auxiliary potential", fontsize=9.0, fontweight="bold", transform=ax.transAxes)

    top_y, top_h = 0.655, 0.190
    bot_y, bot_h = 0.205, 0.175
    modules = {
        "env": (0.055, top_y, 0.132, top_h),
        "mask": (0.242, top_y, 0.150, top_h),
        "archive": (0.448, top_y, 0.132, top_h),
        "mcts": (0.635, top_y, 0.148, top_h),
        "eval": (0.840, top_y, 0.122, top_h),
        "demo": (0.105, bot_y, 0.180, bot_h),
        "phi": (0.398, bot_y, 0.165, bot_h),
        "shape": (0.675, bot_y, 0.185, bot_h),
    }

    light = "#fafafa"
    shaded = "#f2f2f2"
    box(*modules["env"], "Simulator", ["exact transition", "$s'=f(s,a)$"], face=light)
    box(*modules["mask"], "Action mask", ["reachability BFS", "$\\mathcal{A}_{\\mathrm{valid}}(s)$"], face=light)
    box(*modules["archive"], "Archive", ["cell $\\chi(s)$", "top-$K$ states"], face=shaded)
    box(*modules["mcts"], "PUCT-MCTS", ["$Q(s,a)+U(s,a)$", "MDP backup"], face=shaded)
    box(*modules["eval"], "Replay", ["terminal event", "$\\mathrm{HP}_T>0$"], face=light)
    box(*modules["demo"], "Reference", ["ordered states", "ranking pairs"], face="#ffffff", edge="#555555")
    box(*modules["phi"], "Potential", ["$\\Phi_\\phi(s,g)$", "factor weights"], face="#ffffff", edge="#555555")
    box(*modules["shape"], "Shaped return", ["$r_{task}+\\lambda\\Delta\\Phi$", "evaluation fixed"], face="#ffffff", edge="#555555")

    for left, right in [("env", "mask"), ("mask", "archive"), ("archive", "mcts"), ("mcts", "eval")]:
        lx, ly, lw, lh = modules[left]
        rx, ry, rw, rh = modules[right]
        arrow((lx + lw, ly + lh / 2), (rx, ry + rh / 2))

    # Archive expansion is iterative: PUCT adds newly validated states back to
    # the archive.  Draw it as a thin loop rather than another module.
    ax.add_patch(
        FancyArrowPatch(
            (0.723, 0.842),
            (0.513, 0.842),
            connectionstyle="arc3,rad=0.28",
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.75,
            color="#222222",
            linestyle=(0, (3.2, 2.2)),
            transform=ax.transAxes,
            zorder=4,
        )
    )
    ax.text(0.612, 0.892, "new cells", fontsize=6.9, ha="center", transform=ax.transAxes)

    arrow((0.285, bot_y + bot_h / 2), (0.398, bot_y + bot_h / 2), dashed=True)
    arrow((0.563, bot_y + bot_h / 2), (0.675, bot_y + bot_h / 2), dashed=True)
    arrow((0.768, bot_y + bot_h), (0.708, top_y), dashed=True, rad=-0.10, label="$\\Delta\\Phi$")

    ax.text(
        0.055,
        0.065,
        "Solid arrows denote simulator/search data flow. Dashed arrows denote auxiliary shaping; strict replay remains the success criterion.",
        fontsize=7.0,
        color="#333333",
        transform=ax.transAxes,
    )

    fig.savefig(OUT_DIR / "system_diagram.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT_DIR / "system_diagram.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def plot_agentic_architecture() -> None:
    fig, ax = plt.subplots(figsize=(11.0, 3.8))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def box(x: float, y: float, w: float, h: float, title: str, lines: list[str], face: str = "#f5f5f5") -> None:
        ax.add_patch(Rectangle((x, y), w, h, linewidth=1.0, edgecolor="#333333", facecolor=face))
        title_y = y + h * (0.68 if lines else 0.50)
        ax.text(x + w / 2, title_y, title, ha="center", va="center", fontsize=9.5, fontweight="bold")
        for i, line in enumerate(lines):
            ax.text(x + w / 2, y + h * 0.34 - 0.038 * i, line, ha="center", va="center", fontsize=7.8)

    def arrow(x1: float, y1: float, x2: float, y2: float, dashed: bool = False) -> None:
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=9,
                linewidth=0.95,
                linestyle="--" if dashed else "-",
                color="#333333",
            )
        )

    box(0.045, 0.45, 0.13, 0.20, "State", ["map and hero", "flags and keys"], face="#ffffff")
    box(0.225, 0.45, 0.16, 0.20, "Simulator", ["legal actions", "after-states"], face="#ffffff")
    box(0.435, 0.75, 0.18, 0.14, "Stage", ["milestone progress"])
    box(0.435, 0.57, 0.18, 0.14, "Resource", ["key and item economy"])
    box(0.435, 0.39, 0.18, 0.14, "Combat", ["damage thresholds"])
    box(0.435, 0.21, 0.18, 0.14, "Terminal", ["gate and boss readiness"])
    box(0.675, 0.45, 0.15, 0.20, "Beam arbiter", ["weighted scores", "top-K states"], face="#ffffff")
    box(0.875, 0.45, 0.10, 0.20, "Route", ["strict replay", "JSONL"], face="#ffffff")
    box(0.67, 0.20, 0.16, 0.13, "Memory", ["trace and revisits"], face="#ffffff")

    arrow(0.175, 0.55, 0.225, 0.55)
    arrow(0.385, 0.55, 0.435, 0.82)
    arrow(0.385, 0.55, 0.435, 0.64)
    arrow(0.385, 0.55, 0.435, 0.46)
    arrow(0.385, 0.55, 0.435, 0.28)
    arrow(0.615, 0.82, 0.675, 0.59)
    arrow(0.615, 0.64, 0.675, 0.56)
    arrow(0.615, 0.46, 0.675, 0.52)
    arrow(0.615, 0.28, 0.675, 0.50)
    arrow(0.825, 0.55, 0.875, 0.55)
    arrow(0.75, 0.33, 0.75, 0.45, dashed=True)
    arrow(0.745, 0.45, 0.745, 0.33, dashed=True)

    ax.text(
        0.5,
        0.06,
        "All decisions are made over simulator-verified legal candidates; reference alignment is used only in this planner.",
        ha="center",
        va="center",
        fontsize=8.2,
        color="#333333",
    )
    fig.savefig(OUT_DIR / "agentic_architecture.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT_DIR / "agentic_architecture.pdf", bbox_inches="tight")
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
        ("Reference", 6, 403),
        ("Self-generated sword", 1, 0),
        ("Self-generated shield", 2, 0),
        ("Policy-guided suffix", 4, 0),
        ("Archive search", 3, 162),
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


def plot_agentic_route_summary() -> None:
    checkpoints = [
        ("Start", 0, 400, 10, 10),
        ("Sword", 19, 754, 20, 10),
        ("Shield", 75, 338, 21, 20),
        ("F10-B", 172, 176, 26, 27),
        ("F10-R", 235, 715, 27, 27),
        ("R-key", 248, 497, 27, 27),
        ("R-door", 281, 1070, 27, 27),
        ("Gate", 282, 995, 27, 27),
        ("Done", 290, 436, 27, 27),
    ]
    labels = [row[0] for row in checkpoints]
    steps = np.array([row[1] for row in checkpoints])
    hp = np.array([row[2] for row in checkpoints])
    atk = np.array([row[3] for row in checkpoints])
    defense = np.array([row[4] for row in checkpoints])

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.15), constrained_layout=True)
    ax = axes[0]
    y = np.zeros_like(steps)
    ax.hlines(0, steps.min(), steps.max(), color="#9aa4b2", linewidth=1.0)
    ax.scatter(steps, y, s=38, color="#4c78a8", zorder=3)
    for idx, (label, step) in enumerate(zip(labels, steps)):
        if label in {"R-door", "Gate", "Done"}:
            continue
        offset = 0.22 if idx % 2 == 0 else -0.24
        va = "bottom" if offset > 0 else "top"
        ax.text(step, offset, f"{label}\n{step}", ha="center", va=va, fontsize=7.3)
        ax.vlines(step, 0, offset * 0.72, color="#c7cdd6", linewidth=0.7)
    ax.text(286, 0.25, "Final\n281-290", ha="center", va="bottom", fontsize=7.3)
    ax.vlines([281, 282, 290], 0, 0.16, color="#c7cdd6", linewidth=0.7)
    ax.set_ylim(-0.48, 0.48)
    ax.set_yticks([])
    ax.set_xlabel("macro-action step")
    ax.set_title("Reference route checkpoints")
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.grid(True, axis="x", alpha=0.18)

    ax = axes[1]
    hp_line = ax.plot(steps, hp, marker="o", linewidth=1.8, color="#4c78a8", label="HP")
    ax.set_xlabel("macro-action step")
    ax.set_ylabel("HP")
    ax.grid(True, axis="y", alpha=0.22)
    ax2 = ax.twinx()
    atk_line = ax2.plot(steps, atk, marker="s", linewidth=1.6, color="#f58518", label="ATK")
    def_line = ax2.plot(steps, defense, marker="^", linewidth=1.6, color="#54a24b", label="DEF")
    ax2.set_ylabel("ATK / DEF")
    ax.set_title("Checkpoint-level resources")
    lines = hp_line + atk_line + def_line
    ax.legend(lines, [line.get_label() for line in lines], frameon=False, fontsize=8, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    fig.savefig(OUT_DIR / "agentic_route_summary.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT_DIR / "agentic_route_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_experiment_summary() -> None:
    plt.rcParams.update(
        {
            "axes.titleweight": "bold",
            "axes.titlesize": 11.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
        }
    )
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(11.4, 3.35),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.25, 1.35]},
    )

    # Panel A: potential ranking quality.
    ax = axes[0]
    labels = ["rank@1", "rank@3"]
    values = [52.1, 85.8]
    bars = ax.bar(labels, values, color=["#4c78a8", "#7b8f63"], width=0.58)
    ax.set_ylim(0, 100)
    ax.set_ylabel("validated states (%)")
    ax.set_title("(a) Potential ranking")
    ax.text(
        0.5,
        12,
        "mean rank = 2.00",
        ha="center",
        fontsize=8.5,
        color="#3f4a5a",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#f4f6f8", edgecolor="none"),
    )
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 3.0,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#2f3542",
        )
    ax.grid(True, axis="y", alpha=0.18)

    # Panel B: policy/value training curve.
    ax = axes[1]
    steps = np.array([0, 1000, 4000, 9100, 11999])
    total_loss = np.array([1.9283, 0.1051, 0.0127, 0.0064, 0.0139])
    policy_loss = np.array([1.7195, 0.1043, 0.0124, 0.0063, 0.0138])
    ax.plot(steps, total_loss, marker="o", linewidth=1.8, color="#4c78a8", label="total")
    ax.plot(steps, policy_loss, marker="s", linewidth=1.5, color="#f58518", label="policy")
    ax.axvline(9100, color="#6b7280", linestyle="--", linewidth=1.0)
    ax.text(
        9100,
        0.018,
        "selected\ncheckpoint",
        ha="center",
        va="bottom",
        fontsize=7.8,
        color="#3f4a5a",
    )
    ax.set_yscale("log")
    ax.set_xlabel("optimization step")
    ax.set_ylabel("loss")
    ax.set_title("(b) Graph policy/value prior")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, which="both", axis="y", alpha=0.22)

    # Panel C: deterministic validation outcomes.
    ax = axes[2]
    methods = ["agentic ref.", "reference", "PUCT prior", "no-demo archive"]
    hp = np.array([436, 403, 315, 0])
    colors = ["#9467bd", "#7b8f63", "#4c78a8", "#c7cdd6"]
    y = np.arange(len(methods))
    bars = ax.barh(y, hp, color=colors, height=0.58)
    bars[-1].set_hatch("//")
    bars[-1].set_edgecolor("#7b8794")
    bars[-1].set_linewidth(0.7)
    ax.set_yticks(y, methods)
    ax.invert_yaxis()
    ax.set_xlim(0, 475)
    ax.set_xlabel("terminal HP under strict replay")
    ax.set_title("(c) Task-level validation")
    for idx, value in enumerate(hp):
        if value > 0:
            ax.text(value + 8, idx, f"{value}", va="center", fontsize=8.5, color="#2f3542")
        else:
            ax.text(18, idx, "not solved", va="center", fontsize=8.5, color="#3f4a5a")
    ax.grid(True, axis="x", alpha=0.18)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.savefig(OUT_DIR / "experiment_summary.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_map_snapshot()
    plot_system_diagram()
    plot_agentic_architecture()
    plot_agentic_route_summary()
    plot_experiment_summary()
    plot_reward_and_stage_curves()
    plot_search_status()


if __name__ == "__main__":
    main()
