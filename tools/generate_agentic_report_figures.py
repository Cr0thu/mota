from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"
ROUTE = ROOT / "artifacts" / "tmp" / "agentic_expertguide_solved_greedybeam_route.jsonl"


CHECKPOINTS = [
    ("Sword", 19, "go sword1 MT5:11,11"),
    ("Shield", 75, "go shield1 MT9:9,7"),
    ("MT10 blue gem", 172, "go blueGem MT10:2,6"),
    ("MT10 red gem", 235, "go redGem MT10:10,6"),
    ("Red key", 248, "go yellowKey MT8:9,1"),
    ("Red door", 281, "open redDoor MT10:6,9"),
    ("Mechanism", 282, "fight skeletonCaptain MT10:6,4"),
    ("Final boss", 290, "fight skeletonCaptain MT10:6,1"),
]


def load_route(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def after_value(row: dict, key: str, default: int = 0) -> int:
    after = row.get("after", {})
    if key == "def":
        return int(after.get("def", after.get("defense", default)))
    return int(after.get(key, default))


def key_value(row: dict, key: str) -> int:
    return int((row.get("after", {}).get("keys") or {}).get(key, 0))


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_architecture() -> None:
    fig, ax = plt.subplots(figsize=(12, 6.3))
    ax.set_axis_off()
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)

    colors = {
        "env": "#E8F3F1",
        "action": "#F7E8C8",
        "agent": "#E8ECF8",
        "arbiter": "#F3E8EF",
        "output": "#E7F0D8",
    }

    def box(x: float, y: float, w: float, h: float, text: str, color: str) -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.08,rounding_size=0.08",
            linewidth=1.2,
            edgecolor="#334155",
            facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10, color="#111827")

    def arrow(x1: float, y1: float, x2: float, y2: float, *, rad: float = 0.0) -> None:
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.1,
                color="#475569",
                connectionstyle=f"arc3,rad={rad}",
            )
        )

    box(0.4, 3.0, 1.8, 1.0, "MOTA\nstate", colors["env"])
    box(2.8, 3.0, 2.0, 1.0, "Simulator\nlegal macro actions", colors["action"])
    box(5.4, 5.2, 2.0, 0.75, "stage\nnavigator", colors["agent"])
    box(5.4, 4.2, 2.0, 0.75, "resource\neconomy", colors["agent"])
    box(5.4, 3.2, 2.0, 0.75, "combat\nthreshold", colors["agent"])
    box(5.4, 2.2, 2.0, 0.75, "boss\nobjective", colors["agent"])
    box(5.4, 1.2, 2.0, 0.75, "expert\ncurriculum", colors["agent"])
    box(8.2, 3.0, 1.8, 1.0, "Beam-search\narbiter", colors["arbiter"])
    box(10.6, 3.0, 1.0, 1.0, "route\nJSONL", colors["output"])

    arrow(2.2, 3.5, 2.8, 3.5)
    arrow(4.8, 3.5, 5.4, 5.55)
    arrow(4.8, 3.5, 5.4, 4.55)
    arrow(4.8, 3.5, 5.4, 3.55)
    arrow(4.8, 3.5, 5.4, 2.55)
    arrow(4.8, 3.5, 5.4, 1.55)
    arrow(7.4, 5.55, 8.2, 3.55)
    arrow(7.4, 4.55, 8.2, 3.55)
    arrow(7.4, 3.55, 8.2, 3.55)
    arrow(7.4, 2.55, 8.2, 3.55)
    arrow(7.4, 1.55, 8.2, 3.55)
    arrow(10.0, 3.5, 10.6, 3.5)

    ax.text(
        6,
        6.55,
        "Agentic RL-style planner: legal actions, multi-agent scoring, checkpoint-shaped search",
        ha="center",
        va="center",
        fontsize=11.5,
        weight="bold",
    )
    save(fig, "agentic_architecture")


def plot_checkpoint_timeline(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.set_xlim(0, len(rows) + 10)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Macro action step")
    ax.set_title("Checkpoint timeline of the solved agentic route")
    ax.hlines(0.5, 0, len(rows), color="#CBD5E1", linewidth=3)
    palette = ["#2563EB", "#059669", "#7C3AED", "#DC2626", "#D97706", "#0F766E", "#9333EA", "#111827"]
    label_positions = {
        "Sword": (19, 0.78),
        "Shield": (75, 0.23),
        "MT10 blue gem": (172, 0.78),
        "MT10 red gem": (235, 0.23),
        "Red key": (248, 0.78),
        "Red door": (267, 0.23),
        "Mechanism": (287, 0.78),
        "Final boss": (300, 0.23),
    }
    for idx, (name, step, _label) in enumerate(CHECKPOINTS):
        ax.scatter(step, 0.5, s=95, color=palette[idx], zorder=3)
        x_text, y_text = label_positions[name]
        ax.annotate(
            f"{name}\n{step}",
            xy=(step, 0.5),
            xytext=(x_text, y_text),
            ha="center",
            va="center",
            fontsize=8.3,
            arrowprops={"arrowstyle": "-", "color": palette[idx], "linewidth": 1.0},
        )
    save(fig, "checkpoint_timeline")


def plot_resources(rows: list[dict]) -> None:
    x = list(range(1, len(rows) + 1))
    hp = [after_value(r, "hp") for r in rows]
    atk = [after_value(r, "atk") for r in rows]
    defense = [after_value(r, "def") for r in rows]

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(x, hp, color="#2563EB", linewidth=2.0, label="HP")
    ax1.set_ylabel("HP", color="#2563EB")
    ax1.tick_params(axis="y", labelcolor="#2563EB")
    ax1.set_xlabel("Macro action step")
    ax1.set_title("Resource progression during the solved route")
    ax1.grid(True, axis="y", color="#E2E8F0")

    ax2 = ax1.twinx()
    ax2.plot(x, atk, color="#DC2626", linewidth=1.8, label="ATK")
    ax2.plot(x, defense, color="#059669", linewidth=1.8, label="DEF")
    ax2.set_ylabel("ATK / DEF")

    for name, step, _label in CHECKPOINTS:
        ax1.axvline(step, color="#94A3B8", linewidth=0.8, alpha=0.65)
    ax1.text(19, max(hp) * 0.92, "Sword", fontsize=8, rotation=90, va="top")
    ax1.text(75, max(hp) * 0.92, "Shield", fontsize=8, rotation=90, va="top")
    ax1.text(248, max(hp) * 0.92, "Red key", fontsize=8, rotation=90, va="top")
    ax1.text(290, max(hp) * 0.92, "Boss", fontsize=8, rotation=90, va="top")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    save(fig, "resource_progression")


def plot_keys(rows: list[dict]) -> None:
    x = list(range(1, len(rows) + 1))
    yellow = [key_value(r, "yellowKey") for r in rows]
    blue = [key_value(r, "blueKey") for r in rows]
    red = [key_value(r, "redKey") for r in rows]

    fig, ax = plt.subplots(figsize=(12, 4.4))
    ax.step(x, yellow, where="post", color="#D97706", linewidth=2, label="Yellow keys")
    ax.step(x, blue, where="post", color="#2563EB", linewidth=2, label="Blue keys")
    ax.step(x, red, where="post", color="#DC2626", linewidth=2, label="Red keys")
    ax.set_xlabel("Macro action step")
    ax.set_ylabel("Inventory count")
    ax.set_title("Key inventory planning over the solved route")
    ax.grid(True, axis="y", color="#E2E8F0")
    for name, step, _label in CHECKPOINTS:
        ax.axvline(step, color="#CBD5E1", linewidth=0.8, alpha=0.6)
    ax.legend(loc="upper right")
    save(fig, "key_inventory")


def highest_checkpoint(route_path: Path) -> tuple[str, int]:
    if not route_path.exists():
        return "missing", 0
    rows = load_route(route_path)
    labels = [str(r.get("action", {}).get("label", "")) for r in rows]
    checks = [
        ("Start", "always"),
        ("Sword", "sword1"),
        ("Shield", "shield1"),
        ("MT10 resources", "blueGem MT10:2,6"),
        ("Red key", "yellowKey MT8:9,1"),
        ("Red door", "redDoor MT10:6,9"),
        ("Mechanism", "skeletonCaptain MT10:6,4"),
        ("Boss defeated", "skeletonCaptain MT10:6,1"),
    ]
    best_name, best_idx = "Start", 0
    for idx, (name, token) in enumerate(checks):
        if token == "always" or any(token in label for label in labels):
            best_name, best_idx = name, idx
    return best_name, best_idx


def plot_experiment_progress() -> None:
    runs = [
        ("Smoke", "agentic_rl_smoke_route_dl.jsonl"),
        ("Checkpoint\nbeam", "agentic_checkpoint_local_beam360_route.jsonl"),
        ("Expert-token\nbias", "agentic_expertbias_local_beam560_route.jsonl"),
        ("MT10 prep", "agentic_mt10prep_local_beam680_route.jsonl"),
        ("Mechanism\nreached", "agentic_expertguide_exactbonus_greedybeam_route.jsonl"),
        ("Solved", "agentic_expertguide_solved_greedybeam_route.jsonl"),
    ]
    names: list[str] = []
    scores: list[int] = []
    labels: list[str] = []
    for name, filename in runs:
        stage, score = highest_checkpoint(ROOT / "artifacts" / "tmp" / filename)
        names.append(name)
        scores.append(score)
        labels.append(stage)

    fig, ax = plt.subplots(figsize=(11, 4.7))
    bars = ax.bar(names, scores, color=["#94A3B8", "#60A5FA", "#818CF8", "#A78BFA", "#F59E0B", "#10B981"])
    ax.set_ylim(0, 7.7)
    ax.set_ylabel("Highest checkpoint reached")
    ax.set_title("Experimental progression of agentic planning variants")
    ax.set_yticks(range(8))
    ax.set_yticklabels(["Start", "Sword", "Shield", "MT10 res.", "Red key", "Red door", "Mechanism", "Boss"])
    ax.grid(True, axis="y", color="#E2E8F0")
    for bar, label in zip(bars, labels):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.12, label, ha="center", va="bottom", fontsize=8)
    save(fig, "experiment_progression")


def main() -> None:
    rows = load_route(ROUTE)
    plot_architecture()
    plot_checkpoint_timeline(rows)
    plot_resources(rows)
    plot_keys(rows)
    plot_experiment_progress()
    print(f"Generated figures under {OUT}")


if __name__ == "__main__":
    main()
