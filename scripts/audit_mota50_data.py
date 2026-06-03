#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
VIS_DIR = PROJECT_ROOT / "tools" / "visualizer"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(VIS_DIR) not in sys.path:
    sys.path.insert(0, str(VIS_DIR))

from mota_env import MotaSimulator, load_game_data  # noqa: E402
import database as visualizer_db  # noqa: E402


DATA_PATH = PROJECT_ROOT / "artifacts" / "data" / "mota_first10.json"
DEFAULT_REPORT = PROJECT_ROOT / "artifacts" / "runs" / "mota50_data_audit.md"
KEY_ENEMIES = [
    "greenSlime",
    "redSlime",
    "bat",
    "skeleton",
    "skeletonSoldier",
    "skeletonCaptain",
    "bluePriest",
    "yellowGuard",
    "blueGuard",
]
KEY_ITEMS = [
    "redGem",
    "blueGem",
    "redPotion",
    "bluePotion",
    "yellowKey",
    "blueKey",
    "redKey",
    "sword1",
    "shield1",
    "fly",
    "centerFly",
]


def md_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def raw_title(raw: dict) -> str:
    return raw.get("firstData", {}).get("title") or raw.get("firstData", {}).get("name") or "(unknown)"


def source_floor_files(source_project: str) -> list[str]:
    floors_dir = PROJECT_ROOT / source_project / "floors"
    if not floors_dir.exists():
        floors_dir = Path(source_project) / "floors"
    if not floors_dir.exists():
        return []
    return sorted(
        (path.stem for path in floors_dir.glob("MT*.js") if path.stem[2:].isdigit()),
        key=lambda x: int(x[2:]),
    )


def summarize_items(raw: dict) -> list[list[object]]:
    values = raw.get("values", {})
    items = raw.get("items", {})
    rows = []
    for item_id in KEY_ITEMS:
        item = items.get(item_id, {})
        effect = ""
        if item_id == "redGem":
            effect = f"+ATK {values.get('redGem')}"
        elif item_id == "blueGem":
            effect = f"+DEF {values.get('blueGem')}"
        elif item_id == "redPotion":
            effect = f"+HP {values.get('redPotion')}"
        elif item_id == "bluePotion":
            effect = f"+HP {values.get('bluePotion')}"
        elif item_id == "sword1":
            effect = "+ATK 10"
        elif item_id == "shield1":
            effect = "+DEF 10"
        elif item_id in {"yellowKey", "blueKey", "redKey"}:
            effect = "key +1"
        elif item_id == "fly":
            effect = "楼层传送器；当前研究场景删除/禁用"
        elif item_id == "centerFly":
            effect = "中心对称飞行器；当前前十层场景不使用"
        rows.append([item_id, item.get("name", ""), item.get("cls", ""), effect])
    return rows


def summarize_enemies(raw: dict, visualizer: dict) -> list[list[object]]:
    rows = []
    for enemy_id in KEY_ENEMIES:
        enemy = raw.get("enemys", {}).get(enemy_id, {})
        vis_enemy = visualizer.get("enemies", {}).get(enemy_id, {})
        rows.append([
            enemy_id,
            enemy.get("hp", ""),
            enemy.get("atk", ""),
            enemy.get("def", ""),
            enemy.get("money", ""),
            "ok" if {
                k: vis_enemy.get(k)
                for k in ("hp", "atk", "def", "money")
            } == {
                k: enemy.get(k)
                for k in ("hp", "atk", "def", "money")
            } else "DIFF",
        ])
    return rows


def summarize_floors(raw: dict) -> list[list[object]]:
    rows = []
    for floor_id in [f"MT{i}" for i in range(1, 11)]:
        floor = raw["floors"][floor_id]
        events = len(floor.get("events", {}))
        after_battle = len(floor.get("afterBattle", {}))
        ratio = floor.get("ratio", raw.get("values", {}).get("ratio", ""))
        rows.append([
            floor_id,
            f"{floor.get('width')}x{floor.get('height')}",
            ratio,
            floor.get("canFlyFrom"),
            floor.get("canFlyTo"),
            events,
            after_battle,
        ])
    return rows


def summarize_cells(raw: dict, visualizer: dict) -> list[list[object]]:
    counter = Counter()
    for floor_id in [f"MT{i}" for i in range(1, 11)]:
        for row in raw["floors"][floor_id]["map"]:
            counter.update(row)
    visual_counter = Counter(
        cell
        for floor in visualizer["floors"]["map"]
        for row in floor
        for cell in row
    )
    rows = []
    cell_map = getattr(visualizer_db, "_FIRST10_CELL_MAP")
    for cell in sorted(counter):
        raw_meta = raw.get("maps", {}).get(str(cell), {})
        vis_cell = cell_map.get(cell, 0)
        vis_meta = visualizer.get("maps", {}).get(vis_cell, {})
        rows.append([
            cell,
            raw_meta.get("id", ""),
            raw_meta.get("cls", ""),
            counter[cell],
            vis_cell,
            vis_meta.get("id", ""),
            visual_counter[vis_cell],
        ])
    return rows


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_REPORT
    raw = json.loads(DATA_PATH.read_text(encoding="utf8"))
    game_data = load_game_data(DATA_PATH)
    sim = MotaSimulator(game_data)
    simple = sim.reset()
    visualizer = visualizer_db.load_data("10層魔塔")
    hero = raw["firstData"]["hero"]
    source_floors = source_floor_files(raw.get("source_project", ""))
    extracted = sorted(raw["floors"], key=lambda floor_id: int(floor_id[2:]))

    shop_removed = all(
        visualizer["floors"]["map"][3][y][x] == 0
        for y, x in [(1, 5), (1, 6), (1, 7)]
    )
    flyer_removed = all(
        visualizer["maps"].get(cell, {}).get("id") != "centerFly"
        for floor in visualizer["floors"]["map"]
        for row in floor
        for cell in row
    )
    merchant_checks = [
        ("MT6 trader", visualizer["maps"][visualizer["floors"]["map"][5][4][8]]["id"], visualizer["npcs"].get((5, 4, 8))),
        ("MT7 trader", visualizer["maps"][visualizer["floors"]["map"][6][1][6]]["id"], visualizer["npcs"].get((6, 1, 6))),
    ]

    lines: list[str] = []
    lines.extend([
        "# Mota 50 Data Audit",
        "",
        "本报告检查当前项目实际使用的游戏属性、来源和已知简化。结论先放前面：当前数据来自 `50层魔塔` H5 工程，但训练/可视化只抽取并维护 `MT1-MT10` 前十层切片。",
        "",
        "## Source",
        "",
        f"- 数据文件: `{DATA_PATH.relative_to(PROJECT_ROOT)}`",
        f"- H5 标题: `{raw_title(raw)}`",
        f"- source_project: `{raw.get('source_project')}`",
        f"- 原工程楼层文件: `{source_floors[0] if source_floors else 'missing'}` 到 `{source_floors[-1] if source_floors else 'missing'}`，共 `{len(source_floors)}` 个",
        f"- 当前已抽取楼层: `{', '.join(extracted)}`，共 `{len(extracted)}` 个",
        "",
        "## Hero State",
        "",
    ])
    lines.extend(md_table(
        ["环境", "floor", "pos", "HP", "ATK", "DEF", "MDEF", "money", "items/flags"],
        [
            [
                "原始 50层 firstData",
                raw["firstData"].get("floorId"),
                f"{hero['loc']['x']},{hero['loc']['y']}",
                hero.get("hp"),
                hero.get("atk"),
                hero.get("def"),
                hero.get("mdef", 0),
                hero.get("money", 0),
                json.dumps({"items": hero.get("items", {}), "flags": hero.get("flags", {})}, ensure_ascii=False),
            ],
            [
                "当前算法 simple 起点",
                simple.floor_id,
                f"{simple.x},{simple.y}",
                simple.hp,
                simple.atk,
                simple.defense,
                simple.mdef,
                simple.money,
                json.dumps({"items": simple.items, "flags": simple.flags}, ensure_ascii=False),
            ],
            [
                "可视化 reset 起点",
                raw["firstData"].get("floorId"),
                f"{hero['loc']['x']},{hero['loc']['y']}",
                visualizer["player"].get("hp"),
                visualizer["player"].get("atk"),
                visualizer["player"].get("def"),
                visualizer["player"].get("mdef"),
                visualizer["player"].get("money"),
                json.dumps({"items": visualizer["player"].get("items", {})}, ensure_ascii=False),
            ],
        ],
    ))
    lines.extend([
        "",
        "## Floor Metadata",
        "",
    ])
    lines.extend(md_table(
        ["floor", "size", "ratio", "canFlyFrom", "canFlyTo", "events", "afterBattle"],
        summarize_floors(raw),
    ))
    lines.extend([
        "",
        "## Key Items",
        "",
    ])
    lines.extend(md_table(["id", "name", "cls", "current effect/check"], summarize_items(raw)))
    lines.extend([
        "",
        "## Key Enemies",
        "",
    ])
    lines.extend(md_table(["id", "HP", "ATK", "DEF", "money", "visualizer match"], summarize_enemies(raw, visualizer)))
    lines.extend([
        "",
        "## Visualizer Scenario Checks",
        "",
    ])
    lines.extend(md_table(
        ["check", "result"],
        [
            ["4F stat/HP shop physically removed", shop_removed],
            ["fly/centerFly tiles removed from active visualizer map", flyer_removed],
            ["skeletonSoldier spelling normalized", "skeletonSoldier" in visualizer["enemies"] and "skeletonSoilder" not in visualizer["enemies"]],
            ["stale old default NPC commands filtered to actual NPC map positions", all(pos[:3] in {
                (z, y, x)
                for z, floor in enumerate(visualizer["floors"]["map"])
                for y, row in enumerate(floor)
                for x, cell in enumerate(row)
                if visualizer["maps"].get(cell, {}).get("cls") == "npcs"
            } for pos in visualizer["npcs"])],
        ] + [[name, f"{npc_id}: {cmds}"] for name, npc_id, cmds in merchant_checks],
    ))
    lines.extend([
        "",
        "## Used Cell Mapping",
        "",
    ])
    lines.extend(md_table(
        ["raw cell", "raw id", "raw cls", "raw count", "visual cell", "visual id", "active visual count"],
        summarize_cells(raw, visualizer),
    ))
    lines.extend([
        "",
        "## Known Simplifications",
        "",
        "- 当前研究场景不是完整 50 层端到端环境，而是 `50层魔塔` 的前十层切片。",
        "- 算法默认从小偷剧情后开始：`MT2:3,7, HP=400, ATK=10, DEF=10, money=4`；这对应 3F 魔王重置和 2F 小偷剧情被折叠。",
        "- 4F 商店和飞行器在当前场景中被删除/禁用；MT6、MT7 的钥匙商人保留，并按 50 金币门槛检查。",
        "- 10F 机关门、骷髅队长和左上角多余骷髅位置目前是前十层研究目标的显式建模，不是完整事件解释器。",
        "- 可视化器现在使用原始 reset 英雄标量属性，但没有完整表现 `nowWeapon/nowShield/魔法免疫` 这类装备旗标；算法同步起点会覆盖为剧情后标量属性。",
        "",
    ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf8")
    print(out_path)


if __name__ == "__main__":
    main()
