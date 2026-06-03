# Mota 50 Data Audit

本报告检查当前项目实际使用的游戏属性、来源和已知简化。结论先放前面：当前数据来自 `50层魔塔` H5 工程，但训练/可视化只抽取并维护 `MT1-MT10` 前十层切片。

## Source

- 数据文件: `artifacts/data/mota_first10.json`
- H5 标题: `50层魔塔`
- source_project: `/Users/cr0/Documents/项目/mota/game/Falsh原版魔塔合集/51_2/project`
- 原工程楼层文件: `MT0` 到 `MT50`，共 `51` 个
- 当前已抽取楼层: `MT0, MT1, MT2, MT3, MT4, MT5, MT6, MT7, MT8, MT9, MT10`，共 `11` 个

## Hero State

| 环境 | floor | pos | HP | ATK | DEF | MDEF | money | items/flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 原始 50层 firstData | MT1 | 6,11 | 1000 | 100 | 100 | 0 | 0 | {"items": {"constants": {"I300": 1}, "tools": {}, "equips": {}}, "flags": {"nowWeapon": "sword5", "nowShield": "shield5", "魔法免疫": true, "__winskin_opacity__": 1}} |
| 当前算法 simple 起点 | MT2 | 3,7 | 400 | 10 | 10 | 0 | 4 | {"items": {"yellowKey": 0, "blueKey": 0, "redKey": 0, "fly": 0}, "flags": {"开启特性": 0, "addhp": 0, "额外功能开关": true, "fly": false, "nowWeapon": null, "nowShield": null, "魔法免疫": false, "simple": true, "03": 1}} |
| 可视化 reset 起点 | MT1 | 6,11 | 1000 | 100 | 100 | 0 | 0 | {"items": {"yellowKey": 0, "blueKey": 0, "redKey": 0, "I300": 1}} |

## Floor Metadata

| floor | size | ratio | canFlyFrom | canFlyTo | events | afterBattle |
| --- | --- | --- | --- | --- | --- | --- |
| MT1 | 13x13 | 1 | True | True | 1 | 0 |
| MT2 | 13x13 | 1 | True | True | 4 | 2 |
| MT3 | 13x13 | 1 | True | True | 1 | 0 |
| MT4 | 13x13 | 1 | True | True | 1 | 0 |
| MT5 | 13x13 | 1 | True | True | 0 | 0 |
| MT6 | 13x13 | 1 | True | True | 0 | 0 |
| MT7 | 13x13 | 1 | True | True | 0 | 0 |
| MT8 | 13x13 | 1 | True | True | 0 | 2 |
| MT9 | 13x13 | 1 | True | True | 0 | 0 |
| MT10 | 13x13 | 1 | True | True | 5 | 1 |

## Key Items

| id | name | cls | current effect/check |
| --- | --- | --- | --- |
| redGem | 红宝石 | items | +ATK 1 |
| blueGem | 蓝宝石 | items | +DEF 1 |
| redPotion | 红血瓶 | items | +HP 50 |
| bluePotion | 蓝血瓶 | items | +HP 200 |
| yellowKey | 黄钥匙 | tools | key +1 |
| blueKey | 兰钥匙 | tools | key +1 |
| redKey | 红钥匙 | tools | key +1 |
| sword1 | 铁剑 | items | +ATK 10 |
| shield1 | 铁盾 | items | +DEF 10 |
| fly | 魔杖 | constants | 楼层传送器；当前研究场景删除/禁用 |
| centerFly | 瞬移 | tools | 中心对称飞行器；当前前十层场景不使用 |

## Key Enemies

| id | HP | ATK | DEF | money | visualizer match |
| --- | --- | --- | --- | --- | --- |
| greenSlime | 35 | 18 | 1 | 1 | ok |
| redSlime | 45 | 20 | 2 | 2 | ok |
| bat | 35 | 38 | 3 | 3 | ok |
| skeleton | 50 | 42 | 6 | 6 | ok |
| skeletonSoldier | 55 | 52 | 12 | 8 | ok |
| skeletonCaptain | 100 | 65 | 15 | 30 | ok |
| bluePriest | 60 | 32 | 8 | 5 | ok |
| yellowGuard | 50 | 48 | 22 | 12 | ok |
| blueGuard | 100 | 180 | 110 | 50 | ok |

## Visualizer Scenario Checks

| check | result |
| --- | --- |
| 4F stat/HP shop physically removed | True |
| fly/centerFly tiles removed from active visualizer map | True |
| skeletonSoldier spelling normalized | True |
| stale old default NPC commands filtered to actual NPC map positions | True |
| MT6 trader | redMan: [{'type': 'if', 'condition': 'player.money >= 50'}, {'type': 'addValue', 'name': 'player.money', 'value': -50}, {'type': 'addItem', 'name': 'blueKey', 'value': 1}] |
| MT7 trader | redMan: [{'type': 'if', 'condition': 'player.money >= 50'}, {'type': 'addValue', 'name': 'player.money', 'value': -50}, {'type': 'addItem', 'name': 'yellowKey', 'value': 5}] |

## Used Cell Mapping

| raw cell | raw id | raw cls | raw count | visual cell | visual id | active visual count |
| --- | --- | --- | --- | --- | --- | --- |
| 0 |  |  | 442 | 0 | background | 451 |
| 1 | yellowWall | animates | 897 | 1 | yellowWall | 897 |
| 2 | fakeWall | animates | 2 | 0 | background | 451 |
| 7 | blueShop-left | terrains | 1 | 160 | blueShop_l | 0 |
| 8 | blueShop-right | terrains | 1 | 161 | blueShop_r | 0 |
| 17 |  |  | 5 | 17 | dynamicBlock | 0 |
| 21 | yellowKey | items | 56 | 32 | yellowKey | 56 |
| 22 | blueKey | items | 3 | 33 | blueKey | 3 |
| 23 | redKey | items | 1 | 34 | redKey | 1 |
| 27 | redGem | items | 7 | 20 | redJewel | 7 |
| 28 | blueGem | items | 7 | 21 | blueJewel | 7 |
| 31 | redPotion | items | 22 | 16 | redPotion | 22 |
| 32 | bluePotion | items | 12 | 18 | bluePotion | 12 |
| 35 | sword1 | items | 1 | 80 | sword1 | 1 |
| 36 | shield1 | items | 1 | 81 | shield1 | 1 |
| 46 | fly | items | 1 | 50 | centerFly | 0 |
| 73 | wand | items | 1 | 64 | redWand | 1 |
| 81 | yellowDoor | animates | 65 | 36 | yellowDoor | 65 |
| 82 | blueDoor | animates | 6 | 37 | blueDoor | 6 |
| 83 | redDoor | animates | 1 | 38 | redDoor | 1 |
| 85 | specialDoor | animates | 3 | 40 | specialDoor | 2 |
| 86 | steelDoor | animates | 6 | 41 | steelDoor | 8 |
| 87 | upFloor | terrains | 10 | 42 | upFloor | 9 |
| 88 | downFloor | terrains | 9 | 43 | downFloor | 9 |
| 121 | oldman | npcs | 4 | 96 | blueMan | 5 |
| 122 | trader | npcs | 2 | 97 | redMan | 2 |
| 123 | thief | npcs | 2 | 98 | thief | 2 |
| 124 | specialTrader | npcs | 1 | 96 | blueMan | 5 |
| 127 | king | npcs | 1 | 102 | princess | 1 |
| 131 | blueShop | npcs | 1 | 101 | blueShop | 0 |
| 201 | greenSlime | enemys | 24 | 112 | greenSlime | 24 |
| 202 | redSlime | enemys | 18 | 113 | redSlime | 18 |
| 205 | bat | enemys | 18 | 116 | bat | 18 |
| 209 | skeleton | enemys | 17 | 120 | skeleton | 22 |
| 210 | skeletonSoldier | enemys | 15 | 121 | skeletonSoldier | 16 |
| 211 | skeletonCaptain | enemys | 1 | 122 | skeletonCaptain | 1 |
| 217 | bluePriest | enemys | 17 | 128 | bluePriest | 17 |
| 221 | yellowGuard | enemys | 2 | 132 | yellowGuard | 2 |
| 222 | blueGuard | enemys | 2 | 133 | blueGuard | 2 |
| 321 | whiteWall2 | terrains | 1 | 0 | background | 451 |
| 330 | unbreakableWall | animates | 4 | 41 | steelDoor | 8 |

## Known Simplifications

- 当前研究场景不是完整 50 层端到端环境，而是 `50层魔塔` 的前十层切片。
- 算法默认从小偷剧情后开始：`MT2:3,7, HP=400, ATK=10, DEF=10, money=4`；这对应 3F 魔王重置和 2F 小偷剧情被折叠。
- 4F 商店和飞行器在当前场景中被删除/禁用；MT6、MT7 的钥匙商人保留，并按 50 金币门槛检查。
- 10F 机关门、骷髅队长和左上角多余骷髅位置目前是前十层研究目标的显式建模，不是完整事件解释器。
- 可视化器现在使用原始 reset 英雄标量属性，但没有完整表现 `nowWeapon/nowShield/魔法免疫` 这类装备旗标；算法同步起点会覆盖为剧情后标量属性。
