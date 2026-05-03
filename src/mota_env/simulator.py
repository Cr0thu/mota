from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

from .data import GameData


Direction = Literal["up", "down", "left", "right"]

DIRS: dict[Direction, tuple[int, int]] = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


@dataclass
class MotaState:
    floors: dict[str, list[list[int]]]
    floor_id: str
    x: int
    y: int
    hp: int
    atk: int
    defense: int
    mdef: int
    money: int
    exp: int
    items: dict[str, int] = field(default_factory=dict)
    flags: dict[str, Any] = field(default_factory=dict)
    visited_floors: set[str] = field(default_factory=set)
    triggered_events: set[tuple[str, int, int]] = field(default_factory=set)
    steps: int = 0
    dead: bool = False
    done: bool = False
    log: list[str] = field(default_factory=list)

    def clone(self) -> "MotaState":
        return MotaState(
            floors=copy.deepcopy(self.floors),
            floor_id=self.floor_id,
            x=self.x,
            y=self.y,
            hp=self.hp,
            atk=self.atk,
            defense=self.defense,
            mdef=self.mdef,
            money=self.money,
            exp=self.exp,
            items=dict(self.items),
            flags=copy.deepcopy(self.flags),
            visited_floors=set(self.visited_floors),
            triggered_events=set(self.triggered_events),
            steps=self.steps,
            dead=self.dead,
            done=self.done,
            log=list(self.log),
        )


@dataclass(frozen=True)
class Transition:
    ok: bool
    reward: float = 0.0
    message: str = ""


@dataclass(frozen=True)
class SimulatorConfig:
    scenario: Literal["full", "simple"] = "simple"
    enable_shop: bool = False
    enable_fly: bool = False
    first_ten_only: bool = True


class MotaSimulator:
    """Deterministic subset of mota-js mechanics needed for floors MT1-MT10."""

    def __init__(self, data: GameData, config: SimulatorConfig | None = None):
        self.data = data
        self.values = data.values
        self.config = config or SimulatorConfig()
        self.floor_order = [
            floor_id
            for floor_id in data.floor_ids
            if not self.config.first_ten_only or 1 <= int(floor_id[2:]) <= 10
        ]

    def reset(self) -> MotaState:
        if self.config.scenario == "simple":
            return self.reset_simple()
        hero = self.data.first_data["hero"]
        floors = {
            floor_id: copy.deepcopy(self.data.floors[floor_id]["map"])
            for floor_id in self.floor_order
        }
        state = MotaState(
            floors=floors,
            floor_id=self.data.first_data["floorId"],
            x=hero["loc"]["x"],
            y=hero["loc"]["y"],
            hp=hero["hp"],
            atk=hero["atk"],
            defense=hero["def"],
            mdef=hero.get("mdef", 0),
            money=hero.get("money", 0),
            exp=hero.get("exp", 0),
            items={
                "yellowKey": 0,
                "blueKey": 0,
                "redKey": 0,
                "fly": 1,
                "I300": 1,
                "I333": 1,
            },
            flags={
                "开启特性": 0,
                "addhp": 0,
                "额外功能开关": True,
                "fly": False,
                "nowWeapon": hero.get("flags", {}).get("nowWeapon"),
                "nowShield": hero.get("flags", {}).get("nowShield"),
                "魔法免疫": hero.get("flags", {}).get("魔法免疫", True),
            },
            visited_floors={self.data.first_data["floorId"]},
        )
        state.log.append("reset MT1 feature=0 addhp=0")
        return state

    def reset_simple(self) -> MotaState:
        """Start after the MT3 thief plot and remove shop/fly mechanics.

        This scenario is intentionally smaller for algorithm experiments:
        - only MT1-MT10 maps are loaded;
        - the MT3 plot reset has already happened;
        - the MT2 thief has already opened the road;
        - the flyer item and 4F shop blocks are removed;
        - no shop/fly macro actions are generated.
        """

        floors = {
            floor_id: copy.deepcopy(self.data.floors[floor_id]["map"])
            for floor_id in self.floor_order
        }
        state = MotaState(
            floors=floors,
            floor_id="MT2",
            x=3,
            y=7,
            hp=400,
            atk=10,
            defense=10,
            mdef=0,
            money=4,
            exp=0,
            items={
                "yellowKey": 0,
                "blueKey": 0,
                "redKey": 0,
                "fly": 0,
            },
            flags={
                "开启特性": 0,
                "addhp": 0,
                "额外功能开关": True,
                "fly": False,
                "nowWeapon": None,
                "nowShield": None,
                "魔法免疫": False,
                "03": 1,
                "simple": True,
            },
            visited_floors={"MT2"},
            triggered_events={("MT3", 5, 9), ("MT2", 3, 7)},
        )
        for px, py in [(5, 7), (5, 8), (4, 9), (6, 9), (5, 10), (5, 9)]:
            self.set_tile(state, px, py, 0, "MT3")
        self.set_tile(state, 2, 7, 0, "MT2")
        self.set_tile(state, 3, 7, 0, "MT2")
        self.set_tile(state, 2, 11, 0, "MT1")
        for px, py in [(5, 1), (6, 1), (7, 1)]:
            self.set_tile(state, px, py, 0, "MT4")
        state.log.append("reset simple post-thief no-shop no-fly money=4")
        return state

    def tile(self, state: MotaState, x: int, y: int, floor_id: str | None = None) -> int:
        floor = state.floors[floor_id or state.floor_id]
        if y < 0 or y >= len(floor) or x < 0 or x >= len(floor[y]):
            return 1
        return floor[y][x]

    def set_tile(
        self, state: MotaState, x: int, y: int, tile: int | str, floor_id: str | None = None
    ) -> None:
        if isinstance(tile, str):
            tile = self.tile_number(tile)
        state.floors[floor_id or state.floor_id][y][x] = int(tile)

    def tile_number(self, block_id: str) -> int:
        for number, info in self.data.maps.items():
            if info.get("id") == block_id:
                return int(number)
        if block_id == "0":
            return 0
        raise KeyError(block_id)

    def block_info(self, tile: int) -> dict[str, Any]:
        return self.data.maps.get(str(tile), {})

    def block_id(self, tile: int) -> str | None:
        return self.block_info(tile).get("id")

    def block_cls(self, tile: int) -> str | None:
        return self.block_info(tile).get("cls")

    def is_enemy_tile(self, tile: int) -> bool:
        return self.block_cls(tile) == "enemys"

    def is_item_tile(self, tile: int) -> bool:
        return self.block_cls(tile) == "items"

    def is_door_tile(self, tile: int) -> bool:
        return self.block_info(tile).get("trigger") == "openDoor"

    def is_stair_tile(self, tile: int) -> bool:
        return self.block_id(tile) in {"upFloor", "downFloor"}

    def is_wall_tile(self, tile: int) -> bool:
        if tile == 0:
            return False
        info = self.block_info(tile)
        if self.is_enemy_tile(tile) or self.is_item_tile(tile) or self.is_door_tile(tile):
            return False
        return info.get("noPass") is True or info.get("noPass") == "true" or tile == 1

    def can_transit(self, state: MotaState, x: int, y: int) -> bool:
        tile = self.tile(state, x, y)
        if self.is_wall_tile(tile) or self.is_door_tile(tile) or self.is_enemy_tile(tile):
            return False
        return True

    def can_open_door(self, state: MotaState, tile: int) -> bool:
        keys = self.block_info(tile).get("doorInfo", {}).get("keys", {})
        if not keys:
            return True
        return all(state.items.get(key, 0) >= need for key, need in keys.items())

    def open_door(self, state: MotaState, x: int, y: int, consume_key: bool = True) -> bool:
        tile = self.tile(state, x, y)
        if not self.is_door_tile(tile):
            if consume_key:
                return False
            self.set_tile(state, x, y, 0)
            state.log.append(f"event clears block {tile} at {state.floor_id}:{x},{y}")
            return True
        keys = self.block_info(tile).get("doorInfo", {}).get("keys", {})
        if consume_key:
            if not self.can_open_door(state, tile):
                return False
            for key, need in keys.items():
                state.items[key] = state.items.get(key, 0) - int(need)
        self.set_tile(state, x, y, 0)
        state.log.append(f"open door {self.block_id(tile)} at {state.floor_id}:{x},{y}")
        return True

    def close_special_door(self, state: MotaState, x: int, y: int) -> None:
        self.set_tile(state, x, y, "specialDoor")
        state.log.append(f"close specialDoor at {state.floor_id}:{x},{y}")

    def damage_info(self, state: MotaState, enemy_id: str) -> dict[str, int] | None:
        enemy = self.data.enemys[enemy_id]
        mon_hp = int(enemy.get("hp", 0))
        mon_atk = int(enemy.get("atk", 0))
        mon_def = int(enemy.get("def", 0))
        special = enemy.get("special", 0)
        specials = set(special if isinstance(special, list) else [special])

        hero_per_damage = max(state.atk - mon_def, 0)
        if hero_per_damage <= 0:
            return None
        turn = math.ceil(mon_hp / hero_per_damage)
        per_damage = mon_atk if 2 in specials else max(mon_atk - state.defense, 0)
        if 4 in specials:
            per_damage *= 2
        if 5 in specials:
            per_damage *= 3
        if 6 in specials:
            per_damage *= int(enemy.get("n", 4) or 4)

        init_damage = 0
        if 1 in specials:
            init_damage += per_damage
        if 7 in specials:
            init_damage += math.floor(float(enemy.get("defValue") or self.values["breakArmor"]) * state.defense)
        if 9 in specials:
            init_damage += math.floor(float(enemy.get("n") or self.values["purify"]) * state.mdef)
        counter = 0
        if 8 in specials:
            counter += math.floor(float(enemy.get("atkValue") or self.values["counterAttack"]) * state.atk)

        damage = init_damage + (turn - 1) * per_damage + turn * counter - state.mdef
        damage = max(0, int(damage))
        return {"damage": damage, "turn": turn}

    def can_battle(self, state: MotaState, tile: int) -> bool:
        enemy_id = self.block_id(tile)
        info = self.damage_info(state, enemy_id) if enemy_id else None
        return info is not None and state.hp > info["damage"]

    def battle(self, state: MotaState, x: int, y: int) -> bool:
        tile = self.tile(state, x, y)
        enemy_id = self.block_id(tile)
        if not enemy_id:
            return False
        info = self.damage_info(state, enemy_id)
        if info is None or state.hp <= info["damage"]:
            state.dead = True
            state.log.append(f"dead fighting {enemy_id} at {state.floor_id}:{x},{y}")
            return False
        state.hp -= info["damage"]
        enemy = self.data.enemys[enemy_id]
        gain_money = int(enemy.get("money", 0))
        if state.items.get("coin", 0) > 0:
            gain_money *= 2
        state.money += gain_money
        state.exp += int(enemy.get("experience", 0))
        self.set_tile(state, x, y, 0)
        state.log.append(
            f"battle {enemy_id} at {state.floor_id}:{x},{y} damage={info['damage']} hp={state.hp}"
        )
        self.after_battle(state, x, y, enemy_id)
        self.check_auto_events(state)
        return True

    def apply_item(self, state: MotaState, item_id: str) -> None:
        ratio = int(self.data.floors[state.floor_id].get("ratio", 1) or 1)
        if item_id in {"yellowKey", "blueKey", "redKey"}:
            state.items[item_id] = state.items.get(item_id, 0) + 1
        elif item_id == "redGem":
            state.atk += int(self.values["redGem"]) * ratio
        elif item_id == "blueGem":
            state.defense += int(self.values["blueGem"]) * ratio
        elif item_id == "greenGem":
            state.mdef += int(self.values["greenGem"]) * ratio
        elif item_id == "redPotion":
            state.hp += int(self.values["redPotion"]) * ratio
        elif item_id == "bluePotion":
            state.hp += int(self.values["bluePotion"]) * ratio
        elif item_id == "yellowPotion":
            state.hp += int(self.values["yellowPotion"]) * ratio
        elif item_id == "greenPotion":
            state.hp += int(self.values["greenPotion"]) * ratio
        elif item_id == "sword1":
            state.atk += 10
            state.flags["nowWeapon"] = "sword1"
        elif item_id == "sword2":
            state.atk += 20
            state.flags["nowWeapon"] = "sword2"
        elif item_id == "shield1":
            state.defense += 10
            state.flags["nowShield"] = "shield1"
        elif item_id == "shield2":
            state.defense += 20
            state.flags["nowShield"] = "shield2"
        elif item_id == "fly" and not self.config.enable_fly:
            pass
        else:
            state.items[item_id] = state.items.get(item_id, 0) + 1
        state.log.append(f"item {item_id} hp={state.hp} atk={state.atk} def={state.defense}")

    def get_item(self, state: MotaState, x: int, y: int) -> None:
        tile = self.tile(state, x, y)
        item_id = self.block_id(tile)
        if item_id is None:
            return
        self.apply_item(state, item_id)
        self.set_tile(state, x, y, 0)
        if state.floor_id == "MT1" and (x, y) == (2, 11):
            state.flags["fly"] = True

    def step(self, state: MotaState, direction: Direction) -> Transition:
        if state.dead or state.done:
            return Transition(False, -1.0, "terminal")
        dx, dy = DIRS[direction]
        nx, ny = state.x + dx, state.y + dy
        tile = self.tile(state, nx, ny)
        reward = -0.001
        if self.is_wall_tile(tile):
            return Transition(False, -0.05, "wall")
        if self.is_door_tile(tile):
            if not self.open_door(state, nx, ny, consume_key=True):
                return Transition(False, -0.05, "locked")
            reward += 0.02
        elif self.is_enemy_tile(tile):
            if not self.can_battle(state, tile):
                return Transition(False, -0.1, "unwinnable enemy")
            self.battle(state, nx, ny)
            reward += 0.03
        elif self.is_item_tile(tile):
            self.get_item(state, nx, ny)
            reward += 0.02

        state.x, state.y = nx, ny
        state.steps += 1
        self.trigger_event_at(state, nx, ny)
        self.change_floor_if_needed(state)
        self.check_auto_events(state)
        if state.flags.get("10f战胜骷髅队长"):
            state.done = True
            reward += 10.0
        if state.dead:
            reward -= 10.0
        return Transition(True, reward)

    def execute_path(self, state: MotaState, path: Iterable[Direction]) -> Transition:
        total = 0.0
        last = Transition(True)
        for direction in path:
            last = self.step(state, direction)
            total += last.reward
            if not last.ok or state.dead or state.done:
                return Transition(last.ok, total, last.message)
        return Transition(True, total, last.message)

    def change_floor_if_needed(self, state: MotaState) -> None:
        floor = self.data.floors[state.floor_id]
        spec = floor.get("changeFloor", {}).get(f"{state.x},{state.y}")
        if not spec:
            return
        dest = spec.get("floorId")
        if dest == ":next":
            idx = self.floor_order.index(state.floor_id)
            dest = self.floor_order[idx + 1]
        elif dest == ":before":
            idx = self.floor_order.index(state.floor_id)
            if idx == 0:
                return
            dest = self.floor_order[idx - 1]
        if "loc" in spec:
            nx, ny = spec["loc"]
        else:
            stair = spec.get("stair")
            nx, ny = self.data.floors[dest][stair]
        state.floor_id = dest
        state.x, state.y = int(nx), int(ny)
        state.visited_floors.add(dest)
        state.log.append(f"change floor to {dest}:{state.x},{state.y}")

    def trigger_event_at(self, state: MotaState, x: int, y: int) -> None:
        key = (state.floor_id, x, y)
        if key in state.triggered_events:
            return
        if state.floor_id == "MT2" and (x, y) == (3, 7):
            self.open_door(state, 2, 7, consume_key=False)
            self.set_tile(state, 3, 7, 0)
            state.triggered_events.add(key)
        elif state.floor_id == "MT2" and (x, y) == (11, 7):
            state.atk += round(state.atk * 0.03)
            state.defense += round(state.defense * 0.03)
            self.set_tile(state, 11, 7, 0)
            state.triggered_events.add(key)
            state.log.append("special trader blessing +3% atk/def")
        elif state.floor_id == "MT2" and (x, y) in {(1, 9), (10, 11)}:
            self.set_tile(state, x, y, 0)
            state.triggered_events.add(key)
        elif state.floor_id == "MT3" and (x, y) == (5, 9):
            state.hp, state.atk, state.defense = 400, 10, 10
            state.mdef = 0
            state.flags["03"] = 1
            state.flags["nowWeapon"] = None
            state.flags["nowShield"] = None
            state.flags["魔法免疫"] = False
            for px, py in [(5, 7), (5, 8), (4, 9), (6, 9), (5, 10), (5, 9)]:
                self.set_tile(state, px, py, 0, "MT3")
            state.floor_id, state.x, state.y = "MT2", 3, 8
            state.visited_floors.add("MT2")
            state.triggered_events.add(key)
            state.log.append("MT3 plot reset to MT2:3,8 hp=400 atk=10 def=10")
        elif state.floor_id == "MT10" and (x, y) == (6, 5):
            self.trigger_mt10_trap(state)
            state.triggered_events.add(key)

    def trigger_mt10_trap(self, state: MotaState) -> None:
        if state.flags.get("10f机关"):
            return
        for px, py in [(5, 4), (6, 3), (7, 4), (5, 5), (7, 5)]:
            self.set_tile(state, px, py, 0)
        placements = {
            (6, 1): "skeletonCaptain",
            (6, 6): "skeletonSoldier",
            (6, 4): "skeletonSoldier",
            (5, 6): "skeleton",
            (7, 6): "skeleton",
            (5, 5): "skeleton",
            (7, 5): "skeleton",
            (5, 4): "skeleton",
            (7, 4): "skeleton",
        }
        for (px, py), enemy_id in placements.items():
            self.set_tile(state, px, py, enemy_id)
        self.close_special_door(state, 6, 3)
        self.close_special_door(state, 4, 4)
        self.close_special_door(state, 8, 4)
        state.flags["10f机关"] = True
        self.set_tile(state, 6, 5, 0)
        state.log.append("MT10 trap triggered")

    def after_battle(self, state: MotaState, x: int, y: int, enemy_id: str) -> None:
        if state.floor_id == "MT2" and (x, y) in {(6, 2), (8, 2)}:
            if state.flags.get("2", 0):
                for px, py in [(5, 5), (5, 8), (5, 11), (9, 5), (9, 8), (9, 11)]:
                    self.open_door(state, px, py, consume_key=False)
            else:
                state.flags["2"] = 1
        if state.floor_id == "MT8" and (x, y) in {(9, 5), (11, 5)}:
            if state.flags.get("8", 0):
                self.open_door(state, 10, 4, consume_key=False)
            else:
                state.flags["8"] = 1
        if state.floor_id == "MT10" and (x, y) == (6, 1) and enemy_id == "skeletonCaptain":
            self.resolve_mt10_boss(state)

    def resolve_mt10_boss(self, state: MotaState) -> None:
        for px, py, tile in [
            (1, 3, 27),
            (2, 3, 27),
            (3, 3, 27),
            (9, 3, 28),
            (10, 3, 28),
            (11, 3, 28),
            (1, 4, 32),
            (2, 4, 32),
            (3, 4, 32),
            (9, 4, 21),
            (10, 4, 21),
            (11, 4, 21),
        ]:
            self.set_tile(state, px, py, tile)
        for px, py in [(4, 4), (6, 7), (8, 4)]:
            self.open_door(state, px, py, consume_key=False)
        self.set_tile(state, 6, 9, 0)
        state.flags["10f战胜骷髅队长"] = True
        state.done = True
        state.log.append("defeated skeletonCaptain on MT10")

    def check_auto_events(self, state: MotaState) -> None:
        if state.floor_id != "MT10" or not state.flags.get("10f机关"):
            return
        guard_cells = [(5, 4), (6, 4), (7, 4), (5, 5), (7, 5), (5, 6), (6, 6), (7, 6)]
        if all(self.tile(state, x, y) == 0 for x, y in guard_cells):
            if self.tile(state, 6, 3) != 0:
                self.open_door(state, 6, 3, consume_key=False)

    def shop_cost(self, state: MotaState) -> int:
        times = int(state.flags.get("times1", 0) or 0)
        return 20 + 10 * (times + 1) * times

    def shop_buy(self, state: MotaState, kind: Literal["hp", "atk", "def"]) -> bool:
        if not self.config.enable_shop:
            return False
        if state.floor_id != "MT4":
            return False
        cost = self.shop_cost(state)
        if state.money < cost:
            return False
        state.money -= cost
        times = int(state.flags.get("times1", 0) or 0)
        ratio = int(state.flags.get("ratio", 1) or 1)
        if kind == "hp":
            state.hp += 100 * (times + 1)
        elif kind == "atk":
            state.atk += 2 * ratio
        elif kind == "def":
            state.defense += 4 * ratio
        state.flags["times1"] = times + 1
        state.log.append(f"shop {kind} cost={cost} hp={state.hp} atk={state.atk} def={state.defense}")
        return True

    def reachable_cells(self, state: MotaState) -> dict[tuple[int, int], list[Direction]]:
        start = (state.x, state.y)
        queue = deque([start])
        paths: dict[tuple[int, int], list[Direction]] = {start: []}
        while queue:
            x, y = queue.popleft()
            for direction, (dx, dy) in DIRS.items():
                nx, ny = x + dx, y + dy
                if (nx, ny) in paths:
                    continue
                tile = self.tile(state, nx, ny)
                if self.is_wall_tile(tile) or self.is_door_tile(tile) or self.is_enemy_tile(tile):
                    continue
                paths[(nx, ny)] = paths[(x, y)] + [direction]
                # Treat stairs as terminal targets; moving through them changes floor.
                if not self.is_stair_tile(tile):
                    queue.append((nx, ny))
        return paths

    def macro_actions(self, state: MotaState) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        reachable = self.reachable_cells(state)
        for (x, y), path in reachable.items():
            tile = self.tile(state, x, y)
            if (x, y) != (state.x, state.y) and (
                self.is_item_tile(tile)
                or self.is_stair_tile(tile)
                or (
                    f"{x},{y}" in self.data.floors[state.floor_id].get("events", {})
                    and (state.floor_id, x, y) not in state.triggered_events
                )
            ):
                actions.append(
                    {
                        "kind": "move",
                        "target": [state.floor_id, x, y],
                        "path": path,
                        "label": f"go {self.block_id(tile) or 'event'} {state.floor_id}:{x},{y}",
                    }
                )
            for direction, (dx, dy) in DIRS.items():
                nx, ny = x + dx, y + dy
                ntile = self.tile(state, nx, ny)
                if self.is_door_tile(ntile) and self.can_open_door(state, ntile):
                    actions.append(
                        {
                            "kind": "move",
                            "target": [state.floor_id, nx, ny],
                            "path": path + [direction],
                            "label": f"open {self.block_id(ntile)} {state.floor_id}:{nx},{ny}",
                        }
                    )
                elif self.is_enemy_tile(ntile) and self.can_battle(state, ntile):
                    actions.append(
                        {
                            "kind": "move",
                            "target": [state.floor_id, nx, ny],
                            "path": path + [direction],
                            "label": f"fight {self.block_id(ntile)} {state.floor_id}:{nx},{ny}",
                        }
                    )
        if (
            self.config.enable_shop
            and state.floor_id == "MT4"
            and any(pos in reachable for pos in [(6, 1), (5, 1), (7, 1)])
        ):
            for kind in ("atk", "def", "hp"):
                if state.money >= self.shop_cost(state):
                    actions.append({"kind": "shop", "shop": kind, "path": [], "label": f"shop {kind}"})
        if self.config.enable_fly and state.items.get("fly", 0) > 0:
            for floor_id in sorted(state.visited_floors, key=lambda item: int(item[2:])):
                if floor_id == state.floor_id:
                    continue
                floor = self.data.floors[floor_id]
                loc = floor.get("upFloor") or floor.get("downFloor")
                if loc:
                    actions.append(
                        {
                            "kind": "fly",
                            "floor": floor_id,
                            "loc": loc,
                            "path": [],
                            "label": f"fly {floor_id}:{loc[0]},{loc[1]}",
                        }
                    )
            if "MT4" in state.visited_floors and state.money >= self.shop_cost(state):
                for kind in ("atk", "def", "hp"):
                    actions.append(
                        {
                            "kind": "fly_shop",
                            "floor": "MT4",
                            "loc": [6, 1],
                            "shop": kind,
                            "path": [],
                            "label": f"fly shop {kind}",
                        }
                    )
        # Remove duplicate path/target actions while preserving order.
        seen: set[tuple[Any, ...]] = set()
        unique = []
        for action in actions:
            key = (
                action["kind"],
                tuple(action.get("target", [])),
                action.get("shop"),
                tuple(action.get("path", [])),
            )
            if key not in seen:
                seen.add(key)
                unique.append(action)
        return unique

    def apply_macro_action(self, state: MotaState, action: dict[str, Any]) -> Transition:
        if action["kind"] == "shop":
            ok = self.shop_buy(state, action["shop"])
            return Transition(ok, 0.03 if ok else -0.05, "shop" if ok else "shop failed")
        if action["kind"] == "fly":
            state.floor_id = action["floor"]
            state.x, state.y = map(int, action["loc"])
            state.visited_floors.add(state.floor_id)
            state.log.append(f"fly to {state.floor_id}:{state.x},{state.y}")
            return Transition(True, -0.005, "fly")
        if action["kind"] == "fly_shop":
            state.floor_id = action["floor"]
            state.x, state.y = map(int, action["loc"])
            state.visited_floors.add(state.floor_id)
            ok = self.shop_buy(state, action["shop"])
            return Transition(ok, 0.02 if ok else -0.05, "fly_shop")
        return self.execute_path(state, action["path"])

    def state_key(self, state: MotaState) -> tuple[Any, ...]:
        floors_compact = tuple(
            (floor_id, tuple(tuple(row) for row in state.floors[floor_id]))
            for floor_id in self.floor_order
            if floor_id in state.floors
        )
        return (
            state.floor_id,
            state.x,
            state.y,
            state.hp,
            state.atk,
            state.defense,
            state.mdef,
            state.money,
            state.exp,
            tuple(sorted((k, v) for k, v in state.items.items() if v)),
            tuple(sorted((k, str(v)) for k, v in state.flags.items() if v)),
            tuple(sorted(state.triggered_events)),
            floors_compact,
        )

    def dominance_key(self, state: MotaState) -> tuple[Any, ...]:
        floors_compact = tuple(
            (floor_id, tuple(tuple(row) for row in state.floors[floor_id]))
            for floor_id in self.floor_order
            if floor_id in state.floors
        )
        structural_flags = {
            key: value
            for key, value in state.flags.items()
            if key not in {"times1"} and not str(key).endswith("购买hp") and not str(key).endswith("购买atk")
        }
        return (
            state.floor_id,
            state.x,
            state.y,
            tuple(sorted((k, str(v)) for k, v in structural_flags.items() if v)),
            tuple(sorted(state.triggered_events)),
            floors_compact,
        )

    def resource_vector(self, state: MotaState) -> tuple[int, ...]:
        return (
            state.hp,
            state.atk,
            state.defense,
            state.mdef,
            state.money,
            state.exp,
            state.items.get("yellowKey", 0),
            state.items.get("blueKey", 0),
            state.items.get("redKey", 0),
            -int(state.flags.get("times1", 0) or 0),
        )
