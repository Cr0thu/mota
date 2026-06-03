from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .simulator import MotaSimulator, MotaState


MAX_GRAPH_NODES = 256

GRAPH_NODE_TYPES = {
    "pad": 0,
    "hero": 1,
    "item": 2,
    "enemy": 3,
    "door": 4,
    "npc": 5,
    "stair": 6,
    "event": 7,
    "boss": 8,
    "global": 9,
    "action": 10,
}

GRAPH_FEATURE_NAMES = (
    "floor_norm",
    "x_norm",
    "y_norm",
    "type_norm",
    "consumed",
    "reachable",
    "executable",
    "path_len_norm",
    "hp_norm",
    "atk_norm",
    "def_norm",
    "money_norm",
    "yellow_key_norm",
    "blue_key_norm",
    "red_key_norm",
    "item_value_norm",
    "enemy_hp_norm",
    "enemy_atk_norm",
    "enemy_def_norm",
    "enemy_damage_norm",
    "enemy_hp_margin_norm",
    "enemy_killable",
    "damage_drop_atk1_norm",
    "damage_drop_def1_norm",
    "unlock_value_norm",
    "door_openable",
    "missing_yellow_norm",
    "missing_blue_norm",
    "missing_red_norm",
    "required_yellow_norm",
    "required_blue_norm",
    "required_red_norm",
    "stage_norm",
    "boss_damage_norm",
    "boss_margin_norm",
    "remaining_attack_gain_norm",
    "remaining_defense_gain_norm",
    "mt10_resource_remaining_norm",
    "is_mt10",
    "reserved",
)
GRAPH_NODE_FEATURE_DIM = len(GRAPH_FEATURE_NAMES)

KEY_ITEMS = ("yellowKey", "blueKey", "redKey")
RESOURCE_ITEMS = KEY_ITEMS + (
    "redGem",
    "blueGem",
    "greenGem",
    "redPotion",
    "bluePotion",
    "yellowPotion",
    "greenPotion",
    "sword1",
    "shield1",
)


@dataclass
class GraphNode:
    node_id: str
    kind: str
    floor: str
    x: int
    y: int
    block_id: str | None = None
    raw_id: str | None = None
    consumed: bool = False
    reachable: bool = False
    executable: bool = False
    path_len: int | None = None
    action_index: int | None = None
    action_label: str | None = None
    missing_keys: dict[str, int] = field(default_factory=dict)
    required_keys: dict[str, int] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "floor": self.floor,
            "x": self.x,
            "y": self.y,
            "block_id": self.block_id,
            "raw_id": self.raw_id,
            "consumed": self.consumed,
            "reachable": self.reachable,
            "executable": self.executable,
            "path_len": self.path_len,
            "action_index": self.action_index,
            "action_label": self.action_label,
            "missing_keys": self.missing_keys,
            "required_keys": self.required_keys,
            "features": self.features,
        }


class GraphStateBuilder:
    """Build a fixed-size global interaction graph for policy/value search.

    Nodes are all meaningful cells in the first-10-floor scenario, not only the
    currently reachable macro actions. The executable mask is still generated
    from the simulator's legal macro actions, so AlphaZero-style models can
    score all nodes while only executing reachable/legal choices.
    """

    def __init__(
        self,
        sim: MotaSimulator,
        max_nodes: int = MAX_GRAPH_NODES,
        include_unlock_values: bool = True,
    ):
        self.sim = sim
        self.max_nodes = int(max_nodes)
        self.include_unlock_values = bool(include_unlock_values)

    def build(
        self,
        state: MotaState,
        actions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        actions = actions if actions is not None else self.sim.macro_actions(state)
        reachable_parents = self.sim.reachable_parent_map(state)
        reachable = set(reachable_parents)
        distance_by_cell = self._distance_by_cell(reachable_parents)
        action_lookup = self._action_lookup(actions)

        nodes: list[GraphNode] = [
            GraphNode(
                node_id="hero",
                kind="hero",
                floor=state.floor_id,
                x=state.x,
                y=state.y,
                reachable=True,
                executable=False,
                path_len=0,
            )
        ]
        seen: set[str] = {"hero"}

        for floor_id in self.sim.floor_order:
            if floor_id not in state.floors:
                continue
            for node in self._floor_nodes(state, floor_id, reachable, distance_by_cell, action_lookup):
                if node.node_id not in seen:
                    nodes.append(node)
                    seen.add(node.node_id)

        for index, action in enumerate(actions):
            target_key = self._target_key(action)
            if target_key is None or target_key not in action_lookup:
                continue
            node_index = self._find_node_index(nodes, action_lookup[target_key]["node_id"])
            if node_index is None:
                node = self._synthetic_action_node(state, action, index)
                if node.node_id not in seen:
                    nodes.append(node)
                    seen.add(node.node_id)

        nodes.append(
            GraphNode(
                node_id="global",
                kind="global",
                floor=state.floor_id,
                x=state.x,
                y=state.y,
                reachable=True,
                executable=False,
                path_len=0,
            )
        )

        nodes = self._sorted_nodes(nodes)[: self.max_nodes]
        for node in nodes:
            self._attach_node_features(state, node)
        self._attach_action_indices(nodes, actions)
        feature_context = self._feature_context(state)
        feature_rows = [self._node_feature_vector(state, node, feature_context) for node in nodes]
        node_type_ids = [GRAPH_NODE_TYPES.get(node.kind, GRAPH_NODE_TYPES["action"]) for node in nodes]
        node_mask = [True] * len(nodes)
        executable_mask = [node.executable for node in nodes]
        action_node_indices = [
            index
            for index, node in enumerate(nodes)
            if node.action_index is not None and node.executable
        ]
        action_labels = [nodes[index].action_label or "" for index in action_node_indices]
        action_to_node_index = {
            int(nodes[index].action_index): index
            for index in action_node_indices
            if nodes[index].action_index is not None
        }

        pad_count = max(0, self.max_nodes - len(nodes))
        if pad_count:
            feature_rows.extend([[0.0] * GRAPH_NODE_FEATURE_DIM for _ in range(pad_count)])
            node_type_ids.extend([GRAPH_NODE_TYPES["pad"]] * pad_count)
            node_mask.extend([False] * pad_count)
            executable_mask.extend([False] * pad_count)

        return {
            "max_nodes": self.max_nodes,
            "feature_names": list(GRAPH_FEATURE_NAMES),
            "node_feature_dim": GRAPH_NODE_FEATURE_DIM,
            "nodes": [node.as_dict() for node in nodes],
            "node_count": len(nodes),
            "node_features": feature_rows[: self.max_nodes],
            "node_type_ids": node_type_ids[: self.max_nodes],
            "node_mask": node_mask[: self.max_nodes],
            "executable_mask": executable_mask[: self.max_nodes],
            "action_indices": action_node_indices,
            "action_to_node_index": action_to_node_index,
            "action_labels": action_labels,
            "edges": self._edges(nodes),
            "summary": {
                "real_nodes": len(nodes),
                "executable_nodes": sum(1 for node in nodes if node.executable),
                "reachable_nodes": sum(1 for node in nodes if node.reachable),
                "consumed_nodes": sum(1 for node in nodes if node.consumed),
            },
        }

    def _floor_nodes(
        self,
        state: MotaState,
        floor_id: str,
        reachable: set[tuple[int, int]],
        distance_by_cell: dict[tuple[int, int], int],
        action_lookup: dict[tuple[str, int, int], dict[str, Any]],
    ) -> list[GraphNode]:
        nodes: list[GraphNode] = []
        original = self.sim.data.floors[floor_id]["map"]
        current = state.floors[floor_id]
        event_cells = {
            tuple(map(int, key.split(",")))
            for key in self.sim.data.floors[floor_id].get("events", {})
            if "," in key
        }
        coordinates: set[tuple[int, int]] = set(self.sim.graph_floor_coordinates(floor_id))
        for y, row in enumerate(current):
            for x, tile in enumerate(row):
                if self._kind_for_tile(tile):
                    coordinates.add((x, y))

        for x, y in sorted(coordinates, key=lambda pos: (pos[1], pos[0])):
            original_tile = original[y][x] if y < len(original) and x < len(original[y]) else 0
            current_tile = current[y][x] if y < len(current) and x < len(current[y]) else 0
            tile = current_tile if self._kind_for_tile(current_tile) else original_tile
            kind = self._kind_for_tile(tile)
            if (x, y) in event_cells and (floor_id, x, y) not in state.triggered_events:
                kind = "event" if kind is None else kind
            if kind is None:
                continue
            block_id = self.sim.block_id(tile) or ("event" if kind == "event" else None)
            consumed = bool(self._kind_for_tile(original_tile) and not self._kind_for_tile(current_tile))
            if (floor_id, x, y) in state.triggered_events and kind == "event":
                consumed = True
            node_id = self._node_id(floor_id, x, y, kind, block_id)
            node = GraphNode(
                node_id=node_id,
                kind=kind,
                floor=floor_id,
                x=x,
                y=y,
                block_id=block_id,
                raw_id=block_id,
                consumed=consumed,
            )
            self._mark_reachability(state, node, reachable, distance_by_cell)
            target_key = (floor_id, x, y)
            if target_key in action_lookup:
                node.executable = True
                node.action_index = int(action_lookup[target_key]["index"])
                node.action_label = str(action_lookup[target_key]["label"])
                node.path_len = len(action_lookup[target_key].get("path", []))
                action_lookup[target_key]["node_id"] = node.node_id
            nodes.append(node)
        return nodes

    def _kind_for_tile(self, tile: int) -> str | None:
        return self.sim.graph_tile_kind(tile)

    def _node_id(self, floor_id: str, x: int, y: int, kind: str, block_id: str | None) -> str:
        return f"{floor_id}:{x},{y}:{kind}:{block_id or 'none'}"

    def _target_key(self, action: dict[str, Any]) -> tuple[str, int, int] | None:
        target = action.get("target")
        if target and len(target) == 3:
            return str(target[0]), int(target[1]), int(target[2])
        loc = action.get("loc")
        floor = action.get("floor")
        if loc and floor and len(loc) == 2:
            return str(floor), int(loc[0]), int(loc[1])
        return None

    def _action_lookup(self, actions: list[dict[str, Any]]) -> dict[tuple[str, int, int], dict[str, Any]]:
        lookup: dict[tuple[str, int, int], dict[str, Any]] = {}
        for index, action in enumerate(actions):
            key = self._target_key(action)
            if key is None:
                continue
            previous = lookup.get(key)
            if previous is None or len(action.get("path", [])) < len(previous.get("path", [])):
                lookup[key] = {
                    "index": index,
                    "label": action.get("label", ""),
                    "path": action.get("path", []),
                    "node_id": "",
                }
        return lookup

    def _mark_reachability(
        self,
        state: MotaState,
        node: GraphNode,
        reachable: set[tuple[int, int]],
        distance_by_cell: dict[tuple[int, int], int],
    ) -> None:
        if node.floor != state.floor_id or node.consumed:
            return
        cell = (node.x, node.y)
        if node.kind in {"item", "stair", "npc", "event"} and cell in reachable:
            node.reachable = True
            node.path_len = distance_by_cell.get(cell, 0)
            return
        if node.kind in {"door", "enemy", "boss"}:
            best = None
            for nx, ny in ((node.x, node.y - 1), (node.x, node.y + 1), (node.x - 1, node.y), (node.x + 1, node.y)):
                dist = distance_by_cell.get((nx, ny))
                if dist is not None and (best is None or dist + 1 < best):
                    best = dist + 1
            if best is not None:
                node.reachable = True
                node.path_len = best

    def _attach_node_features(self, state: MotaState, node: GraphNode) -> None:
        node.features["item_value"] = 0.0
        node.features["unlock_value"] = 0.0
        if node.kind == "item" and node.block_id:
            node.features["item_value"] = self._item_value(state, node.block_id)
        elif node.kind in {"enemy", "boss"} and node.block_id:
            enemy = self.sim.data.enemys.get(node.block_id, {})
            info = self.sim.damage_info(state, node.block_id)
            killable = False
            if info is not None:
                damage = int(info["damage"])
                killable = state.hp > damage or (
                    self.sim.config.allow_negative_hp
                    and state.hp - damage >= self.sim.config.min_hp
                )
            node.features.update(
                {
                    "enemy_hp": int(enemy.get("hp", 0)),
                    "enemy_atk": int(enemy.get("atk", 0)),
                    "enemy_def": int(enemy.get("def", 0)),
                    "damage": None if info is None else int(info["damage"]),
                    "killable": killable,
                    "hp_margin": None if info is None else state.hp - int(info["damage"]),
                    "damage_drop_atk+1": self._damage_drop(state, node.block_id, atk_delta=1),
                    "damage_drop_def+1": self._damage_drop(state, node.block_id, def_delta=1),
                }
            )
            if self.include_unlock_values:
                node.features["unlock_value"] = self._one_step_unlock_value(state, node)
        elif node.kind == "door":
            tile = self.sim.tile(state, node.x, node.y, node.floor)
            if self.sim.is_door_tile(tile):
                keys = self.sim.block_info(tile).get("doorInfo", {}).get("keys", {})
            else:
                keys = self.sim.block_info(self.sim.tile_number(node.block_id or "0")).get("doorInfo", {}).get("keys", {}) if node.block_id else {}
            node.required_keys = {str(key): int(value) for key, value in keys.items()}
            node.missing_keys = {
                str(key): missing
                for key, value in keys.items()
                if (missing := max(0, int(value) - state.items.get(str(key), 0))) > 0
            }
            node.features["openable"] = bool(not node.missing_keys and not node.consumed)
            if self.include_unlock_values:
                node.features["unlock_value"] = self._door_unlock_value(state, node)

    def _one_step_unlock_value(self, state: MotaState, node: GraphNode) -> float:
        if node.floor != state.floor_id or node.consumed or not node.block_id:
            return 0.0
        tile = self.sim.tile(state, node.x, node.y)
        if not self.sim.is_enemy_tile(tile) or not self.sim.can_battle(state, tile):
            return 0.0
        before = set(self.sim.reachable_parent_map(state))
        child = state.clone()
        if not self.sim.battle(child, node.x, node.y):
            return 0.0
        after = set(self.sim.reachable_parent_map(child))
        value = float(self.sim.data.enemys.get(node.block_id, {}).get("money", 0)) * 0.4
        for x, y in after - before:
            item_id = self.sim.block_id(self.sim.tile(child, x, y))
            if item_id:
                value += self._item_value(child, item_id)
        return value

    def _door_unlock_value(self, state: MotaState, node: GraphNode) -> float:
        if node.floor != state.floor_id or node.consumed:
            return 0.0
        tile = self.sim.tile(state, node.x, node.y)
        if not self.sim.is_door_tile(tile) or not self.sim.can_open_door(state, tile):
            return 0.0
        before = set(self.sim.reachable_parent_map(state))
        child = state.clone()
        if not self.sim.open_door(child, node.x, node.y, consume_key=True):
            return 0.0
        after = set(self.sim.reachable_parent_map(child))
        value = self._new_reachable_value(child, before, after)
        # Hidden doors/fake walls often have no key cost.  Look one extra
        # no-cost door ahead so the graph can value routes that reveal a major
        # resource immediately behind a fake wall, such as the 5F sword.
        zero_cost_doors: set[tuple[int, int]] = set()
        for x, y in after:
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nx, ny = x + dx, y + dy
                tile_after = self.sim.tile(child, nx, ny)
                if self.sim.is_door_tile(tile_after):
                    zero_cost_doors.add((nx, ny))
        for x, y in zero_cost_doors:
            tile_after = self.sim.tile(child, x, y)
            keys = self.sim.block_info(tile_after).get("doorInfo", {}).get("keys", {})
            if keys:
                continue
            grandchild = child.clone()
            if not self.sim.open_door(grandchild, x, y, consume_key=True):
                continue
            second_after = set(self.sim.reachable_parent_map(grandchild))
            value += 0.8 * self._new_reachable_value(grandchild, after, second_after)
        return value

    def _new_reachable_value(
        self,
        state: MotaState,
        before: set[tuple[int, int]],
        after: set[tuple[int, int]],
    ) -> float:
        value = 0.0
        for x, y in after - before:
            tile_after = self.sim.tile(state, x, y)
            block_id = self.sim.block_id(tile_after)
            if not block_id:
                continue
            if self.sim.is_item_tile(tile_after):
                value += self._item_value(state, block_id)
            elif self.sim.is_stair_tile(tile_after):
                value += 500.0 if block_id == "upFloor" else 80.0
            elif self.sim.is_enemy_tile(tile_after):
                info = self.sim.damage_info(state, block_id)
                if info is not None:
                    value += max(0.0, 60.0 - float(info["damage"]) * 0.1)
        return value

    def _damage_drop(self, state: MotaState, enemy_id: str, atk_delta: int = 0, def_delta: int = 0) -> int:
        current = self.sim.damage_info(state, enemy_id)
        if current is None:
            return 0
        improved = self.sim.damage_info_for_stats(
            enemy_id,
            atk=state.atk + atk_delta,
            defense=state.defense + def_delta,
            mdef=state.mdef,
        )
        if improved is None:
            return 0
        return max(0, int(current["damage"]) - int(improved["damage"]))

    @staticmethod
    def _distance_by_cell(
        parents: dict[tuple[int, int], tuple[tuple[int, int], str] | None],
    ) -> dict[tuple[int, int], int]:
        distances: dict[tuple[int, int], int] = {}

        def distance(cell: tuple[int, int]) -> int:
            cached = distances.get(cell)
            if cached is not None:
                return cached
            parent = parents[cell]
            if parent is None:
                distances[cell] = 0
                return 0
            value = distance(parent[0]) + 1
            distances[cell] = value
            return value

        for cell in parents:
            distance(cell)
        return distances

    def _item_value(self, state: MotaState, item_id: str) -> float:
        from .rewards import item_value

        return float(item_value(self.sim, state, item_id))

    def _synthetic_action_node(self, state: MotaState, action: dict[str, Any], index: int) -> GraphNode:
        floor = str(action.get("floor") or state.floor_id)
        loc = action.get("loc") or [state.x, state.y]
        x, y = int(loc[0]), int(loc[1])
        return GraphNode(
            node_id=f"action:{index}:{action.get('label', '')}",
            kind="action",
            floor=floor,
            x=x,
            y=y,
            block_id=str(action.get("shop") or action.get("kind", "action")),
            reachable=True,
            executable=True,
            path_len=len(action.get("path", [])),
            action_index=index,
            action_label=str(action.get("label", "")),
        )

    def _attach_action_indices(self, nodes: list[GraphNode], actions: list[dict[str, Any]]) -> None:
        by_target = {(node.floor, node.x, node.y): node for node in nodes}
        for index, action in enumerate(actions):
            key = self._target_key(action)
            if key is None:
                continue
            node = by_target.get(key)
            if node is None:
                continue
            node.executable = True
            node.action_index = index
            node.action_label = str(action.get("label", ""))
            node.path_len = len(action.get("path", []))

    def _find_node_index(self, nodes: list[GraphNode], node_id: str) -> int | None:
        if not node_id:
            return None
        for index, node in enumerate(nodes):
            if node.node_id == node_id:
                return index
        return None

    def _sorted_nodes(self, nodes: list[GraphNode]) -> list[GraphNode]:
        priority = {"hero": 0, "global": 1, "boss": 2, "item": 3, "enemy": 4, "door": 5, "npc": 6, "stair": 7, "event": 8}
        return sorted(
            nodes,
            key=lambda node: (
                0 if node.executable else 1,
                0 if node.reachable else 1,
                priority.get(node.kind, 9),
                self._floor_index(node.floor),
                node.y,
                node.x,
                node.node_id,
            ),
        )

    def _floor_index(self, floor_id: str) -> int:
        try:
            return int(str(floor_id).removeprefix("MT"))
        except ValueError:
            return 0

    def _feature_context(self, state: MotaState) -> dict[str, float]:
        from .rewards import (
            boss_route_margin,
            boss_route_required_damage,
            current_stage_name,
            remaining_attack_defense_gems,
        )

        stage_order = (
            "sword",
            "pre_shield_gems",
            "shield",
            "mt8_gems",
            "mid_gems",
            "low_gems",
            "all_gems",
            "red_key",
            "boss_ready",
            "trap",
            "boss",
            "done",
        )
        stage = current_stage_name(self.sim, state)
        stage_norm = stage_order.index(stage) / max(1, len(stage_order) - 1) if stage in stage_order else 0.0
        return {
            "stage_norm": stage_norm,
            "boss_damage": float(boss_route_required_damage(self.sim, state)),
            "boss_margin": float(boss_route_margin(self.sim, state)),
            "mt10_remaining": float(self._mt10_resource_remaining(state)),
            "remaining_gems": float(remaining_attack_defense_gems(self.sim, state)),
        }

    def _node_feature_vector(
        self,
        state: MotaState,
        node: GraphNode,
        context: dict[str, float],
    ) -> list[float]:
        f_idx = self._floor_index(node.floor)
        damage = node.features.get("damage") if node.kind in {"enemy", "boss"} else 0
        hp_margin = node.features.get("hp_margin") if node.kind in {"enemy", "boss"} else 0
        missing = node.missing_keys
        required = node.required_keys
        stage_norm = context["stage_norm"]
        boss_damage = context["boss_damage"]
        boss_margin = context["boss_margin"]
        mt10_remaining = context["mt10_remaining"]
        remaining_gems = context["remaining_gems"]
        vector = [
            _clip_norm(f_idx, 10.0),
            _clip_norm(node.x, 12.0),
            _clip_norm(node.y, 12.0),
            _clip_norm(GRAPH_NODE_TYPES.get(node.kind, 0), max(GRAPH_NODE_TYPES.values())),
            float(node.consumed),
            float(node.reachable),
            float(node.executable),
            _clip_norm(node.path_len if node.path_len is not None else 99, 60.0),
            _clip_norm(state.hp, 3000.0),
            _clip_norm(state.atk, 80.0),
            _clip_norm(state.defense, 80.0),
            _clip_norm(state.money, 300.0),
            _clip_norm(state.items.get("yellowKey", 0), 10.0),
            _clip_norm(state.items.get("blueKey", 0), 5.0),
            _clip_norm(state.items.get("redKey", 0), 2.0),
            _clip_norm(node.features.get("item_value", 0.0), 300.0),
            _clip_norm(node.features.get("enemy_hp", 0), 1000.0),
            _clip_norm(node.features.get("enemy_atk", 0), 200.0),
            _clip_norm(node.features.get("enemy_def", 0), 120.0),
            _clip_norm(3000.0 if damage is None else damage, 3000.0),
            _signed_clip_norm(0.0 if hp_margin is None else hp_margin, 3000.0),
            float(node.features.get("killable", False)),
            _clip_norm(node.features.get("damage_drop_atk+1", 0), 800.0),
            _clip_norm(node.features.get("damage_drop_def+1", 0), 800.0),
            _clip_norm(node.features.get("unlock_value", 0.0), 500.0),
            float(node.features.get("openable", False)),
            _clip_norm(missing.get("yellowKey", 0), 5.0),
            _clip_norm(missing.get("blueKey", 0), 3.0),
            _clip_norm(missing.get("redKey", 0), 1.0),
            _clip_norm(required.get("yellowKey", 0), 5.0),
            _clip_norm(required.get("blueKey", 0), 3.0),
            _clip_norm(required.get("redKey", 0), 1.0),
            stage_norm,
            _clip_norm(boss_damage, 3000.0),
            _signed_clip_norm(boss_margin, 3000.0),
            _clip_norm(remaining_gems, 16.0),
            _clip_norm(remaining_gems, 16.0),
            _clip_norm(mt10_remaining, 3.0),
            float(node.floor == "MT10"),
            0.0,
        ]
        return vector

    def _mt10_resource_remaining(self, state: MotaState) -> int:
        count = 0
        if "MT10" not in state.floors:
            return 0
        for x, y in ((2, 6), (10, 6), (11, 11)):
            block_id = self.sim.block_id(self.sim.tile(state, x, y, "MT10"))
            if block_id in {"redGem", "blueGem", "bluePotion"}:
                count += 1
        return count

    def _edges(self, nodes: list[GraphNode]) -> list[dict[str, Any]]:
        edges: list[dict[str, Any]] = []
        node_ids = {node.node_id for node in nodes}
        for node in nodes:
            if node.node_id == "hero":
                continue
            if node.reachable and "hero" in node_ids:
                edges.append(
                    {
                        "source": "hero",
                        "target": node.node_id,
                        "kind": "reachable" if node.executable else "visible",
                        "cost": node.path_len,
                    }
                )
        return edges


def _clip_norm(value: Any, scale: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    if scale <= 0:
        return 0.0
    return max(0.0, min(number / scale, 1.0))


def _signed_clip_norm(value: Any, scale: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    if scale <= 0:
        return 0.0
    return max(-1.0, min(number / scale, 1.0))


def build_graph_state(
    sim: MotaSimulator,
    state: MotaState,
    actions: list[dict[str, Any]] | None = None,
    max_nodes: int = MAX_GRAPH_NODES,
    include_unlock_values: bool | None = None,
) -> dict[str, Any]:
    if max_nodes == MAX_GRAPH_NODES:
        max_nodes_env = os.environ.get("MOTA_GRAPH_MAX_NODES", "").strip()
        if max_nodes_env:
            try:
                max_nodes = max(16, min(MAX_GRAPH_NODES, int(max_nodes_env)))
            except ValueError:
                max_nodes = MAX_GRAPH_NODES
    if include_unlock_values is None:
        fast_env = os.environ.get("MOTA_FAST_GRAPH_STATE", "").strip().lower()
        include_unlock_values = fast_env not in {"1", "true", "yes", "on"}
    cache_key = None
    if hasattr(sim, "graph_state_cache_key"):
        action_signature = None
        if actions is not None:
            action_signature = tuple(
                (
                    action.get("kind"),
                    tuple(action.get("target", [])),
                    action.get("floor"),
                    tuple(action.get("loc", [])),
                    action.get("shop"),
                    action.get("label"),
                    len(action.get("path", [])),
                )
                for action in actions
            )
        cache_key = sim.graph_state_cache_key(
            state,
            max_nodes=max_nodes,
            include_unlock_values=bool(include_unlock_values),
        ) + (action_signature,)
        cached = sim.cached_graph_state(cache_key)
        if cached is not None:
            return cached
    result = GraphStateBuilder(
        sim,
        max_nodes=max_nodes,
        include_unlock_values=include_unlock_values,
    ).build(state, actions=actions)
    if cache_key is not None:
        result["_cache_key"] = cache_key
    if cache_key is not None:
        sim.set_cached_graph_state(cache_key, result)
    return result
