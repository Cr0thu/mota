from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .graph_state import build_graph_state

if False:  # pragma: no cover
    from .simulator import MotaSimulator, MotaState


@dataclass(frozen=True)
class ResourceGraphNode:
    node_id: str
    kind: str
    floor: str
    x: int
    y: int
    raw_id: str | None
    consumed: bool
    reachable: bool
    executable: bool
    path_len: int | None
    action_index: int | None
    action_label: str | None
    features: dict[str, Any]


@dataclass(frozen=True)
class ResourceGraph:
    stage: str
    nodes: tuple[ResourceGraphNode, ...]
    actions: tuple[dict[str, Any], ...]
    action_to_node_id: dict[int, str]
    global_features: dict[str, Any]

    @property
    def executable_node_ids(self) -> list[str]:
        return [node.node_id for node in self.nodes if node.executable]

    def action_for_node(self, node_id: str) -> dict[str, Any] | None:
        for action_index, candidate_node_id in self.action_to_node_id.items():
            if candidate_node_id == node_id:
                return self.actions[action_index]
        return None


class MotaResourceGraphBuilder:
    """Interaction graph used by research search/RL code.

    This is intentionally separate from `tools/visualizer`.  It reuses the
    simulator and graph-state feature extraction, then exposes stable node and
    resource keys for planners, archives, dominance pruning, and learned Q
    models.
    """

    def __init__(self, sim: "MotaSimulator", max_nodes: int = 256):
        self.sim = sim
        self.max_nodes = int(max_nodes)

    def build(
        self,
        state: "MotaState",
        *,
        stage: str = "unknown",
        actions: list[dict[str, Any]] | None = None,
    ) -> ResourceGraph:
        actions = actions if actions is not None else self.sim.macro_actions(state)
        graph = build_graph_state(self.sim, state, actions=actions)
        nodes = tuple(self._node_from_dict(node) for node in graph["nodes"])
        action_to_node_id = {
            int(action_index): nodes[int(node_index)].node_id
            for action_index, node_index in graph["action_to_node_index"].items()
            if int(node_index) < len(nodes)
        }
        return ResourceGraph(
            stage=stage,
            nodes=nodes,
            actions=tuple(actions),
            action_to_node_id=action_to_node_id,
            global_features=self.global_features(state),
        )

    @staticmethod
    def _node_from_dict(node: dict[str, Any]) -> ResourceGraphNode:
        return ResourceGraphNode(
            node_id=str(node["node_id"]),
            kind=str(node["kind"]),
            floor=str(node["floor"]),
            x=int(node["x"]),
            y=int(node["y"]),
            raw_id=node.get("raw_id") or node.get("block_id"),
            consumed=bool(node.get("consumed")),
            reachable=bool(node.get("reachable")),
            executable=bool(node.get("executable")),
            path_len=node.get("path_len"),
            action_index=node.get("action_index"),
            action_label=node.get("action_label"),
            features=dict(node.get("features") or {}),
        )

    def global_features(self, state: "MotaState") -> dict[str, Any]:
        return {
            "floor": state.floor_id,
            "x": state.x,
            "y": state.y,
            "hp": state.hp,
            "atk": state.atk,
            "def": state.defense,
            "money": state.money,
            "yellowKey": state.items.get("yellowKey", 0),
            "blueKey": state.items.get("blueKey", 0),
            "redKey": state.items.get("redKey", 0),
            "flags": dict(state.flags),
            "consumed_hash": consumed_hash(state),
        }


def consumed_hash(state: "MotaState") -> int:
    cache = getattr(state, "_signature_cache", None)
    if cache is not None and "consumed_hash" in cache:
        return int(cache["consumed_hash"])
    consumed: list[tuple[str, int, int, int]] = []
    for floor_id, grid in state.floors.items():
        for y, row in enumerate(grid):
            for x, value in enumerate(row):
                if value == 0:
                    consumed.append((floor_id, x, y, value))
    value = hash(tuple(consumed))
    if cache is not None:
        cache["consumed_hash"] = value
    return value


def resource_vector(state: "MotaState", *, hp_debt: bool = False) -> tuple[int, ...]:
    hp = int(state.hp)
    debt = max(0, -hp) if hp_debt else 0
    return (
        hp,
        int(state.atk),
        int(state.defense),
        int(state.money),
        int(state.items.get("yellowKey", 0)),
        int(state.items.get("blueKey", 0)),
        int(state.items.get("redKey", 0)),
        -debt,
    )


def dominance_signature(state: "MotaState") -> tuple[Any, ...]:
    flags = tuple(sorted((str(key), repr(value)) for key, value in state.flags.items()))
    return (
        state.floor_id,
        int(state.x),
        int(state.y),
        consumed_hash(state),
        flags,
    )


def archive_cell_key(state: "MotaState", stage: str) -> tuple[Any, ...]:
    floor_index = int(state.floor_id[2:]) if state.floor_id.startswith("MT") else 0
    return (
        stage,
        min(10, floor_index),
        int(state.atk) // 2,
        int(state.defense) // 2,
        min(9, max(0, int(state.items.get("yellowKey", 0)))),
        min(4, max(0, int(state.items.get("blueKey", 0)))),
        min(1, max(0, int(state.items.get("redKey", 0)))),
        max(-20, min(20, int(state.hp) // 50)),
        bool(state.flags.get("10f机关")),
        bool(state.flags.get("10f战胜骷髅队长")),
    )
