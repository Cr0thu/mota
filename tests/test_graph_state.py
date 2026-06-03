from __future__ import annotations

from mota_env import MotaSimulator, build_graph_state, load_game_data
from mota_env.graph_state import GRAPH_NODE_FEATURE_DIM, MAX_GRAPH_NODES


def test_full_graph_state_contains_global_nodes_and_masks() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    actions = sim.macro_actions(state)

    graph = build_graph_state(sim, state, actions=actions)

    assert graph["max_nodes"] == MAX_GRAPH_NODES
    assert graph["node_feature_dim"] == GRAPH_NODE_FEATURE_DIM
    assert len(graph["node_features"]) == MAX_GRAPH_NODES
    assert len(graph["node_type_ids"]) == MAX_GRAPH_NODES
    assert len(graph["node_mask"]) == MAX_GRAPH_NODES
    assert len(graph["executable_mask"]) == MAX_GRAPH_NODES
    assert sum(graph["executable_mask"]) == len(actions)
    assert all(graph["executable_mask"][index] for index in graph["action_indices"])
    assert set(graph["action_to_node_index"]) == set(range(len(actions)))
    assert len(set(node["node_id"] for node in graph["nodes"])) == graph["node_count"]

    kinds = {node["kind"] for node in graph["nodes"]}
    assert {"hero", "global", "item", "enemy", "door", "stair"} <= kinds
    assert any(not node["reachable"] for node in graph["nodes"] if node["kind"] in {"item", "enemy", "door"})


def test_graph_enemy_features_include_damage_cliffs_and_unlock_value() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    graph = build_graph_state(sim, state)

    enemy_nodes = [
        node for node in graph["nodes"]
        if node["kind"] in {"enemy", "boss"} and node["features"].get("damage") is not None
    ]
    assert enemy_nodes
    for node in enemy_nodes[:8]:
        assert "damage_drop_atk+1" in node["features"]
        assert "damage_drop_def+1" in node["features"]
        assert "unlock_value" in node["features"]


def test_describe_state_exposes_full_graph_state() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    snapshot = sim.describe_state(state)

    assert "full_graph_state" in snapshot
    assert snapshot["full_graph_state"]["summary"]["executable_nodes"] == len(sim.macro_actions(state))
