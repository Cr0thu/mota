from __future__ import annotations

from mota_env import MotaSimulator, load_game_data


def test_describe_state_contains_research_logging_fields() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    actions = sim.macro_actions(state)
    snapshot = sim.describe_state(state, actions=actions)

    assert snapshot["hero"]["floor"] == "MT2"
    assert snapshot["action_mask"]["valid_count"] == len(actions)
    assert "boss" in snapshot
    assert "damage_drop_atk+1" in snapshot["boss"]
    assert "openable_doors" in snapshot["reachable"]
    assert "blocked_doors" in snapshot["reachable"]
    assert "monsters" in snapshot["reachable"]
    assert "graph_state" in snapshot
    assert snapshot["graph_state"]["node_count"] >= 1
    assert "global_resource_summary" in snapshot
    assert "MT10" in snapshot["global_resource_summary"]["by_floor_remaining"]
    assert "phi" in snapshot


def test_trajectory_step_record_has_before_after_features() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    action = sim.macro_actions(state)[0]
    before = state.clone()
    transition = sim.apply_macro_action(state, action)
    record = sim.trajectory_step_record(before, state, action, transition, index=0)

    assert record["index"] == 0
    assert record["before_features"]["hero"]["floor"] == before.floor_id
    assert record["after_features"]["hero"]["floor"] == state.floor_id
