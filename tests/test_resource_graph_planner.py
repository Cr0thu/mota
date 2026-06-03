from __future__ import annotations

from mota_env import MotaResourceGraphBuilder, MotaSimulator, load_game_data
from mota_solver.resource_planner import (
    DominanceTable,
    ResourcePlannerConfig,
    dump_resource_planner_outputs,
    run_resource_planner,
)


def make_sim() -> MotaSimulator:
    return MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))


def test_resource_graph_wraps_full_graph_without_visualizer_dependency() -> None:
    sim = make_sim()
    state = sim.reset()
    actions = sim.macro_actions(state)
    graph = MotaResourceGraphBuilder(sim).build(state, stage="sword", actions=actions)

    assert graph.stage == "sword"
    assert len(graph.nodes) > len(actions)
    assert len(graph.executable_node_ids) == len(actions)
    assert set(graph.action_to_node_id) == set(range(len(actions)))
    assert graph.global_features["hp"] == 400
    assert graph.global_features["def"] == 10


def test_dominance_table_rejects_weaker_duplicate_state() -> None:
    sim = make_sim()
    stronger = sim.reset()
    weaker = stronger.clone()
    stronger.hp = 500
    stronger.atk = 12
    weaker.hp = 400
    weaker.atk = 10

    table = DominanceTable()

    assert table.accepts(stronger)
    assert not table.accepts(weaker)


def test_resource_planner_smoke_writes_trace(tmp_path) -> None:
    sim = make_sim()
    trace_path = tmp_path / "trace.jsonl"
    result = run_resource_planner(
        sim,
        ResourcePlannerConfig(target_stage="sword", max_expansions=200, archive_top_k=3),
        trace_path=trace_path,
    )

    assert result.expansions > 0
    assert result.archive_cells > 0
    assert result.route
    assert trace_path.exists()
    outputs = dump_resource_planner_outputs(result, tmp_path / "artifacts")
    for output_path in outputs.values():
        assert output_path
    assert (tmp_path / "artifacts" / "summary.json").exists()
    assert (tmp_path / "artifacts" / "archive_cells.jsonl").exists()
    assert (tmp_path / "artifacts" / "best_route.jsonl").exists()
