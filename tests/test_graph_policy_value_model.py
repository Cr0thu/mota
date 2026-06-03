from __future__ import annotations

import pytest

from mota_env import MotaSimulator, build_graph_state, load_game_data


def test_graph_policy_value_shapes_and_masked_policy() -> None:
    torch = pytest.importorskip("torch")
    from mota_rl.graph_policy_value_model import (
        GraphPolicyValueConfig,
        GraphPolicyValueNet,
        gather_graph_batch,
        masked_policy,
    )

    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    graph = build_graph_state(sim, state)
    batch = gather_graph_batch([graph])

    model = GraphPolicyValueNet(GraphPolicyValueConfig(d_model=64, nhead=4, num_layers=1, dropout=0.0))
    logits, value = model(batch["node_features"], batch["node_type_ids"], batch["node_mask"])
    policy = masked_policy(logits, batch["executable_mask"])

    assert logits.shape == torch.Size([1, graph["max_nodes"]])
    assert value.shape == torch.Size([1])
    assert policy.shape == torch.Size([1, graph["max_nodes"]])
    assert torch.isclose(policy[0, batch["executable_mask"][0]].sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.all(policy[0, ~batch["executable_mask"][0]] == 0)
