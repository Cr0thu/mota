from __future__ import annotations

from mota_env import MotaSimulator, load_game_data
from mota_rl.attention_model import tokenize_feature_snapshot


def test_tokenize_feature_snapshot_shapes() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    state = sim.reset()
    snapshot = sim.describe_state(state, actions=sim.macro_actions(state))
    tokenized = tokenize_feature_snapshot(snapshot, max_tokens=32)

    assert len(tokenized.token_types) == 32
    assert len(tokenized.values) == 32
    assert len(tokenized.mask) == 32
    assert tokenized.mask[0]
    assert len(tokenized.values[0]) == 12
