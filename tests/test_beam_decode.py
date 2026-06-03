from __future__ import annotations

from mota_env import MotaSimulator, load_game_data
from mota_rl.beam_decode import beam_search, delayed_refill_penalty, stage_distance_potential


def test_red_key_distance_prefers_mt8_over_start() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    start = sim.reset()
    for state in [start]:
        state.flags["nowWeapon"] = "sword1"
        state.flags["nowShield"] = "shield1"
        state.atk = 26
        state.defense = 25
        state.hp = 1200
        state.items["yellowKey"] = 3
        for floor in state.floors.values():
            for y, row in enumerate(floor):
                for x, tile in enumerate(row):
                    if sim.block_id(tile) in {"redGem", "blueGem"}:
                        row[x] = 0
    near = start.clone()
    near.floor_id = "MT8"
    near.x = 10
    near.y = 3

    assert stage_distance_potential(sim, near, "red_key") > stage_distance_potential(
        sim,
        start,
        "red_key",
    )


def test_beam_decode_smoke_without_model() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    result = beam_search(
        sim=sim,
        target_stage="shield",
        model_bundle=None,
        beam_width=2,
        action_top_k=2,
        max_steps=2,
    )

    assert result.expanded_nodes > 0
    assert result.generated_nodes > 0
    assert result.route


def test_delayed_refill_penalty_targets_pre_mt10_low_floor_potions() -> None:
    sim = MotaSimulator(load_game_data("artifacts/data/mota_first10.json"))
    before = sim.reset()
    after = before.clone()
    action = {"label": "go bluePotion MT2:3,10"}
    weights = {
        "pre_mt10_potion_penalty": 10.0,
        "pre_mt10_blue_potion_penalty": 20.0,
        "pre_mt10_low_floor_potion_penalty": 30.0,
        "pre_mt10_key_refill_potion_penalty": 40.0,
    }

    assert delayed_refill_penalty(
        sim,
        before,
        after,
        action,
        "mt10_resources",
        weights,
    ) == -100.0

    before.floors["MT10"][6][2] = 0
    assert delayed_refill_penalty(
        sim,
        before,
        after,
        action,
        "mt10_resources",
        weights,
    ) == 0.0
