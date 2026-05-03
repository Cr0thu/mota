from __future__ import annotations

from mota_env import load_game_data


def test_first_ten_floor_data_shape() -> None:
    data = load_game_data("artifacts/data/mota_first10.json")
    assert data.first_data["floorId"] == "MT1"
    assert set(f"MT{i}" for i in range(1, 11)).issubset(data.floors)
    for floor_id in [f"MT{i}" for i in range(1, 11)]:
        floor_map = data.floors[floor_id]["map"]
        assert len(floor_map) == 13
        assert all(len(row) == 13 for row in floor_map)


def test_critical_entities_exist() -> None:
    data = load_game_data("artifacts/data/mota_first10.json")
    assert "skeletonCaptain" in data.enemys
    assert data.maps["211"]["id"] == "skeletonCaptain"
    assert data.floors["MT10"]["events"].get("6,5") is not None
