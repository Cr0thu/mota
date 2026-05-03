from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = REPO_ROOT / "artifacts" / "data" / "mota_first10.json"


@dataclass(frozen=True)
class GameData:
    source_project: str
    first_data: dict[str, Any]
    values: dict[str, Any]
    flags: dict[str, Any]
    maps: dict[str, dict[str, Any]]
    enemys: dict[str, dict[str, Any]]
    items: dict[str, dict[str, Any]]
    floors: dict[str, dict[str, Any]]

    @property
    def floor_ids(self) -> list[str]:
        return sorted(self.floors, key=lambda floor_id: int(floor_id[2:]))


def load_game_data(path: str | Path = DEFAULT_DATA_PATH) -> GameData:
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} does not exist. Run `node scripts/extract_mota_data.js` first."
        )
    payload = json.loads(data_path.read_text(encoding="utf8"))
    return GameData(
        source_project=payload["source_project"],
        first_data=payload["firstData"],
        values=payload["values"],
        flags=payload["flags"],
        maps=payload["maps"],
        enemys=payload["enemys"],
        items=payload["items"],
        floors=payload["floors"],
    )

