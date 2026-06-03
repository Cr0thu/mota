from __future__ import annotations

from dataclasses import dataclass
from typing import Any


TOKEN_TYPES = {
    "hero": 0,
    "keys": 1,
    "monster": 2,
    "resource": 3,
    "door": 4,
    "boss": 5,
    "global": 6,
}

STAGE_IDS = {
    "sword": 0,
    "pre_shield_gems": 1,
    "shield": 2,
    "mt8_gems": 3,
    "mid_gems": 4,
    "low_gems": 5,
    "lower_gems": 6,
    "all_gems": 7,
    "red_key": 8,
    "boss_ready": 9,
    "trap": 10,
    "boss": 11,
    "done": 12,
    "gems": 13,
    "mt10_blue_ready": 14,
    "mt10_yellow_ready": 15,
    "mt10_ready": 16,
    "mt10_resources": 17,
    "guard_ready": 18,
}

STAGE_SCALE = max(1, len(STAGE_IDS) - 1)


@dataclass(frozen=True)
class TokenizedState:
    token_types: list[int]
    values: list[list[float]]
    mask: list[bool]


def tokenize_feature_snapshot(snapshot: dict[str, Any], max_tokens: int = 160) -> TokenizedState:
    """Convert `mota_env.features.describe_state` output to Transformer tokens."""

    token_types: list[int] = []
    values: list[list[float]] = []

    def add(kind: str, row: list[float]) -> None:
        if len(token_types) >= max_tokens:
            return
        padded = row[:12] + [0.0] * max(0, 12 - len(row))
        token_types.append(TOKEN_TYPES[kind])
        values.append(padded[:12])

    hero = snapshot["hero"]
    stage = snapshot.get("stage", {})
    keys = hero.get("keys", {})
    add(
        "hero",
        [
            _floor(hero.get("floor", "MT0")) / 10.0,
            hero.get("x", 0) / 12.0,
            hero.get("y", 0) / 12.0,
            hero.get("hp", 0) / 2000.0,
            hero.get("atk", 0) / 80.0,
            hero.get("def", 0) / 80.0,
            hero.get("money", 0) / 200.0,
            STAGE_IDS.get(stage.get("name", "shield"), 0) / STAGE_SCALE,
        ],
    )
    add(
        "keys",
        [
            keys.get("yellowKey", 0) / 10.0,
            keys.get("blueKey", 0) / 3.0,
            keys.get("redKey", 0) / 2.0,
            snapshot.get("reachable", {}).get("cell_count", 0) / 169.0,
        ],
    )
    boss = snapshot.get("boss", {})
    add(
        "boss",
        [
            _none_to(boss.get("damage"), 2000) / 2000.0,
            _none_to(boss.get("hp_margin"), -2000) / 2000.0,
            boss.get("break_atk_needed", 0) / 30.0,
            boss.get("damage_drop_atk+1", 0) / 300.0,
            boss.get("damage_drop_def+1", 0) / 300.0,
            float(bool(boss.get("killable"))),
        ],
    )
    for monster in snapshot.get("reachable", {}).get("monsters", [])[:96]:
        add(
            "monster",
            [
                _floor(monster.get("floor", "MT0")) / 10.0,
                monster.get("x", 0) / 12.0,
                monster.get("y", 0) / 12.0,
                monster.get("enemy_hp", 0) / 1000.0,
                monster.get("enemy_atk", 0) / 100.0,
                monster.get("enemy_def", 0) / 100.0,
                _none_to(monster.get("damage"), 2000) / 2000.0,
                _none_to(monster.get("hp_margin"), -2000) / 2000.0,
                monster.get("damage_drop_atk+1", 0) / 300.0,
                monster.get("damage_drop_def+1", 0) / 300.0,
                float(bool(monster.get("killable"))),
                len(monster.get("blocking_new_resources", [])) / 5.0,
            ],
        )
    for item in snapshot.get("reachable", {}).get("resources", [])[:32]:
        add(
            "resource",
            [
                _floor(item.get("floor", "MT0")) / 10.0,
                item.get("x", 0) / 12.0,
                item.get("y", 0) / 12.0,
                _item_id(item.get("item", "")) / 16.0,
                item.get("path_len", 0) / 40.0,
            ],
        )
    for door in snapshot.get("reachable", {}).get("openable_doors", [])[:16]:
        add(
            "door",
            [
                _floor(door.get("floor", "MT0")) / 10.0,
                door.get("x", 0) / 12.0,
                door.get("y", 0) / 12.0,
                _door_id(door.get("door", "")) / 8.0,
                1.0,
            ],
        )
    for door in snapshot.get("reachable", {}).get("blocked_doors", [])[:16]:
        missing = door.get("missing_keys", {})
        add(
            "door",
            [
                _floor(door.get("floor", "MT0")) / 10.0,
                door.get("x", 0) / 12.0,
                door.get("y", 0) / 12.0,
                _door_id(door.get("door", "")) / 8.0,
                0.0,
                missing.get("yellowKey", 0),
                missing.get("blueKey", 0),
                missing.get("redKey", 0),
            ],
        )
    global_resources = snapshot.get("global_resources", {})
    add(
        "global",
        [
            global_resources.get("redGem", 0) / 10.0,
            global_resources.get("blueGem", 0) / 10.0,
            global_resources.get("yellowKey", 0) / 50.0,
            global_resources.get("blueKey", 0) / 5.0,
            global_resources.get("redKey", 0) / 2.0,
            snapshot.get("remaining_attack_defense_gems", 0) / 20.0,
        ],
    )

    mask = [True] * len(token_types)
    while len(token_types) < max_tokens:
        token_types.append(0)
        values.append([0.0] * 12)
        mask.append(False)
    return TokenizedState(token_types=token_types, values=values, mask=mask)


def build_attention_model(d_model: int = 128, nhead: int = 4, num_layers: int = 4):
    try:
        import torch
        from torch import nn
    except Exception as exc:  # pragma: no cover - depends on optional torch install.
        raise RuntimeError("attention model requires torch: pip install torch") from exc

    class MotaAttentionValueNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.type_embedding = nn.Embedding(len(TOKEN_TYPES), d_model)
            self.value_projection = nn.Linear(12, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                batch_first=True,
                dropout=0.1,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.norm = nn.LayerNorm(d_model)
            self.success_head = nn.Linear(d_model, 1)
            self.stage_value_head = nn.Linear(d_model, 1)
            self.boss_margin_head = nn.Linear(d_model, 1)

        def forward(self, token_types, values, mask):
            x = self.type_embedding(token_types) + self.value_projection(values)
            encoded = self.encoder(x, src_key_padding_mask=~mask)
            weights = mask.unsqueeze(-1).float()
            pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            pooled = self.norm(pooled)
            return {
                "success_logit": self.success_head(pooled).squeeze(-1),
                "stage_value": self.stage_value_head(pooled).squeeze(-1),
                "boss_margin": self.boss_margin_head(pooled).squeeze(-1),
            }

    return MotaAttentionValueNet()


def _floor(floor_id: str) -> int:
    try:
        return int(str(floor_id).replace("MT", ""))
    except ValueError:
        return 0


def _none_to(value: Any, fallback: float) -> float:
    return fallback if value is None else float(value)


def _item_id(item: str) -> int:
    ids = {
        "yellowKey": 1,
        "blueKey": 2,
        "redKey": 3,
        "redGem": 4,
        "blueGem": 5,
        "greenGem": 6,
        "redPotion": 7,
        "bluePotion": 8,
        "yellowPotion": 9,
        "greenPotion": 10,
        "sword1": 11,
        "shield1": 12,
    }
    return ids.get(item, 0)


def _door_id(door: str) -> int:
    ids = {
        "yellowDoor": 1,
        "blueDoor": 2,
        "redDoor": 3,
        "specialDoor": 4,
    }
    return ids.get(door, 0)
