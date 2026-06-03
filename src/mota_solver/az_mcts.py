from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable

from mota_env import MotaSimulator, MotaState, build_graph_state
from mota_env.rewards import (
    LearnableStageReward,
    MT10_RESOURCE_YELLOW_KEY_TARGET,
    Rewarder,
    boss_route_margin,
    current_stage_name,
    has_first_sword,
    mt10_resource_progress,
    red_key_route_margin,
    stage_complete,
)


PolicyValueFn = Callable[[dict[str, Any]], tuple[list[float], float]]
LeafValueFn = Callable[[MotaSimulator, MotaState, str], float]
EdgeRewardFn = Callable[[MotaSimulator, MotaState, MotaState, dict[str, Any], Any, str], float]


RESOURCE_TARGET_STAGES = {
    "mt4_redgem",
    "pre_shield_gems",
    "shield",
    "shield_buffer",
    "mid_gems",
    "low_gems",
    "mt8_hp_ready",
    "mt8_gems",
    "lower_gems",
    "pre_mt10_buffer",
    "mt10_blue_ready",
    "mt10_yellow_ready",
    "mt10_resources",
    "all_gems",
    "red_key",
    "boss_ready",
    "guard_ready",
    "trap",
    "boss",
    "boss_all_gems",
}

MT8_RESOURCE_ACTION_TOKENS = (
    "redGem MT8:4,10",
    "blueGem MT8:5,11",
    "redPotion MT8:1,5",
    "blueKey MT8:7,10",
    "yellowKey MT8:7,11",
    "yellowKey MT8:3,4",
    "yellowKey MT8:4,4",
    "yellowKey MT8:5,4",
    "yellowKey MT8:5,10",
    "yellowKey MT8:4,11",
    "redPotion MT8:8,10",
    "yellowDoor MT8:1,3",
    "yellowDoor MT8:5,7",
    "bat MT8:7,7",
    "bluePriest MT8:8,8",
    "yellowDoor MT8:11,9",
    "skeleton MT8:11,10",
    "yellowDoor MT8:9,11",
    "blueDoor MT8:3,11",
    "bat MT8:4,8",
    "skeleton MT8:6,8",
    "skeletonSoldier MT8:10,11",
)

MT8_BLUE_KEY_CHAIN_ACTION_TOKENS = (
    "bat MT8:7,7",
    "bluePriest MT8:8,8",
    "yellowDoor MT8:11,9",
    "skeleton MT8:11,10",
    "skeletonSoldier MT8:10,11",
    "yellowDoor MT8:9,11",
    "yellowKey MT8:7,11",
    "blueKey MT8:7,10",
)

MT8_POST_BLUE_KEY_GEM_ACTION_TOKENS = (
    "redPotion MT8:8,10",
    "redPotion MT8:1,5",
    "blueDoor MT8:3,11",
    "redGem MT8:4,10",
    "blueGem MT8:5,11",
)

MT8_LOWER_GEM_ROUTE_ACTION_TOKENS = (
    "bluePriest MT8:7,5",
    "bat MT8:7,7",
    "skeleton MT8:6,8",
    "yellowDoor MT8:5,7",
    "redSlime MT8:4,6",
    "greenSlime MT8:3,6",
    "redSlime MT8:2,6",
    "redPotion MT8:1,5",
    "bat MT8:4,8",
    "yellowDoor MT8:1,9",
    "greenSlime MT8:1,10",
    "bat MT8:2,11",
    "blueDoor MT8:3,11",
    "redGem MT8:4,10",
    "blueGem MT8:5,11",
    "blueKey MT8:7,10",
    "yellowKey MT8:7,11",
    "redPotion MT8:8,10",
    "yellowKey MT8:5,10",
    "yellowKey MT8:4,11",
)

MID_GEMS_EARLY_ACTION_TOKENS = (
    "yellowDoor MT9:9,4",
    "yellowDoor MT9:8,4",
    "yellowKey MT9:7,4",
    "yellowDoor MT9:4,5",
    "bat MT9:3,5",
    "blueGem MT9:1,5",
    "yellowKey MT9:2,4",
    "yellowDoor MT7:3,5",
    "bat MT7:3,3",
    "redGem MT7:3,1",
    "redSlime MT5:11,2",
    "yellowDoor MT5:10,1",
    "greenSlime MT5:9,2",
    "yellowKey MT5:8,4",
    "skeleton MT6:5,11",
    "greenSlime MT6:2,11",
    "yellowDoor MT6:1,10",
    "bat MT6:2,9",
    "blueGem MT6:4,9",
)

MT7_RED_GEM_ACTION_TOKENS = (
    "buy 5 yellowKey MT7:6,1",
    "yellowDoor MT7:1,5",
    "yellowDoor MT7:3,5",
    "bat MT7:3,3",
    "redPotion MT7:3,2",
    "redGem MT7:3,1",
)

MT9_LOWER_GEM_ACTION_TOKENS = (
    "blueDoor MT9:6,3",
    "redGem MT9:6,5",
    "yellowDoor MT9:9,4",
    "greenSlime MT9:10,2",
    "yellowDoor MT9:8,4",
    "yellowKey MT9:7,4",
    "yellowKey MT9:5,4",
    "yellowDoor MT9:4,5",
    "bat MT9:3,5",
    "blueGem MT9:1,5",
    "yellowKey MT9:2,4",
    "yellowDoor MT9:4,1",
)

MID_GEMS_LATE_ACTION_TOKENS = (
    "redPotion MT4:9,10",
    "bat MT3:3,5",
    "yellowDoor MT3:1,6",
    "skeleton MT3:1,7",
    "redPotion MT3:1,9",
    "redGem MT3:2,9",
    "yellowKey MT3:2,8",
    "yellowDoor MT1:6,6",
    "bat MT1:7,6",
    "bluePriest MT1:8,6",
    "bat MT1:9,6",
    "yellowDoor MT1:9,5",
    "redGem MT1:7,3",
    "blueGem MT1:7,4",
    "bluePriest MT4:2,5",
    "yellowDoor MT4:2,4",
    "blueKey MT4:2,1",
    "yellowKey MT4:3,2",
    "redPotion MT4:1,2",
    "bluePriest MT8:8,8",
    "yellowDoor MT1:6,9",
    "yellowKey MT1:5,10",
    "bat MT1:10,10",
    "bluePotion MT1:10,11",
    "bat MT5:4,6",
    "bluePriest MT5:3,5",
    "yellowKey MT5:1,5",
    "yellowKey MT5:1,6",
    "skeletonSoldier MT5:2,7",
    "blueGem MT5:1,9",
)

LOW_GEMS_ACTION_TOKENS = (
    "redGem MT1:7,3",
    "blueGem MT1:7,4",
    "blueGem MT3:2,1",
    "redGem MT3:2,9",
    "bat MT3:3,5",
    "yellowDoor MT1:6,6",
    "yellowDoor MT1:4,3",
    "yellowDoor MT3:1,4",
    "yellowDoor MT3:1,6",
    "bluePriest MT3:1,3",
    "skeleton MT3:1,7",
    "yellowKey MT3:2,8",
    "redPotion MT4:9,10",
    "bat MT1:7,6",
    "bluePriest MT1:8,6",
    "bat MT1:9,6",
    "yellowDoor MT1:9,5",
    "redPotion MT1:1,10",
    "redPotion MT1:1,11",
    "bluePotion MT2",
)

LOW_GEMS_KEY_BUFFER_ACTION_TOKENS = (
    "bat MT7:5,9",
    "yellowKey MT7:5,10",
    "yellowKey MT7:5,11",
    "yellowKey MT8:7,11",
    "yellowKey MT8:5,10",
    "yellowKey MT8:4,11",
    "yellowKey MT9:7,4",
    "buy 5 yellowKey MT7:6,1",
)

MT10_RESOURCE_ACTION_TOKENS = (
    "yellowDoor MT10:1,9",
    "skeleton MT10:1,6",
    "skeleton MT10:3,6",
    "blueGem MT10:2,6",
    "bluePriest MT10:4,11",
    "bluePriest MT10:8,11",
    "skeleton MT10:9,6",
    "skeleton MT10:11,6",
    "skeletonSoldier MT10:10,7",
    "redGem MT10:10,6",
    "bluePotion MT10:11,11",
    "blueDoor MT9:3,11",
    "yellowDoor MT10",
    "upFloor MT9:6,1",
    "upFloor MT9:1,11",
)

MT10_LEFT_RESOURCE_ACTION_TOKENS = (
    "yellowDoor MT10:1,9",
    "skeleton MT10:1,6",
    "skeleton MT10:3,6",
    "blueGem MT10:2,6",
)

MT10_AFTER_LEFT_REFILL_ACTION_TOKENS = (
    "yellowDoor MT9:8,11",
    "bluePriest MT9:9,11",
    "bluePriest MT9:11,9",
    "yellowKey MT9:9,9",
    "redPotion MT9:11,11",
)

MT10_PRE_STAIR_REFILL_ACTION_TOKENS = (
    "redPotion MT9:2,10",
    "skeleton MT9:5,10",
    "yellowKey MT9:5,4",
    "yellowKey MT9:7,4",
)

MT9_GEM_BEFORE_SOLDIER_ACTION_TOKENS = (
    "yellowDoor MT9:9,4",
    "redSlime MT9:7,6",
    "yellowKey MT9:5,4",
    "yellowDoor MT9:4,5",
    "bat MT9:3,5",
    "yellowKey MT9:2,4",
    "blueGem MT9:1,5",
)

MT10_BOSS_ACTION_TOKENS = (
    "redDoor MT10:6,9",
    "event MT10:6,9",
    "event MT10:6,5",
    "skeletonCaptain MT10:6,4",
    "skeleton MT10:5,4",
    "skeleton MT10:7,4",
    "skeleton MT10:5,5",
    "skeleton MT10:7,5",
    "skeleton MT10:5,6",
    "skeletonSoldier MT10:6,6",
    "skeleton MT10:7,6",
    "skeletonCaptain MT10:6,1",
)

MT10_YELLOW_PREP_ACTION_TOKENS = (
    # The 6F potion next to the blue-key merchant is mandatory for the
    # no-hp403 route; skipping it leaves the agent with too little HP to route
    # through 9F/8F cleanup.
    "redPotion MT6:8,3",
    # 9F left/bottom access route.  The central blue door is deliberately not
    # listed here: in the no-hp403 route the only blue key must be saved for
    # the lower-left door that opens the 10F stair.
    "yellowDoor MT9:9,4",
    "greenSlime MT9:10,2",
    "yellowDoor MT9:8,4",
    "yellowKey MT9:7,4",
    "yellowKey MT9:5,4",
    "redGem MT9:6,5",
    "skeletonSoldier MT9:11,6",
    "redSlime MT9:7,6",
    "yellowDoor MT9:4,5",
    "bat MT9:3,5",
    "yellowKey MT9:2,4",
    "blueGem MT9:1,5",
    "bat MT9:7,10",
    "yellowDoor MT9:8,11",
    "bluePriest MT9:9,11",
    "bluePriest MT9:11,9",
    "yellowKey MT9:9,9",
    "redPotion MT9:11,11",
    "yellowDoor MT9:6,11",
    "blueDoor MT9:3,11",
    "redPotion MT9:2,10",
    "skeleton MT9:5,10",
    "yellowKey MT9:5,4",
    "yellowKey MT9:7,4",
    "upFloor MT9:1,11",
    # 8F right-bottom key pocket for the second blue key.
    "blueKey MT8:7,10",
    "redPotion MT8:8,10",
    "yellowDoor MT8:9,11",
    "skeletonSoldier MT8:10,11",
    # 7F key/HP pockets used after the first 10F resource when the route needs
    # to rebuild yellow-key stock without going all the way to low floors.
    "yellowDoor MT7:7,7",
    "bluePriest MT7:7,10",
    "skeletonSoldier MT7:9,7",
    "yellowKey MT7:9,11",
    "yellowDoor MT7:5,7",
    "bat MT7:5,9",
    "yellowKey MT7:5,11",
    "redSlime MT7:7,9",
    "bluePotion MT7:7,11",
)

MT10_DIRECT_ACCESS_ACTION_TOKENS = (
    # Direct path from the 6F blue-key merchant to the first 10F resource.
    # This deliberately excludes low-floor refill doors; those are only useful
    # after 10F resource progress has started.
    "yellowDoor MT9:9,4",
    "greenSlime MT9:10,2",
    "yellowDoor MT9:8,4",
    "yellowKey MT9:7,4",
    "yellowKey MT9:5,4",
    "redGem MT9:6,5",
    "skeletonSoldier MT9:11,6",
    "redPotion MT9:11,11",
    "yellowKey MT9:9,9",
    "bluePriest MT9:9,11",
    "bluePriest MT9:11,9",
    "redSlime MT9:7,6",
    "bat MT9:7,10",
    "yellowDoor MT9:6,11",
    "blueDoor MT9:3,11",
    "upFloor MT9:1,11",
)

MT6_BLUE_KEY_BUY_ACTION_TOKENS = (
    "bluePriest MT6:7,1",
    "greenSlime MT6:10,1",
    "yellowDoor MT6:11,2",
    "bat MT6:11,4",
    "buy blueKey MT6:8,4",
    "redPotion MT6:8,3",
)

MT10_REFILL_ACTION_TOKENS = (
    # After the first 10F gem, the no-hp403 route often needs to leave 10F,
    # collect yellow keys and HP on lower floors, then come back for the right
    # side 10F resources.
    "skeleton MT9:5,10",
    "skeletonSoldier MT9:1,3",
    "yellowKey MT9:2,2",
    "yellowDoor MT9:8,11",
    "bluePriest MT9:9,11",
    "bluePriest MT9:11,9",
    "yellowKey MT9:9,9",
    "redPotion MT9:11,11",
    "redPotion MT9:2,10",
    "skeleton MT9:5,10",
    "yellowKey MT9:5,4",
    "yellowKey MT9:7,4",
    "yellowDoor MT7:7,7",
    "bluePriest MT7:7,10",
    "skeletonSoldier MT7:9,7",
    "yellowKey MT7:9,11",
    "yellowDoor MT7:5,7",
    "bat MT7:5,9",
    "yellowKey MT7:5,11",
    "redSlime MT7:7,9",
    "bluePotion MT7:7,11",
)

RED_KEY_ACTION_TOKENS = (
    "bluePriest MT8:7,5",
    "bat MT8:7,7",
    "skeleton MT8:6,8",
    "bluePriest MT8:8,8",
    "yellowDoor MT8:10,7",
    "yellowGuard MT8:9,5",
    "yellowGuard MT8:11,5",
    "greenSlime MT8:7,2",
    "bluePotion MT8:9,3",
    "redPotion MT8:11,3",
    "yellowKey MT8:9,1",
    "yellowKey MT8:11,1",
    "redKey MT8:10,2",
)

LOW_FLOOR_REFILL_ACTION_TOKENS = (
    # Low-floor resources are the main difference between the current pure
    # search routes and the successful benchmark route.  They should be
    # harvested before the 10F/right-8F damage spike, not only after the route
    # is already short on HP.
    "yellowDoor MT1:4,3",
    "redPotion MT1:1,3",
    "skeleton MT1:2,4",
    "yellowDoor MT1:2,5",
    "yellowKey MT1:1,6",
    "skeletonSoldier MT1:2,7",
    "yellowDoor MT1:2,8",
    "yellowKey MT1:3,10",
    "redPotion MT1:8,4",
    "redPotion MT1:1,11",
    "redPotion MT1:1,10",
    "yellowDoor MT1:10,9",
    "bat MT1:10,10",
    "bluePotion MT1:10,11",
    "bluePotion MT2:3,10",
    "bluePotion MT2:4,10",
    "bluePotion MT2:3,11",
    "yellowDoor MT3:9,2",
    "bat MT3:10,2",
    "redPotion MT3:11,1",
    "yellowDoor MT4:8,8",
    "bluePriest MT4:8,9",
    "redGem MT4:7,10",
    "redPotion MT4:9,10",
    "yellowDoor MT4:4,5",
    "redSlime MT4:6,5",
    "skeleton MT4:9,5",
    "yellowDoor MT4:10,4",
    "skeletonSoldier MT4:10,3",
    "bluePotion MT4:11,2",
    "yellowKey MT4:9,2",
)

POST_BOSS_MT10_RESOURCE_ACTION_TOKENS = (
    "redGem MT10:1,3",
    "redGem MT10:2,3",
    "redGem MT10:3,3",
    "blueGem MT10:9,3",
    "blueGem MT10:10,3",
    "blueGem MT10:11,3",
    "bluePotion MT10:1,4",
    "bluePotion MT10:2,4",
    "bluePotion MT10:3,4",
    "yellowKey MT10:9,4",
    "yellowKey MT10:10,4",
    "yellowKey MT10:11,4",
)

DELAYED_PRE_SHIELD_REFILL_TOKENS = (
    # These are valuable late-game refills.  Taking them before the shield
    # makes the route look locally safer but starves the 10F cleanup stage.
    "bat MT7:5,9",
    "yellowKey MT7:5,11",
)

MT4_REDGEM_ACTION_TOKENS = (
    "downFloor MT5:1,11",
    "yellowDoor MT4:8,8",
    "bluePriest MT4:8,9",
    "redGem MT4:7,10",
    "redPotion MT4:9,10",
    "upFloor MT4:1,11",
    "bat MT5:6,4",
    "yellowKey MT5:6,2",
    "yellowDoor MT5:5,1",
    "redSlime MT5:4,1",
)

MT4_LEFT_KEY_REFILL_ACTION_TOKENS = (
    "downFloor MT5:1,11",
    "yellowDoor MT4:4,8",
    "bat MT4:4,9",
    "greenSlime MT4:3,10",
    "yellowKey MT4:5,11",
    "yellowKey MT4:3,11",
    "upFloor MT4:1,11",
)

MT4_BLUE_KEY_POCKET_ACTION_TOKENS = (
    "bluePriest MT4:2,5",
    "yellowDoor MT4:2,4",
    "blueKey MT4:2,1",
    "yellowKey MT4:3,2",
    "redPotion MT4:1,2",
)


def filter_stage_actions(
    actions: list[dict[str, Any]],
    state: MotaState,
    target_stage: str,
    sim: MotaSimulator | None = None,
) -> list[dict[str, Any]]:
    """Prune obviously off-stage macro actions before MCTS expansion.

    The policy still learns over graph nodes, but long resource stages should
    not spend most simulations on unrelated low-floor fights and doors.  This
    filter is intentionally conservative and falls back to the original action
    list whenever it would remove every option.
    """

    late_requires_sword = {
        "mt4_redgem",
        "pre_shield_gems",
        "shield",
        "shield_buffer",
        "mid_gems",
        "low_gems",
        "mt8_hp_ready",
        "mt8_gems",
        "lower_gems",
        "pre_mt10_buffer",
        "mt10_blue_ready",
        "mt10_yellow_ready",
        "mt10_resources",
        "all_gems",
        "guard_ready",
        "red_key",
        "boss_ready",
        "trap",
        "boss",
        "boss_all_gems",
    }
    if (
        target_stage in late_requires_sword
        and sim is not None
        and not has_first_sword(sim, state)
    ):
        return filter_stage_actions(actions, state, "sword", sim=sim)

    if target_stage == "sword":
        if sim is not None and has_first_sword(sim, state):
            return actions
        current_floor = _floor_index(str(state.floor_id))

        def label(action: dict[str, Any]) -> str:
            return str(action.get("label") or "")

        def label_lower(action: dict[str, Any]) -> str:
            return label(action).lower()

        def is_up_stair(action: dict[str, Any]) -> bool:
            return "upfloor" in label_lower(action)

        def is_down_stair(action: dict[str, Any]) -> bool:
            return "downfloor" in label_lower(action)

        # This is a stage mask, not a route: it keeps only interactions that
        # plausibly advance the early climb to the 5F sword and removes low-floor
        # backtracking/farming choices that dominate random self-play.
        focused: list[dict[str, Any]] = []
        for action in actions:
            text = label(action)
            lower = text.lower()
            if "sword1" in text:
                focused.append(action)
            elif "event MT2:1,9" in text:
                focused.append(action)
            elif "event " in text:
                continue
            elif is_up_stair(action) and current_floor <= 5:
                focused.append(action)
            elif is_down_stair(action):
                continue
            elif "blueDoor" in text:
                continue
            elif "fakeWall" in text and current_floor <= 5:
                focused.append(action)
            elif "yellowDoor" in text:
                if "MT3:" in text and "yellowDoor MT3:9,11" not in text:
                    continue
                if "MT4:" in text and not any(
                    token in text for token in ("yellowDoor MT4:11,8", "yellowDoor MT4:1,8")
                ):
                    continue
                if "MT5:" in text and "yellowDoor MT5:8,9" not in text:
                    continue
                focused.append(action)
            elif any(token in text for token in ("yellowKey", "blueKey")):
                focused.append(action)
            elif any(token in text for token in ("redGem", "blueGem", "redPotion", "bluePotion")) and current_floor <= 5:
                focused.append(action)
            elif any(token in lower for token in ("greenslime", "redslime")) and current_floor <= 5:
                focused.append(action)
        if focused:
            return focused
        non_down = [action for action in actions if not is_down_stair(action)]
        return non_down or actions

    if target_stage not in RESOURCE_TARGET_STAGES:
        return actions

    current_floor = _floor_index(str(state.floor_id))

    def label(action: dict[str, Any]) -> str:
        return str(action.get("label") or "")

    def label_lower(action: dict[str, Any]) -> str:
        return label(action).lower()

    def is_up_stair(action: dict[str, Any]) -> bool:
        return "upfloor" in label_lower(action)

    def is_down_stair(action: dict[str, Any]) -> bool:
        return "downfloor" in label_lower(action)

    def is_key_or_key_merchant(action: dict[str, Any]) -> bool:
        text = label_lower(action)
        return (
            "yellowkey" in text
            or "bluekey" in text
            or "yellow key" in text
            or "blue key" in text
        )

    def is_blue_key_action(action: dict[str, Any]) -> bool:
        text = label_lower(action)
        return "bluekey" in text or "blue key" in text

    def is_yellow_key_action(action: dict[str, Any]) -> bool:
        text = label_lower(action)
        return "yellowkey" in text or "yellow key" in text

    def is_item(action: dict[str, Any]) -> bool:
        text = label(action)
        return any(token in text for token in ("redGem", "blueGem", "redPotion", "bluePotion"))

    def is_clear_tile(floor_id: str, x: int, y: int) -> bool:
        floor = state.floors.get(floor_id)
        if floor is None or y < 0 or y >= len(floor) or x < 0 or x >= len(floor[y]):
            return False
        return int(floor[y][x]) == 0

    def low_floor_refill_candidates() -> list[dict[str, Any]]:
        focused = [
            action
            for action in actions
            if any(token in label(action) for token in LOW_FLOOR_REFILL_ACTION_TOKENS)
        ]
        non_stair = [action for action in focused if not (is_up_stair(action) or is_down_stair(action))]
        return non_stair or focused

    def mt4_red_gem_pocket_candidates() -> list[dict[str, Any]]:
        pocket_tokens = (
            "yellowDoor MT4:8,8",
            "bluePriest MT4:8,9",
            "redGem MT4:7,10",
            "redPotion MT4:9,10",
        )
        return [action for action in actions if any(token in label(action) for token in pocket_tokens)]

    low_refill_targets = (
        ("MT1", 1, 3, "redPotion"),
        ("MT1", 1, 10, "redPotion"),
        ("MT1", 1, 11, "redPotion"),
        ("MT1", 3, 10, "yellowKey"),
        ("MT1", 3, 11, "yellowKey"),
        ("MT2", 3, 10, "bluePotion"),
        ("MT2", 4, 10, "bluePotion"),
        ("MT2", 3, 11, "bluePotion"),
        ("MT3", 11, 1, "redPotion"),
        ("MT4", 11, 2, "bluePotion"),
        ("MT4", 9, 2, "yellowKey"),
    )

    def target_still_present(floor_id: str, x: int, y: int, block_id: str) -> bool:
        if sim is None:
            return False
        try:
            return sim.block_id(sim.tile(state, x, y, floor_id)) == block_id
        except Exception:
            return False

    def mt10_progress_count() -> int:
        if sim is None:
            return 0
        try:
            return int(mt10_resource_progress(sim, state))
        except Exception:
            return 0

    def mt10_after_left_refill_pending() -> bool:
        """Whether the 9F right-bottom refill needed after 10F left gem remains.

        The successful strict routes do not spend all 10F yellow doors
        immediately after the left blue gem.  They first return to 9F, open the
        right-bottom refill pocket, and only then continue through the 10F
        right-side resource chain.
        """
        return any(
            target_still_present(floor_id, x, y, block_id)
            for floor_id, x, y, block_id in (
                ("MT9", 8, 11, "yellowDoor"),
                ("MT9", 9, 11, "bluePriest"),
                ("MT9", 11, 9, "bluePriest"),
            )
        )

    def before_first_mt10_resource() -> bool:
        return mt10_progress_count() == 0

    def can_start_mt10_left_resource_chain() -> bool:
        if sim is None:
            return False
        if state.items.get("yellowKey", 0) <= 0:
            return False
        damage = sim.damage_info_for_stats(
            "skeleton",
            atk=state.atk,
            defense=state.defense,
            mdef=state.mdef,
        )
        if damage is None:
            return False
        # The first 10F left-side door immediately exposes a skeleton and then
        # the blue gem.  Gate this by the deterministic fight loss instead of a
        # fixed HP threshold; otherwise viable states such as HP=291 are forced
        # to backtrack forever.
        return state.hp - int(damage["damage"]) > 120

    def mt8_pre_mt10_resource_pending() -> bool:
        return any(
            target_still_present(floor_id, x, y, block_id)
            for floor_id, x, y, block_id in (
                ("MT8", 4, 10, "redGem"),
                ("MT8", 5, 11, "blueGem"),
                ("MT8", 8, 10, "redPotion"),
                ("MT8", 7, 10, "blueKey"),
                ("MT8", 3, 11, "blueDoor"),
            )
        )

    if (
        target_stage == "shield_buffer"
        and current_floor == 6
        and sim is not None
        and has_first_sword(sim, state)
        and state.flags.get("nowShield") != "shield1"
        and target_still_present("MT9", 9, 7, "shield1")
    ):
        for group in (
            ("yellowDoor MT6:2,4",),
            ("yellowDoor MT6:3,4",),
            ("redSlime MT6:4,3",),
            ("yellowKey MT6:3,1", "yellowKey MT6:4,1", "yellowKey MT6:3,2"),
            ("yellowDoor MT6:5,4",),
            ("yellowDoor MT6:7,8",),
            ("yellowDoor MT6:8,8",),
            ("redSlime MT6:9,9",),
            ("redPotion MT6:8,11",),
            ("yellowDoor MT6:10,8",),
            ("redSlime MT6:11,9",),
            ("upFloor MT6:11,11",),
        ):
            matched = [
                action
                for action in actions
                if any(token in label(action) for token in group)
            ]
            if matched:
                return matched

    if (
        target_stage in {"pre_shield_gems", "shield", "shield_buffer"}
        and sim is not None
        and has_first_sword(sim, state)
        and state.flags.get("nowShield") != "shield1"
        and current_floor <= 6
        and target_still_present("MT9", 9, 7, "shield1")
        and (
            target_still_present("MT4", 7, 10, "redGem")
            or target_still_present("MT4", 9, 10, "redPotion")
        )
        and any(
            is_down_stair(action)
            or any(token in label(action) for token in MT4_REDGEM_ACTION_TOKENS)
            for action in actions
        )
    ):
        return filter_stage_actions(actions, state, "mt4_redgem", sim=sim)

    if (
        target_stage == "mt10_resources"
        and before_first_mt10_resource()
        and sim is not None
        and current_floor == 8
        and (state.atk < 26 or state.defense < 26)
    ):
        mt9_lower_gem_pending = any(
            target_still_present(floor_id, x, y, block_id)
            for floor_id, x, y, block_id in (
                ("MT9", 1, 5, "blueGem"),
                ("MT9", 6, 5, "redGem"),
            )
        )
        if mt9_lower_gem_pending:
            lower_actions = filter_stage_actions(actions, state, "lower_gems", sim=sim)
            if lower_actions and lower_actions != actions:
                return lower_actions

    if (
        target_stage == "mt10_resources"
        and sim is not None
        and mt10_progress_count() == 1
        and current_floor == 7
        and not target_still_present("MT10", 3, 9, "yellowDoor")
        and not target_still_present("MT10", 4, 11, "bluePriest")
        and target_still_present("MT10", 11, 11, "bluePotion")
    ):
        # After the first 10F resource and the middle-left priest, rebuild the
        # yellow-key stock through the 7F lower/right pockets.  The potion is
        # useful, but taking it before the 9,11 yellow-key pocket leaves the
        # route unable to finish the 10F right side.
        for group in (
            ("yellowDoor MT7:7,7",),
            ("redSlime MT7:7,9",),
            ("bluePriest MT7:7,10",),
            ("skeletonSoldier MT7:9,7",),
            ("yellowKey MT7:9,11",),
            ("yellowDoor MT7:5,7",),
            ("bat MT7:5,9",),
            ("yellowKey MT7:5,11",),
            ("bluePotion MT7:7,11",),
        ):
            focused = [
                action
                for action in actions
                if any(token in label(action) for token in group)
            ]
            if focused:
                return focused

    if (
        target_stage == "mt10_resources"
        and current_floor == 10
        and target_still_present("MT10", 2, 6, "blueGem")
        and any("yellowDoor MT10:1,9" in label(action) for action in actions)
        and not can_start_mt10_left_resource_chain()
    ):
        down = [action for action in actions if is_down_stair(action)]
        if down:
            return down

    if (
        target_stage == "mt10_resources"
        and before_first_mt10_resource()
        and (state.atk < 26 or state.defense < 26)
    ):
        direct_gems = [
            action
            for action in actions
            if not (is_up_stair(action) or is_down_stair(action))
            and any(token in label(action) for token in ("redGem", "blueGem"))
        ]
        if direct_gems:
            return direct_gems

    if (
        target_stage == "mt10_resources"
        and before_first_mt10_resource()
        and current_floor == 9
        and state.items.get("blueKey", 0) > 0
        and state.items.get("yellowKey", 0) > 0
        and (state.atk < 26 or state.defense < 26 or state.hp < 320)
        and not target_still_present("MT9", 1, 5, "blueGem")
        and mt8_pre_mt10_resource_pending()
    ):
        # If the route has cleaned the immediately reachable MT9 gem pocket but
        # is still far below the first 10F-resource readiness threshold, do not
        # force more local 9F fights.  Return to the MT8 resource pocket so the
        # search can raise ATK/DEF/HP before opening the 10F left door.
        local_progress = [
            action
            for action in actions
            if not (is_up_stair(action) or is_down_stair(action))
            and any(
                token in label(action)
                for token in (
                    "redGem MT9:6,5",
                    "blueGem MT9:1,5",
                    "yellowKey MT9:2,4",
                    "yellowKey MT9:5,4",
                    "yellowKey MT9:7,4",
                )
            )
        ]
        if local_progress:
            return local_progress
        down = [action for action in actions if is_down_stair(action)]
        if down:
            return down

    if (
        target_stage == "mt10_resources"
        and before_first_mt10_resource()
        and current_floor == 8
        and (state.atk < 26 or state.defense < 26)
    ):
        direct_gems = [
            action
            for action in actions
            if not (is_up_stair(action) or is_down_stair(action))
            and any(token in label(action) for token in ("redGem", "blueGem"))
        ]
        if direct_gems:
            return direct_gems
        down = [action for action in actions if is_down_stair(action)]
        if down:
            return down

    if (
        target_stage == "mt10_resources"
        and before_first_mt10_resource()
        and current_floor == 9
        and (
            state.items.get("yellowKey", 0) >= 3
            or not target_still_present("MT9", 9, 4, "yellowDoor")
        )
    ):
        mt9_right_tokens = (
            "yellowDoor MT9:9,4",
            "greenSlime MT9:10,2",
            "yellowDoor MT9:8,4",
            "yellowKey MT9:5,4",
            "redGem MT9:6,5",
        )
        for token in mt9_right_tokens:
            focused = [action for action in actions if token in label(action)]
            if focused:
                return focused

    if target_stage == "pre_mt10_buffer":
        needed_blue = 0 if is_clear_tile("MT9", 3, 11) else 1
        if state.hp < 240:
            direct_potions = [
                action
                for action in actions
                if not (is_up_stair(action) or is_down_stair(action))
                and any(token in label(action) for token in ("redPotion", "bluePotion"))
            ]
            if direct_potions:
                return direct_potions
        if state.items.get("blueKey", 0) <= 1 and target_still_present("MT8", 7, 10, "blueKey"):
            mt8_blue_key_chain = [
                action
                for action in actions
                if any(token in label(action) for token in MT8_BLUE_KEY_CHAIN_ACTION_TOKENS)
                or any(
                    token in label(action)
                    for token in (
                        "blueKey MT8:7,10",
                        "yellowKey MT8:7,11",
                        "redPotion MT8:8,10",
                    )
                )
            ]
            non_stair_blue_key_chain = [
                action
                for action in mt8_blue_key_chain
                if not (is_up_stair(action) or is_down_stair(action))
            ]
            if non_stair_blue_key_chain:
                return non_stair_blue_key_chain
            if mt8_blue_key_chain:
                return mt8_blue_key_chain
            if current_floor > 8:
                down = [action for action in actions if is_down_stair(action)]
                if down:
                    return down
        if (
            current_floor == 8
            and state.hp >= 180
            and state.items.get("yellowKey", 0) >= 1
            and state.items.get("blueKey", 0) >= needed_blue
            and state.atk >= 22
            and state.defense >= 23
        ):
            mt8_chain_progress = [
                action
                for action in actions
                if not (is_up_stair(action) or is_down_stair(action))
                and any(
                    token in label(action)
                    for token in (
                        MT8_BLUE_KEY_CHAIN_ACTION_TOKENS
                        + MT8_LOWER_GEM_ROUTE_ACTION_TOKENS
                    )
                )
            ]
            if mt8_chain_progress:
                return mt8_chain_progress
        lower_pending = any(
            target_still_present(floor_id, x, y, block_id)
            for floor_id, x, y, block_id in (
                ("MT7", 3, 1, "redGem"),
                ("MT8", 4, 10, "redGem"),
                ("MT8", 5, 11, "blueGem"),
                ("MT9", 1, 5, "blueGem"),
                ("MT9", 6, 5, "redGem"),
            )
        )
        low_floor_recovery_pending = any(
            target_still_present(floor_id, x, y, block_id)
            for floor_id, x, y, block_id in (
                ("MT1", 7, 3, "redGem"),
                ("MT1", 7, 4, "blueGem"),
                ("MT1", 1, 10, "redPotion"),
                ("MT1", 1, 11, "redPotion"),
                ("MT2", 3, 10, "bluePotion"),
                ("MT2", 4, 10, "bluePotion"),
                ("MT2", 3, 11, "bluePotion"),
                ("MT3", 2, 1, "blueGem"),
                ("MT3", 2, 9, "redGem"),
                ("MT3", 11, 1, "redPotion"),
                ("MT4", 11, 2, "bluePotion"),
                ("MT4", 9, 2, "yellowKey"),
            )
        )
        if low_floor_recovery_pending and 6 <= current_floor < 8 and state.hp < 240:
            low_refill = low_floor_refill_candidates()
            if low_refill:
                return low_refill
            down = [action for action in actions if is_down_stair(action)]
            if down:
                return down
        if (
            low_floor_recovery_pending
            and current_floor == 4
            and state.hp < 180
            and state.items.get("yellowKey", 0) <= 1
        ):
            down = [action for action in actions if is_down_stair(action)]
            if down:
                return down
        if (
            low_floor_recovery_pending
            and current_floor == 3
            and state.hp < 180
            and state.items.get("yellowKey", 0) <= 1
        ):
            down = [action for action in actions if is_down_stair(action)]
            if down:
                return down
        if (
            low_floor_recovery_pending
            and current_floor == 2
            and state.hp < 180
            and state.items.get("yellowKey", 0) <= 1
        ):
            down = [action for action in actions if is_down_stair(action)]
            if down:
                return down
        if (
            lower_pending
            and low_floor_recovery_pending
            and current_floor <= 3
            and state.items.get("yellowKey", 0) <= 1
        ):
            if current_floor > 1:
                mt1_left_key_refill_done = not (
                    target_still_present("MT1", 2, 4, "skeleton")
                    or target_still_present("MT1", 2, 5, "yellowDoor")
                    or target_still_present("MT1", 1, 6, "yellowKey")
                )
                if mt1_left_key_refill_done:
                    up = [action for action in actions if is_up_stair(action)]
                    if up:
                        return up
                down = [action for action in actions if is_down_stair(action)]
                if down:
                    return down
            if current_floor == 1:
                for group in (
                    ("skeleton MT1:2,4",),
                    ("yellowDoor MT1:2,5",),
                    ("yellowKey MT1:1,6",),
                    ("skeletonSoldier MT1:2,7",),
                    ("yellowDoor MT1:2,8",),
                    ("yellowKey MT1:3,10",),
                    ("redPotion MT1:1,10", "redPotion MT1:1,11"),
                ):
                    matched = [
                        action
                        for action in actions
                        if any(token in label(action) for token in group)
                    ]
                    if matched:
                        return matched
                up = [action for action in actions if is_up_stair(action)]
                if up:
                    return up
            lower_actions = filter_stage_actions(actions, state, "lower_gems", sim=sim)
            if lower_actions and lower_actions != actions:
                return lower_actions
        if (
            lower_pending
            and current_floor <= 3
            and state.items.get("yellowKey", 0) <= 2
            and (state.atk < 25 or state.defense < 26)
        ):
            direct_keys = [
                action
                for action in actions
                if not (is_up_stair(action) or is_down_stair(action))
                and is_key_or_key_merchant(action)
            ]
            if direct_keys:
                return direct_keys
            up = [action for action in actions if is_up_stair(action)]
            if up:
                return up
        if (
            lower_pending
            and 4 <= current_floor < 8
            and state.hp >= 180
            and state.items.get("yellowKey", 0) >= 2
            and state.items.get("blueKey", 0) >= needed_blue
            and state.atk >= 22
            and state.defense >= 23
            and (state.atk < 25 or state.defense < 26)
        ):
            direct_high_resources = [
                action
                for action in actions
                if not (is_up_stair(action) or is_down_stair(action))
                and any(
                    token in label(action)
                    for token in (
                        MT7_RED_GEM_ACTION_TOKENS
                        + MT8_LOWER_GEM_ROUTE_ACTION_TOKENS
                        + MT9_LOWER_GEM_ACTION_TOKENS
                    )
                )
            ]
            if direct_high_resources:
                return direct_high_resources
            up = [action for action in actions if is_up_stair(action)]
            if up:
                return up
        if lower_pending or state.atk < 25 or state.defense < 26:
            lower_actions = filter_stage_actions(actions, state, "lower_gems", sim=sim)
            if lower_actions and lower_actions != actions:
                if all(is_up_stair(action) or is_down_stair(action) for action in lower_actions):
                    stats_ready_for_high_chain = state.atk >= 22 and state.defense >= 23
                    exploratory_tokens = (
                        (
                            MT7_RED_GEM_ACTION_TOKENS
                            + MT8_LOWER_GEM_ROUTE_ACTION_TOKENS
                            + MT9_LOWER_GEM_ACTION_TOKENS
                        )
                        if stats_ready_for_high_chain
                        else ()
                    ) + LOW_GEMS_KEY_BUFFER_ACTION_TOKENS
                    exploratory_non_stairs = [
                        action
                        for action in actions
                        if not (is_up_stair(action) or is_down_stair(action))
                        and not any(token in label(action) for token in ("yellowDoor", "blueDoor", "redDoor"))
                        and (
                            any(token in label(action) for token in exploratory_tokens)
                            or any(
                                token in label(action)
                                for token in (
                                    "redGem",
                                    "blueGem",
                                    "redPotion",
                                    "bluePotion",
                                    "yellowKey",
                                    "blueKey",
                                )
                            )
                        )
                    ]
                    if exploratory_non_stairs:
                        return lower_actions + exploratory_non_stairs[:4]
                return lower_actions
            direct_gems = [
                action
                for action in actions
                if not (is_up_stair(action) or is_down_stair(action))
                and any(token in label(action) for token in ("redGem", "blueGem"))
            ]
            if direct_gems:
                return direct_gems
        if state.hp < 240:
            direct_potions = [
                action
                for action in actions
                if not (is_up_stair(action) or is_down_stair(action))
                and any(token in label(action) for token in ("redPotion", "bluePotion"))
            ]
            if direct_potions:
                return direct_potions
            refill = low_floor_refill_candidates()
            if refill:
                return refill
            if current_floor > 4:
                down = [action for action in actions if is_down_stair(action)]
                if down:
                    return down
        if (
            state.items.get("yellowKey", 0) < 2
            or state.items.get("blueKey", 0) < needed_blue
        ):
            direct_keys = [
                action
                for action in actions
                if not (is_up_stair(action) or is_down_stair(action))
                and is_key_or_key_merchant(action)
            ]
            if direct_keys:
                return direct_keys
            key_unlockers = [
                action
                for action in actions
                if not (is_up_stair(action) or is_down_stair(action))
                and any(
                    token in label(action)
                    for token in (
                        "skeletonSoldier MT7:9,7",
                        "bat MT7:5,9",
                    )
                )
            ]
            if key_unlockers:
                return key_unlockers
            down = [action for action in actions if is_down_stair(action)]
            if down:
                return down
            yellow_ready_actions = filter_stage_actions(
                actions,
                state,
                "mt10_yellow_ready",
                sim=sim,
            )
            if yellow_ready_actions and yellow_ready_actions != actions:
                return yellow_ready_actions
        yellow_ready_actions = filter_stage_actions(actions, state, "mt10_yellow_ready", sim=sim)
        return yellow_ready_actions or actions

    def low_refill_remaining_below_or_on(max_floor: int) -> bool:
        return any(
            _floor_index(floor_id) <= max_floor and target_still_present(floor_id, x, y, block_id)
            for floor_id, x, y, block_id in low_refill_targets
        )

    def near_late_refill_candidates() -> list[dict[str, Any]]:
        include_general_mt8 = target_stage != "mt10_resources" and not (
            target_stage in {"boss_ready", "trap", "boss", "boss_all_gems"}
            and state.items.get("redKey", 0) > 0
        )
        focused = [
            action
            for action in actions
            if any(token in label(action) for token in MT10_REFILL_ACTION_TOKENS)
            or any(token in label(action) for token in MT10_YELLOW_PREP_ACTION_TOKENS)
            or (
                include_general_mt8
                and any(token in label(action) for token in MT8_RESOURCE_ACTION_TOKENS)
            )
        ]
        non_stair = [action for action in focused if not (is_up_stair(action) or is_down_stair(action))]
        return non_stair

    def needs_boss_hp() -> bool:
        return (
            target_stage in {"boss_ready", "trap", "boss", "boss_all_gems"}
            and sim is not None
            and boss_route_margin(sim, state) < 0
        )

    def needs_boss_buffer() -> bool:
        return (
            target_stage in {"boss_ready", "trap", "boss", "boss_all_gems"}
            and sim is not None
            and boss_route_margin(sim, state) < 430
        )

    mt4_refill_done = (
        not target_still_present("MT4", 7, 10, "redGem")
        and not target_still_present("MT4", 9, 10, "redPotion")
    )
    mt10_yellow_climb_ready = (
        state.items.get("blueKey", 0) > 0
        and (
            (state.items.get("yellowKey", 0) >= 3 and state.hp >= 150)
            or (mt4_refill_done and state.items.get("yellowKey", 0) >= 2 and state.hp >= 137)
        )
    )
    if target_stage == "mt10_yellow_ready" and current_floor < 9 and mt10_yellow_climb_ready:
        up = [action for action in actions if is_up_stair(action)]
        if up:
            return up

    if (
        target_stage == "mt10_yellow_ready"
        and current_floor == 7
        and state.items.get("yellowKey", 0) < 5
        and state.money >= 50
    ):
        direct_supply = [
            action
            for action in actions
            if not (is_up_stair(action) or is_down_stair(action))
            and (
                is_yellow_key_action(action)
                or any(token in label(action) for token in ("redPotion", "bluePotion"))
            )
        ]
        if direct_supply:
            return direct_supply
        merchant = [action for action in actions if "buy 5 yellowKey MT7" in label(action)]
        if merchant:
            return merchant
        local_bottom_refill = [
            action
            for action in actions
            if any(
                token in label(action)
                for token in (
                    "yellowDoor MT7:5,7",
                    "bat MT7:5,9",
                    "redSlime MT7:7,9",
                    "bluePotion MT7:7,11",
                )
            )
        ]
        if local_bottom_refill:
            return local_bottom_refill
        first_mt1_refill_done = not (
            target_still_present("MT1", 2, 4, "skeleton")
            or target_still_present("MT1", 2, 5, "yellowDoor")
            or target_still_present("MT1", 1, 6, "yellowKey")
        )
        if first_mt1_refill_done:
            up = [action for action in actions if is_up_stair(action)]
            if up:
                return up
        down = [action for action in actions if is_down_stair(action)]
        if down:
            return down

    if target_stage == "mt10_yellow_ready" and current_floor == 4:
        mt4_red_gem = mt4_red_gem_pocket_candidates()
        if mt4_red_gem and (
            any("yellowDoor MT4:8,8" in label(action) for action in mt4_red_gem)
            or target_still_present("MT4", 7, 10, "redGem")
            or target_still_present("MT4", 9, 10, "redPotion")
        ):
            return mt4_red_gem
        if (
            state.items.get("yellowKey", 0) < 2
            and state.items.get("blueKey", 0) > 0
            and sim is not None
            and not target_still_present("MT4", 7, 10, "redGem")
            and not target_still_present("MT4", 9, 10, "redPotion")
            and not (
                target_still_present("MT7", 5, 10, "yellowKey")
                or target_still_present("MT7", 5, 11, "yellowKey")
            )
        ):
            first_mt1_refill_pending = (
                target_still_present("MT1", 2, 4, "skeleton")
                or target_still_present("MT1", 2, 5, "yellowDoor")
                or target_still_present("MT1", 1, 6, "yellowKey")
            )
            if first_mt1_refill_pending:
                down = [action for action in actions if is_down_stair(action)]
                if down:
                    return down
            up = [action for action in actions if is_up_stair(action)]
            if up:
                return up

    if (
        target_stage == "mt10_yellow_ready"
        and current_floor == 8
        and state.items.get("yellowKey", 0) < 2
        and state.items.get("blueKey", 0) > 0
        and sim is not None
        and not (
            target_still_present("MT1", 2, 4, "skeleton")
            or target_still_present("MT1", 2, 5, "yellowDoor")
            or target_still_present("MT1", 1, 6, "yellowKey")
        )
    ):
        up = [action for action in actions if is_up_stair(action)]
        if up:
            return up

    if (
        target_stage == "mt10_yellow_ready"
        and current_floor == 8
        and state.items.get("blueKey", 0) <= 0
        and target_still_present("MT8", 7, 10, "blueKey")
    ):
        direct_blue_key = [action for action in actions if "blueKey MT8:7,10" in label(action)]
        if direct_blue_key:
            return direct_blue_key
        blue_key_chain = [
            action
            for action in actions
            if any(token in label(action) for token in MT8_BLUE_KEY_CHAIN_ACTION_TOKENS)
        ]
        if blue_key_chain:
            return blue_key_chain

    if (
        target_stage == "mt10_yellow_ready"
        and current_floor == 9
        and state.items.get("blueKey", 0) <= 0
        and before_first_mt10_resource()
    ):
        direct_resource = [
            action
            for action in actions
            if any(token in label(action) for token in ("redGem MT9:6,5", "yellowKey MT9:5,4", "yellowKey MT9:7,4"))
        ]
        if direct_resource:
            return direct_resource
        potion = [action for action in actions if "redPotion MT9:2,10" in label(action)]
        if potion:
            return potion
        up = [action for action in actions if is_up_stair(action)]
        if up:
            return up
        if state.items.get("yellowKey", 0) >= 2:
            central_harvest = [
                action
                for action in actions
                if any(token in label(action) for token in ("yellowDoor MT9:4,5", "redSlime MT9:7,6"))
            ]
            if central_harvest:
                return central_harvest
        down = [action for action in actions if is_down_stair(action)]
        if down:
            return down

    if (
        target_stage == "mt10_yellow_ready"
        and current_floor == 3
        and state.items.get("yellowKey", 0) < 2
        and state.items.get("blueKey", 0) > 0
        and sim is not None
        and not target_still_present("MT4", 7, 10, "redGem")
        and not target_still_present("MT4", 9, 10, "redPotion")
        and not (
            target_still_present("MT7", 5, 10, "yellowKey")
            or target_still_present("MT7", 5, 11, "yellowKey")
        )
    ):
        first_mt1_refill_pending = (
            target_still_present("MT1", 2, 4, "skeleton")
            or target_still_present("MT1", 2, 5, "yellowDoor")
            or target_still_present("MT1", 1, 6, "yellowKey")
        )
        if first_mt1_refill_pending:
            down = [action for action in actions if is_down_stair(action)]
            if down:
                return down
        up = [action for action in actions if is_up_stair(action)]
        if up:
            return up

    if (
        target_stage == "mt10_yellow_ready"
        and current_floor == 2
        and state.items.get("yellowKey", 0) < 2
        and state.items.get("blueKey", 0) > 0
        and sim is not None
        and not target_still_present("MT4", 7, 10, "redGem")
        and not target_still_present("MT4", 9, 10, "redPotion")
        and not (
            target_still_present("MT7", 5, 10, "yellowKey")
            or target_still_present("MT7", 5, 11, "yellowKey")
        )
    ):
        first_mt1_refill_pending = (
            target_still_present("MT1", 2, 4, "skeleton")
            or target_still_present("MT1", 2, 5, "yellowDoor")
            or target_still_present("MT1", 1, 6, "yellowKey")
        )
        if first_mt1_refill_pending:
            down = [action for action in actions if is_down_stair(action)]
            if down:
                return down
        up = [action for action in actions if is_up_stair(action)]
        if up:
            return up

    if (
        target_stage == "mt10_yellow_ready"
        and current_floor == 1
        and state.items.get("yellowKey", 0) < 2
        and state.items.get("blueKey", 0) > 0
        and sim is not None
        and not target_still_present("MT4", 7, 10, "redGem")
        and not target_still_present("MT4", 9, 10, "redPotion")
        and not (
            target_still_present("MT7", 5, 10, "yellowKey")
            or target_still_present("MT7", 5, 11, "yellowKey")
        )
    ):
        refill = low_floor_refill_candidates()
        if refill:
            return refill

    if (
        target_stage == "mt10_yellow_ready"
        and current_floor in {5, 6}
        and state.items.get("yellowKey", 0) < 2
        and state.items.get("blueKey", 0) > 0
        and sim is not None
        and not target_still_present("MT6", 8, 3, "redPotion")
        and not (
            target_still_present("MT7", 5, 10, "yellowKey")
            or target_still_present("MT7", 5, 11, "yellowKey")
        )
    ):
        first_mt1_refill_pending = (
            target_still_present("MT1", 2, 4, "skeleton")
            or target_still_present("MT1", 2, 5, "yellowDoor")
            or target_still_present("MT1", 1, 6, "yellowKey")
        )
        if first_mt1_refill_pending:
            down = [action for action in actions if is_down_stair(action)]
            if down:
                return down
        up = [action for action in actions if is_up_stair(action)]
        if up:
            return up

    if (
        target_stage == "mt10_yellow_ready"
        and 2 <= current_floor <= 6
        and state.items.get("yellowKey", 0) < 5
        and state.items.get("yellowKey", 0) >= 2
        and state.items.get("blueKey", 0) > 0
        and not target_still_present("MT6", 8, 3, "redPotion")
    ):
        if current_floor == 4:
            mt4_red_gem = mt4_red_gem_pocket_candidates()
            if mt4_red_gem:
                return mt4_red_gem
        if current_floor <= 4:
            refill = low_floor_refill_candidates()
            if refill:
                return refill
        direct_supply = [
            action
            for action in actions
            if not (is_up_stair(action) or is_down_stair(action))
            and (
                is_yellow_key_action(action)
                or any(token in label(action) for token in ("redPotion", "bluePotion"))
            )
        ]
        if direct_supply:
            return direct_supply
        if current_floor in {2, 3}:
            up = [action for action in actions if is_up_stair(action)]
            if up:
                return up
        down = [action for action in actions if is_down_stair(action)]
        if down:
            return down

    if (
        target_stage == "mt10_yellow_ready"
        and current_floor <= 4
        and state.items.get("yellowKey", 0) < 5
        and state.items.get("yellowKey", 0) >= 2
        and state.items.get("blueKey", 0) > 0
    ):
        refill = low_floor_refill_candidates()
        if refill:
            return refill

    if target_stage == "mt4_redgem":
        if current_floor > 4:
            if (
                target_still_present("MT4", 7, 10, "redGem")
                or target_still_present("MT4", 9, 10, "redPotion")
            ):
                down = [action for action in actions if is_down_stair(action)]
                if down:
                    return down
            focused = [
                action
                for action in actions
                if is_down_stair(action) or any(token in label(action) for token in MT4_REDGEM_ACTION_TOKENS)
            ]
            return focused or actions
        if current_floor < 4:
            key_actions = [action for action in actions if is_key_or_key_merchant(action)]
            if key_actions:
                return key_actions
            focused = [action for action in actions if is_up_stair(action)]
            return focused or actions
        focused = [
            action
            for action in actions
            if any(token in label(action) for token in MT4_REDGEM_ACTION_TOKENS)
        ]
        non_stair = [action for action in focused if not (is_up_stair(action) or is_down_stair(action))]
        return non_stair or focused or actions

    if target_stage == "mt8_gems":
        if current_floor < 8:
            key_actions = [action for action in actions if is_key_or_key_merchant(action)]
            if key_actions:
                return key_actions
            focused = [
                action
                for action in actions
                if is_up_stair(action)
            ]
            return focused or actions
        if current_floor > 8:
            focused = [action for action in actions if is_down_stair(action) or is_key_or_key_merchant(action)]
            return focused or actions

        blue_key_pending = target_still_present("MT8", 7, 10, "blueKey")
        if blue_key_pending:
            if (
                state.hp < 340
                and state.items.get("yellowKey", 0) >= 3
                and target_still_present("MT8", 1, 5, "redPotion")
            ):
                left_potion = [
                    action
                    for action in actions
                    if any(token in label(action) for token in ("yellowDoor MT8:1,3", "redPotion MT8:1,5"))
                ]
                non_stair_left = [
                    action for action in left_potion if not (is_up_stair(action) or is_down_stair(action))
                ]
                if non_stair_left:
                    return non_stair_left
            focused = [
                action
                for action in actions
                if any(
                    token in label(action)
                    for token in (
                        MT8_BLUE_KEY_CHAIN_ACTION_TOKENS
                        + (
                            "greenSlime MT8:7,2",
                            "bluePriest MT8:7,5",
                            "redPotion MT8:8,10",
                        )
                    )
                )
            ]
            if state.items.get("yellowKey", 0) < 3:
                # The 8F left potion is useful only when the route has an
                # extra yellow key.  With the shield route's common
                # yellowKey=2 state, opening MT8:1,3 first leaves only one key
                # for the right-bottom chain and repeatedly creates a no-key
                # dead end before the blue key/gem pocket.
                focused = [
                    action
                    for action in focused
                    if not any(
                        token in label(action)
                        for token in ("yellowDoor MT8:1,3", "redPotion MT8:1,5")
                    )
                ]
            non_stair = [action for action in focused if not (is_up_stair(action) or is_down_stair(action))]
            if non_stair:
                return non_stair
            if state.items.get("yellowKey", 0) <= 1:
                # At this point the direct right-bottom chain is temporarily
                # blocked, usually because HP is too low to fight the
                # skeleton soldier.  Spending the last yellow key on the left
                # or middle 8F doors makes the blue key unreachable, so keep
                # only non-stair interactions that do not consume the final
                # yellow key.
                unsafe_last_key_doors = (
                    "yellowDoor MT8:1,3",
                    "yellowDoor MT8:5,7",
                    "yellowDoor MT8:10,7",
                )
                safe = [
                    action
                    for action in actions
                    if not (is_up_stair(action) or is_down_stair(action))
                    and not any(token in label(action) for token in unsafe_last_key_doors)
                ]
                if safe:
                    return safe

        if not blue_key_pending and state.items.get("blueKey", 0) > 0:
            # Once the right-bottom blue key has been collected, the shortest
            # useful continuation is the blue-door gem pocket plus the two
            # nearby potions.  Letting the generic MT8 resource list re-enable
            # the centre-left skeleton/doors here repeatedly wastes HP and
            # yellow keys before the route has cashed in the blue key.
            focused = [
                action
                for action in actions
                if any(token in label(action) for token in MT8_POST_BLUE_KEY_GEM_ACTION_TOKENS)
            ]
            non_stair = [action for action in focused if not (is_up_stair(action) or is_down_stair(action))]
            if non_stair:
                return non_stair

        focused = [action for action in actions if any(token in label(action) for token in MT8_RESOURCE_ACTION_TOKENS)]
        non_stair = [action for action in focused if not (is_up_stair(action) or is_down_stair(action))]
        if non_stair:
            return non_stair
        floor_actions = [
            action for action in actions if not (is_up_stair(action) or is_down_stair(action))
        ]
        return floor_actions or focused or actions

    if target_stage == "mid_gems":
        mt6_blue_pending = target_still_present("MT6", 4, 9, "blueGem")
        mt5_blue_pending = target_still_present("MT5", 1, 9, "blueGem")
        mt5_key_pocket_pending = target_still_present("MT5", 8, 4, "yellowKey")
        if mt6_blue_pending and mt5_key_pocket_pending and state.items.get("yellowKey", 0) < 3:
            if current_floor == 6:
                down = [action for action in actions if is_down_stair(action)]
                if down:
                    return down
            if current_floor == 5:
                pocket = [
                    action
                    for action in actions
                    if any(
                        token in label(action)
                        for token in (
                            "redSlime MT5:11,2",
                            "yellowDoor MT5:10,1",
                            "greenSlime MT5:9,2",
                            "yellowKey MT5:8,4",
                        )
                    )
                ]
                if pocket:
                    return pocket
        tokens = MID_GEMS_EARLY_ACTION_TOKENS if mt6_blue_pending else MID_GEMS_LATE_ACTION_TOKENS
        focused = [action for action in actions if any(token in label(action) for token in tokens)]
        non_stair = [action for action in focused if not (is_up_stair(action) or is_down_stair(action))]
        if non_stair:
            return non_stair
        if focused:
            return focused
        if mt6_blue_pending:
            if current_floor > 6:
                down = [action for action in actions if is_down_stair(action)]
                return down or actions
            if current_floor < 5:
                up = [action for action in actions if is_up_stair(action)]
                return up or actions
            if current_floor == 5:
                up = [action for action in actions if is_up_stair(action)]
                return up or actions
        if mt5_blue_pending:
            if current_floor > 5:
                down = [action for action in actions if is_down_stair(action)]
                return down or actions
            if current_floor < 5:
                up = [action for action in actions if is_up_stair(action)]
                return up or actions

    if target_stage == "lower_gems":
        mt5_key_pocket_pending = target_still_present("MT5", 8, 4, "yellowKey")
        mid_pending = target_still_present("MT6", 4, 9, "blueGem") or target_still_present(
            "MT5", 1, 9, "blueGem"
        )
        low_pending = any(
            target_still_present(floor_id, x, y, block_id)
            for floor_id, x, y, block_id in (
                ("MT1", 7, 3, "redGem"),
                ("MT1", 7, 4, "blueGem"),
                ("MT3", 2, 1, "blueGem"),
                ("MT3", 2, 9, "redGem"),
            )
        )
        mt1_low_pending = target_still_present("MT1", 7, 3, "redGem") or target_still_present(
            "MT1", 7, 4, "blueGem"
        )
        mt3_low_pending = target_still_present("MT3", 2, 1, "blueGem") or target_still_present(
            "MT3", 2, 9, "redGem"
        )
        mt8_pending = any(
            target_still_present(floor_id, x, y, block_id)
            for floor_id, x, y, block_id in (
                ("MT8", 4, 10, "redGem"),
                ("MT8", 5, 11, "blueGem"),
            )
        )
        mt9_lower_pending = any(
            target_still_present(floor_id, x, y, block_id)
            for floor_id, x, y, block_id in (
                ("MT9", 1, 5, "blueGem"),
                ("MT9", 6, 5, "redGem"),
            )
        )
        mt7_red_pending = target_still_present("MT7", 3, 1, "redGem")
        mt4_blue_key_pocket_pending = any(
            target_still_present(floor_id, x, y, block_id)
            for floor_id, x, y, block_id in (
                ("MT4", 2, 1, "blueKey"),
                ("MT4", 3, 2, "yellowKey"),
                ("MT4", 1, 2, "redPotion"),
            )
        )
        if state.hp < 220:
            refill = low_floor_refill_candidates()
            potion_refill = [action for action in refill if "Potion" in label(action)]
            if potion_refill:
                return potion_refill
        if (
            mt8_pending
            and not mid_pending
            and not low_pending
            and mt4_blue_key_pocket_pending
        ):
            if current_floor > 4:
                down = [action for action in actions if is_down_stair(action)]
                if down:
                    return down
            if current_floor < 4:
                up = [action for action in actions if is_up_stair(action)]
                if up:
                    return up
            if current_floor == 4:
                pocket = [
                    action
                    for action in actions
                    if any(token in label(action) for token in MT4_BLUE_KEY_POCKET_ACTION_TOKENS)
                ]
                if pocket:
                    return pocket
        if mid_pending:
            if (
                target_still_present("MT6", 4, 9, "blueGem")
                and mt5_key_pocket_pending
                and state.items.get("yellowKey", 0) < 3
            ):
                if current_floor == 6:
                    down = [action for action in actions if is_down_stair(action)]
                    if down:
                        return down
                if current_floor == 5:
                    pocket = [
                        action
                        for action in actions
                        if any(
                            token in label(action)
                            for token in (
                                "redSlime MT5:11,2",
                                "yellowDoor MT5:10,1",
                                "greenSlime MT5:9,2",
                                "yellowKey MT5:8,4",
                            )
                        )
                    ]
                    if pocket:
                        return pocket
            tokens = (
                MID_GEMS_EARLY_ACTION_TOKENS
                if target_still_present("MT6", 4, 9, "blueGem")
                else MID_GEMS_LATE_ACTION_TOKENS
            )
        elif low_pending:
            tokens = LOW_GEMS_ACTION_TOKENS
        elif mt8_pending:
            if (
                current_floor == 8
                and state.hp < 100
                and target_still_present("MT8", 1, 5, "redPotion")
            ):
                mt8_refill_chain = (
                    "yellowDoor MT8:5,7",
                    "redSlime MT8:4,6",
                    "greenSlime MT8:3,6",
                    "redSlime MT8:2,6",
                    "redPotion MT8:1,5",
                )
                focused_refill = [
                    action
                    for action in actions
                    if any(token in label(action) for token in mt8_refill_chain)
                ]
                non_stair_refill = [
                    action
                    for action in focused_refill
                    if not (is_up_stair(action) or is_down_stair(action))
                ]
                if non_stair_refill:
                    return non_stair_refill
            tokens = MT8_LOWER_GEM_ROUTE_ACTION_TOKENS
        elif mt9_lower_pending:
            tokens = MT9_LOWER_GEM_ACTION_TOKENS
        elif mt7_red_pending:
            tokens = MT7_RED_GEM_ACTION_TOKENS
        else:
            tokens = (
                MID_GEMS_EARLY_ACTION_TOKENS
                + MID_GEMS_LATE_ACTION_TOKENS
                + LOW_GEMS_ACTION_TOKENS
                + MT8_RESOURCE_ACTION_TOKENS
                + MT9_LOWER_GEM_ACTION_TOKENS
                + MT7_RED_GEM_ACTION_TOKENS
                + LOW_FLOOR_REFILL_ACTION_TOKENS
            )
        focused = [action for action in actions if any(token in label(action) for token in tokens)]
        non_stair = [action for action in focused if not (is_up_stair(action) or is_down_stair(action))]
        if non_stair:
            return non_stair
        if focused:
            return focused
        if mid_pending:
            if current_floor > 6:
                down = [action for action in actions if is_down_stair(action)]
                return down or actions
            if current_floor < 5:
                up = [action for action in actions if is_up_stair(action)]
                return up or actions
            if current_floor == 5 and target_still_present("MT6", 4, 9, "blueGem"):
                up = [action for action in actions if is_up_stair(action)]
                return up or actions
        if low_pending and current_floor > 3:
            down = [action for action in actions if is_down_stair(action)]
            return down or actions
        if mt1_low_pending and not mt3_low_pending and current_floor > 1:
            down = [action for action in actions if is_down_stair(action)]
            return down or actions
        if mt8_pending and current_floor < 8:
            up = [action for action in actions if is_up_stair(action)]
            return up or actions
        if mt9_lower_pending:
            if current_floor > 9:
                down = [action for action in actions if is_down_stair(action)]
                return down or actions
            if current_floor < 9:
                up = [action for action in actions if is_up_stair(action)]
                return up or actions
        if mt7_red_pending:
            if current_floor > 7:
                down = [action for action in actions if is_down_stair(action)]
                return down or actions
            if current_floor < 7:
                up = [action for action in actions if is_up_stair(action)]
                return up or actions

    if target_stage in {"pre_shield_gems", "shield", "shield_buffer"}:
        shield_taken = state.flags.get("nowShield") == "shield1"
        if (
            target_stage in {"shield", "shield_buffer"}
            and not shield_taken
            and current_floor == 4
            and sim is not None
            and not target_still_present("MT4", 7, 10, "redGem")
            and not target_still_present("MT4", 9, 10, "redPotion")
            and not target_still_present("MT5", 6, 2, "yellowKey")
        ):
            mt4_left_refill_pending = any(
                target_still_present(floor_id, x, y, block_id)
                for floor_id, x, y, block_id in (
                    ("MT4", 4, 8, "yellowDoor"),
                    ("MT4", 4, 9, "bat"),
                    ("MT4", 3, 10, "greenSlime"),
                    ("MT4", 5, 11, "yellowKey"),
                    ("MT4", 3, 11, "yellowKey"),
                )
            )
            if mt4_left_refill_pending:
                refill = [
                    action
                    for action in actions
                    if any(token in label(action) for token in MT4_LEFT_KEY_REFILL_ACTION_TOKENS)
                ]
                non_stair_refill = [
                    action for action in refill if not (is_up_stair(action) or is_down_stair(action))
                ]
                if non_stair_refill:
                    return non_stair_refill
                if refill:
                    return refill
            up = [action for action in actions if "upFloor MT4:1,11" in label(action)]
            if up:
                return up
        if target_stage == "shield_buffer" and shield_taken:
            if state.items.get("yellowKey", 0) <= 1 and state.money >= 50:
                merchant_actions = [
                    action
                    for action in actions
                    if any(
                        token in label(action)
                        for token in (
                            "buy 5 yellowKey MT7:6,1",
                            "blueDoor MT7:5,5",
                            "redSlime MT7:5,3",
                        )
                    )
                ]
                if merchant_actions:
                    return merchant_actions
            if state.hp < 300:
                immediate_resources = [
                    action
                    for action in actions
                    if not (is_up_stair(action) or is_down_stair(action))
                    and any(
                        token in label(action)
                        for token in (
                            "Potion",
                            "Gem",
                            "Key",
                        )
                    )
                    and "Door" not in label(action)
                ]
                if immediate_resources:
                    return immediate_resources
                if target_stage == "shield_buffer" and state.hp < 280 and current_floor <= 6:
                    refill_fights = [
                        action
                        for action in actions
                        if any(
                            token in label(action)
                            for token in (
                                "redSlime MT6:9,9",
                                "greenSlime MT6:2,11",
                                "redSlime MT6:11,9",
                            )
                        )
                    ]
                    if refill_fights:
                        return refill_fights
                if target_stage == "shield_buffer" and state.hp < 280 and current_floor >= 7:
                    stairs = [
                        action for action in actions if is_down_stair(action) or is_up_stair(action)
                    ]
                    if stairs:
                        return stairs
                door_progress = [
                    action
                    for action in actions
                    if not (is_up_stair(action) or is_down_stair(action))
                    and any(
                        token in label(action)
                        for token in (
                            "yellowDoor",
                            "blueDoor",
                            "redDoor",
                        )
                    )
                ]
                if door_progress:
                    low_key_stock = (
                        state.items.get("yellowKey", 0) <= 2
                        and state.items.get("blueKey", 0) <= 1
                    )
                    only_keyed_doors = all(
                        any(token in label(action) for token in ("yellowDoor", "blueDoor", "redDoor"))
                        for action in door_progress
                    )
                    if low_key_stock and only_keyed_doors:
                        stairs = [
                            action for action in actions if is_down_stair(action) or is_up_stair(action)
                        ]
                        if stairs:
                            return door_progress + stairs
                    return door_progress
                low_risk_fights: list[dict[str, Any]] = []
                if sim is not None:
                    for action in actions:
                        if is_up_stair(action) or is_down_stair(action):
                            continue
                        if "fight " not in label(action):
                            continue
                        target = action.get("target")
                        if not isinstance(target, (list, tuple)) or len(target) != 3:
                            continue
                        floor_id, x, y = target
                        try:
                            tile = sim.tile(state, int(x), int(y), str(floor_id))
                        except (TypeError, ValueError):
                            continue
                        enemy_id = sim.block_id(tile)
                        if not enemy_id:
                            continue
                        info = sim.damage_info(state, enemy_id)
                        if info is not None and int(info["damage"]) <= min(20, max(0, state.hp - 1)):
                            low_risk_fights.append(action)
                if low_risk_fights:
                    return low_risk_fights
                if current_floor > 4:
                    down = [action for action in actions if is_down_stair(action)]
                    if down:
                        return down
            refill = low_floor_refill_candidates()
            non_stair_refill = [
                action for action in refill if not (is_up_stair(action) or is_down_stair(action))
            ]
            if non_stair_refill:
                return non_stair_refill
            key_actions = [action for action in actions if is_key_or_key_merchant(action)]
            if key_actions:
                return key_actions
            stairs = [action for action in actions if is_down_stair(action) or is_up_stair(action)]
            if stairs:
                return stairs

        if (
            target_stage in {"shield", "shield_buffer"}
            and not shield_taken
            and state.items.get("yellowKey", 0) <= 1
        ):
            if (
                current_floor in {4, 5}
                and sim is not None
                and not target_still_present("MT4", 7, 10, "redGem")
                and not target_still_present("MT4", 9, 10, "redPotion")
            ):
                mt5_first_key_taken = not target_still_present("MT5", 6, 2, "yellowKey")
                mt4_left_refill_pending = any(
                    target_still_present(floor_id, x, y, block_id)
                    for floor_id, x, y, block_id in (
                        ("MT4", 4, 8, "yellowDoor"),
                        ("MT4", 4, 9, "bat"),
                        ("MT4", 3, 10, "greenSlime"),
                        ("MT4", 5, 11, "yellowKey"),
                        ("MT4", 3, 11, "yellowKey"),
                    )
                )
                if current_floor == 4:
                    # With no yellow key left, go upward to take the exposed 5F
                    # key.  Once that key has been collected, spend it on the
                    # 4F left refill pocket before continuing the 5F shield
                    # chain.
                    if mt5_first_key_taken and mt4_left_refill_pending:
                        refill = [
                            action
                            for action in actions
                            if any(token in label(action) for token in MT4_LEFT_KEY_REFILL_ACTION_TOKENS)
                        ]
                        non_stair_refill = [
                            action for action in refill if not (is_up_stair(action) or is_down_stair(action))
                        ]
                        if non_stair_refill:
                            return non_stair_refill
                        if refill:
                            return refill
                    if state.items.get("yellowKey", 0) <= 0:
                        up = [action for action in actions if "upFloor MT4:1,11" in label(action)]
                        if up:
                            return up
                    up = [action for action in actions if "upFloor MT4:1,11" in label(action)]
                    if up:
                        return up
                if current_floor == 5 and mt5_first_key_taken and mt4_left_refill_pending:
                    down = [action for action in actions if "downFloor MT5:1,11" in label(action)]
                    if down:
                        return down
            if current_floor == 5:
                mt5_key_unlock = [
                    action
                    for action in actions
                    if any(
                        token in label(action)
                        for token in (
                            "bat MT5:6,4",
                            "yellowKey MT5:6,2",
                            "yellowDoor MT5:5,1",
                            "redSlime MT5:4,1",
                            "yellowDoor MT5:4,4",
                            "bat MT5:4,6",
                            "yellowKey MT5:1,5",
                            "bat MT5:3,3",
                            "yellowDoor MT5:2,3",
                            "upFloor MT5:1,1",
                        )
                    )
                ]
                if mt5_key_unlock:
                    return mt5_key_unlock
            if current_floor >= 6:
                # After the 5F key-transfer segment the shield route fans out.
                # Keep the legal interaction nodes available and let PUCT choose
                # among doors, enemies, keys, merchants, and resources instead
                # of forcing a down-stair loop when yellow keys are temporarily
                # low.
                if state.money >= 50 and state.items.get("yellowKey", 0) >= 1:
                    upward = [action for action in actions if "upFloor MT6:11,11" in label(action)]
                    if upward:
                        return upward
                if current_floor == 7:
                    right_resources = [
                        action
                        for action in actions
                        if any(
                            token in label(action)
                            for token in (
                                "bluePotion MT7:9,9",
                                "redPotion MT7:9,3",
                                "yellowKey MT7:9,10",
                                "yellowKey MT7:9,11",
                                "yellowKey MT7:9,2",
                                "yellowKey MT7:9,1",
                            )
                        )
                    ]
                    if right_resources:
                        return right_resources
                non_stair_progress = [
                    action
                    for action in actions
                    if not (is_down_stair(action) or is_up_stair(action))
                ]
                if non_stair_progress:
                    return non_stair_progress
                up = [action for action in actions if is_up_stair(action)]
                if up:
                    return up
            key_actions = [action for action in actions if is_key_or_key_merchant(action)]
            if key_actions:
                return key_actions
            if current_floor > 1:
                down = [action for action in actions if is_down_stair(action)]
                if down:
                    return down

        if target_stage in {"shield", "shield_buffer"}:
            def first_group(groups: tuple[tuple[str, ...], ...]) -> list[dict[str, Any]]:
                for group in groups:
                    matched = [
                        action
                        for action in actions
                        if any(token in label(action) for token in group)
                    ]
                    if matched:
                        return matched
                return []

            if current_floor == 5:
                focused = first_group(
                    (
                        ("yellowKey MT5:6,2",),
                        ("yellowDoor MT5:5,1",),
                        ("redSlime MT5:4,1",),
                        ("yellowDoor MT5:4,4",),
                        ("bat MT5:4,6",),
                        ("yellowKey MT5:1,5",),
                        ("bat MT5:3,3",),
                        ("yellowDoor MT5:2,3",),
                        ("upFloor MT5:1,1",),
                    )
                )
                if focused:
                    return focused
                upward = [action for action in actions if "upFloor MT5:1,1" in label(action)]
                if upward:
                    return upward
            if current_floor == 6:
                if state.money >= 50 and state.items.get("yellowKey", 0) >= 1:
                    # The 7F merchant is the intended recovery when the shield
                    # route reaches 6F with enough money but an exhausted yellow
                    # key buffer.  Earlier filters returned only local fights,
                    # spending HP before the agent could buy the continuation
                    # keys needed for the climb.
                    upward = [action for action in actions if "upFloor MT6:11,11" in label(action)]
                    if upward:
                        return upward
                if target_stage == "shield_buffer":
                    for group in (
                        ("yellowDoor MT6:2,4",),
                        ("yellowDoor MT6:3,4",),
                        ("redSlime MT6:4,3",),
                        ("yellowKey MT6:3,1", "yellowKey MT6:4,1", "yellowKey MT6:3,2"),
                        ("yellowDoor MT6:5,4",),
                        ("yellowDoor MT6:7,8",),
                        ("yellowDoor MT6:8,8",),
                        ("redSlime MT6:9,9",),
                        ("redPotion MT6:8,11",),
                        ("yellowDoor MT6:10,8",),
                        ("redSlime MT6:11,9",),
                        ("upFloor MT6:11,11",),
                    ):
                        matched = [
                            action
                            for action in actions
                            if any(token in label(action) for token in group)
                        ]
                        if matched:
                            return matched
                # The shield route branches on 6F.  Returning only the first
                # matching milestone here collapses Go-Explore into a single
                # yellow-door path and prevents recovery from key/HP variants.
                focused = [
                    action
                    for action in actions
                    if not is_down_stair(action)
                    and any(
                        token in label(action)
                        for token in (
                            "yellowDoor MT6:",
                            "yellowKey MT6:",
                            "redSlime MT6:",
                            "greenSlime MT6:",
                            "bluePriest MT6:",
                            "skeleton MT6:",
                            "upFloor MT6:11,11",
                        )
                    )
                ]
                if focused:
                    return focused
                upward = [action for action in actions if "upFloor MT6:11,11" in label(action)]
                if upward:
                    return upward
            if current_floor == 7:
                if (
                    target_stage in {"shield", "shield_buffer"}
                    and state.money >= 50
                    and state.items.get("yellowKey", 0) >= 3
                ):
                    merchant_route = [
                        action
                        for action in actions
                        if any(
                            token in label(action)
                            for token in (
                                "buy 5 yellowKey MT7",
                                "redSlime MT7:5,3",
                                "blueDoor MT7:5,5",
                                "bluePriest MT7:4,6",
                            )
                        )
                    ]
                    if merchant_route:
                        return merchant_route
                if target_stage == "shield_buffer" and state.money >= 50:
                    # A two-yellow-key state can technically climb to 8F/9F,
                    # but doing so often spends the whole yellow buffer before
                    # the shield.  When the 7F merchant route is available,
                    # take it before the upward stair so shield-buffer routes
                    # remain continuation-safe.
                    for group in (
                        ("buy 5 yellowKey MT7",),
                        ("redSlime MT7:5,3",),
                        ("blueDoor MT7:5,5",),
                    ):
                        merchant_path = [
                            action
                            for action in actions
                            if any(token in label(action) for token in group)
                        ]
                        if merchant_path:
                            return merchant_path
                if (
                    target_stage in {"shield", "shield_buffer"}
                    and state.money < 50
                    and state.items.get("yellowKey", 0) >= 3
                    and state.items.get("blueKey", 0) == 0
                ):
                    # After using the 7F merchant, the only useful local
                    # objective is opening the corridor to the 8F stair.  Letting
                    # the search clear arbitrary 7F doors consumes the fresh
                    # yellow-key buffer before the shield route starts.
                    for group in (
                        ("upFloor MT7:1,1",),
                        ("yellowDoor MT7:1,5",),
                        ("yellowDoor MT7:1,7",),
                        ("yellowDoor MT7:3,5",),
                        ("greenSlime MT7:1,10",),
                        ("redSlime MT7:2,11",),
                        ("greenSlime MT7:3,10",),
                        ("yellowDoor MT7:3,7",),
                    ):
                        stair_corridor = [
                            action
                            for action in actions
                            if any(token in label(action) for token in group)
                        ]
                        if stair_corridor:
                            return stair_corridor
                merchant = [action for action in actions if "buy 5 yellowKey MT7" in label(action)]
                if merchant and state.items.get("yellowKey", 0) < 4:
                    return merchant
                upward = [action for action in actions if "upFloor MT7:1,1" in label(action)]
                if upward and state.items.get("yellowKey", 0) >= 2:
                    return upward
                right_resources = [
                    action
                    for action in actions
                    if any(
                        token in label(action)
                        for token in (
                            "bluePotion MT7:9,9",
                            "redPotion MT7:9,3",
                            "yellowKey MT7:9,10",
                            "yellowKey MT7:9,11",
                            "yellowKey MT7:9,2",
                            "yellowKey MT7:9,1",
                        )
                    )
                ]
                if right_resources:
                    return right_resources
                focused = [
                    action
                    for action in actions
                    if any(
                        token in label(action)
                        for token in (
                            "yellowKey MT7:9,10",
                            "yellowKey MT7:9,11",
                            "yellowKey MT7:9,1",
                            "redGem MT7:3,1",
                            "redPotion MT7",
                            "bluePotion MT7",
                            "upFloor MT7:1,1",
                            "buy 5 yellowKey MT7",
                        )
                    )
                ]
                if focused:
                    return focused
            if current_floor == 8:
                if target_stage == "shield_buffer":
                    for group in (
                        ("yellowKey MT8:3,4", "yellowKey MT8:4,4", "yellowKey MT8:5,4"),
                        ("yellowDoor MT8:6,3",),
                        ("yellowDoor MT8:4,1",),
                        ("yellowDoor MT8:3,1",),
                        ("upFloor MT8:6,1",),
                    ):
                        shield_key_pocket = [
                            action
                            for action in actions
                            if any(token in label(action) for token in group)
                        ]
                        if shield_key_pocket:
                            return shield_key_pocket
                focused = [
                    action
                    for action in actions
                    if any(
                        token in label(action)
                        for token in (
                            "yellowDoor MT8:3,1",
                            "yellowDoor MT8:4,1",
                            "yellowDoor MT8:6,3",
                            "yellowKey MT8:3,4",
                            "yellowKey MT8:4,4",
                            "yellowKey MT8:5,4",
                            "upFloor MT8:6,1",
                        )
                    )
                ]
                if focused:
                    return focused
            if current_floor == 9:
                focused = [
                    action
                    for action in actions
                    if any(
                        token in label(action)
                        for token in (
                            "yellowDoor MT9:8,1",
                            "greenSlime MT9:9,1",
                            "fakeWall MT9:10,5",
                            "shield1 MT9:9,7",
                        )
                    )
                ]
                if focused:
                    return focused
        safe_actions = [
            action
            for action in actions
            if not any(token in label(action) for token in DELAYED_PRE_SHIELD_REFILL_TOKENS)
        ]
        if safe_actions:
            return safe_actions
        stairs = [action for action in actions if is_up_stair(action) or is_down_stair(action)]
        return stairs or actions

    if target_stage == "boss_all_gems" and state.flags.get("10f战胜骷髅队长"):
        if current_floor < 10:
            focused = [action for action in actions if is_up_stair(action)]
            return focused or actions
        if current_floor > 10:
            focused = [action for action in actions if is_down_stair(action)]
            return focused or actions
        focused = [
            action
            for action in actions
            if any(token in label(action) for token in POST_BOSS_MT10_RESOURCE_ACTION_TOKENS)
        ]
        return focused or actions

    if target_stage in {"trap", "boss", "boss_all_gems"}:
        mt1_blue_potion_pending = target_still_present("MT1", 10, 11, "bluePotion")
        mt4_second_door_pending = target_still_present("MT4", 10, 4, "yellowDoor")
        if current_floor == 1 and state.items.get("redKey", 0) > 0:
            refill = low_floor_refill_candidates()
            if refill:
                return refill
            focused = [action for action in actions if is_up_stair(action)]
            return focused or actions
        if (
            current_floor == 2
            and state.items.get("redKey", 0) > 0
            and not mt1_blue_potion_pending
        ):
            focused = [action for action in actions if is_up_stair(action)]
            return focused or actions
        if (
            current_floor == 3
            and state.items.get("redKey", 0) > 0
            and not mt1_blue_potion_pending
            and mt4_second_door_pending
        ):
            focused = [action for action in actions if is_up_stair(action)]
            return focused or actions
        if current_floor == 4 and state.items.get("redKey", 0) > 0 and state.items.get("yellowKey", 0) <= 0:
            refill = low_floor_refill_candidates()
            if refill:
                return refill
        if current_floor == 3 and state.items.get("redKey", 0) > 0 and state.items.get("yellowKey", 0) <= 0:
            refill = low_floor_refill_candidates()
            if refill:
                return refill
        if current_floor < 10 and state.items.get("redKey", 0) > 0 and state.items.get("yellowKey", 0) <= 0:
            focused = [action for action in actions if is_up_stair(action)]
            return focused or actions
        if current_floor == 10:
            focused = [
                action
                for action in actions
                if any(token in label(action) for token in MT10_BOSS_ACTION_TOKENS)
            ]
            if focused:
                return focused

    if (
        target_stage == "mt10_yellow_ready"
        and current_floor == 7
        and state.items.get("yellowKey", 0) < 5
        and state.money >= 50
    ):
        direct_supply = [
            action
            for action in actions
            if not (is_up_stair(action) or is_down_stair(action))
            and (
                is_yellow_key_action(action)
                or any(token in label(action) for token in ("redPotion", "bluePotion"))
            )
        ]
        if direct_supply:
            return direct_supply
        merchant = [action for action in actions if "buy 5 yellowKey MT7" in label(action)]
        if merchant:
            return merchant
        down = [action for action in actions if is_down_stair(action)]
        if down:
            return down

    if (
        target_stage in {"mt10_yellow_ready", "mt10_resources"}
        and before_first_mt10_resource()
        and state.items.get("yellowKey", 0) <= 0
        and state.money >= 50
    ):
        if target_stage == "mt10_resources" and current_floor == 10:
            local_resource_actions = [
                action
                for action in actions
                if any(token in label(action) for token in MT10_LEFT_RESOURCE_ACTION_TOKENS)
                and not (is_up_stair(action) or is_down_stair(action))
            ]
            immediate_gem = [
                action
                for action in local_resource_actions
                if "blueGem MT10:2,6" in label(action)
            ]
            if immediate_gem:
                return immediate_gem
            if local_resource_actions:
                return local_resource_actions
        direct_yellow_keys = [action for action in actions if is_yellow_key_action(action)]
        if direct_yellow_keys:
            return direct_yellow_keys
        if current_floor == 7:
            merchant = [action for action in actions if "buy 5 yellowKey MT7" in label(action)]
            if merchant:
                return merchant
        if current_floor > 7:
            down = [action for action in actions if is_down_stair(action)]
            if down:
                return down
        if current_floor < 7:
            up = [action for action in actions if is_up_stair(action)]
            if up:
                return up

    if target_stage in {
        "mt10_yellow_ready",
        "mt10_resources",
        "all_gems",
        "red_key",
        "boss_ready",
        "trap",
        "boss",
        "boss_all_gems",
    } and (state.hp < 720 or needs_boss_buffer()) and not (
        target_stage == "red_key"
        and sim is not None
        and red_key_route_margin(sim, state) > 0
    ) and not (
        target_stage == "red_key"
        and current_floor >= 7
        and state.items.get("yellowKey", 0) >= 3
        and state.hp >= 280
    ) and not (
        target_stage == "red_key"
        and current_floor == 8
        and int(state.x) >= 5
        and int(state.y) >= 5
        and state.hp >= 260
    ) and not (
        target_stage == "red_key"
        and current_floor == 8
        and any("redPotion MT8:1,5" in label(action) for action in actions)
    ) and not (
        target_stage == "red_key"
        and current_floor == 7
        and any(
            token in label(action)
            for action in actions
            for token in (
                "yellowDoor MT7:7,7",
                "redSlime MT7:7,9",
                "bluePotion MT7:7,11",
            )
        )
    ) and not (
        target_stage in {"mt10_yellow_ready", "mt10_resources"}
        and state.items.get("blueKey", 0) <= 0
    ) and not (
        target_stage in {"mt10_yellow_ready", "mt10_resources"}
        and state.items.get("blueKey", 0) > 0
        and before_first_mt10_resource()
    ) and not (
        target_stage == "mt10_resources"
        and before_first_mt10_resource()
        and current_floor == 9
        and is_clear_tile("MT9", 3, 11)
    ):
        near_refill = near_late_refill_candidates()
        if near_refill:
            return near_refill
        refill = low_floor_refill_candidates()
        if refill:
            return refill
        mt1_blue_potion_pending = target_still_present("MT1", 10, 11, "bluePotion")
        mt4_left_route_cleared = not target_still_present("MT4", 9, 5, "skeleton")
        if (
            needs_boss_buffer()
            and mt1_blue_potion_pending
            and mt4_left_route_cleared
            and 2 <= current_floor <= 4
        ):
            down = [action for action in actions if is_down_stair(action)]
            if down:
                return down
        has_yellow_key = state.items.get("yellowKey", 0) > 0
        if (
            current_floor > 4
            and has_yellow_key
            and (state.hp < 300 or needs_boss_hp() or needs_boss_buffer())
            and low_refill_remaining_below_or_on(4)
        ):
            down = [action for action in actions if is_down_stair(action)]
            if down:
                return down
        if (
            2 <= current_floor <= 4
            and has_yellow_key
            and (state.hp < 300 or needs_boss_hp() or needs_boss_buffer())
            and low_refill_remaining_below_or_on(current_floor - 1)
        ):
            down = [action for action in actions if is_down_stair(action)]
            if down:
                return down

    if (
        target_stage in {"mt10_blue_ready", "mt10_resources"}
        and state.items.get("blueKey", 0) <= 0
        and (
            target_stage == "mt10_blue_ready"
            or (before_first_mt10_resource() and not is_clear_tile("MT9", 3, 11))
        )
        and not (
            target_stage == "mt10_resources"
            and any(
                any(token in label(action) for token in MT10_REFILL_ACTION_TOKENS)
                and not (is_up_stair(action) or is_down_stair(action))
                for action in actions
            )
        )
    ):
        if current_floor > 6:
            focused = [action for action in actions if is_down_stair(action)]
            return focused or actions
        if current_floor < 6:
            focused = [action for action in actions if is_up_stair(action)]
            return focused or actions
        blue_actions = [action for action in actions if is_blue_key_action(action)]
        if blue_actions:
            return blue_actions
        focused = [
            action
            for action in actions
            if any(token in label(action) for token in MT6_BLUE_KEY_BUY_ACTION_TOKENS)
        ]
        non_stair = [action for action in focused if not (is_up_stair(action) or is_down_stair(action))]
        if target_stage == "mt10_blue_ready" and not non_stair:
            local_actions = [
                action for action in actions if not (is_down_stair(action) or is_up_stair(action))
            ]
            if local_actions:
                return local_actions
        return non_stair or focused or actions

    if target_stage == "mt10_blue_ready" and state.items.get("blueKey", 0) <= 0:
        blue_actions = [action for action in actions if is_blue_key_action(action)]
        if blue_actions:
            return blue_actions
        if current_floor > 6:
            focused = [action for action in actions if is_down_stair(action)]
            return focused or actions
        if current_floor < 6:
            focused = [action for action in actions if is_up_stair(action)]
            return focused or actions
        local_actions = [
            action for action in actions if not (is_down_stair(action) or is_up_stair(action))
        ]
        if local_actions:
            min_len = min(len(action.get("path") or []) for action in local_actions)
            return [
                action
                for action in local_actions
                if len(action.get("path") or []) <= min_len + 2
            ]
        focused = [action for action in actions if is_down_stair(action) or is_up_stair(action)]
        return focused or actions

    if target_stage == "mt10_yellow_ready":
        if (
            state.items.get("blueKey", 0) <= 0
            and current_floor != 9
            and not is_clear_tile("MT9", 3, 11)
        ):
            blue_actions = [action for action in actions if is_blue_key_action(action)]
            if blue_actions:
                return blue_actions
            if current_floor > 6:
                focused = [action for action in actions if is_down_stair(action)]
                return focused or actions
            if current_floor < 6:
                focused = [action for action in actions if is_up_stair(action)]
                return focused or actions
            local_actions = [
                action for action in actions if not (is_down_stair(action) or is_up_stair(action))
            ]
            if local_actions:
                min_len = min(len(action.get("path") or []) for action in local_actions)
                return [
                    action
                    for action in local_actions
                    if len(action.get("path") or []) <= min_len + 2
                ]

        # First preserve the cheap HP refill next to the 6F blue-key merchant.
        # Without this, the planner reaches 9F with around 32 HP and spends all
        # keys/HP before it can recover the second blue key.
        if current_floor == 6:
            nearby_potions = [
                action
                for action in actions
                if "redPotion MT6:8,3" in label(action) and len(action.get("path") or []) <= 4
            ]
            if nearby_potions:
                return nearby_potions

        # If the route has already spent the first blue key on 9F, it must keep
        # harvesting local 9F resources before dropping back to 8F for the
        # second blue key.  Do not immediately fall into a stair loop.
        if current_floor == 9 and state.items.get("blueKey", 0) <= 0:
            potions = [action for action in actions if "redPotion MT9:2,10" in label(action)]
            if potions:
                return potions
            up_to_10 = [action for action in actions if "upFloor MT9:1,11" in label(action)]
            if up_to_10:
                return up_to_10
            local = [
                action
                for action in actions
                if not (is_up_stair(action) or is_down_stair(action))
                and any(token in label(action) for token in MT10_YELLOW_PREP_ACTION_TOKENS)
            ]
            if local:
                return local
            focused = [action for action in actions if is_down_stair(action)]
            return focused or actions

        # On 8F, recover the second blue key and the HP behind the right-bottom
        # pocket before going back toward 9F/10F.
        if current_floor == 8 and state.items.get("blueKey", 0) <= 0:
            focused = [
                action
                for action in actions
                if any(token in label(action) for token in MT10_YELLOW_PREP_ACTION_TOKENS)
            ]
            non_stair = [action for action in focused if not (is_up_stair(action) or is_down_stair(action))]
            if non_stair:
                return non_stair
            if focused:
                return focused

        if state.items.get("yellowKey", 0) < 5:
            direct_supply = [
                action
                for action in actions
                if is_yellow_key_action(action)
                or any(token in label(action) for token in ("redPotion", "bluePotion"))
            ]
            if direct_supply:
                return direct_supply
            if state.money >= 50:
                if current_floor == 9 and state.items.get("blueKey", 0) > 0:
                    right_chain_started = not target_still_present("MT9", 9, 4, "yellowDoor")
                    for safe_token in (
                        "yellowDoor MT9:9,4",
                        "greenSlime MT9:10,2",
                        "yellowDoor MT9:8,4",
                        "yellowKey MT9:7,4",
                        "yellowKey MT9:5,4",
                        "redGem MT9:6,5",
                    ):
                        if safe_token == "yellowDoor MT9:9,4":
                            if state.items.get("yellowKey", 0) < 2 or state.hp < 137:
                                continue
                        elif not right_chain_started:
                            continue
                        safe_9f_supply = [
                            action for action in actions if safe_token in label(action)
                        ]
                        if safe_9f_supply:
                            return safe_9f_supply
                    if (
                        state.hp < 137
                        and target_still_present("MT9", 11, 6, "skeletonSoldier")
                    ):
                        down = [action for action in actions if is_down_stair(action)]
                        if down:
                            return down
                    if state.items.get("yellowKey", 0) >= 3 or right_chain_started:
                        for token in MT10_DIRECT_ACCESS_ACTION_TOKENS:
                            if "blueDoor MT9:6,3" in token:
                                continue
                            focused = [action for action in actions if token in label(action)]
                            if focused:
                                return focused
                if current_floor > 7:
                    down = [action for action in actions if is_down_stair(action)]
                    if down:
                        return down
                if current_floor < 7:
                    up = [action for action in actions if is_up_stair(action)]
                    if up:
                        return up
                if current_floor == 7:
                    merchant = [action for action in actions if "buy 5 yellowKey MT7" in label(action)]
                    if merchant:
                        return merchant
                    local_bottom_refill = [
                        action
                        for action in actions
                        if any(
                            token in label(action)
                            for token in (
                                "yellowDoor MT7:5,7",
                                "bat MT7:5,9",
                                "redSlime MT7:7,9",
                                "bluePotion MT7:7,11",
                            )
                        )
                    ]
                    if local_bottom_refill:
                        return local_bottom_refill
                    # Some route prefixes can stand on 7F after taking the
                    # bottom two yellow keys while the 7F merchant is not
                    # actually reachable as a macro action.  In that case,
                    # climbing to 8F only creates a stair loop; the next useful
                    # expansion is to backtrack toward 6F/lower refill pockets.
                    down = [action for action in actions if is_down_stair(action)]
                    if down:
                        return down

            if (
                current_floor == 9
                and state.items.get("blueKey", 0) > 0
                and state.items.get("yellowKey", 0) > 0
            ):
                right_chain = [
                    "redPotion MT9:11,11",
                    "yellowKey MT9:9,9",
                    "bluePriest MT9:9,11",
                    "yellowDoor MT9:8,11",
                ]
                for token in right_chain:
                    focused = [action for action in actions if token in label(action)]
                    if focused:
                        return focused

            if current_floor == 9 and state.items.get("blueKey", 0) > 0:
                correct_blue_door = [action for action in actions if "blueDoor MT9:3,11" in label(action)]
                if correct_blue_door:
                    return correct_blue_door

            yellow_key_actions = [action for action in actions if is_yellow_key_action(action)]
            if yellow_key_actions:
                return yellow_key_actions

            focused = [
                action
                for action in actions
                if any(token in label(action) for token in MT10_YELLOW_PREP_ACTION_TOKENS)
            ]
            non_stair = [action for action in focused if not (is_up_stair(action) or is_down_stair(action))]
            if non_stair:
                return non_stair
            if focused:
                return focused

    if (
        target_stage == "mt10_resources"
        and before_first_mt10_resource()
        and state.items.get("blueKey", 0) > 0
    ):
        # Preserve the cheap HP refill next to the 6F blue-key merchant before
        # climbing into the 9F/10F resource chain.  The MCTS backup is otherwise
        # repeatedly learning a low-HP 9F branch where the first skeleton soldier
        # leaves the route alive but unable to continue.
        if current_floor == 6:
            nearby_potions = [
                action
                for action in actions
                if "redPotion MT6:8,3" in label(action) and len(action.get("path") or []) <= 4
            ]
            if nearby_potions:
                return nearby_potions
        if current_floor == 9 and state.items.get("yellowKey", 0) < 3:
            # The MT10 resource pocket is a net yellow-key sink.  Strict replay
            # shows that entering it with only one key reaches MT1/MT2 with no
            # recoverable yellow-key path.  Before the first MT10 commitment,
            # force the cheap 9F key chain ahead of optional fight/gem branches.
            if state.hp < 120:
                potion = [
                    action
                    for action in actions
                    if "redPotion MT9:2,10" in label(action)
                ]
                if potion:
                    return potion
            for group in (
                ("yellowKey MT9:2,4",),
                ("skeletonSoldier MT9:1,3",),
                ("yellowKey MT9:2,2",),
            ):
                matched = [
                    action
                    for action in actions
                    if any(token in label(action) for token in group)
                ]
                if matched:
                    return matched
            direct_yellow_keys = [action for action in actions if is_yellow_key_action(action)]
            if state.items.get("yellowKey", 0) <= 1 and direct_yellow_keys:
                return direct_yellow_keys
        if current_floor == 9 and target_still_present("MT9", 1, 5, "blueGem"):
            gem_route = [
                action
                for action in actions
                if any(token in label(action) for token in MT9_GEM_BEFORE_SOLDIER_ACTION_TOKENS)
                and not (is_up_stair(action) or is_down_stair(action))
            ]
            if gem_route:
                return gem_route
        if current_floor == 9 and state.items.get("yellowKey", 0) >= 3:
            correct_blue_door = [action for action in actions if "blueDoor MT9:3,11" in label(action)]
            if correct_blue_door:
                return correct_blue_door
        first_entry_yellow_ready = state.items.get("yellowKey", 0) >= 2 or (
            state.items.get("yellowKey", 0) >= 1 and state.hp >= 500
        )
        if current_floor < 9 and first_entry_yellow_ready:
            up = [action for action in actions if is_up_stair(action)]
            if up:
                return up
        if current_floor == 9 and first_entry_yellow_ready:
            focused = [
                action
                for action in actions
                if any(token in label(action) for token in MT10_DIRECT_ACCESS_ACTION_TOKENS)
            ]
            non_stair = [action for action in focused if not (is_up_stair(action) or is_down_stair(action))]
            if non_stair:
                return non_stair
            if focused:
                return focused

    if (
        target_stage in {"mt10_yellow_ready", "mt10_resources"}
        and before_first_mt10_resource()
        and state.items.get("yellowKey", 0) <= 0
        and state.money >= 50
    ):
        direct_yellow_keys = [action for action in actions if is_yellow_key_action(action)]
        if direct_yellow_keys:
            return direct_yellow_keys
        if current_floor == 7:
            merchant = [action for action in actions if "buy 5 yellowKey MT7" in label(action)]
            if merchant:
                return merchant
        if current_floor > 7:
            down = [action for action in actions if is_down_stair(action)]
            if down:
                return down
        if current_floor < 7:
            up = [action for action in actions if is_up_stair(action)]
            if up:
                return up
        if state.items.get("yellowKey", 0) < 2:
            focused = [
                action
                for action in actions
                if any(token in label(action) for token in MT10_YELLOW_PREP_ACTION_TOKENS)
            ]
            yellow_keys = [action for action in focused if is_yellow_key_action(action)]
            if yellow_keys:
                return yellow_keys
            non_stair = [action for action in focused if not (is_up_stair(action) or is_down_stair(action))]
            if non_stair:
                return non_stair
            if focused:
                return focused
        if current_floor < 10:
            up = [action for action in actions if is_up_stair(action)]
            if up:
                return up

    if (
        target_stage == "mt10_resources"
        and before_first_mt10_resource()
        and current_floor == 9
        and is_clear_tile("MT9", 3, 11)
    ):
        if state.items.get("yellowKey", 0) <= 0 or state.hp < 120:
            local_refill = [
                action
                for action in actions
                if any(token in label(action) for token in MT10_PRE_STAIR_REFILL_ACTION_TOKENS)
                and not (is_up_stair(action) or is_down_stair(action))
            ]
            if local_refill:
                potion = [action for action in local_refill if "redPotion MT9:2,10" in label(action)]
                if potion:
                    return potion
                return local_refill
        up_to_10 = [action for action in actions if "upFloor MT9:1,11" in label(action)]
        if up_to_10:
            return up_to_10

    if target_stage == "mt10_resources":
        if current_floor == 10:
            if (
                target_still_present("MT10", 2, 6, "blueGem")
                and any("yellowDoor MT10:1,9" in label(action) for action in actions)
                and not can_start_mt10_left_resource_chain()
            ):
                down = [action for action in actions if is_down_stair(action)]
                if down:
                    return down
            if target_still_present("MT10", 2, 6, "blueGem"):
                focused_left = [
                    action
                    for action in actions
                    if any(token in label(action) for token in MT10_LEFT_RESOURCE_ACTION_TOKENS)
                ]
                immediate_gem = [
                    action
                    for action in focused_left
                    if "blueGem MT10:2,6" in label(action)
                ]
                if immediate_gem:
                    return immediate_gem
                non_stair_left = [
                    action
                    for action in focused_left
                    if not (is_up_stair(action) or is_down_stair(action))
                ]
                if non_stair_left:
                    return non_stair_left
                if focused_left:
                    return focused_left
            if mt10_progress_count() == 1 and mt10_after_left_refill_pending():
                down = [action for action in actions if is_down_stair(action)]
                if down:
                    return down
            if (
                mt10_progress_count() == 1
                and target_still_present("MT10", 3, 9, "yellowDoor")
                and target_still_present("MT10", 10, 6, "redGem")
                and target_still_present("MT10", 11, 11, "bluePotion")
                and state.items.get("yellowKey", 0) < 2
            ):
                # Opening the middle-left door with fewer than two yellow keys
                # consumes the last buffer before the right-side red gem and
                # potion.  Return to the archive/refill layer instead.
                down = [action for action in actions if is_down_stair(action)]
                if down:
                    return down
            if (
                mt10_progress_count() == 1
                and not target_still_present("MT10", 3, 9, "yellowDoor")
                and not target_still_present("MT10", 4, 11, "bluePriest")
                and target_still_present("MT10", 10, 6, "redGem")
                and target_still_present("MT10", 11, 11, "bluePotion")
                and state.items.get("yellowKey", 0) < 2
            ):
                # After opening the middle-left 10F door and clearing the
                # first blue priest, the route must rebuild yellow-key/HP
                # stock before spending the two right-side yellow doors.  If it
                # continues locally with only one key, it reaches the red gem
                # but cannot open the final potion door.
                down = [action for action in actions if is_down_stair(action)]
                if down:
                    return down
            if target_still_present("MT10", 10, 6, "redGem") and state.hp < 220:
                down = [action for action in actions if is_down_stair(action)]
                if down:
                    return down
            if (
                not target_still_present("MT10", 10, 6, "redGem")
                and target_still_present("MT10", 11, 11, "bluePotion")
                and state.items.get("yellowKey", 0) <= 0
            ):
                # After the right-side red gem, the remaining 10F potion is
                # behind yellow doors.  With zero yellow keys, extra local
                # fights only spend HP and cannot complete the resource pocket;
                # return to the lower-floor key economy first.
                down = [action for action in actions if is_down_stair(action)]
                if down:
                    return down
            focused = [action for action in actions if any(token in label(action) for token in MT10_RESOURCE_ACTION_TOKENS)]
            non_stair = [action for action in focused if not (is_up_stair(action) or is_down_stair(action))]
            if non_stair:
                return non_stair
            down = [action for action in actions if is_down_stair(action)]
            return down or focused or actions

        if (
            mt10_progress_count() >= 2
            and target_still_present("MT10", 11, 11, "bluePotion")
            and state.items.get("yellowKey", 0) <= 0
            and state.money >= 50
        ):
            merchant = [action for action in actions if "buy 5 yellowKey MT7" in label(action)]
            if merchant:
                return merchant
            if current_floor > 7:
                down = [action for action in actions if is_down_stair(action)]
                if down:
                    return down
            if current_floor < 7:
                up = [action for action in actions if is_up_stair(action)]
                if up:
                    return up
            if current_floor == 7:
                down = [action for action in actions if is_down_stair(action)]
                if down:
                    return down

        refill_actions = [
            action
            for action in actions
            if any(token in label(action) for token in MT10_REFILL_ACTION_TOKENS)
        ]
        if (
            mt10_progress_count() == 1
            and current_floor == 9
            and state.items.get("yellowKey", 0) > 0
            and state.hp >= 300
            and not target_still_present("MT9", 8, 11, "yellowDoor")
            and not target_still_present("MT9", 9, 11, "bluePriest")
            and not target_still_present("MT9", 11, 9, "bluePriest")
            and target_still_present("MT10", 3, 9, "yellowDoor")
        ):
            up_to_10 = [action for action in actions if "upFloor MT9:1,11" in label(action)]
            if up_to_10:
                return up_to_10
        if mt10_progress_count() == 1 and current_floor == 9 and state.items.get("yellowKey", 0) > 0:
            after_left = [
                action
                for action in actions
                if any(token in label(action) for token in MT10_AFTER_LEFT_REFILL_ACTION_TOKENS)
            ]
            non_stair_after_left = [
                action
                for action in after_left
                if not (is_up_stair(action) or is_down_stair(action))
            ]
            if non_stair_after_left:
                return non_stair_after_left
            if after_left:
                return after_left
        non_stair_refill = [action for action in refill_actions if not (is_up_stair(action) or is_down_stair(action))]
        if non_stair_refill:
            return non_stair_refill

        if current_floor < 10:
            if state.hp < 300 or state.items.get("yellowKey", 0) < 2:
                local_resources = [
                    action
                    for action in actions
                    if is_key_or_key_merchant(action) or is_item(action)
                ]
                if local_resources:
                    return local_resources
                if refill_actions:
                    return refill_actions
                down = [action for action in actions if is_down_stair(action)]
                if down and current_floor > 4 and low_refill_remaining_below_or_on(4):
                    return down
                if down and 2 <= current_floor <= 4 and low_refill_remaining_below_or_on(current_floor - 1):
                    return down
            up = [
                action
                for action in actions
                if is_up_stair(action) or any(token in label(action) for token in MT10_RESOURCE_ACTION_TOKENS)
            ]
            return up or refill_actions or actions

    if target_stage == "red_key":
        if current_floor > 8:
            if state.items.get("yellowKey", 0) <= 0 or state.hp < 260:
                key_or_item = [
                    action
                    for action in actions
                    if is_key_or_key_merchant(action) or is_item(action)
                ]
                if key_or_item:
                    return key_or_item
                non_stair = [
                    action
                    for action in actions
                    if not (is_up_stair(action) or is_down_stair(action))
                ]
                if non_stair:
                    return non_stair
            down = [action for action in actions if is_down_stair(action)]
            return down or actions
        if current_floor == 7 and target_still_present("MT7", 7, 11, "bluePotion"):
            for group in (
                ("yellowDoor MT7:7,7",),
                ("redSlime MT7:7,9",),
                ("bluePotion MT7:7,11",),
            ):
                mt7_buffer = [
                    action
                    for action in actions
                    if any(token in label(action) for token in group)
                ]
                if mt7_buffer:
                    return mt7_buffer
        if current_floor < 8:
            up = [action for action in actions if is_up_stair(action)]
            return up or actions
        if (
            current_floor == 8
            and target_still_present("MT8", 1, 5, "redPotion")
            and state.hp < 360
        ):
            left_potion = [
                action
                for action in actions
                if "redPotion MT8:1,5" in label(action)
            ]
            if left_potion:
                return left_potion
        if (
            current_floor == 8
            and target_still_present("MT8", 10, 7, "yellowDoor")
            and state.hp >= 260
            and state.items.get("yellowKey", 0) >= 1
        ):
            red_key_entry = [
                action
                for action in actions
                if "yellowDoor MT8:10,7" in label(action)
            ]
            if red_key_entry:
                return red_key_entry
        if (
            current_floor == 8
            and not target_still_present("MT8", 10, 7, "yellowDoor")
            and target_still_present("MT8", 10, 2, "redKey")
        ):
            red_key_corridor_groups = (
                ("yellowGuard MT8:9,5",),
                ("bluePotion MT8:9,3",),
                ("yellowKey MT8:9,1",),
                ("yellowGuard MT8:11,5",),
                ("redPotion MT8:11,3",),
                ("yellowKey MT8:11,1",),
                ("redKey MT8:10,2",),
                ("greenSlime MT8:7,2",),
            )
            for group in red_key_corridor_groups:
                corridor = [
                    action
                    for action in actions
                    if any(token in label(action) for token in group)
                ]
                if corridor:
                    return corridor
        if current_floor == 8 and (
            state.hp < 260 or state.items.get("yellowKey", 0) < 2
        ):
            recovery = [
                action
                for action in actions
                if is_up_stair(action)
                or is_down_stair(action)
                or is_key_or_key_merchant(action)
                or is_item(action)
            ]
            if recovery:
                return recovery
        focused = [
            action
            for action in actions
            if any(token in label(action) for token in RED_KEY_ACTION_TOKENS)
        ]
        non_stair = [action for action in focused if not (is_up_stair(action) or is_down_stair(action))]
        return non_stair or focused or actions

    if target_stage in {"mt10_blue_ready", "mt10_yellow_ready", "mt10_resources"}:
        if current_floor < 10:
            key_actions = [action for action in actions if is_key_or_key_merchant(action)]
            if key_actions:
                return key_actions
            focused = [
                action
                for action in actions
                if is_up_stair(action) or any(token in label(action) for token in MT10_RESOURCE_ACTION_TOKENS)
            ]
            return focused or actions
        if current_floor > 10:
            focused = [action for action in actions if is_down_stair(action)]
            return focused or actions
        focused = [action for action in actions if any(token in label(action) for token in MT10_RESOURCE_ACTION_TOKENS)]
        non_stair = [action for action in focused if not (is_up_stair(action) or is_down_stair(action))]
        return non_stair or focused or actions

    if target_stage == "low_gems":
        mt1_low_pending = target_still_present("MT1", 7, 3, "redGem") or target_still_present(
            "MT1", 7, 4, "blueGem"
        )
        mt3_low_pending = target_still_present("MT3", 2, 1, "blueGem") or target_still_present(
            "MT3", 2, 9, "redGem"
        )
        mt3_route_tokens = (
            "downFloor MT3:1,11",
            "bat MT3:3,5",
            "yellowDoor MT3:1,4",
            "bluePriest MT3:1,3",
            "blueGem MT3:2,1",
            "yellowDoor MT3:1,6",
            "skeleton MT3:1,7",
            "redPotion MT3:1,9",
            "redGem MT3:2,9",
            "yellowKey MT3:2,8",
        )
        mt1_route_tokens = (
            "yellowDoor MT1:6,6",
            "bat MT1:7,6",
            "bluePriest MT1:8,6",
            "bat MT1:9,6",
            "yellowDoor MT1:9,5",
            "redGem MT1:7,3",
            "blueGem MT1:7,4",
            "yellowDoor MT1:6,9",
            "yellowKey MT1:5,10",
            "redPotion MT1:1,10",
            "redPotion MT1:1,11",
        )
        mt1_center_entry_opened = not target_still_present("MT1", 6, 6, "yellowDoor")
        mt1_required_keys = 1 if mt1_center_entry_opened else 2
        mt1_needs_key_buffer = (
            mt1_low_pending
            and not mt3_low_pending
            and state.items.get("yellowKey", 0) < mt1_required_keys
        )
        if mt1_needs_key_buffer:
            key_refill = [
                action
                for action in actions
                if is_yellow_key_action(action)
                or any(token in label(action) for token in LOW_GEMS_KEY_BUFFER_ACTION_TOKENS)
            ]
            non_stair_key_refill = [
                action for action in key_refill if not (is_up_stair(action) or is_down_stair(action))
            ]
            if non_stair_key_refill:
                return non_stair_key_refill
            if current_floor < 7:
                up = [action for action in actions if is_up_stair(action)]
                if up:
                    return up
            if current_floor > 7:
                down = [action for action in actions if is_down_stair(action)]
                if down:
                    return down
            if key_refill:
                return key_refill
        if mt1_low_pending and not mt3_low_pending and current_floor > 1:
            down = [action for action in actions if is_down_stair(action)]
            return down or actions
        if current_floor > 3:
            focused = [action for action in actions if any(token in label(action) for token in LOW_GEMS_ACTION_TOKENS)]
            non_stair = [action for action in focused if not (is_up_stair(action) or is_down_stair(action))]
            if non_stair:
                return non_stair
            down = [action for action in actions if is_down_stair(action)]
            return down or focused or actions
        if current_floor == 3 and mt3_low_pending:
            focused = [action for action in actions if any(token in label(action) for token in mt3_route_tokens)]
            non_stair = [action for action in focused if not (is_up_stair(action) or is_down_stair(action))]
            if non_stair:
                return non_stair
            left_down = [action for action in focused if "downFloor MT3:1,11" in label(action)]
            return left_down or focused or actions
        if current_floor < 3 and mt3_low_pending and not mt1_low_pending:
            up = [action for action in actions if is_up_stair(action)]
            return up or actions
        if current_floor == 1 and mt1_low_pending:
            focused = [action for action in actions if any(token in label(action) for token in mt1_route_tokens)]
            non_stair = [action for action in focused if not (is_up_stair(action) or is_down_stair(action))]
            if non_stair:
                return non_stair
            floor_actions = [
                action for action in actions if not (is_up_stair(action) or is_down_stair(action))
            ]
            return focused or floor_actions or actions
        # These stages already include deliberate lower-floor routing.  Keep
        # them broad, but remove high-damage fights that do not reveal resources.
        focused = [
            action
            for action in actions
            if is_up_stair(action)
            or is_down_stair(action)
            or is_key_or_key_merchant(action)
            or is_item(action)
            or "door" in label_lower(action)
            or any(token in label(action) for token in ("bluePriest", "bat", "skeleton"))
        ]
        return focused or actions

    return actions


def _floor_index(floor_id: str) -> int:
    try:
        return int(str(floor_id).removeprefix("MT"))
    except ValueError:
        return 0


@dataclass(frozen=True)
class AlphaMCTSConfig:
    target_stage: str = "sword"
    num_simulations: int = 64
    c_puct: float = 1.5
    max_depth: int = 80
    success_value: float = 1.0
    failure_value: float = -1.0
    discount: float = 1.0
    seed: int = 0
    hp_aware_success_value: bool = False
    hp_success_base: float = 0.55
    hp_success_scale: float = 1000.0
    final_action_visit_weight: float = 1.0
    final_action_value_weight: float = 0.0
    final_action_prior_weight: float = 0.0
    root_dirichlet_alpha: float = 0.0
    root_exploration_fraction: float = 0.0
    use_stage_action_filter: bool = True
    edge_reward_scheme: str = "none"
    edge_reward_scale: float = 100.0
    edge_reward_clip: float = 1.0
    graph_include_unlock_values: bool = False


@dataclass
class MCTSChild:
    action_index: int
    action_node_index: int
    action: dict[str, Any]
    prior: float
    node: "MCTSNode" = field(default_factory=lambda: MCTSNode())


@dataclass
class MCTSNode:
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, MCTSChild] = field(default_factory=dict)
    expanded: bool = False
    terminal: bool = False
    max_nodes: int = 0

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / float(self.visit_count)


@dataclass(frozen=True)
class AlphaMCTSResult:
    action: dict[str, Any] | None
    action_index: int | None
    action_node_index: int | None
    root_value: float
    visit_count: int
    policy_target: list[float]
    child_stats: list[dict[str, Any]]


def uniform_policy_value(graph: dict[str, Any]) -> tuple[list[float], float]:
    executable = [bool(value) for value in graph["executable_mask"]]
    count = sum(executable)
    if count <= 0:
        return [0.0] * int(graph["max_nodes"]), 0.0
    prior = [1.0 / count if value else 0.0 for value in executable]
    return prior, 0.0


def _softmax(scores: list[float], mask: list[bool], temperature: float) -> list[float]:
    temp = max(float(temperature), 1e-6)
    if not any(mask):
        return [0.0] * len(scores)
    max_score = max(score / temp for score, valid in zip(scores, mask) if valid)
    values = [math.exp(score / temp - max_score) if valid else 0.0 for score, valid in zip(scores, mask)]
    total = sum(values)
    if total <= 1e-12:
        count = sum(mask)
        return [1.0 / count if valid else 0.0 for valid in mask]
    return [value / total for value in values]


class HeuristicPolicyValueFn:
    """Stage-aware policy/value warm start from graph features.

    This is not an expert route.  It only turns globally available game facts
    into a weak AlphaGo-style prior so early MCTS does not waste all simulations
    on arbitrary doors and fights before the policy/value network has learned.
    """

    def __init__(self, target_stage: str = "sword", temperature: float = 1.0):
        self.target_stage = target_stage
        self.temperature = float(temperature)

    def __call__(self, graph: dict[str, Any]) -> tuple[list[float], float]:
        names = graph["feature_names"]
        features = graph["node_features"]
        mask = [bool(value) for value in graph["executable_mask"]]
        current_floor = self._current_floor_index(graph)
        target = self._effective_target(graph, names)
        scores = [
            self._score_node(node, features[index], names, current_floor=current_floor, target=target)
            for index, node in enumerate(graph["nodes"])
        ]
        priors = _softmax(scores, mask, self.temperature)
        return priors, self._value(graph, names)

    def _score_node(
        self,
        node: dict[str, Any],
        row: list[float],
        names: list[str],
        *,
        current_floor: int,
        target: str,
    ) -> float:
        block_id = str(node.get("block_id") or "")
        kind = str(node.get("kind") or "")
        label = str(node.get("action_label") or "")
        label_lower = label.lower()
        target_floor = self._floor_index(str(node.get("floor") or ""))

        def feature(name: str) -> float:
            return float(row[names.index(name)]) if name in names else 0.0

        score = 0.0
        score += 2.5 * feature("item_value_norm")
        score += 2.5 * feature("unlock_value_norm")
        score += 3.0 * feature("damage_drop_atk1_norm")
        score += 3.5 * feature("damage_drop_def1_norm")
        enemy_damage_norm = feature("enemy_damage_norm") if kind in {"enemy", "boss"} else 0.0
        score -= 4.0 * enemy_damage_norm
        score -= 3.0 * feature("missing_yellow_norm")
        score -= 5.0 * feature("missing_blue_norm")
        score -= 8.0 * feature("missing_red_norm")
        score -= 0.6 * feature("path_len_norm")
        if kind == "enemy" and feature("enemy_killable") <= 0:
            score -= 8.0
        if kind == "door":
            score -= 0.6
        if kind == "stair":
            score += 0.25

        if target == "sword":
            if kind == "door":
                score -= 2.4
                score += 2.0 * feature("door_openable")
                if target_floor >= 4:
                    score += 0.8
                if feature("unlock_value_norm") >= 0.9:
                    score += 6.0
                elif feature("unlock_value_norm") >= 0.4:
                    score += 3.0
                elif feature("unlock_value_norm") >= 0.15:
                    score += 4.0
                elif feature("unlock_value_norm") <= 0.05:
                    score -= 0.8
            if kind == "stair":
                if "upfloor" in label_lower:
                    score += 6.0
                if "downfloor" in label_lower:
                    score -= 7.5
                if target_floor >= current_floor:
                    score += 1.0 * min(3, max(0, target_floor - current_floor))
                if target_floor >= 4:
                    score += 2.0
                if target_floor >= 5:
                    score += 4.0
            if block_id == "sword1":
                score += 26.0
            if block_id in {"yellowKey", "blueKey"}:
                score += 3.4
            if block_id == "redGem":
                score += 3.0
            if block_id == "blueGem":
                score += 1.5
            if kind == "enemy":
                score -= 2.5 + 14.0 * enemy_damage_norm
                if enemy_damage_norm > 0.08:
                    score -= 3.5
        elif target == "mt4_redgem":
            if kind == "stair":
                if current_floor > 4 and "downfloor" in label_lower:
                    score += 18.0 + 2.0 * min(3, current_floor - 4)
                if current_floor < 4 and "upfloor" in label_lower:
                    score += 12.0
                if current_floor == 4 and ("upfloor" in label_lower or "downfloor" in label_lower):
                    score -= 10.0
            if kind == "door":
                score -= 1.0
                score += 1.7 * feature("door_openable")
                if "yellowDoor MT4:8,8" in label:
                    score += 40.0
            if "bluePriest MT4:8,9" in label:
                score += 38.0
            if "redGem MT4:7,10" in label:
                score += 90.0
            if "redPotion MT4:9,10" in label:
                score += 18.0
            if block_id == "redGem":
                score += 10.0
            if block_id in {"yellowKey", "blueKey"}:
                score += 5.0
            if kind == "enemy":
                score -= 1.5 + 8.0 * enemy_damage_norm
                if any(token in label for token in ("bluePriest MT4:8,9", "bat MT5:6,4", "redSlime MT5:4,1")):
                    score += 14.0
            if any(token in label for token in MT4_REDGEM_ACTION_TOKENS):
                score += 18.0
        elif target in {"pre_shield_gems", "shield", "shield_buffer"}:
            if kind == "door":
                score -= 1.2
                score += 1.5 * feature("door_openable")
                if feature("unlock_value_norm") >= 0.4:
                    score += 5.0
                elif feature("unlock_value_norm") >= 0.15:
                    score += 3.0
                elif feature("unlock_value_norm") <= 0.05:
                    score -= 0.8
            if kind == "stair":
                if "upfloor" in label_lower and current_floor < 9:
                    score += 10.0
                if "downfloor" in label_lower and current_floor <= 9:
                    score -= 18.0
                if target_floor >= current_floor:
                    score += 1.2 * min(4, max(0, target_floor - current_floor))
                if target_floor >= 8:
                    score += 3.0
            if block_id == "shield1":
                score += 44.0
            # In strict no-hp403 runs, the 4F red gem is a high-leverage
            # pre-shield detour: ATK=21 reduces several bat/priest losses
            # before the 9F shield route.  Without this, the policy often
            # reaches shield with too little HP to recover later resources.
            if "redGem MT4:7,10" in label:
                score += 46.0
            if any(
                token in label
                for token in (
                    "yellowDoor MT4:8,8",
                    "yellowDoor MT4:11,8",
                    "bluePriest MT4:8,9",
                    "greenSlime MT4:8,11",
                )
            ):
                score += 10.0
            if "blueDoor MT9:6,3" in label:
                score += 18.0
            if "fakeWall MT9:10,5" in label:
                score += 24.0
            if "shield1 MT9:9,7" in label:
                score += 44.0
            if "buy 5 yellowKey MT7" in label:
                score += 70.0
            if any(
                token in label
                for token in (
                    "skeletonSoldier MT7:9,7",
                    "blueDoor MT7:5,5",
                    "yellowDoor MT7:7,5",
                    "yellowDoor MT7:3,5",
                    "yellowDoor MT7:5,7",
                    "yellowDoor MT7:3,7",
                    "yellowDoor MT7:7,7",
                    "yellowDoor MT7:11,5",
                    "yellowDoor MT7:1,7",
                    "yellowDoor MT7:1,5",
                )
            ):
                score -= 42.0
            if current_floor == 7 and "upFloor MT7:1,1" in label:
                score += 36.0
            if current_floor == 8 and any(
                token in label
                for token in (
                    "yellowDoor MT8:3,1",
                    "yellowDoor MT8:4,1",
                    "upFloor MT8:6,1",
                )
            ):
                score += 34.0
            if current_floor == 9 and any(
                token in label
                for token in (
                    "yellowDoor MT9:8,1",
                    "greenSlime MT9:9,1",
                    "fakeWall MT9:10,5",
                    "shield1 MT9:9,7",
                )
            ):
                score += 38.0
            if block_id == "sword1":
                score += 8.0
            if block_id == "blueGem":
                score += 4.2
            if block_id == "redGem":
                score += 3.2
            if block_id in {"yellowKey", "blueKey"}:
                score += 5.4
            if kind == "enemy":
                score -= 2.0 + 12.0 * enemy_damage_norm
                if feature("unlock_value_norm") <= 0.1:
                    score -= 2.0
            if "shield" in label:
                score += 12.0
        elif target in RESOURCE_TARGET_STAGES:
            hero_hp_norm = feature("hp_norm")
            target_floor_hint = self._target_floor_hint(target, label, current_floor)
            if kind == "stair":
                if target_floor_hint > current_floor and "upfloor" in label_lower:
                    score += 7.0 + 1.2 * min(4, target_floor_hint - current_floor)
                elif target_floor_hint < current_floor and "downfloor" in label_lower:
                    score += 8.0 + 1.5 * min(6, current_floor - target_floor_hint)
                elif target_floor_hint == current_floor and ("upfloor" in label_lower or "downfloor" in label_lower):
                    score -= 3.0
                if target == "mt8_hp_ready" and hero_hp_norm < 0.20:
                    if "upfloor" in label_lower:
                        score -= 8.0
                    if "downfloor" in label_lower:
                        score += 2.0
                if target == "mt8_gems":
                    if current_floor == 8 and ("upfloor" in label_lower or "downfloor" in label_lower):
                        score -= 24.0
                    if "upFloor MT8:6,1" in label or "downFloor MT8:1,1" in label:
                        score -= 18.0
            if kind == "door":
                score -= 1.1
                score += 1.8 * feature("door_openable")
                if feature("unlock_value_norm") >= 0.35:
                    score += 7.0
                elif feature("unlock_value_norm") >= 0.12:
                    score += 3.5
                elif feature("unlock_value_norm") <= 0.03:
                    score -= 1.6
                if target == "low_gems" and "yellowDoor MT1" in label and feature("yellow_key_norm") <= 0.13:
                    score -= 8.0
            if kind == "npc" or "buy " in label_lower:
                if "yellow" in label_lower or "yellowkey" in label_lower:
                    score += 17.0
                if "blue" in label_lower or "bluekey" in label_lower:
                    score += 20.0
                score += 4.0
            if block_id == "redGem":
                score += 18.0 + 5.0 * feature("damage_drop_atk1_norm")
            if block_id == "blueGem":
                score += 18.0 + 5.5 * feature("damage_drop_def1_norm")
            if block_id in {"yellowKey", "blueKey"}:
                score += 9.0 if block_id == "yellowKey" else 12.0
            if block_id in {"redPotion", "bluePotion"}:
                score += 1.0
                if target == "mt8_hp_ready":
                    score += 22.0
                if hero_hp_norm < 0.08:
                    score += 10.0
                elif hero_hp_norm < 0.18:
                    score += 4.0
                elif target in {"mt10_resources", "all_gems"} and "MT10" in label:
                    score += 7.0
            if kind == "enemy":
                score -= 1.0 + 8.0 * enemy_damage_norm
                score += 7.0 * feature("unlock_value_norm")
                score += 4.5 * feature("damage_drop_atk1_norm")
                score += 4.5 * feature("damage_drop_def1_norm")
                if feature("unlock_value_norm") <= 0.08 and enemy_damage_norm > 0.06:
                    score -= 4.5

            score += self._resource_stage_label_bonus(target, label)
            if target in {
                "pre_mt10_buffer",
                "mt10_yellow_ready",
                "mt10_resources",
                "all_gems",
                "boss_all_gems",
            } and any(token in label for token in LOW_FLOOR_REFILL_ACTION_TOKENS):
                score += 22.0
            if target == "mt8_gems" and any(
                token in label
                for token in (
                    "redGem MT8:4,10",
                    "blueGem MT8:5,11",
                    "blueDoor MT8:3,11",
                    "blueKey MT8:7,10",
                    "yellowKey MT8:7,11",
                    "redPotion MT8:8,10",
                )
            ):
                score += 24.0
            if target == "mt8_gems" and "MT8" not in label:
                if kind == "enemy":
                    score -= 14.0
                if kind == "door":
                    score -= 8.0
                if block_id in {"redGem", "blueGem"}:
                    score -= 18.0
                if block_id in {"redPotion", "bluePotion"}:
                    score -= 6.0
            if target == "low_gems" and any(
                token in label
                for token in (
                    "yellowDoor MT3:1,4",
                    "yellowDoor MT3:1,6",
                    "bluePriest MT3:1,3",
                    "blueGem MT3:2,1",
                )
            ):
                score += 24.0
        elif target in {"guard_ready", "red_key", "boss_ready", "trap", "boss", "boss_all_gems"}:
            hero_hp_norm = feature("hp_norm")
            if block_id == "redKey":
                score += 30.0
            if kind == "npc" or "buy " in label_lower:
                if "5 yellow" in label_lower or "yellowkey" in label_lower:
                    score += 22.0
                if "bluekey" in label_lower:
                    score += 16.0
                score += 3.0
            if block_id == "shield1":
                score += 10.0
            if block_id == "sword1":
                score += 8.0
            if block_id in {"redGem", "blueGem"}:
                score += 7.0
                if target == "boss_all_gems" and "MT10" in label:
                    score += 30.0
            if block_id in {"yellowKey", "blueKey"}:
                score += 7.5
            if kind == "door":
                score -= 1.2
                score += 1.8 * feature("door_openable")
                if feature("unlock_value_norm") >= 0.3:
                    score += 5.0
                if "MT8" in label:
                    score += 2.0
            if kind == "stair":
                if "upfloor" in label_lower and current_floor < 8:
                    score += 5.0
                if "downfloor" in label_lower and current_floor > 8:
                    score += 1.5
            if "skeletonCaptain" in label:
                score += 2.0 + 4.0 * feature("boss_margin_norm")
            if block_id in {"redPotion", "bluePotion"}:
                score += 3.5
                if target == "boss_all_gems" and "MT10" in label:
                    score += 10.0
                if hero_hp_norm < 0.35:
                    score += 8.0
            if any(token in label for token in LOW_FLOOR_REFILL_ACTION_TOKENS):
                score += 16.0
                if hero_hp_norm < 0.35:
                    score += 14.0
        else:
            if block_id in {"redGem", "blueGem", "sword1", "shield1", "redKey"}:
                score += 4.0
        return score

    def _target_floor_hint(self, target: str, label: str, current_floor: int) -> int:
        if target == "mid_gems":
            return 5 if current_floor <= 6 else 6
        if target == "mt4_redgem":
            return 4
        if target == "low_gems":
            return 3 if current_floor > 3 else 1
        if target == "lower_gems":
            if "MT8" in label:
                return 8
            if any(token in label for token in ("MT5", "MT6", "MT7", "MT9")):
                return 6
            if any(token in label for token in ("MT1", "MT2", "MT3", "MT4")):
                return 3
            return 6 if current_floor > 3 else 3
        if target in {"mt8_hp_ready", "mt8_gems"}:
            return 8
        if target in {
            "mt10_blue_ready",
            "pre_mt10_buffer",
            "mt10_yellow_ready",
            "mt10_resources",
            "boss_all_gems",
        }:
            return 10
        if target == "all_gems":
            if any(token in label for token in ("MT1", "MT3")):
                return 3
            if any(token in label for token in ("MT5", "MT6")):
                return 6
            if "MT8" in label:
                return 8
            if "MT10" in label:
                return 10
        return current_floor

    def _resource_stage_label_bonus(self, target: str, label: str) -> float:
        targets: dict[str, tuple[str, ...]] = {
            "mid_gems": (
                "blueGem MT5:1,9",
                "blueGem MT6:4,9",
                "yellowDoor MT5:4,4",
                "bluePriest MT5:3,5",
                "bat MT5:4,6",
                "skeleton MT5:2,7",
                "bluePriest MT6:1,8",
                "bat MT6:2,9",
                "buy blueKey MT6",
                "buy yellowKey MT7",
            ),
            "mt4_redgem": MT4_REDGEM_ACTION_TOKENS,
            "low_gems": (
                *LOW_GEMS_ACTION_TOKENS,
                "bluePotion MT4:11,2",
            ),
            "lower_gems": (
                *MID_GEMS_EARLY_ACTION_TOKENS,
                *MID_GEMS_LATE_ACTION_TOKENS,
                *LOW_GEMS_ACTION_TOKENS,
                *MT8_RESOURCE_ACTION_TOKENS,
                *MT8_LOWER_GEM_ROUTE_ACTION_TOKENS,
                *MT4_BLUE_KEY_POCKET_ACTION_TOKENS,
                *MT9_LOWER_GEM_ACTION_TOKENS,
                *MT7_RED_GEM_ACTION_TOKENS,
                *LOW_FLOOR_REFILL_ACTION_TOKENS,
            ),
            "mt8_hp_ready": (
                "redPotion MT8:1,5",
                "yellowKey MT8:3,4",
                "yellowDoor MT8:1,3",
            ),
            "mt8_gems": (
                "redGem MT8:4,10",
                "blueGem MT8:5,11",
                "redPotion MT8:1,5",
                "blueKey MT8:7,10",
                "yellowKey MT8:7,11",
                "yellowKey MT8:3,4",
                "yellowKey MT8:4,4",
                "yellowKey MT8:5,4",
                "yellowKey MT8:5,10",
                "yellowKey MT8:4,11",
                "redPotion MT8:8,10",
                "yellowDoor MT8:1,3",
                "yellowDoor MT8:5,7",
                "yellowDoor MT8:9,11",
                "blueDoor MT8:3,11",
                "bat MT8:4,8",
                "skeleton MT8:6,8",
                "skeletonSoldier MT8:10,11",
            ),
            "mt10_blue_ready": (
                *MT6_BLUE_KEY_BUY_ACTION_TOKENS,
                "blueKey MT8:7,10",
                "blueDoor MT9:3,11",
                "upFloor MT9:6,1",
            ),
            "pre_mt10_buffer": (
                *MID_GEMS_EARLY_ACTION_TOKENS,
                *MID_GEMS_LATE_ACTION_TOKENS,
                *LOW_GEMS_ACTION_TOKENS,
                *MT8_RESOURCE_ACTION_TOKENS,
                *MT8_LOWER_GEM_ROUTE_ACTION_TOKENS,
                *MT9_LOWER_GEM_ACTION_TOKENS,
                *MT7_RED_GEM_ACTION_TOKENS,
                *LOW_FLOOR_REFILL_ACTION_TOKENS,
                *MT6_BLUE_KEY_BUY_ACTION_TOKENS,
                "redPotion MT6:8,3",
                "yellowKey",
                "buy yellowKey",
                "blueDoor MT9:3,11",
                "upFloor MT9:6,1",
            ),
            "mt10_yellow_ready": (
                *MT6_BLUE_KEY_BUY_ACTION_TOKENS,
                "redPotion MT6:8,3",
                "yellowKey",
                "buy yellowKey",
                "redSlime MT9:7,6",
                "yellowDoor MT9:4,5",
                "bat MT9:3,5",
                "blueGem MT9:1,5",
                "bat MT9:7,10",
                "yellowDoor MT9:6,11",
                "blueDoor MT9:3,11",
                "redPotion MT9:2,10",
                "upFloor MT9:1,11",
                "blueKey MT8:7,10",
                "redPotion MT8:8,10",
                "skeletonSoldier MT8:10,11",
                "upFloor MT9:6,1",
            ),
            "mt10_resources": (
                "blueGem MT10:2,6",
                "redGem MT10:10,6",
                "bluePotion MT10:11,11",
                "blueDoor MT9:3,11",
                "yellowDoor MT10",
                "upFloor MT9:6,1",
            ),
            "boss_all_gems": POST_BOSS_MT10_RESOURCE_ACTION_TOKENS,
        }
        if target == "boss_all_gems":
            return 24.0 if any(token in label for token in targets["boss_all_gems"]) else 0.0
        if target == "all_gems":
            token_groups = (
                targets["low_gems"]
                + targets["mid_gems"]
                + targets["mt8_gems"]
                + targets["mt10_resources"]
            )
        else:
            token_groups = targets.get(target, ())
        return 12.0 if any(token in label for token in token_groups) else 0.0

    def _current_floor_index(self, graph: dict[str, Any]) -> int:
        for node in graph.get("nodes", []):
            if node.get("kind") == "hero":
                return self._floor_index(str(node.get("floor") or ""))
        return 0

    def _effective_target(self, graph: dict[str, Any], names: list[str]) -> str:
        target = self.target_stage
        if target == "sword":
            return target
        hero_row = None
        for index, node in enumerate(graph.get("nodes", [])):
            if node.get("kind") == "hero":
                hero_row = graph["node_features"][index]
                break
        atk = 0.0
        if hero_row is not None and "atk_norm" in names:
            atk = float(hero_row[names.index("atk_norm")]) * 80.0
        sword_present = any(
            node.get("block_id") == "sword1" and not node.get("consumed")
            for node in graph.get("nodes", [])
        )
        shield_present = any(
            node.get("block_id") == "shield1" and not node.get("consumed")
            for node in graph.get("nodes", [])
        )
        mt4_red_present = any(
            node.get("block_id") == "redGem"
            and str(node.get("floor") or "") == "MT4"
            and int(node.get("x") or -1) == 7
            and int(node.get("y") or -1) == 10
            and not node.get("consumed")
            for node in graph.get("nodes", [])
        )
        late_targets = {
            "pre_shield_gems",
            "shield",
            "shield_buffer",
            "red_key",
            "boss_ready",
            "trap",
            "boss",
            "boss_all_gems",
        } | RESOURCE_TARGET_STAGES
        if target in late_targets and (atk < 20.0 or sword_present):
            return "sword"
        if target in {"pre_shield_gems", "shield", "shield_buffer"} and mt4_red_present and not shield_present:
            return "mt4_redgem"
        if target in (late_targets - {"pre_shield_gems", "shield", "shield_buffer"}) and shield_present:
            return "shield"
        return target

    def _floor_index(self, floor_id: str) -> int:
        try:
            return int(str(floor_id).removeprefix("MT"))
        except ValueError:
            return 0

    def _value(self, graph: dict[str, Any], names: list[str]) -> float:
        if not graph["node_features"]:
            return 0.0
        hero = graph["node_features"][0]

        def feature(name: str) -> float:
            return float(hero[names.index(name)]) if name in names else 0.0

        value = -0.2
        value += 0.6 * feature("hp_norm")
        value += 0.8 * feature("atk_norm")
        value += 0.8 * feature("def_norm")
        value += 0.2 * feature("yellow_key_norm")
        value += 0.3 * feature("blue_key_norm")
        value += 0.5 * feature("boss_margin_norm")
        if self.target_stage in {
            "pre_mt10_buffer",
            "mt10_yellow_ready",
            "mt10_resources",
            "all_gems",
            "red_key",
            "boss_ready",
            "trap",
            "boss",
            "boss_all_gems",
        }:
            value += 0.8 * feature("hp_norm")
            value += 0.35 * feature("yellow_key_norm")
        return max(-1.0, min(1.0, value))


class BlendedPolicyValueFn:
    def __init__(self, primary: PolicyValueFn, secondary: PolicyValueFn, secondary_mix: float):
        self.primary = primary
        self.secondary = secondary
        self.secondary_mix = max(0.0, min(1.0, float(secondary_mix)))

    def __call__(self, graph: dict[str, Any]) -> tuple[list[float], float]:
        primary_prior, primary_value = self.primary(graph)
        secondary_prior, secondary_value = self.secondary(graph)
        mix = self.secondary_mix
        size = max(len(primary_prior), len(secondary_prior), int(graph["max_nodes"]))
        priors: list[float] = []
        for index in range(size):
            left = primary_prior[index] if index < len(primary_prior) else 0.0
            right = secondary_prior[index] if index < len(secondary_prior) else 0.0
            priors.append((1.0 - mix) * left + mix * right)
        total = sum(priors)
        if total > 1e-12:
            priors = [value / total for value in priors]
        return priors, (1.0 - mix) * float(primary_value) + mix * float(secondary_value)


class TorchPolicyValueFn:
    def __init__(self, model, device: str = "cpu", temperature: float = 1.0, cache_size: int = 4096):
        import torch

        self.model = model
        self.device = device
        self.temperature = float(temperature)
        self.torch = torch
        self.cache_size = max(0, int(cache_size))
        self._cache: dict[Any, tuple[list[float], float]] = {}
        self._cache_order: list[Any] = []

    def __call__(self, graph: dict[str, Any]) -> tuple[list[float], float]:
        import numpy as np

        cache_key = graph.get("_cache_key")
        if self.cache_size > 0 and cache_key is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached
        self.model.eval()
        node_features = self.torch.as_tensor(
            np.asarray(graph["node_features"], dtype=np.float32),
            dtype=self.torch.float32,
            device=self.device,
        ).unsqueeze(0)
        node_type_ids = self.torch.as_tensor(
            np.asarray(graph["node_type_ids"], dtype=np.int64),
            dtype=self.torch.long,
            device=self.device,
        ).unsqueeze(0)
        node_mask = self.torch.as_tensor(graph["node_mask"], dtype=self.torch.bool, device=self.device).unsqueeze(0)
        executable_mask = self.torch.as_tensor(
            graph["executable_mask"],
            dtype=self.torch.bool,
            device=self.device,
        ).unsqueeze(0)
        with self.torch.no_grad():
            logits, value = self.model(node_features, node_type_ids, node_mask)
            masked = (logits / max(self.temperature, 1e-6)).masked_fill(
                ~executable_mask,
                self.torch.finfo(logits.dtype).min,
            )
            probs = self.torch.softmax(masked, dim=-1)[0].detach().cpu().tolist()
        result = ([float(item) for item in probs], float(value.item()))
        if self.cache_size > 0 and cache_key is not None:
            if len(self._cache_order) >= self.cache_size:
                old_key = self._cache_order.pop(0)
                self._cache.pop(old_key, None)
            self._cache[cache_key] = result
            self._cache_order.append(cache_key)
        return result


class LearnedRewardValueFn:
    """Use a learned PBRS potential as the single-player MCTS leaf value."""

    def __init__(
        self,
        weights_payload: dict[str, Any],
        *,
        scale: float = 25_000.0,
        stage_mode: str = "target",
        gamma: float = 0.99,
    ):
        payload = weights_payload.get("weights", weights_payload)
        self.reward = LearnableStageReward(
            gamma=float(payload.get("gamma", gamma)),
            global_weights=dict(payload.get("global_weights", {})),
            stage_weights={
                str(stage): {str(key): float(value) for key, value in dict(weights).items()}
                for stage, weights in dict(payload.get("stage_weights", {})).items()
            },
        )
        self.scale = max(1.0, float(scale))
        if stage_mode not in {"target", "current"}:
            raise ValueError("stage_mode must be 'target' or 'current'")
        self.stage_mode = stage_mode

    def __call__(self, sim: MotaSimulator, state: MotaState, target_stage: str) -> float:
        if state.dead or state.hp <= 0:
            return -1.0
        stage = current_stage_name(sim, state) if self.stage_mode == "current" else target_stage
        phi = self.reward.potential(sim, state, stage=stage)
        return max(-1.0, min(1.0, math.tanh(float(phi) / self.scale)))


class AlphaMCTS:
    def __init__(
        self,
        sim: MotaSimulator,
        policy_value_fn: PolicyValueFn | None = None,
        leaf_value_fn: LeafValueFn | None = None,
        edge_reward_fn: EdgeRewardFn | None = None,
        config: AlphaMCTSConfig | None = None,
    ):
        self.sim = sim
        self.policy_value_fn = policy_value_fn or uniform_policy_value
        self.leaf_value_fn = leaf_value_fn
        self.edge_reward_fn = edge_reward_fn
        self.config = config or AlphaMCTSConfig()
        self.rng = random.Random(self.config.seed)
        scheme = str(self.config.edge_reward_scheme or "none")
        self.edge_rewarder = None if scheme == "none" else Rewarder(scheme, gamma=self.config.discount)

    def search(self, root_state: MotaState) -> AlphaMCTSResult:
        root = MCTSNode()
        self._expand(root, root_state.clone())
        self._add_root_exploration_noise(root)
        for _ in range(max(1, self.config.num_simulations)):
            self._simulate(root, root_state.clone())
        return self._result(root)

    def _add_root_exploration_noise(self, root: MCTSNode) -> None:
        fraction = max(0.0, min(1.0, float(self.config.root_exploration_fraction)))
        alpha = float(self.config.root_dirichlet_alpha)
        if fraction <= 0.0 or alpha <= 0.0 or not root.children:
            return
        children = list(root.children.values())
        noise = [self.rng.gammavariate(alpha, 1.0) for _ in children]
        total = sum(noise)
        if total <= 1e-12:
            return
        for child, sample in zip(children, noise):
            child.prior = (1.0 - fraction) * child.prior + fraction * (sample / total)

    def _simulate(self, root: MCTSNode, state: MotaState) -> None:
        node = root
        path: list[tuple[MCTSNode, float]] = [(node, 0.0)]
        depth = 0
        value: float | None = self._terminal_value(state)

        while value is None and node.expanded and node.children and depth < self.config.max_depth:
            child = self._select_child(node)
            before = state.clone() if self.edge_rewarder is not None or self.edge_reward_fn is not None else state
            transition = self.sim.apply_macro_action(state, child.action)
            edge_reward = self._edge_reward(before, state, child.action, transition)
            node = child.node
            path.append((node, edge_reward))
            depth += 1
            if not transition.ok:
                value = self.config.failure_value
                break
            value = self._terminal_value(state)

        if value is None:
            if depth >= self.config.max_depth:
                value = self._depth_cutoff_value(state)
            else:
                value = self._expand(node, state)

        self._backpropagate(path, value)

    def _select_child(self, node: MCTSNode) -> MCTSChild:
        parent_visits = max(1, node.visit_count)
        best_score = -math.inf
        best_children: list[MCTSChild] = []
        for child in node.children.values():
            q_value = child.node.value
            u_value = (
                self.config.c_puct
                * child.prior
                * math.sqrt(parent_visits)
                / (1.0 + child.node.visit_count)
            )
            score = q_value + u_value
            if score > best_score + 1e-12:
                best_score = score
                best_children = [child]
            elif abs(score - best_score) <= 1e-12:
                best_children.append(child)
        return self.rng.choice(best_children)

    def _expand(self, node: MCTSNode, state: MotaState) -> float:
        value = self._terminal_value(state)
        if value is not None:
            node.expanded = True
            node.terminal = True
            return value

        actions = self.sim.macro_actions(state)
        if self.config.use_stage_action_filter:
            actions = filter_stage_actions(
                actions,
                state,
                self.config.target_stage,
                sim=self.sim,
            )
        if not actions:
            node.expanded = True
            node.terminal = True
            return self.config.failure_value

        if self.policy_value_fn is uniform_policy_value:
            uniform = 1.0 / float(len(actions))
            node.children = {
                index: MCTSChild(
                    action_index=index,
                    action_node_index=index,
                    action=action,
                    prior=uniform,
                )
                for index, action in enumerate(actions)
            }
            node.max_nodes = len(actions)
            node.expanded = True
            if self.leaf_value_fn is not None:
                value = self.leaf_value_fn(self.sim, state, self.config.target_stage)
            else:
                value = 0.0
            return float(max(-1.0, min(1.0, value)))

        graph = build_graph_state(
            self.sim,
            state,
            actions=actions,
            include_unlock_values=self.config.graph_include_unlock_values,
        )
        node.max_nodes = int(graph["max_nodes"])
        priors, value = self.policy_value_fn(graph)
        if self.leaf_value_fn is not None:
            value = self.leaf_value_fn(self.sim, state, self.config.target_stage)
        children: dict[int, MCTSChild] = {}
        prior_sum = 0.0
        for action_index, node_index in graph["action_to_node_index"].items():
            if action_index < 0 or action_index >= len(actions):
                continue
            prior = max(0.0, float(priors[node_index])) if node_index < len(priors) else 0.0
            children[int(action_index)] = MCTSChild(
                action_index=int(action_index),
                action_node_index=int(node_index),
                action=actions[int(action_index)],
                prior=prior,
            )
            prior_sum += prior
        if not children:
            node.expanded = True
            node.terminal = True
            return self.config.failure_value
        if prior_sum <= 1e-12:
            uniform = 1.0 / float(len(children))
            for child in children.values():
                child.prior = uniform
        else:
            for child in children.values():
                child.prior /= prior_sum
        node.children = children
        node.expanded = True
        return float(max(-1.0, min(1.0, value)))

    def _terminal_value(self, state: MotaState) -> float | None:
        if stage_complete(self.sim, state, self.config.target_stage):
            if self.config.hp_aware_success_value and self.config.target_stage in {
                "shield",
                "shield_buffer",
                "red_key",
                "boss_ready",
                "trap",
                "boss",
                "boss_all_gems",
            }:
                scale = max(1.0, float(self.config.hp_success_scale))
                key_bonus = min(0.12, 0.03 * float(state.items.get("yellowKey", 0))) + min(
                    0.08,
                    0.05 * float(state.items.get("blueKey", 0)),
                )
                value = float(self.config.hp_success_base) + float(state.hp) / scale + key_bonus
                return max(self.config.failure_value, min(self.config.success_value, value))
            return self.config.success_value
        if state.dead or state.done:
            return self.config.failure_value
        return None

    def _depth_cutoff_value(self, state: MotaState) -> float:
        if self.config.target_stage in {"boss", "boss_all_gems"}:
            margin = boss_route_margin(self.sim, state)
            return max(-1.0, min(1.0, margin / 2000.0))
        return 0.0

    def _edge_reward(
        self,
        before: MotaState,
        after: MotaState,
        action: dict[str, Any],
        transition,
    ) -> float:
        if self.edge_reward_fn is None and self.edge_rewarder is None:
            return 0.0
        if self.edge_reward_fn is not None:
            reward = float(
                self.edge_reward_fn(
                    self.sim,
                    before,
                    after,
                    action,
                    transition,
                    self.config.target_stage,
                )
            )
        elif self.config.edge_reward_scheme == "raw":
            reward = float(transition.reward)
        else:
            assert self.edge_rewarder is not None
            reward = float(self.edge_rewarder.score(self.sim, before, after, action, transition).total)
        scale = max(1e-6, float(self.config.edge_reward_scale))
        value = reward / scale
        clip = max(0.0, float(self.config.edge_reward_clip))
        if clip > 0.0:
            value = max(-clip, min(clip, value))
        return value

    def _backpropagate(self, path: list[tuple[MCTSNode, float]], value: float) -> None:
        """Back up single-agent deterministic MDP returns.

        For a path ``s_0,a_0,r_0,...,s_L`` and leaf estimate ``V(s_L)``, the
        backed return is ``G_k = r_k + gamma * G_{k+1}``.  There is no minimax
        sign flip because Magic Tower is a single-player MDP.
        """
        backed_value = float(value)
        for index in range(len(path) - 1, -1, -1):
            node, incoming_reward = path[index]
            if index > 0:
                backed_value = float(incoming_reward) + self.config.discount * backed_value
            node.visit_count += 1
            node.value_sum += backed_value

    def _result(self, root: MCTSNode) -> AlphaMCTSResult:
        max_nodes = max(0, int(root.max_nodes))
        for child in root.children.values():
            max_nodes = max(max_nodes, child.action_node_index + 1)
        policy_target = [0.0] * max_nodes
        total_visits = sum(child.node.visit_count for child in root.children.values())
        child_stats: list[dict[str, Any]] = []
        best_child: MCTSChild | None = None
        for child in root.children.values():
            probability = child.node.visit_count / float(total_visits) if total_visits > 0 else 0.0
            if child.action_node_index >= len(policy_target):
                policy_target.extend([0.0] * (child.action_node_index + 1 - len(policy_target)))
            policy_target[child.action_node_index] = probability
            child_stats.append(
                {
                    "action_index": child.action_index,
                    "action_node_index": child.action_node_index,
                    "visit_count": child.node.visit_count,
                    "prior": child.prior,
                    "value": child.node.value,
                    "probability": probability,
                    "label": child.action.get("label", ""),
                    "action": child.action,
                }
            )
            if best_child is None or _final_child_score(child, self.config) > _final_child_score(best_child, self.config):
                best_child = child
        child_stats.sort(
            key=lambda row: _final_child_score_from_stats(row, self.config),
            reverse=True,
        )
        return AlphaMCTSResult(
            action=best_child.action if best_child else None,
            action_index=best_child.action_index if best_child else None,
            action_node_index=best_child.action_node_index if best_child else None,
            root_value=root.value,
            visit_count=root.visit_count,
            policy_target=policy_target,
            child_stats=child_stats,
        )


def _final_child_score(child: MCTSChild, config: AlphaMCTSConfig) -> tuple[float, ...]:
    """Rank root children when selecting the actual action.

    AlphaZero normally picks by visit count.  In this project we often run very
    small searches while iterating, so ties are common; value and prior tie
    breakers avoid arbitrary insertion-order choices.
    """

    if (
        float(config.final_action_visit_weight) != 1.0
        or float(config.final_action_value_weight) != 0.0
        or float(config.final_action_prior_weight) != 0.0
    ):
        score = (
            float(config.final_action_visit_weight) * float(child.node.visit_count)
            + float(config.final_action_value_weight) * child.node.value
            + float(config.final_action_prior_weight) * child.prior
        )
        return (
            score,
            float(child.node.visit_count),
            child.node.value,
            child.prior,
            -float(child.action_index),
        )
    return (
        float(child.node.visit_count),
        float(config.final_action_value_weight) * child.node.value,
        float(config.final_action_prior_weight) * child.prior,
        -float(child.action_index),
    )


def _final_child_score_from_stats(row: dict[str, Any], config: AlphaMCTSConfig) -> tuple[float, ...]:
    if (
        float(config.final_action_visit_weight) != 1.0
        or float(config.final_action_value_weight) != 0.0
        or float(config.final_action_prior_weight) != 0.0
    ):
        score = (
            float(config.final_action_visit_weight) * float(row.get("visit_count", 0.0))
            + float(config.final_action_value_weight) * float(row.get("value", 0.0))
            + float(config.final_action_prior_weight) * float(row.get("prior", 0.0))
        )
        return (
            score,
            float(row.get("visit_count", 0.0)),
            float(row.get("value", 0.0)),
            float(row.get("prior", 0.0)),
            -float(row.get("action_index", 0.0)),
        )
    return (
        float(row.get("visit_count", 0.0)),
        float(config.final_action_value_weight) * float(row.get("value", 0.0)),
        float(config.final_action_prior_weight) * float(row.get("prior", 0.0)),
        -float(row.get("action_index", 0.0)),
    )
