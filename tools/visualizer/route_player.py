# -*- coding: utf-8 -*-
"""Replay JSONL macro-action routes inside the Tkinter visualizer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RouteStep:
    index: int
    action: dict[str, Any]
    before: dict[str, Any] | None
    after: dict[str, Any] | None
    raw: dict[str, Any]

    @property
    def label(self) -> str:
        return str(self.action.get("label") or self.action.get("kind") or f"step {self.index}")

    @property
    def target_pos(self) -> tuple[int, int, int]:
        return target_to_visualizer_pos(self.action.get("target"))

    @property
    def before_pos(self) -> tuple[int, int, int] | None:
        if not self.before:
            return None
        return state_to_visualizer_pos(self.before)

    @property
    def after_pos(self) -> tuple[int, int, int] | None:
        if not self.after:
            return None
        return state_to_visualizer_pos(self.after)

    def path_positions(self) -> list[tuple[int, int, int]]:
        start = self.before_pos
        path = self.action.get("path")
        if start is None or not isinstance(path, list):
            positions = []
            target = self.target_pos
            if target:
                positions.append(target)
            after = self.after_pos
            if after and (not positions or positions[-1] != after):
                positions.append(after)
            return positions
        z, y, x = start
        positions = [start]
        for direction in path:
            if direction == "left":
                x -= 1
            elif direction == "right":
                x += 1
            elif direction == "up":
                y -= 1
            elif direction == "down":
                y += 1
            else:
                return []
            positions.append((z, y, x))
        after = self.after_pos
        if after and positions[-1] != after:
            positions.append(after)
        return positions

    def is_visualizer_noop(self) -> bool:
        return "fakeWall" in self.label and self.is_state_noop()

    def needs_visualizer_state_sync(self) -> bool:
        if self.after is None or self.is_state_noop():
            return False
        if "fakeWall" in self.label:
            return True
        if self.label.startswith("open "):
            return False
        return any(token in self.label for token in ("upFloor", "downFloor")) or self.label.startswith(("fight ", "go "))

    def is_state_noop(self) -> bool:
        """Whether the solver step only changes position/step count.

        The visualizer hides stairs as policy actions, so pure walking/stair
        solver steps sometimes need to be synchronized visually instead of
        executed as a normal macro action.
        """

        if not self.before or not self.after:
            return False
        scalar_keys = ("hp", "atk", "def", "mdef", "money")
        if any(self.before.get(key) != self.after.get(key) for key in scalar_keys):
            return False
        if self.before.get("keys") != self.after.get("keys"):
            return False
        if self.before.get("flags") != self.after.get("flags"):
            return False
        return True


class RoutePlayback:
    def __init__(self, steps: list[RouteStep], path: Path):
        self.steps = steps
        self.path = path
        self.cursor = 0

    @property
    def done(self) -> bool:
        return self.cursor >= len(self.steps)

    @property
    def total(self) -> int:
        return len(self.steps)

    def reset(self) -> None:
        self.cursor = 0

    def current_step(self) -> RouteStep | None:
        if self.done:
            return None
        return self.steps[self.cursor]

    def advance(self) -> None:
        if not self.done:
            self.cursor += 1

    def progress_text(self) -> str:
        name = self.path.name if self.path else "未加载"
        return f"{name}  {self.cursor}/{self.total}"

    def find_visualizer_action(self, env):
        step = self.current_step()
        if step is None:
            return None, [], False, None
        target = step.target_pos
        actions = env.get_feasible_actions()

        current = env.n2p[env.observation[-1]][:3]
        path_match = self._find_path_action(env, actions, current)
        if path_match is not None:
            action, matched_cursor = path_match
            matched_step = self.steps[matched_cursor]
            return action, actions, env.n2p[action][:3] == matched_step.target_pos, matched_cursor

        for action in actions:
            if env.n2p[action][:3] == target:
                return action, actions, True, self.cursor
        return None, actions, False, None

    def align_to_current(self, env, lookahead: int = 32) -> bool:
        """Return whether the current visualizer position still lies on the route corridor."""

        if self.done:
            return True
        current = env.n2p[env.observation[-1]][:3]
        for pos, step_index in self._combined_path_positions(lookahead=lookahead):
            if pos == current:
                if step_index > self.cursor and self._can_skip_to_step(step_index):
                    self.cursor = step_index
                elif step_index > self.cursor:
                    continue
                return True
        return False

    def _find_path_action(self, env, actions, current: tuple[int, int, int]):
        combined = self._combined_path_positions()
        if not combined:
            return None
        current_indices = [idx for idx, (pos, _step_index) in enumerate(combined) if pos == current]
        if not current_indices:
            return None
        current_index = current_indices[0]
        remaining: dict[tuple[int, int, int], tuple[int, int]] = {}
        for idx, (pos, step_index) in enumerate(combined[current_index + 1:], current_index + 1):
            if self._can_skip_to_step(step_index):
                remaining.setdefault(pos, (idx, step_index))
        candidates = []
        for action in actions:
            pos = env.n2p[action][:3]
            if pos in remaining:
                route_index, step_index = remaining[pos]
                candidates.append((route_index, step_index, action))
        if not candidates:
            return None
        _route_index, step_index, action = min(candidates, key=lambda item: item[0])
        return action, step_index

    def _can_skip_to_step(self, step_index: int) -> bool:
        if step_index <= self.cursor:
            return True
        for index in range(self.cursor, min(step_index, self.total)):
            if not self._can_auto_skip_step(self.steps[index]):
                return False
        return True

    @staticmethod
    def _can_auto_skip_step(step: RouteStep) -> bool:
        return step.is_state_noop() or step.is_visualizer_noop()

    def _combined_path_positions(self, lookahead: int = 32) -> list[tuple[tuple[int, int, int], int]]:
        positions: list[tuple[tuple[int, int, int], int]] = []
        end = min(self.total, self.cursor + max(1, lookahead))
        for step_index in range(self.cursor, end):
            step = self.steps[step_index]
            for pos in step.path_positions():
                if positions and positions[-1][0] == pos:
                    continue
                positions.append((pos, step_index))
        return positions


def load_route(path: str | Path) -> RoutePlayback:
    route_path = Path(path)
    steps: list[RouteStep] = []
    for line_no, line in enumerate(route_path.read_text(encoding="utf8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        action = row.get("action")
        if not isinstance(action, dict):
            raise ValueError(f"{route_path}:{line_no} missing action object")
        if "target" not in action:
            raise ValueError(f"{route_path}:{line_no} missing action.target")
        steps.append(
            RouteStep(
                index=int(row.get("index", len(steps))),
                action=action,
                before=row.get("before") if isinstance(row.get("before"), dict) else None,
                after=row.get("after") if isinstance(row.get("after"), dict) else None,
                raw=row,
            )
        )
    if not steps:
        raise ValueError(f"{route_path} has no route steps")
    return RoutePlayback(steps, route_path)


def floor_to_index(floor_id: str) -> int:
    if not floor_id.startswith("MT"):
        raise ValueError(f"Unsupported floor id: {floor_id!r}")
    return int(floor_id[2:]) - 1


def state_to_visualizer_pos(state: dict[str, Any]) -> tuple[int, int, int] | None:
    floor = state.get("floor")
    x = state.get("x")
    y = state.get("y")
    if floor is None or x is None or y is None:
        return None
    return floor_to_index(str(floor)), int(y), int(x)


def target_to_visualizer_pos(target: Any) -> tuple[int, int, int]:
    if not isinstance(target, (list, tuple)) or len(target) != 3:
        raise ValueError(f"Unsupported route target: {target!r}")
    floor_id, x, y = target
    return floor_to_index(str(floor_id)), int(y), int(x)


def visualizer_pos_text(pos: tuple[int, int, int]) -> str:
    z, y, x = pos
    return f"{z + 1}F({y},{x})"
