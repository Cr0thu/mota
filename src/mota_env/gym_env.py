from __future__ import annotations

from typing import Any

from .data import load_game_data
from .rewards import Rewarder
from .simulator import MotaSimulator, MotaState


try:  # Keep the simulator usable on a plain Python install.
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover - exercised only when gymnasium is installed.
    gym = None
    spaces = None


MAX_MACRO_ACTIONS = 256


class MotaMacroEnv(gym.Env if gym else object):
    """Gymnasium-compatible macro-action environment for the first 10 floors.

    The action space is an index into the current state's legal macro actions. Call
    :meth:`action_mask` before selecting actions; illegal or stale action indices are rejected.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        data_path: str = "artifacts/data/mota_first10.json",
        reward_scheme: str = "label_dense",
    ):
        self.sim = MotaSimulator(load_game_data(data_path))
        self.rewarder = Rewarder(reward_scheme)
        self.state: MotaState | None = None
        self.actions: list[dict[str, Any]] = []
        if spaces is not None:
            self.action_space = spaces.Discrete(MAX_MACRO_ACTIONS)
            self.observation_space = spaces.Dict(
                {
                    "floor": spaces.Discrete(len(self.sim.floor_order)),
                    "position": spaces.Box(0, 12, shape=(2,), dtype=int),
                    "hero": spaces.Box(0, 1_000_000, shape=(9,), dtype=int),
                    "grid": spaces.Box(0, 999, shape=(13, 13), dtype=int),
                }
            )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if gym is not None:
            super().reset(seed=seed)
        self.state = self.sim.reset()
        self._refresh_actions()
        observation = self._observation()
        info = {"action_mask": self.action_mask(), "actions": self.actions}
        return observation, info

    def step(self, action: int):
        if self.state is None:
            raise RuntimeError("Call reset() before step().")
        if action < 0 or action >= len(self.actions):
            return self._observation(), -1.0, False, False, {
                "message": "illegal macro action index",
                "action_mask": self.action_mask(),
            }
        macro_action = self.actions[action]
        before = self.state.clone()
        transition = self.sim.apply_macro_action(self.state, macro_action)
        terminated = self.state.done or self.state.dead
        reward = self.rewarder.score(
            self.sim,
            before,
            self.state,
            macro_action,
            transition,
        ).total
        self._refresh_actions()
        return self._observation(), reward, terminated, False, {
            "message": transition.message,
            "action": macro_action,
            "action_mask": self.action_mask(),
            "actions": self.actions,
        }

    def action_mask(self) -> list[bool]:
        return [idx < len(self.actions) for idx in range(MAX_MACRO_ACTIONS)]

    def render(self) -> str:
        if self.state is None:
            return "<uninitialized>"
        return (
            f"{self.state.floor_id} ({self.state.x},{self.state.y}) "
            f"hp={self.state.hp} atk={self.state.atk} def={self.state.defense} "
            f"money={self.state.money} keys="
            f"{self.state.items.get('yellowKey', 0)}/"
            f"{self.state.items.get('blueKey', 0)}/"
            f"{self.state.items.get('redKey', 0)}"
        )

    def _refresh_actions(self) -> None:
        if self.state is None or self.state.done or self.state.dead:
            self.actions = []
        else:
            self.actions = self.sim.macro_actions(self.state)[:MAX_MACRO_ACTIONS]

    def _observation(self) -> dict[str, Any]:
        if self.state is None:
            raise RuntimeError("Call reset() before observation.")
        floor = self.state.floors[self.state.floor_id]
        return {
            "floor": self.sim.floor_order.index(self.state.floor_id),
            "position": [self.state.x, self.state.y],
            "hero": [
                self.state.hp,
                self.state.atk,
                self.state.defense,
                self.state.mdef,
                self.state.money,
                self.state.exp,
                self.state.items.get("yellowKey", 0),
                self.state.items.get("blueKey", 0),
                self.state.items.get("redKey", 0),
            ],
            "grid": floor,
        }
