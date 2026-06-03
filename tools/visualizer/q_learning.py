# -*- coding: utf-8 -*-
"""Small tabular Q-learning baseline for the visualizer macro-action env.

This module intentionally does not depend on torch.  It learns values for the
current macro-action abstraction used by the Tk visualizer: each action is a
reachable interactive target such as an item, monster, door, NPC, or event.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Iterable


class TabularQLearningAgent:
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 0.97,
        epsilon: float = 0.20,
        prior_weight: float = 1.00,
        explore_bonus: float = 1.25,
        seed: int = 0,
    ):
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.prior_weight = float(prior_weight)
        self.explore_bonus = float(explore_bonus)
        self.q: dict[str, dict[str, float]] = {}
        self.visits: dict[str, dict[str, int]] = {}
        self.rng = random.Random(seed)
        self.training_steps = 0
        self.episodes = 0

    def state_key(self, env) -> str:
        p = env.player
        pos = env.n2p[env.observation[-1]][:3]
        flags = tuple(sorted((str(k), self._stable_flag_value(v)) for k, v in getattr(env, "flags", {}).items()))
        consumed_hash = self._consumed_hash(env)
        return json.dumps(
            {
                "version": 2,
                "pos": pos,
                "hp": int(p.hp),
                "atk": int(p.atk),
                "def": int(p.def_),
                "money": int(p.money),
                "exp": int(p.exp),
                "yk": int(p.items.get("yellowKey", 0)),
                "bk": int(p.items.get("blueKey", 0)),
                "rk": int(p.items.get("redKey", 0)),
                "flags": flags,
                "map": consumed_hash,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def _stable_flag_value(self, value):
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value) if value.is_integer() else value
        return str(value)

    def _consumed_hash(self, env) -> str:
        parts = []
        for node, pos in env.n2p.items():
            node_id = getattr(node, "id", "")
            if node is env.player or node_id in {"upFloor", "downFloor", "background"}:
                continue
            if getattr(node, "activated", False) or getattr(node, "disabled", False):
                disabled = "D" if getattr(node, "disabled", False) else "A"
                parts.append(f"{pos}:{getattr(node, 'class_', '')}:{node_id}:{disabled}")
        raw = "|".join(sorted(parts)).encode("utf-8")
        return hashlib.sha1(raw).hexdigest()[:12]

    def action_key(self, env, action) -> str:
        pos = env.n2p.get(action)
        if pos is None:
            pos = ("?", "?", "?")
        pos_text = ":".join(str(v) for v in pos[:3])
        suffix = "" if len(pos) <= 3 else ":" + ":".join(str(v) for v in pos[3:])
        return f"{getattr(action, 'class_', '?')}:{getattr(action, 'id', '?')}:{pos_text}{suffix}"

    def q_value(self, env, action, state_key: str | None = None) -> float:
        s_key = state_key or self.state_key(env)
        return float(self.q.get(s_key, {}).get(self.action_key(env, action), 0.0))

    def action_values(self, env, actions: Iterable, priors: Iterable[float] | None = None):
        actions = list(actions)
        s_key = self.state_key(env)
        prior_list = list(priors) if priors is not None else [0.0] * len(actions)
        values = []
        for action, prior in zip(actions, prior_list):
            action_key = self.action_key(env, action)
            q = self.q_value(env, action, s_key)
            count = int(self.visits.get(s_key, {}).get(action_key, 0))
            bonus = self.explore_bonus / math.sqrt(1.0 + count)
            score = q + self.prior_weight * float(prior) + bonus
            values.append({"q": q, "prior": float(prior), "bonus": bonus, "score": score})
        return values

    def choose_action(self, env, actions: list, priors: Iterable[float] | None = None):
        if not actions:
            raise ValueError("choose_action requires at least one action")
        if self.rng.random() < self.epsilon:
            index = self.rng.randrange(len(actions))
            return actions[index], index, "explore"
        values = self.action_values(env, actions, priors)
        best = max(range(len(actions)), key=lambda idx: (values[idx]["score"], -idx))
        return actions[best], best, "greedy"

    def update(
        self,
        state_key: str,
        action_key: str,
        reward: float,
        next_state_key: str | None,
        next_action_keys: Iterable[str] | None,
        done: bool,
    ) -> dict[str, float]:
        row = self.q.setdefault(state_key, {})
        self.visits.setdefault(state_key, {})[action_key] = self.visits.setdefault(state_key, {}).get(action_key, 0) + 1
        old = float(row.get(action_key, 0.0))
        if done or not next_state_key or not next_action_keys:
            bootstrap = 0.0
        else:
            next_row = self.q.get(next_state_key, {})
            bootstrap = max((float(next_row.get(a_key, 0.0)) for a_key in next_action_keys), default=0.0)
        target = float(reward) + self.gamma * bootstrap
        new_value = old + self.alpha * (target - old)
        row[action_key] = new_value
        self.training_steps += 1
        return {"old": old, "target": target, "new": new_value, "td_error": target - old}

    def greedy_action(self, env, actions: list, priors: Iterable[float] | None = None):
        if not actions:
            raise ValueError("greedy_action requires at least one action")
        values = self.action_values(env, actions, priors)
        best = max(range(len(actions)), key=lambda idx: (values[idx]["score"], -idx))
        return actions[best], best

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "prior_weight": self.prior_weight,
            "explore_bonus": self.explore_bonus,
            "training_steps": self.training_steps,
            "episodes": self.episodes,
            "q": self.q,
            "visits": self.visits,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TabularQLearningAgent":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        agent = cls(
            alpha=payload.get("alpha", 0.25),
            gamma=payload.get("gamma", 0.97),
            epsilon=payload.get("epsilon", 0.20),
            prior_weight=payload.get("prior_weight", 0.35),
            explore_bonus=payload.get("explore_bonus", 1.25),
        )
        agent.training_steps = int(payload.get("training_steps", 0))
        agent.episodes = int(payload.get("episodes", 0))
        agent.q = {
            str(s_key): {str(a_key): float(value) for a_key, value in row.items()}
            for s_key, row in payload.get("q", {}).items()
        }
        agent.visits = {
            str(s_key): {str(a_key): int(value) for a_key, value in row.items()}
            for s_key, row in payload.get("visits", {}).items()
        }
        return agent
