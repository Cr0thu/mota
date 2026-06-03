# -*- coding: utf-8 -*-
"""Standalone tabular Q-learning trainer for the 10-floor visualizer env."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
os.chdir(BASE_DIR)
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from environment import Mota  # noqa: E402
from q_learning import TabularQLearningAgent  # noqa: E402
from stage_reward import stage_action_priors, stage_potential, transition_reward  # noqa: E402

try:
    from PPO import action_prior_logits  # noqa: E402
except Exception:  # pragma: no cover - torch may be unavailable in some envs.
    action_prior_logits = None


def _format_pos(pos):
    z, y, x = pos[:3]
    return f"{z + 1}F({y},{x})"


def _action_prior_values(env, actions):
    stage_priors = stage_action_priors(env, actions)
    if action_prior_logits is None:
        return stage_priors
    try:
        ppo_priors = list(action_prior_logits(env, actions))
    except Exception:
        return stage_priors
    return [stage + 0.25 * ppo for stage, ppo in zip(stage_priors, ppo_priors)]


def train(
    rounds: int,
    max_steps: int,
    save_path: Path,
    seed: int = 0,
    reset_q: bool = False,
    stop_on_stage_advance: bool = False,
    success_replay_bonus: float = 3000.0,
    start_route: Path | None = None,
    start_stage: str | None = None,
):
    env = Mota()
    env.build_env("10層魔塔")
    env.create_nodes()

    if save_path.exists() and not reset_q:
        agent = TabularQLearningAgent.load(save_path)
    else:
        agent = TabularQLearningAgent(seed=seed)
    agent.epsilon = 0.25

    successes = 0
    boss_successes = 0
    best_hp = 0
    best_quality = 0.0
    start_rows = load_route_rows(start_route) if start_route else []
    for episode in range(1, rounds + 1):
        env.reset()
        if start_rows:
            replay_start_route(env, agent, start_rows, start_stage)
        ending = "continue"
        steps = 0
        episode_pairs: list[tuple[str, str]] = []
        while ending == "continue" and steps < max_steps:
            actions = env.get_feasible_actions()
            if not actions:
                ending = "stop"
                break
            priors = _action_prior_values(env, actions)
            state_key = agent.state_key(env)
            action, _index, _mode = agent.choose_action(env, actions, priors)
            action_key = agent.action_key(env, action)
            episode_pairs.append((state_key, action_key))
            before_phi = stage_potential(env)
            before = env.get_player_state().copy()
            ending = env.step(action, return_reward=False)
            after = env.get_player_state().copy()
            reward_info = transition_reward(env, action, before, after, ending, before_phi)
            reward = reward_info.total
            done = ending != "continue"
            if reward_info.after.stage != reward_info.before.stage:
                successes += 1
                best_hp = max(best_hp, env.player.hp)
                replay_value = quality_weighted_success_bonus(env, reward_info, steps + 1, success_replay_bonus)
                best_quality = max(best_quality, replay_value)
                replay_success_path(agent, episode_pairs, replay_value)
                if stop_on_stage_advance:
                    done = True
            if reward_info.after.stage == "done":
                boss_successes += 1
                best_hp = max(best_hp, env.player.hp)
                done = True
            steps += 1
            if not done and steps >= max_steps:
                reward -= 500.0
                done = True
                ending = "timeout"
            next_actions = [] if done else env.get_feasible_actions()
            next_state_key = None if done else agent.state_key(env)
            next_action_keys = [agent.action_key(env, candidate) for candidate in next_actions]
            agent.update(state_key, action_key, reward, next_state_key, next_action_keys, done)
            if done:
                break
        agent.episodes += 1
        if episode % 20 == 0 or episode == rounds:
            print(
                f"[Q {episode}/{rounds}] ending={ending} steps={steps} hp={env.player.hp} "
                f"stage={stage_potential(env).stage} stage_advances={successes} "
                f"boss={boss_successes} best_replay={best_quality:.0f} q_states={len(agent.q)}"
            )

    agent.save(save_path)
    print(f"[Q] saved: {save_path}")
    print(f"[Q] stage_advances={successes}, boss_successes={boss_successes}/{rounds}, best_hp={best_hp}")
    print(f"[Q] best_quality_replay_bonus={best_quality:.1f}")
    return agent


def replay_success_path(agent: TabularQLearningAgent, episode_pairs: list[tuple[str, str]], bonus: float):
    value = float(bonus)
    for state_key, action_key in reversed(episode_pairs):
        agent.update(state_key, action_key, value, None, [], True)
        value *= agent.gamma


def quality_weighted_success_bonus(env: Mota, reward_info, steps: int, base_bonus: float) -> float:
    p = env.player
    yk = p.items.get("yellowKey", 0)
    bk = p.items.get("blueKey", 0)
    rk = p.items.get("redKey", 0)
    stage = reward_info.before.stage
    if stage == "sword":
        quality = (
            0.35
            + min(max(p.hp, 0), 900) / 900.0 * 0.55
            + min(yk, 4) * 0.18
            + min(bk, 2) * 0.22
            - max(0, steps - 38) * 0.018
        )
        if yk <= 0:
            quality -= 0.22
    elif stage == "shield":
        quality = (
            0.45
            + min(max(p.hp, 0), 1400) / 1400.0 * 0.45
            + min(yk, 5) * 0.10
            + min(bk, 2) * 0.18
            - max(0, steps - 55) * 0.012
        )
    elif stage == "gems":
        quality = 0.65 + min(max(p.hp, 0), 1800) / 1800.0 * 0.35 + min(yk, 5) * 0.08
    elif stage == "red_key":
        quality = 0.75 + min(max(p.hp, 0), 2200) / 2200.0 * 0.35 + min(rk, 1) * 0.35
    else:
        quality = 1.0 + min(max(p.hp, 0), 2500) / 2500.0 * 0.4
    quality = max(0.18, min(2.6, quality))
    return float(base_bonus) * quality


def load_route_rows(path: Path | None) -> list[dict]:
    if path is None:
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def replay_start_route(env: Mota, agent: TabularQLearningAgent, rows: list[dict], stop_stage: str | None):
    for row in rows:
        action_key = row.get("action")
        action = find_action_by_key(env, agent, action_key)
        if action is None:
            break
        ending = env.step(action, return_reward=False)
        if ending != "continue":
            break
        if stop_stage and row.get("stage_after") == stop_stage:
            break


def find_action_by_key(env: Mota, agent: TabularQLearningAgent, action_key: str | None):
    if not action_key:
        return None
    for action in env.get_feasible_actions():
        if agent.action_key(env, action) == action_key:
            return action
    return None


def demo(save_path: Path, max_steps: int, route_out: Path | None = None):
    env = Mota()
    env.build_env("10層魔塔")
    env.create_nodes()
    agent = TabularQLearningAgent.load(save_path)
    agent.epsilon = 0.0
    env.reset()
    route = []
    ending = "continue"
    steps = 0
    while ending == "continue" and steps < max_steps:
        actions = env.get_feasible_actions()
        if not actions:
            ending = "stop"
            break
        priors = _action_prior_values(env, actions)
        state_key = agent.state_key(env)
        before_phi = stage_potential(env)
        action, index = agent.greedy_action(env, actions, priors)
        q_before = agent.q_value(env, action, state_key)
        stage_before = before_phi.stage
        before = env.get_player_state().copy()
        before_pos = env.n2p[env.observation[-1]][:3]
        ending = env.step(action, return_reward=False)
        after = env.get_player_state().copy()
        after_pos = env.n2p[env.observation[-1]][:3]
        reward_info = transition_reward(env, action, before, after, ending, before_phi)
        route.append(
            {
                "step": steps + 1,
                "action": agent.action_key(env, action),
                "target_pos": after_pos,
                "target": _format_pos(env.n2p[action]),
                "from": _format_pos(before_pos),
                "to": _format_pos(after_pos),
                "hp": env.player.hp,
                "atk": env.player.atk,
                "def": env.player.def_,
                "keys": {
                    "yellow": env.player.items.get("yellowKey", 0),
                    "blue": env.player.items.get("blueKey", 0),
                    "red": env.player.items.get("redKey", 0),
                },
                "q": q_before,
                "reward": reward_info.total,
                "stage_before": stage_before,
                "stage_after": reward_info.after.stage,
            }
        )
        steps += 1
        if reward_info.after.stage == "done":
            break
    print(f"[Q Demo] ending={ending} steps={steps} hp={env.player.hp}")
    for row in route[:30]:
        print(
            f"{row['step']:03d} {row['from']} -> {row['target']} -> {row['to']} "
            f"q={row['q']:.2f} r={row['reward']:.1f} {row['stage_before']}->{row['stage_after']}"
        )
    if len(route) > 30:
        print(f"... {len(route) - 30} more steps")
    if route_out is not None:
        route_out.parent.mkdir(parents=True, exist_ok=True)
        route_out.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in route), encoding="utf-8")
        print(f"[Q Demo] route saved: {route_out}")


def print_initial_values(save_path: Path):
    env = Mota()
    env.build_env("10層魔塔")
    env.create_nodes()
    env.reset()
    agent = TabularQLearningAgent.load(save_path)
    actions = env.get_feasible_actions()
    priors = _action_prior_values(env, actions)
    values = agent.action_values(env, actions, priors)
    rows = sorted(zip(actions, values), key=lambda pair: pair[1]["score"], reverse=True)
    print("[Q] current initial action ranking:")
    for action, value in rows[:10]:
        print(
            f"  score={value['score']:8.2f} q={value['q']:8.2f} prior={value['prior']:5.2f} "
            f"{_format_pos(env.n2p[action])} {action.class_}:{action.id}"
        )


def _resolve_project_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="10层魔塔 tabular Q-learning")
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=180)
    parser.add_argument("--save", type=Path, default=PROJECT_ROOT / "artifacts/runs/visualizer_qlearning/q_table.json")
    parser.add_argument("--reset-q", action="store_true", help="Ignore any existing Q table at --save")
    parser.add_argument(
        "--stop-on-stage-advance",
        action="store_true",
        help="End an episode as soon as the current route reaches the next stage.",
    )
    parser.add_argument("--success-replay-bonus", type=float, default=3000.0)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--route-out", type=Path, default=None)
    parser.add_argument("--print-values", action="store_true")
    parser.add_argument("--start-route", type=Path, default=None)
    parser.add_argument("--start-stage", type=str, default=None)
    args = parser.parse_args()
    args.save = _resolve_project_path(args.save)
    args.route_out = _resolve_project_path(args.route_out)
    args.start_route = _resolve_project_path(args.start_route)

    if args.demo:
        demo(args.save, args.max_steps, args.route_out)
    else:
        train(
            args.rounds,
            args.max_steps,
            args.save,
            reset_q=args.reset_q,
            stop_on_stage_advance=args.stop_on_stage_advance,
            success_replay_bonus=args.success_replay_bonus,
            start_route=args.start_route,
            start_stage=args.start_stage,
        )
    if args.print_values:
        print_initial_values(args.save)
