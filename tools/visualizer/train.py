# -*- coding: utf-8 -*-
"""
10层魔塔 PPO 独立训练脚本

用法：
    python train.py --rounds 2000 --save model/ppo_10floor.pth
"""
import os
import sys
import argparse
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from environment import Mota
from PPO import PPO, compute_sword_reward, PPO_REWARD_RATE
from GNN import MapGNNEncoder


def train(rounds: int = 2000, save_path: str = 'model/ppo_10floor.pth', max_steps: int = 180):
    # 建立环境
    env = Mota()
    env.build_env('10層魔塔')
    env.create_nodes()

    # 建立 GNN 编码器（与 PPO 端到端训练）
    gnn_encoder = MapGNNEncoder(output_dim=64, hidden_dim=128, num_layers=3)
    # 建立 PPO Agent
    agent = PPO(emb_map_dim=64, emb_state_dim=32, emb_action_dim=32, gnn_encoder=gnn_encoder)

    sword_collected = False
    best_hp = 0

    print(f'[PPO] 开始训练，目标：10层魔塔第5层的剑 (4,11,11)')
    print(f'[PPO] 训练回合数: {rounds}')

    for episode in range(rounds):
        env.reset()
        ending = 'continue'
        step_count = 0
        recent_positions = [env.n2p[env.observation[-1]][:3]]
        last_action_was_stair = False

        while ending == 'continue' and step_count < max_steps:
            actions = env.get_feasible_actions()
            actions = filter_stair_loop_actions(env, actions, recent_positions, last_action_was_stair)
            if not actions:
                ending = 'stop'
                break

            # 策略网络采样动作
            action, info = agent.choose_action(env, actions)
            before_pos = env.n2p[env.observation[-1]][:3]

            # 计算奖励（去掉金币项）
            before = env.get_player_state()
            ending = env.step(action, return_reward=False)
            after_pos = env.n2p[env.observation[-1]][:3]
            was_stair = getattr(action, 'id', '') in {'upFloor', 'downFloor'}
            after = env.get_player_state()

            if ending == 'stop':
                base_reward = -9999.0
            else:
                base_reward = float(np.sum((after - before) * PPO_REWARD_RATE))

            # 目标导向奖励塑形
            reward = compute_sword_reward(env, base_reward, ending, action)

            done = (ending != 'continue')

            # 拿到剑则强制结束本回合并标记成功
            if env.n2p[action] == (4, 11, 11):
                print(f'[Episode {episode + 1}] 拿到剑！剩余生命: {env.player.hp}')
                sword_collected = True
                done = True
                if env.player.hp > best_hp:
                    best_hp = env.player.hp

            agent.store_transition(info, reward, done)
            step_count += 1
            last_action_was_stair = was_stair
            recent_positions.append(after_pos)
            recent_positions = recent_positions[-10:]

            if done:
                break
        if ending == 'continue' and step_count >= max_steps:
            ending = 'timeout'

        # 回合结束，必要时触发 PPO 更新
        updated = agent.end_episode()

        # 打印进度
        if (episode + 1) % 100 == 0 or updated:
            status = 'UPDATED' if updated else f'hp={env.player.hp}'
            print(f'[Episode {episode + 1}/{rounds}] {status}  steps={step_count}')

    # 训练结束，最后更新一次
    agent.update()

    # 保存模型（PPO 与 GNN 打包保存在同一文件中）
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    agent.save(save_path)
    print(f'\n[PPO] 训练完毕，模型已保存至: {save_path}')

    if sword_collected:
        print(f'[PPO] 训练过程中已成功拿到第五层的剑！最佳剩余生命: {best_hp}')
    else:
        print('[PPO] 警告：训练过程中未成功拿到第五层的剑，建议增加训练回数。')


def demo(model_path: str = 'model/ppo_10floor.pth'):
    """使用训练好的模型进行贪婪策略演示"""
    env = Mota()
    env.build_env('10層魔塔')
    env.create_nodes()

    gnn_encoder = MapGNNEncoder(output_dim=64, hidden_dim=128, num_layers=3)
    agent = PPO(emb_map_dim=64, emb_state_dim=32, emb_action_dim=32, gnn_encoder=gnn_encoder)
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f'[Demo] 已加载模型: {model_path}')
    else:
        print(f'[Demo] 模型文件不存在，使用随机初始化: {model_path}')

    env.reset()
    ending = 'continue'
    step_count = 0
    path = [env.n2p[env.observation[-1]]]

    while ending == 'continue':
        actions = env.get_feasible_actions()
        if not actions:
            ending = 'stop'
            break

        if path and len(path) >= 2:
            current = env.n2p[env.observation[-1]]
            filtered = []
            for candidate in actions:
                pos = env.n2p[candidate]
                is_stair = getattr(candidate, 'id', '') in {'upFloor', 'downFloor'}
                if is_stair and pos == current and getattr(env.observation[-2], 'id', '') in {'upFloor', 'downFloor'}:
                    continue
                filtered.append(candidate)
            if filtered:
                actions = filtered

        action = agent.greedy_action(env, actions)
        ending = env.step(action, return_reward=False)
        path.append(env.n2p[action])
        step_count += 1

        if env.n2p[action] == (4, 11, 11):
            print(f'[Demo] 成功拿到剑！')
            break

    print(f'[Demo] 结束状态: {ending}')
    print(f'[Demo] 剩余生命: {env.player.hp}')
    print(f'[Demo] 行动步数: {step_count}')
    print(f'[Demo] 路径: {path}')


def filter_stair_loop_actions(env, actions: list, recent_positions: list, last_action_was_stair: bool) -> list:
    if not actions or not env.observation:
        return actions
    current = env.n2p[env.observation[-1]][:3]
    recent = set((recent_positions or [])[-6:])
    filtered = []
    skipped = []
    for action in actions:
        pos = env.n2p.get(action)
        pos = None if pos is None else pos[:3]
        is_stair = getattr(action, 'id', '') in {'upFloor', 'downFloor'}
        loop_like = False
        if pos is not None and is_stair:
            if last_action_was_stair and pos == current:
                loop_like = True
            elif pos in recent and len(recent) >= 2:
                loop_like = True
        if loop_like:
            skipped.append(action)
        else:
            filtered.append(action)
    return filtered if filtered and skipped else actions


def stair_loop_reward_penalty(action, recent_positions: list, last_action_was_stair: bool, before_pos, after_pos) -> float:
    # Deprecated: stair bounces are filtered/masked, not shaped through reward.
    return 0.0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='10层魔塔 PPO 训练')
    parser.add_argument('--rounds', type=int, default=1000, help='训练回合数 (默认: 2000)')
    parser.add_argument('--save', type=str, default='model/ppo_10floor.pth', help='模型保存路径')
    parser.add_argument('--max-steps', type=int, default=180, help='单回合最大步数，避免楼梯循环卡死')
    parser.add_argument('--demo', action='store_true', help='演示模式（使用已保存的模型）')
    args = parser.parse_args()

    if args.demo:
        demo(args.save)
    else:
        train(args.rounds, args.save, args.max_steps)
