# -*- coding: utf-8 -*-
"""
10层魔塔 PPO 独立训练脚本

用法：
    python train.py --rounds 2000 --save model/ppo_10floor.pth
"""
import os
import argparse
import numpy as np

from environment import Mota
from PPO import PPO, compute_sword_reward, PPO_REWARD_RATE
from GNN import MapGNNEncoder


def train(rounds: int = 2000, save_path: str = 'model/ppo_10floor.pth'):
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

        while ending == 'continue':
            actions = env.get_feasible_actions()
            if not actions:
                ending = 'stop'
                break

            # 策略网络采样动作
            action, info = agent.choose_action(env, actions)

            # 计算奖励（去掉金币项）
            before = env.get_player_state()
            ending = env.step(action, return_reward=False)
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

            if done:
                break

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='10层魔塔 PPO 训练')
    parser.add_argument('--rounds', type=int, default=1000, help='训练回合数 (默认: 2000)')
    parser.add_argument('--save', type=str, default='model/ppo_10floor.pth', help='模型保存路径')
    parser.add_argument('--demo', action='store_true', help='演示模式（使用已保存的模型）')
    args = parser.parse_args()

    if args.demo:
        demo(args.save)
    else:
        train(args.rounds, args.save)
