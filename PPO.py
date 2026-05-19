# -*- coding: utf-8 -*-
import numpy as np
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# ============================================================================
#  PPO        by Hung1 (PPO extension)
# ----------------------------------------------------------------------------
#  使用 PPO (Proximal Policy Optimization) 來做學習，適用於魔塔環境。
#  與 QLearning 最大的差異：
#    1. 用神經網路逼近策略 π(a|s) 與價值函數 V(s)，不需要為每個狀態建立 Q 表。
#    2. 狀態使用主角屬性 + 當前位置的數值向量，而非 LZW 壓縮的歷史路徑字串。
#    3. 動作空間是可變的，因此採用「對每個候選動作打分後 softmax」的方式處理。
#    4. 訓練採用 Clipped Surrogate Objective 與 GAE (Generalized Advantage Estimation)。
#  V 1.0   2026/05
# ============================================================================

# 狀態特徵維度：
# [hp, atk, def, mdef, money, exp, yKey, bKey, rKey, step, floor, y, x]
STATE_DIM = 13

# 動作特徵維度：
# [floor, y, x, isEnemy, e_hp, e_atk, e_def, e_money, e_exp, e_special,
#  isItem, isTerrain, isNPC, isEndFlag, bias, special_flag]
ACTION_FEATURE_DIM = 16

# PPO 专用奖励系数（去掉了金币 reward，其余与环境默认保持一致）
# 对应 [hp, atk, def, mdef, money, exp, yKey, bKey, rKey]
PPO_REWARD_RATE = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)


# ============================================================================
#  Actor-Critic 神經網路
# ----------------------------------------------------------------------------
#  1. 先用 state_encoder 把 state 編碼為 emb_state。
#  2. Actor 接收 (emb_state, action_feature) 拼接後打分，每個候選動作獨立計分。
#     所有分數經 softmax 後構成動作機率分布。
#  3. Critic 只接收 emb_state，輸出狀態價值 V(s)。
# ============================================================================
class ActorCritic(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM,
                 action_feature_dim: int = ACTION_FEATURE_DIM,
                 emb_state_dim: int = 32,
                 emb_action_dim: int = 32,
                 emb_map_dim: int = 32,
                 hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.action_feature_dim = action_feature_dim
        self.emb_state_dim = emb_state_dim
        self.emb_action_dim = emb_action_dim
        self.emb_map_dim = emb_map_dim

        # State Encoder: state -> emb_state
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, emb_state_dim),
            nn.Tanh(),
        )

        # Action Encoder: action_feature -> emb_action
        self.action_encoder = nn.Sequential(
            nn.Linear(action_feature_dim, 64),
            nn.Tanh(),
            nn.Linear(64, emb_action_dim),
            nn.Tanh(),
        )

        # Actor: (emb_state + emb_action + map_emb) -> score
        actor_input_dim = emb_state_dim + emb_action_dim + emb_map_dim
        self.actor = nn.Sequential(
            nn.Linear(actor_input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        # Critic: (emb_state + map_emb) -> value
        critic_input_dim = emb_state_dim + emb_map_dim
        self.critic = nn.Sequential(
            nn.Linear(critic_input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action_features: torch.Tensor,
                map_emb: torch.Tensor = None):
        """
        single-state 推理（用於 choose_action）。
        state: shape (state_dim,)
        action_features: shape (n_actions, action_feature_dim)
        map_emb: shape (emb_map_dim,) or None
        return: (scores (n_actions,), value (scalar))
        """
        n = action_features.shape[0]

        # 1. state encoder
        emb_state = self.state_encoder(state)  # (emb_state_dim,)

        # 2. action encoder
        emb_actions = self.action_encoder(action_features)  # (n, emb_action_dim)

        # 3. actor: 每個候選動作獨立打分
        emb_states = emb_state.unsqueeze(0).expand(n, -1)  # (n, emb_state_dim)
        if self.emb_map_dim > 0 and map_emb is not None:
            map_embs = map_emb.unsqueeze(0).expand(n, -1)  # (n, emb_map_dim)
            actor_input = torch.cat([emb_states, emb_actions, map_embs], dim=-1)
            critic_input = torch.cat([emb_state, map_emb], dim=-1)
        else:
            actor_input = torch.cat([emb_states, emb_actions], dim=-1)
            critic_input = emb_state
        scores = self.actor(actor_input).view(n)  # (n,)

        # 4. critic
        value = self.critic(critic_input).squeeze(-1)
        return scores, value


# ============================================================================
#  PPO 主類別
# ============================================================================
class PPO:
    # ------------------------------------------------------------------------
    #  初始化
    # ------------------------------------------------------------------------
    def __init__(self,
                 learning_rate: float = 3e-4,
                 discount_factor: float = 0.99,
                 clip_epsilon: float = 0.2,
                 gae_lambda: float = 0.95,
                 entropy_coef: float = 0.01,
                 value_coef: float = 0.5,
                 update_epochs: int = 4,
                 batch_size: int = 64,
                 update_interval: int = 4,
                 reward_scale: float = 1e-3,
                 hidden_dim: int = 256,
                 emb_map_dim: int = 64,
                 emb_state_dim: int = 32,
                 emb_action_dim: int = 32,
                 gnn_encoder=None,
                 device: str = None):
        self.gamma = discount_factor
        self.clip_eps = clip_epsilon
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.update_interval = update_interval        # 每 N 個回合更新一次
        self.reward_scale = reward_scale              # 縮放原始 reward 避免梯度爆炸
        self.emb_map_dim = emb_map_dim
        self.gnn_encoder = gnn_encoder

        # 自動偵測 GPU。可顯式傳入 'cuda' / 'cuda:0' / 'cpu' 來覆寫
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        self.device = torch.device(device)
        # 啟用 cuDNN benchmark 提升固定輸入尺寸的卷積/線性運算速度
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            print(f'[PPO] Using GPU: {torch.cuda.get_device_name(self.device)}')
        else:
            print('[PPO] Using CPU')

        self.network = ActorCritic(STATE_DIM, ACTION_FEATURE_DIM, emb_state_dim, emb_action_dim, emb_map_dim, hidden_dim).to(self.device)

        # 优化器参数包含 Actor-Critic 和 GNN（如果启用）
        params = list(self.network.parameters())
        if self.gnn_encoder is not None:
            params += list(self.gnn_encoder.gnn.parameters())
        self.optimizer = optim.Adam(params, lr=learning_rate)

        # 軌跡緩衝區（一個或多個回合的 transitions）
        self.trajectory: List[dict] = []
        self.episodes_collected = 0

    # ------------------------------------------------------------------------
    #  將環境狀態編碼為固定維度向量
    # ------------------------------------------------------------------------
    @staticmethod
    def encode_state(env) -> np.ndarray:
        p = env.player
        floors = env.env_data['floors']
        max_l = max(1, floors['layer'])
        max_h = max(1, floors['height'])
        max_w = max(1, floors['width'])
        pos = env.n2p[env.observation[-1]]
        return np.array([
            p.hp / 1000.0,
            p.atk / 200.0,
            p.def_ / 200.0,
            p.mdef / 100.0,
            p.money / 100.0,
            p.exp / 100.0,
            p.items.get('yellowKey', 0) / 10.0,
            p.items.get('blueKey', 0) / 10.0,
            p.items.get('redKey', 0) / 10.0,
            min(env.get_step_count() / 200.0, 1.0),
            pos[0] / max_l,
            pos[1] / max_h,
            pos[2] / max_w,
        ], dtype=np.float32)

    # ------------------------------------------------------------------------
    #  將候選動作列表編碼為特徵矩陣
    # ------------------------------------------------------------------------
    @staticmethod
    def encode_actions(env, actions: list) -> np.ndarray:
        # 延遲匯入避免循環依賴
        from environment import Enemy, Item, Terrain, NPC, EndFlag

        floors = env.env_data['floors']
        max_l = max(1, floors['layer'])
        max_h = max(1, floors['height'])
        max_w = max(1, floors['width'])

        feats = np.zeros((len(actions), ACTION_FEATURE_DIM), dtype=np.float32)
        for i, action in enumerate(actions):
            pos = env.n2p[action]
            feats[i, 0] = pos[0] / max_l
            feats[i, 1] = pos[1] / max_h
            feats[i, 2] = pos[2] / max_w

            if isinstance(action, Enemy):
                feats[i, 3] = 1.0
                feats[i, 4] = action.hp / 1000.0
                feats[i, 5] = action.atk / 200.0
                feats[i, 6] = action.def_ / 200.0
                feats[i, 7] = action.money / 100.0
                feats[i, 8] = action.exp / 100.0
                feats[i, 9] = action.special / 25.0
            elif isinstance(action, Item):
                feats[i, 10] = 1.0
            elif isinstance(action, Terrain):
                feats[i, 11] = 1.0
                # 標記該地形是否需要鑰匙
                if action.id in ('yellowDoor', 'blueDoor', 'redDoor', 'greenDoor'):
                    feats[i, 15] = 1.0
            elif isinstance(action, NPC):
                feats[i, 12] = 1.0
            elif isinstance(action, EndFlag):
                feats[i, 13] = 1.0
            feats[i, 14] = 1.0  # bias
        return feats

    # ------------------------------------------------------------------------
    #  選擇下一步行動（採樣，用於訓練時）
    # ------------------------------------------------------------------------
    def choose_action(self, env, actions: list) -> Tuple[object, dict]:
        state_np = self.encode_state(env)
        feats_np = self.encode_actions(env, actions)

        state_t = torch.from_numpy(state_np).to(self.device, non_blocking=True)
        feats_t = torch.from_numpy(feats_np).to(self.device, non_blocking=True)

        # 若配置了 GNN 编码器，计算地图 embedding 并保存图数据供 update 重新编码
        map_emb = None
        graph_data = None
        if self.gnn_encoder is not None:
            map_emb = self.gnn_encoder.get_embedding(env)
            graph_data = self.gnn_encoder.get_graph_data(env)

        with torch.no_grad():
            if map_emb is not None:
                map_t = torch.from_numpy(map_emb).to(self.device, non_blocking=True)
                scores, value = self.network(state_t, feats_t, map_t)
            else:
                scores, value = self.network(state_t, feats_t)
            dist = Categorical(logits=scores)
            idx = dist.sample()
            log_prob = dist.log_prob(idx)

        info = {
            'state': state_np,
            'action_features': feats_np,
            'action_idx': int(idx.item()),
            'log_prob': float(log_prob.item()),
            'value': float(value.item()),
        }
        if graph_data is not None:
            info['node_features'], info['edge_index'] = graph_data
        return actions[idx.item()], info

    # ------------------------------------------------------------------------
    #  選擇下一步行動（貪婪，用於部署 / 演示）
    # ------------------------------------------------------------------------
    def greedy_action(self, env, actions: list):
        state_t = torch.from_numpy(self.encode_state(env)).to(self.device, non_blocking=True)
        feats_t = torch.from_numpy(self.encode_actions(env, actions)).to(self.device, non_blocking=True)

        map_emb = None
        if self.gnn_encoder is not None:
            map_emb = self.gnn_encoder.get_embedding(env)

        with torch.no_grad():
            if map_emb is not None:
                map_t = torch.from_numpy(map_emb).to(self.device, non_blocking=True)
                scores, _ = self.network(state_t, feats_t, map_t)
            else:
                scores, _ = self.network(state_t, feats_t)
            idx = int(torch.argmax(scores).item())
        return actions[idx]

    # ------------------------------------------------------------------------
    #  儲存單步轉移
    # ------------------------------------------------------------------------
    def store_transition(self, info: dict, reward: float, done: bool):
        info['reward'] = reward * self.reward_scale
        info['done'] = bool(done)
        self.trajectory.append(info)

    # ------------------------------------------------------------------------
    #  完成一個回合：增加計數，必要時觸發更新
    # ------------------------------------------------------------------------
    def end_episode(self) -> bool:
        self.episodes_collected += 1
        if self.episodes_collected >= self.update_interval:
            self.update()
            self.episodes_collected = 0
            return True
        return False

    # ------------------------------------------------------------------------
    #  GAE 計算優勢函數與回報
    # ------------------------------------------------------------------------
    def _compute_gae(self):
        n = len(self.trajectory)
        advantages = np.zeros(n, dtype=np.float32)
        last_adv = 0.0
        next_value = 0.0  # 軌跡末端 V 視為 0（done=True 時也應為 0）
        for t in reversed(range(n)):
            tr = self.trajectory[t]
            mask = 0.0 if tr['done'] else 1.0
            delta = tr['reward'] + self.gamma * next_value * mask - tr['value']
            last_adv = delta + self.gamma * self.gae_lambda * mask * last_adv
            advantages[t] = last_adv
            next_value = tr['value']
            # 跨回合邊界時，下一步價值要重置為 0
            if tr['done']:
                next_value = 0.0
                last_adv = 0.0
        values = np.array([tr['value'] for tr in self.trajectory], dtype=np.float32)
        returns = advantages + values
        return advantages, returns

    # ------------------------------------------------------------------------
    #  PPO 更新（支持可變動作數量）
    # ------------------------------------------------------------------------
    def update(self):
        if len(self.trajectory) < 2:
            self.trajectory.clear()
            return

        advantages, returns = self._compute_gae()
        # 標準化優勢
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = len(self.trajectory)

        # 預先提取固定長度數據
        states_np = np.stack([tr['state'] for tr in self.trajectory])
        idxs_np = np.array([tr['action_idx'] for tr in self.trajectory], dtype=np.int64)
        old_lp_np = np.array([tr['log_prob'] for tr in self.trajectory], dtype=np.float32)
        # 動作特徵保留為 list，因為每個 transition 的動作數量可能不同
        feats_list = [tr['action_features'] for tr in self.trajectory]
        # 是否使用了 GNN（存的是圖數據，供 update 重新 forward GNN）
        has_gnn = 'node_features' in self.trajectory[0]
        if has_gnn:
            graph_data_list = [(tr['node_features'], tr['edge_index']) for tr in self.trajectory]

        states_t = torch.from_numpy(states_np).to(self.device, non_blocking=True)
        idxs_t = torch.from_numpy(idxs_np).to(self.device, non_blocking=True)
        old_lp_t = torch.from_numpy(old_lp_np).to(self.device, non_blocking=True)
        adv_t = torch.from_numpy(advantages).to(self.device, non_blocking=True)
        ret_t = torch.from_numpy(returns).to(self.device, non_blocking=True)

        indices = np.arange(n)
        for _ in range(self.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, n, self.batch_size):
                bi = indices[start:start + self.batch_size]

                b_states = states_t[bi]
                b_idxs = idxs_t[bi]
                b_old_lp = old_lp_t[bi]
                b_adv = adv_t[bi]
                b_ret = ret_t[bi]
                b_feats = [feats_list[i] for i in bi]
                b_graphs = [graph_data_list[i] for i in bi] if has_gnn else None

                # Critic: batched
                emb_states = self.network.state_encoder(b_states)
                if has_gnn:
                    # 逐樣本重新 forward GNN，讓 GNN 能接收梯度（端到端訓練）
                    map_embs = []
                    for nf, ei in b_graphs:
                        nf_t = torch.from_numpy(nf).to(self.device, non_blocking=True)
                        ei_t = torch.from_numpy(ei).to(self.device, non_blocking=True)
                        map_emb = self.gnn_encoder.gnn(nf_t, ei_t)
                        map_embs.append(map_emb)
                    b_map_embs = torch.stack(map_embs)  # (batch, emb_map_dim)
                    critic_input = torch.cat([emb_states, b_map_embs], dim=-1)
                    values = self.network.critic(critic_input).squeeze(-1)
                else:
                    values = self.network.critic(emb_states).squeeze(-1)

                # Actor: 每個樣本的動作數量不同，逐個計算
                new_lps = []
                entropies = []
                for j, feat in enumerate(b_feats):
                    feat_t = torch.from_numpy(feat).to(self.device, non_blocking=True)
                    emb_state = emb_states[j]
                    emb_actions = self.network.action_encoder(feat_t)  # (n_actions, emb_action_dim)
                    emb_states_exp = emb_state.unsqueeze(0).expand(feat_t.shape[0], -1)
                    if has_gnn:
                        map_emb = b_map_embs[j]
                        map_embs_exp = map_emb.unsqueeze(0).expand(feat_t.shape[0], -1)
                        actor_input = torch.cat([emb_states_exp, emb_actions, map_embs_exp], dim=-1)
                    else:
                        actor_input = torch.cat([emb_states_exp, emb_actions], dim=-1)
                    scores = self.network.actor(actor_input).view(-1)
                    dist = Categorical(logits=scores)
                    new_lps.append(dist.log_prob(b_idxs[j]))
                    entropies.append(dist.entropy())

                new_lp = torch.stack(new_lps)
                entropy = torch.stack(entropies).mean()

                ratio = torch.exp(new_lp - b_old_lp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (values - b_ret).pow(2).mean()

                total_loss = (policy_loss
                              + self.value_coef * value_loss
                              - self.entropy_coef * entropy)

                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                # 梯度裁剪同時覆蓋 network + gnn
                all_params = list(self.network.parameters())
                if self.gnn_encoder is not None:
                    all_params += list(self.gnn_encoder.gnn.parameters())
                torch.nn.utils.clip_grad_norm_(all_params, 0.5)
                self.optimizer.step()

        self.trajectory.clear()

    # ------------------------------------------------------------------------
    #  儲存 / 載入模型
    # ------------------------------------------------------------------------
    def save(self, path: str):
        checkpoint = {'network': self.network.state_dict()}
        if self.gnn_encoder is not None:
            checkpoint['gnn'] = self.gnn_encoder.gnn.state_dict()
        torch.save(checkpoint, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'network' in checkpoint:
            self.network.load_state_dict(checkpoint['network'])
            if self.gnn_encoder is not None and 'gnn' in checkpoint:
                self.gnn_encoder.gnn.load_state_dict(checkpoint['gnn'])
        else:
            # 兼容旧格式（仅保存了 network state_dict）
            self.network.load_state_dict(checkpoint)


# ============================================================================
#  PPO 奖励塑形 —— 目标：取得 10 层魔塔第五层的剑
# ============================================================================
# 第五层剑在 env 坐标中的位置 (z=4, y=11, x=11)
_SWORD_POS = (4, 11, 11)


def compute_sword_reward(env, base_reward: float, ending: str, action) -> float:
    """
    在环境默认奖励（属性变化）之上，加入目标导向的奖励塑形。

    Parameters
    ----------
    env : Mota
        当前魔塔环境实例。
    base_reward : float
        env.step(return_reward=True) 返回的原始奖励（属性差值 * REWARD_RATE）。
    ending : str
        本步的结局：'continue' / 'death' / 'stop' / 'clear' 等。
    action : Node
        本步执行的动作节点。

    Returns
    -------
    float
        塑形后的奖励值。
    """
    # 延迟导入避免循环依赖
    from environment import Terrain

    reward = float(base_reward)
    action_pos = env.n2p[action]
    current_pos = env.n2p[env.observation[-1]]

    # 1. 终极奖励：成功拿到第五层的剑
    if action_pos == _SWORD_POS:
        reward += 10000.0

    # 2. 楼层进度奖励（使用楼梯时给予正向激励）
    if isinstance(action, Terrain) and action.id in ('upFloor'):
        reward += 100.0

    # 3. 距离引导：当角色已在第五层时，离剑越近奖励越高
    if current_pos[0] == _SWORD_POS[0]:
        dist = abs(current_pos[1] - _SWORD_POS[1]) + abs(current_pos[2] - _SWORD_POS[2])
        reward += max(0.0, 30.0 - dist * 2.0)

    # 4. 每步小惩罚，鼓励找到最短路径
    reward -= 10

    return reward
