# -*- coding: utf-8 -*-
import numpy as np
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# ============================================================================
#  PPO Transformer        by Hung1 (PPO extension)
# ----------------------------------------------------------------------------
#  使用 PPO (Proximal Policy Optimization) 來做學習，適用於魔塔環境。
#  與原始 PPO.py 最大的差異：
#    1. Actor-Critic 網路改用 Transformer 統一處理 state / map / action tokens。
#    2. 將 state_encoder、action_encoder、gnn_encoder 的輸出統一投影到 emb_dim，
#       拼接為 [CLS, state, map?, action1, action2, ...] 後送入 Transformer。
#    3. Critic 取 Transformer 最後一層第一個 token (CLS) 經 MLP 頭輸出 value。
#    4. Actor 取 Transformer 最後一層 action tokens 經 MLP 頭輸出 logits，
#       天然支援可變動作數量，且 action token 之間可互相 attention。
#  V 1.0   2026/05
# ============================================================================

# 狀態特徵維度：
# [hp, atk, def, mdef, money, exp, yKey, bKey, rKey, step, floor, y, x]
STATE_DIM = 13

# 動作特徵維度：
# [floor, y, x, isEnemy, e_hp, e_atk, e_def, e_money, e_exp, e_special,
#  isItem, isTerrain, isNPC, isEndFlag, bias, special_flag]
ACTION_FEATURE_DIM = 16

# PPO 专用奖励系数
# 对应 [hp, atk, def, mdef, money, exp, yKey, bKey, rKey]
PPO_REWARD_RATE = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)


# ============================================================================
#  Actor-Critic Transformer 神經網路
# ----------------------------------------------------------------------------
#  Token 順序: [CLS, state, map(optional), action1, action2, ...]
#  - CLS token   : 可學習參數，最後一層輸出經 critic_head -> value
#  - state token : 經 state_encoder 編碼
#  - map token   : 經 gnn_encoder + gnn_proj 編碼（可選）
#  - action tokens: 經 action_encoder 編碼，最後一層輸出經 actor_head -> logits
# ============================================================================
class ActorCriticTransformer(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM,
                 action_feature_dim: int = ACTION_FEATURE_DIM,
                 emb_dim: int = 64,
                 emb_map_dim: int = 64,
                 use_gnn: bool = True,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.emb_dim = emb_dim
        self.use_gnn = use_gnn

        # State Encoder: state -> emb_dim
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, emb_dim),
            nn.Tanh(),
        )

        # Action Encoder: action_feature -> emb_dim
        self.action_encoder = nn.Sequential(
            nn.Linear(action_feature_dim, 64),
            nn.Tanh(),
            nn.Linear(64, emb_dim),
            nn.Tanh(),
        )

        # GNN projection: gnn output -> emb_dim
        if use_gnn and emb_map_dim > 0:
            self.gnn_proj = nn.Sequential(
                nn.Linear(emb_map_dim, emb_dim),
                nn.Tanh(),
            )
        else:
            self.gnn_proj = None

        # CLS token for value estimation
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))

        # Positional embedding (up to 1024 tokens, more than enough for this game)
        self.pos_embedding = nn.Embedding(1024, emb_dim)

        # Transformer encoder (Pre-norm for stability)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Actor head: action token -> logit
        self.actor_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Tanh(),
            nn.Linear(emb_dim, 1),
        )

        # Critic head: cls token -> value
        self.critic_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Tanh(),
            nn.Linear(emb_dim, 1),
        )

    def _build_sequence(self, state_emb: torch.Tensor,
                        action_embs: torch.Tensor,
                        map_emb: torch.Tensor = None):
        """
        將單一樣本的編碼後 embedding 拼接為 token sequence。
        state_emb : (emb_dim,)
        action_embs: (n_actions, emb_dim)
        map_emb   : (emb_dim,) or None
        return    : (seq_len, emb_dim)
        """
        tokens = [self.cls_token.squeeze(0), state_emb.unsqueeze(0)]
        if self.gnn_proj is not None and map_emb is not None:
            tokens.append(map_emb.unsqueeze(0))
        tokens.append(action_embs)
        seq = torch.cat(tokens, dim=0)  # (seq_len, emb_dim)

        positions = torch.arange(seq.shape[0], device=seq.device)
        seq = seq + self.pos_embedding(positions)
        return seq

    def forward(self, state: torch.Tensor, action_features: torch.Tensor,
                map_emb: torch.Tensor = None):
        """
        single-state 推理（用於 choose_action / greedy_action）。
        state: (state_dim,)
        action_features: (n_actions, action_feature_dim)
        map_emb: (emb_map_dim,) or None
        return: (logits (n_actions,), value (scalar))
        """
        state_emb = self.state_encoder(state)
        action_emb = self.action_encoder(action_features)

        map_emb_enc = None
        if self.gnn_proj is not None and map_emb is not None:
            map_emb_enc = self.gnn_proj(map_emb)

        seq = self._build_sequence(state_emb, action_emb, map_emb_enc)
        out = self.transformer(seq.unsqueeze(0)).squeeze(0)  # (seq_len, emb_dim)

        # Critic: first token (CLS)
        value = self.critic_head(out[0]).squeeze(-1)

        # Actor: action tokens
        has_map = self.gnn_proj is not None and map_emb is not None
        offset = 3 if has_map else 2
        n_actions = action_features.shape[0]
        action_out = out[offset:offset + n_actions]
        logits = self.actor_head(action_out).squeeze(-1)

        return logits, value

    def forward_batch(self, states: torch.Tensor,
                      action_features_list: List[torch.Tensor],
                      map_embs: torch.Tensor = None):
        """
        Batched forward for training，支援每個樣本動作數量不同。
        states: (B, state_dim)
        action_features_list: list of B tensors, each (N_i, action_feature_dim)
        map_embs: (B, emb_map_dim) or None
        return: (logits_list [B tensors of shape (N_i,)], values (B,))
        """
        B = states.shape[0]
        device = states.device

        # Encode
        state_embs = self.state_encoder(states)  # (B, emb_dim)
        action_embs_list = [self.action_encoder(f) for f in action_features_list]

        has_map = self.gnn_proj is not None and map_embs is not None
        if has_map:
            map_embs_enc = self.gnn_proj(map_embs)  # (B, emb_dim)
            offset = 3
        else:
            map_embs_enc = None
            offset = 2

        # Build padded sequences
        seq_lens = [offset + a.shape[0] for a in action_embs_list]
        max_len = max(seq_lens)

        padded_seq = torch.zeros(B, max_len, self.emb_dim, device=device)
        padding_mask = torch.ones(B, max_len, dtype=torch.bool, device=device)

        for i in range(B):
            padded_seq[i, 0] = self.cls_token.squeeze(0)
            padded_seq[i, 1] = state_embs[i]
            if map_embs_enc is not None:
                padded_seq[i, 2] = map_embs_enc[i]
            n_a = action_embs_list[i].shape[0]
            padded_seq[i, offset:offset + n_a] = action_embs_list[i]
            padding_mask[i, :seq_lens[i]] = False

        # Positional encoding
        positions = torch.arange(max_len, device=device)
        padded_seq = padded_seq + self.pos_embedding(positions).unsqueeze(0)

        # Transformer
        out = self.transformer(padded_seq, src_key_padding_mask=padding_mask)
        # out: (B, max_len, emb_dim)

        # Critic: first token
        values = self.critic_head(out[:, 0]).squeeze(-1)  # (B,)

        # Actor: gather action token outputs for each sample
        logits_list = []
        for i in range(B):
            n_a = action_embs_list[i].shape[0]
            action_out = out[i, offset:offset + n_a]  # (N_i, emb_dim)
            logits = self.actor_head(action_out).squeeze(-1)  # (N_i,)
            logits_list.append(logits)

        return logits_list, values


# ============================================================================
#  PPO Transformer 主類別
# ============================================================================
class PPOTransformer:
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
                 emb_dim: int = 64,
                 emb_map_dim: int = 64,
                 emb_state_dim: int = 32,
                 emb_action_dim: int = 32,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 gnn_encoder=None,
                 device: str = None):
        self.gamma = discount_factor
        self.clip_eps = clip_epsilon
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.reward_scale = reward_scale
        self.emb_map_dim = emb_map_dim
        self.gnn_encoder = gnn_encoder

        # 自動偵測 GPU
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            print(f'[PPOTransformer] Using GPU: {torch.cuda.get_device_name(self.device)}')
        else:
            print('[PPOTransformer] Using CPU')

        self.network = ActorCriticTransformer(
            state_dim=STATE_DIM,
            action_feature_dim=ACTION_FEATURE_DIM,
            emb_dim=emb_dim,
            emb_map_dim=emb_map_dim,
            use_gnn=(gnn_encoder is not None),
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        ).to(self.device)

        # 优化器参数包含 Actor-Critic 和 GNN（如果启用）
        params = list(self.network.parameters())
        if self.gnn_encoder is not None:
            params += list(self.gnn_encoder.gnn.parameters())
        self.optimizer = optim.Adam(params, lr=learning_rate)

        # 軌跡緩衝區
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
        next_value = 0.0
        for t in reversed(range(n)):
            tr = self.trajectory[t]
            mask = 0.0 if tr['done'] else 1.0
            delta = tr['reward'] + self.gamma * next_value * mask - tr['value']
            last_adv = delta + self.gamma * self.gae_lambda * mask * last_adv
            advantages[t] = last_adv
            next_value = tr['value']
            if tr['done']:
                next_value = 0.0
                last_adv = 0.0
        values = np.array([tr['value'] for tr in self.trajectory], dtype=np.float32)
        returns = advantages + values
        return advantages, returns

    # ------------------------------------------------------------------------
    #  PPO 更新（使用 Transformer batch forward）
    # ------------------------------------------------------------------------
    def update(self):
        if len(self.trajectory) < 2:
            self.trajectory.clear()
            return

        advantages, returns = self._compute_gae()
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = len(self.trajectory)

        # 預先提取固定長度數據
        states_np = np.stack([tr['state'] for tr in self.trajectory])
        idxs_np = np.array([tr['action_idx'] for tr in self.trajectory], dtype=np.int64)
        old_lp_np = np.array([tr['log_prob'] for tr in self.trajectory], dtype=np.float32)
        feats_list = [tr['action_features'] for tr in self.trajectory]

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

                # 準備 map embeddings（帶梯度，端到端訓練）
                b_map_embs = None
                if has_gnn:
                    map_embs = []
                    for nf, ei in b_graphs:
                        nf_t = torch.from_numpy(nf).to(self.device, non_blocking=True)
                        ei_t = torch.from_numpy(ei).to(self.device, non_blocking=True)
                        map_emb = self.gnn_encoder.gnn(nf_t, ei_t)
                        map_embs.append(map_emb)
                    b_map_embs = torch.stack(map_embs)  # (batch, emb_map_dim)

                # 將 numpy action features 轉為 tensor list
                b_feats_t = [torch.from_numpy(f).to(self.device, non_blocking=True) for f in b_feats]

                # Transformer batch forward
                logits_list, values = self.network.forward_batch(
                    b_states, b_feats_t, b_map_embs
                )

                # 計算新 log_prob 與 entropy
                new_lps = []
                entropies = []
                for j, logits in enumerate(logits_list):
                    dist = Categorical(logits=logits)
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
            # 兼容旧格式
            self.network.load_state_dict(checkpoint)


# ============================================================================
#  PPO 奖励塑形 —— 目标：取得 10 层魔塔第五层的剑
# ============================================================================
_SWORD_POS = (4, 11, 11)


def compute_sword_reward(env, base_reward: float, ending: str, action) -> float:
    """
    在环境默认奖励（属性变化）之上，加入目标导向的奖励塑形。
    """
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
