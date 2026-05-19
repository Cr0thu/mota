# -*- coding: utf-8 -*-
"""
MapGNN: 将魔塔10层地图建模为图，通过GNN输出地图级别的embedding向量。

节点初始编码沿用 PPO.py 中 ACTION_FEATURE_DIM=16 的相同方式，并额外加入
节点状态位（activated / disabled / 玩家位置标记 / 当前楼层标记）。
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple

# ============================================================================
#  常量定义（与 PPO.py 保持一致）
# ============================================================================
ACTION_FEATURE_DIM = 16

# 节点特征维度 = ACTION_FEATURE_DIM + 状态扩展
# 扩展维度: [activated, disabled, is_player_pos, is_current_floor]
NODE_FEATURE_DIM = ACTION_FEATURE_DIM + 4  # = 20


# ============================================================================
#  图卷积层（纯 PyTorch 实现，无需 PyG 依赖）
# ============================================================================
class GraphConvLayer(nn.Module):
    """简单图卷积 + 残差：聚合邻居消息，与自身特征融合后线性变换。"""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x          : (N, in_dim)
        edge_index : (2, E)  有向边，src -> dst
        return     : (N, out_dim)
        """
        src, dst = edge_index
        # 1. 收集源节点消息
        msg = x[src]                     # (E, in_dim)
        # 2. 按目标节点求和聚合
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, msg)      # (N, in_dim)
        # 3. 残差融合 + 线性变换
        out = self.linear(agg + x)
        return out


# ============================================================================
#  GNN 主网络
# ============================================================================
class MapGNN(nn.Module):
    """
    输入：魔塔当前全部10层地图的图结构（节点特征 + 边）
    输出：整个地图的图级别 embedding 向量 (output_dim,)
    """

    def __init__(self,
                 node_feature_dim: int = NODE_FEATURE_DIM,
                 hidden_dim: int = 64,
                 output_dim: int = 64,
                 num_layers: int = 3,
                 dropout: float = 0.0):
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # 节点初始嵌入
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 消息传递层
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GraphConvLayer(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 图级别读出 → 输出 embedding
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self,
                node_features: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """
        node_features : (N, node_feature_dim)  所有节点的初始特征
        edge_index    : (2, E)                 无向边（调用前需拆成双向有向）
        return        : (output_dim,)          整个地图的 embedding 向量
        """
        # 1. 初始嵌入
        x = self.node_encoder(node_features)          # (N, hidden_dim)
        x = torch.relu(x)

        # 2. 消息传递
        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, edge_index)
            h = norm(h)
            h = torch.relu(h)
            h = self.dropout(h)
            x = x + h                                     # 残差连接

        # 3. 全局读出：mean pooling 得到图级别表示
        graph_emb = x.mean(dim=0)                     # (hidden_dim,)

        # 4. 输出投影
        out = self.readout(graph_emb)                 # (output_dim,)
        return out


# ============================================================================
#  地图图编码器（负责把魔塔环境转换为图数据）
# ============================================================================
class MapGraphEncoder:
    """
    将 Mota 环境当前状态构建为图：
      - 节点：所有事件节点（Enemy / Item / Terrain / NPC / EndFlag），不包括普通地板
      - 边   ：基于节点已有的 links（可通行路径）构建无向边
      - 节点特征：沿用 ACTION_FEATURE_DIM 编码，扩展节点状态位
    """

    @staticmethod
    def build_graph(env) -> Tuple[np.ndarray, np.ndarray]:
        """
        将环境编码为图数据。

        Parameters
        ----------
        env : Mota
            当前魔塔环境实例。

        Returns
        -------
        node_features : np.ndarray, shape (N, NODE_FEATURE_DIM)
        edge_index    : np.ndarray, shape (2, E)
        """
        from environment import Enemy, Item, Terrain, NPC, EndFlag

        floors = env.env_data['floors']
        max_l = max(1, floors['layer'])
        max_h = max(1, floors['height'])
        max_w = max(1, floors['width'])

        current_pos = env.n2p[env.observation[-1]]
        current_floor = current_pos[0]

        # ---- 收集节点：排除玩家自身 + 排除已触发(activated)的节点 ----
        nodes = []          # [(node_obj, pos), ...]
        node_to_idx = {}    # node_obj -> index
        for node, pos in env.n2p.items():
            if node is env.player:
                continue
            if node.activated:
                continue
            node_to_idx[node] = len(nodes)
            nodes.append((node, pos))

        N = len(nodes)
        if N == 0:
            return (np.zeros((0, NODE_FEATURE_DIM), dtype=np.float32),
                    np.zeros((2, 0), dtype=np.int64))

        # ---- 节点特征编码 ----
        feats = np.zeros((N, NODE_FEATURE_DIM), dtype=np.float32)

        for i, (node, pos) in enumerate(nodes):
            # --- 位置（与 ACTION_FEATURE_DIM 一致）---
            feats[i, 0] = pos[0] / max_l   # floor
            feats[i, 1] = pos[1] / max_h   # y
            feats[i, 2] = pos[2] / max_w   # x

            # --- 节点类型与属性（与 ACTION_FEATURE_DIM 一致）---
            if isinstance(node, Enemy):
                feats[i, 3] = 1.0
                feats[i, 4] = node.hp / 1000.0
                feats[i, 5] = node.atk / 200.0
                feats[i, 6] = node.def_ / 200.0
                feats[i, 7] = node.money / 100.0
                feats[i, 8] = node.exp / 100.0
                feats[i, 9] = node.special / 25.0
            elif isinstance(node, Item):
                feats[i, 10] = 1.0
            elif isinstance(node, Terrain):
                feats[i, 11] = 1.0
                if node.id in ('yellowDoor', 'blueDoor', 'redDoor', 'greenDoor'):
                    feats[i, 15] = 1.0
            elif isinstance(node, NPC):
                feats[i, 12] = 1.0
            elif isinstance(node, EndFlag):
                feats[i, 13] = 1.0

            feats[i, 14] = 1.0   # bias

            # --- 扩展状态位 ---
            feats[i, 16] = 1.0 if node.activated else 0.0          # 已触发
            feats[i, 17] = 1.0 if node.disabled else 0.0           # 已禁用
            feats[i, 18] = 1.0 if pos == current_pos else 0.0      # 玩家当前位置
            feats[i, 19] = 1.0 if pos[0] == current_floor else 0.0 # 是否在当前楼层

        # ---- 构建边（基于 links，转无向双向边） ----
        edge_set = set()
        for node, _ in nodes:
            i = node_to_idx[node]
            for linked in node.links:
                if linked is env.player or linked not in node_to_idx:
                    continue
                j = node_to_idx[linked]
                # 无向边去重存储
                if i <= j:
                    edge_set.add((i, j))
                else:
                    edge_set.add((j, i))

        if edge_set:
            edges = np.array(list(edge_set), dtype=np.int64)   # (E, 2)
            # 拆分为双向有向边（满足 GraphConvLayer 的 src->dst 语义）
            edge_index = np.concatenate([
                edges.T,           # (2, E) 正向
                edges[:, [1, 0]].T # (2, E) 反向
            ], axis=1)             # (2, 2E)
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)

        return feats, edge_index


# ============================================================================
#  封装接口：环境 -> GNN Embedding
# ============================================================================
class MapGNNEncoder:
    """
    封装了图编码 + GNN 前向的完整流程。
    在玩家每步选择动作后调用 get_embedding(env) 即可获得当前地图的 embedding。
    """

    def __init__(self,
                 output_dim: int = 64,
                 hidden_dim: int = 64,
                 num_layers: int = 3,
                 dropout: float = 0.0,
                 device: str = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.output_dim = output_dim

        self.gnn = MapGNN(
            node_feature_dim=NODE_FEATURE_DIM,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)

    def get_embedding(self, env) -> np.ndarray:
        """
        主入口。输入当前环境，返回地图的 GNN embedding 向量。

        Parameters
        ----------
        env : Mota

        Returns
        -------
        embedding : np.ndarray, shape (output_dim,)
        """
        node_feats, edge_index = MapGraphEncoder.build_graph(env)

        if node_feats.shape[0] == 0:
            return np.zeros(self.output_dim, dtype=np.float32)

        x = torch.from_numpy(node_feats).to(self.device, non_blocking=True)
        ei = torch.from_numpy(edge_index).to(self.device, non_blocking=True)

        with torch.no_grad():
            emb = self.gnn(x, ei)

        return emb.cpu().numpy()

    def get_embedding_train(self, env) -> torch.Tensor:
        """
        训练模式入口，返回 torch.Tensor（保留梯度）。
        """
        node_feats, edge_index = MapGraphEncoder.build_graph(env)

        if node_feats.shape[0] == 0:
            return torch.zeros(self.output_dim, dtype=torch.float32, device=self.device)

        x = torch.from_numpy(node_feats).to(self.device, non_blocking=True)
        ei = torch.from_numpy(edge_index).to(self.device, non_blocking=True)
        return self.gnn(x, ei)

    @staticmethod
    def get_graph_data(env) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取当前地图的图数据（节点特征 + 边索引），用于在 update 阶段重新 forward GNN。
        return: (node_features, edge_index)
        """
        return MapGraphEncoder.build_graph(env)

    def save(self, path: str):
        torch.save(self.gnn.state_dict(), path)

    def load(self, path: str):
        self.gnn.load_state_dict(torch.load(path, map_location=self.device))


# ============================================================================
#  与 PPO 集成的辅助函数示例
# ============================================================================
def make_state_with_gnn(env, gnn_encoder: MapGNNEncoder, ppo_state: np.ndarray = None) -> np.ndarray:
    """
    将 PPO 原始 state 与 GNN 地图 embedding 拼接，得到增强后的状态向量。

    Parameters
    ----------
    env          : Mota
    gnn_encoder  : MapGNNEncoder
    ppo_state    : np.ndarray, shape (STATE_DIM,) 或 None

    Returns
    -------
    enhanced_state : np.ndarray, shape (STATE_DIM + output_dim,)
    """
    gnn_emb = gnn_encoder.get_embedding(env)
    if ppo_state is not None:
        return np.concatenate([ppo_state, gnn_emb], axis=0)
    return gnn_emb
