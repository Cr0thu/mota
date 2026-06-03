from __future__ import annotations

from dataclasses import dataclass

from mota_env.graph_state import GRAPH_NODE_FEATURE_DIM, GRAPH_NODE_TYPES, MAX_GRAPH_NODES


@dataclass(frozen=True)
class GraphPolicyValueConfig:
    max_nodes: int = MAX_GRAPH_NODES
    node_feature_dim: int = GRAPH_NODE_FEATURE_DIM
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 4
    dropout: float = 0.10


def _torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover - exercised only without torch.
        raise RuntimeError("GraphPolicyValueNet requires torch") from exc
    return torch, nn


class GraphPolicyValueNet:
    """Transformer policy/value network over the fixed interaction graph.

    The policy head emits one logit per graph node.  Callers apply the executable
    mask, because non-executable nodes are still useful as context tokens.  The
    value head predicts the stage outcome from pooled node embeddings.
    """

    def __new__(cls, config: GraphPolicyValueConfig | None = None):
        torch, nn = _torch()
        config = config or GraphPolicyValueConfig()

        class _GraphPolicyValueNet(nn.Module):
            def __init__(self, cfg: GraphPolicyValueConfig):
                super().__init__()
                self.config = cfg
                self.type_embedding = nn.Embedding(max(GRAPH_NODE_TYPES.values()) + 1, cfg.d_model)
                self.feature_projection = nn.Sequential(
                    nn.LayerNorm(cfg.node_feature_dim),
                    nn.Linear(cfg.node_feature_dim, cfg.d_model),
                    nn.GELU(),
                    nn.Linear(cfg.d_model, cfg.d_model),
                )
                layer = nn.TransformerEncoderLayer(
                    d_model=cfg.d_model,
                    nhead=cfg.nhead,
                    dim_feedforward=cfg.d_model * 4,
                    dropout=cfg.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
                self.policy_head = nn.Sequential(
                    nn.LayerNorm(cfg.d_model),
                    nn.Linear(cfg.d_model, cfg.d_model),
                    nn.GELU(),
                    nn.Linear(cfg.d_model, 1),
                )
                self.value_head = nn.Sequential(
                    nn.LayerNorm(cfg.d_model),
                    nn.Linear(cfg.d_model, cfg.d_model),
                    nn.GELU(),
                    nn.Linear(cfg.d_model, 1),
                    nn.Tanh(),
                )

            def forward(self, node_features, node_type_ids, node_mask=None):
                if node_features.dim() != 3:
                    raise ValueError("node_features must have shape [batch, nodes, features]")
                if node_type_ids.dim() != 2:
                    raise ValueError("node_type_ids must have shape [batch, nodes]")
                x = self.feature_projection(node_features.float()) + self.type_embedding(node_type_ids.long())
                key_padding_mask = None
                mask_float = None
                if node_mask is not None:
                    key_padding_mask = ~node_mask.bool()
                    mask_float = node_mask.float().unsqueeze(-1)
                encoded = self.encoder(x, src_key_padding_mask=key_padding_mask)
                policy_logits = self.policy_head(encoded).squeeze(-1)
                if mask_float is None:
                    pooled = encoded.mean(dim=1)
                else:
                    pooled = (encoded * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(1.0)
                    policy_logits = policy_logits.masked_fill(~node_mask.bool(), 0.0)
                value = self.value_head(pooled).squeeze(-1)
                return policy_logits, value

        return _GraphPolicyValueNet(config)


def masked_policy(policy_logits, executable_mask, temperature: float = 1.0):
    torch, _nn = _torch()
    if policy_logits.shape != executable_mask.shape:
        raise ValueError("policy_logits and executable_mask must have the same shape")
    mask = executable_mask.bool()
    if (mask.sum(dim=-1) == 0).any():
        raise ValueError("masked_policy received a batch row with no executable action")
    temp = max(float(temperature), 1e-6)
    masked = (policy_logits / temp).masked_fill(~mask, torch.finfo(policy_logits.dtype).min)
    return torch.softmax(masked, dim=-1)


def gather_graph_batch(graph_states: list[dict]):
    torch, _nn = _torch()
    node_features = torch.tensor([g["node_features"] for g in graph_states], dtype=torch.float32)
    node_type_ids = torch.tensor([g["node_type_ids"] for g in graph_states], dtype=torch.long)
    node_mask = torch.tensor([g["node_mask"] for g in graph_states], dtype=torch.bool)
    executable_mask = torch.tensor([g["executable_mask"] for g in graph_states], dtype=torch.bool)
    return {
        "node_features": node_features,
        "node_type_ids": node_type_ids,
        "node_mask": node_mask,
        "executable_mask": executable_mask,
    }
