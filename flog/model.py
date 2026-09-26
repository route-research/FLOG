from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch import nn


class _ResidualAttention(nn.Module):
    """Residual agent/city message-passing block."""

    def __init__(self, hidden_dim: int, heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.agent_self = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.agent_from_city = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.city_self = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.agent_mix = nn.Linear(2 * hidden_dim, hidden_dim)
        self.agent_norm1 = nn.LayerNorm(hidden_dim)
        self.agent_norm2 = nn.LayerNorm(hidden_dim)
        self.city_norm1 = nn.LayerNorm(hidden_dim)
        self.city_norm2 = nn.LayerNorm(hidden_dim)
        self.agent_ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim), nn.GELU(), nn.Linear(ff_dim, hidden_dim)
        )
        self.city_ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim), nn.GELU(), nn.Linear(ff_dim, hidden_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, agent: torch.Tensor, city: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        aa, _ = self.agent_self(agent, agent, agent, need_weights=False)
        ac, _ = self.agent_from_city(agent, city, city, need_weights=False)
        agent_delta = self.agent_mix(torch.cat([aa, ac], dim=-1))
        agent = self.agent_norm1(agent + self.dropout(agent_delta))
        agent = self.agent_norm2(agent + self.dropout(self.agent_ff(agent)))

        cc, _ = self.city_self(city, city, city, need_weights=False)
        city = self.city_norm1(city + self.dropout(cc))
        city = self.city_norm2(city + self.dropout(self.city_ff(city)))
        return agent, city


class FLOGPolicy(nn.Module):
    """Neural policy backbone used by FLOG."""

    def __init__(
        self,
        hidden_dim: int = 128,
        encode_layers: int = 2,
        heads: int = 8,
        ff_dim: int = 256,
        dropout: float = 0.0,
        tanh_clipping: float = 10.0,
    ):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        self.hidden_dim = int(hidden_dim)
        self.encode_layers = int(encode_layers)
        self.heads = int(heads)
        self.ff_dim = int(ff_dim)
        self.dropout_value = float(dropout)
        self.tanh_clipping = float(tanh_clipping)

        self.agent_input = nn.Linear(5, hidden_dim)
        self.city_input = nn.Linear(3, hidden_dim)
        self.layers = nn.ModuleList(
            [
                _ResidualAttention(hidden_dim, heads, ff_dim, dropout)
                for _ in range(encode_layers)
            ]
        )
        self.dynamic_agent = nn.Sequential(
            nn.Linear(6, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.agent_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.city_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        pair_hidden = max(16, hidden_dim // 4)
        self.pair_bias = nn.Sequential(
            nn.Linear(3, pair_hidden), nn.GELU(), nn.Linear(pair_hidden, 1)
        )

    def config(self) -> Dict[str, object]:
        return {
            "hidden_dim": self.hidden_dim,
            "encode_layers": self.encode_layers,
            "heads": self.heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout_value,
            "tanh_clipping": self.tanh_clipping,
        }

    def encode(
        self, agent_feature: torch.Tensor, city_feature: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        agent = self.agent_input(agent_feature)
        city = self.city_input(city_feature)
        for layer in self.layers:
            agent, city = layer(agent, city)
        return agent, city

    def score_pairs(
        self,
        agent_static: torch.Tensor,
        city_static: torch.Tensor,
        dynamic_agent: torch.Tensor,
        pair_feature: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits with shape [rollouts, agents, cities]."""

        agent = agent_static + self.dynamic_agent(dynamic_agent)
        key = self.agent_key(agent)
        query = self.city_query(city_static)
        logits = torch.einsum("rah,rch->rac", key, query) / math.sqrt(self.hidden_dim)
        logits = self.tanh_clipping * torch.tanh(logits)
        return logits + self.pair_bias(pair_feature).squeeze(-1)
