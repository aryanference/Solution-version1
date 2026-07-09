"""Graph Attention Encoder — shared backbone for HGA²C (§5.1)."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """One transformer-style layer with travel-time edge bias."""

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        ff_dim: int = 512,
        dropout: float = 0.1,
        use_travel_time_bias: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.use_travel_time_bias = use_travel_time_bias
        if self.use_travel_time_bias:
            # learned per-head scalar — farther nodes get lower attention
            self.edge_bias_weight = nn.Parameter(torch.ones(n_heads) * 0.1)

        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, n_nodes, _ = x.shape

        residual = x
        x_norm = self.norm1(x)

        Q = self.W_q(x_norm).view(batch_size, n_nodes, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x_norm).view(batch_size, n_nodes, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x_norm).view(batch_size, n_nodes, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if self.use_travel_time_bias:
            edge_bias = -self.edge_bias_weight.view(1, self.n_heads, 1, 1) * \
                        edge_features.unsqueeze(1)
            scores = scores + edge_bias

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), float("-inf"))

        attn_weights = self.dropout(F.softmax(scores, dim=-1))
        attn_output = torch.matmul(attn_weights, V)
        attn_output = self.W_o(
            attn_output.transpose(1, 2).contiguous().view(batch_size, n_nodes, self.d_model)
        )
        x = residual + self.dropout(attn_output)
        x = x + self.ff(self.norm2(x))
        return x


class GATEncoder(nn.Module):
    """Encodes the full hub+region graph into contextual node embeddings."""

    def __init__(
        self,
        node_feature_dim: int = 7,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        ff_dim: int = 512,
        dropout: float = 0.1,
        use_travel_time_bias: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.input_proj = nn.Linear(node_feature_dim, d_model)

        self.layers = nn.ModuleList([
            GraphAttentionLayer(d_model, n_heads, ff_dim, dropout, use_travel_time_bias)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        node_features: torch.Tensor,
        travel_times: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.input_proj(node_features)
        for layer in self.layers:
            x = layer(x, travel_times, mask)
        return self.final_norm(x)

    def get_graph_embedding(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        return node_embeddings.mean(dim=1)
