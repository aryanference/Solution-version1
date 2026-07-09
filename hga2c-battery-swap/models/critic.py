"""Critic / Value Network A2C-C part called"""
from __future__ import annotations

import torch
import torch.nn as nn


class Critic(nn.Module):

    def __init__(
        self,
        d_model: int = 128,
        n_global_scalars: int = 2,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # soft attention pooling over node embeddings
        self.pool_gate = nn.Linear(d_model, 1)

        self.mlp = nn.Sequential(
            nn.Linear(d_model + n_global_scalars, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_embeddings: torch.Tensor,
        global_scalars: torch.Tensor,
    ) -> torch.Tensor:
        gate_logits = self.pool_gate(node_embeddings).squeeze(-1)
        gate_weights = torch.softmax(gate_logits, dim=-1)
        pooled = torch.bmm(gate_weights.unsqueeze(1), node_embeddings).squeeze(1)
        combined = torch.cat([pooled, global_scalars], dim=-1)
        return self.mlp(combined)
