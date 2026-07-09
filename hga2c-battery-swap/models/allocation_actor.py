"""Level-1 Allocation Actor — swap + relocation decisions """
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class AllocationActor(nn.Module):

    def __init__(
        self,
        d_model: int = 128,
        max_swap: int = 7,
        max_reloc: int = 5,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_swap = max_swap
        self.max_reloc = max_reloc

        # swap head: region embedding → logits over {0..max_swap}
        self.swap_head = nn.Linear(d_model, max_swap + 1)

        # relocation head: [e_r || e_l] → logits over {0..max_reloc}
        self.reloc_head = nn.Linear(2 * d_model, max_reloc + 1)

    def forward_swap(
        self,
        region_embedding: torch.Tensor,
        swap_mask: torch.Tensor,
        greedy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.swap_head(region_embedding)
        logits = logits.masked_fill(~swap_mask, float("-inf"))
        dist = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if greedy else dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def forward_relocation(
        self,
        src_embedding: torch.Tensor,
        dst_embedding: torch.Tensor,
        reloc_mask: torch.Tensor,
        greedy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pair_emb = torch.cat([src_embedding, dst_embedding], dim=-1)
        logits = self.reloc_head(pair_emb)
        logits = logits.masked_fill(~reloc_mask, float("-inf"))
        dist = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if greedy else dist.sample()
        return action, dist.log_prob(action), dist.entropy()
