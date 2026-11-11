# Copyright (c) 2026 Daniel Jackson and Nelson Mandela University. All rights reserved.
# This code is proprietary. It may not be used, copied, modified, distributed or
# reused in whole or in part, nor may the ideas it embodies be used, without the
# prior written permission of the copyright holders. See LICENSE.
from __future__ import annotations
from typing import List, Sequence
import torch
from torch import nn

class k266(nn.Module):

    def __init__(self, k81: int, k103: Sequence[int]=(128, 64), var1: float=0.1) -> None:
        super().__init__()
        var3: List[nn.Module] = []
        var4 = k81
        for var2 in k103:
            var3.append(nn.Linear(var4, var2))
            var3.append(nn.ReLU())
            var3.append(nn.Dropout(p=var1))
            var4 = var2
        var3.append(nn.Linear(var4, 1))
        self.network = nn.Sequential(*var3)

    def forward(self, var5: torch.Tensor) -> torch.Tensor:
        return self.network(var5).squeeze(-1)

def k395(var6: dict, device: str) -> k266:
    var7 = k266(k81=int(var6['k81']), k103=list(var6['k103'])).to(device)
    var7.load_state_dict(var6['k204'])
    return var7
