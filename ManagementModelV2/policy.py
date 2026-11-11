# Copyright (c) 2026 Daniel Jackson and Nelson Mandela University. All rights reserved.
# This code is proprietary. It may not be used, copied, modified, distributed or
# reused in whole or in part, nor may the ideas it embodies be used, without the
# prior written permission of the copyright holders. See LICENSE.
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
k130: List[str] = ['k308', 'k75', 'k234', 'k349', 'k91', 'k315', 'k95', 'k307', 'k154', 'k87', 'k31', 'k432', 'k60', 'k359', 'k52', 'k133', 'k98', 'k362', 'k211']
k371 = len(k130)
k137 = 12
k380 = 2
k251: List[Tuple[str, Tuple[int, ...]]] = [('w1', (k371, k137)), ('b1', (k137,)), ('w2', (k137, k380)), ('b2', (k380,))]
k316 = sum((int(np.prod(shape)) for (_, shape) in k251))

def unflatten(var2: np.ndarray) -> Dict[str, np.ndarray]:
    var2 = np.asarray(var2, dtype=np.float64)
    var3 = var2.shape[:-1]
    var6: Dict[str, np.ndarray] = {}
    var5 = 0
    for (var4, shape) in k251:
        var1 = int(np.prod(shape))
        var6[var4] = var2[..., var5:var5 + var1].reshape(*var3, *shape)
        var5 += var1
    return var6

def flatten(var8: Dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(var8[var7], dtype=np.float64).reshape(-1) for (var7, _) in k251])

def k326(var14: int, var15: np.random.Generator) -> np.ndarray:
    var13 = np.empty((var14, k316), dtype=np.float64)
    var12 = 0
    for (var9, shape) in k251:
        var10 = int(np.prod(shape))
        var11 = shape[0] if len(shape) > 1 else 1
        var16 = 1.0 / np.sqrt(var11)
        var13[:, var12:var12 + var10] = var15.normal(0.0, var16, size=(var14, var10))
        var12 += var10
    return var13

def forward(var20: Dict[str, np.ndarray], var17: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    var18 = np.tanh(np.einsum('pef,pfh->peh', var17, var20['w1']) + var20['b1'][:, None, :])
    var19 = np.einsum('peh,pho->peo', var18, var20['w2']) + var20['b2'][:, None, :]
    return (np.tanh(var19[..., 0]), var19[..., 1])
