# Copyright (c) 2026 Daniel Jackson and Nelson Mandela University. All rights reserved.
# This code is proprietary. It may not be used, copied, modified, distributed or
# reused in whole or in part, nor may the ideas it embodies be used, without the
# prior written permission of the copyright holders. See LICENSE.
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import environment
import policy as policy_module
from environment import k285, k218
from synthetic_data import k12
k290 = 5.0
k207 = 0.02
k283 = 0.2

@dataclass
class k157:
    k246: np.ndarray
    k232: k285
    k186: float
    k375: dict

def k10(k213, device: str | None=None) -> k157:
    var4 = json.loads(Path(k213).read_text(encoding='utf-8'))
    var1 = var4['k375']
    var3 = var1['k121']
    var2 = k285(k338=np.asarray(var3['k338'], dtype=np.float64), k388=np.asarray(var3['k388'], dtype=np.float64), k358=float(var3['k358']))
    return k157(k246=np.asarray(var4['k246'], dtype=np.float64), k232=var2, k186=float(var1['k186']), k375=var1)

def k58(var6: environment.k1, var29: k218, k124: float=k290) -> environment.k1:

    def action_fn(var13: np.ndarray, var17: int, var11: np.ndarray):
        (var5, var22) = var6(var13, var17, var11)
        var7 = var29.capacity_wh[None, :]
        var14 = var29.shutoff_pct[None, :] / 100.0 * var7
        var30 = np.maximum(0.0, var11 - var14)
        var26 = var29.k64[None, :, var17]
        var25 = np.where(var26 > 0.0, var26 + k124 / 100.0 * var7, 0.0)
        var27 = var29.loadshedding[None, :, var17] > 0.5
        var12 = var29.k17[None, :, var17]
        var10 = var25 - var30 - var12
        var28 = var10 > 0.0
        var18 = np.maximum(var29.k133[None, :, var17], 0.0)
        var16 = np.maximum(var7 - var11, 0.0)
        var23 = var29.import_price[None, :, var17]
        var9 = var29.k261[None, :, var17]
        var19 = var23 > var9 * (1.0 + k207)
        var8 = np.minimum(k283 * var7, var16)
        var24 = var10 <= var8 * np.maximum(var18 - 1.0, 0.0)
        var20 = var28 & ~var27 & ~(var19 & var24)
        var21 = var10 / np.maximum(var18, 1.0)
        var15 = np.clip(var21 / np.maximum(var16, 1.0), 0.0, 1.0)
        var5 = np.where(var20, np.maximum(var5, var15), var5)
        var22 = var22 & ~var28
        return (var5, var22)
    return action_fn

def k94(var40: k157, var32: k12, var36: k12 | None=None, k340: bool=True, k124: float=k290) -> dict:
    var45 = var40.k186
    var33 = environment.k328([var32], var45)
    var37 = environment.k328([var36], var45) if var36 is not None else None
    var38 = var37 or var33
    var42 = policy_module.unflatten(np.atleast_2d(var40.k246))
    var34 = environment.k55(var42)
    var44 = var34
    var39 = k58(var34, var38, k124) if k340 else var34
    var43 = environment.k404(var44, var33, 1, var40.k232, var45, var37)
    var31 = environment.k404(var39, var33, 1, var40.k232, var45, var37)
    var35 = environment.k404(var39, var38, 1, var40.k232, var45, None)
    var41 = ~np.isclose(var43.k298[0, 0], var31.k298[0, 0])
    return {'actual': var31, 'expected': var35, 'raw': var43, 'k167': var41}
