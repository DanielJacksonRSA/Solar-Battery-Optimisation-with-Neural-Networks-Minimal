# Copyright (c) 2026 Daniel Jackson and Nelson Mandela University. All rights reserved.
# This code is proprietary. It may not be used, copied, modified, distributed or
# reused in whole or in part, nor may the ideas it embodies be used, without the
# prior written permission of the copyright holders. See LICENSE.
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from typing import Literal
import numpy as np
import pandas as pd
import registry
k329 = 24
k271 = 0.15
k272 = frozenset({6, 7, 8})
k90 = frozenset({7, 8, 9, 18, 19})
k391 = frozenset({6, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21})
k322 = {('winter', 'peak'): 856.79, ('winter', 'standard'): 294.05, ('winter', 'off_peak'): 192.61, ('summer', 'peak'): 331.17, ('summer', 'standard'): 236.54, ('summer', 'off_peak'): 176.41}
k9 = {('winter', 'peak'): 529.95, ('winter', 'standard'): 145.27, ('winter', 'off_peak'): 102.53, ('summer', 'peak'): 229.91, ('summer', 'standard'): 136.72, ('summer', 'off_peak'): 102.53}
k71 = ('peak', 'standard', 'off_peak')
k322.update({('blended', band): (k322['winter', band] + k322['summer', band]) / 2.0 for band in k71})
k9.update({('blended', band): (k9['winter', band] + k9['summer', band]) / 2.0 for band in k71})
k70 = 117.7
k295 = 'k299'
k399 = 'synthetic'

@dataclass
class k357:
    schedule: Literal['k299', 'synthetic'] = k295
    k247: bool = True
    k451: bool = True
    k250: bool = True
    k264: float = 0.2
    k446: float = 2.0
    k169: float = 0.6
    k194: float = 0.6
    k229: int | None = None
    seed: int = 42

def k406(var1: date) -> str:
    return 'winter' if var1.month in k272 else 'summer'

def k140(var2: date, k375: 'k357') -> str:
    return 'blended' if k375.k451 else k406(var2)

def k339(var3: date, var4: int, k250: bool=True) -> str:
    if k250 and var3.weekday() >= 5:
        return 'off_peak'
    if var4 in k90:
        return 'peak'
    if var4 in k391:
        return 'standard'
    return 'off_peak'

def k32(var5: float, k247: bool) -> float:
    var6 = var5 / 100.0
    return var6 * (1.0 + k271) if k247 else var6

def k114(var9: date, k375: k357) -> tuple[np.ndarray, np.ndarray, list[str]]:
    var13 = k140(var9, k375)
    var8 = [k339(var9, var11, k375.k250) for var11 in range(k329)]
    var12 = np.array([k32(k322[var13, var7], k375.k247) for var7 in var8])
    var10 = np.array([k32(k9[var13, var7], k375.k247) for var7 in var8])
    return (var12, var10, var8)

def k39(days: list[date], k375: k357) -> float:
    if k375.schedule != k295:
        return 0.0
    var14 = k70 * (1.0 + k271 if k375.k247 else 1.0)
    return var14 * len(days) / 30.4375

def k405(var16: list[date], k375: k357) -> pd.DataFrame:
    var33 = registry.k408()
    var31 = []
    for (var18, var17) in enumerate(var16):
        var30 = np.random.default_rng(k375.seed + var18)
        (var27, var26, var23, var32) = var33.k223(var30)
        if k375.k229 is not None:
            var20 = int(k375.k229)
            if var20 >= 4:
                var23 = var33.k108(var30, var33.k314, var33.k321, var33.k252)
                var32 = 4
            else:
                var23 = var33.k108(var30, var20, var33.k174, var33.k444)
                var32 = var20
        if k375.schedule == k295:
            (var22, var19, var15) = k114(var17, k375)
        else:
            (var29, var28) = (var27.min(), var27.max())
            var25 = (var27 - var29) / (var28 - var29) if var28 > var29 else np.zeros(k329)
            var24 = k375.k169 + var25 * (k375.k446 - k375.k169)
            var22 = k375.k264 * var24
            var19 = np.full(k329, k375.k264 * k375.k194)
            var15 = ['synthetic'] * k329
        for var21 in range(k329):
            var31.append({'date': var17, 'hour': var21, 'import_price': float(var22[var21]), 'export_price': float(var19[var21]), 'loadshedding': float(var23[var21]), 'k229': var32, 'k225': var15[var21], 'season': k406(var17), 'k140': k140(var17, k375) if k375.schedule == k295 else 'synthetic'})
    return pd.DataFrame(var31)

def k227(var36: pd.DataFrame) -> dict:
    var35 = var36.loc[var36['loadshedding'] > 0.5].groupby('k229').size().to_dict()
    var34 = var36.loc[var36['loadshedding'] > 0.5].drop_duplicates(subset=['date', 'k229']).groupby('k229').size().to_dict()
    return {'k170': var35, 'k389': var34}

def k182(var40: pd.DataFrame) -> dict:
    if 'k225' not in var40.columns:
        return {}
    var38 = var40.groupby(['season', 'k225'])
    return {f'{var39}_{band}': {'hours': int(len(var37)), 'import_price': float(var37['import_price'].iloc[0]), 'export_price': float(var37['export_price'].iloc[0])} for ((var39, band), var37) in var38}
