# Copyright (c) 2026 Daniel Jackson and Nelson Mandela University. All rights reserved.
# This code is proprietary. It may not be used, copied, modified, distributed or
# reused in whole or in part, nor may the ideas it embodies be used, without the
# prior written permission of the copyright holders. See LICENSE.
from __future__ import annotations
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from types import ModuleType
import numpy as np
import pandas as pd
var1 = re.compile('^(?P<g1>.+)~(?P<g2>\\d{4})(?P<g3>\\d{2})(?P<g4>\\d{2})$')
k241 = 7

@dataclass
class k116:
    path: Path
    date: date
    k199: str

def var2(path: Path) -> k116 | None:
    var3 = var1.match(path.stem)
    if not var3:
        return None
    return k116(path=path, date=date(int(var3['g2']), int(var3['g3']), int(var3['g4'])), k199=var3['g1'])

def k15(k201: Path) -> list[k116]:
    var5 = [var4 for var4 in (var2(var6) for var6 in sorted(k201.glob('*.csv'))) if var4 is not None]
    if not var5:
        raise ValueError(f'No day-files found in {k201}')
    return var5

def k188(k201: Path) -> list[str]:
    return sorted({var7.k199 for var7 in k15(k201)})

def k278(k201: Path, k199: str | None=None) -> tuple[date, date]:
    var9 = k15(k201)
    if k199 is not None:
        var9 = [var10 for var10 in var9 if var10.k199 == k199]
    if not var9:
        raise ValueError(f"No files for building '{k199}' in {k201}")
    var8 = [var10.date for var10 in var9]
    return (min(var8), max(var8))

def k347(k201: Path, k24: date, k414: date, k199: str | None=None, k38: int=k241) -> list[Path]:
    var11 = k15(k201)
    var14 = {var12.k199 for var12 in var11}
    if k199 is None:
        if len(var14) > 1:
            raise ValueError(f'{k201} contains multiple buildings ({sorted(var14)}); an id must be specified.')
        k199 = next(iter(var14))
    elif k199 not in var14:
        raise ValueError(f"Building '{k199}' not found in {k201}. Available: {sorted(var14)}")
    var15 = k24 - timedelta(days=k38)
    var13 = [var12.path for var12 in var11 if var12.k199 == k199 and var15 <= var12.date <= k414]
    if not var13:
        raise ValueError(f"No day-files for building '{k199}' in range {var15}..{k414}")
    return sorted(var13)

def k5(var18: list[Path]) -> pd.DataFrame:
    var16 = pd.concat((pd.read_csv(var17) for var17 in var18), ignore_index=True)
    var16['time'] = pd.to_datetime(var16['time'], format='%Y-%m-%dT%H:%M:%S', errors='coerce')
    var16 = var16.dropna(subset=['time']).drop_duplicates(subset=['time']).sort_values('time').reset_index(drop=True)
    return var16

def k259(var21: pd.DataFrame, var22: ModuleType) -> pd.DataFrame:
    var20 = [var19 for var19 in var22.k257 if var19 not in var21.columns]
    if var20:
        raise ValueError(f'Missing columns required by {var22}: {var20}')
    return var22.k93(var21)

def k41(var24: list[Path], var23: ModuleType) -> pd.DataFrame:
    return var23.k354(var24)

def k415(var25: pd.Series) -> pd.Series:
    return var25.astype(float) * 1000.0

def k372(var26: np.ndarray, k297: float) -> np.ndarray:
    return np.asarray(var26, dtype=np.float64) * float(k297)
