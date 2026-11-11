# Copyright (c) 2026 Daniel Jackson and Nelson Mandela University. All rights reserved.
# This code is proprietary. It may not be used, copied, modified, distributed or
# reused in whole or in part, nor may the ideas it embodies be used, without the
# prior written permission of the copyright holders. See LICENSE.
from __future__ import annotations
from pathlib import Path
from typing import Sequence, Set
import numpy as np
import pandas as pd
k365 = ['k196', 'k65', 'k392', 'k342', 'k274', 'k202', 'k317']
k7 = ['k362', 'k211', 'k381', 'k190', 'k150', 'k287', 'k345', 'k343', 'k413', 'k178', 'k13']
k253 = ['k337', 'k256', 'k26', 'k260', 'k429']
k398 = ['k61', 'k305', 'k216', 'k86', 'k69']
k288 = ['k155', 'k410', 'k420', 'k249', 'k115', 'k361', 'k396', 'k346', 'k313', 'k366']
k130 = [*k365, *k7, *k253, *k398, *k288]
k257 = ['time', 'load', 'k155', 'k420', 'k249', 'k115', 'k396', 'k361', 'k346', *k365]
k120 = 'load'
k263 = 'k337'
k27 = 2
k443: Set[str] = {'2025-01-01', '2025-03-21', '2025-04-18', '2025-04-21', '2025-04-27', '2025-04-28', '2025-05-01', '2025-06-16', '2025-08-09', '2025-09-24', '2025-12-16', '2025-12-25', '2025-12-26', '2026-01-01', '2026-03-21', '2026-04-03', '2026-04-06', '2026-04-27', '2026-05-01', '2026-06-16', '2026-08-09', '2026-08-10', '2026-09-24', '2026-12-16', '2026-12-25', '2026-12-26'}

def k233(var6: pd.Series, var5: pd.Series, var8: pd.Series) -> pd.Series:
    var4 = var5 / 100.0 * 6.105 * np.exp(17.27 * var6 / (237.7 + var6))
    var7 = var8 / 3.6
    var3 = var6 + 0.33 * var4 - 0.7 * var7 - 4.0
    return var3

def k125(var13: pd.DataFrame) -> pd.DataFrame:
    var12 = var13.copy()
    if not pd.api.types.is_datetime64_any_dtype(var12['time']):
        var20 = pd.to_datetime(var12['time'], format='%Y-%m-%dT%H:%M:%S', errors='coerce')
    else:
        var20 = var12['time']
    var14 = var20.dt.hour + var20.dt.minute / 60.0 + var20.dt.second / 3600.0
    var11 = var20.dt.dayofyear.astype(float)
    var10 = var20.dt.dayofweek.astype(float)
    var15 = var20.dt.month.astype(float)
    var12['k362'] = np.sin(2.0 * np.pi * var14 / 24.0)
    var12['k211'] = np.cos(2.0 * np.pi * var14 / 24.0)
    var12['k381'] = np.sin(2.0 * np.pi * var10 / 7.0)
    var12['k190'] = np.cos(2.0 * np.pi * var10 / 7.0)
    var12['k150'] = (var20.dt.dayofweek >= 5).astype(float)
    var12['k287'] = np.sin(2.0 * np.pi * var11 / 366.0)
    var12['k345'] = np.cos(2.0 * np.pi * var11 / 366.0)
    var12['k343'] = np.sin(2.0 * np.pi * var15 / 12.0)
    var12['k413'] = np.cos(2.0 * np.pi * var15 / 12.0)
    var9 = var20.dt.strftime('%Y-%m-%d')
    var12['k178'] = var9.isin(k443).astype(float)
    var12['k13'] = (var12['k249'] > 0.0).astype(float)
    var19 = var12['k155'].astype(float)
    var16 = var12['k420'].astype(float)
    var21 = var12['k346'].astype(float)
    var12['k410'] = k233(var19, var16, var21)
    var12['k313'] = np.maximum(0.0, var19 - 22.0)
    var12['k366'] = np.maximum(0.0, 18.0 - var19)
    var18 = var12[k120]
    var12['k337'] = var18.shift(1)
    var12['k256'] = var18.shift(2)
    var12['k26'] = var18.shift(3)
    var12['k260'] = var18.shift(24)
    var12['k429'] = var18.shift(48)
    var17 = var18.shift(1)
    var12['k61'] = var17.rolling(window=6, min_periods=1).mean()
    var12['k305'] = var17.rolling(window=24, min_periods=1).mean()
    var12['k216'] = var17.rolling(window=48, min_periods=1).mean()
    var12['k86'] = var17.rolling(window=24, min_periods=1).std().fillna(0.0)
    var12['k69'] = var17.rolling(window=24, min_periods=1).max()
    return var12
var1 = pd.Timedelta(hours=24)

def var2(var25: pd.DataFrame) -> pd.DataFrame:
    var23 = var25.set_index('time')
    var22 = pd.date_range(var23.index.min(), var23.index.max(), freq='h')
    var24 = var23.reindex(var22)
    var24.index.name = 'time'
    var24[k365] = var24[k365].ffill().bfill()
    var26 = ['k155', 'k420', 'k249', 'k115', 'k396', 'k361', 'k346']
    var24[var26] = var24[var26].interpolate(method='linear').ffill().bfill()
    var24[k120] = var24[k120].ffill().bfill()
    return var24.reset_index()

def k354(var33: Sequence[Path]) -> pd.DataFrame:
    var28 = pd.concat((pd.read_csv(var32) for var32 in var33), ignore_index=True)
    var28 = var28.drop(columns=['month'], errors='ignore')
    var31 = [var27 for var27 in k257 if var27 not in var28.columns]
    if var31:
        raise ValueError(f"Fine-tune data files missing columns: {', '.join(var31)}")
    var28['time'] = pd.to_datetime(var28['time'], format='%Y-%m-%dT%H:%M:%S', errors='coerce')
    var28 = var28.dropna(subset=['time']).drop_duplicates(subset=['time']).sort_values('time').reset_index(drop=True)
    var30 = var28['time'].diff() > var1
    var35 = var30.cumsum()
    var29 = [k125(var2(var34)) for (_, var34) in var28.groupby(var35)]
    return pd.concat(var29, ignore_index=True)
