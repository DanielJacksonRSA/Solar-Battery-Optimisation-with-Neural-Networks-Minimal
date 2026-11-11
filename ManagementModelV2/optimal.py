# Copyright (c) 2026 Daniel Jackson and Nelson Mandela University. All rights reserved.
# This code is proprietary. It may not be used, copied, modified, distributed or
# reused in whole or in part, nor may the ideas it embodies be used, without the
# prior written permission of the copyright holders. See LICENSE.
from __future__ import annotations
from typing import NamedTuple, Sequence
import numpy as np
import environment
from environment import k285
from synthetic_data import k12, k329
k293 = 1001
k280 = 201

class var1(NamedTuple):
    grid: np.ndarray
    k16: float
    k48: int
    k416: list[np.ndarray]
    k226: list[np.ndarray]
    k279: list[np.ndarray]
    k3: np.ndarray

def var2(var13: k12, k232: k285, k186: float, var19: int, k383: bool=True) -> var1:
    var14 = var15 = float(np.sqrt(k186))
    var5 = float(var13.capacity_wh)
    var17 = float(var13.shutoff_pct) / 100.0 * var5
    var24 = float(var13.max_discharge_w)
    var36 = float(var13.soc0_pct) / 100.0 * var5
    var18 = np.linspace(0.0, var5, var19)
    var34 = var18 / var5 * 100.0
    var4 = np.interp(var34, k232.k338, k232.k388)
    var6 = np.clip(np.minimum(var4 * (var5 / k232.k358), var5 - var18), 0.0, None)
    var30 = np.clip(np.minimum(var24, var18 - var17), 0.0, None)
    var43 = []
    var41 = []
    var8 = []
    for var21 in range(k329):
        var33 = var13.loadshedding[var21] > 0.5
        var35 = float(var13.solar_wh[var21])
        var23 = float(var13.load_wh[var21])
        var29 = float(var13.import_price[var21])
        var28 = float(var13.export_price[var21])
        if var33:
            var20 = max(0.0, var23 - var35) / var15
            var11 = np.minimum(var30, var20)
        else:
            var11 = var30
        var32 = np.clip(var18 - var11, 0.0, var5)
        var31 = np.clip(var18 + var6, 0.0, var5)
        var10 = var18[None, :] - var18[:, None]
        var42 = (var18[None, :] >= var32[:, None]) & (var18[None, :] <= var31[:, None])
        var7 = var10 >= 0
        var12 = np.where(var7, var10 / var14, 0.0)
        var9 = np.where(var7, 0.0, -var10 * var15)
        var27 = var35 + var9 - (var23 + var12)
        if var33:
            var16 = np.zeros_like(var27)
            var22 = np.zeros_like(var27)
            var40 = np.maximum(-var27, 0.0)
        else:
            var16 = np.maximum(var27, 0.0)
            var22 = np.maximum(-var27, 0.0)
            var40 = np.zeros_like(var27)
        var43.append(var42)
        var41.append(var40)
        if k383:
            var8.append(var22 / 1000.0 * var29 - var16 / 1000.0 * var28 + var40 / 1000.0 * var29)
    var25 = np.zeros(var19)
    var26 = np.zeros((k329, var19))
    for var21 in range(k329 - 1, -1, -1):
        var39 = np.where(var43[var21], var41[var21] + var25[None, :], np.inf)
        var38 = np.argmin(var39, axis=1)
        var26[var21] = var39[np.arange(var19), var38]
        var25 = var26[var21]
    var37 = int(np.argmin(np.abs(var18 - var36)))
    return var1(grid=var18, k16=var36, k48=var37, k416=var43, k226=var41, k279=var8, k3=var26)

def var3(var50: k12, k232: k285, k186: float, var54: int=k293) -> np.ndarray:
    var59 = var2(var50, k232, k186, var54)
    var53 = var59.grid
    var60 = var59.k16
    var56 = var59.k3
    var46 = float(var50.capacity_wh)
    var51 = float(np.sqrt(k186))
    var47 = float(var50.import_price.min())
    var61 = (var53 - var60) / 1000.0 * var47 * var51
    var65 = -var61
    var63 = max(1e-06, var46 * 1e-09)
    var45 = np.zeros((k329, var54), dtype=np.int64)
    for var55 in range(k329 - 1, -1, -1):
        var57 = var56[var55 + 1] if var55 + 1 < k329 else np.zeros(var54)
        var58 = var59.k226[var55] + var57[None, :] <= var56[var55][:, None] + var63
        var44 = var59.k416[var55] & var58
        var64 = np.where(var44, var59.k279[var55] + var65[None, :], np.inf)
        var45[var55] = np.argmin(var64, axis=1)
        var65 = var64[np.arange(var54), var45[var55]]
    var52 = var59.k48
    var48 = np.zeros(k329)
    var49 = var60
    for var55 in range(k329):
        var62 = var45[var55, var52]
        var48[var55] = var53[var62] - var49
        var49 = var53[var62]
        var52 = var62
    return var48

def k89(var66: k12, k232: k285, k186: float, var67: int=k280) -> float:
    var68 = var2(var66, k232, k186, var67, k383=False)
    return float(var68.k3[0][var68.k48])

def k286(var70: Sequence[k12], k232: k285, k186: float, var71: int=k280) -> np.ndarray:
    return np.array([k89(var69, k232, k186, var71) for var69 in var70], dtype=float)

def k289(var92: environment.k218, k232: k285, k186: float, var84: int=k293) -> environment.k1:
    var80 = var92.count
    var88 = np.zeros((var80, k329))
    for var77 in range(var80):
        var79 = k12(solar_wh=var92.solar_wh[var77], load_wh=var92.load_wh[var77], import_price=var92.import_price[var77], export_price=var92.export_price[var77], loadshedding=var92.loadshedding[var77], soc0_pct=float(var92.soc0_pct[var77]), capacity_wh=float(var92.capacity_wh[var77]), shutoff_pct=float(var92.shutoff_pct[var77]), max_discharge_w=float(var92.max_discharge_w[var77]))
        var88[var77] = var3(var79, k232, k186, var84)
    var81 = float(np.sqrt(k186))

    def action_fn(var82: np.ndarray, var86: int, var78: np.ndarray):
        var73 = var92.capacity_wh[None, :]
        var83 = var92.shutoff_pct[None, :] / 100.0 * var73
        var91 = var78 / var73 * 100.0
        var74 = np.maximum(np.minimum(environment.k434(var91, var73, k232), var73 - var78), 0.0)
        var89 = np.maximum(np.minimum(np.broadcast_to(var92.max_discharge_w[None, :], var78.shape), var78 - var83), 0.0)
        var90 = var92.loadshedding[None, :, var86] > 0.5
        var85 = np.maximum(0.0, var92.load_wh[None, :, var86] - var92.solar_wh[None, :, var86]) / var81
        var76 = np.where(var90, np.minimum(var89, var85), var89)
        var75 = np.broadcast_to(var88[None, :, var86], var78.shape)
        var72 = np.where(var75 >= 0, np.divide(var75, var74, out=np.zeros_like(var75), where=var74 > 1e-09), np.divide(var75, var76, out=np.zeros_like(var75), where=var76 > 1e-09))
        var72 = np.clip(var72, -1.0, 1.0)
        var87 = np.ones(var78.shape, dtype=bool)
        return (var72, var87)
    return action_fn

def k35(var93: k12, k232: k285, k186: float) -> dict:
    var96 = environment.k328([var93], k186)
    var94 = k289(var96, k232, k186)
    var95 = environment.k404(var94, var96, 1, k232, k186)
    return {'actual': var95, 'expected': var95, 'raw': var95, 'k167': np.zeros_like(var95.k298[0, 0], dtype=bool)}
