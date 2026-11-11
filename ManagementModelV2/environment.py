# Copyright (c) 2026 Daniel Jackson and Nelson Mandela University. All rights reserved.
# This code is proprietary. It may not be used, copied, modified, distributed or
# reused in whole or in part, nor may the ideas it embodies be used, without the
# prior written permission of the copyright holders. See LICENSE.
from __future__ import annotations
import csv
from dataclasses import dataclass
from typing import Callable, Sequence, Tuple
import numpy as np
import policy as policy_module
from synthetic_data import k192, k195, k329, k12
k11 = 10000.0
k440 = 0.92
k255 = 40.0
k4 = 100.0

@dataclass
class k285:
    k338: np.ndarray
    k388: np.ndarray
    k358: float = k11

def k172(path, k358: float=k11) -> k285:
    var7: list[float] = []
    var5: list[float] = []
    with open(path, newline='', encoding='utf-8') as var3:
        for var6 in csv.DictReader(var3):
            var7.append(float(var6['SOC']))
            var5.append(float(var6['k436']))
    var4 = np.argsort(var7)
    return k285(k338=np.asarray(var7, dtype=np.float64)[var4], k388=np.asarray(var5, dtype=np.float64)[var4], k358=float(k358))

def k434(k309: np.ndarray, capacity_wh: np.ndarray, k232: k285) -> np.ndarray:
    var8 = np.interp(k309, k232.k338, k232.k388)
    return var8 * (capacity_wh / k232.k358)

@dataclass
class k218:
    solar_wh: np.ndarray
    load_wh: np.ndarray
    import_price: np.ndarray
    export_price: np.ndarray
    loadshedding: np.ndarray
    soc0_pct: np.ndarray
    capacity_wh: np.ndarray
    shutoff_pct: np.ndarray
    max_discharge_w: np.ndarray
    k304: np.ndarray
    k91: np.ndarray
    k315: np.ndarray
    k95: np.ndarray
    k307: np.ndarray
    k111: np.ndarray
    k382: np.ndarray
    k64: np.ndarray
    k133: np.ndarray
    k156: np.ndarray
    k17: np.ndarray
    k261: np.ndarray

    @property
    def count(self) -> int:
        return self.solar_wh.shape[0]

def var1(var10: np.ndarray) -> list[tuple[int, int]]:
    var9: list[tuple[int, int]] = []
    var13: int | None = None
    for var11 in range(k329):
        var12 = var10[var11] > 0.5
        if var12 and var13 is None:
            var13 = var11
        elif not var12 and var13 is not None:
            var9.append((var13, var11))
            var13 = None
    if var13 is not None:
        var9.append((var13, k329))
    return var9

def var2(var20: k12, var21: float) -> tuple[np.ndarray, ...]:
    var27 = np.zeros(k329)
    var25 = np.full(k329, float(k329))
    var14 = np.zeros(k329)
    var30 = np.zeros(k329)
    var16 = var20.import_price.astype(np.float64).copy()
    var15 = var1(var20.loadshedding)
    if not var15:
        return (var27, var25, var14, var30, var16)
    var23 = np.maximum(0.0, var20.load_wh - var20.solar_wh)
    var24 = np.maximum(0.0, var20.solar_wh - var20.load_wh)
    for var22 in range(k329):
        var26 = [(var28, var18) for (var28, var18) in var15 if var18 > var22]
        if not var26:
            continue
        var14[var22] = sum((var23[max(var22, var28):var18].sum() for (var28, var18) in var26)) / var21
        (var29, var19) = var26[0]
        var17 = max(var22, var29)
        var27[var22] = var23[var17:var19].sum() / var21
        var25[var22] = max(0.0, var29 - var22)
        if var29 > var22:
            var30[var22] = var24[var22 + 1:var29].sum()
            var16[var22] = var20.import_price[var22:var29].min()
    return (var27, var25, var14, var30, var16)

def k328(var36: Sequence[k12], k186: float) -> k218:
    var37 = float(np.sqrt(k186))
    var57 = np.stack([var34.solar_wh for var34 in var36]).astype(np.float64)
    var47 = np.stack([var34.load_wh for var34 in var36]).astype(np.float64)
    var46 = np.stack([var34.import_price for var34 in var36]).astype(np.float64)
    var39 = np.stack([var34.export_price for var34 in var36]).astype(np.float64)
    var56 = np.stack([var34.loadshedding for var34 in var36]).astype(np.float64)
    var48 = np.argsort(var46, axis=1, kind='stable')
    var51 = np.empty_like(var46)
    var54 = np.arange(len(var36))[:, None]
    var51[var54, var48] = np.arange(k329)[None, :] / (k329 - 1)
    var49 = var46 / np.maximum(var46.mean(axis=1, keepdims=True), 1e-09)
    var53 = np.maximum.accumulate(var46[:, ::-1], axis=1)[:, ::-1]
    var41 = np.empty_like(var46)
    var41[:, :-1] = var53[:, 1:]
    var41[:, -1] = var46[:, -1]
    var42 = var41 / np.maximum(var46, 1e-09)
    var55 = np.minimum.accumulate(var46, axis=1)
    var50 = var46 / np.maximum(var55, 1e-09)
    var38 = var39 / np.maximum(var46, 1e-09)
    var58 = np.maximum(0.0, var57 - var47)
    var33 = np.maximum(0.0, var47 - var57)
    var43 = np.cumsum(var58[:, ::-1], axis=1)[:, ::-1] - var58
    var40 = np.cumsum(var33[:, ::-1], axis=1)[:, ::-1] - var33
    var52 = np.zeros_like(var57)
    var44 = np.zeros_like(var57)
    var31 = np.zeros_like(var57)
    var59 = np.zeros_like(var57)
    var32 = np.zeros_like(var57)
    for (var45, var35) in enumerate(var36):
        (var52[var45], var44[var45], var31[var45], var59[var45], var32[var45]) = var2(var35, var37)
    return k218(solar_wh=var57, load_wh=var47, import_price=var46, export_price=var39, loadshedding=var56, soc0_pct=np.asarray([var34.soc0_pct for var34 in var36], dtype=np.float64), capacity_wh=np.asarray([var34.capacity_wh for var34 in var36], dtype=np.float64), shutoff_pct=np.asarray([var34.shutoff_pct for var34 in var36], dtype=np.float64), max_discharge_w=np.asarray([var34.max_discharge_w for var34 in var36], dtype=np.float64), k304=var51, k91=var49, k315=var42, k95=var50, k307=var38, k111=var43, k382=var40, k64=var52, k133=var44, k156=var31, k17=var59, k261=var32)

def k132(var72: k218, var63: np.ndarray, var66: int) -> np.ndarray:
    var61 = var72.capacity_wh[None, :]
    var64 = var72.shutoff_pct[None, :] / 100.0 * var61
    var74 = np.maximum(var61 - var64, 1e-09)
    var70 = np.clip((var63 - var64) / var74, 0.0, 1.0)
    var71 = np.clip(var63 / var61, 0.0, 1.0)
    var65 = np.clip((var61 - var63) / var61, 0.0, 1.0)
    var67 = (var72.load_wh[:, var66] - var72.solar_wh[:, var66])[None, :] / var61
    var73 = var63 - var64
    var68 = (var72.k64[:, var66][None, :] - var73) / var61
    var60 = (var72.k156[:, var66][None, :] - var73) / var61

    def spread(var69: np.ndarray) -> np.ndarray:
        return np.broadcast_to(var69[None, :], var63.shape)
    var62 = [var70, var71, var65, spread(var72.k304[:, var66]), spread(var72.k91[:, var66]), spread(np.clip(var72.k315[:, var66], 0.0, 4.0)), spread(np.clip(var72.k95[:, var66], 0.0, 4.0)), spread(var72.k307[:, var66]), np.clip(spread(var67[0]), -2.0, 2.0), spread(np.clip(var72.k111[:, var66] / var72.capacity_wh, 0.0, 4.0)), spread(np.clip(var72.k382[:, var66] / var72.capacity_wh, 0.0, 4.0)), np.clip(var68, -1.0, 1.0), np.clip(var60, -1.0, 1.0), spread(np.clip(var72.k17[:, var66] / var72.capacity_wh, 0.0, 4.0)), spread(np.clip(var72.k261[:, var66] / np.maximum(var72.import_price[:, var66], 1e-09), 0.0, 1.0)), spread(np.minimum(var72.k133[:, var66] / k329, 1.0)), spread(var72.loadshedding[:, var66]), np.broadcast_to(np.float64(k195[var66]), var63.shape), np.broadcast_to(np.float64(k192[var66]), var63.shape)]
    return np.stack(var62, axis=-1)

@dataclass
class k56:
    k106: np.ndarray
    effective_cost: np.ndarray
    k191: np.ndarray
    k22: np.ndarray
    k123: np.ndarray
    k138: np.ndarray
    k217: np.ndarray
    k453: np.ndarray
    k353: np.ndarray
    k298: np.ndarray
    k309: np.ndarray
    k129: np.ndarray
    k176: np.ndarray
    k30: np.ndarray
    k203: np.ndarray
k1 = Callable[[np.ndarray, int, np.ndarray], Tuple[np.ndarray, np.ndarray]]

def k404(var76: k1, var125: k218, var113: int, k232: k285, k186: float=k440, var96: 'k218 | None'=None, var128: np.ndarray | None=None) -> k56:
    var92 = var93 = float(np.sqrt(k186))
    var91 = var125.count
    var117 = (var113, var91)
    var81 = np.broadcast_to(var125.capacity_wh[None, :], var117)
    var100 = np.broadcast_to((var125.shutoff_pct / 100.0 * var125.capacity_wh)[None, :], var117)
    var107 = np.broadcast_to(var125.max_discharge_w[None, :], var117)
    var90 = np.broadcast_to((var125.soc0_pct / 100.0 * var125.capacity_wh)[None, :], var117).copy()
    var85 = np.zeros(var117)
    var131 = np.zeros(var117)
    var130 = np.zeros(var117)
    var123 = var90.copy()
    var77 = np.zeros((*var117, k329))
    var103 = np.zeros((*var117, k329))
    var94 = np.zeros((*var117, k329))
    var129 = np.zeros((*var117, k329))
    var112 = np.zeros((*var117, k329), dtype=bool)
    var119 = np.zeros((*var117, k329 + 1))
    var119[..., 0] = var90 / var81 * 100.0
    for var102 in range(k329):
        var121 = var125.solar_wh[None, :, var102]
        var105 = var125.load_wh[None, :, var102]
        var118 = var125.loadshedding[None, :, var102] > 0.5
        var110 = var125.import_price[None, :, var102]
        var109 = var125.export_price[None, :, var102]
        var97 = k132(var96 or var125, var90, var102)
        (var75, var111) = var76(var97, var102, var90)
        var120 = var90 / var81 * 100.0
        var83 = np.maximum(np.minimum(k434(var120, var81, k232), var81 - var90), 0.0)
        var87 = np.maximum(np.minimum(var107, var90 - var100), 0.0)
        var101 = np.maximum(0.0, var105 - var121) / var93
        var116 = ~var111 | var118
        var87 = np.where(var116, np.minimum(var87, var101), var87)
        var124 = np.where(var75 >= 0, var75 * var83, var75 * var87)
        var122 = np.maximum(0.0, var121 - var105)
        var78 = np.minimum(var122, var83)
        var124 = np.where(var118 & (var122 > 0.0), np.maximum(var124, var78), var124)
        var84 = var124 >= 0
        var88 = np.where(var84, var124 / var92, 0.0)
        var86 = np.where(var84, 0.0, -var124 * var93)
        var108 = var121 + var86 - (var105 + var88)
        var95 = np.where(var118, 0.0, np.maximum(var108, 0.0))
        var104 = np.where(var118, 0.0, np.maximum(-var108, 0.0))
        var132 = np.where(var118, np.maximum(-var108, 0.0), 0.0)
        var85 += var104 / 1000.0 * var110 - var95 / 1000.0 * var109
        var131 += var132 / 1000.0 * var110
        var130 += var132
        var90 = np.clip(var90 + var124, 0.0, var81)
        var77[..., var102] = var124
        var103[..., var102] = var104
        var94[..., var102] = var95
        var129[..., var102] = var132
        var112[..., var102] = var111
        var119[..., var102 + 1] = var90 / var81 * 100.0
    var82 = var125.import_price.min(axis=1)[None, :]
    var126 = (var90 - var123) / 1000.0 * var82 * var93
    var89 = var85 + var131 - var126
    var80 = k164(var125)[None, :]
    var106 = np.maximum((var125.load_wh * var125.import_price).sum(axis=1) / 1000.0, 1e-09)[None, :]
    var127 = np.maximum(var125.load_wh.sum(axis=1), 1e-09)[None, :]
    var114 = (var80 - var89) / var106
    var99 = np.zeros_like(var127) if var128 is None else np.asarray(var128, dtype=np.float64)[None, :]
    var79 = np.maximum(0.0, var130 - var99)
    var98 = var114 - k255 * (var79 / var127)
    with np.errstate(divide='ignore', invalid='ignore'):
        var115 = np.where(np.abs(var80) > 1e-09, (var80 - var89) / np.abs(var80) * 100.0, 0.0)
    return k56(k106=var85, effective_cost=var89, k191=np.broadcast_to(var80, var117).copy(), k22=var130, k123=var79, k138=var114, k217=var115, k453=var98, k353=var130 > k4, k298=var77, k309=var119, k129=var103, k176=var94, k30=var129, k203=var112)

def k55(var138: dict) -> k1:

    def action_fn(var136: np.ndarray, var137: int, var134: np.ndarray):
        (var133, var135) = policy_module.forward(var138, var136)
        return (var133, var135 > 0.0)
    return action_fn

def k402(var144: np.ndarray, var145: k218, k232: k285, k186: float=k440, var140: 'k218 | None'=None, var146: np.ndarray | None=None, var139=None) -> k56:
    var141 = np.atleast_2d(np.asarray(var144, dtype=np.float64))
    var143 = policy_module.unflatten(var141)
    var142 = k55(var143)
    if var139 is not None:
        var142 = var139(var142, var140 or var145)
    return k404(var142, var145, var141.shape[0], k232, k186, var140, var146)

def k164(var151: k218) -> np.ndarray:
    var150 = var151.loadshedding > 0.5
    var149 = var151.solar_wh - var151.load_wh
    var147 = np.where(var150, 0.0, np.maximum(var149, 0.0))
    var148 = np.where(var150, 0.0, np.maximum(-var149, 0.0))
    var152 = np.where(var150, np.maximum(-var149, 0.0), 0.0)
    return ((var148 + var152) * var151.import_price / 1000.0 - var147 * var151.export_price / 1000.0).sum(axis=1)

def k296(var154: 'k56') -> np.ndarray:
    var155 = (var154.k191 - var154.effective_cost).sum(axis=-1)
    var153 = var154.k191.sum(axis=-1)
    return np.where(np.abs(var153) > 1e-09, var155 / np.abs(var153) * 100.0, 0.0)

def k318() -> k1:

    def action_fn(var157: np.ndarray, var158: int, var156: np.ndarray):
        return (np.zeros(var156.shape), np.zeros(var156.shape, dtype=bool))
    return action_fn

def k166(var172: k218, k232: k285, k186: float) -> k1:
    var164 = float(np.sqrt(k186))

    def action_fn(var165: np.ndarray, var166: int, var163: np.ndarray):
        var170 = var172.solar_wh[None, :, var166]
        var167 = var172.load_wh[None, :, var166]
        var160 = var172.capacity_wh[None, :]
        var171 = np.maximum(0.0, var170 - var167)
        var169 = var163 / var160 * 100.0
        var162 = np.maximum(np.minimum(k434(var169, var160, k232), var160 - var163), 0.0)
        var161 = np.where(var162 > 1e-09, np.minimum(1.0, var171 * var164 / np.maximum(var162, 1e-09)), 0.0)
        var159 = np.where(var171 > 0.0, var161, -1.0)
        var168 = np.zeros(var163.shape, dtype=bool)
        return (np.broadcast_to(var159, var163.shape).copy(), var168)
    return action_fn

def k448(var179: k218) -> k1:

    def action_fn(var175: np.ndarray, var176: int, var174: np.ndarray):
        var178 = var179.loadshedding[None, :, var176] > 0.5
        var173 = np.where(var178, -1.0, 1.0)
        var177 = np.zeros(var174.shape, dtype=bool)
        return (np.broadcast_to(var173, var174.shape).copy(), var177)
    return action_fn
