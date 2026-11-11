# Copyright (c) 2026 Daniel Jackson and Nelson Mandela University. All rights reserved.
# This code is proprietary. It may not be used, copied, modified, distributed or
# reused in whole or in part, nor may the ideas it embodies be used, without the
# prior written permission of the copyright holders. See LICENSE.
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np
k329 = 24
k212 = np.arange(k329, dtype=np.float64)
k195 = np.sin(2.0 * np.pi * k212 / k329)
k192 = np.cos(2.0 * np.pi * k212 / k329)
k425 = (5.0, 7.0)
k34 = (17.5, 19.5)
k235 = (3000.0, 11000.0)
k386 = 1.2
k325 = (2.0, 2.0)
k128 = 0.18
k197 = (0.3, 1.4)
k344 = 0.05
k209 = (0.6, 1.8)
k350 = (280.0, 560.0)
k237 = (6.0, 9.0)
k385 = (1200.0, 2800.0)
k183 = (1.0, 2.0)
k63 = (17.0, 21.0)
k422 = (2000.0, 4200.0)
k100 = (1.5, 3.0)
k239 = 0.1
k90 = np.array([7, 8, 9, 18, 19])
k391 = np.array([6, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21])
k240 = (6.8308, 3.0509, 2.1219)
k244 = (4.3692, 1.6215, 1.1791)
k142 = 2.0 / 7.0
k367 = (0.85, 1.15)
k332 = (0.85, 1.15)
k215 = [0, 1, 2, 3]
k433 = [0.4, 0.35, 0.2, 0.05]
k174 = [2, 2, 4]
k444 = 4.0
k109 = 20
k314 = 3
k321 = [2, 3]
k252 = 2.0
k394 = 0.3
k428 = {0: 0.4, 1: 0.2, 2: 0.15, 3: 0.1, 4: 0.15}
k356 = (6000.0, 16000.0)
k200 = (4000.0, 30000.0)
k403 = (1.3, 2.5)
k42 = (2000.0, 12000.0)
k193 = (10.0, 30.0)

@dataclass
class k12:
    solar_wh: np.ndarray
    load_wh: np.ndarray
    import_price: np.ndarray
    export_price: np.ndarray
    loadshedding: np.ndarray
    soc0_pct: float
    capacity_wh: float
    shutoff_pct: float
    max_discharge_w: float

def var1(hours: np.ndarray, var9: float, var10: float) -> np.ndarray:
    return np.exp(-0.5 * ((hours - var9) / var10) ** 2)

def var2(var17: np.random.Generator) -> np.ndarray:
    var18 = var17.uniform(*k425)
    var19 = var17.uniform(*k34)
    var16 = var17.uniform(*k235)
    var14 = max(var19 - var18, 1e-06)
    var13 = np.clip(np.sin(np.pi * (k212 - var18) / var14), 0.0, 1.0)
    var13 = np.where((k212 >= var18) & (k212 <= var19), var13, 0.0)
    var11 = var16 * var13 ** k386
    var12 = float(var17.beta(*k325))
    var20 = np.clip(var17.normal(1.0, k128 * (1.0 - var12), size=k329), *k197)
    var15 = var17.normal(1.0, k344, size=k329)
    return np.maximum(0.0, np.round(var11 * var12 * var20 * var15))

def var3(var30: np.random.Generator) -> tuple[np.ndarray, float]:
    var25 = var30.uniform(*k209)
    var21 = var30.uniform(*k350) * var25
    var27 = var30.uniform(*k237)
    var28 = var30.uniform(*k385) * var25
    var29 = var30.uniform(*k183)
    var22 = var30.uniform(*k63)
    var23 = var30.uniform(*k422) * var25
    var24 = var30.uniform(*k100)
    var26 = var21 + var28 * var1(k212, var27, var29) + var23 * var1(k212, var22, var24) + var30.normal(0.0, k239 * var21, size=k329)
    return (np.maximum(0.0, var26), var25)

def var4(var32: float, var31: float) -> np.ndarray:
    if var31 <= var32:
        return (k212 >= var32) | (k212 < var31)
    return (k212 >= var32) & (k212 < var31)

def var5(var39: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    (var36, var33) = (k240, k244)
    var38 = var39.uniform(*k367)
    var34 = var38 * var39.uniform(*k332)
    if var39.random() < k142:
        var37 = np.full(k329, var36[2] * var38)
        var35 = np.full(k329, var33[2] * var34)
        return (var37, var35)
    var37 = np.full(k329, var36[2] * var38)
    var35 = np.full(k329, var33[2] * var34)
    var37[k391] = var36[1] * var38
    var35[k391] = var33[1] * var34
    var37[k90] = var36[0] * var38
    var35[k90] = var33[0] * var34
    var35 = np.minimum(var35, var37 * 0.99)
    return (var37, var35)

def k108(var52: np.random.Generator, var41: int, var44: list[int], var50: float) -> np.ndarray:
    var49 = np.zeros(k329, dtype=np.float64)
    var51: list[tuple[int, int]] = []
    for _ in range(var41):
        var43 = int(var52.choice(var44))
        for var40 in range(k109):
            var45 = int(var52.integers(0, k329 - var43 + 1))
            var42 = var45 + var43
            var46 = any((var45 < var47 + var50 and var48 < var42 + var50 for (var48, var47) in var51))
            if not var46:
                var51.append((var45, var42))
                var49[var45:var42] = 1.0
                break
    return var49

def k107(var53: np.random.Generator, var54: int) -> np.ndarray:
    if var54 >= 4:
        return k108(var53, k314, k321, k252)
    return k108(var53, int(var54), k174, k444)

def var6(var55: np.random.Generator) -> tuple[np.ndarray, int]:
    if var55.random() < k394:
        return (k107(var55, 4), 4)
    var56 = int(var55.choice(k215, p=k433))
    return (k107(var55, var56), var56)

def var7(var59: np.random.Generator) -> np.ndarray:
    (var58, var57) = var6(var59)
    return var58

def k223(var63: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    (var61, var60) = var5(var63)
    (var62, var64) = var6(var63)
    return (var61, var60, var62, var64)

def k449(var73: np.random.Generator, k364: int | None=None) -> k12:
    var76 = var2(var73)
    (var69, var67) = var3(var73)
    (var68, var66) = var5(var73)
    if k364 is None:
        var70 = var7(var73)
    else:
        var70 = k107(var73, k364)
    var65 = np.clip(var67 * var73.uniform(*k356), *k200)
    var72 = float(var69.max())
    var71 = np.clip(var72 * var73.uniform(*k403), *k42)
    var74 = var73.uniform(*k193)
    var75 = var73.uniform(var74, 100.0)
    return k12(solar_wh=var76, load_wh=var69, import_price=var68, export_price=var66, loadshedding=var70, soc0_pct=float(var75), capacity_wh=float(var65), shutoff_pct=float(var74), max_discharge_w=float(var71))

def k268(var77: int, var78: np.random.Generator) -> List[k12]:
    return [k449(var78) for _ in range(var77)]

def k267(var81: int) -> List[int]:
    var85 = sorted(k428)
    var80 = {var84: k428[var84] * var81 for var84 in var85}
    var79 = {var84: int(np.floor(var80[var84])) for var84 in var85}
    var82 = var81 - sum(var79.values())
    for var84 in sorted(var85, key=lambda var83: var80[var83] - np.floor(var80[var83]), reverse=True)[:var82]:
        var79[var84] += 1
    return [var84 for var84 in var85 for _ in range(var79[var84])]

def k320(var86: int, var87: np.random.Generator) -> List[k12]:
    return [k449(var87, k364=var88) for var88 in k267(var86)]
k224 = 1.145
k417 = 0.405
k214 = 0.02
k270 = 0.149
k219 = 0.35
k427 = 1.005
k14 = 0.014
k423 = 0.381
k441 = 0.0
k46 = 0.0
k144 = (0.25, 2.5)

def var8(var93: np.random.Generator, var94: float, var92: float) -> np.ndarray:
    if var94 <= 0.0:
        return np.zeros(k329)
    if var92 == 0.0:
        return var93.normal(0.0, var94, size=k329)
    var90 = var94 * np.sqrt(1.0 - var92 ** 2)
    var91 = np.empty(k329)
    var91[0] = var93.normal(0.0, var94)
    for var89 in range(1, k329):
        var91[var89] = var92 * var91[var89 - 1] + var93.normal(0.0, var90)
    return var91

def k222(var100: k12, var104: np.random.Generator) -> k12:

    def perturb(var105, var97, var99, var102, var95, var101):
        var98 = float(np.clip(var104.normal(var97, var99), *k144))
        var103 = 1.0 + var8(var104, var102, var101)
        var96 = var8(var104, var95 * float(var105.mean()), var101) if var95 > 0.0 else 0.0
        return np.maximum(0.0, var105 * var98 * var103 + var96)
    return k12(solar_wh=perturb(var100.solar_wh, k224, k417, k214, k270, k219), load_wh=perturb(var100.load_wh, k427, k14, k423, k441, k46), import_price=var100.import_price, export_price=var100.export_price, loadshedding=var100.loadshedding, soc0_pct=var100.soc0_pct, capacity_wh=var100.capacity_wh, shutoff_pct=var100.shutoff_pct, max_discharge_w=var100.max_discharge_w)
