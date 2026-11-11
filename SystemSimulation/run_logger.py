# Copyright (c) 2026 Daniel Jackson and Nelson Mandela University. All rights reserved.
# This code is proprietary. It may not be used, copied, modified, distributed or
# reused in whole or in part, nor may the ideas it embodies be used, without the
# prior written permission of the copyright holders. See LICENSE.
from __future__ import annotations
import json
from pathlib import Path
from simulation import k319
k165 = Path(__file__).resolve().parent / 'Data' / 'runs'

def var1(var3: str, var5, out: dict) -> None:
    if isinstance(var5, dict):
        for (var2, var4) in var5.items():
            var1(f'{var3}.{var2}' if var3 else var2, var4, out)
    else:
        out[var3] = var5

def k96(var11: k319, var10: Path | None=None) -> Path:
    var10 = var10 or k165 / f'{var11.run_id}.csv'
    var10.parent.mkdir(parents=True, exist_ok=True)
    var6: dict = {}
    var1('', var11.k205, var6)
    var9 = []
    for (var8, var12) in var6.items():
        if isinstance(var12, (dict, list)):
            var12 = json.dumps(var12)
        var9.append(f'# {var8}: {var12}')
    var9.append('#')
    with var10.open('w', encoding='utf-8', newline='') as var7:
        var7.write('\n'.join(var9) + '\n')
        var11.k306.to_csv(var7, index=False)
    return var10

def k45(var14: Path) -> dict:
    var18: dict = {}
    with var14.open('r', encoding='utf-8') as var15:
        for var17 in var15:
            if not var17.startswith('#'):
                break
            var13 = var17[1:].strip()
            if not var13 or ':' not in var13:
                continue
            (var16, _, var19) = var13.partition(':')
            var18[var16.strip()] = var19.strip()
    return var18
