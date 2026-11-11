# Copyright (c) 2026 Daniel Jackson and Nelson Mandela University. All rights reserved.
# This code is proprietary. It may not be used, copied, modified, distributed or
# reused in whole or in part, nor may the ideas it embodies be used, without the
# prior written permission of the copyright holders. See LICENSE.
from __future__ import annotations
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable, Optional
import numpy as np
import pandas as pd
import torch
k44 = Path(__file__).resolve().parent.parent
k134 = ['SolarModelV1', 'SolarModelV2']
k324 = ['LoadModelV1', 'LoadModelV2', 'LoadModelV3']
k168 = ['ManagementModelV1', 'ManagementModelV2']
k80 = 'ManagementModelV2'
k230 = {'solar': 'solar_model.pt', 'load': 'load_model.pt', 'management': 'management_model.json'}
var1 = ('data', 'model', 'training', 'environment', 'synthetic_data', 'predict', 'policy', 'heuristic', 'ga')
var2: dict[str, ModuleType] = {}

@dataclass
class k377:
    version: str
    family: str
    k276: Path
    k213: Path
    k205: Optional[dict]
    label: str

def var3(version: str) -> str:
    if version.startswith('SolarModel'):
        return 'solar_' + version.replace('SolarModel', '').lower()
    if version.startswith('LoadModel'):
        return 'load_' + version.replace('LoadModel', '').lower()
    return 'management_' + version.replace('ManagementModel', '').lower()

def var4(var13: Path, version: str, family: str) -> list[k377]:
    var15 = var13 / version / 'runs'
    if not var15.exists():
        return []
    var7 = k230[family]
    var9: list[k377] = []
    for var14 in sorted((var12 for var12 in var15.iterdir() if var12.is_dir())):
        var6 = var14 / var7
        if not var6.exists():
            continue
        var11 = var14 / 'run_metadata.json'
        var10 = None
        if var11.exists():
            try:
                var10 = json.loads(var11.read_text(encoding='utf-8'))
            except (json.JSONDecodeError, OSError):
                var10 = None
        var9.append(k377(version=version, family=family, k276=var14, k213=var6, k205=var10, label=f'{version} / {var14.name}'))
    var9.sort(key=lambda var8: var8.k213.stat().st_mtime, reverse=True)
    return var9

def k50(var16: Path=k44) -> dict[str, list[k377]]:
    var17: dict[str, list[k377]] = {}
    for var18 in k134:
        var17[var3(var18)] = var4(var16, var18, 'solar')
    for var18 in k324:
        var17[var3(var18)] = var4(var16, var18, 'load')
    for var18 in k168:
        var17[var3(var18)] = var4(var16, var18, 'management')
    return var17

def k185(var28: Path, var21: str, var27: str) -> ModuleType:
    var22 = Path(var21).stem
    var19 = f'{var27}:{var22}'
    if var19 in var2:
        return var2[var19]
    var25 = list(sys.path)
    var24 = {var23: sys.modules[var23] for var23 in var1 if var23 in sys.modules}
    for var23 in var1:
        sys.modules.pop(var23, None)
    sys.path.insert(0, str(var28))
    try:
        var26 = importlib.util.spec_from_file_location(var22, var28 / var21)
        var20 = importlib.util.module_from_spec(var26)
        sys.modules[var22] = var20
        var26.loader.exec_module(var20)
        var2[var19] = var20
        return var20
    finally:
        sys.path[:] = var25
        for var23 in var1:
            sys.modules.pop(var23, None)
        sys.modules.update(var24)

@dataclass
class k348:
    k59: Callable[[pd.DataFrame], np.ndarray]
    k231: list[str]
    k85: k377

@dataclass
class k379:
    k59: Callable[[pd.DataFrame], np.ndarray]
    k231: list[str]
    k354: Callable[[list[Path]], pd.DataFrame]
    k85: k377

def var5(var32, var31: np.ndarray, var35: np.ndarray, k231: list[str], device: str):

    def k59(var29: pd.DataFrame) -> np.ndarray:
        var30 = var29[k231].astype(float).to_numpy(dtype=np.float32)
        var33 = (var30 - var31) / var35
        with torch.no_grad():
            var34 = var32(torch.from_numpy(var33).to(device))
        return var34.cpu().numpy()
    return k59

def k101(var39: k377, device: str) -> k348:
    var44 = k44 / var39.version
    var37 = k185(var44, 'data.py', f'solar_{var39.version}')
    var42 = k185(var44, 'model.py', f'solar_{var39.version}')
    var36 = torch.load(var39.k213, map_location=device)
    var41 = var42.k395(var36, device)
    var41.eval()
    var40 = np.asarray(var36['k419'], dtype=np.float32)
    var43 = np.asarray(var36['k374'], dtype=np.float32)
    var38 = list(var37.k130)
    return k348(k59=var5(var41, var40, var43, var38, device), k231=var38, k85=var39)

def k147(var48: k377, device: str) -> k379:
    var55 = k44 / var48.version
    var46 = k185(var55, 'data.py', f'load_{var48.version}')
    var51 = k185(var55, 'model.py', f'load_{var48.version}')
    var45 = torch.load(var48.k213, map_location=device)
    if hasattr(var51, 'k395'):
        var50 = var51.k395(var45, device)
    else:
        var50 = var51.k113(k81=int(var45['k81']), k103=list(var45['k103'])).to(device)
        var50.load_state_dict(var45['k204'])
    var50.eval()
    var49 = np.asarray(var45['k419'], dtype=np.float32)
    var54 = np.asarray(var45['k374'], dtype=np.float32)
    var47 = list(var45.get('k231', var46.k130))
    if hasattr(var46, 'k384'):
        var53 = None
        if var45.get('k6') is not None and hasattr(var51, 'k92'):
            var53 = var51.k92(var45, device)
            var53.eval()
        var52 = var46.k384(var50, var45, device, refiner=var53)
    else:
        var52 = var5(var50, var49, var54, var47, device)
    return k379(k59=var52, k231=var47, k354=var46.k354, k85=var48)

def k73(var56: k377, device: str):
    var58 = k44 / var56.version
    var57 = k185(var58, 'predict.py', f'management_{var56.version}')
    return var57.k10(var56.k213, device)

def k88(version: str=k80):
    var59 = k44 / version
    return k185(var59, 'environment.py', f'management_{version}')

def k104(version: str=k80):
    var60 = k44 / version
    return k185(var60, 'predict.py', f'management_{version}')

def k438(version: str=k80):
    var61 = k44 / version
    return k185(var61, 'optimal.py', f'management_{version}')

def k408(version: str=k80):
    var62 = k44 / version
    return k185(var62, 'synthetic_data.py', f'management_{version}')
