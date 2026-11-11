# Copyright (c) 2026 Daniel Jackson and Nelson Mandela University. All rights reserved.
# This code is proprietary. It may not be used, copied, modified, distributed or
# reused in whole or in part, nor may the ideas it embodies be used, without the
# prior written permission of the copyright holders. See LICENSE.
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
import feature_adapter
import registry
import tariff_loadshedding
k329 = 24

@dataclass
class k78:
    capacity_wh: float = 10000.0
    shutoff_pct: float = 20.0
    max_discharge_w: float = 10000.0
    k186: float = 0.92
    k297: float = 8.8
    k292: Path | None = None
    k447: float | None = None

@dataclass
class k148:
    date: date
    hours: pd.DataFrame
    k333: float
    k220: float
    k191: float

@dataclass
class k319:
    run_id: str
    days: list[k148]
    k306: pd.DataFrame
    k205: dict

def var1(errors: np.ndarray) -> float:
    return float(np.sqrt(np.mean(errors ** 2))) if len(errors) else float('nan')

def var2(errors: np.ndarray) -> float:
    return float(np.mean(np.abs(errors))) if len(errors) else float('nan')

def var3(var12: np.ndarray, var13: np.ndarray) -> float:
    if len(var12) < 2:
        return float('nan')
    var14 = np.sum((var12 - var13) ** 2)
    var15 = np.sum((var12 - var12.mean()) ** 2)
    return float(1.0 - var14 / var15) if var15 > 0 else float('nan')

def var4(k201: Path, k24: date, k414: date, k199: str | None, var26: 'registry.k348 | None', var18: 'registry.k379 | None') -> tuple[pd.DataFrame, bool, bool]:
    var16 = var6(var18) if var18 is not None else None
    var21 = getattr(var16, 'k27', feature_adapter.k241)
    var23 = feature_adapter.k347(k201, k24, k414, k199, k38=var21)
    var24 = feature_adapter.k5(var23)
    var29 = var26 is None
    if var26 is not None:
        var25 = feature_adapter.k259(var24, var5(var26))
        var27 = var26.k59(var25)
    else:
        var27 = var24['k407'].to_numpy()
    var28 = var18 is None
    if var18 is not None:
        var17 = feature_adapter.k41(var23, var16)
        var17 = var17.dropna(subset=var18.k231)
        var19 = var18.k59(var17)
        var20 = pd.Series(var19, index=var17['time'].values)
    else:
        var20 = None
    var22 = var24[['time', 'k407', 'load', 'k155', 'k249', 'k346']].copy()
    var22['k119'] = var27
    if var20 is not None:
        var22 = var22.set_index('time')
        var22['k173'] = var20
        var22 = var22.reset_index()
        var22['k369'] = var22['k173'].isna()
        var22['k173'] = var22['k173'].fillna(var22['load'])
    else:
        var22['k173'] = var22['load']
        var22['k369'] = False
    var22['date'] = var22['time'].dt.date
    var22['hour'] = var22['time'].dt.hour
    var22 = var22[(var22['date'] >= k24) & (var22['date'] <= k414)].reset_index(drop=True)
    return (var22, var29, var28)

def var5(var30: 'registry.k348'):
    return registry.k185(registry.k44 / var30.k85.version, 'data.py', f'solar_{var30.k85.version}')

def var6(var31: 'registry.k379'):
    return registry.k185(registry.k44 / var31.k85.version, 'data.py', f'load_{var31.k85.version}')

def k40(var34, k232, var35, var33):
    var36 = var33.k328([var34], var35)
    var32 = var33.k448(var36)
    return var33.k404(var32, var36, 1, k232, var35)

def k390(var39, var37, var38, var41, k232=None, k186=None, k340=True, k124=5.0):
    if var39 is None:
        var40 = registry.k438()
        return var40.k35(var37, k232, k186)
    return var41.k94(var39, var37, var38, k340=k340, k124=k124)

def var7(k129: np.ndarray, k176: np.ndarray, import_price: np.ndarray, export_price: np.ndarray) -> np.ndarray:
    return k129 / 1000.0 * import_price - k176 / 1000.0 * export_price

def var8(k22: np.ndarray, import_price: np.ndarray) -> np.ndarray:
    return k22 / 1000.0 * import_price

def k187(var65: date, var67: int, var66: pd.DataFrame, var56: float, k84: k78, k232, var77, var69, var83, var102, var81) -> tuple[pd.DataFrame, float, float, float]:
    var66 = var66.sort_values('hour').reset_index(drop=True)
    assert len(var66) == k329, f'Expected 24 rows for {var65}, got {len(var66)}'
    var89 = feature_adapter.k372(var66['k407'].to_numpy(), k84.k297)
    var90 = feature_adapter.k372(var66['k119'].to_numpy(), k84.k297)
    var75 = feature_adapter.k415(var66['load']).to_numpy()
    var76 = feature_adapter.k415(var66['k173']).to_numpy()
    var74 = var66['import_price'].to_numpy()
    var71 = var66['export_price'].to_numpy()
    var78 = var66['loadshedding'].to_numpy()
    var42 = var102.k12
    var48 = var42(solar_wh=var90, load_wh=var76, import_price=var74, export_price=var71, loadshedding=var78, soc0_pct=var56, capacity_wh=k84.capacity_wh, shutoff_pct=k84.shutoff_pct, max_discharge_w=k84.max_discharge_w)
    var51 = var42(solar_wh=var89, load_wh=var75, import_price=var74, export_price=var71, loadshedding=var78, soc0_pct=var56, capacity_wh=k84.capacity_wh, shutoff_pct=k84.shutoff_pct, max_discharge_w=k84.max_discharge_w)
    var63 = var42(solar_wh=var89, load_wh=var75, import_price=var74, export_price=var71, loadshedding=var78, soc0_pct=100.0, capacity_wh=k84.capacity_wh, shutoff_pct=k84.shutoff_pct, max_discharge_w=k84.max_discharge_w)
    var46 = k390(var77, var51, var48, var83, k232=k232, k186=k84.k186)
    var86 = var46['actual']
    var85 = var46['raw']
    var84 = var85.k298[0, 0]
    var88 = var46['k167']
    var64 = k40(var63, k232, k84.k186, var69)
    var57 = var86.k309[0, 0, 1:]
    var53 = var86.k129[0, 0]
    var49 = var86.k176[0, 0]
    var62 = var86.k30[0, 0]
    var44 = var86.k298[0, 0]
    var70 = var46['expected']
    var45 = var70.k298[0, 0]
    var54 = var70.k129[0, 0]
    var50 = var70.k176[0, 0]
    var58 = var70.k309[0, 0, 1:]
    var97 = var64.k309[0, 0, 1:]
    var95 = var64.k129[0, 0]
    var93 = var64.k176[0, 0]
    var101 = var64.k30[0, 0]
    var91 = var64.k298[0, 0]
    var72 = var81.k35(var51, k232, k84.k186)
    var103 = var72['actual'].k30[0, 0]
    var60 = np.maximum(0.0, var62 - var103)
    var99 = np.maximum(0.0, var101 - var103)
    var52 = var7(var53, var49, var74, var71)
    var94 = var7(var95, var93, var74, var71)
    var80 = var7(var75, np.zeros_like(var75), var74, var71)
    var61 = var8(var62, var74)
    var100 = var8(var101, var74)
    var47 = var52 + var61
    var92 = var94 + var100
    var59 = np.maximum(0.0, var89 - var75 - np.maximum(var44, 0.0) - var49)
    var98 = np.maximum(0.0, var89 - var75 - np.maximum(var91, 0.0) - var93)
    var55 = np.minimum(var89, var75)
    var96 = np.minimum(var89, var75)
    var82 = var46['k167']
    var68 = np.zeros(k329, dtype=bool)
    for var73 in range(k329 - 1):
        if abs(var44[var73]) < 1e-06 and var57[var73] < 95.0:
            var79 = slice(var73 + 1, min(var73 + 4, k329))
            if np.any(var90[var79] - var76[var79] > 0):
                var68[var73] = True
    var43 = np.sign(np.round(var44, 1)) != np.sign(np.round(var91, 1))
    var87 = pd.DataFrame({'date': var65, 'hour': var66['hour'].to_numpy(), 'day_of_week': pd.Timestamp(var65).dayofweek, 'k126': var67, 'block': var66['k229'].to_numpy(), 'k127': var66['k155'].to_numpy(), 'k262': var66['k249'].to_numpy(), 'k363': var66['k346'].to_numpy() / 3.6, 'k327': var89, 'k341': var75, 'k160': var90, 'k238': var90 - var89, 'k442': (var90 - var89) ** 2, 'k53': var76, 'k291': var76 - var75, 'k67': (var76 - var75) ** 2, 'import_price': var74, 'export_price': var71, 'loadshedding': var78, 'k152': var84, 'k143': var44, 'k311': var88, 'k335': var57, 'k435': var57 / 100.0 * k84.capacity_wh, 'k228': var53, 'k29': var49, 'k141': var62, 'k265': var45, 'k153': var54, 'k25': var50, 'k387': var58, 'k378': var52, 'k102': float(var70.k138[0, 0]), 'k430': 0.0, 'k323': float(var69.k255 * var70.k22[0, 0] / max(var76.sum(), 1e-09)), 'k179': bool(var70.k353[0, 0]), 'k452': var82, 'k277': var68, 'k66': var91, 'k72': var97, 'k117': var97 / 100.0 * k84.capacity_wh, 'k181': var95, 'k145': var93, 'k401': var101, 'k397': var94, 'k352': var103, 'k135': var60, 'k21': var99, 'k426': var80, 'k79': var61, 'k151': var100, 'k336': var47, 'k47': var92, 'k118': var94 - var52, 'k424': var57 - var97, 'k412': var43, 'k99': var59, 'k208': var98, 'k310': var55, 'k373': var96, 'k437': np.abs(var44)})
    return (var87, float(var57[-1]), float(var47.sum()), float(var92.sum()))

def var9(var104: list[date]) -> list[list[date]]:
    var106: list[list[date]] = []
    for var105 in var104:
        if var106 and (var105 - var106[-1][-1]).days == 1:
            var106[-1].append(var105)
        else:
            var106.append([var105])
    return var106

def k37(k201: Path, k24: date, k414: date, k199: str | None, k122: 'registry.k377 | None', k312: 'registry.k377 | None', k49: 'registry.k377 | None', k161: tariff_loadshedding.k357, k84: k78, device: str, run_id: str, var109: 'set[date] | None'=None) -> k319:
    var125 = k49.version if k49 is not None else registry.k80
    var120 = registry.k88(var125)
    var130 = registry.k104(var125)
    var136 = registry.k408(var125)
    var129 = registry.k438(var125)
    var112 = k84.k292 or registry.k44 / var125 / 'charge.csv'
    var111 = var120.k172(var112, k358=k84.capacity_wh)
    var135 = registry.k101(k122, device) if k122 is not None else None
    var123 = registry.k147(k312, device) if k312 is not None else None
    var124 = registry.k73(k49, device) if k49 is not None else None
    (var126, var140, var139) = var4(k201, k24, k414, k199, var135, var123)
    var108 = sorted(var126['date'].unique())
    var137 = tariff_loadshedding.k405(var108, k161)
    var126 = var126.merge(var137, on=['date', 'hour'], how='inner')
    var121 = var126.groupby('date').size()
    var141 = var126.groupby('date')['k369'].any()
    var142 = [var113 for var113 in var108 if bool(var141.get(var113, False))]
    var138 = [var113 for var113 in var108 if var121.get(var113, 0) == k329 and (not bool(var141.get(var113, False)))]
    if var109 is not None:
        var138 = [var113 for var113 in var138 if var113 in var109]
        if not var138:
            raise ValueError('No usable days remain after applying the common date set.')
    var134 = [str(var113) for var113 in var108 if var113 not in var138 and var113 not in var142]
    if not var138:
        raise ValueError('No complete 24-hour days found in the requested date range for this site/building.')
    var133 = var9(var138)
    var117: list[pd.DataFrame] = []
    var119: list[k148] = []
    var107 = 100.0
    for (var118, var115) in enumerate(var138):
        var116 = var126[var126['date'] == var115]
        (var131, var107, var128, var110) = k187(var115, var118, var116, var107, k84, var111, var124, var120, var130, var136, var129)
        var117.append(var131)
        var119.append(k148(date=var115, hours=var131, k333=var107, k220=var128, k191=var110))
    var114 = var138
    var137 = var137[var137['date'].isin(var138)]
    var122 = pd.concat(var117, ignore_index=True)
    var122['k110'] = var122['k378'].cumsum()
    var122['k376'] = var122['k397'].cumsum()
    var122['k54'] = var122['k426'].cumsum()
    var122['k303'] = var122['k376'] - var122['k110']
    var127 = var11(run_id, k201, k199, k24, k414, var114, k122, k312, k49, var140, var139, k84, k161, var137, var112, device, var122, var134)
    var127['dataset']['k112'] = [str(var113) for var113 in var142]
    var127['dataset']['k19'] = [{'start': str(var132[0]), 'end': str(var132[-1]), 'days': len(var132)} for var132 in var133]
    var127['k282'] = var10(var119)
    return k319(run_id=run_id, days=var119, k306=var122, k205=var127)

def var10(days: list[k148]) -> dict:
    var144 = [{'date': str(var145.date), 'k191': var145.k191, 'k220': var145.k220, 'savings': var145.k191 - var145.k220} for var145 in days]
    var149: dict[str, dict] = {}
    for var151 in var144:
        var147 = var151['date'][:7]
        var143 = var149.setdefault(var147, {'month': var147, 'k191': 0.0, 'k220': 0.0, 'savings': 0.0})
        var143['k191'] += var151['k191']
        var143['k220'] += var151['k220']
        var143['savings'] += var151['savings']
    var148 = [var149[var146] for var146 in sorted(var149)]
    var150 = {'k191': sum((var151['k191'] for var151 in var144)), 'k220': sum((var151['k220'] for var151 in var144)), 'savings': sum((var151['savings'] for var151 in var144))}
    return {'daily': var144, 'monthly': var148, 'overall': var150}

def var11(run_id, k201, k199, k24, k414, var153, k122, k312, k49, var175, var174, k84, k161, var167, var152, device, k306, var162) -> dict:
    import platform
    import sys
    from datetime import datetime
    import torch as _torch

    def _accuracy_from_metadata(var154) -> dict:
        if var154 is None or var154.k205 is None:
            return {}
        var156 = ('k269', 'k254', 'k370', 'k180')
        return {var155: var154.k205[var155] for var155 in var156 if var155 in var154.k205}
    var160 = tariff_loadshedding.k227(var167)
    var168 = float(k306['k47'].sum())
    var171 = float(k306['k336'].sum())
    var173 = var168 - var171
    var161 = var173 / abs(var168) * 100.0 if var168 else float('nan')
    var169 = float(k306['k397'].sum())
    var172 = float(k306['k378'].sum())
    var170 = float(k306['k426'].sum())
    var164 = k306['k238'].to_numpy()
    var158 = k306['k291'].to_numpy()
    var163 = k306['k327'].to_numpy()
    var157 = k306['k341'].to_numpy()
    var165 = k306['k160'].to_numpy()
    var159 = k306['k53'].to_numpy()
    var166 = {'k281': var168, 'k33': var171, 'k275': var173, 'k217': var161, 'k131': var169, 'k221': var172, 'k57': var170, 'k273': float(k306['k79'].sum()), 'k301': float(k306['k151'].sum()), 'k355': float(k306['k437'].sum() / (2 * k84.capacity_wh)), 'k331': float(k306['k99'].sum()), 'k189': float(k306['k208'].sum()), 'k206': float(k306['k310'].sum() / max(k306['k327'].sum(), 1e-09) * 100.0), 'k62': float(k306['k373'].sum() / max(k306['k327'].sum(), 1e-09) * 100.0), 'k149': int((k306['k141'] > 0).sum()), 'k411': int((k306['k401'] > 0).sum()), 'k400': int(k306['k452'].sum()), 'k284': var1(var164), 'k177': var2(var164), 'k351': var3(var163, var165), 'k146': var1(var158), 'k163': var2(var158), 'k245': var3(var157, var159)}
    if k84.k447:
        var166['k2'] = k84.k447 / max(var168, 1e-09) / 365.0 * len(var153)
        var166['k368'] = k84.k447 / max(var171, 1e-09) / 365.0 * len(var153)
    return {'run_id': run_id, 'timestamp': datetime.now().isoformat(), 'k258': k161.seed, 'dataset': {'k201': str(k201), 'k199': k199, 'k24': str(k24), 'k414': str(k414), 'k393': len(var153), 'k74': len(var153) * k329, 'k83': var162}, 'checkpoints': {'solar': {'version': k122.version, 'k276': str(k122.k276), **_accuracy_from_metadata(k122)} if k122 is not None else None, 'load': {'version': k312.version, 'k276': str(k312.k276), **_accuracy_from_metadata(k312)} if k312 is not None else None, 'k68': var175, 'k105': var174, 'management': {'version': k49.version, 'k276': str(k49.k276), **_accuracy_from_metadata(k49)} if k49 is not None else None, 'k302': k49 is None}, 'k300': {'capacity_wh': k84.capacity_wh, 'shutoff_pct': k84.shutoff_pct, 'max_discharge_w': k84.max_discharge_w, 'k186': k84.k186, 'k297': k84.k297, 'k292': str(var152)}, 'k330': {'schedule': k161.schedule, 'k247': k161.k247, 'k250': k161.k250, 'k264': k161.k264, 'k446': k161.k446, 'k169': k161.k169, 'k194': k161.k194, 'k242': tariff_loadshedding.k39(list(var153), k161), 'bands': tariff_loadshedding.k182(var167)}, 'k23': var160, 'environment': {'python_version': sys.version.split()[0], 'torch_version': _torch.__version__, 'device': device, 'platform': platform.platform()}, 'summary': var166}
