# Copyright (c) 2026 Daniel Jackson and Nelson Mandela University. All rights reserved.
# This code is proprietary. It may not be used, copied, modified, distributed or
# reused in whole or in part, nor may the ideas it embodies be used, without the
# prior written permission of the copyright holders. See LICENSE.
from __future__ import annotations
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import feature_adapter
import registry
import run_logger
import simulation
import tariff_loadshedding
app = FastAPI(title='Solar/Battery System Simulation')
k18 = Path(__file__).resolve().parent / 'static'
k97 = registry.k44 / 'Data'
var1: dict[str, simulation.k319] = {}
var2: Optional[str] = None

def var3() -> str:
    import torch
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

@app.get('/api/checkpoints')
def k77() -> dict:
    var5 = registry.k50()
    return {var8: [{'version': var6.version, 'k276': str(var6.k276), 'label': var6.label, 'k205': var6.k205} for var6 in var7] for (var8, var7) in var5.items()}

@app.get('/api/tariff')
def k8() -> dict:
    var10 = tariff_loadshedding.k357()
    var14 = 'blended' if var10.k451 else 'seasonal'

    def rates_for(var15: str) -> list[dict]:
        return [{'band': var9, 'k20': tariff_loadshedding.k32(tariff_loadshedding.k322[var15, var9], var10.k247), 'k198': tariff_loadshedding.k32(tariff_loadshedding.k9[var15, var9], var10.k247)} for var9 in tariff_loadshedding.k71]
    var12 = {'peak': sorted(tariff_loadshedding.k90), 'standard': sorted(tariff_loadshedding.k391), 'off_peak': [var11 for var11 in range(24) if var11 not in tariff_loadshedding.k90 and var11 not in tariff_loadshedding.k391]}
    return {'schedule': var10.schedule, 'k334': 'Residential time-of-use tariff', 'k247': var10.k247, 'k236': tariff_loadshedding.k271, 'k82': var14, 'k250': var10.k250, 'k294': tariff_loadshedding.k70 * (1.0 + tariff_loadshedding.k271 if var10.k247 else 1.0), 'k158': var12, 'rates': {var13: rates_for(var13) for var13 in ('blended', 'winter', 'summer')}}

@app.get('/api/datasets')
def k439() -> list[dict]:
    if not k97.exists():
        return []
    var18 = []
    for var21 in sorted(k97.glob('*')):
        if not var21.is_dir():
            continue
        try:
            var17 = feature_adapter.k188(var21)
            var20 = []
            for var16 in var17:
                (var22, var19) = feature_adapter.k278(var21, var16)
                var20.append({'k199': var16, 'k24': str(var22), 'k414': str(var19)})
            var18.append({'k201': str(var21), 'name': var21.name, 'buildings': var20})
        except ValueError:
            continue
    return var18

class k139(BaseModel):
    k201: str
    k199: Optional[str] = None
    k24: date
    k414: date
    k409: Optional[str] = None
    k243: Optional[str] = None
    k210: Optional[str] = None
    k445: Optional[str] = None
    k431: Optional[str] = None
    k28: Optional[str] = None
    capacity_wh: float = 10000.0
    shutoff_pct: float = 20.0
    max_discharge_w: float = 10000.0
    k186: float = 0.92
    k297: float = 8.8
    k447: Optional[float] = None
    k248: str = tariff_loadshedding.k295
    k247: bool = True
    k250: bool = True
    k264: float = 0.2
    k446: float = 2.0
    k169: float = 0.6
    k194: float = 0.6
    k229: Optional[int] = None
    k258: int = 42

def var4(var24: str, version: str, k276: str) -> registry.k377:
    var23 = registry.k50()
    for (var27, var26) in var23.items():
        if not var27.startswith(var24):
            continue
        for var25 in var26:
            if var25.version == version and str(var25.k276) == k276:
                return var25
    raise HTTPException(status_code=404, detail=f'Checkpoint not found: {version}/{k276}')

@app.post('/api/simulate')
def k360(req: k139) -> dict:
    var34 = var4('solar_', req.k409, req.k243) if req.k409 and req.k243 else None
    var30 = var4('load_', req.k210, req.k445) if req.k210 and req.k445 else None
    var31 = var4('management', req.k431, req.k28) if req.k431 and req.k28 else None
    var28 = simulation.k78(capacity_wh=req.capacity_wh, shutoff_pct=req.shutoff_pct, max_discharge_w=req.max_discharge_w, k186=req.k186, k297=req.k297, k447=req.k447)
    var35 = tariff_loadshedding.k357(schedule=req.k248, k247=req.k247, k250=req.k250, k264=req.k264, k446=req.k446, k169=req.k169, k194=req.k194, k229=req.k229, seed=req.k258)
    var33 = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    try:
        var32 = simulation.k37(k201=Path(req.k201), k24=req.k24, k414=req.k414, k199=req.k199, k122=var34, k312=var30, k49=var31, k161=var35, k84=var28, device=var3(), run_id=var33)
    except ValueError as var29:
        raise HTTPException(status_code=400, detail=str(var29))
    global var2
    var1[var33] = var32
    run_logger.k96(var32)
    var2 = var33
    return {'run_id': var33}

@app.get('/api/current')
def k421() -> dict:
    return {'run_id': var2}

@app.get('/api/results/{run_id}')
def k184(run_id: str) -> dict:
    var37 = var1.get(run_id)
    if var37 is None:
        raise HTTPException(status_code=404, detail=f'Unknown run_id: {run_id}')
    var36 = var37.k306.copy()
    var36['date'] = var36['date'].astype(str)
    return {'run_id': run_id, 'k205': var37.k205, 'k306': var36.to_dict(orient='records')}

@app.get('/api/runs/{run_id}/csv')
def k76(run_id: str):
    var38 = run_logger.k165 / f'{run_id}.csv'
    if not var38.exists():
        raise HTTPException(status_code=404, detail=f'No CSV found for run_id: {run_id}')
    return FileResponse(var38, media_type='text/csv', filename=var38.name)

class k159(StaticFiles):

    def file_response(self, *var39, **var40):
        var41 = super().file_response(*var39, **var40)
        var41.headers['Cache-Control'] = 'no-store'
        return var41
app.mount('/', k159(directory=str(k18), html=True), name='static')
