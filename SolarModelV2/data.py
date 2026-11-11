# Copyright (c) 2026 Daniel Jackson and Nelson Mandela University. All rights reserved.
# This code is proprietary. It may not be used, copied, modified, distributed or
# reused in whole or in part, nor may the ideas it embodies be used, without the
# prior written permission of the copyright holders. See LICENSE.
from __future__ import annotations
import numpy as np
import pandas as pd
k130 = ['k155', 'k249', 'k115', 'k396', 'k361', 'k346', 'k171', 'k43', 'k362', 'k211', 'k51', 'k136']
k257 = ['time', 'k407', 'k171', 'k43']

def k93(var3: pd.DataFrame) -> pd.DataFrame:
    var2 = var3.copy()
    if not pd.api.types.is_datetime64_any_dtype(var2['time']):
        var5 = pd.to_datetime(var2['time'], format='%Y-%m-%dT%H:%M:%S', errors='coerce')
    else:
        var5 = var2['time']
    var4 = var5.dt.hour + var5.dt.minute / 60.0 + var5.dt.second / 3600.0
    var1 = var5.dt.dayofyear.astype(float)
    var2['k362'] = np.sin(2.0 * np.pi * var4 / 24.0)
    var2['k211'] = np.cos(2.0 * np.pi * var4 / 24.0)
    var2['k51'] = np.sin(2.0 * np.pi * var1 / 366.0)
    var2['k136'] = np.cos(2.0 * np.pi * var1 / 366.0)
    return var2
