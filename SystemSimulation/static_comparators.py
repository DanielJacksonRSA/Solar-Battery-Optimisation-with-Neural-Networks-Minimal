# Copyright (c) 2026 Daniel Jackson and Nelson Mandela University. All rights reserved.
# This code is proprietary. It may not be used, copied, modified, distributed or
# reused in whole or in part, nor may the ideas it embodies be used, without the
# prior written permission of the copyright holders. See LICENSE.
import sys
from pathlib import Path
import numpy as np
import pandas as pd
k162 = Path(__file__).resolve().parent
k450 = k162.parent
sys.path.insert(0, str(k450 / 'ManagementModelV2'))
import environment as env
import heuristic
from synthetic_data import k12
(CAPACITY_WH, SHUTOFF_PCT, MAX_DISCHARGE_W, RTE) = (10000.0, 20.0, 10000.0, 0.92)

def k36(var2, var4, var1, var3):
    return k12(solar_wh=var2[var4].to_numpy(float), load_wh=var2[var1].to_numpy(float), import_price=var2.import_price.to_numpy(float), export_price=var2.export_price.to_numpy(float), loadshedding=var2.loadshedding.to_numpy(float), soc0_pct=var3, capacity_wh=CAPACITY_WH, shutoff_pct=SHUTOFF_PCT, max_discharge_w=MAX_DISCHARGE_W)

def main(var25: str) -> None:
    var16 = next((k162 / 'Data' / 'runs' / 'k175').glob(f'*{var25}.csv'))
    var9 = pd.read_csv(var16, comment='#')
    var7 = env.k172(k450 / 'ManagementModelV2' / 'charge.csv', k358=CAPACITY_WH)
    var17 = var9.import_price.max()
    print(f'run: {var16.name}')
    print(f'simulator static effective cost {var9.k47.sum():.2f}, AI effective cost {var9.k336.sum():.2f}')
    for var6 in ['k418', 'backup', 'selfcons', 'heuristic']:
        var23 = 100.0
        var26 = dict(effective_cost=0.0, grid_cost=0.0, import_kwh=0.0, export_kwh=0.0, peak_import_kwh=0.0, unmet_kwh=0.0, shed_unmet_kwh=0.0, shed_load_kwh=0.0, throughput_kwh=0.0)
        for (_, var19) in var9.groupby('date', sort=True):
            var19 = var19.sort_values('hour')
            var24 = 100.0 if var6 == 'k418' else var23
            var5 = env.k328([k36(var19, 'k327', 'k341', var24)], RTE)
            if var6.startswith('backup'):
                var11 = env.k448(var5)
            elif var6 == 'selfcons':
                var11 = env.k166(var5, var7, RTE)
            else:
                var12 = env.k328([k36(var19, 'k160', 'k53', var24)], RTE)
                var11 = heuristic.heuristic_action_fn(var12, var7, RTE)
            var18 = env.k404(var11, var5, 1, var7, RTE)
            (var14, var10, var27) = (var18.k129[0, 0], var18.k176[0, 0], var18.k30[0, 0])
            (var15, var28) = (var19.import_price.to_numpy(), var19.export_price.to_numpy())
            var22 = var19.loadshedding.to_numpy() > 0.5
            var13 = (var14 * var15 - var10 * var28).sum() / 1000
            var26['grid_cost'] += var13
            var26['effective_cost'] += var13 + (var27 * var15).sum() / 1000
            var26['import_kwh'] += var14.sum() / 1000
            var26['export_kwh'] += var10.sum() / 1000
            var26['peak_import_kwh'] += var14[var15 >= var17 - 1e-09].sum() / 1000
            var26['unmet_kwh'] += var27.sum() / 1000
            var26['shed_unmet_kwh'] += var27[var22].sum() / 1000
            var26['shed_load_kwh'] += var19.k341.to_numpy()[var22].sum() / 1000
            var26['throughput_kwh'] += np.abs(var18.k298[0, 0]).sum() / 1000
            var23 = float(var18.k309[0, 0, -1])
        var8 = var26['throughput_kwh'] / (2 * CAPACITY_WH / 1000)
        var21 = 100 * (1 - var26['shed_unmet_kwh'] / var26['shed_load_kwh']) if var26['shed_load_kwh'] else float('nan')
        var20 = 100 * (1 - var26['effective_cost'] / var9.k47.sum())
        print(f"{var6:13s} effective R{var26['effective_cost']:9.2f}  saving vs baseline {var20:5.1f}%  import {var26['import_kwh']:7.1f} kWh  export {var26['export_kwh']:6.1f}  peak import {var26['peak_import_kwh']:6.1f}  unmet {var26['unmet_kwh']:6.1f}  loadshedding demand served {var21:5.1f}%  cycles {var8:5.1f}")
if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'solar-v2_load-v1_mgmt-v2_shed-mixed')
