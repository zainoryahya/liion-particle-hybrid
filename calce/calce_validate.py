"""25 C validation: parameters fitted on DST 80% -> FUDS/US06/BJDST (80% & 50%) and DST 50%.
Also: with the same parameters, compare exact / FV N=5,10,20 / hybrid N=10 particle solvers on real data."""
import os; ROOT = os.environ.get('LIION_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
import numpy as np, json, sys
sys.path.insert(0, ROOT + '/calce')
import calce_model as M

TEMP = sys.argv[1] if len(sys.argv) > 1 else '25C'
Tk = {'0C': 273.15, '25C': 298.15, '45C': 318.15}[TEMP]
FIT = json.load(open(sys.argv[2] if len(sys.argv) > 2 else f'{ROOT}/calce/fit_{TEMP}_hybrid10.json'))
x = np.array(FIT['x'])
Dz = np.load(ROOT + '/calce/dyn/dyn.npz')
keys = [k for k in Dz.files if k.startswith(TEMP + '_')]
OUT = {'fit': FIT, 'tests': {}}
for key in sorted(keys):
    d = M.load_test(key, Dz)
    m = d['mask']; row = {}
    Vex = M.simulate(x, d, 'exact', T=Tk)
    for kind, N in [('exact', 0), ('hybrid', 10), ('fv', 5), ('fv', 10), ('fv', 20)]:
        V = Vex if kind == 'exact' else M.simulate(x, d, kind, N, T=Tk)
        r = (V[d['gi']] - d['Vd'])[m]
        dv = (V - Vex)[d['gi']][m]
        lab = kind if kind == 'exact' else f'{kind}{N}'
        row[lab] = dict(rmse_mV=float(np.sqrt(np.mean(r ** 2)) * 1e3), p95_mV=float(np.percentile(np.abs(r), 95) * 1e3),
                        vs_exact_max_mV=float(np.abs(dv).max() * 1e3), vs_exact_rms_mV=float(np.sqrt(np.mean(dv ** 2)) * 1e3))
        if lab in ('hybrid10', 'fv10'):
            row[lab]['trace'] = (V[d['gi']][m]).tolist()[::1]
    row['t'] = d['td'][m].tolist(); row['V'] = d['Vd'][m].tolist(); row['I'] = d['Id'][m].tolist()
    OUT['tests'][key] = row
    print(f"{key:14s} RMSE data: exact {row['exact']['rmse_mV']:5.2f}  hybrid10 {row['hybrid10']['rmse_mV']:5.2f}  FV5 {row['fv5']['rmse_mV']:5.2f}  "
          f"FV10 {row['fv10']['rmse_mV']:5.2f}  FV20 {row['fv20']['rmse_mV']:5.2f} mV | max diff vs exact: hybrid10 {row['hybrid10']['vs_exact_max_mV']:.3f}  "
          f"FV10 {row['fv10']['vs_exact_max_mV']:.1f}  FV20 {row['fv20']['vs_exact_max_mV']:.1f} mV", flush=True)
json.dump(OUT, open(f'{ROOT}/calce/validate_{TEMP}.json', 'w'))
print('done')
