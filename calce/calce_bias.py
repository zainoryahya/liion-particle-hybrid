"""Identification bias due to discretisation error: model fitted with FV N=10 / N=20 / hybrid N=10 (each with its own solver),
then validated on all 25 C profiles. Also: prediction of each model against the exact-solver 'truth'."""
import os; ROOT = os.environ.get('LIION_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
import numpy as np, json, sys
sys.path.insert(0, ROOT + '/calce')
import calce_model as M
Dz = np.load(ROOT + '/calce/dyn/dyn.npz')
fits = {'hybrid N=10': ('hybrid', 10, 'fit_25C_hybrid10.json'), 'FV N=20': ('fv', 20, 'fit_25C_fv20.json'), 'FV N=10': ('fv', 10, 'fit_25C_fv10.json')}
OUT = {}
for lab, (kind, N, f) in fits.items():
    F = json.load(open(f)); x = np.array(F['x']); row = {'p': F['p']}
    for key in sorted(k for k in Dz.files if k.startswith('25C_')):
        d = M.load_test(key, Dz); m = d['mask']
        V = M.simulate(x, d, kind, N)
        row[key] = float(np.sqrt(np.mean(((V[d['gi']] - d['Vd'])[m]) ** 2)) * 1e3)
    OUT[lab] = row
    print(f"{lab:12s} tau_n {F['p']['tau_n']:7.0f} s  tau_p {F['p']['tau_p']:7.0f} s  R0 {F['p']['R0']*1e3:5.1f} mOhm  R1 {F['p']['R1']*1e3:5.2f} mOhm  c {F['p']['c']:.3f} | RMSE: " +
          '  '.join(f"{k[4:]} {v:5.1f}" for k, v in row.items() if k != 'p'), flush=True)
json.dump(OUT, open(ROOT + '/calce/bias_25C.json', 'w'))
