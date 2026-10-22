"""
Twin experiment for identification bias due to discretisation, with KNOWN TRUTH.
Synthetic data: EXACT particle model + 'true' parameters (= CALCE 25 C hybrid fit) + real DST 80% current profile + 2 mV noise.
Fit with hybrid N=10, FV N=20, FV N=10 (two starting points each). Report parameter ratios vs truth.
"""
import os; ROOT = os.environ.get('LIION_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
import numpy as np, json, sys, time
from scipy.optimize import least_squares
sys.path.insert(0, ROOT + '/calce')
import calce_model as M

BND = [(1, 5), (0.5, 5), (-2, 2), (-2, 2), (-3, -0.7), (-4, -0.7), (0, 3.5), (0.85, 1.1), (0.9, 1.1)]
lo, hi = np.array([b[0] for b in BND]), np.array([b[1] for b in BND])
xt = np.array(json.load(open(ROOT + '/calce/fit_25C_hybrid10.json'))['x'])
xt[5] = np.log10(5e-3); xt[6] = np.log10(20.0)          # true RC is non-zero, for a fairer test
rng = np.random.default_rng(42)
OUT = {'truth': M.unpack(xt), 'fits': {}}
d = M.load_test('25C_DST_80'); m = d['mask']
t0 = time.time()
Vtrue = M.simulate(xt, d, 'exact')
print(f'synthetic data generated ({time.time()-t0:.0f} s)', flush=True)
Vmeas = Vtrue[d['gi']] + rng.normal(0, 2e-3, len(d['gi']))
dv = M.load_test('25C_FUDS_80'); Vval = M.simulate(xt, dv, 'exact')[dv['gi']]

starts = [xt + np.array([0.3, -0.3, 0.3, -0.3, 0.2, 0.3, 0.3, 0.02, 0.01]),
          xt + np.array([-0.3, 0.3, -0.3, 0.3, -0.2, -0.3, -0.3, -0.02, -0.01])]
for kind, N in [('hybrid', 10), ('fv', 20), ('fv', 10), ('fv', 5)]:
    best = None
    for s0 in starts:
        res = lambda x: (M.simulate(x, d, kind, N)[d['gi']] - Vmeas)[m]
        r = least_squares(res, np.clip(s0, lo + 1e-6, hi - 1e-6), bounds=(lo, hi), x_scale='jac', max_nfev=300)
        if best is None or r.cost < best.cost:
            best = r
    x = best.x; p = M.unpack(x)
    rms = float(np.sqrt(np.mean(best.fun ** 2)) * 1e3)
    val = float(np.sqrt(np.mean(((M.simulate(x, dv, kind, N)[dv['gi']] - Vval)[dv['mask']]) ** 2)) * 1e3)
    ratio = {k: p[k] / OUT['truth'][k] for k in ['tau_n', 'tau_p', 'I0_n', 'I0_p', 'R0', 'R1']}
    lab = f'{kind} N={N}'
    OUT['fits'][lab] = dict(p=p, rmse_fit_mV=rms, rmse_FUDS_vs_truth_mV=val, ratio=ratio)
    print(f"{lab:10s}: fit {rms:5.2f} mV (noise 2.00)  FUDS prediction vs truth {val:5.2f} mV | ratio vs true: "
          + '  '.join(f'{k} {v:.2f}' for k, v in ratio.items()) + f'  ({time.time()-t0:.0f} s)', flush=True)
json.dump(OUT, open(ROOT + '/calce/twin_bias.json', 'w'))
print('done')
