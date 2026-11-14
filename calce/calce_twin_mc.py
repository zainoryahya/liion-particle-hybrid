"""
Twin experiment repeated over independent noise realizations (same truth and current as calce_twin.py).
For each realization: synthetic DST 80% data = exact particle model + 2 mV Gaussian noise (seed 100 + i); every particle
solver is fitted from the same two starting points as in calce_twin.py. Reports mean and range of the identified/true
ratios and of the fit RMSE. Writes calce/twin_mc.json. Environment: TWIN_MC_N (number of realizations, default 10).
"""
import os; ROOT = os.environ.get('LIION_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
import numpy as np, json, sys, time
from scipy.optimize import least_squares
sys.path.insert(0, ROOT + '/calce')
import calce_model as M

BND = [(1, 5), (0.5, 5), (-2, 2), (-2, 2), (-3, -0.7), (-4, -0.7), (0, 3.5), (0.85, 1.1), (0.9, 1.1)]
lo, hi = np.array([b[0] for b in BND]), np.array([b[1] for b in BND])
xt = np.array(json.load(open(ROOT + '/calce/fit_25C_hybrid10.json'))['x'])
xt[5] = np.log10(5e-3); xt[6] = np.log10(20.0)
truth = M.unpack(xt)
d = M.load_test('25C_DST_80'); m = d['mask']
Vtrue = M.simulate(xt, d, 'exact')
starts = [xt + np.array([0.3, -0.3, 0.3, -0.3, 0.2, 0.3, 0.3, 0.02, 0.01]),
          xt + np.array([-0.3, 0.3, -0.3, 0.3, -0.2, -0.3, -0.3, -0.02, -0.01])]
KINDS = [('hybrid', 10), ('fv', 20), ('fv', 10), ('twocell', 20), ('twocell', 10)]
KEYS = ['tau_n', 'tau_p', 'I0_n', 'I0_p', 'R0', 'R1', 'tau1']
NR = int(os.environ.get('TWIN_MC_N', '10'))
OUT = {'truth': truth, 'runs': {f'{k} N={n}': [] for k, n in KINDS}}
t0 = time.time()
for i in range(NR):
    rng = np.random.default_rng(100 + i)
    Vmeas = Vtrue[d['gi']] + rng.normal(0, 2e-3, len(d['gi']))
    for kind, N in KINDS:
        best = None
        for s0 in starts:
            res = lambda x: (M.simulate(x, d, kind, N)[d['gi']] - Vmeas)[m]
            r = least_squares(res, np.clip(s0, lo + 1e-6, hi - 1e-6), bounds=(lo, hi), x_scale='jac', max_nfev=300)
            if best is None or r.cost < best.cost:
                best = r
        p = M.unpack(best.x)
        rec = dict(seed=100 + i, rmse_fit_mV=float(np.sqrt(np.mean(best.fun ** 2)) * 1e3),
                   ratio={k: p[k] / truth[k] for k in KEYS}, at_bound=bool(np.any(np.isclose(best.x, lo) | np.isclose(best.x, hi))))
        OUT['runs'][f'{kind} N={N}'].append(rec)
        print(f"run {i} {kind:8s} N={N:2d}: fit {rec['rmse_fit_mV']:5.2f} mV  " + ' '.join(f'{k} {v:.3f}' for k, v in rec['ratio'].items())
              + f"  ({time.time() - t0:.0f} s)", flush=True)
    json.dump(OUT, open(ROOT + '/calce/twin_mc.json', 'w'))
summ = {}
for lab, runs in OUT['runs'].items():
    R = {k: np.array([r['ratio'][k] for r in runs]) for k in KEYS}
    f = np.array([r['rmse_fit_mV'] for r in runs])
    summ[lab] = dict(rmse_mean=float(f.mean()), rmse_min=float(f.min()), rmse_max=float(f.max()),
                     **{k: dict(mean=float(v.mean()), min=float(v.min()), max=float(v.max())) for k, v in R.items()})
    print(f"{lab:12s} fit {f.mean():.2f} [{f.min():.2f}, {f.max():.2f}] mV  " +
          '  '.join(f"{k} {v.mean():.2f} [{v.min():.2f}, {v.max():.2f}]" for k, v in R.items()))
OUT['summary'] = summ
json.dump(OUT, open(ROOT + '/calce/twin_mc.json', 'w'), indent=1)
print('done')
