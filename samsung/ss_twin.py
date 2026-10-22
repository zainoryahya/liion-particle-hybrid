"""Samsung 30T twin experiment: truth = joint hybrid fit (UDDS+US06), data = EXACT model + 2 mV noise."""
import numpy as np, json, sys, time
from scipy.optimize import least_squares
import ss_model as S
BND = [(1, 5), (0.5, 5), (-2, 2.5), (-2, 2.5), (-3, -0.7), (-4, -0.7), (-1, 3.5), (0.85, 1.1), (0.9, 1.1)]
lo, hi = np.array([b[0] for b in BND]), np.array([b[1] for b in BND])
xt = np.array(json.load(open('fit_UDDS+US06_hybrid10.json'))['x'])
rng = np.random.default_rng(7); t0 = time.time()
DS = [S.load_test(k) for k in ['UDDS', 'US06']]
Ym = [S.simulate(xt, d, 'exact')[d['gi']] + rng.normal(0, 2e-3, len(d['gi'])) for d in DS]
dv = S.load_test('LA92'); Yv = S.simulate(xt, dv, 'exact')[dv['gi']]
truth = S.unpack(xt); OUT = {'truth': truth, 'fits': {}}
print(f'data generated ({time.time()-t0:.0f} s)', flush=True)
starts = [xt + np.array([0.3, -0.3, 0.3, -0.3, 0.2, 0.3, 0.3, 0.02, 0.01]), xt + np.array([-0.3, 0.3, -0.3, 0.3, -0.2, -0.3, -0.3, -0.02, -0.01])]
for kind, N in [('hybrid', 10), ('fv', 20), ('fv', 10)]:
    res = lambda x: np.concatenate([(S.simulate(x, d, kind, N)[d['gi']] - y)[::2] for d, y in zip(DS, Ym)])
    best = None
    for s0 in starts:
        r = least_squares(res, np.clip(s0, lo + 1e-6, hi - 1e-6), bounds=(lo, hi), x_scale='jac', max_nfev=300)
        if best is None or r.cost < best.cost: best = r
    p = S.unpack(best.x)
    val = float(np.sqrt(np.mean((S.simulate(best.x, dv, kind, N)[dv['gi']] - Yv) ** 2)) * 1e3)
    ratio = {k: p[k] / truth[k] for k in ['tau_n', 'tau_p', 'I0_n', 'I0_p', 'R0', 'R1']}
    lab = f'{kind} N={N}'
    OUT['fits'][lab] = dict(p=p, rmse_fit_mV=float(np.sqrt(np.mean(best.fun ** 2)) * 1e3), rmse_LA92_vs_truth_mV=val, ratio=ratio)
    print(f"{lab:10s}: fit {OUT['fits'][lab]['rmse_fit_mV']:5.2f} mV  LA92 vs truth {val:6.2f} mV | " + '  '.join(f'{k} {v:.2f}' for k, v in ratio.items()) + f' ({time.time()-t0:.0f} s)', flush=True)
json.dump(OUT, open('twin_ss.json', 'w'))
