"""Sweep: FV N=10 bias vs the dimensionless number Pi = I_max tau_n / (3*3600*Q_n) (dimensionless peak surface flux).
Truth = Samsung parameters with tau_n, tau_p multiplied by a factor f; data = exact model + 2 mV noise; UDDS+US06 profile."""
import numpy as np, json, time
from scipy.optimize import least_squares
import ss_model as S
BND = [(0.5, 5.5), (0.5, 5.5), (-2, 2.5), (-2, 2.5), (-3, -0.7), (-4, -0.7), (-1, 3.5), (0.85, 1.1), (0.9, 1.1)]
lo, hi = np.array([b[0] for b in BND]), np.array([b[1] for b in BND])
x0 = np.array(json.load(open('fit_UDDS+US06_hybrid10.json'))['x'])
DS = [S.load_test(k) for k in ['UDDS', 'US06']]
Imax = max(np.abs(d['I']).max() for d in DS)
rng = np.random.default_rng(3); OUT = []; t0 = time.time()
for f in [0.25, 1, 4, 16, 36]:
    xt = x0.copy(); xt[0] += np.log10(f); xt[1] += np.log10(f)
    p = S.unpack(xt); Qn = p['c'] * S.M.Q20 / (S.M.W['n100'] - S.M.W['n0'])
    Pi = Imax * p['tau_n'] / (3 * 3600 * Qn)
    Ym = [S.simulate(xt, d, 'exact')[d['gi']] + rng.normal(0, 2e-3, len(d['gi'])) for d in DS]
    row = dict(f=f, Pi=float(Pi), tau_n=p['tau_n'])
    for kind, N in [('fv', 10), ('fv', 20), ('hybrid', 10)]:
        res = lambda x: np.concatenate([(S.simulate(x, d, kind, N)[d['gi']] - y)[::3] for d, y in zip(DS, Ym)])
        r = least_squares(res, np.clip(xt + np.array([0.15, -0.15, 0.2, -0.2, 0.1, 0.2, 0.2, 0.01, 0.005]), lo + 1e-6, hi - 1e-6), bounds=(lo, hi), x_scale='jac', max_nfev=200)
        q = S.unpack(r.x)
        row[f'{kind}{N}'] = dict(tau_n_ratio=q['tau_n'] / p['tau_n'], tau_p_ratio=q['tau_p'] / p['tau_p'], rmse=float(np.sqrt(np.mean(r.fun ** 2)) * 1e3))
    OUT.append(row)
    print(f"f={f:5.2f} Pi={Pi:6.2f} tau_n={p['tau_n']:7.0f}s | FV10 tau_n x{row['fv10']['tau_n_ratio']:.2f} tau_p x{row['fv10']['tau_p_ratio']:.2f} ({row['fv10']['rmse']:.1f} mV) | "
          f"FV20 x{row['fv20']['tau_n_ratio']:.2f}/{row['fv20']['tau_p_ratio']:.2f} ({row['fv20']['rmse']:.1f}) | hyb x{row['hybrid10']['tau_n_ratio']:.2f}/{row['hybrid10']['tau_p_ratio']:.2f} ({row['hybrid10']['rmse']:.1f})  [{time.time()-t0:.0f} s]", flush=True)
    json.dump(OUT, open('sweep_ss.json', 'w'))
