"""Fit Samsung 30T parameters (25 C) on one cycle; local multi-start (+ optional DE)."""
import numpy as np, json, sys, time, os
from scipy.optimize import least_squares, differential_evolution
import ss_model as S
KEY = sys.argv[1] if len(sys.argv) > 1 else 'UDDS'
KIND = sys.argv[2] if len(sys.argv) > 2 else 'hybrid'
NN = int(sys.argv[3]) if len(sys.argv) > 3 else 10
TEMPF = os.environ.get('TEMPF', '25degC')
DS = [S.load_test(k, TEMPF) for k in KEY.split('+')]
BND = [(1, 5), (0.5, 5), (-2, 2.5), (-2, 2.5), (-3, -0.7), (-4, -0.7), (-1, 3.5), (0.85, 1.1), (0.9, 1.1)]
lo, hi = np.array([b[0] for b in BND]), np.array([b[1] for b in BND])
def res(x):
    return np.concatenate([np.nan_to_num(S.simulate(x, d, KIND, NN)[d['gi']] - d['Vd'], nan=1.0)[::2] for d in DS])
t0 = time.time()
starts = [np.array([3.0, 3.0, 1.0, 1.0, np.log10(0.02), np.log10(0.005), 1.0, 1.0, 1.0]),
          np.array([2.0, 2.5, 0.5, 1.5, np.log10(0.015), np.log10(0.01), 1.5, 0.95, 1.0]),
          np.array([3.5, 2.0, 1.5, 0.5, np.log10(0.025), np.log10(0.002), 0.5, 1.02, 1.0])]
if len(sys.argv) > 4:
    starts = [np.array(json.load(open(sys.argv[4]))['x'])]
if os.environ.get('DE'):
    de = differential_evolution(lambda x: float(np.mean(res(x) ** 2)), BND, popsize=8, maxiter=40, seed=1, polish=False, init='sobol', tol=1e-8)
    starts = [de.x] + starts[:1]
    print(f'DE {np.sqrt(de.fun)*1e3:.2f} mV ({time.time()-t0:.0f} s)', flush=True)
best = None
for s in starts:
    r = least_squares(res, np.clip(s, lo + 1e-6, hi - 1e-6), bounds=(lo, hi), x_scale='jac', max_nfev=300)
    print(f'  start -> RMSE {np.sqrt(np.mean(r.fun**2))*1e3:.2f} mV', flush=True)
    if best is None or r.cost < best.cost: best = r
x = best.x; p = S.unpack(x); rm = float(np.sqrt(np.mean(best.fun ** 2)) * 1e3)
J = best.jac
try:
    cov = np.linalg.inv(J.T @ J) * np.mean(best.fun ** 2); sd = np.sqrt(np.diag(cov)); corr = cov / np.outer(sd, sd)
except Exception:
    sd = np.full(len(x), np.nan); corr = np.full((len(x), len(x)), np.nan)
print(f'{KEY} {KIND} N={NN}: RMSE {rm:.2f} mV ({time.time()-t0:.0f} s)')
for n in S.NAMES: print(f'   {n:6s} = {p[n]:.4g}')
print('   correlation > 0.9:', [(S.NAMES[i], S.NAMES[j], round(corr[i, j], 3)) for i in range(len(x)) for j in range(i) if abs(corr[i, j]) > 0.9])
json.dump(dict(x=x.tolist(), p=p, rmse_mV=rm, corr=corr.tolist(), kind=KIND, N=NN, key=KEY), open(f'fit_{KEY}_{KIND}{NN}' + ('' if TEMPF == '25degC' else '_' + TEMPF) + '.json', 'w'))
