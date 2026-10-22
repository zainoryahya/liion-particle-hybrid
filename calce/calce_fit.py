"""Fit 8 parameters on DST (initial SOC 80%) for one temperature; global (DE) + local (TRF). Particle solver: hybrid N=10."""
import os; ROOT = os.environ.get('LIION_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
import numpy as np, json, sys, time
from scipy.optimize import differential_evolution, least_squares
sys.path.insert(0, ROOT + '/calce')
import calce_model as M

TEMP = sys.argv[1] if len(sys.argv) > 1 else '25C'
KIND = sys.argv[2] if len(sys.argv) > 2 else 'hybrid'
NN = int(sys.argv[3]) if len(sys.argv) > 3 else 10
Tk = {'0C': 273.15, '25C': 298.15, '45C': 318.15}[TEMP]
import os
d = M.load_test(f'{TEMP}_DST_80', dt=float(os.environ.get('CALCE_DT', '0.2')))
msk = d['mask']
BND = [(1, 5), (0.5, 5), (-2, 2), (-2, 2), (-3, -0.7), (-4, -0.7), (0, 3.5), (0.85, 1.1), (0.9, 1.1)]


def res(x):
    V = M.simulate(x, d, KIND, NN, T=Tk)
    r = (V[d['gi']] - d['Vd'])[msk]
    return np.nan_to_num(r, nan=1.0)


def cost(x):
    return float(np.mean(res(x) ** 2))


t0 = time.time()
if len(sys.argv) > 5:                     # fix c and soc0 from another fit (e.g. 25 C)
    xr = json.load(open(sys.argv[5]))['x']
    BND[7] = (xr[7] - 1e-9, xr[7] + 1e-9); BND[8] = (xr[8] - 1e-9, xr[8] + 1e-9)
if len(sys.argv) > 4:                     # starting point given -> local only
    x0 = np.clip(np.array(json.load(open(sys.argv[4]))['x']), [b[0] for b in BND], [b[1] for b in BND])
else:
    de = differential_evolution(cost, BND, popsize=int(os.environ.get('DE_POP', '12')), maxiter=int(os.environ.get('DE_IT', '60')), tol=1e-8, seed=1, polish=False, init='sobol')
    x0 = de.x
    print(f'DE: RMSE {np.sqrt(de.fun)*1e3:.2f} mV  ({time.time()-t0:.0f} s)', flush=True)
lsq = least_squares(res, x0, bounds=([b[0] for b in BND], [b[1] for b in BND]), x_scale='jac', max_nfev=400)
x = lsq.x; r = res(x)
J = lsq.jac; s2 = np.mean(r ** 2)
try:
    cov = np.linalg.inv(J.T @ J) * s2; sd = np.sqrt(np.diag(cov)); corr = cov / np.outer(sd, sd)
except np.linalg.LinAlgError:
    sd = np.full(len(x), np.nan); corr = np.full((len(x), len(x)), np.nan)
p = M.unpack(x)
print(f'{TEMP} {KIND} N={NN}: RMSE {np.sqrt(s2)*1e3:.2f} mV  max {np.abs(r).max()*1e3:.1f} mV  ({time.time()-t0:.0f} s)')
for n, v, s_ in zip(M.NAMES, x, sd):
    print(f'   {n:6s} = {p[n]:.4g}   (sd log/lin {s_:.3f})')
bad = [(M.NAMES[i], M.NAMES[j], corr[i, j]) for i in range(len(x)) for j in range(i) if abs(corr[i, j]) > 0.9]
print('   correlation > 0.9:', [(a, b, round(c, 3)) for a, b, c in bad])
json.dump(dict(x=x.tolist(), p=p, rmse_mV=float(np.sqrt(s2) * 1e3), sd=sd.tolist(), corr=corr.tolist(), kind=KIND, N=NN, temp=TEMP),
          open(f'{ROOT}/calce/fit_{TEMP}_{KIND}{NN}.json', 'w'))
