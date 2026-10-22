"""Temperature-dependent model fit (Arrhenius + warped-time diffusion, measured cell temperature) for Samsung 30T at cold temperatures.
argv: TEMPF KIND N [KEY=UDDS+US06]
"""
import os; ROOT = os.environ.get('LIION_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
import numpy as np, json, sys, time
from scipy.optimize import least_squares
sys.path.insert(0, ROOT + '/samsung')
import ss_model as S, ss_thermal as T
TEMPF, KIND, NN = sys.argv[1], sys.argv[2], int(sys.argv[3])
KEY = sys.argv[4] if len(sys.argv) > 4 else 'UDDS+US06'
TREF = 273.15 + float(TEMPF.replace('degC', ''))
DS = [S.load_test(k, TEMPF) for k in KEY.split('+')]
# PAPER=1 (default): settings of the published fits (fitT_*): activation energies in [0, 100] kJ/mol, no entropic term.
# PAPER=0: sensitivity run (fitT2_*): wider bounds and an effective entropic coefficient dU/dT (reported in the text only).
PAPER = os.environ.get('PAPER', '1') == '1'
BND = [(1, 5.5), (0.5, 5.5), (-3, 2.5), (-3, 2.5), (-3, -0.5), (-4, -0.5), (-1, 3.5), (0.8, 1.1), (0.85, 1.1)] + \
      ([(0, 100), (0, 100), (0, 100)] if PAPER else [(0, 150), (0, 200), (0, 150), (-1.5, 1.5)])
lo, hi = np.array([b[0] for b in BND]), np.array([b[1] for b in BND])


def res(x):
    return np.concatenate([np.nan_to_num(T.simulate(x, d, KIND, NN, Tref=TREF)[d['gi']] - d['Vd'], nan=1.0)[::2] for d in DS])


x_iso = np.array(json.load(open(f'fit_UDDS+US06_hybrid10_{TEMPF}.json'))['x'])
starts = [np.r_[x_iso, 30, 50, 30],
          np.array([3.0, 3.0, 0.5, 0.5, np.log10(0.03), np.log10(0.01), 1.5, 0.95, 0.97, 30, 50, 30]),
          np.array([3.5, 2.5, 0.0, 1.0, np.log10(0.02), np.log10(0.02), 2.0, 0.9, 0.95, 50, 30, 20]),
          np.array([2.5, 3.5, 1.0, 0.0, np.log10(0.04), np.log10(0.005), 1.0, 1.0, 1.0, 20, 70, 40]),
          np.array([4.0, 3.0, 0.3, 0.3, np.log10(0.025), np.log10(0.015), 2.5, 0.92, 0.95, 40, 40, 40])]
if len(sys.argv) > 5:
    starts = []
    for fn in sys.argv[5].split(','):
        xs = np.array(json.load(open(fn))['x']); starts.append(np.r_[xs, np.zeros(max(0, len(BND) - len(xs)))][:len(BND)])
starts = [np.r_[s, np.zeros(max(0, len(BND) - len(s)))][:len(BND)] for s in starts]
t0 = time.time(); best = None
for s in starts:
    r = least_squares(res, np.clip(s, lo + 1e-6, hi - 1e-6), bounds=(lo, hi), x_scale='jac', max_nfev=400)
    print(f'  start -> RMSE {np.sqrt(np.mean(r.fun**2))*1e3:.2f} mV ({time.time()-t0:.0f} s)', flush=True)
    if best is None or r.cost < best.cost:
        best = r
x = best.x; p = T.unpack(x); rm = float(np.sqrt(np.mean(best.fun ** 2)) * 1e3)
J = best.jac
try:
    cov = np.linalg.inv(J.T @ J) * np.mean(best.fun ** 2); sd = np.sqrt(np.diag(cov)); corr = cov / np.outer(sd, sd)
except Exception:
    corr = np.full((len(x), len(x)), np.nan)
per = {}
for k, d in zip(KEY.split('+'), DS):
    e = T.simulate(x, d, KIND, NN, Tref=TREF)[d['gi']] - d['Vd']; per[k] = float(np.sqrt(np.mean(e ** 2)) * 1e3)
print(f'{TEMPF} {KEY} {KIND} N={NN} (Arrhenius, measured T): RMSE {rm:.2f} mV  per cycle {per}  ({time.time()-t0:.0f} s)')
for n in T.NAMES:
    print(f'   {n:6s} = {p[n]:.4g}')
print('   correlation > 0.9:', [(T.NAMES[i], T.NAMES[j], round(corr[i, j], 3)) for i in range(len(x)) for j in range(i) if abs(corr[i, j]) > 0.9])
json.dump(dict(x=x.tolist(), p=p, rmse_mV=rm, per_cycle=per, corr=np.nan_to_num(corr).tolist(), kind=KIND, N=NN, key=KEY, Tref=TREF),
          open(f"{'fitT' if PAPER else 'fitT2'}_{KEY}_{KIND}{NN}_{TEMPF}.json", 'w'))
