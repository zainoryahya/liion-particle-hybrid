"""Cold-temperature twin experiment: truth = temperature-dependent hybrid fit (UDDS+US06, measured T), data = EXACT model + 2 mV noise.
Fit hybrid N=10, FV N=20, FV N=10 with the same temperature-dependent model. argv: TEMPF"""
import numpy as np, json, sys, time
from scipy.optimize import least_squares
import ss_model as S, ss_thermal as T
TEMPF = sys.argv[1]
F = json.load(open(f'fitT_UDDS+US06_hybrid10_{TEMPF}.json')); xt = np.array(F['x']); TREF = F['Tref']
BND = [(1, 5.5), (0.5, 5.5), (-3, 2.5), (-3, 2.5), (-3, -0.5), (-4, -0.5), (-1, 3.5), (0.8, 1.1), (0.85, 1.1), (0, 100), (0, 100), (0, 100)]
lo, hi = np.array([b[0] for b in BND]), np.array([b[1] for b in BND])
rng = np.random.default_rng(7); t0 = time.time()
DS = [S.load_test(k, TEMPF) for k in ['UDDS', 'US06']]
Ym = [T.simulate(xt, d, 'exact', Tref=TREF)[d['gi']] + rng.normal(0, 2e-3, len(d['gi'])) for d in DS]
dv = S.load_test('LA92', TEMPF); Yv = T.simulate(xt, dv, 'exact', Tref=TREF)[dv['gi']]
truth = T.unpack(xt); OUT = {'truth': truth, 'fits': {}, 'Tref': TREF}
# Pi at the minimum and mean cell temperature (tau_n(T) from Arrhenius)
Qn = truth['c'] * S.M.Q20 / (S.M.W['n100'] - S.M.W['n0'])
for nm, Tv in [('Tmin', min(d['T'].min() for d in DS)), ('Tmean', np.mean(np.concatenate([d['T'] for d in DS])))]:
    taun = truth['tau_n'] * np.exp(truth['EaD'] * 1e3 / S.M.Rg * (1 / Tv - 1 / TREF))
    OUT['Pi_' + nm] = float(36.0 * taun / (3 * 3600 * Qn))
print(f"data generated ({time.time()-t0:.0f} s); Pi(Tmin)={OUT['Pi_Tmin']:.2f} Pi(Tmean)={OUT['Pi_Tmean']:.2f}", flush=True)
d_ = np.array([0.3, -0.3, 0.3, -0.3, 0.2, 0.3, 0.3, 0.02, 0.01, 10, -10, 10])
starts = [xt + d_, xt - d_]
for kind, N in [('hybrid', 10), ('fv', 20), ('fv', 10)]:
    res = lambda x: np.concatenate([(T.simulate(x, d, kind, N, Tref=TREF)[d['gi']] - y)[::2] for d, y in zip(DS, Ym)])
    best = None
    for s0 in starts:
        r = least_squares(res, np.clip(s0, lo + 1e-6, hi - 1e-6), bounds=(lo, hi), x_scale='jac', max_nfev=300)
        if best is None or r.cost < best.cost:
            best = r
    p = T.unpack(best.x)
    val = float(np.sqrt(np.mean((T.simulate(best.x, dv, kind, N, Tref=TREF)[dv['gi']] - Yv) ** 2)) * 1e3)
    ratio = {k: p[k] / truth[k] for k in ['tau_n', 'tau_p', 'I0_n', 'I0_p', 'R0', 'R1']}
    ratio.update({k: p[k] - truth[k] for k in ['EaD', 'Eak', 'EaR']})
    lab = f'{kind} N={N}'
    OUT['fits'][lab] = dict(p=p, rmse_fit_mV=float(np.sqrt(np.mean(best.fun ** 2)) * 1e3), rmse_LA92_vs_truth_mV=val, ratio=ratio)
    print(f"{TEMPF} {lab:10s}: fit {OUT['fits'][lab]['rmse_fit_mV']:5.2f} mV  LA92 vs truth {val:6.2f} mV | "
          + '  '.join(f'{k} {v:.2f}' for k, v in ratio.items()) + f' ({time.time()-t0:.0f} s)', flush=True)
json.dump(OUT, open(f'twinT_{TEMPF}.json', 'w'))
