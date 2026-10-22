"""
SOC EKF with ROM-FV vs ROM-hybrid. Truth = EXACT particle model (Duhamel + Laplace/eigen-series) + Mohtat2020 OCP,
NASA B0005 parameters, 25 C; voltage sensor noise 2 mV. Profile: synthetic drive cycle x3 (5400 s, dt 1 s).
Wrong initial EKF SOC (50% vs 90% true).
"""
import os; ROOT = os.environ.get('LIION_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
import numpy as np, json, sys, io, contextlib
sys.path.insert(0, ROOT + '/ext')
from scipy.linalg import expm
with contextlib.redirect_stdout(io.StringIO()):
    import hybrid as H
from common import params, voltage
import rom_hybrid as RH

p = params(); QAH = 1.8
rng = np.random.default_rng(3)
I1 = RH.profiles()['synthetic drive cycle']
I = np.r_[I1, I1, I1]; n = len(I); T = np.full(n, 298.15)
soc_true0 = 0.9
s_true = (1 - soc_true0) * QAH + np.r_[0, np.cumsum(I[:-1]) / 3600]
th0 = {'n': p['th_n0'], 'p': p['th_p0']}
qs = {el: p['g' + el] * I * p['R_' + el] ** 2 / p['D_' + el] for el in 'np'}
dts = {el: p['D_' + el] / p['R_' + el] ** 2 for el in 'np'}
sign = {'n': -1.0, 'p': 1.0}; Qel = {'n': p['Qn'], 'p': p['Qp']}
ex = {el: th0[el] + sign[el] * (1 - soc_true0) * QAH / Qel[el] + H.exact_duhamel(qs[el], dts[el]) for el in 'np'}
V_true = voltage(p, ex['n'], ex['p'], I, T)
SIG = 2e-3
V_meas = V_true + rng.normal(0, SIG, n)
soc_true = 1 - s_true / QAH
print(f'mean current {I.mean():.2f} A; true SOC range {soc_true.min():.3f}..{soc_true.max():.3f}')


def run_ekf(rom, Rv, s0=0.5 * QAH):
    A, B, C, D = rom; r = len(B)
    nx = 1 + 2 * r
    Ad, Bd = {}, {}
    for el in 'np':
        Aug = np.zeros((r + 1, r + 1)); Aug[:r, :r] = A; Aug[:r, r] = B
        E = expm(Aug * dts[el]); Ad[el], Bd[el] = E[:r, :r], E[:r, r]
    x = np.r_[s0, np.zeros(2 * r)]; P = np.diag(np.r_[0.6 ** 2, np.full(2 * r, 1e-6)])
    S_hist = np.empty(n); sd = np.empty(n); innov = np.empty(n)
    Fm = np.eye(nx); Fm[1:1 + r, 1:1 + r] = Ad['n']; Fm[1 + r:, 1 + r:] = Ad['p']
    Qn = np.r_[(0.01 / 3600) ** 2, np.full(2 * r, 1e-14)]
    for k in range(n):
        thn = th0['n'] - x[0] / p['Qn'] + C @ x[1:1 + r] + D * qs['n'][k]
        thp = th0['p'] + x[0] / p['Qp'] + C @ x[1 + r:] + D * qs['p'][k]
        h0 = voltage(p, np.array([thn]), np.array([thp]), I[k:k + 1], T[k:k + 1])[0]
        e = 1e-7
        dVn = (voltage(p, np.array([thn + e]), np.array([thp]), I[k:k + 1], T[k:k + 1])[0] - h0) / e
        dVp = (voltage(p, np.array([thn]), np.array([thp + e]), I[k:k + 1], T[k:k + 1])[0] - h0) / e
        Hj = np.r_[-dVn / p['Qn'] + dVp / p['Qp'], dVn * C, dVp * C]
        S = Hj @ P @ Hj + Rv; K = P @ Hj / S
        innov[k] = V_meas[k] - h0
        x = x + K * innov[k]
        IKH = np.eye(nx) - np.outer(K, Hj); P = IKH @ P @ IKH.T + Rv * np.outer(K, K)
        x[0] = np.clip(x[0], 0.0, 1.05 * QAH)
        S_hist[k] = x[0]; sd[k] = np.sqrt(P[0, 0])
        G = np.r_[I[k] / 3600, Bd['n'] * qs['n'][k], Bd['p'] * qs['p'][k]]
        x = Fm @ x + G; P = Fm @ P @ Fm.T + np.diag(Qn)
    return 1 - S_hist / QAH, sd / QAH, innov


OUT = {'t': list(range(n)), 'soc_true': soc_true.tolist(), 'I': I.tolist(), 'runs': {}}
cases = [('ROM-FV r=3 (current practice)', 'FV', 3, SIG), ('ROM-FV r=3, R inflated to 15 mV', 'FV', 3, 15e-3),
         ('ROM-hybrid r=3', 'HYB', 3, SIG), ('ROM-hybrid r=6', 'HYB', 6, SIG)]
for lab, kind, r, rv in cases:
    soc, sd, inn = run_ekf(RH.make_rom(kind, r), rv ** 2)
    err = soc - soc_true; m = np.arange(n) >= 600
    conv = np.flatnonzero(np.abs(err) < 0.02); tc = None
    for c in conv:                       # time after which the error stays < 2%
        if np.all(np.abs(err[c:]) < 0.02):
            tc = int(c); break
    nees = np.mean((err[m] / sd[m]) ** 2)
    OUT['runs'][lab] = dict(soc=soc.tolist(), sd=sd.tolist(), rmse_after600=float(np.sqrt(np.mean(err[m] ** 2))),
                            maxerr_after600=float(np.abs(err[m]).max()), t_conv_2pct=tc, innov_rms_mV=float(np.sqrt(np.mean(inn[m] ** 2)) * 1e3),
                            nees=float(nees), states=1 + 2 * r)
    print(f"{lab:34s}: RMSE SOC (t>600 s) {np.sqrt(np.mean(err[m]**2))*100:5.2f} %  max {np.abs(err[m]).max()*100:5.2f} %  "
          f"convergence time<2% {tc} s  innovation RMS {np.sqrt(np.mean(inn[m]**2))*1e3:5.2f} mV  NEES {nees:7.2f}", flush=True)
json.dump(OUT, open(ROOT + '/ext/out/ekf_hybrid.json', 'w'))
print('done')
