"""
Comparison with transfer-function methods (reviewers will certainly ask):
  (i)  Pade [m-1/m] of the exact transfer function (Forman et al. 2011), with the integrator separated
  (ii) exact mode truncation + static correction (equivalent to residue grouping, Smith et al. 2008, in the limit)
  (iii) FV with the two-cell extrapolation of Zeng et al. 2013:  c_s = (3 c_N - c_{N-1}) / 2
  (iv) half-cell FV (ours) and hybrid
Metric: SHARP l_inf gain (Theorem 1) at dtau = 2.8e-4 (graphite, 1 s), horizon 1800 samples, vs number of states.
"""
import os; ROOT = os.environ.get('LIION_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
import numpy as np, json, sys, io, contextlib, mpmath as mp
sys.path.insert(0, ROOT + '/ext'); sys.path.insert(0, ROOT + '/calce')
with contextlib.redirect_stdout(io.StringIO()):
    import hybrid as H
import calce_model as CM
from scipy.linalg import eigh
sys.path.insert(0, ROOT + '/battery')
from spm import SPM

mp.mp.dps = 60
dtau = 2.8e-4; nst = 1800
lags = np.arange(nst) * dtau
Sx = CM.HT.S_exact(lags); Sx[0] = 0.0


def gain(S_model):
    d = Sx - S_model
    return abs(d[0]) + np.abs(np.diff(d)).sum()


OUT = {}
# (iv) half-cell FV and hybrid
OUT['FV (half-cell flux extrapolation)'] = [(N, float(gain(CM.HT.S_fv(N, lags)))) for N in [5, 10, 20, 40, 80, 160]]
hy = []
for N in [5, 10, 20]:
    for K in [8, 16]:
        H.MU = np.logspace(-1, 6.5, 16) if K == 16 else np.logspace(0, 6, 8)
        cf = H.fit_correction(N, tau_min=dtau / 20)
        Sh = CM.HT.S_fv(N, lags) + cf['Einf'] + np.exp(-np.outer(lags, H.MU)) @ cf['w']
        hy.append((N + K, float(gain(Sh)), f'N={N},K={K}'))
H.MU = np.logspace(-1, 6.5, 16)
OUT['hybrid (this work)'] = sorted(hy)


# (iii) Zeng: (3 c_N - c_{N-1})/2
def S_zeng(N, tau):
    m = SPM(N=N); Mv = np.diag(m.vol); S = Mv @ m.K; S = 0.5 * (S + S.T)
    lam, V = eigh(S, Mv); beta = V.T @ Mv @ m.bsurf
    cvec = 1.5 * V[-1, :] - 0.5 * V[-2, :]
    wgt = beta * cvec; nz = np.abs(lam) > 1e-12
    return tau * wgt[~nz].sum() + (np.expm1(np.outer(tau, lam[nz])) / lam[nz]) @ wgt[nz]
OUT['FV, two-cell extrapolation (Zeng 2013)'] = [(N, float(gain(S_zeng(N, lags)))) for N in [5, 10, 20, 40, 80, 160]]

# (ii) m exact modes + static correction (feedthrough = DC residual)
c_n = H.AN * np.sin(H.LAM)
mt = []
for m in [2, 4, 8, 16, 32, 64]:
    S = -3 * lags - 0.2 + (c_n[:m] * np.exp(-np.outer(lags, H.LAM[:m] ** 2))).sum(1)
    mt.append((m + 1, float(gain(S))))
OUT['exact modes + static correction (residue-grouping limit)'] = mt


# (i) Pade [m-1/m] of G_r(s) = G(s) + 3/s,  G(s) = -1/(sqrt(s) coth sqrt(s) - 1)
def taylor_Gr(n):
    # f(s) = (sqrt(s) coth sqrt(s) - 1)/s = sum_{k>=1} 2^{2k} B_{2k} s^{k-1} / (2k)!
    f = [mp.mpf(2) ** (2 * k) * mp.bernoulli(2 * k) / mp.factorial(2 * k) for k in range(1, n + 3)]
    # 1/f series
    inv = [1 / f[0]]
    for k in range(1, n + 2):
        inv.append(-sum(f[j] * inv[k - j] for j in range(1, k + 1)) / f[0])
    # G = -(1/s) * (1/f) = -inv0/s - inv1 - inv2 s - ... ; inv0 = 3 ; G_r = G + 3/s
    return [-inv[k + 1] for k in range(n + 1)]


pd = []
for m in [1, 2, 3, 4, 5, 6, 8, 10]:
    a = taylor_Gr(2 * m + 2)
    p_, q_ = mp.pade(a, m - 1, m)          # numerator deg m-1, denominator deg m
    num = np.array([float(x) for x in p_][::-1]); den = np.array([float(x) for x in q_][::-1])
    poles = np.roots(den)
    if np.any(poles.real >= 0):
        print('Pade m', m, 'unstable poles', poles); continue
    # step response of G_r: G_r(0) + sum r_i/p_i e^{p_i tau}
    r_i = np.polyval(num, poles) / np.polyval(np.polyder(den), poles)
    step = float(num[-1] / den[-1]) + np.real((r_i / poles)[None, :] * np.exp(np.outer(lags, poles))).sum(1)
    S = -3 * lags + step
    pd.append((m + 1, float(gain(S))))
    print(f'Pade [{m-1}/{m}]: poles {np.round(np.sort(poles.real), 2)[:4]} ... G_r(0)={num[-1]/den[-1]:.5f} (exact -0.2) gain {gain(S):.3e}')
OUT['Pade of exact transfer function (Forman 2011)'] = pd

for k, v in OUT.items():
    print(k); [print('   ', r) for r in v]
json.dump(OUT, open(ROOT + '/ext/out/compare_tf.json', 'w'))
