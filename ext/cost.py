"""
Reviewer request: cost vs accuracy. For each surface-concentration approximation, find the smallest model whose sharp
l_inf gain (Theorem 1) is <= TARGET at dtau = 2.8e-4 (1 s, fitted graphite) and 2.8e-5 (fast sampling / large particles),
and report states, stiffness (max |eigenvalue|) and per-step cost of an exact zero-order-hold update.
"""
import os; ROOT = os.environ.get('LIION_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
import numpy as np, json, sys, io, contextlib, mpmath as mp
from scipy.linalg import expm, eigh
from scipy.optimize import minimize_scalar
sys.path.insert(0, ROOT + '/ext'); sys.path.insert(0, ROOT + '/calce'); sys.path.insert(0, ROOT + '/battery')
with contextlib.redirect_stdout(io.StringIO()):
    import hybrid as H
    import rom_hybrid as RH
import calce_model as CM
import compare_tf3 as T3
_g = {}; _src = open(ROOT + '/ext/graded.py').read(); exec(_src[:_src.index('OUT = {')], _g)
fv_graded, faces_of, step_resp = _g['fv_graded'], _g['faces_of'], _g['step_resp']

TARGET = 1e-3


def gain(Sx, Sm):
    d = Sx - Sm
    return float(abs(d[0]) + np.abs(np.diff(d)).sum())


def taylor_Gr(n):
    f = [mp.mpf(2) ** (2 * k) * mp.bernoulli(2 * k) / mp.factorial(2 * k) for k in range(1, n + 3)]
    inv = [1 / f[0]]
    for k in range(1, n + 2):
        inv.append(-sum(f[j] * inv[k - j] for j in range(1, k + 1)) / f[0])
    return [-inv[k + 1] for k in range(n + 1)]


def pade_S(m, lags):
    mp.mp.dps = 60
    p_, q_ = mp.pade(taylor_Gr(2 * m + 2), m - 1, m)
    num = np.array([float(x) for x in p_][::-1]); den = np.array([float(x) for x in q_][::-1])
    poles = np.roots(den)
    r_i = np.polyval(num, poles) / np.polyval(np.polyder(den), poles)
    return -3 * lags + float(num[-1] / den[-1]) + np.real((r_i / poles)[None, :] * np.exp(np.outer(lags, poles))).sum(1), \
        float(np.max(np.abs(poles)))


def zeng_S(N, lags):
    K, b, V, D = fv_graded(faces_of(N, 0.0))
    Mv = np.diag(V); S = Mv @ K; S = 0.5 * (S + S.T)
    lam, U = eigh(S, Mv); beta = U.T @ Mv @ b
    cvec = 1.5 * U[-1, :] - 0.5 * U[-2, :]
    w = beta * cvec; nz = np.abs(lam) > 1e-12
    return lags * w[~nz].sum() + (np.expm1(np.outer(lags, lam[nz])) / lam[nz]) @ w[nz], float(np.max(np.abs(lam)))


def rom_S(r, lags):
    A, B, C, D = RH.make_rom('HYB', r)
    w, U = np.linalg.eig(A); Bm = np.linalg.solve(U, B); Cm = C @ U
    return -3 * lags + D + np.real((Cm * Bm / w)[None, :] * np.expm1(np.outer(lags, w))).sum(1), float(np.max(np.abs(w)))


def first(cands, Sx, lags):
    """cands: list of (label, states, fn) in increasing size; returns first meeting the target (and the last tried)."""
    last = None
    for lab, n, fn in cands:
        S, stiff = fn()
        g = gain(Sx, S)
        last = dict(config=lab, states=n, gain=g, stiff=stiff)
        if g <= TARGET:
            return last
    last['not_reached'] = True
    return last


OUT = {}
for dtau in [2.8e-4, 2.8e-5]:
    lags = np.arange(1800) * dtau
    Sx = CM.HT.S_exact(lags); Sx[0] = 0.0
    row = {}

    def fvu(N):
        K, b, V, D = fv_graded(faces_of(N, 0.0)); return step_resp(K, b, V, D, lags), float(np.max(np.abs(np.linalg.eigvals(K))))
    row['FV, uniform'] = first([(f'N={N}', N, (lambda N=N: fvu(N))) for N in [100, 200, 400, 700, 1000, 1500]], Sx, lags)

    def fvg(N):
        f = lambda a: gain(Sx, step_resp(*fv_graded(faces_of(N, a)), lags))
        a = minimize_scalar(f, bounds=(0, 4), method='bounded', options=dict(xatol=1e-3)).x
        K, b, V, D = fv_graded(faces_of(N, a))
        return step_resp(K, b, V, D, lags), float(np.max(np.abs(np.linalg.eigvals(K))))
    row['FV, optimally graded'] = first([(f'N={N}', N, (lambda N=N: fvg(N))) for N in [20, 30, 40, 50, 60, 80, 100]], Sx, lags)
    row['FV, two-cell extrapolation'] = first([(f'N={N}', N, (lambda N=N: zeng_S(N, lags))) for N in [100, 200, 300, 400, 600, 800]], Sx, lags)

    def cheb(Nc):
        A, b, C, d, ns = T3.cheb_sys(Nc); return T3.step_lin(A, b, C, d, lags), float(np.max(np.abs(np.linalg.eigvals(A))))
    row['Chebyshev collocation'] = first([(f'Nc={Nc}', T3.cheb_sys(Nc)[4], (lambda Nc=Nc: cheb(Nc))) for Nc in range(40, 161, 8)], Sx, lags)
    row['Pade approximant'] = first([(f'm={m}', m + 1, (lambda m=m: pade_S(m, lags))) for m in range(1, 11)], Sx, lags)

    def dra(n):
        Ac, Bc, Cd = T3.dra_cont(n, dtau, 1500); return T3.step_cont(Ac, Bc, Cd, lags), float(np.max(np.abs(np.linalg.eigvals(Ac))))
    row['DRA (realized at this interval)'] = first([(f'n={n}', n + 1, (lambda n=n: dra(n))) for n in range(2, 13)], Sx, lags)

    def hyb(N, K):
        H.MU = np.logspace(-1, 6.5, 16) if K == 16 else np.logspace(0, 6, 8)
        cf = H.fit_correction(N, tau_min=dtau / 20)
        S = CM.HT.S_fv(N, lags) + cf['Einf'] + np.exp(-np.outer(lags, H.MU)) @ cf['w']
        K_, b, V, D = fv_graded(faces_of(N, 0.0))
        return S, float(max(np.max(np.abs(np.linalg.eigvals(K_))), H.MU.max()))
    row['hybrid (FV + correction)'] = first([(f'N={N},K={K}', N + K, (lambda N=N, K=K: hyb(N, K)))
                                             for N, K in [(5, 8), (10, 8), (5, 16), (10, 16), (20, 16)]], Sx, lags)
    H.MU = np.logspace(-1, 6.5, 16)
    row['BT-SPA of hybrid'] = first([(f'r={r}', r + 1, (lambda r=r: rom_S(r, lags))) for r in range(3, 13)], Sx, lags)
    OUT[str(dtau)] = row
    print(f'dtau = {dtau:.1e}, target gain {TARGET:.0e}')
    for k, v in row.items():
        print(f'   {k:32s} {v["config"]:10s} states {v["states"]:5d} gain {v["gain"]:.2e} max|lambda| {v["stiff"]:.2e}'
              + ('  (target not reached)' if v.get('not_reached') else ''), flush=True)
json.dump(OUT, open(ROOT + '/ext/out/cost.json', 'w'), indent=1)
