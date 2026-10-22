"""Dependence on dtau (sampling rate / D / R) and BT-SPA ROM of the hybrid system."""
import os; ROOT = os.environ.get('LIION_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
import numpy as np, json, sys, io, contextlib, mpmath as mp
sys.path.insert(0, ROOT + '/ext'); sys.path.insert(0, ROOT + '/calce')
with contextlib.redirect_stdout(io.StringIO()):
    import hybrid as H
    import rom_hybrid as RH
import calce_model as CM
from scipy.linalg import expm
import compare_tf as CT   # uses taylor_Gr, S_zeng (re-runs that script; cheap)

def pade_S(m, lags, dps=60):
    mp.mp.dps = dps
    a = CT.taylor_Gr(2 * m + 2)
    p_, q_ = mp.pade(a, m - 1, m)
    num = np.array([float(x) for x in p_][::-1]); den = np.array([float(x) for x in q_][::-1])
    poles = np.roots(den)
    if np.any(poles.real >= 0) or np.any(np.abs(poles.imag) > 1e-6 * np.abs(poles.real)):
        return None
    r_i = np.polyval(num, poles) / np.polyval(np.polyder(den), poles)
    return -3 * lags + float(num[-1] / den[-1]) + np.real((r_i / poles)[None, :] * np.exp(np.outer(lags, poles))).sum(1)

def rom_S(r, lags):
    A, B, C, D = RH.make_rom('HYB', r)
    n = len(B); Ai = np.linalg.solve(A, B)
    out = []
    for t in lags:
        out.append(C @ (expm(A * t) @ Ai - Ai) + D)
    return -3 * lags + np.array(out)

OUT = {}
for dtau in [2.8e-5, 2.8e-4, 2.8e-3, 2.8e-2]:
    lags = np.arange(1800) * dtau
    Sx = CM.HT.S_exact(lags); Sx[0] = 0.0
    g = lambda S: abs(Sx[0]-S[0]) + np.abs(np.diff(Sx - S)).sum()
    row = {}
    for m in [6, 10]:
        S = pade_S(m, lags); row[f'Pade m={m} ({m+1} states)'] = None if S is None else float(g(S))
    for r in [6, 10]:
        row[f'BT-SPA of hybrid r={r} ({r+1} states)'] = float(g(rom_S(r, lags)))
    for N, K in [(5, 8), (10, 16)]:
        H.MU = np.logspace(-1, 6.5, 16) if K == 16 else np.logspace(0, 6, 8)
        cf = H.fit_correction(N, tau_min=min(dtau, 1e-6) / 20)
        S = CM.HT.S_fv(N, lags) + cf['Einf'] + np.exp(-np.outer(lags, H.MU)) @ cf['w']
        row[f'hybrid N={N},K={K} ({N+K} states)'] = float(g(S))
    H.MU = np.logspace(-1, 6.5, 16)
    row['FV N=20 (20 states)'] = float(g(CM.HT.S_fv(20, lags)))
    row['Zeng N=20 (20 states)'] = float(g(CT.S_zeng(20, lags)))
    OUT[str(dtau)] = row
    print(f'dtau = {dtau:.1e}')
    for k, v in row.items(): print(f'    {k:36s} {v:.2e}' if v is not None else f'    {k:36s} unstable')
json.dump(OUT, open(ROOT + '/ext/out/compare_tf2.json', 'w'))
# precision sensitivity: Pade m=10 in double precision (dps=16)
lags = np.arange(1800) * 2.8e-4
for dps in [16, 20, 25, 60]:
    try:
        S = pade_S(10, lags, dps)
    except ZeroDivisionError:
        print(f'Pade m=10 with {dps} digits: singular Pade matrix'); continue
    print(f'Pade m=10 with {dps} digits: ', 'unstable/complex poles' if S is None else f'gain {abs(np.diff(CM.HT.S_exact(lags) - S)).sum():.2e}')
json.dump(OUT, open(ROOT + '/ext/out/compare_tf2.json', 'w'))
