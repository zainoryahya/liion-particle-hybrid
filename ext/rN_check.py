"""Extended numerical evidence for Prop 1(c): sup|r_N| and TV(r_N)/h^2 up to N = 320."""
import os; ROOT = os.environ.get('LIION_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
import numpy as np, sys, json
sys.path.insert(0, ROOT + '/calce')
import calce_model as CM
Mh = 40000; kk = np.arange(Mh); lam_h = -4 * np.sin(kk * np.pi / (2 * Mh)) ** 2
v1sq = np.where(kk == 0, 1.0 / Mh, 2.0 / Mh * np.cos(kk * np.pi / (2 * Mh)) ** 2)
def g_half(sig):
    out = np.empty_like(sig)
    for i in range(0, len(sig), 100):
        s = sig[i:i + 100]; lt = np.outer(s, lam_h)
        with np.errstate(divide='ignore', invalid='ignore'):
            ph = np.where(lam_h == 0, s[:, None], np.expm1(lt) / np.where(lam_h == 0, 1, lam_h))
        out[i:i + 100] = -2 * np.sqrt(s / np.pi) + 0.5 + ph @ v1sq
    return out
rows = []
for N in [10, 20, 40, 80, 160, 320]:
    h = 1 / N
    tau = np.r_[0.0, np.logspace(-11, 1.3, 2500)]
    tau = tau[tau / h ** 2 <= 1e5] if N <= 80 else tau
    sig = tau / h ** 2
    E = CM.HT.S_exact(tau) - CM.HT.S_fv(N, tau)
    ok = sig <= 1e5
    g = np.where(ok, 0.0, 1 / (8 * np.sqrt(np.pi * np.maximum(sig, 1e-30))))
    g[ok] = g_half(sig[ok])
    r = E - h * g; r[0] = 0.0
    rows.append(dict(N=N, sup=float(np.abs(r).max()), sup_h2=float(np.abs(r).max() * N * N), TV_h2=float(np.abs(np.diff(r)).sum() * N * N)))
    print(f"N={N:4d}: sup|r_N|/h^2 = {rows[-1]['sup_h2']:.4f}   TV(r_N)/h^2 = {rows[-1]['TV_h2']:.4f}", flush=True)
json.dump(rows, open(ROOT + '/ext/out/rN_check.json', 'w'))
