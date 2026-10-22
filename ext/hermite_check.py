"""Hermite extrapolation (Xu et al. 2023): quadratic through c_{N-1}, c_N and the surface flux -> y = (9c_N - c_{N-1})/8 - (3h/8) q.
Sharp gain (Theorem 1) vs plain FV, two-cell (Zeng) and hybrid."""
import os; ROOT = os.environ.get('LIION_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
import numpy as np, sys, io, contextlib, json
from scipy.linalg import eigh
sys.path.insert(0, ROOT + '/ext'); sys.path.insert(0, ROOT + '/calce')
with contextlib.redirect_stdout(io.StringIO()):
    import graded as G
import calce_model as CM

def step_resp_C(K, b, V, C, D, lags):
    Mv = np.diag(V); S = Mv @ K; S = 0.5 * (S + S.T)
    lam, U = eigh(S, Mv); beta = U.T @ Mv @ b; w = beta * (C @ U)
    nz = np.abs(lam) > 1e-12
    return D + lags * w[~nz].sum() + (np.expm1(np.outer(lags, lam[nz])) / lam[nz]) @ w[nz]

OUT = {}
for dtau in [2.8e-5, 2.8e-4, 2.8e-3]:
    lags = np.arange(1800) * dtau; Sx = CM.HT.S_exact(lags); Sx[0] = 0.0
    for N in [5, 10, 20, 40]:
        K, b, V, D = G.fv_graded(G.faces_of(N, 0.0)); h = 1.0 / N
        e = np.zeros(N); e[-1] = 1
        Cz = np.zeros(N); Cz[-1] = 1.5; Cz[-2] = -0.5
        Ch = np.zeros(N); Ch[-1] = 9 / 8; Ch[-2] = -1 / 8
        g = dict(fv=G.gain(Sx, step_resp_C(K, b, V, e, -h / 2, lags)),
                 twocell=G.gain(Sx, step_resp_C(K, b, V, Cz, 0.0, lags)),
                 hermite=G.gain(Sx, step_resp_C(K, b, V, Ch, -3 * h / 8, lags)),
                 hybrid=G.gain(Sx, CM.kernel('hybrid', N, lags) if N in (5, 10, 20, 40) else np.nan))
        OUT[f'{dtau:g}_{N}'] = g
        print(f'dtau={dtau:.1e} N={N:3d}: ' + '  '.join(f'{k} {v:.2e}' for k, v in g.items()), flush=True)
json.dump(OUT, open(ROOT + '/ext/out/hermite.json', 'w'))
