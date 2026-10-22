"""
(i) FV with grid refined towards the surface (as proposed by Zeng et al. 2013): faces r = 1 - (10^{a s} - 1)/(10^a - 1), s uniform.
    Sharp gain (Theorem 1) vs number of cells, a optimised for each N and dtau (favouring the non-uniform grid).
(ii) Implicit Euler (BE) with step equal to the sampling interval: does the error floor persist? FV vs hybrid.
"""
import os; ROOT = os.environ.get('LIION_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
import numpy as np, json, sys, io, contextlib
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar
sys.path.insert(0, ROOT + '/ext'); sys.path.insert(0, ROOT + '/calce')
with contextlib.redirect_stdout(io.StringIO()):
    import hybrid as H
import calce_model as CM


def fv_graded(faces):
    N = len(faces) - 1
    V = (faces[1:] ** 3 - faces[:-1] ** 3) / 3
    cen = 0.5 * (faces[1:] + faces[:-1])
    K = np.zeros((N, N))
    for i in range(N - 1):
        g = faces[i + 1] ** 2 / (cen[i + 1] - cen[i])
        K[i, i] -= g / V[i]; K[i, i + 1] += g / V[i]; K[i + 1, i + 1] -= g / V[i + 1]; K[i + 1, i] += g / V[i + 1]
    b = np.zeros(N); b[-1] = -1 / V[-1]
    D = -(faces[-1] - cen[-1])            # extrapolation over the last half-cell using the boundary flux
    return K, b, V, D


def step_resp(K, b, V, D, lags):
    Mv = np.diag(V); S = Mv @ K; S = 0.5 * (S + S.T)
    lam, U = eigh(S, Mv); beta = U.T @ Mv @ b; w = beta * U[-1, :]
    nz = np.abs(lam) > 1e-12
    return D + lags * w[~nz].sum() + (np.expm1(np.outer(lags, lam[nz])) / lam[nz]) @ w[nz]


def faces_of(N, a):
    s = np.linspace(0, 1, N + 1)
    x = s if a < 1e-6 else (10 ** (a * s) - 1) / (10 ** a - 1)
    return np.sort(1 - x)


def gain(Sx, Sm):
    d = Sx - Sm
    return abs(d[0]) + np.abs(np.diff(d)).sum()


OUT = {'graded': {}, 'be': {}}
for dtau in [2.8e-5, 2.8e-4, 2.8e-3]:
    lags = np.arange(1800) * dtau
    Sx = CM.HT.S_exact(lags); Sx[0] = 0.0
    row = []
    for N in [5, 10, 20, 40]:
        f = lambda a: gain(Sx, step_resp(*fv_graded(faces_of(N, a)), lags))
        r = minimize_scalar(f, bounds=(0, 4), method='bounded', options=dict(xatol=1e-3))
        g_uni = f(0.0)
        row.append(dict(N=N, a_opt=float(r.x), gain_graded=float(r.fun), gain_uniform=float(g_uni),
                        h_surface=float(1 - faces_of(N, r.x)[-2])))
        print(f'dtau={dtau:.1e} N={N:3d}: uniform {g_uni:.2e} | non-uniform a*={r.x:.2f} (surface cell {row[-1]["h_surface"]:.2e}) {r.fun:.2e}', flush=True)
    OUT['graded'][str(dtau)] = row

# (ii) implicit Euler with dt = sampling interval
dtau = 2.8e-4; n = 1800; lags = np.arange(n) * dtau
Sx = CM.HT.S_exact(lags); Sx[0] = 0.0
for N in [10, 20]:
    K, b, V, D = fv_graded(faces_of(N, 0.0))
    A_ = np.linalg.inv(np.eye(N) - dtau * K)
    x = np.zeros(N); Sfv = np.empty(n)
    for k in range(n):                                  # unit step response (q = 1)
        Sfv[k] = x[-1] + D; x = A_ @ (x + dtau * b)
    cf = H.fit_correction(N, tau_min=dtau / 20)
    xc = np.zeros(len(H.MU)); Sc = np.empty(n); Dc = cf['Einf'] + cf['w'].sum()
    for k in range(n):
        Sc[k] = xc.sum() + Dc; xc = (xc - dtau * H.MU * cf['w']) / (1 + dtau * H.MU)
    g_fv, g_hy = gain(Sx, Sfv), gain(Sx, Sfv + Sc)
    g_fv_ex = gain(Sx, step_resp(K, b, V, D, lags))
    OUT['be'][str(N)] = dict(fv_be=float(g_fv), hybrid_be=float(g_hy), fv_exact_time=float(g_fv_ex))
    print(f'implicit Euler dt=dtau, N={N}: FV {g_fv:.2e} (exact in time {g_fv_ex:.2e}) | hybrid {g_hy:.2e}', flush=True)
json.dump(OUT, open(ROOT + '/ext/out/graded.json', 'w'))
print('done')

# (iii) FV with BE + correction states integrated EXACTLY (diagonal: e^{-mu dtau}, zero cost)
for N in [10, 20]:
    K, b, V, D = fv_graded(faces_of(N, 0.0))
    A_ = np.linalg.inv(np.eye(N) - dtau * K)
    x = np.zeros(N); Sfv = np.empty(n)
    for k in range(n):
        Sfv[k] = x[-1] + D; x = A_ @ (x + dtau * b)
    cf = H.fit_correction(N, tau_min=dtau / 20)
    Ec = cf['Einf'] + np.exp(-np.outer(lags, H.MU)) @ cf['w']
    g = gain(Sx, Sfv + Ec)
    # BE-consistent version: fit the correction to the error of the FV+BE scheme itself (E_BE = S - S_FV,BE)
    E_be = Sx - Sfv
    Phi = np.exp(-np.outer(lags[1:], H.MU))
    KKT = np.block([[2 * Phi.T @ Phi, np.ones((len(H.MU), 1))], [np.ones((1, len(H.MU))), np.zeros((1, 1))]])
    Einf = E_be[-1]
    w = np.linalg.solve(KKT, np.r_[2 * Phi.T @ (E_be[1:] - Einf), E_be[0] - Einf])[:len(H.MU)]
    g2 = gain(Sx, Sfv + Einf + np.exp(-np.outer(lags, H.MU)) @ w)
    OUT['be'][str(N)].update(hybrid_be_fv_exact_corr=float(g), hybrid_be_refit=float(g2))
    print(f'N={N}: FV(BE) + exact correction {g:.2e} | correction refitted to FV+BE error {g2:.2e}', flush=True)
json.dump(OUT, open(ROOT + '/ext/out/graded.json', 'w'))
