"""Single-precision (float32) implementation of the corrected particle models, as on embedded BMS hardware.

The FV model (N = 10) with the per-mesh (K = 16) or the universal (K = 40) correction is discretized exactly for a
zero-order-hold input, the coefficients are rounded to float32 and the recursion is run entirely in float32, in two forms:
  'dense'    : x_{k+1} = Ad x_k + Bd q_k with the full matrix exponential of the augmented system;
  'diagonal' : FV in modal coordinates plus the K diagonal correction states (every state is a scalar first-order
               recursion), with the conserved mode (particle average) accumulated separately, as in coulomb counting. The result is compared with the
same recursion in float64 on a long zero-mean random BMS-type profile (holds of 1-29 samples, |q| <= 1.5,
100 000 samples at dtau = 2.8e-4, i.e. 28 diffusion times), starting from a uniform stoichiometry u0 = 0.5.
Writes ext/out/float32_check.json.
"""
import os, sys, json, io, contextlib
ROOT = os.environ.get('LIION_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('LIION_ROOT', ROOT)
import numpy as np
from scipy.linalg import expm
sys.path.insert(0, ROOT + '/ext'); sys.path.insert(0, ROOT + '/battery')
with contextlib.redirect_stdout(io.StringIO()):
    import hybrid as H

N, dtau, nst = 10, 2.8e-4, 100_000
rng = np.random.default_rng(7)
q = np.empty(nst); k = 0
while k < nst:
    L = rng.integers(1, 30); q[k:k + L] = rng.uniform(-1.5, 1.5); k += L
q -= q.mean()                                   # zero mean: the stoichiometry stays bounded
u0 = 0.5

UNI = json.load(open(ROOT + '/ext/out/hybrid_theory.json'))['universal']


def system(kind):
    A, b, C, D = H.fv_sys(N); n = len(b); h = 1 / N
    if kind == 'FV':
        return A, b, C, D
    if kind == 'per-mesh':
        cf = H.fit_correction(N); mu, w, Einf = H.MU, cf['w'], cf['Einf']
    else:
        mu = np.array(UNI['nu']) / h ** 2; w = h * np.array(UNI['w']); Einf = 0.0
    K = len(mu)
    Af = np.zeros((n + K, n + K)); Af[:n, :n] = A; Af[n:, n:] = np.diag(-mu)
    return Af, np.r_[b, -mu * w], np.r_[C, np.ones(K)], D + Einf + w.sum()


def run(Ad, Bd, C, D, x0, dt):
    Ad, Bd, C, D = [np.asarray(z, dtype=dt) for z in (Ad, Bd, C, D)]
    x = np.asarray(x0, dtype=dt); y = np.empty(nst, dtype=dt); qq = q.astype(dt)
    for k in range(nst):
        y[k] = C @ x + D * qq[k]
        x = Ad @ x + Bd * qq[k]
    return y.astype(float)


from scipy.linalg import eigh
from spm import SPM


def run_diag(kind, dt):
    A, b, C, D = system(kind)
    m_ = SPM(N=N); Mv = np.diag(m_.vol); Sm = Mv @ m_.K; Sm = 0.5 * (Sm + Sm.T)
    lam, V = eigh(Sm, Mv); beta = V.T @ Mv @ m_.bsurf; nz = np.abs(lam) > 1e-9 * np.abs(lam).max()
    K = len(b) - N; mu = -np.diag(A)[N:]; bc = b[N:]
    ad = np.r_[np.exp(lam[nz] * dtau), np.exp(-mu * dtau)]
    bd = np.r_[beta[nz] * np.expm1(lam[nz] * dtau) / lam[nz], bc * (-np.expm1(-mu * dtau)) / mu if K else []]
    cc = np.r_[V[-1, nz], np.ones(K)]
    zg = (beta[~nz] * V[-1, ~nz]).sum() * dtau       # conserved mode: y gains zg * q_k per sample
    a_, b_, c_ = ad.astype(dt), bd.astype(dt), cc.astype(dt); Dd, zg, qq = dt(D), dt(zg), q.astype(dt)
    z = np.zeros(len(ad), dt); acc = dt(u0); y = np.empty(nst)
    for k in range(nst):
        y[k] = acc + c_ @ z + Dd * qq[k]
        z = a_ * z + b_ * qq[k]; acc = acc + zg * qq[k]
    return y


OUT = dict(N=N, dtau=dtau, n_samples=nst, u0=u0, runs={})
for kind in ['FV', 'per-mesh', 'universal']:
    A, b, C, D = system(kind); m = len(b)
    Aug = np.zeros((m + 1, m + 1)); Aug[:m, :m] = A; Aug[:m, m] = b
    E = expm(Aug * dtau); Ad, Bd = E[:m, :m], E[:m, m]
    x0 = np.r_[np.full(N, u0), np.zeros(m - N)]
    y64 = run(Ad, Bd, C, D, x0, np.float64); y32 = run(Ad, Bd, C, D, x0, np.float32)
    d = np.abs(y32 - y64)
    dd = np.abs(run_diag(kind, np.float32) - run_diag(kind, np.float64))
    OUT['runs'][kind] = dict(states=m, dense_max_abs_diff=float(d.max()), dense_first_1800=float(d[:1800].max()),
                             diagonal_max_abs_diff=float(dd.max()), diagonal_first_1800=float(dd[:1800].max()),
                             max_fp64_vs_reference_y_check=float(np.abs(run_diag(kind, np.float64) - y64).max()))
    print(f'{kind:9s} ({m:2d} states): max |y_float32 - y_float64| over {nst} samples: dense {d.max():.2e}, '
          f'diagonal {dd.max():.2e}   (double-precision forms agree to {OUT["runs"][kind]["max_fp64_vs_reference_y_check"]:.1e})')
json.dump(OUT, open(ROOT + '/ext/out/float32_check.json', 'w'), indent=1)
