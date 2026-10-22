"""
Hybrid analytical-numerical scheme for particle diffusion under piecewise-constant current (extension of
Hashim Ali, Hussin & Arof, J. Power Sources 112 (2002): short-time & long-time solutions via Laplace transform).

Dimensionless problem: u_tau = r^-2 (r^2 u_r)_r, -u_r(1) = q(tau) (outward flux), u(r,0)=0.
Exact step response (q = 1 for tau > 0), at the surface:
  Laplace:  U(1,s) = -1 / ( s (sqrt(s) coth sqrt(s) - 1) )
  short time (large s, coth -> 1):  S(tau) ~ -[ 2 sqrt(tau/pi) + tau + (4/3) tau^{3/2}/sqrt(pi) + tau^2/2 + ... ]
  long time (eigen-series, tan l = l):  S(tau) = -3 tau - 1/5 + sum a_n sin(l_n) exp(-l_n^2 tau)
FV (half-cell extrapolation) has step response S_h(tau) with S_h(0+) = -h/2 (instantaneous jump)
-> O(h) error after EVERY current step.
Correction: E(tau) = S(tau) - S_h(tau) ~= E_inf + sum_k w_k exp(-mu_k tau)  (mu_k fixed, w_k by least squares)
realised as K additional linear states: x_k' = -mu_k x_k - mu_k w_k q,  dy = (E_inf + sum w_k) q + sum x_k.
"""
import os; ROOT = os.environ.get('LIION_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
import numpy as np, json, sys
from scipy.linalg import expm
sys.path.insert(0, ROOT + '/battery')
from spm import SPM

OUT = {}
# ---------- exact step response
from scipy.optimize import brentq
LAM = np.array([brentq(lambda x: np.tan(x) - x, k * np.pi + 1e-9, k * np.pi + np.pi / 2 - 1e-9) for k in range(1, 801)])
xg, wg = np.polynomial.legendre.leggauss(4000); rg = 0.5 * (xg + 1); wg = 0.5 * wg
AN = np.array([np.sum(wg * (rg ** 2 / 2 - 0.3) * np.sin(l * rg) * rg) / (0.5 - np.sin(2 * l) / (4 * l)) for l in LAM])


def S_long(tau):
    tau = np.atleast_1d(tau).astype(float)
    return -3 * tau - 0.2 + (AN * np.sin(LAM) * np.exp(-np.outer(tau, LAM ** 2))).sum(1)


def S_short(tau):
    tau = np.atleast_1d(tau).astype(float)
    return -(2 * np.sqrt(tau / np.pi) + tau + 4 / 3 * tau ** 1.5 / np.sqrt(np.pi) + tau ** 2 / 2)


def S_exact(tau, tau_switch=2e-3):
    tau = np.atleast_1d(tau).astype(float)
    out = np.empty_like(tau)
    m = tau < tau_switch
    out[m] = S_short(tau[m]); out[~m] = S_long(tau[~m])
    out[tau <= 0] = 0.0
    return out


# check: overlap region of short-time / long-time series
for tt in [1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]:
    print(f'tau={tt:.0e}: short time {S_short(tt)[0]:+.8f}  long time (800 terms) {S_long(tt)[0]:+.8f}  diff {abs(S_short(tt)[0]-S_long(tt)[0]):.1e}')


# ---------- FV step response
def fv_sys(N):
    m = SPM(N=N)
    C = np.zeros(N); C[-1] = 1.0
    return m.K, m.bsurf, C, -0.5 * m.dr


def S_fv(N, tau):
    A, b, C, D = fv_sys(N)
    n = len(b); Aug = np.zeros((n + 1, n + 1)); Aug[:n, :n] = A; Aug[:n, n] = b
    return np.array([C @ expm(Aug * t)[:n, n] + D for t in np.atleast_1d(tau)])


MU = np.logspace(-1, 6.5, 16)
def fit_correction(N, tau_min=1e-6):
    tau = np.logspace(np.log10(tau_min), 1, 400)
    E = S_exact(tau) - S_fv(N, tau)
    Einf = E[-1]
    E0 = -S_fv(N, [0.0])[0]          # E(0+) = 0 - S_h(0+) = h/2 (exact)
    Phi = np.exp(-np.outer(tau, MU))
    # constrained least squares: Einf + sum w = E0 (exact match at the instant of a current step)
    KKT = np.block([[2 * Phi.T @ Phi, np.ones((len(MU), 1))], [np.ones((1, len(MU))), np.zeros((1, 1))]])
    rhs = np.r_[2 * Phi.T @ (E - Einf), E0 - Einf]
    w = np.linalg.solve(KKT, rhs)[:len(MU)]
    fit = Einf + Phi @ w
    return dict(w=w, Einf=Einf, tau=tau, E=E, fit_err=float(np.abs(fit - E).max()))


# ---------- simulation: random BMS-type flux profile (1 s steps), in dimensionless time
def simulate(N, q, dtau, corr=None):
    A, b, C, D = fv_sys(N)
    n = len(b)
    if corr is not None:
        K = len(MU)
        Afull = np.zeros((n + K, n + K)); Afull[:n, :n] = A; Afull[n:, n:] = np.diag(-MU)
        bfull = np.r_[b, -MU * corr['w']]
        Cfull = np.r_[C, np.ones(K)]; Dfull = D + corr['Einf'] + corr['w'].sum()
    else:
        Afull, bfull, Cfull, Dfull = A, b, C, D
    m = len(bfull)
    Aug = np.zeros((m + 1, m + 1)); Aug[:m, :m] = Afull; Aug[:m, m] = bfull
    E = expm(Aug * dtau); Ad, Bd = E[:m, :m], E[:m, m]
    x = np.zeros(m); y = np.empty(len(q))
    for k in range(len(q)):
        y[k] = Cfull @ x + Dfull * q[k]
        x = Ad @ x + Bd * q[k]
    return y


def exact_duhamel(q, dtau):
    """u_s(tau_k) with q piecewise constant on [tau_j, tau_j+1); output at tau_k uses q[k] (right limit)."""
    dq = np.diff(np.r_[0.0, q]); idx = np.flatnonzero(dq != 0)
    y = np.zeros(len(q))
    for j in idx:
        lag = (np.arange(len(q)) - j) * dtau
        mk = lag >= 0
        y[mk] += dq[j] * S_exact(lag[mk])
    return y


if __name__ == '__main__':
    rng = np.random.default_rng(1)
    dtau = 2.8e-4                       # 1 s for NASA graphite (D = 2.8e-14, R = 10 um)
    nsteps = 1800
    q = np.zeros(nsteps); k = 0
    while k < nsteps:
        L = rng.integers(1, 30); q[k:k + L] = rng.uniform(-1.0, 1.5); k += L
    ref = exact_duhamel(q, dtau)
    OUT['q'] = q.tolist(); OUT['ref'] = ref.tolist(); OUT['dtau'] = dtau
    rows = []
    for N in [5, 10, 20, 40, 80]:
        yp = simulate(N, q, dtau)
        cf = fit_correction(N)
        yc = simulate(N, q, dtau, corr=cf)
        ep, ec = np.abs(yp - ref), np.abs(yc - ref)
        rows.append(dict(N=N, plain_max=float(ep.max()), plain_rms=float(np.sqrt(np.mean(ep ** 2))),
                         hyb_max=float(ec.max()), hyb_rms=float(np.sqrt(np.mean(ec ** 2))), fit_err=cf['fit_err'],
                         states_plain=N, states_hyb=N + len(MU)))
        print(f'N={N:3d}: plain max {ep.max():.2e} RMS {np.sqrt(np.mean(ep**2)):.2e} | hybrid (+{len(MU)} states) max {ec.max():.2e} RMS {np.sqrt(np.mean(ec**2)):.2e} | E fit error {cf["fit_err"]:.1e}')
        if N == 10:
            OUT['y_plain_N10'] = yp.tolist(); OUT['y_hyb_N10'] = yc.tolist(); OUT['E_N10'] = dict(tau=cf['tau'].tolist(), E=cf['E'].tolist())
    OUT['rows'] = rows
    Ns = np.array([r['N'] for r in rows])
    for key in ['plain_max', 'hyb_max', 'plain_rms', 'hyb_rms']:
        e = np.array([r[key] for r in rows]); p = -np.polyfit(np.log(Ns), np.log(e), 1)[0]
        OUT['order_' + key] = p; print(f'  empirical order {key}: {p:.2f}')
    json.dump(OUT, open(ROOT + '/ext/out/hybrid.json', 'w'))
    print('done')
