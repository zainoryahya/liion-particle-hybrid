"""
Reviewer-requested baselines for the sharp l_inf gain comparison (Theorem 1):
  (a) Chebyshev (Gauss-Lobatto) collocation of the particle problem, as in Bizeray et al. (2015):
      v = r u is odd, v_t = v_rr on [-1, 1], Robin ends v_r(1) - v(1) = -q, v_r(-1) + v(-1) = -q.
      Boundary values eliminated; states = interior nodes with r > 0 (the odd subspace is invariant).
  (b) Discrete-time realization (DRA, Lee, Chemistruck & Plett 2012) built in its ideal form from the EXACT sampled
      step response at the BMS interval dtau: Ho-Kalman/ERA on the Markov parameters of G_r(s) = G(s) + 3/s,
      integrator kept exactly; n + 1 states.
Outputs ext/out/compare_tf3.json: gain vs states at dtau = 2.8e-4 and gain vs dtau for fixed budgets.
"""
import os; ROOT = os.environ.get('LIION_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
import numpy as np, json, sys, io, contextlib
from scipy.linalg import expm, svd, eig
sys.path.insert(0, ROOT + '/ext'); sys.path.insert(0, ROOT + '/calce')
with contextlib.redirect_stdout(io.StringIO()):
    import hybrid as H
import calce_model as CM


def cheb(N):
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.r_[2, np.ones(N - 1), 2] * (-1) ** np.arange(N + 1)
    X = np.tile(x, (N + 1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1 / c) / (dX + np.eye(N + 1))
    D -= np.diag(D.sum(1))
    return D, x


def cheb_sys(Nc):
    """Linear system for the surface value; Nc even. Returns (A, b, C, d, nstates_odd)."""
    D, x = cheb(Nc); D2 = D @ D
    ib = [0, Nc]; ii = np.arange(1, Nc)
    # boundary rows: at x=1 (index 0): D[0]v - v0 = -q ; at x=-1 (index Nc): D[Nc]v + vN = -q
    Rb = np.array([D[0] - np.eye(Nc + 1)[0], D[Nc] + np.eye(Nc + 1)[Nc]])
    Mbb = Rb[:, ib]; Mbi = Rb[:, ii]
    # v_b = Mbb^{-1}(-q*1 - Mbi v_i)
    Minv = np.linalg.inv(Mbb)
    Pb = -Minv @ Mbi; pb = -Minv @ np.ones(2)
    A = D2[np.ix_(ii, ii)] + D2[np.ix_(ii, ib)] @ Pb
    b = D2[np.ix_(ii, ib)] @ pb
    C = Pb[0]; d = pb[0]                       # surface value v(1) = u(1)
    # restrict to the invariant odd subspace v(-r) = -v(r): states = interior nodes with r > 0 (the even subspace
    # contains a spurious unstable mode, which roundoff would excite over long horizons)
    xi = x[ii]; pos = np.flatnonzero(xi > 1e-12); n = len(pos)
    P = np.zeros((len(ii), n))
    for j, k in enumerate(pos):
        P[k, j] = 1.0
        P[np.argmin(np.abs(xi + xi[k])), j] = -1.0
    R = np.zeros((n, len(ii))); R[np.arange(n), pos] = 1.0
    return R @ A @ P, R @ b, C @ P, d, n


def step_lin(A, b, C, d, lags):
    n = len(b); Aug = np.zeros((n + 1, n + 1)); Aug[:n, :n] = A; Aug[:n, n] = b
    dt = lags[1] - lags[0]
    E = expm(Aug * dt); Ad, Bd = E[:n, :n], E[:n, n]
    x = np.zeros(n); out = np.empty(len(lags))
    for k in range(len(lags)):
        out[k] = C @ x + d
        x = Ad @ x + Bd
    return out


def dra(n, dtau, nh=1500):
    """Ho-Kalman realization of order n from exact Markov parameters of G_r at interval dtau (+ exact integrator)."""
    lags = np.arange(2 * nh + 2) * dtau
    Sr = CM.HT.S_exact(lags) + 3 * lags; Sr[0] = 0.0          # step response of G_r (S_r(0+) = 0)
    h = np.diff(Sr)                                           # h[k-1] = Markov parameter k >= 1
    Hk = np.array([h[i:i + nh] for i in range(nh)])
    Hk1 = np.array([h[i + 1:i + 1 + nh] for i in range(nh)])
    U, s, Vt = svd(Hk, full_matrices=False)
    U, s, Vt = U[:, :n], s[:n], Vt[:n]
    sq = np.sqrt(s)
    Ad = (U / sq).T @ Hk1 @ (Vt.T / sq)
    Bd = (sq[:, None] * Vt)[:, 0]
    Cd = (U * sq)[0]
    return Ad, Bd, Cd, s


def step_dra(Ad, Bd, Cd, lags):
    # output at tau_k uses q_k (same convention as the other surrogates): y_k = -3 tau_k + sum_{j<k} Cd Ad^{k-1-j} Bd
    x = np.zeros(len(Bd)); out = np.empty(len(lags))
    for k in range(len(lags)):
        out[k] = -3 * lags[k] + Cd @ x
        x = Ad @ x + Bd
    return out


def dra_cont(n, dt0, nh):
    """DRA realized at dt0, converted to continuous time (poles are real in (0,1), so logm is well defined)."""
    Ad, Bd, Cd, s = dra(n, dt0, nh)
    lam, V = np.linalg.eig(Ad)
    assert np.all(lam.real > 0) and np.abs(lam.imag).max() < 1e-9
    Ac = (V @ np.diag(np.log(lam.real) / dt0) @ np.linalg.inv(V)).real
    Bc = np.linalg.solve(Ad - np.eye(n), Ac @ Bd)
    return Ac, Bc, Cd


def step_cont(Ac, Bc, Cd, lags):
    w, U = np.linalg.eig(Ac); Bm = np.linalg.solve(U, Bc); Cm = Cd @ U
    return -3 * lags + np.real((Cm * Bm / w)[None, :] * np.expm1(np.outer(lags, w))).sum(1)


def gain_at(S_model, dtau, nst=1800):
    lags = np.arange(nst) * dtau
    Sx = CM.HT.S_exact(lags); Sx[0] = 0.0
    d = Sx - S_model
    return float(abs(d[0]) + np.abs(np.diff(d)).sum())


if __name__ == '__main__':
    OUT = {'cheb': [], 'dra': [], 'vs_dtau': {}}
    dtau = 2.8e-4; lags = np.arange(1800) * dtau
    for Nc in [6, 8, 10, 12, 16, 20, 24, 32, 40]:
        A, b, C, d, ns = cheb_sys(Nc)
        ev = np.linalg.eigvals(A)
        S = step_lin(A, b, C, d, lags)
        g = gain_at(S, dtau)
        OUT['cheb'].append(dict(Nc=Nc, states=ns, gain=g, jump=float(-d), stiff=float(np.max(np.abs(ev)))))
        print(f'Chebyshev Nc={Nc:3d}: {ns:3d} states  gain {g:.2e}  surface jump S(0+)={d:+.3e}  |lambda|max {np.max(np.abs(ev)):.1e}')
    for n in [2, 3, 4, 6, 8, 10, 12]:
        Ad, Bd, Cd, s = dra(n, dtau)
        S = step_dra(Ad, Bd, Cd, lags)
        g = gain_at(S, dtau)
        OUT['dra'].append(dict(n=n, states=n + 1, gain=g, rho=float(np.max(np.abs(np.linalg.eigvals(Ad))))))
        print(f'DRA n={n:2d} (+1 integrator): gain {g:.2e}  spectral radius {np.max(np.abs(np.linalg.eigvals(Ad))):.6f}')
    for dt in [2.8e-5, 2.8e-4, 2.8e-3, 2.8e-2]:
        lg = np.arange(1800) * dt; row = {}
        A, b, C, d, ns = cheb_sys(20)
        row[f'Chebyshev Nc=20 ({ns} states)'] = gain_at(step_lin(A, b, C, d, lg), dt)
        Ad, Bd, Cd, s = dra(10, dt)
        row['DRA n=10 (11 states)'] = gain_at(step_dra(Ad, Bd, Cd, lg), dt)
        for (n0, dt0, nh0, lab) in [(10, 2.8e-4, 1500, 'DRA n=10 realized at 2.8e-4, used at this interval'),
                                    (10, 2.8e-5, 3000, 'DRA n=10 realized at 2.8e-5 (long horizon)')]:
            row[lab] = gain_at(step_cont(*dra_cont(n0, dt0, nh0), lg), dt)
        OUT['vs_dtau'][str(dt)] = row
        print(dt, row)
    json.dump(OUT, open(ROOT + '/ext/out/compare_tf3.json', 'w'), indent=1)
