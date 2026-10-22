"""
Reviewer request: hybrid correction with concentration-dependent diffusivity D(u) = D0 f(u).
   u_tau = r^-2 (r^2 f(u) u_r)_r ,  -f(u) u_r(1) = q(tau),  tau = D0 t / R^2.
Reference: finite volumes on a mesh graded geometrically towards r = 1 (smallest cell 2e-5), Radau, tight tolerances.
Tested (N = 10, 20, 40):
   FV          : uniform FV, face f = mean of neighbours, surface y = c_N - (h/2) q / f(c_N)
   hybrid-frozen: universal correction with the rates of the reference diffusivity D0:
                  x_k' = -mu_k x_k - mu_k w_k q,       y = c_N + sum x_k   (feedthrough cancels, f only in FV)
   hybrid-scaled: rates scaled by the local surface diffusivity f_s = f(c_N) (local time scale of the layer):
                  x_k' = -mu_k f_s x_k - mu_k w_k q,   y = c_N + sum x_k
   with mu_k = nu_k / h^2, w_k = h omega_k (universal coefficients, computed once).
   Note: for the FV part the extrapolation -(h/2) q/f is built into y through the KKT constraint sum w = h/2, i.e.
   y = c_N - (h/2) q/f_s + [(h/2) q/f_s + sum x_k]; in steady state x_k -> -w_k q / f_s.
"""
import os; ROOT = os.environ.get('LIION_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
import numpy as np, json, time
from scipy.integrate import solve_ivp
from scipy.sparse import diags, bmat, identity

U = json.load(open(ROOT + '/ext/out/hybrid_theory.json'))['universal']
NU, OM = np.array(U['nu']), np.array(U['w'])

import os
DECADES = float(os.environ.get('DC_DECADES', '1.0'))                                    # D varies by 10^DECADES over u in [0.1, 0.9]
def f(u):
    return 10.0 ** (DECADES * (np.clip(u, 0, 1) - 0.5) / 0.8)


def mesh_uniform(N):
    return np.linspace(0, 1, N + 1)


def mesh_graded(N, hmin):
    # geometric cells from the surface inwards
    from scipy.optimize import brentq
    rho = brentq(lambda q: hmin * (q ** N - 1) / (q - 1) - 1, 1 + 1e-9, 2.0)
    w = hmin * rho ** np.arange(N)                # w[0] at the surface
    faces = np.r_[0.0, np.cumsum(w[::-1])]
    faces[-1] = 1.0
    return faces


class FVnl:
    def __init__(self, faces):
        self.f_ = faces; self.N = len(faces) - 1
        self.V = (faces[1:] ** 3 - faces[:-1] ** 3) / 3
        self.c = 0.5 * (faces[1:] + faces[:-1])
        self.dist = np.diff(self.c); self.A = faces[1:-1] ** 2
        self.hs = faces[-1] - self.c[-1]          # centre-to-surface distance

    def rhs(self, c, q):
        fc = f(c); ff = 0.5 * (fc[1:] + fc[:-1])
        flux = self.A * ff * np.diff(c) / self.dist   # r^2 f u_r at interior faces (inward positive)
        dc = np.zeros_like(c); dc[:-1] += flux; dc[1:] -= flux; dc[-1] -= q   # surface face r=1: r^2 = 1, -f u_r = q
        return dc / self.V

    def jac_sp(self):
        n = self.N; return diags([1, 1, 1], [-1, 0, 1], shape=(n, n)).toarray() != 0

    def surf(self, c, q):
        return c[-1] - self.hs * q / f(c[-1])


def simulate(model, q, dtau, kind='fv', h=None):
    N = model.N; K = len(NU)
    mu = NU / h ** 2 if h else None; w = h * OM if h else None
    n = N + (K if kind != 'fv' else 0)
    y = np.empty(len(q)); x = np.full(n, 0.5); x[N:] = 0.0      # start at u = 0.5 uniform
    J = np.zeros((n, n), bool); J[:N, :N] = model.jac_sp()
    if kind != 'fv':
        J[N:, N:] = np.eye(K, dtype=bool); J[N:, N - 1] = True
    for k in range(len(q)):
        if kind == 'fv':
            y[k] = model.surf(x[:N], q[k])
        else:
            y[k] = x[N - 1] + x[N:].sum()
        def rhs(t, z, qk=q[k]):
            dz = np.empty_like(z); dz[:N] = model.rhs(z[:N], qk)
            if kind == 'frozen':
                dz[N:] = -mu * z[N:] - mu * w * qk
            elif kind == 'scaled':
                fs = f(z[N - 1]); dz[N:] = -mu * fs * z[N:] - mu * w * qk
            elif kind == 'scaled_y':                 # local diffusivity at the corrected surface value
                fs = f(z[N - 1] + z[N:].sum()); dz[N:] = -mu * fs * z[N:] - mu * w * qk
            return dz
        sol = solve_ivp(rhs, (0, dtau), x, method='Radau', rtol=1e-10, atol=1e-13, jac_sparsity=J)
        x = sol.y[:, -1]
    return y


if __name__ == '__main__':
    rng = np.random.default_rng(3)
    dtau = 2.8e-4; nst = 600
    q = np.zeros(nst); k = 0
    while k < nst:
        L = rng.integers(1, 30); q[k:k + L] = rng.uniform(-1.0, 1.5); k += L
    OUT = {'q': q.tolist(), 'dtau': dtau, 'decades': DECADES}
    t0 = time.time()
    ref = simulate(FVnl(mesh_graded(300, 2e-5)), q, dtau)
    ref2 = simulate(FVnl(mesh_graded(450, 1e-5)), q, dtau)
    print(f'reference: |ref(300) - ref(450)|_max = {np.abs(ref - ref2).max():.2e}  ({time.time()-t0:.0f} s)', flush=True)
    OUT['ref_check'] = float(np.abs(ref - ref2).max())
    rows = []
    for N in [10, 20, 40]:
        m = FVnl(mesh_uniform(N)); h = 1 / N
        e = {}
        for kind in ['fv', 'frozen', 'scaled', 'scaled_y']:
            y = simulate(m, q, dtau, kind, h)
            e[kind] = float(np.abs(y - ref2).max())
            if N == 10:
                OUT.setdefault('trace', {})[kind] = (y - ref2).tolist()
        rows.append(dict(N=N, **e))
        print(f"N={N:3d}: " + ' | '.join(f'{k} {v:.2e}' for k, v in e.items()), flush=True)
    OUT['rows'] = rows
    for kind in ['fv', 'frozen', 'scaled', 'scaled_y']:
        p = -np.polyfit(np.log([r['N'] for r in rows]), np.log([r[kind] for r in rows]), 1)[0]
        OUT['order_' + kind] = float(p); print(f'order {kind}: {p:.2f}')
    OUT['urange'] = [float(ref2.min()), float(ref2.max())]
    print('surface u range', OUT['urange'], ' f range', f(ref2.min()), f(ref2.max()))
    json.dump(OUT, open(f'{ROOT}/ext/out/dc_test_{DECADES:g}.json', 'w'))
