"""
Integration of the hybrid scheme into the particle reduced-order model (ROM) used by the EKF/BMS.
ROM-FV : BT-SPA of FV N=20 (as in kalman.py)                  -> approximates FV (which has the h/2 floor)
ROM-hyb: BT-SPA of FV N=20 + K=16 correction states           -> approximates the EXACT solution
The mass mode (particle average, SOC) is kept exact in both.
Reference: exact solution (Duhamel + Laplace/eigen-series). NASA B0005 parameters (Mohtat2020 OCP), 25 C.
"""
import os; ROOT = os.environ.get('LIION_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
import numpy as np, json, sys, io, contextlib
sys.path.insert(0, ROOT + '/ext')
from scipy.linalg import expm, solve_continuous_lyapunov, svd
with contextlib.redirect_stdout(io.StringIO()):
    import hybrid as H
from common import params, voltage, full_system, modal_form

NFV = 20


def sqrt_psd(X):
    w, U = np.linalg.eigh(0.5 * (X + X.T)); return U @ np.diag(np.sqrt(np.clip(w, 0, None)))


def bt_spa(A, B, C, D, r):
    P = solve_continuous_lyapunov(A, -np.outer(B, B)); Q = solve_continuous_lyapunov(A.T, -np.outer(C, C))
    Lp, Lq = sqrt_psd(P), sqrt_psd(Q)
    U, s, Vt = svd(Lq.T @ Lp)
    k = int(np.sum(s > 1e-13 * s[0]))
    S1 = np.diag(s[:k] ** -0.5); T = Lp @ Vt[:k].T @ S1; Ti = S1 @ U[:, :k].T @ Lq.T
    Ab, Bb, Cb = Ti @ A @ T, Ti @ B, C @ T
    if r >= k:
        return Ab, Bb, Cb, D, s
    A11, A12, A21, A22 = Ab[:r, :r], Ab[:r, r:], Ab[r:, :r], Ab[r:, r:]
    X = np.linalg.solve(A22, A21); Y = np.linalg.solve(A22, Bb[r:])
    return A11 - A12 @ X, Bb[:r] - A12 @ Y, Cb[:r] - Cb[r:] @ X, D - Cb[r:] @ Y, s


def deviation_systems(N=NFV, dtau_min=1e-4):
    fs = full_system(N)
    lam, Bm, Cm, unit = modal_form(fs)
    A = np.diag(lam[1:]); B = Bm[1:, 0]; C = Cm[0, 1:]; D = fs['D'][0, 0]
    cf = H.fit_correction(N, tau_min=dtau_min / 20)
    K = len(H.MU)
    Ah = np.zeros((N - 1 + K, N - 1 + K)); Ah[:N - 1, :N - 1] = A; Ah[N - 1:, N - 1:] = np.diag(-H.MU)
    Bh = np.r_[B, -H.MU * cf['w']]; Ch = np.r_[C, np.ones(K)]; Dh = D + cf['Einf'] + cf['w'].sum()
    return dict(FV=(A, B, C, D), HYB=(Ah, Bh, Ch, Dh)), cf


SYS, CF = deviation_systems()


def make_rom(kind, r):
    A, B, C, D = SYS[kind]
    if r is None:
        return A, B, C, D
    Ar, Br, Cr, Dr, _ = bt_spa(A, B, C, D, r)
    return Ar, Br, Cr, Dr


def run_surface(rom, q, dtau):
    """surface output (without theta0) = exact mass mode + deviation ROM; q piecewise constant, output at tau_k uses q_k."""
    A, B, C, D = rom; n = len(B)
    Aug = np.zeros((n + 1, n + 1)); Aug[:n, :n] = A; Aug[:n, n] = B
    E = expm(Aug * dtau); Ad, Bd = E[:n, :n], E[:n, n]
    x = np.zeros(n); y = np.empty(len(q)); mean = 0.0
    for k in range(len(q)):
        y[k] = mean + C @ x + D * q[k]
        x = Ad @ x + Bd * q[k]; mean += -3 * q[k] * dtau
    return y


# ---------------- current profiles (A); NASA cell ~2 Ah
def profiles():
    rng = np.random.default_rng(11)
    P = {}
    Vd = json.load(open(ROOT + '/ext/out/hybrid_voltage.json'))
    P['random BMS (-1C..2C)'] = np.array(Vd['I'])
    t = np.arange(1800)
    P['HPPC-type pulses'] = np.where((t % 120) < 10, 4.0, np.where(((t % 120) >= 50) & ((t % 120) < 60), -3.0, 0.0))
    # synthetic drive cycle: smooth random speed -> power -> current (regenerative braking)
    v = np.zeros(1800); a = 0.0
    for k in range(1, 1800):
        a = 0.9 * a + rng.normal(0, 0.35); v[k] = np.clip(v[k - 1] + a, 0, 30)
        if (k // 300) % 2 == 1 and k % 300 > 240:
            v[k] = max(v[k - 1] - 2.5, 0)
    acc = np.r_[0, np.diff(v)]
    Pw = 1500 * v * acc + 0.5 * 1.2 * 0.6 * v ** 3 + 0.01 * 1500 * 9.81 * v       # W (small vehicle)
    I = Pw / (96 * 3.7) / 20                                                         # 96s20p -> per cell
    P['synthetic drive cycle'] = np.clip(I, -3.0, 6.0)
    return P


if __name__ == '__main__':
    p = params(); OUT = {'rows': {}}
    soc0 = 0.7
    th0 = {'n': p['th_n0'] - (1 - soc0) * 1.8 / p['Qn'], 'p': p['th_p0'] + (1 - soc0) * 1.8 / p['Qp']}
    PR = profiles()
    for name, I in PR.items():
        res = {}
        qs = {el: p['g' + el] * I * p['R_' + el] ** 2 / p['D_' + el] for el in 'np'}
        dts = {el: p['D_' + el] / p['R_' + el] ** 2 for el in 'np'}
        ex = {el: th0[el] + H.exact_duhamel(qs[el], dts[el]) for el in 'np'}
        T = np.full(len(I), 298.15)
        Vex = voltage(p, ex['n'], ex['p'], I, T)
        row = {}
        for kind in ['FV', 'HYB']:
            for r in [2, 3, 4, 6, None]:
                rom = make_rom(kind, r)
                sv = {el: th0[el] + run_surface(rom, qs[el], dts[el]) for el in 'np'}
                Vm = voltage(p, sv['n'], sv['p'], I, T)
                e = np.abs(Vm - Vex) * 1e3
                lab = f"{kind}_r{r if r else 'full'}"
                row[lab] = dict(max_mV=float(e.max()), rms_mV=float(np.sqrt(np.mean(e ** 2))),
                                surf_n_max=float(np.abs(sv['n'] - ex['n']).max()))
                if r == 3:
                    OUT.setdefault('trace', {}).setdefault(name, {})[kind] = (Vm - Vex).tolist()
                print(f'{name:24s} {lab:9s}: voltage max {e.max():7.3f} mV  RMS {np.sqrt(np.mean(e**2)):6.3f} mV', flush=True)
        OUT['rows'][name] = row
        OUT.setdefault('I', {})[name] = I.tolist()
    json.dump(OUT, open(ROOT + '/ext/out/rom_hybrid.json', 'w'))
    print('done')
