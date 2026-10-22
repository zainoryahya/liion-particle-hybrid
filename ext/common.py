"""Shared utilities: fitted NASA model (Mohtat2020), dimensionless diffusion system, voltage from surface concentration."""
import os; ROOT = os.environ.get('LIION_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
import numpy as np, json, sys
sys.path.insert(0, ROOT + '/battery'); sys.path.insert(0, ROOT + '/nasa')
import spm
from spm import SPM, F, Rg, EA, T_REF
from scipy.linalg import expm, eigh, solve_continuous_lyapunov, svd, cholesky

TAB = np.load(ROOT + '/nasa/ocp_tables.npy', allow_pickle=True).item()
STO = TAB['sto']; TB = TAB['tables']['Mohtat2020']
UN = lambda s: np.interp(s, STO, TB['n'])
UP = lambda s: np.interp(s, STO, TB['p'])
dUN = lambda s: np.interp(s, STO, np.gradient(TB['n'], STO))
dUP = lambda s: np.interp(s, STO, np.gradient(TB['p'], STO))
NAMES = ['D_n', 'D_p', 'm_n', 'm_p', 'R_c', 'eps_n', 'eps_p', 'th_n0', 'th_p0']
LOG = [True] * 5 + [False] * 4
Q_HAT = np.array(json.load(open(ROOT + '/nasa/refine_1.json'))['q'])


def params(q=Q_HAT):
    p = dict(spm.BASE); p.update(A=0.0834, cmax_n=TB['cmax_n'], cmax_p=TB['cmax_p'])
    p.update({n: (10 ** v if lg else v) for n, v, lg in zip(NAMES, q, LOG)})
    p['a_n'] = 3 * p['eps_n'] / p['R_n']; p['a_p'] = 3 * p['eps_p'] / p['R_p']
    p['gn'] = 1.0 / (F * p['a_n'] * p['L_n'] * p['A'] * p['cmax_n'] * p['R_n'])
    p['gp'] = -1.0 / (F * p['a_p'] * p['L_p'] * p['A'] * p['cmax_p'] * p['R_p'])
    p['Qn'] = F * p['eps_n'] * p['L_n'] * p['A'] * p['cmax_n'] / 3600
    p['Qp'] = F * p['eps_p'] * p['L_p'] * p['A'] * p['cmax_p'] / 3600
    return p


def arr(key, T):
    return np.exp(EA[key] / Rg * (1.0 / T_REF - 1.0 / np.asarray(T, float)))


def voltage(p, ths_n, ths_p, I, T):
    """Terminal voltage from surface stoichiometry (same as spm.simulate_T)."""
    ths_n = np.clip(ths_n, 1e-6, 1 - 1e-6); ths_p = np.clip(ths_p, 1e-6, 1 - 1e-6)
    jn = I / (F * p['a_n'] * p['L_n'] * p['A']); jp = -I / (F * p['a_p'] * p['L_p'] * p['A'])
    j0n = p['m_n'] * arr('m_n', T) * np.sqrt(p['c_e']) * p['cmax_n'] * np.sqrt(ths_n * (1 - ths_n))
    j0p = p['m_p'] * arr('m_p', T) * np.sqrt(p['c_e']) * p['cmax_p'] * np.sqrt(ths_p * (1 - ths_p))
    RT2F = 2 * Rg * np.asarray(T, float) / F
    eta_n = RT2F * np.arcsinh(F * jn / (2 * j0n)); eta_p = RT2F * np.arcsinh(F * jp / (2 * j0p))
    return UP(ths_p) - UN(ths_n) + eta_p - eta_n - I * p['R_c']


# ---------------- dimensionless diffusion system: x' = K x + b u, y = C x + Dff u  (y = surface theta)
def full_system(N=20):
    m = SPM(N=N)
    C = np.zeros((1, N)); C[0, -1] = 1.0
    return dict(A=m.K.copy(), B=m.bsurf.reshape(-1, 1), C=C, D=np.array([[-0.5 * m.dr]]), vol=m.vol, N=N)


class LinSurf:
    """Linear surface model (full or reduced) in dimensionless time, uniform initial state theta0."""
    def __init__(self, A, B, C, D, x_unit, name):
        self.A, self.B, self.C, self.D = A, B, C, D
        self.x_unit = x_unit         # state representing the uniform profile theta=1
        self.name = name; self.n = A.shape[0]

    def run(self, theta0, t, u_scale, k_scale):
        """u_scale[k]: dimensionless input u on interval k; k_scale[k] = D/R^2 on interval k.
        Returns y at the points t (using point value u = current u_scale)."""
        n = self.n
        x = self.x_unit * theta0
        Y = np.empty(len(t))
        Aug = np.zeros((n + 1, n + 1)); Aug[:n, :n] = self.A; Aug[:n, n] = self.B[:, 0]
        for k in range(len(t)):
            Y[k] = (self.C @ x)[0] + self.D[0, 0] * u_scale[k]
            if k < len(t) - 1:
                E = expm(Aug * k_scale[k] * (t[k + 1] - t[k]))
                x = E[:n, :n] @ x + E[:n, n] * u_scale[k]
        return Y


def modal_form(sysd):
    """M-orthonormal modal form: z = V^T M x ; mode 0 = mass (lambda=0)."""
    A, B, C, vol = sysd['A'], sysd['B'], sysd['C'], sysd['vol']
    M = np.diag(vol); S = M @ A
    S = 0.5 * (S + S.T)
    lam, V = eigh(S, M)                    # V^T M V = I
    idx = np.argsort(-lam); lam = lam[idx]; V = V[:, idx]   # lam[0] = 0 (mass)
    Bm = V.T @ M @ B; Cm = C @ V
    unit = V.T @ M @ np.ones(len(vol))     # uniform profile in modal coordinates
    return lam, Bm, Cm, unit
