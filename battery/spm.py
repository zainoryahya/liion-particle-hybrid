"""
Single Particle Model (SPM) for a lithium-ion cell.
Forward model for the parameter identification inverse problem.

Equations (each electrode k = n, p):
    dc/dt = (1/r^2) d/dr ( D r^2 dc/dr ),   0 < r < R
    dc/dr = 0                     at r = 0
    -D dc/dr = j_k                at r = R   (molar flux, mol m^-2 s^-1)
    j_n = +I / (F a_n L_n A),  j_p = -I / (F a_p L_p A)     (I > 0 = discharge)

Terminal voltage:
    V = U_p(th_p) - U_n(th_n) + eta_p - eta_n - I R_c
    eta_k = (2RT/F) asinh( i_k / (2 j0_k) ),  i_k = F j_k
    j0_k = m_k * c_e^0.5 * cs^0.5 * (cmax - cs)^0.5

Discretisation: finite volume method (mass-conserving) on N uniform spherical
shells; linear ODE system dc/dt = M c + b j(t), solved with BDF (stiff).

OCP parameters and reference values: Marquis et al. (2019) / Dualfoil set (MCMB2528 graphite / LiCoO2).
"""
import numpy as np
from scipy.integrate import solve_ivp

F = 96485.33212
Rg = 8.314462618


def U_n(sto):
    sto = np.clip(sto, 1e-6, 1 - 1e-6)
    return (0.194 + 1.5 * np.exp(-120.0 * sto)
            + 0.0351 * np.tanh((sto - 0.286) / 0.083)
            - 0.0045 * np.tanh((sto - 0.849) / 0.119)
            - 0.035 * np.tanh((sto - 0.9233) / 0.05)
            - 0.0147 * np.tanh((sto - 0.5) / 0.034)
            - 0.102 * np.tanh((sto - 0.194) / 0.142)
            - 0.022 * np.tanh((sto - 0.9) / 0.0164)
            - 0.011 * np.tanh((sto - 0.124) / 0.0226)
            + 0.0155 * np.tanh((sto - 0.105) / 0.029))


def U_p(sto):
    sto = 1.062 * np.clip(sto, 1e-6, 1 - 1e-6)
    return (2.16216 + 0.07645 * np.tanh(30.834 - 54.4806 * sto)
            + 2.1581 * np.tanh(52.294 - 50.294 * sto)
            - 0.14169 * np.tanh(11.0923 - 19.8543 * sto)
            + 0.2051 * np.tanh(1.4684 - 5.4888 * sto)
            + 0.2531 * np.tanh((-sto + 0.56478) / 0.1316)
            - 0.02167 * np.tanh((sto - 0.525) / 0.006))


# Base parameters (Marquis 2019), with electrode area A chosen for a ~2 Ah cell (18650 size)
BASE = dict(
    R_n=1e-5, R_p=1e-5,                 # particle radius [m]
    L_n=1e-4, L_p=1e-4,                 # electrode thickness [m]
    eps_n=0.6, eps_p=0.5,               # active material volume fraction
    cmax_n=24983.2619938437, cmax_p=51217.9257309275,  # [mol/m3]
    th_n0=0.8, th_p0=0.6,               # initial stoichiometry (full charge)
    c_e=1000.0, T=298.15,
    A=0.0284,                            # electrode area [m2]
    # --- parameters to be estimated ---
    D_n=3.9e-14, D_p=1e-13,             # diffusion coefficient [m2/s]
    m_n=2e-5, m_p=6e-7,                 # reaction rate constant [(A/m2)(m3/mol)^1.5]
    R_c=0.0,                            # lumped contact/ohmic resistance [ohm]
)


class SPM:
    def __init__(self, N=20, **over):
        self.N = N
        self.P = dict(BASE)
        self.P.update(over)
        # finite volume grid: shells [r_{i-1/2}, r_{i+1/2}] on the unit sphere
        e = np.linspace(0.0, 1.0, N + 1)           # edges (dimensionless, r/R)
        self.edges = e
        self.vol = (e[1:] ** 3 - e[:-1] ** 3) / 3.0  # volume/(4 pi)
        self.area = e[1:-1] ** 2                     # interior edge area/(4 pi)
        self.dr = 1.0 / N
        # dimensionless diffusion matrix: dc/dt = (D/R^2) K c
        K = np.zeros((N, N))
        for i in range(N - 1):
            g = self.area[i] / self.dr
            K[i, i] -= g / self.vol[i]
            K[i, i + 1] += g / self.vol[i]
            K[i + 1, i + 1] -= g / self.vol[i + 1]
            K[i + 1, i] += g / self.vol[i + 1]
        self.K = K
        self.bsurf = np.zeros(N)
        self.bsurf[-1] = -1.0 / self.vol[-1]         # outward flux at r = 1 (area = 1)

    def set(self, **kw):
        self.P.update(kw)
        return self

    def capacity_Ah(self):
        P = self.P
        return F * P['eps_p'] * P['L_p'] * P['A'] * P['cmax_p'] / 3600.0

    def simulate(self, t, I, rtol=1e-8, atol=1e-10, return_states=False, method='expm'):
        """t: time grid [s]; I: current [A] at t (piecewise constant, value I[k] on [t_k, t_{k+1})).
        method='expm': exact solution of the linear system dc/dt = M c + b I (exponential integrator);
        method='bdf' : stiff BDF solver (scipy), for cross-checking."""
        P = self.P
        N = self.N
        t = np.asarray(t, float)
        I = np.broadcast_to(np.asarray(I, float), t.shape).copy()
        a_n = 3 * P['eps_n'] / P['R_n']
        a_p = 3 * P['eps_p'] / P['R_p']
        # state variables: theta (c/cmax) for n and p
        kn = P['D_n'] / P['R_n'] ** 2
        kp = P['D_p'] / P['R_p'] ** 2
        gn = 1.0 / (F * a_n * P['L_n'] * P['A'] * P['cmax_n'] * P['R_n'])   # j/(cmax) * (1/R) per ampere
        gp = -1.0 / (F * a_p * P['L_p'] * P['A'] * P['cmax_p'] * P['R_p'])
        M = np.zeros((2 * N, 2 * N))
        M[:N, :N] = kn * self.K
        M[N:, N:] = kp * self.K
        b = np.concatenate([gn * self.bsurf, gp * self.bsurf])

        # solve piece by piece where the current changes (avoid jumps inside BDF)
        x0 = np.concatenate([np.full(N, P['th_n0']), np.full(N, P['th_p0'])])
        X = np.empty((len(t), 2 * N))
        X[0] = x0
        if method == 'expm':
            from scipy.linalg import expm
            dts = np.diff(t)
            cache = {}
            # n and p blocks are uncoupled: exponentiate each block separately
            def prop(h):
                key = round(h, 9)
                if key not in cache:
                    out = []
                    for kk, g in [(kn, gn), (kp, gp)]:
                        Aug = np.zeros((N + 1, N + 1))
                        Aug[:N, :N] = kk * self.K
                        Aug[:N, N] = g * self.bsurf
                        Ea = expm(Aug * h)
                        out.append((Ea[:N, :N], Ea[:N, N]))
                    cache[key] = out
                return cache[key]
            x = x0.copy()
            for k, h in enumerate(dts):
                (En, Gn), (Ep, Gp) = prop(h)
                x = np.concatenate([En @ x[:N] + Gn * I[k], Ep @ x[N:] + Gp * I[k]])
                X[k + 1] = x
            brk = None
        else:
            brk = np.flatnonzero(np.diff(I) != 0) + 1
        segs = [] if brk is None else np.split(np.arange(len(t)), brk)
        x = x0
        for s in segs:
            i0 = s[0]
            i1 = s[-1] + 1 if s[-1] + 1 < len(t) else s[-1]
            Ik = I[i0]
            if i1 == i0:
                X[i0] = x
                continue
            tt = t[i0:i1 + 1]
            sol = solve_ivp(lambda _t, y: M @ y + b * Ik, (tt[0], tt[-1]), x,
                            method='BDF', t_eval=tt, jac=M, rtol=rtol, atol=atol)
            X[i0:i1 + 1] = sol.y.T
            x = sol.y[:, -1]
        th_n = X[:, :N]
        th_p = X[:, N:]
        # surface concentration: half-cell extrapolation using the flux boundary condition
        jn = I / (F * a_n * P['L_n'] * P['A'])
        jp = -I / (F * a_p * P['L_p'] * P['A'])
        hs = 0.5 * self.dr * P['R_n']
        ths_n = th_n[:, -1] - jn * hs / (P['D_n'] * P['cmax_n'])
        hs = 0.5 * self.dr * P['R_p']
        ths_p = th_p[:, -1] - jp * hs / (P['D_p'] * P['cmax_p'])
        ths_n = np.clip(ths_n, 1e-6, 1 - 1e-6)
        ths_p = np.clip(ths_p, 1e-6, 1 - 1e-6)
        j0n = P['m_n'] * np.sqrt(P['c_e']) * np.sqrt(ths_n * P['cmax_n']) * np.sqrt(P['cmax_n'] * (1 - ths_n))
        j0p = P['m_p'] * np.sqrt(P['c_e']) * np.sqrt(ths_p * P['cmax_p']) * np.sqrt(P['cmax_p'] * (1 - ths_p))
        RT2F = 2 * Rg * P['T'] / F
        eta_n = RT2F * np.arcsinh(F * jn / (2 * j0n))
        eta_p = RT2F * np.arcsinh(F * jp / (2 * j0p))
        V = U_p(ths_p) - U_n(ths_n) + eta_p - eta_n - I * P['R_c']
        if return_states:
            return V, dict(ths_n=ths_n, ths_p=ths_p, eta_n=eta_n, eta_p=eta_p, th_n=th_n, th_p=th_p)
        return V


# ---------------------------------------------------------------------------
# Version with measured temperature as input (Arrhenius for D and m, varying RT/F)
EA = dict(D_n=42770.0, D_p=18550.0, m_n=37480.0, m_p=39570.0)   # activation energy [J/mol], Marquis 2019
T_REF = 298.15


def simulate_T(model, t, I, T, I_pt=None):
    """t [s], I [A] (+ = discharge, constant on [t_k,t_{k+1})), T [K] measured at t.
    I_pt: measured current at points t_k (for the voltage); default = I."""
    from scipy.linalg import expm
    P, N = model.P, model.N
    t = np.asarray(t, float); I = np.asarray(I, float); T = np.asarray(T, float)
    arr = lambda key, TT: np.exp(EA[key] / Rg * (1.0 / T_REF - 1.0 / TT))
    a_n = 3 * P['eps_n'] / P['R_n']; a_p = 3 * P['eps_p'] / P['R_p']
    gn = 1.0 / (F * a_n * P['L_n'] * P['A'] * P['cmax_n'] * P['R_n'])
    gp = -1.0 / (F * a_p * P['L_p'] * P['A'] * P['cmax_p'] * P['R_p'])
    xn = np.full(N, P['th_n0']); xp = np.full(N, P['th_p0'])
    Xn = np.empty((len(t), N)); Xp = np.empty((len(t), N))
    Xn[0] = xn; Xp[0] = xp
    Aug = np.zeros((N + 1, N + 1))
    for k in range(len(t) - 1):
        h = t[k + 1] - t[k]
        Tm = 0.5 * (T[k] + T[k + 1])
        for which in (0, 1):
            if which == 0:
                kk = P['D_n'] * arr('D_n', Tm) / P['R_n'] ** 2; g = gn; x = xn
            else:
                kk = P['D_p'] * arr('D_p', Tm) / P['R_p'] ** 2; g = gp; x = xp
            Aug[:N, :N] = kk * model.K; Aug[:N, N] = g * model.bsurf; Aug[N, :] = 0
            E = expm(Aug * h)
            xnew = E[:N, :N] @ x + E[:N, N] * I[k]
            if which == 0: xn = xnew
            else: xp = xnew
        Xn[k + 1] = xn; Xp[k + 1] = xp
    # voltage at each point uses the measured current at that point
    I = I if I_pt is None else np.asarray(I_pt, float)
    Dn = P['D_n'] * arr('D_n', T); Dp = P['D_p'] * arr('D_p', T)
    jn = I / (F * a_n * P['L_n'] * P['A']); jp = -I / (F * a_p * P['L_p'] * P['A'])
    ths_n = np.clip(Xn[:, -1] - jn * 0.5 * model.dr * P['R_n'] / (Dn * P['cmax_n']), 1e-6, 1 - 1e-6)
    ths_p = np.clip(Xp[:, -1] - jp * 0.5 * model.dr * P['R_p'] / (Dp * P['cmax_p']), 1e-6, 1 - 1e-6)
    j0n = P['m_n'] * arr('m_n', T) * np.sqrt(P['c_e']) * P['cmax_n'] * np.sqrt(ths_n * (1 - ths_n))
    j0p = P['m_p'] * arr('m_p', T) * np.sqrt(P['c_e']) * P['cmax_p'] * np.sqrt(ths_p * (1 - ths_p))
    RT2F = 2 * Rg * T / F
    eta_n = RT2F * np.arcsinh(F * jn / (2 * j0n)); eta_p = RT2F * np.arcsinh(F * jp / (2 * j0p))
    V = U_p(ths_p) - U_n(ths_n) + eta_p - eta_n - I * P['R_c']
    return V, dict(ths_n=ths_n, ths_p=ths_p, eta_n=eta_n, eta_p=eta_p, j0n=j0n, j0p=j0p, a_n=a_n, a_p=a_p)
