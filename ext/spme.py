"""
Canonical SPMe (Marquis et al. 2019, eq. 40) in dimensional form.
Three independent LINEAR PDEs:
  particle k = n,p :  dc_s/dt = (1/r^2) d/dr (D_k r^2 dc_s/dr),  -D_k dc_s/dr|_R = +-I/(F a_k L_k A)
  electrolyte   :  eps(x) dc_e/dt = d/dx (eps^b D_e dc_e/dx) + (1-t+) * { I/(F A L_n), 0, -I/(F A L_p) }
                   zero flux at x=0, x=L ; continuity of c_e and flux at the interfaces
Voltage (algebraic):
  V = U_p - U_n + eta_p - eta_n + eta_c + dPhi_elec + dPhi_solid - I R_c
  eta_k = (2RT/F) asinh(i_k / (2 jbar0_k)),  jbar0_k = x-average of m_k sqrt(c_e) sqrt(c_s (cmax-c_s))
  eta_c = 2 (1-t+) (RT/F) (cbar_e,p - cbar_e,n)/c_e0
  dPhi_elec = -(I/A)/kappa(c_e0) (L_n/(3 eps_n^b) + L_s/eps_s^b + L_p/(3 eps_p^b))
  dPhi_solid = -(I/(3A)) (L_p/sigma_p + L_n/sigma_n)
Discretisation: finite volume (particle: spherical shells; electrolyte: 3 regions, harmonic mean at faces) ->
linear ODE system x' = A x + b I, solved EXACTLY via the matrix exponential for piecewise-constant current.
"""
import os; ROOT = os.environ.get('LIION_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
import numpy as np, sys
from scipy.linalg import expm
sys.path.insert(0, ROOT + '/battery')
import spm as spmmod
from spm import SPM, F, Rg

DEFAULT_E = dict(eps_en=0.3, eps_es=1.0, eps_ep=0.3, brug=1.5, tplus=0.4, c_e0=1000.0,
                 sigma_n=100.0, sigma_p=10.0, L_s=2.5e-5)


def De_capiglia(c, T=298.15):
    return 5.34e-10 * np.exp(-0.65 * c / 1000) * np.exp(37040 / Rg * (1 / 298.15 - 1 / T))


def kappa_capiglia(c, T=298.15):
    x = c / 1000
    return (0.0911 + 1.9101 * x - 1.052 * x ** 2 + 0.1554 * x ** 3) * np.exp(34700 / Rg * (1 / 298.15 - 1 / T))


class SPMe:
    def __init__(self, Nr=20, Nx=(10, 6, 10), electrolyte=True, **over):
        self.sp = SPM(N=Nr, **{k: v for k, v in over.items() if k in spmmod.BASE})
        self.E = dict(DEFAULT_E); self.E.update({k: v for k, v in over.items() if k in DEFAULT_E})
        self.Nx = Nx; self.electrolyte = electrolyte
        self._build_electrolyte()

    def _build_electrolyte(self, T=298.15):
        P, E = self.sp.P, self.E
        Ln, Ls, Lp = P['L_n'], E['L_s'], P['L_p']
        Nn, Ns, Np = self.Nx
        h = np.r_[np.full(Nn, Ln / Nn), np.full(Ns, Ls / Ns), np.full(Np, Lp / Np)]
        eps = np.r_[np.full(Nn, E['eps_en']), np.full(Ns, E['eps_es']), np.full(Np, E['eps_ep'])]
        De = De_capiglia(E['c_e0'], T)                 # canonical SPMe: D_e evaluated at c_e0
        Deff = eps ** E['brug'] * De
        N = len(h)
        A = np.zeros((N, N))
        for i in range(N - 1):
            Tf = 1.0 / (0.5 * h[i] / Deff[i] + 0.5 * h[i + 1] / Deff[i + 1])
            A[i, i] -= Tf; A[i, i + 1] += Tf; A[i + 1, i + 1] -= Tf; A[i + 1, i] += Tf
        A = A / (eps * h)[:, None]
        src = np.r_[np.full(Nn, 1.0 / Ln), np.zeros(Ns), np.full(Np, -1.0 / Lp)]
        b = (1 - E['tplus']) * src / (F * P['A'] * eps)      # per ampere
        self.Ae, self.be, self.h, self.eps = A, b, h, eps
        self.idx_n = np.arange(Nn); self.idx_p = np.arange(Nn + Ns, N)
        self.De = De

    def _hybrid_corr(self, t, q, kscale):
        """Hybrid correction (Sections 6-7): additional surface output, sum_k x_k + (E_inf + sum w) q."""
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            import hybrid as H
        if getattr(self, '_cf', None) is None:
            dt_min = np.min(np.diff(t)) * kscale
            self._cf = H.fit_correction(self.sp.N, tau_min=dt_min / 20)
        cf = self._cf; mu = H.MU; w = cf['w']
        x = np.zeros(len(mu)); y = np.empty(len(t)); Dc = cf['Einf'] + w.sum()
        for k in range(len(t)):
            y[k] = x.sum() + Dc * q[k]
            if k < len(t) - 1:
                e = np.exp(-mu * kscale * (t[k + 1] - t[k]))
                x = e * x - w * (1 - e) * q[k]
        return y

    def simulate(self, t, I, T=298.15, return_parts=False, particle='fv'):
        """particle: 'fv' (finite volume), 'hybrid' (FV + analytical correction), 'exact' (Duhamel, uniform dt)."""
        """t [s], I [A] constant on [t_k, t_{k+1}) and used for the voltage at t_k (as in spm.simulate)."""
        sp, P, E = self.sp, self.sp.P, self.E
        t = np.asarray(t, float); I = np.broadcast_to(np.asarray(I, float), t.shape).copy()
        N = sp.N
        a_n = 3 * P['eps_n'] / P['R_n']; a_p = 3 * P['eps_p'] / P['R_p']
        kn = P['D_n'] / P['R_n'] ** 2; kp = P['D_p'] / P['R_p'] ** 2
        gn = 1.0 / (F * a_n * P['L_n'] * P['A'] * P['cmax_n'] * P['R_n'])
        gp = -1.0 / (F * a_p * P['L_p'] * P['A'] * P['cmax_p'] * P['R_p'])
        blocks = [(kn * sp.K, gn * sp.bsurf), (kp * sp.K, gp * sp.bsurf)]
        if self.electrolyte:
            blocks.append((self.Ae, self.be))
        x = [np.full(N, P['th_n0']), np.full(N, P['th_p0'])] + ([np.full(len(self.h), E['c_e0'])] if self.electrolyte else [])
        cache = {}
        X = [np.empty((len(t), len(xx))) for xx in x]
        for j in range(len(x)):
            X[j][0] = x[j]
        for k in range(len(t) - 1):
            hdt = round(t[k + 1] - t[k], 9)
            if hdt not in cache:
                cache[hdt] = []
                for (Am, bm) in blocks:
                    n = Am.shape[0]; Aug = np.zeros((n + 1, n + 1)); Aug[:n, :n] = Am; Aug[:n, n] = bm
                    Ex = expm(Aug * hdt); cache[hdt].append((Ex[:n, :n], Ex[:n, n]))
            for j, (Ed, gd) in enumerate(cache[hdt]):
                x[j] = Ed @ x[j] + gd * I[k]
                X[j][k + 1] = x[j]
        # particle surface
        jn = I / (F * a_n * P['L_n'] * P['A']); jp = -I / (F * a_p * P['L_p'] * P['A'])
        hs = 0.5 * sp.dr
        ths_n = np.clip(X[0][:, -1] - jn * hs * P['R_n'] / (P['D_n'] * P['cmax_n']), 1e-6, 1 - 1e-6)
        ths_p = np.clip(X[1][:, -1] - jp * hs * P['R_p'] / (P['D_p'] * P['cmax_p']), 1e-6, 1 - 1e-6)
        qn = jn * P['R_n'] / (P['D_n'] * P['cmax_n']); qp = jp * P['R_p'] / (P['D_p'] * P['cmax_p'])
        if particle == 'hybrid':
            self._cf = None
            yn = self._hybrid_corr(t, qn, kn); self._cf = None
            yp = self._hybrid_corr(t, qp, kp); self._cf = None
            ths_n = np.clip(ths_n + yn, 1e-6, 1 - 1e-6); ths_p = np.clip(ths_p + yp, 1e-6, 1 - 1e-6)
        elif particle == 'exact':
            import io, contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                import hybrid as H
            dt = np.diff(t); assert np.allclose(dt, dt[0]), 'exact requires uniform dt'
            ths_n = np.clip(P['th_n0'] + H.exact_duhamel(qn, kn * dt[0]), 1e-6, 1 - 1e-6)
            ths_p = np.clip(P['th_p0'] + H.exact_duhamel(qp, kp * dt[0]), 1e-6, 1 - 1e-6)
        RT = Rg * T
        if self.electrolyte:
            ce = X[2]
            w_n = self.h[self.idx_n] / self.h[self.idx_n].sum(); w_p = self.h[self.idx_p] / self.h[self.idx_p].sum()
            sq_n = (np.sqrt(np.clip(ce[:, self.idx_n], 1e-6, None)) * w_n).sum(1)
            sq_p = (np.sqrt(np.clip(ce[:, self.idx_p], 1e-6, None)) * w_p).sum(1)
            cbar_n = (ce[:, self.idx_n] * w_n).sum(1); cbar_p = (ce[:, self.idx_p] * w_p).sum(1)
            eta_c = 2 * (1 - E['tplus']) * RT / F * (cbar_p - cbar_n) / E['c_e0']
            kap = kappa_capiglia(E['c_e0'], T); b = E['brug']
            dphi_e = -(I / P['A']) / kap * (P['L_n'] / (3 * E['eps_en'] ** b) + E['L_s'] / E['eps_es'] ** b + P['L_p'] / (3 * E['eps_ep'] ** b))
            dphi_s = -(I / (3 * P['A'])) * (P['L_p'] / E['sigma_p'] + P['L_n'] / E['sigma_n'])
        else:
            sq_n = sq_p = np.sqrt(E['c_e0']); eta_c = dphi_e = dphi_s = 0.0; ce = None
        j0n = P['m_n'] * sq_n * P['cmax_n'] * np.sqrt(ths_n * (1 - ths_n))
        j0p = P['m_p'] * sq_p * P['cmax_p'] * np.sqrt(ths_p * (1 - ths_p))
        eta_n = 2 * RT / F * np.arcsinh(F * jn / (2 * j0n)); eta_p = 2 * RT / F * np.arcsinh(F * jp / (2 * j0p))
        V = spmmod.U_p(ths_p) - spmmod.U_n(ths_n) + eta_p - eta_n + eta_c + dphi_e + dphi_s - I * P['R_c']
        if return_parts:
            return V, dict(eta_c=eta_c, dphi_e=dphi_e, dphi_s=dphi_s, ce=ce, eta_n=eta_n, eta_p=eta_p)
        return V
