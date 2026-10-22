"""
Cell model of CALCE INR18650-20R (SP20-2) for the DST/FUDS/US06/BJDST dynamic profiles.
Lumped physical model (Bizeray-style grouped parameters):
  particles n, p : spherical diffusion, time constant tau_k = R_k^2/D_k [s]; three solvers:
               'exact' (Laplace/eigen series, Duhamel), 'fv' (finite volume N), 'hybrid' (FV N + K=16 correction)
  kinetics   : eta_k = (2RT/F) asinh( I / (2 I0_k sqrt(th(1-th))) )   [I0_k in A]
  ohmic      : R0 I ;  electrolyte polarisation (SPMe ROM r_e=1): v1' = (-v1 + R1 I)/tau1
  OCP        : hybrid (Mohtat2020 graphite + positive from CALCE measured OCV), fixed electrode windows
  capacity   : Q = c * Q_C20(SP20-1)
All linear parts are integrated EXACTLY for piecewise-constant current (Duhamel convolution with the step response, FFT).
"""
import os; ROOT = os.environ.get('LIION_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
import numpy as np, json, sys, io, contextlib
from scipy.signal import fftconvolve
sys.path.insert(0, ROOT + '/ext')
with contextlib.redirect_stdout(io.StringIO()):
    import hybrid as H
from scipy.special import gamma
from scipy.linalg import eigh
sys.path.insert(0, ROOT + '/battery')
from spm import SPM


class HT:          # copy of functions from ext/hybrid_theory2.py (avoids running that script)
    @staticmethod
    def S_short_full(tau, nterms=40):
        n = np.arange(nterms)
        return -(tau[:, None] ** ((n + 1) / 2) / gamma((n + 3) / 2)).sum(1)

    @staticmethod
    def S_exact(tau, sw=5e-3):
        tau = np.atleast_1d(tau).astype(float); out = np.zeros_like(tau)
        m = (tau > 0) & (tau < sw); out[m] = HT.S_short_full(tau[m]); m2 = tau >= sw
        if m2.any():
            # eigen series: only modes not yet decayed (large lag -> linear term only)
            t2 = tau[m2]; out2 = -3 * t2 - 0.2
            for i0 in range(0, len(t2), 4000):
                tt = t2[i0:i0 + 4000]
                out2[i0:i0 + 4000] += (H.AN * np.sin(H.LAM) * np.exp(-np.minimum(np.outer(tt, H.LAM ** 2), 700))).sum(1)
            out[m2] = out2
        return out

    _fv = {}
    @staticmethod
    def S_fv(N, tau):
        if N not in HT._fv:
            m = SPM(N=N); Mv = np.diag(m.vol); S = Mv @ m.K; S = 0.5 * (S + S.T)
            lam, V = eigh(S, Mv); beta = V.T @ Mv @ m.bsurf
            HT._fv[N] = (lam, beta * V[-1, :], -0.5 * m.dr)
        lam, wgt, D = HT._fv[N]
        tau = np.atleast_1d(tau).astype(float)
        nz = np.abs(lam) > 1e-12
        out = np.full(len(tau), D) + (tau * wgt[~nz].sum())
        out += (np.expm1(np.outer(tau, lam[nz])) / lam[nz]) @ wgt[nz]
        return out
F, Rg = 96485.33212, 8.314462618
OCP = json.load(open(ROOT + '/calce/hybrid_ocp.json'))
W = OCP['window']; Q20 = OCP['Qtot_C20']
SP, UPt = np.array(OCP['sto_p']), np.array(OCP['Up']); SN, UNt = np.array(OCP['sto_n']), np.array(OCP['Un'])


def _interp_lin(x, xp, fp):
    y = np.interp(x, xp, fp)
    lo = x < xp[0]; hi = x > xp[-1]
    if lo.any():
        y[lo] = fp[0] + (x[lo] - xp[0]) * (fp[3] - fp[0]) / (xp[3] - xp[0])
    if hi.any():
        y[hi] = fp[-1] + (x[hi] - xp[-1]) * (fp[-1] - fp[-4]) / (xp[-1] - xp[-4])
    return y


def Up(th): return _interp_lin(th, SP, UPt)
def Un(th): return _interp_lin(th, SN, UNt)


_CF = {}
def kernel(kind, N, lags):
    """surface step response (unit flux) at dimensionless lags."""
    if kind == 'exact':
        S = HT.S_exact(lags); S[0] = 0.0; return S
    S = HT.S_fv(N, lags)
    if kind == 'hybrid':
        if N not in _CF:
            _CF[N] = H.fit_correction(N, tau_min=1e-7)
        cf = _CF[N]
        S = S + cf['Einf'] + np.exp(-np.outer(lags, H.MU)) @ cf['w']
    return S


NAMES = ['tau_n', 'tau_p', 'I0_n', 'I0_p', 'R0', 'R1', 'tau1', 'c', 'soc0']
LOG = [True, True, True, True, True, True, True, False, False]


def unpack(x):
    return {n: (10 ** v if lg else v) for n, v, lg in zip(NAMES, x, LOG)}   # x may omit soc0


def simulate(x, d, kind='hybrid', N=10, T=298.15, parts=False):
    """d: dict t (uniform dt), I (A, + discharge), soc0. Voltage at t_k uses I_k."""
    p = unpack(x); I = d['I']; dt = d['dt']; n = len(I)
    Qc = p['c'] * Q20
    Qn = Qc / (W['n100'] - W['n0']); Qp = Qc / (W['p0'] - W['p100'])
    soc0 = p.get('soc0', d['soc0'])
    thn0 = W['n0'] + soc0 * (W['n100'] - W['n0']); thp0 = W['p0'] - soc0 * (W['p0'] - W['p100'])
    dI = np.diff(np.r_[0.0, I]); m = np.arange(n)
    ths = {}
    for el, Qe, th0, sg in [('n', Qn, thn0, 1.0), ('p', Qp, thp0, -1.0)]:
        tau = p['tau_' + el]
        # dimensionless flux: q = sg * I * tau / (3 * 3600 * Qe);  d(theta_bar)/d(tau) = -3 q
        g = sg * tau / (3 * 3600 * Qe)
        K = kernel(kind, N, m * dt / tau)
        ths[el] = th0 + g * fftconvolve(dI, K)[:n]
    RT = Rg * T
    sq = lambda th: np.sqrt(np.clip(th * (1 - th), 1e-6, None))
    eta_n = 2 * RT / F * np.arcsinh(I / (2 * p['I0_n'] * sq(ths['n'])))
    eta_p = 2 * RT / F * np.arcsinh(I / (2 * p['I0_p'] * sq(ths['p'])))
    a = np.exp(-dt / p['tau1'])
    from scipy.signal import lfilter
    v1 = lfilter([0, p['R1'] * (1 - a)], [1, -a], I)
    V = Up(ths['p']) - Un(ths['n']) - eta_p - eta_n - p['R0'] * I - v1
    if parts:
        return V, dict(ths=ths, eta=eta_n + eta_p, v1=v1)
    return V


def load_test(key, D=None, dt=0.2):
    """Data from the start of step 4 (rest after CV) to the end of step 7. Current is held constant between samples
    on a fine grid dt; model voltage is evaluated at the grid indices matching the data sample times (no V interpolation)."""
    if D is None:
        D = np.load(ROOT + '/calce/dyn/dyn.npz')
    A = D[key]
    A = A[np.argsort(A[:, 0], kind='stable')]
    A = A[(A[:, 1] >= 4) & (A[:, 1] <= 7)]
    tt = A[:, 0] - A[0, 0]
    tg = np.arange(0, tt[-1] + dt, dt)
    idx = np.searchsorted(tt, tg + 1e-9, side='right') - 1
    I = -A[idx, 2]                                   # Arbin + charge -> our + discharge
    gi = np.minimum(np.round(tt / dt).astype(int), len(tg) - 1)
    # ensure current at the evaluation indices = that sample's current
    I[gi] = -A[:, 2]
    return dict(t=tg, I=I, dt=dt, soc0=1.0, key=key, gi=gi, Vd=A[:, 3], Id=-A[:, 2], td=tt, step=A[:, 1], mask=(A[:, 1] == 7))
