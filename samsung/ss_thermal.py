"""
Samsung 30T SPM with temperature dependence (Arrhenius) using the MEASURED cell temperature.
  tau_k(T) = tau_k * exp( Ea_D/R (1/T - 1/Tref))        (particle diffusion; warped time, exact for uniform T within the particle)
  I0_k(T)  = I0_k  * exp(-Ea_k/R (1/T - 1/Tref))
  R0(T), R1(T) = R * exp( Ea_R/R (1/T - 1/Tref))
Particle: dimensionless time s(t) = int dt / tau(T(t)); flux q = sg I tau(T) / (3*3600*Qe);
modal states updated EXACTLY for q piecewise constant in s (modal recursion, C code).
x = [log tau_n, log tau_p, log I0_n, log I0_p, log R0, log R1, log tau1, c, soc0, EaD, Eak, EaR, dUdT]  (Ea kJ/mol, dUdT mV/K)
"""
import os; ROOT = os.environ.get('LIION_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
import numpy as np, ctypes, sys, io, contextlib
from scipy.signal import lfilter
sys.path.insert(0, ROOT + '/samsung'); sys.path.insert(0, ROOT + '/ext')
import ss_model as S
M = S.M
with contextlib.redirect_stdout(io.StringIO()):
    import hybrid as H

_lib = ctypes.CDLL(ROOT + '/samsung/csrc/libmodrec.so')
_d = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
_lib.modrec.argtypes = [ctypes.c_int, ctypes.c_int, _d, _d, ctypes.c_double, ctypes.c_double, _d, _d, _d, _d]
_lib.modrec.restype = None

NAMES = M.NAMES + ['EaD', 'Eak', 'EaR', 'dUdT']
F, Rg = M.F, M.Rg
_REP = {}


def rep(kind, N=10):
    """S(s) = D + a0 s + sum c (1 - exp(-kap s))."""
    key = (kind, N)
    if key in _REP:
        return _REP[key]
    if kind == 'exact':
        A = H.AN * np.sin(H.LAM)
        # S(0+)=0: fast-mode tail (decays within one step) lumped into one mode with kap=inf, not a direct feedthrough
        r = (0.0, -3.0, np.r_[-A, -0.2 + A.sum()], np.r_[H.LAM ** 2, 1e300])
    else:
        M.HT.S_fv(N, np.array([0.0]))
        lam, w, D = M.HT._fv[N]
        nz = np.abs(lam) > 1e-12
        c = -w[nz] / lam[nz]; kap = -lam[nz]; a0 = w[~nz].sum()
        if kind == 'hybrid':
            if N not in M._CF:
                M._CF[N] = H.fit_correction(N, tau_min=1e-7)
            cf = M._CF[N]
            D = D + cf['Einf'] + cf['w'].sum()
            c = np.r_[c, -cf['w']]; kap = np.r_[kap, H.MU]
        r = (float(D), float(a0), np.ascontiguousarray(c, float), np.ascontiguousarray(kap, float))
    _REP[key] = r
    return r


def surface(q, ds, kind, N):
    D, a0, c, kap = rep(kind, N)
    n = len(q); y = np.empty(n); z = np.empty(len(c))
    _lib.modrec(n, len(c), np.ascontiguousarray(ds, float), np.ascontiguousarray(q, float), D, a0, c, kap, z, y)
    return y


def unpack(x):
    p = M.unpack(x[:9])
    for i, n in enumerate(['EaD', 'Eak', 'EaR', 'dUdT']):
        p[n] = x[9 + i] if len(x) > 9 + i else 0.0
    return p


def simulate(x, d, kind='hybrid', N=10, Tref=298.15, parts=False, Tconst=None):
    p = unpack(x); I = d['I']; dt = d['dt']; n = len(I)
    T = d['T'] if Tconst is None else np.full(n, float(Tconst))
    arr = lambda Ea: np.exp(Ea * 1e3 / Rg * (1.0 / T - 1.0 / Tref))
    fD, fk, fR = arr(p['EaD']), 1.0 / arr(p['Eak']), arr(p['EaR'])
    Qc = p['c'] * M.Q20; W = M.W
    Qn = Qc / (W['n100'] - W['n0']); Qp = Qc / (W['p0'] - W['p100'])
    soc0 = p.get('soc0', d['soc0'])
    thn0 = W['n0'] + soc0 * (W['n100'] - W['n0']); thp0 = W['p0'] - soc0 * (W['p0'] - W['p100'])
    ths = {}
    for el, Qe, th0, sg in [('n', Qn, thn0, 1.0), ('p', Qp, thp0, -1.0)]:
        tau = p['tau_' + el] * fD
        q = sg * I * tau / (3 * 3600 * Qe)
        ds = dt / tau
        ths[el] = th0 + surface(q, ds, kind, N)
    RT = Rg * T
    sq = lambda th: np.sqrt(np.clip(th * (1 - th), 1e-6, None))
    eta = 2 * RT / F * (np.arcsinh(I / (2 * p['I0_n'] * fk * sq(ths['n']))) + np.arcsinh(I / (2 * p['I0_p'] * fk * sq(ths['p']))))
    a = np.exp(-dt / p['tau1'])
    v1 = lfilter([0, (1 - a)], [1, -a], p['R1'] * fR * I)
    V = M.Up(ths['p']) - M.Un(ths['n']) - eta - p['R0'] * fR * I - v1
    V = V + p['dUdT'] * 1e-3 * (T - 298.15)        # effective full-cell entropic coefficient [mV/K]; OCP measured at 25 C
    if parts:
        return V, dict(ths=ths, eta=eta, v1=v1, T=T)
    return V


if __name__ == '__main__':
    # check: with Ea = 0 and constant T, must match the original FFT simulation
    import json, time
    x = np.array(json.load(open(ROOT + '/samsung/fit_UDDS+US06_hybrid10.json'))['x'])
    d = S.load_test('US06', '25degC')
    for kind, N in [('fv', 10), ('hybrid', 10), ('exact', 10)]:
        t0 = time.time(); V1 = simulate(np.r_[x, 0, 0, 0], d, kind, N, Tconst=298.15); t1 = time.time() - t0
        V0 = M.simulate(x, d, kind, N, T=298.15)
        print(f'{kind:6s}: max |diff| ={np.max(np.abs(V1 - V0))*1e6:.3f} uV  ({t1:.2f} s, n={len(V1)})')
