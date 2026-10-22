"""Quick check (about 1 minute, no battery data needed): recomputes key numbers of the paper from scratch and compares
them with the values printed in the paper.  Run:  python verify_quick.py"""
import os, sys, io, contextlib, json
import numpy as np
ROOT = os.environ.get('LIION_ROOT', os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'ext')); sys.path.insert(0, os.path.join(ROOT, 'battery'))
with contextlib.redirect_stdout(io.StringIO()):
    import hybrid as H
import compare_tf3 as T3

ok_all = True


def check(name, value, expected, rtol):
    global ok_all
    ok = abs(value - expected) <= rtol * abs(expected)
    ok_all &= ok
    print(f"{'OK  ' if ok else 'FAIL'} {name:58s} computed {value:.6e}   paper {expected:.6e}")


print('Proposition 1 (FV step error), N = 10')
N = 10; h = 1 / N
check('E_N(0+) = h/2', -H.S_fv(N, [0.0])[0], h / 2, 1e-12)
tau = np.array([50.0])
check('E_N(inf) = h^2/3 - h^4/30', (H.S_exact(tau) - H.S_fv(N, tau))[0], h ** 2 / 3 - h ** 4 / 30, 1e-6)

print('Exact solution')
check('sum_n a_n sin(lambda_n) = 1/5 (so S(0) = 0)', float(np.sum(H.AN * np.sin(H.LAM))) + 2 / (np.pi ** 2 * 800.5), 0.2, 1e-4)
check('a_n sin(lambda_n) = 2/lambda_n^2 (n = 1)', float(H.AN[0] * np.sin(H.LAM[0])), 2 / H.LAM[0] ** 2, 1e-8)

print('Theorem 1: sharp worst-case gain of plain FV, N = 10, dtau = 2.8e-4, 1800 samples')
dtau = 2.8e-4; lags = np.arange(1800) * dtau
Sx = H.S_exact(lags); Sx[0] = 0.0
d = Sx - H.S_fv(N, lags)
G = abs(d[0]) + np.abs(np.diff(d)).sum()
check('G_FV (Table 1)', G, 9.683178e-2, 1e-5)

print('Per-mesh hybrid correction, N = 10, K = 16')
H.MU = np.logspace(-1, 6.5, 16)
cf = H.fit_correction(N, tau_min=dtau / 20)
Sh = H.S_fv(N, lags) + cf['Einf'] + np.exp(-np.outer(lags, H.MU)) @ cf['w']
dh = Sx - Sh
check('G_hybrid (Table 1, 2.5e-4)', abs(dh[0]) + np.abs(np.diff(dh)).sum(), 2.5e-4, 0.1)

print('Universal correction (Appendix B)')
U = json.load(open(os.path.join(ROOT, 'ext', 'out', 'hybrid_theory.json')))['universal']
check('sum of universal weights = 1/2', float(np.sum(U['w'])), 0.5, 1e-10)

print('Comparison baselines (Fig. 3, Table 2)')
A, b, C, dd, ns = T3.cheb_sys(20)
check('Chebyshev Nc = 20 (9 states): gain', T3.gain_at(T3.step_lin(A, b, C, dd, lags), dtau), 1.64e-2, 0.02)
Ad, Bd, Cd, s = T3.dra(6, dtau)
check('DRA n = 6 (7 states): gain', T3.gain_at(T3.step_dra(Ad, Bd, Cd, lags), dtau), 1.98e-4, 0.05)

print('\nALL CHECKS PASSED' if ok_all else '\nSOME CHECKS FAILED')
