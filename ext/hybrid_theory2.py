"""
Theory of the hybrid scheme -- numerical verification of Proposition 1, Theorem 2, Theorem 3 (see methodology_hybrid.tex).

E_N(tau) = S(tau) - S_h(tau): surface step-response error (unit flux) for FV with N uniform cells, h = 1/N.
Prop 1 (a) E_N(0+) = h/2 ;  (b) E_N(inf) = h^2/3 - h^4/30 (closed form) ;
       (c) E_N(tau) = h g(tau/h^2) + O(h^2), g = error of the half-line FV scheme (universal), g(0)=1/2, g(s) ~ 1/(8 sqrt(pi s)).
Theorem 2 (error representation & sharp bound): for piecewise-constant q, e_k = sum_j dq_j delta((k-j)dtau) and
       sup_{|q|<=1} |e_k| = |delta(0)| + sum_{m=1}^k |delta(m dtau) - delta((m-1) dtau)|   (an equality, not merely a bound).
Theorem 3 (universal correction): Ehat_N(tau) = h ghat(tau/h^2) with ghat(0) = 1/2 -> uniform O(h^2) error, K states independent of N.
"""
import os; ROOT = os.environ.get('LIION_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
import numpy as np, json, sys, math
from scipy.linalg import eigh
from scipy.special import gamma
sys.path.insert(0, ROOT + '/ext'); sys.path.insert(0, ROOT + '/battery')
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import hybrid as H
from spm import SPM

OUT = {}


# ---------- exact step response (full short-time series + eigen-series)
def S_short_full(tau, nterms=40):
    tau = np.atleast_1d(tau).astype(float)
    n = np.arange(nterms)
    return -(tau[:, None] ** ((n + 1) / 2) / gamma((n + 3) / 2)).sum(1)


def S_exact(tau, sw=5e-3):
    tau = np.atleast_1d(tau).astype(float); out = np.zeros_like(tau)
    m = (tau > 0) & (tau < sw); out[m] = S_short_full(tau[m]); m2 = tau >= sw; out[m2] = H.S_long(tau[m2])
    return out


for tt in [1e-3, 5e-3, 1e-2]:
    print(f'series check: tau={tt:.0e} short(40) {S_short_full(tt)[0]:+.12f} long {H.S_long(tt)[0]:+.12f}')


# ---------- FV via modal form (fast, exact)
def fv_modal(N):
    m = SPM(N=N); Mv = np.diag(m.vol); S = Mv @ m.K; S = 0.5 * (S + S.T)
    lam, V = eigh(S, Mv)
    beta = V.T @ Mv @ m.bsurf            # input in modal coordinates
    cN = V[-1, :]                        # output c_N
    return lam, beta * cN, -0.5 * m.dr


def phi(lam, tau):
    lt = np.outer(tau, lam)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.where(np.abs(lam) < 1e-12, tau[:, None], np.expm1(lt) / np.where(np.abs(lam) < 1e-12, 1, lam))
    return r


def S_fv(N, tau):
    lam, wgt, D = fv_modal(N); tau = np.atleast_1d(tau).astype(float)
    return phi(lam, tau) @ wgt + D


# ---------- g(sigma): half-line FV scheme (unit cells), DCT-II closed form, large number of cells M
Mh = 40000
kk = np.arange(Mh); lam_h = -4 * np.sin(kk * np.pi / (2 * Mh)) ** 2
v1sq = np.where(kk == 0, 1.0 / Mh, 2.0 / Mh * np.cos(kk * np.pi / (2 * Mh)) ** 2)


def g_half(sig):
    sig = np.atleast_1d(sig).astype(float); out = np.empty_like(sig)
    for i0 in range(0, len(sig), 200):
        s = sig[i0:i0 + 200]
        out[i0:i0 + 200] = -2 * np.sqrt(s / np.pi) + 0.5 + phi(lam_h, s) @ v1sq
    return out


sg = np.logspace(-4, 5, 600)
gv = g_half(sg)
print(f'g(1e-4)={gv[0]:.5f} (expect 0.5);  g(1e5)*8*sqrt(pi*1e5) = {gv[-1]*8*np.sqrt(np.pi*1e5):.4f} (expect 1)')
OUT['g'] = dict(sigma=sg.tolist(), g=gv.tolist())

# ---------- Prop 1
rows = []
Ns = [5, 10, 20, 40, 80]
for N in Ns:
    h = 1 / N
    tau = np.r_[0.0, np.logspace(-9, 1.5, 3000)]
    E = S_exact(tau) - S_fv(N, tau)
    Einf_cf = h ** 2 / 3 - h ** 4 / 30
    lam, _, _ = fv_modal(N)
    tail = np.exp(lam[np.argsort(-lam)[1]] * tau[-1])
    hg = h * g_half(tau / h ** 2)
    rows.append(dict(N=N, E0_over_h=float(E[0] / h), Einf=float(E[-1]), Einf_closed=float(Einf_cf),
                     Einf_rel_err=float(abs(E[-1] - Einf_cf) / Einf_cf), sup_E_minus_hg=float(np.abs(E - hg).max())))
    print(f'N={N:3d}: E(0)/h={E[0]/h:.6f}  E(inf)={E[-1]:.6e}  closed form h^2/3-h^4/30={Einf_cf:.6e}  '
          f'|E - h g(tau/h^2)|_inf={np.abs(E-hg).max():.2e}  (h^2={h*h:.2e})')
    OUT.setdefault('collapse', {})[str(N)] = dict(sigma=(tau[1:] / h ** 2).tolist()[::10], E_over_h=(E[1:] / h).tolist()[::10])
OUT['prop1'] = rows
o = -np.polyfit(np.log(Ns), np.log([r['sup_E_minus_hg'] for r in rows]), 1)[0]
OUT['order_sup_E_minus_hg'] = o
print(f'order |E - h g|: {o:.2f}  (expect 2)')

# ---------- universal correction: fit ghat once (half-line), constraint ghat(0) = 1/2
NU = np.logspace(-6, 3, 20)
sf = np.logspace(-5, 6, 700)
gf = g_half(sf[sf <= 1e5]); sf_ = sf[sf <= 1e5]
# asymptotic tail for sigma > 1e5
sf2 = sf[sf > 1e5]; gf2 = 1 / (8 * np.sqrt(np.pi * sf2))
sfit = np.r_[sf_, sf2]; gfit = np.r_[gf, gf2]
Wt = 1 / np.maximum(np.abs(gfit), 1e-6) ** 0.5          # mild relative weighting
Phi = np.exp(-np.outer(sfit, NU)) * Wt[:, None]
KKT = np.block([[2 * Phi.T @ Phi, np.ones((len(NU), 1))], [np.ones((1, len(NU))), np.zeros((1, 1))]])
wU = np.linalg.solve(KKT, np.r_[2 * Phi.T @ (gfit * Wt), 0.5])[:len(NU)]
ghat = lambda s: np.exp(-np.outer(np.atleast_1d(s), NU)) @ wU
print(f'ghat: sum w = {wU.sum():.12f};  max |g - ghat| for sigma<=1e5 = {np.abs(ghat(sf_) - gf).max():.2e}')
OUT['universal'] = dict(nu=NU.tolist(), w=wU.tolist())


# ---------- Theorems 2 & 3 on the random profile (hybrid.json): sharp worst-case gain vs observed error
Hd = json.load(open(ROOT + '/ext/out/hybrid.json'))
q = np.array(Hd['q']); dtau = Hd['dtau']; nst = len(q)
lags = np.arange(nst) * dtau


def duhamel(kernel_vals, q):
    dq = np.diff(np.r_[0.0, q]); y = np.zeros(len(q))
    for j in np.flatnonzero(dq):
        y[j:] += dq[j] * kernel_vals[:len(q) - j]
    return y


def gain(delta):            # sharp l_inf gain (Theorem 2)
    return abs(delta[0]) + np.abs(np.diff(delta)).sum()


Sx = S_exact(lags); Sx[0] = 0.0
ref = duhamel(Sx, q)
qmax = np.abs(q).max()
t2 = []
for N in Ns:
    h = 1 / N
    Sh = S_fv(N, lags)
    E = Sx - Sh
    cf = H.fit_correction(N)
    Ehat_N = cf['Einf'] + np.exp(-np.outer(lags, H.MU)) @ cf['w']
    Ehat_U = h * ghat(lags / h ** 2)
    rec = dict(N=N)
    for lab, Ehat in [('FV', 0 * E), ('hybrid_fitN', Ehat_N), ('hybrid_universal', Ehat_U)]:
        d = E - Ehat
        y = duhamel(Sh + Ehat, q)
        err = np.abs(y - ref).max()
        G = gain(d)
        rec[lab] = dict(max_err=float(err), gain=float(G), bound=float(G * qmax), ok=bool(err <= G * qmax * (1 + 1e-9)))
    t2.append(rec)
    print(f"N={N:3d}: FV error {rec['FV']['max_err']:.2e} <= gain*|q| {rec['FV']['bound']:.2e} [{rec['FV']['ok']}] | "
          f"hybrid(fit-N) {rec['hybrid_fitN']['max_err']:.2e} <= {rec['hybrid_fitN']['bound']:.2e} [{rec['hybrid_fitN']['ok']}] | "
          f"hybrid(universal, {len(NU)} states) {rec['hybrid_universal']['max_err']:.2e} <= {rec['hybrid_universal']['bound']:.2e} [{rec['hybrid_universal']['ok']}]")
OUT['thm'] = t2
for lab in ['FV', 'hybrid_fitN', 'hybrid_universal']:
    for key in ['max_err', 'gain']:
        v = [r[lab][key] for r in t2]
        p = -np.polyfit(np.log(Ns), np.log(v), 1)[0]; OUT[f'order_{lab}_{key}'] = p
        print(f'order {lab:17s} {key:8s}: {p:.2f}')

# sharpness: the worst-case input for N=10 FV attains the gain exactly
N = 10; Sh = S_fv(N, lags); d = Sx - Sh
k = nst - 1
sgn = np.sign(np.r_[d[0], np.diff(d)])            # q_k * d0 + sum q_j (d_{k-j} - d_{k-j-1})
qw = np.empty(nst); qw[k] = sgn[0]
for j in range(k):
    qw[j] = sgn[k - j]
ew = duhamel(Sh, qw) - duhamel(Sx, qw)
print(f'sharpness (N=10 FV): error with worst-case input = {abs(ew[k]):.6e}, gain = {gain(d):.6e}')
OUT['sharp'] = dict(err=float(abs(ew[k])), gain=float(gain(d)))
json.dump(OUT, open(ROOT + '/ext/out/hybrid_theory.json', 'w'))
print('done')

# ---------- additional: TV(r_N), r_N = E - h g(tau/h^2), and eta_K(sigma_1)
add = []
for N in Ns:
    h = 1 / N
    tau = np.r_[0.0, np.logspace(-9, 1.5, 4000)]
    E = S_exact(tau) - S_fv(N, tau); rN = E - h * g_half(tau / h ** 2); rN[0] = 0.0
    s1 = dtau / h ** 2
    ss = np.r_[0.0, np.logspace(np.log10(s1), 6, 800)]
    dg = np.r_[0.0, g_half(ss[1:][ss[1:] <= 1e5]) - ghat(ss[1:][ss[1:] <= 1e5])]
    eta = abs(dg[1]) + np.abs(np.diff(dg[1:])).sum()
    add.append(dict(N=N, TV_rN=float(np.abs(np.diff(rN)).sum()), TV_rN_over_h2=float(np.abs(np.diff(rN)).sum() * N * N),
                    sigma1=float(s1), eta=float(eta)))
    print(f"N={N:3d}: TV(r_N)/h^2 = {add[-1]['TV_rN_over_h2']:.4f}   sigma_1 = dtau/h^2 = {s1:.3g}   eta_K(sigma_1) = {eta:.2e}")
OUT['thm3'] = add
json.dump(OUT, open(ROOT + '/ext/out/hybrid_theory.json', 'w'))
