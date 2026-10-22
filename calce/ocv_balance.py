"""
CALCE INR18650-20R (NMC/graphite, 2 Ah), 25 C.
(1) Measured OCV: average of C/20 discharge & charge branches (pseudo-OCV) + incremental OCV rest points
(2) Electrode balancing: OCV(SOC) = U_p(th_p) - U_n(th_n),
    th_n = th_n100 - (1-SOC)(th_n100-th_n0),  th_p = th_p100 + (1-SOC)(th_p0-th_p100)
    fit 4 parameters for each literature OCP pair (global DE + LM)
(3) Differential voltage analysis dV/dQ (industrial diagnostic)
"""
import os; ROOT = os.environ.get('LIION_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
import numpy as np, json, sys
from scipy.optimize import differential_evolution, least_squares
from scipy.signal import savgol_filter

R = np.load(ROOT + '/calce/raw.npz')
lc, inc, cap = R['lc'], R['inc'], R['cap']
OUT = {}

# ---------- (1a) C/20: discharge branch (step 3) and charge branch (step 5)
def branch(step):
    s = lc[lc[:, 1] == step]
    t, V, I = s[:, 0], s[:, 2] / 1000, s[:, 3] / 1000
    Q = np.r_[0, np.cumsum(np.abs(I[1:]) * np.diff(t) / 3600)]
    return Q, V
Qd, Vd = branch(3); Qc, Vc = branch(5)
Qtot = Qd[-1]
print(f'C/20 discharge capacity = {Qtot:.4f} Ah ; charge = {Qc[-1]:.4f} Ah ; 0.5C capacity = {cap[:, 6].max():.4f} Ah')
# SOC axis: discharge from 100% (SOC = 1 - Q/Qtot); charge starts from 0% (SOC = Qc/Qtot)
soc_d = 1 - Qd / Qtot; soc_c = Qc / Qtot
grid = np.linspace(0.0, 1.0, 501)
Vd_g = np.interp(grid, soc_d[::-1], Vd[::-1])
ok_c = grid <= soc_c[-1]
Vc_g = np.interp(grid, soc_c, Vc)
edge = ok_c & (grid > soc_c[-1] - 0.04)
ocv = np.where(ok_c, 0.5 * (Vd_g + Vc_g), Vd_g + np.median((Vc_g - Vd_g)[edge]) / 2)   # local offset at the boundary
hyst = (Vc_g - Vd_g)
print(f'charge/discharge half-gap (median, SOC 20-90%) = {np.median(hyst[(grid>0.2)&(grid<0.9)])/2*1e3:.1f} mV ; C/20 charge stops at SOC {soc_c[-1]*100:.1f}%')

# ---------- (1b) incremental OCV: voltage at the end of each 2 h rest (step 6 discharge; step 9? charge)
st = inc[:, 1].astype(int); seg = np.split(np.arange(len(inc)), np.flatnonzero(np.diff(st) != 0) + 1)
pts_d, pts_c = [], []
qnet = np.r_[0, np.cumsum(inc[1:, 3] * np.diff(inc[:, 0]) / 3600)]   # Ah (+ charge)
for s in seg:
    if np.abs(inc[s, 3]).max() < 1e-3 and inc[s[-1], 0] - inc[s[0], 0] > 3000:
        prev = inc[s[0] - 1, 3] if s[0] > 0 else 0
        rec = (qnet[s[-1]], inc[s[-1], 4])
        (pts_d if prev < 0 else pts_c).append(rec)
pts_d, pts_c = np.array(pts_d), np.array(pts_c)
# SOC reference: first rest (after full charge) = 100%
q100 = qnet[seg[3][-1]]
soc_inc_d = 1 + (pts_d[:, 0] - q100) / Qtot
soc_inc_c = 1 + (pts_c[:, 0] - q100) / Qtot if len(pts_c) else np.array([])
print(f'incremental OCV: {len(pts_d)} discharge points, {len(pts_c)} charge points')
dev = pts_d[:, 1] - np.interp(soc_inc_d, grid, ocv)
print('  diff (incremental discharge OCV - C/20 pseudo-OCV) [mV]:', np.round(dev * 1e3, 1))
# (1c) correction: 2 h rest points (both directions) = equilibrium OCV; smooth offset on the pseudo-OCV
sr = np.r_[soc_inc_d, soc_inc_c[1:]]; vr = np.r_[pts_d[:, 1], pts_c[1:, 1]]
keep = (sr > 0.12) & (sr < 0.95)
dv = vr[keep] - np.interp(sr[keep], grid, ocv)
cf = np.polyfit(sr[keep], dv, 2)
ocv_raw = ocv.copy()
ocv = ocv + np.polyval(cf, grid)
resid_rest = vr[keep] - np.interp(sr[keep], grid, ocv)
print(f'offset correction (quadratic): range {np.polyval(cf, grid).min()*1e3:.1f} .. {np.polyval(cf, grid).max()*1e3:.1f} mV ; residual at rest points RMS {np.sqrt(np.mean(resid_rest**2))*1e3:.1f} mV')
OUT['ocv_raw'] = ocv_raw.tolist(); OUT['offset_coef'] = cf.tolist(); OUT['rest_resid_mV'] = float(np.sqrt(np.mean(resid_rest ** 2)) * 1e3)
OUT.update(grid=grid.tolist(), ocv=ocv.tolist(), Vd=Vd_g.tolist(), Vc=Vc_g.tolist(), soc_c_max=float(soc_c[-1]),
           inc_d=dict(soc=soc_inc_d.tolist(), V=pts_d[:, 1].tolist()),
           inc_c=dict(soc=soc_inc_c.tolist(), V=(pts_c[:, 1].tolist() if len(pts_c) else [])), Qtot=Qtot)

# ---------- (2) electrode balancing
TAB = np.load(ROOT + '/nasa/ocp_tables.npy', allow_pickle=True).item(); STO = TAB['sto']
mask = (grid > 0.02) & (grid < 0.98)          # avoid the polarisation-dominated ends
y = ocv[mask]; s_ = grid[mask]
pairs = ['Mohtat2020', 'Chen2020', 'Ecker2015', 'NCA_Kim2011', 'OKane2022', 'Ai2020', 'Marquis2019']
bal = {}
for nm in pairs:
    tb = TAB['tables'][nm]
    Un = lambda x: np.interp(x, STO, tb['n']); Up = lambda x: np.interp(x, STO, tb['p'])
    def model(p, soc):
        n100, n0, p100, p0 = p
        return Up(p100 + (1 - soc) * (p0 - p100)) - Un(n100 - (1 - soc) * (n100 - n0))
    res = lambda p: model(p, s_) - y
    bnds = [(0.4, 1.0), (0.0, 0.4), (0.0, 0.6), (0.6, 1.0)]
    de = differential_evolution(lambda p: np.sum(res(p) ** 2), bnds, seed=0, tol=1e-10, maxiter=400, polish=False)
    ls = least_squares(res, de.x, bounds=([b[0] for b in bnds], [b[1] for b in bnds]))
    r = res(ls.x); rm = np.sqrt(np.mean(r ** 2))
    bal[nm] = dict(p=ls.x.tolist(), rmse_mV=float(rm * 1e3), max_mV=float(np.abs(r).max() * 1e3),
                   fit=model(ls.x, grid).tolist())
    print(f'{nm:12s} RMSE {rm*1e3:6.2f} mV  max {np.abs(r).max()*1e3:6.1f} mV   th_n: {ls.x[0]:.3f}->{ls.x[1]:.3f}  th_p: {ls.x[2]:.3f}->{ls.x[3]:.3f}')
OUT['balance'] = bal
best = min(bal, key=lambda k: bal[k]['rmse_mV']); OUT['best'] = best
print('best:', best)

# ---------- (3) dV/dQ
Q = (1 - grid) * Qtot
dvdq = -np.gradient(savgol_filter(ocv, 31, 3), Q)
dvdq_fit = -np.gradient(np.array(bal[best]['fit']), Q)
OUT.update(Q=Q.tolist(), dvdq=dvdq.tolist(), dvdq_fit=dvdq_fit.tolist())
json.dump(OUT, open(ROOT + '/calce/ocv_balance.json', 'w'))
print('done')
