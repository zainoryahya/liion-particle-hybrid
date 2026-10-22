"""Samsung 30T OCV from C/20 (average of discharge & charge branches) + hybrid OCP (Mohtat2020 graphite + derived positive)."""
import os; ROOT = os.environ.get('LIION_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
import numpy as np, json, load_ss as L
c = L.load('C20DisCh'); I, t, V = c['I'], c['t'], c['V']
dis = (I > 0.1); ch = (I < -0.1)
def branch(m):
    tt, vv, ii = t[m], V[m], np.abs(I[m])
    Q = np.r_[0, np.cumsum(ii[1:] * np.diff(tt) / 3600)]
    return Q, vv
Qd, Vd = branch(dis); Qc, Vc = branch(ch)
Qtot = Qd[-1]
g = np.linspace(0, 1, 501)
soc_d = 1 - Qd / Qtot; soc_c = Qc / Qtot
Vd_g = np.interp(g, soc_d[::-1], Vd[::-1]); Vc_g = np.interp(g, soc_c, Vc)
ok = g <= soc_c[-1]
edge = ok & (g > soc_c[-1] - 0.04)
ocv = np.where(ok, 0.5 * (Vd_g + Vc_g), Vd_g + np.median((Vc_g - Vd_g)[edge]) / 2)
print(f'C/20 capacity: discharge {Qtot:.4f} Ah, charge {Qc[-1]:.4f} Ah; median half-hysteresis (20-90%) {np.median((Vc_g-Vd_g)[(g>.2)&(g<.9)])/2*1e3:.1f} mV')
TAB = np.load(ROOT + '/nasa/ocp_tables.npy', allow_pickle=True).item(); STO = TAB['sto']; tb = TAB['tables']['Mohtat2020']
n100, n0, p100, p0 = 0.780, 0.015, 0.016, 0.992
th_n = n100 - (1 - g) * (n100 - n0); th_p = p100 + (1 - g) * (p0 - p100)
Up_h = ocv + np.interp(th_n, STO, tb['n'])
o = np.argsort(th_p); Up_s = np.minimum.accumulate(Up_h[o])
print('max isotonic correction', round(float(np.max(Up_h[o] - Up_s)) * 1e3, 2), 'mV')
json.dump(dict(sto_p=th_p[o].tolist(), Up=Up_s.tolist(), sto_n=STO.tolist(), Un=tb['n'].tolist(),
               window=dict(n100=n100, n0=n0, p100=p100, p0=p0), Qtot_C20=float(Qtot), grid=g.tolist(), ocv=ocv.tolist(),
               note='Samsung INR21700-30T; graphite Mohtat2020; positive from C/20 average OCV'), open('hybrid_ocp_ss.json', 'w'))
