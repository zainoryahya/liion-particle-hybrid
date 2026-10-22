"""Samsung 30T model: reuses calce_model (SPM + exact/FV/hybrid particle) with the Samsung OCP and measured temperature."""
import os; ROOT = os.environ.get('LIION_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
import numpy as np, json, sys
sys.path.insert(0, ROOT + '/calce'); sys.path.insert(0, ROOT + '/samsung')
import calce_model as M
import load_ss as L
O = json.load(open(ROOT + '/samsung/hybrid_ocp_ss.json'))
M.SP, M.UPt, M.SN, M.UNt = np.array(O['sto_p']), np.array(O['Up']), np.array(O['sto_n']), np.array(O['Un'])
M.W = O['window']; M.Q20 = O['Qtot_C20']
NAMES, unpack = M.NAMES, M.unpack

def load_test(key, temp='25degC', dt=0.1):
    r = L.load(key, temp)
    tt = r['t'] - r['t'][0]
    tg = np.arange(0, tt[-1] + dt, dt)
    idx = np.clip(np.searchsorted(tt, tg + 1e-9, side='right') - 1, 0, len(tt) - 1)
    I = r['I'][idx].copy()
    gi = np.minimum(np.round(tt / dt).astype(int), len(tg) - 1)
    I[gi] = r['I']
    T = np.interp(tg, tt, r['T'])
    return dict(t=tg, I=I, dt=dt, soc0=1.0, key=key, gi=gi, Vd=r['V'], Id=r['I'], td=tt, T=T, Td=r['T'],
                mask=np.ones(len(tt), bool))

def simulate(x, d, kind='hybrid', N=10, isothermal=False):
    T = 298.15 if isothermal else d['T']
    return M.simulate(x, d, kind, N, T=T)
