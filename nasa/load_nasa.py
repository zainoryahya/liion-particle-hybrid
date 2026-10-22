import os; ROOT = os.environ.get('LIION_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
import numpy as np, scipy.io as sio
def load(path):
    m = sio.loadmat(path, simplify_cells=True)
    key = [k for k in m if not k.startswith('__')][0]
    cyc = m[key]['cycle']
    out = []
    for i, c in enumerate(cyc):
        out.append(dict(idx=i, type=c['type'], T_amb=c['ambient_temperature'], data=c['data']))
    return out
if __name__ == '__main__':
    C = load(ROOT + '/nasa/B0005.mat')
    from collections import Counter
    print(Counter(c['type'] for c in C))
    dis = [c for c in C if c['type']=='discharge']
    imp = [c for c in C if c['type']=='impedance']
    caps = [c['data']['Capacity'] for c in dis]
    print('discharge cycles', len(dis), 'capacity first/last', caps[0], caps[-1])
    d = dis[0]['data']
    for k,v in d.items(): print(k, np.shape(v), np.round(np.asarray(v).ravel()[:6],4))
    t=d['Time']; print('dt stats', np.diff(t)[:10], np.median(np.diff(t)), t[-1])
    I=d['Current_measured']; V=d['Voltage_measured']; T=d['Temperature_measured']
    print('I range', I.min(), I.max(), ' V range', V.min(), V.max(), ' T', T.min(), T.max())
    # where current stops
    on = np.flatnonzero(np.abs(I)>1.0); print('current on idx', on[0], on[-1], 'of', len(I), 't_off', t[on[-1]])
    print('imp first Re/Rct', imp[0]['data']['Re'], imp[0]['data']['Rct'])
    print('imp keys', list(imp[0]['data'].keys()))
    # order of first few cycles
    print([ (c['idx'], c['type']) for c in C[:12]])
