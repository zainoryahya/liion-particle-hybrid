import os; ROOT = os.environ.get('LIION_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
import scipy.io as sio, numpy as np, glob, os
DATA = os.path.join(ROOT, 'data', 'Samsung INR21700 30T 3Ah Li-ion Battery Data')
def find(key, temp='25degC'):
    f = [g for g in glob.glob(f'{DATA}/{temp}/*.mat') if f'_{key}_' in os.path.basename(g)]
    assert len(f) == 1, (key, f)
    return f[0]
def load(key, temp='25degC'):
    m = sio.loadmat(find(key, temp), squeeze_me=True, struct_as_record=False)
    d = m[[k for k in m if not k.startswith('__')][0]]
    t = np.asarray(d.Time, float); o = np.argsort(t, kind='stable')
    return dict(t=t[o], I=-np.asarray(d.Current, float)[o], V=np.asarray(d.Voltage, float)[o],
                Ah=np.asarray(d.Ah, float)[o], T=np.asarray(d.Battery_Temp_degC, float)[o] + 273.15)
