"""SPMe with hybrid particle: voltage error vs SPMe with EXACT particle (same electrolyte), dynamic profiles."""
import os; ROOT = os.environ.get('LIION_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
import numpy as np, json, sys
sys.path.insert(0, ROOT + '/ext')
from spme import SPMe
import rom_hybrid as RH

AREA = 0.137 * 0.207; C1 = 0.680616
OUT = {}
tp = np.arange(0, 2400, 1.0)
prof = {'pulse 3C / -1.5C': (tp, np.where((tp % 300) < 30, 3 * C1, np.where((tp % 300) < 60, -1.5 * C1, 0.0))),
        'synthetic drive cycle': (np.arange(1800.0), RH.profiles()['synthetic drive cycle'] * C1 / 1.8)}
for name, (t, I) in prof.items():
    ref = SPMe(Nr=10, A=AREA).simulate(t, I, particle='exact')
    row = {}
    for N in [10, 20, 40, 80]:
        V = SPMe(Nr=N, A=AREA).simulate(t, I, particle='fv'); e = np.abs(V - ref) * 1e3
        row[f'FV N={N}'] = dict(states=2 * N + 26, max_mV=float(e.max()), rms_mV=float(np.sqrt(np.mean(e ** 2))))
    for N in [10, 20]:
        V = SPMe(Nr=N, A=AREA).simulate(t, I, particle='hybrid'); e = np.abs(V - ref) * 1e3
        row[f'hybrid N={N}'] = dict(states=2 * (N + 16) + 26, max_mV=float(e.max()), rms_mV=float(np.sqrt(np.mean(e ** 2))))
        if N == 10:
            OUT.setdefault('trace', {})[name] = dict(t=t.tolist(), I=I.tolist(), hyb10=(V - ref).tolist())
    OUT.setdefault('trace', {})[name]['fv20'] = (SPMe(Nr=20, A=AREA).simulate(t, I) - ref).tolist()
    OUT[name] = row
    for k, v in row.items():
        print(f"{name:22s} {k:12s} states {v['states']:3d}: max {v['max_mV']:7.3f} mV  RMS {v['rms_mV']:6.3f} mV", flush=True)
json.dump(OUT, open(ROOT + '/ext/out/spme_hybrid.json', 'w'))
print('done')
