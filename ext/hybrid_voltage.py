"""Effect of the hybrid scheme on VOLTAGE (fitted NASA B0005 parameters, Mohtat2020 OCP, 25 C) + effect of the number of correction states K."""
import os; ROOT = os.environ.get('LIION_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
import numpy as np, json, sys
sys.path.insert(0, ROOT + '/ext')
import hybrid as H
from common import params, voltage

OUT = {}
p = params()
rng = np.random.default_rng(7)
dt = 1.0; nst = 1800
I = np.zeros(nst); k = 0
while k < nst:
    L = rng.integers(1, 30); I[k:k + L] = rng.uniform(-2.0, 4.0); k += L     # charge 1C .. discharge 2C
# keep SOC within range: start at 70% SOC (initial stoichiometry shift)
soc0 = 0.7
th0 = {'n': p['th_n0'] - (1 - soc0) * 1.8 / p['Qn'], 'p': p['th_p0'] + (1 - soc0) * 1.8 / p['Qp']}
ths = {}
for MUK in [6, 8, 16]:
    H.MU = np.logspace(-1, 6.5, MUK) if MUK == 16 else np.logspace(0, 6, MUK)
    res = {}
    for el in ['n', 'p']:
        R = p['R_' + el]; D = p['D_' + el]; g = p['g' + el]
        q = g * I * R ** 2 / D                         # dimensionless flux (outward)
        dtau = D * dt / R ** 2
        ref = H.exact_duhamel(q, dtau)
        res[el] = dict(ref=th0[el] + ref)
        for N in [10, 20]:
            res[el][f'plain{N}'] = th0[el] + H.simulate(N, q, dtau)
            cf = H.fit_correction(N, tau_min=dtau / 20)
            res[el][f'hyb{N}'] = th0[el] + H.simulate(N, q, dtau, corr=cf)
    Vref = voltage(p, res['n']['ref'], res['p']['ref'], I, np.full(nst, 298.15))
    row = {}
    for lab in ['plain10', 'plain20', 'hyb10', 'hyb20']:
        if MUK != 16 and lab.startswith('plain'):
            continue
        Vm = voltage(p, res['n'][lab], res['p'][lab], I, np.full(nst, 298.15))
        e = np.abs(Vm - Vref) * 1e3
        N = int(lab[-2:]); st = 2 * N + (2 * MUK if lab.startswith('hyb') else 0)
        row[lab] = dict(max_mV=float(e.max()), rms_mV=float(np.sqrt(np.mean(e ** 2))), states=st)
        print(f'K={MUK:2d}  {lab:8s} total states {st:3d}: voltage error max {e.max():7.3f} mV  RMS {np.sqrt(np.mean(e**2)):7.4f} mV', flush=True)
        if MUK == 16:
            OUT.setdefault('trace', {})[lab] = (Vm - Vref).tolist()
    OUT[f'K{MUK}'] = row
OUT['I'] = I.tolist(); OUT['Vref'] = Vref.tolist()
json.dump(OUT, open(ROOT + '/ext/out/hybrid_voltage.json', 'w'))
print('done')
