import numpy as np, json, sys
import ss_model as S
fits = {'hybrid N=10': ('hybrid', 10, 'fit_UDDS+US06_hybrid10.json'), 'FV N=20': ('fv', 20, 'fit_UDDS+US06_fv20.json'), 'FV N=10': ('fv', 10, 'fit_UDDS+US06_fv10.json')}
keys = ['UDDS', 'HWFET', 'LA92', 'US06'] + [f'Mixed{i}' for i in range(1, 9)]
OUT = {}
which = sys.argv[1:] or list(fits)
for lab in which:
    kind, N, f = fits[lab]
    F = json.load(open(f)); x = np.array(F['x']); row = {'p': F['p']}
    for k in keys:
        d = S.load_test(k)
        V = S.simulate(x, d, kind, N)
        row[k] = float(np.sqrt(np.mean((V[d['gi']] - d['Vd']) ** 2)) * 1e3)
        if lab == 'hybrid N=10':
            Vex = S.simulate(x, d, 'exact')
            row[k + '_hyb_vs_exact_max'] = float(np.abs(V - Vex).max() * 1e3)
            row[k + '_fv10_vs_exact_max'] = float(np.abs(S.simulate(x, d, 'fv', 10) - Vex).max() * 1e3)
            row[k + '_fv20_vs_exact_max'] = float(np.abs(S.simulate(x, d, 'fv', 20) - Vex).max() * 1e3)
    OUT[lab] = row
    print(lab, ' '.join(f"{k} {row[k]:.1f}" for k in keys), flush=True)
    if lab == 'hybrid N=10':
        print('   max diff vs exact (mV): hybrid', [round(row[k + '_hyb_vs_exact_max'], 2) for k in keys[:4]],
              ' FV10', [round(row[k + '_fv10_vs_exact_max']) for k in keys[:4]], ' FV20', [round(row[k + '_fv20_vs_exact_max']) for k in keys[:4]], flush=True)
try:
    old = json.load(open('validate_ss.json')); old.update(OUT); OUT = old
except Exception:
    pass
json.dump(OUT, open('validate_ss.json', 'w'))
