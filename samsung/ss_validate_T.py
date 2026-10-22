"""Validation of the temperature-dependent model (fitted on UDDS+US06) on the other cycles at the same temperature. argv: TEMPF"""
import numpy as np, json, sys
import ss_model as S, ss_thermal as T
TEMPF = sys.argv[1]
keys = ['UDDS', 'US06', 'HWFET', 'LA92'] + [f'Mixed{i}' for i in range(1, 9)]
OUT = {}
for lab, kind, N in [('hybrid N=10', 'hybrid', 10), ('FV N=20', 'fv', 20), ('FV N=10', 'fv', 10)]:
    F = json.load(open(f'fitT_UDDS+US06_{kind}{N}_{TEMPF}.json')); x = np.array(F['x']); row = {'p': F['p']}
    for k in keys:
        try:
            d = S.load_test(k, TEMPF)
        except AssertionError:
            continue
        V = T.simulate(x, d, kind, N, Tref=F['Tref'])
        row[k] = float(np.sqrt(np.mean((V[d['gi']] - d['Vd']) ** 2)) * 1e3)
        if lab == 'hybrid N=10' and k in ('UDDS', 'US06'):
            Vex = T.simulate(x, d, 'exact', Tref=F['Tref'])
            row[k + '_maxdiff_vs_exact_mV'] = {kn: float(np.abs(T.simulate(x, d, kd, n_, Tref=F['Tref']) - Vex).max() * 1e3)
                                              for kn, kd, n_ in [('hybrid10', 'hybrid', 10), ('fv10', 'fv', 10), ('fv20', 'fv', 20)]}
    OUT[lab] = row
    print(TEMPF, lab, ' '.join(f"{k} {row[k]:.1f}" for k in keys if k in row), flush=True)
    for k in ('UDDS', 'US06'):
        if k + '_maxdiff_vs_exact_mV' in row:
            print('   max diff vs exact', k, {a: round(b, 2) for a, b in row[k + '_maxdiff_vs_exact_mV'].items()})
json.dump(OUT, open(f'validateT_{TEMPF}.json', 'w'))
