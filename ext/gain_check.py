"""Check that the sampled horizon of horizon.py reproduces the full-sample gain over tau in [0, T_H],
and report how much the earlier fixed 1800-sample horizon underestimated it."""
import os; ROOT = os.environ.get('LIION_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
import numpy as np, json, sys, io, contextlib
sys.path.insert(0, ROOT + '/ext'); sys.path.insert(0, ROOT + '/calce')
with contextlib.redirect_stdout(io.StringIO()):
    import hybrid as H
import calce_model as CM
import horizon as HZ


def models(lags):
    out = {}
    for N in [10, 40]:
        out[f'FV N={N}'] = CM.HT.S_fv(N, lags)
    out['two-cell N=20'] = CM.S_twocell(20, lags)
    for N in [5, 10, 20]:
        H.MU = np.logspace(-1, 6.5, 16)
        cf = H.fit_correction(N, tau_min=1e-7)
        out[f'hybrid N={N},K=16'] = CM.HT.S_fv(N, lags) + cf['Einf'] + np.exp(-np.outer(lags, H.MU)) @ cf['w']
    return out


OUT = {}
for dtau in [2.8e-4, 2.8e-5]:
    full = np.arange(int(np.ceil(HZ.T_H / dtau)) + 1) * dtau
    sub = HZ.lags_for(dtau)
    old = np.arange(1800) * dtau
    res = {}
    for lags, lab in [(full, 'full'), (sub, 'sub'), (old, 'old1800')]:
        Sx = CM.HT.S_exact(lags); Sx[0] = 0.0
        for k, S in models(lags).items():
            res.setdefault(k, {})[lab] = HZ.gain(Sx - S)
    OUT[str(dtau)] = res
    print(f'dtau = {dtau:.1e}: {len(full)} samples to tau = {HZ.T_H}, {len(sub)} retained')
    for k, v in res.items():
        print(f"   {k:20s} full {v['full']:.4e}  retained {v['sub']:.4e} (rel. diff {abs(v['sub']/v['full']-1):.1e})"
              f"   old 1800-sample {v['old1800']:.4e} (ratio {v['full']/v['old1800']:.2f})", flush=True)
json.dump(OUT, open(ROOT + '/ext/out/gain_check.json', 'w'), indent=1)
