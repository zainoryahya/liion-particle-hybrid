"""
SOC EKF on real CALCE data (25 C), two complete workflows (identification -> ROM -> EKF):
  A 'standard': parameters fitted with FV N=10, ROM = BT-SPA r of FV N=10
  B 'hybrid'  : parameters fitted with hybrid N=10, ROM = BT-SPA r of hybrid N=10
Reference SOC = laboratory Coulomb counting (exact current sensor) with the model capacity. Initial EKF SOC is wrong (50%).
"""
import os; ROOT = os.environ.get('LIION_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
import numpy as np, json, sys, io, contextlib
from scipy.linalg import expm
sys.path.insert(0, ROOT + '/calce'); sys.path.insert(0, ROOT + '/ext')
import calce_model as M
with contextlib.redirect_stdout(io.StringIO()):
    import rom_hybrid as RH
F, Rg = M.F, M.Rg
R_ORD = int(sys.argv[1]) if len(sys.argv) > 1 else 6
Dz = np.load(ROOT + '/calce/dyn/dyn.npz')


def build(kind, fitfile, N=10):
    Fp = json.load(open(fitfile)); p = M.unpack(np.array(Fp['x']))
    sysd, _ = RH.deviation_systems(N=N, dtau_min=1e-5)
    A, B, C, D = sysd['HYB' if kind == 'hybrid' else 'FV']
    Ar, Br, Cr, Dr, _ = RH.bt_spa(A, B, C, D, R_ORD)
    return p, (Ar, Br, Cr, Dr)


def run(p, rom, d, soc_init, Rv=(10e-3) ** 2, T=298.15):
    Ar, Br, Cr, Dr = rom; r = len(Br)
    Qc = p['c'] * M.Q20; W = M.W
    Qe = {'n': Qc / (W['n100'] - W['n0']), 'p': Qc / (W['p0'] - W['p100'])}
    g = {'n': p['tau_n'] / (3 * 3600 * Qe['n']), 'p': -p['tau_p'] / (3 * 3600 * Qe['p'])}
    t, I, V = d['t'], d['I'], d['V']; n = len(t)
    nx = 1 + 2 * r
    x = np.r_[soc_init, np.zeros(2 * r)]; P = np.diag(np.r_[0.3 ** 2, np.full(2 * r, 1e-4)])
    cache = {}
    soc = np.empty(n); inn = np.empty(n); sd = np.empty(n)
    RT = Rg * T

    def volt(socv, thn_s, thp_s, Ik):
        sq = lambda th: np.sqrt(max(th * (1 - th), 1e-6))
        eta = 2 * RT / F * (np.arcsinh(Ik / (2 * p['I0_n'] * sq(thn_s))) + np.arcsinh(Ik / (2 * p['I0_p'] * sq(thp_s))))
        return M.Up(np.array([thp_s]))[0] - M.Un(np.array([thn_s]))[0] - eta - p['R0'] * Ik

    for k in range(n):
        Ik = I[k]
        thn_b = W['n0'] + x[0] * (W['n100'] - W['n0']); thp_b = W['p0'] - x[0] * (W['p0'] - W['p100'])
        thn = thn_b + g['n'] * (Cr @ x[1:1 + r] + Dr * Ik); thp = thp_b + g['p'] * (Cr @ x[1 + r:] + Dr * Ik)
        h0 = volt(x[0], thn, thp, Ik); e = 1e-6
        dn = (volt(x[0], thn + e, thp, Ik) - h0) / e; dp = (volt(x[0], thn, thp + e, Ik) - h0) / e
        Hj = np.r_[dn * (W['n100'] - W['n0']) - dp * (W['p0'] - W['p100']), dn * g['n'] * Cr, dp * g['p'] * Cr]
        S = Hj @ P @ Hj + Rv; K = P @ Hj / S
        inn[k] = V[k] - h0; x = x + K * inn[k]
        IKH = np.eye(nx) - np.outer(K, Hj); P = IKH @ P @ IKH.T + Rv * np.outer(K, K)
        x[0] = np.clip(x[0], -0.05, 1.1)
        soc[k] = x[0]; sd[k] = np.sqrt(P[0, 0])
        if k < n - 1:
            dt = round(t[k + 1] - t[k], 3)
            if dt not in cache:
                blk = {}
                for el in 'np':
                    Aug = np.zeros((r + 1, r + 1)); Aug[:r, :r] = Ar; Aug[:r, r] = Br
                    E = expm(Aug * dt / p['tau_' + el]); blk[el] = (E[:r, :r], E[:r, r])
                Fm = np.eye(nx); Fm[1:1 + r, 1:1 + r] = blk['n'][0]; Fm[1 + r:, 1 + r:] = blk['p'][0]
                cache[dt] = (Fm, blk['n'][1], blk['p'][1])
            Fm, bn, bp = cache[dt]
            G = np.r_[-Ik * dt / (3600 * Qc), bn * Ik, bp * Ik]
            x = Fm @ x + G
            P = Fm @ P @ Fm.T + np.diag(np.r_[(1e-5) ** 2 * dt, np.full(2 * r, 1e-10 * dt)])
    return soc, sd, inn


OUT = {}
PIPE = {'A: FV N=10 (fit + ROM)': ('fv', 'fit_25C_fv10.json'), 'B: hybrid N=10 (fit + ROM)': ('hybrid', 'fit_25C_hybrid10.json')}
built = {k: build(kind, f) for k, (kind, f) in PIPE.items()}
for key in ['25C_FUDS_80', '25C_US06_80', '25C_DST_50', '25C_BJDST_80', '25C_FUDS_50', '25C_US06_50']:
    A_ = Dz[key]; A_ = A_[np.argsort(A_[:, 0], kind='stable')]
    s7 = A_[A_[:, 1] == 7]; before = A_[(A_[:, 1] >= 4) & (A_[:, 1] < 7)]
    t = s7[:, 0] - s7[0, 0]; I = -s7[:, 2]; V = s7[:, 3]
    row = {}
    for lab, (p, rom) in built.items():
        Qc = p['c'] * M.Q20
        q_before = np.sum(-before[:-1, 2] * np.diff(before[:, 0])) / 3600
        soc_true0 = p['soc0'] - q_before / Qc
        soc_true = soc_true0 - np.r_[0, np.cumsum(I[:-1] * np.diff(t))] / 3600 / Qc
        soc, sd, inn = run(p, rom, dict(t=t, I=I, V=V), soc_init=0.5)
        # model-independent error: Ah discharged since full (estimate vs laboratory Coulomb counting), % of 2.0 Ah nominal
        ah_true = q_before + np.r_[0, np.cumsum(I[:-1] * np.diff(t))] / 3600
        ah_est = (p['soc0'] - soc) * Qc
        err = (ah_est - ah_true) / 2.0; m = t > 600
        row[lab] = dict(rmse=float(np.sqrt(np.mean(err[m] ** 2)) * 100), maxerr=float(np.abs(err[m]).max() * 100),
                        innov_rms_mV=float(np.sqrt(np.mean(inn[m] ** 2)) * 1e3), soc0_true=float(soc_true0),
                        err=err[::10].tolist(), t=t[::10].tolist())
        print(f"{key:13s} {lab:28s}: Ah before profile {q_before:.3f}  SOC RMSE (t>600 s) {row[lab]['rmse']:5.2f}%  max {row[lab]['maxerr']:5.2f}%  innovation RMS {row[lab]['innov_rms_mV']:5.1f} mV", flush=True)
    OUT[key] = row
json.dump(OUT, open(f'{ROOT}/calce/ekf_calce_r{R_ORD}.json', 'w'))
print('done')
