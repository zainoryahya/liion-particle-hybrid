import os; ROOT = os.environ.get('LIION_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
import numpy as np, json, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, ROOT + '/calce')

C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 9.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.6,
                     'legend.frameon': False, 'figure.dpi': 150, 'savefig.bbox': 'tight'})
Vd = json.load(open(ROOT + '/calce/validate_25C.json')); B = json.load(open(ROOT + '/calce/bias_25C.json'))
E = json.load(open(ROOT + '/calce/ekf_calce_r6.json'))
fig, ax = plt.subplots(1, 4, figsize=(18.5, 3.9), gridspec_kw=dict(wspace=0.33, width_ratios=[1.25, 1.15, 0.8, 1.1]))

# (a) US06 validation trace (not used in fitting)
T = Vd['tests']['25C_US06_80']; t = np.array(T['t']); t = t - t[0]
w = (t > 3000) & (t < 3600)
ax[0].plot(t[w], np.array(T['V'])[w], color=INK, lw=1.0, label='measured')
ax[0].plot(t[w], np.array(T['hybrid10']['trace'])[w], color=C[0], lw=1.1, label=f"hybrid model (RMSE {T['hybrid10']['rmse_mV']:.0f} mV, whole test)")
ax[0].plot(t[w], np.array(T['fv10']['trace'])[w], color=C[1], lw=0.9, ls='--', label='same parameters, FV N = 10')
ax[0].set(xlabel='time in US06 test [s]', ylabel='voltage [V]', title='(a) CALCE INR18650-20R, US06, 25 °C (validation)')
ax[0].legend(fontsize=7.2, loc='lower left')

# (b) validation RMSE per test, three identification pipelines
tests = ['DST_80', 'DST_50', 'FUDS_80', 'FUDS_50', 'US06_80', 'US06_50', 'BJDST_80', 'BJDST_50']
xx = np.arange(len(tests)); wd = 0.27
for i, (lab, col) in enumerate([('hybrid N=10', C[0]), ('FV N=20', C[3]), ('FV N=10', C[1])]):
    ax[1].bar(xx + (i - 1) * wd, [B[lab]['25C_' + k] for k in tests], wd, color=col, label=f'identified with {lab}')
ax[1].set_xticks(xx, [k.replace('_', '\n') + '%' for k in tests], fontsize=7.5); ax[1].grid(axis='x', visible=False)
ax[1].set(ylabel='voltage RMSE [mV]', title='(b) Fit on DST 80%, validate on all'); ax[1].legend(fontsize=7.2, loc='upper left')

# (c) parameter bias
labs = ['hybrid N=10', 'FV N=20', 'FV N=10']; cols = [C[0], C[3], C[1]]
for j, par in enumerate(['tau_n', 'tau_p']):
    for i, (lab, col) in enumerate(zip(labs, cols)):
        ax[2].bar(j + (i - 1) * 0.27, B[lab]['p'][par] / 3600, 0.27, color=col)
ax[2].set_xticks([0, 1], [r'$\tau_n=R_n^2/D_n$', r'$\tau_p=R_p^2/D_p$']); ax[2].grid(axis='x', visible=False)
ax[2].set(ylabel='identified diffusion time [h]', title='(c) Identified parameters')
for i, lab in enumerate(labs):
    r = B[lab]['p']['tau_n'] / B['hybrid N=10']['p']['tau_n']
    ax[2].text(0 + (i - 1) * 0.27, B[lab]['p']['tau_n'] / 3600 + 0.08, f'{r:.2f}×', ha='center', fontsize=7, color=INK2)

# (d) EKF on real data
for key, ls in [('25C_US06_80', '-'), ('25C_FUDS_50', ':')]:
    for lab, col in [('A: FV N=10 (fit + ROM)', C[1]), ('B: hybrid N=10 (fit + ROM)', C[0])]:
        r = E[key][lab]
        ax[3].plot(np.array(r['t']) / 60, np.array(r['err']) * 100, ls, color=col, lw=1.0,
                   label=f"{key[4:].replace('_', ' ')}%, {lab.split(' (')[0]}: RMSE {r['rmse']:.2f}%, innov. {r['innov_rms_mV']:.0f} mV")
ax[3].axhline(0, color=INK2, lw=0.5)
ax[3].set(xlabel='time [min]', ylabel='SOC error [% of 2 Ah]', title='(d) EKF on measured data (initial SOC 50%)', ylim=(-4, 4))
ax[3].legend(fontsize=6.6, loc='upper center', bbox_to_anchor=(0.5, -0.2))
fig.savefig(ROOT + '/figures/fig20_calce_dynamic.png'); plt.close(fig)
print('ok')
