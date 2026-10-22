import os; ROOT = os.environ.get('LIION_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
import numpy as np, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter

C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 9.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.6,
                     'legend.frameon': False, 'figure.dpi': 150, 'savefig.bbox': 'tight'})
R = json.load(open(ROOT + '/ext/out/rom_hybrid.json')); K = json.load(open(ROOT + '/ext/out/ekf_hybrid.json'))
S = json.load(open(ROOT + '/ext/out/spme_hybrid.json'))
fig, ax = plt.subplots(1, 4, figsize=(18, 3.8), gridspec_kw=dict(wspace=0.32))

# (a) ROM order
rs = [2, 3, 4, 6]; xs = rs + [9]
mk = {'random BMS (-1C..2C)': 'o', 'HPPC-type pulses': 's', 'synthetic drive cycle': '^'}
for name, m in mk.items():
    row = R['rows'][name]
    ax[0].semilogy(xs, [row[f'FV_r{r}']['max_mV'] for r in rs] + [row['FV_rfull']['max_mV']], m + '-', color=C[1], ms=5, lw=1.2)
    ax[0].semilogy(xs, [row[f'HYB_r{r}']['max_mV'] for r in rs] + [row['HYB_rfull']['max_mV']], m + '-', color=C[0], ms=5, lw=1.2)
ax[0].plot([], [], '-', color=C[1], label='ROM of finite volume (N = 20)')
ax[0].plot([], [], '-', color=C[0], label='ROM of hybrid (N = 20, K = 16)')
for name, m in mk.items():
    ax[0].plot([], [], m, color=INK2, ms=5, label=name)
ax[0].xaxis.set_major_locator(FixedLocator(xs)); ax[0].xaxis.set_major_formatter(FixedFormatter(['2', '3', '4', '6', 'full']))
ax[0].set(xlabel='reduced order $r$ per electrode', ylabel='max voltage error vs exact [mV]', title='(a) Balanced-truncation ROM accuracy', ylim=(0.02, 150))
ax[0].legend(fontsize=7, loc='lower left')

# (b) trace
name = 'synthetic drive cycle'; t = np.arange(len(R['trace'][name]['FV']))
w = (t >= 300) & (t < 700)
ax[1].plot(t[w], np.array(R['trace'][name]['FV'])[w] * 1e3, color=C[1], lw=1.0, label='ROM of finite volume, r = 3')
ax[1].plot(t[w], np.array(R['trace'][name]['HYB'])[w] * 1e3, color=C[0], lw=1.2, label='ROM of hybrid, r = 3')
ax[1].axhline(0, color=INK2, lw=0.5)
ax[1].set(xlabel='time [s]', ylabel='voltage error [mV]', title='(b) Drive cycle, 7-state ROMs'); ax[1].legend(fontsize=7.5, loc='lower left')

# (c) EKF
t = np.array(K['t']); st = np.array(K['soc_true'])
cols = [C[1], C[3], C[0], C[2]]
for (lab, run), col in zip(K['runs'].items(), cols):
    e = (np.array(run['soc']) - st) * 100
    ax[2].plot(t, e, color=col, lw=1.0, label=f"{lab}: innov. {run['innov_rms_mV']:.1f} mV, NEES {run['nees']:.1f}")
ax[2].set(xlabel='time [s]', ylabel='SOC error [%]', title='(c) EKF, exact-model truth, 2 mV noise', ylim=(-1.0, 1.0), xlim=(0, 5400))
ax[2].legend(fontsize=6.6, loc='upper center', bbox_to_anchor=(0.5, -0.2))

# (d) SPMe
for name, m in [('pulse 3C / -1.5C', 'o'), ('synthetic drive cycle', '^')]:
    row = S[name]
    fv = [(v['states'], v['max_mV']) for k, v in row.items() if k.startswith('FV')]
    hy = [(v['states'], v['max_mV']) for k, v in row.items() if k.startswith('hyb')]
    ax[3].loglog(*zip(*fv), m + '-', color=C[1], ms=5, lw=1.2)
    ax[3].loglog(*zip(*hy), m, color=C[0], ms=6)
ax[3].plot([], [], '-', color=C[1], label='SPMe, finite-volume particles (N = 10-80)')
ax[3].plot([], [], '-', color=C[0], label='SPMe, hybrid particles (N = 10, 20)')
ax[3].plot([], [], 'o', color=INK2, label='pulse 3C / -1.5C'); ax[3].plot([], [], '^', color=INK2, label='synthetic drive cycle')
ax[3].set(xlabel='total SPMe states', ylabel='max voltage error [mV]', title='(d) SPMe with hybrid particles', ylim=(1e-3, 30))
ax[3].xaxis.set_major_locator(FixedLocator([50, 100, 200])); ax[3].xaxis.set_major_formatter(FixedFormatter(['50', '100', '200'])); ax[3].xaxis.set_minor_formatter(NullFormatter())
ax[3].legend(fontsize=7, loc='center right')
ax[3].annotate('hybrid: error at the level of\nthe exact-reference accuracy', (95, 4e-3), xytext=(52, 2.2e-2), fontsize=7, color=INK2, arrowprops=dict(arrowstyle='->', color=INK2, lw=0.6))
fig.savefig(ROOT + '/figures/fig19_hybrid_rom_ekf_spme.png'); plt.close(fig)
print('ok')
