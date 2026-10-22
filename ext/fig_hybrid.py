import os; ROOT = os.environ.get('LIION_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
import numpy as np, json, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, ROOT + '/ext')
import hybrid as H

C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 9.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.6,
                     'legend.frameon': False, 'figure.dpi': 150, 'savefig.bbox': 'tight'})
Hd = json.load(open(ROOT + '/ext/out/hybrid.json')); Vd = json.load(open(ROOT + '/ext/out/hybrid_voltage.json'))
fig, ax = plt.subplots(1, 4, figsize=(17.5, 3.7), gridspec_kw=dict(wspace=0.34))

tau = np.logspace(-6, 0, 300)
ax[0].loglog(tau, -H.S_long(tau), color=C[0], lw=2.6, alpha=0.5, label='long-time (eigen series)')
ts = tau[tau < 2e-2]
ax[0].loglog(ts, -H.S_short(ts), color=C[1], ls='--', lw=1.4, label='short-time (Laplace)')
for N, col in [(10, C[2]), (40, C[3])]:
    ax[0].loglog(tau, -H.S_fv(N, tau), color=col, lw=1.1, label=f'finite volume, N={N}')
ax[0].set(xlabel=r'$\tau$ after a current step', ylabel=r'$-u_s(\tau)$ (unit flux)', title='Step response at the surface', xlim=(1e-6, 1), ylim=(1e-3, 1.0))
ax[0].legend(fontsize=7.5, loc='upper left')

rows = Hd['rows']; N = np.array([r['N'] for r in rows])
ax[1].loglog(N, [r['plain_max'] for r in rows], 'o-', color=C[1], label=f"FV, max (order {Hd['order_plain_max']:.2f})")
ax[1].loglog(N, [r['hyb_max'] for r in rows], 'o-', color=C[0], label=f"hybrid, max (order {Hd['order_hyb_max']:.2f})")
ax[1].loglog(N, [r['plain_rms'] for r in rows], 's:', color=C[1], ms=4, label='FV, RMS')
ax[1].loglog(N, [r['hyb_rms'] for r in rows], 's:', color=C[0], ms=4, label='hybrid, RMS')
ax[1].set(xlabel='finite volumes $N$', ylabel='surface concentration error', title='Random 1-s current profile'); ax[1].legend(fontsize=7.5)
from matplotlib.ticker import NullFormatter, FixedLocator, FixedFormatter
ax[1].xaxis.set_major_locator(FixedLocator([5, 10, 20, 40, 80])); ax[1].xaxis.set_major_formatter(FixedFormatter(['5', '10', '20', '40', '80'])); ax[1].xaxis.set_minor_formatter(NullFormatter())

t = np.arange(len(Vd['I']))
w = (t >= 600) & (t < 900)
ax[2].plot(t[w], np.array(Vd['trace']['plain20'])[w] * 1e3, color=C[1], lw=1.0, label='FV, N = 20 (40 states)')
ax[2].plot(t[w], np.array(Vd['trace']['hyb10'])[w] * 1e3, color=C[0], lw=1.4, label='hybrid, N = 10, K = 16 (52 states)')
ax[2].axhline(0, color=INK2, lw=0.5)
ax[2].set(xlabel='time [s]', ylabel='voltage error [mV]', title='NASA B0005 cell, dynamic current'); ax[2].legend(fontsize=7.5, loc='lower left')

pts = []
for lab, r in Vd['K16'].items():
    pts.append((lab, r['states'], r['max_mV']))
for lab, r in Vd['plain_more'].items():
    pts.append((lab, r['states'], r['max_mV']))
for K in ['K6', 'K8']:
    for lab, r in Vd[K].items():
        pts.append((lab + '_' + K, r['states'], r['max_mV']))
pl = sorted([p for p in pts if p[0].startswith('plain')], key=lambda p: p[1])
hy = [p for p in pts if p[0].startswith('hyb')]
ax[3].loglog([p[1] for p in pl], [p[2] for p in pl], 'o-', color=C[1], label='finite volume')
for (lab, st, e) in hy:
    K = lab.split('_K')[1] if '_K' in lab else '16'
    ax[3].loglog(st, e, 'D', ms=6, color=C[0])
    ax[3].annotate(f"N={lab[3:5]}, K={K}", (st, e), textcoords='offset points', xytext=(5, 3), fontsize=7, color=INK2)
ax[3].plot([], [], 'D', color=C[0], label='hybrid (analytical correction)')
ax[3].set(xlabel='total model states (both electrodes)', ylabel='max voltage error [mV]', title='Accuracy vs model size'); ax[3].legend(fontsize=7.5, loc='upper right')
ax[3].xaxis.set_major_locator(FixedLocator([20, 40, 80, 160, 320])); ax[3].xaxis.set_major_formatter(FixedFormatter(['20', '40', '80', '160', '320'])); ax[3].xaxis.set_minor_formatter(NullFormatter())
fig.savefig(ROOT + '/figures/fig17_hybrid_analytical_numerical.png'); plt.close(fig)
print('ok')
