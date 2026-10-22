import os; ROOT = os.environ.get('LIION_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
import numpy as np, json, sys
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, ROOT + '/ext'); import hybrid as H
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']; INK2, GRID = '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True,
                     'grid.color': GRID, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                     'pdf.fonttype': 42})
os.makedirs(ROOT + '/figures/jes', exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.2), gridspec_kw=dict(wspace=0.38))
N = 10; tau = np.logspace(-7, 0, 300)
cf = H.fit_correction(N); E = H.S_exact(tau) - H.S_fv(N, tau)
Eh = cf['Einf'] + np.exp(-np.outer(tau, H.MU)) @ cf['w']
ax[0].loglog(tau, np.abs(E), color=C[1], lw=2, label='finite volume (10 cells)')
ax[0].loglog(tau, np.maximum(np.abs(E - Eh), 1e-7), color=C[0], lw=2, label='hybrid: + analytical correction')
ax[0].annotate('jump h/2 at every\ncurrent step', (1e-6, 0.05), xytext=(2e-7, 2e-3), fontsize=8, color=INK2,
               arrowprops=dict(arrowstyle='->', color=INK2, lw=0.7))
ax[0].set(xlabel=r'dimensionless time after a current step', ylabel='surface-concentration error', ylim=(1e-7, 0.2),
          title='Hidden first-order error and its removal')
ax[0].legend(fontsize=7.5, loc='center right', bbox_to_anchor=(1.0, 0.5))
SW = json.load(open(ROOT + '/samsung/sweep_ss.json')); Pi = np.array([r['Pi'] for r in SW])
for lab, col, m, name in [('fv10', C[1], 'o', 'finite volume, 10 cells'), ('fv20', C[3], 's', 'finite volume, 20 cells'), ('hybrid10', C[0], 'D', 'hybrid, 10 cells')]:
    ok = np.array([r[lab]['rmse'] < 50 for r in SW]); y = np.array([r[lab]['tau_n_ratio'] for r in SW])
    ax[1].semilogx(Pi[ok], y[ok], m + '-', color=col, ms=5, lw=1.4, label=name)
    if (~ok).any(): ax[1].semilogx(Pi[~ok], y[~ok], 'x', color=col, ms=7, mew=1.5)
ax[1].plot([2.11], [0.28], 'o', color=C[1], ms=8, mfc='white', mew=1.5)
ax[1].text(1.75, 0.28, 'high-energy 18650\ncell (measured\ncurrent profile)', fontsize=7, color=INK2, ha='right', va='center')
ax[1].axhline(1, color=INK2, lw=0.5)
ax[1].set(xlabel=r'dimensionless peak surface flux $\Pi$', ylabel=r'identified / true diffusion time', ylim=(0.2, 1.15),
          title='Bias in identified parameters')
ax[1].plot([], [], 'x', color=INK2, label='fit failed')
ax[1].legend(fontsize=7.5, loc='center left', bbox_to_anchor=(0.0, 0.40))
for ext in ('png', 'jpg'):
    fig.savefig(f'{ROOT}/figures/jes/GA_Yahya.{ext}', dpi=300, bbox_inches='tight', facecolor='white')
print('ok')
