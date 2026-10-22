import os; ROOT = os.environ.get('LIION_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
import numpy as np, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#7c5cc4']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 9.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.6,
                     'legend.frameon': False, 'figure.dpi': 150, 'savefig.bbox': 'tight'})
A = json.load(open(ROOT + '/ext/out/compare_tf.json')); B = json.load(open(ROOT + '/ext/out/compare_tf2.json')); G = json.load(open(ROOT + '/ext/out/graded.json')); T3 = json.load(open(ROOT + '/ext/out/compare_tf3.json'))
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.0), gridspec_kw=dict(wspace=0.28))
sty = {'FV (half-cell flux extrapolation)': (C[1], 'o-'), 'FV, two-cell extrapolation (Zeng 2013)': (C[3], 's-'),
       'exact modes + static correction (residue-grouping limit)': (C[4], 'v-'),
       'Pade of exact transfer function (Forman 2011)': (C[5], 'D-'), 'hybrid (this work)': (C[0], '*')}
for k, (col, m) in sty.items():
    pts = sorted([tuple(r[:2]) for r in A[k]])
    ax[0].loglog(*zip(*pts), m, color=col, ms=6 if k != 'hybrid (this work)' else 10, lw=1.2, label=k)
gr = G['graded']['0.00028']
ax[0].loglog([r['N'] for r in gr], [r['gain_graded'] for r in gr], 'P--', color='#3c3b37', ms=6, lw=1.2, label='FV, optimally graded mesh (Zeng-type)')
ax[0].loglog([r['states'] for r in T3['cheb']], [r['gain'] for r in T3['cheb']], 'h-', color='#8c5a3c', ms=5, lw=1.2, label='Chebyshev collocation (odd subspace)')
ax[0].loglog([r['states'] for r in T3['dra']], [r['gain'] for r in T3['dra']], 'x-', color='#139a9a', ms=6, mew=1.5, lw=1.2, label='DRA realized at this $\\Delta\\tau$ (+ integrator)')
for r in A['hybrid (this work)']:
    ax[0].annotate(r[2], (r[0], r[1]), textcoords='offset points', xytext=(5, -3), fontsize=6.5, color=INK2)
ax[0].set(xlabel='number of states', ylabel=r'sharp $\ell_\infty$ error gain (Thm. 1)', title=r'(a) Accuracy per state, $\Delta\tau = 2.8\times10^{-4}$')
ax[0].legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2)

dts = sorted(B, key=float); x = np.array([float(d) for d in dts])
pick = [('Pade m=10 (11 states)', C[5], 'D-'), ('BT-SPA of hybrid r=10 (11 states)', C[0], 'o-'),
        ('hybrid N=10,K=16 (26 states)', C[2], 's-'), ('Zeng N=20 (20 states)', C[3], '^-'), ('FV N=20 (20 states)', C[1], 'v-')]
for k, col, m in pick:
    ax[1].loglog(x, [B[d][k] for d in dts], m, color=col, ms=5, lw=1.2, label=k)
gx = sorted(G['graded'], key=float)
ax[1].loglog([float(d) for d in gx], [[r for r in G['graded'][d] if r['N'] == 20][0]['gain_graded'] for d in gx], 'P--', color='#3c3b37', ms=5, lw=1.2, label='FV N=20, optimally graded (20 states)')
for lab, col, m in [('Chebyshev Nc=20 (9 states)', '#8c5a3c', 'h-'), ('DRA n=10 realized at 2.8e-4, used at this interval', '#139a9a', 'x-'),
                     ('DRA n=10 realized at 2.8e-5 (long horizon)', '#139a9a', 'x--')]:
    ax[1].loglog(x, [T3['vs_dtau'][d][lab] for d in dts], m, color=col, ms=5, mew=1.3, lw=1.2,
                 label=lab.replace('realized at 2.8e-4, used at this interval', 'realized at $\\Delta\\tau$=2.8e-4 (11 states)').replace('realized at 2.8e-5 (long horizon)', 'realized at $\\Delta\\tau$=2.8e-5 (11 states)'))
ax[1].axvspan(1e-5, 1.2e-4, color=GRID, alpha=0.6, lw=0)
ax[1].text(1.2e-5, 3e-9, 'fitted CALCE cell,\n1 s sampling', fontsize=7, color=INK2)
ax[1].set(xlabel=r'sampling interval $\Delta\tau = D\,\Delta t/R^2$', ylabel=r'sharp $\ell_\infty$ error gain', title='(b) Dependence on sampling interval', xlim=(1e-5, 5e-2))
ax[1].legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2)
fig.savefig(ROOT + '/figures/fig21_compare_transfer_function.png'); plt.close(fig)
print('ok')
