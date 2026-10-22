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
D = json.load(open(ROOT + '/ext/out/hybrid_theory.json'))
fig, ax = plt.subplots(1, 3, figsize=(14.5, 3.9), gridspec_kw=dict(wspace=0.3))

# (a) scaling collapse
for (N, col) in zip(['5', '10', '20', '40', '80'], ['#bcd7f5', '#8fbcef', '#5e9fe6', '#2a78d6', '#164a8a']):
    s = np.array(D['collapse'][N]['sigma']); e = np.array(D['collapse'][N]['E_over_h'])
    m = (s > 1e-4) & (s < 1e5)
    ax[0].loglog(s[m], e[m], color=col, lw=1.2, label=f'N = {N}')
sg = np.array(D['g']['sigma']); g = np.array(D['g']['g'])
ax[0].loglog(sg, g, 'k--', lw=1.3, label=r'half-line limit $g(\sigma)$')
ss = np.logspace(1, 5, 20)
ax[0].loglog(ss, 1 / (8 * np.sqrt(np.pi * ss)), ':', color=C[1], lw=2, label=r'$1/(8\sqrt{\pi\sigma})$')
ax[0].set(xlabel=r'$\sigma=\tau/h^2$', ylabel=r'$E_N(\tau)/h$', title='(a) Step-error collapse (Prop. 1c)', ylim=(1e-4, 1))
ax[0].legend(fontsize=7.3, loc='lower left')

# (b) closed form E_inf and remainder
P = D['prop1']; N = np.array([r['N'] for r in P]); h = 1 / N
ax[1].loglog(N, [r['Einf'] for r in P], 'o', color=C[0], ms=6, label=r'$E_N(\infty)$ computed')
Nf = np.linspace(4, 100, 100); hf = 1 / Nf
ax[1].loglog(Nf, hf ** 2 / 3 - hf ** 4 / 30, '-', color=C[0], lw=1, label=r'$h^2/3-h^4/30$ (closed form)')
ax[1].loglog(N, [r['sup_E_minus_hg'] for r in P], 's', color=C[2], ms=5, label=r'$\sup_\tau|E_N-h\,g(\tau/h^2)|$')
ax[1].loglog(N, 0.5 * h, '^', color=C[1], ms=6, label=r'$E_N(0^+)=h/2$')
ax[1].loglog(Nf, 0.5 * hf, '-', color=C[1], lw=1)
ax[1].set(xlabel='finite volumes $N$', ylabel='step-response error', title='(b) Exact limits (Prop. 1a,b)')
ax[1].legend(fontsize=7.3, loc='lower left')

# (c) sharp worst-case gain and observed errors
T = D['thm']; N = np.array([r['N'] for r in T])
for lab, col, name in [('FV', C[1], 'finite volume'), ('hybrid_universal', C[2], 'hybrid, universal fit (20 states, any N)'),
                       ('hybrid_fitN', C[0], 'hybrid, fitted per N (16 states)')]:
    ax[2].loglog(N, [r[lab]['gain'] for r in T], '-', color=col, lw=1.4,
                 label=f"{name} (order {D['order_' + lab + '_gain']:.2f})")
    ax[2].loglog(N, [r[lab]['max_err'] / np.abs(1.471) for r in T], 'o', color=col, ms=5, mfc='white')
ax[2].plot([], [], '-', color=INK2, lw=1.4, label='line: sharp worst-case gain (Thm. 2)')
ax[2].plot([], [], 'o', color=INK2, mfc='white', label=r'circle: observed error$/\|q\|_\infty$, random profile')
ax[2].set(xlabel='finite volumes $N$', ylabel=r'$\ell_\infty$ error gain', title='(c) Sharp error gain (Thms. 2-3)', ylim=(5e-7, 0.5))
ax[2].legend(fontsize=7.3, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=1)
for a in ax[1:]:
    a.xaxis.set_major_locator(FixedLocator([5, 10, 20, 40, 80])); a.xaxis.set_major_formatter(FixedFormatter(['5', '10', '20', '40', '80']))
    a.xaxis.set_minor_formatter(NullFormatter())
fig.savefig(ROOT + '/figures/fig18_hybrid_theory.png'); plt.close(fig)
print('ok')
