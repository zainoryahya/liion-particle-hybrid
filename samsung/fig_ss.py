import os; ROOT = os.environ.get('LIION_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
import numpy as np, json, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ss_model as S
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 9.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
                     'lines.linewidth': 1.6, 'legend.frameon': False, 'figure.dpi': 150, 'savefig.bbox': 'tight'})
V = json.load(open('validate_ss.json')); SW = json.load(open('sweep_ss.json'))
fig, ax = plt.subplots(1, 3, figsize=(16, 3.9), gridspec_kw=dict(wspace=0.3, width_ratios=[1.3, 1.25, 1]))
x = np.array(json.load(open('fit_UDDS+US06_hybrid10.json'))['x'])
d = S.load_test('LA92'); Vm = S.simulate(x, d, 'hybrid', 10)[d['gi']]; t = d['td']
w = (t > 1500) & (t < 1900)
ax[0].plot(t[w], d['Vd'][w], color=INK, lw=0.9, label='measured')
ax[0].plot(t[w], Vm[w], color=C[0], lw=1.0, label=f"hybrid SPM (RMSE {V['hybrid N=10']['LA92']:.0f} mV, whole cycle)")
a2 = ax[0].twinx(); a2.plot(t[w], d['Id'][w], color=C[3], lw=0.5, alpha=0.6); a2.set_ylabel('current [A]', color=C[3]); a2.spines['right'].set_visible(True); a2.grid(False)
ax[0].set(xlabel='time in LA92 cycle [s]', ylabel='voltage [V]', title='(a) Samsung INR21700-30T, LA92, 25 °C (validation)'); ax[0].legend(fontsize=7.3, loc='lower left')
keys = ['UDDS', 'US06', 'HWFET', 'LA92'] + [f'Mixed{i}' for i in range(1, 9)]
xx = np.arange(len(keys)); wd = 0.27
for i, (lab, col) in enumerate([('hybrid N=10', C[0]), ('FV N=20', C[3]), ('FV N=10', C[1])]):
    ax[1].bar(xx + (i - 1) * wd, [V[lab][k] for k in keys], wd, color=col, label=f'identified with {lab}')
ax[1].axvspan(-0.5, 1.5, color=GRID, alpha=0.6, lw=0); ax[1].text(0.5, 1.0, 'fitted', ha='center', fontsize=7.5, color=INK2)
ax[1].set_xticks(xx, [k.replace('Mixed', 'Mix') for k in keys], fontsize=7, rotation=45); ax[1].grid(axis='x', visible=False)
ax[1].set(ylabel='voltage RMSE [mV]', title='(b) Fit on UDDS+US06, validate on 10 cycles', ylim=(0, 27)); ax[1].legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=3)
Pi = np.array([r['Pi'] for r in SW])
for lab, col, m, name in [('fv10', C[1], 'o', 'FV N=10'), ('fv20', C[3], 's', 'FV N=20'), ('hybrid10', C[0], 'D', 'hybrid N=10')]:
    ok = np.array([r[lab]['rmse'] < 50 for r in SW])
    y = np.array([r[lab]['tau_n_ratio'] for r in SW])
    ax[2].semilogx(Pi[ok], y[ok], m + '-', color=col, ms=5, lw=1.2, label=name)
    if (~ok).any():
        ax[2].semilogx(Pi[~ok], y[~ok], 'x', color=col, ms=7, mew=1.5)
ax[2].plot([], [], 'x', color=INK2, label='fit failed (residual > 50 mV; noise 2 mV)')
ax[2].plot([2.11], [0.28], 'o', color=C[1], ms=8, mfc='white', mew=1.5); ax[2].plot([2.11], [0.67], 's', color=C[3], ms=8, mfc='white', mew=1.5)
ax[2].annotate('CALCE 18650 twin\n(open markers)', (2.11, 0.28), xytext=(0.12, 0.27), fontsize=7, color=INK2, arrowprops=dict(arrowstyle='->', color=INK2, lw=0.6))
ax[2].axvline(0.40, color=INK2, lw=0.6, ls=':'); ax[2].text(0.42, 1.08, 'Samsung 30T, 25 °C', fontsize=7, color=INK2)
TW = json.load(open('twinT_0degC.json')); P0 = TW['Pi_Tmean']
for lab, col, m in [('fv N=10', C[1], 'o'), ('fv N=20', C[3], 's'), ('hybrid N=10', C[0], 'D')]:
    ax[2].plot([P0], [TW['fits'][lab]['ratio']['tau_n']], m, color=col, ms=8, mec=INK, mew=1.4)
ax[2].annotate('Samsung 30T at 0 °C\n(twin, measured cell T;\nblack-edged markers)', (P0, TW['fits']['fv N=10']['ratio']['tau_n']),
               xytext=(0.12, 0.47), fontsize=7, color=INK2, arrowprops=dict(arrowstyle='->', color=INK2, lw=0.6))
ax[2].axhline(1, color=INK2, lw=0.5)
ax[2].set(xlabel=r'$\Pi = I_{\max}\,\tau_n/(3\cdot3600\,Q_n)$', ylabel=r'identified / true $\tau_n$', title='(c) Twin experiments: bias vs $\\Pi$', ylim=(0.2, 1.15))
ax[2].legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2)
fig.savefig(ROOT + '/figures/fig22_samsung_30T.png'); plt.close(fig); print('ok')
