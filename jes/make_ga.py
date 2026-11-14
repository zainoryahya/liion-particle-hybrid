"""Graphical abstract (v1.1.1): three panels read left to right.
(1) After every current step the particle surface value of a finite-volume model carries an error that decays only over
    the diffusion time of one cell (flux-based extrapolation, 10 cells; PyBaMM default two-cell extrapolation, 20 cells;
    10 cells with the 16-state output correction), shown in seconds for a graphite particle with tau_n = 16.6 ks.
(2) Twin experiment on the CALCE DST current (ten noise realizations, calce/twin_mc.json): identified / true diffusion
    times, with the fit residual of each model.
(3) The screening indicator Pi_n for the two cells (and for the 21700 cell at 0 C), with the regions observed for plain FV
    with 20 cells (h Pi_n <= 0.02: diffusion-time bias <= 2.3 %; h Pi_n >= 0.04: biased parameters or a failed fit in the twin experiments).
Writes figures/jes/GA_Yahya.{png,tif,pdf}."""
import os; ROOT = os.environ.get('LIION_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('LIION_ROOT', ROOT)
import numpy as np, json, sys, io, contextlib
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, ROOT + '/ext'); sys.path.insert(0, ROOT + '/calce'); sys.path.insert(0, ROOT + '/battery')
with contextlib.redirect_stdout(io.StringIO()):
    import hybrid as H
    import calce_model as CM
C = {'fv': '#eb6834', 'tc': '#eda100', 'hy': '#2a78d6', 'ok': '#1baf7a', 'bad': '#d64545'}
INK, INK2, GRID = '#1f1e1c', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True,
                     'grid.color': GRID, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                     'pdf.fonttype': 42, 'axes.titleweight': 'bold', 'axes.titlesize': 9.5})
os.makedirs(ROOT + '/figures/jes', exist_ok=True)
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.4), gridspec_kw=dict(wspace=0.42, width_ratios=[1.15, 1.0, 1.0]))

# (1) error after a current step
TAU_N = 16.6e3
tau = np.logspace(-6, 0.3, 400); t = tau * TAU_N
Sx = CM.HT.S_exact(tau)
E10 = Sx - CM.HT.S_fv(10, tau)
E20tc = Sx - CM.S_twocell(20, tau)
cf = H.fit_correction(10); Ehy = E10 - (cf['Einf'] + np.exp(-np.outer(tau, H.MU)) @ cf['w'])
ax[0].semilogx(t, np.abs(E10), color=C['fv'], lw=2, label='10 cells')
ax[0].semilogx(t, np.abs(E20tc), color=C['tc'], lw=2, label='PyBaMM default (20 cells)')
ax[0].semilogx(t, np.abs(Ehy), color=C['hy'], lw=2, label='10 cells + 16-state correction')
ax[0].set(xlabel='time after a current step [s]', ylabel='surface stoichiometry error\nper unit flux',
          title='1  Every current step leaves an error', xlim=(1e-2, 3e4), ylim=(0, 0.06))
ax[0].legend(fontsize=7.5, loc='upper right')
ax[0].annotate('decays over the diffusion\ntime of one cell (minutes)', (100, 0.0125), xytext=(1.3e-2, 0.024), fontsize=7.5,
               color=INK2, arrowprops=dict(arrowstyle='->', color=INK2, lw=0.7))

# (2) twin experiment
MC = json.load(open(ROOT + '/calce/twin_mc.json'))['summary']
rows = [('fv N=10', '10 cells', C['fv']), ('twocell N=20', 'PyBaMM\ndefault', C['tc']), ('hybrid N=10', 'corrected', C['hy'])]
x = np.arange(len(rows)); w = 0.36
for i, (k, lab, col) in enumerate(rows):
    ax[1].bar(i - w / 2, MC[k]['tau_n']['mean'], w, color=col, alpha=0.95)
    ax[1].bar(i + w / 2, MC[k]['tau_p']['mean'], w, color=col, alpha=0.55, hatch='///', edgecolor='white')
    ax[1].text(i, 1.08, f"fit {MC[k]['rmse_mean']:.0f} mV", ha='center', fontsize=7.5, color=INK2)
ax[1].axhline(1, color=INK, lw=0.8)
ax[1].set_xticks(x, [r[1] for r in rows], fontsize=8); ax[1].grid(axis='x', visible=False)
ax[1].set(ylabel='identified / true diffusion time', ylim=(0, 1.2), title='2  Good fit, wrong parameters')
from matplotlib.patches import Patch
ax[1].legend(handles=[Patch(facecolor='#8a8986', label='negative electrode'),
                      Patch(facecolor='#c9c8c4', hatch='///', edgecolor='white', label='positive electrode')],
             fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, title='twin experiment, known parameters, 2 mV noise',
             title_fontsize=7)

# (3) screening indicator
TW = json.load(open(ROOT + '/samsung/twinT_0degC.json'))
pts = [(0.40, 'high-power 21700\nat 25 °C', C['ok']), (2.11, 'high-energy 18650\nat 25 °C', C['bad']),
       (TW['Pi_Tmean'], '21700 at 0 °C', C['bad'])]
ax[2].axvspan(0.03, 0.4, color=C['ok'], alpha=0.12, lw=0); ax[2].axvspan(0.8, 30, color=C['bad'], alpha=0.10, lw=0)
ax[2].text(0.11, 0.9, 'bias\n< 3 %', ha='center', fontsize=7.5, color=C['ok'], transform=ax[2].get_xaxis_transform())
ax[2].text(6.0, 0.9, 'biased parameters\nor failed fit', ha='center', fontsize=7.5, color=C['bad'], transform=ax[2].get_xaxis_transform())
ys = [0.55, 0.30, 0.62]
for (P, lab, col), y in zip(pts, ys):
    ax[2].plot([P], [y], 'o', color=col, ms=9, mec=INK, mew=0.8)
    ax[2].text(P * 1.25, y, lab, fontsize=7.5, va='center', color=INK)
ax[2].set_xscale('log'); ax[2].set(xlim=(0.03, 30), ylim=(0, 1), yticks=[],
                                   xlabel='peak surface flux $\\Pi$ (from peak current,\ncapacity and diffusion time)',
                                   title='3  Check before fitting (20 cells)')
ax[2].grid(axis='y', visible=False)
for ext in ('png', 'pdf'):
    fig.savefig(f'{ROOT}/figures/jes/GA_Yahya.{ext}', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(f'{ROOT}/figures/jes/GA_Yahya.tif', dpi=300, bbox_inches='tight', facecolor='white',
            pil_kwargs={'compression': 'tiff_lzw'})
print('ok')
