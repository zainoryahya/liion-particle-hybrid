"""Hybrid OCP for CALCE INR18650-20R + Figure 14 (English)."""
import os; ROOT = os.environ.get('LIION_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
import numpy as np, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import subprocess

D = json.load(open(ROOT + '/calce/ocv_balance.json'))
g = np.array(D['grid']); ocv = np.array(D['ocv']); ocv_raw = np.array(D['ocv_raw'])
TAB = np.load(ROOT + '/nasa/ocp_tables.npy', allow_pickle=True).item(); STO = TAB['sto']
tb = TAB['tables']['Mohtat2020']
# graphite window from balancing on the pseudo-OCV (physical: graphite 0.78 -> 0.015), positive linear
n100, n0, p100, p0 = 0.780, 0.015, 0.016, 0.992
th_n = n100 - (1 - g) * (n100 - n0); th_p = p100 + (1 - g) * (p0 - p100)
Up_h = ocv + np.interp(th_n, STO, tb['n'])           # U_p(th_p) = OCV + U_n(th_n)
order = np.argsort(th_p)
Up_sorted = np.minimum.accumulate(Up_h[order])        # monotonically decreasing (isotonic)
print('max isotonic correction:', round(float(np.max(Up_h[order] - Up_sorted))*1e3, 2), 'mV')
hyb = dict(sto_p=th_p[order].tolist(), Up=Up_sorted.tolist(), sto_n=STO.tolist(), Un=tb['n'].tolist(),
           window=dict(n100=n100, n0=n0, p100=p100, p0=p0), Qtot_C20=D['Qtot'],
           note='Hybrid OCP: graphite from Mohtat2020; positive derived from measured equilibrium OCV (C/20 average corrected to 2-h rest points)')
json.dump(hyb, open(ROOT + '/calce/hybrid_ocp.json', 'w'))
print('hybrid positive OCP: range', round(min(Up_h), 3), '-', round(max(Up_h), 3), 'V ; monotonically decreasing (after isotonic):', bool(np.all(np.diff(Up_sorted) <= 0)))

# ---------------- Figure 14
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 9.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.6,
                     'legend.frameon': False, 'figure.dpi': 150, 'savefig.bbox': 'tight'})
fig, ax = plt.subplots(1, 3, figsize=(14.5, 3.6), gridspec_kw=dict(wspace=0.3))
Vd = np.array(D['Vd']); Vc = np.array(D['Vc']); mc = g <= D['soc_c_max']
ax[0].plot(g * 100, Vd, color=C[1], lw=1.1, label='C/20 discharge')
ax[0].plot(g[mc] * 100, Vc[mc], color=C[0], lw=1.1, label='C/20 charge')
ax[0].plot(g * 100, ocv_raw, color=INK2, lw=1.0, ls='--', label='C/20 average')
ax[0].plot(g * 100, ocv, color=INK, lw=1.8, label='equilibrium OCV (corrected)')
ax[0].plot(np.array(D['inc_d']['soc']) * 100, D['inc_d']['V'], 'v', ms=5, color=C[1], label='2-h rest, discharge')
ax[0].plot(np.array(D['inc_c']['soc'][1:]) * 100, D['inc_c']['V'][1:], '^', ms=5, color=C[0], label='2-h rest, charge')
ax[0].set(xlabel='SOC [%]', ylabel='voltage [V]', title='CALCE INR18650-20R, 25 °C'); ax[0].legend(fontsize=7.5, loc='lower right')

# stability of the OCP selection: balancing against two OCV references
import re
log = subprocess.run(['grep', '-h', 'RMSE', ROOT + '/calce/balance_raw.log', ROOT + '/calce/balance_corr.log'],
                     capture_output=True, text=True).stdout.strip().split('\n')
half = len(log) // 2
def parse(lines):
    return {l.split()[0]: float(l.split()[2]) for l in lines}
raw, cor = parse(log[:half]), parse(log[half:])
names = sorted(raw, key=lambda k: raw[k]); y = np.arange(len(names))[::-1]
ax[1].barh(y + 0.19, [raw[n] for n in names], 0.36, color=C[2], label='reference: C/20 average')
ax[1].barh(y - 0.19, [cor[n] for n in names], 0.36, color=C[0], label='reference: corrected OCV')
ax[1].set_yticks(y, names); ax[1].grid(axis='y', visible=False)
ax[1].set(xlabel='electrode-balancing RMSE [mV]', title='Literature OCP pairs: ranking is unstable'); ax[1].legend(fontsize=7.5, loc='center right')

Q = np.array(D['Q']); m = (g > 0.03) & (g < 0.97)
ax[2].plot(Q[m], np.array(D['dvdq'])[m], color=INK, lw=1.4, label='measured')
best_raw = min(raw, key=raw.get)
ax[2].set(xlabel='discharged capacity [Ah]', ylabel='−dV/dQ [V/Ah]', title='Differential voltage', ylim=(0, 1.2))
ax[2].legend(fontsize=8)
fig.savefig(ROOT + '/figures/fig14_calce_ocv_balancing.png'); plt.close(fig)
print('ok', best_raw)
