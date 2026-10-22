"""Regenerate the six paper-1 figures for JES: 300 dpi TIFF (LZW) for submission, vector PDF for the LaTeX build,
print width <= 7 in (double column), fonts >= 6.5 pt at print size. Patches the original scripts at run time."""
import os; ROOT = os.environ.get('LIION_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
import os, sys, re, runpy

OUT = ROOT + '/figures/jes'; os.makedirs(OUT, exist_ok=True)
RC = ("'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, "
      "'legend.fontsize': 7, 'figure.dpi': 300, 'pdf.fonttype': 42")

GS2x2 = ("fig = plt.figure(figsize=({w}, {h})); _gs = fig.add_gridspec(2, 2, hspace={hs}, wspace={ws}); "
         "ax = [fig.add_subplot(_gs[0, 0]), fig.add_subplot(_gs[0, 1]), fig.add_subplot(_gs[1, 0]), fig.add_subplot(_gs[1, 1])]")
GS3_bot = ("fig = plt.figure(figsize=({w}, {h})); _gs = fig.add_gridspec(2, 2, hspace={hs}, wspace={ws}); "
           "ax = [fig.add_subplot(_gs[0, 0]), fig.add_subplot(_gs[0, 1]), fig.add_subplot(_gs[1, 0])]")
GS3_top = ("fig = plt.figure(figsize=({w}, {h})); _gs = fig.add_gridspec(2, 2, hspace={hs}, wspace={ws}); "
           "ax = [fig.add_subplot(_gs[0, :]), fig.add_subplot(_gs[1, 0]), fig.add_subplot(_gs[1, 1])]")

JOBS = [
    # (script, cwd, figure name, layout line, extra replacements)
    (ROOT + '/ext/fig_theory.py', ROOT + '/ext', 'figure1',
     GS3_bot.format(w=6.6, h=5.4, hs=0.5, ws=0.38),
     [("title='(b) Exact limits (Prop. 1a,b)')", "title='(b) Exact limits (Prop. 1a,b)', ylim=(3e-8, 0.4))"),
      ("title='(a) Step-error collapse (Prop. 1c)'", "title='(a) Step-error collapse (Proposition 1c)'"),
      ("title='(b) Exact limits (Prop. 1a,b)'", "title='(b) Exact limits (Proposition 1a, b)'"),
      ("title='(c) Sharp error gain (Thms. 2-3)'", "title='(c) Sharp error gain (Theorems 1, 2)'"),
      ("label='line: sharp worst-case gain (Thm. 2)'", "label='line: sharp worst-case gain (Theorem 1)'"),
      ("loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=1)", "loc='center left', bbox_to_anchor=(1.08, 0.5), ncol=1)")]),
    (ROOT + '/ext/fig_hybrid.py', ROOT + '/ext', 'figure2',
     GS2x2.format(w=6.8, h=5.4, hs=0.45, ws=0.36),
     [("ax[1].legend(fontsize=7.5)", "ax[1].set_ylim(3e-10, 1); ax[1].legend(fontsize=6.5, loc='lower left', ncol=2)"),
      ("xytext=(5, 3), fontsize=7, color=INK2", "xytext=((-4, -13) if (lab[3:5] == '10' and K == '8') else ((7, -3) if (lab[3:5] == '20' and K == '8') else (5, 3))), ha=('center' if (lab[3:5] == '10' and K == '8') else 'left'), fontsize=6.5, color=INK2"),
      ("ax[3].legend(fontsize=7.5, loc='upper right')", "ax[3].set_ylim(top=2e3); ax[3].legend(fontsize=6.8, loc='upper right')"),
      ("title='Step response at the surface'", "title='(a) Step response at the surface'"),
      ("title='Random 1-s current profile'", "title='(b) Random 1-s current profile'"),
      ("title='NASA B0005 cell, dynamic current'", "title='(c) NASA B0005 cell, dynamic current'"),
      ("title='Accuracy vs model size'", "title='(d) Accuracy vs model size'"),
      ("fontsize=7.5", "fontsize=6.8")]),
    (ROOT + '/ext/fig_compare_tf_v2.py', ROOT + '/ext', 'figure3',
     "fig, ax = plt.subplots(1, 2, figsize=(6.8, 3.0), gridspec_kw=dict(wspace=0.32))",
     [("sty = {", "_L = {'FV, two-cell extrapolation (Zeng 2013)': 'FV, two-cell extrapolation', 'Pade of exact transfer function (Forman 2011)': 'Padé approximant of exact transfer function', 'Pade m=10 (11 states)': 'Padé m=10 (11 states)', 'Zeng N=20 (20 states)': 'two-cell extrap. N=20 (20 states)'}\nsty = {"),
      ("label=k)", "label=_L.get(k, k))"),
      ("sharp $\\ell_\\infty$ error gain (Thm. 1)", "sharp $\\ell_\\infty$ error gain"),
      ("bbox_to_anchor=(0.5, -0.17), ncol=2)", "bbox_to_anchor=(0.5, -0.2), ncol=1, fontsize=6.5)"),
      ("fontsize=7, loc='upper center', bbox", "loc='upper center', bbox")]),
    (ROOT + '/ext/fig_integration.py', ROOT + '/ext', 'figure4',
     GS2x2.format(w=6.8, h=5.6, hs=0.5, ws=0.36),
     [("ylim=(0.02, 150))", "ylim=(1e-3, 150))"),
      ("ax[0].legend(fontsize=7, loc='lower left')", "ax[0].legend(fontsize=6.5, loc='lower left')"),
      ("title='(b) Drive cycle, 7-state ROMs')", "title='(b) Drive cycle, 7-state ROMs', ylim=(-100, 50))"),
      ("ylim=(1e-3, 30))", "ylim=(5e-6, 30))"),
      ("ax[3].legend(fontsize=7, loc='center right')", "ax[3].legend(fontsize=6.5, loc='lower left')"),
      ("ax[3].annotate(", "(lambda *a, **k: None)("),
      ("fontsize=7.5", "fontsize=6.8")]),
    (ROOT + '/calce/fig_calce_dyn.py', ROOT + '/calce', 'figure5',
     GS2x2.format(w=6.8, h=5.8, hs=0.55, ws=0.36),
     [("title='(b) Fit on DST 80%, validate on all')", "title='(b) Fit on DST 80%, validate on all', ylim=(0, 50))"),
      ("[k.replace('_', '\\n') + '%' for k in tests], fontsize=7.5)", "[k.replace('_', ' ') + '%' for k in tests], fontsize=6.5, rotation=40, ha='right')"),
      ("title='(a) CALCE INR18650-20R, US06, 25 °C (validation)')", "title='(a) CALCE INR18650-20R, US06, 25 °C (validation)', ylim=(3.15, 3.8))"),
      ("title='(a) CALCE INR18650-20R, US06, 25 °C (validation)'", "title='(a) CALCE INR18650-20R, US06, 25 °C'"),
      ]),
    (ROOT + '/samsung/fig_ss.py', ROOT + '/samsung', 'figure6',
     GS3_top.format(w=6.8, h=5.8, hs=0.45, ws=0.38),
     [("Pi = np.array([r['Pi'] for r in SW])", "_pp = json.load(open('fit_UDDS+US06_hybrid10.json'))['p']; _W = json.load(open('hybrid_ocp_ss.json'))['window']; _RAT = (_pp['tau_p'] * (_W['p0'] - _W['p100'])) / (_pp['tau_n'] * (_W['n100'] - _W['n0'])); Pi = np.array([r['Pi'] for r in SW]) * max(1.0, _RAT)"),
      ("ax[2].axvline(0.40, color=INK2, lw=0.6, ls=':'); ax[2].text(0.42, 1.08,", "ax[2].axvline(0.40 * max(1.0, _RAT), color=INK2, lw=0.6, ls=':'); ax[2].text(0.47, 1.08,"),
      ("xlabel=r'$\\Pi = I_{\\max}\\,\\tau_n/(3\\cdot3600\\,Q_n)$'", "xlabel=r'$\\Pi = \\max_e\\, I_{\\max}\\,\\tau_e/(3\\cdot3600\\,Q_e)$'"),
      ("title='(a) Samsung INR21700-30T, LA92, 25 °C (validation)')", "title='(a) Samsung INR21700-30T, LA92, 25 °C (validation)', ylim=(3.3, 4.0))"),
      ("loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=3)", "loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=1)"),
      ("loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2)", "loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=1)")]),
]

SAVE = '''
def _jes_save(fig, name):
    import matplotlib.pyplot as _p
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches='tight')
    fig.savefig(f"{OUT}/{name}.tif", dpi=300, bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'})
    fig.savefig(f"{OUT}/{name}.png", dpi=300, bbox_inches='tight')
'''

only = sys.argv[1:]
for script, cwd, name, layout, reps in JOBS:
    if only and name not in only:
        continue
    src = open(script).read()
    src = re.sub(r"'figure\.dpi': 150", RC, src)
    src, n = re.subn(r"^fig, ax = plt\.subplots\(.*\)$", layout, src, flags=re.M)
    assert n == 1, (script, 'layout')
    for a, b in reps:
        assert a in src, (script, a)
        src = src.replace(a, b)
    src, n = re.subn(r"fig\.savefig\((?:ROOT \+ )?'[^']+'\)", f"_jes_save(fig, '{name}')", src)
    assert n == 1, (script, 'save')
    src = SAVE.replace('{OUT}', OUT) + src
    os.chdir(cwd); sys.path.insert(0, cwd)
    exec(compile(src, script + ' [JES]', 'exec'), {'__name__': '__jes__', '__file__': script})
    from PIL import Image
    im = Image.open(f'{OUT}/{name}.tif'); print(name, im.size, f'{im.size[0]/300:.2f} x {im.size[1]/300:.2f} in')
