"""Read CALCE SP20-2 files (Arbin; .xls is actually xlsx) -> dyn.npz. Arbin current: + = charge; we use I>0 = discharge."""
import os; ROOT = os.environ.get('LIION_ROOT', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repository root
import openpyxl, numpy as np, glob, re, os
out = {}
import io
# CALCE serves these files with an .xls extension although they are .xlsx; read them through a byte buffer
for f in sorted(glob.glob(ROOT + '/calce/dyn/*.xls') + glob.glob(ROOT + '/calce/dyn/*.xlsx')):
    b = openpyxl.load_workbook(io.BytesIO(open(f, 'rb').read()), read_only=True)
    rows = []
    for n in b.sheetnames:
        if n.startswith('Channel') and hasattr(b[n], 'iter_rows'):
            rows += list(b[n].iter_rows(min_row=2, values_only=True))
    A = np.array([[r[1], r[4], r[6], r[7], r[8], r[9]] for r in rows if r[1] is not None], float)
    name = os.path.basename(f)
    m = re.search(r'SP20-2_(?:(0C|45C)_)?(DST|FUDS|US06|BJDST)_(\d+)SOC', name)
    key = 'capacity' if 'Initial' in name else f"{m.group(1) or '25C'}_{m.group(2)}_{m.group(3)}"
    out[key] = A
    print(f'{key:18s} {len(A):6d} rows  t={A[-1,0]/3600:.2f} h  I[{A[:,2].min():.2f},{A[:,2].max():.2f}]  V[{A[:,3].min():.3f},{A[:,3].max():.3f}]  steps {np.unique(A[:,1]).astype(int).tolist()[:20]}')
np.savez_compressed(ROOT + '/calce/dyn/dyn.npz', **out)
