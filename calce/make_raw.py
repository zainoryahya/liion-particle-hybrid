"""Build calce/raw.npz from the CALCE INR18650-20R (cell SP20-1) characterization spreadsheets.
   lc  : low-current (C/20) OCV test, columns [Duration (s), Pgm step, mV, mA, Temperature]
   cap : initial-capacity test,       columns [Test_Time, Step_Index, Cycle_Index, Current, Voltage, Charge_Ah, Discharge_Ah]
   inc : incremental OCV test (Arbin channel sheets), same 7 columns as cap
Place the three .xlsx files (names as downloaded from CALCE) in calce/ and run:  python calce/make_raw.py"""
import os, glob, numpy as np, openpyxl
ROOT = os.environ.get('LIION_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, 'calce')


def arbin(path):
    b = openpyxl.load_workbook(path, read_only=True); rows = []
    for n in b.sheetnames:
        ws = b[n]
        if not hasattr(ws, 'iter_rows'):
            continue
        it = ws.iter_rows(values_only=True); head = next(it, None)
        if not head or 'Test_Time(s)' not in head:
            continue
        ix = [head.index(c) for c in ['Test_Time(s)', 'Step_Index', 'Cycle_Index', 'Current(A)', 'Voltage(V)',
                                      'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)']]
        rows += [[r[i] for i in ix] for r in it if r and r[ix[0]] is not None]
    return np.array(rows, float)


def lowcurrent(path):
    ws = openpyxl.load_workbook(path, read_only=True).worksheets[0]
    it = ws.iter_rows(values_only=True); next(it)
    return np.array([[np.nan if v is None else v for v in r[:5]] for r in it if r and r[0] is not None], float)


out = dict(lc=lowcurrent(glob.glob(os.path.join(D, '*low current OCV test_SP20-1.xlsx'))[0]),
           cap=arbin(glob.glob(os.path.join(D, '*Initial capacity_SP20-1.xlsx'))[0]),
           inc=arbin(glob.glob(os.path.join(D, '*Incremental OCV test_SP20-1.xlsx'))[0]))
for k, v in out.items():
    print(k, v.shape)
np.savez_compressed(os.path.join(D, 'raw.npz'), **out)
