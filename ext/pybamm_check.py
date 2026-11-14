"""Check that the particle kernels used in this repository reproduce PyBaMM's own particle surface value.

PyBaMM (tested with version 26.10) is run with its single particle model (SPM), the Chen2020 parameter set (constant
particle diffusivities) and its default particle discretization (finite volumes, 20 cells per particle, surface value by
linear extrapolation from the last two cells, option "use bcs": False). A constant 1C discharge from 50% SOC is applied,
which is a unit flux step for each particle. The surface stoichiometry is normalized by the flux,
S_pb(tau) = (y(tau) - u0) / q, u0 the uniform initial stoichiometry (= particle average at t = 0), with tau = D t / R^2 and q obtained from the change of the particle-averaged stoichiometry
(d ubar / d tau = -3 q), and compared with
  - the repository's two-cell kernel calce_model.S_twocell(20, tau)  (PyBaMM default);
  - the flux-based kernel S_fv(20, tau), Eq. (7) of the paper, against PyBaMM run with "use bcs": True;
  - the exact solution.
Writes ext/out/pybamm_check.json.
"""
import os, sys, json, io, contextlib
ROOT = os.environ.get('LIION_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('LIION_ROOT', ROOT)
import numpy as np
import pybamm
sys.path.insert(0, ROOT + '/calce'); sys.path.insert(0, ROOT + '/ext'); sys.path.insert(0, ROOT + '/battery')
with contextlib.redirect_stdout(io.StringIO()):
    import calce_model as CM

N = 20
OUT = dict(pybamm_version=pybamm.__version__, N=N, parameter_set='Chen2020', runs={})


def run(use_bcs):
    model = pybamm.lithium_ion.SPM()
    pv = pybamm.ParameterValues('Chen2020')
    pv['Current function [A]'] = pv['Nominal cell capacity [A.h]']          # 1C, constant from t = 0
    sm = model.default_spatial_methods
    if use_bcs:
        opts = {'extrapolation': {'order': {'gradient': 'quadratic', 'value': 'linear'}, 'use bcs': True}}
        for k in ['negative particle', 'positive particle']:
            sm[k] = pybamm.FiniteVolume(options=opts)
    sim = pybamm.Simulation(model, parameter_values=pv, spatial_methods=sm,
                            solver=pybamm.IDAKLUSolver(rtol=1e-10, atol=1e-12))
    t = np.r_[0.0, np.geomspace(1e-3, 1800.0, 600)]
    sol = sim.solve(t_eval=t, t_interp=t, initial_soc=0.5)
    res = {}
    for e, E in [('n', 'negative'), ('p', 'positive')]:
        D = pv[f'{E.capitalize()} particle diffusivity [m2.s-1]']
        R = pv[f'{E.capitalize()} particle radius [m]']
        y = sol[f'X-averaged {E} particle surface stoichiometry'].entries
        ub = sol[f'Average {E} particle stoichiometry'].entries
        tt = sol['Time [s]'].entries
        tau = D * tt / R ** 2
        q = -(ub[-1] - ub[0]) / (3 * tau[-1])          # d ubar/d tau = -3 q
        Spb = (y - ub[0]) / q             # PyBaMM evaluates t = 0 with the current already on
        m = tau > 0
        ref_tc = CM.S_twocell(N, tau[m]); ref_fv = CM.HT.S_fv(N, tau[m]); ref_ex = CM.HT.S_exact(tau[m])
        lin = np.abs(np.diff(ub[m]) / np.diff(tau[m]) + 3 * q).max() / (3 * abs(q))
        res[e] = dict(D_over_R2=float(D / R ** 2), tau_max=float(tau[-1]), q=float(q), ubar_linearity=float(lin),
                      jump_at_first_sample=float(Spb[m][0]),
                      maxdiff_twocell=float(np.abs(Spb[m] - ref_tc).max()),
                      maxdiff_fluxbased=float(np.abs(Spb[m] - ref_fv).max()),
                      max_err_vs_exact=float(np.abs(Spb[m] - ref_ex).max()),
                      tau_first=float(tau[m][0]))
    return res


for use_bcs in [False, True]:
    lab = 'use_bcs_True' if use_bcs else 'default'
    OUT['runs'][lab] = run(use_bcs)
    for e, r in OUT['runs'][lab].items():
        print(f"PyBaMM {pybamm.__version__} {lab:13s} {e}: |S_pb - S_twocell| = {r['maxdiff_twocell']:.2e}   "
              f"|S_pb - S_fluxbased| = {r['maxdiff_fluxbased']:.2e}   |S_pb - S_exact| = {r['max_err_vs_exact']:.3e}   "
              f"(tau up to {r['tau_max']:.2f})")
json.dump(OUT, open(ROOT + '/ext/out/pybamm_check.json', 'w'), indent=1)
