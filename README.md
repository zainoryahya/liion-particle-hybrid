# Particle diffusion under battery-management-system currents: error analysis and a hybrid correction

Code and derived results for the paper

> Z. R. Yahya and S. Mat Zin, *Discretization error in particle diffusion biases identified lithium-ion battery parameters: error analysis and a hybrid correction*, manuscript (2026).

The repository regenerates every figure and table of the paper.

- **From the stored results** (`*.json` files), regenerating the figures takes seconds.
- **From scratch**, the public battery data must be downloaded first (see [data/README.md](data/README.md)). The parameter fits take minutes to hours.

## What is in the repository

| Folder | Contents |
|---|---|
| `ext/` | Theory and numerics. Exact solution, finite-volume (FV) step error (Proposition 1), sharp worst-case gain (Theorem 1), hybrid correction (per-mesh and universal, Theorem 2), comparison with graded meshes, Hermite and two-cell extrapolation, Chebyshev collocation, Padé, DRA and BT-SPA. Also: cost table, reduced-order models, EKF, SPMe, and the concentration-dependent diffusivity test. Results are in `ext/out/`. |
| `battery/` | Single particle model (FV) used by `ext/`. |
| `nasa/` | Open-circuit potential tables and the identified parameter vector of the NASA B0005 particle model (derived data, inputs to `ext/common.py`). |
| `calce/` | CALCE INR18650-20R: data conversion, OCV balancing, fits with three particle solvers, validation, twin experiment, bias, EKF on measured data. |
| `samsung/` | McMaster Samsung INR21700-30T: OCP, fits, validation, twin experiments, Π sweep, Arrhenius model with exact warped-time modal recursion (C library in `samsung/csrc/`). |
| `jes/` | Scripts that produce the journal figures (`figure1`–`figure6`, 300 dpi TIFF/PDF/PNG) and the graphical abstract. |
| `figures/` | Output folder for figures. |
| `data/` | Place for the downloaded raw data (not redistributed). |

## Installation

Python 3.10 or newer (tested with Python 3.13):

```bash
pip install -r requirements.txt
make -C samsung/csrc        # builds libmodrec.so (needs a C compiler: gcc or clang)
```

All scripts locate the repository root automatically. To use another location, set the environment variable `LIION_ROOT`.

## Quick check (about 10 seconds, no battery data needed)

```bash
python verify_quick.py
```

The script recomputes key numbers of the paper from scratch and compares them with the published values:

- the exact step-error limits of Proposition 1;
- the residue identity of the exact solution;
- the sharp gain of plain FV (Table 1);
- the hybrid gain;
- the universal weights;
- the Chebyshev and DRA baselines.

It ends with `ALL CHECKS PASSED`.

## Quick start: regenerate the paper figures from the stored results

```bash
python jes/make_figs_jes.py figure1 figure2 figure3 figure4 figure5   # need only the stored results
python jes/make_figs_jes.py figure6                                   # panel (a) also needs the raw Samsung data (see data/README.md)
python jes/make_ga.py                                         # graphical abstract (needs the stored Samsung sweep)
```

The figures are written to `figures/jes/`.

## Map from the paper to the code

Run the scripts in `ext/` from any folder. Run the scripts in `calce/` and `samsung/` from inside their own folder, because several of them read and write files in the working directory.

| Paper item | Script(s) | Output |
|---|---|---|
| Exact solution, Proposition 1, Table 1, Fig. 1, Appendix A (r_N for N = 2–320), Table B.1 (universal correction at small dtau) | `ext/hybrid_theory2.py`, `ext/rN_check.py`; plot `ext/fig_theory.py` | `ext/out/hybrid_theory.json`, `rN_check.json` |
| Fig. 2 (step responses, NASA voltage test, plain FV with N = 10–160) | `ext/hybrid.py`, `ext/hybrid_voltage.py`; plot `ext/fig_hybrid.py` | `hybrid.json`, `hybrid_voltage.json` |
| Comparison of approximations, Fig. 3 | `ext/compare_tf.py`, `ext/compare_tf2.py`, `ext/compare_tf3.py` (Chebyshev, DRA), `ext/graded.py`, `ext/hermite_check.py`; plot `ext/fig_compare_tf_v2.py`. All gains are taken over five diffusion times (`ext/horizon.py`); `ext/gain_check.py` compares this with using every sample. | `compare_tf*.json`, `graded.json`, `hermite.json`, `gain_check.json` |
| Cost table (Table 2) and offline set-up times quoted in the text | `ext/cost.py` (about 2 min); per-step timing `ext/bench.c` (`gcc -O2 -o bench ext/bench.c && ./bench`). Timings depend on the machine; the stored files hold the values in the paper. | `cost.json`, `offline_cost.json`, `bench.csv` |
| ROMs, EKF, SPMe (Fig. 4) | `ext/rom_hybrid.py`, `ext/ekf_hybrid.py`, `ext/spme_hybrid.py`; plot `ext/fig_integration.py` | `rom_hybrid.json`, `ekf_hybrid.json`, `spme_hybrid.json` |
| Concentration-dependent diffusivity | `DC_DECADES=1 python ext/dc_test.py` and `DC_DECADES=2.5 python ext/dc_test.py` | `dc_test_1.json`, `dc_test_2.5.json` |
| CALCE data preparation | `python calce/make_raw.py`, `python calce/dyn/load_dyn.py`, then `calce/ocv_balance.py`, `calce/ocv_balance_raw.py`, `calce/hybrid_and_fig.py` | `calce/raw.npz`, `calce/dyn/dyn.npz`, `hybrid_ocp.json` |
| CALCE fits (Table 3) | `python calce_fit.py 25C hybrid 10`, `... 25C fv 20`, `... 25C fv 10`; 0 °C: `python calce_fit.py 0C hybrid 10 fit_25C_hybrid10.json fit_25C_hybrid10.json`; 45 °C (discussed in the text): `python calce_fit.py 45C hybrid 10 fit_25C_hybrid10.json fit_25C_hybrid10.json` | `fit_*.json` |
| CALCE validation, bias, twin, EKF (Tables 3–4, Fig. 5) | `calce_validate.py`, `calce_bias.py`, `calce_twin.py` (`TWIN_KINDS=twocell` for the two-cell extrapolation, the PyBaMM default), `calce_twin_mc.py` (ten noise realizations, about 45 min), `calce_ekf.py`; plot `fig_calce_dyn.py` | `validate_25C.json`, `bias_25C.json`, `twin_bias.json`, `twin_bias_twocell.json`, `twin_mc.json`, `ekf_calce_r6.json` |
| Samsung OCP and 25 °C fits (Table D.1, 25 °C column; Fig. 6a,b) | `ocp_ss.py`; `python ss_fit.py UDDS+US06 hybrid 10` (and `fv 10`, `fv 20`); `ss_validate.py`; `ss_twin.py` | `hybrid_ocp_ss.json`, `fit_UDDS+US06_*.json`, `validate_ss.json`, `twin_ss.json` |
| Π sweep (Fig. 6c, graphical abstract) | `ss_sweep.py`; plots `fig_ss.py`, `jes/make_ga.py` | `sweep_ss.json` |
| Low temperature (Table D.1, 0 °C; Fig. 6c, black-edged markers) | `python ss_fit_T.py 0degC hybrid 10` (also `fv 10`, `fv 20`, and `-10degC`), `python ss_validate_T.py 0degC`, `python ss_twin_T.py 0degC`; plot `fig_ss.py` | `fitT_*.json`, `validateT_*.json`, `twinT_*.json` |
| Universal correction coefficients (Table B.2) and its fit term (Appendix B) | `ext/hybrid_theory2.py`, `ext/universal_eta.py` | `ext/out/hybrid_theory.json` → `universal`, `ext/out/universal_eta.json` |
| NASA B0005 parameters (Table C.1), used by `ext/common.py` | `nasa/load_nasa.py` (data preparation) | `nasa/refine_1.json`, `nasa/ocp_tables.npy` |

Notes:

- `samsung/ss_fit_T.py` reproduces the published fits by default (`PAPER=1`: activation energies in [0, 100] kJ/mol, no entropic term). `PAPER=0` runs the wider-bound sensitivity check mentioned in the text and writes `fitT2_*.json`.
- The isothermal 0 °C and −10 °C fits that start `ss_fit_T.py` are produced by `TEMPF=0degC python ss_fit.py UDDS+US06 hybrid 10`.
- Differential evolution uses fixed seeds. Results can differ in the last digits across library versions and processors.

## Data

The raw data are public but are not redistributed here. Download links and the expected folder layout are given in [data/README.md](data/README.md).

## How to cite

Please cite the paper above and this archive: doi:[10.5281/zenodo.23556199](https://doi.org/10.5281/zenodo.23556199) (see `CITATION.cff`).

## Changes

See [CHANGELOG.md](CHANGELOG.md).

## Licence

MIT (see `LICENSE`). The battery data remain under the terms of their providers.
