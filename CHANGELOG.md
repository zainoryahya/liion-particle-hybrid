# Changelog

## v1.1.1 (2026-11-14)

Corrections after an independent audit of the manuscript and the code. No conclusion of the paper changes.

- Gain horizon: up to v1.1.0 the gains were evaluated only to tau = 5, which underestimates the gain of the corrected schemes, whose slowest correction states decay more slowly. `ext/horizon.py` now continues the samples geometrically to tau = 2000 (every per-mesh correction state has decayed), and `ext/hybrid_theory2.py` adds a closed-form far tail for the universal correction (rates down to 1e-6/h^2). Effect: per-mesh gains +0.4 to +4 % (N = 10: 2.57e-4 -> 2.59e-4); universal gains +10.5 % (they now equal E_N(inf) = h^2/3 - h^4/30 to three digits); orders unchanged (local orders at dtau = 2.8e-7: 1.89-2.00). `gain_tau5` keeps the old values. `ext/gain_check.py` also checks the tail. `ext/universal_eta.py` uses the same tail.
- Zero eigenvalue of the FV matrix: identified with a relative threshold (|lambda| < 1e-9 max|lambda|) instead of 1e-12, which misclassified it for N = 40 over long horizons (affected only gains beyond tau = 5).
- New `ext/pybamm_check.py`: the two-cell kernel reproduces PyBaMM 26.10's default particle surface value, and the flux-based kernel its "use bcs" option, to below 1e-7.
- New `ext/float32_check.py`: single-precision implementation of the corrected models.
- `samsung/fig_ss.py`: the CALCE twin markers in Fig. 6c are read from `calce/twin_mc.json` (ten-realization means) instead of hard-coded single-realization values.
- `jes/make_ga.py`: new graphical abstract. `jes/make_figs_jes.py`: panel titles follow the renumbered results (Remark 1, Lemma 1, Theorem 1).
- `verify_quick.py`: expected values updated (G_FV = 9.6832e-2 over the sampled horizon; per-mesh hybrid gain 2.591e-4 with tau_min = 1e-6, tolerance 0.5 %).
- README, CITATION.cff, .zenodo.json: new title, appendix numbering, DOI note.

## v1.1.0 (2026-11-14)

Version accompanying the submission to the Journal of Energy Storage (revised after an editorial decision at J. Electrochem. Soc.).

- Gains are now evaluated over five dimensionless diffusion times (`ext/horizon.py`) instead of a fixed 1800 samples; `ext/gain_check.py` checks this against using every sample. Affects Table 1 (gains of the corrected schemes), Table 2 and Fig. 3.
- Universal correction: 40 rates in [1e-6, 1e8] (was 20 in [1e-6, 1e3]); second order in the tests down to dtau/h^2 of about 7e-6 (`ext/hybrid_theory2.py`, `universal_small_dtau`); fit term in `ext/universal_eta.py`.
- `calce/calce_model.py`: new particle kernel `twocell` (linear two-cell surface extrapolation, the PyBaMM default).
- `calce/calce_twin.py`: `TWIN_KINDS=twocell` option; new `calce/calce_twin_mc.py` (ten noise realizations).
- `ext/compare_tf3.py`: closed-form step responses (exact zero-order hold) at arbitrary sample times.
- Fig. 1c uses the long-horizon gains; Fig. 6c plots tau_n against Pi_n; Fig. 3b shaded band 6-8e-5.

## v1.0.1 (2026-11-08)

Reproducibility fixes after an independent check of the repository. No result of the paper changes.

- `ext/cost.py`: fixed a `NameError: __file__` that stopped the script. Added timing of the offline set-up steps quoted next to Table 2. These are now written to `ext/out/offline_cost.json`; before this release, no script produced that file.
- `ext/hybrid_voltage.py`: now also computes plain finite volumes with N = 40, 80 and 160, and writes them to `hybrid_voltage.json` under `plain_more`. These are the values quoted in the text and plotted in Fig. 2d. Previously, re-running the script removed this block and `figure2` stopped with a `KeyError`.
- `ext/rN_check.py`: now covers N = 2–8 as well, as stated in Remark 1, and documents the horizon used for N ≤ 80.
- `jes/make_ga.py`: the x axis is now documented as Π of the negative electrode (Π_n). No change to the plot.
- `README.md`:
  - completed the map from the paper to the code (Fig. 5 needs only stored results; Fig. 6a needs the raw Samsung data; Fig. 6a,b; Table 5; Appendix C; the 45 °C fit);
  - added the Zenodo DOI.
- `CITATION.cff`, `.zenodo.json`: added the DOI and the ORCID iDs; version 1.0.1.
- `verify_quick.py`: corrected the stated run time.

## v1.0.0 (2026-10-22)

Initial release accompanying the submission to the Journal of The Electrochemical Society.
