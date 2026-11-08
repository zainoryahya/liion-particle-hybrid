# Changelog

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
