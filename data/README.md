# Raw data

All battery data used in the paper are public. They are not redistributed here; download them from their providers and place them as described below. The derived results (fits, validations, OCP tables) are stored in the repository, so the figures can be regenerated without the raw data except where noted in the main README.

## 1. CALCE INR18650-20R (University of Maryland)

**Source:** CALCE Battery Research Group, battery data page, cell type INR 18650-20R. Cite F. Zheng et al., *Appl. Energy* 183, 513 (2016) and Y. Xing et al., *Appl. Energy* 113, 106 (2014).

### Characterization files (cell SP20-1)

Put these three files in `calce/`:

- `10_16_2015_Initial capacity_SP20-1.xlsx`
- `11_5_2015_low current OCV test_SP20-1.xlsx`
- `12_2_2015_Incremental OCV test_SP20-1.xlsx`

Then build `calce/raw.npz`:

```bash
python calce/make_raw.py
```

### Dynamic tests (cell SP20-2)

Put the following files in `calce/dyn/`. They are served with an `.xls` extension although they are `.xlsx`; both extensions are accepted.

- DST, FUDS, US06 and BJDST at 25 °C, from 80% and 50% SOC
- DST, FUDS and US06 at 0 °C and 45 °C, from 80% and 50% SOC
- `10_16_2015_Initial capacity_SP20-2`

Example names: `11_05_2015_SP20-2_DST_80SOC.xls`, `02_24_2016_SP20-2_0C_DST_50SOC.xls`, `12_16_2015_SP20-2_45C_US06_80SOC.xls`.

Then build `calce/dyn/dyn.npz`:

```bash
python calce/dyn/load_dyn.py
```

## 2. Samsung INR21700-30T (McMaster University)

**Source:** P. Kollmeyer, *Samsung INR21700 30T 3Ah Li-ion Battery Data*, Mendeley Data, Version 2 (2023), doi:10.17632/9xyvy2njj3.2 (CC BY 4.0).

Unzip the archive so that this folder exists:

```
data/Samsung INR21700 30T 3Ah Li-ion Battery Data/
    25degC/   ... 734_UDDS_25degC_IN21700_30T.mat, 734_US06_..., 734_LA92_..., 740_Mixed3_..., 729_C20DisCh_... etc.
    0degC/
    -10degC/
```

`samsung/load_ss.py` finds each test by its name (`_UDDS_`, `_US06_`, `_C20DisCh_`, …) in the temperature folder.

## 3. NASA B0005 (NASA Ames Prognostics Center of Excellence)

**Source:** B. Saha and K. Goebel, *Battery Data Set*, NASA Prognostics Data Repository (2007), cell B0005.

This dataset is only needed to repeat the original identification of the NASA particle model. The identified parameters and the OCP tables are stored in `nasa/refine_1.json` and `nasa/ocp_tables.npy`, and these are all that the paper's scripts read. If needed, place `B0005.mat` in `nasa/` and read it with `nasa/load_nasa.py`.
