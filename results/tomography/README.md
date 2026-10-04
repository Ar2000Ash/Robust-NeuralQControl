# Tomography results and uncertainty layers

This directory separates the deterministic central reconstruction from the
successive uncertainty analyses used in the hardware-validation workflow.

## Central reconstruction

`experimental_hs_correlations.csv` and `pps_reference.csv` are reproduced by

```bash
python scripts/06_reconstruct_tomography.py
```

using the exact digitized spectra in
`data/experimental/digitized_spectra.zip`.

## Standalone Monte-Carlo reconstruction uncertainty

`tomography_uncertainty_mc.csv` and
`hardware_difference_uncertainty_mc.csv` are the frozen outputs of
`scripts/07_tomography_uncertainty_mc.py`.

The standalone model combines:

- independent pointwise spectral-amplitude perturbations of ±2%;
- a smooth linear frequency drift across the 11 sequential readouts;
- independently sampled start/end drift offsets in ±5 Hz;
- a 0.1 Hz precomputed shift grid;
- 10,000 Monte-Carlo samples per condition and uncertainty mode;
- base random seed 12345.

Because fixed-window integration is linear in the digitized spectrum, the
pointwise amplitude model is propagated to the exact first two moments of the
integrated transition amplitudes and sampled with a moment-matched
multivariate Gaussian. The frequency drift is propagated explicitly.

The central deviation-matrix correlations reproduced by this analysis match
the deterministic reconstruction to machine precision. Re-running the frozen
Monte-Carlo analysis on a different NumPy/SciPy platform can change the last
few digits of covariance-square-root sampling statistics because the symmetric
eigendecomposition is only defined numerically; in our regression check the
difference was at approximately the 1e-9 level.

These two files are intentionally labeled **standalone MC**. They are not the
final manuscript effective-uncertainty intervals. The systematic sensitivity
analysis and the final combined effective uncertainty are handled separately
by `scripts/08_tomography_systematic_sensitivity.py` and
`scripts/09_tomography_joint_effective_uncertainty.py`.
