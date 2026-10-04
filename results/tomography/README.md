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


## Systematic-sensitivity diagnostics

`tomography_systematic_sensitivity.csv` and
`tomography_systematic_sensitivity_compact.csv` are the frozen outputs of
`scripts/08_tomography_systematic_sensitivity.py`.

This analysis keeps the standard reconstruction as the central result and
measures sensitivity to alternative analysis choices:

- linear and quadratic baseline subtraction in signal-free regions;
- independent receiver-phase perturbations of ±1 degree;
- a wider ±3 degree receiver-phase diagnostic;
- moving one integration-window edge at a time by one digitized frequency bin
  (approximately 2.5601565 Hz);
- dropping either or both central Q3 transition windows while confirming the
  reduced tomography matrix remains full rank (63).

The routine sensitivity envelope uses the baseline, ±1 degree phase, and
one-bin window-placement tests. The Q3-removal results are reported separately
as leverage diagnostics. Neither set is interpreted as a statistical
confidence interval.

A local regression run against the exact digitized spectra reproduced the
frozen systematic tables to floating-point roundoff (maximum discrepancy about
1e-15).


## Final joint/effective uncertainty

`tomography_joint_effective_uncertainty.csv` is reproduced by

```bash
python scripts/09_tomography_joint_effective_uncertainty.py
```

using the same digitized spectra and central tomography model.

This is the final manuscript-facing uncertainty layer. Each of 10,000
realizations jointly propagates:

- ±2% pointwise spectral-amplitude uncertainty;
- a linear ±5 Hz frequency drift across the sequential readouts;
- lower and upper integration-window perturbations within one digitized
  frequency bin (±2.5601565 Hz), shared across readouts;
- independent ±1 degree receiver-phase perturbations for the three spin groups;
- uncertainty between no baseline correction and fitted linear/quadratic
  baseline subtraction;
- conservative trust weights for the two partially overlapped central Q3
  transitions, handled through weighted least squares.

The original experimental reconstruction remains the reported central
`C_HS`. The joint distribution supplies the effective standard deviation and
the 2.5--97.5 percentile interval.

The archived Step-09 source did not include a tracked frozen output CSV.
Therefore the table in this directory was regenerated from the archived
analysis code and the exact released spectra. A direct manuscript regression
check confirms that all 11 conditions reproduce the published effective SD and
joint 95% interval exactly at the manuscript's five-decimal reporting
precision.

Representative final intervals are:

- PPS: SD 0.01450, [0.82378, 0.88055];
- nominal -10%: SD 0.02470, [0.32423, 0.41910];
- robust 0%: SD 0.00795, [0.80872, 0.84023];
- robust +10%: SD 0.01020, [0.76360, 0.80417].

At the ±10% RF endpoints, the nominal and robust effective intervals remain
clearly separated, preserving the principal hardware robustness conclusion
under the joint reconstruction-uncertainty model.
