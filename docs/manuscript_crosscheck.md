# Manuscript ↔ repository cross-check

This audit was performed against the current 20-page manuscript source supplied
with the project and the live `main` branch.

## Numerically matched sections

| Manuscript quantity | Repository source | Status |
|---|---|---|
| nominal held-out mean / median / p05 | `results/nominal/final_validation_summary.json` | exact at reported precision |
| routine ±5% nominal/robust means | `results/robust/robust_model_comparison.csv` | exact |
| ±10% robust mean | robust comparison table | exact |
| ±15% conservative/OOD comparison | robust comparison table | exact |
| neural/GRAPE benchmark means and latency | `results/benchmarks/` | exact |
| Hadamard simulated/experimental (C_{HS}) table | `experimental_hs_correlations.csv` | exact |
| PPS central (C_{HS}) | `pps_reference.csv` | exact |
| systematic-sensitivity ranges | Step-08 frozen tables | exact to floating-point roundoff |
| final effective SD and joint intervals | `tomography_joint_effective_uncertainty.csv` | exact at manuscript precision |

## Standalone Monte-Carlo column

The released Step-07 implementation reproduces the same uncertainty model and
central reconstruction, but its finite Monte-Carlo realization is not
bit-for-bit identical to the particular realization frozen into the manuscript
table. The difference is small and consistent with platform/RNG/eigendecomposition
sampling variation; it does not affect the final joint/effective uncertainty
layer.

For direct manuscript comparison, the rounded values printed in the current
manuscript are frozen in:

```text
results/tomography/manuscript_uncertainty_reference.csv
```

The executable Step-07 output remains separately available as
`tomography_uncertainty_mc.csv`; it is not silently altered to force agreement
with a different finite random realization.

## Model provenance

- `nominal_best_state_dict.pt` is a pure inference state dictionary. A
  tensor-by-tensor comparison with the original nominal publication export
  found identical parameter keys and values (maximum absolute difference 0).
- `robust_final_state_dict.pt` is byte-identical to the robust state
  dictionary in the frozen robust-training export.
- `scripts/verify_models.py` performs strict architectural loading and a
  forward-pass sanity check for both models.

## Experimental archive scope

The original laboratory digitization working archive contains 363 CSV exports:
242 R/I reconstruction inputs plus 121 pre-combined display/export traces.

The released canonical tomography dataset contains the 242 R/I inputs actually
consumed by every reconstruction and uncertainty script. The original full
working archive has SHA-256

```text
7560282110335180c691789db62781888544e335728d646e5f72ddcc5d6721a5
```

and size 4,996,965 bytes. The released R/I-only ZIP has its own checksum in
`MANIFEST_SHA256.csv`.

## Known scope limitation

The underlying raw acquisition export for the manuscript's earlier
continuous-parameter alpha-sweep hardware demonstration was not present in the
experiment source archive supplied for this repository rebuild. The final
manuscript figure assets exist in the manuscript package, but that one
sub-experiment cannot be independently regenerated from raw acquisition data
using this Git repository alone.

All Hadamard RF-sweep hardware data, device waveforms, digitized R/I spectra,
tomography, uncertainty analyses, and final matrix/correlation figures are
covered by the released repository.
