# Data dictionary

- `data/experimental/digitized_spectra.zip` — digitized NMR spectra for the PPS reference and nominal/robust Hadamard RF-sweep conditions.
- `results/tables/robust_model_comparison.csv` — nominal versus robust compiler evaluation, including out-of-training-range RF tests.
- `results/tables/stress_*.csv` — frozen nominal sensitivity sweeps that identify RF-gain error as the dominant perturbation in the tested ranges.
- `results/tables/benchmark_*.csv` — matched neural/GRAPE/robust-GRAPE/NN-seeded-GRAPE/CRAB-SPSA results used in the optimizer comparison.
- `results/tables/experimental_hs_correlations.csv` — reconstructed deviation-matrix Hilbert–Schmidt correlations.
- `results/tables/tomography_*.csv` and `hardware_difference_uncertainty.csv` — statistical and systematic uncertainty analyses for the tomography results.

The released robust-training code uses the fixed ±5% RF training regime used for the selected model. Wider RF ranges are retained only as post-training evaluation tests.