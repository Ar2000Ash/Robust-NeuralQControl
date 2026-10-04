# Manuscript figure reproduction map

This page maps every figure in the current 20-page manuscript source to its
released numerical source or regeneration path.

## Fully regenerable numerical figures

| Manuscript figure asset | Reproduction source |
|---|---|
| `stress_rf_gain.pdf` | `notebooks/02_stress_test_nominal.ipynb` + `results/stress_test/stress_rf_gain.csv` |
| `stress_common_B0_gamma.pdf` | stress notebook + `stress_common_b0.csv` |
| `stress_phase_bias.pdf` | stress notebook + `stress_phase_bias.csv` |
| `stress_J_offset.pdf` | stress notebook + `stress_j_offset.csv` |
| `robust_training_convergence.pdf` | `notebooks/03_train_robust.ipynb` + `robust_training_history.csv` |
| `nominal_haar_distribution.pdf` | nominal validation arrays under `data/nominal/` |
| `rf_sweep_nominal_vs_robust.pdf` | `results/robust/rf_sweep_nominal_vs_robust.csv` |
| `benchmark_fidelity_grouped.pdf` | `scripts/04_benchmark_optimizers.py` + benchmark tables |
| `benchmark_latency_bars.pdf` | benchmark script + `neural_latency_throughput.csv` |
| `IIINP10.pdf` | `scripts/10_plot_hardware_figures.py` + NP10 III R/I spectra |
| `IIIRP10.pdf` | hardware figure script + RP10 III R/I spectra |
| `hardware_simulation_vs_experiment.pdf` | hardware figure script + `experimental_hs_correlations.csv` |
| `density_NP10_real.pdf` | hardware figure script + released tomography reconstruction |
| `density_RP10_real.pdf` | hardware figure script + released tomography reconstruction |

Fresh figure outputs are written under `outputs/` by default so the frozen
numerical release is not overwritten.

## Continuous-family hardware demonstration

The manuscript also contains the earlier continuous-parameter hardware
demonstration using the frozen assets:

- `12Peaks.pdf`
- `20.pdf`
- `40.pdf`
- `60.pdf`
- `80.pdf`
- `Phase.pdf`

These six final manuscript figure assets are present in the manuscript source
package supplied for submission, but the corresponding raw acquisition export
was **not present in the Git experiment source archive supplied for this
reproducibility rebuild**. Consequently, the public repository does not claim
raw-data regeneration of this one hardware sub-experiment.

For provenance, the frozen manuscript-asset SHA-256 values are:

| asset | SHA-256 |
|---|---|
| `12Peaks.pdf` | `c19d649df7a9418506df49d260a878fefbc895096dc73a58b172005d3fd42810` |
| `20.pdf` | `5bc344c52754ef763a6b3aeb56a6b13813995ed7b7c40f94d1d98f6e2a1f2efd` |
| `40.pdf` | `96354fb6aae7bf113309ba2335bf605e1b3bd4ce425d65f21832ebae441ed241` |
| `60.pdf` | `c59d6201e6236b89e61c2e47ecda715e32d1f8d9c78e5580f6203ba6b5766ef8` |
| `80.pdf` | `54d7aada6e01eab14262b64a7a3902ee101b750b57f7dd3053357fd9b09110c8` |
| `Phase.pdf` | `e289e4ce7b7b56d28d07f996d764a7df36def5dd7e116a0956c79d1a5bd48690` |

This is the only known manuscript-figure raw-data gap found in the final audit.
