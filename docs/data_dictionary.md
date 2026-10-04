# Data dictionary

This document describes the canonical data and result assets in the release.

## Trained models

| Path | Format | Description |
|---|---|---|
| <code>data/checkpoints/nominal_best_state_dict.pt</code> | PyTorch state dict | Selected nominal neural compiler weights from training step 48,600; architecture/physics settings are stored in <code>configs/nominal.json</code>. |
| <code>data/checkpoints/robust_final_state_dict.pt</code> | PyTorch state dict | Selected risk-aware neural compiler trained under the ±5% RF envelope. |

The public repository keeps one canonical state dictionary for each trained model. The released nominal tensor values are exactly equal to the original publication-export state dictionary even though the serialized file container is a compact reserialization.

## Nominal validation data

| Path | Shape / format | Description |
|---|---|---|
| <code>data/nominal/validation_quaternions.npy</code> | \((512,4)\), float32 | Frozen Haar-random SU(2) target quaternions. |
| <code>data/nominal/validation_fidelities.npy</code> | \((512,)\), float32 | Corresponding nominal unitary fidelities. |
| <code>results/nominal/final_validation_summary.json</code> | JSON | Aggregate statistics for the 512-gate validation set. |

## SpinQ hardware waveforms

Root:

~~~text
data/hardware/spinq/
~~~

The canonical device set contains ten <code>.spinq</code> files:

$$
2\ {\rm models}
\times
5\ {\rm acquired\ RF\ conditions}
=
10.
$$

Conditions:

$$
-10,-5,0,+5,+10\%.
$$

Each file contains exactly 300 rows with:

1. SpinQ amplitude;
2. phase in degrees;
3. dwell time in microseconds.

The dwell time is 35 μs for every row.

Supporting files:

| Path | Purpose |
|---|---|
| <code>device_pulse_manifest.csv</code> | source mapping plus amplitude/phase summary |
| <code>hardware_run_sheet.csv</code> | acquired-condition order and simulated observables |
| <code>paper_acquired_conditions.csv</code> | direct device-file ↔ tomography-condition mapping |
| <code>amplitude_conversion_reference.csv</code> | Hz-to-device working-scale reference points |
| <code>SHA256SUMS.txt</code> | checksums for the canonical device waveforms |

## Digitized experimental spectra

Canonical R/I reconstruction archive:

~~~text
data/experimental/digitized_spectra.zip
~~~

Original pre-combined display/export archive:

~~~text
data/experimental/combined_display_exports.zip
~~~

Browsable reconstruction inputs:

~~~text
data/experimental/spectra/
~~~

Inventory used by tomography:

- 11 experimental conditions;
- 11 readout rotations per condition;
- real and imaginary quadratures;
- 242 CSV traces total.

Condition labels:

| Label | Meaning |
|---|---|
| <code>PPS</code> | pseudo-pure-state reference |
| <code>NM10</code>, <code>NM5</code>, <code>N0</code>, <code>NP5</code>, <code>NP10</code> | nominal pulse at −10%, −5%, 0%, +5%, +10% RF gain |
| <code>RM10</code>, <code>RM5</code>, <code>R0</code>, <code>RP5</code>, <code>RP10</code> | risk-aware pulse at the same RF-gain settings |

Readout labels:

~~~text
III, XII, IXI, IIX, IXX, XXX, YII, IYI, IIY, YYI, YYY
~~~

Each released reconstruction readout has:

- <code>,R.csv</code> — digitized real spectral trace;
- <code>,I.csv</code> — digitized imaginary spectral trace.

The separate archival ZIP contains the 121 original <code>combined.csv</code> display/export traces (11 conditions × 11 readouts). These denser pre-combined exports are not inputs to any released reconstruction or uncertainty calculation and are preserved exactly rather than regenerated from rounded R/I data.

## Stress-test results

Root:

~~~text
results/stress_test/
~~~

Files:

- <code>stress_rf_gain.csv</code>
- <code>stress_common_b0.csv</code>
- <code>stress_spin1_offset.csv</code>
- <code>stress_spin2_offset.csv</code>
- <code>stress_spin3_offset.csv</code>
- <code>stress_j_offset.csv</code>
- <code>stress_phase_bias.csv</code>
- <code>stress_clock_error.csv</code>
- <code>stress_sweep_summary.csv</code>

Units follow the column names: Hz, degrees, ppm, fractional RF gain, or fidelity.

## Risk-aware training and evaluation results

Root:

~~~text
results/robust/
~~~

| File | Description |
|---|---|
| <code>robust_training_history.csv</code> | validation history of the reported risk-aware fine-tuning run |
| <code>robust_model_comparison.csv</code> | nominal vs risk-aware compiler across nominal/routine/OOD suites |
| <code>rf_sweep_nominal_vs_robust.csv</code> | dense RF-gain comparison for plotting |

Only the ±5% RF envelope is used for training. Wider RF ranges in these files are evaluation-only.

## Optimizer benchmark results

Root:

~~~text
results/benchmarks/
~~~

| File | Description |
|---|---|
| <code>benchmark_method_summary.csv</code> | per-method aggregate fidelity/control/time statistics |
| <code>benchmark_joint_robustness.csv</code> | routine and conservative joint uncertainty evaluation |
| <code>benchmark_rf_sweep.csv</code> | mean/p05/min fidelity versus RF error |
| <code>benchmark_b0_sweep.csv</code> | mean/p05/min fidelity versus common B0 offset |
| <code>benchmark_optimizer_convergence.csv</code> | optimizer iteration histories |
| <code>neural_latency_throughput.csv</code> | neural batch latency and throughput |

Compilation/optimization times are in seconds.

## Tomography results

Root:

~~~text
results/tomography/
~~~

| File | Meaning |
|---|---|
| <code>experimental_hs_correlations.csv</code> | central nominal/robust Hadamard deviation-matrix correlations |
| <code>pps_reference.csv</code> | PPS reference \(C_{\mathrm{HS}}\) |
| <code>tomography_uncertainty_mc.csv</code> | standalone stochastic reconstruction-MC summaries |
| <code>hardware_difference_uncertainty_mc.csv</code> | standalone robust-minus-nominal MC differences |
| <code>tomography_systematic_sensitivity.csv</code> | full systematic-sensitivity diagnostics |
| <code>tomography_systematic_sensitivity_compact.csv</code> | compact sensitivity envelope |
| <code>tomography_joint_effective_uncertainty.csv</code> | final manuscript-facing effective uncertainty |

The tomography objects are **Hermitian traceless deviation matrices**, not unit-trace density matrices.

The normalized Hilbert–Schmidt correlation is

$$
C_{\mathrm{HS}}(D_1,D_2)
=
\frac{\operatorname{Re}\operatorname{Tr}(D_1D_2)}
{\sqrt{\operatorname{Tr}(D_1^2)\operatorname{Tr}(D_2^2)}}.
$$

## Configurations

Root:

~~~text
configs/
~~~

- <code>nominal.json</code>
- <code>stress_test.json</code>
- <code>robust_training.json</code>
- <code>benchmark.json</code>
- <code>hardware_experiment.json</code>

These are compact human-readable records of the frozen settings. The executable notebooks/scripts remain authoritative for control flow.
