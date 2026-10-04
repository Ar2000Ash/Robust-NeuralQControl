# Data dictionary

This document describes the canonical data and result assets in the release.

## Trained models

| Path | Format | Description |
|---|---|---|
| `data/checkpoints/nominal_frozen_checkpoint.pt` | PyTorch checkpoint | Selected nominal neural compiler; includes model weights plus nominal run metadata/configuration. |
| `data/checkpoints/robust_final_state_dict.pt` | PyTorch state dict | Selected risk-aware neural compiler trained under the ±5% RF envelope. |

The public repository keeps one canonical copy of each trained model.

## Nominal validation data

| Path | Shape / format | Description |
|---|---|---|
| `data/nominal/validation_quaternions.npy` | ((512,4)), float32 | Frozen Haar-random SU(2) target quaternions. |
| `data/nominal/validation_fidelities.npy` | ((512,)), float32 | Corresponding nominal unitary fidelities. |
| `results/nominal/final_validation_summary.json` | JSON | Aggregate statistics for the 512-gate validation set. |

## SpinQ hardware waveforms

Root:

```text
data/hardware/spinq/
```

The canonical device set contains ten `.spinq` files:

[
2 {m models}
	imes
5 {m acquired RF conditions}
=
10.
]

Conditions:

[
-10,-5,0,+5,+10%.
]

Each file contains exactly 300 rows with:

1. SpinQ amplitude;
2. phase in degrees;
3. dwell time in microseconds.

The dwell time is 35 μs for every row.

Supporting files:

| Path | Purpose |
|---|---|
| `device_pulse_manifest.csv` | source mapping plus amplitude/phase summary |
| `hardware_run_sheet.csv` | acquired-condition order and simulated observables |
| `paper_acquired_conditions.csv` | direct device-file ↔ tomography-condition mapping |
| `amplitude_conversion_reference.csv` | Hz-to-device working-scale reference points |
| `SHA256SUMS.txt` | checksums for the canonical device waveforms |

## Digitized experimental spectra

Exact reconstruction archive:

```text
data/experimental/digitized_spectra.zip
```

Browsable copy:

```text
data/experimental/spectra/
```

Inventory:

- 11 experimental conditions;
- 11 readout rotations per condition;
- real and imaginary quadratures;
- 242 CSV traces total.

Condition labels:

| Label | Meaning |
|---|---|
| `PPS` | pseudo-pure-state reference |
| `NM10`, `NM5`, `N0`, `NP5`, `NP10` | nominal pulse at −10%, −5%, 0%, +5%, +10% RF gain |
| `RM10`, `RM5`, `R0`, `RP5`, `RP10` | risk-aware pulse at the same RF-gain settings |

Readout labels:

```text
III, XII, IXI, IIX, IXX, XXX, YII, IYI, IIY, YYI, YYY
```

Each released readout has:

- `,R.csv` — digitized real spectral trace;
- `,I.csv` — digitized imaginary spectral trace.

The original laboratory working archive also contained 121 pre-combined display/export CSVs. They are not inputs to any released reconstruction or uncertainty calculation and are not treated as canonical numerical inputs.

## Stress-test results

Root:

```text
results/stress_test/
```

Files:

- `stress_rf_gain.csv`
- `stress_common_b0.csv`
- `stress_spin1_offset.csv`
- `stress_spin2_offset.csv`
- `stress_spin3_offset.csv`
- `stress_j_offset.csv`
- `stress_phase_bias.csv`
- `stress_clock_error.csv`
- `stress_sweep_summary.csv`

Units follow the column names: Hz, degrees, ppm, fractional RF gain, or fidelity.

## Risk-aware training and evaluation results

Root:

```text
results/robust/
```

| File | Description |
|---|---|
| `robust_training_history.csv` | validation history of the reported risk-aware fine-tuning run |
| `robust_model_comparison.csv` | nominal vs risk-aware compiler across nominal/routine/OOD suites |
| `rf_sweep_nominal_vs_robust.csv` | dense RF-gain comparison for plotting |

Only the ±5% RF envelope is used for training. Wider RF ranges in these files are evaluation-only.

## Optimizer benchmark results

Root:

```text
results/benchmarks/
```

| File | Description |
|---|---|
| `benchmark_method_summary.csv` | per-method aggregate fidelity/control/time statistics |
| `benchmark_joint_robustness.csv` | routine and conservative joint uncertainty evaluation |
| `benchmark_rf_sweep.csv` | mean/p05/min fidelity versus RF error |
| `benchmark_b0_sweep.csv` | mean/p05/min fidelity versus common B0 offset |
| `benchmark_optimizer_convergence.csv` | optimizer iteration histories |
| `neural_latency_throughput.csv` | neural batch latency and throughput |

Compilation/optimization times are in seconds.

## Tomography results

Root:

```text
results/tomography/
```

| File | Meaning |
|---|---|
| `experimental_hs_correlations.csv` | central nominal/robust Hadamard deviation-matrix correlations |
| `pps_reference.csv` | PPS reference (C_{HS}) |
| `tomography_uncertainty_mc.csv` | standalone stochastic reconstruction-MC summaries |
| `hardware_difference_uncertainty_mc.csv` | standalone robust-minus-nominal MC differences |
| `tomography_systematic_sensitivity.csv` | full systematic-sensitivity diagnostics |
| `tomography_systematic_sensitivity_compact.csv` | compact sensitivity envelope |
| `tomography_joint_effective_uncertainty.csv` | final manuscript-facing effective uncertainty |

The tomography objects are **Hermitian traceless deviation matrices**, not unit-trace density matrices.

The normalized Hilbert–Schmidt correlation is

[
C_{HS}(D_1,D_2)
=
rac{operatorname{Re}operatorname{Tr}(D_1D_2)}
{sqrt{operatorname{Tr}(D_1^2)operatorname{Tr}(D_2^2)}}.
]

## Configurations

Root:

```text
configs/
```

- `nominal.json`
- `stress_test.json`
- `robust_training.json`
- `benchmark.json`
- `hardware_experiment.json`

These are compact human-readable records of the frozen settings. The executable notebooks/scripts remain authoritative for control flow.
