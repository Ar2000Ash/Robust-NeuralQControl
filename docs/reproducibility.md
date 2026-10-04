# Reproducing the study

This guide separates **frozen-result reproduction** from **full recomputation**. The repository contains the trained models and principal numerical outputs, so most manuscript figures and tables can be inspected without retraining or rerunning expensive optimal-control searches.

## 1. Validate the release snapshot

From the repository root:

```bash
python scripts/verify_release.py
```

The verifier checks:

- every SHA-256 entry in `MANIFEST_SHA256.csv`;
- the presence of the nominal and risk-aware trained models;
- JSON validity for all public experiment configs;
- exactly 242 released tomography R/I CSV traces;
- exactly 121 real and 121 imaginary spectra;
- exactly ten canonical SpinQ device waveforms;
- exactly 300 rows in each canonical device waveform.

## 2. Environment

Portable dependency ranges are in:

```text
requirements.txt
environment.yml
```

The publication workflows use Python 3.13. GPU workloads require a CUDA-enabled PyTorch installation compatible with the target driver.

The historically recorded environments differ slightly between experiment stages:

- nominal/robust training: PyTorch 2.11.0, CUDA 12.8;
- frozen two-GPU benchmark export: PyTorch 2.14.0, CUDA 13.0.

The released equations, Hamiltonian, target set, uncertainty definitions, and optimizer hyperparameters are frozen independently of those environment differences.

## 3. Nominal neural compiler

Notebook:

```text
notebooks/01_train_nominal.ipynb
```

Main settings:

- 512 held-out Haar validation targets;
- 300 control slices;
- 35 μs per slice;
- 10-layer width-256 GELU MLP;
- dropout 0.25;
- AdamW, learning rate (5	imes10^{-4});
- 50,000 training steps;
- seed 20260917.

Canonical frozen model:

```text
data/checkpoints/nominal_frozen_checkpoint.pt
```

Canonical held-out validation data:

```text
data/nominal/validation_quaternions.npy
data/nominal/validation_fidelities.npy
results/nominal/final_validation_summary.json
```

The selected frozen nominal checkpoint corresponds to training step 48,600.

## 4. Frozen nominal stress diagnosis

Notebook:

```text
notebooks/02_stress_test_nominal.ipynb
```

The notebook loads the frozen nominal model and evaluates RF gain, common B0, independent spin offsets, J-coupling offsets, phase bias, clock error, and joint uncertainty profiles.

Frozen publication tables:

```text
results/stress_test/
```

The routine profile includes ±5% RF-gain uncertainty; the conservative profile extends to ±15%.

## 5. Risk-aware fine-tuning

Notebook:

```text
notebooks/03_train_robust.ipynb
```

The public experiment is one direct risk-aware fine-tuning run under a fixed **±5% RF-gain training envelope**.

Per gate, training uses:

- 8 random joint uncertainty scenarios;
- 2 exact RF-boundary scenarios;
- CVaR (alpha=0.20);
- CVaR weight 1.5;
- 1,800 maximum fine-tuning steps.

The ±7.5%, ±10%, and ±15% RF ranges are evaluation-only tests.

Canonical frozen model:

```text
data/checkpoints/robust_final_state_dict.pt
```

The selected robust checkpoint corresponds to validation step 1,500.

Frozen outputs:

```text
results/robust/
```

## 6. Optimal-control benchmark

Driver:

```text
scripts/04_benchmark_optimizers.py
```

Methods:

1. Neural nominal
2. Neural robust
3. GRAPE
4. Robust GRAPE
5. NN-seeded GRAPE
6. CRAB-SPSA

Target set:

- 11 named single-qubit gates;
- 8 additional Haar-random SU(2) targets;
- 19 targets total.

The recommended publication run uses the portable two-GPU launcher:

```bash
sbatch hpc/benchmark_two_gpu.slurm
```

The launcher contains no site-specific account information. If a cluster requires account/partition/QoS flags, supply them at submission time.

Each GPU runs an independent Python worker on a disjoint target subset. There is no distributed-gradient synchronization. Rank 0 merges the per-target records and produces the final sweeps, summaries, figures, environment record, and checksums.

Frozen benchmark tables:

```text
results/benchmarks/
```

## 7. Hadamard hardware preparation

Notebook:

```text
notebooks/05_prepare_hardware.ipynb
```

The representative experiment applies a Hadamard gate to spin 1 with ideal deviation transfer

[
Z_1 ightarrow X_1.
]

The notebook produces Hz-equivalent I/Q controls on the nine-point preparation grid

[
-10,-7.5,-5,-2.5,0,+2.5,+5,+7.5,+10%.
]

The reported hardware acquisition uses the five points

[
-10,-5,0,+5,+10%
]

for each of the two neural models.

## 8. SpinQ device export

Exporter:

```bash
python scripts/05_export_spinq_waveforms.py
```

Recorded device convention:

[
A_{m SpinQ}
=
100,
rac{sqrt{u_x^2+u_y^2}}
{8333.333333333334 {m Hz}},
]

[
phi_{m SpinQ}
=
operatorname{atan2}(u_y,u_x)
rac{180}{pi}
pmod{360^circ}.
]

Each `.spinq` file contains 300 rows with:

1. device amplitude;
2. phase in degrees;
3. 35 μs dwell time.

Canonical acquired-condition waveforms:

```text
data/hardware/spinq/device_pulses/
```

The requested RF perturbation is already baked into each waveform and must not be applied again inadvertently at the instrument.

## 9. Experimental spectra and central tomography

Exact archive:

```text
data/experimental/digitized_spectra.zip
```

Browsable reconstruction inputs:

```text
data/experimental/spectra/
```

The released reconstruction data contain:

[
11 {m conditions}
	imes
11 {m readouts}
	imes
2 {m quadratures}
=
242 {m CSV traces}.
]

Central reconstruction:

```bash
python scripts/06_reconstruct_tomography.py
```

No manual extraction is necessary.

The linear inversion uses:

- 11 readout rotations;
- 12 single-quantum transitions per readout;
- real and imaginary components;
- 264 real measurements;
- 63 Hermitian-traceless parameters.

Outputs:

```text
results/tomography/experimental_hs_correlations.csv
results/tomography/pps_reference.csv
```

## 10. Tomography uncertainty

Run in order:

```bash
python scripts/07_tomography_uncertainty_mc.py
python scripts/08_tomography_systematic_sensitivity.py
python scripts/09_tomography_joint_effective_uncertainty.py
```

### Step 07 — standalone reconstruction MC

Propagates:

- ±2% pointwise spectral-amplitude uncertainty;
- smooth linear frequency drift with endpoints in ±5 Hz;
- 10,000 samples;
- base seed 12345.

This is retained as a diagnostic layer and is not the final manuscript uncertainty model.

### Step 08 — systematic sensitivity

Evaluates sensitivity to:

- linear/quadratic baseline subtraction;
- ±1° receiver-phase perturbations;
- wider ±3° phase diagnostic;
- one-bin integration-window edge shifts;
- removal of either/both central Q3 transitions.

These are sensitivity diagnostics, not statistical confidence intervals.

### Step 09 — final joint/effective uncertainty

Jointly propagates:

- ±2% spectral-amplitude uncertainty;
- ±5 Hz linear readout drift;
- ±1-bin window-edge uncertainty;
- ±1° receiver-phase uncertainty;
- baseline-model uncertainty;
- conservative trust weights for the two central Q3 transitions.

It uses 10,000 realizations with base seed 24680 and reproduces the manuscript-facing uncertainty intervals.

## 11. Generated versus frozen artifacts

Frozen publication assets are committed under `data/` and `results/`.

Fresh executions should write to:

```text
outputs/
```

which is ignored by Git.

Small last-digit differences can occur across numerical libraries or GPU architectures because matrix exponentials, reductions, covariance eigendecompositions, and optimizer trajectories are floating-point operations. The frozen release tables/checkpoints are the numerical reference snapshot.
