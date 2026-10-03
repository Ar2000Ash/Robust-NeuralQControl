# Robust Neural Quantum Control

**Reproducibility repository for _Robust neural pulse compilation of continuous quantum gates with optimal-control benchmarking and NMR hardware validation_.**

This project learns a reusable neural map from arbitrary single-qubit SU(2) targets to 300-slice RF waveforms for a three-spin liquid-state NMR processor. The nominal compiler is trained end-to-end through the differentiable propagator, stress-tested under structured model mismatch, and then fine-tuned with a mean-plus-CVaR objective under a fixed ±5% global RF/B1 gain envelope. The resulting compiler is compared with target-specific optimal-control methods and tested experimentally with NMR tomography.

## Repository map

```text
Robust-NeuralQControl/
├── notebooks/
│   ├── 01_train_nominal.ipynb
│   ├── 02_stress_test_nominal.ipynb
│   ├── 03_train_robust.ipynb
│   └── 05_prepare_hardware.ipynb
├── scripts/
│   ├── 04_benchmark_optimizers.py
│   ├── 06_reconstruct_tomography.py
│   ├── 07_tomography_uncertainty_mc.py
│   ├── 08_tomography_systematic_sensitivity.py
│   ├── 09_tomography_joint_effective_uncertainty.py
│   └── densitymatrix.py
├── hpc/benchmark_two_gpu.slurm
├── configs/
├── data/
│   └── experimental/digitized_spectra.zip
├── results/tables/
└── docs/
```

## Scientific pipeline

`Haar SU(2) target → nominal neural compiler → frozen robustness diagnosis → CVaR robust fine-tuning → optimal-control benchmark → NMR hardware preparation → tomography + uncertainty`

### Physical model

The numerical model uses the same three-spin Hamiltonian throughout training and benchmarking,

```text
H0 = π Σ_i ν_i Z_i + π Σ_{i<j} J_ij (X_i X_j + Y_i Y_j + Z_i Z_j)
```

with 300 control slices of 35 μs each (10.5 ms total). The network is a 10-layer width-256 GELU MLP that emits the two RF quadratures for every slice.

### Robust training

The publication code has one robust-training regime: **±5% global RF/B1 gain uncertainty** together with the other routine uncertainty channels. Every training microbatch includes eight random joint scenarios and the exact ±5% RF boundary cases. The loss is

```text
L = L_nominal + L_mean + 1.5 L_CVaR + L_control.
```

The ±7.5%, ±10%, and ±15% RF envelopes appear only in post-training evaluation and are not used to update the robust model.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
unzip data/experimental/digitized_spectra.zip -d data/experimental
```

For the full experiment order and hardware notes, see [`docs/reproducibility.md`](docs/reproducibility.md). The frozen paper tables are in [`results/tables`](results/tables), and every repository file is listed in `MANIFEST_SHA256.csv`.

## Reproducibility notes

- The digitized real/imaginary spectra required by the tomography pipeline are included in full; redundant pre-combined CSVs are omitted because the reconstruction does not use them.
- Machine-specific `/content`, Windows desktop, and personal scratch paths have been removed from the publication code.
- Generated outputs are written under `outputs/` and are ignored by Git.
- The 50 kHz amplitude cap in the model is a numerical training cap; hardware waveforms must be mapped to instrument units using the experimental RF/Rabi calibration.

## Citation

Citation metadata is provided in `CITATION.cff`.