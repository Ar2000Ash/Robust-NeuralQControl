# Risk-Aware Neural Pulse Compilation for NMR Quantum Control

**Reproducibility repository for _Risk-aware neural pulse compilation of continuous quantum gates with optimal-control benchmarking and NMR hardware validation_.**

This repository contains the trained neural compilers, publication notebooks and scripts, frozen numerical results, exact SpinQ device waveforms, digitized NMR spectra, and the full tomography/uncertainty pipeline used in the study.

The core task is an amortized map

$$
q \in S^3 \longmapsto \{u_x(t_k),u_y(t_k)\}_{k=1}^{300},
$$

from an arbitrary single-qubit SU(2) target to a 10.5 ms pulse for a three-spin liquid-state NMR system. A nominal compiler is trained end-to-end through the differentiable propagator, diagnosed under structured model mismatch, and then fine-tuned with a mean-plus-CVaR objective under a **fixed ±5% global RF-gain training envelope**. Wider RF ranges are used only for post-training evaluation.

---

## What is released

- **Trained models** — frozen nominal and risk-aware neural compiler state dictionaries used by the released workflows.
- **Training and evaluation code** — nominal training, frozen stress diagnosis, risk-aware fine-tuning, and optimizer benchmarking.
- **Classical baselines** — GRAPE, robust GRAPE, NN-seeded GRAPE, and CRAB-SPSA.
- **Hardware preparation** — Hadamard pulse generation plus the recorded Ankara/SpinQ device conversion.
- **Exact device waveforms** — the ten <code>.spinq</code> files used for the reported nominal/robust RF-sweep conditions.
- **Experimental data** — 242 individually browsable real/imaginary digitized spectra, the canonical R/I reconstruction-input ZIP, and the original 121 pre-combined display/export spectra in a separate archival ZIP.
- **Tomography** — central deviation-matrix reconstruction, standalone Monte-Carlo uncertainty, systematic-sensitivity diagnostics, and final joint/effective uncertainty propagation.
- **Frozen paper results** — compact CSV/JSON/NPY assets for replotting the principal numerical results without rerunning expensive optimization.

---

## Representative frozen results

### Neural compilers

| Evaluation | Nominal compiler | Risk-aware compiler |
|---|---:|---:|
| Nominal evaluation | 0.992978 | 0.973617 |
| Routine joint uncertainty, RF ±5% | 0.918983 | **0.966034** |
| Routine joint uncertainty, RF ±10% | 0.751187 | **0.943869** |
| Conservative/OOD uncertainty, RF ±15% | 0.574994 | **0.895094** |

The risk-aware model is trained only within the ±5% RF envelope; ±7.5%, ±10%, and ±15% results are evaluation-only tests.

### Optimizer benchmark

| Method | Mean nominal fidelity | Mean per-target compile / optimization time |
|---|---:|---:|
| Neural nominal | 0.991268 | **0.000445 s** |
| Neural robust | 0.973257 | **0.000438 s** |
| GRAPE | **0.999547** | 748.19 s |
| Robust GRAPE | 0.998888 | 1776.11 s |
| NN-seeded GRAPE | 0.999384 | 29.97 s |
| CRAB-SPSA | 0.175207 | 193.80 s |

The neural contribution is amortized sub-millisecond compilation; the classical optimizers remain stronger in absolute target-specific fidelity.

### Hardware tomography

At the RF endpoints, the reconstructed Hadamard deviation-matrix correlations are:

| RF error | Nominal \(C_{\mathrm{HS}}\) | Risk-aware \(C_{\mathrm{HS}}\) |
|---:|---:|---:|
| −10% | 0.402463 | **0.774062** |
| +10% | 0.369366 | **0.780671** |

The final joint uncertainty analysis keeps the robust and nominal endpoint intervals clearly separated.

---

## Scientific pipeline

~~~text
Haar SU(2) targets
        │
        ▼
01_train_nominal.ipynb
        │
        ▼
frozen nominal compiler
        │
        ├──► 02_stress_test_nominal.ipynb
        │
        ▼
03_train_robust.ipynb
        │
        ▼
frozen risk-aware compiler
        │
        ├──► 04_benchmark_optimizers.py
        │
        ▼
05_prepare_hardware.ipynb
        │
        ▼
Hz-equivalent Hadamard I/Q controls
        │
        ▼
05_export_spinq_waveforms.py
        │
        ▼
SpinQ device waveforms → NMR acquisition
        │
        ▼
digitized spectra
        │
        ▼
06 reconstruction → 07 MC → 08 sensitivity → 09 joint uncertainty
~~~

---

## Quick start

### 1. Create an environment

~~~bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~

A Conda specification is also provided in <code>environment.yml</code>. Install a CUDA-enabled PyTorch build compatible with the target GPU/driver when running the training or benchmark workloads.

### 2. Verify the release

~~~bash
python scripts/verify_release.py
~~~

This checks the repository-wide SHA-256 manifest, model/config assets, experimental-spectrum inventories, and canonical device waveforms.

### 3. Reproduce the experimental tomography

No manual ZIP extraction is required.

~~~bash
python scripts/06_reconstruct_tomography.py
python scripts/07_tomography_uncertainty_mc.py
python scripts/08_tomography_systematic_sensitivity.py
python scripts/09_tomography_joint_effective_uncertainty.py
~~~

The frozen reference outputs are under <code>results/tomography/</code>.

### 4. Run the two-GPU optimizer benchmark

~~~bash
sbatch hpc/benchmark_two_gpu.slurm
~~~

The SLURM file contains no user/account/site-specific identifiers. Supply any cluster-required account, partition, QoS, or other scheduling options externally.

---

## Repository map

~~~text
configs/                    frozen experiment settings
data/
  checkpoints/              trained nominal + risk-aware models
  nominal/                  held-out Haar targets and fidelities
  hardware/spinq/           exact device waveforms and manifests
  experimental/
    digitized_spectra.zip           canonical R/I reconstruction archive
    combined_display_exports.zip    original 121 pre-combined display exports
    spectra/                        242 browsable R/I CSV traces
notebooks/                  training, stress, robust, hardware workflows
scripts/                    benchmark, device export, tomography, verification
hpc/                        portable two-GPU benchmark launcher
results/                    frozen publication outputs
docs/                       reproducibility, data dictionary, structure
~~~

See:

- [Reproducibility guide](docs/reproducibility.md)
- [Data dictionary](docs/data_dictionary.md)
- [Repository structure](docs/repository_structure.md)
- [HPC execution notes](hpc/README.md)

---

## Physical model

The three-spin drift Hamiltonian is

$$
H_0
=
\pi\sum_i \nu_i Z_i
+
\pi\sum_{i<j} J_{ij}
\left(
X_iX_j+Y_iY_j+Z_iZ_j
\right),
$$

with

$$
\nu=(-921,\;40.75,\;700)\ {\rm Hz},
\qquad
J=(-64,\;24.4,\;34.1)\ {\rm Hz}.
$$

Each pulse contains 300 slices of 35 μs, giving a total duration of 10.5 ms. The 50 kHz model amplitude cap is a **numerical optimization bound**, not the SpinQ hardware full-scale calibration.

---

## Reproducibility and provenance

- The frozen trained-model state dictionaries are tracked under <code>data/checkpoints/</code>; the nominal weights are tensor-identical to the state dictionary exported from the selected nominal training checkpoint.
- The ten canonical hardware <code>.spinq</code> files are byte-identical to the archived device-ready waveforms.
- <code>data/experimental/digitized_spectra.zip</code> is the canonical R/I reconstruction archive consumed by the released tomography scripts.
- <code>data/experimental/combined_display_exports.zip</code> preserves the 121 original pre-combined laboratory display/export traces exactly; these files are archival and are not tomography inputs.
- The 242 R/I reconstruction traces are also exposed individually under <code>data/experimental/spectra/</code>.
- Central tomography values reproduce the archived result table exactly.
- Systematic-sensitivity outputs reproduce to floating-point roundoff.
- The final joint/effective uncertainty table reproduces the manuscript values at the reported precision.
- <code>MANIFEST_SHA256.csv</code> records SHA-256 hashes for the complete release snapshot, excluding only the manifest itself.

Generated runs belong under <code>outputs/</code> and are ignored by Git.

---

## Citation

Please use the metadata in [CITATION.cff](CITATION.cff) when citing the code or data.

## Rights

No open-source license has been granted in this release. See [LICENSE](LICENSE) for the current rights notice.
