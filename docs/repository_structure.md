# Repository structure

The release is organized by **scientific role**, not by development chronology.

```text
Robust-NeuralQControl/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── environment.yml
├── MANIFEST_SHA256.csv
├── configs/
│   ├── README.md
│   ├── nominal.json
│   ├── stress_test.json
│   ├── robust_training.json
│   ├── benchmark.json
│   └── hardware_experiment.json
├── notebooks/
│   ├── 01_train_nominal.ipynb
│   ├── 02_stress_test_nominal.ipynb
│   ├── 03_train_robust.ipynb
│   └── 05_prepare_hardware.ipynb
├── scripts/
│   ├── 04_benchmark_optimizers.py
│   ├── 05_export_spinq_waveforms.py
│   ├── 06_reconstruct_tomography.py
│   ├── 07_tomography_uncertainty_mc.py
│   ├── 08_tomography_systematic_sensitivity.py
│   ├── 09_tomography_joint_effective_uncertainty.py
│   └── verify_release.py
├── hpc/
│   ├── README.md
│   └── benchmark_two_gpu.slurm
├── data/
│   ├── checkpoints/
│   │   ├── nominal_frozen_checkpoint.pt
│   │   └── robust_final_state_dict.pt
│   ├── nominal/
│   │   ├── validation_quaternions.npy
│   │   └── validation_fidelities.npy
│   ├── hardware/
│   │   └── spinq/
│   │       ├── device_pulses/
│   │       ├── device_pulse_manifest.csv
│   │       ├── hardware_run_sheet.csv
│   │       └── paper_acquired_conditions.csv
│   └── experimental/
│       ├── digitized_spectra.zip
│       ├── spectra_manifest.csv
│       └── spectra/
├── results/
│   ├── README.md
│   ├── nominal/
│   ├── stress_test/
│   ├── robust/
│   ├── benchmarks/
│   └── tomography/
└── docs/
    ├── reproducibility.md
    ├── data_dictionary.md
    └── repository_structure.md
```

## Canonical versus generated files

### Canonical release assets

Files under the following locations are part of the frozen publication snapshot:

```text
configs/
data/
notebooks/
scripts/
hpc/
results/
docs/
```

along with the top-level metadata/environment files.

### Generated artifacts

Fresh executions should write transient or regenerated outputs under:

```text
outputs/
logs/
```

These directories are excluded by `.gitignore`.

## Why some stages are notebooks and others scripts

The neural training/stress/hardware-preparation stages remain notebooks because their original scientific workflow includes inspection and staged execution.

The benchmark, device conversion, and tomography stages are scripts because they are deterministic batch analyses with clear command-line inputs/outputs.

## Why the experimental spectra appear twice

`data/experimental/digitized_spectra.zip` is the exact archived reconstruction bundle and is retained for byte-level provenance.

`data/experimental/spectra/` exposes the same 242 real/imaginary CSV traces individually so reviewers can inspect them directly in GitHub without extracting the archive.

## Why only ten SpinQ waveforms are canonical

The preparation workflow generates nine RF offsets per model, but the reported hardware acquisition uses five offsets per model:

[
-10,-5,0,+5,+10%.
]

Those ten actual acquired-condition waveforms are retained as the canonical device dataset. The exporter can regenerate the intermediate ±2.5% and ±7.5% preparation points when needed.

## Benchmark parallelism

The two-GPU benchmark does not use DDP or synchronized gradients. It launches two independent Python workers, each with one visible GPU and a disjoint subset of target gates. Rank 0 subsequently merges the saved gate records and runs the common post-processing analyses.

## Release integrity

`MANIFEST_SHA256.csv` contains SHA-256 hashes for every tracked release file except itself. Run

```bash
python scripts/verify_release.py
```

to verify the snapshot.
