# Reproducing the paper

The repository follows the order of the experiments in the manuscript.

1. **Nominal compiler** — `notebooks/01_train_nominal.ipynb`
2. **Frozen stress diagnosis** — `notebooks/02_stress_test_nominal.ipynb`
3. **Risk-aware fine-tuning** — `notebooks/03_train_robust.ipynb`
4. **Optimal-control benchmark** — `scripts/04_benchmark_optimizers.py`
5. **Hadamard hardware preparation** — `notebooks/05_prepare_hardware.ipynb`
6. **Tomography reconstruction and uncertainty** — `scripts/06_*` through `scripts/09_*`

## Experimental spectra

`data/experimental/digitized_spectra.zip` contains the digitized complex spectra used by the tomography analysis. Extract it before running the tomography scripts:

```bash
unzip data/experimental/digitized_spectra.zip -d data/experimental
```

## Determinism

The scripts record fixed random seeds and write provenance information and SHA-256 checksums with generated result bundles. GPU matrix exponentials and floating-point reductions can still introduce small platform-dependent differences.

## Hardware requirements

Nominal training, robust fine-tuning, and the full benchmark are GPU workloads. The tomography scripts are CPU-friendly. The benchmark SLURM helper in `hpc/benchmark_two_gpu.slurm` uses two independent GPU workers and can be adapted to another scheduler by changing only the resource header.