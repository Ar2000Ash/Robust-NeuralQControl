# HPC benchmark execution

`benchmark_two_gpu.slurm` reproduces the two-GPU optimizer benchmark using two
independent Python workers. Each worker sees one GPU and processes a disjoint
subset of the 19 target gates; rank 0 then merges the saved per-gate records and
runs the common RF/B0 sweeps, joint robustness evaluation, timing analysis, and
figure/table export.

The SLURM file intentionally contains **no user, account, partition, QoS,
email, home-directory, scratch-directory, or virtual-environment identifiers**.
Cluster-specific scheduler options should be supplied at submission time if
required by the local site.

The launcher expects the repository root to contain:

- `configs/nominal.json`;
- `data/checkpoints/nominal_frozen_checkpoint.pt`;
- `data/checkpoints/robust_final_state_dict.pt`;
- `scripts/04_benchmark_optimizers.py`.

It creates a temporary benchmark input ZIP from those public files because the
validated benchmark driver uses an archive-loading interface. No private
training archive is required.

## Environment

The reported nominal/robust training environment used Python 3.13, PyTorch
2.11.0 with CUDA 12.8, NumPy 2.1.3, and pandas 2.2.3. The frozen two-GPU
benchmark export records Python 3.13, PyTorch 2.14.0 with CUDA 13.0, and NumPy
2.2.6. These differences reflect the execution environments of separate
experiments, not different physical or optimizer definitions.

`requirements.txt` and `environment.yml` specify portable dependency ranges.
Install a CUDA-enabled PyTorch build appropriate for the target system.

## Submission

Run the job from the repository root and supply any site-required scheduler
options externally. Results are written under
`outputs/benchmark_<job-id>/` unless `OUT_DIR` is overridden.

The launcher does not use distributed gradient synchronization. The two GPUs
are used only for independent target-level optimization shards.
