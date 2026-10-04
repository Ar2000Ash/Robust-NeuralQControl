# Frozen publication results

This directory contains the compact numerical outputs used to reproduce the
paper's reported values and principal figures. Expensive simulations can be
rerun from the notebooks/scripts, but these frozen tables provide a direct
reference against the reported experiment.

## nominal/

- `final_validation_summary.json`: held-out 512-gate nominal validation
  statistics.
- The exact validation quaternions and fidelities are stored under
  `data/nominal/`.

## stress_test/

Frozen one-dimensional sensitivity sweeps for RF gain, common B0, independent
spin offsets, J-coupling offsets, phase bias, and clock error, plus the combined
sweep table.

## robust/

- `robust_training_history.csv`: validation history of the reported ±5% RF
  risk-aware fine-tuning run.
- `robust_model_comparison.csv`: nominal and robust compiler performance from
  nominal through conservative ±15% evaluation.
- `rf_sweep_nominal_vs_robust.csv`: dense RF-gain comparison used for the
  nominal-versus-robust robustness figure.

Training used only the ±5% RF envelope. Wider RF ranges in these files are
evaluation-only tests.

## benchmarks/

Frozen neural/classical optimal-control comparison tables:

- method summary;
- RF-gain sweep;
- B0 sweep;
- joint routine/OOD robustness;
- optimizer convergence histories;
- neural batching latency/throughput.

## tomography/

Central experimental reconstruction, standalone Monte-Carlo uncertainty,
systematic-sensitivity diagnostics, and the final joint/effective uncertainty
table.

Whenever an identical historical Git blob existed, it was reused directly.
Files that had only been retained inside the original experiment export ZIPs
were copied from those frozen exports without recomputing their numerical
contents.
