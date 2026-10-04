# Reproducibility configurations

These JSON files collect the frozen settings used by the publication
workflows. The executable notebooks/scripts remain the authoritative
implementation; the configs provide a compact, human-readable record.

- `nominal.json`: full-Haar nominal neural-compiler training.
- `stress_test.json`: frozen nominal robustness sweeps and joint uncertainty
  profiles.
- `robust_training.json`: direct ±5% risk-aware fine-tuning. Wider RF ranges
  are evaluation-only.
- `benchmark.json`: target set and classical/neural optimizer benchmark
  settings.
- `hardware_experiment.json`: Hadamard RF-sweep preparation and the recorded
  SpinQ device-export convention.

No legacy staged-training labels are used in the public robust-training
configuration because only the reported ±5% training run is part of the
publication workflow.
