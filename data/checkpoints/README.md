# Frozen neural-compiler checkpoints

These are the canonical model files used by the downstream stress-test,
benchmark, and hardware workflows.

## nominal_frozen_checkpoint.pt

Frozen nominal compiler selected from the full-Haar nominal training run.

- selected training step: 48,600;
- contains the model weights together with the nominal run configuration and
  stored validation metadata;
- exact historical Git blob:
  `646a8f43955b24a716285f08c8b809febf882d0a`;
- size: 2,999,065 bytes.

The stress-test and robust-training notebooks read this file directly.

## robust_final_state_dict.pt

Frozen risk-aware compiler obtained by fine-tuning the nominal compiler under
the ±5% RF training envelope.

- selected robust validation step: 1,500;
- stored as the final model `state_dict`;
- exact historical Git blob:
  `f1dc4f44f80cb0a96e6c07a06ecfdc6aff257ddb`;
- size: 2,999,329 bytes.

Training used only the ±5% RF envelope. The ±7.5%, ±10%, and ±15% RF ranges
are evaluation-only robustness tests.

The repository intentionally keeps one canonical copy of each frozen model.
Full training artifacts and optimizer-state exports are reproducible from the
training notebooks but are not duplicated here.
