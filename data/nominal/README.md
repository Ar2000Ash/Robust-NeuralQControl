# Frozen nominal validation data

These arrays are copied directly from the original nominal-training publication
export.

- `validation_quaternions.npy`: 512 held-out Haar-random SU(2) target
  quaternions, shape `(512, 4)`, float32.
- `validation_fidelities.npy`: corresponding frozen nominal unitary
  fidelities, shape `(512,)`, float32.

Together with `results/nominal/final_validation_summary.json`, these arrays
make the reported nominal validation distribution independently inspectable
without retraining the neural compiler.

The frozen nominal model itself is
`data/checkpoints/nominal_frozen_checkpoint.pt`.
