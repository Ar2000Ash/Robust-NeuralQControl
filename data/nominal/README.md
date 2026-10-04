# Frozen nominal validation data

These arrays are copied directly from the original nominal-training publication export.

- <code>validation_quaternions.npy</code>: 512 held-out Haar-random SU(2) target quaternions, shape <code>(512, 4)</code>, float32.
- <code>validation_fidelities.npy</code>: corresponding frozen nominal unitary fidelities, shape <code>(512,)</code>, float32.

Together with <code>results/nominal/final_validation_summary.json</code>, these arrays make the reported nominal validation distribution independently inspectable without retraining the neural compiler.

The frozen nominal model itself is:

~~~text
data/checkpoints/nominal_best_state_dict.pt
~~~
