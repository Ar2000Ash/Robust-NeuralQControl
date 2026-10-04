# Digitized NMR tomography spectra

This directory contains the experimental spectra used for the pseudo-pure-state
(PPS) reference and the Hadamard RF-gain hardware validation.

## Released reconstruction data

The same reconstruction inputs are available in two forms:

- `digitized_spectra.zip`: the canonical R/I-only reconstruction-input bundle used by the released tomography scripts;
- `spectra/`: the same 242 real/imaginary CSV traces unpacked so they can be
  inspected directly on GitHub.

`spectra_manifest.csv` enumerates every released trace by experimental
condition, readout rotation, and quadrature.

The reconstruction dataset contains 11 conditions × 11 readouts × 2
quadratures = **242 CSV files**. No numerical transformation is applied when
the archived traces are exposed under `spectra/`.

The original laboratory working archive contains 363 digitized CSV exports in total: the 242 released R/I reconstruction inputs plus 121 `combined.csv` display/export traces. Those pre-combined files are not inputs
to any tomography or uncertainty calculation and are therefore kept distinct
from the canonical reconstruction dataset rather than being silently
regenerated from rounded R/I exports.

## Experimental conditions

The eleven condition folders are:

- `PPS`: pseudo-pure-state preparation reference;
- `NM10`, `NM5`, `N0`, `NP5`, `NP10`: nominal-compiler Hadamard
  pulses at -10%, -5%, 0%, +5%, +10% RF gain;
- `RM10`, `RM5`, `R0`, `RP5`, `RP10`: risk-aware-compiler Hadamard
  pulses at the same RF-gain settings.

The ten Hadamard conditions map directly to the exact device waveforms listed
in `data/hardware/spinq/paper_acquired_conditions.csv`.

## Tomography readouts

Each condition contains real (`,R.csv`) and imaginary (`,I.csv`) traces for
eleven readout rotations:

`III, XII, IXI, IIX, IXX, XXX, YII, IYI, IIY, YYI, YYY`.

The reconstruction in `scripts/06_reconstruct_tomography.py` integrates
twelve fixed single-quantum transition windows from each complex spectrum,
applies the frozen receiver-phase corrections used in the experiment, and
solves a 264-by-63 linear least-squares system for an 8×8 Hermitian traceless
deviation matrix.

The normalized Hilbert–Schmidt correlation is

```text
C_HS(D1,D2) = Re Tr(D1 D2) / sqrt[Tr(D1^2) Tr(D2^2)].
```

These are deviation-matrix correlations, not unit-trace density-matrix
fidelities.

## Direct reproduction

From the repository root:

```bash
python scripts/06_reconstruct_tomography.py
```

No manual extraction is required; the script reads the ZIP directly. The
generated central table reproduces
`results/tomography/experimental_hs_correlations.csv`, including

`C_HS(PPS) = 0.8277429528354344`.
