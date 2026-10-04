# Digitized NMR tomography spectra

`digitized_spectra.zip` contains the experimental spectra used for the PPS reference and the Hadamard RF-gain hardware validation.

The archive contains the real and imaginary spectral traces required by the linear tomography reconstruction. Redundant pre-combined CSV files from the laboratory working archive are intentionally omitted because the reconstruction never uses them.

## Experimental conditions

The eleven condition folders are:

- `PPS`: pseudo-pure-state preparation reference;
- `NM10`, `NM5`, `N0`, `NP5`, `NP10`: nominal-compiler Hadamard pulses at -10%, -5%, 0%, +5%, +10% RF gain;
- `RM10`, `RM5`, `R0`, `RP5`, `RP10`: risk-aware-compiler Hadamard pulses at the same RF-gain settings.

The ten Hadamard conditions map directly to the exact device waveforms in `data/hardware/spinq/paper_acquired_conditions.csv`.

## Tomography readouts

Each condition contains real (`,R.csv`) and imaginary (`,I.csv`) traces for eleven readout rotations:

`III, XII, IXI, IIX, IXX, XXX, YII, IYI, IIY, YYI, YYY`.

The reconstruction in `scripts/06_reconstruct_tomography.py` integrates twelve fixed single-quantum transition windows from each complex spectrum, applies the fixed receiver-phase corrections used in the experiment, and solves a 264-by-63 linear least-squares system for an 8x8 Hermitian traceless deviation matrix.

The normalized Hilbert-Schmidt correlation is

[
C_{HS}(D_1,D_2)=
rac{operatorname{Re}operatorname{Tr}(D_1D_2)}
{sqrt{operatorname{Tr}(D_1^2)operatorname{Tr}(D_2^2)}}.
]

These are deviation-matrix correlations, not unit-trace density-matrix fidelities.

## Direct reproduction

From the repository root:

```bash
python scripts/06_reconstruct_tomography.py
```

No manual extraction is required; the script reads the ZIP directly. The generated central table reproduces `results/tomography/experimental_hs_correlations.csv`, including the PPS reference (C_{HS}=0.8277429528354344).
