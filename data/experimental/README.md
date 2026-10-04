# Digitized NMR tomography spectra

This directory contains the experimental spectra used for the pseudo-pure-state (PPS) reference and the Hadamard RF-gain hardware validation.

## Released reconstruction data

The reconstruction inputs are available in two forms:

- <code>digitized_spectra.zip</code>: canonical R/I-only reconstruction-input bundle used by the released tomography scripts;
- <code>spectra/</code>: the same 242 real/imaginary CSV traces unpacked for direct inspection on GitHub.

<code>spectra_manifest.csv</code> enumerates every released reconstruction trace by experimental condition, readout rotation, and quadrature.

The reconstruction dataset contains 11 conditions × 11 readouts × 2 quadratures = **242 CSV files**.

## Original combined display exports

<code>combined_display_exports.zip</code> contains the **121 original combined.csv laboratory display/export traces** (11 conditions × 11 readouts).

These files are preserved exactly for archival completeness. They are denser pre-combined exports and are **not** inputs to the tomography or uncertainty scripts. They were not regenerated from the rounded R/I CSV files.

Archive provenance:

~~~text
SHA-256: 20e52816dcaf11c2e694b5841407bee19a992a70809cf03a3d57aa4418156bd4
Git blob: 3d12bf1e6709b16ad7160dc2692587fe224ac12a
Files:    121 combined.csv exports
~~~

## Experimental conditions

The eleven condition folders are:

- <code>PPS</code>: pseudo-pure-state preparation reference;
- <code>NM10</code>, <code>NM5</code>, <code>N0</code>, <code>NP5</code>, <code>NP10</code>: nominal-compiler Hadamard pulses at −10%, −5%, 0%, +5%, +10% RF gain;
- <code>RM10</code>, <code>RM5</code>, <code>R0</code>, <code>RP5</code>, <code>RP10</code>: risk-aware-compiler Hadamard pulses at the same RF-gain settings.

The ten Hadamard conditions map directly to the exact device waveforms listed in <code>data/hardware/spinq/paper_acquired_conditions.csv</code>.

## Tomography readouts

Each condition contains real (<code>,R.csv</code>) and imaginary (<code>,I.csv</code>) traces for eleven readout rotations:

<code>III, XII, IXI, IIX, IXX, XXX, YII, IYI, IIY, YYI, YYY</code>.

The reconstruction in <code>scripts/06_reconstruct_tomography.py</code> integrates twelve fixed single-quantum transition windows from each complex spectrum, applies the frozen receiver-phase corrections used in the experiment, and solves a 264-by-63 linear least-squares system for an 8×8 Hermitian traceless deviation matrix.

The normalized Hilbert–Schmidt correlation is

$$
C_{\mathrm{HS}}(D_1,D_2)
=
\frac{\operatorname{Re}\operatorname{Tr}(D_1D_2)}
{\sqrt{\operatorname{Tr}(D_1^2)\operatorname{Tr}(D_2^2)}}.
$$

These are deviation-matrix correlations, not unit-trace density-matrix fidelities.

## Direct reproduction

From the repository root:

~~~bash
python scripts/06_reconstruct_tomography.py
~~~

No manual extraction is required; the script reads the canonical R/I ZIP directly. The generated central table reproduces <code>results/tomography/experimental_hs_correlations.csv</code>, including

~~~text
C_HS(PPS) = 0.8277429528354344
~~~
