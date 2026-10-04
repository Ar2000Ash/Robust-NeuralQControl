# SpinQ device-ready Hadamard pulses

This directory contains the device-facing waveforms for the Hadamard hardware experiment reported in the manuscript.

## Conversion from the neural-control representation

<code>notebooks/05_prepare_hardware.ipynb</code> produces 300-slice Hadamard controls in the model's Hz-equivalent I/Q convention. <code>scripts/05_export_spinq_waveforms.py</code> converts those controls into the three-column SpinQ format recorded with the experiment.

For each slice,

$$
A_{\mathrm{SpinQ}}
=
100\,
\frac{\sqrt{u_x^2+u_y^2}}
{8333.333333333334\ {\rm Hz}},
$$

$$
\phi_{\mathrm{SpinQ}}
=
\operatorname{atan2}(u_y,u_x)
\frac{180}{\pi}
\pmod{360^\circ},
$$

and the third column is the fixed dwell time of 35 microseconds.

This is the working Ankara/SpinQ convention preserved in the original device-ready archive. It is separate from the neural model's 50 kHz numerical control bound; that training bound is not treated as the hardware full-scale calibration.

## Device file format

Every <code>.spinq</code> file has exactly 300 rows and three comma-separated numeric columns:

1. SpinQ amplitude;
2. phase in degrees, wrapped to \([0,360)\);
3. dwell time in microseconds (35 for every row).

The exporter was validated against the archived device-ready package and reproduces all 18 originally prepared waveforms byte-for-byte.

## Files retained for the paper experiment

The historical preparation package contained nine RF-gain settings per model (−10%, −7.5%, −5%, −2.5%, 0%, +2.5%, +5%, +7.5%, +10%). The digitized tomography dataset and the manuscript hardware experiment contain the five settings −10%, −5%, 0%, +5%, +10% for both nominal and risk-aware pulses. Those ten device files are retained here as the canonical experimental waveforms.

The requested RF scaling is **already baked into each waveform**. It must not be applied a second time on the instrument unless an additional perturbation is deliberately intended.

<code>paper_acquired_conditions.csv</code> maps each of the ten device waveforms to the digitized-spectrum condition labels used by the tomography pipeline: <code>NM10</code>, <code>NM5</code>, <code>N0</code>, <code>NP5</code>, <code>NP10</code>, <code>RM10</code>, <code>RM5</code>, <code>R0</code>, <code>RP5</code>, and <code>RP10</code>. The PPS dataset is a separate state-preparation reference and therefore has no corresponding Hadamard device waveform.

## Supporting files

- <code>device_pulses/nominal_nn/</code>: five nominal-compiler waveforms used in the reported hardware experiment.
- <code>device_pulses/robust_nn/</code>: five risk-aware-compiler waveforms used in the reported hardware experiment.
- <code>device_pulse_manifest.csv</code>: source mapping and amplitude/phase statistics for those ten files.
- <code>hardware_run_sheet.csv</code>: the ten hardware conditions corresponding to the reported experiment.
- <code>paper_acquired_conditions.csv</code>: device-file to digitized-spectrum mapping.
- <code>amplitude_conversion_reference.csv</code>: reference points for the Hz-to-device-amplitude scale.
- <code>SHA256SUMS.txt</code>: checksums for this curated device package.

The large nested source copy from the historical archive is intentionally not duplicated. Its scientific inputs are represented elsewhere in the repository by the cleaned preparation notebook, frozen checkpoints, and reproducibility data.
