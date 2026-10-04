#!/usr/bin/env python3
"""Export Hadamard pulse CSV files to the SpinQ three-column pulse format.

The input CSV files are produced by ``notebooks/05_prepare_hardware.ipynb`` and
contain the already RF-scaled I/Q controls in Hz-equivalent units. This script
applies the working Ankara/SpinQ amplitude convention recorded with the hardware
experiment and writes the exact 300-row device pulse format used for acquisition.

Each output row is

    SpinQ amplitude, phase in degrees, dwell time in microseconds

The RF perturbation must not be applied again at the instrument because it is
already included in each source pulse CSV.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SPINQ_FULL_SCALE_HZ = 8333.333333333334
SPINQ_FULL_SCALE_UNITS = 100.0
EXPECTED_SLICES = 300
EXPECTED_DWELL_US = 35.0


@dataclass(frozen=True)
class PulseSummary:
    method: str
    rf_error_percent: float
    rf_scale: float
    source_csv: str
    device_file: str
    rows: int
    dt_us: float
    peak_amplitude_hz: float
    rms_amplitude_hz: float
    peak_device_amplitude: float
    min_device_amplitude: float
    phase_min_deg: float
    phase_max_deg: float


def method_label(method_dir: str) -> str:
    labels = {"nominal_nn": "Nominal NN", "robust_nn": "Robust NN"}
    try:
        return labels[method_dir]
    except KeyError as exc:
        raise ValueError(f"Unsupported method directory: {method_dir}") from exc


def device_suffix(source_stem: str) -> str:
    if source_stem == "rf_plus00p0pct":
        return "rf_zero00p0pct"
    if source_stem.startswith("rf_"):
        return source_stem
    raise ValueError(f"Unexpected pulse filename: {source_stem}")


def read_source_csv(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"ux_hz", "uy_hz", "dwell_us", "rf_error_percent", "rf_scale"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows = [
            {key: float(value) for key, value in row.items() if value is not None}
            for row in reader
        ]

    if len(rows) != EXPECTED_SLICES:
        raise ValueError(
            f"{path}: expected {EXPECTED_SLICES} slices, found {len(rows)}"
        )
    if any(
        not math.isclose(row["dwell_us"], EXPECTED_DWELL_US, abs_tol=1e-12)
        for row in rows
    ):
        raise ValueError(
            f"{path}: dwell time is not uniformly {EXPECTED_DWELL_US} us"
        )
    return rows


def convert_rows(
    rows: list[dict[str, float]]
) -> tuple[list[tuple[float, float, float]], list[float]]:
    converted: list[tuple[float, float, float]] = []
    amplitude_hz_values: list[float] = []

    for row in rows:
        ux_hz = row["ux_hz"]
        uy_hz = row["uy_hz"]
        amplitude_hz = math.hypot(ux_hz, uy_hz)
        device_amplitude = (
            SPINQ_FULL_SCALE_UNITS * amplitude_hz / SPINQ_FULL_SCALE_HZ
        )
        phase_deg = math.degrees(math.atan2(uy_hz, ux_hz)) % 360.0

        converted.append((device_amplitude, phase_deg, row["dwell_us"]))
        amplitude_hz_values.append(amplitude_hz)

    return converted, amplitude_hz_values


def write_spinq(
    path: Path, rows: Iterable[tuple[float, float, float]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="\n") as handle:
        for amplitude, phase_deg, dt_us in rows:
            handle.write(
                f"  {amplitude:.6e},  {phase_deg:.6e},  {dt_us:.6e}\n"
            )


def export_one(
    source: Path, input_root: Path, output_root: Path
) -> PulseSummary:
    method_dir = source.parent.name
    rows = read_source_csv(source)
    converted, amplitude_hz_values = convert_rows(rows)

    out_name = (
        f"hadamard_{method_dir}_{device_suffix(source.stem)}.spinq"
    )
    out_path = output_root / method_dir / out_name
    write_spinq(out_path, converted)

    device_amplitudes = [row[0] for row in converted]
    phases = [row[1] for row in converted]
    rf_error = rows[0]["rf_error_percent"]
    rf_scale = rows[0]["rf_scale"]
    rms_hz = math.sqrt(
        sum(value * value for value in amplitude_hz_values)
        / len(amplitude_hz_values)
    )

    if max(device_amplitudes) > SPINQ_FULL_SCALE_UNITS + 1e-12:
        raise ValueError(
            f"{source}: converted amplitude exceeds the 100-unit SpinQ working scale"
        )

    return PulseSummary(
        method=method_label(method_dir),
        rf_error_percent=rf_error,
        rf_scale=rf_scale,
        source_csv=str(source.relative_to(input_root.parent)),
        device_file=str(out_path.relative_to(output_root.parent)),
        rows=len(converted),
        dt_us=EXPECTED_DWELL_US,
        peak_amplitude_hz=max(amplitude_hz_values),
        rms_amplitude_hz=rms_hz,
        peak_device_amplitude=max(device_amplitudes),
        min_device_amplitude=min(device_amplitudes),
        phase_min_deg=min(phases),
        phase_max_deg=max(phases),
    )


def write_manifest(path: Path, summaries: list[PulseSummary]) -> None:
    fields = list(PulseSummary.__dataclass_fields__)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary.__dict__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("outputs/hardware_preparation/pulses"),
        help="Pulse directory generated by notebooks/05_prepare_hardware.ipynb",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/spinq_device_ready/device_pulses"),
        help="Destination for regenerated device-ready .spinq files",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/spinq_device_ready/device_pulse_manifest.csv"),
        help="Summary manifest written after conversion",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    manifest_path = args.manifest.resolve()

    sources = sorted(input_root.glob("*/*.csv"))
    if not sources:
        raise FileNotFoundError(f"No pulse CSV files found under {input_root}")

    summaries = [
        export_one(source, input_root, output_root)
        for source in sources
    ]

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(manifest_path, summaries)

    print(f"Exported {len(summaries)} device-ready pulses to {output_root}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
