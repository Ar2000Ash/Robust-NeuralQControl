#!/usr/bin/env python3
"""Reconstruct three-spin deviation density matrices from digitized NMR spectra.

This is the publication reconstruction used for the PPS reference and the
nominal/risk-aware Hadamard RF-gain hardware experiment. The released data are
stored as a ZIP archive containing real and imaginary digitized spectra for 11
readout rotations and 11 experimental conditions.

The reconstruction is linear and deterministic:

* 11 readout rotations (III, XII, IXI, IIX, IXX, XXX, YII, IYI, IIY, YYI, YYY)
* 12 observed single-quantum transitions per readout
* real and imaginary components, giving 264 real measurements
* a 63-dimensional Hermitian traceless basis for an 8x8 deviation matrix
* fixed integration windows, receiver phases, and the original global scale

The script reads the ZIP directly; no manual extraction is required. An
already-extracted data directory may also be supplied with --data.
"""

from __future__ import annotations

import argparse
import io
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from numpy import kron
from scipy.linalg import expm

DIM = 8
GLOBAL_SIGNAL_SCALE = 3800.0

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)


def kron3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return kron(kron(a, b), c)


# 90-degree readout rotations used in the tomography acquisition.
UX = expm(-1j * X * np.pi / 4)
UY = expm(-1j * Y * np.pi / 4)

READOUT_LABELS = [
    "III", "XII", "IXI", "IIX", "IXX", "XXX",
    "YII", "IYI", "IIY", "YYI", "YYY",
]

READOUT_UNITARIES = [
    np.eye(8),
    kron3(UX, I2, I2),
    kron3(I2, UX, I2),
    kron3(I2, I2, UX),
    kron3(I2, UX, UX),
    kron3(UX, UX, UX),
    kron3(UY, I2, I2),
    kron3(I2, UY, I2),
    kron3(I2, I2, UY),
    kron3(UY, UY, I2),
    kron3(UY, UY, UY),
]

# Matrix elements corresponding to the 12 single-quantum transitions.
TRANSITION_INDICES = [
    (4, 0), (6, 2), (5, 1), (7, 3),
    (2, 0), (6, 4), (3, 1), (7, 5),
    (1, 0), (3, 2), (5, 4), (7, 6),
]

# Fixed frequency windows used in the original experimental analysis.
FREQUENCY_WINDOWS_HZ = [
    (-1001, -930), (-866, -765), (-1072, -1001), (-930, -866),
    (-45, 40), (100, 200), (-153, -45), (40, 100),
    (721, 840), (647, 684), (684, 721), (571, 647),
]

# Receiver-phase corrections for the three spin groups (radians).
RECEIVER_PHASES_RAD = np.array([
    4.593571789275965,
    4.729046676847299,
    3.8289005158044778,
])

CONDITIONS = [
    (-10, "NM10", "RM10"),
    (-5, "NM5", "RM5"),
    (0, "N0", "R0"),
    (5, "NP5", "RP5"),
    (10, "NP10", "RP10"),
]

# State-transfer simulation values preserved from the original hardware
# analysis. These are not the gate-level unitary fidelities reported in the
# optimizer benchmark.
SIMULATED_NOMINAL_STATE_FIDELITY = {
    -10: 0.47740,
    -5: 0.82272,
    0: 0.99130,
    5: 0.82560,
    10: 0.46757,
}
SIMULATED_ROBUST_STATE_FIDELITY = {
    -10: 0.95995,
    -5: 0.98524,
    0: 0.98171,
    5: 0.97507,
    10: 0.95644,
}


class SpectrumSource(Protocol):
    def load(self, condition: str, filename: str) -> tuple[np.ndarray, np.ndarray]:
        ...


@dataclass
class DirectorySource:
    root: Path

    def load(self, condition: str, filename: str) -> tuple[np.ndarray, np.ndarray]:
        return load_two_column_csv(self.root / condition / filename)


class ZipSource:
    def __init__(self, path: Path):
        self.path = path
        self.archive = zipfile.ZipFile(path)
        self.member_map = self._index_members()

    def _index_members(self) -> dict[tuple[str, str], str]:
        mapping: dict[tuple[str, str], str] = {}
        required_conditions = {"PPS"}
        required_conditions.update(n for _, n, _ in CONDITIONS)
        required_conditions.update(r for _, _, r in CONDITIONS)

        for member in self.archive.namelist():
            parts = Path(member).parts
            if len(parts) < 2:
                continue
            condition = parts[-2]
            filename = parts[-1]
            if condition in required_conditions and (
                filename.endswith(",R.csv") or filename.endswith(",I.csv")
            ):
                mapping[(condition, filename)] = member
        return mapping

    def load(self, condition: str, filename: str) -> tuple[np.ndarray, np.ndarray]:
        try:
            member = self.member_map[(condition, filename)]
        except KeyError as exc:
            raise FileNotFoundError(
                f"Missing {condition}/{filename} in {self.path}"
            ) from exc
        text = self.archive.read(member).decode("utf-8-sig")
        return load_two_column_text(text)

    def close(self) -> None:
        self.archive.close()


def hermitian_traceless_basis() -> list[np.ndarray]:
    basis: list[np.ndarray] = []

    # Seven independent diagonal traceless matrices.
    for i in range(DIM - 1):
        matrix = np.zeros((DIM, DIM), dtype=complex)
        matrix[i, i] = 1.0
        matrix[DIM - 1, DIM - 1] = -1.0
        basis.append(matrix)

    # 28 real-symmetric + 28 imaginary-antisymmetric matrices.
    for i in range(DIM):
        for j in range(i + 1, DIM):
            matrix = np.zeros((DIM, DIM), dtype=complex)
            matrix[i, j] = 1.0
            matrix[j, i] = 1.0
            basis.append(matrix)

            matrix = np.zeros((DIM, DIM), dtype=complex)
            matrix[i, j] = -1j
            matrix[j, i] = 1j
            basis.append(matrix)

    assert len(basis) == 63
    return basis


BASIS = hermitian_traceless_basis()


def build_measurement_matrix() -> np.ndarray:
    rows: list[list[float]] = []
    for unitary in READOUT_UNITARIES:
        for i, j in TRANSITION_INDICES:
            row_real: list[float] = []
            row_imag: list[float] = []
            for basis_matrix in BASIS:
                rotated = unitary @ basis_matrix @ unitary.conj().T
                value = rotated[i, j]
                row_real.append(float(value.real))
                row_imag.append(float(value.imag))
            rows.extend([row_real, row_imag])

    matrix = np.asarray(rows, dtype=float)
    assert matrix.shape == (264, 63)
    return matrix


MEASUREMENT_MATRIX = build_measurement_matrix()


def load_two_column_text(text: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(io.StringIO(text), delimiter=",", skiprows=1)
    data = data[np.argsort(data[:, 0])]
    return data[:, 0], data[:, 1]


def load_two_column_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    data = data[np.argsort(data[:, 0])]
    return data[:, 0], data[:, 1]


def integrate_window(
    window: tuple[float, float],
    frequency_hz: np.ndarray,
    spectrum: np.ndarray,
) -> float:
    dense_frequency = np.linspace(window[0], window[1], 1000)
    interpolated = np.interp(dense_frequency, frequency_hz, spectrum)
    return float(
        np.trapezoid(interpolated, dense_frequency)
        / (window[1] - window[0])
    )


def reconstruct_deviation_matrix(
    source: SpectrumSource,
    condition: str,
) -> np.ndarray:
    measurements: list[float] = []

    for label in READOUT_LABELS:
        real_f, real_s = source.load(condition, f"{label},R.csv")
        imag_f, imag_s = source.load(condition, f"{label},I.csv")

        peaks = np.asarray([
            integrate_window(window, real_f, real_s)
            + 1j * integrate_window(window, imag_f, imag_s)
            for window in FREQUENCY_WINDOWS_HZ
        ])

        peaks[0:4] *= np.exp(-1j * RECEIVER_PHASES_RAD[0])
        peaks[4:8] *= np.exp(-1j * RECEIVER_PHASES_RAD[1])
        peaks[8:12] *= np.exp(-1j * RECEIVER_PHASES_RAD[2])

        for value in peaks:
            measurements.extend([float(value.real), float(value.imag)])

    vector = np.asarray(measurements, dtype=float) / GLOBAL_SIGNAL_SCALE
    if vector.shape != (264,):
        raise RuntimeError(f"Unexpected tomography vector shape: {vector.shape}")

    coefficients, *_ = np.linalg.lstsq(
        MEASUREMENT_MATRIX,
        vector,
        rcond=None,
    )

    deviation = np.zeros((DIM, DIM), dtype=complex)
    for coefficient, basis_matrix in zip(coefficients, BASIS):
        deviation += coefficient * basis_matrix
    return deviation


def hs_correlation(d1: np.ndarray, d2: np.ndarray) -> float:
    numerator = float(np.real(np.trace(d1 @ d2)))
    denominator = math.sqrt(
        float(np.real(np.trace(d1 @ d1)))
        * float(np.real(np.trace(d2 @ d2)))
    )
    return numerator / denominator


def ideal_deviation_matrices() -> tuple[np.ndarray, np.ndarray]:
    psi0 = np.zeros((8, 1), dtype=complex)
    psi0[0, 0] = 1.0
    d_pps = psi0 @ psi0.conj().T - np.eye(8) / 8

    hadamard = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    psi_h = kron3(hadamard, I2, I2) @ psi0
    d_h = psi_h @ psi_h.conj().T - np.eye(8) / 8
    return d_pps, d_h


def state_fidelity_to_deviation_hs(fidelity: float) -> float:
    # For pure simulated output and pure target states in d=8.
    return (8.0 * fidelity - 1.0) / 7.0


def validate_source(source: SpectrumSource) -> None:
    conditions = {"PPS"}
    conditions.update(n for _, n, _ in CONDITIONS)
    conditions.update(r for _, _, r in CONDITIONS)

    for condition in sorted(conditions):
        for label in READOUT_LABELS:
            source.load(condition, f"{label},R.csv")
            source.load(condition, f"{label},I.csv")


def reconstruct_all(
    source: SpectrumSource,
) -> tuple[pd.DataFrame, float, dict[str, np.ndarray]]:
    d_pps_ideal, d_h_ideal = ideal_deviation_matrices()

    matrices: dict[str, np.ndarray] = {}
    matrices["PPS"] = reconstruct_deviation_matrix(source, "PPS")
    c_pps = hs_correlation(matrices["PPS"], d_pps_ideal)

    rows: list[dict[str, float | int]] = []
    for rf_error, nominal_condition, robust_condition in CONDITIONS:
        matrices[nominal_condition] = reconstruct_deviation_matrix(
            source, nominal_condition
        )
        matrices[robust_condition] = reconstruct_deviation_matrix(
            source, robust_condition
        )

        c_nominal = hs_correlation(matrices[nominal_condition], d_h_ideal)
        c_robust = hs_correlation(matrices[robust_condition], d_h_ideal)

        rows.append({
            "epsilon_RF_percent": rf_error,
            "sim_nominal_C_HS": state_fidelity_to_deviation_hs(
                SIMULATED_NOMINAL_STATE_FIDELITY[rf_error]
            ),
            "exp_nominal_C_H": c_nominal,
            "exp_nominal_R_H=C_H/C_PPS": c_nominal / c_pps,
            "sim_robust_C_HS": state_fidelity_to_deviation_hs(
                SIMULATED_ROBUST_STATE_FIDELITY[rf_error]
            ),
            "exp_robust_C_H": c_robust,
            "exp_robust_R_H=C_H/C_PPS": c_robust / c_pps,
            "robust_minus_nominal_C_H": c_robust - c_nominal,
        })

    return pd.DataFrame(rows), c_pps, matrices


def open_source(path: Path) -> SpectrumSource:
    if path.is_dir():
        # Accept either data/experimental/digitized_csv or its parent.
        nested = path / "digitized_csv"
        return DirectorySource(nested if nested.is_dir() else path)
    if path.suffix.lower() == ".zip":
        return ZipSource(path)
    raise ValueError(f"--data must be a directory or ZIP archive: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/experimental/digitized_spectra.zip"),
        help="Digitized spectra ZIP or extracted digitized_csv directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/tomography/experimental_hs_correlations.csv"),
        help="Destination for the central reconstruction table",
    )
    parser.add_argument(
        "--pps-output",
        type=Path,
        default=Path("results/tomography/pps_reference.csv"),
        help="Destination for the PPS reference correlation",
    )
    parser.add_argument(
        "--matrix-output",
        type=Path,
        default=Path("outputs/tomography/reconstructed_deviation_matrices.npz"),
        help="Generated reconstructed-matrix bundle",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = open_source(args.data.resolve())

    try:
        validate_source(source)
        table, c_pps, matrices = reconstruct_all(source)
    finally:
        if isinstance(source, ZipSource):
            source.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.pps_output.parent.mkdir(parents=True, exist_ok=True)
    args.matrix_output.parent.mkdir(parents=True, exist_ok=True)

    table.to_csv(args.output, index=False)
    pd.DataFrame([{"condition": "PPS", "C_HS": c_pps}]).to_csv(
        args.pps_output,
        index=False,
    )
    np.savez(args.matrix_output, **matrices)

    print(f"Measurement matrix: {MEASUREMENT_MATRIX.shape}")
    print(f"C_PPS = {c_pps:.15f}")
    print(
        table.to_string(
            index=False,
            float_format=lambda value: f"{value:.9f}",
        )
    )
    print(f"\nCentral table: {args.output}")
    print(f"PPS reference: {args.pps_output}")
    print(f"Matrices: {args.matrix_output}")


if __name__ == "__main__":
    main()
