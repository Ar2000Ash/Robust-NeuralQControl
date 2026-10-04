#!/usr/bin/env python3
"""Regenerate Hadamard hardware-validation figures from released data."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASIS_LABELS = [format(index, "03b") for index in range(8)]


def load_reconstruction_module():
    path = ROOT / "scripts" / "06_reconstruct_tomography.py"
    spec = importlib.util.spec_from_file_location(
        "tomography_reconstruction",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_figure(fig, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(
        output_dir / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_hardware_summary(
    table: pd.DataFrame,
    output_dir: Path,
) -> None:
    rf_error = table["epsilon_RF_percent"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.plot(
        rf_error,
        table["sim_nominal_C_HS"],
        marker="o",
        label="simulation, nominal",
    )
    ax.plot(
        rf_error,
        table["sim_robust_C_HS"],
        marker="o",
        label="simulation, robust",
    )
    ax.plot(
        rf_error,
        table["exp_nominal_C_H"],
        marker="s",
        linestyle="--",
        label="experiment, nominal",
    )
    ax.plot(
        rf_error,
        table["exp_robust_C_H"],
        marker="s",
        linestyle="--",
        label="experiment, robust",
    )

    ax.set_xlabel("Commanded RF-amplitude error (%)")
    ax.set_ylabel("Normalized Hilbert–Schmidt correlation")
    ax.set_xticks(rf_error)
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    save_figure(
        fig,
        output_dir,
        "hardware_simulation_vs_experiment",
    )


def plot_complex_spectrum(
    condition: str,
    output_dir: Path,
) -> None:
    root = (
        ROOT
        / "data"
        / "experimental"
        / "spectra"
        / condition
    )
    real = np.loadtxt(
        root / "III,R.csv",
        delimiter=",",
        skiprows=1,
    )
    imag = np.loadtxt(
        root / "III,I.csv",
        delimiter=",",
        skiprows=1,
    )

    real = real[np.argsort(real[:, 0])]
    imag = imag[np.argsort(imag[:, 0])]

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(real[:, 0], real[:, 1], label="real")
    ax.plot(
        imag[:, 0],
        imag[:, 1],
        label="imaginary",
        alpha=0.8,
    )
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Digitized amplitude (mV)")
    ax.set_title(f"{condition}: III readout")
    ax.grid(alpha=0.15)
    ax.legend(frameon=False)

    save_figure(fig, output_dir, f"III{condition}")


def plot_density_real(
    matrix: np.ndarray,
    condition: str,
    output_dir: Path,
) -> None:
    values = np.asarray(matrix.real, dtype=float)
    x, y = np.meshgrid(
        np.arange(8),
        np.arange(8),
        indexing="ij",
    )

    xpos = x.ravel()
    ypos = y.ravel()
    zpos = np.zeros(64)
    dx = np.full(64, 0.72)
    dy = np.full(64, 0.72)
    dz = values.ravel()

    fig = plt.figure(figsize=(7.4, 5.8))
    ax = fig.add_subplot(111, projection="3d")
    ax.bar3d(
        xpos,
        ypos,
        zpos,
        dx,
        dy,
        dz,
        shade=True,
    )
    ax.set_xticks(
        np.arange(8) + 0.36,
        BASIS_LABELS,
        rotation=45,
        ha="right",
    )
    ax.set_yticks(
        np.arange(8) + 0.36,
        BASIS_LABELS,
        rotation=-30,
        va="center",
    )
    ax.set_xlabel("ket basis")
    ax.set_ylabel("bra basis")
    ax.set_zlabel("Re(deviation matrix)")
    ax.set_title(
        f"{condition}: reconstructed real deviation matrix"
    )

    save_figure(
        fig,
        output_dir,
        f"density_{condition}_real",
    )


def reconstruct_matrices(
    data_path: Path,
) -> dict[str, np.ndarray]:
    reconstruction = load_reconstruction_module()
    source = reconstruction.open_source(data_path.resolve())

    try:
        reconstruction.validate_source(source)
        _, _, matrices = reconstruction.reconstruct_all(source)
    finally:
        if isinstance(source, reconstruction.ZipSource):
            source.close()

    return matrices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=(
            ROOT
            / "data"
            / "experimental"
            / "digitized_spectra.zip"
        ),
        help="R/I digitized-spectrum ZIP used by central tomography",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=(
            ROOT
            / "results"
            / "tomography"
            / "experimental_hs_correlations.csv"
        ),
        help="Central hardware correlation table",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "hardware_figures",
        help="Destination for regenerated PDF/PNG figures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = pd.read_csv(args.table)
    matrices = reconstruct_matrices(args.data)

    plot_hardware_summary(table, args.output_dir)

    for condition in ["NP10", "RP10"]:
        plot_complex_spectrum(condition, args.output_dir)
        plot_density_real(
            matrices[condition],
            condition,
            args.output_dir,
        )

    print("HARDWARE FIGURE REGENERATION PASSED")
    print("Output directory:", args.output_dir)
    print("Generated stems:")
    for stem in [
        "hardware_simulation_vs_experiment",
        "IIINP10",
        "IIIRP10",
        "density_NP10_real",
        "density_RP10_real",
    ]:
        print(" -", stem)


if __name__ == "__main__":
    main()
