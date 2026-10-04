#!/usr/bin/env python3
"""Verify the frozen publication release without external dependencies."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST_SHA256.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest() -> int:
    if not MANIFEST.exists():
        raise FileNotFoundError(
            "MANIFEST_SHA256.csv is missing. Use the frozen release snapshot."
        )

    seen: set[str] = set()
    checked = 0

    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["sha256", "path"]:
            raise ValueError(
                "Manifest columns must be exactly: sha256,path"
            )

        for row in reader:
            rel = row["path"]
            expected = row["sha256"].lower()

            if rel in seen:
                raise ValueError(f"Duplicate manifest path: {rel}")
            seen.add(rel)

            path = ROOT / rel
            if not path.is_file():
                raise FileNotFoundError(f"Manifest file missing: {rel}")

            actual = sha256_file(path)
            if actual != expected:
                raise RuntimeError(
                    f"SHA-256 mismatch for {rel}\n"
                    f"expected: {expected}\n"
                    f"actual:   {actual}"
                )
            checked += 1

    return checked


def verify_configs() -> None:
    config_dir = ROOT / "configs"
    required = {
        "nominal.json",
        "stress_test.json",
        "robust_training.json",
        "benchmark.json",
        "hardware_experiment.json",
    }
    present = {path.name for path in config_dir.glob("*.json")}
    missing = required - present
    if missing:
        raise FileNotFoundError(
            f"Missing publication config(s): {sorted(missing)}"
        )

    for path in sorted(config_dir.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            json.load(handle)


def verify_models() -> None:
    required = [
        ROOT / "data" / "checkpoints" / "nominal_frozen_checkpoint.pt",
        ROOT / "data" / "checkpoints" / "robust_final_state_dict.pt",
    ]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing trained model: {path}")


def verify_spectra() -> None:
    root = ROOT / "data" / "experimental" / "spectra"
    files = sorted(root.rglob("*.csv"))
    real = [path for path in files if path.name.endswith(",R.csv")]
    imag = [path for path in files if path.name.endswith(",I.csv")]
    combined = [
        path for path in files if path.name.endswith(",combined.csv")
    ]

    if len(files) != 242:
        raise RuntimeError(
            f"Expected 242 released spectrum CSVs, found {len(files)}"
        )
    if len(real) != 121 or len(imag) != 121:
        raise RuntimeError(
            f"Expected 121 R and 121 I spectra; "
            f"found R={len(real)}, I={len(imag)}"
        )
    if combined:
        raise RuntimeError(
            "Canonical reconstruction directory unexpectedly contains "
            "combined.csv files"
        )


def verify_device_pulses() -> None:
    root = ROOT / "data" / "hardware" / "spinq" / "device_pulses"
    pulses = sorted(root.rglob("*.spinq"))

    if len(pulses) != 10:
        raise RuntimeError(
            f"Expected 10 canonical device pulses, found {len(pulses)}"
        )

    for path in pulses:
        with path.open(encoding="utf-8") as handle:
            rows = [line for line in handle if line.strip()]
        if len(rows) != 300:
            raise RuntimeError(
                f"{path.relative_to(ROOT)} has {len(rows)} rows; expected 300"
            )


def main() -> None:
    count = verify_manifest()
    verify_configs()
    verify_models()
    verify_spectra()
    verify_device_pulses()

    print("RELEASE VERIFICATION PASSED")
    print(f"SHA-256 verified files: {count}")
    print("Configs: valid")
    print("Trained models: present")
    print("Digitized spectra: 242 CSV files (121 R + 121 I)")
    print("Canonical SpinQ waveforms: 10 files × 300 rows")


if __name__ == "__main__":
    main()
