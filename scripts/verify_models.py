#!/usr/bin/env python3
"""Load and sanity-check both released neural compiler state dictionaries."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]


class PulseCompiler(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        width = int(cfg["width"])
        hidden_layers = int(cfg["hidden_layers"])
        dropout = float(cfg["dropout"])
        n_slices = int(cfg["n_slices"])
        self.rabi_max_hz = float(cfg["rabi_max_hz"])
        self.n_slices = n_slices

        layers = []
        in_dim = 4
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(in_dim, width), nn.GELU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = width
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, 2 * n_slices)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        raw = self.head(self.backbone(q)).view(-1, self.n_slices, 2)
        radius_sq = (raw * raw).sum(dim=-1, keepdim=True)
        return self.rabi_max_hz * raw / torch.sqrt(1.0 + radius_sq)


def extract_state_dict(obj):
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    return obj


def main() -> None:
    config_path = ROOT / "configs" / "nominal.json"
    nominal_path = ROOT / "data" / "checkpoints" / "nominal_best_state_dict.pt"
    robust_path = ROOT / "data" / "checkpoints" / "robust_final_state_dict.pt"

    with config_path.open(encoding="utf-8") as handle:
        cfg = json.load(handle)

    expected = {
        "n_slices": 300,
        "width": 256,
        "hidden_layers": 10,
        "dropout": 0.25,
        "rabi_max_hz": 50_000.0,
    }
    for key, value in expected.items():
        if cfg[key] != value:
            raise RuntimeError(
                f"Unexpected nominal config {key}: {cfg[key]} != {value}"
            )

    models = {}
    states = {}
    for label, path in [("nominal", nominal_path), ("robust", robust_path)]:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        state = extract_state_dict(obj)
        model = PulseCompiler(cfg)
        model.load_state_dict(state, strict=True)
        model.eval()
        models[label] = model
        states[label] = state

    parameter_count = sum(
        parameter.numel()
        for parameter in models["nominal"].parameters()
    )
    if parameter_count != 747_608:
        raise RuntimeError(
            f"Unexpected parameter count: {parameter_count}"
        )

    if list(states["nominal"].keys()) != list(states["robust"].keys()):
        raise RuntimeError(
            "Nominal and robust state dictionaries have different parameter keys"
        )
    for key in states["nominal"]:
        if states["nominal"][key].shape != states["robust"][key].shape:
            raise RuntimeError(f"Shape mismatch for {key}")

    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    with torch.inference_mode():
        for label, model in models.items():
            controls = model(q)
            if controls.shape != (1, 300, 2):
                raise RuntimeError(
                    f"{label}: unexpected output shape {tuple(controls.shape)}"
                )
            peak = torch.linalg.vector_norm(
                controls,
                dim=-1,
            ).max().item()
            if not peak < float(cfg["rabi_max_hz"]):
                raise RuntimeError(
                    f"{label}: radial amplitude bound violated: {peak}"
                )
            if not torch.isfinite(controls).all():
                raise RuntimeError(
                    f"{label}: non-finite control value"
                )

    print("MODEL VERIFICATION PASSED")
    print(f"Parameters: {parameter_count:,}")
    print("Nominal state dict: strict load OK")
    print("Robust state dict: strict load OK")
    print(
        "Forward output: (1, 300, 2), finite, radial bound satisfied"
    )


if __name__ == "__main__":
    main()
