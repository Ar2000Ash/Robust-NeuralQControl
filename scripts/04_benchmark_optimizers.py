"""Classical optimal-control benchmark for the neural pulse compilers.

This script reproduces the paper's neural-versus-optimal-control benchmark.
It uses independent workers for target-gate sharding and writes reproducibility
artifacts under results/benchmarks/.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, fields
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# CLI / independent multi-GPU worker setup
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-zip", type=str, default=None,
                   help="ZIP containing the nominal config and frozen nominal/robust model weights.")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--rank", type=int, default=0,
                   help="Logical worker rank. GPUs are isolated externally with CUDA_VISIBLE_DEVICES.")
    p.add_argument("--world-size", type=int, default=1,
                   help="Number of independent workers used to shard target gates.")
    p.add_argument("--prepare-only", action="store_true",
                   help="Extract/validate the model archive, then exit. No GPU work.")
    p.add_argument("--worker-only", action="store_true",
                   help="Run only this rank's assigned target gates, then exit.")
    p.add_argument("--merge-only", action="store_true",
                   help="Merge all saved gate records and run final analysis on this GPU.")
    p.add_argument("--self-check-only", action="store_true",
                   help="Load models and run one physics/model self-check on this GPU.")
    p.add_argument("--seed", type=int, default=20260919)
    return p.parse_args()


def init_device(rank):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for worker/merge modes.")
    # Each process is launched with CUDA_VISIBLE_DEVICES containing exactly
    # one physical GPU, so that GPU is always cuda:0 inside the process.
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    print(
        f"[rank {rank}] device={device}, gpu={props.name}, "
        f"memory={props.total_memory / 1024**3:.2f} GB",
        flush=True,
    )
    return device


def rprint(rank, *args, **kwargs):
    print(f"[rank {rank}]", *args, **kwargs, flush=True)


def prepare_input_archive(input_zip, out):
    out.mkdir(parents=True, exist_ok=True)
    (out / "rank_results").mkdir(exist_ok=True)
    extraction = out / "_input_archive"

    if not extraction.exists() or not any(extraction.iterdir()):
        extraction.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(input_zip, "r") as zf:
            zf.extractall(extraction)

    config_path = first_match(extraction, ["nominal_config.json", "config.json"])
    nominal_path = first_match(
        extraction,
        ["nominal_frozen_checkpoint.pt", "frozen_best_checkpoint.pt",
         "best_full_checkpoint.pt", "best.pt"],
    )
    robust_path = first_match(
        extraction,
        ["robust_final_model.pt", "robust_final_state_dict.pt"],
    )

    if config_path is None or nominal_path is None or robust_path is None:
        raise FileNotFoundError(
            "Need config + nominal + robust checkpoints. "
            f"config={config_path}, nominal={nominal_path}, robust={robust_path}"
        )

    return extraction, config_path, nominal_path, robust_path

def sha256_file(path, block=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# -----------------------------------------------------------------------------
# Input extraction / discovery
# -----------------------------------------------------------------------------
def resolve_input_zip(arg, cwd):
    if arg:
        path = Path(arg).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    default_bundle = (cwd / "publication_model_bundle.zip").resolve()
    if default_bundle.exists():
        return default_bundle

    raise FileNotFoundError(
        "No benchmark model bundle found. Pass --input-zip explicitly or "
        "create publication_model_bundle.zip from the public config/checkpoints."
    )


def first_match(root, patterns):
    for pat in patterns:
        hits = list(root.rglob(pat))
        if hits:
            return hits[0]
    return None


# -----------------------------------------------------------------------------
# Configuration / physics / models
# -----------------------------------------------------------------------------
@dataclass
class Config:
    v_hz: tuple = (-921.0, 40.75, 700.0)
    j_hz: tuple = (-64.0, 24.4, 34.1)
    n_slices: int = 300
    dt_s: float = 35e-6
    rabi_max_hz: float = 50_000.0
    width: int = 256
    hidden_layers: int = 10
    dropout: float = 0.25
    batch_size: int = 32
    train_steps: int = 50_000
    lr: float = 5e-4
    weight_decay: float = 1e-3
    grad_clip: float = 1.0
    smooth_lambda: float = 1e-4
    power_lambda: float = 1e-6
    val_gates: int = 512
    val_batch: int = 64
    validate_every: int = 100
    print_every: int = 20
    seed: int = 20260917
    out_dir: str = "outputs/nominal"


def load_cfg(path):
    d = json.load(open(path))
    allowed = {f.name for f in fields(Config)}
    cfg = Config(**{k: v for k, v in d.items() if k in allowed})
    cfg.v_hz = tuple(cfg.v_hz)
    cfg.j_hz = tuple(cfg.j_hz)
    assert cfg.n_slices == 300, cfg.n_slices
    assert abs(cfg.dt_s - 35e-6) < 1e-12, cfg.dt_s
    return cfg


def kron3(a, b, c):
    return torch.kron(torch.kron(a, b), c)


class NMRSystem:
    def __init__(self, cfg, device):
        ct = torch.complex64
        I = torch.eye(2, dtype=ct, device=device)
        X = torch.tensor([[0, 1], [1, 0]], dtype=ct, device=device)
        Y = torch.tensor([[0, -1j], [1j, 0]], dtype=ct, device=device)
        Z = torch.tensor([[1, 0], [0, -1]], dtype=ct, device=device)
        self.I2 = I
        self.I4 = torch.eye(4, dtype=ct, device=device)
        self.I8 = torch.eye(8, dtype=ct, device=device)
        self.X, self.Y, self.Z = X, Y, Z
        self.Xs = [kron3(X, I, I), kron3(I, X, I), kron3(I, I, X)]
        self.Ys = [kron3(Y, I, I), kron3(I, Y, I), kron3(I, I, Y)]
        self.Zs = [kron3(Z, I, I), kron3(I, Z, I), kron3(I, I, Z)]
        self.Hx = sum(self.Xs)
        self.Hy = sum(self.Ys)
        self.pair_ops = []
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            self.pair_ops.append(
                self.Xs[i] @ self.Xs[j]
                + self.Ys[i] @ self.Ys[j]
                + self.Zs[i] @ self.Zs[j]
            )
        v = torch.tensor(cfg.v_hz, dtype=torch.float32, device=device)[None, :]
        j = torch.tensor(cfg.j_hz, dtype=torch.float32, device=device)[None, :]
        self.H0 = self.build_h0(v, j)[0]

    def build_h0(self, v, j):
        H = torch.zeros((len(v), 8, 8), dtype=torch.complex64, device=v.device)
        for i in range(3):
            H += math.pi * v[:, i, None, None].to(H.dtype) * self.Zs[i]
        for k in range(3):
            H += math.pi * j[:, k, None, None].to(H.dtype) * self.pair_ops[k]
        return H


class PulseCompiler(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        layers = []
        d = 4
        for _ in range(cfg.hidden_layers):
            layers += [nn.Linear(d, cfg.width), nn.GELU()]
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            d = cfg.width
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(d, 2 * cfg.n_slices)
        self.cfg = cfg

    def forward(self, q):
        raw = self.head(self.backbone(q)).view(-1, self.cfg.n_slices, 2)
        r2 = (raw * raw).sum(-1, keepdim=True)
        return self.cfg.rabi_max_hz * raw / torch.sqrt(1 + r2)


def extract_state(obj):
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    return obj


# -----------------------------------------------------------------------------
# Benchmark context: populated in main after device/config are known
# -----------------------------------------------------------------------------
CFG = None
SYS = None
DEVICE = None
SEED = None

# optimizer parameters used in the publication benchmark
GRAPE_RESTARTS = 6
GRAPE_STEPS = 2000
GRAPE_LR = 8e-4
GRAPE_INIT_SCALES = [0.003, 0.008, 0.015, 0.03, 0.06, 0.10]
GRAPE_EARLY_STOP_FID = 0.9995

ROBUST_GRAPE_STEPS = 2200
ROBUST_GRAPE_LR = 6e-4
ROBUST_SCENARIOS_RANDOM = 8
ROBUST_RF_ABS = 0.05
ROBUST_CVAR_ALPHA = 0.20
ROBUST_LAMBDA_NOM = 1.0
ROBUST_LAMBDA_MEAN = 1.0
ROBUST_LAMBDA_CVAR = 1.5

USE_LBFGS = True
LBFGS_MAX_ITER = 200
LBFGS_LR = 0.3

NN_SEEDED_ADAM_STEPS = 400
NN_SEEDED_LR = 2e-4
NN_SEEDED_USE_LBFGS = True
NN_SEEDED_LBFGS_MAX_ITER = 120
NN_SEEDED_LBFGS_LR = 0.25

CRAB_MODES = 16
CRAB_RESTARTS = 4
CRAB_STEPS = 1500
CRAB_A0 = 0.05
CRAB_C0 = 0.03
CRAB_INIT_SCALE = 0.005
CRAB_GRAD_CLIP = 1.0

ROUTINE = {
    "common_B0_abs_hz": 2.0,
    "independent_v_abs_hz": 2.0,
    "independent_J_abs_hz": 0.25,
    "phase_bias_abs_deg": 1.0,
    "clock_abs_ppm": 2.0,
    "timing_jitter_rms_ps": 10.0,
}
OOD = {
    "common_B0_abs_hz": 6.0,
    "independent_v_abs_hz": 6.0,
    "independent_J_abs_hz": 1.0,
    "rf_gain_abs_fraction": 0.15,
    "phase_bias_abs_deg": 3.0,
    "clock_abs_ppm": 5.0,
    "timing_jitter_rms_ps": 50.0,
}
ROBUST_SELECTION_SCENARIOS = None


# -----------------------------------------------------------------------------
# Core math helpers
# -----------------------------------------------------------------------------
def unitary_fidelity(target, U):
    ov = (target.conj() * U).sum((-2, -1))
    return ov.abs().square() / 64.0


def lift_target(U2):
    return torch.einsum("bij,kl->bikjl", U2, SYS.I4).reshape(-1, 8, 8)


def q_to_u2(q):
    q0, qx, qy, qz = q.unbind(-1)
    return q0[:, None, None].to(torch.complex64) * SYS.I2 - 1j * (
        qx[:, None, None].to(torch.complex64) * SYS.X
        + qy[:, None, None].to(torch.complex64) * SYS.Y
        + qz[:, None, None].to(torch.complex64) * SYS.Z
    )


def radial_bound(raw):
    r2 = (raw * raw).sum(-1, keepdim=True)
    return CFG.rabi_max_hz * raw / torch.sqrt(1 + r2)


def pulse_to_raw_exact(u):
    y = (u / CFG.rabi_max_hz).detach()
    y2 = (y * y).sum(-1, keepdim=True).clamp(max=1 - 1e-7)
    return y / torch.sqrt(1 - y2)


def control_metrics(u):
    amp = torch.linalg.vector_norm(u, dim=-1)
    diff = u[:, 1:] - u[:, :-1] if u.ndim == 3 else u[1:] - u[:-1]
    return {
        "peak_amp_hz": float(amp.max().detach().cpu()),
        "rms_amp_hz": float(torch.sqrt((amp ** 2).mean()).detach().cpu()),
        "roughness": float((diff.square().sum(-1) / (CFG.rabi_max_hz ** 2)).mean().detach().cpu()),
    }


def propagate_nominal(u):
    ux, uy = u[..., 0], u[..., 1]
    H = (
        SYS.H0[None, None]
        + math.pi * ux[..., None, None].to(torch.complex64) * SYS.Hx
        + math.pi * uy[..., None, None].to(torch.complex64) * SYS.Hy
    )
    Us = torch.matrix_exp((-1j * CFG.dt_s) * H)
    U = SYS.I8[None].expand(len(u), -1, -1).clone()
    for t in range(CFG.n_slices):
        U = Us[:, t] @ U
    return U


def haar_q(n, seed):
    g = torch.Generator(device=DEVICE)
    g.manual_seed(seed)
    q = torch.randn(n, 4, device=DEVICE, generator=g)
    return q / q.norm(dim=-1, keepdim=True)


def make_targets():
    I = torch.eye(2, dtype=torch.complex64, device=DEVICE)
    X, Y, Z = SYS.X, SYS.Y, SYS.Z

    def Rx(a):
        return torch.cos(torch.tensor(a / 2, device=DEVICE)) * I - 1j * torch.sin(torch.tensor(a / 2, device=DEVICE)) * X

    def Ry(a):
        return torch.cos(torch.tensor(a / 2, device=DEVICE)) * I - 1j * torch.sin(torch.tensor(a / 2, device=DEVICE)) * Y

    def Rz(a):
        return torch.cos(torch.tensor(a / 2, device=DEVICE)) * I - 1j * torch.sin(torch.tensor(a / 2, device=DEVICE)) * Z

    def su2_from_u2(U):
        return U / torch.sqrt(torch.linalg.det(U))

    def u2_to_q(U):
        U = su2_from_u2(U)
        q0 = torch.trace(U).real / 2
        qx = (0.5j * torch.trace(X @ U)).real
        qy = (0.5j * torch.trace(Y @ U)).real
        qz = (0.5j * torch.trace(Z @ U)).real
        q = torch.stack([q0, qx, qy, qz]).float()
        return q / q.norm()

    H = (X + Z) / math.sqrt(2)
    S = torch.diag(torch.tensor([1, 1j], dtype=torch.complex64, device=DEVICE))
    Tgate = torch.diag(torch.tensor([1, np.exp(1j * math.pi / 4)], dtype=torch.complex64, device=DEVICE))
    axis = torch.tensor([1.0, 2.0, 3.0], device=DEVICE)
    axis /= axis.norm()
    generic = math.cos(0.73 / 2) * I - 1j * math.sin(0.73 / 2) * (axis[0] * X + axis[1] * Y + axis[2] * Z)

    named = {
        "I": I,
        "X": X,
        "Y": Y,
        "Z": Z,
        "H": H,
        "S": S,
        "T": Tgate,
        "sqrtX": Rx(math.pi / 2),
        "sqrtY": Ry(math.pi / 2),
        "Rz_pi2": Rz(math.pi / 2),
        "generic_Rn": generic,
    }
    n_haar = 8
    hq = haar_q(n_haar, SEED + 10)
    hu = q_to_u2(hq)
    names = list(named.keys()) + [f"Haar_{i:02d}" for i in range(n_haar)]
    U2_list = [su2_from_u2(named[k]) for k in named] + [hu[i] for i in range(n_haar)]
    Q_list = [u2_to_q(named[k]) for k in named] + [hq[i] for i in range(n_haar)]
    U2 = torch.stack(U2_list)
    Q = torch.stack(Q_list)
    targets = lift_target(U2)
    return names, named, Q, targets


# -----------------------------------------------------------------------------
# Uncertainty / perturbed propagation
# -----------------------------------------------------------------------------
def zeros_scen(S):
    z = lambda *shape: torch.zeros(*shape, device=DEVICE)
    return {
        "common_B0_hz": z(S),
        "v_offset_hz": z(S, 3),
        "J_offset_hz": z(S, 3),
        "rf_gain": torch.ones(S, device=DEVICE),
        "phase_bias_rad": z(S),
        "clock_ppm": z(S),
        "timing_jitter_s": z(S, CFG.n_slices),
    }


def random_routine(S, rf_abs, seed):
    g = torch.Generator(device=DEVICE)
    g.manual_seed(seed)
    unif = lambda *shape: 2 * torch.rand(*shape, device=DEVICE, generator=g) - 1
    s = zeros_scen(S)
    s["common_B0_hz"] = unif(S) * ROUTINE["common_B0_abs_hz"]
    s["v_offset_hz"] = unif(S, 3) * ROUTINE["independent_v_abs_hz"]
    s["J_offset_hz"] = unif(S, 3) * ROUTINE["independent_J_abs_hz"]
    s["rf_gain"] = 1 + unif(S) * rf_abs
    s["phase_bias_rad"] = torch.deg2rad(unif(S) * ROUTINE["phase_bias_abs_deg"])
    s["clock_ppm"] = unif(S) * ROUTINE["clock_abs_ppm"]
    s["timing_jitter_s"] = torch.randn(S, CFG.n_slices, device=DEVICE, generator=g) * ROUTINE["timing_jitter_rms_ps"] * 1e-12
    return s


def boundaries(r):
    s = zeros_scen(2)
    s["rf_gain"] = torch.tensor([1 - r, 1 + r], device=DEVICE)
    return s


def cat_scen(a, b):
    return {k: torch.cat([a[k], b[k]], 0) for k in a}


def sobol_scen(S, profile, rf_abs, seed):
    p = ROUTINE if profile == "routine" else OOD
    eng = torch.quasirandom.SobolEngine(10, scramble=True, seed=seed)
    z = 2 * eng.draw(S).to(DEVICE) - 1
    k = 0
    s = zeros_scen(S)
    s["common_B0_hz"] = z[:, k] * p["common_B0_abs_hz"]; k += 1
    s["v_offset_hz"] = z[:, k:k+3] * p["independent_v_abs_hz"]; k += 3
    s["J_offset_hz"] = z[:, k:k+3] * p["independent_J_abs_hz"]; k += 3
    r = rf_abs if profile == "routine" else p["rf_gain_abs_fraction"]
    s["rf_gain"] = 1 + z[:, k] * r; k += 1
    s["phase_bias_rad"] = torch.deg2rad(z[:, k] * p["phase_bias_abs_deg"]); k += 1
    s["clock_ppm"] = z[:, k] * p["clock_abs_ppm"]; k += 1
    g = torch.Generator(device=DEVICE); g.manual_seed(seed + 999)
    s["timing_jitter_s"] = torch.randn(S, CFG.n_slices, device=DEVICE, generator=g) * p["timing_jitter_rms_ps"] * 1e-12
    return s


def propagate_scen(u, s):
    G, T, _ = u.shape
    S = len(s["rf_gain"])
    v0 = torch.tensor(CFG.v_hz, dtype=torch.float32, device=DEVICE)
    j0 = torch.tensor(CFG.j_hz, dtype=torch.float32, device=DEVICE)
    v = v0[None] + s["common_B0_hz"][:, None] + s["v_offset_hz"]
    j = j0[None] + s["J_offset_hz"]
    H0 = SYS.build_h0(v, j)
    c = u[:, None].expand(G, S, -1, -1)
    ph = s["phase_bias_rad"][None, :, None]
    gain = s["rf_gain"][None, :, None]
    cp, sp = torch.cos(ph), torch.sin(ph)
    ux = gain * (cp * c[..., 0] - sp * c[..., 1])
    uy = gain * (sp * c[..., 0] + cp * c[..., 1])
    ux = ux.reshape(G * S, T)
    uy = uy.reshape(G * S, T)
    H0 = H0[None].expand(G, S, -1, -1).reshape(G * S, 8, 8)
    H = H0[:, None] + math.pi * ux[..., None, None].to(torch.complex64) * SYS.Hx + math.pi * uy[..., None, None].to(torch.complex64) * SYS.Hy
    dt = CFG.dt_s * (1 + s["clock_ppm"] * 1e-6)[:, None] + s["timing_jitter_s"]
    dt = dt[None].expand(G, S, -1).reshape(G * S, T)
    Us = torch.matrix_exp((-1j * dt[..., None, None]) * H)
    U = SYS.I8[None].expand(G * S, -1, -1).clone()
    for t in range(T):
        U = Us[:, t] @ U
    return U.reshape(G, S, 8, 8)


def scen_fid(target, u, s):
    U = propagate_scen(u, s)
    G, S = U.shape[:2]
    targ = target[:, None].expand(G, S, -1, -1)
    ov = (targ.conj() * U).sum((-2, -1))
    return ov.abs().square() / 64.0


def cvar_loss(infid, alpha):
    k = max(1, int(math.ceil(alpha * infid.shape[1])))
    return torch.topk(infid, k, dim=1).values.mean()


# -----------------------------------------------------------------------------
# GRAPE / robust GRAPE / NN-seeded GRAPE
# -----------------------------------------------------------------------------
def control_penalty(u):
    diff = u[:, 1:] - u[:, :-1]
    smooth = (diff.square().sum(-1) / (CFG.rabi_max_hz ** 2)).mean()
    power = (u.square().sum(-1) / (CFG.rabi_max_hz ** 2)).mean()
    return CFG.smooth_lambda * smooth + CFG.power_lambda * power


def grape_objective(target, u, robust=False, scenarios=None):
    F0 = unitary_fidelity(target[None], propagate_nominal(u))
    if not robust:
        core = (1 - F0).mean()
        return core + control_penalty(u), F0.mean(), None, None
    if scenarios is None:
        raise ValueError("Robust GRAPE requires scenarios")
    Fs = scen_fid(target[None], u, scenarios)
    inf = 1 - Fs
    cvar = cvar_loss(inf, ROBUST_CVAR_ALPHA)
    core = ROBUST_LAMBDA_NOM * (1 - F0).mean() + ROBUST_LAMBDA_MEAN * inf.mean() + ROBUST_LAMBDA_CVAR * cvar
    return core + control_penalty(u), F0.mean(), Fs.mean(), cvar


@torch.inference_mode()
def candidate_score(target, u, robust=False):
    loss, F0, Frob, cvar = grape_objective(
        target, u, robust=robust,
        scenarios=ROBUST_SELECTION_SCENARIOS if robust else None,
    )
    return {
        "score": float(loss),
        "nominal_fidelity": float(F0),
        "robust_mean_fidelity": None if Frob is None else float(Frob),
        "cvar_infidelity": None if cvar is None else float(cvar),
    }


def grape_one(target, seed, restart_index, steps, lr, robust=False):
    g = torch.Generator(device=DEVICE); g.manual_seed(seed)
    scale = GRAPE_INIT_SCALES[restart_index % len(GRAPE_INIT_SCALES)]
    raw = (scale * torch.randn(CFG.n_slices, 2, device=DEVICE, generator=g)).requires_grad_(True)
    opt = torch.optim.Adam([raw], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.05)
    train_scen = None
    if robust:
        train_scen = cat_scen(random_routine(ROBUST_SCENARIOS_RANDOM, ROBUST_RF_ABS, seed + 5000), boundaries(ROBUST_RF_ABS))

    best_raw = raw.detach().clone()
    best_train_score = float("inf")
    best_iteration = 0
    hist = []
    sync(DEVICE); t0 = time.perf_counter()

    for it in range(steps):
        opt.zero_grad(set_to_none=True)
        u = radial_bound(raw)[None]
        loss, F0, Frob, cvar = grape_objective(target, u, robust=robust, scenarios=train_scen)
        train_score = float(loss.detach())
        fid = float(F0.detach())
        if train_score < best_train_score:
            best_train_score = train_score
            best_raw = raw.detach().clone()
            best_iteration = it + 1
        loss.backward()
        torch.nn.utils.clip_grad_norm_([raw], 5.0)
        opt.step(); scheduler.step()
        if it == 0 or (it + 1) % 25 == 0 or it == steps - 1:
            hist.append({
                "iteration": it + 1,
                "fidelity": fid,
                "loss": train_score,
                "learning_rate": float(opt.param_groups[0]["lr"]),
                "robust_mean_fidelity": None if Frob is None else float(Frob.detach()),
                "cvar_infidelity": None if cvar is None else float(cvar.detach()),
            })
        if (not robust) and fid >= GRAPE_EARLY_STOP_FID:
            break

    with torch.no_grad():
        u_final = radial_bound(raw)[None]
        loss_final, _, _, _ = grape_objective(target, u_final, robust=robust, scenarios=train_scen)
        if float(loss_final) < best_train_score:
            best_train_score = float(loss_final)
            best_raw = raw.detach().clone()
            best_iteration = it + 1

    sync(DEVICE); elapsed = time.perf_counter() - t0
    u_best = radial_bound(best_raw)[None].detach()
    sel = candidate_score(target, u_best, robust=robust)
    return {
        "pulse": u_best[0],
        "nominal_fidelity": sel["nominal_fidelity"],
        "selection_score": sel["score"],
        "robust_mean_fidelity": sel["robust_mean_fidelity"],
        "cvar_infidelity": sel["cvar_infidelity"],
        "elapsed_s": elapsed,
        "history": hist,
        "restart": restart_index,
        "init_scale": scale,
        "best_iteration": best_iteration,
        "best_training_score": best_train_score,
    }


def lbfgs_refine(target, u0, robust=False, max_iter=None, lr=None):
    if max_iter is None: max_iter = LBFGS_MAX_ITER
    if lr is None: lr = LBFGS_LR
    raw = pulse_to_raw_exact(u0).clone().requires_grad_(True)
    opt = torch.optim.LBFGS([raw], lr=lr, max_iter=max_iter, history_size=30, line_search_fn="strong_wolfe")
    scenarios = ROBUST_SELECTION_SCENARIOS if robust else None

    def closure():
        opt.zero_grad(set_to_none=True)
        u = radial_bound(raw)[None]
        loss, _, _, _ = grape_objective(target, u, robust=robust, scenarios=scenarios)
        loss.backward()
        return loss

    sync(DEVICE); t0 = time.perf_counter()
    try:
        opt.step(closure)
        ok = True
    except Exception as e:
        print("L-BFGS warning:", repr(e), flush=True)
        ok = False
    sync(DEVICE); elapsed = time.perf_counter() - t0
    if not ok:
        return u0.detach(), elapsed, False
    return radial_bound(raw.detach()), elapsed, True


def grape_best(target, base_seed, robust=False, rank=0):
    candidates = []
    total_time = 0.0
    for r in range(GRAPE_RESTARTS):
        rec = grape_one(
            target, base_seed + r, r,
            ROBUST_GRAPE_STEPS if robust else GRAPE_STEPS,
            ROBUST_GRAPE_LR if robust else GRAPE_LR,
            robust=robust,
        )
        total_time += rec["elapsed_s"]
        candidates.append(rec)
        rprint(rank, f"restart {r}: init={rec['init_scale']:.3f}, Fnom={rec['nominal_fidelity']:.6f}, selection={rec['selection_score']:.6f}, best_it={rec['best_iteration']}")

    best = min(candidates, key=lambda x: x["selection_score"])
    lbfgs_used = False
    if USE_LBFGS:
        refined, t_ref, ok = lbfgs_refine(target, best["pulse"], robust=robust, max_iter=LBFGS_MAX_ITER, lr=LBFGS_LR)
        total_time += t_ref
        if ok:
            ev = candidate_score(target, refined[None], robust=robust)
            if ev["score"] < best["selection_score"]:
                best = copy.deepcopy(best)
                best["pulse"] = refined
                best["selection_score"] = ev["score"]
                best["nominal_fidelity"] = ev["nominal_fidelity"]
                best["robust_mean_fidelity"] = ev["robust_mean_fidelity"]
                best["cvar_infidelity"] = ev["cvar_infidelity"]
                lbfgs_used = True

    best["total_search_time_s"] = total_time
    best["lbfgs_used"] = lbfgs_used
    rprint(rank, f"SELECTED FINAL: Fnom={best['nominal_fidelity']:.6f}, score={best['selection_score']:.6f}, restart={best['restart']}, init={best['init_scale']:.3f}, LBFGS={best['lbfgs_used']}, total_time={total_time:.2f}s")
    return best


def nn_seeded_grape(target, u_seed, rank=0):
    raw = pulse_to_raw_exact(u_seed).clone().requires_grad_(True)
    opt = torch.optim.Adam([raw], lr=NN_SEEDED_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NN_SEEDED_ADAM_STEPS, eta_min=NN_SEEDED_LR * 0.05)
    best_raw = raw.detach().clone()
    best_score = float("inf")
    best_iteration = 0
    hist = []
    sync(DEVICE); t0 = time.perf_counter()
    for it in range(NN_SEEDED_ADAM_STEPS):
        opt.zero_grad(set_to_none=True)
        u = radial_bound(raw)[None]
        loss, F0, _, _ = grape_objective(target, u, robust=False)
        score = float(loss.detach()); fid = float(F0.detach())
        if score < best_score:
            best_score = score; best_raw = raw.detach().clone(); best_iteration = it + 1
        loss.backward(); torch.nn.utils.clip_grad_norm_([raw], 5.0); opt.step(); scheduler.step()
        if it == 0 or (it + 1) % 20 == 0 or it == NN_SEEDED_ADAM_STEPS - 1:
            hist.append({"iteration": it + 1, "fidelity": fid, "loss": score, "learning_rate": float(opt.param_groups[0]["lr"]), "robust_mean_fidelity": None, "cvar_infidelity": None})
        if fid >= GRAPE_EARLY_STOP_FID:
            break
    sync(DEVICE); adam_time = time.perf_counter() - t0
    u_best = radial_bound(best_raw).detach()
    ev = candidate_score(target, u_best[None], robust=False)
    total_time = adam_time; lbfgs_used = False
    if NN_SEEDED_USE_LBFGS:
        refined, t_ref, ok = lbfgs_refine(target, u_best, robust=False, max_iter=NN_SEEDED_LBFGS_MAX_ITER, lr=NN_SEEDED_LBFGS_LR)
        total_time += t_ref
        if ok:
            rev = candidate_score(target, refined[None], robust=False)
            if rev["score"] < ev["score"]:
                u_best = refined; ev = rev; lbfgs_used = True
    rprint(rank, f"NN-SEEDED FINAL: Fnom={ev['nominal_fidelity']:.6f}, Adam_best_it={best_iteration}, LBFGS={lbfgs_used}, total_time={total_time:.2f}s")
    return {
        "pulse": u_best.detach(),
        "nominal_fidelity": ev["nominal_fidelity"],
        "selection_score": ev["score"],
        "elapsed_s": total_time,
        "history": hist,
        "best_iteration": best_iteration,
        "lbfgs_used": lbfgs_used,
    }


# -----------------------------------------------------------------------------
# CRAB-style SPSA
# -----------------------------------------------------------------------------
def crab_controls(theta, freq):
    K = len(freq)
    t = torch.linspace(0, 1, CFG.n_slices, device=DEVICE)
    vals = []
    for ch in range(2):
        y = theta[ch, 0] * torch.ones_like(t)
        a = theta[ch, 1:1+K]
        b = theta[ch, 1+K:1+2*K]
        ang = 2 * math.pi * freq[:, None] * t[None]
        y = y + (a[:, None] * torch.sin(ang) + b[:, None] * torch.cos(ang)).sum(0)
        vals.append(y)
    return radial_bound(torch.stack(vals, -1))


@torch.inference_mode()
def objective_u(target, u):
    F = unitary_fidelity(target[None], propagate_nominal(u[None]))
    return float((1 - F).mean() + control_penalty(u[None]))


def crab_spsa_one(target, seed, restart_index):
    rng = np.random.default_rng(seed)
    K = CRAB_MODES
    freq = torch.tensor(np.arange(1, K + 1) + rng.uniform(-0.35, 0.35, K), dtype=torch.float32, device=DEVICE)
    if restart_index == 0:
        theta = torch.zeros((2, 1 + 2 * K), dtype=torch.float32, device=DEVICE)
    else:
        theta = torch.tensor(rng.normal(0, CRAB_INIT_SCALE, size=(2, 1 + 2 * K)), dtype=torch.float32, device=DEVICE)
    best_theta = theta.clone(); best_loss = objective_u(target, crab_controls(theta, freq)); best_iteration = 0; hist = []
    sync(DEVICE); t0 = time.perf_counter()
    for k in range(CRAB_STEPS):
        ak = CRAB_A0 / ((k + 10) ** 0.602)
        ck = CRAB_C0 / ((k + 1) ** 0.101)
        delta = torch.tensor(rng.choice([-1.0, 1.0], size=theta.shape), dtype=torch.float32, device=DEVICE)
        yp = objective_u(target, crab_controls(theta + ck * delta, freq))
        ym = objective_u(target, crab_controls(theta - ck * delta, freq))
        ghat = ((yp - ym) / (2 * ck)) * delta
        gn = torch.linalg.vector_norm(ghat)
        if gn > CRAB_GRAD_CLIP:
            ghat = ghat * (CRAB_GRAD_CLIP / gn)
        theta = theta - ak * ghat
        current_loss = objective_u(target, crab_controls(theta, freq))
        if current_loss < best_loss:
            best_loss = current_loss; best_theta = theta.clone(); best_iteration = k + 1
        if k == 0 or (k + 1) % 25 == 0 or k == CRAB_STEPS - 1:
            hist.append({"iteration": k + 1, "fidelity": 1 - current_loss, "loss": current_loss})
    sync(DEVICE); elapsed = time.perf_counter() - t0
    u = crab_controls(best_theta, freq).detach()
    F = float(unitary_fidelity(target[None], propagate_nominal(u[None])).item())
    return {"pulse": u, "fidelity": F, "objective": best_loss, "elapsed_s": elapsed, "history": hist, "restart": restart_index, "best_iteration": best_iteration}


def crab_best(target, base_seed, rank=0):
    total_time = 0.0; candidates = []
    for r in range(CRAB_RESTARTS):
        rec = crab_spsa_one(target, base_seed + r, r)
        total_time += rec["elapsed_s"]; candidates.append(rec)
        rprint(rank, f"CRAB restart {r}: F={rec['fidelity']:.6f}, objective={rec['objective']:.6f}, best_it={rec['best_iteration']}")
    best = min(candidates, key=lambda x: x["objective"])
    best = copy.deepcopy(best); best["total_search_time_s"] = total_time
    return best


# -----------------------------------------------------------------------------
# Persistence / merging
# -----------------------------------------------------------------------------
def safe_method(m):
    return m.replace(" ", "_").replace("/", "_")


def save_gate_record(rank_dir, idx, name, rows, pulses, histories):
    payload = {
        "index": idx,
        "name": name,
        "rows": rows,
        "pulses": {m: p.detach().cpu() for m, p in pulses.items()},
        "histories": histories,
    }
    tmp = rank_dir / f"gate_{idx:03d}.pt.tmp"
    final = rank_dir / f"gate_{idx:03d}.pt"
    torch.save(payload, tmp)
    os.replace(tmp, final)


def load_all_gate_records(output_dir, n_targets):
    records = {}
    for p in sorted((output_dir / "rank_results").rglob("gate_*.pt")):
        rec = torch.load(p, map_location="cpu", weights_only=False)
        records[int(rec["index"])] = rec
    missing = [i for i in range(n_targets) if i not in records]
    if missing:
        raise RuntimeError(f"Missing gate records after distributed run: {missing}")
    return [records[i] for i in range(n_targets)]


# -----------------------------------------------------------------------------
# Rank-0 post-processing
# -----------------------------------------------------------------------------
def warmed_neural_timing(nominal_model, robust_model):
    timing = []
    batches = [1, 16, 128, 1024, 4096]
    with torch.inference_mode():
        dummy = haar_q(256, SEED + 123456)
        for _ in range(20):
            _ = nominal_model(dummy); _ = robust_model(dummy)
        sync(DEVICE)
        for method, model in [("Neural nominal", nominal_model), ("Neural robust", robust_model)]:
            for n in batches:
                q = haar_q(n, SEED + 1000 + n)
                for _ in range(5): _ = model(q)
                sync(DEVICE)
                reps = 50 if n <= 128 else 10
                times = []
                for _ in range(reps):
                    sync(DEVICE); t0 = time.perf_counter(); _ = model(q); sync(DEVICE)
                    times.append(time.perf_counter() - t0)
                med = float(np.median(times))
                timing.append({"method": method, "batch_size": n, "total_time_s": med, "latency_per_gate_s": med / n, "gates_per_s": n / med, "repetitions": reps})
    return pd.DataFrame(timing)


def summarize_methods(results, methods):
    out = []
    for method in methods:
        d = results[results.method == method]
        vals = d.fidelity.values.astype(float)
        out.append({
            "method": method,
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
            "p05": float(np.quantile(vals, 0.05)),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "mean_compile_time_s": float(d.compile_time_s.mean()),
            "median_compile_time_s": float(d.compile_time_s.median()),
            "mean_peak_amp_hz": float(d.peak_amp_hz.mean()),
            "mean_rms_amp_hz": float(d.rms_amp_hz.mean()),
            "mean_roughness": float(d.roughness.mean()),
        })
    return pd.DataFrame(out)


def savefig(figs, name):
    plt.tight_layout()
    plt.savefig(figs / f"{name}.png", dpi=600, bbox_inches="tight")
    plt.savefig(figs / f"{name}.pdf", bbox_inches="tight")
    plt.close()


def postprocess_rank0(output_dir, input_zip, config_path, nominal_path, robust_path, nominal_model, robust_model, names, named, Q, targets, gate_records):
    methods = ["Neural nominal", "Neural robust", "GRAPE", "Robust GRAPE", "NN-seeded GRAPE", "CRAB-SPSA"]
    raw_dir = output_dir / "raw"; tables = output_dir / "tables"; figs = output_dir / "figures"; pulses_dir = output_dir / "pulses"
    for d in [raw_dir, tables, figs, pulses_dir]: d.mkdir(parents=True, exist_ok=True)

    rows = []
    conv_rows = []
    pulse_bank = {m: [] for m in methods}
    for rec in gate_records:
        rows.extend(rec["rows"])
        for m in methods:
            pulse_bank[m].append(rec["pulses"][m])
        for method, hist in rec["histories"].items():
            for h in hist:
                rr = {"gate": rec["name"], "method": method}
                rr.update(h)
                conv_rows.append(rr)

    results = pd.DataFrame(rows)
    timing = warmed_neural_timing(nominal_model, robust_model)
    single = {r.method: float(r.latency_per_gate_s) for _, r in timing[timing.batch_size == 1].iterrows()}
    for m in ["Neural nominal", "Neural robust"]:
        results.loc[results.method == m, "compile_time_s"] = single[m]
    results.to_csv(tables / "per_gate_method_results.csv", index=False)
    timing.to_csv(tables / "neural_latency_throughput.csv", index=False)

    pulse_bank = {m: torch.stack(pulse_bank[m]).to(DEVICE) for m in methods}
    for m, u in pulse_bank.items():
        np.save(raw_dir / f"pulses_{safe_method(m)}.npy", u.detach().cpu().numpy())

    summary = summarize_methods(results, methods)
    summary.to_csv(tables / "benchmark_method_summary.csv", index=False)
    pd.DataFrame(conv_rows).to_csv(tables / "benchmark_optimizer_convergence.csv", index=False)

    # RF and B0 sweeps
    RF_VALUES = np.linspace(-0.15, 0.15, 25, dtype=np.float32)
    B0_VALUES = np.linspace(-10, 10, 21, dtype=np.float32)
    def rf_suite(vals):
        s = zeros_scen(len(vals)); s["rf_gain"] = 1 + torch.tensor(vals, device=DEVICE); return s
    def b0_suite(vals):
        s = zeros_scen(len(vals)); s["common_B0_hz"] = torch.tensor(vals, device=DEVICE); return s
    RFSC, B0SC = rf_suite(RF_VALUES), b0_suite(B0_VALUES)
    rf_rows, b0_rows = [], []
    for method in methods:
        print("[rank 0] sweep", method, flush=True)
        Fr = scen_fid(targets, pulse_bank[method], RFSC).detach().cpu().numpy()
        Fb = scen_fid(targets, pulse_bank[method], B0SC).detach().cpu().numpy()
        np.save(raw_dir / f"rf_sweep_{safe_method(method)}.npy", Fr)
        np.save(raw_dir / f"b0_sweep_{safe_method(method)}.npy", Fb)
        for j, x in enumerate(RF_VALUES):
            rf_rows.append({"method": method, "rf_error_percent": float(100 * x), "mean": float(Fr[:, j].mean()), "p05": float(np.quantile(Fr[:, j], 0.05)), "min": float(Fr[:, j].min())})
        for j, x in enumerate(B0_VALUES):
            b0_rows.append({"method": method, "b0_offset_hz": float(x), "mean": float(Fb[:, j].mean()), "p05": float(np.quantile(Fb[:, j], 0.05)), "min": float(Fb[:, j].min())})
    RFDF, B0DF = pd.DataFrame(rf_rows), pd.DataFrame(b0_rows)
    RFDF.to_csv(tables / "benchmark_rf_sweep.csv", index=False)
    B0DF.to_csv(tables / "benchmark_b0_sweep.csv", index=False)

    # Joint robustness
    JOINT_SCENARIOS = 64
    routine_s = sobol_scen(JOINT_SCENARIOS, "routine", 0.05, SEED + 7001)
    ood_s = sobol_scen(JOINT_SCENARIOS, "ood", 0.15, SEED + 7002)
    joint_rows = []
    for suite_name, s in [("routine_joint_rf05", routine_s), ("conservative_ood_rf15", ood_s)]:
        np.savez(raw_dir / f"{suite_name}_scenarios.npz", **{k: v.cpu().numpy() for k, v in s.items()})
        for method in methods:
            print("[rank 0]", suite_name, method, flush=True)
            F = scen_fid(targets, pulse_bank[method], s).detach().cpu().numpy()
            np.save(raw_dir / f"{suite_name}_{safe_method(method)}.npy", F)
            gate_mean, gate_worst = F.mean(1), F.min(1)
            L = 1 - F; k = max(1, int(np.ceil(0.10 * F.shape[1])))
            cvar = np.partition(L, F.shape[1] - k, axis=1)[:, F.shape[1] - k:].mean(1)
            joint_rows.append({
                "suite": suite_name, "method": method, "mean": float(F.mean()),
                "p05": float(np.quantile(F, 0.05)), "p01": float(np.quantile(F, 0.01)),
                "min": float(F.min()), "gate_worst_mean": float(gate_worst.mean()),
                "gate_mean_p05": float(np.quantile(gate_mean, 0.05)),
                "mean_gate_cvar10_infidelity": float(cvar.mean()),
            })
    JOINT = pd.DataFrame(joint_rows)
    JOINT.to_csv(tables / "benchmark_joint_robustness.csv", index=False)

    # Pulse CSVs
    for method in methods:
        mdir = pulses_dir / safe_method(method); mdir.mkdir(exist_ok=True)
        U = pulse_bank[method].detach().cpu().numpy()
        for i, name in enumerate(names):
            ux, uy = U[i, :, 0], U[i, :, 1]
            amp = np.sqrt(ux ** 2 + uy ** 2); phase = np.arctan2(uy, ux)
            pd.DataFrame({
                "slice": np.arange(CFG.n_slices),
                "time_us": np.arange(CFG.n_slices) * CFG.dt_s * 1e6,
                "ux_hz": ux, "uy_hz": uy, "amplitude_hz": amp, "phase_rad": phase,
            }).to_csv(mdir / f"{name}.csv", index=False)

    # Figures
    named_names = list(named.keys())
    D = results[results.gate.isin(named_names)]
    pivot = D.pivot(index="gate", columns="method", values="fidelity").loc[named_names]
    pivot.plot(kind="bar", figsize=(13, 5)); plt.ylabel("Nominal unitary fidelity"); plt.ylim(0, 1.005); plt.xticks(rotation=30, ha="right"); plt.grid(axis="y", alpha=0.2); savefig(figs, "named_gate_fidelity_all_methods")
    S = summary.sort_values("median_compile_time_s")
    plt.figure(figsize=(9, 4.8)); plt.bar(S.method, S.median_compile_time_s); plt.yscale("log"); plt.ylabel("Median online compile/optimization time per gate (s)"); plt.xticks(rotation=25, ha="right"); plt.grid(axis="y", alpha=0.2); savefig(figs, "compile_runtime_log")
    plt.figure(figsize=(8, 5))
    for _, r in summary.iterrows():
        plt.scatter(r.mean_compile_time_s, r["mean"], s=70); plt.annotate(r.method, (r.mean_compile_time_s, r["mean"]), xytext=(5, 4), textcoords="offset points")
    plt.xscale("log"); plt.xlabel("Mean compilation time per gate (s)"); plt.ylabel("Mean nominal fidelity"); plt.grid(alpha=0.2); savefig(figs, "fidelity_vs_compile_time")

    plt.figure(figsize=(10, 5.5))
    for method in methods:
        d = RFDF[RFDF.method == method]
        plt.plot(d.rf_error_percent, d["mean"], label=method)
    plt.xlabel("Global RF gain error (%)")
    plt.ylabel("Mean fidelity")
    plt.ylim(0, 1.005)
    plt.legend()
    plt.grid(alpha=0.2)
    savefig(figs, "rf_robustness_all_methods")

    plt.figure(figsize=(10, 5.5))
    for method in methods:
        d = B0DF[B0DF.method == method]
        plt.plot(d.b0_offset_hz, d["mean"], label=method)
    plt.xlabel("Common B0 / carrier offset (Hz)")
    plt.ylabel("Mean fidelity")
    plt.ylim(0, 1.005)
    plt.legend()
    plt.grid(alpha=0.2)
    savefig(figs, "b0_robustness_all_methods")

    P = JOINT.pivot(index="method", columns="suite", values="mean").loc[methods]
    P.plot(kind="bar", figsize=(10, 5))
    plt.ylabel("Mean gate × scenario fidelity")
    plt.ylim(0, 1.005)
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.2)
    savefig(figs, "joint_robustness_all_methods")

    convdf = pd.DataFrame(conv_rows)
    plt.figure(figsize=(9, 5))
    for method in ["GRAPE", "Robust GRAPE", "NN-seeded GRAPE", "CRAB-SPSA"]:
        d = convdf[(convdf.gate == "H") & (convdf.method == method)]
        if len(d):
            plt.plot(d.iteration, d.fidelity, label=method)
    plt.xlabel("Optimization iteration")
    plt.ylabel("Nominal fidelity")
    plt.ylim(0, 1.005)
    plt.legend()
    plt.grid(alpha=0.2)
    savefig(figs, "hadamard_optimizer_convergence")

    # Preserve the exact model/config inputs with the benchmark output.
    shutil.copy2(config_path, output_dir / "nominal_config.json")
    shutil.copy2(nominal_path, output_dir / "nominal_checkpoint.pt")
    shutil.copy2(robust_path, output_dir / "robust_checkpoint.pt")
    np.save(raw_dir / "H0_full_XX_YY_ZZ.npy", SYS.H0.detach().cpu().numpy())

    report = {
        "version": "publication_optimizer_benchmark",
        "input_bundle_sha256": sha256_file(input_zip),
        "nominal_checkpoint_sha256": sha256_file(nominal_path),
        "robust_checkpoint_sha256": sha256_file(robust_path),
        "method_summary": summary.to_dict(orient="records"),
        "joint_robustness": JOINT.to_dict(orient="records"),
        "neural_timing": timing.to_dict(orient="records"),
        "notes": {
            "GRAPE": "Standalone 600-pixel I/Q GRAPE with six diversified nonzero starts, cosine decay, best-iterate retention, and L-BFGS refinement. Runtime includes all restarts and refinement.",
            "Robust GRAPE": "Direct robust optimizer using routine uncertainty plus exact ±5% RF boundaries; selected on a shared held-out robust scenario ensemble.",
            "NN-seeded GRAPE": "Neural pulse initialization followed by short target-specific GRAPE/L-BFGS refinement.",
            "CRAB-SPSA": "16-mode chopped Fourier basis with stabilized SPSA.",
            "parallelism": "Independent Python workers process disjoint target-gate subsets; each target is optimized on one GPU.",
        },
    }
    json.dump(
        report,
        open(output_dir / "benchmark_report.json", "w"),
        indent=2,
        default=str,
    )

    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(DEVICE),
    }
    try:
        env["nvidia_smi"] = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv",
            ],
            text=True,
        )
    except Exception as exc:
        env["nvidia_smi_error"] = str(exc)
    json.dump(env, open(output_dir / "environment.json", "w"), indent=2)

    with open(output_dir / "SUMMARY.txt", "w") as handle:
        handle.write(
            "NEURAL PULSE COMPILER — OPTIMAL-CONTROL BENCHMARK\n"
            + "=" * 64
            + "\n\n"
        )
        handle.write("METHOD SUMMARY\n" + summary.to_string(index=False) + "\n\n")
        handle.write("JOINT ROBUSTNESS\n" + JOINT.to_string(index=False) + "\n\n")
        handle.write("NEURAL TIMING\n" + timing.to_string(index=False) + "\n")

    extraction = output_dir / "_input_archive"
    if extraction.exists():
        shutil.rmtree(extraction)

    manifest = [
        str(path.relative_to(output_dir))
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    ]
    (output_dir / "MANIFEST.txt").write_text("\n".join(manifest))

    checks = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            checks.append(
                f"{sha256_file(path)}  {path.relative_to(output_dir)}"
            )
    (output_dir / "SHA256SUMS.txt").write_text("\n".join(checks))

    archive_base = output_dir.parent / f"{output_dir.name}_results"
    archive_path = shutil.make_archive(
        str(archive_base),
        "zip",
        root_dir=output_dir.parent,
        base_dir=output_dir.name,
    )
    print("=" * 72)
    print("OPTIMAL-CONTROL BENCHMARK COMPLETE")
    print("Output:", output_dir)
    print("Result archive:", archive_path)
    print(summary.to_string(index=False))
    print("=" * 72, flush=True)
    return archive_path


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    global CFG, SYS, DEVICE, SEED, ROBUST_SELECTION_SCENARIOS

    args = parse_args()
    rank = int(args.rank)
    world = int(args.world_size)

    if world < 1:
        raise ValueError("--world-size must be >= 1")
    if not (0 <= rank < world):
        raise ValueError(
            f"--rank must satisfy 0 <= rank < world_size; "
            f"got rank={rank}, world={world}"
        )

    SEED = args.seed
    random.seed(SEED + rank)
    np.random.seed(SEED + rank)
    torch.manual_seed(SEED + rank)
    torch.backends.cuda.matmul.allow_tf32 = True

    out = Path(args.output_dir).expanduser().resolve()
    input_zip = resolve_input_zip(args.input_zip, Path.cwd())

    if args.prepare_only:
        extraction, config_path, nominal_path, robust_path = prepare_input_archive(
            input_zip, out
        )
        print("PREPARE PASSED", flush=True)
        print("Input bundle:", input_zip, flush=True)
        print("Input SHA256:", sha256_file(input_zip), flush=True)
        print("Extraction:", extraction, flush=True)
        print("Config:", config_path, flush=True)
        print("Nominal checkpoint:", nominal_path, flush=True)
        print("Robust checkpoint:", robust_path, flush=True)
        return

    extraction = out / "_input_archive"
    if not extraction.exists() or not any(extraction.iterdir()):
        raise RuntimeError(
            f"Prepared input bundle not found at {extraction}. "
            "Run once with --prepare-only before launching workers."
        )

    config_path = first_match(extraction, ["nominal_config.json", "config.json"])
    nominal_path = first_match(
        extraction,
        [
            "nominal_frozen_checkpoint.pt",
            "frozen_best_checkpoint.pt",
            "best_full_checkpoint.pt",
            "best.pt",
        ],
    )
    robust_path = first_match(
        extraction,
        ["robust_final_model.pt", "robust_final_state_dict.pt"],
    )
    if config_path is None or nominal_path is None or robust_path is None:
        raise FileNotFoundError(
            "Need config + nominal + robust checkpoints. "
            f"config={config_path}, nominal={nominal_path}, robust={robust_path}"
        )

    DEVICE = init_device(rank)
    torch.cuda.manual_seed_all(SEED + rank)

    CFG = load_cfg(config_path)
    SYS = NMRSystem(CFG, DEVICE)

    nominal_model = PulseCompiler(CFG).to(DEVICE)
    robust_model = PulseCompiler(CFG).to(DEVICE)
    nominal_model.load_state_dict(
        extract_state(
            torch.load(
                nominal_path,
                map_location=DEVICE,
                weights_only=False,
            )
        ),
        strict=True,
    )
    robust_model.load_state_dict(
        extract_state(
            torch.load(
                robust_path,
                map_location=DEVICE,
                weights_only=False,
            )
        ),
        strict=True,
    )
    for model in [nominal_model, robust_model]:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    names, named, Q, targets = make_targets()
    ROBUST_SELECTION_SCENARIOS = cat_scen(
        sobol_scen(20, "routine", ROBUST_RF_ABS, SEED + 424242),
        boundaries(ROBUST_RF_ABS),
    )

    with torch.inference_mode():
        q = Q[0:1]
        un = nominal_model(q)
        ur = robust_model(q)
        fn = float(
            unitary_fidelity(
                targets[0:1],
                propagate_nominal(un),
            ).item()
        )
        fr = float(
            unitary_fidelity(
                targets[0:1],
                propagate_nominal(ur),
            ).item()
        )

    if not (np.isfinite(fn) and np.isfinite(fr)):
        raise RuntimeError("Non-finite self-check fidelity")

    rprint(
        rank,
        f"SELF CHECK: nominal F(I)={fn:.6f}, robust F(I)={fr:.6f}; "
        f"targets={len(names)}, world_size={world}",
    )

    if args.self_check_only:
        print(
            f"SELF CHECK PASSED on logical rank {rank}.",
            flush=True,
        )
        return

    if args.merge_only:
        if rank != 0:
            raise ValueError("--merge-only must be run with --rank 0")
        gate_records = load_all_gate_records(out, len(names))
        postprocess_rank0(
            out,
            input_zip,
            config_path,
            nominal_path,
            robust_path,
            nominal_model,
            robust_model,
            names,
            named,
            Q,
            targets,
            gate_records,
        )
        return

    local_indices = list(range(rank, len(names), world))
    rank_dir = out / "rank_results" / f"rank_{rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)

    rprint(rank, "assigned target indices", local_indices)

    for idx in local_indices:
        record_path = rank_dir / f"gate_{idx:03d}.pt"
        if record_path.exists():
            try:
                rec = torch.load(
                    record_path,
                    map_location="cpu",
                    weights_only=False,
                )
                if int(rec.get("index", -1)) == idx:
                    rprint(
                        rank,
                        f"SKIP existing completed gate {idx}: "
                        f"{rec.get('name', names[idx])}",
                    )
                    continue
            except Exception:
                rprint(
                    rank,
                    f"existing record for gate {idx} is unreadable; recomputing",
                )

        name = names[idx]
        target = targets[idx]
        q = Q[idx:idx + 1]
        rprint(rank, f"[{idx + 1}/{len(names)}] {name}")

        with torch.inference_mode():
            u_nom = nominal_model(q)[0]
            u_rob = robust_model(q)[0]

        f_nom = float(
            unitary_fidelity(
                target[None],
                propagate_nominal(u_nom[None]),
            ).item()
        )
        f_rob = float(
            unitary_fidelity(
                target[None],
                propagate_nominal(u_rob[None]),
            ).item()
        )

        rows = []
        pulses = {
            "Neural nominal": u_nom.detach(),
            "Neural robust": u_rob.detach(),
        }
        histories = {}

        for method, pulse, fidelity in [
            ("Neural nominal", u_nom, f_nom),
            ("Neural robust", u_rob, f_rob),
        ]:
            rows.append({
                "gate": name,
                "method": method,
                "fidelity": fidelity,
                "compile_time_s": np.nan,
                "restart": np.nan,
                "init_scale": np.nan,
                "best_iteration": np.nan,
                "optimization_score": np.nan,
                "lbfgs_used": False,
                **control_metrics(pulse),
            })

        rprint(rank, f"{name} — standalone GRAPE")
        rec = grape_best(
            target,
            SEED + 10000 * idx,
            robust=False,
            rank=rank,
        )
        pulse = rec["pulse"]
        pulses["GRAPE"] = pulse
        histories["GRAPE"] = rec["history"]
        rows.append({
            "gate": name,
            "method": "GRAPE",
            "fidelity": rec["nominal_fidelity"],
            "compile_time_s": rec["total_search_time_s"],
            "restart": rec["restart"],
            "init_scale": rec["init_scale"],
            "best_iteration": rec["best_iteration"],
            "optimization_score": rec["selection_score"],
            "lbfgs_used": rec["lbfgs_used"],
            **control_metrics(pulse),
        })

        rprint(rank, f"{name} — robust GRAPE")
        rec = grape_best(
            target,
            SEED + 20000 * idx,
            robust=True,
            rank=rank,
        )
        pulse = rec["pulse"]
        pulses["Robust GRAPE"] = pulse
        histories["Robust GRAPE"] = rec["history"]
        rows.append({
            "gate": name,
            "method": "Robust GRAPE",
            "fidelity": rec["nominal_fidelity"],
            "compile_time_s": rec["total_search_time_s"],
            "restart": rec["restart"],
            "init_scale": rec["init_scale"],
            "best_iteration": rec["best_iteration"],
            "optimization_score": rec["selection_score"],
            "robust_selection_mean_fidelity": rec["robust_mean_fidelity"],
            "robust_selection_cvar_infidelity": rec["cvar_infidelity"],
            "lbfgs_used": rec["lbfgs_used"],
            **control_metrics(pulse),
        })

        rprint(rank, f"{name} — NN-seeded GRAPE")
        rec = nn_seeded_grape(
            target,
            u_nom.detach(),
            rank=rank,
        )
        pulse = rec["pulse"]
        pulses["NN-seeded GRAPE"] = pulse
        histories["NN-seeded GRAPE"] = rec["history"]
        rows.append({
            "gate": name,
            "method": "NN-seeded GRAPE",
            "fidelity": rec["nominal_fidelity"],
            "compile_time_s": rec["elapsed_s"],
            "restart": np.nan,
            "init_scale": np.nan,
            "best_iteration": rec["best_iteration"],
            "optimization_score": rec["selection_score"],
            "lbfgs_used": rec["lbfgs_used"],
            **control_metrics(pulse),
        })

        rprint(rank, f"{name} — CRAB-SPSA")
        rec = crab_best(
            target,
            SEED + 30000 * idx,
            rank=rank,
        )
        pulse = rec["pulse"]
        pulses["CRAB-SPSA"] = pulse
        histories["CRAB-SPSA"] = rec["history"]
        rows.append({
            "gate": name,
            "method": "CRAB-SPSA",
            "fidelity": rec["fidelity"],
            "compile_time_s": rec["total_search_time_s"],
            "restart": rec["restart"],
            "init_scale": np.nan,
            "best_iteration": rec["best_iteration"],
            "optimization_score": rec["objective"],
            "lbfgs_used": False,
            **control_metrics(pulse),
        })

        save_gate_record(
            rank_dir,
            idx,
            name,
            rows,
            pulses,
            histories,
        )
        rprint(rank, f"saved gate {name}")

    (rank_dir / "COMPLETE").write_text("ok\n")
    rprint(
        rank,
        "WORKER COMPLETE — independent target shard finished.",
    )


if __name__ == "__main__":
    main()
