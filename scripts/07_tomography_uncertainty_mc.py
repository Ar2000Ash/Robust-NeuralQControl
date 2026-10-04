#!/usr/bin/env python3
"""Standalone Monte-Carlo uncertainty analysis for NMR tomography.

This script preserves the uncertainty model used in the intermediate
tomography analysis:

1. independent pointwise spectral-amplitude perturbations
       S_i -> S_i * (1 + delta_i), delta_i ~ Uniform[-0.02,+0.02]
2. a smooth linear frequency drift across the 11 sequential readouts, with
   independently sampled start/end offsets in Uniform[-5,+5] Hz.

The central reconstruction is identical to scripts/06_reconstruct_tomography.py:
11 readout operations, 12 transition windows, the same receiver phases,
the /3800 scale, 63-parameter Hermitian-traceless inversion, and normalized
Hilbert-Schmidt correlation.

For computational efficiency, the pointwise amplitude model is propagated
analytically to the exact first two moments of the integrated transition
amplitudes and sampled with a moment-matched multivariate Gaussian. Frequency
drift is propagated explicitly on the same 0.1-Hz shift grid.

This standalone table is retained as a reproducibility diagnostic. The final
manuscript uncertainty intervals also include the later systematic/effective
uncertainty analysis and are reproduced by scripts/08 and 09.
"""

import argparse
import zipfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from numpy import kron
from scipy.linalg import expm

# ============================================================
# DEFAULT UNCERTAINTY SETTINGS
# ============================================================

DEFAULT_DATA = Path("data/experimental/digitized_spectra.zip")
DEFAULT_OUTPUT = Path("results/tomography/tomography_uncertainty_mc.csv")
DEFAULT_DIFFERENCE_OUTPUT = Path(
    "results/tomography/hardware_difference_uncertainty_mc.csv"
)

N_MC = 10_000
AMP_FRAC = 0.02
SHIFT_MAX_HZ = 5.0
SHIFT_GRID_STEP_HZ = 0.1
SEED = 12345

# ============================================================
# TOMOGRAPHY DEFINITIONS
# ============================================================

DIM = 8
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)

def kron3(a, b, c):
    return kron(kron(a, b), c)

Ux = expm(-1j * X * np.pi / 4)
Uy = expm(-1j * Y * np.pi / 4)

U_LIST = [
    np.eye(8),
    kron3(Ux, I2, I2),
    kron3(I2, Ux, I2),
    kron3(I2, I2, Ux),
    kron3(I2, Ux, Ux),
    kron3(Ux, Ux, Ux),
    kron3(Uy, I2, I2),
    kron3(I2, Uy, I2),
    kron3(I2, I2, Uy),
    kron3(Uy, Uy, I2),
    kron3(Uy, Uy, Uy),
]

LABELS = ["III", "XII", "IXI", "IIX", "IXX", "XXX",
          "YII", "IYI", "IIY", "YYI", "YYY"]

INDEX_LIST = [
    (4,0), (6,2), (5,1), (7,3),
    (2,0), (6,4), (3,1), (7,5),
    (1,0), (3,2), (5,4), (7,6),
]

FREQ_WINDOWS = [
    (-1001,-930), (-866,-765), (-1072,-1001), (-930,-866),
    (-45,40), (100,200), (-153,-45), (40,100),
    (721,840), (647,684), (684,721), (571,647),
]

PHASES = np.array([
    4.593571789275965,
    4.729046676847299,
    3.8289005158044778,
])

PHASE_VECTOR = np.array(
    [np.exp(-1j*PHASES[0])] * 4
    + [np.exp(-1j*PHASES[1])] * 4
    + [np.exp(-1j*PHASES[2])] * 4
)

# ============================================================
# 63-ELEMENT HERMITIAN TRACELESS BASIS
# ============================================================

def hermitian_basis():
    out = []
    for i in range(DIM - 1):
        M = np.zeros((DIM, DIM), dtype=complex)
        M[i, i] = 1
        M[DIM - 1, DIM - 1] = -1
        out.append(M)

    for i in range(DIM):
        for j in range(i + 1, DIM):
            M = np.zeros((DIM, DIM), dtype=complex)
            M[i, j] = 1
            M[j, i] = 1
            out.append(M)

            M = np.zeros((DIM, DIM), dtype=complex)
            M[i, j] = -1j
            M[j, i] = 1j
            out.append(M)
    return out

BASIS = hermitian_basis()

def build_measurement_matrix():
    A = []
    for U in U_LIST:
        for i, j in INDEX_LIST:
            rr, ii = [], []
            for B in BASIS:
                R = U @ B @ U.conj().T
                rr.append(R[i, j].real)
                ii.append(R[i, j].imag)
            A.extend([rr, ii])
    return np.asarray(A, dtype=float)

A = build_measurement_matrix()
PINV_A = np.linalg.pinv(A)

# Gram matrix for fast HS evaluation in coefficient space
G = np.empty((63, 63), dtype=float)
for i, Bi in enumerate(BASIS):
    for j, Bj in enumerate(BASIS):
        G[i, j] = np.real(np.trace(Bi @ Bj))

# ============================================================
# IDEAL DEVIATION TARGETS
# ============================================================

psi0 = np.zeros((8, 1), dtype=complex)
psi0[0, 0] = 1.0

D_PPS = psi0 @ psi0.conj().T - np.eye(8) / 8

H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
UH = kron3(H, I2, I2)
psiH = UH @ psi0
D_H = psiH @ psiH.conj().T - np.eye(8) / 8

def target_projection(D):
    q = np.array([np.real(np.trace(B @ D)) for B in BASIS])
    norm = np.real(np.trace(D @ D))
    return q, norm

Q_PPS, NORM_PPS = target_projection(D_PPS)
Q_H, NORM_H = target_projection(D_H)

# ============================================================
# FILE / SPECTRUM UTILITIES
# ============================================================

def find_data_root(extract_dir):
    candidates = list(extract_dir.rglob("PPS"))
    for p in candidates:
        if p.is_dir() and (p / "III,R.csv").exists():
            return p.parent
    raise FileNotFoundError("Could not find PPS/III,R.csv after extraction.")

def load_spectrum(path):
    d = np.loadtxt(path, delimiter=",", skiprows=1)
    d = d[np.argsort(d[:, 0])]
    return d[:, 0], d[:, 1]

# ============================================================
# EXACT LINEAR WEIGHTS FOR THE EXISTING INTEGRATOR
# ============================================================

def integration_weights(freq, window, shift_hz=0.0, n_dense=1000):
    """
    Returns W such that W @ spectrum equals the existing
    1000-point interpolation + normalized trapezoidal integral.

    A positive shift moves the displayed spectrum to higher frequency:
        S_shift(f) = S_original(f - shift).
    """
    query = np.linspace(window[0], window[1], n_dense) - shift_hz

    # Normalized trapezoid weights. Because the dense grid is uniform,
    # division by window width reduces to these coefficients.
    tw = np.ones(n_dense, dtype=float) / (n_dense - 1)
    tw[0] *= 0.5
    tw[-1] *= 0.5

    W = np.zeros(len(freq), dtype=float)

    # Match np.interp endpoint behavior exactly: constant extrapolation.
    left = query <= freq[0]
    right = query >= freq[-1]
    mid = ~(left | right)

    if np.any(left):
        W[0] += tw[left].sum()
    if np.any(right):
        W[-1] += tw[right].sum()

    if np.any(mid):
        q = query[mid]
        wt = tw[mid]

        idx = np.searchsorted(freq, q, side="left")
        x0 = freq[idx - 1]
        x1 = freq[idx]
        a = (q - x0) / (x1 - x0)

        np.add.at(W, idx - 1, wt * (1 - a))
        np.add.at(W, idx, wt * a)

    return W

def covariance_root(C):
    """
    Symmetric square root L with C = L L^T.
    More robust than Cholesky for nearly singular adjacent-window covariances.
    """
    vals, vecs = np.linalg.eigh((C + C.T) / 2)
    vals = np.clip(vals, 0.0, None)
    return vecs @ np.diag(np.sqrt(vals))

SHIFT_GRID = np.arange(
    -SHIFT_MAX_HZ,
    SHIFT_MAX_HZ + 0.5 * SHIFT_GRID_STEP_HZ,
    SHIFT_GRID_STEP_HZ
)
ZERO_SHIFT_INDEX = int(np.argmin(np.abs(SHIFT_GRID)))

def precompute_condition(folder):
    """
    For every readout/quadrature and every frequency shift:
      mean[k] = integrated amplitudes
      root[k] = square-root covariance generated by pointwise +/-2% noise
    """
    readouts = []

    for label in LABELS:
        entry = {}

        for comp in ("R", "I"):
            freq, y = load_spectrum(folder / f"{label},{comp}.csv")

            means = []
            roots = []

            for shift in SHIFT_GRID:
                W = np.stack([
                    integration_weights(freq, window, shift)
                    for window in FREQ_WINDOWS
                ])

                mean = W @ y

                # delta_i ~ Uniform[-a,+a] => Var(delta_i)=a^2/3.
                # Integrated noise = W diag(y) delta.
                Wy = W * y[None, :]
                C = (AMP_FRAC**2 / 3.0) * (Wy @ Wy.T)

                means.append(mean)
                roots.append(covariance_root(C))

            entry[comp] = (np.stack(means), np.stack(roots))

        readouts.append(entry)

    return readouts

# ============================================================
# FAST HS CORRELATION
# ============================================================

def correlations_from_F(F, q_target, target_norm):
    r = F @ PINV_A.T
    numerator = r @ q_target
    reconstructed_norm = np.einsum("bi,ij,bj->b", r, G, r)
    return numerator / np.sqrt(np.maximum(reconstructed_norm, 0.0) * target_norm)

def nominal_correlation(precomputed, q_target, target_norm):
    F = []

    for entry in precomputed:
        R = entry["R"][0][ZERO_SHIFT_INDEX]
        I = entry["I"][0][ZERO_SHIFT_INDEX]
        peaks = (R + 1j * I) * PHASE_VECTOR

        for z in peaks:
            F.extend([z.real, z.imag])

    F = np.asarray(F, dtype=float)[None, :] / 3800.0
    return correlations_from_F(F, q_target, target_norm)[0]

# ============================================================
# MONTE CARLO
# ============================================================

def draw_shift_indices(rng, n_mc, n_readouts, model="linear"):
    """
    Frequency-position model.

    linear:
        Draw start/end offsets independently from U[-5,+5] Hz and
        linearly interpolate across the 11 sequential readouts.

    common:
        One U[-5,+5] Hz offset shared by all 11 readouts.

    independent:
        Separate U[-5,+5] Hz offset for every readout.
        This is available only as a deliberately aggressive sensitivity test.
    """
    if model == "linear":
        start = rng.uniform(-SHIFT_MAX_HZ, SHIFT_MAX_HZ, size=n_mc)
        end = rng.uniform(-SHIFT_MAX_HZ, SHIFT_MAX_HZ, size=n_mc)

        shifts = np.empty((n_mc, n_readouts), dtype=float)

        for j in range(n_readouts):
            t = j / (n_readouts - 1)
            shifts[:, j] = (1.0 - t) * start + t * end

    elif model == "common":
        s = rng.uniform(-SHIFT_MAX_HZ, SHIFT_MAX_HZ, size=n_mc)
        shifts = np.repeat(s[:, None], n_readouts, axis=1)

    elif model == "independent":
        shifts = rng.uniform(
            -SHIFT_MAX_HZ,
            SHIFT_MAX_HZ,
            size=(n_mc, n_readouts)
        )

    else:
        raise ValueError("model must be 'linear', 'common', or 'independent'")

    # Quantize to the 0.1-Hz precomputed grid.
    idx = np.rint(
        (shifts + SHIFT_MAX_HZ) / SHIFT_GRID_STEP_HZ
    ).astype(int)

    return np.clip(idx, 0, len(SHIFT_GRID) - 1)


def monte_carlo(precomputed, q_target, target_norm,
                n_mc=N_MC, seed=SEED, mode="combined",
                drift_model="linear"):
    """
    mode:
      "amplitude" : +/-2% pointwise spectral uncertainty only
      "shift"     : frequency drift only
      "combined"  : both effects

    The default drift_model="linear" is the recommended model for the paper.
    """
    rng = np.random.default_rng(seed)
    F = np.empty((n_mc, 264), dtype=float)

    if mode in ("shift", "combined"):
        all_shift_idx = draw_shift_indices(
            rng, n_mc, len(precomputed), model=drift_model
        )
    else:
        all_shift_idx = np.full(
            (n_mc, len(precomputed)),
            ZERO_SHIFT_INDEX,
            dtype=int
        )

    col = 0

    for readout_index, entry in enumerate(precomputed):
        shift_idx = all_shift_idx[:, readout_index]

        mean_R = entry["R"][0][shift_idx]
        mean_I = entry["I"][0][shift_idx]

        if mode in ("amplitude", "combined"):
            root_R = entry["R"][1][shift_idx]
            root_I = entry["I"][1][shift_idx]

            zR = rng.standard_normal((n_mc, 12))
            zI = rng.standard_normal((n_mc, 12))

            R = mean_R + np.einsum("bij,bj->bi", root_R, zR)
            I = mean_I + np.einsum("bij,bj->bi", root_I, zI)
        else:
            R, I = mean_R, mean_I

        peaks = (R + 1j * I) * PHASE_VECTOR[None, :]

        for k in range(12):
            F[:, col] = peaks[:, k].real / 3800.0
            F[:, col + 1] = peaks[:, k].imag / 3800.0
            col += 2

    return correlations_from_F(F, q_target, target_norm)


def summarize(x):
    return {
        "mean": float(np.mean(x)),
        "variance": float(np.var(x, ddof=1)),
        "sd": float(np.std(x, ddof=1)),
        "q2.5": float(np.quantile(x, 0.025)),
        "q97.5": float(np.quantile(x, 0.975)),
    }

# ============================================================
# COMMAND LINE
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="Digitized spectra ZIP archive",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Standalone Monte-Carlo condition summary",
    )
    parser.add_argument(
        "--difference-output",
        type=Path,
        default=DEFAULT_DIFFERENCE_OUTPUT,
        help="Standalone robust-minus-nominal difference summary",
    )
    parser.add_argument(
        "--n-mc",
        type=int,
        default=N_MC,
        help="Number of Monte-Carlo samples per uncertainty mode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Base random seed",
    )
    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()
    data_path = args.data.resolve()
    output_path = args.output.resolve()
    difference_output_path = args.difference_output.resolve()

    if not data_path.exists():
        raise FileNotFoundError(f"Digitized spectra archive not found: {data_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    difference_output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        with zipfile.ZipFile(data_path, "r") as zf:
            zf.extractall(td)

        root = find_data_root(td)

        conditions = [
            "PPS",
            "NM10", "NM5", "N0", "NP5", "NP10",
            "RM10", "RM5", "R0", "RP5", "RP10",
        ]

        rows = []
        samples = {}

        for ci, condition in enumerate(conditions):
            folder = root / condition
            pre = precompute_condition(folder)

            if condition == "PPS":
                q_target, target_norm = Q_PPS, NORM_PPS
            else:
                q_target, target_norm = Q_H, NORM_H

            nominal = nominal_correlation(pre, q_target, target_norm)

            row = {
                "condition": condition,
                "nominal_C_HS": nominal,
            }

            for mi, mode in enumerate(("amplitude", "shift", "combined")):
                x = monte_carlo(
                    pre,
                    q_target,
                    target_norm,
                    n_mc=args.n_mc,
                    seed=args.seed + 100 * ci + mi + 1,
                    mode=mode,
                    drift_model="linear",
                )

                samples[(condition, mode)] = x
                s = summarize(x)

                for key, value in s.items():
                    row[f"{mode}_{key}"] = value

            rows.append(row)

        results = pd.DataFrame(rows)
        results.to_csv(output_path, index=False)

        # Robust-minus-nominal differences
        pairs = [
            (-10, "NM10", "RM10"),
            (-5,  "NM5",  "RM5"),
            (0,   "N0",   "R0"),
            (5,   "NP5",  "RP5"),
            (10,  "NP10", "RP10"),
        ]

        diff_rows = []
        for eps, ncond, rcond in pairs:
            d = samples[(rcond, "combined")] - samples[(ncond, "combined")]

            rn = results.loc[results.condition == ncond].iloc[0]
            rr = results.loc[results.condition == rcond].iloc[0]

            diff_rows.append({
                "epsilon_RF_percent": eps,
                "nominal_C_HS": rn["nominal_C_HS"],
                "nominal_combined_sd": rn["combined_sd"],
                "nominal_95_low": rn["combined_q2.5"],
                "nominal_95_high": rn["combined_q97.5"],
                "robust_C_HS": rr["nominal_C_HS"],
                "robust_combined_sd": rr["combined_sd"],
                "robust_95_low": rr["combined_q2.5"],
                "robust_95_high": rr["combined_q97.5"],
                "robust_minus_nominal_central":
                    rr["nominal_C_HS"] - rn["nominal_C_HS"],
                "difference_MC_mean": np.mean(d),
                "difference_MC_sd": np.std(d, ddof=1),
                "difference_95_low": np.quantile(d, 0.025),
                "difference_95_high": np.quantile(d, 0.975),
            })

        diffs = pd.DataFrame(diff_rows)
        diffs.to_csv(difference_output_path, index=False)

        print("\n=== Reconstruction uncertainty ===")
        print(results.to_string(index=False))

        print("\n=== Robust - nominal differences ===")
        print(diffs.to_string(index=False))

        print("\nMonte-Carlo settings:")
        print(f"N_MC = {args.n_mc}")
        print(f"pointwise spectral perturbation = +/- {100*AMP_FRAC:.1f}%")
        print(f"frequency drift endpoints = +/- {SHIFT_MAX_HZ:.1f} Hz")
        print(f"frequency-shift grid = {SHIFT_GRID_STEP_HZ:.1f} Hz")
        print(f"standalone MC table = {output_path}")
        print(f"difference table = {difference_output_path}")


if __name__ == "__main__":
    main()
