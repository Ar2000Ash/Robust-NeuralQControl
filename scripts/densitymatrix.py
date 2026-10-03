import os
import numpy as np
from numpy import kron
from scipy.linalg import expm

# ==========================================================
# SETTINGS
# ==========================================================

DIM = 8

I2 = np.eye(2, dtype=complex)

X = np.array([
    [0, 1],
    [1, 0]
], dtype=complex)

Y = np.array([
    [0, -1j],
    [1j, 0]
], dtype=complex)


def kron3(a, b, c):
    return kron(kron(a, b), c)


# ==========================================================
# TOMOGRAPHY READOUT ROTATIONS
# ==========================================================

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
    kron3(Uy, Uy, Uy)
]

LABELS = [
    "III",
    "XII",
    "IXI",
    "IIX",
    "IXX",
    "XXX",
    "YII",
    "IYI",
    "IIY",
    "YYI",
    "YYY"
]


# ==========================================================
# OBSERVED SINGLE-QUANTUM TRANSITIONS
# ==========================================================

INDEX_LIST = [
    (4, 0), (6, 2), (5, 1), (7, 3),   # Q1
    (2, 0), (6, 4), (3, 1), (7, 5),   # Q2
    (1, 0), (3, 2), (5, 4), (7, 6)    # Q3
]


# ==========================================================
# FIXED INTEGRATION WINDOWS
# ==========================================================

FREQ_WINDOWS = [
    (-1001, -930),
    (-866, -765),
    (-1072, -1001),
    (-930, -866),

    (-45, 40),
    (100, 200),
    (-153, -45),
    (40, 100),

    (721, 840),
    (647, 684),
    (684, 721),
    (571, 647)
]


# ==========================================================
# FIXED RECEIVER PHASES
# ==========================================================

PHASES = [
    4.593571789275965,
    4.729046676847299,
    3.8289005158044778
]


# ==========================================================
# 63-ELEMENT HERMITIAN TRACELESS BASIS
# ==========================================================

def hermitian_basis():

    basis = []

    # 7 diagonal traceless matrices
    for i in range(DIM - 1):

        M = np.zeros((DIM, DIM), dtype=complex)

        M[i, i] = 1
        M[DIM - 1, DIM - 1] = -1

        basis.append(M)

    # 28 real-symmetric + 28 imaginary-antisymmetric
    for i in range(DIM):

        for j in range(i + 1, DIM):

            # Real part
            M = np.zeros((DIM, DIM), dtype=complex)

            M[i, j] = 1
            M[j, i] = 1

            basis.append(M)

            # Imaginary part
            M = np.zeros((DIM, DIM), dtype=complex)

            M[i, j] = -1j
            M[j, i] = 1j

            basis.append(M)

    return basis


BASIS = hermitian_basis()


# ==========================================================
# BUILD TOMOGRAPHY MEASUREMENT MATRIX
# ==========================================================

def build_measurement_matrix():

    A = []

    for U in U_LIST:

        for i, j in INDEX_LIST:

            row_real = []
            row_imag = []

            for B in BASIS:

                rho_rot = U @ B @ U.conj().T

                value = rho_rot[i, j]

                row_real.append(value.real)
                row_imag.append(value.imag)

            A.append(row_real)
            A.append(row_imag)

    return np.asarray(A)


A = build_measurement_matrix()

print("Measurement matrix shape:", A.shape)
# expected: (264, 63)


# ==========================================================
# LOAD R / I CSV
# ==========================================================

def load_data(filename):

    data = np.loadtxt(
        filename,
        delimiter=",",
        skiprows=1
    )

    # Sort frequency axis
    data = data[np.argsort(data[:, 0])]

    freq = data[:, 0]
    signal = data[:, 1]

    return freq, signal


# ==========================================================
# FIXED-WINDOW INTEGRATION
# ==========================================================

def integrate_window(freq_range, freq, spectrum):

    dense_freq = np.linspace(
        freq_range[0],
        freq_range[1],
        1000
    )

    interp_signal = np.interp(
        dense_freq,
        freq,
        spectrum
    )

    # Average signal over window
    return (
        np.trapezoid(interp_signal, dense_freq)
        /
        (freq_range[1] - freq_range[0])
    )


def integrate_complex(freq_range, Rf, If, Rs, Is):

    real = integrate_window(
        freq_range,
        Rf,
        Rs
    )

    imag = integrate_window(
        freq_range,
        If,
        Is
    )

    return real + 1j * imag


# ==========================================================
# DEVIATION-DENSITY-MATRIX RECONSTRUCTION
# ==========================================================

def reconstruct_deviation_matrix(folder):

    F = []

    for label in LABELS:

        # Example filenames:
        # III,R.csv
        # III,I.csv

        Rf, Rs = load_data(
            os.path.join(
                folder,
                label + ",R.csv"
            )
        )

        If, Is = load_data(
            os.path.join(
                folder,
                label + ",I.csv"
            )
        )

        # --------------------------------------
        # Integrate 12 complex transition peaks
        # --------------------------------------

        peaks = []

        for window in FREQ_WINDOWS:

            value = integrate_complex(
                window,
                Rf,
                If,
                Rs,
                Is
            )

            peaks.append(value)

        peaks = np.asarray(
            peaks,
            dtype=complex
        )

        # --------------------------------------
        # Fixed receiver-phase correction
        # --------------------------------------

        peaks[0:4] *= np.exp(
            -1j * PHASES[0]
        )

        peaks[4:8] *= np.exp(
            -1j * PHASES[1]
        )

        peaks[8:12] *= np.exp(
            -1j * PHASES[2]
        )

        # --------------------------------------
        # Append real / imaginary measurements
        # --------------------------------------

        for value in peaks:

            F.append(value.real)
            F.append(value.imag)

    # Same global scale used in original tomography
    F = np.asarray(F) / 3800.0

    # --------------------------------------
    # Linear least-squares inversion
    # --------------------------------------

    r, *_ = np.linalg.lstsq(
        A,
        F,
        rcond=None
    )

    # --------------------------------------
    # Construct traceless deviation matrix
    # --------------------------------------

    D = np.zeros(
        (DIM, DIM),
        dtype=complex
    )

    for coefficient, B in zip(r, BASIS):

        D += coefficient * B

    return D


# ==========================================================
# NORMALIZED HILBERT-SCHMIDT CORRELATION
# ==========================================================

def hs_corr(D1, D2):

    numerator = np.real(
        np.trace(D1 @ D2)
    )

    denominator = np.sqrt(
        np.real(np.trace(D1 @ D1))
        *
        np.real(np.trace(D2 @ D2))
    )

    return numerator / denominator


# ==========================================================
# IDEAL PPS DEVIATION MATRIX
# ==========================================================

psi0 = np.zeros(
    (8, 1),
    dtype=complex
)

psi0[0, 0] = 1.0

rho_pps_ideal = (
    psi0 @ psi0.conj().T
)

D_PPS_IDEAL = (
    rho_pps_ideal
    -
    np.eye(8) / 8
)


# ==========================================================
# IDEAL HADAMARD DEVIATION MATRIX
# ==========================================================

H = (
    np.array([
        [1, 1],
        [1, -1]
    ], dtype=complex)
    /
    np.sqrt(2)
)

U_H = kron3(
    H,
    I2,
    I2
)

psi_H = U_H @ psi0

rho_H_ideal = (
    psi_H @ psi_H.conj().T
)

D_H_IDEAL = (
    rho_H_ideal
    -
    np.eye(8) / 8
)


# ==========================================================
# EXAMPLE USAGE
# ==========================================================

# Change these paths to your folders


# The functions above are imported by the tomography analysis.
# Example dataset paths are intentionally omitted from the publication version.