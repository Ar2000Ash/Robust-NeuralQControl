#!/usr/bin/env python3
"""
Systematic-sensitivity analysis for the 3-qubit NMR deviation-matrix tomography.

This script DOES NOT turn systematic effects into a statistical error bar.
It keeps the standard reconstruction as the central result and asks how much
C_HS changes under several alternative analyses / diagnostics:

1. Baseline treatment:
   - no subtraction (central)
   - linear baseline fit in signal-free regions
   - quadratic baseline fit in signal-free regions

2. Receiver-phase sensitivity:
   - independently perturb each of the three frozen receiver phases by
     -1, 0, or +1 degree and report the min/max over all 27 combinations.
   - a +/-3 degree diagnostic is also calculated.

3. Integration-window placement sensitivity:
   - move ONE integration-window edge at a time by one digitized frequency bin
     (~2.56 Hz) and report the min/max C_HS.

4. Central-Q3 overlap leverage:
   - remove transition window [647,684] Hz from ALL 11 readouts
   - remove transition window [684,721] Hz from ALL 11 readouts
   - remove both central Q3 transitions
   The reduced tomography matrix remains full rank (63) in all three cases.
   These tests are diagnostics of how strongly the unresolved Q3 information
   pulls the reconstruction; they are NOT corrections and NOT confidence
   intervals.

Central tomography is exactly the same fixed-window, fixed-phase, linear
deviation-density-matrix reconstruction used in the manuscript.
"""

from pathlib import Path
import zipfile, tempfile, itertools
import numpy as np
import pandas as pd
from scipy.linalg import expm

ZIP_PATH = Path("data/experimental/digitized_spectra.zip")
OUT_FULL = Path("tomography_systematic_sensitivity_full.csv")
OUT_COMPACT = Path("tomography_systematic_sensitivity_compact.csv")

CONDITIONS = ["PPS","NM10","NM5","N0","NP5","NP10",
              "RM10","RM5","R0","RP5","RP10"]

LABELS = ["III","XII","IXI","IIX","IXX","XXX",
          "YII","IYI","IIY","YYI","YYY"]

FREQ_WINDOWS = [
    (-1001,-930),(-866,-765),(-1072,-1001),(-930,-866),
    (-45,40),(100,200),(-153,-45),(40,100),
    (721,840),(647,684),(684,721),(571,647)
]

PHASES = np.array([
    4.593571789275965,
    4.729046676847299,
    3.8289005158044778
])

NOISE_REGIONS = [(-700,-250),(300,500),(900,1100)]

DIM = 8
I2 = np.eye(2,dtype=complex)
X = np.array([[0,1],[1,0]],dtype=complex)
Y = np.array([[0,-1j],[1j,0]],dtype=complex)

def kron3(a,b,c):
    return np.kron(np.kron(a,b),c)

Ux = expm(-1j*X*np.pi/4)
Uy = expm(-1j*Y*np.pi/4)

U_LIST = [
    np.eye(8),
    kron3(Ux,I2,I2),
    kron3(I2,Ux,I2),
    kron3(I2,I2,Ux),
    kron3(I2,Ux,Ux),
    kron3(Ux,Ux,Ux),
    kron3(Uy,I2,I2),
    kron3(I2,Uy,I2),
    kron3(I2,I2,Uy),
    kron3(Uy,Uy,I2),
    kron3(Uy,Uy,Uy)
]

INDEX_LIST = [
    (4,0),(6,2),(5,1),(7,3),
    (2,0),(6,4),(3,1),(7,5),
    (1,0),(3,2),(5,4),(7,6)
]

def build_basis():
    basis=[]
    for i in range(DIM-1):
        M=np.zeros((DIM,DIM),complex)
        M[i,i]=1
        M[-1,-1]=-1
        basis.append(M)
    for i in range(DIM):
        for j in range(i+1,DIM):
            M=np.zeros((DIM,DIM),complex)
            M[i,j]=1
            M[j,i]=1
            basis.append(M)

            M=np.zeros((DIM,DIM),complex)
            M[i,j]=-1j
            M[j,i]=1j
            basis.append(M)
    return basis

BASIS=build_basis()

def build_A():
    A=[]
    for U in U_LIST:
        for i,j in INDEX_LIST:
            rr=[]
            ii=[]
            for B in BASIS:
                D=U@B@U.conj().T
                rr.append(D[i,j].real)
                ii.append(D[i,j].imag)
            A.extend([rr,ii])
    return np.asarray(A,float)

A=build_A()

psi0=np.zeros((8,1),complex)
psi0[0,0]=1
D_PPS=psi0@psi0.conj().T-np.eye(8)/8

H=np.array([[1,1],[1,-1]],complex)/np.sqrt(2)
psiH=kron3(H,I2,I2)@psi0
D_H=psiH@psiH.conj().T-np.eye(8)/8

def hs_corr(D1,D2):
    return float(
        np.real(np.trace(D1@D2)) /
        np.sqrt(np.real(np.trace(D1@D1))*np.real(np.trace(D2@D2)))
    )

def find_root(td):
    for p in td.rglob("PPS"):
        if p.is_dir() and (p/"III,R.csv").exists():
            return p.parent
    raise FileNotFoundError("PPS/III,R.csv not found")

def load_all(root):
    data={}
    for cond in CONDITIONS:
        data[cond]={}
        for lab in LABELS:
            data[cond][lab]={}
            for comp in ("R","I"):
                a=np.loadtxt(root/cond/f"{lab},{comp}.csv",
                             delimiter=",",skiprows=1)
                a=a[np.argsort(a[:,0])]
                data[cond][lab][comp]=(a[:,0],a[:,1])
    return data

def integrate_window(window,freq,spec):
    dense=np.linspace(window[0],window[1],1000)
    interp=np.interp(dense,freq,spec)
    return np.trapezoid(interp,dense)/(window[1]-window[0])

def remove_baseline(freq,y,degree):
    mask=np.zeros(freq.shape,dtype=bool)
    for lo,hi in NOISE_REGIONS:
        mask |= (freq>=lo)&(freq<=hi)
    coef=np.polyfit(freq[mask],y[mask],degree)
    return y-np.polyval(coef,freq)

def make_F(data,cond,windows=FREQ_WINDOWS,
           phase_offsets_deg=(0,0,0),baseline_degree=None):
    corrected_phases=PHASES+np.deg2rad(np.asarray(phase_offsets_deg))
    pvec=np.array(
        [np.exp(-1j*corrected_phases[0])]*4+
        [np.exp(-1j*corrected_phases[1])]*4+
        [np.exp(-1j*corrected_phases[2])]*4
    )

    F=[]
    for lab in LABELS:
        fr,R=data[cond][lab]["R"]
        fi,I=data[cond][lab]["I"]

        if baseline_degree is not None:
            R=remove_baseline(fr,R,baseline_degree)
            I=remove_baseline(fi,I,baseline_degree)

        peaks=np.array([
            integrate_window(w,fr,R)+1j*integrate_window(w,fi,I)
            for w in windows
        ])
        peaks *= pvec

        for z in peaks:
            F.extend([z.real,z.imag])

    return np.asarray(F,float)/3800.0

def reconstruct(F,Ause=A):
    r,*_=np.linalg.lstsq(Ause,F,rcond=None)
    D=np.zeros((8,8),complex)
    for c,B in zip(r,BASIS):
        D += c*B
    return D

def correlation(cond,F,Ause=A):
    D=reconstruct(F,Ause)
    target=D_PPS if cond=="PPS" else D_H
    return hs_corr(D,target)

def keep_without_transitions(remove):
    bad=set()
    for readout in range(11):
        for t in remove:
            bad.update([24*readout+2*t,24*readout+2*t+1])
    return np.array([i for i in range(264) if i not in bad])

def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp)
        with zipfile.ZipFile(ZIP_PATH) as z:
            z.extractall(tmp)
        root=find_root(tmp)
        data=load_all(root)

        # One digitized frequency bin, used for the window-edge sensitivity.
        f0,_=data["PPS"]["III"]["R"]
        bin_hz=float(np.median(np.diff(f0)))

        # Q3 central transition subsets.
        keep_q3a=keep_without_transitions([9])      # [647,684] Hz
        keep_q3b=keep_without_transitions([10])     # [684,721] Hz
        keep_q3both=keep_without_transitions([9,10])

        assert np.linalg.matrix_rank(A[keep_q3a])==63
        assert np.linalg.matrix_rank(A[keep_q3b])==63
        assert np.linalg.matrix_rank(A[keep_q3both])==63

        rows=[]

        for cond in CONDITIONS:
            F0=make_F(data,cond)
            C0=correlation(cond,F0)

            # Baseline model sensitivity
            C_bl1=correlation(cond,make_F(data,cond,baseline_degree=1))
            C_bl2=correlation(cond,make_F(data,cond,baseline_degree=2))

            # Receiver phase +/-1 degree and +/-3 degree
            def phase_range(d):
                vals=[]
                for offs in itertools.product([-d,0,d],repeat=3):
                    vals.append(correlation(
                        cond,make_F(data,cond,phase_offsets_deg=offs)
                    ))
                return min(vals),max(vals)

            p1min,p1max=phase_range(1)
            p3min,p3max=phase_range(3)

            # Move one window edge at a time by one digitized bin.
            wvals=[]
            for wi in range(12):
                for edge in (0,1):
                    for sign in (-1,1):
                        wins=[list(w) for w in FREQ_WINDOWS]
                        wins[wi][edge] += sign*bin_hz
                        if wins[wi][0] >= wins[wi][1]:
                            continue
                        wins=[tuple(w) for w in wins]
                        wvals.append(correlation(
                            cond,make_F(data,cond,windows=wins)
                        ))
            wmin,wmax=min(wvals),max(wvals)

            # Q3 overlap leverage diagnostic
            C_q3a=correlation(cond,F0[keep_q3a],A[keep_q3a])
            C_q3b=correlation(cond,F0[keep_q3b],A[keep_q3b])
            C_q3both=correlation(cond,F0[keep_q3both],A[keep_q3both])

            routine_values=[C0,C_bl1,C_bl2,p1min,p1max,wmin,wmax]
            q3_values=[C0,C_q3a,C_q3b,C_q3both]

            rows.append({
                "condition":cond,
                "central_C_HS":C0,

                "baseline_linear_C":C_bl1,
                "baseline_linear_delta":C_bl1-C0,
                "baseline_quadratic_C":C_bl2,
                "baseline_quadratic_delta":C_bl2-C0,

                "phase_1deg_min_C":p1min,
                "phase_1deg_max_C":p1max,
                "phase_1deg_delta_min":p1min-C0,
                "phase_1deg_delta_max":p1max-C0,

                "phase_3deg_min_C":p3min,
                "phase_3deg_max_C":p3max,
                "phase_3deg_delta_min":p3min-C0,
                "phase_3deg_delta_max":p3max-C0,

                "window_one_bin_Hz":bin_hz,
                "window_one_bin_min_C":wmin,
                "window_one_bin_max_C":wmax,
                "window_one_bin_delta_min":wmin-C0,
                "window_one_bin_delta_max":wmax-C0,

                "drop_Q3_647_684_C":C_q3a,
                "drop_Q3_647_684_delta":C_q3a-C0,
                "drop_Q3_684_721_C":C_q3b,
                "drop_Q3_684_721_delta":C_q3b-C0,
                "drop_both_Q3_central_C":C_q3both,
                "drop_both_Q3_central_delta":C_q3both-C0,

                "routine_sensitivity_min_C":min(routine_values),
                "routine_sensitivity_max_C":max(routine_values),
                "routine_delta_min":min(routine_values)-C0,
                "routine_delta_max":max(routine_values)-C0,

                "Q3_diagnostic_min_C":min(q3_values),
                "Q3_diagnostic_max_C":max(q3_values),

                "deficit_from_unity":1-C0,
            })

        df=pd.DataFrame(rows)
        df.to_csv(OUT_FULL,index=False)

        compact=df[[
            "condition","central_C_HS",
            "routine_sensitivity_min_C","routine_sensitivity_max_C",
            "Q3_diagnostic_min_C","Q3_diagnostic_max_C",
            "deficit_from_unity"
        ]].copy()
        compact.to_csv(OUT_COMPACT,index=False)

        print("\nSystematic-sensitivity summary")
        print(compact.to_string(index=False))
        print("\nOne digitized frequency bin =",bin_hz,"Hz")
        print("\nNOTE: these ranges are sensitivity diagnostics, not statistical confidence intervals.")

if __name__=="__main__":
    main()