import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from numpy import kron
from scipy.linalg import expm

parser = argparse.ArgumentParser(description='Reconstruct deviation density matrices from digitized NMR spectra.')
parser.add_argument('--data-dir', type=Path, default=Path('data/experimental/digitized_csv'), help='Directory containing PPS/NM10/... tomography folders')
parser.add_argument('--output', type=Path, default=Path('outputs/tomography/experimental_HS_results.csv'))
args = parser.parse_args()
BASE = str(args.data_dir.resolve()) + os.sep

DIM = 8
I2 = np.eye(2, dtype=complex)
X = np.array([[0,1],[1,0]], dtype=complex)
Y = np.array([[0,-1j],[1j,0]], dtype=complex)

def kron3(a,b,c): return kron(kron(a,b),c)
Ux = expm(-1j*X*np.pi/4)
Uy = expm(-1j*Y*np.pi/4)
U_LIST = [
    np.eye(8), kron3(Ux,I2,I2), kron3(I2,Ux,I2), kron3(I2,I2,Ux),
    kron3(I2,Ux,Ux), kron3(Ux,Ux,Ux), kron3(Uy,I2,I2),
    kron3(I2,Uy,I2), kron3(I2,I2,Uy), kron3(Uy,Uy,I2), kron3(Uy,Uy,Uy)
]
LABELS = ['III','XII','IXI','IIX','IXX','XXX','YII','IYI','IIY','YYI','YYY']
INDEX_LIST = [(4,0),(6,2),(5,1),(7,3),(2,0),(6,4),(3,1),(7,5),(1,0),(3,2),(5,4),(7,6)]
FREQ_WINDOWS = [
    (-1001,-930),(-866,-765),(-1072,-1001),(-930,-866),
    (-45,40),(100,200),(-153,-45),(40,100),
    (721,840),(647,684),(684,721),(571,647)
]
PHASES = [4.593571789275965, 4.729046676847299, 3.8289005158044778]

def hermitian_basis():
    out=[]
    for i in range(DIM-1):
        M=np.zeros((DIM,DIM),complex); M[i,i]=1; M[DIM-1,DIM-1]=-1; out.append(M)
    for i in range(DIM):
        for j in range(i+1,DIM):
            M=np.zeros((DIM,DIM),complex); M[i,j]=1; M[j,i]=1; out.append(M)
            M=np.zeros((DIM,DIM),complex); M[i,j]=-1j; M[j,i]=1j; out.append(M)
    return out
BASIS=hermitian_basis()

def build_A():
    A=[]
    for U in U_LIST:
        for i,j in INDEX_LIST:
            rr=[]; ri=[]
            for B in BASIS:
                R=U@B@U.conj().T
                rr.append(R[i,j].real); ri.append(R[i,j].imag)
            A.extend([rr,ri])
    return np.asarray(A)
A=build_A()

def load_data(path):
    d=np.loadtxt(path, delimiter=',', skiprows=1)
    d=d[np.argsort(d[:,0])]
    return d[:,0],d[:,1]

def integrate_window(window,freq,spec):
    dense=np.linspace(window[0],window[1],1000)
    interp=np.interp(dense,freq,spec)
    return np.trapezoid(interp,dense)/(window[1]-window[0])

def reconstruct(folder):
    F=[]
    for label in LABELS:
        Rf,Rs=load_data(os.path.join(folder,label+',R.csv'))
        If,Is=load_data(os.path.join(folder,label+',I.csv'))
        p=np.array([integrate_window(w,Rf,Rs)+1j*integrate_window(w,If,Is) for w in FREQ_WINDOWS])
        p[:4] *= np.exp(-1j*PHASES[0])
        p[4:8] *= np.exp(-1j*PHASES[1])
        p[8:] *= np.exp(-1j*PHASES[2])
        for z in p: F.extend([z.real,z.imag])
    F=np.asarray(F)/3800.0
    r,*_=np.linalg.lstsq(A,F,rcond=None)
    D=sum((c*B for c,B in zip(r,BASIS)), start=np.zeros((DIM,DIM),complex))
    return D

def hs_corr(D1,D2):
    return np.real(np.trace(D1@D2))/np.sqrt(np.real(np.trace(D1@D1))*np.real(np.trace(D2@D2)))

psi0=np.zeros((8,1),complex); psi0[0,0]=1
D0=psi0@psi0.conj().T-np.eye(8)/8
H=np.array([[1,1],[1,-1]],complex)/np.sqrt(2)
UH=kron3(H,I2,I2)
psiH=UH@psi0
DH=psiH@psiH.conj().T-np.eye(8)/8

Dpps=reconstruct(os.path.join(BASE,'PPS'))
Cpps=hs_corr(Dpps,D0)
print('C_PPS =',Cpps)

conds=[(-10,'NM10','RM10'),(-5,'NM5','RM5'),(0,'N0','R0'),(5,'NP5','RP5'),(10,'NP10','RP10')]
sim_nom_F={-10:0.47740,-5:0.82272,0:0.99130,5:0.82560,10:0.46757}
sim_rob_F={-10:0.95995,-5:0.98524,0:0.98171,5:0.97507,10:0.95644}
rows=[]
for eps,nf,rf in conds:
    Cn=hs_corr(reconstruct(os.path.join(BASE,nf)),DH)
    Cr=hs_corr(reconstruct(os.path.join(BASE,rf)),DH)
    # for a pure simulated output vs pure target in d=8:
    # C_HS=(8F_state-1)/7
    Sn=(8*sim_nom_F[eps]-1)/7
    Sr=(8*sim_rob_F[eps]-1)/7
    rows.append({
        'epsilon_RF_percent':eps,
        'sim_nominal_C_HS':Sn,
        'exp_nominal_C_H':Cn,
        'exp_nominal_R_H=C_H/C_PPS':Cn/Cpps,
        'sim_robust_C_HS':Sr,
        'exp_robust_C_H':Cr,
        'exp_robust_R_H=C_H/C_PPS':Cr/Cpps,
        'robust_minus_nominal_C_H':Cr-Cn,
    })

df=pd.DataFrame(rows)
print(df.to_string(index=False, float_format=lambda x:f'{x:.6f}'))