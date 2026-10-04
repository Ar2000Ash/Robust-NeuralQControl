#!/usr/bin/env python3
"""Joint effective uncertainty propagation for experimental NMR tomography.

This is the final manuscript-facing uncertainty model. It jointly propagates
the stochastic spectral perturbations and the reconstruction-sensitivity
sources identified in the preceding analyses through the same 63-parameter
deviation-matrix tomography used for the central experimental results.

Each Monte-Carlo realization includes:

1. independent ±2% pointwise spectral-amplitude uncertainty, propagated to the
   integrated transition amplitudes through their exact first two moments;
2. a smooth linear frequency drift whose start/end offsets are sampled
   independently from ±5 Hz across the 11 sequential readouts;
3. integration-window lower/upper edge perturbations within one digitized
   frequency bin (±2.5601565 Hz), shared across readouts in a realization;
4. receiver-phase perturbations sampled uniformly within ±1 degree for each of
   the three spin groups;
5. baseline-model uncertainty interpolating between no correction and either
   a fitted linear or quadratic baseline;
6. conservative trust weights for the two partially overlapped central Q3
   transitions, sampled from a triangular distribution with mode at full
   weight and applied through weighted least squares.

The experimentally reconstructed C_HS values remain the reported central
values. The effective uncertainty interval is the 2.5--97.5 percentile range
of the joint distribution. Default settings reproduce the final manuscript
uncertainty values with 10,000 realizations per condition.
"""

import argparse
import tempfile
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import expm

DEFAULT_DATA = Path("data/experimental/digitized_spectra.zip")
DEFAULT_OUTPUT = Path(
    "results/tomography/tomography_joint_effective_uncertainty.csv"
)
N_MC = 10_000
SEED = 24680
AMP_FRAC = 0.02
SHIFT_MAX = 5.0
SHIFT_STEP = 0.1
PHASE_MAX_DEG = 1.0
BIN_HZ = 2.5601565

CONDITIONS=['PPS','NM10','NM5','N0','NP5','NP10','RM10','RM5','R0','RP5','RP10']
LABELS=['III','XII','IXI','IIX','IXX','XXX','YII','IYI','IIY','YYI','YYY']
WINDOWS=[(-1001,-930),(-866,-765),(-1072,-1001),(-930,-866),(-45,40),(100,200),(-153,-45),(40,100),(721,840),(647,684),(684,721),(571,647)]
PHASES=np.array([4.593571789275965,4.729046676847299,3.8289005158044778])
NOISE_REGIONS=[(-700,-250),(300,500),(900,1100)]

DIM=8
I2=np.eye(2,dtype=complex)
X=np.array([[0,1],[1,0]],complex)
Y=np.array([[0,-1j],[1j,0]],complex)
def kron3(a,b,c): return np.kron(np.kron(a,b),c)
Ux=expm(-1j*X*np.pi/4); Uy=expm(-1j*Y*np.pi/4)
U_LIST=[np.eye(8),kron3(Ux,I2,I2),kron3(I2,Ux,I2),kron3(I2,I2,Ux),kron3(I2,Ux,Ux),kron3(Ux,Ux,Ux),kron3(Uy,I2,I2),kron3(I2,Uy,I2),kron3(I2,I2,Uy),kron3(Uy,Uy,I2),kron3(Uy,Uy,Uy)]
INDEX_LIST=[(4,0),(6,2),(5,1),(7,3),(2,0),(6,4),(3,1),(7,5),(1,0),(3,2),(5,4),(7,6)]

def basis():
    out=[]
    for i in range(7):
        M=np.zeros((8,8),complex); M[i,i]=1; M[-1,-1]=-1; out.append(M)
    for i in range(8):
        for j in range(i+1,8):
            M=np.zeros((8,8),complex); M[i,j]=M[j,i]=1; out.append(M)
            M=np.zeros((8,8),complex); M[i,j]=-1j; M[j,i]=1j; out.append(M)
    return out
BASIS=basis()

def build_A():
    rows=[]
    for U in U_LIST:
        for i,j in INDEX_LIST:
            rr=[]; ii=[]
            for B in BASIS:
                R=U@B@U.conj().T; rr.append(R[i,j].real); ii.append(R[i,j].imag)
            rows.extend([rr,ii])
    return np.asarray(rows,float)
A=build_A()

G=np.array([[np.real(np.trace(Bi@Bj)) for Bj in BASIS] for Bi in BASIS])
psi0=np.zeros((8,1),complex); psi0[0,0]=1
D_PPS=psi0@psi0.conj().T-np.eye(8)/8
H=np.array([[1,1],[1,-1]],complex)/np.sqrt(2)
psiH=kron3(H,I2,I2)@psi0
D_H=psiH@psiH.conj().T-np.eye(8)/8

def target_info(D):
    return np.array([np.real(np.trace(B@D)) for B in BASIS]), np.real(np.trace(D@D))
Q_PPS,N_PPS=target_info(D_PPS); Q_H,N_H=target_info(D_H)

SHIFT_GRID=np.arange(-SHIFT_MAX,SHIFT_MAX+0.5*SHIFT_STEP,SHIFT_STEP)
ZERO_IDX=int(np.argmin(abs(SHIFT_GRID)))

def find_root(td):
    for p in td.rglob('PPS'):
        if p.is_dir() and (p/'III,R.csv').exists(): return p.parent
    raise RuntimeError('data root not found')

def load_spec(p):
    a=np.loadtxt(p,delimiter=',',skiprows=1); a=a[np.argsort(a[:,0])]
    return a[:,0],a[:,1]

def int_weights(freq,window,shift=0,n_dense=1000):
    a,b=window; q=np.linspace(a,b,n_dense)-shift
    tw=np.ones(n_dense)/(n_dense-1); tw[0]*=.5; tw[-1]*=.5
    W=np.zeros(len(freq))
    left=q<=freq[0]; right=q>=freq[-1]; mid=~(left|right)
    W[0]+=tw[left].sum(); W[-1]+=tw[right].sum()
    if np.any(mid):
        qm=q[mid]; wt=tw[mid]; idx=np.searchsorted(freq,qm,side='left')
        x0=freq[idx-1]; x1=freq[idx]; t=(qm-x0)/(x1-x0)
        np.add.at(W,idx-1,wt*(1-t)); np.add.at(W,idx,wt*t)
    return W

def covroot(C):
    vals,vecs=np.linalg.eigh((C+C.T)/2); vals=np.clip(vals,0,None)
    return vecs@np.diag(np.sqrt(vals))

def fit_baselines(freq,y):
    mask=np.zeros(len(freq),bool)
    for lo,hi in NOISE_REGIONS: mask |= (freq>=lo)&(freq<=hi)
    return [np.polyval(np.polyfit(freq[mask],y[mask],deg),freq) for deg in (1,2)]

def precompute(folder):
    outs=[]
    for lab in LABELS:
        ent={}
        for comp in ('R','I'):
            freq,y=load_spec(folder/f'{lab},{comp}.csv')
            bls=fit_baselines(freq,y)
            means=[]; roots=[]; blmeans=[[],[]]
            draw_lo=[]; draw_hi=[]; dbl_lo=[[],[]]; dbl_hi=[[],[]]
            for sh in SHIFT_GRID:
                Ws=[]
                for win in WINDOWS: Ws.append(int_weights(freq,win,sh))
                W=np.stack(Ws)
                m=W@y
                means.append(m)
                Wy=W*y[None,:]
                roots.append(covroot((AMP_FRAC**2/3.0)*(Wy@Wy.T)))
                bm=[]
                for bi,bl in enumerate(bls):
                    bb=W@bl; blmeans[bi].append(bb); bm.append(bb)
                # exact first derivatives of normalized window average wrt bounds
                dlo=np.empty(12); dhi=np.empty(12)
                dblo=[np.empty(12),np.empty(12)]; dbhi=[np.empty(12),np.empty(12)]
                for j,(a,b) in enumerate(WINDOWS):
                    width=b-a
                    sval_a=np.interp(a-sh,freq,y); sval_b=np.interp(b-sh,freq,y)
                    dlo[j]=(m[j]-sval_a)/width
                    dhi[j]=(sval_b-m[j])/width
                    for bi,bl in enumerate(bls):
                        ba=np.interp(a-sh,freq,bl); bbnd=np.interp(b-sh,freq,bl)
                        dblo[bi][j]=(bm[bi][j]-ba)/width
                        dbhi[bi][j]=(bbnd-bm[bi][j])/width
                draw_lo.append(dlo); draw_hi.append(dhi)
                for bi in range(2):
                    dbl_lo[bi].append(dblo[bi]); dbl_hi[bi].append(dbhi[bi])
            ent[comp]={
                'mean':np.stack(means), 'root':np.stack(roots),
                'bl1':np.stack(blmeans[0]), 'bl2':np.stack(blmeans[1]),
                'dlo':np.stack(draw_lo), 'dhi':np.stack(draw_hi),
                'bl1_dlo':np.stack(dbl_lo[0]), 'bl1_dhi':np.stack(dbl_hi[0]),
                'bl2_dlo':np.stack(dbl_lo[1]), 'bl2_dhi':np.stack(dbl_hi[1]),
            }
        outs.append(ent)
    return outs

def central_F(pre):
    F=[]
    pvec=np.array([np.exp(-1j*PHASES[0])]*4+[np.exp(-1j*PHASES[1])]*4+[np.exp(-1j*PHASES[2])]*4)
    for e in pre:
        R=e['R']['mean'][ZERO_IDX]; I=e['I']['mean'][ZERO_IDX]
        z=(R+1j*I)*pvec
        for zz in z: F.extend([zz.real/3800,zz.imag/3800])
    return np.asarray(F)

def hs_from_r(r,q,norm):
    num=r@q
    den=np.sqrt(np.maximum(np.einsum('bi,ij,bj->b',r,G,r),0)*norm)
    return num/den

def weighted_maps():
    maps={}
    # grid 0...1 in steps of 0.1 for the two central Q3 transition trust weights
    for i,w9 in enumerate(np.linspace(0,1,11)):
        for j,w10 in enumerate(np.linspace(0,1,11)):
            wr=np.ones(264)
            for ro in range(11):
                for t,w in ((9,w9),(10,w10)):
                    wr[24*ro+2*t:24*ro+2*t+2]=w
            sw=np.sqrt(wr)
            Aw=A*sw[:,None]
            maps[(i,j)] = np.linalg.pinv(Aw) * sw[None,:]  # 63x264
    return maps
WMAPS=weighted_maps()
PINV=np.linalg.pinv(A)

def draw_shift_idx(rng,n):
    s=rng.uniform(-SHIFT_MAX,SHIFT_MAX,n); e=rng.uniform(-SHIFT_MAX,SHIFT_MAX,n)
    out=np.empty((n,11),int)
    for j in range(11):
        t=j/10; v=(1-t)*s+t*e
        out[:,j]=np.clip(np.rint((v+SHIFT_MAX)/SHIFT_STEP).astype(int),0,len(SHIFT_GRID)-1)
    return out

def joint_mc(pre,q,norm,seed,n_mc=N_MC):
    rng=np.random.default_rng(seed)
    n=n_mc
    shifts=draw_shift_idx(rng,n)
    # Same analysis-window definitions across all readouts in a realization.
    da=rng.uniform(-BIN_HZ,BIN_HZ,size=(n,12))
    db=rng.uniform(-BIN_HZ,BIN_HZ,size=(n,12))
    # Baseline uncertainty between no correction and the fitted correction;
    # line/quadratic model selected with equal probability.
    beta=rng.uniform(0,1,n)
    bmodel=rng.integers(0,2,n)  # 0 linear, 1 quadratic
    # Frozen receiver-phase uncertainty.
    dphi=np.deg2rad(rng.uniform(-PHASE_MAX_DEG,PHASE_MAX_DEG,size=(n,3)))
    # Conservative Type-B Q3 trust factors. Triangular distribution with mode=1:
    # full measured Q3 information is most likely; complete downweighting is allowed but rare.
    w9=rng.triangular(0,1,1,n); w10=rng.triangular(0,1,1,n)
    i9=np.clip(np.rint(w9*10).astype(int),0,10); i10=np.clip(np.rint(w10*10).astype(int),0,10)

    F=np.empty((n,264),float); col=0
    for ri,e in enumerate(pre):
        idx=shifts[:,ri]
        comps=[]
        for comp in ('R','I'):
            p=e[comp]
            m=p['mean'][idx]
            dlo=p['dlo'][idx]; dhi=p['dhi'][idx]
            # select baseline model arrays row-wise
            B=np.where(bmodel[:,None]==0,p['bl1'][idx],p['bl2'][idx])
            Bdlo=np.where(bmodel[:,None]==0,p['bl1_dlo'][idx],p['bl2_dlo'][idx])
            Bdhi=np.where(bmodel[:,None]==0,p['bl1_dhi'][idx],p['bl2_dhi'][idx])
            mu = m + dlo*da + dhi*db - beta[:,None]*(B + Bdlo*da + Bdhi*db)
            root=p['root'][idx]
            z=rng.standard_normal((n,12))
            val=mu+np.einsum('bij,bj->bi',root,z)
            comps.append(val)
        R,I=comps
        ph=np.empty((n,12))
        ph[:,:4]=PHASES[0]+dphi[:,0,None]
        ph[:,4:8]=PHASES[1]+dphi[:,1,None]
        ph[:,8:]=PHASES[2]+dphi[:,2,None]
        zz=(R+1j*I)*np.exp(-1j*ph)
        for k in range(12):
            F[:,col]=zz[:,k].real/3800; F[:,col+1]=zz[:,k].imag/3800; col+=2

    # Weighted LS according to Q3 trust factors; batch by 11x11 grid pair.
    r=np.empty((n,63),float)
    code=i9*11+i10
    for c in np.unique(code):
        mask=code==c; a=c//11; b=c%11
        r[mask]=F[mask]@WMAPS[(a,b)].T
    return hs_from_r(r,q,norm)

def summarize(x,central):
    lo,hi=np.quantile(x,[.025,.975])
    return dict(central_C_HS=central, effective_mean=float(np.mean(x)), effective_sd=float(np.std(x,ddof=1)), effective_95_low=float(lo), effective_95_high=float(hi), effective_minus=float(central-lo), effective_plus=float(hi-central))

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
        help="Final joint/effective uncertainty table",
    )
    parser.add_argument(
        "--n-mc",
        type=int,
        default=N_MC,
        help="Monte-Carlo realizations per condition",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Base random seed",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = args.data.resolve()
    output_path = args.output.resolve()

    if not data_path.exists():
        raise FileNotFoundError(f"Digitized spectra archive not found: {data_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows=[]
    with tempfile.TemporaryDirectory() as td:
        td=Path(td)
        with zipfile.ZipFile(data_path) as z:
            z.extractall(td)
        root=find_root(td)
        for ci,c in enumerate(CONDITIONS):
            print('precompute',c,flush=True)
            pre=precompute(root/c)
            F0=central_F(pre); r0=PINV@F0
            q,norm=(Q_PPS,N_PPS) if c=='PPS' else (Q_H,N_H)
            central=float(hs_from_r(r0[None,:],q,norm)[0])
            print('MC',c,'central',central,'n=',args.n_mc,flush=True)
            x=joint_mc(pre,q,norm,args.seed+1000*ci,n_mc=args.n_mc)
            s=summarize(x,central); s['condition']=c; rows.append(s)
            print(c,s,flush=True)
    df=pd.DataFrame(rows)[['condition','central_C_HS','effective_mean','effective_sd','effective_95_low','effective_95_high','effective_minus','effective_plus']]
    df.to_csv(output_path,index=False)
    print('\n',df.to_string(index=False))
    print('\nSaved',output_path)

if __name__=='__main__': main()
