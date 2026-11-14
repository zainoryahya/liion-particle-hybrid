"""Fit term eta_K(sigma_1) of Theorem 2 (variation carried to sigma = 1e12, plus the remaining |g - ghat| there) for the
40-rate universal correction and for the earlier 20-rate choice, and gains of the 20-rate choice at dtau = 2.8e-7.
Writes ext/out/universal_eta.json."""
import sys, numpy as np, io, contextlib, json
import os; ROOT = os.environ.get('LIION_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT+'/ext')
src=open(ROOT+'/ext/hybrid_theory2.py').read(); g={'__file__':ROOT+'/ext/hybrid_theory2.py'}
with contextlib.redirect_stdout(io.StringIO()):
    exec(src[:src.index('# ---------- Prop 1')], g)
g_half=g['g_half']; S_exact=g['S_exact']; S_fv=g['S_fv']
import horizon as HZ
def fit(NU, sf):
    m=sf<=1e5; gf=np.r_[g_half(sf[m]), 1/(8*np.sqrt(np.pi*sf[~m]))]
    Wt=1/np.maximum(np.abs(gf),1e-6)**0.5; Phi=np.exp(-np.outer(sf,NU))*Wt[:,None]
    KKT=np.block([[2*Phi.T@Phi,np.ones((len(NU),1))],[np.ones((1,len(NU))),np.zeros((1,1))]])
    return np.linalg.solve(KKT,np.r_[2*Phi.T@(gf*Wt),0.5])[:len(NU)]
def eta(NU,w,s1):
    ss=np.r_[s1, np.geomspace(s1*1.0001,1e12,20000)]
    gg=np.where(ss<=1e5,0.0,1/(8*np.sqrt(np.pi*ss))); m=ss<=1e5; gg[m]=g_half(ss[m])
    dg=gg-np.exp(-np.outer(ss,NU))@w
    return abs(dg[0])+np.abs(np.diff(dg)).sum()+abs(dg[-1])
U=json.load(open(ROOT+'/ext/out/hybrid_theory.json'))['universal']; NUn=np.array(U['nu']); wn=np.array(U['w'])
NUo=np.logspace(-6,3,20); wo=fit(NUo, np.logspace(-5,6,700))
OUT={'eta':[]}
for s1 in [1e-9,1e-6,1e-4,1e-3,1,10]:
    OUT['eta'].append(dict(sigma1=s1, new=float(eta(NUn,wn,s1)), old20=float(eta(NUo,wo,s1))))
    print('sigma1 %.0e  eta new %.2e  old %.2e'%(s1, eta(NUn,wn,s1), eta(NUo,wo,s1)))
dt=2.8e-7; LH=HZ.lags_for(dt); Sx=S_exact(LH); Sx[0]=0
G=[HZ.gain(Sx-S_fv(N,LH)-(1/N)*(np.exp(-np.outer(LH*N*N,NUo))@wo)) for N in [5,10,20,40,80,160]]
print('old 20-rate gains at 2.8e-7', ['%.2e'%x for x in G], 'orders', ['%.2f'%(np.log(G[i]/G[i+1])/np.log(2)) for i in range(5)])
OUT['old20_gains_2.8e-7']=[float(x) for x in G]
json.dump(OUT, open(ROOT+'/ext/out/universal_eta.json','w'), indent=1)
