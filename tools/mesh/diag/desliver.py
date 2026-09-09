#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
import numpy as np, re, sys, os, collections
src, dst, THR = sys.argv[1], sys.argv[2], float(sys.argv[3])
pm = os.path.join(src, 'constant', 'polyMesh')
def body(p):
    t = open(p).read(); i = t.index('// * * *'); j = t.index('\n', i); return t[j:], t[:j+1]
def rl(p):
    b,_ = body(p); m = re.search(r'(\d+)\s*\(', b)
    return np.fromstring(b[m.end():b.rindex(')')], dtype=np.int64, sep=' ')
b,hdr = body(os.path.join(pm,'points'))
m = re.search(r'(\d+)\s*\(', b); npnt = int(m.group(1))
pts = np.fromstring(b[m.end():b.rindex(')')].replace('(',' ').replace(')',' '), dtype=np.float64, sep=' ').reshape(npnt,3)
owner = rl(os.path.join(pm,'owner')); neigh = rl(os.path.join(pm,'neighbour'))
bfx,_ = body(os.path.join(pm,'faces')); m = re.search(r'(\d+)\s*\(', bfx)
tri = np.array([[int(x) for x in t.split()] for t in re.findall(r'3\(([^)]*)\)', bfx[m.end():])], dtype=np.int64)
nf, nif = len(owner), len(neigh); ncells = int(max(owner.max(), neigh.max()))+1
print(f"points {npnt} faces {nf} internal {nif} cells {ncells}", flush=True)
cf = [[] for _ in range(ncells)]
for f in range(nf): cf[owner[f]].append(f)
for f in range(nif): cf[neigh[f]].append(f)
sg = [[1.0]*len(x) for x in cf]
for c in range(ncells):
    sg[c] = [1.0 if owner[f]==c else -1.0 for f in cf[c]]
cf = np.array(cf, dtype=np.int64); sg = np.array(sg)
cv = np.array([np.unique(tri[cf[c]]) for c in range(ncells)], dtype=np.int64)
p2c = collections.defaultdict(list)
for c in range(ncells):
    for v in cv[c]: p2c[v].append(c)
print("topology built", flush=True)

def cellgeom(P, cells):
    F = cf[cells]; S = sg[cells]                    # (m,4)
    a=P[tri[F][:,:,0]]; bq=P[tri[F][:,:,1]]; c2=P[tri[F][:,:,2]]
    Sf = 0.5*np.cross(bq-a, c2-a); Cf=(a+bq+c2)/3.0
    mg = np.linalg.norm(Sf, axis=2)
    ap = Cf.mean(axis=1)
    V = np.einsum('mfi,mfi->m', Sf*S[:,:,None], Cf-ap[:,None,:])/3.0
    return V, mg.sum(axis=1), mg

allc = np.arange(ncells)
P = pts.copy()
V, A, _ = cellgeom(P, allc)
print(f"start: minV {V.min():.3e} thin(<{THR}) {(3*V/A<THR).sum()} min3V/A {(3*V/A).min():.3e}", flush=True)
moved=0
for it in range(40):
    th = 3*V/A; bad = np.nonzero(th < THR)[0]
    if len(bad)==0: break
    nm=0
    for c in bad:
        faces=cf[c]; verts=cv[c]
        Vc, Ac, mgc = cellgeom(P, np.array([c]))
        best=None
        for v in verts:
            k=[i for i,f in enumerate(faces) if v not in tri[f]][0]
            if best is None or mgc[0,k]>best[0]: best=(mgc[0,k], v, faces[k])
        Aopp, v, opp = best
        need = 1.6*THR*Ac[0]/3.0 - Vc[0]
        if need <= 0: continue
        t0=tri[opp]; n=np.cross(P[t0[1]]-P[t0[0]], P[t0[2]]-P[t0[0]]); nn=np.linalg.norm(n)
        if nn<=0: continue
        n/=nn
        sgn = 1.0 if np.dot(P[v]-P[t0[0]], n) >= 0 else -1.0
        aff = np.array(p2c[v]); Vb, Ab, _ = cellgeom(P, aff); base = (3*Vb/Ab).min(); baseV = Vb.min()
        step = 3*need/Aopp; old = P[v].copy(); okmove=False
        for k in range(10):
            P[v] = old + sgn*n*step
            Va, Aa, _ = cellgeom(P, aff)
            if Va.min() > 0 and (3*Va/Aa).min() >= min(base, THR*1.0)*0.9:
                okmove=True; break
            step *= 0.5
        if okmove: nm+=1; moved+=1
        else: P[v]=old
    V, A, _ = cellgeom(P, allc)
    print(f"  pass {it}: moved {nm} minV {V.min():.3e} thin {(3*V/A<THR).sum()} min3V/A {(3*V/A).min():.3e}", flush=True)
    if nm==0: break
d = np.linalg.norm(P-pts, axis=1)
print(f"points moved {(d>0).sum()} max {d.max():.4f} m mean {d[d>0].mean() if (d>0).any() else 0:.4f} m")
os.makedirs(os.path.join(dst,'constant','polyMesh'), exist_ok=True)
open(os.path.join(dst,'constant','polyMesh','points'),'w').write(
    hdr + f"\n{npnt}\n(\n" + "\n".join(f"({x:.14g} {y:.14g} {z:.14g})" for x,y,z in P) + "\n)\n")
print("written")
