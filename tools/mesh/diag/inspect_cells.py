#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
# Geometry of given cells (and of all cells within a radius of a point) in a tet polyMesh:
# volume, thickness 3V/Amax, face angles, patches, neighbours.
#   python inspect_cells.py <case> --cells 1,2,3 [--near x y z r]
import os, re, sys, math
import numpy as np
case = sys.argv[1]
cells = []; near = None
a = sys.argv[2:]
while a:
    k = a.pop(0)
    if k == '--cells': cells = [int(v) for v in a.pop(0).split(',') if v]
    elif k == '--near': near = [float(a.pop(0)) for _ in range(4)]
pm = os.path.join(case, 'constant', 'polyMesh')


def read_list(path):
    with open(path, 'rb') as f:
        data = f.read()
    m = re.search(rb'\n(\d+)\s*\n\(', data)
    n = int(m.group(1))
    body = data[m.end():]
    body = body[:body.rfind(b')')]
    return np.fromstring(body.replace(b'(', b' ').replace(b')', b' '), sep=' ').reshape(n, -1)


pts = read_list(os.path.join(pm, 'points'))
faces = read_list(os.path.join(pm, 'faces')).astype(np.int64)[:, 1:]
owner = read_list(os.path.join(pm, 'owner')).astype(np.int64).ravel()
neigh = read_list(os.path.join(pm, 'neighbour')).astype(np.int64).ravel()
ni = len(neigh); nf = len(faces)
bnd = open(os.path.join(pm, 'boundary'), encoding='utf-8', errors='replace').read()
patches = [(m.group(1), int(m.group(3)), int(m.group(4))) for m in
           re.finditer(r'(\w+)\s*\{\s*type\s+(\w+);.*?nFaces\s+(\d+);\s*startFace\s+(\d+);', bnd, re.S)]


def patch_of(f):
    for name, n, s in patches:
        if s <= f < s + n: return name
    return 'internal'


P = pts[faces]; fc = P.mean(axis=1)
Sf = 0.5 * np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]); area = np.linalg.norm(Sf, axis=1)
ncell = int(max(owner.max(), neigh.max())) + 1
if near:
    x, y, z, r = near
    sel_f = np.where(np.linalg.norm(fc - np.array([x, y, z]), axis=1) < r)[0]
    cells = sorted(set(cells) | set(owner[sel_f].tolist()) | set(neigh[sel_f[sel_f < ni]].tolist()))
    print('cells within %.0f m of (%.1f, %.1f, %.1f): %d' % (r, x, y, z, len(cells)))
cells = np.array(sorted(set(cells)), dtype=np.int64)
# faces of the selected cells
mo = np.isin(owner, cells); mn = np.zeros(nf, bool); mn[:ni] = np.isin(neigh, cells)
fsel = np.where(mo | mn)[0]
cf = {}
for f in fsel:
    cf.setdefault(int(owner[f]), []).append(int(f))
    if f < ni: cf.setdefault(int(neigh[f]), []).append(int(f))
# cell centres of the selected cells and their neighbours (for face angles)
need = set(cells.tolist())
for f in fsel:
    need.add(int(owner[f]))
    if f < ni: need.add(int(neigh[f]))
need = np.array(sorted(need)); mo2 = np.isin(owner, need); mn2 = np.zeros(nf, bool); mn2[:ni] = np.isin(neigh, need)
f2 = np.where(mo2 | mn2)[0]
cc = {}; cnt = {}
for f in f2:
    for c in ([int(owner[f])] + ([int(neigh[f])] if f < ni else [])):
        if c in need:
            cc[c] = cc.get(c, 0) + fc[f]; cnt[c] = cnt.get(c, 0) + 1
cc = {c: cc[c] / cnt[c] for c in cc}
rows = []
for c in cells.tolist():
    fl = cf.get(c, [])
    # tet volume from the four vertices
    vs = sorted(set(faces[fl].ravel().tolist()))
    vol = None
    if len(vs) == 4:
        q = pts[vs]; vol = abs(np.dot(q[1] - q[0], np.cross(q[2] - q[0], q[3] - q[0]))) / 6.0
    amax = area[fl].max(); thick = 3 * vol / amax if vol else float('nan')
    angs = []
    for f in fl:
        if f < ni:
            o, n = int(owner[f]), int(neigh[f])
            d = cc[n] - cc[o]
            ca = np.dot(d, Sf[f]) / (np.linalg.norm(d) * area[f] + 1e-300)
            angs.append(math.degrees(math.acos(max(-1, min(1, ca)))))
    pats = sorted(set(patch_of(f) for f in fl if f >= ni))
    edges = [np.linalg.norm(pts[vs[i]] - pts[vs[j]]) for i in range(len(vs)) for j in range(i + 1, len(vs))]
    rows.append((c, vol, thick, max(angs) if angs else 0, min(edges), max(edges), pats, cc[c]))
rows.sort(key=lambda r: (-(r[3] or 0)))
print('%-9s %11s %9s %8s %8s %8s  %s' % ('cell', 'V m3', '3V/Amax', 'maxAng', 'eMin', 'eMax', 'patches / centre'))
for c, vol, thick, ang, emin, emax, pats, ctr in rows[:40]:
    print('%-9d %11.4e %9.4f %8.2f %8.3f %8.3f  %s (%.1f, %.1f, %.1f)' % (c, vol or 0, thick, ang, emin, emax, ','.join(pats) or '-', ctr[0], ctr[1], ctr[2]))
print('summary: %d cells; thickness < 0.05 m: %d, < 0.2 m: %d; max face angle > 85: %d, > 80: %d; min volume %.3e' % (
    len(rows), sum(1 for r in rows if r[2] < 0.05), sum(1 for r in rows if r[2] < 0.2),
    sum(1 for r in rows if r[3] > 85), sum(1 for r in rows if r[3] > 80), min(r[1] or 1e9 for r in rows)))
