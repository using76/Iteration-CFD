#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
# Every face of one polyMesh cell: nodes, owner/neighbour, patch, area vector, and the
# cell's face-based volume and closure - to see why the loader calls it inside out.
#   python cell_faces.py <case> <cell>
import os, re, sys
import numpy as np
case, cell = sys.argv[1], int(sys.argv[2])
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
faces = read_list(os.path.join(pm, 'faces')).astype(np.int64)
owner = read_list(os.path.join(pm, 'owner')).astype(np.int64).ravel()
neigh = read_list(os.path.join(pm, 'neighbour')).astype(np.int64).ravel()
ni = len(neigh)
bnd = open(os.path.join(pm, 'boundary'), encoding='utf-8', errors='replace').read()
patches = [(m.group(1), int(m.group(3)), int(m.group(4))) for m in
           re.finditer(r'(\w+)\s*\{\s*type\s+(\w+);.*?nFaces\s+(\d+);\s*startFace\s+(\d+);', bnd, re.S)]


def patch_of(f):
    for name, n, s in patches:
        if s <= f < s + n: return name
    return 'internal'


fo = np.where(owner == cell)[0]; fn = np.where(neigh == cell)[0]
fl = sorted(set(fo.tolist()) | set(fn.tolist()))
print('cell %d: %d faces (%d as owner, %d as neighbour)' % (cell, len(fl), len(fo), len(fn)))
vol = 0.0; ssum = np.zeros(3)
for f in fl:
    nodes = faces[f, 1:1 + int(faces[f, 0])]
    P = pts[nodes]
    Sf = 0.5 * np.cross(P[1] - P[0], P[2] - P[0])
    for k in range(3, len(P)):
        Sf += 0.5 * np.cross(P[k - 1] - P[0], P[k] - P[0])
    sign = 1.0 if owner[f] == cell else -1.0          # outward for this cell
    xf = P.mean(axis=0)
    vol += sign * np.dot(xf, Sf) / 3.0
    ssum += sign * Sf
    other = int(neigh[f]) if f < ni and owner[f] == cell else int(owner[f]) if f < ni else -1
    print('  face %-9d nodes %-32s %s  owner %d neigh %s  |Sf| %.3f  centre (%.2f, %.2f, %.2f)' % (
        f, nodes.tolist(), patch_of(f), owner[f], neigh[f] if f < ni else '-', np.linalg.norm(Sf), *xf))
print('face-based volume %.4f, |sum Sf| %.4f' % (vol, np.linalg.norm(ssum)))
vs = sorted(set(v for f in fl for v in faces[f, 1:1 + int(faces[f, 0])].tolist()))
print('distinct nodes', vs)
for v in vs:
    print('   node %d (%.4f, %.4f, %.4f)' % (v, *pts[v]))
# other cells sharing 3 of these nodes (potential twins)
