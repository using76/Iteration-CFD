#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
import gmsh, math, sys, json
import numpy as np
from scipy.spatial import cKDTree
brep = sys.argv[1]; tol = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
ck = json.load(open(brep[:-5] + '.json', encoding='utf-8'))
gmsh.initialize(); gmsh.option.setNumber('General.Terminal', 0)
gmsh.option.setNumber('Geometry.OCCScaling', 1.0)
gmsh.model.occ.importShapes(brep); gmsh.model.occ.synchronize()
pts = gmsh.model.getEntities(0)
tags = np.array([p for d, p in pts])
xyz = np.array([gmsh.model.getValue(0, p, []) for d, p in pts])
print('vertices:', len(tags))
tree = cKDTree(xyz)
pairs = tree.query_pairs(tol)
print('pairs closer than %.3f m: %d' % (tol, len(pairs)))
def solids_at(x):
    out = []
    for tg, bb in ck['solid_bboxes'].items():
        if bb[0] - 0.01 <= x[0] <= bb[3] + 0.01 and bb[1] - 0.01 <= x[1] <= bb[4] + 0.01 and bb[2] - 0.01 <= x[2] <= bb[5] + 0.01:
            out.append(int(tg))
    return out
rows = []
for i, j in pairs:
    d = float(np.linalg.norm(xyz[i] - xyz[j]))
    if d < 1e-3:
        continue                      # (near-)coincident vertices: the mesher merges them
    rows.append((d, i, j))
rows.sort()
import collections
hist = collections.Counter()
for i, j in pairs:
    d = float(np.linalg.norm(xyz[i] - xyz[j]))
    hist['<1e-6' if d < 1e-6 else '<1e-3' if d < 1e-3 else '<1e-2' if d < 1e-2 else '<5e-2'] += 1
print('histogram', dict(hist))
print('pairs with 1 mm < d < %.3f m: %d' % (tol, len(rows)))
seen = set()
for d, i, j in rows:
    key = tuple(np.round(0.5 * (xyz[i] + xyz[j]), 1))
    if key in seen:
        continue
    seen.add(key)
    s = sorted(set(solids_at(xyz[i]) + solids_at(xyz[j])))
    r = math.hypot(xyz[i][0] - 18.5, xyz[i][1] + 7.5)
    print('  d %.4f m at (%.2f, %.2f, %.2f)  solids %s  dist from QCDC %.0f m' % (d, xyz[i][0], xyz[i][1], xyz[i][2], s, r))
gmsh.finalize()
