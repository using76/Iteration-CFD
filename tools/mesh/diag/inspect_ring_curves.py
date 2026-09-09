#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
import gmsh, math, sys, json
brep = sys.argv[1]; cx, cy, r = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
gmsh.initialize(); gmsh.option.setNumber('General.Terminal', 0)
gmsh.option.setNumber('Geometry.OCCScaling', 1.0)
gmsh.model.occ.importShapes(brep); gmsh.model.occ.synchronize()
vol = gmsh.model.getEntities(3)[0]
surfs = gmsh.model.getBoundary([vol], oriented=False)
pad = 1.0
zg = 5.0
near = []
for d, s in surfs:
    b = gmsh.model.getBoundingBox(2, s)
    if b[3] < cx - r - pad or b[0] > cx + r + pad or b[4] < cy - r - pad or b[1] > cy + r + pad: continue
    if b[5] < zg - 0.5 or b[2] > zg + 1.0: continue
    near.append(s)
print('faces near the ring:', near)
curves = {}
for s in near:
    cs = gmsh.model.getBoundary([(2, s)], oriented=False, recursive=False)
    for d, c in cs:
        curves.setdefault(c, []).append(s)
print('curves of those faces (tag, length, bbox centre z, faces using it):')
rows = []
for c, ss in curves.items():
    L = gmsh.model.occ.getMass(1, c)
    bb = gmsh.model.getBoundingBox(1, c)
    if bb[3] < cx - r - pad or bb[0] > cx + r + pad or bb[4] < cy - r - pad or bb[1] > cy + r + pad: continue
    rows.append((L, c, ss, bb))
rows.sort()
for L, c, ss, bb in rows:
    print('  c%-6d L %9.4f  x[%.2f,%.2f] y[%.2f,%.2f] z[%.3f,%.3f]  faces %s  type %s' % (
        c, L, bb[0], bb[3], bb[1], bb[4], bb[2], bb[5], ss, gmsh.model.getType(1, c)))
# any two curves with the same length and bbox (coincident duplicates)?
print('coincident pairs:')
for i in range(len(rows)):
    for j in range(i + 1, len(rows)):
        Li, ci, si, bi = rows[i]; Lj, cj, sj, bj = rows[j]
        if abs(Li - Lj) < 1e-3 and all(abs(bi[k] - bj[k]) < 1e-3 for k in range(6)):
            print('  c%d ~ c%d  L %.4f  faces %s / %s' % (ci, cj, Li, si, sj))
# vertices near the ring: any duplicates (same xyz, different tags)?
pts = {}
for L, c, ss, bb in rows:
    for d, p in gmsh.model.getBoundary([(1, c)], oriented=False):
        xyz = gmsh.model.getValue(0, p, [])
        pts[p] = tuple(round(v, 4) for v in xyz)
inv = {}
for p, xyz in pts.items():
    inv.setdefault(xyz, []).append(p)
print('vertices:', len(pts), 'distinct positions:', len(inv))
for xyz, ps in inv.items():
    if len(ps) > 1:
        print('  duplicate vertices at', xyz, ps)
for p, xyz in sorted(pts.items()):
    print('  p%-6d (%.3f, %.3f, %.3f)  r %.3f' % (p, xyz[0], xyz[1], xyz[2], math.hypot(xyz[0]-cx, xyz[1]-cy)))
gmsh.finalize()
