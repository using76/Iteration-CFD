#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
import gmsh, math, sys, json
brep = sys.argv[1]; px, py, pz = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
pad = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5
ck = json.load(open(brep[:-5] + '.json', encoding='utf-8'))
gmsh.initialize(); gmsh.option.setNumber('General.Terminal', 0)
gmsh.option.setNumber('Geometry.OCCScaling', 1.0)
gmsh.model.occ.importShapes(brep); gmsh.model.occ.synchronize()
vol = gmsh.model.getEntities(3)[0]
def near(b):
    return (b[0] - pad <= px <= b[3] + pad and b[1] - pad <= py <= b[4] + pad and b[2] - pad <= pz <= b[5] + pad)
print('solids (imported tags) whose bbox contains the point:')
for tg, bb in ck['solid_bboxes'].items():
    if near(bb):
        print('  solid %s bbox x[%.2f,%.2f] y[%.2f,%.2f] z[%.2f,%.2f]' % (tg, bb[0], bb[3], bb[1], bb[4], bb[2], bb[5]))
print('fluid boundary faces near the point:')
faces = []
for d, s in gmsh.model.getBoundary([vol], oriented=False):
    b = gmsh.model.getBoundingBox(2, s)
    if near(b):
        a = gmsh.model.occ.getMass(2, s)
        faces.append(s)
        print('  s%-6d area %10.3f  x[%.2f,%.2f] y[%.2f,%.2f] z[%.3f,%.3f]  %s' % (s, a, b[0], b[3], b[1], b[4], b[2], b[5], gmsh.model.getType(2, s)))
print('curves of those faces near the point:')
seen = set()
for s in faces:
    for d, c in gmsh.model.getBoundary([(2, s)], oriented=False, recursive=False):
        if c in seen: continue
        b = gmsh.model.getBoundingBox(1, c)
        if not near(b): continue
        seen.add(c)
        L = gmsh.model.occ.getMass(1, c)
        ends = [p for d0, p in gmsh.model.getBoundary([(1, c)], oriented=False)]
        ep = ['p%d(%.3f,%.3f,%.3f)' % ((p,) + tuple(gmsh.model.getValue(0, p, []))) for p in ends]
        up = [t for t in gmsh.model.getAdjacencies(1, c)[0]]
        print('  c%-6d L %8.4f  x[%.2f,%.2f] y[%.2f,%.2f] z[%.3f,%.3f] %s faces %s ends %s' % (
            c, L, b[0], b[3], b[1], b[4], b[2], b[5], gmsh.model.getType(1, c), up, ep))
print('vertices near the point:')
for d, p in gmsh.model.getEntities(0):
    x = gmsh.model.getValue(0, p, [])
    if abs(x[0] - px) < pad and abs(x[1] - py) < pad and abs(x[2] - pz) < pad:
        up = gmsh.model.getAdjacencies(0, p)[0]
        print('  p%-6d (%.4f, %.4f, %.4f) dist %.4f  curves %s' % (p, x[0], x[1], x[2], math.dist(x, (px, py, pz)), list(up)))
gmsh.finalize()
