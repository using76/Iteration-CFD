#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
# A small tester for the Fluent writer: a hexahedral fluid box with a tetrahedral solid in its
# middle, cut out. Outer faces are named; the solid's four faces are 'wall_solid'.
#   python tester_mesh.py [size]   -> tester_tet.msh (Gmsh 4.1 ASCII) + tester_tet_geometry.json
import gmsh, sys, json, itertools
import numpy as np
size = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
LX, LY, LZ = 10.0, 6.0, 4.0
TET = [(4.0, 2.0, 1.0), (6.5, 2.5, 1.2), (5.0, 4.0, 1.0), (5.2, 3.0, 3.2)]      # a general (non-axis-aligned) tetrahedron
gmsh.initialize(); gmsh.option.setNumber('General.Terminal', 0)
occ = gmsh.model.occ
box = occ.addBox(0, 0, 0, LX, LY, LZ)
pts = [occ.addPoint(*p) for p in TET]
lines = {}
def line(a, b):
    key = (min(a, b), max(a, b))
    if key not in lines:
        lines[key] = occ.addLine(pts[key[0]], pts[key[1]])
    return lines[key] if a < b else -lines[key]
faces = []
for tri in [(0, 1, 2), (0, 3, 1), (1, 3, 2), (0, 2, 3)]:
    a, b, c = tri
    faces.append(occ.addPlaneSurface([occ.addCurveLoop([line(a, b), line(b, c), line(c, a)])]))
tet = occ.addVolume([occ.addSurfaceLoop(faces)])
occ.synchronize()
v_tet = occ.getMass(3, tet)
out, _ = occ.cut([(3, box)], [(3, tet)], removeObject=True, removeTool=True)
occ.synchronize()
fluid = out[0][1]
v_fluid = occ.getMass(3, fluid)
groups = {'west': [], 'east': [], 'south': [], 'north': [], 'bottom': [], 'top': [], 'wall_solid': []}
for d, s in gmsh.model.getBoundary([(3, fluid)], oriented=False):
    b = gmsh.model.getBoundingBox(2, s); tol = 1e-6
    if abs(b[0]) < tol and abs(b[3]) < tol: groups['west'].append(s)
    elif abs(b[0] - LX) < tol and abs(b[3] - LX) < tol: groups['east'].append(s)
    elif abs(b[1]) < tol and abs(b[4]) < tol: groups['south'].append(s)
    elif abs(b[1] - LY) < tol and abs(b[4] - LY) < tol: groups['north'].append(s)
    elif abs(b[2]) < tol and abs(b[5]) < tol: groups['bottom'].append(s)
    elif abs(b[2] - LZ) < tol and abs(b[5] - LZ) < tol: groups['top'].append(s)
    else: groups['wall_solid'].append(s)
assert len(groups['wall_solid']) == 4, groups
gmsh.model.addPhysicalGroup(3, [fluid], name='fluid')
for k, v in groups.items():
    gmsh.model.addPhysicalGroup(2, v, name=k)
gmsh.option.setNumber('Mesh.MeshSizeMin', size * 0.5); gmsh.option.setNumber('Mesh.MeshSizeMax', size)
gmsh.option.setNumber('Mesh.Algorithm', 6); gmsh.option.setNumber('Mesh.Algorithm3D', 1); gmsh.option.setNumber('Mesh.Optimize', 1)
gmsh.model.mesh.generate(3)
ntet = sum(len(e) for e in gmsh.model.mesh.getElements(3)[1])
gmsh.option.setNumber('Mesh.MshFileVersion', 4.1); gmsh.option.setNumber('Mesh.Binary', 0); gmsh.option.setNumber('Mesh.SaveAll', 0)
gmsh.write('tester_tet.msh')
json.dump({'box': [LX, LY, LZ], 'tet_vertices': TET, 'v_tet': v_tet, 'v_fluid': v_fluid, 'v_box': LX * LY * LZ, 'tets': ntet,
           'patches': {k: len(v) for k, v in groups.items()}}, open('tester_tet_geometry.json', 'w'), indent=1)
print('tester: %d tets, fluid volume %.6f = box %.1f - tet %.6f (check %.2e)' % (ntet, v_fluid, LX * LY * LZ, v_tet, v_fluid - (LX * LY * LZ - v_tet)))
gmsh.finalize()
