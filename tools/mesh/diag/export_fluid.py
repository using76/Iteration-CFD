#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
# Export the fluid region (one solid) of a step_mesh checkpoint as STEP (mm, like the original
# STEP) and as a binary STL surface mesh (metres).
#   python export_fluid.py <fluid.brep> <out_basename> [--trim z] [--stl-size m]
import sys, os, time, gmsh
src = sys.argv[1]; base = sys.argv[2]
trim = None; stl_size = 2.0; stl_only = False
a = sys.argv[3:]
while a:
    k = a.pop(0)
    if k == '--trim': trim = float(a.pop(0))
    elif k == '--stl-size': stl_size = float(a.pop(0))
    elif k == '--stl-only': stl_only = True
t0 = time.time()
gmsh.initialize(); gmsh.option.setNumber('General.Terminal', 0)
gmsh.option.setNumber('Geometry.OCCScaling', 1.0)
gmsh.model.occ.importShapes(src); gmsh.model.occ.synchronize()
vols = gmsh.model.getEntities(3)
print('%s: %d volume(s)' % (os.path.basename(src), len(vols)))
if trim is not None:
    bb = gmsh.model.getBoundingBox(-1, -1)
    box = gmsh.model.occ.addBox(bb[0] - 10, bb[1] - 10, bb[2] - 10, (bb[3] - bb[0]) + 20, (bb[4] - bb[1]) + 20, trim - (bb[2] - 10))
    gmsh.model.occ.cut(vols, [(3, box)], removeObject=True, removeTool=True)
    gmsh.model.occ.synchronize()
    vols = gmsh.model.getEntities(3)
    if len(vols) > 1:
        keep = max(vols, key=lambda dt: gmsh.model.occ.getMass(3, dt[1]))
        gmsh.model.occ.remove([v for v in vols if v != keep], recursive=True)
        gmsh.model.occ.synchronize(); vols = gmsh.model.getEntities(3)
    print('trimmed below z = %.2f: %d volume(s)' % (trim, len(vols)))
mass = sum(gmsh.model.occ.getMass(3, t) for d, t in vols)
nsurf = len(gmsh.model.getEntities(2))
bb = gmsh.model.getBoundingBox(-1, -1)
print('fluid: %.4e m^3, %d faces, bbox x[%.1f,%.1f] y[%.1f,%.1f] z[%.2f,%.2f]' % (mass, nsurf, bb[0], bb[3], bb[1], bb[4], bb[2], bb[5]))
if not stl_only:
    # STEP in millimetres (OCC's STEP writer declares mm), so the file reads back at the right size
    gmsh.model.occ.dilate(gmsh.model.getEntities(), 0, 0, 0, 1000, 1000, 1000)
    gmsh.model.occ.synchronize()
    gmsh.write(base + '.step')
    print('wrote %s.step (%.0f MB, mm) in %.0f s' % (base, os.path.getsize(base + '.step') / 1e6, time.time() - t0), flush=True)
    gmsh.model.occ.dilate(gmsh.model.getEntities(), 0, 0, 0, 0.001, 0.001, 0.001)
    gmsh.model.occ.synchronize()
# STL: a surface mesh in metres
gmsh.option.setNumber('Mesh.MeshSizeMin', stl_size / 4)
gmsh.option.setNumber('Mesh.MeshSizeMax', stl_size)
gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 8)
gmsh.option.setNumber('Mesh.Algorithm', 6)
gmsh.option.setNumber('Mesh.MaxNumThreads2D', 8)
gmsh.model.mesh.generate(2)
ntri = sum(len(e) for e in gmsh.model.mesh.getElements(2)[1])
gmsh.option.setNumber('Mesh.Binary', 1)
gmsh.option.setNumber('Mesh.SaveAll', 1)
gmsh.write(base + '.stl')
print('wrote %s.stl (%d triangles, %.0f MB, metres) in %.0f s' % (base, ntri, os.path.getsize(base + '.stl') / 1e6, time.time() - t0))
gmsh.finalize()
