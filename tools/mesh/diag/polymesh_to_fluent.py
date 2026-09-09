#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
# polyMesh (as ofgpu-convert-mesh writes it) -> ANSYS Fluent ASCII mesh (.msh).
#   python polymesh_to_fluent.py <caseDir> <out.msh> [name=type ...]
# Zone types default from the patch type (wall -> wall, patch -> pressure-outlet) and from the
# names this project uses (east, nh3_source -> velocity-inlet); override with name=type where
# type is one of wall, velocity-inlet, pressure-inlet, pressure-outlet, outflow, symmetry,
# interior. Fluent lets you change zone types after reading anyway.
#
# The format is ANSYS's, documented in the ANSYS FLUENT 12.0 User's Guide, Appendix B "Mesh File
# Format" (public mirror: afs.enea.it/project/neptunius/docs/fluent/html/ug/node1471.htm):
#   - every index in the file is 1-based and hexadecimal
#   - B.3.7: "if you curl the fingers of your right hand in the order of the nodes, your thumb
#     will point toward c1"; a polyMesh face's node order points from owner to neighbour (or out
#     of the domain), so internal faces carry c0 = owner, c1 = neighbour and boundary faces
#     c0 = owner, c1 = 0, both with the polyMesh node order unchanged
# Fluent zone-type numbers: interior 2, wall 3, pressure-inlet 4, pressure-outlet 5, symmetry 7,
# velocity-inlet 10, outflow 36.
import sys, os, re, time
import numpy as np

case, out = sys.argv[1], sys.argv[2]
overrides = dict(a.split('=', 1) for a in sys.argv[3:])
TYPE_NO = {'interior': 2, 'wall': 3, 'pressure-inlet': 4, 'pressure-outlet': 5, 'symmetry': 7, 'velocity-inlet': 10, 'outflow': 36}
PM = os.path.join(case, 'constant', 'polyMesh')
T0 = time.time()


def log(m):
    print('%6.1f s  %s' % (time.time() - T0, m), flush=True)


def body(name):
    """The text between the count and the closing paren of a polyMesh list file, plus the count."""
    t = open(os.path.join(PM, name), encoding='utf-8', errors='replace').read()
    t = re.sub(r'/\*.*?\*/', ' ', t, flags=re.S)
    t = re.sub(r'//[^\n]*', ' ', t)
    i = t.index('}') + 1                               # end of the FoamFile header
    m = re.search(r'(\d+)\s*\(', t[i:])
    n = int(m.group(1))
    start = i + m.end()
    end = t.rindex(')')
    return n, t[start:end]


n_pts, txt = body('points')
pts = np.fromstring(txt.replace('(', ' ').replace(')', ' '), sep=' ', dtype=np.float64).reshape(-1, 3)
assert len(pts) == n_pts, (len(pts), n_pts)
log('points %d' % n_pts)
n_faces, txt = body('faces')
raw = np.fromstring(txt.replace('(', ' ').replace(')', ' '), sep=' ', dtype=np.int64)
# each face: count then nodes; all faces of a tet mesh are triangles
if len(raw) == 4 * n_faces and (raw[::4] == 3).all():
    faces = raw.reshape(-1, 4)[:, 1:]
else:
    faces = None
    sizes = []
    pos = 0
    rows = []
    while pos < len(raw):
        k = int(raw[pos]); rows.append(raw[pos + 1:pos + 1 + k]); sizes.append(k); pos += 1 + k
    assert len(rows) == n_faces
log('faces %d (%s)' % (n_faces, 'all triangles' if faces is not None else 'mixed'))
n_own, txt = body('owner'); owner = np.fromstring(txt, sep=' ', dtype=np.int64); assert len(owner) == n_own == n_faces
n_nei, txt = body('neighbour'); neigh = np.fromstring(txt, sep=' ', dtype=np.int64); assert len(neigh) == n_nei
n_int = n_nei
n_cells = int(max(owner.max(), neigh.max())) + 1
log('owner/neighbour: %d cells, %d internal faces, %d boundary faces' % (n_cells, n_int, n_faces - n_int))
btxt = open(os.path.join(PM, 'boundary'), encoding='utf-8', errors='replace').read()
patches = re.findall(r'\n\s*([A-Za-z_][\w]*)\s*\{\s*type\s+(\w+);[^}]*?nFaces\s+(\d+);\s*startFace\s+(\d+);', btxt)
assert patches, 'no patches parsed from boundary'


def zone_type(name, ptype):
    if name in overrides:
        return overrides[name]
    if ptype == 'wall':
        return 'wall'
    if name in ('east', 'inlet') or name.endswith('_source') or name.startswith('inlet'):
        return 'velocity-inlet'
    if ptype in ('symmetry', 'symmetryPlane'):
        return 'symmetry'
    return 'pressure-outlet'


zones = []
zid = 10                                             # foamMeshToFluent numbers the patches from 10
for name, ptype, nf, sf in patches:
    zt = zone_type(name, ptype)
    assert zt in TYPE_NO, 'unknown Fluent zone type %r for %s' % (zt, name)
    zones.append((zid, name, zt, int(sf), int(nf)))
    zid += 1
    log('  zone %2d  %-20s %-16s %8d faces' % (zid - 1, name, zt, int(nf)))

with open(out, 'w', encoding='ascii', newline='\n') as f:
    w = f.write
    w('(0 "Fluent mesh written by polymesh_to_fluent.py from %s")\n' % os.path.basename(os.path.abspath(case)))
    # the declarations exactly as foamMeshToFluent writes them (fluentFvMesh.C)
    w('(0 "Dimension:")\n(2 3)\n(0 "Grid dimensions:")\n')
    w('(10 (0 1 %x 0 3))\n' % n_pts)
    w('(12 (0 1 %x 0 0))\n' % n_cells)
    w('(13 (0 1 %x 0 0))\n' % n_faces)
    # nodes
    w('(10 (1 1 %x 1 3)(\n' % n_pts)
    for i in range(0, n_pts, 200000):
        w('\n'.join('%.9g %.9g %.9g' % (x, y, z) for x, y, z in pts[i:i + 200000]))
        w('\n')
    w('))\n')
    log('nodes written')
    # cells: one zone of tets (element type 2); mixed meshes would need a per-cell type list
    if faces is None:
        raise SystemExit('mixed-element polyMesh: not handled by this writer')
    # ANSYS FLUENT 12.0 User's Guide B.3.7: "curl the fingers of your right hand in the order
    # of the nodes, your thumb will point toward c1". A polyMesh face's node order points from
    # the owner to the neighbour, so c0 = owner, c1 = neighbour, nodes unchanged.
    # Element type 0 (mixed): every line starts with its node count.
    w('(13 (2 1 %x 2 0)(\n' % n_int)
    F = faces[:n_int] + 1
    c0 = owner[:n_int] + 1
    c1 = neigh + 1
    for i in range(0, n_int, 500000):
        blk = np.column_stack([F[i:i + 500000], c0[i:i + 500000], c1[i:i + 500000]])
        w('\n'.join('3 %x %x %x %x %x' % tuple(r) for r in blk))
        w('\n')
    w('))\n')
    log('interior faces written')
    # boundary faces: the polyMesh node order already points outwards, so c0 = owner and the
    # outside is c1 = 0 (the same ANSYS rule)
    for zid, name, zt, sf, nf in zones:
        w('(13 (%x %x %x %x 0)(\n' % (zid, sf + 1, sf + nf, TYPE_NO[zt]))
        Fb = faces[sf:sf + nf] + 1
        cb = owner[sf:sf + nf] + 1
        blk = np.column_stack([Fb, cb])
        w('\n'.join('3 %x %x %x %x 0' % tuple(r) for r in blk))
        w('\n))\n')
    log('boundary faces written')
    # the cell zone: mixed element type with one type per cell (2 = tet), as foamMeshToFluent
    w('(12 (1 1 %x 1 0)(\n' % n_cells)
    for i in range(0, n_cells, 40):
        w(' '.join(['2'] * min(40, n_cells - i)) + '\n')
    w('))\n')
    log('cell types written')
    # zone names in the (39 ...) form foamMeshToFluent uses (decimal ids there)
    w('(39 (1 fluid fluid)())\n')
    w('(39 (2 interior interior)())\n')
    for zid, name, zt, sf, nf in zones:
        w('(39 (%d %s %s)())\n' % (zid, zt, name))
log('wrote %s (%.0f MB)' % (out, os.path.getsize(out) / 1e6))
