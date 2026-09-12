# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
#
# sections.py <caseDir> <x0> <y0> <z0> <out_dir> [--stl PATH] [--walls a,b]
#             [--half 160] [--ztop 70] [--dpi 150]
#
r"""Three sections of a REAL-POINT polyMesh written by ofgpu-automesher.

The section of the mesh by a plane coord[axis] = v is the set of segments where
the mesh's FACES cross that plane: the exact cell-edge network on the cut plane.
It needs no bounding boxes, no node planes and no assumption about cell shape -
each face is the actual polygon of the actual cells over real points, of any
length. An edge that straddles the plane is cut at the crossing; a face that
crosses gives an even number of hits, paired in the order found, one segment per
pair. A point exactly on the plane would leave an edge lying in it, so every
plane value is nudged by 1e-9 of the domain size once at the start.

Segments are coloured by the face they come from:
    internal faces                       thin grey  #606060, linewidth 0.35
    boundary faces on WALL patches       orange     #ff7f0e, linewidth 1.0
    boundary faces on any other patch    blue       #1f77b4, linewidth 0.8
Over the top, in black at linewidth 1.6, comes the STL's own cross-section by
the same plane: what shows whether the snap actually landed on the geometry.

Arguments:
    caseDir   the case; the mesh is <caseDir>/constant/polyMesh, and the case
              root, the constant directory or the polyMesh directory itself
              are accepted, as io::polymesh::read_poly_mesh accepts them
    x0 y0 z0  the three planes: z = z0 drawn as xy, x = x0 as yz, y = y0 as zx.
              Three PNGs in out_dir: mesh_xy_z<z0>.png, mesh_yz_x<x0>.png,
              mesh_zx_y<y0>.png
    --stl     ASCII STL for the black overlay; without it the overlay is
              simply absent and the log says so
    --walls   comma-separated patch names to draw as walls. Default: every
              patch whose type in the boundary file is wall, falling back to
              every patch NOT one of the six the octree emits (xMin, xMax,
              yMin, yMax, zMin, zMax) when the file types every patch patch
    --half    the window half-width in metres about (x0, y0), default 160
    --ztop    the top of the two vertical windows, default 70; they start at
              the mesh's own minimum z
    --dpi     PNG resolution, default 150
"""
import argparse
import os
import re
import sys
import time

try:
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError as e:
    sys.exit('sections.py: %s - numpy and matplotlib are required and are not vendored' % e)
from matplotlib.collections import LineCollection

ap = argparse.ArgumentParser(description='Three sections of an automesher polyMesh.',
                             epilog='example: python sections.py case 5 5 5 out --stl geom.stl --half 6 --ztop 10')
ap.add_argument('case', help='case root, constant directory or polyMesh directory')
ap.add_argument('x0', type=float, help='plane x = x0 (yz view)')
ap.add_argument('y0', type=float, help='plane y = y0 (zx view)')
ap.add_argument('z0', type=float, help='plane z = z0 (xy view)')
ap.add_argument('out', help='directory for the three PNGs')
ap.add_argument('--stl', help='ASCII STL for the black geometry overlay')
ap.add_argument('--walls', help='comma-separated patch names to draw as walls')
ap.add_argument('--half', type=float, default=160.0, help='window half-width about (x0, y0) [m]')
ap.add_argument('--ztop', type=float, default=70.0, help='top of the vertical windows [m]')
ap.add_argument('--dpi', type=int, default=150, help='PNG resolution')
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True)
t0 = time.time()


def say(m):
    print('[%5.1f s] %s' % (time.time() - t0, m), flush=True)


def body_after_count(text):
    """The list body of a FoamFile: (count on its own line, then the items)."""
    m = re.search(r'\n(\d+)\s*\n\(', text)
    if not m:
        sys.exit('sections.py: no "<count> (" header line found')
    return int(m.group(1)), text[m.end():text.rfind(')')]


def find_polymesh(case):
    """Accept the case root, the constant directory or the polyMesh directory."""
    for p in (os.path.join(case, 'constant', 'polyMesh'), os.path.join(case, 'polyMesh'), case):
        if os.path.isfile(os.path.join(p, 'points')):
            return p
    sys.exit('sections.py: no polyMesh under %s - expected <case>/constant/polyMesh/points' % case)


PM = find_polymesh(A.case)
say('reading points from %s' % PM)
n, b = body_after_count(open(os.path.join(PM, 'points'), encoding='ascii', errors='replace').read())
pts = np.fromstring(b.replace('(', ' ').replace(')', ' '), sep=' ').reshape(-1, 3)
assert len(pts) == n, (len(pts), n)
say('reading faces')
n, b = body_after_count(open(os.path.join(PM, 'faces'), encoding='ascii', errors='replace').read())
counts = np.array(re.findall(r'(\d+)\s*\(', b), dtype=np.int64)   # the N of every N(...)
if len(counts) != n:
    sys.exit('sections.py: the faces header says %d faces but %d face polygons were found' % (n, len(counts)))
if counts.min() < 3:
    sys.exit('sections.py: a face with %d vertices - a face polygon needs at least 3' % int(counts.min()))
nums = np.fromstring(b.replace('(', ' ').replace(')', ' '), sep=' ', dtype=np.int64)
if len(nums) != n + int(counts.sum()):
    sys.exit('sections.py: the faces body holds %d numbers, expected %d = %d counts + %d vertices'
             % (len(nums), n + int(counts.sum()), n, int(counts.sum())))
# the counts sit interleaved with the vertex ids, one slot before each face's ids: drop them
drop = np.arange(n) + np.concatenate(([0], np.cumsum(counts)[:-1]))
fv = np.delete(nums, drop)                        # flat vertex ids, face after face
fo = np.concatenate(([0], np.cumsum(counts)))     # offsets: face i is fv[fo[i]:fo[i+1]]
n_faces = n
_names = {3: 'triangles', 4: 'quads', 5: 'pentagons', 6: 'hexagons', 7: 'heptagons', 8: 'octagons'}
_vals, _cnts = np.unique(counts, return_counts=True)
say('faces: %s' % ', '.join('%d %s' % (c, _names.get(v, '%d-gons' % v)) for v, c in zip(_vals, _cnts)))

say('reading owner/neighbour/boundary')
n, b = body_after_count(open(os.path.join(PM, 'owner'), encoding='ascii', errors='replace').read())
owner = np.fromstring(b.replace('(', ' ').replace(')', ' '), sep=' ', dtype=np.int64)
n, b = body_after_count(open(os.path.join(PM, 'neighbour'), encoding='ascii', errors='replace').read())
neigh = np.fromstring(b.replace('(', ' ').replace(')', ' '), sep=' ', dtype=np.int64)
assert len(owner) == n_faces and len(neigh) < n_faces, (len(owner), len(neigh), n_faces)
n_int = len(neigh)                # startFace in the boundary file counts internal faces: absolute
n_cells = int(max(owner.max(), neigh.max())) + 1
btxt = open(os.path.join(PM, 'boundary'), encoding='utf-8', errors='replace').read()
patches = {}
for m in re.finditer(r'\n\s*(\w+)\s*\n\s*\{([^}]*)\}', btxt):
    ty = re.search(r'type\s+(\w+)\s*;', m.group(2))
    nf = re.search(r'nFaces\s+(\d+)\s*;', m.group(2))
    sf = re.search(r'startFace\s+(\d+)\s*;', m.group(2))
    if nf and sf:
        patches[m.group(1)] = (int(nf.group(1)), int(sf.group(1)), ty.group(1) if ty else '')
say('%d points, %d faces (%d internal), %d cells' % (len(pts), n_faces, n_int, n_cells))
say('patches: %s' % '; '.join('%s (type %s, %d faces)' % (k, v[2] or '?', v[0]) for k, v in patches.items()))

SIX = ('xMin', 'xMax', 'yMin', 'yMax', 'zMin', 'zMax')
if A.walls:
    walls = [s.strip() for s in A.walls.split(',') if s.strip()]
else:
    walls = sorted(k for k, v in patches.items() if v[2] == 'wall')
    if not walls:                 # every patch typed patch: fall back to the six-names rule
        walls = sorted(k for k in patches if k not in SIX)
missing = [w for w in walls if w not in patches]
if missing:
    sys.exit('sections.py: the wall names the boundary file does not have: %s' % ', '.join(missing))
say('wall patches: %s' % (', '.join(walls) if walls else 'none'))
cls = np.zeros(n_faces, np.int8)          # 0 internal, 1 wall boundary, 2 other boundary
cls[n_int:] = 2
for k, (nf, sf, _ty) in patches.items():
    if sf + nf > n_faces:
        sys.exit('sections.py: patch %s runs to face %d but the mesh has %d faces' % (k, sf + nf, n_faces))
    if k in walls:
        cls[sf:sf + nf] = 1

# every polygon edge, wrapped, in polygon order: flat arrays over the edges of all faces
_counts = np.diff(fo)
_idx = np.arange(len(fv))
fid = np.repeat(np.arange(n_faces), _counts)
pos = _idx - fo[:-1].repeat(_counts)
ea = fv[_idx]
eb = fv[np.where(pos == _counts[fid] - 1, fo[fid], _idx + 1)]


def section_segments(axis, v):
    """Segments where the mesh's faces cross coord[axis] = v, grouped by face class."""
    dv = pts[:, axis] - v                            # one offset per POINT: ea/eb are point ids
    da, db = dv[ea], dv[eb]
    hit = da * db < 0
    g = fid[hit]
    nh = np.bincount(g, minlength=n_faces)
    if (nh % 2 == 1).any():
        sys.exit('sections.py: a face crosses the plane v=%r an odd number of times' % v)
    s = (da[hit] / (da[hit] - db[hit]))[:, None]
    H = pts[ea[hit]] + s * (pts[eb[hit]] - pts[ea[hit]])
    _, first, inv = np.unique(g, return_index=True, return_inverse=True)
    loc = np.arange(len(g)) - first[inv]             # position of each hit within its face
    one = loc % 2 == 0
    ga, out = g[one], {}
    for k in (0, 1, 2):
        m = cls[ga] == k
        if m.any():
            out[k] = np.stack([H[one][m], H[~one][m]], axis=1)
    return out, int((nh > 0).sum())


def read_stl(path):
    """The ASCII STL's triangles, one (n, 3, 3) array over all its solids."""
    txt = open(path, encoding='ascii', errors='replace').read()
    tris, names = [], []
    for name, blk in re.findall(r'solid\s+(\w+)(.*?)endsolid', txt, flags=re.S):
        v = np.fromstring(blk.replace('vertex', ' ').replace('facet normal', ' ').replace('outer loop', ' ')
                          .replace('endloop', ' ').replace('endfacet', ' '), sep=' ').reshape(-1, 4, 3)[:, 1:, :]
        tris.append(v)
        names += [name] * len(v)
    if not tris:
        sys.exit('sections.py: no "solid" found in %s' % path)
    say('STL %s: %d triangles, solids %s' % (path, sum(len(t) for t in tris), sorted(set(names))))
    return np.vstack(tris)


def plane_segments(tri, axis, v):
    """Segments where triangles cross the plane coord[axis] = v."""
    segs = []
    d = tri[:, :, axis] - v
    cross = (np.sign(d).min(axis=1) < 0) & (np.sign(d).max(axis=1) > 0)
    for t, dd in zip(tri[cross], d[cross]):
        p2 = []
        for e0, e1 in ((0, 1), (1, 2), (2, 0)):
            if dd[e0] * dd[e1] < 0:
                s = dd[e0] / (dd[e0] - dd[e1])
                p2.append(t[e0] + s * (t[e1] - t[e0]))
        if len(p2) == 2:
            segs.append(np.array(p2))
    return segs


T = None
if A.stl:
    say('reading STL')
    T = read_stl(A.stl)
else:
    say('no --stl given: the black geometry overlay is not drawn')

COLS = ((0, '#606060', 0.35), (1, '#ff7f0e', 1.0), (2, '#1f77b4', 0.8))


def draw(axis, v, h_ax, v_ax, win_h, win_v, title, fname):
    """The section by plane coord[axis] = v onto axes (h_ax, v_ax) of the figure."""
    segs, ncut = section_segments(axis, v)
    stl_segs = plane_segments(T, axis, v) if T is not None else []
    fig, ax = plt.subplots(figsize=(12, max(3.2, 12 * (win_v[1] - win_v[0]) / (win_h[1] - win_h[0]) + 1.4)))
    for k, col, lw in COLS:
        if k in segs:
            ax.add_collection(LineCollection(segs[k][:, :, [h_ax, v_ax]], colors=col, linewidths=lw))
    if stl_segs:
        ax.add_collection(LineCollection([s[:, [h_ax, v_ax]] for s in stl_segs], colors='black', linewidths=1.6))
    ax.set_xlim(*win_h)
    ax.set_ylim(*win_v)
    ax.set_aspect('equal')
    ax.set_xlabel('xyz'[h_ax] + ' [m]')
    ax.set_ylabel('xyz'[v_ax] + ' [m]')
    ax.set_title(title)
    ax.text(0.0, -0.16 if v_ax == 2 else -0.05,
            'grey: internal faces   orange: wall patch   blue: domain sides   black: STL section'
            + ('' if T is not None else ' (no --stl: overlay absent)'),
            transform=ax.transAxes, fontsize=8, va='top', ha='left')
    fig.tight_layout()
    fig.savefig(os.path.join(A.out, fname), dpi=A.dpi)
    plt.close(fig)
    say('%s: %d faces cut, %d internal + %d wall + %d domain-side segments, %d STL segments'
        % (fname, ncut, len(segs.get(0, ())), len(segs.get(1, ())), len(segs.get(2, ())), len(stl_segs)))

# the nudge: a point exactly on a plane would leave an edge lying in it, so every
# plane value moves by 1e-9 of the domain size, once, before anything is cut
dom = float((pts.max(axis=0) - pts.min(axis=0)).max())
nudge = 1e-9 * dom
zmin = float(pts[:, 2].min())
x0, y0, z0 = A.x0 + nudge, A.y0 + nudge, A.z0 + nudge
hx, hy = A.half, A.half
say('windows: half %g m about (%g, %g), vertical z %.6g to %g' % (A.half, A.x0, A.y0, zmin, A.ztop))
draw(2, z0, 0, 1, (x0 - hx, x0 + hx), (y0 - hy, y0 + hy),
     'automesher mesh, plane z = %g m (xy)' % A.z0, 'mesh_xy_z%g.png' % A.z0)
draw(0, x0, 1, 2, (y0 - hy, y0 + hy), (zmin, A.ztop),
     'automesher mesh, plane x = %g m (yz)' % A.x0, 'mesh_yz_x%g.png' % A.x0)
draw(1, y0, 0, 2, (x0 - hx, x0 + hx), (zmin, A.ztop),
     'automesher mesh, plane y = %g m (zx)' % A.y0, 'mesh_zx_y%g.png' % A.y0)
say('done -> %s' % A.out)
