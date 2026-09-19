#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.

"""stl_repair - repair an STL to watertight where it can be, and report
what it could not.  The repair is written to a NEW file: the Rust reader's
weld stays bit-exact (*DESIGN*, SPEC-LIT §23.1), so an epsilon weld has to
live outside the files the reader is pointed at.

    python tools/geom/stl_repair.py <file.stl> [--out <path.stl>] [--json <out.json>] [--weld REL] [--max-hole-edges N] [--ascii | --binary]
    python tools/geom/geom_tool.py repair <file.stl> [--out <path.stl>] [--json <out.json>] [--weld REL] [--max-hole-edges N] [--ascii | --binary]

What is repaired, in order: a bit-exact weld is measured first, duplicate
corners are welded at a tolerance (a coordinate a cluster already had is
kept), degenerate triangles are dropped, each component is oriented by
propagation across edges the input uses exactly twice - a flip that would
make one of the triangle's own edges same-direction is skipped and counted,
not applied - and flipped outward when closed, a boundary loop of up to
--max-hole-edges edges is filled (three edges with one triangle, four or
more by ear clipping in the plane of the loop's Newell normal), and the
result is written with recomputed normals.  The defect counts after every
stage and everything the tool could not repair are reported, never hidden.
"""

import argparse
import json
import math
import os
import re
import struct
import sys

import numpy as np
from scipy.spatial import cKDTree

BIN_HEADER = 80
BIN_TRI_BYTES = 50
TOOL = 'stl_repair'


def die(msg, code=1):
    sys.stderr.write('%s: %s\n' % (TOOL, msg))
    sys.stderr.flush()
    raise SystemExit(code)


# --- reading -------------------------------------------------------------
# Detection is SPEC-LIT §23.1's: ASCII iff the bytes start with "solid"
# (case-insensitive) AND the ASCII grammar parses; else binary, refused
# unless the size is exactly 84 + 50*n.

_ASCII_FLOAT = r'[-+0-9.eE]+'
_FACET = re.compile(r'facet\s+normal\s+(%s)\s+(%s)\s+(%s)' % ((_ASCII_FLOAT,) * 3), re.I)
_VERTEX = re.compile(r'vertex\s+(%s)\s+(%s)\s+(%s)' % ((_ASCII_FLOAT,) * 3), re.I)


def _parse_ascii(text, stem, origin):
    """The triangles and `solid` names of an ASCII STL; None when the text
    is not ASCII STL (the caller then falls back to binary)."""
    tris, patches = [], []
    cur = None            # (name, [vertices so far])
    pending_normal = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith('solid'):
            if cur is not None:
                return None
            name = line[5:].strip()
            cur = (name if name else stem, [])
        elif low.startswith('endsolid'):
            if cur is None or cur[1]:
                return None
            cur = None
        elif low.startswith('facet'):
            if cur is None or not _FACET.match(line):
                return None
            pending_normal = True
        elif low.startswith('outer'):
            if not pending_normal or low != 'outer loop':
                return None
        elif low.startswith('vertex'):
            m = _VERTEX.match(line)
            if cur is None or not pending_normal or not m:
                return None
            cur[1].append((float(m.group(1)), float(m.group(2)), float(m.group(3))))
            if len(cur[1]) == 3:
                pending_normal = False
        elif low.startswith('endloop'):
            pass
        elif low.startswith('endfacet'):
            if cur is None or len(cur[1]) != 3:
                return None
            tris.append((cur[0], tuple(cur[1])))
            cur = (cur[0], [])
        else:
            return None
    if cur is not None:
        return None
    return tris


def _parse_binary(bytes_, stem, origin):
    """The triangles of a binary STL as (patch_name, tri) pairs; the count
    must determine the size exactly (rust/src/surface/stl.rs's rule)."""
    if len(bytes_) < BIN_HEADER + 4:
        return None
    n = struct.unpack_from('<I', bytes_, BIN_HEADER)[0]
    expected = BIN_HEADER + 4 + BIN_TRI_BYTES * n
    if len(bytes_) != expected:
        die('%s is neither binary STL (84 + 50*n bytes; the file is %d) '
            'nor ASCII STL (binary declares %d triangles, which requires '
            'exactly %d bytes)' % (origin, len(bytes_), n, expected), 1)
    tris = []
    for t in range(n):
        base = BIN_HEADER + 4 + BIN_TRI_BYTES * t + 12   # stored normal ignored
        v = struct.unpack_from('<9f', bytes_, base)
        tris.append((stem, ((v[0], v[1], v[2]), (v[3], v[4], v[5]), (v[6], v[7], v[8]))))
    return tris


def read_stl(path, stem):
    """(tris, format_in): tris is a list of (patch_name, ((x,y,z),)*3)."""
    with open(path, 'rb') as f:
        bytes_ = f.read()
    ascii_reason = 'the file does not start with "solid"'
    if bytes_[:5].lower() == b'solid':
        text = bytes_.decode('utf-8', errors='replace')
        tris = _parse_ascii(text, stem, path)
        if tris is not None:
            return tris, 'stl-ascii'
        ascii_reason = 'the text does not follow the facet grammar'
    tris = _parse_binary(bytes_, stem, path)
    if tris is None:
        die('%s is neither binary STL (84 + 50*n bytes; the file is %d) '
            'nor ASCII STL (%s)' % (path, len(bytes_), ascii_reason), 1)
    return tris, 'stl-binary'


# --- the weld (tolerance from the house convention, 1e-6 x diagonal) -----
# SPEC-LIT §23.1 keeps bit-exact welding in the Rust reader (*DESIGN*): this
# tool welds at a tolerance into a NEW file and prints what it moved.

class _DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def _weld(raw, tol_abs):
    """(points, tri_idx, tri_bit, stats) - raw is the (3n, 3) float64 corner
    array; tri_bit is the bit-exact weld the before numbers measure.  The
    representative of a cluster is the raw point of lowest raw index."""
    nb_exact_keys = raw.view(np.uint64).reshape(-1, 3)
    uniq, first, inv = np.unique(nb_exact_keys, axis=0, return_index=True,
                                 return_inverse=True)
    nb = len(uniq)
    tri_bit = inv.reshape(-1, 3)     # the bit-exact weld: what before measures
    rep_of = np.arange(nb)           # unique-point id -> representative unique id
    merged = 0
    if tol_abs > 0:
        dsu = _DSU(nb)
        # the tree runs on the float coordinates, never on the bit patterns
        for i, j in cKDTree(raw[first]).query_pairs(r=tol_abs):
            dsu.union(i, j)
        root = np.array([dsu.find(i) for i in range(nb)])
        best = {}
        for i in range(nb):
            r = root[i]
            if r not in best or first[i] < first[best[r]]:
                best[r] = i
        rep_of = np.array([best[root[i]] for i in range(nb)])
        merged = nb - len(set(best.values()))
    uids = np.unique(rep_of)         # the representative of each cluster, ascending
    points = raw[first[uids]].copy()  # a coordinate the file already had
    corner_rep = np.searchsorted(uids, rep_of)[inv]   # raw corner -> cluster id
    tri_idx = corner_rep.reshape(-1, 3)
    # the distance every raw corner moved to its representative
    moved = np.linalg.norm(raw - points[corner_rep], axis=1)
    max_move = float(moved.max()) if len(moved) else 0.0
    return points, tri_idx, tri_bit, {'points_bit_exact': int(nb),
        'points_welded': int(len(uids)),
        'merged': int(merged), 'max_move': max_move}


# --- closedness (E2) and components --------------------------------------
# The rule restated in numpy from rust/src/surface/mod.rs edge_defects
# (SPEC-LIT §23.2): per undirected edge, one appearance is open, two with
# opposite directions is closed, anything else is non-manifold.

def _edge_key(T, nP):
    a = T[:, [0, 1, 2]].reshape(-1)
    b = T[:, [1, 2, 0]].reshape(-1)
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    return a, b, lo * nP + hi


def _defects(T, nP):
    """(open, non_manifold) over the triangles T on nP points."""
    if len(T) == 0:
        return 0, 0
    a, b, key = _edge_key(T, nP)
    _, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    fwd = np.bincount(inv[a < b], minlength=len(counts))
    tot = fwd + (counts - fwd)
    open_e = int(np.count_nonzero(tot == 1))
    ok = np.count_nonzero((tot == 2) & (fwd == 1) & (counts - fwd == 1))
    return open_e, int(len(counts) - open_e - ok)


def _area_normals(T, P):
    pa, pb, pc = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    cr = np.cross(pb - pa, pc - pa)
    return 0.5 * np.linalg.norm(cr, axis=1), cr


def _components(T, nP):
    """Vertex-connected components, numbered by their lowest triangle."""
    dsu = _DSU(nP)
    for t in T:
        dsu.union(int(t[0]), int(t[1]))
        dsu.union(int(t[1]), int(t[2]))
    roots = np.array([dsu.find(int(t[0])) for t in T])
    uniq, first, inv = np.unique(roots, return_index=True, return_inverse=True)
    rank = np.empty(len(uniq), dtype=np.int64)
    rank[np.argsort(first)] = np.arange(len(uniq))
    return rank[inv]          # component id per triangle


def _flip_makes_same(T, t2, tri2, uk, ustart, ucount, nP):
    """True iff flipping triangle t2 would turn one of its own two-triangle
    edges from an opposite pair into a same-direction pair - the only move
    that raises non_manifold_edges.  An open edge stays open and an edge
    shared three ways or more stays non-manifold whatever the winding, so
    neither ever costs anything."""
    for k in range(3):
        a, b = int(tri2[k]), int(tri2[(k + 1) % 3])
        kk = min(a, b) * nP + max(a, b)
        i = np.searchsorted(uk, kk)
        if i >= len(uk) or uk[i] != kk or ucount[i] != 2:
            continue
        for e in ustart[i] + np.arange(2):
            t3 = int(e) // 3
            if t3 == t2:
                continue
            tri3 = T[t3]
            if not [j for j in range(3)
                    if (int(tri3[j]), int(tri3[(j + 1) % 3])) == (a, b)]:
                return True    # the partner runs b->a today; after the flip both would
    return False


def _orient(T, P, nP):
    """Per component a BFS from its largest-area triangle flips every
    triangle that traverses a shared two-triangle edge the same way.  The
    BFS walks only edges the input uses exactly twice, so a flip decision
    is never taken through a non-manifold edge, and a flip is applied only
    when it does not raise the non-manifold count: a triangle whose flip
    would make one of its own edges same-direction is left alone, counted
    in left_alone.  A component's remainder the walk did not reach from
    the previous seed is given a fresh seed of its own, counted in
    reseeded_patches.  Returns (flips, left_alone, left_alone_reasons,
    reseeded_patches)."""
    areas, _ = _area_normals(T, P)
    comp = _components(T, nP)
    ncomp = int(comp.max()) + 1 if len(comp) else 0
    _, _, key = _edge_key(T, nP)
    uk, ustart, ucount = np.unique(key, return_index=True, return_counts=True)
    visited = np.zeros(len(T), dtype=bool)
    flips = 0
    refused = 0
    refused_reason = 'flip would make an edge same-direction'
    reseeded = 0
    for c in range(ncomp):
        members = np.nonzero(comp == c)[0]
        first = True
        while True:
            todo = members[~visited[members]]
            if len(todo) == 0:
                break
            seed = int(todo[np.argmax(areas[todo])])
            if not first:
                reseeded += 1
            first = False
            visited[seed] = True
            stack = [seed]
            while stack:
                t = stack.pop()
                tri = T[t]
                for k in range(3):
                    ca, cb = int(tri[k]), int(tri[(k + 1) % 3])
                    kk = min(ca, cb) * nP + max(ca, cb)
                    i = np.searchsorted(uk, kk)
                    if i >= len(uk) or uk[i] != kk or ucount[i] != 2:
                        continue
                    for e in ustart[i] + np.arange(2):
                        t2 = int(e) // 3
                        if visited[t2]:
                            continue
                        tri2 = T[t2]
                        pos = [j for j in range(3) if (int(tri2[j]), int(tri2[(j + 1) % 3])) == (ca, cb)]
                        if pos:
                            if _flip_makes_same(T, t2, tri2, uk, ustart, ucount, nP):
                                refused += 1     # the flip would break an opposite edge
                            else:
                                T[t2] = tri2[[0, 2, 1]]
                                flips += 1
                        visited[t2] = True
                        stack.append(t2)
    reasons = [refused_reason] if refused > 0 else []
    return flips, refused, reasons, reseeded


# --- holes and outward ----------------------------------------------------

def _clip_loop(P, loop):
    """Ear-clip a simple boundary loop in its own plane.  The polygon
    normal is the Newell normal n = sum_i p_i x p_(i+1) (indices mod m);
    an ear at p_i is a corner convex about n whose ear triangle, projected
    onto the plane orthogonal to n, holds no other loop vertex strictly
    inside it - Meisters's two-ears theorem (G. H. Meisters, "Polygons
    have ears", Amer. Math. Monthly 82 (1975) 648-651) guarantees a simple
    polygon has one.  loop holds the m distinct point indices in walk
    order WITHOUT the repeated first point.  Returns m - 2 index triples,
    each in reversed walk winding, or None when the Newell normal has zero
    length or a full pass over the working copy finds no ear."""
    m = len(loop)
    pts = P[np.asarray(loop, dtype=np.int64)]
    n = np.cross(pts, np.roll(pts, -1, axis=0)).sum(axis=0)
    ln = float(np.linalg.norm(n))
    if ln == 0.0:
        return None
    nh = n / ln
    ref = np.array([1.0, 0.0, 0.0])
    if abs(float(ref @ nh)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = ref - float(ref @ nh) * nh
    u /= np.linalg.norm(u)
    v = np.cross(nh, u)
    p3 = pts.tolist()
    q = (pts @ np.array([u, v]).T).tolist()
    nh3 = nh.tolist()
    work = list(range(m))
    tris = []
    while len(work) > 3:
        ear, L = -1, len(work)
        for i in range(L):
            im, ic, ip = work[i - 1], work[i], work[(i + 1) % L]
            ax, ay, az = p3[im]
            bx, by, bz = p3[ic]
            cx, cy, cz = p3[ip]
            ux, uy, uz = bx - ax, by - ay, bz - az
            wx, wy, wz = cx - bx, cy - by, cz - bz
            if (uy * wz - uz * wy) * nh3[0] + (uz * wx - ux * wz) * nh3[1] \
                    + (ux * wy - uy * wx) * nh3[2] <= 0.0:
                continue
            x0, y0 = q[im]
            e1x, e1y = q[ic][0] - x0, q[ic][1] - y0
            e2x, e2y = q[ip][0] - x0, q[ip][1] - y0
            det = e1x * e2y - e1y * e2x
            if abs(det) <= 1e-12 * (e1x * e1x + e1y * e1y
                                    + e2x * e2x + e2y * e2y):
                continue    # a flat projected triangle contains nothing
            inside = False
            for j in work:
                if j in (im, ic, ip):
                    continue
                dx, dy = q[j][0] - x0, q[j][1] - y0
                uu = (dx * e2y - dy * e2x) / det
                vv = (e1x * dy - e1y * dx) / det
                if uu > 0.0 and vv > 0.0 and uu + vv < 1.0:
                    inside = True
                    break
            if not inside:
                ear = i
                break
        if ear < 0:
            return None
        im, ic, ip = work[ear - 1], work[ear], work[(ear + 1) % len(work)]
        tris.append((loop[ip], loop[ic], loop[im]))
        del work[ear]
    tris.append((loop[work[2]], loop[work[1]], loop[work[0]]))
    return tris


def _fill_holes(T, P, tri_patch, nP, max_hole):
    """(T, tri_patch, filled, filled_triangles, unfilled, per_hole): walk
    the open directed edges, fill 3-edge loops with one triangle and
    ear-clip loops of up to max_hole edges, report every other loop.  A
    fill is applied only if it leaves every one of its new triangles'
    undirected edges used at most twice (the use tally is counted once,
    before the walk; a clipped loop's whole candidate fan is tallied
    before any of it is committed)."""
    a, b, key = _edge_key(T, nP)
    uk, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    use = dict(zip(uk.tolist(), counts.tolist()))
    is_open = counts[inv] == 1
    open_e = np.nonzero(is_open)[0]
    out_map = {}
    for e in open_e:
        out_map.setdefault(int(a[e]), []).append(int(e))
    used = set()
    filled, filled_tris, unfilled, per_hole = 0, 0, [], []
    newT, newP = [], []
    for e0 in open_e:
        e0 = int(e0)
        if e0 in used:
            continue
        p0, p1 = int(a[e0]), int(b[e0])
        used.add(e0)
        loop, dead = [p0, p1], None
        while loop[-1] != p0:
            outs = out_map.get(loop[-1], [])
            if len(outs) != 1 or outs[0] in used:
                dead = (len(loop) - 1,
                        'boundary vertex with %d outgoing open edges' % len(outs))
                break
            ne = outs[0]
            used.add(ne)
            loop.append(int(b[ne]))
        if dead is not None:
            unfilled.append({'edges': dead[0], 'reason': dead[1]})
            per_hole.append({'edges': dead[0], 'outcome': 'skipped',
                             'triangles': 0, 'reason': dead[1]})
            continue
        m = len(loop) - 1
        if m > max_hole:
            unfilled.append({'edges': m, 'reason':
                             'loop of %d edges > max_hole_edges %d' % (m, max_hole)})
            per_hole.append({'edges': m, 'outcome': 'skipped', 'triangles': 0,
                             'reason': 'loop of %d edges > max_hole_edges %d'
                                       % (m, max_hole)})
        elif m == 3:
            keys = [min(loop[i], loop[i + 1]) * nP + max(loop[i], loop[i + 1])
                    for i in range(3)]
            bad = sum(1 for k in keys if use[k] >= 2)
            if bad:
                unfilled.append({'edges': m, 'reason':
                                 'filling it would make %d edge(s) non-manifold' % bad})
                per_hole.append({'edges': m, 'outcome': 'skipped', 'triangles': 0,
                                 'reason': 'filling it would make %d edge(s) non-manifold'
                                           % bad})
                continue
            owner = int(e0) // 3
            newT.append((loop[2], loop[1], loop[0]))
            newP.append(int(tri_patch[owner]))
            for k in keys:
                use[k] += 1
            filled += 1
            filled_tris += 1
            per_hole.append({'edges': m, 'outcome': 'filled',
                             'triangles': 1, 'reason': None})
        else:
            tris = _clip_loop(P, loop[:-1])
            if tris is None:
                unfilled.append({'edges': m, 'reason':
                                 'no ear found: the loop is not simple in its own plane'})
                per_hole.append({'edges': m, 'outcome': 'skipped', 'triangles': 0,
                                 'reason': 'no ear found: the loop is not simple '
                                           'in its own plane'})
                continue
            cand = {}
            for t in tris:
                for i, j in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                    k = min(i, j) * nP + max(i, j)
                    cand[k] = cand.get(k, 0) + 1
            bad = sum(1 for k in cand if use.get(k, 0) + cand[k] > 2)
            if bad:
                unfilled.append({'edges': m, 'reason':
                                 'filling it would make %d edge(s) non-manifold' % bad})
                per_hole.append({'edges': m, 'outcome': 'skipped', 'triangles': 0,
                                 'reason': 'filling it would make %d edge(s) '
                                           'non-manifold' % bad})
                continue
            owner = int(e0) // 3
            newT.extend(tris)
            newP.extend([int(tri_patch[owner])] * len(tris))
            for k, c in cand.items():
                use[k] = use.get(k, 0) + c
            filled += 1
            filled_tris += m - 2
            per_hole.append({'edges': m, 'outcome': 'filled',
                             'triangles': m - 2, 'reason': None})
    if newT:
        T = np.vstack([T, np.array(newT, dtype=T.dtype)])
        tri_patch = np.concatenate([tri_patch, np.array(newP, dtype=tri_patch.dtype)])
    return T, tri_patch, filled, filled_tris, unfilled, per_hole


def _flip_outward(T, P, nP):
    """(E4): a closed component with negative divergence-theorem volume is
    flipped whole.  Returns (T, flipped_component_ids, comp_id_per_tri)."""
    comp = _components(T, nP)
    ncomp = int(comp.max()) + 1 if len(comp) else 0
    flipped = set()
    for c in range(ncomp):
        members = np.nonzero(comp == c)[0]
        pa, pb, pc = P[T[members, 0]], P[T[members, 1]], P[T[members, 2]]
        vol = float(np.einsum('ij,ij->i', pa, np.cross(pb, pc)).sum() / 6.0)
        if _defects(T[members], nP) == (0, 0) and vol < 0:
            T[members] = T[members][:, [0, 2, 1]]
            flipped.add(c)
    return T, flipped, comp


# --- writing: normals recomputed from the winding, never the stored ones --

def _normals(T, P):
    _, cr = _area_normals(T, P)
    n = np.linalg.norm(cr, axis=1)
    out = np.zeros((len(T), 3))
    ok = n > 0
    out[ok] = cr[ok] / n[ok, None]
    return out


def write_binary(path, T, P, stem):
    with open(path, 'wb') as f:
        f.write(('stl_repair %s' % stem).encode('ascii', 'replace')[:80].ljust(80, b'\0'))
        f.write(struct.pack('<I', len(T)))
        nr = _normals(T, P)
        for t in range(len(T)):
            f.write(struct.pack('<3f', *nr[t]))
            for k in range(3):
                f.write(struct.pack('<3f', *P[T[t, k]]))
            f.write(struct.pack('<H', 0))


def write_ascii(path, T, P, tri_patch, patch_names):
    parts = []
    for p, name in enumerate(patch_names):
        sel = np.nonzero(tri_patch == p)[0]
        if len(sel) == 0:
            continue
        nr = _normals(T[sel], P)
        lines = ['solid %s' % name]
        for i, t in enumerate(sel):
            lines.append('facet normal %s' % ' '.join('%.17g' % v for v in nr[i]))
            lines.append('outer loop')
            for k in range(3):
                lines.append('vertex %s' % ' '.join('%.17g' % v for v in P[T[t, k]]))
            lines.append('endloop')
            lines.append('endfacet')
        lines.append('endsolid %s' % name)
        parts.append('\n'.join(lines))
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(parts) + '\n')


# --- the command ---------------------------------------------------------

def add_arguments(parser):
    parser.add_argument('file', help='a .stl file, binary or ASCII')
    parser.add_argument('--out', metavar='PATH', default=None,
                        help='the repaired STL; without it the tool only reports')
    parser.add_argument('--json', metavar='OUT',
                        help='write the report to this file, never stdout')
    parser.add_argument('--weld', type=float, default=1e-6,
                        help='weld tolerance as a fraction of the bounding-box '
                             'diagonal (default 1e-6; 0 = bit-exact only)')
    parser.add_argument('--max-hole-edges', type=int, default=32,
                        dest='max_hole_edges',
                        help='fill a boundary loop of at most this many edges (default 32)')
    g = parser.add_mutually_exclusive_group()
    g.add_argument('--ascii', action='store_true',
                   help="force the output format (default: the input's)")
    g.add_argument('--binary', action='store_true',
                   help="force the output format (default: the input's)")


def _patch_names(tris):
    names = []
    for name, _ in tris:
        if name not in names:
            names.append(name)
    return names


def run_repair(args):
    path = args.file
    if not os.path.isfile(path):
        die('input file does not exist: %s' % path, 2)
    ext = os.path.splitext(path)[1].lower()
    if ext != '.stl':
        die("unsupported extension '%s' of %s (I read .stl only)" % (ext, path), 2)
    if args.out is not None:
        oext = os.path.splitext(args.out)[1].lower()
        if oext != '.stl':
            die("unsupported extension '%s' of %s (I read .stl only)" % (oext, args.out), 2)
        if os.path.abspath(args.out) == os.path.abspath(path):
            die('--out %s is the input file; refusing to overwrite it' % args.out, 2)
    if args.weld < 0:
        die('--weld must be >= 0 (got %s)' % args.weld, 2)
    if args.max_hole_edges < 3:
        die('--max-hole-edges must be >= 3 (got %d)' % args.max_hole_edges, 2)
    stem = os.path.splitext(os.path.basename(path))[0]
    tris, fmt = read_stl(path, stem)
    patch_names = _patch_names(tris)
    if args.binary and len(patch_names) > 1:
        die('--binary carries one patch (the file stem, SPEC-LIT \xa723.1); '
            '%s has %d: %s' % (os.path.basename(path), len(patch_names),
                               ', '.join(patch_names)), 2)
    n_in = len(tris)
    raw = np.array([v for _, t in tris for v in t], dtype=np.float64)
    nf = int(np.count_nonzero(~np.isfinite(raw)))
    if nf:
        die('%d non-finite coordinate(s) in %s' % (nf, path), 1)
    tri_patch0 = np.array([patch_names.index(name) for name, _ in tris], dtype=np.int64)
    bbox_lo, bbox_hi = raw.min(axis=0), raw.max(axis=0)
    diag = float(np.linalg.norm(bbox_hi - bbox_lo))
    tol_abs = args.weld * diag if diag > 0 else 1e-9
    # degenerate first: zero area at float64 exactly, on the raw corners
    corners = raw.reshape(-1, 3, 3)
    cr = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    keep = ~np.all(cr == 0.0, axis=1)
    dropped = int(np.count_nonzero(~keep))
    raw_k = raw.reshape(-1, 3, 3)[keep].reshape(-1, 3)
    patch_k = tri_patch0[keep]
    P, T, Tbit, wst = _weld(raw_k, tol_abs)
    nP_bit = wst['points_bit_exact']
    dup = np.all(np.sort(T, axis=1)[:, 1:] != np.sort(T, axis=1)[:, :-1], axis=1)
    T, Tbit, patch_k = T[dup], Tbit[dup], patch_k[dup]
    dropped += int(np.count_nonzero(~dup))
    T = T.astype(np.int64)
    Tbit = Tbit.astype(np.int64)
    nP = len(P)
    before = _defects(Tbit, nP_bit)
    weld_stage = _defects(T, nP)
    flip3, left_alone, alone_reasons, reseeded = _orient(T, P, nP)
    orient_stage = _defects(T, nP)
    T, tri_patch, filled, filled_tris, unfilled, per_hole = _fill_holes(
        T, P, patch_k, nP, args.max_hole_edges)
    T, flipped_set, comp = _flip_outward(T, P, nP)
    after_open, after_nm = _defects(T, nP)
    stages = [
        {'stage': 'input', 'open_edges': before[0],
         'non_manifold_edges': before[1]},
        {'stage': 'weld', 'open_edges': weld_stage[0],
         'non_manifold_edges': weld_stage[1]},
        {'stage': 'orient', 'open_edges': orient_stage[0],
         'non_manifold_edges': orient_stage[1]},
        {'stage': 'fill', 'open_edges': after_open,
         'non_manifold_edges': after_nm},
    ]
    if after_nm > before[1]:
        die('the fill raised non_manifold_edges from %d to %d - this is a bug, '
            'report it' % (before[1], after_nm), 3)
    after_closed = after_open == 0 and after_nm == 0
    areas, _ = _area_normals(T, P)
    ncomp = int(comp.max()) + 1 if len(comp) else 0
    comps, vol_sum, all_closed = [], 0.0, True
    for c in range(ncomp):
        members = np.nonzero(comp == c)[0]
        o, nm = _defects(T[members], nP)
        closed = o == 0 and nm == 0
        vol = None
        if closed:
            pa, pb, pc = P[T[members, 0]], P[T[members, 1]], P[T[members, 2]]
            vol = float(np.einsum('ij,ij->i', pa, np.cross(pb, pc)).sum() / 6.0)
            vol_sum += vol
        else:
            all_closed = False
        seed = int(members[np.argmax(areas[members])])
        comps.append({'index': c, 'patch': patch_names[int(tri_patch[seed])],
                      'n_triangles': int(len(members)), 'open_edges': o,
                      'non_manifold_edges': nm, 'closed': closed,
                      'volume': vol, 'flipped': c in flipped_set})
    after_vol = vol_sum if all_closed and ncomp else None
    fmt_out = 'stl-binary' if args.binary else ('stl-ascii' if args.ascii else fmt)
    out_base = os.path.basename(args.out) if args.out else None
    rep = {'version': 1, 'tool': 'stl_repair', 'file': os.path.basename(path),
           'format_in': fmt, 'out': out_base, 'format_out': fmt_out if args.out else None,
           'units': 'file',
           'bbox': [float(v) for v in (*bbox_lo, *bbox_hi)], 'diagonal': diag,
           'triangles_in': n_in, 'triangles_out': int(len(T)),
           'degenerate_dropped': dropped,
           'weld': dict({'tol_rel': args.weld, 'tol_abs': tol_abs,
                         'points_raw': 3 * n_in}, **wst),
           'orientation': {'reoriented_triangles': flip3,
                           'left_alone': left_alone,
                           'left_alone_reason': alone_reasons,
                           'reseeded_patches': reseeded,
                           'flipped_components': len(flipped_set)},
           'holes': {'max_hole_edges': args.max_hole_edges, 'filled': filled,
                     'filled_triangles': filled_tris, 'unfilled': unfilled,
                     'per_hole': per_hole},
           'stages': stages,
           'before': {'open_edges': before[0], 'non_manifold_edges': before[1],
                      'closed': before == (0, 0)},
           'after': {'open_edges': after_open, 'non_manifold_edges': after_nm,
                     'closed': after_closed, 'volume': after_vol},
           'n_components': ncomp, 'components': comps,
           'patches': list(patch_names)}
    if args.json:
        with open(args.json, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(rep, f, indent=1)
            f.write('\n')
    sys.stdout.write('%s %s: %s, %d triangle(s), %d component(s), before: %d open edge(s), %d non-manifold edge(s)\n'
                     % (TOOL, rep['file'], fmt, n_in, ncomp, before[0], before[1]))
    sys.stdout.write('  weld: %d point(s) merged at %.6g (%g x diagonal %.6g), max move %.3g; %d degenerate dropped\n'
                     % (wst['merged'], tol_abs, args.weld, diag, wst['max_move'], dropped))
    sys.stdout.write('  orient: %d triangle(s) reoriented, %d left alone, %d patch(es) re-seeded, %d component(s) flipped outward\n'
                     % (flip3, left_alone, reseeded, len(flipped_set)))
    ear_tris = sum(h['triangles'] for h in per_hole
                   if h['outcome'] == 'filled' and h['triangles'] > 1)
    sys.stdout.write('  holes: %d filled with %d triangle(s), %d unfilled, %d by ear clipping\n'
                     % (filled, filled_tris, len(unfilled), ear_tris))
    sys.stdout.write('  after: %d open edge(s), %d non-manifold edge(s) -> closed %s, volume %s\n'
                     % (after_open, after_nm, 'yes' if after_closed else 'no',
                        '%.9e' % after_vol if after_vol is not None else '-'))
    for st in stages:
        sys.stdout.write('  stage %s: %d open edge(s), %d non-manifold edge(s)\n'
                         % (st['stage'], st['open_edges'], st['non_manifold_edges']))
    if args.out:
        if fmt_out == 'stl-binary':
            write_binary(args.out, T, P, stem)
        else:
            write_ascii(args.out, T, P, tri_patch, patch_names)
        sys.stdout.write('  wrote %s (%s, %d triangle(s))\n'
                         % (out_base, fmt_out, len(T)))
    else:
        sys.stdout.write('  nothing written (no --out)\n')
    if not after_closed:
        for c in [k for k in comps if not k['closed']][:20]:
            sys.stdout.write('  component %d (%s, %d triangle(s)): %d open, %d non-manifold\n'
                             % (c['index'], c['patch'], c['n_triangles'],
                                c['open_edges'], c['non_manifold_edges']))
    sys.stdout.flush()
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog='stl_repair', description=__doc__.splitlines()[0])
    add_arguments(parser)
    return run_repair(parser.parse_args(argv))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.exit(main())
