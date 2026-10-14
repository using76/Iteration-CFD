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
kept), degenerate triangles are dropped, verbatim duplicate facets are
removed - exact copies, and coincident opposite pairs unless one of the
two runs against the surrounding surface - a vertex lying on another
triangle's open edge within the weld tolerance splits that edge (a
T-junction), and each component is oriented
patch by patch: a patch is a maximal set of triangles joined by edges the
input uses twice in opposite directions, so the windings inside a patch
already agree, and a reached patch is flipped whole when more of its
edges to the decided patches run same-direction than opposite.  Then a
closed component is flipped outward when its volume is negative, a
boundary loop of up to --max-hole-edges edges is filled (three edges with
one triangle, four or more by ear clipping in the plane of the loop's
Newell normal), and the result is written with recomputed normals.  The
defect counts after every stage and everything the tool could not repair
are reported, never hidden.
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


def _edge_map(key):
    """Group the flat edge indices by undirected edge key.  key[3*t + k] is
    the key of triangle t's corner k, so an edge's users are scattered all
    over it: np.unique(..., return_index=True) finds only the FIRST of
    them, which is why the partner has to be looked up here instead.
    Returns (uk, eidx, estart, ucount): uk holds the distinct keys
    ascending, and eidx[estart[i] : estart[i] + ucount[i]] holds the flat
    index 3*t + k of EVERY corner that uses uk[i], ascending.  eidx is a
    permutation of range(len(key)), so every index a caller reads is a
    real corner of a real triangle and t = e // 3 is always in range."""
    uk, inv = np.unique(key, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    eidx = np.argsort(inv, kind='stable').astype(np.int64)
    ucount = np.bincount(inv, minlength=len(uk)).astype(np.int64)
    estart = np.zeros(len(uk), dtype=np.int64)
    np.cumsum(ucount[:-1], out=estart[1:])
    return uk, eidx, estart, ucount


def _dedup(T, tri_patch, nP):
    """(T, tri_patch, n_exact, n_opposite): drop coincident triangles.  A
    group is the set of triangles sharing one sorted vertex triple, and
    two of them run the SAME way iff one is a cyclic rotation of the
    other.  Within a group only the first triangle (lowest row index) of
    each way survives; the rest are exact duplicates (n_exact).  A group
    that then holds one triangle of each way is a coincident opposite
    pair.  With o_k the outside triangles on the pair's k-th undirected
    edge - all o_k counted on one post-exact snapshot, the same for every
    pair - a pair whose every o_k is even is a zero-thickness sheet and
    both triangles go; otherwise the triangle running opposite to the
    outside users on more edges survives (on an edge with a single
    outside user that is the one traversing the edge the other way round;
    a tie keeps the lower row index).  The removals of this step are
    n_opposite and the survivors keep their relative order."""
    n = len(T)
    if n == 0:
        return T, tri_patch, 0, 0
    srt = np.sort(T, axis=1)
    _, inv = np.unique(srt, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    v0, v1 = T[:, 0], T[:, 1]
    even = ((v0 == srt[:, 0]) & (v1 == srt[:, 1])) \
        | ((v0 == srt[:, 1]) & (v1 == srt[:, 2])) \
        | ((v0 == srt[:, 2]) & (v1 == srt[:, 0]))
    cls = (~even).astype(np.int64)          # 0 = cyclic way, 1 = reversed
    _, first = np.unique(inv * 2 + cls, return_index=True)
    keep = np.zeros(n, dtype=bool)
    keep[first] = True
    n_exact = int(np.count_nonzero(~keep))
    T2, patch2, inv2, cls2 = T[keep], tri_patch[keep], inv[keep], cls[keep]
    keep2 = np.ones(len(T2), dtype=bool)
    if np.any(np.bincount(inv2) == 2):       # coincident opposite pairs
        a2, _, key2 = _edge_key(T2, nP)
        uk2, eidx2, estart2, ucnt2 = _edge_map(key2)
        order = np.argsort(inv2, kind='stable')
        s_inv = inv2[order]
        bnd = np.nonzero(np.diff(s_inv))[0] + 1
        spans = zip(np.concatenate(([0], bnd)).tolist(),
                    np.concatenate((bnd, [len(s_inv)])).tolist())
        for s, e in spans:
            if e - s != 2:
                continue
            i, j = int(order[s]), int(order[s + 1])
            if cls2[i] == cls2[j]:
                continue
            even_all, si, sj = True, 0, 0
            for k in range(3):
                pos = int(np.searchsorted(uk2, key2[3 * i + k]))
                if ucnt2[pos] % 2:
                    even_all = False         # o_k = ucnt2 - 2 is odd
                users = eidx2[estart2[pos]:estart2[pos] + ucnt2[pos]]
                out = users[(users // 3 != i) & (users // 3 != j)]
                if len(out) == 0:
                    continue
                tails = a2[out]
                opp_i = int(np.count_nonzero(tails != a2[3 * i + k]))
                same_i = len(tails) - opp_i
                if opp_i > same_i:
                    si += 1
                elif same_i > opp_i:
                    sj += 1
            if even_all:
                keep2[i] = keep2[j] = False  # a zero-thickness sheet: both go
            elif sj > si:
                keep2[i] = False
            else:
                keep2[j] = False             # i wins; a tie keeps the lower row
    n_opposite = int(np.count_nonzero(~keep2))
    return T2[keep2], patch2[keep2], n_exact, n_opposite


# --- T-junctions (AUT-02b) -------------------------------------------------

TJ_MAX_PASSES = 16


def _split_t_junctions(T, P, tri_patch, nP, tol_abs):
    """(T, tri_patch, stats): a vertex lying on another triangle's edge,
    within the weld tolerance, splits that edge.  Only OPEN edges (one use)
    and only boundary vertices (an endpoint of an open edge) take part, so
    a closed surface is returned untouched.  Vertex m splits the open edge
    (a, b) of triangle (a, b, c) iff m is none of a, b, c, its parameter
    s = (m - a).(b - a) / |b - a|^2 lies in (0, 1), its distance to
    a + s (b - a) is <= tol_abs, and it is farther than tol_abs from a and
    from b.  The hits sorted by s, a = q_0, q_1 .. q_k, q_k+1 = b, replace
    the triangle in place by the fan (q_i, q_i+1, c), which keeps its
    winding and its patch.  A triangle splits at most one edge per pass
    (its lowest corner with a hit).  A fan is not made when one of its
    triangles has an exactly-zero cross product, when one of its new
    spokes (q_i, c), 1 <= i <= k, is already an edge (it would lay a
    triangle over one already there, as around a collinear sliver), or
    when a sub-edge (q_i, q_i+1) already has two uses - the uses counted
    on the pass's input plus the fans planned before it in the pass, the
    rule the hole fill applies, so no edge ends with three.  Passes repeat
    until none splits (at most TJ_MAX_PASSES).  stats counts the split
    edges, the vertex insertions, the triangles added and the passes that
    split."""
    stats = {'split_edges': 0, 'inserted_vertices': 0,
             'triangles_added': 0, 'passes': 0}
    for _ in range(TJ_MAX_PASSES):
        if len(T) == 0:
            break
        a, b, key = _edge_key(T, nP)
        uk, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
        inv = np.asarray(inv).reshape(-1)
        open_c = np.nonzero(counts[inv] == 1)[0]     # flat corner 3 t + k
        if len(open_c) == 0:
            break
        bverts = np.unique(np.concatenate((a[open_c], b[open_c])))
        tree = cKDTree(P[bverts])
        uses = dict(zip(uk.tolist(), counts.tolist()))   # undirected edge -> uses
        ekey = lambda p, q: min(p, q) * nP + max(p, q)
        plan = {}
        for e in open_c.tolist():
            t, k = divmod(e, 3)
            if t in plan:
                continue
            ia, ib, ic = int(a[e]), int(b[e]), int(T[t, (k + 2) % 3])
            pa, pb = P[ia], P[ib]
            d = pb - pa
            L2 = float(d @ d)
            if L2 == 0.0:
                continue
            hits = []
            for j in sorted(tree.query_ball_point(0.5 * (pa + pb),
                                                  0.5 * math.sqrt(L2) + tol_abs)):
                m = int(bverts[j])
                if m in (ia, ib, ic):
                    continue
                pm = P[m]
                s = float((pm - pa) @ d) / L2
                if not 0.0 < s < 1.0:
                    continue
                if np.linalg.norm(pm - (pa + s * d)) > tol_abs:
                    continue
                if np.linalg.norm(pm - pa) <= tol_abs or \
                        np.linalg.norm(pm - pb) <= tol_abs:
                    continue
                hits.append((s, m))
            if not hits:
                continue
            hits.sort()
            chain = [ia] + [m for _, m in hits] + [ib]
            fan = [(chain[i], chain[i + 1], ic) for i in range(len(chain) - 1)]
            if any(np.all(np.cross(P[q1] - P[q0], P[qc] - P[q0]) == 0.0)
                   for q0, q1, qc in fan):
                continue
            spokes = [ekey(m, ic) for _, m in hits]
            subs = [ekey(q0, q1) for q0, q1, _ in fan]
            if any(uses.get(x, 0) for x in spokes) or \
                    any(uses.get(x, 0) >= 2 for x in subs):
                continue
            for x in spokes:
                uses[x] = 2
            for x in subs:
                uses[x] = uses.get(x, 0) + 1
            uses[ekey(ia, ib)] -= 1
            plan[t] = fan
        if not plan:
            break
        rows, patches = [], []
        for t in range(len(T)):
            pieces = plan.get(t)
            if pieces is None:
                rows.append(T[t])
                patches.append(tri_patch[t])
            else:
                rows.extend(np.array(f, dtype=np.int64) for f in pieces)
                patches.extend([tri_patch[t]] * len(pieces))
        added = sum(len(f) - 1 for f in plan.values())
        T = np.array(rows, dtype=np.int64).reshape(-1, 3)
        tri_patch = np.array(patches, dtype=tri_patch.dtype)
        stats['split_edges'] += len(plan)
        stats['inserted_vertices'] += added
        stats['triangles_added'] += added
        stats['passes'] += 1
    return T, tri_patch, stats


def _orient(T, P, nP):
    """Per component the walk now runs over PATCHES, not single triangles:
    a patch is a maximal set of triangles joined by two-use edges whose
    two runs oppose in the input (one union-find pass, on the input
    windings), so the windings inside a patch already agree and every
    two-use edge between two different patches runs the same way in the
    input - which is why a decided patch can never raise
    non_manifold_edges.  Each component is seeded at its not-yet-decided
    patch with the largest total triangle area (tie: the patch holding the
    lowest triangle index); the seed is never flipped and every seed after
    a component's first is counted in reseeded_patches.  A patch first
    reached counts S same-way and O opposite-way two-use edges to the
    patches already decided: S > O flips the patch whole (flips grows by
    its triangle count); S > 0 with S <= O leaves it as it is (left_alone
    grows by its triangle count).  Returns (flips, left_alone,
    left_alone_reasons, reseeded_patches)."""
    areas, _ = _area_normals(T, P)
    comp = _components(T, nP)
    ncomp = int(comp.max()) + 1 if len(comp) else 0
    a, _, key = _edge_key(T, nP)
    _, eidx, estart, ucount = _edge_map(key)
    two = np.nonzero(ucount == 2)[0]
    dsu = _DSU(len(T))
    for i in two:
        e1, e2 = int(eidx[estart[i]]), int(eidx[estart[i] + 1])
        if int(a[e1]) != int(a[e2]):     # the two runs oppose in the input
            dsu.union(e1 // 3, e2 // 3)
    roots = np.array([dsu.find(t) for t in range(len(T))], dtype=np.int64)
    if len(roots) == 0:
        return 0, 0, [], 0
    up, first, pinv = np.unique(roots, return_index=True, return_inverse=True)
    rank = np.empty(len(up), dtype=np.int64)
    rank[np.argsort(first)] = np.arange(len(up))
    patch = rank[np.asarray(pinv).reshape(-1)]
    npatch = int(len(up))
    members = [np.nonzero(patch == p)[0] for p in range(npatch)]
    parea = np.array([float(areas[m].sum()) for m in members])
    adj = [[] for _ in range(npatch)]
    for i in two:
        p1 = int(patch[int(eidx[estart[i]]) // 3])
        p2 = int(patch[int(eidx[estart[i] + 1]) // 3])
        if p1 != p2:                     # inter-patch: same way in the input
            adj[p1].append(p2)
            adj[p2].append(p1)
    decided = np.zeros(npatch, dtype=bool)
    state = np.zeros(npatch, dtype=bool) # True = the patch was flipped whole
    flips = left_alone = reseeded = 0
    refused_reason = 'flip would make an edge same-direction'
    for c in range(ncomp):
        cpatch = np.unique(patch[comp == c]).tolist()
        seen = 0
        first_seed = True
        while seen < len(cpatch):
            seed, best = -1, -1.0
            for p in cpatch:
                if not decided[p] and parea[p] > best:
                    seed, best = p, float(parea[p])
            if not first_seed:
                reseeded += 1
            first_seed = False
            decided[seed] = True         # the seed is never flipped
            seen += 1
            stack = [seed]
            while stack:
                for q in adj[stack.pop()]:
                    if decided[q]:
                        continue
                    s = sum(1 for r in adj[q] if decided[r] and not state[r])
                    o = sum(1 for r in adj[q] if decided[r] and state[r])
                    if s > o:
                        T[members[q]] = T[members[q]][:, [0, 2, 1]]
                        state[q] = True
                        flips += len(members[q])
                    elif s > 0:
                        left_alone += len(members[q])
                    decided[q] = True
                    seen += 1
                    stack.append(q)
    reasons = [refused_reason] if left_alone > 0 else []
    return flips, left_alone, reasons, reseeded


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
    before any of it is committed).  A candidate triangle with zero area
    at float64 - the same exact test the input degenerate drop applies -
    is refused with its whole loop, so the tool never writes a triangle
    it would itself drop on the way back in."""
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
            zc = np.cross(P[loop[1]] - P[loop[0]], P[loop[2]] - P[loop[0]])
            if bool(np.all(zc == 0.0)):
                unfilled.append({'edges': m, 'reason':
                                 'filling it would add 1 zero-area triangle(s)'})
                per_hole.append({'edges': m, 'outcome': 'skipped', 'triangles': 0,
                                 'reason': 'filling it would add 1 zero-area '
                                           'triangle(s)'})
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
            ta = np.asarray(tris, dtype=np.int64)
            zc = np.cross(P[ta[:, 1]] - P[ta[:, 0]], P[ta[:, 2]] - P[ta[:, 0]])
            nz = int(np.count_nonzero(np.all(zc == 0.0, axis=1)))
            if nz:
                unfilled.append({'edges': m, 'reason':
                                 'filling it would add %d zero-area triangle(s)' % nz})
                per_hole.append({'edges': m, 'outcome': 'skipped', 'triangles': 0,
                                 'reason': 'filling it would add %d zero-area '
                                           'triangle(s)' % nz})
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
    T, patch_k, n_exact, n_opposite = _dedup(T, patch_k, nP)
    dedup_stage = _defects(T, nP)
    T, patch_k, tj = _split_t_junctions(T, P, patch_k, nP, tol_abs)
    tj_stage = _defects(T, nP)
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
        {'stage': 'dedup', 'open_edges': dedup_stage[0],
         'non_manifold_edges': dedup_stage[1]},
        {'stage': 'tjunction', 'open_edges': tj_stage[0],
         'non_manifold_edges': tj_stage[1]},
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
           'duplicates': {'exact_removed': n_exact,
                          'opposite_removed': n_opposite},
           't_junctions': tj,
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
    sys.stdout.write('  duplicates: %d exact, %d opposite-coincident removed\n'
                     % (n_exact, n_opposite))
    sys.stdout.write('  t-junctions: %d edge(s) split at %d vertex insertion(s), %d triangle(s) added\n'
                     % (tj['split_edges'], tj['inserted_vertices'], tj['triangles_added']))
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
