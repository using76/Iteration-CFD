#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
# An independent check of an ANSYS Fluent ASCII mesh: parse it from scratch, rebuild every cell
# from its faces and verify geometry and topology against the tester's known answer.
#   python fluent_check.py <mesh.msh> [geometry.json]
# Checks: header counts vs bodies; every face's cells exist; each cell closes (sum of its outward
# face vectors ~ 0) and has positive volume; total volume = box - tetrahedron; boundary faces have
# exactly one cell and their right-hand normal points OUT of the domain; the c0/c1 orientation
# convention actually used (ANSYS B.3.7: thumb toward c1) is measured, not assumed; every zone's
# faces lie on the plane or solid they are named after.
import re, sys, json
import numpy as np

path = sys.argv[1]
geo = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else None
txt = open(path, encoding='ascii', errors='replace').read()
sections = re.findall(r'\((\d+)\s*\(([^()]*)\)(?:\s*\(([^)]*)\))?\)?', txt)   # rough: header then optional body


def hexs(s):
    return [int(x, 16) for x in s.split()]


# nodes
m = re.search(r'\(10 \([0-9a-f]+ 1 ([0-9a-f]+) 1 3\)\s*\((.*?)\)\s*\)', txt, flags=re.S)
n_nodes = int(m.group(1), 16)
nodes = np.fromstring(m.group(2), sep=' ').reshape(-1, 3)
assert len(nodes) == n_nodes, (len(nodes), n_nodes)
decl_nodes = int(re.search(r'\(10 \(0 1 ([0-9a-f]+) 0', txt).group(1), 16)
decl_cells = int(re.search(r'\(12 \(0 1 ([0-9a-f]+) 0', txt).group(1), 16)
decl_faces = int(re.search(r'\(13 \(0 1 ([0-9a-f]+) 0', txt).group(1), 16)
assert decl_nodes == n_nodes
# zones (39 ...) or (45 ...)
zones = {int(z, 10): (t, n) for z, t, n in re.findall(r'\((?:39|45) \((\d+) ([\w-]+) ([\w-]+)\)\(\)\)', txt)}
# face sections
faces = []            # (zone, nodes[list], c0, c1)
for hdr, body in re.findall(r'\(13 \(([0-9a-f]+ [0-9a-f]+ [0-9a-f]+ [0-9a-f]+ [0-9a-f]+)\)\s*\((.*?)\n\s*\)\s*\)', txt, flags=re.S):
    zid, first, last, btype, etype = hexs(hdr)
    rows = [hexs(l) for l in body.strip().split('\n') if l.strip()]
    assert len(rows) == last - first + 1, (zid, len(rows), last - first + 1)
    for r in rows:
        if etype == 0:
            k = r[0]; nd, c0, c1 = r[1:1 + k], r[1 + k], r[2 + k]
        else:
            nd, c0, c1 = r[:etype], r[etype], r[etype + 1]
        faces.append((zid, [v - 1 for v in nd], c0, c1))
assert len(faces) == decl_faces, (len(faces), decl_faces)
n_cells = decl_cells
print('%s: %d nodes, %d faces, %d cells, zones %s' % (path.split('/')[-1], n_nodes, len(faces), n_cells, {z: v for z, v in sorted(zones.items())}))

# geometry per face
S = np.zeros((len(faces), 3)); C = np.zeros((len(faces), 3))
for i, (z, nd, c0, c1) in enumerate(faces):
    P = nodes[nd]; c = P.mean(axis=0); s = np.zeros(3)
    for a in range(len(nd)):
        s += 0.5 * np.cross(P[a] - c, P[(a + 1) % len(nd)] - c)
    S[i] = s; C[i] = c
# which convention does the file follow? ANSYS: the right-hand normal points toward c1.
# Test on internal faces with the cell centroids (approximate centroid = mean of face centres).
cell_faces = [[] for _ in range(n_cells + 1)]
for i, (z, nd, c0, c1) in enumerate(faces):
    if c0: cell_faces[c0].append(i)
    if c1: cell_faces[c1].append(i)
cent = np.zeros((n_cells + 1, 3))
for c in range(1, n_cells + 1):
    cent[c] = C[cell_faces[c]].mean(axis=0)
internal = [i for i, f in enumerate(faces) if f[2] and f[3]]
toward_c1 = sum(1 for i in internal if np.dot(S[i], cent[faces[i][3]] - cent[faces[i][2]]) > 0)
print('internal faces %d: right-hand normal points toward c1 on %d, toward c0 on %d  -> convention: %s' % (
    len(internal), toward_c1, len(internal) - toward_c1, 'ANSYS (thumb -> c1)' if toward_c1 == len(internal) else ('inverted (thumb -> c0)' if toward_c1 == 0 else 'MIXED - BROKEN')))
rule_c1 = toward_c1 > len(internal) / 2
# closure and volume per cell, using the measured convention: outward for the cell the normal points AWAY from
vol = np.zeros(n_cells + 1); closure = np.zeros(n_cells + 1)
acc = np.zeros((n_cells + 1, 3))
for i, (z, nd, c0, c1) in enumerate(faces):
    n_to = c1 if rule_c1 else c0                  # the cell the normal points INTO
    n_from = c0 if rule_c1 else c1                # the cell the normal points OUT of
    if n_from: acc[n_from] += S[i]; vol[n_from] += np.dot(C[i], S[i]) / 3
    if n_to: acc[n_to] -= S[i]; vol[n_to] -= np.dot(C[i], S[i]) / 3
closure = np.linalg.norm(acc[1:], axis=1) / np.maximum(np.abs(vol[1:]), 1e-300) ** (2 / 3)
print('cells: volume min %.3e max %.3e total %.6f; negative %d; closure max %.2e' % (vol[1:].min(), vol[1:].max(), vol[1:].sum(), (vol[1:] <= 0).sum(), closure.max()))
ok = (vol[1:] > 0).all() and closure.max() < 1e-9
# boundary faces: one cell, normal outward
bnd = [i for i, f in enumerate(faces) if (f[2] == 0) != (f[3] == 0)]
outward = 0
for i in bnd:
    z, nd, c0, c1 = faces[i]; c = c0 or c1
    outward += np.dot(S[i], C[i] - cent[c]) > 0
print('boundary faces %d: normal points out of its cell on %d; c1 == 0 on %d' % (len(bnd), outward, sum(1 for i in bnd if faces[i][3] == 0)))
ok = ok and outward == len(bnd)
if geo:
    v_expect = geo['v_fluid']
    print('expected fluid volume %.6f (box %.1f - tet %.6f): difference %.2e' % (v_expect, geo['v_box'], geo['v_tet'], vol[1:].sum() - v_expect))
    ok = ok and abs(vol[1:].sum() - v_expect) < 1e-6 * v_expect
    LX, LY, LZ = geo['box']
    planes = {'west': (0, 0.0), 'east': (0, LX), 'south': (1, 0.0), 'north': (1, LY), 'bottom': (2, 0.0), 'top': (2, LZ)}
    for zid, (t, name) in zones.items():
        fi = [i for i, f in enumerate(faces) if f[0] == zid]
        if not fi: continue
        if name in planes:
            ax, val = planes[name]; dev = max(abs(nodes[v][ax] - val) for i in fi for v in faces[i][1])
            print('  zone %-11s %5d faces on plane %s=%g (max deviation %.1e) type %s' % (name, len(fi), 'xyz'[ax], val, dev, t)); ok = ok and dev < 1e-6
        elif name == 'wall_solid':
            T = np.array(geo['tet_vertices']); inside = all(T.min(axis=0)[k] - 1e-6 <= nodes[v][k] <= T.max(axis=0)[k] + 1e-6 for i in fi for v in faces[i][1] for k in range(3))
            print('  zone %-11s %5d faces, all nodes inside the tetrahedron bbox: %s, type %s' % (name, len(fi), inside, t)); ok = ok and inside
        elif name in ('fluid', 'interior'):
            print('  zone %-11s %5d faces, type %s' % (name, len(fi), t))
print('RESULT:', 'PASS' if ok else 'FAIL')
