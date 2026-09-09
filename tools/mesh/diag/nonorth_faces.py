#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
# Non-orthogonality of every internal face of a triangle-faced polyMesh, and where the worst ones are.
#   python nonorth_faces.py <case> [pool_x pool_y]
import os, re, sys, math
import numpy as np
case = sys.argv[1]
px, py = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 else (18.5, -7.5)
pm = os.path.join(case, 'constant', 'polyMesh')


def read_list(path, vector=False):
    with open(path, 'rb') as f:
        data = f.read()
    m = re.search(rb'\n(\d+)\s*\n\(', data)
    n = int(m.group(1))
    body = data[m.end():]
    body = body[:body.rfind(b')')]
    if vector:
        return np.fromstring(body.replace(b'(', b' ').replace(b')', b' '), sep=' ').reshape(n, -1)
    return np.fromstring(body, sep=' ')


pts = read_list(os.path.join(pm, 'points'), vector=True)
faces = read_list(os.path.join(pm, 'faces'), vector=True).astype(np.int64)
assert (faces[:, 0] == 3).all()
faces = faces[:, 1:]
owner = read_list(os.path.join(pm, 'owner')).astype(np.int64)
neigh = read_list(os.path.join(pm, 'neighbour')).astype(np.int64)
ncell = int(max(owner.max(), neigh.max())) + 1
P = pts[faces]
fc = P.mean(axis=1)
Sf = 0.5 * np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
area = np.linalg.norm(Sf, axis=1)
# cell centres: area-weighted mean of face centres (good enough to locate faces)
cc = np.zeros((ncell, 3)); w = np.zeros(ncell)
np.add.at(cc, owner, fc * area[:, None]); np.add.at(w, owner, area)
ni = len(neigh)
np.add.at(cc, neigh, fc[:ni] * area[:ni, None]); np.add.at(w, neigh, area[:ni])
cc /= w[:, None]
d = cc[neigh] - cc[owner[:ni]]
dn = np.linalg.norm(d, axis=1)
cosang = np.einsum('ij,ij->i', d, Sf[:ni]) / (dn * area[:ni])
ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
print('internal faces %d: non-orth max %.2f mean %.2f; >70: %d, >80: %d, >85: %d, >87: %d, >89: %d' % (
    ni, ang.max(), ang.mean(), (ang > 70).sum(), (ang > 80).sum(), (ang > 85).sum(), (ang > 87).sum(), (ang > 89).sum()))
bad = np.where(ang > 85)[0]
r = np.hypot(fc[bad, 0] - px, fc[bad, 1] - py)
print('faces > 85 deg: within 45 m of the pool centre: %d, elsewhere: %d' % ((r < 45).sum(), (r >= 45).sum()))
# cluster the bad faces on a 20 m grid
if len(bad):
    keys = {}
    for i in bad:
        k = (int(fc[i, 0] // 20) * 20, int(fc[i, 1] // 20) * 20)
        keys.setdefault(k, []).append(i)
    print('clusters (20 m cells) of faces > 85 deg, largest first:')
    for k, idx in sorted(keys.items(), key=lambda kv: -len(kv[1]))[:12]:
        zs = fc[idx, 2]
        print('   x %5d..%5d y %5d..%5d  %4d faces  z %.2f..%.2f  worst %.2f deg  |d| min %.3f m' % (
            k[0], k[0] + 20, k[1], k[1] + 20, len(idx), zs.min(), zs.max(), ang[idx].max(), dn[idx].min()))
order = bad[np.argsort(-ang[bad])[:10]]
print('worst 10 faces:')
for i in order:
    print('   face %-9d %.3f deg at (%.2f, %.2f, %.2f)  area %.4f m2  |d| %.4f m  cells %d/%d  r_pool %.0f' % (
        i, ang[i], fc[i, 0], fc[i, 1], fc[i, 2], area[i], dn[i], owner[i], neigh[i], math.hypot(fc[i, 0] - px, fc[i, 1] - py)))
np.save(os.path.join(case, 'nonorth_deg.npy'), ang.astype(np.float32))
