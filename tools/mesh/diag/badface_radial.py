#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
# Radial (distance from the pool centre) and per-box distribution of the faces above 85 deg,
# using the angles saved by nonorth_faces.py.
#   python badface_radial.py <case> pool_x pool_y
import os, re, sys
import numpy as np
case = sys.argv[1]; px, py = float(sys.argv[2]), float(sys.argv[3])
pm = os.path.join(case, 'constant', 'polyMesh')
ang = np.load(os.path.join(case, 'nonorth_deg.npy'))


def read_list(path):
    with open(path, 'rb') as f:
        data = f.read()
    m = re.search(rb'\n(\d+)\s*\n\(', data)
    n = int(m.group(1))
    body = data[m.end():]
    body = body[:body.rfind(b')')]
    return np.fromstring(body.replace(b'(', b' ').replace(b')', b' '), sep=' ').reshape(n, -1)


pts = read_list(os.path.join(pm, 'points'))
faces = read_list(os.path.join(pm, 'faces')).astype(np.int64)[:, 1:]
bad = np.where(ang > 85)[0]
fc = pts[faces[bad]].mean(axis=1)
r = np.hypot(fc[:, 0] - px, fc[:, 1] - py)
print('faces > 85 deg: %d; > 87: %d; > 89: %d' % (len(bad), (ang[bad] > 87).sum(), (ang[bad] > 89).sum()))
edges = [0, 100, 200, 300, 400, 500, 600, 800, 1000, 1200, 1800]
h, _ = np.histogram(r, bins=edges)
for a, b, n in zip(edges[:-1], edges[1:], h):
    print('  %4d..%4d m from the pool: %4d faces (%d > 87)' % (a, b, n, ((r >= a) & (r < b) & (ang[bad] > 87)).sum()))
for half in (500, 600, 700, 800):
    inside = (np.abs(fc[:, 0] - px) < half) & (np.abs(fc[:, 1] - py) < half)
    print('  box +-%d m around the pool would keep %d of the > 85 deg faces (%d > 87)' % (
        half, inside.sum(), (inside & (ang[bad] > 87)).sum()))
print('  z of the > 85 faces: min %.1f  median %.1f  max %.1f' % (fc[:, 2].min(), np.median(fc[:, 2]), fc[:, 2].max()))
