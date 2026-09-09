#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
# Extremes of U and p in a written foam time directory, with the cell centres of the extreme
# cells (tet mesh: every face is a triangle, so face centres are the mean of three points).
#   python field_extremes.py <case> <time> [pool_x pool_y]
import os, re, sys, math
import numpy as np

case, tdir = sys.argv[1], sys.argv[2]
px, py = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else (18.5, -7.5)
pm = os.path.join(case, 'constant', 'polyMesh')


def read_list(path, vector=False):
    """internalField / points / faces / owner: the OpenFOAM ASCII list after its size line."""
    with open(path, 'rb') as f:
        data = f.read()
    # find "<n>\n(" : the list header
    m = re.search(rb'\n(\d+)\s*\n\(', data)
    n = int(m.group(1))
    start = m.end()
    body = data[start:]
    end = body.rfind(b')')
    body = body[:end]
    if vector:
        # "(x y z)" per row
        arr = np.fromstring(body.replace(b'(', b' ').replace(b')', b' '), sep=' ', dtype=np.float64)
        return arr.reshape(n, -1)
    return np.fromstring(body, sep=' ', dtype=np.float64), n


def read_faces(path):
    with open(path, 'rb') as f:
        data = f.read()
    m = re.search(rb'\n(\d+)\s*\n\(', data)
    n = int(m.group(1))
    body = data[m.end():]
    body = body[:body.rfind(b')')]
    arr = np.fromstring(body.replace(b'(', b' ').replace(b')', b' '), sep=' ', dtype=np.int64)
    # every row "3(a b c)" -> 3 a b c
    arr = arr.reshape(n, 4)
    assert (arr[:, 0] == 3).all(), 'not a pure triangle-face mesh'
    return arr[:, 1:]


def read_field(path):
    with open(path, 'rb') as f:
        data = f.read()
    i = data.find(b'internalField')
    j = data.find(b'boundaryField')
    seg = data[i:j]
    if b'nonuniform' not in seg:
        val = re.search(rb'uniform\s+\(?([^;)]*)\)?;', seg).group(1)
        return np.fromstring(val, sep=' '), True
    m = re.search(rb'\n(\d+)\s*\n\(', seg)
    n = int(m.group(1))
    body = seg[m.end():]
    body = body[:body.rfind(b')')]
    if b'(' in body[:200]:
        arr = np.fromstring(body.replace(b'(', b' ').replace(b')', b' '), sep=' ').reshape(n, -1)
    else:
        arr = np.fromstring(body, sep=' ')
    return arr, False


print('reading mesh ...', flush=True)
pts = read_list(os.path.join(pm, 'points'), vector=True)
faces = read_faces(os.path.join(pm, 'faces'))
owner, nf = read_list(os.path.join(pm, 'owner'))
neigh, ni = read_list(os.path.join(pm, 'neighbour'))
owner = owner.astype(np.int64); neigh = neigh.astype(np.int64)
ncell = int(max(owner.max(), neigh.max())) + 1
fc = pts[faces].mean(axis=1)                           # face centres
cc = np.zeros((ncell, 3)); cnt = np.zeros(ncell)
np.add.at(cc, owner, fc); np.add.at(cnt, owner, 1)
np.add.at(cc, neigh, fc[:len(neigh)]); np.add.at(cnt, neigh, 1)
cc /= cnt[:, None]
print('cells %d, faces %d, points %d' % (ncell, len(faces), len(pts)), flush=True)

td = os.path.join(case, tdir)
U, uni = read_field(os.path.join(td, 'U'))
p, puni = read_field(os.path.join(td, 'p'))
print('U uniform' if uni else 'U cells %d' % len(U), '| p uniform' if puni else '| p cells %d' % len(p))
if uni or puni:
    sys.exit(0)
mag = np.linalg.norm(U, axis=1)


def where(i):
    x, y, z = cc[i]
    return '(%.1f, %.1f, %.1f) r_pool %.0f m' % (x, y, z, math.hypot(x - px, y - py))


print('|U|: max %.3f m/s at cell %d %s' % (mag.max(), mag.argmax(), where(mag.argmax())))
print('     percentiles 50/99/99.9/99.99: %.2f %.2f %.2f %.2f;  cells > 15 m/s: %d, > 20: %d, > 30: %d' % (
    np.percentile(mag, 50), np.percentile(mag, 99), np.percentile(mag, 99.9), np.percentile(mag, 99.99),
    (mag > 15).sum(), (mag > 20).sum(), (mag > 30).sum()))
for k, name in enumerate('xyz'):
    print('     U%s min %.3f at %s | max %.3f at %s' % (name, U[:, k].min(), where(U[:, k].argmin()),
                                                       U[:, k].max(), where(U[:, k].argmax())))
print('p  : min %.3f at cell %d %s' % (p.min(), p.argmin(), where(p.argmin())))
print('     max %.3f at cell %d %s' % (p.max(), p.argmax(), where(p.argmax())))
print('     percentiles 0.01/1/50/99/99.99: %.2f %.2f %.2f %.2f %.2f' % tuple(np.percentile(p, [0.01, 1, 50, 99, 99.99])))
print('     |p| > 200: %d cells, > 500: %d, > 1000: %d' % ((np.abs(p) > 200).sum(), (np.abs(p) > 500).sum(), (np.abs(p) > 1000).sum()))
# the 10 fastest cells and the 10 most extreme p cells
order = np.argsort(-mag)[:10]
print('fastest cells:')
for i in order:
    print('   cell %-9d |U| %7.3f  U (%.2f %.2f %.2f) p %9.2f at %s' % (i, mag[i], U[i, 0], U[i, 1], U[i, 2], p[i], where(i)))
order = np.argsort(-np.abs(p))[:10]
print('most extreme p cells:')
for i in order:
    print('   cell %-9d p %9.2f  |U| %7.3f at %s' % (i, p[i], mag[i], where(i)))
# how the extremes cluster: distance from the pool centre, height
near = math.hypot(cc[mag.argmax()][0] - px, cc[mag.argmax()][1] - py)
print('pool region (r < 45 m of the centre): max |U| %.3f, p min %.2f max %.2f' % (
    mag[np.hypot(cc[:, 0] - px, cc[:, 1] - py) < 45].max(),
    p[np.hypot(cc[:, 0] - px, cc[:, 1] - py) < 45].min(), p[np.hypot(cc[:, 0] - px, cc[:, 1] - py) < 45].max()))
