#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""
regions_check.py - refuse a region layout that breaks R1, R2, R3 or R6.

Reads a regions.json manifest (the schema of docs/10-fsi-solid-mesh-plan.md §C)
and every region's polyMesh under it: R1 a region complete and standalone (owner
< neighbour, internal faces sorted, patches contiguous, every cell carried,
positive volumes by geometry.rs's pyramid decomposition); R2 the k-th faces of
a patch pair coincident (cht.rs's three PairingTolerances inequalities,
dimensionless, within the manifest's tolerance); R3 patches named <a>_to_<b> /
<b>_to_<a> and typed patch; R6 relative paths holding the five files and
exactly the §C keys. Exit 0, or 1 with one line per violation, or 2 on usage.

    python tools/mesh/regions_check.py <path/to/regions.json>
"""
import argparse
import json
import os
import sys

import numpy as np

import polymesh_write

SMALL = 1e-150     # geometry.rs's SMALL: below it a face's area weighting is meaningless
FILES = ('points', 'faces', 'owner', 'neighbour', 'boundary')
TOP_KEYS = {'version', 'units', 'regions', 'interfaces', 'source'}
REGION_KEYS = {'name', 'kind', 'polyMesh'}
INTERFACE_KEYS = {'regions', 'patches', 'faces', 'tolerance'}


def _die(msg):
    sys.stderr.write('regions_check: %s\n' % msg)
    sys.stderr.flush()
    raise SystemExit(2)


def face_geometry(points, face):
    """(Sf, Cf) by the fan about the vertex average - geometry.rs:111-160."""
    n = len(face)
    if n == 0:
        return np.zeros(3), np.zeros(3)
    xavg = points[face].mean(axis=0)
    if n < 3:
        return np.zeros(3), xavg
    sf, cf, area = np.zeros(3), np.zeros(3), 0.0
    for i in range(n):
        a, b = points[face[i]], points[face[(i + 1) % n]]
        tn = np.cross(a - xavg, b - xavg)
        ta = float(np.sqrt(tn.dot(tn))) * 0.5
        sf += tn * 0.5
        cf += (xavg + a + b) / 3.0 * ta
        area += ta
    return (sf, cf / area) if area > SMALL else (sf, xavg)


def cell_geometry(points, faces, owner, neighbour, n_cells):
    """(C, V) by the pyramid decomposition about the mean face centroid - geometry.rs:385-445."""
    n_if = len(neighbour)
    geo = [face_geometry(points, face) for face in faces]
    apex = np.zeros((n_cells, 3))
    n_cf = np.zeros(n_cells, dtype=np.int64)
    for f in range(len(faces)):
        for c in ([owner[f], neighbour[f]] if f < n_if else [owner[f]]):
            apex[c] += geo[f][1]
            n_cf[c] += 1
    for c in np.nonzero(n_cf)[0]:
        apex[c] /= n_cf[c]
    vol = np.zeros(n_cells)
    c_acc = np.zeros((n_cells, 3))
    for f in range(len(faces)):
        for c, s in ([(owner[f], 1.0), (neighbour[f], -1.0)] if f < n_if else [(owner[f], 1.0)]):
            vp = float(geo[f][0].dot(geo[f][1] - apex[c])) * s / 3.0
            vol[c] += vp
            c_acc[c] += (geo[f][1] * 0.75 + apex[c] * 0.25) * vp
    cent = apex.copy()
    pos = np.nonzero(vol > 0.0)[0]
    cent[pos] = c_acc[pos] / vol[pos, None]
    return cent, vol


def _n_cells(pm):
    return int(max(pm['owner'].max(), pm['neighbour'].max() if len(pm['neighbour']) else -1)) + 1


def _find(pm, name):
    return next((p for p in pm['patches'] if p['name'] == name), None)


def check_region(name, pm):
    """R1 for one region: (violations, info). Every violation names region and index."""
    v = []
    faces, owner, neighbour = pm['faces'], pm['owner'], pm['neighbour']
    n_faces, n_if, n_pts = len(faces), len(neighbour), len(pm['points'])
    if len(owner) != n_faces:
        v.append("R1: region '%s': owner has %d entries but there are %d faces" % (name, len(owner), n_faces))
    if n_if > n_faces:
        v.append("R1: region '%s': %d internal faces out of %d faces" % (name, n_if, n_faces))
    prev = (-1, -1)
    for f in range(n_if):
        o, n = int(owner[f]), int(neighbour[f])
        if o >= n:
            v.append("R1: region '%s' face %d: owner %d >= neighbour %d - not upper-triangular" % (name, f, o, n))
        if (o, n) < prev:
            v.append("R1: region '%s' face %d: internal faces not in (owner, neighbour) order" % (name, f))
        prev = (o, n)
    for fi, face in enumerate(faces):
        if len(face) < 3:
            v.append("R1: region '%s' face %d: %d vertices" % (name, fi, len(face)))
        for vtx in face:
            if vtx < 0 or vtx >= n_pts:
                v.append("R1: region '%s' face %d: vertex %d outside 0..%d" % (name, fi, vtx, n_pts - 1))
    expect = n_if
    for p in pm['patches']:
        if p['startFace'] != expect:
            v.append("R1: region '%s': patch '%s' starts at face %d, expected %d - boundary faces must be contiguous and in patch order" % (name, p['name'], p['startFace'], expect))
        expect = p['startFace'] + p['nFaces']
    if expect != n_faces:
        v.append("R1: region '%s': the patches cover faces %d..%d but the mesh has %d" % (name, n_if, expect, n_faces))
    n_cells = _n_cells(pm) if n_faces else 0
    used = np.zeros(n_cells, dtype=bool)
    used[np.asarray(owner, dtype=np.int64)] = True
    used[np.asarray(neighbour, dtype=np.int64)] = True
    for c in np.nonzero(~used)[0]:
        v.append("R1: region '%s': cell %d is in no owner/neighbour list" % (name, c))
    cent, vol = cell_geometry(pm['points'], faces, owner, neighbour, n_cells)
    for c in np.nonzero(vol <= 0.0)[0]:
        v.append("R1: region '%s' cell %d: volume %.3e <= 0" % (name, c, vol[c]))
    info = {'name': name, 'nCells': n_cells, 'nPoints': n_pts, 'nFaces': n_faces, 'nInternalFaces': n_if,
            'nPatches': len(pm['patches']), 'minV': float(vol.min()) if n_cells else 0.0, 'ok': not v}
    return v, info


def _theta(sf, mag, cf, cc):
    """The face's lean off its own cell-to-face direction, in degrees (report only)."""
    d = cf - cc
    nd = float(np.sqrt(d.dot(d)))
    if mag <= 0.0 or nd <= 0.0:
        return 0.0
    cosv = float(sf.dot(d)) / (mag * nd)
    return float(np.degrees(np.arccos(min(1.0, max(-1.0, cosv)))))


def check_interface(iface, pm_a, pm_b, names):
    """R2 + R3 for one interface: (violations, report). Side A (names[0]) defines the order."""
    v = []
    a, b = names
    want = [a + '_to_' + b, b + '_to_' + a]
    pair = '%s/%s' % (want[0], want[1])
    rep = {'a': a, 'b': b, 'pa': want[0], 'pb': want[1], 'faces': 0, 'area': 0.0,
           'wc': 0.0, 'wa': 0.0, 'wn': 0.0, 'ta': 0.0, 'tb': 0.0}
    if list(iface.get('patches', [])) != want:
        v.append('R3: interface %s: patches %s should be %s' % (pair, iface.get('patches'), want))
    pa, pb = _find(pm_a, want[0]), _find(pm_b, want[1])
    for nm, wp, p in ((a, want[0], pa), (b, want[1], pb)):
        if p is None:
            v.append("R3: interface %s: region '%s' has no patch '%s'" % (pair, nm, wp))
        elif p['type'] != 'patch':
            v.append("R3: interface %s: patch '%s' has type '%s', not 'patch'" % (pair, wp, p['type']))
    if pa is None or pb is None:
        return v, rep
    na, nb, nman = pa['nFaces'], pb['nFaces'], iface.get('faces')
    rep['faces'] = na
    if na != nb or nb != nman:
        v.append('R2: interface %s: face counts disagree - patch %r has %s, %r has %s, the manifest says %s; the k-th-face pairing is not checked' % (pair, want[0], na, want[1], nb, nman))
        return v, rep
    tol = float(iface.get('tolerance', 1e-9))
    ca = cell_geometry(pm_a['points'], pm_a['faces'], pm_a['owner'], pm_a['neighbour'], _n_cells(pm_a))[0]
    cb = cell_geometry(pm_b['points'], pm_b['faces'], pm_b['owner'], pm_b['neighbour'], _n_cells(pm_b))[0]
    worst = [0.0, 0.0, 0.0, 0.0, 0.0]
    area_sum = 0.0
    for k in range(na):
        sfa, cfa = face_geometry(pm_a['points'], pm_a['faces'][pa['startFace'] + k])
        sfb, cfb = face_geometry(pm_b['points'], pm_b['faces'][pb['startFace'] + k])
        ma, mb = float(np.sqrt(sfa.dot(sfa))), float(np.sqrt(sfb.dot(sfb)))
        area_sum += ma
        cd = float(np.sqrt((cfa - cfb).dot(cfa - cfb))) / np.sqrt(ma) if ma > 0.0 else float('inf')
        ad = abs(ma - mb) / ma if ma > 0.0 else float('inf')
        dot = float(sfa.dot(sfb)) / (ma * mb) if ma > 0.0 and mb > 0.0 else -2.0
        worst[0], worst[1], worst[2] = max(worst[0], cd), max(worst[1], ad), max(worst[2], dot + 1.0)
        if cd > tol:
            v.append('R2: interface %s face %d: centroid distance %.1e > tolerance %g (relative to sqrt(area))' % (pair, k, cd, tol))
        if ad > tol:
            v.append('R2: interface %s face %d: areas %.1e and %.1e differ by %.1e relative > tolerance %g' % (pair, k, ma, mb, ad, tol))
        if dot > -1.0 + tol:
            v.append('R2: interface %s face %d: the two face normals have n_A . n_B = %.1e, not -1 (tolerance %g)' % (pair, k, dot, tol))
        worst[3] = max(worst[3], _theta(sfa, ma, cfa, ca[int(pm_a['owner'][pa['startFace'] + k])]))
        worst[4] = max(worst[4], _theta(sfb, mb, cfb, cb[int(pm_b['owner'][pb['startFace'] + k])]))
    rep.update({'area': area_sum, 'wc': worst[0], 'wa': worst[1], 'wn': worst[2],
                'ta': worst[3], 'tb': worst[4]})
    return v, rep


def check_layout(manifest_path):
    """The whole layout: (violations, report). _die(2) on an unreadable manifest."""
    try:
        with open(manifest_path, encoding='utf-8') as f:
            man = json.load(f)
    except (OSError, ValueError) as e:
        _die('cannot read %s: %s' % (manifest_path, e))
    if not isinstance(man, dict):
        _die('%s: the manifest is not a JSON object' % manifest_path)
    v = []
    for k in sorted(set(man) - TOP_KEYS):
        v.append("R6: manifest carries unknown key '%s'" % k)
    for k in sorted(TOP_KEYS - set(man)):
        v.append("R6: manifest lacks key '%s'" % k)
    if man.get('version') != 1:
        v.append('R6: manifest version %r is not 1' % man.get('version'))
    regions = man.get('regions', [])
    names = [r.get('name') for r in regions]
    if len(set(names)) != len(names):
        v.append('R6: region names are not unique: %s' % names)
    pms = {}
    base = os.path.dirname(manifest_path)
    for r in regions:
        nm = r.get('name')
        extra = {'material'} if r.get('kind') == 'solid' else set()
        for k in sorted(set(r) - REGION_KEYS - extra):
            v.append("R6: region '%s' carries unknown key '%s' (only a solid may carry material)" % (nm, k))
        if r.get('kind') not in ('fluid', 'solid'):
            v.append("R6: region '%s': kind %r is not fluid or solid" % (nm, r.get('kind')))
        rel = r.get('polyMesh', '')
        if not rel or os.path.isabs(rel) or '..' in rel.replace(os.sep, '/').split('/'):
            v.append("R6: region '%s': polyMesh path %r is not relative to the manifest" % (nm, rel))
            continue
        pm_dir = os.path.join(base, rel)
        missing = [fn for fn in FILES if not os.path.isfile(os.path.join(pm_dir, fn))]
        if missing:
            v.append("R6: region '%s': %s is missing %s" % (nm, rel, ', '.join(missing)))
            continue
        pms[nm] = polymesh_write.read_polymesh(pm_dir)
    rinfo = []
    for r in regions:
        pm = pms.get(r.get('name'))
        if pm is not None:
            rv, info = check_region(r['name'], pm)
            v += rv
            rinfo.append(info)
    irep = []
    for i in man.get('interfaces', []):
        for k in sorted(set(i) - INTERFACE_KEYS):
            v.append("R6: interface carries unknown key '%s'" % k)
        nm = i.get('regions', [])
        for rn in nm:
            if rn not in names:
                v.append("R6: interface names region '%s', which the manifest does not declare" % rn)
        if len(nm) == 2 and nm[0] in pms and nm[1] in pms:
            iv, rep = check_interface(i, pms[nm[0]], pms[nm[1]], nm)
            v += iv
            irep.append(rep)
    return v, {'regions': rinfo, 'interfaces': irep, 'n_regions': len(regions), 'n_interfaces': len(man.get('interfaces', []))}


def print_report(report, violations):
    """The [check] lines of C8: one per region and interface, then OK or every violation."""
    for ri in report['regions']:
        print('[check] %s: %d cells, %d points, %d faces (%d internal), %d patches, min V %.3e, R1 %s'
              % (ri['name'], ri['nCells'], ri['nPoints'], ri['nFaces'], ri['nInternalFaces'],
                 ri['nPatches'], ri['minV'], 'ok' if ri['ok'] else 'VIOLATED'))
    for ri in report['interfaces']:
        print('[check] interface %s / %s: %d faces, area %.3e, worst centroid %.1e, worst area %.1e, worst normal %.1e, worst non-orth %.1f deg (A) %.1f deg (B) [report only; solver gate 5.0 deg]'
              % (ri['pa'], ri['pb'], ri['faces'], ri['area'], ri['wc'], ri['wa'], ri['wn'], ri['ta'], ri['tb']))
    for msg in violations:
        print('[check] VIOLATION %s' % msg)
    if violations:
        print('[check] FAIL: %d violation(s)' % len(violations))
    else:
        print('[check] OK: R1 R2 R3 R6 hold for %d regions, %d interface%s'
              % (report['n_regions'], report['n_interfaces'], '' if report['n_interfaces'] == 1 else 's'))


def main(argv=None):
    ap = argparse.ArgumentParser(description='Check a region layout (regions.json) against rules R1, R2, R3 and R6.')
    ap.add_argument('manifest', help='the regions.json to check')
    args = ap.parse_args(argv)
    violations, report = check_layout(args.manifest)
    print_report(report, violations)
    return 1 if violations else 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    sys.exit(main())
