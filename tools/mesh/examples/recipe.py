#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""
recipe.py - what tools/mesh/examples/*.py share: the gmsh session, the bbox
surface classification, the physical groups, the transfinite box, the .msh
write, M4's two subprocess calls, the polyMesh readers and the pairing error.

A recipe builds its geometry in gmsh, meshes it hex-only, writes the .msh and
hands it to regions_from_msh.py / regions_check.py (M4) - never the reverse.
Nothing here edits M4's tools or step_mesh.py; the three imported functions
are polymesh_write.read_polymesh and regions_check.face_geometry /
cell_geometry (SPEC-LIT §2.1 / §2.2 as code).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))    # tools/mesh: M4's two modules

from polymesh_write import read_polymesh                  # noqa: E402
from regions_check import cell_geometry, face_geometry    # noqa: E402

T0 = time.time()


def log(msg):
    print('%7.1f s  %s' % (time.time() - T0, msg), flush=True)


def die(msg, code=1):
    sys.stderr.write('recipes: %s\n' % msg)
    sys.stderr.flush()
    raise SystemExit(code)


def repo_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
    # dirname x4 of __file__: examples -> mesh -> tools -> repo


def fresh_dir(out_dir):
    """The ONLY delete a recipe does: once, in main, before anything is written."""
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)


def gmsh_begin(threads=4):
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber('General.Terminal', 0)
    gmsh.option.setNumber('General.NumThreads', threads)


def gmsh_end():
    import gmsh
    gmsh.finalize()


def surfaces_of(volume):
    import gmsh
    return [int(t) for d, t in gmsh.model.getBoundary([(3, volume)], oriented=False)]


def lies_in_plane(tag, axis, value, tol=1e-6):
    """Both ends of the surface's bbox along the axis sit on the plane."""
    import gmsh
    b = gmsh.model.getBoundingBox(2, tag)
    return abs(b[axis] - value) < tol and abs(b[axis + 3] - value) < tol


def centre_of(dim, tag):
    import gmsh
    x, y, z = gmsh.model.occ.getCenterOfMass(dim, tag)
    return (x, y, z)


def add_groups(volumes, surfaces):
    """Mirror of step_mesh.report_groups' refusals, then the groups by name."""
    import gmsh
    used = {}
    for name in sorted(surfaces):
        if not surfaces[name]:
            die('surface group %r is empty' % name)
        for t in surfaces[name]:
            if t in used:
                die('surface %d is in two groups: %r and %r' % (t, used[t], name))
            used[t] = name
    for name in sorted(volumes):
        for t in surfaces_of(volumes[name]):
            if t not in used:
                die('surface %d bounds volume %r but is in no group' % (t, name))
    for name in sorted(surfaces):
        gmsh.model.addPhysicalGroup(2, surfaces[name], name=name)
    for name in sorted(volumes):
        gmsh.model.addPhysicalGroup(3, [volumes[name]], name=name)


def transfinite_box(volume, n):
    """A whole box structured: per curve the longest bbox axis picks n[axis]+1 points."""
    import gmsh
    curves = set()
    for s in surfaces_of(volume):
        for d, c in gmsh.model.getBoundary([(2, s)], oriented=False):
            curves.add(int(c))
    for c in sorted(curves):
        b = gmsh.model.getBoundingBox(1, c)
        axis = int(np.argmax([b[i + 3] - b[i] for i in range(3)]))
        gmsh.model.mesh.setTransfiniteCurve(c, n[axis] + 1)
    for s in surfaces_of(volume):
        gmsh.model.mesh.setTransfiniteSurface(s)
        gmsh.model.mesh.setRecombine(2, s)
    gmsh.model.mesh.setTransfiniteVolume(volume)


def generate_and_count(volumes):
    """mesh.generate(3), then the 3-D element types per volume - hex/prism only."""
    import gmsh
    gmsh.model.mesh.generate(3)
    counts = {}
    for name in sorted(volumes):
        per = {}
        types, etags = gmsh.model.mesh.getElements(3, volumes[name])[:2]
        for et, etg in zip(types, etags):
            t = int(et)
            if t not in (5, 6):
                die('recipe wants hex/prism only, got type %d in %s' % (t, name))
            per[t] = len(etg)
        counts[name] = per
    return counts


def write_msh(path):
    """The three options step_mesh.py writes its .msh with, then gmsh.write."""
    import gmsh
    gmsh.option.setNumber('Mesh.MshFileVersion', 4.1)
    gmsh.option.setNumber('Mesh.Binary', 0)
    gmsh.option.setNumber('Mesh.SaveAll', 0)
    gmsh.write(path)


def make_layout(msh, out_dir, fluid, materials):
    """M4's converter, then M4's checker; echoes both stdouts, returns regions.json."""
    root = repo_root()
    argv = [sys.executable, os.path.join(root, 'tools', 'mesh', 'regions_from_msh.py'),
            msh, out_dir, '--fluid', fluid or 'none']
    for r in sorted(materials):
        argv += ['--material', '%s=%s' % (r, materials[r])]
    p = subprocess.run(argv, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if p.returncode != 0:
        print((p.stdout + p.stderr)[-3000:])
        die('regions_from_msh.py exited %d on %s' % (p.returncode, os.path.basename(msh)),
            p.returncode)
    c = subprocess.run([sys.executable, os.path.join(root, 'tools', 'mesh', 'regions_check.py'),
                        os.path.join(out_dir, 'regions.json')],
                       capture_output=True, text=True, encoding='utf-8', errors='replace')
    if c.returncode != 0 or '[check] OK' not in c.stdout:
        print((c.stdout + c.stderr)[-3000:])
        die('regions_check.py exited %d (no [check] OK)' % c.returncode, c.returncode or 1)
    print(p.stdout, end='')
    print(c.stdout, end='')
    with open(os.path.join(out_dir, 'regions.json'), encoding='utf-8') as f:
        return json.load(f)


def load_region(layout_dir, regions, name):
    """read_polymesh of one region of the layout, plus 'n_cells' (R5: its own numbering)."""
    entry = next((r for r in regions['regions'] if r['name'] == name), None)
    if entry is None:
        die('no region named %r in regions.json' % name)
    pm = read_polymesh(os.path.join(layout_dir, entry['polyMesh']))
    pm['n_cells'] = int(max(pm['owner'].max(),
                            pm['neighbour'].max() if len(pm['neighbour']) else -1)) + 1
    return pm


def patch_faces(pm, name):
    p = next((q for q in pm['patches'] if q['name'] == name), None)
    if p is None:
        die('no patch named %r (have %s)' % (name, ', '.join(q['name'] for q in pm['patches'])))
    return p['type'], range(p['startFace'], p['startFace'] + p['nFaces'])


def face_arrays(pm, ids):
    """regions_check.face_geometry per face -> (Cf (m,3), |Sf| (m,), unit n (m,3))."""
    cf, area, unit = [], [], []
    for f in ids:
        sf, c = face_geometry(pm['points'], pm['faces'][f])
        a = float(np.sqrt(sf.dot(sf)))
        cf.append(c)
        area.append(a)
        unit.append(sf / a if a > 0.0 else sf)
    return np.array(cf), np.array(area), np.array(unit)


def cell_centres(pm):
    """regions_check.cell_geometry's C - SPEC-LIT §2.2's C_P, the pyramid centres."""
    cent, _ = cell_geometry(pm['points'], pm['faces'], pm['owner'], pm['neighbour'], pm['n_cells'])
    return cent


def pairing_error(layout_dir, regions):
    """R2 by index, recomputed from the two polyMeshes of every interface."""
    out = []
    for it in regions.get('interfaces', []):
        ra, rb = it['regions']
        pa, pb = it['patches']
        ma = load_region(layout_dir, regions, ra)
        mb = load_region(layout_dir, regions, rb)
        ta, ia = patch_faces(ma, pa)
        tb, ib = patch_faces(mb, pb)
        if ta != 'patch' or tb != 'patch':
            die('interface %s/%s: patch type %r/%r, R3 wants patch' % (pa, pb, ta, tb))
        if len(ia) != len(ib):
            die('interface %s/%s: %d faces vs %d' % (pa, pb, len(ia), len(ib)))
        ca, aa, na = face_arrays(ma, ia)
        cb, ab, nb = face_arrays(mb, ib)
        out.append({'regions': [ra, rb], 'patches': [pa, pb], 'faces': len(ia),
                    'max_centroid': float(np.max(np.sqrt(((ca - cb) ** 2).sum(axis=1)))),
                    'max_area_rel': float(np.max(np.abs(aa - ab) / aa)),
                    'min_opposed': float(np.min(-(na * nb).sum(axis=1)))})
    return out


def write_json(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write('\n')


def summarize(regions, layout_dir, counts):
    """The '[recipe]' console lines and the dict the sidecar embeds."""
    side = {'regions': {}, 'interfaces': []}
    for r in regions['regions']:
        pm = load_region(layout_dir, regions, r['name'])
        per = counts.get(r['name'], {})
        print('[recipe] region %s: %d cells ({%s})' % (
            r['name'], pm['n_cells'],
            ', '.join('%d: %d' % (t, per[t]) for t in sorted(per))), flush=True)
        side['regions'][r['name']] = {
            'cells': pm['n_cells'],
            'patches': {p['name']: [p['type'], p['nFaces']] for p in pm['patches']},
            'elements': {str(t): per[t] for t in sorted(per)}}
    for pe in pairing_error(layout_dir, regions):
        print('[recipe] pairing %s: centroid %.1e area %.1e opposed %.1e' % (
            pe['patches'][0], pe['max_centroid'], pe['max_area_rel'], pe['min_opposed']),
            flush=True)
        side['interfaces'].append(pe)
    return side
