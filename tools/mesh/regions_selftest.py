#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""
regions_selftest.py - the M4 unit end to end, on meshes gmsh builds for it.

gmsh (an external GPL process; its source is never read) meshes a box with a
building on the floor four ways and a transfinite hex slab two ways:
a single volume (tet, hex) must come out byte-identical to the Rust converter;
two regions must pair conformally by index, the fluid region again byte-equal;
the checker must refuse a broken layout (R2 counts, R2 order, R3 name, R6 path,
R6 unknown key); the hex layout must be all-quad with gmsh's own counts; the
face tables must give analytic volumes on a hand-written prism and pyramid;
every
refusal (--fluent, an existing manifest, --fluid naming no volume, a SaveAll
mesh with no volume name, no $PhysicalNames) must name its reason and write
nothing; '--fluid none' makes every region solid; --msh PATH runs the tool on
an external two-region mesh.

    python tools/mesh/regions_selftest.py [--keep [DIR]] [--msh PATH]
"""
import argparse
import filecmp
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

import polymesh_write
import regions_check
import regions_from_msh

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, 'regions_from_msh.py')
CHECK = os.path.join(HERE, 'regions_check.py')
REPO = os.path.dirname(os.path.dirname(HERE))
CONVERTER = os.environ.get('OFGPU_CONVERT') or os.path.join(REPO, 'rust', 'target', 'release', 'ofgpu-convert-mesh.exe')


def run(argv):
    return subprocess.run(argv, capture_output=True, text=True, encoding='utf-8', errors='replace')


def tool(msh, out, *extra): return run([sys.executable, TOOL, msh, out] + list(extra))


def cmp_polymesh(a, b):
    ok = [filecmp.cmp(os.path.join(a, fn), os.path.join(b, fn), shallow=False) for fn in polymesh_write.FILES]
    assert all(ok), 'polyMesh bytes differ between %s and %s: %s' % (a, b, ok)
    return len(ok)


def _bbox_plane(gmsh, s):
    """Which axis a model surface is flat across (asserts exactly one)."""
    x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(2, s)
    flat = (abs(x1 - x0), abs(y1 - y0), abs(z1 - z0))
    assert sum(f < 1e-6 for f in flat) == 1, 'surface %d does not lie in one plane' % s
    return flat


def build_tet_meshes(work):
    """Four tet meshes from one gmsh session: the 100 x 60 x 40 domain, a 20 x 20 x 8 building on the floor."""
    import gmsh
    paths = {k: os.path.join(work, k + '.msh') for k in ('two_region', 'fluid_only', 'no_volume_name', 'no_physical_names')}
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal', 0); gmsh.model.add('two_region')
        land, bld = gmsh.model.occ.addBox(0, 0, 0, 100, 60, 40), gmsh.model.occ.addBox(40, 20, 0, 20, 20, 8)
        out = gmsh.model.occ.fragment([(3, land)], [(3, bld)]); gmsh.model.occ.synchronize()
        vf, vb = out[1][0][0][1], out[1][1][0][1]
        assert gmsh.model.getBoundingBox(3, vf)[0] < 1 < gmsh.model.getBoundingBox(3, vb)[0], 'fragment swapped the volumes'
        surf = lambda v: [s[1] for s in gmsh.model.getBoundary([(3, v)], oriented=False)]
        shared = sorted(set(surf(vf)) & set(surf(vb)))
        groups = {'fluid_to_building': shared}
        for s in surf(vf):
            if s in shared: continue
            flat, b = _bbox_plane(gmsh, s), gmsh.model.getBoundingBox(2, s)
            ax = flat.index(min(flat))
            key = ('top' if b[2] > 20 else 'wall_ground_land') if ax == 2 else ('west' if b[0] < 1 else 'east') if ax == 0 else ('south' if b[1] < 1 else 'north')
            groups.setdefault(key, []).append(s)
        base = [s for s in surf(vb) if _bbox_plane(gmsh, s)[2] and gmsh.model.getBoundingBox(2, s)[5] < 1e-6]
        groups['building_base'] = base; assert len(base) == 1 and base[0] not in shared, 'the building bottom is not one ground face'
        pg = {k: gmsh.model.addPhysicalGroup(2, ss, name=k) for k, ss in groups.items()}
        pf = gmsh.model.addPhysicalGroup(3, [vf], name='fluid'); pb = gmsh.model.addPhysicalGroup(3, [vb], name='building')
        for opt, val in (('Mesh.MeshSizeMin', 4), ('Mesh.MeshSizeMax', 8), ('Mesh.Algorithm', 6), ('Mesh.Algorithm3D', 1), ('Mesh.SaveAll', 0), ('Mesh.MshFileVersion', 4.1), ('Mesh.Binary', 0)): gmsh.option.setNumber(opt, val)
        gmsh.model.mesh.generate(3); gmsh.write(paths['two_region'])
        counts = {'n_fluid': len(gmsh.model.mesh.getElements(3, vf)[1][0]), 'n_bld': len(gmsh.model.mesh.getElements(3, vb)[1][0]),
                  'n_iface': sum(len(gmsh.model.mesh.getElements(2, s)[1][0]) for s in shared)}
        # gmsh corrupts its heap if removePhysicalGroups gets no argument: pass explicit lists, never a group twice.
        gmsh.model.removePhysicalGroups([(3, pb), (2, pg['building_base'])]); gmsh.write(paths['fluid_only'])
        gmsh.option.setNumber('Mesh.SaveAll', 1)
        gmsh.model.removePhysicalGroups([(3, pf)]); gmsh.write(paths['no_volume_name'])
        gmsh.model.removePhysicalGroups([(2, t) for k, t in pg.items() if k != 'building_base']); gmsh.write(paths['no_physical_names'])
    finally:
        gmsh.finalize()
    return paths, counts


def _tf_curves(gmsh, h):  # gmsh's size-based transfinite gives the thin direction 3 layers; pin the divisions
    for dim, tag in gmsh.model.getEntities(1):
        bb = gmsh.model.getBoundingBox(1, tag)
        ax = max(range(3), key=lambda i: bb[i + 3] - bb[i])
        gmsh.model.mesh.setTransfiniteCurve(tag, int(round((bb[ax + 3] - bb[ax]) / h)) + 1)


def build_hex_meshes(work):
    """hex_one: one 2 x 1 x 0.05 box in transfinite hexes at size 0.1; hex_two: the same slab split down x = 1."""
    import gmsh
    one, two = os.path.join(work, 'hex_one.msh'), os.path.join(work, 'hex_two.msh')
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal', 0); gmsh.model.add('hex_two')
        a, b = gmsh.model.occ.addBox(0, 0, 0, 2, 1, 0.05), gmsh.model.occ.addBox(1, 0, 0, 1, 1, 0.05)
        out = gmsh.model.occ.fragment([(3, a)], [(3, b)]); gmsh.model.occ.synchronize()
        va, vb = out[1][0][0][1], out[1][1][0][1]
        assert gmsh.model.getBoundingBox(3, va)[0] < 0.5 < gmsh.model.getBoundingBox(3, vb)[0], 'fragment swapped the slabs'
        surf = lambda v: [s[1] for s in gmsh.model.getBoundary([(3, v)], oriented=False)]
        shared = sorted(set(surf(va)) & set(surf(vb)))
        assert len(shared) == 1, 'the two slabs share %d faces' % len(shared)
        groups = {3: {'fluid': [va], 'plate': [vb]}, 2: {'fluid_to_plate': shared}}
        for v in (va, vb):
            for s in surf(v):
                if s in shared: continue
                flat, b = _bbox_plane(gmsh, s), gmsh.model.getBoundingBox(2, s)
                ax = flat.index(min(flat))
                key = ('inlet' if b[0] < 0.5 else 'outlet') if ax == 0 else ('wall_bottom' if b[1] < 0.5 else 'wall_top') if ax == 1 else ('empty_back' if b[2] < 0.025 else 'empty_front')
                groups[2].setdefault(key, []).append(s)
        for dim, names in groups.items():
            for nm, ss in names.items(): gmsh.model.addPhysicalGroup(dim, ss, name=nm)
        for opt, val in (('Mesh.MeshSizeMin', 0.05), ('Mesh.MeshSizeMax', 0.05), ('Mesh.RecombineAll', 1), ('Mesh.SaveAll', 0), ('Mesh.MshFileVersion', 4.1), ('Mesh.Binary', 0)): gmsh.option.setNumber(opt, val)
        gmsh.model.mesh.setTransfiniteAutomatic([(3, va), (3, vb)]); _tf_curves(gmsh, 0.05)
        gmsh.model.mesh.generate(3); gmsh.write(two)
        counts = {'n_a': len(gmsh.model.mesh.getElements(3, va)[1][0]), 'n_b': len(gmsh.model.mesh.getElements(3, vb)[1][0]), 'n_iface': len(gmsh.model.mesh.getElements(2, shared[0])[1][0])}
        gmsh.clear(); gmsh.model.add('hex_one')
        c = gmsh.model.occ.addBox(0, 0, 0, 2, 1, 0.05); gmsh.model.occ.synchronize(); gmsh.model.addPhysicalGroup(3, [c], name='fluid')
        gmsh.option.setNumber('Mesh.MeshSizeMin', 0.1); gmsh.option.setNumber('Mesh.MeshSizeMax', 0.1)
        gmsh.model.mesh.setTransfiniteAutomatic([(3, c)]); gmsh.model.mesh.generate(3); gmsh.write(one)
    finally:
        gmsh.finalize()
    return counts


def test_single_and_two_region(work, paths, counts):
    """Reqs 1 + 2: a single volume (tet, hex) is byte-identical to the Rust converter;
    two regions pair by index, and the fluid region IS the single-volume tet mesh."""
    for msh, label in ((paths['fluid_only'], 'tet'), (os.path.join(work, 'hex_one.msh'), 'hex')):
        p = tool(msh, os.path.join(work, 'bit_' + label)); assert p.returncode == 0, p.stdout[-500:] + p.stderr[-400:]
        run([CONVERTER, msh, os.path.join(work, 'bit_case_' + label)])
        cmp_polymesh(os.path.join(work, 'bit_' + label, 'fluid', 'polyMesh'), os.path.join(work, 'bit_case_' + label, 'constant', 'polyMesh'))
        print('  [ok] %s single volume: 5 files byte-identical to ofgpu-convert-mesh' % label)
    out = os.path.join(work, 'two_region')
    p = tool(paths['two_region'], out, '--material', 'building=concrete'); assert p.returncode == 0, p.stdout[-600:] + p.stderr[-400:]
    with open(os.path.join(out, 'regions.json'), encoding='utf-8') as f: man = json.load(f)
    assert [r['name'] for r in man['regions']] == ['fluid', 'building']
    assert [r['kind'] for r in man['regions']] == ['fluid', 'solid'] and man['regions'][1]['material'] == 'concrete'
    assert man['interfaces'] == [{'regions': ['fluid', 'building'], 'patches': ['fluid_to_building', 'building_to_fluid'],
                                  'faces': counts['n_iface'], 'tolerance': 1e-9}], man['interfaces']
    assert man['source'] == {'tool': 'regions_from_msh', 'version': '1', 'geometry': 'two_region.msh',
                             'config': '--material building=concrete'}, man['source']
    fpm = polymesh_write.read_polymesh(os.path.join(out, 'fluid', 'polyMesh')); bpm = polymesh_write.read_polymesh(os.path.join(out, 'building', 'polyMesh'))
    assert regions_check._n_cells(fpm) == counts['n_fluid'] and regions_check._n_cells(bpm) == counts['n_bld']
    assert {pp['name'] for pp in fpm['patches']} == {'fluid_to_building', 'top', 'west', 'east', 'south', 'north', 'wall_ground_land'}
    cmp_polymesh(os.path.join(out, 'fluid', 'polyMesh'), os.path.join(work, 'bit_tet', 'fluid', 'polyMesh'))
    pa = next(pp for pp in fpm['patches'] if pp['name'] == 'fluid_to_building'); pb = next(pp for pp in bpm['patches'] if pp['name'] == 'building_to_fluid')
    assert pa['nFaces'] == pb['nFaces'] == counts['n_iface']
    for k in range(pa['nFaces']):
        fa, fb = fpm['faces'][pa['startFace'] + k], bpm['faces'][pb['startFace'] + k]
        assert np.array_equal(fpm['points'][fa], bpm['points'][fb][::-1]), 'the index pairing breaks at %d' % k
    ck = run([sys.executable, CHECK, os.path.join(out, 'regions.json')]); assert ck.returncode == 0, ck.stdout[-400:]; print('  [ok] two regions: fluid == the single-volume tet bytes, %d interface faces pair by index' % counts['n_iface'])


def test_checker_refusals(work):
    """Req 3: the checker refuses a layout edited to break R2, R3 or R6."""
    src = os.path.join(work, 'two_region')
    with open(os.path.join(src, 'regions.json'), encoding='utf-8') as f: man = json.load(f)
    def with_manifest(label, fn):
        d = os.path.join(work, 'bad_' + label)
        shutil.copytree(src, d)
        m = json.loads(json.dumps(man)); fn(m)
        with open(os.path.join(d, 'regions.json'), 'w', encoding='utf-8', newline='\n') as f: json.dump(m, f, indent=2); f.write('\n')
        return d
    cases = [
        ('r2count', with_manifest('r2count', lambda m: m['interfaces'][0].update(faces=man['interfaces'][0]['faces'] + 1)), ('face counts disagree',)),
        ('r3name', with_manifest('r3name', lambda m: m['interfaces'][0].update(patches=['x_to_y', 'y_to_x'])), ('should be',)),
        ('r6abs', with_manifest('r6abs', lambda m: m['regions'][0].update(polyMesh='C:/elsewhere/polyMesh')), ('not relative',)),
        ('r6key', with_manifest('r6key', lambda m: m.update(comment='hello')), ('unknown key', 'comment')),
    ]

    d = os.path.join(work, 'bad_r2order')   # swap two face lines inside the
    shutil.copytree(src, d)                 # building_to_fluid range on disk:
    pm = polymesh_write.read_polymesh(os.path.join(d, 'building', 'polyMesh'))
    pb = next(p for p in pm['patches'] if p['name'] == 'building_to_fluid')
    fp = os.path.join(d, 'building', 'polyMesh', 'faces')
    with open(fp, encoding='utf-8') as f: lines = f.read().split('\n')
    j = next(i for i, ln in enumerate(lines) if ln.rstrip().endswith('(')) + 1 + pb['startFace']
    lines[j], lines[j + 1] = lines[j + 1], lines[j]
    with open(fp, 'w', encoding='utf-8', newline='\n') as f: f.write('\n'.join(lines))
    cases.append(('r2order', d, ('VIOLATION R2', 'face 0')))
    for label, bad, wants in cases:
        ck = run([sys.executable, CHECK, os.path.join(bad, 'regions.json')])
        assert ck.returncode == 1, '%s: exit %d\n%s' % (label, ck.returncode, ck.stdout[-400:])
        assert all(w in ck.stdout for w in wants), '%s: %r not in\n%s' % (label, wants, ck.stdout[-400:])
    print('  [ok] the checker refused %d broken layouts (R2 count, R2 order, R3 name, R6 path, R6 key)' % len(cases))


def test_hex_one_cell_thick(work, counts):
    """Reqs 4 + 7: a transfinite hex slab splits cleanly, every face a quad;
    the converter's stdout counts match the tool's polyMesh for hex_one."""
    out = os.path.join(work, 'hex_two')
    p = tool(os.path.join(work, 'hex_two.msh'), out, '--material', 'plate=steel'); assert p.returncode == 0, p.stdout[-600:] + p.stderr[-400:]
    with open(os.path.join(out, 'regions.json'), encoding='utf-8') as f: man = json.load(f)
    assert [r['name'] for r in man['regions']] == ['fluid', 'plate'] and man['regions'][1]['material'] == 'steel'
    assert man['interfaces'][0]['faces'] == counts['n_iface'], man['interfaces']
    types = {}
    for r, n in zip(man['regions'], (counts['n_a'], counts['n_b'])):
        pm = polymesh_write.read_polymesh(os.path.join(out, r['name'], 'polyMesh')); assert regions_check._n_cells(pm) == n, (r['name'], regions_check._n_cells(pm), n)
        for face in pm['faces']: assert len(face) == 4, 'hex region %s has a %d-vertex face' % (r['name'], len(face))
        types.update({pp['name']: pp['type'] for pp in pm['patches']})
    assert types['empty_back'] == types['empty_front'] == 'empty'
    assert types['wall_bottom'] == types['wall_top'] == 'wall'
    assert types['inlet'] == types['outlet'] == types['fluid_to_plate'] == 'patch'
    cp = run([CONVERTER, os.path.join(work, 'hex_one.msh'), os.path.join(work, 'hex_case')])
    m = re.search(r'(\d+) cells, (\d+) points, (\d+) faces \((\d+) internal', cp.stdout); assert m, cp.stdout[-300:]
    pm = polymesh_write.read_polymesh(os.path.join(work, 'bit_hex', 'fluid', 'polyMesh'))
    got = (regions_check._n_cells(pm), len(pm['points']), len(pm['faces']), len(pm['neighbour']))
    assert tuple(int(x) for x in m.groups()) == got, (m.groups(), got)
    ck = run([sys.executable, CHECK, os.path.join(out, 'regions.json')]); assert ck.returncode == 0, ck.stdout[-400:]
    print('  [ok] hex layout: %d + %d hexes, %d interface quads, all faces 4-vertex, converter counts agree'
          % (counts['n_a'], counts['n_b'], counts['n_iface']))


def test_face_tables(work):
    """Req 8: the face tables on a hand-written prism + pyramid .msh naming no
    surfaces (C11): analytic volumes, outward winding, the defaultFaces warning."""
    msh = os.path.join(work, 'prism_pyr.msh')
    with open(msh, 'w', encoding='utf-8', newline='\n') as f:
        f.write('$MeshFormat\n4.1 0 8\n$EndMeshFormat\n$PhysicalNames\n1\n3 1 "solid"\n$EndPhysicalNames\n')
        f.write('$Entities\n0 0 0 1\n1 0 0 0 10 10 10 1 1 0\n$EndEntities\n')
        f.write('$Nodes\n1 11 1 11\n3 1 0 11\n' + ' '.join(str(i) for i in range(1, 12)) + '\n')
        for p in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (5, 0, 0), (6, 0, 0), (6, 1, 0), (5, 1, 0), (5.5, 0.5, 1)):
            f.write('%r %r %r\n' % p)
        f.write('$EndNodes\n$Elements\n2 2 1 2\n3 1 6 1\n1 1 2 3 4 5 6\n3 1 7 1\n2 7 8 9 10 11\n$EndElements\n')
    m = regions_from_msh.read_msh(msh)
    assert len(m['cells']) == 2 and m['volume_names'] == ['solid']
    regions, interfaces = regions_from_msh.split_regions(m, 'none', {}, 1e-9)
    assert interfaces == [] and [rr['kind'] for rr in regions] == ['solid']; r = regions[0]
    assert r['patches'] == [('defaultFaces', 'patch', 10)] and len(r['faces']) == 10 and len(r['neighbour']) == 0
    assert sorted(r['owner']) == [0] * 5 + [1] * 5, 'the prism and the pyramid contribute 5 faces each'
    cent, vol = regions_check.cell_geometry(r['points'], r['faces'], r['owner'], r['neighbour'], 2)
    for got, want in ((vol[0], 0.5), (vol[1], 1.0 / 3.0)):
        assert abs(got - want) <= 1e-14 * want, (got, want)
    for fi, face in enumerate(r['faces']):
        sf, cf = regions_check.face_geometry(r['points'], face)
        assert sf.dot(cf - cent[r['owner'][fi]]) > 0.0, 'face %d does not point out of its cell' % fi
    p = tool(msh, os.path.join(work, 'prism_pyr_out'), '--fluid', 'none'); assert p.returncode == 0 and p.stdout.count('not covered by any physical surface') == 1, p.stdout[-400:]
    with open(msh, encoding='utf-8') as g: text = g.read()
    assert text.count('3 1 6 1') == 1, 'the prism block header is not unique in the hand-written mesh'
    bad = os.path.join(work, 'bad_type.msh')
    with open(bad, 'w', encoding='utf-8', newline='\n') as f: f.write(text.replace('3 1 6 1', '3 1 11 1'))
    p = tool(bad, os.path.join(work, 'bad_type_out')); assert p.returncode == 1 and 'unsupported Gmsh element type 11' in p.stderr, (p.returncode, p.stderr[-300:])
    print('  [ok] face tables: prism V=1/2 and pyramid V=1/3, 5 faces each, every face outward, defaultFaces x 10')


def test_refusals(work, paths):
    """Req 6 + 8: every refusal names its reason and writes nothing; --fluid none."""
    out = os.path.join(work, 'refusal_out')
    for msh, extra, code, want in (
            (paths['two_region'], ('--fluent', 'x.msh'), 2, '--fluent is not supported'),
            (paths['two_region'], ('--fluid', 'nosuch'), 2, 'no volume named'),
            (paths['no_volume_name'], (), 1, 'no physical name'),
            (paths['no_physical_names'], (), 1, '$PhysicalNames')):
        p = tool(msh, out, *extra)
        assert p.returncode == code and want in p.stderr, (p.returncode, p.stderr[-300:])
    assert not os.path.isdir(out), 'a refusal created the output directory'
    p = tool(paths['two_region'], os.path.join(work, 'two_region')); assert p.returncode == 2 and '--overwrite' in p.stderr, (p.returncode, p.stderr[-300:])
    p = tool(paths['two_region'], os.path.join(work, 'fluid_none'), '--fluid', 'none')
    assert p.returncode == 0, p.stdout[-500:] + p.stderr[-400:]
    with open(os.path.join(work, 'fluid_none', 'regions.json'), encoding='utf-8') as f: man = json.load(f)
    assert [r['name'] for r in man['regions']] == ['fluid', 'building'] and [r['kind'] for r in man['regions']] == ['solid', 'solid'] and man['interfaces'][0]['patches'] == ['fluid_to_building', 'building_to_fluid']
    ck = run([sys.executable, CHECK, os.path.join(work, 'fluid_none', 'regions.json')]); assert ck.returncode == 0, ck.stdout[-300:]
    print('  [ok] 5 refusals write nothing; --fluid none: both regions solid, checker green')


def main(argv=None):
    ap = argparse.ArgumentParser(description='Self-test the M4 regions unit end to end.')
    ap.add_argument('--keep', nargs='?', const='', metavar='DIR', help='keep the work directory (bare: keep the temp dir; DIR: keep it there)'); ap.add_argument('--msh', metavar='PATH', help='an external two-region .msh to run the tool on (req 9)')
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    work = os.path.abspath(args.keep) if args.keep else tempfile.mkdtemp(prefix='regions_selftest_')
    print('  [..] work directory: %s' % work); os.makedirs(work, exist_ok=True)
    try:
        paths, counts = build_tet_meshes(work)
        hexc = build_hex_meshes(work)
        test_single_and_two_region(work, paths, counts)
        test_checker_refusals(work)
        test_hex_one_cell_thick(work, hexc)
        test_face_tables(work)
        test_refusals(work, paths)
        if args.msh:
            out = os.path.join(work, 'external')
            p = tool(args.msh, out, '--material', 'building=concrete'); assert p.returncode == 0, p.stdout[-500:] + p.stderr[-400:]
            with open(os.path.join(out, 'regions.json'), encoding='utf-8') as f: man = json.load(f)
            assert [r['name'] for r in man['regions']] == ['fluid', 'building'], man['regions']; assert man['interfaces'][0]['patches'] == ['fluid_to_building', 'building_to_fluid']
            ck = run([sys.executable, CHECK, os.path.join(out, 'regions.json')]); assert ck.returncode == 0, ck.stdout[-400:]
            print('  [ok] external mesh %s: two regions, conformal interface, checker green' % os.path.basename(args.msh))
        print('SELFTEST PASS'); code = 0
    except AssertionError as e:
        print('SELFTEST FAIL: %s' % (e,)); print('  work kept: %s' % work); code = 1
    if args.keep is None and code == 0: shutil.rmtree(work, ignore_errors=True)
    return code


if __name__ == '__main__':
    sys.exit(main())
