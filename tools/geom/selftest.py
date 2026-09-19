#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.

"""selftest.py - builds its own inputs (gmsh) and runs geom_tool.py on them; --keep keeps the scratch dir."""

import argparse
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time

import gmsh

GEOM_TOOL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'geom_tool.py')


def build_two_boxes(path):
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.model.occ.addBox(0, 0, 0, 1000, 1000, 1000)
        gmsh.model.occ.addBox(1000, 0, 0, 2000, 1000, 1000)
        gmsh.model.occ.synchronize()
        gmsh.model.occ.fragment([(3, 1)], [(3, 2)])
        gmsh.model.occ.synchronize()
        gmsh.write(path)
    finally:
        gmsh.finalize()


def build_one_box(path):
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.model.occ.addBox(0, 0, 0, 1000, 1000, 1000)
        gmsh.model.occ.synchronize()
        gmsh.write(path)
    finally:
        gmsh.finalize()


def write_open_stl(path):
    lines = ['solid one', 'facet normal 0 0 1', 'outer loop', 'vertex 0 0 0',
             'vertex 1 0 0', 'vertex 0 1 0', 'endloop', 'endfacet', 'endsolid one']
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


RACECAR = os.path.join(os.path.dirname(GEOM_TOOL), '..', '..', 'cases', 'racecar.stl')


def cube_tris(origin=(0, 0, 0)):
    """The unit cube at origin as 12 outward triangles, two per face."""
    ox, oy, oz = origin
    p = lambda x, y, z: (ox + x, oy + y, oz + z)
    return [
        (p(0, 0, 0), p(0, 1, 0), p(1, 1, 0)), (p(0, 0, 0), p(1, 1, 0), p(1, 0, 0)),
        (p(0, 0, 1), p(1, 0, 1), p(1, 1, 1)), (p(0, 0, 1), p(1, 1, 1), p(0, 1, 1)),
        (p(0, 0, 0), p(1, 0, 0), p(1, 0, 1)), (p(0, 0, 0), p(1, 0, 1), p(0, 0, 1)),
        (p(0, 1, 0), p(0, 1, 1), p(1, 1, 1)), (p(0, 1, 0), p(1, 1, 1), p(1, 1, 0)),
        (p(0, 0, 0), p(0, 0, 1), p(0, 1, 1)), (p(0, 0, 0), p(0, 1, 1), p(0, 1, 0)),
        (p(1, 0, 0), p(1, 1, 0), p(1, 1, 1)), (p(1, 0, 0), p(1, 1, 1), p(1, 0, 1)),
    ]


def _tri_normal(t):
    (ax, ay, az), (bx, by, bz), (cx, cy, cz) = t
    u, v = (bx - ax, by - ay, bz - az), (cx - ax, cy - ay, cz - az)
    n = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
    m = math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])
    return (0.0, 0.0, 0.0) if m == 0 else (n[0] / m, n[1] / m, n[2] / m)


def write_stl_ascii(path, blocks):
    lines = []
    for name, tris in blocks:
        lines.append('solid %s' % name)
        for t in tris:
            lines.append('facet normal %s' % ' '.join('%.17g' % v for v in _tri_normal(t)))
            lines.append('outer loop')
            for v in t:
                lines.append('vertex %s' % ' '.join('%.17g' % c for c in v))
            lines.append('endloop')
            lines.append('endfacet')
        lines.append('endsolid %s' % name)
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')


def write_stl_binary(path, tris, name):
    with open(path, 'wb') as f:
        f.write(name.encode('ascii', 'replace')[:80].ljust(80, b'\0'))
        f.write(struct.pack('<I', len(tris)))
        for t in tris:
            f.write(struct.pack('<3f', 0.0, 0.0, 0.0))
            for v in t:
                f.write(struct.pack('<3f', *v))
            f.write(struct.pack('<H', 0))


def flip(tri):
    return (tri[0], tri[2], tri[1])


def repair(*args, expect=0):
    return run('repair', *args, expect=expect)


def run(*args, expect=0):
    t = time.time()
    p = subprocess.run([sys.executable, GEOM_TOOL] + list(args), capture_output=True,
                       text=True, encoding='utf-8', errors='replace')
    print('  [%s] geom_tool %s  %.1f s' % ('ok' if p.returncode == expect else 'FAIL', ' '.join(args), time.time() - t), flush=True)
    assert p.returncode == expect, (list(args), p.returncode, p.stderr[-400:])
    return p


def load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def write_sidecar(path, records):
    doc = {'version': 1, 'tool': 'geom_tool', 'units': 'm', 'scale': 0.001, 'ops': [], 'solids': records,
           'file': os.path.basename(path).replace('.geom.json', '')}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(doc, f, indent=1)


def rel(a, b):
    return abs(a - b) / abs(b)


def close3(a, b, tol):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def work(top, name):
    os.makedirs(os.path.join(top, name), exist_ok=True)
    return os.path.join(top, name)


def check_solids(rep, vols, names=None, matched=None, dupes='keep'):
    ss = rep['solids']
    assert len(ss) == len(vols) and [s['tag'] for s in ss] == [1, 2] and \
        [rel(s['volume'], v) <= 1e-9 for s, v in zip(ss, vols)] == [True] * len(vols), ss
    if names is not None:
        assert [s['name'] for s in ss] == names, ss
    if matched is not None:
        assert all(s['matched'] == matched for s in ss), ss
    if dupes != 'keep':
        assert rep['duplicates_removed'] == dupes, rep['duplicates_removed']


def test_info_two_boxes(top):
    d = work(top, 'info_two')
    two = os.path.join(d, 'two.step')
    build_two_boxes(two)
    r = run('info', two, '--json', os.path.join(d, 'i.json'))
    rep = load(os.path.join(d, 'i.json'))
    assert rep['format'] == 'step' and rep['scale'] == 0.001 and rep['units'] == 'm', rep
    assert rep['sidecar'] is None and rep['surfaces'] == [] and rep['discrete'] == [], rep
    check_solids(rep, (1.0, 2.0), names=['solid_1', 'solid_2'], matched='fallback',
                 dupes={'surfaces_before': 12, 'surfaces_after': 11})
    ss = rep['solids']
    assert all(s['material'] is None and s['n_faces'] == 6 and s['closed'] is True for s in ss), ss
    assert close3(ss[0]['centroid'], [0.5, 0.5, 0.5], 1e-9) and \
        close3(ss[1]['centroid'], [2.0, 0.5, 0.5], 1e-9) and \
        close3(ss[0]['bbox'], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0], 1e-6) and \
        close3(ss[1]['bbox'], [1.0, 0.0, 0.0, 3.0, 1.0, 1.0], 1e-6), ss
    # the bbox carries gmsh's FIXED OCC bounding-gap padding (1e-07 on every
    # getBoundingBox, probed); the centroid and volume checks carry none
    assert '2 solids' in r.stdout and '1 duplicate surface removed' in r.stdout, r.stdout[-400:]


def test_export_round_trip(top):
    d = work(top, 'roundtrip')
    two = os.path.join(d, 'two.step')
    build_two_boxes(two)
    rt = work(d, 'rt')
    j = os.path.join(rt, 'i.json')
    for ext, dupes in (('step', {'surfaces_before': 12, 'surfaces_after': 11}),
                       ('brep', {'surfaces_before': 11, 'surfaces_after': 11}), ('xao', None)):
        out = os.path.join(rt, 'two_rt.' + ext)
        run('export', two, '--out', out)
        assert os.path.isfile(out) and os.path.isfile(os.path.join(rt, 'two_rt.geom.json')), ext
        if ext == 'step':
            doc = load(os.path.join(rt, 'two_rt.geom.json'))
            assert (doc['version'], doc['scale'], doc['file'], doc['ops']) == \
                (1, 0.001, 'two_rt.step', []) and len(doc['solids']) == 2, doc
            assert [rel(s['volume'], v) <= 1e-9 for s, v in zip(doc['solids'], (1.0, 2.0))] == [True, True] \
                and close3(doc['solids'][0]['centroid'], [0.5, 0.5, 0.5], 1e-9) \
                and close3(doc['solids'][1]['centroid'], [2.0, 0.5, 0.5], 1e-9), doc['solids']
        run('info', out, '--json', j)
        rep = load(j)
        check_solids(rep, (1.0, 2.0), names=['solid_1', 'solid_2'], matched='centroid', dupes=dupes)
        assert rep['sidecar'] == 'two_rt.geom.json', rep['sidecar']


def test_sidecar_names(top):
    d = work(top, 'sidecar_names')
    two = os.path.join(d, 'two.step')
    build_two_boxes(two)
    sc1 = work(d, 'sc1')
    shutil.copy(two, os.path.join(sc1, 'two.step'))
    # the records of C6 with the TAGS SWAPPED: the centroids carry the identity
    write_sidecar(os.path.join(sc1, 'two.geom.json'), [
        {'tag': 2, 'name': 'left', 'material': None, 'volume': 1.0, 'centroid': [0.5, 0.5, 0.5]},
        {'tag': 1, 'name': 'right', 'material': 'steel', 'volume': 2.0, 'centroid': [2.0, 0.5, 0.5]}])
    run('info', os.path.join(sc1, 'two.step'), '--json', os.path.join(sc1, 'i.json'))
    s1, s2 = load(os.path.join(sc1, 'i.json'))['solids']
    assert (s1['tag'], s1['name'], s1['matched'], s2['name'], s2['material']) == \
        (1, 'left', 'centroid', 'right', 'steel'), (s1, s2)
    sc2 = work(d, 'sc2')
    shutil.copy(two, os.path.join(sc2, 'two.step'))
    write_sidecar(os.path.join(sc2, 'two.geom.json'), [
        {'tag': 1, 'name': 'left', 'material': None, 'volume': 1.0, 'centroid': [0.6, 0.5, 0.5]},
        {'tag': 2, 'name': 'right', 'material': None, 'volume': 2.0, 'centroid': [2.1, 0.5, 0.5]}])
    r = run('info', os.path.join(sc2, 'two.step'), '--json', os.path.join(sc2, 'i.json'))
    rep = load(os.path.join(sc2, 'i.json'))
    assert [s['name'] for s in rep['solids']] == ['left', 'right'] and \
        all(s['matched'] == 'tag' for s in rep['solids']) and 'matched by tag only' in r.stdout, \
        rep['solids']
    named = os.path.join(sc1, 'out', 'named.step')
    run('export', os.path.join(sc1, 'two.step'), '--out', named)
    doc = load(os.path.join(sc1, 'out', 'named.geom.json'))
    assert {s['name'] for s in doc['solids']} == {'left', 'right'} and \
        {s['name']: s['material'] for s in doc['solids']}['right'] == 'steel', doc['solids']
    run('info', named, '--json', os.path.join(sc1, 'out', 'i.json'))
    by_tag = {s['tag']: s for s in load(os.path.join(sc1, 'out', 'i.json'))['solids']}
    assert [by_tag[i]['name'] for i in (1, 2)] == ['left', 'right'] and \
        by_tag[2]['material'] == 'steel' and \
        all(s['matched'] == 'centroid' for s in by_tag.values()), by_tag


def test_iges_surfaces_only(top):
    d = work(top, 'iges')
    build_two_boxes(os.path.join(d, 'two.iges'))
    r = run('info', os.path.join(d, 'two.iges'), '--json', os.path.join(d, 'i.json'))
    rep = load(os.path.join(d, 'i.json'))
    assert rep['solids'] == [] and rep['discrete'] == [] and len(rep['surfaces']) == 11, rep
    area = sum(f['area'] for f in rep['surfaces'])
    assert rel(area, 15.0) <= 1e-9, area
    assert rep['duplicates_removed'] == {'surfaces_before': 12, 'surfaces_after': 11}, rep
    assert rep['note'] == 'IGES imports as surfaces only: no solids are listed', rep['note']
    assert 'surfaces only' in r.stdout, r.stdout[-400:]


def test_stl_discrete(top):
    d = work(top, 'stl')
    build_one_box(os.path.join(d, 'one.step'))
    run('export', os.path.join(d, 'one.step'), '--out', os.path.join(d, 'one.stl'))
    assert os.path.isfile(os.path.join(d, 'one.stl')) and not os.path.isfile(os.path.join(d, 'one.geom.json')), 'stl export outputs'
    run('info', os.path.join(d, 'one.stl'), '--json', os.path.join(d, 'i.json'))
    rep = load(os.path.join(d, 'i.json'))
    assert rep['format'] == 'stl' and rep['scale'] == 1.0 and rep['solids'] == [] \
        and rep['surfaces'] == [] and rep['duplicates_removed'] is None, rep
    k = rep['discrete'][0]
    assert k['n_triangles'] > 0 and k['closed'] is True and k['name'], k
    assert rel(k['volume'], 1.0) <= 1e-6, k['volume']
    assert close3(k['centroid'], [0.5, 0.5, 0.5], 1e-6) and \
        close3(k['bbox'], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0], 1e-6), k
    write_open_stl(os.path.join(d, 'open.stl'))
    run('info', os.path.join(d, 'open.stl'), '--json', os.path.join(d, 'i2.json'))
    k = load(os.path.join(d, 'i2.json'))['discrete'][0]
    assert k['n_triangles'] == 1 and k['closed'] is False and k['volume'] is None, k


def test_refusals(top):
    d = work(top, 'refuse')
    two = os.path.join(d, 'two.step')
    build_two_boxes(two)
    iges = os.path.join(d, 'two.iges')
    build_two_boxes(iges)
    open(os.path.join(d, 'two.obj'), 'w').close()
    assert 'unsupported extension' in run('info', os.path.join(d, 'two.obj'), expect=2).stderr, 'obj'
    os.remove(os.path.join(d, 'two.obj'))
    cases = [(('info', os.path.join(d, 'nope.step')), 'does not exist'),
             (('info', two, '--scale', '0'), None), (('export', two, '--out', two), None),
             (('export', two, '--out', os.path.join(d, 'x.iges')), 'IGES'),
             (('export', two, '--out', os.path.join(d, 'x.msh')), 'unsupported extension'),
             (('export', iges, '--out', os.path.join(d, 'x.step')), 'holds none')]
    for args, needle in cases:
        r = run(*args, expect=2)
        assert 'geom_tool:' in r.stderr and (needle is None or needle in r.stderr), r.stderr[-300:]
    assert sorted(os.listdir(d)) == ['two.iges', 'two.step'], os.listdir(d)
    open(os.path.join(top, 'empty.stl'), 'w').close()
    assert 'import failed' in run('info', os.path.join(top, 'empty.stl'), expect=1).stderr, 'empty stl'
    os.remove(os.path.join(top, 'empty.stl'))


def build_brep(path, boxes, cylinders=()):
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        for x, y, z, dx, dy, dz in boxes:
            gmsh.model.occ.addBox(x, y, z, dx, dy, dz)
        for x, y, z, dx, dy, dz, r in cylinders:
            gmsh.model.occ.addCylinder(x, y, z, dx, dy, dz, r)
        gmsh.model.occ.synchronize()
        gmsh.write(path)
    finally:
        gmsh.finalize()


def write_ops(path, ops):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'version': 1, 'ops': ops}, f, indent=1)
    return path


def tag_at(rep, centroid, tol=1e-6):
    hits = [s for s in rep['solids'] if close3(s['centroid'], centroid, tol)]
    assert len(hits) == 1, (centroid, [(s['tag'], s['centroid']) for s in rep['solids']])
    return hits[0]['tag']


def expect_refusal(src, ops, out, *needles):
    ops_path = write_ops(os.path.join(os.path.dirname(out), 'refuse_ops.json'), ops)
    r = run('edit', src, '--ops', ops_path, '--out', out, expect=2)
    for needle in needles:
        assert needle in r.stderr, (needle, r.stderr[-400:])
    assert not os.path.exists(out), out
    return r


def test_edit_refusals(top):
    d = work(top, 'edit_refuse')
    src = os.path.join(d, 'two.brep')
    build_brep(src, [(0, 0, 0, 1, 1, 1), (3, 0, 0, 1, 1, 1)])
    run('info', src, '--json', os.path.join(d, 'i.json'))
    rep = load(os.path.join(d, 'i.json'))
    assert [s['name'] for s in rep['solids']] == ['solid_1', 'solid_2'], rep['solids']
    out = os.path.join(d, 'edited.step')
    expect_refusal(src, [{'op': 'shrink'}], out, 'ops[0].op', 'shrink', 'set_material')
    expect_refusal(src, [{'op': 'rename', 'solids': ['solid_1'], 'name': 'x'}], out,
                   'ops[0].solids', 'did you mean "solid"')
    expect_refusal(src, [{'op': 'box', 'name': 'bx', 'origin': [0, 0, 0]}], out,
                   'ops[0].size')
    expect_refusal(src, [{'op': 'translate', 'solids': ['solid_1'], 'by': [1, 2]}], out,
                   'ops[0].by')
    expect_refusal(src, [{'op': 'rename', 'solid': 'nothing', 'name': 'x'}], out,
                   'nothing', 'solid_1')
    ops_path = write_ops(os.path.join(d, 'ops_same.json'),
                         [{'op': 'rename', 'solid': 'solid_1', 'name': 'x'}])
    r = run('edit', src, '--ops', ops_path, '--out', src, expect=2)
    assert 'same file' in r.stderr and os.path.isfile(src), r.stderr[-300:]
    expect_refusal(src, [{'op': 'rename', 'solid': 'solid_1', 'name': 'x'}],
                   os.path.join(d, 'x.stl'), '.stl')
    stl = os.path.join(d, 'in.stl')
    with open(stl, 'w', encoding='utf-8') as f:
        f.write('solid x\nendsolid x\n')
    expect_refusal(stl, [{'op': 'rename', 'solid': 'solid_1', 'name': 'x'}], out, 'surfaces')
    assert not os.path.exists(out), out


def test_edit_cut_closed_form(top):
    d = work(top, 'edit_cut')
    src = os.path.join(d, 'cut.brep')
    build_brep(src, [(0, 0, 0, 2, 1, 1)])
    run('info', src, '--json', os.path.join(d, 'i.json'))
    t = tag_at(load(os.path.join(d, 'i.json')), (1.0, 0.5, 0.5))
    ops_path = write_ops(os.path.join(d, 'ops.json'), [
        {'op': 'cylinder', 'name': 'hole', 'origin': [1, 0.5, -1], 'axis': [0, 0, 3],
         'radius': 0.2},
        {'op': 'rename', 'solid': t, 'name': 'block'},
        {'op': 'cut', 'object': ['block'], 'tools': ['hole']}])
    out = os.path.join(d, 'cut.step')
    run('edit', src, '--ops', ops_path, '--out', out)
    v_closed = 2.0 - math.pi * 0.04
    doc = load(os.path.join(d, 'cut.geom.json'))
    ss = doc['solids']
    assert len(ss) == 1 and ss[0]['name'] == 'block', ss
    assert rel(ss[0]['volume'], v_closed) <= 1e-9, (ss[0]['volume'], v_closed)
    assert close3(ss[0]['centroid'], (1.0, 0.5, 0.5), 1e-9), ss[0]
    assert [e['op']['op'] for e in doc['ops']] == ['cylinder', 'rename', 'cut'], doc['ops']
    run('info', out, '--json', os.path.join(d, 'i2.json'))
    s = load(os.path.join(d, 'i2.json'))['solids']
    assert len(s) == 1 and s[0]['name'] == 'block' and s[0]['matched'] == 'centroid', s
    assert rel(s[0]['volume'], v_closed) <= 1e-9, (s[0]['volume'], v_closed)
    assert close3(s[0]['centroid'], (1.0, 0.5, 0.5), 1e-9), s[0]
    print('  [ok] cut V=%.12f  sidecar rel=%.2e  info rel=%.2e'
          % (ss[0]['volume'], rel(ss[0]['volume'], v_closed), rel(s[0]['volume'], v_closed)),
          flush=True)


def test_edit_fragment_three_pieces(top):
    d = work(top, 'edit_frag')
    # the two operands OVERLAP, and M1's import runs removeAllDuplicates - on
    # overlapping solids OCC resolves the overlap itself, i.e. the input would
    # arrive pre-fragmented (probed: a brep of these three boxes imports as
    # FOUR solids). xao is the one input format the loader does not dedup.
    src = os.path.join(d, 'frag.xao')
    build_brep(src, [(0, 0, 0, 1, 1, 1), (0.5, 0, 0, 1, 1, 1), (5, 5, 5, 1, 1, 1)])
    run('info', src, '--json', os.path.join(d, 'i.json'))
    rep = load(os.path.join(d, 'i.json'))
    ops_path = write_ops(os.path.join(d, 'ops.json'), [
        {'op': 'rename', 'solid': tag_at(rep, (0.5, 0.5, 0.5)), 'name': 'a'},
        {'op': 'rename', 'solid': tag_at(rep, (1.0, 0.5, 0.5)), 'name': 'b'},
        {'op': 'rename', 'solid': tag_at(rep, (5.5, 5.5, 5.5)), 'name': 'c'},
        {'op': 'set_material', 'solid': 'c', 'material': 'steel'},
        {'op': 'fragment', 'object': ['a'], 'tools': ['b']}])
    out = os.path.join(d, 'frag.step')
    run('edit', src, '--ops', ops_path, '--out', out)
    doc = load(os.path.join(d, 'frag.geom.json'))
    ss = {s['name']: s for s in doc['solids']}
    assert set(ss) == {'a', 'a_b', 'b', 'c'} and len(doc['solids']) == 4, doc['solids']
    for nm, c in (('a', (0.25, 0.5, 0.5)), ('a_b', (0.75, 0.5, 0.5)),
                  ('b', (1.25, 0.5, 0.5))):
        assert rel(ss[nm]['volume'], 0.5) <= 1e-9, (nm, ss[nm]['volume'])
        assert close3(ss[nm]['centroid'], c, 1e-9), (nm, ss[nm]['centroid'])
    v_sum = ss['a']['volume'] + ss['a_b']['volume'] + ss['b']['volume']
    assert rel(v_sum, 1.5) <= 1e-9, v_sum
    assert rel(ss['c']['volume'], 1.0) <= 1e-9 and ss['c']['material'] == 'steel', ss['c']
    assert len(doc['ops']) == 5 and doc['ops'][4]['op']['op'] == 'fragment', doc['ops']
    print('  [ok] fragment V(a)=%.12f V(a_b)=%.12f V(b)=%.12f sum=%.12f'
          % (ss['a']['volume'], ss['a_b']['volume'], ss['b']['volume'], v_sum), flush=True)


def test_edit_rename_round_trip(top):
    d = work(top, 'edit_rt')
    src = os.path.join(d, 'rt.brep')
    build_brep(src, [(0, 0, 0, 1, 1, 1), (3, 0, 0, 1, 2, 1), (6, 0, 0, 1, 3, 1)])
    run('info', src, '--json', os.path.join(d, 'i.json'))
    rep = load(os.path.join(d, 'i.json'))
    op1 = [{'op': 'rename', 'solid': tag_at(rep, (0.5, 0.5, 0.5)), 'name': 'keep'},
           {'op': 'set_material', 'solid': 'keep', 'material': 'steel'},
           {'op': 'rename', 'solid': tag_at(rep, (3.5, 1.0, 0.5)), 'name': 'gone'},
           {'op': 'rename', 'solid': tag_at(rep, (6.5, 1.5, 0.5)), 'name': 'other'}]
    rt1 = os.path.join(d, 'rt1.step')
    run('edit', src, '--ops', write_ops(os.path.join(d, 'ops1.json'), op1), '--out', rt1)
    j1 = os.path.join(d, 'i1.json')
    run('info', rt1, '--json', j1)
    s1 = {s['name']: s for s in load(j1)['solids']}
    assert set(s1) == {'keep', 'gone', 'other'} and s1['keep']['material'] == 'steel', s1
    rt2 = os.path.join(d, 'rt2.step')
    run('edit', rt1, '--ops', write_ops(os.path.join(d, 'ops2.json'),
        [{'op': 'delete', 'solids': ['gone']}]), '--out', rt2)
    j2 = os.path.join(d, 'i2.json')
    run('info', rt2, '--json', j2)
    rep2 = load(j2)
    s2 = {s['name']: s for s in rep2['solids']}
    assert len(rep2['solids']) == 2 and {s['tag'] for s in rep2['solids']} == {1, 2}, \
        rep2['solids']
    assert all(s['matched'] == 'centroid' for s in rep2['solids']), rep2['solids']
    assert rel(s2['keep']['volume'], 1.0) <= 1e-9 and s2['keep']['material'] == 'steel', \
        s2['keep']
    assert close3(s2['keep']['centroid'], (0.5, 0.5, 0.5), 1e-9), s2['keep']
    assert rel(s2['other']['volume'], 3.0) <= 1e-9, s2['other']
    assert close3(s2['other']['centroid'], (6.5, 1.5, 0.5), 1e-9), s2['other']
    doc2 = load(os.path.join(d, 'rt2.geom.json'))
    assert len(doc2['ops']) == 5 and doc2['ops'][4]['op']['op'] == 'delete', doc2['ops']
    assert doc2['ops'][0]['op'] == op1[0], doc2['ops'][0]
    print('  [ok] round trip: tags after the 2nd edit: %s'
          % sorted(s['tag'] for s in rep2['solids']), flush=True)


def test_edit_transforms(top):
    d = work(top, 'edit_tf')
    src = os.path.join(d, 'tf.brep')
    build_brep(src, [(0, 0, 0, 2, 1, 1), (10, 0, 0, 1, 2, 3), (20, 0, 0, 1, 1, 1),
                     (30, 0, 0, 1, 1, 1), (60, 0, 0, 2, 2, 2)])
    run('info', src, '--json', os.path.join(d, 'i.json'))
    rep = load(os.path.join(d, 'i.json'))
    before = {'t': (1, 0.5, 0.5), 'r': (10.5, 1.0, 1.5), 's': (20.5, 0.5, 0.5),
              'm': (30.5, 0.5, 0.5), 'f': (61.0, 1.0, 1.0)}
    tags = {nm: tag_at(rep, c) for nm, c in before.items()}
    ops_path = write_ops(os.path.join(d, 'ops.json'),
        [{'op': 'rename', 'solid': tags[nm], 'name': nm} for nm in ('t', 'r', 's', 'm', 'f')]
        + [{'op': 'translate', 'solids': ['t'], 'by': [1, 2, 3]},
           {'op': 'rotate', 'solids': ['r'], 'point': [10, 0, 0], 'axis': [0, 0, 1],
            'angle_deg': 90},
           {'op': 'scale', 'solids': ['s'], 'point': [20, 0, 0], 'factors': [2, 3, 4]},
           {'op': 'mirror', 'solids': ['m'], 'plane': [1, 0, 0, -40]},
           {'op': 'scale', 'solids': ['f'], 'point': [60, 0, 0], 'factor': 0.5}])
    out = os.path.join(d, 'tf.step')
    run('edit', src, '--ops', ops_path, '--out', out)
    want = {'t': (2, (2, 2.5, 3.5)), 'r': (6, (9, 0.5, 1.5)), 's': (24, (21, 1.5, 2)),
            'm': (1, (49.5, 0.5, 0.5)), 'f': (1, (60.5, 0.5, 0.5))}
    docs = (load(os.path.join(d, 'tf.geom.json')),)
    run('info', out, '--json', os.path.join(d, 'i2.json'))
    docs = (docs[0], load(os.path.join(d, 'i2.json')))
    for doc in docs:
        ss = {s['name']: s for s in doc['solids']}
        assert set(ss) == set(want), sorted(ss)
        for nm, (v, c) in want.items():
            assert rel(ss[nm]['volume'], v) <= 1e-9, (nm, ss[nm]['volume'], v)
            for x, y in zip(ss[nm]['centroid'], c):
                assert abs(x - y) <= 1e-9 * (1 + abs(y)), (nm, ss[nm]['centroid'], c)
    sc_by_name = {s['name']: s for s in docs[0]['solids']}
    assert {nm: sc_by_name[nm]['tag'] for nm in tags} == tags, docs[0]['solids']
    print('  [ok] transforms centroids: %s'
          % '; '.join('%s (%.3f, %.3f, %.3f)' % (nm, *ss[nm]['centroid'])
                      for nm in ('t', 'r', 's', 'm', 'f')), flush=True)


def test_repair_weld(top):
    d = work(top, 'repair_weld')
    eps = 1e-8
    shifted = []
    for t, tri in enumerate(cube_tris()):
        row = []
        for k, v in enumerate(tri):
            i = 3 * t + k
            row.append((v[0] + eps * ((i % 5) - 2), v[1] + eps * ((i % 7) - 3),
                        v[2] + eps * ((i % 3) - 1)))
        shifted.append(tuple(row))
    src = os.path.join(d, 'weld.stl')
    write_stl_ascii(src, [('cube', shifted)])   # %.17g keeps every 1e-8 shift;
    out = os.path.join(d, 'weld_r.stl')         # float32 bytes could not carry them
    repair(src, '--out', out, '--json', os.path.join(d, 'r.json'), '--binary')
    rep = load(os.path.join(d, 'r.json'))
    assert rep['format_in'] == 'stl-ascii' and rep['format_out'] == 'stl-binary', rep
    assert rep['before']['open_edges'] == 36 and rep['before']['non_manifold_edges'] == 0, rep['before']
    w = rep['weld']
    assert w['points_bit_exact'] == 36 and w['points_welded'] == 8 and w['merged'] == 28, w
    assert w['max_move'] <= 1e-7 and rep['degenerate_dropped'] == 0, (w, rep['degenerate_dropped'])
    assert rep['after']['closed'] is True and rel(rep['after']['volume'], 1.0) <= 1e-6, rep['after']
    assert os.path.getsize(out) == 84 + 50 * 12, os.path.getsize(out)
    run('info', out, '--json', os.path.join(d, 'i.json'))
    k = load(os.path.join(d, 'i.json'))['discrete'][0]
    assert k['closed'] is True and rel(k['volume'], 1.0) <= 1e-6, k


def test_repair_hole3(top):
    d = work(top, 'repair_hole3')
    src = os.path.join(d, 'hole3.stl')
    write_stl_ascii(src, [('cube', cube_tris()[:-1])])
    out = os.path.join(d, 'hole3_r.stl')
    repair(src, '--out', out, '--json', os.path.join(d, 'r.json'))
    rep = load(os.path.join(d, 'r.json'))
    assert rep['before'] == {'open_edges': 3, 'non_manifold_edges': 0, 'closed': False}, rep['before']
    assert rep['patches'] == ['cube'], rep['patches']
    assert rep['holes']['filled'] == 1 and rep['holes']['filled_triangles'] == 1, rep['holes']
    assert rep['triangles_out'] == 12 and rep['after']['closed'] is True, rep
    assert rel(rep['after']['volume'], 1.0) <= 1e-9, rep['after']
    text = open(out, encoding='utf-8').read()
    assert 'solid cube' in text and 'endsolid cube' in text, text[:200]
    run('info', out, '--json', os.path.join(d, 'i.json'))
    k = load(os.path.join(d, 'i.json'))['discrete'][0]
    assert k['closed'] is True and rel(k['volume'], 1.0) <= 1e-9, k


def test_repair_flip(top):
    d = work(top, 'repair_flip')
    tris = cube_tris()
    tris[5] = flip(tris[5])
    src = os.path.join(d, 'flip.stl')
    write_stl_binary(src, tris, 'flipcube')
    out = os.path.join(d, 'flip_r.stl')
    repair(src, '--out', out, '--json', os.path.join(d, 'r.json'))
    rep = load(os.path.join(d, 'r.json'))
    assert rep['before']['open_edges'] == 0 and rep['before']['non_manifold_edges'] == 3, rep['before']
    o = rep['orientation']
    assert set(o) == {'reoriented_triangles', 'left_alone', 'left_alone_reason',
                      'reseeded_patches', 'flipped_components'}, o
    assert o['reoriented_triangles'] == 1 and o['left_alone'] == 0, o
    assert o['left_alone_reason'] == [] and o['flipped_components'] == 0, o
    assert isinstance(o['reseeded_patches'], int) and o['reseeded_patches'] >= 0, o
    assert rep['after']['closed'] is True and rel(rep['after']['volume'], 1.0) <= 1e-9, rep['after']


def test_repair_inward(top):
    d = work(top, 'repair_inward')
    src = os.path.join(d, 'in.stl')
    write_stl_binary(src, [flip(t) for t in cube_tris()], 'inward')
    out = os.path.join(d, 'in_r.stl')
    repair(src, '--out', out, '--json', os.path.join(d, 'r.json'), '--ascii')
    rep = load(os.path.join(d, 'r.json'))
    assert rep['before']['closed'] is True, rep['before']
    assert rep['orientation']['reoriented_triangles'] == 0 and \
        rep['orientation']['flipped_components'] == 1, rep['orientation']
    assert rel(rep['after']['volume'], 1.0) <= 1e-9 and rep['format_out'] == 'stl-ascii', rep
    run('info', src, '--json', os.path.join(d, 'i0.json'))
    k = load(os.path.join(d, 'i0.json'))['discrete'][0]
    assert k['closed'] is True and abs(k['volume'] + 1.0) <= 1e-9, k
    run('info', out, '--json', os.path.join(d, 'i.json'))
    k = load(os.path.join(d, 'i.json'))['discrete'][0]
    assert k['closed'] is True and rel(k['volume'], 1.0) <= 1e-9, k


def test_repair_patch_identity(top):
    d = work(top, 'repair_patch_identity')
    src = os.path.join(d, 'two.stl')
    write_stl_ascii(src, [('a', cube_tris((0, 0, 0))), ('b', cube_tris((3, 0, 0)))])
    out = os.path.join(d, 'two_r.stl')
    repair(src, '--out', out, '--json', os.path.join(d, 'r.json'))
    rep = load(os.path.join(d, 'r.json'))
    assert rep['n_components'] == 2 and rep['patches'] == ['a', 'b'], rep
    assert [c['patch'] for c in rep['components']] == ['a', 'b'], rep['components']
    assert rep['after']['closed'] is True and rel(rep['after']['volume'], 2.0) <= 1e-9, rep['after']
    text = open(out, encoding='utf-8').read()
    assert 'solid a' in text and 'solid b' in text, text[:200]
    run('info', out, '--json', os.path.join(d, 'i.json'))
    disc = load(os.path.join(d, 'i.json'))['discrete']
    assert [e['name'] for e in disc] == ['a', 'b'] and all(e['closed'] for e in disc), disc


def test_repair_reports_unrepairable(top):
    d = work(top, 'repair_unrepairable')
    src = os.path.join(d, 'mix.stl')
    b = [t for t in cube_tris((3, 0, 0)) if not all(abs(v[2] - 1.0) < 1e-12 for v in t)]
    write_stl_ascii(src, [('a', cube_tris()), ('b', b)])
    out = os.path.join(d, 'mix_r.stl')
    p = repair(src, '--out', out, '--json', os.path.join(d, 'r.json'),
               '--max-hole-edges', '3')
    rep = load(os.path.join(d, 'r.json'))
    assert rep['holes']['unfilled'] == [
        {'edges': 4, 'reason': 'loop of 4 edges > max_hole_edges 3'}], rep['holes']
    assert rep['holes']['per_hole'] == [
        {'edges': 4, 'outcome': 'skipped', 'triangles': 0,
         'reason': 'loop of 4 edges > max_hole_edges 3'}], rep['holes']
    assert rep['components'][0]['closed'] is True and rep['components'][1]['closed'] is False, rep['components']
    assert rep['components'][1]['open_edges'] == 4, rep['components']
    assert rep['after']['volume'] is None and rep['triangles_out'] == 22, rep
    assert 'component 1 (b, 10 triangle(s)): 4 open, 0 non-manifold' in p.stdout, p.stdout
    src2 = os.path.join(d, 'fin.stl')
    v0, v1, _ = cube_tris()[0]
    write_stl_binary(src2, cube_tris() + [(v0, v1, (0.5, 0.5, 2.0))], 'fincube')
    p = repair(src2, '--out', os.path.join(d, 'fin_r.stl'), '--json', os.path.join(d, 'r2.json'))
    rep = load(os.path.join(d, 'r2.json'))
    assert rep['n_components'] == 1 and rep['before']['non_manifold_edges'] == 1, rep
    assert rep['after']['non_manifold_edges'] == 1 and rep['after']['open_edges'] == 2, rep['after']
    assert rep['after']['closed'] is False and rep['orientation']['flipped_components'] == 0, rep
    assert rep['holes']['filled'] == 0 and len(rep['holes']['unfilled']) == 1, rep['holes']
    assert rep['holes']['unfilled'][0]['reason'].startswith('boundary vertex with'), rep['holes']
    assert rep['triangles_out'] == 13, rep
    assert 'component 0 (' in p.stdout and ': 2 open, 1 non-manifold' in p.stdout, p.stdout


def test_repair_racecar(top):
    d = work(top, 'repair_racecar')
    out = os.path.join(d, 'racecar_r.stl')
    repair(RACECAR, '--out', out, '--json', os.path.join(d, 'r.json'))
    rep = load(os.path.join(d, 'r.json'))
    assert rep['format_in'] == 'stl-binary' and rep['triangles_in'] == 1040, rep
    assert rep['patches'] == ['racecar'], rep['patches']
    assert rep['before'] == {'open_edges': 0, 'non_manifold_edges': 0, 'closed': True}, rep['before']
    w = rep['weld']
    assert w['points_raw'] == 3120 and w['points_bit_exact'] == 564 and \
        w['merged'] == 0 and w['max_move'] == 0.0, w
    assert rep['degenerate_dropped'] == 0 and rep['n_components'] == 22, rep
    assert rep['orientation']['reoriented_triangles'] == 0 and \
        rep['orientation']['flipped_components'] == 4, rep['orientation']
    flipped = sorted(c['n_triangles'] for c in rep['components'] if c['flipped'])
    assert flipped == [20, 36, 36, 348], flipped
    assert rep['after']['closed'] is True and \
        rel(rep['after']['volume'], 2.7319858267646144e-3) <= 1e-6, rep['after']
    assert close3(rep['bbox'][:3], [0.1, 0.0015, 0.39044], 1e-5), rep['bbox']
    assert rel(rep['diagonal'], 0.70532065) <= 1e-6, rep['diagonal']
    assert os.path.getsize(out) == 84 + 50 * 1040, os.path.getsize(out)
    run('info', out, '--json', os.path.join(d, 'i.json'))
    k = load(os.path.join(d, 'i.json'))['discrete'][0]
    assert k['closed'] is True and k['volume'] > 0, k
    run('info', RACECAR, '--json', os.path.join(d, 'i0.json'))
    k = load(os.path.join(d, 'i0.json'))['discrete'][0]
    assert abs(k['volume'] + 7.887749871e-04) <= 1e-9, k['volume']


def test_repair_refusals(top):
    d = work(top, 'repair_refusals')
    bad = os.path.join(d, 'x.step')
    open(bad, 'wb').close()
    p = repair(bad, expect=2)
    assert "unsupported extension '.step'" in p.stderr, p.stderr
    src = os.path.join(d, 'cube.stl')
    write_stl_binary(src, cube_tris(), 'cube')
    p = repair(src, '--out', src, '--json', os.path.join(d, 'r.json'), expect=2)
    assert 'refusing to overwrite' in p.stderr, p.stderr
    two = os.path.join(d, 'two.stl')
    write_stl_ascii(two, [('a', cube_tris()), ('b', cube_tris((3, 0, 0)))])
    p = repair(two, '--out', os.path.join(d, 'o.stl'), '--binary', expect=2)
    assert 'has 2: a, b' in p.stderr, p.stderr
    p = repair(os.path.join(d, 'missing.stl'), expect=2)
    assert 'does not exist' in p.stderr, p.stderr
    one = os.path.join(d, 'one.stl')
    write_open_stl(one)
    p = repair(one, '--out', os.path.join(d, 'one_r.stl'), '--json', os.path.join(d, 'r3.json'))
    rep = load(os.path.join(d, 'r3.json'))
    assert rep['n_components'] == 1 and rep['holes']['filled'] == 1, rep
    assert rep['after']['closed'] is True and abs(rep['after']['volume']) <= 1e-12, rep['after']
    deg = os.path.join(d, 'deg.stl')
    write_stl_binary(deg, cube_tris() + [((0, 0, 0), (1, 0, 0), (2, 0, 0))], 'deg')
    repair(deg, '--out', os.path.join(d, 'deg_r.stl'), '--json', os.path.join(d, 'r4.json'))
    rep = load(os.path.join(d, 'r4.json'))
    assert rep['degenerate_dropped'] == 1 and rep['triangles_out'] == 12 and \
        rep['after']['closed'] is True, rep
    p = repair(src, '--weld', '-1', expect=2)
    assert '--weld must be >= 0' in p.stderr, p.stderr
    p = repair(src, '--max-hole-edges', '2', expect=2)
    assert '--max-hole-edges must be >= 3' in p.stderr, p.stderr


def test_repair_fill_guard(top):
    """A fin glued on one cube edge (that edge used three times) must not
    make non-manifold worse: the clean 3-edge hole away from the fin fills,
    the fin's dead-end chain is reported, and after.nm == before.nm.  The
    walk only ever closes loops of single-use edges, so the guard's
    skip-with-reason branch is unreachable today; it stands against a
    future walk change.  (A tetrahedron base cannot host this: every face
    shares a vertex with the fin edge, and the pinched hole never fills.)"""
    d = work(top, 'repair_fill_guard')
    tris = cube_tris()[:-1]
    tris.append(((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (-0.5, 0.5, 0.5)))
    src = os.path.join(d, 'guard.stl')
    write_stl_ascii(src, [('cube', tris)])
    out = os.path.join(d, 'guard_r.stl')
    repair(src, '--out', out, '--json', os.path.join(d, 'r.json'))
    rep = load(os.path.join(d, 'r.json'))
    assert rep['before'] == {'open_edges': 5, 'non_manifold_edges': 1,
                             'closed': False}, rep['before']
    assert rep['holes']['filled'] == 1 and rep['holes']['filled_triangles'] == 1, rep['holes']
    assert len(rep['holes']['unfilled']) == 1, rep['holes']
    assert rep['holes']['unfilled'][0]['reason'].startswith('boundary vertex with'), rep['holes']
    assert rep['after']['non_manifold_edges'] == rep['before']['non_manifold_edges'], rep['after']
    assert rep['after']['open_edges'] == 2 and rep['after']['closed'] is False, rep['after']
    assert rep['degenerate_dropped'] == 0 and rep['triangles_out'] == 13, rep


def test_repair_orient_cost(top):
    """A fin triangle glued to one tetrahedron edge in the same direction as
    a face: that edge is used three ways, so the BFS never crosses it, so
    the fin is reached only by a fresh seed and is counted in
    `reseeded_patches`.  The tetrahedron keeps its outward winding - the
    output's signed 6*volume stays +1 (the fin holds the origin, so it adds
    nothing to the sum) - and non_manifold_edges is exactly the input's."""
    d = work(top, 'repair_orient_cost')
    A, B, C, D = (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
    E = (0.5, 0.5, -1.0)
    tris = [(A, C, B), (A, B, D), (A, D, C), (B, C, D), (A, B, E)]
    src = os.path.join(d, 'tet.stl')
    write_stl_ascii(src, [('tet', tris)])
    out = os.path.join(d, 'tet_r.stl')
    p = repair(src, '--out', out, '--json', os.path.join(d, 'r.json'), '--ascii')
    rep = load(os.path.join(d, 'r.json'))
    assert p.returncode == 0, p
    assert rep['before'] == {'open_edges': 2, 'non_manifold_edges': 1,
                             'closed': False}, rep['before']
    assert rep['after']['non_manifold_edges'] == \
        rep['before']['non_manifold_edges'], rep['after']
    assert rep['after']['open_edges'] == 2 and rep['after']['closed'] is False, rep['after']
    assert rep['orientation']['reseeded_patches'] >= 1, rep['orientation']
    assert rep['orientation']['reoriented_triangles'] == 0, rep['orientation']
    assert rep['triangles_out'] == 5, rep
    text = open(out, encoding='utf-8').read()
    vs = [tuple(float(x) for x in l.split()[1:4])
          for l in text.splitlines() if l.startswith('vertex ')]
    vol6 = sum(va[0] * (vb[1] * vc[2] - vb[2] * vc[1])
               + va[1] * (vb[2] * vc[0] - vb[0] * vc[2])
               + va[2] * (vb[0] * vc[1] - vb[1] * vc[0])
               for va, vb, vc in zip(vs[0::3], vs[1::3], vs[2::3]))
    assert abs(vol6 - 1.0) <= 1e-12, vol6    # the tet's faces are still outward


def test_repair_hole4(top):
    d = work(top, 'repair_hole4')
    tris = [t for t in cube_tris() if not all(abs(v[2] - 1.0) < 1e-12 for v in t)]
    src = os.path.join(d, 'hole4.stl')
    write_stl_ascii(src, [('cube', tris)])
    out = os.path.join(d, 'hole4_r.stl')
    repair(src, '--out', out, '--json', os.path.join(d, 'r.json'))
    rep = load(os.path.join(d, 'r.json'))
    assert rep['before'] == {'open_edges': 4, 'non_manifold_edges': 0,
                             'closed': False}, rep['before']
    assert rep['patches'] == ['cube'], rep['patches']
    assert rep['holes']['filled'] == 1 and rep['holes']['filled_triangles'] == 2, rep['holes']
    assert rep['triangles_out'] == 12 and rep['after']['closed'] is True, rep
    assert rel(rep['after']['volume'], 1.0) <= 1e-9, rep['after']
    assert rep['holes']['per_hole'] == [
        {'edges': 4, 'outcome': 'filled', 'triangles': 2, 'reason': None}], rep['holes']
    text = open(out, encoding='utf-8').read()
    assert 'solid cube' in text and 'endsolid cube' in text, text[:200]
    run('info', out, '--json', os.path.join(d, 'i.json'))
    k = load(os.path.join(d, 'i.json'))['discrete'][0]
    assert k['closed'] is True and rel(k['volume'], 1.0) <= 1e-9, k


def test_repair_hole_bent(top):
    d = work(top, 'repair_hole_bent')
    gone = {(1.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1.0, 0.0, 1.0)}
    tris = [t for t in cube_tris()
            if not (all(abs(v[2] - 1.0) < 1e-12 for v in t) or set(t) == gone)]
    src = os.path.join(d, 'hole_bent.stl')
    write_stl_ascii(src, [('cube', tris)])
    out = os.path.join(d, 'hole_bent_r.stl')
    repair(src, '--out', out, '--json', os.path.join(d, 'r.json'))
    rep = load(os.path.join(d, 'r.json'))
    assert rep['before']['open_edges'] == 5 and rep['before']['non_manifold_edges'] == 0, rep['before']
    assert rep['holes']['filled'] == 1 and rep['holes']['filled_triangles'] == 3, rep['holes']
    assert rep['triangles_out'] == 12 and rep['after']['closed'] is True, rep
    assert rel(rep['after']['volume'], 1.0) <= 1e-9, rep['after']


TESTS = (test_info_two_boxes, test_export_round_trip, test_sidecar_names, test_iges_surfaces_only, test_stl_discrete, test_refusals, test_edit_refusals, test_edit_cut_closed_form, test_edit_fragment_three_pieces, test_edit_rename_round_trip, test_edit_transforms, test_repair_weld, test_repair_hole3, test_repair_flip, test_repair_inward, test_repair_patch_identity, test_repair_reports_unrepairable, test_repair_racecar, test_repair_refusals, test_repair_fill_guard, test_repair_orient_cost, test_repair_hole4, test_repair_hole_bent)


def main(argv=None):
    ap = argparse.ArgumentParser(description='Self-test geom_tool: build, run, assert.')
    ap.add_argument('--keep', action='store_true', help='keep the scratch directory')
    args = ap.parse_args(argv)
    top = tempfile.mkdtemp(prefix='geom_tool_selftest_')
    for t in TESTS:
        try:
            t(top)
        except AssertionError as e:
            print('SELFTEST FAIL: %s\nthe scratch dir is kept: %s' % (e, top))
            return 1
        print('[ok] %s' % t.__name__, flush=True)
    print('SELFTEST PASS')
    if not args.keep:
        shutil.rmtree(top, ignore_errors=True)
    return 0


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.exit(main())
