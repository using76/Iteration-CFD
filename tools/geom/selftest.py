#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.

"""selftest.py - builds its own inputs (gmsh) and runs geom_tool.py on them; --keep keeps the scratch dir."""

import argparse
import json
import os
import shutil
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


TESTS = (test_info_two_boxes, test_export_round_trip, test_sidecar_names, test_iges_surfaces_only, test_stl_discrete, test_refusals)


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
