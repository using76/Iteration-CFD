#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""
selftest_recipes.py - meshes every recipe of tools/mesh/examples through M4
and asserts the M5 table: the Turek-Hron layouts at levels 1 and 2 (3 with
--full), the three solid gates, and the two case skeletons' parse + numbers.

    python tools/mesh/examples/selftest_recipes.py [--keep] [--full]

Everything builds under tempfile.mkdtemp(prefix='recipes_selftest_'); --keep
preserves it for inspection instead of deleting it after a pass. --full adds
Turek-Hron level 3 and the three solids at level 2.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

import recipe

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = recipe.repo_root()
CHECK = os.path.join(ROOT, 'tools', 'mesh', 'regions_check.py')


def run_recipe(name, level, out):
    t = time.time()
    p = subprocess.run([sys.executable, os.path.join(HERE, '%s.py' % name),
                        '--level', str(level), '--out', out],
                       capture_output=True, text=True, encoding='utf-8', errors='replace')
    print('  [%s] %s --level %d  %.1f s' % (
        'ok' if p.returncode == 0 else 'FAIL', name, level, time.time() - t), flush=True)
    if p.returncode != 0:
        print('--- %s stdout ---' % name)
        print(p.stdout[-4000:])
        print('--- %s stderr ---' % name)
        print(p.stderr[-2000:])
    return p


def load_jsonc(path):
    """Comments and trailing commas stripped, then json.loads."""
    with open(path, encoding='utf-8') as f:
        txt = re.sub(r'/\*.*?\*/', '', f.read(), flags=re.S)
    txt = re.sub(r'//[^\n]*', '', txt)
    return json.loads(re.sub(r',\s*([}\]])', r'\1', txt))


def checker(out):
    return subprocess.run([sys.executable, CHECK, os.path.join(out, 'regions.json')],
                          capture_output=True, text=True, encoding='utf-8', errors='replace')


def load_regions(out):
    with open(os.path.join(out, 'regions.json'), encoding='utf-8') as f:
        return json.load(f)


def sidecar(out, name):
    with open(os.path.join(out, '%s.json' % name), encoding='utf-8') as f:
        return json.load(f)


def zone_of(centres, zones):
    """S9's rule: cell in zone iff min[i] <= c[i] <= max[i]; -1 none, -2 two."""
    idx = np.full(len(centres), -1, dtype=np.int64)
    for zi, z in enumerate(zones.values()):
        lo, hi = np.asarray(z['bounds']['min'], dtype=float), np.asarray(z['bounds']['max'], dtype=float)
        inside = np.all((centres >= lo) & (centres <= hi), axis=1)
        idx[inside & (idx >= 0)] = -2
        idx[inside & (idx == -1)] = zi
    return idx


def check_repo_root_is_the_repo():
    assert os.path.isfile(os.path.join(recipe.repo_root(), 'tools', 'mesh', 'step_mesh.py')), \
        'repo_root() is not the repo: %s' % recipe.repo_root()


def check_th_layout(out, label):
    """Requirement 1 at any level: the files, the checker, the regions, the interface."""
    for rel in ('regions.json', 'turek_hron.msh', 'turek_hron.json',
                'fluid/polyMesh/points', 'fluid/polyMesh/boundary',
                'flap/polyMesh/points', 'flap/polyMesh/boundary'):
        assert os.path.isfile(os.path.join(out, rel)), '%s: %s missing' % (label, rel)
    p = checker(out)
    assert p.returncode == 0, '%s: regions_check exited %d\n%s' % (
        label, p.returncode, (p.stdout + p.stderr)[-2000:])
    regions = load_regions(out)
    got = {r['name']: r['kind'] for r in regions['regions']}
    assert got == {'fluid': 'fluid', 'flap': 'solid'}, '%s: regions %s' % (label, got)
    its = regions['interfaces']
    assert len(its) == 1 and its[0]['patches'] == ['fluid_to_flap', 'flap_to_fluid'], \
        '%s: interfaces %s' % (label, its)
    return regions


def check_th_pairing_one_cell(out, regions, label):
    """Requirements 3 + 4: the pairing worsts by index, one cell thick, patch types."""
    it = regions['interfaces'][0]
    fa = recipe.patch_faces(recipe.load_region(out, regions, 'fluid'), it['patches'][0])[1]
    fb = recipe.patch_faces(recipe.load_region(out, regions, 'flap'), it['patches'][1])[1]
    assert len(fa) == len(fb) == it['faces'], '%s: %d/%d faces vs %d' % (
        label, len(fa), len(fb), it['faces'])
    for region in ('fluid', 'flap'):
        for patch in ('empty_front', 'empty_back'):
            pm = recipe.load_region(out, regions, region)
            typ, ids = recipe.patch_faces(pm, patch)
            assert typ == 'empty', '%s: %s %s is typed %r' % (label, region, patch, typ)
            assert len(ids) == pm['n_cells'], '%s: %s %s: %d faces, %d cells' % (
                label, region, patch, len(ids), pm['n_cells'])
    pe = recipe.pairing_error(out, regions)[0]
    assert pe['max_centroid'] <= 1e-12, '%s: max_centroid %r' % (label, pe['max_centroid'])
    assert pe['max_area_rel'] <= 1e-12, '%s: max_area_rel %r' % (label, pe['max_area_rel'])
    assert pe['min_opposed'] >= 1.0 - 1e-12, '%s: min_opposed %r' % (label, pe['min_opposed'])
    return pe


def check_th_elements(side, label):
    """Requirement 4's element half, at every level: types 5/6 only, hex >= 95 %."""
    for name in ('fluid', 'flap'):
        el = side['regions'][name]['elements']
        assert set(el) <= {'5', '6'}, '%s: %s element types %s' % (label, name, el)
        frac = el.get('5', 0) / max(1, sum(el.values()))
        assert frac >= 0.95, '%s: %s hex fraction %.3f' % (label, name, frac)


def check_th_counts_elements(side, label):
    """Requirements 4 + 5's band gate: the cells, the element types, hex >= 95 %."""
    cells = {name: side['regions'][name]['cells'] for name in ('fluid', 'flap')}
    assert 5000 <= cells['fluid'] <= 8000, '%s: fluid cells %d' % (label, cells['fluid'])
    assert 200 <= cells['flap'] <= 400, '%s: flap cells %d' % (label, cells['flap'])
    check_th_elements(side, label)
    return cells


def check_th_cylinder(out, regions, side, least, label):
    """Requirement 2 / 6's cylinder gate, measured on the polyMesh and the sidecar."""
    pm = recipe.load_region(out, regions, 'fluid')
    _, ids = recipe.patch_faces(pm, 'cylinder')
    assert len(ids) >= least, '%s: cylinder has %d faces, wants >= %d' % (label, len(ids), least)
    assert side['cylinder_faces'] == len(ids), '%s: sidecar cylinder_faces %r' % (
        label, side['cylinder_faces'])


def check_solid(out, name, region, side_keys):
    """The shape the three solid recipes share: checker 0, one solid region, sidecar keys."""
    p = checker(out)
    assert p.returncode == 0, '%s L: regions_check exited %d\n%s' % (
        name, p.returncode, (p.stdout + p.stderr)[-2000:])
    got = {r['name']: r['kind'] for r in load_regions(out)['regions']}
    assert got == {region: 'solid'}, '%s: regions %s' % (name, got)
    side = sidecar(out, name)
    for key in ('recipe', 'level', 'gmsh', 'units', 'timings_s') + tuple(side_keys):
        assert key in side, '%s sidecar lacks %r' % (name, key)
    assert side['recipe'] == name and side['units'] == 'm', '%s sidecar header' % name
    return side


def check_bimetal(out):
    """Requirement 8: the zones partition, recomputed from cell_centres with S9's rule."""
    side = check_solid(out, 'bimetal_strip', 'strip', ('L', 'w', 'h1', 'h2', 'n', 'zones'))
    pm = recipe.load_region(out, load_regions(out), 'strip')
    centres = recipe.cell_centres(pm)
    idx = zone_of(centres, side['zones'])
    bad = int((idx < 0).sum())
    assert bad == 0, 'bimetal: %d cells in no zone or in two zones' % bad
    nx, ny, nz = side['n']
    half = nx * ny * nz // 2
    counts = {}
    for zi, name in enumerate(('lower', 'upper')):
        counts[name] = int((idx == zi).sum())
        assert counts[name] == side['zones'][name]['cells'] == half, \
            'bimetal zone %s: %d cells, sidecar %d, half %d' % (
                name, counts[name], side['zones'][name]['cells'], half)
    assert (centres[idx == 0][:, 2] < side['h1']).all(), 'bimetal: a lower centroid at or above h1'
    assert (centres[idx == 1][:, 2] > side['h1']).all(), 'bimetal: an upper centroid at or below h1'
    return counts


def check_thick(out, level):
    """Requirement 9 (level 1) / the level-2 rerun: cells = nr*nt*nz, six patches, no other."""
    side = check_solid(out, 'thick_cylinder', 'cylinder', ('r_in', 'r_out', 'length', 'n'))
    nr, nt, nz = ((8, 12, 2), (16, 24, 4), (32, 48, 8))[level - 1]
    assert side['n'] == [nr, nt, nz], 'thick_cylinder: sidecar n %s' % side['n']
    pm = recipe.load_region(out, load_regions(out), 'cylinder')
    assert pm['n_cells'] == nr * nt * nz, 'thick_cylinder: %d cells, wants %d' % (
        pm['n_cells'], nr * nt * nz)
    got = {p['name']: p['nFaces'] for p in pm['patches']}
    want = {'inner': nt * nz, 'outer': nt * nz, 'symmetry_x0': nr * nz,
            'symmetry_y0': nr * nz, 'zmin': nr * nt, 'zmax': nr * nt}
    assert got == want, 'thick_cylinder: patches %s, wants %s' % (got, want)
    return {'cells': pm['n_cells'], 'patches': got}


def check_cant(out, level):
    """Requirement 10 (level 1) / the level-2 rerun: cells = nx*ny*nz, the six patch faces."""
    side = check_solid(out, 'cantilever', 'beam', ('L', 'b', 'h', 'n', 'I', 'A', 'beta1_L',
                                                   'f1_formula', 'w_tip_formula'))
    nx, ny, nz = ((40, 4, 4), (80, 8, 8), (160, 16, 16))[level - 1]
    assert side['n'] == [nx, ny, nz], 'cantilever: sidecar n %s' % side['n']
    pm = recipe.load_region(out, load_regions(out), 'beam')
    assert pm['n_cells'] == nx * ny * nz, 'cantilever: %d cells, wants %d' % (
        pm['n_cells'], nx * ny * nz)
    got = {p['name']: p['nFaces'] for p in pm['patches']}
    want = {'fixed_end': ny * nz, 'free_end': ny * nz, 'bottom': nx * ny,
            'top': nx * ny, 'side_y0': nx * nz, 'side_y1': nx * nz}
    assert got == want, 'cantilever: patches %s, wants %s' % (got, want)
    return {'cells': pm['n_cells'], 'patches': got}


def check_case_skeletons():
    """Requirement 12: JSONC parses, the mesh paths, the benchmark numbers as spelled."""
    fsi = load_jsonc(os.path.join(ROOT, 'cases', 'turekHron', 'fsi1.jsonc'))
    cfd = load_jsonc(os.path.join(ROOT, 'cases', 'turekHron', 'cfd1.jsonc'))
    assert fsi['mesh']['regions'] == 'mesh/regions.json', 'fsi1: mesh.regions %r' % fsi['mesh']
    assert cfd['regions'][0]['mesh']['polyMesh'] == 'mesh/fluid/polyMesh', \
        'cfd1: polyMesh %r' % cfd['regions'][0]['mesh']
    ref = fsi['reference']
    assert ref['ux_A'] == 2.270493e-05, 'fsi1: ux_A %r' % ref['ux_A']
    assert ref['uy_A'] == 8.208773e-04, 'fsi1: uy_A %r' % ref['uy_A']
    assert ref['drag'] == 1.429426e+01, 'fsi1: drag %r' % ref['drag']
    assert ref['lift'] == 7.637460e-01, 'fsi1: lift %r' % ref['lift']
    assert cfd['reference']['drag'] == 14.2929, 'cfd1: drag %r' % cfd['reference']['drag']
    assert cfd['reference']['lift'] == 1.11905, 'cfd1: lift %r' % cfd['reference']['lift']


def main(argv=None):
    ap = argparse.ArgumentParser(description='Mesh every recipe and assert the M5 table.')
    ap.add_argument('--keep', action='store_true', help='keep the scratch directory after a pass')
    ap.add_argument('--full', action='store_true', help='add Turek-Hron level 3 and the solids at level 2')
    args = ap.parse_args(argv)
    work = tempfile.mkdtemp(prefix='recipes_selftest_')
    t_all = time.time()
    try:
        check_repo_root_is_the_repo()
        print('  [--] scratch: %s' % work)

        th1 = os.path.join(work, 'th_L1')
        p = run_recipe('turek_hron', 1, th1)
        assert p.returncode == 0, 'turek_hron --level 1 exited %d' % p.returncode
        assert p.stdout.count('[check] OK') >= 2, 'th L1: fewer than two [check] OK lines'
        assert '[recipe] region fluid:' in p.stdout and '[recipe] pairing fluid_to_flap:' in p.stdout, \
            'th L1: the [recipe] lines are missing'
        regions = check_th_layout(th1, 'th L1')
        pe = check_th_pairing_one_cell(th1, regions, 'th L1')
        side = sidecar(th1, 'turek_hron')
        cells = check_th_counts_elements(side, 'th L1')
        check_th_cylinder(th1, regions, side, 40, 'th L1')
        print('  [ok] th L1: fluid %d / flap %d hex, cylinder %d, interface %d, '
              'pairing %.1e / %.1e / %.4f' % (
                  cells['fluid'], cells['flap'], side['cylinder_faces'], side['interface_faces'],
                  pe['max_centroid'], pe['max_area_rel'], pe['min_opposed']))

        th2 = os.path.join(work, 'th_L2')
        p = run_recipe('turek_hron', 2, th2)
        assert p.returncode == 0, 'turek_hron --level 2 exited %d' % p.returncode
        regions = check_th_layout(th2, 'th L2')
        check_th_pairing_one_cell(th2, regions, 'th L2')
        check_th_elements(sidecar(th2, 'turek_hron'), 'th L2')
        check_th_cylinder(th2, regions, sidecar(th2, 'turek_hron'), 80, 'th L2')
        print('  [ok] th L2: layout, pairing, one cell thick, cylinder >= 80')

        bim1 = os.path.join(work, 'bim_L1')
        assert run_recipe('bimetal_strip', 1, bim1).returncode == 0, 'bimetal L1 failed'
        print('  [ok] bimetal L1 zones: %s' % check_bimetal(bim1))
        cyl1 = os.path.join(work, 'cyl_L1')
        assert run_recipe('thick_cylinder', 1, cyl1).returncode == 0, 'thick_cylinder L1 failed'
        print('  [ok] thick_cylinder L1: %s' % check_thick(cyl1, 1))
        beam1 = os.path.join(work, 'beam_L1')
        assert run_recipe('cantilever', 1, beam1).returncode == 0, 'cantilever L1 failed'
        print('  [ok] cantilever L1: %s' % check_cant(beam1, 1))
        check_case_skeletons()
        print('  [ok] fsi1.jsonc / cfd1.jsonc parse; the reference numbers are verbatim')

        if args.full:
            th3 = os.path.join(work, 'th_L3')
            t3 = time.time()
            p = run_recipe('turek_hron', 3, th3)
            took = time.time() - t3
            assert p.returncode == 0, 'turek_hron --level 3 exited %d' % p.returncode
            regions = check_th_layout(th3, 'th L3')
            pe = check_th_pairing_one_cell(th3, regions, 'th L3')
            side3 = sidecar(th3, 'turek_hron')
            check_th_elements(side3, 'th L3')
            print('  [ok] th L3: fluid %d / flap %d cells, %.1f s recipe, timings %s, '
                  'pairing %.1e / %.1e / %.4f' % (
                      side3['regions']['fluid']['cells'], side3['regions']['flap']['cells'],
                      took, side3['timings_s'], pe['max_centroid'], pe['max_area_rel'],
                      pe['min_opposed']))
            cyl2 = os.path.join(work, 'cyl_L2')
            assert run_recipe('thick_cylinder', 2, cyl2).returncode == 0, 'thick_cylinder L2 failed'
            print('  [ok] thick_cylinder L2: %s' % check_thick(cyl2, 2))
            beam2 = os.path.join(work, 'beam_L2')
            assert run_recipe('cantilever', 2, beam2).returncode == 0, 'cantilever L2 failed'
            print('  [ok] cantilever L2: %s' % check_cant(beam2, 2))
            bim2 = os.path.join(work, 'bim_L2')
            assert run_recipe('bimetal_strip', 2, bim2).returncode == 0, 'bimetal L2 failed'
            print('  [ok] bimetal L2 zones: %s' % check_bimetal(bim2))

        print('SELFTEST PASS (%.1f s)' % (time.time() - t_all))
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)
        else:
            print('the scratch dir is kept: %s' % work)
        return 0
    except AssertionError as e:
        print('SELFTEST FAIL: %s' % e)
        print('the scratch dir is kept: %s' % work)
        return 1


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    sys.exit(main())
