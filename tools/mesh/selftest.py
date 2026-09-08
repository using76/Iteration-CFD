#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""
selftest.py - meshes a tiny STEP end to end through step_mesh.py, in seconds.

Builds its own geometry with gmsh (a 100 x 60 x 40 box minus a 20 x 20 x 8
building standing on the z = 0 floor), writes a config for it (one pool point,
one roof patch), and runs the tool four ways:

    1. --dry-run                 import + cut + ground scan, then stop
    2. --stop-after-checkpoint   import + cut + pools, checkpoint, then stop
    3. the full run              -> .msh, .vtk, _summary.json
    4. --from-checkpoint         the same full run, from the checkpoint of 2

and asserts, for every full run: exit 0; the patches top, west, east, south,
north, wall_ground_land, wall_buildings, roof and pool_yard exist; tets > 0;
no negative volumes in the summary. When rust/target/release/ofgpu-convert-mesh.exe
is built, the mesh is also converted to a case's polyMesh and to an ANSYS
Fluent mesh (-fluent) and both must be written.

    python tools/mesh/selftest.py [--keep]

The scratch directory is printed; --keep preserves it for inspection instead
of deleting it after a pass.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
STEP_MESH = os.path.join(HERE, 'step_mesh.py')
CONVERTER = os.path.join(REPO, 'rust', 'target', 'release', 'ofgpu-convert-mesh.exe')

PATCHES = ('top', 'west', 'east', 'south', 'north',
           'wall_ground_land', 'wall_buildings', 'roof', 'pool_yard')


def build_step(step_path):
    """A 100 x 60 x 40 box (the fluid) minus a 20 x 20 x 8 building on the floor."""
    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        fluid = gmsh.model.occ.addBox(0, 0, 0, 100, 60, 40)      # tag 1
        building = gmsh.model.occ.addBox(40, 20, 0, 20, 20, 8)   # tag 2, on the floor
        gmsh.model.occ.synchronize()
        assert fluid == 1 and building == 2, (fluid, building)
        gmsh.write(step_path)
    finally:
        gmsh.finalize()


def write_config(cfg_path, step_path, out_dir):
    cfg = {
        'step': step_path,
        'scale': 1.0,
        'out_dir': out_dir,
        'name': 'selftest',
        'fluid': {'tag': 1},
        'domain_box': [0.0, 0.0, 0.0, 100.0, 60.0, 40.0],
        'solids': {'sink_m': 0.0, 'fuse': False, 'exclude_tags': []},
        'trim': None,
        'points': {'yard': [20.0, 30.0]},
        'pool_radius_m': 5.0,
        'roof_patches': {'roof': 2},
        'sizes': {'min': 2.0, 'max': 12.0, 'pool': 3.0, 'box': 5.0, 'growth_from': 4.0,
                  'near_struct': 6.0, 'far_struct': 9.0, 'near_radius': 30.0,
                  'size_mult': 1.0, 'roof_boxes': [3.0, 5.0, 7.0]},
        'mesh': {'algo2d': 6, 'algo3d': 1, 'optimize_passes': 1, 'threads': 4},
        'post': {'flat_tets': True, 'flat_threshold': 1e-7, 'seam_merge_m': 0.02,
                 'sliver_edge_m': 0.6, 'sliver_vol_m3': 0.2, 'thin_push_m': 0.0},
        'classification': {'wall_prefix': 'wall_', 'big_roof_is_ground_m2': 2000.0},
    }
    with open(cfg_path, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=1)
    return cfg


def run(cfg_path, out, extra=()):
    t = time.time()
    p = subprocess.run([sys.executable, STEP_MESH, cfg_path, *extra],
                       capture_output=True, text=True, encoding='utf-8', errors='replace')
    took = time.time() - t
    if p.returncode != 0:
        print('--- step_mesh.py stdout ----------------------------------------')
        print(p.stdout[-6000:])
        print('--- step_mesh.py stderr ----------------------------------------')
        print(p.stderr[-2000:])
    print('  [%s] step_mesh.py %s  %.1f s' % (
        'ok' if p.returncode == 0 else 'FAIL', ' '.join(extra) or '(full run)', took))
    return p


def check_summary(label, out_dir, work):
    with open(os.path.join(out_dir, 'selftest_summary.json'), encoding='utf-8') as f:
        s = json.load(f)
    for patch in PATCHES:
        assert patch in s['groups'], '%s: patch %s is missing (have %s)' % (
            label, patch, sorted(s['groups']))
    assert s['tetrahedra'] > 0, '%s: no tetrahedra' % label
    for key in ('quality', 'quality_after_flat_removal'):
        q = s[key]['minSICN']
        assert q['negative'] == 0, '%s: %d negative volumes in %s' % (label, q['negative'], key)
    assert s['pools']['yard'] == 'imprinted', '%s: pool not imprinted: %s' % (label, s['pools'])
    assert s['points']['yard']['z_ground'] == 0.0, '%s: ground under the point: %s' % (
        label, s['points']['yard'])
    print('  [%s] %d tets, %d tris, minSICN min %.4f, patches %s' % (
        label, s['tetrahedra'], s['triangles'], s['quality_after_flat_removal']['minSICN']['min'],
        ', '.join(sorted(s['groups']))))
    return s


def check_conversion(msh, work):
    if not os.path.exists(CONVERTER):
        print('  [--] %s is not built; skipping the conversion check' % CONVERTER)
        return
    case = os.path.join(work, 'case')
    fluent = os.path.join(work, 'selftest_fluent.msh')
    p = subprocess.run([CONVERTER, msh, case, '-fluent', fluent],
                       capture_output=True, text=True, encoding='utf-8', errors='replace')
    assert p.returncode == 0, 'convert-mesh failed:\n%s\n%s' % (p.stdout[-2000:], p.stderr[-2000:])
    assert os.path.isfile(fluent) and os.path.getsize(fluent) > 0, 'no Fluent mesh written'
    assert os.path.isfile(os.path.join(case, 'constant', 'polyMesh', 'faces')), \
        'no polyMesh written'
    print('  [ok] ofgpu-convert-mesh: polyMesh + %s (%.1f MB)'
          % (os.path.basename(fluent), os.path.getsize(fluent) / 1e6))


def main(argv=None):
    ap = argparse.ArgumentParser(description='Mesh a tiny STEP through step_mesh.py, in seconds.')
    ap.add_argument('--keep', action='store_true', help='keep the scratch directory')
    args = ap.parse_args(argv)

    work = tempfile.mkdtemp(prefix='step_mesh_selftest_')
    step = os.path.join(work, 'selftest.step')
    out_dir = os.path.join(work, 'out')
    cfg_path = os.path.join(work, 'selftest.json')
    print('selftest scratch dir: %s' % work)
    try:
        build_step(step)
        cfg = write_config(cfg_path, step, out_dir)
        print('built %s (fluid = 100x60x40 box, building = 20x20x8 on the floor)' % step)

        p = run(cfg_path, out_dir, ('--dry-run',))
        assert p.returncode == 0, '--dry-run exited %d' % p.returncode
        assert 'DRY RUN' in p.stdout and 'surface count' in p.stdout, \
            '--dry-run did not print the volumes, masses, surface counts and ground heights'
        assert 'z_g = 0.0' in p.stdout, '--dry-run did not print the ground heights'

        p = run(cfg_path, out_dir, ('--stop-after-checkpoint',))
        assert p.returncode == 0, '--stop-after-checkpoint exited %d' % p.returncode
        assert os.path.isfile(os.path.join(out_dir, 'work', 'selftest_pools.brep')) and \
            os.path.isfile(os.path.join(out_dir, 'work', 'selftest_pools.json')), \
            '--stop-after-checkpoint wrote no checkpoint'

        p = run(cfg_path, out_dir)
        assert p.returncode == 0, 'the full run exited %d' % p.returncode
        s = check_summary('full run', out_dir, work)
        check_conversion(os.path.join(out_dir, 'selftest.msh'), work)

        p = run(cfg_path, out_dir, ('--from-checkpoint',))
        assert p.returncode == 0, '--from-checkpoint exited %d' % p.returncode
        s2 = check_summary('--from-checkpoint', out_dir, work)
        # the brep round-trip perturbs coordinates by ~1e-7, which re-seeds the mesher, so the
        # counts only have to agree to within a few per cent, not exactly
        assert abs(s2['tetrahedra'] - s['tetrahedra']) <= 0.05 * s['tetrahedra'], \
            'the checkpoint run disagrees with the full run (%d vs %d tets)' % (
                s2['tetrahedra'], s['tetrahedra'])

        print('SELFTEST PASS')
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)
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
