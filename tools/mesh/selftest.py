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
    4. --from-checkpoint         the same full run, from the checkpoint of 2, with the post
                                 stage in relative sliver mode (sliver_rel 0.01)

and asserts, for every full run: exit 0; the patches top, west, east, south,
north, wall_ground_land, wall_buildings, roof and pool_yard exist; tets > 0;
no negative volumes in the summary. When rust/target/release/ofgpu-convert-mesh.exe
is built, the mesh is also converted to a case's polyMesh and to an ANSYS
Fluent mesh (-fluent) and both must be written. With the building declared as
a region, the tool runs three more times (dry-run, full, from-checkpoint)
plus one with `interface_names: false`, and four refusals are checked.

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

REGION_PATCHES = ('top', 'west', 'east', 'south', 'north', 'wall_ground_land', 'pool_yard',
                  'fluid_to_building', 'building_outer')
MSH_TRIS = {}   # read_msh_groups side table: {2-D group name: triangle count in the .msh}


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


def write_regions_config(cfg_path, step_path, out_dir, interface_names=True, **overrides):
    """write_config's dict with the building declared as a region (no roof_patches, post gains
    flat_tets: false, sizes gains the per-region field); overrides replace top-level keys.
    Returns cfg."""
    cfg = write_config(cfg_path, step_path, out_dir)
    del cfg['roof_patches']
    cfg['regions'] = {'solids': [{'tag': 2, 'name': 'building', 'kind': 'solid',
                                  'material': 'concrete'}],
                      'interface_names': interface_names}
    cfg['post']['flat_tets'] = False
    cfg['sizes']['regions'] = {'building': {'size': 3.0, 'reach_m': 20.0}}
    cfg.update(overrides)
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


def read_msh_groups(msh):
    """gmsh.initialize(); gmsh.open(msh); returns (names {(dim, tag): name},
    entities {name: [entity tags]}, n_volumes, tets {name: n}, tri_nodes {name: set of node
    tags}, tet_nodes {name: set}); fills MSH_TRIS {name: triangle count}; finalize."""
    import gmsh
    gmsh.initialize()
    try:
        gmsh.open(msh)
        names = {(d, t): gmsh.model.getPhysicalName(d, t)
                 for d, t in gmsh.model.getPhysicalGroups()}
        entities = {nm: list(gmsh.model.getEntitiesForPhysicalGroup(d, t))
                    for (d, t), nm in names.items()}
        n_volumes = len(gmsh.model.getEntities(3))
        tets, tri_nodes, tet_nodes = {}, {}, {}
        MSH_TRIS.clear()
        for (d, t), nm in names.items():
            for ent_t in entities[nm]:
                for et, etg, enodes in zip(*gmsh.model.mesh.getElements(d, ent_t)):
                    if d == 3:
                        tets[nm] = tets.get(nm, 0) + len(etg)
                        tet_nodes.setdefault(nm, set()).update(int(x) for x in enodes)
                    elif et == 2:
                        MSH_TRIS[nm] = MSH_TRIS.get(nm, 0) + len(etg)
                        tri_nodes.setdefault(nm, set()).update(int(x) for x in enodes)
        return names, entities, n_volumes, tets, tri_nodes, tet_nodes
    finally:
        gmsh.finalize()


def check_regions(label, out_dir, msh, interface_names=True):
    """Requirements 2-5 (and 6's interface_names=false variant); returns the summary dict."""
    with open(os.path.join(out_dir, 'selftest_summary.json'), encoding='utf-8') as f:
        s = json.load(f)
    for patch in REGION_PATCHES:
        assert patch in s['groups'], '%s: %s missing from the summary groups (have %s)' % (
            label, patch, sorted(s['groups']))
    assert 'wall_buildings' not in s['groups'], '%s: wall_buildings should not exist' % label
    assert 'roof' not in s['groups'], '%s: roof should not exist' % label
    assert abs(s['regions']['building']['mass_m3'] - 3200.0) <= 1e-9 * 3200.0, \
        '%s: building mass %r' % (label, s['regions']['building']['mass_m3'])
    assert abs(s['fluid_mass_m3'] - 236800.0) <= 1e-9 * 236800.0, \
        '%s: fluid mass %r' % (label, s['fluid_mass_m3'])
    assert abs(s['groups']['fluid_to_building']['area_m2'] - 1040.0) <= 0.05, \
        '%s: interface area %r' % (label, s['groups']['fluid_to_building']['area_m2'])
    assert abs(s['regions']['building']['groups']['building_outer']['area_m2'] - 400.0) <= 0.05, \
        '%s: outer area %r' % (label,
                               s['regions']['building']['groups']['building_outer']['area_m2'])
    names, ents, nvol, tets, tri_nodes, tet_nodes = read_msh_groups(msh)

    assert nvol == 2, '%s: %d volumes in the .msh' % (label, nvol)
    dim3 = {nm for (d, t), nm in names.items() if d == 3}
    assert dim3 == {'fluid', 'building'}, '%s: dim-3 names %s' % (label, sorted(dim3))
    dim2 = {nm for (d, t), nm in names.items() if d == 2}
    assert dim2 >= {'building_outer', 'top', 'west', 'east', 'south', 'north',
                    'wall_ground_land', 'pool_yard'}, '%s: dim-2 names %s' % (label, sorted(dim2))
    assert ('fluid_to_building' in dim2) == interface_names, \
        '%s: fluid_to_building in the .msh: %s' % (label, 'fluid_to_building' in dim2)
    assert tets['building'] == s['regions']['building']['tetrahedra'] > 0, \
        '%s: building tets %r vs summary %r' % (label, tets.get('building'),
                                                s['regions']['building'].get('tetrahedra'))
    assert tets['fluid'] + tets['building'] == s['tetrahedra'], \
        '%s: %d fluid + %d building != %d total' % (label, tets['fluid'], tets['building'],
                                                    s['tetrahedra'])
    assert s['quality']['minSICN']['negative'] == 0, '%s: negative tets' % label
    if interface_names:
        inter = tri_nodes['fluid_to_building']
        assert inter and inter == tet_nodes['fluid'] & tet_nodes['building'], \
            '%s: interface nodes %d, shared tet nodes %d' % (
                label, len(inter), len(tet_nodes['fluid'] & tet_nodes['building']))
        n_tri = MSH_TRIS['fluid_to_building']
    else:
        n_tri = 0
    print('  [%s] %d fluid tets + %d building tets, interface %d tris, minSICN min %.4f' % (
        label, s['fluid_tetrahedra'], s['regions']['building']['tetrahedra'], n_tri,
        s['quality']['minSICN']['min']))
    return s


def check_region_refusals(step_path, out_dir, work):
    """Requirement 7: four configs, each run must exit 1 with the named key in stderr, and
    <out_dir>/selftest.msh must not have been rewritten (mtime before == after)."""
    msh = os.path.join(out_dir, 'selftest.msh')
    cases = (
        ('a roof_patches solid', {'roof_patches': {'roof': 2}}, 'regions.solids[0].tag'),
        ('trim with regions', {'trim': {'below_z': 1.0}}, 'config.trim'),
        ('flat_tets with regions', {'post': {'flat_tets': True}}, 'post.flat_tets'),
        ('a tag outside the STEP', {'regions': {'solids': [{'tag': 7, 'name': 'ghost'}],
                                                'interface_names': True},
                                    'sizes': {'regions': {}}}, 'tag 7'),
    )
    for label, overrides, needle in cases:
        before = os.path.getmtime(msh)
        cfg_path = os.path.join(work, 'selftest_refuse.json')
        write_regions_config(cfg_path, step_path, out_dir, **overrides)
        p = run(cfg_path, out_dir)
        assert p.returncode == 1, '%s: exit %d, expected 1' % (label, p.returncode)
        assert needle in p.stderr, '%s: stderr lacks %r (tail: %s)' % (
            label, needle, p.stderr[-300:])
        assert os.path.getmtime(msh) == before, '%s: selftest.msh was rewritten' % label
        print('  [refusal: %s] exit 1, stderr names "%s"' % (label, needle))


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
        assert s['near_touching'] == [], \
            'the full run recorded near-touching solid pairs: %s' % s['near_touching']
        check_conversion(os.path.join(out_dir, 'selftest.msh'), work)

        # the checkpoint path reruns the post stage, so it carries the relative sliver
        # thresholds; the other three paths keep the absolute ones
        cfg['post']['sliver_rel'] = 0.01
        cfg['post']['sliver_edge_rel'] = 0.25
        with open(cfg_path, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=1)
        p = run(cfg_path, out_dir, ('--from-checkpoint',))
        assert p.returncode == 0, '--from-checkpoint exited %d' % p.returncode
        s2 = check_summary('--from-checkpoint', out_dir, work)
        print('  [--] flat-tet notes (relative sliver mode): %s' % s2['flat_tets_notes'])
        # the brep round-trip perturbs coordinates by ~1e-7, which re-seeds the mesher, so the
        # counts only have to agree to within a few per cent, not exactly
        assert abs(s2['tetrahedra'] - s['tetrahedra']) <= 0.05 * s['tetrahedra'], \
            'the checkpoint run disagrees with the full run (%d vs %d tets)' % (
                s2['tetrahedra'], s['tetrahedra'])

        # the building declared as a region: fragmented with the fluid, conformal interface
        out_r = os.path.join(work, 'out_regions')
        cfg_r = os.path.join(work, 'selftest_regions.json')
        msh_r = os.path.join(out_r, 'selftest.msh')
        write_regions_config(cfg_r, step, out_r)
        p = run(cfg_r, out_r, ('--dry-run',))
        assert p.returncode == 0, 'the regions --dry-run exited %d' % p.returncode
        assert 'volumes        : 2' in p.stdout, \
            'the regions --dry-run did not print two volumes: %s' % p.stdout[-2000:]
        p = run(cfg_r, out_r)
        assert p.returncode == 0, 'the regions full run exited %d' % p.returncode
        s = check_regions('regions full run', out_r, msh_r)
        p = run(cfg_r, out_r, ('--from-checkpoint',))
        assert p.returncode == 0, 'the regions --from-checkpoint exited %d' % p.returncode
        s2 = check_regions('regions --from-checkpoint', out_r, msh_r)
        assert abs(s2['tetrahedra'] - s['tetrahedra']) <= 0.05 * s['tetrahedra'], \
            'the regions checkpoint run disagrees with the full run (%d vs %d tets)' % (
                s2['tetrahedra'], s['tetrahedra'])
        # its own out dir, so out_regions/selftest.msh stays the named-interface mesh M4 reads
        out_n = os.path.join(work, 'out_regions_unnamed')
        cfg_n = os.path.join(work, 'selftest_regions_unnamed.json')
        write_regions_config(cfg_n, step, out_n, interface_names=False)
        p = run(cfg_n, out_n)
        assert p.returncode == 0, 'the interface_names=false run exited %d' % p.returncode
        check_regions('regions interface_names=false', out_n,
                      os.path.join(out_n, 'selftest.msh'), interface_names=False)
        check_region_refusals(step, out_r, work)
        print('  [--] %s is left for the M4 converter (a --keep run preserves it)' % msh_r)

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
