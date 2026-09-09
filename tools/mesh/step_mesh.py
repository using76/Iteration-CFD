#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""
step_mesh.py - a configuration-driven STEP -> Gmsh tetrahedral mesh tool.

    python tools/mesh/step_mesh.py <config.json> [--from-checkpoint]
          [--stop-after-checkpoint] [--tag NAME] [--dry-run]

One JSON config describes the geometry (which solid is the fluid, which solids
are cut away, which get replaced by a repaired proxy), the sizing (pool,
refinement boxes, near/far structures) and the post-processing (flat-tet
removal). One run writes

    <out_dir>/<name>[_TAG].msh            Gmsh 4.1 ASCII, physical groups = patches
    <out_dir>/<name>[_TAG].vtk            binary, for viewing only
    <out_dir>/<name>[_TAG]_summary.json   counts, groups, quality, timings, the config
    <out_dir>/work/                       checkpoints, repaired solids, run.log

and `ofgpu-convert-mesh <out_dir>/<name>.msh <caseDir> -fluent <name>_fluent.msh`
takes the mesh to the solver and to Fluent.

The pipeline, its order and its checkpoints are ported stage for stage from the
ammonia site mesh (nh3_site_mesh.py, proven on a 3.65 M-cell mesh), with the
ship repair route E of work/probe_repair.py and the trim cut of
work/probe_trim.py. Every number that script hard-coded is a config key here,
with the same default; tools/mesh/examples/nh3_site.json is that site as a
config, and tools/mesh/selftest.py builds its own tiny STEP and meshes it end
to end in seconds.

Known limits (see tools/mesh/README.md): 3-D Delaunay only - HXT hits an
unfinished Steiner-point path on this class of geometry and kills the process,
and Netgen's optimiser dies with an access violation on this gmsh build.
"""
from __future__ import annotations

import argparse
import difflib
import json
import math
import os
import sys
import time

import gmsh
import numpy as np
try:
    from scipy.spatial import cKDTree      # only the near-touching scan of the cut stage needs it
except ImportError:                        # the scan is a diagnostic: without scipy it is skipped
    cKDTree = None

# ---------------------------------------------------------------- constants
# Numbers the reference script hard-codes and the config schema does not name:
# the field layout around the points and roof patches, the scan of the ground,
# the classification tolerances. Everything the ground rules name by value
# (sizes, radii, thresholds, the seam merge) is a config key instead.
GROUND_SCAN = (-1.0, 0.25, 60.0)   # first z, step, last z of the isInside scan
FLAT_TOL = 0.05                    # a face whose bbox is this thin is "flat"
SEA_TOL = 0.06                     # tolerance of the sea_z match
POOL_Z_TOL = 0.1                   # a pool piece must sit this near its z_g
POOL_PAD = 0.5                     # slack of the pool bbox / centre-radius tests
ROOF_PAD = 0.3                     # slack of the "face belongs to solid X" test
ROOF_Z_TOL = 0.1                   # a roof face sits this near its solid's top
HULL_PAD = 0.5                     # slack of the ship-hull bbox test
BIG_ROOF_XY = 100.0                # a slab longer than this, both ways, is terrain
POINT_BOX = (40.0, 1.0, 10.0, 30.0)     # +-x/y extent, z below, z above, Thickness
POINT_BOX_FAR = (150.0, 1.0, 30.0, 80.0)
GROWTH = (40.0, 700.0)             # DistMin, DistMax of the from-source growth
NEAR_FAR = (80.0, 120.0)           # DistMax of the near / far structure fields
ROOF_BOXES = (                     # per roof box: +-x/y extent, z below, z above, Thickness
    (60.0, 2.0, 40.0, 60.0),
    (250.0, 2.0, 80.0, 120.0),
)
ROOF_DOWNWIND = (700.0, 60.0, 250.0, 2.0, 120.0, 150.0)   # -x reach, +x, +-y, z below/above, Thickness
NEAR_FAR_SAMPLING = (30, 20)       # Distance-field sampling of the near / far fields
REPAIR_SAMPLE_SIZE = 3.0           # surface mesh size the resample route starts from
REPAIR_LADDER_TAIL = (10000, 16000)  # fallback face counts, coarsest first
REPAIR_MIN_COMPONENT = 200         # crumbs smaller than this come from the resampler
EXPECT_SLACK = 0.02                # the cut may miss the expected mass by this much
TRIM_MARGIN = (50.0, 10.0)         # xy margin, z depth below the model of the trim box
PRE3D_MERGE = 0.0                  # OFF: merging seams BEFORE the 3-D pass makes gmsh
                                   # drop every tet (nh3_site_mesh.py, 2026-09-08); the
                                   # seams are merged after it, in the flat-tet stage
THIN_SICN = 0.02                   # a thin tet for the push stage: gamma below this
SLIVER_ROUNDS = 8
HULL_FLAT_M = 1.0                  # a hull corner less than this off its neighbours' line is dropped
SLIVER_TRI = 0.05                  # a surface triangle whose height is below this x its longest edge
PUSH_ROUNDS = 3

DEFAULTS = {
    'scale': 0.001,                 # Geometry.OCCScaling, applied BEFORE the import (mm -> m)
    'name': 'site',
    'fluid': {'tag': 1, 'largest': False},
    'outer_tol': 0.05,              # tolerance of the top/west/east/south/north tests
    'solids': {'sink_m': 2.0, 'fuse': False, 'exclude_tags': [], 'touch_warn_m': 0.05,
               'hull_beyond_m': 0.0, 'hull_pad_m': 1.0, 'hull_snap_m': 0.05,
               'boolean_tol_m': 0.0},
    'repairs': [],                  # [{'tag', 'method', 'cell_m', 'target_faces', 'lift_z', 'brep'}]
    'trim': {'below_z': 3.05},      # or null: no trim
    'sea_z': 3.05,
    'points': {},                   # {'tank_shell': [x, y], ...}
    'pool_radius_m': 26.0,
    'roof_patches': {},             # {'nh3_source': 306}
    'sizes': {'min': 1.5, 'max': 40.0, 'pool': 2.0, 'box': 4.0, 'growth_from': 2.5,
              'near_struct': 4.0, 'far_struct': 12.0, 'near_radius': 400.0,
              'size_mult': 1.0, 'roof_boxes': [2.5, 5.0, 10.0], 'gap_ratio': 0.0,
              'gap_min_m': 0.0},
    'mesh': {'algo2d': 6, 'algo3d': 1, 'optimize_passes': 5, 'threads': 32},
    'post': {'flat_tets': True, 'flat_threshold': 1e-7, 'seam_merge_m': 0.02,
             'sliver_edge_m': 0.6, 'sliver_vol_m3': 0.2, 'thin_push_m': 0.0,
             'sliver_rel': 0.0, 'sliver_edge_rel': 0.25, 'min_thickness': 0.0,
             'repair_rounds': 3},
    'classification': {'wall_prefix': 'wall_', 'big_roof_is_ground_m2': 2000.0},
}
REQUIRED = ('step', 'out_dir', 'domain_box')
REPAIR_KEYS = {'tag', 'method', 'cell_m', 'target_faces', 'lift_z', 'brep'}


# ------------------------------------------------------------------- output
T0 = time.time()
SUMMARY = {}


def _install_tee() -> None:
    """run_step_mesh.cmd sets STEP_MESH_LOG: every line also goes to that file."""
    path = os.environ.get('STEP_MESH_LOG')
    if not path:
        return

    class Tee:
        def __init__(self, stream):
            self.stream = stream
            self.file = open(path, 'w', encoding='utf-8')

        def write(self, s):
            try:
                self.stream.write(s)
            except Exception:
                pass
            self.file.write(s)

        def flush(self):
            try:
                self.stream.flush()
            except Exception:
                pass
            self.file.flush()

    sys.stdout = Tee(sys.stdout)
    sys.stderr = Tee(sys.stderr)


def stage(n, name):
    print('')
    print('========== [%2d/10] %-18s  (elapsed %6.0f s) ==========' % (n, name, time.time() - T0), flush=True)


def log(msg):
    print('%7.1f s  %s' % (time.time() - T0, msg), flush=True)


def tick(name, t):
    SUMMARY['timings_s'][name] = round(time.time() - t, 1)


def die(msg, code=1):
    sys.stderr.write('step_mesh: %s\n' % msg)
    sys.stderr.flush()
    raise SystemExit(code)


# ------------------------------------------------------------------- config
def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _nums(v, n, what, errors):
    if not (isinstance(v, list) and len(v) == n and all(_is_num(x) for x in v)):
        errors.append('%s: expected %d numbers, got %r' % (what, n, v))
        return False
    return True


def _merge(user, defaults, path, errors, extra_ok=()):
    """User values over defaults; every key the schema does not name is an error.
    The required keys carry no default but are known all the same (extra_ok)."""
    if not isinstance(user, dict):
        errors.append('%s: expected a JSON object, got %r' % (path, user))
        return dict(defaults)
    out = {}
    for k, dv in defaults.items():
        if k not in user:
            out[k] = dv
        elif isinstance(dv, dict) and dv == {}:
            out[k] = user[k]          # an open mapping: points, roof_patches - any names allowed
        elif isinstance(dv, dict) and user[k] is not None:
            out[k] = _merge(user[k], dv, '%s.%s' % (path, k), errors)
        else:
            out[k] = user[k]
    for k in extra_ok:
        if k in user:
            out[k] = user[k]
    for k in user:
        if k not in defaults and k not in extra_ok:
            near = difflib.get_close_matches(k, list(defaults) + list(extra_ok), 1)
            errors.append('unknown key "%s.%s"%s' % (
                path, k, ' - did you mean "%s"?' % near[0] if near else ''))
    return out


def load_config(path):
    try:
        with open(path, encoding='utf-8-sig') as f:
            user = json.load(f)
    except OSError as e:
        die('cannot read the config %s: %s' % (path, e))
    except json.JSONDecodeError as e:
        die('the config %s is not valid JSON: %s' % (path, e))
    if not isinstance(user, dict):
        die('the config must be a JSON object')
    errors = []
    cfg = _merge(user, DEFAULTS, 'config', errors, extra_ok=REQUIRED)

    for key in REQUIRED:
        if cfg.get(key) in (None, '', []):
            errors.append('missing required key "%s"' % key)
    if 'step' in cfg and not isinstance(cfg['step'], str):
        errors.append('config.step: expected a path string')
    if not isinstance(cfg['name'], str) or not cfg['name']:
        errors.append('config.name: expected a non-empty string')
    if cfg.get('domain_box') is not None:
        _nums(cfg['domain_box'], 6, 'config.domain_box', errors)
    if _is_num(cfg['outer_tol']) and cfg['outer_tol'] <= 0:
        errors.append('config.outer_tol: expected a positive tolerance')
    if not isinstance(cfg['fluid'], dict):
        errors.append('config.fluid: expected {"tag": n} or {"largest": true}')
    else:
        if not isinstance(cfg['fluid']['tag'], int) or isinstance(cfg['fluid']['tag'], bool):
            errors.append('config.fluid.tag: expected an integer solid tag')
        if not isinstance(cfg['fluid']['largest'], bool):
            errors.append('config.fluid.largest: expected true or false')

    sol = cfg['solids']
    if not _is_num(sol['sink_m']) or sol['sink_m'] < 0:
        errors.append('config.solids.sink_m: expected a non-negative number of metres')
    if not isinstance(sol['fuse'], bool):
        errors.append('config.solids.fuse: expected true or false')
    if not (isinstance(sol['exclude_tags'], list) and
            all(isinstance(t, int) and not isinstance(t, bool) for t in sol['exclude_tags'])):
        errors.append('config.solids.exclude_tags: expected a list of integer solid tags')
    if not _is_num(sol['hull_beyond_m']) or sol['hull_beyond_m'] < 0:
        errors.append('config.solids.hull_beyond_m: expected a distance >= 0 (0 disables)')
    if not _is_num(sol['hull_pad_m']) or sol['hull_pad_m'] < 0:
        errors.append('config.solids.hull_pad_m: expected a non-negative pad in metres')
    if not _is_num(sol['hull_snap_m']) or sol['hull_snap_m'] < 0:
        errors.append('config.solids.hull_snap_m: expected a snapping distance >= 0')
    if not _is_num(sol['boolean_tol_m']) or sol['boolean_tol_m'] < 0:
        errors.append('config.solids.boolean_tol_m: expected a fuzzy boolean tolerance >= 0 (0 = exact)')
    if not _is_num(sol['touch_warn_m']) or sol['touch_warn_m'] < 0:
        errors.append('config.solids.touch_warn_m: expected a non-negative distance (0 disables)')

    rep_tags = set()
    for i, r in enumerate(cfg['repairs']):
        if not isinstance(r, dict):
            errors.append('config.repairs[%d]: expected an object' % i)
            continue
        for k in r:
            if k not in REPAIR_KEYS:
                errors.append('unknown key "config.repairs[%d].%s"' % (i, k))
        if not isinstance(r.get('tag'), int) or isinstance(r.get('tag'), bool):
            errors.append('config.repairs[%d].tag: expected an integer solid tag' % i)
        else:
            if r['tag'] in rep_tags:
                errors.append('config.repairs[%d].tag: solid %d is repaired twice' % (i, r['tag']))
            rep_tags.add(r['tag'])
            if r['tag'] in sol['exclude_tags']:
                errors.append('config.repairs[%d].tag: solid %d is both repaired and excluded '
                              'from the cut - pick one' % (i, r['tag']))
        if r.get('method') != 'resample':
            errors.append('config.repairs[%d].method: only "resample" is implemented '
                          '(probe_repair.py route E)' % i)
        if not _is_num(r.get('cell_m')) or r['cell_m'] <= 0:
            errors.append('config.repairs[%d].cell_m: expected a positive cell size' % i)
        if not isinstance(r.get('target_faces'), int) or isinstance(r.get('target_faces'), bool) \
                or r['target_faces'] < 200:
            errors.append('config.repairs[%d].target_faces: expected a face count >= 200' % i)
        if not _is_num(r.get('lift_z')):
            errors.append('config.repairs[%d].lift_z: expected a number of metres (0 to keep it)' % i)
        if 'brep' in r and not isinstance(r['brep'], str):
            errors.append('config.repairs[%d].brep: expected a path string' % i)
    if 'trim' in user and cfg['trim'] is not None:
        if not isinstance(cfg['trim'], dict) or not _is_num(cfg['trim'].get('below_z')):
            errors.append('config.trim: expected {"below_z": metres} or null')
    if not _is_num(cfg['sea_z']):
        errors.append('config.sea_z: expected a height in metres')
    if not isinstance(cfg['points'], dict):
        errors.append('config.points: expected {"name": [x, y] or {"x", "y", "r", "r_inner"}, ...}')
    else:
        # a point is [x, y] (radius = pool_radius_m) or {x, y, r, r_inner}: r is the disc's
        # radius; r_inner > 0 makes it an annulus (the ring between r_inner and r), so one
        # centre can carry an inner disc and an outer ring as two separately named patches
        cfg['pool_specs'] = {}
        for pname, spec in list(cfg['points'].items()):
            if isinstance(spec, dict):
                extra = set(spec) - {'x', 'y', 'r', 'r_inner', 'h'}
                if extra:
                    errors.append('config.points.%s: unknown keys %s' % (pname, sorted(extra)))
                if not (_is_num(spec.get('x')) and _is_num(spec.get('y'))):
                    errors.append('config.points.%s: expected numbers x and y' % pname)
                    continue
                r = spec.get('r', cfg['pool_radius_m'])
                ri = spec.get('r_inner', 0.0)
                if not _is_num(r) or r <= 0:
                    errors.append('config.points.%s.r: expected a positive radius' % pname)
                    continue
                if not _is_num(ri) or ri < 0 or ri >= r:
                    errors.append('config.points.%s.r_inner: expected 0 <= r_inner < r' % pname)
                    continue
                h = spec.get('h', 0.0)
                if not _is_num(h) or h < 0:
                    errors.append('config.points.%s.h: expected a height >= 0 (0 = flat disc)' % pname)
                    continue
                cfg['pool_specs'][pname] = {'x': float(spec['x']), 'y': float(spec['y']),
                                            'r': float(r), 'r_inner': float(ri), 'h': float(h)}
                cfg['points'][pname] = [float(spec['x']), float(spec['y'])]
            else:
                _nums(spec, 2, 'config.points.%s' % pname, errors)
                ok = (isinstance(spec, list) and len(spec) == 2 and all(_is_num(v) for v in spec)
                      and _is_num(cfg['pool_radius_m']))
                if ok:
                    cfg['pool_specs'][pname] = {'x': float(spec[0]), 'y': float(spec[1]),
                                                'r': float(cfg['pool_radius_m']), 'r_inner': 0.0,
                                                'h': 0.0}
    if not _is_num(cfg['pool_radius_m']) or cfg['pool_radius_m'] <= 0:
        errors.append('config.pool_radius_m: expected a positive radius')
    if not isinstance(cfg['roof_patches'], dict):
        errors.append('config.roof_patches: expected {"name": solid_tag, ...}')
    else:
        for pname, tag in cfg['roof_patches'].items():
            if not isinstance(tag, int) or isinstance(tag, bool):
                errors.append('config.roof_patches.%s: expected an integer solid tag' % pname)
            if pname in cfg['points']:
                errors.append('config.roof_patches.%s: a pool and a roof patch share this name' % pname)

    s = cfg['sizes']
    for k in ('min', 'max', 'pool', 'box', 'growth_from', 'near_struct', 'far_struct',
              'near_radius', 'size_mult'):
        if not _is_num(s[k]) or s[k] <= 0:
            errors.append('config.sizes.%s: expected a positive number' % k)
    if not _is_num(cfg['sizes']['gap_ratio']) or cfg['sizes']['gap_ratio'] < 0:
        errors.append('config.sizes.gap_ratio: expected a ratio >= 0 (0 disables the gap pass)')
    if not _is_num(cfg['sizes']['gap_min_m']) or cfg['sizes']['gap_min_m'] < 0:
        errors.append('config.sizes.gap_min_m: expected a size >= 0 (0 = sizes.min)')
    if _is_num(s['min']) and _is_num(s['max']) and s['min'] >= s['max']:
        errors.append('config.sizes: min %s must be below max %s' % (s['min'], s['max']))
    if not (isinstance(s['roof_boxes'], list) and len(s['roof_boxes']) == 3
            and all(_is_num(v) and v > 0 for v in s['roof_boxes'])):
        errors.append('config.sizes.roof_boxes: expected three positive sizes [near, mid, far]')

    m = cfg['mesh']
    if not isinstance(m['algo2d'], int) or isinstance(m['algo2d'], bool) or not 1 <= m['algo2d'] <= 9:
        errors.append('config.mesh.algo2d: expected a gmsh 2-D algorithm number (1..9)')
    if not isinstance(m['algo3d'], int) or isinstance(m['algo3d'], bool):
        errors.append('config.mesh.algo3d: expected a gmsh 3-D algorithm number (1 = Delaunay)')
    if not isinstance(m['optimize_passes'], int) or isinstance(m['optimize_passes'], bool) \
            or m['optimize_passes'] < 0:
        errors.append('config.mesh.optimize_passes: expected a pass count >= 0')
    if not isinstance(m['threads'], int) or isinstance(m['threads'], bool) or m['threads'] < 1:
        errors.append('config.mesh.threads: expected a thread count >= 1')

    p = cfg['post']
    if not isinstance(p['flat_tets'], bool):
        errors.append('config.post.flat_tets: expected true or false')
    if not _is_num(p['flat_threshold']) or p['flat_threshold'] <= 0:
        errors.append('config.post.flat_threshold: expected a positive volume fraction')
    for k in ('seam_merge_m', 'sliver_edge_m', 'sliver_vol_m3', 'thin_push_m'):
        if not _is_num(p[k]) or p[k] < 0:
            errors.append('config.post.%s: expected a non-negative length/volume' % k)
    for k in ('sliver_rel', 'sliver_edge_rel'):
        if not _is_num(p[k]) or p[k] < 0:
            errors.append('config.post.%s: expected a non-negative ratio' % k)
    if not _is_num(p['min_thickness']) or p['min_thickness'] < 0:
        errors.append('config.post.min_thickness: expected a thickness ratio >= 0 (0 = no gate)')
    if not isinstance(p['repair_rounds'], int) or isinstance(p['repair_rounds'], bool) or p['repair_rounds'] < 0:
        errors.append('config.post.repair_rounds: expected a round count >= 0')

    c = cfg['classification']
    if not isinstance(c['wall_prefix'], str):
        errors.append('config.classification.wall_prefix: expected a string')
    if not _is_num(c['big_roof_is_ground_m2']) or c['big_roof_is_ground_m2'] <= 0:
        errors.append('config.classification.big_roof_is_ground_m2: expected an area in m^2')

    if errors:
        die('the config %s is rejected:\n  - %s' % (path, '\n  - '.join(errors)))
    cfg['step'] = os.path.abspath(os.path.expanduser(cfg['step']))
    cfg['out_dir'] = os.path.abspath(os.path.expanduser(cfg['out_dir']))
    return cfg


# ------------------------------------------------------------ stage 1: import
def import_stage(cfg, args, work):
    """The STEP in, the fluid solid found, its mass reported (or the checkpoint in)."""
    stage(1, 'import')
    t = time.time()
    gmsh.option.setNumber('General.Terminal', 1)
    gmsh.option.setNumber('General.NumThreads', cfg['mesh']['threads'])
    gmsh.model.add(cfg['name'])

    if args.from_checkpoint:
        ckpt_brep = os.path.join(work, '%s_pools.brep' % cfg['name'])
        ckpt_json = os.path.join(work, '%s_pools.json' % cfg['name'])
        if not (os.path.exists(ckpt_brep) and os.path.exists(ckpt_json)):
            die('--from-checkpoint but the checkpoint is missing (%s[.json]): '
                'run once without it' % ckpt_brep)
        gmsh.option.setNumber('Geometry.OCCScaling', 1.0)   # the checkpoint is in metres already
        gmsh.model.occ.importShapes(ckpt_brep)
        gmsh.model.occ.synchronize()
        vols = gmsh.model.getEntities(3)
        if len(vols) != 1:
            die('the checkpoint %s holds %d volumes, expected 1' % (ckpt_brep, len(vols)))
        fluid = vols[0][1]
        with open(ckpt_json, encoding='utf-8') as f:
            ck = json.load(f)
        for k in ('points', 'pools', 'solid_bboxes', 'ship_hulls', 'fluid_mass_m3', 'pockets'):
            SUMMARY[k] = ck[k]
        SUMMARY['near_touching'] = ck.get('near_touching', [])   # checkpoints predating U46 lack it
        SUMMARY['solid_bboxes'] = {int(tg): bb for tg, bb in SUMMARY['solid_bboxes'].items()}
        if set(cfg['points']) != set(SUMMARY['points']):
            die("the config's points %s and the checkpoint's points %s differ; the ground "
                "heights and the pools belong to the checkpoint - rerun without "
                '--from-checkpoint' % (sorted(cfg['points']), sorted(SUMMARY['points'])))
        SUMMARY['from_checkpoint'] = True
        # faces that survived a failed pool imprint (a disc embedded in a slope) are not part
        # of the boundary; left in the model they get meshed and confuse the 3-D pass
        bset = set(sf for d, sf in gmsh.model.getBoundary([(3, fluid)], oriented=False))
        loose = [(2, sf) for d, sf in gmsh.model.getEntities(2) if sf not in bset]
        if loose:
            gmsh.model.occ.remove(loose, recursive=True)
            gmsh.model.occ.synchronize()
            log('removed %d loose surfaces that do not bound the fluid' % len(loose))
        log('checkpoint %s: fluid tag %d, mass %.4e m^3, %d boundary surfaces '
            '(cut + pools %s)' % (os.path.basename(ckpt_brep), fluid,
                                  gmsh.model.occ.getMass(3, fluid),
                                  len(gmsh.model.getBoundary([(3, fluid)], oriented=False)),
                                  'loaded' if ck['pools'] else 'skipped'))
        SUMMARY['fluid'] = fluid
        tick('import', t)
        return

    if not os.path.exists(cfg['step']):
        die('the STEP file does not exist: %s' % cfg['step'])
    gmsh.option.setNumber('Geometry.OCCScaling', cfg['scale'])   # mm -> m, before the import
    gmsh.model.occ.importShapes(cfg['step'])
    gmsh.model.occ.synchronize()
    vols = gmsh.model.getEntities(3)
    if not vols:
        die('the STEP %s contains no solid' % cfg['step'])
    tags = [tg for d, tg in vols]
    if cfg['fluid']['largest']:
        fluid = max(vols, key=lambda dt: gmsh.model.occ.getMass(3, dt[1]))[1]
        log('fluid = the largest of the %d imported solids (tag %d)' % (len(vols), fluid))
    else:
        fluid = cfg['fluid']['tag']
        if fluid not in tags:
            die('the fluid tag %d is not among the imported solids %s%s'
                % (fluid, tags[:20], ' ... (%d total)' % len(tags) if len(tags) > 20 else ''))
    mass = gmsh.model.occ.getMass(3, fluid)
    log('imported at scale %g: %d volumes, fluid tag %d, mass %.4e m^3'
        % (cfg['scale'], len(vols), fluid, mass))
    SUMMARY['fluid'] = fluid
    SUMMARY['fluid_mass_m3_imported'] = mass
    SUMMARY['solids_imported'] = len(vols) - 1
    # every solid's bbox drives the classification; healing may renumber but a bbox does not move
    SUMMARY['solid_bboxes'] = {tg: list(gmsh.model.getBoundingBox(3, tg)) for tg in tags if tg != fluid}
    tick('import', t)


# ------------------------------------------------------- stage 2: repairs + cut
def repair_one(cfg, entry, work):
    """Replace one self-intersecting solid by a faceted proxy: the probe_repair.py E route.

    2-D mesh at 3 m -> pymeshlab uniform resampling on a cell_m grid -> quadric
    decimation (coarsest of the target ladder that is watertight, manifold and
    free of self-intersections) -> an OCC solid built of the planar triangles.
    The result is cached as a .brep in out_dir/work; the original STEP's scale
    applies to the sampling import, the cached brep is already in metres.
    """
    try:
        import pymeshlab
    except ImportError as e:
        die('a repair needs pymeshlab and it does not import: %s' % e)

    tag = entry['tag']
    brep = entry.get('brep') or os.path.join(work, 'repaired_%d.brep' % tag)
    if os.path.exists(brep):
        log('repair tag %d: reusing %s' % (tag, brep))
        return brep

    repair_model = 'repair_%d' % tag
    main_model = cfg['name']
    gmsh.model.add(repair_model)
    gmsh.model.setCurrent(repair_model)
    try:
        gmsh.option.setNumber('Geometry.OCCScaling', cfg['scale'])
        gmsh.model.occ.importShapes(cfg['step'])
        gmsh.model.occ.synchronize()
        vols = gmsh.model.getEntities(3)
        if tag not in [tg for d, tg in vols]:
            die('the repair tag %d is not among the solids of %s' % (tag, cfg['step']))
        others = [(3, tg) for d, tg in vols if tg != tag]
        if others:
            gmsh.model.occ.remove(others, recursive=True)
            gmsh.model.occ.synchronize()
        for k, v in (('Mesh.MeshSizeMin', REPAIR_SAMPLE_SIZE / 3.0),
                     ('Mesh.MeshSizeMax', REPAIR_SAMPLE_SIZE),
                     ('Mesh.MeshSizeFromCurvature', 0),
                     ('Mesh.MeshSizeExtendFromBoundary', 0),
                     ('Mesh.MeshSizeFromPoints', 0),
                     ('Mesh.Algorithm', 6)):
            gmsh.option.setNumber(k, v)
        gmsh.model.mesh.generate(2)
        ntags, coords, _ = gmsh.model.mesh.getNodes()
        V = coords.reshape(-1, 3)
        idx = {int(t): i for i, t in enumerate(ntags)}
        F = []
        for d, sf in gmsh.model.getEntities(2):
            et, _, enodes = gmsh.model.mesh.getElements(2, sf)
            for ty, nodes in zip(et, enodes):
                if ty == 2:
                    F.append(np.array([idx[int(v)] for v in nodes]).reshape(-1, 3))
        if not F:
            die('repair tag %d: its surface produced no triangles' % tag)
        F = np.vstack(F)
        gmsh.model.mesh.clear()
    finally:
        gmsh.model.setCurrent(main_model)
        gmsh.model.setCurrent(repair_model)
        gmsh.model.remove()
        gmsh.model.setCurrent(main_model)

    def status(label):
        m = ms.current_mesh()
        E = np.sort(np.vstack([m.face_matrix()[:, [0, 1]], m.face_matrix()[:, [1, 2]],
                               m.face_matrix()[:, [2, 0]]]), axis=1)
        _, cnt = np.unique(E, axis=0, return_counts=True)
        ms.compute_selection_by_self_intersections_per_face()
        nbad = int(ms.current_mesh().face_selection_array().sum())
        log('repair tag %d: %s: %d faces, %d boundary edges, %d non-manifold edges, '
            '%d self-intersecting' % (tag, label, m.face_number(), int((cnt == 1).sum()),
                                      int((cnt > 2).sum()), nbad))
        return int((cnt == 1).sum()), int((cnt > 2).sum()), nbad

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(V, F))
    # global repair: resample the surface on a cell_m distance-field grid (marching cubes),
    # which yields a closed manifold surface whatever the input soup looks like
    ms.generate_resampled_uniform_mesh(cellsize=pymeshlab.PureValue(entry['cell_m']),
                                       offset=pymeshlab.PureValue(0.0), mergeclosevert=True,
                                       discretize=False, multisample=True, absdist=False)
    ms.meshing_remove_connected_component_by_face_number(mincomponentsize=REPAIR_MIN_COMPONENT)
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_remove_unreferenced_vertices()
    status('resampled at %g m' % entry['cell_m'])
    base = ms.current_mesh_id()
    ladder = [entry['target_faces']] + [t for t in REPAIR_LADDER_TAIL if t > entry['target_faces']]
    b1 = nm = nbad = 1
    for target in ladder:                        # the coarsest clean decimation wins
        ms.set_current_mesh(base)
        ms.add_mesh(ms.current_mesh(), 'dec%d' % target)
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target, qualitythr=0.3,
                                                    preservenormal=True, preservetopology=True,
                                                    planarquadric=True, autoclean=True)
        b1, nm, nbad = status('decimated to %d faces' % target)
        if not (b1 or nm or nbad):
            break
    if b1 or nm or nbad:
        die('repair tag %d: no decimation of the ladder %s is watertight, manifold and '
            'free of self-intersections' % (tag, ladder))
    Vd, Fd = ms.current_mesh().vertex_matrix(), ms.current_mesh().face_matrix()

    # OCC solid from the planar triangles, built in a scratch model and written as the cache
    gmsh.model.add(repair_model)
    gmsh.model.setCurrent(repair_model)
    try:
        occ = gmsh.model.occ
        pts = [occ.addPoint(*p) for p in Vd]
        edges = {}

        def line(a, b):
            key = (a, b) if a < b else (b, a)
            if key not in edges:
                edges[key] = occ.addLine(pts[key[0]], pts[key[1]])
            return edges[key] if a < b else -edges[key]

        surfs = []
        for a, b, c in Fd:
            cl = occ.addCurveLoop([line(a, b), line(b, c), line(c, a)])
            surfs.append(occ.addPlaneSurface([cl]))
        sl = occ.addSurfaceLoop(surfs)
        vol = occ.addVolume([sl])
        occ.synchronize()
        mass = occ.getMass(3, vol)
        log('repair tag %d: OCC solid from %d triangles: mass %.0f m^3' % (tag, len(Fd), mass))
        gmsh.write(brep)
    finally:
        gmsh.model.setCurrent(main_model)
        gmsh.model.setCurrent(repair_model)
        gmsh.model.remove()
        gmsh.model.setCurrent(main_model)
    return brep


def hull_corners(tag, bbox, pad):
    """The footprint of solid `tag` as its convex hull pushed out by `pad` (ccw corners), with
    the solid's z range; None when the outline is degenerate (fewer than three hull corners)."""
    verts = gmsh.model.getBoundary([(3, tag)], combined=False, oriented=False, recursive=True)
    xy = np.array([gmsh.model.getValue(0, p, [])[:2] for d, p in verts if d == 0])
    if len(xy) < 3:
        return None
    # convex hull of the footprint (monotone chain), counter-clockwise
    pts = sorted(set(map(tuple, np.round(xy, 6))))
    if len(pts) < 3:
        return None

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def turns_left(o, a, b):
        # a corner survives only if it stands more than HULL_FLAT_M off the line o-b: a wall
        # whose vertices are millimetres out of line would otherwise leave hull edges of
        # millimetres, and the fuse of neighbouring prisms turns those into sliver faces
        c = cross(o, a, b)
        base = math.hypot(b[0] - o[0], b[1] - o[1])
        return c > 0 and (base < 1e-9 or c / base > HULL_FLAT_M)
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and not turns_left(lower[-2], lower[-1], p):
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and not turns_left(upper[-2], upper[-1], p):
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return None
    # push every edge outwards by pad and intersect consecutive edge lines
    n = len(hull)
    lines = []
    for i in range(n):
        (x0, y0), (x1, y1) = hull[i], hull[(i + 1) % n]
        ex, ey = x1 - x0, y1 - y0
        L = math.hypot(ex, ey)
        if L < 1e-9:
            continue
        nx, ny = ey / L, -ex / L                      # outward normal of a ccw polygon
        lines.append(((x0 + pad * nx, y0 + pad * ny), (ex / L, ey / L)))
    if len(lines) < 3:
        return None
    corners = []
    for i in range(len(lines)):
        (p0, d0), (p1, d1) = lines[i - 1], lines[i]
        den = d0[0] * d1[1] - d0[1] * d1[0]
        if abs(den) < 1e-12:                          # parallel consecutive edges: keep the joint
            corners.append((p1[0], p1[1]))
            continue
        t = ((p1[0] - p0[0]) * d1[1] - (p1[1] - p0[1]) * d1[0]) / den
        corners.append((p0[0] + t * d0[0], p0[1] + t * d0[1]))
    corners = clean_polygon(corners, HULL_FLAT_M)
    if corners is None:
        return None
    z0, z1 = bbox[2], bbox[5]
    if z1 - z0 < 0.05:
        return None
    return corners, z0, z1


def clean_polygon(corners, tol):
    """Drop corners closer than `tol` to their predecessor and corners less than `tol` off the
    line through their neighbours (the wrap-around pair included), until nothing changes;
    None when fewer than three corners remain."""
    pts = [tuple(c) for c in corners]
    changed = True
    while changed and len(pts) >= 3:
        changed = False
        n = len(pts)
        for i in range(n):
            a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
            if math.hypot(b[0] - a[0], b[1] - a[1]) < tol:
                del pts[i]
                changed = True
                break
            base = math.hypot(c[0] - a[0], c[1] - a[1])
            off = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / max(base, 1e-12)
            if off < tol:
                del pts[i]
                changed = True
                break
    return pts if len(pts) >= 3 else None


def snap_corners(prisms, tol):
    """Corners of different prisms closer than `tol` become one point (the first seen), so
    neighbouring prisms share their vertices exactly and the fuse merges their edges instead
    of leaving two vertical edges millimetres apart - the mesher treats those as duplicate
    points and the 3-D boundary recovery fails on them."""
    if tol <= 0 or cKDTree is None:
        return 0
    flat = [(i, j, c) for i, (corners, z0, z1) in prisms.items() for j, c in enumerate(corners)]
    if len(flat) < 2:
        return 0
    P = np.array([c for i, j, c in flat])
    tree = cKDTree(P)
    parent = list(range(len(flat)))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b in tree.query_pairs(tol):
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    moved = 0
    for k, (i, j, c) in enumerate(flat):
        r = root(k)
        if r != k:
            prisms[i][0][j] = tuple(P[r])
            moved += 1
    return moved


def build_prism(corners, z0, z1):
    """An OCC prism over the polygon `corners` (ccw) from z0 to z1: (tag, bbox) or None."""
    corners = clean_polygon(corners, HULL_FLAT_M)      # snapping may have merged neighbours
    if corners is None:
        return None
    ptags = [gmsh.model.occ.addPoint(x, y, z0) for x, y in corners]
    ltags = [gmsh.model.occ.addLine(ptags[i], ptags[(i + 1) % len(ptags)]) for i in range(len(ptags))]
    loop = gmsh.model.occ.addCurveLoop(ltags)
    face = gmsh.model.occ.addPlaneSurface([loop])
    out = gmsh.model.occ.extrude([(2, face)], 0, 0, z1 - z0)
    vols = [t for d, t in out if d == 3]
    if len(vols) != 1:
        return None
    xs, ys = [c[0] for c in corners], [c[1] for c in corners]
    # the bbox from the corners: the prism is not synchronised into the model yet
    return vols[0], [min(xs), min(ys), z0, max(xs), max(ys), z1]


def cut_stage(cfg, args, work):
    """Repairs, the far-solid hull smear, the sink stretch, the optional fuse, the boolean cut,
    pockets dropped."""
    stage(2, 'cut')
    t = time.time()
    fluid = SUMMARY['fluid']
    fluid_mass0 = SUMMARY['fluid_mass_m3_imported']
    solid_bbox = SUMMARY['solid_bboxes']
    repaired = {}
    for entry in cfg['repairs']:
        brep = repair_one(cfg, entry, work)
        gmsh.model.occ.remove([(3, entry['tag'])], recursive=True)
        gmsh.option.setNumber('Geometry.OCCScaling', 1.0)   # the brep is already in metres
        imported = gmsh.model.occ.importShapes(brep)
        gmsh.model.occ.synchronize()
        new = [dt for dt in imported if dt[0] == 3]
        if len(new) != 1:
            die('the repaired solid %s came back as %d volumes, expected 1' % (brep, len(new)))
        if entry['lift_z']:
            gmsh.model.occ.translate(new, 0, 0, entry['lift_z'])
            gmsh.model.occ.synchronize()
            log('repaired tag %d lifted by %.2f m' % (entry['tag'], entry['lift_z']))
        new_tag = new[0][1]
        bbox = list(gmsh.model.getBoundingBox(3, new_tag))
        repaired[new_tag] = entry['tag']
        SUMMARY.setdefault('ship_hulls', []).append(bbox)
        SUMMARY.setdefault('replacements', []).append(
            {'file': os.path.basename(brep), 'old_tag': entry['tag'], 'new_tag': new_tag,
             'mass_m3': gmsh.model.occ.getMass(3, new_tag), 'bbox': [round(v, 2) for v in bbox],
             'lift_z': entry['lift_z']})
        log('tag %d replaced by %s (now tag %d): mass %.0f m^3, bbox x[%.1f,%.1f] y[%.1f,%.1f] '
            'z[%.1f,%.1f]' % (entry['tag'], os.path.basename(brep), new_tag,
                              gmsh.model.occ.getMass(3, new_tag), bbox[0], bbox[3],
                              bbox[1], bbox[4], bbox[2], bbox[5]))
        solid_bbox.pop(entry['tag'], None)

    # far from every release point a building's exact outline does not matter, but the slits
    # between neighbouring solids do: walls a few centimetres to a metre apart leave a slot
    # narrower than the local cell, which the mesher can only fill with slivers, and those are
    # where the steady solution blows up. So every solid farther than hull_beyond_m from the
    # nearest point is replaced by the prism of its footprint's convex hull, pushed out by
    # hull_pad_m: neighbours then overlap and the cut merges them, slits and all.
    hull_beyond = cfg['solids']['hull_beyond_m']
    if hull_beyond > 0:
        n_hull, n_kept, n_skipped = 0, 0, 0
        keep_tags = set(cfg['solids']['exclude_tags']) | {r['tag'] for r in cfg['repairs']}
        pool_xy = list(cfg['points'].values())
        prisms = {}
        for tg in sorted(solid_bbox):
            if tg in keep_tags:
                continue
            b = solid_bbox[tg]
            cx, cy = 0.5 * (b[0] + b[3]), 0.5 * (b[1] + b[4])
            if min(math.hypot(cx - px, cy - py) for px, py in pool_xy) <= hull_beyond:
                n_kept += 1
                continue
            made = hull_corners(tg, b, cfg['solids']['hull_pad_m'])
            if made is None:
                n_skipped += 1
                continue
            prisms[tg] = [list(made[0]), made[1], made[2]]
        n_snapped = snap_corners(prisms, cfg['solids']['hull_snap_m'])
        for tg, (corners, z0, z1) in prisms.items():
            made = build_prism(corners, z0, z1)
            if made is None:
                n_skipped += 1
                continue
            new_tag, new_bbox = made
            gmsh.model.occ.remove([(3, tg)], recursive=True)
            solid_bbox.pop(tg)
            solid_bbox[new_tag] = new_bbox
            n_hull += 1
        gmsh.model.occ.synchronize()
        log('solids beyond %.0f m of the points replaced by padded convex-hull prisms: %d '
            '(pad %.2f m, %d corners snapped onto neighbours within %.2f m); %d kept as '
            'modelled (near a point), %d skipped (degenerate outline)'
            % (hull_beyond, n_hull, cfg['solids']['hull_pad_m'], n_snapped,
               cfg['solids']['hull_snap_m'], n_kept, n_skipped))
        SUMMARY['solids_hulled'] = n_hull
        SUMMARY['hull_corners_snapped'] = n_snapped

    # the buildings' bases can sit above the terrain under them, which leaves a hairline air
    # layer the mesher fills with slivers the solver cannot survive: stretch every other solid
    # downwards about its roof, so the base sinks sink_m into the ground; roofs and walls stay
    sink = cfg['solids']['sink_m']
    tools = []
    for tg in sorted(solid_bbox):
        if tg in cfg['solids']['exclude_tags']:
            continue
        tools.append((3, tg))
        if sink > 0:
            b = solid_bbox[tg]
            h, w, dp = b[5] - b[2], b[3] - b[0], b[4] - b[1]
            if h > 0.05 and w > 0.05 and dp > 0.05:
                gmsh.model.occ.dilate([(3, tg)], 0.5 * (b[0] + b[3]), 0.5 * (b[1] + b[4]), b[5],
                                      1.0, 1.0, (h + sink) / h)
    if sink > 0 and tools:
        gmsh.model.occ.synchronize()
        log('solids stretched %.1f m below their bases (about the roof centre): none floats' % sink)
        SUMMARY['solids_sink_m'] = sink

    # solids left out of the cut entirely leave the model: they would else stay as volumes
    # the mesher has no physical group for
    excluded = [(3, tg) for tg in cfg['solids']['exclude_tags'] if tg in solid_bbox]
    if excluded:
        gmsh.model.occ.remove(excluded, recursive=True)
        gmsh.model.occ.synchronize()
        for tg in cfg['solids']['exclude_tags']:
            solid_bbox.pop(tg, None)
        log('excluded from the cut and removed: tags %s' % sorted(cfg['solids']['exclude_tags']))
        SUMMARY['solids_excluded'] = sorted(cfg['solids']['exclude_tags'])

    # a fuzzy boolean merges the near-coincident vertices and edges that overlapping tools
    # (padded hull prisms crossing each other) would otherwise leave millimetres apart; it must
    # be on for the fuse as well as for the cut, or the fuse creates exactly those vertices
    if cfg['solids']['boolean_tol_m'] > 0:
        gmsh.option.setNumber('Geometry.ToleranceBoolean', cfg['solids']['boolean_tol_m'])
        log('boolean tolerance (fuzzy fuse and cut): %.4f m' % cfg['solids']['boolean_tol_m'])

    # neighbouring solids touch, overlap or stand a few centimetres apart in the STEP, which
    # leaves coincident faces and hairline slits in the fluid; fusing them first merges all that
    if cfg['solids']['fuse'] and len(tools) > 1:
        t_f = time.time()
        fused, _ = gmsh.model.occ.fuse(tools[:1], tools[1:], removeObject=True, removeTool=True)
        gmsh.model.occ.synchronize()
        tools = [dt for dt in fused if dt[0] == 3]
        log('solids fused: -> %d solids, %.0f s' % (len(tools), time.time() - t_f))
        SUMMARY['solids_fused'] = len(tools)
        solid_bbox = {tg: list(gmsh.model.getBoundingBox(3, tg)) for d, tg in tools}
    # the union of two prisms whose walls are nearly in line keeps a step of centimetres
    # between them - a face of a few square centimetres with edges of millimetres, which the
    # mesher can only fill with slivers (OCC's shape healing removes them but leaves a shape
    # the 3-D mesher refuses). So each fused group of far solids is replaced once more by the
    # prism of its own convex hull: one convex block per group, no steps, no slits
    if hull_beyond > 0 and cfg['solids']['fuse'] and len(tools) > 1:
        t_g = time.time()
        far_tags = [tg for d, tg in tools if tg not in repaired]
        prisms = {}
        for tg in far_tags:
            made = hull_corners(tg, solid_bbox[tg], 0.0)
            if made is not None:
                prisms[tg] = [list(made[0]), made[1], made[2]]
        n_snapped = snap_corners(prisms, cfg['solids']['hull_snap_m'])
        n_groups = 0
        for tg, (corners, z0, z1) in prisms.items():
            made = build_prism(corners, z0, z1)
            if made is None:
                continue
            new_tag, new_bbox = made
            gmsh.model.occ.remove([(3, tg)], recursive=True)
            solid_bbox.pop(tg, None)
            solid_bbox[new_tag] = new_bbox
            n_groups += 1
        gmsh.model.occ.synchronize()
        tools = [(3, tg) for tg in solid_bbox if tg not in repaired]
        log('fused groups replaced by the prisms of their convex hulls: %d (%d corners '
            'snapped), %.0f s' % (n_groups, n_snapped, time.time() - t_g))
        SUMMARY['fused_groups_hulled'] = n_groups
    tools += [(3, tg) for tg in repaired]

    tool_mass = sum(gmsh.model.occ.getMass(3, tg) for d, tg in tools)
    out, _ = gmsh.model.occ.cut([(3, fluid)], tools, removeObject=True, removeTool=True)
    gmsh.model.occ.synchronize()
    if cfg['solids']['boolean_tol_m'] > 0:
        gmsh.option.setNumber('Geometry.ToleranceBoolean', 0.0)   # the pool cuts stay exact
    masses = sorted(((gmsh.model.occ.getMass(3, tg), tg) for d, tg in out), reverse=True)
    if not masses:
        die('the cut returned no volume at all - nothing to mesh')
    expected = fluid_mass0 - tool_mass
    if abs(masses[0][0] - expected) / abs(expected) > EXPECT_SLACK:
        die('the cut gave a largest volume of %.4e m^3, expected %.4e (fluid %.4e minus the '
            '%d tool solids)' % (masses[0][0], expected, fluid_mass0, len(tools)))
    fluid = masses[0][1]
    SUMMARY['pockets'] = []
    for m, tg in masses[1:]:
        b = gmsh.model.getBoundingBox(3, tg)
        SUMMARY['pockets'].append({'mass_m3': m, 'bbox': [round(v, 2) for v in b]})
        log('sealed pocket removed: %.3e m^3 at x[%.0f,%.0f] y[%.0f,%.0f] z[%.1f,%.1f]'
            % (m, b[0], b[3], b[1], b[4], b[2], b[5]))
    if masses[1:]:
        gmsh.model.occ.remove([(3, tg) for m, tg in masses[1:]], recursive=True)
        gmsh.model.occ.synchronize()
    vols = gmsh.model.getEntities(3)
    if len(vols) != 1:
        die('after the cut the model holds %d volumes, expected 1 (the fluid)' % len(vols))
    n_surf = len(gmsh.model.getBoundary([(3, fluid)], oriented=False))
    log('cut done: fluid tag %d, mass %.4e m^3, %d boundary surfaces, %d pockets dropped'
        % (fluid, masses[0][0], n_surf, len(masses) - 1))
    # two solids whose vertices stand less than touch_warn_m apart leave a razor-thin fluid gap
    # the mesher may treat as duplicate points, and whether the 3-D boundary recovery survives
    # one depends on node numbering luck. Diagnostic only: record the pairs, move and fuse
    # nothing. Pairs closer than 1 mm OCC/gmsh already merges and are harmless.
    warn = cfg['solids']['touch_warn_m']
    near = []
    if warn > 0 and cKDTree is None:
        log('near-touching solids: scan skipped (scipy is not installed)')
    elif warn > 0:
        vtags = [tg for d, tg in gmsh.model.getEntities(0)]
        if len(vtags) > 1:
            V = np.array([gmsh.model.getValue(0, tg, []) for tg in vtags])

            def solids_at(p):
                return {tg for tg, bb in solid_bbox.items()
                        if bb[0] - 0.01 <= p[0] <= bb[3] + 0.01
                        and bb[1] - 0.01 <= p[1] <= bb[4] + 0.01
                        and bb[2] - 0.01 <= p[2] <= bb[5] + 0.01}

            seen = {}
            for i, j in cKDTree(V).query_pairs(warn):
                d = float(np.linalg.norm(V[i] - V[j]))
                if d <= 1e-3:
                    continue
                mid = 0.5 * (V[i] + V[j])
                key = tuple(np.round(mid, 2))
                if key not in seen:
                    seen[key] = {'d': round(d, 4), 'xyz': [round(v, 2) for v in mid],
                                 'solids': sorted(solids_at(V[i]) | solids_at(V[j]))}
            near = sorted(seen.values(), key=lambda r: r['d'])
    for rec in near[:30]:
        log('near-touching solids: d %.4f m at (%.2f, %.2f, %.2f) solids %s'
            % (rec['d'], rec['xyz'][0], rec['xyz'][1], rec['xyz'][2], rec['solids']))
    if len(near) > 30:
        log('... %d near-touching pair(s) more; the full list is in the summary'
            % (len(near) - 30))
    if near:
        log('near-touching solid vertices: %d pairs - a 3-D boundary-recovery failure usually '
            'sits at one of them; exclude one solid of the pair with solids.exclude_tags'
            % len(near))
    SUMMARY['near_touching'] = near
    gmsh.write(os.path.join(work, '%s_cut.brep' % cfg['name']))   # the pool-failure restore point
    SUMMARY['fluid'] = fluid
    SUMMARY['fluid_mass_m3'] = masses[0][0]
    SUMMARY['solid_bboxes'] = solid_bbox
    tick('cut', t)


# --------------------------------------------------- stage 3: ground heights
def ground_stage(cfg, args, work):
    stage(3, 'ground heights')
    t = time.time()
    fluid = SUMMARY['fluid']
    z0, dz, z1 = GROUND_SCAN

    def ground_z(x, y):
        z = z0
        while z < z1:
            if gmsh.model.isInside(3, fluid, [x, y, z]):
                return z
            z += dz
        return None

    SUMMARY['points'] = {}
    for name, (x, y) in cfg['points'].items():
        zg = ground_z(x, y)
        SUMMARY['points'][name] = {'x': x, 'y': y, 'z_ground': zg,
                                   'r': cfg['pool_specs'][name]['r'],
                                   'r_inner': cfg['pool_specs'][name]['r_inner'],
                                   'h': cfg['pool_specs'][name]['h']}
        log('ground under %-14s (%8.1f, %7.1f) -> z_g = %s' % (name, x, y, zg))
        if zg is None:
            die('no fluid found under the point %s at (%.1f, %.1f) between z = %g and %g'
                % (name, x, y, z0, z1))
    tick('ground', t)

    if args.dry_run:
        n_surf = len(gmsh.model.getBoundary([(3, fluid)], oriented=False))
        print('', flush=True)
        log('DRY RUN: stopping after the import and the boolean cut')
        print('  volumes        : 1 (fluid tag %d)%s' % (
            fluid, ', %d sealed pockets dropped' % len(SUMMARY['pockets'])
            if SUMMARY['pockets'] else ''), flush=True)
        print('  masses         : fluid %.4e m^3' % SUMMARY['fluid_mass_m3'], flush=True)
        print('  surface count  : %d boundary surfaces' % n_surf, flush=True)
        for name, p in SUMMARY['points'].items():
            print('  ground %-14s (%8.1f, %7.1f) -> z_g = %s'
                  % (name, p['x'], p['y'], p['z_ground']), flush=True)
        with open(os.path.join(work, 'dry_run_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(SUMMARY, f, indent=1, ensure_ascii=False)
        raise SystemExit(0)


# ---------------------------------------------------- stage 4: classification
def classify(cfg):
    """Every boundary surface of the fluid into exactly one named group."""
    prefix = cfg['classification']['wall_prefix']
    names = ('top', 'west', 'east', 'south', 'north',
             prefix + 'sea_surface', prefix + 'ship_hull', prefix + 'buildings',
             prefix + 'ground_land')
    groups = {k: [] for k in names}
    for name in cfg['points']:
        groups['pool_' + name] = []
        if cfg['pool_specs'][name]['h'] > 0:
            groups[prefix + 'pool_' + name] = []      # the cylinder's side
    for name in cfg['roof_patches']:
        groups[name] = []
    box = cfg['domain_box']
    btol = cfg['outer_tol']
    sea_z = cfg['sea_z']
    hulls = SUMMARY.get('ship_hulls', [])
    big_m2 = cfg['classification']['big_roof_is_ground_m2']
    pools = SUMMARY['pools']

    def inside(sb, bb, pad):
        return (sb[0] >= bb[0] - pad and sb[3] <= bb[3] + pad and
                sb[1] >= bb[1] - pad and sb[4] <= bb[4] + pad and
                sb[2] >= bb[2] - pad and sb[5] <= bb[5] + pad)

    fluid = SUMMARY['fluid']
    for d, sf in gmsh.model.getBoundary([(3, fluid)], oriented=False):
        b = gmsh.model.getBoundingBox(2, sf)
        flat = (b[5] - b[2]) < FLAT_TOL
        # a pool piece is a flat face at z_g whose bbox lies inside the disc's square and whose
        # centre lies inside the circle. A disc imprinted on a roof comes back as several pieces
        # (the roof's own edges cut it); the lens shared by two overlapping discs goes to the
        # nearer centre.
        pooled = False
        if flat:
            best = None
            for name, (x, y) in cfg['points'].items():
                zg = SUMMARY['points'][name]['z_ground']
                pool_r = cfg['pool_specs'][name]['r']
                pool_ri = cfg['pool_specs'][name]['r_inner']
                zg = zg + cfg['pool_specs'][name]['h']             # a raised pool's top
                if pools.get(name) != 'imprinted' or abs(b[2] - zg) > POOL_Z_TOL:
                    continue
                if (b[0] < x - pool_r - POOL_PAD or b[3] > x + pool_r + POOL_PAD or
                        b[1] < y - pool_r - POOL_PAD or b[4] > y + pool_r + POOL_PAD):
                    continue
                # a ring's pieces are the ones that do NOT fit inside the inner square: the
                # inner disc (its own point) fits there, the ring's bbox cannot
                if pool_ri > 0 and (b[0] >= x - pool_ri - POOL_PAD and b[3] <= x + pool_ri + POOL_PAD
                                    and b[1] >= y - pool_ri - POOL_PAD
                                    and b[4] <= y + pool_ri + POOL_PAD):
                    continue
                cx, cy = 0.5 * (b[0] + b[3]), 0.5 * (b[1] + b[4])
                r = math.hypot(cx - x, cy - y)
                # the smaller radius wins where two pools of one centre both fit (the inner
                # disc before its ring); otherwise the nearer centre, as before
                key = (pool_r, r)
                if r <= pool_r + POOL_PAD and (best is None or key < best[0]):
                    best = (key, name)
            if best is not None:
                groups['pool_' + best[1]].append(sf)
                pooled = True
        if pooled:
            continue
        if not flat:
            # the side of a raised pool: a curved face inside the disc's square between the
            # ground and the top; where an inner disc and its ring share a centre, the side
            # belongs to the larger radius (the smaller cylinder was cut away by the larger)
            side = None
            for name, (x, y) in cfg['points'].items():
                sp = cfg['pool_specs'][name]
                if sp['h'] <= 0 or pools.get(name) != 'imprinted':
                    continue
                zg = SUMMARY['points'][name]['z_ground']
                if (b[0] >= x - sp['r'] - POOL_PAD and b[3] <= x + sp['r'] + POOL_PAD
                        and b[1] >= y - sp['r'] - POOL_PAD and b[4] <= y + sp['r'] + POOL_PAD
                        and b[2] >= zg - POOL_Z_TOL and b[5] <= zg + sp['h'] + POOL_Z_TOL):
                    if side is None or sp['r'] > cfg['pool_specs'][side]['r']:
                        side = name
            if side is not None:
                groups[prefix + 'pool_' + side].append(sf)
                continue
        roofed = False
        for rname, rtag in cfg['roof_patches'].items():
            rb = SUMMARY['solid_bboxes'].get(rtag)
            if (rb is not None and flat and inside(b, rb, ROOF_PAD)
                    and abs(b[2] - rb[5]) < ROOF_Z_TOL):
                groups[rname].append(sf)            # the release building's roof
                roofed = True
                break
        if roofed:
            continue
        if abs(b[2] - box[5]) < btol and abs(b[5] - box[5]) < btol:
            groups['top'].append(sf)
        elif abs(b[0] - box[0]) < btol and abs(b[3] - box[0]) < btol:
            groups['west'].append(sf)
        elif abs(b[0] - box[3]) < btol and abs(b[3] - box[3]) < btol:
            groups['east'].append(sf)
        elif abs(b[1] - box[1]) < btol and abs(b[4] - box[1]) < btol:
            groups['south'].append(sf)
        elif abs(b[1] - box[4]) < btol and abs(b[4] - box[4]) < btol:
            groups['north'].append(sf)
        elif flat and abs(b[2] - sea_z) < SEA_TOL:
            groups[prefix + 'sea_surface'].append(sf)
        elif any(inside(b, hb, HULL_PAD) for hb in hulls):
            groups[prefix + 'ship_hull'].append(sf)
        else:
            hit = None
            for tg, bb in SUMMARY['solid_bboxes'].items():
                if inside(b, bb, ROOF_PAD):
                    hit = tg
                    break
            if hit is not None:
                # the flat roof of a very large slab is terrain, not a building
                bb = SUMMARY['solid_bboxes'][hit]
                big = bb[3] - bb[0] > BIG_ROOF_XY and bb[4] - bb[1] > BIG_ROOF_XY
                if (flat and big and abs(b[2] - bb[5]) < ROOF_Z_TOL
                        and gmsh.model.occ.getMass(2, sf) > big_m2):
                    groups[prefix + 'ground_land'].append(sf)
                else:
                    groups[prefix + 'buildings'].append(sf)
            else:
                groups[prefix + 'ground_land'].append(sf)
    return groups


def report_groups(cfg, groups):
    total = len(gmsh.model.getBoundary([(3, SUMMARY['fluid'])], oriented=False))
    n = sum(len(v) for v in groups.values())
    if n != total:
        die('classification covers %d of %d boundary surfaces' % (n, total))
    for k in ('top', 'west', 'east', 'south', 'north'):
        if not groups[k]:
            die('the outer group %s came out empty - check domain_box %s' % (k, cfg['domain_box']))
    for k, v in groups.items():
        if v:
            area = sum(gmsh.model.occ.getMass(2, sf) for sf in v)
            SUMMARY['groups'][k] = {'surfaces': len(v), 'area_m2': round(area, 1)}
            log('  %-18s %5d surface(s)  %12.1f m^2' % (k, len(v), area))


# -------------------------------------------------------- stage 5: pool discs
def pool_stage(cfg, args, work):
    stage(5, 'pool discs')
    t = time.time()
    cut_brep = os.path.join(work, '%s_cut.brep' % cfg['name'])
    fluid = SUMMARY['fluid']
    for name, (x, y) in cfg['points'].items():
        zg = SUMMARY['points'][name]['z_ground']
        pool_r = cfg['pool_specs'][name]['r']
        pool_ri = cfg['pool_specs'][name]['r_inner']
        pool_h = cfg['pool_specs'][name]['h']
        try:
            if pool_h > 0:
                # the circle drawn on the ground and pulled up h: a cylinder cut out of the
                # fluid, whose top at z_g + h is the pool and whose side becomes a wall
                cyl = gmsh.model.occ.addCylinder(x, y, zg, 0, 0, pool_h, pool_r)
                gmsh.model.occ.cut([(3, fluid)], [(3, cyl)], removeObject=True, removeTool=True)
                gmsh.model.occ.synchronize()
                vols = gmsh.model.getEntities(3)
                fluid = max(vols, key=lambda dt: gmsh.model.occ.getMass(3, dt[1]))[1]
                if pool_ri > 0:                      # split the top into the inner disc and the ring
                    disk = gmsh.model.occ.addDisk(x, y, zg + pool_h, pool_ri, pool_ri)
                    _, outmap = gmsh.model.occ.fragment([(3, fluid)], [(2, disk)])
                    gmsh.model.occ.synchronize()
                    vols = gmsh.model.getEntities(3)
                    fluid = max(vols, key=lambda dt: gmsh.model.occ.getMass(3, dt[1]))[1]
                    bset = set(s for d, s in gmsh.model.getBoundary([(3, fluid)], oriented=False))
                    loose = [(2, s) for d, s in outmap[1] if d == 2 and s not in bset]
                    if loose:
                        gmsh.model.occ.remove(loose, recursive=False)
                        gmsh.model.occ.synchronize()
                top = []
                for d, s in gmsh.model.getBoundary([(3, fluid)], oriented=False):
                    b = gmsh.model.getBoundingBox(2, s)
                    if ((b[5] - b[2]) < FLAT_TOL and abs(b[2] - (zg + pool_h)) < POOL_Z_TOL
                            and b[0] >= x - pool_r - POOL_PAD and b[3] <= x + pool_r + POOL_PAD
                            and b[1] >= y - pool_r - POOL_PAD and b[4] <= y + pool_r + POOL_PAD):
                        top.append(s)
                if not top:
                    SUMMARY['pools'][name] = 'not imprinted: no top face at z_g + h'
                    log('pool %-14s NOT raised (no top face at z=%.2f)' % (name, zg + pool_h))
                else:
                    SUMMARY['pools'][name] = 'imprinted'
                    log('pool %-14s raised as a cylinder, h %.2f m (%d top face(s), r %.2f%s)' % (
                        name, pool_h, len(top), pool_r,
                        ', r_inner %.2f' % pool_ri if pool_ri > 0 else ''))
                continue
            disk = gmsh.model.occ.addDisk(x, y, zg, pool_r, pool_r)
            if pool_ri > 0:                          # an annulus: the ring between r_inner and r
                hole = gmsh.model.occ.addDisk(x, y, zg, pool_ri, pool_ri)
                ring, _ = gmsh.model.occ.cut([(2, disk)], [(2, hole)], removeObject=True,
                                             removeTool=True)
                disk = ring[0][1]
            _, outmap = gmsh.model.occ.fragment([(3, fluid)], [(2, disk)])
            gmsh.model.occ.synchronize()
            vols = gmsh.model.getEntities(3)
            fluid = max(vols, key=lambda dt: gmsh.model.occ.getMass(3, dt[1]))[1]
            bset = set(s for d, s in gmsh.model.getBoundary([(3, fluid)], oriented=False))
            disk_faces = [s for d, s in outmap[1] if d == 2]
            on_boundary = [s for s in disk_faces if s in bset]
            if not on_boundary:
                # not coplanar with the ground here: the disc was embedded, not imprinted
                gmsh.model.occ.remove([(2, s) for s in disk_faces if s not in bset],
                                      recursive=False)
                gmsh.model.occ.synchronize()
                SUMMARY['pools'][name] = 'not imprinted: ground not planar at z_g'
                log('pool %-14s NOT imprinted (ground not planar at z=%.2f)' % (name, zg))
            else:
                SUMMARY['pools'][name] = 'imprinted'
                log('pool %-14s imprinted (%d face(s), r %.2f%s)' % (
                    name, len(on_boundary), pool_r,
                    ', r_inner %.2f' % pool_ri if pool_ri > 0 else ''))
        except Exception as e:                      # restore the checkpoint, go on without it
            log('pool %-14s FAILED: %s -> restoring the post-cut checkpoint' % (name, e))
            SUMMARY['pools'][name] = 'failed: %s' % e
            gmsh.model.remove()
            gmsh.model.add(cfg['name'])
            gmsh.option.setNumber('Geometry.OCCScaling', 1.0)   # the checkpoint is in metres
            gmsh.model.occ.importShapes(cut_brep)
            gmsh.model.occ.synchronize()
            vols = gmsh.model.getEntities(3)
            fluid = max(vols, key=lambda dt: gmsh.model.occ.getMass(3, dt[1]))[1]
            for k in list(SUMMARY['pools']):
                if SUMMARY['pools'][k] == 'imprinted':
                    SUMMARY['pools'][k] = 'lost when a later pool failed and the checkpoint was restored'
    SUMMARY['fluid'] = fluid
    tick('pools', t)


# ------------------------------------------------------- stage 6: checkpoint
def checkpoint_stage(cfg, args, work):
    stage(6, 'checkpoint')
    ckpt_brep = os.path.join(work, '%s_pools.brep' % cfg['name'])
    if not args.from_checkpoint:
        gmsh.write(ckpt_brep)
        with open(ckpt_brep[:-5] + '.json', 'w', encoding='utf-8') as f:
            json.dump({'points': SUMMARY['points'], 'pools': SUMMARY['pools'],
                       'ship_hulls': SUMMARY.get('ship_hulls', []),
                       'solid_bboxes': SUMMARY['solid_bboxes'],
                       'fluid_mass_m3': SUMMARY['fluid_mass_m3'],
                       'pockets': SUMMARY['pockets'],
                       'near_touching': SUMMARY.get('near_touching', [])}, f, indent=1)
        log('checkpoint written: %s + .json (--from-checkpoint reruns from here)'
            % os.path.basename(ckpt_brep))
    else:
        log('checkpoint stage skipped (the run started from it)')
    if args.stop_after_checkpoint:
        log('--stop-after-checkpoint: stopping here')
        raise SystemExit(0)


# ------------------------------------------------------------- stage 7: trim
def trim_stage(cfg, work):
    trim = cfg['trim']
    if not trim or trim.get('below_z') is None:
        stage(7, 'trim')
        log('no trim (config.trim is null): the model keeps everything below')
        return
    stage(7, 'trim')
    t = time.time()
    z_trim = float(trim['below_z'])
    bb = gmsh.model.getBoundingBox(-1, -1)
    z0 = bb[2] - TRIM_MARGIN[1]
    box = gmsh.model.occ.addBox(bb[0] - TRIM_MARGIN[0], bb[1] - TRIM_MARGIN[0], z0,
                                (bb[3] - bb[0]) + 2 * TRIM_MARGIN[0],
                                (bb[4] - bb[1]) + 2 * TRIM_MARGIN[0], z_trim - z0)
    fluid = SUMMARY['fluid']
    m0 = gmsh.model.occ.getMass(3, fluid)
    n0 = len(gmsh.model.getBoundary([(3, fluid)], oriented=False))
    out, _ = gmsh.model.occ.cut([(3, fluid)], [(3, box)], removeObject=True, removeTool=True)
    gmsh.model.occ.synchronize()
    vols = gmsh.model.getEntities(3)
    masses = sorted(((gmsh.model.occ.getMass(3, tg), tg) for d, tg in vols), reverse=True)
    if not masses:
        die('the trim cut at z = %g left no volume' % z_trim)
    for m, tg in masses[1:]:
        b = gmsh.model.getBoundingBox(3, tg)
        log('   extra volume %.3e m^3 at x[%.0f,%.0f] y[%.0f,%.0f] z[%.2f,%.2f] -> removed'
            % (m, b[0], b[3], b[1], b[4], b[2], b[5]))
        gmsh.model.occ.remove([(3, tg)], recursive=True)
    gmsh.model.occ.synchronize()
    fluid = gmsh.model.getEntities(3)[0][1]
    surfs = [s for d, s in gmsh.model.getBoundary([(3, fluid)], oriented=False)]
    sea = [s for s in surfs if abs(gmsh.model.getBoundingBox(2, s)[2] - z_trim) < 0.005
           and abs(gmsh.model.getBoundingBox(2, s)[5] - z_trim) < 0.005]
    below = [s for s in surfs if gmsh.model.getBoundingBox(2, s)[2] < z_trim - 0.005]
    log('trim cut: %d volume(s), boundary faces %d (before %d); flat faces at z = %g: %d, '
        'area %.0f m^2; faces reaching below: %d'
        % (len(masses), len(surfs), n0, z_trim, len(sea),
           sum(gmsh.model.occ.getMass(2, s) for s in sea), len(below)))
    gmsh.write(os.path.join(work, '%s_trimmed.brep' % cfg['name']))
    SUMMARY['fluid'] = fluid
    SUMMARY['trim_below_z'] = z_trim
    SUMMARY['fluid_mass_m3'] = masses[0][0]
    SUMMARY['trim_mass_removed_m3'] = m0 - masses[0][0]
    tick('trim', t)


# --------------------------------------- stage 8: physical groups + size fields
def groups_and_fields_stage(cfg):
    stage(8, 'groups + fields')
    t = time.time()
    SUMMARY['groups'] = {}
    groups = classify(cfg)
    log('classification after pools and trim:')
    report_groups(cfg, groups)

    gmsh.model.addPhysicalGroup(3, [SUMMARY['fluid']], name='fluid')
    for k, v in groups.items():
        if v:
            gmsh.model.addPhysicalGroup(2, v, name=k)

    mult = cfg['sizes']['size_mult']
    smin, smax = cfg['sizes']['min'] * mult, cfg['sizes']['max'] * mult
    gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 0)
    gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 0)
    gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 0)
    gmsh.option.setNumber('Mesh.MeshSizeMin', smin)
    gmsh.option.setNumber('Mesh.MeshSizeMax', smax)
    F = gmsh.model.mesh.field
    fields = []

    def box(x0, x1, y0, y1, z0, z1, vin, thick):
        f = F.add('Box')
        F.setNumber(f, 'XMin', x0)
        F.setNumber(f, 'XMax', x1)
        F.setNumber(f, 'YMin', y0)
        F.setNumber(f, 'YMax', y1)
        F.setNumber(f, 'ZMin', z0)
        F.setNumber(f, 'ZMax', z1)
        F.setNumber(f, 'VIn', vin * mult)
        F.setNumber(f, 'VOut', smax)
        F.setNumber(f, 'Thickness', thick * mult)
        return f

    def threshold(dist_field, smin_l, dmin, smax_l, dmax):
        f = F.add('Threshold')
        F.setNumber(f, 'InField', dist_field)
        F.setNumber(f, 'SizeMin', smin_l * mult)
        F.setNumber(f, 'DistMin', dmin)
        F.setNumber(f, 'SizeMax', smax_l * mult)
        F.setNumber(f, 'DistMax', dmax)
        return f

    for name, (x, y) in cfg['points'].items():
        zg = SUMMARY['points'][name]['z_ground']
        half = max(POINT_BOX[0], cfg['pool_specs'][name]['r'] + 15.0)   # a big pool widens its box
        fields.append(box(x - half, x + half, y - half, y + half,
                          zg - POINT_BOX[1], zg + POINT_BOX[2], cfg['sizes']['pool'],
                          POINT_BOX[3]))                                  # pool + first 10 m
        fields.append(box(x - POINT_BOX_FAR[0], x + POINT_BOX_FAR[0],
                          y - POINT_BOX_FAR[0], y + POINT_BOX_FAR[0],
                          zg - POINT_BOX_FAR[1], zg + POINT_BOX_FAR[2], cfg['sizes']['box'],
                          POINT_BOX_FAR[3]))                              # the refinement box
        d = F.add('Distance')
        F.setNumbers(d, 'PointsList', [gmsh.model.occ.addPoint(x, y, zg + 2)])
        gmsh.model.occ.synchronize()
        fields.append(threshold(d, cfg['sizes']['growth_from'], GROWTH[0],
                                cfg['sizes']['max'], GROWTH[1]))          # growth from the source

    for rname, rtag in cfg['roof_patches'].items():
        sb = SUMMARY['solid_bboxes'].get(rtag)
        if sb is None:
            die('the roof patch %s names solid tag %d, which is not in the model' % (rname, rtag))
        if not groups[rname]:
            die('no roof face found for the roof patch %s (solid tag %d)' % (rname, rtag))
        top = sb[5]
        cy = 0.5 * (sb[1] + sb[4])
        fields.append(box(sb[0] - ROOF_BOXES[0][0], sb[3] + ROOF_BOXES[0][0],
                          sb[1] - ROOF_BOXES[0][0], sb[4] + ROOF_BOXES[0][0],
                          top - ROOF_BOXES[0][1], top + ROOF_BOXES[0][2],
                          cfg['sizes']['roof_boxes'][0], ROOF_BOXES[0][3]))   # roof + first 40 m
        fields.append(box(sb[0] - ROOF_BOXES[1][0], sb[3] + ROOF_BOXES[1][0],
                          sb[1] - ROOF_BOXES[1][0], sb[4] + ROOF_BOXES[1][0],
                          top - ROOF_BOXES[1][1], top + ROOF_BOXES[1][2],
                          cfg['sizes']['roof_boxes'][1], ROOF_BOXES[1][3]))   # the near field
        fields.append(box(sb[0] - ROOF_DOWNWIND[0], sb[3] + ROOF_DOWNWIND[1],          # downwind (-x)
                          cy - ROOF_DOWNWIND[2], cy + ROOF_DOWNWIND[2],
                          top - ROOF_DOWNWIND[3], top + ROOF_DOWNWIND[4],
                          cfg['sizes']['roof_boxes'][2], ROOF_DOWNWIND[5]))
        log('roof patch %s (tag %d): %d face(s), %.0f m^2 at z = %.1f; refinement boxes '
            '%g / %g / %g m' % (rname, rtag, len(groups[rname]),
                                sum(gmsh.model.occ.getMass(2, sf) for sf in groups[rname]), top,
                                *cfg['sizes']['roof_boxes']))

    def near_a_point(sf, R):
        b = gmsh.model.getBoundingBox(2, sf)
        cx, cy = 0.5 * (b[0] + b[3]), 0.5 * (b[1] + b[4])
        return any(math.hypot(cx - x, cy - y) < R for x, y in cfg['points'].values())

    prefix = cfg['classification']['wall_prefix']
    structures = groups[prefix + 'buildings'] + groups[prefix + 'ship_hull']
    near = [s for s in structures if near_a_point(s, cfg['sizes']['near_radius'])]
    far = [s for s in structures if not near_a_point(s, cfg['sizes']['near_radius'])]
    if near:
        d = F.add('Distance')
        F.setNumbers(d, 'SurfacesList', near)
        F.setNumber(d, 'Sampling', NEAR_FAR_SAMPLING[0])
        fields.append(threshold(d, cfg['sizes']['near_struct'], 0, cfg['sizes']['max'], NEAR_FAR[0]))
    if far:
        d = F.add('Distance')
        F.setNumbers(d, 'SurfacesList', far)
        F.setNumber(d, 'Sampling', NEAR_FAR_SAMPLING[1])
        fields.append(threshold(d, cfg['sizes']['far_struct'], 0, cfg['sizes']['max'], NEAR_FAR[1]))
    fmin = F.add('Min')
    F.setNumbers(fmin, 'FieldsList', fields)
    F.setAsBackgroundMesh(fmin)
    BASE_FIELDS[:] = fields
    log('size fields: %d (near-source structures %d surfaces, far %d)'
        % (len(fields), len(near), len(far)))
    SUMMARY['structures_near_points'] = len(near)
    tick('fields', t)


BASE_FIELDS = []                   # the size fields of stage 8, for the gap pass of stage 9
GAP_GRID = (10.0, 5.0)             # xy and z cell of the grid the gap refinements are boxed on
GAP_MAX_BOXES = 2500               # the most box fields the gap pass adds (each is cheap to evaluate)


def gap_size_pass(cfg, ratio):
    """Local-feature-size rule h <= g / ratio. From the surface mesh just built, every triangle
    looks along its inward normal for the nearest triangle facing it (normals opposed, within
    the triangle's own footprint); that distance g is the width of the slot the triangle sits
    on, and a cell wider than g / ratio cannot fit in it without turning into a sliver. Where
    g / ratio is below the local size the triangle is marked for refinement; the marks are
    binned on a coarse grid and become Box fields so the volume mesh sees them too. Slots
    narrower than ratio x sizes.min cannot be resolved at all: those are reported, they are
    what the geometry smear (solids.hull_*) is for. Returns (boxes added, unresolvable spots)."""
    if cKDTree is None:
        log('gap size pass skipped (scipy is not installed)')
        return 0, 0
    mult = cfg['sizes']['size_mult']
    smin = cfg['sizes']['min'] * mult
    fluid = SUMMARY['fluid']
    ntags, coords, _ = gmsh.model.mesh.getNodes()
    xyz = np.array(coords).reshape(-1, 3)
    idx = np.zeros(int(ntags.max()) + 1, dtype=np.int64)
    idx[ntags] = np.arange(len(ntags))
    cents, norms, sizes, lmax, areas = [], [], [], [], []
    for d, s in gmsh.model.getBoundary([(3, fluid)], oriented=True, combined=True):
        sign = 1.0 if s > 0 else -1.0
        et, etags, en = gmsh.model.mesh.getElements(2, abs(s))
        for typ, nodes in zip(et, en):
            if typ != 2:
                continue
            tri = np.array(nodes, dtype=np.int64).reshape(-1, 3)
            P = xyz[idx[tri]]
            n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
            a2 = np.linalg.norm(n, axis=1)
            keep = a2 > 1e-14
            n = -sign * n[keep] / a2[keep][:, None]          # into the fluid
            cents.append(P[keep].mean(axis=1))
            norms.append(n)
            sizes.append(1.52 * np.sqrt(0.5 * a2[keep]))     # the equilateral edge of that area
            Pk = P[keep]
            lmax.append(np.max(np.stack([np.linalg.norm(Pk[:, 1] - Pk[:, 0], axis=1),
                                         np.linalg.norm(Pk[:, 2] - Pk[:, 1], axis=1),
                                         np.linalg.norm(Pk[:, 0] - Pk[:, 2], axis=1)], axis=1), axis=1))
            areas.append(0.5 * a2[keep])
    if not cents:
        return 0, 0
    C = np.concatenate(cents); N = np.concatenate(norms); E = np.concatenate(sizes)
    Lm = np.concatenate(lmax); Ar = np.concatenate(areas)
    tree = cKDTree(C)
    gap = np.full(len(C), np.inf)
    radius = np.maximum(4.0 * E, 2.0)
    for i, nb in enumerate(tree.query_ball_point(C, radius)):
        if len(nb) < 2:
            continue
        nb = np.array(nb)
        nb = nb[nb != i]
        v = C[nb] - C[i]
        t = v @ N[i]
        facing = (N[nb] @ N[i] < -0.5) & (t > 0.01)
        if not facing.any():
            continue
        lat = np.linalg.norm(v[facing] - np.outer(t[facing], N[i]), axis=1)
        ok = lat < E[i] + E[nb][facing]
        if ok.any():
            gap[i] = t[facing][ok].min()
    have = np.isfinite(gap)
    h_gap = gap / ratio
    unres = have & (h_gap < smin)
    # the floor: far from the points a slot is not worth cells below gap_min_m (the gap pass
    # tripled a site mesh to 24 M tets without it); the smear (solids.hull_*) is for those
    gmin = max(cfg['sizes']['gap_min_m'] * mult, smin)
    h_gap = np.maximum(h_gap, gmin)
    refine = have & (h_gap < 0.9 * E)
    # a surface sliver - a triangle whose height is under SLIVER_TRI x its longest edge (three
    # outline nodes nearly in line, a corner half a metre off a 20 m wall) - carries a needle
    # tet whose boundary face the converter cannot wind consistently; refine it to three cells
    # across its height so the outline is resolved instead
    height = 2.0 * Ar / np.maximum(Lm, 1e-12)
    sliver = height < SLIVER_TRI * Lm
    h_sl = np.where(sliver, np.maximum(3.0 * height, gmin), np.inf)
    h_gap = np.minimum(h_gap, h_sl)
    refine = refine | sliver
    SUMMARY['gap_pass'] = {'triangles': int(len(C)), 'with a facing surface': int(have.sum()),
                           'refined': int(refine.sum()), 'unresolvable': int(unres.sum()),
                           'surface slivers': int(sliver.sum())}
    if unres.any():
        pts = C[unres]; g = gap[unres]
        order = np.argsort(g)
        spots = [{'xyz': [round(float(v), 2) for v in pts[k]], 'gap_m': round(float(g[k]), 3)}
                 for k in order[:200]]
        SUMMARY['gap_unresolvable'] = spots
        for sp in spots[:12]:
            log('gap %.3f m at (%.1f, %.1f, %.1f) is narrower than %g x sizes.min: not meshable '
                'without slivers - smear or remove that geometry' % (sp['gap_m'], *sp['xyz'], ratio))
        if len(spots) > 12:
            log('... %d unresolvable gap spot(s) more; the full list is in the summary' % (len(spots) - 12))
    if sliver.any():
        log('gap pass: %d surface sliver triangle(s) (height < %g x longest edge) marked for '
            'refinement' % (int(sliver.sum()), SLIVER_TRI))
    if not refine.any():
        return 0, int(unres.sum())
    # bin the refinements on the grid: one Box per occupied cell with the smallest size in it
    gx, gz = GAP_GRID
    keys = np.stack([np.floor(C[refine, 0] / gx), np.floor(C[refine, 1] / gx),
                     np.floor(C[refine, 2] / gz)], axis=1).astype(np.int64)
    hv = np.maximum(h_gap[refine], smin)
    cells = {}
    for k, h in zip(map(tuple, keys), hv):
        cells[k] = min(cells.get(k, np.inf), h)
    F = gmsh.model.mesh.field
    boxes = []
    for (kx, ky, kz), h in sorted(cells.items(), key=lambda kv: kv[1])[:GAP_MAX_BOXES]:
        f = F.add('Box')
        F.setNumber(f, 'XMin', kx * gx - 1.0); F.setNumber(f, 'XMax', (kx + 1) * gx + 1.0)
        F.setNumber(f, 'YMin', ky * gx - 1.0); F.setNumber(f, 'YMax', (ky + 1) * gx + 1.0)
        F.setNumber(f, 'ZMin', kz * gz - 1.0); F.setNumber(f, 'ZMax', (kz + 1) * gz + 1.0)
        F.setNumber(f, 'VIn', float(h)); F.setNumber(f, 'VOut', cfg['sizes']['max'] * mult)
        F.setNumber(f, 'Thickness', max(4.0 * float(h), gx))
        boxes.append(f)
    fmin = F.add('Min')
    F.setNumbers(fmin, 'FieldsList', BASE_FIELDS + boxes)
    F.setAsBackgroundMesh(fmin)
    if len(cells) > GAP_MAX_BOXES:
        log('gap pass: %d grid cells wanted refinement, only the %d finest became boxes'
            % (len(cells), GAP_MAX_BOXES))
    return len(boxes), int(unres.sum())


# ------------------------------------------------------------ stage 9: mesh
def unify_coincident_curves():
    """Give every family of coincident curves (same bbox within 2 cm, same length within 1 cm)
    one and the same 1-D mesh: the finest member's nodes, copied onto the others with their own
    parametrisation. Returns (families, curves rewritten)."""
    curves = set()
    for d, sf in gmsh.model.getBoundary([(3, SUMMARY['fluid'])], oriented=False):
        for d1, c in gmsh.model.getBoundary([(2, sf)], oriented=False):
            curves.add(abs(c))
    fam = {}
    for c in curves:
        L = gmsh.model.occ.getMass(1, c)
        if L < 0.05:
            continue
        b = gmsh.model.getBoundingBox(1, c)
        fam.setdefault(tuple(np.round(np.array(b) / 0.02).astype(int))
                       + (int(round(L / 0.01)),), []).append(c)
    fams = [v for v in fam.values() if len(v) > 1]
    node_tags = gmsh.model.mesh.getNodes()[0]
    next_tag = int(node_tags.max()) + 1 if len(node_tags) else 1
    all_et = gmsh.model.mesh.getElements()[1]
    next_el = int(max(int(e.max()) for e in all_et)) + 1 if len(all_et) else 1
    rewritten = 0
    for members in fams:
        counts = {c: len(gmsh.model.mesh.getNodes(1, c)[0]) for c in members}
        src = max(members, key=lambda c: counts[c])
        stags, scoords, sparams = gmsh.model.mesh.getNodes(1, src)
        P = scoords.reshape(-1, 3)
        if len(P) == 0:
            continue
        for dst in members:
            if dst == src:
                continue
            # start and end point by geometry: getBoundary's signs are not reliable on these curves
            ends = list(set(abs(t) for d0, t in gmsh.model.getBoundary([(1, dst)], oriented=False)))
            if not ends:
                continue
            tb = gmsh.model.getParametrizationBounds(1, dst)
            x0 = np.array(gmsh.model.getValue(1, dst, [tb[0][0]]))
            x1 = np.array(gmsh.model.getValue(1, dst, [tb[1][0]]))
            pts = {q: np.array(gmsh.model.getValue(0, q, [])) for q in ends}
            p_start = min(pts, key=lambda q: np.linalg.norm(pts[q] - x0))
            p_end = min(pts, key=lambda q: np.linalg.norm(pts[q] - x1))
            n_start = int(gmsh.model.mesh.getNodes(0, p_start)[0][0])
            n_end = int(gmsh.model.mesh.getNodes(0, p_end)[0][0])
            params = gmsh.model.getParametrization(1, dst, P.ravel())
            order = np.argsort(params)
            tags = list(range(next_tag, next_tag + len(P)))
            next_tag += len(P)
            gmsh.model.mesh.clear([(1, dst)])
            gmsh.model.mesh.addNodes(1, dst, tags, P[order].ravel(), params[order])
            chain = [n_start] + tags + [n_end]
            lines = [v for i in range(len(chain) - 1) for v in (chain[i], chain[i + 1])]
            gmsh.model.mesh.addElementsByType(dst, 1,
                                              list(range(next_el, next_el + len(chain) - 1)), lines)
            next_el += len(chain) - 1
            rewritten += 1
    return len(fams), rewritten


def faces_with_overlapping_triangles():
    """Faces that own a triangle on an edge used by three or more triangles anywhere on the
    boundary (a fold or overlap), or on an edge used only once (a hole): the frontal mesher
    occasionally leaves such elements on sliver faces."""
    rows, owner = [], []
    for d, sf in gmsh.model.getBoundary([(3, SUMMARY['fluid'])], oriented=False):
        st, _, snodes = gmsh.model.mesh.getElements(2, sf)
        if 2 not in list(st):
            continue
        N = snodes[list(st).index(2)].reshape(-1, 3).astype(np.int64)
        rows.append(N)
        owner += [sf] * len(N)
    if not rows:
        return []
    N = np.vstack(rows)
    owner = np.array(owner)
    E = np.sort(np.vstack([N[:, [0, 1]], N[:, [1, 2]], N[:, [2, 0]]]), axis=1)
    _, inv, cnt = np.unique(E, axis=0, return_inverse=True, return_counts=True)
    badedge = (cnt[inv] != 2)
    return sorted(set(owner[np.tile(np.arange(len(N)), 3)[badedge]].tolist()))


def mesh_stage(cfg, work):
    stage(9, 'mesh')
    t = time.time()
    m = cfg['mesh']
    gmsh.option.setNumber('Mesh.Algorithm', m['algo2d'])
    # 3-D Delaunay by default: HXT (10) is faster, but on this class of geometry it hits its
    # unfinished Steiner-point path during boundary recovery and kills the whole process (no
    # exception, so nothing downstream can catch it)
    gmsh.option.setNumber('Mesh.Algorithm3D', m['algo3d'])
    gmsh.option.setNumber('Mesh.MaxNumThreads2D', m['threads'])
    gmsh.option.setNumber('Mesh.MaxNumThreads3D', m['threads'])
    gmsh.option.setNumber('Mesh.Optimize', 1)

    def surface_mesh():
        gmsh.model.mesh.generate(1)
        n_fam, n_rew = unify_coincident_curves()
        SUMMARY['coincident_curve_families'] = n_fam
        log('coincident curves: %d families, %d curves given their family\'s 1-D nodes'
            % (n_fam, n_rew))
        gmsh.model.mesh.generate(2)
        for algo in (5, 1):                                # Delaunay, then MeshAdapt
            bad = faces_with_overlapping_triangles()
            if not bad:
                break
            for sf in bad:
                gmsh.model.mesh.setAlgorithm(2, sf, algo)
            gmsh.model.mesh.clear([(2, sf) for sf in bad])
            gmsh.model.mesh.generate(2)
            log('remeshed %d face(s) with overlapping triangles using 2-D algorithm %d: %s'
                % (len(bad), algo, bad[:10]))
        bad = faces_with_overlapping_triangles()
        if bad:
            log('WARNING: %d face(s) still carry overlapping triangles: %s' % (len(bad), bad[:10]))
        SUMMARY['faces_remeshed'] = bad
        return sum(len(e) for e in gmsh.model.mesh.getElements(2)[1])

    ntri = surface_mesh()
    log('2D: %d triangles' % ntri)
    ratio = cfg['sizes']['gap_ratio']
    if ratio > 0:
        t_g = time.time()
        n_boxes, n_unres = gap_size_pass(cfg, ratio)
        gp = SUMMARY.get('gap_pass', {})
        log('gap pass (h <= g/%g): %d triangles, %d face a surface, %d need refinement -> %d box '
            'fields, %d unresolvable spots, %.0f s' % (ratio, gp.get('triangles', 0),
                                                       gp.get('with a facing surface', 0),
                                                       gp.get('refined', 0), n_boxes, n_unres,
                                                       time.time() - t_g))
        if n_boxes:
            gmsh.model.mesh.clear()
            ntri = surface_mesh()
            log('2D after the gap pass: %d triangles' % ntri)
    if PRE3D_MERGE > 0:
        # merging + rebuilding the surface BEFORE the 3-D pass makes gmsh drop every tet
        # ('No elements in volume'); the seams are merged after it, in the flat-tet stage
        bb = gmsh.model.getBoundingBox(-1, -1)
        lc = math.sqrt((bb[3] - bb[0]) ** 2 + (bb[4] - bb[1]) ** 2 + (bb[5] - bb[2]) ** 2)
        gmsh.option.setNumber('Geometry.Tolerance', PRE3D_MERGE / lc)
        n_before = len(gmsh.model.mesh.getNodes()[0])
        gmsh.model.mesh.removeDuplicateNodes()
        n_after = len(gmsh.model.mesh.getNodes()[0])
        if n_before != n_after:
            log('merged %d duplicate nodes within %.2f m before the 3-D pass'
                % (n_before - n_after, PRE3D_MERGE))
    tick('mesh2d', t)
    t = time.time()

    def tet_count():
        return sum(len(e) for e in gmsh.model.mesh.getElements(3)[1])

    SUMMARY['algorithm3d'] = {1: 'delaunay', 10: 'hxt'}.get(m['algo3d'], str(m['algo3d']))
    try:
        gmsh.model.mesh.generate(3)
    except Exception as e:
        log('generate(3) raised: %s' % e)
    if tet_count() == 0 and m['algo3d'] != 1:
        log('%s produced no tetrahedra (a self-intersecting surface mesh) -> retrying with '
            'Delaunay' % SUMMARY['algorithm3d'])
        gmsh.option.setNumber('Mesh.Algorithm3D', 1)
        SUMMARY['algorithm3d'] = 'delaunay'
        try:
            gmsh.model.mesh.generate(3)
        except Exception as e:
            log('generate(3) raised: %s' % e)
    ntet = tet_count()
    log('3D: %d tetrahedra' % ntet)
    # gmsh's own tet optimiser (edge/face swaps + smoothing) runs on this model, unlike Netgen;
    # the default pass (threshold 0.3) still leaves thin cells that break the pressure equation
    if ntet and m['optimize_passes']:
        gmsh.option.setNumber('Mesh.OptimizeThreshold', 0.5)
        t_opt = time.time()
        gmsh.model.mesh.optimize('', force=False, niter=m['optimize_passes'])
        log('gmsh optimiser: %d pass(es), threshold 0.5, %.0f s -> %d tetrahedra'
            % (m['optimize_passes'], time.time() - t_opt, tet_count()))
        ntet = tet_count()
    if ntet == 0:
        nt = SUMMARY.get('near_touching') or []
        if nt:
            log('near-touching solid vertices (1 mm..5 cm apart) recorded at the cut: %d - '
                "see summary 'near_touching' - the boundary recovery usually fails at one "
                'of them' % len(nt))
        surf = os.path.join(work, 'surface_only.msh')
        gmsh.option.setNumber('Mesh.MshFileVersion', 4.1)
        gmsh.option.setNumber('Mesh.Binary', 0)
        gmsh.option.setNumber('Mesh.SaveAll', 1)
        gmsh.write(surf)
        SUMMARY['error'] = '3D meshing failed (Delaunay); surface mesh saved for inspection'
        with open(os.path.join(work, 'failed_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(SUMMARY, f, indent=1, ensure_ascii=False)
        die('the 3-D mesh came out empty; the surface mesh is at %s (see the PLC error above '
            'for the point)' % surf, code=3)
    SUMMARY['triangles'] = ntri
    SUMMARY['tetrahedra'] = ntet
    tick('mesh3d', t)


# ------------------------------------------------- stage 10: post + write
def quality():
    tags = gmsh.model.mesh.getElements(3)[1][0]
    q = {}
    for measure in ('minSICN', 'gamma'):
        vals = sorted(gmsh.model.mesh.getElementQualities(tags, measure))
        n = len(vals)
        q[measure] = {'min': vals[0], 'p01': vals[n // 100], 'p05': vals[n // 20],
                      'p50': vals[n // 2], 'below_0.1': sum(1 for v in vals if v < 0.1),
                      'negative': sum(1 for v in vals if v < 0)}
        log('  %-7s min %.4f  p1 %.3f  p5 %.3f  p50 %.3f  <0.1: %d  <0: %d'
            % (measure, vals[0], q[measure]['p01'], q[measure]['p05'], q[measure]['p50'],
               q[measure]['below_0.1'], q[measure]['negative']))
    return q


def remove_flat_boundary_tets(fluid, flat_thr, seam_merge, sliver_edge, sliver_vol, thin_push,
                              sliver_rel, sliver_edge_rel, min_thickness=0.0, repair_rounds=3,
                              keep_boxes=()):
    """Delete zero-volume tets whose four nodes lie in one planar boundary by re-triangulating
    the boundary underneath them; merge seam node pairs, collapse sliver edges, push thin tets,
    nudge an interior node for the few that touch no boundary triangle.
    Returns (removed, found, rounds, notes)."""
    ntags, coords, _ = gmsh.model.mesh.getNodes()
    ntags = ntags.astype(np.int64)
    xyz = coords.reshape(-1, 3).copy()
    nmap = np.zeros(int(ntags.max()) + 1, dtype=np.int64)
    nmap[ntags] = np.arange(len(ntags))
    et, etags, enodes = gmsh.model.mesh.getElements(3, fluid)
    k = list(et).index(4)
    T = enodes[k].reshape(-1, 4).astype(np.int64)
    Ttag = etags[k].astype(np.int64)

    def volumes(idx):
        P = xyz[nmap[T[idx]]]
        return np.einsum('ij,ij->i', np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]),
                         P[:, 3] - P[:, 0]) / 6.0

    P = xyz[nmap[T]]
    vol = volumes(np.arange(len(T)))
    L = np.linalg.norm(P - P[:, [0]], axis=2).max(axis=1)
    flat = np.abs(vol) < flat_thr * L ** 3                 # a 2 m cell of 1 m^3 must be under 1e-6 m^3
    del P
    # boundary triangles as mutable per-surface lists; tri maps a sorted node triple to (surface, row)
    rows, tags, tri = {}, {}, {}
    max_tag = int(Ttag.max())
    for d, sf in gmsh.model.getBoundary([(3, fluid)], oriented=False):
        st, stags, snodes = gmsh.model.mesh.getElements(2, sf)
        if 2 not in list(st):
            continue
        kk = list(st).index(2)
        rows[sf] = snodes[kk].reshape(-1, 3).astype(np.int64).tolist()
        tags[sf] = stags[kk].astype(np.int64).tolist()
        max_tag = max(max_tag, max(tags[sf]))
        for i, tr in enumerate(rows[sf]):
            tri[tuple(sorted(tr))] = (sf, i)
    bnodes = set()
    for sf in rows:
        for tr in rows[sf]:
            bnodes.update(tr)

    def oriented(new, like):
        a_, b_, c_ = xyz[nmap[np.array(like)]]
        n_old = np.cross(b_ - a_, c_ - a_)
        a_, b_, c_ = xyz[nmap[np.array(new)]]
        return list(new) if np.dot(np.cross(b_ - a_, c_ - a_), n_old) >= 0 \
            else [new[0], new[2], new[1]]

    def put(sf, row, new, like):
        rows[sf][row] = oriented(new, like)
        tri[tuple(sorted(new))] = (sf, row)

    def add(sf, new, like):
        nonlocal max_tag
        max_tag += 1
        rows[sf].append(oriented(new, like))
        tags[sf].append(max_tag)
        tri[tuple(sorted(new))] = (sf, len(rows[sf]) - 1)

    removed = np.zeros(len(T), dtype=bool)
    straddled = set()
    rounds = 0
    while True:
        rounds += 1
        changed = 0
        for i in np.where(flat & ~removed)[0]:
            a, b, c, d = T[i].tolist()
            keys = [tuple(sorted(f)) for f in ((a, b, c), (a, b, d), (a, c, d), (b, c, d))]
            hits = [(kk, tri[kk]) for kk in keys if kk in tri]
            if not hits:
                continue                                    # not (yet) resting on the boundary
            hidden = [kk for kk in keys if kk not in [h[0] for h in hits]]
            # the planar configuration: is one node inside the triangle of the other three?
            Q = xyz[nmap[np.array([a, b, c, d])]]
            inside = None
            for m4 in range(4):
                o = [j for j in range(4) if j != m4]
                p0, p1, p2 = Q[o[0]], Q[o[1]], Q[o[2]]
                nrm = np.cross(p1 - p0, p2 - p0)
                tot = np.dot(nrm, nrm)
                if tot <= 0:
                    continue
                bary = [np.dot(np.cross(q1 - q0, Q[m4] - q0), nrm) / tot
                        for q0, q1 in ((p0, p1), (p1, p2), (p2, p0))]
                if min(bary) > 1e-6:
                    inside = (a, b, c, d)[m4]
                    break
            hit_keys = [h[0] for h in hits]
            if inside is None:                              # convex quad
                if len(hits) != 2:
                    continue
                move = 'swap'
            else:                                           # cap around node `inside`
                small = [kk for kk in keys if inside in kk]
                big = [kk for kk in keys if inside not in kk][0]
                if len(hits) == 3 and all(kk in small for kk in hit_keys):
                    move = 'cap3to1'
                elif len(hits) == 1 and hit_keys[0] == big:
                    move = 'cap1to3'
                else:
                    continue
            gs_of = set()
            for kk, (sf, row) in hits:
                gs = gmsh.model.getPhysicalGroupsForEntity(2, sf)
                if gs:
                    gs_of.add(int(gs[0]))
            if len(gs_of) > 1:
                straddled.add(int(i))
            like = rows[hits[0][1][0]][hits[0][1][1]]
            if move == 'swap':
                for (kk_old, (sf, row)), kk_new in zip(hits, hidden):
                    del tri[kk_old]
                    put(sf, row, kk_new, like)
            elif move == 'cap3to1':
                kk0, (sf0, row0) = hits[0]
                for kk_old, (sf, row) in hits:
                    del tri[kk_old]
                    rows[sf][row] = None
                rows[sf0][row0] = oriented(hidden[0], like)
                tri[hidden[0]] = (sf0, row0)
            else:                                           # cap1to3
                kk0, (sf0, row0) = hits[0]
                del tri[kk0]
                put(sf0, row0, hidden[0], like)
                add(sf0, hidden[1], like)
                add(sf0, hidden[2], like)
            removed[i] = True
            changed += 1
        if not changed:
            break
    notes = {'tets straddling two patches': len(straddled)}
    # leftovers that contain a pair of nodes closer than the seam merge (a roof or tile seam the
    # 2-D pass discretised twice): merge the pair everywhere, then drop the tets and triangles
    # that collapsed. Merging after the 3-D pass keeps gmsh's own volume mesh intact.
    alias = {}
    if seam_merge > 0:
        for i in np.where(flat & ~removed)[0]:
            pts = xyz[nmap[T[i]]]
            for u in range(4):
                for v in range(u + 1, 4):
                    if np.linalg.norm(pts[u] - pts[v]) < seam_merge:
                        hi, lo = max(int(T[i][u]), int(T[i][v])), min(int(T[i][u]), int(T[i][v]))
                        alias[hi] = lo
    if alias:
        def root(t):
            while t in alias:
                t = alias[t]
            return t
        amap = np.arange(int(T.max()) + 1, dtype=np.int64)
        for hi in alias:
            amap[hi] = root(hi)
        T = amap[T]
        collapsed = ((T[:, 0] == T[:, 1]) | (T[:, 0] == T[:, 2]) | (T[:, 0] == T[:, 3]) |
                     (T[:, 1] == T[:, 2]) | (T[:, 1] == T[:, 3]) | (T[:, 2] == T[:, 3]))
        removed |= collapsed
        n_tri_dropped = 0
        for sf in rows:
            for r_i, r in enumerate(rows[sf]):
                if r is None:
                    continue
                m = [int(amap[v]) if v < len(amap) else v for v in r]
                if len(set(m)) < 3:
                    rows[sf][r_i] = None
                    n_tri_dropped += 1
                else:
                    rows[sf][r_i] = m
        # tri keys are stale now; rebuild them for the nudge step's bnodes
        bnodes = set()
        for sf in rows:
            for r in rows[sf]:
                if r is not None:
                    bnodes.update(r)
        vol = volumes(np.arange(len(T)))
        flat = np.abs(vol) < flat_thr * L ** 3
        notes['seam node pairs merged'] = len(alias)
        notes['tets collapsed by the merge'] = int(collapsed.sum())
        notes['triangles collapsed by the merge'] = n_tri_dropped
    # ---- sliver wedges: collapse edges shorter than the limit inside thin tets. The absolute
    # keys are for meshes whose smallest cells are metres; with sliver_rel > 0 the thresholds
    # follow the locally refined cell size instead: a tet is thin when |V| < sliver_rel *
    # e_mean^3 (a regular tet has |V| = 0.11785 e^3, so 0.01 selects tets flatter than about
    # 8 % of regular) and an edge is collapsible below sliver_edge_rel * e_mean of that tet.
    n_collapsed_edges = n_collapse_rounds = n_reverted = 0
    if sliver_edge > 0 or sliver_rel > 0:

        def edge_mean():
            Pt = xyz[nmap[T]]
            em = np.stack([np.linalg.norm(Pt[:, a] - Pt[:, b], axis=1)
                           for a in range(4) for b in range(a + 1, 4)], axis=1).mean(axis=1)
            del Pt
            return em

        tri_of_node = {}
        for sf in rows:
            for r_i, r in enumerate(rows[sf]):
                if r is not None:
                    for v in r:
                        tri_of_node.setdefault(v, []).append((sf, r_i))
        for rnd in range(SLIVER_ROUNDS):
            order_all = np.argsort(T.ravel())
            starts_all = np.searchsorted(T.ravel()[order_all], np.arange(int(T.max()) + 2))

            def tets_around(node):
                a = order_all[starts_all[node]:starts_all[node + 1]] // 4
                return a[~removed[a]]

            vol = volumes(np.arange(len(T)))
            if sliver_rel > 0:
                e_mean = edge_mean()                          # per tet, recomputed per round
                edge_lim = sliver_edge_rel * e_mean
                thin = np.where((~removed) & (np.abs(vol) < sliver_rel * e_mean ** 3))[0]
            else:
                edge_lim = np.full(len(T), sliver_edge)
                thin = np.where((~removed) & (np.abs(vol) < sliver_vol))[0]
            merged_now = 0
            touched = set()
            for i in thin:
                if removed[i]:
                    continue
                Q = T[i]
                pts = xyz[nmap[Q]]
                best = None
                for u in range(4):
                    for v in range(u + 1, 4):
                        d = np.linalg.norm(pts[u] - pts[v])
                        if d < edge_lim[i] and (best is None or d < best[0]):
                            best = (d, int(Q[u]), int(Q[v]))
                if best is None:
                    continue
                d, a, b = best
                if a == b or a in touched or b in touched:
                    continue
                # keep: the boundary node if only one is on the boundary, else the lower tag
                if (a in bnodes) != (b in bnodes):
                    keep, drop = (a, b) if a in bnodes else (b, a)
                else:
                    keep, drop = (min(a, b), max(a, b))
                old_keep = xyz[nmap[keep]].copy()
                around = np.unique(np.concatenate([tets_around(keep), tets_around(drop)]))
                Tsave = T[around].copy()
                T[around] = np.where(T[around] == drop, keep, T[around])
                degenerate = np.array([len(set(T[j].tolist())) < 4 for j in around])
                live = around[~degenerate]
                # try the merged node at three places: its own spot, the midpoint, the dropped
                # node's spot (a boundary node stays put when the other one is interior)
                candidates = [old_keep] if (keep in bnodes) != (drop in bnodes) else \
                    [old_keep, 0.5 * (old_keep + xyz[nmap[drop]]), xyz[nmap[drop]].copy()]
                ok = False
                for cpos in candidates:
                    xyz[nmap[keep]] = cpos
                    if not len(live) or (volumes(live) > 1e-9 * L[live] ** 3).all():
                        ok = True
                        break
                if not ok:
                    T[around] = Tsave
                    xyz[nmap[keep]] = old_keep
                    n_reverted += 1
                    continue
                removed[around[degenerate]] = True
                for sf, r_i in tri_of_node.get(drop, []):
                    r = rows[sf][r_i]
                    if r is None:
                        continue
                    mm = [keep if v == drop else v for v in r]
                    rows[sf][r_i] = None if len(set(mm)) < 3 else mm
                    if rows[sf][r_i] is not None:
                        tri_of_node.setdefault(keep, []).append((sf, r_i))
                tri_of_node[drop] = []
                if drop in bnodes:
                    bnodes.discard(drop)
                    bnodes.add(keep)
                touched.update((keep, drop))
                merged_now += 1
            n_collapsed_edges += merged_now
            n_collapse_rounds = rnd + 1
            if merged_now == 0:
                break
        vol = volumes(np.arange(len(T)))
        flat = np.abs(vol) < flat_thr * L ** 3
        notes['sliver edges collapsed'] = n_collapsed_edges
        notes['collapse rounds'] = n_collapse_rounds
        notes['collapses reverted (would invert a tet)'] = n_reverted
        if sliver_rel > 0:
            notes['thin tets (|V| < %g e^3) left' % sliver_rel] = \
                int(((~removed) & (np.abs(vol) < sliver_rel * edge_mean() ** 3)).sum())
        else:
            notes['tets with V < %g m3 left' % sliver_vol] = \
                int(((~removed) & (np.abs(vol) < sliver_vol)).sum())
    # ---- twins: merging nodes (the seam merge, the edge collapses) can make two tets that shared
    # three nodes coincide in all four - both kept, a face is then shared by three tets and the
    # converter refuses the mesh. Keep the first of every set of identical tets.
    live_idx = np.where(~removed)[0]
    if len(live_idx):
        srt = np.sort(T[live_idx], axis=1)
        _, first = np.unique(srt, axis=0, return_index=True)
        twins = np.ones(len(live_idx), dtype=bool)
        twins[first] = False
        if twins.any():
            removed[live_idx[twins]] = True
        notes['twin tets removed'] = int(twins.sum())
        # and a face shared by more than two tets is reported (it should not survive the above)
        Tk = T[~removed]
        base_k = int(T.max()) + 1
        Fk = np.concatenate([Tk[:, [0, 1, 2]], Tk[:, [0, 1, 3]], Tk[:, [0, 2, 3]], Tk[:, [1, 2, 3]]])
        Fk.sort(axis=1)
        fk = Fk[:, 0].astype(np.int64) * base_k * base_k + Fk[:, 1].astype(np.int64) * base_k + Fk[:, 2]
        del Fk
        _, cnt_k = np.unique(fk, return_counts=True)
        del fk
        notes['faces shared by more than two tets'] = int((cnt_k > 2).sum())
        vol = volumes(np.arange(len(T)))
        flat = np.abs(vol) < flat_thr * L ** 3

    # ---- thin tets with no short edge (a wedge between a hull facet and the sea, a roof and a
    # slab): push one node away from the tet's largest face by up to thin_push metres, guarded
    # by every tet around that node
    n_pushed = n_push_failed = 0
    if thin_push > 0:
        for rnd in range(PUSH_ROUNDS):
            order_all = np.argsort(T.ravel())
            starts_all = np.searchsorted(T.ravel()[order_all], np.arange(int(T.max()) + 2))
            vol = volumes(np.arange(len(T)))
            P = xyz[nmap[T]]
            e = np.stack([np.linalg.norm(P[:, a] - P[:, b], axis=1)
                          for a in range(4) for b in range(a + 1, 4)], axis=1).mean(axis=1)
            thin_q = 6.0 * math.sqrt(2.0) * np.abs(vol) / np.maximum(e, 1e-12) ** 3   # 1 = regular
            cand = np.where((~removed) & (thin_q < THIN_SICN))[0]
            if not len(cand):
                break
            pushed_now = 0
            for i in cand:
                Q = T[i]
                pts = xyz[nmap[Q]]
                best = None
                for m4 in range(4):
                    o = [j for j in range(4) if j != m4]
                    nrm = np.cross(pts[o[1]] - pts[o[0]], pts[o[2]] - pts[o[0]])
                    area = 0.5 * np.linalg.norm(nrm)
                    if best is None or area > best[0]:
                        best = (area, m4, nrm / (2 * area + 1e-300), o[0])
                area, m4, n, o0 = best
                node = int(Q[m4])
                sign = np.sign(np.dot(pts[m4] - pts[o0], n)) or 1.0
                a = order_all[starts_all[node]:starts_all[node + 1]] // 4
                a = a[~removed[a]]
                old = xyz[nmap[node]].copy()
                done = False
                for step in (thin_push, 0.5 * thin_push):
                    xyz[nmap[node]] = old + sign * step * n
                    v_a = volumes(a)
                    if (v_a > 1e-9 * L[a] ** 3).all() and \
                            abs(volumes(np.array([i]))[0]) > abs(vol[i]):
                        done = True
                        break
                    xyz[nmap[node]] = old
                if done:
                    pushed_now += 1
                else:
                    n_push_failed += 1
            n_pushed += pushed_now
            if pushed_now == 0:
                break
        vol = volumes(np.arange(len(T)))
        flat = np.abs(vol) < flat_thr * L ** 3
        notes['thin tets pushed'] = n_pushed
        notes['thin tets not pushable'] = n_push_failed
    # leftovers: nudge one node off the plane, guarded by every tet around that node
    left = np.where(flat & ~removed)[0]
    if len(left):
        order = np.argsort(T.ravel())
        starts = np.searchsorted(T.ravel()[order], np.arange(int(T.max()) + 2))
        nudged = 0
        for i in left:
            P = xyz[nmap[T[i]]]
            n = np.cross(P[1] - P[0], P[2] - P[0])
            n = n / (np.linalg.norm(n) + 1e-300)
            done = False
            # interior nodes first; failing that a boundary node (a 1 mm bump on a wall, checked
            # against every tet around it, is harmless for the flow and removes a zero-volume cell)
            for node in sorted(T[i].tolist(), key=lambda v: v in bnodes):
                around = order[starts[node]:starts[node + 1]] // 4
                around = around[~removed[around]]
                old = xyz[nmap[node]].copy()
                for sign in (1.0, -1.0):
                    xyz[nmap[node]] = old + sign * 1e-3 * L[i] * n
                    if (volumes(around) > 1e-9 * L[i] ** 3).all():
                        done = True
                        break
                    xyz[nmap[node]] = old
                if done:
                    break
            nudged += done
        notes['nudged a node off the plane'] = nudged
        notes['still flat'] = int(len(left) - nudged)

    # ---- the thickness gate, checked and repaired in rounds. tau = 3V / A_max^1.5 is the
    # cell's height over its largest face divided by that face's size (§92.3 of SPEC-LIT; a
    # regular tet has tau = 1.24, the ammonia slivers 0.01). Every tet under min_thickness gets
    # the node opposite its largest face pushed along the face normal by the height the gate
    # asks for, guarded by every tet around that node; nodes inside a pool's refinement box
    # are never moved. What is left after the rounds is listed with its position.
    def thickness(P):
        v = np.abs(np.einsum('ij,ij->i', P[:, 1] - P[:, 0],
                             np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 0]))) / 6.0
        areas = np.stack([0.5 * np.linalg.norm(np.cross(P[:, b] - P[:, a], P[:, c] - P[:, a]), axis=1)
                          for a, b, c in ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))], axis=1)
        amax = areas.max(axis=1)
        return v, areas, amax, 3.0 * v / np.maximum(amax, 1e-300) ** 1.5

    def frozen(node):
        x, y = xyz[nmap[node]][:2]
        return any(abs(x - kx) < kh and abs(y - ky) < kh for kx, ky, kh in keep_boxes)

    n_thick_pushed = n_thick_failed = 0
    if min_thickness > 0:
        for rnd in range(repair_rounds):
            order_all = np.argsort(T.ravel())
            starts_all = np.searchsorted(T.ravel()[order_all], np.arange(int(T.max()) + 2))
            P = xyz[nmap[T]]
            v, areas, amax, tau = thickness(P)
            cand = np.where((~removed) & (tau < min_thickness))[0]
            if not len(cand):
                break
            pushed_now = 0
            for i in cand:
                Q = T[i]
                m4 = int(np.argmax(areas[i]))
                o = [j for j in range(4) if j != m4]
                pts = xyz[nmap[Q]]
                nrm = np.cross(pts[o[1]] - pts[o[0]], pts[o[2]] - pts[o[0]])
                nrm = nrm / (np.linalg.norm(nrm) + 1e-300)
                node = int(Q[m4])
                if frozen(node):
                    n_thick_failed += 1
                    continue
                sign = np.sign(np.dot(pts[m4] - pts[o[0]], nrm)) or 1.0
                h_now = 3.0 * v[i] / max(amax[i], 1e-300)
                h_req = 1.1 * min_thickness * math.sqrt(amax[i])
                step = h_req - h_now
                if step <= 0:
                    continue
                a = order_all[starts_all[node]:starts_all[node + 1]] // 4
                a = a[~removed[a]]
                old = xyz[nmap[node]].copy()
                done = False
                for frac in (1.0, 0.5, 0.25):
                    xyz[nmap[node]] = old + sign * frac * step * nrm
                    v_a = volumes(a)
                    if (v_a > 1e-9 * L[a] ** 3).all() and \
                            abs(volumes(np.array([i]))[0]) > abs(v[i]):
                        done = True
                        break
                    xyz[nmap[node]] = old
                if done:
                    pushed_now += 1
                else:
                    n_thick_failed += 1
            n_thick_pushed += pushed_now
            if pushed_now == 0:
                break
        vol = volumes(np.arange(len(T)))
        flat = np.abs(vol) < flat_thr * L ** 3
    P = xyz[nmap[T]]
    v, areas, amax, tau = thickness(P)
    live = ~removed
    tl = tau[live]
    notes['thickness: min'] = round(float(tl.min()), 5) if len(tl) else None
    notes['thickness: cells below 0.05'] = int((tl < 0.05).sum())
    notes['thickness: cells below 0.02'] = int((tl < 0.02).sum())
    if min_thickness > 0:
        notes['thickness: cells below the gate %g' % min_thickness] = int((tl < min_thickness).sum())
        notes['thickness: nodes pushed'] = n_thick_pushed
        notes['thickness: pushes refused'] = n_thick_failed
    worst = np.where(live)[0][np.argsort(tl)[:10]] if len(tl) else []
    SUMMARY['thickness_gate'] = [{'tet': int(i), 'tau': round(float(tau[i]), 5),
                                  'xyz': [round(float(c), 2) for c in P[i].mean(axis=0)]}
                                 for i in worst]
    # reconcile the triangles with the surviving tets before the rebuild: a patch triangle
    # must be the face of exactly one tet. The edits above leave triangles that belong to no
    # tet any more (orphans), that two tets share (an interior face wearing a patch: the
    # converter turns it into a boundary inside the fluid and seals a pocket) or that appear
    # twice (a cell with five faces that does not close) - all three break the solver's mesh
    keep = ~removed
    Tk = T[keep]
    base = int(max(int(T.max()), max((max(r) for sf in rows for r in rows[sf] if r), default=0))) + 1
    Fk = np.concatenate([Tk[:, [0, 1, 2]], Tk[:, [0, 1, 3]], Tk[:, [0, 2, 3]], Tk[:, [1, 2, 3]]])
    Fk.sort(axis=1)
    fkey = Fk[:, 0].astype(np.int64) * base * base + Fk[:, 1].astype(np.int64) * base + Fk[:, 2]
    del Fk
    ukey, ucnt = np.unique(fkey, return_counts=True)
    del fkey
    bkey = ukey[ucnt == 1]
    tri_ref = [(sf, i) for sf in rows for i, r in enumerate(rows[sf]) if r is not None]
    if tri_ref:
        Tri = np.array([sorted(rows[sf][i]) for sf, i in tri_ref], dtype=np.int64)
        tkey = Tri[:, 0] * base * base + Tri[:, 1] * base + Tri[:, 2]
        exists = np.isin(tkey, ukey)
        on_boundary = np.isin(tkey, bkey)
        first = np.zeros(len(tkey), dtype=bool)
        first[np.unique(tkey, return_index=True)[1]] = True
        good = on_boundary & first
        n_orphan = int((~exists).sum())
        n_interior = int((exists & ~on_boundary).sum())
        n_dup = int((on_boundary & ~first).sum())
        for (sf, i), ok_ in zip(tri_ref, good):
            if not ok_:
                rows[sf][i] = None
        n_missing = int(len(bkey) - np.unique(tkey[good]).size)
        notes['triangles dropped: orphan'] = n_orphan
        notes['triangles dropped: interior'] = n_interior
        notes['triangles dropped: duplicate'] = n_dup
        notes['boundary faces without a triangle'] = n_missing
        if n_orphan or n_interior or n_dup or n_missing:
            log('triangles reconciled with the tets: dropped %d orphan, %d interior, %d duplicate; '
                '%d boundary tet face(s) carry no patch triangle' % (n_orphan, n_interior, n_dup, n_missing))
    # rebuild the mesh: all nodes on the volume, the surviving tets, the edited triangles
    gmsh.model.mesh.clear()
    gmsh.model.mesh.addNodes(3, fluid, ntags, xyz.ravel())
    gmsh.model.mesh.addElementsByType(fluid, 4, Ttag[keep], T[keep].ravel())
    for sf in rows:
        ok = [i for i, r in enumerate(rows[sf]) if r is not None]
        if ok:
            gmsh.model.mesh.addElementsByType(sf, 2, [tags[sf][i] for i in ok],
                                              [v for i in ok for v in rows[sf][i]])
    return int(removed.sum()), int(flat.sum()), rounds, notes


def post_and_write_stage(cfg, out_dir, tag):
    stage(10, 'post + write')
    t = time.time()
    fluid = SUMMARY['fluid']
    p = cfg['post']

    log('quality before the flat-tet stage:')
    q = quality()
    flat = sum(1 for v in gmsh.model.mesh.getElementQualities(
        gmsh.model.mesh.getElements(3)[1][0], 'minSICN') if v < 0.01)
    SUMMARY['flat_tets'] = flat
    SUMMARY['quality'] = q
    tick('quality', t)

    t = time.time()
    if p['flat_tets']:
        keep_boxes = [(x, y, max(POINT_BOX[0], cfg['pool_specs'][name]['r'] + 15.0))
                      for name, (x, y) in cfg['points'].items()]
        n_removed, n_flat, n_rounds, skipped = remove_flat_boundary_tets(
            fluid, p['flat_threshold'], p['seam_merge_m'], p['sliver_edge_m'],
            p['sliver_vol_m3'], p['thin_push_m'], p['sliver_rel'], p['sliver_edge_rel'],
            p['min_thickness'], p['repair_rounds'], keep_boxes)
        gate = SUMMARY.get('thickness_gate') or []
        if gate:
            log('thickness gate: worst tau %.4f at (%.1f, %.1f, %.1f); %s' % (
                gate[0]['tau'], *gate[0]['xyz'],
                ', '.join('%s %s' % (k.replace('thickness: ', ''), v) for k, v in skipped.items()
                          if k.startswith('thickness:'))))
        log('flat tets: %d found (SICN<0.01), %d tets removed by re-triangulating the boundary '
            'under them (%d rounds); %s' % (n_flat, n_removed, n_rounds, skipped))
        SUMMARY['flat_tets_found'] = n_flat
        SUMMARY['flat_tets_removed'] = n_removed
        SUMMARY['flat_tets_notes'] = skipped
        log('quality after the removal:')
        q = quality()
        SUMMARY['quality_after_flat_removal'] = q
    else:
        log('flat-tet stage disabled (config.post.flat_tets = false)')
    SUMMARY['tetrahedra'] = int(len(gmsh.model.mesh.getElements(3)[1][0]))
    SUMMARY['triangles'] = sum(len(e) for e in gmsh.model.mesh.getElements(2)[1])
    tick('flat', t)

    t = time.time()
    gmsh.option.setNumber('Mesh.MshFileVersion', 4.1)
    gmsh.option.setNumber('Mesh.Binary', 0)
    gmsh.option.setNumber('Mesh.SaveAll', 0)
    suffix = '_' + tag if tag else ''
    msh = os.path.join(out_dir, '%s%s.msh' % (cfg['name'], suffix))
    vtk = os.path.join(out_dir, '%s%s.vtk' % (cfg['name'], suffix))
    gmsh.write(msh)
    gmsh.option.setNumber('Mesh.Binary', 1)   # the .vtk is for viewing only; ASCII is gigabytes
    gmsh.write(vtk)
    gmsh.option.setNumber('Mesh.Binary', 0)
    SUMMARY['files'] = {os.path.basename(pth): os.path.getsize(pth) for pth in (msh, vtk)}
    tick('write', t)
    SUMMARY['total_s'] = round(time.time() - T0, 1)
    SUMMARY['config'] = cfg
    summary_path = os.path.join(out_dir, '%s%s_summary.json' % (cfg['name'], suffix))
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(SUMMARY, f, indent=1, ensure_ascii=False)
    log('wrote %s (%.0f MB), %s (%.0f MB) and %s'
        % (msh, os.path.getsize(msh) / 1e6, vtk, os.path.getsize(vtk) / 1e6,
           os.path.basename(summary_path)))


# -------------------------------------------------------------------- main
def parse_args(argv):
    ap = argparse.ArgumentParser(
        prog='step_mesh.py',
        description='A configuration-driven STEP -> Gmsh tetrahedral mesh tool '
                    '(see tools/mesh/README.md for the schema).')
    ap.add_argument('config', help='the JSON config (tools/mesh/examples/nh3_site.json)')
    ap.add_argument('--from-checkpoint', action='store_true',
                    help='start from <out_dir>/work/<name>_pools.brep (cut + pools done)')
    ap.add_argument('--stop-after-checkpoint', action='store_true',
                    help='stop after the checkpoint, before the trim and the mesh')
    ap.add_argument('--tag', default='', metavar='NAME',
                    help='suffix the outputs: <name>_<NAME>.msh, .vtk, _summary.json')
    ap.add_argument('--dry-run', action='store_true',
                    help='stop after the import and the cut; print volumes, masses, '
                         'surface counts and ground heights')
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    _install_tee()
    out_dir, work = cfg['out_dir'], os.path.join(cfg['out_dir'], 'work')
    os.makedirs(work, exist_ok=True)
    SUMMARY.update({'gmsh': gmsh.__version__, 'timings_s': {}, 'points': {}, 'groups': {},
                    'pools': {}, 'pockets': []})
    if args.from_checkpoint and args.stop_after_checkpoint:
        die('--from-checkpoint and --stop-after-checkpoint together would do nothing')
    gmsh.initialize()
    try:
        import_stage(cfg, args, work)
        if not args.from_checkpoint:
            cut_stage(cfg, args, work)
            ground_stage(cfg, args, work)
            t = time.time()
            SUMMARY['groups'] = {}
            log('classification before pools:')
            report_groups(cfg, classify(cfg))
            tick('classify', t)
            pool_stage(cfg, args, work)
        else:
            # the pool outcomes ride in with the checkpoint; classify later, after the trim
            SUMMARY['pools'] = SUMMARY.get('pools', {})
        checkpoint_stage(cfg, args, work)
        trim_stage(cfg, work)
        groups_and_fields_stage(cfg)
        mesh_stage(cfg, work)
        post_and_write_stage(cfg, out_dir, args.tag)
    finally:
        try:
            gmsh.finalize()
        except Exception:
            pass
    log('DONE')


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    main()
