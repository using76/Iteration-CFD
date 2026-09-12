#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""
turek_hron.py - the Turek-Hron FSI benchmark as a two-region hex mesh.

S. Turek & J. Hron, LNCSE 53, Springer (2006) 371-385, DOI
10.1007/3-540-34596-5_15: the channel 2.5 x 0.41, the cylinder r = 0.05 at
(0.2, 0.2), the elastic flap [0.2, 0.6] x [0.19, 0.21] minus the disk (the
cylinder is subtracted from the flap BEFORE the fragment, so no lens remains).
A 2-D fragment, extruded one cell with recombine - hex only, dz thick, forces
per unit depth are F / dz.

    python tools/mesh/examples/turek_hron.py [--level 1|2|3] [--out DIR] [--dz 0.02]

Level 1 lands in cases/turekHron/mesh (the two case skeletons point there);
levels 2 and 3 default to mesh_L2 and mesh_L3. The layout is written through
M4 (regions_from_msh.py, then regions_check.py) and the sidecar
<out>/turek_hron.json carries the counts, the pairing worsts and the
benchmark constants.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import gmsh

import recipe

L, H = 2.5, 0.41                              # the channel, m
CY, R = (0.2, 0.2), 0.05                      # cylinder centre and radius
FX0, FX1, FY0, FY1 = 0.2, 0.6, 0.19, 0.21     # the flap rectangle
OUT = {1: ('cases', 'turekHron', 'mesh'), 2: ('cases', 'turekHron', 'mesh_L2'),
       3: ('cases', 'turekHron', 'mesh_L3')}


def build(level, dz):
    """The 2-D fragment (bigger face = fluid, smaller = flap), extruded one cell."""
    occ = gmsh.model.occ
    h_near, h_far = 0.01 / 2 ** level, 0.08 / 2 ** level
    ch = occ.addRectangle(0, 0, 0, L, H)
    dk = occ.addDisk(CY[0], CY[1], 0, R, R)
    fl = occ.addRectangle(FX0, FY0, 0, FX1 - FX0, FY1 - FY0)
    fl = occ.cut([(2, fl)], [(2, dk)], removeObject=True, removeTool=False)[0][0][1]
    ch = occ.cut([(2, ch)], [(2, dk)], removeObject=True, removeTool=True)[0][0][1]
    out, _ = occ.fragment([(2, ch)], [(2, fl)])
    occ.synchronize()
    area = {t: occ.getMass(2, t) for d, t in out}
    fluid_face, flap_face = max(area, key=area.get), min(area, key=area.get)
    size_field(flap_face, fluid_face, h_near, h_far)
    ext = occ.extrude([(2, fluid_face), (2, flap_face)], 0, 0, dz,
                      numElements=[1], recombine=True)
    occ.synchronize()
    mass = {t: occ.getMass(3, t) for d, t in ext if d == 3}
    fluid, flap = max(mass, key=mass.get), min(mass, key=mass.get)
    return {'fluid': fluid, 'flap': flap, 'groups': {'fluid': fluid, 'flap': flap},
            'h_near': h_near, 'h_far': h_far}


def size_field(flap_face, fluid_face, h_near, h_far):
    """Distance/Threshold about the flap and the wetted arc; the extrusion copies it."""
    near = [int(t) for d, t in gmsh.model.getBoundary([(2, flap_face)], oriented=False)]
    for d, t in gmsh.model.getBoundary([(2, fluid_face)], oriented=False):
        x, y, _ = recipe.centre_of(1, t)
        if math.hypot(x - CY[0], y - CY[1]) < 0.06:
            near.append(int(t))
    dist = gmsh.model.mesh.field.add('Distance')
    gmsh.model.mesh.field.setNumbers(dist, 'CurvesList', near)
    gmsh.model.mesh.field.setNumber(dist, 'Sampling', 200)
    thr = gmsh.model.mesh.field.add('Threshold')
    gmsh.model.mesh.field.setNumber(thr, 'InField', dist)
    gmsh.model.mesh.field.setNumber(thr, 'SizeMin', h_near)
    gmsh.model.mesh.field.setNumber(thr, 'SizeMax', h_far)
    gmsh.model.mesh.field.setNumber(thr, 'DistMin', 0.05)
    gmsh.model.mesh.field.setNumber(thr, 'DistMax', 0.6)
    gmsh.model.mesh.field.setAsBackgroundMesh(thr)
    for key, val in (('Mesh.MeshSizeExtendFromBoundary', 0), ('Mesh.MeshSizeFromPoints', 0),
                     ('Mesh.MeshSizeFromCurvature', 0), ('Mesh.Algorithm', 8),
                     ('Mesh.RecombineAll', 1), ('Mesh.RecombinationAlgorithm', 3)):
        gmsh.option.setNumber(key, val)


def classify(fluid, flap, dz):
    """Every boundary surface of the two volumes into its named group (tol 1e-6)."""
    on_fluid, on_flap = set(recipe.surfaces_of(fluid)), set(recipe.surfaces_of(flap))
    groups = {k: [] for k in ('fluid_to_flap', 'empty_back', 'empty_front', 'inlet',
                              'outlet', 'wall_bottom', 'wall_top', 'cylinder', 'flap_fixed')}
    for s in sorted(on_fluid | on_flap):
        if s in on_fluid and s in on_flap:
            groups['fluid_to_flap'].append(s)
        elif recipe.lies_in_plane(s, 2, 0.0):
            groups['empty_back'].append(s)
        elif recipe.lies_in_plane(s, 2, dz):
            groups['empty_front'].append(s)
        elif recipe.lies_in_plane(s, 0, 0.0):
            groups['inlet'].append(s)
        elif recipe.lies_in_plane(s, 0, L):
            groups['outlet'].append(s)
        elif recipe.lies_in_plane(s, 1, 0.0):
            groups['wall_bottom'].append(s)
        elif recipe.lies_in_plane(s, 1, H):
            groups['wall_top'].append(s)
        else:
            x, y, _ = recipe.centre_of(2, s)
            if math.hypot(x - CY[0], y - CY[1]) >= 0.06:
                recipe.die('surface %d (centre %.4f, %.4f) is in no group' % (s, x, y))
            groups['cylinder' if s in on_fluid else 'flap_fixed'].append(s)
    return groups


def main(argv=None):
    ap = argparse.ArgumentParser(
        description='The Turek-Hron FSI benchmark as a two-region hex mesh through M4.')
    ap.add_argument('--level', type=int, default=1, choices=(1, 2, 3))
    ap.add_argument('--out', default=None, help='default cases/turekHron/mesh[_L2|_L3], repo-root relative')
    ap.add_argument('--dz', type=float, default=0.02, help='the one-cell depth, m')
    args = ap.parse_args(argv)
    out = os.path.abspath(args.out or os.path.join(recipe.repo_root(), *OUT[args.level]))
    recipe.fresh_dir(out)
    t0 = time.time()
    recipe.gmsh_begin()
    try:
        b = build(args.level, args.dz)
        groups = classify(b['fluid'], b['flap'], args.dz)
        recipe.add_groups(b['groups'], groups)
        counts = recipe.generate_and_count(b['groups'])
        recipe.write_msh(os.path.join(out, 'turek_hron.msh'))
        version = gmsh.__version__
    finally:
        recipe.gmsh_end()
    t_gmsh = time.time() - t0
    t1 = time.time()
    regions = recipe.make_layout(os.path.join(out, 'turek_hron.msh'), out,
                                 'fluid', {'flap': 'flap'})
    t_m4 = time.time() - t1
    side = recipe.summarize(regions, out, counts)
    sidecar = {'recipe': 'turek_hron', 'level': args.level, 'gmsh': version, 'units': 'm',
               'timings_s': {'gmsh': round(t_gmsh, 1), 'regions': round(t_m4, 1)},
               'dz': args.dz, 'h_near': b['h_near'], 'h_far': b['h_far'],
               'flap_length_m': FX1 - (CY[0] + math.sqrt(R * R - (FY1 - CY[1]) ** 2)),
               'point_A': [0.6, 0.2], 'flap_cells_across': round((FY1 - FY0) / b['h_near']),
               'cylinder_faces': side['regions']['fluid']['patches']['cylinder'][1],
               'interface_faces': side['interfaces'][0]['faces'] if side['interfaces'] else 0,
               'regions': side['regions'], 'interfaces': side['interfaces']}
    recipe.write_json(os.path.join(out, 'turek_hron.json'), sidecar)
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    sys.exit(main())
