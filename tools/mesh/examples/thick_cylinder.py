#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""
thick_cylinder.py - the thick-walled cylinder of an axisymmetric thermal gate,
as one transfinite hex region: a quarter annulus r 0.05 -> 0.10 m, z 0..0.02,
in the first quadrant (so the flat sides are symmetry_x0 / symmetry_y0).

    python tools/mesh/examples/thick_cylinder.py [--level 1|2|3] [--out DIR]

Levels (nr, nt, nz) = (8, 12, 2), (16, 24, 4), (32, 48, 8) - cells =
nr*nt*nz = 192 at level 1. Names only, in the sidecar: Timoshenko & Goodier,
Theory of Elasticity, 3rd ed. (1970); Boley & Weiner, Theory of Thermal
Stresses, ch. 9. Default --out cases/thick_cylinder_L<level> (git-ignored).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import gmsh

import recipe

R_IN, R_OUT, ZL = 0.05, 0.10, 0.02
LEVELS = ((8, 12, 2), (16, 24, 4), (32, 48, 8))
PATCHES = ('inner', 'outer', 'zmin', 'zmax', 'symmetry_x0', 'symmetry_y0')


def build(level):
    """A transfinite quarter annulus, extruded nz layers: (volume, (nr, nt, nz))."""
    occ = gmsh.model.occ
    nr, nt, nz = LEVELS[level - 1]
    ctr = occ.addPoint(0, 0, 0)
    p_in0, p_out0 = occ.addPoint(R_IN, 0, 0), occ.addPoint(R_OUT, 0, 0)
    p_out90, p_in90 = occ.addPoint(0, R_OUT, 0), occ.addPoint(0, R_IN, 0)
    l_x = occ.addLine(p_in0, p_out0)               # y = 0,  r_in -> r_out
    arc_o = occ.addCircleArc(p_out0, ctr, p_out90)  # r_out,   0 -> 90 deg
    l_y = occ.addLine(p_out90, p_in90)             # x = 0,  r_out -> r_in
    arc_i = occ.addCircleArc(p_in90, ctr, p_in0)   # r_in,  90 -> 360 deg
    s = occ.addPlaneSurface([occ.addCurveLoop([l_x, arc_o, l_y, arc_i])])
    occ.synchronize()
    for c, m in ((l_x, nr), (l_y, nr), (arc_o, nt), (arc_i, nt)):
        gmsh.model.mesh.setTransfiniteCurve(c, m + 1)
    gmsh.model.mesh.setTransfiniteSurface(s)
    gmsh.model.mesh.setRecombine(2, s)
    vols = [t for d, t in occ.extrude([(2, s)], 0, 0, ZL,
                                      numElements=[nz], recombine=True) if d == 3]
    occ.synchronize()
    if len(vols) != 1:
        recipe.die('the extrusion made %d volumes' % len(vols))
    return vols[0], (nr, nt, nz)


def classify(volume):
    groups = {k: [] for k in PATCHES}
    mid = 0.5 * (R_IN + R_OUT)
    for s in sorted(recipe.surfaces_of(volume)):
        if recipe.lies_in_plane(s, 2, 0.0):
            groups['zmin'].append(s)
        elif recipe.lies_in_plane(s, 2, ZL):
            groups['zmax'].append(s)
        elif recipe.lies_in_plane(s, 1, 0.0):
            groups['symmetry_y0'].append(s)
        elif recipe.lies_in_plane(s, 0, 0.0):
            groups['symmetry_x0'].append(s)
        else:
            x, y, _ = recipe.centre_of(2, s)
            groups['inner' if math.hypot(x, y) < mid else 'outer'].append(s)
    return groups


def main(argv=None):
    ap = argparse.ArgumentParser(description='The thick-walled cylinder quadrant as one hex region.')
    ap.add_argument('--level', type=int, default=1, choices=(1, 2, 3))
    ap.add_argument('--out', default=None, help='default cases/thick_cylinder_L<level>')
    args = ap.parse_args(argv)
    out = os.path.abspath(args.out or os.path.join(
        recipe.repo_root(), 'cases', 'thick_cylinder_L%d' % args.level))
    recipe.fresh_dir(out)
    t0 = time.time()
    recipe.gmsh_begin()
    try:
        vol, n = build(args.level)
        recipe.add_groups({'cylinder': vol}, classify(vol))
        counts = recipe.generate_and_count({'cylinder': vol})
        recipe.write_msh(os.path.join(out, 'thick_cylinder.msh'))
        version = gmsh.__version__
    finally:
        recipe.gmsh_end()
    t_gmsh = time.time() - t0
    t1 = time.time()
    regions = recipe.make_layout(os.path.join(out, 'thick_cylinder.msh'), out,
                                 None, {'cylinder': 'cylinder'})
    t_m4 = time.time() - t1
    side = recipe.summarize(regions, out, counts)
    sidecar = {'recipe': 'thick_cylinder', 'level': args.level, 'gmsh': version, 'units': 'm',
               'timings_s': {'gmsh': round(t_gmsh, 1), 'regions': round(t_m4, 1)},
               'r_in': R_IN, 'r_out': R_OUT, 'length': ZL, 'n': list(n),
               'reference': 'Timoshenko & Goodier, Theory of Elasticity, 3rd ed. (1970); '
                            'Boley & Weiner, Theory of Thermal Stresses, ch. 9',
               'regions': side['regions'], 'interfaces': side['interfaces']}
    recipe.write_json(os.path.join(out, 'thick_cylinder.json'), sidecar)
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    sys.exit(main())
