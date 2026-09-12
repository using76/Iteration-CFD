#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""
cantilever.py - the S7/S8 bending gates' beam as one transfinite hex region:
L = 1.0 x b = 0.1 x h = 0.1 m, clamped at x = 0. The sidecar carries the
closed forms the gates check against (nothing is computed here):
f_1 = (beta_1 L)^2/(2 pi L^2) sqrt(E I/(rho A)),  cos(beta_1 L) cosh(beta_1 L) = -1,
beta_1 L = 1.87510406871;  w_tip = q L^4/(8 E I);  rectangle I = b h^3/12,
A = b h. Sources: Blevins, Formulas for Natural Frequency and Mode Shape
(1979) Table 8-1; Rao, Mechanical Vibrations, 5th ed. SS 8.5.

    python tools/mesh/examples/cantilever.py [--level 1|2|3] [--out DIR]

Levels (nx, ny, nz) = (40, 4, 4), (80, 8, 8), (160, 16, 16) - 640 cells at
level 1. Default --out cases/cantilever_L<level> (git-ignored).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import gmsh

import recipe

L, B, H = 1.0, 0.1, 0.1
LEVELS = ((40, 4, 4), (80, 8, 8), (160, 16, 16))
PATCHES = ('fixed_end', 'free_end', 'bottom', 'top', 'side_y0', 'side_y1')


def build(level):
    occ = gmsh.model.occ
    vol = occ.addBox(0, 0, 0, L, B, H)
    occ.synchronize()
    return vol, LEVELS[level - 1]


def classify(volume):
    groups = {k: [] for k in PATCHES}
    for s in sorted(recipe.surfaces_of(volume)):
        if recipe.lies_in_plane(s, 0, 0.0):
            groups['fixed_end'].append(s)
        elif recipe.lies_in_plane(s, 0, L):
            groups['free_end'].append(s)
        elif recipe.lies_in_plane(s, 2, 0.0):
            groups['bottom'].append(s)
        elif recipe.lies_in_plane(s, 2, H):
            groups['top'].append(s)
        elif recipe.lies_in_plane(s, 1, 0.0):
            groups['side_y0'].append(s)
        elif recipe.lies_in_plane(s, 1, B):
            groups['side_y1'].append(s)
        else:
            recipe.die('surface %d is in no group' % s)
    return groups


def main(argv=None):
    ap = argparse.ArgumentParser(description='The bending-gate cantilever as one hex region.')
    ap.add_argument('--level', type=int, default=1, choices=(1, 2, 3))
    ap.add_argument('--out', default=None, help='default cases/cantilever_L<level>')
    args = ap.parse_args(argv)
    out = os.path.abspath(args.out or os.path.join(
        recipe.repo_root(), 'cases', 'cantilever_L%d' % args.level))
    recipe.fresh_dir(out)
    t0 = time.time()
    recipe.gmsh_begin()
    try:
        vol, n = build(args.level)
        recipe.add_groups({'beam': vol}, classify(vol))
        recipe.transfinite_box(vol, n)
        counts = recipe.generate_and_count({'beam': vol})
        recipe.write_msh(os.path.join(out, 'cantilever.msh'))
        version = gmsh.__version__
    finally:
        recipe.gmsh_end()
    t_gmsh = time.time() - t0
    t1 = time.time()
    regions = recipe.make_layout(os.path.join(out, 'cantilever.msh'), out,
                                 None, {'beam': 'beam'})
    t_m4 = time.time() - t1
    side = recipe.summarize(regions, out, counts)
    sidecar = {'recipe': 'cantilever', 'level': args.level, 'gmsh': version, 'units': 'm',
               'timings_s': {'gmsh': round(t_gmsh, 1), 'regions': round(t_m4, 1)},
               'L': L, 'b': B, 'h': H, 'n': list(n), 'I': B * H ** 3 / 12.0, 'A': B * H,
               'beta1_L': 1.87510406871,
               'f1_formula': 'f_1 = (beta_1 L)^2/(2 pi L^2) sqrt(E I/(rho A))',
               'w_tip_formula': 'w_tip = q L^4/(8 E I)',
               'reference': 'Blevins, Formulas for Natural Frequency and Mode Shape (1979) '
                            'Table 8-1; Rao, Mechanical Vibrations, 5th ed. SS 8.5',
               'regions': side['regions'], 'interfaces': side['interfaces']}
    recipe.write_json(os.path.join(out, 'cantilever.json'), sidecar)
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    sys.exit(main())
