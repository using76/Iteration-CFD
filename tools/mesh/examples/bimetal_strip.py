#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""
bimetal_strip.py - a bonded two-material strip as ONE transfinite hex region.

L = 0.1 x w = 0.01 x 2 h1 = 0.002 m, the bond plane z = h1 a mesh plane (nz
even). S9's case format assigns materials per bounds box - a cell is in a
zone iff its centroid is in the closed box - so NO cell-index list is written
anywhere: the sidecar's `zones` carries the two boxes and their measured cell
counts. The strip curvature (S. Timoshenko, J. Opt. Soc. Am. 11 (1925) 233,
DOI 10.1364/JOSA.11.000233) is S8's to write; this recipe is its mesh.

    python tools/mesh/examples/bimetal_strip.py [--level 1|2|3] [--out DIR]

Default --out cases/bimetal_strip_L<level> (git-ignored; regenerate).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import gmsh

import recipe

L, W, H1, H2 = 0.1, 0.01, 0.001, 0.001
LEVELS = ((50, 5, 8), (100, 10, 16), (200, 20, 32))
PATCHES = ('fixed_end', 'free_end', 'bottom', 'top', 'side_y0', 'side_y1')


def build(level):
    occ = gmsh.model.occ
    vol = occ.addBox(0, 0, 0, L, W, H1 + H2)
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
        elif recipe.lies_in_plane(s, 2, H1 + H2):
            groups['top'].append(s)
        elif recipe.lies_in_plane(s, 1, 0.0):
            groups['side_y0'].append(s)
        elif recipe.lies_in_plane(s, 1, W):
            groups['side_y1'].append(s)
        else:
            recipe.die('surface %d is in no group' % s)
    return groups


def zones(layout_dir, regions, length, w, h1):
    """The sidecar's `zones`: the two S9 bounds boxes, cells counted by centroid."""
    centres = recipe.cell_centres(recipe.load_region(layout_dir, regions, 'strip'))
    boxes = {'lower': ((0.0, 0.0, 0.0), (length, w, h1)),
             'upper': ((0.0, 0.0, h1), (length, w, h1 + H2))}
    out = {name: {'bounds': {'min': list(lo), 'max': list(hi)}, 'cells': 0}
           for name, (lo, hi) in boxes.items()}
    for c in centres:
        hit = [name for name, (lo, hi) in boxes.items()
               if all(lo[i] <= c[i] <= hi[i] for i in range(3))]
        if len(hit) != 1:
            recipe.die('cell centroid %s is in %d zones (%s): the bond plane must be a mesh plane'
                       % (c.tolist(), len(hit), '+'.join(hit) or 'none'))
        out[hit[0]]['cells'] += 1
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description='The bonded bimetal strip as one transfinite hex region.')
    ap.add_argument('--level', type=int, default=1, choices=(1, 2, 3))
    ap.add_argument('--out', default=None, help='default cases/bimetal_strip_L<level>')
    args = ap.parse_args(argv)
    out = os.path.abspath(args.out or os.path.join(
        recipe.repo_root(), 'cases', 'bimetal_strip_L%d' % args.level))
    recipe.fresh_dir(out)
    t0 = time.time()
    recipe.gmsh_begin()
    try:
        vol, n = build(args.level)
        recipe.add_groups({'strip': vol}, classify(vol))
        recipe.transfinite_box(vol, n)
        counts = recipe.generate_and_count({'strip': vol})
        recipe.write_msh(os.path.join(out, 'bimetal_strip.msh'))
        version = gmsh.__version__
    finally:
        recipe.gmsh_end()
    t_gmsh = time.time() - t0
    t1 = time.time()
    regions = recipe.make_layout(os.path.join(out, 'bimetal_strip.msh'), out,
                                 None, {'strip': 'strip'})
    t_m4 = time.time() - t1
    side = recipe.summarize(regions, out, counts)
    sidecar = {'recipe': 'bimetal_strip', 'level': args.level, 'gmsh': version, 'units': 'm',
               'timings_s': {'gmsh': round(t_gmsh, 1), 'regions': round(t_m4, 1)},
               'L': L, 'w': W, 'h1': H1, 'h2': H2, 'n': list(n),
               'zones': zones(out, regions, L, W, H1),
               'reference': 'S. Timoshenko, J. Opt. Soc. Am. 11 (1925) 233, '
                            'DOI 10.1364/JOSA.11.000233',
               'regions': side['regions'], 'interfaces': side['interfaces']}
    recipe.write_json(os.path.join(out, 'bimetal_strip.json'), sidecar)
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    sys.exit(main())
