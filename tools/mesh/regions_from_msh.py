#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""
regions_from_msh.py - one multi-volume Gmsh MSH 4.1 mesh becomes the region
layout: one complete, standalone polyMesh per named volume, every shared face
a boundary face of BOTH regions in a patch pair whose k-th faces coincide,
and a regions.json naming it all (docs/10-fsi-solid-mesh-plan.md §C).

    python tools/mesh/regions_from_msh.py <mesh.msh> <outDir>
        [--fluid NAME] [--material REGION=NAME]... [--units U] [--tolerance T] [--overwrite]

--fluid NAME (default 'fluid') names the fluid region; '--fluid none' makes
every region solid. Every other volume is a solid, ordered by ascending 3-D
physical tag. A shared face becomes the pair <a>_to_<b> / <b>_to_<a> in
identical order - the k-th face of one is the reversed k-th face of the
other - so the solver can pair by index. polymesh_write emits the polyMesh
bytes and regions_check runs on the finished layout as the last step.
Refused by name: a volume without exactly one physical name, a missing $PhysicalNames section,
a face touched by three cells, a bad volume name, --fluent (tet-only and single cell zone), an
existing regions.json without --overwrite, --fluid/--material naming no volume.
"""
import argparse
import json
import os
import sys

import numpy as np

import polymesh_write
import regions_check

TOOL_VERSION = '1'

# Reference-element face tables, verbatim from rust/src/io/msh.rs:77-113 (outward winding).
TET_FACES = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]
HEX_FACES = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [3, 7, 6, 2], [0, 4, 7, 3], [1, 2, 6, 5]]
PRISM_FACES = [[0, 2, 1], [3, 4, 5], [0, 1, 4, 3], [1, 2, 5, 4], [0, 3, 5, 2]]
PYRAMID_FACES = [[0, 3, 2, 1], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]

NODE_COUNT = {1: 2, 2: 3, 3: 4, 4: 4, 5: 8, 6: 6, 7: 5, 15: 1}
FACES_OF = {4: TET_FACES, 5: HEX_FACES, 6: PRISM_FACES, 7: PYRAMID_FACES}


def die(msg, code=1):
    sys.stderr.write('regions_from_msh: %s\n' % msg)
    sys.stderr.flush()
    raise SystemExit(code)


class _Tok:
    """A whitespace tokeniser over the whole ASCII .msh - msh.rs's Lex."""

    def __init__(self, text, path):
        self.t, self.i, self.path = text.split(), 0, path

    def peek(self): return self.t[self.i] if self.i < len(self.t) else None

    def next(self):
        if self.i >= len(self.t): die('%s: unexpected end of file' % self.path)
        self.i += 1; return self.t[self.i - 1]

    def Int(self): return int(self.next())

    def Num(self): return float(self.next())

    def expect(self, want):
        got = self.next()
        if got != want: die("%s: expected '%s', found '%s'" % (self.path, want, got))

    def quoted(self):
        """A `dim tag "name"` name - tokens until the closing quote (names hold spaces)."""
        parts = [self.next()]
        while not parts[-1].endswith('"') or len(parts[-1]) < 2:
            parts.append(self.next())
        return ' '.join(parts)[1:-1]


def _bbox_entity(lx):
    """A curve/surface/volume record: `tag bbox*6 numPhys phys* numBounding bounding*`."""
    tag = lx.Int()
    for _ in range(6): lx.Num()
    ph = [lx.Int() for _ in range(lx.Int())]
    for _ in range(lx.Int()): lx.Int()
    return tag, ph


def read_msh(path):
    """Read an MSH 4.1 ASCII file, keeping the volume names msh.rs discards:
    {'points', 'cells' (file order), 'cell_volume', 'volume_names' (ascending 3-D tag),
    'surface_name' keyed by sorted vertex tuple}."""
    try:
        with open(path, encoding='utf-8', errors='replace') as f:
            lx = _Tok(f.read(), path)
    except OSError as e:
        die('cannot read %s: %s' % (path, e), 2)
    lx.expect('$MeshFormat')
    if lx.next() != '4.1': die('%s: unsupported MSH version (only 4.1 ASCII; re-export with -format msh41)' % path)
    if lx.Int() != 0: die('%s: binary MSH is not supported; re-export as ASCII' % path)
    lx.Int(); lx.expect('$EndMeshFormat')
    phys = {}
    if lx.peek() == '$PhysicalNames':
        lx.next()
        for _ in range(lx.Int()):
            dim, tag = lx.Int(), lx.Int()
            phys[(dim, tag)] = lx.quoted()
        lx.expect('$EndPhysicalNames')
    else:
        die("%s: no $PhysicalNames section - the mesh has no region or patch identity; group the volumes and surfaces and re-export" % path)
    lx.expect('$Entities')
    n_pt, n_cur, n_surf, n_vol = lx.Int(), lx.Int(), lx.Int(), lx.Int()
    for _ in range(n_pt):
        lx.Int(), lx.Num(), lx.Num(), lx.Num()
        for _ in range(lx.Int()): lx.Int()
    for _ in range(n_cur): _bbox_entity(lx)
    surf_phys, vol_phys = {}, {}
    for _ in range(n_surf):
        tag, ph = _bbox_entity(lx); surf_phys[tag] = ph
    for _ in range(n_vol):
        tag, ph = _bbox_entity(lx); vol_phys[tag] = ph
    lx.expect('$EndEntities'); lx.expect('$Nodes')
    n_blocks, n_nodes = lx.Int(), lx.Int(); lx.Int(); lx.Int()
    tag_to_idx, pts = {}, []
    for _ in range(n_blocks):
        edim = lx.Int(); lx.Int(); param = lx.Int(); n_in = lx.Int()
        extra = max(edim, 0) if param != 0 else 0   # a parametric node carries them AFTER xyz
        for tg in [lx.Int() for _ in range(n_in)]:
            x, y, z = lx.Num(), lx.Num(), lx.Num()
            for _ in range(extra): lx.Num()
            if tg in tag_to_idx: die('%s: node tag %d appears more than once' % (path, tg))
            tag_to_idx[tg] = len(pts); pts.append((x, y, z))
    if len(pts) != n_nodes: die('%s: $Nodes declared %d nodes but the blocks contained %d' % (path, n_nodes, len(pts)))
    lx.expect('$EndNodes')
    lx.expect('$Elements')
    n_blocks, n_elements = lx.Int(), lx.Int(); lx.Int(); lx.Int()
    cells, cell_volume, surface_name, n_read = [], [], {}, 0
    for _ in range(n_blocks):
        lx.Int(); etag = lx.Int(); etype = lx.Int(); n_in = lx.Int()
        if etype not in NODE_COUNT:
            die('%s: unsupported Gmsh element type %d (regions_from_msh reads point/line/triangle/quadrangle/tetrahedron/hexahedron/prism/pyramid, types 1,2,3,4,5,6,7,15)' % (path, etype))
        name = None
        if etype in FACES_OF:
            ph = vol_phys.get(etag, [])
            if not ph:
                die('%s: volume entity %d carries no physical name; every volume must be a named region - Mesh.SaveAll=0 writes only grouped volumes' % (path, etag))
            if len(ph) >= 2: die('%s: volume entity %d carries %d physical names; a region has one' % (path, etag, len(ph)))
            name = phys.get((3, ph[0]))
            if name is None: die('%s: volume entity %d has physical tag %d, which $PhysicalNames does not name' % (path, etag, ph[0]))
        elif etype in (2, 3):
            ph = surf_phys.get(etag, []); name = phys.get((2, ph[0])) if ph else None
        for _ in range(n_in):
            lx.Int(); verts = []
            for _ in range(NODE_COUNT[etype]):
                tg = lx.Int()
                if tg not in tag_to_idx: die('%s: element references node tag %d, which $Nodes never defined' % (path, tg))
                verts.append(tag_to_idx[tg])
            if etype in FACES_OF:
                cells.append((etype, verts)); cell_volume.append(name)
            elif etype in (2, 3) and name is not None and tuple(sorted(verts)) not in surface_name:
                surface_name[tuple(sorted(verts))] = name
        n_read += n_in
    if n_read != n_elements: die('%s: $Elements declared %d elements but the blocks contained %d' % (path, n_elements, n_read))
    lx.expect('$EndElements')
    min_tag, used_names = {}, set(cell_volume)
    for (dim, tag), nm in phys.items():
        if dim == 3 and nm in used_names: min_tag[nm] = min(min_tag.get(nm, tag), tag)
    volume_names = sorted(min_tag, key=min_tag.get)
    for nm in volume_names:
        try: polymesh_write.check_patch_name(nm, 'volume')
        except ValueError as e: die('%s: %s' % (path, e))
        if '_to_' in nm:
            die("%s: volume name '%s' carries '_to_' - the interface patch names <a>_to_<b> would be ambiguous" % (path, nm))
    return {'points': np.array(pts, dtype=np.float64), 'cells': cells, 'cell_volume': cell_volume,
            'volume_names': volume_names, 'surface_name': surface_name}


def shared_faces(cells, cell_volume):
    """{sorted-vertex-tuple: (name_a, name_b)} for faces two DIFFERENT volumes touch;
    a third touch is fatal, as in msh.rs's build_raw_mesh."""
    counts, first, shared = {}, {}, {}
    for ci, (etype, verts) in enumerate(cells):
        vol = cell_volume[ci]
        for face in FACES_OF[etype]:
            key = tuple(sorted(verts[li] for li in face))
            c = counts.get(key, 0) + 1
            counts[key] = c
            if c == 1: first[key] = vol
            elif c == 2:
                if first[key] != vol: shared[key] = (first[key], vol)
            else: die('face with vertices %s is shared by more than two cells; the mesh is not a valid volume mesh' % (key,))
    return shared


def region_faces(cells, cell_ids):
    """One region's faces over its LOCAL cell numbers - msh.rs:458-596 per region:
    (internal [(owner, neighbour, verts)] stable-sorted by (owner, neighbour),
    boundary [(key, owner, verts)] in first-touch order, winding outward)."""
    recs, index = [], {}
    for local, ci in enumerate(cell_ids):
        etype, verts = cells[ci]
        for face in FACES_OF[etype]:
            vs = tuple(verts[li] for li in face)
            key = tuple(sorted(vs))
            i = index.get(key)
            if i is None:
                index[key] = len(recs); recs.append([vs, local, None])
            else:
                rec = recs[i]
                if rec[2] is not None: die('face with vertices %s is shared by more than two cells; the mesh is not a valid volume mesh' % (key,))
                rec[2] = local
    internal, boundary = [], []
    for vs, o, n in recs:
        if n is not None: internal.append((o, n, vs))
        else: boundary.append((tuple(sorted(vs)), o, vs))
    internal.sort(key=lambda t: (t[0], t[1]))
    return internal, boundary


def _convention_type(name):
    # convert_mesh.rs's convention_type: case-insensitive prefix, else patch.
    return next((t for pre, t in (('wall', 'wall'), ('empty', 'empty'), ('symmetry', 'symmetry')) if name.lower().startswith(pre)), 'patch')


def _pieces(n_cells, internal):  # connected pieces through internal faces - report only
    parent = list(range(n_cells))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    for o, n, _ in internal: parent[find(o)] = find(n)
    return len({find(c) for c in range(n_cells)})


def split_regions(msh, fluid, materials, tolerance):
    """([region dicts], [interface dicts]) - C5: the split, the ordering, the pairing."""
    cells, cell_volume, vol_names = msh['cells'], msh['cell_volume'], msh['volume_names']
    shared = shared_faces(cells, cell_volume)
    if fluid == 'none':
        if 'none' in vol_names: die("no volume may be named 'none' in a --fluid none layout", 2)
        order, kinds = list(vol_names), dict.fromkeys(vol_names, 'solid')
    else:
        if fluid not in vol_names: die('--fluid %s: no volume named %r; the volumes are: %s' % (fluid, fluid, ', '.join(vol_names)), 2)
        order = [fluid] + [nm for nm in vol_names if nm != fluid]
        kinds = {nm: ('fluid' if nm == fluid else 'solid') for nm in order}
    built = {}
    for nm in order:
        cell_ids = [i for i, cv in enumerate(cell_volume) if cv == nm]
        used = sorted({v for ci in cell_ids for v in cells[ci][1]})
        g2l = {g: i for i, g in enumerate(used)}
        internal, boundary = region_faces(cells, cell_ids)
        internal = [(o, n, tuple(g2l[g] for g in vs)) for o, n, vs in internal]
        iface_names, defaults, patch_order, by_patch = set(), 0, [], {}
        for key, o, gverts in boundary:
            pair = shared.get(key)
            if pair is None:
                gname = msh['surface_name'].get(key)
                pname, defaults = (gname, defaults) if gname else ('defaultFaces', defaults + 1)
            else:
                a, b = pair
                pname = '%s_to_%s' % (nm, b if a == nm else a)
                iface_names.add(pname)
                gname = msh['surface_name'].get(key)
                if gname is not None and gname not in (a + '_to_' + b, b + '_to_' + a):
                    die("2-D physical group '%s' covers the interface face with vertices %s between '%s' and '%s'; an interface face carries only '%s' or '%s'" % (gname, key, a, b, a + '_to_' + b, b + '_to_' + a))
            if pname not in by_patch: by_patch[pname] = []; patch_order.append(pname)
            by_patch[pname].append((o, tuple(g2l[g] for g in gverts)))
        built[nm] = {'points': msh['points'][used], 'internal': internal, 'patch_order': patch_order,
                     'by_patch': by_patch, 'g2l': g2l, 'l2g': used, 'n_cells': len(cell_ids),
                     'iface_names': iface_names, 'defaults': defaults, 'pieces': _pieces(len(cell_ids), internal)}
    interfaces = []
    for pa, pb in sorted({tuple(sorted((a, b))) for a, b in shared.values()}):
        a, b = (pa, pb) if order.index(pa) < order.index(pb) else (pb, pa)
        ra, rb = built[a], built[b]
        list_a = ra['by_patch'].get(a + '_to_' + b)
        own_b = rb['by_patch'].get(b + '_to_' + a)
        if list_a is None or own_b is None: die("internal: the shared faces of '%s' and '%s' did not become boundary patches" % (a, b))
        # R2 (C5.7): side B is rebuilt as A's k-th face, global vertex list reversed.
        bkey = {tuple(sorted(rb['l2g'][g] for g in lv)): o for o, lv in own_b}
        new_b = []
        for o_a, lv_a in list_a:
            gv = [ra['l2g'][g] for g in lv_a]
            ob = bkey.get(tuple(sorted(gv)))
            if ob is None: die('internal: a face of %s is not among the first-touch faces of %s' % (b + '_to_' + a, b))
            new_b.append((ob, tuple(rb['g2l'][g] for g in reversed(gv))))
        if len(new_b) != len(bkey): die('internal: the rebuilt %s does not cover the first-touch faces of %s' % (b + '_to_' + a, b))
        rb['by_patch'][b + '_to_' + a] = new_b
        interfaces.append({'a': a, 'b': b, 'faces': len(new_b), 'tolerance': tolerance})
    regions = []
    for nm in order:
        r = built[nm]
        faces = [list(vs) for _, _, vs in r['internal']]
        owner = [o for o, _, _ in r['internal']]
        neighbour = [n for _, n, _ in r['internal']]
        patches = []
        for pname in r['patch_order']:
            grp = r['by_patch'][pname]
            patches.append((pname, 'patch' if pname in r['iface_names'] else _convention_type(pname), len(grp)))
            for o, lv in grp:
                faces.append(list(lv)); owner.append(o)
        regions.append({'name': nm, 'kind': kinds[nm], 'material': materials.get(nm), 'points': r['points'],
                        'faces': faces, 'owner': owner, 'neighbour': neighbour, 'patches': patches,
                        'n_cells': r['n_cells'], 'pieces': r['pieces'], 'defaults': r['defaults']})
    return regions, interfaces


def write_layout(out_dir, regions, interfaces, units, geometry_basename, config):
    """One polyMesh per region, regions.json written LAST; returns the manifest path."""
    os.makedirs(out_dir, exist_ok=True)
    mregions = []
    for r in regions:
        polymesh_write.write_polymesh(os.path.join(out_dir, r['name'], 'polyMesh'), r['points'], r['faces'], r['owner'], r['neighbour'], r['patches'])
        e = {'name': r['name'], 'kind': r['kind'], 'polyMesh': '%s/polyMesh' % r['name']}
        if r['material'] is not None: e['material'] = r['material']
        mregions.append(e)
    ifaces = [{'regions': [i['a'], i['b']], 'patches': ['%s_to_%s' % (i['a'], i['b']), '%s_to_%s' % (i['b'], i['a'])], 'faces': i['faces'], 'tolerance': i['tolerance']} for i in interfaces]
    manifest = {'version': 1, 'units': units, 'regions': mregions, 'interfaces': ifaces,
                'source': {'tool': 'regions_from_msh', 'version': TOOL_VERSION, 'geometry': geometry_basename, 'config': config}}
    path = os.path.join(out_dir, 'regions.json')
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(manifest, f, indent=2); f.write('\n')
    return path


def parse_args(argv):
    ap = argparse.ArgumentParser(description='Split a multi-volume Gmsh MSH 4.1 mesh into the region layout: one polyMesh per named volume, conformal interface patch pairs, regions.json.')
    ap.add_argument('mesh', help='the .msh (MSH 4.1 ASCII, one physical group per volume)')
    ap.add_argument('out_dir', help='the layout directory to write')
    ap.add_argument('--fluid', default='fluid', metavar='NAME', help="the fluid volume's name (default 'fluid'; 'none' makes every region solid)")
    ap.add_argument('--material', action='append', default=[], metavar='REGION=NAME', help='a solid region carries one material name')
    ap.add_argument('--units', default='m', metavar='U', help='the manifest units (default m)')
    ap.add_argument('--tolerance', type=float, default=1e-9, metavar='T', help='the R2 pairing tolerance (default 1e-9)')
    ap.add_argument('--overwrite', action='store_true', help='replace an existing regions.json')
    ap.add_argument('--fluent', nargs='?', const='', default=None, metavar='PATH', help='not supported here: refused by name')
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.fluent is not None:
        die('--fluent is not supported here: the Fluent writer is tet-only and single cell zone (ofgpu-convert-mesh -fluent converts one tet mesh into one Fluent cell zone); a region layout is one polyMesh per volume', 2)
    materials = {}
    for spec in args.material:
        nm, sep, mat = spec.partition('=')
        if not sep or not nm or not mat: die('--material %s: expected REGION=NAME' % spec, 2)
        materials[nm] = mat
    manifest = os.path.join(args.out_dir, 'regions.json')
    if os.path.isfile(manifest):
        if not args.overwrite: die('%s already holds a regions.json; pass --overwrite to replace it' % args.out_dir, 2)
        try: f = open(manifest, encoding='utf-8'); old = json.load(f)
        except (OSError, ValueError) as e: die('cannot read the old %s: %s' % (manifest, e), 2)
        for nm in [r.get('name') for r in old.get('regions', []) if isinstance(r, dict)]:
            d = os.path.join(args.out_dir, nm or '')
            if not (nm and os.path.isdir(d)): continue
            for root, dirs, files in os.walk(d, topdown=False):
                for fn in files: os.remove(os.path.join(root, fn))
                for sub in dirs: os.rmdir(os.path.join(root, sub))
            os.rmdir(d)
    msh = read_msh(args.mesh)
    vols = msh['volume_names']
    if args.fluid != 'none' and args.fluid not in vols:
        die('--fluid %s: no volume named %r; the volumes are: %s' % (args.fluid, args.fluid, ', '.join(vols)), 2)
    for nm in materials:
        if nm not in vols: die('--material %s=...: no volume named %r; the volumes are: %s' % (nm, nm, ', '.join(vols)), 2)
        if args.fluid != 'none' and nm == args.fluid: die('--material %s=...: the fluid region carries no material' % nm, 2)
    regions, interfaces = split_regions(msh, args.fluid, materials, args.tolerance)
    per = dict.fromkeys(vols, 0)
    for nm in msh['cell_volume']: per[nm] += 1
    print('[regions] %s: %d cells, %d points, %d volumes: %s' % (os.path.basename(args.mesh), len(msh['cells']), len(msh['points']), len(vols), ', '.join('%s (%d)' % (nm, per[nm]) for nm in vols)))
    for r in regions:
        print('[regions] %s: %d cells, %d points, %d faces (%d internal), %d patches, pieces %d' % (r['name'], r['n_cells'], len(r['points']), len(r['faces']), len(r['neighbour']), len(r['patches']), r['pieces']))
        for pname, ptype, n_f in r['patches']: print("[regions]   patch '%s' (%s): %d face(s)" % (pname, ptype, n_f))
        if r['defaults']: print("[regions] warning: %d boundary face(s) of region '%s' were not covered by any physical surface; assigned to patch 'defaultFaces'" % (r['defaults'], r['name']))
        if r['kind'] == 'solid' and r['material'] is None: print("[regions] note: solid '%s' has no --material; regions.json carries none" % r['name'])
    for i in interfaces: print('[regions] interface %s_to_%s / %s_to_%s: %d faces' % (i['a'], i['b'], i['b'], i['a'], i['faces']))
    flags = (['--fluid', args.fluid] if args.fluid != 'fluid' else []) + [x for spec in args.material for x in ('--material', spec)] + (['--units', args.units] if args.units != 'm' else []) + (['--tolerance', repr(args.tolerance)] if args.tolerance != 1e-9 else [])
    path = write_layout(args.out_dir, regions, interfaces, args.units, os.path.basename(args.mesh), ' '.join(flags))
    print('[regions] wrote %s  (%d regions, %d interface%s)' % (path, len(regions), len(interfaces), '' if len(interfaces) == 1 else 's'))
    violations, report = regions_check.check_layout(path)
    regions_check.print_report(report, violations)
    return 1 if violations else 0


if __name__ == '__main__':
    try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception: pass
    sys.exit(main())
