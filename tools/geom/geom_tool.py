#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.

"""geom_tool - list and export the solids of a geometry file on gmsh's
OpenCASCADE kernel.

    python tools/geom/geom_tool.py info   <file> [--json <out.json>] [--scale S]
    python tools/geom/geom_tool.py export <file> --out <path> [--scale S] [--stl-size M]

STEP/STP, BREP, IGES, XAO and STL are read; STEP, BREP, XAO and STL are
written, by the extension of --out. A STEP keeps neither names nor materials
and OCC renumbers tags on the way back in, so the names travel in a
<stem>.geom.json sidecar beside the file and are matched to the solids by
centroid and volume first, by tag only as the fallback. Known limits: IGES
imports as surfaces only and is never written; an STL is a discrete
triangulation, so it has no solids and its volume is the divergence-theorem
sum over the triangles, null while the surface is open; gmsh's own STEP
writer does not carry entity names - the sidecar is the store. edit
(transforms and booleans, ops history in the sidecar) is the next unit.
"""

import argparse
import json
import math
import os
import sys

import gmsh
import numpy as np

FORMATS = {'.step': 'step', '.stp': 'step', '.brep': 'brep', '.iges': 'iges',
           '.igs': 'iges', '.stl': 'stl', '.xao': 'xao'}
DEFAULT_SCALE = {'step': 0.001, 'iges': 0.001, 'brep': 1.0, 'xao': 1.0, 'stl': 1.0}
EXPORTABLE = ('step', 'brep', 'xao', 'stl')
MATCH_REL = 1e-6
OCC_LABEL_PREFIX = 'Open CASCADE'
LOAD = {}   # filled by load_model: format, scale, surfaces_before/after (None for xao/stl)
HEADER = ('%5s  %-16s  %-10s  %-15s   %-30s  %6s  %s'
          % ('tag', 'name', 'material', 'volume [m^3]', 'centroid [m]', 'faces', 'closed'))


def die(msg, code=1):
    sys.stderr.write('geom_tool: %s\n' % msg)
    sys.stderr.flush()
    raise SystemExit(code)


def log(msg):
    print(msg, flush=True)


def format_of(path):
    ext = os.path.splitext(path)[1].lower()
    if ext not in FORMATS:
        die("unsupported extension '%s' of %s (I read %s)"
            % (ext, path, ', '.join(sorted(FORMATS))), 2)
    return FORMATS[ext]


def sidecar_path(path):
    base = os.path.basename(path)
    stem = base.rsplit('.', 1)[0] if '.' in base else base
    return os.path.join(os.path.dirname(path), stem + '.geom.json')


def write_sidecar(path, data):
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(data, f, indent=1)
        f.write('\n')


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def read_sidecar(path):
    sp = sidecar_path(path)
    if not os.path.isfile(sp):
        return None
    name = os.path.basename(sp)
    try:
        with open(sp, encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        die("sidecar %s is not valid JSON: %s" % (name, e), 2)
    if not isinstance(data, dict) or data.get('version') != 1:
        die("sidecar %s: 'version' must be 1" % name, 2)
    solids = data.get('solids', [])
    if not isinstance(solids, list):
        die("sidecar %s: 'solids' must be a list" % name, 2)
    for r in solids:
        if not isinstance(r, dict) or not isinstance(r.get('name'), str):
            die("sidecar %s: every solid needs a string 'name'" % name, 2)
        if not isinstance(r.get('tag'), int) or isinstance(r['tag'], bool):
            die("sidecar %s: solid '%s' needs an int 'tag'" % (name, r['name']), 2)
        if not _num(r.get('volume')):
            die("sidecar %s: solid '%s' needs a number 'volume'" % (name, r['name']), 2)
        c = r.get('centroid')
        if not (isinstance(c, list) and len(c) == 3 and all(_num(x) for x in c)):
            die("sidecar %s: solid '%s' needs a 3-number 'centroid'" % (name, r['name']), 2)
    return data


def load_model(path, scale=None):
    fmt = format_of(path)
    if fmt == 'stl':
        if scale is not None and scale != 1.0:
            die('an STL is read in its own units; --scale %s is refused' % scale, 2)
        used = 1.0
    else:
        used = DEFAULT_SCALE[fmt] if scale is None else scale
        if used <= 0:
            die('--scale must be > 0 (got %s)' % scale, 2)
    LOAD.clear()
    LOAD['format'] = fmt
    LOAD['scale'] = used
    gmsh.option.setNumber('Geometry.OCCImportLabels', 1)
    gmsh.option.setNumber('Geometry.OCCScaling', used)
    if fmt in ('stl', 'xao'):
        # importShapes refuses both ('Unknown file type'); merge reads them
        gmsh.merge(path)
        if fmt == 'xao' and used != 1.0:
            # OCCScaling does not act on merge (probed 2026-09-12): dilate instead
            gmsh.model.occ.dilate(gmsh.model.getEntities(), 0, 0, 0, used, used, used)
    else:
        # mm -> m (or the --scale given), before the import, as the mesher's import stage does
        gmsh.model.occ.importShapes(path, highestDimOnly=True, format='')
    gmsh.model.occ.synchronize()
    if fmt in ('step', 'brep', 'iges'):
        # a STEP round trip duplicates a shared face; before/after is the report
        LOAD['surfaces_before'] = len(gmsh.model.getEntities(2))
        gmsh.model.occ.removeAllDuplicates()
        gmsh.model.occ.synchronize()
        LOAD['surfaces_after'] = len(gmsh.model.getEntities(2))
    else:
        LOAD['surfaces_before'] = None   # xao keeps its faces, an stl is not an OCC shape
        LOAD['surfaces_after'] = None
    LOAD['n_surfaces'] = len(gmsh.model.getEntities(2))   # the header line counts every face
    return sorted(t for d, t in gmsh.model.getEntities(3))


def bbox_diagonal():
    x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(-1, -1)
    return math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)


def label_name(tag):
    # the last non-empty path component, refused when it is OCC's own default
    name = gmsh.model.getEntityName(3, tag) or ''
    parts = [p.strip() for p in name.split('/') if p.strip()]
    if not parts or parts[-1].startswith(OCC_LABEL_PREFIX):
        return None
    return parts[-1]


def physical_name(tag):
    # getPhysicalGroupsForEntity returns a (possibly empty) numpy array: len(), not truthiness
    groups = gmsh.model.getPhysicalGroupsForEntity(3, tag)
    if len(groups) == 0:
        return None
    return gmsh.model.getPhysicalName(3, int(groups[0][1])) or None


def attach_names(sidecar, tags):
    tol = MATCH_REL * bbox_diagonal()
    recs = list(sidecar.get('solids') or []) if sidecar else []
    ratio = 1.0
    if sidecar:
        sc = sidecar.get('scale')
        ratio = LOAD['scale'] / (sc if _num(sc) and sc > 0 else LOAD['scale'])
    mass = {t: gmsh.model.occ.getMass(3, t) for t in tags}
    com = {t: np.array(gmsh.model.occ.getCenterOfMass(3, t), dtype=float) for t in tags}
    names, placed = {}, set()
    # 1. centroid + volume first: a round trip perturbs both by ~1e-15, and a
    #    moved solid must fail this step on purpose
    for i, r in enumerate(recs):
        c = np.array(r['centroid'], dtype=float) * ratio
        v = r['volume'] * ratio ** 3
        best, best_d = None, None
        for t in tags:
            if t in names:
                continue
            d = float(np.linalg.norm(com[t] - c))
            if d <= tol and abs(mass[t] - v) <= MATCH_REL * abs(v):
                if best is None or d < best_d:
                    best, best_d = t, d
        if best is not None:
            placed.add(i)
            names[best] = {'name': r['name'], 'material': r.get('material'),
                           'matched': 'centroid'}
    # 2. the tag is only the fallback key, and it says so
    for i, r in enumerate(recs):
        if i in placed or r['tag'] not in tags or r['tag'] in names:
            continue
        placed.add(i)
        names[r['tag']] = {'name': r['name'], 'material': r.get('material'),
                           'matched': 'tag'}
    for i, r in enumerate(recs):
        if i not in placed:
            log("  sidecar record '%s' (tag %s) matched no solid; dropped"
                % (r['name'], r['tag']))
    # 3. names from the model: physical group, OCC label, then solid_<tag>
    for t in tags:
        if t in names:
            continue
        nm = physical_name(t)
        kind = 'physical' if nm else None
        nm = nm or label_name(t)
        kind = kind or ('label' if nm else None)
        names[t] = {'name': nm or 'solid_%d' % t, 'material': None,
                    'matched': kind or 'fallback'}
    # 4. names are the key the next unit's edit resolves references by
    seen = {}
    for t in tags:
        if names[t]['name'] in seen:
            die("duplicate solid name '%s' (tags %d, %d): fix the sidecar"
                % (names[t]['name'], seen[names[t]['name']], t), 2)
        seen[names[t]['name']] = t
    return names


def solid_info(tag, rec):
    x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(3, tag)
    vol = gmsh.model.occ.getMass(3, tag)
    return {'tag': tag, 'name': rec['name'], 'material': rec['material'],
            'matched': rec['matched'], 'volume': vol,
            'bbox': [x0, y0, z0, x1, y1, z1],
            'centroid': list(gmsh.model.occ.getCenterOfMass(3, tag)),
            'n_faces': len(gmsh.model.getBoundary([(3, tag)], combined=False,
                                                  oriented=False, recursive=False)),
            # an OCC volume entity is bounded by closed shells by construction
            'closed': vol > 0}


def surface_info(tag):
    x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(2, tag)
    return {'tag': tag, 'name': gmsh.model.getEntityName(2, tag) or '',
            'area': gmsh.model.occ.getMass(2, tag),
            'bbox': [x0, y0, z0, x1, y1, z1],
            'centroid': list(gmsh.model.occ.getCenterOfMass(2, tag))}


def node_xyz():
    tags, coords, _ = gmsh.model.mesh.getNodes()
    # merged STL node tags are NOT contiguous (max tag >> count): by tag, not position
    return {int(t): np.array(coords[3 * i:3 * i + 3], dtype=float)
            for i, t in enumerate(tags)}


def discrete_info(tag, xyz):
    types, _, node_tags = gmsh.model.mesh.getElements(2, tag)
    flat = node_tags[list(types).index(2)] if 2 in list(types) else node_tags[0]
    tri = np.array(flat, dtype=np.int64).reshape(-1, 3)
    p = [np.array([xyz[int(t)] for t in tri[:, k]], dtype=float) for k in range(3)]
    pa, pb, pc = p
    # closed iff every undirected edge sits in exactly two triangles
    e = np.sort(np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    closed = bool(np.all(counts == 2))
    vol = None
    if closed:
        # divergence theorem on the closed triangulation; gmsh writes outward, so +
        vol = float(np.einsum('ij,ij->i', pa, np.cross(pb, pc)).sum() / 6.0)
    area = 0.5 * np.linalg.norm(np.cross(pb - pa, pc - pa), axis=1)
    centre = (area[:, None] * ((pa + pb + pc) / 3.0)).sum(axis=0) / area.sum()
    x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(2, tag)
    return {'tag': tag, 'name': gmsh.model.getEntityName(2, tag) or '',
            'n_triangles': int(len(tri)), 'bbox': [x0, y0, z0, x1, y1, z1],
            'centroid': [float(v) for v in centre], 'closed': closed, 'volume': vol}


def info_report(path, scale):
    fmt = format_of(path)
    tags = load_model(path, scale)
    sidecar = read_sidecar(path)
    names = attach_names(sidecar, tags)
    before = LOAD['surfaces_before']
    doc = {'version': 1, 'tool': 'geom_tool', 'gmsh': gmsh.__version__,
           'file': os.path.basename(path), 'format': fmt,
           'scale': LOAD['scale'], 'units': 'm',
           'duplicates_removed': None if before is None else
                                 {'surfaces_before': before,
                                  'surfaces_after': LOAD['surfaces_after']},
           'sidecar': os.path.basename(sidecar_path(path)) if sidecar else None,
           'note': None,
           'solids': [solid_info(t, names[t]) for t in tags],
           'surfaces': [], 'discrete': []}
    if fmt == 'iges':
        doc['note'] = 'IGES imports as surfaces only: no solids are listed'
        doc['surfaces'] = [surface_info(t) for d, t in gmsh.model.getEntities(2)]
    elif fmt == 'stl':
        xyz = node_xyz()
        doc['discrete'] = [discrete_info(t, xyz) for d, t in gmsh.model.getEntities(2)]
    return doc


def print_info(rep):
    head = 'geom_tool info %s: format %s, scale %s, %d solids, %d surfaces' % (
        rep['file'], rep['format'], rep['scale'], len(rep['solids']),
        LOAD.get('n_surfaces', len(rep['surfaces']) + len(rep['discrete'])))
    d = rep['duplicates_removed']
    if d and d['surfaces_before'] > d['surfaces_after']:
        head += ' (%d duplicate surface%s removed after import)' % (
            d['surfaces_before'] - d['surfaces_after'],
            '' if d['surfaces_before'] - d['surfaces_after'] == 1 else 's')
    log(head)
    if rep['note']:
        log('  ' + rep['note'])
    if rep['solids']:
        log(HEADER)
    for s in rep['solids']:
        log('%5d  %-16s  %-10s  %.9e   (%.6f, %.6f, %.6f)  %6d  %s' % (
            s['tag'], s['name'] or '-', s['material'] or '-', s['volume'],
            s['centroid'][0], s['centroid'][1], s['centroid'][2],
            s['n_faces'], 'yes' if s['closed'] else 'no'))
    for s in rep['solids']:
        if s['matched'] == 'tag':
            log("  name '%s': matched by tag only (its centroid or volume moved)" % s['name'])
    for f in rep['surfaces']:
        log('%5d  %-16s  %.9e   (%.6f, %.6f, %.6f)' % (
            f['tag'], f['name'] or '-', f['area'],
            f['centroid'][0], f['centroid'][1], f['centroid'][2]))
    for k in rep['discrete']:
        log('%5d  %-16s  %8d  %-5s  %s' % (
            k['tag'], k['name'] or '-', k['n_triangles'],
            'yes' if k['closed'] else 'no',
            '-' if k['volume'] is None else '%.9e' % k['volume']))


def export_model(out_path, tags, names, stl_size=None):
    fmt = format_of(out_path)
    if fmt == 'step':
        # occ.dilate is a GENERAL transform that re-approximates curved faces
        # (probed: 4.4e-4 relative on a sphere), so the exact mm export goes
        # out through a BREP and back in at OCCScaling = 1000, a uniform
        # gp_Trsf that is exact (2.5e-15 on the cut box, 0 on a sphere).
        # Tags may renumber: nothing reads the model by tag after this -
        # sidecar_for runs before it (cmd_export, run_edit).
        tmp = out_path + '.tmp.brep'
        gmsh.write(tmp)                      # every entity, metres, exact
        gmsh.model.remove(); gmsh.model.add('mm')
        gmsh.option.setNumber('Geometry.OCCScaling', 1000.0)
        gmsh.model.occ.importShapes(tmp); gmsh.model.occ.synchronize()
        gmsh.write(out_path)                 # the model IS in mm now
        gmsh.model.remove(); gmsh.model.add('m')
        gmsh.option.setNumber('Geometry.OCCScaling', 1.0)
        gmsh.model.occ.importShapes(tmp); gmsh.model.occ.synchronize()
        os.remove(tmp)
    elif fmt == 'brep':
        gmsh.write(out_path)     # metres as is; a shared face stays single
    elif fmt == 'xao':
        # the one format that carries the names natively
        gmsh.model.removePhysicalGroups()
        for t in sorted(tags):
            gmsh.model.addPhysicalGroup(3, [t], name=names[t]['name'])
            gmsh.model.setEntityName(3, t, names[t]['name'])
        gmsh.write(out_path)
    elif fmt == 'stl':
        size = stl_size if stl_size else bbox_diagonal() / 20.0
        gmsh.option.setNumber('Mesh.MeshSizeMin', size / 4.0)
        gmsh.option.setNumber('Mesh.MeshSizeMax', size)
        gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 8)
        gmsh.option.setNumber('Mesh.Algorithm', 6)
        gmsh.model.mesh.generate(2)
        ntri = sum(len(e) for e in gmsh.model.mesh.getElements(2)[1])
        gmsh.option.setNumber('Mesh.Binary', 1)
        gmsh.option.setNumber('Mesh.SaveAll', 1)
        gmsh.write(out_path)
        log('geom_tool export %s: %d triangles at mesh size %.6g m'
            % (os.path.basename(out_path), ntri, size))
    return fmt


def sidecar_for(out_path, tags, names, ops):
    # measured in metres, before the export replaces the model (E1); scale is the OUTPUT
    # format's default, so info on the new file converts records correctly
    return {'version': 1, 'tool': 'geom_tool', 'units': 'm',
            'scale': DEFAULT_SCALE[format_of(out_path)],
            'file': os.path.basename(out_path),
            'solids': [{'tag': t, 'name': names[t]['name'],
                        'material': names[t]['material'],
                        'volume': gmsh.model.occ.getMass(3, t),
                        'centroid': list(gmsh.model.occ.getCenterOfMass(3, t))}
                       for t in sorted(tags)],
            'ops': list(ops or [])}


def cmd_info(args):
    path = args.file
    if not os.path.isfile(path):
        die('input file does not exist: %s' % path, 2)
    fmt = format_of(path)
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        rep = info_report(path, args.scale)
        if not (rep['solids'] or rep['surfaces'] or rep['discrete']):
            die('no entities in %s (format %s)' % (path, fmt), 2)
    except Exception as e:
        die('%s import failed: %s' % (LOAD.get('format', fmt), e), 1)
    finally:
        gmsh.finalize()
    if args.json:
        with open(args.json, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(rep, f, indent=1)
            f.write('\n')
    print_info(rep)
    return 0


def cmd_export(args):
    path, out = args.file, args.out
    if not os.path.isfile(path):
        die('input file does not exist: %s' % path, 2)
    fmt = format_of(path)
    fmt_out = format_of(out)
    if fmt_out == 'iges':
        die('IGES imports as surfaces only; the tool does not write it', 2)
    if fmt_out not in EXPORTABLE:    # unreachable while FORMATS stands; a guard
        die('the tool does not write %s files' % fmt_out, 2)
    if os.path.normcase(os.path.abspath(out)) == os.path.normcase(os.path.abspath(path)):
        die('--out %s is the input file; refusing to overwrite it' % out, 2)
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        try:
            tags = load_model(path, args.scale)
        except Exception as e:
            die('%s import failed: %s' % (LOAD.get('format', fmt), e), 1)
        if not tags:
            die('export needs solids; %s holds none (format %s)'
                % (os.path.basename(path), LOAD['format']), 2)
        sc = read_sidecar(path)
        names = attach_names(sc, tags)
        doc = sidecar_for(out, tags, names, (sc or {}).get('ops'))
        if sc:                       # unknown top-level keys ride along untouched
            known = {'version', 'tool', 'units', 'scale', 'file', 'solids', 'ops'}
            for k in sc:
                if k not in known:
                    doc[k] = sc[k]
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        export_model(out, tags, names, args.stl_size)
        if fmt_out != 'stl':         # nothing reads names off a discrete surface
            write_sidecar(sidecar_path(out), doc)
    except Exception as e:
        die('export failed: %s' % e, 1)
    finally:
        gmsh.finalize()
    log('geom_tool export %s: wrote %s (format %s)'
        % (os.path.basename(path), out, fmt_out))
    return 0


def parse_args(argv):
    p = argparse.ArgumentParser(
        prog='geom_tool',
        description="List and export geometry files on gmsh's OpenCASCADE kernel.")
    sub = p.add_subparsers(dest='cmd', required=True, metavar='{info,export,edit}')
    pi = sub.add_parser('info', help='list the solids (or IGES surfaces / STL faces) of a file')
    pi.add_argument('file', help='a .step/.stp/.brep/.iges/.igs/.xao/.stl file')
    pi.add_argument('--json', metavar='OUT',
                    help='write the info document to this file, never stdout')
    pi.add_argument('--scale', type=float, default=None,
                    help='file units -> metres (default: STEP/IGES 0.001, others 1.0)')
    pe = sub.add_parser('export', help='write the model to another format, with the sidecar')
    pe.add_argument('file', help='the file to read')
    pe.add_argument('--out', required=True, help='output path; the extension picks the format')
    pe.add_argument('--scale', type=float, default=None, help='file units -> metres')
    pe.add_argument('--stl-size', type=float, default=None, dest='stl_size',
                    help='surface mesh size in metres for an STL export (default D/20)')
    pd = sub.add_parser('edit', help='apply the named operations of an ops file, write the result')
    pd.add_argument('file', help='the file to read (.step/.stp/.brep/.xao)')
    pd.add_argument('--ops', required=True, help='the JSON file of operations')
    pd.add_argument('--out', required=True, help='output path; the extension picks the format')
    pd.add_argument('--scale', type=float, default=None,
                    help='file units -> metres (default: STEP/IGES 0.001, others 1.0)')
    return p.parse_args(argv)


def main(argv=None):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    args = parse_args(argv)
    if args.cmd == 'info':
        return cmd_info(args)
    if args.cmd == 'edit':
        import geom_edit
        return geom_edit.run_edit(args, sys.modules[__name__])
    return cmd_export(args)


if __name__ == '__main__':
    sys.exit(main())
