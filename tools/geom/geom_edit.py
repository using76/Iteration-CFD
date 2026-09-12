#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.

"""geom_edit - the edit side of geom_tool (docs/10-fsi-solid-mesh-plan.md §I M2).

    geom_tool.py edit <file> --ops ops.json --out <path> [--scale S]

applies fourteen named operations on gmsh's OpenCASCADE kernel and writes the
result by extension through the M1 export plus its sidecar, the op history
appended. geom_tool.py imports this module inside its `edit` branch and hands
it its own module object as `api`; this module never imports geom_tool (the
tool runs as __main__, and a second import would load a second copy with its
own LOAD).

Names survive OCC's tag renumbering two ways, and the distinction is the
point of this module: WITHIN a run names ride gmsh's out_map, because a
boolean's products have new centroids and volumes no matcher could recognise;
ACROSS files they are matched by M1's sidecar rule (centroid + volume, then
tag). No second matcher lives here.
"""

import difflib
import json
import math
import os
import re

import gmsh

OP_NAMES = ('rename', 'set_material', 'delete', 'translate', 'rotate', 'scale', 'mirror',
            'fuse', 'cut', 'intersect', 'fragment', 'box', 'cylinder', 'sphere')
NAME_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
# per op: required and optional keys, each with the kind _check_kind enforces:
# 'ref'/'refs' a solid name or tag, 'name' a NAME_RE identifier, 'material'
# a non-empty string or null, 'vec3'/'vec3pos' three numbers (each positive
# for vec3pos), 'vec3nonzero' a direction (not all zero), 'vec3nz' three
# scale factors (each non-zero), 'plane' [a, b, c, d] with (a, b, c) not all
# zero, 'pos'/'nonzero'/'num' single numbers
OPS = {
    'rename': {'required': {'solid': 'ref', 'name': 'name'}, 'optional': {}},
    'set_material': {'required': {'solid': 'ref', 'material': 'material'}, 'optional': {}},
    'delete': {'required': {'solids': 'refs'}, 'optional': {}},
    'translate': {'required': {'solids': 'refs', 'by': 'vec3'}, 'optional': {}},
    'rotate': {'required': {'solids': 'refs', 'point': 'vec3',
                            'axis': 'vec3nonzero', 'angle_deg': 'num'}, 'optional': {}},
    'scale': {'required': {'solids': 'refs', 'point': 'vec3'},
              'optional': {'factor': 'nonzero', 'factors': 'vec3nz'}},
    'mirror': {'required': {'solids': 'refs', 'plane': 'plane'}, 'optional': {}},
    'fuse': {'required': {'object': 'refs', 'tools': 'refs'}, 'optional': {'name': 'name'}},
    'cut': {'required': {'object': 'refs', 'tools': 'refs'}, 'optional': {'name': 'name'}},
    'intersect': {'required': {'object': 'refs', 'tools': 'refs'},
                  'optional': {'name': 'name'}},
    'fragment': {'required': {'object': 'refs', 'tools': 'refs'}, 'optional': {}},
    'box': {'required': {'name': 'name', 'origin': 'vec3', 'size': 'vec3pos'},
            'optional': {'material': 'material'}},
    'cylinder': {'required': {'name': 'name', 'origin': 'vec3',
                              'axis': 'vec3nonzero', 'radius': 'pos'},
                 'optional': {'material': 'material'}},
    'sphere': {'required': {'name': 'name', 'centre': 'vec3', 'radius': 'pos'},
               'optional': {'material': 'material'}},
}


class OpsError(Exception):
    """A refusal before anything is written: the message is already in the
    'ops[k].<key>: ...' shape; run_edit maps it to exit 2."""


def _is_num(v):
    # JSON booleans are ints in Python; step_mesh.py:183's rule excludes them
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _check_kind(k, key, v, kind):
    def bad(expect):
        raise OpsError('ops[%d].%s: expected %s, got %r' % (k, key, expect, v))
    if kind == 'ref':
        if not (isinstance(v, str) or isinstance(v, int) and not isinstance(v, bool)):
            bad('a solid name (string) or tag (int)')
    elif kind == 'refs':
        if not (isinstance(v, list) and v and
                all(isinstance(r, str) or isinstance(r, int) and not isinstance(r, bool)
                    for r in v)):
            bad('a non-empty list of solid names or tags')
    elif kind == 'name':
        if not (isinstance(v, str) and NAME_RE.match(v)):
            bad('a name matching ^[A-Za-z_][A-Za-z0-9_]*$')
    elif kind == 'material':
        if v is not None and not (isinstance(v, str) and v):
            bad('a non-empty string, or null to clear')
    elif kind == 'vec3':
        if not (isinstance(v, list) and len(v) == 3 and all(_is_num(x) for x in v)):
            bad('3 numbers')
    elif kind == 'vec3pos':
        if not (isinstance(v, list) and len(v) == 3 and all(_is_num(x) and x > 0 for x in v)):
            bad('3 numbers > 0')
    elif kind == 'vec3nonzero':
        # a direction: the vector as a whole must be non-zero - rotate's
        # [0, 0, 1] and the cut cylinder's [0, 0, 3] carry zero components
        if not (isinstance(v, list) and len(v) == 3 and all(_is_num(x) for x in v)
                and any(x != 0 for x in v)):
            bad('3 numbers, not all zero')
    elif kind == 'vec3nz':
        # scale factors: every component is itself a scale and must be non-zero
        if not (isinstance(v, list) and len(v) == 3 and all(_is_num(x) and x != 0 for x in v)):
            bad('3 non-zero numbers')
    elif kind == 'plane':
        if not (isinstance(v, list) and len(v) == 4 and all(_is_num(x) for x in v)
                and any(x != 0 for x in v[:3])):
            bad('4 numbers [a, b, c, d] with (a, b, c) not all zero')
    elif kind == 'pos':
        if not (_is_num(v) and v > 0):
            bad('a number > 0')
    elif kind == 'nonzero':
        if not (_is_num(v) and v != 0):
            bad('a non-zero number')
    else:   # 'num'
        if not _is_num(v):
            bad('a number')


def validate_ops(doc):
    """The C3 structural checks of the whole ops file, run before the model
    file is opened; every failure is an OpsError naming ops[k].<key>."""
    if not isinstance(doc, dict):
        raise OpsError('the ops file must hold a JSON object, got %r' % (doc,))
    for key in doc:
        if key not in ('version', 'ops'):
            near = difflib.get_close_matches(key, ['version', 'ops'], 1)
            raise OpsError('ops file: unknown key "%s"%s'
                           % (key, ' - did you mean "%s"?' % near[0] if near else ''))
    version = doc.get('version')
    if isinstance(version, bool) or version != 1:   # True == 1 in Python
        raise OpsError('ops file: "version" must be 1, got %r' % (version,))
    ops = doc.get('ops')
    if not isinstance(ops, list) or not ops:
        raise OpsError('ops file: "ops" must be a non-empty list')
    for k, op in enumerate(ops):
        if not isinstance(op, dict):
            raise OpsError('ops[%d]: expected an object, got %r' % (k, op))
        what = op.get('op')
        if not isinstance(what, str) or what not in OP_NAMES:
            raise OpsError('ops[%d].op: unknown operation %r; the operations are: %s'
                           % (k, what, ', '.join(OP_NAMES)))
        spec = OPS[what]
        keys = list(spec['required']) + list(spec['optional'])
        for key in op:
            if key == 'op' or key in spec['required'] or key in spec['optional']:
                continue
            near = difflib.get_close_matches(key, keys, 1)
            raise OpsError('ops[%d].%s: unknown key for %s%s'
                           % (k, key, what, ' - did you mean "%s"?' % near[0] if near else ''))
        for key in spec['required']:
            if key not in op:
                raise OpsError('ops[%d].%s: missing; %s needs it' % (k, key, what))
        for key, kind in list(spec['required'].items()) + list(spec['optional'].items()):
            if key in op:
                _check_kind(k, key, op[key], kind)
        if what == 'scale' and ('factor' in op) == ('factors' in op):
            raise OpsError('ops[%d]: scale needs exactly one of "factor" and "factors"' % k)
    return ops


def resolve(ref, names, k, key):
    """A name (against the registry AS IT STANDS at this op) or a tag -> tag."""
    if isinstance(ref, str):
        for t, rec in names.items():
            if rec['name'] == ref:
                return t
    elif ref in names:
        return ref
    raise OpsError('ops[%d].%s: no solid %r; the names that exist: %s'
                   % (k, key, ref,
                      ', '.join(sorted(rec['name'] for rec in names.values())) or '(none)'))


def resolve_list(refs, names, k, key):
    tags = [resolve(r, names, k, key) for r in refs]
    if len(set(tags)) != len(tags):
        raise OpsError('ops[%d].%s: the list repeats a solid' % (k, key))
    return tags


def check_new_name(name, names, keep, k):
    """Names are unique at all times: `name` must not belong to a solid
    outside `keep` (the operands, whose records are about to be popped)."""
    for t, rec in names.items():
        if t not in keep and rec['name'] == name:
            raise OpsError('ops[%d]: the name "%s" is already taken by solid %d'
                           % (k, name, t))


def numbered(name, n):
    """C5's one suffix rule: n == 1 keeps the bare name, n > 1 becomes
    name_1 .. name_n in out order - every piece suffixed, no bare first."""
    return [name] if n == 1 else ['%s_%d' % (name, i) for i in range(1, n + 1)]


def _dimtags(tags):
    return [(3, t) for t in tags]


def _guard_tags(k, what, names):
    """After every boolean: the registry and the model must hold the SAME
    solids - a bystander whose tag OCC lost is a bug, never a silent re-match
    by centroid (a boolean's products have new centroids anyway)."""
    model = {t for d, t in gmsh.model.getEntities(3)}
    if set(names) != model:
        raise RuntimeError('the registry holds tags %s but the model holds %s'
                           % (sorted(names), sorted(model)))
    if not names:
        raise RuntimeError('the edit leaves no solid')


def _dedupe(raw):
    """Two fragment outputs with the same joined name (a disconnected overlap)
    get _1, _2, ... in out order - EVERY one suffixed, none left bare (C5)."""
    counts = {}
    for _t, nm, _m in raw:
        counts[nm] = counts.get(nm, 0) + 1
    seen = {}
    out = []
    for t, nm, m in raw:
        if counts[nm] > 1:
            seen[nm] = seen.get(nm, 0) + 1
            nm = '%s_%d' % (nm, seen[nm])
        out.append((t, nm, m))
    return out


def _apply_boolean(k, op, names):
    """fuse / cut / intersect / fragment, with C5's naming: the inputs are the
    object list then the tools list (= gmsh's out_map order), every operand's
    record is popped first, and the products are named from out and out_map."""
    what = op['op']
    obj = resolve_list(op['object'], names, k, 'object')
    tools = resolve_list(op['tools'], names, k, 'tools')
    shared = sorted(set(obj) & set(tools))
    if shared:
        raise OpsError('ops[%d]: object and tools are not disjoint (%s)'
                       % (k, ', '.join(names[t]['name'] for t in shared)))
    inputs = obj + tools
    in_names = [names[t]['name'] for t in inputs]
    in_mats = [names[t]['material'] for t in inputs]
    rec_in = [[t, nm] for t, nm in zip(inputs, in_names)]
    base = op.get('name') or in_names[0]
    if 'name' in op:            # the check possible before the run: the bare name
        check_new_name(base, names, set(inputs), k)
    for t in inputs:
        del names[t]
    fn = getattr(gmsh.model.occ, what)   # fuse / cut / intersect / fragment
    out, out_map = fn(_dimtags(obj), _dimtags(tools))
    out = [t for d, t in out]
    gmsh.model.occ.synchronize()
    if not out:
        raise RuntimeError('produced no solid')
    if what == 'fragment':
        # S(o) = the inputs whose out_map entry lists o; the overlap piece is in
        # BOTH lists, so its name joins both contributors in input order
        raw = []
        for t in out:
            contrib = [i for i, lst in enumerate(out_map) if (3, t) in lst]
            if not contrib:
                raise RuntimeError('output %d appears in no out_map entry' % t)
            mat = next((in_mats[i] for i in contrib if in_mats[i] is not None), None)
            raw.append((t, '_'.join(in_names[i] for i in contrib), mat))
    else:
        # the op's name or the first object's name; a cut that splits its
        # object gives base_1 .. base_n (a cut's tool out_map entry is junk)
        raw = [(t, nm, in_mats[0]) for t, nm in zip(out, numbered(base, len(out)))]
    named = _dedupe(raw)
    for _t, nm, _m in named:    # bystanders only: the operands are popped
        check_new_name(nm, names, set(), k)
    for t, nm, mat in named:
        names[t] = {'name': nm, 'material': mat, 'matched': 'op'}
    _guard_tags(k, what, names)
    return {'op': op, 'in': rec_in, 'out': [[t, nm] for t, nm, _m in named]}


def apply_op(k, op, names):
    """One op per C4's call table, in place on `names`; returns the C6 history
    entry. OpsError -> exit 2; RuntimeError -> exit 1 (run_edit formats both)."""
    what = op['op']
    if what == 'rename':
        t = resolve(op['solid'], names, k, 'solid')
        check_new_name(op['name'], names, {t}, k)
        old = names[t]['name']
        names[t]['name'] = op['name']
        # a courtesy for XAO/info, not the store: a STEP keeps no entity names
        gmsh.model.setEntityName(3, t, op['name'])
        return {'op': op, 'in': [[t, old]], 'out': [[t, op['name']]]}
    if what == 'set_material':
        t = resolve(op['solid'], names, k, 'solid')
        names[t]['material'] = op['material']
        nm = names[t]['name']
        return {'op': op, 'in': [[t, nm]], 'out': [[t, nm]]}
    if what == 'delete':
        tags = resolve_list(op['solids'], names, k, 'solids')
        rec_in = [[t, names[t]['name']] for t in tags]
        # recursive: a neighbour's shared face survives (probed 2026-09-12)
        gmsh.model.occ.remove(_dimtags(tags), recursive=True)
        gmsh.model.occ.synchronize()
        for t in tags:
            del names[t]
        _guard_tags(k, what, names)
        return {'op': op, 'in': rec_in, 'out': []}
    if what in ('translate', 'rotate', 'scale', 'mirror'):
        tags = resolve_list(op['solids'], names, k, 'solids')
        dt = _dimtags(tags)
        occ = gmsh.model.occ
        if what == 'translate':
            occ.translate(dt, *op['by'])
        elif what == 'rotate':
            # the ops file carries degrees; OCC's rotate takes radians
            occ.rotate(dt, *op['point'], *op['axis'], math.radians(op['angle_deg']))
        elif what == 'scale':
            if 'factor' in op:
                occ.dilate(dt, *op['point'], op['factor'], op['factor'], op['factor'])
            else:
                occ.dilate(dt, *op['point'], *op['factors'])
        else:
            occ.mirror(dt, *op['plane'])
        occ.synchronize()
        rec = [[t, names[t]['name']] for t in tags]   # transforms keep tags (probed)
        _guard_tags(k, what, names)
        return {'op': op, 'in': rec, 'out': list(rec)}
    if what in ('box', 'cylinder', 'sphere'):
        check_new_name(op['name'], names, set(), k)
        occ = gmsh.model.occ
        if what == 'box':
            t = occ.addBox(*op['origin'], *op['size'])
        elif what == 'cylinder':
            t = occ.addCylinder(*op['origin'], *op['axis'], op['radius'])
        else:
            t = occ.addSphere(*op['centre'], op['radius'])
        occ.synchronize()
        names[t] = {'name': op['name'], 'material': op.get('material'), 'matched': 'op'}
        _guard_tags(k, what, names)
        return {'op': op, 'in': [], 'out': [[t, op['name']]]}
    return _apply_boolean(k, op, names)


def run_edit(args, api):
    """geom_tool.py's `edit` branch: everything refusable is refused before
    gmsh starts, the ops apply one by one, then sidecar -> export -> sidecar
    file (C6's order: the export replaces the model, nothing reads tags after)."""
    if not os.path.isfile(args.ops):
        api.die('edit: the ops file does not exist: %s' % args.ops, 2)
    if not os.path.isfile(args.file):
        api.die('edit: input file does not exist: %s' % args.file, 2)
    try:
        with open(args.ops, encoding='utf-8') as f:
            ops = validate_ops(json.load(f))
    except OpsError as e:
        api.die('edit: %s' % e, 2)
    except (OSError, ValueError) as e:   # unreadable, or not JSON
        api.die('edit: %s is not a readable ops file: %s' % (args.ops, e), 2)
    out = args.out
    if os.path.normcase(os.path.abspath(out)) == os.path.normcase(os.path.abspath(args.file)):
        api.die('edit: --out %s is the input (same file); refusing to overwrite it' % out, 2)
    if os.path.splitext(out)[1].lower() in ('.stl', '.iges', '.igs'):
        api.die('edit: an edit output must re-import as solids; .stl/.iges/.igs '
                'do not (write .step/.brep/.xao)', 2)
    if os.path.splitext(args.file)[1].lower() in ('.stl', '.iges', '.igs'):
        api.die('edit: %s imports as surfaces only; edit needs solids'
                % os.path.basename(args.file), 2)
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        try:
            tags = api.load_model(args.file, args.scale)
        except Exception as e:
            api.die('edit: %s import failed: %s' % (os.path.basename(args.file), e), 1)
        if not tags:
            # the second net: gmsh.merge on a two-line STL succeeds with zero
            # entities, so an empty tag list here is the same surfaces refusal
            api.die('edit: %s: no solid was imported (surfaces only?)'
                    % os.path.basename(args.file), 2)
        sc = api.read_sidecar(args.file)
        names = api.attach_names(sc, tags)
        entries = []
        for k, op in enumerate(ops):
            try:
                entry = apply_op(k, op, names)
            except OpsError as e:
                api.die('edit: %s' % e, 2)
            except Exception as e:
                api.die('edit: ops[%d] (%s): %s' % (k, op['op'], e), 1)
            api.log('ops[%d] %s: %s -> %s'
                    % (k, op['op'],
                       ', '.join(nm for _t, nm in entry['in']) or '(nothing)',
                       ', '.join(nm for _t, nm in entry['out']) or '(nothing)'))
            entries.append(entry)
        doc = api.sidecar_for(out, sorted(names), names,
                              list((sc or {}).get('ops') or []) + entries)
        if sc:      # unknown top-level sidecar keys ride along, as in export
            for key in sc:
                if key not in ('version', 'tool', 'units', 'scale', 'file',
                               'solids', 'ops'):
                    doc[key] = sc[key]
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        try:
            api.export_model(out, sorted(names), names)
            api.write_sidecar(api.sidecar_path(out), doc)
        except Exception as e:
            api.die('edit: writing %s failed: %s' % (os.path.basename(out), e), 1)
    finally:
        gmsh.finalize()
    api.log('wrote %s (%d solids)' % (os.path.basename(out), len(names)))
    return 0
