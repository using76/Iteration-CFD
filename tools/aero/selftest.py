#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.

"""selftest.py - builds the two-cell polyMesh fixture and runs drag_post.py on it; --keep keeps the scratch dir."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

DRAG_POST = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'drag_post.py')

# The fixture: two hex cells, cell 0 = [0,1]x[0,1]x[0,1] and cell 1 =
# [0,1]x[1,3]x[0,1]. Each face's point order gives Newell's normal OUT of its
# owner, so f1/f2/f3 carry the car's Sf_x = -1/-2/0 and f4/f5 the outlet's
# Sf_x = 1/2 - a wrong sign or area moves every closed form below.
POINTS = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1),
          (1, 1, 1), (0, 1, 1), (0, 3, 0), (1, 3, 0), (1, 3, 1), (0, 3, 1)]
FACES = [(2, 3, 7, 6), (0, 4, 7, 3), (3, 7, 11, 8), (0, 3, 2, 1), (1, 2, 6, 5),
         (2, 9, 10, 6), (0, 1, 5, 4), (8, 11, 10, 9), (4, 5, 6, 7), (3, 8, 9, 2),
         (7, 6, 10, 11)]
OWNERS = [0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1]
NEIGHBOURS = [1]
PATCHES = [('car', 'wall', 3, 1), ('outlet', 'patch', 2, 4), ('walls', 'wall', 5, 6)]

RULE = '// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n'
FOOTER = '\n\n// ************************************************************************* //\n'


def foam_header(cls, location, obj):
    return ('FoamFile\n{\n    version     2.0;\n    format      ascii;\n'
            '    class       %s;\n    location    "%s";\n    object      %s;\n}\n%s'
            % (cls, location, obj, RULE))


def write(path, text):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)


def write_polymesh(case):
    """constant/polyMesh: points, faces, owner, neighbour, boundary, in the
    layout src/io/polymesh.rs writes - the header words are not parsed, the
    counted lists are."""
    pm = os.path.join(case, 'constant', 'polyMesh')
    os.makedirs(pm)
    write(os.path.join(pm, 'points'),
          foam_header('vectorField', 'constant/polyMesh', 'points')
          + '12\n(\n' + ''.join('(%s %s %s)\n' % p for p in POINTS) + ')' + FOOTER)
    write(os.path.join(pm, 'faces'),
          foam_header('faceList', 'constant/polyMesh', 'faces')
          + '11\n(\n' + ''.join('4(%d %d %d %d)\n' % f for f in FACES) + ')' + FOOTER)
    write(os.path.join(pm, 'owner'),
          foam_header('labelList', 'constant/polyMesh', 'owner')
          + '11\n(\n' + ''.join('%d\n' % o for o in OWNERS) + ')' + FOOTER)
    write(os.path.join(pm, 'neighbour'),
          foam_header('labelList', 'constant/polyMesh', 'neighbour')
          + '%d\n(\n%s)\n' % (len(NEIGHBOURS), ''.join('%d\n' % n for n in NEIGHBOURS)) + FOOTER)
    rows = ['%d' % len(PATCHES), '(']
    for name, typ, nf, sf in PATCHES:
        rows += ['    %s' % name, '    {', '        type            %s;' % typ,
                 '        nFaces          %d;' % nf, '        startFace       %d;' % sf, '    }']
    rows.append(')')
    write(os.path.join(pm, 'boundary'),
          foam_header('polyBoundaryMesh', 'constant/polyMesh', 'boundary') + '\n'.join(rows) + '\n' + FOOTER)


def scalar_entry(v):
    """One field entry as src/io/fields.rs writes it: `uniform v` or the
    counted `nonuniform List<scalar> \n<N>\n(...)\n;` form."""
    if isinstance(v, list):
        return 'nonuniform List<scalar> \n%d\n(\n%s)\n;\n' % (len(v), ''.join('%s\n' % x for x in v))
    return 'uniform %s;\n' % v


def vector_entry(v):
    if v and isinstance(v[0], list):
        return 'nonuniform List<vector> \n%d\n(\n%s)\n;\n' % (len(v), ''.join('(%s %s %s)\n' % tuple(x) for x in v))
    return 'uniform (%s %s %s);\n' % tuple(v)


def write_field(path, obj, internal, patches):
    """One field file: dimensions, `internalField   <entry>`, then
    boundaryField with `type` and, when the patch carries one, `value`.
    patches is [(name, type, value-or-None), ...]."""
    vec = obj == 'U'
    ent = vector_entry if vec else scalar_entry
    parts = [foam_header('volVectorField' if vec else 'volScalarField',
                         os.path.basename(os.path.dirname(path)), obj),
             'dimensions      %s;\n' % ('[0 1 -1 0 0 0 0]' if vec else '[0 0 0 0 0 0 0]'),
             'internalField   %s\n' % ent(internal),
             'boundaryField\n{\n']
    for name, typ, val in patches:
        parts.append('    %s\n    {\n        type            %s;\n' % (name, typ))
        if val is not None:
            parts.append('        value           %s' % ent(val))
        parts.append('    }\n')
    parts.append('}\n')
    write(path, ''.join(parts) + FOOTER)


# The five field variants: v1 zeroGradient car (owner branch), v2/v3 value
# branches, v4 a free-stream face to skip, v5 a short patch value that must
# be refused. Closed forms at U_inf = 10 are in the tests.
VARIANTS = {
    'v1': dict(p=([100, 50], [('car', 'zeroGradient', None), ('outlet', 'fixedValue', 0), ('walls', 'zeroGradient', None)]),
               U=([[5, 0, 0], [8, 0, 0]], [('car', 'fixedValue', [0, 0, 0]), ('outlet', 'zeroGradient', None), ('walls', 'fixedValue', [0, 0, 0])])),
    'v2': dict(p=(0, [('car', 'fixedValue', [10, 20, 999]), ('outlet', 'zeroGradient', None), ('walls', 'zeroGradient', None)]),
               U=([10, 0, 0], [('car', 'zeroGradient', None), ('outlet', 'fixedValue', [[5, 0, 0], [5, 0, 0]]), ('walls', 'zeroGradient', None)])),
    'v3': dict(p=(0, [('car', 'fixedValue', 7), ('outlet', 'zeroGradient', None), ('walls', 'zeroGradient', None)]),
               U=([10, 0, 0], [('car', 'zeroGradient', None), ('outlet', 'fixedValue', [5, 0, 0]), ('walls', 'zeroGradient', None)])),
}
VARIANTS['v4'] = dict(p=VARIANTS['v1']['p'], U=([[5, 0, 0], [10, 0, 0]], VARIANTS['v1']['U'][1]))
VARIANTS['v5'] = dict(p=(0, [('car', 'fixedValue', [10, 20]), ('outlet', 'zeroGradient', None), ('walls', 'zeroGradient', None)]),
                      U=VARIANTS['v2']['U'])


def make_case(top, name, variant):
    """One variant as a runnable case in scratch dir `name`; the time dir is
    `<case>/0`."""
    case = work(top, name)
    write_polymesh(case)
    td = os.path.join(case, '0')
    os.makedirs(td)
    write_field(os.path.join(td, 'p'), 'p', VARIANTS[variant]['p'][0], VARIANTS[variant]['p'][1])
    write_field(os.path.join(td, 'U'), 'U', VARIANTS[variant]['U'][0], VARIANTS[variant]['U'][1])
    return case


def work(top, name):
    os.makedirs(os.path.join(top, name), exist_ok=True)
    return os.path.join(top, name)


def run(*args, expect=0):
    t = time.time()
    p = subprocess.run([sys.executable, DRAG_POST] + list(args), capture_output=True,
                       text=True, encoding='utf-8', errors='replace')
    print('  [%s] drag_post %s  %.1f s' % ('ok' if p.returncode == expect else 'FAIL',
                                           ' '.join(args), time.time() - t), flush=True)
    assert p.returncode == expect, (list(args), p.returncode, p.stderr[-400:])
    return p


def load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def close(got, want):
    assert abs(got - want) <= 1e-9 * max(1.0, abs(want)), (got, want)


def test_usage(top):
    p = run(expect=2)
    assert 'usage' in p.stderr, p.stderr


def test_refusals(top):
    case = make_case(top, 'ref_body', 'v1')
    p = run(case, os.path.join(case, '0'), '10', '--body', 'nose', expect=1)
    assert 'the boundary file has no patch named nose (patches: car, outlet, walls)' in p.stderr, p.stderr
    p = run(case, os.path.join(case, '0'), '10', '--outlet', 'lip', expect=1)
    assert 'the boundary file has no patch named lip' in p.stderr, p.stderr
    case = make_case(top, 'ref_short', 'v5')
    want = '%s: patch car carries 2 value(s), the patch has 3 faces' % os.path.join(case, '0', 'p')
    p = run(case, os.path.join(case, '0'), '10', expect=1)
    assert want in p.stderr, p.stderr
    case = make_case(top, 'ref_missing', 'v1')
    os.remove(os.path.join(case, '0', 'p'))
    p = run(case, os.path.join(case, '0'), '10', expect=1)
    assert ('missing file %s' % os.path.join(case, '0', 'p')) in p.stderr, p.stderr


def test_area_vectors(top):
    """The Sf column of the fixture through the closed forms - car faces
    f1/f2/f3 carry Sf_x -1/-2/0 and outlet faces f4/f5 Sf_x 1/2 - plus the
    exact shape of the --json document."""
    case = make_case(top, 'area', 'v1')
    out = os.path.join(top, 'area.json')
    p = run(case, os.path.join(case, '0'), '10', '--json', out)
    assert 'pressure drag Fx = -200.00 N' in p.stdout, p.stdout
    assert 'momentum-deficit drag = 30.10 N (wake faces: 1)' in p.stdout, p.stdout
    doc = load(out)
    assert doc['tool'] == 'drag_post'
    assert doc['case'] == case and doc['time_dir'] == os.path.join(case, '0')
    assert (doc['u_inf'], doc['rho'], doc['a_ref']) == (10.0, 1.2041, 1.6022)
    close(doc['q_pa'], 60.205)
    assert doc['band'] == {'y_max': 1.8, 'z_min': 0.3}
    assert sorted(doc) == ['a_ref', 'band', 'body', 'case', 'q_pa', 'rho', 'time_dir', 'tool', 'u_inf', 'wake']
    assert sorted(doc['body']) == ['cd', 'faces', 'fx_n', 'patch', 'source']
    assert sorted(doc['wake']) == ['cd', 'drag_n', 'faces', 'patch', 'source', 'used']
    assert (doc['body']['patch'], doc['body']['faces'], doc['body']['source']) == ('car', 3, 'owner')
    close(doc['body']['fx_n'], -200.0)
    close(doc['body']['cd'], -200.0 / (60.205 * 1.6022))
    assert (doc['wake']['patch'], doc['wake']['faces'], doc['wake']['used'], doc['wake']['source']) == ('outlet', 2, 1, 'owner')
    close(doc['wake']['drag_n'], 30.1025)
    close(doc['wake']['cd'], 30.1025 / (60.205 * 1.6022))


def test_drag_closed_form(top):
    """V1-V3 with every default: the pressure integral (owner branch and
    both value branches) and the wake survey against their closed forms."""
    for name, fx, src_p, src_u in (('v1', -200.0, 'owner', 'owner'),
                                   ('v2', -50.0, 'value', 'value'),
                                   ('v3', -21.0, 'value', 'value')):
        case = make_case(top, 'cf_' + name, name)
        out = os.path.join(top, 'cf_' + name + '.json')
        run(case, os.path.join(case, '0'), '10', '--json', out)
        doc = load(out)
        assert doc['body']['source'] == src_p
        assert doc['wake']['source'] == src_u
        close(doc['body']['fx_n'], fx)
        close(doc['body']['cd'], fx / (60.205 * 1.6022))
        close(doc['wake']['drag_n'], 30.1025)
        assert doc['wake']['used'] == 1


def test_band_and_skip(top):
    """--band 3 0 widens the survey to both outlet faces; --rho rescales it;
    V4's second face sits at free stream and must be skipped, not counted."""
    case = make_case(top, 'band_v1', 'v1')
    out = os.path.join(top, 'band_v1.json')
    run(case, os.path.join(case, '0'), '10', '--band', '3', '0', '--json', out)
    doc = load(out)
    assert doc['band'] == {'y_max': 3.0, 'z_min': 0.0}
    close(doc['wake']['drag_n'], 68.6337)
    assert doc['wake']['used'] == 2
    case = make_case(top, 'band_v2', 'v2')
    out = os.path.join(top, 'band_v2.json')
    run(case, os.path.join(case, '0'), '10', '--band', '3', '0', '--json', out)
    close(load(out)['wake']['drag_n'], 90.3075)
    case = make_case(top, 'band_v2_rho', 'v2')
    out = os.path.join(top, 'band_v2_rho.json')
    run(case, os.path.join(case, '0'), '10', '--rho', '1', '--band', '3', '0', '--json', out)
    doc = load(out)
    assert doc['rho'] == 1.0 and doc['q_pa'] == 50.0
    close(doc['wake']['drag_n'], 75.0)
    assert doc['wake']['used'] == 2
    case = make_case(top, 'skip_v4', 'v4')
    out = os.path.join(top, 'skip_v4.json')
    run(case, os.path.join(case, '0'), '10', '--band', '3', '0', '--json', out)
    doc = load(out)
    close(doc['wake']['drag_n'], 30.1025)
    assert doc['wake']['used'] == 1


TESTS = (test_usage, test_refusals, test_area_vectors, test_drag_closed_form, test_band_and_skip)


def main(argv=None):
    ap = argparse.ArgumentParser(description='Self-test drag_post: build the fixture, run, assert.')
    ap.add_argument('--keep', action='store_true', help='keep the scratch directory')
    args = ap.parse_args(argv)
    top = tempfile.mkdtemp(prefix='aero_selftest_')
    for t in TESTS:
        try:
            t(top)
        except AssertionError as e:
            print('SELFTEST FAIL: %s\nthe scratch dir is kept: %s' % (e, top))
            return 1
        print('[ok] %s' % t.__name__, flush=True)
    print('SELFTEST PASS')
    if not args.keep:
        shutil.rmtree(top, ignore_errors=True)
    return 0


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.exit(main())
