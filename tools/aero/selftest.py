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


# ---------------------------------------------------------------- fast_patch

FAST_PATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fast_patch.py')


def run_fast(*args, expect=0):
    t = time.time()
    p = subprocess.run([sys.executable, FAST_PATCH] + list(args), capture_output=True,
                       text=True, encoding='utf-8', errors='replace')
    print('  [%s] fast_patch %s  %.1f s' % ('ok' if p.returncode == expect else 'FAIL',
                                            ' '.join(args), time.time() - t), flush=True)
    assert p.returncode == expect, (list(args), p.returncode, p.stderr[-400:])
    return p


def read_text(path):
    with open(path, encoding='utf-8', newline='') as f:
        return f.read()


def list_scalar(vals):
    """A counted scalar list even at length 1 - the writer collapses a single
    entry to `uniform`, and the swallow test needs the list form to survive."""
    return 'nonuniform List<scalar> \n%d\n(\n%s)\n;\n' % (len(vals), ''.join('%s\n' % v for v in vals))


def fast_field(obj, internal, patches):
    """One 0/ field file in the writer's layout: a third keyword (inletValue)
    where a patch carries one, nonuniform lists left uncollapsed."""
    vec = obj == 'U'
    ent = vector_entry if vec else scalar_entry
    parts = [foam_header('volVectorField' if vec else 'volScalarField', '0', obj),
             'dimensions      %s;\n' % ('[0 1 -1 0 0 0 0]' if vec else '[0 0 0 0 0 0 0]'),
             'internalField   %s\n' % ent(internal),
             'boundaryField\n{\n']
    for name, typ, *kvs in patches:
        parts.append('    %s\n    {\n        type            %s;\n' % (name, typ))
        for key, val in kvs:
            parts.append('        %-16s%s' % (key, val if isinstance(val, str) else ent(val)))
        parts.append('    }\n')
    parts.append('}\n')
    return ''.join(parts) + FOOTER


def write_fvsolution(path):
    """system/fvSolution, byte for byte as src/blockgen.rs write_system writes
    it - the Rust literal's \\x20 is one space, keywords padded to 16 columns."""
    def block(fld, prec, tol, rel, it):
        return ('    %s\n    {\n        solver          PBiCGStab;\n'
                '        preconditioner  %s;\n        tolerance       %s;\n'
                '        relTol          %s;\n        maxIter         %s;\n    }\n\n'
                % (fld, prec, tol, rel, it))
    blocks = [block('p', 'DIC', '1e-08', '0.01', '1000'),
              block('Phi', 'DIC', '1e-12', '0', '5000'),
              block('U', 'diagonal', '1e-08', '0.1', '200'),
              block('T', 'diagonal', '1e-08', '0.01', '200'),
              block('k', 'diagonal', '1e-08', '0.01', '200'),
              block('epsilon', 'diagonal', '1e-08', '0.01', '200'),
              block('omega', 'diagonal', '1e-08', '0.01', '200'),
              block('nuTilda', 'diagonal', '1e-08', '0.01', '200')]
    body = ('solvers\n{\n' + ''.join(blocks[:-1]) + blocks[-1][:-1] + '}\n\n'
            'SIMPLE\n{\n    nNonOrthogonalCorrectors 0;\n}\n\n'
            "// U 0.7 with p 0.3 is what OpenFOAM's buoyant cases use, and the two\n"
            '// summing to one is the usual rule of thumb behind it.\n'
            'relaxationFactors\n{\n    fields\n    {\n        p               0.3;\n    }\n\n'
            '    equations\n    {\n'
            '        U               0.7;\n        T               0.7;\n        k               0.7;\n'
            '        epsilon         0.7;\n        omega           0.7;\n        nuTilda         0.7;\n    }\n}')
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write(foam_header('dictionary', 'system', 'fvSolution') + body + FOOTER)


def write_fast_case(top, name):
    """The fixture in the generator's layout - no polyMesh, fast_patch never
    reads one: two-cell fields, physicalProperties and fvSolution. Written LF
    (newline='') so the counted-list assertion and the refusal byte-compare
    stay exact."""
    case = work(top, name)
    for d in ('0', 'constant', 'system'):
        os.makedirs(os.path.join(case, d))

    def put(rel, text):
        with open(os.path.join(case, *rel.split('/')), 'w', encoding='utf-8', newline='') as f:
            f.write(text)

    put('0/U', fast_field('U', [[1, 0, 0], [2, 0, 0]],
                          [('inlet', 'fixedValue', ('value', [1, 0, 0])),
                           ('outlet', 'inletOutlet', ('inletValue', [0, 0, 0]), ('value', [1, 0, 0])),
                           ('car', 'noSlip'), ('walls', 'noSlip')]))
    put('0/p', fast_field('p', [3, 4],
                          [('inlet', 'zeroGradient'),
                           ('outlet', 'fixedValue', ('value', list_scalar([0]))),
                           ('car', 'zeroGradient'), ('walls', 'zeroGradient')]))
    put('0/rho', fast_field('rho', 1.2, [('inlet', 'zeroGradient'), ('outlet', 'zeroGradient'),
                                         ('car', 'zeroGradient'), ('walls', 'zeroGradient')]))
    put('0/k', fast_field('k', 0.375,
                          [('inlet', 'fixedValue', ('value', 0.375)),
                           ('outlet', 'inletOutlet', ('inletValue', 0.375), ('value', 0.375)),
                           ('car', 'kqRWallFunction', ('value', 0.375)),
                           ('walls', 'kqRWallFunction', ('value', 0.375))]))
    put('0/epsilon', fast_field('epsilon', 0.1,
                                [('inlet', 'fixedValue', ('value', 0.1)),
                                 ('outlet', 'inletOutlet'),
                                 ('car', 'epsilonWallFunction', ('value', 0.1)),
                                 ('walls', 'epsilonWallFunction', ('value', 0.1))]))
    put('constant/physicalProperties',
        foam_header('dictionary', 'constant', 'physicalProperties')
        + 'viscosityModel  constant;\n\nnu              [0 2 -1 0 0 0 0] 1e-05;\n' + FOOTER)
    write_fvsolution(os.path.join(case, 'system', 'fvSolution'))
    return case


def test_fast_usage(top):
    p = run_fast(expect=2)
    assert 'usage' in p.stderr, p.stderr


def test_fast_patch_counts(top):
    """The fixture run: every recorded count in the original's order, the
    epsilon kqRWallFunction row at exactly 0 without failing, 0/rho gone and
    every expected text in the patched files (the 0/p outlet's counted value
    is the swallow test - it must survive the re.S internalField rewrite)."""
    case = write_fast_case(top, 'fast_counts')
    p = run_fast(case)
    want = [('0/U', 'internalField', 1), ('0/U', 'inlet', 1),
            ('0/p', 'internalField', 1),
            ('0/k', 'internalField', 1), ('0/k', 'inlet', 1), ('0/k', 'kqRWallFunction', 2),
            ('0/epsilon', 'internalField', 1), ('0/epsilon', 'inlet', 1),
            ('0/epsilon', 'kqRWallFunction', 0),
            ('constant/physicalProperties', 'nu', 1),
            ('system/fvSolution', 'PBiCGStab tol', 7), ('system/fvSolution', 'p relTol', 1),
            ('system/fvSolution', 'p maxIter', 1), ('system/fvSolution', 'relax p', 1),
            ('system/fvSolution', 'relax U', 1)]
    got = [ln for ln in p.stdout.splitlines() if 'hit(s)' in ln]
    assert got == ['fast_patch: %s %s: %d hit(s)' % r for r in want], (got, p.stdout)
    assert 'fast_patch: 0/epsilon kqRWallFunction: 0 hit(s)' in p.stdout, p.stdout
    assert p.stdout.rstrip().endswith('fast-mode patch done: relTol 0.05, p maxIter 200, U=83.3333 m/s')
    assert not os.path.isfile(os.path.join(case, '0', 'rho'))
    checks = [('0/U', 'internalField   uniform (83.3333 0 0);'),
              ('0/U', 'value           uniform (83.3333 0 0);'),
              ('0/U', 'inletValue      uniform (0 0 0);'),
              ('0/p', 'internalField   uniform 0;'),
              ('0/p', 'nonuniform List<scalar> \n1\n(\n0\n)\n;'),
              ('0/k', 'internalField   uniform 4.1667;'),
              ('0/epsilon', 'internalField   uniform 18.632;'),
              ('0/epsilon', 'value           uniform 18.632;'),
              ('constant/physicalProperties', 'nu              [0 2 -1 0 0 0 0] 1.5e-05;')]
    for rel, needle in checks:
        assert needle in read_text(os.path.join(case, *rel.split('/'))), rel
    k = read_text(os.path.join(case, '0', 'k'))
    assert k.count('value           uniform 4.1667;') == 3, k
    e = read_text(os.path.join(case, '0', 'epsilon'))
    assert 'epsilonWallFunction' in e and e.count('value           uniform 0.1;') == 2, e
    fvs = read_text(os.path.join(case, 'system', 'fvSolution'))
    assert fvs.count('tolerance       1e-06;') == 7 and 'tolerance       1e-08;' not in fvs, fvs
    assert 'tolerance       1e-12;\n        relTol          0;\n        maxIter         5000;' in fvs, fvs
    assert 'tolerance       1e-06;\n        relTol          0.05;\n        maxIter         200;' in fvs, fvs
    assert 'tolerance       1e-06;\n        relTol          0.1;' in fvs, fvs
    assert 'p               0.25;' in fvs and 'U               0.5;' in fvs and 'T               0.7;' in fvs, fvs


def test_fast_refusal_untouched(top):
    """A required row with no hit refuses by name and writes nothing: every
    file's bytes are what they were, and the stale rho survives. A missing
    0/U refuses with the path as given."""
    case = write_fast_case(top, 'fast_refuse')
    names = ('0/U', '0/p', '0/rho', '0/k', '0/epsilon',
             'constant/physicalProperties', 'system/fvSolution')
    u = os.path.join(case, '0', 'U')
    renamed = read_text(u).replace('    inlet\n', '    inflow\n')
    with open(u, 'w', encoding='utf-8', newline='') as f:
        f.write(renamed)
    before = {n: read_text(os.path.join(case, *n.split('/'))) for n in names}
    p = run_fast(case, expect=1)
    assert 'fast_patch: no match in 0/U for inlet' in p.stderr, p.stderr
    for n in names:
        assert read_text(os.path.join(case, *n.split('/'))) == before[n], n
    assert os.path.isfile(os.path.join(case, '0', 'rho'))
    empty = work(top, 'fast_missing')
    p = run_fast(empty, expect=1)
    assert 'fast_patch: missing file %s' % os.path.join(empty, '0', 'U') in p.stderr, p.stderr


def test_fast_idempotent(top):
    """The second run exits 0: the rows that match only the un-patched
    literals accept their own output, and rho stays gone."""
    case = write_fast_case(top, 'fast_twice')
    run_fast(case)
    assert not os.path.isfile(os.path.join(case, '0', 'rho'))
    p = run_fast(case)
    assert 'fast_patch: system/fvSolution PBiCGStab tol: 0 hit(s)' in p.stdout, p.stdout
    assert 'fast_patch: system/fvSolution p maxIter: 0 hit(s)' in p.stdout, p.stdout
    assert not os.path.isfile(os.path.join(case, '0', 'rho'))


TESTS = (test_usage, test_refusals, test_area_vectors, test_drag_closed_form, test_band_and_skip,
         test_fast_usage, test_fast_patch_counts, test_fast_refusal_untouched, test_fast_idempotent)


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
