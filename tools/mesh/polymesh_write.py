#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""
polymesh_write.py - the polyMesh writer and reader the region tools share.

write_polymesh emits, byte for byte, what this repository's own Rust writer
(rust/src/io/polymesh.rs, write_poly_mesh_raw) emits: the counted `N (`
list bodies, points at 17 significant digits ('%.17g', C's %.*g), the
boundary dictionary carrying type/nFaces/startFace, LF line endings.
read_polymesh reads the same five files back. A file format is not a work:
this is an independent implementation written from this repository's Rust
source, not from anyone's polyMesh writer. regions_selftest.py verifies the
bytes against rust/target/release/ofgpu-convert-mesh.
"""
import os
import re

import numpy as np

# The header of every file - byte for byte polymesh.rs's BANNER, which ends
# mid-line after "class" so the class name completes it.
BANNER = r"""/*---------------------------------------------------------------------------*\
| ofgpu  --  GPU-native finite volume CFD                                     |
|                                                                             |
| Written in the OpenFOAM ASCII case format so that existing pre- and         |
| post-processing tools can read it. A file format is not a work: ofgpu is    |
| an independent implementation, neither derived from nor affiliated with     |
| OpenFOAM.                                                                   |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       """

SEPARATOR = '// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n'
FOOTER_RULE = '// ************************************************************************* //\n'

FILES = ('points', 'faces', 'owner', 'neighbour', 'boundary')


def fmt_real(x):
    """'%.17g' - C's %.*g at 17 significant digits, polymesh.rs's fmt_g_prec(x, 17)."""
    return '%.17g' % x


def check_patch_name(name, what):
    """The three refusals of polymesh.rs's check_patch_name, as ValueError."""
    if not name:
        raise ValueError('empty %s name - a patch the boundary file carries has to be named' % what)
    for ch in name:
        if ch.isspace() or ord(ch) < 32 or 127 <= ord(ch) <= 159:
            raise ValueError("%s name '%s' carries a whitespace or control character (%r)"
                             ' - the boundary file holds each name as one bare token' % (what, name, ch))
        if ch in '{}();"\'':
            raise ValueError("%s name '%s' carries '%s', which the boundary file's patch "
                             'entry grammar reserves' % (what, name, ch))


def _header(cls, obj, note):
    s = BANNER + cls + ';\n'
    if note:
        s += '    note        "%s";\n' % note
    s += '    location    "constant/polyMesh";\n    object      %s;\n}\n' % obj
    return s + SEPARATOR + '\n'


def _write(path, body, cls, obj, note=''):
    text = _header(cls, obj, note) + body + '\n\n' + FOOTER_RULE
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)


def write_polymesh(pm_dir, points, faces, owner, neighbour, patches):
    """Write the five files of one polyMesh into pm_dir.

    points is an (n, 3) float64 array; faces a list of int lists, internal
    FIRST; owner one label per face; neighbour the n_internal_faces labels;
    patches [(name, type, n_faces)] in order, startFace derived. Returns
    {'nPoints', 'nCells', 'nFaces', 'nInternalFaces'} - the note's numbers.
    """
    os.makedirs(pm_dir, exist_ok=True)
    for name, _, _ in patches:
        check_patch_name(name, 'patch')
    n_if = len(neighbour)
    n_cells = max(list(owner) + list(neighbour)) + 1
    note = 'nPoints:%d  nCells:%d  nFaces:%d  nInternalFaces:%d' % (
        len(points), n_cells, len(faces), n_if)
    out = ['%d\n(\n' % len(points)]
    for p in points:
        out.append('(%s %s %s)\n' % (fmt_real(p[0]), fmt_real(p[1]), fmt_real(p[2])))
    _write(os.path.join(pm_dir, 'points'), ''.join(out) + ')', 'vectorField', 'points')
    out = ['%d\n(\n' % len(faces)]
    for f in faces:
        out.append('%d(%s)\n' % (len(f), ' '.join(str(v) for v in f)))
    _write(os.path.join(pm_dir, 'faces'), ''.join(out) + ')', 'faceList', 'faces')
    for obj, vals in (('owner', owner), ('neighbour', neighbour)):
        out = ['%d\n(\n' % len(vals)] + ['%d\n' % v for v in vals]
        _write(os.path.join(pm_dir, obj), ''.join(out) + ')', 'labelList', obj, note)
    out = ['%d\n(\n' % len(patches)]
    start = n_if
    for name, tname, n_f in patches:
        out.append('    %s\n    {\n        type            %s;\n' % (name, tname))
        out.append('        nFaces          %d;\n' % n_f)
        out.append('        startFace       %d;\n    }\n' % start)
        start += n_f
    _write(os.path.join(pm_dir, 'boundary'), ''.join(out) + ')', 'polyBoundaryMesh', 'boundary')
    return {'nPoints': len(points), 'nCells': n_cells, 'nFaces': len(faces),
            'nInternalFaces': n_if}


def _body(pm_dir, name):
    """The counted `N (` ... `)` list of one polyMesh file, comments stripped -
    the diag/polymesh_to_fluent.py pattern, as a function."""
    with open(os.path.join(pm_dir, name), encoding='utf-8') as f:
        t = f.read()
    t = re.sub(r'/\*.*?\*/', ' ', t, flags=re.S)
    t = re.sub(r'//[^\n]*', ' ', t)
    i = t.index('}') + 1
    m = re.search(r'(\d+)\s*\(', t[i:])
    return int(m.group(1)), t[i + m.end():t.rindex(')')]


def read_polymesh(pm_dir):
    """Read the five files back.

    Returns {'points': (n, 3) float64, 'faces': list[int lists],
    'owner': int64 array, 'neighbour': int64 array,
    'patches': [{'name', 'type', 'startFace', 'nFaces'}, ...]}.
    """
    n_pts, txt = _body(pm_dir, 'points')
    vals = np.array(txt.replace('(', ' ').replace(')', ' ').split(), dtype=np.float64)
    assert len(vals) == 3 * n_pts, (len(vals), n_pts)
    n_faces, txt = _body(pm_dir, 'faces')
    raw = np.array(txt.replace('(', ' ').replace(')', ' ').split(), dtype=np.int64)
    faces = []
    pos = 0
    while pos < len(raw):
        k = int(raw[pos])
        faces.append([int(v) for v in raw[pos + 1:pos + 1 + k]])
        pos += 1 + k
    assert len(faces) == n_faces, (len(faces), n_faces)
    n_own, txt = _body(pm_dir, 'owner')
    owner = np.array(txt.split(), dtype=np.int64)
    assert len(owner) == n_own == n_faces, (len(owner), n_own, n_faces)
    n_nei, txt = _body(pm_dir, 'neighbour')
    neighbour = np.array(txt.split(), dtype=np.int64)
    assert len(neighbour) == n_nei <= n_faces, (len(neighbour), n_nei, n_faces)
    with open(os.path.join(pm_dir, 'boundary'), encoding='utf-8') as f:
        btxt = re.sub(r'/\*.*?\*/', ' ', f.read(), flags=re.S)
    patches = [{'name': nm, 'type': tp, 'nFaces': int(nf), 'startFace': int(sf)}
               for nm, tp, nf, sf in re.findall(
                   r'\n\s*([A-Za-z_][\w]*)\s*\{\s*type\s+(\w+);[^}]*?nFaces\s+(\d+);'
                   r'\s*startFace\s+(\d+);', btxt)]
    return {'points': vals.reshape(-1, 3), 'faces': faces, 'owner': owner,
            'neighbour': neighbour, 'patches': patches}
