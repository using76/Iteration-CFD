#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""
mesh_identity.py - the identity block both mesh summaries carry.

step_mesh.py imports this to stamp every summary it writes with six keys that
name the mesh and the run: schema, mesh_id, run_id, tool, written_at, host.
`mesh_id` is deterministic - FNV-1a 64 over the case directory as configured
plus the mesh name - because the mesh at a directory is one mesh however many
times it is re-made. rust/src/automesher/identity.rs is the same algorithm in
Rust, and the published vectors both must reproduce are in the --selftest.

    python tools/mesh/mesh_identity.py --selftest

prints `mesh_identity selftest: PASS` (exit 0) or the first failed vector
(exit 1). Standard library only - no gmsh, no numpy - so it is a two-second
check.
"""
import os
import re
import sys
import time

IDENTITY_SCHEMA = 1
RUN_ID_RE = re.compile(r'^[A-Za-z0-9._-]{1,64}$')


def fnv1a64(text: str) -> int:
    """FNV-1a, 64-bit, over the UTF-8 bytes (the published two-line loop)."""
    h = 0xcbf29ce484222325
    for b in text.encode('utf-8'):
        h ^= b
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


def identity_key(case_dir: str, name: str) -> str:
    """The directory as configured, no absolutising, plus the name."""
    return case_dir.replace('\\', '/').rstrip('/') + '\n' + name


def mesh_id(case_dir: str, name: str) -> str:
    return 'm_%016x' % fnv1a64(identity_key(case_dir, name))


def is_run_id(value) -> bool:
    """`^[A-Za-z0-9._-]{1,64}$`, exact - a trailing newline is NOT inside
    the set, which is why the match is anchored at both ends via fullmatch
    and not `$` alone."""
    return isinstance(value, str) and RUN_ID_RE.fullmatch(value) is not None


def run_id_from_env(tool: str):
    """OFGPU_RUN_ID when it is a run id; None otherwise, having printed
    the warning to stderr when the variable was set but malformed."""
    v = os.environ.get('OFGPU_RUN_ID')
    if v is None:
        return None
    if is_run_id(v):
        return v
    sys.stderr.write("%s: OFGPU_RUN_ID='%s' is not a run id (1 to 64 characters "
                     "from [A-Za-z0-9._-]); the summary records run_id: null\n"
                     % (tool, v))
    sys.stderr.flush()
    return None


def utc_stamp(epoch_seconds=None) -> str:
    """`YYYY-MM-DDTHH:MM:SSZ`, UTC, seconds precision, literal Z."""
    if epoch_seconds is None:
        epoch_seconds = time.time()
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(epoch_seconds))


def host():
    """COMPUTERNAME, else HOSTNAME, else None. An empty value is None."""
    for k in ('COMPUTERNAME', 'HOSTNAME'):
        v = os.environ.get(k)
        if v:
            return v
    return None


def identity_block(tool: str, case_dir: str, name: str, run_id) -> dict:
    """C1's six keys. run_id may be None - the key is present, null-valued."""
    return {
        'schema': IDENTITY_SCHEMA,
        'mesh_id': mesh_id(case_dir, name),
        'run_id': run_id,
        'tool': tool,
        'written_at': utc_stamp(),
        'host': host(),
    }


def _selftest() -> int:
    """Every published vector of C2 and C4, vector by vector."""
    fails = []

    def check(what, got, want):
        if got != want:
            fails.append('mesh_identity selftest: FAIL %s %r != %r' % (what, got, want))

    for text, want in (('', 'cbf29ce484222325'), ('a', 'af63dc4c8601ec8c'),
                       ('abc', 'e71fa2190541574b'), ('foobar', '85944171f73967e8')):
        check('fnv1a64(%r)' % text, '%016x' % fnv1a64(text), want)
    check("mesh_id('C:/out/cube', 'cube')", mesh_id('C:/out/cube', 'cube'),
          'm_ede67abc39071c48')
    for secs, want in ((0, '1970-01-01T00:00:00Z'), (951782400, '2000-02-29T00:00:00Z'),
                       (1700000000, '2023-11-14T22:13:20Z'),
                       (1789019553, '2026-09-10T05:52:33Z')):
        check('utc_stamp(%d)' % secs, utc_stamp(secs), want)
    for good in ('r_1', 'r_221', 'a', 'A.b-c_9', 'x' * 64):
        check('is_run_id(%r)' % good, is_run_id(good), True)
    for bad in ('', 'x' * 65, 'r 1', '../x', 'a/b', 'a' + chr(92) + 'b',
                'a:b', 'a"b', 'a\nb', 'a\tb'):
        check('is_run_id(%r)' % bad, is_run_id(bad), False)
    blk = identity_block('step_mesh', 'C:/out/cube', 'cube', None)
    check('identity_block keys', sorted(blk),
          ['host', 'mesh_id', 'run_id', 'schema', 'tool', 'written_at'])
    check('identity_block run_id', blk['run_id'], None)
    check('identity_block schema', blk['schema'], 1)
    if fails:
        print(fails[0])
        return 1
    print('mesh_identity selftest: PASS')
    return 0


if __name__ == '__main__':
    sys.exit(_selftest())
