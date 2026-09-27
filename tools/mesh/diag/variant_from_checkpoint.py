#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""
variant_from_checkpoint.py - a config variant that reruns a case from its pools checkpoint.

    python tools/mesh/diag/variant_from_checkpoint.py <case.json> <variant> key.sub=value ...

Copies the checkpoint (<name>_pools.brep + .json) from <out_dir>/work of the case - the
config's own out_dir, resolved as step_mesh.py resolves it, never an assumed <case dir>/mesh -
into <case dir>/<variant>/mesh/work, writes <case dir>/<variant>/<variant>.json with the
overrides applied (dotted keys, JSON values), and prints the run_step_mesh.cmd line to run it
with --from-checkpoint. A missing checkpoint, out_dir or name is refused by name before
anything is written. Used for "does pool 0.34 pass where 0.35 failed", "what does HXT say
about the surface" (mesh.algo3d=10) and "what does the gap pass find" (sizes.gap_ratio=2.5)
without touching the case itself.
"""
import json, io, os, shutil, sys


def die(msg, code=1):
    sys.stderr.write('variant_from_checkpoint: %s\n' % msg)
    sys.stderr.flush()
    raise SystemExit(code)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    cfg_path, variant = sys.argv[1], sys.argv[2]
    d = json.load(open(cfg_path, encoding='utf-8-sig'))
    case_dir = os.path.dirname(os.path.abspath(cfg_path))
    if 'out_dir' not in d:
        die('the config %s has no "out_dir" - the checkpoint lives under <out_dir>/work'
            % cfg_path)
    if 'name' not in d:
        die('the config %s has no "name"' % cfg_path)
    name = d['name']
    for kv in sys.argv[3:]:
        key, val = kv.split('=', 1)
        node = d
        parts = key.split('.')
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        try:
            node[parts[-1]] = json.loads(val)
        except ValueError:
            node[parts[-1]] = val
    src_out = os.path.abspath(os.path.expanduser(d['out_dir']))
    src_work = os.path.join(src_out, 'work')
    brep = os.path.join(src_work, name + '_pools.brep')
    if not (os.path.exists(brep) and os.path.exists(os.path.join(src_work, name + '_pools.json'))):
        die('the checkpoint is missing: %s[.json] (the case\'s out_dir is %s) - run the case '
            'once without --from-checkpoint, or with --stop-after-checkpoint' % (brep, src_out))
    out = os.path.join(case_dir, variant, 'mesh')
    d['out_dir'] = out.replace('\\', '/')
    os.makedirs(os.path.join(out, 'work'), exist_ok=True)
    for ext in ('.brep', '.json'):
        shutil.copy(os.path.join(src_work, name + '_pools' + ext), os.path.join(out, 'work'))
    vj = os.path.join(case_dir, variant, variant + '.json')
    with io.open(vj, 'w', encoding='ascii', newline='\n') as f:
        json.dump(d, f, indent=2, ensure_ascii=True)
    print('checkpoint copied from', src_work)
    print('variant written:', vj)
    print('run: tools\\mesh\\run_step_mesh.cmd "%s" --from-checkpoint' % vj)


if __name__ == '__main__':
    main()
