#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""usage: python tools/aero/fast_patch.py <caseDir>

Fast-mode patch for the F1 case: 300 km/h physics + a pressure linear
solve that does not grind. The dominant cost of the slow run was the
pressure PBiCGStab grinding to relTol 0.01 with maxIter 1000 on a cut-cell
mesh with 79-degree non-orthogonality - hundreds to a thousand inner
iterations per outer iteration. relTol 0.05 + maxIter 200 trades a little
per-iteration accuracy for an order of magnitude less inner work.

A rerun starts from the previous solve's 0/ - its pressure field carries
the converged M spikes and NaNs the first pressure solve, so the patch
resets p to a clean uniform gauge 0 and drops the stale rho (recomputed
from p0 and T).
"""
import argparse
import os
import re
import sys

U_X = 83.3333
K_IN = 4.1667
EPS_IN = 18.632


def die(msg, code=1):
    sys.stderr.write('fast_patch: %s\n' % msg)
    sys.stderr.flush()
    raise SystemExit(code)


def sub(text, pattern, repl, flags=0):
    """The original's re.sub, counted: a pattern that matches nothing is a
    silent no-op, so every call reports how many places it rewrote."""
    return re.subn(pattern, repl, text, flags=flags)


def patch_uniform(text, value):
    """The internalField pair for one 0/ field: the nonuniform counted list
    first, then the uniform form - after the first has rewritten a nonuniform
    field the second always matches, so the pair's count is the second's."""
    text, _ = sub(text, r"internalField\s+nonuniform\s+List<[^>]+>\s*\d+\s*\(.*?\n\)\s*;",
                  f"internalField   uniform {value};", re.S)
    return sub(text, r"internalField\s+uniform\s+[^;]+;",
               f"internalField   uniform {value};")


def patch_inlet(text, value):
    return sub(text, r"(inlet\s*\{\s*type\s+fixedValue;)\s*value\s+uniform\s+[^;]+;",
               r"\g<1> value           uniform " + value + ";")


def plan(case_dir):
    """Read every file and apply every substitution in memory, refusing before
    anything is written: a required row with no hit fails by name, while a
    rerun accepts its own output - the two fvSolution rows whose patterns
    carry the un-patched literals pass when their OWN block already reads
    the patched form (`done`, a regex scoped to that block, never a bare
    substring: `maxIter         200;` also stands in the U block and would
    swallow a real refusal of p's row). Returns the new texts, the
    (file, label, count, required) rows in the original's order, and the
    unlink list."""
    texts, rows, unlink = {}, [], []

    def read(name):
        path = os.path.join(case_dir, *name.split('/'))
        if not os.path.isfile(path):
            die('missing file %s' % path)
        with open(path, encoding='utf-8', newline='') as f:
            return f.read()

    def record(name, label, n, text, required=True, done=None):
        rows.append((name, label, n, required))
        if required and n == 0 and not (done and re.search(done, text)):
            die('no match in %s for %s' % (name, label))

    text = read('0/U')
    text, n = patch_uniform(text, '(%s 0 0)' % U_X)
    record('0/U', 'internalField', n, text)
    text, n = patch_inlet(text, '(%s 0 0)' % U_X)
    record('0/U', 'inlet', n, text)
    texts['0/U'] = text

    text = read('0/p')
    text, n = patch_uniform(text, '0')
    record('0/p', 'internalField', n, text)
    texts['0/p'] = text

    rho = os.path.join(case_dir, '0', 'rho')
    if os.path.isfile(rho):
        unlink.append(rho)

    text = read('0/k')
    text, n = patch_uniform(text, str(K_IN))
    record('0/k', 'internalField', n, text)
    text, n = patch_inlet(text, str(K_IN))
    record('0/k', 'inlet', n, text)
    text, n = sub(text, r"(kqRWallFunction;\s*value\s+)uniform\s+[^;]+;",
                  r"\g<1>uniform " + str(K_IN) + ";")
    record('0/k', 'kqRWallFunction', n, text, required=False)
    texts['0/k'] = text

    text = read('0/epsilon')
    text, n = patch_uniform(text, str(EPS_IN))
    record('0/epsilon', 'internalField', n, text)
    text, n = patch_inlet(text, str(EPS_IN))
    record('0/epsilon', 'inlet', n, text)
    text, n = sub(text, r"(kqRWallFunction;\s*value\s+)uniform\s+[^;]+;",
                  r"\g<1>uniform " + str(EPS_IN) + ";")
    record('0/epsilon', 'kqRWallFunction', n, text, required=False)
    texts['0/epsilon'] = text

    text = read('constant/physicalProperties')
    text, n = sub(text, r"nu\s+\[0 2 -1 0 0 0 0\]\s*[^;]+;",
                  "nu              [0 2 -1 0 0 0 0] 1.5e-05;")
    record('constant/physicalProperties', 'nu', n, text)
    texts['constant/physicalProperties'] = text

    # the speed knobs
    text = read('system/fvSolution')
    text, n = sub(text, r"(solver\s+PBiCGStab;)([^}]*?)tolerance\s+1e-08;",
                  r"\g<1>\g<2>tolerance       1e-06;")
    record('system/fvSolution', 'PBiCGStab tol', n, text,
           done=r"solver\s+PBiCGStab;[^}]*?tolerance\s+1e-06;")
    text, n = sub(text, r"(preconditioner\s+DIC;\s*tolerance\s+1e-06;\s*relTol\s+)[\d.]+;",
                  r"\g<1>0.05;")
    record('system/fvSolution', 'p relTol', n, text)
    text, n = sub(text, r"(preconditioner\s+DIC;\s*tolerance\s+1e-06;\s*relTol\s+[\d.]+;\s*maxIter\s+)1000;",
                  r"\g<1>200;")
    record('system/fvSolution', 'p maxIter', n, text,
           done=r"preconditioner\s+DIC;\s*tolerance\s+1e-06;\s*relTol\s+[\d.]+;\s*maxIter\s+200;")
    text, n = sub(text, r"(fields\s*\{\s*p\s+)[\d.]+;", r"\g<1>0.25;")
    record('system/fvSolution', 'relax p', n, text)
    text, n = sub(text, r"(equations\s*\{[^}]*?U\s+)[\d.]+;", r"\g<1>0.5;", re.S)
    record('system/fvSolution', 'relax U', n, text)
    texts['system/fvSolution'] = text

    return texts, rows, unlink


def apply(case_dir, texts, unlink):
    """The second phase: write every patched file, then drop the stale ones -
    a refusal in plan() has already left the case untouched."""
    for name in texts:
        path = os.path.join(case_dir, *name.split('/'))
        with open(path, 'w', encoding='utf-8', newline='') as f:
            f.write(texts[name])
    for path in unlink:
        os.remove(path)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description='Fast-mode patch of a generated case: counted regex rewrites of the run settings.')
    ap.add_argument('case_dir', help='the case directory to patch in place')
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    texts, rows, unlink = plan(args.case_dir)
    for name, label, n, _required in rows:
        print('fast_patch: %s %s: %d hit(s)' % (name, label, n))
    apply(args.case_dir, texts, unlink)
    print(f"fast-mode patch done: relTol 0.05, p maxIter 200, U={U_X} m/s")
    return 0


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.exit(main())
