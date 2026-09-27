#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""selftest.py - the package gate of tools/cad: runs every module's --selftest as a child and exits non-zero on any miss.

Usage:
  python selftest.py
"""

import contextlib
import io
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKS = (("common.py", 14), ("schema.py", 6), ("schema_fixtures.py", 13), ("measure.py", 43), ("hints.py", 4), ("runner.py", 12), ("template_gc3.py", 12), ("export.py", 15), ("readiness.py", 23), ("reqs.py", 45), ("verify.py", 15), ("mutate.py", 18), ("wedge_mesh.py", 13), ("thwaites.py", 15), ("turb_integral.py", 15), ("mesh_fidelity.py", 11), ("case_writer.py", 9), ("post.py", 13), ("solve.py", 15))   # (script, min [ok]); later units append

_TALLY = []                                     # each child's n_ok, for the final count


def run_child(script_path: str, min_ok: int) -> tuple:
    """Run one module's --selftest; (ok, n_ok, message); a timeout is a failure, not an exception."""
    try:
        p = subprocess.run([sys.executable, script_path, "--selftest"],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                           timeout=600)
    except subprocess.TimeoutExpired:
        return False, 0, "timed out after 600s"
    n_ok = sum(1 for l in p.stdout.splitlines() if l.startswith("[ok]"))
    if p.returncode == 0 and n_ok >= min_ok and "SELFTEST PASS" in p.stdout:
        return True, n_ok, ""
    tail = (p.stdout + p.stderr)[-1500:]
    return False, n_ok, "exit %d, %d [ok] (min %d): %s" % (p.returncode, n_ok, min_ok, tail)


def run_checks(checks, here: str) -> int:
    """Print one line per child; run EVERY check even after a miss; 0 only when all passed."""
    del _TALLY[:]
    missed = False
    for script, min_ok in checks:
        path = script if os.path.isabs(script) else os.path.join(here, script)
        ok, n_ok, message = run_child(path, min_ok)
        _TALLY.append(n_ok)
        if ok:
            print("[ok] %s: %d [ok]" % (script, n_ok))
        else:
            print("[FAIL] %s: %s" % (script, message))
            missed = True
    return 1 if missed else 0


def harness_selftest() -> None:
    """Prove the gate itself against three failing fake children and one good one."""
    with tempfile.TemporaryDirectory() as td:
        def child(name, lines):
            path = os.path.join(td, name)
            with open(path, "w", encoding="utf-8") as f:
                f.write(chr(10).join(lines) + chr(10))
            return path
        a = child("a_exit1.py", ['print("[ok] x")', 'print("SELFTEST PASS")',
                                 'raise SystemExit(1)'])
        b = child("b_few_ok.py", ['print("[ok] only-one")', 'print("SELFTEST PASS")'])
        c = child("c_no_pass.py", ['print("[ok] a")', 'print("[ok] b")'])
        d = child("d_good.py", ['print("[ok] a")', 'print("[ok] b")',
                                'print("SELFTEST PASS")'])
        for fake, want, why in ((a, 1, "prints PASS then exits 1"),
                                (b, 1, "too few [ok]"),
                                (c, 1, "no SELFTEST PASS"),
                                (d, 0, "the good child")):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                got = run_checks([(fake, 2)], td)
            assert got == want, "%s (%s): run_checks returned %d, want %d: %s" % (
                fake, why, got, want, buf.getvalue())
    print("[ok] harness refuses 3 of 3 failing children and passes the good one")


def main() -> int:
    harness_selftest()
    rc = run_checks(CHECKS, HERE)
    if rc == 0:
        print("SELFTEST PASS (%d [ok])" % (1 + sum(_TALLY)))
        return 0
    print("SELFTEST FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
