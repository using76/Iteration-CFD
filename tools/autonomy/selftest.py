#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""
selftest.py - the tools/autonomy package gate.

Runs schema.py --selftest (the schemas, fixtures, knob table and lock), then
checks that README.md still carries docs/15 §D verbatim between its markers and
that deps_licences.py --python sees numpy, scipy and scikit-learn installed as
BSD-3-Clause.

    python tools/autonomy/selftest.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PLAN = os.path.join(REPO, "docs", "15-autonomous-setup-plan.md")
README = os.path.join(HERE, "README.md")


def main():
    """Run schema.py --selftest as a child; echo its output; fail below 8 [ok] lines."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    p = subprocess.run([sys.executable, os.path.join(HERE, "schema.py"), "--selftest"],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env, timeout=300)
    n_ok = sum(1 for l in p.stdout.splitlines() if l.startswith("[ok]"))
    assert p.returncode == 0 and n_ok >= 8, \
        "schema.py --selftest failed (exit %d, %d [ok]): %s" % (p.returncode, n_ok, (p.stdout + p.stderr)[-2000:])
    assert "SELFTEST PASS" in p.stdout
    for l in p.stdout.splitlines():
        if l != "SELFTEST PASS":
            print(l)
    return n_ok


def _section_d(plan_lines):
    """docs/15 §D: from the D heading up to but excluding E, trailing blanks stripped."""
    i = plan_lines.index("## D. What we measure, exactly")
    j = plan_lines.index("## E. The corpus")
    body = plan_lines[i:j]
    while body and body[-1].strip() == "":
        body.pop()
    return "\n".join(body)


def main_rest():
    lines = []
    try:
        plan = open(PLAN, encoding="utf-8").read().replace("\r\n", "\n").splitlines()
        want = _section_d(plan)
        readme = open(README, encoding="utf-8").read().replace("\r\n", "\n")
        begin = "<!-- BEGIN VERBATIM docs/15-autonomous-setup-plan.md lines 136-203 at f636fa6 -->"
        assert begin in readme and "<!-- END VERBATIM -->" in readme, "README markers missing"
        have = readme[readme.index(begin) + len(begin):readme.index("<!-- END VERBATIM -->")]
        have = have.strip("\n")
        assert have == want, "README.md does not carry docs/15 §D verbatim"
        lines.append("[ok] README: docs/15 section D present verbatim (%d lines)"
                     % len(want.splitlines()))

        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        q = subprocess.run([sys.executable, os.path.join(REPO, "tools", "deps_licences.py"),
                            "--python", "--json"], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", env=env, timeout=300)
        assert q.returncode == 0, "deps_licences --python exited %d: %s" % (q.returncode, q.stderr[-500:])
        data = json.loads(q.stdout)
        parts = []
        for name in ("numpy", "scipy", "scikit-learn"):
            rec = next(r for r in data["packages"] if r["name"] == name)
            assert rec["installed"] and rec["detected"] == "BSD-3-Clause" \
                and rec["verdict"] == "ok", "%s: %r" % (name, rec)
            parts.append("%s %s BSD-3-Clause" % (name, rec["version"]))
        assert data["ok"] is True
        lines.append("[ok] licences: " + ", ".join(parts))
    except (AssertionError, KeyError, OSError, ValueError) as e:
        print("SELFTEST FAIL: %s" % e)
        return 1
    for l in lines:
        print(l)
    print("SELFTEST PASS")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    try:
        if main():
            sys.exit(main_rest())
    except AssertionError as e:
        print("SELFTEST FAIL: %s" % e)
        sys.exit(1)
    sys.exit(1)
