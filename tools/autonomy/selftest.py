#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""
selftest.py - the tools/autonomy package gate.

Runs schema.py --selftest (the schemas, fixtures, knob table and lock), then
score.py --selftest (the G-SCORER probe fixtures, the refusal grammar, the
content hash and the live automesher), then the ten corpus selftests of
docs/15 §E/§F (corpus/stl_io.py, corpus/gen_wing.py, corpus/gen_lathe.py,
corpus/meshkit.py, corpus/gen_bluff.py, corpus/gen_gap.py, corpus/gen_thin.py,
corpus/inject.py (family G), corpus/gate.py (G-CORPUS on families A, B, D,
E, F and G) and corpus/split.py (the sealed 420/180 split)), then
features.py --selftest
(the AM-4 geometry fingerprint: refusals, sphere curvature, cube sharp edges,
two-sphere gap, NACA0012 thickness, planar fraction, commensurability,
patches, invariance, schema and speed), then preflight.py --selftest (the L0 checks: the mesher schema mirror, 54 hand cases and 300 random configs against -dryRun, one fixture per refusal, the survey configs, C-THIN and stage 0), then rules.py --selftest
(the L1 setup rules: the worked example, the §D.3 window and fit_growth, R-DOM,
R-PLANE on cubep and corpus boxes, R-CURV, R-GAP and R-FEAT, the R-BUDGET
predictor and ladder, setup on five corpus rows against preflight and -dryRun,
the whitelist, the records, determinism and the CLI), then remedies.py --selftest (the L2 remedies: the table, diagnose, the 31 probe fixtures and 32 sequences, box_sphere CAPABILITY-LIMITED after 1 try, the static scan, 600 single steps and 300 loops, the synthetic outcomes, the records, determinism, the veto and the CLI), then sensitivity.py --selftest
(the G-PILOT builder: the 12 geometries, the 288+3 jobs and their whitelist
edits, R-WIN, the verdict function, a live small-cube run, resume and the
report), then checks that README.md still
carries docs/15 §D verbatim between its markers and that deps_licences.py
--python sees numpy, scipy and scikit-learn installed as BSD-3-Clause.

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
    """Run schema.py and score.py --selftest as children; echo; fail on either."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    p = subprocess.run([sys.executable, os.path.join(HERE, "schema.py"), "--selftest"],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env, timeout=300)
    n_ok = sum(1 for l in p.stdout.splitlines() if l.startswith("[ok]"))
    assert p.returncode == 0 and n_ok >= 8, \
        "schema.py --selftest failed (exit %d, %d [ok]): %s" % (p.returncode, n_ok, (p.stdout + p.stderr)[-2000:])
    assert "SELFTEST PASS" in p.stdout
    q = subprocess.run([sys.executable, os.path.join(HERE, "score.py"), "--selftest"],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env, timeout=300)
    s_ok = sum(1 for l in q.stdout.splitlines() if l.startswith("[ok]"))
    assert q.returncode == 0 and s_ok >= 9 and "SELFTEST PASS" in q.stdout, \
        "score.py --selftest failed (exit %d, %d [ok]): %s" % (q.returncode, s_ok,
                                                               (q.stdout + q.stderr)[-2000:])
    corpus = (("corpus/stl_io.py", 4), ("corpus/gen_wing.py", 7),
              ("corpus/gen_lathe.py", 7), ("corpus/meshkit.py", 7),
              ("corpus/gen_bluff.py", 7), ("corpus/gen_gap.py", 7),
              ("corpus/gen_thin.py", 7), ("corpus/inject.py", 7),
              ("corpus/gate.py", 7), ("corpus/split.py", 7),
              ("features.py", 11), ("preflight.py", 12), ("rules.py", 14),
              ("remedies.py", 13))
    total = n_ok + s_ok
    outs = [p.stdout, q.stdout]
    for rel, min_ok in corpus:
        c = subprocess.run([sys.executable, os.path.join(HERE, rel), "--selftest"],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", env=env, timeout=300)
        k = sum(1 for l in c.stdout.splitlines() if l.startswith("[ok]"))
        assert c.returncode == 0 and k >= min_ok and "SELFTEST PASS" in c.stdout, \
            "%s --selftest failed (exit %d, %d [ok]): %s" % (rel, c.returncode, k,
                                                             (c.stdout + c.stderr)[-2000:])
        total += k
        outs.append(c.stdout)
    r = subprocess.run([sys.executable, os.path.join(HERE, "sensitivity.py"), "--selftest"],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env, timeout=300)
    s_ok = sum(1 for l in r.stdout.splitlines() if l.startswith("[ok]"))
    msg = ("sensitivity.py --selftest failed (exit %d, %d [ok]): %s"
           % (r.returncode, s_ok, (r.stdout + r.stderr)[-2000:]))
    assert r.returncode == 0 and s_ok >= 9 and "SELFTEST PASS" in r.stdout, msg
    total += s_ok
    outs.append(r.stdout)
    for l in "".join(outs).splitlines():
        if l != "SELFTEST PASS":
            print(l)
    return total


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
