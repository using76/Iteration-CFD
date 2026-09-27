#!/usr/bin/env python3
"""
Regenerate the "Third-party components" block of NOTICE from Cargo.lock.

    python tools/deps_licences.py            # print the block
    python tools/deps_licences.py --check    # exit 1 if NOTICE disagrees

The point is that NOTICE should never be a hand-maintained list that drifts
from the lock file. This reads the resolved graph that `cargo` itself reports
(`cargo metadata --locked`, so it reflects Cargo.lock and not whatever happens
to be newest on crates.io) and prints every package with the licence its own
manifest declares. Nothing here is typed by hand.

It also enforces the one licence rule this project cannot bend: no GPL, LGPL
or AGPL anywhere in the graph, direct or transitive.

Needs `cargo` on PATH. On the development machine that is
`C:/Users/sdd32/.cargo/bin`.

    python tools/deps_licences.py --python         # the Python-side packages
    python tools/deps_licences.py --python --json  # the same, as JSON

The `--python` branch (docs/15 AM-1) is the Python twin: it reads only
`importlib.metadata`, reports the declared and the detected licence of the
autonomy engine's numpy/scipy/scikit-learn (plus its optional jsonschema and
referencing), lists the components their wheels bundle, and refuses any
GPL/LGPL/AGPL outside the GCC runtime library exception. It never calls cargo.
"""

import json
import re
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, files as dist_files, metadata as dist_metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "rust" / "Cargo.toml"
NOTICE = ROOT / "NOTICE"

# Substring match, case-insensitive, against the SPDX expression. "LGPL" is
# listed separately from "GPL" only for the error message - "GPL" catches it.
FORBIDDEN = ("GPL", "AGPL", "LGPL")

BEGIN = "<!-- BEGIN GENERATED: cargo metadata --locked -->"
END = "<!-- END GENERATED -->"


def packages():
    """Every package in the resolved graph, including the crate itself."""
    out = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--locked",
         "--manifest-path", str(MANIFEST)],
        capture_output=True, check=True,
    )
    meta = json.loads(out.stdout.decode("utf-8"))
    return sorted(meta["packages"], key=lambda p: p["name"].lower())


def render(pkgs):
    lines = [BEGIN,
             "Resolved dependency graph: %d packages (this crate plus %d)."
             % (len(pkgs), len(pkgs) - 1),
             "",
             "%-24s %-10s %s" % ("PACKAGE", "VERSION", "LICENCE"),
             "%-24s %-10s %s" % ("-" * 24, "-" * 10, "-" * 30)]
    for p in pkgs:
        lic = p.get("license") or ("see " + str(p.get("license_file")))
        lines.append("%-24s %-10s %s" % (p["name"], p["version"], lic))
    lines += ["", END]
    return "\n".join(lines)


REQUIRED_PY = ("numpy", "scipy", "scikit-learn")
OPTIONAL_PY = ("jsonschema", "referencing")

OK_PY_LICENCES = ("BSD-3-Clause", "BSD-2-Clause", "MIT", "Apache-2.0")


def _dist_licence_text(meta):
    """The licence text: the License field, else the dist-info LICEN*/COPYING* file."""
    lic = meta.get("License") or ""
    if lic.strip():
        return lic
    try:
        for f in dist_files(meta["Name"]) or []:
            stem = str(f).rsplit("/", 1)[-1].upper()
            if stem.startswith(("LICEN", "COPYING")):
                return f.locate().read_text(encoding="utf-8", errors="replace")
    except Exception:
        pass
    return ""


def _licence_head(text):
    """The licence text cut at the first `Name:` line (the bundled list starts there)."""
    head = []
    for line in text.splitlines():
        if line.strip().startswith("Name:"):
            break
        head.append(line)
    return "\n".join(head)


def _detect(head, expression):
    """License-Expression if present, else BSD-3/BSD-2/MIT by the head's own words."""
    if expression:
        return expression
    low = head.lower()
    has_use = "redistribution and use in source and binary forms" in low
    has_name = "neither the name" in low
    if has_use and has_name:
        return "BSD-3-Clause"
    if has_use:
        return "BSD-2-Clause"
    if "permission is hereby granted, free of charge" in low:
        return "MIT"
    return "UNKNOWN"


def _bundled(text):
    """The `Name: X` / `License: Y` components the wheel's licence text declares."""
    comps = []
    cur = None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("Name:"):
            cur = {"name": s[5:].strip(), "licence": "(not stated)"}
            comps.append(cur)
        elif s.startswith("License:") and cur is not None:
            cur["licence"] = s[8:].strip()
    return comps


def _py_record(name):
    """One package's licence record for --python, from importlib.metadata only."""
    try:
        meta = dist_metadata(name)
        version = meta["Version"]
    except PackageNotFoundError:
        return {"name": name, "installed": False, "version": None, "declared": None,
                "classifiers": [], "detected": None, "bundled": [], "verdict": "missing"}
    expression = meta.get("License-Expression") or None
    classifiers = [c for c in meta.get_all("Classifier") or [] if c.startswith("License ::")]
    declared = expression
    if not declared:
        for line in (meta.get("License") or "").splitlines():
            if line.strip():
                declared = line.strip()
                break
    if not declared and classifiers:
        declared = "; ".join(classifiers)
    text = _dist_licence_text(meta)
    head = _licence_head(text)
    detected = _detect(head, expression)
    bundled = _bundled(text)
    notes = [c for c in bundled
             if c["licence"].rstrip().endswith("-with-GCC-exception")]
    gpl = [c for c in bundled
           if "GPL" in c["licence"].upper()
           and not c["licence"].rstrip().endswith("-with-GCC-exception")]
    if detected in OK_PY_LICENCES and not gpl:
        verdict = "ok"
    else:
        verdict = "refused"
    return {"name": name, "installed": True, "version": version, "declared": declared,
            "classifiers": classifiers, "detected": detected, "bundled": bundled,
            "verdict": verdict, "gcc_exception_notes": notes}


def python_main(as_json):
    """Report the Python-side licences; exit 0 ok, 1 a required package missing, 2 refused."""
    records = [_py_record(n) for n in REQUIRED_PY] + [_py_record(n) for n in OPTIONAL_PY]
    if as_json:
        print(json.dumps({"packages": records,
                          "ok": all(r["verdict"] == "ok" for r in records
                                    if r["name"] in REQUIRED_PY)},
                         indent=2))
    else:
        print("%-14s %-9s %-9s %-40s %-16s %s"
              % ("PACKAGE", "VERSION", "INSTALLED", "DECLARED", "DETECTED", "VERDICT"))
        print("%-14s %-9s %-9s %-40s %-16s %s"
              % ("-" * 14, "-" * 9, "-" * 9, "-" * 40, "-" * 16, "-" * 7))
        for r in records:
            print("%-14s %-9s %-9s %-40s %-16s %s"
                  % (r["name"], r["version"] or "-", "yes" if r["installed"] else "NO",
                     (r["declared"] or "-")[:40], r["detected"] or "-", r["verdict"]))
            for c in r["bundled"]:
                print("  bundled: %-42s %s" % (c["name"], c["licence"]))
            for c in r.get("gcc_exception_notes", []):
                print("  NOTE: %s carries %s - the GCC runtime library exception, "
                      "which is what numpy's and scipy's OpenBLAS DLL carry"
                      % (c["name"], c["licence"]))
    required = [r for r in records if r["name"] in REQUIRED_PY]
    if any(r["verdict"] == "refused" for r in required):
        return 2
    if any(r["verdict"] == "missing" for r in required):
        return 1
    return 0


def main():
    if "--python" in sys.argv:
        return python_main("--json" in sys.argv)
    pkgs = packages()

    bad = []
    for p in pkgs:
        lic = (p.get("license") or "").upper()
        # Word-boundary match so "GPL" does not fire on, say, a name containing
        # it; SPDX ids are separated by spaces, slashes, parentheses.
        for tok in re.split(r"[^A-Z0-9.+-]+", lic):
            if tok.split("-")[0] in FORBIDDEN or tok in FORBIDDEN:
                bad.append((p["name"], p.get("license")))
                break
    if bad:
        print("FORBIDDEN LICENCE IN THE GRAPH:", file=sys.stderr)
        for n, l in bad:
            print("  %s: %s" % (n, l), file=sys.stderr)
        return 2

    block = render(pkgs)

    if "--check" in sys.argv:
        text = NOTICE.read_text(encoding="utf-8")
        if BEGIN not in text or END not in text:
            print("NOTICE has no generated block", file=sys.stderr)
            return 1
        have = text[text.index(BEGIN):text.index(END) + len(END)]
        if have.strip() != block.strip():
            print("NOTICE is stale - rerun without --check and paste the block",
                  file=sys.stderr)
            return 1
        print("NOTICE matches Cargo.lock (%d packages, no GPL/LGPL/AGPL)"
              % len(pkgs))
        return 0

    print(block)
    return 0


if __name__ == "__main__":
    sys.exit(main())
