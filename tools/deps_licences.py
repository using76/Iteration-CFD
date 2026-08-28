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
"""

import json
import re
import subprocess
import sys
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


def main():
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
