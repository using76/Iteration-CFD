#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
#
# The lines between the two VERBATIM markers below are copied unchanged from
# AI-CAD (https://github.com/ai-cad-labs/ai-cad, commit c7503b4),
# .shared/tools/cadquery_executor.py lines 93-146:
#
#   AI-CAD
#   Copyright 2026 The ai-cad-labs project
#
#   This product includes software developed by Saifuddin Raja and the AI-CAD Labs contributors and maintainers.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# This file was modified by Iteration-CFD (Iterations Co., Ltd.) on 2026-09-26:
# the _HINTS table and match_hint() were moved out of cadquery_executor.py into
# this module, and the docstring, the pinned digest, selftest() and main() were
# added around them. The lines between the markers are unchanged and remain
# under Apache-2.0 (see NOTICE and LICENSE-APACHE-2.0.ai-cad at the repository
# root); the rest of this file is under LICENSE. docs/16 §A, decision D-5.
"""hints.py - ai-cad's corrective hints for known CadQuery hallucinations, verbatim, used to enrich a child's error (docs/16 §A).

Usage:
  python hints.py --selftest
"""

import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VERBATIM_SHA256 = "587e0e47b828a597c15ee2f7549167c86007999f730488a2b75fe9d02f8ad769"
VERBATIM_LINES = 54
BEGIN_MARK = "# ---- BEGIN VERBATIM: ai-cad c7503b4 .shared/tools/cadquery_executor.py:93-146 (Apache-2.0) ----"
END_MARK = "# ---- END VERBATIM ----"

# ---- BEGIN VERBATIM: ai-cad c7503b4 .shared/tools/cadquery_executor.py:93-146 (Apache-2.0) ----
_HINTS = {
    "has no attribute 'hull'": (
        "HINT: .hull() does not exist in CadQuery. "
        "To create an I-beam or H-beam cross section, use .polyline() to draw "
        "the profile shape, then .close().extrude(). For connecting two circular "
        "ends with a tapered beam, use boolean operations: create each cylinder "
        "separately, then .union() them with a rectangular beam body."
    ),
    "has no attribute 'fillet2D'": (
        "HINT: .fillet2D() does not exist. Use .fillet() on 3D edges AFTER extrude."
    ),
    "has no attribute 'cone'": (
        "HINT: .cone() does not exist as a Workplane method. "
        "Use cq.Solid.makeCone(radius1, radius2, height) instead."
    ),
    "has no attribute 'And'": (
        "HINT: cadquery.selectors.And does not exist. "
        "Use string selector combinations: .edges('|Z and >Y') or "
        "cadquery.selectors.AndSelector(sel1, sel2)."
    ),
    "has no attribute 'OrSelector'": (
        "HINT: cadquery.selectors.OrSelector does not exist. "
        "Use string selectors with 'or': .edges('|Z or |X')."
    ),
    "has no attribute 'Circle'": (
        "HINT: cq.Circle does not exist. Use .circle(radius) on a Workplane."
    ),
    "has no attribute 'makeHull'": (
        "HINT: Wire.makeHull() does not exist. "
        "Use .polyline() and .close() to create custom profiles."
    ),
    "Unknown color name": (
        "HINT: Valid CadQuery color names: red, green, blue, gray, lightgray, "
        "white, black, yellow, orange, cyan, magenta, brown, pink. "
        "Do NOT use 'silver', 'gold', or other CSS color names."
    ),
    "unexpected keyword argument 'centered'": (
        "HINT: CadQuery's .extrude() does NOT have a 'centered' kwarg. "
        "That's a Build123d API. Use .extrude(length) + .translate() to center. "
        "CadQuery extrude signature: extrude(until, combine=True, clean=True, both=False, taper=None)."
    ),
    "has no attribute 'EdgeCylinderSelector'": (
        "HINT: cq.selectors.EdgeCylinderSelector does not exist. "
        'Use the string selector "%CIRCLE" to select circular edges.'
    ),
}


def match_hint(error_msg: str) -> str | None:
    """Return the corrective hint for a known hallucination pattern, else None."""
    for pattern, hint in _HINTS.items():
        if pattern in error_msg:
            return hint
    return None
# ---- END VERBATIM ----


def verbatim_span(path=None) -> list:
    """The lines strictly between the first BEGIN and the next END marker line (whole-line equality)."""
    target = os.path.abspath(__file__) if path is None else path
    with open(target, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    begin = None
    end = None
    for i, line in enumerate(lines):
        if begin is None:
            if line == BEGIN_MARK:
                begin = i
        elif line == END_MARK:
            end = i
            break
    if begin is None or end is None:
        raise ValueError("hints.py: VERBATIM markers not found")
    return lines[begin + 1:end]


def verbatim_sha256(path=None) -> str:
    """sha256 over the span joined with LF and closed by one LF (the pinned digest of ai-cad 93-146)."""
    span = verbatim_span(path)
    return hashlib.sha256(("\n".join(span) + "\n").encode("utf-8")).hexdigest()


def selftest() -> int:
    span = verbatim_span()
    assert len(span) == VERBATIM_LINES, len(span)
    assert span[0] == "_HINTS = {", span[0]
    assert span[-1] == "    return None", span[-1]
    assert verbatim_sha256() == VERBATIM_SHA256, verbatim_sha256()
    assert len(_HINTS) == 10, len(_HINTS)
    print("[ok] verbatim: 54 lines between the markers, sha256 587e0e47b828 matches ai-cad c7503b4 lines 93-146")
    for pattern, hint_text in _HINTS.items():
        assert match_hint("prefix " + pattern + " suffix") == hint_text, pattern
    print("[ok] match_hint returns each of the 10 hints for its own pattern")
    assert match_hint("AttributeError: 'Workplane' object has no attribute 'hull'") == _HINTS["has no attribute 'hull'"]
    assert match_hint("ZeroDivisionError: division by zero") is None
    assert match_hint("") is None
    print("[ok] match_hint: a real AttributeError text gets the hull hint, an unknown error gets None")
    with open(os.path.abspath(__file__), "r", encoding="utf-8") as f:
        whole = f.read()
    for needle in ("Copyright 2026 The ai-cad-labs project",
                   'Licensed under the Apache License, Version 2.0 (the "License");',
                   "modified by Iteration-CFD (Iterations Co., Ltd.)",
                   "LICENSE-APACHE-2.0.ai-cad",
                   "No GPL-licensed source was consulted.",
                   "Iterations Co., Ltd."):
        assert needle in whole, needle
    print("[ok] attribution: AI-CAD copyright, Apache-2.0 notice, the modified-by mark and the provenance line are in the header")
    print("SELFTEST PASS")
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["--selftest"]:
        return selftest()
    sys.stderr.write("usage: python hints.py --selftest" + chr(10))
    return 2


if __name__ == "__main__":
    sys.exit(main())
