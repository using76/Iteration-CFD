#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""Runner fixture: floods stderr with 6000 x's, then calls a Workplane method that does not exist."""
import sys

import cadquery as cq


def build(params, out_dir):
    sys.stderr.write("x" * 6000 + chr(10))
    sys.stderr.flush()
    return cq.Workplane("XY").box(1, 1, 1).hull()
