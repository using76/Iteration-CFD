#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""Runner fixture: builds a 1x2x3 box and returns its volume - the ok path."""
import cadquery as cq


def build(params, out_dir):
    solid = cq.Workplane("XY").box(params["a"], params["b"], params["c"]).val()
    return {"volume": solid.Volume()}
