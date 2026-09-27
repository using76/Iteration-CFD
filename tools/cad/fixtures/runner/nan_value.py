#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""Runner fixture: returns a NaN, which canonical JSON refuses before any result is written."""
def build(params, out_dir):
    return {"x": float("nan")}
