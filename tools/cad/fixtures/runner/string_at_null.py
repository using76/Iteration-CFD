#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""Runner fixture: the plan's ctypes.string_at(0), which ctypes turns into OSError on Windows, not a crash."""
import ctypes


def build(params, out_dir):
    return {"n": len(ctypes.string_at(0))}
