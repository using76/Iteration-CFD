#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""Runner fixture: reads address 0 through from_address - a real access violation past ctypes' handler."""
import ctypes


def build(params, out_dir):
    return {"v": ctypes.c_int.from_address(0).value}
