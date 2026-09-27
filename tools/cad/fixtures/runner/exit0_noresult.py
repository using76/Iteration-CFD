#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""Runner fixture: leaves with os._exit(0) and writes no result - the RUN-NORESULT path."""
import os


def build(params, out_dir):
    os._exit(0)
