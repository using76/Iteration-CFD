#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""Runner fixture: starts a sleeping grandchild and sleeps 60 s itself - the timeout path."""
import os
import subprocess
import sys
import time


def build(params, out_dir):
    g = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    with open(os.path.join(out_dir, "grandchild.pid"), "w", encoding="utf-8") as f:
        f.write(str(g.pid))
    time.sleep(60)
    return {"slept": 60}
