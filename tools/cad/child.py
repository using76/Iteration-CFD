#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""child.py - the child process of the CAD runner: loads ONE module by path, calls its entry with the job's params and writes result.json (docs/16 §A row 74, §D S3).

Usage (spawned by runner.py only):
  python child.py JOB_JSON
"""

import faulthandler
import importlib.util
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402
import hints  # noqa: E402

JOB_KEYS = ("module_path", "entry", "params", "out_dir")
RESULT_KEYS = ("status", "value", "error_type", "message", "traceback", "hint")
RESULT_NAME = "result.json"
MODULE_NAME = "cad_child_job"
SEM_FLAGS = 0x8003      # SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX


def quiet_crash_dialogs() -> None:
    """Stop Windows Error Reporting from holding a crashed child open with a dialog box."""
    if os.name != "nt":
        return
    try:
        import ctypes
        ctypes.windll.kernel32.SetErrorMode(SEM_FLAGS)
    except Exception:
        pass


def load_module(path: str):
    """Import the job's module by path - in THIS child only, never in the runner's process."""
    spec = importlib.util.spec_from_file_location(MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load module from %s" % path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = mod
    spec.loader.exec_module(mod)
    return mod


def result_record(status, value=None, error_type=None, message=None, tb=None, hint=None) -> dict:
    """One result.json record: exactly RESULT_KEYS in this order."""
    return {"status": status, "value": value, "error_type": error_type,
            "message": message, "traceback": tb, "hint": hint}


def main(argv) -> int:
    if len(argv) != 1:
        sys.stderr.write("usage: python child.py JOB_JSON" + chr(10))
        return 2
    faulthandler.enable()
    quiet_crash_dialogs()
    job = common.read_json(argv[0])
    if not isinstance(job, dict) or sorted(job) != sorted(JOB_KEYS):
        sys.stderr.write("child: malformed job" + chr(10))
        return 2
    out = os.path.join(job["out_dir"], RESULT_NAME)
    try:
        mod = load_module(job["module_path"])
        fn = getattr(mod, job["entry"], None)
        if not callable(fn):
            raise AttributeError("module has no callable entry %r" % job["entry"])
        value = fn(job["params"], job["out_dir"])
        common.canonical_json(value)    # refuses NaN and non-JSON BEFORE anything is written
    except Exception as e:              # NOT BaseException: SystemExit and os._exit pass through
        tb = traceback.format_exc()
        sys.stderr.write(tb)
        sys.stderr.flush()
        common.write_json(out, result_record("error", None, type(e).__name__,
                                             "%s: %s" % (type(e).__name__, e), tb,
                                             hints.match_hint(tb)))
        return 1
    common.write_json(out, result_record("ok", value))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
