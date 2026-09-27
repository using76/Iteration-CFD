#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""runner.py - runs one module's entry point in a killable child process and returns one run record (docs/16 §A rows 56-57 and 74, §D S3, §I).

The parent NEVER imports or runs the job's module: only child.py does, in its own
process, so a crash or a hang in the module cannot touch this interpreter. On
timeout the whole PID tree dies - taskkill /T /F /PID on Windows, the child's
process group via killpg on POSIX - always by PID, never by image name or
command line. Exit codes are mapped to ok / exit / crash: Windows reports
crashes as unsigned 32-bit NTSTATUS values (Microsoft [MS-ERREF] §2.3.1,
https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-erref/596a1078-e883-4972-9bbc-49e60bebca55),
POSIX shells as death by signal. The plan's spawn line `python -m tools.cad.child
job.json` is realised as [sys.executable, child.py, job.json], because every
tools/cad module runs as a script by path here (no package, siblings on
sys.path) and a path keeps the spawn independent of the cwd. The plan's crash
fixture ctypes.string_at(0) is an OSError on Windows - ctypes wraps its foreign
calls in a handler - so the real access-violation probe is from_address(0).

Usage:
  python runner.py --selftest
"""

import ctypes
import math
import os
import signal
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402
import hints  # noqa: E402

CHILD = os.path.join(HERE, "child.py")
FIXTURE_DIR = os.path.join(HERE, "fixtures", "runner")
IS_WINDOWS = os.name == "nt"
DEFAULT_TIMEOUT_S = 120.0
KILL_WAIT_S = 5.0
STDERR_TAIL_BYTES = 4096
STILL_ACTIVE = 259
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
RUN_KEYS = ("status", "rule", "exit_code", "exit_hex", "exit_name", "timed_out", "pid", "wall_s", "value",
            "error_type", "message", "hint", "stderr_tail", "job_sha", "out_dir")
RESULT_KEYS = ("status", "value", "error_type", "message", "traceback", "hint")   # child.py's, not imported
RULES = ("RUN-JOB", "RUN-RAISE", "RUN-EXIT", "RUN-NORESULT", "RUN-CRASH", "RUN-TIMEOUT")
NTSTATUS = {
    0xC0000005: "STATUS_ACCESS_VIOLATION",
    0xC0000006: "STATUS_IN_PAGE_ERROR",
    0xC0000017: "STATUS_NO_MEMORY",
    0xC000001D: "STATUS_ILLEGAL_INSTRUCTION",
    0xC000008E: "STATUS_FLOAT_DIVIDE_BY_ZERO",
    0xC0000094: "STATUS_INTEGER_DIVIDE_BY_ZERO",
    0xC0000096: "STATUS_PRIVILEGED_INSTRUCTION",
    0xC00000FD: "STATUS_STACK_OVERFLOW",
    0xC000013A: "STATUS_CONTROL_C_EXIT",
    0xC0000374: "STATUS_HEAP_CORRUPTION",
    0xC0000409: "STATUS_STACK_BUFFER_OVERRUN",
}
SIGNALS = {4: "SIGILL", 6: "SIGABRT", 7: "SIGBUS", 8: "SIGFPE", 9: "SIGKILL", 11: "SIGSEGV", 15: "SIGTERM"}
SHELL_CRASH = (4, 6, 7, 8, 11)      # a shell reports death by signal n as exit 128 + n


def classify_exit(rc: int) -> tuple:
    """Map a child exit code to (kind, name, hex): kind is ok, exit or crash (docs/16 §A row 57)."""
    rc = int(rc)
    if rc == 0:
        return ("ok", None, "0x00000000")
    if rc < 0:
        return ("crash", SIGNALS.get(-rc, "SIG%d" % -rc), None)
    if rc - 128 in SHELL_CRASH:
        return ("crash", SIGNALS[rc - 128], "0x%08X" % rc)
    if 0xC0000000 <= rc <= 0xFFFFFFFF:
        return ("crash", NTSTATUS.get(rc, "NTSTATUS_0x%08X" % rc), "0x%08X" % rc)
    return ("exit", None, "0x%08X" % rc)


def pid_alive(pid: int) -> bool:
    """True while the process still runs; on Windows via OpenProcess + GetExitCodeProcess."""
    if IS_WINDOWS:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = ctypes.c_void_p
        k32.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
        k32.GetExitCodeProcess.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32))
        k32.CloseHandle.argtypes = (ctypes.c_void_p,)
        h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, int(pid))
        if not h:
            return False
        try:
            code = ctypes.c_uint32()
            ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
            return bool(ok) and code.value == STILL_ACTIVE
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def kill_tree(pid: int) -> None:
    """Kill the process and its descendants by PID - never by image name or command line."""
    try:
        if IS_WINDOWS:
            tk = os.path.join(os.environ.get("SystemRoot", ""), "System32", "taskkill.exe")
            if not os.path.isfile(tk):
                tk = "taskkill"
            subprocess.run([tk, "/T", "/F", "/PID", str(int(pid))], stdin=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
        else:
            os.killpg(int(pid), signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        pass    # an already-dead PID is a no-op


def stderr_tail(path: str, n: int = STDERR_TAIL_BYTES) -> str:
    """The last n BYTES of a file, utf-8-decoded; an empty string when the file is missing."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - n))
            return f.read().decode("utf-8", errors="ignore")
    except OSError:
        return ""


def read_result(path: str):
    """The child's result dict, or None when missing, malformed, wrongly keyed or badly statused."""
    try:
        result = common.read_json(path)
    except Exception:
        return None
    if not isinstance(result, dict):
        return None
    if sorted(result) != sorted(RESULT_KEYS):
        return None
    if result.get("status") not in ("ok", "error"):
        return None
    return result


def new_record(**fields) -> dict:
    """A run record with every RUN_KEYS key in order; an unknown key raises KeyError."""
    for key in fields:
        if key not in RUN_KEYS:
            raise KeyError(key)
    rec = {}
    for key in RUN_KEYS:
        rec[key] = fields.get(key, False) if key == "timed_out" else fields.get(key)
    return rec


def refuse_job(reason: str, out_dir) -> dict:
    """The RUN-JOB record for a refused job: nothing written, nothing spawned, no directory made."""
    return new_record(status="error", rule="RUN-JOB", message="RUN-JOB: " + reason,
                      out_dir=out_dir if isinstance(out_dir, str) else None)


def run_job(module_path, params, out_dir, entry="build", timeout_s=DEFAULT_TIMEOUT_S) -> dict:
    """Run one module's entry in a killable child; refuse a malformed job before any spawn."""
    if (not isinstance(module_path, str) or not os.path.isabs(module_path)
            or not module_path.endswith(".py") or not os.path.isfile(module_path)):
        return refuse_job("module_path is not an existing absolute .py file: %r" % (module_path,), out_dir)
    if not isinstance(entry, str) or not entry.isidentifier():
        return refuse_job("entry is not an identifier: %r" % (entry,), out_dir)
    if not isinstance(params, dict):
        return refuse_job("params is not a dict", out_dir)
    try:
        common.canonical_json(params)
    except (TypeError, ValueError) as e:
        return refuse_job("params are not canonical JSON: %s" % e, out_dir)
    if (isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s) or timeout_s <= 0):
        return refuse_job("timeout_s is not a positive finite number: %r" % (timeout_s,), out_dir)
    if not isinstance(out_dir, str) or not os.path.isabs(out_dir):
        return refuse_job("out_dir is not an absolute path: %r" % (out_dir,), out_dir)
    os.makedirs(out_dir, exist_ok=True)
    for stale in ("result.json", "run.json"):
        stale_path = os.path.join(out_dir, stale)
        if os.path.exists(stale_path):
            os.remove(stale_path)
    job = {"module_path": os.path.abspath(module_path), "entry": entry, "params": params,
           "out_dir": os.path.abspath(out_dir)}
    job_path = os.path.join(out_dir, "job.json")
    common.write_json(job_path, job)
    job_sha = common.sha256_of(job)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONHASHSEED="0")
    if IS_WINDOWS:
        extra = {"creationflags": subprocess.CREATE_NO_WINDOW}
    else:
        extra = {"start_new_session": True}     # the child leads its own process group
    t0 = time.monotonic()
    with open(os.path.join(out_dir, "child.stdout.txt"), "wb") as fo:
        with open(os.path.join(out_dir, "child.stderr.txt"), "wb") as fe:
            # FILES, never PIPE: a grandchild holding a pipe would hang this parent
            proc = subprocess.Popen([sys.executable, CHILD, job_path], stdin=subprocess.DEVNULL,
                                    stdout=fo, stderr=fe, cwd=out_dir, env=env, **extra)
            timed_out = False
            try:
                rc = proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                kill_tree(proc.pid)
                try:
                    rc = proc.wait(timeout=KILL_WAIT_S)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    rc = proc.wait(timeout=KILL_WAIT_S)
    wall_s = round(time.monotonic() - t0, 3)
    tail = stderr_tail(os.path.join(out_dir, "child.stderr.txt"))
    result = read_result(os.path.join(out_dir, "result.json"))
    kind, name, hexs = classify_exit(rc)
    status = rule = message = value = error_type = hint = None
    if timed_out:
        status, rule = "timeout", "RUN-TIMEOUT"
        message = "killed after %.1f s timeout (PID tree)" % timeout_s
    elif kind == "crash":
        status, rule = "crash", "RUN-CRASH"
        message = "child crashed: %s (%s)" % (name, hexs if hexs else rc)
        hint = hints.match_hint(tail)
    elif result is not None and result["status"] == "error":
        status, rule = "error", "RUN-RAISE"
        message = result["message"]
        error_type = result["error_type"]
        hint = result["hint"]
    elif kind == "exit":
        status, rule = "error", "RUN-EXIT"
        message = "child exited %d without a result" % rc
        hint = hints.match_hint(tail)
    elif result is not None and result["status"] == "ok":
        status = "ok"
        value = result["value"]
    else:
        status, rule = "error", "RUN-NORESULT"
        message = "child exited 0 but result.json is missing or malformed"
    rec = new_record(status=status, rule=rule, exit_code=rc, exit_hex=hexs, exit_name=name,
                     timed_out=timed_out, pid=proc.pid, wall_s=wall_s, value=value,
                     error_type=error_type, message=message, hint=hint, stderr_tail=tail,
                     job_sha=job_sha, out_dir=job["out_dir"])
    common.write_json(os.path.join(out_dir, "run.json"), rec)
    return rec


def selftest() -> int:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        fx = lambda name: os.path.join(FIXTURE_DIR, name)
        expected = [
            (0, "ok", None, "0x00000000"),
            (1, "exit", None, "0x00000001"),
            (2, "exit", None, "0x00000002"),
            (3, "exit", None, "0x00000003"),
            (255, "exit", None, "0x000000FF"),
            (137, "exit", None, "0x00000089"),
            (-11, "crash", "SIGSEGV", None),
            (-6, "crash", "SIGABRT", None),
            (-9, "crash", "SIGKILL", None),
            (139, "crash", "SIGSEGV", "0x0000008B"),
            (134, "crash", "SIGABRT", "0x00000086"),
            (0xC0000005, "crash", "STATUS_ACCESS_VIOLATION", "0xC0000005"),
            (0xC00000FD, "crash", "STATUS_STACK_OVERFLOW", "0xC00000FD"),
            (0xC0000409, "crash", "STATUS_STACK_BUFFER_OVERRUN", "0xC0000409"),
            (0xC0000374, "crash", "STATUS_HEAP_CORRUPTION", "0xC0000374"),
            (0xC0001234, "crash", "NTSTATUS_0xC0001234", "0xC0001234"),
        ]
        for rc, kind, name, hx in expected:
            assert classify_exit(rc) == (kind, name, hx), (rc, classify_exit(rc))
        print("[ok] classify_exit maps 16 codes: 0 ok; 1 2 3 255 137 exit; -11 -6 -9 139 134 and 5 NTSTATUS values crash")
        d = os.path.join(td, "ok_box")
        rec = run_job(fx("ok_box.py"), {"a": 1.0, "b": 2.0, "c": 3.0}, d)
        assert rec["status"] == "ok" and rec["rule"] is None and rec["exit_code"] == 0, rec
        assert abs(rec["value"]["volume"] - 6.0) <= 1e-9, rec["value"]
        assert tuple(rec) == RUN_KEYS, tuple(rec)
        assert common.read_json(os.path.join(d, "run.json")) == rec
        assert rec["job_sha"] == common.sha256_of(common.read_json(os.path.join(d, "job.json")))
        print("[ok] ok: volume %s from the child, run.json equals the record, job_sha matches job.json, wall %s s"
              % (rec["value"]["volume"], rec["wall_s"]))
        d = os.path.join(td, "raise_hull")
        rec = run_job(fx("raise_hull.py"), {}, d)
        assert rec["status"] == "error" and rec["rule"] == "RUN-RAISE", rec
        assert rec["error_type"] == "AttributeError", rec["error_type"]
        assert "has no attribute 'hull'" in rec["message"], rec["message"]
        assert rec["hint"] == hints._HINTS["has no attribute 'hull'"], rec["hint"]
        assert len(rec["stderr_tail"].encode("utf-8")) == STDERR_TAIL_BYTES, len(rec["stderr_tail"])
        assert "x" * 100 in rec["stderr_tail"] and "has no attribute 'hull'" in rec["stderr_tail"]
        print("[ok] raise: RUN-RAISE AttributeError with the ai-cad hull hint; stderr tail 4096 bytes ends in the traceback")
        d = os.path.join(td, "sleep_tree")
        rec = run_job(fx("sleep_tree.py"), {}, d, timeout_s=3.0)
        assert rec["status"] == "timeout" and rec["rule"] == "RUN-TIMEOUT", rec
        assert rec["timed_out"] is True, rec["timed_out"]
        assert rec["wall_s"] < 5.0, rec["wall_s"]
        with open(os.path.join(d, "grandchild.pid"), "r", encoding="utf-8") as f:
            gpid = int(f.read().strip())
        for _ in range(20):
            if not pid_alive(rec["pid"]) and not pid_alive(gpid):
                break
            time.sleep(0.05)
        assert not pid_alive(rec["pid"]), rec["pid"]
        assert not pid_alive(gpid), gpid
        print("[ok] timeout: killed at 3 s, wall %s s < 5 s, child PID %d and grandchild PID %d gone"
              % (rec["wall_s"], rec["pid"], gpid))
        child_pid = rec["pid"]
        d = os.path.join(td, "crash_null")
        rec = run_job(fx("crash_null.py"), {}, d)
        assert rec["status"] == "crash" and rec["rule"] == "RUN-CRASH", rec
        if IS_WINDOWS:
            assert rec["exit_code"] == 0xC0000005, rec["exit_code"]
            assert rec["exit_hex"] == "0xC0000005", rec["exit_hex"]
            assert rec["exit_name"] == "STATUS_ACCESS_VIOLATION", rec["exit_name"]
            assert "access violation" in rec["stderr_tail"], rec["stderr_tail"][-200:]
        else:
            assert rec["exit_name"] == "SIGSEGV", rec["exit_name"]
        print("[ok] crash: %s %s classified crash, faulthandler text in the stderr tail"
              % (rec["exit_name"], rec["exit_hex"] or rec["exit_code"]))
        d = os.path.join(td, "exit3")
        os.makedirs(d)
        common.write_json(os.path.join(d, "result.json"),
                          {"status": "ok", "value": 1, "error_type": None, "message": None,
                           "traceback": None, "hint": None})
        rec = run_job(fx("exit3.py"), {}, d)
        assert rec["status"] == "error" and rec["rule"] == "RUN-EXIT", rec
        assert rec["exit_code"] == 3, rec["exit_code"]
        assert rec["value"] is None, rec["value"]
        assert not os.path.exists(os.path.join(d, "result.json"))
        print("[ok] exit: sys.exit(3) is RUN-EXIT with exit 3; a planted stale ok result.json was removed, not read")
        d = os.path.join(td, "string_at_null")
        rec = run_job(fx("string_at_null.py"), {}, d)
        if IS_WINDOWS:
            assert rec["status"] == "error" and rec["rule"] == "RUN-RAISE", rec
            assert rec["error_type"] == "OSError", rec["error_type"]
            assert "access violation" in rec["message"], rec["message"]
        else:
            assert rec["status"] == "crash", rec
        print("[ok] plan correction: ctypes.string_at(0) is %s on Windows, not a crash; from_address(0) is the real 0xC0000005"
              % rec["error_type"])
        bad = [
            (fx("no_such.py"), "build", {}, DEFAULT_TIMEOUT_S),
            ("ok_box.py", "build", {}, DEFAULT_TIMEOUT_S),
            (fx("ok_box.py"), "not an id", {}, DEFAULT_TIMEOUT_S),
            (fx("ok_box.py"), "build", [1], DEFAULT_TIMEOUT_S),
            (fx("ok_box.py"), "build", {"x": float("nan")}, DEFAULT_TIMEOUT_S),
            (fx("ok_box.py"), "build", {}, 0),
            (fx("ok_box.py"), "build", {}, float("nan")),
            (fx("ok_box.py"), "build", {}, True),
        ]
        for i, (mp, entry, params, to) in enumerate(bad):
            out = os.path.join(td, "never%d" % i)
            rec = run_job(mp, params, out, entry=entry, timeout_s=to)
            assert rec["status"] == "error" and rec["rule"] == "RUN-JOB", rec
            assert rec["pid"] is None and rec["exit_code"] is None, rec
            assert not os.path.exists(out), out
        rec = run_job(fx("ok_box.py"), {}, "relative_dir")
        assert rec["status"] == "error" and rec["rule"] == "RUN-JOB", rec
        assert rec["pid"] is None and rec["exit_code"] is None, rec
        print("[ok] job refusals: 9 bad jobs refused as RUN-JOB before any spawn, no directory created")
        d = os.path.join(td, "exit0_noresult")
        rec = run_job(fx("exit0_noresult.py"), {}, d)
        assert rec["status"] == "error" and rec["rule"] == "RUN-NORESULT", rec
        assert rec["exit_code"] == 0, rec["exit_code"]
        d = os.path.join(td, "nan_value")
        rec = run_job(fx("nan_value.py"), {}, d)
        assert rec["status"] == "error" and rec["rule"] == "RUN-RAISE", rec
        assert rec["error_type"] == "ValueError", rec["error_type"]
        assert "non-finite" in rec["message"], rec["message"]
        print("[ok] no result: exit 0 without result.json is RUN-NORESULT; a NaN return is RUN-RAISE ValueError")
        assert "cadquery" not in sys.modules
        assert "cad_child_job" not in sys.modules
        norm_fix = os.path.normcase(FIXTURE_DIR)
        for mod in list(sys.modules.values()):
            modfile = getattr(mod, "__file__", None)
            if modfile:
                assert not os.path.normcase(os.path.abspath(modfile)).startswith(norm_fix), modfile
        print("[ok] isolation: the parent imported no fixture module and no cadquery after 8 runs")
        pats = ("exe" + "c(", "eva" + "l(")
        scanned = 0
        hits = 0
        for root, dirnames, filenames in os.walk(HERE):
            dirnames[:] = [dn for dn in dirnames if dn != "__pycache__"]
            for fname in filenames:
                if not fname.endswith(".py"):
                    continue
                scanned += 1
                with open(os.path.join(root, fname), "r", encoding="utf-8") as f:
                    for line in f:
                        if pats[0] in line or pats[1] in line:
                            hits += 1
        assert scanned >= 16, scanned
        assert hits == 0, hits
        print("[ok] exec/eval gate: 0 hits in %d .py files under tools/cad" % scanned)
        kill_tree(child_pid)
        assert pid_alive(os.getpid())
        assert not pid_alive(child_pid)
        print("[ok] kill_tree on a dead PID is a no-op; pid_alive sees this process and not the killed child")
    print("SELFTEST PASS")
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["--selftest"]:
        return selftest()
    sys.stderr.write("usage: python runner.py --selftest" + chr(10))
    return 2


if __name__ == "__main__":
    sys.exit(main())
