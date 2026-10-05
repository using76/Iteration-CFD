#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""solve.py - stage S8 of the CAD loop (docs/16 §D S8, §I; docs/16a §E SOLVE-BIND): one written cad-case/1
directory is solved once by the pinned ofgpu-lowmach binary in a visible console, cold, for a fixed -iters budget,
never retried with changed numerics. The log's iter lines (lowmach.rs print_report at bceb799) and its last line
`run ended: <word> | <detail> | exit code <n>` (run_end_line) are parsed, and the run is classified steady,
unsteady, diverged or refused by the stop_rule of tools/cad/gates.json: steady iff the |U| and |p| residuals fall
at least residual_decades decades from iteration 0 to the last iteration, |contErr| <= cont_err_max, and dp and Cd
(post.py on each written time of the last window_iters iterations) each change by less than their limit relative
to the final value. A case of kind "pipe" is classified by contErr and the two window changes alone (SPEC-LIT
114.4: in the periodic pipe two of three velocity components and p are round-off fields, and the binary prints
only the largest component's residual), the residual decades still computed and reported. The case directory's
sha map is taken before the launch and after the run ended; a case file
that changed marks the solve refused (SOLVE-BIND), never steady.

Usage:
  python solve.py --selftest
  python solve.py classify LOG HISTORY_JSON ITERS OUT_JSON
  python solve.py run CASE_DIR GEOM_DIR OUT_DIR ITERS
  python solve.py tee LOG -- COMMAND...
"""

import json
import math
import os
import re
import subprocess
import sys
import tempfile
import traceback
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common
import schema

VERSION = "cad-solve/1"
GATES = os.path.join(HERE, "gates.json")
BIN_JSON = os.path.join(HERE, "bin.json")
BIN_NAME = "ofgpu-lowmach"
CHECK_EVERY = 50            # -check: an iter line every 50 steps (the binary's default)
WRITE_EVERY = 50            # -writeEvery: a time directory every 50 iterations, so the window has 5 samples
MACH_LIMIT = 0.3            # energy.rs:846 LOW_MACH_LIMIT at bceb799
CLASSES = ("steady", "unsteady", "diverged", "refused")
LAUNCH_IDS = ("SOLVE-OUT", "SOLVE-GATES", "SOLVE-ARGS", "SOLVE-BIN", "SOLVE-COLD", "SOLVE-BIND")   # pre-launch, in order
RESULT_IDS = ("SOLVE-LOG", "SOLVE-DIVERGED", "SOLVE-MACH", "SOLVE-REFUSED", "SOLVE-ERROR", "SOLVE-POST",
              "SOLVE-UNSTEADY")
END_CODES = {"budget": 0, "error": 1, "diverged": 2, "refused": 3}
CRITERIA = ("U_decades", "p_decades", "cont_err", "dp_rel_change", "Cd_rel_change")
CRITERIA_PERIODIC = ("cont_err", "dp_rel_change", "Cd_rel_change")   # SPEC-LIT 114.4/114.5 O3: the periodic pipe
ITER_KEYS = ("iter", "U_res", "p_res", "cont_err", "T_min", "T_max", "rho_min", "rho_max", "p0", "dp0dt",
             "M_max", "M_cell", "M_mean", "finite")
RESULT_KEYS = ("class", "reason_id", "detail", "run_end", "n_lines", "iterations", "mach_max", "mach_refusal",
               "criteria", "failed", "window")
HISTORY_KEYS = ("iter", "time", "status", "reason_id", "dp", "Cd", "post_sha256")
SOLVE_KEYS = ("version", "class", "reason_id", "message", "case_dir", "geom_dir", "binary", "command", "iters",
              "check_every", "write_every", "p0_Pa", "gates", "before", "after", "written", "returncode", "log",
              "result", "gating")
TOP_LEVEL = ("0", "case.json", "constant", "system")     # a cold case directory holds exactly these
CASE_VERSIONS = ("cad-case/1", "cad-case-turb/1")        # the cold case.json versions launch admits


class Refused(Exception):
    """A refusal with its id and a human detail, the shape of post.Refused."""

    def __init__(self, rule, detail=""):
        super().__init__("%s: %s" % (rule, detail))
        self.rule = rule
        self.detail = detail


def load_rule(gates_path=None):
    """The stop_rule dict and the gates file's sha256 (the plan constants, cad-gates/1); refuses SOLVE-GATES."""
    path = GATES if gates_path is None else gates_path
    snap = common.stable_file_snapshot(path)
    if not snap["stable"]:
        raise Refused("SOLVE-GATES", "gates file is missing or not a stable regular file: %s" % path)
    doc = common.read_json(path)
    errs = schema.errors(doc, "cad-gates/1")
    if errs:
        raise Refused("SOLVE-GATES", str(errs[0]))
    return doc["stop_rule"], snap["sha256"]


# One number as print_report writes it through g()/sci(): optional -, then nan, inf, or digits with an optional
# .digits part and an optional signed exponent of two or more digits (common/mod.rs:582-643 at bceb799).
_N = r"-?(?:nan|inf|\d+(?:\.\d+)?(?:e[+-]\d+)?)"
_ITER_RE = re.compile(
    r"^iter +(\d+)  \|U\| res (%(N)s)  \|p\| res (%(N)s)  contErr (%(N)s)"
    r"  T \[(%(N)s), (%(N)s)\] K  rho \[(%(N)s), (%(N)s)\] kg/m3"
    r"  p0 (%(N)s) Pa  dp0/dt (%(N)s) Pa/s  M max (%(N)s) \(cell (\d+)\) mean (%(N)s)"
    r"(  \*\*\* NaN/Inf \*\*\*)?$" % {"N": _N})
_END_RE = re.compile(r"^run ended: (budget|diverged|refused|error) \| (.*) \| exit code (\d+)$")
_MACH_MSG_RE = re.compile(r"^ofgpu-lowmach: max Mach number (\S+) at cell (\d+) \(([^)]*)\) is above (\S+);")
_RES_KEYS = ("U_res", "p_res", "cont_err", "T_min", "T_max", "rho_min", "rho_max", "p0", "dp0dt", "M_max",
             "M_mean")
_ROW_GROUPS = (("U_res", 2), ("p_res", 3), ("cont_err", 4), ("T_min", 5), ("T_max", 6), ("rho_min", 7),
               ("rho_max", 8), ("p0", 9), ("dp0dt", 10), ("M_max", 11), ("M_mean", 13))


def _num(token):
    """float(token) when finite, else the token exactly as written (nan, inf, -inf)."""
    if token.lstrip("-").isalpha():
        return token
    return float(token)


def _iter_row(m):
    """One ITER_KEYS row from a matched iter line."""
    row = {"iter": int(m.group(1))}
    for key, gi in _ROW_GROUPS:
        row[key] = _num(m.group(gi))
    row["M_cell"] = int(m.group(12))
    row["finite"] = m.group(14) is None
    return row


def _parse_end(lines):
    """The run ended dict {word, detail, exit_code} when exactly one line starts with `run ended: `, it is the
    last non-empty line and it matches in full; None otherwise (classify's own step 3 rule)."""
    ends = [i for i, l in enumerate(lines) if l.startswith("run ended: ")]
    if len(ends) != 1:
        return None
    last_nonempty = max(j for j, l in enumerate(lines) if l.strip())
    if ends[0] != last_nonempty:
        return None
    m = _END_RE.match(lines[ends[0]])
    if m is None:
        return None
    return {"word": m.group(1), "detail": m.group(2), "exit_code": int(m.group(3))}


def _base_result(lines):
    """The refused-by-default classify doc (RESULT_KEYS) before the steps run."""
    return {"class": "refused", "reason_id": "SOLVE-LOG", "detail": "", "run_end": None,
            "n_lines": len(lines), "iterations": [], "mach_max": None, "mach_refusal": None,
            "criteria": None, "failed": [], "window": None}


def classify(log_text, history, stop_rule, iters, write_every, returncode=None, gating=CRITERIA):
    """Classify one ofgpu-lowmach log by the stop_rule (docs/16 §D S8); never raises on any log text.
    gating is CRITERIA or CRITERIA_PERIODIC - the criteria rows that decide the class; all five are
    always computed and reported."""
    if tuple(gating) != CRITERIA and tuple(gating) != CRITERIA_PERIODIC:
        raise ValueError("gating %r is neither CRITERIA nor CRITERIA_PERIODIC" % (gating,))
    lines = log_text.splitlines()
    res = _base_result(lines)
    bad_iter = False
    for line in lines:                                  # 1. every iter line, in log order
        if not line.startswith("iter "):
            continue
        m = _ITER_RE.match(line)
        if m is None:
            bad_iter = True
            continue
        res["iterations"].append(_iter_row(m))
    m_max = [r["M_max"] for r in res["iterations"] if isinstance(r["M_max"], float)]
    res["mach_max"] = max(m_max) if m_max else None
    if bad_iter:
        res["detail"] = "a line starting with 'iter ' does not match the print_report format"
        return res
    seq = [r["iter"] for r in res["iterations"]]
    if any(b <= a for a, b in zip(seq, seq[1:])):       # 2. strictly increasing
        res["detail"] = "the iter numbers are not strictly increasing"
        return res
    end = _parse_end(lines)                             # 3. the end line
    if end is None:
        res["detail"] = "the run ended line is missing, duplicated, not last or malformed"
        return res
    res["run_end"] = end
    if END_CODES[end["word"]] != end["exit_code"] or (returncode is not None and returncode != end["exit_code"]):
        res["detail"] = "the exit code %d does not match the word %s or the process's %s" % (
            end["exit_code"], end["word"], returncode)  # 4. the code pairs
        return res
    if end["word"] == "diverged":                       # 5.
        res["class"], res["reason_id"] = "diverged", "SOLVE-DIVERGED"
        return res
    if end["word"] == "refused":                        # 6.
        res["class"] = "refused"
        mm = _MACH_MSG_RE.match(end["detail"])
        if mm is None:
            res["reason_id"] = "SOLVE-REFUSED"
        else:
            res["reason_id"] = "SOLVE-MACH"
            res["mach_refusal"] = {"M": float(mm.group(1)), "cell": int(mm.group(2)),
                                   "when": mm.group(3), "limit": float(mm.group(4))}
        return res
    if end["word"] == "error":                          # 7.
        res["class"], res["reason_id"] = "refused", "SOLVE-ERROR"
        return res
    if end["detail"] != "%d iterations reached" % iters:    # 8. the budget detail
        res["detail"] = "the budget detail %r is not %d iterations" % (end["detail"], iters)
        return res
    return _steady_steps(res, history, stop_rule, iters, write_every, gating)


def _decades(a, b, limit):
    """Decades from a to b (both > 0) against limit; pass iff v is not None and v >= limit."""
    v = math.log10(a / b) if a > 0 and b > 0 else None
    return {"value": v, "limit": limit, "pass": v is not None and v >= limit}


def _rel(xs, limit):
    """The largest relative change to the final value; pass iff v < limit."""
    xn = xs[-1]
    if any(x is None for x in xs) or xn == 0:
        return {"value": None, "limit": limit, "pass": False}
    v = max(abs(x - xn) / abs(xn) for x in xs)
    return {"value": v, "limit": limit, "pass": v < limit}


def _steady_steps(res, history, stop_rule, iters, write_every, gating=CRITERIA):
    """classify's steps 9-13: the budget rows, the Mach sweep and the five criteria; the class is
    decided by the gating rows alone (the periodic pipe gates on contErr and the two window changes,
    SPEC-LIT 114.4), the other rows reported with their pass untouched."""
    by_iter = {r["iter"]: r for r in res["iterations"]}
    if 0 not in by_iter or iters - 1 not in by_iter:    # 9. first and last iteration present
        res["detail"] = "the log lacks the iter 0 or the iter %d line" % (iters - 1)
        return res
    for r in res["iterations"]:                         # 10. finite rows only
        if not r["finite"] or any(not isinstance(r[k], float) for k in _RES_KEYS):
            res["detail"] = "iteration %d is not fully finite" % r["iter"]
            return res
    for r in res["iterations"]:                         # 11. the Mach limit from an iter line
        if r["M_max"] > MACH_LIMIT:
            res["reason_id"] = "SOLVE-MACH"
            res["mach_refusal"] = {"M": r["M_max"], "cell": r["M_cell"],
                                   "when": "iter %d" % r["iter"], "limit": MACH_LIMIT}
            res["detail"] = "max Mach %.6g at iteration %d is above %g" % (r["M_max"], r["iter"], MACH_LIMIT)
            return res
    want = list(range(iters - stop_rule["window_iters"], iters + 1, write_every))
    rows = list(history)                                # 12. the post window
    if [r["iter"] for r in rows] != want or any(r["status"] != "ok" for r in rows):
        res["reason_id"] = "SOLVE-POST"
        res["detail"] = "the post window %s is incomplete or a window time was refused" % (want,)
        return res
    res["window"] = {"iters": want, "dp": [r["dp"] for r in rows], "Cd": [r["Cd"] for r in rows]}
    first, last = by_iter[0], by_iter[iters - 1]        # 13. the five criteria
    v_ce = abs(last["cont_err"])
    crit = {"U_decades": _decades(first["U_res"], last["U_res"], stop_rule["residual_decades"]),
            "p_decades": _decades(first["p_res"], last["p_res"], stop_rule["residual_decades"]),
            "cont_err": {"value": v_ce, "limit": stop_rule["cont_err_max"],
                         "pass": v_ce <= stop_rule["cont_err_max"]},
            "dp_rel_change": _rel(res["window"]["dp"], stop_rule["dp_rel_change_max"]),
            "Cd_rel_change": _rel(res["window"]["Cd"], stop_rule["cd_rel_change_max"])}
    res["criteria"] = crit
    res["failed"] = [k for k in CRITERIA if k in gating and not crit[k]["pass"]]
    if res["failed"]:
        res["class"], res["reason_id"] = "unsteady", "SOLVE-UNSTEADY"
    else:
        res["class"], res["reason_id"] = "steady", None
    return res


def history(case_dir, geom_dir, iters_list):
    """One HISTORY_KEYS row per written time of the window, from post.post (docs/16 §D S9); never raises."""
    import post                                         # post.py is heavy; only the window needs it
    rows = []
    for it in iters_list:
        doc = post.post(case_dir, str(it), geom_dir)
        row = {"iter": it, "time": str(it), "status": doc["status"], "reason_id": doc["reason_id"]}
        metrics = doc.get("metrics") or {}
        for name in ("dp", "Cd"):
            value = None
            if doc["status"] == "ok" and metrics.get(name) is not None:
                metric = metrics[name]
                if isinstance(metric, dict):
                    value = None if metric.get("reason_id") == post.UNDEFINED_ID else metric.get("value")
                else:
                    value = metric
            row[name] = value
        row["post_sha256"] = common.sha256_of(doc)
        rows.append(row)
    return rows


def _stable_sha(path):
    """The stable snapshot's sha256, or None when the file is missing or not a stable regular file."""
    snap = common.stable_file_snapshot(path)
    return snap["sha256"] if snap["stable"] else None


def _case_files(root):
    """{relative /-separated path: stable sha or None} for every regular file under root."""
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            rel = os.path.relpath(os.path.join(dirpath, name), root).replace(os.sep, "/")
            out[rel] = _stable_sha(os.path.join(dirpath, name))
    return out


def _is_time_top(rel):
    """True when a relative path's top-level directory is a time name: all digits and not the cold start 0."""
    top = rel.split("/", 1)[0]
    return top.isdigit() and top != "0"


def _times(case_dir):
    """The sorted (by int) top-level time directory names of a case after a run."""
    return sorted((n for n in os.listdir(case_dir) if n.isdigit() and n != "0"), key=int)


def launch(case_dir, geom_dir, out_dir, iters, exe=None, bin_json=None, gates_path=None, history_fn=None,
           visible=True):
    """Solve one cold case once with the pinned binary (docs/16 §D S8); never raises Refused - the refusal is
    the returned doc with its class "refused", and every outcome but SOLVE-OUT lands in out_dir/solve.json."""
    if history_fn is None:
        history_fn = history
    case_dir = os.path.abspath(case_dir)
    geom_dir = os.path.abspath(geom_dir)
    out_dir = os.path.abspath(out_dir)
    doc = {"version": VERSION, "class": None, "reason_id": None, "message": "", "case_dir": case_dir,
           "geom_dir": geom_dir, "binary": None, "command": None, "iters": iters, "check_every": CHECK_EVERY,
           "write_every": WRITE_EVERY, "p0_Pa": None, "gates": None, "before": None, "after": None,
           "written": [], "returncode": None, "log": None, "result": None, "gating": None}

    def refuse(rid, message):
        doc["class"], doc["reason_id"], doc["message"] = "refused", rid, message
        if rid != "SOLVE-OUT":                          # no out_dir to hold the doc yet
            common.atomic_write(os.path.join(out_dir, "solve.json"), common.canonical_json(doc) + chr(10))
        return doc

    if out_dir == case_dir or out_dir.startswith(case_dir + os.sep):
        return refuse("SOLVE-OUT", "out_dir equals or lies inside case_dir")          # 1.
    if os.path.exists(out_dir) and (not os.path.isdir(out_dir) or os.listdir(out_dir)):
        return refuse("SOLVE-OUT", "out_dir exists and is not an empty directory")
    os.makedirs(out_dir, exist_ok=True)
    try:
        stop_rule, gates_sha = load_rule(gates_path)                                  # 2.
    except Refused as r:
        return refuse(r.rule, r.detail)
    doc["gates"] = {"path": "tools/cad/gates.json" if gates_path is None else gates_path,
                    "sha256": gates_sha, "stop_rule": stop_rule}
    if not isinstance(iters, int) or isinstance(iters, bool):                         # 3.
        return refuse("SOLVE-ARGS", "iters %r is not an int" % (iters,))
    if iters < stop_rule["window_iters"] or iters % WRITE_EVERY != 0 \
            or stop_rule["window_iters"] % WRITE_EVERY != 0:
        return refuse("SOLVE-ARGS", "iters %d does not fit window_iters %d at -writeEvery %d"
                      % (iters, stop_rule["window_iters"], WRITE_EVERY))
    if exe is None:                                                                   # 4.
        bj = common.read_json(BIN_JSON if bin_json is None else bin_json)
        entry = (bj.get("binaries") or {}).get(BIN_NAME)
        if entry is None:
            return refuse("SOLVE-BIN", "the binary record has no %s entry" % BIN_NAME)
        bin_path = os.path.join(common.REPO, entry["path"])
        sha = _stable_sha(bin_path)
        if sha != entry["sha256"]:
            return refuse("SOLVE-BIN", "the pinned %s is missing or its sha256 differs" % BIN_NAME)
        prefix = [bin_path]
        doc["binary"] = {"name": BIN_NAME, "path": entry["path"], "sha256": entry["sha256"]}
    else:
        sha = _stable_sha(exe[-1])
        if sha is None:
            return refuse("SOLVE-BIN", "the exe target %s is not a stable regular file" % exe[-1])
        prefix = list(exe)
        doc["binary"] = {"name": "exe", "path": exe[-1], "sha256": sha}
    return _launch_run(doc, case_dir, geom_dir, out_dir, iters, stop_rule, prefix, exe, history_fn, visible)


def _launch_run(doc, case_dir, geom_dir, out_dir, iters, stop_rule, prefix, exe, history_fn, visible):
    """launch's SOLVE-COLD and SOLVE-BIND checks, the one run and the classification of its log."""

    def refuse(rid, message):
        doc["class"], doc["reason_id"], doc["message"] = "refused", rid, message
        common.atomic_write(os.path.join(out_dir, "solve.json"), common.canonical_json(doc) + chr(10))
        return doc

    if sorted(os.listdir(case_dir)) != list(TOP_LEVEL):                               # 5.
        return refuse("SOLVE-COLD", "a cold case holds exactly %s, found %s"
                      % (list(TOP_LEVEL), sorted(os.listdir(case_dir))))
    case = common.read_json(os.path.join(case_dir, "case.json"))
    p0 = (case.get("operating_point") or {}).get("p0_Pa") if isinstance(case, dict) else None
    if (not isinstance(case, dict) or case.get("version") not in CASE_VERSIONS or case.get("status") != "ok"
            or case.get("cold_start") is not True or isinstance(p0, bool)
            or not isinstance(p0, (int, float)) or not math.isfinite(p0) or p0 <= 0):
        return refuse("SOLVE-COLD", "case.json is not an ok cold cad-case/1 or cad-case-turb/1 with a"
                      " positive finite p0_Pa")
    p0 = float(p0)
    doc["p0_Pa"] = p0
    gating = CRITERIA_PERIODIC if case.get("kind") == "pipe" else CRITERIA   # SPEC-LIT 114.5 O3
    doc["gating"] = list(gating)
    disk = _case_files(case_dir)                                                      # 6.
    want = {"case.json"} | set(case.get("files") or {})
    if set(disk) != want:
        return refuse("SOLVE-BIND", "the case directory's files are not exactly case.json's files map")
    moved = sorted(rel for rel, sha in disk.items() if sha is None)
    if moved:
        return refuse("SOLVE-BIND", "not a stable regular file: %s" % moved[0])
    changed = sorted(rel for rel, sha in disk.items() if rel != "case.json" and sha != case["files"][rel])
    if changed:
        return refuse("SOLVE-BIND", "sha256 differs from case.json for %s" % changed[0])
    before = dict(disk)
    doc["before"] = before
    command = prefix + [case_dir, "-iters", str(iters), "-check", str(CHECK_EVERY), "-writeEvery",
                        str(WRITE_EVERY), "-p0", repr(p0)]
    doc["command"] = command
    log_path = os.path.join(out_dir, "solve.log")
    tee_cmd = [sys.executable, os.path.join(HERE, "solve.py"), "tee", log_path, "--"] + command
    if visible and os.name == "nt":
        proc = subprocess.run(tee_cmd, cwd=out_dir, creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        proc = subprocess.run(tee_cmd, cwd=out_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    doc["returncode"] = proc.returncode
    with open(log_path, "rb") as f:
        log_bytes = f.read()
    log_text = log_bytes.decode("utf-8", "replace")
    doc["written"] = _times(case_dir)
    doc["log"] = {"file": "solve.log", "sha256": common.sha256_bytes(log_bytes),
                  "n_lines": len(log_text.splitlines())}
    after = dict((rel, _stable_sha(os.path.join(case_dir, rel))) for rel in before)   # 7.
    doc["after"] = after
    grew = sorted(rel for rel in _case_files(case_dir) if rel not in before and not _is_time_top(rel))
    if any(after[rel] != before[rel] for rel in before) or grew:
        doc["result"] = classify(log_text, [], stop_rule, iters, WRITE_EVERY, proc.returncode, gating=gating)
        if grew:
            return refuse("SOLVE-BIND", "a new file outside a time directory: %s" % grew[0])
        moved = sorted(rel for rel in before if after[rel] != before[rel])
        return refuse("SOLVE-BIND", "a case file changed during the run: %s" % moved[0])
    result = classify(log_text, [], stop_rule, iters, WRITE_EVERY, proc.returncode, gating=gating)   # 8.
    if result["run_end"] is not None and result["run_end"]["word"] == "budget":
        window = list(range(iters - stop_rule["window_iters"], iters + 1, WRITE_EVERY))
        result = classify(log_text, history_fn(case_dir, geom_dir, window), stop_rule, iters, WRITE_EVERY,
                          proc.returncode, gating=gating)
    doc["result"] = result
    doc["class"], doc["reason_id"] = result["class"], result["reason_id"]
    common.atomic_write(os.path.join(out_dir, "solve.json"), common.canonical_json(doc) + chr(10))
    return doc


def tee_run(args):
    """solve.py tee LOG -- COMMAND...: run COMMAND once, tee its stdout+stderr byte for byte into LOG."""
    if len(args) < 3 or args[1] != "--":
        print(USAGE, file=sys.stderr)
        return 2
    log_path, command = args[0], args[2:]
    try:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except OSError as e:
        with open(log_path, "wb") as f:
            f.write(("solve.py tee: cannot start: %s" % e).encode("utf-8"))
        return 1
    with open(log_path, "wb") as log:
        for line in iter(proc.stdout.readline, b""):
            log.write(line)
            log.flush()
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()
    return proc.wait()


USAGE = ("usage: solve.py --selftest | classify LOG HISTORY_JSON ITERS OUT_JSON | "
         "run CASE_DIR GEOM_DIR OUT_DIR ITERS | tee LOG -- COMMAND...")


def main(argv):
    if argv == ["--selftest"]:
        try:
            selftest()
        except Exception:
            traceback.print_exc()
            return 1
        return 0
    if len(argv) == 5 and argv[0] == "classify":
        with open(argv[1], "rb") as f:
            text = f.read().decode("utf-8", "replace")
        hist_doc = common.read_json(argv[2])
        rule, _ = load_rule()
        result = classify(text, hist_doc.get("rows") or [], rule, int(argv[3]), WRITE_EVERY,
                          hist_doc.get("returncode"))
        common.atomic_write(argv[4], common.canonical_json(result) + chr(10))
        print(common.canonical_json({"class": result["class"], "reason_id": result["reason_id"]}))
        return 0 if result["class"] == "steady" else 1
    if len(argv) == 5 and argv[0] == "run":
        doc = launch(os.path.abspath(argv[1]), os.path.abspath(argv[2]), os.path.abspath(argv[3]), int(argv[4]))
        print(common.canonical_json({"class": doc["class"], "reason_id": doc["reason_id"]}))
        return 0 if doc["class"] == "steady" else 1
    if argv and argv[0] == "tee":
        return tee_run(argv[1:])
    print(USAGE, file=sys.stderr)
    return 2



def selftest():
    ok = 0
    with tempfile.TemporaryDirectory() as tmp:
        F = os.path.join(common.FIXTURES, "solve")
        rule = load_rule()[0]

        def strip_detail(obj):
            return dict((k, v) for k, v in obj.items() if k != "detail")    # the free-text top-level detail only

        def same(a, b):
            return common.canonical_json(strip_detail(a)) == common.canonical_json(strip_detail(b))

        def rep(t, a, b):
            assert t.count(a) == 1, "expected exactly one occurrence of %r" % a
            return t.replace(a, b)

        gates_path = os.path.join(HERE, "gates.json")
        with open(gates_path, "rb") as f:
            gates_sha = common.sha256_bytes(f.read())
        assert gates_sha == "8c59f024d1241e16ce959c09487a79f229caa0ec805ab573794dd328e3cb103c"
        want_rule = {"residual_decades": 4, "cont_err_max": 1e-06, "window_iters": 200,
                     "dp_rel_change_max": 1e-05, "cd_rel_change_max": 1e-05}
        assert rule == want_rule
        rule2, sha2 = load_rule()
        assert rule2 == want_rule and sha2 == gates_sha
        bad = common.read_json(gates_path)
        bad["stop_rule"]["window_iters"] = 200.5
        bad_path = os.path.join(tmp, "gates_bad.json")
        common.write_json(bad_path, bad)
        for bp in (bad_path, os.path.join(tmp, "missing.json")):
            try:
                load_rule(bp)
                raise AssertionError("load_rule accepted a bad gates file")
            except Refused as r:
                assert r.rule == "SOLVE-GATES"
        print("[ok] gates.json is the pinned plan constants and load_rule refuses a bad or missing one")
        ok += 1

        fixtures = ("steady", "unsteady", "diverged", "refused_mach")
        texts, hists, exps = {}, {}, {}
        for name in fixtures:
            with open(os.path.join(F, name, "solve.log"), "rb") as f:
                texts[name] = f.read().decode("utf-8", "replace")
            hists[name] = common.read_json(os.path.join(F, name, "history.json"))["rows"]
            exps[name] = common.read_json(os.path.join(F, name, "expected.json"))
        ST, H = texts["steady"], hists["steady"]
        for name in fixtures:
            exp = exps[name]
            got = classify(texts[name], hists[name], rule, exp["iters"], exp["write_every"], exp["returncode"])
            assert same(got, exp["result"]), name
        s = classify(ST, H, rule, 600, 50, 0)
        assert s["class"] == "steady" and s["reason_id"] is None and len(s["iterations"]) == 13
        print("[ok] the steady fixture is steady with 13 finite iterations and no reason")
        ok += 1
        u = classify(texts["unsteady"], hists["unsteady"], rule, 600, 50, 0)
        assert u["class"] == "unsteady" and u["reason_id"] == "SOLVE-UNSTEADY"
        assert u["failed"] == list(CRITERIA)
        print("[ok] the unsteady fixture fails all five criteria as SOLVE-UNSTEADY")
        ok += 1

        # P1 (SPEC-LIT 114.4/114.5 O3, TG0-REDO): the residual decades alone fail the default gating;
        # under CRITERIA_PERIODIC the same log is steady and the decades rows are reported, not gating
        STp = rep(rep(ST, "|U| res 2.19539e-05", "|U| res 0.1"), "|p| res 5.012e-05", "|p| res 0.05")
        r = classify(STp, H, rule, 600, 50, 0)
        assert r["class"] == "unsteady" and r["reason_id"] == "SOLVE-UNSTEADY"
        assert r["failed"] == ["U_decades", "p_decades"]
        assert r["criteria"]["U_decades"]["value"] == 0.9415114326344031
        assert r["criteria"]["p_decades"]["value"] == 1.3010299956639813
        rp = classify(STp, H, rule, 600, 50, 0, gating=CRITERIA_PERIODIC)
        assert rp["class"] == "steady" and rp["reason_id"] is None and rp["failed"] == []
        assert rp["criteria"]["U_decades"]["value"] == 0.9415114326344031
        assert rp["criteria"]["p_decades"]["value"] == 1.3010299956639813
        assert rp["criteria"]["U_decades"]["pass"] is False and rp["criteria"]["p_decades"]["pass"] is False
        print("[ok] P1 periodic gating: residual decades alone no longer gate, the same two values"
              " reported and passing nowhere")
        ok += 1

        # P2: under CRITERIA_PERIODIC the contErr and the two window changes still bite, alone
        h2p = [dict(row) for row in H]
        h2p[0]["dp"] = h2p[-1]["dp"] * 1.00002
        r = classify(STp, h2p, rule, 600, 50, 0, gating=CRITERIA_PERIODIC)
        assert r["class"] == "unsteady" and r["reason_id"] == "SOLVE-UNSTEADY"
        assert r["failed"] == ["dp_rel_change"]
        assert r["criteria"]["dp_rel_change"]["value"] == 1.9999999999818947e-05
        r = classify(rep(STp, "contErr 1.32501e-09", "contErr 1.5e-06"), H, rule, 600, 50, 0,
                     gating=CRITERIA_PERIODIC)
        assert r["class"] == "unsteady" and r["failed"] == ["cont_err"]
        try:
            classify(STp, H, rule, 600, 50, 0, gating=("U_decades",))
            raise AssertionError("classify accepted a bogus gating")
        except ValueError:
            pass
        print("[ok] P2 periodic gating still bites: the dp window alone, contErr alone, and a"
              " partial gating tuple raises ValueError")
        ok += 1
        d = classify(texts["diverged"], hists["diverged"], rule, 600, 50, 2)
        assert d["class"] == "diverged" and d["reason_id"] == "SOLVE-DIVERGED"
        assert len(d["iterations"]) == 4
        assert d["iterations"][-1]["finite"] is False and d["iterations"][-1]["U_res"] == "nan"
        assert d["mach_max"] == 0.2412
        print("[ok] the diverged fixture keeps its non-finite row and reaches the diverged class")
        ok += 1
        m = classify(texts["refused_mach"], hists["refused_mach"], rule, 600, 50, 3)
        assert m["class"] == "refused" and m["reason_id"] == "SOLVE-MACH"
        assert m["mach_refusal"] == {"M": 0.312, "cell": 4711, "when": "step 118", "limit": 0.3}
        assert m["mach_max"] == 0.287
        print("[ok] the refused_mach fixture parses the refusal detail into mach_refusal")
        ok += 1

        def unst(t, h=None):
            return classify(t, H if h is None else h, rule, 600, 50, 0)

        r = unst(rep(ST, "|U| res 2.19539e-05", "|U| res 0.0001"))
        assert r["failed"] == ["U_decades"]
        assert r["criteria"]["U_decades"]["value"] == 3.941511432634403
        r = unst(rep(ST, "|p| res 5.012e-05", "|p| res 1.001e-04"))
        assert r["failed"] == ["p_decades"]
        assert r["criteria"]["p_decades"]["value"] == 3.999565922520681
        r = unst(rep(ST, "contErr 1.32501e-09", "contErr 1.5e-06"))
        assert r["failed"] == ["cont_err"] and r["criteria"]["cont_err"]["value"] == 1.5e-06
        h2 = [dict(row) for row in H]
        h2[0]["dp"] = h2[-1]["dp"] * 1.00002
        r = unst(ST, h2)
        assert r["failed"] == ["dp_rel_change"]
        assert r["criteria"]["dp_rel_change"]["value"] == 1.9999999999818947e-05
        h3 = [dict(row) for row in H]
        h3[2]["Cd"] = h3[-1]["Cd"] * (1 - 1.5e-5)
        r = unst(ST, h3)
        assert r["failed"] == ["Cd_rel_change"]
        assert r["criteria"]["Cd_rel_change"]["value"] == 1.500000000001862e-05
        r = unst(rep(ST, "contErr 1.32501e-09", "contErr 1e-06"))
        assert r["class"] == "steady" and r["criteria"]["cont_err"]["value"] == 1e-06
        r = unst(rep(rep(ST, "|U| res 0.874  ", "|U| res 1  "), "|U| res 2.19539e-05", "|U| res 0.0001"))
        assert r["class"] == "steady" and r["criteria"]["U_decades"]["value"] == 4.0
        print("[ok] each criterion bites alone at its exact value and the >= and <= edges stay steady")
        ok += 1

        def logsolve(t, rc=0, it=600):
            res = classify(t, H, rule, it, 50, rc)
            assert res["class"] == "refused" and res["reason_id"] == "SOLVE-LOG", (res["reason_id"], res["detail"])
            return res

        r = logsolve(chr(10).join(ST.splitlines()[:-1]))
        assert r["run_end"] is None and len(r["iterations"]) == 13
        r = logsolve(ST + "stray line" + chr(10))
        assert r["run_end"] is None and len(r["iterations"]) == 13
        r = logsolve(ST, rc=1)
        assert r["run_end"]["word"] == "budget" and len(r["iterations"]) == 13
        r = logsolve(rep(ST, "contErr 1.32501e-09", "contErr ?"))
        assert r["run_end"] is None and len(r["iterations"]) == 12
        r = logsolve(ST, it=650)
        assert r["run_end"]["word"] == "budget" and len(r["iterations"]) == 13
        r = logsolve(chr(10).join(l for l in ST.splitlines() if not l.startswith("iter    599")))
        assert r["run_end"]["word"] == "budget" and len(r["iterations"]) == 12
        r = logsolve(rep(ST, "exit code 0", "exit code 3"), rc=None)
        assert r["run_end"]["word"] == "budget" and len(r["iterations"]) == 13
        t = rep(ST, "iter    500", "@@SWAP@@")
        t = rep(t, "iter    550", "iter    500")
        r = logsolve(rep(t, "@@SWAP@@", "iter    550"))
        assert r["run_end"] is None and len(r["iterations"]) == 13
        print("[ok] truncated, trailing, mismatched, malformed, mis-budgeted and disordered logs are SOLVE-LOG")
        ok += 1

        h5 = [dict(row) for row in H if row["iter"] != 450]
        r = classify(ST, h5, rule, 600, 50, 0)
        assert r["class"] == "refused" and r["reason_id"] == "SOLVE-POST" and r["window"] is None
        h6 = [dict(row) for row in H]
        h6[2]["status"] = "refused"
        r = classify(ST, h6, rule, 600, 50, 0)
        assert r["reason_id"] == "SOLVE-POST" and r["window"] is None
        r = classify(rep(ST, "M max 0.0662017 (cell 20311)", "M max 0.31 (cell 20311)"), H, rule, 600, 50, 0)
        assert r["class"] == "refused" and r["reason_id"] == "SOLVE-MACH"
        assert r["mach_refusal"] == {"M": 0.31, "cell": 20311, "when": "iter 100", "limit": 0.3}
        mline = texts["refused_mach"]
        a0 = mline.index("run ended: refused | ") + len("run ended: refused | ")
        a1 = mline.index(" | exit code 3")
        t = mline.replace(mline[a0:a1], '-writeInterval: "10" is not supported by ofgpu')
        r = classify(t, [], rule, 600, 50, 3)
        assert r["class"] == "refused" and r["reason_id"] == "SOLVE-REFUSED" and r["mach_refusal"] is None
        t = rep(ST, "run ended: budget | 600 iterations reached | exit code 0",
                "run ended: error | boom | exit code 1")
        r = classify(t, H, rule, 600, 50, 1)
        assert r["class"] == "refused" and r["reason_id"] == "SOLVE-ERROR"
        assert r["run_end"]["word"] == "error"
        r = classify(rep(ST, "contErr 1.32501e-09", "contErr 1.32501e-109"), H, rule, 600, 50, 0)
        assert r["class"] == "steady" and r["criteria"]["cont_err"]["value"] == 1.32501e-109, r["reason_id"]
        print("[ok] an incomplete window, a Mach bite mid-log, a generic refusal and an error are classified")
        ok += 1

        solve_py = os.path.join(HERE, "solve.py")
        out1 = os.path.join(tmp, "cli_steady.json")
        p = subprocess.run([sys.executable, solve_py, "classify", os.path.join(F, "steady", "solve.log"),
                            os.path.join(F, "steady", "history.json"), "600", out1], capture_output=True)
        assert p.returncode == 0, p.stderr
        assert same(common.read_json(out1), exps["steady"]["result"])
        out2 = os.path.join(tmp, "cli_unsteady.json")
        p = subprocess.run([sys.executable, solve_py, "classify", os.path.join(F, "unsteady", "solve.log"),
                            os.path.join(F, "unsteady", "history.json"), "600", out2], capture_output=True)
        assert p.returncode == 1
        p = subprocess.run([sys.executable, solve_py, "nonsense"], capture_output=True)
        assert p.returncode == 2
        print("[ok] the classify CLI reproduces the fixtures in fresh processes and exits 0, 1 and 2")
        ok += 1

        def _fx_case(root):
            files_text = {
                "0/U": "internalField uniform (0 0 0);",
                "0/p": "internalField uniform 0;",
                "0/T": "internalField uniform 293.15;",
                "constant/polyMesh/boundary": "3(inlet{type patch;} outlet{type patch;} wall{type wall;})",
                "constant/polyMesh/faces": "0()",
                "constant/polyMesh/neighbour": "0()",
                "constant/polyMesh/owner": "0()",
                "constant/polyMesh/points": "0()",
                "constant/physicalProperties": "nu [0 2 -1 0 0 0 0] 1.5e-05;",
                "constant/momentumTransport": "laminar;",
                "system/controlDict": "application ofgpu-lowmach;",
                "system/fvSchemes": "ddtSchemes steadyState;",
                "system/fvSolution": "solver SIMPLE;",
            }
            root = os.path.abspath(root)
            os.makedirs(root, exist_ok=True)
            files = {}
            for rel, text in files_text.items():
                blob = text.encode("utf-8")
                target = os.path.join(root, rel.replace("/", os.sep))
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with open(target, "wb") as f:
                    f.write(blob)
                files[rel] = common.sha256_bytes(blob)
            case = {"version": "cad-case/1", "status": "ok", "cold_start": True,
                    "operating_point": {"p0_Pa": 101325.0}, "files": files}
            with open(os.path.join(root, "case.json"), "wb") as f:
                f.write((common.canonical_json(case) + chr(10)).encode("utf-8"))
            return root

        def _fx_stub(root):
            stub = os.path.join(os.path.abspath(root), "stub.py")
            os.makedirs(os.path.dirname(stub), exist_ok=True)
            L = chr(10)
            with open(stub, "w", encoding="utf-8", newline=L) as f:
                f.write(L.join([
                    "# a stub of the pinned solver: its behaviour is set by the environment",
                    "import json, os, sys",
                    "case = sys.argv[1]",
                    "count = os.environ.get('SOLVE_STUB_COUNT')",
                    "if count:",
                    "    with open(count, 'a', encoding='utf-8') as f:",
                    "        f.write(json.dumps(sys.argv[1:]) + chr(10))",
                    "times = os.environ.get('SOLVE_STUB_TIMES')",
                    "if times:",
                    "    for t in times.split(','):",
                    "        d = os.path.join(case, t)",
                    "        os.makedirs(d, exist_ok=True)",
                    "        with open(os.path.join(d, 'U'), 'w') as f:",
                    "            f.write('internalField uniform (0 0 0);' + chr(10))",
                    "touch = os.environ.get('SOLVE_STUB_TOUCH')",
                    "if touch:",
                    "    p = os.path.join(case, touch)",
                    "    os.makedirs(os.path.dirname(p), exist_ok=True)",
                    "    with open(p, 'ab') as f:",
                    "        f.write(b'x')",
                    "log = os.environ.get('SOLVE_STUB_LOG')",
                    "if log:",
                    "    with open(log, 'rb') as f:",
                    "        sys.stdout.buffer.write(f.read())",
                    "    sys.stdout.buffer.flush()",
                    "sys.exit(int(os.environ.get('SOLVE_STUB_RC', '0')))",
                ]) + L)
            return stub

        _STUB_ENV = ("SOLVE_STUB_COUNT", "SOLVE_STUB_TIMES", "SOLVE_STUB_TOUCH", "SOLVE_STUB_LOG",
                     "SOLVE_STUB_RC")

        def set_env(**kw):
            saved = dict((k, os.environ.pop(k, None)) for k in _STUB_ENV)
            for k, v in kw.items():
                os.environ[k] = v
            return saved

        def restore_env(saved):
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        stub = _fx_stub(os.path.join(tmp, "bin"))
        exe = [sys.executable, stub]

        def run_stub(tag, log_path, rc):
            case = _fx_case(os.path.join(tmp, "case_" + tag))
            geom = os.path.join(tmp, "geom_" + tag)
            os.makedirs(geom, exist_ok=True)
            out = os.path.join(tmp, "out_" + tag)
            count = os.path.join(tmp, "count_" + tag + ".txt")
            rec = []

            def rec_fn(case_dir, geom_dir, window):
                rec.append((case_dir, geom_dir, list(window)))
                return H

            saved = set_env(SOLVE_STUB_LOG=log_path, SOLVE_STUB_TIMES="400,450,500,550,600",
                            SOLVE_STUB_RC=str(rc), SOLVE_STUB_COUNT=count)
            try:
                doc = launch(case, geom, out, 600, exe=exe, visible=False, history_fn=rec_fn)
            finally:
                restore_env(saved)
            return doc, rec, count, out, case

        doc, rec, count, out, case7 = run_stub("a", os.path.join(F, "steady", "solve.log"), 0)
        assert doc["class"] == "steady" and doc["reason_id"] is None
        assert doc["gating"] == list(CRITERIA)
        assert common.read_json(os.path.join(out, "solve.json")) == doc
        assert doc["command"][2:] == [case7, "-iters", "600", "-check", "50", "-writeEvery", "50",
                                      "-p0", "101325.0"]
        with open(count, "r", encoding="utf-8") as f:
            stub_calls = [json.loads(l) for l in f]
        assert stub_calls == [doc["command"][2:]]
        with open(os.path.join(out, "solve.log"), "rb") as f:
            tee_bytes = f.read()
        with open(os.path.join(F, "steady", "solve.log"), "rb") as f:
            fx_bytes = f.read()
        assert tee_bytes == fx_bytes
        assert doc["log"]["sha256"] == common.sha256_bytes(fx_bytes) and doc["log"]["n_lines"] == 23
        assert doc["before"] == doc["after"] and len(doc["before"]) == 14
        assert doc["written"] == ["400", "450", "500", "550", "600"]
        assert rec == [(case7, os.path.abspath(os.path.join(tmp, "geom_a")), [400, 450, 500, 550, 600])]
        assert same(doc["result"], exps["steady"]["result"])
        assert doc["returncode"] == 0 and doc["binary"]["name"] == "exe"
        print("[ok] one cold launch runs the stub once with the exact argv and tees the log byte for byte")
        ok += 1

        # P3 (SPEC-LIT 114.5 O3): a case.json of kind "pipe" launches with CRITERIA_PERIODIC in the doc
        # and solve.json; the laminar stub above keeps CRITERIA; refusals carry gating None
        case_p = _fx_case(os.path.join(tmp, "case_pipe_kind"))
        cj_p = common.read_json(os.path.join(case_p, "case.json"))
        cj_p["kind"] = "pipe"
        with open(os.path.join(case_p, "case.json"), "wb") as f:
            f.write((common.canonical_json(cj_p) + chr(10)).encode("utf-8"))
        geom_p = os.path.join(tmp, "geom_pipe_kind")
        os.makedirs(geom_p, exist_ok=True)
        out_p = os.path.join(tmp, "out_pipe_kind")
        count_p = os.path.join(tmp, "count_pipe_kind.txt")
        saved = set_env(SOLVE_STUB_LOG=os.path.join(F, "steady", "solve.log"),
                        SOLVE_STUB_TIMES="400,450,500,550,600", SOLVE_STUB_RC="0",
                        SOLVE_STUB_COUNT=count_p)
        try:
            doc_p = launch(case_p, geom_p, out_p, 600, exe=exe, visible=False, history_fn=lambda d, g, w: H)
        finally:
            restore_env(saved)
        assert doc_p["class"] == "steady" and doc_p["reason_id"] is None
        assert doc_p["gating"] == list(CRITERIA_PERIODIC)
        assert common.read_json(os.path.join(out_p, "solve.json"))["gating"] == list(CRITERIA_PERIODIC)
        print("[ok] P3 a kind-pipe case launch carries gating %s in solve.json, the laminar stub"
              " keeps CRITERIA, and a pre-admission refusal keeps gating None"
              % (list(CRITERIA_PERIODIC),))
        ok += 1

        def count_lines(path):
            with open(path, "r", encoding="utf-8") as f:
                return len(f.readlines())

        def bind_case(tag, touch):
            case_b = _fx_case(os.path.join(tmp, "case8" + tag))
            geom_b = os.path.join(tmp, "geom8" + tag)
            os.makedirs(geom_b, exist_ok=True)
            out_b = os.path.join(tmp, "out8" + tag)
            count_b = os.path.join(tmp, "count8" + tag + ".txt")
            rec_b = []

            def rec_fn(case_dir, geom_dir, window):
                rec_b.append((case_dir, geom_dir, list(window)))
                return H

            saved = set_env(SOLVE_STUB_LOG=os.path.join(F, "steady", "solve.log"),
                            SOLVE_STUB_TIMES="400,450,500,550,600", SOLVE_STUB_RC="0",
                            SOLVE_STUB_COUNT=count_b, SOLVE_STUB_TOUCH=touch)
            try:
                doc_b = launch(case_b, geom_b, out_b, 600, exe=exe, visible=False, history_fn=rec_fn)
            finally:
                restore_env(saved)
            return doc_b, rec_b, count_b

        doc, rec, count = bind_case("a", "system/controlDict")
        assert doc["class"] == "refused" and doc["reason_id"] == "SOLVE-BIND"
        assert doc["after"]["system/controlDict"] != doc["before"]["system/controlDict"]
        assert doc["result"]["run_end"]["word"] == "budget"
        assert rec == [] and count_lines(count) == 1
        doc, rec, count = bind_case("b", "constant/extra")
        assert doc["reason_id"] == "SOLVE-BIND" and rec == [] and count_lines(count) == 1
        print("[ok] a case file changed or created during the run refuses the solve as SOLVE-BIND")
        ok += 1

        doc, rec, count, _o, _c = run_stub("c", os.path.join(F, "diverged", "solve.log"), 2)
        assert doc["class"] == "diverged" and doc["reason_id"] == "SOLVE-DIVERGED"
        assert rec == [] and count_lines(count) == 1
        doc, rec, count, _o, _c = run_stub("d", os.path.join(F, "refused_mach", "solve.log"), 3)
        assert doc["class"] == "refused" and doc["reason_id"] == "SOLVE-MACH"
        assert rec == [] and count_lines(count) == 1
        print("[ok] a diverged and a Mach refused run are classified and the binary never runs twice")
        ok += 1

        count10 = os.path.join(tmp, "count10.txt")
        saved = set_env(SOLVE_STUB_COUNT=count10)

        def expect(d, rid, json_out=None, gating=None):
            assert d["class"] == "refused" and d["reason_id"] == rid, (d["reason_id"], d["message"])
            assert d["command"] is None and d["returncode"] is None
            assert d["gating"] == gating
            assert not os.path.exists(count10)
            if json_out is not None:
                assert common.read_json(os.path.join(json_out, "solve.json")) == d

        try:
            case_x = _fx_case(os.path.join(tmp, "case10a"))
            d = launch(case_x, os.path.join(tmp, "g10a"), os.path.join(case_x, "sub"), 600, exe=exe,
                       visible=False)
            expect(d, "SOLVE-OUT")
            holder = os.path.join(tmp, "out10b")
            os.makedirs(holder)
            with open(os.path.join(holder, "occupied.txt"), "wb") as f:
                f.write(b"x")
            d = launch(case_x, os.path.join(tmp, "g10b"), holder, 600, exe=exe, visible=False)
            expect(d, "SOLVE-OUT")
            case_x = _fx_case(os.path.join(tmp, "case10c"))
            d = launch(case_x, os.path.join(tmp, "g10c"), os.path.join(tmp, "o10c"), 610, exe=exe,
                       visible=False)
            expect(d, "SOLVE-ARGS", os.path.join(tmp, "o10c"))
            d = launch(case_x, os.path.join(tmp, "g10c"), os.path.join(tmp, "o10d"), 150, exe=exe,
                       visible=False)
            expect(d, "SOLVE-ARGS", os.path.join(tmp, "o10d"))
            badbin = os.path.join(tmp, "bin_empty.json")
            common.write_json(badbin, {"binaries": {}})
            d = launch(case_x, os.path.join(tmp, "g10e"), os.path.join(tmp, "o10e"), 600, bin_json=badbin,
                       visible=False)
            expect(d, "SOLVE-BIN", os.path.join(tmp, "o10e"))
            badbin2 = os.path.join(tmp, "bin_sha.json")
            common.write_json(badbin2, {"binaries": {"ofgpu-lowmach": {"path": stub, "sha256": "0" * 64}}})
            d = launch(case_x, os.path.join(tmp, "g10f"), os.path.join(tmp, "o10f"), 600, bin_json=badbin2,
                       visible=False)
            expect(d, "SOLVE-BIN", os.path.join(tmp, "o10f"))

            case_x = _fx_case(os.path.join(tmp, "case10g"))
            with open(os.path.join(case_x, "restart.mcr"), "wb") as f:
                f.write(b"reset")
            d = launch(case_x, os.path.join(tmp, "g10g"), os.path.join(tmp, "o10g"), 600, exe=exe,
                       visible=False)
            expect(d, "SOLVE-COLD", os.path.join(tmp, "o10g"))
            case_x = _fx_case(os.path.join(tmp, "case10h"))
            os.makedirs(os.path.join(case_x, "600"))
            d = launch(case_x, os.path.join(tmp, "g10h"), os.path.join(tmp, "o10h"), 600, exe=exe,
                       visible=False)
            expect(d, "SOLVE-COLD", os.path.join(tmp, "o10h"))
            case_x = _fx_case(os.path.join(tmp, "case10i"))
            case_doc = common.read_json(os.path.join(case_x, "case.json"))
            case_doc["cold_start"] = False
            with open(os.path.join(case_x, "case.json"), "wb") as f:
                f.write((common.canonical_json(case_doc) + chr(10)).encode("utf-8"))
            d = launch(case_x, os.path.join(tmp, "g10i"), os.path.join(tmp, "o10i"), 600, exe=exe,
                       visible=False)
            expect(d, "SOLVE-COLD", os.path.join(tmp, "o10i"))
            case_x = _fx_case(os.path.join(tmp, "case10j"))
            with open(os.path.join(case_x, "0", "U"), "ab") as f:
                f.write(b" // edited after case.json")
            d = launch(case_x, os.path.join(tmp, "g10j"), os.path.join(tmp, "o10j"), 600, exe=exe,
                       visible=False)
            expect(d, "SOLVE-BIND", os.path.join(tmp, "o10j"), gating=list(CRITERIA))
            case_x = _fx_case(os.path.join(tmp, "case10k"))
            with open(os.path.join(case_x, "constant", "polyMesh", "extra"), "wb") as f:
                f.write(b"x")
            d = launch(case_x, os.path.join(tmp, "g10k"), os.path.join(tmp, "o10k"), 600, exe=exe,
                       visible=False)
            expect(d, "SOLVE-BIND", os.path.join(tmp, "o10k"), gating=list(CRITERIA))
        finally:
            restore_env(saved)
        print("[ok] the pre-launch refusals fire in order, write their doc and launch nothing")
        ok += 1

        fake = types.ModuleType("post")
        fake.UNDEFINED_ID = "POST-UNDEFINED"
        docs = {
            "400": {"status": "ok", "reason_id": None,
                    "metrics": {"dp": {"value": 1.5, "reason_id": None},
                                "Cd": {"value": 0.9, "reason_id": None}}},
            "450": {"status": "ok", "reason_id": None,
                    "metrics": {"dp": {"value": 7.0, "reason_id": "POST-UNDEFINED"},
                                "Cd": {"value": 0.8, "reason_id": None}}},
            "500": {"status": "refused", "reason_id": "POST-BIND", "metrics": None},
        }

        def fake_post(case_dir, time_name, geom_dir):
            return docs[time_name]

        fake.post = fake_post
        sys.modules["post"] = fake
        try:
            rows = history("c", "g", [400, 450, 500])
        finally:
            del sys.modules["post"]
        assert [r["iter"] for r in rows] == [400, 450, 500]
        assert rows[0] == {"iter": 400, "time": "400", "status": "ok", "reason_id": None, "dp": 1.5,
                           "Cd": 0.9, "post_sha256": common.sha256_of(docs["400"])}
        assert rows[1]["dp"] is None and rows[1]["Cd"] == 0.8
        assert rows[1]["post_sha256"] == common.sha256_of(docs["450"])
        assert rows[2]["status"] == "refused" and rows[2]["reason_id"] == "POST-BIND"
        assert rows[2]["dp"] is None and rows[2]["Cd"] is None
        assert rows[2]["post_sha256"] == common.sha256_of(docs["500"])
        print("[ok] history maps post docs to window rows with POST-UNDEFINED values as None")
        ok += 1

        trimmed = os.path.join(tmp, "killed.log")
        with open(trimmed, "wb") as f:
            f.write((chr(10).join(ST.splitlines()[:-1]) + chr(10)).encode("utf-8"))
        doc, rec, count, _o, _c = run_stub("e", trimmed, 1)
        assert doc["class"] == "refused" and doc["reason_id"] == "SOLVE-LOG"
        assert doc["result"]["run_end"] is None
        assert rec == [] and count_lines(count) == 1
        assert doc["returncode"] == 1 and doc["written"] == ["400", "450", "500", "550", "600"]
        print("[ok] a killed run without its run ended line is SOLVE-LOG and is never retried")
        ok += 1

        assert ok == 18, ok
        print("SELFTEST PASS")

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
