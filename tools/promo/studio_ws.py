#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.

"""The F1 promo studio workspace: one `build` lays out the tree the GUI
studio runs on, so tools/promo/capture.py can drive it headlessly.

Subcommands:
  build       --solve DIR --post DIR --sim DIR --gui-tree DIR --out DIR [--time 0.5]
  --selftest  runs T1-T6 on temp dirs only (no network, no GPU), < 60 s.

Layout under --out: ws/docs/schema/<every schema json of the gui tree>,
ws/cases/f1-promo/{constant/polyMesh/<the five files>, constant/<the two
transport files>, system/<every file>, <time>/<every file>} copied
byte-identical from the solve; <time>/Cp derived (Cp = (p - p_ref)/q_kin
over the internalField, p_ref/q_kin from post/numbers.json, the boundary
values converted line-for-line); ws/cases/f1-promo/post/{numbers.json,
forces.csv,residuals.csv}; ws/cases/f1-promo/LICENSE.txt, ws/geometry/
f1_sim.stl + ws/geometry/LICENSE.txt; state/server/tools.defaults.json;
state/runs/promo-solve-full/{run.json,residuals.jsonl,log.txt} - the solve
imported as a past run the studio's runs pane can replay; studio_ws.json,
the manifest; LICENSE.txt beside it all.

Refusals (exit 2, nothing written before every check passes): SW-OUT (--out
inside the repository or the gui tree, or non-empty and not a previous
output of this tool), SW-SOLVE (a required solve file missing), SW-POST
(numbers.json missing, not finite p_ref/q_kin, or a time mismatch), SW-SIM
(sim_surface.stl missing), SW-SCHEMA (the gui tree's case-1.json or
tools.defaults.json missing), SW-CP (non-finite Cp or a count that differs
from the owner header's nCells).

The mesh and geometry derive from a CC BY 4.0 model ("F1 2026 concept" by
Qvist_Designs, via Sketchfab), so NOTHING made from them is written inside
the repository: every output goes under --out, which must lie outside the
repository and the gui tree, with the model's LICENSE.txt copied beside it.
"""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import socket
import sys
import time
import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solve import REPO, read_boundary, read_ints, read_foam_scalar  # noqa: E402
from tunnel_mesh import refuse  # noqa: E402

TOOL = "tools/promo/studio_ws.py"
VERSION = "promo-studio-ws/1"
CASE_REL = "cases/f1-promo"
GEOM_REL = "geometry/f1_sim.stl"
RUN_ID = "promo-solve-full"
POLY_FILES = ["boundary", "faces", "owner", "neighbour", "points"]
END_WORDS = ("budget", "error", "refused", "diverged")
_VAL_LINE = re.compile(r"^(\s*value\s+uniform\s+)(\S+)(;.*)$")


# ---------------------------------------------------------------- helpers

def _posix(path):
    return os.path.abspath(path).replace(os.sep, "/")


def is_inside(child, parent):
    """True when child is parent or lies under it."""
    c = os.path.normcase(os.path.abspath(child))
    p = os.path.normcase(os.path.abspath(parent))
    if c == p:
        return True
    return c.startswith(p.rstrip("/" + os.sep) + os.sep)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _copy(src, dst, rel, copies):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    copies.append({"rel": rel, "bytes": os.path.getsize(dst),
                   "same": _sha256(src) == _sha256(dst)})


def owner_n_cells(path):
    """nCells from the owner header's note ("nCells:3235813")."""
    with open(path, encoding="utf-8", errors="replace") as f:
        head = f.read(8192)
    m = re.search(r"nCells:(\d+)", head)
    if not m:
        refuse("SW-SOLVE", path + ": owner header carries no nCells note")
    return int(m.group(1))


def iso_ms(epoch):
    """Epoch seconds -> ISO 8601 UTC with milliseconds and Z (rounded to
    the nearest millisecond first; isoformat would truncate)."""
    ms = int(round(epoch * 1000.0))
    dt = datetime.datetime.fromtimestamp(ms / 1000.0, tz=datetime.timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S") + ".%03dZ" % (dt.microsecond // 1000)


# ------------------------------------------------------------- pure: C2/C3

def cp_values(p, p_ref, q_kin):
    """Cp = (p - p_ref) / q_kin over the internalField values."""
    p = np.asarray(p, dtype=np.float64)
    return (p - float(p_ref)) / float(q_kin)


def cp_boundary_text(p_boundary_text, p_ref, q_kin):
    """p's boundaryField block with every `value uniform <v>;` converted,
    every other line identical; a nonuniform value refuses SW-CP."""
    out = []
    for line in p_boundary_text.split("\n"):
        s = line.strip()
        if s.startswith("value") and "nonuniform" in s:
            refuse("SW-CP", "boundaryField carries a nonuniform value; "
                    "Cp cannot convert it")
        m = _VAL_LINE.match(line)
        if m:
            v = float(m.group(2))
            out.append(m.group(1) + ("%.12g" % ((v - p_ref) / q_kin)) + m.group(3))
        else:
            out.append(line)
    return "\n".join(out)


def patch_uniform(p_boundary_text, patch):
    """The `value uniform <v>` of one patch's block, or None."""
    m = re.search(r"\n\s{4}" + re.escape(patch) + r"\s*\n\s*\{(.*?)\n\s*\}",
                  p_boundary_text, re.S)
    if not m:
        return None
    v = re.search(r"value\s+uniform\s+(\S+)\s*;", m.group(1))
    return float(v.group(1)) if v else None


def residual_records(csv_text):
    """One record per residuals.csv data row, seq from 1 (the studio's
    ResidualRecord shape, lowmach fields U/p/continuity)."""
    out = []
    for raw in csv_text.splitlines()[1:]:
        if not raw.strip():
            continue
        c = raw.split(",")
        out.append({
            "seq": len(out) + 1,
            "iter": int(c[0]),
            "time": None,
            "wall": None,
            "fields": {"U": float(c[1]), "p": float(c[2]),
                       "continuity": float(c[3])},
            "solverIters": None,
            "raw": raw,
        })
    return out


def run_info(log_text, returncode, wall_seconds, ended_epoch, command,
             log_lines, records, time_value, time_dir, hostname):
    """The studio's past-run RunInfo dict for the imported solve."""
    lines = [ln for ln in log_text.splitlines() if ln.strip()]
    first = lines[0] if lines else ""
    last = lines[-1] if lines else ""
    device = first.split("|")[1].strip() if first.count("|") >= 2 else ""
    end_word, end_detail = None, None
    if last.startswith("run ended:"):
        parts = last.split("|")
        w = parts[0][len("run ended:"):].strip()
        if w in END_WORDS:
            end_word = w
            if len(parts) >= 3:
                end_detail = parts[1].strip()
    target = None
    if end_detail:
        m = re.search(r"in\s+(\d+)\s+steps", end_detail)
        if m:
            target = int(m.group(1))
    last_fields = dict(records[-1]["fields"]) if records else None
    return {
        "id": RUN_ID,
        "binary": "ofgpu-lowmach",
        "argv": list(command[1:]),
        "cwd": CASE_REL,
        "casePath": CASE_REL,
        "outputRoot": CASE_REL,
        "status": "done" if returncode == 0 else "failed",
        "pid": None,
        "startedAt": iso_ms(ended_epoch - wall_seconds),
        "endedAt": iso_ms(ended_epoch),
        "exitCode": int(returncode),
        "signal": None,
        "iter": records[-1]["iter"] if records else 0,
        "targetIter": target,
        "time": time_value,
        "endTime": time_value,
        "lastResidual": last_fields,
        "written": [CASE_REL + "/" + time_dir],
        "error": None,
        "converged": False,
        "device": device,
        "logLines": int(log_lines),
        "mode": "real",
        "label": "PROMO-SOLVE (imported; run outside the studio)",
        "gitSha": None,
        "gitDirty": None,
        "caseId": CASE_REL,
        "meshId": None,
        "machine": {"hostname": hostname, "gpu": device, "platform": "win32"},
        "endWord": end_word,
        "endDetail": end_detail,
    }


# ------------------------------------------------------------------ build

def _check_inputs(a):
    """Every refusal, in order, before anything is written. Returns the
    parsed numbers.json, p_ref, q_kin."""
    if is_inside(a.out, REPO) or is_inside(a.out, a.gui_tree):
        refuse("SW-OUT", _posix(a.out) + " lies inside the repository or the gui tree")
    prev = os.path.join(a.out, "studio_ws.json")
    if os.path.isdir(a.out) and os.listdir(a.out) and not os.path.isfile(prev):
        refuse("SW-OUT", _posix(a.out) + " is not empty and not a previous output of this tool")
    solve = {}
    for rel in (["case/constant/polyMesh/" + f for f in POLY_FILES] +
                ["case/" + a.time + "/p", "residuals.csv", "forces.csv",
                 "solve.log", "run.json", "solve.json", "LICENSE.txt"]):
        p = os.path.join(a.solve, rel)
        if not os.path.isfile(p):
            refuse("SW-SOLVE", p + " is missing")
        solve[rel] = p
    for name in ("momentumTransport", "physicalProperties"):
        p = os.path.join(a.solve, "case", "constant", name)
        if not os.path.isfile(p):
            refuse("SW-SOLVE", p + " is missing")
        solve[name] = p
    sys_dir = os.path.join(a.solve, "case", "system")
    if not os.path.isdir(sys_dir):
        refuse("SW-SOLVE", sys_dir + " is missing")
    n_json = os.path.join(a.post, "numbers.json")
    if not os.path.isfile(n_json):
        refuse("SW-POST", n_json + " is missing")
    try:
        numbers = json.load(open(n_json, encoding="utf-8"))
    except Exception as e:
        refuse("SW-POST", n_json + " does not parse: " + repr(e))
    try:
        p_ref = float(numbers["params"]["p_ref"])
        q_kin = float(numbers["params"]["q_kin"])
        num_time = float(numbers["time"])
    except (KeyError, TypeError, ValueError) as e:
        refuse("SW-POST", n_json + " lacks params.p_ref/q_kin/time: " + repr(e))
    if not (math.isfinite(p_ref) and math.isfinite(q_kin) and q_kin > 0):
        refuse("SW-POST", "p_ref %r / q_kin %r not finite-positive" % (p_ref, q_kin))
    if abs(num_time - float(a.time)) > 1e-12:
        refuse("SW-POST", "numbers.time %r != the build time %r" % (num_time, a.time))
    sim_stl = os.path.join(a.sim, "sim_surface.stl")
    if not os.path.isfile(sim_stl):
        refuse("SW-SIM", sim_stl + " is missing")
    for p in (os.path.join(a.gui_tree, "docs", "schema", "case-1.json"),
              os.path.join(a.gui_tree, "gui", "server", "tools.defaults.json")):
        if not os.path.isfile(p):
            refuse("SW-SCHEMA", p + " is missing")
    return numbers, p_ref, q_kin, solve, sim_stl, sys_dir


def cmd_build(a):
    t0 = time.time()
    numbers, p_ref, q_kin, solve, sim_stl, sys_dir = _check_inputs(a)
    time_dir = a.time
    ws_cases = os.path.join(a.out, "ws", CASE_REL)

    owner = solve["case/constant/polyMesh/owner"]
    n_cells = owner_n_cells(owner)
    p_text = open(solve["case/" + time_dir + "/p"], encoding="utf-8",
                  errors="replace").read()
    m = re.search(r"^boundaryField\s*$", p_text, re.M)
    if not m:
        refuse("SW-SOLVE", "p: no boundaryField block")
    p_boundary = p_text[m.start():]
    cp = cp_values(read_foam_scalar(solve["case/" + time_dir + "/p"], n_cells),
                   p_ref, q_kin)
    if cp.size != n_cells or not bool(np.all(np.isfinite(cp))):
        refuse("SW-CP", "Cp count %d != nCells %d or non-finite values"
               % (cp.size, n_cells))
    cp_boundary = cp_boundary_text(p_boundary, p_ref, q_kin)
    patches = read_boundary(solve["case/constant/polyMesh/boundary"])
    print("[studio_ws] checks %.1f s" % (time.time() - t0))

    if os.path.isfile(os.path.join(a.out, "studio_ws.json")):
        for sub in ("ws", "state"):
            p = os.path.join(a.out, sub)
            if os.path.isdir(p):
                shutil.rmtree(p)
    copies = []
    for src in sorted(os.listdir(os.path.join(a.gui_tree, "docs", "schema"))):
        if src.endswith(".json"):
            _copy(os.path.join(a.gui_tree, "docs", "schema", src),
                  os.path.join(a.out, "ws", "docs", "schema", src),
                  "ws/docs/schema/" + src, copies)
    for f in POLY_FILES:
        _copy(solve["case/constant/polyMesh/" + f],
              os.path.join(ws_cases, "constant", "polyMesh", f),
              CASE_REL + "/constant/polyMesh/" + f, copies)
    for name in ("momentumTransport", "physicalProperties"):
        _copy(solve[name], os.path.join(ws_cases, "constant", name),
              CASE_REL + "/constant/" + name, copies)
    for src in sorted(os.listdir(sys_dir)):
        if os.path.isfile(os.path.join(sys_dir, src)):
            _copy(os.path.join(sys_dir, src),
                  os.path.join(ws_cases, "system", src),
                  CASE_REL + "/system/" + src, copies)
    t_dir = os.path.join(a.solve, "case", time_dir)
    for src in sorted(os.listdir(t_dir)):
        if os.path.isfile(os.path.join(t_dir, src)):
            _copy(os.path.join(t_dir, src),
                  os.path.join(ws_cases, time_dir, src),
                  CASE_REL + "/" + time_dir + "/" + src, copies)
    post_dir = os.path.join(ws_cases, "post")
    _copy(os.path.join(a.post, "numbers.json"),
          os.path.join(post_dir, "numbers.json"),
          CASE_REL + "/post/numbers.json", copies)
    for f in ("forces.csv", "residuals.csv"):
        _copy(solve[f], os.path.join(post_dir, f), CASE_REL + "/post/" + f, copies)
    _copy(solve["LICENSE.txt"], os.path.join(ws_cases, "LICENSE.txt"),
          CASE_REL + "/LICENSE.txt", copies)
    _copy(sim_stl, os.path.join(a.out, "ws", "geometry", "f1_sim.stl"),
          GEOM_REL, copies)
    _copy(solve["LICENSE.txt"], os.path.join(a.out, "ws", "geometry", "LICENSE.txt"),
          "geometry/LICENSE.txt", copies)
    _copy(os.path.join(a.gui_tree, "gui", "server", "tools.defaults.json"),
          os.path.join(a.out, "state", "server", "tools.defaults.json"),
          "state/server/tools.defaults.json", copies)
    _copy(solve["LICENSE.txt"], os.path.join(a.out, "LICENSE.txt"),
          "LICENSE.txt", copies)
    print("[studio_ws] copy %.1f s" % (time.time() - t0))
    return _finish(a, t0, numbers, p_ref, q_kin, cp, cp_boundary, patches,
                   owner, n_cells, copies, solve, time_dir)


def _finish(a, t0, numbers, p_ref, q_kin, cp, cp_boundary, patches, owner,
            n_cells, copies, solve, time_dir):
    ws_cases = os.path.join(a.out, "ws", CASE_REL)
    cp_path = os.path.join(ws_cases, time_dir, "Cp")
    with open(cp_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("FoamFile\n{\n    format      ascii;\n"
                "    class       volScalarField;\n    location    \"" +
                time_dir + "\";\n    object      Cp;\n}\n\n")
        f.write("dimensions      [0 0 0 0 0 0 0];\n\n")
        f.write("internalField   nonuniform List<scalar> \n%d\n(\n" % cp.size)
        step = 200000
        for k in range(0, cp.size, step):
            f.write("".join("%.12g\n" % v for v in cp[k:k + step]))
        f.write(")\n;\n")
        f.write(cp_boundary)
        if not cp_boundary.endswith("\n"):
            f.write("\n")
    print("[studio_ws] cp %.1f s" % (time.time() - t0))

    run_dir = os.path.join(a.out, "state", "runs", RUN_ID)
    os.makedirs(run_dir, exist_ok=True)
    run_text = open(solve["solve.log"], encoding="utf-8", errors="replace").read()
    records = residual_records(
        open(solve["residuals.csv"], encoding="utf-8", errors="replace").read())
    run_meta = json.load(open(solve["run.json"], encoding="utf-8"))
    solve_json = json.load(open(solve["solve.json"], encoding="utf-8"))
    log_lines = len(run_text.splitlines())
    info = run_info(run_text, int(run_meta.get("returncode", 1)),
                    float(run_meta.get("wall_seconds", 0.0)),
                    os.path.getmtime(solve["solve.log"]),
                    list(solve_json.get("command") or []), log_lines,
                    records, float(time_dir), time_dir, socket.gethostname())
    with open(os.path.join(run_dir, "run.json"), "w", encoding="utf-8",
              newline="\n") as f:
        json.dump(info, f, indent=1)
        f.write("\n")
    with open(os.path.join(run_dir, "residuals.jsonl"), "w",
              encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    shutil.copyfile(solve["solve.log"], os.path.join(run_dir, "log.txt"))
    print("[studio_ws] run %.1f s" % (time.time() - t0))

    owner_vals = read_ints(owner)
    per_patch = {}
    for name, nf, sf in patches:
        if nf <= 0:
            continue
        v = cp[owner_vals[sf:sf + nf]]
        per_patch[name] = {"n": int(nf), "min": float(v.min()),
                           "max": float(v.max())}
    # cp_boundary already carries the converted uniform values
    ov = patch_uniform(cp_boundary, "outlet")
    manifest = {
        "tool": TOOL, "version": VERSION,
        "solve_dir": _posix(a.solve), "post_dir": _posix(a.post),
        "sim_dir": _posix(a.sim), "gui_tree": _posix(a.gui_tree),
        "out_dir": _posix(a.out),
        "time": float(time_dir), "case": CASE_REL, "geometry": GEOM_REL,
        "attribution": solve_json.get("attribution"),
        "cells": n_cells, "p_ref": p_ref, "q_kin": q_kin,
        "cp": {"n": int(cp.size), "min": float(cp.min()),
               "max": float(cp.max()), "mean": float(cp.mean()),
               "sum": float(cp.sum()), "first": float(cp[0]),
               "per_patch": per_patch,
               "outlet_value": ov},
        "copies": copies,
        "run": {"id": RUN_ID, "residuals": len(records),
                "log_lines": log_lines, "status": info["status"],
                "converged": False},
        "seconds": round(time.time() - t0, 3),
    }
    with open(os.path.join(a.out, "studio_ws.json"), "w", encoding="utf-8",
              newline="\n") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")
    print("[studio_ws] manifest %.1f s" % (time.time() - t0))
    print("BUILD OK %.1f s" % (time.time() - t0))
    return manifest


# --------------------------------------------------------------- selftest

T2_BLOCK = """boundaryField
{
    inlet
    {
        type            zeroGradient;
    }
    outlet
    {
        type            fixedValue;
        value           uniform 0;
    }
    top
    {
        type            slip;
    }
}"""
T1_P = [18.5753, 18.5614, 18.5381]
P_REF = 18.058075173611112
Q_KIN = 2411.2654320987654
T3_CSV = ("step,U_res,p_res,cont_err,M_max,M_cell\n"
          "0,0.0203503,0.03537,0.000457716,0.997176,971629\n"
          "1,0.0139745,0.05724,0.000551187,0.604966,62536\n")
T3_FIRST = ('{"seq": 1, "iter": 0, "time": null, "wall": null, "fields": '
            '{"U": 0.0203503, "p": 0.03537, "continuity": 0.000457716}, '
            '"solverIters": null, "raw": '
            '"0,0.0203503,0.03537,0.000457716,0.997176,971629"}')


def make_fixture(root):
    """A tiny synthetic solve/post/sim/gui tree for T5/T6."""
    def w(rel, text):
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w", encoding="utf-8", newline="\n").write(text)
    foam = ('FoamFile\n{\n    version     2.0;\n    format      ascii;\n'
            '    class       labelList;\n    note        "nPoints:4  nCells:3'
            '  nFaces:3  nInternalFaces:0";\n    object      owner;\n}\n3\n(\n0\n1\n2\n)\n')
    w("solve/case/constant/polyMesh/owner", foam)
    for f in ("boundary", "faces", "neighbour", "points"):
        w("solve/case/constant/polyMesh/" + f, "FoamFile\n{\n}\n0\n(\n)\n")
    w("solve/case/constant/polyMesh/boundary",
      'FoamFile\n{\n    class       polyBoundaryMesh;\n    object      boundary;\n}\n'
      '3\n(\n    inlet\n    {\n        type            patch;\n'
      '        nFaces          1;\n        startFace       0;\n    }\n'
      '    outlet\n    {\n        type            patch;\n'
      '        nFaces          1;\n        startFace       1;\n    }\n'
      '    walls\n    {\n        type            wall;\n'
      '        nFaces          1;\n        startFace       2;\n    }\n)\n')
    for f in ("T", "U", "k", "nut", "omega", "rho"):
        w("solve/case/0.5/" + f, "FoamFile\n{\n}\ninternalField   uniform 0;\n")
    w("solve/case/0.5/p",
      'FoamFile\n{\n    format      ascii;\n    class       volScalarField;\n'
      '    location    "0.5";\n    object      p;\n}\n'
      'dimensions      [0 2 -2 0 0 0 0];\n'
      'internalField   nonuniform List<scalar> \n3\n(\n18.5753\n18.5614\n18.5381\n)\n;\n'
      + T2_BLOCK + "\n")
    w("solve/case/constant/momentumTransport", "FoamFile\n{\n}\n")
    w("solve/case/constant/physicalProperties", "FoamFile\n{\n}\n")
    w("solve/case/system/controlDict", "application     ofgpu;\n")
    w("solve/residuals.csv", T3_CSV)
    w("solve/forces.csv", "time,fx\n0.5,1\n")
    w("solve/solve.log",
      "ofgpu lowmach | NVIDIA GeForce RTX 5070 Ti sm_120 | 16302 MiB | precision double\n"
      "run ended: budget | endTime 0.5 s reached in 1000 steps | exit code 0\n")
    w("solve/run.json", '{"returncode": 0, "wall_seconds": 12.5}')
    w("solve/solve.json",
      '{"tool": "t", "command": ["X.exe", "case", "-endTime", "0.5"], '
      '"attribution": "attr"}')
    w("solve/LICENSE.txt", "CC BY 4.0 fixture\n")
    w("post/numbers.json",
      '{"params": {"p_ref": %r, "q_kin": %r}, "time": 0.5}' % (P_REF, Q_KIN))
    w("sim/sim_surface.stl", "solid s\nendsolid s\n")
    w("gui/docs/schema/case-1.json", "{}\n")
    w("gui/gui/server/tools.defaults.json", "{}\n")
    return (os.path.join(root, "solve"), os.path.join(root, "post"),
            os.path.join(root, "sim"), os.path.join(root, "gui"))


def _selftest():
    import contextlib
    import io
    import subprocess
    import tempfile
    t0 = time.time()
    ok = []

    # T1 cp_values formatting
    vals = ["%.12g" % v for v in cp_values(np.array(T1_P), P_REF, Q_KIN)]
    assert vals == ["0.00021450348", "0.000208738872", "0.000199075896"], vals
    ok.append("T1")

    # T2 cp_boundary_text: only the outlet value line converts; nonuniform refuses
    out = cp_boundary_text(T2_BLOCK, P_REF, Q_KIN)
    exp = T2_BLOCK.replace("uniform 0;", "uniform -0.007489044936;")
    assert out == exp, out
    bad = T2_BLOCK.replace("value           uniform 0;",
                           "value           nonuniform List<scalar> 2(1 2);")
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            cp_boundary_text(bad, P_REF, Q_KIN)
        raise AssertionError("nonuniform value was not refused")
    except SystemExit as e:
        assert e.code == 2 and "SW-CP" in buf.getvalue(), buf.getvalue()
    ok.append("T2")

    # T3 residual_records
    recs = residual_records(T3_CSV)
    assert len(recs) == 2 and recs[1]["seq"] == 2 and recs[1]["iter"] == 1
    assert abs(recs[1]["fields"]["U"] - 0.0139745) < 1e-15
    assert json.dumps(recs[0]) == T3_FIRST, json.dumps(recs[0])
    ok.append("T3")

    # T4 run_info
    info = run_info(
        "ofgpu lowmach | NVIDIA GeForce RTX 5070 Ti sm_120 | 16302 MiB | precision double\n"
        "run ended: budget | endTime 0.5 s reached in 1000 steps | exit code 0\n",
        0, 8216.01832150016, 1791106211.847, ["X.exe", "case", "-endTime", "0.5"],
        1142, recs, 0.5, "0.5", "host")
    assert info["status"] == "done" and info["device"] == "NVIDIA GeForce RTX 5070 Ti sm_120"
    assert info["endWord"] == "budget"
    assert info["endDetail"] == "endTime 0.5 s reached in 1000 steps"
    assert info["targetIter"] == 1000 and info["iter"] == 1
    assert info["startedAt"] == "2026-10-04T07:13:15.829Z", info["startedAt"]
    assert info["endedAt"] == "2026-10-04T09:30:11.847Z", info["endedAt"]
    assert info["argv"] == ["case", "-endTime", "0.5"] and info["converged"] is False
    keys = ("id binary argv cwd casePath outputRoot status pid startedAt endedAt "
            "exitCode signal iter targetIter time endTime lastResidual written "
            "error converged device logLines mode label gitSha gitDirty caseId "
            "meshId machine endWord endDetail").split()
    assert sorted(info) == sorted(keys), sorted(set(info) ^ set(keys))
    ok.append("T4")
    print("[studio_ws] selftest T1-T4 %.1f s" % (time.time() - t0))

    this = os.path.abspath(__file__)

    def run_build(args):
        return subprocess.run([sys.executable, this, "build"] + args,
                              capture_output=True, text=True)

    # T5 refusals in subprocesses on temp dirs
    with tempfile.TemporaryDirectory() as root:
        solve, post, sim, gui = make_fixture(root)
        r = run_build(["--solve", solve, "--post", post, "--sim", sim,
                       "--gui-tree", gui, "--out", os.path.join(REPO, "sw_selftest_out")])
        assert r.returncode == 2 and "refused: SW-OUT" in r.stdout, r.stdout + r.stderr
        os.makedirs(os.path.join(root, "s2"), exist_ok=True)
        solve2, post2, sim2, gui2 = make_fixture(os.path.join(root, "s2"))
        os.remove(os.path.join(solve2, "case", "0.5", "p"))
        out2 = os.path.join(root, "out2")
        r = run_build(["--solve", solve2, "--post", post2, "--sim", sim2,
                       "--gui-tree", gui2, "--out", out2])
        assert r.returncode == 2 and "refused: SW-SOLVE" in r.stdout, r.stdout + r.stderr
        solve3, post3, sim3, gui3 = make_fixture(os.path.join(root, "s3"))
        nj = os.path.join(post3, "numbers.json")
        open(nj, "w", encoding="utf-8").write(
            '{"params": {"p_ref": %r, "q_kin": %r}, "time": 0.25}' % (P_REF, Q_KIN))
        r = run_build(["--solve", solve3, "--post", post3, "--sim", sim3,
                       "--gui-tree", gui3, "--out", os.path.join(root, "out3")])
        assert r.returncode == 2 and "refused: SW-POST" in r.stdout, r.stdout + r.stderr
    ok.append("T5")
    print("[studio_ws] selftest T5 %.1f s" % (time.time() - t0))

    # T6 end-to-end on the tiny synthetic case
    with tempfile.TemporaryDirectory() as root:
        solve, post, sim, gui = make_fixture(root)
        out = os.path.join(root, "out")
        r = run_build(["--solve", solve, "--post", post, "--sim", sim,
                       "--gui-tree", gui, "--out", out])
        assert r.returncode == 0, r.stdout + r.stderr
        man = json.load(open(os.path.join(out, "studio_ws.json"), encoding="utf-8"))
        assert man["copies"] and all(c["same"] for c in man["copies"])
        assert man["cp"]["n"] == 3 and man["cells"] == 3
        got = read_foam_scalar(os.path.join(out, "ws", CASE_REL, "0.5", "Cp"), 3)
        exp = cp_values(np.array(T1_P), P_REF, Q_KIN)
        assert float(np.max(np.abs(got - exp))) < 1e-12, got
        rd = os.path.join(out, "state", "runs", RUN_ID)
        for f in ("run.json", "residuals.jsonl", "log.txt"):
            assert os.path.isfile(os.path.join(rd, f)), f
    ok.append("T6")
    print("SELFTEST PASS %d/%d (%.1f s)" % (len(ok), 6, time.time() - t0))


def main():
    ap = argparse.ArgumentParser(description=TOOL)
    sub = ap.add_subparsers(dest="cmd")
    b = sub.add_parser("build")
    for name in ("solve", "post", "sim", "gui-tree", "out"):
        b.add_argument("--" + name, required=True)
    b.add_argument("--time", default="0.5")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    elif a.cmd == "build":
        cmd_build(a)
    else:
        ap.print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()
