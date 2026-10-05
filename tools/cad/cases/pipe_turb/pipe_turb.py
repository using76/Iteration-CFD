#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""pipe_turb.py - TG0 (docs/16 §H.5, §I): the fully developed turbulent pipe on the nozzle's
wedge machinery.

The periodic 5-degree pipe wedge of pipe_mesh.py (R 0.025 m, periodic length 0.1 m, 4 axial
cells, radial 40/80/160 graded to the wall, y+1 <= 1) is written as a cold cad-case-turb/1 of
case_writer.write_pipe_case at Re_tau 576.69 (u_tau 0.34971 m/s, g_x 9.78348 m/s^2) and
Re_tau 2358.00 (u_tau 1.42989 m/s, g_x 163.567 m/s^2) plus the report-only Re_tau 550 DNS leg,
solved with the pinned D-1 binary through solve.launch, and judged on NACA TM 1218 (Schlichting,
Lecture series "Boundary layer theory", Part II - Turbulent flows, 1949,
https://ntrs.nasa.gov/citations/20050040758): TB1 f within 5 % of Prandtl's universal law
1/sqrt(f) = 2.0 log10(Re sqrt(f)) - 0.8 (eq. 16.26) at the input Re_tau, TB2 f within 5 % of
Blasius f = 0.3164 Re_D^(-1/4) (eq. 16.4) at the MEASURED Re_D, TB3 u+ within 1.0 wall unit of
u+ = 2.5 ln y+ + 5.5 (eq. 16.14, Nikuradse's kappa 0.4, B 5.5) with at least one band cell,
TB4 (U_axis - U_b)/u_tau = 4.07 within 10 % (eq. 16.22), each on top of TB0's 1 % wall-shear
force balance and 1e-5 x-invariance and of a steady solve class; Nikuradse's smooth-pipe
measurements stand behind the laws (NASA TT F-10,359) and TB5 is Roache 1997's three-grid GCI
(DOI 10.1115/1.2910291) of f over L0, L1, L2 per gated tag: monotone with gci_fine at most 1 %.

Definitions (fixed before any run):
  names L0_lo, L1_lo, L2_lo, L0_hi, L1_hi, L2_hi, L0_dns; the tag gives Re_tau {lo 576.69,
  hi 2358.0, dns 550.0} and the prefix the level; the gate level is L2 and the gated tags are
  lo and hi; the iteration budgets are 4000/8000/20000 per level, fixed before any run.
  build meshes the three levels with pipe_mesh.run (TG0-MESH when not ok or gc7_pass not True),
  writes each requested case (TG0-CASE when one is refused, the message carrying its rule) and
  build.json; TG0-OUT when out_dir exists and is not empty.  run solves cases/<name> into
  runs/<name> through solve.launch with nvidia-smi before and after exactly as poiseuille.run
  (TG0-NAME for a bad name, TG0-OUT when the run directory exists).  history posts each window
  time into a row {"iter", "time", "status", "reason_id", dp = U_b_m_s, Cd = f, post_sha256},
  so the stop rule's 1e-5 window changes apply to U_b and f.  The final time of a run is
  solve.json's iters; nut_boundary is "empty" when that time's nut has no patch entry in its
  parsed boundaryField, else "written", None when the nut is absent.  judge_run checks TB0-TB4
  and the class; tb5 is gci3's row
  plus pass; the TG0 verdict is PASS iff the G0 record's verdict is PASS and both gated tags'
  L2 reasons are empty and TB5 passes, else OPEN with the ids present in VERDICT_IDS order;
  per tag the record also reports the finest run with an ok pipe, indicative only.  reduced is
  FULL only when every name has a solve.json.  TB6 reports U_b+ and the profile against the
  El Khoury et al. 2013 DNS row when L0_dns has an ok pipe (the DNS files state no licence and
  are not in this tree; nothing is compared); TB7 is the constant NOT_RUN wall-function row;
  TB8 reports whether the first run in NAMES order whose final time directory exists carries
  a wall-distance field, against y_axis_ref_m = R/sqrt(2), Tucker's Poisson y on the axis.
  TG0-REDO: the pipe cases write k and omega at tolerance 1e-12 (SPEC-LIT 114.5 O1), solve.launch
  classifies them by contErr and the U_b / f window (O3), and the record's model_form quotes
  SPEC-LIT 114.3.

Usage:
  python pipe_turb.py --selftest
  python pipe_turb.py build OUT_DIR
  python pipe_turb.py run OUT_DIR NAME [ITERS]     (NAME one of L0_lo..L0_dns; the GPU solve, supervisor only)
  python pipe_turb.py record OUT_DIR RECORD_JSON
  python pipe_turb.py stage-bin               (copy the pinned GPU binary to its single-link path)
"""
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAD = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, CAD)
sys.path.insert(0, os.path.join(CAD, "cases", "poiseuille"))
sys.path.insert(0, os.path.join(CAD, "cases", "nozzle_nominal"))
import common
import case_writer
import wedge_mesh
import post
import solve
import pipe_mesh
import turb_integral
import poiseuille
import nozzle_nominal

VERSION = "cad-tg0/1"
NAMES = ("L0_lo", "L1_lo", "L2_lo", "L0_hi", "L1_hi", "L2_hi", "L0_dns")
RE_OF = {"lo": 576.69, "hi": 2358.0, "dns": 550.0}
TAGS_GATED = ("lo", "hi")
LEVELS = (0, 1, 2)
GATE_LEVEL = 2
ITERS = {0: 4000, 1: 8000, 2: 20000}          # each level's -iters budget, fixed before any run
BANDS = {"balance_rel": 0.01, "x_invariance": 1e-5, "f_prandtl_rel": 0.05, "f_blasius_rel": 0.05,
         "loglaw_dev": 1.0, "core_defect": 4.07, "core_rel": 0.10, "gci_fine_f": 0.01}
Y_AXIS_REF_M = pipe_mesh.RECIPE["R_m"] / math.sqrt(2.0)     # docs/16 §H.5: Tucker's Poisson y on the axis
WALLDIST_NAMES = ("y", "wallDist", "yWall", "nearWallDist")
G0_RECORD = os.path.join(CAD, "cases", "poiseuille", "g0_record.json")
REFUSAL_IDS = ("TG0-OUT", "TG0-NAME", "TG0-MESH", "TG0-CASE")
VERDICT_IDS = ("TG0-G0", "TG0-MISSING", "TG0-UNSTEADY", "TG0-TB0", "TG0-TB1", "TG0-TB2", "TG0-TB3",
               "TG0-TB4", "TG0-TB5")
BUILD_KEYS = ("version", "pipe_recipe_sha", "mesh_report_sha256", "levels", "cases")
BUILD_LEVEL_KEYS = ("level", "nr", "cells", "h1_max_m", "yplus1_max", "check_tau_min", "gc7_pass")
BUILD_CASE_KEYS = ("name", "re_tau", "level", "u_tau_m_s", "g_x_m_s2", "U_b_case_m_s", "yplus1_apriori",
                   "case_sha256")
RECORD_KEYS = ("version", "pipe_recipe_sha", "bands", "iters", "gate_level", "binary", "g0_verdict",
               "reduced", "runs", "tb5", "tg0", "tb6", "tb7", "tb8", "model_form")
RUN_KEYS = ("name", "re_tau", "level", "cells", "solve", "wall_s", "gpu", "nut_boundary", "post", "checks")
SOLVE_ROW_KEYS = ("class", "reason_id", "failed", "gating", "criteria", "iters", "n_iter_lines", "log_sha256",
                  "binary_sha256")
POST_ROW_KEYS = ("status", "reason_id", "U_b_m_s", "Re_D", "u_tau_m_s", "Re_tau", "f", "U_axis_m_s",
                 "core_defect", "x_invariance", "balance_rel", "loglaw", "yplus1_max")
CHECK_KEYS = ("steady", "TB0", "TB1", "TB2", "TB3", "TB4", "f_rel_prandtl", "f_rel_blasius", "core_rel",
              "reasons")
TG0_KEYS = ("verdict", "reasons", "per_tag")
PER_TAG_KEYS = ("gate_run", "gate_reasons", "tb5_pass", "finest_run", "finest_reasons")
DNS_ROW = {"status": "not closed", "missing": ["U_b_plus", "u_plus(y_plus)"],
           "reference": "El Khoury et al. 2013, Flow Turb. Combust. 91, 475-495, DOI 10.1007/s10494-013-9482-8",
           "source": "https://www.lstm.tf.fau.de/database/simulation-database/",
           "note": "the DNS files state no licence and are not in this tree; nothing is compared"}
TB7_ROW = {"status": "NOT_RUN",
           "reasons": ["pipe_mesh has no y+1 30-60 recipe; the three TG0 levels are resolved-wall meshes",
                       "the solver writes nut without a boundaryField, so a wall-function wall nut is not "
                       "readable and post_turb refuses it by POST-FIELD"]}
MODEL_FORM = {"bands": ["TB1", "TB3", "TB4"], "source": "SPEC-LIT 114.3, solver tree f3f145c",
              "quote": "No mesh and no solver setting reaches 4.07 within 10 % with this model, and the f and"
                       " log-law bands at Re_tau 576.69 are short by the model too, +5.81 % and 1.239.",
              "reading": "kOmegaSST's centreline eddy viscosity is 0.124 to 0.131 of u_tau R against"
                         " Reichardt's kappa/6 = 0.0667 (SPEC-LIT 114.3 table 3); TB1, TB3 and TB4 are measured"
                         " model-form results for SST and their bands are not loosened"}
USAGE = ("usage: python pipe_turb.py --selftest" + chr(10)
         + "       python pipe_turb.py build OUT_DIR" + chr(10)
         + "       python pipe_turb.py run OUT_DIR NAME [ITERS]" + chr(10)
         + "       python pipe_turb.py record OUT_DIR RECORD_JSON" + chr(10)
         + "       python pipe_turb.py stage-bin")


class Refused(wedge_mesh.Refused):
    """A refusal by id (case_writer.Refused's shape): rule and detail."""


def level_of(name):
    """The (level, tag) of a name like L0_lo: the prefix's digit and the part after the underscore."""
    prefix, tag = name.split("_")
    return int(prefix[1:]), tag


def build(out_dir, names=NAMES):
    """The three pipe meshes, then one cad-case-turb/1 case per requested name, then build.json;
    TG0-OUT unless out_dir is missing or empty, TG0-MESH when the mesh run is not ok or its
    gc7_pass is not True, TG0-CASE when a case is refused (the message carries its rule)."""
    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise Refused("TG0-OUT", "%s exists and is not empty" % out_dir)
    os.makedirs(out_dir, exist_ok=True)
    res = pipe_mesh.run(os.path.join(out_dir, "mesh"), levels=LEVELS)
    if res["status"] != "ok" or res["report"].get("gc7_pass") is not True:
        raise Refused("TG0-MESH", "%s: %s" % (res["rule"], res["message"]))
    report = res["report"]
    levels = []
    for row in report["levels"]:
        levels.append({"level": row["level"], "nr": row["nr"], "cells": row["cells"],
                       "h1_max_m": row["h1_max_m"], "yplus1_max": list(row["yplus1_max"]),
                       "check_tau_min": row["check"]["tau_min"], "gc7_pass": row["gc7"]["pass"]})
    cases = []
    for name in names:
        level, tag = level_of(name)
        cdir = os.path.join(out_dir, "cases", name)
        w = case_writer.write_pipe_case(os.path.join(out_dir, "mesh"), level, RE_OF[tag], cdir)
        if w["status"] != "ok":
            raise Refused("TG0-CASE", "%s refused by %s: %s" % (name, w["rule"], w["message"]))
        cdoc = w["case"]
        op, turb = cdoc["operating_point"], cdoc["turbulence"]
        cases.append({"name": name, "re_tau": RE_OF[tag], "level": level,
                      "u_tau_m_s": op["u_tau_m_s"], "g_x_m_s2": op["g_x_m_s2"],
                      "U_b_case_m_s": op["U_b_m_s"], "yplus1_apriori": turb["yplus1_apriori"],
                      "case_sha256": common.sha256_file(os.path.join(cdir, "case.json"))})
    doc = {"version": VERSION, "pipe_recipe_sha": report["recipe_sha"],
           "mesh_report_sha256": common.sha256_file(os.path.join(out_dir, "mesh", "pipe_mesh.json")),
           "levels": levels, "cases": cases}
    with open(os.path.join(out_dir, "build.json"), "wb") as f:
        f.write((common.canonical_json(doc) + chr(10)).encode("utf-8"))
    return doc


def nut_boundary(time_dir):
    """None when the time directory has no nut, else "empty" when the written field's PARSED
    boundaryField has no patch entry (the solver's format, post.nut_boundary_empty), else
    "written"; the mesh sizes come from the case's own constant/polyMesh."""
    path = os.path.join(time_dir, "nut")
    if not os.path.isfile(path):
        return None
    mesh = post.load_mesh(os.path.join(os.path.dirname(time_dir), "constant", "polyMesh"))
    patch_sizes = dict((name, rng[1]) for name, rng in mesh["patch_range"].items())
    return "empty" if post.nut_boundary_empty(path, mesh["n_cells"], patch_sizes) else "written"


def history(case_dir, geom_dir, iters_list):
    """One row per window time from post.post_turb (the geom_dir argument is solve.launch's
    calling convention only - the pipe has no BRep): {"iter", "time", "status", "reason_id",
    dp = the pipe's U_b_m_s, Cd = its f, post_sha256}, None dp/Cd unless ok; never raises."""
    rows = []
    for it in iters_list:
        doc = post.post_turb(case_dir, str(it))
        pipe = doc.get("pipe") if doc.get("status") == "ok" else None
        rows.append({"iter": it, "time": str(it), "status": doc["status"],
                     "reason_id": doc.get("reason_id"),
                     "dp": None if pipe is None else pipe["U_b_m_s"],
                     "Cd": None if pipe is None else pipe["f"],
                     "post_sha256": common.sha256_of(doc)})
    return rows


def run(out_dir, name, iters=None, exe=None, visible=True, snapshot_fn=None):
    """One solve of cases/<name> into runs/<name> through solve.launch (the pinned D-1 binary
    of poiseuille's bin_gpu when exe is None; TG0-NAME for a bad name, TG0-OUT when the run
    directory exists); timed, the solve doc returned and wall.json written. An nvidia-smi
    snapshot (snapshot_fn, poiseuille.gpu_snapshot when None) is taken just before and just
    after launch and written to runs/<name>/gpu.json with poiseuille's shared flag; nothing
    lands in the run directory before launch returns."""
    if name not in NAMES:
        raise Refused("TG0-NAME", "name %r is not one of %r" % (name, list(NAMES)))
    level, _tag = level_of(name)
    rdir = os.path.join(out_dir, "runs", name)
    if os.path.exists(rdir):
        raise Refused("TG0-OUT", "%s exists" % rdir)
    if iters is None:
        iters = ITERS[level]
    if snapshot_fn is None:
        snapshot_fn = poiseuille.gpu_snapshot
    print("[tg0] solving %s for %d iterations..." % (name, iters))
    before = snapshot_fn()
    t0 = time.monotonic()
    doc = solve.launch(os.path.join(out_dir, "cases", name), os.path.join(out_dir, "mesh"), rdir, iters,
                       exe=exe, bin_json=None if exe else poiseuille.BIN_GPU, history_fn=history,
                       visible=visible)
    wall_s = time.monotonic() - t0
    after = snapshot_fn()
    os.makedirs(rdir, exist_ok=True)
    with open(os.path.join(rdir, "wall.json"), "wb") as f:
        f.write((common.canonical_json({"wall_s": wall_s}) + chr(10)).encode("utf-8"))
    gpu_doc = {"before": before, "after": after, "shared": poiseuille.shared_flag(before, after),
               "note": poiseuille.GPU_NOTE}
    with open(os.path.join(rdir, "gpu.json"), "wb") as f:
        f.write((common.canonical_json(gpu_doc) + chr(10)).encode("utf-8"))
    print("[tg0] %s class %s in %.1f s" % (name, doc.get("class"), wall_s))
    print("[tg0] %s gpu shared %s" % (name, gpu_doc["shared"]))
    return doc


def judge_run(pipe, re_tau, solve_class):
    """The per-run checks: the class and TB0-TB4 as bools, the three relative deviations as
    floats or None, and the reasons in VERDICT_IDS order (TG0-MISSING when pipe is None, then
    no TB id is added); re_tau is the INPUT of TB1, Re_D the measured one of TB2."""
    steady = solve_class == "steady"
    if pipe is None:
        return {"steady": steady, "TB0": False, "TB1": False, "TB2": False, "TB3": False, "TB4": False,
                "f_rel_prandtl": None, "f_rel_blasius": None, "core_rel": None,
                "reasons": ["TG0-MISSING"] + ([] if steady else ["TG0-UNSTEADY"])}
    f, re_d = pipe["f"], pipe["Re_D"]
    rel_p = None if f is None else f / turb_integral.f_prandtl(re_tau) - 1.0
    rel_b = None if f is None or re_d is None else f / turb_integral.f_blasius(re_d) - 1.0
    core = pipe["core_defect"]
    rel_c = None if core is None else core / BANDS["core_defect"] - 1.0
    bal, xin = pipe["balance_rel"], pipe["x_invariance"]
    loglaw = pipe["loglaw"] or {}
    n_cells, dev = loglaw.get("n_cells"), loglaw.get("dev_max")
    tb0 = bal is not None and xin is not None and abs(bal) <= BANDS["balance_rel"] \
        and xin <= BANDS["x_invariance"]
    tb1 = rel_p is not None and abs(rel_p) <= BANDS["f_prandtl_rel"]
    tb2 = rel_b is not None and abs(rel_b) <= BANDS["f_blasius_rel"]
    tb3 = n_cells is not None and n_cells >= 1 and dev is not None and dev <= BANDS["loglaw_dev"]
    tb4 = rel_c is not None and abs(rel_c) <= BANDS["core_rel"]
    reasons = []
    for flag, rid in ((not steady, "TG0-UNSTEADY"), (not tb0, "TG0-TB0"), (not tb1, "TG0-TB1"),
                      (not tb2, "TG0-TB2"), (not tb3, "TG0-TB3"), (not tb4, "TG0-TB4")):
        if flag:
            reasons.append(rid)
    return {"steady": steady, "TB0": tb0, "TB1": tb1, "TB2": tb2, "TB3": tb3, "TB4": tb4,
            "f_rel_prandtl": rel_p, "f_rel_blasius": rel_b, "core_rel": rel_c, "reasons": reasons}


def tb5(f_values):
    """One gated tag's TB5 row: nozzle_nominal.gci3 of the three levels' f (None where a level
    has no ok pipe), plus pass - monotone True with a gci_fine at most 1 %."""
    row = nozzle_nominal.gci3(list(f_values))
    row["pass"] = row["monotone"] is True and row["gci_fine"] is not None \
        and row["gci_fine"] <= BANDS["gci_fine_f"]
    return row


def verdict(g0_verdict, gate_reasons, tb5_rows):
    """The TG0 verdict over the gated tags: PASS iff the G0 verdict is PASS, both tags' L2
    reasons are empty (a None gate_reasons is a missing L2 run) and both TB5 rows pass;
    otherwise OPEN with the ids present, in VERDICT_IDS order."""
    bad = set()
    if g0_verdict != "PASS":
        bad.add("TG0-G0")
    for tag in TAGS_GATED:
        reasons = gate_reasons.get(tag)
        if reasons is None:
            bad.add("TG0-MISSING")
        else:
            bad.update(reasons)
        row = tb5_rows.get(tag)
        if row is None or row.get("pass") is not True:
            bad.add("TG0-TB5")
    reasons_out = [rid for rid in VERDICT_IDS if rid in bad]
    return {"verdict": "PASS" if not reasons_out else "OPEN", "reasons": reasons_out}


def _run_read(out_dir, name):
    """(solve row, wall_s, gpu, final time) of one run, all None without a solve.json."""
    sj = os.path.join(out_dir, "runs", name, "solve.json")
    if not os.path.isfile(sj):
        return None, None, None, None
    s = common.read_json(sj)
    res = s.get("result") or {}
    solve_row = {"class": s.get("class"), "reason_id": s.get("reason_id"), "failed": res.get("failed"),
                 "gating": s.get("gating"),
                 "criteria": res.get("criteria"), "iters": s.get("iters"),
                 "n_iter_lines": len(res.get("iterations") or []),
                 "log_sha256": (s.get("log") or {}).get("sha256"),
                 "binary_sha256": (s.get("binary") or {}).get("sha256")}
    wj = os.path.join(out_dir, "runs", name, "wall.json")
    wall_s = common.read_json(wj).get("wall_s") if os.path.isfile(wj) else None
    gj = os.path.join(out_dir, "runs", name, "gpu.json")
    gpu = common.read_json(gj) if os.path.isfile(gj) else None
    return solve_row, wall_s, gpu, s.get("iters")


def _post_row(doc):
    """The POST_ROW_KEYS row of one post_turb doc: the numbers from its pipe block, yplus1_max
    from wall_shear's wall row, all numbers None unless ok."""
    pipe = doc.get("pipe") if doc.get("status") == "ok" else None
    if pipe is None:
        row = dict((k, None) for k in POST_ROW_KEYS)
        row["status"], row["reason_id"] = doc.get("status"), doc.get("reason_id")
        return row
    ws = (doc.get("wall_shear") or {}).get("wall") or {}
    return {"status": doc["status"], "reason_id": doc["reason_id"],
            "U_b_m_s": pipe["U_b_m_s"], "Re_D": pipe["Re_D"], "u_tau_m_s": pipe["u_tau_m_s"],
            "Re_tau": pipe["Re_tau"], "f": pipe["f"], "U_axis_m_s": pipe["U_axis_m_s"],
            "core_defect": pipe["core_defect"], "x_invariance": pipe["x_invariance"],
            "balance_rel": pipe["balance_rel"], "loglaw": pipe["loglaw"],
            "yplus1_max": ws.get("yplus1_max")}


def record(out_dir):
    """The cad-tg0/1 record over the seven names: per-run solve, wall, gpu, nut_boundary and
    post rows with their checks, TB5 per gated tag, the verdict with its per-tag detail, and
    the report-only TB6/TB7/TB8. No absolute path, no out_dir and no time stamp in the doc."""
    build_doc = common.read_json(os.path.join(out_dir, "build.json"))
    lv_rows = dict((lv["level"], lv) for lv in build_doc["levels"])
    reads = dict((name, _run_read(out_dir, name)) for name in NAMES)
    binary = None
    for name in NAMES:
        sj = os.path.join(out_dir, "runs", name, "solve.json")
        if os.path.isfile(sj):
            b = common.read_json(sj).get("binary")
            if b:
                binary = dict(b)
                if os.path.isabs(binary.get("path", "")):
                    binary["path"] = os.path.basename(binary["path"])   # no absolute path in the record
                break
    runs, pipe_of, checks_of = [], {}, {}
    for name in NAMES:
        level, tag = level_of(name)
        solve_row, wall_s, gpu, final_t = reads[name]
        nut_b = post_row = checks = None
        if solve_row is not None:
            nut_b = nut_boundary(os.path.join(out_dir, "cases", name, str(final_t)))
            doc = post.post_turb(os.path.join(out_dir, "cases", name), str(final_t))
            post_row = _post_row(doc)
            pipe = doc.get("pipe") if doc.get("status") == "ok" else None
            if pipe is not None:
                pipe_of[name] = pipe
            checks = judge_run(pipe, RE_OF[tag], solve_row["class"])
            checks_of[name] = checks
        runs.append({"name": name, "re_tau": RE_OF[tag], "level": level,
                     "cells": lv_rows[level]["cells"], "solve": solve_row, "wall_s": wall_s,
                     "gpu": gpu, "nut_boundary": nut_b, "post": post_row, "checks": checks})
    tb5_rows, gate, per_tag = {}, {}, {}
    for tag in TAGS_GATED:
        vals = []
        for lv in LEVELS:
            p = pipe_of.get("L%d_%s" % (lv, tag))
            vals.append(None if p is None else p["f"])
        row = tb5(vals)
        tb5_rows[tag] = row
        gate_name = "L%d_%s" % (GATE_LEVEL, tag)
        ch = checks_of.get(gate_name)
        gate[tag] = None if ch is None else list(ch["reasons"])
        finest, finest_reasons = None, None
        for lv in (2, 1, 0):
            nm = "L%d_%s" % (lv, tag)
            if pipe_of.get(nm) is not None:
                finest = nm
                ch2 = checks_of.get(nm)
                finest_reasons = None if ch2 is None else list(ch2["reasons"])
                break
        per_tag[tag] = {"gate_run": gate_name, "gate_reasons": gate[tag], "tb5_pass": row["pass"],
                        "finest_run": finest, "finest_reasons": finest_reasons}
    g0v = None
    if os.path.isfile(G0_RECORD):
        g0v = (common.read_json(G0_RECORD).get("g0") or {}).get("verdict")
    ran = [name for name in NAMES if reads[name][0] is not None]
    reduced = {"status": "FULL" if len(ran) == len(NAMES) else "REDUCED", "ran": ran,
               "deferred": [name for name in NAMES if name not in set(ran)]}
    v = verdict(g0v, gate, tb5_rows)
    tg0 = {"verdict": v["verdict"], "reasons": v["reasons"], "per_tag": per_tag}
    tb6 = {"status": "NOT_RUN", "run": "L0_dns", "re_tau": RE_OF["dns"], "U_b_plus": None,
           "profile": None, "dns": dict(DNS_ROW)}
    p_dns = pipe_of.get("L0_dns")
    if p_dns is not None:
        tb6["status"] = "REPORTED"
        tb6["U_b_plus"] = p_dns["U_b_m_s"] / p_dns["u_tau_m_s"]
        tb6["profile"] = [[row["y_plus"], row["u_plus"]] for row in p_dns["profile"]]
    tb8 = {"status": "NOT_RUN", "fields_written": None, "y_axis_ref_m": Y_AXIS_REF_M}
    for name in NAMES:
        if reads[name][0] is None or reads[name][3] is None:
            continue
        tdir = os.path.join(out_dir, "cases", name, str(reads[name][3]))
        if not os.path.isdir(tdir):
            continue
        written = sorted(os.listdir(tdir))
        tb8["status"] = "WRITTEN" if any(nm in WALLDIST_NAMES for nm in written) else "NOT_WRITTEN"
        tb8["fields_written"] = written
        break
    return {"version": VERSION, "pipe_recipe_sha": build_doc["pipe_recipe_sha"], "bands": dict(BANDS),
            "iters": dict((str(k), v2) for k, v2 in ITERS.items()), "gate_level": GATE_LEVEL,
            "binary": binary, "g0_verdict": g0v, "reduced": reduced, "runs": runs, "tb5": tb5_rows,
            "tg0": tg0, "tb6": tb6, "tb7": dict(TB7_ROW), "tb8": tb8, "model_form": dict(MODEL_FORM)}


def main(argv):
    """--selftest | build OUT_DIR | run OUT_DIR NAME [ITERS] | record OUT_DIR RECORD_JSON |
    stage-bin (poiseuille.main's style; refusals to stderr, exit 2; bad argv the usage, exit 2)."""
    if argv == ["--selftest"]:
        try:
            selftest()
        except Exception:
            traceback.print_exc()
            return 1
        return 0
    if len(argv) == 2 and argv[0] == "build":
        try:
            doc = build(os.path.abspath(argv[1]))
        except wedge_mesh.Refused as r:
            sys.stderr.write("refused %s: %s%s" % (r.rule, r.detail, chr(10)))
            return 2
        for lv in doc["levels"]:
            print("L%d cells %d tau %.6f gc7_pass %s"
                  % (lv["level"], lv["cells"], float("nan") if lv["check_tau_min"] is None
                     else lv["check_tau_min"], lv["gc7_pass"]))
        for c in doc["cases"]:
            print("%s re_tau %s level %d u_tau %s yplus1 %s"
                  % (c["name"], c["re_tau"], c["level"], c["u_tau_m_s"], c["yplus1_apriori"]))
        return 0
    if len(argv) in (3, 4) and argv[0] == "run":
        try:
            iters = None
            if len(argv) == 4:
                iters = int(argv[3])
            doc = run(os.path.abspath(argv[1]), argv[2], iters=iters)
        except ValueError:
            sys.stderr.write(USAGE + chr(10))
            return 2
        except wedge_mesh.Refused as r:
            sys.stderr.write("refused %s: %s%s" % (r.rule, r.detail, chr(10)))
            return 2
        print("class %s reason_id %s" % (doc.get("class"), doc.get("reason_id")))
        return 0 if doc.get("class") == "steady" else 1
    if len(argv) == 3 and argv[0] == "record":
        try:
            rec = record(os.path.abspath(argv[1]))
        except wedge_mesh.Refused as r:
            sys.stderr.write("refused %s: %s%s" % (r.rule, r.detail, chr(10)))
            return 2
        with open(os.path.abspath(argv[2]), "wb") as f:
            f.write((common.canonical_json(rec) + chr(10)).encode("utf-8"))
        print("tg0 %s reasons %s" % (rec["tg0"]["verdict"], ",".join(rec["tg0"]["reasons"])))
        return 0 if rec["tg0"]["verdict"] == "PASS" else 1
    if argv == ["stage-bin"]:
        try:
            print(common.canonical_json(poiseuille.stage_pinned()))
        except wedge_mesh.Refused as r:
            sys.stderr.write("refused %s: %s%s" % (r.rule, r.detail, chr(10)))
            return 2
        return 0
    sys.stderr.write(USAGE + chr(10))
    return 2


def _plant_fields(case_dir, t):
    """The selftest's planted Poiseuille pipe at time t (the plant of post's pipe test): the
    u = 2 (1 - (y^2 + z^2)/r_p^2) profile along x with r_p = R cos(2.5 deg), p and nut zero,
    written through post._fx_plant_c and post._fx_write_turb, then the time's nut rewritten so
    that everything from the word boundaryField on is the solver's exact written tail (an empty
    boundaryField, two blank lines, the 79-byte // ***...*** // closing line)."""
    case = common.read_json(os.path.join(case_dir, "case.json"))
    types = post._fx_types_turb(case)
    mesh = post.load_mesh(os.path.join(case_dir, "constant", "polyMesh"))
    r_p = pipe_mesh.RECIPE["R_m"] * math.cos(math.radians(2.5))

    def fn_u(pt):
        u = 2.0 * (1.0 - (pt[:, 1] ** 2 + pt[:, 2] ** 2) / r_p ** 2)
        return np.stack([u, np.zeros_like(u), np.zeros_like(u)], axis=1)

    U, p, nut = post._fx_plant_c(mesh, fn_u, lambda pt: np.zeros(len(pt)),
                                 lambda pt: np.zeros(len(pt)), types)
    post._fx_write_turb(case_dir, str(t), U, p, nut)
    npath = os.path.join(case_dir, str(t), "nut")
    with open(npath, "r", encoding="utf-8") as f:
        text = f.read()
    idx = text.find("boundaryField")
    tail = ("boundaryField" + chr(10) + "{" + chr(10) + "}" + chr(10) + chr(10) + chr(10)
            + "// " + "*" * 73 + " //" + chr(10))
    text = text[:idx] + tail
    with open(npath, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def _fake_bin(td):
    """The test's fake solver binary in the temp dir (never a tree file): it plants the
    Poiseuille pipe on the five window times of its -iters and prints the steady fixture log."""
    path = os.path.join(td, "fake_tg0.py")
    lines = ["import os, sys",
             "sys.path.insert(0, %r)" % os.path.dirname(os.path.abspath(__file__)),
             "import pipe_turb",
             "case = sys.argv[1]",
             "iters = int(sys.argv[sys.argv.index('-iters') + 1])",
             "for t in range(iters - 200, iters + 1, 50):",
             "    pipe_turb._plant_fields(case, str(t))",
             "with open(os.path.join(%r, 'fixtures', 'solve', 'steady', 'solve.log'), 'rb') as f:" % CAD,
             "    sys.stdout.buffer.write(f.read())"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(chr(10).join(lines) + chr(10))
    return path


def _t1():
    assert RE_OF == {"lo": 576.69, "hi": 2358.0, "dns": 550.0} and GATE_LEVEL == 2
    assert ITERS == {0: 4000, 1: 8000, 2: 20000} and TAGS_GATED == ("lo", "hi")
    assert NAMES == ("L0_lo", "L1_lo", "L2_lo", "L0_hi", "L1_hi", "L2_hi", "L0_dns")
    assert abs(turb_integral.f_prandtl(576.69) / 0.025789172770595198 - 1.0) <= 1e-12
    assert abs(turb_integral.f_prandtl(2358.0) / 0.01801605735288456 - 1.0) <= 1e-12
    assert Y_AXIS_REF_M == 0.025 / math.sqrt(2.0)
    assert abs(Y_AXIS_REF_M / 0.017677669529663688 - 1.0) <= 1e-15
    assert TB7_ROW["status"] == "NOT_RUN"
    assert WALLDIST_NAMES == ("y", "wallDist", "yWall", "nearWallDist")
    print("[ok] T1 constants: the seven names, Re_of 576.69/2358.0/550.0, budgets 4000/8000/20000,"
          " gate level 2, f_prandtl 0.0257891728 and 0.0180160574, y_axis_ref_m R/sqrt(2), TB7 NOT_RUN")


def _t2(out, build_doc):
    assert list(build_doc) == list(BUILD_KEYS)
    assert [lv["level"] for lv in build_doc["levels"]] == [0, 1, 2]
    for lv in build_doc["levels"]:
        assert list(lv) == list(BUILD_LEVEL_KEYS)
    assert [lv["cells"] for lv in build_doc["levels"]] == [160, 320, 640]
    assert all(lv["gc7_pass"] is True for lv in build_doc["levels"])
    cases = build_doc["cases"]
    assert len(cases) == 1 and list(cases[0]) == list(BUILD_CASE_KEYS)
    c = cases[0]
    assert c["name"] == "L0_lo" and c["re_tau"] == 576.69 and c["level"] == 0
    assert abs(c["u_tau_m_s"] / 0.34970481600000003 - 1.0) <= 1e-12, c["u_tau_m_s"]
    case = common.read_json(os.path.join(out, "cases", "L0_lo", "case.json"))
    assert case["version"] == "cad-case-turb/1" and case["kind"] == "pipe"
    try:
        build(out, names=("L0_lo",))
        raise AssertionError("build(out) was not refused")
    except Refused as r:
        assert r.rule == "TG0-OUT"
    print("[ok] T2 build: 160/320/640 cells all gc7 pass, one L0_lo case at Re_tau 576.69 with"
          " u_tau %.14f, cad-case-turb/1 kind pipe, a second build refused TG0-OUT" % c["u_tau_m_s"])


def _t3(td, out):
    base = os.path.join(out, "cases", "L0_lo")
    copy_a = os.path.join(td, "case_a")
    shutil.copytree(base, copy_a)
    _plant_fields(copy_a, "9")
    doc = post.post_turb(copy_a, "9")
    assert doc["status"] == "ok", (doc["status"], doc["reason_id"], doc["message"])
    pb = doc["pipe"]
    assert abs(pb["f"] * pb["Re_D"] / 63.97539743517225 - 1.0) <= 1e-9, pb["f"] * pb["Re_D"]
    assert abs(pb["U_b_m_s"] / 1.0005960972173311 - 1.0) <= 1e-12, pb["U_b_m_s"]
    assert abs(pb["Re_D"] / 3298.024866681143 - 1.0) <= 1e-9, pb["Re_D"]
    assert abs(pb["f"] / 0.019398094320480896 - 1.0) <= 1e-9, pb["f"]
    assert nut_boundary(os.path.join(copy_a, "9")) == "empty"
    line79 = "// " + "*" * 73 + " //"
    assert len(line79) == 79
    solver_tail = ("boundaryField" + chr(10) + "{" + chr(10) + "}" + chr(10) + chr(10) + chr(10)
                   + line79 + chr(10)).encode("utf-8")
    with open(os.path.join(copy_a, "9", "nut"), "rb") as f:
        blob_a = f.read()
    assert blob_a.endswith(solver_tail)
    copy_b = os.path.join(td, "case_b")
    shutil.copytree(base, copy_b)
    _plant_fields(copy_b, "9")
    with open(os.path.join(copy_b, "0", "nut"), "a", encoding="utf-8", newline="") as f:
        f.write("// x" + chr(10))
    doc_b = post.post_turb(copy_b, "9")
    assert doc_b["status"] == "refused" and doc_b["reason_id"] == "POST-BIND", \
        (doc_b["reason_id"], doc_b["message"])
    assert doc_b["message"] == "0/nut: sha differs from case.json", doc_b["message"]
    copy_c = os.path.join(td, "case_c")
    shutil.copytree(base, copy_c)
    _plant_fields(copy_c, "9")
    npath = os.path.join(copy_c, "0", "nut")
    with open(npath, "r", encoding="utf-8") as f:
        text = f.read()
    assert text.count("fixedValue") == 1
    with open(npath, "w", encoding="utf-8", newline="") as f:
        f.write(text.replace("fixedValue", "nutkWallFunction", 1))
    case_doc = common.read_json(os.path.join(copy_c, "case.json"))
    case_doc["files"]["0/nut"] = common.sha256_file(npath)
    common.atomic_write(os.path.join(copy_c, "case.json"), common.canonical_json(case_doc) + chr(10))
    doc_c = post.post_turb(copy_c, "9")
    assert doc_c["status"] == "refused" and doc_c["reason_id"] == "POST-FIELD", \
        (doc_c["reason_id"], doc_c["message"])
    assert "wall patch wall has 0/nut type nutkWallFunction" in doc_c["message"], doc_c["message"]
    copy_d = os.path.join(td, "case_d")
    shutil.copytree(base, copy_d)
    _plant_fields(copy_d, "9")
    npath_d = os.path.join(copy_d, "9", "nut")
    with open(npath_d, "rb") as f:
        blob_d = f.read()
    assert blob_d.endswith(solver_tail)
    short_tail = ("boundaryField" + chr(10) + "{" + chr(10) + "}" + chr(10)).encode("utf-8")
    with open(npath_d, "wb") as f:
        f.write(blob_d[:len(blob_d) - len(solver_tail)] + short_tail)
    doc_d = post.post_turb(copy_d, "9")
    assert doc_d["status"] == "ok", (doc_d["status"], doc_d["reason_id"], doc_d["message"])
    pd = doc_d["pipe"]
    assert abs(pd["f"] * pd["Re_D"] / 63.97539743517225 - 1.0) <= 1e-9, pd["f"] * pd["Re_D"]
    assert nut_boundary(os.path.join(copy_d, "9")) == "empty"
    print("[ok] T3 the empty-boundaryField nut read: the planted pipe ok at f Re_D %.14f,"
          " U_b %.16f, an edited 0/nut POST-BIND, a nutkWallFunction wall POST-FIELD naming"
          " patch wall, the comment-tailed and bare empty forms both ok at f Re_D %.14f"
          % (pb["f"] * pb["Re_D"], pb["U_b_m_s"], pd["f"] * pd["Re_D"]))


def _t4(td, out):
    fake = _fake_bin(td)

    def snap():
        return {"status": "ok", "gpu": {"name": "fake", "memory_used_mib": 0, "memory_total_mib": 0,
                                        "utilization_pct": 0}, "compute_apps": [], "detail": ""}

    doc = run(out, "L0_lo", iters=600, exe=[sys.executable, fake], visible=False, snapshot_fn=snap)
    assert doc["class"] == "steady", (doc["class"], doc["reason_id"], doc["message"])
    assert doc["gating"] == list(solve.CRITERIA_PERIODIC), doc["gating"]
    with open(os.path.join(out, "cases", "L0_lo", "system", "fvSolution"), "r", encoding="utf-8") as f:
        fsol_t4 = f.read()
    assert fsol_t4.count("tolerance       1e-12;") == 2, fsol_t4.count("tolerance       1e-12;")
    rdir = os.path.join(out, "runs", "L0_lo")
    assert sorted(os.listdir(rdir)) == ["gpu.json", "solve.json", "solve.log", "wall.json"]
    rows = history(os.path.join(out, "cases", "L0_lo"), os.path.join(out, "mesh"), [600])
    assert len(rows) == 1 and rows[0]["iter"] == 600 and rows[0]["time"] == "600"
    assert rows[0]["status"] == "ok" and rows[0]["reason_id"] is None
    assert abs(rows[0]["dp"] / 1.0005960972173311 - 1.0) <= 1e-12, rows[0]["dp"]
    assert abs(rows[0]["Cd"] / 0.019398094320480896 - 1.0) <= 1e-9, rows[0]["Cd"]
    assert rows[0]["post_sha256"]
    try:
        run(out, "L0_lo", iters=600, exe=[sys.executable, fake], visible=False, snapshot_fn=snap)
        raise AssertionError("run was not refused")
    except Refused as r:
        assert r.rule == "TG0-OUT"
    try:
        run(out, "L3_lo", iters=600, exe=[sys.executable, fake], visible=False, snapshot_fn=snap)
        raise AssertionError("run was not refused")
    except Refused as r:
        assert r.rule == "TG0-NAME"
    print("[ok] T4 launch: the fake binary's run is steady in the four run files, the history"
          " window row has U_b %.16f and f %.18f, a second run TG0-OUT and L3_lo TG0-NAME"
          % (rows[0]["dp"], rows[0]["Cd"]))


def _t5():
    pipe = {"f": 0.0266, "Re_D": 19976.4, "balance_rel": -8.2e-7, "x_invariance": 8.25e-7,
            "core_defect": 3.5975, "loglaw": {"n_cells": 36, "dev_max": 1.0125}}
    ch = judge_run(pipe, 576.69, "unsteady")
    assert list(ch) == list(CHECK_KEYS)
    assert abs(ch["f_rel_prandtl"] / 0.03144060635901069 - 1.0) <= 1e-12, ch["f_rel_prandtl"]
    assert abs(ch["f_rel_blasius"] / -0.0005191713427487032 - 1.0) <= 1e-12, ch["f_rel_blasius"]
    assert abs(ch["core_rel"] / -0.11609336609336607 - 1.0) <= 1e-12, ch["core_rel"]
    assert ch["steady"] is False and ch["TB0"] is True and ch["TB1"] is True and ch["TB2"] is True
    assert ch["TB3"] is False and ch["TB4"] is False
    assert ch["reasons"] == ["TG0-UNSTEADY", "TG0-TB3", "TG0-TB4"]
    pipe2 = {"f": 0.0262, "Re_D": 20250.0, "balance_rel": 0.004, "x_invariance": 9e-6,
             "core_defect": 3.9, "loglaw": {"n_cells": 40, "dev_max": 0.9}}
    ch2 = judge_run(pipe2, 576.69, "steady")
    assert abs(ch2["f_rel_prandtl"] / 0.01593022130098065 - 1.0) <= 1e-12, ch2["f_rel_prandtl"]
    assert abs(ch2["f_rel_blasius"] / -0.012195333737513026 - 1.0) <= 1e-12, ch2["f_rel_blasius"]
    assert abs(ch2["core_rel"] / -0.04176904176904184 - 1.0) <= 1e-12, ch2["core_rel"]
    assert ch2["steady"] is True and ch2["reasons"] == []
    assert judge_run(None, 576.69, None)["reasons"] == ["TG0-MISSING", "TG0-UNSTEADY"]
    j4a = judge_run(dict(pipe2, x_invariance=2e-5), 576.69, "steady")
    assert j4a["TB0"] is False and j4a["reasons"] == ["TG0-TB0"]
    j4b = judge_run(dict(pipe2, loglaw={"n_cells": 0, "dev_max": None}), 576.69, "steady")
    assert j4b["TB3"] is False and j4b["reasons"] == ["TG0-TB3"]
    j4c = judge_run({"f": 0.0185, "Re_D": 101000.0, "balance_rel": 0.004, "x_invariance": 9e-6,
                     "core_defect": 3.9, "loglaw": {"n_cells": 40, "dev_max": 0.9}}, 2358.0, "steady")
    assert abs(j4c["f_rel_prandtl"] / 0.0268617399265747 - 1.0) <= 1e-12, j4c["f_rel_prandtl"]
    assert abs(j4c["f_rel_blasius"] / 0.04235486330407423 - 1.0) <= 1e-12, j4c["f_rel_blasius"]
    assert j4c["reasons"] == []
    print("[ok] T5 judge_run: the two oracle judges and the None pipe, the x_invariance,"
          " loglaw and Re_tau 2358 variants, reasons in VERDICT_IDS order")


def _t6():
    row = tb5([0.027, 0.0265, 0.0263])
    assert row["monotone"] is True and abs(row["p"] / 1.3219280948873724 - 1.0) <= 1e-12
    assert abs(row["gci_fine"] / 0.0063371356147020426 - 1.0) <= 1e-12, row["gci_fine"]
    assert row["pass"] is True
    row2 = tb5([0.0263, 0.0265, 0.027])
    assert row2["gci_fine"] is None and row2["pass"] is False
    row3 = tb5([0.027, None, 0.0265])
    assert row3["values"] is None and row3["pass"] is False
    good = {"pass": True}
    v1 = verdict("PASS", {"lo": [], "hi": []}, {"lo": dict(good), "hi": dict(good)})
    assert v1 == {"verdict": "PASS", "reasons": []}
    v2 = verdict("PASS", {"lo": None, "hi": []}, {"lo": {"pass": False}, "hi": dict(good)})
    assert v2 == {"verdict": "OPEN", "reasons": ["TG0-MISSING", "TG0-TB5"]}
    v3 = verdict("OPEN", {"lo": ["TG0-TB4"], "hi": ["TG0-UNSTEADY", "TG0-TB4"]},
                 {"lo": dict(good), "hi": dict(good)})
    assert v3 == {"verdict": "OPEN", "reasons": ["TG0-G0", "TG0-UNSTEADY", "TG0-TB4"]}
    print("[ok] T6 tb5 and verdict: the monotone GCI pass, the increasing and None-value"
          " fails, the three verdict oracles with reasons in VERDICT_IDS order")


def _t7(td, out):
    rec = record(out)
    assert list(rec) == list(RECORD_KEYS)
    assert len(rec["runs"]) == 7 and [r["name"] for r in rec["runs"]] == list(NAMES)
    for r in rec["runs"]:
        assert list(r) == list(RUN_KEYS)
    r0 = rec["runs"][0]
    assert r0["name"] == "L0_lo" and r0["re_tau"] == 576.69 and r0["level"] == 0 and r0["cells"] == 160
    assert r0["solve"]["class"] == "steady" and r0["solve"]["iters"] == 600
    assert r0["solve"]["gating"] == ["cont_err", "dp_rel_change", "Cd_rel_change"], r0["solve"]["gating"]
    assert rec["model_form"] == MODEL_FORM
    assert r0["nut_boundary"] == "empty"
    assert r0["post"]["status"] == "ok"
    assert abs(r0["post"]["f"] / 0.019398094320480896 - 1.0) <= 1e-9, r0["post"]["f"]
    assert abs(r0["post"]["U_b_m_s"] / 1.0005960972173311 - 1.0) <= 1e-12
    for r in rec["runs"][1:]:
        assert r["solve"] is None and r["wall_s"] is None and r["gpu"] is None
        assert r["nut_boundary"] is None and r["post"] is None and r["checks"] is None
    assert rec["reduced"] == {"status": "REDUCED", "ran": ["L0_lo"],
                              "deferred": ["L1_lo", "L2_lo", "L0_hi", "L1_hi", "L2_hi", "L0_dns"]}
    assert rec["g0_verdict"] == "PASS"
    assert rec["tg0"]["verdict"] == "OPEN" and rec["tg0"]["reasons"] == ["TG0-MISSING", "TG0-TB5"]
    assert rec["tb6"]["status"] == "NOT_RUN" and rec["tb7"] == TB7_ROW
    assert rec["tb8"]["status"] == "NOT_WRITTEN"
    assert rec["tb8"]["fields_written"] == ["U", "nut", "p"]
    assert abs(rec["tb8"]["y_axis_ref_m"] / 0.017677669529663688 - 1.0) <= 1e-15
    text = common.canonical_json(rec)
    assert "C:/" not in text and ("C:" + chr(92)) not in text and td not in text
    rec2 = record(out)
    assert common.canonical_json(rec2) == text
    dns_dir = os.path.join(out, "runs", "L0_dns")
    os.makedirs(dns_dir)
    sdoc = common.read_json(os.path.join(out, "runs", "L0_lo", "solve.json"))
    sdoc["iters"] = 999
    common.atomic_write(os.path.join(dns_dir, "solve.json"), common.canonical_json(sdoc) + chr(10))
    rec3 = record(out)
    r_dns = [r for r in rec3["runs"] if r["name"] == "L0_dns"][0]
    assert r_dns["nut_boundary"] is None and r_dns["post"]["status"] == "refused"
    assert r_dns["checks"]["reasons"] == ["TG0-MISSING"]
    assert rec3["reduced"]["ran"] == ["L0_lo", "L0_dns"]
    assert rec3["tb6"]["status"] == "NOT_RUN"
    assert rec3["tb8"]["status"] == "NOT_WRITTEN" and rec3["tb8"]["fields_written"] == ["U", "nut", "p"]
    shutil.rmtree(dns_dir)
    print("[ok] T7 record: the seven RUN_KEYS rows in NAMES order with only L0_lo solved,"
          " REDUCED, tg0 OPEN on TG0-MISSING and TG0-TB5, tb6 NOT_RUN, tb7 the constant row,"
          " tb8 NOT_WRITTEN over [U, nut, p], no absolute path, a second record identical, and"
          " a solve.json without its time directory a refused TG0-MISSING row in ran with tb8"
          " unchanged")


def _t8(td, out):
    this = os.path.abspath(__file__)
    rec_path = os.path.join(td, "rec.json")
    p = subprocess.run([sys.executable, this, "record", out, rec_path], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", cwd=HERE)
    assert p.returncode == 1, (p.returncode, p.stdout[-300:], p.stderr[-300:])
    doc = common.read_json(rec_path)
    assert doc["tg0"]["verdict"] == "OPEN" and doc["reduced"]["ran"] == ["L0_lo"]
    p2 = subprocess.run([sys.executable, this, "run", out, "L9_xx"], capture_output=True, text=True,
                        encoding="utf-8", errors="replace", cwd=HERE)
    assert p2.returncode == 2 and "refused TG0-NAME" in p2.stderr, (p2.returncode, p2.stderr[-300:])
    p3 = subprocess.run([sys.executable, this], capture_output=True, text=True,
                        encoding="utf-8", errors="replace", cwd=HERE)
    assert p3.returncode == 2, (p3.returncode, p3.stderr[-200:])
    assert p3.stderr.startswith("usage: python pipe_turb.py --selftest"), p3.stderr[:80]
    print("[ok] T8 CLI: record exits 1 writing an OPEN cad-tg0/1, run L9_xx refuses TG0-NAME"
          " on stderr with exit 2, no argv the usage and exit 2")


def selftest():
    """T1-T8 in one TemporaryDirectory, build(out) once for T2-T8; SELFTEST PASS at the end."""
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "out")
        _t1()
        build_doc = build(out, names=("L0_lo",))
        _t2(out, build_doc)
        _t3(td, out)
        _t4(td, out)
        _t5()
        _t6()
        _t7(td, out)
        _t8(td, out)
    print("SELFTEST PASS")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
