#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""nozzle_nominal.py - CAD-20 (docs/16 §I CAD-20, §H.2 the case, §H.4 G1, G2, G3 and G-REPEAT):
the nominal laminar nozzle's REDUCED run set, its builds, solves and the cad-g123/1 record.

The nominal is the GC-3 template nominal (D_i 0.06, CR 9, poly5, L/D_i 1, Lx/De 0.5, t 3 mm)
at Re_De 3e4 (U_e 22.5 m/s) and 1e4 (U_e 7.5 m/s), wedge-meshed at two levels (three with
--full), written as cold cad-case/1 cases of case_writer.py, and solved by the supervisor with
the D-1 binary of bin_gpu.json. G1 is conservation (mass, axis total-pressure loss, axial
momentum closure), G2 feeds the CFD edge velocity sqrt(2 (p0_core - p_wall)) to thwaites.py and
judges theta_e against it plus the -1/2 Reynolds slope and the reversal flag, G3 is the Roache
1994 grid-convergence index over L0, L1, L2 (NOT_RUN without L2), G-REPEAT is the L0/L0r noise
band with bit identity reported, never assumed, and the exit-tube sensitivity Lx/De 0.5 vs 1.0
on theta_e is reported, never gated. A result outside its band leaves the gate OPEN; it is never
tuned and never fixed by changing numerics (docs/16 §H.4).

Definitions (fixed before any run):
  N1 names. L<level><suffix>_<re>: level 0/1/2, suffix "" (nominal), "r" (repeat: a second cold
     case of the same nominal mesh and requirement set) or "x" (the Lx/De 1.0 geometry); re
     "re3e4" or "re1e4". A name not in RUN_SET_REDUCED + FULL_EXTRA is NOZ-NAME.
     parse_name(name) -> {"level", "kind": "nominal"|"repeat"|"exit", "re"}.
  N2 build(out_dir, full=False). NOZ-OUT unless out_dir is missing or empty. geom/ = the export
     nominal; geom_x/ the same with Lx_over_De = 1.0 (NOZ-GEOM on a refusal). wedge/ = the wedge
     mesh at levels (0, 1) reduced or (0, 1, 2) full; wedge_x/ at (0,) reduced or (0, 1) full
     (NOZ-MESH on a refusal or a false gc6_pass). req_re3e4/, req_re1e4/: the golden
     v3_exit_velocity requirements with only operating_point.U_exit_m_s set to RE_POINTS[re] and
     the lock recomputed, written by reqs.write_locked. cases/<name>/ by case_writer.write_case
     (NOZ-CASE with its rule on a refusal). build.json: version, full, names, the two wedge
     level tables and per case level, kind, re, cells and the two U values; no absolute path.
  N3 run(out_dir, name, ...). NOZ-NAME for a bad name, NOZ-OUT when runs/<name> exists; iters =
     ITERS[level], geom_dir geom_x for kind "exit" else geom; an nvidia-smi snapshot
     (poiseuille.gpu_snapshot) before and after solve.launch with bin_gpu.json, wall.json
     {"wall_s"} and gpu.json {before, after, shared, note} exactly like poiseuille.run; then,
     when cases/<name>/<iters> exists, post.post into runs/<name>/post.json.
  N4 per-run values. m[k] = post["metrics"][k]["value"] for Cd, theta_exit, mass_imbalance,
     p0_loss_axis, momentum_closure, separation_free, dp, exit_nonuniformity, H_exit, plus
     x_exit = stations.exit_plane.x_m, nu = operating_point.nu_m2_s and edge = post["edge"];
     m is None when post.json is missing or its status is not ok.
  N5 G1 judge_g1(m, solve_class): mass <= 1e-5, p0 loss <= 0.005 (signed, a negative loss
     passes), |momentum closure| <= 0.02; reasons in G1_IDS order. The record's G1 is judged on
     BOTH Re at the gate level, overall PASS iff both PASS, reasons the union in G1_IDS order.
  N6 theta_thwaites(edge, nu, x_exit): thwaites.solve on the edge rows in their given order;
     a None edge velocity or a ValueError is (None, "G2-EDGE"); theta_thw = interp(x_exit).
  N7 G2 judge_g2(runs_by_re, classes_by_re): per Re theta_rel = theta_exit/theta_thw - 1 with
     |theta_rel| <= 0.10, reversal pass iff separation_free == 1.0; slope = ln(theta_exit(3e4)/
     theta_exit(1e4))/ln 3 with |slope - (-0.50)| <= 0.03; reasons in G2_IDS order;
     theta_thw_nominal = theta at the exit station of thwaites.nominal_case(Re), report only.
  N8 G3 gci3(values): e01, e12, monotone iff e01*e12 > 0, p = ln(e01/e12)/ln 2, f_ext and
     gci_fine = 1.25 |f2 - f1|/(|f2| (2^p - 1)) only when p > 0, gci_fine_abs = gci_fine |f2|;
     gci2(f0, f1) = 3 |f1 - f0|/(|f1| 3) report only. judge_g3: NOT_RUN with G3-LEVELS unless
     L0, L1 and L2 nominal at BOTH Re have solve.json; otherwise per Re gci3 of Cd and
     theta_exit, reasons in G3_IDS order (GCI bands 0.002 and 0.03).
  N9 G-REPEAT and sensitivity. Repeat pair (L1_re3e4, L1r_re3e4) when L1r has a solve.json else
     (L0_re3e4, L0r_re3e4): band |a - b| of Cd, theta_exit and dp, field bit identity and
     solve.log iter-line identity exactly as poiseuille's _record_tail; status "recorded" when
     both have ok values. Sensitivity pair (L1_re3e4, L1x_re3e4) when L1x has a solve.json else
     (L0_re3e4, L0x_re3e4): theta_rel, cd_rel, reduced, status "reported" (never gated).
  N10 record(out_dir). gate_level = 2 when both L2 nominal names have solve.json, else 1 when
     both L1 do, else 0; reduced = gate_level < 2; binary is the first solve.json's binary with
     the basename of its path; one runs row per name of the build's set in set order; cfd_u is
     the E.4 u of Re 3e4 (the G3 gci_fine_abs and the G-REPEAT band), status OPEN, with the note
     that the loop must not lock on it while gci_fine is None. No absolute path anywhere.

Usage:
  python nozzle_nominal.py --selftest
  python nozzle_nominal.py build OUT_DIR [--full]
  python nozzle_nominal.py run OUT_DIR NAME            (the GPU solve, supervisor only)
  python nozzle_nominal.py record OUT_DIR RECORD_JSON
"""
import copy
import math
import os
import re
import sys
import tempfile
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAD = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, CAD)
sys.path.insert(0, os.path.join(CAD, "cases", "poiseuille"))
import common
import case_writer
import wedge_mesh
import post
import solve
import export
import reqs
import thwaites
import poiseuille

VERSION = "cad-g123/1"
RE_POINTS = {"re3e4": 22.5, "re1e4": 7.5}            # U_exit_m_s per Re tag (docs/16 H.2)
RE_VALUES = {"re3e4": 3.0e4, "re1e4": 1.0e4}
LX_VARIANT = 1.0                                     # Lx_over_De of the exit-tube variant; nominal is 0.5
ITERS = {0: 1000, 1: 1200, 2: 6000}                  # -iters per mesh level, fixed before any run
RUN_SET_REDUCED = ("L0_re3e4", "L0_re1e4", "L1_re3e4", "L1_re1e4", "L0r_re3e4", "L0x_re3e4")
FULL_EXTRA = ("L2_re3e4", "L2_re1e4", "L1r_re3e4", "L1x_re3e4")
ALL_NAMES = tuple(RUN_SET_REDUCED) + tuple(FULL_EXTRA)
BANDS = {"mass_imbalance": 1e-5, "p0_loss_axis": 0.005, "momentum_closure": 0.02, "theta_rel": 0.10,
         "slope": -0.50, "slope_tol": 0.03, "gci_cd": 0.002, "gci_theta": 0.03}
GCI_FS = 1.25                                        # three-grid safety factor (Roache 1994)
GCI_FS_2LEVEL = 3.0                                  # two-grid safety factor, formal order 2, report only
P_FORMAL = 2.0
BIN_GPU = os.path.join(CAD, "bin_gpu.json")
GOLDEN_REQ = os.path.join(CAD, "fixtures", "reqs", "golden", "v3_exit_velocity.json")
REFUSAL_IDS = ("NOZ-OUT", "NOZ-NAME", "NOZ-GEOM", "NOZ-MESH", "NOZ-CASE")
G1_IDS = ("G1-MISSING", "G1-UNSTEADY", "G1-MASS", "G1-P0LOSS", "G1-MOMENTUM")
G2_IDS = ("G2-MISSING", "G2-UNSTEADY", "G2-EDGE", "G2-THETA", "G2-SLOPE", "G2-REVERSAL")
G3_IDS = ("G3-LEVELS", "G3-MISSING", "G3-UNSTEADY", "G3-MONO", "G3-GCI-CD", "G3-GCI-THETA")
VALUE_KEYS = ("Cd", "theta_exit", "mass_imbalance", "p0_loss_axis", "momentum_closure", "separation_free",
              "dp", "exit_nonuniformity", "H_exit")
# name, metric key, limit, |value|: mass and the SIGNED p0 loss judge v <= limit (a negative
# loss passes, docs/16 H.4 N5); only the momentum closure takes |v| <= limit.
G1_LIMITS = (("mass", "mass_imbalance", BANDS["mass_imbalance"], False),
             ("p0loss", "p0_loss_axis", BANDS["p0_loss_axis"], False),
             ("momentum", "momentum_closure", BANDS["momentum_closure"], True))
G1_REASON = {"mass": "G1-MASS", "p0loss": "G1-P0LOSS", "momentum": "G1-MOMENTUM"}
REPEAT_KEYS = ("Cd", "theta_exit", "dp")
GATE_LEVELS = (2, 1)
RECORD_KEYS = ("version", "bands", "iters", "gate_level", "reduced", "binary", "runs", "g1", "g2", "g3",
               "g_repeat", "sensitivity", "cfd_u")
RUN_ROW_KEYS = ("name", "level", "kind", "re", "cells", "solve", "wall_s", "gpu", "values", "theta_thw")
SOLVE_ROW_KEYS = ("class", "reason_id", "failed", "criteria", "n_iter_lines", "log_sha256", "binary_sha256")
CFD_U_NOTE = ("The loop must not lock on this u while gci_fine is None; u = max(GCI_fine at the nominal, "
              "the G-REPEAT band) only once the L2 confirmation holds (docs/16 E.4).")
USAGE = ("usage: python nozzle_nominal.py --selftest" + chr(10)
         + "       python nozzle_nominal.py build OUT_DIR [--full]" + chr(10)
         + "       python nozzle_nominal.py run OUT_DIR NAME" + chr(10)
         + "       python nozzle_nominal.py record OUT_DIR RECORD_JSON")
_NAME_RE = re.compile(r"^L([0-2])(r|x)?_re(3e4|1e4)$")


class Refused(wedge_mesh.Refused):
    """A refusal by id (case_writer.Refused's shape): rule and detail."""


def parse_name(name):
    """N1: {"level", "kind": "nominal"|"repeat"|"exit", "re"}; NOZ-NAME when the name is off grammar."""
    m = _NAME_RE.match(name) if isinstance(name, str) else None
    if m is None:
        raise Refused("NOZ-NAME", "name %r is not L<0|1|2><r|x|>_re<3e4|1e4>" % (name,))
    kind = "nominal" if m.group(2) is None else ("repeat" if m.group(2) == "r" else "exit")
    return {"level": int(m.group(1)), "kind": kind, "re": "re" + m.group(3)}


def judge_g1(m, solve_class):
    """N5 into {"verdict", "reasons", "checks"}: mass and the signed p0 loss judge v <= limit
    (a negative loss passes); the momentum closure judges |v| <= limit."""
    checks = {}
    for name, key, limit, absv in G1_LIMITS:
        v = m.get(key) if isinstance(m, dict) else None
        w = abs(v) if absv and v is not None else v
        checks[name] = {"value": v, "limit": limit, "pass": bool(v is not None and w <= limit)}
    reasons = []
    if m is None or any(checks[name]["value"] is None for name, _k, _l, _a in G1_LIMITS):
        reasons.append("G1-MISSING")
    if solve_class != "steady":
        reasons.append("G1-UNSTEADY")
    for name, _k, _l, _a in G1_LIMITS:
        if checks[name]["value"] is not None and not checks[name]["pass"]:
            reasons.append(G1_REASON[name])
    return {"verdict": "PASS" if not reasons else "OPEN", "reasons": reasons, "checks": checks}


def theta_thwaites(edge, nu, x_exit):
    """N6: theta at x_exit from thwaites.solve on the edge rows in their given order; a None
    edge velocity or a thwaites refusal is (None, "G2-EDGE")."""
    x = edge.get("x_m") if isinstance(edge, dict) else None
    u = edge.get("U_edge_m_s") if isinstance(edge, dict) else None
    r = edge.get("r_wall_m") if isinstance(edge, dict) else None
    if x is None or u is None or r is None or any(v is None for v in u):
        return None, "G2-EDGE"
    try:
        res = thwaites.solve(x, u, nu, r=r, theta0=0.0)
    except ValueError:
        return None, "G2-EDGE"
    return float(np.interp(x_exit, res["x"], res["theta"])), None


def _theta_thw_nominal(re, nu):
    """N7 report only: theta at the exit station of thwaites.nominal_case(Re) fed to thwaites.solve."""
    try:
        x, u, r, ie, _re_ = thwaites.nominal_case(RE_VALUES[re], nu=nu)
        res = thwaites.solve(x, u, nu, r=r, theta0=0.0)
    except ValueError:
        return None
    return float(res["theta"][ie])


def _g2_row(re, vals):
    """One Re's G2 numbers from an N4 dict: theta_thw, theta_rel and the two passes."""
    row = {"theta_exit": None, "separation_free": None, "theta_thw": None, "theta_rel": None,
           "theta_pass": False, "reversal_pass": False, "theta_thw_nominal": None, "missing": True}
    if not isinstance(vals, dict):
        return row
    row["theta_exit"] = vals.get("theta_exit")
    row["separation_free"] = vals.get("separation_free")
    row["reversal_pass"] = bool(vals.get("separation_free") is not None
                                and vals["separation_free"] == 1.0)
    edge, nu, x_exit = vals.get("edge"), vals.get("nu_m2_s"), vals.get("x_exit")
    row["missing"] = any(v is None for v in (row["theta_exit"], row["separation_free"], edge, nu, x_exit))
    if edge is not None and nu is not None and x_exit is not None:
        row["theta_thw"], _err = theta_thwaites(edge, nu, x_exit)
        row["theta_thw_nominal"] = _theta_thw_nominal(re, nu)
        if row["theta_thw"] is not None and row["theta_exit"] is not None:
            row["theta_rel"] = row["theta_exit"] / row["theta_thw"] - 1.0
            row["theta_pass"] = bool(abs(row["theta_rel"]) <= BANDS["theta_rel"])
    return row


def judge_g2(runs_by_re, classes_by_re):
    """N7: per Re theta_rel against Thwaites on the CFD edge velocity, the Re slope and the
    reversal flags; reasons in G2_IDS order, theta_thw_nominal report only."""
    per_re = dict((re, _g2_row(re, runs_by_re.get(re) if isinstance(runs_by_re, dict) else None))
                  for re in RE_POINTS)
    classes = classes_by_re if isinstance(classes_by_re, dict) else {}
    t3 = per_re["re3e4"]["theta_exit"]
    t1 = per_re["re1e4"]["theta_exit"]
    slope = None if t3 is None or t1 is None or t3 <= 0.0 or t1 <= 0.0 \
        else math.log(t3 / t1) / math.log(RE_VALUES["re3e4"] / RE_VALUES["re1e4"])
    slope_pass = bool(slope is not None and abs(slope - BANDS["slope"]) <= BANDS["slope_tol"])
    flags = {"G2-MISSING": any(per_re[re]["missing"] for re in RE_POINTS),
             "G2-UNSTEADY": any(classes.get(re) != "steady" for re in RE_POINTS),
             "G2-EDGE": any(per_re[re]["theta_thw"] is None for re in RE_POINTS),
             "G2-THETA": any(per_re[re]["theta_rel"] is not None and not per_re[re]["theta_pass"]
                             for re in RE_POINTS),
             "G2-SLOPE": slope is not None and not slope_pass,
             "G2-REVERSAL": any(not per_re[re]["reversal_pass"] for re in RE_POINTS)}
    reasons = [g for g in G2_IDS if flags[g]]
    for re in RE_POINTS:
        del per_re[re]["missing"]
    return {"verdict": "PASS" if not reasons else "OPEN", "reasons": reasons, "slope": slope,
            "slope_pass": slope_pass, "per_re": per_re}


def gci3(values):
    """N8: the Roache 1994 three-grid GCI of [f0, f1, f2] (L0, L1, L2, r = 2); f_ext and
    gci_fine only when the observed order is positive, else None."""
    f0, f1, f2 = values
    if any(v is None for v in values):
        return {"values": None, "monotone": None, "p": None, "f_ext": None, "gci_fine": None,
                "gci_fine_abs": None}
    e01, e12 = f0 - f1, f1 - f2
    monotone = bool(e01 * e12 > 0.0)
    p = math.log(e01 / e12) / math.log(2.0) if monotone and e12 != 0.0 else None
    ok = p is not None and p > 0.0
    den = 2.0 ** p - 1.0 if ok else None
    gci = GCI_FS * abs(f2 - f1) / (abs(f2) * den) if ok else None
    return {"values": [f0, f1, f2], "monotone": monotone, "p": p,
            "f_ext": f2 + (f2 - f1) / den if ok else None, "gci_fine": gci,
            "gci_fine_abs": None if gci is None else gci * abs(f2)}


def gci2(f0, f1):
    """N8 report only: the two-grid GCI at the formal order 2 with the 3.0 safety factor."""
    if f0 is None or f1 is None or f1 == 0.0:
        return None
    return GCI_FS_2LEVEL * abs(f1 - f0) / (abs(f1) * (2.0 ** P_FORMAL - 1.0))


def judge_g3(levels_by_re):
    """N8: {re: {level: {"solve": doc|None, "values": m|None}}} into the G3 doc. NOT_RUN with
    G3-LEVELS unless L0, L1 and L2 nominal at BOTH Re have solve.json; two_level always
    reported; otherwise per Re gci3 of Cd and theta_exit, reasons in G3_IDS order."""
    per_re, two_level, complete = {}, {}, True
    for re in RE_POINTS:
        lv = levels_by_re.get(re) if isinstance(levels_by_re, dict) else None
        lv = lv if isinstance(lv, dict) else {}
        two = {}
        for q in ("Cd", "theta_exit"):
            v0 = lv.get(0, {}).get("values")
            v1 = lv.get(1, {}).get("values")
            two[q] = gci2(v0.get(q) if isinstance(v0, dict) else None,
                          v1.get(q) if isinstance(v1, dict) else None)
        two_level[re] = two
        ok = all(isinstance(lv.get(n), dict) and lv[n].get("solve") is not None for n in (0, 1, 2))
        complete = complete and ok
        if ok:
            vs = [lv[n].get("values") for n in (0, 1, 2)]
            per_re[re] = {"Cd": gci3([(v.get("Cd") if isinstance(v, dict) else None) for v in vs]),
                          "theta_exit": gci3([(v.get("theta_exit") if isinstance(v, dict) else None)
                                              for v in vs]),
                          "classes": [lv[n]["solve"].get("class") for n in (0, 1, 2)]}
        else:
            per_re[re] = None
    if not complete:
        return {"verdict": "NOT_RUN", "reasons": ["G3-LEVELS"], "per_re": per_re,
                "two_level": two_level}
    docs = [(re, q, per_re[re][q]) for re in RE_POINTS for q in ("Cd", "theta_exit")]
    flags = {"G3-MISSING": any(d["values"] is None for _r, _q, d in docs),
             "G3-UNSTEADY": any(per_re[re]["classes"][n] != "steady"
                                for re in RE_POINTS for n in (0, 1, 2)),
             "G3-MONO": any(d["values"] is not None and not d["monotone"] for _r, _q, d in docs),
             "G3-GCI-CD": any(q == "Cd" and d["values"] is not None
                              and (d["gci_fine"] is None or d["gci_fine"] > BANDS["gci_cd"])
                              for _r, q, d in docs),
             "G3-GCI-THETA": any(q == "theta_exit" and d["values"] is not None
                                 and (d["gci_fine"] is None or d["gci_fine"] > BANDS["gci_theta"])
                                 for _r, q, d in docs)}
    reasons = [g for g in G3_IDS if g in flags and g != "G3-LEVELS"]
    return {"verdict": "PASS" if not reasons else "OPEN", "reasons": reasons, "per_re": per_re,
            "two_level": two_level}


def _level_rows(wres):
    """The build.json level table of one wedge_mesh.run result: level, cells, h1_max_m."""
    return [{"level": lv["level"], "cells": lv["cells"], "h1_max_m": lv["h1_max_m"]}
            for lv in wres["report"]["levels"]]


def _write_reqs(out_dir):
    """The two locked requirement sets: the golden v3_exit_velocity requirements with only
    operating_point.U_exit_m_s set to RE_POINTS[re] and the lock recomputed (N2)."""
    golden = common.read_json(GOLDEN_REQ)["requirements"]
    for re in RE_POINTS:
        doc = copy.deepcopy(golden)
        doc["operating_point"]["U_exit_m_s"] = RE_POINTS[re]
        doc["lock_sha"] = reqs.lock_sha_of(doc)
        reqs.write_locked(os.path.join(out_dir, "req_" + re), doc)


def build(out_dir, full=False):
    """N2: the geometry (nominal and Lx/De 1.0), the wedge meshes, the two locked requirement
    sets and the cold cases of the run set, then build.json with no absolute path."""
    out_dir = os.path.abspath(out_dir)
    if os.path.exists(out_dir) and (not os.path.isdir(out_dir) or os.listdir(out_dir)):
        raise Refused("NOZ-OUT", "%s exists and is not empty" % out_dir)
    os.makedirs(out_dir, exist_ok=True)
    names = tuple(RUN_SET_REDUCED) + (tuple(FULL_EXTRA) if full else ())
    levels = (0, 1, 2) if full else (0, 1)
    levels_x = (0, 1) if full else (0,)
    geom = export.run_pipeline(export.TEMPLATE, dict(export.NOMINAL), os.path.join(out_dir, "geom"))
    if geom["status"] != "ok":
        raise Refused("NOZ-GEOM", "%s: %s" % (geom["rule"], geom["message"]))
    geom_x = export.run_pipeline(export.TEMPLATE, dict(export.NOMINAL, Lx_over_De=LX_VARIANT),
                                 os.path.join(out_dir, "geom_x"))
    if geom_x["status"] != "ok":
        raise Refused("NOZ-GEOM", "%s: %s" % (geom_x["rule"], geom_x["message"]))
    wedges = {}
    for tag, gdir, lvl in (("wedge", "geom", levels), ("wedge_x", "geom_x", levels_x)):
        w = wedge_mesh.run(os.path.join(out_dir, gdir), os.path.join(out_dir, tag), levels=lvl)
        if w["status"] != "ok" or not (w.get("report") or {}).get("gc6_pass"):
            raise Refused("NOZ-MESH", "%s: %s" % (w.get("rule"), w.get("message")))
        wedges[tag] = w
    _write_reqs(out_dir)
    for name in names:
        info = parse_name(name)
        exit_kind = info["kind"] == "exit"
        r = case_writer.write_case(os.path.join(out_dir, "wedge_x" if exit_kind else "wedge"),
                                   info["level"],
                                   os.path.join(out_dir, "geom_x" if exit_kind else "geom"),
                                   os.path.join(out_dir, "req_" + info["re"]),
                                   os.path.join(out_dir, "cases", name))
        if r["status"] != "ok":
            raise Refused("NOZ-CASE", "%s: %s" % (r["rule"], r["message"]))
    doc = {"version": VERSION, "full": bool(full), "names": list(names),
           "levels": dict((tag, _level_rows(w)) for tag, w in wedges.items()), "cases": {}}
    for name in names:
        info = parse_name(name)
        cj = common.read_json(os.path.join(out_dir, "cases", name, "case.json"))
        doc["cases"][name] = {"level": info["level"], "kind": info["kind"], "re": info["re"],
                              "cells": cj["wedge"]["cells"],
                              "U_exit_m_s": cj["operating_point"]["U_exit_m_s"],
                              "U_inlet_m_s": cj["operating_point"]["U_inlet_m_s"]}
    common.atomic_write(os.path.join(out_dir, "build.json"), common.canonical_json(doc) + common.NL)
    return doc


def values(run_dir):
    """N4: the m dict of the nine metric values plus x_exit, nu, edge and boundary_other_rel, from
    runs/<name>/post.json; None when the file is missing or its status is not ok."""
    p = os.path.join(run_dir, "post.json")
    if not os.path.isfile(p):
        return None
    doc = common.read_json(p)
    if doc.get("status") != "ok" or not isinstance(doc.get("metrics"), dict):
        return None
    m = {}
    for k in VALUE_KEYS:
        met = doc["metrics"].get(k)
        m[k] = met.get("value") if isinstance(met, dict) else None
    ep = (doc.get("stations") or {}).get("exit_plane") or {}
    m["x_exit"] = ep.get("x_m")
    m["nu_m2_s"] = (doc.get("operating_point") or {}).get("nu_m2_s")
    m["edge"] = doc.get("edge")
    bf = doc.get("boundary_flux")
    m["boundary_other_rel"] = bf.get("other_rel") if isinstance(bf, dict) else None
    return m


def run(out_dir, name, exe=None, visible=True, snapshot_fn=None):
    """N3: one solve of cases/<name> into runs/<name> through solve.launch (the D-1 binary of
    bin_gpu.json when exe is None; NOZ-NAME for a bad name, NOZ-OUT when the run directory
    exists); timed, the solve doc returned, wall.json and gpu.json written exactly like
    poiseuille.run, and post.json written when cases/<name>/<iters> exists."""
    out_dir = os.path.abspath(out_dir)
    if name not in ALL_NAMES:
        raise Refused("NOZ-NAME", "name %r is not one of %r" % (name, list(ALL_NAMES)))
    rdir = os.path.join(out_dir, "runs", name)
    if os.path.exists(rdir):
        raise Refused("NOZ-OUT", "%s exists" % rdir)
    info = parse_name(name)
    iters = ITERS[info["level"]]
    geom_dir = os.path.join(out_dir, "geom_x" if info["kind"] == "exit" else "geom")
    if snapshot_fn is None:
        snapshot_fn = poiseuille.gpu_snapshot
    print("[noz] solving %s for %d iterations..." % (name, iters))
    before = snapshot_fn()
    t0 = time.monotonic()
    doc = solve.launch(os.path.join(out_dir, "cases", name), geom_dir, rdir, iters,
                       exe=exe, bin_json=None if exe else BIN_GPU, visible=visible)
    wall_s = time.monotonic() - t0
    after = snapshot_fn()
    os.makedirs(rdir, exist_ok=True)
    with open(os.path.join(rdir, "wall.json"), "wb") as f:
        f.write((common.canonical_json({"wall_s": wall_s}) + chr(10)).encode("utf-8"))
    gpu_doc = {"before": before, "after": after, "shared": poiseuille.shared_flag(before, after),
               "note": poiseuille.GPU_NOTE}
    with open(os.path.join(rdir, "gpu.json"), "wb") as f:
        f.write((common.canonical_json(gpu_doc) + chr(10)).encode("utf-8"))
    tdir = os.path.join(out_dir, "cases", name, str(iters))
    if os.path.isdir(tdir):
        pdoc = post.post(os.path.join(out_dir, "cases", name), str(iters), geom_dir)
        with open(os.path.join(rdir, "post.json"), "wb") as f:
            f.write((common.canonical_json(pdoc) + chr(10)).encode("utf-8"))
    print("[noz] %s class %s in %.1f s" % (name, doc.get("class"), wall_s))
    print("[noz] %s gpu shared %s" % (name, gpu_doc["shared"]))
    return doc


def gate_level_of(solves):
    """N10: 2 when both L2 nominal names have solve.json, else 1 when both L1 do, else 0."""
    for lv in GATE_LEVELS:
        if all(solves.get("L%d_%s" % (lv, re)) is not None for re in RE_POINTS):
            return lv
    return 0


def _solve_row(s):
    """The SOLVE_ROW_KEYS row of one solve.json (None when there is no solve)."""
    if s is None:
        return None
    res = s.get("result") or {}
    return {"class": s.get("class"), "reason_id": s.get("reason_id"), "failed": res.get("failed"),
            "criteria": res.get("criteria"), "n_iter_lines": len(res.get("iterations") or []),
            "log_sha256": (s.get("log") or {}).get("sha256"),
            "binary_sha256": (s.get("binary") or {}).get("sha256")}


def _iter_lines(out_dir, name):
    """solve.log's iter lines of one run, or None without the log."""
    p = os.path.join(out_dir, "runs", name, "solve.log")
    if not os.path.isfile(p):
        return None
    with open(p, "r", encoding="utf-8") as fh:
        return [l for l in fh.read().splitlines() if l.startswith("iter ")]


def _field_shas(out_dir, a, b, iters):
    """{U: [sha_a, sha_b], p: [sha_a, sha_b]} at the pair's final time (poiseuille's _record_tail)."""
    sha = {}
    for fld in ("U", "p"):
        pair = []
        for nm in (a, b):
            p = os.path.join(out_dir, "cases", nm, str(iters), fld)
            pair.append(common.sha256_file(p) if os.path.isfile(p) else None)
        sha[fld] = pair
    return sha


def _bit_identity(sha):
    """True only when both fields of both runs exist and hash equal."""
    return bool(all(sha[f][0] is not None and sha[f][0] == sha[f][1] for f in ("U", "p")))


def repeat_doc(out_dir, solves, vals_of):
    """N9: the G-REPEAT pair, its |a - b| band of Cd, theta_exit and dp, the field sha pair,
    bit identity and iter-line identity; status recorded when both have ok values."""
    if solves.get("L1r_re3e4") is not None:
        a, b, reduced = "L1_re3e4", "L1r_re3e4", False
    else:
        a, b, reduced = "L0_re3e4", "L0r_re3e4", True
    va, vb = vals_of.get(a), vals_of.get(b)
    band = dict((k, None if va is None or vb is None or va.get(k) is None or vb.get(k) is None
                 else abs(va[k] - vb[k])) for k in REPEAT_KEYS)
    sha = _field_shas(out_dir, a, b, ITERS[parse_name(a)["level"]])
    la, lb = _iter_lines(out_dir, a), _iter_lines(out_dir, b)
    reached = all(solves.get(n) is not None and solves[n].get("class") is not None for n in (a, b)) \
        and va is not None and vb is not None
    return {"names": [a, b], "reduced": bool(reduced), "band": band,
            "fields_bit_identical": _bit_identity(sha), "field_sha256": sha,
            "log_iter_lines_identical": None if la is None or lb is None else la == lb,
            "status": "recorded" if reached else "OPEN"}


def sensitivity_doc(solves, vals_of):
    """N9: the exit-tube sensitivity theta_rel = theta_x/theta_nom - 1 and cd_rel, reported,
    never gated."""
    if solves.get("L1x_re3e4") is not None:
        a, b, reduced = "L1_re3e4", "L1x_re3e4", False
    else:
        a, b, reduced = "L0_re3e4", "L0x_re3e4", True
    va, vb = vals_of.get(a), vals_of.get(b)
    ta = va.get("theta_exit") if va else None
    tb = vb.get("theta_exit") if vb else None
    ca = va.get("Cd") if va else None
    cb = vb.get("Cd") if vb else None
    return {"names": [a, b], "reduced": bool(reduced),
            "theta_rel": None if ta is None or tb is None or ta == 0.0 else tb / ta - 1.0,
            "cd_rel": None if ca is None or cb is None or ca == 0.0 else cb / ca - 1.0,
            "status": "reported"}


def record(out_dir):
    """N10: the cad-g123/1 record over the build's run set - the runs rows, G1 on both Re at
    the gate level, G2, G3, G-REPEAT, the exit-tube sensitivity and the cfd_u of E.4. No
    absolute path goes into the record."""
    out_dir = os.path.abspath(out_dir)
    build_doc = common.read_json(os.path.join(out_dir, "build.json"))
    names = list(build_doc["names"])
    cases = build_doc.get("cases") or {}
    solves = {}
    for name in names:
        sj = os.path.join(out_dir, "runs", name, "solve.json")
        solves[name] = common.read_json(sj) if os.path.isfile(sj) else None
    binary = None
    for name in names:
        s = solves[name]
        if s and s.get("binary"):
            binary = {"name": s["binary"].get("name"), "path": os.path.basename(s["binary"].get("path") or ""),
                      "sha256": s["binary"].get("sha256")}
            break
    vals_of = {}
    runs = []
    for name in names:
        info = parse_name(name)
        rdir = os.path.join(out_dir, "runs", name)
        v = values(rdir)
        vals_of[name] = v
        tt = None
        if v is not None and v.get("edge") is not None and v.get("nu_m2_s") is not None \
                and v.get("x_exit") is not None:
            tt, _err = theta_thwaites(v["edge"], v["nu_m2_s"], v["x_exit"])
        wj = os.path.join(rdir, "wall.json")
        gj = os.path.join(rdir, "gpu.json")
        runs.append({"name": name, "level": info["level"], "kind": info["kind"], "re": info["re"],
                     "cells": (cases.get(name) or {}).get("cells"), "solve": _solve_row(solves[name]),
                     "wall_s": common.read_json(wj).get("wall_s") if os.path.isfile(wj) else None,
                     "gpu": common.read_json(gj) if os.path.isfile(gj) else None,
                     "values": None if v is None else dict((k, v.get(k)) for k in VALUE_KEYS),
                     "theta_thw": tt})
    gate = gate_level_of(solves)
    g1_per, g1_reasons = {}, []
    for re in RE_POINTS:
        gname = "L%d_%s" % (gate, re)
        j = judge_g1(vals_of.get(gname), solves[gname].get("class") if solves[gname] else None)
        g1_per[re] = j
        g1_reasons = [g for g in G1_IDS if g in g1_reasons or g in j["reasons"]]
    g1 = {"verdict": "PASS" if not g1_reasons else "OPEN", "reasons": g1_reasons,
          "gate_level": gate, "per_re": g1_per}
    g2 = judge_g2(dict((re, vals_of.get("L%d_%s" % (gate, re))) for re in RE_POINTS),
                  dict((re, solves["L%d_%s" % (gate, re)].get("class") if solves["L%d_%s" % (gate, re)]
                        else None) for re in RE_POINTS))
    g3 = judge_g3(dict((re, dict((n, {"solve": solves.get("L%d_%s" % (n, re)),
                                      "values": vals_of.get("L%d_%s" % (n, re))}) for n in (0, 1, 2)))
                       for re in RE_POINTS))
    rep = repeat_doc(out_dir, solves, vals_of)
    sens = sensitivity_doc(solves, vals_of)

    def gci_abs(q):
        pr = g3["per_re"].get("re3e4")
        d = pr.get(q) if isinstance(pr, dict) else None
        return d.get("gci_fine_abs") if isinstance(d, dict) else None

    cfd_u = {"Cd": {"gci_fine": gci_abs("Cd"), "repeat_band": rep["band"]["Cd"]},
             "theta_exit": {"gci_fine": gci_abs("theta_exit"), "repeat_band": rep["band"]["theta_exit"]},
             "status": "OPEN", "note": CFD_U_NOTE}
    return {"version": VERSION, "bands": BANDS, "iters": dict((str(k), v) for k, v in ITERS.items()),
            "gate_level": gate,
            "reduced": bool(gate < 2), "binary": binary, "runs": runs, "g1": g1, "g2": g2,
            "g3": g3, "g_repeat": rep, "sensitivity": sens, "cfd_u": cfd_u}


def main(argv):
    """--selftest | build OUT_DIR [--full] | run OUT_DIR NAME | record OUT_DIR RECORD_JSON
    (poiseuille.main's style)."""
    if argv == ["--selftest"]:
        try:
            selftest()
        except Exception:
            traceback.print_exc()
            return 1
        return 0
    if len(argv) in (2, 3) and argv[0] == "build" and (len(argv) == 2 or argv[2] == "--full"):
        try:
            doc = build(os.path.abspath(argv[1]), full=len(argv) == 3)
        except wedge_mesh.Refused as r:
            sys.stderr.write("refused %s: %s%s" % (r.rule, r.detail, chr(10)))
            return 2
        for tag in ("wedge", "wedge_x"):
            for lv in doc["levels"][tag]:
                print("%s L%d cells %d h1_max %.6e m" % (tag, lv["level"], lv["cells"], lv["h1_max_m"]))
        print("cases %d (%s)" % (len(doc["names"]), "full" if doc["full"] else "reduced"))
        return 0
    if len(argv) == 3 and argv[0] == "run":
        try:
            doc = run(os.path.abspath(argv[1]), argv[2])
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
        for g in ("g1", "g2", "g3"):
            print("%s %s reasons %s" % (g, rec[g]["verdict"], ",".join(rec[g]["reasons"])))
        return 0
    sys.stderr.write(USAGE + chr(10))
    return 2


def rel(a, b):
    """The relative difference the selftest asserts with."""
    return abs(a / b - 1.0)


def _t1():
    assert parse_name("L0x_re3e4") == {"level": 0, "kind": "exit", "re": "re3e4"}
    assert parse_name("L1r_re3e4") == {"level": 1, "kind": "repeat", "re": "re3e4"}
    assert parse_name("L2_re1e4") == {"level": 2, "kind": "nominal", "re": "re1e4"}
    for bad in ("L3_re3e4", "L0_re2e4", "L0q_re1e4"):
        try:
            parse_name(bad)
            raise AssertionError("parse_name accepted %r" % bad)
        except Refused as r:
            assert r.rule == "NOZ-NAME", r.rule
    print("[ok] T1 parse_name: L0x exit, L1r repeat, and L3_re3e4 / L0_re2e4 / L0q_re1e4 all"
          " refused NOZ-NAME")


def _t2():
    m = {"mass_imbalance": 5e-6, "p0_loss_axis": 0.004, "momentum_closure": 0.019}
    j = judge_g1(m, "steady")
    assert j["verdict"] == "PASS" and j["reasons"] == [], j
    assert j["checks"]["p0loss"]["pass"] is True and j["checks"]["momentum"]["value"] == 0.019
    j = judge_g1(dict(m, p0_loss_axis=-0.006), "steady")
    assert j["verdict"] == "PASS" and j["reasons"] == [] and j["checks"]["p0loss"]["pass"] is True, j
    j = judge_g1(dict(m, p0_loss_axis=0.006), "steady")
    assert j["verdict"] == "OPEN" and j["reasons"] == ["G1-P0LOSS"], j
    assert judge_g1(dict(m, p0_loss_axis=-0.001), "steady")["verdict"] == "PASS"
    j = judge_g1({"mass_imbalance": 2e-5, "p0_loss_axis": 0.006, "momentum_closure": -0.03}, "steady")
    assert j["verdict"] == "OPEN" and j["reasons"] == ["G1-MASS", "G1-P0LOSS", "G1-MOMENTUM"], j
    j = judge_g1(m, "unsteady")
    assert j["verdict"] == "OPEN" and j["reasons"] == ["G1-UNSTEADY"], j
    j = judge_g1(None, None)
    assert j["reasons"][0] == "G1-MISSING" and j["verdict"] == "OPEN"
    print("[ok] T2 judge_g1: the passing triple PASS, the signed p0 loss -0.006 PASS and 0.006"
          " OPEN G1-P0LOSS alone, the triple miss OPEN in G1_IDS order, unsteady and missing"
          " flagged")


def _t3():
    n = 71
    edge = {"x_m": [float(v) for v in np.linspace(0.0, 0.07, n)], "U_edge_m_s": [22.5] * n,
            "r_wall_m": [0.01] * n}
    edge1 = dict(edge, U_edge_m_s=[7.5] * n)
    tt, err = theta_thwaites(edge, 1.5e-5, 0.06)
    assert err is None and rel(tt, 1.3416407864998739e-4) <= 1e-9, (tt, err)
    tt1, err1 = theta_thwaites(edge1, 1.5e-5, 0.06)
    assert rel(tt1, 1.3416407864998739e-4 * 3 ** 0.5) <= 1e-9, tt1
    edge_bad = dict(edge, U_edge_m_s=[22.5] * (n - 1) + [None])
    assert theta_thwaites(edge_bad, 1.5e-5, 0.06) == (None, "G2-EDGE")
    classes = {"re3e4": "steady", "re1e4": "steady"}

    def vals(th3, th1, sep1=1.0):
        return {"re3e4": {"theta_exit": th3, "separation_free": 1.0, "edge": edge,
                          "nu_m2_s": 1.5e-5, "x_exit": 0.06},
                "re1e4": {"theta_exit": th1, "separation_free": sep1, "edge": edge1,
                          "nu_m2_s": 1.5e-5, "x_exit": 0.06}}

    j = judge_g2(vals(tt * 1.05, tt * 1.05 * 3 ** 0.5), classes)
    assert j["verdict"] == "PASS" and j["reasons"] == [], j
    assert abs(j["slope"] - (-0.5)) <= 1e-12 and j["slope_pass"] is True
    assert rel(j["per_re"]["re1e4"]["theta_thw_nominal"],
               _theta_thw_nominal("re1e4", 1.5e-5)) <= 1e-12
    j = judge_g2(vals(tt * 1.05, tt * 1.05 * 3 ** 0.4), classes)
    assert j["reasons"] == ["G2-SLOPE"] and abs(j["slope"] - (-0.4)) <= 1e-12, j
    j = judge_g2(vals(tt * 0.85, tt * 0.85 * 3 ** 0.5), classes)
    assert j["reasons"] == ["G2-THETA"] and j["verdict"] == "OPEN", j
    j = judge_g2(vals(tt * 1.05, tt * 1.05 * 3 ** 0.5, sep1=0.0), classes)
    assert j["reasons"] == ["G2-REVERSAL"] and j["verdict"] == "OPEN", j
    print("[ok] T3 judge_g2: theta_thwaites is the flat-plate sqrt(0.45 nu x / U) at both Re, a"
          " None edge velocity is G2-EDGE, and theta, slope -0.5/-0.4 and reversal judge in"
          " G2_IDS order")


def _t4():
    a = gci3([0.95, 0.96, 0.9625])
    assert a["monotone"] is True and abs(a["p"] - 2.0) <= 1e-9, a
    assert rel(a["gci_fine"], 0.0010822510822511393) <= 1e-9, a["gci_fine"]
    assert rel(a["gci_fine_abs"], 0.0010416666666667215) <= 1e-9, a["gci_fine_abs"]
    b = gci3([7.0e-5, 6.4e-5, 6.25e-5])
    assert abs(b["p"] - 2.0) <= 1e-9 and rel(b["gci_fine"], 0.01) <= 1e-9, b
    c = gci3([0.95, 0.97, 0.96])
    assert c["monotone"] is False and c["p"] is None and c["gci_fine"] is None, c
    d = gci3([0.90, 0.94, 0.95])
    assert rel(d["gci_fine"], 0.0043859649122807215) <= 1e-9, d["gci_fine"]
    e = gci3([7.0e-5, 6.6e-5, 6.0e-5])
    assert e["monotone"] is True and abs(e["p"] - (-0.5849625007211611)) <= 1e-9, e
    assert e["gci_fine"] is None and e["f_ext"] is None
    assert gci2(0.95, 0.96) == 0.010416666666666676, gci2(0.95, 0.96)
    assert gci2(7e-5, 6.4e-5) == 0.09374999999999994, gci2(7e-5, 6.4e-5)
    empty = dict((re, dict((n, {"solve": None, "values": None}) for n in (0, 1, 2)))
                 for re in RE_POINTS)
    j = judge_g3(empty)
    assert j["verdict"] == "NOT_RUN" and j["reasons"] == ["G3-LEVELS"], j
    solved_none = dict((re, dict((n, {"solve": {"class": "steady"}, "values": None})
                                 for n in (0, 1, 2))) for re in RE_POINTS)
    j = judge_g3(solved_none)
    assert j["verdict"] == "OPEN" and j["reasons"][0] == "G3-MISSING", j
    print("[ok] T4 gci3/gci2/judge_g3: the oracle GCI numbers exact to rel 1e-9, the non-monotone"
          " and negative-order cases None, the two-grid values bit-exact, a set without L2 is"
          " NOT_RUN G3-LEVELS, and all-solved levels with None values are OPEN G3-MISSING without"
          " a crash")


def _plant_run(out, name, theta, cd, cls="steady"):
    """T5's planted files for one name: solve.json, post.json (status ok), a solve.log with iter
    lines, and the final-time U and p files the record hashes."""
    v = {"Cd": cd, "theta_exit": theta, "mass_imbalance": 1e-7, "p0_loss_axis": 0.001,
         "momentum_closure": 0.005, "separation_free": 1.0, "dp": 300.0,
         "exit_nonuniformity": 0.005, "H_exit": 1.4}
    post_doc = {"version": "cad-post/1", "status": "ok", "reason_id": None, "message": "",
                "metrics": dict((k, {"value": v[k]}) for k in VALUE_KEYS),
                "stations": {"exit_plane": {"x_m": 0.07}},
                "operating_point": {"nu_m2_s": 1.5e-5},
                "edge": {"x_m": [0.0, 0.07], "U_edge_m_s": [22.5, 22.5], "r_wall_m": [0.01, 0.01]}}
    solve_doc = {"class": cls, "reason_id": None, "message": "",
                 "binary": {"name": "exe", "path": "fake_%s.py" % name, "sha256": "0" * 64},
                 "result": {"failed": [], "criteria": {"U_decades": {"pass": True}},
                            "iterations": [1, 2, 3]}}
    rdir = os.path.join(out, "runs", name)
    os.makedirs(rdir)
    with open(os.path.join(rdir, "solve.json"), "wb") as f:
        f.write((common.canonical_json(solve_doc) + chr(10)).encode("utf-8"))
    with open(os.path.join(rdir, "post.json"), "wb") as f:
        f.write((common.canonical_json(post_doc) + chr(10)).encode("utf-8"))
    with open(os.path.join(rdir, "solve.log"), "w", encoding="utf-8") as f:
        f.write("iter 0  |U| res 1.0e-2  |p| res 1.0e-1  contErr 1.0e-4  T [293.15, 293.15] K"
                "  rho [1.205, 1.205] kg/m3  p0 101325 Pa  dp0/dt 0 Pa/s  M max 1.0e-2 (cell 0)"
                " mean 5.0e-3" + chr(10))
        f.write("run ended: budget | 1000 iterations reached | exit code 0" + chr(10))
    tdir = os.path.join(out, "cases", name, str(ITERS[parse_name(name)["level"]]))
    os.makedirs(tdir)
    for fld, blob in (("U", "dimensions [0 1 -1 0 0 0 0]; internal uniform (0 0 0)"),
                      ("p", "dimensions [0 2 -2 0 0 0 0]; internal uniform 0")):
        with open(os.path.join(tdir, fld), "w", encoding="utf-8") as f:
            f.write(blob + chr(10))
    return v


def _t5(td):
    out = os.path.join(td, "planted")
    os.makedirs(out)
    names = RUN_SET_REDUCED
    build_doc = {"version": VERSION, "full": False, "names": list(names),
                 "levels": {"wedge": [], "wedge_x": []},
                 "cases": dict((n, {"level": parse_name(n)["level"], "kind": parse_name(n)["kind"],
                                    "re": parse_name(n)["re"], "cells": 10140,
                                    "U_exit_m_s": RE_POINTS[parse_name(n)["re"]],
                                    "U_inlet_m_s": RE_POINTS[parse_name(n)["re"]] / 9.0})
                               for n in names)}
    common.atomic_write(os.path.join(out, "build.json"), common.canonical_json(build_doc) + common.NL)
    v0 = _plant_run(out, "L0_re3e4", 1.0e-4, 0.96)
    _plant_run(out, "L0r_re3e4", 1.0e-4, 0.96)
    _plant_run(out, "L0x_re3e4", 1.02e-4, 0.9696)
    rec = record(out)
    assert rec["gate_level"] == 0 and rec["reduced"] is True
    rep = rec["g_repeat"]
    assert rep["names"] == ["L0_re3e4", "L0r_re3e4"] and rep["reduced"] is True, rep
    assert rep["band"]["Cd"] == 0.0 and rep["band"]["theta_exit"] == 0.0, rep["band"]
    assert rep["fields_bit_identical"] is True and rep["log_iter_lines_identical"] is True
    assert rep["status"] == "recorded", rep["status"]
    sens = rec["sensitivity"]
    assert sens["names"] == ["L0_re3e4", "L0x_re3e4"] and sens["reduced"] is True, sens
    assert rel(sens["theta_rel"], 0.02) <= 1e-9 and rel(sens["cd_rel"], 0.01) <= 1e-9
    assert sens["status"] == "reported"
    assert rec["runs"][0]["values"]["Cd"] == v0["Cd"] and rec["runs"][0]["solve"]["class"] == "steady"
    p = os.path.join(out, "cases", "L0r_re3e4", "1000", "U")
    with open(p, "rb") as f:
        data = f.read()
    with open(p, "wb") as f:
        f.write(data + b" ")
    rec2 = record(out)
    assert rec2["g_repeat"]["fields_bit_identical"] is False, rec2["g_repeat"]
    print("[ok] T5 the planted L0/L0r repeat: band 0.0, bit and iter-line identity True, status"
          " recorded, and the Lx sensitivity theta_rel 0.02 reported; one changed byte of U breaks"
          " the bit identity")


def _case_snapshot(root):
    """{relative /-separated path: sha256} of every file under one case directory."""
    out = {}
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            p = os.path.join(dirpath, fn)
            out[os.path.relpath(p, root).replace(os.sep, "/")] = common.sha256_file(p)
    return out


def _t6(out, build_doc):
    assert build_doc["names"] == list(RUN_SET_REDUCED), build_doc["names"]
    rows = dict((lv["level"], lv) for lv in build_doc["levels"]["wedge"])
    assert rows[0]["cells"] == 10115 and rows[1]["cells"] == 40460, rows
    xrows = dict((lv["level"], lv) for lv in build_doc["levels"]["wedge_x"])
    # The brief expected L0x denser than L0; the tree wins: the laminar recipe re-targets
    # cells_l0, so the longer exit tube shrinks nr and L0x comes out with FEWER cells.
    span_a = common.read_json(os.path.join(out, "geom", "geom.json"))["x_span_m"]
    span_x = common.read_json(os.path.join(out, "geom_x", "geom.json"))["x_span_m"]
    assert span_x > span_a and xrows[0]["cells"] != rows[0]["cells"], (span_a, span_x, xrows[0])
    assert build_doc["cases"]["L0_re3e4"]["cells"] == 10115
    sa, sb = (_case_snapshot(os.path.join(out, "cases", n)) for n in ("L0_re3e4", "L0r_re3e4"))
    assert sa and sa == sb, sorted(set(sa) ^ set(sb))
    cj3 = common.read_json(os.path.join(out, "cases", "L0_re3e4", "case.json"))
    cj1 = common.read_json(os.path.join(out, "cases", "L0_re1e4", "case.json"))
    assert rel(cj3["operating_point"]["U_exit_m_s"], 22.5) <= 1e-12, cj3["operating_point"]
    assert rel(cj1["operating_point"]["U_exit_m_s"], 7.5) <= 1e-12, cj1["operating_point"]
    assert rel(build_doc["cases"]["L0_re1e4"]["U_exit_m_s"], 7.5) <= 1e-12
    for re in RE_POINTS:
        rdir = os.path.join(out, "req_" + re)
        doc = common.read_json(os.path.join(rdir, "requirements.json"))
        assert rel(doc["operating_point"]["U_exit_m_s"], RE_POINTS[re]) <= 1e-12
        assert reqs.lock_ok(doc), re
        assert os.path.isfile(os.path.join(rdir, "requirements.lock"))
    try:
        build(out)
        raise AssertionError("a second build into a non-empty directory was accepted")
    except Refused as r:
        assert r.rule == "NOZ-OUT", r.rule
    print("[ok] T6 build reduced: %d names, the L0 nominal mesh %d cells and L1 %d exact, the"
          " Lx/De 1.0 variant longer (x_span %+.6f m) at L0 %d cells (the recipe re-targets 10k,"
          " so fewer, not more), cases L0_re3e4 and L0r_re3e4 byte-identical, U_exit 22.5 / 7.5,"
          " both requirement sets lock, and a second build is NOZ-OUT"
          % (len(build_doc["names"]), rows[0]["cells"], rows[1]["cells"], span_x - span_a,
             xrows[0]["cells"]))


def _t7(td, out):
    fake = os.path.join(td, "fake_solver.py")
    lines = ["import os, shutil, sys",
             "case = sys.argv[1]",
             "iters = int(sys.argv[sys.argv.index('-iters') + 1])",
             "zero = os.path.join(case, '0')",
             "for t in range(50, iters + 1, 50):",
             "    d = os.path.join(case, str(t))",
             "    if not os.path.isdir(d):",
             "        shutil.copytree(zero, d)",
             "for it in range(0, iters, 50):",
             "    print('iter %d  |U| res 1.000000e-02  |p| res 1.000000e-01  contErr 1.000000e-04"
             "  T [293.15, 293.15] K  rho [1.205, 1.205] kg/m3  p0 101325 Pa  dp0/dt 0 Pa/s"
             "  M max 1.000000e-02 (cell 0) mean 5.000000e-03' % it)",
             "print('run ended: budget | %d iterations reached | exit code 0' % iters)"]
    with open(fake, "w", encoding="utf-8") as f:
        f.write(chr(10).join(lines) + chr(10))
    snap = {"status": "ok", "gpu": {"name": "FAKE GPU", "memory_used_mib": 10,
                                    "memory_total_mib": 100, "utilization_pct": 0},
            "compute_apps": [], "detail": ""}
    doc = run(out, "L0_re3e4", exe=[sys.executable, fake], visible=False,
              snapshot_fn=lambda: dict(snap))
    assert doc.get("class") is not None
    rdir = os.path.join(out, "runs", "L0_re3e4")
    assert sorted(os.listdir(rdir)) == ["gpu.json", "post.json", "solve.json", "solve.log",
                                        "wall.json"], sorted(os.listdir(rdir))
    wj = common.read_json(os.path.join(rdir, "wall.json"))
    gj = common.read_json(os.path.join(rdir, "gpu.json"))
    assert isinstance(wj["wall_s"], float) and wj["wall_s"] >= 0.0
    assert gj["shared"] is False and gj["note"] == poiseuille.GPU_NOTE
    assert gj["before"]["gpu"]["name"] == "FAKE GPU" and gj["after"]["compute_apps"] == []
    pj = common.read_json(os.path.join(rdir, "post.json"))
    assert pj["status"] in ("ok", "refused"), pj["status"]
    rec = record(out)
    assert rec["gate_level"] == 0 and rec["reduced"] is True
    assert rec["g3"]["verdict"] == "NOT_RUN" and rec["g3"]["reasons"] == ["G3-LEVELS"]
    assert rec["cfd_u"]["status"] == "OPEN" and rec["cfd_u"]["Cd"]["gci_fine"] is None
    text = common.canonical_json(rec)
    assert ":/" not in text and (":" + chr(92)) not in text, "an absolute path leaked into the record"
    try:
        run(out, "L0_re3e4", exe=[sys.executable, fake], visible=False,
            snapshot_fn=lambda: dict(snap))
        raise AssertionError("a second run of the same name was accepted")
    except Refused as r:
        assert r.rule == "NOZ-OUT", r.rule
    print("[ok] T7 the fake-exe run of L0_re3e4 leaves solve.json, wall.json, gpu.json and"
          " post.json, the record is gate_level 0 reduced with g3 NOT_RUN, holds no absolute"
          " path, and a second run is NOZ-OUT")


def selftest():
    """T1-T7 in one TemporaryDirectory: T1-T4 pure, T5 on a planted out_dir, T6 the real
    reduced build, T7 the fake-exe run and record on T6's build; SELFTEST PASS at the end."""
    _t1()
    _t2()
    _t3()
    _t4()
    with tempfile.TemporaryDirectory() as td:
        _t5(td)
        out = os.path.join(td, "out")
        build_doc = build(out)
        _t6(out, build_doc)
        _t7(td, out)
    print("SELFTEST PASS")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
