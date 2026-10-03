#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""nozzle_turb.py - TG1, TG2a-f, TG3, TG-Cd and TG-REPEAT (docs/16 §H.5, §I): the turbulent
nozzle nozzle_turb_nominal at U_e 30 and 60 m/s.

The requirements set is docs/16 §H.5's nozzle_turb_nominal (D_i 0.30 m, CR 2, poly5, L/D_i 1.5,
a 0.60 m no-slip upstream pipe, I 1 %, l 3 mm) at U_e 30 and 60 m/s, wedge-meshed at two levels
(three with --full), written as cold cad-case-turb/1 cases whose inlet k and omega are the
fixedValue k_ref / omega_ref and whose internal U is (U_inlet 0 0), and solved by the supervisor
with the D-1 binary of bin_gpu.json. The references the gates judge against: the 1/7-power
axisymmetric momentum integral of NACA TM 1218 §17 (eqs. 17.3-17.11) made axisymmetric after
Rott & Crabtree 1952, run on the CFD's own edge velocity sqrt(2 (p0_core - p_wall)) upstream of
the contraction (TG2a), from the CFD's entry theta through the nozzle (TG2b), and for the
Reynolds slope -0.20 (TG2c); the K 3e-6 relaminarisation guard and the laminarescent p -0.005
report of Kline et al. 1967 / arXiv:2306.05972 (TG2d); the exact separation count (TG2e); Head's
entrainment method (ARC R&M 3152) with Ludwieg & Tillmann friction, report only (TG2f); Roache
1997 for the three-grid GCI (TG3). Per docs/16 §H.5 a TG0 miss leaves no turbulent nozzle number
counting: while the TG0 record's verdict is not PASS every TG gate carries TG-TG0 first and a
PASS verdict reads OPEN.
"""
#
# Definitions (fixed before any run):
#   D1 names L<level><suffix>_<point>: level 0/1/2; suffix "" (nominal, I 0.01), "r" (repeat:
#      a second cold case of the same mesh and requirement set), "i05" (I 0.005) or "i2" (I 0.02);
#      point "u30" or "u60". r, i05 and i2 exist only at u60. A name not in ALL_NAMES is NTB-NAME.
#      parse_name(name) -> {"level", "kind": "nominal"|"repeat"|"intensity", "point", "intensity"}.
#   D2 build(out_dir, full=False, names=None, levels=None): names default RUN_SET_REDUCED
#      (+ FULL_EXTRA when full), levels default (0, 1, 2) when full else (0, 1); a name whose
#      level is not in levels is NTB-NAME. NTB-OUT unless out_dir is missing or empty.
#      turb_params.json = export.TURB_NOMINAL; geom/ by export.py run TEMPLATE params geom in a
#      FRESH child process (NTB-GEOM when its exit code is not 0, message the last 300 characters
#      of its stderr); wedge/ by wedge_mesh.run_turb(geom, wedge, U_e=60.0, levels=levels)
#      (NTB-MESH when its status is not ok or a level's gc7 pass is not True); req_u30, req_u60:
#      the golden v3_exit_velocity requirements with REQ-001 value 0.30, REQ-002 value
#      0.30 / sqrt(2), operating_point.U_exit_m_s = POINTS[point], flow_quote "at %r m/s", the
#      lock recomputed and written by reqs.write_locked; cases/<name> by case_writer
#      .write_turb_case(wedge, level, geom, req_<point>, cases/<name>, turb=TURB_DEFAULT with the
#      suffix's intensity) (NTB-CASE with its rule); build.json = canonical JSON {"version",
#      "full", "names", "levels": [{"level", "cells", "h1_max_m", "gc7_pass"}], "geom_sha256",
#      "cases": {name: {"level", "kind", "point", "intensity", "cells", "U_exit_m_s",
#      "U_inlet_m_s", "Re_De", "K_max_apriori", "yplus1_apriori"}}} with no absolute path.
#   D3 run(out_dir, name, iters=None, exe=None, visible=True, snapshot_fn=None): NTB-NAME,
#      NTB-OUT when runs/<name> exists; iters = ITERS[level] unless given; nvidia-smi
#      before/after, wall.json, gpu.json exactly as nozzle_nominal.run; solve.launch(cases/<name>,
#      geom, runs/<name>, iters, exe=exe, bin_json=None if exe else BIN_GPU, history_fn=history,
#      visible=visible); then when cases/<name>/<iters> exists, runs/<name>/post.json = canonical
#      post.post_turb(cases/<name>, str(iters), geom). Prints the [ntb] progress lines.
#   D4 history(case_dir, geom_dir, iters_list): per time post.post_turb(case_dir, str(t),
#      geom_dir); row {"iter": t, "time": str(t), "status", "reason_id", "dp", "Cd",
#      "post_sha256": common.sha256_of(doc)}, dp and Cd the nozzle metrics' values (None unless
#      status ok and the metric's reason_id is None). Never raises.
#   D5 values(run_dir) from runs/<name>/post.json (None when missing or not ok): the VALUE_KEYS
#      scalars - Cd, theta_exit, dstar_exit, H_exit, dp, mass_imbalance, mach_max,
#      separation_free, exit_nonuniformity from nozzle.metrics[k].value; closure_turb =
#      nozzle.momentum_turb.closure; yplus1_frac_le1, yplus1_max from wall_shear.wall_nozzle;
#      K_max, p_min from nozzle.accel; x_exit = nozzle.stations.exit_plane.x_m;
#      D_e = 2 * nozzle.stations.exit_plane.R_m; nu_m2_s = operating_point.nu_m2_s - plus edge =
#      nozzle.edge, edge_upstream = nozzle.edge_upstream, entry = nozzle.entry.
#   D6 TG1 judge_tg1(m, solve_class): mass v <= 1e-5, momentum abs(closure_turb) <= 0.02, yplus
#      yplus1_frac_le1 >= 0.99, mach mach_max <= 0.3, each {"value", "limit", "pass"}; reasons in
#      TG1_IDS order: TG1-MISSING (m None or any of the four None), TG1-UNSTEADY (class not
#      "steady"), TG1-MASS, TG1-MOMENTUM, TG1-YPLUS, TG1-MACH (each only when its value exists
#      and fails). The y+ is post's cell-centre wall-shear y+ d_P sqrt(tau_w,f) / nu, not the
#      solver's printed k-based y*; a TG1-YPLUS miss asks for one re-mesh from the measured u_tau
#      (a mesh change, a later run) - this unit only reports it.
#   D7 TG2a theta_entry_ref(edge_up, entry, nu): stations x = [entry.x_inlet_m] + edge_up.x_m,
#      U = [U0] + edge_up.U_edge_m_s with U0 the first row's U, r = [r0] + edge_up.r_wall_m with
#      r0 the first row's r; turb_integral.solve(x, U, nu, r=r, theta0=0.0); theta_ref_m =
#      np.interp(entry.x_m, res x, res theta); U_edge0_m_s = np.interp(entry.x_m, x, U),
#      L_m = entry.x_m - entry.x_inlet_m, theta_plate_m = 0.037 L (U_edge0 L / nu)^-0.2 (report
#      only); n_stations = len(x), reason_id None. Any None U, a ValueError from solve, or an
#      entry without x_m: every number None and reason_id TG2-EDGE.
#   D8 TG2b theta_exit_ref(edge_up, edge, entry, x_exit, nu): the combined rows = edge_up rows
#      then edge rows (x, U, r); TG2-EDGE unless the combined x is strictly increasing and no U
#      is None; x0 = entry.x_m, U0 / r0 = np.interp(x0, combined x, combined U / r); stations
#      [x0] + every combined row with x > x0; turb_integral.solve(..., theta0=entry.theta_m);
#      theta_ref_m = np.interp(x_exit, ...), dstar_ref_m = turb_integral.H17 * theta_ref_m,
#      n_stations. TG2f head_exit(edge_up, edge, entry, x_exit, nu): turb_integral.head on the
#      SAME stations with theta0 = entry.theta_m, H0 = entry.H, curves="digitised":
#      {"status": "ok", "message": "", "theta_m" and "H" interpolated at x_exit,
#      "n_extrapolated", "finite", "residual": turb_integral.head_residual()}; on TG2-EDGE or a
#      ValueError {"status": "refused", "message": str(error), the numbers None, "residual"}.
#      Report only.
#   D9 TG2 judge_tg2(points, classes): points = {"u30": row|None, "u60": row|None}, row keys
#      theta0_m, theta0_ref_m, theta_e_m, theta_e_ref_m, K_max, p_min, separation_free, Re_De.
#      Per point theta0_rel = theta0/theta0_ref - 1 (entry_pass abs <= 0.10), theta_e_rel =
#      theta_e/theta_e_ref - 1 (exit_pass abs <= 0.15), guard_pass = K_max <= 3e-6,
#      laminarescent = p_min < -0.005 (report only), separation_pass = separation_free == 1.0.
#      slope = ln(theta_e(u60)/theta_e(u30)) / ln(Re_De(u60)/Re_De(u30)), slope_pass
#      abs(slope - (-0.20)) <= 0.03. guard_pass overall = both points pass; when it fails
#      exit_status and slope_status are "REPORT_ONLY" and TG2-EXIT / TG2-SLOPE are NOT added,
#      else both "GATED". Reasons in TG2_IDS order: TG2-MISSING (a row None or any of theta0_m,
#      theta_e_m, K_max, separation_free None), TG2-UNSTEADY (either class not steady), TG2-EDGE
#      (a CFD theta exists but its ref is None), TG2-ENTRY, TG2-EXIT (gated only), TG2-SLOPE
#      (gated only, slope not None), TG2-GUARD (guard fails), TG2-SEPARATION.
#   D10 TG3 judge_tg3(levels_by_point): {point: {level: {"solve": doc|None, "values": m|None}}};
#      nozzle_nominal's judge_g3 shape with the quantities theta_exit and dp per point: NOT_RUN
#      with ["TG3-LEVELS"] unless L0, L1, L2 nominal at BOTH points have a solve; two_level =
#      gci2 of L0/L1 per quantity always; otherwise gci3 per point and quantity, reasons in
#      TG3_IDS order: TG3-MISSING, TG3-UNSTEADY, TG3-MONO, TG3-GCI-THETA (gci_fine None or
#      > 0.05), TG3-GCI-DP (gci_fine None or > 0.01).
#   D11 TG-Cd tg_cd(m, cr, theta_e_ref, gci_fine, gci2_cd) (report only): {"status": "REPORTED",
#      "Cd", "cd_over_ideal": Cd sqrt(1 - 1/cr^2), "gci_fine", "gci2", "blockage_method":
#      1 - 4 H17 theta_e_ref / D_e, "blockage_cfd": 1 - 4 dstar_exit / D_e} (a None input gives a
#      None number). cr is export.TURB_NOMINAL["CR"].
#   D12 TG0: tg0_verdict = tools/cad/cases/pipe_turb/tg0_record.json's ["tg0"]["verdict"] (None
#      when absent); counts = tg0_verdict == "PASS". counted(doc, tg0_verdict) returns a copy of
#      a gate doc: when tg0_verdict is not "PASS", "TG-TG0" goes FIRST in its reasons and a PASS
#      verdict becomes OPEN (NOT_RUN stays NOT_RUN); otherwise the doc unchanged. A TG0 miss
#      leaves no turbulent nozzle number counting.
#   D13 record(out_dir): gate_level = 2 when L2_u30 and L2_u60 have solve.json, else 1 when both
#      L1 do, else 0; reduced = gate_level < 2; run_set = {"ran": [build names with solve.json],
#      "deferred": [the others]}; per run a RUN_ROW_KEYS row with refs = {"entry":
#      theta_entry_ref, "exit": theta_exit_ref, "head": head_exit} (None without values); TG1 per
#      point at the gate level, overall reasons the union in TG1_IDS order, then counted; TG2 at
#      the gate level (rows built from values and refs), counted; TG3, counted; tg_cd per point
#      at the gate level with gci_fine = gci3 of Cd over L0-L2 (None without L2) and gci2 of Cd
#      over L0/L1; tg_repeat: pair (L2_u60, L2r_u60) when L2r_u60 has a solve.json else
#      (L1_u60, L1r_u60) with reduced True: band abs(a - b) of Cd, theta_exit, dp,
#      fields_bit_identical, field_sha256, log_iter_lines_identical, status "recorded" when both
#      have values else "OPEN" (nozzle_nominal.repeat_doc's shape); sensitivity: base level 1
#      when L1i05_u60 or L1i2_u60 has a solve else 0, rows {"i05": ..., "i2": ...} each {"name",
#      "intensity", "theta_rel": theta_e/theta_e(base nominal) - 1, "cd_rel", "K_max"}, "base"
#      the nominal name, status "reported". binary is the first solve.json's binary with the
#      basename of its path. No absolute path, no out_dir, no time stamp.
#
# Usage:
#   python nozzle_turb.py --selftest
#   python nozzle_turb.py build OUT_DIR [--full]
#   python nozzle_turb.py run OUT_DIR NAME [ITERS]
#   python nozzle_turb.py record OUT_DIR RECORD_JSON
#   python nozzle_turb.py stage-bin
import math
import os
import re
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
import export
import reqs
import turb_integral
import poiseuille
import nozzle_nominal

VERSION = "cad-tg123/1"
POINTS = {"u30": 30.0, "u60": 60.0}                  # U_e m/s (docs/16 §H.5 nozzle_turb_nominal)
INTENSITY = {"": 0.01, "r": 0.01, "i05": 0.005, "i2": 0.02}
ITERS = {0: 2000, 1: 3000, 2: 6000}                  # -iters per level, fixed before any run
RUN_SET_REDUCED = ("L0_u30", "L0_u60", "L1_u30", "L1_u60", "L1r_u60", "L0i05_u60", "L0i2_u60")
FULL_EXTRA = ("L2_u30", "L2_u60", "L2r_u60", "L1i05_u60", "L1i2_u60")
ALL_NAMES = tuple(RUN_SET_REDUCED) + tuple(FULL_EXTRA)
GATE_LEVELS = (2, 1)
BANDS = {"mass_imbalance": 1e-5, "momentum_closure": 0.02, "yplus1_frac_min": 0.99, "mach_max": 0.3,
         "theta0_rel": 0.10, "theta_e_rel": 0.15, "slope": -0.20, "slope_tol": 0.03, "K_lim": 3e-6,
         "p_laminarescent": -0.005, "gci_theta": 0.05, "gci_dp": 0.01}
BIN_GPU = poiseuille.BIN_GPU
TG0_RECORD = os.path.join(CAD, "cases", "pipe_turb", "tg0_record.json")
GOLDEN_REQ = os.path.join(CAD, "fixtures", "reqs", "golden", "v3_exit_velocity.json")
REFUSAL_IDS = ("NTB-OUT", "NTB-NAME", "NTB-GEOM", "NTB-MESH", "NTB-CASE")
TG_TG0 = "TG-TG0"
TG1_IDS = ("TG-TG0", "TG1-MISSING", "TG1-UNSTEADY", "TG1-MASS", "TG1-MOMENTUM", "TG1-YPLUS", "TG1-MACH")
TG2_IDS = ("TG-TG0", "TG2-MISSING", "TG2-UNSTEADY", "TG2-EDGE", "TG2-ENTRY", "TG2-EXIT", "TG2-SLOPE",
           "TG2-GUARD", "TG2-SEPARATION")
TG3_IDS = ("TG-TG0", "TG3-LEVELS", "TG3-MISSING", "TG3-UNSTEADY", "TG3-MONO", "TG3-GCI-THETA", "TG3-GCI-DP")
VALUE_KEYS = ("Cd", "theta_exit", "dstar_exit", "H_exit", "dp", "mass_imbalance", "mach_max", "separation_free",
              "exit_nonuniformity", "closure_turb", "yplus1_frac_le1", "yplus1_max", "K_max", "p_min",
              "x_exit", "D_e", "nu_m2_s")
RECORD_KEYS = ("version", "bands", "iters", "gate_level", "reduced", "run_set", "binary", "tg0_verdict",
               "counts", "runs", "tg1", "tg2", "tg3", "tg_cd", "tg_repeat", "sensitivity")
RUN_ROW_KEYS = ("name", "level", "kind", "point", "intensity", "cells", "solve", "wall_s", "gpu", "values",
                "refs")
REPEAT_KEYS = ("Cd", "theta_exit", "dp")
_NAME_RE = re.compile(r"^L([0-2])(r|i05|i2)?_(u30|u60)$")
USAGE = ("usage: python nozzle_turb.py --selftest" + chr(10)
         + "       python nozzle_turb.py build OUT_DIR [--full]" + chr(10)
         + "       python nozzle_turb.py run OUT_DIR NAME [ITERS]" + chr(10)
         + "       python nozzle_turb.py record OUT_DIR RECORD_JSON" + chr(10)
         + "       python nozzle_turb.py stage-bin")


class Refused(wedge_mesh.Refused):
    """A refusal by id (case_writer.Refused's shape): rule and detail."""


def parse_name(name):
    """D1: L<level><suffix>_<point} into its parts; a name the regex rejects or that is not in
    ALL_NAMES (r, i05 and i2 exist only at u60) is NTB-NAME."""
    m = _NAME_RE.match(name) if isinstance(name, str) else None
    if m is None or name not in ALL_NAMES:
        raise Refused("NTB-NAME", "name %r is not one of %r" % (name, list(ALL_NAMES)))
    suffix = m.group(2) or ""
    return {"level": int(m.group(1)),
            "kind": {"": "nominal", "r": "repeat", "i05": "intensity", "i2": "intensity"}[suffix],
            "point": m.group(3), "intensity": INTENSITY[suffix]}


def _write_reqs(out_dir):
    """D2: req_<point> - the golden requirements with only U_exit_m_s and the flow quote set to
    the point and the lock recomputed (post._fx_turb_chain's recipe)."""
    for point in POINTS:
        doc = common.read_json(GOLDEN_REQ)["requirements"]
        for row in doc["rows"]:
            if row["id"] == "REQ-001":
                row["value"] = 0.30
            if row["id"] == "REQ-002":
                row["value"] = 0.30 / math.sqrt(2.0)
        doc["operating_point"]["U_exit_m_s"] = POINTS[point]
        doc["operating_point"]["flow_quote"] = "at %r m/s" % POINTS[point]
        doc["lock_sha"] = reqs.lock_sha_of(doc)
        reqs.write_locked(os.path.join(out_dir, "req_" + point), doc)


def build(out_dir, full=False, names=None, levels=None):
    """D2: the TURB_NOMINAL geometry (a fresh export child), the turbulent wedge meshes, the two
    locked requirement sets and the cold cases of the name set, then build.json with no
    absolute path."""
    out_dir = os.path.abspath(out_dir)
    if os.path.exists(out_dir) and (not os.path.isdir(out_dir) or os.listdir(out_dir)):
        raise Refused("NTB-OUT", "%s exists and is not empty" % out_dir)
    if names is None:
        names = tuple(RUN_SET_REDUCED) + (tuple(FULL_EXTRA) if full else ())
    if levels is None:
        levels = (0, 1, 2) if full else (0, 1)
    infos = []
    for name in names:
        info = parse_name(name)
        if info["level"] not in levels:
            raise Refused("NTB-NAME", "name %r needs level %d, not one of %r"
                          % (name, info["level"], list(levels)))
        infos.append((name, info))
    os.makedirs(out_dir, exist_ok=True)
    params = os.path.join(out_dir, "turb_params.json")
    common.write_json(params, dict(export.TURB_NOMINAL))
    geom = os.path.join(out_dir, "geom")
    r = subprocess.run([sys.executable, os.path.join(CAD, "export.py"), "run", export.TEMPLATE,
                        params, geom], capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if r.returncode != 0:
        raise Refused("NTB-GEOM", "export.py run failed: %s" % (r.stderr or r.stdout)[-300:])
    wres = wedge_mesh.run_turb(geom, os.path.join(out_dir, "wedge"), U_e=60.0, levels=levels)
    rec = common.read_json(os.path.join(out_dir, "wedge", "wedge_turb.json"))
    if wres["status"] != "ok" or not all(lv["gc7"]["pass"] is True for lv in rec["levels"]):
        raise Refused("NTB-MESH", "%s: %s" % (wres.get("rule"), wres.get("message")))
    _write_reqs(out_dir)
    for name, info in infos:
        cr = case_writer.write_turb_case(os.path.join(out_dir, "wedge"), info["level"], geom,
                                         os.path.join(out_dir, "req_" + info["point"]),
                                         os.path.join(out_dir, "cases", name),
                                         turb=dict(case_writer.TURB_DEFAULT,
                                                   intensity=info["intensity"]))
        if cr["status"] != "ok":
            raise Refused("NTB-CASE", "%s: %s" % (cr["rule"], cr["message"]))
    doc = {"version": VERSION, "full": bool(full), "names": list(names),
           "levels": [{"level": lv["level"], "cells": lv["cells"], "h1_max_m": lv["h1_max_m"],
                       "gc7_pass": lv["gc7"]["pass"]} for lv in rec["levels"]],
           "geom_sha256": rec["geom_sha256"], "cases": {}}
    for name, info in infos:
        cj = common.read_json(os.path.join(out_dir, "cases", name, "case.json"))
        doc["cases"][name] = {"level": info["level"], "kind": info["kind"], "point": info["point"],
                              "intensity": info["intensity"], "cells": cj["mesh"]["cells"],
                              "U_exit_m_s": cj["operating_point"]["U_exit_m_s"],
                              "U_inlet_m_s": cj["operating_point"]["U_inlet_m_s"],
                              "Re_De": cj["turbulence"]["Re_De"],
                              "K_max_apriori": cj["turbulence"]["K_max_apriori"],
                              "yplus1_apriori": cj["turbulence"]["yplus1_apriori"]}
    common.atomic_write(os.path.join(out_dir, "build.json"), common.canonical_json(doc) + common.NL)
    return doc


def _metric_value(doc, key):
    """The nozzle metric's value, None unless the doc is ok and the metric carries no reason."""
    met = ((doc.get("nozzle") or {}).get("metrics") or {}).get(key)
    if doc.get("status") != "ok" or not isinstance(met, dict) or met.get("reason_id") is not None:
        return None
    return met.get("value")


def history(case_dir, geom_dir, iters_list):
    """D4: one history row per time; never raises."""
    rows = []
    for t in iters_list:
        try:
            doc = post.post_turb(case_dir, str(t), geom_dir)
        except Exception:
            doc = {"status": "refused", "reason_id": "POST-ERROR", "message": "post_turb failed"}
        rows.append({"iter": t, "time": str(t), "status": doc.get("status"),
                     "reason_id": doc.get("reason_id"), "dp": _metric_value(doc, "dp"),
                     "Cd": _metric_value(doc, "Cd"), "post_sha256": common.sha256_of(doc)})
    return rows


def values(run_dir):
    """D5: the VALUE_KEYS scalars plus the entry and edge rows of runs/<name>/post.json; None
    when the file is missing or its status is not ok."""
    p = os.path.join(run_dir, "post.json")
    if not os.path.isfile(p):
        return None
    doc = common.read_json(p)
    if doc.get("status") != "ok":
        return None
    noz = doc.get("nozzle") if isinstance(doc.get("nozzle"), dict) else {}
    met = noz.get("metrics") if isinstance(noz.get("metrics"), dict) else {}
    m = {}
    for k in VALUE_KEYS[:9]:
        m[k] = met[k].get("value") if isinstance(met.get(k), dict) else None
    mt = noz.get("momentum_turb") if isinstance(noz.get("momentum_turb"), dict) else {}
    m["closure_turb"] = mt.get("closure")
    ws = doc.get("wall_shear") if isinstance(doc.get("wall_shear"), dict) else {}
    ws = ws.get("wall_nozzle") if isinstance(ws.get("wall_nozzle"), dict) else {}
    m["yplus1_frac_le1"] = ws.get("yplus1_frac_le1")
    m["yplus1_max"] = ws.get("yplus1_max")
    ac = noz.get("accel") if isinstance(noz.get("accel"), dict) else {}
    m["K_max"] = ac.get("K_max")
    m["p_min"] = ac.get("p_min")
    st = noz.get("stations") if isinstance(noz.get("stations"), dict) else {}
    ep = st.get("exit_plane") if isinstance(st.get("exit_plane"), dict) else {}
    m["x_exit"] = ep.get("x_m")
    m["D_e"] = None if ep.get("R_m") is None else 2.0 * ep["R_m"]
    op = doc.get("operating_point") if isinstance(doc.get("operating_point"), dict) else {}
    m["nu_m2_s"] = op.get("nu_m2_s")
    m["edge"] = noz.get("edge")
    m["edge_upstream"] = noz.get("edge_upstream")
    m["entry"] = noz.get("entry")
    return m


def run(out_dir, name, iters=None, exe=None, visible=True, snapshot_fn=None):
    """D3: one solve of cases/<name> into runs/<name> through solve.launch (the D-1 binary of
    bin_gpu.json when exe is None; NTB-NAME for a bad name, NTB-OUT when the run directory
    exists); timed, the solve doc returned, wall.json and gpu.json written exactly like
    nozzle_nominal.run, and post.json written when cases/<name>/<iters> exists."""
    out_dir = os.path.abspath(out_dir)
    if name not in ALL_NAMES:
        raise Refused("NTB-NAME", "name %r is not one of %r" % (name, list(ALL_NAMES)))
    rdir = os.path.join(out_dir, "runs", name)
    if os.path.exists(rdir):
        raise Refused("NTB-OUT", "%s exists" % rdir)
    info = parse_name(name)
    if iters is None:
        iters = ITERS[info["level"]]
    if snapshot_fn is None:
        snapshot_fn = poiseuille.gpu_snapshot
    geom_dir = os.path.join(out_dir, "geom")
    print("[ntb] solving %s for %d iterations..." % (name, iters))
    before = snapshot_fn()
    t0 = time.monotonic()
    doc = solve.launch(os.path.join(out_dir, "cases", name), geom_dir, rdir, iters,
                       exe=exe, bin_json=None if exe else BIN_GPU, history_fn=history,
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
    tdir = os.path.join(out_dir, "cases", name, str(iters))
    if os.path.isdir(tdir):
        pdoc = post.post_turb(os.path.join(out_dir, "cases", name), str(iters), geom_dir)
        with open(os.path.join(rdir, "post.json"), "wb") as f:
            f.write((common.canonical_json(pdoc) + chr(10)).encode("utf-8"))
    print("[ntb] %s class %s in %.1f s" % (name, doc.get("class"), wall_s))
    print("[ntb] %s gpu shared %s" % (name, gpu_doc["shared"]))
    return doc


def theta_entry_ref(edge_up, entry, nu):
    """D7 (TG2a): the 1/7-power integral from the inlet plane (theta 0) over the upstream edge
    rows, read at the entry layer; TG2-EDGE with every number None when the rows cannot feed it."""
    empty = {"theta_ref_m": None, "U_edge0_m_s": None, "L_m": None, "theta_plate_m": None,
             "n_stations": None, "reason_id": "TG2-EDGE"}
    try:
        if not (isinstance(edge_up, dict) and isinstance(entry, dict)):
            return empty
        u_up = list(edge_up["U_edge_m_s"])
        x_up = list(edge_up["x_m"])
        r_up = list(edge_up["r_wall_m"])
        x0, x_in = entry["x_m"], entry["x_inlet_m"]
        if x0 is None or x_in is None or not u_up or any(u is None for u in u_up) \
                or not x_up or not r_up:
            return empty
        xs = [x_in] + x_up
        U = [u_up[0]] + u_up
        r = [r_up[0]] + r_up
        res = turb_integral.solve(xs, U, nu, r=r, theta0=0.0)
        u_e0 = float(np.interp(x0, xs, U))
        L = x0 - x_in
        return {"theta_ref_m": float(np.interp(x0, res["x"], res["theta"])),
                "U_edge0_m_s": u_e0, "L_m": L,
                "theta_plate_m": 0.037 * L * (u_e0 * L / nu) ** -0.2,
                "n_stations": len(xs), "reason_id": None}
    except (ValueError, TypeError, KeyError, IndexError):
        return empty


def _exit_stations(edge_up, edge, entry):
    """D8's combined stations: the upstream edge rows then the nozzle wall rows, interpolated at
    the entry x; TG2-EDGE unless the combined x is strictly increasing with no None U."""
    xs = list(edge_up["x_m"]) + list(edge["x_m"])
    U = list(edge_up["U_edge_m_s"]) + list(edge["U_edge_m_s"])
    r = list(edge_up["r_wall_m"]) + list(edge["r_wall_m"])
    if any(u is None for u in U) or any(xs[i] >= xs[i + 1] for i in range(len(xs) - 1)) \
            or not isinstance(entry, dict):
        raise ValueError("TG2-EDGE")
    x0 = entry.get("x_m")
    if x0 is None or entry.get("theta_m") is None:
        raise ValueError("TG2-EDGE")
    keep = [i for i, x in enumerate(xs) if x > x0]
    if not keep:
        raise ValueError("TG2-EDGE")
    return ([x0] + [xs[i] for i in keep],
            [float(np.interp(x0, xs, U))] + [U[i] for i in keep],
            [float(np.interp(x0, xs, r))] + [r[i] for i in keep])


def theta_exit_ref(edge_up, edge, entry, x_exit, nu):
    """D8 (TG2b): the same integral from the CFD's entry theta through the nozzle wall rows, read
    at x_exit with its 9/7 delta*."""
    empty = {"theta_ref_m": None, "dstar_ref_m": None, "n_stations": None, "reason_id": "TG2-EDGE"}
    try:
        if not (isinstance(edge_up, dict) and isinstance(edge, dict)):
            return empty
        st_x, st_U, st_r = _exit_stations(edge_up, edge, entry)
        if x_exit is None:
            return empty
        res = turb_integral.solve(st_x, st_U, nu, r=st_r, theta0=entry["theta_m"])
        theta_ref = float(np.interp(x_exit, res["x"], res["theta"]))
        return {"theta_ref_m": theta_ref, "dstar_ref_m": turb_integral.H17 * theta_ref,
                "n_stations": len(st_x), "reason_id": None}
    except (ValueError, TypeError, KeyError, IndexError):
        return empty


def head_exit(edge_up, edge, entry, x_exit, nu):
    """D8 (TG2f, report only): Head's entrainment method on the SAME stations as TG2b."""
    try:
        if not (isinstance(edge_up, dict) and isinstance(edge, dict)):
            raise ValueError("TG2-EDGE")
        st_x, st_U, st_r = _exit_stations(edge_up, edge, entry)
        if x_exit is None:
            raise ValueError("TG2-EDGE")
        res = turb_integral.head(st_x, st_U, nu, r=st_r, theta0=entry["theta_m"], H0=entry["H"],
                                 curves="digitised")
        return {"status": "ok", "message": "",
                "theta_m": float(np.interp(x_exit, res["x"], res["theta"])),
                "H": float(np.interp(x_exit, res["x"], res["H"])),
                "n_extrapolated": res["n_extrapolated"], "finite": res["finite"],
                "residual": turb_integral.head_residual()}
    except (ValueError, TypeError, KeyError, IndexError) as error:
        return {"status": "refused", "message": str(error), "theta_m": None, "H": None,
                "n_extrapolated": None, "finite": None, "residual": turb_integral.head_residual()}


def judge_tg1(m, solve_class):
    """D6: conservation and the resolved-wall y+ and Mach guards of one run; the y+ is post's
    cell-centre wall-shear y+, not the solver's printed k-based y*."""
    row = m if isinstance(m, dict) else {}
    mass = row.get("mass_imbalance")
    mom = row.get("closure_turb")
    yp = row.get("yplus1_frac_le1")
    mach = row.get("mach_max")
    checks = (("mass", mass, BANDS["mass_imbalance"], "TG1-MASS",
               lambda v: v <= BANDS["mass_imbalance"]),
              ("momentum", mom, BANDS["momentum_closure"], "TG1-MOMENTUM",
               lambda v: abs(v) <= BANDS["momentum_closure"]),
              ("yplus", yp, BANDS["yplus1_frac_min"], "TG1-YPLUS",
               lambda v: v >= BANDS["yplus1_frac_min"]),
              ("mach", mach, BANDS["mach_max"], "TG1-MACH", lambda v: v <= BANDS["mach_max"]))
    docs, reasons = {}, []
    if m is None or any(v is None for _k, v, _l, _r, _f in checks):
        reasons.append("TG1-MISSING")
    if solve_class != "steady":
        reasons.append("TG1-UNSTEADY")
    for key, val, limit, rid, fn in checks:
        ok = None if val is None else bool(fn(val))
        docs[key] = {"value": val, "limit": limit, "pass": ok}
        if ok is False:
            reasons.append(rid)
    reasons = [g for g in TG1_IDS if g in reasons]
    return {"verdict": "PASS" if not reasons else "OPEN", "reasons": reasons, "checks": docs}


def counted(doc, tg0_verdict):
    """D12: a TG0 miss (the record's verdict not PASS) carries TG-TG0 first and demotes a PASS
    verdict to OPEN; NOT_RUN stays NOT_RUN; a PASS TG0 leaves the doc unchanged."""
    if tg0_verdict == "PASS":
        return doc
    out = dict(doc)
    out["reasons"] = [TG_TG0] + list(out.get("reasons") or [])
    if out.get("verdict") == "PASS":
        out["verdict"] = "OPEN"
    return out


def judge_tg2(points, classes):
    """D9: the entry and exit layers against the 1/7-power references, the Reynolds slope, the
    relaminarisation guard and the separation flag; a failing guard demotes TG2-EXIT and
    TG2-SLOPE to report only."""
    pts = points if isinstance(points, dict) else {}
    cls = classes if isinstance(classes, dict) else {}
    per, missing = {}, False
    for point in POINTS:
        row = pts.get(point) if isinstance(pts.get(point), dict) else None
        d = {"row": row, "theta0_rel": None, "theta_e_rel": None, "entry_pass": None,
             "exit_pass": None, "guard_pass": None, "laminarescent": None,
             "separation_pass": None, "edge": False}
        if row is None or any(row.get(k) is None
                              for k in ("theta0_m", "theta_e_m", "K_max", "separation_free")):
            missing = True
            per[point] = d
            continue
        t0r, ter = row.get("theta0_ref_m"), row.get("theta_e_ref_m")
        d["theta0_rel"] = None if t0r is None else row["theta0_m"] / t0r - 1.0
        d["theta_e_rel"] = None if ter is None else row["theta_e_m"] / ter - 1.0
        d["entry_pass"] = None if d["theta0_rel"] is None \
            else bool(abs(d["theta0_rel"]) <= BANDS["theta0_rel"])
        d["exit_pass"] = None if d["theta_e_rel"] is None \
            else bool(abs(d["theta_e_rel"]) <= BANDS["theta_e_rel"])
        d["edge"] = t0r is None or ter is None
        d["guard_pass"] = bool(row["K_max"] <= BANDS["K_lim"])
        d["laminarescent"] = None if row.get("p_min") is None \
            else bool(row["p_min"] < BANDS["p_laminarescent"])
        d["separation_pass"] = bool(row["separation_free"] == 1.0)
        per[point] = d
    reasons = []
    if missing:
        reasons.append("TG2-MISSING")
    if any(cls.get(p) != "steady" for p in POINTS):
        reasons.append("TG2-UNSTEADY")
    if any(per[p]["edge"] for p in POINTS):
        reasons.append("TG2-EDGE")
    if any(per[p]["entry_pass"] is False for p in POINTS):
        reasons.append("TG2-ENTRY")
    guard = all(per[p]["guard_pass"] is True for p in POINTS)
    slope = None
    ra, rb = pts.get("u30") if isinstance(pts.get("u30"), dict) else {}, \
        pts.get("u60") if isinstance(pts.get("u60"), dict) else {}
    if not missing and ra.get("theta_e_m") is not None and rb.get("theta_e_m") is not None \
            and ra.get("Re_De") and rb.get("Re_De"):
        slope = math.log(rb["theta_e_m"] / ra["theta_e_m"]) / math.log(rb["Re_De"] / ra["Re_De"])
    slope_pass = None if slope is None else bool(abs(slope - BANDS["slope"]) <= BANDS["slope_tol"])
    status = "GATED" if guard else "REPORT_ONLY"
    if guard:
        if any(per[p]["exit_pass"] is False for p in POINTS):
            reasons.append("TG2-EXIT")
        if slope is not None and slope_pass is False:
            reasons.append("TG2-SLOPE")
    if not missing and not guard:
        reasons.append("TG2-GUARD")
    if any(per[p]["separation_pass"] is False for p in POINTS):
        reasons.append("TG2-SEPARATION")
    reasons = [g for g in TG2_IDS if g in reasons]
    return {"verdict": "PASS" if not reasons else "OPEN", "reasons": reasons, "per_point": per,
            "slope": slope, "slope_pass": slope_pass, "guard_pass": guard,
            "exit_status": status, "slope_status": status}


def judge_tg3(levels_by_point):
    """D10: nozzle_nominal.judge_g3's shape over theta_exit and dp per point (NOT_RUN without L2
    at both points; the two-grid estimate always reported)."""
    lvs = levels_by_point if isinstance(levels_by_point, dict) else {}
    per_point, two_level, complete = {}, {}, True
    for point in POINTS:
        lv = lvs.get(point) if isinstance(lvs.get(point), dict) else {}
        two = {}
        for q in ("theta_exit", "dp"):
            v0 = lv.get(0, {}).get("values")
            v1 = lv.get(1, {}).get("values")
            two[q] = nozzle_nominal.gci2(v0.get(q) if isinstance(v0, dict) else None,
                                         v1.get(q) if isinstance(v1, dict) else None)
        two_level[point] = two
        ok = all(isinstance(lv.get(n), dict) and lv[n].get("solve") is not None for n in (0, 1, 2))
        complete = complete and ok
        if ok:
            vs = [lv[n].get("values") for n in (0, 1, 2)]
            per_point[point] = {
                "theta_exit": nozzle_nominal.gci3(
                    [(v.get("theta_exit") if isinstance(v, dict) else None) for v in vs]),
                "dp": nozzle_nominal.gci3(
                    [(v.get("dp") if isinstance(v, dict) else None) for v in vs]),
                "classes": [lv[n]["solve"].get("class") for n in (0, 1, 2)]}
        else:
            per_point[point] = None
    if not complete:
        return {"verdict": "NOT_RUN", "reasons": ["TG3-LEVELS"], "per_point": per_point,
                "two_level": two_level}
    docs = [(p, q, per_point[p][q]) for p in POINTS for q in ("theta_exit", "dp")]
    flags = {"TG3-MISSING": any(d["values"] is None for _p, _q, d in docs),
             "TG3-UNSTEADY": any(per_point[p]["classes"][n] != "steady"
                                 for p in POINTS for n in (0, 1, 2)),
             "TG3-MONO": any(d["values"] is not None and not d["monotone"] for _p, _q, d in docs),
             "TG3-GCI-THETA": any(q == "theta_exit" and d["values"] is not None
                                  and (d["gci_fine"] is None or d["gci_fine"] > BANDS["gci_theta"])
                                  for _p, q, d in docs),
             "TG3-GCI-DP": any(q == "dp" and d["values"] is not None
                               and (d["gci_fine"] is None or d["gci_fine"] > BANDS["gci_dp"])
                               for _p, q, d in docs)}
    reasons = [g for g in TG3_IDS if flags.get(g)]
    return {"verdict": "PASS" if not reasons else "OPEN", "reasons": reasons,
            "per_point": per_point, "two_level": two_level}


def tg_cd(m, cr, theta_e_ref, gci_fine, gci2_cd):
    """D11 (report only): Cd over its ideal value and the two blockage estimates."""
    row = m if isinstance(m, dict) else {}
    cd, de, ds = row.get("Cd"), row.get("D_e"), row.get("dstar_exit")
    return {"status": "REPORTED", "Cd": cd,
            "cd_over_ideal": None if cd is None or not cr else cd * math.sqrt(1.0 - 1.0 / cr ** 2),
            "gci_fine": gci_fine, "gci2": gci2_cd,
            "blockage_method": None if theta_e_ref is None or not de
            else 1.0 - 4.0 * turb_integral.H17 * theta_e_ref / de,
            "blockage_cfd": None if ds is None or not de else 1.0 - 4.0 * ds / de}


def gate_level_of(solves):
    """D13: 2 when both L2 nominal points have solve.json, else 1 when both L1 do, else 0."""
    for lv in GATE_LEVELS:
        if all(solves.get("L%d_%s" % (lv, p)) is not None for p in POINTS):
            return lv
    return 0


def _repeat_doc(out_dir, solves, vals_of):
    """D13: the TG-REPEAT pair at u60 - (L2, L2r) when the L2r solve exists, else (L1, L1r) with
    reduced True; the band of Cd, theta_exit and dp with field and iter-line identity reported."""
    if solves.get("L2r_u60") is not None:
        a, b, reduced = "L2_u60", "L2r_u60", False
    else:
        a, b, reduced = "L1_u60", "L1r_u60", True
    va, vb = vals_of.get(a), vals_of.get(b)
    band = dict((k, None if va is None or vb is None or va.get(k) is None or vb.get(k) is None
                 else abs(va[k] - vb[k])) for k in REPEAT_KEYS)
    iters = ITERS[parse_name(a)["level"]]
    sha = nozzle_nominal._field_shas(out_dir, a, b, iters)
    la, lb = nozzle_nominal._iter_lines(out_dir, a), nozzle_nominal._iter_lines(out_dir, b)
    reached = all(solves.get(n) is not None and solves[n].get("class") is not None for n in (a, b)) \
        and va is not None and vb is not None
    return {"names": [a, b], "reduced": bool(reduced), "band": band,
            "fields_bit_identical": nozzle_nominal._bit_identity(sha), "field_sha256": sha,
            "log_iter_lines_identical": None if la is None or lb is None else la == lb,
            "status": "recorded" if reached else "OPEN"}


def _sensitivity_doc(solves, vals_of):
    """D13: the inlet-turbulence sensitivity theta_rel and cd_rel against the same-level nominal,
    reported, never gated."""
    base_lv = 1 if solves.get("L1i05_u60") is not None or solves.get("L1i2_u60") is not None else 0
    base = "L%d_u60" % base_lv
    vb = vals_of.get(base)
    rows = {}
    for tag in ("i05", "i2"):
        name = "L%d%s_u60" % (base_lv, tag)
        v = vals_of.get(name)
        rows[tag] = {"name": name, "intensity": INTENSITY[tag],
                     "theta_rel": None if v is None or vb is None or v.get("theta_exit") is None
                     or vb.get("theta_exit") is None else v["theta_exit"] / vb["theta_exit"] - 1.0,
                     "cd_rel": None if v is None or vb is None or v.get("Cd") is None
                     or vb.get("Cd") is None else v["Cd"] / vb["Cd"] - 1.0,
                     "K_max": None if v is None else v.get("K_max")}
    return {"base": base, "rows": rows, "status": "reported"}


def _refs_of(vals):
    """D13: the three reference docs of one run's values (all None without them)."""
    refs = {"entry": None, "exit": None, "head": None}
    if vals is None or vals.get("nu_m2_s") is None or not isinstance(vals.get("entry"), dict):
        return refs
    refs["entry"] = theta_entry_ref(vals.get("edge_upstream"), vals["entry"], vals["nu_m2_s"])
    if vals.get("x_exit") is not None and isinstance(vals.get("edge"), dict):
        refs["exit"] = theta_exit_ref(vals.get("edge_upstream"), vals["edge"], vals["entry"],
                                      vals["x_exit"], vals["nu_m2_s"])
        refs["head"] = head_exit(vals.get("edge_upstream"), vals["edge"], vals["entry"],
                                 vals["x_exit"], vals["nu_m2_s"])
    return refs


def record(out_dir):
    """D13: the cad-tg123/1 record over the build's run set - the runs rows, TG1, TG2, TG3,
    TG-Cd, TG-REPEAT and the inlet-turbulence sensitivity, every gate through counted. No
    absolute path goes into the record."""
    out_dir = os.path.abspath(out_dir)
    build_doc = common.read_json(os.path.join(out_dir, "build.json"))
    names = list(build_doc["names"])
    cases = build_doc.get("cases") or {}
    solves = {}
    for name in names:
        sj = os.path.join(out_dir, "runs", name, "solve.json")
        solves[name] = common.read_json(sj) if os.path.isfile(sj) else None
    tg0_verdict = None
    if os.path.isfile(TG0_RECORD):
        tg0_verdict = (common.read_json(TG0_RECORD).get("tg0") or {}).get("verdict")
    gate = gate_level_of(solves)
    vals_of, refs_of, runs = {}, {}, []
    for name in names:
        info = parse_name(name)
        rdir = os.path.join(out_dir, "runs", name)
        v = values(rdir)
        vals_of[name] = v
        refs = _refs_of(v)
        refs_of[name] = refs
        wj, gj = os.path.join(rdir, "wall.json"), os.path.join(rdir, "gpu.json")
        runs.append({"name": name, "level": info["level"], "kind": info["kind"],
                     "point": info["point"], "intensity": info["intensity"],
                     "cells": (cases.get(name) or {}).get("cells"),
                     "solve": nozzle_nominal._solve_row(solves[name]),
                     "wall_s": common.read_json(wj).get("wall_s") if os.path.isfile(wj) else None,
                     "gpu": common.read_json(gj) if os.path.isfile(gj) else None,
                     "values": None if v is None else dict((k, v.get(k)) for k in VALUE_KEYS),
                     "refs": refs})
    tg1_per, tg1_reasons = {}, []
    for point in POINTS:
        gname = "L%d_%s" % (gate, point)
        j = judge_tg1(vals_of.get(gname), (solves.get(gname) or {}).get("class"))
        tg1_per[point] = j
        tg1_reasons = [g for g in TG1_IDS if g in tg1_reasons or g in j["reasons"]]
    tg1 = counted({"verdict": "PASS" if not tg1_reasons else "OPEN", "reasons": tg1_reasons,
                   "gate_level": gate, "per_point": tg1_per}, tg0_verdict)
    rows2, classes2, head2 = {}, {}, {}
    for point in POINTS:
        gname = "L%d_%s" % (gate, point)
        v = vals_of.get(gname)
        refs = refs_of.get(gname) or {}
        row = None
        if v is not None and isinstance(v.get("entry"), dict):
            er, xr = refs.get("entry") or {}, refs.get("exit") or {}
            row = {"theta0_m": v["entry"].get("theta_m"), "theta0_ref_m": er.get("theta_ref_m"),
                   "theta_e_m": v.get("theta_exit"), "theta_e_ref_m": xr.get("theta_ref_m"),
                   "K_max": v.get("K_max"), "p_min": v.get("p_min"),
                   "separation_free": v.get("separation_free"),
                   "Re_De": (cases.get(gname) or {}).get("Re_De")}
        rows2[point] = row
        classes2[point] = (solves.get(gname) or {}).get("class")
        head2[point] = refs.get("head")
    tg2_doc = judge_tg2(rows2, classes2)
    tg2_doc["gate_level"] = gate
    tg2_doc["head"] = head2
    tg2 = counted(tg2_doc, tg0_verdict)
    tg3 = counted(judge_tg3(dict((p, dict((n, {"solve": solves.get("L%d_%s" % (n, p)),
                                               "values": vals_of.get("L%d_%s" % (n, p))})
                                         for n in (0, 1, 2))) for p in POINTS)), tg0_verdict)
    cr = export.TURB_NOMINAL["CR"]
    tg_cd_doc = {}
    for point in POINTS:
        gname = "L%d_%s" % (gate, point)
        xr = (refs_of.get(gname) or {}).get("exit") or {}
        if all(solves.get("L%d_%s" % (n, point)) is not None for n in (0, 1, 2)):
            g3 = nozzle_nominal.gci3([(vals_of.get("L%d_%s" % (n, point)) or {}).get("Cd")
                                      for n in (0, 1, 2)])
            gci_fine = g3["gci_fine"]
        else:
            gci_fine = None
        gci2_cd = nozzle_nominal.gci2((vals_of.get("L0_%s" % point) or {}).get("Cd"),
                                      (vals_of.get("L1_%s" % point) or {}).get("Cd"))
        tg_cd_doc[point] = tg_cd(vals_of.get(gname), cr, xr.get("theta_ref_m"), gci_fine, gci2_cd)
    binary = None
    for name in names:
        s = solves[name]
        if s and s.get("binary"):
            binary = {"name": s["binary"].get("name"),
                      "path": os.path.basename(s["binary"].get("path") or ""),
                      "sha256": s["binary"].get("sha256")}
            break
    return {"version": VERSION, "bands": BANDS,
            "iters": dict((str(k), v) for k, v in ITERS.items()),
            "gate_level": gate, "reduced": bool(gate < 2),
            "run_set": {"ran": [n for n in names if solves[n] is not None],
                        "deferred": [n for n in names if solves[n] is None]},
            "binary": binary, "tg0_verdict": tg0_verdict, "counts": tg0_verdict == "PASS",
            "runs": runs, "tg1": tg1, "tg2": tg2, "tg3": tg3, "tg_cd": tg_cd_doc,
            "tg_repeat": _repeat_doc(out_dir, solves, vals_of),
            "sensitivity": _sensitivity_doc(solves, vals_of)}


def main(argv):
    """--selftest | build OUT_DIR [--full] | run OUT_DIR NAME [ITERS] | record OUT_DIR
    RECORD_JSON | stage-bin (poiseuille.stage_pinned) - refusals exit 2, bad argv usage exit 2."""
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
        for lv in doc["levels"]:
            print("wedge L%d cells %d h1_max %.6e m gc7 %s"
                  % (lv["level"], lv["cells"], lv["h1_max_m"], lv["gc7_pass"]))
        print("cases %d (%s)" % (len(doc["names"]), "full" if doc["full"] else "reduced"))
        return 0
    if len(argv) in (3, 4) and argv[0] == "run":
        iters = None
        if len(argv) == 4:
            try:
                iters = int(argv[3])
            except ValueError:
                sys.stderr.write(USAGE + chr(10))
                return 2
        try:
            doc = run(os.path.abspath(argv[1]), argv[2], iters=iters)
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
        print("tg1 %s tg2 %s tg3 %s counts %s"
              % (rec["tg1"]["verdict"], rec["tg2"]["verdict"], rec["tg3"]["verdict"],
                 rec["counts"]))
        return 0 if all(rec[g]["verdict"] == "PASS" for g in ("tg1", "tg2", "tg3")) else 1
    if argv == ["stage-bin"]:
        doc = poiseuille.stage_pinned()
        print("staged %s sha256 %s" % (doc["path"], doc["sha256"]))
        return 0
    sys.stderr.write(USAGE + chr(10))
    return 2


def _plant_fields(case_dir, t, geom_dir):
    """The selftest's planted turbulent entry layer at time t (post's T21 plant): u = 30
    min(1, y/delta)^(1/7) with y = max(R cos(2.5 deg) - hypot(y, z), 0) for x < 0 and u = 30 for
    x >= 0, p = -50 x, nut 0, written through post._fx_plant_c and post._fx_write_turb, then the
    time's nut rewritten so everything from the word boundaryField on is the solver's exact
    written tail (an empty boundaryField, two blank lines, the 79-byte closing rule)."""
    case = common.read_json(os.path.join(case_dir, "case.json"))
    types = post._fx_types_turb(case)
    mesh = post.load_mesh(os.path.join(case_dir, "constant", "polyMesh"))
    tags = common.read_json(os.path.join(geom_dir, "tags.json"))
    planes = dict((q["name"], q["x"]) for q in tags["planes"])
    lays = post.layers(mesh)
    lc = [lay for lay in lays if abs(lay["x"] - planes["contraction_start"]) <= post.FLAT_TOL_M][0]
    x_target = planes["contraction_start"] - post.ENTRY_DI * 2.0 * lc["R"]
    lay = min(lays, key=lambda l: abs(l["x"] - x_target))
    R = lay["R"]
    delta = 0.1 * R

    def fn_u(pt):
        y = np.maximum(R * math.cos(math.radians(2.5)) - np.hypot(pt[:, 1], pt[:, 2]), 0.0)
        u = 30.0 * np.minimum(1.0, y / delta) ** (1.0 / 7.0)
        u = np.where(pt[:, 0] < 0.0, u, 30.0)
        return np.stack([u, np.zeros_like(u), np.zeros_like(u)], axis=1)

    U, p, nut = post._fx_plant_c(mesh, fn_u, lambda pt: -50.0 * pt[:, 0],
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
    """The test's fake solver binary in the temp dir (never a tree file): it plants the turbulent
    entry layer on the five window times of its -iters and prints the steady fixture log."""
    path = os.path.join(td, "fake_ntb.py")
    lines = ["import os, sys",
             "sys.path.insert(0, %r)" % os.path.dirname(os.path.abspath(__file__)),
             "import nozzle_turb",
             "case = sys.argv[1]",
             "geom = os.path.join(os.path.dirname(os.path.dirname(case)), 'geom')",
             "iters = int(sys.argv[sys.argv.index('-iters') + 1])",
             "for t in range(iters - 200, iters + 1, 50):",
             "    nozzle_turb._plant_fields(case, str(t), geom)",
             "with open(os.path.join(%r, 'fixtures', 'solve', 'steady', 'solve.log'), 'rb') as f:" % CAD,
             "    sys.stdout.buffer.write(f.read())"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(chr(10).join(lines) + chr(10))
    return path


def _t1():
    """T1: the constants and the name grammar."""
    assert len(ALL_NAMES) == 12, len(ALL_NAMES)
    assert parse_name("L1r_u60") == {"level": 1, "kind": "repeat", "point": "u60",
                                    "intensity": 0.01}
    assert parse_name("L0i05_u60")["intensity"] == 0.005
    assert parse_name("L2i2_u30" if "L2i2_u30" in ALL_NAMES else "L1i2_u60")["kind"] == "intensity"
    for bad in ("L0r_u30", "L3_u30", "L0_u45"):
        try:
            parse_name(bad)
            raise AssertionError("%s was not refused" % bad)
        except Refused as r:
            assert r.rule == "NTB-NAME", (bad, r.rule)
    assert ITERS == {0: 2000, 1: 3000, 2: 6000} and BANDS["K_lim"] == 3e-6
    assert BANDS["slope"] == -0.20 and BANDS["slope_tol"] == 0.03 and BANDS["gci_theta"] == 0.05
    print("[ok] T1 the 12 names, parse_name's level/kind/point/intensity, the NTB-NAME refusals"
          " and the fixed bands and iters")


def _t2():
    """T2: the three references on synthetic rows at nu 1.516e-5."""
    nu = 1.516e-5
    up = {"x_m": [-0.595 + 0.01 * k for k in range(60)],
          "U_edge_m_s": [15.0] * 60, "r_wall_m": [0.15] * 60}
    entry = {"x_inlet_m": -0.6, "x_m": -0.015, "theta_m": 1.0e-3, "H": 1.3}
    r1 = theta_entry_ref(up, entry, nu)
    assert abs(r1["theta_ref_m"] / 0.0015235194583118456 - 1.0) <= 1e-9, r1
    assert abs(r1["theta_plate_m"] / 0.001523519458311846 - 1.0) <= 1e-9, r1["theta_plate_m"]
    assert abs(r1["L_m"] - 0.585) <= 1e-12, r1["L_m"]
    assert r1["n_stations"] == 61 and r1["reason_id"] is None
    noz = {"x_m": [0.01 * k for k in range(1, 56)], "U_edge_m_s": [15.0] * 55,
           "r_wall_m": [0.15] * 55}
    r2 = theta_exit_ref(up, noz, entry, 0.45, nu)
    assert abs(r2["theta_ref_m"] / 0.001977771611112404 - 1.0) <= 1e-9, r2
    assert abs(r2["dstar_ref_m"] / 0.0025428492142873767 - 1.0) <= 1e-9, r2["dstar_ref_m"]
    assert r2["n_stations"] == 57
    u3 = [15.0 * (1.0 + min(x, 0.45) / 0.45) for x in noz["x_m"]]
    noz3 = {"x_m": noz["x_m"], "U_edge_m_s": u3,
            "r_wall_m": [0.15 / math.sqrt(u / 15.0) for u in u3]}
    r3 = theta_exit_ref(up, noz3, entry, 0.45, nu)
    assert abs(r3["theta_ref_m"] / 0.0006497452625859042 - 1.0) <= 1e-9, r3
    h3 = head_exit(up, noz3, entry, 0.45, nu)
    assert h3["status"] == "ok" and h3["finite"] is True and h3["theta_m"] > 0.0, h3
    up4 = dict(up)
    up4["U_edge_m_s"] = list(up["U_edge_m_s"])
    up4["U_edge_m_s"][10] = None
    r4 = theta_entry_ref(up4, entry, nu)
    assert all(r4[k] is None for k in ("theta_ref_m", "U_edge0_m_s", "L_m", "theta_plate_m",
                                       "n_stations")) and r4["reason_id"] == "TG2-EDGE"
    noz4 = {"x_m": [-0.007 + 0.01 * k for k in range(55)], "U_edge_m_s": [15.0] * 55,
            "r_wall_m": [0.15] * 55}
    r5 = theta_exit_ref(up, noz4, entry, 0.45, nu)
    assert r5["reason_id"] == "TG2-EDGE" and r5["theta_ref_m"] is None
    h5 = head_exit(up, noz4, entry, 0.45, nu)
    assert h5["status"] == "refused", h5
    print("[ok] T2 the references on synthetic rows: theta0 %.18f and its plate limit, theta_e"
          " %.18f with dstar %.18f, the accelerated r(u) case %.18f with Head ok, and the three"
          " TG2-EDGE refusals"
          % (r1["theta_ref_m"], r2["theta_ref_m"], r2["dstar_ref_m"], r3["theta_ref_m"]))


def _t3():
    """T3: judge_tg1 on the measured u60 row and its failures."""
    m60 = {"mass_imbalance": 2.99e-6, "closure_turb": 1.3689e-6, "yplus1_frac_le1": 1.0,
           "mach_max": 0.179}
    j1 = judge_tg1(m60, "steady")
    assert j1["verdict"] == "PASS" and j1["reasons"] == [], j1
    assert judge_tg1(m60, "unsteady")["reasons"] == ["TG1-UNSTEADY"]
    j2 = judge_tg1({"mass_imbalance": 1.2e-5, "closure_turb": -0.025, "yplus1_frac_le1": 0.98,
                    "mach_max": 0.31}, "steady")
    assert j2["reasons"] == ["TG1-MASS", "TG1-MOMENTUM", "TG1-YPLUS", "TG1-MACH"], j2["reasons"]
    j3 = judge_tg1(None, None)
    assert j3["reasons"] == ["TG1-MISSING", "TG1-UNSTEADY"], j3["reasons"]
    assert j1["checks"]["mass"]["limit"] == 1e-5 and j1["checks"]["mach"]["limit"] == 0.3
    print("[ok] T3 judge_tg1: the measured u60 row passes, unsteady hits TG1-UNSTEADY, the four"
          " failing bands and the missing doc hit their reasons in TG1_IDS order")


def _t4():
    """T4: judge_tg2 on the two synthetic points and its reasons."""
    re30, re60 = 419786.3476701141, 839572.6953402282
    b30 = {"theta0_m": 1.40e-3, "theta0_ref_m": 1.50e-3, "theta_e_m": 0.70e-3,
           "theta_e_ref_m": 0.75e-3, "K_max": 2.1e-6, "p_min": -0.004, "separation_free": 1.0,
           "Re_De": re30}
    b60 = {"theta0_m": 1.22e-3, "theta0_ref_m": 1.30e-3, "theta_e_m": 0.61e-3,
           "theta_e_ref_m": 0.66e-3, "K_max": 1.1e-6, "p_min": -0.012, "separation_free": 1.0,
           "Re_De": re60}
    st = {"u30": "steady", "u60": "steady"}
    d = judge_tg2({"u30": b30, "u60": b60}, st)
    assert d["verdict"] == "PASS" and d["reasons"] == [], d["reasons"]
    assert abs(d["per_point"]["u30"]["theta0_rel"] / (-0.06666666666666665) - 1.0) <= 1e-12
    assert abs(d["per_point"]["u60"]["theta0_rel"] / (-0.06153846153846154) - 1.0) <= 1e-12
    assert abs(d["per_point"]["u60"]["theta_e_rel"] / (-0.0757575757575758) - 1.0) <= 1e-12
    assert abs(d["slope"] / (-0.19854567938208015) - 1.0) <= 1e-12, d["slope"]
    assert d["per_point"]["u30"]["laminarescent"] is False
    assert d["per_point"]["u60"]["laminarescent"] is True
    assert d["exit_status"] == "GATED" and d["slope_status"] == "GATED"
    b = judge_tg2({"u30": b30, "u60": dict(b60, theta_e_ref_m=0.50e-3)}, st)
    assert abs(b["per_point"]["u60"]["theta_e_rel"] / 0.21999999999999997 - 1.0) <= 1e-12
    assert b["reasons"] == ["TG2-EXIT"], b["reasons"]
    c = judge_tg2({"u30": b30, "u60": dict(b60, theta_e_m=0.50e-3, theta_e_ref_m=0.55e-3)}, st)
    assert abs(c["slope"] / (-0.4854268271702417) - 1.0) <= 1e-12, c["slope"]
    assert c["reasons"] == ["TG2-SLOPE"], c["reasons"]
    d4 = judge_tg2({"u30": dict(b30, K_max=3.5e-6),
                    "u60": dict(b60, theta_e_m=0.50e-3, theta_e_ref_m=0.55e-3)}, st)
    assert d4["reasons"] == ["TG2-GUARD"], d4["reasons"]
    assert d4["exit_status"] == "REPORT_ONLY" and d4["slope_status"] == "REPORT_ONLY"
    e1 = judge_tg2({"u30": dict(b30, separation_free=0.0), "u60": b60}, st)
    assert e1["reasons"] == ["TG2-SEPARATION"], e1["reasons"]
    e2 = judge_tg2({"u30": b30, "u60": None}, st)
    assert e2["reasons"][0] == "TG2-MISSING", e2["reasons"]
    e3 = judge_tg2({"u30": dict(b30, theta0_ref_m=None), "u60": b60}, st)
    assert "TG2-EDGE" in e3["reasons"], e3["reasons"]
    e4 = judge_tg2({"u30": b30, "u60": b60}, {"u30": "unsteady", "u60": "steady"})
    assert e4["reasons"][0] == "TG2-UNSTEADY", e4["reasons"]
    print("[ok] T4 judge_tg2: the base passes at slope %.17f with the laminarescent flag, and the"
          " exit, slope, guard, separation, missing, edge and unsteady reasons hit in order"
          % d["slope"])


def _t5():
    """T5: judge_tg3, tg_cd and counted."""
    def lv(theta, dp):
        return dict((n, {"solve": {"class": "steady"},
                         "values": {"theta_exit": theta[n], "dp": dp[n]}}) for n in (0, 1, 2))
    th = [6.2e-4, 6.0e-4, 5.95e-4]
    good = dict((p, lv(th, [1600.0, 1620.0, 1625.0])) for p in POINTS)
    d = judge_tg3(good)
    assert d["verdict"] == "PASS" and d["reasons"] == [], d["reasons"]
    g_th = d["per_point"]["u30"]["theta_exit"]["gci_fine"]
    g_dp = d["per_point"]["u30"]["dp"]["gci_fine"]
    assert abs(g_th / 0.003501400560223921 - 1.0) <= 1e-9, g_th
    assert abs(g_dp / 0.001282051282051282 - 1.0) <= 1e-9, g_dp
    assert abs(d["per_point"]["u30"]["theta_exit"]["p"] - 2.0) <= 1e-12
    db = judge_tg3(dict((p, lv(th, [1600.0, 1620.0, 1660.0])) for p in POINTS))
    assert db["reasons"] == ["TG3-GCI-DP"], db["reasons"]
    dn = judge_tg3(dict((p, dict((n, good[p][n]) for n in (0, 1))) for p in POINTS))
    assert dn["verdict"] == "NOT_RUN" and dn["reasons"] == ["TG3-LEVELS"]
    tc = tg_cd({"Cd": 1.13874934, "dstar_exit": 7.963417644135802e-4,
                "D_e": 0.21213203435596426}, 2.0, 0.62e-3, None, None)
    assert tc["status"] == "REPORTED"
    assert abs(tc["cd_over_ideal"] / 0.986185856982763 - 1.0) <= 1e-12, tc["cd_over_ideal"]
    assert abs(tc["blockage_method"] / 0.9849689301370631 - 1.0) <= 1e-12
    assert abs(tc["blockage_cfd"] / 0.9849840356864292 - 1.0) <= 1e-12
    assert tc["gci_fine"] is None and tc["gci2"] is None
    assert counted({"verdict": "PASS", "reasons": []}, "OPEN") == {"verdict": "OPEN",
                                                                  "reasons": ["TG-TG0"]}
    nr = counted({"verdict": "NOT_RUN", "reasons": ["TG3-LEVELS"]}, "OPEN")
    assert nr == {"verdict": "NOT_RUN", "reasons": ["TG-TG0", "TG3-LEVELS"]}
    assert counted({"verdict": "PASS", "reasons": []}, "PASS") == {"verdict": "PASS",
                                                                  "reasons": []}
    print("[ok] T5 judge_tg3 passes at theta gci %.18f and dp gci %.18f with p 2.0, the dp band"
          " and TG3-LEVELS hit, tg_cd's three numbers match and counted carries TG-TG0"
          % (g_th, g_dp))


def _t6(out):
    """T6: the real L0_u60 build, its build.json and case, and the two refusals."""
    doc = build(out, names=("L0_u60",), levels=(0,))
    assert sorted(doc.keys()) == sorted(("version", "full", "names", "levels", "geom_sha256",
                                         "cases")), sorted(doc.keys())
    assert len(doc["levels"]) == 1
    l0 = doc["levels"][0]
    assert l0["cells"] == 5499 and l0["gc7_pass"] is True, l0
    crow = doc["cases"]["L0_u60"]
    assert crow["point"] == "u60" and crow["intensity"] == 0.01 and crow["kind"] == "nominal"
    cj = common.read_json(os.path.join(out, "cases", "L0_u60", "case.json"))
    assert cj["version"] == "cad-case-turb/1" and cj["kind"] == "nozzle"
    with open(os.path.join(out, "cases", "L0_u60", "0", "k"), "r", encoding="utf-8") as f:
        ktxt = f.read()
    assert "    inlet" + chr(10) + "    {" + chr(10) + "        type            fixedValue;" in ktxt
    try:
        build(out, names=("L0_u60",), levels=(0,))
        raise AssertionError("the second build was not refused")
    except Refused as r:
        assert r.rule == "NTB-OUT", r.rule
    try:
        build(os.path.join(os.path.dirname(out), "out6b"), names=("L1_u60",), levels=(0,))
        raise AssertionError("the L1 build was not refused")
    except Refused as r:
        assert r.rule == "NTB-NAME", r.rule
    print("[ok] T6 build: the L0_u60 case row (5499 cells, gc7 pass, I 0.01), the case.json kind"
          " nozzle with the fixedValue inlet k, a second build NTB-OUT and the level mismatch"
          " NTB-NAME")


def _t7(td, out):
    """T7: the launch path through the fake binary, the history row and the record."""
    fake = _fake_bin(td)

    def snap():
        return {"status": "ok", "gpu": {"name": "fake", "memory_used_mib": 0,
                                        "memory_total_mib": 0, "utilization_pct": 0},
                "compute_apps": [], "detail": ""}

    doc = run(out, "L0_u60", iters=600, exe=[sys.executable, fake], visible=False,
              snapshot_fn=snap)
    assert doc["class"] == "steady", (doc["class"], doc["reason_id"], doc["message"])
    rdir = os.path.join(out, "runs", "L0_u60")
    assert sorted(os.listdir(rdir)) == ["gpu.json", "post.json", "solve.json", "solve.log",
                                        "wall.json"]
    rows = history(os.path.join(out, "cases", "L0_u60"), os.path.join(out, "geom"), [600])
    pj = common.read_json(os.path.join(rdir, "post.json"))
    cd_post = pj["nozzle"]["metrics"]["Cd"]["value"]
    assert len(rows) == 1 and rows[0]["iter"] == 600 and rows[0]["time"] == "600"
    assert rows[0]["status"] == "ok" and rows[0]["reason_id"] is None
    assert rows[0]["Cd"] == cd_post, (rows[0]["Cd"], cd_post)
    assert rows[0]["post_sha256"]
    mesh = post.load_mesh(os.path.join(out, "cases", "L0_u60", "constant", "polyMesh"))
    tags = common.read_json(os.path.join(out, "geom", "tags.json"))
    planes = dict((q["name"], q["x"]) for q in tags["planes"])
    lays = post.layers(mesh)
    lc = [lay for lay in lays if abs(lay["x"] - planes["contraction_start"]) <= post.FLAT_TOL_M][0]
    x_target = planes["contraction_start"] - post.ENTRY_DI * 2.0 * lc["R"]
    lay = min(lays, key=lambda l: abs(l["x"] - x_target))
    delta = 0.1 * lay["R"]
    en = pj["nozzle"]["entry"]
    assert abs(en["theta_m"] / delta / 0.09689205233387911 - 1.0) <= 1e-9, en["theta_m"] / delta
    rec = record(out)
    assert list(rec.keys()) == list(RECORD_KEYS), list(rec.keys())
    assert len(rec["runs"]) == 1
    assert sorted(rec["runs"][0].keys()) == sorted(RUN_ROW_KEYS)
    assert rec["gate_level"] == 0 and rec["reduced"] is True
    assert rec["run_set"] == {"ran": ["L0_u60"], "deferred": []}
    assert rec["tg0_verdict"] == "OPEN" and rec["counts"] is False
    assert rec["tg1"]["verdict"] == "OPEN" and rec["tg1"]["reasons"][0] == "TG-TG0"
    assert rec["tg2"]["verdict"] == "OPEN" and rec["tg2"]["reasons"][0] == "TG-TG0"
    assert rec["tg3"]["verdict"] == "NOT_RUN"
    assert rec["tg3"]["reasons"] == ["TG-TG0", "TG3-LEVELS"]
    assert rec["runs"][0]["refs"]["entry"]["theta_ref_m"] > 0.0
    blob = common.canonical_json(rec)
    assert "C:/" not in blob and ("C:" + chr(92)) not in blob and td not in blob
    assert common.canonical_json(record(out)) == blob
    print("[ok] T7 the fake binary's run is steady in the five run files with the planted entry"
          " theta/delta %.17f, the history row's Cd equals the post doc's, and the record is"
          " keyed, TG0-counted and byte-identical across two calls" % (en["theta_m"] / delta))


def _t8(td, out):
    """T8: the CLI through subprocess."""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nozzle_turb.py")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    rec_path = os.path.join(td, "rec_cli.json")
    p = subprocess.run([sys.executable, script, "record", out, rec_path], capture_output=True,
                       text=True, encoding="utf-8", errors="replace", env=env)
    assert p.returncode == 1, (p.returncode, p.stdout[-300:], p.stderr[-300:])
    rec = common.read_json(rec_path)
    assert rec["tg3"]["verdict"] == "NOT_RUN"
    assert "tg1 OPEN tg2 OPEN tg3 NOT_RUN counts False" in p.stdout, p.stdout
    p = subprocess.run([sys.executable, script, "run", out, "L9_u30"], capture_output=True,
                       text=True, encoding="utf-8", errors="replace", env=env)
    assert p.returncode == 2 and "NTB-NAME" in p.stderr, (p.returncode, p.stderr[-200:])
    p = subprocess.run([sys.executable, script], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env)
    assert p.returncode == 2 and "usage" in p.stderr, p.stderr[-200:]
    print("[ok] T8 the CLI: record exits 1 with the NOT_RUN tg3 written, run L9_u30 exits 2 with"
          " NTB-NAME on stderr, no argv exits 2 with the usage")


def selftest():
    """T1-T8 in one TemporaryDirectory, build(out, names=("L0_u60",), levels=(0,)) once for
    T6-T8; SELFTEST PASS at the end."""
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "out")
        _t1()
        _t2()
        _t3()
        _t4()
        _t5()
        _t6(out)
        _t7(td, out)
        _t8(td, out)
    print("SELFTEST PASS")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
