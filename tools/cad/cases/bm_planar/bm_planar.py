#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""bm_planar.py - CAD-22 (docs/16 §I CAD-22, §H.4 G5 the report-only gate): Bell & Mehta's planar
CR 7.7, L/H_i 0.89 contraction built with the four wall laws, one GPU solve per law per call, and
the cad-g5/1 comparison with their Table 4, REPORTED, never gated.

The convention (settled by the supervisor from the report; put in the doc string and the record
verbatim): Bell & Mehta's Table 1 lists tunnel B (the 5th-order 2-D contraction this replication
copies) as CR 7.7, inlet 91 x 137 cm, exit 91 x 18 cm, length 244 cm, L/H 0.89. 244/137 = 1.78,
not 0.89, but 244/(2 x 137) = 0.891. The same factor 2 holds for tunnel A (91/(2 x 38) = 1.197,
listed 1.20), while the 3-D tunnels C (120/120 = 1.00, listed 1.0) and D (122/114 = 1.070, listed
1.07) use their full inlet height. A and B are the splitter-plate tunnels; Table 1 lists one
stream's 137 cm, the panel code models the splitter plate as an inviscid image, so the
contraction's full height is 274 cm. Fig. 14 is consistent: L/H_i 0.89 is attached, while 1.79
(= 244/137 read the other way) is marked separated, which tunnel B is not (Figs. 9, Table 2).
So H_i is the full inlet height across the symmetry plane: H_i = 2 h_i, h_i the distance from the
symmetry plane to the curved wall. In the nozzle template that is exactly D_i = H_i with the
meridian read as the planar half-channel y(x), so L_over_Di = L/H_i = 0.89 and
CR_template = 7.7 ** 2 (the template sets R_e = R_i/sqrt(CR), so h_e = h_i/7.7).

Definitions (fixed before any run):
  B1 convention. convention() returns {"H_i": "full inlet height across the symmetry plane,
     H_i = 2 h_i", "evidence": [per TABLE1 row {"tunnel", "L_cm", "listed_cm", "factor", "listed",
     "computed": L/(factor * listed_cm), "matches": round(computed, 2) == listed}],
     "rejected": {"reading": "H_i = h_i", "B": 244/137, "why": "Fig. 14 marks L/H_i 1.79
     separated; tunnel B is attached"}}. Derived: h_i = H_i/2 = 1.37, h_e = h_i/7.7,
     L = 0.89 H_i = 2.4386, U_inlet = U_exit/7.7.
  B2 params_for(law). BM-NAME for a law not in LAWS. {"D_i": 2.74, "CR": 7.7 ** 2,
     "L_over_Di": 0.89, "law": law, "x_m": 0.5 if law == "cubic_matched" else None,
     "Lx_over_De": 1.0, "Lu_over_Di": 0.5, "upstream_role": "slip", "t_wall": 0.003}.
     Geometry: export.run_pipeline(export.TEMPLATE, params_for(law), out_dir/geom_<law>);
     status not "ok" -> BM-GEOM with its rule and message.
  B3 the planar mesh (gmsh-build GEOM_DIR MSH_PATH, a fresh process per build, wedge_mesh's
     options: Terminal 0, NumThreads 1, OCCTargetUnit "M", MSH 4.1 ASCII). edges =
     wedge_mesh.stations(planes) from geom_dir/tags.json; surfs, area, ybar =
     wedge_mesh._split_meridian(gmsh, geom_dir, edges) (12 blocks in x order); extrude every
     block surface by (0, 0, dz) with numElements=[1], recombine=True; synchronize. Then on each
     SOURCE block surface (z = 0): every boundary curve whose x-extent (bounding box) is below
     1e-6 m is a "vertical" curve: transfinite with nr + 1 nodes, Progression q =
     wedge_mesh.solve_q(h1, |y1 - y0|, nr) from its two ends (wedge_mesh._curve_ends),
     coefficient q when the first end has the larger y else 1/q (so the first cell h1 is at the
     wall end); every other curve gets n + 1 uniform nodes with n = upstream (block 0), block
     (blocks 1..10), exit (block 11) of RECIPE n_axial; then setTransfiniteSurface and
     setRecombine(2, s). Surface groups by bounding box (tolerance 1e-6 m): z-extent below tol at
     z = 0 -> front, at z = dz -> back; x-extent below tol at the inlet plane x -> inlet, at the
     outlet x -> outlet (other x-flat surfaces are internal station faces, in no group);
     y-extent below tol at y = 0 -> symmetry; otherwise x_max <= contraction_start + tol ->
     slip_upstream; otherwise wall_nozzle. Want counts {inlet 1, outlet 1, slip_upstream 1,
     wall_nozzle 11, symmetry 12, front 12, back 12} and 11 internal, else BM-MESH. Physical
     groups by PATCHES name plus volume "fluid"; generate(3); the volume elements must be hex
     only (gmsh type 5), else BM-MESH. Print one canonical JSON line: {"cells", "nr",
     "n_axial": [12 counts], "meridian_area_m2": area, "gmsh": gmsh.GMSH_API_VERSION}; a Refused
     prints {"refused", "detail"} and exits 3 (wedge_mesh.main's pattern).
  B4 mesh_law(geom_dir, mesh_dir, bins). Runs the child into mesh_dir/planar.msh, then
     wedge_mesh.convert(bins, msh, mesh_dir/case, TYPE_ARGS), wedge_mesh.patch_types must equal
     WANT_TYPES (else BM-MESH), wedge_mesh.run_check(bins, mesh_dir/case, mesh_dir/check.json,
     min_tau=RECIPE["check_tau"]) must have exit 0 (else BM-MESH; tau_min and non_orth_max_deg
     are REPORTED). Measured from post.load_mesh: cells; per-patch face counts; volume_m3 =
     sum V; volume_ref_m3 = meridian_area * dz; volume_rel = volume_m3/volume_ref_m3 - 1
     (|.| <= 3e-3 else BM-MESH); h1_exit_m = h_e - min vertex y of the exit-layer face (B6) with
     the largest Cf_y; h1_rel = h1_exit_m/h1 - 1 (|.| <= 0.05 else BM-MESH). Returns the mesh row
     {"cells", "patch_faces", "types", "volume_m3", "volume_ref_m3", "volume_rel", "h1_exit_m",
     "h1_rel", "tau_min", "non_orth_max_deg", "msh_sha256"}.
  B5 write_case(mesh_dir, law, out_dir). poiseuille.write_case's shape: BM-OUT when out_dir
     exists; the five polyMesh files copied byte for byte; one row per boundary patch in boundary
     order with role ROLES[name]; U/p/T BCs: velocity_inlet, pressure_outlet, wall, slip from
     case_writer.bc_table(role, [U_inlet, 0, 0], T_K); symmetry -> {"type": "symmetryPlane"} and
     empty -> {"type": "empty"} for U, p and T; a patch without a role -> BM-CASE. Write
     case_writer's system_files, constant_files and field_files; then case.json LAST:
     {"version": "cad-case/1", "case_writer_version", "status": "ok", "kind": "bm_planar", "law",
     "mesh": {"recipe_sha", "msh_sha256", "cells", "polymesh_sha256"}, "operating_point": {"fluid",
     "T_K", "p0_Pa", "nu_m2_s", "U_inlet_m_s", "U_exit_m_s", "H_i_m", "h_i_m", "h_e_m", "L_m"},
     "patches", "fields", "numerics", "sources", "cold_start": true, "files"} exactly as
     poiseuille fills fields/numerics/sources/files.
  B6 the exit layer. From post.load_mesh(case/constant/polyMesh): internal faces (index <
     n_internal) with fx_max - fx_min <= FLAT_TOL_M and |fx_min - x_exit| <= 1e-9, sorted by Cf_y
     ascending; x_exit = the "exit_plane" plane of geom_<law>/tags.json. BM-STATION when fewer
     than 3 faces or |sum |Sf_x| / (h_e dz) - 1| > 1e-6. A cell field f takes the face value
     f_o + w (f_n - f_o), w = post._x_weights(mesh)[face], o/n its owner and neighbour.
  B7 non-uniformity (core = faces with Cf_y <= core_frac * h_e). a = |Sf_x|, u = U_x face
     values; mean = sum(a u)/sum(a); nu_std = sqrt(sum(a (u - mean)^2)/sum(a)) / mean (Table 4's
     measure); nu_range = (max u - min u)/mean; nu_cl = max |u - u_cl|/u_cl, u_cl the face with
     the smallest Cf_y.
  B8 exit boundary layer. U_c = max u over ALL exit-layer faces; walk the faces from the wall
     (largest Cf_y) inward; points start with (eta 0, u 0), then (eta = h_e - Cf_y, u) per face,
     up to and including the first face with u >= U_c (1 - EDGE_REL). With r = u/U_c:
     theta_exit = trapezoid of r (1 - r) d eta, dstar_exit = trapezoid of (1 - r) d eta,
     H_exit = dstar/theta, Re_theta_exit = U_c theta / nu.
  B9 reversal (post._wall_block's rule). wall_nozzle faces ordered by owner C_x (ties by face
     index); a face is reversed iff its owner's U_x < 0; n_wall_cells, n_reversed, n_bands,
     bands_x_m ([first, last] owner C_x per run of reversed), separated_cfd = n_reversed > 0.
  B10 Thwaites on the CFD edge velocity (Bell & Mehta's own method). p0_core = sum over the
     inlet faces of a (p_f + |U_f|^2 / 2) / sum a, with post.face_values for U and p (a = |Sf|).
     For the wall_nozzle faces in B9 order: U_edge = sqrt(2 (p0_core - p_owner)); any negative
     argument -> the block is {"status": "refused", "reason_id": "BM-EDGE"}. The stations are
     s = 0 at the contraction start carrying the first wall face's U_edge (zero-order
     extrapolation), then every wall_nozzle face in B9 order at s_k = s_0 + the cumulative
     distance between consecutive face centres, s_0 the distance from (contraction_start, h_i)
     to the first face's (Cf_x, Cf_y); thwaites.solve runs on these n + 1 stations with
     theta0 0 (planar); a ValueError (or a non-finite theta/Re_theta row) -> refused
     "BM-EDGE". s_exit = numpy.interp(x_exit, [contraction_start] + Cf_x, s), and
     theta_thw_exit, Re_theta_thw_exit and x_sep_thw interpolate on the same n + 1 stations:
     theta_thw_exit = interp(s_exit, s, theta), Re_theta_thw_exit = interp(s_exit, s, Re_theta),
     separated_thw = res["separated"], x_sep_thw = interp(res["x_sep"], s,
     [contraction_start] + Cf_x) or None.
  B11 metrics(case_dir, time_name, geom_dir). Never raises: read U (3) and p (1) with
     post.read_field (post.Refused -> status refused BM-FIELD); p dimensions must be
     "[0 2 -2 0 0 0 0]" else BM-UNITS. Returns {"status", "reason_id", "detail", "time",
     "nu_std", "nu_range", "nu_cl", "n_core_faces", "theta_exit", "dstar_exit", "H_exit",
     "Re_theta_exit", "U_c", "reversal": {B9}, "separated_cfd", "thwaites": {B10 keys plus
     "status"}, "dp", "Cd"}; dp = area-mean p over inlet - area-mean p over outlet (face values,
     a = |Sf|); Cd = Q_in / (A_out sqrt(2 dp)) with Q_in = -post.patch_flux(mesh, U, "inlet"),
     A_out = sum |Sf_x| over the outlet; None when dp <= 0. history(case_dir, geom_dir,
     iters_list) = poiseuille.history's shape with the dp and Cd slots from metrics (solve's
     stop rule then applies to them).
  B12 compare(rows) and record(out_dir). rows = {law: metrics doc or None}. o1: values nu_std of
     poly3, poly5, poly7; "NOT_RUN" when any is missing, "AGREE" iff poly3 > poly5 > poly7
     strictly, else "DISAGREE"; the same rule on nu_range as o1_range (reported). o2_cfd and
     o2_thw: per run law {"cfd" or "thw": the flag, "table4": TABLE4 flag, "agree": equal};
     status "NOT_RUN" when no law has the flag, "DISAGREE" when any run law disagrees, "AGREE"
     when all four ran and agree, "PARTIAL" when every run law agrees but fewer than four ran.
     re_theta: per law {"cfd": Re_theta_exit, "thw": Re_theta_thw_exit, "table4", "cfd_rel":
     cfd/table4 - 1, "thw_rel"} and "order_cfd"/"order_thw": True iff poly3 < poly5 < poly7
     (None when missing), report only. record: {"version", "status": "REPORTED", "convention":
     B1, "recipe", "recipe_sha", "iters", "table4", "source", "binary" (first solve.json's
     binary with the basename of its path only), "runs": [per LAWS law {"law", "cells", "mesh"
     (build row), "solve": poiseuille's SOLVE_ROW_KEYS row or None, "wall_s", "gpu", "metrics"}],
     "all_steady", "o1", "o1_range", "o2_cfd", "o2_thw", "re_theta", "reduced": true,
     "reduced_note", "deferred"}. G5 is reported only: NO pass/fail verdict anywhere. No
     absolute path anywhere.

Usage:
  python bm_planar.py --selftest
  python bm_planar.py build OUT_DIR [LAW ...]
  python bm_planar.py run OUT_DIR LAW              (the GPU solve, supervisor only)
  python bm_planar.py metrics OUT_DIR LAW          (rewrite runs/<law>/metrics.json)
  python bm_planar.py record OUT_DIR RECORD_JSON
  python bm_planar.py gmsh-build GEOM_DIR MSH_PATH   (the fresh child of mesh_law)
"""
import json
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
import common
import case_writer
import wedge_mesh
import post
import solve
import export
import thwaites
import poiseuille

VERSION = "cad-g5/1"
CASE_KIND = "bm_planar"
LAWS = ("poly3", "poly5", "poly7", "cubic_matched")
RECIPE = {"version": "cad-bm-planar/1", "H_i_m": 2.74, "CR": 7.7, "L_over_Hi": 0.89, "Lu_over_Hi": 0.5,
          "Lx_over_He": 1.0, "x_m_cubic": 0.5, "t_wall_m": 0.003, "U_exit_m_s": 15.0, "dz_m": 0.01,
          "h1_m": 4e-5, "nr": 80, "n_axial": {"upstream": 40, "block": 16, "exit": 40}, "check_tau": 1e-4,
          "core_frac": 0.8}
ITERS = 2000                      # -iters of every solve: 3000 was set first, cut to 2000 (supervisor) after the first solve ran 4.5 it/s on 19200 cells, so one solve fits the 10-minute rule
TABLE4 = {"poly3": {"separation": False, "Re_theta": 359, "std": 0.0102},
          "poly5": {"separation": False, "Re_theta": 428, "std": 0.0040},
          "poly7": {"separation": True, "Re_theta": 478, "std": 0.0024},
          "cubic_matched": {"separation": True, "Re_theta": 484, "std": 0.0024}}
TABLE1 = (("A", 91.0, 38.0, 2, 1.20), ("B", 244.0, 137.0, 2, 0.89), ("C", 120.0, 120.0, 1, 1.0),
          ("D", 122.0, 114.0, 1, 1.07))   # (tunnel, L cm, listed inlet dimension cm, factor, listed L/H)
SOURCE = {"title": "Bell & Mehta 1988, Contraction Design for Small Low-Speed Wind Tunnels, NASA CR-177488",
          "url": "https://ntrs.nasa.gov/api/citations/19890004382/downloads/19890004382.pdf",
          "pdf_sha256": "f6093cebe6265814cc6606f03a54f8237947206b5765801b58710124104a130a",
          "table4_note": "Table 4 is their panel-method-plus-Thwaites prediction at CR 7.7, L/H 0.89, 15 m/s, not a measurement"}
PATCHES = ("inlet", "outlet", "wall_nozzle", "slip_upstream", "symmetry", "front", "back")
TYPE_ARGS = ["-type", "front=empty", "-type", "back=empty", "-type", "symmetry=symmetryPlane"]
WANT_TYPES = {"inlet": "patch", "outlet": "patch", "slip_upstream": "patch", "wall_nozzle": "wall",
              "symmetry": "symmetryPlane", "front": "empty", "back": "empty"}
ROLES = {"inlet": "velocity_inlet", "outlet": "pressure_outlet", "wall_nozzle": "wall", "slip_upstream": "slip",
         "symmetry": "symmetry", "front": "empty", "back": "empty"}
BIN_GPU = os.path.join(CAD, "bin_gpu.json")
REFUSAL_IDS = ("BM-OUT", "BM-NAME", "BM-GEOM", "BM-MESH", "BM-CASE", "BM-FIELD", "BM-UNITS", "BM-STATION")
EDGE_REL = 1e-12
FLAT_TOL_M = 1e-12
# derived numbers of B1 (h_i = H_i/2, h_e = h_i/7.7, L = 0.89 H_i, U_inlet = U_exit/7.7)
H_I = RECIPE["H_i_m"] / 2.0
H_E = H_I / RECIPE["CR"]
U_INLET = RECIPE["U_exit_m_s"] / RECIPE["CR"]
L_M = RECIPE["L_over_Hi"] * RECIPE["H_i_m"]
METRIC_KEYS = ("status", "reason_id", "detail", "time", "nu_std", "nu_range", "nu_cl", "n_core_faces",
               "theta_exit", "dstar_exit", "H_exit", "Re_theta_exit", "U_c", "reversal", "separated_cfd",
               "thwaites", "dp", "Cd")
THW_OK_KEYS = ("status", "theta_thw_exit", "Re_theta_thw_exit", "separated_thw", "x_sep_thw")
RECORD_KEYS = ("version", "status", "convention", "recipe", "recipe_sha", "iters", "table4", "source",
               "binary", "runs", "all_steady", "o1", "o1_range", "o2_cfd", "o2_thw", "re_theta", "reduced",
               "reduced_note", "deferred")
RUN_KEYS = ("law", "cells", "mesh", "solve", "wall_s", "gpu", "metrics")
SOLVE_ROW_KEYS = ("class", "reason_id", "failed", "criteria", "n_iter_lines", "log_sha256", "binary_sha256")
REDUCED_NOTE = "one mesh level and one 2000-iteration solve per law; no grid study"
DEFERRED = ["a second mesh level (nr 160, n_axial doubled) to bound nu_std's discretisation"]
P_DIMS = "[0 2 -2 0 0 0 0]"
USAGE = ("usage: python bm_planar.py --selftest" + chr(10)
         + "       python bm_planar.py build OUT_DIR [LAW ...]" + chr(10)
         + "       python bm_planar.py run OUT_DIR LAW" + chr(10)
         + "       python bm_planar.py metrics OUT_DIR LAW" + chr(10)
         + "       python bm_planar.py record OUT_DIR RECORD_JSON" + chr(10)
         + "       python bm_planar.py gmsh-build GEOM_DIR MSH_PATH")


class Refused(wedge_mesh.Refused):
    """A refusal by id (case_writer.Refused's shape): rule and detail."""


def convention():
    """B1: the H_i convention with the Table 1 evidence rows and the rejected reading."""
    evidence = []
    for tunnel, l_cm, listed_cm, factor, listed in TABLE1:
        computed = l_cm / (factor * listed_cm)
        evidence.append({"tunnel": tunnel, "L_cm": l_cm, "listed_cm": listed_cm, "factor": factor,
                         "listed": listed, "computed": computed,
                         "matches": round(computed, 2) == listed})
    return {"H_i": "full inlet height across the symmetry plane, H_i = 2 h_i", "evidence": evidence,
            "rejected": {"reading": "H_i = h_i", "B": 244.0 / 137.0,
                         "why": "Fig. 14 marks L/H_i 1.79 separated; tunnel B is attached"}}


def params_for(law):
    """B2: the template parameters of one wall law; BM-NAME for a law not in LAWS."""
    if law not in LAWS:
        raise Refused("BM-NAME", "law %r is not one of %r" % (law, list(LAWS)))
    return {"D_i": RECIPE["H_i_m"], "CR": RECIPE["CR"] ** 2, "L_over_Di": RECIPE["L_over_Hi"], "law": law,
            "x_m": RECIPE["x_m_cubic"] if law == "cubic_matched" else None,
            "Lx_over_De": RECIPE["Lx_over_He"], "Lu_over_Di": RECIPE["Lu_over_Hi"],
            "upstream_role": "slip", "t_wall": RECIPE["t_wall_m"]}


def _plane_x(geom_dir, name):
    """The x of one named plane of geom_dir/tags.json."""
    for p in common.read_json(os.path.join(geom_dir, "tags.json"))["planes"]:
        if p["name"] == name:
            return float(p["x"])
    raise wedge_mesh.Refused("BM-GEOM", "tags.json has no plane %r" % (name,))


def _order1(values):
    """One o1 row over the three values in poly3, poly5, poly7 order."""
    if any(v is None for v in values):
        return {"status": "NOT_RUN", "values": list(values)}
    ok = bool(values[0] > values[1] > values[2])
    return {"status": "AGREE" if ok else "DISAGREE", "values": list(values)}


def _o2(rows, key, row_key):
    """One o2 row: per run law the flag against TABLE4's separation."""
    laws = {}
    for law in LAWS:
        m = rows.get(law)
        if not isinstance(m, dict) or m.get("status") != "ok":
            continue
        flag = m.get(key) if key == "separated_cfd" else (m.get("thwaites") or {}).get(key)
        if flag is None:
            continue
        t4 = TABLE4[law]["separation"]
        laws[law] = {row_key: bool(flag), "table4": t4, "agree": bool(flag) == t4}
    ran = list(laws)
    if not ran:
        status = "NOT_RUN"
    elif any(not laws[l]["agree"] for l in ran):
        status = "DISAGREE"
    elif len(ran) == len(LAWS):
        status = "AGREE"
    else:
        status = "PARTIAL"
    return {"status": status, "laws": laws}


def compare(rows):
    """B12: o1, o1_range, o2_cfd, o2_thw and the report-only re_theta rows."""
    def val(law, key):
        m = rows.get(law)
        return m.get(key) if isinstance(m, dict) and m.get("status") == "ok" else None

    three = ("poly3", "poly5", "poly7")
    o1 = _order1([val(l, "nu_std") for l in three])
    o1_range = _order1([val(l, "nu_range") for l in three])
    per = {}
    for law in LAWS:
        m = rows.get(law) if isinstance(rows.get(law), dict) else {}
        thw = m.get("thwaites") if isinstance(m.get("thwaites"), dict) else {}
        cfd = m.get("Re_theta_exit") if m.get("status") == "ok" else None
        thw_re = thw.get("Re_theta_thw_exit") if thw.get("status") == "ok" else None
        t4 = TABLE4[law]["Re_theta"]
        per[law] = {"cfd": cfd, "thw": thw_re, "table4": t4,
                    "cfd_rel": None if cfd is None else cfd / t4 - 1.0,
                    "thw_rel": None if thw_re is None else thw_re / t4 - 1.0}

    def order(key):
        vals = [per[l][key] for l in three]
        if any(v is None for v in vals):
            return None
        return bool(vals[0] < vals[1] < vals[2])

    re_theta = dict(per)
    re_theta["order_cfd"] = order("cfd")
    re_theta["order_thw"] = order("thw")
    return {"o1": o1, "o1_range": o1_range, "o2_cfd": _o2(rows, "separated_cfd", "cfd"),
            "o2_thw": _o2(rows, "separated_thw", "thw"), "re_theta": re_theta}


def _n_axial():
    """The 12 per-block axial node counts: upstream, 10 contraction blocks, exit."""
    return ([RECIPE["n_axial"]["upstream"]] + [RECIPE["n_axial"]["block"]] * (wedge_mesh.RECIPE["k_stations"])
            + [RECIPE["n_axial"]["exit"]])


def _classify_surfaces(gmsh, planes, tol=wedge_mesh.TOL_GEOM):
    """B3's surface groups by bounding box, plus the internal station faces; BM-MESH unless the
    counts are exactly {inlet 1, outlet 1, slip_upstream 1, wall_nozzle 11, symmetry 12,
    front 12, back 12} and 11 internal."""
    groups = dict((g, []) for g in PATCHES)
    internal = []
    for _d, t in gmsh.model.getEntities(2):
        b = gmsh.model.getBoundingBox(2, t)
        if b[5] - b[2] < tol:
            groups["front" if abs(b[2]) < tol else "back"].append(t)
        elif b[3] - b[0] < tol:
            if abs(b[0] - planes["inlet"]) < tol:
                groups["inlet"].append(t)
            elif abs(b[0] - planes["outlet"]) < tol:
                groups["outlet"].append(t)
            else:
                internal.append(t)
        elif b[4] - b[1] < tol and abs(b[1]) < tol:
            groups["symmetry"].append(t)
        elif b[3] <= planes["contraction_start"] + tol:
            groups["slip_upstream"].append(t)
        else:
            groups["wall_nozzle"].append(t)
    want = {"inlet": 1, "outlet": 1, "slip_upstream": 1, "wall_nozzle": 11, "symmetry": 12,
            "front": 12, "back": 12}
    got = dict((g, len(v)) for g, v in groups.items())
    if got != want or len(internal) != 11:
        raise wedge_mesh.Refused("BM-MESH", "surface groups %r and %d internal, want %r and 11"
                                 % (got, len(internal), want))
    return groups


def _set_transfinite(gmsh, surfs, n_of_block, tol=wedge_mesh.TOL_GEOM):
    """B3's transfinite settings on the SOURCE block surfaces: vertical curves (x-extent below
    tol) get nr + 1 Progression-graded nodes with the first cell h1 at the wall end, every other
    curve n + 1 uniform nodes of its block, then setTransfiniteSurface and setRecombine."""
    nr = RECIPE["nr"]
    done = set()
    for i, s in enumerate(surfs):
        for c in gmsh.model.getBoundary([(2, s)], oriented=False):
            c = abs(c[1])
            if c in done:
                continue
            done.add(c)
            b = gmsh.model.getBoundingBox(1, c)
            if b[3] - b[0] < tol:
                e = wedge_mesh._curve_ends(gmsh, c)
                q = wedge_mesh.solve_q(RECIPE["h1_m"], abs(e[1][1] - e[0][1]), nr)
                gmsh.model.mesh.setTransfiniteCurve(c, nr + 1, "Progression",
                                                    q if e[0][1] > e[1][1] else 1.0 / q)
            else:
                gmsh.model.mesh.setTransfiniteCurve(c, n_of_block[i] + 1)
        gmsh.model.mesh.setTransfiniteSurface(s)
        gmsh.model.mesh.setRecombine(2, s)


def gmsh_build(geom_dir, msh_path):
    """B3, the gmsh-build child's whole job: meridian.step -> 12 transfinite blocks -> extrude by
    (0, 0, dz) with one recombined layer -> the seven patch groups and the fluid volume -> MSH 4.1
    ASCII at msh_path. Runs only in a fresh process; returns the build dict; a Refused exits 3."""
    import gmsh
    planes = dict((p["name"], float(p["x"]))
                  for p in common.read_json(os.path.join(geom_dir, "tags.json"))["planes"])
    edges = wedge_mesh.stations(planes)
    n_of_block = _n_axial()
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.option.setString("Geometry.OCCTargetUnit", "M")
        surfs, area, _ybar = wedge_mesh._split_meridian(gmsh, geom_dir, edges)
        gmsh.model.occ.extrude([(2, s) for s in surfs], 0.0, 0.0, RECIPE["dz_m"],
                               numElements=[1], recombine=True)
        gmsh.model.occ.synchronize()
        _set_transfinite(gmsh, surfs, n_of_block)
        groups = _classify_surfaces(gmsh, planes)
        for name in PATCHES:
            gmsh.model.addPhysicalGroup(2, sorted(groups[name]), name=name)
        gmsh.model.addPhysicalGroup(3, sorted(t for _d, t in gmsh.model.getEntities(3)), name="fluid")
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.model.mesh.generate(3)
        types, etags, _n = gmsh.model.mesh.getElements(3)
        elements = dict((wedge_mesh.ELEMENT_NAMES.get(int(t), "type%d" % int(t)), len(g))
                        for t, g in zip(types, etags))
        if sorted(elements) != ["hex"]:
            raise wedge_mesh.Refused("BM-MESH", "volume elements %r, want hex only" % (elements,))
        os.makedirs(os.path.dirname(os.path.abspath(msh_path)), exist_ok=True)
        gmsh.write(msh_path)
        version = gmsh.GMSH_API_VERSION
    finally:
        gmsh.finalize()
    return {"cells": sum(elements.values()), "nr": RECIPE["nr"], "n_axial": n_of_block,
            "meridian_area_m2": area, "gmsh": version}


def build_child(geom_dir, msh_path):
    """One gmsh-build child (a fresh process per build, wedge_mesh.build_level's protocol); the
    build dict from its last stdout line, wedge_mesh.Refused on the child's REFUSED_EXIT."""
    pr = subprocess.run([sys.executable, os.path.abspath(__file__), "gmsh-build", geom_dir, msh_path],
                        capture_output=True, text=True, encoding="utf-8", timeout=wedge_mesh.CHILD_TIMEOUT_S,
                        env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    lines = [s for s in pr.stdout.splitlines() if s.strip()]
    line = json.loads(lines[-1]) if lines else {}
    if pr.returncode == 0:
        return line
    if pr.returncode == wedge_mesh.REFUSED_EXIT:
        raise wedge_mesh.Refused(line["refused"], line["detail"])
    raise RuntimeError("gmsh-build exited %d: %s" % (pr.returncode, pr.stderr[-600:]))


def _exit_layer(mesh, x_exit):
    """B6: the exit plane's internal faces, sorted by Cf_y ascending."""
    n = mesh["n_internal"]
    flat = (mesh["fx_max"][:n] - mesh["fx_min"][:n]) <= FLAT_TOL_M
    at = np.abs(mesh["fx_min"][:n] - x_exit) <= 1e-9
    faces = np.nonzero(flat & at)[0]
    return faces[np.argsort(mesh["Cf"][faces, 1], kind="stable")]


def _exit_faces_checked(mesh, x_exit):
    """_exit_layer with B6's two checks (BM-STATION): at least 3 faces and the |Sf_x| sum against h_e dz."""
    faces = _exit_layer(mesh, x_exit)
    if len(faces) < 3:
        raise wedge_mesh.Refused("BM-STATION", "%d internal faces at x_exit %r, want >= 3" % (len(faces), x_exit))
    sum_a = float(np.sum(np.abs(mesh["Sf"][faces, 0])))
    rel = abs(sum_a / (H_E * RECIPE["dz_m"]) - 1.0)
    if rel > 1e-6:
        raise wedge_mesh.Refused("BM-STATION", "the exit faces' |Sf_x| sum %r is %.3e off h_e dz %r"
                                 % (sum_a, rel, H_E * RECIPE["dz_m"]))
    return faces


def mesh_law(geom_dir, mesh_dir, bins):
    """B4: one law's planar mesh - the fresh gmsh-build child into mesh_dir/planar.msh, the
    converter into mesh_dir/case, the seven patch types, the -check at RECIPE's tau, the volume
    against meridian_area dz and the first cell at the exit plane; the B4 row."""
    os.makedirs(mesh_dir, exist_ok=True)
    msh_path = os.path.join(mesh_dir, "planar.msh")
    b = build_child(geom_dir, msh_path)
    case_dir = os.path.join(mesh_dir, "case")
    wedge_mesh.convert(bins, msh_path, case_dir, TYPE_ARGS)
    types = wedge_mesh.patch_types(case_dir)
    if types != WANT_TYPES:
        raise wedge_mesh.Refused("BM-MESH", "boundary is %r, want %r" % (types, WANT_TYPES))
    chk = wedge_mesh.run_check(bins, case_dir, os.path.join(mesh_dir, "check.json"), RECIPE["check_tau"])
    if chk["exit"] != 0:
        raise wedge_mesh.Refused("BM-MESH", "-check exited %d with gate %r" % (chk["exit"], chk["gate"]))
    mesh = post.load_mesh(os.path.join(case_dir, "constant", "polyMesh"))
    volume_m3 = float(np.sum(mesh["V"]))
    volume_ref_m3 = float(b["meridian_area_m2"]) * RECIPE["dz_m"]
    volume_rel = volume_m3 / volume_ref_m3 - 1.0
    if abs(volume_rel) > 3e-3:
        raise wedge_mesh.Refused("BM-MESH", "volume_rel %r is outside +-3e-3" % volume_rel)
    faces = _exit_faces_checked(mesh, _plane_x(geom_dir, "exit_plane"))
    f = int(faces[int(np.argmax(mesh["Cf"][faces, 1]))])
    h1_exit_m = H_E - float(np.min(mesh["points"][mesh["faces"][f], 1]))
    h1_rel = h1_exit_m / RECIPE["h1_m"] - 1.0
    if abs(h1_rel) > 0.05:
        raise wedge_mesh.Refused("BM-MESH", "h1_rel %r is outside +-0.05" % h1_rel)
    return {"cells": int(mesh["n_cells"]),
            "patch_faces": dict((n, mesh["patch_range"][n][1]) for n in PATCHES), "types": types,
            "volume_m3": volume_m3, "volume_ref_m3": volume_ref_m3, "volume_rel": volume_rel,
            "h1_exit_m": h1_exit_m, "h1_rel": h1_rel, "tau_min": chk["tau_min"],
            "non_orth_max_deg": chk["non_orth_max_deg"], "msh_sha256": common.sha256_file(msh_path)}


def write_case(mesh_dir, law, out_dir):
    """B5: one cold cad-case/1 case in out_dir (refused BM-OUT when it exists): the five polyMesh
    files copied byte for byte, case_writer's system, constant and field files of the boundary's
    own rows (symmetry/empty typed explicitly), then case.json LAST."""
    if os.path.exists(out_dir):
        raise Refused("BM-OUT", "%s exists" % out_dir)
    pm_dir = os.path.join(mesh_dir, "case", "constant", "polyMesh")
    mesh = post.load_mesh(pm_dir)
    missing = [n for n in PATCHES if n not in mesh["patch_range"]]
    if missing:
        raise wedge_mesh.Refused("BM-MESH", "boundary lacks %r" % (missing,))
    os.makedirs(os.path.join(out_dir, "constant", "polyMesh"))
    files = {}
    for nm in case_writer.POLYMESH_FILES:
        with open(os.path.join(pm_dir, nm), "rb") as f:
            blob = f.read()
        with open(os.path.join(out_dir, "constant", "polyMesh", nm), "wb") as f:
            f.write(blob)
        files["constant/polyMesh/" + nm] = common.sha256_bytes(blob)
    rows = []
    for p in mesh["patches"]:
        nm = p["name"]
        role = ROLES.get(nm)
        if role is None:
            raise Refused("BM-CASE", "patch %s has no G5 role" % nm)
        if role == "symmetry":
            bcs = dict((f, {"type": "symmetryPlane"}) for f in ("U", "p", "T"))
        elif role == "empty":
            bcs = dict((f, {"type": "empty"}) for f in ("U", "p", "T"))
        else:
            bcs = case_writer.bc_table(role, [U_INLET, 0.0, 0.0], case_writer.STATE["T_K"])
        rows.append({"name": nm, "type": p["type"], "n_faces": p["nFaces"], "start_face": p["startFace"],
                     "role": role, "U": bcs["U"], "p": bcs["p"], "T": bcs["T"]})

    def put(rel, text):
        blob = text.encode("utf-8")
        target = os.path.join(out_dir, rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "wb") as f:
            f.write(blob)
        files[rel] = common.sha256_bytes(blob)

    for rel, text in case_writer.system_files().items():
        put(rel, text)
    for rel, text in case_writer.constant_files().items():
        put(rel, text)
    for rel, text in sorted(case_writer.field_files(rows, case_writer.STATE["T_K"]).items()):
        put(rel, text)
    msh_path = os.path.join(mesh_dir, "planar.msh")
    case = {"version": "cad-case/1", "case_writer_version": case_writer.CASE_WRITER_VERSION, "status": "ok",
            "kind": CASE_KIND, "law": law,
            "mesh": {"recipe_sha": common.sha256_of(RECIPE), "msh_sha256": common.sha256_file(msh_path),
                     "cells": mesh["n_cells"],
                     "polymesh_sha256": dict((nm, files["constant/polyMesh/" + nm])
                                             for nm in case_writer.POLYMESH_FILES)},
            "operating_point": {"fluid": case_writer.STATE["fluid"], "T_K": case_writer.STATE["T_K"],
                                "p0_Pa": case_writer.STATE["p0_Pa"], "nu_m2_s": case_writer.NU_AIR,
                                "U_inlet_m_s": U_INLET, "U_exit_m_s": RECIPE["U_exit_m_s"],
                                "H_i_m": RECIPE["H_i_m"], "h_i_m": H_I, "h_e_m": H_E, "L_m": L_M},
            "patches": rows,
            "fields": {"U": {"dimensions": "[0 1 -1 0 0 0 0]", "internal": [0.0, 0.0, 0.0]},
                       "p": {"dimensions": P_DIMS, "internal": 0.0},
                       "T": {"dimensions": "[0 0 0 1 0 0 0]", "internal": case_writer.STATE["T_K"]}},
            "numerics": {"transcribed": [list(r) for r in case_writer.TRANSCRIBED]
                         + [list(r) for r in case_writer.TRANSCRIBED_LAMINAR],
                         "differences": list(case_writer.DIFFERENCES)},
            "sources": [{"id": s["id"], "file": s["file"], "commit": s["commit"], "blob": s["blob"],
                         "lines": s["lines"], "text_sha256": s["text_sha256"]}
                        for s in common.read_json(case_writer.SOURCES)["sources"]],
            "cold_start": True, "files": files}
    with open(os.path.join(out_dir, "case.json"), "wb") as f:
        f.write((common.canonical_json(case) + chr(10)).encode("utf-8"))
    return case


def _num(v):
    """A check value for the build print: %.6g, or nan when missing."""
    return "%.6g" % v if isinstance(v, float) else "nan"


def build(out_dir, laws=LAWS):
    """Per law the geometry (B2), the planar mesh (B4) and the cold case (B5), then build.json
    with no absolute path; BM-OUT unless out_dir is missing or empty, BM-NAME for a bad law."""
    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise Refused("BM-OUT", "%s exists and is not empty" % out_dir)
    for law in laws:
        if law not in LAWS:
            raise Refused("BM-NAME", "law %r is not one of %r" % (law, list(LAWS)))
    os.makedirs(out_dir, exist_ok=True)
    bins = wedge_mesh.load_bins()
    meshes, cases = {}, {}
    for law in laws:
        r = export.run_pipeline(export.TEMPLATE, params_for(law), os.path.join(out_dir, "geom_" + law))
        if r["status"] != "ok":
            raise Refused("BM-GEOM", "%s: %s" % (r["rule"], r["message"]))
        row = mesh_law(os.path.join(out_dir, "geom_" + law), os.path.join(out_dir, "mesh_" + law), bins)
        meshes[law] = row
        write_case(os.path.join(out_dir, "mesh_" + law), law, os.path.join(out_dir, "cases", law))
        cj = common.read_json(os.path.join(out_dir, "cases", law, "case.json"))
        cases[law] = {"cells": cj["mesh"]["cells"], "U_inlet_m_s": cj["operating_point"]["U_inlet_m_s"]}
        print("[bm] %s cells %d tau %s non-orth %s deg volume_rel %+.3e h1_rel %+.3e"
              % (law, row["cells"], _num(row["tau_min"]), _num(row["non_orth_max_deg"]),
                 row["volume_rel"], row["h1_rel"]))
    doc = {"version": VERSION, "laws": list(laws), "recipe_sha": common.sha256_of(RECIPE),
           "convention": convention(), "meshes": meshes, "cases": cases}
    with open(os.path.join(out_dir, "build.json"), "wb") as f:
        f.write((common.canonical_json(doc) + chr(10)).encode("utf-8"))
    return doc


def _laws_of(rest):
    """A build call's laws (F2): all of LAWS when no LAW argument is given, else the given ones
    in order (build itself refuses a law not in LAWS)."""
    return LAWS if not rest else tuple(rest)


_MESH_CACHE = {}


def _mesh(case_dir):
    """post.load_mesh of the case's polyMesh, cached per directory in this process."""
    pm = os.path.abspath(os.path.join(case_dir, "constant", "polyMesh"))
    if pm not in _MESH_CACHE:
        _MESH_CACHE[pm] = post.load_mesh(pm)
    return _MESH_CACHE[pm]


def _nonuniform(name, internal, tag):
    """The internalField replacement of _fx_plant: a nonuniform List<tag> block."""
    rows = np.asarray(internal)
    if rows.ndim == 2 and rows.shape[1] == 1:
        rows = rows[:, 0]
    body = ["internalField   nonuniform List<%s> %d" % (tag, len(rows)), "("]
    body += [post._fx_num(r) for r in rows]
    return chr(10).join(body + [")", ";"])


def _fx_plant(case_dir, time_name, u_x, p_x):
    """Plant U and p at <case_dir>/<time_name>: the internal fields from the two callables over
    the cell centres, every 0/ boundary entry kept verbatim."""
    mesh = _mesh(case_dir)
    cent = mesh["C"]
    u_int = np.zeros((mesh["n_cells"], 3))
    u_int[:, 0] = np.asarray(u_x(cent), dtype=float)
    p_int = np.asarray(p_x(cent), dtype=float)
    tdir = os.path.join(case_dir, time_name)
    os.makedirs(tdir, exist_ok=True)
    for name, internal, tag in (("U", u_int, "vector"), ("p", p_int, "scalar")):
        with open(os.path.join(case_dir, "0", name), "r", encoding="utf-8") as f:
            text = f.read()
        head, tail = text.split("boundaryField", 1)
        i0 = head.index("internalField")
        i1 = head.index(";", i0)
        head = head[:i0] + _nonuniform(name, internal, tag) + head[i1 + 1:]
        with open(os.path.join(tdir, name), "wb") as f:
            f.write((head + "boundaryField" + tail).encode("utf-8"))


def _metrics_ok(mesh, u_field, p_field, planes, time_name):
    """B6-B11's body after the fields are read: B6 the exit layer, B7 the core non-uniformity,
    B8 the exit boundary layer, B9 the reversal, B10 Thwaites on the CFD edge, then dp and Cd."""
    n_int = mesh["n_internal"]
    w = post._x_weights(mesh)
    o, nb = mesh["owner"][:n_int], mesh["neighbour"]
    u_int, p_int = u_field["internal"], p_field["internal"]
    faces = _exit_faces_checked(mesh, planes["exit_plane"])
    u_f = u_int[o[faces]] + w[faces][:, None] * (u_int[nb[faces]] - u_int[o[faces]])
    p_f = p_int[o[faces]] + w[faces] * (p_int[nb[faces]] - p_int[o[faces]])
    cfy = mesh["Cf"][faces, 1]
    a = np.abs(mesh["Sf"][faces, 0])
    core = cfy <= RECIPE["core_frac"] * H_E
    ac, uc = a[core], u_f[core, 0]
    mean = float(np.sum(ac * uc)) / float(np.sum(ac))
    if not mean > 0.0:
        raise Refused("BM-FIELD", "the exit core's area-mean u_x is %r" % mean)
    nu_std = float(np.sqrt(float(np.sum(ac * (uc - mean) ** 2)) / float(np.sum(ac)))) / mean
    nu_range = (float(np.max(uc)) - float(np.min(uc))) / mean
    u_cl = float(uc[int(np.argmin(cfy[core]))])
    nu_cl = None if not u_cl > 0.0 else float(np.max(np.abs(uc - u_cl))) / u_cl
    u_c = float(np.max(u_f[:, 0]))
    if not u_c > 0.0:
        raise Refused("BM-FIELD", "the exit plane's max u_x is %r" % u_c)
    eta, uu = [0.0], [0.0]
    for k in range(len(faces) - 1, -1, -1):                     # from the wall (largest Cf_y) inward
        eta.append(H_E - float(cfy[k]))
        uu.append(float(u_f[k, 0]))
        if u_f[k, 0] >= u_c * (1.0 - EDGE_REL):
            break
    eta_a = np.array(eta)
    r = np.array(uu) / u_c
    de = np.diff(eta_a)

    def trapz(g):
        """The trapezoid of g over eta_a (numpy 1/2 agnostic, no np.trapz deprecation)."""
        return float(np.sum(0.5 * (g[1:] + g[:-1]) * de))

    theta_exit = trapz(r * (1.0 - r))
    dstar_exit = trapz(1.0 - r)
    h_exit = None if not theta_exit > 0.0 else dstar_exit / theta_exit
    re_theta_exit = u_c * theta_exit / case_writer.NU_AIR
    wst, wnf, _t = mesh["patch_range"]["wall_nozzle"]
    wf = np.arange(wst, wst + wnf, dtype=np.int64)
    oc = mesh["owner"][wf]
    order = np.lexsort((wf, mesh["C"][oc, 0]))
    wall_f, cells = wf[order], oc[order]
    rev = u_int[cells, 0] < 0.0
    n_rev = int(np.count_nonzero(rev))
    xs_c = mesh["C"][cells, 0]
    bands = []
    i = 0
    while i < len(rev):
        if rev[i]:
            j = i
            while j < len(rev) and rev[j]:
                j += 1
            bands.append([float(xs_c[i]), float(xs_c[j - 1])])
            i = j
        else:
            i += 1
    reversal = {"n_wall_cells": len(cells), "n_reversed": n_rev, "n_bands": len(bands), "bands_x_m": bands}
    thw = _thwaites_block(mesh, u_field, p_field, planes, cells, wall_f)

    def area_mean_p(name):
        st, nf_, _t2 = mesh["patch_range"][name]
        sf = mesh["Sf"][st:st + nf_]
        af = np.sqrt(np.einsum("ij,ij->i", sf, sf))
        pf = np.asarray(post.face_values(mesh, p_field, name)).reshape(-1)
        return float(np.sum(af * pf)) / float(np.sum(af))

    dp = area_mean_p("inlet") - area_mean_p("outlet")
    q_in = -post.patch_flux(mesh, u_field, "inlet")
    st_o, nf_o, _t2 = mesh["patch_range"]["outlet"]
    a_out = float(np.sum(np.abs(mesh["Sf"][st_o:st_o + nf_o, 0])))
    cd = None if not dp > 0.0 else q_in / (a_out * math.sqrt(2.0 * dp))
    return {"status": "ok", "reason_id": None, "detail": "", "time": time_name, "nu_std": nu_std,
            "nu_range": nu_range, "nu_cl": nu_cl, "n_core_faces": int(np.count_nonzero(core)),
            "theta_exit": theta_exit, "dstar_exit": dstar_exit, "H_exit": h_exit,
            "Re_theta_exit": re_theta_exit, "U_c": u_c, "reversal": reversal,
            "separated_cfd": n_rev > 0, "thwaites": thw, "dp": dp, "Cd": cd}


def _thwaites_block(mesh, u_field, p_field, planes, cells, wall_f):
    """B10: p0_core over the inlet faces, the CFD edge velocity on the wall_nozzle owners in B9
    order, the planar Thwaites march on the n + 1 stations s (s = 0 at the contraction start,
    theta0 0), and the exit-plane interpolations on the same stations."""
    refused = {"status": "refused", "reason_id": "BM-EDGE"}
    st_in, nf_in, _t = mesh["patch_range"]["inlet"]
    sf = mesh["Sf"][st_in:st_in + nf_in]
    a = np.sqrt(np.einsum("ij,ij->i", sf, sf))
    u_in = post.face_values(mesh, u_field, "inlet")
    p_in = np.asarray(post.face_values(mesh, p_field, "inlet")).reshape(-1)
    p0_core = float(np.sum(a * (p_in + 0.5 * np.einsum("ij,ij->i", u_in, u_in)))) / float(np.sum(a))
    arg = 2.0 * (p0_core - p_field["internal"][cells])
    if bool(np.any(arg < 0.0)):
        return refused
    cfx, cfy_w = mesh["Cf"][wall_f, 0], mesh["Cf"][wall_f, 1]
    try:
        u_edge = np.sqrt(arg)
        s_0 = math.hypot(float(cfx[0]) - planes["contraction_start"], float(cfy_w[0]) - H_I)
        s = np.concatenate(([0.0, s_0], s_0 + np.cumsum(np.hypot(np.diff(cfx), np.diff(cfy_w)))))
        res = thwaites.solve(s, np.concatenate(([u_edge[0]], u_edge)), case_writer.NU_AIR)
        if not bool(np.all(np.isfinite(res["theta"]))) or not bool(np.all(np.isfinite(res["Re_theta"]))):
            return refused
        x_stat = np.concatenate(([planes["contraction_start"]], cfx))
        s_exit = float(np.interp(planes["exit_plane"], x_stat, s))
        x_sep = None
        if res["x_sep"] is not None:
            v = float(np.interp(res["x_sep"], s, x_stat))
            x_sep = v if math.isfinite(v) else None
        return {"status": "ok", "theta_thw_exit": float(np.interp(s_exit, s, res["theta"])),
                "Re_theta_thw_exit": float(np.interp(s_exit, s, res["Re_theta"])),
                "separated_thw": bool(res["separated"]), "x_sep_thw": x_sep}
    except ValueError:
        return refused


def metrics(case_dir, time_name, geom_dir):
    """B11 into METRIC_KEYS; never raises on a bad case - a refusal comes back as status
    "refused" with its reason_id, the detail, the time and None elsewhere."""
    tdir = os.path.join(case_dir, time_name)

    def _ref(rid, detail):
        row = dict((k, None) for k in METRIC_KEYS)
        row["status"], row["reason_id"], row["detail"], row["time"] = "refused", rid, detail, time_name
        return row

    try:
        mesh = _mesh(case_dir)
    except Exception as e:                                   # polyMesh unreadable at all
        return _ref("BM-MESH", "%s" % e)
    ranges = mesh["patch_range"]
    missing = [n for n in PATCHES if n not in ranges]
    if missing:
        return _ref("BM-MESH", "boundary lacks %r" % (missing,))
    try:
        planes = dict((p["name"], float(p["x"]))
                      for p in common.read_json(os.path.join(geom_dir, "tags.json"))["planes"])
        u_field = post.read_field(os.path.join(tdir, "U"), 3, mesh["n_cells"],
                                  dict((n, ranges[n][1]) for n in PATCHES))
        p_field = post.read_field(os.path.join(tdir, "p"), 1, mesh["n_cells"],
                                  dict((n, ranges[n][1]) for n in PATCHES))
    except (post.Refused, OSError, ValueError) as e:
        return _ref("BM-FIELD", "%s" % e)
    if p_field["dimensions"] != P_DIMS:
        return _ref("BM-UNITS", "p carries %s, want %s" % (p_field["dimensions"], P_DIMS))
    try:
        return _metrics_ok(mesh, u_field, p_field, planes, time_name)
    except wedge_mesh.Refused as r:
        return _ref(r.rule, r.detail)


def history(case_dir, geom_dir, iters_list):
    """One row per time of the window, the metrics doc behind it: the dp and Cd slots carry B11's
    dp and Cd, so solve.launch's stop rule applies to them; never raises."""
    rows = []
    for it in iters_list:
        m = metrics(case_dir, str(it), geom_dir)
        rows.append({"iter": it, "time": str(it), "status": m["status"], "reason_id": m["reason_id"],
                     "dp": m["dp"], "Cd": m["Cd"], "post_sha256": common.sha256_of(m)})
    return rows


def run(out_dir, law, exe=None, visible=True, snapshot_fn=None):
    """One solve of cases/<law> into runs/<law> through solve.launch (the D-1 binary of bin_gpu
    when exe is None; BM-NAME for a bad law, BM-OUT when the run directory exists); timed, the
    solve doc returned, wall.json and gpu.json written exactly like poiseuille.run, then
    runs/<law>/metrics.json when cases/<law>/<ITERS> exists."""
    if law not in LAWS:
        raise Refused("BM-NAME", "law %r is not one of %r" % (law, list(LAWS)))
    rdir = os.path.join(out_dir, "runs", law)
    if os.path.exists(rdir):
        raise Refused("BM-OUT", "%s exists" % rdir)
    geom_dir = os.path.join(out_dir, "geom_" + law)
    if snapshot_fn is None:
        snapshot_fn = poiseuille.gpu_snapshot
    print("[bm] solving %s for %d iterations..." % (law, ITERS))
    before = snapshot_fn()
    t0 = time.monotonic()
    doc = solve.launch(os.path.join(out_dir, "cases", law), geom_dir, rdir, ITERS,
                       exe=exe, bin_json=None if exe else BIN_GPU, history_fn=history, visible=visible)
    wall_s = time.monotonic() - t0
    after = snapshot_fn()
    os.makedirs(rdir, exist_ok=True)
    with open(os.path.join(rdir, "wall.json"), "wb") as f:
        f.write((common.canonical_json({"wall_s": wall_s}) + chr(10)).encode("utf-8"))
    gpu_doc = {"before": before, "after": after, "shared": poiseuille.shared_flag(before, after),
               "note": poiseuille.GPU_NOTE}
    with open(os.path.join(rdir, "gpu.json"), "wb") as f:
        f.write((common.canonical_json(gpu_doc) + chr(10)).encode("utf-8"))
    if os.path.isdir(os.path.join(out_dir, "cases", law, str(ITERS))):
        m = metrics(os.path.join(out_dir, "cases", law), str(ITERS), geom_dir)
        _write_metrics(rdir, m)
    print("[bm] %s class %s in %.1f s" % (law, doc.get("class"), wall_s))
    print("[bm] %s gpu shared %s" % (law, gpu_doc["shared"]))
    return doc


def _write_metrics(rdir, m):
    """run's metrics.json write, shared with the metrics sub-command: canonical JSON + newline."""
    with open(os.path.join(rdir, "metrics.json"), "wb") as f:
        f.write((common.canonical_json(m) + chr(10)).encode("utf-8"))


def metrics_rewrite(out_dir, law):
    """The metrics sub-command: recompute runs/<law>/metrics.json from cases/<law>/<ITERS> with
    the same write run made; BM-NAME for a law not in LAWS, BM-OUT when runs/<law>/solve.json
    or the cases/<law>/<ITERS> directory is missing; prints the one-line summary."""
    if law not in LAWS:
        raise Refused("BM-NAME", "law %r is not one of %r" % (law, list(LAWS)))
    rdir = os.path.join(out_dir, "runs", law)
    sj = os.path.join(rdir, "solve.json")
    if not os.path.isfile(sj):
        raise Refused("BM-OUT", "%s is missing" % sj)
    tdir = os.path.join(out_dir, "cases", law, str(ITERS))
    if not os.path.isdir(tdir):
        raise Refused("BM-OUT", "%s is missing" % tdir)
    m = metrics(os.path.join(out_dir, "cases", law), str(ITERS), os.path.join(out_dir, "geom_" + law))
    _write_metrics(rdir, m)
    thw = m.get("thwaites") if isinstance(m.get("thwaites"), dict) else {}
    print("[bm] %s metrics %s nu_std %s theta_thw_exit %s"
          % (law, m["status"], m["nu_std"], thw.get("theta_thw_exit")))
    return m


def record(out_dir):
    """B12: the cad-g5/1 record over the four laws, REPORTED, no verdict, no absolute path."""
    build_doc = common.read_json(os.path.join(out_dir, "build.json"))
    meshes = build_doc.get("meshes") or {}
    solves, rows, runs = {}, {}, []
    for law in LAWS:
        sj = os.path.join(out_dir, "runs", law, "solve.json")
        solves[law] = common.read_json(sj) if os.path.isfile(sj) else None
        mj = os.path.join(out_dir, "runs", law, "metrics.json")
        rows[law] = common.read_json(mj) if os.path.isfile(mj) else None
    binary = None
    for law in LAWS:
        s = solves[law]
        if s and s.get("binary"):
            binary = dict(s["binary"])
            if os.path.isabs(binary.get("path", "")):
                binary["path"] = os.path.basename(binary["path"])   # no absolute path in the record
            break
    for law in LAWS:
        s, mesh_row = solves[law], meshes.get(law)
        solve_row, wall_s = None, None
        if s is not None:
            res = s.get("result") or {}
            solve_row = {"class": s.get("class"), "reason_id": s.get("reason_id"), "failed": res.get("failed"),
                         "criteria": res.get("criteria"), "n_iter_lines": len(res.get("iterations") or []),
                         "log_sha256": (s.get("log") or {}).get("sha256"),
                         "binary_sha256": (s.get("binary") or {}).get("sha256")}
            wj = os.path.join(out_dir, "runs", law, "wall.json")
            wall_s = common.read_json(wj).get("wall_s") if os.path.isfile(wj) else None
        gj = os.path.join(out_dir, "runs", law, "gpu.json")
        gpu = common.read_json(gj) if os.path.isfile(gj) else None
        runs.append({"law": law, "cells": None if mesh_row is None else mesh_row["cells"],
                     "mesh": mesh_row, "solve": solve_row, "wall_s": wall_s, "gpu": gpu,
                     "metrics": rows[law]})
    classes = [s.get("class") for s in solves.values() if s is not None]
    cmp_rows = compare(rows)
    return {"version": VERSION, "status": "REPORTED", "convention": convention(), "recipe": RECIPE,
            "recipe_sha": common.sha256_of(RECIPE), "iters": ITERS, "table4": TABLE4, "source": SOURCE,
            "binary": binary, "runs": runs, "all_steady": all(c == "steady" for c in classes),
            "o1": cmp_rows["o1"], "o1_range": cmp_rows["o1_range"], "o2_cfd": cmp_rows["o2_cfd"],
            "o2_thw": cmp_rows["o2_thw"], "re_theta": cmp_rows["re_theta"], "reduced": True,
            "reduced_note": REDUCED_NOTE, "deferred": list(DEFERRED)}


def main(argv):
    """--selftest | build OUT_DIR [LAW ...] | run OUT_DIR LAW | metrics OUT_DIR LAW | record
    OUT_DIR RECORD_JSON | gmsh-build GEOM_DIR MSH_PATH (nozzle_nominal.main's style)."""
    if argv == ["--selftest"]:
        try:
            selftest()
        except Exception:
            traceback.print_exc()
            return 1
        return 0
    if len(argv) == 3 and argv[0] == "gmsh-build":
        try:
            build_dict = gmsh_build(argv[1], argv[2])
        except wedge_mesh.Refused as r:
            print(common.canonical_json({"refused": r.rule, "detail": r.detail}))
            return wedge_mesh.REFUSED_EXIT
        print(common.canonical_json(build_dict))
        return 0
    if len(argv) >= 2 and argv[0] == "build":
        try:
            doc = build(os.path.abspath(argv[1]), _laws_of(argv[2:]))
        except wedge_mesh.Refused as r:
            sys.stderr.write("refused %s: %s%s" % (r.rule, r.detail, chr(10)))
            return 2
        return 0
    if len(argv) == 3 and argv[0] == "run":
        try:
            doc = run(os.path.abspath(argv[1]), argv[2])
        except wedge_mesh.Refused as r:
            sys.stderr.write("refused %s: %s%s" % (r.rule, r.detail, chr(10)))
            return 2
        print("class %s reason_id %s" % (doc.get("class"), doc.get("reason_id")))
        return 0 if doc.get("class") == "steady" else 1
    if len(argv) == 3 and argv[0] == "metrics":
        try:
            metrics_rewrite(os.path.abspath(argv[1]), argv[2])
        except wedge_mesh.Refused as r:
            sys.stderr.write("refused %s: %s%s" % (r.rule, r.detail, chr(10)))
            return 2
        return 0
    if len(argv) == 3 and argv[0] == "record":
        try:
            rec = record(os.path.abspath(argv[1]))
        except wedge_mesh.Refused as r:
            sys.stderr.write("refused %s: %s%s" % (r.rule, r.detail, chr(10)))
            return 2
        with open(os.path.abspath(argv[2]), "wb") as f:
            f.write((common.canonical_json(rec) + chr(10)).encode("utf-8"))
        print("o1 %s %s" % (rec["o1"]["status"], rec["o1"]["values"]))
        print("o2_cfd %s" % rec["o2_cfd"]["status"])
        print("o2_thw %s" % rec["o2_thw"]["status"])
        for r in rec["runs"]:
            m = r["metrics"] if isinstance(r["metrics"], dict) else {}
            thw = m.get("thwaites") if isinstance(m.get("thwaites"), dict) else {}
            print("%s nu_std %s Re_theta cfd %s thw %s table4 %s sep cfd %s thw %s"
                  % (r["law"], m.get("nu_std"), m.get("Re_theta_exit"), thw.get("Re_theta_thw_exit"),
                     (TABLE4[r["law"]]["Re_theta"]), m.get("separated_cfd"), thw.get("separated_thw")))
        return 0
    sys.stderr.write(USAGE + chr(10))
    return 2


def rel(a, b):
    """The relative difference the selftest asserts with."""
    return abs(a / b - 1.0)


def _mdoc(nu_std=None, nu_range=None, sep=None, re_theta=None, thw_sep=None, thw_re=None):
    """A planted metrics doc for compare's tests."""
    thw = None
    if thw_sep is not None or thw_re is not None:
        thw = {"status": "ok", "separated_thw": thw_sep, "Re_theta_thw_exit": thw_re}
    return {"status": "ok", "nu_std": nu_std, "nu_range": nu_range, "separated_cfd": sep,
            "Re_theta_exit": re_theta, "thwaites": thw}


def _t1():
    c = convention()
    want = (91.0 / 76.0, 244.0 / 274.0, 1.0, 122.0 / 114.0)
    for row, w in zip(c["evidence"], want):
        assert row["computed"] == w and row["matches"] is True
    assert c["H_i"] == "full inlet height across the symmetry plane, H_i = 2 h_i"
    assert c["rejected"]["B"] == 244.0 / 137.0 and c["rejected"]["reading"] == "H_i = h_i"
    p = params_for("cubic_matched")
    assert p["x_m"] == 0.5 and params_for("poly5")["x_m"] is None
    assert p["CR"] == 7.7 ** 2 and p["L_over_Di"] == 0.89 and p["D_i"] == 2.74
    try:
        params_for("poly9")
        raise AssertionError("params_for(poly9) was not refused")
    except Refused as r:
        assert r.rule == "BM-NAME"
    assert H_E == 1.37 / 7.7 and U_INLET == 15.0 / 7.7 and L_M == 0.89 * 2.74
    assert _laws_of([]) == LAWS and _laws_of(["poly5"]) == ("poly5",)
    assert _laws_of(("poly7", "poly3")) == ("poly7", "poly3")
    print("[ok] T1 convention: A %.13f B %.13f C %.1f D %.13f all match, rejected B %.12f,"
          " CR %r, h_e U_inlet exact, build laws [] -> LAWS, ['poly5'] -> ('poly5',)"
          % (want[0], want[1], want[2], want[3],
             244.0 / 137.0, params_for("poly5")["CR"]))


def _t2():
    base = {"poly3": _mdoc(nu_std=0.010, nu_range=0.030, sep=False),
            "poly5": _mdoc(nu_std=0.004, nu_range=0.012, sep=False),
            "poly7": _mdoc(nu_std=0.002, nu_range=0.006, sep=True),
            "cubic_matched": _mdoc(sep=True)}
    c = compare(base)
    assert c["o1"]["status"] == "AGREE" and c["o1_range"]["status"] == "AGREE"
    assert c["o2_cfd"]["status"] == "AGREE" and c["o2_thw"]["status"] == "NOT_RUN"
    assert compare(dict(base, poly5=_mdoc(nu_std=0.012, nu_range=0.036, sep=False)))["o1"]["status"] \
        == "DISAGREE"
    assert compare(dict(base, poly7=None))["o1"]["status"] == "NOT_RUN"
    flipped = dict(base, poly3=_mdoc(nu_std=0.010, nu_range=0.030, sep=True))
    assert compare(flipped)["o2_cfd"]["status"] == "DISAGREE"
    partial = {"poly3": _mdoc(sep=False), "poly5": _mdoc(sep=False), "poly7": None, "cubic_matched": None}
    assert compare(partial)["o2_cfd"]["status"] == "PARTIAL"
    one = {"poly3": None, "poly5": _mdoc(re_theta=428.0 * 1.1), "poly7": None, "cubic_matched": None}
    rt = compare(one)["re_theta"]
    assert rel(rt["poly5"]["cfd_rel"], 0.1) <= 1e-9 and rt["poly5"]["table4"] == 428
    assert rt["poly5"]["thw_rel"] is None and rt["order_cfd"] is None and rt["order_thw"] is None
    print("[ok] T2 compare on planted rows: o1 AGREE / DISAGREE (poly5 0.012) / NOT_RUN (poly7"
          " missing), o2_cfd AGREE / DISAGREE / PARTIAL, poly5 cfd_rel 0.1, orders None")


def _t3(td):
    out = os.path.join(td, "outA")
    doc = build(out, ("poly5",))
    assert doc["laws"] == ["poly5"] and doc["cases"]["poly5"]["U_inlet_m_s"] == U_INLET
    row = doc["meshes"]["poly5"]
    assert row["cells"] == 19200
    assert row["patch_faces"] == {"inlet": 80, "outlet": 80, "slip_upstream": 40, "wall_nozzle": 200,
                                  "symmetry": 240, "front": 19200, "back": 19200}
    assert row["types"] == WANT_TYPES
    assert abs(row["volume_rel"]) <= 1e-6, row["volume_rel"]
    assert abs(row["h1_rel"]) <= 1e-4, row["h1_rel"]
    assert isinstance(row["tau_min"], float) and 40 < row["non_orth_max_deg"] < 45
    with open(os.path.join(out, "cases", "poly5", "0", "U"), "r", encoding="utf-8") as f:
        u_text = f.read()
    assert "value           uniform (%r 0.0 0.0);" % U_INLET in u_text
    case = common.read_json(os.path.join(out, "cases", "poly5", "case.json"))
    rows = dict((r["name"], r) for r in case["patches"])
    assert rows["symmetry"]["U"] == {"type": "symmetryPlane"}
    assert rows["front"]["U"] == {"type": "empty"} and rows["back"]["p"] == {"type": "empty"}
    assert rows["inlet"]["U"]["type"] == "fixedValue" and rows["wall_nozzle"]["U"]["type"] == "noSlip"
    assert rows["outlet"]["p"] == {"type": "fixedValue", "value": 0.0}
    assert case["kind"] == "bm_planar" and case["cold_start"] is True
    try:
        build(out, ("poly5",))
        raise AssertionError("a second build into the non-empty directory was accepted")
    except Refused as r:
        assert r.rule == "BM-OUT"
    print("[ok] T3 the real poly5 build: %d cells, patch faces %r, types ok, volume_rel %+.2e,"
          " h1_rel %+.2e, tau %.6f, non-orth %.2f deg, the case cold with U_inlet %r and a second"
          " build BM-OUT" % (row["cells"], row["patch_faces"], row["volume_rel"], row["h1_rel"],
                             row["tau_min"], row["non_orth_max_deg"], U_INLET))
    return out


def _t4(td, out, geom_dir):
    case = os.path.join(out, "cases", "poly5")
    u_e = RECIPE["U_exit_m_s"]

    def planted(name, u_x, p_x):
        copy = os.path.join(td, name)
        shutil.copytree(case, copy)
        _fx_plant(copy, "100", u_x, p_x)
        return metrics(copy, "100", geom_dir)

    u_a = lambda C: u_e * (1.0 + 0.01 * (C[:, 1] / H_E) ** 2)
    m = planted("t4a", u_a, lambda C: np.zeros(len(C)))
    assert m["status"] == "ok" and list(m.keys()) == list(METRIC_KEYS)
    assert m["n_core_faces"] == 22, m["n_core_faces"]
    assert rel(m["nu_std"], 0.0019060696475369304) <= 1e-3, m["nu_std"]
    assert rel(m["nu_range"], 0.00626159273828457) <= 1e-3, m["nu_range"]
    assert m["separated_cfd"] is False and m["reversal"]["n_reversed"] == 0
    u_b = lambda C: u_e * np.minimum(1.0, (H_E - C[:, 1]) / 0.005)
    mb = planted("t4b", u_b, lambda C: np.zeros(len(C)))
    assert rel(mb["theta_exit"], 0.0008347659389646812) <= 1e-3, mb["theta_exit"]
    assert rel(mb["dstar_exit"], 0.0025048329019165473) <= 1e-3, mb["dstar_exit"]
    assert mb["H_exit"] == mb["dstar_exit"] / mb["theta_exit"]
    assert rel(mb["Re_theta_exit"], u_e * mb["theta_exit"] / case_writer.NU_AIR) <= 1e-12
    copy = os.path.join(td, "t4c")
    shutil.copytree(case, copy)
    mesh = _mesh(copy)
    wst, wnf, _t = mesh["patch_range"]["wall_nozzle"]
    wf = np.arange(wst, wst + wnf, dtype=np.int64)
    oc = mesh["owner"][wf]
    mask = np.zeros(mesh["n_cells"], dtype=bool)
    mask[oc[(mesh["C"][oc, 0] >= 0.1) & (mesh["C"][oc, 0] <= 0.3)]] = True
    u_c = lambda C: np.where(mask, -1.0, u_e * (1.0 + 0.01 * (C[:, 1] / H_E) ** 2))
    _fx_plant(copy, "100", u_c, lambda C: np.zeros(len(C)))
    mc = metrics(copy, "100", geom_dir)
    assert mc["separated_cfd"] is True and mc["reversal"]["n_reversed"] == 13, mc["reversal"]
    assert mc["reversal"]["n_bands"] == 1
    assert mc["reversal"]["bands_x_m"][0][0] >= 0.1 and mc["reversal"]["bands_x_m"][0][1] <= 0.3
    for k in ("nu_std", "nu_range"):
        assert mc[k] == m[k]
    print("[ok] T4 planted post: (a) n_core 22, nu_std %.13f, nu_range %.13f; (b) theta %.13f,"
          " dstar %.13f; (c) 13 reversed in one band in [0.1, 0.3], else separated_cfd False"
          % (m["nu_std"], m["nu_range"], mb["theta_exit"], mb["dstar_exit"]))


def _t5(td, out, geom_dir):
    case = os.path.join(out, "cases", "poly5")
    copy = os.path.join(td, "t5")
    shutil.copytree(case, copy)
    mesh = _mesh(copy)
    cent = mesh["C"]
    wst, wnf, _t = mesh["patch_range"]["wall_nozzle"]
    wf = np.arange(wst, wst + wnf, dtype=np.int64)
    wx, wy = mesh["Cf"][wf, 0], mesh["Cf"][wf, 1]
    y_w = wy[np.argmin(np.abs(wx[None, :] - cent[:, 0][:, None]), axis=1)]
    p_plant = 0.5 * (U_INLET ** 2 - (U_INLET * H_I / y_w) ** 2)
    _fx_plant(copy, "100", lambda C: np.full(len(C), U_INLET), lambda C: p_plant)
    m = metrics(copy, "100", geom_dir)
    assert m["status"] == "ok" and m["thwaites"]["status"] == "ok", (m["status"], m["thwaites"])
    sizes = dict((n, mesh["patch_range"][n][1]) for n in PATCHES)
    u_field = post.read_field(os.path.join(copy, "100", "U"), 3, mesh["n_cells"], sizes)
    p_field = post.read_field(os.path.join(copy, "100", "p"), 1, mesh["n_cells"], sizes)
    oc = mesh["owner"][wf]
    order = np.lexsort((wf, mesh["C"][oc, 0]))
    wall_f, cells = wf[order], oc[order]
    st_in, nf_in, _t2 = mesh["patch_range"]["inlet"]
    sf = mesh["Sf"][st_in:st_in + nf_in]
    a = np.sqrt(np.einsum("ij,ij->i", sf, sf))
    u_in = post.face_values(mesh, u_field, "inlet")
    p_in = np.asarray(post.face_values(mesh, p_field, "inlet")).reshape(-1)
    p0_core = float(np.sum(a * (p_in + 0.5 * np.einsum("ij,ij->i", u_in, u_in)))) / float(np.sum(a))
    u_edge = np.sqrt(2.0 * (p0_core - p_field["internal"][cells]))
    cfx, cfy_w = mesh["Cf"][wall_f, 0], mesh["Cf"][wall_f, 1]
    cs_x = _plane_x(geom_dir, "contraction_start")
    s_0 = math.hypot(float(cfx[0]) - cs_x, float(cfy_w[0]) - H_I)
    s = np.concatenate(([0.0, s_0], s_0 + np.cumsum(np.hypot(np.diff(cfx), np.diff(cfy_w)))))
    res = thwaites.solve(s, np.concatenate(([u_edge[0]], u_edge)), case_writer.NU_AIR)
    x_stat = np.concatenate(([cs_x], cfx))
    s_exit = float(np.interp(_plane_x(geom_dir, "exit_plane"), x_stat, s))
    want_xsep = None
    if res["x_sep"] is not None:
        v = float(np.interp(res["x_sep"], s, x_stat))
        want_xsep = v if math.isfinite(v) else None
    assert rel(m["thwaites"]["theta_thw_exit"], float(np.interp(s_exit, s, res["theta"]))) <= 1e-12
    assert rel(m["thwaites"]["Re_theta_thw_exit"], float(np.interp(s_exit, s, res["Re_theta"]))) <= 1e-12
    assert m["thwaites"]["separated_thw"] == res["separated"]
    assert m["thwaites"]["x_sep_thw"] == want_xsep
    copy2 = os.path.join(td, "t5edge")
    shutil.copytree(case, copy2)
    x_in, x_out = _plane_x(geom_dir, "inlet"), _plane_x(geom_dir, "outlet")
    _fx_plant(copy2, "100", lambda C: np.full(len(C), U_INLET),
              lambda C: 1e6 * (C[:, 0] - x_in) / (x_out - x_in))
    me = metrics(copy2, "100", geom_dir)
    assert me["status"] == "ok" and me["thwaites"] == {"status": "refused", "reason_id": "BM-EDGE"}
    copy3 = os.path.join(td, "t5flat")
    shutil.copytree(case, copy3)
    _fx_plant(copy3, "100", lambda C: np.full(len(C), U_INLET), lambda C: np.zeros(len(C)))
    mf = metrics(copy3, "100", geom_dir)
    want_theta = math.sqrt(0.45 * case_writer.NU_AIR * s_exit / U_INLET)
    assert rel(mf["thwaites"]["theta_thw_exit"], want_theta) <= 1e-4, \
        (mf["thwaites"]["theta_thw_exit"], want_theta)
    assert mf["thwaites"]["separated_thw"] is False
    print("[ok] T5 Thwaites plumbing: the metrics block equals a direct thwaites.solve on the same"
          " n + 1 stations (theta_thw %.9e m, Re_theta %.3f, separated %s); the uniform U_INLET"
          " / p 0 field gives the flat plate theta_thw %.9e m vs sqrt(0.45 nu s_exit / U_INLET)"
          " %.9e m at s_exit %.6f (separated False), and p_owner above p0_core is refused BM-EDGE"
          % (m["thwaites"]["theta_thw_exit"], m["thwaites"]["Re_theta_thw_exit"],
             m["thwaites"]["separated_thw"], mf["thwaites"]["theta_thw_exit"], want_theta, s_exit))


def _fake_exe(td, tag):
    """The selftest's fake solver binary in the temp dir (never a tree file): it plants a
    positive everywhere BL-on-the-inlet-height U and a linear p on the five window times of its
    -iters, prints the steady log and exits 0."""
    fake = os.path.join(td, "fake_%s.py" % tag)
    lines = ["import sys",
             "sys.path.insert(0, %r)" % HERE,
             "import numpy as np",
             "import bm_planar as bm",
             "case = sys.argv[1]",
             "iters = int(sys.argv[sys.argv.index('-iters') + 1])",
             "U_E, HI = bm.RECIPE['U_exit_m_s'], bm.H_I",
             "def u_x(C):",
             "    return U_E * np.minimum(1.0, (HI - C[:, 1]) / 0.005)",
             "def p_x(C):",
             "    return 5.0 * (4.17 - C[:, 0])",
             "for t in range(iters - 200, iters + 1, 50):",
             "    bm._fx_plant(case, str(t), u_x, p_x)",
             "for it in range(iters):",
             "    r = 10.0 ** (-2.0 - 5.0 * it / max(1, iters - 1))",
             "    print('iter %d  |U| res %.6e  |p| res %.6e  contErr 1.000000e-09  T [293.15,"
             " 293.15] K  rho [1.20412, 1.20412] kg/m3  p0 101325 Pa  dp0/dt 0 Pa/s  M max"
             " 1.000000e-02 (cell 0) mean 5.000000e-03' % (it, r, 10.0 * r))",
             "print('run ended: budget | %d iterations reached | exit code 0' % iters)"]
    with open(fake, "w", encoding="utf-8") as f:
        f.write(chr(10).join(lines) + chr(10))
    return fake


def _t6(td, out):
    fake = _fake_exe(td, "poly5")
    snap = {"status": "ok", "gpu": {"name": "FAKE GPU", "memory_used_mib": 10, "memory_total_mib": 100,
                                    "utilization_pct": 0}, "compute_apps": [], "detail": ""}
    doc = run(out, "poly5", exe=[sys.executable, fake], visible=False, snapshot_fn=lambda: dict(snap))
    assert doc.get("class") == "steady", (doc.get("class"), doc.get("reason_id"), doc.get("detail"))
    rdir = os.path.join(out, "runs", "poly5")
    assert sorted(os.listdir(rdir)) == ["gpu.json", "metrics.json", "solve.json", "solve.log", "wall.json"]
    rec = record(out)
    assert rec["status"] == "REPORTED" and rec["reduced"] is True and rec["iters"] == 2000
    assert rec["all_steady"] is True
    assert rec["o1"]["status"] == "NOT_RUN" and rec["o1_range"]["status"] == "NOT_RUN"
    assert rec["o2_cfd"]["status"] == "PARTIAL" and rec["o2_thw"]["status"] == "PARTIAL"
    assert [r["law"] for r in rec["runs"]] == ["poly3", "poly5", "poly7", "cubic_matched"]
    assert rec["runs"][0]["solve"] is None and rec["runs"][0]["mesh"] is None
    assert rec["runs"][1]["metrics"]["status"] == "ok"
    text = common.canonical_json(rec)
    stripped = text.replace(SOURCE["url"], "")
    assert ":/" not in stripped and (":" + chr(92)) not in stripped, "an absolute path leaked"
    try:
        run(out, "poly9", exe=[sys.executable, fake], visible=False, snapshot_fn=lambda: dict(snap))
        raise AssertionError("run poly9 was not refused")
    except Refused as r:
        assert r.rule == "BM-NAME"
    try:
        run(out, "poly5", exe=[sys.executable, fake], visible=False, snapshot_fn=lambda: dict(snap))
        raise AssertionError("a second run of poly5 was accepted")
    except Refused as r:
        assert r.rule == "BM-OUT"
    mpath = os.path.join(out, "runs", "poly5", "metrics.json")
    with open(mpath, "rb") as f:
        blob = f.read()
    assert main(["metrics", out, "poly5"]) == 0
    with open(mpath, "rb") as f:
        assert f.read() == blob, "the metrics rewrite is not byte-identical"
    try:
        metrics_rewrite(out, "poly3")
        raise AssertionError("metrics poly3 without a run was accepted")
    except Refused as r:
        assert r.rule == "BM-OUT"
    try:
        metrics_rewrite(out, "poly9")
        raise AssertionError("metrics poly9 was accepted")
    except Refused as r:
        assert r.rule == "BM-NAME"
    assert main(["metrics", out, "poly3"]) == 2
    print("[ok] T6 the fake-exe run of poly5 is steady and leaves gpu.json, metrics.json,"
          " solve.json, solve.log and wall.json; the record is REPORTED reduced with o1 NOT_RUN,"
          " runs 4 rows (poly3's solve None), no absolute path; the metrics rewrite of poly5 is"
          " byte-identical and poly3 (BM-OUT) / poly9 (BM-NAME) are refused")


def selftest():
    """T1-T2 pure, T3 the real poly5 build, T4-T5 planted post on T3's mesh, T6 the fake-exe run,
    record and metrics rewrite on T3's build; SELFTEST PASS at the end."""
    _t1()
    _t2()
    with tempfile.TemporaryDirectory() as td:
        out = _t3(td)
        geom_dir = os.path.join(out, "geom_poly5")
        _t4(td, out, geom_dir)
        _t5(td, out, geom_dir)
        _t6(td, out)
    print("SELFTEST PASS")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
