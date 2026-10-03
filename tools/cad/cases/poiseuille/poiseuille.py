#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""poiseuille.py - CAD-19 (docs/16 §I CAD-19, §H.4 G0 and G-REPEAT): G0, the Hagen-Poiseuille
physics gate of chain P, on the nozzle's own wedge machinery.

A 5-degree wedge of one pipe, R = 0.005 m and L = 0.3 m (30 D), Re_D = 100, meshed at three
levels (nr 10/20/40, nx 150/300/600, uniform in both directions) by the geo-kernel wedge builder
of pipe_mesh.py, converted and -checked by wedge_mesh.py, written as a cold cad-case/1 of
case_writer.py's own system, constant and field files, and measured over x in [20 D, 30 D), a
window that starts about 14 D past the 5.7 D development length of Durst et al. 2005, J. Fluids
Eng. 127(6):1154-1160, DOI 10.1115/1.2063088. Hagen-Poiseuille flow is the exact reference
(Sutera & Skalak 1993, Annu. Rev. Fluid Mech. 25:1-20, DOI 10.1146/annurev.fl.25.010193.000245):
u = 2 U_IN (1 - r^2/R^2), p = G_p (L - x) with G_p = 8 nu U_IN / R^2, f Re = 64 and
u_axis/u_mean = 2 exactly. The observed order of f Re, u_ratio and shear_balance across L0, L1
and L2 is the grid-convergence order of Roache 1997, DOI 10.1115/1.2910291, reported, never gated.

Definitions (fixed before any run):
  D1 mesh and fields: post.load_mesh(case/constant/polyMesh), post.read_field of <case>/<t>/U
     (3 components) and /p (1) with the patch face counts; p carries [0 2 -2 0 0 0 0] (kinematic)
     else G0-UNITS; a field read error is G0-FIELD (post.Refused caught).
  D2 theta from the two wedge sides' summed face normals exactly as post.py line 1099,
     factor = 2 pi / theta; A = sum |Sf_x| over the inlet; Q_in = -patch_flux(inlet), Q_out =
     patch_flux(outlet); dQ_rel = |Q_out - Q_in| / Q_in; U_m = Q_in / A; D = 2 sqrt(factor A / pi)
     (the area-equivalent diameter, as post's pipe path); Re_D = U_m D / nu.
  D3 columns: cells sorted by C_x (stable), a new column when C_x exceeds the column's first by
     more than COLUMN_TOL_M, each column sorted by |C| (stable); G0-STATION unless at least 3
     columns of one equal size >= 3; a column's x_c is the mean C_x of its cells; the window is
     the columns with 0.2 <= x_c < 0.3.
  D4 p_col = sum(p V)/sum(V) per column; G = -slope of p_col against x_c over the window columns
     (numpy.polyfit(x, y, 1)[0]); fRe = 2 G D^2 / (nu U_m); fRe_rel = fRe / 64 - 1.
  D5 u_axis = post._axis_fit on the two innermost cells of the column (u = U_x, r = |C|); u_mean
     = sum(u_x V)/sum(V); ratio = u_axis/u_mean; u_ratio is their mean over the window columns,
     with min and max; u_ratio_rel = u_ratio / 2 - 1.
  D6 wall shear, second order: per wall face with 0.2 <= Cf_x < 0.3, P1 its owner (the outermost
     cell of its column, G0-STATION otherwise), P2 the next cell inward; n = Sf/|Sf|;
     d1, d2 the wall-normal distances of P1, P2; u1, u2 the x components of u - (u.n)n;
     a = (u1 d2^2 - u2 d1^2)/(d1 d2 (d2 - d1)); tau_x = nu a; F_shear = sum tau_x |Sf|;
     F_shear_first_order = sum nu (u1/d1) |Sf| over the same faces.
  D7 Lw = max(fx_max) - min(fx_min) over the window wall faces; dp_window = G Lw;
     shear_balance = F_shear/(dp_window A) - 1, first order likewise.
  D8 observed order on L0, L1, L2 (r = 2): monotone iff (v0-v1)(v1-v2) > 0;
     p = ln((v0-v1)/(v1-v2))/ln 2 when monotone and the second difference is nonzero, else None.
  D9 the G0 verdict on the gate level L2 only: PASS iff the solve class is steady and
     |fRe_rel| <= 0.01 and |u_ratio_rel| <= 0.01 and |shear_balance| <= 0.01 and dQ_rel <= 1e-6;
     otherwise OPEN with the failing reasons G0-MISSING, G0-UNSTEADY, G0-FRE, G0-URATIO,
     G0-SHEAR, G0-MASS in that order. L0 and L1 are reported, never gated.
  D10 G-REPEAT: L1r is a second cold case of the same L1 mesh, solved the same way; the band is
     |a - b| of fRe, u_ratio, shear_balance, dQ_rel and dp_window at the final time;
     fields_bit_identical and log_iter_lines_identical are reported, never assumed (docs/16 §D).

Usage:
  python poiseuille.py --selftest
  python poiseuille.py build OUT_DIR
  python poiseuille.py run OUT_DIR NAME            (NAME one of L0 L1 L2 L1r; the GPU solve, supervisor only)
  python poiseuille.py metrics CASE_DIR TIME OUT_JSON
  python poiseuille.py record OUT_DIR RECORD_JSON
  python poiseuille.py gmsh-build LEVEL MSH_PATH   (the fresh child of build_level)
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

CAD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, CAD)
import common
import case_writer
import wedge_mesh
import post
import solve

VERSION = "cad-g0/1"
CASE_KIND = "poiseuille"
RECIPE = {"version": "cad-poiseuille/1", "R_m": 0.005, "L_m": 0.3, "theta_deg": 5.0, "re_d": 100.0,
          "levels": [0, 1, 2], "nr": [10, 20, 40], "nx": [150, 300, 600], "radial": "uniform", "axial": "uniform"}
U_IN = RECIPE["re_d"] * case_writer.NU_AIR / (2.0 * RECIPE["R_m"])     # 0.15 m/s
WINDOW = (0.2, 0.3)            # m: x in [20 D, 30 D), D = 2 R
COLUMN_TOL_M = 1e-9
VOLUME_REL_MAX = 3e-3          # GC-6's 0.3 %, against theta/2 R^2 L
BANDS = {"fRe": 64.0, "fRe_rel": 0.01, "u_ratio": 2.0, "u_ratio_rel": 0.01, "shear_balance": 0.01, "dQ_rel": 1e-6}
ITERS = 20000                  # each solve's -iters budget, fixed before any run
NAMES = ("L0", "L1", "L2", "L1r")
LEVEL_OF = {"L0": 0, "L1": 1, "L2": 2, "L1r": 1}
GATE_NAME = "L2"
REPEAT_NAMES = ("L1", "L1r")
PATCHES = ("inlet", "outlet", "wall", "wedge_front", "wedge_back")
ROLES = {"inlet": "velocity_inlet", "outlet": "pressure_outlet", "wall": "wall", "wedge_front": "wedge",
         "wedge_back": "wedge"}
BIN_GPU = os.path.join(CAD, "bin_gpu.json")
REFUSAL_IDS = ("G0-OUT", "G0-NAME", "G0-MESH", "G0-FIELD", "G0-UNITS", "G0-STATION")
VERDICT_IDS = ("G0-MISSING", "G0-UNSTEADY", "G0-FRE", "G0-URATIO", "G0-SHEAR", "G0-MASS")
LEVEL_KEYS = ("name", "level", "nr", "nx", "cells", "elements", "volume_m3", "volume_ref_m3", "volume_rel",
              "patches", "check", "msh_sha256", "polymesh_sha256", "mesh_pass")
BUILD_KEYS = ("version", "recipe", "recipe_sha", "u_in_m_s", "levels", "cases")
CASE_KEYS = ("version", "case_writer_version", "status", "kind", "level", "mesh", "operating_point", "patches",
             "fields", "numerics", "sources", "cold_start", "files")
METRIC_KEYS = ("status", "reason_id", "detail", "time", "n_cells", "theta_mesh_rad", "factor", "A_sector_m2",
               "Q_in_m3_s", "Q_out_m3_s", "dQ_rel", "U_m_m_s", "D_m", "Re_D", "x_window_m", "n_window_columns",
               "G_m_s2", "dp_window", "fRe", "fRe_rel", "u_ratio", "u_ratio_rel", "u_ratio_min", "u_ratio_max",
               "F_shear", "F_shear_first_order", "shear_balance", "shear_balance_first_order")
RECORD_KEYS = ("version", "recipe", "recipe_sha", "bands", "iters", "gate_level", "binary", "runs",
               "observed_order", "repeat", "g0", "g_repeat")
RUN_KEYS = ("name", "level", "cells", "mesh", "solve", "wall_s", "metrics")
SOLVE_ROW_KEYS = ("class", "reason_id", "failed", "criteria", "n_iter_lines", "log_sha256", "binary_sha256")
ORDER_KEYS = ("values", "monotone", "p")
REPEAT_KEYS = ("names", "classes", "delta", "fields_bit_identical", "field_sha256", "log_iter_lines_identical")
G0_KEYS = ("verdict", "reasons", "checks")
G_REPEAT_KEYS = ("status", "band", "bit_identical", "note")
USAGE = ("usage: python poiseuille.py --selftest" + chr(10)
         + "       python poiseuille.py build OUT_DIR" + chr(10)
         + "       python poiseuille.py run OUT_DIR NAME" + chr(10)
         + "       python poiseuille.py metrics CASE_DIR TIME OUT_JSON" + chr(10)
         + "       python poiseuille.py record OUT_DIR RECORD_JSON" + chr(10)
         + "       python poiseuille.py gmsh-build LEVEL MSH_PATH")


class Refused(wedge_mesh.Refused):
    """A refusal by id (case_writer.Refused's shape): rule and detail."""


def _level_dims(level):
    """(nr, nx) of one level, refused G0-MESH for an unknown level."""
    if level not in RECIPE["levels"]:
        raise wedge_mesh.Refused("G0-MESH", "level %r is not one of %r" % (level, RECIPE["levels"]))
    i = RECIPE["levels"].index(level)
    return RECIPE["nr"][i], RECIPE["nx"][i]


def gmsh_build(level, msh_path):
    """The gmsh child's whole job (pipe_mesh.gmsh_build's pattern): the meridian rectangle
    x in [0, L], y in [0, R] on z = 0, rotated by -theta/2 about +x and revolved by theta with one
    recombined layer, transfinite nx + 1 points on the axis and outer lines and nr + 1 on both
    radial lines, UNIFORM (no Progression), General.NumThreads 1; the five patch groups and the
    fluid volume -> MSH 4.1 ASCII at msh_path. Refused G0-MESH on an unknown level, element types
    other than hex + prism, or a group count other than one surface each."""
    import gmsh
    nr, nx = _level_dims(level)
    r, ln = RECIPE["R_m"], RECIPE["L_m"]
    th = math.radians(RECIPE["theta_deg"])
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        geo = gmsh.model.geo
        p1 = geo.addPoint(0.0, 0.0, 0.0, 1.0)
        p2 = geo.addPoint(ln, 0.0, 0.0, 1.0)
        p3 = geo.addPoint(ln, r, 0.0, 1.0)
        p4 = geo.addPoint(0.0, r, 0.0, 1.0)
        axis_l = geo.addLine(p1, p2)
        rad_far = geo.addLine(p3, p2)                  # outer -> axis at x = L
        outer_l = geo.addLine(p3, p4)
        rad_near = geo.addLine(p4, p1)                 # outer -> axis at x = 0
        surf = geo.addPlaneSurface([geo.addCurveLoop([axis_l, -rad_far, outer_l, rad_near])])
        geo.synchronize()
        src = [(2, surf)]
        geo.rotate(src, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -th / 2.0)
        geo.revolve(src, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, th, [1], recombine=True)
        geo.synchronize()
        for t in (axis_l, outer_l):
            gmsh.model.mesh.setTransfiniteCurve(t, nx + 1)
        for t in (rad_far, rad_near):
            gmsh.model.mesh.setTransfiniteCurve(t, nr + 1)
        gmsh.model.mesh.setTransfiniteSurface(surf)
        gmsh.model.mesh.setRecombine(2, surf)
        groups = _classify(gmsh)
        for name in PATCHES:
            gmsh.model.addPhysicalGroup(2, sorted(groups[name]), name=name)
        gmsh.model.addPhysicalGroup(3, sorted(t for d, t in gmsh.model.getEntities(3)), name="fluid")
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.model.mesh.generate(3)
        types, etags, _ = gmsh.model.mesh.getElements(3)
        elements = dict((wedge_mesh.ELEMENT_NAMES.get(int(t), "type%d" % int(t)), len(g))
                        for t, g in zip(types, etags))
        if sorted(elements) != ["hex", "prism"]:
            raise wedge_mesh.Refused("G0-MESH", "volume elements %r, want hex and prism only" % (elements,))
        os.makedirs(os.path.dirname(os.path.abspath(msh_path)), exist_ok=True)
        gmsh.write(msh_path)
        version = gmsh.GMSH_API_VERSION
    finally:
        gmsh.finalize()
    return {"level": level, "nr": nr, "nx": nx, "elements": elements,
            "n_cells": sum(elements.values()), "gmsh": version}


def _classify(gmsh):
    """Every surface of the revolved model into one of PATCHES by bounding box, as pipe_mesh's:
    a plane face at x = 0 is the inlet and at x = L the outlet, a face spanning both wedge sides
    is the wall, and the two flat sides separate by the sign of their z extent. Refused G0-MESH
    unless each of the five groups holds exactly one surface."""
    rs = RECIPE["R_m"] * math.sin(math.radians(RECIPE["theta_deg"]) / 2.0)
    ln = RECIPE["L_m"]
    groups = dict((g, []) for g in PATCHES)
    for d, t in gmsh.model.getEntities(2):
        b = gmsh.model.getBoundingBox(2, t)
        if b[3] - b[0] < wedge_mesh.TOL_GEOM:
            if abs(b[0]) < wedge_mesh.TOL_GEOM:
                groups["inlet"].append(t)
            elif abs(b[0] - ln) < wedge_mesh.TOL_GEOM:
                groups["outlet"].append(t)
        elif b[2] < -rs / 2.0 and b[5] > rs / 2.0:
            groups["wall"].append(t)
        elif b[5] < rs / 2.0:
            groups["wedge_front"].append(t)
        else:
            groups["wedge_back"].append(t)
    got = dict((g, len(v)) for g, v in groups.items())
    want = dict((g, 1) for g in PATCHES)
    if got != want:
        raise wedge_mesh.Refused("G0-MESH", "surface groups %r, want %r" % (got, want))
    return groups


def build_level(level, msh_path):
    """One fresh gmsh-build child (pipe_mesh.build_level's protocol); the build dict from its
    last stdout line, wedge_mesh.Refused on the child's REFUSED_EXIT."""
    pr = subprocess.run([sys.executable, os.path.abspath(__file__), "gmsh-build", str(level), msh_path],
                        capture_output=True, text=True, encoding="utf-8", timeout=wedge_mesh.CHILD_TIMEOUT_S,
                        env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    lines = [s for s in pr.stdout.splitlines() if s.strip()]
    line = json.loads(lines[-1]) if lines else {}
    if pr.returncode == 0:
        return line
    if pr.returncode == wedge_mesh.REFUSED_EXIT:
        raise wedge_mesh.Refused(line["refused"], line["detail"])
    raise RuntimeError("gmsh-build exited %d: %s" % (pr.returncode, pr.stderr[-600:]))


def mesh_level(level, out_dir, bins):
    """One level's mesh: the fresh gmsh-build child into mesh/L<level>.msh, the converter into
    mesh/L<level> with wedge_mesh.TYPE_ARGS, the five F1 patch types required (G0-MESH otherwise),
    the -check at wedge_mesh.GC6's tau_min, the volume against theta/2 R^2 L; the LEVEL_KEYS row."""
    nr, nx = _level_dims(level)
    msh_path = os.path.join(out_dir, "mesh", "L%d.msh" % level)
    b = build_level(level, msh_path)
    case_dir = os.path.join(out_dir, "mesh", "L%d" % level)
    wedge_mesh.convert(bins, msh_path, case_dir, wedge_mesh.TYPE_ARGS)
    types = wedge_mesh.patch_types(case_dir)
    want_types = {"wedge_front": "wedge", "wedge_back": "wedge", "inlet": "patch", "wall": "wall",
                  "outlet": "patch"}
    if types != want_types:
        raise wedge_mesh.Refused("G0-MESH", "boundary is %r, want %r" % (types, want_types))
    chk = wedge_mesh.run_check(bins, case_dir, os.path.join(out_dir, "mesh", "check_L%d.json" % level),
                               wedge_mesh.GC6["tau_min"])
    check = dict((k, v) for k, v in chk.items() if k != "text")
    pm_dir = os.path.join(case_dir, "constant", "polyMesh")
    mesh = post.load_mesh(pm_dir)
    volume = float(np.sum(mesh["V"]))
    volume_ref = math.radians(RECIPE["theta_deg"]) / 2.0 * RECIPE["R_m"] ** 2 * RECIPE["L_m"]
    volume_rel = volume / volume_ref - 1.0
    patches = dict((name, {"type": types[name], "n_faces": mesh["patch_range"][name][1]}) for name in PATCHES)
    return {"name": "L%d" % level, "level": level, "nr": b["nr"], "nx": b["nx"], "cells": b["n_cells"],
            "elements": b["elements"], "volume_m3": volume, "volume_ref_m3": volume_ref, "volume_rel": volume_rel,
            "patches": patches, "check": check, "msh_sha256": common.sha256_file(msh_path),
            "polymesh_sha256": dict((nm, common.sha256_file(os.path.join(pm_dir, nm)))
                                    for nm in case_writer.POLYMESH_FILES),
            "mesh_pass": bool(chk["exit"] == 0 and chk["gate"] == "passed" and abs(volume_rel) <= VOLUME_REL_MAX)}


def write_case(mesh_dir, level, out_dir):
    """One cold cad-case/1 case in out_dir (refused G0-OUT when it exists): the five polyMesh
    files copied byte for byte, case_writer's system, constant and field files of the boundary's
    own bc_table rows, then case.json LAST, keyed CASE_KEYS, carrying no name, path or time - so
    the L1 and L1r cases are byte-identical."""
    if os.path.exists(out_dir):
        raise Refused("G0-OUT", "%s exists" % out_dir)
    pm_dir = os.path.join(mesh_dir, "constant", "polyMesh")
    mesh = post.load_mesh(pm_dir)
    missing = [n for n in PATCHES if n not in mesh["patch_range"]]
    if missing:
        raise wedge_mesh.Refused("G0-MESH", "boundary lacks %r" % (missing,))
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
        if nm not in ROLES:
            raise wedge_mesh.Refused("G0-MESH", "patch %s has no G0 role" % nm)
        role = ROLES[nm]
        bcs = case_writer.bc_table(role, [U_IN, 0.0, 0.0], case_writer.STATE["T_K"])
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
    st, nf, _t = mesh["patch_range"]["inlet"]
    a_in = float(np.sum(np.abs(mesh["Sf"][st:st + nf, 0])))
    msh_path = os.path.join(os.path.dirname(mesh_dir), "L%d.msh" % level)
    case = {"version": "cad-case/1", "case_writer_version": case_writer.CASE_WRITER_VERSION, "status": "ok",
            "kind": CASE_KIND, "level": level,
            "mesh": {"recipe_sha": common.sha256_of(RECIPE), "msh_sha256": common.sha256_file(msh_path),
                     "cells": mesh["n_cells"],
                     "polymesh_sha256": dict((nm, files["constant/polyMesh/" + nm])
                                             for nm in case_writer.POLYMESH_FILES)},
            "operating_point": {"fluid": case_writer.STATE["fluid"], "T_K": case_writer.STATE["T_K"],
                                "p0_Pa": case_writer.STATE["p0_Pa"], "nu_m2_s": case_writer.NU_AIR,
                                "Re_D": RECIPE["re_d"], "U_inlet_m_s": U_IN, "A_inlet_m2": a_in,
                                "Q_m3_s": U_IN * a_in},
            "patches": rows,
            "fields": {"U": {"dimensions": "[0 1 -1 0 0 0 0]", "internal": [0.0, 0.0, 0.0]},
                       "p": {"dimensions": "[0 2 -2 0 0 0 0]", "internal": 0.0},
                       "T": {"dimensions": "[0 0 0 1 0 0 0]", "internal": case_writer.STATE["T_K"]}},
            "numerics": {"transcribed": [list(r) for r in case_writer.TRANSCRIBED],
                         "differences": list(case_writer.DIFFERENCES)},
            "sources": [{"id": s["id"], "file": s["file"], "commit": s["commit"], "blob": s["blob"],
                         "lines": s["lines"], "text_sha256": s["text_sha256"]}
                        for s in common.read_json(case_writer.SOURCES)["sources"]],
            "cold_start": True, "files": files}
    with open(os.path.join(out_dir, "case.json"), "wb") as f:
        f.write((common.canonical_json(case) + chr(10)).encode("utf-8"))
    return case


def build(out_dir):
    """The three meshes (G0-MESH when any mesh_pass is false), then the four cold cases (L1r from
    the L1 mesh), then build.json; refused G0-OUT unless out_dir is missing or empty."""
    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise Refused("G0-OUT", "%s exists and is not empty" % out_dir)
    os.makedirs(out_dir, exist_ok=True)
    bins = wedge_mesh.load_bins()
    levels = [mesh_level(level, out_dir, bins) for level in RECIPE["levels"]]
    failed = [lv["name"] for lv in levels if not lv["mesh_pass"]]
    if failed:
        raise wedge_mesh.Refused("G0-MESH", "mesh_pass false for %r" % (failed,))
    cases = {}
    for name in NAMES:
        level = LEVEL_OF[name]
        write_case(os.path.join(out_dir, "mesh", "L%d" % level), level, os.path.join(out_dir, "cases", name))
        cases[name] = "cases/" + name
    doc = {"version": VERSION, "recipe": RECIPE, "recipe_sha": common.sha256_of(RECIPE), "u_in_m_s": U_IN,
           "levels": levels, "cases": cases}
    with open(os.path.join(out_dir, "build.json"), "wb") as f:
        f.write((common.canonical_json(doc) + chr(10)).encode("utf-8"))
    return doc


_MESH_CACHE = {}


def _mesh(case_dir):
    """post.load_mesh of the case's polyMesh, cached per directory in this process."""
    pm = os.path.abspath(os.path.join(case_dir, "constant", "polyMesh"))
    if pm not in _MESH_CACHE:
        _MESH_CACHE[pm] = post.load_mesh(pm)
    return _MESH_CACHE[pm]


def _columns(C):
    """D3: the cells as axial columns - sorted by C_x (stable), a new column when C_x exceeds the
    column's first by more than COLUMN_TOL_M, each column sorted by |C| (stable)."""
    rr = np.hypot(C[:, 1], C[:, 2])
    order = np.argsort(C[:, 0], kind="stable")
    cols = []
    cur = [int(order[0])]
    for k in range(1, len(order)):
        idx = int(order[k])
        if C[idx, 0] - C[cur[0], 0] > COLUMN_TOL_M:
            cols.append(cur)
            cur = [idx]
        else:
            cur.append(idx)
    cols.append(cur)
    for col in cols:
        col.sort(key=lambda i: rr[i])
    return cols


def metrics(case_dir, time_name):
    """D1-D7 into METRIC_KEYS; never raises on a bad case - a refusal comes back as
    status "refused" with its reason_id, the detail, the time and None elsewhere."""
    tdir = os.path.join(case_dir, time_name)

    def _ref(rid, detail):
        row = dict((k, None) for k in METRIC_KEYS)
        row["status"], row["reason_id"], row["detail"], row["time"] = "refused", rid, detail, time_name
        return row

    try:
        mesh = _mesh(case_dir)
    except Exception as e:                                   # polyMesh unreadable at all
        return _ref("G0-MESH", "%s" % e)
    ranges = mesh["patch_range"]
    missing = [n for n in PATCHES if n not in ranges]
    if missing:
        return _ref("G0-MESH", "boundary lacks %r" % (missing,))
    try:
        u_field = post.read_field(os.path.join(tdir, "U"), 3, mesh["n_cells"],
                                  dict((n, ranges[n][1]) for n in PATCHES))
        p_field = post.read_field(os.path.join(tdir, "p"), 1, mesh["n_cells"],
                                  dict((n, ranges[n][1]) for n in PATCHES))
    except (post.Refused, OSError, ValueError) as e:
        return _ref("G0-FIELD", "%s" % e)
    if p_field["dimensions"] != "[0 2 -2 0 0 0 0]":
        return _ref("G0-UNITS", "p carries %s, want [0 2 -2 0 0 0 0]" % p_field["dimensions"])
    nf_, nb_ = post._wedge_normals(mesh)
    theta = float(math.atan2(float(np.linalg.norm(np.cross(nf_, nb_))), abs(float(np.dot(nf_, nb_)))))
    factor = 2.0 * math.pi / theta
    st_in, nf_in, _t = ranges["inlet"]
    area = float(np.sum(np.abs(mesh["Sf"][st_in:st_in + nf_in, 0])))
    q_in = -post.patch_flux(mesh, u_field, "inlet")
    q_out = post.patch_flux(mesh, u_field, "outlet")
    if not q_in > 0.0:
        return _ref("G0-FIELD", "the inlet flux %r is not positive" % q_in)
    d_q_rel = abs(q_out - q_in) / q_in
    u_mean = q_in / area
    d_h = 2.0 * math.sqrt(factor * area / math.pi)
    re_d = u_mean * d_h / case_writer.NU_AIR
    cols = _columns(mesh["C"])
    sizes = set(len(c) for c in cols)
    if len(cols) < 3 or len(sizes) != 1 or min(sizes) < 3:
        return _ref("G0-STATION", "%d columns of sizes %r, want >= 3 columns of one size >= 3"
                    % (len(cols), sorted(sizes)))
    cell_col = np.empty(mesh["n_cells"], dtype=np.int64)
    for j, col in enumerate(cols):
        for i in col:
            cell_col[i] = j
    lo, hi = WINDOW
    win = []
    for col in cols:
        x_c = float(np.mean(mesh["C"][col, 0]))
        if lo <= x_c < hi:
            win.append((col, x_c))
    if not win:
        return _ref("G0-STATION", "no column in x in [%r, %r)" % (lo, hi))
    vol = mesh["V"]
    p_int = p_field["internal"]
    u_int = u_field["internal"]
    xs = []
    p_cols = []
    ratios = []
    for col, x_c in win:
        idx = np.array(col)
        v_sum = float(np.sum(vol[idx]))
        xs.append(x_c)
        p_cols.append(float(np.sum(p_int[idx] * vol[idx]) / v_sum))
        uc = u_int[idx, 0]
        u_col_mean = float(np.sum(uc * vol[idx]) / v_sum)
        r0, r1 = float(np.hypot(mesh["C"][col[0], 1], mesh["C"][col[0], 2])), \
            float(np.hypot(mesh["C"][col[1], 1], mesh["C"][col[1], 2]))
        ratios.append(post._axis_fit(float(uc[0]), float(uc[1]), r0, r1) / u_col_mean)
    g_m = -float(np.polyfit(np.array(xs), np.array(p_cols), 1)[0])
    f_re = 2.0 * g_m * d_h * d_h / (case_writer.NU_AIR * u_mean)
    u_ratio = float(np.mean(ratios))
    u_ratio_rel = u_ratio / 2.0 - 1.0
    wst, wnf, _t = ranges["wall"]
    sf_w = mesh["Sf"][wst:wst + wnf]
    cf_w = mesh["Cf"][wst:wst + wnf]
    fx_w = cf_w[:, 0]
    sel = np.nonzero((fx_w >= lo) & (fx_w < hi))[0]
    if len(sel) == 0:
        return _ref("G0-STATION", "no wall face in x in [%r, %r)" % (lo, hi))
    cent = mesh["C"]
    f_shear = 0.0
    f_shear1 = 0.0
    for k in sel:
        f = wst + int(k)
        owner = int(mesh["owner"][f])
        col = cols[int(cell_col[owner])]
        if col[-1] != owner:
            return _ref("G0-STATION", "wall face %d owns cell %d, not its column's outermost cell %d"
                        % (f, owner, col[-1]))
        inner = col[-2]
        n_vec = sf_w[k] / np.linalg.norm(sf_w[k])
        d1 = float(np.dot(cf_w[k] - cent[owner], n_vec))
        d2 = float(np.dot(cf_w[k] - cent[inner], n_vec))
        u1v = u_int[owner] - float(np.dot(u_int[owner], n_vec)) * n_vec
        u2v = u_int[inner] - float(np.dot(u_int[inner], n_vec)) * n_vec
        u1, u2 = float(u1v[0]), float(u2v[0])
        a_fit = (u1 * d2 * d2 - u2 * d1 * d1) / (d1 * d2 * (d2 - d1))
        f_shear += case_writer.NU_AIR * a_fit * float(np.linalg.norm(sf_w[k]))
        f_shear1 += case_writer.NU_AIR * (u1 / d1) * float(np.linalg.norm(sf_w[k]))
    lw = float(np.max(mesh["fx_max"][wst:wst + wnf][sel]) - np.min(mesh["fx_min"][wst:wst + wnf][sel]))
    dp_window = g_m * lw
    shear_balance = f_shear / (dp_window * area) - 1.0
    shear_balance1 = f_shear1 / (dp_window * area) - 1.0
    x_window = [float(np.min(mesh["fx_min"][wst:wst + wnf][sel])),
                float(np.max(mesh["fx_max"][wst:wst + wnf][sel]))]
    return {"status": "ok", "reason_id": None, "detail": "", "time": time_name, "n_cells": mesh["n_cells"],
            "theta_mesh_rad": theta, "factor": factor, "A_sector_m2": area, "Q_in_m3_s": q_in,
            "Q_out_m3_s": q_out, "dQ_rel": d_q_rel, "U_m_m_s": u_mean, "D_m": d_h, "Re_D": re_d,
            "x_window_m": x_window, "n_window_columns": len(win), "G_m_s2": g_m, "dp_window": dp_window,
            "fRe": f_re, "fRe_rel": f_re / BANDS["fRe"] - 1.0, "u_ratio": u_ratio,
            "u_ratio_rel": u_ratio_rel, "u_ratio_min": float(np.min(ratios)), "u_ratio_max": float(np.max(ratios)),
            "F_shear": f_shear, "F_shear_first_order": f_shear1, "shear_balance": shear_balance,
            "shear_balance_first_order": shear_balance1}


def history(case_dir, geom_dir, iters_list):
    """One solve.HISTORY_KEYS row per time of the window, the metrics doc behind it: the dp slot
    carries dp_window and the Cd slot carries f Re, so the stop rule's 1e-5 relative change then
    applies to f Re; never raises."""
    rows = []
    for it in iters_list:
        m = metrics(case_dir, str(it))
        rows.append({"iter": it, "time": str(it), "status": m["status"], "reason_id": m["reason_id"],
                     "dp": m["dp_window"], "Cd": m["fRe"], "post_sha256": common.sha256_of(m)})
    return rows


def observed_order(values):
    """D8 into ORDER_KEYS (Roache 1997, r = 2); None in every key when a value is None."""
    if any(v is None for v in values):
        return {"values": None, "monotone": None, "p": None}
    e01, e12 = values[0] - values[1], values[1] - values[2]
    monotone = bool(e01 * e12 > 0.0)
    p_hat = math.log(e01 / e12) / math.log(2.0) if monotone and e12 != 0.0 else None
    return {"values": list(values), "monotone": monotone, "p": p_hat}


def judge(m, solve_class):
    """D9 into G0_KEYS; the checks dict carries each value and limit, pass false when missing."""
    ok = isinstance(m, dict) and m.get("status") == "ok"
    checks = {"steady": {"value": solve_class, "pass": solve_class == "steady"}}
    limits = (("fRe", "fRe_rel", BANDS["fRe_rel"]), ("u_ratio", "u_ratio_rel", BANDS["u_ratio_rel"]),
              ("shear_balance", "shear_balance", BANDS["shear_balance"]), ("dQ_rel", "dQ_rel", BANDS["dQ_rel"]))
    for name, key, limit in limits:
        v = m.get(key) if ok else None
        checks[name] = {"value": v, "limit": limit, "pass": bool(ok and v is not None and abs(v) <= limit)}
    reasons = []
    if not ok:
        reasons.append("G0-MISSING")
    if solve_class != "steady":
        reasons.append("G0-UNSTEADY")
    if ok:
        ids = {"fRe": "G0-FRE", "u_ratio": "G0-URATIO", "shear_balance": "G0-SHEAR", "dQ_rel": "G0-MASS"}
        for name, _key, _limit in limits:
            if not checks[name]["pass"]:
                reasons.append(ids[name])
    return {"verdict": "PASS" if not reasons else "OPEN", "reasons": reasons, "checks": checks}


def run(out_dir, name, iters=ITERS, exe=None, visible=True):
    """One solve of cases/<name> into runs/<name> through solve.launch, the D-1 binary of bin_gpu
    when exe is None (G0-NAME for a bad name, G0-OUT when the run directory exists); timed, the
    solve doc returned and wall.json written."""
    if name not in NAMES:
        raise Refused("G0-NAME", "name %r is not one of %r" % (name, list(NAMES)))
    rdir = os.path.join(out_dir, "runs", name)
    if os.path.exists(rdir):
        raise Refused("G0-OUT", "%s exists" % rdir)
    print("[g0] solving %s for %d iterations..." % (name, iters))
    t0 = time.monotonic()
    doc = solve.launch(os.path.join(out_dir, "cases", name), os.path.join(out_dir, "mesh"), rdir, iters,
                       exe=exe, bin_json=None if exe else BIN_GPU, history_fn=history, visible=visible)
    wall_s = time.monotonic() - t0
    os.makedirs(rdir, exist_ok=True)
    with open(os.path.join(rdir, "wall.json"), "wb") as f:
        f.write((common.canonical_json({"wall_s": wall_s}) + chr(10)).encode("utf-8"))
    print("[g0] %s class %s in %.1f s" % (name, doc.get("class"), wall_s))
    return doc


def _fx_plant(case_dir, time_name, p_scale=1.0, outlet_scale=1.0, axis_scale=1.0):
    """The F4 plant into <case_dir>/<time_name>/U and /p: the exact parabola on the wall faces'
    mean radius, p = G_p (L - C_x), the per-face boundary values, the outlet's inletValue line,
    and three perturbation knobs (p_scale on p, outlet_scale on the outlet's per-face U,
    axis_scale on every column's innermost cell U)."""
    mesh = post.load_mesh(os.path.join(case_dir, "constant", "polyMesh"))
    cent = mesh["C"]
    r_sq = cent[:, 1] ** 2 + cent[:, 2] ** 2
    wst, wnf, _t = mesh["patch_range"]["wall"]
    r_p = float(np.mean(np.hypot(mesh["Cf"][wst:wst + wnf, 1], mesh["Cf"][wst:wst + wnf, 2])))
    g_p = 8.0 * case_writer.NU_AIR * U_IN / (r_p * r_p)
    u_int = np.zeros((mesh["n_cells"], 3))
    u_int[:, 0] = 2.0 * U_IN * (1.0 - r_sq / (r_p * r_p))
    if axis_scale != 1.0:
        for col in _columns(cent):
            u_int[col[0], 0] *= axis_scale
    p_int = g_p * (RECIPE["L_m"] - cent[:, 0]) * p_scale
    u_out = np.zeros((mesh["patch_range"]["outlet"][1], 3))
    u_out[:, 0] = U_IN * outlet_scale
    u_patches = [("inlet", "fixedValue", np.tile(np.array([U_IN, 0.0, 0.0]), (mesh["patch_range"]["inlet"][1], 1)),
                  None),
                 ("outlet", "inletOutlet", u_out, "        inletValue      uniform (0 0 0);"),
                 ("wall", "noSlip", None, None),
                 ("wedge_front", "wedge", None, None), ("wedge_back", "wedge", None, None)]
    p_patches = [("inlet", "zeroGradient", None, None),
                 ("outlet", "fixedValue", np.zeros(mesh["patch_range"]["outlet"][1]), None),
                 ("wall", "zeroGradient", None, None),
                 ("wedge_front", "wedge", None, None), ("wedge_back", "wedge", None, None)]
    post._fx_write_field(os.path.join(case_dir, time_name, "U"), "[0 1 -1 0 0 0 0]", "volVectorField", "U",
                         time_name, u_int, u_patches)
    post._fx_write_field(os.path.join(case_dir, time_name, "p"), "[0 2 -2 0 0 0 0]", "volScalarField", "p",
                         time_name, p_int, p_patches)


def record(out_dir):
    """RECORD_KEYS over the four runs (docs/16 §H.4 G0 and G-REPEAT): the runs' SOLVE_ROW_KEYS
    rows, the observed order over L0, L1, L2, the L1/L1r repeat band, the G0 verdict on L2 and
    the g_repeat row. No absolute path, no out_dir and no time stamp goes into the record."""
    build_doc = common.read_json(os.path.join(out_dir, "build.json"))
    lv_rows = dict((lv["level"], lv) for lv in build_doc["levels"])
    solves = {}
    for name in NAMES:
        sj = os.path.join(out_dir, "runs", name, "solve.json")
        solves[name] = common.read_json(sj) if os.path.isfile(sj) else None
    iters_set = set(s["iters"] for s in solves.values() if s)
    if len(iters_set) > 1:
        raise Refused("G0-OUT", "the four solve.json records disagree on iters: %r" % (sorted(iters_set),))
    iters = iters_set.pop() if iters_set else None
    binary = None
    for name in NAMES:
        if solves[name] and solves[name].get("binary"):
            binary = dict(solves[name]["binary"])
            if os.path.isabs(binary.get("path", "")):
                binary["path"] = os.path.basename(binary["path"])   # no absolute path in the record
            break
    runs = []
    for name in NAMES:
        lv = lv_rows[LEVEL_OF[name]]
        s = solves[name]
        if s is None:
            solve_row, wall_s, m = None, None, None
        else:
            res = s.get("result") or {}
            solve_row = {"class": s.get("class"), "reason_id": s.get("reason_id"), "failed": res.get("failed"),
                         "criteria": res.get("criteria"), "n_iter_lines": len(res.get("iterations") or []),
                         "log_sha256": (s.get("log") or {}).get("sha256"),
                         "binary_sha256": (s.get("binary") or {}).get("sha256")}
            wj = os.path.join(out_dir, "runs", name, "wall.json")
            wall_s = common.read_json(wj).get("wall_s") if os.path.isfile(wj) else None
            m = metrics(os.path.join(out_dir, "cases", name), str(iters))
        runs.append({"name": name, "level": LEVEL_OF[name], "cells": lv["cells"],
                     "mesh": dict((k, lv[k]) for k in LEVEL_KEYS if k != "name"),
                     "solve": solve_row, "wall_s": wall_s, "metrics": m})
    return _record_tail(out_dir, iters, binary, runs, solves)


def _record_tail(out_dir, iters, binary, runs, solves):
    """record's tail: the observed order, the L1/L1r repeat band, g0 and g_repeat."""
    m_of = dict((r["name"], r["metrics"]) for r in runs)

    def val(name, key):
        m = m_of[name]
        return m.get(key) if m is not None and m.get("status") == "ok" else None

    observed = {"fRe": observed_order([val("L0", "fRe"), val("L1", "fRe"), val("L2", "fRe")]),
                "u_ratio": observed_order([val("L0", "u_ratio"), val("L1", "u_ratio"), val("L2", "u_ratio")]),
                "shear_balance": observed_order([val("L0", "shear_balance"), val("L1", "shear_balance"),
                                                 val("L2", "shear_balance")])}
    delta = {}
    for key in ("fRe", "u_ratio", "shear_balance", "dQ_rel", "dp_window"):
        a, b = val("L1", key), val("L1r", key)
        delta[key] = None if a is None or b is None else abs(a - b)
    fsha = {}
    for fld in ("U", "p"):
        pair = []
        for nm in REPEAT_NAMES:
            p = os.path.join(out_dir, "cases", nm, str(iters), fld)
            pair.append(common.sha256_file(p) if os.path.isfile(p) else None)
        fsha[fld] = pair
    bit = all(x is not None and x == y for x, y in ((fsha["U"][0], fsha["U"][1]),
                                                    (fsha["p"][0], fsha["p"][1])))

    def iter_lines(name):
        p = os.path.join(out_dir, "runs", name, "solve.log")
        if not os.path.isfile(p):
            return None
        with open(p, "r", encoding="utf-8") as fh:
            return [l for l in fh.read().splitlines() if l.startswith("iter ")]

    la, lb = iter_lines("L1"), iter_lines("L1r")
    log_same = None if la is None or lb is None else la == lb
    repeat = {"names": list(REPEAT_NAMES),
              "classes": [solves[n].get("class") if solves[n] else None for n in REPEAT_NAMES],
              "delta": delta, "fields_bit_identical": bool(bit), "field_sha256": fsha,
              "log_iter_lines_identical": log_same}
    g0 = judge(m_of[GATE_NAME], solves[GATE_NAME].get("class") if solves[GATE_NAME] else None)
    reached = all(solves[n] is not None and solves[n].get("class") is not None for n in REPEAT_NAMES) \
        and all(m_of[n] is not None and m_of[n].get("status") == "ok" for n in REPEAT_NAMES)
    g_repeat = {"status": "recorded" if reached else "OPEN", "band": delta, "bit_identical": bool(bit),
                "note": "The nozzle nominal's |dCd| and |dtheta_e| bands are CAD-20's, the unit that builds "
                        "the nominal; this repeat band is G0's own."}
    return {"version": VERSION, "recipe": RECIPE, "recipe_sha": common.sha256_of(RECIPE), "bands": BANDS,
            "iters": iters, "gate_level": GATE_NAME, "binary": binary, "runs": runs,
            "observed_order": observed, "repeat": repeat, "g0": g0, "g_repeat": g_repeat}


def main(argv):
    """--selftest | build OUT_DIR | run OUT_DIR NAME | metrics CASE_DIR TIME OUT_JSON | record OUT_DIR
    RECORD_JSON | gmsh-build LEVEL MSH_PATH (pipe_mesh.main's style)."""
    if argv == ["--selftest"]:
        try:
            selftest()
        except Exception:
            traceback.print_exc()
            return 1
        return 0
    if len(argv) == 3 and argv[0] == "gmsh-build":
        try:
            build_dict = gmsh_build(int(argv[1]), argv[2])
        except wedge_mesh.Refused as r:
            print(common.canonical_json({"refused": r.rule, "detail": r.detail}))
            return wedge_mesh.REFUSED_EXIT
        print(common.canonical_json(build_dict))
        return 0
    if len(argv) == 2 and argv[0] == "build":
        try:
            doc = build(os.path.abspath(argv[1]))
        except wedge_mesh.Refused as r:
            sys.stderr.write("refused %s: %s%s" % (r.rule, r.detail, chr(10)))
            return 2
        for lv in doc["levels"]:
            print("L%d cells %d tau %.6f volume_rel %+.6f mesh_pass %s"
                  % (lv["level"], lv["cells"], float("nan") if lv["check"]["tau_min"] is None
                     else lv["check"]["tau_min"], lv["volume_rel"], lv["mesh_pass"]))
        return 0 if all(lv["mesh_pass"] for lv in doc["levels"]) else 1
    if len(argv) == 3 and argv[0] == "run":
        try:
            doc = run(os.path.abspath(argv[1]), argv[2])
        except wedge_mesh.Refused as r:
            sys.stderr.write("refused %s: %s%s" % (r.rule, r.detail, chr(10)))
            return 2
        print("class %s reason_id %s" % (doc.get("class"), doc.get("reason_id")))
        return 0 if doc.get("class") == "steady" else 1
    if len(argv) == 4 and argv[0] == "metrics":
        m = metrics(os.path.abspath(argv[1]), argv[2])
        with open(os.path.abspath(argv[3]), "wb") as f:
            f.write((common.canonical_json(m) + chr(10)).encode("utf-8"))
        print("metrics status %s reason_id %s" % (m["status"], m["reason_id"]))
        return 0
    if len(argv) == 3 and argv[0] == "record":
        try:
            rec = record(os.path.abspath(argv[1]))
        except wedge_mesh.Refused as r:
            sys.stderr.write("refused %s: %s%s" % (r.rule, r.detail, chr(10)))
            return 2
        with open(os.path.abspath(argv[2]), "wb") as f:
            f.write((common.canonical_json(rec) + chr(10)).encode("utf-8"))
        print("g0 %s reasons %s" % (rec["g0"]["verdict"], ",".join(rec["g0"]["reasons"])))
        return 0 if rec["g0"]["verdict"] == "PASS" else 1
    sys.stderr.write(USAGE + chr(10))
    return 2



def rel(a, b):
    """The relative difference the selftest asserts with."""
    return abs(a / b - 1.0)


def _fake_bin(td, tag, p_scale):
    """The test's fake solver binary in the temp dir (never a tree file): it plants the F4 field
    on the five window times of its -iters, prints the steady fixture log and exits 0."""
    path = os.path.join(td, "fake_%s.py" % tag)
    lines = ["import os, sys",
             "sys.path.insert(0, %r)" % os.path.dirname(os.path.abspath(__file__)),
             "import poiseuille",
             "case = sys.argv[1]",
             "iters = int(sys.argv[sys.argv.index('-iters') + 1])",
             "for t in range(iters - 200, iters + 1, 50):",
             "    poiseuille._fx_plant(case, str(t), p_scale=%r)" % p_scale,
             "with open(os.path.join(%r, 'fixtures', 'solve', 'steady', 'solve.log'), 'rb') as f:" % CAD,
             "    sys.stdout.buffer.write(f.read())"]
    with open(path, "w", encoding="utf-8") as f:
        f.write(chr(10).join(lines) + chr(10))
    return path


def _walk(d):
    """{relative /-separated path: bytes} of every file under d."""
    out = {}
    for root, _dirs, names in os.walk(d):
        for name in names:
            p = os.path.join(root, name)
            with open(p, "rb") as f:
                out[os.path.relpath(p, d).replace(os.sep, "/")] = f.read()
    return out


def _t1():
    assert U_IN == 0.15
    rsha = common.sha256_of(RECIPE)
    doc = common.read_json(BIN_GPU)
    assert list(doc["binaries"].keys()) == ["ofgpu-lowmach"]
    ent = doc["binaries"]["ofgpu-lowmach"]
    assert ent["path"] == "../Iteration-CFD-solver/rust/target/release/ofgpu-lowmach.exe"
    assert ent["sha256"] == "50471caaa54125e0c2eee4fb34eebdfc2da44727d2b818a2b96bb4234c02103c"
    disk = common.sha256_file(os.path.normpath(os.path.join(common.REPO, ent["path"])))
    assert disk == ent["sha256"]
    print("[ok] T1 U_IN 0.15, recipe %s sha %s, bin_gpu pins ofgpu-lowmach and the disk sha matches"
          % (",".join(RECIPE.keys()), rsha))


def _t2(td, out_A, build_doc):
    p1, p2 = os.path.join(td, "t2a.msh"), os.path.join(td, "t2b.msh")
    build_level(0, p1)
    build_level(0, p2)
    assert common.sha256_file(p1) == common.sha256_file(p2)
    l0, l1, l2 = build_doc["levels"]
    assert l0["cells"] == 1500 and l0["elements"] == {"hex": 1350, "prism": 150}
    assert l0["patches"]["inlet"] == {"type": "patch", "n_faces": 10}
    assert l0["patches"]["outlet"] == {"type": "patch", "n_faces": 10}
    assert l0["patches"]["wall"] == {"type": "wall", "n_faces": 150}
    assert l0["patches"]["wedge_front"]["type"] == "wedge" and l0["patches"]["wedge_back"]["type"] == "wedge"
    assert l0["check"]["exit"] == 0 and l0["check"]["gate"] == "passed"
    assert abs(l0["check"]["tau_min"] - 0.065367) <= 1e-6
    assert abs(l0["volume_rel"] - (-0.001268756046250763)) <= 1e-9 and l0["mesh_pass"] is True
    assert l1["cells"] == 6000 and l2["cells"] == 24000 and l2["elements"] == {"hex": 23400, "prism": 600}
    try:
        build(out_A)
        raise AssertionError("build(out_A) was not refused")
    except Refused as r:
        assert r.rule == "G0-OUT"
    print("[ok] T2 two L0 builds byte-identical; 1500 = 1350 hex + 150 prism, L1 6000, L2 24000 = 23400 + 600,"
          " -check passed at tau 0.065367, volume_rel -0.001268756, second build refused G0-OUT")


def _t3(out_A):
    c1, c2 = os.path.join(out_A, "cases", "L1"), os.path.join(out_A, "cases", "L1r")
    m1, m2 = _walk(c1), _walk(c2)
    assert set(m1) == set(m2) and all(m1[k] == m2[k] for k in m1)
    assert sorted(os.listdir(c1)) == ["0", "case.json", "constant", "system"]
    case = common.read_json(os.path.join(c1, "case.json"))
    assert sorted(case.keys()) == sorted(CASE_KEYS)
    assert case["version"] == "cad-case/1" and case["kind"] == CASE_KIND and case["cold_start"] is True
    assert set(case["files"]) == set(k for k in m1 if k != "case.json")
    for k, blob in m1.items():
        if k != "case.json":
            assert case["files"][k] == common.sha256_bytes(blob)
    want = dict(case_writer.system_files())
    want.update(case_writer.constant_files())
    for k, text in want.items():
        assert m1[k].decode("utf-8") == text
    assert case_writer.scan_bcs(c1) == {"U": [], "p": [], "T": []}
    assert "value           uniform (0.15 0.0 0.0);" in m1["0/U"].decode("utf-8")
    print("[ok] T3 cases/L1 and cases/L1r byte-identical incl. case.json; cold cad-case/1 of case_writer's"
          " own files, scan_bcs clean, the inlet block carries uniform (0.15 0.0 0.0)")


def _t4(td, out_A):
    copy = os.path.join(td, "t4case")
    shutil.copytree(os.path.join(out_A, "cases", "L0"), copy)
    _fx_plant(copy, "1")
    m = metrics(copy, "1")
    assert m["status"] == "ok" and list(m.keys()) == list(METRIC_KEYS)
    assert m["n_window_columns"] == 50
    assert rel(m["G_m_s2"], 0.7213725201130465) <= 1e-9
    assert rel(m["D_m"], 0.009993654206313863) <= 1e-9
    assert rel(m["fRe"], 64.04064661036026) <= 1e-9
    assert rel(m["u_ratio"], 1.9966958372833867) <= 1e-9
    assert rel(m["u_ratio_min"], 1.9966958372822996) <= 1e-9
    assert rel(m["u_ratio_max"], 1.9966958372847623) <= 1e-9
    assert rel(m["F_shear"], 7.858969723512004e-08) <= 1e-9
    assert rel(m["shear_balance_first_order"], -0.024561403507880164) <= 1e-9
    assert abs(m["shear_balance"]) <= 1e-9 and m["dQ_rel"] <= 1e-12
    assert rel(m["theta_mesh_rad"], 0.08726646259971661) <= 1e-9
    assert rel(m["A_sector_m2"], 1.0894467843457273e-06) <= 1e-9
    assert abs(m["x_window_m"][0] - 0.2) <= 1e-9 and abs(m["x_window_m"][1] - 0.3) <= 1e-12
    j = judge(m, "steady")
    assert j["verdict"] == "PASS" and j["reasons"] == []
    print("[ok] T4 the planted L0 oracle: fRe %.9f, u_ratio %.13f, shear_balance %.3e, first order %.15f,"
          " dQ %.1e, G %.13f, window [%.13f, %.1f] - PASS"
          % (m["fRe"], m["u_ratio"], m["shear_balance"], m["shear_balance_first_order"], m["dQ_rel"],
             m["G_m_s2"], m["x_window_m"][0], m["x_window_m"][1]))
    return copy, m


def _t5(td, out_A, clean):
    made = [0]

    def planted(**kw):
        made[0] += 1
        copy = os.path.join(td, "t5_%d" % made[0])
        shutil.copytree(os.path.join(out_A, "cases", "L0"), copy)
        _fx_plant(copy, "1", **kw)
        return copy, metrics(copy, "1")

    _c, m_mass = planted(outlet_scale=1.000002)
    assert abs(m_mass["dQ_rel"] - 2e-6) <= 1e-9
    assert judge(m_mass, "steady")["reasons"] == ["G0-MASS"]
    _c, m_p = planted(p_scale=1.02)
    assert rel(m_p["fRe_rel"], 0.020647805352616677) <= 1e-9
    assert rel(m_p["shear_balance"], -0.019607843137254943) <= 1e-9
    assert judge(m_p, "steady")["reasons"] == ["G0-FRE", "G0-SHEAR"]
    _c, m_ax = planted(axis_scale=1.03)
    assert m_ax["u_ratio_rel"] > 0.01
    assert judge(m_ax, "steady")["reasons"] == ["G0-URATIO"]
    for k in ("fRe", "shear_balance", "dQ_rel"):
        assert abs(m_ax[k] - clean[k]) <= 1e-12
    assert judge(clean, "unsteady")["reasons"] == ["G0-UNSTEADY"]
    j = judge(None, None)
    assert j["reasons"] == ["G0-MISSING", "G0-UNSTEADY"] and j["verdict"] == "OPEN"
    copy4, m_u = planted()
    p_path = os.path.join(copy4, "1", "p")
    with open(p_path, "r", encoding="utf-8") as f:
        text = f.read()
    with open(p_path, "w", encoding="utf-8") as f:
        f.write(text.replace("[0 2 -2 0 0 0 0]", "[1 -1 -2 0 0 0 0]"))
    m_u = metrics(copy4, "1")
    assert m_u["status"] == "refused" and m_u["reason_id"] == "G0-UNITS"
    os.remove(p_path)
    m_f = metrics(copy4, "1")
    assert m_f["status"] == "refused" and m_f["reason_id"] == "G0-FIELD"
    for m in (m_u, m_f):
        assert list(m.keys()) == list(METRIC_KEYS) and m["time"] == "1"
        for k in METRIC_KEYS:
            if k not in ("status", "reason_id", "detail", "time"):
                assert m[k] is None
    print("[ok] T5 each check fails alone: dQ 2e-6 -> G0-MASS, p 1.02 -> G0-FRE and G0-SHEAR, axis 1.03 ->"
          " G0-URATIO only, unsteady/missing judges, G0-UNITS and G0-FIELD refusals with every number None")


def _t6(td, out_A):
    out_B = os.path.join(td, "outB")
    os.makedirs(out_B)
    shutil.copytree(os.path.join(out_A, "mesh"), os.path.join(out_B, "mesh"))
    shutil.copytree(os.path.join(out_A, "cases"), os.path.join(out_B, "cases"))
    shutil.copy(os.path.join(out_A, "build.json"), os.path.join(out_B, "build.json"))
    fake = _fake_bin(td, "104", 1.04)
    doc = run(out_B, "L0", iters=600, exe=[sys.executable, fake], visible=False)
    assert doc["class"] == "steady" and doc["reason_id"] is None
    win = doc["result"]["window"]
    assert win["iters"] == [400, 450, 500, 550, 600]
    for v in win["Cd"]:
        assert rel(v, 64.04064661036026 * 1.04) <= 1e-9
    assert len(set(win["dp"])) == 1
    wj = common.read_json(os.path.join(out_B, "runs", "L0", "wall.json"))
    assert isinstance(wj["wall_s"], float) and wj["wall_s"] >= 0.0
    assert sorted(os.listdir(os.path.join(out_B, "runs", "L0"))) == ["solve.json", "solve.log", "wall.json"]
    print("[ok] T6 the fake-binary L0 run is classified steady (Cd = f Re * 1.04 on the five window times,"
          " dp constant), wall.json holds a non-negative wall_s")
    return out_B, fake


def _t7(td, out_B, fake104):
    run(out_B, "L1", iters=600, exe=[sys.executable, _fake_bin(td, "101", 1.01)], visible=False)
    run(out_B, "L2", iters=600, exe=[sys.executable, _fake_bin(td, "1025", 1.0025)], visible=False)
    run(out_B, "L1r", iters=600, exe=[sys.executable, _fake_bin(td, "101b", 1.01)], visible=False)
    rec = record(out_B)
    assert list(rec.keys()) == list(RECORD_KEYS)
    assert rec["g0"]["verdict"] == "PASS" and rec["g0"]["reasons"] == []
    assert rel(rec["g0"]["checks"]["fRe"]["value"], 0.003136691045007023) <= 1e-9
    assert rel(rec["g0"]["checks"]["shear_balance"]["value"], -0.0024937655860348684) <= 1e-9
    order = rec["observed_order"]["fRe"]
    assert order["monotone"] is True and abs(order["p"] - 2.0) <= 1e-6
    for v in rec["repeat"]["delta"].values():
        assert v == 0.0
    assert rec["repeat"]["fields_bit_identical"] is True and rec["repeat"]["log_iter_lines_identical"] is True
    assert rec["g_repeat"]["status"] == "recorded"
    assert rec["binary"]["sha256"] == common.sha256_file(fake104)
    assert [r["name"] for r in rec["runs"]] == ["L0", "L1", "L2", "L1r"]
    text = common.canonical_json(rec)
    assert out_B not in text and os.path.basename(td) not in text
    assert "C:/" not in text and ("C:" + chr(92)) not in text
    print("[ok] T7 record: g0 PASS (fRe_rel %.12f, shear_balance %.12f), observed order 2 (%.9f), the L1/L1r"
          " band all 0.0 and bit-identical, no path in the canonical JSON"
          % (rec["g0"]["checks"]["fRe"]["value"], rec["g0"]["checks"]["shear_balance"]["value"], order["p"]))


def _t8(td, out_B):
    a = os.path.join(td, "outB_a")
    shutil.copytree(out_B, a)
    shutil.rmtree(os.path.join(a, "runs", "L2"))
    for name in os.listdir(os.path.join(a, "cases", "L2")):     # back to a cold case
        if name.isdigit() and name != "0":
            shutil.rmtree(os.path.join(a, "cases", "L2", name))
    run(a, "L2", iters=600, exe=[sys.executable, _fake_bin(td, "102", 1.02)], visible=False)
    assert record(a)["g0"]["reasons"] == ["G0-FRE", "G0-SHEAR"]
    b = os.path.join(td, "outB_b")
    shutil.copytree(out_B, b)
    sj = os.path.join(b, "runs", "L2", "solve.json")
    doc = common.read_json(sj)
    doc["class"] = "unsteady"
    with open(sj, "wb") as f:
        f.write((common.canonical_json(doc) + chr(10)).encode("utf-8"))
    assert record(b)["g0"]["reasons"] == ["G0-UNSTEADY"]
    c = os.path.join(td, "outB_c")
    shutil.copytree(out_B, c)
    shutil.rmtree(os.path.join(c, "runs", "L2"))
    rec_c = record(c)
    assert rec_c["g0"]["reasons"] == ["G0-MISSING", "G0-UNSTEADY"] and rec_c["runs"][2]["solve"] is None
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "poiseuille.py")
    child = dict(os.environ, PYTHONIOENCODING="utf-8")
    rec_path = os.path.join(td, "rec.json")
    pr = subprocess.run([sys.executable, script, "record", out_B, rec_path], capture_output=True,
                        text=True, encoding="utf-8", env=child)
    assert pr.returncode == 0, pr.stderr[-400:]
    with open(rec_path, "rb") as f:
        assert f.read() == (common.canonical_json(record(out_B)) + chr(10)).encode("utf-8")
    m_path = os.path.join(td, "m.json")
    pr = subprocess.run([sys.executable, script, "metrics", os.path.join(td, "t4case"), "1", m_path],
                        capture_output=True, text=True, encoding="utf-8", env=child)
    assert pr.returncode == 0, pr.stderr[-400:]
    with open(m_path, "rb") as f:
        assert f.read() == (common.canonical_json(metrics(os.path.join(td, "t4case"), "1"))
                            + chr(10)).encode("utf-8")
    pr = subprocess.run([sys.executable, script], capture_output=True, text=True, encoding="utf-8")
    assert pr.returncode == 2
    try:
        run(out_B, "L9")
        raise AssertionError("run L9 was not refused")
    except Refused as r:
        assert r.rule == "G0-NAME"
    try:
        run(out_B, "L0")
        raise AssertionError("run L0 was not refused")
    except Refused as r:
        assert r.rule == "G0-OUT"
    print("[ok] T8 the OPEN variants (G0-FRE + G0-SHEAR, G0-UNSTEADY, G0-MISSING + G0-UNSTEADY with runs[2]"
          " solve None) and the CLI (record byte-identical, metrics, usage 2, G0-NAME and G0-OUT)")


def selftest():
    """T1-T8 in one TemporaryDirectory, build(out_A) once and shared; SELFTEST PASS at the end."""
    with tempfile.TemporaryDirectory() as td:
        out_A = os.path.join(td, "outA")
        build_doc = build(out_A)
        _t1()
        _t2(td, out_A, build_doc)
        _t3(out_A)
        _t4_copy, clean = _t4(td, out_A)
        _t5(td, out_A, clean)
        out_B, fake104 = _t6(td, out_A)
        _t7(td, out_B, fake104)
        _t8(td, out_B)
    print("SELFTEST PASS")

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
