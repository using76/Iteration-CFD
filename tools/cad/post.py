#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""post.py - stage S9 of the CAD loop (docs/16 §D, §E.2, §H.4): the performance quantities of one lowmach wedge
case, read from its polyMesh and ASCII fields.

Every input (case.json, the five polyMesh files, the time directory's U and p, geom.json and tags.json) is hashed
through common.stable_file_snapshot before and after the reading and bound to case.json (POST-BIND); the wedge
factor 2 pi / theta is applied only after mesh_fidelity.check reports the mesh at scale 1 with its measured wedge
angle (POST-SCALE, docs/16a §H). Values are raw floats, never rounded, and every metric carries repr cfd. The
pattern of reading a polyMesh with its fields (a patch's own value, else the owner cell) is tools/aero/drag_post.py
at e63c61f; the field format is the solver's own writer (rust/src/io/fields.rs), read here, not copied.
Pohlhausen's quartic profile (1921, ZAMM 1(4):252-290, DOI 10.1002/zamm.19210010402; theta/delta = 37/315,
delta*/delta = 3/10) and the degree-2 triangle (edge midpoints) and tetrahedron (4-point, Keast 1986, DOI
10.1016/0045-7825(86)90059-9) quadrature rules are used only by the selftest's planted fields.

post_turb (docs/16 §H.5, §I CAD-27) reads a turbulent lowmach case - TG0's periodic pipe (geom_dir None,
driven by its constant/fvSources momentumSource) or the turbulent nozzle - into a cad-post-turb/1 doc:
the wall shear tau_w = (nu + nut_f) |u_t,P| / d_P and u_tau from it, y+, u+ and f, the log-law band and
the core defect of the pipe with its rho g_x V balance and x-invariance, and for the nozzle the laminar
metrics plus the wall-shear term inside the axial momentum balance and K(x), p(x) by -nu d(1/U_e)/dx
(Kline et al. 1967; arXiv:2306.05972) from the CFD edge velocity. The laminar post() and its cad-post/1
doc are unchanged.

Usage:
  python post.py --selftest
  python post.py run CASE_DIR TIME GEOM_DIR OUT_JSON
  python post.py run-turb CASE_DIR TIME GEOM_DIR OUT_JSON   (GEOM_DIR - for a pipe)
  python post.py fx-tet GEOM_DIR MSH_PATH
"""
import ast
import math
import os
import shutil
import subprocess
import sys
import tempfile
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common
import reqs
import thwaites
import turb_integral
sys.path.insert(0, os.path.join(common.REPO, "tools", "mesh"))
import polymesh_write
import regions_check

VERSION = "cad-post/1"
REFUSAL_IDS = ("POST-BIND", "POST-PATCH", "POST-SCALE", "POST-UNITS", "POST-FIELD", "POST-STATION")
UNDEFINED_ID = "POST-UNDEFINED"
DIMS_U = "[0 1 -1 0 0 0 0]"
DIMS_P = {"[0 2 -2 0 0 0 0]": "kinematic", "[1 -1 -2 0 0 0 0]": "static"}
PATCHES = ("inlet", "outlet", "wall_nozzle", "slip_upstream", "wedge_front", "wedge_back")
WALL_PATCH = "wall_nozzle"
NEED_VALUE = ("fixedValue", "inletOutlet", "calculated")
NO_VALUE = ("empty", "cyclic", "cyclicSlip", "processor", "wedge", "symmetry", "symmetryPlane",
            "zeroGradient", "noSlip", "slip")
FLAT_TOL_M = 1e-12            # a face is flat in x when its vertices' x agree within this; layers merge within it
EDGE_REL = 1e-12              # the BL edge face: the first face from the wall with u >= U_c (1 - EDGE_REL)
NONUNIF_R = 0.8               # docs/16 §E.2: exit non-uniformity over r <= 0.8 R_e
UP_DI = 0.25                  # docs/16 §E.2 / §H.4: upstream station x = contraction_start - 0.25 D_i
ENTRY_DI = 0.05               # docs/16 §H.5 TG2a: the entry layer at x = contraction_start - 0.05 D_i
DOWN_DE = 0.25                # docs/16 §H.4 G1: downstream station x = exit_plane + 0.25 D_e
STATIONS = ("upstream", "exit_plane", "downstream")
INPUT_KEYS = ("case.json", "polyMesh/boundary", "polyMesh/faces", "polyMesh/neighbour", "polyMesh/owner",
              "polyMesh/points", "fields/U", "fields/p", "geom/geom.json", "geom/tags.json")
VERSION_TURB = "cad-post-turb/1"
CASE_TURB_VERSION = "cad-case-turb/1"
DIMS_NUT = "[0 2 -1 0 0 0 0]"
PATCHES_PIPE = {"periodic_a": "cyclic", "periodic_b": "cyclic", "wall": "wall", "wedge_front": "wedge",
                "wedge_back": "wedge"}
PATCHES_TURB = ("inlet", "outlet", "wall_nozzle", "wall_upstream", "wedge_front", "wedge_back")
WALLS_PIPE = ("wall",)
WALLS_TURB = ("wall_nozzle", "wall_upstream")
LOGLAW_YPLUS_MIN = 30.0          # docs/16 §H.5 TB3: 30 <= y+ <= 0.2 Re_tau
LOGLAW_RE_TAU_FRAC = 0.2
YPLUS_LIMIT = 1.0                # docs/16 §H.5 TG1: y+1 <= 1
INPUT_KEYS_TURB_PIPE = ("case.json", "polyMesh/boundary", "polyMesh/faces", "polyMesh/neighbour",
                        "polyMesh/owner", "polyMesh/points", "fields/U", "fields/p", "fields/nut")
INPUT_KEYS_TURB_NOZZLE = INPUT_KEYS_TURB_PIPE + ("geom/geom.json", "geom/tags.json")
DOC_KEYS = ("version", "status", "reason_id", "message", "time", "inputs", "mesh_fidelity", "wedge",
            "operating_point", "units", "stations", "exit_profile", "metrics", "momentum", "reversal", "edge",
            "boundary_flux")
MFID_KEYS = ("status", "reason_id", "fields", "theta_mesh_rad", "scale_pass", "report_sha256")
WEDGE_KEYS = ("theta_mesh_rad", "factor", "factor_formula", "inlet_area_mesh_m2", "inlet_area_geom_m2",
              "inlet_area_rel", "sin_ratio_minus_1")
OP_KEYS = ("fluid", "rho_kg_m3", "c_m_s", "nu_m2_s", "Q_case_m3_s")
UNITS_KEYS = ("U_dimensions", "p_dimensions", "p_kind")
STATION_KEYS = ("x_m", "layers_x_m", "weight", "n_faces", "R_m", "area_m2", "Q_m3_s", "u_mean_m_s", "p_mean",
                "p0_flux_mean", "u_axis_m_s", "p_axis", "p0_axis")
EXIT_PROFILE_KEYS = ("U_c_m_s", "n_faces", "edge_rank", "y_edge_m")
METRIC_KEYS = ("value", "unit", "repr", "where", "reason_id", "definition")
MOMENTUM_KEYS = ("x_a_m", "x_b_m", "I_a", "I_b", "W", "P_a", "P_b", "residual")
REVERSAL_KEYS = ("n_wall_cells", "n_reversed", "n_bands", "n_sign_changes", "bands_x_m", "area_fraction")
EDGE_KEYS = ("p0_core", "x_m", "r_wall_m", "U_edge_m_s", "n_undefined")
PRESCRIBED_MESH = ("empty", "symmetry", "symmetryPlane", "wedge")     # mesh.rs:84 + the empty kind
PRESCRIBED_U = ("fixedValue", "noSlip")                                # fr = 1 conditions
BFLUX_KEYS = ("patches", "Q_net_m3_s", "Q_other_m3_s", "other_rel", "net_rel", "method")
BFLUX_ROW_KEYS = ("mesh_type", "U_bc", "rule", "Q_m3_s")
BFLUX_METHOD = ("solver rule (momentum.cu momFluxIsPrescribed): prescribed when the mesh type is empty, symmetry, "
                "symmetryPlane or wedge or U is fixedValue/noSlip, Q = F sum(U_b . Sf) with U_b by face_values; "
                "otherwise owner, Q = F sum(U_P . Sf), the owner cell's flux standing for HbyA_P . Sf - rAU_P |Sf| "
                "snGrad(p), whose rAU the written fields do not carry")
ENTRY_KEYS = ("x_target_m", "x_m", "x_inlet_m", "R_m", "n_faces", "U_c_m_s", "edge_rank", "y_edge_m", "theta_m", "dstar_m", "H")
METRICS = (   # (name, unit, where, definition) - the order of the doc's metrics and of records()
    ("Q_in", "m3/s", ["inlet"], "-F sum(U_f . Sf) over the inlet patch"),
    ("Q_out", "m3/s", ["outlet"], "F sum(U_f . Sf) over the outlet patch"),
    ("mass_imbalance", "1", ["inlet", "outlet"], "|Q_out - Q_in| / Q_in"),
    ("dp", "Pa", ["upstream", "exit_plane"], "rho (p_mean(upstream) - p_mean(exit_plane))"),
    ("Cd", "1", ["upstream", "exit_plane"], "Q_in / (A_e sqrt(2 dp_kin)), A_e = area(exit_plane)"),
    ("dp_loss", "Pa", ["upstream", "downstream"], "rho (p0_flux_mean(upstream) - p0_flux_mean(downstream))"),
    ("p0_loss_axis", "1", ["upstream", "downstream"],
     "(p0_axis(upstream) - p0_axis(downstream)) / (U_e^2 / 2), U_e = Q_in / A_e"),
    ("exit_nonuniformity", "1", ["exit_plane"], "(max - min) / area-mean of u_x over faces with r_c <= 0.8 R_e"),
    ("theta_exit", "m", ["exit_plane"], "sum from the wall to the edge face of (u/U_c)(1 - u/U_c) dy"),
    ("dstar_exit", "m", ["exit_plane"], "sum from the wall to the edge face of (1 - u/U_c) dy"),
    ("H_exit", "1", ["exit_plane"], "dstar_exit / theta_exit"),
    ("u_axis_ratio_exit", "1", ["exit_plane"], "u_axis / u_mean at exit_plane"),
    ("mach_max", "1", ["fluid"], "max over cells |U| / c"),
    ("momentum_closure", "1", ["upstream", "downstream"], "(I_a - I_b - W) / |P_a - P_b|"),
    ("reversal_fraction", "1", ["wall_nozzle"], "area of wall_nozzle faces whose owner has u_x < 0 / area"),
    ("separation_free", "1", ["wall_nozzle"], "1 when no wall_nozzle owner cell has u_x < 0, else 0"),
)
DOC_TURB_KEYS = ("version", "status", "reason_id", "message", "time", "kind", "inputs", "mesh_fidelity",
                 "wedge", "operating_point", "units", "wall_shear", "pipe", "nozzle")
WEDGE_TURB_KEYS = ("theta_mesh_rad", "factor", "factor_formula", "theta_source", "sin_ratio_minus_1")
OP_POST_TURB_KEYS = ("fluid", "rho_kg_m3", "nu_m2_s", "c_m_s", "model", "wall", "re_tau", "u_tau_case_m_s",
                     "g_x_m_s2", "U_b_case_m_s", "U_exit_case_m_s", "Q_case_m3_s", "yplus1_apriori")
UNITS_TURB_KEYS = ("U_dimensions", "p_dimensions", "p_kind", "nut_dimensions")
WALL_SHEAR_KEYS = ("n_faces", "area_m2", "tau_mean_m2_s2", "u_tau_m_s", "yplus1_max", "yplus1_mean",
                   "yplus1_frac_le1", "method")
WALL_SHEAR_METHOD = ("tau_w = (nu + nut_f) |u_t,P| / d_P, d_P = (C_f - C_P) . n_f; u_tau = sqrt(area-mean tau_w); "
                     "y+ = d_P sqrt(tau_w,f) / nu")
PIPE_KEYS = ("R_m", "L_m", "V_m3", "D_m", "R_wall_face_m", "n_radial", "U_b_m_s", "Re_D", "u_tau_m_s",
             "Re_tau", "f", "U_axis_m_s", "core_defect", "x_invariance", "profile", "loglaw",
             "shear_force_N", "body_force_N", "balance_rel")
PROFILE_KEYS = ("r_m", "y_m", "u_m_s", "y_plus", "u_plus")
LOGLAW_KEYS = ("y_plus_min", "y_plus_max", "n_cells", "dev_max")
NOZZLE_KEYS = ("stations", "exit_profile", "metrics", "momentum", "reversal", "edge", "momentum_turb",
               "accel", "entry", "edge_upstream")
MOMENTUM_TURB_KEYS = ("x_a_m", "x_b_m", "I_a", "I_b", "W", "S", "P_a", "P_b", "residual", "closure")
ACCEL_KEYS = ("x_m", "U_edge_m_s", "u_tau_m_s", "K", "K_max", "p", "p_min", "method", "reason_id")
ACCEL_METHOD = "K = -nu d(1/U_e)/dx by thwaites.sg_derivative; p = -K (U_e/u_tau)^3"
USAGE = ("usage: python post.py --selftest" + chr(10)
         + "       python post.py run CASE_DIR TIME GEOM_DIR OUT_JSON" + chr(10)
         + "       python post.py run-turb CASE_DIR TIME GEOM_DIR OUT_JSON   (GEOM_DIR - for a pipe)"
         + chr(10)
         + "       python post.py fx-tet GEOM_DIR MSH_PATH")


class Refused(Exception):
    """A refusal by id: .rule (one of REFUSAL_IDS) and .detail."""
    def __init__(self, rule, detail):
        super().__init__("%s: %s" % (rule, detail))
        self.rule = rule
        self.detail = detail


def _tokens(text):
    """The field file as (kind, text) tokens: words, quoted strings, numbers, and {}();[]<>."""
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == " " or c == chr(9) or c == chr(10) or c == chr(13):
            i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            j = text.find("*/", i + 2)
            if j < 0:
                raise ValueError("unterminated comment")
            i = j + 2
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            j = text.find(chr(10), i)
            i = n if j < 0 else j + 1
            continue
        if c == '"':
            j = text.find('"', i + 1)
            if j < 0:
                raise ValueError("unterminated string")
            out.append(("str", text[i + 1:j]))
            i = j + 1
            continue
        if c in "{}();[]<>":
            out.append((c, c))
            i += 1
            continue
        if ("a" <= c <= "z") or ("A" <= c <= "Z") or c == "_":
            j = i + 1
            while j < n and (("a" <= text[j] <= "z") or ("A" <= text[j] <= "Z")
                             or ("0" <= text[j] <= "9") or text[j] == "_"):
                j += 1
            out.append(("word", text[i:j]))
            i = j
            continue
        j = i
        if c == "+" or c == "-":
            j += 1
        k = j
        while j < n and (("0" <= text[j] <= "9") or text[j] == "." or text[j] == "e" or text[j] == "E"
                         or ((text[j] == "+" or text[j] == "-") and (text[j - 1] == "e" or text[j - 1] == "E"))):
            j += 1
        if j == k:
            raise ValueError("cannot tokenise at %r" % text[i:i + 8])
        out.append(("num", text[i:j]))
        i = j
    return out


class _Scan:
    def __init__(self, toks):
        self.t = toks
        self.i = 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else ("end", "")

    def take(self):
        kind, val = self.peek()
        if kind == "end":
            raise ValueError("unexpected end of field text")
        self.i += 1
        return kind, val

    def expect(self, want):
        kind, val = self.take()
        if val != want:
            raise ValueError("expected %r, got %r" % (want, val))


def _float_of(tok):
    try:
        return float(tok[1])
    except ValueError:
        raise ValueError("%r is not a number" % (tok[1],))


def _value(s, n_comp):
    """One uniform value: a float, or a list of n_comp floats from ( x y z )."""
    tok = s.take()
    if tok[1] == "(":
        vs = []
        for _ in range(n_comp):
            vs.append(_float_of(s.take()))
        s.expect(")")
        return vs
    return _float_of(tok)


def _entry(s, n_comp, count, what):
    """One uniform or nonuniform entry expanded to `count` rows (None: size not checked)."""
    kind, val = s.take()
    if val == "uniform":
        v = _value(s, n_comp)
        s.expect(";")
        if count == 0:
            return []
        if count is None:
            return [v]
        if n_comp == 1:
            return [v] * count
        return [list(v)] * count
    if val != "nonuniform":
        raise ValueError("%s: expected uniform or nonuniform, got %r" % (what, val))
    kind2, val2 = s.take()
    rows = []
    if val2 == "0":
        s.expect("(")
        s.expect(")")
        if count is not None and count != 0:
            raise ValueError("%s: empty list, want %d" % (what, count))
    else:
        if val2 != "List":
            raise ValueError("%s: expected List or 0(), got %r" % (what, val2))
        s.expect("<")
        kind3, cls = s.take()
        if cls not in ("scalar", "vector"):
            raise ValueError("%s: unsupported List<%s>" % (what, cls))
        if (cls == "scalar") != (n_comp == 1):
            raise ValueError("%s: List<%s> in a %d-component field" % (what, cls, n_comp))
        s.expect(">")
        kind4, cnt_txt = s.take()
        if not cnt_txt.isdigit():
            raise ValueError("%s: count %r is not a number" % (what, cnt_txt))
        cnt = int(cnt_txt)
        s.expect("(")
        if n_comp == 1:
            while s.peek()[1] != ")":
                rows.append(_float_of(s.take()))
        else:
            while s.peek()[1] != ")":
                rows.append(_value(s, n_comp))
        s.take()
        if len(rows) != cnt:
            raise ValueError("%s: count %d, %d value(s) read" % (what, cnt, len(rows)))
        if count is not None and cnt != count:
            raise ValueError("%s: count %d, want %d" % (what, cnt, count))
    s.expect(";")
    return rows


def _read_field_text(text, n_comp, n_cells, patch_sizes):
    s = _Scan(_tokens(text))
    if s.take()[1] != "FoamFile":
        raise ValueError("no FoamFile header")
    s.expect("{")
    depth = 1
    while depth:
        val = s.take()[1]
        if val == "{":
            depth += 1
        elif val == "}":
            depth -= 1
    s.expect("dimensions")
    s.expect("[")
    nums = []
    while True:
        val = s.take()[1]
        if val == "]":
            break
        nums.append(val)
    s.expect(";")
    dims = "[" + " ".join(nums) + "]"
    if s.take()[1] != "internalField":
        raise ValueError("no internalField")
    internal = _entry(s, n_comp, n_cells, "internalField")
    arr = np.array(internal, dtype=np.float64)
    if arr.ndim == 1 and n_comp > 1:
        raise ValueError("internalField: scalar rows in a %d-component field" % n_comp)
    if arr.shape[0] != n_cells:
        raise ValueError("internalField: %d value(s), want %d" % (arr.shape[0], n_cells))
    if arr.size and not bool(np.isfinite(arr).all()):
        raise ValueError("internalField: a value is not finite")
    if s.take()[1] != "boundaryField":
        raise ValueError("no boundaryField")
    s.expect("{")
    blocks = {}
    while True:
        kind, name = s.take()
        if name == "}":
            break
        s.expect("{")
        ptype = None
        value = None
        while True:
            kind, key = s.take()
            if key == "}":
                break
            kind2, val2 = s.take()
            if val2 == "uniform" or val2 == "nonuniform":
                s.i -= 1
                entry = _entry(s, n_comp, patch_sizes.get(name), "patch %s %s" % (name, key))
                if key == "value":
                    value = entry
            else:
                s.expect(";")
                if key == "type":
                    ptype = val2
        blocks[name] = (ptype, value)
    if s.peek()[0] != "end":
        raise ValueError("trailing tokens after boundaryField")
    return dims, arr, blocks


def read_field(path, n_comp, n_cells, patch_sizes):
    """One ASCII field (C5): {"dimensions": str, "internal": ndarray (n_cells,) or (n_cells, 3),
    "patches": {name: {"type": str, "value": ndarray or None}}}; POST-FIELD on any defect."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    try:
        dims, arr, blocks = _read_field_text(text, n_comp, n_cells, patch_sizes)
    except Refused:
        raise
    except ValueError as e:
        raise Refused("POST-FIELD", "%s: %s" % (os.path.basename(path), e))
    patches = {}
    for name, size in patch_sizes.items():
        if name not in blocks:
            raise Refused("POST-FIELD", "%s: patch %s has no boundaryField entry"
                          % (os.path.basename(path), name))
        ptype, value = blocks[name]
        if ptype is None:
            raise Refused("POST-FIELD", "%s: patch %s has no type" % (os.path.basename(path), name))
        if value is None and (ptype in NEED_VALUE or (ptype not in NEED_VALUE and ptype not in NO_VALUE)):
            raise Refused("POST-FIELD", "%s: patch %s of type %s carries no value"
                          % (os.path.basename(path), name, ptype))
        varr = None if value is None else np.array(value, dtype=np.float64)
        if varr is not None and varr.size and not bool(np.isfinite(varr).all()):
            raise Refused("POST-FIELD", "%s: patch %s carries a value that is not finite"
                          % (os.path.basename(path), name))
        patches[name] = {"type": ptype, "value": varr}
    return {"dimensions": dims, "internal": arr, "patches": patches}


def nut_boundary_empty(path, n_cells, patch_sizes):
    """True when the written field's PARSED boundaryField has no patch entry (the solver's nut
    format, comments and whitespace irrelevant); read as n_comp 1, POST-FIELD when the file
    does not parse."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    try:
        _dims, _arr, blocks = _read_field_text(text, 1, n_cells, patch_sizes)
    except Refused:
        raise
    except ValueError as e:
        raise Refused("POST-FIELD", "%s: %s" % (os.path.basename(path), e))
    return not blocks


def read_nut(path, case_dir, case, kind, n_cells, patch_sizes):
    """One nut field whose written boundaryField may be empty (the solver writes nut without
    patch rows): the written file is parsed once and the decision comes from the PARSED
    boundaryField - with at least one patch entry this is exactly read_field of the file;
    with none the dimensions and internal values come from that one parse and every patch row
    from the case's own 0/nut - whose sha256 must match case.json (POST-BIND) and whose wall
    patches must all be fixedValue there (POST-FIELD naming the patch), keeping the 0/nut
    value, while every other patch keeps its 0/nut type with a None value so face_values takes
    the owner cell."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    try:
        dims, arr, blocks = _read_field_text(text, 1, n_cells, patch_sizes)
    except Refused:
        raise
    except ValueError as e:
        raise Refused("POST-FIELD", "%s: %s" % (os.path.basename(path), e))
    if blocks:
        return read_field(path, 1, n_cells, patch_sizes)
    zero = os.path.join(case_dir, "0", "nut")
    snap = common.stable_file_snapshot(zero)
    if snap["stable"] is not True or snap["sha256"] != (case.get("files") or {}).get("0/nut"):
        raise Refused("POST-BIND", "0/nut: sha differs from case.json")
    ref = read_field(zero, 1, n_cells, patch_sizes)
    walls = WALLS_TURB if kind == "nozzle" else WALLS_PIPE
    patches = {}
    for name in patch_sizes:
        row = ref["patches"][name]
        if name in walls:
            if row["type"] != "fixedValue":
                raise Refused("POST-FIELD", "nut: written without a boundaryField and wall patch %s has "
                              "0/nut type %s" % (name, row["type"]))
            patches[name] = {"type": row["type"], "value": row["value"]}
        else:
            patches[name] = {"type": row["type"], "value": None}
    return {"dimensions": dims, "internal": arr, "patches": patches}


def face_values(mesh, field, name):
    """The values on one patch's faces by the rule of (C5) (its value; noSlip zero; the Symmetry family
    tangential - slip, symmetry, symmetryPlane and wedge, which the solver's field.rs:431 maps to one
    condition; else owner)."""
    st, nf, _t = mesh["patch_range"][name]
    row = field["patches"][name]
    if row["value"] is not None:
        return row["value"]
    ftype = row["type"]
    vector = field["internal"].ndim == 2 and field["internal"].shape[1] == 3
    if vector and ftype == "noSlip":
        return np.zeros((nf, 3))
    if vector and ftype in ("slip", "symmetry", "symmetryPlane", "wedge"):
        v = field["internal"][mesh["owner"][st:st + nf]]
        Sf = mesh["Sf"][st:st + nf]
        n = Sf / np.sqrt(Sf[:, 0] ** 2 + Sf[:, 1] ** 2 + Sf[:, 2] ** 2)[:, None]
        return v - np.einsum("ij,ij->i", v, n)[:, None] * n
    return field["internal"][mesh["owner"][st:st + nf]]


def patch_flux(mesh, U, name):
    """sum(U_f . Sf) over one patch, UNSCALED (the 3-D tet comparison uses it with factor 1)."""
    st, nf, _t = mesh["patch_range"][name]
    uv = face_values(mesh, U, name)
    Sf = mesh["Sf"][st:st + nf]
    return float(np.sum(uv[:, 0] * Sf[:, 0] + uv[:, 1] * Sf[:, 1] + uv[:, 2] * Sf[:, 2]))


def boundary_flux(mesh, U, factor):
    """The solver-rule flux of every boundary patch, outward positive, scaled by the wedge factor (BFLUX_KEYS)."""
    patches = {}
    for name, rng in mesh["patch_range"].items():
        st, nf, mesh_type = rng
        u_bc = U["patches"][name]["type"]
        prescribed = mesh_type in PRESCRIBED_MESH or u_bc in PRESCRIBED_U
        if prescribed:
            ub = face_values(mesh, U, name)
        else:
            ub = U["internal"][mesh["owner"][st:st + nf]]
        Sf = mesh["Sf"][st:st + nf]
        q = factor * float(np.sum(ub[:, 0] * Sf[:, 0] + ub[:, 1] * Sf[:, 1] + ub[:, 2] * Sf[:, 2]))
        patches[name] = {"mesh_type": mesh_type, "U_bc": u_bc,
                         "rule": "prescribed" if prescribed else "owner", "Q_m3_s": q}
    q_net = sum(row["Q_m3_s"] for row in patches.values())
    q_other = sum(row["Q_m3_s"] for name, row in patches.items() if name not in ("inlet", "outlet"))
    q_in = None if "inlet" not in patches else -patches["inlet"]["Q_m3_s"]
    other_rel = None if q_in is None or q_in == 0 else q_other / q_in
    net_rel = None if q_in is None or q_in == 0 else q_net / q_in
    return {"patches": patches, "Q_net_m3_s": q_net, "Q_other_m3_s": q_other,
            "other_rel": other_rel, "net_rel": net_rel, "method": BFLUX_METHOD}


def _dot3(x, y):
    d = x[..., 0] * y[..., 0]
    d = d + x[..., 1] * y[..., 1]
    d = d + x[..., 2] * y[..., 2]
    return d


def _face_groups(faces):
    groups = {}
    for i, f in enumerate(faces):
        groups.setdefault(len(f), []).append(i)
    out = {}
    for k, idxs in groups.items():
        out[k] = np.array(idxs, dtype=np.int64)
    return out


def _fan(points, faces, idxs):
    arr = points[np.array([faces[j] for j in idxs], dtype=np.int64)]
    return arr, arr.mean(axis=1), np.roll(arr, -1, axis=1)


def load_mesh(pm_dir):
    """read_polymesh plus the geometry of (C6): keys points, faces, owner, neighbour, patches, n_cells, n_internal,
    Sf (n_faces, 3), Cf (n_faces, 3), C (n_cells, 3), V (n_cells,), fx_min, fx_max (per-face vertex x range),
    r_min, r_max (per-face vertex radius range), patch_range {name: (startFace, nFaces, type)}."""
    pm = polymesh_write.read_polymesh(pm_dir)
    points = pm["points"]
    faces = pm["faces"]
    owner = np.asarray(pm["owner"], dtype=np.int64)
    neighbour = np.asarray(pm["neighbour"], dtype=np.int64)
    n_faces = len(faces)
    n_internal = len(neighbour)
    mx = int(owner.max())
    if n_internal:
        mx = max(mx, int(neighbour.max()))
    n_cells = mx + 1
    Sf = np.zeros((n_faces, 3))
    Cf = np.zeros((n_faces, 3))
    fx_min = np.zeros(n_faces)
    fx_max = np.zeros(n_faces)
    r_min = np.zeros(n_faces)
    r_max = np.zeros(n_faces)
    for k, idxs in _face_groups(faces).items():
        arr, xavg, b = _fan(points, faces, idxs)
        m = len(idxs)
        sf_g = np.zeros((m, 3))
        ta_g = np.zeros((m, k))
        cf_g = np.zeros((m, 3))
        for j in range(k):
            a = arr[:, j, :]
            bb = b[:, j, :]
            tn = np.cross(a - xavg, bb - xavg)
            ta = np.sqrt(_dot3(tn, tn)) * 0.5
            sf_g += tn * 0.5
            ta_g[:, j] = ta
            cf_g += (xavg + a + bb) / 3.0 * ta[:, None]
        area = np.zeros(m)
        for j in range(k):
            area = area + ta_g[:, j]
        big = area > regions_check.SMALL
        Sf[idxs] = sf_g
        Cf[idxs[big]] = cf_g[big] / area[big, None]
        Cf[idxs[~big]] = xavg[~big]
        fx_min[idxs] = arr[:, :, 0].min(axis=1)
        fx_max[idxs] = arr[:, :, 0].max(axis=1)
        rr = np.hypot(arr[:, :, 1], arr[:, :, 2])
        r_min[idxs] = rr.min(axis=1)
        r_max[idxs] = rr.max(axis=1)
    visits = np.empty(n_faces + n_internal, dtype=np.int64)
    visits[0:2 * n_internal:2] = owner[:n_internal]
    visits[1:2 * n_internal:2] = neighbour
    visits[2 * n_internal:] = owner[n_internal:]
    cf_vis = np.empty((n_faces + n_internal, 3))
    cf_vis[0:2 * n_internal:2] = Cf[:n_internal]
    cf_vis[1:2 * n_internal:2] = Cf[:n_internal]
    cf_vis[2 * n_internal:] = Cf[n_internal:]
    apex = np.zeros((n_cells, 3))
    n_cf = np.zeros(n_cells, dtype=np.int64)
    np.add.at(apex, visits, cf_vis)
    np.add.at(n_cf, visits, 1)
    has = n_cf > 0
    apex[has] /= n_cf[has, None]
    ao = apex[owner[:n_internal]]
    an = apex[neighbour]
    ab = apex[owner[n_internal:]]
    vp = np.empty(n_faces + n_internal)
    vp[0:2 * n_internal:2] = _dot3(Sf[:n_internal], Cf[:n_internal] - ao) * 1.0 / 3.0
    vp[1:2 * n_internal:2] = _dot3(Sf[:n_internal], Cf[:n_internal] - an) * -1.0 / 3.0
    vp[2 * n_internal:] = _dot3(Sf[n_internal:], Cf[n_internal:] - ab) * 1.0 / 3.0
    wgt = np.empty((n_faces + n_internal, 3))
    wgt[0:2 * n_internal:2] = Cf[:n_internal] * 0.75 + ao * 0.25
    wgt[1:2 * n_internal:2] = Cf[:n_internal] * 0.75 + an * 0.25
    wgt[2 * n_internal:] = Cf[n_internal:] * 0.75 + ab * 0.25
    vol = np.zeros(n_cells)
    c_acc = np.zeros((n_cells, 3))
    np.add.at(vol, visits, vp)
    np.add.at(c_acc, visits, wgt * vp[:, None])
    cent = apex.copy()
    pos = np.nonzero(vol > 0.0)[0]
    cent[pos] = c_acc[pos] / vol[pos, None]
    patches = dict((p["name"], (int(p["startFace"]), int(p["nFaces"]), p["type"])) for p in pm["patches"])
    return {"points": points, "faces": faces, "owner": owner, "neighbour": neighbour, "patches": pm["patches"],
            "n_cells": n_cells, "n_internal": n_internal, "Sf": Sf, "Cf": Cf, "C": cent, "V": vol,
            "fx_min": fx_min, "fx_max": fx_max, "r_min": r_min, "r_max": r_max, "patch_range": patches}


def layers(mesh):
    """The complete flat layers of (C7), sorted by x: [{"x": float, "faces": int ndarray, "R": float}, ...]."""
    flat = (mesh["fx_max"] - mesh["fx_min"]) <= FLAT_TOL_M
    idx = np.nonzero(flat)[0]
    order = idx[np.argsort(mesh["fx_min"][idx], kind="stable")]
    fxs = mesh["fx_min"][order]
    out = []
    i, n = 0, len(order)
    while i < n:
        x0 = fxs[i]
        j = i
        while j < n and fxs[j] - x0 <= FLAT_TOL_M:
            j += 1
        grp = order[i:j]
        g = grp[np.argsort(mesh["r_min"][grp], kind="stable")]
        rm = mesh["r_min"][g]
        rx = mesh["r_max"][g]
        if rm[0] <= FLAT_TOL_M and bool(np.all(np.abs(rm[1:] - rx[:-1]) <= FLAT_TOL_M)):
            out.append({"x": float(x0), "faces": g, "R": float(rx[-1])})
        i = j
    return out


def _x_weights(mesh):
    """Per internal face, the (C7) interpolation weight in x between owner and neighbour centroids."""
    o = mesh["owner"][:mesh["n_internal"]]
    nb = mesh["neighbour"]
    cx = mesh["C"][:, 0]
    den = cx[nb] - cx[o]
    num = mesh["Cf"][:mesh["n_internal"], 0] - cx[o]
    return np.where(den == 0.0, 0.5, num / np.where(den == 0.0, 1.0, den))


def _field_faces(mesh, field, w):
    """The (C7) value of `field` on every face: internal faces interpolated in x, boundary faces their patch."""
    o = mesh["owner"][:mesh["n_internal"]]
    nb = mesh["neighbour"]
    vi = field["internal"]
    out = np.empty((len(mesh["faces"]),) + vi.shape[1:])
    if vi.ndim == 2:
        out[:mesh["n_internal"]] = vi[o] * (1.0 - w)[:, None] + vi[nb] * w[:, None]
    else:
        out[:mesh["n_internal"]] = vi[o] * (1.0 - w) + vi[nb] * w
    for name in mesh["patch_range"]:
        st, nf, _t = mesh["patch_range"][name]
        out[st:st + nf] = face_values(mesh, field, name)
    return out


def _axis_fit(v0, v1, r0, r1):
    """The even fit a + b r^2 through the two innermost faces, at r = 0."""
    return v0 - (v1 - v0) / (r1 * r1 - r0 * r0) * (r0 * r0)


_STATION_NUM = ("R_m", "area_m2", "Q_m3_s", "u_mean_m_s", "p_mean", "p0_flux_mean",
                "u_axis_m_s", "p_axis", "p0_axis")


def _layer_agg(mesh, layer, u_face, p_face, factor):
    """The (C7) aggregates of one complete layer, scaled by the wedge factor where they are extensive."""
    fi = layer["faces"]
    a = np.abs(mesh["Sf"][fi, 0])
    uxf = u_face[fi, 0]
    pf = np.asarray(p_face[fi]).reshape(-1)
    A = float(np.sum(a))
    q = float(np.sum(uxf * a))
    mag2 = u_face[fi, 0] ** 2 + u_face[fi, 1] ** 2 + u_face[fi, 2] ** 2
    p0_num = float(np.sum((pf + mag2 * 0.5) * uxf * a))
    rc = np.hypot(mesh["Cf"][fi, 1], mesh["Cf"][fi, 2])
    o = np.argsort(rc, kind="stable")
    u_axis = _axis_fit(uxf[o[0]], uxf[o[1]], rc[o[0]], rc[o[1]])
    p_axis = _axis_fit(pf[o[0]], pf[o[1]], rc[o[0]], rc[o[1]])
    return {"R_m": layer["R"], "area_m2": factor * A, "Q_m3_s": factor * q, "u_mean_m_s": q / A,
            "p_mean": float(np.sum(pf * a)) / A, "p0_flux_mean": (None if q == 0.0 else p0_num / q),
            "u_axis_m_s": u_axis, "p_axis": p_axis, "p0_axis": p_axis + u_axis * u_axis * 0.5}


def _station_row(x, la, aga, lb, agb, w):
    """One STATION_KEYS row: one exact layer (lb None) or the (1-w)/w blend of two."""
    row = {"x_m": float(x),
           "layers_x_m": [la["x"], x_la_of(la, lb)],
           "weight": 0.0 if lb is None else w,
           "n_faces": [len(la["faces"]), len(la["faces"]) if lb is None else len(lb["faces"])]}
    for key in _STATION_NUM:
        if lb is None:
            row[key] = aga[key]
        else:
            va, vb = aga[key], agb[key]
            row[key] = None if va is None or vb is None else (1.0 - w) * va + w * vb
    return row


def x_la_of(la, lb):
    return lb["x"] if lb is not None else la["x"]


def _metrics_run(mesh, U, p_kin, ctx):
    """The computed part of the doc: {"stations", "exit_profile", "metrics", "momentum", "reversal", "edge"} for the
    kinematic pressure field p_kin and ctx {"factor", "theta_mesh_rad", "planes" {name: x}, "rho_kg_m3", "c_m_s"}.
    Raises Refused("POST-STATION", ...) when a station has no bracketing layer or exit_plane no exact layer."""
    lays = layers(mesh)
    factor = ctx["factor"]
    theta = ctx["theta_mesh_rad"]
    planes = ctx["planes"]
    w = _x_weights(mesh)
    u_face = _field_faces(mesh, U, w)
    p_face = _field_faces(mesh, p_kin, w)

    def exact(x):
        for lay in lays:
            if abs(lay["x"] - x) <= FLAT_TOL_M:
                return lay
        return None

    def nearest(x):
        best = None
        for lay in lays:
            if best is None or abs(lay["x"] - x) < abs(best["x"] - x):
                best = lay
        return best

    lc = exact(planes["contraction_start"]) if "contraction_start" in planes else None
    le = exact(planes["exit_plane"]) if "exit_plane" in planes else None
    if lc is None or le is None:
        raise Refused("POST-STATION", "contraction_start or exit_plane has no exact complete layer")
    d_i = 2.0 * lc["R"]
    d_e = 2.0 * le["R"]
    xs = {"upstream": planes["contraction_start"] - UP_DI * d_i,
          "exit_plane": planes["exit_plane"],
          "downstream": planes["exit_plane"] + DOWN_DE * d_e}
    aggs = {}

    def agg_of(lay):
        key = id(lay)
        if key not in aggs:
            aggs[key] = _layer_agg(mesh, lay, u_face, p_face, factor)
        return aggs[key]

    def bracket(x):
        below = [lay for lay in lays if lay["x"] < x]
        above = [lay for lay in lays if lay["x"] > x]
        if not below or not above:
            raise Refused("POST-STATION", "station x=%r has no bracketing complete layer" % x)
        la = below[-1]
        lb = above[0]
        wgt = (x - la["x"]) / (lb["x"] - la["x"])
        return la, lb, wgt

    stations = {}
    for name in STATIONS:
        x = xs[name]
        lay = exact(x)
        if lay is not None:
            stations[name] = _station_row(x, lay, agg_of(lay), None, None, 0.0)
        else:
            la, lb, wgt = bracket(x)
            stations[name] = _station_row(x, la, agg_of(la), lb, agg_of(lb), wgt)
    return {"stations": stations, "lc": lc, "le": le, "xs": xs, "u_face": u_face, "p_face": p_face,
            "lays": lays, "factor": factor, "theta": theta, "ctx": ctx,
            "_exact": exact, "_nearest": nearest, "_agg": agg_of}


def _exit_profile(mesh, le, u_face, theta):
    """The exit-layer profile from the wall inward (C8): U_c, the edge face, theta and delta* sums."""
    fi = le["faces"]
    uxf = u_face[fi, 0]
    rc = np.hypot(mesh["Cf"][fi, 1], mesh["Cf"][fi, 2])
    o = np.argsort(-rc, kind="stable")
    u_d = uxf[o]
    u_c = float(np.max(uxf))
    prof = {"U_c_m_s": u_c, "n_faces": len(fi), "edge_rank": None, "y_edge_m": None}
    if u_c <= 0.0:
        return prof, None, None
    hit = np.nonzero(u_d >= u_c * (1.0 - EDGE_REL))[0]
    rank = int(hit[0])
    dy = (mesh["r_max"][fi[o]] - mesh["r_min"][fi[o]]) * math.cos(theta * 0.5)
    m = rank + 1
    un = u_d[:m] / u_c
    th = float(np.sum(un * (1.0 - un) * dy[:m]))
    ds = float(np.sum((1.0 - un) * dy[:m]))
    prof["edge_rank"] = rank
    prof["y_edge_m"] = float(np.sum(dy[:m]))
    return prof, th, ds


def metrics(mesh, U, p_kin, ctx):
    """The computed part of the doc: {"stations", "exit_profile", "metrics", "momentum", "reversal", "edge"} for the
    kinematic pressure field p_kin and ctx {"factor", "theta_mesh_rad", "planes" {name: x}, "rho_kg_m3", "c_m_s"}.
    Raises Refused("POST-STATION", ...) when a station has no bracketing layer or exit_plane no exact layer."""
    st = _metrics_run(mesh, U, p_kin, ctx)
    up = st["stations"]["upstream"]
    ex = st["stations"]["exit_plane"]
    down = st["stations"]["downstream"]
    rho = st["ctx"]["rho_kg_m3"]
    c_m_s = st["ctx"]["c_m_s"]
    factor = st["factor"]
    u_face = st["u_face"]
    p_face = st["p_face"]
    mesh_m = mesh
    q_in = -factor * patch_flux(mesh_m, U, "inlet")
    q_out = factor * patch_flux(mesh_m, U, "outlet")
    prof, th_exit, ds_exit = _exit_profile(mesh_m, st["le"], u_face, st["theta"])
    p0_flux_up = up["p0_flux_mean"]
    p0_flux_dn = down["p0_flux_mean"]
    h_exit = None if th_exit is None or th_exit == 0.0 else (ds_exit / th_exit if ds_exit is not None else None)
    mom, closure = _momentum_block(mesh_m, st, p_kin)
    rev, edge = _wall_block(mesh_m, U, p_kin, st, up)
    mass_imb = None if q_in <= 0.0 else abs(q_out - q_in) / q_in
    dp_kin = up["p_mean"] - ex["p_mean"]
    cd = None if (q_in <= 0.0 or dp_kin <= 0.0) else q_in / (ex["area_m2"] * math.sqrt(2.0 * dp_kin))
    dp = rho * dp_kin
    dp_loss = None if p0_flux_up is None or p0_flux_dn is None else rho * (p0_flux_up - p0_flux_dn)
    u_e = None if q_in <= 0.0 else q_in / ex["area_m2"]
    p0_axis_loss = None if q_in <= 0.0 else (up["p0_axis"] - down["p0_axis"]) / (u_e ** 2 / 2)
    nonunif = _exit_nonuniformity(mesh_m, st["le"], u_face, ex)
    u_axis_ratio = None if ex["u_mean_m_s"] == 0.0 else ex["u_axis_m_s"] / ex["u_mean_m_s"]
    mach = float(np.max(np.sqrt(np.einsum("ij,ij->i", U["internal"], U["internal"])))) / c_m_s
    vals = _metric_values(q_in, q_out, mass_imb, dp, cd, dp_loss, p0_axis_loss, nonunif, th_exit,
                          ds_exit, h_exit, u_axis_ratio, mach, closure, rev)
    computed = {"stations": st["stations"], "exit_profile": prof, "metrics": vals,
                "momentum": mom, "reversal": rev, "edge": edge}
    return computed


def _momentum_block(mesh, st, p_kin):
    """The axial momentum closure (C8): I, P on the layers nearest the upstream/downstream stations, W the wall
    pressure on every boundary face between them except inlet and outlet."""
    factor = st["factor"]
    la = st["_nearest"](st["xs"]["upstream"])
    lb = st["_nearest"](st["xs"]["downstream"])

    def moment(lay):
        fi = lay["faces"]
        a = np.abs(mesh["Sf"][fi, 0])
        uxf = st["u_face"][fi, 0]
        pf = np.asarray(st["p_face"][fi]).reshape(-1)
        return (factor * float(np.sum((pf + uxf * uxf) * a)), factor * float(np.sum(pf * a)))

    i_a, p_a = moment(la)
    i_b, p_b = moment(lb)
    x_a = la["x"]
    x_b = lb["x"]
    w_sum = 0.0
    for name, rng in mesh["patch_range"].items():
        if name in ("inlet", "outlet"):
            continue
        stf, nf, _t = rng
        pf = face_values(mesh, p_kin, name)
        cfx = mesh["Cf"][stf:stf + nf, 0]
        sel = (cfx > x_a) & (cfx < x_b)
        w_sum += factor * float(np.sum(pf[sel] * mesh["Sf"][stf:stf + nf, 0][sel]))
    residual = i_a - i_b - w_sum
    mom = {"x_a_m": x_a, "x_b_m": x_b, "I_a": i_a, "I_b": i_b, "W": w_sum, "P_a": p_a, "P_b": p_b,
           "residual": residual}
    closure = None if p_a == p_b else residual / abs(p_a - p_b)
    return mom, closure


def _wall_block(mesh, U, p_kin, st, up):
    """The wall_nozzle owner cells in x order (ties by face index): the reversal row and the edge velocity (C8)."""
    stf, nf, _t = mesh["patch_range"][WALL_PATCH]
    wf = np.arange(stf, stf + nf, dtype=np.int64)
    oc = mesh["owner"][wf]
    order = np.lexsort((wf, mesh["C"][oc, 0]))
    wall_f = wf[order]
    cells = oc[order]
    rev = U["internal"][cells, 0] < 0.0
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
    n_chg = int(np.count_nonzero(rev[1:] != rev[:-1])) if len(rev) > 1 else 0
    mags = np.sqrt(np.einsum("ij,ij->i", mesh["Sf"][wall_f], mesh["Sf"][wall_f]))
    wall_area = float(np.sum(mags))
    rev_row = {"n_wall_cells": len(cells), "n_reversed": n_rev, "n_bands": len(bands),
               "n_sign_changes": n_chg, "bands_x_m": bands,
               "area_fraction": None if wall_area == 0.0 else float(np.sum(mags[rev])) / wall_area}
    p0_core = up["p0_axis"]
    arg = 2.0 * (p0_core - p_kin["internal"][cells])
    u_edge = [None if a < 0.0 else math.sqrt(float(a)) for a in arg]
    edge = {"p0_core": p0_core, "x_m": [float(v) for v in xs_c],
            "r_wall_m": [float(v) for v in np.hypot(mesh["Cf"][wall_f, 1], mesh["Cf"][wall_f, 2])],
            "U_edge_m_s": u_edge, "n_undefined": sum(1 for v in u_edge if v is None)}
    return rev_row, edge


def edge_rows(mesh, p_kin, p0_core, patch):
    """An EDGE_KEYS row for `patch`, built like _wall_block's edge: the patch's owner cells in x order (ties by face index), U_edge = sqrt(2 (p0_core - p_owner)) None where negative.
    r_wall_m is the face-centre radius, _wall_block's own definition, so the upstream and nozzle wall
    rows join on one r(x)."""
    stf, nf, _t = mesh["patch_range"][patch]
    wf = np.arange(stf, stf + nf, dtype=np.int64)
    oc = mesh["owner"][wf]
    order = np.lexsort((wf, mesh["C"][oc, 0]))
    wall_f = wf[order]
    cells = oc[order]
    arg = 2.0 * (p0_core - p_kin["internal"][cells])
    u_edge = [None if a < 0.0 else math.sqrt(float(a)) for a in arg]
    return {"p0_core": p0_core, "x_m": [float(v) for v in mesh["C"][cells, 0]],
            "r_wall_m": [float(v) for v in np.hypot(mesh["Cf"][wall_f, 1], mesh["Cf"][wall_f, 2])],
            "U_edge_m_s": u_edge, "n_undefined": sum(1 for v in u_edge if v is None)}


def _exit_nonuniformity(mesh, le, u_face, ex):
    """(max - min) / area-mean of u_x over the exit layer's faces with r_c <= 0.8 R_e."""
    fi = le["faces"]
    rc = np.hypot(mesh["Cf"][fi, 1], mesh["Cf"][fi, 2])
    sel = rc <= NONUNIF_R * ex["R_m"]
    uxf = u_face[fi, 0][sel]
    if not len(uxf):
        return None
    a = np.abs(mesh["Sf"][fi, 0])[sel]
    mean = float(np.sum(uxf * a)) / float(np.sum(a))
    if mean <= 0.0:
        return None
    return (float(np.max(uxf)) - float(np.min(uxf))) / mean


def _metric_values(q_in, q_out, mass_imb, dp, cd, dp_loss, p0_axis_loss, nonunif, th, ds, h, uar, mach,
                   closure, rev):
    """The 16 METRICS rows in order, each {"value", "unit", "repr", "where", "reason_id", "definition"}."""
    got = {"Q_in": q_in, "Q_out": q_out, "mass_imbalance": mass_imb, "dp": dp, "Cd": cd,
           "dp_loss": dp_loss, "p0_loss_axis": p0_axis_loss, "exit_nonuniformity": nonunif,
           "theta_exit": th, "dstar_exit": ds, "H_exit": h, "u_axis_ratio_exit": uar,
           "mach_max": mach, "momentum_closure": closure, "reversal_fraction": rev["area_fraction"],
           "separation_free": 1.0 if rev["n_reversed"] == 0 else 0.0}
    out = {}
    for name, unit, where, definition in METRICS:
        v = got[name]
        out[name] = {"value": None if v is None else float(v), "unit": unit, "repr": "cfd",
                     "where": where, "reason_id": None if v is not None else UNDEFINED_ID,
                     "definition": definition}
    return out


def _refused_doc(rule, detail, time_name, inputs):
    doc = dict((k, None) for k in DOC_KEYS)
    doc["version"] = VERSION
    doc["status"] = "refused"
    doc["reason_id"] = rule
    doc["message"] = detail
    doc["time"] = time_name
    doc["inputs"] = dict(inputs)
    return doc


def momentum_turb(mesh, st, p_kin, shear):
    """The MOMENTUM_TURB_KEYS block (CAD-27): x_a, x_b, I, W, P exactly the laminar momentum block's, plus
    S = factor sum of tau_x,f |S_f| over the faces of BOTH wall patches with x_a < C_f,x < x_b, the residual
    I_a - I_b - W - S, and closure residual / |P_a - P_b| (None when P_a == P_b)."""
    mom, _c = _momentum_block(mesh, st, p_kin)
    x_a, x_b = mom["x_a_m"], mom["x_b_m"]
    s_sum = 0.0
    for name in WALLS_TURB:
        stf, nf, _t = mesh["patch_range"][name]
        cfx = mesh["Cf"][stf:stf + nf, 0]
        sel = (cfx > x_a) & (cfx < x_b)
        Sf = mesh["Sf"][stf:stf + nf]
        mag = np.sqrt(_dot3(Sf, Sf))
        s_sum += st["factor"] * float(np.sum(shear["faces"][name]["tau_x"][sel] * mag[sel]))
    residual = mom["I_a"] - mom["I_b"] - mom["W"] - s_sum
    return {"x_a_m": x_a, "x_b_m": x_b, "I_a": mom["I_a"], "I_b": mom["I_b"], "W": mom["W"], "S": s_sum,
            "P_a": mom["P_a"], "P_b": mom["P_b"], "residual": residual,
            "closure": None if mom["P_a"] == mom["P_b"] else residual / abs(mom["P_a"] - mom["P_b"])}


def _wedge_normals(mesh):
    """The two wedge sides' summed face normals, unit (the angle mesh_fidelity.scale_row measures)."""
    out = []
    for name in ("wedge_front", "wedge_back"):
        st, nf, _t = mesh["patch_range"][name]
        s = np.sum(mesh["Sf"][st:st + nf], axis=0)
        out.append(s / np.linalg.norm(s))
    return out


def _wall_vertex_rmax(mesh, names):
    """The largest vertex radius over the named wall patches."""
    vs = set()
    for name in names:
        st, nf, _t = mesh["patch_range"][name]
        for f in range(st, st + nf):
            vs.update(mesh["faces"][f])
    pts = mesh["points"][np.array(sorted(vs), dtype=np.int64)]
    return float(np.max(np.hypot(pts[:, 1], pts[:, 2])))


def _turb_input_paths(case_dir, time_name, geom_dir):
    """The turbulent inputs by key; the geom entries only for a nozzle (the pipe has no BRep)."""
    pm = os.path.join(case_dir, "constant", "polyMesh")
    paths = {"case.json": os.path.join(case_dir, "case.json"),
             "polyMesh/boundary": os.path.join(pm, "boundary"),
             "polyMesh/faces": os.path.join(pm, "faces"),
             "polyMesh/neighbour": os.path.join(pm, "neighbour"),
             "polyMesh/owner": os.path.join(pm, "owner"),
             "polyMesh/points": os.path.join(pm, "points"),
             "fields/U": os.path.join(case_dir, time_name, "U"),
             "fields/p": os.path.join(case_dir, time_name, "p"),
             "fields/nut": os.path.join(case_dir, time_name, "nut")}
    if geom_dir is not None:
        paths["geom/geom.json"] = os.path.join(geom_dir, "geom.json")
        paths["geom/tags.json"] = os.path.join(geom_dir, "tags.json")
    return paths


def _refused_turb(rule, detail, time_name, geom_dir, inputs):
    """A refused cad-post-turb/1 doc: every DOC_TURB_KEYS key, kind from geom_dir, None elsewhere."""
    doc = dict((k, None) for k in DOC_TURB_KEYS)
    doc["version"] = VERSION_TURB
    doc["status"] = "refused"
    doc["reason_id"] = rule
    doc["message"] = detail
    doc["time"] = time_name
    doc["kind"] = "nozzle" if geom_dir is not None else "pipe"
    doc["inputs"] = dict(inputs)
    return doc


def post_turb(case_dir, time_name, geom_dir=None, between_hook=None):
    """The cad-post-turb/1 doc (DOC_TURB_KEYS) of docs/16 §H.5 and §I CAD-27: the turbulent reader for
    TG0's periodic pipe (geom_dir None) and the turbulent nozzle; never raises Refused - a refusal comes
    back as the doc with status "refused". between_hook(case_dir) runs after the blocks, before the second
    hash."""
    inputs = {}
    try:
        return _post_turb(case_dir, time_name, geom_dir, between_hook, inputs)
    except Refused as r:
        return _refused_turb(r.rule, r.detail, time_name, geom_dir, inputs)


def _turb_bind(case_dir, time_name, geom_dir, kind, inputs):
    """The first POST-BIND: every input of the kind's key set hashed (unbound files None), case.json's
    version, kind-against-geom_dir, the polyMesh shas, and the nozzle's geom.json sha all checked."""
    paths = _turb_input_paths(case_dir, time_name, geom_dir)
    keys = INPUT_KEYS_TURB_NOZZLE if kind == "nozzle" else INPUT_KEYS_TURB_PIPE
    for key in keys:
        snap = common.stable_file_snapshot(paths[key])
        if snap["stable"] is not True:
            inputs[key] = None
            raise Refused("POST-BIND", "%s: not a stable regular file (unbound)" % key)
        inputs[key] = snap["sha256"]
    case = common.read_json(paths["case.json"])
    if case.get("version") != CASE_TURB_VERSION:
        raise Refused("POST-BIND", "case.json version %r is not %s" % (case.get("version"), CASE_TURB_VERSION))
    ckind = case.get("kind")
    if ckind not in ("pipe", "nozzle") or (ckind == "pipe") != (geom_dir is None):
        raise Refused("POST-BIND", "case kind %r against geom_dir %s"
                      % (ckind, "absent" if geom_dir is None else "given"))
    for n in ("boundary", "faces", "neighbour", "owner", "points"):
        key = "polyMesh/" + n
        if case["files"]["constant/polyMesh/" + n] != inputs[key]:
            raise Refused("POST-BIND", "%s: sha differs from case.json" % key)
    if kind == "nozzle" and case["inputs"]["geom.json"] != inputs["geom/geom.json"]:
        raise Refused("POST-BIND", "geom/geom.json: sha differs from case.json")
    return paths, keys, case


def _post_turb(case_dir, time_name, geom_dir, between_hook, inputs):
    """post_turb's body: the rules in order - BIND, PATCH, SCALE, FIELD, UNITS - then the blocks."""
    kind = "nozzle" if geom_dir is not None else "pipe"
    if (not isinstance(time_name, str) or not time_name or time_name in (".", "..")
            or "/" in time_name or chr(92) in time_name):
        raise Refused("POST-BIND", "time %r is not a single path component" % (time_name,))
    paths, keys, case = _turb_bind(case_dir, time_name, geom_dir, kind, inputs)
    mesh = load_mesh(os.path.join(case_dir, "constant", "polyMesh"))
    _turb_patches(mesh, kind)
    theta, factor, theta_source, mfid_block = _turb_scale(mesh, kind, geom_dir, case_dir)
    patch_sizes = dict((name, rng[1]) for name, rng in mesh["patch_range"].items())
    U = read_field(paths["fields/U"], 3, mesh["n_cells"], patch_sizes)
    p = read_field(paths["fields/p"], 1, mesh["n_cells"], patch_sizes)
    nut = read_nut(paths["fields/nut"], case_dir, case, kind, mesh["n_cells"], patch_sizes)
    dims_u = " ".join(U["dimensions"].split())
    dims_p = " ".join(p["dimensions"].split())
    dims_n = " ".join(nut["dimensions"].split())
    if dims_u != DIMS_U:
        raise Refused("POST-UNITS", "U dimensions %s, want %s" % (dims_u, DIMS_U))
    p_kind = DIMS_P.get(dims_p)
    if p_kind is None:
        raise Refused("POST-UNITS", "p dimensions %s are neither kinematic nor static" % dims_p)
    if dims_n != DIMS_NUT:
        raise Refused("POST-UNITS", "nut dimensions %s, want %s" % (dims_n, DIMS_NUT))
    op = case["operating_point"]
    if op["fluid"] not in reqs.RHO_TABLE:
        raise Refused("POST-UNITS", "fluid %r is not in the density table" % (op["fluid"],))
    rho = reqs.RHO_TABLE[op["fluid"]]
    p_kin = p
    if p_kind == "static":
        p_kin = {"dimensions": p["dimensions"], "internal": p["internal"] / rho,
                 "patches": dict((name, {"type": row["type"],
                                         "value": None if row["value"] is None else row["value"] / rho})
                                 for name, row in p["patches"].items())}
    units = {"U_dimensions": dims_u, "p_dimensions": dims_p, "p_kind": p_kind, "nut_dimensions": dims_n}
    return _turb_blocks(case, time_name, kind, inputs, paths, keys, mesh, U, nut, p_kin, op, units,
                        rho, theta, factor, theta_source, mfid_block, between_hook, case_dir)


def _turb_blocks(case, time_name, kind, inputs, paths, keys, mesh, U, nut, p_kin, op, units,
                 rho, theta, factor, theta_source, mfid_block, between_hook, case_dir):
    """The pipe or nozzle blocks (POST-STATION inside), between_hook, the second hash, and the ok doc."""
    nu = op["nu_m2_s"]
    shear = wall_shear(mesh, U, nut, nu, WALLS_TURB if kind == "nozzle" else WALLS_PIPE, factor)
    pipe_row = None
    nozzle_row = None
    if kind == "pipe":
        ctx = {"factor": factor, "nu_m2_s": nu, "g_x_m_s2": op["g_x_m_s2"], "rho_kg_m3": rho, "shear": shear}
        pipe_row = pipe_block(mesh, U, nut, ctx)
    else:
        tags = common.read_json(paths["geom/tags.json"])
        ctx = {"factor": factor, "theta_mesh_rad": theta,
               "planes": dict((row["name"], row["x"]) for row in tags["planes"]),
               "rho_kg_m3": rho, "c_m_s": op["c_m_s"]}
        computed = metrics(mesh, U, p_kin, ctx)
        nozzle_row = dict(computed)
        st = _metrics_run(mesh, U, p_kin, ctx)
        nozzle_row["momentum_turb"] = momentum_turb(mesh, st, p_kin, shear)
        x_target = ctx["planes"]["contraction_start"] - ENTRY_DI * 2.0 * st["lc"]["R"]
        lay_e = st["_nearest"](x_target)
        prof_e, th_e, ds_e = _exit_profile(mesh, lay_e, st["u_face"], theta)
        nozzle_row["entry"] = {"x_target_m": x_target, "x_m": lay_e["x"],
                               "x_inlet_m": ctx["planes"]["inlet"], "R_m": lay_e["R"],
                               "n_faces": prof_e["n_faces"], "U_c_m_s": prof_e["U_c_m_s"],
                               "edge_rank": prof_e["edge_rank"], "y_edge_m": prof_e["y_edge_m"],
                               "theta_m": th_e, "dstar_m": ds_e,
                               "H": None if not th_e else ds_e / th_e}
        nozzle_row["edge_upstream"] = edge_rows(mesh, p_kin, computed["edge"]["p0_core"],
                                                "wall_upstream")
        stf, nfwf, _t = mesh["patch_range"]["wall_nozzle"]
        wf = np.arange(stf, stf + nfwf, dtype=np.int64)
        order = np.lexsort((wf, mesh["C"][mesh["owner"][wf], 0]))
        nozzle_row["accel"] = accel(computed["edge"]["x_m"], computed["edge"]["U_edge_m_s"],
                                    [float(v) for v in shear["faces"]["wall_nozzle"]["u_tau_f"][order]],
                                    nu)
    if between_hook is not None:
        between_hook(case_dir)
    _turb_rebind(paths, keys, inputs)
    turb = case["turbulence"]
    doc = dict((k, None) for k in DOC_TURB_KEYS)
    doc["version"] = VERSION_TURB
    doc["status"] = "ok"
    doc["reason_id"] = None
    doc["message"] = ""
    doc["time"] = time_name
    doc["kind"] = kind
    doc["inputs"] = dict(inputs)
    doc["mesh_fidelity"] = mfid_block
    doc["wedge"] = {"theta_mesh_rad": theta, "factor": factor, "factor_formula": "2*pi/theta_mesh_rad",
                    "theta_source": theta_source, "sin_ratio_minus_1": math.sin(theta) / theta - 1.0}
    doc["operating_point"] = {"fluid": op["fluid"], "rho_kg_m3": rho, "nu_m2_s": op["nu_m2_s"],
                              "c_m_s": op["c_m_s"], "model": turb["model"], "wall": turb["wall"],
                              "re_tau": op["re_tau"], "u_tau_case_m_s": op["u_tau_m_s"],
                              "g_x_m_s2": op["g_x_m_s2"], "U_b_case_m_s": op["U_b_m_s"],
                              "U_exit_case_m_s": op["U_exit_m_s"], "Q_case_m3_s": op["Q_m3_s"],
                              "yplus1_apriori": turb["yplus1_apriori"]}
    doc["units"] = units
    doc["wall_shear"] = shear["rows"]
    doc["pipe"] = pipe_row
    doc["nozzle"] = nozzle_row
    return doc


def _turb_patches(mesh, kind):
    """POST-PATCH: the kind's boundary names exactly, the pipe's types by PATCHES_PIPE, the nozzle's
    wedges typed wedge and its WALLS_TURB patches wall."""
    want_names = sorted(PATCHES_PIPE if kind == "pipe" else PATCHES_TURB)
    if sorted(mesh["patch_range"]) != want_names:
        raise Refused("POST-PATCH", "patches %s are not exactly %s"
                      % (",".join(sorted(mesh["patch_range"])), ",".join(want_names)))
    if kind == "pipe":
        want_types = dict(PATCHES_PIPE)
    else:
        want_types = dict((n, "wall") for n in WALLS_TURB)
        want_types.update(wedge_front="wedge", wedge_back="wedge")
    for name, want in want_types.items():
        got = mesh["patch_range"][name][2]
        if got != want:
            raise Refused("POST-PATCH", "%s is typed %s, want %s" % (name, got, want))


def _turb_scale(mesh, kind, geom_dir, case_dir):
    """POST-SCALE: (theta, factor, theta_source, mesh_fidelity block or None). The nozzle takes theta from
    mesh_fidelity.check after its scale-1 pass; the pipe from the two wedge sides' summed face normals
    (mesh_fidelity.scale_row's angle), refused unless it equals pipe_mesh.RECIPE["theta_deg"] within
    TOL["theta_rad"] and the wall vertices' largest radius and the points' x span equal RECIPE R_m and L_m
    within TOL["scale_rel"] (relative). factor = 2 pi / theta."""
    import mesh_fidelity
    import pipe_mesh
    if kind == "nozzle":
        rep = mesh_fidelity.check(geom_dir, case_dir)
        if rep["status"] != "ok" or rep["scale"]["pass"] is not True:
            raise Refused("POST-SCALE", "mesh_fidelity %s: %s" % (rep["reason_id"], ",".join(rep["fields"])))
        theta = rep["scale"]["theta_mesh_rad"]
        block = {"status": rep["status"], "reason_id": rep["reason_id"], "fields": list(rep["fields"]),
                 "theta_mesh_rad": theta, "scale_pass": rep["scale"]["pass"],
                 "report_sha256": common.sha256_of(rep)}
        return theta, 2.0 * math.pi / theta, "mesh_fidelity", block
    nf_, nb_ = _wedge_normals(mesh)
    theta = float(math.atan2(float(np.linalg.norm(np.cross(nf_, nb_))), abs(float(np.dot(nf_, nb_)))))
    theta_want = math.radians(pipe_mesh.RECIPE["theta_deg"])
    if abs(theta - theta_want) > mesh_fidelity.TOL["theta_rad"]:
        raise Refused("POST-SCALE", "wedge normals angle %r rad, want %r within %r"
                      % (theta, theta_want, mesh_fidelity.TOL["theta_rad"]))
    rel = mesh_fidelity.TOL["scale_rel"]
    r_wall = _wall_vertex_rmax(mesh, WALLS_PIPE)
    if abs(r_wall - pipe_mesh.RECIPE["R_m"]) > rel * pipe_mesh.RECIPE["R_m"]:
        raise Refused("POST-SCALE", "wall vertex radius %r m, want %r within %r rel"
                      % (r_wall, pipe_mesh.RECIPE["R_m"], rel))
    span = float(np.max(mesh["points"][:, 0]) - np.min(mesh["points"][:, 0]))
    if abs(span - pipe_mesh.RECIPE["L_m"]) > rel * pipe_mesh.RECIPE["L_m"]:
        raise Refused("POST-SCALE", "points x span %r m, want %r within %r rel"
                      % (span, pipe_mesh.RECIPE["L_m"], rel))
    return theta, 2.0 * math.pi / theta, "wedge_normals", None


def _turb_rebind(paths, keys, inputs):
    """The second POST-BIND: any input changed while it was read is a refusal."""
    for key in keys:
        snap = common.stable_file_snapshot(paths[key])
        if snap["stable"] is not True or snap["sha256"] != inputs[key]:
            raise Refused("POST-BIND", "%s changed while it was read" % key)


def accel(x, u_edge, u_tau, nu):
    """K = -nu d(1/U_e)/dx by thwaites.sg_derivative and p = -K (U_e/u_tau)^3 (CAD-27, decisions D1/D5);
    POST-UNDEFINED (K and p None) when any U_e is None or <= 0, fewer than thwaites.SG_WINDOW stations, or
    x not strictly increasing; p is None only where u_tau is None or <= 0. Pure; lists of floats out."""
    xs = [float(v) for v in x]
    ue = [None if v is None else float(v) for v in u_edge]
    ut = [None if v is None else float(v) for v in u_tau]
    out = {"x_m": xs, "U_edge_m_s": ue, "u_tau_m_s": ut, "K": None, "K_max": None, "p": None,
           "p_min": None, "method": ACCEL_METHOD, "reason_id": UNDEFINED_ID}
    defined = (len(xs) >= thwaites.SG_WINDOW and all(v is not None and v > 0.0 for v in ue)
               and all(xs[i + 1] > xs[i] for i in range(len(xs) - 1)))
    if not defined:
        return out
    k = [-nu * float(d) for d in thwaites.sg_derivative(np.asarray(xs), [1.0 / v for v in ue])]
    ps = [None if t is None or t <= 0.0 else -k[i] * (ue[i] / t) ** 3 for i, t in enumerate(ut)]
    good = [v for v in ps if v is not None]
    out["K"] = k
    out["K_max"] = max(k)
    out["p"] = ps
    out["p_min"] = min(good) if good else None
    out["reason_id"] = None
    return out


def _radial_groups(mesh):
    """The radial groups of F7: cells sorted by |C_P| (stable), a new group when |C_P| exceeds the
    group's first by more than FLAT_TOL_M. Returns (groups, rc)."""
    rc = np.hypot(mesh["C"][:, 1], mesh["C"][:, 2])
    order = np.argsort(rc, kind="stable")
    groups = []
    start = 0
    for i in range(1, len(order) + 1):
        if i == len(order) or rc[order[i]] - rc[order[start]] > FLAT_TOL_M:
            groups.append(order[start:i])
            start = i
    return groups, rc


def pipe_block(mesh, U, nut, ctx):
    """The PIPE_KEYS block of the periodic pipe (CAD-27, F7 = decision D4; TB0-TB4): bulk velocity, the
    area-equivalent diameter, f and Re from the wall u_tau, the axis fit through the two innermost radial
    groups, the core defect, x-invariance, the wall-inward profile, the log-law band, and the rho g_x V
    force balance. ctx: factor, nu_m2_s, g_x_m_s2, rho_kg_m3, shear (wall_shear's internal dict).
    POST-STATION under 2 radial groups or when the groups are of unequal size."""
    groups, rc = _radial_groups(mesh)
    sizes = sorted(set(int(g.size) for g in groups))
    if len(groups) < 2 or len(sizes) != 1:
        raise Refused("POST-STATION", "%d radial group(s) of sizes %s"
                      % (len(groups), ",".join(str(s) for s in sizes)))
    factor = ctx["factor"]
    nu = ctx["nu_m2_s"]
    ux = U["internal"][:, 0]
    vol = mesh["V"]
    v_tot = factor * float(np.sum(vol))
    u_b = float(np.sum(ux * vol)) / float(np.sum(vol))
    u_tau = ctx["shear"]["rows"]["wall"]["u_tau_m_s"]
    stw, nfw, _t = mesh["patch_range"]["wall"]
    vs = set()
    for f in range(stw, stw + nfw):
        vs.update(mesh["faces"][f])
    pts = mesh["points"][np.array(sorted(vs), dtype=np.int64)]
    r_wf = float(np.mean(np.hypot(mesh["Cf"][stw:stw + nfw, 1], mesh["Cf"][stw:stw + nfw, 2])))
    l_span = float(np.max(mesh["points"][:, 0]) - np.min(mesh["points"][:, 0]))
    row = {"R_m": float(np.max(np.hypot(pts[:, 1], pts[:, 2]))), "L_m": l_span, "V_m3": v_tot,
           "D_m": 2.0 * math.sqrt(v_tot / (math.pi * l_span)), "R_wall_face_m": r_wf,
           "n_radial": len(groups), "U_b_m_s": u_b,
           "Re_D": None if u_b == 0.0 else u_b * (2.0 * math.sqrt(v_tot / (math.pi * l_span))) / nu,
           "u_tau_m_s": u_tau,
           "Re_tau": None if u_tau in (None, 0.0) else u_tau * math.sqrt(v_tot / (math.pi * l_span)) / nu,
           "f": None if u_tau in (None, 0.0) or u_b == 0.0 else 8.0 * (u_tau / u_b) ** 2}
    means = sorted((float(rc[g[0]]), float(np.mean(ux[g]))) for g in groups)
    row["U_axis_m_s"] = _axis_fit(means[0][1], means[1][1], means[0][0], means[1][0])
    row["core_defect"] = None if u_tau in (None, 0.0) else (row["U_axis_m_s"] - u_b) / u_tau
    devs = []
    for g in groups:
        m = float(np.mean(ux[g]))
        devs.extend(np.abs(ux[g] - m))
    row["x_invariance"] = None if u_b == 0.0 else float(np.max(devs)) / u_b
    row["profile"] = [{"r_m": r, "y_m": r_wf - r, "u_m_s": um,
                       "y_plus": None if u_tau in (None, 0.0) else (r_wf - r) * u_tau / nu,
                       "u_plus": None if u_tau in (None, 0.0) else um / u_tau}
                      for r, um in sorted(means, reverse=True)]
    if u_tau in (None, 0.0):
        row["loglaw"] = {"y_plus_min": None, "y_plus_max": None, "n_cells": 0, "dev_max": None}
    else:
        yp = (r_wf - rc) * u_tau / nu
        sel = (yp >= LOGLAW_YPLUS_MIN) & (yp <= LOGLAW_RE_TAU_FRAC * row["Re_tau"])
        n_sel = int(np.count_nonzero(sel))
        if n_sel == 0:
            row["loglaw"] = {"y_plus_min": None, "y_plus_max": None, "n_cells": 0, "dev_max": None}
        else:
            dev = np.abs(ux[sel] / u_tau - turb_integral.u_plus_log(yp[sel]))
            row["loglaw"] = {"y_plus_min": float(np.min(yp[sel])), "y_plus_max": float(np.max(yp[sel])),
                             "n_cells": n_sel, "dev_max": float(np.max(dev))}
    Sf = mesh["Sf"][stw:stw + nfw]
    mag = np.sqrt(_dot3(Sf, Sf))
    shear_f = factor * float(np.sum(ctx["shear"]["faces"]["wall"]["tau_x"] * mag))
    g_x = ctx["g_x_m_s2"]
    row["shear_force_N"] = ctx["rho_kg_m3"] * shear_f
    row["body_force_N"] = None if g_x is None else ctx["rho_kg_m3"] * g_x * v_tot
    row["balance_rel"] = None if g_x is None or row["body_force_N"] == 0.0 \
        else row["shear_force_N"] / row["body_force_N"] - 1.0
    return row


def wall_shear(mesh, U, nut, nu, names, factor):
    """The wall shear of docs/16 §H.5 (CAD-27, decisions D2/D3) per patch: tau_w,f = (nu + nut_f) |u_t,P| / d_P
    with u_t,P the owner's tangential velocity, d_P = (C_f - C_P) . n_f, and y+_f = d_P sqrt(tau_w,f) / nu.
    Returns {"rows": {name: WALL_SHEAR_KEYS row}} for the doc plus "faces" {name: {"tau_w", "tau_x",
    "u_tau_f"}} arrays for pipe_block and momentum_turb - stripped before the doc is built."""
    rows, faces = {}, {}
    for name in names:
        st, nf, _t = mesh["patch_range"][name]
        oc = mesh["owner"][st:st + nf]
        Sf = mesh["Sf"][st:st + nf]
        mag = np.sqrt(_dot3(Sf, Sf))
        n = Sf / mag[:, None]
        u_P = U["internal"][oc]
        u_t = u_P - _dot3(u_P, n)[:, None] * n
        nut_f = face_values(mesh, nut, name)
        d_P = _dot3(mesh["Cf"][st:st + nf] - mesh["C"][oc], n)
        tau_w = (nu + nut_f) * np.sqrt(_dot3(u_t, u_t)) / d_P
        tau_x = (nu + nut_f) * u_t[:, 0] / d_P
        if nf == 0:
            rows[name] = {"n_faces": 0, "area_m2": None, "tau_mean_m2_s2": None, "u_tau_m_s": None,
                          "yplus1_max": None, "yplus1_mean": None, "yplus1_frac_le1": None,
                          "method": WALL_SHEAR_METHOD}
        else:
            tau_mean = float(np.sum(mag * tau_w)) / float(np.sum(mag))
            yplus = d_P * np.sqrt(tau_w) / nu
            rows[name] = {"n_faces": int(nf), "area_m2": factor * float(np.sum(mag)),
                          "tau_mean_m2_s2": tau_mean, "u_tau_m_s": math.sqrt(tau_mean),
                          "yplus1_max": float(np.max(yplus)), "yplus1_mean": float(np.mean(yplus)),
                          "yplus1_frac_le1": float(np.count_nonzero(yplus <= YPLUS_LIMIT)) / int(nf),
                          "method": WALL_SHEAR_METHOD}
        faces[name] = {"tau_w": tau_w, "tau_x": tau_x, "u_tau_f": np.sqrt(tau_w)}
    return {"rows": rows, "faces": faces}


def post(case_dir, time_name, geom_dir, between_hook=None):
    """The cad-post/1 doc (DOC_KEYS) by the rules of (C4); never raises Refused - a refusal comes back as the doc
    with status "refused". between_hook(case_dir), a selftest hook, runs after the metrics and before the second
    hash."""
    inputs = {}
    try:
        return _post(case_dir, time_name, geom_dir, between_hook, inputs)
    except Refused as r:
        return _refused_doc(r.rule, r.detail, time_name, inputs)


def _input_paths(case_dir, time_name, geom_dir):
    pm = os.path.join(case_dir, "constant", "polyMesh")
    return {"case.json": os.path.join(case_dir, "case.json"),
            "polyMesh/boundary": os.path.join(pm, "boundary"),
            "polyMesh/faces": os.path.join(pm, "faces"),
            "polyMesh/neighbour": os.path.join(pm, "neighbour"),
            "polyMesh/owner": os.path.join(pm, "owner"),
            "polyMesh/points": os.path.join(pm, "points"),
            "fields/U": os.path.join(case_dir, time_name, "U"),
            "fields/p": os.path.join(case_dir, time_name, "p"),
            "geom/geom.json": os.path.join(geom_dir, "geom.json"),
            "geom/tags.json": os.path.join(geom_dir, "tags.json")}


def _post(case_dir, time_name, geom_dir, between_hook, inputs):
    if (not isinstance(time_name, str) or not time_name or time_name in (".", "..")
            or "/" in time_name or chr(92) in time_name):
        raise Refused("POST-BIND", "time %r is not a single path component" % (time_name,))
    paths = _input_paths(case_dir, time_name, geom_dir)
    for key in INPUT_KEYS:
        snap = common.stable_file_snapshot(paths[key])
        if snap["stable"] is not True:
            inputs[key] = None
            raise Refused("POST-BIND", "%s: not a stable regular file (unbound)" % key)
        inputs[key] = snap["sha256"]
    case = common.read_json(paths["case.json"])
    for n in ("boundary", "faces", "neighbour", "owner", "points"):
        key = "polyMesh/" + n
        if case["files"]["constant/polyMesh/" + n] != inputs[key]:
            raise Refused("POST-BIND", "%s: sha differs from case.json" % key)
    if case["inputs"]["geom.json"] != inputs["geom/geom.json"]:
        raise Refused("POST-BIND", "geom/geom.json: sha differs from case.json")
    mesh = load_mesh(os.path.join(case_dir, "constant", "polyMesh"))
    if sorted(mesh["patch_range"]) != sorted(PATCHES):
        raise Refused("POST-PATCH", "patches %s are not exactly %s"
                      % (",".join(sorted(mesh["patch_range"])), ",".join(PATCHES)))
    for side in ("wedge_front", "wedge_back"):
        if mesh["patch_range"][side][2] != "wedge":
            raise Refused("POST-PATCH", "%s is typed %s, want wedge" % (side, mesh["patch_range"][side][2]))
    import mesh_fidelity
    rep = mesh_fidelity.check(geom_dir, case_dir)
    if rep["status"] != "ok" or rep["scale"]["pass"] is not True:
        raise Refused("POST-SCALE", "mesh_fidelity %s: %s" % (rep["reason_id"], ",".join(rep["fields"])))
    theta = rep["scale"]["theta_mesh_rad"]
    factor = 2.0 * math.pi / theta
    patch_sizes = dict((name, rng[1]) for name, rng in mesh["patch_range"].items())
    U = read_field(paths["fields/U"], 3, mesh["n_cells"], patch_sizes)
    p = read_field(paths["fields/p"], 1, mesh["n_cells"], patch_sizes)
    dims_u = " ".join(U["dimensions"].split())
    dims_p = " ".join(p["dimensions"].split())
    if dims_u != DIMS_U:
        raise Refused("POST-UNITS", "U dimensions %s, want %s" % (dims_u, DIMS_U))
    p_kind = DIMS_P.get(dims_p)
    if p_kind is None:
        raise Refused("POST-UNITS", "p dimensions %s are neither kinematic nor static" % dims_p)
    fluid = case["operating_point"]["fluid"]
    if fluid not in reqs.RHO_TABLE:
        raise Refused("POST-UNITS", "fluid %r is not in the density table" % (fluid,))
    rho = reqs.RHO_TABLE[fluid]
    p_kin = p
    if p_kind == "static":
        p_kin = {"dimensions": p["dimensions"], "internal": p["internal"] / rho,
                 "patches": dict((name, {"type": row["type"],
                                         "value": None if row["value"] is None else row["value"] / rho})
                                 for name, row in p["patches"].items())}
    tags = common.read_json(paths["geom/tags.json"])
    planes = dict((row["name"], row["x"]) for row in tags["planes"])
    ctx = {"factor": factor, "theta_mesh_rad": theta, "planes": planes, "rho_kg_m3": rho,
           "c_m_s": case["operating_point"]["c_m_s"]}
    computed = metrics(mesh, U, p_kin, ctx)
    geom = common.read_json(paths["geom/geom.json"])
    st_in, nf_in, _t = mesh["patch_range"]["inlet"]
    a_mesh = factor * float(np.sum(np.sqrt(np.einsum("ij,ij->i", mesh["Sf"][st_in:st_in + nf_in],
                                                     mesh["Sf"][st_in:st_in + nf_in]))))
    a_geom = geom["tags"]["fluid_faces"]["inlet"]["area_m2"]
    op = case["operating_point"]
    if between_hook is not None:
        between_hook(case_dir)
    for key in INPUT_KEYS:
        snap = common.stable_file_snapshot(paths[key])
        if snap["stable"] is not True or snap["sha256"] != inputs[key]:
            raise Refused("POST-BIND", "%s changed while it was read" % key)
    doc = dict((k, None) for k in DOC_KEYS)
    doc["version"] = VERSION
    doc["status"] = "ok"
    doc["reason_id"] = None
    doc["message"] = ""
    doc["time"] = time_name
    doc["inputs"] = dict(inputs)
    doc["mesh_fidelity"] = {"status": rep["status"], "reason_id": rep["reason_id"],
                            "fields": list(rep["fields"]), "theta_mesh_rad": theta,
                            "scale_pass": rep["scale"]["pass"], "report_sha256": common.sha256_of(rep)}
    doc["wedge"] = {"theta_mesh_rad": theta, "factor": factor, "factor_formula": "2*pi/theta_mesh_rad",
                    "inlet_area_mesh_m2": a_mesh, "inlet_area_geom_m2": a_geom,
                    "inlet_area_rel": a_mesh / a_geom - 1.0,
                    "sin_ratio_minus_1": math.sin(theta) / theta - 1.0}
    doc["operating_point"] = {"fluid": op["fluid"], "rho_kg_m3": rho, "c_m_s": op["c_m_s"],
                              "nu_m2_s": op["nu_m2_s"], "Q_case_m3_s": op["Q_m3_s"]}
    doc["units"] = {"U_dimensions": dims_u, "p_dimensions": dims_p, "p_kind": p_kind}
    doc.update(computed)
    doc["boundary_flux"] = boundary_flux(mesh, U, factor)
    return doc


def records(doc):
    """{metric name: cad-measure/1 record} for every METRICS row, in METRICS order (C10)."""
    detail = "inputs " + common.sha256_of(doc["inputs"])
    out = {}
    for name, unit, where, definition in METRICS:
        if doc["status"] != "ok":
            out[name] = {"schema": "cad-measure/1", "primitive": name, "feature": None, "where": where,
                         "value": None, "unit": unit, "u_meas": None, "method": "cad-post/1: " + definition,
                         "status": "refused", "reason_id": doc["reason_id"], "detail": detail}
            continue
        row = doc["metrics"][name]
        undefined = row["reason_id"] == UNDEFINED_ID
        out[name] = {"schema": "cad-measure/1", "primitive": name, "feature": None, "where": where,
                     "value": None if undefined else row["value"], "unit": unit, "u_meas": None,
                     "method": "cad-post/1: " + definition,
                     "status": "refused" if undefined else "ok",
                     "reason_id": UNDEFINED_ID if undefined else None, "detail": detail}
    return out


def write_doc(path, doc):
    """canonical_json(doc) + a newline, written with common.atomic_write."""
    common.atomic_write(path, common.canonical_json(doc) + chr(10))


def _fx_tet(geom_dir, msh_path):
    """The selftest's 3-D tet child: gmsh meshes fluid.brep, physical groups name the patches (C11)."""
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.NumThreads", 1)
    gmsh.model.occ.importShapes(os.path.join(geom_dir, "fluid.brep"))
    gmsh.model.occ.synchronize()
    tags = common.read_json(os.path.join(geom_dir, "tags.json"))
    planes = dict((row["name"], row["x"]) for row in tags["planes"])
    groups = {"inlet": [], "outlet": [], "wall": []}
    for dim, tag in gmsh.model.getEntities(2):
        box = gmsh.model.getBoundingBox(dim, tag)
        if box[3] - box[0] < 1e-6 and abs(0.5 * (box[0] + box[3]) - planes["inlet"]) <= 1e-6:
            groups["inlet"].append((dim, tag))
        elif box[3] - box[0] < 1e-6 and abs(0.5 * (box[0] + box[3]) - planes["outlet"]) <= 1e-6:
            groups["outlet"].append((dim, tag))
        else:
            groups["wall"].append((dim, tag))
    for name, ents in groups.items():
        if ents:
            gmsh.model.addPhysicalGroup(2, [t for _d, t in ents], name=name)
    vols = gmsh.model.getEntities(3)
    gmsh.model.addPhysicalGroup(3, [t for _d, t in vols], name="fluid")
    gmsh.option.setNumber("Mesh.MeshSizeMax", 3e-3)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 1e-4)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.model.mesh.setSize(gmsh.model.getBoundary(groups["outlet"], recursive=True), 5e-4)
    gmsh.model.mesh.setSize(gmsh.model.getBoundary(groups["inlet"], recursive=True), 1.5e-3)
    gmsh.model.mesh.generate(3)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.write(msh_path)
    n_tets = len(gmsh.model.mesh.getElementsByType(4)[0])
    gmsh.finalize()
    print(common.canonical_json({"tets": n_tets}))
    return 0


def main(argv):
    if argv == ["--selftest"]:
        try:
            selftest()
        except Exception:
            traceback.print_exc()
            return 1
        return 0
    if len(argv) == 5 and argv[0] == "run":
        doc = post(os.path.abspath(argv[1]), argv[2], os.path.abspath(argv[3]))
        write_doc(argv[4], doc)
        print(common.canonical_json({"status": doc["status"], "reason_id": doc["reason_id"]}))
        return 0 if doc["status"] == "ok" else 1
    if len(argv) == 5 and argv[0] == "run-turb":
        geom_dir = None if argv[3] == "-" else os.path.abspath(argv[3])
        doc = post_turb(os.path.abspath(argv[1]), argv[2], geom_dir)
        write_doc(argv[4], doc)
        print(common.canonical_json({"reason_id": doc["reason_id"], "status": doc["status"]}))
        return 0 if doc["status"] == "ok" else 1
    if len(argv) == 3 and argv[0] == "fx-tet":
        return _fx_tet(argv[1], argv[2])
    print(USAGE, file=sys.stderr)
    return 2


def _fx_chain(td):
    """The nominal geometry, its L1 wedge and the case (the pattern of case_writer.py's selftest, level 1)."""
    import case_writer
    import export
    import wedge_mesh
    import reqs
    gdir = os.path.join(td, "geom")
    res = export.run_pipeline(export.TEMPLATE, dict(export.NOMINAL), gdir)
    assert res["status"] == "ok", (res["status"], res["rule"], res["message"])
    wdir = os.path.join(td, "wedge")
    res = wedge_mesh.run(gdir, wdir, levels=(1,))
    assert res["status"] == "ok", (res["status"], res["rule"], res["message"])
    sdir = os.path.join(td, "study")
    study = common.read_json(os.path.join(common.FIXTURES, "reqs", "golden",
                                          "v3_exit_velocity.json"))["requirements"]
    reqs.write_locked(sdir, study)
    cdir = os.path.join(td, "case")
    res = case_writer.write_case(wdir, 1, gdir, sdir, cdir)
    assert res["status"] == "ok", (res["status"], res["rule"], res["message"])
    return gdir, cdir, res["case"]


def _r_wall(mesh, names):
    """R_w(x): the largest vertex radius per distinct x over the named wall patches, interpolated."""
    points = mesh["points"]
    faces = mesh["faces"]
    vs = set()
    for name in names:
        st, nf, _t = mesh["patch_range"][name]
        for f in range(st, st + nf):
            vs.update(faces[f])
    v = points[np.array(sorted(vs), dtype=np.int64)]
    x = v[:, 0]
    r = np.hypot(v[:, 1], v[:, 2])
    xs = np.unique(x)
    mr = np.array([r[x == t].max() for t in xs])
    def Rw(q):
        return np.interp(q, xs, mr)
    return Rw


_A_Q = 0.5854101966249685
_B_Q = 0.1381966011250105


def _out_width(fn):
    return np.shape(fn(np.zeros((2, 3))))[-1] if np.ndim(fn(np.zeros((2, 3)))) > 1 else 1


def _face_avg(mesh, fn):
    """Per face, the fan-triangle degree-2 rule (C12): the mean of fn at the three edge midpoints, area weighted."""
    m = _out_width(fn)
    res = np.zeros((len(mesh["faces"]), m))
    for k, idxs in _face_groups(mesh["faces"]).items():
        arr, xavg, b = _fan(mesh["points"], mesh["faces"], idxs)
        tn = np.cross(arr - xavg[:, None, :], b - xavg[:, None, :])
        ta = np.sqrt(_dot3(tn, tn)) * 0.5
        area = np.zeros(len(idxs))
        for j in range(k):
            area = area + ta[:, j]
        mid = np.concatenate([((xavg[:, None, :] + arr) * 0.5).reshape(-1, 3),
                              ((arr + b) * 0.5).reshape(-1, 3),
                              ((b + xavg[:, None, :]) * 0.5).reshape(-1, 3)])
        fv = fn(mid).reshape(3, len(idxs), k, m).mean(axis=0)
        res[idxs] = np.sum(fv * ta[:, :, None], axis=1) / area[:, None]
    return res


def _cell_avg(mesh, fn):
    """Per cell, the 4-point degree-2 tet rule (C12) over (apex, xavg, a, b) tets of every face, volume weighted."""
    m = _out_width(fn)
    apex = np.zeros((mesh["n_cells"], 3))
    n_cf = np.zeros(mesh["n_cells"], dtype=np.int64)
    np.add.at(apex, mesh["owner"], mesh["Cf"])
    np.add.at(n_cf, mesh["owner"], 1)
    np.add.at(apex, mesh["neighbour"], mesh["Cf"][:mesh["n_internal"]])
    np.add.at(n_cf, mesh["neighbour"], 1)
    apex[n_cf > 0] /= n_cf[n_cf > 0, None]
    acc = np.zeros((mesh["n_cells"], m))
    for lo, hi, with_nb in ((0, mesh["n_internal"], True),
                            (mesh["n_internal"], len(mesh["faces"]), False)):
        for k, idxs in _face_groups(mesh["faces"][lo:hi]).items():
            idxs = idxs + lo
            arr, xavg, b = _fan(mesh["points"], mesh["faces"], idxs)
            cr = np.cross(arr - xavg[:, None, :], b - xavg[:, None, :])
            passes = [(mesh["owner"][idxs], 1.0)]
            if with_nb:
                passes.append((mesh["neighbour"][idxs], -1.0))
            for cells, sgn in passes:
                apx = apex[cells]
                s = apx[:, None, :] + xavg[:, None, :] + arr + b
                qs = np.empty((4, len(idxs), k, 3))
                qs[0] = _B_Q * s + (_A_Q - _B_Q) * apx[:, None, :]
                qs[1] = _B_Q * s + (_A_Q - _B_Q) * xavg[:, None, :]
                qs[2] = _B_Q * s + (_A_Q - _B_Q) * arr
                qs[3] = _B_Q * s + (_A_Q - _B_Q) * b
                fv = fn(qs.reshape(-1, 3)).reshape(4, len(idxs), k, m).mean(axis=0)
                d = xavg[:, None, :] - apx[:, None, :]
                vt = (cr[:, :, 0] * d[:, :, 0] + cr[:, :, 1] * d[:, :, 1]
                      + cr[:, :, 2] * d[:, :, 2]) / 6.0 * sgn
                np.add.at(acc, cells, np.sum(fv * vt[:, :, None], axis=1))
    return acc / mesh["V"][:, None]


def _fx_types_wedge(case):
    """Each patch's U and p types from the case's own patch rows."""
    return dict((row["name"], (row["U"]["type"], row["p"]["type"])) for row in case["patches"])


def _flat(a):
    """read_field's shapes: a scalar field is 1-D, only vectors are (n, 3)."""
    a = np.asarray(a)
    return a.reshape(-1) if a.ndim == 2 and a.shape[1] == 1 else a


def _fx_plant(mesh, fn_u, fn_p, types):
    """U and p in read_field's structure (C12): cell averages inside, face averages where the BC carries a value."""
    u_int = _flat(_cell_avg(mesh, fn_u))
    p_int = _flat(_cell_avg(mesh, fn_p))
    u_fa = _flat(_face_avg(mesh, fn_u))
    p_fa = _flat(_face_avg(mesh, fn_p))
    U = {"dimensions": DIMS_U, "internal": u_int, "patches": {}}
    p = {"dimensions": "[0 2 -2 0 0 0 0]", "internal": p_int, "patches": {}}
    for name, (ut, pt) in types.items():
        U["patches"][name] = {"type": ut,
                              "value": None if ut not in NEED_VALUE else u_fa[mesh["patch_range"][name][0]:
                                                                              mesh["patch_range"][name][0]
                                                                              + mesh["patch_range"][name][1]]}
        p["patches"][name] = {"type": pt,
                              "value": None if pt not in NEED_VALUE else p_fa[mesh["patch_range"][name][0]:
                                                                              mesh["patch_range"][name][0]
                                                                              + mesh["patch_range"][name][1]]}
    return U, p


def _fx_num(row):
    vals = [repr(float(v)) for v in np.atleast_1d(np.asarray(row))]
    if len(vals) == 3:
        return "(" + " ".join(vals) + ")"
    return vals[0]


def _fx_write_field(path, dims, cls, obj, location, internal, patches):
    """The solver's field layout (rust/src/io/fields.rs), every number repr(float) so it reads back bit-identical."""
    internal = np.asarray(internal)
    n_comp = 3 if internal.ndim > 1 and internal.shape[1] == 3 else 1
    rows = internal[:, 0] if internal.ndim > 1 and internal.shape[1] == 1 else internal
    tag = "vector" if n_comp == 3 else "scalar"
    lines = ["FoamFile", "{", "    format      ascii;", "    class       %s;" % cls,
             '    location    "%s";' % location, "    object      %s;" % obj, "}",
             "dimensions      %s;" % dims, "", "internalField   nonuniform List<%s> " % tag,
             str(len(rows)), "("]
    for row in rows:
        lines.append(_fx_num(row))
    lines += [")", ";", "", "boundaryField", "{"]
    for name, ptype, value, extra in patches:
        lines += ["    %s" % name, "    {", "        type            %s;" % ptype]
        if extra:
            lines.append(extra)
        if value is not None:
            lines += ["        value           nonuniform List<%s> " % tag, str(len(value)), "("]
            for row in np.asarray(value):
                lines.append(_fx_num(row))
            lines += [")", ";"]
        lines.append("    }")
    lines.append("}")
    common.atomic_write(path, chr(10).join(lines) + chr(10))


def _t1(cdir, gdir):
    """Geometry equals regions_check everywhere; the complete layers of the L1 wedge."""
    pm = os.path.join(cdir, "constant", "polyMesh")
    mesh = load_mesh(pm)
    points = mesh["points"]
    faces = mesh["faces"]
    d_cf = 0.0
    d_sf = 0.0
    for i in range(len(faces)):
        s, c = regions_check.face_geometry(points, faces[i])
        d_cf = max(d_cf, float(np.abs(c - mesh["Cf"][i]).max()))
        d_sf = max(d_sf, float(np.abs(s - mesh["Sf"][i]).max() / max(float(np.abs(s).max()), 1e-300)))
    assert d_cf <= 1e-15 and d_sf <= 1e-12, (d_cf, d_sf)
    ref_c, ref_v = regions_check.cell_geometry(points, faces, mesh["owner"], mesh["neighbour"], mesh["n_cells"])
    d_c = float(np.abs(ref_c - mesh["C"]).max())
    d_v = float((np.abs(ref_v - mesh["V"]) / np.abs(ref_v)).max())
    assert d_c <= 1e-15 and d_v <= 1e-12, (d_c, d_v)
    lays = layers(mesh)
    assert len(lays) == 515, len(lays)
    assert set(len(lay["faces"]) for lay in lays) == {30}, sorted(set(len(lay["faces"]) for lay in lays))
    tags = common.read_json(os.path.join(gdir, "tags.json"))
    planes = dict((q["name"], q["x"]) for q in tags["planes"])
    xs = [lay["x"] for lay in lays]
    for want in (-0.03, -0.015, 0.0, 0.06, 0.065, planes["outlet"]):
        assert any(abs(x - want) <= FLAT_TOL_M for x in xs), want
    print("[ok] geometry equals regions_check on %s faces and %d cells; %d complete layers"
          % (format(len(faces), ","), mesh["n_cells"], len(lays)))
    return mesh


_T2_LITERAL = chr(10).join([
    "FoamFile { format ascii; class volScalarField; object p; }",
    "dimensions [0 2 -2 0 0 0 0];",
    "internalField nonuniform List<scalar> 3 (1 2.5 -3e-2);",
    "boundaryField",
    "{",
    "    a { type fixedValue; value nonuniform List<scalar> 2 (4 5); }",
    "    b { type fixedValue; value nonuniform 0(); }",
    '    "c" { type zeroGradient; }',
    "    d { type inletOutlet; inletValue uniform 0; value uniform 7; }",
    "}",
])


def _read_text(text, n_comp, n_cells, patch_sizes, td, name):
    path = os.path.join(td, name)
    common.atomic_write(path, text)
    return read_field(path, n_comp, n_cells, patch_sizes)


def _t2(mesh, td):
    """The field reader: the literal compact text, a bit-identical round trip, and five POST-FIELD defects."""
    f = _read_text(_T2_LITERAL, 1, 3, {"a": 2, "b": 0, "c": 1, "d": 3}, td, "t2a")
    assert f["dimensions"] == "[0 2 -2 0 0 0 0]", f["dimensions"]
    assert np.array_equal(f["internal"], np.array([1.0, 2.5, -0.03])), f["internal"]
    assert np.array_equal(f["patches"]["a"]["value"], np.array([4.0, 5.0]))
    assert f["patches"]["b"]["value"].shape == (0,)
    assert f["patches"]["c"]["type"] == "zeroGradient" and f["patches"]["c"]["value"] is None
    assert np.array_equal(f["patches"]["d"]["value"], np.array([7.0, 7.0, 7.0]))
    rng = np.random.default_rng(7)
    internal = rng.random((mesh["n_cells"], 3))
    sizes = dict((name, r[1]) for name, r in mesh["patch_range"].items())
    vals = {}
    rows = []
    for name, r in mesh["patch_range"].items():
        ptype = {"inlet": "fixedValue", "outlet": "inletOutlet", "wall_nozzle": "noSlip",
                 "slip_upstream": "slip", "wedge_front": "wedge", "wedge_back": "wedge"}[name]
        value = None
        if ptype in NEED_VALUE:
            value = rng.random((r[1], 3))
            vals[name] = value
        extra = "        inletValue      uniform (%s);" % " ".join(repr(float(v)) for v in [0.0, 0.0, 0.0]) \
            if ptype == "inletOutlet" else None
        rows.append((name, ptype, value, extra))
    path = os.path.join(td, "t2b_U")
    _fx_write_field(path, DIMS_U, "volVectorField", "U", "7", internal, rows)
    back = read_field(path, 3, mesh["n_cells"], sizes)
    assert np.array_equal(back["internal"], internal)
    for name, value in vals.items():
        assert np.array_equal(back["patches"][name]["value"], value), name
    defects = [
        _T2_LITERAL.replace("3 (1 2.5 -3e-2)", "3 (1 2.5)"),
        _T2_LITERAL.replace("-3e-2", "nan"),
        _T2_LITERAL.replace("value nonuniform List<scalar> 2 (4 5); ", ""),
        _T2_LITERAL.replace("    d { type inletOutlet; inletValue uniform 0; value uniform 7; }" + chr(10), ""),
        _T2_LITERAL.replace("type zeroGradient", "type myBC"),
    ]
    n_def = 0
    for text in defects:
        try:
            _read_text(text, 1, 3, {"a": 2, "b": 0, "c": 1, "d": 3}, td, "t2c")
        except Refused as r:
            assert r.rule == "POST-FIELD", r.rule
            n_def += 1
    assert n_def == 5, n_def
    print("[ok] field reader: 3 forms read, round trip bit-identical, 5 defects POST-FIELD")


def _fx_planes(gdir):
    tags = common.read_json(os.path.join(gdir, "tags.json"))
    return dict((q["name"], q["x"]) for q in tags["planes"])


def _fx_rows_of(field, inlet_value=False):
    rows = []
    for name, row in field["patches"].items():
        extra = None
        if inlet_value and row["type"] == "inletOutlet":
            shape = np.shape(row["value"])[1:] if row["value"] is not None else ()
            zero = [0.0] * (shape[0] if shape else 1)
            extra = "        inletValue      uniform (%s);" % " ".join(repr(float(v)) for v in zero)
        rows.append((name, row["type"], row["value"], extra))
    return rows


def _fx_u_poise(Q, Rw):
    def fn(pt):
        x = pt[:, 0]
        r2 = pt[:, 1] ** 2 + pt[:, 2] ** 2
        r = Rw(x)
        return (2.0 * (Q / (math.pi * r ** 2)) * (1.0 - r2 / r ** 2))[:, None] * np.array([1.0, 0.0, 0.0])
    return fn


def _t3(td, gdir, cdir, case, mesh):
    """The plan's Poiseuille gate end to end: a planted analytic profile through post() on the real L1 wedge."""
    q_case = case["operating_point"]["Q_m3_s"]
    rw = _r_wall(mesh, ["wall_nozzle", "slip_upstream"])
    u_field, p_field = _fx_plant(mesh, _fx_u_poise(q_case, rw), lambda pt: (-50.0 * pt[:, 0])[:, None],
                                 _fx_types_wedge(case))
    _fx_write_field(os.path.join(cdir, "7", "U"), DIMS_U, "volVectorField", "U", "7", u_field["internal"],
                    _fx_rows_of(u_field, True))
    _fx_write_field(os.path.join(cdir, "7", "p"), "[0 2 -2 0 0 0 0]", "volScalarField", "p", "7",
                    p_field["internal"], _fx_rows_of(p_field))
    doc = post(cdir, "7", gdir)
    assert doc["status"] == "ok", (doc["status"], doc["reason_id"], doc["message"])
    assert doc["mesh_fidelity"]["status"] == "ok" and doc["mesh_fidelity"]["scale_pass"] is True
    theta = doc["wedge"]["theta_mesh_rad"]
    assert abs(theta - math.radians(5.0)) <= 1e-9, theta
    factor = doc["wedge"]["factor"]
    assert factor == 2.0 * math.pi / theta
    assert abs(doc["wedge"]["inlet_area_rel"] - doc["wedge"]["sin_ratio_minus_1"]) <= 1e-9
    qs = [doc["metrics"]["Q_in"]["value"], doc["metrics"]["Q_out"]["value"]]
    qs += [doc["stations"][n]["Q_m3_s"] for n in STATIONS]
    for q in qs:
        assert abs(q / q_case - 1.0) <= 1e-3, q / q_case - 1.0
    ex = doc["stations"]["exit_plane"]
    dn = doc["stations"]["downstream"]
    ratio = doc["metrics"]["u_axis_ratio_exit"]["value"]
    assert abs(ratio / 2.0 - 1.0) <= 5e-3, ratio
    assert abs(dn["u_axis_m_s"] / dn["u_mean_m_s"] / 2.0 - 1.0) <= 5e-3
    disc = doc["metrics"]["Q_out"]["value"] / factor * (2.0 * math.pi / math.sin(theta))
    assert abs(disc / q_case - 1.0) > 1e-3, disc / q_case - 1.0
    print("[ok] Poiseuille on the L1 wedge: Q_out/Q_case-1 = %.3e, u_axis_ratio_exit = %.5f"
          % (doc["metrics"]["Q_out"]["value"] / q_case - 1.0, ratio))
    return doc, u_field, p_field


def _t4(td, gdir, doc, case):
    """The plan's gate: wedge-scaled Q matches a 3-D tet mesh of the same fluid within 0.5 %."""
    import wedge_mesh
    msh = os.path.join(td, "tet.msh")
    tet_case = os.path.join(td, "tet_case")
    pr = subprocess.run([sys.executable, os.path.abspath(__file__), "fx-tet", gdir, msh],
                        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
                        env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    assert pr.returncode == 0, (pr.returncode, pr.stdout[-300:], pr.stderr[-300:])
    wedge_mesh.convert(wedge_mesh.load_bins(), msh, tet_case, type_args=[])
    tet = load_mesh(os.path.join(tet_case, "constant", "polyMesh"))
    assert 30000 <= tet["n_cells"] <= 150000, tet["n_cells"]
    q_case = case["operating_point"]["Q_m3_s"]
    rw = _r_wall(tet, ["wall"])
    types = {"inlet": ("fixedValue", "fixedValue"), "outlet": ("fixedValue", "zeroGradient"),
             "wall": ("noSlip", "zeroGradient")}
    u_field, _p = _fx_plant(tet, _fx_u_poise(q_case, rw), lambda pt: np.zeros((len(pt), 1)), types)
    q_out = patch_flux(tet, u_field, "outlet")
    q_in = -patch_flux(tet, u_field, "inlet")
    r_out = q_out / doc["metrics"]["Q_out"]["value"] - 1.0
    r_in = q_in / doc["metrics"]["Q_in"]["value"] - 1.0
    assert abs(r_out) <= 5e-3 and abs(r_in) <= 5e-3, (r_out, r_in)
    print("[ok] 3-D tet of the same fluid: %d cells, Q_out ratio %.3e, Q_in ratio %.3e"
          % (tet["n_cells"], r_out, r_in))
    return tet


def _fx_ctx(doc, gdir, case):
    """ctx for in-memory metrics() calls: factor, theta, planes, density and sound speed."""
    return {"factor": doc["wedge"]["factor"], "theta_mesh_rad": doc["wedge"]["theta_mesh_rad"],
            "planes": _fx_planes(gdir), "rho_kg_m3": 1.2041,
            "c_m_s": case["operating_point"]["c_m_s"]}


def _t5(mesh, doc, gdir, cdir, case):
    """The plan's boundary-layer gate: a planted Pohlhausen-style profile's theta within 1 %."""
    theta = doc["wedge"]["theta_mesh_rad"]
    ctx = _fx_ctx(doc, gdir, case)
    rw = _r_wall(mesh, ["wall_nozzle", "slip_upstream"])
    st, nf, _t = mesh["patch_range"]["outlet"]
    e2 = mesh["Cf"][st:st + nf, 1:3].mean(axis=0)
    e = e2 / math.sqrt(float(e2[0]) ** 2 + float(e2[1]) ** 2)
    delta = 5e-4
    half = math.cos(theta * 0.5)

    def fn_u(pt):
        s = (rw(pt[:, 0]) * half - (pt[:, 1] * e[0] + pt[:, 2] * e[1])) / delta
        eta = np.clip(s, 0.0, 1.0)
        f = 2.0 * eta - 2.0 * eta ** 3 + eta ** 4
        return (22.5 * f)[:, None] * np.array([1.0, 0.0, 0.0])
    u_field, p_field = _fx_plant(mesh, fn_u, lambda pt: np.zeros((len(pt), 1)), _fx_types_wedge(case))
    res = metrics(mesh, u_field, p_field, ctx)
    m = res["metrics"]
    assert abs(m["theta_exit"]["value"] / (37.0 / 315.0 * delta) - 1.0) <= 1e-2, m["theta_exit"]["value"]
    assert abs(m["dstar_exit"]["value"] / (0.3 * delta) - 1.0) <= 1e-2, m["dstar_exit"]["value"]
    assert abs(m["H_exit"]["value"] / (0.3 * 315.0 / 37.0) - 1.0) <= 2e-2, m["H_exit"]["value"]
    assert abs(res["exit_profile"]["U_c_m_s"] / 22.5 - 1.0) <= 1e-12, res["exit_profile"]["U_c_m_s"]
    print("[ok] planted boundary layer: theta err %+.2e, dstar err %+.2e, H err %+.2e"
          % (m["theta_exit"]["value"] / (37.0 / 315.0 * delta) - 1.0,
             m["dstar_exit"]["value"] / (0.3 * delta) - 1.0,
             m["H_exit"]["value"] / (0.3 * 315.0 / 37.0) - 1.0))


def _t6(mesh, doc, gdir, cdir, case):
    """The plan's reversed-band gate: planted reversed bands counted exactly."""
    ctx = _fx_ctx(doc, gdir, case)
    q_case = case["operating_point"]["Q_m3_s"]
    rw = _r_wall(mesh, ["wall_nozzle", "slip_upstream"])
    u_field, p_field = _fx_plant(mesh, _fx_u_poise(q_case, rw), lambda pt: np.zeros((len(pt), 1)),
                                 _fx_types_wedge(case))
    stf, nf, _t = mesh["patch_range"][WALL_PATCH]
    wf = np.arange(stf, stf + nf, dtype=np.int64)
    oc = mesh["owner"][wf]
    order = np.lexsort((wf, mesh["C"][oc, 0]))
    cells = oc[order]
    assert len(np.unique(cells)) == len(cells) == 900, len(cells)

    def run(ranks):
        u = u_field
        if ranks:
            internal = u_field["internal"].copy()
            for lo, hi in ranks:
                sel = cells[lo:hi + 1]
                internal[sel, 0] = -0.1 * (q_case / (math.pi * rw(mesh["C"][sel, 0]) ** 2))
            u = dict(u_field)
            u["internal"] = internal
        return metrics(mesh, u, p_field, ctx)

    row0 = run(None)["reversal"]
    assert (row0["n_reversed"], row0["n_bands"], row0["n_sign_changes"]) == (0, 0, 0), row0
    assert row0["area_fraction"] == 0.0
    assert run(None)["metrics"]["separation_free"]["value"] == 1.0
    row = run([(300, 339)])["reversal"]
    assert (row["n_reversed"], row["n_bands"], row["n_sign_changes"]) == (40, 1, 2), row
    x_lo = float(mesh["C"][cells[300], 0])
    x_hi = float(mesh["C"][cells[339], 0])
    assert row["bands_x_m"] == [[x_lo, x_hi]], (row["bands_x_m"], [[x_lo, x_hi]])
    s_ref = 0.0
    for i in range(300, 340):
        s, _c = regions_check.face_geometry(mesh["points"], mesh["faces"][wf[order][i]])
        s_ref += float(np.sqrt(s[0] ** 2 + s[1] ** 2 + s[2] ** 2))
    s_all = 0.0
    for i in range(nf):
        s, _c = regions_check.face_geometry(mesh["points"], mesh["faces"][stf + i])
        s_all += float(np.sqrt(s[0] ** 2 + s[1] ** 2 + s[2] ** 2))
    frac = row["area_fraction"]
    assert abs(frac - s_ref / s_all) <= 1e-12 * abs(s_ref / s_all), (frac, s_ref / s_all)
    assert run([(300, 339)])["metrics"]["separation_free"]["value"] == 0.0
    row2 = run([(300, 339), (600, 609)])["reversal"]
    assert (row2["n_reversed"], row2["n_bands"], row2["n_sign_changes"]) == (50, 2, 4), row2
    row3 = run([(0, 4)])["reversal"]
    assert (row3["n_reversed"], row3["n_bands"], row3["n_sign_changes"]) == (5, 1, 1), row3
    print("[ok] reversed bands counted exactly: 0, 40 (1 band), 50 (2 bands), 5 at the first wall cell")


def _t7(mesh, doc, gdir, cdir, case):
    """Planted plug-Bernoulli fields with known Cd, p0 loss, Mach, non-uniformity and edge velocity."""
    ctx = _fx_ctx(doc, gdir, case)
    q_case = case["operating_point"]["Q_m3_s"]
    c_m_s = case["operating_point"]["c_m_s"]
    rw = _r_wall(mesh, ["wall_nozzle", "slip_upstream"])
    planes = ctx["planes"]

    def u_plug(pt):
        return (q_case / (math.pi * rw(pt[:, 0]) ** 2))[:, None] * np.array([1.0, 0.0, 0.0])

    def plant(g):
        return _fx_plant(mesh, u_plug, lambda pt: (-(q_case / (math.pi * rw(pt[:, 0]) ** 2)) ** 2 / 2.0
                                                   - g * pt[:, 0])[:, None], _fx_types_wedge(case))
    u0, p0 = plant(0.0)
    res = metrics(mesh, u0, p0, ctx)
    m = res["metrics"]
    cd = m["Cd"]["value"]
    want_cd = 1.0 / math.sqrt(1.0 - 1.0 / 81.0)
    assert abs(cd / want_cd - 1.0) <= 1e-6, (cd, want_cd)
    assert abs(m["p0_loss_axis"]["value"]) <= 1e-12, m["p0_loss_axis"]["value"]
    assert m["mass_imbalance"]["value"] <= 1e-12, m["mass_imbalance"]["value"]
    mach_want = q_case / (math.pi * rw(planes["outlet"]) ** 2) / c_m_s
    assert abs(m["mach_max"]["value"] / mach_want - 1.0) <= 1e-9, (m["mach_max"]["value"], mach_want)
    assert m["exit_nonuniformity"]["value"] <= 1e-9, m["exit_nonuniformity"]["value"]
    edge = res["edge"]
    n_bad = 0
    for k, x in enumerate(edge["x_m"]):
        if x > planes["exit_plane"]:
            u1d = q_case / (math.pi * rw(x) ** 2)
            if abs(edge["U_edge_m_s"][k] / u1d - 1.0) > 1e-9:
                n_bad += 1
    assert n_bad == 0 and edge["n_undefined"] == 0, (n_bad, edge["n_undefined"])
    u50, p50 = plant(50.0)
    res50 = metrics(mesh, u50, p50, ctx)
    m50 = res50["metrics"]
    x_up = res50["stations"]["upstream"]["x_m"]
    x_dn = res50["stations"]["downstream"]["x_m"]
    u_e = res50["metrics"]["Q_in"]["value"] / res50["stations"]["exit_plane"]["area_m2"]
    want50 = 50.0 * (x_dn - x_up) / (u_e ** 2 / 2.0)
    got50 = m50["p0_loss_axis"]["value"]
    assert abs(got50 / want50 - 1.0) <= 1e-9, (got50, want50)
    assert abs(want50 - 0.015802469135802469) <= 1e-12, want50
    u_step = dict(u0)
    internal = u0["internal"].copy()
    cell_r = np.hypot(mesh["C"][:, 1], mesh["C"][:, 2])
    internal[cell_r < 5e-3, 0] = internal[cell_r < 5e-3, 0] * 1.02
    u_step["internal"] = internal
    res_step = metrics(mesh, u_step, p0, ctx)
    nunif = res_step["metrics"]["exit_nonuniformity"]["value"]
    le = layers(mesh)
    x_ex = ctx["planes"]["exit_plane"]
    lay = [l for l in le if abs(l["x"] - x_ex) <= FLAT_TOL_M][0]
    fi = lay["faces"]
    rc = np.hypot(mesh["Cf"][fi, 1], mesh["Cf"][fi, 2])
    a = np.abs(mesh["Sf"][fi, 0])
    a_in = float(np.sum(a[rc < 5e-3]))
    a_08 = float(np.sum(a[rc <= 0.8 * lay["R"]]))
    want_step = 0.02 / (1.0 + 0.02 * a_in / a_08)
    assert abs(nunif / want_step - 1.0) <= 1e-7, (nunif, want_step)
    u_hot = dict(u0)
    hot = int(np.argmax(mesh["C"][:, 0]))
    internal_hot = u0["internal"].copy()
    internal_hot[hot] = np.array([0.35 * c_m_s, 0.0, 0.0])
    u_hot["internal"] = internal_hot
    res_hot = metrics(mesh, u_hot, p0, ctx)
    mach_hot = res_hot["metrics"]["mach_max"]["value"]
    assert abs(mach_hot / 0.35 - 1.0) <= 1e-12, mach_hot
    print("[ok] plug-Bernoulli g=0: Cd %.12f (want %.16f), p0 loss %.1e, M %.6f, nonuniformity %.1e;"
          " g=50 p0 loss %.15f, step nonuniformity %.12f, hot-cell M %.12f"
          % (cd, want_cd, m["p0_loss_axis"]["value"], m["mach_max"]["value"],
             m["exit_nonuniformity"]["value"], got50, nunif, mach_hot))
    return u0, p0, res, mach_hot


def _t8(mesh, doc, gdir, cdir, case):
    """The momentum closure is exact on uniform pressure; undefined metrics are POST-UNDEFINED, never a number."""
    ctx = _fx_ctx(doc, gdir, case)
    u0, p0 = _fx_plant(mesh, lambda pt: np.zeros((len(pt), 3)), lambda pt: np.full((len(pt), 1), 7.0),
                       _fx_types_wedge(case))
    res = metrics(mesh, u0, p0, ctx)
    m = res["metrics"]
    assert abs(res["momentum"]["residual"]) <= 1e-12 * max(abs(res["momentum"]["I_a"]), 1.0), res["momentum"]
    assert abs(m["momentum_closure"]["value"]) <= 1e-12, m["momentum_closure"]["value"]
    assert abs(m["dp"]["value"]) <= 1e-14, m["dp"]["value"]
    assert m["mach_max"]["value"] == 0.0
    assert m["separation_free"]["value"] == 1.0
    for name in ("Cd", "mass_imbalance", "dp_loss", "p0_loss_axis", "exit_nonuniformity", "theta_exit",
                 "dstar_exit", "H_exit", "u_axis_ratio_exit"):
        assert m[name]["value"] is None and m[name]["reason_id"] == "POST-UNDEFINED", (name, m[name])
    stf, nf, _t = mesh["patch_range"][WALL_PATCH]
    wf = np.arange(stf, stf + nf, dtype=np.int64)
    target = wf[int(np.argmin(np.abs(mesh["Cf"][wf, 0] - 0.03)))]
    cell = int(mesh["owner"][target])
    p_mod = dict(p0)
    internal = p0["internal"].copy()
    internal[cell] = 8.0
    p_mod["internal"] = internal
    res2 = metrics(mesh, u0, p_mod, ctx)
    sf, _cf = regions_check.face_geometry(mesh["points"], mesh["faces"][target])
    factor = ctx["factor"]
    want = -factor * sf[0]
    got = res2["momentum"]["residual"]
    assert abs(got - want) <= 1e-9 * abs(want), (got, want)
    closure = res2["metrics"]["momentum_closure"]["value"]
    want_c = got / abs(res2["momentum"]["P_a"] - res2["momentum"]["P_b"])
    assert abs(closure - want_c) <= 1e-9 * abs(want_c), (closure, want_c)
    print("[ok] momentum closure exact (residual %.1e -> one-face defect %.3e) and 9 metrics POST-UNDEFINED"
          % (res["momentum"]["residual"], got))
    return res


def _t9(doc7, doc_refused, res8, sys_mach, rec_mach0, rec_mach_hot):
    """records(): schema-valid cad-measure/1 rows in METRICS order, judged by verify."""
    import schema
    import verify
    recs = records(doc7)
    names = [row[0] for row in METRICS]
    assert list(recs.keys()) == names, list(recs.keys())
    for name, unit, where, definition in METRICS:
        r = recs[name]
        assert schema.errors(r, "cad-measure/1") == [], (name, schema.errors(r, "cad-measure/1"))
        assert r["value"] == doc7["metrics"][name]["value"], name
        assert r["u_meas"] is None and r["unit"] == unit and r["where"] == where, name
        assert doc7["metrics"][name]["repr"] == "cfd"
    doc8 = {"version": VERSION, "status": "ok", "reason_id": None, "message": "", "time": "t8",
            "inputs": {"case.json": "0"}, "metrics": res8["metrics"]}
    recs8 = records(doc8)
    assert recs8["Cd"]["status"] == "refused" and recs8["Cd"]["reason_id"] == "POST-UNDEFINED" \
        and recs8["Cd"]["value"] is None
    recs_r = records(doc_refused)
    assert len(recs_r) == 16 and all(r["status"] == "refused" and r["reason_id"] == doc_refused["reason_id"]
                                     and r["value"] is None for r in recs_r.values())
    v0 = verify.judge(sys_mach, rec_mach0, cfd={"gci_fine": 0.0})
    vh = verify.judge(sys_mach, rec_mach_hot, cfd={"gci_fine": 0.0})
    assert v0["verdict"] == "pass", v0
    assert vh["verdict"] == "fail", vh
    print("[ok] records: 16 cad-measure/1 rows, verify judges SYS-MACH pass (%.4f) and fail (%.4f)"
          % (rec_mach0["value"], rec_mach_hot["value"]))


def _t10(td, gdir, cdir, case, mesh, doc7, u_field):
    """Raw values: no rounding anywhere (AST), Cd bit-equal to its formula, a Pa p file gives the same metrics."""
    src = open(os.path.abspath(__file__), encoding="utf-8").read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            bad = (isinstance(fn, ast.Name) and fn.id == "round") or \
                (isinstance(fn, ast.Attribute) and fn.attr in ("round", "around"))
            assert not bad, ast.dump(node)[:120]
    q_in = doc7["metrics"]["Q_in"]["value"]
    up = doc7["stations"]["upstream"]
    ex = doc7["stations"]["exit_plane"]
    want_cd = q_in / (ex["area_m2"] * math.sqrt(2.0 * (up["p_mean"] - ex["p_mean"])))
    assert doc7["metrics"]["Cd"]["value"] == want_cd, (doc7["metrics"]["Cd"]["value"], want_cd)
    rho = reqs.RHO_TABLE[case["operating_point"]["fluid"]]
    p8_field = _fx_plant(mesh, lambda pt: np.zeros((len(pt), 3)),
                         lambda pt: (rho * -50.0 * pt[:, 0])[:, None], _fx_types_wedge(case))[1]
    _fx_write_field(os.path.join(cdir, "8", "U"), DIMS_U, "volVectorField", "U", "8", u_field["internal"],
                    _fx_rows_of(u_field, True))
    _fx_write_field(os.path.join(cdir, "8", "p"), "[1 -1 -2 0 0 0 0]", "volScalarField", "p", "8",
                    p8_field["internal"], _fx_rows_of(p8_field))
    doc8 = post(cdir, "8", gdir)
    assert doc8["status"] == "ok", (doc8["reason_id"], doc8["message"])
    assert doc8["units"]["p_kind"] == "static" and doc8["units"]["p_dimensions"] == "[1 -1 -2 0 0 0 0]"
    assert doc8["metrics"]["Q_in"]["value"] == doc7["metrics"]["Q_in"]["value"]
    assert doc8["metrics"]["Q_out"]["value"] == doc7["metrics"]["Q_out"]["value"]
    for name in doc7["metrics"]:
        v7 = doc7["metrics"][name]["value"]
        v8 = doc8["metrics"][name]["value"]
        if v7 is None or v8 is None:
            assert v7 is None and v8 is None, name
        elif v7 == 0.0:
            assert v8 == 0.0, (name, v7, v8)
        else:
            assert abs(v8 / v7 - 1.0) <= 1e-12, (name, v7, v8)
    print("[ok] raw values: no rounding in the AST, Cd bit-equal, the Pa p file matches within 1e-12 rel")


def _fx_rebind(case_dir, names):
    """After an edit of a copied case's polyMesh files, rewrite that copy's case.json with the new shas."""
    case = common.read_json(os.path.join(case_dir, "case.json"))
    for n in names:
        case["files"]["constant/polyMesh/" + n] = common.sha256_file(
            os.path.join(case_dir, "constant", "polyMesh", n))
    common.atomic_write(os.path.join(case_dir, "case.json"), common.canonical_json(case) + chr(10))


def _fx_copy_tree(src, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    return shutil.copytree(src, dst)


def _append(path):
    with open(path, "ab") as f:
        f.write(b" ")


def _t11(td, gdir, cdir, doc7):
    """Every input bound by its stable sha (before and after) and to case.json; POST-BIND four ways."""
    pm = os.path.join(cdir, "constant", "polyMesh")
    for key, path in zip(INPUT_KEYS, [os.path.join(cdir, "case.json"), os.path.join(pm, "boundary"),
                                      os.path.join(pm, "faces"), os.path.join(pm, "neighbour"),
                                      os.path.join(pm, "owner"), os.path.join(pm, "points"),
                                      os.path.join(cdir, "7", "U"), os.path.join(cdir, "7", "p"),
                                      os.path.join(gdir, "geom.json"), os.path.join(gdir, "tags.json")]):
        assert doc7["inputs"][key] == common.stable_file_snapshot(path)["sha256"], key
    def refused(rule, doc):
        assert doc["status"] == "refused" and doc["reason_id"] == rule, (doc["reason_id"], doc["message"])
        assert sorted(doc) == sorted(DOC_KEYS) and doc["metrics"] is None, sorted(doc)
        return doc
    shutil.copytree(os.path.join(cdir, "7"), os.path.join(cdir, "9"))
    os.remove(os.path.join(cdir, "9", "p"))
    doc_a = refused("POST-BIND", post(cdir, "9", gdir))
    c10 = os.path.join(cdir, "10")
    shutil.copytree(os.path.join(cdir, "7"), c10)

    def hook(case_dir):
        _append(os.path.join(case_dir, "10", "U"))
    doc = post(cdir, "10", gdir, between_hook=hook)
    assert doc["status"] == "refused" and doc["reason_id"] == "POST-BIND", (doc["reason_id"], doc["message"])
    assert "fields/U" in doc["message"], doc["message"]
    refused("POST-BIND", doc)
    cc = _fx_copy_tree(cdir, os.path.join(td, "case_bind_c"))
    _append(os.path.join(cc, "constant", "polyMesh", "points"))
    doc = post(cc, "7", gdir)
    assert doc["status"] == "refused" and "polyMesh/points" in doc["message"], (doc["reason_id"], doc["message"])
    refused("POST-BIND", doc)
    gd = _fx_copy_tree(gdir, os.path.join(td, "geom_bind_d"))
    _append(os.path.join(gd, "geom.json"))
    doc = post(cdir, "7", gd)
    assert doc["status"] == "refused" and "geom/geom.json" in doc["message"], (doc["reason_id"], doc["message"])
    print("[ok] binding: 10 inputs sha-equal to snapshots; POST-BIND on missing p, changed U, points, geom.json")
    return doc_a


def _t12(td, gdir, cdir, case):
    """POST-PATCH, POST-SCALE, POST-UNITS x2, POST-FIELD x3, POST-STATION x2, each by id."""
    def check(doc, rule, needle=None):
        assert doc["status"] == "refused" and doc["reason_id"] == rule, (doc["reason_id"], doc["message"])
        if needle is not None:
            assert needle in doc["message"], doc["message"]
        return doc
    cc = _fx_copy_tree(cdir, os.path.join(td, "r_patch"))
    bp = os.path.join(cc, "constant", "polyMesh", "boundary")
    text = open(bp, encoding="utf-8").read()
    assert "outlet" in text
    common.atomic_write(bp, text.replace("outlet", "exitpatch", 1))
    _fx_rebind(cc, ["boundary"])
    check(post(cc, "7", gdir), "POST-PATCH")
    cc = _fx_copy_tree(cdir, os.path.join(td, "r_scale"))
    import mesh_fidelity
    mesh_fidelity._fx_scale_points(cc, 1.001)
    _fx_rebind(cc, ["boundary", "faces", "neighbour", "owner", "points"])
    doc = check(post(cc, "7", gdir), "POST-SCALE")
    assert "MFID-SCALE" in doc["message"], doc["message"]
    cc = _fx_copy_tree(cdir, os.path.join(td, "r_units_p"))
    pp = os.path.join(cc, "7", "p")
    text = open(pp, encoding="utf-8").read()
    assert "dimensions      [0 2 -2 0 0 0 0];" in text
    common.atomic_write(pp, text.replace("dimensions      [0 2 -2 0 0 0 0];",
                                         "dimensions      [1 0 0 0 0 0 0];"))
    check(post(cc, "7", gdir), "POST-UNITS")
    cc = _fx_copy_tree(cdir, os.path.join(td, "r_units_u"))
    up = os.path.join(cc, "7", "U")
    text = open(up, encoding="utf-8").read()
    assert "dimensions      [0 1 -1 0 0 0 0];" in text
    common.atomic_write(up, text.replace("dimensions      [0 1 -1 0 0 0 0];",
                                         "dimensions      [0 1 0 0 0 0 0];"))
    check(post(cc, "7", gdir), "POST-UNITS")
    cc = _fx_copy_tree(cdir, os.path.join(td, "r_field_u"))
    up = os.path.join(cc, "7", "U")
    lines = open(up, encoding="utf-8").read().split(chr(10))
    k = lines.index("(")
    del lines[k + 1]
    common.atomic_write(up, chr(10).join(lines))
    check(post(cc, "7", gdir), "POST-FIELD")
    cc = _fx_copy_tree(cdir, os.path.join(td, "r_field_nan"))
    pp = os.path.join(cc, "7", "p")
    lines = open(pp, encoding="utf-8").read().split(chr(10))
    k = lines.index("(")
    lines[k + 1] = "nan"
    common.atomic_write(pp, chr(10).join(lines))
    check(post(cc, "7", gdir), "POST-FIELD")
    cc = _fx_copy_tree(cdir, os.path.join(td, "r_field_novalue"))
    _fx_write_field(os.path.join(cc, "7", "p"), "[0 2 -2 0 0 0 0]", "volScalarField", "p", "7",
                    np.full(load_mesh(os.path.join(cdir, "constant", "polyMesh"))["n_cells"], 1.0),
                    [("inlet", "zeroGradient", None, None), ("outlet", "fixedValue", None, None),
                     ("wall_nozzle", "zeroGradient", None, None), ("slip_upstream", "zeroGradient", None, None),
                     ("wedge_front", "wedge", None, None), ("wedge_back", "wedge", None, None)])
    check(post(cc, "7", gdir), "POST-FIELD")
    for name, x_new, want in (("exit_plane", 0.0655, "exit_plane"),
                              ("contraction_start", -0.2, "upstream")):
        gd = _fx_copy_tree(gdir, os.path.join(td, "r_station_" + name))
        tags = common.read_json(os.path.join(gd, "tags.json"))
        for row in tags["planes"]:
            if row["name"] == name:
                row["x"] = x_new
        common.write_json(os.path.join(gd, "tags.json"), tags)
        doc = post(cdir, "7", gd)
        assert doc["status"] == "refused" and doc["reason_id"] == "POST-STATION", \
            (doc["reason_id"], doc["message"])
    print("[ok] refusals by id: POST-PATCH, POST-SCALE, POST-UNITS x2, POST-FIELD x3, POST-STATION x2")


def _t13(td, gdir, cdir, doc7):
    """Two runs give identical canonical bytes with no path inside; the CLI writes the same bytes."""
    text1 = common.canonical_json(post(cdir, "7", gdir))
    text2 = common.canonical_json(post(cdir, "7", gdir))
    assert text1 == text2
    assert text1 == common.canonical_json(doc7)
    for bad in (td, td.replace(chr(92), "/"), "C:"):
        assert bad not in text1, bad
    out = os.path.join(td, "out.json")
    pr = subprocess.run([sys.executable, os.path.abspath(__file__), "run", cdir, "7", gdir, out],
                        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
                        env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    assert pr.returncode == 0, (pr.returncode, pr.stdout[-200:], pr.stderr[-300:])
    with open(out, "rb") as f:
        blob = f.read()
    assert blob == (common.canonical_json(doc7) + chr(10)).encode("utf-8")
    out2 = os.path.join(td, "out2.json")
    pr = subprocess.run([sys.executable, os.path.abspath(__file__), "run", cdir, "nope", gdir, out2],
                        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
                        env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    assert pr.returncode == 1 and "POST-BIND" in pr.stdout, (pr.returncode, pr.stdout[:200])
    for argv in ([], ["run", "x"]):
        pr = subprocess.run([sys.executable, os.path.abspath(__file__)] + argv, capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=600,
                            env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        assert pr.returncode == 2, (argv, pr.returncode)
    print("[ok] determinism and the CLI: identical bytes, no path, exits 0 / 1 / 2")


def _t22(mesh, case):
    """The solver's boundary-flux rule (G1-MASS): the Symmetry family projects tangential, a prescribed
    patch's Q is its projected (or value) flux, an owner patch leaks the owner cell's flux - the leak the
    CAD-20 slip_upstream=patch typing produced, now measured by the doc's own boundary_flux block."""
    factor = 2.0 * math.pi / 0.08726646259971649
    n = mesh["n_cells"]
    types = _fx_types_wedge(case)
    U_a = {"dimensions": DIMS_U, "internal": np.tile(np.array([0.0, -1.0, 0.0]), (n, 1)),
           "patches": dict((nm, {"type": ut, "value": None}) for nm, (ut, _pt) in types.items())}

    def proj(name, u):
        st, nf, _t = mesh["patch_range"][name]
        Sf = mesh["Sf"][st:st + nf]
        nn = Sf / np.sqrt(Sf[:, 0] ** 2 + Sf[:, 1] ** 2 + Sf[:, 2] ** 2)[:, None]
        got = face_values(mesh, U_a, name)
        assert np.abs(np.sum(got * nn, axis=1)).max() <= 1e-14, name
        assert np.abs(got - (u[None, :] - np.sum(u * nn, axis=1)[:, None] * nn)).max() <= 1e-14, name

    proj("slip_upstream", np.array([0.0, -1.0, 0.0]))
    proj("wedge_front", np.array([0.0, -1.0, 0.0]))
    rows = boundary_flux(mesh, U_a, factor)["patches"]
    assert rows["slip_upstream"]["rule"] == "prescribed" \
        and abs(rows["slip_upstream"]["Q_m3_s"]) <= 1e-15, rows["slip_upstream"]
    for side in ("wedge_front", "wedge_back"):
        assert rows[side]["rule"] == "prescribed" and abs(rows[side]["Q_m3_s"]) <= 1e-15, rows[side]
    assert rows["wall_nozzle"]["rule"] == "prescribed" and rows["wall_nozzle"]["Q_m3_s"] == 0.0, \
        rows["wall_nozzle"]
    st, nf, _t = mesh["patch_range"]["slip_upstream"]
    mesh_c = dict(mesh, patch_range=dict(mesh["patch_range"], slip_upstream=(st, nf, "patch")))
    U_c = dict(U_a, patches=dict(U_a["patches"], slip_upstream={"type": "slip", "value": None}))
    q_own = boundary_flux(mesh_c, U_c, factor)["patches"]["slip_upstream"]
    want_q = -0.005653072602547545
    assert q_own["rule"] == "owner" and abs(q_own["Q_m3_s"] / want_q - 1.0) <= 1e-12, q_own
    U_d = {"dimensions": DIMS_U, "internal": np.tile(np.array([2.5, 0.0, 0.0]), (n, 1)),
           "patches": dict((nm, {"type": ut, "value": None}) for nm, (ut, _pt) in types.items())}
    for nm, ut in (("inlet", "fixedValue"), ("outlet", "inletOutlet")):
        stn, nfn, _t = mesh["patch_range"][nm]
        U_d["patches"][nm] = {"type": ut, "value": np.tile(np.array([2.5, 0.0, 0.0]), (nfn, 1))}
    bd = boundary_flux(mesh, U_d, factor)
    rd = bd["patches"]
    assert abs(bd["Q_other_m3_s"]) <= 1e-15 and abs(bd["other_rel"]) <= 1e-15, \
        (bd["Q_other_m3_s"], bd["other_rel"])
    assert bd["Q_net_m3_s"] == rd["outlet"]["Q_m3_s"] + rd["inlet"]["Q_m3_s"], bd["Q_net_m3_s"]
    assert sorted(bd) == sorted(BFLUX_KEYS), sorted(bd)
    for row in rd.values():
        assert sorted(row) == sorted(BFLUX_ROW_KEYS), sorted(row)
    print("[ok] T22 boundary_flux: slip prescribed |Q| <= 1e-15, owner Q %r (want %r), other_rel %r"
          % (q_own["Q_m3_s"], want_q, bd["other_rel"]))


def _t14(pcase, case):
    """A planted Poiseuille pipe (docs/16 §I CAD-27): f Re_D = 64 through the same f path, the oracle
    numbers, x-invariance, and one perturbed cell caught."""
    import pipe_mesh
    mesh = load_mesh(os.path.join(pcase, "constant", "polyMesh"))
    types = _fx_types_turb(case)
    r_p = pipe_mesh.RECIPE["R_m"] * math.cos(math.radians(2.5))

    def fn_u(pt):
        u = 2.0 * (1.0 - (pt[:, 1] ** 2 + pt[:, 2] ** 2) / r_p ** 2)
        return np.stack([u, np.zeros_like(u), np.zeros_like(u)], axis=1)

    U, p, nut = _fx_plant_c(mesh, fn_u, lambda pt: np.zeros(len(pt)),
                            lambda pt: np.zeros(len(pt)), types)
    _fx_write_turb(pcase, "9", U, p, nut)
    doc = post_turb(pcase, "9")
    assert doc["status"] == "ok", (doc["status"], doc["reason_id"], doc["message"])
    assert doc["kind"] == "pipe" and doc["wedge"]["theta_source"] == "wedge_normals"
    assert doc["pipe"] is not None and doc["nozzle"] is None and doc["mesh_fidelity"] is None
    pb = doc["pipe"]
    assert abs(pb["f"] * pb["Re_D"] / 64.0 - 1.0) <= 1e-3, pb["f"] * pb["Re_D"]
    assert abs(pb["f"] * pb["Re_D"] / 63.97539743517225 - 1.0) <= 1e-9
    assert abs(pb["U_b_m_s"] / 1.0005960972173311 - 1.0) <= 1e-12, pb["U_b_m_s"]
    assert abs(pb["u_tau_m_s"] / 0.0492712232359498 - 1.0) <= 1e-9
    assert abs(pb["Re_D"] / 3298.024866681143 - 1.0) <= 1e-9
    assert abs(pb["D_m"] / 0.04996827103156937 - 1.0) <= 1e-12
    assert pb["n_radial"] == 40
    assert abs(pb["U_axis_m_s"] / 2.0 - 1.0) <= 1e-12
    assert abs(pb["core_defect"] / 20.2837241932624 - 1.0) <= 1e-9
    assert pb["x_invariance"] <= 1e-12, pb["x_invariance"]
    assert len(pb["profile"]) == 40
    assert all(pb["profile"][i]["y_m"] < pb["profile"][i + 1]["y_m"] for i in range(39))
    u2 = U["internal"].copy()
    u2[87, 0] += 1e-3
    _fx_write_turb(pcase, "9", dict(U, internal=u2), p, nut)
    doc2 = post_turb(pcase, "9")
    assert abs(doc2["pipe"]["x_invariance"] / 0.0007495439047455829 - 1.0) <= 1e-6, doc2["pipe"]["x_invariance"]
    print("[ok] pipe Poiseuille: f Re_D %.9f, U_b %.13f, x_invariance %.1e -> %.9f perturbed"
          % (pb["f"] * pb["Re_D"], pb["U_b_m_s"], pb["x_invariance"], doc2["pipe"]["x_invariance"]))


def _t15(pcase, case, mesh):
    """A planted log-law pipe: u_tau back from the wall shear to 1e-12, the log-law band clean."""
    import turb_integral as ti
    types = _fx_types_turb(case)
    u_tau0 = 0.34970481600000003
    nu = case["operating_point"]["nu_m2_s"]
    stw, nfw, _t = mesh["patch_range"]["wall"]
    r_wf = float(np.mean(np.hypot(mesh["Cf"][stw:stw + nfw, 1], mesh["Cf"][stw:stw + nfw, 2])))

    def fn_u(pt):
        yp = (r_wf - np.hypot(pt[:, 1], pt[:, 2])) * u_tau0 / nu
        up = yp.copy()
        m = yp > 11.0
        up[m] = ti.u_plus_log(yp[m])
        return np.stack([u_tau0 * up, np.zeros_like(up), np.zeros_like(up)], axis=1)

    U, p, nut = _fx_plant_c(mesh, fn_u, lambda pt: np.zeros(len(pt)),
                            lambda pt: np.zeros(len(pt)), types)
    _fx_write_turb(pcase, "9", U, p, nut)
    doc = post_turb(pcase, "9")
    assert doc["status"] == "ok", (doc["status"], doc["reason_id"], doc["message"])
    ws = doc["wall_shear"]["wall"]
    assert abs(ws["u_tau_m_s"] / u_tau0 - 1.0) <= 1e-12, ws["u_tau_m_s"]
    assert abs(doc["pipe"]["Re_tau"] / 576.3240444239149 - 1.0) <= 1e-9
    assert abs(ws["yplus1_max"] / 0.12215872536815109 - 1.0) <= 1e-9
    assert ws["yplus1_frac_le1"] == 1.0
    ll = doc["pipe"]["loglaw"]
    assert ll["n_cells"] == 36 and ll["dev_max"] <= 1e-9, ll
    rc = np.hypot(mesh["C"][:, 1], mesh["C"][:, 2])
    ux = U["internal"][:, 0]
    for r in doc["pipe"]["profile"]:
        g = np.abs(rc - r["r_m"]) <= 1e-12
        assert abs(r["u_plus"] - float(np.mean(ux[g])) / u_tau0) <= 1e-9
        assert abs(r["y_plus"] / ((r_wf - r["r_m"]) * u_tau0 / nu) - 1.0) <= 1e-9
    print("[ok] pipe log law: u_tau to %.1e rel, loglaw %d cells dev %.1e"
          % (abs(ws["u_tau_m_s"] / u_tau0 - 1.0), ll["n_cells"], ll["dev_max"]))


def _t16(pcase, case, mesh):
    """A planted balanced pipe field: the wedge-scaled wall shear force equals rho g_x V."""
    types = _fx_types_turb(case)
    nu = case["operating_point"]["nu_m2_s"]
    g_x = case["operating_point"]["g_x_m_s2"]
    rho = reqs.RHO_TABLE["air"]
    stw, nfw, _t = mesh["patch_range"]["wall"]
    Sf = mesh["Sf"][stw:stw + nfw]
    mag = np.sqrt(_dot3(Sf, Sf))

    def side(name):
        st, nf, _t2 = mesh["patch_range"][name]
        s = np.sum(mesh["Sf"][st:st + nf], axis=0)
        return s / np.linalg.norm(s)

    nf_, nb_ = side("wedge_front"), side("wedge_back")
    theta = math.atan2(float(np.linalg.norm(np.cross(nf_, nb_))), abs(float(np.dot(nf_, nb_))))
    factor = 2.0 * math.pi / theta
    V = factor * float(np.sum(mesh["V"]))
    a_w = factor * float(np.sum(mag))
    tau0 = g_x * V / a_w
    oc = mesh["owner"][stw:stw + nfw]
    d_p0 = float(np.mean(_dot3(mesh["Cf"][stw:stw + nfw] - mesh["C"][oc], Sf / mag[:, None])))
    rc_wall = float(np.mean(np.hypot(mesh["C"][oc, 1], mesh["C"][oc, 2])))

    def fn_u(pt):
        hit = np.abs(np.hypot(pt[:, 1], pt[:, 2]) - rc_wall) <= 1e-12
        u = np.where(hit, tau0 * d_p0 / nu, 0.0)
        return np.stack([u, np.zeros_like(u), np.zeros_like(u)], axis=1)

    U, p, nut = _fx_plant_c(mesh, fn_u, lambda pt: np.zeros(len(pt)),
                            lambda pt: np.zeros(len(pt)), types)
    _fx_write_turb(pcase, "9", U, p, nut)
    doc = post_turb(pcase, "9")
    assert doc["status"] == "ok", (doc["status"], doc["reason_id"], doc["message"])
    pb = doc["pipe"]
    assert abs(pb["body_force_N"] / 0.0023101187039260922 - 1.0) <= 1e-12, pb["body_force_N"]
    assert abs(pb["shear_force_N"] / (rho * g_x * V) - 1.0) <= 1e-9
    assert abs(pb["balance_rel"]) <= 1e-9, pb["balance_rel"]
    assert abs(doc["wall_shear"]["wall"]["tau_mean_m2_s2"] / 0.12217706205927206 - 1.0) <= 1e-9
    print("[ok] pipe balance: shear/(rho g_x V) - 1 = %.2e, balance_rel %.2e"
          % (pb["shear_force_N"] / (rho * g_x * V) - 1.0, pb["balance_rel"]))


def _t17(ncase, gdir, case, mesh):
    """A planted 1/7-power exit layer: theta/delta = 7/72 within 0.5 % through the laminar path."""
    types = _fx_types_turb(case)
    tags = common.read_json(os.path.join(gdir, "tags.json"))
    planes = dict((q["name"], q["x"]) for q in tags["planes"])
    le = [lay for lay in layers(mesh) if abs(lay["x"] - planes["exit_plane"]) <= FLAT_TOL_M]
    assert len(le) == 1, len(le)
    r_e = le[0]["R"]
    delta = 0.1 * r_e

    def fn_u(pt):
        y = np.maximum(r_e * math.cos(math.radians(2.5)) - np.hypot(pt[:, 1], pt[:, 2]), 0.0)
        u = 50.0 * np.minimum(1.0, y / delta) ** (1.0 / 7.0)
        return np.stack([u, np.zeros_like(u), np.zeros_like(u)], axis=1)

    U, p, nut = _fx_plant_c(mesh, fn_u, lambda pt: np.zeros(len(pt)),
                            lambda pt: np.zeros(len(pt)), types)
    _fx_write_turb(ncase, "9", U, p, nut)
    doc = post_turb(ncase, "9", gdir)
    assert doc["status"] == "ok", (doc["status"], doc["reason_id"], doc["message"])
    assert doc["kind"] == "nozzle" and doc["pipe"] is None
    assert doc["mesh_fidelity"]["status"] == "ok" and doc["mesh_fidelity"]["scale_pass"] is True
    assert doc["wedge"]["theta_source"] == "mesh_fidelity"
    me = doc["nozzle"]["metrics"]
    assert abs(me["theta_exit"]["value"] / delta / (7.0 / 72.0) - 1.0) <= 5e-3
    assert abs(me["theta_exit"]["value"] / delta / 0.09704604709164305 - 1.0) <= 1e-9
    assert abs(me["H_exit"]["value"] / 1.2859604217303642 - 1.0) <= 1e-9
    print("[ok] nozzle 1/7 layer: theta/delta = %.11f (7/72 within 0.5 %%), H = %.13f"
          % (me["theta_exit"]["value"] / delta, me["H_exit"]["value"]))


def _t18(ncase, gdir, case, mesh):
    """An analytic sink flow: K = nu/30 at every station by the 1/U form, every p None at u_tau 0,
    and the accel() undefined cases."""
    types = _fx_types_turb(case)
    nu = case["operating_point"]["nu_m2_s"]

    def fn_u(pt):
        z = np.zeros(len(pt))
        return np.stack([z, z, z], axis=1)

    def fn_p(pt):
        us = 30.0 / (1.0 - pt[:, 0])
        return np.where(pt[:, 0] >= 0.0, -us * us / 2.0, 0.0)

    U, p, nut = _fx_plant_c(mesh, fn_u, fn_p, lambda pt: np.zeros(len(pt)), types)
    _fx_write_turb(ncase, "9", U, p, nut)
    doc = post_turb(ncase, "9", gdir)
    assert doc["status"] == "ok", (doc["status"], doc["reason_id"], doc["message"])
    ac = doc["nozzle"]["accel"]
    assert len(ac["x_m"]) == 60 and ac["reason_id"] is None
    k_want = nu / 30.0
    assert max(abs(k / k_want - 1.0) for k in ac["K"]) <= 1e-6, max(abs(k / k_want - 1.0) for k in ac["K"])
    assert abs(ac["K_max"] / k_want - 1.0) <= 1e-6
    assert all(v is None for v in ac["p"]) and ac["p_min"] is None
    xs = list(np.linspace(0.0, 0.5, 41))
    ue = [30.0 / (1.0 - v) for v in xs]
    a = accel(xs, ue, [1.5] * 41, nu)
    assert a["reason_id"] is None and max(abs(k / k_want - 1.0) for k in a["K"]) <= 1e-9
    assert all(abs(a["p"][i] / (-a["K"][i] * (ue[i] / 1.5) ** 3) - 1.0) <= 1e-12 for i in range(41))
    a2 = accel(xs, [None] + ue[1:], [1.5] * 41, nu)
    assert a2["reason_id"] == UNDEFINED_ID and a2["K"] is None and a2["p"] is None
    ut2 = [1.5] * 41
    ut2[7] = 0.0
    a3 = accel(xs, ue, ut2, nu)
    assert a3["reason_id"] is None and a3["p"][7] is None
    assert all(a3["p"][i] is not None for i in range(41) if i != 7)
    a4 = accel(xs[:10], ue[:10], [1.5] * 10, nu)
    assert a4["reason_id"] == UNDEFINED_ID
    print("[ok] sink flow: K = nu/30 to %.1e rel on %d stations, p None at u_tau 0, POST-UNDEFINED 3 ways"
          % (max(abs(k / k_want - 1.0) for k in ac["K"]), len(ac["x_m"])))


def _t19(ncase, gdir, case, mesh):
    """The nozzle wall-shear block and the S term inside the axial momentum residual."""
    types = _fx_types_turb(case)

    def fn_u(pt):
        u = np.where(pt[:, 0] < 0.2, 30.0, 1.0)
        return np.stack([u, np.zeros_like(u), np.zeros_like(u)], axis=1)

    U, p, nut = _fx_plant_c(mesh, fn_u, lambda pt: -50.0 * pt[:, 0],
                            lambda pt: np.zeros(len(pt)), types)
    _fx_write_turb(ncase, "9", U, p, nut)
    doc = post_turb(ncase, "9", gdir)
    assert doc["status"] == "ok", (doc["status"], doc["reason_id"], doc["message"])
    ws = doc["wall_shear"]
    noz, ups = ws["wall_nozzle"], ws["wall_upstream"]
    assert noz["n_faces"] == 60 and ups["n_faces"] == 57
    assert abs(noz["area_m2"] / 0.4349966540758174 - 1.0) <= 1e-12, noz["area_m2"]
    for got, want in ((noz["tau_mean_m2_s2"], 68.31392631410176), (noz["u_tau_m_s"], 8.265223911915621),
                      (noz["yplus1_max"], 2.3960394284990674), (noz["yplus1_mean"], 1.149147798783713),
                      (ups["u_tau_m_s"], 12.526063833801757)):
        assert abs(got / want - 1.0) <= 1e-9, (got, want)
    assert noz["yplus1_frac_le1"] == 38 / 60 and ups["yplus1_frac_le1"] == 0.0
    mt = doc["nozzle"]["momentum_turb"]
    assert mt["x_a_m"] == -0.07368421052631585 and mt["x_b_m"] == 0.5030330085889999
    assert abs(mt["S"] / 40.26774747920834 - 1.0) <= 1e-9, mt["S"]
    tol = 1e-12 * max(1.0, abs(mt["I_a"]))
    assert abs(mt["residual"] - (mt["I_a"] - mt["I_b"] - mt["W"] - mt["S"])) <= tol
    assert abs(mt["residual"] - (doc["nozzle"]["momentum"]["residual"] - mt["S"])) <= tol
    assert mt["closure"] == mt["residual"] / abs(mt["P_a"] - mt["P_b"])
    assert sorted(doc["nozzle"]) == sorted(NOZZLE_KEYS)
    print("[ok] nozzle shear: S %.9f, residual the laminar one minus S, closure residual/|dP|" % mt["S"])
    return doc


def _t20(td, gdir_lam, cdir_lam, pcase, gdir, ncase):
    """The refusals by id: the laminar case, both kind mismatches, a missing nut, a changed nut, nut
    units, a scaled pipe, a retyped wall_upstream; the laminar reader still refuses the turbulent case;
    then determinism (identical canonical bytes, no path inside) and the run-turb CLI 0 / 0 / 1 / 2."""
    import mesh_fidelity
    common.atomic_write(os.path.join(cdir_lam, "7", "nut"),
                        "FoamFile { format ascii; class volScalarField; object nut; }" + chr(10)
                        + "dimensions [0 2 -1 0 0 0 0];" + chr(10)
                        + "internalField uniform 0;" + chr(10) + "boundaryField { }" + chr(10))
    doc = post_turb(cdir_lam, "7", gdir_lam)
    assert doc["status"] == "refused" and doc["reason_id"] == "POST-BIND", (doc["reason_id"], doc["message"])
    assert "cad-case-turb/1" in doc["message"], doc["message"]
    doc = post_turb(ncase, "9")
    assert doc["reason_id"] == "POST-BIND", doc["message"]
    doc = post_turb(pcase, "9", gdir)
    assert doc["reason_id"] == "POST-BIND", doc["message"]
    c = _fx_copy_tree(ncase, os.path.join(td, "t20_nonut"))
    os.remove(os.path.join(c, "9", "nut"))
    doc = post_turb(c, "9", gdir)
    assert doc["reason_id"] == "POST-BIND" and doc["inputs"]["fields/nut"] is None, doc["message"]
    c = _fx_copy_tree(ncase, os.path.join(td, "t20_changed"))
    doc = post_turb(c, "9", gdir, between_hook=lambda d: _append(os.path.join(d, "9", "nut")))
    assert doc["reason_id"] == "POST-BIND" and "changed" in doc["message"], doc["message"]
    c = _fx_copy_tree(ncase, os.path.join(td, "t20_units"))
    pth = os.path.join(c, "9", "nut")
    with open(pth, encoding="utf-8") as f:
        text = f.read()
    common.atomic_write(pth, text.replace("[0 2 -1 0 0 0 0]", "[0 2 -2 0 0 0 0]"))
    doc = post_turb(c, "9", gdir)
    assert doc["reason_id"] == "POST-UNITS", doc["message"]
    c = _fx_copy_tree(pcase, os.path.join(td, "t20_scaled"))
    mesh_fidelity._fx_scale_points(c, 1.001)
    _fx_rebind(c, ("boundary", "faces", "neighbour", "owner", "points"))
    doc = post_turb(c, "9")
    assert doc["reason_id"] == "POST-SCALE", doc["message"]
    c = _fx_copy_tree(ncase, os.path.join(td, "t20_retyped"))
    mesh_fidelity._fx_retype(c, "wall_upstream", new_type="patch")
    _fx_rebind(c, ("boundary",))
    doc = post_turb(c, "9", gdir)
    assert doc["reason_id"] == "POST-PATCH", doc["message"]
    doc = post(ncase, "9", gdir)
    assert doc["status"] == "refused" and doc["reason_id"] == "POST-PATCH", doc["message"]
    text1 = common.canonical_json(post_turb(ncase, "9", gdir))
    text2 = common.canonical_json(post_turb(ncase, "9", gdir))
    assert text1 == text2
    for bad in (td, td.replace(chr(92), "/"), "C:"):
        assert bad not in text1, bad
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    out = os.path.join(td, "out_turb.json")
    pr = subprocess.run([sys.executable, os.path.abspath(__file__), "run-turb", ncase, "9", gdir, out],
                        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600, env=env)
    assert pr.returncode == 0, (pr.returncode, pr.stdout[-200:], pr.stderr[-300:])
    with open(out, "rb") as f:
        blob = f.read()
    assert blob == (text1 + chr(10)).encode("utf-8")
    out2 = os.path.join(td, "out_pipe.json")
    pr = subprocess.run([sys.executable, os.path.abspath(__file__), "run-turb", pcase, "9", "-", out2],
                        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600, env=env)
    assert pr.returncode == 0, (pr.returncode, pr.stdout[-200:], pr.stderr[-300:])
    pr = subprocess.run([sys.executable, os.path.abspath(__file__), "run-turb", pcase, "nope", "-", out2],
                        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600, env=env)
    assert pr.returncode == 1 and "POST-BIND" in pr.stdout, (pr.returncode, pr.stdout[:200])
    pr = subprocess.run([sys.executable, os.path.abspath(__file__), "run-turb", "x"],
                        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600, env=env)
    assert pr.returncode == 2, pr.returncode
    print("[ok] refusals, determinism, run-turb CLI: POST-BIND/POST-UNITS/POST-SCALE/POST-PATCH, "
          "identical bytes with no path inside, exits 0 / 0 / 1 (POST-BIND) / 2")


def _fx_types_turb(case):
    """Each patch's U, p and nut types from the case's own patch rows."""
    return dict((row["name"], (row["U"]["type"], row["p"]["type"], row["nut"]["type"]))
                for row in case["patches"])


def _fx_plant_c(mesh, fn_u, fn_p, fn_n, types):
    """CENTROID plants: internal = fn(C_P); a patch whose BC carries a value gets fn's face average (the
    degree-2 fan rule of _face_avg) on its faces."""
    u_fa = _flat(_face_avg(mesh, fn_u))
    p_fa = _flat(_face_avg(mesh, fn_p))
    n_fa = _flat(_face_avg(mesh, fn_n))
    U = {"dimensions": DIMS_U, "internal": fn_u(mesh["C"]), "patches": {}}
    p = {"dimensions": "[0 2 -2 0 0 0 0]", "internal": fn_p(mesh["C"]).reshape(-1), "patches": {}}
    nut = {"dimensions": DIMS_NUT, "internal": fn_n(mesh["C"]).reshape(-1), "patches": {}}
    for name, (ut, pt, nt) in types.items():
        st, nf, _t = mesh["patch_range"][name]
        U["patches"][name] = {"type": ut, "value": None if ut not in NEED_VALUE else u_fa[st:st + nf]}
        p["patches"][name] = {"type": pt, "value": None if pt not in NEED_VALUE else p_fa[st:st + nf]}
        nut["patches"][name] = {"type": nt, "value": None if nt not in NEED_VALUE else n_fa[st:st + nf]}
    return U, p, nut


def _fx_write_turb(cdir, time_name, U, p, nut):
    """U, p and nut into the time directory (created); U's inletOutlet row carries inletValue 0."""
    os.makedirs(os.path.join(cdir, time_name), exist_ok=True)
    _fx_write_field(os.path.join(cdir, time_name, "U"), DIMS_U, "volVectorField", "U", time_name,
                    U["internal"], _fx_rows_of(U, True))
    _fx_write_field(os.path.join(cdir, time_name, "p"), "[0 2 -2 0 0 0 0]", "volScalarField", "p",
                    time_name, p["internal"], _fx_rows_of(p))
    _fx_write_field(os.path.join(cdir, time_name, "nut"), DIMS_NUT, "volScalarField", "nut", time_name,
                    nut["internal"], _fx_rows_of(nut))


def _fx_turb_chain(td):
    """The L0 pipe, the TURB_NOMINAL geometry (export.py run in a FRESH child process, as case_writer.py's
    selftest does - OCCT numbers STEP products with a process-global counter), its L0 turbulent wedge at
    U_e 60, the locked turbulent study, and both cases (docs/16 §H.5, the pattern of case_writer.py)."""
    import case_writer
    import export
    import pipe_mesh
    import wedge_mesh
    pdir = os.path.join(td, "pipe")
    res = pipe_mesh.run(pdir, levels=(0,))
    assert res["status"] == "ok", (res["status"], res["rule"], res["message"])
    gdir = os.path.join(td, "geom_turb")
    pparams = os.path.join(td, "turb_params.json")
    common.write_json(pparams, dict(export.TURB_NOMINAL))
    r = subprocess.run([sys.executable, os.path.join(HERE, "export.py"), "run", export.TEMPLATE, pparams, gdir],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, (r.returncode, r.stdout[-200:], r.stderr[-300:])
    wdir = os.path.join(td, "wedge_turb")
    res = wedge_mesh.run_turb(gdir, wdir, U_e=60.0, levels=(0,))
    assert res["status"] == "ok", (res["status"], res["rule"], res["message"])
    sdir = os.path.join(td, "study")
    doc = common.read_json(os.path.join(common.FIXTURES, "reqs", "golden",
                                        "v3_exit_velocity.json"))["requirements"]
    for row_ in doc["rows"]:
        if row_["id"] == "REQ-001":
            row_["value"] = 0.30
        if row_["id"] == "REQ-002":
            row_["value"] = 0.30 / math.sqrt(2.0)
    doc["operating_point"]["U_exit_m_s"] = 60.0
    doc["operating_point"]["flow_quote"] = "at %r m/s" % 60.0
    doc["lock_sha"] = reqs.lock_sha_of(doc)
    reqs.write_locked(sdir, doc)
    pcase = os.path.join(td, "case_pipe")
    res = case_writer.write_pipe_case(pdir, 0, 576.69, pcase)
    assert res["status"] == "ok", (res["status"], res["rule"], res["message"])
    ncase = os.path.join(td, "case_nozzle")
    res = case_writer.write_turb_case(wdir, 0, gdir, sdir, ncase)
    assert res["status"] == "ok", (res["status"], res["rule"], res["message"])
    return gdir, pcase, ncase


def _wrap(inputs_shas, res):
    """An ok doc wrapper around an in-memory metrics() result, for records() in the selftest."""
    doc = dict((k, None) for k in DOC_KEYS)
    doc["version"] = VERSION
    doc["status"] = "ok"
    doc["reason_id"] = None
    doc["message"] = ""
    doc["time"] = "t"
    doc["inputs"] = inputs_shas
    doc.update(res)
    return doc


def _t21(ncase, gdir, case, mesh):
    """The entry layer nearest x = contraction_start - 0.05 D_i (the TG2a station) measured by _exit_profile, and the wall_upstream edge rows."""
    types = _fx_types_turb(case)
    tags = common.read_json(os.path.join(gdir, "tags.json"))
    planes = dict((q["name"], q["x"]) for q in tags["planes"])
    lays = layers(mesh)
    lc = [lay for lay in lays if abs(lay["x"] - planes["contraction_start"]) <= FLAT_TOL_M][0]
    x_target = planes["contraction_start"] - ENTRY_DI * 2.0 * lc["R"]
    best = min(lays, key=lambda lay: abs(lay["x"] - x_target))
    R = best["R"]
    delta = 0.1 * R

    def fn_u(pt):
        y = np.maximum(R * math.cos(math.radians(2.5)) - np.hypot(pt[:, 1], pt[:, 2]), 0.0)
        u = 30.0 * np.minimum(1.0, y / delta) ** (1.0 / 7.0)
        u = np.where(pt[:, 0] < 0.0, u, 30.0)
        return np.stack([u, np.zeros_like(u), np.zeros_like(u)], axis=1)

    U, p, nut = _fx_plant_c(mesh, fn_u, lambda pt: -50.0 * pt[:, 0],
                            lambda pt: np.zeros(len(pt)), types)
    _fx_write_turb(ncase, "9", U, p, nut)
    doc = post_turb(ncase, "9", gdir)
    assert doc["status"] == "ok", (doc["status"], doc["reason_id"], doc["message"])
    en = doc["nozzle"]["entry"]
    assert en["x_target_m"] == -0.015000000000000006, en["x_target_m"]
    assert en["x_m"] == -0.01052631578947372, en["x_m"]
    assert en["x_inlet_m"] == -0.6 and en["n_faces"] == 47, (en["x_inlet_m"], en["n_faces"])
    assert en["U_c_m_s"] == 30.0 and en["edge_rank"] == 34, (en["U_c_m_s"], en["edge_rank"])
    assert abs(en["theta_m"] / delta / 0.09689205233387911 - 1.0) <= 1e-9, en["theta_m"] / delta
    assert abs(en["H"] / 1.2858457977538857 - 1.0) <= 1e-9, en["H"]
    eu = doc["nozzle"]["edge_upstream"]
    assert eu["p0_core"] == doc["nozzle"]["edge"]["p0_core"]
    assert abs(eu["p0_core"] / 453.75 - 1.0) <= 1e-9, eu["p0_core"]
    assert len(eu["x_m"]) == 57, len(eu["x_m"])
    assert all(eu["x_m"][i] < eu["x_m"][i + 1] for i in range(len(eu["x_m"]) - 1))
    pc = eu["p0_core"]
    for u_got, x_got in zip(eu["U_edge_m_s"], eu["x_m"]):
        assert abs(u_got / math.sqrt(2.0 * (pc + 50.0 * x_got)) - 1.0) <= 1e-12, (u_got, x_got)
    assert eu["n_undefined"] == 0
    for r_got in eu["r_wall_m"]:
        assert abs(r_got / (R * math.cos(math.radians(2.5))) - 1.0) <= 1e-9, r_got
    assert sorted(doc["nozzle"]) == sorted(NOZZLE_KEYS)
    assert sorted(en) == sorted(ENTRY_KEYS)
    print("[ok] entry layer at x_target -0.015: x_m %.17f, theta/delta %.17f, H %.16f;"
          " edge_upstream 57 rows at p0_core %.13f with r_wall the layer R cos(2.5 deg)"
          % (en["x_m"], en["theta_m"] / delta, en["H"], eu["p0_core"]))


def selftest() -> None:
    """T1..T13 of docs/16 section I CAD-14: geometry, the reader, the four plan gates, the refusals, the
    CLI; T14..T20 of section I CAD-27: the turbulent pipe and nozzle through post_turb."""
    import time
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        gdir, cdir, case = _fx_chain(td)
        mesh = _t1(cdir, gdir)
        _t2(mesh, td)
        doc7, u_field, p_field = _t3(td, gdir, cdir, case, mesh)
        _t4(td, gdir, doc7, case)
        _t5(mesh, doc7, gdir, cdir, case)
        _t6(mesh, doc7, gdir, cdir, case)
        u0, p0, res0, mach_hot = _t7(mesh, doc7, gdir, cdir, case)
        res8 = _t8(mesh, doc7, gdir, cdir, case)
        doc_a = _t11(td, gdir, cdir, doc7)
        sys_mach = [c for c in common.read_json(os.path.join(common.FIXTURES, "reqs", "golden",
                                                             "v3_exit_velocity.json"))["checks"]["checks"]
                    if c["req_id"] == "SYS-MACH"][0]
        rec_mach0 = records(_wrap(doc7["inputs"], res0))["mach_max"]
        u_hot = dict(u0)
        internal = u0["internal"].copy()
        internal[int(np.argmax(mesh["C"][:, 0]))] = np.array(
            [0.35 * case["operating_point"]["c_m_s"], 0.0, 0.0])
        u_hot["internal"] = internal
        rec_mach_hot = records(_wrap(doc7["inputs"],
                                     metrics(mesh, u_hot, p0, _fx_ctx(doc7, gdir, case))))["mach_max"]
        _t9(doc7, doc_a, res8, sys_mach, rec_mach0, rec_mach_hot)
        _t10(td, gdir, cdir, case, mesh, doc7, u_field)
        _t12(td, gdir, cdir, case)
        _t13(td, gdir, cdir, doc7)
        _t22(mesh, case)
        del p_field
        ttd = os.path.join(td, "turb")
        gt, pcase, ncase = _fx_turb_chain(ttd)
        case_p = common.read_json(os.path.join(pcase, "case.json"))
        case_n = common.read_json(os.path.join(ncase, "case.json"))
        _t14(pcase, case_p)
        mesh_p = load_mesh(os.path.join(pcase, "constant", "polyMesh"))
        _t15(pcase, case_p, mesh_p)
        _t16(pcase, case_p, mesh_p)
        mesh_n = load_mesh(os.path.join(ncase, "constant", "polyMesh"))
        _t17(ncase, gt, case_n, mesh_n)
        _t18(ncase, gt, case_n, mesh_n)
        _t19(ncase, gt, case_n, mesh_n)
        _t20(ttd, gdir, cdir, pcase, gt, ncase)
        _t21(ncase, gt, case_n, mesh_n)
    print("SELFTEST PASS (%.1f s)" % (time.time() - t0))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
