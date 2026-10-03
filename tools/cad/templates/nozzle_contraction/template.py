#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""nozzle_contraction/1 - an axisymmetric subsonic contraction revolved on +x (docs/16 §H.1).

The wetted wall runs on +x through four laws, each an exact Bezier curve: Bell & Mehta's
3rd-, 5th- and 7th-order polynomials (NASA CR-177488, 1988, US Government work,
https://ntrs.nasa.gov/api/citations/19890004382/downloads/19890004382.pdf; the 5th is
r = R_i - (R_i - R_e)(10 xi^3 - 15 xi^4 + 6 xi^5) with xi = x/L) and Morel's two matched
cubics (T. Morel, J. Fluids Eng. 97(2):225-233, 1975, DOI 10.1115/1.3447255; the law
only, no chart transcribed), joined C1 at x_m with r' = 0 at both ends of both pieces.
The upstream pipe from x = -Lu to x = 0 is a slip wall or a no-slip wall by the intent
parameter upstream_role, so a turbulent boundary layer can be carried to the law start
(docs/16 section H.5 item 4).

Sections in order: params, profile, PRF, faces, solids, COMPOSE. The runner entries are
build(params, out_dir), which returns one result record and writes four BREP files, and
declare(params, out_dir), which returns DECLARATION (template.json is generated from it).

Inputs are refused by rule id before any 3-D operation, in the order PRF-BOX, PRF-RMIN,
PRF-MONO, PRF-DERIV, PRF-SELFX, PRF-FACE2D (the profile_rules list). DERIV is a property
of the law, so it is evaluated on the wetted curve BEFORE the offset and before SELFX. It
is the only rule that catches a law with a non-zero end slope: with DERIV switched off, a
cone law passes SELFX and FACE2D, because the trimmed offset closes the corner at the exit.

The body's outer wall is the TRUE normal offset of the wetted curve by t_wall: an exact
Geom_OffsetCurve, with any swallowtail loop found by sampling the distance to the wetted
curve and cut by bisection, each kept piece re-fitted as a B-spline by
GeomConvert_ApproxCurve at 1e-11 m. A raw Geom_OffsetCurve never becomes an edge: its
revolved STEP reads back as a shell, not a solid. Faces are made with
cq.Face.makeFromWires (it runs ShapeFix) and every tag is assigned geometrically, so all
of them survive a BREP reload unchanged.

This file imports only math, cadquery and OCP, as docs/16 §G admission requires; its only
outputs are the four BREP files, written through cadquery, and it has no main guard.
"""

import math

import cadquery as cq
from OCP.gp import gp_Pnt, gp_Dir, gp_Vec, gp_Pln, gp_Ax1
from OCP.Geom import Geom_BezierCurve, Geom_OffsetCurve, Geom_Line, Geom_TrimmedCurve
from OCP.TColgp import TColgp_Array1OfPnt
from OCP.GeomAbs import GeomAbs_C2
from OCP.GeomConvert import GeomConvert_ApproxCurve
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeVertex, BRepBuilderAPI_MakeFace
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.BRepPrimAPI import BRepPrimAPI_MakeRevol
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepAlgoAPI import BRepAlgoAPI_Check
from OCP.ShapeAnalysis import ShapeAnalysis_Wire
from OCP.TopExp import TopExp

# ---------------------------------------------------------------- params
TEMPLATE_ID = "nozzle_contraction/1"
N_SAMPLE = 200          # offset samples per piece when looking for a loop
N_BISECT = 60           # bisection steps for each loop cut
DELTA = 1e-12           # m: an offset point closer than t - DELTA to the wetted curve is inside a loop
JOIN_TOL = 1e-9         # m: the two cut points of one loop must coincide within this
APPROX_TOL = 1e-11      # m: GeomConvert_ApproxCurve tolerance for each kept offset piece
APPROX_MAX_ERR = 1e-10  # m: a larger reported MaxError raises
FIDELITY_TOL = 1e-9     # m: every outer sample must be at distance t from the wetted curve within this
N_FIDELITY = 101        # samples per outer edge for the fidelity check
N_MONO = 201            # samples per law piece for PRF-RMIN / PRF-MONO
DERIV_TOL = 1e-9        # dimensionless: |dr/dx| at the ends, and its jump at a junction
CLASS_TOL = 1e-10       # m: geometric tag classification
LAWS = ("poly3", "poly5", "poly7", "cubic_matched")
ROLES = ("slip", "wall")
UPSTREAM_TAG = {"slip": "slip_upstream", "wall": "wall_upstream"}
POLY = {"poly3": (0.0, 0.0, 3.0, -2.0),
        "poly5": (0.0, 0.0, 0.0, 10.0, -15.0, 6.0),
        "poly7": (0.0, 0.0, 0.0, 0.0, 35.0, -84.0, 70.0, -20.0)}   # f(xi) monomial coefficients, r = R_i - (R_i - R_e) f
RESULT_KEYS = ("template_id", "status", "rule", "detail", "params", "derived", "planes", "files",
               "face_tags", "meridian_edges", "wall_edges", "offset", "checks")
PARAMS = [
    {"name": "D_i", "kind": "real", "unit": "m", "min": None, "max": None, "default_real": 0.06,
     "choices": [], "default_choice": None, "role": "intent", "only_when": None},
    {"name": "CR", "kind": "real", "unit": "1", "min": None, "max": None, "default_real": 9.0,
     "choices": [], "default_choice": None, "role": "intent", "only_when": None},
    {"name": "L_over_Di", "kind": "real", "unit": "1", "min": 0.5, "max": 1.5, "default_real": 1.0,
     "choices": [], "default_choice": None, "role": "design", "only_when": None},
    {"name": "law", "kind": "choice", "unit": "1", "min": None, "max": None, "default_real": None,
     "choices": ["poly3", "poly5", "poly7", "cubic_matched"], "default_choice": "poly5",
     "role": "design", "only_when": None},
    {"name": "x_m", "kind": "real", "unit": "1", "min": 0.2, "max": 0.8, "default_real": 0.5,
     "choices": [], "default_choice": None, "role": "design", "only_when": "law=cubic_matched"},
    {"name": "Lx_over_De", "kind": "real", "unit": "1", "min": 0.25, "max": 1.0, "default_real": 0.5,
     "choices": [], "default_choice": None, "role": "design", "only_when": None},
    {"name": "Lu_over_Di", "kind": "real", "unit": "1", "min": 0.5, "max": 2.0, "default_real": 0.5,
     "choices": [], "default_choice": None, "role": "intent", "only_when": None},
    {"name": "upstream_role", "kind": "choice", "unit": "1", "min": None, "max": None, "default_real": None,
     "choices": ["slip", "wall"], "default_choice": "slip", "role": "intent", "only_when": None},
    {"name": "t_wall", "kind": "real", "unit": "m", "min": 0.001, "max": 0.01, "default_real": 0.003,
     "choices": [], "default_choice": None, "role": "design", "only_when": None},
]
PLANES = [{"name": "inlet", "description": "x = -Lu, the velocity inlet"},
          {"name": "contraction_start", "description": "x = 0, where the wall law starts"},
          {"name": "exit_plane", "description": "x = L, where the wall law ends"},
          {"name": "outlet", "description": "x = L + Lx, the pressure outlet"}]
TAGS = [{"name": "inlet", "kind": "face", "description": "velocity inlet disc at x = -Lu"},
        {"name": "outlet", "kind": "face", "description": "pressure outlet disc at x = L + Lx"},
        {"name": "slip_upstream", "kind": "face", "description": "slip pipe from x = -Lu to x = 0 (upstream_role slip)"},
        {"name": "wall_upstream", "kind": "face", "description": "no-slip pipe from x = -Lu to x = 0 (upstream_role wall)"},
        {"name": "wall_contraction", "kind": "face", "description": "no-slip wall of the contraction law, x = 0 to L"},
        {"name": "wall_exit", "kind": "face", "description": "no-slip exit tube wall, x = L to L + Lx"},
        {"name": "wetted", "kind": "edge", "description": "meridian wetted curve: the law, then the exit tube"},
        {"name": "outer", "kind": "edge", "description": "meridian outer wall: the true normal offset of the wetted curve by t_wall"}]
# watertight / axis / units are measured by export.py (the stl_repair report, the fluid's faces and
# geom.json); the performance rows (docs/16 §E.2) are measured by post.py from the solved case,
# never by measure.py. k_max_apriori is the a priori acceleration parameter of docs/16 §H.5 item 5
# on the 1-D area rule (turb_integral.k_max_1d) at the row's condition Re (Re_De), computed from
# the parameters, never from the BREP.
CATALOGUE = [
    {"quantity": "inlet_diameter", "primitive": "diameter_at_plane", "where": ["contraction_start"],
     "kind": "geometric", "method": "geometry", "unit": "m", "u_kind": "abs", "u_meas": 1e-9},
    {"quantity": "exit_diameter", "primitive": "diameter_at_plane", "where": ["exit_plane"],
     "kind": "geometric", "method": "geometry", "unit": "m", "u_kind": "abs", "u_meas": 1e-9},
    {"quantity": "contraction_ratio", "primitive": "area_ratio",
     "where": ["contraction_start", "exit_plane"], "kind": "geometric", "method": "geometry",
     "unit": "1", "u_kind": "rel", "u_meas": 1e-9},
    {"quantity": "total_length", "primitive": "extent_along_axis", "where": ["body"],
     "kind": "geometric", "method": "geometry", "unit": "m", "u_kind": "abs", "u_meas": 1e-9},
    {"quantity": "contraction_length", "primitive": "plane_distance",
     "where": ["contraction_start", "exit_plane"], "kind": "geometric", "method": "geometry",
     "unit": "m", "u_kind": "abs", "u_meas": 1e-9},
    {"quantity": "min_wall_normal", "primitive": "meridian_min_wall", "where": ["wetted", "outer"],
     "kind": "geometric", "method": "geometry", "unit": "m", "u_kind": "abs", "u_meas": 1e-8},
    {"quantity": "max_wall_slope", "primitive": "slope_max", "where": ["wall_contraction"],
     "kind": "geometric", "method": "geometry", "unit": "rad", "u_kind": "rel", "u_meas": 1e-6},
    {"quantity": "min_curvature_radius", "primitive": "curvature_radius_min",
     "where": ["wall_contraction"], "kind": "geometric", "method": "geometry", "unit": "m",
     "u_kind": "rel", "u_meas": 1e-6},
    {"quantity": "n_solids", "primitive": "n_solids", "where": ["fluid"], "kind": "geometric",
     "method": "geometry", "unit": "1", "u_kind": "exact", "u_meas": 0.0},
    {"quantity": "valid", "primitive": "valid", "where": ["fluid"], "kind": "geometric",
     "method": "geometry", "unit": "1", "u_kind": "exact", "u_meas": 0.0},
    {"quantity": "watertight", "primitive": "watertight", "where": ["fluid"], "kind": "geometric",
     "method": "geometry", "unit": "1", "u_kind": "exact", "u_meas": 0.0},
    {"quantity": "axis", "primitive": "axis_x", "where": ["fluid"], "kind": "geometric",
     "method": "geometry", "unit": "1", "u_kind": "exact", "u_meas": 0.0},
    {"quantity": "units", "primitive": "units_m", "where": ["fluid"], "kind": "geometric",
     "method": "geometry", "unit": "1", "u_kind": "exact", "u_meas": 0.0},
    {"quantity": "separation_free", "primitive": "separation_free",
     "where": ["wall_contraction", "wall_exit"], "kind": "performance", "method": "cfd",
     "unit": "1", "u_kind": "cfd", "u_meas": None},
    {"quantity": "exit_nonuniformity", "primitive": "exit_nonuniformity", "where": ["exit_plane"],
     "kind": "performance", "method": "cfd", "unit": "1", "u_kind": "cfd", "u_meas": None},
    {"quantity": "Cd", "primitive": "Cd", "where": ["exit_plane"], "kind": "performance",
     "method": "cfd", "unit": "1", "u_kind": "cfd", "u_meas": None},
    {"quantity": "dp_loss", "primitive": "dp_loss", "where": ["fluid"], "kind": "performance",
     "method": "cfd", "unit": "Pa", "u_kind": "cfd", "u_meas": None},
    {"quantity": "p0_loss_axis", "primitive": "p0_loss_axis", "where": ["fluid"],
     "kind": "performance", "method": "cfd", "unit": "1", "u_kind": "cfd", "u_meas": None},
    {"quantity": "theta_exit", "primitive": "theta_exit", "where": ["exit_plane"],
     "kind": "performance", "method": "cfd", "unit": "m", "u_kind": "cfd", "u_meas": None},
    {"quantity": "mach_max", "primitive": "mach_max", "where": ["fluid"], "kind": "performance",
     "method": "cfd", "unit": "1", "u_kind": "cfd", "u_meas": None},
    {"quantity": "mass_imbalance", "primitive": "mass_imbalance", "where": ["fluid"],
     "kind": "performance", "method": "cfd", "unit": "1", "u_kind": "cfd", "u_meas": None},
    {"quantity": "k_max_apriori", "primitive": "k_max_1d", "where": ["wall_contraction"],
     "kind": "geometric", "method": "geometry", "unit": "1", "u_kind": "rel", "u_meas": 1e-9},
]
PROFILE_RULES = ["PRF-BOX", "PRF-RMIN", "PRF-MONO", "PRF-DERIV", "PRF-SELFX", "PRF-FACE2D"]
STANDARDS = []   # docs/16a §F: the frozen table a row sourced standard must cite (REQ-STD); none for the nozzle
DECLARATION = {"schema": "cad-template/1", "template_id": TEMPLATE_ID,
               "title": "axisymmetric subsonic contraction on +x", "axis": "+x", "units": "m",
               "params": PARAMS, "planes": PLANES, "tags": TAGS, "catalogue": CATALOGUE,
               "profile_rules": PROFILE_RULES, "standards": STANDARDS}


def domain_rules(p):
    """The first refusing rule for a raw params dict, in the order PRF-BOX, PRF-RMIN; None accepts."""
    if not isinstance(p, dict):
        return ("PRF-BOX", "parameters missing: [%s]; unknown: []"
                % ", ".join(row["name"] for row in PARAMS))
    names = [row["name"] for row in PARAMS]
    missing = [n for n in names if n not in p]
    unknown = sorted(k for k in p if k not in set(names))
    if missing or unknown:
        return ("PRF-BOX", "parameters missing: [%s]; unknown: [%s]"
                % (", ".join(missing), ", ".join(unknown)))
    if not isinstance(p["law"], str) or p["law"] not in LAWS:
        return ("PRF-BOX", "law %r is not one of poly3, poly5, poly7, cubic_matched" % (p["law"],))
    if not isinstance(p["upstream_role"], str) or p["upstream_role"] not in ROLES:
        return ("PRF-BOX", "upstream_role %r is not one of slip, wall" % (p["upstream_role"],))
    for row in PARAMS:
        if row["kind"] != "real":
            continue
        name, mn, mx = row["name"], row["min"], row["max"]
        if row["only_when"] is not None:
            key, want = row["only_when"].split("=", 1)
            if p.get(key) != want:
                continue
        v = p[name]
        if not isinstance(v, bool) and isinstance(v, (int, float)) and not math.isfinite(v):
            return ("PRF-BOX", "%s = %r is not a finite number" % (name, v))
        if (isinstance(v, bool) or not isinstance(v, (int, float))
                or (mn is not None and v < mn) or (mx is not None and v > mx)):
            return ("PRF-BOX", "%s = %r is outside [%r, %r]" % (name, v, mn, mx))
    if p["D_i"] <= 0:
        return ("PRF-RMIN", "R_i = D_i / 2 must be > 0, got D_i = %r" % (p["D_i"],))
    if p["CR"] <= 0:
        return ("PRF-RMIN", "R_e = R_i / sqrt(CR) needs CR > 0, got CR = %r" % (p["CR"],))
    return None


def resolved(p):
    """A params dict with every real a float, and x_m None unless law is cubic_matched."""
    q = dict(p)
    for row in PARAMS:
        if row["kind"] != "real":
            continue
        if row["name"] == "x_m":
            q["x_m"] = float(p["x_m"]) if p["law"] == "cubic_matched" else None
        else:
            q[row["name"]] = float(p[row["name"]])
    return q


def derived(p):
    """The ten derived lengths of the template, from resolved params."""
    r_i = p["D_i"] / 2.0
    r_e = r_i / math.sqrt(p["CR"])
    return {"R_i": r_i, "R_e": r_e, "D_e": 2.0 * r_e, "L": p["L_over_Di"] * p["D_i"],
            "Lx": p["Lx_over_De"] * 2.0 * r_e, "Lu": p["Lu_over_Di"] * p["D_i"], "t": p["t_wall"],
            "x_inlet": -p["Lu_over_Di"] * p["D_i"], "x_outlet": p["L_over_Di"] * p["D_i"] + p["Lx_over_De"] * 2.0 * r_e,
            "x_m": p["x_m"]}


def planes(d):
    return [{"name": "inlet", "x": d["x_inlet"]}, {"name": "contraction_start", "x": 0.0},
            {"name": "exit_plane", "x": d["L"]}, {"name": "outlet", "x": d["x_outlet"]}]


# ---------------------------------------------------------------- profile
def bernstein(a):
    """Monomial coefficients a[0..n] of a polynomial on [0, 1] -> its Bernstein coefficients."""
    n = len(a) - 1
    return [sum(math.comb(j, k) / math.comb(n, k) * a[k] for k in range(j + 1)) for j in range(n + 1)]


def bezier(points):
    """A Geom_BezierCurve in the plane z = 0 through the given (x, y) poles."""
    arr = TColgp_Array1OfPnt(1, len(points))
    for i, (x, y) in enumerate(points):
        arr.SetValue(i + 1, gp_Pnt(float(x), float(y), 0.0))
    return Geom_BezierCurve(arr)


def law_pieces(p):
    """The wall law as (xi0, xi1, a) pieces: f = sum a_k s^k on s in [0, 1], xi = xi0 + (xi1 - xi0) s."""
    if p["law"] == "cubic_matched":
        x = p["x_m"]
        return [(0.0, x, (0.0, 0.0, 0.0, x)), (x, 1.0, (x, 3.0 * (1.0 - x), -3.0 * (1.0 - x), 1.0 - x))]
    return [(0.0, 1.0, POLY[p["law"]])]


def law_curves(p, d):
    """Each law piece as an exact Geom_BezierCurve: x linear in the parameter, r from its Bernstein poles."""
    out = []
    for xi0, xi1, a in law_pieces(p):
        b = bernstein(a)
        n = len(a) - 1
        poles = [(d["L"] * (xi0 + (xi1 - xi0) * j / n), d["R_i"] - (d["R_i"] - d["R_e"]) * b[j])
                 for j in range(n + 1)]
        out.append(bezier(poles))
    return out


def vertex(x, y):
    """A new TopoDS_Vertex at (x, y, 0)."""
    return BRepBuilderAPI_MakeVertex(gp_Pnt(float(x), float(y), 0.0)).Vertex()


def first_vertex(edge):
    return TopExp.FirstVertex_s(edge.wrapped)


def last_vertex(edge):
    return TopExp.LastVertex_s(edge.wrapped)


def line_between(va, vb):
    """A straight edge whose two ends ARE the given vertices (shared, never copied)."""
    mk = BRepBuilderAPI_MakeEdge(va, vb)
    if not mk.IsDone():
        raise RuntimeError("line between shared vertices failed: error %r" % (mk.Error(),))
    return cq.Edge(mk.Edge())


def curve_between(curve, va, vb, u0, u1):
    """The curve on [u0, u1] as an edge whose two ends ARE the given vertices."""
    mk = BRepBuilderAPI_MakeEdge(curve, va, vb, u0, u1)
    if not mk.IsDone():
        raise RuntimeError("curve between shared vertices failed: error %r" % (mk.Error(),))
    return cq.Edge(mk.Edge())


def wetted_edges(d, curves):
    """The meridian wetted curve: the law curves, then the straight exit tube to the outlet plane.

    Adjacent edges SHARE one vertex object at every junction. Two separate vertices a few ulp
    apart make the wire builder keep one of them by an order that changes between processes,
    and then the BREP bytes change too (docs/16 §H.3 GC-4).
    """
    vs = [vertex(0.0, d["R_i"])]
    for c in curves[:-1]:
        q = c.Value(1.0)
        vs.append(vertex(q.X(), q.Y()))
    vs.append(vertex(d["L"], d["R_e"]))
    edges = [curve_between(c, vs[i], vs[i + 1], 0.0, 1.0) for i, c in enumerate(curves)]
    edges.append(line_between(vs[-1], vertex(d["x_outlet"], d["R_e"])))
    return edges


def fluid_edges(d, wall):
    """The closed fluid meridian wire: inlet, slip pipe, the wall, exit tube, outlet, axis."""
    a, b = vertex(d["x_inlet"], 0.0), vertex(d["x_inlet"], d["R_i"])
    e, f = vertex(d["x_outlet"], d["R_e"]), vertex(d["x_outlet"], 0.0)
    edges = [line_between(a, b), line_between(b, first_vertex(wall[0]))]
    edges.extend(wall)
    edges.append(line_between(last_vertex(wall[-1]), e))
    edges.append(line_between(e, f))
    edges.append(line_between(f, a))
    return edges


def body_edges(wetted, outer):
    """The closed body meridian wire: two radial ends, the outer wall, and the wetted curve back."""
    edges = [line_between(first_vertex(wetted[0]), first_vertex(outer[0]))]
    edges.extend(outer)
    edges.append(line_between(last_vertex(outer[-1]), last_vertex(wetted[-1])))
    edges.extend(list(reversed(wetted)))
    return edges


def outer_edges(d, curves, wetted):
    """The outer wall as the TRUE normal offset of the wetted curve by t, with any loop cut out.

    Returns (edges, info, None) or (None, None, (rule, detail)).
    """
    t = d["t"]
    wet = cq.Compound.makeCompound(list(wetted)).wrapped
    pieces = [(Geom_OffsetCurve(c, -t, gp_Dir(0, 0, 1)), 0.0, 1.0) for c in curves]
    pieces.append((Geom_Line(gp_Pnt(d["L"], d["R_e"] + t, 0.0), gp_Dir(1, 0, 0)), 0.0, d["Lx"]))

    def inside(k, u):
        return point_distance(pieces[k][0].Value(u), wet) - t < -DELTA

    samples = []
    for k in range(len(pieces)):
        u0, u1 = pieces[k][1], pieces[k][2]
        for i in range(0 if k == 0 else 1, N_SAMPLE + 1):
            u = u0 + (u1 - u0) * i / N_SAMPLE
            samples.append((k, u, inside(k, u)))
    if samples[0][2] or samples[-1][2]:
        return (None, None, ("PRF-SELFX", "the normal offset of the wetted curve by t_wall "
                                          "self-intersects past an end of the wall"))
    runs = []
    i = 0
    while i < len(samples):
        if samples[i][2]:
            j = i
            while j + 1 < len(samples) and samples[j + 1][2]:
                j = j + 1
            runs.append((i, j))
            i = j + 1
        else:
            i = i + 1

    def bisect(k, u_kept, u_in):
        for _ in range(N_BISECT):
            m = 0.5 * (u_kept + u_in)
            if inside(k, m):
                u_in = m
            else:
                u_kept = m
        return u_kept

    cuts = []
    for i0, i1 in runs:
        ka, ua, _ = samples[i0 - 1]
        kr, ur, _ = samples[i0]
        cut_a = bisect(kr, ua if ka == kr else pieces[kr][1], ur)
        kr2, ur2, _ = samples[i1]
        kb, ub, _ = samples[i1 + 1]
        cut_b = bisect(kr2, ub if kb == kr2 else pieces[kr2][2], ur2)
        gap = pieces[kr][0].Value(cut_a).Distance(pieces[kr2][0].Value(cut_b))
        if gap > JOIN_TOL:
            return (None, None, ("PRF-SELFX", "the offset loop does not close at one "
                                              "self-intersection: gap %r m" % (gap,)))
        cuts.append((kr, cut_a, kr2, cut_b))

    def span(a, b):
        ka, sa = a
        kb, sb = b
        out = []
        for k in range(ka, kb + 1):
            s = sa if k == ka else pieces[k][1]
            e = sb if k == kb else pieces[k][2]
            if e - s > 1e-12:
                out.append((k, s, e))
        return out

    segs = []
    cur = (0, pieces[0][1])
    for kr, cut_a, kr2, cut_b in cuts:
        segs.extend(span(cur, (kr, cut_a)))
        cur = (kr2, cut_b)
    segs.extend(span(cur, (len(pieces) - 1, pieces[-1][2])))
    edges = []
    max_error = 0.0
    k0, s00 = segs[0][0], segs[0][1]
    p0 = pieces[k0][0].Value(s00)
    ends = [vertex(p0.X(), p0.Y())]            # one shared vertex per junction of the outer wall
    for k, s0, s1 in segs:
        curve = pieces[k][0]
        pb = curve.Value(s1)
        ends.append(vertex(pb.X(), pb.Y()))
        if k == len(pieces) - 1:
            edges.append(line_between(ends[-2], ends[-1]))
        else:
            ap = GeomConvert_ApproxCurve(Geom_TrimmedCurve(curve, s0, s1), APPROX_TOL, GeomAbs_C2, 200, 9)
            err = ap.MaxError() if ap.IsDone() and ap.HasResult() else -1.0
            if not ap.IsDone() or not ap.HasResult() or err > APPROX_MAX_ERR:
                raise RuntimeError("offset piece %d: B-spline refit failed (done %r, error %r m)"
                                   % (k, ap.IsDone(), err))
            max_error = max(max_error, err)
            bs = ap.Curve()
            edges.append(curve_between(bs, ends[-2], ends[-1], bs.FirstParameter(), bs.LastParameter()))
    fidelity = 0.0
    for edge in edges:
        for i in range(N_FIDELITY):
            q = edge.positionAt(i / (N_FIDELITY - 1))
            err = abs(point_distance(gp_Pnt(q.x, q.y, q.z), wet) - t)
            if err > fidelity:
                fidelity = err
    if fidelity > FIDELITY_TOL:
        raise RuntimeError("outer wall sits %r m off the true offset, limit %r" % (fidelity, FIDELITY_TOL))
    info = {"trimmed": bool(cuts), "cuts": len(cuts), "approx_error_m": max_error, "fidelity_m": fidelity}
    return (edges, info, None)


# ---------------------------------------------------------------- PRF
def point_distance(P, shape):
    d = BRepExtrema_DistShapeShape(BRepBuilderAPI_MakeVertex(P).Vertex(), shape)
    d.Perform()
    if not d.IsDone():
        raise RuntimeError("BRepExtrema_DistShapeShape not done")
    return d.Value()


def wetted_rules(d, curves):
    """PRF-RMIN, PRF-MONO and PRF-DERIV on the wetted law curves; None accepts."""
    rows = []
    for c in curves:
        row = []
        for i in range(N_MONO):
            u = i / (N_MONO - 1)
            P = gp_Pnt()
            V = gp_Vec()
            c.D1(u, P, V)
            row.append((P, V))
        rows.append(row)
    low = min(P.Y() for row in rows for P, V in row)
    if low <= 0:
        return ("PRF-RMIN", "the wetted radius reaches %r m <= 0" % (low,))
    if d["R_e"] >= d["R_i"]:
        return ("PRF-MONO", "R_e >= R_i: not a contraction")
    for row in rows:
        for P, V in row:
            if V.X() <= 0 or V.Y() > 1e-12:
                return ("PRF-MONO", "r is not non-increasing along +x")

    def slope(v):
        return v.Y() / v.X()

    a, b = slope(rows[0][0][1]), slope(rows[-1][-1][1])
    if abs(a) > DERIV_TOL or abs(b) > DERIV_TOL:
        return ("PRF-DERIV", "r'(0) = %r, r'(L) = %r, both must be 0" % (a, b))
    for i in range(len(rows) - 1):
        jump = slope(rows[i][-1][1]) - slope(rows[i + 1][0][1])
        if abs(jump) > DERIV_TOL:
            return ("PRF-DERIV", "r' jumps by %r at a junction" % (jump,))
    return None


def edge_list_rules(lists):
    """PRF-SELFX and PRF-FACE2D on the closed profile wires; (faces, None) or (None, (rule, detail))."""
    try:
        wires = [cq.Wire.assembleEdges(l) for l in lists]
    except Exception as e:
        return (None, ("PRF-FACE2D", "the edges do not form one wire: %s" % (e,)))
    plane = BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))).Face()
    for i, w in enumerate(wires):
        if ShapeAnalysis_Wire(w.wrapped, plane, 1e-7).CheckSelfIntersection():
            return (None, ("PRF-SELFX", "profile wire %d intersects itself" % (i,)))
    faces = []
    for i, w in enumerate(wires):
        f = make_face(w)
        if f is None:
            return (None, ("PRF-FACE2D", "profile wire %d does not bound a valid planar face" % (i,)))
        faces.append(f)
    return (faces, None)


def profile_stage(p, d, curves):
    """The 2-D stage: wetted rules, wetted and outer edges, both profile faces; (prof, None) or (None, ref)."""
    ref = wetted_rules(d, curves)
    if ref is not None:
        return (None, ref)
    wet = wetted_edges(d, curves)
    outer, info, ref = outer_edges(d, curves, wet)
    if ref is not None:
        return (None, ref)
    fluid = fluid_edges(d, wet[:-1])
    body = body_edges(wet, outer)
    faces, ref = edge_list_rules([fluid, body])
    if ref is not None:
        return (None, ref)
    prof = {"wetted": wet, "outer": outer, "fluid_edges": fluid, "body_edges": body,
            "fluid_face": faces[0], "wall_face": faces[1], "offset": info}
    return (prof, None)


# ---------------------------------------------------------------- faces
def make_face(w):
    """A planar face from a closed wire, only when BRepCheck passes and the area is positive."""
    try:
        f = cq.Face.makeFromWires(w)
    except Exception:
        return None
    if BRepCheck_Analyzer(f.wrapped).IsValid() and f.Area() > 0:
        return f
    return None


def classify_meridian(face, d, role="slip"):
    """Tag every edge of the fluid meridian face by the geometry of its midpoint."""
    tags = {"axis": [], "inlet": [], "outlet": [], UPSTREAM_TAG[role]: [], "wall_contraction": [],
            "wall_exit": []}
    for i, e in enumerate(face.Edges()):
        q = e.positionAt(0.5)
        if abs(q.y) <= CLASS_TOL:
            tags["axis"].append(i)
        elif abs(q.x - d["x_inlet"]) <= CLASS_TOL:
            tags["inlet"].append(i)
        elif abs(q.x - d["x_outlet"]) <= CLASS_TOL:
            tags["outlet"].append(i)
        elif d["x_inlet"] < q.x < 0:
            tags[UPSTREAM_TAG[role]].append(i)
        elif 0 < q.x < d["L"]:
            tags["wall_contraction"].append(i)
        elif d["L"] < q.x < d["x_outlet"]:
            tags["wall_exit"].append(i)
        else:
            raise RuntimeError("meridian edge %d at (%r, %r) matches no tag" % (i, q.x, q.y))
    return tags


def classify_wall(face, wetted, t):
    """Tag every edge of the body meridian face: wetted, outer (distance t), or one of the two ends."""
    wet = cq.Compound.makeCompound(list(wetted)).wrapped
    tags = {"wetted": [], "outer": [], "ends": []}
    for i, e in enumerate(face.Edges()):
        q = e.positionAt(0.5)
        g = point_distance(gp_Pnt(q.x, q.y, q.z), wet)
        if g <= FIDELITY_TOL:
            tags["wetted"].append(i)
        elif abs(g - t) <= FIDELITY_TOL:
            tags["outer"].append(i)
        else:
            tags["ends"].append(i)
    if len(tags["ends"]) != 2:
        raise RuntimeError("wall meridian has %d end edges, want exactly 2" % (len(tags["ends"]),))
    return tags


# ---------------------------------------------------------------- solids
def revolve(face):
    """One full revolution of a meridian face about the +x axis, as a single solid."""
    rv = BRepPrimAPI_MakeRevol(face.wrapped, gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0)), 2 * math.pi)
    rv.Build()
    sol = cq.Shape.cast(rv.Shape()).Solids()
    if len(sol) != 1:
        raise RuntimeError("revolve produced %d solids, want 1" % (len(sol),))
    return sol[0]


def stage_check(name, s):
    """BRepCheck and BRepAlgoAPI on one stage solid; a failure raises."""
    ok = BRepCheck_Analyzer(s.wrapped).IsValid() and BRepAlgoAPI_Check(s.wrapped).IsValid()
    if not ok:
        raise RuntimeError("stage check failed: %s is not a valid solid" % (name,))


def classify_fluid(solid, d, role="slip"):
    """Tag every face of the fluid solid by the geometry of its centre of mass."""
    tags = {"inlet": [], "outlet": [], UPSTREAM_TAG[role]: [], "wall_contraction": [], "wall_exit": []}
    for i, f in enumerate(solid.Faces()):
        c = f.Center()
        plane = f.geomType() == "PLANE"
        if plane and abs(c.x - d["x_inlet"]) <= CLASS_TOL:
            tags["inlet"].append(i)
        elif plane and abs(c.x - d["x_outlet"]) <= CLASS_TOL:
            tags["outlet"].append(i)
        elif not plane and d["x_inlet"] < c.x < 0:
            tags[UPSTREAM_TAG[role]].append(i)
        elif not plane and 0 < c.x < d["L"]:
            tags["wall_contraction"].append(i)
        elif not plane and d["L"] < c.x < d["x_outlet"]:
            tags["wall_exit"].append(i)
        else:
            raise RuntimeError("fluid face %d at x = %r (%s) matches no tag" % (i, c.x, f.geomType()))
    return tags


# ---------------------------------------------------------------- COMPOSE
def refused(params, ref):
    """The refusal record: exactly RESULT_KEYS, everything 3-D left empty."""
    return {"template_id": TEMPLATE_ID, "status": "refused", "rule": ref[0], "detail": ref[1],
            "params": params, "derived": {}, "planes": [], "files": {}, "face_tags": {},
            "meridian_edges": {}, "wall_edges": {}, "offset": {}, "checks": {}}


def build(params, out_dir):
    """The runner entry: refuse by rule id first, then the 2-D stage, then revolve, tag and export."""
    ref = domain_rules(params)
    if ref is not None:
        return refused(params, ref)
    p = resolved(params)
    d = derived(p)
    prof, ref = profile_stage(p, d, law_curves(p, d))
    if ref is not None:
        return refused(params, ref)
    fluid = revolve(prof["fluid_face"])
    stage_check("fluid", fluid)
    body = revolve(prof["wall_face"])
    stage_check("body", body)
    face_tags = classify_fluid(fluid, d, p["upstream_role"])
    meridian_edges = classify_meridian(prof["fluid_face"], d, p["upstream_role"])
    wall_edges = classify_wall(prof["wall_face"], prof["wetted"], d["t"])
    for name, shape in (("fluid.brep", fluid), ("body.brep", body),
                        ("meridian.brep", prof["fluid_face"]), ("wall_meridian.brep", prof["wall_face"])):
        if not shape.exportBrep(out_dir + "/" + name):
            raise RuntimeError("exportBrep returned False for %s" % (name,))
    return {"template_id": TEMPLATE_ID, "status": "ok", "rule": None, "detail": "",
            "params": p, "derived": d, "planes": planes(d),
            "files": {"fluid": "fluid.brep", "body": "body.brep", "meridian": "meridian.brep",
                      "wall_meridian": "wall_meridian.brep"},
            "face_tags": face_tags, "meridian_edges": meridian_edges, "wall_edges": wall_edges,
            "offset": prof["offset"],
            "checks": {"fluid_solids": 1, "fluid_valid": 1, "body_solids": 1, "body_valid": 1,
                       "fluid_volume_m3": fluid.Volume(), "body_volume_m3": body.Volume()}}


def declare(params, out_dir):
    """The runner entry template.json is generated from; both arguments are ignored."""
    return DECLARATION
