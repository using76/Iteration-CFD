#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""admission fixture bad_determ_outdir of CAD-23: the good pipe whose length depends on hash of the run directory; the first refusing rule is ADM-DETERM.

Built like the nozzle: meridian faces in the z = 0 plane, revolved 2 pi about the x axis with
BRepPrimAPI_MakeRevol. Sections in order: params, profile, faces, solids, COMPOSE. The runner
entries are build(params, out_dir), which returns one result record and writes four BREP files,
and declare(params, out_dir), which returns DECLARATION. Imports only math, cadquery and OCP, as
docs/16 §G admission requires; no main guard.
"""

import math

import cadquery as cq
from OCP.gp import gp_Pnt, gp_Dir, gp_Ax1
from OCP.BRepPrimAPI import BRepPrimAPI_MakeRevol
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepAlgoAPI import BRepAlgoAPI_Check

# ---------------------------------------------------------------- params
TEMPLATE_ID = "pipe_straight/1"
RESULT_KEYS = ("template_id", "status", "rule", "detail", "params", "derived", "planes", "files",
               "face_tags", "meridian_edges", "wall_edges", "offset", "checks")
CLASS_TOL = 1e-10       # m: geometric tag classification
PARAMS = [
    {"name": "D", "kind": "real", "unit": "m", "min": 0.01, "max": 0.2, "default_real": 0.05,
     "choices": [], "default_choice": None, "role": "intent", "only_when": None},
    {"name": "L_over_D", "kind": "real", "unit": "1", "min": 2.0, "max": 8.0, "default_real": 4.0,
     "choices": [], "default_choice": None, "role": "design", "only_when": None},
    {"name": "t_wall", "kind": "real", "unit": "m", "min": 0.001, "max": 0.01, "default_real": 0.003,
     "choices": [], "default_choice": None, "role": "design", "only_when": None},
]
PLANES = [{"name": "inlet", "description": "x = 0, the velocity inlet"},
          {"name": "mid", "description": "x = L / 2, the measurement station"},
          {"name": "outlet", "description": "x = L, the pressure outlet"}]
TAGS = [{"name": "inlet", "kind": "face", "description": "inlet disc at x = 0"},
        {"name": "outlet", "kind": "face", "description": "outlet disc at x = L"},
        {"name": "wall", "kind": "face", "description": "the wetted cylinder r = R"},
        {"name": "wetted", "kind": "edge", "description": "meridian wetted line r = R"},
        {"name": "outer", "kind": "edge", "description": "meridian outer line r = R + t"}]
CATALOGUE = [
    {"quantity": "pipe_diameter", "primitive": "diameter_at_plane", "where": ["mid"],
     "kind": "geometric", "method": "geometry", "unit": "m", "u_kind": "abs", "u_meas": 1e-9},
    {"quantity": "total_length", "primitive": "extent_along_axis", "where": ["body"],
     "kind": "geometric", "method": "geometry", "unit": "m", "u_kind": "abs", "u_meas": 1e-9},
    {"quantity": "pipe_length", "primitive": "plane_distance", "where": ["inlet", "outlet"],
     "kind": "geometric", "method": "geometry", "unit": "m", "u_kind": "abs", "u_meas": 1e-9},
    {"quantity": "min_wall_normal", "primitive": "meridian_min_wall", "where": ["wetted", "outer"],
     "kind": "geometric", "method": "geometry", "unit": "m", "u_kind": "abs", "u_meas": 1e-8},
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
]
DRIVERS = {"pipe_diameter": ["D"], "total_length": ["D", "L_over_D"], "pipe_length": ["D", "L_over_D"],
           "min_wall_normal": ["t_wall"], "n_solids": [], "valid": [], "watertight": [], "axis": [],
           "units": []}
PROFILE_RULES = ["PRF-BOX"]
STANDARDS = []
DECLARATION = {"schema": "cad-template/1", "template_id": TEMPLATE_ID,
               "title": "a straight circular pipe on +x", "axis": "+x", "units": "m",
               "params": PARAMS, "planes": PLANES, "tags": TAGS, "catalogue": CATALOGUE,
               "profile_rules": PROFILE_RULES, "standards": STANDARDS}


def domain_rules(p):
    """PRF-BOX on a missing or unknown name, a non-finite or non-number real, or a value outside
    its box; None accepts."""
    if not isinstance(p, dict):
        return ("PRF-BOX", "parameters missing: [%s]; unknown: []"
                % ", ".join(row["name"] for row in PARAMS))
    names = [row["name"] for row in PARAMS]
    missing = [n for n in names if n not in p]
    unknown = sorted(k for k in p if k not in set(names))
    if missing or unknown:
        return ("PRF-BOX", "parameters missing: [%s]; unknown: [%s]"
                % (", ".join(missing), ", ".join(unknown)))
    for row in PARAMS:
        v = p[row["name"]]
        if not isinstance(v, bool) and isinstance(v, (int, float)) and not math.isfinite(v):
            return ("PRF-BOX", "%s = %r is not a finite number" % (row["name"], v))
        if (isinstance(v, bool) or not isinstance(v, (int, float))
                or (row["min"] is not None and v < row["min"])
                or (row["max"] is not None and v > row["max"])):
            return ("PRF-BOX", "%s = %r is outside [%r, %r]" % (row["name"], v, row["min"], row["max"]))
    return None


def resolved(p):
    """A params dict with every real a float."""
    q = dict(p)
    for row in PARAMS:
        if row["kind"] == "real":
            q[row["name"]] = float(p[row["name"]])
    return q


def derived(p):
    """R, D_e, L, t and the two axis ends, from resolved params."""
    return {"R": p["D"] / 2.0, "D_e": p["D"], "L": p["L_over_D"] * p["D"], "t": p["t_wall"],
            "x_inlet": 0.0, "x_outlet": p["L_over_D"] * p["D"]}


def planes(d):
    return [{"name": "inlet", "x": d["x_inlet"]}, {"name": "mid", "x": d["L"] / 2.0},
            {"name": "outlet", "x": d["x_outlet"]}]


# ---------------------------------------------------------------- profile
def rectangle(x0, r0, x1, r1):
    """A meridian face in the z = 0 plane: the rectangle [x0, x1] x [r0, r1] as one wire."""
    a = cq.Vector(x0, r0, 0.0)
    b = cq.Vector(x1, r0, 0.0)
    c = cq.Vector(x1, r1, 0.0)
    e = cq.Vector(x0, r1, 0.0)
    edges = [cq.Edge.makeLine(a, b), cq.Edge.makeLine(b, c), cq.Edge.makeLine(c, e),
             cq.Edge.makeLine(e, a)]
    return cq.Face.makeFromWires(cq.Wire.assembleEdges(edges))


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


# ---------------------------------------------------------------- faces
def classify_fluid(solid, d):
    """Tag every face of the fluid solid by the geometry of its centre of mass."""
    tags = {"inlet": [], "outlet": [], "wall": []}
    for i, f in enumerate(solid.Faces()):
        c = f.Center()
        plane = f.geomType() == "PLANE"
        if plane and abs(c.x - d["x_inlet"]) <= CLASS_TOL:
            tags["inlet"].append(i)
        elif plane and abs(c.x - d["x_outlet"]) <= CLASS_TOL:
            tags["outlet"].append(i)
        elif not plane and d["x_inlet"] < c.x < d["x_outlet"]:
            tags["wall"].append(i)
        else:
            raise RuntimeError("fluid face %d at x = %r (%s) matches no tag" % (i, c.x, f.geomType()))
    return tags


def classify_meridian(face, d):
    """Tag every edge of the fluid meridian face by the geometry of its midpoint."""
    tags = {"axis": [], "inlet": [], "outlet": [], "wall": []}
    for i, e in enumerate(face.Edges()):
        q = e.positionAt(0.5)
        if abs(q.y) <= CLASS_TOL:
            tags["axis"].append(i)
        elif abs(q.x - d["x_inlet"]) <= CLASS_TOL:
            tags["inlet"].append(i)
        elif abs(q.x - d["x_outlet"]) <= CLASS_TOL:
            tags["outlet"].append(i)
        elif d["x_inlet"] < q.x < d["x_outlet"]:
            tags["wall"].append(i)
        else:
            raise RuntimeError("meridian edge %d at (%r, %r) matches no tag" % (i, q.x, q.y))
    return tags


def classify_wall(face, d):
    """Tag every edge of the body meridian face: wetted, outer, or one of the two ends."""
    tags = {"wetted": [], "outer": [], "ends": []}
    for i, e in enumerate(face.Edges()):
        q = e.positionAt(0.5)
        if abs(q.y - d["R"]) <= CLASS_TOL:
            tags["wetted"].append(i)
        elif abs(q.y - (d["R"] + d["t"])) <= CLASS_TOL:
            tags["outer"].append(i)
        else:
            tags["ends"].append(i)
    if len(tags["ends"]) != 2:
        raise RuntimeError("wall meridian has %d end edges, want exactly 2" % (len(tags["ends"]),))
    return tags


# ---------------------------------------------------------------- COMPOSE
def refused(params, ref):
    """The refusal record: exactly RESULT_KEYS, everything 3-D left empty."""
    return {"template_id": TEMPLATE_ID, "status": "refused", "rule": ref[0], "detail": ref[1],
            "params": params, "derived": {}, "planes": [], "files": {}, "face_tags": {},
            "meridian_edges": {}, "wall_edges": {}, "offset": {}, "checks": {}}


def build(params, out_dir):
    """The runner entry: refuse by rule id first, then revolve, tag and export."""
    ref = domain_rules(params)
    if ref is not None:
        return refused(params, ref)
    p = resolved(params)
    d = derived(p)
    d["L"] = d["L"] * (1.0 + (hash(out_dir) & 0xFFFFFF) * 1e-13)
    d["x_outlet"] = d["L"]
    fluid_face = rectangle(0.0, 0.0, d["L"], d["R"])
    wall_face = rectangle(0.0, d["R"], d["L"], d["R"] + d["t"])
    fluid = revolve(fluid_face)
    stage_check("fluid", fluid)
    body = revolve(wall_face)
    stage_check("body", body)
    face_tags = classify_fluid(fluid, d)
    meridian_edges = classify_meridian(fluid_face, d)
    wall_edges = classify_wall(wall_face, d)
    for name, shape in (("fluid.brep", fluid), ("body.brep", body),
                        ("meridian.brep", fluid_face), ("wall_meridian.brep", wall_face)):
        if not shape.exportBrep(out_dir + "/" + name):
            raise RuntimeError("exportBrep returned False for %s" % (name,))
    return {"template_id": TEMPLATE_ID, "status": "ok", "rule": None, "detail": "",
            "params": p, "derived": d, "planes": planes(d),
            "files": {"fluid": "fluid.brep", "body": "body.brep", "meridian": "meridian.brep",
                      "wall_meridian": "wall_meridian.brep"},
            "face_tags": face_tags, "meridian_edges": meridian_edges, "wall_edges": wall_edges,
            "offset": {},
            "checks": {"fluid_solids": 1, "fluid_valid": 1, "body_solids": 1, "body_valid": 1,
                       "fluid_volume_m3": fluid.Volume(), "body_volume_m3": body.Volume()}}


def declare(params, out_dir):
    """The runner entry template.json is generated from; both arguments are ignored."""
    return DECLARATION
