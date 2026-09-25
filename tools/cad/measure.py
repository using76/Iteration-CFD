#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""measure.py - the typed measurement primitives of the CAD loop (docs/16 §E.2, gate GC-1 of §H.3): each returns one cad-measure/1 record, and wall thickness is measured on the 2-D meridian, never between 3-D surfaces.

Usage:
  python measure.py --selftest
"""

import math
import os
import re
import sys

import numpy as np
import cadquery as cq
from scipy.optimize import minimize, minimize_scalar

from OCP.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Circle
from OCP.BRepAlgoAPI import BRepAlgoAPI_Section, BRepAlgoAPI_Check
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCP.gp import gp_Pln, gp_Pnt, gp_Dir, gp_Vec, gp_Ax1
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepLProp import BRepLProp_CLProps
from OCP.BRepPrimAPI import BRepPrimAPI_MakeRevol   # selftest only

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import schema

SCHEMA_KIND = "cad-measure/1"
AXIS_TOL = 1e-12          # m: a circle centre / meridian edge may leave the axis / plane z = 0 by this much
XCHECK_TOL = 1e-6         # m: BRepExtrema vs dense sampling on the meridian (docs/16 E.2)
N_DENSE = 2001            # points per edge for the dense cross-check
N_SCAN = 2001             # parameter samples per edge before the bounded refinement (slope, curvature)
U_MEAS = {                # (kind, value): "abs" in the record's unit, "rel" times |value|; docs/16 E.2
    "cylinder_radius": ("rel", 1e-12), "cone_semi_angle": ("abs", 1e-9), "volume": ("rel", 1e-8),
    "diameter_at_plane": ("abs", 1e-9), "area_ratio": ("rel", 1e-9), "extent_along_axis": ("abs", 1e-9),
    "plane_distance": ("abs", 1e-9), "meridian_min_wall": ("abs", 1e-8), "slope_max": ("rel", 1e-6),
    "curvature_radius_min": ("rel", 1e-6), "n_solids": ("abs", 0.0), "valid": ("abs", 0.0),
}
REFUSED = {"wall_distance_3d": "MEAS-3D-WALL"}


def record(primitive, value, unit, u_meas, method, feature=None, where=(), status="ok", reason_id=None, detail=""):
    """One cad-measure/1 dict, keys in the schema's required order."""
    return {"schema": SCHEMA_KIND, "primitive": primitive, "feature": feature, "where": list(where),
            "value": value, "unit": unit, "u_meas": u_meas, "method": method, "status": status,
            "reason_id": reason_id, "detail": detail}

_NAME_RE = re.compile("^[A-Za-z][A-Za-z0-9_]{0,63}$")


def _ok(primitive, value, unit, method, feature=None, where=(), detail=""):
    value = float(value)
    if not math.isfinite(value):
        return _error(primitive, unit, method, "MEAS-NONFINITE", "value %r is not finite" % (value,), feature, where)
    kind, num = U_MEAS[primitive]
    u_meas = num if kind == "abs" else num * abs(value)
    return record(primitive, value, unit, u_meas, method, feature=feature, where=where, detail=detail)


def _refused(primitive, unit, method, reason_id, detail, feature=None, where=()):
    return record(primitive, None, unit, None, method, feature=feature, where=where,
                  status="refused", reason_id=reason_id, detail=detail)


def _error(primitive, unit, method, reason_id, detail, feature=None, where=()):
    return record(primitive, None, unit, None, method, feature=feature, where=where,
                  status="error", reason_id=reason_id, detail=detail)


def _plane_ok(p):
    """True iff p is a template plane {name, x}: the name pattern and a finite non-bool x."""
    if not isinstance(p, dict):
        return False
    name, x = p.get("name"), p.get("x")
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        return False
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        return False
    return True


def _meridian_ok(edge):
    """True iff the edge is a cq.Edge lying in the meridian plane z = 0 at 11 sampled parameters."""
    if not isinstance(edge, cq.Edge):
        return False
    c = BRepAdaptor_Curve(edge.wrapped)
    for t in np.linspace(c.FirstParameter(), c.LastParameter(), 11):
        if abs(c.Value(float(t)).Z()) > AXIS_TOL:
            return False
    return True


def cylinder_radius(face, feature=None):
    method = "BRepAdaptor_Surface cylinder radius"
    try:
        a = BRepAdaptor_Surface(face.wrapped)
        if a.GetType() != GeomAbs_Cylinder:
            return _refused("cylinder_radius", "m", method, "MEAS-NOTCYL",
                            "the face is not a cylindrical surface", feature)
        return _ok("cylinder_radius", a.Cylinder().Radius(), "m", method, feature=feature)
    except Exception as e:
        return _error("cylinder_radius", "m", method, "MEAS-ERROR",
                      ("%s: %s" % (type(e).__name__, e))[:300], feature)


def cone_semi_angle(face, feature=None):
    method = "BRepAdaptor_Surface cone semi-angle"
    try:
        a = BRepAdaptor_Surface(face.wrapped)
        if a.GetType() != GeomAbs_Cone:
            return _refused("cone_semi_angle", "rad", method, "MEAS-NOTCONE",
                            "the face is not a conical surface", feature)
        return _ok("cone_semi_angle", abs(a.Cone().SemiAngle()), "rad", method, feature=feature)
    except Exception as e:
        return _error("cone_semi_angle", "rad", method, "MEAS-ERROR",
                      ("%s: %s" % (type(e).__name__, e))[:300], feature)


def volume(shape, feature=None):
    method = "BRepGProp volume"
    try:
        if len(shape.Solids()) == 0:
            return _refused("volume", "m3", method, "MEAS-NOSOLID",
                            "the shape holds no solid", feature)
        p = GProp_GProps()
        BRepGProp.VolumeProperties_s(shape.wrapped, p)
        return _ok("volume", p.Mass(), "m3", method, feature=feature)
    except Exception as e:
        return _error("volume", "m3", method, "MEAS-ERROR",
                      ("%s: %s" % (type(e).__name__, e))[:300], feature)


def diameter_at_plane(shape, plane, feature=None):
    method = "BRepAlgoAPI_Section circle radius at a named plane"
    try:
        if not _plane_ok(plane):
            return _refused("diameter_at_plane", "m", method, "MEAS-BADPLANE",
                            "the plane is not a {name, x} dict with a finite non-bool x", feature)
        x = float(plane["x"])
        pl = BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(x, 0, 0), gp_Dir(1, 0, 0))).Face()
        sec = BRepAlgoAPI_Section(shape.wrapped, pl)
        sec.Build()
        edges = cq.Shape.cast(sec.Shape()).Edges()
        if not edges:
            return _refused("diameter_at_plane", "m", method, "MEAS-EMPTY",
                            "the section at plane %s has no edge" % (plane["name"],), feature, (plane["name"],))
        not_circle = [e for e in edges if BRepAdaptor_Curve(e.wrapped).GetType() != GeomAbs_Circle]
        if not_circle:
            return _refused("diameter_at_plane", "m", method, "MEAS-NOTCIRCLE",
                            "%d of %d section edges are not circles" % (len(not_circle), len(edges)),
                            feature, (plane["name"],))
        radii = []
        for e in edges:
            circ = BRepAdaptor_Curve(e.wrapped).Circle()
            loc, ax = circ.Location(), circ.Axis().Direction()
            if (abs(loc.Y()) > AXIS_TOL or abs(loc.Z()) > AXIS_TOL or abs(loc.X() - x) > AXIS_TOL
                    or abs(ax.X()) < 1 - AXIS_TOL):
                return _refused("diameter_at_plane", "m", method, "MEAS-OFFAXIS",
                                "a section circle at plane %s is off the axis or not normal to +x"
                                % (plane["name"],), feature, (plane["name"],))
            radii.append(circ.Radius())
        return _ok("diameter_at_plane", 2 * max(radii), "m", method, feature=feature, where=(plane["name"],))
    except Exception as e:
        return _error("diameter_at_plane", "m", method, "MEAS-ERROR",
                      ("%s: %s" % (type(e).__name__, e))[:300], feature,
                      (plane["name"],) if _plane_ok(plane) else ())


def area_ratio(shape, plane_a, plane_b, feature=None):
    method = "(diameter_at_plane a / diameter_at_plane b) squared"
    try:
        for p in (plane_a, plane_b):
            if not _plane_ok(p):
                return _refused("area_ratio", "1", method, "MEAS-BADPLANE",
                                "the plane is not a {name, x} dict with a finite non-bool x", feature)
        where = (plane_a["name"], plane_b["name"])
        d = {}
        for p in (plane_a, plane_b):
            rec = diameter_at_plane(shape, p, feature)
            if rec["status"] != "ok":       # propagate status and rule id as an area_ratio record
                return record("area_ratio", None, "1", None, method, feature=feature, where=where,
                              status=rec["status"], reason_id=rec["reason_id"],
                              detail="plane %s: %s" % (p["name"], rec["detail"]))
            d[p["name"]] = rec["value"]
        return _ok("area_ratio", (d[plane_a["name"]] / d[plane_b["name"]]) ** 2, "1", method,
                   feature=feature, where=where)
    except Exception as e:
        return _error("area_ratio", "1", method, "MEAS-ERROR",
                      ("%s: %s" % (type(e).__name__, e))[:300], feature)


def extent_along_axis(shape, feature=None):
    """Span along +x from BRepExtrema distances to two far planes (an optimal Bnd_Box is 2e-7 m off on a torus)."""
    method = "BRepExtrema distance to two far planes normal to +x"
    try:
        b = Bnd_Box()
        BRepBndLib.AddOptimal_s(shape.wrapped, b, False, False)
        x0, y0, z0, x1, y1, z1 = b.Get()
        m = 10.0 * max(x1 - x0, y1 - y0, z1 - z0) + 1.0

        def gap(xp):
            pl = BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(xp, 0, 0), gp_Dir(1, 0, 0)), -1e3, 1e3, -1e3, 1e3).Face()
            d = BRepExtrema_DistShapeShape(shape.wrapped, pl)
            d.Perform()
            if not d.IsDone():
                raise RuntimeError("BRepExtrema not done")
            return d.Value()

        lo = (x0 - m) + gap(x0 - m)
        hi = (x1 + m) - gap(x1 + m)
        return _ok("extent_along_axis", hi - lo, "m", method, feature=feature)
    except Exception as e:
        return _error("extent_along_axis", "m", method, "MEAS-ERROR",
                      ("%s: %s" % (type(e).__name__, e))[:300], feature)


def plane_distance(plane_a, plane_b, feature=None):
    method = "difference of named plane x"
    try:
        for p in (plane_a, plane_b):
            if not _plane_ok(p):
                return _refused("plane_distance", "m", method, "MEAS-BADPLANE",
                                "the plane is not a {name, x} dict with a finite non-bool x", feature)
        return _ok("plane_distance", abs(float(plane_b["x"]) - float(plane_a["x"])), "m", method,
                   feature=feature, where=(plane_a["name"], plane_b["name"]))
    except Exception as e:
        return _error("plane_distance", "m", method, "MEAS-ERROR",
                      ("%s: %s" % (type(e).__name__, e))[:300], feature)


def _edge_points(edge, n):
    c = BRepAdaptor_Curve(edge.wrapped)
    ts = np.linspace(c.FirstParameter(), c.LastParameter(), n)
    return np.array([[c.Value(t).X(), c.Value(t).Y(), c.Value(t).Z()] for t in ts])


def _points_to_polyline(P, Q):
    A, B = Q[:-1], Q[1:]
    AB = B - A
    L2 = np.maximum(np.einsum("ij,ij->i", AB, AB), 1e-300)
    best = np.inf
    for i in range(0, len(P), 256):
        p = P[i:i + 256][:, None, :]
        t = np.clip(np.einsum("pij,ij->pi", p - A[None], AB) / L2[None], 0.0, 1.0)
        q = A[None] + t[..., None] * AB[None]
        best = min(best, float(np.sqrt(((p - q) ** 2).sum(-1)).min()))
    return best


def _dense_min_distance(wetted, outer, n):
    pw = [_edge_points(e, n) for e in wetted]
    po = [_edge_points(e, n) for e in outer]
    m = np.inf
    for a in pw:
        for b in po:
            m = min(m, _points_to_polyline(a, b), _points_to_polyline(b, a))
    return m


def meridian_min_wall(wetted, outer, feature=None, n_dense=N_DENSE):
    """Wall thickness on the 2-D meridian: BRepExtrema between the tagged edges, dense polyline cross-check."""
    method = "BRepExtrema_DistShapeShape on 2-D meridian edges, dense polyline cross-check"
    try:
        if not wetted or not outer:
            return _refused("meridian_min_wall", "m", method, "MEAS-EMPTY",
                            "the wetted or the outer edge list is empty", feature, ("wetted", "outer"))
        for item in list(wetted) + list(outer):
            if isinstance(item, cq.Shape) and not isinstance(item, cq.Edge):
                return _refused("meridian_min_wall", "m", method, "MEAS-3D-WALL",
                                "wall thickness on a solid of revolution is measured on the 2-D meridian edges "
                                "tagged wetted and outer; a 3-D surface or solid was passed",
                                feature, ("wetted", "outer"))
        for item in list(wetted) + list(outer):
            if not _meridian_ok(item):
                return _refused("meridian_min_wall", "m", method, "MEAS-3D-WALL",
                                "an edge leaves the meridian plane z = 0", feature, ("wetted", "outer"))
        d = BRepExtrema_DistShapeShape(cq.Compound.makeCompound(list(wetted)).wrapped,
                                       cq.Compound.makeCompound(list(outer)).wrapped)
        d.Perform()
        if not d.IsDone():
            return _error("meridian_min_wall", "m", method, "MEAS-ERROR",
                          "BRepExtrema_DistShapeShape not done", feature, ("wetted", "outer"))
        brep = d.Value()
        dense = _dense_min_distance(wetted, outer, n_dense)
        if abs(brep - dense) > XCHECK_TOL:
            return _refused("meridian_min_wall", "m", method, "MEAS-XCHECK",
                            "BRepExtrema %.12g m vs dense %.12g m differ by more than 1e-6 m" % (brep, dense),
                            feature, ("wetted", "outer"))
        return _ok("meridian_min_wall", brep, "m", method, feature=feature, where=("wetted", "outer"),
                   detail="dense %.12g m" % (dense,))
    except Exception as e:
        return _error("meridian_min_wall", "m", method, "MEAS-ERROR",
                      ("%s: %s" % (type(e).__name__, e))[:300], feature, ("wetted", "outer"))


def wall_distance_3d(*args, **kwargs):
    return _refused("wall_distance_3d", "m", "refused: 3-D surface-to-surface wall distance", "MEAS-3D-WALL",
                    "the 3-D free-form wall distance is refused: wall thickness is the normal wall on the "
                    "2-D meridian (meridian_min_wall, docs/16 E.2)")


def _scan_max(f, first, last):
    us = np.linspace(first, last, N_SCAN)
    vals = [f(u) for u in us]
    k = int(np.argmax(vals))
    lo, hi = us[max(k - 1, 0)], us[min(k + 1, N_SCAN - 1)]
    best = vals[k]
    if hi > lo:
        r = minimize_scalar(lambda u: -f(u), bounds=(lo, hi), method="bounded",
                            options={"xatol": 1e-12 * (last - first)})
        best = max(best, -float(r.fun))
    return best


def slope_max(edges, feature=None):
    """Max tangent angle to +x over meridian edges; the cad-measure/1 unit enum has rad and no deg."""
    method = "max tangent angle to +x on meridian edges, scanned and refined"
    try:
        if not edges:
            return _refused("slope_max", "rad", method, "MEAS-EMPTY",
                            "no edge was passed", feature)
        for e in edges:
            if not _meridian_ok(e):
                return _refused("slope_max", "rad", method, "MEAS-NOTMERIDIAN",
                                "an edge leaves the meridian plane z = 0", feature)
        best = -math.inf
        for e in edges:
            c = BRepAdaptor_Curve(e.wrapped)

            def angle(u, c=c):
                P = gp_Pnt()
                V = gp_Vec()
                c.D1(float(u), P, V)
                return math.atan2(abs(V.Y()), abs(V.X()))

            best = max(best, _scan_max(angle, c.FirstParameter(), c.LastParameter()))
        return _ok("slope_max", best, "rad", method, feature=feature)
    except Exception as e:
        return _error("slope_max", "rad", method, "MEAS-ERROR",
                      ("%s: %s" % (type(e).__name__, e))[:300], feature)


def curvature_radius_min(edges, feature=None):
    method = "1 / max curvature (BRepLProp_CLProps) on meridian edges, scanned and refined"
    try:
        if not edges:
            return _refused("curvature_radius_min", "m", method, "MEAS-EMPTY",
                            "no edge was passed", feature)
        for e in edges:
            if not _meridian_ok(e):
                return _refused("curvature_radius_min", "m", method, "MEAS-NOTMERIDIAN",
                                "an edge leaves the meridian plane z = 0", feature)
        kmax = -math.inf
        for e in edges:
            c = BRepAdaptor_Curve(e.wrapped)

            def curvature(u, c=c):
                return BRepLProp_CLProps(c, float(u), 2, 1e-12).Curvature()

            kmax = max(kmax, _scan_max(curvature, c.FirstParameter(), c.LastParameter()))
        if kmax <= 1e-12:
            return _refused("curvature_radius_min", "m", method, "MEAS-NOCURV",
                            "max curvature %.12g: a straight generator has no finite radius" % (kmax,), feature)
        return _ok("curvature_radius_min", 1.0 / kmax, "m", method, feature=feature)
    except Exception as e:
        return _error("curvature_radius_min", "m", method, "MEAS-ERROR",
                      ("%s: %s" % (type(e).__name__, e))[:300], feature)


def n_solids(shape, feature=None):
    method = "count of solids"
    try:
        return _ok("n_solids", len(shape.Solids()), "1", method, feature=feature)
    except Exception as e:
        return _error("n_solids", "1", method, "MEAS-ERROR",
                      ("%s: %s" % (type(e).__name__, e))[:300], feature)


def valid(shape, feature=None):
    method = "BRepCheck_Analyzer and BRepAlgoAPI_Check"
    try:
        a = BRepCheck_Analyzer(shape.wrapped).IsValid()
        b = BRepAlgoAPI_Check(shape.wrapped).IsValid()
        return _ok("valid", 1 if (a and b) else 0, "1", method, feature=feature,
                   detail="BRepCheck=%s BOPCheck=%s" % (a, b))
    except Exception as e:
        return _error("valid", "1", method, "MEAS-ERROR",
                      ("%s: %s" % (type(e).__name__, e))[:300], feature)


PRIMITIVES = {"cylinder_radius": cylinder_radius, "cone_semi_angle": cone_semi_angle, "volume": volume,
              "diameter_at_plane": diameter_at_plane, "area_ratio": area_ratio,
              "extent_along_axis": extent_along_axis, "plane_distance": plane_distance,
              "meridian_min_wall": meridian_min_wall, "slope_max": slope_max,
              "curvature_radius_min": curvature_radius_min, "n_solids": n_solids, "valid": valid}

assert set(PRIMITIVES) == set(U_MEAS), "the primitive registry and U_MEAS must name the same set"


def run(name, *args, **kwargs):
    """Measure by primitive name; a refused method answers by its rule id, an unknown name by MEAS-UNKNOWN."""
    if name in PRIMITIVES:
        return PRIMITIVES[name](*args, **kwargs)
    if name in REFUSED:
        return wall_distance_3d(*args, **kwargs)
    return _refused("unknown", "1", "no such primitive", "MEAS-UNKNOWN", "no primitive named %r" % (name,))


# ---------------------------------------------------------------- selftest fixtures (docs/16 H.1)

V = cq.Vector
_FX_RI, _FX_RE = 0.030, 0.010                                # poly5 law, D_i 0.06, D_e 0.02


def _fx_r(x, L):
    xi = x / L
    return _FX_RI - (_FX_RI - _FX_RE) * (10 * xi ** 3 - 15 * xi ** 4 + 6 * xi ** 5)


def _fx_dr(x, L):
    xi = x / L
    return -(_FX_RI - _FX_RE) * (30 * xi ** 2 - 60 * xi ** 3 + 30 * xi ** 4) / L


def _fx_ddr(x, L):
    xi = x / L
    return -(_FX_RI - _FX_RE) * (60 * xi - 180 * xi ** 2 + 120 * xi ** 3) / L ** 2


def _fx_revolve_polygon(pts):
    return cq.Workplane("XY").polyline(pts).close().revolve(360, (0, 0, 0), (1, 0, 0)).val()


def _fx_poly5_spline(L, offset=None, w=0.003, N=200):
    pts = []
    for k in range(N + 1):
        x = L * k / N
        r = _fx_r(x, L)
        if offset is None:
            pts.append(V(x, r, 0))
        elif offset == "radial":
            pts.append(V(x, r + w, 0))
        else:
            d = _fx_dr(x, L)
            s = math.sqrt(1.0 + d * d)
            pts.append(V(x - w * d / s, r + w / s, 0))
    return cq.Edge.makeSpline(pts, tangents=[V(1, 0, 0), V(1, 0, 0)])


def _fx_poly5_fluid(L_over_Di=1.0):
    L, Lu, Lx = L_over_Di * 0.06, 0.03, 0.01
    edges = [cq.Edge.makeLine(V(-Lu, 0, 0), V(-Lu, _FX_RI, 0)),
             cq.Edge.makeLine(V(-Lu, _FX_RI, 0), V(0, _FX_RI, 0)),
             _fx_poly5_spline(L),
             cq.Edge.makeLine(V(L, _FX_RE, 0), V(L + Lx, _FX_RE, 0)),
             cq.Edge.makeLine(V(L + Lx, _FX_RE, 0), V(L + Lx, 0, 0)),
             cq.Edge.makeLine(V(L + Lx, 0, 0), V(-Lu, 0, 0))]
    solid = cq.Solid.revolve(cq.Face.makeFromWires(cq.Wire.assembleEdges(edges)), 360, V(0, 0, 0), V(1, 0, 0))
    planes = {"contraction_start": {"name": "contraction_start", "x": 0.0},
              "exit_plane": {"name": "exit_plane", "x": L}}
    return solid, planes


def _fx_wall(offset, L=0.03, w=0.003, Lx=0.01):
    """poly5 L/D_i 0.5 wall: wetted spline+exit line, outer the same offset radially or along the normal."""
    wet = [_fx_poly5_spline(L), cq.Edge.makeLine(V(L, _FX_RE, 0), V(L + Lx, _FX_RE, 0))]
    if offset == "radial":
        out = [_fx_poly5_spline(L, "radial", w), cq.Edge.makeLine(V(L, _FX_RE + w, 0), V(L + Lx, _FX_RE + w, 0))]
    else:
        sL = math.sqrt(1.0 + _fx_dr(L, L) ** 2)
        out = [_fx_poly5_spline(L, "normal", w),
               cq.Edge.makeLine(V(L - w * _fx_dr(L, L) / sL, _FX_RE + w / sL, 0), V(L + Lx, _FX_RE + w, 0))]
    return wet, out


def _fx_wall_truth(L, w):
    """Continuous truth of the radial offset's minimum wall, from the analytic law only."""
    g = np.linspace(0, L, 801)
    XO, XI = np.meshgrid(g, g, indexing="ij")
    D = np.hypot(XO - XI, _fx_r(XO, L) + w - _fx_r(XI, L))
    i, j = np.unravel_index(D.argmin(), D.shape)
    f = lambda v: math.hypot(v[0] - v[1], _fx_r(v[0], L) + w - _fx_r(v[1], L))
    res = minimize(f, [g[i], g[j]], method="Nelder-Mead",
                   options={"xatol": 1e-15, "fatol": 1e-18, "maxiter": 20000})
    return float(res.fun)


def _fx_cone_band(R1=0.025, R2=0.0125, L=0.040, t=0.003):
    alpha = math.atan((R1 - R2) / L)
    n0, n1 = math.sin(alpha), math.cos(alpha)
    wet = [cq.Edge.makeLine(V(0, R1, 0), V(L, R2, 0))]
    out_n = [cq.Edge.makeLine(V(t * n0, R1 + t * n1, 0), V(L + t * n0, R2 + t * n1, 0))]
    out_r = [cq.Edge.makeLine(V(0, R1 + t, 0), V(L, R2 + t, 0))]
    return wet, out_n, out_r, alpha, t


def _fx_poly5_kappa_truth(L):
    xs = np.linspace(0, L, 20001)

    def kappa(x):
        d, dd = _fx_dr(x, L), _fx_ddr(x, L)
        return abs(dd) / (1.0 + d * d) ** 1.5

    vals = [kappa(x) for x in xs]
    k = int(np.argmax(vals))
    lo, hi = xs[max(k - 1, 0)], xs[min(k + 1, 20000)]
    best = vals[k]
    if hi > lo:
        r = minimize_scalar(lambda x: -kappa(x), bounds=(lo, hi), method="bounded")
        best = max(best, -float(r.fun))
    return 1.0 / best


def _fx_torus():
    return cq.Solid.makeTorus(0.05, 0.01, pnt=V(0, 0, 0), dir=V(0, 0, 1))


def _fx_bowtie():
    return cq.Face.makeFromWires(cq.Wire.makePolygon(
        [V(0, 0.01, 0), V(0.04, 0.03, 0), V(0.04, 0.01, 0), V(0, 0.03, 0)], close=True))


def _fx_arc():
    return cq.Edge.makeCircle(0.05, V(0, 0, 0), V(0, 0, 1), 10, 80)


def selftest():
    """GC-1: every primitive against an analytic answer; 21 [ok] lines, then SELFTEST PASS."""
    seen = []

    def keep(rec):
        seen.append(rec)
        return rec

    cylinder = _fx_revolve_polygon([(0, 0), (0, 0.03), (0.1, 0.03), (0.1, 0)])
    cyl_face = [f for f in cylinder.Faces() if f.geomType() == "CYLINDER"][0]
    frustum = _fx_revolve_polygon([(0, 0), (0, 0.03), (0.1, 0.01), (0.1, 0)])
    cone_face = [f for f in frustum.Faces() if f.geomType() == "CONE"][0]
    fluid, planes = _fx_poly5_fluid()
    cs, ep = planes["contraction_start"], planes["exit_plane"]

    # (M1)
    rec = keep(cylinder_radius(cyl_face))
    err = abs(rec["value"] - 0.03) / 0.03
    assert err <= 1e-12, "M1 cylinder radius rel %g > 1e-12" % (err,)
    print("[ok] cylinder_radius rel err %g" % (err,))

    # (M2)
    rec = keep(cone_semi_angle(cone_face))
    err = abs(rec["value"] - math.atan(0.2))
    assert err <= 1e-9, "M2 cone semi-angle err %g rad > 1e-9" % (err,)
    print("[ok] cone_semi_angle err %g rad" % (err,))

    # (M3)
    vf = math.pi * 0.1 * (0.03 * 0.03 + 0.03 * 0.01 + 0.01 * 0.01) / 3
    vc = math.pi * 0.03 * 0.03 * 0.1
    recf = keep(volume(frustum))
    recc = keep(volume(cylinder))
    recb = keep(volume(_fx_bowtie()))
    errf, errc = abs(recf["value"] - vf) / vf, abs(recc["value"] - vc) / vc
    assert errf <= 1e-8, "M3 frustum volume rel %g > 1e-8" % (errf,)
    assert errc <= 1e-8, "M3 cylinder volume rel %g > 1e-8" % (errc,)
    assert (recb["status"] == "refused" and recb["reason_id"] == "MEAS-NOSOLID"
            and recb["value"] is None), "M3 bow-tie not refused MEAS-NOSOLID: %r" % (recb,)
    print("[ok] volume frustum rel %g cylinder rel %g, bow-tie refused MEAS-NOSOLID" % (errf, errc))

    # (M4)
    six = [(cylinder, {"name": "cyl_mid", "x": 0.05}, 0.06),
           (frustum, {"name": "fr_start", "x": 0.0}, 0.06),
           (frustum, {"name": "fr_037", "x": 0.037}, 2 * (0.03 + (0.01 - 0.03) * 0.037 / 0.1)),
           (frustum, {"name": "fr_end", "x": 0.1}, 0.02),
           (fluid, cs, 0.06), (fluid, ep, 0.02)]
    worst = 0.0
    for shp, pl, truth_d in six:
        rec = keep(diameter_at_plane(shp, pl))
        worst = max(worst, abs(rec["value"] - truth_d) / truth_d)
    assert worst <= 1e-9, "M4 diameter worst rel %g > 1e-9" % (worst,)
    print("[ok] diameter_at_plane worst rel %g over six named planes" % (worst,))

    # (M5)
    rec = keep(diameter_at_plane(fluid, {"name": "mid", "x": 0.017}))
    assert (rec["status"] == "refused" and rec["reason_id"] == "MEAS-NOTCIRCLE"
            and rec["value"] is None), "M5 mid not NOTCIRCLE: %r" % (rec,)
    rec = keep(diameter_at_plane(fluid, {"name": "far", "x": 1.0}))
    assert (rec["status"] == "refused" and rec["reason_id"] == "MEAS-EMPTY"
            and rec["value"] is None), "M5 far not EMPTY: %r" % (rec,)
    rec = keep(diameter_at_plane(fluid, {"name": "bad name", "x": 0.0}))
    assert (rec["status"] == "refused" and rec["reason_id"] == "MEAS-BADPLANE"
            and rec["value"] is None), "M5 bad name not BADPLANE: %r" % (rec,)
    print("[ok] refusals MEAS-NOTCIRCLE MEAS-EMPTY MEAS-BADPLANE, all value None")

    # (M6)
    rec = keep(area_ratio(fluid, cs, ep))
    rel = abs(rec["value"] - 9.0) / 9.0
    assert rel <= 1e-9, "M6 area ratio rel %g > 1e-9" % (rel,)
    rec = keep(area_ratio(fluid, cs, {"name": "far", "x": 1.0}))
    assert (rec["status"] == "refused" and rec["reason_id"] == "MEAS-EMPTY"
            and rec["detail"].startswith("plane far: ")
            and rec["primitive"] == "area_ratio" and rec["unit"] == "1"
            and rec["where"] == ["contraction_start", "far"]), "M6 chained refusal wrong: %r" % (rec,)
    print("[ok] area_ratio 9 rel err %g; chained refusal 'plane far: ' MEAS-EMPTY" % (rel,))

    # (M7)
    rec = keep(extent_along_axis(fluid))
    rec_t = keep(extent_along_axis(_fx_torus()))
    err, err_t = abs(rec["value"] - 0.1), abs(rec_t["value"] - 0.12)
    assert err <= 1e-9 and err_t <= 1e-9, "M7 extent err %g / %g > 1e-9" % (err, err_t)
    print("[ok] extent_along_axis err %g m (fluid) %g m (torus)" % (err, err_t))

    # (M8)
    rec = keep(plane_distance(cs, ep))
    err = abs(rec["value"] - 0.06)
    assert (err <= 1e-15 and rec["where"] == ["contraction_start", "exit_plane"]), "M8 plane_distance %r" % (rec,)
    print("[ok] plane_distance err %g m, where ['contraction_start', 'exit_plane']" % (err,))

    # (M9)
    wet_c, out_n, out_r, alpha, t_band = _fx_cone_band()
    rec = keep(meridian_min_wall(wet_c, out_n))
    err = abs(rec["value"] - t_band)
    assert rec["status"] == "ok" and err <= 1e-9, "M9 cone band normal wall err %g > 1e-9" % (err,)
    print("[ok] meridian_min_wall cone band normal offset err %g m (t)" % (err,))

    # (M10)
    rec = keep(meridian_min_wall(wet_c, out_r))
    err = abs(rec["value"] - t_band * math.cos(alpha))
    assert rec["status"] == "ok" and err <= 1e-9, "M10 cone band radial wall err %g > 1e-9" % (err,)
    print("[ok] meridian_min_wall cone band radial offset = t cos(alpha) (%.5f mm), err %g m"
          % (t_band * math.cos(alpha) * 1e3, err))

    # (M11)
    truth = _fx_wall_truth(0.03, 0.003)
    err = abs(truth - 1.8759e-3)
    assert err <= 5e-9, "M11 wall truth %.12g off %g > 5e-9 from 1.8759e-3" % (truth, err)
    print("[ok] wall truth %.9f mm within %g m of 1.8759e-3" % (truth * 1e3, err))

    # (M12)
    wet_r, out_rr = _fx_wall("radial")
    rec12 = keep(meridian_min_wall(wet_r, out_rr))
    err = abs(rec12["value"] - truth)
    assert rec12["status"] == "ok" and err <= 1e-8, "M12 poly5 radial wall err %g > 1e-8" % (err,)
    print("[ok] meridian_min_wall poly5 L/D_i 0.5 radial err %g m vs dense truth" % (err,))

    # (M13)
    wet_n, out_nn = _fx_wall("normal")
    rec = keep(meridian_min_wall(wet_n, out_nn))
    err = abs(rec["value"] - 0.003)
    assert rec["status"] == "ok" and err <= 1e-8, "M13 poly5 normal wall err %g > 1e-8" % (err,)
    print("[ok] meridian_min_wall poly5 L/D_i 0.5 normal err %g m vs w = 0.003" % (err,))

    # (M14)
    def S(es):
        ax = gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0))
        return cq.Compound.makeCompound(
            [cq.Shape.cast(BRepPrimAPI_MakeRevol(e.wrapped, ax, 2 * math.pi).Shape()) for e in es])

    refusals = [keep(run("wall_distance_3d", wet_r, out_rr)), keep(wall_distance_3d()),
                keep(meridian_min_wall([S(wet_r)], [S(out_rr)])),
                keep(meridian_min_wall([wet_r[0].translate(V(0, 0, 1e-3))], out_rr))]
    assert all(r["status"] == "refused" and r["reason_id"] == "MEAS-3D-WALL" and r["value"] is None
               for r in refusals), "M14 not refused four ways: %r" % (refusals,)
    print("[ok] MEAS-3D-WALL refused four ways (run, direct, revolved surfaces, off-plane edge), value None each")

    # (M15)
    L, w, Lx = 0.03, 0.003, 0.01
    inner = [(L * i / 60, _fx_r(L * i / 60, L)) for i in range(61)]
    outer = [(x, _fx_r(x, L) + w) for x, _r0 in inner]
    sel_i = cq.Workplane("XY").moveTo(*inner[0]).spline(inner[1:], includeCurrent=True).lineTo(
        L + Lx, _FX_RE).edges().vals()
    sel_o = cq.Workplane("XY").moveTo(*outer[0]).spline(outer[1:], includeCurrent=True).lineTo(
        L + Lx, _FX_RE + w).edges().vals()
    assert len(sel_i) == 1 and sel_i[0].geomType() == "LINE", "M15 probe selection %r" % (
        [e.geomType() for e in sel_i],)
    rec = keep(meridian_min_wall(sel_i, sel_o))
    assert rec["status"] == "ok" and abs(rec["value"] - 0.003) <= 1e-9, "M15 probe wall %r" % (rec,)
    d3 = BRepExtrema_DistShapeShape(S(wet_r).wrapped, S(out_rr).wrapped)
    d3.Perform()
    err3 = abs(d3.Value() - truth)
    assert err3 <= 1e-8, "M15 3-D full generators err %g > 1e-8" % (err3,)
    print("[ok] trap: the probe's selection is 1 LINE edge reading 3.000 mm; the full generators read "
          "%.5f mm in 2-D and 3-D; 3-D stays refused by definition" % (rec12["value"] * 1e3,))

    # (M16)
    rec = keep(meridian_min_wall(wet_r, out_rr, n_dense=3))
    assert (rec["status"] == "refused" and rec["reason_id"] == "MEAS-XCHECK"
            and "BRepExtrema" in rec["detail"] and "dense" in rec["detail"]), "M16 n_dense=3 %r" % (rec,)
    assert "dense" in rec12["detail"], "M12 detail lacks the dense value: %r" % (rec12["detail"],)
    print("[ok] dense cross-check: n_dense=3 refused MEAS-XCHECK; n_dense=2001 agrees, detail '%s'"
          % (rec12["detail"],))

    # (M17)
    rec = keep(curvature_radius_min([_fx_arc()]))
    rel_arc = abs(rec["value"] - 0.05) / 0.05
    assert rel_arc <= 1e-9, "M17 arc curvature rel %g > 1e-9" % (rel_arc,)
    k1, k05 = _fx_poly5_kappa_truth(0.06), _fx_poly5_kappa_truth(0.03)
    rec1 = keep(curvature_radius_min([_fx_poly5_spline(0.06)]))
    rec05 = keep(curvature_radius_min([_fx_poly5_spline(0.03)]))
    rel1 = abs(rec1["value"] - k1) / k1
    rel05 = abs(rec05["value"] - k05) / k05
    assert max(rel1, rel05) <= 1e-3, "M17 poly5 curvature rel %g / %g > 1e-3" % (rel1, rel05)
    rec = keep(curvature_radius_min([cq.Edge.makeLine(V(0, 0.02, 0), V(0.05, 0.02, 0))]))
    assert rec["status"] == "refused" and rec["reason_id"] == "MEAS-NOCURV", "M17 straight line %r" % (rec,)
    print("[ok] curvature_radius_min arc rel %g, poly5 rel %g (L/D_i 1) and %g (0.5), straight refused MEAS-NOCURV"
          % (rel_arc, rel1, rel05))

    # (M18)
    rec = keep(slope_max([cq.Edge.makeLine(V(0, 0.03, 0), V(0.1, 0.01, 0))]))
    err = abs(rec["value"] - math.atan(0.2))
    assert err <= 1e-12, "M18 frustum slope err %g rad > 1e-12" % (err,)
    rec = keep(slope_max([_fx_poly5_spline(0.03)]))
    rel = abs(rec["value"] - math.atan(1.25)) / math.atan(1.25)
    assert rel <= 1e-3, "M18 poly5 slope rel %g > 1e-3" % (rel,)
    print("[ok] slope_max frustum err %g rad, poly5 rel %g of atan(1.25)" % (err, rel))

    # (M19)
    rec = keep(n_solids(fluid))
    rec2 = keep(n_solids(cq.Compound.makeCompound([fluid, cq.Solid.makeBox(0.01, 0.01, 0.01, pnt=V(1, 1, 1))])))
    assert rec["value"] == 1 and rec2["value"] == 2, "M19 n_solids %r / %r" % (rec["value"], rec2["value"])
    recv = keep(valid(fluid))
    recb = keep(valid(_fx_bowtie()))
    assert recv["value"] == 1, "M19 fluid not valid: %r" % (recv,)
    assert recb["value"] == 0 and "BRepCheck=False" in recb["detail"], "M19 bow-tie %r" % (recb,)
    print("[ok] n_solids fluid 1 compound 2; valid fluid 1, bow-tie 0 with BRepCheck=False")

    # (M20)
    plane_face = [f for f in cylinder.Faces() if f.geomType() == "PLANE"][0]
    off = cq.Edge.makeLine(V(0, 0.02, 0), V(0.05, 0.02, 0)).translate(V(0, 0, 1e-3))
    guards = [keep(cylinder_radius(None)), keep(run("no_such")), keep(cylinder_radius(plane_face)),
              keep(cone_semi_angle(cyl_face)), keep(slope_max([off]))]
    ids = [r["reason_id"] for r in guards]
    assert ids == ["MEAS-ERROR", "MEAS-UNKNOWN", "MEAS-NOTCYL", "MEAS-NOTCONE", "MEAS-NOTMERIDIAN"], (
        "M20 guard ids %r" % (ids,))
    assert guards[0]["status"] == "error", "M20 None face status %r" % (guards[0]["status"],)
    print("[ok] guards: MEAS-ERROR MEAS-UNKNOWN MEAS-NOTCYL MEAS-NOTCONE MEAS-NOTMERIDIAN")

    # (M21)
    bad = [r for r in seen if schema.errors(r, "cad-measure/1") != []]
    statuses = set(r["status"] for r in seen)
    assert not bad, "M21 %d records fail the schema, first: %r" % (len(bad), bad[0])
    assert len(seen) >= 40 and {"ok", "refused", "error"} <= statuses, (
        "M21 %d records, statuses %r" % (len(seen), sorted(statuses)))
    print("[ok] %d records pass the cad-measure/1 schema; statuses ok refused error" % (len(seen),))
    print("SELFTEST PASS")
    return 0


def main(argv):
    if argv == ["--selftest"]:
        return selftest()
    sys.stderr.write("usage: python measure.py --selftest" + chr(10))
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
