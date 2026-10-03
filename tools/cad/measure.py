#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""measure.py - the typed measurement primitives of the CAD loop (docs/16 §E.2, gate GC-1 of §H.3): each returns one cad-measure/1 record; wall thickness is measured on the 2-D meridian, or in 3-D only between complete tagged face sets.

Integral properties are adaptive (docs/16a §B.2): the idea of passing an eps to BRepGProp and keeping the returned
error estimate is from Amagine3D (https://github.com/amagine-ai/Amagine3D, e608dc6,
skills/text-a3d/brep_measurements.py, _surface_properties), reimplemented here; no code was copied.
Sections on an explicit plane (islands, holes as inner wires, the cutting face recentred on the solid's projection) are an
idea from the same Amagine3D file's measure_section, reimplemented in raw OCP; the solid is scaled to a 1e5 bounding-box
diagonal first because the kernel's plane-surface intersection tolerance is absolute (2.09e-6 rel off in metres on the trap
station, 1.7e-13 at 1e5); no code was copied.
A 3-D wall distance is measured only between COMPLETE tagged face sets (docs/16a D-9, §D.15): wall_min_tagged takes
BRepExtrema between the two selections after proving each selection's adaptive area equals its geom.json tag row
(MEAS-COVER), refuses sets that touch (MEAS-TOUCH) and cross-checks the distance against a BRep normal-ray field
sampled outside a boundary band (MEAS-XCHECK); an untagged or free-form pair stays refused MEAS-3D-WALL. The ray
field's report (minimum, area-weighted p05, violating-area ratio, a full sample account and the argmin location) is
the report of Amagine3D's skills/text-a3d/qa_check.py thickness_observation, reimplemented around a true BRep normal
ray (BRepClass_FaceClassifier inside the face, IntCurvesFace_ShapeIntersector along the inward normal); their
max-sphere method is not taken and no code was copied.

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
from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Circle, GeomAbs_Plane, GeomAbs_SurfaceOfRevolution
from OCP.BRepAlgoAPI import BRepAlgoAPI_Section, BRepAlgoAPI_Check, BRepAlgoAPI_Common
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_Transform
from OCP.gp import gp_Pln, gp_Pnt, gp_Dir, gp_Vec, gp_Ax1, gp_Ax3, gp_Trsf, gp_Lin, gp_Pnt2d
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepLProp import BRepLProp_CLProps, BRepLProp_SLProps
from OCP.BRepPrimAPI import BRepPrimAPI_MakeRevol   # selftest only
from OCP.GCPnts import GCPnts_AbscissaPoint
from OCP.BRep import BRep_Tool
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_WIRE, TopAbs_SOLID, TopAbs_IN, TopAbs_REVERSED
from OCP.TopoDS import TopoDS
from OCP.BRepTools import BRepTools
from OCP.BRepClass import BRepClass_FaceClassifier
from OCP.IntCurvesFace import IntCurvesFace_ShapeIntersector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import schema
import turb_integral

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
    "watertight": ("abs", 0.0), "axis_x": ("abs", 0.0), "units_m": ("abs", 0.0), "section_at_plane": ("rel", 1e-9),
    "wall_min_tagged": ("abs", 1e-8), "k_max_1d": ("rel", 1e-9),
}
REFUSED = {"wall_distance_3d": "MEAS-3D-WALL"}
UNITS_TOL = 1e-9          # m: gmsh vs BREP x-span in geom.json (docs/16 §H.3 GC-5)
GPROP_EPS = 1e-12         # rel eps of every adaptive integral (BRepGProp Eps overloads, GCPnts length)
GPROP_EPS_CHECK = 1e-9    # the coarser length pass; |L(GPROP_EPS) - L(GPROP_EPS_CHECK)| / L is the length's estimate
GPROP_REL_MAX = 1e-8      # rel: the export helpers' bound, the volume primitive's u_meas (docs/16 §E.2)
SECTION_DIAG = 1e5        # model units: each solid is scaled to this bounding-box diagonal before the cut (the kernel's section tolerance is absolute)
SECTION_PERP_TOL = 1e-12  # |n . u| of the unit normal and unit u axis above this is refused MEAS-BADPLANE
COVER_REL = 1e-8          # rel: a tag selection's adaptive area must equal its geom.json row's area_m2 (MEAS-COVER)
TOUCH_TOL = 1e-9          # m: tagged sets nearer than this touch, and distance 0 is not a wall (MEAS-TOUCH, 16a §D.10)
RAY_N = 81                # ray samples per face and parameter direction, at cell midpoints (16a G.1 AMG-10: 81 x 81)
RAY_BAND = 0.02           # the boundary band: no ray within this fraction of either end of u or v (16a §D.14)
RAY_XCHECK_REL = 5e-4     # rel: the ray-field minimum must lie in [extrema - RAY_XCHECK_ABS, extrema (1 + this)]
RAY_XCHECK_ABS = 1e-12    # m: the lower slack of the cross-check (a normal ray is never shorter than the extrema)
RAY_TOL = 1e-9            # the classifier, normal and intersector tolerance
RAY_W_MIN_REL = 1e-6      # a hit nearer than this times the solid's bounding-box diagonal is the ray's own start
RAY_P = 0.05              # the area-weighted percentile of the ray-field report


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


def gprop(shape, kind):
    """(mass, est_rel) by adaptive integration at GPROP_EPS.

    'volume' and 'area' use BRepGProp's Eps overloads, which return the relative error estimate; 'length' sums
    GCPnts_AbscissaPoint.Length at GPROP_EPS over every non-degenerate edge occurrence (this OCP has no Eps
    overload of LinearProperties), its estimate the relative difference to a pass at GPROP_EPS_CHECK (0.0 when the
    length is 0). The default quadrature is off by -2.16e-4 on a 200-span spline solid (docs/16a §D.13).
    """
    w = shape.wrapped if hasattr(shape, "wrapped") else shape
    if kind in ("volume", "area"):
        p = GProp_GProps()
        if kind == "volume":
            est = BRepGProp.VolumeProperties_s(w, p, GPROP_EPS, False)
        else:
            est = BRepGProp.SurfaceProperties_s(w, p, GPROP_EPS, False)
        return float(p.Mass()), float(est)
    if kind == "length":
        total, diff = 0.0, 0.0
        ex = TopExp_Explorer(w, TopAbs_EDGE)
        while ex.More():
            e = TopoDS.Edge_s(ex.Current())
            if not BRep_Tool.Degenerated_s(e):
                c = BRepAdaptor_Curve(e)
                fine = GCPnts_AbscissaPoint.Length_s(c, GPROP_EPS)
                total += fine
                diff += abs(fine - GCPnts_AbscissaPoint.Length_s(c, GPROP_EPS_CHECK))
            ex.Next()
        return total, (diff / total if total > 0.0 else 0.0)
    raise ValueError("gprop kind %r is not volume, area or length" % (kind,))


def volume(shape, feature=None):
    method = "BRepGProp adaptive volume"
    try:
        if len(shape.Solids()) == 0:
            return _refused("volume", "m3", method, "MEAS-NOSOLID",
                            "the shape holds no solid", feature)
        v, est = gprop(shape, "volume")
        rec = _ok("volume", v, "m3", method, feature=feature,
                  detail="eps %.0e; relative error estimate %.3e" % (GPROP_EPS, est))
        if rec["status"] == "ok" and not (est * abs(v) <= rec["u_meas"]):
            return _refused("volume", "m3", method, "MEAS-GPROP",
                            "eps %.0e; relative error estimate %.3e exceeds u_meas %.3e m3"
                            % (GPROP_EPS, est, rec["u_meas"]), feature)
        return rec
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


def units_m(geom, feature=None):
    """1 iff geom.json says m, scale 1, a METRE STEP, and gmsh read that STEP at the BREP's own x-span."""
    method = "geom.json: units m, scale 1, STEP length unit METRE, gmsh (OCCTargetUnit M) x-span = BREP x-span within 1e-9 m"
    try:
        if not isinstance(geom, dict):
            return _refused("units_m", "1", method, "MEAS-BADGEOM", "not a geom.json dict", feature)
        sc, sb, sg = geom["scale"], geom["x_span_m"], geom["gmsh_import"]["x_span_m"]
        nums = all(not isinstance(v, bool) and isinstance(v, (int, float)) for v in (sc, sb, sg))
        ok = (nums and geom["units"] == "m" and sc == 1 and geom["step_length_unit"] == "METRE"
              and abs(sg - sb) <= UNITS_TOL)
        return _ok("units_m", 1 if ok else 0, "1", method, feature=feature,
                   detail="units %r scale %r step %r; x-span brep %r gmsh %r" % (geom["units"], sc,
                                                                               geom["step_length_unit"], sb, sg))
    except (KeyError, TypeError) as e:
        return _refused("units_m", "1", method, "MEAS-BADGEOM", "geom field missing: %s" % (e,), feature)
    except Exception as e:
        return _error("units_m", "1", method, "MEAS-ERROR", ("%s: %s" % (type(e).__name__, e))[:300], feature)


def axis_x(shape, feature=None):
    """1 iff every face is a plane normal to x or a surface revolved about the x axis itself."""
    method = "every face a plane with normal +-x, or a cylinder, cone or surface of revolution about the x axis"
    try:
        faces = shape.Faces()
        if not faces:
            return _refused("axis_x", "1", method, "MEAS-EMPTY", "the shape has no face", feature)
        for i, f in enumerate(faces):
            a = BRepAdaptor_Surface(f.wrapped)
            t = a.GetType()
            if t == GeomAbs_Plane:
                ax, on_axis = a.Plane().Axis(), False
            elif t == GeomAbs_Cylinder:
                ax, on_axis = a.Cylinder().Axis(), True
            elif t == GeomAbs_Cone:
                ax, on_axis = a.Cone().Axis(), True
            elif t == GeomAbs_SurfaceOfRevolution:
                ax, on_axis = a.AxeOfRevolution(), True
            else:
                return _ok("axis_x", 0, "1", method, feature=feature,
                           detail="face %d is %s, not a plane or a surface of revolution" % (i, t))
            dd, loc = ax.Direction(), ax.Location()
            if abs(abs(dd.X()) - 1.0) > AXIS_TOL:
                return _ok("axis_x", 0, "1", method, feature=feature,
                           detail="face %d axis direction (%r, %r, %r) is not +-x" % (i, dd.X(), dd.Y(), dd.Z()))
            if on_axis and (abs(loc.Y()) > AXIS_TOL or abs(loc.Z()) > AXIS_TOL):
                return _ok("axis_x", 0, "1", method, feature=feature,
                           detail="face %d axis passes (%r, %r) off the x axis" % (i, loc.Y(), loc.Z()))
        return _ok("axis_x", 1, "1", method, feature=feature, detail="%d faces on the x axis" % (len(faces),))
    except Exception as e:
        return _error("axis_x", "1", method, "MEAS-ERROR", ("%s: %s" % (type(e).__name__, e))[:300], feature)


def watertight(report, feature=None):
    """1 iff an stl_repair --weld 0 report says closed before and after with nothing repaired, one component."""
    method = "stl_repair --weld 0 report: closed before and after, 0 open, 0 non-manifold, 0 reoriented, 0 flipped, 0 filled, 0 dropped, 1 component"
    try:
        if not isinstance(report, dict) or report.get("tool") != "stl_repair":
            return _refused("watertight", "1", method, "MEAS-BADREPORT", "not an stl_repair report", feature)
        tol = report["weld"]["tol_rel"]
        if tol != 0:
            return _refused("watertight", "1", method, "MEAS-BADREPORT",
                            "the report was made with --weld %r, not --weld 0" % (tol,), feature)
        o, a = report["orientation"], report["after"]
        counts = (a["open_edges"], a["non_manifold_edges"], o["reoriented_triangles"], o["flipped_components"],
                  report["holes"]["filled"], report["degenerate_dropped"])
        ok = (bool(report["before"]["closed"]) and bool(a["closed"]) and counts == (0, 0, 0, 0, 0, 0)
              and report["n_components"] == 1)
        return _ok("watertight", 1 if ok else 0, "1", method, feature=feature,
                   detail="before closed %s; after closed %s; open %d non-manifold %d reoriented %d flipped %d "
                          "filled %d dropped %d; components %d" % ((report["before"]["closed"], a["closed"])
                                                                   + counts + (report["n_components"],)))
    except (KeyError, TypeError) as e:
        return _refused("watertight", "1", method, "MEAS-BADREPORT", "report field missing: %s" % (e,), feature)
    except Exception as e:
        return _error("watertight", "1", method, "MEAS-ERROR", ("%s: %s" % (type(e).__name__, e))[:300], feature)


def _vec3(a):
    """A finite non-bool 3-vector as a numpy array, or None."""
    if not isinstance(a, (list, tuple)) or len(a) != 3:
        return None
    if any(isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(float(c)) for c in a):
        return None
    return np.array([float(c) for c in a])


def _section_plane(p):
    """(origin, n, u, v) of a {name, origin, normal, u} plane dict, n and u unit and perpendicular, or None."""
    if not isinstance(p, dict):
        return None
    if not isinstance(p.get("name"), str) or not _NAME_RE.fullmatch(p.get("name")):
        return None
    origin, nv, uv = _vec3(p.get("origin")), _vec3(p.get("normal")), _vec3(p.get("u"))
    if origin is None or nv is None or uv is None:
        return None
    if not np.linalg.norm(nv) > 0 or not np.linalg.norm(uv) > 0:
        return None
    n, u = nv / np.linalg.norm(nv), uv / np.linalg.norm(uv)
    if abs(float(np.dot(n, u))) > SECTION_PERP_TOL:
        return None
    return origin, n, u, np.cross(n, u)


def _uv_extent(shape, d, o_s, k):
    """(lo, hi) in metres along the unit direction d relative to o_s, by BRepExtrema distances to far planes."""
    b = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape, b, False, False)
    x0, y0, z0, x1, y1, z1 = b.Get()
    cs = np.array([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
    m = 10.0 * math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2) + 1.0
    gap = {}
    for sgn in (-1, +1):
        pf = cs + sgn * m * d
        far = BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(*pf), gp_Dir(*d)), -3 * m, 3 * m, -3 * m, 3 * m).Face()
        dd = BRepExtrema_DistShapeShape(shape, far)
        dd.Perform()
        if not dd.IsDone():
            raise RuntimeError("BRepExtrema not done")
        gap[sgn] = dd.Value()
    base = float(np.dot(cs - o_s, d))
    return ((base - (m - gap[-1])) / k, (base + (m - gap[1])) / k)


def _section_islands(solid, origin, n, u, diag_units):
    """The islands of one solid cut by a planar face recentred on its projection, as a list of dicts.

    solid is a TopoDS_Shape of one solid; diag_units is SECTION_DIAG in production and None for the unscaled
    selftest cut (k = 1). The transform copies, so the caller's solid is never modified.
    """
    b = Bnd_Box()
    BRepBndLib.AddOptimal_s(solid, b, False, False)
    x0, y0, z0, x1, y1, z1 = b.Get()
    c = np.array([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
    diag = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)
    k = 1.0 if diag_units is None else diag_units / diag
    t = gp_Trsf()
    t.SetScale(gp_Pnt(*c), k)
    s = BRepBuilderAPI_Transform(solid, t, True).Shape()
    o_s = c + k * (origin - c)
    p0 = c - np.dot(c - o_s, n) * n
    h = 2.0 * k * diag
    face = BRepBuilderAPI_MakeFace(gp_Pln(gp_Ax3(gp_Pnt(*p0), gp_Dir(*n), gp_Dir(*u))), -h, h, -h, h).Face()
    op = BRepAlgoAPI_Common(s, face)
    op.Build()
    if not op.IsDone():
        raise RuntimeError("BRepAlgoAPI_Common failed")
    v = np.cross(n, u)
    islands = []
    ex = TopExp_Explorer(op.Shape(), TopAbs_FACE)
    while ex.More():
        f = TopoDS.Face_s(ex.Current())
        ex.Next()
        a_s, est = gprop(f, "area")
        if not a_s > 0:
            continue
        ow = BRepTools.OuterWire_s(f)
        holes = []
        ew = TopExp_Explorer(f, TopAbs_WIRE)
        while ew.More():
            w = TopoDS.Wire_s(ew.Current())
            ew.Next()
            if not w.IsSame(ow):
                holes.append(w)
        lo, elo = gprop(ow, "length")
        lh, elh = [], []
        for w in holes:
            li, ei = gprop(w, "length")
            lh.append(li)
            elh.append(ei)
        umin, umax = _uv_extent(ow, u, o_s, k)
        vmin, vmax = _uv_extent(ow, v, o_s, k)
        islands.append({"area_m2": a_s / k ** 2, "area_est_rel": est, "outer_perimeter_m": lo / k,
                        "hole_perimeters_m": [li / k for li in lh], "perimeter_est_rel": max([elo] + elh),
                        "envelope_uv_m": [umin, umax, vmin, vmax]})
    return islands


def section_props(shape, plane):
    """The section of one solid at an explicit {name, origin, normal, u} plane as ONE plain dict (AMG-5 reads it).

    status is ok, refused or error; a plane that misses is refused MEAS-NOSECTION, never an area of 0.
    """
    sp = {"name": None, "origin": None, "normal": None, "u": None, "v": None, "diag_units": SECTION_DIAG,
          "status": "error", "reason_id": None, "detail": "", "area_m2": None, "area_est_rel": None,
          "n_islands": 0, "n_holes": 0, "outer_perimeter_m": None, "hole_perimeter_m": None,
          "perimeter_est_rel": None, "dh_m": None, "envelope_uv_m": None, "islands": []}
    try:
        p = _section_plane(plane)
        if p is None:
            sp["status"] = "refused"
            sp["reason_id"] = "MEAS-BADPLANE"
            sp["detail"] = ("the plane is not a {name, origin, normal, u} dict with finite 3-vectors, "
                            "nonzero normal and u, and u normal to the normal")
            return sp
        origin, n, u, v = p
        sp["name"] = plane["name"]
        sp["origin"] = [float(q) for q in origin]
        sp["normal"] = [float(q) for q in n]
        sp["u"] = [float(q) for q in u]
        sp["v"] = [float(q) for q in v]
        solids = shape.Solids()
        if len(solids) == 0:
            sp["status"] = "refused"
            sp["reason_id"] = "MEAS-NOSOLID"
            sp["detail"] = "the shape holds no solid"
            return sp
        if len(solids) > 1:
            sp["status"] = "refused"
            sp["reason_id"] = "MEAS-MULTISOLID"
            sp["detail"] = ("the shape holds %d solids; a section measures one "
                            "(summed islands of overlapping solids double count)" % (len(solids),))
            return sp
        islands = _section_islands(solids[0].wrapped, origin, n, u, SECTION_DIAG)
        if not islands:
            sp["status"] = "refused"
            sp["reason_id"] = "MEAS-NOSECTION"
            sp["detail"] = "the plane %s misses the solid" % (plane["name"],)
            return sp
        sp["islands"] = [dict({"island_index": i}, **isl) for i, isl in enumerate(islands)]
        area = sum(isl["area_m2"] for isl in islands)
        sp["status"] = "ok"
        sp["area_m2"] = area
        sp["area_est_rel"] = sum(isl["area_est_rel"] * isl["area_m2"] for isl in islands) / area
        sp["n_islands"] = len(islands)
        sp["n_holes"] = sum(len(isl["hole_perimeters_m"]) for isl in islands)
        sp["outer_perimeter_m"] = sum(isl["outer_perimeter_m"] for isl in islands)
        sp["hole_perimeter_m"] = sum(sum(isl["hole_perimeters_m"]) for isl in islands)
        sp["perimeter_est_rel"] = max(isl["perimeter_est_rel"] for isl in islands)
        sp["dh_m"] = 4 * area / (sp["outer_perimeter_m"] + sp["hole_perimeter_m"])
        sp["envelope_uv_m"] = [min(isl["envelope_uv_m"][0] for isl in islands),
                               max(isl["envelope_uv_m"][1] for isl in islands),
                               min(isl["envelope_uv_m"][2] for isl in islands),
                               max(isl["envelope_uv_m"][3] for isl in islands)]
        return sp
    except Exception as e:
        sp["status"] = "error"
        sp["reason_id"] = "MEAS-ERROR"
        sp["detail"] = ("%s: %s" % (type(e).__name__, e))[:300]
        return sp


def section_at_plane(shape, plane, feature=None):
    """One cad-measure/1 record of the section of one solid at an explicit plane (docs/16a §B.2, AMG-2).

    The plane is explicit; islands are the positive-area faces of solid ∩ a planar face recentred on the solid's
    projection, holes are inner wires, and the solid is scaled to SECTION_DIAG first because the kernel's
    plane-surface intersection tolerance is absolute (in metres the trap station was 2.09e-6 rel off, at 1e5
    1.7e-13). The in-plane envelope is measured from the input origin along u and v = normal × u by BRepExtrema to
    far planes; a plane that misses is refused MEAS-NOSECTION, never 0.
    """
    method = "BRepAlgoAPI_Common with a recentred planar face at bounding-box diagonal 1e5, adaptive BRepGProp"
    where = (plane["name"],) if _section_plane(plane) is not None else ()
    sp = section_props(shape, plane)
    if sp["status"] == "refused":
        return _refused("section_at_plane", "m2", method, sp["reason_id"], sp["detail"], feature, where)
    if sp["status"] == "error":
        return _error("section_at_plane", "m2", method, sp["reason_id"], sp["detail"], feature, where)
    D = ("islands %d holes %d; outer perimeter %.12g m; hole perimeter %.12g m; 4A/P %.12g m; "
         "envelope u [%.12g, %.12g] v [%.12g, %.12g] m; eps %.0e; area estimate %.3e"
         % (sp["n_islands"], sp["n_holes"], sp["outer_perimeter_m"], sp["hole_perimeter_m"], sp["dh_m"],
            sp["envelope_uv_m"][0], sp["envelope_uv_m"][1], sp["envelope_uv_m"][2], sp["envelope_uv_m"][3],
            GPROP_EPS, sp["area_est_rel"]))
    rec = _ok("section_at_plane", sp["area_m2"], "m2", method, feature=feature, where=where, detail=D)
    est = sp["area_est_rel"]
    if rec["status"] == "ok" and not (est * abs(sp["area_m2"]) <= rec["u_meas"]):
        return _refused("section_at_plane", "m2", method, "MEAS-GPROP",
                        "eps %.0e; relative error estimate %.3e exceeds u_meas %.3e m2"
                        % (GPROP_EPS, est, rec["u_meas"]), feature, where)
    return rec


def tag_faces(shape, rows, name):
    """The tag selection {name, faces, area_m2} of one row of a geom.json face-tag group (geom["tags"]["fluid_faces"]):
    faces are shape.Faces()[i] for the row's index list and area_m2 is the row's recorded area."""
    row = rows[name]
    fs = shape.Faces()
    return {"name": name, "faces": [fs[i] for i in row["index"]], "area_m2": row["area_m2"]}


def _tag_ok(tag):
    """True iff tag is exactly {name, faces, area_m2}: a template name, a list of cq.Face, a finite area > 0."""
    if not isinstance(tag, dict) or set(tag) != {"name", "faces", "area_m2"}:
        return False
    name, faces, a = tag["name"], tag["faces"], tag["area_m2"]
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        return False
    if isinstance(a, bool) or not isinstance(a, (int, float)) or not math.isfinite(float(a)) or not a > 0:
        return False
    return isinstance(faces, (list, tuple)) and all(isinstance(f, cq.Face) for f in faces)


def _face_indices(own, faces):
    """Each face's index in own (the shape's faces, IsSame); None if one is missing or selected twice."""
    idx = []
    for f in faces:
        hit = [k for k, g in enumerate(own) if g.wrapped.IsSame(f.wrapped)]
        if not hit or hit[0] in idx:
            return None
        idx.append(hit[0])
    return idx


def _cover(own, tag):
    """(indices, rel, detail): the selection is distinct faces of the shape whose adaptive area equals area_m2."""
    idx = _face_indices(own, tag["faces"])
    if idx is None:
        return None, None, "tag %s: a selected face is not a face of the shape, or is selected twice" % (tag["name"],)
    a = sum(gprop(own[k], "area")[0] for k in idx)
    rel = (a - tag["area_m2"]) / tag["area_m2"]
    if not abs(rel) <= COVER_REL:
        return None, rel, ("tag %s: the %d selected faces hold %.12g m2 against the tag's %.12g m2 (rel %+.3e, over "
                           "%.0e): the selection is not the complete tag" % (tag["name"], len(idx), a, tag["area_m2"],
                                                                            rel, COVER_REL))
    return idx, rel, ""


def _ray_cast(X, own, src, dst, name, n, band, w_min, acc, out):
    """Cast one normal ray per band-free, inside, midpoint sample of every src face into the material; a sample is
    valid when the first hit beyond w_min lies on a dst face. Appends (w, weight, name, k, u, v, P, H) to out."""
    for k in src:
        f = own[k]
        a = BRepAdaptor_Surface(f.wrapped)
        u0, u1, v0, v1 = a.FirstUParameter(), a.LastUParameter(), a.FirstVParameter(), a.LastVParameter()
        du, dv = (u1 - u0) / n, (v1 - v0) / n
        s = 1.0 if f.wrapped.Orientation() == TopAbs_REVERSED else -1.0
        for i in range(n):
            fu = (i + 0.5) / n
            for j in range(n):
                fv = (j + 0.5) / n
                acc["candidate"] += 1
                if fu < band or fu > 1.0 - band or fv < band or fv > 1.0 - band:
                    acc["banded"] += 1
                    continue
                u, v = u0 + fu * (u1 - u0), v0 + fv * (v1 - v0)
                if BRepClass_FaceClassifier(f.wrapped, gp_Pnt2d(u, v), RAY_TOL).State() != TopAbs_IN:
                    acc["outside"] += 1
                    continue
                acc["selected"] += 1
                pr = BRepLProp_SLProps(a, u, v, 1, RAY_TOL)
                wgt = pr.D1U().Crossed(pr.D1V()).Magnitude() * du * dv
                acc["selected_area_m2"] += wgt
                if not pr.IsNormalDefined():
                    acc["no_normal"] += 1
                    continue
                N, P = pr.Normal(), pr.Value()
                X.Perform(gp_Lin(P, gp_Dir(s * N.X(), s * N.Y(), s * N.Z())), 0.0, 1e100)
                hits = [(X.WParameter(h), h) for h in range(1, X.NbPnt() + 1) if X.WParameter(h) > w_min]
                if not hits:
                    acc["no_hit"] += 1
                    continue
                w, h = min(hits)
                if not any(X.Face(h).IsSame(own[d].wrapped) for d in dst):
                    acc["other_face"] += 1
                    continue
                acc["valid"] += 1
                H = X.Pnt(h)
                out.append((w, wgt, name, k, u, v, [P.X(), P.Y(), P.Z()], [H.X(), H.Y(), H.Z()]))


def wall_ray_field(shape, tag_a, tag_b, n=RAY_N, band=RAY_BAND, target_m=None):
    """The BRep normal-ray wall field between two tag selections of ONE solid, as one plain dict (a diagnostic, never
    a gate): rays from each set's faces along the inward normal, a sample valid when its first hit is on the other
    set. Reports min_m (both directions), min_ab_m, min_ba_m, the area-weighted p05_m, violating_area_ratio (valid
    area thinner than target_m, None without a target), the sample account and the argmin location."""
    rep = {"status": "error", "reason_id": None, "detail": "", "tags": [None, None], "n": n, "band": band,
           "target_m": target_m, "min_m": None, "min_ab_m": None, "min_ba_m": None, "p05_m": None,
           "violating_area_ratio": None, "account": None, "argmin": None}
    try:
        for t in (tag_a, tag_b):
            if (not isinstance(t, dict) or not isinstance(t.get("name"), str) or not _NAME_RE.fullmatch(t["name"])
                    or not isinstance(t.get("faces"), (list, tuple)) or not t["faces"]
                    or not all(isinstance(f, cq.Face) for f in t["faces"])):
                rep.update(status="refused", reason_id="MEAS-EMPTY",
                           detail="a tag is not {name, faces} with at least one cq.Face")
                return rep
        rep["tags"] = [tag_a["name"], tag_b["name"]]
        solids = shape.Solids()
        if len(solids) != 1:
            rep.update(status="refused", reason_id="MEAS-NOSOLID" if not solids else "MEAS-MULTISOLID",
                       detail="the shape holds %d solids; a ray field is cast in one" % (len(solids),))
            return rep
        own = shape.Faces()
        ia, ib = _face_indices(own, tag_a["faces"]), _face_indices(own, tag_b["faces"])
        if ia is None or ib is None:
            rep.update(status="refused", reason_id="MEAS-COVER",
                       detail="a selected face is not a face of the solid, or is selected twice")
            return rep
        bb = Bnd_Box()
        BRepBndLib.AddOptimal_s(solids[0].wrapped, bb, False, False)
        w_min = RAY_W_MIN_REL * math.sqrt(bb.SquareExtent())
        X = IntCurvesFace_ShapeIntersector()
        X.Load(solids[0].wrapped, RAY_TOL)
        acc = {"candidate": 0, "banded": 0, "outside": 0, "selected": 0, "valid": 0, "no_normal": 0, "no_hit": 0,
               "other_face": 0, "selected_area_m2": 0.0}
        ab, ba = [], []
        _ray_cast(X, own, ia, ib, tag_a["name"], n, band, w_min, acc, ab)
        _ray_cast(X, own, ib, ia, tag_b["name"], n, band, w_min, acc, ba)
        total = sum(gprop(own[k], "area")[0] for k in ia + ib)
        sel_area = acc.pop("selected_area_m2")
        acc["invalid"] = acc["no_normal"] + acc["no_hit"] + acc["other_face"]
        acc["sampled_area_ratio"] = sel_area / total
        rep["account"] = acc
        rep["status"] = "ok"
        rep["min_ab_m"] = min(q[0] for q in ab) if ab else None
        rep["min_ba_m"] = min(q[0] for q in ba) if ba else None
        allq = ab + ba
        if not allq:
            rep["detail"] = "no valid sample: no normal ray from either set reaches the other"
            return rep
        vals = np.array([q[0] for q in allq])
        wts = np.array([q[1] for q in allq])
        order = np.argsort(vals, kind="stable")
        cum = np.cumsum(wts[order])
        p = int(np.searchsorted(cum, cum[-1] * RAY_P, side="left"))
        best = allq[int(order[0])]
        rep["min_m"] = float(best[0])
        rep["p05_m"] = float(vals[order][min(p, len(vals) - 1)])
        if target_m is not None:
            rep["violating_area_ratio"] = float(wts[vals < target_m].sum() / wts.sum())
        rep["argmin"] = {"from": best[2], "face_index": best[3], "uv": [best[4], best[5]], "point_m": best[6],
                         "hit_m": best[7]}
        return rep
    except Exception as e:
        rep.update(status="error", reason_id="MEAS-ERROR", detail=("%s: %s" % (type(e).__name__, e))[:300])
        return rep


def wall_min_tagged(shape, tag_a, tag_b, feature=None, n_ray=RAY_N):
    """3-D minimum wall between two COMPLETE tag selections of one solid (docs/16a D-9): MEAS-COVER, BRepExtrema,
    MEAS-TOUCH, then the ray-field cross-check MEAS-XCHECK. An untagged set is refused MEAS-3D-WALL."""
    method = "BRepExtrema_DistShapeShape between complete tagged face sets, BRep normal-ray cross-check"
    try:
        if not _tag_ok(tag_a) or not _tag_ok(tag_b):
            return _refused("wall_min_tagged", "m", method, "MEAS-3D-WALL",
                            "an untagged set: a 3-D wall distance is measured only between complete tagged face "
                            "sets {name, faces, area_m2} (docs/16a D-9); a revolved wall is meridian_min_wall",
                            feature)
        where = (tag_a["name"], tag_b["name"])
        if not tag_a["faces"] or not tag_b["faces"]:
            return _refused("wall_min_tagged", "m", method, "MEAS-EMPTY", "a tag selects no face", feature, where)
        own = shape.Faces()
        idx, rels = [], []
        for t in (tag_a, tag_b):
            i, rel, why = _cover(own, t)
            if i is None:
                return _refused("wall_min_tagged", "m", method, "MEAS-COVER", why, feature, where)
            idx.append(i)
            rels.append(rel)
        d = BRepExtrema_DistShapeShape(cq.Compound.makeCompound([own[k] for k in idx[0]]).wrapped,
                                       cq.Compound.makeCompound([own[k] for k in idx[1]]).wrapped)
        d.Perform()
        if not d.IsDone():
            return _error("wall_min_tagged", "m", method, "MEAS-ERROR", "BRepExtrema_DistShapeShape not done",
                          feature, where)
        ext = d.Value()
        if not ext > TOUCH_TOL:
            return _refused("wall_min_tagged", "m", method, "MEAS-TOUCH",
                            "the tagged sets %s and %s touch: BRepExtrema %.3g m is not above %.0e m, and distance 0 "
                            "is not a wall" % (where[0], where[1], ext, TOUCH_TOL), feature, where)
        rf = wall_ray_field(shape, tag_a, tag_b, n=n_ray)
        if rf["status"] != "ok":
            return _refused("wall_min_tagged", "m", method, rf["reason_id"], rf["detail"], feature, where)
        acc = rf["account"]
        if rf["min_m"] is None:
            return _refused("wall_min_tagged", "m", method, "MEAS-XCHECK",
                            "BRepExtrema %.12g m, but no normal ray from either set reaches the other outside the band "
                            "(%d selected samples, %d on another face)" % (ext, acc["selected"], acc["other_face"]),
                            feature, where)
        lo, hi = ext - RAY_XCHECK_ABS, ext * (1.0 + RAY_XCHECK_REL)
        if not lo <= rf["min_m"] <= hi:
            return _refused("wall_min_tagged", "m", method, "MEAS-XCHECK",
                            "BRepExtrema %.12g m vs ray-field minimum %.12g m outside [%.12g, %.12g] m (n %d, band %g)"
                            % (ext, rf["min_m"], lo, hi, n_ray, RAY_BAND), feature, where)
        return _ok("wall_min_tagged", ext, "m", method, feature=feature, where=where,
                   detail="ray min %.12g m over %d valid samples (n %d, band %g); cover rel %+.2e %+.2e"
                   % (rf["min_m"], acc["valid"], n_ray, RAY_BAND, rels[0], rels[1]))
    except Exception as e:
        return _error("wall_min_tagged", "m", method, "MEAS-ERROR", ("%s: %s" % (type(e).__name__, e))[:300], feature)


def k_max_1d(params, re_de, feature=None, where=("wall_contraction",)):
    """The a priori acceleration parameter K_max of docs/16 §H.5 item 5 on the 1-D area rule, exact
    from the wall law's stationary points (turb_integral.k_max_1d) at Re_De re_de: a record computed
    from the parameters alone, never from the BREP. A re_de that is not a finite positive number is
    refused MEAS-RE; a turb_integral refusal (its TI-* ids) is an error record MEAS-ERROR."""
    method = "turb_integral.k_max_1d/1-D area rule"
    if isinstance(re_de, bool) or not isinstance(re_de, (int, float)) \
            or not math.isfinite(float(re_de)) or float(re_de) <= 0.0:
        return _refused("k_max_1d", "1", method, "MEAS-RE",
                        "re_de %r is not a finite positive number" % (re_de,), feature, where)
    try:
        value = turb_integral.k_max_1d(params["law"], params["CR"], params["L_over_Di"],
                                       float(re_de), params.get("x_m"))["K_max"]
    except ValueError as e:
        return _error("k_max_1d", "1", method, "MEAS-ERROR", str(e)[:300], feature, where)
    return _ok("k_max_1d", value, "1", method, feature=feature, where=where)


PRIMITIVES = {"cylinder_radius": cylinder_radius, "cone_semi_angle": cone_semi_angle, "volume": volume,
              "diameter_at_plane": diameter_at_plane, "area_ratio": area_ratio,
              "extent_along_axis": extent_along_axis, "plane_distance": plane_distance,
              "meridian_min_wall": meridian_min_wall, "slope_max": slope_max,
              "curvature_radius_min": curvature_radius_min, "n_solids": n_solids, "valid": valid,
              "watertight": watertight, "axis_x": axis_x, "units_m": units_m,
              "section_at_plane": section_at_plane, "wall_min_tagged": wall_min_tagged,
              "k_max_1d": k_max_1d}

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


def _fx_tag(shape, name, idx):
    """A geom.json-style face-tag row made the way export.tag_table makes it (index list, summed adaptive area)."""
    fs = shape.Faces()
    return {name: {"index": list(idx), "area_m2": sum(gprop(fs[k], "area")[0] for k in idx)}}


def _fx_tilted_tube():
    """A tube of radii 10 and 12.5 mm, 60 mm long, tilted 30 deg about y and 20 deg about z: wall 2.5 mm."""
    tube = cq.Workplane("XY").circle(0.0125).circle(0.010).extrude(0.06).val()
    tube = tube.rotate(V(0, 0, 0), V(0, 1, 0), 30).rotate(V(0, 0, 0), V(0, 0, 1), 20)
    rad = [(k, BRepAdaptor_Surface(f.wrapped).Cylinder().Radius()) for k, f in enumerate(tube.Faces())
           if f.geomType() == "CYLINDER"]
    rows = dict(_fx_tag(tube, "inner", [k for k, r in rad if abs(r - 0.010) < 1e-12]),
                **_fx_tag(tube, "outer", [k for k, r in rad if abs(r - 0.0125) < 1e-12]))
    return tube, rows


def _fx_ramp(z_top=None):
    """Two slanted faces z = x tan 50 deg and 3 mm above it (x 0..20 mm, y 0..40 mm): wall 3 cos 50 deg mm.
    With z_top = (z0, z1) the top face runs from z0 at x = 0 to z1 at x = 20 mm above a flat bottom instead."""
    if z_top is None:
        t = math.tan(math.radians(50))
        pts = [(0, 0), (0.02, 0.02 * t), (0.02, 0.02 * t + 0.003), (0, 0.003)]
    else:
        pts = [(0, 0), (0.02, 0), (0.02, z_top[1]), (0, z_top[0])]
    body = cq.Workplane("XZ").polyline(pts).close().extrude(-0.04).val()
    fs = body.Faces()
    sl = [k for k, f in enumerate(fs) if abs(f.normalAt().z) > 1e-9]
    sl.sort(key=lambda k: fs[k].Center().z)
    return body, dict(_fx_tag(body, "lower", sl[:1]), **_fx_tag(body, "upper", sl[1:]))


def selftest():
    """GC-1: every primitive against an analytic answer; 44 [ok] lines, then SELFTEST PASS."""
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

    # (M24)
    def geo(**over):
        d = {"units": "m", "scale": 1, "step_length_unit": "METRE", "x_span_m": 0.1,
             "gmsh_import": {"x_span_m": 0.1}}
        d.update(over)
        return d

    r = keep(units_m(geo()))
    assert r["value"] == 1, "M24 good %r" % (r,)
    r = keep(units_m(geo(units="mm")))
    assert r["value"] == 0, "M24 mm %r" % (r,)
    r = keep(units_m(geo(scale=1000)))
    assert r["value"] == 0, "M24 scale 1000 %r" % (r,)
    r = keep(units_m(geo(scale=True)))
    assert r["value"] == 0, "M24 scale True %r" % (r,)
    r = keep(units_m(geo(step_length_unit="MILLI.METRE")))
    assert r["value"] == 0, "M24 MILLI %r" % (r,)
    r = keep(units_m(geo(gmsh_import={"x_span_m": 100.0})))
    assert r["value"] == 0, "M24 span x1000 %r" % (r,)
    r = keep(units_m(geo(gmsh_import={"x_span_m": 0.1 + 2e-9})))
    assert r["value"] == 0, "M24 span +2e-9 %r" % (r,)
    r = keep(units_m({"units": "m", "scale": 1, "step_length_unit": "METRE", "x_span_m": 0.1}))
    assert r["status"] == "refused" and r["reason_id"] == "MEAS-BADGEOM", "M24 missing %r" % (r,)
    print("[ok] units_m: good 1; mm, scale 1000, scale True, MILLI STEP, span x1000, span +2e-9 all 0; "
          "missing key refused")

    # (M23)
    r = keep(axis_x(cylinder))
    assert r["value"] == 1, "M23 cylinder %r" % (r,)
    r = keep(axis_x(fluid))
    assert r["value"] == 1, "M23 poly5 fluid %r" % (r,)
    r = keep(axis_x(cylinder.rotate(V(0, 0, 0), V(0, 1, 0), 90)))
    assert r["value"] == 0, "M23 rotated %r" % (r,)
    r = keep(axis_x(_fx_torus()))
    assert r["value"] == 0, "M23 torus %r" % (r,)
    r = keep(axis_x(cq.Solid.makeBox(0.01, 0.02, 0.03)))
    assert r["value"] == 0, "M23 box %r" % (r,)
    print("[ok] axis_x: cylinder 1, poly5 fluid 1, rotated 0, torus 0, box 0")

    # (M22)
    def rep(**over):
        d = {"tool": "stl_repair", "weld": {"tol_rel": 0.0}, "before": {"closed": True},
             "after": {"closed": True, "open_edges": 0, "non_manifold_edges": 0},
             "orientation": {"reoriented_triangles": 0, "flipped_components": 0}, "holes": {"filled": 0},
             "degenerate_dropped": 0, "n_components": 1}
        d.update(over)
        return d

    r = keep(watertight(rep()))
    assert r["value"] == 1, "M22 clean %r" % (r,)
    r = keep(watertight(rep(orientation={"reoriented_triangles": 3, "flipped_components": 0})))
    assert r["value"] == 0, "M22 reoriented %r" % (r,)
    r = keep(watertight(rep(after={"closed": False, "open_edges": 2, "non_manifold_edges": 0})))
    assert r["value"] == 0, "M22 open %r" % (r,)
    r = keep(watertight(rep(weld={"tol_rel": 1e-6})))
    assert r["status"] == "refused" and r["reason_id"] == "MEAS-BADREPORT", "M22 weld %r" % (r,)
    r = keep(watertight({}))
    assert r["status"] == "refused" and r["reason_id"] == "MEAS-BADREPORT", "M22 empty %r" % (r,)
    print("[ok] watertight: clean 1, reoriented 0, open 0, weld 1e-6 refused, non-report refused")

    import warnings
    from scipy.integrate import quad
    sys.path.insert(0, os.path.join(HERE, "fixtures", "trap"))
    import trap
    body = trap.build()
    assert body.isValid() and len(body.Solids()) == 1, "M25 trap build"

    # (M25)
    pd = GProp_GProps()
    BRepGProp.VolumeProperties_s(body.wrapped, pd)
    rel_d = (pd.Mass() - trap.VOLUME_M3) / trap.VOLUME_M3
    assert abs(rel_d) > 1e-5, "M25 the default call no longer misses: fixture does not discriminate"
    rec = keep(volume(body))
    assert rec["status"] == "ok", "M25 volume refused or errored: %r" % (rec,)
    rel = (rec["value"] - trap.VOLUME_M3) / trap.VOLUME_M3
    assert abs(rel) <= 1e-8, "M25 adaptive volume rel %+.3e > 1e-8" % (rel,)
    assert rec["detail"].startswith("eps 1e-12; relative error estimate "), "M25 detail %r" % (rec["detail"],)
    est = float(rec["detail"].rsplit(" ", 1)[1])
    assert est <= 1e-8, "M25 estimate %.3e > 1e-8" % (est,)
    print("[ok] trap volume: default call rel %+.3e (misses > 1e-5), adaptive rel %+.3e, estimate %.3e"
          % (rel_d, rel, est))

    # (M26)
    mod = sys.modules[__name__]
    saved = mod.GPROP_EPS
    try:
        mod.GPROP_EPS = 1e-6
        rec = keep(volume(body))
    finally:
        mod.GPROP_EPS = saved
    assert (rec["status"] == "refused" and rec["reason_id"] == "MEAS-GPROP" and rec["value"] is None
            and rec["u_meas"] is None and "exceeds u_meas" in rec["detail"]), "M26 %r" % (rec,)
    rec2 = keep(volume(body))
    assert rec2["status"] == "ok", "M26 restored call %r" % (rec2,)
    print("[ok] MEAS-GPROP: at eps 1e-6 the trap volume estimate exceeds u_meas and is refused, value None")

    # (M27)
    face = trap.wetted_face(body)
    ad = BRepAdaptor_Curve(trap.spline(0.0).wrapped)
    us = np.linspace(ad.FirstParameter(), ad.LastParameter(), 20001)
    res = max(abs(ad.Value(u).Y() - trap.r(ad.Value(u).X())) for u in us)
    assert res <= 1e-8, "M27 spline-vs-law residual %g > 1e-8" % (res,)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        law = 2 * math.pi * quad(lambda x: trap.r(x) * math.sqrt(1 + trap.dr(x) ** 2), 0, trap.L,
                                 epsabs=0, epsrel=1e-13, limit=200)[0]
    a, est_a = gprop(face, "area")
    assert abs(a - law) / law <= 1e-7, "M27 adaptive area rel %+.3e > 1e-7" % ((a - law) / law,)
    assert est_a <= 1e-8, "M27 area estimate %g > 1e-8" % (est_a,)
    pf = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face.wrapped, pf)
    assert abs(pf.Mass() - law) / law > 1e-5, "M27 default does not miss: rel %g" % (abs(pf.Mass() - law) / law,)
    print("[ok] trap wetted spline face: spline-vs-law residual %.3e m, adaptive area rel %+.3e (estimate %.3e), "
          "default rel %+.3e" % (res, (a - law) / law, est_a, (pf.Mass() - law) / law))

    # (M28)
    e = trap.spline(0.0)
    ad = BRepAdaptor_Curve(e.wrapped)
    bs = ad.BSpline()
    kn = [bs.Knot(i) for i in range(1, bs.NbKnots() + 1)]

    def sp(u):
        p = gp_Pnt()
        v = gp_Vec()
        ad.D1(u, p, v)
        return v.Magnitude()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        q = sum(quad(sp, a0, a1, epsabs=0, epsrel=1e-13)[0] for a0, a1 in zip(kn[:-1], kn[1:]))
    ln, est_l = gprop(e, "length")
    assert abs(ln - q) / q <= 1e-12, "M28 spline length rel %+.3e > 1e-12" % ((ln - q) / q,)
    assert est_l <= 1e-8, "M28 length estimate %g > 1e-8" % (est_l,)
    la = gprop(_fx_arc(), "length")[0]
    ta = 0.05 * math.radians(70)
    assert abs(la - ta) / ta <= 1e-12, "M28 arc length rel %+.3e > 1e-12" % ((la - ta) / ta,)
    assert gprop(body, "area")[0] > 0, "M28 body area not positive"
    raised = False
    try:
        gprop(body, "mass")
    except ValueError:
        raised = True
    assert raised, "M28 unknown kind did not raise ValueError"
    print("[ok] gprop length: trap spline rel %+.3e vs per-span quad, arc rel %+.3e, unknown kind raises"
          % ((ln - q) / q, (la - ta) / ta))

    # (M29)
    from scipy.special import ellipe
    from scipy.optimize import brentq
    Ro, Ri = 0.02, 0.012
    tube = cq.Workplane("YZ").circle(Ro).circle(Ri).extrude(0.1).val()
    pm = {"name": "mid", "origin": [0.05, 0, 0], "normal": [1, 0, 0], "u": [0, 1, 0]}
    sp = section_props(tube, pm)
    rec = keep(section_at_plane(tube, pm))
    assert rec["status"] == "ok", "M29 %r" % (rec,)
    assert rec["value"] == sp["area_m2"] and rec["unit"] == "m2" and rec["where"] == ["mid"], "M29 %r" % (rec,)
    assert sp["n_islands"] == 1 and sp["n_holes"] == 1, "M29 islands %r" % (sp,)
    tr_a, tr_h, tr_o, tr_d = (math.pi * (Ro ** 2 - Ri ** 2), 2 * math.pi * Ri, 2 * math.pi * Ro, 2 * (Ro - Ri))
    rel = (sp["area_m2"] - tr_a) / tr_a
    relh = (sp["hole_perimeter_m"] - tr_h) / tr_h
    relo = (sp["outer_perimeter_m"] - tr_o) / tr_o
    reld = (sp["dh_m"] - tr_d) / tr_d
    assert abs(rel) <= 1e-12 and abs(relh) <= 1e-12 and abs(relo) <= 1e-12 and abs(reld) <= 1e-12, (
        "M29 tube rels %+.3e %+.3e %+.3e %+.3e > 1e-12" % (rel, relh, relo, reld))
    env_err = max(abs(a - b) for a, b in zip(sp["envelope_uv_m"], [-Ro, Ro, -Ro, Ro]))
    assert env_err <= 1e-12, "M29 envelope err %.3e > 1e-12 m" % (env_err,)
    pf = {"name": "mid_far", "origin": [0.05, 7.0, -3.0], "normal": [1, 0, 0], "u": [0, 1, 0]}
    spf = section_props(tube, pf)
    recf = keep(section_at_plane(tube, pf))
    assert recf["status"] == "ok", "M29 far %r" % (recf,)
    relf = (spf["area_m2"] - tr_a) / tr_a
    assert abs(relf) <= 1e-12, "M29 far area rel %+.3e > 1e-12" % (relf,)
    envf = max(abs(a - b) for a, b in zip(spf["envelope_uv_m"], [-Ro - 7.0, Ro - 7.0, -Ro + 3.0, Ro + 3.0]))
    assert envf <= 1e-11, "M29 far envelope err %.3e > 1e-11 m" % (envf,)
    print("[ok] section tube: annulus rel %+.2e, hole perimeter rel %+.2e, outer rel %+.2e, 4A/P rel %+.2e, "
          "envelope err %.1e m; far in-plane origin area rel %+.2e envelope err %.1e m"
          % (rel, relh, relo, reld, env_err, relf, envf))

    # (M30)
    r = 0.01
    cyl = cq.Workplane("YZ").circle(r).extrude(0.1).val()
    c30, s30 = math.cos(math.radians(30)), math.sin(math.radians(30))
    pe = {"name": "oblique", "origin": [0.05, 0, 0], "normal": [c30, s30, 0], "u": [-s30, c30, 0]}
    a = r / c30
    m = 1 - (r / a) ** 2
    P = 4 * a * ellipe(m)
    sp2 = section_props(cyl, pe)
    rec2 = keep(section_at_plane(cyl, pe))
    assert rec2["status"] == "ok", "M30 %r" % (rec2,)
    assert sp2["n_islands"] == 1 and sp2["n_holes"] == 0, "M30 islands %r" % (sp2,)
    tr_e = math.pi * r * r / c30
    rel_e = (sp2["area_m2"] - tr_e) / tr_e
    assert abs(rel_e) <= 1e-12, "M30 area rel %+.3e > 1e-12" % (rel_e,)
    rel_p = (sp2["outer_perimeter_m"] - P) / P
    assert abs(rel_p) <= 1e-9, "M30 perimeter rel %+.3e > 1e-9 vs 4a E(m)" % (rel_p,)
    assert sp2["hole_perimeter_m"] == 0.0, "M30 hole perimeter %r" % (sp2["hole_perimeter_m"],)
    env_e = max(abs(x - y) for x, y in zip(sp2["envelope_uv_m"], [-a, a, -r, r]))
    assert env_e <= 1e-12, "M30 envelope err %.3e > 1e-12 m" % (env_e,)
    print("[ok] section 30 deg ellipse: area rel %+.2e, perimeter rel %+.2e vs 4a E(m), envelope err %.1e m"
          % (rel_e, rel_p, env_e))

    # (M31)
    x = 0.014269
    cc = BRepAdaptor_Curve(trap.spline(0.0).wrapped)
    t = brentq(lambda s: cc.Value(s).X() - x, cc.FirstParameter(), cc.LastParameter(), xtol=1e-16, rtol=1e-15)
    rs = cc.Value(t).Y()
    truth = math.pi * ((rs + trap.W) ** 2 - rs ** 2)
    ps = {"name": "station", "origin": [x, 0, 0], "normal": [1, 0, 0], "u": [0, 1, 0]}
    sp3 = section_props(body, ps)
    rec3 = keep(section_at_plane(body, ps))
    assert rec3["status"] == "ok", "M31 %r" % (rec3,)
    assert sp3["n_islands"] == 1 and sp3["n_holes"] == 1, "M31 islands %r" % (sp3,)
    rel_t = (sp3["area_m2"] - truth) / truth
    assert abs(rel_t) <= 1e-9, "M31 area rel %+.3e > 1e-9" % (rel_t,)
    rel_hp = (sp3["hole_perimeter_m"] - 2 * math.pi * rs) / (2 * math.pi * rs)
    assert abs(rel_hp) <= 1e-9, "M31 hole perimeter rel %+.3e > 1e-9" % (rel_hp,)
    raw = _section_islands(body.Solids()[0].wrapped, np.array([x, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
                           np.array([0.0, 1.0, 0.0]), None)
    rel_raw = sum(i["area_m2"] for i in raw) / truth - 1
    assert abs(rel_raw) > 1e-7, "M31 the fixture no longer discriminates the scale step: rel %+.3e" % (rel_raw,)
    po = {"name": "oblique_trap", "origin": [0.02, 0, 0], "normal": [c30, 0, s30], "u": [0, 1, 0]}
    spo = section_props(body, po)
    assert spo["status"] == "ok", "M31 oblique %r" % (spo,)
    print("[ok] section trap x 14.269 mm: area rel %+.2e, hole perimeter rel %+.2e vs the spline radius; "
          "unscaled cut rel %+.2e (misses > 1e-7); oblique 30 deg %.9f mm2"
          % (rel_t, rel_hp, rel_raw, spo["area_m2"] * 1e6))

    # (M32)
    pmer = {"name": "meridian", "origin": [0, 0, 0], "normal": [0, 0, 1], "u": [1, 0, 0]}
    truth_m = 2 * trap.W * (trap.L + trap.LX)
    sp4 = section_props(body, pmer)
    rec4 = keep(section_at_plane(body, pmer))
    assert rec4["status"] == "ok", "M32 %r" % (rec4,)
    assert sp4["n_islands"] == 2 and sp4["n_holes"] == 0, "M32 islands %r" % (sp4,)
    rel_m = (sp4["area_m2"] - truth_m) / truth_m
    assert abs(rel_m) <= 1e-12, "M32 area rel %+.3e > 1e-12" % (rel_m,)
    print("[ok] section trap meridian: 2 islands 0 holes, area rel %+.2e vs 2 W (L + LX)" % (rel_m,))

    # (M33)
    pmiss = {"name": "upstream", "origin": [-1.0, 0, 0], "normal": [1, 0, 0], "u": [0, 1, 0]}
    rm = keep(section_at_plane(body, pmiss))
    assert (rm["status"] == "refused" and rm["reason_id"] == "MEAS-NOSECTION" and rm["value"] is None
            and rm["u_meas"] is None and rm["where"] == ["upstream"]), "M33 %r" % (rm,)
    rr = keep(run("section_at_plane", body, pmiss))
    assert (rr["status"] == "refused" and rr["reason_id"] == "MEAS-NOSECTION" and rr["value"] is None
            and rr["u_meas"] is None and rr["where"] == ["upstream"]), "M33 %r" % (rr,)
    spm = section_props(body, pmiss)
    assert (spm["status"] == "refused" and spm["reason_id"] == "MEAS-NOSECTION" and spm["area_m2"] is None
            and spm["n_islands"] == 0), "M33 %r" % (spm,)
    mod = sys.modules[__name__]
    saved = mod.GPROP_EPS
    try:
        mod.GPROP_EPS = 1e-2
        rg = keep(section_at_plane(body, ps))
    finally:
        mod.GPROP_EPS = saved
    assert (rg["status"] == "refused" and rg["reason_id"] == "MEAS-GPROP" and "exceeds u_meas" in rg["detail"]
            and rg["value"] is None), "M33 %r" % (rg,)
    rok = keep(section_at_plane(body, ps))
    assert rok["status"] == "ok", "M33 restored %r" % (rok,)
    print("[ok] section x = -1 m refused MEAS-NOSECTION (direct and run), value None; eps 1e-2 refused MEAS-GPROP")

    # (M34)
    def bad_plane(p):
        rb = keep(section_at_plane(tube, p))
        assert (rb["status"] == "refused" and rb["reason_id"] == "MEAS-BADPLANE"
                and rb["value"] is None), "M34 %r" % (rb,)

    bad_plane(dict(pm, name="1bad"))
    bad_plane(dict(pm, normal=[0, 0, 0]))
    bad_plane(dict(pm, normal=[1, 0, 0], u=[1, 0, 0]))
    bad_plane(dict(pm, normal=[1, 0, 0], u=[1e-9, 1, 0]))
    bad_plane(dict(pm, origin=[True, 0, 0]))
    bad_plane(dict(pm, origin=[float("nan"), 0, 0]))
    bad_plane(dict(pm, origin=[0, 0]))
    nb = dict(pm)
    del nb["u"]
    bad_plane(nb)
    bad_plane("x")
    rface = keep(section_at_plane(cq.Face.makePlane(0.1, 0.1), pm))
    assert rface["status"] == "refused" and rface["reason_id"] == "MEAS-NOSOLID", "M34 %r" % (rface,)
    rcomp = keep(section_at_plane(cq.Compound.makeCompound([tube, tube.translate(cq.Vector(0.2, 0, 0))]), pm))
    assert rcomp["status"] == "refused" and rcomp["reason_id"] == "MEAS-MULTISOLID", "M34 %r" % (rcomp,)
    rint = keep(section_at_plane(42, pm))
    assert rint["status"] == "error" and rint["reason_id"] == "MEAS-ERROR", "M34 %r" % (rint,)
    print("[ok] section refusals: 9 bad planes MEAS-BADPLANE, a face MEAS-NOSOLID, two solids MEAS-MULTISOLID, "
          "a non-shape MEAS-ERROR")
    # (M35)
    ttube, trows = _fx_tilted_tube()
    tin, tout = tag_faces(ttube, trows, "inner"), tag_faces(ttube, trows, "outer")
    rt = keep(wall_min_tagged(ttube, tin, tout))
    rt2 = keep(run("wall_min_tagged", ttube, tin, tout))
    assert rt["status"] == "ok" and rt["where"] == ["inner", "outer"] and rt == rt2, "M35 %r" % (rt,)
    err_t = abs(rt["value"] - 0.0025)
    assert err_t <= 1e-12, "M35 tilted tube err %.3e > 1e-12 m" % (err_t,)
    ft = wall_ray_field(ttube, tin, tout)
    err_ab, err_ba = abs(ft["min_ab_m"] - 0.0025), abs(ft["min_ba_m"] - 0.0025)
    assert max(err_ab, err_ba) <= 1e-12, "M35 ray field from both sides %.3e %.3e > 1e-12 m" % (err_ab, err_ba)
    print("[ok] wall_min_tagged tilted tube: extrema err %.1e m vs t = 2.5 mm, ray field from both sides err "
          "%.1e / %.1e m (direct and run)" % (err_t, err_ab, err_ba))

    # (M36)
    ramp, rrows = _fx_ramp()
    rr = keep(wall_min_tagged(ramp, tag_faces(ramp, rrows, "lower"), tag_faces(ramp, rrows, "upper")))
    truth_r = 0.003 * math.cos(math.radians(50))
    assert abs(truth_r - 1.928362829e-3) <= 1e-12, "M36 3 cos 50 deg %.15g" % (truth_r,)
    assert rr["status"] == "ok", "M36 %r" % (rr,)
    err_r = abs(rr["value"] - truth_r)
    assert err_r <= 1e-12, "M36 ramp err %.3e > 1e-12 m" % (err_r,)
    print("[ok] wall_min_tagged ramp: %.9f mm, err %.1e m vs 3 cos 50 deg = 1.928362829 mm"
          % (rr["value"] * 1e3, err_r))

    # (M37)
    tb = trap.build()
    tfs = tb.Faces()
    lim = {"REVOLUTION": trap.RI + trap.W / 2, "CYLINDER": trap.RE + trap.W / 2}
    wet_k = [k for k, f in enumerate(tfs) if f.geomType() in lim and f.BoundingBox().ymax < lim[f.geomType()]]
    out_k = [k for k, f in enumerate(tfs) if f.geomType() in lim and k not in wet_k]
    assert len(wet_k) == 2 and len(out_k) == 2, "M37 trap faces %r %r" % (wet_k, out_k)
    grows = dict(_fx_tag(tb, "wetted", wet_k), **_fx_tag(tb, "outer", out_k))
    twet, tout2 = tag_faces(tb, grows, "wetted"), tag_faces(tb, grows, "outer")
    rtr = keep(wall_min_tagged(tb, twet, tout2))
    assert rtr["status"] == "ok" and rtr["where"] == ["wetted", "outer"], "M37 %r" % (rtr,)
    err_tr = abs(rtr["value"] - 1.8758968327e-3)
    assert err_tr <= 1e-8, "M37 trap err %.3e > 1e-8 m vs the dense 2-D truth" % (err_tr,)
    print("[ok] wall_min_tagged trap complete tagged sets: %.12f mm, err %.1e m vs the dense 2-D truth 1.8758968327 mm"
          % (rtr["value"] * 1e3, err_tr))

    # (M38)
    cyl_w = [tfs[k] for k in wet_k if tfs[k].geomType() == "CYLINDER"]
    cyl_o = [tfs[k] for k in out_k if tfs[k].geomType() == "CYLINDER"]
    d_line = BRepExtrema_DistShapeShape(cq.Compound.makeCompound(cyl_w).wrapped,
                                        cq.Compound.makeCompound(cyl_o).wrapped)
    d_line.Perform()
    assert abs(d_line.Value() - 0.003) <= 1e-12, "M38 exit-line extrema %.15g" % (d_line.Value(),)
    bad = [dict(twet, faces=cyl_w), dict(twet, faces=twet["faces"] + cyl_w), dict(twet, faces=[tin["faces"][0]])]
    for tg in bad:
        rc = keep(wall_min_tagged(tb, tg, dict(tout2, faces=cyl_o)))
        assert rc["status"] == "refused" and rc["reason_id"] == "MEAS-COVER" and rc["value"] is None, "M38 %r" % (rc,)
    print("[ok] trap exit-line faces only refused MEAS-COVER (their extrema reads the false %.12f mm); a face twice "
          "and a foreign face refused MEAS-COVER" % (d_line.Value() * 1e3,))

    # (M39)
    b1, b2 = cq.Solid.makeBox(0.01, 0.01, 0.01), cq.Solid.makeBox(0.01, 0.01, 0.01, pnt=V(0.01, 0, 0))
    bb2 = cq.Compound.makeCompound([b1, b2])
    ball = dict(_fx_tag(bb2, "left", range(6)), **_fx_tag(bb2, "right", range(6, 12)))
    rb = keep(wall_min_tagged(bb2, tag_faces(bb2, ball, "left"), tag_faces(bb2, ball, "right")))
    top = [k for k, f in enumerate(b1.Faces()) if abs(f.Center().z - 0.01) < 1e-12]
    side = [k for k, f in enumerate(b1.Faces()) if abs(f.Center().x) < 1e-12]
    b1rows = dict(_fx_tag(b1, "top", top), **_fx_tag(b1, "side", side))
    re_ = keep(wall_min_tagged(b1, tag_faces(b1, b1rows, "top"), tag_faces(b1, b1rows, "side")))
    for rx in (rb, re_):
        assert rx["status"] == "refused" and rx["reason_id"] == "MEAS-TOUCH" and rx["value"] is None, "M39 %r" % (rx,)
    print("[ok] two touching boxes refused MEAS-TOUCH; two faces sharing an edge refused MEAS-TOUCH")

    # (M40)
    fr = wall_ray_field(tb, twet, tout2)
    a40 = fr["account"]
    ext = rtr["value"]
    assert fr["status"] == "ok" and ext <= fr["min_m"] <= ext * (1 + 5e-4), "M40 ray min %r vs %r" % (fr["min_m"], ext)
    assert (a40["candidate"] == 4 * 81 * 81 == a40["banded"] + a40["outside"] + a40["selected"]
            and a40["selected"] == a40["valid"] + a40["invalid"]
            and a40["invalid"] == a40["no_normal"] + a40["no_hit"] + a40["other_face"]), "M40 account %r" % (a40,)
    assert fr["argmin"]["face_index"] in wet_k + out_k and fr["p05_m"] >= fr["min_m"], "M40 %r" % (fr["argmin"],)
    print("[ok] ray field trap 81 x 81: min %.6f mm in [extrema, extrema (1 + 5e-4)] (rel %+.2e), p05 %.6f mm, "
          "%d valid of %d candidates, sampled area ratio %.4f" % (fr["min_m"] * 1e3, fr["min_m"] / ext - 1,
                                                                  fr["p05_m"] * 1e3, a40["valid"], a40["candidate"],
                                                                  a40["sampled_area_ratio"]))

    # (M41)
    f3, f2 = wall_ray_field(ttube, tin, tout, target_m=0.003), wall_ray_field(ttube, tin, tout, target_m=0.002)
    a41 = f3["account"]
    ratio = (77 / 81) ** 2
    assert (a41["candidate"] == 2 * 81 * 81 and a41["selected"] == a41["valid"] == 2 * 77 * 77
            and a41["banded"] == 2 * (81 * 81 - 77 * 77) and a41["outside"] == 0), "M41 account %r" % (a41,)
    assert abs(a41["sampled_area_ratio"] / ratio - 1) <= 1e-12, "M41 sampled area ratio %r" % (a41,)
    assert abs(f3["p05_m"] - 0.0025) <= 1e-12, "M41 p05 %r" % (f3["p05_m"],)
    assert f3["violating_area_ratio"] == 1.0 and f2["violating_area_ratio"] == 0.0, "M41 violating %r %r" % (
        f3["violating_area_ratio"], f2["violating_area_ratio"])
    assert ft["violating_area_ratio"] is None and set(f3["argmin"]) == {"from", "face_index", "uv", "point_m",
                                                                        "hit_m"}, "M41 %r" % (f3["argmin"],)
    print("[ok] ray field report on the tube: p05 2.5 mm, violating area 1 at 3 mm and 0 at 2 mm, sampled area "
          "ratio (77/81)^2, account %d = %d banded + %d valid" % (a41["candidate"], a41["banded"], a41["valid"]))

    # (M42)
    taper, prow = _fx_ramp(z_top=(0.003, 0.001))
    tl, tu = tag_faces(taper, prow, "lower"), tag_faces(taper, prow, "upper")
    rx1 = keep(wall_min_tagged(taper, tl, tu))
    fx1 = wall_ray_field(taper, tl, tu)
    pb = cq.Solid.makeBox(0.04, 0.03, 0.02).cut(cq.Solid.makeBox(0.0345, 0.025, 0.02, pnt=V(0.0015, 0.002, 0.0025)))
    pk = [k for k, f in enumerate(pb.Faces()) if f.geomType() == "PLANE" and abs(abs(f.normalAt().x) - 1) < 1e-12]
    px0 = [k for k in pk if abs(pb.Faces()[k].Center().x) < 1e-12]
    px1 = [k for k in pk if abs(pb.Faces()[k].Center().x - 0.036) < 1e-12]
    prows = dict(_fx_tag(pb, "outside", px0), **_fx_tag(pb, "pocket", px1))
    rx2 = keep(wall_min_tagged(pb, tag_faces(pb, prows, "outside"), tag_faces(pb, prows, "pocket")))
    fx2 = wall_ray_field(pb, tag_faces(pb, prows, "outside"), tag_faces(pb, prows, "pocket"))
    for rx in (rx1, rx2):
        assert rx["status"] == "refused" and rx["reason_id"] == "MEAS-XCHECK" and rx["value"] is None, "M42 %r" % (rx,)
    assert fx1["min_m"] > 0.001 * (1 + 5e-4) and "BRepExtrema 0.001 m" in rx1["detail"], "M42 taper %r" % (fx1,)
    assert (fx2["min_m"] is None and fx2["account"]["valid"] == 0
            and fx2["account"]["other_face"] == fx2["account"]["selected"] > 0), "M42 pocket %r" % (fx2,)
    print("[ok] MEAS-XCHECK: a taper whose 1 mm minimum sits in the band (ray min %.6f mm), and a pocket pair no ray "
          "joins (%d rays hit another face)" % (fx1["min_m"] * 1e3, fx2["account"]["other_face"]))

    # (M43)
    two = cq.Compound.makeCompound([ttube, ttube.translate(V(0.1, 0, 0))])
    un = [keep(wall_min_tagged(ttube, tin["faces"], tout["faces"])), keep(wall_min_tagged(ttube, {"name": "inner",
          "faces": tin["faces"]}, tout)), keep(wall_min_tagged(ttube, dict(tin, area_m2=True), tout)),
          keep(run("wall_distance_3d", tin["faces"], tout["faces"]))]
    assert all(r["status"] == "refused" and r["reason_id"] == "MEAS-3D-WALL" and r["value"] is None for r in un), (
        "M43 untagged %r" % (un,))
    rem = keep(wall_min_tagged(ttube, dict(tin, faces=[]), tout))
    rms = keep(wall_min_tagged(two, tin, tout))
    assert rem["status"] == "refused" and rem["reason_id"] == "MEAS-EMPTY", "M43 empty %r" % (rem,)
    assert rms["status"] == "refused" and rms["reason_id"] == "MEAS-MULTISOLID", "M43 two solids %r" % (rms,)
    print("[ok] untagged sets refused MEAS-3D-WALL four ways (face lists, no area, a bool area, wall_distance_3d); "
          "an empty tag MEAS-EMPTY; two solids MEAS-MULTISOLID")

    # (M44)
    tg4_params = {"D_i": 0.30, "CR": 2.0, "L_over_Di": 0.5, "law": "poly7", "x_m": None,
                  "Lx_over_De": 0.5, "Lu_over_Di": 2.0, "upstream_role": "wall", "t_wall": 0.004}
    kp = k_max_1d(tg4_params, 630000.0)
    kp3 = k_max_1d(dict(tg4_params, law="poly3", L_over_Di=0.6), 630000.0)
    kr = k_max_1d(tg4_params, None)
    ke = k_max_1d({"law": "nope", "CR": 2.0, "L_over_Di": 0.5, "x_m": None}, 630000.0)
    assert abs(kp["value"] / 4.966618287407307e-06 - 1.0) <= 1e-12, kp["value"]
    assert abs(kp3["value"] / 2.8505770491696314e-06 - 1.0) <= 1e-12, kp3["value"]
    assert kp["u_meas"] == 1e-9 * kp["value"] and kp["unit"] == "1" and kp["status"] == "ok", kp
    assert kp["method"] == "turb_integral.k_max_1d/1-D area rule" and kp["where"] == ["wall_contraction"], kp
    assert kr["status"] == "refused" and kr["reason_id"] == "MEAS-RE" and kr["value"] is None, kr
    assert ke["status"] == "error" and ke["reason_id"] == "MEAS-ERROR" and ke["value"] is None, ke
    print("[ok] k_max_1d: poly7 L/D 0.5 gives %r and poly3 L/D 0.6 %r at Re_De 630000 (1e-12 rel),"
          " re_de None refused MEAS-RE, a bad law errors MEAS-ERROR, all records schema-valid"
          % (kp["value"], kp3["value"]))

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
