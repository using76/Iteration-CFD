#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""readiness.py - stage S5 of the CAD loop (docs/16 §D, §I): ordered RDY-* CFD-readiness refusals on one S3 export directory, before any mesh.

Usage:
  python readiness.py --selftest
  python readiness.py check OUT_DIR H_M
"""
import math
import os
import shutil
import subprocess
import sys
import tempfile

import cadquery as cq
from OCP.BRep import BRep_Tool
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.ShapeAnalysis import ShapeAnalysis_FreeBounds, ShapeAnalysis_Wire
from OCP.TopAbs import TopAbs_EDGE, TopAbs_WIRE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS
from OCP.gp import gp_Dir, gp_Pln, gp_Pnt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common
import export
import measure

# readiness judges the FILES an S3 export wrote (fluid.brep, meridian.brep, wall_meridian.brep, tags.json,
# geom.json, stl_repair.json), never the template's parameters: the template's PRF rules refuse bad inputs, and
# readiness refuses bad OUTPUTS, including ones no PRF rule can see (docs/16 §C fact 1: an axis-crossing and a
# bow-tie profile both revolve into "valid" solids). h is the caller's mesh size in metres; the rules stop at the
# first refusal, in RULE_ORDER, and a rule whose input file is missing refuses, never passes.
VERSION = 1
RULE_ORDER = ("RDY-PROFILE", "RDY-BREP", "RDY-FACEW", "RDY-EDGE", "RDY-STL", "RDY-THROAT", "RDY-TAGS")
READY_KEYS = ("version", "status", "rule", "check", "detail", "h_m", "template_id", "params_sha", "results")
RESULT_KEYS = ("rule", "ok", "check", "detail", "values")
AXIS_TOL = measure.AXIS_TOL    # m: a meridian vertex with |y| <= this is on the axis; |z| <= this is in-plane
SELFX_PREC = 1e-7              # m: ShapeAnalysis_Wire precision, as the template's PRF-SELFX uses
N_SAMPLE = 2001                # points per meridian edge (measure._edge_points, uniform in the parameter)
EDGE_FRAC = 0.5                # RDY-EDGE: every edge >= EDGE_FRAC * h
THROAT_MIN = 20.0              # RDY-THROAT: D_throat / h >= THROAT_MIN
H_SELFTEST_M = 5e-4            # m: about the nominal's L1 core cell (meridian 2.2e-3 m2 over 10k cells)
READY_FILE = "readiness.json"
USAGE = ("usage: python readiness.py --selftest" + chr(10)
         + "       python readiness.py check OUT_DIR H_M")


class MissingInput(Exception):
    """A rule input file is absent; str(e) is the file's basename."""


def _load_shape(ctx, name):
    """The cq.Shape of OUT_DIR/name; MissingInput(basename) when absent; cached per check."""
    key = "shape:" + name
    if key not in ctx["cache"]:
        path = os.path.join(ctx["out_dir"], name)
        if not os.path.isfile(path):
            raise MissingInput(name)
        ctx["cache"][key] = cq.Shape.importBrep(path)
    return ctx["cache"][key]


def _load_json(ctx, name):
    """The json dict of OUT_DIR/name; MissingInput(basename) when absent; cached per check."""
    key = "json:" + name
    if key not in ctx["cache"]:
        path = os.path.join(ctx["out_dir"], name)
        if not os.path.isfile(path):
            raise MissingInput(name)
        ctx["cache"][key] = common.read_json(path)
    return ctx["cache"][key]


def _load_decl(ctx):
    """The template declaration (template.json); MissingInput(basename) when absent."""
    if "json:declaration" not in ctx["cache"]:
        path = ctx["declaration_path"]
        if path is None or not os.path.isfile(path):
            raise MissingInput("template.json")
        ctx["cache"]["json:declaration"] = common.read_json(path)
    return ctx["cache"]["json:declaration"]


def _scrub(text, out_dir):
    """Replace every spelling of OUT_DIR by <out>: a report never holds an absolute path."""
    forms = {out_dir, os.path.abspath(out_dir), os.path.abspath(out_dir).replace(os.sep, "/")}
    for bad in sorted((f for f in forms if f), key=len, reverse=True):
        text = text.replace(bad, "<out>")
    return text


def face_width(face):
    """2A/P: twice the area over the summed length of every edge occurrence; degenerate edges skipped; P == 0 gives 0.0."""
    perimeter = 0.0
    explorer = TopExp_Explorer(face.wrapped, TopAbs_EDGE)
    while explorer.More():
        edge = TopoDS.Edge_s(explorer.Current())
        if not BRep_Tool.Degenerated_s(edge):
            perimeter += export.edge_length(cq.Edge(edge))
        explorer.Next()
    if perimeter == 0.0:
        return 0.0
    return 2.0 * export.face_props(face) / perimeter


def free_wire_count(shape):
    """The wires in the shape's free bounds: closed plus open, as ShapeAnalysis_FreeBounds finds them."""
    fb = ShapeAnalysis_FreeBounds(shape.wrapped)
    n = 0
    for bounds in (fb.GetClosedWires(), fb.GetOpenWires()):
        explorer = TopExp_Explorer(bounds, TopAbs_WIRE)
        while explorer.More():
            n += 1
            explorer.Next()
    return n


def profile_values(face):
    """The RDY-PROFILE numbers of one meridian face: sampling, wall edges, self-intersection, 2-D validity."""
    z_max = 0.0
    r_min_all = None
    r_min_wall = None
    n_wall = 0
    for edge in face.Edges():
        pts = measure._edge_points(edge, N_SAMPLE)
        for r in pts:
            z = float(r[2])
            if z < 0.0:
                z = -z
            if z > z_max:
                z_max = z
            y = float(r[1])
            if r_min_all is None or y < r_min_all:
                r_min_all = y
        if abs(float(pts[0][1])) > AXIS_TOL and abs(float(pts[-1][1])) > AXIS_TOL:
            n_wall += 1
            for r in pts:
                y = float(r[1])
                if r_min_wall is None or y < r_min_wall:
                    r_min_wall = y
    plane = BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(0.0, 0.0, 1.0))).Face()
    selfx = 0
    for w in face.Wires():
        if ShapeAnalysis_Wire(w.wrapped, plane, SELFX_PREC).CheckSelfIntersection():
            selfx = 1
            break
    face2d = 1 if (BRepCheck_Analyzer(face.wrapped).IsValid() and face.Area() > 0.0) else 0
    return {"n_edges": len(face.Edges()), "z_max_m": z_max, "r_min_all_m": r_min_all,
            "r_min_wall_m": r_min_wall, "n_wall_edges": n_wall, "selfx": selfx, "face2d": face2d}


def rule_profile(ctx):
    """RDY-PROFILE: the meridian is one planar face in z = 0, never below the axis, its wall edges strictly
    off it, no self-intersection, BRepCheck valid (docs/16 §C facts 1 and 4)."""
    shape = _load_shape(ctx, "meridian.brep")
    faces = shape.Faces()
    values = {"n_faces": len(faces)}
    if len(faces) != 1:
        return False, "one_face", ("meridian.brep holds %d faces, not exactly 1" % (len(faces),)), values
    v = profile_values(faces[0])
    values = dict(v)
    if v["z_max_m"] > AXIS_TOL:
        return False, "plane", ("meridian leaves the plane z = 0: max |z| is %.6e m" % (v["z_max_m"],)), values
    if v["r_min_all_m"] < -AXIS_TOL:
        return False, "r_min", ("meridian dips below the axis: r_min is %.6e m" % (v["r_min_all_m"],)), values
    if v["n_wall_edges"] == 0:
        return False, "r_min", "meridian has no wall edge strictly off the axis", values
    if v["r_min_wall_m"] <= AXIS_TOL:
        return False, "r_min", ("a wall edge reaches the axis: r_min_wall is %.6e m" % (v["r_min_wall_m"],)), values
    if v["selfx"] == 1:
        return False, "selfx", "the meridian wire self-intersects (selfx is 1)", values
    if v["face2d"] == 0:
        return False, "face2d", "the 2-D meridian face fails BRepCheck or has zero area (face2d is 0)", values
    ctx["cache"]["r_min_wall_m"] = v["r_min_wall_m"]
    return True, None, "", values


def rule_brep(ctx):
    """RDY-BREP: the fluid is valid by BRepCheck and BOPCheck, exactly one solid, no free wires."""
    fluid = _load_shape(ctx, "fluid.brep")
    rv = measure.valid(fluid)
    if rv["status"] != "ok":
        raise RuntimeError(rv["detail"])
    rs = measure.n_solids(fluid)
    if rs["status"] != "ok":
        raise RuntimeError(rs["detail"])
    values = {"valid": rv["value"], "n_solids": rs["value"], "free_wires": free_wire_count(fluid)}
    if values["valid"] != 1:
        return False, "valid", ("fluid fails BRepCheck or BOPCheck: %s" % (rv["detail"],)), values
    if values["n_solids"] != 1:
        return False, "n_solids", ("fluid holds %d solids, not 1" % (values["n_solids"],)), values
    if values["free_wires"] != 0:
        return False, "free_wires", ("fluid has %d free wires" % (values["free_wires"],)), values
    return True, None, "", values


def rule_facew(ctx):
    """RDY-FACEW: every fluid face has the 2A/P face width of at least the mesh size h."""
    fluid = _load_shape(ctx, "fluid.brep")
    faces = fluid.Faces()
    values = {"n_faces": len(faces), "face_width_min_m": None, "face_index": None}
    if not faces:
        return False, "face_width", "fluid has no faces", values
    widths = [face_width(f) for f in faces]
    idx = 0
    for i, w in enumerate(widths):
        if w < widths[idx]:
            idx = i
    values["face_width_min_m"] = widths[idx]
    values["face_index"] = idx
    if widths[idx] < ctx["h"]:
        return False, "face_width", ("narrowest fluid face %d has width 2A/P %.6e m, below h %.6e m"
                                     % (idx, widths[idx], ctx["h"])), values
    return True, None, "", values


def rule_edge(ctx):
    """RDY-EDGE: every non-degenerate edge of the fluid and then the meridian is at least EDGE_FRAC * h."""
    fluid = _load_shape(ctx, "fluid.brep")
    meridian = _load_shape(ctx, "meridian.brep")
    n = 0
    best = None
    for on_m, shape in ((0, fluid), (1, meridian)):
        for i, e in enumerate(shape.Edges()):
            if BRep_Tool.Degenerated_s(e.wrapped):
                continue
            n += 1
            ln = export.edge_length(e)
            if best is None or ln < best[0]:
                best = (ln, i, on_m)
    values = {"n_edges": n, "edge_min_m": None, "edge_index": None, "on_meridian": None}
    if best is None:
        return False, "edge_length", "fluid and meridian have no non-degenerate edge", values
    values["edge_min_m"], values["edge_index"], values["on_meridian"] = best
    where = "meridian" if best[2] == 1 else "fluid"
    if best[0] < EDGE_FRAC * ctx["h"]:
        return False, "edge_length", ("shortest %s edge %d is %.6e m, below %.6e m"
                                      % (where, best[1], best[0], EDGE_FRAC * ctx["h"])), values
    return True, None, "", values


def rule_stl(ctx):
    """RDY-STL: the stl_repair report is the one geom.json hashed, watertight, agrees with geom.json,
    and within the STL volume tolerance."""
    report = _load_json(ctx, "stl_repair.json")
    geom = _load_json(ctx, "geom.json")
    values = {"watertight": None, "volume_rel": None}
    sha = common.sha256_file(os.path.join(ctx["out_dir"], "stl_repair.json"))
    if sha != geom["files"]["stl_repair.json"]:
        return False, "report_sha", ("stl_repair.json sha %s is not the geom.json hash %s"
                                     % (sha, geom["files"]["stl_repair.json"])), values
    wt = measure.watertight(report)
    values["watertight"] = wt["value"] if wt["status"] == "ok" else None
    if not (wt["status"] == "ok" and wt["value"] == 1):
        return False, "watertight", ("the STL is not watertight: %s" % (wt["detail"],)), values
    if geom["watertight"] != export._wt_dict(report):
        return False, "geom_agrees", "geom.json's watertight block disagrees with stl_repair.json", values
    vr = geom["stl"]["volume_rel"]
    if isinstance(vr, bool) or not isinstance(vr, (int, float)) or not math.isfinite(vr):
        values["volume_rel"] = None
        return False, "volume", ("geom.json stl.volume_rel is not a finite number: %r" % (vr,)), values
    vr = float(vr)
    values["volume_rel"] = vr
    if abs(vr) > export.STL_VOL_TOL:
        return False, "volume", ("STL volume error %.6e rel exceeds the tolerance %.6e"
                                 % (vr, export.STL_VOL_TOL)), values
    return True, None, "", values


def rule_throat(ctx):
    """RDY-THROAT: the smallest wall diameter D_throat = 2 * r_min_wall is at least THROAT_MIN * h."""
    d = 2.0 * float(ctx["cache"]["r_min_wall_m"])
    values = {"d_throat_m": d, "d_over_h": d / ctx["h"]}
    if values["d_over_h"] < THROAT_MIN:
        return False, "throat", ("smallest wall diameter %.6e m is %.6g h, below the minimum %g"
                                 % (d, values["d_over_h"], THROAT_MIN)), values
    return True, None, "", values


def _decl_lists(decl):
    """The declaration's face-kind tag names, edge-kind tag names and plane names."""
    faces = [t["name"] for t in decl["tags"] if t["kind"] == "face"]
    edges = [t["name"] for t in decl["tags"] if t["kind"] == "edge"]
    planes = [p["name"] for p in decl["planes"]]
    return faces, edges, planes


def _ints_ok(idxs):
    """Every index is an int and not a bool."""
    for i in idxs:
        if isinstance(i, bool) or not isinstance(i, int):
            return False
    return True


def rule_tags(ctx):
    """RDY-TAGS: tags.json, geom.json and the declaration agree, and every declared tag is present
    exactly once, partitioning the fluid faces and the meridian edges."""
    tags = _load_json(ctx, "tags.json")
    geom = _load_json(ctx, "geom.json")
    decl = _load_decl(ctx)
    wall_m = _load_shape(ctx, "wall_meridian.brep")
    fluid = _load_shape(ctx, "fluid.brep")
    meridian = _load_shape(ctx, "meridian.brep")
    decl_faces, decl_edges, decl_planes = _decl_lists(decl)
    values = {"n_face_tags": len(tags["face_tags"]), "n_fluid_faces": len(fluid.Faces()),
              "n_meridian_edges": len(meridian.Edges())}
    if not (tags["template_id"] == geom["template_id"] == decl["template_id"]):
        return False, "template_id", ("template_id disagrees: tags %r, geom %r, declaration %r"
                                      % (tags["template_id"], geom["template_id"],
                                         decl["template_id"])), values
    if geom["declaration_sha"] != common.sha256_file(ctx["declaration_path"]):
        return False, "declaration_sha", "geom.json's declaration_sha does not match the declaration file", values
    if sorted(tags["face_tags"]) != sorted(decl_faces):
        return False, "face_tags", ("face tags %s are not the declared %s"
                                    % (sorted(tags["face_tags"]), sorted(decl_faces))), values
    fidx = []
    for name in sorted(tags["face_tags"]):
        fidx.extend(tags["face_tags"][name])
    if not _ints_ok(fidx) or sorted(fidx) != list(range(len(fluid.Faces()))):
        return False, "face_partition", ("the face tag indices %s do not partition the %d fluid faces"
                                         % (sorted(fidx), len(fluid.Faces()))), values
    patches = tags["stl_patches"]
    if len(set(patches)) != len(patches):
        return False, "stl_patches", "tags.json has duplicate stl patch names", values
    if sorted(patches) != sorted(decl_faces):
        return False, "stl_patches", ("stl patches %s are not the declared face names %s"
                                      % (sorted(patches), sorted(decl_faces))), values
    for name in sorted(tags["face_tags"]):
        area = 0.0
        for i in tags["face_tags"][name]:
            area += export.face_props(fluid.Faces()[i])
        if not area > 0.0:
            return False, "face_area", ("face tag %s has total area %.6e m^2, not > 0" % (name, area)), values
    want = sorted(decl_faces + ["axis"])
    if sorted(tags["meridian_edges"]) != want:
        return False, "meridian_partition", ("meridian edge tags %s are not the declared %s"
                                             % (sorted(tags["meridian_edges"]), want)), values
    midx = []
    for name in sorted(tags["meridian_edges"]):
        midx.extend(tags["meridian_edges"][name])
    if not _ints_ok(midx) or sorted(midx) != list(range(len(meridian.Edges()))):
        return False, "meridian_partition", ("the meridian edge tag indices %s do not partition the %d edges"
                                             % (sorted(midx), len(meridian.Edges()))), values
    n_we = len(wall_m.Edges())
    for name in decl_edges:
        if name not in tags["wall_edges"]:
            return False, "wall_edges", ("declared edge tag %s is missing from tags.json" % (name,)), values
        lst = tags["wall_edges"][name]
        if not lst:
            return False, "wall_edges", ("declared edge tag %s has an empty index list" % (name,)), values
        if len(set(lst)) != len(lst):
            return False, "wall_edges", ("edge tag %s has duplicate indices" % (name,)), values
        if not _ints_ok(lst) or not all(0 <= i < n_we for i in lst):
            return False, "wall_edges", ("edge tag %s has an index outside 0..%d" % (name, n_we - 1)), values
    pnames = [p["name"] for p in tags["planes"]]
    if len(set(pnames)) != len(pnames):
        return False, "planes", "tags.json has duplicate plane names", values
    if sorted(pnames) != sorted(decl_planes):
        return False, "planes", ("plane names %s are not the declared %s"
                                 % (sorted(pnames), sorted(decl_planes))), values
    for p in tags["planes"]:
        x = p.get("x") if isinstance(p, dict) else None
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
            return False, "planes", ("plane %r has a missing or non-finite x"
                                     % (p.get("name") if isinstance(p, dict) else p,)), values
    return True, None, "", values


RULES = {"RDY-PROFILE": rule_profile, "RDY-BREP": rule_brep, "RDY-FACEW": rule_facew,
         "RDY-EDGE": rule_edge, "RDY-STL": rule_stl, "RDY-THROAT": rule_throat, "RDY-TAGS": rule_tags}


def _run_rule(fn, rule, ctx):
    """One rule through the tri-state net: a row, a missing input refusal, or an error with the cause."""
    try:
        ok, chk, det, vals = fn(ctx)
    except MissingInput as e:
        return {"rule": rule, "ok": False, "check": "missing_input",
                "detail": _scrub("%s is missing" % (e,), ctx["out_dir"]), "values": {}}
    except Exception as e:
        return {"rule": rule, "ok": False, "check": "raised",
                "detail": _scrub(("%s: %s" % (type(e).__name__, e))[:300], ctx["out_dir"]), "values": {}}
    return {"rule": rule, "ok": bool(ok), "check": None if ok else chk,
            "detail": _scrub("" if ok else det, ctx["out_dir"]), "values": vals}


def _peek_json(out_dir, name):
    """OUT_DIR/name as a dict, or None when it is absent, unreadable or not a dict."""
    try:
        obj = common.read_json(os.path.join(out_dir, name))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _peek_template_id(out_dir):
    """tags.json's template_id, or None: the default declaration path needs it."""
    tags = _peek_json(out_dir, "tags.json")
    tid = tags.get("template_id") if tags is not None else None
    return tid if isinstance(tid, str) and tid else None


def check(out_dir, h_m, declaration_path=None):
    """The S5 report: the seven RDY-* rules of RULE_ORDER, stopped at the first failing row."""
    if isinstance(h_m, bool) or not isinstance(h_m, (int, float)) or not math.isfinite(h_m) or h_m <= 0:
        raise ValueError("h_m must be a finite number > 0, got %r" % (h_m,))
    ctx = {"out_dir": out_dir, "h": float(h_m), "declaration_path": declaration_path, "cache": {}}
    if ctx["declaration_path"] is None:
        tid = _peek_template_id(out_dir)
        if tid:
            ctx["declaration_path"] = os.path.join(HERE, "templates", tid.split("/")[0], "template.json")
    rows = []
    status = "ready"
    for rule in RULE_ORDER:
        row = _run_rule(RULES[rule], rule, ctx)
        rows.append(row)
        if not row["ok"]:
            status = "error" if row["check"] == "raised" else "refused"
            break
    geom = _peek_json(out_dir, "geom.json")     # read here too: an early refusal never loads it
    template_id = geom.get("template_id") if isinstance(geom, dict) else None
    params_sha = geom.get("params_sha") if isinstance(geom, dict) else None
    fail = None if status == "ready" else rows[-1]
    return {
        "version": VERSION,
        "status": status,
        "rule": None if fail is None else fail["rule"],
        "check": None if fail is None else fail["check"],
        "detail": "" if fail is None else fail["detail"],
        "h_m": float(h_m),
        "template_id": template_id if isinstance(template_id, str) else None,
        "params_sha": params_sha if isinstance(params_sha, str) else None,
        "results": [dict((k, r[k]) for k in RESULT_KEYS) for r in rows],
    }


def write_report(out_dir, report):
    """write_json the report to OUT_DIR/readiness.json; returns that path."""
    path = os.path.join(out_dir, READY_FILE)
    common.write_json(path, report)
    return path


def main(argv):
    """--selftest runs the fixture suite; check OUT_DIR H_M writes readiness.json and prints the verdict."""
    if len(argv) == 1 and argv[0] == "--selftest":
        try:
            selftest()
        except Exception:
            import traceback
            traceback.print_exc()
            return 1
        return 0
    if len(argv) == 3 and argv[0] == "check":
        out_dir = os.path.abspath(argv[1])
        report = check(out_dir, float(argv[2]))
        write_report(out_dir, report)
        print(common.canonical_json({"status": report["status"], "rule": report["rule"],
                                     "check": report["check"]}))
        return 0 if report["status"] == "ready" else 1
    sys.stderr.write(USAGE + chr(10))
    return 2


V = cq.Vector
_FX = {"R_i": 0.03, "R_e": 0.01, "Lu": 0.03, "L": 0.06, "Lx": 0.01}     # the nominal's meridian, poly5


def _fx_blend(xi):
    return 10 * xi ** 3 - 15 * xi ** 4 + 6 * xi ** 5


def _fx_line(a, b):
    return cq.Edge.makeLine(V(a[0], a[1], 0), V(b[0], b[1], 0))


def _fx_face(edges):
    return cq.Face.makeFromWires(cq.Wire.assembleEdges(edges))


def _fx_revolve(face):
    return cq.Solid.revolve(face, 360, V(0, 0, 0), V(1, 0, 0))


def _fx_nozzle_meridian(lip=0.0, split=0.0, n=200):
    """The nominal poly5 meridian; lip > 0 steps the wall in by lip at x = 0 (the law then starts at R_i - lip);
    split > 0 cuts the exit-tube line split metres before its end into two collinear edges."""
    Ri, Re, Lu, L, Lx = (_FX[k] for k in ("R_i", "R_e", "Lu", "L", "Lx"))
    r0 = Ri - lip
    pts = [V(L * k / n, r0 - (r0 - Re) * _fx_blend(k / n), 0) for k in range(n + 1)]
    edges = [_fx_line((-Lu, 0), (-Lu, Ri)), _fx_line((-Lu, Ri), (0, Ri))]
    if lip > 0:
        edges.append(_fx_line((0, Ri), (0, r0)))
    edges.append(cq.Edge.makeSpline(pts, tangents=[V(1, 0, 0), V(1, 0, 0)]))
    if split > 0:
        edges += [_fx_line((L, Re), (L + Lx - split, Re)), _fx_line((L + Lx - split, Re), (L + Lx, Re))]
    else:
        edges.append(_fx_line((L, Re), (L + Lx, Re)))
    edges += [_fx_line((L + Lx, Re), (L + Lx, 0)), _fx_line((L + Lx, 0), (-Lu, 0))]
    return _fx_face(edges)


def _fx_crossing_wp():
    """docs/16 §C: the planner's axis-crossing spline (r dips to -0.010 m), as a fresh closed Workplane."""
    return (cq.Workplane("XY").moveTo(0, 0).lineTo(0, 0.05)
            .spline([(0.1, 0.03), (0.15, -0.01), (0.2, 0.03), (0.3, 0.035)], includeCurrent=True)
            .lineTo(0.3, 0).close())


def _fx_crossing():
    face = cq.Face.makeFromWires(_fx_crossing_wp().wires().val())
    fluid = _fx_crossing_wp().revolve(360, (0, 0, 0), (1, 0, 0)).val()
    return face, fluid


def _fx_bowtie():
    P = [(0, 0), (0.1, 0), (0.1, 0.01), (0, 0.03), (0.05, 0.03)]
    face = _fx_face([_fx_line(P[i], P[(i + 1) % 5]) for i in range(5)])
    return face, _fx_revolve(face)


def _fx_variant(nom_dir, dst, fluid=None, meridian=None):
    """A copy of the nominal export dir with fluid.brep and/or meridian.brep replaced."""
    shutil.copytree(nom_dir, dst)
    for name, shp in (("fluid.brep", fluid), ("meridian.brep", meridian)):
        if shp is not None and not shp.exportBrep(os.path.join(dst, name)):
            raise RuntimeError("exportBrep returned False for %s" % (name,))
    return dst


def selftest():
    """The fourteen fixture lines: every RDY-* rule refuses its planted defect, the nominal is ready."""
    with tempfile.TemporaryDirectory() as td:
        nom = os.path.join(td, "nom")
        res = export.run_pipeline(export.TEMPLATE, dict(export.NOMINAL), nom)
        if res["status"] != "ok":
            raise RuntimeError("nominal export did not build: %s %s" % (res["rule"], res["message"]))
        h = H_SELFTEST_M
        rep = check(nom, h)
        assert rep["status"] == "ready", rep["status"]
        assert rep["rule"] is None and rep["check"] is None and rep["detail"] == ""
        assert [r["rule"] for r in rep["results"]] == list(RULE_ORDER)
        assert all(r["ok"] for r in rep["results"])
        assert list(rep.keys()) == list(READY_KEYS)
        assert rep["template_id"] == "nozzle_contraction/1", rep["template_id"]
        print("[ok] nominal ready at h %g: r_min_wall %.6g face width %.6g edge %.6g D/h %.6g"
              % (h, rep["results"][0]["values"]["r_min_wall_m"],
                 rep["results"][2]["values"]["face_width_min_m"],
                 rep["results"][3]["values"]["edge_min_m"],
                 rep["results"][5]["values"]["d_over_h"]))
        face, fluid = _fx_crossing()
        d = _fx_variant(nom, os.path.join(td, "crossing"), fluid=fluid, meridian=face)
        rep = check(d, h)
        assert rep["status"] == "refused" and rep["rule"] == "RDY-PROFILE" and rep["check"] == "r_min", \
            (rep["status"], rep["rule"], rep["check"])
        assert len(rep["results"]) == 1
        assert rep["template_id"] == "nozzle_contraction/1" and isinstance(rep["params_sha"], str), rep["template_id"]
        assert rep["results"][0]["values"]["r_min_all_m"] < -0.009, rep["results"][0]["values"]
        assert measure.valid(fluid)["value"] == 1 and measure.n_solids(fluid)["value"] == 1
        sag = _fx_face([_fx_line((-0.03, 0), (-0.03, 0.03)), _fx_line((-0.03, 0.03), (0.07, 0.01)),
                        _fx_line((0.07, 0.01), (0.07, 0)),
                        cq.Edge.makeThreePointArc(V(0.07, 0, 0), V(0.02, -0.005, 0), V(-0.03, 0, 0))])
        d = _fx_variant(nom, os.path.join(td, "sag"), meridian=sag)
        rs = check(d, h)
        vs = rs["results"][0]["values"]
        assert rs["rule"] == "RDY-PROFILE" and rs["check"] == "r_min", (rs["rule"], rs["check"])
        assert vs["r_min_all_m"] < -0.004 and vs["r_min_wall_m"] > 0.009, vs
        print("[ok] axis crossing refused RDY-PROFILE r_min (r_min %.6g), though its fluid is valid with 1 solid;"
              " an axis edge sagging to %.6g is refused too" % (rep["results"][0]["values"]["r_min_all_m"],
                                                              vs["r_min_all_m"]))
        face, fluid = _fx_bowtie()
        d = _fx_variant(nom, os.path.join(td, "bowtie"), fluid=fluid, meridian=face)
        rep = check(d, h)
        assert rep["rule"] == "RDY-PROFILE" and rep["check"] == "selfx", (rep["rule"], rep["check"])
        vals = rep["results"][0]["values"]
        assert vals["r_min_wall_m"] >= 0.01 - 1e-12, vals["r_min_wall_m"]
        assert measure.n_solids(fluid)["value"] == 1
        print("[ok] bow-tie refused RDY-PROFILE selfx (r_min_wall %.6g over %d wall edges, fluid 1 solid)"
              % (vals["r_min_wall_m"], vals["n_wall_edges"]))
        m = _fx_nozzle_meridian(lip=5e-5)
        lip_dir = _fx_variant(nom, os.path.join(td, "lip"), fluid=_fx_revolve(m), meridian=m)
        rep = check(lip_dir, h)
        assert rep["rule"] == "RDY-FACEW" and rep["check"] == "face_width", (rep["rule"], rep["check"])
        assert rep["results"][0]["ok"] and rep["results"][1]["ok"]
        w = rep["results"][2]["values"]["face_width_min_m"]
        assert abs(w - 5e-5) <= 1e-9, w
        print("[ok] 0.05 mm lip refused RDY-FACEW after PROFILE and BREP passed: min face width %.6e m" % (w,))
        nom_fluid = cq.Shape.importBrep(os.path.join(nom, "fluid.brep"))
        tags = common.read_json(os.path.join(nom, "tags.json"))
        shell = cq.Shell.makeShell([f for i, f in enumerate(nom_fluid.Faces())
                                    if i not in tags["face_tags"]["inlet"]])
        d = _fx_variant(nom, os.path.join(td, "noinlet"), fluid=shell)
        rep = check(d, h)
        assert rep["rule"] == "RDY-BREP" and rep["check"] == "n_solids", (rep["rule"], rep["check"])
        assert rep["results"][0]["ok"]
        assert rep["results"][1]["values"] == {"valid": 1, "n_solids": 0, "free_wires": 1}, \
            rep["results"][1]["values"]
        print("[ok] no-inlet shell refused RDY-BREP n_solids (values %s)"
              % (common.canonical_json(rep["results"][1]["values"]),))
        m = _fx_nozzle_meridian(split=1e-4)
        d = _fx_variant(nom, os.path.join(td, "split"), meridian=m)
        rep = check(d, h)
        assert rep["rule"] == "RDY-EDGE" and rep["check"] == "edge_length", (rep["rule"], rep["check"])
        assert all(r["ok"] for r in rep["results"][:3])
        vals = rep["results"][3]["values"]
        assert vals["on_meridian"] == 1 and abs(vals["edge_min_m"] - 1e-4) <= 1e-9, vals
        print("[ok] split exit tube refused RDY-EDGE on the meridian after 3 ok rows: min edge %.6e m"
              % (vals["edge_min_m"],))
        stl_a = shutil.copytree(nom, os.path.join(td, "stl_a"))
        tags = common.read_json(os.path.join(stl_a, "tags.json"))
        order = [t for t in tags["stl_patches"] if t != "outlet"]
        export.write_named_stl(cq.Shape.importBrep(os.path.join(stl_a, "fluid.brep")), tags["face_tags"], order,
                               os.path.join(stl_a, "fluid_named.stl"), export.STL_LIN_REL * 0.02,
                               export.STL_ANG_RAD)
        export.stl_report(os.path.join(stl_a, "fluid_named.stl"), os.path.join(stl_a, "stl_repair.json"))
        geom = common.read_json(os.path.join(stl_a, "geom.json"))
        geom["files"]["fluid_named.stl"] = common.sha256_file(os.path.join(stl_a, "fluid_named.stl"))
        geom["files"]["stl_repair.json"] = common.sha256_file(os.path.join(stl_a, "stl_repair.json"))
        common.write_json(os.path.join(stl_a, "geom.json"), geom)
        rep = check(stl_a, h)
        assert rep["rule"] == "RDY-STL" and rep["check"] == "watertight", (rep["rule"], rep["check"])
        stl_b = shutil.copytree(nom, os.path.join(td, "stl_b"))
        rp = common.read_json(os.path.join(stl_b, "stl_repair.json"))
        rp["before"]["closed"] = False
        common.write_json(os.path.join(stl_b, "stl_repair.json"), rp)
        rep = check(stl_b, h)
        assert rep["rule"] == "RDY-STL" and rep["check"] == "report_sha", (rep["rule"], rep["check"])
        print("[ok] stl refused two ways: dropped outlet patch -> watertight, doctored report -> report_sha")
        rep = check(nom, 1.5e-3)
        assert rep["rule"] == "RDY-THROAT" and rep["check"] == "throat", (rep["rule"], rep["check"])
        assert all(r["ok"] for r in rep["results"][:5])
        dh = rep["results"][5]["values"]["d_over_h"]
        assert abs(dh - 0.02 / 1.5e-3) <= 1e-6, dh
        print("[ok] nominal at h 0.0015 refused RDY-THROAT after 5 ok rows: D/h %.6g" % (dh,))
        ta = shutil.copytree(nom, os.path.join(td, "tags_a"))
        tg = common.read_json(os.path.join(ta, "tags.json"))
        del tg["face_tags"]["inlet"]
        common.write_json(os.path.join(ta, "tags.json"), tg)
        rep = check(ta, h)
        assert rep["rule"] == "RDY-TAGS" and rep["check"] == "face_tags", (rep["rule"], rep["check"])
        assert all(r["ok"] for r in rep["results"][:6])
        tb = shutil.copytree(nom, os.path.join(td, "tags_b"))
        tg = common.read_json(os.path.join(tb, "tags.json"))
        tg["face_tags"]["outlet"] = [0]
        common.write_json(os.path.join(tb, "tags.json"), tg)
        rep = check(tb, h)
        assert rep["rule"] == "RDY-TAGS" and rep["check"] == "face_partition", (rep["rule"], rep["check"])
        assert all(r["ok"] for r in rep["results"][:6])
        print("[ok] tags refused two ways: dropped inlet -> face_tags, duplicated outlet index -> face_partition")
        m = _fx_nozzle_meridian(lip=5e-5)
        d = _fx_variant(nom, os.path.join(td, "order"), fluid=_fx_revolve(m), meridian=m)
        tg = common.read_json(os.path.join(d, "tags.json"))
        del tg["face_tags"]["inlet"]
        common.write_json(os.path.join(d, "tags.json"), tg)
        rep = check(d, h)
        assert rep["rule"] == "RDY-FACEW" and len(rep["results"]) == 3, (rep["rule"], len(rep["results"]))
        print("[ok] order wins: lip + dropped inlet refused by %s with %d rows"
              % (rep["rule"], len(rep["results"])))
        d = shutil.copytree(nom, os.path.join(td, "miss"))
        os.remove(os.path.join(d, "fluid.brep"))
        rep = check(d, h)
        assert rep["status"] == "refused" and rep["rule"] == "RDY-BREP" and rep["check"] == "missing_input", \
            (rep["status"], rep["rule"], rep["check"])
        assert rep["detail"] == "fluid.brep is missing", rep["detail"]
        assert rep["results"][0]["ok"]
        empty = os.path.join(td, "empty")
        os.mkdir(empty)
        rep = check(empty, h)
        assert rep["status"] == "refused" and rep["rule"] == "RDY-PROFILE" and rep["check"] == "missing_input", \
            (rep["status"], rep["rule"], rep["check"])
        print("[ok] missing: deleted fluid -> RDY-BREP missing_input, empty dir -> RDY-PROFILE missing_input")
        d = shutil.copytree(nom, os.path.join(td, "garbage"))
        with open(os.path.join(d, "fluid.brep"), "w", encoding="utf-8") as f:
            f.write("garbage")
        rep = check(d, h)
        assert rep["status"] == "error" and rep["rule"] == "RDY-BREP" and rep["check"] == "raised", \
            (rep["status"], rep["rule"], rep["check"])
        cj = common.canonical_json(rep)
        for bad in (td, os.path.abspath(td), os.path.abspath(td).replace(os.sep, "/")):
            assert bad not in cj
            assert bad.replace(chr(92), chr(92) + chr(92)) not in cj
        assert "<out>" in cj
        print("[ok] garbage fluid is an error (raised), and the canonical report holds <out>, no path")
        a = common.canonical_json(check(nom, h))
        b = common.canonical_json(check(nom, h))
        assert a == b
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        blobs = []
        for i in (0, 1):
            p = subprocess.run([sys.executable, os.path.abspath(__file__), "check", nom, "0.0005"],
                               capture_output=True, text=True, encoding="utf-8", env=env, timeout=300)
            assert p.returncode == 0, (p.returncode, p.stdout[-400:], p.stderr[-400:])
            assert '"status":"ready"' in p.stdout, p.stdout
            blobs.append(open(os.path.join(nom, READY_FILE), "rb").read())
        assert blobs[0] == blobs[1]
        assert common.read_json(os.path.join(nom, READY_FILE)) == check(nom, h)
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "check", lip_dir, "0.0005"],
                           capture_output=True, text=True, encoding="utf-8", env=env, timeout=300)
        assert p.returncode == 1, (p.returncode, p.stdout[-400:])
        assert '"status":"refused"' in p.stdout and '"rule":"RDY-FACEW"' in p.stdout, p.stdout
        print("[ok] two fresh CLI runs byte-identical and ready, the refused lip CLI exits 1 with RDY-FACEW")
        for x in (0, -1e-3, float("nan"), float("inf"), True, "5e-4", None):
            raised = False
            try:
                check(nom, x)
            except ValueError:
                raised = True
            assert raised, x
        print("[ok] h guard raises ValueError on all 7 bad h values")
    print("SELFTEST PASS")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
