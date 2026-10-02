#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""readiness.py - stage S5 of the CAD loop (docs/16 §D, §I): ordered RDY-* CFD-readiness refusals on one S3 export directory, before any mesh.

RDY-BIND and RDY-READBACK come first and RDY-SECTION, RDY-MERIDIAN and RDY-FLUIDBODY last (docs/16a §E, §G AMG-5).
They reimplement Amagine3D ideas (https://github.com/amagine-ai/Amagine3D, e608dc6, skills/text-a3d/):
build_manifest.py file_binding_errors (every consumed file re-hashed against its recorded sha, and again after the
audit), brep_measurements.py measure_step (a result discarded if its input changed while it was measured),
cad_helpers.py _intersection_volume and assembly_check.py's part_overlaps check (a pairwise BRep common volume, an
empty result counting 0). No code was copied; the file hashes go through common.stable_file_snapshot, the one
verbatim Amagine3D block of this tree (AMG-3).

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
from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.ShapeAnalysis import ShapeAnalysis_FreeBounds, ShapeAnalysis_Wire
from OCP.TopAbs import TopAbs_EDGE, TopAbs_WIRE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Compound
from OCP.gp import gp_Dir, gp_Pln, gp_Pnt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common
import export
import measure

# readiness judges the FILES an S3 export wrote (geom.json and every file its files map binds, RDY-BIND), never
# the template's parameters: the template's PRF rules refuse bad inputs, and
# readiness refuses bad OUTPUTS, including ones no PRF rule can see (docs/16 §C fact 1: an axis-crossing and a
# bow-tie profile both revolve into "valid" solids). h is the caller's mesh size in metres; the rules stop at the
# first refusal, in RULE_ORDER, and a rule whose input file is missing refuses, never passes.
VERSION = 2
RULE_ORDER = ("RDY-BIND", "RDY-READBACK", "RDY-PROFILE", "RDY-BREP", "RDY-FACEW", "RDY-EDGE", "RDY-STL",
              "RDY-THROAT", "RDY-TAGS", "RDY-SECTION", "RDY-MERIDIAN", "RDY-FLUIDBODY")
READY_KEYS = ("version", "status", "rule", "check", "detail", "h_m", "template_id", "params_sha", "inputs",
              "results")
RESULT_KEYS = ("rule", "ok", "check", "detail", "values")
AXIS_TOL = measure.AXIS_TOL    # m: a meridian vertex with |y| <= this is on the axis; |z| <= this is in-plane
SELFX_PREC = 1e-7              # m: ShapeAnalysis_Wire precision, as the template's PRF-SELFX uses
N_SAMPLE = 2001                # points per meridian edge (measure._edge_points, uniform in the parameter)
EDGE_FRAC = 0.5                # RDY-EDGE: every edge >= EDGE_FRAC * h
THROAT_MIN = 20.0              # RDY-THROAT: D_throat / h >= THROAT_MIN
H_SELFTEST_M = 5e-4            # m: about the nominal's L1 core cell (meridian 2.2e-3 m2 over 10k cells)
READY_FILE = "readiness.json"
BOUND_FILES = tuple(export.BREP_FILES) + tuple(f for f in export.EXPORT_FILES if f != "geom.json")
MERIDIAN_REL = 1e-9            # RDY-MERIDIAN: fluid meridian-plane section vs 2 x the meridian face area, rel
OVERLAP_REL = 1e-12            # RDY-FLUIDBODY: fluid-body common volume <= OVERLAP_REL x V_fluid
WETTED_REL = 1e-9              # RDY-FLUIDBODY: coincident wetted area vs the WET_TAGS tag area, rel
WET_TAGS = ("wall_contraction", "wall_exit")   # the fluid face tags the body wets (docs/16a line 84)
MERIDIAN_PLANE = {"name": "meridian", "origin": [0.0, 0.0, 0.0], "normal": [0.0, 0.0, 1.0], "u": [1.0, 0.0, 0.0]}
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
    # docs/16 section H.5 item 4: the box carries both upstream tags and a build carries exactly
    # the one its upstream_role names, so the other role's <role>_upstream tag is not wanted here.
    # A geom.json without the key is a slip build (the box's original, and only, role until then).
    role = (geom.get("params") or {}).get("upstream_role", "slip")
    other = {"slip": "wall", "wall": "slip"}.get(role)
    want_faces = sorted(t for t in decl_faces if t != (other + "_upstream" if other else None))
    values = {"n_face_tags": len(tags["face_tags"]), "n_fluid_faces": len(fluid.Faces()),
              "n_meridian_edges": len(meridian.Edges())}
    if not (tags["template_id"] == geom["template_id"] == decl["template_id"]):
        return False, "template_id", ("template_id disagrees: tags %r, geom %r, declaration %r"
                                      % (tags["template_id"], geom["template_id"],
                                         decl["template_id"])), values
    if geom["declaration_sha"] != common.sha256_file(ctx["declaration_path"]):
        return False, "declaration_sha", "geom.json's declaration_sha does not match the declaration file", values
    if sorted(tags["face_tags"]) != want_faces:
        return False, "face_tags", ("face tags %s are not the declared %s"
                                    % (sorted(tags["face_tags"]), want_faces)), values
    fidx = []
    for name in sorted(tags["face_tags"]):
        fidx.extend(tags["face_tags"][name])
    if not _ints_ok(fidx) or sorted(fidx) != list(range(len(fluid.Faces()))):
        return False, "face_partition", ("the face tag indices %s do not partition the %d fluid faces"
                                         % (sorted(fidx), len(fluid.Faces()))), values
    patches = tags["stl_patches"]
    if len(set(patches)) != len(patches):
        return False, "stl_patches", "tags.json has duplicate stl patch names", values
    if sorted(patches) != want_faces:
        return False, "stl_patches", ("stl patches %s are not the declared face names %s"
                                      % (sorted(patches), want_faces)), values
    for name in sorted(tags["face_tags"]):
        area = 0.0
        for i in tags["face_tags"][name]:
            area += export.face_props(fluid.Faces()[i])
        if not area > 0.0:
            return False, "face_area", ("face tag %s has total area %.6e m^2, not > 0" % (name, area)), values
    want = sorted(want_faces + ["axis"])
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


def _snap(path):
    """The sha256 of one file through common.stable_file_snapshot, or None when it is not a stable regular file."""
    s = common.stable_file_snapshot(path)
    return s["sha256"] if s["stable"] is True else None


def rule_bind(ctx):
    """RDY-BIND (docs/16a §E): geom.json, the declaration and every file in geom.json's files map are stable
    regular files, and each file's sha256 is the one geom.json recorded; check() hashes them all again after the
    last rule and discards the judgement if any changed."""
    out = ctx["out_dir"]
    inputs = ctx["inputs"]
    inputs["geom.json"] = _snap(os.path.join(out, "geom.json"))
    if inputs["geom.json"] is None:
        raise MissingInput("geom.json")
    dpath = ctx["declaration_path"]
    inputs["template.json"] = None if dpath is None else _snap(dpath)
    if inputs["template.json"] is None:
        raise MissingInput("template.json")
    geom = _load_json(ctx, "geom.json")
    files = geom.get("files") if isinstance(geom, dict) else None
    values = {"n_bound": 0, "unbound": [], "mismatch": [], "changed": [], "after": None}
    if (not isinstance(files, dict)
            or not all(isinstance(k, str) and isinstance(v, str) for k, v in files.items())
            or not all(k == os.path.basename(k) and not k.startswith(".") for k in files)
            or not all(n in files for n in BOUND_FILES)):
        return False, "files_map", "geom.json's files map is not a name -> sha map holding every exported file", values
    names = sorted(files)
    for n in names:
        inputs[n] = _snap(os.path.join(out, n))
    ctx["bind"] = [("geom.json", os.path.join(out, "geom.json")), ("template.json", dpath)]
    ctx["bind"] += [(n, os.path.join(out, n)) for n in names]
    values["n_bound"] = len(names)
    values["unbound"] = [n for n in names if inputs[n] is None]
    if values["unbound"]:
        return False, "unbound", ("%s is missing or not a stable regular file" % (values["unbound"][0],)), values
    values["mismatch"] = [n for n in names if inputs[n] != files[n]]
    if values["mismatch"]:
        n = values["mismatch"][0]
        return False, "sha", ("%s sha %s is not the geom.json hash %s" % (n, inputs[n], files[n])), values
    if geom.get("declaration_sha") != inputs["template.json"]:
        return False, "declaration", "the declaration's sha is not geom.json's declaration_sha", values
    return True, None, "", values


def rule_readback(ctx):
    """RDY-READBACK (docs/16a §E): geom.json records the export readback (export.readback, AMG-4) as ok, no field."""
    geom = _load_json(ctx, "geom.json")
    rb = geom.get("readback")
    values = {"status": None, "reason_id": None, "field": None, "n_fields": None}
    if not isinstance(rb, dict):
        return False, "missing", "geom.json records no readback", values
    fields = rb.get("fields")
    values["status"], values["reason_id"], values["field"] = rb.get("status"), rb.get("reason_id"), rb.get("field")
    values["n_fields"] = len(fields) if isinstance(fields, list) else None
    if rb.get("status") != "ok" or fields != []:
        return False, "status", ("geom.json's readback is %r (%s: %s)"
                                 % (rb.get("status"), rb.get("reason_id"), rb.get("field"))), values
    return True, None, "", values


def _xplane(q):
    """The {name, origin, normal, u} section plane of one tags.json plane row {name, x}: normal +x, u +y."""
    return {"name": q["name"], "origin": [float(q["x"]), 0.0, 0.0], "normal": [1.0, 0.0, 0.0], "u": [0.0, 1.0, 0.0]}


def _section(shape, plane, what):
    """measure.section_props, with a status error raised: an unmeasurable section is an error, never a pass."""
    sp = measure.section_props(shape, plane)
    if sp["status"] == "error":
        raise RuntimeError("%s section at %s: %s" % (what, plane["name"], sp["detail"]))
    return sp


def rule_section(ctx):
    """RDY-SECTION (docs/16a §E): at every named plane the fluid's section is 1 island with 0 holes, and the body's
    is 1 island with 1 hole wherever the plane lies within the body's x bounds (outside them it must miss)."""
    tags = _load_json(ctx, "tags.json")
    fluid = _load_shape(ctx, "fluid.brep")
    body = _load_shape(ctx, "body.brep")
    b = export._bounds(body)
    bx0, bx1 = b[0], b[3]
    rows = []
    values = {"planes": rows, "n_body_planes": 0}
    for q in tags["planes"]:
        sp = _section(fluid, _xplane(q), "fluid")
        rows.append({"name": q["name"], "x_m": float(q["x"]), "fluid_islands": sp["n_islands"],
                     "fluid_holes": sp["n_holes"], "fluid_area_m2": sp["area_m2"], "body": None,
                     "body_islands": None, "body_holes": None, "body_area_m2": None})
        if sp["status"] != "ok":
            return False, "fluid_section", ("the fluid section at %s is refused %s"
                                            % (q["name"], sp["reason_id"])), values
        if sp["n_islands"] != 1 or sp["n_holes"] != 0:
            return False, "fluid_section", ("the fluid section at %s has %d islands and %d holes, not 1 and 0"
                                            % (q["name"], sp["n_islands"], sp["n_holes"])), values
    for q, row in zip(tags["planes"], rows):
        sp = _section(body, _xplane(q), "body")
        x = float(q["x"])
        if sp["status"] == "refused" and sp["reason_id"] == "MEAS-NOSECTION" and (x < bx0 or x > bx1):
            row["body"] = "outside"
            continue
        row["body"] = "cut"
        row["body_islands"], row["body_holes"], row["body_area_m2"] = sp["n_islands"], sp["n_holes"], sp["area_m2"]
        if sp["status"] != "ok":
            return False, "body_section", ("the body section at %s is refused %s"
                                           % (q["name"], sp["reason_id"])), values
        if sp["n_islands"] != 1 or sp["n_holes"] != 1:
            return False, "body_section", ("the body section at %s has %d islands and %d holes, not 1 and 1"
                                           % (q["name"], sp["n_islands"], sp["n_holes"])), values
        values["n_body_planes"] += 1
    if values["n_body_planes"] == 0:
        return False, "body_section", "no named plane cuts the body", values
    return True, None, "", values


def rule_meridian(ctx):
    """RDY-MERIDIAN (docs/16a §E): the fluid's section by the meridian plane z = 0 is 1 island with 0 holes and its
    area is 2 x the meridian face area of meridian.brep AND of meridian.step (read at METRE), each within
    MERIDIAN_REL; the plan names meridian.brep only, the tree also judges the STEP the mesher is handed."""
    fluid = _load_shape(ctx, "fluid.brep")
    meridian = _load_shape(ctx, "meridian.brep")
    step_path = os.path.join(ctx["out_dir"], "meridian.step")
    if not os.path.isfile(step_path):
        raise MissingInput("meridian.step")
    sp = _section(fluid, MERIDIAN_PLANE, "fluid")
    values = {"n_islands": sp["n_islands"], "n_holes": sp["n_holes"], "section_area_m2": sp["area_m2"],
              "meridian_area_m2": None, "step_area_m2": None, "rel_brep": None, "rel_step": None}
    if sp["status"] != "ok" or sp["n_islands"] != 1 or sp["n_holes"] != 0:
        return False, "section", ("the fluid's meridian section is %s with %d islands and %d holes, not ok, 1 and 0"
                                  % (sp["status"], sp["n_islands"], sp["n_holes"])), values
    s = sp["area_m2"]
    a = measure.gprop(meridian, "area")[0]
    values["meridian_area_m2"] = a
    values["rel_brep"] = (s - 2.0 * a) / (2.0 * a)
    if not abs(values["rel_brep"]) <= MERIDIAN_REL:
        return False, "meridian_brep", ("the fluid's meridian section %.12e m^2 is not 2 x meridian.brep's %.12e m^2"
                                        " (rel %.3e)" % (s, a, values["rel_brep"])), values
    a = measure.gprop(export.read_step(step_path), "area")[0]
    values["step_area_m2"] = a
    values["rel_step"] = (s - 2.0 * a) / (2.0 * a)
    if not abs(values["rel_step"]) <= MERIDIAN_REL:
        return False, "meridian_step", ("the fluid's meridian section %.12e m^2 is not 2 x meridian.step's %.12e m^2"
                                        " (rel %.3e)" % (s, a, values["rel_step"])), values
    return True, None, "", values


def _faces_compound(shape):
    """A TopoDS_Compound of every face of the shape (no solid): the argument of a face-face common."""
    c = TopoDS_Compound()
    bb = BRep_Builder()
    bb.MakeCompound(c)
    for f in shape.Faces():
        bb.Add(c, f.wrapped)
    return c


def _common(a, b):
    """BRepAlgoAPI_Common of two TopoDS shapes as a cq.Shape; a failed operation raises."""
    op = BRepAlgoAPI_Common(a, b)
    op.Build()
    if not op.IsDone():
        raise RuntimeError("BRepAlgoAPI_Common failed")
    return cq.Shape.cast(op.Shape())


def rule_fluidbody(ctx):
    """RDY-FLUIDBODY (docs/16a §E): the fluid and the body do not overlap (common volume <= OVERLAP_REL x V_fluid,
    an empty common counting 0) and their COINCIDENT wetted area - the face-face common, never a distance
    (docs/16a §D.10) - equals geom.json's WET_TAGS tag area within WETTED_REL."""
    fluid = _load_shape(ctx, "fluid.brep")
    body = _load_shape(ctx, "body.brep")
    geom = _load_json(ctx, "geom.json")
    v_f = measure.gprop(fluid, "volume")[0]
    ov = _common(fluid.wrapped, body.wrapped)
    overlap = measure.gprop(ov, "volume")[0]
    values = {"fluid_volume_m3": v_f, "overlap_m3": overlap, "n_overlap_solids": len(ov.Solids()),
              "wetted_area_m2": None, "wall_tag_area_m2": None, "wetted_rel": None}
    if not overlap <= OVERLAP_REL * v_f:
        return False, "overlap", ("fluid and body overlap by %.6e m^3, above %.0e x V_fluid %.6e m^3"
                                  % (overlap, OVERLAP_REL, v_f)), values
    rows = geom["tags"]["fluid_faces"]
    missing = [t for t in WET_TAGS if t not in rows]
    if missing:
        return False, "wall_tags", ("geom.json has no fluid face tag %s" % (missing[0],)), values
    wall = sum(float(rows[t]["area_m2"]) for t in WET_TAGS)
    wet = measure.gprop(_common(_faces_compound(fluid), _faces_compound(body)), "area")[0]
    values["wetted_area_m2"], values["wall_tag_area_m2"] = wet, wall
    values["wetted_rel"] = (wet - wall) / wall
    if not abs(values["wetted_rel"]) <= WETTED_REL:
        return False, "wetted", ("the coincident fluid-body area %.12e m^2 is not the wall tag area %.12e m^2"
                                 " (rel %.3e)" % (wet, wall, values["wetted_rel"])), values
    return True, None, "", values


RULES = {"RDY-PROFILE": rule_profile, "RDY-BREP": rule_brep, "RDY-FACEW": rule_facew,
         "RDY-EDGE": rule_edge, "RDY-STL": rule_stl, "RDY-THROAT": rule_throat, "RDY-TAGS": rule_tags,
         "RDY-BIND": rule_bind, "RDY-READBACK": rule_readback, "RDY-SECTION": rule_section,
         "RDY-MERIDIAN": rule_meridian, "RDY-FLUIDBODY": rule_fluidbody}


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


def _after_bind(ctx, rows, between_hook):
    """The RDY-BIND after-check: every bound file hashed again; the rows, or one refused RDY-BIND row if any changed.

    between_hook(out_dir), a selftest hook, runs after the last rule and before the second hash.
    """
    if between_hook is not None:
        between_hook(ctx["out_dir"])
    changed = [n for n, p in ctx["bind"] if _snap(p) != ctx["inputs"][n]]
    first = rows[0]
    if not changed:
        first["values"]["after"] = "unchanged"
        return rows
    vals = dict(first["values"], changed=changed, after="changed")
    det = "%s changed during judgement; the judgement is discarded" % (changed[0],)
    return [{"rule": "RDY-BIND", "ok": False, "check": "changed", "detail": det, "values": vals}]


def check(out_dir, h_m, declaration_path=None, between_hook=None):
    """The S5 report: the twelve RDY-* rules of RULE_ORDER, stopped at the first failing row, bound by RDY-BIND.

    Every consumed file is hashed by common.stable_file_snapshot before the rules (RDY-BIND, the first rule) and
    again after them; `inputs` holds the first hashes (None: not a stable regular file). If any file changed in
    between, the judgement is discarded and the report is one refused RDY-BIND row, check "changed".
    """
    if isinstance(h_m, bool) or not isinstance(h_m, (int, float)) or not math.isfinite(h_m) or h_m <= 0:
        raise ValueError("h_m must be a finite number > 0, got %r" % (h_m,))
    ctx = {"out_dir": out_dir, "h": float(h_m), "declaration_path": declaration_path, "cache": {},
           "inputs": {}, "bind": None}
    if ctx["declaration_path"] is None:
        tid = _peek_template_id(out_dir)
        if tid:
            ctx["declaration_path"] = os.path.join(HERE, "templates", tid.split("/")[0], "template.json")
    rows = []
    for rule in RULE_ORDER:
        row = _run_rule(RULES[rule], rule, ctx)
        rows.append(row)
        if not row["ok"]:
            break
    if rows[0]["ok"]:
        rows = _after_bind(ctx, rows, between_hook)
    fail = None
    for r in rows:
        if not r["ok"]:
            fail = r
    status = "ready" if fail is None else ("error" if fail["check"] == "raised" else "refused")
    geom = _peek_json(out_dir, "geom.json")     # read here too: an early refusal never loads it
    template_id = geom.get("template_id") if isinstance(geom, dict) else None
    params_sha = geom.get("params_sha") if isinstance(geom, dict) else None
    return {
        "version": VERSION,
        "status": status,
        "rule": None if fail is None else fail["rule"],
        "check": None if fail is None else fail["check"],
        "detail": "" if fail is None else fail["detail"],
        "h_m": float(h_m),
        "template_id": template_id if isinstance(template_id, str) else None,
        "params_sha": params_sha if isinstance(params_sha, str) else None,
        "inputs": dict(sorted(ctx["inputs"].items())),
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


def _fx_rebind(d, names):
    """Re-hash the named files into d/geom.json's files map, so a planted defect reaches its own rule, not RDY-BIND."""
    gp = os.path.join(d, "geom.json")
    geom = common.read_json(gp)
    for n in names:
        geom["files"][n] = common.sha256_file(os.path.join(d, n))
    common.write_json(gp, geom)


def _fx_variant(nom_dir, dst, fluid=None, meridian=None, body=None):
    """A copy of the nominal export dir with fluid.brep, meridian.brep and/or body.brep replaced and re-bound."""
    shutil.copytree(nom_dir, dst)
    names = []
    for name, shp in (("fluid.brep", fluid), ("meridian.brep", meridian), ("body.brep", body)):
        if shp is not None:
            if not shp.exportBrep(os.path.join(dst, name)):
                raise RuntimeError("exportBrep returned False for %s" % (name,))
            names.append(name)
    _fx_rebind(dst, names)
    return dst


def _row(rep, rule):
    """The results row of one rule in a report; KeyError when the report stopped before it."""
    for r in rep["results"]:
        if r["rule"] == rule:
            return r
    raise KeyError(rule)


def _fx_append_byte(out_dir):
    """The between_hook of the selftest: one byte appended to tags.json (the JSON still parses)."""
    with open(os.path.join(out_dir, "tags.json"), "ab") as f:
        f.write(b" ")


def selftest():
    """The twenty-three fixture lines: every RDY-* rule refuses its planted defect, the nominal is ready."""
    with tempfile.TemporaryDirectory() as td:
        nom = os.path.join(td, "nom")
        res = export.run_pipeline(export.TEMPLATE, dict(export.NOMINAL), nom)
        if res["status"] != "ok":
            raise RuntimeError("nominal export did not build: %s %s" % (res["rule"], res["message"]))
        h = H_SELFTEST_M
        rep = check(nom, h)
        assert rep["status"] == "ready", (rep["status"], rep["rule"], rep["check"], rep["detail"])
        assert rep["rule"] is None and rep["check"] is None and rep["detail"] == ""
        assert [r["rule"] for r in rep["results"]] == list(RULE_ORDER)
        assert all(r["ok"] for r in rep["results"])
        assert list(rep.keys()) == list(READY_KEYS)
        assert rep["template_id"] == "nozzle_contraction/1", rep["template_id"]
        vm, vf = _row(rep, "RDY-MERIDIAN")["values"], _row(rep, "RDY-FLUIDBODY")["values"]
        print("[ok] nominal ready at h %g: r_min_wall %.6g face width %.6g edge %.6g D/h %.6g;"
              " meridian rel %.1e / %.1e, overlap %.3g m^3, wetted rel %.1e"
              % (h, _row(rep, "RDY-PROFILE")["values"]["r_min_wall_m"],
                 _row(rep, "RDY-FACEW")["values"]["face_width_min_m"],
                 _row(rep, "RDY-EDGE")["values"]["edge_min_m"], _row(rep, "RDY-THROAT")["values"]["d_over_h"],
                 vm["rel_brep"], vm["rel_step"], vf["overlap_m3"], vf["wetted_rel"]))
        vb = _row(rep, "RDY-BIND")["values"]
        want = sorted(list(BOUND_FILES) + ["geom.json", "template.json"])
        assert sorted(rep["inputs"]) == want and len(want) == 13, sorted(rep["inputs"])
        decl = os.path.join(HERE, "templates", "nozzle_contraction", "template.json")
        for n in want:
            p = decl if n == "template.json" else os.path.join(nom, n)
            assert rep["inputs"][n] == common.sha256_file(p), n
        assert vb == {"n_bound": 11, "unbound": [], "mismatch": [], "changed": [], "after": "unchanged"}, vb
        sp = _row(rep, "RDY-SECTION")["values"]
        assert [q["body"] for q in sp["planes"]] == ["outside", "cut", "cut", "cut"], sp["planes"]
        print("[ok] inputs map: %d shas equal sha256_file, hashed before and after (unchanged); sections at %d planes,"
              " the body cut at %d" % (len(want), len(sp["planes"]), sp["n_body_planes"]))
        face, fluid = _fx_crossing()
        d = _fx_variant(nom, os.path.join(td, "crossing"), fluid=fluid, meridian=face)
        rep = check(d, h)
        assert rep["status"] == "refused" and rep["rule"] == "RDY-PROFILE" and rep["check"] == "r_min", \
            (rep["status"], rep["rule"], rep["check"])
        assert len(rep["results"]) == 3
        assert rep["template_id"] == "nozzle_contraction/1" and isinstance(rep["params_sha"], str), rep["template_id"]
        vc = _row(rep, "RDY-PROFILE")["values"]
        assert vc["r_min_all_m"] < -0.009, vc
        assert measure.valid(fluid)["value"] == 1 and measure.n_solids(fluid)["value"] == 1
        sag = _fx_face([_fx_line((-0.03, 0), (-0.03, 0.03)), _fx_line((-0.03, 0.03), (0.07, 0.01)),
                        _fx_line((0.07, 0.01), (0.07, 0)),
                        cq.Edge.makeThreePointArc(V(0.07, 0, 0), V(0.02, -0.005, 0), V(-0.03, 0, 0))])
        d = _fx_variant(nom, os.path.join(td, "sag"), meridian=sag)
        rs = check(d, h)
        vs = _row(rs, "RDY-PROFILE")["values"]
        assert rs["rule"] == "RDY-PROFILE" and rs["check"] == "r_min", (rs["rule"], rs["check"])
        assert vs["r_min_all_m"] < -0.004 and vs["r_min_wall_m"] > 0.009, vs
        print("[ok] axis crossing refused RDY-PROFILE r_min (r_min %.6g), though its fluid is valid with 1 solid;"
              " an axis edge sagging to %.6g is refused too" % (vc["r_min_all_m"], vs["r_min_all_m"]))
        face, fluid = _fx_bowtie()
        d = _fx_variant(nom, os.path.join(td, "bowtie"), fluid=fluid, meridian=face)
        rep = check(d, h)
        assert rep["rule"] == "RDY-PROFILE" and rep["check"] == "selfx", (rep["rule"], rep["check"])
        vals = _row(rep, "RDY-PROFILE")["values"]
        assert vals["r_min_wall_m"] >= 0.01 - 1e-12, vals["r_min_wall_m"]
        assert measure.n_solids(fluid)["value"] == 1
        print("[ok] bow-tie refused RDY-PROFILE selfx (r_min_wall %.6g over %d wall edges, fluid 1 solid)"
              % (vals["r_min_wall_m"], vals["n_wall_edges"]))
        m = _fx_nozzle_meridian(lip=5e-5)
        lip_dir = _fx_variant(nom, os.path.join(td, "lip"), fluid=_fx_revolve(m), meridian=m)
        rep = check(lip_dir, h)
        assert rep["rule"] == "RDY-FACEW" and rep["check"] == "face_width", (rep["rule"], rep["check"])
        assert all(r["ok"] for r in rep["results"][:4])
        w = _row(rep, "RDY-FACEW")["values"]["face_width_min_m"]
        assert abs(w - 5e-5) <= 1e-9, w
        print("[ok] 0.05 mm lip refused RDY-FACEW after PROFILE and BREP passed: min face width %.6e m" % (w,))
        nom_fluid = cq.Shape.importBrep(os.path.join(nom, "fluid.brep"))
        tags = common.read_json(os.path.join(nom, "tags.json"))
        shell = cq.Shell.makeShell([f for i, f in enumerate(nom_fluid.Faces())
                                    if i not in tags["face_tags"]["inlet"]])
        d = _fx_variant(nom, os.path.join(td, "noinlet"), fluid=shell)
        rep = check(d, h)
        assert rep["rule"] == "RDY-BREP" and rep["check"] == "n_solids", (rep["rule"], rep["check"])
        assert all(r["ok"] for r in rep["results"][:3])
        assert _row(rep, "RDY-BREP")["values"] == {"valid": 1, "n_solids": 0, "free_wires": 1}, \
            _row(rep, "RDY-BREP")["values"]
        print("[ok] no-inlet shell refused RDY-BREP n_solids (values %s)"
              % (common.canonical_json(_row(rep, "RDY-BREP")["values"]),))
        m = _fx_nozzle_meridian(split=1e-4)
        d = _fx_variant(nom, os.path.join(td, "split"), meridian=m)
        rep = check(d, h)
        assert rep["rule"] == "RDY-EDGE" and rep["check"] == "edge_length", (rep["rule"], rep["check"])
        assert all(r["ok"] for r in rep["results"][:5])
        vals = _row(rep, "RDY-EDGE")["values"]
        assert vals["on_meridian"] == 1 and abs(vals["edge_min_m"] - 1e-4) <= 1e-9, vals
        print("[ok] split exit tube refused RDY-EDGE on the meridian after 5 ok rows: min edge %.6e m"
              % (vals["edge_min_m"],))
        stl_a = shutil.copytree(nom, os.path.join(td, "stl_a"))
        tags = common.read_json(os.path.join(stl_a, "tags.json"))
        order = [t for t in tags["stl_patches"] if t != "outlet"]
        export.write_named_stl(cq.Shape.importBrep(os.path.join(stl_a, "fluid.brep")), tags["face_tags"], order,
                               os.path.join(stl_a, "fluid_named.stl"), export.STL_LIN_REL * 0.02,
                               export.STL_ANG_RAD)
        export.stl_report(os.path.join(stl_a, "fluid_named.stl"), os.path.join(stl_a, "stl_repair.json"))
        _fx_rebind(stl_a, ["fluid_named.stl", "stl_repair.json"])
        rep = check(stl_a, h)
        assert rep["rule"] == "RDY-STL" and rep["check"] == "watertight", (rep["rule"], rep["check"])
        stl_b = shutil.copytree(nom, os.path.join(td, "stl_b"))
        rp = common.read_json(os.path.join(stl_b, "stl_repair.json"))
        rp["before"]["closed"] = False
        common.write_json(os.path.join(stl_b, "stl_repair.json"), rp)
        rep = check(stl_b, h)
        assert rep["rule"] == "RDY-BIND" and rep["check"] == "sha", (rep["rule"], rep["check"])
        assert _row(rep, "RDY-BIND")["values"]["mismatch"] == ["stl_repair.json"], _row(rep, "RDY-BIND")["values"]
        stl_c = shutil.copytree(nom, os.path.join(td, "stl_c"))
        geom = common.read_json(os.path.join(stl_c, "geom.json"))
        geom["watertight"]["n_components"] = 2
        common.write_json(os.path.join(stl_c, "geom.json"), geom)
        rep = check(stl_c, h)
        assert rep["rule"] == "RDY-STL" and rep["check"] == "geom_agrees", (rep["rule"], rep["check"])
        print("[ok] stl refused three ways: dropped outlet -> watertight, doctored report -> RDY-BIND sha,"
              " doctored geom.json watertight block -> geom_agrees")
        rep = check(nom, 1.5e-3)
        assert rep["rule"] == "RDY-THROAT" and rep["check"] == "throat", (rep["rule"], rep["check"])
        assert all(r["ok"] for r in rep["results"][:7])
        dh = _row(rep, "RDY-THROAT")["values"]["d_over_h"]
        assert abs(dh - 0.02 / 1.5e-3) <= 1e-6, dh
        print("[ok] nominal at h 0.0015 refused RDY-THROAT after 7 ok rows: D/h %.6g" % (dh,))
        ta = shutil.copytree(nom, os.path.join(td, "tags_a"))
        tg = common.read_json(os.path.join(ta, "tags.json"))
        del tg["face_tags"]["inlet"]
        common.write_json(os.path.join(ta, "tags.json"), tg)
        _fx_rebind(ta, ["tags.json"])
        rep = check(ta, h)
        assert rep["rule"] == "RDY-TAGS" and rep["check"] == "face_tags", (rep["rule"], rep["check"])
        assert all(r["ok"] for r in rep["results"][:8])
        tb = shutil.copytree(nom, os.path.join(td, "tags_b"))
        tg = common.read_json(os.path.join(tb, "tags.json"))
        tg["face_tags"]["outlet"] = [0]
        common.write_json(os.path.join(tb, "tags.json"), tg)
        _fx_rebind(tb, ["tags.json"])
        rep = check(tb, h)
        assert rep["rule"] == "RDY-TAGS" and rep["check"] == "face_partition", (rep["rule"], rep["check"])
        assert all(r["ok"] for r in rep["results"][:8])
        print("[ok] tags refused two ways: dropped inlet -> face_tags, duplicated outlet index -> face_partition")
        m = _fx_nozzle_meridian(lip=5e-5)
        d = _fx_variant(nom, os.path.join(td, "order"), fluid=_fx_revolve(m), meridian=m)
        tg = common.read_json(os.path.join(d, "tags.json"))
        del tg["face_tags"]["inlet"]
        common.write_json(os.path.join(d, "tags.json"), tg)
        _fx_rebind(d, ["tags.json"])
        rep = check(d, h)
        assert rep["rule"] == "RDY-FACEW" and len(rep["results"]) == 5, (rep["rule"], len(rep["results"]))
        print("[ok] order wins: lip + dropped inlet refused by %s with %d rows"
              % (rep["rule"], len(rep["results"])))
        d = shutil.copytree(nom, os.path.join(td, "miss"))
        os.remove(os.path.join(d, "fluid.brep"))
        rep = check(d, h)
        assert rep["status"] == "refused" and rep["rule"] == "RDY-BIND" and rep["check"] == "unbound", \
            (rep["status"], rep["rule"], rep["check"])
        assert rep["detail"] == "fluid.brep is missing or not a stable regular file", rep["detail"]
        assert len(rep["results"]) == 1 and rep["inputs"]["fluid.brep"] is None
        empty = os.path.join(td, "empty")
        os.mkdir(empty)
        rep = check(empty, h)
        assert rep["status"] == "refused" and rep["rule"] == "RDY-BIND" and rep["check"] == "missing_input", \
            (rep["status"], rep["rule"], rep["check"])
        assert rep["detail"] == "geom.json is missing" and rep["inputs"] == {"geom.json": None}, rep["inputs"]
        print("[ok] missing: deleted fluid -> RDY-BIND unbound, empty dir -> RDY-BIND missing_input (geom.json)")
        d = shutil.copytree(nom, os.path.join(td, "garbage"))
        with open(os.path.join(d, "fluid.brep"), "w", encoding="utf-8") as f:
            f.write("garbage")
        _fx_rebind(d, ["fluid.brep"])
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
        cor = os.path.join(td, "corner")
        res = export.run_pipeline(export.TEMPLATE, dict(export.CORNER), cor)
        assert res["status"] == "ok", (res["rule"], res["message"])
        d = shutil.copytree(nom, os.path.join(td, "swap"))
        shutil.copyfile(os.path.join(cor, "fluid.brep"), os.path.join(d, "fluid.brep"))
        rep = check(d, h)
        assert rep["rule"] == "RDY-BIND" and rep["check"] == "sha" and len(rep["results"]) == 1, \
            (rep["rule"], rep["check"])
        assert _row(rep, "RDY-BIND")["values"]["mismatch"] == ["fluid.brep"], _row(rep, "RDY-BIND")["values"]
        print("[ok] the corner variant's fluid.brep swapped in after export is refused RDY-BIND sha (fluid.brep)")
        d = shutil.copytree(nom, os.path.join(td, "byte"))
        _fx_append_byte(d)
        assert common.read_json(os.path.join(d, "tags.json")) == common.read_json(os.path.join(nom, "tags.json"))
        rep = check(d, h)
        assert rep["rule"] == "RDY-BIND" and rep["check"] == "sha" and len(rep["results"]) == 1, \
            (rep["rule"], rep["check"])
        assert _row(rep, "RDY-BIND")["values"]["mismatch"] == ["tags.json"], _row(rep, "RDY-BIND")["values"]
        print("[ok] one byte appended to tags.json (same JSON) is refused RDY-BIND sha (tags.json)")
        d = shutil.copytree(nom, os.path.join(td, "hook"))
        rep = check(d, h, between_hook=_fx_append_byte)
        assert rep["status"] == "refused" and rep["rule"] == "RDY-BIND" and rep["check"] == "changed", \
            (rep["status"], rep["rule"], rep["check"])
        assert len(rep["results"]) == 1 and rep["results"][0]["values"]["changed"] == ["tags.json"], rep["results"]
        assert rep["inputs"]["tags.json"] == common.sha256_file(os.path.join(nom, "tags.json"))
        rep2 = check(d, h)
        assert rep2["rule"] == "RDY-BIND" and rep2["check"] == "sha", (rep2["rule"], rep2["check"])
        print("[ok] a file rewritten between the before and after hash is refused RDY-BIND changed, the judgement"
              " discarded (1 row); the next check refuses it RDY-BIND sha")
        d = shutil.copytree(nom, os.path.join(td, "readback"))
        geom = common.read_json(os.path.join(d, "geom.json"))
        geom["readback"].update(status="refused", reason_id="EXP-READBACK", field="fluid.step:bounds",
                                fields=["fluid.step:bounds"])
        common.write_json(os.path.join(d, "geom.json"), geom)
        rep = check(d, h)
        assert rep["rule"] == "RDY-READBACK" and rep["check"] == "status" and len(rep["results"]) == 2, \
            (rep["rule"], rep["check"])
        assert rep["results"][0]["ok"]
        print("[ok] a geom.json recording a refused readback is refused RDY-READBACK status after RDY-BIND passed")
        tags = common.read_json(os.path.join(nom, "tags.json"))
        xp = dict((q["name"], q["x"]) for q in tags["planes"])
        sph = cq.Solid.makeSphere(0.003, V(xp["exit_plane"], 0, 0), angleDegrees1=-90, angleDegrees2=90)
        void = nom_fluid.cut(sph)
        assert len(void.Faces()) == 6 and void.Faces()[5].geomType() == "SPHERE", [f.geomType() for f in void.Faces()]
        d = _fx_variant(nom, os.path.join(td, "void"), fluid=void)
        tg = common.read_json(os.path.join(d, "tags.json"))
        tg["face_tags"]["wall_exit"] = tg["face_tags"]["wall_exit"] + [5]
        common.write_json(os.path.join(d, "tags.json"), tg)
        _fx_rebind(d, ["tags.json"])
        rep = check(d, h)
        assert rep["rule"] == "RDY-SECTION" and rep["check"] == "fluid_section", (rep["rule"], rep["check"])
        assert all(r["ok"] for r in rep["results"][:9])
        pl = _row(rep, "RDY-SECTION")["values"]["planes"][-1]
        assert pl["name"] == "exit_plane" and (pl["fluid_islands"], pl["fluid_holes"]) == (1, 1), pl
        print("[ok] a spherical void on the throat plane is refused RDY-SECTION fluid_section after 9 ok rows:"
              " exit_plane has 1 island and 1 hole")
        p = dict(export.NOMINAL, CR=export.NOMINAL["CR"] / 1.0201)
        de = os.path.join(td, "de1")
        res = export.run_pipeline(export.TEMPLATE, p, de)
        assert res["status"] == "ok", (res["rule"], res["message"])
        d = shutil.copytree(nom, os.path.join(td, "mstep"))
        shutil.copyfile(os.path.join(de, "meridian.step"), os.path.join(d, "meridian.step"))
        _fx_rebind(d, ["meridian.step"])
        rep = check(d, h)
        assert rep["rule"] == "RDY-MERIDIAN" and rep["check"] == "meridian_step", (rep["rule"], rep["check"])
        assert all(r["ok"] for r in rep["results"][:10])
        vm = _row(rep, "RDY-MERIDIAN")["values"]
        assert abs(vm["rel_brep"]) <= 1e-12 and -2.5e-3 < vm["rel_step"] < -2.0e-3, vm
        print("[ok] a meridian.step from D_e +1 %% is refused RDY-MERIDIAN meridian_step: rel %.4e (brep rel %.1e)"
              % (vm["rel_step"], vm["rel_brep"]))
        nom_body = cq.Shape.importBrep(os.path.join(nom, "body.brep"))
        d = _fx_variant(nom, os.path.join(td, "shift"), body=nom_body.translate(V(0, 1e-4, 0)))
        rep = check(d, h)
        assert rep["rule"] == "RDY-FLUIDBODY" and rep["check"] == "overlap", (rep["rule"], rep["check"])
        assert all(r["ok"] for r in rep["results"][:11])
        ov = _row(rep, "RDY-FLUIDBODY")["values"]["overlap_m3"]
        assert abs(ov / 2.6e-7 - 1.0) < 1e-3, ov
        print("[ok] a body shifted 0.1 mm radially is refused RDY-FLUIDBODY overlap %.6e m^3 after 11 ok rows" % (ov,))
        box = cq.Solid.makeBox(1, 1, 1, V(xp["outlet"] - 1e-3, -0.5, -0.5))
        d = _fx_variant(nom, os.path.join(td, "short"), body=nom_body.cut(box))
        rep = check(d, h)
        assert rep["rule"] == "RDY-FLUIDBODY" and rep["check"] == "wetted", (rep["rule"], rep["check"])
        vf = _row(rep, "RDY-FLUIDBODY")["values"]
        assert vf["overlap_m3"] == 0.0 and -7.3e-3 < vf["wetted_rel"] < -7.1e-3, vf
        print("[ok] a body 1 mm short of the exit plane is refused RDY-FLUIDBODY wetted: rel %.4e, overlap 0"
              % (vf["wetted_rel"],))
    print("SELFTEST PASS")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
