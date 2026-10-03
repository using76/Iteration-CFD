#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""export.py - the S3 export of the CAD loop (docs/16 §D S3, §I, gates GC-5 and GC-4 of §H.3): fluid / body / meridian STEP declared in metres, a named-patch STL, tags.json, probes.json and geom.json (units m, scale 1, axis +x), every file read back by an independent reader (EXP-READBACK).

The export readback reimplements three Amagine3D ideas (https://github.com/amagine-ai/Amagine3D, e608dc6,
skills/text-a3d/): export_audit.py audit_exports / _audit_step / _geometry_errors (every file re-read and compared
with the kernel record, here with metre-relative tolerances, because their absolute-mm ones pass a 0.7x and a 1.3x
scaled STL of ours), brep_tessellation.py tessellate_brep (mesh a cleaned copy) and mesh_topology.py
physical_body_count (count bodies after a weld, here with trimesh instead of manifold3d). No code was copied.

Usage:
  python export.py --selftest
  python export.py run TEMPLATE_PY PARAMS_JSON OUT_DIR
  python export.py determinism TEMPLATE_PY N
  python export.py gmsh-span STEP_PATH TARGET_UNIT
"""
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

import cadquery as cq
from OCP.STEPControl import STEPControl_Writer, STEPControl_Reader, STEPControl_AsIs, STEPControl_Controller
from OCP.Interface import Interface_Static
from OCP.IFSelect import IFSelect_RetDone
from OCP.APIHeaderSection import APIHeaderSection_MakeHeader
from OCP.TCollection import TCollection_HAsciiString
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRep import BRep_Tool
from OCP.TopLoc import TopLoc_Location
from OCP.TopAbs import TopAbs_REVERSED
from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
from OCP.BRepTools import BRepTools
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepAlgoAPI import BRepAlgoAPI_Check
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common
import measure
import runner
import schema

# The template never runs in this process: every build goes through runner.run_job or the run
# CLI, whose child is the only place the template module is loaded. STEP is written with OCCT's
# write.step.unit set to M together with xstep.cascade.unit, AFTER the controller init and the
# writer construction, so the file declares SI_UNIT($,.METRE.) over its metre values; CadQuery's
# default declares SI_UNIT(.MILLI.,.METRE.) over the same numbers - a silent 1000x. gmsh always
# runs in a fresh child process (the gmsh-span sub-command) because it shares OCCT state with the
# writer and reader in this process. The STL is ONE BRepMesh_IncrementalMesh on a cleaned
# copy of the fluid, reversed faces flipped on write, ASCII so each face tag is its own solid;
# tools/geom/stl_repair.py --weld 0 is the watertightness oracle.
SCHEMA_MEASURE = "cad-measure/1"
STL_LIN_REL = 1e-3            # BRepMesh linear deflection = STL_LIN_REL * D_e, in m
STL_ANG_RAD = 0.05            # BRepMesh angular deflection, rad
STEP_UNIT = "M"
STEP_TIMESTAMP = "1970-01-01T00:00:00"
STEP_AUTHOR = "meteor-cfd tools/cad/export.py"
STEP_ORG = "Iterations Co., Ltd."
STEP_SYSTEM = "meteor-cfd tools/cad"
SPAN_TOL = 1e-9               # m, GC-5 gmsh x-span
ROUNDTRIP_TOL = 1e-9          # rel, GC-5 STEP volume / area
STL_VOL_TOL = 2e-3            # rel, GC-5 STL volume vs BREP
READBACK_BOUNDS_M = 1e-9      # m, each of the 6 STEP readback bounds vs the BREP's (docs/16a §B.2)
READBACK_VOL_REL = 1e-9       # rel, STEP readback volume (solids) or area (the meridian face) vs the BREP
READBACK_AREA_REL = STL_LIN_REL   # rel, per-patch STL area vs the geom.json tag area, tied to the deflection
READBACK_CX_M = 1e-9          # m, a planar patch's STL area-weighted centroid x vs its named plane's x
READBACK_STEP = (("fluid.step", "fluid.brep", "solid"), ("body.step", "body.brep", "solid"),
                 ("meridian.step", "meridian.brep", "face"))
BUILD_TIMEOUT_S = 180
STL_REPAIR = os.path.join(common.REPO, "tools", "geom", "stl_repair.py")
TEMPLATE = os.path.join(HERE, "templates", "nozzle_contraction", "template.py")
NOMINAL = {"D_i": 0.06, "CR": 9.0, "L_over_Di": 1.0, "law": "poly5", "x_m": None,
           "Lx_over_De": 0.5, "Lu_over_Di": 0.5, "upstream_role": "slip", "t_wall": 0.003}
CORNER = {"D_i": 0.06, "CR": 9.0, "L_over_Di": 0.5, "law": "cubic_matched", "x_m": 0.8,
          "Lx_over_De": 1.0, "Lu_over_Di": 0.5, "upstream_role": "slip", "t_wall": 0.01}
# The turbulent nominal of docs/16 section H.5 (lines 521-523): D_i 0.30, CR 2, poly5, L/D_i 1.5,
# a 0.60 m no-slip upstream pipe, air at 293.15 K; the turbulent recipe meshes it at y+1 = 1.
TURB_NOMINAL = {"D_i": 0.30, "CR": 2.0, "L_over_Di": 1.5, "law": "poly5", "x_m": None,
                "Lx_over_De": 0.5, "Lu_over_Di": 2.0, "upstream_role": "wall", "t_wall": 0.003}
BREP_FILES = ("fluid.brep", "body.brep", "meridian.brep", "wall_meridian.brep")
EXPORT_FILES = ("fluid.step", "body.step", "meridian.step", "fluid_named.stl", "stl_repair.json",
                "tags.json", "probes.json", "geom.json")
GEOM_KEYS = ("version", "template_id", "template_sha", "declaration_sha", "params", "params_sha", "units",
             "scale", "axis", "step_length_unit", "x_span_m", "x_span_expected_m", "gmsh_import", "stl",
             "watertight", "step_roundtrip", "readback", "tags", "files", "env")
TAGS_KEYS = ("version", "template_id", "face_tags", "meridian_edges", "wall_edges", "planes", "stl_patches")
PROBES_KEYS = ("version", "template_id", "params_sha", "rows")
USAGE = ("usage: python export.py --selftest" + chr(10)
         + "       python export.py run TEMPLATE_PY PARAMS_JSON OUT_DIR" + chr(10)
         + "       python export.py determinism TEMPLATE_PY N" + chr(10)
         + "       python export.py gmsh-span STEP_PATH TARGET_UNIT")


def write_step(shape, path, unit=STEP_UNIT):
    """One shape as STEP declaring `unit` (M = METRE): init, writer, THEN the two statics, transfer, header."""
    STEPControl_Controller.Init_s()
    w = STEPControl_Writer()
    Interface_Static.SetCVal_s("xstep.cascade.unit", unit)
    Interface_Static.SetCVal_s("write.step.unit", unit)
    if w.Transfer(shape.wrapped, STEPControl_AsIs) != IFSelect_RetDone:
        raise RuntimeError("STEPControl transfer failed for %s" % (os.path.basename(path),))
    h = APIHeaderSection_MakeHeader(w.Model())
    h.SetName(TCollection_HAsciiString(os.path.splitext(os.path.basename(path))[0]))
    h.SetTimeStamp(TCollection_HAsciiString(STEP_TIMESTAMP))
    h.SetAuthorValue(1, TCollection_HAsciiString(STEP_AUTHOR))
    h.SetOrganizationValue(1, TCollection_HAsciiString(STEP_ORG))
    h.SetOriginatingSystem(TCollection_HAsciiString(STEP_SYSTEM))
    h.SetAuthorisation(TCollection_HAsciiString(""))
    if w.Write(path) != IFSelect_RetDone:
        raise RuntimeError("STEP write failed: %s" % (path,))


def read_step(path):
    """One shape back from STEP at scale 1 (metres); the static is set only after the reader exists."""
    STEPControl_Controller.Init_s()
    r = STEPControl_Reader()
    Interface_Static.SetCVal_s("xstep.cascade.unit", "M")
    if r.ReadFile(path) != IFSelect_RetDone:
        raise RuntimeError("STEP read failed: %s" % (path,))
    r.TransferRoots()
    return cq.Shape.cast(r.OneShape())


def step_length_unit(path):
    """METRE, MILLI.METRE or None, from the STEP line carrying LENGTH_UNIT()."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "LENGTH_UNIT()" in line:
                if "SI_UNIT($,.METRE.)" in line:
                    return "METRE"
                if "SI_UNIT(.MILLI.,.METRE.)" in line:
                    return "MILLI.METRE"
                return None
    return None


def gmsh_span(path, target_unit=STEP_UNIT):
    """The x-span gmsh reads at scale `target_unit`, from a FRESH process (gmsh shares OCCT state)."""
    pr = subprocess.run([sys.executable, os.path.abspath(__file__), "gmsh-span", path, target_unit],
                        capture_output=True, text=True, encoding="utf-8", timeout=120,
                        env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    if pr.returncode != 0:
        raise RuntimeError("gmsh-span exited %d: %s" % (pr.returncode, pr.stderr[-400:]))
    lines = [s for s in pr.stdout.splitlines() if s.strip()]
    return json.loads(lines[-1])


def gmsh_span_main(path, target):
    """The gmsh-span sub-command: import the STEP in gmsh at OCCTargetUnit `target`, print one JSON line."""
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.NumThreads", 1)
    gmsh.option.setString("Geometry.OCCTargetUnit", target)
    gmsh.model.occ.importShapes(path)
    gmsh.model.occ.synchronize()
    xs = [gmsh.model.getValue(0, tag, [])[0] for dim, tag in gmsh.model.getEntities(0)]
    print(common.canonical_json({"target_unit": target, "n_volumes": len(gmsh.model.getEntities(3)),
                                 "n_vertices": len(xs), "x_min": min(xs), "x_max": max(xs),
                                 "x_span_m": max(xs) - min(xs), "gmsh": gmsh.GMSH_API_VERSION}))
    gmsh.finalize()
    return 0


def write_named_stl(shape, face_tags, order, path, lin, ang, flip_reversed=True):
    """ASCII STL, one `solid <tag>` per face tag in `order`, from ONE BRepMesh_IncrementalMesh.

    A REVERSED face is written with its second and third corners swapped, so the written
    corner order carries the outward normal (a cq.Solid.makeBox has 3 REVERSED faces of 6;
    without the flip stl_repair sees non-manifold edges and an open shell).

    The mesh is made on a BRepBuilderAPI_Copy (geometry copied, no mesh) cleaned by
    BRepTools.Clean_s, so a triangulation already on the caller's shape - finer or coarser -
    can never stand in for the requested deflection (docs/16a §B.2, after
    Amagine3D's tessellate_brep).
    """
    copy = BRepBuilderAPI_Copy(shape.wrapped, True, False).Shape()
    BRepTools.Clean_s(copy)
    shape = cq.Shape.cast(copy)
    m = BRepMesh_IncrementalMesh(shape.wrapped, lin, False, ang, False)
    m.Perform()
    faces = shape.Faces()
    lines = []
    total = 0
    per_tag = {}
    for tag in order:
        out = ["solid %s" % (tag,)]
        n_tag = 0
        for fi in face_tags[tag]:
            face = faces[fi]
            loc = TopLoc_Location()
            tri = BRep_Tool.Triangulation_s(face.wrapped, loc)
            if tri is None:
                raise RuntimeError("face %d (tag %s) has no triangulation" % (fi, tag))
            tr = loc.Transformation()
            rev = face.wrapped.Orientation() == TopAbs_REVERSED
            pts = [tri.Node(k).Transformed(tr) for k in range(1, tri.NbNodes() + 1)]
            for k in range(1, tri.NbTriangles() + 1):
                a, b, c = tri.Triangle(k).Get()
                if rev and flip_reversed:
                    b, c = c, b
                p, q, r = pts[a - 1], pts[b - 1], pts[c - 1]
                ux, uy, uz = q.X() - p.X(), q.Y() - p.Y(), q.Z() - p.Z()
                vx, vy, vz = r.X() - p.X(), r.Y() - p.Y(), r.Z() - p.Z()
                nx = uy * vz - uz * vy
                ny = uz * vx - ux * vz
                nz = ux * vy - uy * vx
                nn = math.sqrt(nx * nx + ny * ny + nz * nz)
                if nn == 0.0:
                    nx = ny = nz = 0.0
                else:
                    nx, ny, nz = nx / nn, ny / nn, nz / nn
                out.append("facet normal %s %s %s" % (repr(nx), repr(ny), repr(nz)))
                out.append("outer loop")
                for g in (p, q, r):
                    out.append("vertex %s %s %s" % (repr(g.X()), repr(g.Y()), repr(g.Z())))
                out.append("endloop")
                out.append("endfacet")
                n_tag += 1
        out.append("endsolid %s" % (tag,))
        lines.extend(out)
        per_tag[tag] = n_tag
        total += n_tag
    common.atomic_write(path, chr(10).join(lines) + chr(10))
    return {"triangles": total, "per_tag": per_tag}


def stl_report(stl_path, json_path):
    """tools/geom/stl_repair.py --weld 0 on the STL: the watertightness oracle, as a report dict."""
    pr = subprocess.run([sys.executable, STL_REPAIR, stl_path, "--weld", "0", "--json", json_path],
                        capture_output=True, text=True, encoding="utf-8", timeout=300,
                        env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    if pr.returncode != 0:
        raise RuntimeError("stl_repair exited %d: %s" % (pr.returncode, (pr.stderr or pr.stdout)[-400:]))
    return common.read_json(json_path)


def _gprop_checked(shape, kind):
    """measure.gprop's value, or ValueError MEAS-GPROP when its estimate exceeds measure.GPROP_REL_MAX."""
    value, est = measure.gprop(shape, kind)
    if not (est <= measure.GPROP_REL_MAX):
        raise ValueError("MEAS-GPROP: %s relative error estimate %.3e exceeds %.0e at eps %.0e"
                         % (kind, est, measure.GPROP_REL_MAX, measure.GPROP_EPS))
    return value


def face_props(face):
    """Face area in m^2 by adaptive integration (measure.gprop)."""
    return _gprop_checked(face, "area")


def edge_length(edge):
    """Edge length in m by adaptive integration (measure.gprop)."""
    return _gprop_checked(edge, "length")


def solid_volume(shape):
    """Solid volume in m^3 by adaptive integration (measure.gprop)."""
    return _gprop_checked(shape, "volume")


def tag_table(fluid, meridian, wall_m, value):
    """Per-tag area / length, type names and indices, grouped by the shape the tag indexes.

    Grouped, not flat: the five fluid face tags reappear as meridian edge tags (inlet, outlet,
    slip_upstream, wall_contraction, wall_exit), so one flat dict would drop every face area.
    """
    faces = {}
    for tag, idx in value["face_tags"].items():
        fs = [fluid.Faces()[i] for i in idx]
        faces[tag] = {"kind": "face", "shape": "fluid", "index": list(idx),
                      "type": sorted(set(f.geomType() for f in fs)),
                      "area_m2": sum(face_props(f) for f in fs),
                      "gprop_eps": measure.GPROP_EPS,
                      "gprop_est_rel": max([measure.gprop(f, "area")[1] for f in fs] + [0.0])}
    groups = {"fluid_faces": faces}
    for key, shape_name, shape, tags in (("meridian_edges", "meridian", meridian, value["meridian_edges"]),
                                         ("wall_edges", "wall_meridian", wall_m, value["wall_edges"])):
        rows = {}
        for tag, idx in tags.items():
            es = [shape.Edges()[i] for i in idx]
            rows[tag] = {"kind": "edge", "shape": shape_name, "index": list(idx),
                         "type": sorted(set(e.geomType() for e in es)),
                         "length_m": sum(edge_length(e) for e in es),
                         "gprop_eps": measure.GPROP_EPS,
                         "gprop_est_rel": max([measure.gprop(e, "length")[1] for e in es] + [0.0])}
        groups[key] = rows
    return groups


def _bounds(shape):
    """The 6 bounds (xmin, ymin, zmin, xmax, ymax, zmax) from the geometry, never a triangulation or tolerance."""
    b = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape.wrapped, b, False, False)
    return list(b.Get())


def _step_readback(step_path, brep_path, kind):
    """One STEP re-read at METRE against its BREP: (row, failing fields); nothing raises out of it."""
    name = os.path.basename(step_path)
    what = "volume" if kind == "solid" else "area"
    row = {"read": False, "valid": None, "bopcheck": None, "n_solids": None, "n_faces": None,
           "bounds_dmax_m": None, "measure": what, "measure_rel": None, "error": None}
    try:
        back = read_step(step_path)
        ref = cq.Shape.importBrep(brep_path)
        row["read"] = True
        row["valid"] = bool(BRepCheck_Analyzer(back.wrapped).IsValid())
        row["bopcheck"] = bool(BRepAlgoAPI_Check(back.wrapped).IsValid())
        row["n_solids"] = len(back.Solids())
        row["n_faces"] = len(back.Faces())
        row["bounds_dmax_m"] = max(abs(a - b) for a, b in zip(_bounds(back), _bounds(ref)))
        vb, vr = measure.gprop(back, what)[0], measure.gprop(ref, what)[0]
        row["measure_rel"] = (vb - vr) / vr
        topo_ok = (row["n_faces"] == len(ref.Faces())
                   and row["n_solids"] == (1 if kind == "solid" else 0))
    except Exception as exc:
        row["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:200])
        return row, [name + ":read"]
    bad = []
    if not row["valid"]:
        bad.append(name + ":valid")
    if not row["bopcheck"]:
        bad.append(name + ":bopcheck")
    if not topo_ok:
        bad.append(name + ":topology")
    if not row["bounds_dmax_m"] <= READBACK_BOUNDS_M:
        bad.append(name + ":bounds")
    if not abs(row["measure_rel"]) <= READBACK_VOL_REL:
        bad.append(name + ":" + what)
    return row, bad


def _stl_readback(stl_path, order, face_rows, planes):
    """The named STL re-read by trimesh (an independent reader): (row, failing fields); nothing raises out of it.

    Per-patch meshes come from force="scene" (one geometry per ASCII solid); the body count from force="mesh"
    after merge_vertices(), because trimesh does not weld across solid blocks (docs/16a §D.12).
    """
    row = {"file": os.path.basename(stl_path), "reader": "trimesh " + trimesh.__version__, "patches": None,
           "per_patch": {}, "welded_vertices": None, "bodies": None, "watertight_bodies": None, "error": None}
    try:
        scene = trimesh.load(stl_path, force="scene")
        mesh = trimesh.load(stl_path, force="mesh")
    except Exception as exc:
        row["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:200])
        return row, ["stl:read"]
    names = list(scene.geometry.keys())
    row["patches"] = names
    bad = []
    if sorted(names) != sorted(order):
        bad.append("stl:patches")
    px = dict((q["name"], q["x"]) for q in planes)
    for tag in order:
        if tag not in scene.geometry:
            continue
        m = scene.geometry[tag]
        ar = m.area_faces
        a = float(ar.sum())
        ta = face_rows[tag]["area_m2"]
        cx = float((m.triangles[:, :, 0].mean(axis=1) * ar).sum() / a) if a > 0 else None
        r = {"area_m2": a, "tag_area_m2": ta, "area_rel": (a - ta) / ta, "centroid_x_m": cx,
             "plane_x_m": px.get(tag), "centroid_dx_m": None}
        if tag in px and cx is not None:
            r["centroid_dx_m"] = cx - px[tag]
        row["per_patch"][tag] = r
        if not abs(r["area_rel"]) <= READBACK_AREA_REL:
            bad.append("stl:area:" + tag)
        if tag in px and not (r["centroid_dx_m"] is not None and abs(r["centroid_dx_m"]) <= READBACK_CX_M):
            bad.append("stl:centroid:" + tag)
    mesh.merge_vertices()
    bodies = mesh.split(only_watertight=False)
    row["welded_vertices"] = int(len(mesh.vertices))
    row["bodies"] = int(len(bodies))
    row["watertight_bodies"] = int(sum(1 for b in bodies if b.is_watertight))
    if not (row["bodies"] == 1 and row["watertight_bodies"] == 1):
        bad.append("stl:bodies")
    return row, bad


def readback(out_dir, face_rows):
    """EXP-READBACK (docs/16a §B.2, §E): every STEP re-read at METRE and the named STL re-read by trimesh.

    `face_rows` is geom.json's tags["fluid_faces"] (tag -> row with area_m2); tags.json gives stl_patches and planes.
    The record lists EVERY failing field in check order; `field` is the first. Nothing raises out of it.
    """
    tags = common.read_json(os.path.join(out_dir, "tags.json"))
    fields = []
    steps = {}
    for step_name, brep_name, kind in READBACK_STEP:
        row, bad = _step_readback(os.path.join(out_dir, step_name), os.path.join(out_dir, brep_name), kind)
        steps[step_name] = row
        fields.extend(bad)
    stl, bad = _stl_readback(os.path.join(out_dir, "fluid_named.stl"), tags["stl_patches"], face_rows,
                             tags["planes"])
    fields.extend(bad)
    return {"status": "refused" if fields else "ok", "reason_id": "EXP-READBACK" if fields else None,
            "field": fields[0] if fields else None, "fields": fields,
            "tolerances": {"bounds_m": READBACK_BOUNDS_M, "volume_rel": READBACK_VOL_REL,
                           "area_rel": READBACK_AREA_REL, "centroid_m": READBACK_CX_M},
            "step": steps, "stl": stl}


def roundtrip(shape, path, kind):
    """GC-5 STEP round trip for one shape: counts equal and the volume / area relative drift."""
    back = read_step(path)
    if kind == "solid":
        vb, vs = solid_volume(shape), solid_volume(back)
        return {"solids": len(back.Solids()), "faces_brep": len(shape.Faces()),
                "faces_step": len(back.Faces()), "volume_rel": (vs - vb) / vb}
    ab, as_ = sum(face_props(f) for f in shape.Faces()), sum(face_props(f) for f in back.Faces())
    return {"faces_brep": len(shape.Faces()), "faces_step": len(back.Faces()),
            "edges_brep": len(shape.Edges()), "edges_step": len(back.Edges()),
            "area_rel": (as_ - ab) / ab}


def determinism(template_path, cases, n, root):
    """n fresh runner builds per case; {name: number of distinct (4 BREP sha, result sha) tuples}."""
    out = {}
    for name, params in cases:
        seen = set()
        for i in range(n):
            od = os.path.join(root, "%s_%d" % (name, i))
            r = runner.run_job(template_path, params, od, entry="build", timeout_s=BUILD_TIMEOUT_S)
            if r["status"] != "ok":
                raise RuntimeError("determinism build %s_%d: %s %s" % (name, i, r["status"], r["message"]))
            key = tuple([common.sha256_file(os.path.join(od, f)) for f in BREP_FILES]
                        + [common.sha256_of(r["value"])])
            seen.add(key)
        out[name] = len(seen)
    return out


def _stl_mutate(path, kind, s=None):
    """Rewrite an ASCII STL in place for the readback proofs: "swap" inlet/outlet names, "drop" outlet, "scale" s."""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().split(chr(10))
    out = []
    skip = False
    names = {"solid inlet": "solid outlet", "solid outlet": "solid inlet",
             "endsolid inlet": "endsolid outlet", "endsolid outlet": "endsolid inlet"}
    for ln in lines:
        if kind == "swap":
            ln = names.get(ln, ln)
        elif kind == "drop":
            if ln == "solid outlet":
                skip = True
            if skip:
                if ln == "endsolid outlet":
                    skip = False
                continue
        elif kind == "scale" and ln.startswith("vertex "):
            ln = "vertex %s %s %s" % tuple(repr(float(t) * s) for t in ln.split()[1:])
        out.append(ln)
    common.atomic_write(path, chr(10).join(out))


def selftest():
    """GC-5 and GC-4 end to end, plus the can-fail proofs; fifteen [ok] lines, then SELFTEST PASS."""
    import time
    t0 = time.monotonic()
    me = os.path.abspath(__file__)
    with tempfile.TemporaryDirectory() as td:
        decl = common.read_json(os.path.join(os.path.dirname(TEMPLATE), "template.json"))
        quantities = [r["quantity"] for r in decl["catalogue"]
                      if r["method"] == "geometry" and r["primitive"] != "k_max_1d"]

        def run_cli(params, out):
            pj = os.path.join(td, os.path.basename(out) + "_params.json")
            common.write_json(pj, params)
            pr = subprocess.run([sys.executable, me, "run", TEMPLATE, pj, out], capture_output=True,
                                text=True, encoding="utf-8", timeout=400,
                                env=dict(os.environ, PYTHONIOENCODING="utf-8"))
            lines = [s for s in pr.stdout.splitlines() if s.strip()]
            return pr.returncode, (json.loads(lines[-1]) if lines else {}), pr.stderr

        # (X1)
        det = determinism(TEMPLATE, [("nominal", NOMINAL), ("corner", CORNER)], 6, td)
        assert det == {"nominal": 1, "corner": 1}, "X1 determinism %r" % (det,)
        print("[ok] determinism: 6 fresh builds each of nominal and corner give one byte-identical set of 4 BREPs")

        # (X2)
        A = {}
        geoms = {}
        for name, params in (("A_nom", NOMINAL), ("A_cor", CORNER)):
            code, line, err = run_cli(params, os.path.join(td, name))
            assert code == 0 and line["status"] == "ok", "X2 run %s: %r %s" % (name, line, err[-400:])
            A[name] = os.path.join(td, name)
            missing = [f for f in BREP_FILES + EXPORT_FILES if not os.path.isfile(os.path.join(A[name], f))]
            assert not missing, "X2 %s missing %r" % (name, missing)
            geoms[name] = common.read_json(os.path.join(A[name], "geom.json"))
        an, ac = geoms["A_nom"], geoms["A_cor"]
        assert list(an) == list(GEOM_KEYS), "X2 geom keys %r" % (list(an),)
        ff = an["tags"].get("fluid_faces", {})
        want_faces = ["inlet", "outlet", "slip_upstream", "wall_contraction", "wall_exit"]
        assert sorted(ff) == sorted(want_faces) and all(ff[t]["area_m2"] > 0 for t in want_faces), (
            "X2 per-tag face areas %r" % (an["tags"],))
        for t, r0 in (("inlet", NOMINAL["D_i"] / 2), ("outlet", NOMINAL["D_i"] / 2 / math.sqrt(NOMINAL["CR"]))):
            a0 = math.pi * r0 * r0
            assert abs(ff[t]["area_m2"] - a0) / a0 <= 1e-9, "X2 %s area %r vs pi r^2 %r" % (t, ff[t]["area_m2"], a0)
        tn, tc = an["stl"]["triangles"], ac["stl"]["triangles"]
        for name in ("A_nom", "A_cor"):
            t = common.read_json(os.path.join(A[name], "tags.json"))
            pr = common.read_json(os.path.join(A[name], "probes.json"))
            assert list(t) == list(TAGS_KEYS), "X2 tags keys %r" % (list(t),)
            assert list(pr) == list(PROBES_KEYS), "X2 probes keys %r" % (list(pr),)
            assert len(pr["rows"]) == 13, "X2 %d probe rows" % (len(pr["rows"]),)
            assert [r["quantity"] for r in pr["rows"]] == quantities, "X2 probe order"
            bad = [r["quantity"] for r in pr["rows"]
                   if r["record"]["status"] != "ok" or schema.errors(r["record"], SCHEMA_MEASURE)]
            assert not bad, "X2 bad probes %r" % (bad,)
        print("[ok] run: 4 BREPs and 8 export files each case, geom/tags/probes keys exact, "
              "13 probes ok and schema-valid, inlet/outlet areas = pi r^2 within 1e-9; triangles nominal %d, corner %d" % (tn, tc))

        # (X3)
        rels = []
        for g in (an, ac):
            rt = g["step_roundtrip"]
            for key in ("fluid", "body"):
                r1 = rt[key]
                assert r1["solids"] == 1 and r1["faces_step"] == r1["faces_brep"], "X3 %s %r" % (key, r1)
                assert abs(r1["volume_rel"]) <= ROUNDTRIP_TOL, "X3 %s volume_rel %r" % (key, r1["volume_rel"])
                rels.append(r1["volume_rel"])
            r3 = rt["meridian"]
            assert r3["faces_brep"] == r3["faces_step"] == 1 and r3["edges_step"] == r3["edges_brep"], (
                "X3 meridian %r" % (r3,))
            assert abs(r3["area_rel"]) <= ROUNDTRIP_TOL, "X3 meridian area_rel %r" % (r3["area_rel"],)
            rels.append(r3["area_rel"])
        print("[ok] GC-5 STEP round trip: rel " + " ".join("%+.2e" % (v,) for v in rels)
              + " (fluid, body, meridian; nominal then corner, limit %g)" % (ROUNDTRIP_TOL,))

        # (X4)
        prows = {r["quantity"]: r["record"]["value"]
                 for r in common.read_json(os.path.join(A["A_nom"], "probes.json"))["rows"]}
        for name in ("A_nom", "A_cor"):
            rep = common.read_json(os.path.join(A[name], "stl_repair.json"))
            o, af = rep["orientation"], rep["after"]
            assert rep["weld"]["tol_rel"] == 0 and rep["before"]["closed"] and af["closed"], "X4 %s" % (name,)
            assert af["open_edges"] == 0 and af["non_manifold_edges"] == 0, "X4 %s edges" % (name,)
            assert o["reoriented_triangles"] == 0 and o["flipped_components"] == 0, "X4 %s orient" % (name,)
            assert rep["n_components"] == 1, "X4 %s components" % (name,)
            assert rep["patches"] == ["inlet", "outlet", "slip_upstream", "wall_contraction", "wall_exit"], (
                "X4 %s patches %r" % (name, rep["patches"]))
        assert prows["watertight"] == 1, "X4 watertight probe %r" % (prows["watertight"],)
        print("[ok] GC-5 watertight: weld 0, closed, 0 open, 0 non-manifold, 0 reoriented, 1 component, "
              "5 named patches, probe 1")

        # (X5)
        for g in (an, ac):
            assert abs(g["stl"]["volume_rel"]) <= STL_VOL_TOL, "X5 stl volume_rel %r" % (g["stl"]["volume_rel"],)
        print("[ok] GC-5 STL volume rel nominal %+.2e corner %+.2e (limit %g)"
              % (an["stl"]["volume_rel"], ac["stl"]["volume_rel"], STL_VOL_TOL))

        # (X6)
        for name, params in (("A_nom", NOMINAL), ("A_cor", CORNER)):
            assert step_length_unit(os.path.join(A[name], "fluid.step")) == "METRE", "X6 %s unit" % (name,)
            span = (params["Lu_over_Di"] * params["D_i"] + params["L_over_Di"] * params["D_i"]
                    + params["Lx_over_De"] * params["D_i"] / math.sqrt(params["CR"]))
            gm = an if name == "A_nom" else ac
            assert abs(gm["gmsh_import"]["x_span_m"] - span) <= SPAN_TOL, (
                "X6 %s span %r vs %r" % (name, gm["gmsh_import"]["x_span_m"], span))
            assert gm["gmsh_import"]["n_volumes"] == 1, "X6 %s n_volumes" % (name,)
        assert prows["axis"] == 1 and prows["units"] == 1, "X6 probes %r" % (prows,)
        print("[ok] GC-5 units: METRE STEP, gmsh x-span nominal %.12f corner %.12f m, 1 volume, "
              "axis and units probes 1" % (an["gmsh_import"]["x_span_m"], ac["gmsh_import"]["x_span_m"]))

        # (X7)
        B = {}
        for name, params in (("B_nom", NOMINAL), ("B_cor", CORNER)):
            code, line, err = run_cli(params, os.path.join(td, name))
            assert code == 0 and line["status"] == "ok", "X7 run %s: %r %s" % (name, line, err[-400:])
            B[name] = os.path.join(td, name)
        for a, b in (("A_nom", "B_nom"), ("A_cor", "B_cor")):
            for f in BREP_FILES + EXPORT_FILES:
                assert common.sha256_file(os.path.join(td, a, f)) == common.sha256_file(os.path.join(td, b, f)), (
                    "X7 %s differs between %s and %s" % (f, a, b))
        print("[ok] GC-4: two fresh processes give byte-identical probes.json and fluid_named.stl "
              "and all 12 files, both cases")

        # (X8)
        rp = lambda n: os.path.join(td, n)
        box = cq.Solid.makeBox(0.01, 0.02, 0.03)
        write_named_stl(box, {"box": list(range(6))}, ["box"], rp("box0.stl"), 1e-3, STL_ANG_RAD,
                        flip_reversed=False)
        rep0 = stl_report(rp("box0.stl"), rp("box0.json"))
        assert rep0["before"]["non_manifold_edges"] > 0, "X8a unflipped box %r" % (rep0["before"],)
        assert measure.watertight(rep0)["value"] == 0, "X8a unflipped box not caught"
        box1 = cq.Solid.makeBox(0.01, 0.02, 0.03)
        write_named_stl(box1, {"box": list(range(6))}, ["box"], rp("box1.stl"), 1e-3, STL_ANG_RAD)
        rep1 = stl_report(rp("box1.stl"), rp("box1.json"))
        assert measure.watertight(rep1)["value"] == 1, "X8a flipped box %r" % (measure.watertight(rep1),)
        tags_nom = common.read_json(os.path.join(A["A_nom"], "tags.json"))
        order_d = [t for t in tags_nom["stl_patches"] if t != "outlet"]
        write_named_stl(cq.Shape.importBrep(os.path.join(A["A_nom"], "fluid.brep")), tags_nom["face_tags"],
                        order_d, rp("no_outlet.stl"), an["stl"]["lin_deflection_m"], STL_ANG_RAD)
        rep2 = stl_report(rp("no_outlet.stl"), rp("no_outlet.json"))
        assert rep2["after"]["open_edges"] > 0 and measure.watertight(rep2)["value"] == 0, (
            "X8b dropped outlet %r" % (rep2["after"],))
        sd = _stl_dict(an["stl"]["lin_deflection_m"], {"triangles": 0, "per_tag": {}}, rep2, 1.0)
        assert sd["volume_m3"] is None and sd["volume_rel"] is None, "X8b open STL volume %r" % (sd,)
        fluid_nom = cq.Shape.importBrep(os.path.join(A["A_nom"], "fluid.brep"))
        write_step(fluid_nom, rp("milli.step"), unit="MM")
        assert step_length_unit(rp("milli.step")) == "MILLI.METRE", "X8c MILLI not declared"
        span0 = an["gmsh_import"]["x_span_m"]
        s_mm = gmsh_span(rp("milli.step"))
        assert abs(s_mm["x_span_m"] - 1e-3 * span0) <= 1e-12, (
            "X8c mm span %r vs %r" % (s_mm["x_span_m"], 1e-3 * span0))
        gmm = dict(an)
        gmm["step_length_unit"] = "MILLI.METRE"
        gmm["gmsh_import"] = s_mm
        assert measure.units_m(gmm)["value"] == 0, "X8c MILLI geom not caught"
        rot = fluid_nom.rotate(cq.Vector(0, 0, 0), cq.Vector(0, 1, 0), 90)
        assert measure.axis_x(rot)["value"] == 0, "X8d rotated axis not caught"
        print("[ok] can fail: unflipped box, dropped outlet, MILLI STEP and rotated axis are each caught")

        # (X9)
        bad_params = dict(NOMINAL)
        bad_params["L_over_Di"] = 0.4
        out_r = os.path.join(td, "R")
        code, line, err = run_cli(bad_params, out_r)
        assert code == 1 and line["status"] == "refused" and line["rule"] == "PRF-BOX", (
            "X9 %r %r %s" % (code, line, err[-200:]))
        leftovers = [f for f in EXPORT_FILES if os.path.exists(os.path.join(out_r, f))]
        assert not leftovers and not [f for f in os.listdir(out_r) if f.endswith(".brep")], (
            "X9 leftovers %r" % (leftovers,))
        print("[ok] refusal: L_over_Di 0.4 exits 1 with PRF-BOX and writes no export file")

        # (X10)
        assert "cad_child_job" not in sys.modules, "X10 cad_child_job imported"
        tdir = os.path.dirname(TEMPLATE)
        loaded = [n for n, m2 in sys.modules.items()
                  if getattr(m2, "__file__", None) and os.path.normcase(os.path.dirname(os.path.abspath(m2.__file__)))
                  == os.path.normcase(tdir)]
        assert not loaded, "X10 template modules %r" % (loaded,)
        print("[ok] isolation: this process never imported template.py")

        # (X11)
        from scipy.integrate import quad
        from OCP.BRepAdaptor import BRepAdaptor_Curve
        from OCP.gp import gp_Pnt, gp_Vec
        from OCP.GProp import GProp_GProps
        from OCP.BRepGProp import BRepGProp
        sys.path.insert(0, os.path.join(HERE, "fixtures", "trap"))
        import trap
        body_t = trap.build()
        assert solid_volume(body_t) == measure.gprop(body_t, "volume")[0], "X11 solid helper not routed"
        assert face_props(trap.wetted_face(body_t)) == measure.gprop(trap.wetted_face(body_t), "area")[0], (
            "X11 face helper not routed")
        saved_eps = measure.GPROP_EPS
        raised = False
        try:
            measure.GPROP_EPS = 1e-6
            try:
                solid_volume(body_t)
            except ValueError as exc:
                raised = str(exc).startswith("MEAS-GPROP")
        finally:
            measure.GPROP_EPS = saved_eps
        assert raised, "X11 MEAS-GPROP not raised at eps 1e-6"
        moves = {}
        len_def = len_ada = None
        for name in ("A_nom", "A_cor"):
            g = common.read_json(os.path.join(A[name], "geom.json"))
            fl = cq.Shape.importBrep(os.path.join(A[name], "fluid.brep"))
            mer = cq.Shape.importBrep(os.path.join(A[name], "meridian.brep"))
            tg = common.read_json(os.path.join(A[name], "tags.json"))
            move = 0.0
            for tag, row in g["tags"]["fluid_faces"].items():
                assert row["gprop_eps"] == 1e-12 and row["gprop_est_rel"] <= 1e-8, (
                    "X11 %s %s row %r" % (name, tag, row))
                default = 0.0
                for i in row["index"]:
                    pd = GProp_GProps()
                    BRepGProp.SurfaceProperties_s(fl.Faces()[i].wrapped, pd)
                    default += pd.Mass()
                move = max(move, abs(default - row["area_m2"]) / row["area_m2"])
            moves[name] = move
            assert move <= 1e-8, "X11 %s tag area move %.3e > 1e-8" % (name, move)
            for key in ("meridian_edges", "wall_edges"):
                for tag, row in g["tags"][key].items():
                    assert row["gprop_eps"] == 1e-12 and row["gprop_est_rel"] <= 1e-8, (
                        "X11 %s %s %s row %r" % (name, key, tag, row))
            if name == "A_cor":
                for i in tg["meridian_edges"]["wall_contraction"]:
                    e = mer.Edges()[i]
                    ad = BRepAdaptor_Curve(e.wrapped)

                    def sp(u, ad=ad):
                        p = gp_Pnt()
                        v = gp_Vec()
                        ad.D1(u, p, v)
                        return v.Magnitude()
                    q = quad(sp, ad.FirstParameter(), ad.LastParameter(), epsabs=0, epsrel=1e-13, limit=200)[0]
                    la = edge_length(e)
                    assert abs(la - q) / q <= 1e-11, "X11 corner Bezier adaptive rel %+.3e > 1e-11" % ((la - q) / q,)
                    pl = GProp_GProps()
                    BRepGProp.LinearProperties_s(e.wrapped, pl)
                    assert abs(pl.Mass() - q) / q > 1e-8, "X11 corner Bezier default does not miss: %r" % (i,)
                    len_def, len_ada = (pl.Mass() - q) / q, (la - q) / q
        assert len_def is not None and len_ada is not None, "X11 corner lengths not measured"
        print("[ok] adaptive GProp: trap helpers equal measure.gprop, MEAS-GPROP raised at eps 1e-6; "
              "tag areas move nominal %.2e corner %.2e (<= 1e-8); corner Bezier length default rel %+.2e, "
              "adaptive %+.2e" % (moves["A_nom"], moves["A_cor"], len_def, len_ada))

        # (X12)
        bmax = mmax = dxmax = 0.0
        arels = []
        for name, g in (("A_nom", an), ("A_cor", ac)):
            rb = g["readback"]
            assert rb["status"] == "ok" and rb["reason_id"] is None and rb["field"] is None and rb["fields"] == [], (
                "X12 %s readback %r" % (name, rb["fields"],))
            for sn, sr in rb["step"].items():
                assert sr["read"] and sr["valid"] and sr["bopcheck"], "X12 %s %s %r" % (name, sn, sr["error"],)
            bmax = max(bmax, max(sr["bounds_dmax_m"] for sr in rb["step"].values()))
            mmax = max(mmax, max(abs(sr["measure_rel"]) for sr in rb["step"].values()))
            assert len(rb["stl"]["per_patch"]) == 5, "X12 %s patches %d" % (name, len(rb["stl"]["per_patch"]),)
            for tg, pr in rb["stl"]["per_patch"].items():
                assert abs(pr["area_rel"]) <= READBACK_AREA_REL, "X12 %s %s area %r" % (name, tg, pr["area_rel"],)
                arels.append(pr["area_rel"])
            for tg in ("inlet", "outlet"):
                dx = rb["stl"]["per_patch"][tg]["centroid_dx_m"]
                assert dx is not None and abs(dx) <= READBACK_CX_M, "X12 %s %s dx %r" % (name, tg, dx,)
                dxmax = max(dxmax, abs(dx))
            assert rb["stl"]["bodies"] == 1 and rb["stl"]["watertight_bodies"] == 1, (
                "X12 %s bodies %r" % (name, (rb["stl"]["bodies"], rb["stl"]["watertight_bodies"]),))
            assert readback(A[name], g["tags"]["fluid_faces"]) == rb, "X12 %s re-run differs" % (name,)
        print("[ok] readback: nominal and corner pass: STEP bounds max %.1e m, volume/area max |rel| %.1e, "
              "BRepCheck and BOPCheck true; STL patch area rel %+.2e..%+.2e, planar centroid max |dx| %.1e m, "
              "1 welded watertight body" % (bmax, mmax, min(arels), max(arels), dxmax))

        # (X13)
        mids = ("control", "milli", "swap", "drop", "s0.7", "s1.3", "s1.001", "other")
        first_also = {"milli": ("fluid.step:bounds", ["fluid.step:volume"]),
                      "swap": ("stl:area:inlet", ["stl:centroid:inlet", "stl:area:outlet", "stl:centroid:outlet"]),
                      "drop": ("stl:patches", ["stl:bodies"]),
                      "s0.7": ("stl:area:inlet", ["stl:centroid:outlet"]),
                      "s1.3": ("stl:area:inlet", ["stl:centroid:outlet"]),
                      "s1.001": ("stl:area:inlet", ["stl:centroid:inlet"]),
                      "other": ("fluid.step:topology", ["fluid.step:bounds", "fluid.step:volume"])}
        for mid in mids:
            d = rp("MX_" + mid.replace(".", "_"))
            shutil.copytree(A["A_nom"], d)
            if mid == "milli":
                write_step(cq.Shape.importBrep(os.path.join(d, "fluid.brep")), os.path.join(d, "fluid.step"),
                           unit="MM")
            elif mid in ("swap", "drop"):
                _stl_mutate(os.path.join(d, "fluid_named.stl"), mid)
            elif mid.startswith("s"):
                _stl_mutate(os.path.join(d, "fluid_named.stl"), "scale", float(mid[1:]))
            elif mid == "other":
                shutil.copyfile(os.path.join(A["A_cor"], "fluid.step"), os.path.join(d, "fluid.step"))
            r = readback(d, an["tags"]["fluid_faces"])
            if mid == "control":
                assert r["status"] == "ok" and r["fields"] == [], "X13 control %r" % (r["fields"],)
                continue
            first, also = first_also[mid]
            assert r["status"] == "refused" and r["reason_id"] == "EXP-READBACK", (
                "X13 %s status %r" % (mid, r["status"],))
            assert r["field"] == first, "X13 %s field %r != %r" % (mid, r["field"], first)
            for fld in also:
                assert fld in r["fields"], "X13 %s missing %r in %r" % (mid, fld, r["fields"])
        print("[ok] readback refuses 7 of 7 mutants EXP-READBACK by field: "
              + "; ".join("%s %s" % (m, first_also[m][0]) for m in mids[1:]) + "; the unmutated copy passes")

        # (X14)
        mod = sys.modules[__name__]
        orig_wns = mod.write_named_stl

        def scaled(shape, face_tags, order, path, lin, ang, flip_reversed=True):
            out = orig_wns(shape, face_tags, order, path, lin, ang, flip_reversed)
            _stl_mutate(path, "scale", 1.001)
            return out

        mod.write_named_stl = scaled
        try:
            res = run_pipeline(TEMPLATE, dict(NOMINAL), rp("RB"))
        finally:
            mod.write_named_stl = orig_wns
        assert res["status"] == "refused" and res["rule"] == "EXP-READBACK", "X14 status %r" % (res["status"],)
        assert res["message"].startswith("EXP-READBACK: stl:area:inlet"), "X14 msg %r" % (res["message"],)
        gd = common.read_json(os.path.join(rp("RB"), "geom.json"))
        assert gd["readback"]["status"] == "refused" and gd["readback"]["field"] == "stl:area:inlet", (
            "X14 geom %r" % (gd["readback"]["field"],))
        assert gd["readback"] == res["geom"]["readback"], "X14 geom readback differs from the result"
        print("[ok] run_pipeline refuses EXP-READBACK (field stl:area:inlet) on an STL scaled 1.001 in the "
              "export, and geom.json records the refused readback")

        # (X15)
        lin_n = an["stl"]["lin_deflection_m"]
        with open(os.path.join(A["A_nom"], "fluid_named.stl"), "rb") as f:
            ref_bytes = f.read()
        for label, fl, fa in (("coarse", 10.0, 4.0), ("fine", 0.3, 0.5)):
            sh = cq.Shape.importBrep(os.path.join(A["A_nom"], "fluid.brep"))
            BRepMesh_IncrementalMesh(sh.wrapped, fl * lin_n, False, fa * STL_ANG_RAD, False).Perform()
            pth = rp("pre_%s.stl" % label)
            write_named_stl(sh, tags_nom["face_tags"], tags_nom["stl_patches"], pth, lin_n, STL_ANG_RAD)
            with open(pth, "rb") as f:
                got = f.read()
            assert got == ref_bytes, "X15 %s: %d bytes vs %d" % (label, len(got), len(ref_bytes))
        print("[ok] a shape meshed first coarsely (10x) or finely (0.3x) gives an STL byte-identical to the "
              "export (%d bytes)" % len(ref_bytes))

    print("selftest wall %.1f s" % (time.monotonic() - t0,))
    print("SELFTEST PASS")
    return 0


def export_build(out_dir, value, template_path):
    """The S3 export of one ok template result: 3 STEP, named STL, stl_repair report, tags, probes, geom.

    geom.json records the readback; a refused one is refused by run_pipeline, never raised here
    (mutate.py's M10 exports through this function).
    """
    p = lambda *a: os.path.join(out_dir, *a)
    fluid = cq.Shape.importBrep(p("fluid.brep"))
    body = cq.Shape.importBrep(p("body.brep"))
    meridian = cq.Shape.importBrep(p("meridian.brep"))
    wall_m = cq.Shape.importBrep(p("wall_meridian.brep"))
    for name, shp in (("fluid.step", fluid), ("body.step", body), ("meridian.step", meridian)):
        write_step(shp, p(name))
    lin = STL_LIN_REL * value["derived"]["D_e"]
    decl = common.read_json(os.path.join(os.path.dirname(template_path), "template.json"))
    order = [t["name"] for t in decl["tags"] if t["kind"] == "face" and t["name"] in value["face_tags"]]
    stl = write_named_stl(cq.Shape.importBrep(p("fluid.brep")), value["face_tags"], order,
                          p("fluid_named.stl"), lin, STL_ANG_RAD)
    report = stl_report(p("fluid_named.stl"), p("stl_repair.json"))
    span = gmsh_span(p("fluid.step"))
    ext = measure.extent_along_axis(fluid)
    geom_units = {"units": "m", "scale": 1, "axis": "+x", "step_length_unit": step_length_unit(p("fluid.step")),
                  "x_span_m": ext["value"], "gmsh_import": span}
    planes = dict((q["name"], q) for q in value["planes"])
    rows = measure_catalogue(decl["catalogue"], {"fluid": fluid, "body": body}, planes,
                             meridian, wall_m, value, report, geom_units)
    common.write_json(p("tags.json"), {"version": 1, "template_id": value["template_id"],
                                       "face_tags": value["face_tags"],
                                       "meridian_edges": value["meridian_edges"],
                                       "wall_edges": value["wall_edges"],
                                       "planes": value["planes"], "stl_patches": order})
    common.write_json(p("probes.json"), {"version": 1, "template_id": value["template_id"],
                                         "params_sha": common.sha256_of(value["params"]), "rows": rows})
    geom = _geom_dict(out_dir, value, template_path, lin, span, ext, geom_units, stl, report)
    common.write_json(p("geom.json"), geom)
    return geom


def run_pipeline(template_path, params, out_dir, timeout_s=BUILD_TIMEOUT_S):
    """runner build then export; ok / refused / error with the rule, the export files only when ok.

    A refused export readback is refused EXP-READBACK here, not in export_build (docs/16a line 80 says
    export_build; the tree wins because mutate.py's M10 calls export_build and counts an exception as an
    error).
    """
    r = runner.run_job(template_path, params, out_dir, entry="build", timeout_s=timeout_s)
    if r["status"] != "ok":
        return {"status": "error", "rule": r["rule"], "message": r["message"], "geom": None}
    if r["value"]["status"] == "refused":
        return {"status": "refused", "rule": r["value"]["rule"], "message": r["value"]["detail"], "geom": None}
    geom = export_build(out_dir, r["value"], template_path)
    rb = geom["readback"]
    if rb["status"] != "ok":
        return {"status": "refused", "rule": "EXP-READBACK",
                "message": "EXP-READBACK: " + ", ".join(rb["fields"]), "geom": geom}
    return {"status": "ok", "rule": None, "message": "", "geom": geom}


def _wt_dict(report):
    """The geom.json watertight summary of one stl_repair report."""
    return {"weld_tol_rel": report["weld"]["tol_rel"], "before_closed": bool(report["before"]["closed"]),
            "after_closed": bool(report["after"]["closed"]), "open_edges": report["after"]["open_edges"],
            "non_manifold_edges": report["after"]["non_manifold_edges"],
            "reoriented_triangles": report["orientation"]["reoriented_triangles"],
            "flipped_components": report["orientation"]["flipped_components"],
            "holes_filled": report["holes"]["filled"], "degenerate_dropped": report["degenerate_dropped"],
            "n_components": report["n_components"], "patches": report["patches"]}


def _stl_dict(lin, stl, report, brep_vol):
    """The geom.json stl block; an STL stl_repair cannot close has no volume, so volume_rel is None."""
    vol = report["after"]["volume"]
    return {"file": "fluid_named.stl", "lin_deflection_m": lin, "ang_deflection_rad": STL_ANG_RAD,
            "triangles": stl["triangles"], "per_tag": stl["per_tag"], "volume_m3": vol,
            "brep_volume_m3": brep_vol, "volume_rel": None if vol is None else (vol - brep_vol) / brep_vol}


def _geom_dict(out_dir, value, template_path, lin, span, ext, geom_units, stl, report):
    """The geom dict: exactly GEOM_KEYS in order, no timestamp, wall time, pid or absolute path."""
    p = lambda *a: os.path.join(out_dir, *a)
    brep_vol = solid_volume(cq.Shape.importBrep(p("fluid.brep")))
    files = {}
    for name in list(BREP_FILES) + [f for f in EXPORT_FILES if f != "geom.json"]:
        files[name] = common.sha256_file(p(name))
    decl = os.path.join(os.path.dirname(template_path), "template.json")
    tags = tag_table(cq.Shape.importBrep(p("fluid.brep")), cq.Shape.importBrep(p("meridian.brep")),
                     cq.Shape.importBrep(p("wall_meridian.brep")), value)
    return {"version": 1, "template_id": value["template_id"],
            "template_sha": common.sha256_file(template_path), "declaration_sha": common.sha256_file(decl),
            "params": value["params"], "params_sha": common.sha256_of(value["params"]),
            "units": "m", "scale": 1, "axis": "+x", "step_length_unit": geom_units["step_length_unit"],
            "x_span_m": ext["value"],
            "x_span_expected_m": value["derived"]["x_outlet"] - value["derived"]["x_inlet"],
            "gmsh_import": span,
            "stl": _stl_dict(lin, stl, report, brep_vol),
            "watertight": _wt_dict(report),
            "step_roundtrip": {"fluid": roundtrip(cq.Shape.importBrep(p("fluid.brep")), p("fluid.step"), "solid"),
                               "body": roundtrip(cq.Shape.importBrep(p("body.brep")), p("body.step"), "solid"),
                               "meridian": roundtrip(cq.Shape.importBrep(p("meridian.brep")),
                                                     p("meridian.step"), "face")},
            "readback": readback(out_dir, tags["fluid_faces"]),
            "tags": tags,
            "files": files, "env": common.env_fingerprint()}


def measure_catalogue(catalogue, shapes, planes, meridian, wall_m, value, report, geom_units):
    """One {quantity, primitive, where, record} row per catalogue row, in order, each schema-valid."""
    def edges_of(tag):
        if tag in value["meridian_edges"]:
            return [meridian.Edges()[i] for i in value["meridian_edges"][tag]]
        return [wall_m.Edges()[i] for i in value["wall_edges"][tag]]

    def one(row):
        w, prim = row["where"], row["primitive"]
        if prim == "diameter_at_plane":
            return measure.diameter_at_plane(shapes["fluid"], planes[w[0]])
        if prim == "area_ratio":
            return measure.area_ratio(shapes["fluid"], planes[w[0]], planes[w[1]])
        if prim == "extent_along_axis":
            return measure.extent_along_axis(shapes[w[0]])
        if prim == "plane_distance":
            return measure.plane_distance(planes[w[0]], planes[w[1]])
        if prim == "meridian_min_wall":
            return measure.meridian_min_wall(edges_of(w[0]), edges_of(w[1]))
        if prim == "slope_max":
            return measure.slope_max(edges_of(w[0]))
        if prim == "curvature_radius_min":
            return measure.curvature_radius_min(edges_of(w[0]))
        if prim in ("n_solids", "valid", "axis_x"):
            return measure.PRIMITIVES[prim](shapes[w[0]])
        if prim == "watertight":
            return measure.watertight(report)
        if prim == "units_m":
            return measure.units_m(geom_units)
        return measure.run(prim)

    rows = []
    for row in catalogue:
        if row.get("method") != "geometry":
            continue                      # performance rows (docs/16 §E.2): post.py measures them
        if row["primitive"] == "k_max_1d":
            continue                      # computed from the parameters at the check's Re by optimise_cad._probes_row, never measured on the BREP
        rec = one(row)
        errs = schema.errors(rec, SCHEMA_MEASURE)
        if errs:
            raise RuntimeError("probe %s: record fails cad-measure/1: %r" % (row["quantity"], errs[:1]))
        rows.append({"quantity": row["quantity"], "primitive": row["primitive"],
                     "where": row["where"], "record": rec})
    return rows


def main(argv):
    if argv == ["--selftest"]:
        try:
            return selftest()
        except Exception:
            import traceback
            traceback.print_exc()
            return 1
    if len(argv) == 4 and argv[0] == "run":
        params = common.read_json(argv[2])
        res = run_pipeline(argv[1], params, os.path.abspath(argv[3]))
        print(common.canonical_json({"status": res["status"], "rule": res["rule"], "message": res["message"]}))
        return 0 if res["status"] == "ok" else 1
    if len(argv) == 3 and argv[0] == "determinism":
        with tempfile.TemporaryDirectory() as td:
            res = determinism(argv[1], [("nominal", NOMINAL), ("corner", CORNER)], int(argv[2]), td)
        code = 0
        for name, k in (("nominal", res["nominal"]), ("corner", res["corner"])):
            print("%s: %d distinct of %s" % (name, k, int(argv[2])))
            if k != 1:
                code = 1
        return code
    if len(argv) == 3 and argv[0] == "gmsh-span":
        return gmsh_span_main(argv[1], argv[2])
    sys.stderr.write(USAGE + chr(10))
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
