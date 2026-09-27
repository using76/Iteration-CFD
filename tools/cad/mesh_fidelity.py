#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""mesh_fidelity.py - AMG-9 (docs/16a §B.2, §G.1 AMG-9): the wedge polyMesh boundary that wedge_mesh.py (CAD-11)
writes, judged per patch against the BRep it was meshed from, at scale 1.

Every boundary vertex must lie within TOL["vertex_m"] of its patch's tagged BRep faces (the points file is written
with 17 significant digits, so the 1e-9 m gate is the binding one), and every boundary face centre within the
analytic sag: R (1 - cos(theta/2)) for a revolved patch, plus the meridian chord sag h^2 / (8 rho_min) from the
face's longest meridian edge h and the tag's minimum curvature radius, plus TOL["centre_extra_m"]. The scale is
asserted first: geom.json declares metres at scale 1, the mesh's x span and largest radius equal the meridian's
within TOL["scale_rel"], and the angle between the two wedge sides equals theta within TOL["theta_rad"] (the
precondition CAD-14's 2 pi / theta scaling rests on). The idea of seeded bidirectional sampling with unit_scale 1
is Amagine3D's shape_consistency.py compare_meshes (Apache-2.0, e608dc6), reimplemented as an exact projection
onto the tagged faces; no code was copied.

Usage:
  python mesh_fidelity.py --selftest
  python mesh_fidelity.py check GEOM_DIR CASE_DIR OUT_JSON
"""
import math
import os
import re
import shutil
import sys
import tempfile

import numpy as np
import cadquery as cq
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex, BRepBuilderAPI_Transform
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.BRepTopAdaptor import BRepTopAdaptor_FClass2d
from OCP.ShapeAnalysis import ShapeAnalysis_Surface
from OCP.TopAbs import TopAbs_OUT
from OCP.TopoDS import TopoDS
from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common
import measure
sys.path.insert(0, os.path.join(common.REPO, "tools", "mesh"))
import polymesh_write
import regions_check

VERSION = "cad-mesh-fidelity/1"
TOL = {"vertex_m": 1e-9, "centre_extra_m": 1e-9, "scale_rel": 1e-9, "theta_rad": 1e-9}
THETA_DEG = 5.0               # wedge_mesh.RECIPE["theta_deg"]; the selftest proves the two agree
PROJ_TOL = 1e-12              # m: ShapeAnalysis_Surface.ValueOfUV precision
UV_TOL = 1e-9                 # BRepTopAdaptor_FClass2d tolerance in (u, v)
AXIS_R = 1e-12                # m: a vertex this close to the axis has no azimuth
N_EDGE_SAMPLES = 1024         # per meridian edge, for the reference extents
REVOLVED = (("inlet", ("inlet",)), ("outlet", ("outlet",)),
            ("wall_nozzle", ("wall_contraction", "wall_exit")), ("slip_upstream", ("slip_upstream",)))
SIDES = (("wedge_front", -0.5), ("wedge_back", 0.5))   # the meridian rotated about +x by this fraction of theta
PATCH_ORDER = ("inlet", "outlet", "wall_nozzle", "slip_upstream", "wedge_front", "wedge_back")
GEOM_FILES = ("geom.json", "tags.json", "fluid.brep", "meridian.brep")
MESH_FILES = ("boundary", "faces", "neighbour", "owner", "points")
REFUSAL_IDS = ("MFID-BIND", "MFID-PATCH", "MFID-SCALE", "MFID-VERTEX", "MFID-CENTRE")
DECLARED = {"units": "m", "scale": 1, "axis": "+x", "step_length_unit": "METRE"}
REPORT_KEYS = ("version", "status", "reason_id", "fields", "tolerances", "theta_deg", "inputs", "scale", "patches")
SCALE_KEYS = ("declared", "ref_x_min_m", "ref_x_max_m", "ref_r_max_m", "mesh_x_min_m", "mesh_x_max_m",
              "mesh_r_max_m", "s_x", "s_r", "theta_mesh_rad", "theta_rad", "pass")
PATCH_KEYS = ("tags", "type", "n_faces", "n_vertices", "rho_min_m", "vertex_max_m", "vertex_p99_m",
              "vertex_argmax_m", "n_vertex_over", "n_vertex_outside", "centre_max_m", "centre_p99_m",
              "centre_ratio_max", "n_centre_over", "pass")
USAGE = ("usage: python mesh_fidelity.py --selftest" + chr(10)
         + "       python mesh_fidelity.py check GEOM_DIR CASE_DIR OUT_JSON")


class FaceSet:
    """A patch's tagged BRep faces, each with a surface projector and a (u, v) classifier."""

    def __init__(self, faces):
        self.items = [(ShapeAnalysis_Surface(BRep_Tool.Surface_s(f)), BRepTopAdaptor_FClass2d(f, UV_TOL), f)
                      for f in faces]

    def distance(self, p):
        """(distance, inside): the smallest projection gap over the faces whose foot classifies IN or ON; when no
        foot lies in a face, the exact BRepExtrema distance to the nearest face (inside False). Either value is the
        distance to a real point of the face set, so it never under-reports."""
        pt = gp_Pnt(float(p[0]), float(p[1]), float(p[2]))
        best = math.inf
        for sa, cl, f in self.items:
            uv = sa.ValueOfUV(pt, PROJ_TOL)
            if cl.Perform(uv) != TopAbs_OUT:
                best = min(best, sa.Gap())
        if best < math.inf:
            return best, True
        v = BRepBuilderAPI_MakeVertex(pt).Vertex()
        return min(BRepExtrema_DistShapeShape(v, f).Value() for sa, cl, f in self.items), False


def _rotated(face, angle):
    """A copy of the face rotated about +x through the origin by angle (rad)."""
    t = gp_Trsf()
    t.SetRotation(gp_Ax1(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(1.0, 0.0, 0.0)), angle)
    return TopoDS.Face_s(BRepBuilderAPI_Transform(face, t, True).Shape())


def _rho_min(edges):
    """The smallest curvature radius of the edges (measure.curvature_radius_min); None for straight edges."""
    rec = measure.curvature_radius_min(edges)
    if rec["status"] == "ok":
        return rec["value"]
    if rec["reason_id"] == "MEAS-NOCURV":
        return None
    raise RuntimeError("curvature_radius_min: %s %s" % (rec["reason_id"], rec["detail"]))


def load_brep(geom_dir, theta):
    """{patch: (tags, FaceSet, rho_min or None)} for the six patches, plus the meridian shape."""
    tags = common.read_json(os.path.join(geom_dir, "tags.json"))
    fluid = cq.Shape.importBrep(os.path.join(geom_dir, "fluid.brep"))
    meridian = cq.Shape.importBrep(os.path.join(geom_dir, "meridian.brep"))
    ff, me, medges = fluid.Faces(), tags["meridian_edges"], meridian.Edges()
    out = {}
    for name, names in REVOLVED:
        faces = [ff[i].wrapped for t in names for i in tags["face_tags"][t]]
        out[name] = (list(names), FaceSet(faces), _rho_min([medges[i] for t in names for i in me[t]]))
    mface = meridian.Faces()[0].wrapped
    rho_all = _rho_min(medges)
    for name, frac in SIDES:
        out[name] = (["meridian"], FaceSet([_rotated(mface, frac * theta)]), rho_all)
    return out, meridian


def ref_extents(meridian):
    """(x_min, x_max, r_max) of the meridian: every edge sampled at N_EDGE_SAMPLES + 1 parameters, ends included."""
    xs, rs = [], []
    for e in meridian.Edges():
        c = BRepAdaptor_Curve(e.wrapped)
        t0, t1 = c.FirstParameter(), c.LastParameter()
        for i in range(N_EDGE_SAMPLES + 1):
            p = c.Value(t0 + (t1 - t0) * i / N_EDGE_SAMPLES)
            xs.append(p.X())
            rs.append(math.hypot(p.Y(), p.Z()))
    return min(xs), max(xs), max(rs)


def _patch_faces(pm, name):
    p = [q for q in pm["patches"] if q["name"] == name][0]
    return range(p["startFace"], p["startFace"] + p["nFaces"]), p["type"]


def face_table(points, faces, ids, revolved):
    """(Sf, Cf, r_max, h) arrays over the faces ids: Sf and Cf by the fan about the vertex average exactly as
    regions_check.face_geometry, vectorised by vertex count (the selftest checks the two agree); r_max the largest
    vertex radius; h the longest edge in one meridian half-plane (every edge when not revolved)."""
    m = len(ids)
    sf, cf, r_max, h = np.zeros((m, 3)), np.zeros((m, 3)), np.zeros(m), np.zeros(m)
    by_n = {}
    for k, f in enumerate(ids):
        by_n.setdefault(len(faces[f]), []).append(k)
    for n, ks in by_n.items():
        if n == 0:
            continue
        P = points[np.array([faces[ids[k]] for k in ks], dtype=np.int64)]
        xavg = P.mean(axis=1)
        rr = np.hypot(P[:, :, 1], P[:, :, 2])
        phi = np.arctan2(P[:, :, 2], P[:, :, 1])
        r_max[ks] = rr.max(axis=1)
        if n < 3:
            cf[ks] = xavg
            continue
        s, c, area, hh = np.zeros((len(ks), 3)), np.zeros((len(ks), 3)), np.zeros(len(ks)), np.zeros(len(ks))
        for i in range(n):
            j = (i + 1) % n
            a, b = P[:, i], P[:, j]
            tn = np.cross(a - xavg, b - xavg)
            ta = np.sqrt(np.einsum("ij,ij->i", tn, tn)) * 0.5
            s += tn * 0.5
            c += (xavg + a + b) / 3.0 * ta[:, None]
            area += ta
            same = ((rr[:, i] <= AXIS_R) | (rr[:, j] <= AXIS_R)
                    | (np.abs(phi[:, i] - phi[:, j]) <= TOL["theta_rad"]))
            if not revolved:
                same = np.ones(len(ks), dtype=bool)
            hh = np.maximum(hh, np.where(same, np.linalg.norm(a - b, axis=1), 0.0))
        ok = area > regions_check.SMALL
        sf[ks] = s
        cf[ks] = np.where(ok[:, None], c / np.where(ok, area, 1.0)[:, None], xavg)
        h[ks] = hh
    return sf, cf, r_max, h


def _mean_normal(pm, name):
    rng, _ = _patch_faces(pm, name)
    s = face_table(pm["points"], pm["faces"], rng, False)[0].sum(axis=0)
    return s / np.linalg.norm(s)


def scale_row(pm, geom, meridian, theta):
    """The scale precondition's row and its failure fields (MFID-SCALE:declared, :s_x, :s_r, :theta)."""
    declared = all(geom.get(k) == v for k, v in DECLARED.items())
    rx0, rx1, rr = ref_extents(meridian)
    vs = sorted(set(v for q in pm["patches"] for f in range(q["startFace"], q["startFace"] + q["nFaces"])
                    for v in pm["faces"][f]))
    pts = pm["points"][vs]
    mx0, mx1 = float(pts[:, 0].min()), float(pts[:, 0].max())
    mr = float(np.hypot(pts[:, 1], pts[:, 2]).max())
    nf, nb = _mean_normal(pm, "wedge_front"), _mean_normal(pm, "wedge_back")
    th = float(math.atan2(np.linalg.norm(np.cross(nf, nb)), abs(float(np.dot(nf, nb)))))
    row = {"declared": declared, "ref_x_min_m": rx0, "ref_x_max_m": rx1, "ref_r_max_m": rr,
           "mesh_x_min_m": mx0, "mesh_x_max_m": mx1, "mesh_r_max_m": mr,
           "s_x": (mx1 - mx0) / (rx1 - rx0), "s_r": mr / rr, "theta_mesh_rad": th, "theta_rad": theta}
    fields = []
    if not declared:
        fields.append("MFID-SCALE:declared")
    for k in ("s_x", "s_r"):
        if not abs(row[k] - 1.0) <= TOL["scale_rel"]:
            fields.append("MFID-SCALE:" + k)
    if not abs(th - theta) <= TOL["theta_rad"]:
        fields.append("MFID-SCALE:theta")
    row["pass"] = not fields
    return row, fields


def centre_bound(r_max, h, rho, revolved, theta):
    """R (1 - cos(theta/2)) on a revolved patch, plus h^2 / (8 rho) on a curved generator, plus the extra."""
    b = r_max * (1.0 - math.cos(theta / 2.0)) if revolved else 0.0
    if rho is not None:
        b += h * h / (8.0 * rho)
    return b + TOL["centre_extra_m"]


def judge_patch(vertex_d, centre_d, bounds):
    """(n_vertex_over, n_centre_over): vertices over TOL["vertex_m"], centres over their own bound."""
    nv = int(sum(1 for d in vertex_d if not d <= TOL["vertex_m"]))
    nc = int(sum(1 for d, b in zip(centre_d, bounds) if not d <= b))
    return nv, nc


def patch_row(pm, name, entry, theta):
    """One patch's PATCH_KEYS row: every vertex and every face centre projected onto its tagged faces."""
    tags, fs, rho = entry
    revolved = name not in dict(SIDES)
    rng, ptype = _patch_faces(pm, name)
    pts, faces = pm["points"], pm["faces"]
    vs = sorted(set(v for f in rng for v in faces[f]))
    vd, n_out = [], 0
    for v in vs:
        d, inside = fs.distance(pts[v])
        vd.append(d)
        n_out += 0 if inside else 1
    _, cf, r_max, h = face_table(pts, faces, rng, revolved)
    cd = [fs.distance(c)[0] for c in cf]
    bounds = [centre_bound(float(r), float(e), rho, revolved, theta) for r, e in zip(r_max, h)]
    nv, nc = judge_patch(vd, cd, bounds)
    vd_a, cd_a = np.array(vd), np.array(cd)
    k = int(np.argmax(vd_a))
    return {"tags": tags, "type": ptype, "n_faces": len(rng), "n_vertices": len(vs), "rho_min_m": rho,
            "vertex_max_m": float(vd_a.max()), "vertex_p99_m": float(np.percentile(vd_a, 99)),
            "vertex_argmax_m": [float(c) for c in pts[vs[k]]], "n_vertex_over": nv, "n_vertex_outside": n_out,
            "centre_max_m": float(cd_a.max()), "centre_p99_m": float(np.percentile(cd_a, 99)),
            "centre_ratio_max": float(max(d / b for d, b in zip(cd, bounds))), "n_centre_over": nc,
            "pass": nv == 0 and nc == 0}


def _paths(geom_dir, case_dir):
    pm_dir = os.path.join(case_dir, "constant", "polyMesh")
    return ([("geom/" + n, os.path.join(geom_dir, n)) for n in GEOM_FILES]
            + [("polyMesh/" + n, os.path.join(pm_dir, n)) for n in MESH_FILES])


def _snapshots(geom_dir, case_dir):
    return [(n, common.stable_file_snapshot(p)) for n, p in _paths(geom_dir, case_dir)]


def _report(status_fields, inputs, theta_deg, scale, patches):
    fields = list(status_fields)
    bind = [f for f in fields if f.startswith("MFID-BIND:")]
    reason = "MFID-BIND" if bind else (fields[0].split(":")[0] if fields else None)
    return {"version": VERSION, "status": "refused" if fields else "ok", "reason_id": reason, "fields": fields,
            "tolerances": dict(TOL), "theta_deg": float(theta_deg), "inputs": inputs, "scale": scale,
            "patches": patches}


def check(geom_dir, case_dir, theta_deg=THETA_DEG, between_hook=None):
    """The fidelity report (REPORT_KEYS). Order: MFID-BIND (every input a stable regular file), MFID-PATCH (the six
    names, both wedge sides typed wedge), then MFID-SCALE, MFID-VERTEX and MFID-CENTRE, every failure listed in
    that order; the inputs are hashed again after the judgement (between_hook, a selftest hook, runs first) and
    a change refuses MFID-BIND whatever else failed. reason_id is the first failing stage."""
    theta = math.radians(theta_deg)
    before = _snapshots(geom_dir, case_dir)
    inputs = dict((n, s["sha256"]) for n, s in before)
    unstable = ["MFID-BIND:" + n for n, s in before if s["stable"] is not True]
    if unstable:
        return _report(unstable, inputs, theta_deg, None, {})
    pm = polymesh_write.read_polymesh(os.path.join(case_dir, "constant", "polyMesh"))
    types = dict((p["name"], p["type"]) for p in pm["patches"])
    bad = []
    if len(pm["patches"]) != len(PATCH_ORDER) or sorted(types) != sorted(PATCH_ORDER):
        bad.append("MFID-PATCH:names")
    elif types["wedge_front"] != "wedge" or types["wedge_back"] != "wedge":
        bad.append("MFID-PATCH:wedge_types")
    if bad:
        return _report(bad, inputs, theta_deg, None, {})
    geom = common.read_json(os.path.join(geom_dir, "geom.json"))
    brep, meridian = load_brep(geom_dir, theta)
    scale, fields = scale_row(pm, geom, meridian, theta)
    patches = dict((name, patch_row(pm, name, brep[name], theta)) for name in PATCH_ORDER)
    fields += ["MFID-VERTEX:" + n for n in PATCH_ORDER if patches[n]["n_vertex_over"]]
    fields += ["MFID-CENTRE:" + n for n in PATCH_ORDER if patches[n]["n_centre_over"]]
    if between_hook is not None:
        between_hook(case_dir)
    after = dict(_snapshots(geom_dir, case_dir))
    changed = ["MFID-BIND:" + n for n, s in before
               if after[n]["stable"] is not True or after[n]["sha256"] != s["sha256"]]
    return _report(changed + fields, inputs, theta_deg, scale, patches)


def write_report(path, report):
    """The report as canonical JSON plus a newline, written atomically."""
    common.atomic_write(path, common.canonical_json(report) + common.NL)


def main(argv):
    """--selftest | check GEOM_DIR CASE_DIR OUT_JSON (exit 0 ok, 1 refused); anything else prints usage, exit 2."""
    if argv == ["--selftest"]:
        try:
            selftest()
        except Exception:
            import traceback
            traceback.print_exc()
            return 1
        return 0
    if len(argv) == 4 and argv[0] == "check":
        rep = check(os.path.abspath(argv[1]), os.path.abspath(argv[2]))
        write_report(argv[3], rep)
        print(common.canonical_json({"status": rep["status"], "reason_id": rep["reason_id"], "fields": rep["fields"]}))
        return 0 if rep["status"] == "ok" else 1
    sys.stderr.write(USAGE + common.NL)
    return 2


def _fx_copy_case(src, dst):
    shutil.copytree(src, dst)
    return dst


def _fx_scale_points(case_dir, s):
    """Every point of the polyMesh multiplied by s (17 significant digits, as the converter writes them)."""
    path = os.path.join(case_dir, "constant", "polyMesh", "points")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    num = r"([-+0-9.eE]+)"
    pat = re.compile(r"^\(" + num + " " + num + " " + num + r"\)$", re.M)
    out = pat.sub(lambda m: "(%s)" % " ".join(repr(float(g) * s) for g in m.groups()), text)
    common.atomic_write(path, out)


def _fx_swap_names(case_dir, a, b):
    """The patch names a and b exchanged in the boundary file (face ranges and types stay where they were)."""
    path = os.path.join(case_dir, "constant", "polyMesh", "boundary")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    text = re.sub(r"^(\s*)" + a + r"$", r"\1@@SWAP@@", text, flags=re.M)
    text = re.sub(r"^(\s*)" + b + r"$", r"\1" + a, text, flags=re.M)
    text = text.replace("@@SWAP@@", b)
    common.atomic_write(path, text)


def _fx_retype(case_dir, name, new_name=None, new_type=None):
    """One boundary entry renamed and/or retyped."""
    path = os.path.join(case_dir, "constant", "polyMesh", "boundary")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if new_type is not None:
        text = re.sub(r"(^\s*" + name + r"\s*\{\s*type\s+)\w+;", r"\1" + new_type + ";", text, flags=re.M)
    if new_name is not None:
        text = re.sub(r"^(\s*)" + name + r"$", r"\1" + new_name, text, flags=re.M)
    common.atomic_write(path, text)


def _fx_append_byte(case_dir):
    with open(os.path.join(case_dir, "constant", "polyMesh", "points"), "ab") as f:
        f.write(b" ")


def _fx_mutant_mesh(td, name, params):
    """An export of params and its L0 wedge; returns (geom_dir, case_dir)."""
    import export
    import wedge_mesh
    g = os.path.join(td, name + "_geom")
    res = export.run_pipeline(export.TEMPLATE, params, g)
    assert res["status"] == "ok", (name, res["status"], res["rule"], res["message"])
    w = wedge_mesh.run(g, os.path.join(td, name + "_wedge"), levels=(0,))
    assert w["status"] == "ok", (name, w["rule"], w["message"])
    return g, os.path.join(td, name + "_wedge", "L0", "case")


def selftest():
    """The nominal L1 and L2 wedges pass; the 1.001 scale, the label swap and D_e +1 % are refused."""
    import time
    import export
    import wedge_mesh
    t0 = time.time()
    assert THETA_DEG == wedge_mesh.RECIPE["theta_deg"] and PATCH_ORDER == wedge_mesh.GROUPS
    with tempfile.TemporaryDirectory() as td:
        geom = os.path.join(td, "geom")
        res = export.run_pipeline(export.TEMPLATE, dict(export.NOMINAL), geom)
        assert res["status"] == "ok", (res["status"], res["rule"], res["message"])
        w = wedge_mesh.run(geom, os.path.join(td, "wedge"), levels=(1, 2))
        assert w["status"] == "ok", (w["rule"], w["message"])
        case = dict((n, os.path.join(td, "wedge", "L%d" % n, "case")) for n in (1, 2))

        # T1, T2: the nominal L1 and L2 pass, at the written precision, with the cylinder's centre bound tight
        sag = 0.03 * (1.0 - math.cos(math.radians(2.5)))
        reps = {}
        for n in (1, 2):
            rep = check(geom, case[n])
            reps[n] = rep
            assert rep["status"] == "ok" and rep["fields"] == [] and rep["reason_id"] is None, rep["fields"]
            assert rep["scale"]["pass"] and abs(rep["scale"]["s_x"] - 1.0) <= 1e-15, rep["scale"]
            vmax = max(p["vertex_max_m"] for p in rep["patches"].values())
            assert vmax < 1e-13, vmax
            for p in rep["patches"].values():
                assert p["vertex_p99_m"] <= p["vertex_max_m"] and p["centre_p99_m"] <= p["centre_max_m"], p
                assert p["n_vertex_outside"] == 0 and p["centre_ratio_max"] <= 1.0, p
            su = rep["patches"]["slip_upstream"]
            assert abs(su["centre_max_m"] - sag) <= 1e-12 and su["centre_ratio_max"] > 0.9999, su
            print("[ok] L%d passes: vertex max %.2e m (wall_nozzle p99 %.2e), theta %.12f deg, slip_upstream centre "
                  "%.6e m = 0.03 (1 - cos 2.5 deg), centre ratio max %.6f"
                  % (n, vmax, rep["patches"]["wall_nozzle"]["vertex_p99_m"],
                     math.degrees(rep["scale"]["theta_mesh_rad"]), su["centre_max_m"],
                     max(p["centre_ratio_max"] for p in rep["patches"].values())))

        # T3: face_table's vectorised fan agrees with regions_check.face_geometry on every face of L1
        pm = polymesh_write.read_polymesh(os.path.join(case[1], "constant", "polyMesh"))
        worst = 0.0
        for q in pm["patches"]:
            rng = range(q["startFace"], q["startFace"] + q["nFaces"])
            sf, cf = face_table(pm["points"], pm["faces"], rng, True)[:2]
            for k, f in enumerate(rng):
                s1, c1 = regions_check.face_geometry(pm["points"], pm["faces"][f])
                worst = max(worst, float(np.abs(sf[k] - s1).max()), float(np.abs(cf[k] - c1).max()))
        assert worst <= 1e-15, worst
        print("[ok] face_table agrees with regions_check.face_geometry within %.1e (<= 1e-15) on all %d boundary "
              "faces of L1" % (worst, sum(q["nFaces"] for q in pm["patches"])))

        # T4: the report is canonical, path-free and exactly the declared keys; the CLI writes the same bytes
        out = os.path.join(td, "fid.json")
        assert main(["check", geom, case[1], out]) == 0
        with open(out, "rb") as f:
            blob = f.read()
        assert blob == (common.canonical_json(reps[1]) + common.NL).encode("ascii"), "CLI bytes differ"
        assert sorted(reps[1]) == sorted(REPORT_KEYS) and sorted(reps[1]["scale"]) == sorted(SCALE_KEYS)
        for p in reps[1]["patches"].values():
            assert sorted(p) == sorted(PATCH_KEYS), sorted(p)
        text = blob.decode("ascii")
        for poison in (td, td.replace(chr(92), "/"), td.replace(chr(92), chr(92) * 2), "C:"):
            assert poison not in text, poison
        assert sorted(reps[1]["inputs"]) == sorted(n for n, p in _paths(geom, case[1]))
        assert main([]) == 2 and main(["check", geom]) == 2
        print("[ok] report canonical, path-free, %d inputs hashed; the CLI writes the same bytes; usage exits 2"
              % len(reps[1]["inputs"]))

        # T5: the mesh scaled 1.001 is refused MFID-SCALE first
        sc = _fx_copy_case(case[1], os.path.join(td, "scaled", "case"))
        _fx_scale_points(sc, 1.001)
        rep = check(geom, sc)
        assert rep["status"] == "refused" and rep["reason_id"] == "MFID-SCALE", rep["fields"]
        assert rep["fields"][:2] == ["MFID-SCALE:s_x", "MFID-SCALE:s_r"], rep["fields"]
        assert abs(rep["scale"]["s_x"] - 1.001) <= 1e-12 and abs(rep["scale"]["s_r"] - 1.001) <= 1e-12
        assert "MFID-VERTEX:inlet" in rep["fields"], rep["fields"]
        print("[ok] scaled 1.001: refused MFID-SCALE (s_x %.12f, s_r %.12f), also %s"
              % (rep["scale"]["s_x"], rep["scale"]["s_r"], " ".join(rep["fields"][2:])))

        # T6: wall_nozzle and slip_upstream swapped: refused MFID-VERTEX on both, scale intact
        sw = _fx_copy_case(case[1], os.path.join(td, "swap", "case"))
        _fx_swap_names(sw, "wall_nozzle", "slip_upstream")
        rep = check(geom, sw)
        assert rep["reason_id"] == "MFID-VERTEX" and rep["scale"]["pass"], rep["fields"]
        assert rep["fields"][:2] == ["MFID-VERTEX:wall_nozzle", "MFID-VERTEX:slip_upstream"], rep["fields"]
        assert all(rep["patches"][n]["pass"] for n in ("inlet", "outlet", "wedge_front", "wedge_back"))
        print("[ok] wall_nozzle/slip_upstream swapped: refused %s (vertex max %.4e m, %.4e m)"
              % (" ".join(rep["fields"]), rep["patches"]["wall_nozzle"]["vertex_max_m"],
                 rep["patches"]["slip_upstream"]["vertex_max_m"]))

        # T7: a mesh from D_e +1 % (Lx follows D_e): refused by the x span first; against its own BRep it passes
        de_params = dict(export.NOMINAL, CR=9.0 / 1.0201)
        g_de, c_de = _fx_mutant_mesh(td, "de", de_params)
        rep = check(geom, c_de)
        assert rep["reason_id"] == "MFID-SCALE" and rep["fields"][0] == "MFID-SCALE:s_x", rep["fields"]
        assert abs(rep["scale"]["s_x"] - 1.001) <= 1e-9 and abs(rep["scale"]["s_r"] - 1.0) <= 1e-12, rep["scale"]
        assert "MFID-VERTEX:wall_nozzle" in rep["fields"], rep["fields"]
        own = check(g_de, c_de)
        assert own["status"] == "ok", own["fields"]
        print("[ok] D_e +1 %%: refused %s (s_x %.9f); the same mesh against its own BRep passes"
              % (" ".join(rep["fields"]), rep["scale"]["s_x"]))

        # T8: D_e +1 % with Lx held (Lx_over_De 0.5 / 1.01): the scale holds and the wall refuses MFID-VERTEX
        g_dl, c_dl = _fx_mutant_mesh(td, "del", dict(de_params, Lx_over_De=0.5 / 1.01))
        rep = check(geom, c_dl)
        assert rep["scale"]["pass"] and rep["reason_id"] == "MFID-VERTEX", (rep["scale"], rep["fields"])
        assert "MFID-VERTEX:wall_nozzle" in rep["fields"], rep["fields"]
        assert check(g_dl, c_dl)["status"] == "ok"
        print("[ok] D_e +1 %% with Lx held: scale passes, refused %s (wall_nozzle vertex max %.4e m); passes "
              "against its own BRep" % (" ".join(rep["fields"]), rep["patches"]["wall_nozzle"]["vertex_max_m"]))

        # T9: MFID-PATCH (a renamed side, a side typed patch), MFID-SCALE:declared and :theta
        for tag, kw, want in (("rename", {"new_name": "wedge_side"}, "MFID-PATCH:names"),
                              ("retype", {"new_type": "patch"}, "MFID-PATCH:wedge_types")):
            c = _fx_copy_case(case[1], os.path.join(td, tag, "case"))
            _fx_retype(c, "wedge_front", **kw)
            rep = check(geom, c)
            assert rep["reason_id"] == "MFID-PATCH" and rep["fields"] == [want], rep["fields"]
            assert rep["scale"] is None and rep["patches"] == {}
        g_mm = os.path.join(td, "geom_mm")
        shutil.copytree(geom, g_mm)
        gj = common.read_json(os.path.join(g_mm, "geom.json"))
        gj["units"] = "mm"
        common.write_json(os.path.join(g_mm, "geom.json"), gj)
        rep = check(g_mm, case[1])
        assert rep["fields"] == ["MFID-SCALE:declared"], rep["fields"]
        rep = check(geom, case[1], theta_deg=4.0)
        assert rep["fields"][0] == "MFID-SCALE:theta", rep["fields"]
        print("[ok] refused MFID-PATCH:names, MFID-PATCH:wedge_types, MFID-SCALE:declared (units mm) and "
              "MFID-SCALE:theta (theta 4 deg against a 5 deg mesh)")

        # T10: MFID-BIND: a file rewritten between the two hashes, and a hard-linked points file
        rep = check(geom, case[1], between_hook=_fx_append_byte)
        assert rep["reason_id"] == "MFID-BIND" and rep["fields"] == ["MFID-BIND:polyMesh/points"], rep["fields"]
        hl = _fx_copy_case(case[2], os.path.join(td, "hl", "case"))
        os.link(os.path.join(hl, "constant", "polyMesh", "points"), os.path.join(td, "hl", "points_link"))
        rep = check(geom, hl)
        assert rep["fields"] == ["MFID-BIND:polyMesh/points"] and rep["inputs"]["polyMesh/points"] is None
        print("[ok] refused MFID-BIND: points rewritten between the two hashes; a hard-linked points file")

        # T11: each judged miss alone: a centre over its bound, a vertex over 1e-9 m
        assert judge_patch([0.0, 1e-9], [1e-5, 2e-5], [1e-5, 2e-5]) == (0, 0)
        assert judge_patch([0.0, 1e-9], [1e-5, 2e-5 + 1e-15], [1e-5, 2e-5]) == (0, 1)
        assert judge_patch([0.0, 1.0000001e-9], [1e-5, 2e-5], [1e-5, 2e-5]) == (1, 0)
        assert judge_patch([math.nan], [0.0], [1.0]) == (1, 0)
        b = centre_bound(0.03, 1e-4, 0.034, True, math.radians(5.0))
        assert abs(b - (sag + 1e-8 / (8 * 0.034) + 1e-9)) <= 1e-18, b
        print("[ok] judge_patch: a centre 1e-15 m over its bound and a vertex 1e-16 m over 1e-9 m each fail alone; "
              "NaN fails; centre_bound is R(1 - cos 2.5 deg) + h^2/(8 rho) + 1e-9")
    print("selftest wall %.1f s" % (time.time() - t0))
    print("SELFTEST PASS")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
