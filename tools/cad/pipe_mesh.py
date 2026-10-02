#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""pipe_mesh.py - TG0's periodic pipe wedge (docs/16 §H.5 TG0, gate GC-7 of §H.3).

A 5-degree wedge of one pipe, R = 0.025 m and periodic length 0.1 m with 4 axial
cells, at three radial refinements 40/80/160 (r = 2), graded geometrically to the
wall so the first cell is the a priori y+1 = 1 height of the larger Re_tau. The two
end faces are a cyclic pair under the +x translation (L, 0, 0).

The converter writes both patches `type cyclic` but no neighbourPatch, and the
solver's polyMesh reader then refuses the case ("cyclic patch 'periodic_a' has no
neighbourPatch entry", rust/src/io/polymesh.rs:171), so the pair is written here:
pair_cyclic orders periodic_b so face k of b is the partner of face k of a (the
reader checks face k against face k under the translation and Sf_a == -Sf_b,
polymesh.rs:426-530) and rewrites the polyMesh through polymesh_write with both
patches naming each other in write_boundary_file's layout. The resolved wall's tau
is about 4e-3, far under GC-6's 0.05, so -check runs at the supervisor's
min_thickness_ratio 1e-4 (its F3 decision) and REPORTS tau_min; GC-7 asks only
that -check exits 0.

Usage:
  python pipe_mesh.py --selftest
  python pipe_mesh.py run OUT_DIR
  python pipe_mesh.py gmsh-build LEVEL MSH_PATH
"""
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common
sys.path.insert(0, os.path.join(common.REPO, "tools", "mesh"))
import polymesh_write
import regions_check
import wedge_mesh

RECIPE = {"version": "cad-pipe/1", "R_m": 0.025, "L_m": 0.1, "theta_deg": 5.0, "nx": 4, "nr": [40, 80, 160],
          "levels": [0, 1, 2], "re_tau": [576.69, 2358.0], "nu_m2_s": 1.516e-5,
          "h1_rule": "h1_0 = R / max(re_tau) (y+1 = 1 at the larger Re_tau), h1 = h1_0 * 2 ** -level",
          "grading": "geometric to the wall", "axial": "nx uniform cells at every level"}
GC7_PIPE = {"volume_rel": 3e-3, "cyclic_gap_m": 1e-9, "sf_rel": 1e-9, "first_cell_rel": 0.05, "yplus_max": 1.0}
CHECK_TAU_TURB = 1e-4
PATCHES = ("periodic_a", "periodic_b", "wall", "wedge_front", "wedge_back")
TYPE_ARGS = ["-type", "wedge_front=wedge", "-type", "wedge_back=wedge",
             "-type", "periodic_a=cyclic", "-type", "periodic_b=cyclic"]
REPORT_KEYS = ("version", "recipe", "recipe_sha", "gc7_tolerances", "check_tau", "bins", "h1_0_m",
               "volume_ref_m3", "levels", "gc7_pass")
LEVEL_KEYS = ("level", "nr", "nx", "q", "cells", "elements", "h1_target_m", "h1_max_m", "h1_min_m", "h1_rel",
              "yplus1_max", "volume_m3", "volume_rel", "patches", "cyclic", "check", "msh_sha256",
              "polymesh_sha256", "gc7")
CYCLIC_KEYS = ("n_a", "n_b", "neighbour_a", "neighbour_b", "max_gap_m", "sf_rel_max", "reordered")
GC7_KEYS = ("volume", "cyclic", "check", "first_cell", "cells", "pass")
USAGE = ("usage: python pipe_mesh.py --selftest" + chr(10)
         + "       python pipe_mesh.py run OUT_DIR" + chr(10)
         + "       python pipe_mesh.py gmsh-build LEVEL MSH_PATH")
PM_FILES = ("boundary", "faces", "neighbour", "owner", "points")


def gmsh_build(level, msh_path):
    """The gmsh child's whole job: the meridian rectangle x in [0, L], y in [0, R] on z = 0, rotated
    by -theta/2 about +x and revolved by theta with one recombined layer, transfinite nx uniform axial
    cells and nr radial cells geometric to the wall from h1, the five patch groups and the fluid
    volume -> MSH 4.1 ASCII at msh_path. Runs only in a fresh process; returns the build dict.

    Built with the geo kernel and both radial edges oriented outer -> axis with the same Progression
    beta: the two caps then share one transfinite recurrence and the cyclic pair closes at machine
    precision. The occ kernel re-meshes the far cap's radial edge with the reciprocal beta and misses
    the pair at 1.4e-8 of |Sf|, over GC7_PIPE's 1e-9."""
    import gmsh
    if level not in RECIPE["levels"]:
        raise wedge_mesh.Refused("PIPE-LEVEL", "level %r is not one of %r" % (level, RECIPE["levels"]))
    r, ln = RECIPE["R_m"], RECIPE["L_m"]
    th = math.radians(RECIPE["theta_deg"])
    nx = RECIPE["nx"]
    nr = RECIPE["nr"][RECIPE["levels"].index(level)]
    h1 = (r / max(RECIPE["re_tau"])) * 2.0 ** -level
    q = wedge_mesh.solve_q(h1, r, nr)
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
            gmsh.model.mesh.setTransfiniteCurve(t, nr + 1, "Progression", q)
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
            raise wedge_mesh.Refused("PIPE-TOPO", "volume elements %r, want hex and prism only" % (elements,))
        gmsh.write(msh_path)
        version = gmsh.GMSH_API_VERSION
    finally:
        gmsh.finalize()
    return {"level": level, "nr": nr, "nx": nx, "h1_target_m": h1, "q": q, "elements": elements,
            "n_cells": sum(elements.values()), "gmsh": version}


def _classify(gmsh):
    """Every surface of the revolved model into one of PATCHES, by bounding box: a plane face at
    x = 0 is periodic_a and at x = L periodic_b, a face spanning both wedge sides is the wall, and
    the two flat sides separate by the sign of their z extent."""
    rs = RECIPE["R_m"] * math.sin(math.radians(RECIPE["theta_deg"]) / 2.0)
    ln = RECIPE["L_m"]
    groups = dict((g, []) for g in PATCHES)
    for d, t in gmsh.model.getEntities(2):
        b = gmsh.model.getBoundingBox(2, t)
        if b[3] - b[0] < wedge_mesh.TOL_GEOM:
            if abs(b[0]) < wedge_mesh.TOL_GEOM:
                groups["periodic_a"].append(t)
            elif abs(b[0] - ln) < wedge_mesh.TOL_GEOM:
                groups["periodic_b"].append(t)
        elif b[2] < -rs / 2.0 and b[5] > rs / 2.0:
            groups["wall"].append(t)
        elif b[5] < rs / 2.0:
            groups["wedge_front"].append(t)
        else:
            groups["wedge_back"].append(t)
    got = dict((g, len(v)) for g, v in groups.items())
    want = dict((g, 1) for g in PATCHES)
    if got != want:
        raise wedge_mesh.Refused("PIPE-TOPO", "surface groups %r, want %r" % (got, want))
    return groups


def build_level(level, msh_path):
    """One fresh gmsh-build child (a fresh process per build); the build dict from its last stdout line."""
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


def _read_cyclic(case_dir):
    pm = polymesh_write.read_polymesh(os.path.join(case_dir, "constant", "polyMesh"))
    rows = dict((p["name"], p) for p in pm["patches"])
    missing = [n for n in ("periodic_a", "periodic_b") if n not in rows]
    if missing:
        raise wedge_mesh.Refused("PIPE-CYCLIC", "the boundary has no %s" % ", ".join(missing))
    return pm, rows["periodic_a"], rows["periodic_b"]


def _face_data(pm, patch):
    """(centroids, Sf) of one patch's faces, in face order."""
    pts, faces = pm["points"], pm["faces"]
    cf, sf = [], []
    for f in range(patch["startFace"], patch["startFace"] + patch["nFaces"]):
        s, c = regions_check.face_geometry(pts, faces[f])
        cf.append(c)
        sf.append(s)
    return np.array(cf), np.array(sf)


def measure_cyclic(case_dir, translation):
    """The cyclic pair face k to face k (no matching): the counts, the neighbourPatch each side
    carries (None when absent), the worst |C_b(k) - C_a(k) - translation| and the worst
    |Sf_a(k) + Sf_b(k)| / |Sf_a(k)|. When the counts differ the two maxima are None."""
    pm, a, b = _read_cyclic(case_dir)
    out = {"n_a": int(a["nFaces"]), "n_b": int(b["nFaces"]),
           "neighbour_a": a.get("neighbourPatch"), "neighbour_b": b.get("neighbourPatch"),
           "max_gap_m": None, "sf_rel_max": None, "reordered": False}
    if out["n_a"] != out["n_b"]:
        return out
    ca, sa = _face_data(pm, a)
    cb, sb = _face_data(pm, b)
    tr = np.asarray(translation, dtype=np.float64)
    out["max_gap_m"] = float(np.linalg.norm(cb - (ca + tr), axis=1).max())
    out["sf_rel_max"] = float((np.linalg.norm(sa + sb, axis=1) / np.linalg.norm(sa, axis=1)).max())
    return out


def pair_cyclic(case_dir, translation):
    """Match each periodic_a face to the periodic_b face whose centroid is nearest to its centroid
    plus the translation, refuse unless the match is a bijection within GC7_PIPE's cyclic gap, then
    rewrite the five polyMesh files with periodic_b's faces (and owners) reordered so face k of b is
    the partner of face k of a, and both cyclic patches written as 4-tuples naming each other.
    Returns the CYCLIC_KEYS dict measured AFTER the rewrite."""
    pm, a, b = _read_cyclic(case_dir)
    if a["nFaces"] != b["nFaces"]:
        raise wedge_mesh.Refused("PIPE-CYCLIC", "periodic_a has %d faces and periodic_b %d"
                                 % (a["nFaces"], b["nFaces"]))
    ca, _ = _face_data(pm, a)
    cb, _ = _face_data(pm, b)
    tr = np.asarray(translation, dtype=np.float64)
    order = [int(np.argmin(np.sum((cb - (ca[k] + tr)) ** 2, axis=1))) for k in range(len(ca))]
    if len(set(order)) != len(order):
        raise wedge_mesh.Refused("PIPE-CYCLIC", "the nearest-centroid match is not a bijection")
    gaps = np.linalg.norm(cb[order] - (ca + tr), axis=1)
    if float(gaps.max()) > GC7_PIPE["cyclic_gap_m"]:
        raise wedge_mesh.Refused("PIPE-CYCLIC", "the best match misses by %r m, over the gap %r"
                                 % (float(gaps.max()), GC7_PIPE["cyclic_gap_m"]))
    faces, owner = pm["faces"], pm["owner"]
    fb = [list(faces[f]) for f in range(b["startFace"], b["startFace"] + b["nFaces"])]
    ob = [int(owner[f]) for f in range(b["startFace"], b["startFace"] + b["nFaces"])]
    moved = order != list(range(len(order)))
    if moved:
        new_faces = [list(f) for f in faces[:b["startFace"]]] + [fb[k] for k in order]
        new_faces.extend(list(f) for f in faces[b["startFace"] + b["nFaces"]:])
        new_owner = [int(v) for v in owner[:b["startFace"]]] + [ob[k] for k in order]
        new_owner.extend(int(v) for v in owner[b["startFace"] + b["nFaces"]:])
    else:
        new_faces = [list(f) for f in faces]
        new_owner = [int(v) for v in owner]
    # the rewrite always happens: it is what adds the neighbourPatch entries the reader needs
    rows = [(p["name"], p["type"], p["nFaces"],
             "periodic_b" if p["name"] == "periodic_a"
             else ("periodic_a" if p["name"] == "periodic_b" else None))
            for p in pm["patches"]]
    polymesh_write.write_polymesh(os.path.join(case_dir, "constant", "polyMesh"),
                                  pm["points"], new_faces, new_owner, pm["neighbour"], rows)
    out = measure_cyclic(case_dir, translation)
    out["reordered"] = moved
    return out


def judge(row):
    """The five GC-7 predicates plus their conjunction, each a bool; None never passes a predicate."""
    cy = row["cyclic"]
    c = row["check"]
    out = {"volume": abs(row["volume_rel"]) <= GC7_PIPE["volume_rel"],
           "cyclic": cy["n_a"] == cy["n_b"] == row["nr"]
                     and cy["neighbour_a"] == "periodic_b" and cy["neighbour_b"] == "periodic_a"
                     and cy["max_gap_m"] is not None and cy["max_gap_m"] <= GC7_PIPE["cyclic_gap_m"]
                     and cy["sf_rel_max"] is not None and cy["sf_rel_max"] <= GC7_PIPE["sf_rel"],
           "check": c["exit"] == 0 and c["gate"] == "passed" and c["n_regions"] == 1
                    and c["n_cells"] == row["cells"],
           "first_cell": abs(row["h1_rel"]) <= GC7_PIPE["first_cell_rel"]
                         and max(row["yplus1_max"]) <= GC7_PIPE["yplus_max"],
           "cells": row["cells"] == row["nx"] * row["nr"]}
    out["pass"] = all(out[k] for k in GC7_KEYS[:-1])
    return out


def run(out_dir, levels=(0, 1, 2)):
    """All levels in one go: build, convert, pair, check, measure, judge; ok with the report or
    refused by id. A refusal is returned, never raised; any other exception propagates."""
    try:
        return _run(out_dir, levels)
    except wedge_mesh.Refused as r:
        return {"status": "refused", "rule": r.rule, "message": r.detail, "report": None}


def _run(out_dir, levels):
    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise wedge_mesh.Refused("PIPE-OUT", "%s exists and is not empty" % out_dir)
    os.makedirs(out_dir, exist_ok=True)
    bins = wedge_mesh.load_bins()
    r, ln = RECIPE["R_m"], RECIPE["L_m"]
    th = math.radians(RECIPE["theta_deg"])
    h1_0 = r / max(RECIPE["re_tau"])
    vol_ref = r * r * ln * th / 2.0
    levels_out = []
    for level in levels:
        ld = os.path.join(out_dir, "L%d" % level)
        os.makedirs(ld)
        build = build_level(level, os.path.join(ld, "pipe.msh"))
        msh_sha = common.sha256_file(os.path.join(ld, "pipe.msh"))
        case = os.path.join(ld, "case")
        wedge_mesh.convert(bins, os.path.join(ld, "pipe.msh"), case, type_args=TYPE_ARGS)
        cy = pair_cyclic(case, (ln, 0.0, 0.0))
        pm_dir = os.path.join(case, "constant", "polyMesh")
        polymesh_sha = {}
        for nm in PM_FILES:
            snap = common.stable_file_snapshot(os.path.join(pm_dir, nm))
            if snap["stable"] is not True:
                raise wedge_mesh.Refused("PIPE-CYCLIC", "polyMesh/%s is not a stable regular file" % nm)
            polymesh_sha[nm] = snap["sha256"]
        check = wedge_mesh.run_check(bins, case, os.path.join(ld, "check_config.json"), min_tau=CHECK_TAU_TURB)
        common.atomic_write(os.path.join(ld, "check.txt"), check["text"])
        meas = wedge_mesh.measure(case, wall_patch="wall")
        h1_target = h1_0 * 2.0 ** -level
        row = {"level": level, "nr": build["nr"], "nx": build["nx"], "q": build["q"], "cells": meas["n_cells"],
               "elements": build["elements"], "h1_target_m": h1_target,
               "h1_max_m": meas["h1_max_m"], "h1_min_m": meas["h1_min_m"],
               "h1_rel": (meas["h1_max_m"] - h1_target) / h1_target,
               "yplus1_max": [meas["h1_max_m"] * rt / r for rt in RECIPE["re_tau"]],
               "volume_m3": meas["volume_m3"], "volume_rel": (meas["volume_m3"] - vol_ref) / vol_ref,
               "patches": meas["patches"], "cyclic": dict(cy),
               "check": dict((k, v) for k, v in check.items() if k != "text"),
               "msh_sha256": msh_sha, "polymesh_sha256": dict(polymesh_sha)}
        row["gc7"] = judge(row)
        levels_out.append(row)
    decl = common.read_json(wedge_mesh.BIN_JSON)
    rep = {"version": 1, "recipe": RECIPE, "recipe_sha": common.sha256_of(RECIPE),
           "gc7_tolerances": GC7_PIPE, "check_tau": CHECK_TAU_TURB,
           "bins": dict((name, brow["sha256"]) for name, brow in decl["binaries"].items()),
           "h1_0_m": h1_0, "volume_ref_m3": vol_ref, "levels": levels_out,
           "gc7_pass": all(lv["gc7"]["pass"] for lv in levels_out)}
    common.atomic_write(os.path.join(out_dir, "pipe_mesh.json"), common.canonical_json(rep) + common.NL)
    return {"status": "ok", "rule": None, "message": "", "report": rep}


def main(argv):
    """--selftest | gmsh-build LEVEL MSH_PATH | run OUT_DIR."""
    if argv == ["--selftest"]:
        try:
            selftest()
        except Exception:
            import traceback
            traceback.print_exc()
            return 1
        return 0
    if len(argv) == 3 and argv[0] == "gmsh-build":
        try:
            build = gmsh_build(int(argv[1]), argv[2])
        except wedge_mesh.Refused as r:
            print(common.canonical_json({"refused": r.rule, "detail": r.detail}))
            return wedge_mesh.REFUSED_EXIT
        print(common.canonical_json(build))
        return 0
    if len(argv) == 2 and argv[0] == "run":
        res = run(os.path.abspath(argv[1]))
        if res["status"] != "ok":
            print(common.canonical_json({"status": res["status"], "rule": res["rule"], "message": res["message"]}))
            return 2
        for lv in res["report"]["levels"]:
            tau = lv["check"]["tau_min"]
            print("L%d cells %d tau %.6f h1_rel %+.4f volume_rel %+.6f cyclic gap %.1e gc7 %s"
                  % (lv["level"], lv["cells"], float("nan") if tau is None else tau, lv["h1_rel"],
                     lv["volume_rel"], float("nan") if lv["cyclic"]["max_gap_m"] is None
                     else lv["cyclic"]["max_gap_m"], "pass" if lv["gc7"]["pass"] else "FAIL"))
        print("GC-7 pipe %s" % ("pass" if res["report"]["gc7_pass"] else "FAIL"))
        return 0 if res["report"]["gc7_pass"] else 1
    sys.stderr.write(USAGE + common.NL)
    return 2


def selftest():
    """The periodic pipe wedge at three levels proves GC-7: the converter's pairless cyclic is
    refused by the solver's reader, the written pair passes it, and both gate sides can fail."""
    import time
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        bins = wedge_mesh.load_bins()
        ln = RECIPE["L_m"]

        # T1: the converter writes a pairless cyclic the reader refuses; the writer writes the pair
        ld = os.path.join(td, "t1", "L0")
        os.makedirs(ld)
        build_level(0, os.path.join(ld, "pipe.msh"))
        case = os.path.join(ld, "case")
        wedge_mesh.convert(bins, os.path.join(ld, "pipe.msh"), case, type_args=TYPE_ARGS)
        pm_dir = os.path.join(case, "constant", "polyMesh")
        with open(os.path.join(pm_dir, "boundary"), encoding="utf-8") as f:
            btxt = f.read()
        assert btxt.count("type            cyclic;") == 2, "the converter did not write two cyclic patches"
        assert "neighbourPatch" not in btxt, "the converter wrote a neighbourPatch after all"
        chk = wedge_mesh.run_check(bins, case, os.path.join(td, "t1", "cfg.json"), min_tau=CHECK_TAU_TURB)
        assert chk["exit"] != 0 and "has no neighbourPatch entry" in chk["text"], (chk["exit"], chk["text"][-300:])
        pm = polymesh_write.read_polymesh(pm_dir)
        rt = os.path.join(td, "t1", "rt")
        polymesh_write.write_polymesh(rt, pm["points"], pm["faces"], pm["owner"], pm["neighbour"],
                                      [(p["name"], p["type"], p["nFaces"]) for p in pm["patches"]])
        for nm in PM_FILES:
            with open(os.path.join(pm_dir, nm), "rb") as f:
                a = f.read()
            with open(os.path.join(rt, nm), "rb") as f:
                b = f.read()
            assert a == b, nm
        rt2 = os.path.join(td, "t1", "rt2")
        rows4 = [(p["name"], p["type"], p["nFaces"],
                  "periodic_b" if p["name"] == "periodic_a"
                  else ("periodic_a" if p["name"] == "periodic_b" else None)) for p in pm["patches"]]
        polymesh_write.write_polymesh(rt2, pm["points"], pm["faces"], pm["owner"], pm["neighbour"], rows4)
        pa = [p for p in pm["patches"] if p["name"] == "periodic_a"][0]
        with open(os.path.join(rt2, "boundary"), encoding="utf-8") as f:
            btxt2 = f.read()
        block = chr(10).join(["    periodic_a", "    {", "        type            cyclic;",
                              "        neighbourPatch  periodic_b;", "        nFaces          %d;" % pa["nFaces"],
                              "        startFace       %d;" % pa["startFace"], "    }", ""])
        assert block in btxt2, btxt2[:400]
        pm2 = polymesh_write.read_polymesh(rt2)
        nb = dict((p["name"], p.get("neighbourPatch")) for p in pm2["patches"])
        assert nb == {"periodic_a": "periodic_b", "periodic_b": "periodic_a", "wall": None,
                      "wedge_front": None, "wedge_back": None}, nb
        print("[ok] converter writes cyclic without neighbourPatch and -check refuses it; "
              "polymesh_write writes the pair in the solver's layout")

        # T2: run() at the three levels, GC-7 pass and the pinned counts
        ORACLE = {0: (40, 160, 156, 4, 1.159959), 1: (80, 320, 316, 4, 1.076399),
                  2: (160, 640, 636, 4, 1.037351)}
        H1 = (RECIPE["R_m"] / max(RECIPE["re_tau"]))
        res = run(os.path.join(td, "run"))
        assert res["status"] == "ok", (res["status"], res["rule"], res["message"])
        rep = res["report"]
        assert rep["gc7_pass"] is True, common.canonical_json(rep)
        assert abs(rep["volume_ref_m3"] - 2.7270769562411405e-06) <= 1e-18, rep["volume_ref_m3"]
        for lv in rep["levels"]:
            n = lv["level"]
            bad = [k for k, v in lv["gc7"].items() if not v and k != "pass"]
            assert lv["gc7"]["pass"], (n, bad, common.canonical_json(lv))
            want = ORACLE[n]
            assert (lv["nr"], lv["cells"], lv["elements"]["hex"], lv["elements"]["prism"]) == want[:4], \
                (n, lv["nr"], lv["cells"], lv["elements"])
            assert abs(lv["q"] - want[4]) <= 1e-5, (n, lv["q"])
            assert abs(lv["h1_target_m"] - H1 * 2.0 ** -n) <= 1e-15 * lv["h1_target_m"], (n, lv["h1_target_m"])
            print("[ok] pipe L%d: %d cells (hex %d, prism %d), q %.6f, first cell %+.6f, y+1 %.5f / %.5f, "
                  "volume %+.7f, tau %.6f, GC-7 pass"
                  % (n, lv["cells"], lv["elements"]["hex"], lv["elements"]["prism"], lv["q"], lv["h1_rel"],
                     lv["yplus1_max"][0], lv["yplus1_max"][1], lv["volume_rel"], lv["check"]["tau_min"]))

        # T3: the paired cyclic at every level
        for lv in rep["levels"]:
            cy = lv["cyclic"]
            assert cy["n_a"] == cy["n_b"] == lv["nr"], (lv["level"], cy["n_a"], cy["n_b"])
            assert cy["neighbour_a"] == "periodic_b" and cy["neighbour_b"] == "periodic_a", (lv["level"], cy)
            assert cy["max_gap_m"] <= GC7_PIPE["cyclic_gap_m"], (lv["level"], cy["max_gap_m"])
            assert cy["sf_rel_max"] <= GC7_PIPE["sf_rel"], (lv["level"], cy["sf_rel_max"])
        print("[ok] cyclic pair at L0/L1/L2: %s faces each side, max gap %.2e m, Sf rel %.2e, "
              "neighbourPatch both ways"
              % ("/".join(str(lv["nr"]) for lv in rep["levels"]),
                 max(lv["cyclic"]["max_gap_m"] for lv in rep["levels"]),
                 max(lv["cyclic"]["sf_rel_max"] for lv in rep["levels"])))

        # T5: MSH bytes identical across two fresh processes, equal to the recorded sha
        det = os.path.join(td, "det")
        os.makedirs(det)
        for lv in rep["levels"]:
            n = lv["level"]
            pa = os.path.join(det, "L%d_a.msh" % n)
            pb = os.path.join(det, "L%d_b.msh" % n)
            build_level(n, pa)
            build_level(n, pb)
            sa, sb = common.sha256_file(pa), common.sha256_file(pb)
            assert sa == sb == lv["msh_sha256"], (n, sa, sb, lv["msh_sha256"])
        print("[ok] MSH bytes identical across two fresh processes at L0, L1, L2")

        # T4: a reversed pair is refused by the reader's bijection and repaired by pair_cyclic
        cp = os.path.join(td, "t4", "case")
        shutil.copytree(os.path.join(td, "run", "L0", "case"), cp)
        pm3 = polymesh_write.read_polymesh(os.path.join(cp, "constant", "polyMesh"))
        a3 = [p for p in pm3["patches"] if p["name"] == "periodic_a"][0]
        b3 = [p for p in pm3["patches"] if p["name"] == "periodic_b"][0]
        i0, i1 = b3["startFace"], b3["startFace"] + b3["nFaces"]
        faces_rv = [list(f) for f in pm3["faces"][:i0]] + [list(f) for f in reversed(pm3["faces"][i0:i1])]
        faces_rv.extend(list(f) for f in pm3["faces"][i1:])
        own_rv = [int(v) for v in pm3["owner"][:i0]] + [int(v) for v in reversed(pm3["owner"][i0:i1])]
        own_rv.extend(int(v) for v in pm3["owner"][i1:])
        rows_rv = [(p["name"], p["type"], p["nFaces"],
                    "periodic_b" if p["name"] == "periodic_a"
                    else ("periodic_a" if p["name"] == "periodic_b" else None)) for p in pm3["patches"]]
        polymesh_write.write_polymesh(os.path.join(cp, "constant", "polyMesh"),
                                      pm3["points"], faces_rv, own_rv, pm3["neighbour"], rows_rv)
        cy_bad = measure_cyclic(cp, (ln, 0.0, 0.0))
        assert cy_bad["max_gap_m"] > 1e-3, cy_bad
        row0 = json.loads(json.dumps(rep["levels"][0]))
        row0["cyclic"] = dict(cy_bad)
        assert judge(row0)["cyclic"] is False and judge(row0)["pass"] is False, judge(row0)
        chk_bad = wedge_mesh.run_check(bins, cp, os.path.join(td, "t4", "cfg.json"), min_tau=CHECK_TAU_TURB)
        assert chk_bad["exit"] != 0 and "not a bijection" in chk_bad["text"], (chk_bad["exit"], chk_bad["text"][-300:])
        cy_fix = pair_cyclic(cp, (ln, 0.0, 0.0))
        assert cy_fix["reordered"] is True and cy_fix["max_gap_m"] <= GC7_PIPE["cyclic_gap_m"], cy_fix
        chk_fix = wedge_mesh.run_check(bins, cp, os.path.join(td, "t4", "cfg2.json"), min_tau=CHECK_TAU_TURB)
        assert chk_fix["exit"] == 0 and chk_fix["gate"] == "passed", (chk_fix["exit"], chk_fix["gate"])
        print("[ok] a reversed pair: gap %.3f m, -check refuses the bijection; pair_cyclic restores it "
              "(gap %.2e m, -check exit 0)" % (cy_bad["max_gap_m"], cy_fix["max_gap_m"]))

        # T6: the report is canonical, path-free, exactly the declared keys; usage exits 2
        path = os.path.join(td, "run", "pipe_mesh.json")
        with open(path, "rb") as f:
            blob = f.read()
        doc = common.read_json(path)
        assert blob == (common.canonical_json(doc) + common.NL).encode("ascii"), "pipe_mesh.json is not canonical"
        assert sorted(doc) == sorted(REPORT_KEYS), sorted(doc)
        assert sorted(doc["levels"][0]) == sorted(LEVEL_KEYS), sorted(doc["levels"][0])
        assert sorted(doc["levels"][0]["cyclic"]) == sorted(CYCLIC_KEYS)
        assert sorted(doc["levels"][0]["gc7"]) == sorted(GC7_KEYS)
        text = blob.decode("ascii")
        for poison in (td, os.path.abspath(td), td.replace(chr(92), "/"), "C:"):
            assert poison not in text, poison
        assert main([]) == 2 and main(["run"]) == 2, "usage must exit 2"
        print("[ok] pipe_mesh.json canonical and path-free; usage exits 2")

        # T7: each GC-7 predicate fails on its own planted miss, alone
        base = rep["levels"][0]
        assert judge(base)["pass"] is True
        misses = [("volume", ["volume_rel"], 0.0031),
                  ("cyclic", ["cyclic", "max_gap_m"], 2e-9),
                  ("check", ["check", "exit"], 1),
                  ("first_cell", ["h1_rel"], 0.051),
                  ("cells", ["cells"], base["cells"] + 1)]
        for rule, path_keys, value in misses:
            m = json.loads(json.dumps(base))
            node = m
            for k in path_keys[:-1]:
                node = node[k]
            node[path_keys[-1]] = value
            if rule == "cells":            # keep check's own cross-count consistent
                m["check"]["n_cells"] = m["cells"]
            j = judge(m)
            false_pred = [k for k, v in j.items() if not v and k != "pass"]
            assert j["pass"] is False and false_pred == [rule], (rule, false_pred)
        print("[ok] judge: each of 5 planted misses fails exactly its GC-7 predicate")

        # T8: five refusals, each by its exact id
        def refused_run(out, rule):
            os.makedirs(out, exist_ok=True)   # the L0-L2 smoke above already created it
            with open(os.path.join(out, "occupied.txt"), "w", encoding="utf-8") as f:
                f.write("occupied")
            got = run(out)
            assert got["status"] == "refused" and got["rule"] == rule and got["report"] is None, (rule, got)
        refused_run(os.path.join(td, "run"), "PIPE-OUT")

        def refused(fn, rule):
            try:
                fn()
            except wedge_mesh.Refused as r:
                assert r.rule == rule, (rule, r.rule, r.detail)
                return
            raise AssertionError("expected %s, got none" % rule)
        refused(lambda: build_level(3, os.path.join(td, "lvl3.msh")), "PIPE-LEVEL")
        refused(lambda: pair_cyclic(os.path.join(td, "run", "L0", "case"), (ln + 1e-6, 0.0, 0.0)),
                "PIPE-CYCLIC")
        lost = os.path.join(td, "t8", "case")
        shutil.copytree(os.path.join(td, "run", "L0", "case"), lost)
        pm4 = polymesh_write.read_polymesh(os.path.join(lost, "constant", "polyMesh"))
        a4 = [p for p in pm4["patches"] if p["name"] == "periodic_a"][0]
        b4 = [p for p in pm4["patches"] if p["name"] == "periodic_b"][0]
        w4 = [p for p in pm4["patches"] if p["name"] == "wall"][0]
        last = b4["startFace"] + b4["nFaces"] - 1
        wend = w4["startFace"] + w4["nFaces"]
        keep = list(range(0, last)) + list(range(last + 1, wend)) + [last] + list(range(wend, len(pm4["faces"])))
        faces_ls = [list(pm4["faces"][i]) for i in keep]
        own_ls = [int(pm4["owner"][i]) for i in keep]
        rows_ls = [(p["name"], p["type"],
                    b4["nFaces"] - 1 if p["name"] == "periodic_b"
                    else (w4["nFaces"] + 1 if p["name"] == "wall" else p["nFaces"]))
                   for p in pm4["patches"]]
        polymesh_write.write_polymesh(os.path.join(lost, "constant", "polyMesh"),
                                      pm4["points"], faces_ls, own_ls, pm4["neighbour"], rows_ls)
        refused(lambda: pair_cyclic(lost, (ln, 0.0, 0.0)), "PIPE-CYCLIC")
        print("[ok] refusals: PIPE-OUT, PIPE-LEVEL, PIPE-CYCLIC (wrong translation), "
              "PIPE-CYCLIC (counts differ)")

    print("selftest wall %.1f s" % (time.time() - t0))
    print("SELFTEST PASS")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
