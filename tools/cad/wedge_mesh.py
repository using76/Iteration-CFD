#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""wedge_mesh.py - the S6 mesh of the CAD loop (docs/16 §D S6, §H.2, gate GC-6 of §H.3): the exported meridian
becomes a 5 degree, one-layer axisymmetric wedge at three levels (r = 2), converted by ofgpu-convert-mesh with the
two wedge patches typed wedge, gated by ofgpu-automesher -check, and measured against the CAD.

Usage:
  python wedge_mesh.py --selftest
  python wedge_mesh.py run GEOM_DIR OUT_DIR [H1_FINE]
  python wedge_mesh.py gmsh-build GEOM_DIR LEVEL H1_FINE MSH_PATH
"""
import json
import math
import os
import re
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

# gmsh runs only in the gmsh-build child (a fresh process per build, as export.py's gmsh-span): the parent never
# imports gmsh or cadquery. Every convert goes into a directory that does not exist yet, because the converter
# refuses to overwrite a polyMesh (exit 1). The plan's "-type wedge_*=wedge" is not a glob in this converter (it
# leaves both sides type patch, silently), so TYPE_ARGS names the two patches and the boundary is read back.
BIN_JSON = os.path.join(HERE, "bin.json")
GC6 = {"volume_rel": 3e-3, "area_rel": 5e-3, "tau_min": 0.05, "first_cell_rel": 0.05, "cells_rel": 0.15}
PAPPUS_TOL = 1e-9             # rel: 2 pi A ybar of the meridian against geom.json's BREP volume
TYPE_ARGS = ["-type", "wedge_front=wedge", "-type", "wedge_back=wedge"]
CHECK_CONFIG = {"input": {"surfaces": [{"path": "unused.stl"}]},
                "domain": {"extent": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0], "base_size": 0.1},
                "quality": {"min_thickness_ratio": 0.05},
                "output": {"case_dir": "unused", "name": "unused"}}
CHILD_TIMEOUT_S = 600
REFUSED_EXIT = 3              # the gmsh-build child's exit status for a refusal
REPORT_KEYS = ("version", "recipe", "recipe_sha", "gc6_tolerances", "geom_sha256", "template_sha", "params_sha",
               "bins", "h1_fine_m", "pappus", "levels", "gc6_pass")
LEVEL_KEYS = ("level", "target_cells", "cells", "cells_rel", "elements", "nr", "nb", "h1_target_m", "h1_max_m",
              "h1_min_m", "h1_mean_m", "h1_rel", "volume_m3", "volume_pappus_m3", "volume_rel", "patches",
              "check", "msh_sha256", "polymesh_sha256", "gc6")
GC6_KEYS = ("volume", "areas", "check", "tau", "first_cell", "cells", "types", "pass")
USAGE = ("usage: python wedge_mesh.py --selftest" + chr(10)
         + "       python wedge_mesh.py run GEOM_DIR OUT_DIR [H1_FINE]" + chr(10)
         + "       python wedge_mesh.py gmsh-build GEOM_DIR LEVEL H1_FINE MSH_PATH")

RECIPE = {"version": "cad-wedge/1", "theta_deg": 5.0, "k_stations": 10, "tau_design": 0.06,
          "cells_l0": 10000, "ratio": 2, "levels": [0, 1, 2], "nr_min": 4,
          "grading": "geometric to the wall, first cell h1_fine * 2 ** (2 - level)",
          "axial": "uniform per block, n from the tau budget at level 2"}
H1_FINE_NOMINAL = 5.9e-6      # m: docs/16 §C (Thwaites planning numbers) and §L: the L2 first cell at Re_De 3e4
TOL_GEOM = 1e-6               # m: OCC bounding boxes are padded by about 1e-7 m
TOL_ANGLE = 1e-6              # rad: a wedge side's centre of mass sits at -theta/2 or +theta/2
GROUPS = ("inlet", "outlet", "wall_nozzle", "slip_upstream", "wedge_front", "wedge_back")
ELEMENT_NAMES = {5: "hex", 6: "prism"}


class Refused(Exception):
    """A refusal by id: the caller records rule and detail and writes nothing further."""
    def __init__(self, rule, detail):
        Exception.__init__(self, "%s: %s" % (rule, detail))
        self.rule = rule
        self.detail = detail


def solve_q(h1, length, n):
    """The geometric ratio q > 1 with h1 (q^n - 1) / (q - 1) = length: 200 fixed bisection steps on (1, 4)."""
    if h1 * n >= length:
        raise Refused("WEDGE-GRADE", "h1 %r times %d cells is not below the line length %r" % (h1, n, length))
    lo, hi = 1.0, 4.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if h1 * (mid ** n - 1.0) / (mid - 1.0) > length:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def stations(planes):
    """Block edges in x: inlet, contraction_start, k - 1 equal interior stations, exit_plane, outlet."""
    x_cs, x_ep, k = planes["contraction_start"], planes["exit_plane"], RECIPE["k_stations"]
    inner = [x_cs + (x_ep - x_cs) * i / k for i in range(1, k)]
    return [planes["inlet"], x_cs] + inner + [x_ep, planes["outlet"]]


def _bb(gmsh, dim, tag):
    return gmsh.model.getBoundingBox(dim, tag)


def _curve_ends(gmsh, c):
    """The curve's points at its two parameter bounds, in parameter order (the order a Progression follows)."""
    pb = gmsh.model.getParametrizationBounds(1, c)
    return [gmsh.model.getValue(1, c, [float(pb[0][0])]), gmsh.model.getValue(1, c, [float(pb[1][0])])]


def _wall_shape(gmsh, c):
    """(r_max, beta_max) of one wall curve from 257 samples: radius hypot(y, z), slope angle folded to [0, pi/2]."""
    pb = gmsh.model.getParametrizationBounds(1, c)
    t0, t1 = float(pb[0][0]), float(pb[1][0])
    pts = [gmsh.model.getValue(1, c, [t0 + (t1 - t0) * i / 256]) for i in range(257)]
    rr = [math.hypot(p[1], p[2]) for p in pts]
    beta = max(abs(math.atan2(rr[i + 1] - rr[i], pts[i + 1][0] - pts[i][0])) for i in range(256))
    return max(rr), min(beta, math.pi - beta)


def _split_meridian(gmsh, geom_dir, edges):
    """Import meridian.step at scale 1 (metres), fragment it by the interior station lines, drop the dangling
    pieces; returns (the block surfaces sorted by x, meridian area, meridian centroid radius)."""
    occ = gmsh.model.occ
    s0 = [e for e in occ.importShapes(os.path.join(geom_dir, "meridian.step")) if e[0] == 2]
    occ.synchronize()
    if len(s0) != 1:
        raise Refused("WEDGE-TOPO", "meridian.step holds %d faces, want 1" % (len(s0),))
    area = occ.getMass(2, s0[0][1])
    ybar = occ.getCenterOfMass(2, s0[0][1])[1]
    top = 2.0 * _bb(gmsh, 2, s0[0][1])[4]
    lines = [(1, occ.addLine(occ.addPoint(x, 0, 0), occ.addPoint(x, top, 0))) for x in edges[1:-1]]
    occ.fragment(s0, lines)
    occ.synchronize()
    dangling = [(1, t) for d, t in gmsh.model.getEntities(1) if len(gmsh.model.getAdjacencies(1, t)[0]) == 0]
    occ.remove(dangling, recursive=True)
    occ.synchronize()
    surfs = sorted((t for d, t in gmsh.model.getEntities(2)), key=lambda t: occ.getCenterOfMass(2, t)[0])
    if len(surfs) != len(edges) - 1:
        raise Refused("WEDGE-TOPO", "%d blocks after the split, want %d" % (len(surfs), len(edges) - 1))
    for i, s in enumerate(surfs):
        # the span from the block's corner points: a bounding box is padded, loosely on a Bezier wall
        xs = [gmsh.model.getValue(0, p[1], [])[0]
              for p in gmsh.model.getBoundary([(2, s)], combined=False, oriented=False, recursive=True)]
        if len(xs) != 4 or abs(min(xs) - edges[i]) > TOL_GEOM or abs(max(xs) - edges[i + 1]) > TOL_GEOM:
            raise Refused("WEDGE-TOPO", "block %d has corners at x %r, want 4 spanning %r..%r"
                          % (i, sorted(xs), edges[i], edges[i + 1]))
    return surfs, area, ybar


def _blocks(gmsh, surfs, edges, h1_fine):
    """Each block's two radial curves, axis curve and wall curve, and its level-0 axial count from the tau
    budget at level 2: dx <= 9 h1^2 cos^3(beta) / (tau^2 2 r sin(theta/2)), at the block's r_max and beta_max."""
    th = math.radians(RECIPE["theta_deg"])
    tau = RECIPE["tau_design"]
    out = []
    for i, s in enumerate(surfs):
        cs = [abs(c[1]) for c in gmsh.model.getBoundary([(2, s)], oriented=False)]
        rad = [c for c in cs if _bb(gmsh, 1, c)[3] - _bb(gmsh, 1, c)[0] < TOL_GEOM]
        axis = [c for c in cs if max(abs(v) for v in _bb(gmsh, 1, c)[1:3] + _bb(gmsh, 1, c)[4:6]) < TOL_GEOM]
        wall = [c for c in cs if c not in rad and c not in axis]
        if len(rad) != 2 or len(axis) != 1 or len(wall) != 1:
            raise Refused("WEDGE-TOPO", "block %d has %d radial, %d axis, %d wall curves, want 2, 1, 1"
                          % (i, len(rad), len(axis), len(wall)))
        r_max, beta = _wall_shape(gmsh, wall[0])
        dx = 9.0 * h1_fine ** 2 * math.cos(beta) ** 3 / (tau ** 2 * 2.0 * r_max * math.sin(th / 2.0))
        nb2 = math.ceil((edges[i + 1] - edges[i]) / dx)
        out.append({"surface": s, "radial": sorted(rad), "axis": axis[0], "wall": wall[0],
                    "x0": edges[i], "x1": edges[i + 1], "r_max_m": r_max,
                    "beta_max_deg": math.degrees(beta), "nb0": max(1, math.ceil(nb2 / 4))})
    return out


def _set_transfinite(gmsh, blocks, level, h1):
    """Axial curves uniform with nb0 * 2^level cells; each radial curve nr cells, geometric to the wall end
    (its end with the larger radius) with first cell h1; every block a recombined transfinite surface."""
    f = 2 ** level
    nr = max(RECIPE["nr_min"], round(RECIPE["cells_l0"] / sum(b["nb0"] for b in blocks))) * f
    done = set()
    for b in blocks:
        for c in (b["axis"], b["wall"]):
            gmsh.model.mesh.setTransfiniteCurve(c, b["nb0"] * f + 1)
        for c in b["radial"]:
            if c in done:
                continue
            done.add(c)
            e = _curve_ends(gmsh, c)
            r0, r1 = math.hypot(e[0][1], e[0][2]), math.hypot(e[1][1], e[1][2])
            q = solve_q(h1, abs(r1 - r0), nr)
            # a Progression coefficient c makes each cell c times the previous one in parameter order
            gmsh.model.mesh.setTransfiniteCurve(c, nr + 1, "Progression", q if r0 > r1 else 1.0 / q)
        gmsh.model.mesh.setTransfiniteSurface(b["surface"])
        gmsh.model.mesh.setRecombine(2, b["surface"])
    return nr


def _classify(gmsh, planes):
    """Every surface of the revolved model into one of GROUPS, or internal (a revolved interior station line)."""
    th = math.radians(RECIPE["theta_deg"])
    groups = dict((g, []) for g in GROUPS)
    internal = []
    for d, t in gmsh.model.getEntities(2):
        c = gmsh.model.occ.getCenterOfMass(2, t)
        b = _bb(gmsh, 2, t)
        phi = math.atan2(c[2], c[1])
        if abs(phi + th / 2.0) < TOL_ANGLE:
            groups["wedge_front"].append(t)
        elif abs(phi - th / 2.0) < TOL_ANGLE:
            groups["wedge_back"].append(t)
        elif b[3] - b[0] < TOL_GEOM:
            if abs(b[0] - planes["inlet"]) < TOL_GEOM:
                groups["inlet"].append(t)
            elif abs(b[0] - planes["outlet"]) < TOL_GEOM:
                groups["outlet"].append(t)
            else:
                internal.append(t)
        elif b[3] <= planes["contraction_start"] + TOL_GEOM:
            groups["slip_upstream"].append(t)
        else:
            groups["wall_nozzle"].append(t)
    n_blocks = RECIPE["k_stations"] + 2
    want = {"inlet": 1, "outlet": 1, "slip_upstream": 1, "wall_nozzle": n_blocks - 1,
            "wedge_front": n_blocks, "wedge_back": n_blocks}
    got = dict((g, len(v)) for g, v in groups.items())
    if got != want or len(internal) != n_blocks - 1:
        raise Refused("WEDGE-TOPO", "surface groups %r and %d internal, want %r and %d"
                      % (got, len(internal), want, n_blocks - 1))
    return groups


def gmsh_build(geom_dir, level, h1_fine, msh_path):
    """The gmsh child's whole job: meridian.step -> k + 2 transfinite blocks -> rotate by -theta/2 -> revolve by
    theta with one layer -> the six patch groups and the fluid volume -> MSH 4.1 ASCII at msh_path.
    Runs only in a fresh process (the gmsh-build sub-command); returns the build dict."""
    import gmsh
    if level not in RECIPE["levels"]:
        raise Refused("WEDGE-LEVEL", "level %r is not one of %r" % (level, RECIPE["levels"]))
    planes = dict((p["name"], p["x"]) for p in common.read_json(os.path.join(geom_dir, "tags.json"))["planes"])
    edges = stations(planes)
    th = math.radians(RECIPE["theta_deg"])
    h1 = h1_fine * 2.0 ** (2 - level)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.option.setString("Geometry.OCCTargetUnit", "M")
        surfs, area, ybar = _split_meridian(gmsh, geom_dir, edges)
        src = [(2, s) for s in surfs]
        gmsh.model.occ.rotate(src, 0, 0, 0, 1, 0, 0, -th / 2.0)
        gmsh.model.occ.revolve(src, 0, 0, 0, 1, 0, 0, th, numElements=[1], recombine=True)
        gmsh.model.occ.synchronize()
        blocks = _blocks(gmsh, surfs, edges, h1_fine)
        nr = _set_transfinite(gmsh, blocks, level, h1)
        groups = _classify(gmsh, planes)
        for name in GROUPS:
            gmsh.model.addPhysicalGroup(2, sorted(groups[name]), name=name)
        gmsh.model.addPhysicalGroup(3, sorted(t for d, t in gmsh.model.getEntities(3)), name="fluid")
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.model.mesh.generate(3)
        types, etags, _ = gmsh.model.mesh.getElements(3)
        elements = dict((ELEMENT_NAMES.get(int(t), "type%d" % int(t)), len(g)) for t, g in zip(types, etags))
        if sorted(elements) != ["hex", "prism"]:
            raise Refused("WEDGE-TOPO", "volume elements %r, want hex and prism only" % (elements,))
        gmsh.write(msh_path)
        version = gmsh.GMSH_API_VERSION
    finally:
        gmsh.finalize()
    f = 2 ** level
    return {"level": level, "h1_target_m": h1, "nr": nr, "nb": [b["nb0"] * f for b in blocks],
            "elements": elements, "n_cells": sum(elements.values()),
            "blocks": [dict((k, b[k]) for k in ("x0", "x1", "r_max_m", "beta_max_deg")) for b in blocks],
            "meridian": {"area_m2": area, "ybar_m": ybar}, "gmsh": version}


def load_bins(path=BIN_JSON):
    """The two binaries by name, at the absolute path bin.json records, refused unless each sha256 matches."""
    try:
        decl = common.read_json(path)
    except OSError:
        raise Refused("WEDGE-BIN", "cannot read %s" % os.path.basename(path))
    out = {}
    for name, row in decl["binaries"].items():
        p = os.path.normpath(os.path.join(common.REPO, row["path"]))
        if not os.path.isfile(p):
            raise Refused("WEDGE-BIN", "%s is missing at %s" % (name, row["path"]))
        got = common.sha256_file(p)
        if got != row["sha256"]:
            raise Refused("WEDGE-BIN", "%s sha256 %s, bin.json records %s" % (name, got, row["sha256"]))
        out[name] = p
    return out


def check_geom(geom_dir):
    """The geom dict, refused unless it declares metres at scale 1 on +x with a METRE STEP."""
    for name in ("geom.json", "tags.json", "meridian.step"):
        if not os.path.isfile(os.path.join(geom_dir, name)):
            raise Refused("WEDGE-GEOM", "%s is missing in %s" % (name, os.path.basename(geom_dir)))
    geom = common.read_json(os.path.join(geom_dir, "geom.json"))
    want = {"units": "m", "scale": 1, "axis": "+x", "step_length_unit": "METRE"}
    for field, value in want.items():
        if geom[field] != value:
            raise Refused("WEDGE-GEOM", "%s is %r, want %r" % (field, geom[field], value))
    return geom


def build_level(geom_dir, level, h1_fine, msh_path):
    """One gmsh-build child (a fresh process per build); the build dict from its last stdout line."""
    pr = subprocess.run([sys.executable, os.path.abspath(__file__), "gmsh-build", geom_dir, str(level),
                         repr(float(h1_fine)), msh_path],
                        capture_output=True, text=True, encoding="utf-8", timeout=CHILD_TIMEOUT_S,
                        env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    lines = [s for s in pr.stdout.splitlines() if s.strip()]
    line = json.loads(lines[-1]) if lines else {}
    if pr.returncode == 0:
        return line
    if pr.returncode == REFUSED_EXIT:
        raise Refused(line["refused"], line["detail"])
    raise RuntimeError("gmsh-build exited %d: %s" % (pr.returncode, pr.stderr[-600:]))


def convert(bins, msh_path, case_dir, type_args=TYPE_ARGS):
    """ofgpu-convert-mesh into a case directory that does not exist yet (the converter refuses to overwrite)."""
    if os.path.exists(case_dir):
        raise Refused("WEDGE-OUT", "%s exists; the converter refuses to overwrite a polyMesh" % case_dir)
    pr = subprocess.run([bins["ofgpu-convert-mesh"], msh_path, case_dir] + list(type_args),
                        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    if pr.returncode != 0:
        raise Refused("WEDGE-CONVERT", "exit %d: %s" % (pr.returncode, (pr.stdout + pr.stderr)[-400:]))
    return pr.stdout


def patch_types(case_dir):
    """{patch name: boundary type} read back from the written polyMesh."""
    return dict((p["name"], p["type"]) for p in
                polymesh_write.read_polymesh(os.path.join(case_dir, "constant", "polyMesh"))["patches"])


def require_wedge_types(types):
    """Refused unless the six patches are there and both wedge sides came out type wedge."""
    if sorted(types) != sorted(GROUPS) or types["wedge_front"] != "wedge" or types["wedge_back"] != "wedge":
        raise Refused("WEDGE-TYPE", "boundary is %r, want %r with both wedge sides typed wedge"
                      % (types, sorted(GROUPS)))


def run_check(bins, case_dir, cfg_path, min_tau=GC6["tau_min"]):
    """ofgpu-automesher <config> -check <case>: the summary's five numbers, the gate word and the exit code."""
    common.write_json(cfg_path, dict(CHECK_CONFIG, quality={"min_thickness_ratio": min_tau}))
    pr = subprocess.run([bins["ofgpu-automesher"], cfg_path, "-check", case_dir],
                        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    text = pr.stdout + pr.stderr
    def grab(pattern, cast):
        m = re.search(pattern, text)
        return None if m is None else cast(m.group(1))
    return {"exit": pr.returncode,
            "gate": grab(r"gate: (passed|FAILED)", str),
            "tau_min": grab(r"min tau ([0-9.eE+-]+)", float),
            "non_orth_max_deg": grab(r"non-orthogonality: max ([0-9.]+) deg", float),
            "n_regions": grab(r"regions: (\d+) ", int),
            "n_cells": grab(r"automesher: (\d+) cells", int),
            "text": text}


def measure(case_dir):
    """Areas and volume from the written polyMesh (docs/16 §I: read_polymesh + face_geometry): each patch's area is
    sum |Sf|; the volume is the divergence theorem over the boundary, sum Cf . Sf / 3; the first cell of each
    wall_nozzle face is the wall-normal distance from the face centre to the mean of its owner's other vertices."""
    pm = polymesh_write.read_polymesh(os.path.join(case_dir, "constant", "polyMesh"))
    pts, faces, own, nei = pm["points"], pm["faces"], pm["owner"], pm["neighbour"]
    volume, patches = 0.0, {}
    for p in pm["patches"]:
        area = 0.0
        for f in range(p["startFace"], p["startFace"] + p["nFaces"]):
            sf, cf = regions_check.face_geometry(pts, faces[f])
            area += float(np.linalg.norm(sf))
            volume += float(np.dot(cf, sf)) / 3.0
        patches[p["name"]] = {"type": p["type"], "n_faces": int(p["nFaces"]), "area_m2": area}
    wall = [p for p in pm["patches"] if p["name"] == "wall_nozzle"][0]
    wall_faces = range(wall["startFace"], wall["startFace"] + wall["nFaces"])
    cells = set(int(own[f]) for f in wall_faces)
    verts = {}
    for f in range(len(faces)):
        for c in ((int(own[f]), int(nei[f])) if f < len(nei) else (int(own[f]),)):
            if c in cells:
                verts.setdefault(c, set()).update(faces[f])
    h1 = []
    for f in wall_faces:
        sf, cf = regions_check.face_geometry(pts, faces[f])
        opp = sorted(verts[int(own[f])] - set(faces[f]))
        if len(opp) != 4:
            raise RuntimeError("wall face %d: its owner has %d vertices off the face, want 4" % (f, len(opp)))
        h1.append(abs(float(np.dot(pts[opp].mean(axis=0) - cf, sf / np.linalg.norm(sf)))))
    return {"n_cells": int(max(own.max(), nei.max())) + 1, "n_points": int(len(pts)), "volume_m3": volume,
            "patches": patches, "n_wall_faces": len(h1), "h1_max_m": max(h1), "h1_min_m": min(h1),
            "h1_mean_m": sum(h1) / len(h1)}


def cad_refs(geom, build):
    """The CAD-side areas and Pappus volume the mesh is judged against: revolved patches get the geom.json face
    area times theta/2pi, the wedge sides get the meridian area itself, the volume is theta A ybar."""
    k = math.radians(RECIPE["theta_deg"]) / (2 * math.pi)
    ff = geom["tags"]["fluid_faces"]
    A = build["meridian"]["area_m2"]
    yb = build["meridian"]["ybar_m"]
    areas = {"inlet": ff["inlet"]["area_m2"] * k, "outlet": ff["outlet"]["area_m2"] * k,
             "slip_upstream": ff["slip_upstream"]["area_m2"] * k,
             "wall_nozzle": (ff["wall_contraction"]["area_m2"] + ff["wall_exit"]["area_m2"]) * k,
             "wedge_front": A, "wedge_back": A}
    pappus = {"A_meridian_m2": A, "ybar_m": yb, "volume_full_m3": 2 * math.pi * A * yb,
              "brep_volume_m3": geom["stl"]["brep_volume_m3"],
              "rel": (2 * math.pi * A * yb - geom["stl"]["brep_volume_m3"]) / geom["stl"]["brep_volume_m3"]}
    return {"areas": areas, "volume_pappus_m3": math.radians(RECIPE["theta_deg"]) * A * yb, "pappus": pappus}


def judge(rep):
    """The seven GC-6 predicates plus their conjunction, each a bool; None never passes a predicate."""
    c = rep["check"]
    out = {"volume": abs(rep["volume_rel"]) <= GC6["volume_rel"],
           "areas": sorted(rep["patches"]) == sorted(GROUPS)
                    and all(p["rel"] is not None and abs(p["rel"]) <= GC6["area_rel"]
                            for p in rep["patches"].values()),
           "check": c["exit"] == 0 and c["gate"] == "passed" and c["n_regions"] == 1
                    and c["n_cells"] == rep["cells"],
           "tau": c["tau_min"] is not None and c["tau_min"] >= GC6["tau_min"],
           "first_cell": abs(rep["h1_rel"]) <= GC6["first_cell_rel"],
           "cells": abs(rep["cells_rel"]) <= GC6["cells_rel"],
           "types": rep["patches"]["wedge_front"]["type"] == "wedge"
                    and rep["patches"]["wedge_back"]["type"] == "wedge"}
    out["pass"] = all(out[k] for k in GC6_KEYS[:-1])
    return out


def level_report(build, meas, check, cad, msh_sha, polymesh_sha):
    """One level's LEVEL_KEYS row: counts, first cell, volume and areas against the CAD, the check, the sha."""
    target = RECIPE["cells_l0"] * 4 ** build["level"]
    check_row = dict((k, v) for k, v in check.items() if k != "text")
    rep = {"level": build["level"], "target_cells": target, "cells": meas["n_cells"],
           "cells_rel": (meas["n_cells"] - target) / target,
           "elements": build["elements"], "nr": build["nr"], "nb": build["nb"],
           "h1_target_m": build["h1_target_m"], "h1_max_m": meas["h1_max_m"], "h1_min_m": meas["h1_min_m"],
           "h1_mean_m": meas["h1_mean_m"], "h1_rel": (meas["h1_max_m"] - build["h1_target_m"]) / build["h1_target_m"],
           "volume_m3": meas["volume_m3"], "volume_pappus_m3": cad["volume_pappus_m3"],
           "volume_rel": (meas["volume_m3"] - cad["volume_pappus_m3"]) / cad["volume_pappus_m3"],
           "patches": dict((name, {"type": p["type"], "n_faces": p["n_faces"], "area_m2": p["area_m2"],
                                   "cad_area_m2": cad["areas"].get(name),
                                   "rel": (None if cad["areas"].get(name) is None
                                           else (p["area_m2"] - cad["areas"][name]) / cad["areas"][name])})
                           for name, p in meas["patches"].items()),
           "check": check_row, "msh_sha256": msh_sha, "polymesh_sha256": dict(polymesh_sha)}
    rep["gc6"] = judge(rep)
    return rep


def run(geom_dir, out_dir, h1_fine=H1_FINE_NOMINAL, levels=(0, 1, 2)):
    """All levels in one go: build, convert, check, measure, judge; ok with the report or refused by id.
    A refusal is returned, never raised; any other exception propagates."""
    try:
        return _run(geom_dir, out_dir, h1_fine, levels)
    except Refused as r:
        return {"status": "refused", "rule": r.rule, "message": r.detail, "report": None}


def _run(geom_dir, out_dir, h1_fine, levels):
    """run's body: raises Refused at the first refusal."""
    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise Refused("WEDGE-OUT", "%s exists and is not empty" % out_dir)
    os.makedirs(out_dir, exist_ok=True)
    bins = load_bins()
    geom = check_geom(geom_dir)
    levels_out, cad0 = [], None
    for level in levels:
        ld = os.path.join(out_dir, "L%d" % level)
        os.makedirs(ld)
        build = build_level(geom_dir, level, h1_fine, os.path.join(ld, "wedge.msh"))
        msh_sha = common.sha256_file(os.path.join(ld, "wedge.msh"))
        convert(bins, os.path.join(ld, "wedge.msh"), os.path.join(ld, "case"))
        require_wedge_types(patch_types(os.path.join(ld, "case")))
        polymesh_sha = {}
        for pm_name in ("boundary", "faces", "neighbour", "owner", "points"):
            pm_snap = common.stable_file_snapshot(
                os.path.join(ld, "case", "constant", "polyMesh", pm_name))
            if pm_snap["stable"] is not True:
                raise Refused("WEDGE-CONVERT", "polyMesh/%s is not a stable regular file" % pm_name)
            polymesh_sha[pm_name] = pm_snap["sha256"]
        check = run_check(bins, os.path.join(ld, "case"), os.path.join(ld, "check_config.json"))
        common.atomic_write(os.path.join(ld, "check.txt"), check["text"])
        meas = measure(os.path.join(ld, "case"))
        cad = cad_refs(geom, build)
        if cad0 is None:
            cad0 = cad
            if abs(cad["pappus"]["rel"]) > PAPPUS_TOL:
                raise Refused("WEDGE-GEOM", "Pappus 2 pi A ybar against the BREP volume is %r, over %r"
                              % (cad["pappus"]["rel"], PAPPUS_TOL))
        levels_out.append(level_report(build, meas, check, cad, msh_sha, polymesh_sha))
    decl = common.read_json(BIN_JSON)
    rep = {"version": 1, "recipe": RECIPE, "recipe_sha": common.sha256_of(RECIPE), "gc6_tolerances": GC6,
           "geom_sha256": common.sha256_file(os.path.join(geom_dir, "geom.json")),
           "template_sha": geom["template_sha"], "params_sha": geom["params_sha"],
           "bins": dict((name, row["sha256"]) for name, row in decl["binaries"].items()),
           "h1_fine_m": float(h1_fine), "pappus": cad0["pappus"], "levels": levels_out,
           "gc6_pass": all(lv["gc6"]["pass"] for lv in levels_out)}
    common.atomic_write(os.path.join(out_dir, "wedge_mesh.json"), common.canonical_json(rep) + common.NL)
    return {"status": "ok", "rule": None, "message": "", "report": rep}


def main(argv):
    """--selftest | gmsh-build GEOM_DIR LEVEL H1_FINE MSH_PATH | run GEOM_DIR OUT_DIR [H1_FINE]."""
    if argv == ["--selftest"]:
        try:
            selftest()
        except Exception:
            import traceback
            traceback.print_exc()
            return 1
        return 0
    if len(argv) == 5 and argv[0] == "gmsh-build":
        try:
            build = gmsh_build(argv[1], int(argv[2]), float(argv[3]), argv[4])
        except Refused as r:
            print(common.canonical_json({"refused": r.rule, "detail": r.detail}))
            return REFUSED_EXIT
        print(common.canonical_json(build))
        return 0
    if len(argv) in (3, 4) and argv[0] == "run":
        res = run(os.path.abspath(argv[1]), os.path.abspath(argv[2]),
                  float(argv[3]) if len(argv) == 4 else H1_FINE_NOMINAL)
        if res["status"] != "ok":
            print(common.canonical_json({"status": res["status"], "rule": res["rule"], "message": res["message"]}))
            return 2
        for lv in res["report"]["levels"]:
            tau = lv["check"]["tau_min"]
            print("L%d cells %d tau %.6f h1_rel %+.4f volume_rel %+.6f gc6 %s"
                  % (lv["level"], lv["cells"], float("nan") if tau is None else tau, lv["h1_rel"], lv["volume_rel"],
                     "pass" if lv["gc6"]["pass"] else "FAIL"))
        print("GC-6 %s" % ("pass" if res["report"]["gc6_pass"] else "FAIL"))
        return 0 if res["report"]["gc6_pass"] else 1
    sys.stderr.write(USAGE + common.NL)
    return 2


def selftest():
    """The nominal export becomes the three-level wedge; GC-6, determinism, the gates and the refusals prove out."""
    import time
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        import export

        # T1: bin.json loads, and a wrong sha refuses WEDGE-BIN
        bins = load_bins()
        assert sorted(bins) == ["ofgpu-automesher", "ofgpu-convert-mesh"], sorted(bins)
        for name, p in bins.items():
            assert os.path.isfile(p), (name, p)
        bad = os.path.join(td, "bin_bad.json")
        decl = common.read_json(BIN_JSON)
        decl["binaries"]["ofgpu-convert-mesh"]["sha256"] = "0" * 64
        common.write_json(bad, decl)
        try:
            load_bins(bad)
            raise AssertionError("a wrong convert sha did not refuse")
        except Refused as r:
            assert r.rule == "WEDGE-BIN", r.rule
        print("[ok] bin.json: 2 binaries at the recorded sha256; a wrong sha refuses WEDGE-BIN")

        # T2: the nominal export
        geom_dir = os.path.join(td, "geom")
        res = export.run_pipeline(export.TEMPLATE, dict(export.NOMINAL), geom_dir)
        assert res["status"] == "ok", (res["status"], res["rule"], res["message"])
        print("[ok] nominal export ok")

        # T3: GC-6 at the three levels, with the pinned counts
        PIN = {0: (15, 676, 9464, 676), 1: (30, 1352, 39208, 1352), 2: (60, 2704, 159536, 2704)}
        res = run(geom_dir, os.path.join(td, "out"))
        assert res["status"] == "ok", (res["status"], res["rule"], res["message"])
        rep = res["report"]
        for lv in rep["levels"]:
            n = lv["level"]
            bad_pred = [k for k, v in lv["gc6"].items() if not v and k != "pass"]
            assert lv["gc6"]["pass"], (n, bad_pred, common.canonical_json(lv))
            got = (lv["nr"], sum(lv["nb"]), lv["elements"]["hex"], lv["elements"]["prism"])
            assert got == PIN[n], (n, got, PIN[n], common.canonical_json(lv))
            assert lv["cells"] == lv["elements"]["hex"] + lv["elements"]["prism"], common.canonical_json(lv)
            worst = max(abs(p["rel"]) for p in lv["patches"].values())
            print("[ok] L%d: %d cells (hex %d, prism %d), tau %.6f >= 0.05, first cell %+.4f, volume %+.5f, "
                  "worst area %.5f, GC-6 pass"
                  % (n, lv["cells"], lv["elements"]["hex"], lv["elements"]["prism"], lv["check"]["tau_min"],
                     lv["h1_rel"], lv["volume_rel"], worst))

        # T4: Pappus against the BREP volume
        assert abs(rep["pappus"]["rel"]) <= PAPPUS_TOL, rep["pappus"]["rel"]
        for lv in rep["levels"]:
            assert lv["patches"]["wedge_front"]["cad_area_m2"] == rep["pappus"]["A_meridian_m2"], lv["level"]
        print("[ok] Pappus: 2 pi A ybar of the meridian equals the BREP volume (rel %.1e)" % rep["pappus"]["rel"])

        # T5: MSH bytes identical across two fresh processes, and equal to the recorded sha
        det = os.path.join(td, "det")
        os.makedirs(det)
        for lv in rep["levels"]:
            n = lv["level"]
            pa = os.path.join(det, "L%d_a.msh" % n)
            pb = os.path.join(det, "L%d_b.msh" % n)
            build_level(geom_dir, n, H1_FINE_NOMINAL, pa)
            build_level(geom_dir, n, H1_FINE_NOMINAL, pb)
            sa, sb = common.sha256_file(pa), common.sha256_file(pb)
            assert sa == sb == lv["msh_sha256"], (n, sa, sb, lv["msh_sha256"])
        print("[ok] MSH bytes identical across two fresh processes at L0, L1, L2")

        # T6: the report is canonical, path-free, exactly the declared keys; usage exits 2
        path = os.path.join(td, "out", "wedge_mesh.json")
        with open(path, "rb") as f:
            blob = f.read()
        doc = common.read_json(path)
        assert blob == (common.canonical_json(doc) + common.NL).encode("ascii"), "wedge_mesh.json is not canonical"
        assert sorted(doc) == sorted(REPORT_KEYS), sorted(doc)
        for lv in doc["levels"]:
            assert sorted(lv) == sorted(LEVEL_KEYS), sorted(lv)
        text = blob.decode("ascii")
        for poison in (td, os.path.abspath(td), td.replace(chr(92), "/"), td.replace(chr(92), chr(92) * 2), "C:"):
            assert poison not in text, poison
        assert main([]) == 2 and main(["run"]) == 2, "usage must exit 2"
        print("[ok] wedge_mesh.json canonical and path-free; usage exits 2")

        # T6b: every level row's polymesh_sha256 equals the five written polyMesh files
        for lv in doc["levels"]:
            n = lv["level"]
            want = dict((nm, common.sha256_file(os.path.join(td, "out", "L%d" % n, "case",
                                                              "constant", "polyMesh", nm)))
                        for nm in ("boundary", "faces", "neighbour", "owner", "points"))
            assert lv["polymesh_sha256"] == want, n
        print("[ok] polymesh_sha256 of L0, L1, L2 equals the five written polyMesh files")

        # T7: each GC-6 predicate fails on its own planted miss, alone
        base = rep["levels"][0]
        assert judge(base)["pass"] is True
        misses = [("volume", ["volume_rel"], 0.0031), ("areas", ["patches", "inlet", "rel"], 0.0051),
                  ("check", ["check", "exit"], 1), ("tau", ["check", "tau_min"], 0.0499),
                  ("first_cell", ["h1_rel"], 0.051), ("cells", ["cells_rel"], 0.151),
                  ("types", ["patches", "wedge_front", "type"], "patch")]
        for rule, path_keys, value in misses:
            m = json.loads(json.dumps(base))
            node = m
            for k in path_keys[:-1]:
                node = node[k]
            node[path_keys[-1]] = value
            j = judge(m)
            false_pred = [k for k, v in j.items() if not v and k != "pass"]
            assert j["pass"] is False and false_pred == [rule], (rule, false_pred)
        print("[ok] judge: each of 7 planted misses fails exactly its GC-6 predicate")

        # T8: the binary's own gate can fail (a stricter threshold exits 1 with gate FAILED)
        c = run_check(load_bins(), os.path.join(td, "out", "L0", "case"), os.path.join(td, "strict.json"),
                      min_tau=0.2)
        assert c["exit"] == 1 and c["gate"] == "FAILED", (c["exit"], c["gate"])
        m = json.loads(json.dumps(base))
        m["check"] = dict((k, v) for k, v in c.items() if k != "text")
        assert judge(m)["check"] is False, "a FAILED gate did not fail the check predicate"
        print("[ok] -check can fail: min_thickness_ratio 0.2 on L0 exits 1, gate FAILED")

        # T9: the plan's glob spelling is silently not a glob here: caught as WEDGE-TYPE
        os.makedirs(os.path.join(td, "glob"))
        convert(load_bins(), os.path.join(td, "out", "L0", "wedge.msh"), os.path.join(td, "glob", "case"),
                type_args=["-type", "wedge_*=wedge"])
        types = patch_types(os.path.join(td, "glob", "case"))
        assert types["wedge_front"] == "patch", types
        try:
            require_wedge_types(types)
            raise AssertionError("a patch-typed wedge side did not refuse")
        except Refused as r:
            assert r.rule == "WEDGE-TYPE", r.rule
        print("[ok] the plan's -type wedge_*=wedge leaves wedge_front a patch: refused WEDGE-TYPE")

        # T10: five refusals, each by its exact id
        def refused_by(fn, rule):
            # run() returns its refusal (never raises it); build_level raises Refused
            if fn.__name__ == "build":
                try:
                    fn()
                except Refused as r:
                    assert r.rule == rule, (rule, r.rule, r.detail)
                    return
                raise AssertionError("expected %s, got none" % rule)
            got = fn()
            assert got["status"] == "refused" and got["rule"] == rule and got["report"] is None, (rule, got)
        refused_by(lambda: run(geom_dir, os.path.join(td, "out")), "WEDGE-OUT")

        def make_g(name, files, units=None):
            gd = os.path.join(td, name)
            os.makedirs(gd)
            for f in files:
                shutil.copy(os.path.join(geom_dir, f), os.path.join(gd, f))
            if units is not None:
                g = common.read_json(os.path.join(gd, "geom.json"))
                g["units"] = units
                common.write_json(os.path.join(gd, "geom.json"), g)
            return gd
        refused_by(lambda: run(make_g("g_mm", ("geom.json", "tags.json", "meridian.step"), units="mm"),
                               os.path.join(td, "o_mm")), "WEDGE-GEOM")
        refused_by(lambda: run(make_g("g_nomer", ("geom.json", "tags.json")),
                               os.path.join(td, "o_nomer")), "WEDGE-GEOM")
        refused_by(lambda: run(geom_dir, os.path.join(td, "o_grade"), h1_fine=1e-3, levels=(0,)), "WEDGE-GRADE")

        def build():
            return build_level(geom_dir, 3, H1_FINE_NOMINAL, os.path.join(td, "o_lvl.msh"))
        refused_by(build, "WEDGE-LEVEL")
        print("[ok] refusals: WEDGE-OUT, WEDGE-GEOM (units), WEDGE-GEOM (missing meridian.step), "
              "WEDGE-GRADE, WEDGE-LEVEL")

    print("selftest wall %.1f s" % (time.time() - t0))
    print("SELFTEST PASS")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
