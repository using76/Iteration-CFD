#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.

"""The drone showreel's forward-flight GPU case: write, run, post.

Writes the 15 m/s forward-flight case on the PX4 x500 free-flight tunnel
mesh (tools/drone/tunnel_mesh.py, variant "forward": the airframe sits
untransformed in its PX4 FLU frame flying +x at level attitude, and the
freestream U = (-15, 0, 0) m/s enters through the inlet patch at xMax)
and drives the pinned GPU solver ``ofgpu-lowmach`` with the four rotors
as prescribed axial-thrust actuator disks: a disk is the annulus of cells
whose centres satisfy hub_radius <= r <= radius within half the blade
thickness of the disk plane, tiled by disjoint axis-aligned boxes (the
solver selects source cells by centre-in-box only, so each box carries a
uniform ``momentumSource`` body force on U whose volume integral is the
rotor's thrust share, pushing the fluid against the disk normal - the
drone gets +normal thrust; no swirl). The cell centres are computed by
the solver's own face-centroid rule (area-weighted triangle fan,
SPEC-LIT §18), so the boxes select the same cells in ``ofgpu-lowmach``
as they do here. The pinned binary reads
``constant/fvSources`` (SPEC-LIT §18); its fan curve / porous jump acts
only on boundary patches of the datacentre binary, so a momentum source
is the one rotor model it can apply to this baffle-free mesh.

The mesh and everything made from it derive from the BSD-3-Clause PX4
x500 model ("(c) 2022 Rudis Laboratories / PX4 Autopilot for Drones"), so
NOTHING is written inside the repository: every output goes under --out,
which must lie outside the repository, with the model's LICENSE.txt,
LICENSE and rotors.json copied beside it.

Subcommands: write (case files, CPU only), run (write + sha256 pin check
+ GPU preflight + launch + post), post (solve.json, residuals.csv,
forces.csv and the rotor gate from an existing out). --selftest runs
T1-T11 on the CPU.
"""

import argparse
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile
import time
import types

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOL = "tools/drone/solve.py"
VERSION = "drone-solve/1"
CREDIT = "(c) 2022 Rudis Laboratories / PX4 Autopilot for Drones, BSD-3-Clause"
PATCHES = ["inlet", "outlet", "side_ymin", "side_ymax", "bottom", "top",
           "body", "arms", "motors", "skids"]
FARFIELD = ["side_ymin", "side_ymax", "bottom", "top"]   # slip, like the promo's SLIP
WALLS = ["body", "arms", "motors", "skids"]              # noSlip + wall functions
MASS_KG = 2.0 + 4 * 0.016076923076923075                 # model.sdf base_link + 4 rotors
G = 9.80665
THRUST_N_DEFAULT = MASS_KG * G / 4                       # 5.060985757692308 N per disk
BOX_EPS = 1e-6                                           # m
THRUST_TOL = 1e-3                                        # relative, applied vs target
MIN_FREE_MIB = 8000                                      # drone preflight (1.29 M cells)
BIN = "rust/target/gpu-pin/ofgpu-lowmach.exe"
BIN_SHA256 = "50471caaa54125e0c2eee4fb34eebdfc2da44727d2b818a2b96bb4234c02103c"
ROTOR_MODEL = "axial momentum source, uniform over the annulus cells, no swirl"


def load_promo() -> types.ModuleType:
    """R1: reuse the promo solve helpers by import; NEVER import solve by name."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.normpath(os.path.join(here, "..", "promo", "solve.py"))
    spec = importlib.util.spec_from_file_location("promo_solve", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PROMO = load_promo()


def _num(v) -> bool:
    """True for a real number (bools are not numbers here)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _num3(v) -> bool:
    """True for a list of exactly 3 real numbers."""
    return isinstance(v, list) and len(v) == 3 and all(_num(t) for t in v)


# ------------------------------------------------------------ rotor model

def disk_cells(centres: np.ndarray, rotor: dict) -> np.ndarray:
    """R2: the annulus cells by centre - hub_radius <= r <= radius and
    |cz - zc| <= thickness/2; the normal must be (0, 0, +1) or (0, 0, -1)."""
    nx, ny, nz = (float(v) for v in rotor["normal"])
    if (abs(nx) > 1e-12 or abs(ny) > 1e-12
            or (abs(nz - 1.0) > 1e-12 and abs(nz + 1.0) > 1e-12)):
        PROMO.refuse("DS-ROTOR", "rotor " + str(rotor.get("name", "?")) +
                     ": the disk normal must be (0, 0, +1) or (0, 0, -1), got (" +
                     repr(nx) + " " + repr(ny) + " " + repr(nz) + ")")
    c = rotor["centre_m"]
    d = centres - np.array([float(c[0]), float(c[1]), float(c[2])])
    r = np.hypot(d[:, 0], d[:, 1])
    mask = ((r >= float(rotor["hub_radius_m"])) & (r <= float(rotor["radius_m"]))
            & (np.abs(d[:, 2]) <= float(rotor["thickness_m"]) / 2.0))
    return np.flatnonzero(mask).astype(np.int64)


def box_select(centres: np.ndarray, lo, hi) -> np.ndarray:
    """R3: the solver's inclusive centre-in-box rule (read_selector in
    sources.rs: lo <= c <= hi on all three axes, both ends inclusive)."""
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    return np.flatnonzero(np.all((centres >= lo) & (centres <= hi), axis=1)).astype(np.int64)


def _run_box(centres, run, ylo, yhi, zlo, zhi, eps):
    """One box over a maximal run: x spans the run's centres +/- eps."""
    xs = centres[run, 0]
    return ((float(xs.min()) - eps, ylo, float(zlo)),
            (float(xs.max()) + eps, yhi, float(zhi)))


def disk_boxes(centres: np.ndarray, selected: np.ndarray, zlo: float, zhi: float,
               eps: float = BOX_EPS) -> list:
    """R3: tile exactly the selected set with disjoint axis-aligned boxes:
    y-clusters of the selected centres (gap > 2 eps), each cluster's band
    walked in x, each maximal run of selected cells one box."""
    ys = centres[selected, 1]
    order = np.argsort(ys, kind="stable")
    ys_sorted = ys[order]
    cuts = np.flatnonzero(np.diff(ys_sorted) > 2.0 * eps) + 1
    sel = np.zeros(centres.shape[0], dtype=bool)
    sel[selected] = True
    inz = (centres[:, 2] >= zlo) & (centres[:, 2] <= zhi)
    boxes = []
    for vals in np.split(ys_sorted, cuts):
        ylo = float(vals[0]) - eps
        yhi = float(vals[-1]) + eps
        band = np.flatnonzero(inz & (centres[:, 1] >= ylo) & (centres[:, 1] <= yhi))
        band = band[np.argsort(centres[band, 0], kind="stable")]
        run = []
        for ci in band:
            if sel[ci]:
                run.append(int(ci))
            elif run:
                boxes.append(_run_box(centres, run, ylo, yhi, zlo, zhi, eps))
                run = []
        if run:
            boxes.append(_run_box(centres, run, ylo, yhi, zlo, zhi, eps))
    return boxes


def verify_boxes(centres: np.ndarray, selected: np.ndarray, boxes: list) -> dict:
    """R3: prove the boxes select exactly `selected` under box_select and
    no cell lies in more than one box."""
    n = centres.shape[0]
    if boxes:
        los = np.array([b[0] for b in boxes], dtype=np.float64)
        his = np.array([b[1] for b in boxes], dtype=np.float64)
        sub = np.flatnonzero(np.all((centres >= los.min(axis=0))
                                    & (centres <= his.max(axis=0)), axis=1)).astype(np.int64)
    else:
        sub = np.zeros(0, dtype=np.int64)
    counts = np.zeros(sub.size, dtype=np.int64)
    union = np.zeros(sub.size, dtype=bool)
    sub_centres = centres[sub]
    for lo, hi in boxes:
        s = box_select(sub_centres, lo, hi)
        counts[s] += 1
        union[s] = True
    want = np.zeros(n, dtype=bool)
    want[selected] = True
    got = np.zeros(n, dtype=bool)
    if sub.size:
        got[sub[union]] = True
    return {"union_equal": bool(np.array_equal(want, got)),
            "max_overlap": int(counts.max()) if counts.size else 0,
            "n_boxes": int(len(boxes))}


def face_centroids(pm, start: int, count: int) -> tuple:
    """(Sf (n,3), Cf (n,3)) by the solver's own face-centroid rule
    (rust/src/mesh/geometry.rs face_geometry): x_avg is the vertex mean;
    the face fans into triangles (x_avg, P_i, P_{i+1}) with t_n their
    cross product, t_a = |t_n|/2 and t_c the triangle centroid; then
    Sf = sum t_n / 2 and Cf = sum(t_c t_a)/sum(t_a), falling back to
    x_avg when sum(t_a) <= 1e-150 or the face has fewer than 3 vertices.
    Same CSR walk as the promo's face_sf_centres, no Python loop."""
    if count == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    lo = int(pm.ptr[start])
    hi = int(pm.ptr[start + count])
    coords = pm.points[pm.flat[lo:hi]]
    cnt = pm.arity[start:start + count].astype(np.int64)
    lp = pm.ptr[start:start + count + 1] - lo
    x_avg = np.add.reduceat(coords, lp[:-1], axis=0) / cnt[:, None]
    rep = np.repeat(lp[:-1], cnt)
    local = np.arange(coords.shape[0], dtype=np.int64) - rep
    nxt = rep + (local + 1) % np.repeat(cnt, cnt)
    cc = np.repeat(x_avg, cnt, axis=0)
    t_n = np.cross(coords - cc, coords[nxt] - cc)
    t_a = 0.5 * np.linalg.norm(t_n, axis=1)
    t_c = (cc + coords + coords[nxt]) / 3.0
    sf = 0.5 * np.add.reduceat(t_n, lp[:-1], axis=0)
    sum_a = np.add.reduceat(t_a, lp[:-1], axis=0)
    num = np.add.reduceat(t_c * t_a[:, None], lp[:-1], axis=0)
    cf = x_avg.copy()
    good = (cnt >= 3) & (sum_a > 1e-150)
    cf[good] = num[good] / sum_a[good, None]
    return sf, cf


def cell_geometry(pm, owner: np.ndarray, neighbour: np.ndarray, n_cells: int) -> tuple:
    """R4: cell volumes and centroids from a polyMesh - the textbook
    polyhedron sums, vectorised (no Python loop over cells). The face
    centres follow the solver's own face-centroid rule (the area-weighted
    triangle fan of face_centroids), not the vertex mean, so on the
    octree mesh's hanging-node polygons the cell centroids are the ones
    ``ofgpu-lowmach`` itself sees and the rotor boxes select the same
    cells there."""
    nf = int(pm.arity.size)
    sf, xf = face_centroids(pm, 0, nf)
    own = owner.astype(np.int64)
    nb = neighbour.astype(np.int64)
    xn = xf[:nb.size]
    sn = sf[:nb.size]
    sumc = np.zeros((n_cells, 3))
    cnt = np.zeros(n_cells)
    np.add.at(sumc, own, xf)
    np.add.at(cnt, own, 1.0)
    np.add.at(sumc, nb, xn)
    np.add.at(cnt, nb, 1.0)
    a = sumc / cnt[:, None]
    V = np.zeros(n_cells)
    d = (sf * xf).sum(axis=1) / 3.0
    np.add.at(V, own, d)
    np.add.at(V, nb, -d[:nb.size])
    num = np.zeros((n_cells, 3))
    ao = a[own]
    vo = (sf * (xf - ao)).sum(axis=1) / 3.0
    np.add.at(num, own, (ao + 0.75 * (xf - ao)) * vo[:, None])
    an = a[nb]
    vn = -(sn * (xn - an)).sum(axis=1) / 3.0
    np.add.at(num, nb, (an + 0.75 * (xn - an)) * vn[:, None])
    return V, num / V[:, None]


def body_force_z(thrust_n: float, rho: float, volume_m3: float, normal_z: float) -> float:
    """R5: the fluid is pushed AGAINST the disk normal, so the drone gets
    +normal thrust from the reaction."""
    return -normal_z * thrust_n / (rho * volume_m3)


def fv_sources_text(entries: list) -> str:
    """R5: constant/fvSources - one momentumSource box block per entry."""
    fnum = PROMO.fnum
    text = PROMO.foam_header("fvSources", "dictionary", "constant")
    for name, b, lo, hi in entries:
        text += name + "\n{\n"
        text += "    " + "type".ljust(12) + "momentumSource;\n"
        text += "    " + "field".ljust(12) + "U;\n"
        text += ("    " + "bodyForce".ljust(12) + "(" + fnum(b[0]) + " " +
                 fnum(b[1]) + " " + fnum(b[2]) + ");\n")
        text += "    " + "selection".ljust(12) + "box;\n"
        text += ("    " + "min".ljust(12) + "(" + fnum(lo[0]) + " " +
                 fnum(lo[1]) + " " + fnum(lo[2]) + ");\n")
        text += ("    " + "max".ljust(12) + "(" + fnum(hi[0]) + " " +
                 fnum(hi[1]) + " " + fnum(hi[2]) + ");\n")
        text += "}\n"
    return text


def parse_source_lines(text: str) -> dict:
    """R6: the solver's per-source start-up lines -> {name: {n, V, b}}."""
    out = {}
    head = 'source on U: "'
    for line in text.splitlines():
        if not line.startswith(head):
            continue
        try:
            rest = line[len(head):]
            k = rest.index('": box ')
            name = rest[:k]
            body = rest[k + len('": box '):]
            for _ in range(2):
                a = body.index("(")
                b = body.index(")", a)
                body = body[b + 1:]
            body = body[body.index(" -> ") + len(" -> "):]
            n = int(body[:body.index(" cells,")])
            body = body[body.index(" cells,") + len(" cells,"):]
            v = float(body[:body.index(" m3")])
            p = body.index("body force (")
            q = body.index(")", p)
            bf = [float(t) for t in body[p + len("body force ("):q].split()]
        except (ValueError, IndexError):
            continue
        out[name] = {"n": n, "V": v, "b": bf}
    return out


def rotor_gate(rotors_out: list, rho: float) -> dict:
    """R6: applied vs target thrust per rotor from the solver's own source
    lines (each rotor dict carries its `source_text` and `thrust_target_N`)."""
    entries = []
    bare = bool(rotors_out) and all(float(r.get("thrust_target_N", 0.0)) == 0.0
                                    for r in rotors_out)
    for r in rotors_out:
        parsed = parse_source_lines(r.get("source_text") or "")
        pref = "disk" + str(r["index"]) + "_b"
        mine = {nm: d for nm, d in parsed.items() if nm.startswith(pref)}
        n_solver = int(sum(d["n"] for d in mine.values()))
        v_solver = float(sum(d["V"] for d in mine.values()))
        target = float(r.get("thrust_target_N", 0.0))
        applied = rho * abs(float(r.get("bodyforce_z", 0.0))) * v_solver
        rel = abs(applied / target - 1.0) if target != 0.0 else 0.0
        count_match = bool(n_solver == int(r["n_cells_tool"]))
        has_lines = bool(mine)
        entries.append({"index": r["index"], "name": r.get("name"),
                        "n_cells_solver": n_solver, "V_solver_m3": v_solver,
                        "n_boxes_solver": len(mine), "thrust_applied_N": applied,
                        "rel_err": rel, "count_match": count_match,
                        "has_lines": has_lines,
                        "pass": bool(count_match and has_lines and rel <= THRUST_TOL)})
    return {"pass": bool(bare or all(e["pass"] for e in entries)),
            "bare": bool(bare), "rotors": entries}


def boundary_entries(names, U_vec, k_in, omega_in) -> dict:
    """R7: field -> patch -> [(key, value)] for the ten forward patches."""
    fnum = PROMO.fnum
    uvec = "uniform (" + fnum(U_vec[0]) + " " + fnum(U_vec[1]) + " " + fnum(U_vec[2]) + ")"
    fields = {f: {} for f in ("U", "p", "T", "k", "omega", "nut")}
    for nm in names:
        if nm == "inlet":
            fields["U"][nm] = [("type", "fixedValue"), ("value", uvec)]
            fields["p"][nm] = [("type", "zeroGradient")]
            fields["T"][nm] = [("type", "fixedValue"), ("value", "uniform " + fnum(PROMO.T_K))]
            fields["k"][nm] = [("type", "fixedValue"), ("value", "uniform " + fnum(k_in))]
            fields["omega"][nm] = [("type", "fixedValue"),
                                   ("value", "uniform " + fnum(omega_in))]
            fields["nut"][nm] = [("type", "calculated"), ("value", "uniform 0")]
        elif nm == "outlet":
            fields["U"][nm] = [("type", "inletOutlet"), ("inletValue", "uniform (0 0 0)"),
                               ("value", uvec)]
            fields["p"][nm] = [("type", "fixedValue"), ("value", "uniform 0")]
            fields["T"][nm] = [("type", "zeroGradient")]
            fields["k"][nm] = [("type", "zeroGradient")]
            fields["omega"][nm] = [("type", "zeroGradient")]
            fields["nut"][nm] = [("type", "calculated"), ("value", "uniform 0")]
        elif nm in FARFIELD:
            for f in fields:
                fields[f][nm] = PROMO._slip()
        else:  # WALLS: noSlip + wall functions
            fields["U"][nm] = [("type", "noSlip")]
            fields["p"][nm] = [("type", "zeroGradient")]
            fields["T"][nm] = [("type", "zeroGradient")]
            fields["k"][nm] = [("type", "kqRWallFunction"), ("value", "uniform " + fnum(k_in))]
            fields["omega"][nm] = [("type", "omegaWallFunction"),
                                   ("value", "uniform " + fnum(omega_in))]
            fields["nut"][nm] = [("type", "nutkWallFunction"), ("value", "uniform 0")]
    return fields


# ---------------------------------------------------------------- refusals

def _check_thrust(thrust_n) -> float:
    """DS-THRUST: finite and >= 0 (0 is the bare airframe)."""
    t = float(thrust_n)
    if not math.isfinite(t) or t < 0.0:
        PROMO.refuse("DS-THRUST", "thrust " + repr(t) + " N: must be finite and >= 0")
    return t


def _check_out(out: str) -> None:
    """DS-OUT: outside the repository and not already occupied."""
    try:
        rel = os.path.relpath(out, REPO)
    except ValueError:
        rel = ".."
    if rel != ".." and not rel.startswith(".." + os.sep):
        PROMO.refuse("DS-OUT", out + ": --out resolves inside the repository")
    if os.path.isdir(out):
        if os.listdir(out):
            PROMO.refuse("DS-OUT", out + ": exists and is not empty")
    elif os.path.exists(out):
        PROMO.refuse("DS-OUT", out + ": exists and is not a directory")


def _check_mesh_files(mesh: str) -> str:
    """DS-MESH (existence): the five polyMesh files and the metadata."""
    pm_dir = os.path.join(mesh, "case", "constant", "polyMesh")
    five = ["boundary", "faces", "owner", "neighbour", "points"]
    missing = [f for f in five if not os.path.isfile(os.path.join(pm_dir, f))]
    if missing:
        PROMO.refuse("DS-MESH", pm_dir + ": missing " + ", ".join(missing))
    for f in ("tunnel_mesh.json", "rotors.json", "LICENSE.txt"):
        if not os.path.isfile(os.path.join(mesh, f)):
            PROMO.refuse("DS-MESH", mesh + ": missing " + f)
    return pm_dir


def _load_rotors(mesh: str) -> list:
    """DS-ROTOR: exactly 4 disks, +-z normals, radius > hub, thickness > 0."""
    path = os.path.join(mesh, "rotors.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as exc:
        PROMO.refuse("DS-ROTOR", path + ": unreadable (" + str(exc) + ")")
    rotors = doc.get("rotors") if isinstance(doc, dict) else None
    if not isinstance(rotors, list) or len(rotors) != 4:
        PROMO.refuse("DS-ROTOR", path + ": rotors must be a list of exactly 4 entries")
    for r in rotors:
        ok = (isinstance(r, dict) and "index" in r and _num3(r.get("centre_m"))
              and _num3(r.get("normal")))
        if ok:
            nx, ny, nz = (float(v) for v in r["normal"])
            ok = (abs(nx) <= 1e-12 and abs(ny) <= 1e-12
                  and (abs(nz - 1.0) <= 1e-12 or abs(nz + 1.0) <= 1e-12)
                  and _num(r.get("radius_m")) and _num(r.get("hub_radius_m"))
                  and _num(r.get("thickness_m"))
                  and float(r["radius_m"]) > float(r["hub_radius_m"])
                  and float(r["thickness_m"]) > 0.0)
        if not ok:
            PROMO.refuse("DS-ROTOR", path + ": every rotor needs index, centre_m"
                                            " (3 numbers), a (0, 0, +1) or (0, 0, -1)"
                                            " normal, thickness_m > 0 and radius_m >"
                                            " hub_radius_m")
    return sorted(rotors, key=lambda r: r["index"])


def _load_mesh(mesh: str):
    """Files, variant, patch set and rotors, in refusal order (R8)."""
    pm_dir = _check_mesh_files(mesh)
    tm_path = os.path.join(mesh, "tunnel_mesh.json")
    try:
        with open(tm_path, "r", encoding="utf-8") as fh:
            tm = json.load(fh)
    except (OSError, ValueError) as exc:
        PROMO.refuse("DS-MESH", tm_path + ": unreadable (" + str(exc) + ")")
    variant = tm.get("variant") if isinstance(tm, dict) else None
    if variant != "forward":
        PROMO.refuse("DS-VARIANT", tm_path + ': variant must be "forward", got ' +
                     repr(variant))
    boundary = PROMO.read_boundary(os.path.join(pm_dir, "boundary"))
    names = [b[0] for b in boundary]
    if sorted(names) != sorted(PATCHES):
        PROMO.refuse("DS-MESH", "boundary patches " + ",".join(names) +
                     " are not exactly the 10 expected")
    rotors = _load_rotors(mesh)
    return pm_dir, tm, names, boundary, rotors


def _verify_or_refuse(centres, selected, boxes, rotor) -> dict:
    """DS-BOXES: the boxes must reproduce the annulus exactly, once each."""
    v = verify_boxes(centres, selected, boxes)
    if not v["union_equal"] or v["max_overlap"] != 1:
        PROMO.refuse("DS-BOXES", "rotor " + str(rotor["name"]) + ": the " +
                     str(v["n_boxes"]) + " boxes do not reproduce the " +
                     str(int(selected.size)) + " annulus cells (union_equal " +
                     str(v["union_equal"]) + ", max_overlap " +
                     str(v["max_overlap"]) + ")")
    return v


# ---------------------------------------------------------------- commands

def cmd_write(args) -> dict:
    """R9: the full forward case on the real mesh, CPU only."""
    fnum = PROMO.fnum
    mesh = os.path.abspath(args.mesh)
    out = os.path.abspath(args.out)
    end_time = float(args.end_time)
    dt = float(args.dt)
    write_interval = float(args.write_interval)
    thrust_n = _check_thrust(args.thrust_n)
    _check_out(out)
    pm_dir, tm, names, boundary, rotors = _load_mesh(mesh)
    nu = float(tm["params"]["nu"])
    U_vec = [float(v) for v in tm["flow"]["velocity_m_s"]]
    speed = float(np.linalg.norm(np.array(U_vec, dtype=np.float64)))
    k_in = 1.5 * (PROMO.INTENSITY * speed) ** 2
    omega_in = k_in / (PROMO.NUT_RATIO * nu)
    q = 0.5 * PROMO.RHO * speed * speed

    t0 = time.monotonic()
    points = PROMO.read_points(os.path.join(pm_dir, "points"))
    arity, ptr, flat = PROMO.read_faces(os.path.join(pm_dir, "faces"))
    pm = PROMO.PolyMesh(points, arity, ptr, flat)
    owner = PROMO.read_ints(os.path.join(pm_dir, "owner"))
    neighbour = PROMO.read_ints(os.path.join(pm_dir, "neighbour"))
    n_cells = int(owner.max()) + 1 if owner.size else 0
    V, centres = cell_geometry(pm, owner, neighbour, n_cells)
    print("[drone-solve] mesh parsed in " + "%.1f" % (time.monotonic() - t0) +
          " s (" + str(points.shape[0]) + " points, " + str(arity.size) +
          " faces, " + str(n_cells) + " cells)", flush=True)

    rotors_out = []
    entries = []
    for r in rotors:
        sel = disk_cells(centres, r)
        if sel.size == 0:
            PROMO.refuse("DS-BOXES", "rotor " + str(r["name"]) +
                         ": no cell centre falls in the disk")
        zc = float(r["centre_m"][2])
        zlo = zc - float(r["thickness_m"]) / 2.0
        zhi = zc + float(r["thickness_m"]) / 2.0
        boxes = disk_boxes(centres, sel, zlo, zhi)
        v = _verify_or_refuse(centres, sel, boxes, r)
        v_tool = float(V[sel].sum())
        bfz = body_force_z(thrust_n, PROMO.RHO, v_tool, float(r["normal"][2]))
        tag = "disk" + str(int(r["index"]))
        for kk, blo, bhi in ((kk, b[0], b[1]) for kk, b in enumerate(boxes)):
            entries.append((tag + "_b" + "%03d" % kk, (0.0, 0.0, bfz), blo, bhi))
        rotors_out.append({"index": int(r["index"]), "name": r["name"],
                           "spin": r.get("spin"),
                           "centre_m": [float(x) for x in r["centre_m"]],
                           "radius_m": float(r["radius_m"]),
                           "hub_radius_m": float(r["hub_radius_m"]),
                           "thickness_m": float(r["thickness_m"]),
                           "n_cells_tool": int(sel.size), "V_tool_m3": v_tool,
                           "n_boxes": int(len(boxes)), "bodyforce_z": bfz,
                           "verify": v})
        print("[drone-solve] disk " + str(int(r["index"])) + ": " + str(int(sel.size)) +
              " cells, V " + fnum(v_tool) + " m3, " + str(len(boxes)) +
              " boxes, bodyForce z " + fnum(bfz), flush=True)

    case = os.path.join(out, "case")
    os.makedirs(os.path.join(case, "constant", "polyMesh"), exist_ok=True)
    os.makedirs(os.path.join(case, "system"), exist_ok=True)
    os.makedirs(os.path.join(case, "0"), exist_ok=True)
    five = ["boundary", "faces", "owner", "neighbour", "points"]
    sizes = {}
    for f in five:
        shutil.copyfile(os.path.join(pm_dir, f),
                        os.path.join(case, "constant", "polyMesh", f))
        sizes[f] = os.path.getsize(os.path.join(pm_dir, f))
    for f in ("LICENSE.txt", "LICENSE", "rotors.json"):
        src = os.path.join(mesh, f)
        if os.path.isfile(src):
            shutil.copyfile(src, os.path.join(out, f))

    wt = PROMO._write_text
    wt(os.path.join(case, "constant", "physicalProperties"),
       PROMO.foam_header("physicalProperties", "dictionary", "constant") +
       "viscosityModel  constant;\n\n"
       "nu              [0 2 -1 0 0 0 0] " + fnum(nu) + ";\n\n")
    wt(os.path.join(case, "constant", "momentumTransport"),
       PROMO.foam_header("momentumTransport", "dictionary", "constant") +
       "simulationType  RAS;\n\nRAS\n{\n    model       kOmegaSST;\n"
       "    turbulence  on;\n    printCoeffs  on;\n}\n\n")
    wt(os.path.join(case, "system", "controlDict"),
       PROMO.foam_header("controlDict", "dictionary", "system") +
       "application     foamRun;\n"
       "startFrom       startTime;\n"
       "startTime       0;\n"
       "stopAt          endTime;\n"
       "endTime         1;\n"
       "deltaT          1;\n"
       "writeControl    timeStep;\n"
       "writeInterval   1;\n"
       "writeFormat     ascii;\n"
       "writePrecision  8;\n\n")
    wt(os.path.join(case, "system", "fvSchemes"),
       PROMO.foam_header("fvSchemes", "dictionary", "system") +
       "ddtSchemes { default Euler; }\n"
       "gradSchemes { default Gauss linear; }\n"
       "divSchemes { default none; div(phi,U) bounded Gauss linearUpwind grad(U);"
       " div(phi,T) bounded Gauss upwind; div(phi,k) bounded Gauss upwind;"
       " div(phi,omega) bounded Gauss upwind; }\n"
       "laplacianSchemes { default Gauss linear corrected; }\n"
       "interpolationSchemes { default linear; }\n"
       "snGradSchemes { default corrected; }\n\n")
    wt(os.path.join(case, "system", "fvSolution"),
       PROMO.foam_header("fvSolution", "dictionary", "system") +
       "solvers\n{\n"
       "    p { solver PCG; preconditioner DIC; tolerance 1e-07; relTol 0.01;"
       " maxIter 1000; }\n"
       "    U { solver PBiCGStab; preconditioner diagonal; tolerance 1e-08;"
       " relTol 0.1; maxIter 200; }\n"
       "    T { solver PBiCGStab; preconditioner diagonal; tolerance 1e-08;"
       " relTol 0.1; maxIter 200; }\n"
       "    k { solver PBiCGStab; preconditioner diagonal; tolerance 1e-08;"
       " relTol 0.1; maxIter 200; }\n"
       "    omega { solver PBiCGStab; preconditioner diagonal; tolerance 1e-08;"
       " relTol 0.1; maxIter 200; }\n"
       "}\n"
       "PIMPLE { nCorrectors 2; nNonOrthogonalCorrectors 1; }\n"
       "relaxationFactors { fields { p 0.3; } equations"
       " { U 0.7; T 0.7; k 0.7; omega 0.7; } }\n\n")
    if thrust_n != 0.0:
        wt(os.path.join(case, "constant", "fvSources"), fv_sources_text(entries))

    bf = boundary_entries(names, U_vec, k_in, omega_in)
    c0 = os.path.join(case, "0")
    uin = "uniform (" + fnum(U_vec[0]) + " " + fnum(U_vec[1]) + " " + fnum(U_vec[2]) + ")"
    PROMO.write_field(c0, "U", "volVectorField", "[0 1 -1 0 0 0 0]", uin, names, bf["U"])
    PROMO.write_field(c0, "p", "volScalarField", "[0 2 -2 0 0 0 0]", "uniform 0",
                      names, bf["p"])
    PROMO.write_field(c0, "T", "volScalarField", "[0 0 0 1 0 0 0]",
                      "uniform " + fnum(PROMO.T_K), names, bf["T"])
    PROMO.write_field(c0, "k", "volScalarField", "[0 2 -2 0 0 0 0]",
                      "uniform " + fnum(k_in), names, bf["k"])
    PROMO.write_field(c0, "omega", "volScalarField", "[0 0 -1 0 0 0 0]",
                      "uniform " + fnum(omega_in), names, bf["omega"])
    PROMO.write_field(c0, "nut", "volScalarField", "[0 2 -1 0 0 0 0]", "uniform 0",
                      names, bf["nut"])

    bpath, bsha = PROMO.binary_info()
    wj = {"tool": TOOL, "version": VERSION, "mesh_dir": mesh, "out_dir": out,
          "credit": CREDIT,
          "params": {"U_vec": U_vec, "speed_m_s": speed, "nu": nu, "rho": PROMO.RHO,
                     "dt": dt, "end_time": end_time, "write_interval": write_interval,
                     "intensity": PROMO.INTENSITY, "nut_ratio": PROMO.NUT_RATIO,
                     "k_in": k_in, "omega_in": omega_in, "q": q,
                     "thrust_target_N": thrust_n, "mass_kg": MASS_KG,
                     "rotor_model": ROTOR_MODEL},
          "rotors": rotors_out,
          "binary": {"path": bpath, "sha256": bsha},
          "command": PROMO.solve_command(bpath, end_time, dt, write_interval),
          "sizes": sizes, "license_copied": True}
    with open(os.path.join(out, "write.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(wj, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    print("[drone-solve] case written: " + out + " (" +
          "%.1f" % (time.monotonic() - t0) + " s total, U = " + fnum(speed) +
          " m/s, thrust " + fnum(thrust_n) + " N, boxes " +
          str(len(entries)) + ")", flush=True)
    return wj


def _gpu_preflight():
    """DS-GPU: (before, apps, shared) against the drone's 8000 MiB bar."""
    before, apps = PROMO.nvidia_query()
    if before is None:
        PROMO.refuse("DS-GPU", "nvidia-smi query failed")
    first = before.splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    if len(parts) < 3:
        PROMO.refuse("DS-GPU", "nvidia-smi output has no total,used,utilization: " + first)
    free = float(parts[0]) - float(parts[1])
    if free < MIN_FREE_MIB:
        PROMO.refuse("DS-GPU", "only " + PROMO.fnum(free) + " MiB free (" +
                     parts[0] + " - " + parts[1] + "), need " + str(MIN_FREE_MIB))
    shared = bool(apps) or float(parts[2]) > 0
    return before, apps, shared


def cmd_run(args):
    """write, then the sha256 pin, the drone preflight, the launch, post."""
    out = os.path.abspath(args.out)
    cmd_write(args)
    wj = json.load(open(os.path.join(out, "write.json"), encoding="utf-8"))
    params = wj["params"]
    bpath, bsha = PROMO.binary_info()
    if bsha is None:
        PROMO.refuse("DS-BIN", bpath + ": the pinned solver binary is missing")
    if bsha != BIN_SHA256:
        PROMO.refuse("DS-BIN", bpath + ": sha256 " + bsha +
                     " does not match the pin " + BIN_SHA256)
    before, apps, shared = _gpu_preflight()
    cmd = wj["command"]
    n_total = int(round(float(params["end_time"]) / float(params["dt"])))
    print("[drone-solve] launching " + os.path.basename(bpath) +
          " (shared GPU: " + str(shared) + ")", flush=True)
    rc, wall = PROMO.launch(out, cmd, float(params["dt"]), n_total, float(args.timeout))
    after, _ = PROMO.nvidia_query()
    with open(os.path.join(out, "run.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"returncode": rc, "wall_seconds": wall,
                   "gpu": {"before": before, "after": after, "shared": shared}},
                  fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    cmd_post(argparse.Namespace(out=out))


def make_gate(parsed, xcheck_ok, rotor_out) -> dict:
    """R10: the four gate keys of post."""
    end = parsed["end"]
    g = {"run_ok": bool(end and end[0] == "budget"),
         "permissive_ok": parsed["permissive_non_mach"] == 0,
         "rotor_ok": bool(rotor_out.get("pass")),
         "xcheck_ok": bool(xcheck_ok)}
    g["pass"] = g["run_ok"] and g["permissive_ok"] and g["rotor_ok"] and g["xcheck_ok"]
    return g


def cmd_post(args) -> dict:
    """R10: solve.json, residuals.csv, forces.csv and the gate line."""
    fnum = PROMO.fnum
    out = os.path.abspath(args.out)
    wj_path = os.path.join(out, "write.json")
    if not os.path.isdir(out) or not os.path.isfile(wj_path):
        PROMO.refuse("DS-OUT", out + ": post needs the directory to exist with write.json")
    wj = json.load(open(wj_path, encoding="utf-8"))
    params = wj["params"]
    q = float(params["q"])
    dt = float(params["dt"])
    log_path = os.path.join(out, "solve.log")
    text = ""
    if os.path.isfile(log_path):
        text = open(log_path, encoding="utf-8", errors="replace").read()
    parsed = PROMO.parse_log(text)

    pm_dir = os.path.join(out, "case", "constant", "polyMesh")
    boundary = PROMO.read_boundary(os.path.join(pm_dir, "boundary"))
    bd = {b[0]: (b[1], b[2]) for b in boundary}
    points = PROMO.read_points(os.path.join(pm_dir, "points"))
    arity, ptr, flat = PROMO.read_faces(os.path.join(pm_dir, "faces"))
    pm = PROMO.PolyMesh(points, arity, ptr, flat)
    owner = PROMO.read_ints(os.path.join(pm_dir, "owner"))
    n_cells = int(owner.max()) + 1 if owner.size else 0

    case_dir = os.path.join(out, "case")
    snaps = []
    for nm in os.listdir(case_dir):
        full = os.path.join(case_dir, nm)
        if os.path.isdir(full) and PROMO.RE_SNAP.fullmatch(nm) and float(nm) > 0:
            if os.path.isfile(os.path.join(full, "p")):
                snaps.append((float(nm), nm))
    snaps.sort()

    rows = []
    for t, nm in snaps:
        pvals = PROMO.read_foam_scalar(os.path.join(case_dir, nm, "p"), n_cells)
        fv = np.zeros(3)
        per = {}
        for name in WALLS:
            if name not in bd:
                continue
            cnt, st = bd[name]
            sf, _ = pm.face_sf_centres(st, cnt)
            fvec = PROMO.RHO * pvals[owner[st:st + cnt]][:, None] * sf
            per[name] = [float(x) for x in fvec.sum(axis=0)]
            fv = fv + fvec.sum(axis=0)
        rows.append({"time": t, "step": int(round(t / dt)), "F": fv, "per": per})

    with open(os.path.join(out, "residuals.csv"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("step,U_res,p_res,cont_err,M_max,M_cell\n")
        for it in parsed["iters"]:
            fh.write(",".join([str(it["step"]), fnum(it["U_res"]), fnum(it["p_res"]),
                               fnum(it["cont_err"]), fnum(it["M"]),
                               str(it["cell"])]) + "\n")
    with open(os.path.join(out, "forces.csv"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("time,step,Fx,Fy,Fz\n")
        for r in rows:
            fh.write(",".join([fnum(r["time"]), str(r["step"]), fnum(r["F"][0]),
                               fnum(r["F"][1]), fnum(r["F"][2])]) + "\n")

    cls = PROMO.classify(parsed, [r["step"] for r in rows],
                         [float(r["F"][0]) for r in rows],
                         [float(r["F"][2]) for r in rows])

    n_steps = (parsed["iters"][-1]["step"] + 1) if parsed["iters"] else 0
    wsel = [r for r in rows if r["step"] >= n_steps - PROMO.STOP_RULE["window_iters"]]

    def stats(v):
        if not v:
            return {"mean": None, "min": None, "max": None}
        a = np.array(v, dtype=np.float64)
        if a.size == 0:
            return {"mean": None, "min": None, "max": None}
        return {"mean": float(a.mean()), "min": float(a.min()), "max": float(a.max())}

    fx = stats([float(r["F"][0]) for r in wsel])
    fz = stats([float(r["F"][2]) for r in wsel])
    window = {"times": [r["time"] for r in wsel],
              "Fx": [float(r["F"][0]) for r in wsel],
              "Fz": [float(r["F"][2]) for r in wsel],
              "Fx_mean": fx["mean"], "Fx_min": fx["min"], "Fx_max": fx["max"],
              "Fz_mean": fz["mean"], "Fz_min": fz["min"], "Fz_max": fz["max"]}

    xrel = {}
    if rows:
        last = rows[-1]
        for name in WALLS:
            if name in last["per"] and name in parsed["forces"]:
                ft = np.array(last["per"][name])
                fs = np.array(parsed["forces"][name]["F_pres"])
                xrel[name] = float(np.linalg.norm(ft - fs) /
                                   max(float(np.linalg.norm(fs)), 1.0))
    max_rel = max(xrel.values()) if xrel else None
    xcheck_ok = max_rel is not None and max_rel <= PROMO.XCHECK_LIMIT

    per_out = {}
    cv = np.zeros(3)
    cp = np.zeros(3)
    for name in WALLS:
        if name in parsed["forces"]:
            blk = parsed["forces"][name]
            per_out[name] = {"F_visc": blk["F_visc"], "F_pres": blk["F_pres"]}
            cv = cv + np.array(blk["F_visc"])
            cp = cp + np.array(blk["F_pres"])
    fvec = cv + cp
    drag = -float(fvec[0])
    final = {"per_patch": per_out,
             "drone": {"F_visc": [float(x) for x in cv],
                       "F_pres": [float(x) for x in cp],
                       "F": [float(x) for x in fvec]},
             "drag_N": drag, "side_N": float(fvec[1]), "lift_N": float(fvec[2]),
             "drag_area_m2": (drag / q if q != 0 else None)}

    src_text = "\n".join([ln for ln in text.splitlines()
                          if ln.startswith("source on U:")])
    rotors_out = [dict(r) for r in (wj.get("rotors") or [])]
    for r in rotors_out:
        r["source_text"] = src_text
        r["thrust_target_N"] = float(params["thrust_target_N"])
    rg = rotor_gate(rotors_out, float(params["rho"]))
    for e in rg["rotors"]:
        for r in rotors_out:
            if r["index"] == e["index"]:
                r.update({k: e[k] for k in ("n_cells_solver", "V_solver_m3",
                                            "n_boxes_solver", "thrust_applied_N",
                                            "rel_err", "count_match", "has_lines",
                                            "pass")})

    runj = {}
    runj_path = os.path.join(out, "run.json")
    if os.path.isfile(runj_path):
        runj = json.load(open(runj_path, encoding="utf-8"))
    endw, enddetail, endcode = (parsed["end"] if parsed["end"] else (None, None, None))
    run = {"returncode": runj.get("returncode", endcode),
           "wall_seconds": runj.get("wall_seconds"),
           "solver_seconds": parsed["solver_seconds"],
           "end": {"word": endw, "detail": enddetail, "exit_code": endcode},
           "n_steps": n_steps,
           "permissive_lines": parsed["permissive"],
           "permissive_non_mach": parsed["permissive_non_mach"]}
    gpu = runj.get("gpu") or {"before": None, "after": None, "shared": None}

    its = parsed["iters"]
    if its:
        mx = max(its, key=lambda d: d["M"] if d["M"] == d["M"] else float("-inf"))
        residuals = {"first": its[0]["U_res"], "last": its[-1]["U_res"], "n": len(its)}
        mach = {"last": its[-1]["M"], "last_cell": its[-1]["cell"],
                "max": mx["M"], "max_step": mx["step"]}
    else:
        residuals = {"first": None, "last": None, "n": 0}
        mach = {"last": None, "last_cell": None, "max": None, "max_step": None}

    gate = make_gate(parsed, xcheck_ok, rg)
    sol = {"tool": wj.get("tool", TOOL), "version": wj.get("version", VERSION),
           "mesh_dir": wj.get("mesh_dir"), "out_dir": wj.get("out_dir", out),
           "credit": wj.get("credit", CREDIT),
           "binary": dict(wj.get("binary") or {"path": None, "sha256": None}),
           "command": wj.get("command"), "params": params,
           "rotors": rotors_out,
           "rotor_gate": {"pass": rg["pass"], "bare": rg["bare"]},
           "gpu": gpu, "run": run, "residuals": residuals, "mach": mach,
           "final_forces": final, "window": window,
           "xcheck": {"per_patch_rel": xrel, "max_rel": max_rel,
                      "limit": PROMO.XCHECK_LIMIT, "pass": bool(xcheck_ok)},
           "criteria": cls["criteria"], "failed": cls["failed"],
           "class": cls["class"], "reason_id": cls["reason_id"],
           "gate": gate, "license_copied": bool(wj.get("license_copied"))}
    with open(os.path.join(out, "solve.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(sol, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    failed = [k for k in ("run_ok", "permissive_ok", "rotor_ok", "xcheck_ok")
              if not gate[k]]
    if gate["pass"]:
        print("DRONE-SOLVE GATE PASS", flush=True)
    else:
        print("DRONE-SOLVE GATE FAIL: " + ",".join(failed), flush=True)
    return sol


# ---------------------------------------------------------------- selftest

def _lattice():
    """Lattice L: h = 0.01, centres ((i+0.5)h, (j+0.5)h, (k+0.5)h), 1024 cells."""
    h = 0.01
    pts = [[(i + 0.5) * h, (j + 0.5) * h, (k + 0.5) * h]
           for i in range(-8, 8) for j in range(-8, 8) for k in range(-2, 2)]
    return np.array(pts, dtype=np.float64)


ROTOR_A = {"index": 0, "name": "A", "centre_m": [0.0, 0.0, 0.0],
           "normal": [0.0, 0.0, 1.0], "radius_m": 0.05, "hub_radius_m": 0.012,
           "thickness_m": 0.012}


def _t1():
    L = _lattice()
    assert L.shape == (1024, 3), L.shape
    assert disk_cells(L, ROTOR_A).size == 152
    assert disk_cells(L, dict(ROTOR_A, hub_radius_m=0.0)).size == 160
    shifted = dict(ROTOR_A, centre_m=[0.003, -0.002, 0.001])
    assert disk_cells(L, shifted).size == 150
    text = PROMO._refused(lambda: disk_cells(L, dict(ROTOR_A, normal=[1.0, 0.0, 0.0])))
    assert "DS-ROTOR" in text, text


def _t2():
    L = _lattice()
    sel = disk_cells(L, ROTOR_A)
    boxes = disk_boxes(L, sel, -0.006, 0.006)
    assert verify_boxes(L, sel, boxes) == {"union_equal": True, "max_overlap": 1,
                                           "n_boxes": 12}
    assert verify_boxes(L, sel, boxes[:-1])["union_equal"] is False
    assert verify_boxes(L, sel, [boxes[0], boxes[0]])["max_overlap"] == 2


def _t3():
    L = _lattice()
    sel = box_select(L, (0.005, 0.005, 0.005), (0.015, 0.015, 0.005))
    assert sel.size == 4, sel
    got = L[sel]
    col = lambda a: sorted(set(np.round(got[:, a] / 0.005).tolist()))
    assert col(0) == [1.0, 3.0] and col(1) == [1.0, 3.0] and col(2) == [1.0], got


def _t4():
    td = tempfile.mkdtemp()
    try:
        PROMO.write_poly_mesh(td, PROMO._tiny_points(), PROMO._tiny_faces(),
                              PROMO._tiny_owner(), [1],
                              [(p, 1, i + 1) for i, p in enumerate(PATCHES)])
        pm, owner = PROMO.load_tiny(td)
        V, C = cell_geometry(pm, owner, np.array([1], dtype=np.int64), 2)
        assert np.allclose(V, [1.0, 1.0], rtol=0.0, atol=1e-12), V
        assert np.allclose(C, [[0.5, 0.5, 0.5], [1.5, 0.5, 0.5]], rtol=0.0,
                           atol=1e-12), C
    finally:
        shutil.rmtree(td, ignore_errors=True)


def _t5():
    v = body_force_z(5.0, 1.2041, 0.001, 1.0)
    assert abs(v - (-4152.479029980898)) <= 1e-12 * 4152.479029980898, v
    v2 = body_force_z(5.0, 1.2041, 0.001, -1.0)
    assert abs(v2 - 4152.479029980898) <= 1e-12 * 4152.479029980898, v2
    assert abs(THRUST_N_DEFAULT - 5.060985757692308) <= 1e-12 * THRUST_N_DEFAULT


def _t6():
    entries = [("disk0_b000", (0.0, 0.0, -10309.5),
                (0.0, -0.05, -0.006), (0.05, 0.05, 0.006)),
               ("disk1_b000", (0.0, 0.0, -9999.25),
                (-0.05, 0.0, -0.006), (0.0, 0.05, 0.006))]
    text = fv_sources_text(entries)
    assert text.count("FoamFile") == 1, "FoamFile"
    assert text.count("type        momentumSource;") == 2, text
    assert text.count("selection   box;") == 2, text
    for name, b, lo, hi in entries:
        blk = text[text.index(name):]
        i = blk.index("bodyForce")
        line = blk[i:blk.index(";", i)]
        trip = [float(t) for t in line[line.index("(") + 1:line.index(")")].split()]
        assert trip == [float(x) for x in b], (trip, b)


def _t7():
    line0 = ('source on U: "disk0_b000": box (0.0 0.0 0.0) (0.1 0.1 0.001)'
             ' -> 3 cells, 0.0002 m3  |  body force (0.0 0.0 -13999.9)'
             ' m/s2 per unit mass')
    line1 = ('source on U: "disk0_b001": box (0.0 0.0 0.001) (0.1 0.1 0.002)'
             ' -> 2 cells, 0.0001 m3  |  body force (0.0 0.0 -13999.9)'
             ' m/s2 per unit mass')
    text = line0 + "\n" + line1 + "\nsources: none\n"
    parsed = parse_source_lines(text)
    assert set(parsed) == {"disk0_b000", "disk0_b001"}, sorted(parsed)
    assert parsed["disk0_b000"]["n"] == 3 and parsed["disk0_b000"]["V"] == 0.0002
    assert parsed["disk0_b001"]["n"] == 2 and parsed["disk0_b001"]["V"] == 0.0001
    base = {"index": 0, "name": "prop_0", "n_cells_tool": 5, "bodyforce_z": -13999.9,
            "thrust_target_N": 5.0577, "source_text": text}
    rg = rotor_gate([dict(base)], 1.2041)
    assert rg["pass"] is True and rg["bare"] is False, rg
    e = rg["rotors"][0]
    assert abs(e["thrust_applied_N"] - 5.057183877) <= 1e-9, e
    assert e["count_match"] is True and e["rel_err"] <= THRUST_TOL, e
    rg2 = rotor_gate([dict(base, n_cells_tool=6)], 1.2041)
    assert rg2["pass"] is False, rg2
    assert rg2["rotors"][0]["count_match"] is False, rg2
    rg3 = rotor_gate([dict(base, thrust_target_N=0.0)], 1.2041)
    assert rg3["pass"] is True and rg3["bare"] is True, rg3


def _t8():
    bf = boundary_entries(PATCHES, [-15.0, 0.0, 0.0], 0.16875, 112500.0)
    assert set(bf) == {"U", "p", "T", "k", "omega", "nut"}
    assert set(bf["U"]) == set(PATCHES)
    assert dict(bf["U"]["inlet"])["value"] == "uniform (-15.0 0.0 0.0)"
    assert dict(bf["p"]["outlet"])["type"] == "fixedValue"
    assert dict(bf["p"]["outlet"])["value"] == "uniform 0"
    for f in bf:
        for nm in FARFIELD:
            assert dict(bf[f][nm])["type"] == "slip", (f, nm)
    for nm in WALLS:
        assert dict(bf["U"][nm])["type"] == "noSlip", nm
        assert dict(bf["k"][nm])["type"] == "kqRWallFunction", nm


def _t9():
    PROMO._refused(lambda: _check_out(os.path.join(REPO, "temp-drone-solve-t9")))
    td = tempfile.mkdtemp()
    try:
        empty = os.path.join(td, "empty")
        os.makedirs(empty)
        assert "DS-MESH" in PROMO._refused(lambda: _load_mesh(empty))
        mesh = os.path.join(td, "mesh")
        PROMO.write_poly_mesh(os.path.join(mesh, "case"), PROMO._tiny_points(),
                              PROMO._tiny_faces(), PROMO._tiny_owner(), [1],
                              [(p, 1, i + 1) for i, p in enumerate(PATCHES)])
        with open(os.path.join(mesh, "LICENSE.txt"), "w", encoding="utf-8") as fh:
            fh.write("test\n")
        with open(os.path.join(mesh, "rotors.json"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"rotors": [dict(ROTOR_A, index=i) for i in range(4)]}))
        with open(os.path.join(mesh, "tunnel_mesh.json"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"variant": "hover"}))
        assert "DS-VARIANT" in PROMO._refused(lambda: _load_mesh(mesh))
        with open(os.path.join(mesh, "tunnel_mesh.json"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"variant": "forward"}))
        with open(os.path.join(mesh, "rotors.json"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"rotors": [dict(ROTOR_A, radius_m=0.01, index=i)
                                            for i in range(4)]}))
        assert "DS-ROTOR" in PROMO._refused(lambda: _load_mesh(mesh))
        assert "DS-THRUST" in PROMO._refused(lambda: _check_thrust(-1.0))
        L = _lattice()
        sel = disk_cells(L, ROTOR_A)
        bad = [((0.0, 0.0, 0.0), (0.001, 0.001, 0.001))]
        assert "DS-BOXES" in PROMO._refused(
            lambda: _verify_or_refuse(L, sel, bad, ROTOR_A))
    finally:
        shutil.rmtree(td, ignore_errors=True)


def _t10():
    td = tempfile.mkdtemp()
    try:
        it0 = ("iter 0  |U| res 3.2e-01  |p| res 9.9e+00  contErr 1.2e-03"
               "  T [293.1, 293.3] K  rho [1.2, 1.21] kg/m3  p0 101325 Pa"
               "  dp0/dt 0 Pa/s  M max 0.03 (cell 12) mean 0.02")
        it1 = ("iter 1  |U| res 2.1e-01  |p| res 6.6e+00  contErr 8.0e-04"
               "  T [293.1, 293.3] K  rho [1.2, 1.21] kg/m3  p0 101325 Pa"
               "  dp0/dt 0 Pa/s  M max 0.03 (cell 12) mean 0.02")
        force = []
        for nm, a in (("body", "0.01"), ("arms", "0.02"), ("motors", "0.005"),
                      ("skids", "0.008")):
            force.append("  " + nm + ": area " + a +
                         " m2 | F_visc = (0.1 0.0 0.0) N | F_pres = (1.0 0.0 5.0) N")
        src = ('source on U: "disk0_b000": box (0.0 0.0 0.0) (0.1 0.1 0.001)'
               ' -> 3 cells, 0.0002 m3  |  body force (0.0 0.0 -13999.9)'
               ' m/s2 per unit mass\n'
               'source on U: "disk0_b001": box (0.0 0.0 0.001) (0.1 0.1 0.002)'
               ' -> 2 cells, 0.0001 m3  |  body force (0.0 0.0 -13999.9)'
               ' m/s2 per unit mass')
        log = "\n".join([it0, it1, "done in 1.5 s"] + force +
                        ["run ended: budget | endTime reached | exit code 0",
                         src]) + "\n"
        wj = {"params": {"q": 135.46125, "dt": 0.0005, "rho": 1.2041,
                         "thrust_target_N": 5.0577},
              "rotors": [{"index": i, "name": "prop_" + str(i), "n_cells_tool": 5,
                          "bodyforce_z": -13999.9} for i in range(4)]}
        with open(os.path.join(td, "write.json"), "w", encoding="utf-8") as fh:
            json.dump(wj, fh)
        with open(os.path.join(td, "solve.log"), "w", encoding="utf-8") as fh:
            fh.write(log)
        parsed = PROMO.parse_log(log)
        assert parsed["end"] is not None and parsed["end"][0] == "budget"
        rotors_out = [dict(r, source_text=src, thrust_target_N=5.0577)
                      for r in wj["rotors"]]
        rg = rotor_gate(rotors_out, 1.2041)
        g = make_gate(parsed, True, rg)
        assert g["run_ok"] is True, g
        assert g["permissive_ok"] is True, g
        assert g["rotor_ok"] is False, g
        assert g["pass"] is False, g
    finally:
        shutil.rmtree(td, ignore_errors=True)


def _t11():
    # A unit-square pentagon (extra vertex mid-bottom-edge): the solver's
    # area-weighted centroid is (0.5, 0.5, 0), the vertex mean is
    # (0.5, 0.4, 0) - face_centroids must give the former, and
    # face_sf_centres the latter, so the difference is visible.
    pts = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    arity = np.array([5, 3], dtype=np.int64)
    ptr = np.array([0, 5, 8], dtype=np.int64)
    flat = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
    pm = PROMO.PolyMesh(pts, arity, ptr, flat)
    sf, cf = face_centroids(pm, 0, 2)
    assert np.allclose(sf[0], [0.0, 0.0, 1.0], rtol=0.0, atol=1e-12), sf[0]
    assert np.allclose(cf[0], [0.5, 0.5, 0.0], rtol=0.0, atol=1e-12), cf[0]
    _, xf = pm.face_sf_centres(0, 2)
    assert np.allclose(xf[0], [0.5, 0.4, 0.0], rtol=0.0, atol=1e-12), xf[0]
    # Three collinear points: Sf = 0 and Cf falls back to the vertex mean.
    assert np.allclose(sf[1], [0.0, 0.0, 0.0], rtol=0.0, atol=1e-12), sf[1]
    assert np.allclose(cf[1], [1.0, 0.0, 0.0], rtol=0.0, atol=1e-12), cf[1]


def selftest() -> int:
    """R11: T1-T11, CPU only; temp files only under tempfile.mkdtemp()."""
    tests = [("T1", _t1), ("T2", _t2), ("T3", _t3), ("T4", _t4), ("T5", _t5),
             ("T6", _t6), ("T7", _t7), ("T8", _t8), ("T9", _t9), ("T10", _t10),
             ("T11", _t11)]
    npass = 0
    for name, fn in tests:
        try:
            fn()
            print("[ok] " + name, flush=True)
            npass += 1
        except Exception as exc:
            print("[FAIL] " + name + ": " + str(exc), flush=True)
    if npass == len(tests):
        print("SELFTEST PASS " + str(npass) + "/" + str(len(tests)), flush=True)
        return 0
    print("SELFTEST FAIL " + str(npass) + "/" + str(len(tests)), flush=True)
    return 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog="solve.py",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    for name in ("write", "run", "post"):
        sp = sub.add_parser(name)
        sp.add_argument("--mesh")
        sp.add_argument("--out", required=True)
        if name != "post":
            sp.add_argument("--thrust-n", type=float, default=THRUST_N_DEFAULT)
            sp.add_argument("--end-time", type=float, default=0.5)
            sp.add_argument("--dt", type=float, default=0.0005)
            sp.add_argument("--write-interval", type=float, default=0.025)
        if name == "run":
            sp.add_argument("--timeout", type=float, default=0.0)
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if args.cmd == "write":
        cmd_write(args)
    elif args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "post":
        cmd_post(args)
    else:
        ap.print_help()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
