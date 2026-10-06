#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.

"""Sample the drone's forward-flight solution onto a dense lattice and pack
it as an Isaac Sim scene: OpenVDB grids in a .usda beside the drone.

The solver's own -output vdb,usda writes .vdb only on a uniform Cartesian
box mesh, so it refuses the snapped octree polyMesh before its time loop
(refuse_visualisation_on_a_non_cartesian_mesh in rust/src/io/output_plan.rs).
This tool therefore samples the finished solution (cell centres and values,
the promo helpers' inverse-distance-squared interpolation) onto a dense
lattice in Python, writes the .vdb through tools/drone/isaac_vdb.py run
under Isaac Sim's own Python (its omni.volume openvdb writes file format
224, which Isaac Sim 6.0 reads; Blender's bundled module writes 225, which
Isaac refuses - the gate's G-VDBVER), then a .usda in the solver's own
scene conventions (rust/src/io/usda.rs: one Volume prim per field with a
field relationship to an
OpenVDBAsset child; the vector U as four scalar grids). Isaac Sim renders
the scene headless through tools/drone/isaac_probe.py (run by the
supervisor on the GPU box, never here).

The mesh and everything made from it derive from the BSD-3-Clause PX4 x500
model ("(c) 2022 Rudis Laboratories / PX4 Autopilot for Drones"), so
NOTHING is written inside the repository: every output goes under --out,
which must lie outside the repository, with the model's LICENSE.txt,
LICENSE and rotors.json copied beside the scene. The flow values are
demonstration numbers from an unsteady run, not validated; wake is
|U - U_inf| / |U_inf|, derived from U, not a transported scalar (the solve
transported no passive scalar).

Subcommands: export (solve dir + geom dir -> lattice, vdb, usdc, scene,
export.json), validate (one .usda -> the check dict as JSON). --selftest
runs T1-T10 on the CPU (T8 drives tools/drone/isaac_vdb.py under Isaac
Sim's own Python).
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import types
import warnings

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdUtils, UsdVol, Vt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOL = "tools/drone/isaac_export.py"
VERSION = "drone-isaac/1"
CREDIT = "(c) 2022 Rudis Laboratories / PX4 Autopilot for Drones, BSD-3-Clause"
GRIDS = ["U.x", "U.y", "U.z", "U.mag", "p", "wake"]
VOXEL_DEFAULT = 0.01                                   # m
BOX_DEFAULT = (-1.2, -0.6, -0.8, 0.6, 0.6, 0.4)        # xlo, ylo, zlo, xhi, yhi, zhi (m): the drone plus its wake and downwash
MAX_VOXELS = 8_000_000
SOLID_VALUE = 0.0                                      # every grid's value where the lattice point is not fluid
FLUID_MIN_FRAC = 0.9
FREESTREAM_TOL = 0.05                                  # relative
BLENDER = "C:/Program Files/Blender Foundation/Blender 5.1/blender.exe"
ISAAC_PYTHON = "C:/iss/env/Scripts/python.exe"
VDB_MAX_FILE_VERSION = 224  # Isaac Sim 6.0's reader (measured 2026-10-06: 225 refused)
SCENE_NAME = "drone_flow.usda"
DRONE_REL = "./drone/x500.usdc"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_drone():
    here = os.path.dirname(os.path.abspath(__file__))
    return _load("drone_solve", os.path.normpath(os.path.join(here, "solve.py")))


def _load_post():
    here = os.path.dirname(os.path.abspath(__file__))
    return _load("promo_post", os.path.normpath(os.path.join(here, "..", "promo", "post.py")))


DRONE = _load_drone()
PROMO_POST = _load_post()
PROMO = DRONE.PROMO   # the promo solve helpers, loaded once by drone_solve itself


def refuse(code, text):
    print("refused: " + code + ": " + text, flush=True)
    raise SystemExit(2)


# ------------------------------------------------------------------ lattice

def lattice_axes(lo, hi, voxel):
    """R2: the axis coordinate arrays of the dense lattice - n_a nodes from
    lo_a (inclusive) stepping voxel, the last node within one voxel of hi_a."""
    lo = tuple(float(v) for v in lo)
    hi = tuple(float(v) for v in hi)
    v = float(voxel)
    if not (v > 0.0):
        refuse("DI-BOX", "voxel " + repr(v) + " m: must be > 0")
    for a in range(3):
        if lo[a] >= hi[a]:
            refuse("DI-BOX", "axis " + "xyz"[a] + ": lo " + repr(lo[a]) +
                   " >= hi " + repr(hi[a]))
    dims = [int(math.floor((hi[a] - lo[a]) / v + 1e-9)) + 1 for a in range(3)]
    n = dims[0] * dims[1] * dims[2]
    if n > MAX_VOXELS:
        refuse("DI-BOX", "lattice " + repr(dims) + " is " + str(n) +
               " voxels, over the cap " + str(MAX_VOXELS))
    axes = [lo[a] + v * np.arange(dims[a], dtype=np.float64) for a in range(3)]
    return axes[0], axes[1], axes[2]


def lattice_points(xs, ys, zs):
    """R2: (nx*ny*nz, 3) float64, meshgrid indexing="ij" in C order, so the
    flat index of (i, j, k) is (i*ny + j)*nz + k."""
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float64)


def sample_cells(centres, h, values, points, chunk=200000):
    """R3: promo_post.make_vel once, then chunk-by-chunk queries - (vals
    (n, m) float64, fluid (n,) bool); every non-fluid row is SOLID_VALUE."""
    centres = np.asarray(centres, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64).reshape(-1)
    values = np.asarray(values, dtype=np.float64)
    nc = int(centres.shape[0])
    vel = PROMO_POST.make_vel(centres, h, values, nc)
    n = int(points.shape[0])
    vals = np.zeros((n, int(values.shape[1])), dtype=np.float64)
    fluid = np.zeros(n, dtype=bool)
    for a in range(0, n, int(chunk)):
        b = min(n, a + int(chunk))
        v, fl = vel(points[a:b])
        vals[a:b] = v
        fluid[a:b] = fl
    vals[~fluid] = SOLID_VALUE
    return vals, fluid


def derived_grids(U, p, fluid, U_inf):
    """R4: the six GRIDS as (n,) float32, SOLID_VALUE off the fluid - wake is
    |U - U_inf| / |U_inf|, derived from U, not a transported scalar."""
    U = np.asarray(U, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    fluid = np.asarray(fluid, dtype=bool)
    uinf = np.asarray(U_inf, dtype=np.float64).reshape(3)
    speed = float(np.linalg.norm(uinf))
    raw = {"U.x": U[:, 0], "U.y": U[:, 1], "U.z": U[:, 2],
           "U.mag": np.linalg.norm(U, axis=1), "p": p,
           "wake": (np.linalg.norm(U - uinf[None, :], axis=1) / speed
                    if speed > 0.0 else np.zeros(U.shape[0]))}
    out = {}
    for name in GRIDS:
        arr = np.array(raw[name], dtype=np.float32)
        arr[~fluid] = SOLID_VALUE
        out[name] = arr
    return out


# ------------------------------------------------------------------ scene

def safe_prim_name(grid):
    """R5: every char not in [A-Za-z0-9_] becomes "_"; a leading digit gets a
    "_" prefix - "U.x" -> "U_x"."""
    s = "".join(c if (c.isascii() and c.isalnum()) or c == "_" else "_" for c in str(grid))
    if s and s[0].isdigit():
        s = "_" + s
    return s


def write_scene(path, drone_rel, vdb_rel, grids, lo, voxel, dims, doc):
    """R5: the solver's own .usda conventions (rust/src/io/usda.rs) - one
    UsdVol.Volume per grid with a float3[] extent and a field:<safe-name>
    relationship to an OpenVDBAsset child carrying filePath + fieldName."""
    lo = np.asarray(lo, dtype=np.float64).reshape(3)
    dims = np.asarray(dims, dtype=np.int64).reshape(3)
    voxel = float(voxel)
    half = voxel / 2.0
    ext_lo = Gf.Vec3f(float(lo[0] - half), float(lo[1] - half), float(lo[2] - half))
    ext_hi = Gf.Vec3f(float(lo[0] + (dims[0] - 1) * voxel + half),
                      float(lo[1] + (dims[1] - 1) * voxel + half),
                      float(lo[2] + (dims[2] - 1) * voxel + half))
    stage = Usd.Stage.CreateNew(str(path))
    layer = stage.GetRootLayer()
    layer.defaultPrim = "World"
    layer.documentation = doc
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    drone = UsdGeom.Xform.Define(stage, "/World/Drone")
    drone.GetPrim().GetReferences().AddReference(str(drone_rel))
    UsdGeom.Scope.Define(stage, "/World/Flow")
    for g in grids:
        s = safe_prim_name(g)
        vol = UsdVol.Volume.Define(stage, "/World/Flow/" + s)
        UsdGeom.Boundable(vol.GetPrim()).CreateExtentAttr(Vt.Vec3fArray([ext_lo, ext_hi]))
        field_path = Sdf.Path("/World/Flow/" + s + "/" + s)
        vol.GetPrim().CreateRelationship("field:" + s).SetTargets([field_path])
        asset = UsdVol.OpenVDBAsset.Define(stage, field_path)
        asset.CreateFilePathAttr(Sdf.AssetPath(str(vdb_rel)))
        asset.CreateFieldNameAttr(str(g))
    stage.Save()


def validate_scene(path, expect_grids):
    """R5: open the stage back and check everything - never raises."""
    res = {"opens": False, "up_axis": "", "meters_per_unit": 0.0, "default_prim": "",
           "volumes": 0, "fields": [], "drone_meshes": 0, "unresolved": [],
           "compliance_errors": [], "compliance_failed": [], "ok": False}
    expect = [str(g) for g in expect_grids]
    try:
        stage = Usd.Stage.Open(str(path))
    except Exception:
        return res
    if stage is None:
        return res
    res["opens"] = True
    try:
        res["up_axis"] = str(UsdGeom.GetStageUpAxis(stage) or "")
        res["meters_per_unit"] = float(UsdGeom.GetStageMetersPerUnit(stage))
        dp = stage.GetDefaultPrim()
        res["default_prim"] = str(dp.GetPath()) if dp.IsValid() else ""
        vols = [p for p in stage.Traverse() if p.IsA(UsdVol.Volume)]
        res["volumes"] = len(vols)
        fields = []
        for v in vols:
            for rel in v.GetPrim().GetRelationships():
                if not rel.GetName().startswith("field:"):
                    continue
                for tgt in rel.GetTargets():
                    fp = stage.GetPrimAtPath(tgt)
                    if fp.IsValid():
                        fa = fp.GetAttribute("fieldName")
                        val = fa.Get() if fa.IsValid() else None
                        fields.append(str(val) if val is not None else "")
        res["fields"] = fields
        res["drone_meshes"] = sum(1 for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)
                                  and str(p.GetPath()).startswith("/World/Drone"))
        missing = []
        _layers, _assets, unresolved = UsdUtils.ComputeAllDependencies(str(path))
        missing.extend(str(u) for u in unresolved)
        scene_dir = os.path.dirname(os.path.abspath(str(path)))
        for p in stage.Traverse():
            if not p.IsA(UsdVol.OpenVDBAsset):
                continue
            fa = p.GetAttribute("filePath")
            if not fa.IsValid() or fa.Get() is None:
                continue
            ap = str(getattr(fa.Get(), "path", fa.Get()))
            rp = ap if os.path.isabs(ap) else os.path.normpath(os.path.join(scene_dir, ap))
            if not os.path.isfile(rp):
                missing.append(ap)
        res["unresolved"] = missing
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cc = UsdUtils.ComplianceChecker(arkit=False, skipARKitRootLayerCheck=True,
                                            rootPackageOnly=False, skipVariants=False,
                                            verbose=False)
            cc.CheckCompliance(str(path))
            res["compliance_errors"] = [str(e) for e in cc.GetErrors()]
            res["compliance_failed"] = [str(e) for e in cc.GetFailedChecks()]
    except Exception:
        pass
    res["ok"] = bool(res["opens"] and res["up_axis"] == "Z" and res["meters_per_unit"] == 1.0
                     and res["default_prim"] == "/World" and res["volumes"] == len(expect)
                     and res["fields"] == expect and res["drone_meshes"] >= 1
                     and res["unresolved"] == [] and res["compliance_errors"] == []
                     and res["compliance_failed"] == [])
    return res


# ------------------------------------------------------------------ gate

GATE_IDS = ["G-GRIDS", "G-DENSE", "G-ROUNDTRIP", "G-PLACE", "G-FLUID",
            "G-FREESTREAM", "G-USD", "G-DRONE", "G-VDBVER"]


def judge(ex):
    """R7: the export gate, ids in table order."""
    failed = []
    lat = ex["lattice"]
    grids = ex["grids"]
    if [g["name"] for g in grids] != GRIDS:
        failed.append("G-GRIDS")
    if any(int(g["active_voxels"]) != int(lat["n_voxels"]) for g in grids):
        failed.append("G-DENSE")
    if any(float(g["roundtrip_max_abs"]) != 0.0 for g in grids):
        failed.append("G-ROUNDTRIP")
    lo = [float(v) for v in lat["lo"]]
    last = [float(v) for v in lat["last"]]
    if any(any(abs(float(g["world_first"][i]) - lo[i]) > 1e-9
               or abs(float(g["world_last"][i]) - last[i]) > 1e-9 for i in range(3))
           for g in grids):
        failed.append("G-PLACE")
    if float(ex["fluid_frac"]) < FLUID_MIN_FRAC:
        failed.append("G-FLUID")
    if float(ex["freestream"]["rel_err"]) > FREESTREAM_TOL:
        failed.append("G-FREESTREAM")
    if not ex["scene"].get("ok"):
        failed.append("G-USD")
    usd = ex["usd"]
    if int(usd.get("mesh_objects", 0)) < 1 or int(ex["scene"].get("drone_meshes", -1)) != int(usd.get("mesh_objects", -2)):
        failed.append("G-DRONE")
    fv = (ex.get("vdb") or {}).get("file_version")
    if not isinstance(fv, (int, float)) or isinstance(fv, bool) or int(fv) > VDB_MAX_FILE_VERSION:
        failed.append("G-VDBVER")
    return {"pass": not failed, "failed": failed}


# ------------------------------------------------------------- field reader

def read_foam_scalar(path, n_expected):
    """internalField of one volScalarField -> (n_expected,) float64 (the
    promo reader's rule, regex-free)."""
    text = open(path, encoding="utf-8", errors="replace").read()
    i = text.index("internalField")
    seg = text[i:]
    u = seg.find("uniform")
    v = seg.find("nonuniform")
    if v != -1 and (u == -1 or v < u):
        j = seg.find("List<scalar>", v)
        if j == -1:
            refuse("DI-FIELD", path + ": unparseable nonuniform internalField")
        tail = seg[j + len("List<scalar>"):].lstrip()
        k = 0
        while k < len(tail) and tail[k].isdigit():
            k += 1
        cnt = int(tail[:k]) if k else -1
        a = seg.index("(", v)
        b = seg.index(")", a)
        vals = PROMO_POST.np_nums(seg[a + 1:b], np.float64)
        if cnt != n_expected or vals.size != cnt:
            refuse("DI-FIELD", path + ": " + str(vals.size) + " values for " +
                   str(cnt) + " declared, " + str(n_expected) + " cells")
        return vals
    if u != -1:
        rest = seg[u + len("uniform"):].strip().rstrip(";").strip()
        tok = rest.split()
        if not tok:
            refuse("DI-FIELD", path + ": uniform internalField without a value")
        return np.full(max(n_expected, 1), float(tok[0]))
    refuse("DI-FIELD", path + ": internalField is neither uniform nor nonuniform")


# ------------------------------------------------------------------ checks

def _posix(path):
    return str(path).replace(os.sep, "/")


def _check_out(out):
    """DI-OUT: outside the repository (the drone geometry and everything
    made from it is never written inside it)."""
    try:
        rel = os.path.relpath(out, REPO)
    except ValueError:
        rel = ".."
    if rel != ".." and not rel.startswith(".." + os.sep):
        refuse("DI-OUT", _posix(out) + ": --out resolves inside the repository")


def _check_inputs(solve, time_sel):
    """DI-INPUT: the solve dir, its polyMesh, write.json and the time dir;
    DI-FIELD: U and p inside the time dir. latest = the largest numeric time
    directory other than 0."""
    if not os.path.isdir(solve):
        refuse("DI-INPUT", _posix(solve) + ": not a directory")
    if not os.path.isfile(os.path.join(solve, "write.json")):
        refuse("DI-INPUT", _posix(solve) + ": missing write.json")
    pm_dir = os.path.join(solve, "case", "constant", "polyMesh")
    five = ["boundary", "faces", "owner", "neighbour", "points"]
    missing = [f for f in five if not os.path.isfile(os.path.join(pm_dir, f))]
    if missing:
        refuse("DI-INPUT", _posix(pm_dir) + ": missing " + ", ".join(missing))
    case = os.path.join(solve, "case")
    if time_sel == "latest":
        names = []
        for nm in os.listdir(case):
            p = os.path.join(case, nm)
            if not os.path.isdir(p):
                continue
            try:
                fv = float(nm)
            except ValueError:
                continue
            if fv != 0.0:
                names.append((fv, nm))
        if not names:
            refuse("DI-INPUT", _posix(case) + ": no time directories")
        time_sel = max(names)[1]
    tdir = os.path.join(case, time_sel)
    if not os.path.isdir(tdir):
        refuse("DI-INPUT", _posix(tdir) + ": no such time directory")
    for f in ("U", "p"):
        if not os.path.isfile(os.path.join(tdir, f)):
            refuse("DI-FIELD", _posix(tdir) + ": missing " + f)
    return pm_dir, tdir, time_sel


def _parse_box(text):
    """DI-BOX text form: six comma-joined numbers."""
    parts = str(text).split(",")
    if len(parts) != 6:
        refuse("DI-BOX", "--box needs xlo,ylo,zlo,xhi,yhi,zhi, got " + str(text))
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        refuse("DI-BOX", "--box is not six numbers: " + str(text))
    return tuple(vals[0:3]), tuple(vals[3:6])


def _run_writer(cmd, log_path, first=False):
    """One writer subprocess - isaac_vdb.py under Isaac's Python, or blender
    -b - with stdout+stderr streamed into writers.log; both writers append.
    Returns the exit code; the caller refuses."""
    d = os.path.dirname(log_path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(log_path, "wb" if first else "ab") as log:
        log.write(("=== " + " ".join(_posix(a) for a in cmd) + chr(10)).encode("utf-8"))
        log.flush()
        return subprocess.call([str(a) for a in cmd], stdout=log, stderr=subprocess.STDOUT)


def _digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# ------------------------------------------------------------------ export

def cmd_export(args):
    """R11: the real solve -> the full Isaac output set, gate-judged."""
    t_all = time.monotonic()
    times = {}

    def stage(name, t0):
        dt = time.monotonic() - t0
        times[name] = round(dt, 1)
        print("[drone-isaac] " + name + " " + ("%.1f" % dt) + " s", flush=True)

    solve = os.path.abspath(args.solve)
    geom = os.path.abspath(args.geom)
    out = os.path.abspath(args.out)
    blender = os.path.abspath(args.blender)
    vdb_python = os.path.abspath(args.vdb_python)
    t0 = time.monotonic()
    _check_out(out)
    pm_dir, tdir, time_sel = _check_inputs(solve, str(args.time))
    lo, hi = _parse_box(args.box)
    voxel = float(args.voxel)
    xs, ys, zs = lattice_axes(lo, hi, voxel)
    if not os.path.isfile(blender):
        refuse("DI-BLENDER", _posix(blender) + ": blender.exe not found")
    if not os.path.isfile(vdb_python):
        refuse("DI-VDB", _posix(vdb_python) + ": the vdb python not found")
    stage("checks", t0)

    t0 = time.monotonic()
    points = PROMO.read_points(os.path.join(pm_dir, "points"))
    arity, ptr, flat = PROMO.read_faces(os.path.join(pm_dir, "faces"))
    pm = PROMO.PolyMesh(points, arity, ptr, flat)
    owner = PROMO.read_ints(os.path.join(pm_dir, "owner"))
    neighbour = PROMO.read_ints(os.path.join(pm_dir, "neighbour"))
    n_cells = int(owner.max()) + 1 if owner.size else 0
    V, centres = DRONE.cell_geometry(pm, owner, neighbour, n_cells)
    h = np.cbrt(V)
    U_cells = PROMO_POST.read_foam_vector(os.path.join(tdir, "U"), n_cells)
    p_cells = read_foam_scalar(os.path.join(tdir, "p"), n_cells)
    with open(os.path.join(solve, "write.json"), encoding="utf-8") as fh:
        uinf = np.asarray(json.load(fh)["params"]["U_vec"], dtype=np.float64)
    stage("mesh+fields", t0)

    t0 = time.monotonic()
    pts = lattice_points(xs, ys, zs)
    nx, ny, nz = int(xs.size), int(ys.size), int(zs.size)
    n_vox = int(pts.shape[0])
    values = np.column_stack([U_cells, p_cells.reshape(-1, 1)])
    vals, fluid = sample_cells(centres, h, values, pts)
    U = vals[:, 0:3]
    p = vals[:, 3]
    grids = derived_grids(U, p, fluid, uinf)
    fluid_frac = float(fluid.mean()) if n_vox else 0.0
    jj, kk = np.meshgrid(np.arange(ny), np.arange(nz), indexing="ij")
    plane = ((nx - 1) * ny + jj.ravel()) * nz + kk.ravel()
    sel = plane[fluid[plane]]
    if sel.size == 0:
        refuse("DI-FREESTREAM", "the xMax lattice plane has no fluid points")
    median_ux = float(np.median(U[sel, 0]))
    rel_err = abs(median_ux - float(uinf[0])) / abs(float(uinf[0]))
    stage("sample", t0)

    t0 = time.monotonic()
    os.makedirs(out, exist_ok=True)
    npz_path = os.path.join(out, "lattice.npz")
    payload = {"lo": np.asarray(lo, dtype=np.float64), "voxel": np.float64(voxel),
               "dims": np.array([nx, ny, nz], dtype=np.int64), "grids": np.array(GRIDS)}
    for g in GRIDS:
        payload[g] = np.ascontiguousarray(grids[g].reshape(nx, ny, nz), dtype=np.float32)
    np.savez(npz_path, **payload)
    stage("npz", t0)

    t0 = time.monotonic()
    log_path = os.path.join(out, "writers.log")
    vdb_script = os.path.join(REPO, "tools", "drone", "isaac_vdb.py")
    vdb_name = "drone_flow_t" + time_sel + ".vdb"
    vdb_path = os.path.join(out, "VDB", vdb_name)
    man_v = os.path.join(out, "vdb_manifest.json")
    rc = _run_writer([vdb_python, vdb_script, "--npz", npz_path, "--out", vdb_path,
                      "--manifest", man_v], log_path, first=True)
    if rc != 0:
        refuse("DI-VDB", "isaac_vdb.py exited " + str(rc) + " (see writers.log)")
    if not os.path.isfile(man_v):
        refuse("DI-VDB", _posix(man_v) + ": vdb manifest missing (see writers.log)")
    bpy_script = os.path.join(REPO, "tools", "drone", "isaac_bpy.py")
    usdc_path = os.path.join(out, "drone", "x500.usdc")
    man_u = os.path.join(out, "usd_manifest.json")
    rc = _run_writer([blender, "-b", "--factory-startup",
                      os.path.join(geom, "x500_assembled.blend"), "--python", bpy_script,
                      "--", "usd", "--out", usdc_path, "--manifest", man_u], log_path)
    if rc != 0:
        refuse("DI-BLENDER", "blender exited " + str(rc) + " (see writers.log)")
    if not os.path.isfile(man_u):
        refuse("DI-BLENDER", _posix(man_u) + ": usd manifest missing (see writers.log)")
    with open(man_v, encoding="utf-8") as fh:
        vdb_manifest = json.load(fh)
    with open(man_u, encoding="utf-8") as fh:
        usd_manifest = json.load(fh)
    stage("writers", t0)

    t0 = time.monotonic()
    scene_path = os.path.join(out, SCENE_NAME)
    doc = (CREDIT + "; drone solve time " + time_sel + " s; flow values are demonstration "
           "numbers from an unsteady run, not validated; wake = |U - U_inf| / |U_inf|, "
           "derived from U, not a transported scalar")
    write_scene(scene_path, DRONE_REL, "./VDB/" + vdb_name, GRIDS,
                np.asarray(lo, dtype=np.float64), voxel, (nx, ny, nz), doc)
    scene = validate_scene(scene_path, GRIDS)
    stage("scene", t0)

    t0 = time.monotonic()
    for f in ("LICENSE.txt", "LICENSE", "rotors.json"):
        src = os.path.join(solve, f)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(out, f))
    stage("copy", t0)

    t0 = time.monotonic()
    files = {}
    for p in (scene_path, vdb_path, usdc_path, npz_path):
        rel = _posix(os.path.relpath(p, out))
        files[rel] = {"bytes": int(os.path.getsize(p)), "sha256": _digest(p)}
    lo_l = [float(v) for v in lo]
    last = [lo_l[i] + (d - 1) * voxel for i, d in enumerate((nx, ny, nz))]
    ex = {
        "tool": TOOL, "version": VERSION, "credit": CREDIT,
        "solve_dir": _posix(solve), "time": time_sel,
        "lattice": {"lo": lo_l, "last": last, "voxel": voxel,
                    "dims": [nx, ny, nz], "n_voxels": n_vox},
        "U_inf": [float(v) for v in uinf],
        "fluid_frac": fluid_frac,
        "freestream": {"plane": "xMax lattice plane, i = nx-1",
                       "median_Ux": median_ux, "rel_err": rel_err},
        "grids": vdb_manifest.get("grids", []),
        "vdb": {"file_version": vdb_manifest.get("file_version"),
                "openvdb": vdb_manifest.get("openvdb"),
                "library_version": vdb_manifest.get("library_version")},
        "usd": usd_manifest,
        "scene": scene,
        "files": files,
        "timing_s": times,
        "notes": [
            "the solver's -output vdb,usda writes .vdb only on a uniform cartesian box mesh "
            "and refuse_visualisation_on_a_non_cartesian_mesh rejects this snapped octree, "
            "so the finished solution is sampled onto a dense lattice instead",
            "wake = |U - U_inf| / |U_inf|, derived from U, not a transported scalar",
            "flow values are demonstration numbers from an unsteady run, not validated",
            "no passive scalar: the solve transported none",
        ],
    }
    gate = judge(ex)
    ex["gate"] = gate
    with open(os.path.join(out, "export.json"), "w", encoding="utf-8") as fh:
        json.dump(ex, fh, indent=2)
    stage("report", t0)
    print("[drone-isaac] total " + ("%.1f" % (time.monotonic() - t_all)) + " s", flush=True)
    if gate["pass"]:
        print("DRONE-ISAAC GATE PASS", flush=True)
        return 0
    print("DRONE-ISAAC GATE FAIL " + ",".join(gate["failed"]), flush=True)
    return 1


def cmd_validate(args):
    d = validate_scene(args.scene, GRIDS)
    print(json.dumps(d, indent=2))
    return 0 if d["ok"] else 1


# ---------------------------------------------------------------- selftest

def _t1():
    xs, ys, zs = lattice_axes((-1.2, -0.6, -0.8), (0.6, 0.6, 0.4), 0.01)
    assert (len(xs), len(ys), len(zs)) == (181, 121, 121), (len(xs), len(ys), len(zs))
    assert abs(xs[0] - (-1.2)) < 1e-12, xs[0]
    assert abs(xs[-1] - 0.6) < 1e-12, xs[-1]
    bad = ((0.0, (-1.2, -0.6, -0.8), (0.6, 0.6, 0.4)),
           (0.01, (0.0, 0.0, 0.0), (0.0, 1.0, 1.0)),
           (0.001, (-1.2, -0.6, -0.8), (0.6, 0.6, 0.4)))
    for v, blo, bhi in bad:
        code = 0
        try:
            lattice_axes(blo, bhi, v)
        except SystemExit as exc:
            code = int(exc.code or 0)
        assert code == 2, (v, code)


def _t2():
    pts = lattice_points([0, 1, 2, 3], [10, 11, 12], [20, 21, 22, 23, 24])
    assert pts.shape == (60, 3), pts.shape
    assert tuple(pts[38]) == (2, 11, 23), pts[38]


def _t3():
    cen = np.array([[float(x), float(y), float(z)] for x in range(4)
                    for y in range(4) for z in range(4)])
    h = np.ones(64)
    f = (2.0 * cen[:, 0] + 3.0 * cen[:, 1] - cen[:, 2] + 1.0).reshape(64, 1)
    probe = np.array([[1.5, 1.5, 1.5], [2.0, 1.0, 3.0], [10.0, 10.0, 10.0]])
    vals, fluid = sample_cells(cen, h, f, probe)
    assert abs(vals[0, 0] - 7.0) < 1e-12 and bool(fluid[0]), (vals[0], fluid[0])
    assert abs(vals[1, 0] - 5.0) < 1e-12, vals[1]
    assert not fluid[2] and vals[2, 0] == 0.0, (vals[2], fluid[2])
    vals3, _ = sample_cells(cen, h, np.column_stack([f[:, 0], 2.0 * f[:, 0], -f[:, 0]]),
                            np.array([[1.5, 1.5, 1.5]]))
    assert np.allclose(vals3[0], [7.0, 14.0, -7.0], rtol=0.0, atol=1e-12), vals3[0]


def _t4():
    U = np.array([[-15.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-12.0, 0.0, -4.0]])
    p = np.array([1.0, 2.0, 3.0])
    g = derived_grids(U, p, np.array([True, True, True]), (-15.0, 0.0, 0.0))
    assert list(g.keys()) == GRIDS, list(g.keys())
    assert np.allclose(g["wake"], [0.0, 1.0, 1.0 / 3.0], rtol=0.0, atol=1e-6), g["wake"]
    assert abs(float(g["U.mag"][2]) - 12.649110640673518) < 1e-5, g["U.mag"][2]
    for name in GRIDS:
        assert g[name].dtype == np.float32, (name, g[name].dtype)
    g2 = derived_grids(U, p, np.array([True, False, True]), (-15.0, 0.0, 0.0))
    for name in GRIDS:
        assert g2[name][1] == 0.0, (name, g2[name][1])
        assert g2[name][0] == g[name][0] and g2[name][2] == g[name][2], name


def _t5():
    assert safe_prim_name("U.x") == "U_x", safe_prim_name("U.x")
    assert safe_prim_name("U.mag") == "U_mag", safe_prim_name("U.mag")
    assert safe_prim_name("p") == "p"
    assert safe_prim_name("1a") == "_1a", safe_prim_name("1a")
    assert safe_prim_name("a-b c") == "a_b_c", safe_prim_name("a-b c")


def _t6_fixture(tmp):
    from pxr import Usd, UsdGeom
    os.makedirs(os.path.join(tmp, "drone"), exist_ok=True)
    os.makedirs(os.path.join(tmp, "VDB"), exist_ok=True)
    st = Usd.Stage.CreateNew(os.path.join(tmp, "drone", "x500.usda"))
    UsdGeom.Xform.Define(st, "/root")
    UsdGeom.Mesh.Define(st, "/root/cube")
    st.GetRootLayer().defaultPrim = "root"
    st.Save()
    with open(os.path.join(tmp, "VDB", "f.vdb"), "wb") as fh:
        fh.write(b"vdb")


def _t6():
    from pxr import Usd, UsdGeom
    tmp = tempfile.mkdtemp(prefix="drone-isaac-t6-")
    try:
        _t6_fixture(tmp)
        scene = os.path.join(tmp, "scene.usda")
        write_scene(scene, "./drone/x500.usda", "./VDB/f.vdb", GRIDS,
                    (-1.0, -1.0, -1.0), 0.5, (3, 3, 3), "doc")
        v = validate_scene(scene, GRIDS)
        assert v["ok"] is True, v
        assert v["volumes"] == 6 and v["fields"] == GRIDS, v
        assert v["drone_meshes"] == 1 and v["unresolved"] == [], v
        s = Usd.Stage.Open(scene)
        ext = UsdGeom.Boundable(s.GetPrimAtPath("/World/Flow/U_x")).GetExtentAttr().Get()
        assert np.allclose(np.array(ext[0]), [-1.25, -1.25, -1.25], rtol=0.0, atol=0.0), ext
        assert np.allclose(np.array(ext[1]), [0.25, 0.25, 0.25], rtol=0.0, atol=0.0), ext
        fa = s.GetPrimAtPath("/World/Flow/U_x/U_x").GetAttribute("fieldName")
        assert str(fa.Get()) == "U.x", fa.Get()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _t7():
    tmp = tempfile.mkdtemp(prefix="drone-isaac-t7-")
    try:
        _t6_fixture(tmp)
        scene = os.path.join(tmp, "scene.usda")
        write_scene(scene, "./drone/x500.usda", "./VDB/f.vdb", GRIDS,
                    (-1.0, -1.0, -1.0), 0.5, (3, 3, 3), "doc")
        os.remove(os.path.join(tmp, "VDB", "f.vdb"))
        v = validate_scene(scene, GRIDS)
        assert v["ok"] is False and v["unresolved"], v
        with open(os.path.join(tmp, "VDB", "f.vdb"), "wb") as fh:
            fh.write(b"vdb")
        v = validate_scene(scene, GRIDS[:-1])
        assert v["ok"] is False, v
        junk = os.path.join(tmp, "junk.usda")
        with open(junk, "w", encoding="utf-8") as fh:
            fh.write("this is not a usd file" + chr(10))
        v = validate_scene(junk, GRIDS)
        assert v["ok"] is False and v["opens"] is False, v
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _run_vdb(npz, out, man, log_path):
    """T8: the vdb writer under Isaac Sim's own Python (never Blender - the
    Blender-bundled openvdb writes file format 225, which Isaac refuses)."""
    cmd = [ISAAC_PYTHON, os.path.join(REPO, "tools", "drone", "isaac_vdb.py"),
           "--npz", npz, "--out", out, "--manifest", man]
    with open(log_path, "wb") as log:
        return subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)


def _t8():
    tmp = tempfile.mkdtemp(prefix="drone-isaac-t8-")
    try:
        npz = os.path.join(tmp, "l.npz")
        lo = np.array([-0.5, 0.25, 1.0])
        a = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
        np.savez(npz, lo=lo, voxel=np.float64(0.1),
                 dims=np.array([3, 4, 5], dtype=np.int64),
                 grids=np.array(["a", "b", "z"]), a=a, b=-a,
                 z=np.zeros((3, 4, 5), dtype=np.float32))
        man = os.path.join(tmp, "m.json")
        log = os.path.join(tmp, "w.log")
        rc = _run_vdb(npz, os.path.join(tmp, "o.vdb"), man, log)
        if rc != 0:
            tail = open(log, encoding="utf-8", errors="replace").read()[-2000:]
        assert rc == 0, "isaac_vdb.py exited " + str(rc) + chr(10) + tail
        with open(man, encoding="utf-8") as fh:
            d = json.load(fh)
        assert [g["name"] for g in d["grids"]] == ["a", "b", "z"], d["grids"]
        want_last = (-0.3, 0.55, 1.4)
        for g in d["grids"]:
            assert int(g["active_voxels"]) == 60, g
            assert float(g["roundtrip_max_abs"]) == 0.0, g
            wf = [float(v) for v in g["world_first"]]
            wl = [float(v) for v in g["world_last"]]
            for i in range(3):
                assert abs(wf[i] - float(lo[i])) < 1e-9, (g["name"], wf)
                assert abs(wl[i] - want_last[i]) < 1e-9, (g["name"], wl)
        assert int(d["bytes"]) > 0 and d["openvdb"] == "isaac-omni.volume", d
        assert int(d["file_version"]) == VDB_MAX_FILE_VERSION, d
        assert isinstance(d["library_version"], list) and d["library_version"], d
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _t9():
    lat = {"lo": [0.0, 0.0, 0.0], "last": [2.0, 2.0, 2.0], "voxel": 0.4,
           "dims": [6, 6, 6], "n_voxels": 216}
    grids = [{"name": n, "active_voxels": 216, "roundtrip_max_abs": 0.0,
              "world_first": [0.0, 0.0, 0.0], "world_last": [2.0, 2.0, 2.0]}
             for n in GRIDS]
    ex = {"lattice": lat, "grids": grids, "fluid_frac": 0.95,
          "freestream": {"plane": "xMax lattice plane, i = nx-1",
                         "median_Ux": -14.9, "rel_err": 0.01},
          "scene": {"ok": True, "drone_meshes": 25},
          "usd": {"mesh_objects": 25},
          "vdb": {"file_version": 224, "openvdb": "isaac-omni.volume",
                  "library_version": [12, 0, 0]}}
    g = judge(ex)
    assert g["pass"] is True and g["failed"] == [], g

    def broken(fn):
        e = json.loads(json.dumps(ex))
        fn(e)
        r = judge(e)
        assert r["pass"] is False and len(r["failed"]) == 1, r
        return r["failed"][0]

    order = ["G-GRIDS", "G-DENSE", "G-ROUNDTRIP", "G-PLACE", "G-FLUID",
             "G-FREESTREAM", "G-USD", "G-DRONE", "G-VDBVER"]

    def b_grids(e):
        e["grids"][2]["name"] = "zzz"

    def b_dense(e):
        e["grids"][1]["active_voxels"] = 5

    def b_round(e):
        e["grids"][0]["roundtrip_max_abs"] = 1e-3

    def b_place(e):
        e["grids"][4]["world_last"] = [9.0, 2.0, 2.0]

    def b_fluid(e):
        e["fluid_frac"] = 0.5

    def b_free(e):
        e["freestream"]["rel_err"] = 0.5

    def b_usd(e):
        e["scene"]["ok"] = False

    def b_drone(e):
        e["usd"]["mesh_objects"] = 24

    def b_vdb(e):
        e["vdb"]["file_version"] = 225

    for want, fn in zip(order, (b_grids, b_dense, b_round, b_place, b_fluid,
                                b_free, b_usd, b_drone, b_vdb)):
        got = broken(fn)
        assert got == want, (want, got)


def _t10_case(root, with_p):
    pm = os.path.join(root, "case", "constant", "polyMesh")
    os.makedirs(pm, exist_ok=True)
    for f in ("boundary", "faces", "owner", "neighbour", "points"):
        with open(os.path.join(pm, f), "w", encoding="utf-8") as fh:
            fh.write("dummy" + chr(10))
    os.makedirs(os.path.join(root, "case", "0.5"), exist_ok=True)
    with open(os.path.join(root, "write.json"), "w", encoding="utf-8") as fh:
        json.dump({"params": {"U_vec": [-15.0, 0.0, 0.0]}}, fh)
    with open(os.path.join(root, "case", "0.5", "U"), "w", encoding="utf-8") as fh:
        fh.write("dummy" + chr(10))
    if with_p:
        with open(os.path.join(root, "case", "0.5", "p"), "w", encoding="utf-8") as fh:
            fh.write("dummy" + chr(10))


def _t10():
    tmp = tempfile.mkdtemp(prefix="drone-isaac-t10-")
    real_out = os.path.join(REPO, "tools", "drone", "_x")
    try:
        _t10_case(os.path.join(tmp, "partial"), False)
        _t10_case(os.path.join(tmp, "full"), True)
        tool = os.path.join(REPO, "tools", "drone", "isaac_export.py")
        cases = [
            ("DI-OUT", ["export", "--solve", os.path.join(tmp, "full"), "--geom",
                        os.path.join(tmp, "geom"), "--out", real_out]),
            ("DI-INPUT", ["export", "--solve", os.path.join(tmp, "missing"), "--geom",
                          os.path.join(tmp, "geom"), "--out", os.path.join(tmp, "o1")]),
            ("DI-FIELD", ["export", "--solve", os.path.join(tmp, "partial"), "--geom",
                          os.path.join(tmp, "geom"), "--out", os.path.join(tmp, "o2")]),
            ("DI-BOX", ["export", "--solve", os.path.join(tmp, "full"), "--geom",
                        os.path.join(tmp, "geom"), "--out", os.path.join(tmp, "o3"),
                        "--voxel", "0"]),
            ("DI-BLENDER", ["export", "--solve", os.path.join(tmp, "full"), "--geom",
                            os.path.join(tmp, "geom"), "--out", os.path.join(tmp, "o4"),
                            "--blender", os.path.join(tmp, "no-blender.exe")]),
            ("DI-VDB", ["export", "--solve", os.path.join(tmp, "full"), "--geom",
                        os.path.join(tmp, "geom"), "--out", os.path.join(tmp, "o5"),
                        "--vdb-python", os.path.join(tmp, "no-python.exe")]),
        ]
        for code, argv in cases:
            out_arg = argv[argv.index("--out") + 1]
            if os.path.isdir(out_arg):
                shutil.rmtree(out_arg, ignore_errors=True)
            if os.path.exists(real_out):
                shutil.rmtree(real_out, ignore_errors=True)
            p = subprocess.run([sys.executable, tool] + argv, capture_output=True,
                               text=True, encoding="utf-8", errors="replace", cwd=REPO)
            assert p.returncode == 2, (code, p.returncode, p.stdout[-600:], p.stderr[-300:])
            assert "refused: " + code in p.stdout + p.stderr, (code, p.stdout[-400:])
            assert not os.path.exists(out_arg) or not os.listdir(out_arg), code
            if code == "DI-OUT":
                assert not os.path.exists(real_out), "DI-OUT left the repo path behind"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        if os.path.exists(real_out):
            shutil.rmtree(real_out, ignore_errors=True)


def selftest():
    """R9: T1-T10, CPU only (T8 drives tools/drone/isaac_vdb.py under Isaac
    Sim's own Python); temp files only under tempfile.mkdtemp()."""
    tests = [("T1", _t1), ("T2", _t2), ("T3", _t3), ("T4", _t4), ("T5", _t5),
             ("T6", _t6), ("T7", _t7), ("T8", _t8), ("T9", _t9), ("T10", _t10)]
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
    ap = argparse.ArgumentParser(prog="isaac_export.py",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    ep = sub.add_parser("export")
    ep.add_argument("--solve", required=True)
    ep.add_argument("--geom", required=True)
    ep.add_argument("--out", required=True)
    ep.add_argument("--time", default="latest")
    ep.add_argument("--voxel", type=float, default=VOXEL_DEFAULT)
    ep.add_argument("--box", default=",".join(str(v) for v in BOX_DEFAULT))
    ep.add_argument("--blender", default=BLENDER)
    ep.add_argument("--vdb-python", default=ISAAC_PYTHON)
    vp = sub.add_parser("validate")
    vp.add_argument("--scene", required=True)
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if args.cmd == "export":
        return cmd_export(args)
    if args.cmd == "validate":
        return cmd_validate(args)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
