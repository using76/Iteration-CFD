#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.

"""Stage the drone's finished solution as a case carrying a `wake` cell
field, run the solver's own offline writer `ofgpu-sample volume -isaac`
(SPEC-LIT 117.11, 117.9), read the .vdb back with Isaac's openvdb, export
the drone with Blender, write a wrapper scene, and judge the gate.

The writer reads <case>/constant/polyMesh and every volScalarField and
volVectorField of <case>/<time>, lays the voxel lattice of SPEC-LIT 117.3,
scales a density grid derived from the `wake` field so its densest column
has optical depth exactly tau ((117.12) of SPEC-LIT 117.9), and writes the
lattice .vdb plus the native Isaac scene. This tool stages that case (polyMesh, U, p, and wake =
|U - U_inf| / |U_inf| derived from U per cell), runs the writer, reads the
.vdb back through tools/drone/isaac_vdb.py under Isaac Sim's own Python
(its omni.volume openvdb writes file format 224, which Isaac Sim 6.0
reads; Blender's bundled module writes 225, which Isaac refuses - the
gate's G-VDBVER), exports the drone .usdc with Blender, writes a wrapper
scene that sublayers the native scene, hides its boundary surface (the
far-field box would enclose the cameras) and references the drone, and
judges the nine gates on the native outputs. Isaac Sim renders the wrapper
headless through tools/isaac/probe.py (run by the supervisor on the GPU box,
never here) with --profile given explicitly - its auto detection reads only
the root layer's text, and the volume lives in the sublayer - and --box set
to the lattice box.

The mesh and everything made from it derive from the BSD-3-Clause PX4 x500
model ("(c) 2022 Rudis Laboratories / PX4 Autopilot for Drones"), so
NOTHING is written inside the repository: every output goes under --out,
which must lie outside the repository, with the model's LICENSE.txt,
LICENSE and rotors.json copied beside the scene. The flow values are
demonstration numbers from an unsteady run, not validated; wake is
|U - U_inf| / |U_inf|, derived from U, not a transported scalar (the solve
transported no passive scalar).

Subcommands: export (solve dir + geom dir -> staged case, native vdb +
scene, drone usdc, wrapper scene, export.json), validate (one .usda -> the
check dict as JSON). --selftest runs T1-T12 (T11 drives
rust/target/release/ofgpu-sample.exe, ofgpu-generate-mesh.exe and
tools/drone/isaac_vdb.py under Isaac Sim's own Python).
"""

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
from pxr import Usd, UsdGeom, UsdUtils, UsdVol

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOL = "tools/drone/isaac_export.py"
VERSION = "drone-isaac/2"
CREDIT = "(c) 2022 Rudis Laboratories / PX4 Autopilot for Drones, BSD-3-Clause"
GRIDS = ["U.x", "U.y", "U.z", "U.mag", "p", "wake"]
VOXEL_DEFAULT = 0.01                                   # m
BOX_DEFAULT = (-1.2, -0.6, -0.8, 0.6, 0.6, 0.4)        # xlo, ylo, zlo, xhi, yhi, zhi (m): the drone plus its wake and downwash
MAX_VOXELS = 8_000_000
FLUID_MIN_FRAC = 0.9
FREESTREAM_TOL = 0.05                                  # relative
BLENDER = "C:/Program Files/Blender Foundation/Blender 5.1/blender.exe"
ISAAC_PYTHON = "C:/iss/env/Scripts/python.exe"
VDB_MAX_FILE_VERSION = 224  # Isaac Sim 6.0's reader (measured 2026-10-06: 225 refused)
SCENE_NAME = "drone_flow.usda"                         # the WRAPPER scene
DRONE_REL = "./drone/x500.usdc"
SAMPLE_EXE = os.path.join(REPO, "rust", "target", "release", "ofgpu-sample.exe")
GENMESH_EXE = os.path.join(REPO, "rust", "target", "release", "ofgpu-generate-mesh.exe")
USDA_CHECK = os.path.join(REPO, "tools", "isaac", "usda_check.py")
DENSITY = "density"
PROFILES = ("index", "pathtrace")
TAU_DEFAULT = 2.0
DEPTH_TOL = 0.01                                       # VIO-09's bar: within 1 %
STEM = "drone_flow"
NATIVE_DIR = "native"
WAKE = "wake"


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


# ----------------------------------------------------------------- lattice

def lattice_shape(lo, hi, voxel):
    """(nx, ny, nz) by (117.1) of SPEC-LIT 117.3: n_a = max(1, ceil(L_a / h - 1e-9)) -
    the voxel-centred lattice the native writer lays."""
    lo = tuple(float(v) for v in lo)
    hi = tuple(float(v) for v in hi)
    v = float(voxel)
    if not (math.isfinite(v) and v > 0.0):
        refuse("DI-BOX", "voxel " + repr(v) + " m: must be finite and > 0")
    for a in range(3):
        if lo[a] >= hi[a]:
            refuse("DI-BOX", "axis " + "xyz"[a] + ": lo " + repr(lo[a]) +
                   " >= hi " + repr(hi[a]))
    dims = tuple(max(1, int(math.ceil((hi[a] - lo[a]) / v - 1e-9))) for a in range(3))
    n = dims[0] * dims[1] * dims[2]
    if n > MAX_VOXELS:
        refuse("DI-BOX", "lattice " + repr(dims) + " is " + str(n) +
               " voxels, over the cap " + str(MAX_VOXELS))
    return dims


def lattice_centres(lo, voxel, dims):
    """(first, last) 3-tuples of float: the world positions of voxel
    centres (0,0,0) and (n-1,n-1,n-1) by (117.2) of SPEC-LIT 117.3,
    x_ijk = lo + h (i + 1/2, j + 1/2, k + 1/2)."""
    lo = tuple(float(v) for v in lo)
    h = float(voxel)
    first = tuple(lo[a] + h / 2.0 for a in range(3))
    last = tuple(lo[a] + h * (int(dims[a]) - 0.5) for a in range(3))
    return first, last


def wake_cells(U, U_inf):
    """(n,) float64 |U - U_inf| / |U_inf| - the writer's density source
    field; all zeros when |U_inf| == 0."""
    U = np.asarray(U, dtype=np.float64).reshape(-1, 3)
    uinf = np.asarray(U_inf, dtype=np.float64).reshape(3)
    speed = float(np.linalg.norm(uinf))
    if speed == 0.0:
        return np.zeros(U.shape[0], dtype=np.float64)
    return np.linalg.norm(U - uinf[None, :], axis=1) / speed


def write_foam_field(path, values):
    """One ascii OpenFOAM field: shape (n,) -> volScalarField /
    List<scalar>, (n, 3) -> volVectorField / List<vector>, one value per
    line (repr of each float; a vector as (x y z)); anything else raises
    ValueError. The solver reads only internalField."""
    v = np.asarray(values)
    if v.ndim == 1:
        cls, lst = "volScalarField", "List<scalar>"
        body = [repr(float(x)) for x in v]
    elif v.ndim == 2 and v.shape[1] == 3:
        cls, lst = "volVectorField", "List<vector>"
        body = ["(" + " ".join(repr(float(c)) for c in row) + ")" for row in v]
    else:
        raise ValueError("values shape " + repr(v.shape) + ": want (n,) or (n, 3)")
    lines = ["FoamFile", "{", "    version 2.0;", "    format ascii;",
             "    class " + cls + ";",
             "    object " + os.path.basename(str(path)) + ";", "}", "",
             "dimensions [0 0 0 0 0 0 0];",
             "internalField nonuniform " + lst, str(len(body)), "("]
    lines += body
    lines += [")", ";", "", "boundaryField", "{", "}", ""]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("".join(line + chr(10) for line in lines))


def stage_case(pm_dir, tdir, time_sel, out_case, uinf):
    """Copy the five polyMesh files, U and p, then write the writer's third
    field: wake = |U - U_inf| / |U_inf| per cell. Returns n_cells. Never
    writes into pm_dir/tdir."""
    pm_out = os.path.join(out_case, "constant", "polyMesh")
    os.makedirs(pm_out, exist_ok=True)
    for f in ("boundary", "faces", "owner", "neighbour", "points"):
        shutil.copy2(os.path.join(pm_dir, f), os.path.join(pm_out, f))
    t_out = os.path.join(out_case, str(time_sel))
    os.makedirs(t_out, exist_ok=True)
    for f in ("U", "p"):
        shutil.copy2(os.path.join(tdir, f), os.path.join(t_out, f))
    owner = PROMO.read_ints(os.path.join(pm_dir, "owner"))
    neighbour = PROMO.read_ints(os.path.join(pm_dir, "neighbour"))
    n_cells = int(owner.max()) + 1
    if neighbour.size:
        n_cells = max(n_cells, int(neighbour.max()) + 1)
    U = PROMO_POST.read_foam_vector(os.path.join(tdir, "U"), n_cells)
    write_foam_field(os.path.join(t_out, WAKE), wake_cells(U, uinf))
    return n_cells


# ------------------------------------------------------------- the writer

def lattice_json(lo, hi, voxel):
    """-lattice payload (SPEC-LIT 117.3): the box and the voxel."""
    return json.dumps({"voxel": float(voxel),
                       "box": [float(v) for v in tuple(lo) + tuple(hi)]})


def isaac_json(profile, field, tau):
    """-isaac payload (SPEC-LIT 117.9): the profile, the density source
    field and the optical-depth scaling target."""
    if profile not in PROFILES:
        refuse("DI-PROFILE", repr(profile) + ": profile is one of " +
               ", ".join(PROFILES))
    return json.dumps({"profile": profile, "field": field,
                       "scaling": "opticalDepth", "tau": float(tau)})


def sample_cmd(exe, case, time_sel, lo, hi, voxel, profile, tau, ux, out_dir):
    """The one writer command: volume <case> <time> -vdb -usda -lattice
    ... -isaac ... -freestream ... -out DIR -stem NAME."""
    return [str(exe), "volume", str(case), str(time_sel), "-vdb", "-usda",
            "-lattice", lattice_json(lo, hi, voxel),
            "-isaac", isaac_json(profile, WAKE, tau),
            "-freestream", repr(float(ux)), "-out", str(out_dir), "-stem", STEM]


_LATTICE_RE = re.compile(
    r"^\s*lattice (\d+) x (\d+) x (\d+) of h = (\S+) \(requested (\S+), "
    r"coarsen (\S+)\): (\d+) voxels, (\d+) fluid, (\d+) gap-filled; "
    r"fluid fraction (\S+)\s*$", re.M)
_FREESTREAM_RE = re.compile(
    r"^\s*freestream: median U\.x over the (\d+ .+) = (\S+); "
    r"\|median - Ux\|/\|Ux\| = (\S+) against Ux = \S+\s*$", re.M)
_HELD_RE = re.compile(
    r"optical depth held from this write on: range \[(\S+), (\S+)\], "
    r"D = (\S+) m \(tau = (\S+)\)")
_WROTE_RE = re.compile(r"^\s*wrote (.+?)\s*$", re.M)


def parse_sample_report(text):
    """The writer's stdout -> dict. Refuses DI-SAMPLE on a uniform
    Cartesian box (it lays no lattice: not this tool's mesh), or a report
    without a lattice line or a freestream line."""
    if "uniform Cartesian box" in text:
        refuse("DI-SAMPLE", "the writer laid a uniform Cartesian box: "
               "not this tool's mesh")
    m = _LATTICE_RE.search(text)
    if not m:
        refuse("DI-SAMPLE", "the report has no lattice line")
    f = _FREESTREAM_RE.search(text)
    if not f:
        refuse("DI-SAMPLE", "the report has no freestream line")
    held = _HELD_RE.search(text)
    return {
        "n": [int(m.group(1)), int(m.group(2)), int(m.group(3))],
        "voxel": float(m.group(4)),
        "n_voxels": int(m.group(7)),
        "n_fluid": int(m.group(8)),
        "fluid_frac": float(m.group(10)),
        "freestream": {"plane": f.group(1), "median_Ux": float(f.group(2)),
                       "rel_err": float(f.group(3))},
        "held": ({"range": [float(held.group(1)), float(held.group(2))],
                  "D": float(held.group(3)), "tau": float(held.group(4))}
                 if held else None),
        "density_zero": "writing an all-zero density grid" in text,
        "wrote": [w for w in _WROTE_RE.findall(text)],
    }


def read_vdb(vdb_python, vdb_path, dims, manifest, log_path):
    """tools/drone/isaac_vdb.py --read under Isaac Sim's own Python: the
    file as Isaac's reader sees it, measured into the manifest. Returns
    the loaded manifest; non-zero exit or no manifest refuses DI-VDB."""
    dims_s = ",".join(str(int(v)) for v in dims)
    rc = _run_writer([vdb_python, os.path.join(REPO, "tools", "drone",
                                              "isaac_vdb.py"),
                      "--read", str(vdb_path), "--dims", dims_s,
                      "--manifest", str(manifest)], log_path)
    if rc != 0:
        refuse("DI-VDB", "isaac_vdb.py --read exited " + str(rc) +
               " (see writers.log)")
    if not os.path.isfile(str(manifest)):
        refuse("DI-VDB", _posix(manifest) + ": vdb manifest missing (see writers.log)")
    with open(str(manifest), encoding="utf-8") as fh:
        return json.load(fh)


def write_wrapper(path, native_rel, drone_rel, doc):
    """The wrapper scene: the native writer's scene as the first
    sublayer, the stage's metadata, the native /World/boundary made
    invisible in this layer, and the drone referenced at /World/Drone."""
    stage = Usd.Stage.CreateNew(str(path))
    layer = stage.GetRootLayer()
    layer.subLayerPaths.append(str(native_rel))
    layer.defaultPrim = "World"
    layer.documentation = doc
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    bnd = stage.GetPrimAtPath("/World/boundary")
    if bool(bnd) and bnd.IsActive():
        UsdGeom.Imageable(bnd).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    if drone_rel is not None:
        drone = UsdGeom.Xform.Define(stage, "/World/Drone")
        drone.GetPrim().GetReferences().AddReference(str(drone_rel))
    stage.Save()


def validate_scene(path, profile, want_drone=True):
    """Open the wrapper scene back and check everything - never raises.
    The profile rule: index -> one Volume carrying the density field and
    no volume mesh; pathtrace -> one volume Mesh and no Volume."""
    res = {"opens": False, "up_axis": "", "meters_per_unit": 0.0, "default_prim": "",
           "volumes": 0, "volume_meshes": 0, "fields": [], "boundary_hidden": False,
           "drone_meshes": 0, "unresolved": [], "ok": False}
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
        prims = list(stage.Traverse())
        vols = [p for p in prims if p.IsA(UsdVol.Volume)]
        res["volumes"] = len(vols)
        meshes = [p for p in prims if p.IsA(UsdGeom.Mesh)]
        res["volume_meshes"] = sum(
            1 for p in meshes if p.GetAttribute("primvars:isVolume").Get() is True)
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
        bnd = stage.GetPrimAtPath("/World/boundary")
        res["boundary_hidden"] = bool(
            bnd and bnd.IsActive()
            and str(UsdGeom.Imageable(bnd).ComputeVisibility())
            == str(UsdGeom.Tokens.invisible))
        res["drone_meshes"] = sum(1 for p in meshes
                                  if str(p.GetPath()).startswith("/World/Drone"))
        missing = []
        _layers, _assets, unresolved = UsdUtils.ComputeAllDependencies(str(path))
        missing.extend(str(u) for u in unresolved if not str(u).endswith(".mdl"))
        for p in prims:
            for attr in p.GetAttributes():
                if str(attr.GetTypeName()) not in ("asset", "asset[]"):
                    continue
                for val in [attr.Get()] + [attr.Get(t) for t in attr.GetTimeSamples()]:
                    if val is None or not str(getattr(val, "path", "")).endswith(".vdb"):
                        continue
                    rp = str(getattr(val, "resolvedPath", "") or "")
                    if not rp or not os.path.isfile(rp):
                        missing.append(str(val.path))
        res["unresolved"] = missing
    except Exception:
        pass
    if profile == "pathtrace":
        prof_ok = res["volumes"] == 0 and res["volume_meshes"] == 1
    else:
        prof_ok = (res["volumes"] == 1 and res["volume_meshes"] == 0
                   and res["fields"] == ["density"])
    drone_ok = res["drone_meshes"] >= 1 if want_drone else True
    res["ok"] = bool(res["opens"] and res["up_axis"] == "Z"
                     and res["meters_per_unit"] == 1.0
                     and res["default_prim"] == "/World" and res["boundary_hidden"]
                     and res["unresolved"] == [] and prof_ok and drone_ok)
    return res


def run_usda_check(path, profile):
    """tools/isaac/usda_check.py on the native scene (a subprocess of this
    interpreter, so it works under plain python and Isaac's alike)."""
    p = subprocess.run([sys.executable, USDA_CHECK, str(path), "--profile", profile],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    return {"ok": p.returncode == 0, "returncode": p.returncode}


# ------------------------------------------------------------------ gate

GATE_IDS = ["G-GRIDS", "G-DENSE", "G-ROUNDTRIP", "G-PLACE", "G-FLUID",
            "G-FREESTREAM", "G-USD", "G-DRONE", "G-VDBVER"]


def judge(ex):
    """The export gate on the native outputs, ids in table order."""
    failed = []
    lat = ex["lattice"]
    grids = ex["grids"]
    if sorted(g["name"] for g in grids) != sorted(GRIDS + [DENSITY]):
        failed.append("G-GRIDS")
    exp = int(lat["expected_active"])
    if any(int(g["active_voxels"]) != exp for g in grids):
        failed.append("G-DENSE")
    rel = float((ex.get("density") or {}).get("rel_err", float("inf")))
    if any(not bool(g.get("finite")) for g in grids) or rel > DEPTH_TOL:
        failed.append("G-ROUNDTRIP")
    first = [float(v) for v in lat["first"]]
    last = [float(v) for v in lat["last"]]
    if ([int(v) for v in lat["report_dims"]] != [int(v) for v in lat["dims"]]
            or any(any(abs(float(g["world_first"][i]) - first[i]) > 1e-9
                       or abs(float(g["world_last"][i]) - last[i]) > 1e-9
                       for i in range(3)) for g in grids)):
        failed.append("G-PLACE")
    if float(ex["fluid_frac"]) < FLUID_MIN_FRAC:
        failed.append("G-FLUID")
    if float(ex["freestream"]["rel_err"]) > FREESTREAM_TOL:
        failed.append("G-FREESTREAM")
    if not ex["scene"].get("ok") or not (ex.get("native_check") or {}).get("ok"):
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
    """One writer subprocess - ofgpu-sample, isaac_vdb.py under Isaac's
    Python, or blender -b - with stdout+stderr streamed into the log; all
    append. Returns the exit code; the caller refuses."""
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
    """Stage the case, run ofgpu-sample volume -isaac, read the .vdb back,
    export the drone, write the wrapper scene, judge the nine gates."""
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
    sample_exe = os.path.abspath(args.sample_exe)
    voxel = float(args.voxel)
    tau = float(args.tau)
    profile = str(args.profile)
    t0 = time.monotonic()
    _check_out(out)
    pm_dir, tdir, time_sel = _check_inputs(solve, str(args.time))
    lo, hi = _parse_box(args.box)
    dims = lattice_shape(lo, hi, voxel)
    if not os.path.isfile(blender):
        refuse("DI-BLENDER", _posix(blender) + ": blender.exe not found")
    if not os.path.isfile(sample_exe):
        refuse("DI-SAMPLE", _posix(sample_exe) + ": ofgpu-sample not found")
    if not os.path.isfile(vdb_python):
        refuse("DI-VDB", _posix(vdb_python) + ": the vdb python not found")
    stage("checks", t0)

    t0 = time.monotonic()
    with open(os.path.join(solve, "write.json"), encoding="utf-8") as fh:
        uinf = np.asarray(json.load(fh)["params"]["U_vec"], dtype=np.float64)
    stage_case(pm_dir, tdir, time_sel, os.path.join(out, "case"), uinf)
    stage("stage", t0)

    t0 = time.monotonic()
    cmd = sample_cmd(sample_exe, os.path.join(out, "case"), time_sel, lo, hi,
                     voxel, profile, tau, float(uinf[0]), os.path.join(out, NATIVE_DIR))
    rc = _run_writer(cmd, os.path.join(out, "sample.log"), first=True)
    if rc != 0:
        refuse("DI-SAMPLE", "ofgpu-sample exited " + str(rc) +
               " (see sample.log)")
    with open(os.path.join(out, "sample.log"), encoding="utf-8", errors="replace") as fh:
        rep = parse_sample_report(fh.read())
    stage("sample", t0)

    t0 = time.monotonic()
    vdb_path = os.path.join(out, NATIVE_DIR, "VDB", STEM + "_000000.vdb")
    native = os.path.join(out, NATIVE_DIR, STEM + ".usda")
    man_v = os.path.join(out, "vdb_manifest.json")
    wlog = os.path.join(out, "writers.log")
    open(wlog, "wb").close()   # a fresh log: read_vdb and Blender append
    vdb_manifest = read_vdb(vdb_python, vdb_path, dims, man_v, wlog)
    bpy_script = os.path.join(REPO, "tools", "drone", "isaac_bpy.py")
    usdc_path = os.path.join(out, "drone", "x500.usdc")
    man_u = os.path.join(out, "usd_manifest.json")
    rc = _run_writer([blender, "-b", "--factory-startup",
                      os.path.join(geom, "x500_assembled.blend"), "--python", bpy_script,
                      "--", "usd", "--out", usdc_path, "--manifest", man_u], wlog)
    if rc != 0:
        refuse("DI-BLENDER", "blender exited " + str(rc) + " (see writers.log)")
    if not os.path.isfile(man_u):
        refuse("DI-BLENDER", _posix(man_u) + ": usd manifest missing (see writers.log)")
    with open(man_u, encoding="utf-8") as fh:
        usd_manifest = json.load(fh)
    stage("writers", t0)

    t0 = time.monotonic()
    scene_path = os.path.join(out, SCENE_NAME)
    doc = (CREDIT + "; drone solve time " + time_sel + " s; flow values are demonstration "
           "numbers from an unsteady run, not validated; wake = |U - U_inf| / |U_inf|, "
           "derived from U, not a transported scalar")
    write_wrapper(scene_path, "./" + NATIVE_DIR + "/" + STEM + ".usda", DRONE_REL, doc)
    scene = validate_scene(scene_path, profile)
    native_check = run_usda_check(native, profile)
    stage("scene", t0)

    t0 = time.monotonic()
    for f in ("LICENSE.txt", "LICENSE", "rotors.json"):
        src = os.path.join(solve, f)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(out, f))
    stage("copy", t0)

    t0 = time.monotonic()
    files = {}
    for p in (scene_path, native, vdb_path, usdc_path):
        rel = _posix(os.path.relpath(p, out))
        files[rel] = {"bytes": int(os.path.getsize(p)), "sha256": _digest(p)}
    stage("files", t0)

    t0 = time.monotonic()
    rep_voxel = float(rep["voxel"])
    first, last = lattice_centres(lo, rep_voxel, dims)
    grids = vdb_manifest.get("grids", [])
    dens = [g for g in grids if g.get("name") == DENSITY]
    max_col = float(dens[0]["max_column"]) if dens else None
    ex = {
        "tool": TOOL, "version": VERSION, "credit": CREDIT,
        "profile": profile, "tau": tau,
        "solve_dir": _posix(solve), "time": time_sel,
        "sample_cmd": " ".join(_posix(a) for a in cmd),
        "lattice": {"lo": [float(v) for v in lo], "hi": [float(v) for v in hi],
                    "voxel": rep_voxel, "dims": list(dims),
                    "report_dims": [int(v) for v in rep["n"]],
                    "first": [float(v) for v in first],
                    "last": [float(v) for v in last],
                    "n_voxels": rep["n_voxels"], "n_fluid": rep["n_fluid"],
                    "activation": "fluid" if profile == "index" else "dense",
                    "expected_active": (rep["n_fluid"] if profile == "index"
                                        else rep["n_voxels"])},
        "U_inf": [float(v) for v in uinf],
        "fluid_frac": rep["fluid_frac"],
        "freestream": rep["freestream"],
        "held": rep["held"],
        "density_zero": rep["density_zero"],
        "grids": grids,
        "density": {"tau": tau, "max_column": max_col,
                    "rel_err": (abs(max_col - tau) / tau if dens else float("inf"))},
        "vdb": {"file_version": vdb_manifest.get("file_version"),
                "openvdb": vdb_manifest.get("openvdb"),
                "library_version": vdb_manifest.get("library_version")},
        "usd": usd_manifest,
        "scene": scene,
        "native_check": native_check,
        "files": files,
        "timing_s": times,
        "notes": [
            "the solver's ofgpu-sample volume -isaac wrote the lattice .vdb and the "
            "native scene; this tool stages the case and judges",
            "wake = |U - U_inf| / |U_inf|, derived from U per cell, not a "
            "transported scalar",
            "flow values are demonstration numbers from an unsteady run, not validated",
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
    d = validate_scene(args.scene, args.profile)
    print(json.dumps(d, indent=2))
    return 0 if d["ok"] else 1


# ---------------------------------------------------------------- selftest

def _t1():
    assert lattice_shape((-1.2, -0.6, -0.8), (0.6, 0.6, 0.4), 0.01) == (180, 120, 120)
    assert lattice_shape((0, 0, 0), (4, 2, 0.1), 0.05) == (80, 40, 2)
    bad = ((0.0, (-1.2, -0.6, -0.8), (0.6, 0.6, 0.4)),
           (float("nan"), (-1.2, -0.6, -0.8), (0.6, 0.6, 0.4)),
           (0.01, (0.0, 0.0, 0.0), (0.0, 1.0, 1.0)),
           (0.001, (-1.2, -0.6, -0.8), (0.6, 0.6, 0.4)))
    for v, blo, bhi in bad:
        code = 0
        try:
            lattice_shape(blo, bhi, v)
        except SystemExit as exc:
            code = int(exc.code or 0)
        assert code == 2, (v, blo, bhi, code)


def _t2():
    first, last = lattice_centres((-1.2, -0.6, -0.8), 0.01, (180, 120, 120))
    wf, wl = (-1.195, -0.595, -0.795), (0.595, 0.595, 0.395)
    for i in range(3):
        assert abs(first[i] - wf[i]) < 1e-12, (first, wf)
        assert abs(last[i] - wl[i]) < 1e-12, (last, wl)


def _t3():
    tmp = tempfile.mkdtemp(prefix="drone-isaac-t3-")
    try:
        vals = [0.1, -2.5e-7, 3.0]
        p = os.path.join(tmp, "w")
        write_foam_field(p, np.asarray(vals, dtype=np.float64))
        assert read_foam_scalar(p, 3).tolist() == vals, read_foam_scalar(p, 3)
        text = open(p, encoding="utf-8").read()
        assert "class volScalarField;" in text, text
        vmat = [[0.1, -2.5e-7, 3.0], [4.0, 5.5, 6.25]]
        q = os.path.join(tmp, "wv")
        write_foam_field(q, np.asarray(vmat, dtype=np.float64))
        back = PROMO_POST.read_foam_vector(q, 2)
        assert back.tolist() == vmat, back
        tv = open(q, encoding="utf-8").read()
        assert "class volVectorField;" in tv, tv
        bad = False
        try:
            write_foam_field(os.path.join(tmp, "wb"), np.zeros((2, 2)))
        except ValueError:
            bad = True
        assert bad, "shape (2, 2) must raise ValueError"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _t4():
    U = np.array([[-15.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-12.0, 0.0, -4.0]])
    w = wake_cells(U, (-15.0, 0.0, 0.0))
    assert w.dtype == np.float64, w.dtype
    assert np.allclose(w, [0.0, 1.0, 1.0 / 3.0], rtol=0.0, atol=1e-15), w
    z = wake_cells(np.ones((4, 3)), (0.0, 0.0, 0.0))
    assert z.dtype == np.float64 and np.array_equal(z, np.zeros(4)), z


def _t5():
    assert lattice_json((-1.2, -0.6, -0.8), (0.6, 0.6, 0.4), 0.01) == \
        '{"voxel": 0.01, "box": [-1.2, -0.6, -0.8, 0.6, 0.6, 0.4]}'
    assert isaac_json("index", "wake", 2.0) == \
        '{"profile": "index", "field": "wake", "scaling": "opticalDepth", "tau": 2.0}'
    code = 0
    try:
        isaac_json("nvdb", "wake", 2.0)
    except SystemExit as exc:
        code = int(exc.code or 0)
    assert code == 2, code
    cmd = sample_cmd("s.exe", "c", "0.5", (0, 0, 0), (1, 1, 1), 0.5, "pathtrace", 2.0,
                     -15.0, "o")
    assert cmd == ["s.exe", "volume", "c", "0.5", "-vdb", "-usda", "-lattice",
                   '{"voxel": 0.5, "box": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]}', "-isaac",
                   '{"profile": "pathtrace", "field": "wake", "scaling": "opticalDepth", '
                   '"tau": 2.0}', "-freestream", "-15.0", "-out", "o",
                   "-stem", "drone_flow"], cmd


def _t6():
    from pxr import Usd, UsdGeom
    tmp = tempfile.mkdtemp(prefix="drone-isaac-t6-")
    try:
        os.makedirs(os.path.join(tmp, "native", "VDB"))
        os.makedirs(os.path.join(tmp, "drone"))
        shutil.copy2(os.path.join(REPO, "tools", "isaac", "fixtures", "index",
                                  "isaac.usda"),
                     os.path.join(tmp, "native", "drone_flow.usda"))
        with open(os.path.join(tmp, "native", "VDB", "isaac_000000.vdb"), "wb") as fh:
            fh.write(b"")
        st = Usd.Stage.CreateNew(os.path.join(tmp, "drone", "x500.usda"))
        UsdGeom.Xform.Define(st, "/root")
        UsdGeom.Mesh.Define(st, "/root/cube")
        st.GetRootLayer().defaultPrim = "root"
        st.Save()
        scene = os.path.join(tmp, "drone_flow.usda")
        write_wrapper(scene, "./native/drone_flow.usda", "./drone/x500.usda", "doc")
        v = validate_scene(scene, "index")
        assert v["ok"] is True, v
        assert v["volumes"] == 1 and v["volume_meshes"] == 0, v
        assert v["fields"] == ["density"] and v["boundary_hidden"] is True, v
        assert v["drone_meshes"] == 1 and v["unresolved"] == [], v
        v2 = validate_scene(scene, "pathtrace")
        assert v2["ok"] is False, v2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _t7():
    tmp = tempfile.mkdtemp(prefix="drone-isaac-t7-")
    try:
        os.makedirs(os.path.join(tmp, "native", "VDB"))
        shutil.copy2(os.path.join(REPO, "tools", "isaac", "fixtures", "index",
                                  "isaac.usda"),
                     os.path.join(tmp, "native", "drone_flow.usda"))
        vdbp = os.path.join(tmp, "native", "VDB", "isaac_000000.vdb")
        open(vdbp, "wb").close()
        scene = os.path.join(tmp, "drone_flow.usda")
        write_wrapper(scene, "./native/drone_flow.usda", None, "doc")
        os.remove(vdbp)
        v = validate_scene(scene, "index", want_drone=False)
        assert v["ok"] is False and v["unresolved"], v
        junk = os.path.join(tmp, "junk.usda")
        with open(junk, "w", encoding="utf-8") as fh:
            fh.write("this is not a usd file" + chr(10))
        vj = validate_scene(junk, "index", want_drone=False)
        assert vj["opens"] is False and vj["ok"] is False, vj
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    tmp = tempfile.mkdtemp(prefix="drone-isaac-t7b-")
    try:
        os.makedirs(os.path.join(tmp, "native", "VDB"))
        shutil.copy2(os.path.join(REPO, "tools", "isaac", "fixtures", "pathtrace",
                                  "isaac.usda"),
                     os.path.join(tmp, "native", "drone_flow.usda"))
        with open(os.path.join(tmp, "native", "VDB", "isaac_000000.vdb"), "wb") as fh:
            fh.write(b"")
        scene = os.path.join(tmp, "drone_flow.usda")
        write_wrapper(scene, "./native/drone_flow.usda", None, "doc")
        vp = validate_scene(scene, "pathtrace", want_drone=False)
        assert vp["ok"] is True, vp
        assert vp["volumes"] == 0 and vp["volume_meshes"] == 1, vp
        assert vp["drone_meshes"] == 0, vp
        assert validate_scene(scene, "pathtrace", want_drone=True)["ok"] is False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


_T8_TEXT = chr(10).join([
    "volume of t at time 1: 192 cells; fields U, wake (of U, wake)",
    "  lattice 80 x 40 x 2 of h = 0.05 (requested 0.05, coarsen 1): 6400 voxels, "
    "6400 fluid, 0 gap-filled; fluid fraction 1",
    "  lattice built in 0.000 s",
    "  freestream: median U.x over the 80 fluid voxels of the xMax lattice plane "
    "(i = 79) = 1.75; |median - Ux|/|Ux| = 0.75 against Ux = 1",
    "output.visualisation.isaac: optical depth held from this write on: range "
    "[0.0, 0.19], D = 3.200263157894737 m (tau = 2.0), at t = 1 s (SPEC-LIT 117.9)",
    "  wrote t/out/VDB/isaac_000000.vdb",
    "  wrote t/out/isaac.usda",
    "  total 0.0 s",
    ""])


def _t8():
    rep = parse_sample_report(_T8_TEXT)
    assert rep["n"] == [80, 40, 2] and rep["voxel"] == 0.05, rep
    assert rep["n_voxels"] == 6400 and rep["n_fluid"] == 6400, rep
    assert rep["fluid_frac"] == 1.0, rep
    fs = rep["freestream"]
    assert fs["plane"] == "80 fluid voxels of the xMax lattice plane (i = 79)", fs
    assert fs["median_Ux"] == 1.75 and fs["rel_err"] == 0.75, fs
    assert rep["held"]["range"] == [0.0, 0.19], rep
    assert rep["held"]["D"] == 3.200263157894737 and rep["held"]["tau"] == 2.0, rep
    assert rep["density_zero"] is False, rep
    assert rep["wrote"] == ["t/out/VDB/isaac_000000.vdb", "t/out/isaac.usda"], rep
    nofree = chr(10).join(l for l in _T8_TEXT.splitlines() if "freestream" not in l)
    boxed = _T8_TEXT.replace(
        "  lattice 80 x 40 x 2 of h = 0.05 (requested 0.05, coarsen 1): 6400 voxels, "
        "6400 fluid, 0 gap-filled; fluid fraction 1",
        "  uniform Cartesian box 8 x 6 x 4: the cells are the voxels")
    for bad in (nofree, boxed):
        code = 0
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                parse_sample_report(bad)
        except SystemExit as exc:
            code = int(exc.code or 0)
        assert code == 2, code
        assert "refused: DI-SAMPLE" in buf.getvalue(), buf.getvalue()
    d0 = parse_sample_report(_T8_TEXT.replace(
        "output.visualisation.isaac: optical depth held from this write on: range "
        "[0.0, 0.19], D = 3.200263157894737 m (tau = 2.0), at t = 1 s (SPEC-LIT 117.9)",
        "output.visualisation.isaac: D = 0 at t = 1 s - writing an all-zero density "
        "grid (SPEC-LIT 117.9)"))
    assert d0["held"] is None and d0["density_zero"] is True, d0


def _t9():
    lat = {"lo": [0.0, 0.0, 0.0], "hi": [2.4, 2.4, 2.4], "voxel": 0.4,
           "dims": [6, 6, 6], "report_dims": [6, 6, 6],
           "first": [0.2, 0.2, 0.2], "last": [2.2, 2.2, 2.2],
           "n_voxels": 216, "n_fluid": 216,
           "activation": "dense", "expected_active": 216}
    grids = [{"name": n, "active_voxels": 216, "finite": True,
              "world_first": [0.2, 0.2, 0.2], "world_last": [2.2, 2.2, 2.2]}
             for n in sorted(GRIDS + [DENSITY])]
    ex = {"lattice": lat, "grids": grids,
          "density": {"tau": 2.0, "max_column": 2.00000001, "rel_err": 5e-9},
          "fluid_frac": 0.95,
          "freestream": {"plane": "80 fluid voxels of the xMax lattice plane",
                         "median_Ux": 1.0, "rel_err": 0.01},
          "scene": {"ok": True, "drone_meshes": 25},
          "native_check": {"ok": True, "returncode": 0},
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

    def b_grids(e):
        e["grids"][2]["name"] = "zzz"

    def b_dense(e):
        e["grids"][1]["active_voxels"] = 5

    def b_round(e):
        e["density"]["rel_err"] = 0.02

    def b_place(e):
        e["grids"][4]["world_last"] = [9.0, 2.2, 2.2]

    def b_fluid(e):
        e["fluid_frac"] = 0.5

    def b_free(e):
        e["freestream"]["rel_err"] = 0.5

    def b_usd(e):
        e["native_check"]["ok"] = False

    def b_drone(e):
        e["usd"]["mesh_objects"] = 24

    def b_vdb(e):
        e["vdb"]["file_version"] = 225

    for want, fn in zip(GATE_IDS, (b_grids, b_dense, b_round, b_place, b_fluid,
                                   b_free, b_usd, b_drone, b_vdb)):
        got = broken(fn)
        assert got == want, (want, got)
    extras = [("G-ROUNDTRIP", lambda e: e["grids"][3].__setitem__("finite", False)),
              ("G-PLACE", lambda e: e["lattice"].__setitem__("report_dims", [6, 6, 5])),
              ("G-USD", lambda e: e["scene"].__setitem__("ok", False))]
    for want, fn in extras:
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
            ("DI-SAMPLE", ["export", "--solve", os.path.join(tmp, "full"), "--geom",
                           os.path.join(tmp, "geom"), "--out", os.path.join(tmp, "o5"),
                           "--sample-exe", os.path.join(tmp, "no-sample.exe")]),
            ("DI-VDB", ["export", "--solve", os.path.join(tmp, "full"), "--geom",
                        os.path.join(tmp, "geom"), "--out", os.path.join(tmp, "o6"),
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


def _t11():
    tmp = tempfile.mkdtemp(prefix="drone-isaac-t11-")
    try:
        gen = os.path.join(tmp, "gen")
        p = subprocess.run([GENMESH_EXE, "channel", gen, "8", "6", "4",
                            "-grading", "x=2"], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", cwd=REPO)
        assert p.returncode == 0, (p.returncode, p.stdout[-800:], p.stderr[-800:])
        gen_pm = os.path.join(gen, "constant", "polyMesh")
        if not os.path.isfile(os.path.join(gen_pm, "owner")):
            gen_pm = gen
        solve = os.path.join(tmp, "solve")
        pm = os.path.join(solve, "case", "constant", "polyMesh")
        os.makedirs(pm)
        for f in ("boundary", "faces", "owner", "neighbour", "points"):
            shutil.copy2(os.path.join(gen_pm, f), os.path.join(pm, f))
        with open(os.path.join(solve, "write.json"), "w", encoding="utf-8") as fh:
            json.dump({"params": {"U_vec": [1.0, 0.0, 0.0]}}, fh)
        t1 = os.path.join(solve, "case", "1")
        os.makedirs(t1)
        n0 = int(PROMO.read_ints(os.path.join(pm, "owner")).max()) + 1
        assert n0 == 192, n0
        write_foam_field(os.path.join(t1, "U"),
                         np.column_stack([1.0 + 0.01 * np.arange(n0, dtype=np.float64),
                                          np.zeros(n0), np.zeros(n0)]))
        write_foam_field(os.path.join(t1, "p"), np.zeros(n0))
        blo, bhi = (0.0, -1.0, 0.0), (4.0, 1.0, 0.1)
        want_dims = lattice_shape(blo, bhi, 0.05)
        want_first, want_last = lattice_centres(blo, 0.05, want_dims)
        assert want_dims == (80, 40, 2), want_dims
        for profile in PROFILES:
            _t11_profile(tmp, solve, t1, profile, blo, bhi, want_dims,
                         want_first, want_last)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _t11_profile(tmp, solve, t1, profile, blo, bhi, want_dims, want_first, want_last):
    o = os.path.join(tmp, "o_" + profile)
    n = stage_case(os.path.join(solve, "case", "constant", "polyMesh"), t1, "1",
                   os.path.join(o, "case"), np.array([1.0, 0.0, 0.0]))
    assert n == 192, n
    cmd = sample_cmd(SAMPLE_EXE, os.path.join(o, "case"), "1", blo, bhi, 0.05,
                     profile, 2.0, 1.0, os.path.join(o, NATIVE_DIR))
    rc = _run_writer(cmd, os.path.join(o, "sample.log"), first=True)
    with open(os.path.join(o, "sample.log"), encoding="utf-8", errors="replace") as fh:
        log = fh.read()
    assert rc == 0, (profile, rc, log[-1500:])
    rep = parse_sample_report(log)
    assert rep["n"] == [80, 40, 2] == list(want_dims), (profile, rep["n"])
    assert rep["n_voxels"] == 6400 and rep["n_fluid"] == 6400, (profile, rep)
    vdbp = os.path.join(o, NATIVE_DIR, "VDB", STEM + "_000000.vdb")
    native = os.path.join(o, NATIVE_DIR, STEM + ".usda")
    man = read_vdb(ISAAC_PYTHON, vdbp, want_dims,
                   os.path.join(o, "vdb_manifest.json"), os.path.join(o, "writers.log"))
    assert [g["name"] for g in man["grids"]] == sorted(GRIDS + [DENSITY]), \
        (profile, [g["name"] for g in man["grids"]])
    for g in man["grids"]:
        assert int(g["active_voxels"]) == 6400, (profile, g["name"])
        assert g["finite"] is True, (profile, g["name"])
        for i in range(3):
            assert abs(float(g["world_first"][i]) - want_first[i]) < 1e-9, \
                (profile, g["name"], g["world_first"])
            assert abs(float(g["world_last"][i]) - want_last[i]) < 1e-9, \
                (profile, g["name"], g["world_last"])
    mc = float([g for g in man["grids"] if g["name"] == DENSITY][0]["max_column"])
    assert abs(mc - 2.0) <= DEPTH_TOL * 2.0, (profile, mc)
    assert int(man["file_version"]) <= VDB_MAX_FILE_VERSION, man["file_version"]
    chk = run_usda_check(native, profile)
    assert chk["ok"] is True, (profile, chk)
    sc = os.path.join(o, SCENE_NAME)
    write_wrapper(sc, "./" + NATIVE_DIR + "/" + STEM + ".usda", None, "doc")
    v = validate_scene(sc, profile, want_drone=False)
    assert v["ok"] is True, (profile, v)


def _t12():
    tmp = tempfile.mkdtemp(prefix="drone-isaac-t12-")
    try:
        pm = os.path.join(tmp, "pm")
        os.makedirs(pm)

        def labels(name, values):
            path = os.path.join(pm, name)
            lines = ["FoamFile", "{", "    version 2.0;", "    format ascii;",
                     "    class labelList;", "    object " + name + ";", "}",
                     "", str(len(values)), "("]
            lines += [str(int(v)) for v in values]
            lines += [")", ""]
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write("".join(line + chr(10) for line in lines))

        labels("owner", [0, 0, 1])
        labels("neighbour", [1, 2])
        assert PROMO.read_ints(os.path.join(pm, "owner")).tolist() == [0, 0, 1], \
            "owner layout not parsed by PROMO.read_ints"
        assert PROMO.read_ints(os.path.join(pm, "neighbour")).tolist() == [1, 2], \
            "neighbour layout not parsed by PROMO.read_ints"
        for f in ("boundary", "faces", "points"):
            with open(os.path.join(pm, f), "w", encoding="utf-8", newline="") as fh:
                fh.write("dummy" + chr(10))
        t1 = os.path.join(tmp, "1")
        os.makedirs(t1)
        write_foam_field(os.path.join(t1, "U"),
                         [[1, 0, 0], [2, 0, 0], [4, 0, 0]])
        write_foam_field(os.path.join(t1, "p"), np.zeros(3))
        out_case = os.path.join(tmp, "out_case")
        n = stage_case(pm, t1, "1", out_case, (1, 0, 0))
        assert n == 3, n
        wake = read_foam_scalar(os.path.join(out_case, "1", WAKE), 3)
        assert wake.tolist() == [0.0, 1.0, 3.0], wake.tolist()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def selftest():
    """T1-T12; T11 drives rust/target/release/ofgpu-sample.exe,
    ofgpu-generate-mesh.exe and tools/drone/isaac_vdb.py under Isaac Sim's
    own Python. Temp files only under tempfile.mkdtemp()."""
    tests = [("T1", _t1), ("T2", _t2), ("T3", _t3), ("T4", _t4), ("T5", _t5),
             ("T6", _t6), ("T7", _t7), ("T8", _t8), ("T9", _t9), ("T10", _t10),
             ("T11", _t11), ("T12", _t12)]
    npass = 0
    for name, fn in tests:
        try:
            fn()
            print("[ok] " + name, flush=True)
            npass += 1
        except (Exception, SystemExit) as exc:   # a refuse() inside a test fails that test only
            print("[FAIL] " + name + ": " + type(exc).__name__ + " " + str(exc), flush=True)
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
    ep.add_argument("--profile", choices=PROFILES, default="index")
    ep.add_argument("--tau", type=float, default=TAU_DEFAULT)
    ep.add_argument("--sample-exe", default=SAMPLE_EXE)
    vp = sub.add_parser("validate")
    vp.add_argument("--scene", required=True)
    vp.add_argument("--profile", choices=PROFILES, default="index")
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


















