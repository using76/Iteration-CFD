#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.

"""The F1 promo video's wind-tunnel mesh: config, run, -check, gate.

The tunnel around the car (L = the car's bbox length, nose near x = 0): the
inlet is 3 L ahead of the nose, the outlet 7 L behind the tail, 2 L of
clearance each side and 2 L above, each face snapped outward onto the
base-size grid.

The ground is a closed slab STL named `ground` whose top face is z = 0 and
which spans PAST the domain in x and y: the mesher's stage 0 refuses a
surface that ENDS inside the domain without one base_size of margin, so the
car cannot touch a domain face and the road must be part of the input.
Cells inside the slab are solid; the one-cell fluid layer trapped under it
is a sealed pocket that `keep_region: "seed"` drops; the domain's zMin
therefore ends with 0 faces, which is expected and left unrenamed.

The moving ground IS the `ground` patch, with NO layers on it: it moves at
the free-stream speed, so it grows no boundary layer upstream of the car
(decided).

Wall-function sizing of the first layer: Prandtl's 1/5-power turbulent
flat-plate skin friction at x = L, cf = 0.0592 Re_L**(-0.2) - the lowest
u_tau along the car - u_tau = U*sqrt(cf/2), y_p = yplus*nu/u_tau (first
cell centre), t1 = 2*y_p (first layer thickness).

The wake (to 1.5 L behind the car), a field envelope around the whole car
and the two wings are refinement boxes (`box_table`, SPEC-LIT §92.16, the
mesher's `refinement.boxes`); the distance bands are near-wall only
(`band_table`).

Snap's feature attraction asks for half a FINEST cell
(`snap.feature_tolerance = 0.5 * 2**-max_level`, SPEC-LIT §92.12's erratum:
the default 0.5 is half a BASE cell, 2^(max_level-1) finest cells wide).

CC BY 4.0 rule: the source is the "F1 2026 concept" render model by
Qvist_Designs (CC BY 4.0, via Sketchfab). The geometry and every file made
from it are NEVER written inside the repository - all outputs go to --out -
and LICENSE.txt is copied beside them.
"""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

REQUIRED_PATCHES = ["inlet", "outlet", "side_ymin", "side_ymax", "top", "ground",
                    "body", "front_wing", "rear_wing", "wheels", "floor"]
LAYER_PATCHES = ["body", "front_wing", "rear_wing", "wheels", "floor"]
# SPEC-LIT (92.74): the fraction of a patch's wall area that carries the
# WHOLE stack thickness - stack_area_frac, the KEEP faces plus the faces of M
# - which the face-mode ladder's cuts and merges leave standing. The film's
# two hero patches must HOLD 80 % of their wall; the wings and wheels ride
# whatever the ladder leaves them.
LAYER_COVERAGE = {"body": 0.80, "floor": 0.80}
BAND_PATCHES = ["body", "wheels", "floor", "front_wing", "rear_wing"]
BANDS_FULL = {
    "body": [(0.15, 5), (0.45, 4), (0.8, 3)],
    "wheels": [(0.15, 5), (0.45, 4), (0.8, 3)],
    "floor": [(0.06, 6), (0.45, 4), (0.8, 3)],
    "front_wing": [(0.04, 6), (0.15, 5)],
    "rear_wing": [(0.04, 6), (0.15, 5)],
}
BOX_ORDER = ["field", "wake", "front_wing", "rear_wing"]
_NUMPAT = re.compile("[-+]?[0-9]*[.]?[0-9]+(?:[eE][-+]?[0-9]+)?")
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def refuse(code, text):
    print("refused: " + code + ": " + text)
    sys.exit(2)


def _posix(path):
    return path.replace(os.sep, "/")


def _read_stl_raw(path):
    """(normals (n,3), triangles (n,3,3)) of one binary STL, float64."""
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        refuse("TM-STL", path + ": cannot stat (" + str(exc) + ")")
    if size < 84 or (size - 84) % 50 != 0:
        refuse("TM-STL", path + ": size " + str(size) + " is not 84 + 50 n")
    with open(path, "rb") as fh:
        blob = fh.read()
    n = (size - 84) // 50
    rows = np.frombuffer(blob[84:84 + 50 * n], dtype=np.uint8).reshape(n, 50)
    floats = rows[:, :48].copy().view(np.float32).reshape(n, 12)
    return floats[:, 0:3].astype(np.float64), floats[:, 3:12].astype(np.float64).reshape(n, 3, 3)


def read_stl_bin(path):
    """Binary STL only -> (n, 3, 3) float64 vertex array."""
    return _read_stl_raw(path)[1]


def patch_bbox(geom_dir, name):
    """(lo, hi) over one patch STL; a missing one refuses TM-GEOM by name."""
    path = os.path.join(geom_dir, name + ".stl")
    if not os.path.isfile(path):
        refuse("TM-GEOM", path + ": missing patch STL " + name)
    corners = read_stl_bin(path).reshape(-1, 3)
    return corners.min(axis=0), corners.max(axis=0)


def car_bbox(geom_dir):
    """(lo, hi) over the five car patch STLs; a missing one refuses TM-GEOM by name."""
    lo = None
    hi = None
    for name in ("body", "front_wing", "rear_wing", "wheels", "floor"):
        mn, mx = patch_bbox(geom_dir, name)
        lo = mn if lo is None else np.minimum(lo, mn)
        hi = mx if hi is None else np.maximum(hi, mx)
    return lo, hi


def tunnel_extent(lo, hi, base):
    """[xlo, xhi, ylo, yhi, zlo, zhi]: 3 L upstream, 7 L downstream, 2 L each side and above."""
    L = hi[0] - lo[0]
    return [math.floor((lo[0] - 3.0 * L) / base) * base,
            math.ceil((hi[0] + 7.0 * L) / base) * base,
            math.floor((lo[1] - 2.0 * L) / base) * base,
            math.ceil((hi[1] + 2.0 * L) / base) * base,
            -2.0 * base,
            math.ceil((hi[2] + 2.0 * L) / base) * base]


def base_grid(extent, base):
    """Background cell count per axis: round(span / base)."""
    return [round((extent[1] - extent[0]) / base),
            round((extent[3] - extent[2]) / base),
            round((extent[5] - extent[4]) / base)]


def slab_extent(extent, base):
    """The ground slab: one base cell past the domain in x and y, down to -base, top at 0."""
    return [extent[0] - base, extent[1] + base, extent[2] - base,
            extent[3] + base, -base, 0.0]


def write_ground_stl(path, slab6):
    """A closed axis-aligned box of 12 triangles as a binary STL.

    Outward normals (the stored normal is the unit cross product of the
    triangle's edges). The 80-byte header never starts with "solid" - the
    automesher's binary-STL reader takes such a file for ASCII and panics.
    """
    x0, x1, y0, y1, z0, z1 = slab6
    c000, c100, c110, c010 = (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)
    c001, c101, c111, c011 = (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)
    faces = [
        (c000, c010, c110), (c000, c110, c100),  # z = z0, outward -z
        (c001, c101, c111), (c001, c111, c011),  # z = z1, outward +z
        (c000, c100, c101), (c000, c101, c001),  # y = y0, outward -y
        (c010, c011, c111), (c010, c111, c110),  # y = y1, outward +y
        (c000, c001, c011), (c000, c011, c010),  # x = x0, outward -x
        (c100, c110, c111), (c100, c111, c101),  # x = x1, outward +x
    ]
    header = b"tunnel_mesh.py ground slab" + b" " * (80 - len(b"tunnel_mesh.py ground slab"))
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(struct.pack("<I", len(faces)))
        for a, b, d in faces:
            va, vb, vd = (np.array(v, dtype=np.float64) for v in (a, b, d))
            nrm = np.cross(vb - va, vd - va)
            nrm = nrm / float(np.linalg.norm(nrm))
            fh.write(struct.pack("<3f", *nrm.astype(np.float32)))
            for v in (va, vb, vd):
                fh.write(struct.pack("<3f", *v.astype(np.float32)))
            fh.write(struct.pack("<H", 0))


def first_layer(L, speed_kmh, nu, yplus):
    """Wall-function sizing at x = L: Prandtl cf = 0.0592 Re**(-0.2)."""
    U = speed_kmh / 3.6
    Re_L = U * L / nu
    cf = 0.0592 * Re_L ** (-0.2)
    u_tau = U * math.sqrt(cf / 2.0)
    y_p = yplus * nu / u_tau
    t1 = 2.0 * y_p
    return {"U": U, "Re_L": Re_L, "cf": cf, "u_tau": u_tau, "y_p": y_p, "t1": t1}


def band_table(scale):
    """The per-patch distance bands with every distance scaled (levels unchanged)."""
    return [{"patch": name,
             "bands": [{"distance": d * scale, "level": lvl} for d, lvl in BANDS_FULL[name]]}
            for name in BAND_PATCHES]


def box_table(lo, hi, wings):
    """The four refinement boxes (SPEC-LIT §92.16) in BOX_ORDER, grown from
    the car bbox `lo`/`hi` and the two wings' own bboxes: a level-1 field
    envelope, a level-3 wake reaching 1.5 L behind the tail, and the two
    level-4 wings. Levels are ABSOLUTE - the mesher caps them at
    --max-level, and --band-scale does NOT scale boxes."""
    L = hi[0] - lo[0]
    f_lo, f_hi = wings["front_wing"]
    r_lo, r_hi = wings["rear_wing"]
    return [
        {"min": [lo[0] - 0.5 * L, lo[1] - 0.5 * L, lo[2]],
         "max": [hi[0] + 3.0 * L, hi[1] + 0.5 * L, hi[2] + 0.5 * L], "level": 1},
        {"min": [lo[0] - 0.1 * L, lo[1] - 0.1 * L, lo[2]],
         "max": [hi[0] + 1.5 * L, hi[1] + 0.1 * L, hi[2] + 0.2 * L], "level": 3},
        {"min": [f_lo[0] - 0.04 * L, f_lo[1] - 0.04 * L, max(f_lo[2] - 0.04 * L, lo[2])],
         "max": [f_hi[0] + 0.04 * L, f_hi[1] + 0.04 * L, f_hi[2] + 0.04 * L], "level": 4},
        {"min": [r_lo[0] - 0.04 * L, r_lo[1] - 0.04 * L, max(r_lo[2] - 0.04 * L, lo[2])],
         "max": [r_hi[0] + 0.1 * L, r_hi[1] + 0.04 * L, r_hi[2] + 0.04 * L], "level": 4},
    ]


def build_config(geom_dir, out_dir, lo, hi, args, wings):
    """The AutomeshConfig. `snap.feature_tolerance` asks for half a FINEST
    cell (SPEC-LIT §92.12's erratum: the default 0.5 is half a base cell,
    2^(max_level-1) finest cells wide at the wall's level - the far
    attraction drags whole bands of points onto one feature line); the rest
    of snap uses the mesher's defaults."""
    L = hi[0] - lo[0]
    extent = tunnel_extent(lo, hi, args.base)
    surfaces = [{"path": _posix(os.path.join(geom_dir, name + ".stl")), "name": name}
                for name in ("body", "front_wing", "rear_wing", "wheels", "floor")]
    surfaces.append({"path": _posix(os.path.join(out_dir, "ground.stl")), "name": "ground"})
    return {
        "input": {"surfaces": surfaces},
        "domain": {"extent": extent, "base_size": args.base},
        "refinement": {"levels": band_table(args.band_scale), "max_level": args.max_level,
                       "boxes": box_table(lo, hi, wings)},
        "castellation": {"keep_region": "seed",
                         "seed_point": [lo[0] - 1.5 * L, 0.0, 0.5 * extent[5]]},
        "snap": {"feature_tolerance": 0.5 * 2.0 ** -args.max_level},
        "layers": {"patches": list(LAYER_PATCHES), "n": args.layers,
                   "first_thickness": first_layer(L, args.speed_kmh, args.nu, args.yplus)["t1"],
                   "growth": args.growth, "terminate": "face"},
        "quality": {"max_closure": 1e-10, "max_non_orth_deg": 70.0,
                    "report_non_orth_deg": 60.0, "min_thickness_ratio": 0.05,
                    "max_cond": 10000.0},
        "output": {"case_dir": _posix(os.path.join(out_dir, "case")), "name": "f1_tunnel",
                   "patch_names": {"xMin": "inlet", "xMax": "outlet",
                                   "yMin": "side_ymin", "yMax": "side_ymax",
                                   "zMax": "top"}},
    }


def run_stream(cmd, cwd, log_path, timeout=0.0):
    """Run cmd with stdout+stderr merged; echo every line to the console AND
    write it to log_path as it arrives (lines starting "[stl]" skip the
    console only). With timeout > 0 a watchdog kills the child once timeout
    seconds have passed since it started, even while it prints nothing, and
    the returncode becomes 124; timeout 0 = no limit. Returns
    (returncode, wall seconds, text)."""
    t0 = time.monotonic()
    text = []
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, encoding="utf-8",
                                errors="replace", bufsize=1)
        killed = False
        watchdog = None
        if timeout > 0:
            def _bite():
                nonlocal killed
                killed = True
                proc.kill()
            watchdog = threading.Timer(timeout, _bite)
            watchdog.daemon = True
            watchdog.start()
        try:
            for line in proc.stdout:
                text.append(line)
                log.write(line)
                log.flush()
                if not line.startswith("[stl]"):
                    sys.stdout.write(line)
                    sys.stdout.flush()
            rc = proc.wait(timeout=(timeout if timeout > 0 else None))
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            rc = 124
        finally:
            if watchdog is not None:
                watchdog.cancel()
        if killed:
            rc = 124
    return rc, time.monotonic() - t0, "".join(text)


def parse_check(text):
    """Parse a -check report of the F-CHECK shape; a missing line gives
    passed False and the missing numbers None."""
    out = {"n_cells": None, "n_internal_faces": None, "n_boundary_faces": None,
           "n_points": None, "min_volume": None, "n_regions": None,
           "max_closure": None, "max_non_orth_deg": None, "mean_non_orth_deg": None,
           "n_over_report": None, "min_tau": None, "max_cond": None,
           "n_duplicate_faces": None, "ldu_ordered": False, "passed": False,
           "quality_line": ""}
    quality = []
    for raw in text.splitlines():
        s = raw.strip()
        nums = _NUMPAT.findall(s)
        if s.startswith("automesher:") and " cells," in s and len(nums) >= 4:
            out["n_cells"], out["n_internal_faces"], out["n_boundary_faces"], out["n_points"] = (
                int(v) for v in nums[:4])
            quality.append(s)
        elif s.startswith("volume:") and len(nums) >= 3:
            out["min_volume"], out["n_regions"] = float(nums[0]), int(nums[2])
            quality.append(s)
        elif s.startswith("closure:") and nums:
            out["max_closure"] = float(nums[0])
            quality.append(s)
        elif s.startswith("non-orthogonality:") and len(nums) >= 3:
            out["max_non_orth_deg"] = float(nums[0])
            out["mean_non_orth_deg"] = float(nums[1])
            out["n_over_report"] = int(nums[2])
            quality.append(s)
        elif s.startswith("thickness:") and len(nums) >= 3:
            out["min_tau"], out["max_cond"] = float(nums[0]), float(nums[2])
            quality.append(s)
        elif s.startswith("duplicate faces:") and nums:
            out["n_duplicate_faces"] = int(nums[0])
            out["ldu_ordered"] = "ldu ordered: yes" in s
            out["passed"] = s.endswith("gate: passed")
            quality.append(s)
    out["quality_line"] = chr(10).join(quality)
    return out


def judge(run_rc, summary, check_rc, chk, reduced):
    """The gate: run, check, non_orth (STRICT < 70), patches, cells (or
    REDUCED), and layers - LAYER_COVERAGE's patches must keep their STACK
    share of the wall (SPEC-LIT (92.74), stack_area_frac)."""
    reasons = []
    run = run_rc == 0 and summary is not None
    check = check_rc == 0 and bool(chk.get("passed"))
    mo = chk.get("max_non_orth_deg")
    non_orth = mo is not None and mo < 70.0
    missing = []
    n = None
    cells = False
    stack_rows = {}
    if summary is not None:
        stage = next((r for r in summary.get("stages", [])
                      if r.get("stage") == "layers"), None)
        stack_rows = {r.get("name"): r.get("stack_area_frac")
                      for r in (stage or {}).get("patches", [])}
    if summary is None:
        patches = False
    else:
        rows = {r.get("name"): r.get("size", 0)
                for r in summary.get("mesh", {}).get("patches", [])}
        missing = [p for p in REQUIRED_PATCHES if rows.get(p, 0) <= 0]
        patches = not missing
        n = summary.get("mesh", {}).get("n_cells")
        cells = "REDUCED" if reduced else (isinstance(n, int) and 3_000_000 <= n <= 6_000_000)
    layers_ok = True
    for p, bar in LAYER_COVERAGE.items():
        stack = stack_rows.get(p)
        if isinstance(stack, (int, float)) and stack >= bar:
            continue
        layers_ok = False
        shown = f"{stack:.3f}" if isinstance(stack, (int, float)) else repr(stack)
        reasons.append(f"layers: {p} stack {shown} < {bar:.2f}")
    ok = (run and check and non_orth and patches
          and (cells is True or cells == "REDUCED") and layers_ok)
    if not run:
        reasons.append("run: returncode " + str(run_rc) +
                       ("" if summary is not None else ", no summary"))
    if not check:
        reasons.append("check: returncode " + str(check_rc) + ", passed " + repr(chk.get("passed")))
    if not non_orth:
        reasons.append("non_orth: " + repr(mo))
    if not patches:
        reasons.append("patches: " + (", ".join(missing) if summary is not None else "no summary"))
    if not (cells is True or cells == "REDUCED"):
        reasons.append("cells: " + (repr(n) if summary is not None else "no summary"))
    return {"run": run, "check": check, "non_orth": non_orth, "patches": patches,
            "cells": cells, "layers": layers_ok, "pass": ok, "reasons": reasons}


def build_report(args, lo, hi, L, floor_gap_m, attribution, extent, slab, wf,
                 reduced, cfg_path, run_rc, wall_s, summary, check_rc, chk, gate,
                 boxes):
    """The tunnel_mesh.json report (R11)."""
    digest = hashlib.sha256()
    with open(args.binary, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    mesher = {"binary": _posix(os.path.abspath(args.binary)),
              "binary_sha256": digest.hexdigest()[:8],
              "wall_seconds": wall_s, "returncode": run_rc,
              "total_seconds": None, "stages": [], "n_cells": None,
              "patches": {}, "quality": None}
    layers = []
    if summary is not None:
        mesher["total_seconds"] = summary.get("total_seconds")
        mesher["stages"] = [{"stage": r.get("stage"), "seconds": r.get("seconds")}
                            for r in summary.get("stages", [])]
        mesher["n_cells"] = summary.get("mesh", {}).get("n_cells")
        mesher["patches"] = {r.get("name"): r.get("size")
                             for r in summary.get("mesh", {}).get("patches", [])}
        mesher["quality"] = summary.get("quality")
        stage = next((r for r in summary.get("stages", []) if r.get("stage") == "layers"), None)
        if stage is not None:
            keep = ("name", "n_layers", "t1_requested", "t1_min", "t1_mean",
                    "full_area_frac", "drop_cause", "dropped",
                    "kept_area_frac", "ring_area_frac", "n_keep_faces",
                    "n_ring_faces", "n_off_faces", "n_anchored_points",
                    "stack_area_frac", "n_merged_faces", "merged_area_frac",
                    "n_cut_faces")
            layers = [{k: r.get(k) for k in keep} for r in stage.get("patches", [])]
    return {
        "tool": "tools/promo/tunnel_mesh.py",
        "geom_dir": _posix(os.path.abspath(args.geom)),
        "attribution": attribution,
        "reduced": reduced,
        "params": {"base": args.base, "max_level": args.max_level,
                   "band_scale": args.band_scale, "layers": args.layers,
                   "growth": args.growth, "speed_kmh": args.speed_kmh,
                   "nu": args.nu, "yplus": args.yplus},
        "car": {"lo": lo, "hi": hi, "length_m": L, "floor_gap_m": floor_gap_m},
        "boxes": [{"name": n, **box} for n, box in zip(BOX_ORDER, boxes)],
        "domain": {"extent": extent, "base_grid": base_grid(extent, args.base), "slab": slab},
        "wall_function": wf,
        "floor_gap_cells_design": floor_gap_m / (args.base / 2 ** min(args.max_level, 6)),
        "config_path": _posix(cfg_path),
        "mesher": mesher,
        "check": {"returncode": check_rc, "quality_line": chk.get("quality_line"),
                  "parsed": chk},
        "layers": layers,
        "gate": gate,
    }


F_CHECK = """ofgpu-automesher: checking C:/x/p5_case (SPEC-LIT §92.14.5)
automesher: 22599 cells, 65515 internal faces, 7090 boundary faces, 32884 points
  volume: min 0.000022 (cell 1296); regions: 1 (22599 cells)
  closure: max 4.730e-15 (cell 1090)
  non-orthogonality: max 69.951 deg, mean 3.308 deg, 126 face(s) past the report mark
  thickness: min tau 0.052409 (cell 3034); conditioning: max cond 267.106 (cell 1296)
  duplicate faces: 0; ldu ordered: yes; gate: passed
"""

_LO = [0.00023655073891859502, -0.9408786296844482, 0.0]
_HI = [5.393801689147949, 0.9408786296844482, 1.0961036682128906]


def _close(a, b, rel=1e-9):
    return abs(a - b) <= rel * max(abs(a), abs(b))


def _t1():
    e05 = tunnel_extent(_LO, _HI, 0.5)
    if e05 != [-16.5, 43.5, -12.0, 12.0, -1.0, 12.0]:
        raise AssertionError("extent base 0.5 = " + repr(e05))
    if base_grid(e05, 0.5) != [120, 48, 26]:
        raise AssertionError("grid base 0.5 = " + repr(base_grid(e05, 0.5)))
    e10 = tunnel_extent(_LO, _HI, 1.0)
    if e10 != [-17.0, 44.0, -12.0, 12.0, -2.0, 12.0]:
        raise AssertionError("extent base 1.0 = " + repr(e10))
    if base_grid(e10, 1.0) != [61, 24, 14]:
        raise AssertionError("grid base 1.0 = " + repr(base_grid(e10, 1.0)))
    sl = slab_extent(e05, 0.5)
    if sl != [-17.0, 44.0, -12.5, 12.5, -0.5, 0.0]:
        raise AssertionError("slab = " + repr(sl))


def _t2():
    got = first_layer(5.393565138409031, 250, 1.5e-5, 50)
    want = {"U": 69.44444444444444, "Re_L": 24970208.974115882,
            "cf": 0.001962624750218175, "u_tau": 2.1754101260182086,
            "y_p": 0.00034476257650449234, "t1": 0.0006895251530089847}
    if set(got) != set(want):
        raise AssertionError("keys " + repr(sorted(got)))
    for k, v in want.items():
        if not _close(got[k], v):
            raise AssertionError(k + " = " + repr(got[k]) + " want " + repr(v))


def _t3():
    slab = slab_extent(tunnel_extent(_LO, _HI, 0.5), 0.5)
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "ground.stl")
        write_ground_stl(path, slab)
        with open(path, "rb") as fh:
            blob = fh.read()
        if len(blob) != 684:
            raise AssertionError("size " + str(len(blob)) + " want 684")
        if blob[:5] == b"solid":
            raise AssertionError("header starts with solid")
        nrm, tris = _read_stl_raw(path)
        if tris.shape != (12, 3, 3):
            raise AssertionError("shape " + repr(tris.shape))
        counts = {}
        for tri in tris:
            vs = [tuple(v) for v in tri]
            if len(set(vs)) != 3:
                raise AssertionError("degenerate triangle")
            for i in range(3):
                edge = tuple(sorted((vs[i], vs[(i + 1) % 3])))
                counts[edge] = counts.get(edge, 0) + 1
        bad = [e for e, n in counts.items() if n != 2]
        if bad:
            raise AssertionError(str(len(bad)) + " edges not shared by exactly 2 triangles")
        centre = np.array([(slab[0] + slab[1]) / 2.0, (slab[2] + slab[3]) / 2.0,
                           (slab[4] + slab[5]) / 2.0])
        for n, tri in zip(nrm, tris):
            if float(np.dot(n, tri.mean(axis=0) - centre)) <= 0.0:
                raise AssertionError("stored normal not outward")
        for axis in range(3):
            if (tris[:, :, axis].min() != slab[2 * axis]
                    or tris[:, :, axis].max() != slab[2 * axis + 1]):
                raise AssertionError("bbox axis " + str(axis) + " != slab extent")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _t4():
    want = {"body": [(0.15, 5), (0.45, 4), (0.8, 3)],
            "wheels": [(0.15, 5), (0.45, 4), (0.8, 3)],
            "floor": [(0.06, 6), (0.45, 4), (0.8, 3)],
            "front_wing": [(0.04, 6), (0.15, 5)],
            "rear_wing": [(0.04, 6), (0.15, 5)]}
    full = band_table(1.0)
    if [e["patch"] for e in full] != list(want):
        raise AssertionError("patch order " + repr([e["patch"] for e in full]))
    for e in full:
        got = [(b["distance"], b["level"]) for b in e["bands"]]
        if got != want[e["patch"]]:
            raise AssertionError(e["patch"] + " bands " + repr(got))
    body = next(e for e in band_table(0.3) if e["patch"] == "body")
    for b, (d, lvl) in zip(body["bands"], [(0.045, 5), (0.135, 4), (0.24, 3)]):
        if abs(b["distance"] - d) > 1e-12 or b["level"] != lvl:
            raise AssertionError("scaled band " + repr(b) + " want " + repr((d, lvl)))


def _t5():
    args = argparse.Namespace(base=0.5, max_level=6, band_scale=1.0, layers=3,
                              growth=1.2, speed_kmh=250.0, nu=1.5e-5, yplus=50.0)
    wings = {"front_wing": ([0.0002, -0.9, 0.0879], [0.9151, 0.9, 0.5538]),
             "rear_wing": ([4.7974, -0.575, 0.3956], [5.3938, 0.575, 0.9072])}
    cfg = build_config("C:/g", "C:/o", _LO, _HI, args, wings)
    if set(cfg) != {"input", "domain", "refinement", "castellation", "snap",
                    "layers", "quality", "output"}:
        raise AssertionError("top-level keys " + repr(sorted(cfg)))
    if cfg["snap"] != {"feature_tolerance": 0.0078125}:
        raise AssertionError("snap " + repr(cfg["snap"]))
    sur = cfg["input"]["surfaces"]
    if [s["name"] for s in sur] != ["body", "front_wing", "rear_wing", "wheels",
                                    "floor", "ground"]:
        raise AssertionError("surface names " + repr([s["name"] for s in sur]))
    if sur[0]["path"] != "C:/g/body.stl" or sur[-1]["path"] != "C:/o/ground.stl":
        raise AssertionError("surface paths " + repr([s["path"] for s in sur]))
    if cfg["domain"]["extent"] != [-16.5, 43.5, -12.0, 12.0, -1.0, 12.0]:
        raise AssertionError("extent " + repr(cfg["domain"]["extent"]))
    want_seed = [-8.090111156874627, 0.0, 6.0]
    for a, b in zip(cfg["castellation"]["seed_point"], want_seed):
        if abs(a - b) > 1e-12:
            raise AssertionError("seed_point " + repr(cfg["castellation"]["seed_point"]))
    t1 = first_layer(5.393565138409031, 250, 1.5e-5, 50)["t1"]
    if not _close(cfg["layers"]["first_thickness"], t1):
        raise AssertionError("first_thickness " + repr(cfg["layers"]["first_thickness"]))
    if cfg["layers"].get("terminate") != "face":
        raise AssertionError("layers.terminate " + repr(cfg["layers"].get("terminate")))
    if cfg["quality"] != {"max_closure": 1e-10, "max_non_orth_deg": 70.0,
                          "report_non_orth_deg": 60.0, "min_thickness_ratio": 0.05,
                          "max_cond": 10000.0}:
        raise AssertionError("quality " + repr(cfg["quality"]))
    if set(cfg["refinement"]) != {"levels", "max_level", "boxes"}:
        raise AssertionError("refinement keys " + repr(sorted(cfg["refinement"])))
    if [e["patch"] for e in cfg["refinement"]["levels"]] != ["body", "wheels", "floor",
                                                             "front_wing", "rear_wing"]:
        raise AssertionError("refinement order " +
                             repr([e["patch"] for e in cfg["refinement"]["levels"]]))
    if [b["level"] for b in cfg["refinement"]["boxes"]] != [1, 3, 4, 4]:
        raise AssertionError("box levels " +
                             repr([b["level"] for b in cfg["refinement"]["boxes"]]))


def _t6():
    chk = parse_check(F_CHECK)
    want = {"n_cells": 22599, "n_internal_faces": 65515, "n_boundary_faces": 7090,
            "n_points": 32884, "min_volume": 0.000022, "n_regions": 1,
            "max_closure": 4.73e-15, "max_non_orth_deg": 69.951,
            "mean_non_orth_deg": 3.308, "n_over_report": 126, "min_tau": 0.052409,
            "max_cond": 267.106, "n_duplicate_faces": 0}
    for k, v in want.items():
        if chk[k] != v:
            raise AssertionError(k + " = " + repr(chk[k]) + " want " + repr(v))
    if chk["ldu_ordered"] is not True or chk["passed"] is not True:
        raise AssertionError("ldu/passed " + repr((chk["ldu_ordered"], chk["passed"])))
    qlines = chk["quality_line"].split(chr(10))
    if len(qlines) != 6 or not qlines[0].startswith("automesher: 22599 cells") \
            or not qlines[-1].endswith("gate: passed"):
        raise AssertionError("quality_line " + repr(chk["quality_line"]))
    failed = parse_check(F_CHECK.replace("gate: passed", "gate: FAILED"))
    if failed["passed"] is not False:
        raise AssertionError("gate FAILED still parses as passed")


def _t7():
    chk = parse_check(F_CHECK)

    def summ(n_cells, floor_size=10, kept_body=0.9, kept_floor=0.9):
        rows = [{"name": p, "size": floor_size} for p in REQUIRED_PATCHES]
        lays = [{"name": p, "stack_area_frac": k} for p, k in
                (("body", kept_body), ("floor", kept_floor))]
        return {"mesh": {"n_cells": n_cells, "patches": rows},
                "stages": [{"stage": "layers", "patches": lays}]}

    g = judge(0, summ(3500000), 0, chk, False)
    if g["pass"] is not True or g["cells"] is not True or g["layers"] is not True:
        raise AssertionError("(a) " + repr(g))
    g = judge(0, summ(2900000), 0, chk, False)
    if g["pass"] is not False or not any(r.startswith("cells") for r in g["reasons"]):
        raise AssertionError("(b) " + repr(g))
    g = judge(0, summ(22557), 0, chk, True)
    if g["pass"] is not True or g["cells"] != "REDUCED":
        raise AssertionError("(c) " + repr(g))
    bad = dict(chk)
    bad["max_non_orth_deg"] = 70.0
    g = judge(0, summ(3500000), 0, bad, False)
    if g["pass"] is not False or not any(r.startswith("non_orth") for r in g["reasons"]):
        raise AssertionError("(d) " + repr(g))
    g = judge(0, summ(3500000, floor_size=0), 0, chk, False)
    if g["pass"] is not False or not any(r.startswith("patches") for r in g["reasons"]):
        raise AssertionError("(e) " + repr(g))
    g = judge(0, summ(3500000, kept_body=0.412), 0, chk, False)
    if g["pass"] is not False or g["layers"] is not False \
            or "layers: body stack 0.412 < 0.80" not in g["reasons"]:
        raise AssertionError("(f) " + repr(g))
    g = judge(0, summ(3500000, kept_floor=0.4116), 0, chk, False)
    if g["pass"] is not False or "layers: floor stack 0.412 < 0.80" not in g["reasons"]:
        raise AssertionError("(g) " + repr(g))


def _t8():
    tmp = tempfile.mkdtemp()
    try:
        log = os.path.join(tmp, "t8.log")
        rc, wall, text = run_stream(
            [sys.executable, "-c", "import time; print('a', flush=True); time.sleep(20)"],
            ".", log, timeout=2)
        if rc != 124:
            raise AssertionError("timeout rc " + str(rc) + " want 124")
        if not wall < 10.0:
            raise AssertionError("timeout wall " + repr(wall) + " s, want < 10")
        if "a" not in text:
            raise AssertionError("timeout text missing 'a': " + repr(text))
        with open(log, "r", encoding="utf-8") as fh:
            if "a" not in fh.read():
                raise AssertionError("log missing 'a' after the timeout kill")
        rc, _, text = run_stream(
            [sys.executable, "-c", "import time; print('a', flush=True); time.sleep(1)"],
            ".", log, timeout=10)
        if rc != 0:
            raise AssertionError("within-timeout rc " + str(rc) + " want 0")
        if "a" not in text:
            raise AssertionError("within-timeout text missing 'a': " + repr(text))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _t9():
    wings = {"front_wing": ([0.0002, -0.9, 0.0879], [0.9151, 0.9, 0.5538]),
             "rear_wing": ([4.7974, -0.575, 0.3956], [5.3938, 0.575, 0.9072])}
    boxes = box_table(_LO, _HI, wings)
    if [b["level"] for b in boxes] != [1, 3, 4, 4]:
        raise AssertionError("box levels " + repr([b["level"] for b in boxes]))
    want = [
        ([-2.6965460184655967, -3.6376611988889636, 0.0],
         [21.57449710437504, 3.6376611988889636, 3.792886237417406], 1),
        ([-0.5391199631019845, -1.4802351435253513, 0.0],
         [13.484149396761495, 1.4802351435253513, 2.174816695894697], 3),
        ([-0.21554260553636123, -1.1157426055363613, 0.0],
         [1.1308426055363612, 1.1157426055363613, 0.7695426055363612], 4),
        ([4.5816573944636385, -0.7907426055363612, 0.17985739446363877],
         [5.933156513840903, 0.7907426055363612, 1.1229426055363612], 4),
    ]
    for got, (mn, mx, lvl) in zip(boxes, want):
        if got["level"] != lvl:
            raise AssertionError("level " + repr(got["level"]) + " want " + repr(lvl))
        for a in range(3):
            if abs(got["min"][a] - mn[a]) > 1e-12:
                raise AssertionError("min[" + str(a) + "] " + repr(got["min"][a])
                                     + " want " + repr(mn[a]))
            if abs(got["max"][a] - mx[a]) > 1e-12:
                raise AssertionError("max[" + str(a) + "] " + repr(got["max"][a])
                                     + " want " + repr(mx[a]))
    L = _HI[0] - _LO[0]
    if abs(boxes[1]["max"][0] - _HI[0] - 1.5 * L) > 1e-12:
        raise AssertionError("wake reach " + repr(boxes[1]["max"][0] - _HI[0])
                             + " want " + repr(1.5 * L))


def selftest():
    """T1-T9, no geometry and no mesher; temp files only under tempfile.mkdtemp()."""
    tests = [("T1", _t1), ("T2", _t2), ("T3", _t3), ("T4", _t4),
             ("T5", _t5), ("T6", _t6), ("T7", _t7), ("T8", _t8), ("T9", _t9)]
    npass = 0
    for name, fn in tests:
        try:
            fn()
            print("[ok] " + name)
            npass += 1
        except Exception as exc:
            print("[FAIL] " + name + ": " + str(exc))
    if npass == len(tests):
        print("SELFTEST PASS " + str(npass) + "/" + str(len(tests)))
        return 0
    print("SELFTEST FAIL " + str(npass) + "/" + str(len(tests)))
    return 1


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="tunnel_mesh.py",
        description="Write the automesher config for the F1 sim surface in a "
                    "moving-ground wind tunnel, run the release automesher and "
                    "its -check, and report cells, time, the quality line and "
                    "its own gate.")
    p.add_argument("--geom", help="directory of the five patch STLs, sim_geom.json and LICENSE.txt")
    p.add_argument("--out", help="output directory, created if absent (never inside the repository)")
    p.add_argument("--base", type=float, default=0.5, help="background cell size, m (default 0.5)")
    p.add_argument("--max-level", type=int, default=6, help="refinement level cap (default 6)")
    p.add_argument("--band-scale", type=float, default=1.0,
                   help="scale on every band distance (default 1.0)")
    p.add_argument("--layers", type=int, default=3,
                   help="boundary layers on the car patches (default 3)")
    p.add_argument("--growth", type=float, default=1.2, help="layer growth ratio (default 1.2)")
    p.add_argument("--speed-kmh", type=float, default=250.0,
                   help="free-stream speed, km/h (default 250)")
    p.add_argument("--nu", type=float, default=1.5e-5,
                   help="kinematic viscosity, m2/s (default 1.5e-5)")
    p.add_argument("--yplus", type=float, default=50.0,
                   help="target y+ of the first cell centre (default 50)")
    p.add_argument("--binary", default=os.path.join(
        _REPO, "rust", "target", "release",
        "ofgpu-automesher" + (".exe" if os.name == "nt" else "")),
        help="the release automesher binary")
    p.add_argument("--dry-run", action="store_true",
                   help="run the mesher -dryRun only, write dryrun.log, exit")
    p.add_argument("--selftest", action="store_true", help="run T1-T9 and exit")
    p.add_argument("--timeout", type=float, default=0.0,
                   help="seconds for the mesher subprocess, 0 = none (default 0)")
    args = p.parse_args(argv)

    if args.selftest:
        return selftest()
    if not args.geom or not args.out:
        p.error("--geom and --out are required unless --selftest")
    if not os.path.isfile(args.binary):
        refuse("TM-BINARY", args.binary + ": not found")

    lo_np, hi_np = car_bbox(args.geom)
    lo = [float(v) for v in lo_np]
    hi = [float(v) for v in hi_np]
    L = hi[0] - lo[0]
    floor_tris = read_stl_bin(os.path.join(args.geom, "floor.stl"))
    floor_gap_m = float(floor_tris[:, :, 2].min())
    attribution = None
    sim_json = os.path.join(args.geom, "sim_geom.json")
    if os.path.isfile(sim_json):
        with open(sim_json, "r", encoding="utf-8") as fh:
            attribution = json.load(fh).get("source", {}).get("attribution")
    os.makedirs(args.out, exist_ok=True)

    extent = tunnel_extent(lo, hi, args.base)
    slab = slab_extent(extent, args.base)
    write_ground_stl(os.path.join(args.out, "ground.stl"), slab)
    wings = {}
    for name in ("front_wing", "rear_wing"):
        w_lo, w_hi = patch_bbox(args.geom, name)
        wings[name] = ([float(v) for v in w_lo], [float(v) for v in w_hi])
    cfg = build_config(args.geom, args.out, lo, hi, args, wings)
    cfg_path = os.path.abspath(os.path.join(args.out, "f1_tunnel.json"))
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=1)

    wf = first_layer(L, args.speed_kmh, args.nu, args.yplus)
    reduced = (args.base != 0.5) or (args.max_level != 6) or (args.band_scale != 1.0)

    print("geom " + _posix(os.path.abspath(args.geom)))
    print("car L " + repr(L) + " m, floor gap " + repr(floor_gap_m) + " m")
    print("extent " + repr(extent) + ", base grid " + repr(base_grid(extent, args.base)))
    print("wall function " + json.dumps(wf))
    print("config " + _posix(cfg_path))

    if args.dry_run:
        rc, _, _ = run_stream([args.binary, _posix(cfg_path), "-dryRun"], args.out,
                              os.path.join(args.out, "dryrun.log"), timeout=args.timeout)
        print("dry-run returncode " + str(rc))
        return 0 if rc == 0 else 1

    run_rc, wall_s, _ = run_stream([args.binary, _posix(cfg_path)], args.out,
                                   os.path.join(args.out, "automesher.log"),
                                   timeout=args.timeout)
    summary = None
    sum_path = os.path.join(args.out, "case", "f1_tunnel_summary.json")
    if os.path.isfile(sum_path):
        with open(sum_path, "r", encoding="utf-8") as fh:
            summary = json.load(fh)
    check_rc, _, chk_text = run_stream([args.binary, _posix(cfg_path), "-check"],
                                       args.out,
                                       os.path.join(args.out, "check.log"),
                                       timeout=args.timeout)
    chk = parse_check(chk_text)

    if summary is not None:
        stage = next((r for r in summary.get("stages", [])
                      if r.get("stage") == "layers"), None)
        for r in (stage or {}).get("patches", []):
            kept = r.get("kept_area_frac")
            ring = r.get("ring_area_frac")
            stack = r.get("stack_area_frac")
            print("layers " + str(r.get("name"))
                  + ": n " + str(r.get("n_layers"))
                  + " kept " + (f"{kept * 100:.1f} %" if isinstance(kept, (int, float)) else repr(kept))
                  + " ring " + (f"{ring * 100:.1f} %" if isinstance(ring, (int, float)) else repr(ring))
                  + " stack " + (f"{stack * 100:.1f} %" if isinstance(stack, (int, float)) else repr(stack))
                  + " cut " + str(r.get("n_cut_faces"))
                  + " off " + str(r.get("n_off_faces")) + " face(s)"
                  + ", drop " + str(r.get("drop_cause")))

    gate = judge(run_rc, summary, check_rc, chk, reduced)
    report = build_report(args, lo, hi, L, floor_gap_m, attribution, extent, slab, wf,
                          reduced, cfg_path, run_rc, wall_s, summary, check_rc, chk, gate,
                          box_table(lo, hi, wings))
    with open(os.path.join(args.out, "tunnel_mesh.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    lic = os.path.join(args.geom, "LICENSE.txt")
    if os.path.isfile(lic):
        shutil.copyfile(lic, os.path.join(args.out, "LICENSE.txt"))

    if gate["pass"]:
        print("GATE PASS REDUCED" if reduced else "GATE PASS")
        return 0
    print("GATE FAIL " + "; ".join(gate["reasons"]))
    return 1


if __name__ == "__main__":
    sys.exit(main())
