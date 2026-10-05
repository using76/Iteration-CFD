#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.

"""The drone showreel's free-flight tunnel mesh: config, run, -check, gate.

The tunnel follows the drone-and-rotor-disk envelope: 10 D ahead of the
drone, 20 D behind it and 6 D around it, where D is the largest rotor
tip-to-tip distance (the max pairwise 3-D distance between rotor centres
plus 2 x the max rotor radius); every domain face is snapped OUTWARD
onto the base-size grid.

Forward-flight convention: the drone flies +x at level attitude, no
pitch, the geometry left untransformed in its own PX4 FLU frame so it
lines up with rotors.json and the render model; the freestream enters
at xMax and flows along -x. The hover variant turns the flow axis to
-z: 10 D above the drone, 20 D below, 6 D around in x and y, and no
freestream at all - the rotor downwash leaves through zMin.

Refinement boxes (SPEC-LIT §92.16): a level-1 field envelope, a level-3
wake, a level-4 near-wake, a level-4 drone box and one level-5 slab per
actuator disk. The forward wake leans down behind the drone because the
freestream carries the rotor downwash back and down; the hover wake is
the downwash column hanging below it. Distance bands sit on the four
airframe walls only.

--non-orth-cap defaults to 65, not the unit's 70 gate: with the config
cap at 70 the snapped mesh lands on it at 69.9996 deg, which -check
prints as 70.000 - three decimals - and the strict gate fails; at 65
the snapped mesh is the same to within a few faces and -check prints
65.000.

BSD-3-Clause credit rule: the geometry is the PX4 x500 assembly,
"(c) 2022 Rudis Laboratories / PX4 Autopilot for Drones", and the
airframe and every mesh made from it are NEVER written inside the
repository - all outputs go to --out, with rotors.json, LICENSE.txt and
LICENSE copied beside them.
"""

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import shutil
import sys
import tempfile
import types

DRONE_PATCHES = ["body", "arms", "motors", "skids"]
BANDS = [(0.006, 6), (0.02, 5), (0.05, 4)]          # (distance m, level), the same for every drone patch
BOX_ORDER = ["field", "wake", "near_wake", "drone", "disk_0", "disk_1", "disk_2", "disk_3"]
CELLS_MIN, CELLS_MAX = 1_000_000, 3_000_000
NON_ORTH_GATE_DEG = 70.0
CREDIT = "(c) 2022 Rudis Laboratories / PX4 Autopilot for Drones, BSD-3-Clause"
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_promo() -> types.ModuleType:
    """R2: reuse the promo tunnel helpers by import; NEVER import tunnel_mesh by name."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.normpath(os.path.join(here, "..", "promo", "tunnel_mesh.py"))
    spec = importlib.util.spec_from_file_location("promo_tunnel_mesh", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PROMO = load_promo()


def _pos_num(v):
    """True for a real number > 0 (bools are not numbers here)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0


def read_rotors(geom_dir: str) -> list:
    """The four actuator disks of <geom>/rotors.json, sorted by index; DT-ROTORS
    when the file is missing, unreadable, or an entry lacks index, centre_m
    (3 numbers), radius_m > 0 or thickness_m > 0."""
    path = os.path.join(geom_dir, "rotors.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as exc:
        PROMO.refuse("DT-ROTORS", path + ": unreadable (" + str(exc) + ")")
    rotors = doc.get("rotors") if isinstance(doc, dict) else None
    if not isinstance(rotors, list) or len(rotors) != 4:
        PROMO.refuse("DT-ROTORS", path + ": rotors must be a list of exactly 4 entries")
    for r in rotors:
        c = r.get("centre_m") if isinstance(r, dict) else None
        if (not isinstance(r, dict) or "index" not in r
                or not isinstance(c, list) or len(c) != 3
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in c)
                or not _pos_num(r.get("radius_m")) or not _pos_num(r.get("thickness_m"))):
            PROMO.refuse("DT-ROTORS", path + ": every entry needs index, centre_m"
                                            " (3 numbers), radius_m > 0, thickness_m > 0")
    return sorted(rotors, key=lambda r: r["index"])


def airframe_bbox(geom_dir: str) -> tuple:
    """(lo, hi) over the four patch STLs (through the reused read_stl_bin), as
    lists of floats; DT-GEOM when one of them is missing."""
    lo = [math.inf] * 3
    hi = [-math.inf] * 3
    for name in DRONE_PATCHES:
        path = os.path.join(geom_dir, name + ".stl")
        if not os.path.isfile(path):
            PROMO.refuse("DT-GEOM", path + ": missing patch STL " + name)
        corners = PROMO.read_stl_bin(path).reshape(-1, 3)
        mn = corners.min(axis=0).tolist()
        mx = corners.max(axis=0).tolist()
        lo = [min(a, b) for a, b in zip(lo, mn)]
        hi = [max(a, b) for a, b in zip(hi, mx)]
    return lo, hi


def envelope(af_lo, af_hi, rotors) -> dict:
    """The drone-and-rotor-disk envelope: the airframe bbox unioned with every
    rotor disk box (centre x,y +- radius_m; centre z +- thickness_m/2), and
    D = the largest rotor tip-to-tip distance (max pairwise 3-D distance
    between centres + 2 x the max radius_m)."""
    lo = list(af_lo)
    hi = list(af_hi)
    for r in rotors:
        c, rad, t2 = r["centre_m"], r["radius_m"], r["thickness_m"] / 2.0
        lo = [min(lo[0], c[0] - rad), min(lo[1], c[1] - rad), min(lo[2], c[2] - t2)]
        hi = [max(hi[0], c[0] + rad), max(hi[1], c[1] + rad), max(hi[2], c[2] + t2)]
    d = 0.0
    for i in range(len(rotors)):
        for j in range(i + 1, len(rotors)):
            ci, cj = rotors[i]["centre_m"], rotors[j]["centre_m"]
            d = max(d, math.sqrt(sum((a - b) ** 2 for a, b in zip(ci, cj))))
    return {"lo": lo, "hi": hi, "airframe_lo": list(af_lo),
            "airframe_hi": list(af_hi), "D": d + 2.0 * max(r["radius_m"] for r in rotors)}


def tunnel_extent(env: dict, base: float, variant: str) -> list:
    """[xlo, xhi, ylo, yhi, zlo, zhi], every face snapped OUTWARD onto the
    base-size grid: forward 10 D ahead (+x), 20 D behind (-x), 6 D around in
    y and z; hover the same with the flow axis -z (10 D above, 20 D below)."""
    lo, hi, d = env["lo"], env["hi"], env["D"]
    if variant == "forward":
        want = [lo[0] - 20.0 * d, hi[0] + 10.0 * d, lo[1] - 6.0 * d,
                hi[1] + 6.0 * d, lo[2] - 6.0 * d, hi[2] + 6.0 * d]
    else:
        want = [lo[0] - 6.0 * d, hi[0] + 6.0 * d, lo[1] - 6.0 * d,
                hi[1] + 6.0 * d, lo[2] - 20.0 * d, hi[2] + 10.0 * d]
    return [math.floor(want[0] / base) * base, math.ceil(want[1] / base) * base,
            math.floor(want[2] / base) * base, math.ceil(want[3] / base) * base,
            math.floor(want[4] / base) * base, math.ceil(want[5] / base) * base]


def patch_names(variant: str) -> dict:
    """The six domain faces renamed for the case; the inlet is where the flow
    enters (xMax forward, zMax hover)."""
    if variant == "forward":
        return {"xMax": "inlet", "xMin": "outlet", "yMin": "side_ymin",
                "yMax": "side_ymax", "zMin": "bottom", "zMax": "top"}
    return {"zMax": "inlet", "zMin": "outlet", "xMin": "side_xmin",
            "xMax": "side_xmax", "yMin": "side_ymin", "yMax": "side_ymax"}


def required_patches(variant: str) -> list:
    """The six renamed domain patches in value order, then the four airframe
    walls - the ten patches the gate requires to be present with faces."""
    return list(patch_names(variant).values()) + list(DRONE_PATCHES)


def seed_point(env: dict, variant: str) -> list:
    """The keep_region seed: forward 5 D ahead of the drone (+x, upstream),
    hover 5 D above it."""
    if variant == "forward":
        return [env["hi"][0] + 5.0 * env["D"], 0.0, 0.0]
    return [0.0, 0.0, env["hi"][2] + 5.0 * env["D"]]


def box_table(env: dict, rotors: list, variant: str) -> list:
    """The eight refinement boxes (SPEC-LIT §92.16) in BOX_ORDER, grown from
    the envelope lo/hi: a level-1 field envelope, a level-3 wake, a level-4
    near-wake, a level-4 drone box and one level-5 slab per actuator disk
    (rotors in index order). Levels are ABSOLUTE - the mesher caps them at
    max_level."""
    lo, hi, d = env["lo"], env["hi"], env["D"]
    if variant == "forward":
        field = [[lo[0] - 8.0 * d, lo[1] - 2.0 * d, lo[2] - 3.0 * d],
                 [hi[0] + 2.0 * d, hi[1] + 2.0 * d, hi[2] + 2.0 * d]]
        wake = [[lo[0] - 3.0 * d, lo[1] - 0.1 * d, lo[2] - 1.5 * d],
                [hi[0] + 0.1 * d, hi[1] + 0.1 * d, hi[2] + 0.1 * d]]
        near = [[lo[0] - 1.0 * d, lo[1] - 0.05 * d, lo[2] - 0.5 * d],
                [hi[0] + 0.0, hi[1] + 0.05 * d, hi[2] + 0.05 * d]]
    else:
        field = [[lo[0] - 2.0 * d, lo[1] - 2.0 * d, lo[2] - 8.0 * d],
                 [hi[0] + 2.0 * d, hi[1] + 2.0 * d, hi[2] + 2.0 * d]]
        wake = [[lo[0] - 0.1 * d, lo[1] - 0.1 * d, lo[2] - 3.0 * d],
                [hi[0] + 0.1 * d, hi[1] + 0.1 * d, hi[2] + 0.1 * d]]
        near = [[lo[0] - 0.05 * d, lo[1] - 0.05 * d, lo[2] - 1.0 * d],
                [hi[0] + 0.05 * d, hi[1] + 0.05 * d, hi[2] + 0.0]]
    out = [
        {"name": "field", "min": field[0], "max": field[1], "level": 1},
        {"name": "wake", "min": wake[0], "max": wake[1], "level": 3},
        {"name": "near_wake", "min": near[0], "max": near[1], "level": 4},
        {"name": "drone", "min": [v - 0.1 * d for v in lo],
         "max": [v + 0.1 * d for v in hi], "level": 4},
    ]
    for k, r in enumerate(rotors):
        c, s = r["centre_m"], 1.05 * r["radius_m"]
        out.append({"name": "disk_" + str(k), "level": 5,
                    "min": [c[0] - s, c[1] - s, c[2] - 0.025],
                    "max": [c[0] + s, c[1] + s, c[2] + 0.025]})
    return out


def build_config(geom_dir, out_dir, env, rotors, args) -> dict:
    """The AutomeshConfig: four surface patches, the tunnel extent at base_size,
    the per-patch distance bands and refinement boxes, seed-point keep_region,
    half-a-finest-cell feature attraction (SPEC-LIT §92.12's erratum), the
    wall-function first layer (L = the airframe x length, the forward-flight
    sizing for hover too) and the 65 deg non-orthogonality cap. The mesher
    refuses unknown keys, so every refinement.boxes entry carries ONLY
    min, max, level."""
    L = env["airframe_hi"][0] - env["airframe_lo"][0]
    wf = PROMO.first_layer(L, args.speed * 3.6, args.nu, args.yplus)
    levels = [{"patch": name, "bands": [{"distance": dist, "level": lvl} for dist, lvl in BANDS]}
              for name in DRONE_PATCHES]
    return {
        "input": {"surfaces": [{"path": PROMO._posix(os.path.join(geom_dir, name + ".stl")),
                                "name": name} for name in DRONE_PATCHES]},
        "domain": {"extent": tunnel_extent(env, args.base, args.variant),
                   "base_size": args.base},
        "refinement": {"levels": levels, "max_level": args.max_level,
                       "boxes": [{"min": b["min"], "max": b["max"], "level": b["level"]}
                                 for b in box_table(env, rotors, args.variant)]},
        "castellation": {"keep_region": "seed",
                         "seed_point": seed_point(env, args.variant)},
        "snap": {"feature_tolerance": 0.5 * 2.0 ** -args.max_level},
        "layers": {"patches": list(DRONE_PATCHES), "n": args.layers,
                   "first_thickness": wf["t1"], "growth": args.growth},
        "quality": {"max_closure": 1e-10, "max_non_orth_deg": args.non_orth_cap,
                    "report_non_orth_deg": 60.0, "min_thickness_ratio": 0.05,
                    "max_cond": 10000.0},
        "output": {"case_dir": PROMO._posix(os.path.join(out_dir, "case")),
                   "name": "drone_" + args.variant,
                   "patch_names": patch_names(args.variant)},
    }


def judge(run_rc, summary, check_rc, chk, reduced, variant) -> dict:
    """The gate: run, check, non_orth (STRICTLY under 70), all ten patches of
    required_patches(variant) present, and the cell count in 1-3 M - or
    REDUCED, which waives only the cell count."""
    reasons = []
    run = run_rc == 0 and summary is not None
    check = check_rc == 0 and bool(chk.get("passed"))
    mo = chk.get("max_non_orth_deg")
    non_orth = mo is not None and mo < NON_ORTH_GATE_DEG
    missing = []
    n = None
    cells = False
    if summary is None:
        patches = False
    else:
        rows = {r.get("name"): r.get("size", 0)
                for r in summary.get("mesh", {}).get("patches", [])}
        missing = [p for p in required_patches(variant) if rows.get(p, 0) <= 0]
        patches = not missing
        n = summary.get("mesh", {}).get("n_cells")
        cells = "REDUCED" if reduced else (isinstance(n, int) and CELLS_MIN <= n <= CELLS_MAX)
    ok = run and check and non_orth and patches and (cells is True or cells == "REDUCED")
    if not run:
        reasons.append("run: returncode " + str(run_rc)
                       + ("" if summary is not None else ", no summary"))
    if not check:
        reasons.append("check: returncode " + str(check_rc) + ", passed " + repr(chk.get("passed")))
    if not non_orth:
        reasons.append("non_orth: " + repr(mo))
    if not patches:
        reasons.append("patches: " + (", ".join(missing) if summary is not None else "no summary"))
    if not (cells is True or cells == "REDUCED"):
        reasons.append("cells: " + (repr(n) if summary is not None else "no summary"))
    return {"run": run, "check": check, "non_orth": non_orth, "patches": patches,
            "cells": cells, "pass": ok, "reasons": reasons}


def layer_coverage(summary) -> dict:
    """Per-patch prism-layer coverage from the layers stage row - reported,
    never gated: the per-patch rows, which patches kept layers, the layer
    cell count and a none/partial/all claim."""
    empty = {"patches": [], "patches_with_layers": [], "n_layer_cells": 0, "claim": "none"}
    if not summary:
        return empty
    row = next((r for r in summary.get("stages", []) if r.get("stage") == "layers"), None)
    if row is None:
        return empty
    keep = ("name", "n_layers", "t1_requested", "t1_min", "t1_mean",
            "full_area_frac", "drop_cause")
    patches = [{k: r.get(k) for k in keep} for r in row.get("patches", [])]
    kept = [p["name"] for p in patches if (p.get("n_layers") or 0) > 0]
    claim = "none" if not kept else ("all" if len(kept) == len(patches) else "partial")
    return {"patches": patches, "patches_with_layers": kept,
            "n_layer_cells": int(row.get("n_layer_cells") or 0), "claim": claim}


def snap_digest(summary):
    """The snap stage row (or None): the point/freeze/residual counters plus
    the per-patch STL-vs-snapped area table; missing keys read None."""
    if not summary:
        return None
    row = next((r for r in summary.get("stages", []) if r.get("stage") == "snap"), None)
    if row is None:
        return None
    keys = ("n_boundary_points", "within_tolerance_frac", "n_within_tolerance",
            "n_frozen", "n_local_undo", "n_abandoned", "n_pinned", "max_residual",
            "p99_residual", "converged", "iterations", "seconds")
    out = {k: row.get(k) for k in keys}
    out["area_ratio"] = [{"name": a.get("name"), "stl_area_m2": a.get("stl_area_m2"),
                          "castellated_area_m2": a.get("castellated_area_m2"),
                          "snapped_area_m2": a.get("snapped_area_m2"),
                          "ratio": a.get("ratio")}
                         for a in row.get("area_ratio", [])]
    return out


def build_report(args, env, extent, wf, reduced, cfg_path, run_rc, wall_s,
                 summary, check_rc, chk, gate, boxes) -> dict:
    """The tunnel_mesh.json report: flow convention, envelope, domain, boxes,
    wall-function sizing, the mesher's own summary rows, snap and layer
    digests, the parsed -check report and the gate."""
    digest = hashlib.sha256()
    with open(args.binary, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    mesher = {"binary": PROMO._posix(os.path.abspath(args.binary)),
              "binary_sha256": digest.hexdigest()[:8],
              "wall_seconds": wall_s, "returncode": run_rc,
              "total_seconds": None, "stages": [], "n_cells": None,
              "patches": {}, "quality": None}
    if summary is not None:
        mesher["total_seconds"] = summary.get("total_seconds")
        mesher["stages"] = [{"stage": r.get("stage"), "seconds": r.get("seconds")}
                            for r in summary.get("stages", [])]
        mesher["n_cells"] = summary.get("mesh", {}).get("n_cells")
        mesher["patches"] = {r.get("name"): r.get("size")
                             for r in summary.get("mesh", {}).get("patches", [])}
        mesher["quality"] = summary.get("quality")
    if args.variant == "forward":
        flow = {"variant": "forward", "velocity_m_s": [-args.speed, 0.0, 0.0],
                "inlet_patch": "inlet", "inlet_face": "xMax", "outlet_face": "xMin",
                "note": "drone flies +x at level attitude; freestream enters at xMax"
                        " and flows along -x"}
    else:
        flow = {"variant": "hover", "velocity_m_s": [0.0, 0.0, 0.0],
                "inlet_patch": "inlet", "inlet_face": "zMax", "outlet_face": "zMin",
                "note": "hover: no freestream; the rotor downwash leaves along -z"
                        " through zMin"}
    return {
        "tool": "tools/drone/tunnel_mesh.py",
        "variant": args.variant,
        "reduced": reduced,
        "geom_dir": PROMO._posix(os.path.abspath(args.geom)),
        "credit": CREDIT,
        "params": {"base": args.base, "max_level": args.max_level,
                   "speed_m_s": args.speed, "nu": args.nu, "yplus": args.yplus,
                   "layers": args.layers, "growth": args.growth,
                   "non_orth_cap_deg": args.non_orth_cap},
        "flow": flow,
        "envelope": {"lo": env["lo"], "hi": env["hi"],
                     "airframe_lo": env["airframe_lo"],
                     "airframe_hi": env["airframe_hi"], "D_m": env["D"]},
        "domain": {"extent": extent, "base_grid": PROMO.base_grid(extent, args.base),
                   "finest_cell_m": args.base / 2 ** args.max_level},
        "boxes": [{"name": n, **b} for n, b in zip(BOX_ORDER, boxes)],
        "wall_function": wf,
        "config_path": PROMO._posix(cfg_path),
        "mesher": mesher,
        "snap": snap_digest(summary),
        "layers": layer_coverage(summary),
        "check": {"returncode": check_rc, "quality_line": chk.get("quality_line"),
                  "parsed": chk},
        "gate": gate,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="tunnel_mesh.py",
        description="Write the automesher config for the PX4 x500 drone's free-flight "
                    "tunnel (forward flight at 15 m/s or hover), run the release "
                    "automesher and its -check, and judge its own gate.")
    p.add_argument("--geom", help="directory of the four patch STLs, rotors.json and the licence files")
    p.add_argument("--out", help="output directory, created if absent (never inside the repository)")
    p.add_argument("--variant", choices=("forward", "hover"), default="forward",
                   help="forward flight (+x, freestream from xMax) or hover (downwash to -z)")
    p.add_argument("--base", type=float, default=0.2, help="background cell size, m (default 0.2)")
    p.add_argument("--max-level", type=int, default=6, help="refinement level cap (default 6)")
    p.add_argument("--speed", type=float, default=15.0,
                   help="free-stream speed, m/s (default 15)")
    p.add_argument("--nu", type=float, default=1.5e-5,
                   help="kinematic viscosity, m2/s (default 1.5e-5)")
    p.add_argument("--yplus", type=float, default=5.0,
                   help="target y+ of the first cell centre (default 5)")
    p.add_argument("--layers", type=int, default=3,
                   help="boundary layers on the drone patches (default 3)")
    p.add_argument("--growth", type=float, default=1.2, help="layer growth ratio (default 1.2)")
    p.add_argument("--non-orth-cap", type=float, default=65.0,
                   help="config max_non_orth_deg; 65 keeps the snapped mesh under the "
                        "unit's strict 70 gate in -check's three decimals (default 65)")
    p.add_argument("--binary", default=os.path.join(
        _REPO, "rust", "target", "release",
        "ofgpu-automesher" + (".exe" if os.name == "nt" else "")),
        help="the release automesher binary")
    p.add_argument("--timeout", type=float, default=0.0,
                   help="seconds for the mesher subprocess, 0 = none (default 0)")
    p.add_argument("--dry-run", action="store_true",
                   help="run the mesher -dryRun only, write dryrun.log, exit")
    p.add_argument("--selftest", action="store_true", help="run T1-T8 and exit")
    args = p.parse_args(argv)

    if args.selftest:
        return selftest()
    if not args.geom or not args.out:
        p.error("--geom and --out are required unless --selftest")
    if not (0.0 < args.non_orth_cap <= NON_ORTH_GATE_DEG):
        PROMO.refuse("DT-CAP", "--non-orth-cap " + repr(args.non_orth_cap)
                     + ": the cap must be in (0, 70]; above 70 would loosen the unit's gate")
    out_abs = os.path.abspath(args.out)
    repo_case = os.path.normcase(os.path.abspath(_REPO))
    try:
        inside = os.path.commonpath([os.path.normcase(out_abs), repo_case]) == repo_case
    except ValueError:
        inside = False
    if inside:
        PROMO.refuse("DT-OUT", PROMO._posix(out_abs)
                     + ": --out is the repository root or inside it")
    if not os.path.isfile(args.binary):
        PROMO.refuse("DT-BINARY", args.binary + ": not found")
    rotors = read_rotors(args.geom)
    af_lo, af_hi = airframe_bbox(args.geom)
    env = envelope(af_lo, af_hi, rotors)
    os.makedirs(args.out, exist_ok=True)

    cfg = build_config(args.geom, args.out, env, rotors, args)
    cfg_path = os.path.abspath(os.path.join(args.out, "drone_tunnel.json"))
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=1)

    L = env["airframe_hi"][0] - env["airframe_lo"][0]
    wf = PROMO.first_layer(L, args.speed * 3.6, args.nu, args.yplus)
    extent = cfg["domain"]["extent"]
    reduced = (args.base != 0.2) or (args.max_level != 6)

    print("geom " + PROMO._posix(os.path.abspath(args.geom)))
    print("variant " + args.variant)
    print("D " + repr(env["D"]) + " m")
    print("envelope lo " + repr(env["lo"]) + " hi " + repr(env["hi"]))
    print("extent " + repr(extent) + ", base grid " + repr(PROMO.base_grid(extent, args.base)))
    print("wall function " + json.dumps(wf))
    print("config " + PROMO._posix(cfg_path))

    if args.dry_run:
        rc, _, _ = PROMO.run_stream([args.binary, PROMO._posix(cfg_path), "-dryRun"],
                                    args.out, os.path.join(args.out, "dryrun.log"),
                                    timeout=args.timeout)
        print("dry-run returncode " + str(rc))
        return 0 if rc == 0 else 1

    run_rc, wall_s, _ = PROMO.run_stream([args.binary, PROMO._posix(cfg_path)], args.out,
                                         os.path.join(args.out, "automesher.log"),
                                         timeout=args.timeout)
    summary = None
    sum_path = os.path.join(args.out, "case", "drone_" + args.variant + "_summary.json")
    if os.path.isfile(sum_path):
        with open(sum_path, "r", encoding="utf-8") as fh:
            summary = json.load(fh)
    check_rc, _, chk_text = PROMO.run_stream([args.binary, PROMO._posix(cfg_path), "-check"],
                                             args.out, os.path.join(args.out, "check.log"),
                                             timeout=args.timeout)
    chk = PROMO.parse_check(chk_text)

    gate = judge(run_rc, summary, check_rc, chk, reduced, args.variant)
    report = build_report(args, env, extent, wf, reduced, cfg_path, run_rc, wall_s,
                          summary, check_rc, chk, gate,
                          box_table(env, rotors, args.variant))
    with open(os.path.join(args.out, "tunnel_mesh.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    for name in ("rotors.json", "LICENSE.txt", "LICENSE"):
        src = os.path.join(args.geom, name)
        if os.path.isfile(src):
            shutil.copyfile(src, os.path.join(args.out, name))

    lay = layer_coverage(summary)
    print("layers: " + str(len(lay["patches_with_layers"])) + "/" + str(len(lay["patches"]))
          + " patches kept layers, " + str(lay["n_layer_cells"]) + " layer cells, claim "
          + lay["claim"])
    if gate["pass"]:
        print("GATE PASS REDUCED" if reduced else "GATE PASS")
        return 0
    print("GATE FAIL " + "; ".join(gate["reasons"]))
    return 1


_AF_LO = [-0.2035, -0.2035, -0.2276]
_AF_HI = [0.2035, 0.2035, 0.0514]
_ROTORS = [
    {"index": 0, "centre_m": [0.174, -0.174, 0.0512], "radius_m": 0.1468, "thickness_m": 0.0070},
    {"index": 1, "centre_m": [-0.174, 0.174, 0.0512], "radius_m": 0.1468, "thickness_m": 0.0070},
    {"index": 2, "centre_m": [0.174, 0.174, 0.0512], "radius_m": 0.1468, "thickness_m": 0.0070},
    {"index": 3, "centre_m": [-0.174, -0.174, 0.0512], "radius_m": 0.1468, "thickness_m": 0.0070},
]


def _close_list(got, want, tol):
    if len(got) != len(want) or any(abs(a - b) > tol for a, b in zip(got, want)):
        raise AssertionError(repr(got) + " != " + repr(want) + " (tol " + repr(tol) + ")")


def _t1():
    env = envelope(_AF_LO, _AF_HI, _ROTORS)
    if abs(env["D"] - 0.7857463197058371) > 1e-12:
        raise AssertionError("D " + repr(env["D"]))
    _close_list(env["lo"], [-0.3208, -0.3208, -0.2276], 1e-12)
    _close_list(env["hi"], [0.3208, 0.3208, 0.054700000000000006], 1e-12)
    if env["airframe_lo"] != _AF_LO or env["airframe_hi"] != _AF_HI:
        raise AssertionError("airframe_lo/hi changed")


def _t2():
    env = envelope(_AF_LO, _AF_HI, _ROTORS)
    for variant, base, want_ext, want_grid in (
            ("forward", 0.2, [-16.2, 8.2, -5.2, 5.2, -5.0, 4.8], [122, 52, 49]),
            ("hover", 0.2, [-5.2, 5.2, -5.2, 5.2, -16.0, 8.0], [52, 52, 120]),
            ("forward", 0.4, [-16.4, 8.4, -5.2, 5.2, -5.2, 4.8], [62, 26, 25]),
            ("hover", 0.4, [-5.2, 5.2, -5.2, 5.2, -16.0, 8.0], [26, 26, 60])):
        ext = tunnel_extent(env, base, variant)
        _close_list(ext, want_ext, 1e-9)
        grid = PROMO.base_grid(ext, base)
        if grid != want_grid:
            raise AssertionError(variant + " base " + repr(base) + ": grid " + repr(grid))


_DRONE_BOX = ([-0.3993746319705837, -0.3993746319705837, -0.3061746319705837],
              [0.3993746319705837, 0.3993746319705837, 0.13327463197058373])


def _t3():
    env = envelope(_AF_LO, _AF_HI, _ROTORS)
    disks = {
        "disk_0": ([0.01985999999999996, -0.32814, 0.0262],
                   [0.32814, -0.01985999999999996, 0.0762]),
        "disk_3": ([-0.32814, -0.32814, 0.0262],
                   [-0.01985999999999996, -0.01985999999999996, 0.0762]),
    }
    want = {
        "field": {"forward": ([-6.606770557646697, -1.8922926394116741, -2.584838959117511],
                              [1.8922926394116741, 1.8922926394116741, 1.6261926394116741]),
                  "hover": ([-1.8922926394116741, -1.8922926394116741, -6.5135705576466965],
                            [1.8922926394116741, 1.8922926394116741, 1.6261926394116741])},
        "wake": {"forward": ([-2.6780389591175116, -0.3993746319705837, -1.4062194795587557],
                             [0.3993746319705837, 0.3993746319705837, 0.13327463197058373]),
                 "hover": ([-0.3993746319705837, -0.3993746319705837, -2.584838959117511],
                           [0.3993746319705837, 0.3993746319705837, 0.13327463197058373])},
        "near_wake": {"forward": ([-1.1065463197058372, -0.3600873159852918, -0.6204731598529185],
                                  [0.3208, 0.3600873159852918, 0.09398731598529186]),
                      "hover": ([-0.3600873159852918, -0.3600873159852918, -1.013346319705837],
                                [0.3600873159852918, 0.3600873159852918, 0.054700000000000006])},
        "drone": {"forward": _DRONE_BOX, "hover": _DRONE_BOX},
    }
    for k, v in disks.items():
        want[k] = {"forward": v, "hover": v}
    for variant in ("forward", "hover"):
        boxes = box_table(env, _ROTORS, variant)
        if [b["name"] for b in boxes] != BOX_ORDER:
            raise AssertionError("names " + repr([b["name"] for b in boxes]))
        if [b["level"] for b in boxes] != [1, 3, 4, 4, 5, 5, 5, 5]:
            raise AssertionError("levels " + repr([b["level"] for b in boxes]))
        for b in boxes:
            if b["name"] not in want:
                continue
            exp = want[b["name"]][variant]
            _close_list(b["min"], exp[0], 1e-12)
            _close_list(b["max"], exp[1], 1e-12)


_F65 = None


def _f65():
    global _F65
    if _F65 is None:
        _F65 = PROMO.parse_check(PROMO.F_CHECK.replace("max 69.951 deg", "max 65.000 deg"))
    return _F65


def _t4():
    env = envelope(_AF_LO, _AF_HI, _ROTORS)
    for variant, seed_want, name_want in (
            ("forward", [4.2495315985291855, 0.0, 0.0], "drone_forward"),
            ("hover", [0.0, 0.0, 3.9834315985291853], "drone_hover")):
        args = argparse.Namespace(base=0.2, max_level=6, speed=15.0, nu=1.5e-5,
                                  yplus=5.0, layers=3, growth=1.2, non_orth_cap=65.0,
                                  variant=variant)
        cfg = build_config("C:/g", "C:/o", env, _ROTORS, args)
        if set(cfg) != {"input", "domain", "refinement", "castellation", "snap",
                        "layers", "quality", "output"}:
            raise AssertionError("top keys " + repr(sorted(cfg)))
        if set(cfg["refinement"]) != {"levels", "max_level", "boxes"}:
            raise AssertionError("refinement keys " + repr(sorted(cfg["refinement"])))
        boxes = cfg["refinement"]["boxes"]
        if len(boxes) != 8 or any(set(b) != {"min", "max", "level"} for b in boxes):
            raise AssertionError("boxes shape")
        surfs = cfg["input"]["surfaces"]
        if [s["name"] for s in surfs] != ["body", "arms", "motors", "skids"]:
            raise AssertionError("surface order " + repr([s["name"] for s in surfs]))
        if surfs[0]["path"] != "C:/g/body.stl":
            raise AssertionError("first path " + repr(surfs[0]["path"]))
        if [lv["patch"] for lv in cfg["refinement"]["levels"]] != ["body", "arms", "motors", "skids"]:
            raise AssertionError("levels patch order")
        for lv in cfg["refinement"]["levels"]:
            if lv["bands"] != [{"distance": 0.006, "level": 6},
                               {"distance": 0.02, "level": 5},
                               {"distance": 0.05, "level": 4}]:
                raise AssertionError("bands " + repr(lv["bands"]))
        if cfg["snap"] != {"feature_tolerance": 0.0078125}:
            raise AssertionError("snap " + repr(cfg["snap"]))
        if cfg["quality"] != {"max_closure": 1e-10, "max_non_orth_deg": 65.0,
                              "report_non_orth_deg": 60.0, "min_thickness_ratio": 0.05,
                              "max_cond": 10000.0}:
            raise AssertionError("quality " + repr(cfg["quality"]))
        lay = cfg["layers"]
        if lay["n"] != 3 or lay["growth"] != 1.2 or lay["patches"] != ["body", "arms", "motors", "skids"]:
            raise AssertionError("layers shape " + repr(lay))
        t1 = lay["first_thickness"]
        if abs(t1 - 0.00021150156505846693) > 1e-9 * abs(t1):
            raise AssertionError("t1 " + repr(t1))
        _close_list(cfg["castellation"]["seed_point"], seed_want, 1e-12)
        if cfg["output"]["case_dir"] != "C:/o/case" or cfg["output"]["name"] != name_want:
            raise AssertionError("output " + repr(cfg["output"]))
        if cfg["output"]["patch_names"] != patch_names(variant):
            raise AssertionError("patch_names")
        want_ext = ([-16.2, 8.2, -5.2, 5.2, -5.0, 4.8] if variant == "forward"
                    else [-5.2, 5.2, -5.2, 5.2, -16.0, 8.0])
        _close_list(cfg["domain"]["extent"], want_ext, 1e-9)
        if cfg["domain"]["base_size"] != 0.2 or cfg["refinement"]["max_level"] != 6:
            raise AssertionError("domain/refinement scalars")


def _summ(n, variant):
    return {"mesh": {"n_cells": n,
                     "patches": [{"name": p, "size": 10} for p in required_patches(variant)]}}


def _t5():
    if required_patches("forward") != ["inlet", "outlet", "side_ymin", "side_ymax",
                                       "bottom", "top", "body", "arms", "motors", "skids"]:
        raise AssertionError("required forward")
    if required_patches("hover") != ["inlet", "outlet", "side_xmin", "side_xmax",
                                     "side_ymin", "side_ymax", "body", "arms",
                                     "motors", "skids"]:
        raise AssertionError("required hover")
    chk = _f65()
    g = judge(0, _summ(1500000, "forward"), 0, chk, False, "forward")
    if g["pass"] is not True or g["cells"] is not True:
        raise AssertionError("(a) " + repr(g))
    g = judge(0, _summ(900000, "forward"), 0, chk, False, "forward")
    if g["pass"] or not any(r.startswith("cells") for r in g["reasons"]):
        raise AssertionError("(b) " + repr(g))
    g = judge(0, _summ(3100000, "forward"), 0, chk, False, "forward")
    if g["pass"] or not any(r.startswith("cells") for r in g["reasons"]):
        raise AssertionError("(c) " + repr(g))
    g = judge(0, _summ(150000, "forward"), 0, chk, True, "forward")
    if g["pass"] is not True or g["cells"] != "REDUCED":
        raise AssertionError("(d) " + repr(g))
    chk70 = dict(chk)
    chk70["max_non_orth_deg"] = 70.0
    g = judge(0, _summ(1500000, "forward"), 0, chk70, False, "forward")
    if g["pass"] or not any(r.startswith("non_orth") for r in g["reasons"]):
        raise AssertionError("(e) " + repr(g))
    rows = [r for r in _summ(1500000, "hover")["mesh"]["patches"] if r["name"] != "side_xmin"]
    g = judge(0, {"mesh": {"n_cells": 1500000, "patches": rows}}, 0, chk, False, "hover")
    if g["pass"] or not any(r.startswith("patches") and "side_xmin" in r for r in g["reasons"]):
        raise AssertionError("(f) " + repr(g))
    chk_bad = dict(chk)
    chk_bad["passed"] = False
    g = judge(0, _summ(1500000, "forward"), 0, chk_bad, False, "forward")
    if g["pass"] or not any(r.startswith("check") for r in g["reasons"]):
        raise AssertionError("(g) " + repr(g))


def _t6():
    chk = _f65()
    if chk["max_non_orth_deg"] != 65.0 or chk["passed"] is not True:
        raise AssertionError(repr(chk["max_non_orth_deg"]) + " " + repr(chk["passed"]))


def _t7():
    tmp = tempfile.mkdtemp()
    empty = os.path.join(tmp, "empty")
    os.makedirs(empty)

    def run(argv, needle):
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                main(argv)
        except SystemExit as exc:
            if exc.code != 2:
                raise AssertionError("exit " + repr(exc.code) + ": " + buf.getvalue())
        else:
            raise AssertionError("no SystemExit: " + buf.getvalue())
        if needle not in buf.getvalue():
            raise AssertionError("missing " + needle + ": " + buf.getvalue())

    run(["--geom", tmp, "--out", tmp, "--non-orth-cap", "75"], "DT-CAP")
    run(["--geom", tmp, "--out", os.path.join(_REPO, "cases", "x")], "DT-OUT")
    out3 = os.path.join(tmp, "out3")
    run(["--geom", empty, "--out", out3, "--binary", sys.executable], "DT-ROTORS")
    if os.path.exists(out3):
        raise AssertionError("wrote into --out")


def _t8():
    def row(name, n):
        return {"name": name, "n_layers": n, "t1_requested": 0.0004, "t1_min": 0.0,
                "t1_mean": 0.0, "full_area_frac": 0.0,
                "drop_cause": "zero_disp" if n == 0 else None}

    def stage(n_cells, rows):
        return {"stages": [{"stage": "layers", "seconds": 13.6,
                            "n_layer_cells": n_cells, "patches": rows}]}

    lay = layer_coverage(stage(1234, [row("body", 2), row("arms", 0),
                                      row("motors", 0), row("skids", 2)]))
    if lay["patches_with_layers"] != ["body", "skids"] or lay["n_layer_cells"] != 1234:
        raise AssertionError("(a) " + repr(lay))
    if lay["claim"] != "partial" or len(lay["patches"]) != 4:
        raise AssertionError("(a) " + repr(lay))
    if set(lay["patches"][0]) != {"name", "n_layers", "t1_requested", "t1_min",
                                  "t1_mean", "full_area_frac", "drop_cause"}:
        raise AssertionError("(a) row keys " + repr(sorted(lay["patches"][0])))
    lay = layer_coverage(stage(0, [row("body", 0), row("arms", 0),
                                   row("motors", 0), row("skids", 0)]))
    if lay["claim"] != "none" or lay["patches_with_layers"] != []:
        raise AssertionError("(b) " + repr(lay))
    lay = layer_coverage(stage(900, [row("body", 3), row("arms", 3),
                                     row("motors", 3), row("skids", 3)]))
    if lay["claim"] != "all":
        raise AssertionError("(c) " + repr(lay))
    if layer_coverage({"stages": []}) != {"patches": [], "patches_with_layers": [],
                                          "n_layer_cells": 0, "claim": "none"}:
        raise AssertionError("(d)")


def selftest() -> int:
    """T1-T8, no geometry and no mesher; temp files only under tempfile.mkdtemp()."""
    tests = [("T1", _t1), ("T2", _t2), ("T3", _t3), ("T4", _t4),
             ("T5", _t5), ("T6", _t6), ("T7", _t7), ("T8", _t8)]
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


if __name__ == "__main__":
    sys.exit(main())
