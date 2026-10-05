#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.

"""The drone showreel's run review: card, rule-table remedy, re-mesh, compare.

Three subcommands. `review` reads a finished tunnel mesh and solve and writes
the review card: eight checks, one per number the run already reported, each
with an ok/warn/fail status and a plain sentence. `improve` re-meshes the base
case with the ONE remedy the card's fixed rule table picks, runs the mesher's
-check and re-judges the gate. `compare` lines the before and after cards up
metric by metric into an improved / not-improved verdict.

The remedy is chosen by a fixed rule table walked in order - no model, no
randomness: every (finding, status) match is recorded in "considered" and the
first APPLICABLE one wins.

The numbers are demonstration numbers from an unvalidated setup: no validated
aerodynamic coefficient anywhere.

BSD-3-Clause credit rule: the geometry is the PX4 x500 assembly, "(c) 2022
Rudis Laboratories / PX4 Autopilot for Drones", and the drone meshes, solves
and review cards are NEVER written inside the repository - every --out lives
outside it. tunnel_mesh is reached only through load_tunnel_mesh().
"""

import argparse
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import types

TOOL = "tools/drone/review.py"
VERSION = "drone-review/1"
CREDIT = "(c) 2022 Rudis Laboratories / PX4 Autopilot for Drones, BSD-3-Clause"
WALLS = ["body", "arms", "motors", "skids"]
RHO = 1.2041
NON_ORTH_FAIL_DEG = 70.0
SNAP_WITHIN_MIN = 0.99
AREA_MIN = 0.8
YPLUS_LO, YPLUS_HI = 30.0, 300.0
MASS_REL_MAX = 1e-6
FULL_END_TIME = 0.5
OCTREE_MAX_LEVEL = 6   # the automesher's octree cap, SPEC-LIT §74.2 (rust/src/automesher/octree.rs MAX_LEVEL)
FINDING_ORDER = ["RV-CHECK", "RV-NONORTH", "RV-SNAP", "RV-AREA", "RV-LAYERS",
                 "RV-YPLUS", "RV-RESID", "RV-MASS"]
# (finding id, status that triggers, remedy id) - every match is recorded in
# "considered"; the FIRST applicable one is chosen.
RULES = [
    ("RV-CHECK", "fail", "REM-STOP"), ("RV-MASS", "fail", "REM-STOP"),
    ("RV-NONORTH", "fail", "REM-STOP"), ("RV-YPLUS", "fail", "REM-STOP"),
    ("RV-RESID", "fail", "REM-STOP"), ("RV-AREA", "warn", "REM-WALL-LEVEL"),
    ("RV-YPLUS", "warn", "REM-FIRST-LAYER"), ("RV-RESID", "warn", "REM-EXTEND"),
]
HONESTY = "Demonstration numbers from an unvalidated setup: no validated aerodynamic coefficients."
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_BINARY = os.path.join(_REPO, "rust", "target", "release",
                              "ofgpu-automesher" + (".exe" if os.name == "nt" else ""))


def load_tunnel_mesh() -> types.ModuleType:
    """R1: reach tunnel_mesh only through this import; NEVER import it by name."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.normpath(os.path.join(here, "tunnel_mesh.py"))
    spec = importlib.util.spec_from_file_location("drone_tunnel_mesh", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TM = load_tunnel_mesh()
PROMO = TM.PROMO


def _in_repo(path) -> bool:
    """True when path is the repository root or inside it (DR-OUT)."""
    rp = os.path.normcase(os.path.realpath(path))
    repo = os.path.normcase(os.path.realpath(_REPO))
    return rp == repo or rp.startswith(repo + os.sep)


def _fnum(v, d=0.0):
    """v when it is a real number (bools are not numbers here), else d."""
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else d


def _isnum(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def parse_yplus(text: str) -> dict:
    """{patch: {"min","mean","max"}} from every "y+ at <patch>: min a | mean b |
    max c" line; the last occurrence of a patch wins, {} when none."""
    out = {}
    for raw in text.splitlines():
        s = raw.strip()
        if not s.startswith("y+ at ") or ":" not in s:
            continue
        head, _, rest = s.partition(":")
        patch = head[len("y+ at "):].strip()
        if not patch:
            continue
        row = {}
        for chunk in rest.split("|"):
            parts = chunk.split()
            if len(parts) == 2 and parts[0] in ("min", "mean", "max"):
                try:
                    row[parts[0]] = float(parts[1])
                except ValueError:
                    pass
        if len(row) == 3:
            out[patch] = row
    return out


def parse_mass(text: str):
    """{"net_kg_s","l1_kg_s","worst_kg_s"} from the LAST
    "net = a kg/s | L1 = b kg/s | worst cell = c kg/s" line; None when none."""
    for raw in reversed(text.splitlines()):
        s = raw.strip()
        if not (s.startswith("net = ") and "L1 = " in s and "worst cell = " in s):
            continue
        vals = []
        for part in s.split("|"):
            tok = part.split("=", 1)
            nums = tok[1].split() if len(tok) == 2 else []
            try:
                vals.append(float(nums[0]))
            except (ValueError, IndexError):
                vals = None
                break
        if vals is not None and len(vals) == 3:
            return {"net_kg_s": vals[0], "l1_kg_s": vals[1], "worst_kg_s": vals[2]}
    return None


def inlet_mass_flow(tunnel: dict, rho: float = RHO):
    """Forward only: rho x |U| x the y-z extent, the xMax inlet face."""
    v = (tunnel.get("flow") or {}).get("velocity_m_s")
    ext = (tunnel.get("domain") or {}).get("extent")
    if not isinstance(v, list) or len(v) != 3 or not isinstance(ext, list) or len(ext) != 6:
        return None
    speed = sum(_fnum(c) * _fnum(c) for c in v) ** 0.5
    return rho * speed * (ext[3] - ext[2]) * (ext[5] - ext[4])


def _mass_terms(solve, log_text, tunnel):
    """(parsed mass, mdot_in, net, l1, mass_rel); any missing piece reads None."""
    m = parse_mass(log_text)
    if m is None:
        return None, None, None, None, None
    rho = (solve.get("params") or {}).get("rho")
    mdot = inlet_mass_flow(tunnel, rho if _isnum(rho) else RHO)
    rel = abs(m["net_kg_s"]) / mdot if mdot else None
    return m, mdot, m["net_kg_s"], m["l1_kg_s"], rel


def finding(fid, status, value, limit, detail, say) -> dict:
    """One review finding; status in ok | warn | fail."""
    return {"id": fid, "status": status, "value": value, "limit": limit,
            "detail": detail, "say": say}


def review_mesh(tunnel: dict) -> list:
    """The five mesh findings, in FINDING_ORDER (R4)."""
    chk = tunnel.get("check", {}).get("parsed") or {}
    snap = tunnel.get("snap") or {}
    lay = tunnel.get("layers") or {}
    out = []

    passed = chk.get("passed")
    n_cells = chk.get("n_cells")
    n_regions = chk.get("n_regions")
    if passed is True and n_regions == 1:
        out.append(finding("RV-CHECK", "ok", passed, True, None,
                           f"The mesh passes its own check: {_fnum(n_cells) / 1e6:.2f}"
                           " million cells in one connected region."))
    else:
        out.append(finding("RV-CHECK", "fail", passed, True, None,
                           "The mesh fails its own check."))

    v = chk.get("max_non_orth_deg")
    n_over = chk.get("n_over_report") or 0
    if _isnum(v) and v >= NON_ORTH_FAIL_DEG:
        out.append(finding("RV-NONORTH", "fail", v, NON_ORTH_FAIL_DEG, None,
                           f"Face skew reaches {v:.0f} degrees, past the 70-degree limit."))
    elif n_over > 0:
        out.append(finding("RV-NONORTH", "warn", v, NON_ORTH_FAIL_DEG, None,
                           f"The worst face is skewed {_fnum(v):.0f} degrees, and {n_over}"
                           " faces pass 60 degrees: the pressure solve works harder there."))
    else:
        out.append(finding("RV-NONORTH", "ok", v, NON_ORTH_FAIL_DEG, None,
                           f"The worst face is skewed {_fnum(v):.0f} degrees, inside the limit."))

    v = snap.get("within_tolerance_frac")
    if _isnum(v) and v < SNAP_WITHIN_MIN:
        out.append(finding("RV-SNAP", "warn", v, SNAP_WITHIN_MIN, None,
                           f"Only {v * 100:.1f} percent of the surface points"
                           " snapped onto the geometry."))
    else:
        out.append(finding("RV-SNAP", "ok", v, SNAP_WITHIN_MIN, None,
                           f"{_fnum(v) * 100:.1f} percent of the surface points"
                           " snapped onto the geometry."))

    rows = {r.get("name"): r.get("ratio") for r in (snap.get("area_ratio") or [])}
    best = None
    for w in WALLS:
        r = rows.get(w)
        if _isnum(r) and (best is None or r < best[0]):
            best = (r, w)
    if best is None:
        out.append(finding("RV-AREA", "warn", None, AREA_MIN, None,
                           "No wall surface capture was reported."))
    else:
        r, patch = best
        if r < AREA_MIN:
            out.append(finding("RV-AREA", "warn", r, AREA_MIN, patch,
                               f"The {patch} surface is under-resolved: the mesh captures"
                               f" only {r * 100:.0f} percent of it."))
        else:
            out.append(finding("RV-AREA", "ok", r, AREA_MIN, patch,
                               "The mesh captures at least 80 percent of every surface."))

    claim = lay.get("claim")
    kept = lay.get("patches_with_layers") or []
    if claim == "all":
        out.append(finding("RV-LAYERS", "ok", claim, "all", None,
                           "Prism layers cover every wall."))
    elif claim == "partial":
        out.append(finding("RV-LAYERS", "warn", claim, "all", None,
                           f"Prism layers cover only the {' and '.join(kept)} surfaces."))
    else:
        out.append(finding("RV-LAYERS", "warn", claim, "all", None,
                           "No prism layers on the walls yet: wall functions carry"
                           " the boundary layer."))
    return out


def review_solve(solve: dict, log_text: str, tunnel: dict) -> list:
    """The three solve findings, in FINDING_ORDER (R5)."""
    out = []
    crit = solve.get("criteria") or {}
    yp = parse_yplus(log_text)
    missing = [w for w in WALLS if not isinstance(yp.get(w), dict)]
    if not yp or missing:
        out.append(finding("RV-YPLUS", "fail", [], [YPLUS_LO, YPLUS_HI], None,
                           "No wall y-plus was reported."))
    else:
        bad = [w for w in WALLS if not (YPLUS_LO <= yp[w]["mean"] <= YPLUS_HI)]
        det = {w: yp[w]["mean"] for w in bad}
        if bad:
            names = " and ".join(bad)
            means = ", ".join(f"{yp[w]['mean']:.0f}" for w in bad)
            out.append(finding("RV-YPLUS", "warn", bad, [YPLUS_LO, YPLUS_HI], det,
                               f"Wall y-plus leaves the wall-function band on the"
                               f" {names} surface: mean {means}."))
        else:
            out.append(finding("RV-YPLUS", "ok", [], [YPLUS_LO, YPLUS_HI], None,
                               "Wall y-plus sits between 30 and 300 on every surface,"
                               " where wall functions hold."))

    cls = solve.get("class")
    if cls == "steady":
        out.append(finding("RV-RESID", "ok", cls, "steady", solve.get("failed"),
                           "Residuals and forces have settled: the run is steady."))
    elif cls == "unsteady":
        cd = _fnum((crit.get("Cd_rel_change") or {}).get("value"))
        cl = _fnum((crit.get("Cl_rel_change") or {}).get("value"))
        out.append(finding("RV-RESID", "warn", cls, "steady", solve.get("failed"),
                           f"The run is unsteady: drag still moves {cd * 100:.2f} percent"
                           f" and lift {cl * 100:.1f} percent over the last window."))
    else:
        out.append(finding("RV-RESID", "fail", cls, "steady", solve.get("failed"),
                           f"The run ended {cls}."))

    m, mdot, net, l1, rel = _mass_terms(solve, log_text, tunnel)
    if m is None or rel is None:
        out.append(finding("RV-MASS", "fail", None, MASS_REL_MAX, None,
                           "No mass balance was reported."))
    elif rel > MASS_REL_MAX:
        out.append(finding("RV-MASS", "fail", rel, MASS_REL_MAX, None,
                           f"Mass balance is off by {rel:.1e} of the inflow."))
    else:
        out.append(finding("RV-MASS", "ok", rel, MASS_REL_MAX, None,
                           f"Mass balance closes to {rel:.1e} of the inflow."))
    return out


def card_from(label, mesh_dir, solve_dir, tunnel, cfg, solve, log_text) -> dict:
    """The review card, pure: the four files are already read (see contract)."""
    chk = tunnel.get("check", {}).get("parsed") or {}
    snap = tunnel.get("snap") or {}
    lay = tunnel.get("layers") or {}
    crit = solve.get("criteria") or {}
    run = solve.get("run") or {}
    ff = solve.get("final_forces") or {}
    params = solve.get("params") or {}

    def crit_val(key):
        row = crit.get(key)
        return row.get("value") if isinstance(row, dict) else None

    ar = {r.get("name"): r.get("ratio") for r in (snap.get("area_ratio") or [])}
    yp = parse_yplus(log_text)
    m, mdot, net, l1, rel = _mass_terms(solve, log_text, tunnel)
    findings = review_mesh(tunnel) + review_solve(solve, log_text, tunnel)
    counts = {"ok": sum(1 for f in findings if f["status"] == "ok"),
              "warn": sum(1 for f in findings if f["status"] == "warn"),
              "fail": sum(1 for f in findings if f["status"] == "fail")}
    verdict = "fail" if counts["fail"] else ("warn" if counts["warn"] else "ok")
    end_time = params.get("end_time")
    reduced = {"mesh": bool(tunnel.get("reduced")),
               "solve": bool(_isnum(end_time) and end_time < FULL_END_TIME),
               "end_time": end_time, "n_steps": run.get("n_steps")}
    metrics = {
        "n_cells": chk.get("n_cells"),
        "max_non_orth_deg": chk.get("max_non_orth_deg"),
        "n_non_orth_over_60": chk.get("n_over_report"),
        "check_passed": chk.get("passed"),
        "n_regions": chk.get("n_regions"),
        "snap_within_tolerance_frac": snap.get("within_tolerance_frac"),
        "area_ratio": {w: ar.get(w) for w in WALLS},
        "layers_claim": lay.get("claim"),
        "yplus": {w: yp.get(w) for w in WALLS},
        "class": solve.get("class"),
        "failed": solve.get("failed"),
        "U_decades": crit_val("U_decades"),
        "p_decades": crit_val("p_decades"),
        "Cd_rel_change": crit_val("Cd_rel_change"),
        "Cl_rel_change": crit_val("Cl_rel_change"),
        "cont_err": crit_val("cont_err"),
        "mass_net_kg_s": net,
        "mass_l1_kg_s": l1,
        "mdot_in_kg_s": mdot,
        "mass_rel": rel,
        "drag_N": ff.get("drag_N"),
        "lift_N": ff.get("lift_N"),
        "drag_area_m2": ff.get("drag_area_m2"),
        "solver_seconds": run.get("solver_seconds"),
        "end_time": end_time,
        "base_size_m": (cfg.get("domain") or {}).get("base_size"),
    }
    card = {"tool": TOOL, "version": VERSION, "credit": CREDIT, "label": label,
            "mesh_dir": PROMO._posix(mesh_dir), "solve_dir": PROMO._posix(solve_dir),
            "reduced": reduced, "metrics": metrics, "findings": findings,
            "counts": counts, "verdict": verdict}
    remedy = choose_remedy(card, cfg)
    card["mesh_remedy"] = tunnel.get("remedy")
    card["remedy"] = remedy
    summary = (f"Review of the {label} run: {counts['ok']} checks pass,"
               f" {counts['warn']} warnings, {counts['fail']} failures.")
    card["say"] = [summary] + [f["say"] for f in findings] + [remedy["say"]]
    card["honesty"] = HONESTY
    return card


def _wall_level_state(row, metrics, cfg):
    """(applicable, reason, fields) for REM-WALL-LEVEL. The rule reasons on the
    EFFECTIVE level min(smallest-band level, refinement.max_level) - the
    mesher silently caps band levels at max_level - and refines EVERY
    under-resolved wall one level together (refining one wall alone puts a
    level jump at the wall junctions, which sank the arms-only trial mesh), never
    past the octree's cap (SPEC-LIT §74.2); the snap feature tolerance follows
    the mesher's half-a-finest-cell rule."""
    ref = cfg.get("refinement") or {}
    max_from = ref.get("max_level")
    base = (cfg.get("domain") or {}).get("base_size")
    targets = []
    for w in WALLS:
        r = (metrics.get("area_ratio") or {}).get(w)
        if not (_isnum(r) and r < AREA_MIN):
            continue
        entry = next((e for e in (ref.get("levels") or [])
                      if e.get("patch") == w), None)
        if entry is None:
            continue
        bands = [b for b in (entry.get("bands") or []) if _isnum(b.get("distance"))]
        if not bands:
            continue
        band = min(bands, key=lambda b: b["distance"])
        if _isnum(band.get("level")) and _isnum(max_from):
            level_from = min(band["level"], max_from)
        elif _isnum(band.get("level")):
            level_from = band["level"]
        else:
            continue
        if not (isinstance(level_from, int) and not isinstance(level_from, bool)
                and level_from + 1 <= OCTREE_MAX_LEVEL):
            continue
        targets.append({"patch": w, "band_distance_m": band.get("distance"),
                        "level_from": level_from, "level_to": level_from + 1})
    if not targets:
        return False, (f"every surface under 80 percent capture is already at"
                       f" the octree's level cap {OCTREE_MAX_LEVEL}"
                       " (SPEC-LIT §74.2)"), {}
    patches = [t["patch"] for t in targets]
    worst = row.get("detail")
    top = next((t for t in targets if t["patch"] == worst), targets[0])
    max_to = (max([max_from] + [t["level_to"] for t in targets])
              if _isnum(max_from)
              else max(t["level_to"] for t in targets))
    if _isnum(base) and _isnum(top["level_from"]) and _isnum(top["level_to"]):
        cell_from, cell_to = base / 2 ** top["level_from"], base / 2 ** top["level_to"]
    else:
        cell_from, cell_to = 0.0, 0.0
    snap = cfg.get("snap")
    ft_from = snap.get("feature_tolerance") if isinstance(snap, dict) else None
    ft_to = (0.5 * 2.0 ** -max_to
             if max_to != max_from and ft_from is not None else ft_from)
    fields = {"patch": top["patch"], "patches": patches, "bands": targets,
              "band_distance_m": top["band_distance_m"],
              "level_from": top["level_from"], "level_to": top["level_to"],
              "max_level_from": max_from, "max_level_to": max_to,
              "base_size_m": base, "cell_from_m": cell_from, "cell_to_m": cell_to,
              "feature_tolerance_from": ft_from, "feature_tolerance_to": ft_to}
    return True, None, fields


def choose_remedy(card: dict, cfg: dict) -> dict:
    """Walk RULES in order; the first APPLICABLE match is the remedy (R6)."""
    findings = {f.get("id"): f for f in card.get("findings", [])}
    metrics = card.get("metrics") or {}
    considered = []
    chosen = None
    for fid, status, rid in RULES:
        row = findings.get(fid)
        if row is None or row.get("status") != status:
            continue
        if rid == "REM-STOP":
            ok, reason, fields = True, None, {}
        elif rid == "REM-WALL-LEVEL":
            ok, reason, fields = _wall_level_state(row, metrics, cfg)
        elif rid == "REM-FIRST-LAYER":
            if metrics.get("layers_claim") == "none":
                ok, reason, fields = False, ("no wall keeps prism layers in this"
                                             " mesh, so a first-layer change has"
                                             " no effect"), {}
            else:
                ok, reason, fields = True, None, {}
        else:
            t0 = metrics.get("end_time")
            ok, reason, fields = True, None, {
                "end_time_from": t0, "end_time_to": t0 * 2 if _isnum(t0) else 0.0}
        considered.append({"finding": fid, "remedy": rid, "applicable": ok,
                           "reason": reason})
        if ok and chosen is None:
            chosen = (rid, fid, fields)
    if chosen is None:
        return {"id": "REM-NONE", "finding": None, "considered": considered,
                "say": "Nothing to improve by rule."}
    rid, fid, fields = chosen
    if rid == "REM-WALL-LEVEL":
        ps = fields["patches"]
        names = ps[0] if len(ps) == 1 else ", ".join(ps[:-1]) + " and " + ps[-1]
        say = (f"Remedy: refine every surface under 80 percent capture one"
               f" level ({names}): the {fields['patch']} from"
               f" {fields['cell_from_m'] * 1000:g} to {fields['cell_to_m'] * 1000:g}"
               " millimetre cells, then re-mesh and re-solve.")
    elif rid == "REM-FIRST-LAYER":
        say = "Remedy: change the first layer thickness to bring y-plus into the band."
    elif rid == "REM-EXTEND":
        say = (f"Remedy: run twice as long, to {fields['end_time_to']:g} seconds,"
               " and review again.")
    else:
        say = "No automatic remedy: a person must look at this run."
    out = {"id": rid, "finding": fid, "considered": considered, "say": say}
    out.update(fields)
    return out


def apply_remedy(cfg: dict, remedy: dict) -> dict:
    """A deep copy of cfg, changed in exactly FOUR ordered steps (R7): (1)
    CLAMP - every band level in refinement.levels (every patch) and every
    refinement.boxes[*].level above max_level_from drops to it, which changes
    nothing in the mesh (the mesher already caps them there) and only makes
    the config say what the mesh is; (2) EVERY band listed in remedy["bands"]
    sits at its level_to - the remedy refines every under-resolved wall one
    level together, so no junction is left with a level jump; (3)
    refinement.max_level sits at max_level_to; (4) when the remedy carries a
    numeric feature_tolerance_to and the config has snap.feature_tolerance,
    it follows to the mesher's half-a-finest-cell rule. Refuses DR-REMEDY
    otherwise: the id must be REM-WALL-LEVEL with a non-empty "bands" list,
    each entry's level check is on the EFFECTIVE level min(band level,
    refinement.max_level), and no level_to passes the octree's cap
    (SPEC-LIT §74.2)."""
    rem = remedy or {}
    if rem.get("id") != "REM-WALL-LEVEL":
        PROMO.refuse("DR-REMEDY", repr(rem.get("id"))
                     + " is not a mesh remedy; improve re-meshes only")
    specs = rem.get("bands")
    if not isinstance(specs, list) or not specs:
        PROMO.refuse("DR-REMEDY", "the remedy lists no bands to refine")
    out = copy.deepcopy(cfg)
    ref = out.get("refinement") or {}
    max_from = ref.get("max_level")
    for spec in specs:
        patch = spec.get("patch")
        entry = next((e for e in (ref.get("levels") or [])
                      if e.get("patch") == patch), None)
        if entry is None:
            PROMO.refuse("DR-REMEDY", "patch " + repr(patch)
                         + " has no level entry in refinement.levels")
        band = next((b for b in (entry.get("bands") or [])
                     if b.get("distance") == spec.get("band_distance_m")), None)
        if band is None:
            PROMO.refuse("DR-REMEDY", "no band at distance "
                         + repr(spec.get("band_distance_m")) + " on patch "
                         + repr(patch))
        effective = (min(band["level"], max_from)
                     if _isnum(band.get("level")) and _isnum(max_from)
                     else band.get("level"))
        if effective != spec.get("level_from"):
            PROMO.refuse("DR-REMEDY", "the " + repr(patch) + " band sits at"
                         " effective level " + repr(effective)
                         + ", not the remedy's level_from "
                         + repr(spec.get("level_from")))
        if not (_isnum(spec.get("level_to")) and spec["level_to"] <= OCTREE_MAX_LEVEL):
            PROMO.refuse("DR-REMEDY", "level_to " + repr(spec.get("level_to"))
                         + " passes the octree's level cap " + str(OCTREE_MAX_LEVEL)
                         + " (SPEC-LIT §74.2)")
    for e in (ref.get("levels") or []):
        for b in (e.get("bands") or []):
            if _isnum(b.get("level")) and _isnum(max_from) and b["level"] > max_from:
                b["level"] = max_from
    for bx in (ref.get("boxes") or []):
        if _isnum(bx.get("level")) and _isnum(max_from) and bx["level"] > max_from:
            bx["level"] = max_from
    for spec in specs:
        entry = next(e for e in ref["levels"] if e.get("patch") == spec.get("patch"))
        next(b for b in entry["bands"]
             if b.get("distance") == spec.get("band_distance_m"))["level"] = spec.get("level_to")
    ref["max_level"] = rem.get("max_level_to")
    ft_to = rem.get("feature_tolerance_to")
    snap = out.get("snap")
    if _isnum(ft_to) and isinstance(snap, dict) and "feature_tolerance" in snap:
        snap["feature_tolerance"] = ft_to
    return out


_ROW_SPEC = ([("n_cells", None), ("max_non_orth_deg", "lower"),
              ("n_non_orth_over_60", "lower"), ("snap_within_tolerance_frac", "higher")]
             + [("area_ratio." + w, "higher") for w in WALLS]
             + [("yplus_mean." + w, None) for w in WALLS]
             + [("Cd_rel_change", "lower"), ("Cl_rel_change", "lower"),
                ("mass_rel", "lower"), ("drag_N", None), ("solver_seconds", None)])
_SEVERITY = {"ok": 0, "warn": 1, "fail": 2}


def _metric(metrics, key):
    if key.startswith("area_ratio."):
        return (metrics.get("area_ratio") or {}).get(key.split(".", 1)[1])
    if key.startswith("yplus_mean."):
        row = (metrics.get("yplus") or {}).get(key.split(".", 1)[1])
        return row.get("mean") if isinstance(row, dict) else None
    return metrics.get(key)


def compare_cards(before: dict, after: dict) -> dict:
    """Line the two cards up metric by metric (R8); refuses DR-INPUT unless
    the after card carries a REM-WALL-LEVEL mesh_remedy."""
    rem = after.get("mesh_remedy")
    if not isinstance(rem, dict) or rem.get("id") != "REM-WALL-LEVEL":
        PROMO.refuse("DR-INPUT", "the after card carries no REM-WALL-LEVEL"
                     " mesh_remedy; compare judges one automatic re-mesh")
    mb, ma = before.get("metrics") or {}, after.get("metrics") or {}
    rows = []
    for key, better in _ROW_SPEC:
        b, a = _metric(mb, key), _metric(ma, key)
        delta = (a - b) if (_isnum(b) and _isnum(a)) else None
        improved = None
        if better is not None and _isnum(b) and _isnum(a):
            improved = (a > b) if better == "higher" else (a < b)
        rows.append({"metric": key, "before": b, "after": a, "delta": delta,
                     "better": better, "improved": improved})
    patch = rem.get("patch")
    tkey = "area_ratio." + str(patch)
    tb, ta = _metric(mb, tkey), _metric(ma, tkey)
    target = {"metric": tkey, "before": tb, "after": ta,
              "improved": bool(_isnum(tb) and _isnum(ta) and ta > tb)}
    fb = {f.get("id"): f for f in before.get("findings", [])}
    fa = {f.get("id"): f for f in after.get("findings", [])}
    changes, regressions = [], []
    for fid in FINDING_ORDER:
        s0 = (fb.get(fid) or {}).get("status")
        s1 = (fa.get(fid) or {}).get("status")
        if s0 is not None and s1 is not None and s0 != s1:
            changes.append({"id": fid, "before": s0, "after": s1})
            if _SEVERITY.get(s1, 2) > _SEVERITY.get(s0, 2):
                regressions.append(fid)
    aft_fails = [f.get("id") for f in after.get("findings", [])
                 if f.get("status") == "fail"]
    verdict = "improved" if (target["improved"] and not aft_fails) else "not improved"
    say = [f"After one automatic iteration the {patch} surface capture went from"
           f" {_fnum(tb) * 100:.0f} to {_fnum(ta) * 100:.0f} percent."]
    for ch in changes:
        say.append(f"{ch['id']} went from {ch['before']} to {ch['after']}.")
    if not target["improved"]:
        say.append("The fix did not help: the targeted check did not improve.")
    elif aft_fails:
        say.append(f"The fix did not help: {', '.join(aft_fails)} now fail.")
    else:
        say.append("The fix worked: the targeted check improved and nothing failed.")

    def side(card):
        return {"label": card.get("label"), "mesh_dir": card.get("mesh_dir"),
                "solve_dir": card.get("solve_dir"), "reduced": card.get("reduced")}

    return {"tool": TOOL, "version": VERSION, "credit": CREDIT,
            "before": side(before), "after": side(after), "remedy": rem,
            "rows": rows, "target": target, "status_changes": changes,
            "regressions": regressions, "verdict": verdict, "say": say,
            "honesty": HONESTY}


def _cell(v):
    """One markdown table cell: floats as .4g, lists joined by ', '."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, (int, float)):
        return format(v, ".4g")
    if isinstance(v, (list, tuple)):
        return ", ".join(_cell(x) for x in v)
    if isinstance(v, dict):
        return ", ".join(str(k) + ": " + _cell(v[k]) for k in sorted(v, key=str))
    return str(v)


def card_markdown(card: dict) -> str:
    """The review card as markdown (R9)."""
    lines = ["# Review card: " + str(card.get("label"))]
    red = card.get("reduced") or {}
    if red.get("mesh") or red.get("solve"):
        lines.append("REDUCED (mesh/solve)")
    lines.append("Mesh: " + str(card.get("mesh_dir")))
    lines.append("Solve: " + str(card.get("solve_dir")))
    lines.append("")
    lines.append("| Check | Status | Value | Limit | Detail |")
    lines.append("|---|---|---|---|---|")
    for f in card.get("findings", []):
        lines.append("| " + " | ".join([str(f.get("id")), str(f.get("status")),
                                        _cell(f.get("value")), _cell(f.get("limit")),
                                        _cell(f.get("detail"))]) + " |")
    lines.append("")
    lines.append("## What the review says")
    lines.extend("- " + s for s in card.get("say", []))
    lines.append("")
    lines.append("## Remedy")
    rem = card.get("remedy") or {}
    lines.append(str(rem.get("id")) + ": " + str(rem.get("say")))
    lines.append("")
    lines.append(str(card.get("honesty") or HONESTY))
    lines.append("Geometry: " + CREDIT + ", no endorsement implied.")
    return chr(10).join(lines) + chr(10)


def compare_markdown(cmp: dict) -> str:
    """The before/after comparison as markdown (R9)."""
    b, a = cmp.get("before") or {}, cmp.get("after") or {}
    lines = ["# Before and after: " + str(b.get("label")) + " -> " + str(a.get("label"))]
    lines.append("")
    lines.append(str((cmp.get("remedy") or {}).get("say")))
    lines.append("")
    lines.append("| Metric | Before | After | Delta | Better |")
    lines.append("|---|---|---|---|---|")
    for r in cmp.get("rows", []):
        lines.append("| " + " | ".join([str(r.get("metric")), _cell(r.get("before")),
                                        _cell(r.get("after")), _cell(r.get("delta")),
                                        str(r.get("better") or "-")]) + " |")
    tgt = cmp.get("target") or {}
    lines.append("")
    lines.append("Target: " + str(tgt.get("metric")) + " " + _cell(tgt.get("before"))
                 + " -> " + _cell(tgt.get("after"))
                 + (", improved" if tgt.get("improved") else ", not improved"))
    lines.append("")
    lines.append("## What the review says")
    lines.extend("- " + s for s in cmp.get("say", []))
    lines.append("")
    lines.append("Verdict: " + str(cmp.get("verdict")))
    lines.append(str(cmp.get("honesty") or HONESTY))
    lines.append("Geometry: " + CREDIT + ", no endorsement implied.")
    return chr(10).join(lines) + chr(10)


# --- the fixtures: exact values from the supervisor's independent oracle ---

def _tun():
    return {"variant": "forward", "reduced": False,
            "domain": {"extent": [-16.2, 8.2, -5.2, 5.2, -5.0, 4.8]},
            "flow": {"velocity_m_s": [-15.0, 0.0, 0.0]},
            "check": {"parsed": {"n_cells": 1286601, "max_non_orth_deg": 65.0,
                                 "n_over_report": 23, "n_regions": 1, "passed": True}},
            "snap": {"within_tolerance_frac": 0.9971800686086401,
                     "area_ratio": [{"name": "body", "ratio": 0.613687495640983},
                                    {"name": "arms", "ratio": 0.5884430759695871},
                                    {"name": "motors", "ratio": 0.7786442765964805},
                                    {"name": "skids", "ratio": 0.841915899187408}]},
            "layers": {"claim": "none", "patches_with_layers": []}}


def _cfg():
    return {"domain": {"base_size": 0.2},
            "refinement": {"levels": [{"patch": p,
                                       "bands": [{"distance": 0.006, "level": 6},
                                                 {"distance": 0.02, "level": 5},
                                                 {"distance": 0.05, "level": 4}]}
                                      for p in WALLS],
                           "max_level": 6,
                           "boxes": [{"min": [0, 0, 0], "max": [1, 1, 1], "level": 3}]},
            "snap": {"feature_tolerance": 0.0078125},
            "output": {"case_dir": "x", "name": "drone_forward"}}


def _cfg5():
    """Exactly what tunnel_mesh --max-level 5 writes: refinement.max_level
    drops to 5 while the bands stay 6/5/4 - the mesher caps the 6 bands at 5 -
    and the snap feature tolerance follows the half-a-finest-cell rule."""
    cfg = _cfg()
    cfg["refinement"]["max_level"] = 5
    cfg["snap"]["feature_tolerance"] = 0.5 * 2.0 ** -5   # 0.015625
    return cfg


def _solve():
    return {"class": "unsteady",
            "failed": ["U_decades", "Cl_rel_change", "Cd_rel_change"],
            "criteria": {"U_decades": {"value": 2.500915855252043},
                         "p_decades": {"value": 4.540708793603613},
                         "cont_err": {"value": 2.82259e-09},
                         "Cl_rel_change": {"value": 0.04934106094893092},
                         "Cd_rel_change": {"value": 0.0007938383249786806}},
            "params": {"end_time": 0.5, "rho": 1.2041},
            "run": {"n_steps": 1000, "solver_seconds": 806.981},
            "final_forces": {"drag_N": 3.03986, "lift_N": 0.176511,
                             "drag_area_m2": 0.02244}}


def _log(with_yplus=True):
    """The solver's own lines: four y+ rows, an unrelated line, the two mass
    lines verbatim."""
    lines = []
    if with_yplus:
        for name, lo, me, hi in (("body", 0.963897, 57.7303, 221.511),
                                 ("arms", 0.865745, 47.4249, 198.803),
                                 ("motors", 1.65587, 60.4847, 281.91),
                                 ("skids", 0.0264991, 28.7891, 134.205)):
            lines.append("  y+ at " + name + ": min " + str(lo)
                         + " | mean " + str(me) + " | max " + str(hi))
    lines.append("=== an unrelated line ===")
    lines.append("discrete mass-flux divergence Sum_f (rho phi)_f, per cell: ")
    lines.append("                   net = -8.56074e-08 kg/s | L1 = 2.63852e-06 kg/s"
                 " | worst cell = 3.39868e-09 kg/s")
    return chr(10).join(lines)


def _mass_line(net="-8.56074e-08"):
    return ("                   net = " + net + " kg/s | L1 = 2.63852e-06 kg/s"
            " | worst cell = 3.39868e-09 kg/s")


def _refused(fn, *args, code):
    """Run fn(*args) with stdout captured; a refusal is SystemExit 2 and the
    line "refused: <code>"."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            fn(*args)
    except SystemExit as exc:
        if exc.code != 2:
            raise AssertionError("exit code " + repr(exc.code))
    else:
        raise AssertionError("no SystemExit at all")
    if "refused: " + code not in buf.getvalue():
        raise AssertionError(buf.getvalue())


def _t1():
    yp = parse_yplus(_log())
    if sorted(yp) != sorted(WALLS):
        raise AssertionError("(a) " + repr(sorted(yp)))
    if yp["arms"]["mean"] != 47.4249:
        raise AssertionError("(b) " + repr(yp["arms"]))
    if yp["skids"]["min"] != 0.0264991:
        raise AssertionError("(c) " + repr(yp["skids"]))
    if parse_yplus("") != {}:
        raise AssertionError("(d)")
    two = ("y+ at body: min 1.0 | mean 2.0 | max 3.0" + chr(10)
           + "y+ at body: min 4.0 | mean 5.0 | max 6.0")
    if parse_yplus(two)["body"]["min"] != 4.0:
        raise AssertionError("(e) last occurrence must win")


def _t2():
    m = parse_mass(_log())
    if m != {"net_kg_s": -8.56074e-08, "l1_kg_s": 2.63852e-06,
             "worst_kg_s": 3.39868e-09}:
        raise AssertionError("(a) " + repr(m))
    if parse_mass("no balance in here at all") is not None:
        raise AssertionError("(b)")


def _t3():
    q = inlet_mass_flow(_tun())
    if q is None or abs(q - 1840.8280800000002) > 1840.8280800000002 * 1e-12:
        raise AssertionError("(a) " + repr(q))
    rel = abs(-8.56074e-08) / q
    if abs(rel - 4.6504831673362996e-11) > rel * 1e-9:
        raise AssertionError("(b) " + repr(rel))


def _t4():
    fs = review_mesh(_tun())
    if [f["status"] for f in fs] != ["ok", "warn", "ok", "warn", "warn"]:
        raise AssertionError("(a) " + repr([f["status"] for f in fs]))
    if fs[3]["value"] != 0.5884430759695871 or fs[3]["detail"] != "arms":
        raise AssertionError("(b) " + repr(fs[3]))
    want = ["The mesh passes its own check: 1.29 million cells in one connected region.",
            ("The worst face is skewed 65 degrees, and 23 faces pass 60 degrees:"
             " the pressure solve works harder there."),
            "99.7 percent of the surface points snapped onto the geometry.",
            ("The arms surface is under-resolved: the mesh captures only"
             " 59 percent of it."),
            "No prism layers on the walls yet: wall functions carry the boundary layer."]
    says = [f["say"] for f in fs]
    if says != want:
        raise AssertionError("(c) " + repr(says))
    t = copy.deepcopy(_tun())
    t["check"]["parsed"]["max_non_orth_deg"] = 70.0
    if review_mesh(t)[1]["status"] != "fail":
        raise AssertionError("(d)")
    t = copy.deepcopy(_tun())
    t["check"]["parsed"]["n_over_report"] = 0
    if review_mesh(t)[1]["status"] != "ok":
        raise AssertionError("(e)")
    t = copy.deepcopy(_tun())
    for r in t["snap"]["area_ratio"]:
        r["ratio"] = 0.9
    if review_mesh(t)[3]["status"] != "ok":
        raise AssertionError("(f)")
    t = copy.deepcopy(_tun())
    t["check"]["parsed"]["passed"] = False
    if review_mesh(t)[0]["status"] != "fail":
        raise AssertionError("(g)")


def _t5():
    fs = review_solve(_solve(), _log(), _tun())
    if [f["id"] for f in fs] != ["RV-YPLUS", "RV-RESID", "RV-MASS"]:
        raise AssertionError("(a) " + repr([f["id"] for f in fs]))
    if fs[0]["status"] != "warn" or fs[0]["value"] != ["skids"]:
        raise AssertionError("(b) " + repr(fs[0]))
    if fs[0]["detail"] != {"skids": 28.7891}:
        raise AssertionError("(b2) " + repr(fs[0]["detail"]))
    if fs[0]["say"] != ("Wall y-plus leaves the wall-function band on the skids"
                        " surface: mean 29."):
        raise AssertionError("(c) " + fs[0]["say"])
    if fs[1]["status"] != "warn":
        raise AssertionError("(d0) " + fs[1]["status"])
    if fs[1]["say"] != ("The run is unsteady: drag still moves 0.08 percent and"
                        " lift 4.9 percent over the last window."):
        raise AssertionError("(d) " + fs[1]["say"])
    if fs[2]["status"] != "ok" or fs[2]["say"] != "Mass balance closes to 4.7e-11 of the inflow.":
        raise AssertionError("(e) " + repr(fs[2]))
    fs2 = review_solve(_solve(), _log(with_yplus=False), _tun())
    if fs2[0]["status"] != "fail" or fs2[0]["say"] != "No wall y-plus was reported.":
        raise AssertionError("(f) " + repr(fs2[0]))
    lines = ["  y+ at " + n + ": min 1.0 | mean " + str(me) + " | max 100.0"
             for n, me in (("body", 57.7303), ("arms", 47.4249),
                           ("motors", 60.4847), ("skids", 31.0))]
    lines.append(_mass_line())
    fs3 = review_solve(_solve(), chr(10).join(lines), _tun())
    if fs3[0]["status"] != "ok":
        raise AssertionError("(g) " + fs3[0]["say"])
    if review_solve(_solve(), _mass_line("0.01"), _tun())[2]["status"] != "fail":
        raise AssertionError("(h)")
    sv = copy.deepcopy(_solve())
    sv["class"] = "diverged"
    f4 = review_solve(sv, _log(), _tun())[1]
    if f4["status"] != "fail" or f4["say"] != "The run ended diverged.":
        raise AssertionError("(i) " + repr(f4))


def _t6():
    card = card_from("full", "m", "s", _tun(), _cfg(), _solve(), _log())
    c = card["counts"]
    if (c["ok"], c["warn"], c["fail"]) != (3, 5, 0):
        raise AssertionError("(a) " + repr(c))
    rem = card["remedy"]
    if rem["id"] != "REM-EXTEND":
        raise AssertionError("(b) " + rem["id"])
    if rem.get("end_time_from") != 0.5 or rem.get("end_time_to") != 1.0:
        raise AssertionError("(b2) " + repr(rem))
    if rem["say"] != "Remedy: run twice as long, to 1 seconds, and review again.":
        raise AssertionError("(b3) " + rem["say"])
    cons = rem["considered"]
    if [(r["finding"], r["remedy"], r["applicable"]) for r in cons] != [
            ("RV-AREA", "REM-WALL-LEVEL", False),
            ("RV-YPLUS", "REM-FIRST-LAYER", False),
            ("RV-RESID", "REM-EXTEND", True)]:
        raise AssertionError("(d) " + repr(cons))
    if cons[0]["reason"] != ("every surface under 80 percent capture is already"
                             " at the octree's level cap 6 (SPEC-LIT §74.2)"):
        raise AssertionError("(d2) " + repr(cons[0]["reason"]))
    if card["say"][0] != "Review of the full run: 3 checks pass, 5 warnings, 0 failures.":
        raise AssertionError("(f) " + card["say"][0])
    if len(card["say"]) != 10:
        raise AssertionError("(g) " + repr(len(card["say"])))
    card5 = card_from("reduced", "m", "s", _tun(), _cfg5(), _solve(), _log())
    r5 = card5["remedy"]
    if r5["id"] != "REM-WALL-LEVEL":
        raise AssertionError("(h) " + r5["id"])
    for k, v in (("patch", "arms"), ("patches", ["body", "arms", "motors"]),
                 ("band_distance_m", 0.006), ("level_from", 5),
                 ("level_to", 6), ("max_level_from", 5), ("max_level_to", 6),
                 ("cell_from_m", 0.00625), ("cell_to_m", 0.003125),
                 ("feature_tolerance_from", 0.015625),
                 ("feature_tolerance_to", 0.0078125), ("base_size_m", 0.2)):
        if r5.get(k) != v:
            raise AssertionError("(i) " + k + " " + repr(r5.get(k)))
    if r5["bands"] != [{"patch": p, "band_distance_m": 0.006, "level_from": 5,
                        "level_to": 6} for p in ("body", "arms", "motors")]:
        raise AssertionError("(i2) " + repr(r5["bands"]))
    if r5["say"] != ("Remedy: refine every surface under 80 percent capture one"
                     " level (body, arms and motors): the arms from 6.25 to"
                     " 3.125 millimetre cells, then re-mesh and re-solve."):
        raise AssertionError("(j) " + r5["say"])
    if [(r["finding"], r["remedy"], r["applicable"]) for r in r5["considered"]] != [
            ("RV-AREA", "REM-WALL-LEVEL", True),
            ("RV-YPLUS", "REM-FIRST-LAYER", False),
            ("RV-RESID", "REM-EXTEND", True)]:
        raise AssertionError("(k) " + repr(r5["considered"]))
    t = copy.deepcopy(_tun())
    for r in t["snap"]["area_ratio"]:
        r["ratio"] = 0.9
    rem2 = card_from("full", "m", "s", t, _cfg(), _solve(), _log())["remedy"]
    if rem2["id"] != "REM-EXTEND" or rem2["end_time_to"] != 1.0:
        raise AssertionError("(l) " + repr(rem2))
    t = copy.deepcopy(_tun())
    t["check"]["parsed"]["passed"] = False
    if card_from("full", "m", "s", t, _cfg(), _solve(), _log())["remedy"]["id"] != "REM-STOP":
        raise AssertionError("(m)")
    t = copy.deepcopy(_tun())
    for r in t["snap"]["area_ratio"]:
        r["ratio"] = 0.9
    sv = copy.deepcopy(_solve())
    sv["class"] = "steady"
    lines = ["  y+ at " + n + ": min 1.0 | mean " + str(me) + " | max 100.0"
             for n, me in (("body", 57.7303), ("arms", 47.4249),
                           ("motors", 60.4847), ("skids", 31.0))]
    lines.append(_mass_line())
    card3 = card_from("full", "m", "s", t, _cfg(), sv, chr(10).join(lines))
    if card3["remedy"]["id"] != "REM-NONE":
        raise AssertionError("(n) " + card3["remedy"]["id"])


def _t8_pair():
    """The T8 before/after pair, on CFG5: arms ratio 0.75, 1450000 cells, the
    remedy carried on the after tunnel report."""
    before = card_from("full", "m", "s", _tun(), _cfg5(), _solve(), _log())
    rem = before["remedy"]
    t2 = copy.deepcopy(_tun())
    for r in t2["snap"]["area_ratio"]:
        if r["name"] == "arms":
            r["ratio"] = 0.75
    t2["check"]["parsed"]["n_cells"] = 1450000
    t2["remedy"] = rem
    cfg2 = apply_remedy(_cfg5(), rem)
    after = card_from("after", "m2", "s2", t2, cfg2, _solve(), _log())
    return before, after, cfg2, rem


def _t7():
    rem = _t8_pair()[3]
    base = _cfg5()
    cfg2 = apply_remedy(base, rem)
    ref = cfg2["refinement"]
    arms = next(e for e in ref["levels"] if e["patch"] == "arms")
    b0 = next(b for b in arms["bands"] if b["distance"] == 0.006)
    if b0["level"] != 6 or ref["max_level"] != 6:
        raise AssertionError("(a) " + repr(b0) + " max " + repr(ref["max_level"]))
    lv = {(e["patch"], b["distance"]): b["level"]
          for e in ref["levels"] for b in e["bands"]}
    want = {(p, 0.006): (6 if p in ("body", "arms", "motors") else 5)
            for p in WALLS}
    want.update({(p, d): l for p in WALLS for d, l in ((0.02, 5), (0.05, 4))})
    if lv != want:
        raise AssertionError("(a2) " + repr(lv))
    if [bx["level"] for bx in ref["boxes"]] != [3]:
        raise AssertionError("(a3) " + repr(ref["boxes"]))
    if cfg2.get("snap", {}).get("feature_tolerance") != 0.0078125:
        raise AssertionError("(a4) " + repr(cfg2.get("snap")))
    put = copy.deepcopy(cfg2)
    for e in put["refinement"]["levels"]:
        next(b for b in e["bands"] if b["distance"] == 0.006)["level"] = 6
    put["refinement"]["max_level"] = 5
    put["snap"]["feature_tolerance"] = 0.015625
    if put != _cfg5():
        raise AssertionError("(b) the copy differs from CFG5 beyond the remedy")
    if base != _cfg5():
        raise AssertionError("(c) CFG5 itself changed")
    if base["refinement"]["max_level"] != 5:
        raise AssertionError("(d) CFG5 itself changed")
    _refused(apply_remedy, _cfg5(),
             dict(rem, bands=[dict(rem["bands"][0], patch="rotor")]), code="DR-REMEDY")
    _refused(apply_remedy, _cfg5(),
             dict(rem, bands=[dict(rem["bands"][0], level_from=4)]), code="DR-REMEDY")
    _refused(apply_remedy, _cfg5(), dict(rem, id="REM-EXTEND"), code="DR-REMEDY")
    _refused(apply_remedy, _cfg5(), dict(rem, bands=[]), code="DR-REMEDY")
    _refused(apply_remedy, _cfg5(),
             dict(rem, bands=[dict(rem["bands"][0], level_to=7)]), code="DR-REMEDY")


def _t8():
    before, after, cfg2, rem = _t8_pair()
    c = compare_cards(before, after)
    tgt = c["target"]
    if (tgt["metric"] != "area_ratio.arms" or tgt["before"] != 0.5884430759695871
            or tgt["after"] != 0.75 or tgt["improved"] is not True):
        raise AssertionError("(a) " + repr(tgt))
    if c["verdict"] != "improved" or c["status_changes"] != []:
        raise AssertionError("(b) " + repr(c["verdict"]) + " " + repr(c["status_changes"]))
    if len(c["rows"]) != 17:
        raise AssertionError("(c) " + repr(len(c["rows"])))
    if c["rows"][0]["metric"] != "n_cells" or c["rows"][0]["delta"] != 163399:
        raise AssertionError("(d) " + repr(c["rows"][0]))
    if c["say"][0] != ("After one automatic iteration the arms surface capture"
                       " went from 59 to 75 percent."):
        raise AssertionError("(e) " + c["say"][0])
    t3 = copy.deepcopy(after["metrics"])
    t3["area_ratio"]["arms"] = 0.55
    a3 = copy.deepcopy(after)
    a3["metrics"] = t3
    c2 = compare_cards(before, a3)
    if c2["verdict"] != "not improved" or c2["say"][-1] != ("The fix did not help:"
                                                           " the targeted check did not improve."):
        raise AssertionError("(f) " + repr(c2["verdict"]) + " " + repr(c2["say"][-1]))
    t4 = copy.deepcopy(_tun())
    for r in t4["snap"]["area_ratio"]:
        if r["name"] == "arms":
            r["ratio"] = 0.75
    t4["check"]["parsed"]["n_cells"] = 1450000
    t4["check"]["parsed"]["max_non_orth_deg"] = 70.5
    t4["remedy"] = rem
    a4 = card_from("after", "m2", "s2", t4, cfg2, _solve(), _log())
    c3 = compare_cards(before, a4)
    if c3["regressions"] != ["RV-NONORTH"] or c3["verdict"] != "not improved":
        raise AssertionError("(g) " + repr(c3["regressions"]) + " " + c3["verdict"])
    if c3["say"][-1] != "The fix did not help: RV-NONORTH now fail.":
        raise AssertionError("(h) " + c3["say"][-1])
    a5 = copy.deepcopy(after)
    a5.pop("mesh_remedy", None)
    _refused(compare_cards, before, a5, code="DR-INPUT")


def _t9():
    before, after, cfg2, rem = _t8_pair()
    card = card_from("full", "m", "s", _tun(), _cfg(), _solve(), _log())
    md = card_markdown(card)
    for needle in ("| RV-AREA | warn |", "## What the review says", HONESTY, CREDIT):
        if needle not in md:
            raise AssertionError("(a) missing: " + needle)
    if "REDUCED" in md:
        raise AssertionError("(b)")
    sv = copy.deepcopy(_solve())
    sv["params"]["end_time"] = 0.25
    md2 = card_markdown(card_from("full", "m", "s", _tun(), _cfg(), sv, _log()))
    if "REDUCED" not in md2:
        raise AssertionError("(c)")
    md3 = compare_markdown(compare_cards(before, after))
    for needle in ("Verdict: improved", "| area_ratio.arms |", HONESTY, CREDIT):
        if needle not in md3:
            raise AssertionError("(d) missing: " + needle)


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)


def _t10():
    tmp = tempfile.mkdtemp()
    empty = os.path.join(tmp, "empty")
    os.makedirs(empty)
    _refused(main, ["review", "--label", "x", "--mesh", empty, "--solve", empty,
                    "--out", os.path.join(_REPO, "tools", "drone",
                                          "review_selftest_out.json")],
             code="DR-OUT")
    _refused(main, ["review", "--label", "x", "--mesh", empty, "--solve", empty,
                    "--out", os.path.join(tmp, "c.json")], code="DR-INPUT")
    d = os.path.join(tmp, "hover")
    os.makedirs(d)
    _write_json(os.path.join(d, "tunnel_mesh.json"), {"variant": "hover"})
    _write_json(os.path.join(d, "drone_tunnel.json"), {})
    _write_json(os.path.join(d, "solve.json"), {})
    with open(os.path.join(d, "solve.log"), "w", encoding="utf-8") as fh:
        fh.write(_log())
    _refused(main, ["review", "--label", "x", "--mesh", d, "--solve", d,
                    "--out", os.path.join(tmp, "h.json")], code="DR-VARIANT")
    base = os.path.join(tmp, "base")
    os.makedirs(base)
    _write_json(os.path.join(base, "drone_tunnel.json"), _cfg5())
    _write_json(os.path.join(base, "tunnel_mesh.json"), _tun())
    for nm in ("rotors.json", "LICENSE.txt"):
        with open(os.path.join(base, nm), "w", encoding="utf-8") as fh:
            fh.write("x" + chr(10))
    cardp = os.path.join(tmp, "card_ext.json")
    _write_json(cardp, {"remedy": {"id": "REM-EXTEND"}})
    _refused(main, ["improve", "--card", cardp, "--base-mesh", base,
                    "--out", os.path.join(tmp, "improve_out")], code="DR-REMEDY")
    card_ok = card_from("full", "m", "s", _tun(), _cfg5(), _solve(), _log())
    cardp2 = os.path.join(tmp, "card_ok.json")
    _write_json(cardp2, card_ok)
    _refused(main, ["improve", "--card", cardp2, "--base-mesh", base,
                    "--out", os.path.join(tmp, "improve_out2"),
                    "--binary", os.path.join(tmp, "no_such_binary.exe")],
             code="DR-BINARY")


def _t11():
    tmp = tempfile.mkdtemp()
    m = os.path.join(tmp, "mesh")
    s = os.path.join(tmp, "solve")
    os.makedirs(m)
    os.makedirs(s)
    _write_json(os.path.join(m, "tunnel_mesh.json"), _tun())
    _write_json(os.path.join(m, "drone_tunnel.json"), _cfg5())
    _write_json(os.path.join(s, "solve.json"), _solve())
    with open(os.path.join(s, "solve.log"), "w", encoding="utf-8") as fh:
        fh.write(_log())
    outp = os.path.join(tmp, "card.json")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = main(["review", "--label", "full", "--mesh", m, "--solve", s,
                   "--out", outp])
    if rc != 0:
        raise AssertionError("(a) rc " + repr(rc))
    if not os.path.isfile(outp) or not os.path.isfile(os.path.join(tmp, "card.md")):
        raise AssertionError("(b)")
    card = _read_json(outp)
    c = card["counts"]
    if (c["ok"], c["warn"], c["fail"]) != (3, 5, 0):
        raise AssertionError("(c) " + repr(c))
    lines = [l for l in buf.getvalue().splitlines() if l.strip()]
    if lines[-1] != "DRONE-REVIEW CARD full: 3 ok, 5 warn, 0 fail; remedy REM-WALL-LEVEL":
        raise AssertionError("(d) " + repr(lines[-1]))


def _t12():
    """Arms band at 4 with max_level 6: the effective level 4 sits below the
    cap, so the remedy refines arms alone - body and motors already sit at the
    cap - and max_level and the feature tolerance stay put."""
    cfg = _cfg()
    arms = next(e for e in cfg["refinement"]["levels"] if e["patch"] == "arms")
    next(b for b in arms["bands"] if b["distance"] == 0.006)["level"] = 4
    rem = card_from("full", "m", "s", _tun(), cfg, _solve(), _log())["remedy"]
    if rem["id"] != "REM-WALL-LEVEL":
        raise AssertionError("(a) " + rem["id"])
    for k, v in (("patch", "arms"), ("patches", ["arms"]), ("band_distance_m", 0.006),
                 ("level_from", 4), ("level_to", 5), ("max_level_from", 6),
                 ("max_level_to", 6), ("cell_from_m", 0.0125), ("cell_to_m", 0.00625),
                 ("feature_tolerance_from", 0.0078125),
                 ("feature_tolerance_to", 0.0078125)):
        if rem.get(k) != v:
            raise AssertionError("(b) " + k + " " + repr(rem.get(k)))
    if rem["bands"] != [{"patch": "arms", "band_distance_m": 0.006,
                         "level_from": 4, "level_to": 5}]:
        raise AssertionError("(b2) " + repr(rem["bands"]))
    if rem["say"] != ("Remedy: refine every surface under 80 percent capture one"
                      " level (arms): the arms from 12.5 to 6.25 millimetre"
                      " cells, then re-mesh and re-solve."):
        raise AssertionError("(c) " + rem["say"])


def selftest() -> int:
    """T1-T12, no mesher and no GPU; temp files only under tempfile.mkdtemp()."""
    tests = [("T1", _t1), ("T2", _t2), ("T3", _t3), ("T4", _t4), ("T5", _t5),
             ("T6", _t6), ("T7", _t7), ("T8", _t8), ("T9", _t9), ("T10", _t10),
             ("T11", _t11), ("T12", _t12)]
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


def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (ValueError, OSError) as exc:
        PROMO.refuse("DR-INPUT", PROMO._posix(path) + ": unreadable (" + str(exc) + ")")


def build_card(label, mesh_dir, solve_dir) -> dict:
    """Read the four files (DR-INPUT names the missing one, DR-VARIANT off
    forward) and hand card_from the pieces."""
    mesh_dir = os.path.abspath(mesh_dir)
    solve_dir = os.path.abspath(solve_dir)
    tm_path = os.path.join(mesh_dir, "tunnel_mesh.json")
    cfg_path = os.path.join(mesh_dir, "drone_tunnel.json")
    sv_path = os.path.join(solve_dir, "solve.json")
    lg_path = os.path.join(solve_dir, "solve.log")
    for pth in (tm_path, cfg_path, sv_path, lg_path):
        if not os.path.isfile(pth):
            PROMO.refuse("DR-INPUT", PROMO._posix(pth) + ": not found")
    tunnel = _read_json(tm_path)
    if tunnel.get("variant") != "forward":
        PROMO.refuse("DR-VARIANT", PROMO._posix(tm_path) + ": variant "
                     + repr(tunnel.get("variant"))
                     + " - review reads the forward variant only")
    cfg = _read_json(cfg_path)
    solve = _read_json(sv_path)
    with open(lg_path, "r", encoding="utf-8", errors="replace") as fh:
        log_text = fh.read()
    return card_from(label, mesh_dir, solve_dir, tunnel, cfg, solve, log_text)


def cmd_review(a) -> int:
    out = os.path.abspath(a.out)
    if _in_repo(out):
        PROMO.refuse("DR-OUT", PROMO._posix(out)
                     + ": --out is the repository root or inside it")
    card = build_card(a.label, a.mesh, a.solve)
    parent = os.path.dirname(out)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(card, fh, indent=1)
    md_path = os.path.splitext(out)[0] + ".md"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(card_markdown(card))
    for s in card["say"]:
        print("[drone-review] " + s)
    c = card["counts"]
    print("DRONE-REVIEW CARD " + str(card["label"]) + ": " + str(c["ok"]) + " ok, "
          + str(c["warn"]) + " warn, " + str(c["fail"]) + " fail; remedy "
          + str(card["remedy"]["id"]))
    return 0


def cmd_improve(a) -> int:
    out_dir = os.path.abspath(a.out)
    if _in_repo(out_dir):
        PROMO.refuse("DR-OUT", PROMO._posix(out_dir)
                     + ": --out is the repository root or inside it")
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        PROMO.refuse("DR-OUT", PROMO._posix(out_dir) + ": exists and is not empty")
    if os.path.exists(out_dir) and not os.path.isdir(out_dir):
        PROMO.refuse("DR-OUT", PROMO._posix(out_dir) + ": exists and is not a directory")
    if not os.path.isfile(a.card):
        PROMO.refuse("DR-INPUT", PROMO._posix(a.card) + ": not found")
    card = _read_json(a.card)
    base_files = [os.path.join(a.base_mesh, n) for n in
                  ("drone_tunnel.json", "tunnel_mesh.json", "rotors.json", "LICENSE.txt")]
    for pth in base_files:
        if not os.path.isfile(pth):
            PROMO.refuse("DR-INPUT", PROMO._posix(pth) + ": not found")
    base_cfg = _read_json(base_files[0])
    base_report = _read_json(base_files[1])
    if base_report.get("variant") != "forward":
        PROMO.refuse("DR-VARIANT", PROMO._posix(base_files[1]) + ": variant "
                     + repr(base_report.get("variant"))
                     + " - improve re-meshes the forward variant only")
    rem = card.get("remedy")
    if not isinstance(rem, dict) or rem.get("id") != "REM-WALL-LEVEL":
        rid = rem.get("id") if isinstance(rem, dict) else None
        PROMO.refuse("DR-REMEDY", repr(rid) + " is not a mesh remedy; improve re-meshes only")
    cfg2 = apply_remedy(base_cfg, rem)
    binary = a.binary or DEFAULT_BINARY
    if not os.path.isfile(binary):
        PROMO.refuse("DR-BINARY", str(binary) + ": not found")
    os.makedirs(out_dir, exist_ok=True)
    cfg2.setdefault("output", {})["case_dir"] = PROMO._posix(os.path.join(out_dir, "case"))
    cfg_path = os.path.join(out_dir, "drone_tunnel.json")
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(cfg2, fh, indent=1)
    print("[drone-review] remedy " + str(rem.get("id")) + ": " + str(rem.get("say")))
    for spec in (rem.get("bands") or []):
        print("[drone-review] " + str(spec.get("patch")) + " band "
              + repr(spec.get("band_distance_m")) + " m level "
              + repr(spec.get("level_from")) + " -> " + repr(spec.get("level_to")))
    print("[drone-review] max_level " + repr(rem.get("max_level_from")) + " -> "
          + repr(rem.get("max_level_to")) + ", feature_tolerance "
          + repr(rem.get("feature_tolerance_from")) + " -> "
          + repr(rem.get("feature_tolerance_to")))
    print("[drone-review] config " + PROMO._posix(cfg_path))
    run_rc, wall_s, _ = PROMO.run_stream([binary, PROMO._posix(cfg_path)], out_dir,
                                         os.path.join(out_dir, "automesher.log"),
                                         timeout=a.timeout)
    name = (cfg2.get("output") or {}).get("name") or "drone_forward"
    sum_path = os.path.join(out_dir, "case", name + "_summary.json")
    summary = _read_json(sum_path) if os.path.isfile(sum_path) else None
    check_rc, _, chk_text = PROMO.run_stream([binary, PROMO._posix(cfg_path), "-check"],
                                             out_dir, os.path.join(out_dir, "check.log"),
                                             timeout=a.timeout)
    chk = PROMO.parse_check(chk_text)
    gate = TM.judge(run_rc, summary, check_rc, chk, base_report.get("reduced"), "forward")
    digest = hashlib.sha256()
    with open(binary, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    mesher = {"binary": PROMO._posix(os.path.abspath(binary)),
              "binary_sha256": digest.hexdigest()[:8], "wall_seconds": wall_s,
              "returncode": run_rc, "total_seconds": None, "stages": [],
              "n_cells": None, "patches": {}, "quality": None}
    if summary is not None:
        mesher["total_seconds"] = summary.get("total_seconds")
        mesher["stages"] = [{"stage": r.get("stage"), "seconds": r.get("seconds")}
                            for r in summary.get("stages", [])]
        mesher["n_cells"] = summary.get("mesh", {}).get("n_cells")
        mesher["patches"] = {r.get("name"): r.get("size")
                             for r in summary.get("mesh", {}).get("patches", [])}
        mesher["quality"] = summary.get("quality")
    report = copy.deepcopy(base_report)
    report["config_path"] = PROMO._posix(cfg_path)
    report["mesher"] = mesher
    report["snap"] = TM.snap_digest(summary)
    report["layers"] = TM.layer_coverage(summary)
    report["check"] = {"returncode": check_rc,
                       "quality_line": chk.get("quality_line"), "parsed": chk}
    report["gate"] = gate
    mlv = rem.get("max_level_to")
    report.setdefault("params", {})["max_level"] = mlv
    bs = (cfg2.get("domain") or {}).get("base_size")
    report.setdefault("domain", {})["finest_cell_m"] = (
        bs / 2 ** mlv if _isnum(bs) and isinstance(mlv, int) and not isinstance(mlv, bool) else None)
    report["remedy"] = rem
    report["base_mesh"] = PROMO._posix(os.path.abspath(a.base_mesh))
    report["remeshed_by"] = TOOL
    with open(os.path.join(out_dir, "tunnel_mesh.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    for nm in ("rotors.json", "LICENSE.txt", "LICENSE"):
        src = os.path.join(a.base_mesh, nm)
        if os.path.isfile(src):
            shutil.copyfile(src, os.path.join(out_dir, nm))
    if gate.get("pass"):
        print("DRONE-REVIEW REMESH GATE PASS")
        return 0
    print("DRONE-REVIEW REMESH GATE FAIL: " + "; ".join(gate.get("reasons", [])))
    return 1


def cmd_compare(a) -> int:
    for pth in (a.before, a.after):
        if not os.path.isfile(pth):
            PROMO.refuse("DR-INPUT", PROMO._posix(pth) + ": not found")
    before = _read_json(a.before)
    after = _read_json(a.after)
    cmpd = compare_cards(before, after)
    out = os.path.abspath(a.out)
    if _in_repo(out):
        PROMO.refuse("DR-OUT", PROMO._posix(out)
                     + ": --out is the repository root or inside it")
    parent = os.path.dirname(out)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(cmpd, fh, indent=1)
    with open(os.path.splitext(out)[0] + ".md", "w", encoding="utf-8") as fh:
        fh.write(compare_markdown(cmpd))
    for s in cmpd["say"]:
        print("[drone-review] " + s)
    tgt = cmpd["target"]
    b4, af = tgt.get("before"), tgt.get("after")
    vals = (f"{b4:.4f} -> {af:.4f}" if _isnum(b4) and _isnum(af)
            else repr(b4) + " -> " + repr(af))
    print("DRONE-REVIEW BEFORE/AFTER " + str(cmpd["verdict"]) + ": "
          + str(tgt.get("metric")) + " " + vals)
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--selftest" in argv:
        return selftest()
    p = argparse.ArgumentParser(
        prog="review.py",
        description="Review a drone run (mesh quality, y+, residual class, mass"
                    " balance) into a card, pick one remedy by a fixed rule"
                    " table, re-mesh with it, and compare the before/after cards.")
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("review", help="read a tunnel mesh and solve, write the review card")
    pr.add_argument("--label", required=True, help="run label, e.g. full")
    pr.add_argument("--mesh", required=True, help="directory of tunnel_mesh.json and drone_tunnel.json")
    pr.add_argument("--solve", required=True, help="directory of solve.json and solve.log")
    pr.add_argument("--out", required=True, help="card json path, outside the repository")
    pr.set_defaults(fn=cmd_review)
    pi = sub.add_parser("improve", help="re-mesh the base case with the card's remedy")
    pi.add_argument("--card", required=True, help="the review card json")
    pi.add_argument("--base-mesh", required=True, help="the base mesh directory")
    pi.add_argument("--out", required=True, help="output directory, outside the repository")
    pi.add_argument("--binary", default=DEFAULT_BINARY, help="the release automesher binary")
    pi.add_argument("--timeout", type=float, default=0.0,
                    help="seconds for the mesher subprocess, 0 = none (default 0)")
    pi.set_defaults(fn=cmd_improve)
    pc = sub.add_parser("compare", help="compare the before and after cards")
    pc.add_argument("--before", required=True, help="the before card json")
    pc.add_argument("--after", required=True, help="the after card json")
    pc.add_argument("--out", required=True, help="comparison json path, outside the repository")
    pc.set_defaults(fn=cmd_compare)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
