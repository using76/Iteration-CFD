#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""
remedies.py - the L2 remedies (docs/15 §C's L2 row, AM-10).

After a failed attempt, `diagnose` names the key and the earliest failing
stage from score.py's closed failure enum; `propose` walks a table of twelve
remedies in priority order, each firing at most MAX_FIRES times per geometry
and never revisiting a config sha, and ends the geometry in one of four
terminals: PASS, CAPABILITY-LIMITED, EXHAUSTED or NO-REMEDY. `loop` runs the
attempts to a terminal; `gate` writes the G-REMEDIES report.

Guards: every config write goes through `_set` (whitelisted leaf pointers and
the one levels container only), every edit is re-checked against the locked
knob table by `_commit`, and knobs of a later stage than the row's stay
frozen. No remedy ever writes the quality block, a limiter, or a command
line. The static scan re-reads this file's own AST: inside a `_rm_*` remedy
function no process may be spawned and no subprocess import may appear (the
gate's own git call and child runs live outside the remedy functions, where
the scan allows them).
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import schema  # noqa: E402
import score  # noqa: E402
import preflight  # noqa: E402
import rules  # noqa: E402

REPORT_DIR = os.path.join(HERE, "remedies")
FIXTURE_DIR = os.path.join(HERE, "fixtures", "remedies")
RESULT_SCHEMA = "autonomy-remedy/1"
LOOP_SCHEMA = "autonomy-remedy-loop/1"
GATE_SCHEMA = "autonomy-remedies-gate/1"
MAX_FIRES = 2                        # docs/15 §C L2: at most two fires per geometry
STAGES = score.STAGES
TERMINALS = ("PASS", "CAPABILITY-LIMITED", "EXHAUSTED", "NO-REMEDY")
TERMINAL_ID = {"PASS": "RM-PASS", "CAPABILITY-LIMITED": "RM-CAPABILITY-LIMITED",
               "EXHAUSTED": "RM-EXHAUSTED", "NO-REMEDY": "RM-NO-REMEDY"}
DROP_KEYS = ("layer:min_thickness", "layer:retreat_snapped")
NO_REMEDY_KEYS = ("surface_closed", "config", "io", "crash", "gate@octree", "gate@castellate",
                  "gate@split", "gate@layers", "F2")
CONTAINERS = ("/refinement/levels",)     # the one non-leaf pointer _set may write
LAYER_KINDS = ("min_thickness", "retreat_snapped", "no_full_stack")
SYNTHETIC_KINDS = ("pass", "F3a", "F3b", "F3c", "F3d", "F4", "F5", "F2", "min_thickness",
                   "retreat_snapped", "no_full_stack", "timeout", "crash", "io", "config",
                   "surface_closed", "gate_G4@castellate", "gate_G4@snap", "layer_t1_G5")


class RemedyError(ValueError):
    """A caller or harness error - never a verdict."""


REMEDIES = (
    {"id": "RM-BUDGET-FAR", "keys": ("F5", "timeout"), "stage": "octree",
     "fn": "_rm_budget_far", "what": "halve the far-field band distances",
     "formula": "d' = max(d/2, 3*base/2**level, d'_prev) for every band below the wall level",
     "cite": "docs/15 §C L2; rules.py R-BUDGET's ladder (far-field bands first); docs/15 §D.1 F5"},
    {"id": "RM-BUDGET-FEAT", "keys": ("F5", "timeout"), "stage": "octree",
     "fn": "_rm_budget_feat", "what": "drop the feature bump to the wall level",
     "formula": "feature_level' = wall_level where feature_level > wall_level; "
                "max_level' = the largest level left",
     "cite": "docs/15 §C L2; rules.py R-BUDGET's ladder (the feature bump); docs/15 §D.1 F5"},
    {"id": "RM-BUDGET-WALL", "keys": ("F5", "timeout"), "stage": "octree",
     "fn": "_rm_budget_wall",
     "what": "coarsen the whole refinement ladder one level, not below the y+ floor",
     "formula": "every band level and feature_level - 1, max_level - 1; "
                "wall' >= floor = the smallest L with base/2**L <= 3*t1/min_thickness_ratio",
     "cite": "docs/15 §C L2, §D.3 (y+ <= 1 needs h_wall <= 60 t1); "
             "rules.py R-BUDGET's ladder (the wall level last)"},
    {"id": "RM-TOPO-REFINE", "keys": ("F4",), "stage": "castellate",
     "fn": "_rm_topo_refine", "what": "refine the whole refinement ladder one level",
     "formula": "every band level + 1 (none past max_level 6), feature_level + 1 capped at 6; "
                "predicted cells <= 0.7*cell_budget",
     "cite": "docs/15 §C L2, §D.1 F4 (a wall patch absent or n_regions != 1); rules.predict_cells"},
    {"id": "RM-SNAP-WALL", "keys": ("F3", "gate@snap"), "stage": "snap",
     "fn": "_rm_snap_wall",
     "what": "coarsen the whole refinement ladder one level, not below the y+ floor",
     "formula": "as RM-BUDGET-WALL; never on the R-PLANE path",
     "cite": "docs/15 §K G-PILOT (wall level -1 was the only F3-clean knob with the attraction "
             "on: B-1-002, B-1-004); docs/15 §D.3"},
    {"id": "RM-SNAP-FT", "keys": ("F3", "gate@snap"), "stage": "snap",
     "fn": "_rm_snap_ft",
     "what": "switch the feature attraction off (snap.feature_tolerance = 0)",
     "formula": "feature_tolerance' = 0.0 when the body has sharp edges and the attraction is on",
     "cite": "docs/15 §K G-PILOT (feature_tolerance 0 unpinned 10 of 10 feature-bearing "
             "geometries; caution 1: the edges are then not captured, G-FID guards it)"},
    {"id": "RM-SNAP-REFINE", "keys": ("F3d",), "stage": "snap",
     "fn": "_rm_snap_refine",
     "what": "refine the whole refinement ladder one level (the snapped surface misses area: "
             "the lattice is too coarse for the body)",
     "formula": "as RM-TOPO-REFINE",
     "cite": "docs/15 §D.1 F3d (area ratio outside [0.98, 1.02]); rules.predict_cells"},
    {"id": "RM-T1-RAISE", "keys": ("layer_t1_G5",), "stage": "layers",
     "fn": "_rm_t1_raise", "what": "raise the first layer to the y+ = 1 bound",
     "formula": "t1' = t1_floor(y+ nu / u_tau) (R-YP) when t1 < t1'; growth kept while the "
                "stack fits cell_frac*h_wall, else the largest growth that fits",
     "cite": "docs/15 §D.3 (R-YP); layers.rs:1292-1316 (92.51): "
             "3*t1/h_min >= min_thickness_ratio"},
    {"id": "RM-T1-REFINE", "keys": ("layer_t1_G5",), "stage": "layers",
     "fn": "_rm_t1_refine",
     "what": "refine the whole refinement ladder one level so the wall edges shrink under 60 t1",
     "formula": "as RM-TOPO-REFINE, then the growth refit at the new h_wall",
     "cite": "layers.rs:1292-1316 (92.51); rules.predict_cells"},
    {"id": "RM-PLANE", "keys": ("layer:min_thickness", "layer:retreat_snapped"), "stage": "layers",
     "fn": "_rm_plane",
     "what": "put the body on cell planes with R-PLANE (rules.setup's config for this geometry "
             "and flow)",
     "formula": "domain, refinement, snap.feature_tolerance = 0, snap.smoothing_passes = 0 and "
                "the layer stack from rules.setup when its R-PLANE applies; every other key kept",
     "cite": "docs/15 §C L2 (a snapped-wall drop goes to R-PLANE if the geometry qualifies); "
             "SPEC-LIT §92.15.5; rules.py R-PLANE"},
    {"id": "RM-PLANE-FINER", "keys": ("layer:min_thickness", "layer:retreat_snapped"),
     "stage": "layers", "fn": "_rm_plane_finer",
     "what": "take the next lattice divisor h = s/(m+1), the extent re-aligned on the body's faces",
     "formula": "m' = round(s/h_wall) + 1; h' = s/m' >= n*t1/cell_frac; "
                "level' = round(log2(0.5 L_ref/h')); base' = h'*2**level'; "
                "growth' = the largest step fitting cell_frac*h'",
     "cite": "docs/15 §K G-RULES (the box-corner bound: 42.9 t1 delivers, 43.0 drops on cubep); "
             "rules.py R-PLANE"},
    {"id": "RM-LAYER-FIT", "keys": ("layer:no_full_stack",), "stage": "layers",
     "fn": "_rm_layer_fit", "what": "fit the stack under the cell_frac limiter",
     "formula": "growth' = the largest k/1000 with t1*S(g) <= cell_frac*h_wall; if none, "
                "t1' = t1_floor(cell_frac*h_wall/n) and growth 1.0, kept only while "
                "3*t1'/h_wall >= min_thickness_ratio",
     "cite": "docs/15 §D.3; layers.rs:440-450 (t_i = min(T, medial, cell_frac*h_i)); "
             "cell_frac is read, never written (docs/15 §I-5)"},
)


# --- reading a config (C4): _get is the ONLY reader of pointers ---------------

def _get(config, pointer):
    """The value at a JSON pointer, or the mesher's default when a key is missing."""
    node = config
    for seg in pointer.split("/")[1:]:
        if isinstance(node, dict) and seg in node:
            node = node[seg]
        else:
            return preflight.MESHER_DEFAULTS.get(pointer)
    return node


def _requested(config):
    return _get(config, "/layers/n") > 0 and bool(_get(config, "/layers/patches"))


def wall_level(config: dict) -> int:
    """The largest band level over every refinement entry; 0 when there is no band."""
    top = 0
    for e in _get(config, "/refinement/levels"):
        for b in e.get("bands") or []:
            top = max(top, b["level"])
    return top


def floor_level(config: dict, ctx: dict | None, knobs: dict) -> int | None:
    """The y+ floor: the smallest level whose h is under 60 t1; None when none fits."""
    if not _requested(config):
        return 1
    base = config["domain"]["base_size"]
    t1 = _get(config, "/layers/first_thickness")
    hi = rules.G5_FACTOR * t1 / rules.G5_RATIO * (1 - rules.EDGE_MARGIN)
    cap = schema._knob_row("/refinement/max_level", knobs)["max"]
    level, capped = rules.level_for(base, hi, cap)
    if capped:
        return None
    win = (ctx or {}).get("win_level")
    if isinstance(win, int) and not isinstance(win, bool):
        level = max(level, win)
    return level


def on_plane(config: dict, fingerprint: dict) -> bool:
    """True when the body already lies on the R-PLANE path (faces on cell planes)."""
    if fingerprint.get("commensurate") is not True:
        return False
    s = fingerprint.get("lattice_base_size_m")
    if not (isinstance(s, (int, float)) and not isinstance(s, bool) and s > 0):
        return False
    if _get(config, "/snap/feature_tolerance") != 0:
        return False
    if _get(config, "/snap/smoothing_passes") != 0:
        return False
    h = config["domain"]["base_size"] / 2 ** wall_level(config)
    q = s / h
    if abs(q - round(q)) > 1e-9 or round(q) < 1:
        return False
    bb = fingerprint["bbox"]
    ext = config["domain"]["extent"]
    for a in range(3):
        q = (bb[2 * a] - ext[2 * a]) / h
        if abs(q - round(q)) > 1e-9:
            return False
    return True


def _patches(ctx: dict) -> list:
    """The fingerprint's patch names, first-appearance order, no duplicates."""
    out = []
    for p in ctx["fingerprint"]["patches"]:
        if p["name"] not in out:
            out.append(p["name"])
    return out


def _forbidden(pointer: str, knobs: dict) -> bool:
    for f in knobs.get("forbidden", []):
        if pointer == f["pointer"] or \
                (f["match"] == "prefix" and pointer.startswith(f["pointer"] + "/")):
            return True
    return False


# --- writing a config (C5): _set is the ONLY writer; _commit re-checks --------

def _set(after, pointer, value):
    """Store one whitelisted value; RemedyError on anything the table may not write."""
    if pointer not in CONTAINERS:
        row = schema._knob_row(pointer, _knobs())
        if row is None or "*" in row["pointer"]:
            raise RemedyError("_set: %s is not a whitelisted leaf pointer" % pointer)
    node = after
    segs = pointer.split("/")[1:]
    for seg in segs[:-1]:
        if not isinstance(node.get(seg), dict):
            node[seg] = {}
        node = node[seg]
    node[segs[-1]] = copy.deepcopy(value)


def _commit(before, after, stage, knobs):
    """diff_edits plus every guard of docs/15 §C L2; the edits out, RemedyError otherwise."""
    edits = rules.diff_edits(before, after)
    if edits == []:
        raise RemedyError("_commit: the remedy changed nothing")
    si = STAGES.index(stage)
    for e in edits:
        p = e["pointer"]
        row = schema._knob_row(p, knobs)
        if e["to"] is None:
            if row is None or _forbidden(p, knobs):
                raise RemedyError("_commit: the removal of %s is not allowed" % p)
        else:
            bad = schema.check_edit(p, e["to"], knobs)
            if bad is not None:
                raise RemedyError("_commit: %s" % bad["message"])
        st = row["stage"] if row is not None else "octree"
        if STAGES.index(st) > si:
            raise RemedyError("_commit: %s (stage %s) is later than the row's stage %s"
                              % (p, st, stage))
    for k in ("input", "output", "quality"):
        if json.dumps(before.get(k), sort_keys=True) != json.dumps(after.get(k), sort_keys=True):
            raise RemedyError("_commit: the remedy wrote the %s block" % k)
    bl = dict(preflight.config_leaves(before))
    al = dict(preflight.config_leaves(after))
    for p, v in bl.items():
        if schema._knob_row(p, knobs) is None and p in al \
                and json.dumps(al[p], sort_keys=True) != json.dumps(v, sort_keys=True):
            raise RemedyError("_commit: the unlisted leaf %s changed" % p)
    return edits


_KNOBS = {"v": None}          # the knob table, loaded once (it is locked and read-only)


def _knobs() -> dict:
    if _KNOBS["v"] is None:
        _KNOBS["v"] = schema.load_knobs()
    return _KNOBS["v"]


def _shift(config, d, ctx, knobs):
    """Every band and feature level + d; (levels, max_level) or (None, why)."""
    cap = schema._knob_row("/refinement/max_level", knobs)["max"]
    levels = _get(config, "/refinement/levels")
    has_band = any(e.get("bands") for e in levels)
    if has_band:
        out = copy.deepcopy(levels)
        top = 0
        for e in out:
            for b in e.get("bands") or []:
                b["level"] = b["level"] + d
                if b["level"] > cap:
                    return None, "a band would pass max_level %d" % cap
                if b["level"] < 0:
                    return None, "a band would fall below level 0"
                top = max(top, b["level"])
            if "feature_level" in e:
                e["feature_level"] = min(max(e["feature_level"] + d, 0), cap)
                top = max(top, e["feature_level"])
        ml = min(max(_get(config, "/refinement/max_level") + d, top), cap)
        return out, max(0, ml)
    if d > 0:
        base = config["domain"]["base_size"]
        bands = rules.bands_for(ctx["flow"]["l_ref_m"], base, 1, 1.0, 1.0)
        out = [{"patch": p, "bands": copy.deepcopy(bands)} for p in _patches(ctx)]
        return out, min(max(1, _get(config, "/refinement/max_level") + 1), cap)
    return None, "no wall band to coarsen"


def _refit_growth(config, t1, n, h):
    """g_cur while the stack fits cell_frac*h, else the largest fitting step (1.0 at worst)."""
    cf = _get(config, "/layers/cell_frac")
    g_cur = _get(config, "/layers/growth")
    if rules.stack_total(t1, g_cur, n) <= cf * h * (1 - rules.EDGE_MARGIN):
        return g_cur
    g = rules.fit_growth(t1, n, h * cf / rules.CELL_FRAC, _knobs())
    return g if g is not None else 1.0


def _cells_ok(after, ctx, gates):
    """rules.predict_cells against 0.7 * cell_budget (R-BUDGET's safety)."""
    n = _get(after, "/layers/n")
    total = rules.predict_cells(after, ctx["fingerprint"],
                                n if _requested(after) else 0)["total"]
    return total <= rules.PRED_SAFETY * gates["cell_budget"]


def _cells_why(after, ctx, gates):
    total = rules.predict_cells(after, ctx["fingerprint"],
                                _get(after, "/layers/n") if _requested(after) else 0)["total"]
    return "predicted %d cells > 0.7*cell_budget = %d" \
        % (round(total), round(rules.PRED_SAFETY * gates["cell_budget"]))


def _qualifies(ctx, gates, knobs):
    """rules.setup's config when its R-PLANE applies; (None, why) otherwise (cached)."""
    before = ctx["config"]
    key = (ctx["geometry_id"], schema.canonical_sha256(before), ctx["flow"]["l_ref_m"],
           ctx["flow"]["u_ref_m_s"], ctx["flow"]["nu_m2_s"])
    if key in _QUAL_CACHE:
        return _QUAL_CACHE[key]
    inp = before.get("input") or {}
    stl = (inp.get("surfaces") or [{}])[0].get("path", "surface.stl")
    out = before.get("output") or {}
    res = rules.setup({"geometry_id": ctx["geometry_id"], "flow": ctx["flow"]},
                      ctx["fingerprint"], stl, out.get("case_dir", "case"),
                      out.get("name", ctx["geometry_id"]),
                      flow=ctx["flow"], gates=gates, knobs=knobs)
    if res["verdict"] != "apply":
        out2 = (None, "rules.setup refuses (%s)" % res["refused"])
    elif res["summary"]["plane"] is not True:
        out2 = (None, "R-PLANE does not apply (features.py: commensurate %s)"
                % ctx["fingerprint"].get("commensurate"))
    else:
        out2 = (res["config"], None)
    _QUAL_CACHE[key] = out2
    return out2


_QUAL_CACHE = {}


# --- the twelve remedies (C5): (after | None, why | None, extra inputs) -------

def _coarsen(ctx, gates, knobs, plane_guard):
    """The shared coarsen guards; the pack (levels, max_level) goes back for _set."""
    before = ctx["config"]
    if plane_guard and on_plane(before, ctx["fingerprint"]):
        return None, "on the R-PLANE path (the plane owns the wall level)", None, []
    fl = floor_level(before, ctx, knobs)
    if fl is None:
        return None, "no y+ floor (no level is under 60 t1)", None, []
    w = wall_level(before)
    if w - 1 < fl:
        return None, "below the y+ floor (wall %d -> %d < floor %d)" % (w, w - 1, fl), None, []
    levels, ml = _shift(before, -1, ctx, knobs)
    if levels is None:
        return None, ml, None, []
    return copy.deepcopy(before), None, (levels, ml), []


def _refine(ctx, gates, knobs):
    """The shared refine; the pack (levels, max_level) goes back for _set."""
    before = ctx["config"]
    levels, ml = _shift(before, +1, ctx, knobs)
    if levels is None:
        return None, ml, None, []
    return copy.deepcopy(before), None, (levels, ml), []


def _rm_budget_far(ctx, gates, knobs):
    before = ctx["config"]
    w = wall_level(before)
    base = before["domain"]["base_size"]
    levels = copy.deepcopy(_get(before, "/refinement/levels"))
    changed = False
    for e in levels:
        prev = 0.0
        for b in e.get("bands") or []:
            if b["level"] < w:
                nd = max(b["distance"] / 2.0, rules.BAND_MIN_CELLS * base / 2 ** b["level"], prev)
                changed = changed or nd != b["distance"]
                b["distance"] = nd
            prev = b["distance"]
    if not changed:
        return None, "every far-field band is at its 3-cell minimum", []
    after = copy.deepcopy(before)
    _set(after, "/refinement/levels", levels)
    return after, None, []


def _rm_budget_feat(ctx, gates, knobs):
    before = ctx["config"]
    w = wall_level(before)
    levels = copy.deepcopy(_get(before, "/refinement/levels"))
    hit = False
    for e in levels:
        if e.get("feature_level", 0) > w:
            e["feature_level"] = w
            hit = True
    if not hit:
        return None, "no feature level above the wall level", []
    ml = max([w] + [e.get("feature_level", 0) for e in levels])
    after = copy.deepcopy(before)
    _set(after, "/refinement/levels", levels)
    _set(after, "/refinement/max_level", ml)
    return after, None, []


def _rm_budget_wall(ctx, gates, knobs):
    after, why, pack, extra = _coarsen(ctx, gates, knobs, False)
    if after is None:
        return None, why, extra
    _set(after, "/refinement/levels", pack[0])
    _set(after, "/refinement/max_level", pack[1])
    return after, None, extra


def _rm_snap_wall(ctx, gates, knobs):
    after, why, pack, extra = _coarsen(ctx, gates, knobs, True)
    if after is None:
        return None, why, extra
    _set(after, "/refinement/levels", pack[0])
    _set(after, "/refinement/max_level", pack[1])
    return after, None, extra


def _rm_topo_refine(ctx, gates, knobs):
    after, why, pack, extra = _refine(ctx, gates, knobs)
    if after is None:
        return None, why, extra
    _set(after, "/refinement/levels", pack[0])
    _set(after, "/refinement/max_level", pack[1])
    if not _cells_ok(after, ctx, gates):
        return None, _cells_why(after, ctx, gates), extra
    return after, None, extra


def _rm_snap_refine(ctx, gates, knobs):
    after, why, pack, extra = _refine(ctx, gates, knobs)
    if after is None:
        return None, why, extra
    _set(after, "/refinement/levels", pack[0])
    _set(after, "/refinement/max_level", pack[1])
    if not _cells_ok(after, ctx, gates):
        return None, _cells_why(after, ctx, gates), extra
    return after, None, extra


def _rm_snap_ft(ctx, gates, knobs):
    before = ctx["config"]
    if ctx["fingerprint"].get("sharp_edge_length_m", 0) <= 0:
        return None, "no sharp edge (features.py)", []
    if _get(before, "/snap/feature_tolerance") == 0:
        return None, "the attraction is already off", []
    after = copy.deepcopy(before)
    _set(after, "/snap/feature_tolerance", 0.0)
    return after, None, []


def _rm_t1_raise(ctx, gates, knobs):
    before = ctx["config"]
    flow = ctx["flow"]
    extra = []
    line = ctx["outcome"].get("refusal_line")
    h_min = None
    if isinstance(line, str):
        m = preflight.G5_LINE_RE.search(line)
        if m:
            try:
                h_min = float(m.group(2))
            except ValueError:
                h_min = None
    w = schema.a_priori_wall(flow, gates["yplus_max_a_priori"])
    t1a = w.get("t1_a_priori_m")
    extra.append({"name": "h_min", "value": h_min, "unit": "m"})
    t1 = _get(before, "/layers/first_thickness")
    if t1a is None:
        extra.append({"name": "predicted_ratio", "value": None, "unit": "1"})
        return None, "no skin-friction correlation for this flow", extra
    t1_max = rules.t1_floor(t1a)
    extra.append({"name": "predicted_ratio",
                  "value": (3 * t1_max / h_min) if h_min else None, "unit": "1"})
    if t1 >= t1_max * (1 - 1e-12):
        return None, "t1 = %g m is already at the y+ = 1 bound %g m" % (t1, t1_max), extra
    after = copy.deepcopy(before)
    _set(after, "/layers/first_thickness", t1_max)
    g = _refit_growth(after, t1_max, _get(before, "/layers/n"),
                      before["domain"]["base_size"] / 2 ** wall_level(before))
    if g != _get(before, "/layers/growth"):
        _set(after, "/layers/growth", g)
    return after, None, extra


def _rm_t1_refine(ctx, gates, knobs):
    before = ctx["config"]
    after, why, pack, extra = _refine(ctx, gates, knobs)
    if after is None:
        return None, why, extra
    _set(after, "/refinement/levels", pack[0])
    _set(after, "/refinement/max_level", pack[1])
    if not _cells_ok(after, ctx, gates):
        return None, _cells_why(after, ctx, gates), extra
    if _requested(before):
        t1 = _get(before, "/layers/first_thickness")
        g = _refit_growth(after, t1, _get(before, "/layers/n"),
                          after["domain"]["base_size"] / 2 ** wall_level(after))
        if g != _get(before, "/layers/growth"):
            _set(after, "/layers/growth", g)
    return after, None, extra


_PLANE_POINTERS = ("/domain/extent", "/domain/base_size", "/refinement/levels",
                   "/refinement/max_level", "/snap/feature_tolerance",
                   "/snap/smoothing_passes", "/layers/patches", "/layers/n",
                   "/layers/first_thickness", "/layers/growth")


def _rm_plane(ctx, gates, knobs):
    rc, why = _qualifies(ctx, gates, knobs)
    if rc is None:
        return None, why, []
    after = copy.deepcopy(ctx["config"])
    _set(after, "/domain/extent", rc["domain"]["extent"])
    _set(after, "/domain/base_size", rc["domain"]["base_size"])
    _set(after, "/refinement/levels", rc["refinement"]["levels"])
    _set(after, "/refinement/max_level", rc["refinement"]["max_level"])
    _set(after, "/snap/feature_tolerance", rc["snap"]["feature_tolerance"])
    _set(after, "/snap/smoothing_passes", rc["snap"]["smoothing_passes"])
    _set(after, "/layers/patches", rc["layers"]["patches"])
    _set(after, "/layers/n", rc["layers"]["n"])
    _set(after, "/layers/first_thickness", rc["layers"]["first_thickness"])
    _set(after, "/layers/growth", rc["layers"]["growth"])
    return after, None, []


def _rm_plane_finer(ctx, gates, knobs):
    before = ctx["config"]
    fp = ctx["fingerprint"]
    s = fp["lattice_base_size_m"]
    base = before["domain"]["base_size"]
    h = base / 2 ** wall_level(before)
    m = max(1, round(s / h))
    h2 = s / (m + 1)
    t1 = _get(before, "/layers/first_thickness")
    n = _get(before, "/layers/n")
    cf = _get(before, "/layers/cell_frac")
    lo1 = n * t1 / cf * (1 + rules.EDGE_MARGIN) if _requested(before) else 0
    if h2 < lo1:
        return None, ("below the window's lower edge: the next divisor s/%d = %.4g m "
                      "< n*t1/cell_frac = %.4g m" % (m + 1, h2, lo1)), []
    l_ref = ctx["flow"]["l_ref_m"]
    cap = schema._knob_row("/refinement/max_level", knobs)["max"]
    lvl = min(cap, max(0, round(math.log2(rules.BASE_FRAC * l_ref / h2))))
    base2 = h2 * 2 ** lvl
    bb = fp["bbox"]
    mg = [f * l_ref for f in rules.DOMAIN_MARGINS]
    ext = []
    for a in range(3):
        lo = bb[2 * a] - math.ceil(mg[2 * a] / base2) * base2
        kk = math.ceil((bb[2 * a + 1] + mg[2 * a + 1] - lo) / base2)
        ext += [lo, lo + kk * base2]
    bands = rules.bands_for(l_ref, base2, lvl, 1.0, 1.0)
    levels2 = [{"patch": p, "bands": copy.deepcopy(bands)} for p in _patches(ctx)] if bands else []
    g = rules.fit_growth(t1, n, h2 * cf / rules.CELL_FRAC, knobs)
    if g is None:
        return None, "no growth fits cell_frac*h' = %.4g m" % (cf * h2), []
    after = copy.deepcopy(before)
    _set(after, "/domain/extent", ext)
    _set(after, "/domain/base_size", base2)
    _set(after, "/refinement/levels", levels2)
    _set(after, "/refinement/max_level", lvl if bands else 0)
    _set(after, "/layers/growth", g)
    if not _cells_ok(after, ctx, gates):
        return None, _cells_why(after, ctx, gates), []
    return after, None, []


def _rm_layer_fit(ctx, gates, knobs):
    before = ctx["config"]
    t1 = _get(before, "/layers/first_thickness")
    n = _get(before, "/layers/n")
    g_cur = _get(before, "/layers/growth")
    cf = _get(before, "/layers/cell_frac")
    h = before["domain"]["base_size"] / 2 ** wall_level(before)
    if rules.stack_total(t1, g_cur, n) <= cf * h * (1 - rules.EDGE_MARGIN):
        return None, "the stack already fits cell_frac*h_wall (the limiter is not why)", []
    g = rules.fit_growth(t1, n, h * cf / rules.CELL_FRAC, knobs)
    after = copy.deepcopy(before)
    if g is not None:
        _set(after, "/layers/growth", g)
        return after, None, []
    t2 = rules.t1_floor(cf * h * (1 - rules.EDGE_MARGIN) / n)
    if rules.G5_FACTOR * t2 / h < rules.G5_RATIO * (1 + rules.EDGE_MARGIN):
        return None, ("t1' = %g m would fail the G5 early check (3*t1'/h_wall = %.4g < 0.05)"
                      % (t2, rules.G5_FACTOR * t2 / h)), []
    _set(after, "/layers/first_thickness", t2)
    _set(after, "/layers/growth", 1.0)
    return after, None, []


# --- diagnose (C2): the key and the earliest failing stage --------------------

def diagnose(outcome: dict, gates: dict) -> dict:
    """{"key", "stage", "trigger"} - first match wins, in the §D.1 flag order."""
    gates = gates if gates is not None else schema.load_gates()
    fl = outcome.get("flags") or {}
    cls = outcome.get("failure_class")

    def trig(ob, value, th, op, src):
        return {"observable": ob, "value": value, "threshold": th, "op": op, "source": src}
    if fl.get("F1") is True:
        t = trig("outcome.failure_class", cls, cls, "==",
                 "score.py failure enum (docs/15 §D.1 F1)")
        if cls in ("surface_closed", "config", "io", "crash"):
            return {"key": cls, "stage": None, "trigger": t}
        if cls == "timeout":
            return {"key": "timeout", "stage": "octree", "trigger": t}
        if cls == "layer_t1_G5":
            return {"key": "layer_t1_G5", "stage": "layers", "trigger": t}
        if isinstance(cls, str) and cls.startswith("gate_") and "@" in cls:
            stage = cls.split("@", 1)[1]
            return {"key": "gate@" + stage, "stage": stage, "trigger": t}
        raise RemedyError("diagnose: an F1 outcome carries the unknown failure class %r" % (cls,))
    if fl.get("F5") is True:
        return {"key": "F5", "stage": "octree", "trigger": trig(
            "outcome.n_cells", outcome.get("n_cells"), gates["cell_budget"], ">",
            "gates.json cell_budget (docs/15 §D.1 F5)")}
    if fl.get("F4") is True:
        return {"key": "F4", "stage": "castellate", "trigger": trig(
            "outcome.flags.F4", True, True, "==",
            "score.py stages[castellate].wall_patches, quality.n_regions (docs/15 §D.1 F4)")}
    if fl.get("F3a") is True or fl.get("F3b") is True or fl.get("F3c") is True:
        for ob, val, th, src in (
            ("outcome.pinned_frac", outcome.get("pinned_frac"), gates["pinned_frac_max"],
             "score.py stages[snap] n_pinned_boundary / n_boundary_points (docs/15 §D.1 F3a)"),
            ("outcome.p99_over_hf", outcome.get("p99_over_hf"),
             gates["p99_residual_over_hf_max"],
             "score.py stages[snap] p99_residual / h_f (docs/15 §D.1 F3b)"),
            ("outcome.max_over_hf", outcome.get("max_over_hf"),
             gates["max_residual_over_hf_max"],
             "score.py stages[snap] max_residual / h_f (docs/15 §D.1 F3c)")):
            if val is not None and val > th:
                return {"key": "F3", "stage": "snap", "trigger": trig(ob, val, th, ">", src)}
        raise RemedyError("diagnose: an F3 flag is set but no snap metric exceeds its gate")
    if fl.get("F3d") is True:
        return {"key": "F3d", "stage": "snap", "trigger": trig(
            "outcome.flags.F3d", True, [gates["area_ratio_min"], gates["area_ratio_max"]],
            "not_in", "score.py stages[snap].area_ratio (docs/15 §D.1 F3d)")}
    if fl.get("F2") is True:
        return {"key": "F2", "stage": None, "trigger": trig(
            "outcome.flags.F2", True, True, "==", "-check (docs/15 §D.1 F2)")}
    for p in outcome.get("patches") or []:
        if p.get("requested") is True and p.get("layer_class") is not None:
            return {"key": "layer:" + p["layer_class"], "stage": "layers", "trigger": trig(
                "outcome.patches[%s].layer_class" % p["name"], p["layer_class"],
                p["layer_class"], "==", "score.classify_drop (the layers.rs drop text)")}
    return {"key": "pass", "stage": None, "trigger": trig(
        "outcome.failure", False, False, "==", "score.py (docs/15 §D.1, §D.2)")}


# --- propose (C6) --------------------------------------------------------------

def _fmt(v) -> str:
    if isinstance(v, float):
        return "%.6g" % v
    if isinstance(v, list):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    if v is None:
        return "absent"
    return json.dumps(v)


def _rec(rule_id, verdict, trigger, inputs, formula, edits, cite, message):
    """One DecisionRecord of layer remedy, validated before it leaves this module."""
    rec = {"schema": "autonomy-decision/1", "layer": "remedy", "rule_id": rule_id,
           "verdict": verdict, "trigger": trigger, "inputs": inputs, "formula": formula,
           "edits": edits, "cite": cite, "message": message, "uncertainty": 0,
           "t": schema._now_iso()}
    errs = schema.errors(rec, "DecisionRecord")
    if errs:
        raise RemedyError("the remedy record is invalid: %s" % errs[0])
    return rec


def _no_remedy_why(key):
    if key == "surface_closed":
        return "a surface defect: tools/geom/stl_repair.py first"
    if key == "config":
        return "preflight.py refuses a config the mesher rejects"
    if key in ("io", "crash"):
        return "a harness fault, not a mesh the table can change"
    if key.startswith("gate@"):
        return "a quality gate at %s: no remedy touches the quality block" % key[len("gate@"):]
    return "the written mesh fails -check: no remedy touches the quality block"


def propose(ctx: dict, history: list, *, k=None, gates=None, knobs=None, veto=None) -> dict:
    """The tabled remedy for one failed attempt, or the geometry's terminal."""
    gates = gates if gates is not None else schema.load_gates()
    knobs = knobs if knobs is not None else schema.load_knobs()
    fp = ctx["fingerprint"]
    errs = schema.errors(fp, "Fingerprint")
    if errs:
        raise RemedyError("propose: the fingerprint is invalid: %s" % errs[0])
    errs = schema.errors(ctx["flow"], "FlowSpec")
    if errs:
        raise RemedyError("propose: the flow is invalid: %s" % errs[0])
    if not history:
        raise RemedyError("propose: the history is empty")
    config = ctx["config"]
    sha_before = schema.canonical_sha256(config)
    if history[-1].get("config_sha256") != sha_before:
        raise RemedyError("propose: history[-1] carries sha %s, the ctx config's is %s"
                          % (str(history[-1].get("config_sha256"))[:12], sha_before[:12]))
    outcome = ctx["outcome"]
    k = k or gates["attempts_k"]
    fired = {}
    seen = set()
    for h in history:
        if h.get("rule_id"):
            fired[h["rule_id"]] = fired.get(h["rule_id"], 0) + 1
        seen.add(h["config_sha256"])
    d = diagnose(outcome, gates)
    key = d["key"]
    T = "%s = %s %s %s; %s" % (d["trigger"]["observable"], _fmt(d["trigger"]["value"]),
                               d["trigger"]["op"], _fmt(d["trigger"]["threshold"]),
                               d["trigger"]["source"])
    skipped = []
    names = [p["name"] for p in outcome.get("patches") or []
             if p.get("requested") and p.get("layer_class")
             in ("min_thickness", "retreat_snapped")]

    def base_inputs(fires, sha_after, extra):
        return [{"name": "attempt", "value": len(history), "unit": "1"},
                {"name": "k", "value": k, "unit": "1"},
                {"name": "key", "value": key, "unit": "1"},
                {"name": "stage_focus", "value": d["stage"], "unit": "1"},
                {"name": "fires_before", "value": fires, "unit": "1"},
                {"name": "wall_level", "value": wall_level(config), "unit": "1"},
                {"name": "floor_level", "value": floor_level(config, ctx, knobs), "unit": "1"},
                {"name": "on_plane", "value": on_plane(config, fp), "unit": "1"},
                {"name": "config_sha256_before", "value": sha_before, "unit": ""},
                {"name": "config_sha256_after", "value": sha_after, "unit": ""},
                {"name": "skipped", "value": copy.deepcopy(skipped), "unit": "1"}] + list(extra)

    def result(rid, verdict, terminal, after, sha, edits, rec):
        return {"schema": RESULT_SCHEMA, "geometry_id": ctx["geometry_id"],
                "attempt": len(history), "verdict": verdict, "terminal": terminal,
                "rule_id": rid, "key": key, "stage_focus": d["stage"], "config": after,
                "config_sha256": sha, "edits": edits, "record": rec, "skipped": skipped,
                "capability_limited": names if terminal == "CAPABILITY-LIMITED" else []}

    def terminal(name, detail, cap_names=None):
        rid = TERMINAL_ID[name]
        verdict = {"PASS": "pass", "CAPABILITY-LIMITED": "refuse",
                   "EXHAUSTED": "refuse", "NO-REMEDY": "abstain"}[name]
        cite = {"PASS": "docs/15 §D.1, §D.2",
                "CAPABILITY-LIMITED":
                    "SPEC-LIT §92.13 (layers on a snapped wall are attempted, retreated and "
                    "given up by name); docs/15 §C L2, §I-1: only AM-L changes this",
                "EXHAUSTED":
                    "docs/15 §C L2 (each remedy at most twice per geometry, no config sha "
                    "revisited); gates.json attempts_k",
                "NO-REMEDY": "docs/15 §C L2: outside the remedy table"}[name]
        if name == "PASS":
            msg = ("RM-PASS: attempt %d passes: no F flag of docs/15 §D.1 and no requested "
                   "layer patch dropped or trimmed") % len(history)
        elif name == "CAPABILITY-LIMITED":
            msg = ("RM-CAPABILITY-LIMITED: %s at stage layers (%s) on a snapped wall - %s; "
                   "SPEC-LIT §92.13 drops layers there, so %s are CAPABILITY-LIMITED and no "
                   "more trials are spent on their layers"
                   % (key, T, detail, ", ".join(cap_names if cap_names is not None else names)))
        elif name == "EXHAUSTED":
            msg = "RM-EXHAUSTED: %s at stage %s (%s) remains: %s" % (key, d["stage"], T, detail)
        else:
            msg = ("RM-NO-REMEDY: %s at stage %s (%s) is outside the remedy table: %s"
                   % (key, d["stage"], T, detail))
        rec = _rec(rid, verdict, d["trigger"], base_inputs(0, None, []), "", [], cite, msg)
        return result(rid, "terminal", name, None, None, [], rec)

    if key == "pass":
        return terminal("PASS", "")
    if key in NO_REMEDY_KEYS:
        return terminal("NO-REMEDY", _no_remedy_why(key))
    cap_route = False
    if key in DROP_KEYS:
        if on_plane(config, fp):
            rows = [r for r in REMEDIES if r["id"] == "RM-PLANE-FINER"]
        else:
            rc, why = _qualifies(ctx, gates, knobs)
            if rc is None:
                return terminal("CAPABILITY-LIMITED",
                                "the body does not qualify for R-PLANE: " + why)
            rows = [r for r in REMEDIES if r["id"] == "RM-PLANE"]
            cap_route = True
    else:
        rows = [r for r in REMEDIES if key in r["keys"]]
    if len(history) >= k:
        return terminal("EXHAUSTED", "K = %d attempts spent" % k)
    for row in rows:
        rid = row["id"]
        if fired.get(rid, 0) >= MAX_FIRES:
            skipped.append({"rule_id": rid, "why": "fired twice"})
            continue
        after, why, extra = globals()[row["fn"]](ctx, gates, knobs)
        if after is None:
            skipped.append({"rule_id": rid, "why": why})
            continue
        edits = _commit(config, after, row["stage"], knobs)
        sha = schema.canonical_sha256(after)
        if sha in seen:
            skipped.append({"rule_id": rid, "why": "revisits config sha %s" % sha[:12]})
            continue
        if veto is not None:
            refused = veto(after)
            if refused:
                skipped.append({"rule_id": rid,
                                "why": "preflight refused: " + ", ".join(refused)})
                continue
        fires = fired.get(rid, 0)
        etext = "; ".join("%s %s -> %s" % (e["pointer"], _fmt(e["from"]), _fmt(e["to"]))
                          for e in edits[:6])
        if len(edits) > 6:
            etext += " (+%d more)" % (len(edits) - 6)
        msg = ("%s: %s at stage %s (%s) -> %s (fire %d of %d, attempt %d of %d): %s"
               % (rid, key, row["stage"], T, row["what"], fires + 1, MAX_FIRES,
                  len(history) + 1, k, etext))
        rec = _rec(rid, "apply", d["trigger"], base_inputs(fires, sha, extra),
                   row["formula"], edits, row["cite"], msg)
        return result(rid, "apply", None, after, sha, edits, rec)
    detail = "; ".join("%s %s" % (s["rule_id"], s["why"]) for s in skipped)
    if cap_route:
        return terminal("CAPABILITY-LIMITED", "R-PLANE could not fire: " + detail, names)
    return terminal("EXHAUSTED", detail)


# --- loop (C7) -----------------------------------------------------------------

def loop(ctx: dict, attempt_fn, *, k=None, gates=None, knobs=None, veto=None) -> dict:
    """Run attempts through propose until a terminal; the whole history out."""
    config, rid, history, results = ctx["config"], None, [], []
    while True:
        outcome = attempt_fn(config, len(history) + 1)
        history.append({"attempt": len(history) + 1,
                        "config_sha256": schema.canonical_sha256(config), "rule_id": rid})
        res = propose(dict(ctx, config=config, outcome=outcome), history, k=k,
                      gates=gates, knobs=knobs, veto=veto)
        results.append(res)
        if res["verdict"] == "terminal":
            break
        config, rid = res["config"], res["rule_id"]
    return {"schema": LOOP_SCHEMA, "geometry_id": ctx["geometry_id"],
            "terminal": res["terminal"], "attempts": len(history), "history": history,
            "results": results, "capability_limited": res["capability_limited"]}


# --- synthetic outcomes (C8) ----------------------------------------------------

_SYNTH_BASE = {"v": None}
_PROBE_LBL = {"v": None}


def _probe_labels():
    if _PROBE_LBL["v"] is None:
        with open(os.path.join(score.PROBES_DIR, "labels.json"), encoding="utf-8") as fh:
            _PROBE_LBL["v"] = json.load(fh)
    return _PROBE_LBL["v"]


def _synth_base():
    if _SYNTH_BASE["v"] is None:
        _SYNTH_BASE["v"] = score.score_probe("cubep_nofeat", _probe_labels())["outcome"]
    return copy.deepcopy(_SYNTH_BASE["v"])


_F1_SPECS = {
    "timeout": (None, "=== stage 3/5 snap ===\n", "", True),
    "crash": (101, "", "thread 'main' panicked (synthetic)\n", False),
    "io": (0, "", "", False),
    "config": (1, "", "error: domain/base_size: must be positive (synthetic)\n", False),
    "surface_closed": (1, "", 'error: surface/closed: "2 open edge(s), 0 non-manifold edge(s)" '
                              'is not supported by ofgpu (synthetic)\n', False),
    "gate_G4@castellate":
        (1, "=== stage 2/5 castellate ===\n",
         "error: automesher: quality gate G4 (non-orthogonality) failed on 12 face(s) "
         "(synthetic)\n", False),
    "gate_G4@snap":
        (1, "=== stage 3/5 snap ===\n",
         "error: automesher: quality gate G4 (non-orthogonality) failed on 12 face(s) "
         "(synthetic)\n", False),
    "layer_t1_G5": (1, "=== stage 5/5 layers ===\n",
                    "error: layers: the first layer is thinner than the quality gate allows - "
                    "3 * 0.0001 / 0.04 = 0.0075 < min_thickness_ratio = 0.05 (92.51); every "
                    "layer cell would fail G5, so none is inserted\n", False),
}


def synthetic_outcome(kind: str, config: dict, fingerprint: dict, flow: dict,
                      gates=None) -> dict:
    """A real scorer-shaped outcome of the named kind, for tests and the property cases."""
    gates = gates if gates is not None else schema.load_gates()
    areas = {}
    for p in fingerprint["patches"]:
        areas[p["name"]] = areas.get(p["name"], 0.0) + p["area_m2"]
    if kind in _F1_SPECS:
        e, out, err, to = _F1_SPECS[kind]
        return score.score_run(exit_code=e, stdout=out, stderr=err, summary=None,
                               config=config, patch_areas_m2=areas, flow=flow,
                               timed_out=to)["outcome"]
    oc = _synth_base()
    oc["exit_code"] = 0
    oc["last_stage"] = "layers"
    oc["n_cells"] = 10000
    oc["pinned_frac"] = 0.0
    oc["p99_over_hf"] = 0.001
    oc["max_over_hf"] = 0.001
    for f in oc["flags"]:
        oc["flags"][f] = False
    oc["refusal_line"] = None
    oc["failure_class"] = None
    oc["blc8_a_priori"] = 0.0
    oc["blc_full_a_priori"] = 0.0
    for b in oc["blc_beta_a_priori"]:
        b["lo"] = 0.0
        b["hi"] = 0.0
    oc["missing_signals"] = []
    t1 = _get(config, "/layers/first_thickness")
    n = _get(config, "/layers/n")
    cpatch = _get(config, "/layers/patches")
    requested = n > 0 and bool(cpatch)
    rows = []
    for name, a in areas.items():
        req = requested and name in cpatch
        rows.append({"name": name, "area_m2": a, "requested": req,
                     "n_layers": n if req else 0, "dropped": None,
                     "full_area_frac": 1.0 if req else None,
                     "t1_requested_m": t1 if req else None,
                     "yplus_a_priori": schema.yplus_a_priori(t1, flow) if req else None,
                     "delivered": bool(req and n >= gates["delivered_min_layers"]),
                     "capability_limited": False, "layer_class": None,
                     "mean_frac": 1.0 if req else None, "t1_min_m": t1 if req else None})
    oc["patches"] = rows
    first = next((r for r in rows if r["requested"]), None)
    if kind == "pass":
        pass
    elif kind == "F3a":
        oc["pinned_frac"] = 0.2
        oc["flags"]["F3a"] = True
    elif kind == "F3b":
        oc["p99_over_hf"] = 0.4
        oc["flags"]["F3b"] = True
    elif kind == "F3c":
        oc["max_over_hf"] = 0.8
        oc["flags"]["F3c"] = True
    elif kind in ("F3d", "F4", "F2"):
        oc["flags"][kind] = True
    elif kind == "F5":
        oc["n_cells"] = gates["cell_budget"] + 1
        oc["flags"]["F5"] = True
    elif kind in ("min_thickness", "retreat_snapped", "no_full_stack"):
        if first is None:
            raise RemedyError("synthetic_outcome: %s needs a config that requests layers" % kind)
        first["n_layers"] = 0
        if kind == "no_full_stack":
            first["full_area_frac"] = 0.0
            first["mean_frac"] = 0.5
        else:
            drop = ('patch "%s": the thickness fell below min_thickness * T = 1.000e-3 '
                    "(synthetic)" % first["name"]) if kind == "min_thickness" else \
                   ('patch "%s": the gate still failed after 4 retreat(s) (synthetic)'
                    % first["name"])
            first["dropped"] = drop
            first["full_area_frac"] = 0.0
            first["mean_frac"] = 0.0
        first["delivered"] = False
        first["layer_class"] = kind
        oc["failure_class"] = "layer_dropped:" + kind
    else:
        raise RemedyError("synthetic_outcome: unknown kind %r" % (kind,))
    oc["failure"] = any(v is True for v in oc["flags"].values())
    oc["verdict"] = "fail" if oc["failure"] else "pass"
    oc["strict_failure"] = bool(oc["failure"]) or any(
        r["requested"] and r["dropped"] is not None for r in rows)
    return oc


def synthetic_kinds(config: dict) -> tuple:
    """The kinds a config can take: the layer kinds only when layers are requested."""
    if _requested(config):
        return SYNTHETIC_KINDS
    return tuple(k for k in SYNTHETIC_KINDS if k not in LAYER_KINDS)


# --- the static scan (C9) --------------------------------------------------------

_SCAN_METH = ("update", "setdefault", "pop", "popitem", "clear", "append", "extend",
              "insert", "remove")


def _root_name(node):
    while isinstance(node, (ast.Subscript, ast.Attribute)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _str_bad(s, bans, flags):
    for b in bans:
        bp = b["pointer"]
        if b["match"] == "prefix" and (s.startswith(bp) or s.startswith(bp + "/")):
            return "a forbidden pointer literal %s" % bp
        if b["match"] == "exact" and s == bp:
            return "a forbidden pointer literal %s" % bp
    for fl in flags:
        if fl in s:
            return "a forbidden flag literal"
    return None


def static_scan(path: str | None = None, knobs=None) -> dict:
    """ast over this module: every write guarded, no forbidden literal, no spawn in a _rm_*."""
    knobs = knobs if knobs is not None else _knobs()
    src = path or os.path.abspath(__file__)
    with open(src, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=src)
    rows = {r["fn"]: r for r in REMEDIES}
    bans = knobs.get("forbidden", [])
    flags = [f["flag"] for f in knobs.get("forbidden_flags", [])]
    viol, writes, owner = [], {}, {}
    n_set = n_const = 0
    for fn in tree.body:
        if isinstance(fn, ast.FunctionDef):
            for sub in ast.walk(fn):
                owner[id(sub)] = fn.name
    exc_args = set()   # a pointer literal in a _get/_set call is judged by its own rule
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in ("_get", "_set") and len(node.args) >= 2 \
                and isinstance(node.args[1], ast.Constant):
            exc_args.add(id(node.args[1]))
    for node in ast.walk(tree):
        ofn = owner.get(id(node))
        in_rm = ofn is not None and ofn.startswith("_rm_")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            n_const += 1
            if id(node) not in exc_args:
                why = _str_bad(node.value, bans, flags)
                if why:
                    viol.append("%s: %s" % (ofn or "<module>", why))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "_set":
            n_set += 1
            if not in_rm:
                viol.append("%s: _set outside a _rm_* function" % (ofn or "<module>"))
                continue
            a1 = node.args[1] if len(node.args) >= 2 else None
            if not (isinstance(a1, ast.Constant) and isinstance(a1.value, str)):
                viol.append("%s: _set pointer is not a literal" % ofn)
                continue
            ptr = a1.value
            writes.setdefault(ofn, [])
            if ptr not in writes[ofn]:
                writes[ofn].append(ptr)
            if ptr not in CONTAINERS:
                row = schema._knob_row(ptr, knobs)
                if row is None or "*" in row["pointer"]:
                    viol.append("%s: _set pointer %s is not a whitelisted leaf" % (ofn, ptr))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr in _SCAN_METH and _root_name(node.func.value) == "after" \
                and ofn != "_set":
            viol.append("%s: a store into the config outside _set" % (ofn or "<module>"))
        tgts = []
        if isinstance(node, ast.Assign):
            tgts = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            tgts = [node.target]
        elif isinstance(node, ast.Delete):
            tgts = node.targets
        for tgt in tgts:
            root = _root_name(tgt)
            if root != "after":
                continue
            if isinstance(tgt, ast.Name):
                if isinstance(node, ast.Assign) and ofn != "_set":
                    val = node.value
                    ok = isinstance(val, ast.Call) and isinstance(val.func, ast.Attribute) \
                        and val.func.attr == "deepcopy"
                    if not ok:
                        viol.append("%s: after is not assigned a copy.deepcopy" % (ofn or "<module>"))
            elif ofn != "_set":
                viol.append("%s: a store into the config outside _set" % (ofn or "<module>"))
        if in_rm:
            if isinstance(node, ast.Import) and any(a.name.split(".")[0] == "subprocess"
                                                    for a in node.names):
                viol.append("%s: a remedy imports subprocess" % ofn)
            if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "subprocess":
                viol.append("%s: a remedy imports subprocess" % ofn)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and isinstance(node.func.value, ast.Name):
                nm = node.func.value.id
                if nm == "subprocess" or (nm == "os" and node.func.attr in ("system", "popen")):
                    viol.append("%s: a remedy spawns a process" % ofn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "run_automesher" \
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "score":
            third = node.args[2] if len(node.args) > 2 else None
            if not (isinstance(third, ast.List) and not third.elts):
                viol.append("%s: score.run_automesher without an empty argument list"
                            % (ofn or "<module>"))
    for fname, ptrs in writes.items():
        row = rows.get(fname)
        if row is None:
            continue
        for ptr in ptrs:
            krow = schema._knob_row(ptr, knobs)
            st = krow["stage"] if (krow is not None and ptr not in CONTAINERS) else "octree"
            if STAGES.index(st) > STAGES.index(row["stage"]):
                viol.append("%s: writes %s of a later stage" % (fname, ptr))
    return {"ok": not viol, "violations": viol,
            "writes": dict((k, sorted(v)) for k, v in writes.items()),
            "n_set_calls": n_set, "n_constants": n_const}


# --- the fixture harness (C11) --------------------------------------------------

_PROBE_OUT = {"v": {}}
_RULES_CTX = {"v": {}}


def _probe_outcome(pid, plabels):
    cache = _PROBE_OUT["v"]
    if pid not in cache:
        cache[pid] = score.score_probe(pid, plabels)["outcome"]
    return copy.deepcopy(cache[pid])


def _load_fixtures():
    with open(os.path.join(score.PROBES_DIR, "labels.json"), encoding="utf-8") as fh:
        plabels = json.load(fh)
    with open(os.path.join(FIXTURE_DIR, "labels.json"), encoding="utf-8") as fh:
        rl = json.load(fh)
    with open(os.path.join(FIXTURE_DIR, "fingerprints.json"), encoding="utf-8") as fh:
        fps = json.load(fh)["fingerprints"]
    return plabels, rl, fps


def probe_ctx(pid, plabels, fps):
    """The ctx of one frozen probe: its config, its scored outcome, its fingerprint."""
    row = next((r for r in plabels["probes"] if r["id"] == pid), None)
    if row is None:
        raise RemedyError("no probe %r in the probe labels" % pid)
    with open(os.path.join(score.PROBES_DIR, pid, "config.json"), encoding="utf-8") as fh:
        config = json.load(fh)
    return {"geometry_id": pid, "fingerprint": fps[row["stl"]], "flow": plabels["flow"],
            "config": config, "outcome": _probe_outcome(pid, plabels)}


def rules_ctx(stl, flow_name, plabels, rl, fps):
    """rules.setup's attempt-1 ctx for one fingerprint and flow (cached; asserted apply)."""
    key = (stl, flow_name)
    if key not in _RULES_CTX["v"]:
        flow = plabels["flow"] if flow_name == "probe" else rl["flows"][flow_name]
        stem = stl[:-4]
        res = rules.setup({"geometry_id": stem, "flow": flow}, fps[stl], stl,
                          "case_" + stem, stem, flow=flow)
        if res["verdict"] != "apply":
            raise RemedyError("rules.setup refuses %s (%s)" % (stl, res["refused"]))
        _RULES_CTX["v"][key] = (flow, res["config"])
    flow, config = _RULES_CTX["v"][key]
    return {"geometry_id": stl[:-4], "fingerprint": fps[stl], "flow": flow,
            "config": copy.deepcopy(config)}


def _fixture_set(cfg, pointer, value):
    """The test-only pointer setter (digit segments index lists); never named after."""
    segs = pointer.split("/")[1:]
    node = cfg
    for i, seg in enumerate(segs):
        key = int(seg) if seg.isdigit() else seg
        if i == len(segs) - 1:
            if value is None:
                del node[key]
            else:
                node[key] = value
            return
        if isinstance(node, dict) and key not in node:
            node[key] = [] if segs[i + 1].isdigit() else {}
        node = node[key]


def seq_ctx(case, plabels, rl, fps):
    """The ctx of one sequence case: a probe or rules config plus a probe or synthetic outcome."""
    spec = case["config"]
    if spec.startswith("probe:"):
        ctx = probe_ctx(spec[len("probe:"):], plabels, fps)
    else:
        parts = spec.split(":")
        ctx = rules_ctx(parts[1], parts[2], plabels, rl, fps)
    if case.get("config_edits"):
        for p, v in case["config_edits"].items():
            _fixture_set(ctx["config"], p, v)
    if "win_level" in case:
        ctx["win_level"] = case["win_level"]
    oc = case["outcome"]
    if oc.startswith("probe:"):
        ctx["outcome"] = _probe_outcome(oc[len("probe:"):], plabels)
    else:
        ctx["outcome"] = synthetic_outcome(oc[len("synthetic:"):], ctx["config"],
                                           ctx["fingerprint"], ctx["flow"])
    return ctx


def seq_history(ctx, case, gates, knobs):
    """The firing history of one sequence case, ending at the ctx config's real sha."""
    fired = case.get("fired") or {}
    ids = [None]
    for rid, c in fired.items():
        ids.extend([rid] * c)
    hist = [{"attempt": i, "config_sha256": "%064x" % i, "rule_id": rid}
            for i, rid in enumerate(ids, 1)]
    hist[-1]["config_sha256"] = schema.canonical_sha256(ctx["config"])
    for rid in case.get("seed_seen") or []:
        row = next(r for r in REMEDIES if r["id"] == rid)
        seeded = globals()[row["fn"]](ctx, gates, knobs)[0]
        if seeded is not None:
            hist.insert(0, {"attempt": 0, "config_sha256": schema.canonical_sha256(seeded),
                            "rule_id": None})
    for i, h in enumerate(hist, 1):
        h["attempt"] = i
    return hist


def _close(a, b) -> bool:
    """Numbers within 1e-12 of the label, bools and None exact, lists elementwise."""
    if a is None or b is None:
        return a is b
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= 1e-12 * max(1.0, abs(float(b)))
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    return a == b


def check_result(res, exp, is_seq, errs):
    """One labelled expectation against one propose result; the mismatches into errs."""
    if res["verdict"] != exp["verdict"]:
        errs.append("verdict %r != %r" % (res["verdict"], exp["verdict"]))
    if res["rule_id"] != exp["rule_id"]:
        errs.append("rule_id %r != %r" % (res["rule_id"], exp["rule_id"]))
    if res["terminal"] != exp["terminal"]:
        errs.append("terminal %r != %r" % (res["terminal"], exp["terminal"]))
    if is_seq and res["attempt"] != exp["attempt"]:
        errs.append("attempt %r != %r" % (res["attempt"], exp["attempt"]))
    if not is_seq:
        if res["key"] != exp["key"]:
            errs.append("key %r != %r" % (res["key"], exp["key"]))
        if res["stage_focus"] != exp["stage"]:
            errs.append("stage_focus %r != %r" % (res["stage_focus"], exp["stage"]))
    got = dict((e["pointer"], e["to"]) for e in res["edits"])
    want = exp.get("edits") or {}
    if set(got) != set(want):
        errs.append("edit pointers %s != %s" % (sorted(got), sorted(want)))
    else:
        for p, v in want.items():
            if not _close(got.get(p), v):
                errs.append("edit %s: %r != %r" % (p, got.get(p), v))
    if "capability_limited" in exp and res["capability_limited"] != exp["capability_limited"]:
        errs.append("capability_limited %s != %s"
                    % (res["capability_limited"], exp["capability_limited"]))
    for rid, sub in (exp.get("skipped_why") or {}).items():
        if not any(s["rule_id"] == rid and sub in s["why"] for s in res["skipped"]):
            errs.append("no skip of %s carrying %r (have %s)"
                        % (rid, sub, [(s["rule_id"], s["why"]) for s in res["skipped"]]))
    if "message_has" in exp and exp["message_has"] not in res["record"]["message"]:
        errs.append("the record message lacks %r" % exp["message_has"])


def _kind_pool(ctx):
    """synthetic_kinds, with the layer kinds dropped when the config's patch names and
    the fingerprint's never meet (wb.stl: the STL names its patch, the config another)."""
    kinds = synthetic_kinds(ctx["config"])
    cpatch = _get(ctx["config"], "/layers/patches")
    if cpatch and not any(p["name"] in cpatch for p in ctx["fingerprint"]["patches"]):
        kinds = tuple(k for k in kinds if k not in LAYER_KINDS)
    return kinds


def _pool(plabels, rl, fps):
    """The 51 ctxs of the property cases: the 31 probes plus 10 fingerprints x 2 flows."""
    out = []
    for row in plabels["probes"]:
        out.append((probe_ctx(row["id"], plabels, fps), True))
    for stl in sorted(fps):
        for fname in ("worked", "re5e4"):
            out.append((rules_ctx(stl, fname, plabels, rl, fps), False))
    return out


def _property(rng, pool, n, gates, knobs, records, det=False):
    """n single-step cases; every invariant of group 7; the counts out."""
    n_apply = n_term = v = 0
    for _ in range(n):
        ctx, is_probe = pool[rng.randrange(len(pool))]
        before = copy.deepcopy(ctx["config"])
        if is_probe and rng.random() < 0.5:
            oc = ctx["outcome"]
        else:
            oc = synthetic_outcome(rng.choice(_kind_pool(dict(ctx, config=before))), before,
                                   ctx["fingerprint"], ctx["flow"])
        hist = []
        ids = [None] + [r["id"] for r in REMEDIES]
        for i in range(rng.randint(0, 3)):
            hist.append({"attempt": i + 1, "config_sha256": "%064x" % rng.getrandbits(256),
                         "rule_id": rng.choice(ids)})
        c2 = dict(ctx, config=before, outcome=oc)
        hist.append({"attempt": len(hist) + 1,
                     "config_sha256": schema.canonical_sha256(before),
                     "rule_id": rng.choice(ids)})
        cfg_json = json.dumps(ctx["config"], sort_keys=True)
        res = propose(c2, hist, gates=gates, knobs=knobs)
        v += _property_checks(res, c2, hist, before, cfg_json, gates, knobs, records)
        if det:
            res2 = propose(c2, hist, gates=gates, knobs=knobs)
            a = copy.deepcopy(res)
            b = copy.deepcopy(res2)
            a["record"].pop("t")
            b["record"].pop("t")
            if json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True):
                v += 1
        if res["verdict"] == "apply":
            n_apply += 1
        else:
            n_term += 1
    return n_apply, n_term, v


def _property_checks(res, ctx, hist, before, cfg_json, gates, knobs, records):
    """The independent re-check of one single-step result; the violation count out."""
    v = 0
    rec = res["record"]
    records.append(rec)
    if schema.errors(rec, "DecisionRecord") or rec["layer"] != "remedy":
        v += 1
    if not rec["message"].startswith(rec["rule_id"]):
        v += 1
    if json.dumps(ctx["config"], sort_keys=True) != cfg_json:
        return v + 1
    if res["verdict"] == "terminal":
        if res["config"] is not None or res["config_sha256"] is not None or res["edits"]:
            v += 1
        return v
    rid = res["rule_id"]
    if rid not in {r["id"] for r in REMEDIES}:
        return v + 1
    fires = sum(1 for h in hist if h["rule_id"] == rid)
    if fires > 1 or len(hist) >= gates["attempts_k"]:
        v += 1
    cfg2 = res["config"]
    if res["config_sha256"] != schema.canonical_sha256(cfg2):
        v += 1
    if res["config_sha256"] in {h["config_sha256"] for h in hist}:
        v += 1
    if res["edits"] != rules.diff_edits(before, cfg2) or not res["edits"]:
        v += 1
    si = STAGES.index(res["stage_focus"])
    for e in res["edits"]:
        p = e["pointer"]
        row = schema._knob_row(p, knobs)
        if e["to"] is None:
            if row is None or _forbidden(p, knobs):
                v += 1
        elif schema.check_edit(p, e["to"], knobs) is not None:
            v += 1
        st = row["stage"] if row is not None else "octree"
        if STAGES.index(st) > si:
            v += 1
    for k in ("input", "output", "quality"):
        if json.dumps(before.get(k), sort_keys=True) != json.dumps(cfg2.get(k), sort_keys=True):
            v += 1
    bl = dict(preflight.config_leaves(before))
    al = dict(preflight.config_leaves(cfg2))
    for p, val in bl.items():
        if schema._knob_row(p, knobs) is None \
                and json.dumps(al.get(p), sort_keys=True) != json.dumps(val, sort_keys=True):
            v += 1
    for e2 in cfg2["refinement"]["levels"]:
        for b in e2.get("bands") or []:
            if not 0 <= b["level"] <= 6:
                v += 1
        if not 0 <= e2.get("feature_level", 0) <= 6:
            v += 1
    # The y+ floor of the config the remedy wrote, for its own t1 and base: equal to the
    # floor of `before` for every remedy that keeps them (the ladder shifts), and the
    # rules.setup window for RM-PLANE / RM-PLANE-FINER, which set t1 and base afresh.
    w0, w1 = wall_level(before), wall_level(cfg2)
    if w1 < w0:
        fl1 = floor_level(cfg2, ctx, knobs)
        if fl1 is not None and w1 < fl1:
            v += 1
    return v


def _sweep(pool, gates, knobs, records):
    """Every pool ctx x every synthetic kind it can take x three firing histories (none,
    RM-SNAP-WALL once, RM-PLANE twice), each re-checked as group 7; (n, violations) out.
    Added by the supervisor: the seeded 600 never drew a layer drop on box_L4, the one
    probe where RM-PLANE lowers the wall while it raises t1 to its y+ bound."""
    n = v = 0
    for ctx, _ in pool:
        for kind in _kind_pool(ctx):
            oc = synthetic_outcome(kind, ctx["config"], ctx["fingerprint"], ctx["flow"])
            for fired in ({}, {"RM-SNAP-WALL": 1}, {"RM-PLANE": 2}):
                ids = [None] + [rid for rid, c in fired.items() for _ in range(c)]
                hist = [{"attempt": i, "config_sha256": "%064x" % i, "rule_id": rid}
                        for i, rid in enumerate(ids, 1)]
                hist[-1]["config_sha256"] = schema.canonical_sha256(ctx["config"])
                c2 = dict(ctx, outcome=oc)
                cfg_json = json.dumps(ctx["config"], sort_keys=True)
                res = propose(c2, hist, gates=gates, knobs=knobs)
                v += _property_checks(res, c2, hist, ctx["config"], cfg_json, gates, knobs,
                                      records)
                n += 1
    return n, v


def _loops(rng, pool, n, gates, knobs, records):
    """n random loops; every invariant of group 8; the counts out."""
    terms, atts = {}, {}
    v = 0
    ids = [None] + [r["id"] for r in REMEDIES]
    for _ in range(n):
        ctx, is_probe = pool[rng.randrange(len(pool))]
        oc1 = ctx.get("outcome")
        fp, flow = ctx["fingerprint"], ctx["flow"]

        def attempt_fn(cfg, a, _fp=fp, _flow=flow, _oc=oc1, _probe=is_probe):
            if a == 1 and _probe:
                return _oc
            return synthetic_outcome(rng.choice(_kind_pool(dict(ctx, config=cfg))), cfg, _fp, _flow)

        lr = loop(ctx, attempt_fn, gates=gates, knobs=knobs)
        res = lr["results"][-1]
        records.extend(r["record"] for r in lr["results"])
        terms[lr["terminal"]] = terms.get(lr["terminal"], 0) + 1
        atts[lr["attempts"]] = atts.get(lr["attempts"], 0) + 1
        if lr["terminal"] not in TERMINALS or not 1 <= lr["attempts"] <= 4:
            v += 1
        if lr["attempts"] != len(lr["history"]) or lr["attempts"] != len(lr["results"]):
            v += 1
        shas = [h["config_sha256"] for h in lr["history"]]
        if len(set(shas)) != len(shas):
            v += 1
        fires = {}
        for h in lr["history"]:
            if h["rule_id"]:
                fires[h["rule_id"]] = fires.get(h["rule_id"], 0) + 1
        if any(c > MAX_FIRES for c in fires.values()):
            v += 1
        if not all(r["verdict"] == "apply" for r in lr["results"][:-1]) \
                or lr["results"][-1]["verdict"] != "terminal":
            v += 1
        if any(lr["history"][i + 1]["config_sha256"] != lr["results"][i]["config_sha256"]
               for i in range(len(lr["results"]) - 1)):
            v += 1
        key = res["key"]
        if (lr["terminal"] == "CAPABILITY-LIMITED" and key not in DROP_KEYS) \
                or (lr["terminal"] == "PASS") != (key == "pass") \
                or (lr["terminal"] == "NO-REMEDY") != (key in NO_REMEDY_KEYS):
            v += 1
    return terms, atts, v


# --- the selftest (C11) -----------------------------------------------------------

def _fix_probe_rows(plabels, rl, fps, gates, knobs):
    """Group 2/3/4 runners: the per-probe and per-sequence results with their checks."""
    def run_probes():
        out = []
        for row in rl["probes"]:
            ctx = probe_ctx(row["id"], plabels, fps)
            res = propose(ctx, [{"attempt": 1,
                                 "config_sha256": schema.canonical_sha256(ctx["config"]),
                                 "rule_id": None}], gates=gates, knobs=knobs)
            out.append((row, res))
        return out

    def run_seqs():
        out = []
        for case in rl["sequences"]:
            ctx = seq_ctx(case, plabels, rl, fps)
            hist = seq_history(ctx, case, gates, knobs)
            res = propose(ctx, hist, k=case.get("k"), gates=gates, knobs=knobs)
            out.append((case, res))
        return out

    return run_probes, run_seqs


def _strip_t(obj):
    if isinstance(obj, dict):
        return {k: _strip_t(v) for k, v in obj.items() if k != "t"}
    if isinstance(obj, (list, tuple)):
        return [_strip_t(v) for v in obj]
    return copy.deepcopy(obj)


def selftest() -> int:
    """13 [ok] groups over the frozen fixtures; no mesher process; under 60 s."""
    t0 = time.perf_counter()
    plabels, rl, fps = _load_fixtures()
    gates = schema.load_gates()
    knobs = schema.load_knobs()
    lines = []
    records = []

    def step(g, name, fn):
        try:
            lines.append(fn())
        except (AssertionError, RemedyError, OSError, ValueError) as e:
            sys.stderr.write("SELFTEST FAIL: group %d (%s): %s%s" % (g, name, e, chr(10)))
            return True
        return False
    run_probes, run_seqs = _fix_probe_rows(plabels, rl, fps, gates, knobs)

    def g1():
        ids = [r["id"] for r in REMEDIES]
        assert len(ids) == len(set(ids)) == 12, "the table needs 12 unique rows"
        pat = __import__("re").compile(r"^[A-Z][A-Z0-9]*(-[A-Z0-9]+)+$")
        for r in REMEDIES:
            assert pat.match(r["id"]), "%s is not a DecisionRecord rule id" % r["id"]
            assert r["stage"] in STAGES, "%s: bad stage" % r["id"]
            assert callable(globals()[r["fn"]]), "%s: fn missing" % r["id"]
            assert r["what"] and r["formula"] != "" and r["cite"], "%s: text missing" % r["id"]
        assert len(TERMINALS) == 4 and all(t in TERMINAL_ID for t in TERMINALS)
        keys = set()
        for r in REMEDIES:
            keys.update(r["keys"])
        routed = keys | set(NO_REMEDY_KEYS) | {"gate@snap", "pass"}
        assert len(routed) == 20, "20 keys routed, have %d: %s" % (len(routed), sorted(routed))
        for k in ("surface_closed", "gate@snap", "pass", "layer:min_thickness"):
            assert k in routed, "%s unrouted" % k
        return ("[ok] table: 12 remedies, 4 terminals, 20 keys routed "
                "(RM-BUDGET-FAR ... RM-LAYER-FIT)")

    def g2():
        counts = {}
        for row in rl["probes"]:
            ctx = probe_ctx(row["id"], plabels, fps)
            d = diagnose(ctx["outcome"], gates)
            if d["key"] != row["key"] or d["stage"] != row["stage"]:
                raise AssertionError("%s: key %r/%r != %r/%r"
                                     % (row["id"], d["key"], d["stage"], row["key"], row["stage"]))
            counts[d["key"]] = counts.get(d["key"], 0) + 1
        rank = sorted(counts.items(), key=lambda kv: -kv[1])
        txt = ", ".join("%s %d" % (k, n) for k, n in rank)
        assert len(plabels["probes"]) == 31
        return "[ok] diagnose: 31 probes keyed as labelled (%s)" % txt

    def g34():
        errs = []
        p_results = run_probes()
        for row, res in p_results:
            e = []
            check_result(res, row, False, e)
            errs.extend("%s: %s" % (row["id"], x) for x in e)
        s_results = run_seqs()
        for case, res in s_results:
            e = []
            check_result(res, case["expect"], True, e)
            errs.extend("%s: %s" % (case["id"], x) for x in e)
        assert not errs, "%d mismatch(es): %s" % (len(errs), errs[:3])
        rc = {}
        for _, res in p_results + s_results:
            rc[res["rule_id"]] = rc.get(res["rule_id"], 0) + 1
        records.extend(res["record"] for _, res in p_results + s_results)
        pc = {}
        for row, res in p_results:
            pc[res["rule_id"]] = pc.get(res["rule_id"], 0) + 1
        ptxt = ", ".join("%s %d" % (k, n) for k, n in
                         sorted(pc.items(), key=lambda kv: (-kv[1], kv[0])))
        return p_results, s_results, ("[ok] fixtures: 31/31 probes get the tabled remedy (%s)"
                                      % ptxt), "[ok] sequences: 32/32 as labelled"

    def g5():
        ctx = probe_ctx("box_sphere", plabels, fps)
        oc = ctx["outcome"]

        def att(cfg, a):
            return oc
        lr = loop(ctx, att, gates=gates, knobs=knobs)
        assert lr["terminal"] == "CAPABILITY-LIMITED" and lr["attempts"] == 1, \
            "box_sphere: terminal %r after %d attempts" % (lr["terminal"], lr["attempts"])
        assert lr["capability_limited"] == ["sphere"], lr["capability_limited"]
        assert len(lr["results"]) == 1 and lr["results"][0]["verdict"] == "terminal"
        records.append(lr["results"][0]["record"])
        return "[ok] box_sphere: CAPABILITY-LIMITED after 1 try, not 4 (patches sphere)"

    def g6():
        sc = static_scan(knobs=knobs)
        assert sc["ok"], "static scan: %s" % sc["violations"]
        n_fn = len(sc["writes"])
        assert n_fn == 12, "12 _rm_* functions with writes, have %d" % n_fn
        wr = "; ".join("%s[%s]" % (k[3:], ",".join(
            "levels" if p in CONTAINERS else p.rsplit("/", 1)[-1] for p in v))
            for k, v in sorted(sc["writes"].items()))
        d = tempfile.mkdtemp(prefix="remedies-scan-")
        try:
            q = "/" + "qua"
            ptr = q + "lity/max_non_orth_deg"
            flag = "-" + "per" + "missive"
            bad = ("import copy" + chr(10) +
                   "def _rm_bad(ctx, gates, knobs):" + chr(10) +
                   "    after = copy.deepcopy(ctx['config'])" + chr(10) +
                   "    _set(after, %r, 70.0)" % ptr + chr(10) +
                   "    after['domain'] = {}" + chr(10) +
                   "    return after" + chr(10) +
                   "TXT = %r" % (flag + " here") + chr(10))
            bp = os.path.join(d, "bad.py")
            with open(bp, "w", encoding="utf-8") as fh:
                fh.write(bad)
            sc2 = static_scan(bp, knobs=knobs)
            assert len(sc2["violations"]) == 3, sc2["violations"]
            assert any("not a whitelisted leaf" in v for v in sc2["violations"])
            assert any("store into the config" in v for v in sc2["violations"])
            assert any("flag literal" in v for v in sc2["violations"])
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)
        return ("[ok] static scan: %d _set calls in 12 _rm_* functions, 0 violations; "
                "writes %s" % (sc["n_set_calls"], wr))

    def g7():
        pool = _pool(plabels, rl, fps)
        rng = random.Random(10)
        n_apply, n_term, v = _property(rng, pool, 600, gates, knobs, records, det=True)
        assert v == 0, "%d property violations" % v
        n_sw, v_sw = _sweep(pool, gates, knobs, records)
        assert v_sw == 0, "%d sweep violations in %d cases" % (v_sw, n_sw)
        return ("[ok] property: 600 single steps, 0 violations (apply %d, terminal %d); "
                "exhaustive sweep %d cases, 0 violations" % (n_apply, n_term, n_sw))

    def g8():
        pool = _pool(plabels, rl, fps)
        rng = random.Random(11)
        terms, atts, v = _loops(rng, pool, 300, gates, knobs, records)
        assert v == 0, "%d loop violations" % v
        tt = ", ".join("%s %d" % (k, n) for k, n in sorted(terms.items()))
        aa = ", ".join("%d:%d" % (k, n) for k, n in sorted(atts.items()))
        return "[ok] loops: 300 random loops, 0 violations (terminals %s; attempts %s)" \
            % (tt, aa)

    def g9():
        contexts = [("rules wing_a", rules_ctx("wing_a.stl", "worked", plabels, rl, fps), 19),
                    ("rules cubep", rules_ctx("cubep.stl", "worked", plabels, rl, fps), 19),
                    ("probe NO26", probe_ctx("NO26", plabels, fps), 16)]
        key_of = {"pass": "pass", "F3a": "F3", "F3b": "F3", "F3c": "F3", "F3d": "F3d",
                  "F4": "F4", "F5": "F5", "F2": "F2", "timeout": "timeout", "crash": "crash",
                  "io": "io", "config": "config", "surface_closed": "surface_closed",
                  "gate_G4@castellate": "gate@castellate", "gate_G4@snap": "gate@snap",
                  "layer_t1_G5": "layer_t1_G5", "min_thickness": "layer:min_thickness",
                  "retreat_snapped": "layer:retreat_snapped",
                  "no_full_stack": "layer:no_full_stack"}
        n = 0
        for name, ctx, want_n in contexts:
            kinds = synthetic_kinds(ctx["config"])
            assert len(kinds) == want_n, "%s: %d kinds != %d" % (name, len(kinds), want_n)
            for kind in kinds:
                oc = synthetic_outcome(kind, ctx["config"], ctx["fingerprint"], ctx["flow"])
                errs = score.outcome_errors(oc, gates, knobs)
                assert not errs, "%s/%s: %s" % (name, kind, errs[:2])
                d = diagnose(oc, gates)
                assert d["key"] == key_of[kind], \
                    "%s/%s: key %r != %r" % (name, kind, d["key"], key_of[kind])
                n += 1
        return "[ok] synthetic: 19 kinds x 3 contexts pass score.outcome_errors " \
            "and diagnose to their key (%d outcomes)" % n

    def g10():
        assert len(records) >= 100, "only %d records collected" % len(records)
        for rec in records:
            errs = schema.errors(rec, "DecisionRecord")
            assert not errs, "%s: %s" % (rec.get("rule_id"), errs[:2])
            assert rec["layer"] == "remedy"
            assert rec["message"].startswith(rec["rule_id"])
        return "[ok] records: %d records valid, layer remedy, every message starts with " \
            "its rule id" % len(records)

    def g11():
        p1, s1 = g34()[0:2]
        p2, s2 = g34()[0:2]
        assert json.dumps(_strip_t(p1), sort_keys=True) == json.dumps(_strip_t(p2), sort_keys=True)
        assert json.dumps(_strip_t(s1), sort_keys=True) == json.dumps(_strip_t(s2), sort_keys=True)
        pool = _pool(plabels, rl, fps)
        rng = random.Random(11)
        la = _loops(rng, pool, 50, gates, knobs, [])
        rng = random.Random(11)
        lb = _loops(rng, pool, 50, gates, knobs, [])
        assert json.dumps(_strip_t(la), sort_keys=True) == json.dumps(_strip_t(lb), sort_keys=True)
        return "[ok] determinism: groups 3-4 twice and 50 loops twice equal apart from t"

    def g12():
        veto = lambda c: ["PF-TEST"]
        seqs = {c["id"]: c for c in rl["sequences"]}
        out = []
        for sid in ("S03", "S30"):
            case = seqs[sid]
            ctx = seq_ctx(case, plabels, rl, fps)
            hist = seq_history(ctx, case, gates, knobs)
            res = propose(ctx, hist, gates=gates, knobs=knobs, veto=veto)
            assert res["skipped"], "%s: nothing skipped" % sid
            for s in res["skipped"]:
                assert "preflight refused: PF-TEST" in s["why"], \
                    "%s: %s" % (sid, s["why"])
            out.append("%s %s" % (sid, res["terminal"]))
        assert out == ["S03 EXHAUSTED", "S30 CAPABILITY-LIMITED"], out
        return "[ok] veto: a refusing veto skips every candidate by name (S03 EXHAUSTED, " \
            "S30 CAPABILITY-LIMITED)"

    def g13():
        import subprocess
        env = dict(os.environ, PYTHONIOENCODING="utf-8")

        def run(*args):
            return subprocess.run([sys.executable, os.path.abspath(__file__), *args],
                                  capture_output=True, text=True, encoding="utf-8",
                                  errors="replace", env=env, timeout=300)
        p = run("--probe", "wing_a_L3")
        last = (p.stdout or "").strip().splitlines()[-1]
        assert p.returncode == 0 and last.startswith("REMEDY RM-SNAP-FT "), \
            "exit %d, last %r" % (p.returncode, last)
        q = run("--probe", "NOPE")
        assert q.returncode == 2 and "remedies:" in (q.stderr or ""), \
            "exit %d, err %r" % (q.returncode, (q.stderr or "")[:80])
        return "[ok] cli: --probe wing_a_L3 exit 0 ends REMEDY RM-SNAP-FT; --probe NOPE exit 2"

    for g, name, fn in ((1, "the table", g1), (2, "diagnose", g2),
                        (3, "fixtures", lambda: g34()[2]),
                        (4, "sequences", lambda: g34()[3]),
                        (5, "box_sphere", g5), (6, "static scan", g6),
                        (7, "property", g7), (8, "loops", g8), (9, "synthetic", g9),
                        (10, "records", g10), (11, "determinism", g11),
                        (12, "veto", g12), (13, "cli", g13)):
        if step(g, name, fn):
            return 1
        print(lines[-1])
    print("SELFTEST PASS (%.1f s)" % (time.perf_counter() - t0))
    return 0


# --- the gate (C10) and its report ----------------------------------------------

def _file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head():
    import subprocess
    try:
        p = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=60)
        return (p.stdout or "").strip()
    except OSError:
        return None


def _part1(plabels, rl, fps, gates, knobs):
    run_probes, run_seqs = _fix_probe_rows(plabels, rl, fps, gates, knobs)
    probes, hist_r, hist_t, bad = [], {}, {}, 0
    for row, res in run_probes():
        errs = []
        check_result(res, row, False, errs)
        bad += bool(errs)
        probes.append({"id": row["id"], "key": res["key"], "verdict": res["verdict"],
                       "rule_id": res["rule_id"], "terminal": res["terminal"],
                       "ok": not errs, "error": "; ".join(errs)})
        hist_r[res["rule_id"]] = hist_r.get(res["rule_id"], 0) + 1
        hist_t[res["terminal"] or "apply"] = hist_t.get(res["terminal"] or "apply", 0) + 1
    seqs = []
    for case, res in run_seqs():
        errs = []
        check_result(res, case["expect"], True, errs)
        bad += bool(errs)
        seqs.append({"id": case["id"], "verdict": res["verdict"], "rule_id": res["rule_id"],
                     "terminal": res["terminal"], "ok": not errs, "error": "; ".join(errs)})
        hist_r[res["rule_id"]] = hist_r.get(res["rule_id"], 0) + 1
        hist_t[res["terminal"] or "apply"] = hist_t.get(res["terminal"] or "apply", 0) + 1
    ctx = probe_ctx("box_sphere", plabels, fps)
    oc = ctx["outcome"]

    def att(cfg, a):
        return oc
    lr = loop(ctx, att, gates=gates, knobs=knobs)
    ok5 = (lr["terminal"] == "CAPABILITY-LIMITED" and lr["attempts"] == 1
           and lr["capability_limited"] == ["sphere"])
    bad += 0 if ok5 else 1
    return {"verdict": "PASS" if bad == 0 else "FAIL", "n_bad": bad, "probes": probes,
            "sequences": seqs, "rule_histogram": hist_r, "terminal_histogram": hist_t,
            "box_sphere_loop": {"terminal": lr["terminal"], "attempts": lr["attempts"],
                                "capability_limited": lr["capability_limited"],
                                "ok": ok5}}


def _part2(knobs):
    sc = static_scan(knobs=knobs)
    return {"verdict": "PASS" if sc["ok"] else "FAIL", "violations": sc["violations"],
            "n_set_calls": sc["n_set_calls"], "n_constants": sc["n_constants"],
            "writes": dict((k, v) for k, v in sorted(sc["writes"].items()))}


def _part3(plabels, rl, fps, gates, knobs):
    pool = _pool(plabels, rl, fps)
    records = []
    n_apply, n_term, v1 = _property(random.Random(10), pool, 600, gates, knobs, records)
    terms, atts, v2 = _loops(random.Random(11), pool, 300, gates, knobs, records)
    n_sw, v3 = _sweep(pool, gates, knobs, records)
    rec_h, term_h = {}, {}
    for rec in records:
        rec_h[rec["rule_id"]] = rec_h.get(rec["rule_id"], 0) + 1
    bad = v1 + v2 + v3
    return {"verdict": "PASS" if bad == 0 else "FAIL", "n_bad": bad,
            "property": {"n": 600, "apply": n_apply, "terminal": n_term, "violations": v1},
            "sweep": {"n": n_sw, "violations": v3},
            "loops": {"n": 300, "terminals": terms,
                      "attempts": dict((str(k), v) for k, v in sorted(atts.items())),
                      "violations": v2},
            "rule_histogram": rec_h, "n_records": len(records)}


P4_IDS = ("cube_cf", "cube_n5", "cube_ok", "cubep_cf", "cubep_defaults", "cubep_ok")


def _part4_one(pid, row, ctx0, out_dir, binary, gates, knobs, plabels):
    cfg = copy.deepcopy(ctx0["config"])
    cfg["input"]["surfaces"][0]["path"] = os.path.join(
        HERE, "fixtures", "stl", row["stl"]).replace(os.sep, "/")
    case_dir = os.path.join(out_dir, "p4", "case_" + pid).replace(os.sep, "/")
    cfg["output"] = {"case_dir": case_dir, "name": pid}
    ctx = dict(ctx0, config=cfg)
    outs = {}
    t_start = time.perf_counter()

    def attempt_fn(c, a, _pid=pid):
        if a == 1:
            return ctx0["outcome"]
        path = os.path.join(out_dir, "p4", "%s_a%d.json" % (_pid, a))
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(c, fh, indent=1, sort_keys=True)
        run = score.run_automesher(binary, path, [], cwd=os.path.join(out_dir, "p4"),
                                   timeout_s=300)
        summary = None
        sp = os.path.join(case_dir, "%s_summary.json" % _pid)
        if run["exit_code"] == 0 and os.path.isfile(sp):
            with open(sp, encoding="utf-8") as fh:
                summary = json.load(fh)
        oc = score.score_run(exit_code=run["exit_code"], stdout=run["stdout"],
                             stderr=run["stderr"], summary=summary, config=c,
                             patch_areas_m2=row["patch_areas_m2"], flow=plabels["flow"],
                             timed_out=run["timed_out"],
                             wall_seconds=run["seconds"])["outcome"]
        outs[a] = oc
        return oc
    lr = loop(ctx, attempt_fn, gates=gates, knobs=knobs)
    sec = time.perf_counter() - t_start
    res = lr["results"][-1]
    oc2 = outs.get(2)
    fired = [h["rule_id"] for h in lr["history"] if h["rule_id"]]
    row_out = {"id": pid, "rule_ids": fired, "terminal": lr["terminal"],
               "attempts": lr["attempts"],
               "failure_class": oc2.get("failure_class") if oc2 else None,
               "n_layers": (oc2["patches"][0]["n_layers"] if oc2 and oc2["patches"] else None),
               "full_area_frac": (oc2["patches"][0]["full_area_frac"]
                                  if oc2 and oc2["patches"] else None),
               "blc8_a_priori": oc2.get("blc8_a_priori") if oc2 else None,
               "blc_full_a_priori": oc2.get("blc_full_a_priori") if oc2 else None,
               "seconds": round(sec, 2),
               "skip_why": [s["why"] for s in res["skipped"]]}
    return row_out, lr


def _part4(plabels, rl, fps, gates, knobs, out_dir, streams, binary):
    os.makedirs(os.path.join(out_dir, "p4"), exist_ok=True)
    ctxs = {}
    for pid in P4_IDS:
        ctxs[pid] = (probe_ctx(pid, plabels, fps),
                      next(r for r in plabels["probes"] if r["id"] == pid))
    rows, lrs = {}, {}
    def one(pid):
        ctx0, row = ctxs[pid]
        return pid, _part4_one(pid, row, ctx0, out_dir, binary, gates, knobs, plabels)
    with ThreadPoolExecutor(max_workers=min(streams, 6)) as ex:
        for pid, (row_out, lr) in ex.map(one, P4_IDS):
            rows[pid], lrs[pid] = row_out, lr
    bad = 0
    for pid in ("cube_n5", "cube_ok", "cubep_defaults", "cubep_ok"):
        r = rows[pid]
        if not (r["terminal"] == "PASS" and r["attempts"] == 2
                and r["blc8_a_priori"] == 1.0 and r["blc_full_a_priori"] == 1.0):
            bad += 1
    for pid in ("cube_cf", "cubep_cf"):
        r = rows[pid]
        if not (r["terminal"] == "EXHAUSTED" and r["attempts"] == 2
                and any("RM-PLANE-FINER" == s["rule_id"]
                        and "below the window's lower edge" in s["why"]
                        for s in lrs[pid]["results"][-1]["skipped"])):
            bad += 1
    return {"verdict": "PASS" if bad == 0 else "FAIL", "n_bad": bad,
            "probes": [rows[pid] for pid in P4_IDS]}


def _write_md(report):
    """G-REMEDIES.md: every number read back from the JSON report; 60 lines at most."""
    with open(os.path.join(HERE, "README.md"), encoding="utf-8") as fh:
        header = fh.readline().rstrip()
    L = []
    L.append(header)
    L.append("")
    L.append("# G-REMEDIES (AM-10), %s: %s" % (report.get("date", "?"), report["verdict"]))
    L.append("")
    L.append("Run at tree `%s`, binary sha256 `%s...%s`. The report JSON is "
             "`remedies/G-REMEDIES.json`." % ((report.get("git_head") or "?")[:7],
                                              (report.get("binary_sha256") or "?")[:8],
                                              (report.get("binary_sha256") or "?")[-4:]))
    L.append("")
    p1 = report["parts"].get("1", {})
    if p1:
        L.append("- Part 1 %s: %d probes, %d sequences, box_sphere %s after %d attempt(s); "
                 "%d mismatches."
                 % (p1["verdict"], len(p1["probes"]), len(p1["sequences"]),
                    p1["box_sphere_loop"]["terminal"], p1["box_sphere_loop"]["attempts"],
                    p1["n_bad"]))
        L.append("  Probe/sequence rule ids: %s." % ", ".join(
            "%s %d" % (k, n) for k, n in sorted(p1["rule_histogram"].items(),
                                                key=lambda kv: (-kv[1], kv[0]))))
    p2 = report["parts"].get("2", {})
    if p2:
        L.append("- Part 2 %s: the static scan, %d _set calls in %d _rm_* functions, "
                 "%d violations, %d string constants scanned."
                 % (p2["verdict"], p2["n_set_calls"], len(p2["writes"]),
                    len(p2["violations"]), p2["n_constants"]))
    p3 = report["parts"].get("3", {})
    if p3:
        pr, lp = p3["property"], p3["loops"]
        sw = p3.get("sweep", {"n": 0, "violations": 0})
        L.append("- Part 3 %s: %d single steps (%d apply, %d terminal, %d violations), "
                 "%d loops (%s; attempts %s), %d violations; exhaustive sweep %d cases, "
                 "%d violations; %d records."
                 % (p3["verdict"], pr["n"], pr["apply"], pr["terminal"], pr["violations"],
                    lp["n"], ", ".join("%s %d" % kv for kv in sorted(lp["terminals"].items())),
                    ", ".join("%s:%s" % kv for kv in sorted(lp["attempts"].items())),
                    lp["violations"], sw["n"], sw["violations"], p3["n_records"]))
    p4 = report["parts"].get("4", {})
    if p4:
        L.append("- Part 4 %s: the six cube loops live (attempt 1 frozen, attempt 2 run by the "
                 "binary):" % p4["verdict"])
        for r in p4["probes"]:
            L.append("  - %s: %s after %d attempt(s), attempt 2 %s, n_layers %s, full %s, "
                     "BLC_8 %s, %.2f s."
                     % (r["id"], r["terminal"], r["attempts"],
                        r["failure_class"] or "no drop", r["n_layers"],
                        _fmt(r["full_area_frac"]), _fmt(r["blc8_a_priori"]), r["seconds"]))
    L.append("")
    out = os.path.join(REPORT_DIR, "G-REMEDIES.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(chr(10).join(L))
    return len(L)


def gate(parts, out_dir: str, streams: int = 6, binary: str | None = None) -> int:
    """G-REMEDIES: the labelled fixtures, the static scan, the property cases, six live loops."""
    os.makedirs(out_dir, exist_ok=True)
    binary = binary or score.BINARY_DEFAULT
    bsha = _file_sha(binary)
    head = _git_head()
    run_parts = [int(x) for x in str(parts).split(",")] if isinstance(parts, str) else list(parts)
    plabels, rl, fps = _load_fixtures()
    gates = schema.load_gates()
    knobs = schema.load_knobs()
    report_path = os.path.join(REPORT_DIR, "G-REMEDIES.json")
    report = None
    if os.path.isfile(report_path):
        try:
            with open(report_path, encoding="utf-8") as fh:
                report = json.load(fh)
        except (ValueError, OSError):
            report = None
    if not isinstance(report, dict) or report.get("schema") != GATE_SCHEMA \
            or not isinstance(report.get("parts"), dict):
        report = {"schema": GATE_SCHEMA, "parts": {}}
    report["binary_sha256"] = bsha
    report["git_head"] = head
    report["date"] = schema._now_iso()[:10]
    failed = False
    if 1 in run_parts:
        r = _part1(plabels, rl, fps, gates, knobs)
        report["parts"]["1"] = r
        print("[gate] part 1 %s: %d/%d probes, %d/%d sequences, box_sphere %s, %d bad"
              % (r["verdict"], sum(1 for x in r["probes"] if x["ok"]), len(r["probes"]),
                 sum(1 for x in r["sequences"] if x["ok"]), len(r["sequences"]),
                 r["box_sphere_loop"]["terminal"], r["n_bad"]))
        failed = failed or r["verdict"] != "PASS"
    if 2 in run_parts:
        r = _part2(knobs)
        report["parts"]["2"] = r
        print("[gate] part 2 %s: %d _set calls, %d _rm_* functions, %d violations"
              % (r["verdict"], r["n_set_calls"], len(r["writes"]), len(r["violations"])))
        failed = failed or r["verdict"] != "PASS"
    if 3 in run_parts:
        r = _part3(plabels, rl, fps, gates, knobs)
        report["parts"]["3"] = r
        print("[gate] part 3 %s: %d single steps, %d loops, %d sweep cases, %d violations"
              % (r["verdict"], r["property"]["n"], r["loops"]["n"], r["sweep"]["n"],
                 r["n_bad"]))
        failed = failed or r["verdict"] != "PASS"
    if 4 in run_parts:
        r = _part4(plabels, rl, fps, gates, knobs, out_dir, streams, binary)
        report["parts"]["4"] = r
        print("[gate] part 4 %s: %s" % (r["verdict"], ", ".join(
            "%s %s/%d" % (x["id"], x["terminal"], x["attempts"]) for x in r["probes"])))
        failed = failed or r["verdict"] != "PASS"
    have = sorted(report["parts"])
    all_pass = all(report["parts"][k].get("verdict") == "PASS" for k in have)
    if have == ["1", "2", "3", "4"] and all_pass:
        report["verdict"] = "PASS"
    elif have != ["1", "2", "3", "4"]:
        report["verdict"] = "PARTIAL"
    else:
        report["verdict"] = "FAIL"
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, sort_keys=True, ensure_ascii=False)
    _write_md(report)
    if failed:
        print("G-REMEDIES FAIL: %s" % ", ".join(k for k in have
                                                if report["parts"][k]["verdict"] != "PASS"))
        return 1
    if have == ["1", "2", "3", "4"]:
        print("G-REMEDIES PASS")
    else:
        print("G-REMEDIES PARTIAL: have %s" % ", ".join(have))
    return 0


# --- the CLI ---------------------------------------------------------------------

def _cli_gate(argv):
    out, parts, streams, binary = None, [1, 2, 3, 4], 6, None
    i = 0
    try:
        while i < len(argv):
            a = argv[i]
            if a == "--out":
                out = argv[i + 1]
                i += 2
            elif a == "--parts":
                parts = [int(x) for x in argv[i + 1].split(",")]
                i += 2
            elif a == "--streams":
                streams = int(argv[i + 1])
                i += 2
            elif a == "--binary":
                binary = argv[i + 1]
                i += 2
            else:
                sys.stderr.write("remedies: unknown gate argument %r" % a + chr(10))
                return 2
    except (IndexError, ValueError):
        sys.stderr.write("remedies: a gate argument is missing or not a number" + chr(10))
        return 2
    if out is None:
        sys.stderr.write("remedies: --gate needs --out DIR" + chr(10))
        return 2
    return gate(parts, out, streams=streams, binary=binary)


def _cli_probe(argv):
    as_json = "--json" in argv
    ids = [a for a in argv if a != "--json"]
    if len(ids) != 1:
        sys.stderr.write("remedies: --probe needs exactly one probe id" + chr(10))
        return 2
    try:
        plabels, rl, fps = _load_fixtures()
        ctx = probe_ctx(ids[0], plabels, fps)
        res = propose(ctx, [{"attempt": 1,
                             "config_sha256": schema.canonical_sha256(ctx["config"]),
                             "rule_id": None}])
    except (RemedyError, OSError, KeyError) as e:
        sys.stderr.write("remedies: %s" % e + chr(10))
        return 2
    if as_json:
        print(json.dumps(res, indent=1, ensure_ascii=False))
    print(res["record"]["message"])
    if res["verdict"] == "apply":
        print("REMEDY %s %s" % (res["rule_id"], res["config_sha256"][:12]))
    else:
        print("TERMINAL %s" % res["terminal"])
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        sys.stderr.write("remedies: give --selftest, --gate --out DIR or --probe ID" + chr(10))
        return 2
    if argv[0] == "--selftest":
        return selftest()
    if argv[0] == "--gate":
        return _cli_gate(argv[1:])
    if argv[0] == "--probe":
        return _cli_probe(argv[1:])
    sys.stderr.write("remedies: unknown argument %r" % argv[0] + chr(10))
    return 2


if __name__ == "__main__":
    sys.exit(main())
