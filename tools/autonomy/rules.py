#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""
rules.py - the L1 setup rules: attempt 1 from a geometry's fingerprint and
flow (AM-9, docs/15 §C).

Eight pure rules run in a fixed order and each returns ONE DecisionRecord:
R-YP (the a priori first layer), R-DOM (the domain in L_ref multiples),
R-PLANE (commensurate planar bodies put on cell planes, attraction off,
SPEC-LIT §92.15.5), R-WIN (the coarsest wall level inside the §D.3 y+ window
and the largest growth under the stack limiter), R-CURV (h <= r_p5/8),
R-GAP (h <= gap/3), R-FEAT (feature_level +1 on sharp edges, the attraction
on at tau = h_f / 2: snap.feature_tolerance = 0.5 * 2**-max_level, SPEC-LIT
§92.12's erratum) and R-BUDGET (a predicted cell ladder that coarsens far-field
bands before wall bands).  A rule that refuses sets `stop`; the rules after
it abstain.  On a box the delivered stack needs h <= ~42.9 t1 (the
box-corner G5 bound measured on cubep 2026-09-24), so R-PLANE uses 0.70 of
the §D.3 G5 edge; every other wall keeps the full edge (docs/15 §H, §I-1).

    python tools/autonomy/rules.py --id GEOMETRY_ID --out-dir DIR [--records OUT.jsonl]
    python tools/autonomy/rules.py STL --flow FLOW.json --out CONFIG.json
    python tools/autonomy/rules.py --selftest
    python tools/autonomy/rules.py --gate --out DIR [--parts 1,2,3]
    python tools/autonomy/rules.py --ft-sample [--out IDS.txt]
    python tools/autonomy/rules.py --ft-gate --campaign DIR
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "corpus"))
import schema  # noqa: E402
import score  # noqa: E402
import preflight  # noqa: E402

BINARY_DEFAULT = score.BINARY_DEFAULT
REPORT_DIR = os.path.join(HERE, "rules")
RESULT_SCHEMA = "autonomy-rules/1"
GATE_SCHEMA = "autonomy-rules-gate/1"
RULES = ("R-YP", "R-DOM", "R-PLANE", "R-WIN", "R-CURV", "R-GAP", "R-FEAT", "R-BUDGET")
CELL_FRAC = preflight.MESHER_DEFAULTS["/layers/cell_frac"]        # 0.5: a limiter (§I-5)
G5_RATIO = preflight.REFERENCE_QUALITY["min_thickness_ratio"]     # 0.05
G5_FACTOR = preflight.G5_FACTOR                                   # 3.0 (layers.rs:1306)
EDGE_MARGIN = 1e-6      # both window edges are kept this far inside (float h_i)
CORNER_KAPPA = 0.70     # the box-corner G5 bound, measured 42.9 t1 = 0.715*60 t1 on cubep
FT_HALF_H_F = 0.5       # R-FEAT's attraction radius tau = 0.5 h_f (SPEC-LIT §92.12 erratum)
T1_SIG = 4              # t1 floored to 4 significant digits, so the a priori y+ stays <= 1
GROWTH_STEP_DIV = 1000  # growth steps are k / 1000
BASE_FRAC = 0.5         # base_size = 0.5 L_ref (docs/15 §B's L4 recipe)
DOMAIN_MARGINS = (3.0, 6.0, 2.5, 2.5, 2.5, 2.5)   # x lo, x hi, y lo, y hi, z lo, z hi (L_ref)
BAND_SHAPE = ((0.1, 0), (0.5, 1), (1.5, 2))       # (distance / L_ref, levels below wall)
BAND_MIN_CELLS = 3      # a band is never thinner than 3 cells of its own level
CURV_CELLS = 8          # R-CURV: h <= r_p5 / 8
GAP_CELLS = 3           # R-GAP: h <= gap / 3
PRED_SAFETY = 0.7       # R-BUDGET target = 0.7 * cell_budget
BUDGET_RUNGS = ((1.0, 1.0, True), (0.5, 1.0, True), (0.25, 1.0, True), (0.25, 0.5, True),
                (0.25, 0.25, True), (0.25, 0.25, False))  # (far, wall, feature bump kept)
WORKED_FLOW = {"u_ref_m_s": 0.3, "l_ref_m": 1.0, "nu_m2_s": 1.5e-5}   # Re_L 2e4, §D.3
T1_WORKED = 7.29699e-4
GEN_OF = {"A": "gen_wing", "B": "gen_lathe", "D": "gen_bluff", "E": "gen_gap", "F": "gen_thin"}


class RulesError(ValueError):
    """A caller or harness error (bad argument, invalid record) - never a verdict."""


# --- the small pure helpers (C3) ---------------------------------------------

def t1_floor(t1: float, sig: int = T1_SIG) -> float:
    """The first-layer thickness rounded DOWN to sig significant digits, so
    schema.yplus_a_priori stays <= 1 (the raw t1 can give 1.0000000000000002)."""
    k = sig - 1 - math.floor(math.log10(t1))
    return math.floor(t1 * 10 ** k) / 10 ** k


def stack_total(t1: float, growth: float, n: int) -> float:
    """t1 * (1 + g + ... + g^(n-1)) - the layers.rs:127-133 sum, not the closed form."""
    return sum(t1 * growth ** k for k in range(n))


def fit_growth(t1: float, n: int, h: float, knobs: dict):
    """The largest k/1000 growth whose n-layer stack fits cell_frac*h, or None."""
    row = schema._knob_row("/layers/growth", knobs)
    budget = CELL_FRAC * h * (1 - EDGE_MARGIN)
    best = None
    for k in range(round(row["min"] * GROWTH_STEP_DIV), round(row["max"] * GROWTH_STEP_DIV) + 1):
        g = k / GROWTH_STEP_DIV
        if stack_total(t1, g, n) <= budget:
            best = g
        else:
            break   # the sum grows with g, so stop at the first failure
    return best


def level_for(base: float, h_max: float, cap: int) -> tuple:
    """The smallest L with base / 2**L <= h_max; (cap, True) when none fits."""
    for level in range(cap + 1):
        if base / 2 ** level <= h_max:
            return level, False
    return cap, True


def bands_for(l_ref: float, base: float, wall_level: int, far: float, wall: float) -> list:
    """§B's L4 band shape, scaled: distances are max(shape, 3 cells, previous)."""
    out, prev = [], 0.0
    for frac, below in BAND_SHAPE:
        level = wall_level - below
        if level < 1:
            continue
        f = wall if below == 0 else far
        d = max(frac * l_ref * f, BAND_MIN_CELLS * base / 2 ** level, prev)
        out.append({"distance": d, "level": level})
        prev = d
    return out


def _apply_levels(state: dict, wall_level: int, far: float, wall: float, feature: bool) -> None:
    """The refinement block of one (wall_level, far, wall, feature) rung."""
    config = state["config"]
    fl = min(wall_level + 1, state["cap"]) if feature else None
    bands = bands_for(state["L"], state["base"], wall_level, far, wall)
    if bands == [] and fl is None:
        config["refinement"] = {"levels": [], "max_level": 0}
    else:
        entries = [{"patch": p, "bands": copy.deepcopy(bands)} for p in state["patches"]]
        if fl is not None:
            for e in entries:
                e["feature_level"] = fl
        config["refinement"] = {"levels": entries, "max_level": max(wall_level, fl or 0)}
    state["wall_level"] = wall_level
    state["rung"] = (far, wall)
    state["feature"] = feature
    state["feature_level"] = fl


def _set_growth(state: dict, h: float):
    """The largest growth fitting cell_frac*h, written to the config; None says
    the limiter will trim the stack (the message says so, at row["min"])."""
    g = fit_growth(state["t1"], state["n"], h, state["knobs"])
    row = schema._knob_row("/layers/growth", state["knobs"])
    state["growth"] = g if g is not None else row["min"]
    state["config"]["layers"]["growth"] = state["growth"]
    return g


# --- the state, the edits and the records (C2) -------------------------------

def diff_edits(before: dict, after: dict) -> list:
    """Every leaf-pointer whose value changed, in document order."""
    b = dict(preflight.config_leaves(before))
    a = dict(preflight.config_leaves(after))
    out = []
    for p, v in a.items():
        if p not in b or json.dumps(b[p], sort_keys=True) != json.dumps(v, sort_keys=True):
            out.append({"pointer": p, "from": b.get(p), "to": v})
    for p, v in b.items():
        if p not in a:
            out.append({"pointer": p, "from": v, "to": None})
    return out


def _rec(rule_id, verdict, trigger, inputs, formula, edits, cite, message):
    """One validated DecisionRecord of layer `rule`; a bug raises RulesError."""
    rec = {"schema": "autonomy-decision/1", "layer": "rule", "rule_id": rule_id,
           "verdict": verdict, "trigger": trigger, "inputs": inputs, "formula": formula,
           "edits": edits, "cite": cite, "message": message, "uncertainty": 0,
           "t": schema._now_iso()}
    errs = schema.errors(rec, "DecisionRecord")
    if errs:
        raise RulesError("%s produced an invalid DecisionRecord: %s" % (rule_id, errs[0]))
    return rec


def _stopped(rec_id: str, stop_id: str) -> dict:
    """The abstention a rule returns once an earlier rule has refused."""
    return _rec(rec_id, "abstain", None, [], "", [],
                "docs/15 §C L1", "%s: not run: %s refused" % (rec_id, stop_id))


def _edit_records(state: dict) -> list:
    """diff_edits over a deep copy the rule took before it acted."""
    return diff_edits(state["_before"], state["config"])


# --- R-YP and R-DOM (C4) ------------------------------------------------------

def r_yp(state: dict) -> dict:
    """The a priori first layer: t1 = y+ * nu / u_tau, floored to 4 sig digits."""
    flow = state["flow"]
    bad = schema.errors(flow, "FlowSpec")
    if bad:
        return _rec("R-YP", "refuse",
                    {"observable": "flow", "value": flow, "threshold": "FlowSpec",
                     "op": "==", "source": "the manifest row"}, [], "", [],
                    "docs/15 §C L1; docs/15 §D.3",
                    "R-YP: the flow spec is invalid: %s" % bad[0])
    w = schema.a_priori_wall(flow, state["gates"]["yplus_max_a_priori"])
    t1_raw, re_l = w["t1_a_priori_m"], w["re_l"]
    if t1_raw is None:
        return _rec("R-YP", "refuse",
                    {"observable": "flow.re_l", "value": re_l, "threshold": 0.0,
                     "op": ">", "source": "the manifest row's flow (docs/15 §D.3)"},
                    [{"name": "re_l", "value": re_l, "unit": "1"}],
                    "Re_L = U*L/nu; C_F = 1.328/sqrt(Re_L) (Re_L < 5e5) else "
                    "0.455/log10(Re_L)**2.58; u_tau = U*sqrt(C_F/2); "
                    "t1 = y+ * nu / u_tau, floored to 4 significant digits",
                    [], "docs/15 §D.3 (R-YP)",
                    "R-YP: Re_L = %.6g has no skin-friction correlation" % re_l)
    t1 = t1_floor(t1_raw)
    yp = schema.yplus_a_priori(t1, flow)
    if not yp <= state["gates"]["yplus_max_a_priori"]:
        raise RulesError("R-YP: the floored t1 %r gives a priori y+ %r > %g"
                         % (t1, yp, state["gates"]["yplus_max_a_priori"]))
    n = state["gates"]["delivered_min_layers"]
    patches = state["patches"]
    state["t1"], state["n"] = t1, n
    state["config"]["layers"] = {"patches": list(patches), "n": n,
                                 "first_thickness": t1}
    edits = _edit_records(state)
    laminar = "laminar, Re_L < 5e5" if re_l < 5e5 else "turbulent, Re_L >= 5e5"
    return _rec("R-YP", "apply",
                {"observable": "flow.re_l", "value": re_l, "threshold": 5e5,
                 "op": "<" if re_l < 5e5 else ">=",
                 "source": "the manifest row's flow (docs/15 §D.3)"},
                [{"name": "u_ref", "value": flow["u_ref_m_s"], "unit": "m/s"},
                 {"name": "l_ref", "value": flow["l_ref_m"], "unit": "m"},
                 {"name": "nu", "value": flow["nu_m2_s"], "unit": "m2/s"},
                 {"name": "re_l", "value": re_l, "unit": "1"},
                 {"name": "cf_a_priori", "value": w["cf_a_priori"], "unit": "1"},
                 {"name": "u_tau_a_priori", "value": w["u_tau_a_priori_m_s"], "unit": "m/s"},
                 {"name": "t1_raw", "value": t1_raw, "unit": "m"},
                 {"name": "t1", "value": t1, "unit": "m"},
                 {"name": "yplus_a_priori", "value": yp, "unit": "1"},
                 {"name": "n", "value": n, "unit": "1"}],
                "Re_L = U*L/nu; C_F = 1.328/sqrt(Re_L) (Re_L < 5e5) else "
                "0.455/log10(Re_L)**2.58; u_tau = U*sqrt(C_F/2); "
                "t1 = y+ * nu / u_tau, floored to 4 significant digits",
                edits, schema.FLAT_PLATE_CITE + "; docs/15 §D.3 (R-YP)",
                "R-YP: Re_L = %.6g, C_F = %.6g (%s), u_tau = %.6g m/s; "
                "t1 = y+ * nu / u_tau = %.6g m, floored to %.4g m "
                "(a priori y+ = %.6f <= %g); %d layers on %s"
                % (re_l, w["cf_a_priori"], laminar, w["u_tau_a_priori_m_s"], t1_raw,
                   t1, yp, state["gates"]["yplus_max_a_priori"], n, ", ".join(patches)))


def r_dom(state: dict) -> dict:
    """The domain: bbox + L_ref margins, floored/ceiled onto base_size multiples."""
    L = state["L"]
    base = round(BASE_FRAC * L, 4) + 0.0
    if base <= 0:
        base = BASE_FRAC * L
    bb = state["fp"]["bbox"]
    mg = [f * L for f in DOMAIN_MARGINS]
    ext = []
    n_cell = 1
    for a in range(3):
        lo = round(math.floor((bb[2 * a] - mg[2 * a]) / base) * base, 10) + 0.0
        hi = round(math.ceil((bb[2 * a + 1] + mg[2 * a + 1]) / base) * base, 10) + 0.0
        ext += [lo, hi]
        n_cell *= max(1, round((hi - lo) / base))
    state["base"] = base
    state["config"]["domain"] = {"extent": ext, "base_size": base}
    edits = _edit_records(state)
    return _rec("R-DOM", "apply",
                {"observable": "fingerprint.bbox", "value": bb, "threshold": mg,
                 "op": ">=", "source": "features.py bbox; docs/15 §C R-DOM"},
                [{"name": "l_ref", "value": L, "unit": "m"},
                 {"name": "base_size", "value": base, "unit": "m"},
                 {"name": "bbox", "value": bb, "unit": "m"},
                 {"name": "margins", "value": mg, "unit": "m"},
                 {"name": "n_base_cells", "value": n_cell, "unit": "cells"}],
                "base = 0.5 L_ref (4 decimals); extent = bbox - [3, 2.5, 2.5] L_ref "
                "and + [6, 2.5, 2.5] L_ref, floored/ceiled to multiples of base",
                edits, "docs/15 §C L1 R-DOM; docs/15 §B L4 recipe; automesher "
                "octree.rs:446 (n_a = round((hi - lo)/base_size))",
                "R-DOM: base_size = 0.5 * L_ref = %.4g m; domain = bbox + 3 L_ref "
                "upstream, 6 L_ref downstream and 2.5 L_ref on each side, on "
                "multiples of base_size: %d x %d x %d = %d base cells"
                % (base, round((ext[1] - ext[0]) / base), round((ext[3] - ext[2]) / base),
                   round((ext[5] - ext[4]) / base), n_cell))


# --- R-PLANE (C5) -------------------------------------------------------------

def r_plane(state: dict) -> dict:
    """Commensurate planar bodies: h = s/m on a cell plane, attraction off."""
    fp = state["fp"]
    s = fp.get("lattice_base_size_m")
    if fp.get("commensurate") is not True \
            or not (isinstance(s, (int, float)) and not isinstance(s, bool) and s > 0):
        return _rec("R-PLANE", "pass",
                    {"observable": "fingerprint.commensurate",
                     "value": fp.get("commensurate"), "threshold": True, "op": "==",
                     "source": "features.py"}, [], "", [], "docs/15 §C L1 R-PLANE",
                    "R-PLANE: the body is not commensurate with an octree lattice "
                    "(features.py): its walls will be snapped")
    t1, n = state["t1"], state["n"]
    s = float(s)
    hi_k = CORNER_KAPPA * G5_FACTOR * t1 / G5_RATIO * (1 - EDGE_MARGIN)
    h_max = hi_k
    og = fp.get("outer_gap_m")
    if isinstance(og, (int, float)) and not isinstance(og, bool):
        h_max = min(h_max, og / GAP_CELLS)
    rp = fp.get("curvature_radius_p5_m")
    if isinstance(rp, (int, float)) and not isinstance(rp, bool):
        h_max = min(h_max, rp / CURV_CELLS)
    g = state.get("requested_growth")
    lo1 = (stack_total(t1, g, n) if g is not None else n * t1) \
        / CELL_FRAC * (1 + EDGE_MARGIN)
    m = max(1, math.ceil(s / h_max))
    while s / m > h_max:
        m += 1
    h = s / m
    if h < lo1:
        return _rec("R-PLANE", "abstain", None,
                    [{"name": "s", "value": s, "unit": "m"},
                     {"name": "m", "value": m, "unit": "1"},
                     {"name": "h", "value": h, "unit": "m"},
                     {"name": "h_over_t1", "value": h / t1, "unit": "1"},
                     {"name": "lo", "value": lo1, "unit": "m"}],
                    "h = s/m, m = ceil(s/h_max), h_max = min(0.70*3*t1/"
                    "min_thickness_ratio, gap/3, r_p5/8)",
                    [], "docs/15 §C L1 R-PLANE; SPEC-LIT §92.15.5",
                    "R-PLANE: the lattice spacing s = %.6g m has no divisor s/m in "
                    "the delivery window [%.4g, %.4g] m (s/%d = %.4g m, h/t1 = %.2f "
                    "< %.2f): the body is meshed as a snapped one"
                    % (s, lo1, h_max, m, h, h / t1, lo1 / t1))
    L = state["L"]
    level = min(state["cap"], max(0, round(math.log2(BASE_FRAC * L / h))))
    base = h * 2 ** level
    bb = fp["bbox"]
    mg = [f * L for f in DOMAIN_MARGINS]
    ext = []
    for a in range(3):
        lo = bb[2 * a] - math.ceil(mg[2 * a] / base) * base
        k = math.ceil((bb[2 * a + 1] + mg[2 * a + 1] - lo) / base)
        ext += [lo, lo + k * base]
    state["base"] = base
    state["plane"] = {"s": s, "m": m, "h": h, "level": level, "h_max": h_max}
    state["config"]["domain"] = {"extent": ext, "base_size": base}
    state["config"]["snap"] = {"feature_tolerance": 0.0, "smoothing_passes": 0}
    edits = _edit_records(state)
    return _rec("R-PLANE", "apply",
                {"observable": "fingerprint.lattice_base_size_m / m", "value": h,
                 "threshold": [lo1, h_max], "op": "in",
                 "source": "features.py lattice; docs/15 §C R-PLANE"},
                [{"name": "s", "value": s, "unit": "m"},
                 {"name": "m", "value": m, "unit": "1"},
                 {"name": "h", "value": h, "unit": "m"},
                 {"name": "h_over_t1", "value": h / t1, "unit": "1"},
                 {"name": "kappa", "value": CORNER_KAPPA, "unit": "1"},
                 {"name": "h_max", "value": h_max, "unit": "m"},
                 {"name": "lo", "value": lo1, "unit": "m"},
                 {"name": "wall_level", "value": level, "unit": "1"},
                 {"name": "base_size", "value": base, "unit": "m"}],
                "h = s/m, m = ceil(s/h_max), h_max = min(0.70*3*t1/"
                "min_thickness_ratio, gap/3, r_p5/8); level = round(log2(0.5 "
                "L_ref / h)); base = h * 2**level; extent_lo = bbox_lo - "
                "ceil(margin/base)*base",
                edits, "docs/15 §C L1 R-PLANE; SPEC-LIT §92.15.5; features.py "
                "lattice_base_size_m; the box-corner G5 bound measured on cubep "
                "2026-09-24 (docs/15 §K)",
                "R-PLANE: the body is commensurate with lattice spacing s = %.6g m; "
                "h = s/%d = %.6g m (h/t1 = %.2f <= %.2f, 0.70 of the G5 edge: the "
                "box-corner bound); wall level %d at base_size %.6g m (%.3g L_ref); "
                "the extent starts on the body's own faces, so every face lies on a "
                "cell plane; snap.feature_tolerance = 0 and snap.smoothing_passes = 0 "
                "(SPEC-LIT §92.15.5): the edges already lie on lattice lines, so the "
                "attraction has nothing to do"
                % (s, m, h, h / t1, h_max / t1, level, base, base / L))


# --- the §D.3 window and R-WIN (C6) -------------------------------------------

def window_level(t1, n, base, cap, *, kappa=1.0, growth=None, h_fixed=None,
                 knobs=None) -> dict:
    """The §D.3 feasible window: the stack limiter below, the G5 edge above;
    the coarsest landing level, or h_fixed kept (the R-PLANE path)."""
    hi = kappa * G5_FACTOR * t1 / G5_RATIO * (1 - EDGE_MARGIN)
    s = n if growth is None else stack_total(1.0, growth, n)
    lo = t1 * s / CELL_FRAC * (1 + EDGE_MARGIN)
    out = {"verdict": "apply", "reason": None, "lo": lo, "hi": hi, "level": None,
           "h": None, "growth": None, "fit": None, "T": None}
    if lo > hi:
        out["verdict"], out["reason"] = "refuse", "empty"
        return out
    if h_fixed is not None:
        if not (lo <= h_fixed <= hi * (1 + 1e-12)):
            out["verdict"], out["reason"] = "refuse", "none"
            return out
        h = h_fixed
    else:
        h, level = None, None
        for lv in range(cap + 1):
            cand = base / 2 ** lv
            if lo <= cand <= hi:
                h, level = cand, lv
                break
        if level is None:
            out["verdict"] = "refuse"
            out["reason"] = "fine" if base / 2 ** cap > hi else ("coarse" if base < lo else "none")
            return out
        out["level"] = level
    fit = fit_growth(t1, n, h, knobs) if growth is None else growth
    out["h"] = h
    out["fit"] = fit
    out["growth"] = fit if fit is not None else 1.0
    out["T"] = stack_total(t1, out["growth"], n)
    return out


_WIN_REFUSE = {
 "empty": "R-WIN: the §D.3 window is empty at n = %d, g = %g: T/cell_frac = %.4g m "
          "> 3*t1/min_thickness_ratio = %.4g m (t1 = %.4g m)",
 "fine": "R-WIN: y+ <= 1 needs h_wall <= %.4g mm, the finest reachable is %.4g mm "
         "at max_level %d",
 "coarse": "R-WIN: the stack needs h_wall >= %.4g mm, the coarsest reachable is "
           "%.4g mm at level 0",
 "none": "R-WIN: no level 0..%d lands in [%.4g, %.4g] mm",
}


def r_win(state: dict) -> dict:
    """The coarsest wall level inside the §D.3 window and the largest growth."""
    t1, n, base, cap = state["t1"], state["n"], state["base"], state["cap"]
    plane = state["plane"]
    kw = dict(knobs=state["knobs"], growth=state.get("requested_growth"))
    if plane:
        w = window_level(t1, n, base, cap, kappa=CORNER_KAPPA, h_fixed=plane["h"], **kw)
    else:
        w = window_level(t1, n, base, cap, kappa=1.0, **kw)
    if w["verdict"] == "refuse":
        state["stop"] = "R-WIN"
        reason = w["reason"]
        g_arg = state.get("requested_growth")
        if reason == "empty":
            msg = _WIN_REFUSE["empty"] % (n, g_arg if g_arg is not None else 1.0,
                                          w["lo"], w["hi"], t1)
        elif reason == "fine":
            msg = _WIN_REFUSE["fine"] % (w["hi"] * 1000.0, base / 2 ** cap * 1000.0, cap)
        elif reason == "coarse":
            msg = _WIN_REFUSE["coarse"] % (w["lo"] * 1000.0, base * 1000.0)
        else:
            msg = _WIN_REFUSE["none"] % (cap, w["lo"] * 1000.0, w["hi"] * 1000.0)
        trig = {"observable": "/layers/growth" if reason == "empty" else "/domain/base_size",
                "value": (g_arg or base), "threshold": [w["lo"], w["hi"]], "op": "in",
                "source": "docs/15 §D.3"}
        return _rec("R-WIN", "refuse", trig,
                    [{"name": "t1", "value": t1, "unit": "m"},
                     {"name": "n", "value": n, "unit": "1"},
                     {"name": "base_size", "value": base, "unit": "m"},
                     {"name": "kappa", "value": CORNER_KAPPA if plane else 1.0, "unit": "1"},
                     {"name": "lo", "value": w["lo"], "unit": "m"},
                     {"name": "hi", "value": w["hi"], "unit": "m"}],
                    "lo = t1*S(g)/cell_frac with S(g) = sum g^k, k < n (g -> 1: S = n); "
                    "hi = kappa*3*t1/min_thickness_ratio; the coarsest level with "
                    "lo <= base/2**L <= hi; growth = the largest k/1000 with "
                    "t1*S(g) <= cell_frac*h",
                    [], "docs/15 §D.3 (R-WIN); layers.rs:127-133 (the stack), "
                    "layers.rs:449 (t_i = min(T, medial, cell_frac*h_i)), "
                    "layers.rs:1292-1316 (92.51)", msg)
    level = plane["level"] if plane else w["level"]
    state["win_level"] = level
    _apply_levels(state, level, 1.0, 1.0, False)
    state["config"]["layers"]["growth"] = w["growth"]
    state["growth"] = w["growth"]
    edits = _edit_records(state)
    extra = ", times 0.70 for the box corners" if plane else ""
    return _rec("R-WIN", "apply",
                {"observable": "h_wall / t1", "value": w["h"] / t1,
                 "threshold": [w["lo"] / t1, w["hi"] / t1], "op": "in",
                 "source": "docs/15 §D.3"},
                [{"name": "t1", "value": t1, "unit": "m"},
                 {"name": "n", "value": n, "unit": "1"},
                 {"name": "cell_frac", "value": CELL_FRAC, "unit": "1"},
                 {"name": "min_thickness_ratio", "value": G5_RATIO, "unit": "1"},
                 {"name": "kappa", "value": CORNER_KAPPA if plane else 1.0, "unit": "1"},
                 {"name": "lo", "value": w["lo"], "unit": "m"},
                 {"name": "hi", "value": w["hi"], "unit": "m"},
                 {"name": "wall_level", "value": level, "unit": "1"},
                 {"name": "h", "value": w["h"], "unit": "m"},
                 {"name": "h_over_t1", "value": w["h"] / t1, "unit": "1"},
                 {"name": "growth", "value": w["growth"], "unit": "1"},
                 {"name": "T", "value": w["T"], "unit": "m"},
                 {"name": "cell_frac_h", "value": CELL_FRAC * w["h"], "unit": "m"}],
                "lo = t1*S(g)/cell_frac with S(g) = sum g^k, k < n (g -> 1: S = n); "
                "hi = kappa*3*t1/min_thickness_ratio; the coarsest level with "
                "lo <= base/2**L <= hi; growth = the largest k/1000 with "
                "t1*S(g) <= cell_frac*h",
                edits, "docs/15 §D.3 (R-WIN); layers.rs:127-133 (the stack), "
                "layers.rs:449 (t_i = min(T, medial, cell_frac*h_i)), "
                "layers.rs:1292-1316 (92.51)",
                "R-WIN: at n = %d the window is h in [%.4g, %.4g] m (h/t1 in "
                "[%.2f, %.2f]: the stack limiter T <= cell_frac*h below, the G5 edge "
                "3*t1/h >= %g above%s); wall level %d gives h = %.6g m (h/t1 = %.2f); "
                "growth %.3f is the largest step with T = %.6g m <= cell_frac*h = %.6g m"
                % (n, w["lo"], w["hi"], w["lo"] / t1, w["hi"] / t1, G5_RATIO, extra,
                   level, w["h"], w["h"] / t1, w["growth"], w["T"], CELL_FRAC * w["h"]))


# --- R-CURV, R-GAP, R-FEAT (C7) -----------------------------------------------

def _curv_gap(state: dict, rid: str, key: str, cells: int, word: str, noun: str,
              none_noun: str, cite: str, observable: str) -> dict:
    """The shared shape of R-CURV (h <= r/8) and R-GAP (h <= gap/3)."""
    if state["plane"]:
        return _rec(rid, "abstain", None, [], "", [], cite,
                    "%s: R-PLANE owns the wall level (it bounds h by %s/%d already)"
                    % (rid, word, cells))
    fp = state["fp"]
    r = fp.get(key)
    if not (isinstance(r, (int, float)) and not isinstance(r, bool)):
        return _rec(rid, "pass",
                    {"observable": observable, "value": None, "threshold": None,
                     "op": "==", "source": "features.py"}, [], "", [],
                    cite, "%s: no %s (features.py)" % (rid, none_noun))
    base = state["base"]
    h_max = r / cells
    h = base / 2 ** state["wall_level"]
    formula = "h <= %s / %d: the smallest L with base/2**L <= %s/%d, capped at max_level" \
        % (word, cells, word, cells)
    if h <= h_max:
        return _rec(rid, "pass",
                    {"observable": observable, "value": h_max, "threshold": h,
                     "op": "<", "source": "features.py; docs/15 §C " + rid},
                    [{"name": word, "value": r, "unit": "m"},
                     {"name": "h_max", "value": h_max, "unit": "m"},
                     {"name": "h", "value": h, "unit": "m"}],
                    formula, [], cite,
                    "%s: h = %.4g m <= %s/%d = %.4g m already"
                    % (rid, h, word, cells, h_max))
    L, capped = level_for(base, h_max, state["cap"])
    if L <= state["wall_level"]:
        return _rec(rid, "pass",
                    {"observable": observable, "value": h_max, "threshold": h,
                     "op": "<", "source": "features.py; docs/15 §C " + rid},
                    [{"name": word, "value": r, "unit": "m"},
                     {"name": "h_max", "value": h_max, "unit": "m"},
                     {"name": "h", "value": h, "unit": "m"},
                     {"name": "capped", "value": 1, "unit": "1"}],
                    formula, [], cite,
                    "%s: h = %.4g m > %s/%d = %.4g m, but the wall level is already "
                    "at max_level %d" % (rid, h, word, cells, h_max, state["wall_level"]))
    return _curv_gap_apply(state, rid, r, h_max, L, capped, base, h, cells, word,
                           noun, cite, observable, formula)


def _curv_gap_apply(state: dict, rid, r, h_max, L, capped, base, h, cells, word,
                    noun, cite, observable, formula):
    """The apply arm of R-CURV / R-GAP: raise the wall level, refit the growth."""
    t1 = state["t1"]
    before_level = state["wall_level"]
    _apply_levels(state, L, *state["rung"], state["feature"])
    fit = _set_growth(state, base / 2 ** L)
    edits = _edit_records(state)
    tail = (" (capped at max_level %d: %s/h = %.2f < %d)"
            % (state["cap"], word, r / (base / 2 ** L), cells)) if capped else ""
    if fit is None:
        g_txt = "%.3f" % state["growth"]
        tail2 = ("; no growth fits cell_frac*h at this level (n*t1 = %.4g m > %.4g m): "
                 "the cell_frac limiter will trim the stack"
                 % (state["n"] * t1, CELL_FRAC * base / 2 ** L))
    else:
        g_txt, tail2 = "%.3f" % fit, ""
    return _rec(rid, "apply",
                {"observable": observable, "value": h_max, "threshold": h,
                 "op": "<", "source": "features.py; docs/15 §C " + rid},
                [{"name": word, "value": r, "unit": "m"},
                 {"name": "h_max", "value": h_max, "unit": "m"},
                 {"name": "h_before", "value": h, "unit": "m"},
                 {"name": "h_after", "value": base / 2 ** L, "unit": "m"},
                 {"name": "wall_level_before", "value": before_level, "unit": "1"},
                 {"name": "wall_level_after", "value": L, "unit": "1"},
                 {"name": "capped", "value": int(capped), "unit": "1"},
                 {"name": "growth", "value": state["growth"], "unit": "1"}],
                formula, edits, cite,
                "%s: the %s is %.4g m, so h <= %s/%d = %.4g m; wall level %d -> %d "
                "(h %.4g -> %.4g m)%s; growth %s%s"
                % (rid, noun, r, word, cells, h_max, before_level, L, h,
                   base / 2 ** L, tail, g_txt, tail2))


def r_curv(state: dict) -> dict:
    """h <= r_p5 / 8 near small curvature radii (features.py's p5)."""
    return _curv_gap(state, "R-CURV", "curvature_radius_p5_m", CURV_CELLS, "r",
                     "5th-percentile curvature radius", "curved vertex",
                     "docs/15 §C L1 R-CURV; features.py curvature_radius_p5_m",
                     "fingerprint.curvature_radius_p5_m / 8")


def r_gap(state: dict) -> dict:
    """h <= outer_gap / 3 so the two bodies' layers do not close the gap."""
    return _curv_gap(state, "R-GAP", "outer_gap_m", GAP_CELLS, "gap", "outer gap",
                     "outer gap", "docs/15 §C L1 R-GAP; features.py outer_gap_m",
                     "fingerprint.outer_gap_m / 3")


def r_feat(state: dict) -> dict:
    """feature_level = wall level + 1 on sharp edges; the attraction stays on."""
    if state["plane"]:
        return _rec("R-FEAT", "abstain", None, [], "", [],
                    "docs/15 §C L1 R-FEAT; docs/15 §K G-PILOT caution 1",
                    "R-FEAT: R-PLANE owns this body: its edges lie on lattice lines "
                    "and the attraction is off")
    fp = state["fp"]
    sel = fp["sharp_edge_length_m"]
    fa = fp.get("feature_angle_deg", 30.0)
    if sel == 0:
        return _rec("R-FEAT", "pass",
                    {"observable": "fingerprint.sharp_edge_length_m", "value": 0.0,
                     "threshold": 0.0, "op": ">", "source": "features.py"},
                    [{"name": "sharp_edge_length_m", "value": 0.0, "unit": "m"}],
                    "feature_level = wall level + 1 when sharp_edge_length_m > 0",
                    [], "docs/15 §C L1 R-FEAT; features.py sharp_edge_length_m",
                    "R-FEAT: no sharp edge at %g deg" % fa)
    wall_level = state["wall_level"]
    cap = state["cap"]
    fl = min(wall_level + 1, cap)
    _apply_levels(state, wall_level, *state["rung"], True)
    if state["ft_radius"]:
        return _feat_radius(state, sel, fa, wall_level, fl, cap)
    edits = _edit_records(state)
    cap_txt = (" (capped at max_level %d)" % cap) if wall_level + 1 > cap else ""
    return _rec("R-FEAT", "apply",
                {"observable": "fingerprint.sharp_edge_length_m", "value": sel,
                 "threshold": 0.0, "op": ">", "source": "features.py"},
                [{"name": "sharp_edge_length_m", "value": sel, "unit": "m"},
                 {"name": "feature_angle_deg", "value": fa, "unit": "deg"},
                 {"name": "wall_level", "value": wall_level, "unit": "1"},
                 {"name": "feature_level", "value": fl, "unit": "1"}],
                "feature_level = wall level + 1 when sharp_edge_length_m > 0",
                edits, "docs/15 §C L1 R-FEAT; docs/15 §K G-PILOT caution 1; "
                "features.py sharp_edge_length_m",
                "R-FEAT: %.4g m of sharp edge at %g deg, so feature_level = "
                "wall level + 1 = %d%s. snap.feature_tolerance stays at its default "
                "0.5: 0 would switch the feature attraction off and leave the edges "
                "unsnapped (docs/15 §K G-PILOT caution 1; G-FID guards it)"
                % (sel, fa, fl, cap_txt))


def _feat_radius(state: dict, sel, fa, wall_level: int, fl: int, cap: int) -> dict:
    """R-FEAT's apply since FT-RADIUS: the feature bump, then the attraction radius
    tau = h_f / 2 (snap.feature_tolerance = 0.5 * 2**-max_level; SPEC-LIT §92.12's
    erratum: the default 0.5 is half a BASE cell and pins a refined sharp body)."""
    ml = state["config"]["refinement"]["max_level"]
    ft = FT_HALF_H_F * 2.0 ** -ml
    tau = ft * state["base"]
    state["config"].setdefault("snap", {})["feature_tolerance"] = ft
    state["ft_set"] = True
    edits = _edit_records(state)
    cap_txt = (" (capped at max_level %d)" % cap) if wall_level + 1 > cap else ""
    return _rec("R-FEAT", "apply",
                {"observable": "fingerprint.sharp_edge_length_m", "value": sel,
                 "threshold": 0.0, "op": ">", "source": "features.py"},
                [{"name": "sharp_edge_length_m", "value": sel, "unit": "m"},
                 {"name": "feature_angle_deg", "value": fa, "unit": "deg"},
                 {"name": "wall_level", "value": wall_level, "unit": "1"},
                 {"name": "feature_level", "value": fl, "unit": "1"},
                 {"name": "max_level", "value": ml, "unit": "1"},
                 {"name": "feature_tolerance", "value": ft, "unit": "1"},
                 {"name": "tau_m", "value": tau, "unit": "m"}],
                "feature_level = wall level + 1 when sharp_edge_length_m > 0; "
                "snap.feature_tolerance = 0.5 * 2**-max_level, so tau = "
                "feature_tolerance * base_size = h_f / 2",
                edits, "docs/15 §C L1 R-FEAT; docs/15 §K G-PILOT caution 1; "
                "features.py sharp_edge_length_m; SPEC-LIT §92.12 (92.38) erratum 2026-09-26",
                "R-FEAT: %.4g m of sharp edge at %g deg, so feature_level = wall level + 1 "
                "= %d%s, and snap.feature_tolerance = 0.5 * 2^-%d = %.6g: the attraction "
                "radius tau = %.4g m is half a cell of level %d. The default 0.5 is half a "
                "BASE cell, 2^(L-1) wall cells at level L, and pins the points of a sharp "
                "body (SPEC-LIT §92.12 erratum); 0 would switch the attraction off "
                "(WL-SHARP-FT0)" % (sel, fa, fl, cap_txt, ml, ft, tau, ml))


# --- the cell predictor and R-BUDGET (C8) -------------------------------------

def predict_cells(config: dict, fingerprint: dict, n: int) -> dict:
    """The (C8) prediction: base blocks, one parent-padded shell per level, the
    feature cells along the sharp edges and n * A / h_wall^2 layer cells."""
    dom = config["domain"]; base = dom["base_size"]; ext = dom["extent"]
    n_base = 1
    for a in range(3):
        n_base *= max(1, round((ext[2 * a + 1] - ext[2 * a]) / base))
    bb = fingerprint["bbox"]; dims = [bb[1] - bb[0], bb[3] - bb[2], bb[5] - bb[4]]
    area = fingerprint["area_m2"]
    ref = config.get("refinement", {}); ml = ref.get("max_level", 2)
    bands = [b for e in ref.get("levels", []) for b in e.get("bands", [])]
    top = min(max([b["level"] for b in bands] + [0]), ml)
    leaves = float(n_base)
    for l in range(1, top + 1):
        d = max(b["distance"] for b in bands if b["level"] >= l) + base / 2 ** (l - 1)
        vol = min((dims[0] + 2 * d) * (dims[1] + 2 * d) * (dims[2] + 2 * d),
                  2 * area * d + 4.0 / 3.0 * math.pi * d ** 3)
        leaves += 7.0 / 8.0 * vol / (base / 2 ** l) ** 3
    fl = max([e.get("feature_level", 0) for e in ref.get("levels", [])] + [0])
    feature = 0.0
    sel = fingerprint["sharp_edge_length_m"]
    if sel > 0:
        for l in range(top + 1, min(fl, ml) + 1):
            feature += 7.0 / 8.0 * sel * math.pi * (3 * base / 2 ** (l - 1)) ** 2 \
                / (base / 2 ** l) ** 3
    layer_cells = n * area / (base / 2 ** top) ** 2 if n else 0.0
    return {"n_base": n_base, "leaves": leaves, "feature": feature,
            "layer_cells": layer_cells,
            "total": leaves + feature + layer_cells}


_BUDGET_CITE = ("docs/15 §C L1 R-BUDGET; gates.json cell_budget; docs/15 §D.1 F5; "
                "predict_cells calibrated on 11 octree probes 2026-09-24 "
                "(predicted/measured leaves 0.795-1.523)")
_BUDGET_FORMULA = ("leaves = n_base + sum_l 7/8 * V_l / h_l^3, V_l = min(offset bbox, "
                   "2*A*d_l + 4/3*pi*d_l^3), d_l = the widest band at level >= l + one "
                   "parent cell; + feature cells along the sharp edges; + n * A / "
                   "h_wall^2 layer cells")


def r_budget(state: dict) -> dict:
    """The predicted ladder: far-field bands first, then the wall band, then the
    feature bump, then the wall level - predicted, never probed."""
    target = PRED_SAFETY * state["gates"]["cell_budget"]
    floor = state["plane"]["level"] if state["plane"] else state["win_level"]
    base = state["base"]
    entry_wall = state["wall_level"]
    tried, first_p, last_p, chosen = [], None, None, None
    for lw in range(entry_wall, floor - 1, -1):
        for far, wall, keep in BUDGET_RUNGS:
            if not keep and not state["feature"]:
                continue
            feat = state["feature"] and keep
            cand = copy.deepcopy(state)
            _apply_levels(cand, lw, far, wall, feat)
            if lw != entry_wall:
                _set_growth(cand, base / 2 ** lw)
            p = predict_cells(cand["config"], state["fp"], state["n"])
            tried.append({"wall_level": lw, "far": far, "wall": wall,
                          "feature": feat, "predicted": round(p["total"])})
            first_p = first_p if first_p is not None else p
            last_p = p
            if p["total"] <= target:
                chosen = (cand, p, lw, far, wall, feat)
                break
        if chosen is not None:
            break
    trig = {"observable": "predicted_cells", "value": tried[0]["predicted"],
            "threshold": target, "op": "<=",
            "source": "rules.predict_cells (docs/15 §C R-BUDGET)"}
    base_in = [{"name": "cell_budget", "value": state["gates"]["cell_budget"],
                "unit": "cells"},
               {"name": "target", "value": target, "unit": "cells"},
               {"name": "ladder", "value": tried, "unit": "cells"}]
    if chosen is None:
        state["stop"] = "R-BUDGET"
        state["predicted"] = last_p
        return _rec("R-BUDGET", "refuse", trig,
                    base_in + [{"name": "predicted", "value": tried[-1]["predicted"],
                                "unit": "cells"}], _BUDGET_FORMULA, [], _BUDGET_CITE,
                    "R-BUDGET: even the leanest rung at R-WIN's wall level %d predicts "
                    "%d cells > 0.7 * cell_budget = %d: no config keeps y+ <= 1 within "
                    "the cell budget (docs/15 §D.1 F5)"
                    % (floor, tried[-1]["predicted"], round(target)))
    cand, p, lw, far, wall, feat = chosen
    state["predicted"] = p
    pred_in = base_in + [{"name": "predicted", "value": round(p["total"]),
                          "unit": "cells"}]
    if len(tried) == 1:
        return _rec("R-BUDGET", "pass", trig, pred_in, _BUDGET_FORMULA,
                    _edit_records(state), _BUDGET_CITE,
                    "R-BUDGET: the predicted %d cells (%d octree leaves, %d feature, "
                    "%d layer cells) fit 0.7 * cell_budget = %d; nothing is coarsened"
                    % (round(p["total"]), round(p["leaves"]), round(p["feature"]),
                       round(p["layer_cells"]), round(target)))
    ft_txt = ""
    if state["ft_set"]:
        ml2 = cand["config"]["refinement"]["max_level"]
        ft2 = FT_HALF_H_F * 2.0 ** -ml2
        cand["config"]["snap"]["feature_tolerance"] = ft2
        if ml2 != state["config"]["refinement"]["max_level"]:
            ft_txt = ("; snap.feature_tolerance follows max_level %d: 0.5 * 2^-%d = %.6g "
                      "(tau = h_f / 2)" % (ml2, ml2, ft2))
    state["config"] = cand["config"]
    state["wall_level"] = cand["wall_level"]
    state["rung"] = cand["rung"]
    state["feature"] = cand["feature"]
    state["feature_level"] = cand["feature_level"]
    state["growth"] = cand["growth"]
    suffix = ("; the wall level is back at R-WIN's %d" % floor) \
        if lw == floor and lw != entry_wall else ""
    suffix += ft_txt
    return _rec("R-BUDGET", "apply", trig, pred_in, _BUDGET_FORMULA,
                _edit_records(state), _BUDGET_CITE,
                "R-BUDGET: the rules' bands predict %d cells > %d; %d rung(s) later "
                "(far-field bands first, then the wall band, then the feature bump, "
                "then the wall level) wall level %d, far x%g, wall x%g, feature bump "
                "%s predicts %d cells <= %d%s"
                % (tried[0]["predicted"], round(target), len(tried) - 1, lw, far,
                   wall, "kept" if feat else "dropped", round(p["total"]),
                   round(target), suffix))


# --- setup (C9) ----------------------------------------------------------------

RULE_FN = {"R-YP": r_yp, "R-DOM": r_dom, "R-PLANE": r_plane, "R-WIN": r_win,
           "R-CURV": r_curv, "R-GAP": r_gap, "R-FEAT": r_feat, "R-BUDGET": r_budget}


def setup(row: dict, fingerprint: dict, stl_path: str, case_dir: str, name: str, *,
          flow=None, gates=None, knobs=None, requested_growth=None,
          ft_radius=True) -> dict:
    """The eight L1 rules on one geometry; the autonomy-rules/1 result out."""
    gates = gates or schema.load_gates()
    knobs = knobs or schema.load_knobs()
    errs = schema.errors(fingerprint, "Fingerprint")
    if errs:
        raise RulesError("the fingerprint is invalid: %s" % errs[0])
    flow = flow if flow is not None else row.get("flow")
    patches = []
    for p in fingerprint["patches"]:
        if p["name"] not in patches:
            patches.append(p["name"])
    state = {"config": {"input": {"surfaces": [{"path": stl_path}]},
                        "output": {"case_dir": case_dir, "name": name}},
             "fp": fingerprint, "flow": flow, "L": flow["l_ref_m"], "gates": gates,
             "knobs": knobs, "cap": schema._knob_row("/refinement/max_level",
                                                     knobs)["max"],
             "patches": patches, "requested_growth": requested_growth,
             "t1": None, "n": None, "base": None, "plane": None, "win_level": None,
             "wall_level": None, "rung": (1.0, 1.0), "feature": False,
             "feature_level": None, "growth": None, "stop": None, "predicted": None,
             "ft_radius": bool(ft_radius), "ft_set": False}
    records = []
    for rid in RULES:
        if state["stop"] is not None:
            records.append(_stopped(rid, state["stop"]))
            continue
        state["_before"] = copy.deepcopy(state["config"])
        rec = RULE_FN[rid](state)
        if rec["rule_id"] != rid or rec["message"][:len(rid) + 2] != rid + ": ":
            raise RulesError("record %s carries rule_id %r / message %r"
                             % (rid, rec["rule_id"], rec["message"][:24]))
        records.append(rec)
    refused = [state["stop"]] if state["stop"] else []
    h_wall = state["base"] / 2 ** state["wall_level"] \
        if state["base"] is not None and state["wall_level"] is not None else None
    pred = state["predicted"]
    return {"schema": RESULT_SCHEMA, "geometry_id": row.get("geometry_id"),
            "verdict": "refuse" if refused else "apply", "refused": refused,
            "config": None if refused else state["config"],
            "config_sha256": None if refused else schema.canonical_sha256(state["config"]),
            "records": records,
            "edits": [e for r in records for e in r["edits"]],
            "summary": {"t1_m": state["t1"], "n": state["n"],
                        "base_size_m": state["base"], "wall_level": state["wall_level"],
                        "win_level": state["win_level"], "h_wall_m": h_wall,
                        "h_over_t1": (h_wall / state["t1"]) if h_wall is not None else None,
                        "growth": state["growth"], "feature_level": state["feature_level"],
                        "plane": state["plane"] is not None,
                        "rung": list(state["rung"]) if state["wall_level"] is not None else None,
                        "predicted_total": round(pred["total"]) if pred else None,
                        "predicted_leaves":
                            round(pred["leaves"] + pred["feature"]) if pred else None}}


def ft_radius_of(records) -> bool:
    """Which rules a recorded campaign ran under: True when one of its R-FEAT apply
    records edits /snap/feature_tolerance (FT-RADIUS, tau = h_f / 2), False for a
    campaign recorded before it.  A reader that rebuilds recorded attempt-1 configs
    calls setup(..., ft_radius=ft_radius_of(the campaign's records)); `records` are
    DecisionRecords or {"record": DecisionRecord, ...} lines, as bundles carry them."""
    for item in records:
        rec = item.get("record", item) if isinstance(item, dict) else None
        if not isinstance(rec, dict) or rec.get("rule_id") != "R-FEAT" \
                or rec.get("verdict") != "apply":
            continue
        if any(e.get("pointer") == "/snap/feature_tolerance" for e in rec.get("edits") or []):
            return True
    return False


# --- the face-on-a-cell-plane check (C10) -------------------------------------

def plane_check(config: dict, planes, level: int) -> dict:
    """ok when every generator plane lies on base/2**level of the extent, and
    the extent is a whole number of base cells per axis."""
    base = config["domain"]["base_size"]
    ext = config["domain"]["extent"]
    h = base / 2 ** level
    worst_block = worst_face = 0.0
    for a in range(3):
        q = (ext[2 * a + 1] - ext[2 * a]) / base
        worst_block = max(worst_block, abs(q - round(q)))
        for v in planes[a]:
            q = (v - ext[2 * a]) / h
            worst_face = max(worst_face, abs(q - round(q)))
    return {"ok": worst_face <= 1e-9 and worst_block <= 1e-9,
            "worst_face": worst_face, "worst_block": worst_block}


# --- the gate (C11) ------------------------------------------------------------

def gate_rows(family: str, rows: list) -> list:
    """12 tuning rows of one family: strata in manifest order, round robin
    easy, medium, hard, skipping an exhausted stratum."""
    by = {}
    for r in rows:
        if r["family"] == family:
            by.setdefault(r["stratum"], []).append(r)
    picked, idx = [], {s: 0 for s in ("easy", "medium", "hard")}
    while len(picked) < 12:
        progressed = False
        for s in ("easy", "medium", "hard"):
            if idx[s] < len(by.get(s, [])):
                picked.append(by[s][idx[s]])
                idx[s] += 1
                progressed = True
                if len(picked) == 12:
                    break
        if not progressed:
            break
    return picked


def _gen_of(family: str):
    """The corpus generator module of one family (numpy lives there, import late)."""
    import importlib
    return importlib.import_module(GEN_OF[family])


def _fp_of(path: str, geometry_id: str) -> dict:
    """features.fingerprint with features.py's known planar divide-by-zero noise
    silenced (features.py:227 - the result is correct)."""
    import features
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return features.fingerprint(path, geometry_id)[0]


def _fwd(p: str) -> str:
    """A forward-slash path for a config (the callers pass forward slashes)."""
    return p.replace(os.sep, "/")


def _sha256_of(path: str):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _last_error_line(stderr: str):
    lines = [ln for ln in stderr.splitlines() if ln.strip().startswith("error:")]
    return lines[-1] if lines else None


_CUBEP_EXT_FACTS = [-2.4285714285714284, 8.83673469387755, -1.9387755102040813,
                    5.408163265306123, -1.9387755102040813, 5.408163265306123]


def _gate_part1(out_dir: str, streams: int, binary: str, gates: dict, knobs: dict) -> tuple:
    """The worked examples and the five live cube runs A, B, C, D, F."""
    p1 = os.path.join(out_dir, "p1")
    os.makedirs(p1, exist_ok=True)
    stl = _fwd(preflight.CUBEP_STL)
    fp = _fp_of(stl, "CUBEP")
    cube = {"geometry_id": "CUBEP", "flow": WORKED_FLOW}
    res = setup(cube, fp, stl, out_dir + "/p1/case_CUBEP", "CUBEP",
                gates=gates, knobs=knobs)
    ins = {i["name"]: i["value"] for i in res["records"][0]["inputs"]}
    checks = {
        "a_t1_raw": abs(ins["t1_raw"] - T1_WORKED) <= 0.01 * T1_WORKED,
        "a_t1_floored": res["summary"]["t1_m"] == 0.0007296,
        "a_yplus": ins["yplus_a_priori"] <= 1.0,
    }
    w = window_level(0.0007296, 8, 0.5, 6, knobs=knobs)
    w14 = window_level(0.0007296, 8, 0.5, 6, knobs=knobs, growth=1.4)
    checks["b_window"] = (w["level"] == 4 and w["h"] == 0.03125
                          and w["growth"] == 1.27
                          and abs(w["T"] - 0.015585129963795324) <= 1e-15
                          and w14["verdict"] == "refuse" and w14["reason"] == "empty")
    res14 = setup(cube, fp, stl, out_dir + "/p1/case_G14", "CUBEPG14",
                  gates=gates, knobs=knobs, requested_growth=1.4)
    checks["c_refused"] = res14["refused"] == ["R-WIN"] and res14["config"] is None
    pin = {i["name"]: i["value"] for i in res["records"][2]["inputs"]}
    ext = res["config"]["domain"]["extent"]
    checks["d_plane"] = (pin["m"] == 49
                         and abs(pin["h"] - 0.030612244897959183) <= 1e-15
                         and res["summary"]["wall_level"] == 4
                         and res["summary"]["growth"] == 1.265
                         and all(abs(x - y) <= 1e-12 for x, y in zip(ext, _CUBEP_EXT_FACTS)))
    pf = preflight.preflight(res["config"], fingerprint=fp, flow=WORKED_FLOW,
                             gates=gates, knobs=knobs)
    checks["d_preflight"] = pf["verdict"] == "pass"
    t1, h = res["summary"]["t1_m"], res["summary"]["h_wall_m"]
    variants = {}
    for tag in ("A", "B", "C", "D", "F"):
        c = copy.deepcopy(res["config"])
        c["output"] = {"case_dir": out_dir + "/p1/case_" + tag, "name": "cube" + tag}
        if tag == "B":
            g = next(k / GROWTH_STEP_DIV for k in range(1000, 2001)
                     if stack_total(t1, k / GROWTH_STEP_DIV, 8) > CELL_FRAC * h * 1.001)
            c["layers"]["growth"] = g
        elif tag == "C":
            c["layers"]["first_thickness"] = h / 60 * (1 - 1e-3)
        elif tag == "D":
            c["layers"]["first_thickness"] = h / 60 * (1 + 1e-3)
        elif tag == "F":
            c["layers"]["first_thickness"] = h / 44.0
            c["layers"]["growth"] = 1.2
        variants[tag] = c
    cfg_dir = os.path.join(p1, "cfg")
    os.makedirs(cfg_dir, exist_ok=True)
    for tag, c in variants.items():
        with open(os.path.join(cfg_dir, "cube%s.json" % tag), "w", encoding="utf-8") as fh:
            json.dump(c, fh, indent=1)

    def run(tag):
        cp = os.path.join(cfg_dir, "cube%s.json" % tag)
        return tag, score.run_automesher(binary, cp, [], cwd=p1, timeout_s=600)

    with ThreadPoolExecutor(max_workers=min(streams, 5)) as ex:
        results = dict(ex.map(run, list(variants)))
    runs, ok = {}, True
    for tag, r in results.items():
        g = variants[tag]["layers"]["growth"]
        tv = variants[tag]["layers"]["first_thickness"]   # C, D, F change it
        entry = {"t1": tv, "growth": g, "h_over_t1": round(h / tv, 3),
                 "T_over_cell_frac_h": round(stack_total(tv, g, 8) / (CELL_FRAC * h), 6),
                 "exit_code": r["exit_code"]}
        if r["exit_code"] == 0:
            sp = os.path.join(p1, "case_" + tag, "cube%s_summary.json" % tag)
            with open(sp, encoding="utf-8") as fh:
                row = score.layer_rows(json.load(fh)).get("cube")
            entry.update({"n_layers": row["n_layers"],
                          "full_area_frac": round(row["full_area_frac"], 6),
                          "mean_frac": round(row["mean_frac"], 6),
                          "dropped": row["dropped"]})
        else:
            line = _last_error_line(r["stderr"])
            mch = preflight.G5_LINE_RE.search(line) if line else None
            entry["g5_line"] = line
            entry["g5_ratio"] = float(mch.group(3)) if mch else None
        runs[tag] = entry
    ok = (all(checks.values())
          and runs["A"]["exit_code"] == 0 and runs["A"]["n_layers"] == 8
          and runs["A"]["dropped"] is None and runs["A"]["full_area_frac"] == 1.0
          and runs["B"]["exit_code"] == 0 and runs["B"]["n_layers"] == 8
          and runs["B"]["dropped"] is None and runs["B"]["full_area_frac"] < 1.0
          and runs["C"]["exit_code"] == 1 and runs["C"]["g5_ratio"] is not None
          and runs["C"]["g5_ratio"] < 0.05
          and all(runs[t]["exit_code"] == 0 and runs[t]["n_layers"] == 0
                  and isinstance(runs[t]["dropped"], str)
                  and runs[t]["dropped"].startswith(
                      'patch "cube": the thickness fell below min_thickness')
                  for t in ("D", "F")))
    return {"verdict": "PASS" if ok else "FAIL", "checks": checks, "runs": runs}, \
        ("t1 %g, window level %d g %.3f, plane m %d, runs A %s B %s C exit %s "
         "D %s F %s" % (t1, w["level"], w["growth"], pin["m"],
                        _run_tag(runs["A"]), _run_tag(runs["B"]),
                        runs["C"]["exit_code"], _run_tag(runs["D"]),
                        _run_tag(runs["F"])))


def _run_tag(e: dict) -> str:
    if e["exit_code"] != 0:
        return "exit %s" % e["exit_code"]
    return "n%d full %s" % (e["n_layers"], e["full_area_frac"])


def _gate_part2(family: str, out_dir: str, streams: int, binary: str,
                gates: dict, knobs: dict) -> tuple:
    """12 tuning rows of one family: setup, one live octree probe each, preflight,
    -dryRun; refusals re-derived."""
    import split
    gen = _gen_of(family)
    picked = gate_rows(family, split.load("tuning", "rules"))
    base = os.path.join(out_dir, "2" + family)
    stl_dir = os.path.join(base, "stl")
    os.makedirs(stl_dir, exist_ok=True)
    phase1 = []
    for row in picked:
        gid = row["geometry_id"]
        t0 = time.perf_counter()
        stl = _fwd(gen.write_row(row, stl_dir))
        fp = _fp_of(stl, gid)
        res = setup(row, fp, stl, base + "/case/" + gid, gid, gates=gates, knobs=knobs)
        phase1.append([row, fp, res, time.perf_counter() - t0])
    os.makedirs(os.path.join(base, "cfg"), exist_ok=True)

    def work(item):
        row, fp, res, secs = item
        gid = row["geometry_id"]
        out = {"geometry_id": gid, "family": row["family"], "stratum": row["stratum"],
               "verdict": res["verdict"], "refused": res["refused"],
               "fired": [r["rule_id"] for r in res["records"] if r["verdict"] == "apply"],
               "verdicts": {r["rule_id"]: r["verdict"] for r in res["records"]},
               "seconds": round(secs, 3)}
        out.update({k: res["summary"][k] for k in
                    ("wall_level", "win_level", "h_over_t1", "growth", "feature_level",
                     "plane", "rung", "predicted_total", "predicted_leaves")})
        if res["verdict"] != "apply":
            out.update({"rederived": _rederive(res, gates, knobs), "probe_n_leaves": None,
                        "probe_exit": None, "pred_over_probe": None,
                        "preflight_verdict": None, "preflight_refused": None,
                        "pf_verdicts": None, "dryrun_exit": None, "dryrun_error": None})
            return out
        cfg = res["config"]
        probe = preflight.octree_probe(binary, cfg, os.path.join(base, "w_" + gid),
                                       timeout_s=900)
        pf = preflight.preflight(cfg, fingerprint=fp, flow=row["flow"],
                                 octree_probe=probe, gates=gates, knobs=knobs)
        cfg_path = os.path.join(base, "cfg", gid + ".json")
        with open(cfg_path, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=1)
        d = preflight.run_dryrun(binary, cfg_path)
        out.update({"probe_n_leaves": probe["n_leaves"], "probe_exit": probe["exit_code"],
                    "pred_over_probe": (round(res["summary"]["predicted_leaves"]
                                              / probe["n_leaves"], 4)
                                        if probe["n_leaves"] else None),
                    "preflight_verdict": pf["verdict"],
                    "preflight_refused": pf["refused"],
                    "pf_verdicts": {r["rule_id"]: r["verdict"] for r in pf["records"]},
                    "dryrun_exit": d["exit_code"], "dryrun_error": d["error"]})
        return out

    with ThreadPoolExecutor(max_workers=min(streams, 6)) as ex:
        rows_out = list(ex.map(work, phase1))
    with open(os.path.join(base, "rows.jsonl"), "w", encoding="utf-8") as fh:
        for out in rows_out:
            fh.write(json.dumps(out, sort_keys=True, ensure_ascii=False) + "\n")
    applied = [o for o in rows_out if o["verdict"] == "apply"]
    refused = [o for o in rows_out if o["verdict"] == "refuse"]
    ratios = sorted(o["pred_over_probe"] for o in applied if o["pred_over_probe"])
    leaves = [o["probe_n_leaves"] for o in applied if o["probe_n_leaves"]]
    ok = (len(rows_out) == 12
          and all(o["probe_exit"] == 0 and o["preflight_verdict"] == "pass"
                  and o["pf_verdicts"]["PF-BUDGET"] == "pass"
                  and o["pf_verdicts"]["PF-YPLUS"] == "pass"
                  and o["pf_verdicts"]["PF-THIN"] == "pass"
                  and o["dryrun_exit"] == 0 for o in applied)
          and all(o["verdicts"]["R-BUDGET"] != "refuse"
                  or o["rederived"] for o in refused)
          and all(o["verdicts"]["R-WIN"] != "refuse"
                  or o["rederived"] for o in refused)
          and all(set(o["refused"]) <= {"R-BUDGET", "R-WIN"} for o in refused))
    fired = {}
    for o in rows_out:
        for rid in o["fired"]:
            fired[rid] = fired.get(rid, 0) + 1
    walls = {}
    for o in applied:
        walls[o["wall_level"]] = walls.get(o["wall_level"], 0) + 1
    med = ratios[len(ratios) // 2] if ratios else None
    info = ("%d rows, %d applied, %d refused (%s), preflight pass %d/%d, -dryRun 0 %d/%d, "
            "probe leaves max %s, predicted/probe min %s median %s max %s"
            % (len(rows_out), len(applied), len(refused),
               ",".join(o["geometry_id"] for o in refused) or "-",
               sum(1 for o in applied if o["preflight_verdict"] == "pass"), len(applied),
               sum(1 for o in applied if o["dryrun_exit"] == 0), len(applied),
               max(leaves) if leaves else "-",
               ratios[0] if ratios else "-", med if ratios else "-",
               ratios[-1] if ratios else "-"))
    detail = {"applied": len(applied), "refused": len(refused),
              "refused_ids": [o["geometry_id"] for o in refused],
              "fired": fired, "wall_levels": walls, "rows": rows_out,
              "failed": [o["geometry_id"] for o in rows_out if not _row_ok(o)]}
    return {"verdict": "PASS" if ok else "FAIL", **detail}, info


def _row_ok(o: dict) -> bool:
    if o["verdict"] == "refuse":
        return bool(o.get("rederived")) and set(o["refused"]) <= {"R-BUDGET", "R-WIN"}
    return (o["probe_exit"] == 0 and o["preflight_verdict"] == "pass"
            and o["pf_verdicts"]["PF-BUDGET"] == "pass"
            and o["pf_verdicts"]["PF-YPLUS"] == "pass"
            and o["pf_verdicts"]["PF-THIN"] == "pass" and o["dryrun_exit"] == 0)


def _rederive(res: dict, gates: dict, knobs: dict) -> bool:
    """Re-derive an R-BUDGET / R-WIN refusal from the result the rules returned."""
    rid = res["refused"][0]
    s = res["summary"]
    rec = next(r for r in res["records"] if r["rule_id"] == rid)
    if rid == "R-BUDGET":
        ladder = next(i["value"] for i in rec["inputs"] if i["name"] == "ladder")
        target = PRED_SAFETY * gates["cell_budget"]
        return (bool(ladder) and all(e["predicted"] > target for e in ladder)
                and ladder[-1]["wall_level"] == s["win_level"]
                and ladder[-1]["far"] == 0.25 and ladder[-1]["wall"] == 0.25)
    if rid == "R-WIN":
        w2 = window_level(s["t1_m"], s["n"], s["base_size_m"],
                          schema._knob_row("/refinement/max_level", knobs)["max"],
                          kappa=CORNER_KAPPA if s["plane"] else 1.0, knobs=knobs)
        if w2["verdict"] != "refuse":
            return False
        want = {"empty": "window is empty", "fine": "finest reachable",
                "coarse": "coarsest reachable", "none": "no level"}
        return want[w2["reason"]] in rec["message"]
    return False


def _gate_part3(out_dir: str, streams: int, binary: str, gates: dict, knobs: dict) -> tuple:
    """R-PLANE on every commensurate tuning row; three live full runs' snap."""
    import split
    base = os.path.join(out_dir, "3")
    stl_dir = os.path.join(base, "stl")
    os.makedirs(stl_dir, exist_ok=True)
    rows = [r for r in split.load("tuning", "rules") if r.get("commensurate") is True]
    shapes = ("box_c", "plate_c", "lcorner_c")
    counts = {s: [0, 0] for s in shapes}
    first_applied, rows_out, worst = {}, [], 0.0
    checks_ok, applied_n, ok_count = True, 0, 0
    for row in rows:
        gid = row["geometry_id"]
        gen = _gen_of(row["family"])
        stl = _fwd(gen.write_row(row, stl_dir))
        fp = _fp_of(stl, gid)
        res = setup(row, fp, stl, base + "/case/" + gid, gid, gates=gates, knobs=knobs)
        shape = row["params"]["shape"]
        v = {r["rule_id"]: r["verdict"] for r in res["records"]}
        out = {"geometry_id": gid, "shape": shape, "r_plane": v["R-PLANE"],
               "applied": res["verdict"] == "apply"}
        if v["R-PLANE"] == "apply":
            applied_n += 1
            if shape in counts:
                counts[shape][0] += 1
            pc = plane_check(res["config"], gen.planes(row["params"]),
                             res["summary"]["wall_level"])
            worst = max(worst, pc["worst_face"], pc["worst_block"])
            checks_ok = checks_ok and pc["ok"]
            ok_count += 1 if pc["ok"] else 0
            out["plane_check"] = pc
            if shape in shapes and shape not in first_applied:
                first_applied[shape] = (row, res)
        elif v["R-PLANE"] == "abstain":
            rec = res["records"][2]
            out["h_over_t1"] = next(i["value"] for i in rec["inputs"]
                                    if i["name"] == "h_over_t1")
        if shape in counts:
            counts[shape][1] += 1
        rows_out.append(out)
    box_total = counts["box_c"][1]
    live = []
    for shape in shapes:
        entry = {"shape": shape}
        if shape not in first_applied:
            entry["skipped"] = True
            live.append(entry)
            continue
        row, res = first_applied[shape]
        gid = row["geometry_id"]
        cfg_dir = os.path.join(base, "cfg")
        os.makedirs(cfg_dir, exist_ok=True)
        cfg_path = os.path.join(cfg_dir, gid + ".json")
        with open(cfg_path, "w", encoding="utf-8") as fh:
            json.dump(res["config"], fh, indent=1)
        r = score.run_automesher(binary, cfg_path, [], cwd=base, timeout_s=900)
        snap = lay = None
        if r["exit_code"] == 0:
            sp = os.path.join(res["config"]["output"]["case_dir"],
                              "%s_summary.json" % gid)
            with open(sp, encoding="utf-8") as fh:
                summ = json.load(fh)
            st = {s2["stage"]: s2 for s2 in summ["stages"]}
            snap = st.get("snap", {}).get("max_over_h")
            lr = list(score.layer_rows(summ).values())
            lay = {k: lr[0][k] for k in ("n_layers", "full_area_frac", "mean_frac",
                                         "dropped")} if lr else {}
        entry.update({"geometry_id": gid, "exit_code": r["exit_code"],
                      "snap_max_over_h": snap, "layers": lay})
        live.append(entry)
    abst_ids = [o["geometry_id"] for o in rows_out if o["r_plane"] == "abstain"]
    snaps = [e["snap_max_over_h"] for e in live if e.get("snap_max_over_h") is not None]
    ok = (counts["box_c"][0] == box_total and checks_ok and len(live) == 3
          and all(e["exit_code"] == 0 for e in live) and len(snaps) == 3
          and all(v <= 1e-9 for v in snaps))
    info = ("%d commensurate rows, R-PLANE applied %d (box_c %d/%d, plate_c %d/%d, "
            "lcorner_c %d/%d), abstained %s, faces on cell planes %d/%d (worst %.3g), "
            "live snap max_over_h %s"
            % (len(rows), applied_n, counts["box_c"][0], box_total,
               counts["plate_c"][0], counts["plate_c"][1],
               counts["lcorner_c"][0], counts["lcorner_c"][1],
               ",".join(abst_ids) or "-", ok_count, applied_n,
               worst, " ".join("%.3g" % v for v in snaps)))
    detail = {"rows": len(rows), "applied": applied_n, "faces_ok": ok_count,
              "counts": counts,
              "abstained": abst_ids, "rows_out": rows_out, "live": live}
    return {"verdict": "PASS" if ok else "FAIL", **detail}, info


# --- the gate driver and the report (C11) --------------------------------------

def _write_report(report: dict) -> None:
    """The ONLY in-tree writes of --gate: REPORT_DIR/G-RULES.json and .md."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, "G-RULES.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, sort_keys=True, ensure_ascii=False)
    with open(os.path.join(REPORT_DIR, "G-RULES.md"), "w", encoding="utf-8") as fh:
        fh.write(_report_md(report))


def _report_md(report: dict) -> str:
    """G-RULES.md: every number read from the merged JSON."""
    p = report["parts"]
    lines = ["<!-- meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 "
             "(Iterations Co., Ltd.). Source-available, not Open Source. "
             "No GPL-licensed source was consulted. -->",
             "", "# G-RULES — the AM-9 gate of rules.py", "",
             "Date %s - binary sha256 `%s` - git HEAD `%s`."
             % (report.get("date"), (report.get("binary_sha256") or "?")[:16],
                (report.get("git_head") or "?")[:12]), ""]
    p1 = p.get("1")
    if p1:
        lines += ["## Part 1 - %s" % p1["verdict"], ""]
        for tag in ("A", "B", "C", "D", "F"):
            e = p1["runs"].get(tag, {})
            lines.append("- run %s: exit %s, growth %s, h/t1 %s, T/(cf*h) %s%s"
                         % (tag, e.get("exit_code"), e.get("growth"),
                            e.get("h_over_t1"), e.get("T_over_cell_frac_h"),
                            (", n_layers %s, full %s, mean %s, dropped %s"
                             % (e.get("n_layers"), e.get("full_area_frac"),
                                e.get("mean_frac"),
                                (e.get("dropped") or "")[:80]) if "n_layers" in e
                             else (", G5 %.6f" % e["g5_ratio"]
                                   if e.get("g5_ratio") is not None else ""))))
        lines.append("")
    for key in ("2A", "2B", "2D", "2E", "2F"):
        px = p.get(key)
        if px:
            lines += ["## Part %s - %s" % (key, px["verdict"]), "",
                      "- %d rows, %d applied, %d refused %s" % (len(px["rows"]),
                       px["applied"], px["refused"], px["refused_ids"] or ""),
                      "- fired %s, wall levels %s" % (px["fired"], px["wall_levels"])]
            if px.get("failed"):
                lines.append("- FAILED ROWS: %s" % px["failed"])
            lines.append("")
    if p.get("2"):
        lines += ["## Part 2 - %s" % p["2"]["verdict"], "",
                  "- %d rows, %d applied, %d refused" % (p["2"]["rows"],
                  p["2"]["applied"], p["2"]["refused"]), ""]
    p3 = p.get("3")
    if p3:
        c = p3["counts"]
        lines += ["## Part 3 - %s" % p3["verdict"], "",
                  "- %d commensurate rows, R-PLANE applied %d "
                  "(box_c %d/%d, plate_c %d/%d, lcorner_c %d/%d), abstained %s"
                  % (p3["rows"], p3["applied"], c["box_c"][0], c["box_c"][1],
                     c["plate_c"][0], c["plate_c"][1], c["lcorner_c"][0],
                     c["lcorner_c"][1], p3["abstained"] or "none")]
        for e in p3["live"]:
            lines.append("- live %s (%s): exit %s, snap max_over_h %s, layers %s"
                         % (e["shape"], e.get("geometry_id"), e.get("exit_code"),
                            e.get("snap_max_over_h"), e.get("layers")))
        lines += ["", "On a box the delivered stack needs h/t1 <= about 42.9: runs D "
                  "and F pass the §D.3 early check and still lose their layers, so "
                  "R-PLANE uses 0.70 of the G5 edge.", ""]
    return "\n".join(lines) + "\n"


def gate(parts, out_dir: str, streams: int = 6, binary: str | None = None) -> int:
    """G-RULES, one call per part; the merged report lives in REPORT_DIR."""
    binary = binary or BINARY_DEFAULT
    os.makedirs(out_dir, exist_ok=True)
    gates, knobs = schema.load_gates(), schema.load_knobs()
    path = os.path.join(REPORT_DIR, "G-RULES.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            report = json.load(fh)
    else:
        report = {"schema": GATE_SCHEMA, "parts": {}}
    report.setdefault("parts", {})
    report["schema"] = GATE_SCHEMA
    report["binary_sha256"] = _sha256_of(binary)
    report["git_head"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                        capture_output=True, text=True).stdout.strip()
    report["date"] = schema._now_iso()[:10]
    failed, ran = [], []
    for key in parts:
        key = str(key)
        if key == "1":
            d, info = _gate_part1(out_dir, streams, binary, gates, knobs)
        elif key == "3":
            d, info = _gate_part3(out_dir, streams, binary, gates, knobs)
        elif key in ("2A", "2B", "2D", "2E", "2F"):
            d, info = _gate_part2(key[1], out_dir, streams, binary, gates, knobs)
        else:
            raise RulesError("gate part %r is not 1, 2A, 2B, 2D, 2E, 2F or 3" % key)
        report["parts"][key] = d
        ran.append(key)
        if d["verdict"] != "PASS":
            failed.append(key)
        ks = ["2A", "2B", "2D", "2E", "2F"]
        if all(k in report["parts"] for k in ks):
            applied = sum(report["parts"][k]["applied"] for k in ks)
            refused = sum(report["parts"][k]["refused"] for k in ks)
            allpass = all(report["parts"][k]["verdict"] == "PASS" for k in ks)
            report["parts"]["2"] = {"rows": 60, "applied": applied, "refused": refused,
                                    "verdict": "PASS" if allpass and applied >= 54
                                    else "FAIL"}
        core = all(k in report["parts"] for k in ("1", "2", "3"))
        if core:
            cores = [report["parts"][k].get("verdict") for k in ("1", "2", "3")]
            report["verdict"] = "PASS" if all(v == "PASS" for v in cores) else "FAIL"
        else:
            report["verdict"] = "PARTIAL"
        _write_report(report)
        print("[gate] part %s %s: %s" % (key, d["verdict"], info), flush=True)
        if key != "2" and "2" in report["parts"]:
            print("[gate] part 2 %s: %d rows, %d applied, %d refused"
                  % (report["parts"]["2"]["verdict"], report["parts"]["2"]["rows"],
                     report["parts"]["2"]["applied"], report["parts"]["2"]["refused"]),
                  flush=True)
    core = all(k in report["parts"] and report["parts"][k].get("verdict") == "PASS"
               for k in ("1", "2", "3"))
    if failed:
        print("G-RULES FAIL: %s" % " ".join(failed))
        return 1
    if core:
        print("G-RULES PASS")
        return 0
    print("G-RULES PARTIAL: have %s" % " ".join(sorted(report["parts"])))
    return 0


# --- the selftest (C12) --------------------------------------------------------

def selftest() -> int:
    """16 [ok] groups, no full mesher run, only -dryRun; under 60 s."""
    import random
    import shutil
    import tempfile
    import split
    gates, knobs = schema.load_gates(), schema.load_knobs()
    by_id = {r["geometry_id"]: r for r in split.load("tuning", "rules")}
    tmp = tempfile.mkdtemp(prefix="rules_selftest_")
    lines = []

    def group(name, fn):
        try:
            lines.append("[ok] %s: %s" % (name, fn()))
        except Exception as e:  # noqa: BLE001 - a selftest group names its failure
            print("SELFTEST FAIL: %s: %s" % (name, e))
            return 1
        return 0

    stl_dir = os.path.join(tmp, "stl")
    os.makedirs(stl_dir, exist_ok=True)
    stls, fps, setups, all_records = {}, {}, {}, []
    extra_setups = []

    def stl_of(gid: str) -> str:
        if gid not in stls:
            gen = _gen_of(by_id[gid]["family"])
            stls[gid] = _fwd(gen.write_row(by_id[gid], stl_dir))
        return stls[gid]

    def fp_of(gid: str, overrides: dict | None = None) -> dict:
        key = (gid, tuple(sorted((overrides or {}).items())))
        if key not in fps:
            fp = _fp_of(stl_of(gid), gid)
            fp.update(overrides or {})
            fps[key] = fp
        return fps[key]

    def setup_of(gid: str, fp: dict | None = None, **kw) -> dict:
        fresh = fp is not None or kw
        if not fresh and gid in setups:
            return setups[gid]
        res = setup(by_id[gid], fp or fp_of(gid), stl_of(gid),
                    tmp + "/case_" + gid, gid, gates=gates, knobs=knobs, **kw)
        all_records.extend(res["records"])
        if not fresh:
            setups[gid] = res
        else:
            extra_setups.append(res)
        return res

    def cube_setup(**kw) -> dict:
        stl = _fwd(preflight.CUBEP_STL)
        res = setup({"geometry_id": "CUBEP", "flow": WORKED_FLOW},
                    _fp_of(stl, "CUBEP"), stl, tmp + "/case_cubep", "CUBEP",
                    gates=gates, knobs=knobs, **kw)
        all_records.extend(res["records"])
        extra_setups.append(res)
        return res

    def g1():
        res = cube_setup()
        ins = {i["name"]: i["value"] for i in res["records"][0]["inputs"]}
        assert abs(ins["t1_raw"] - T1_WORKED) <= 0.01 * T1_WORKED, ins["t1_raw"]
        assert res["summary"]["t1_m"] == 0.0007296
        assert ins["yplus_a_priori"] <= 1
        return ("Re_L %g, t1 %r -> %r m (within 1 %% of 7.29699e-4), y+ %.6f"
                % (ins["re_l"], ins["t1_raw"], ins["t1"], ins["yplus_a_priori"]))

    def g2():
        w = window_level(0.0007296, 8, 0.5, 6, knobs=knobs)
        assert (w["level"], w["h"], w["growth"]) == (4, 0.03125, 1.27), w
        assert abs(w["T"] - 0.015585129963795324) <= 1e-15 and w["T"] <= 0.015625
        res = cube_setup(requested_growth=1.4)
        assert res["refused"] == ["R-WIN"] and res["config"] is None
        w14 = window_level(0.0007296, 8, 0.5, 6, knobs=knobs, growth=1.4)
        assert w14["verdict"] == "refuse" and w14["reason"] == "empty"
        return ("base 0.5 -> wall level %d, h %g m (h/t1 %.2f), growth %.3f, "
                "T 0.015585129963795324 <= 0.015625; g 1.4 refused: window empty "
                "(%.2f t1 > 60 t1)"
                % (w["level"], w["h"], w["h"] / 0.0007296, w["growth"],
                   w14["lo"] / 0.0007296))

    def g3():
        vals = [stack_total(1, g, 8) / CELL_FRAC for g in (1.0, 1.2, 1.3, 1.4)]
        for got, want in zip(vals, (16.00, 33.00, 47.72, 68.79)):
            assert abs(got - want) <= 5e-3, (got, want)
        lo, hi = 1.0, 1.4
        while hi - lo > 1e-9:
            mid = (lo + hi) / 2
            if stack_total(1, mid, 8) / CELL_FRAC > 60:
                hi = mid
            else:
                lo = mid
        assert 1.36 < hi < 1.365, hi
        return ("S/cell_frac at n 8 = %.2f, %.2f, %.2f, %.2f at g 1, "
                "1.2, 1.3, 1.4; empty above g %.4f"
                % (vals[0], vals[1], vals[2], vals[3], hi))

    def g4():
        rng = random.Random(9)
        for _ in range(200):
            t1 = 10 ** rng.uniform(-5, -3)
            h = t1 * rng.uniform(10, 70)
            g = fit_growth(t1, 8, h, knobs)
            budget = 0.5 * h * (1 - EDGE_MARGIN)
            if g is None:
                assert 8 * t1 > budget, (t1, h)
            else:
                assert stack_total(t1, g, 8) <= budget, (t1, h, g)
                if g != 2.0:
                    assert stack_total(t1, g + 0.001, 8) > budget, (t1, h, g)
        return "200 random (t1, h) cases exact"

    FIVE = ("A-1-000", "B-1-000", "D-1-001", "E-1-000", "F-1-001")

    def g5():
        for gid in FIVE:
            ext = setup_of(gid)["config"]["domain"]["extent"]
            base = setup_of(gid)["config"]["domain"]["base_size"]
            bb = fp_of(gid)["bbox"]
            L = by_id[gid]["flow"]["l_ref_m"]
            mg = [f * L for f in DOMAIN_MARGINS]
            for a in range(3):
                q = (ext[2 * a + 1] - ext[2 * a]) / base
                assert abs(q - round(q)) <= 1e-9, (gid, a, q)
                assert bb[2 * a] - ext[2 * a] >= mg[2 * a] - 1e-9, (gid, a)
                assert ext[2 * a + 1] - bb[2 * a + 1] >= mg[2 * a + 1] - 1e-9, (gid, a)
        return ("5 fingerprints, extents on base multiples, margins >= 3 / 6 / "
                "2.5 L_ref")

    def lattice_rows():
        rows = split_rows()
        shape = lambda r: r["params"].get("shape")
        box = [r for r in rows if r["family"] == "D" and shape(r) == "box_c"][:5]
        plate = next(r for r in rows if shape(r) == "plate_c")
        lc = next(r for r in rows if shape(r) == "lcorner_c")
        return box + [plate, lc]

    _TUNING = None

    def split_rows():
        nonlocal _TUNING
        import split as sp
        if _TUNING is None:
            _TUNING = sp.load("tuning", "rules")
        return _TUNING

    def g6():
        stl = _fwd(preflight.CUBEP_STL)
        fp = _fp_of(stl, "CUBEP")
        res = cube_setup()
        pin = {i["name"]: i["value"] for i in res["records"][2]["inputs"]}
        assert pin["m"] == 49 and abs(pin["h"] - 0.030612244897959183) <= 1e-15
        assert res["summary"]["wall_level"] == 4
        assert abs(res["summary"]["base_size_m"] - 0.4897959183673469) <= 1e-15
        assert res["summary"]["growth"] == 1.265
        n_on = 0
        for row in lattice_rows():
            r = setup_of(row["geometry_id"])
            assert r["records"][2]["verdict"] == "apply", row["geometry_id"]
            pc = plane_check(r["config"], _gen_of(row["family"]).planes(row["params"]),
                             r["summary"]["wall_level"])
            assert pc["ok"], (row["geometry_id"], pc)
            n_on += 1
        assert n_on == 7
        f9 = setup_of("F-1-009")
        assert f9["records"][2]["verdict"] == "abstain"
        h_t1 = next(i["value"] for i in f9["records"][2]["inputs"]
                    if i["name"] == "h_over_t1")
        assert abs(h_t1 - 15.2) < 0.05, h_t1
        return ("cubep m %d, h %r, level %d, base %r, growth %.3f; %d box_c, 1 "
                "plate_c, 1 lcorner_c rows on the lattice; F-1-009 abstains "
                "(h/t1 %.1f < 16)"
                % (pin["m"], pin["h"], res["summary"]["wall_level"],
                   res["summary"]["base_size_m"], res["summary"]["growth"],
                   n_on - 2, h_t1))

    def curv_gap_case(gid, over, which):
        res = setup_of(gid, fp_of(gid, over))
        idx = {"R-CURV": 4, "R-GAP": 5}[which]
        return res, res["records"][idx]

    def g7():
        res, rec = curv_gap_case("A-1-000", {}, "R-CURV")
        ins = {i["name"]: i["value"] for i in rec["inputs"]}
        assert rec["verdict"] == "apply" and ins["wall_level_before"] == 4 \
            and ins["wall_level_after"] == 6, (rec["verdict"], ins)
        h5, h6 = 0.5624 / 32, 0.5624 / 64
        assert h6 < ins["h_max"] < h5 and abs(ins["h_max"] - 0.1012 / 8) < 2e-4, \
            (ins["h_max"], h5, h6)
        _, rec2 = curv_gap_case("A-1-000", {"curvature_radius_p5_m": 10.0}, "R-CURV")
        assert rec2["verdict"] == "pass", rec2["message"]
        _, rec3 = curv_gap_case("A-1-000", {"curvature_radius_p5_m": 1e-4}, "R-CURV")
        ins3 = {i["name"]: i["value"] for i in rec3["inputs"]}
        assert rec3["verdict"] == "apply" and ins3["wall_level_after"] == 6 \
            and ins3["capped"] == 1 and "capped" in rec3["message"]
        _, rec4 = curv_gap_case("A-1-000", {"curvature_radius_p5_m": None,
                                            "outer_gap_m": 0.06}, "R-GAP")
        ins4 = {i["name"]: i["value"] for i in rec4["inputs"]}
        assert rec4["verdict"] == "apply" and ins4["wall_level_before"] == 4 \
            and ins4["wall_level_after"] == 5
        _, rec5 = curv_gap_case("A-1-000", {"curvature_radius_p5_m": None,
                                            "outer_gap_m": 10.0}, "R-GAP")
        assert rec5["verdict"] == "pass"
        _, rec6 = curv_gap_case("A-1-000", {"curvature_radius_p5_m": None,
                                            "outer_gap_m": 1e-4}, "R-GAP")
        ins6 = {i["name"]: i["value"] for i in rec6["inputs"]}
        assert rec6["verdict"] == "apply" and ins6["wall_level_after"] == 6 \
            and ins6["capped"] == 1
        for rec_a in (rec, rec3, rec4, rec6):
            growth_edits = [e for e in rec_a["edits"]
                            if e["pointer"] == "/layers/growth"]
            new_h = ins_of(rec_a)["h_after"] if "h_after" in ins_of(rec_a) else None
            if new_h is not None:
                fit = fit_growth(res["summary"]["t1_m"], 8, new_h, knobs)
                want = fit if fit is not None else 1.0
                for e in growth_edits:
                    assert abs(e["to"] - want) < 1e-12, (e, want)
        return ("raise, pass and cap on 6 synthetic fingerprints (R-CURV 4->6, "
                "pass, 6 capped; R-GAP 4->5, pass, 6 capped)")

    def ins_of(rec):
        return {i["name"]: i["value"] for i in rec["inputs"]}

    def g8():
        res_a = setup_of("A-1-000", fp_of("A-1-000", {"curvature_radius_p5_m": 10.0}))
        assert res_a["summary"]["feature_level"] == 5, res_a["summary"]
        res_b = setup_of("A-1-000", fp_of("A-1-000", {"curvature_radius_p5_m": 1e-4}))
        assert res_b["summary"]["feature_level"] == 6
        rec_f = res_b["records"][6]
        assert "capped" in rec_f["message"] and rec_f["verdict"] == "apply"
        res_c = setup_of("B-1-000")
        assert res_c["records"][6]["verdict"] == "pass", res_c["records"][6]["message"]
        res_d = cube_setup()
        assert res_d["records"][6]["verdict"] == "abstain"
        for res in (res_a, res_b, res_c, res_d):
            for r in res["records"]:
                for e in r["edits"]:
                    assert not e["pointer"].startswith("/snap/") \
                        or r["rule_id"] in ("R-PLANE", "R-FEAT", "R-BUDGET"), (r["rule_id"], e)
                    assert not e["pointer"].startswith("/snap/") or r["rule_id"] == "R-PLANE" \
                        or e["pointer"] == "/snap/feature_tolerance", (r["rule_id"], e)
        mls = []
        for res in (res_a, res_b):
            ml = res["config"]["refinement"]["max_level"]
            mls.append(ml)
            assert res["config"]["snap"] == {"feature_tolerance": 0.5 * 2.0 ** -ml}, \
                res["config"]["snap"]
            assert any(e["pointer"] == "/snap/feature_tolerance"
                       for e in res["records"][6]["edits"]), res["records"][6]["edits"]
        assert "snap" not in res_c["config"], res_c["config"].get("snap")
        assert res_d["config"]["snap"]["feature_tolerance"] == 0
        return ("+1, capped at 6, pass on smooth, abstain under R-PLANE; R-FEAT asks for "
                "tau = h_f/2 (feature_tolerance 0.5 * 2^-max_level at max_level %d and %d); "
                "only R-PLANE, R-FEAT and R-BUDGET write /snap/*" % tuple(mls))

    def g9():
        res = cube_setup()
        p = predict_cells(res["config"], _fp_of(_fwd(preflight.CUBEP_STL), "CUBEP"), 8)
        for k, want in (("n_base", 5175), ("leaves", 244639.2659425842),
                        ("feature", 0.0), ("layer_cells", 115248.0),
                        ("total", 359887.2659425842)):
            assert abs(p[k] - want) <= 1e-9 * max(1.0, abs(want)), (k, p[k], want)
        fp = dict(fp_of("A-1-000")); fp["area_m2"] = fp_of("A-1-000")["area_m2"] * 8
        res8 = setup_of("A-1-000", fp)
        rec8 = res8["records"][7]
        ladder = next(i["value"] for i in rec8["inputs"] if i["name"] == "ladder")
        assert _ladder_in_order(ladder), ladder[:3]
        assert rec8["verdict"] == "apply", rec8["message"]
        fp4 = dict(fp_of("A-1-000")); fp4["area_m2"] = fp_of("A-1-000")["area_m2"] * 400
        res400 = setup_of("A-1-000", fp4)
        assert res400["refused"] == ["R-BUDGET"] and res400["config"] is None
        rec400 = res400["records"][7]
        ladder4 = next(i["value"] for i in rec400["inputs"] if i["name"] == "ladder")
        target = 0.7 * gates["cell_budget"]
        assert all(e["predicted"] > target for e in ladder4)
        assert ladder4[-1]["wall_level"] == res400["summary"]["win_level"] \
            and ladder4[-1]["far"] == 0.25 and ladder4[-1]["wall"] == 0.25
        return ("cubep predicts %d cells (leaves %.2f, layer cells %.0f); the ladder "
                "runs far -> wall -> feature -> level; a refusal re-derives"
                % (round(p["total"]), p["leaves"], p["layer_cells"]))

    def _ladder_in_order(ladder: list) -> bool:
        segs, i = [], 0
        while i < len(ladder):
            lv = ladder[i]["wall_level"]
            seg = []
            while i < len(ladder) and ladder[i]["wall_level"] == lv:
                seg.append((ladder[i]["far"], ladder[i]["wall"], ladder[i]["feature"]))
                i += 1
            segs.append((lv, seg))
        pat = list(BUDGET_RUNGS)

        def is_prefix(seg):
            return len(seg) <= len(pat) and all(a == b for a, b in zip(seg, pat))

        return all(lv2 <= lv1 for (lv1, _), (lv2, _) in zip(segs, segs[1:])) \
            and all(is_prefix(seg) for _, seg in segs)

    def g10():
        for gid in FIVE:
            res = setup_of(gid)
            assert len(res["records"]) == 8 and res["verdict"] == "apply", gid
            cfg = res["config"]
            pf = preflight.preflight(cfg, fingerprint=fp_of(gid),
                                     flow=by_id[gid]["flow"], gates=gates, knobs=knobs)
            assert pf["verdict"] == "pass", (gid, pf["refused"])
        cfgs = [(gid, os.path.join(tmp, "pf_" + gid + ".json")) for gid in FIVE]
        for gid, cp in cfgs:
            with open(cp, "w", encoding="utf-8") as fh:
                json.dump(setups[gid]["config"], fh, indent=1)
        with ThreadPoolExecutor(max_workers=4) as ex:
            codes = dict(ex.map(lambda t: (t[0], preflight.run_dryrun(
                BINARY_DEFAULT, t[1])["exit_code"]), cfgs))
        assert all(c == 0 for c in codes.values()), codes
        return ("A-1-000, B-1-000, D-1-001, E-1-000, F-1-001 -> 8 valid records each, "
                "preflight pass, -dryRun exit 0")

    def g11():
        n = 0
        for res in list(setups.values()) + extra_setups:
            for r in res["records"]:
                for e in r["edits"]:
                    if e["to"] is None:
                        continue
                    n += 1
                    assert schema.check_edit(e["pointer"], e["to"], knobs) is None, \
                        (r["rule_id"], e)
                    assert not e["pointer"].startswith("/quality"), e
                    assert e["pointer"] not in ("/layers/cell_frac",
                                                "/layers/medial_frac",
                                                "/layers/min_thickness"), e
        return "%d edits, every one passes schema.check_edit; none under /quality, " \
            "/layers/cell_frac, /layers/medial_frac, /layers/min_thickness" % n

    def g12():
        n = 0
        for res in list(setups.values()) + extra_setups:
            assert [r["rule_id"] for r in res["records"]] == list(RULES)
            for r in res["records"]:
                n += 1
                assert not schema.errors(r, "DecisionRecord"), r["rule_id"]
                assert r["message"].startswith(r["rule_id"] + ": "), r["message"][:20]
                assert r["verdict"] in ("apply", "pass", "abstain", "refuse")
        return ("%d records valid; every message starts with its rule id; verdicts in "
                "{apply, pass, abstain, refuse}; 8 per setup in RULES order" % n)

    def g13():
        a = setup_of("A-1-000")
        b = setup_of("A-1-000", {}, )
        assert a["config_sha256"] == b["config_sha256"]
        ra = [dict(r, t="") for r in a["records"]]
        rb = [dict(r, t="") for r in b["records"]]
        assert ra == rb
        return ("two setups equal (config sha %s), records equal apart from t"
                % a["config_sha256"][:12])

    def g14():
        import split as sp
        out14 = os.path.join(tmp, "cli")
        os.makedirs(out14, exist_ok=True)
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        me = os.path.abspath(__file__)

        def run(args, timeout=180):
            return subprocess.run([sys.executable, me, *args], capture_output=True,
                                  text=True, encoding="utf-8", errors="replace",
                                  env=env, timeout=timeout)

        p = run(["--id", "A-1-000", "--out-dir", out14])
        assert p.returncode == 0, (p.returncode, p.stderr[-800:])
        rec_lines = [l for l in p.stdout.splitlines() if l.startswith("[")]
        assert len(rec_lines) == 8, rec_lines
        assert p.stdout.strip().splitlines()[-1].startswith("RULES APPLY A-1-000 ")
        lock = sp.read_lock(os.path.join(sp.MANIFEST_DIR, "split.lock"))
        tid = lock["test_ids"].split()[0]
        q = run(["--id", tid, "--out-dir", out14])
        assert q.returncode == 2, (q.returncode, q.stderr[-400:])
        assert "held-out test split" in q.stderr, q.stderr[-400:]
        flow_path = os.path.join(out14, "flow.json")
        with open(flow_path, "w", encoding="utf-8") as fh:
            json.dump(WORKED_FLOW, fh)
        cfg_out = os.path.join(out14, "cubep_cli.json")
        r = run([_fwd(preflight.CUBEP_STL), "--flow", flow_path, "--out", cfg_out,
                 "--name", "cubep_cli", "--case-dir", _fwd(out14) + "/case_cli"])
        assert r.returncode == 0, (r.returncode, r.stderr[-800:])
        assert r.stdout.strip().splitlines()[-1].startswith("RULES APPLY cubep_cli ")
        with open(cfg_out, encoding="utf-8") as fh:
            json.load(fh)
        return ("--id A-1-000 exit 0 with 8 record lines, a test id exits 2 naming "
                "the seal, the STL form writes a config")

    def g15():
        assert ft_radius_of([]) is False
        res_n = setup_of("A-1-000")
        res_o = setup(by_id["A-1-000"], fp_of("A-1-000"), stl_of("A-1-000"),
                      tmp + "/case_A-1-000", "A-1-000", gates=gates, knobs=knobs,
                      ft_radius=False)
        assert ft_radius_of(res_n["records"]) is True
        assert ft_radius_of([{"record": r} for r in res_n["records"]]) is True
        assert ft_radius_of(res_o["records"]) is False
        assert "snap" not in res_o["config"], res_o["config"].get("snap")
        assert "stays at its default 0.5" in res_o["records"][6]["message"]
        ml = res_n["config"]["refinement"]["max_level"]
        assert diff_edits(res_o["config"], res_n["config"]) == [
            {"pointer": "/snap/feature_tolerance", "from": None, "to": 0.5 * 2.0 ** -ml}]
        # R-BUDGET drops A-1-002's feature bump (max_level 6 -> 5): the radius follows it
        res_b = setup_of("A-1-002")
        rb = res_b["records"][7]
        assert res_b["config"]["refinement"]["max_level"] == 5, res_b["summary"]
        assert res_b["config"]["snap"] == {"feature_tolerance": 0.015625}
        assert {"pointer": "/snap/feature_tolerance", "from": 0.0078125,
                "to": 0.015625} in rb["edits"], rb["edits"]
        assert "follows max_level 5" in rb["message"], rb["message"]
        # a smooth body and the R-PLANE cube: FT-RADIUS changes nothing
        old_b = setup(by_id["B-1-000"], fp_of("B-1-000"), stl_of("B-1-000"),
                      tmp + "/case_B-1-000", "B-1-000", gates=gates, knobs=knobs,
                      ft_radius=False)
        assert old_b["config_sha256"] == setup_of("B-1-000")["config_sha256"]
        stl = _fwd(preflight.CUBEP_STL)
        old_c = setup({"geometry_id": "CUBEP", "flow": WORKED_FLOW}, _fp_of(stl, "CUBEP"),
                      stl, tmp + "/case_cubep", "CUBEP", gates=gates, knobs=knobs,
                      ft_radius=False)
        assert old_c["config_sha256"] == cube_setup()["config_sha256"]
        return ("A-1-000 differs from its pre-FT-RADIUS config in /snap/feature_tolerance "
                "alone (%g); A-1-002's radius follows R-BUDGET to max_level 5 (0.015625); "
                "B-1-000 and cubep are unchanged; ft_radius_of tells the two apart"
                % (0.5 * 2.0 ** -ml))

    def g16():
        pop = []
        for fam in FT_FAMILIES:
            for k in range(12):
                pop.append({"geometry_id": "%s-9-%03d" % (fam, k), "family": fam,
                            "stratum": ("easy", "medium", "hard")[k % 3], "class": "sharp"})
            pop.append({"geometry_id": fam + "-9-900", "family": fam, "stratum": "easy",
                        "class": "plane"})
        s1 = [p["geometry_id"] for p in ft_sample(pop)]
        s2 = [p["geometry_id"] for p in ft_sample(list(reversed(pop)))]
        assert s1 == s2 and len(s1) == 60 and len(set(s1)) == 60, s1
        assert all(sum(1 for g in s1 if g[0] == fam) == 10 for fam in FT_FAMILIES), s1
        assert not any(g.endswith("-900") for g in s1), s1
        short = [p for p in pop if p["geometry_id"] not in ("G-9-000", "G-9-001", "G-9-002")]
        try:
            ft_sample(short)
        except RulesError as e:
            assert "family G holds 9" in str(e), str(e)
        else:
            raise AssertionError("a family with 9 sharp rows was sampled")

        def rows(n_clean):
            return [{"family": FT_FAMILIES[i // 10], "f3_clean": i < n_clean,
                     "feature_capture": 0.5, "terminal": "EXHAUSTED",
                     "flags": {"F1": False}, "tau_over_h_f": 0.5,
                     "recorded_f3abcd_clean": False} for i in range(60)]
        good, h0 = {"ok": True}, {"harness_errors": 0}
        assert ft_summary(rows(20), good, h0)["verdict"] == "PASS"
        assert ft_summary(rows(19), good, h0)["verdict"] == "FAIL"
        assert ft_summary(rows(60), {"ok": False}, h0)["verdict"] == "FAIL"
        assert ft_summary(rows(60), good, {"harness_errors": 1})["verdict"] == "FAIL"
        off = rows(60)
        off[5]["tau_over_h_f"] = 16.0
        assert ft_summary(off, good, h0)["verdict"] == "FAIL"
        assert ft_summary(rows(20)[:59], good, h0)["verdict"] == "FAIL"
        return ("ft_sample draws 10 sharp rows per family in a fixed order and refuses a "
                "short family; ft_summary passes at 20 of 60 F3-clean and fails at 19, on a "
                "broken identity, a harness error, a radius off h_f/2 or a missing row")

    failed = 0
    try:
        for name, fn in (("R-YP", g1), ("R-WIN", g2), ("window table", g3),
                         ("fit_growth", g4), ("R-DOM", g5), ("R-PLANE", g6),
                         ("R-CURV / R-GAP", g7), ("R-FEAT", g8), ("R-BUDGET", g9),
                         ("setup", g10), ("whitelist", g11), ("records", g12),
                         ("determinism", g13), ("cli", g14), ("FT-RADIUS rule", g15),
                         ("FT-RADIUS gate", g16)):
            failed += group(name, fn)
            if failed:
                for l in lines:
                    print(l)
                return 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for l in lines:
        print(l)
    print("SELFTEST PASS")
    return 0


# --- the FT-RADIUS gate: sharp tuning bodies at tau = h_f / 2 ------------------

FT_GATE_SCHEMA = "autonomy-ft-radius-gate/1"
FT_FAMILIES = ("A", "B", "D", "E", "F", "G")
FT_PER_FAMILY = 10
FT_CLEAN_MIN = 20      # the gate, fixed before the run: >= 1/3 of 60 F3-clean (G-PILOT's bar)
FT_SALT = "ft-radius/1"
FT_F3 = ("F3a", "F3b", "F3c", "F3d", "F3e")
FT_BUNDLE = os.path.join(HERE, "prior", "tuning_rules.json.gz")
FT_HEADER = ("meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). "
             "Source-available, not Open Source. No GPL-licensed source was consulted.")


def ft_population(bundle, rows, gates=None, knobs=None) -> list:
    """Every tuning row, classed by today's setup on the fingerprint the committed rules
    campaign recorded: no_fingerprint, refused, plane (R-PLANE), smooth (no sharp edge)
    or sharp (the rows FT-RADIUS changes); the stl and case paths are campaign.py's."""
    ends = {g["geometry_id"]: g for g in bundle["geometries"]}
    out = []
    for row in rows:
        gid = row["geometry_id"]
        fp = (ends.get(gid) or {}).get("fingerprint")
        p = {"geometry_id": gid, "family": row["family"], "stratum": row["stratum"],
             "row": row, "fingerprint": fp, "result": None, "class": "no_fingerprint"}
        if fp is not None:
            res = setup(row, fp, "stl/%s.stl" % gid, "cases/%s" % gid, gid,
                        gates=gates, knobs=knobs)
            p["result"] = res
            if res["verdict"] != "apply":
                p["class"] = "refused"
            elif res["summary"]["plane"]:
                p["class"] = "plane"
            elif fp["sharp_edge_length_m"] == 0:
                p["class"] = "smooth"
            else:
                p["class"] = "sharp"
        out.append(p)
    return out


def ft_sample(pop) -> list:
    """FT_PER_FAMILY sharp rows of each family in FT_FAMILIES: the strata taken round
    robin (easy, medium, hard), each stratum in sha256(FT_SALT|id) order; a family
    short of sharp rows raises RulesError."""
    import hashlib
    picked = []
    for fam in FT_FAMILIES:
        by = {}
        for p in pop:
            if p["family"] == fam and p["class"] == "sharp":
                by.setdefault(p["stratum"], []).append(p)
        for s in by:
            by[s].sort(key=lambda p: hashlib.sha256(
                ("%s|%s" % (FT_SALT, p["geometry_id"])).encode("ascii")).hexdigest())
        got, idx = [], {"easy": 0, "medium": 0, "hard": 0}
        while len(got) < FT_PER_FAMILY:
            moved = False
            for s in ("easy", "medium", "hard"):
                if len(got) < FT_PER_FAMILY and idx[s] < len(by.get(s, [])):
                    got.append(by[s][idx[s]])
                    idx[s] += 1
                    moved = True
            if not moved:
                raise RulesError("ft_sample: family %s holds %d sharp rows, not %d"
                                 % (fam, len(got), FT_PER_FAMILY))
        picked.extend(got)
    return picked


def ft_identity(pop, bundle, gates=None, knobs=None) -> dict:
    """The byte-identity check against the committed rules campaign, recorded before
    FT-RADIUS: setup(..., ft_radius=False) rebuilds every recorded attempt-1 config
    sha; plane and smooth configs are unchanged by FT-RADIUS; a sharp config differs
    in /snap/feature_tolerance alone, at 0.5 * 2**-max_level; a refusal is recorded."""
    rec1 = {r["geometry_id"]: r for r in bundle["attempts"] if r["attempt"] == 1}
    ends = {g["geometry_id"]: g for g in bundle["geometries"]}
    counts = {c: {"n": 0, "ok": 0, "recorded": 0}
              for c in ("plane", "smooth", "sharp", "refused")}
    bad = []
    for p in pop:
        cls, gid, new = p["class"], p["geometry_id"], p["result"]
        if cls == "no_fingerprint":
            continue
        r1 = rec1.get(gid)
        if cls == "refused":
            end = ends[gid]
            ok = end["terminal"] == "REFUSED" and \
                sorted(end["refused"]) == sorted(new["refused"])
        else:
            old = setup(p["row"], p["fingerprint"], "stl/%s.stl" % gid, "cases/%s" % gid,
                        gid, gates=gates, knobs=knobs, ft_radius=False)
            ok = r1 is None or r1["config_sha"] == old["config_sha256"]
            if cls in ("plane", "smooth"):
                ok = ok and new["config_sha256"] == old["config_sha256"]
            else:
                ml = new["config"]["refinement"]["max_level"]
                eds = diff_edits(old["config"], new["config"])
                ok = ok and eds == [{"pointer": "/snap/feature_tolerance", "from": None,
                                     "to": FT_HALF_H_F * 2.0 ** -ml}]
        c = counts[cls]
        c["n"] += 1
        c["ok"] += 1 if ok else 0
        c["recorded"] += 1 if r1 is not None else 0
        if not ok:
            bad.append(gid)
    return {"counts": counts, "bad": bad, "ok": not bad}


def ft_read(cdir, ids) -> tuple:
    """(header, {gid: (end record, attempt-1 row, attempt-1 config)}, campaign end) of a
    `campaign.py --run --manifest tuning --mode rules --ablate remedies` run over
    exactly `ids`; any other campaign raises RulesError."""
    import campaign
    with open(os.path.join(cdir, campaign.FILES["campaign"]), encoding="utf-8") as fh:
        head = json.load(fh)
    for k, v in (("mode", "rules"), ("system", "rules"), ("ablate", ["remedies"])):
        if head.get(k) != v:
            raise RulesError("ft_read: %s says %s = %r, not %r" % (cdir, k, head.get(k), v))
    if (head.get("manifest") or {}).get("source") != "tuning":
        raise RulesError("ft_read: %s is not a tuning campaign" % cdir)
    if sorted(head.get("geometry_ids") or []) != sorted(ids):
        raise RulesError("ft_read: %s ran %d geometries, not the %d sampled ones"
                         % (cdir, len(head.get("geometry_ids") or []), len(ids)))
    ends = {g["geometry_id"]: g for g in campaign.load_geometries(cdir)}
    rows = {r["geometry_id"]: r for r in campaign.load_rows(cdir) if r["attempt"] == 1}
    got = {}
    for gid in ids:
        cfg = None
        if gid in rows:
            with open(os.path.join(cdir, campaign.config_rel(gid, 1)), encoding="utf-8") as fh:
                cfg = json.load(fh)
        got[gid] = (ends.get(gid), rows.get(gid), cfg)
    end = None
    end_path = os.path.join(cdir, campaign.FILES["end"])
    if os.path.isfile(end_path):
        with open(end_path, encoding="utf-8") as fh:
            end = json.load(fh)
    return head, got, end


def ft_row(p, end, row, cfg, recorded) -> dict:
    """One sampled geometry: its attempt 1 at tau = h_f / 2 (`row`, `cfg`) beside the
    committed rules campaign's attempt 1 at the default radius (`recorded`).  F3-clean
    is a written mesh (F1 false) with none of F3a-F3e true."""
    oc = row["outcome"] if row else None
    fl = oc["flags"] if oc else {}
    clean = oc is not None and fl.get("F1") is False \
        and not any(fl.get(k) is True for k in FT_F3)
    roc = (recorded or {}).get("outcome") or {}
    rfl = roc.get("flags") or {}
    rclean = bool(rfl) and rfl.get("F1") is False \
        and not any(rfl.get(k) is True for k in FT_F3[:4])
    tau = None
    if cfg is not None:
        ft = (cfg.get("snap") or {}).get("feature_tolerance")
        ml = (cfg.get("refinement") or {}).get("max_level")
        tau = ft * 2 ** ml if ft is not None and ml is not None else None
    got = (lambda k: oc.get(k)) if oc else (lambda k: None)
    return {"geometry_id": p["geometry_id"], "family": p["family"],
            "stratum": p["stratum"], "terminal": end["terminal"] if end else None,
            "f3_clean": clean,
            "flags": {k: fl.get(k) for k in ("F1",) + FT_F3} if oc else None,
            "feature_capture": got("feature_capture"), "failure_class": got("failure_class"),
            "pinned_frac": got("pinned_frac"), "p99_over_hf": got("p99_over_hf"),
            "max_over_hf": got("max_over_hf"), "n_cells": got("n_cells"),
            "seconds": got("seconds"), "tau_over_h_f": tau,
            "recorded_f3abcd_clean": rclean,
            "recorded_failure_class": roc.get("failure_class")}


def ft_summary(rows, identity, end) -> dict:
    """The gate's numbers from the rows alone.  PASS needs all 60 rows run, at least
    FT_CLEAN_MIN of them F3-clean, the identity held, every attempt 1 at tau = h_f / 2
    and zero harness errors; the capture share is reported, never gated (D-L5)."""
    import statistics
    caps = sorted(r["feature_capture"] for r in rows if r["feature_capture"] is not None)
    fams = {}
    for fam in FT_FAMILIES:
        fr = [r for r in rows if r["family"] == fam]
        fc = [r["feature_capture"] for r in fr if r["feature_capture"] is not None]
        fams[fam] = {"n": len(fr), "f3_clean": sum(1 for r in fr if r["f3_clean"]),
                     "capture_median": statistics.median(fc) if fc else None,
                     "recorded_f3abcd_clean": sum(1 for r in fr if r["recorded_f3abcd_clean"])}
    flags = {k: sum(1 for r in rows if r["flags"] and r["flags"].get(k) is True)
             for k in ("F1",) + FT_F3}
    n_clean = sum(1 for r in rows if r["f3_clean"])
    ran = sum(1 for r in rows if r["terminal"] is not None)
    at_radius = all(r["tau_over_h_f"] == FT_HALF_H_F for r in rows
                    if r["tau_over_h_f"] is not None)
    harness = (end or {}).get("harness_errors")
    ok = (len(rows) == FT_PER_FAMILY * len(FT_FAMILIES) and ran == len(rows)
          and n_clean >= FT_CLEAN_MIN and identity["ok"] and at_radius and harness == 0)
    return {"n": len(rows), "ran": ran, "f3_clean": n_clean, "f3_clean_min": FT_CLEAN_MIN,
            "share": n_clean / len(rows) if rows else 0.0,
            "capture": {"n": len(caps), "median": statistics.median(caps) if caps else None,
                        "min": caps[0] if caps else None, "max": caps[-1] if caps else None},
            "flags": flags, "families": fams, "at_radius": at_radius,
            "harness_errors": harness,
            "recorded_f3abcd_clean": sum(1 for r in rows if r["recorded_f3abcd_clean"]),
            "verdict": "PASS" if ok else "FAIL"}


def _ft4(x) -> str:
    return "-" if x is None else "%.4f" % x


def _ft_md_head(report: dict) -> list:
    """G-FT-RADIUS.md's header and verdict lines, every number read from the report."""
    s, ic = report["summary"], report["identity"]["counts"]
    cap = s["capture"]
    return [
        "<!-- %s -->" % FT_HEADER, "",
        "# G-FT-RADIUS - R-FEAT's attraction radius tau = h_f / 2 on sharp tuning bodies", "",
        "Date %s - binary sha256 `%s` - git HEAD `%s` - campaign `%s` (mode rules, remedies "
        "ablated: attempt 1 only)." % (report["date"], (report["binary_sha256"] or "?")[:16],
                                       report["git_head"][:12],
                                       report["campaign"]["campaign_id"]), "",
        "## Verdict - %s" % s["verdict"], "",
        "- F3-clean (a mesh, none of F3a-F3e): **%d of %d** (the gate, fixed before the run: "
        ">= %d); the committed rules campaign's attempt 1 at the default radius was "
        "F3a-F3d-clean on %d of them." % (s["f3_clean"], s["n"], s["f3_clean_min"],
                                          s["recorded_f3abcd_clean"]),
        "- Feature-edge capture (92.62) over %d meshes: median %s, min %s, max %s; F3e is "
        "still capture 0 (D-L5 is the user's)." % (cap["n"], _ft4(cap["median"]),
                                                  _ft4(cap["min"]), _ft4(cap["max"])),
        "- Flags set: " + ", ".join("%s %d" % kv for kv in sorted(s["flags"].items())) + ".",
        "- Every attempt 1 at tau = h_f / 2: %s; harness errors %s; %d of %d ran."
        % ("yes" if s["at_radius"] else "NO", s["harness_errors"], s["ran"], s["n"]),
        "- Identity against `%s`: " % report["bundle"] + "; ".join(
            "%s %d/%d (recorded %d)" % (c, ic[c]["ok"], ic[c]["n"], ic[c]["recorded"])
            for c in ("plane", "smooth", "sharp", "refused"))
        + "; bad: %s." % (", ".join(report["identity"]["bad"]) or "none"), ""]


def _ft_write(report: dict, report_dir: str) -> None:
    """The ONLY in-tree writes of --ft-gate: report_dir/G-FT-RADIUS.json and .md."""
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(report_dir, "G-FT-RADIUS.json"), "w", encoding="utf-8",
              newline="\n") as fh:
        fh.write(json.dumps(report, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    L = _ft_md_head(report)
    L += ["## By family", "",
          "| family | rows | F3-clean | recorded F3a-F3d-clean | median capture |",
          "|---|---|---|---|---|"]
    for fam, f in sorted(report["summary"]["families"].items()):
        L.append("| %s | %d | %d | %d | %s |" % (fam, f["n"], f["f3_clean"],
                                                 f["recorded_f3abcd_clean"],
                                                 _ft4(f["capture_median"])))
    L += ["", "## The 60 rows", "",
          "| geometry | family | stratum | terminal | F3-clean | capture | flags set |",
          "|---|---|---|---|---|---|---|"]
    for r in report["rows"]:
        on = [k for k, v in sorted((r["flags"] or {}).items()) if v is True]
        L.append("| %s | %s | %s | %s | %s | %s | %s |"
                 % (r["geometry_id"], r["family"], r["stratum"], r["terminal"],
                    "yes" if r["f3_clean"] else "no", _ft4(r["feature_capture"]),
                    " ".join(on) or "-"))
    with open(os.path.join(report_dir, "G-FT-RADIUS.md"), "w", encoding="utf-8",
              newline="\n") as fh:
        fh.write("\n".join(L) + "\n")


def ft_gate(cdir, bundle_path=FT_BUNDLE, report_dir=REPORT_DIR) -> int:
    """The FT-RADIUS gate over a finished attempt-1 campaign: the sample, the identity
    check, the rows and the verdict; writes report_dir/G-FT-RADIUS.json and .md."""
    import lreplay
    import split
    bundle = lreplay.load_bundle(bundle_path)
    gates, knobs = schema.load_gates(), schema.load_knobs()
    pop = ft_population(bundle, split.load("tuning", "rules"), gates=gates, knobs=knobs)
    picked = ft_sample(pop)
    ids = [p["geometry_id"] for p in picked]
    identity = ft_identity(pop, bundle, gates=gates, knobs=knobs)
    head, got, end = ft_read(cdir, ids)
    rec1 = {r["geometry_id"]: r for r in bundle["attempts"] if r["attempt"] == 1}
    rows = [ft_row(p, *got[p["geometry_id"]], rec1.get(p["geometry_id"])) for p in picked]
    summ = ft_summary(rows, identity, end)
    classes = {}
    for p in pop:
        classes[p["class"]] = classes.get(p["class"], 0) + 1
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                         text=True)
    report = {"$comment": FT_HEADER, "schema": FT_GATE_SCHEMA,
              "date": schema._now_iso()[:10], "git_head": git.stdout.strip(),
              "binary_sha256": head.get("binary_sha256"),
              "campaign": {k: head.get(k) for k in ("campaign_id", "git_sha", "streams",
                                                    "t_start")},
              "bundle": os.path.relpath(bundle_path, REPO).replace(os.sep, "/"),
              "population": classes, "sample_ids": ids, "identity": identity,
              "summary": summ, "rows": rows}
    _ft_write(report, report_dir)
    ic = identity["counts"]
    print("G-FT-RADIUS %s: %d of %d F3-clean (gate >= %d), median capture %s over %d "
          "meshes; identity plane %d/%d, smooth %d/%d, sharp %d/%d, refused %d/%d"
          % (summ["verdict"], summ["f3_clean"], summ["n"], FT_CLEAN_MIN,
             _ft4(summ["capture"]["median"]), summ["capture"]["n"],
             ic["plane"]["ok"], ic["plane"]["n"], ic["smooth"]["ok"], ic["smooth"]["n"],
             ic["sharp"]["ok"], ic["sharp"]["n"], ic["refused"]["ok"], ic["refused"]["n"]))
    return 0 if summ["verdict"] == "PASS" else 1


# --- the CLI (C9) ---------------------------------------------------------------

class _ArgParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write("rules: %s\n" % message)
        sys.exit(2)


def _emit(res: dict, cfg_path: str, records_path: str | None, as_json: bool) -> int:
    """The record lines / --json dump, the config write and the final line."""
    if records_path:
        with open(records_path, "a", encoding="utf-8") as fh:
            for rec in res["records"]:
                fh.write(json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n")
    if as_json:
        print(json.dumps(res, indent=1, ensure_ascii=False))
    else:
        for rec in res["records"]:
            print("[%s] %s" % (rec["verdict"], rec["message"]))
    if res["verdict"] == "apply":
        with open(cfg_path, "w", encoding="utf-8") as fh:
            json.dump(res["config"], fh, indent=1)
        print("RULES APPLY %s %s" % (res["geometry_id"], res["config_sha256"][:12]))
        return 0
    print("RULES REFUSED: %s" % res["refused"][0])
    return 3


def _cmd_id(args) -> int:
    import split as sp
    try:
        sp.refuse_test(args.id, "rules")
    except (sp.SplitSealed, sp.SplitError) as e:
        sys.stderr.write("rules: %s\n" % e)
        return 2
    row = next((r for r in sp.load("tuning", "rules")
                if r["geometry_id"] == args.id), None)
    if row is None:
        sys.stderr.write("rules: %s is not a tuning row\n" % args.id)
        return 2
    if row["family"] == "G":
        sys.stderr.write("rules: family G rows need tools/geom/stl_repair.py first\n")
        return 2
    if not args.out_dir:
        sys.stderr.write("rules: --id needs --out-dir\n")
        return 2
    os.makedirs(args.out_dir, exist_ok=True)
    stl = _fwd(_gen_of(row["family"]).write_row(row, args.out_dir))
    fp = _fp_of(stl, args.id)
    case = _fwd(os.path.join(args.out_dir, "case_" + args.id))
    res = setup(row, fp, stl, case, args.id)
    return _emit(res, os.path.join(args.out_dir, args.id + ".json"),
                 args.records, args.json)


def _cmd_stl(args) -> int:
    if not args.flow or not args.out:
        sys.stderr.write("rules: the STL form needs --flow and --out\n")
        return 2
    with open(args.flow, encoding="utf-8") as fh:
        flow = json.load(fh)
    stl = _fwd(os.path.abspath(args.stl))
    name = args.name or os.path.splitext(os.path.basename(stl))[0]
    if args.fingerprint:
        with open(args.fingerprint, encoding="utf-8") as fh:
            fp = json.load(fh)
    else:
        fp = _fp_of(stl, name)
    case = args.case_dir or _fwd(os.path.join(
        os.path.dirname(os.path.abspath(args.out)), "case_" + name))
    res = setup({"geometry_id": name, "flow": flow}, fp, stl, case, name)
    return _emit(res, args.out, args.records, args.json)


def _cli_gate(argv: list[str]) -> int:
    ap = _ArgParser(prog="rules.py --gate")
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", default="1,2,3")
    ap.add_argument("--streams", type=int, default=6)
    ap.add_argument("--binary", default=BINARY_DEFAULT)
    args = ap.parse_args(argv)
    parts = []
    for tok in args.parts.split(","):
        tok = tok.strip()
        if tok == "2":
            parts += ["2A", "2B", "2D", "2E", "2F"]
        elif tok in ("1", "2A", "2B", "2D", "2E", "2F", "3"):
            parts.append(tok)
        else:
            sys.stderr.write("rules: unknown gate part %r\n" % tok)
            return 2
    if not os.path.exists(args.binary):
        sys.stderr.write("rules: the binary %s is missing\n" % args.binary)
        return 2
    try:
        return gate(parts, args.out, streams=max(1, args.streams), binary=args.binary)
    except (OSError, schema.SchemaError, schema.LockError) as e:
        sys.stderr.write("rules: %s\n" % e)
        return 2


def _cli_ft_sample(argv: list[str]) -> int:
    ap = _ArgParser(prog="rules.py --ft-sample")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    import hashlib
    import lreplay
    import split
    try:
        pop = ft_population(lreplay.load_bundle(FT_BUNDLE), split.load("tuning", "rules"))
        ids = [p["geometry_id"] for p in ft_sample(pop)]
    except (RulesError, OSError, ValueError) as e:
        sys.stderr.write("rules: %s\n" % e)
        return 2
    text = ",".join(ids)
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text + "\n")
    print("FT-SAMPLE %d ids, sha256 %s"
          % (len(ids), hashlib.sha256(text.encode("ascii")).hexdigest()[:16]))
    print(text)
    return 0


def _cli_ft_gate(argv: list[str]) -> int:
    ap = _ArgParser(prog="rules.py --ft-gate")
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--report-dir", default=REPORT_DIR)
    args = ap.parse_args(argv)
    try:
        return ft_gate(args.campaign, report_dir=args.report_dir)
    except (RulesError, OSError, ValueError, schema.SchemaError) as e:
        sys.stderr.write("rules: %s\n" % e)
        return 2


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        sys.stderr.write("rules: give --id GID, an STL path, --selftest or --gate\n")
        return 2
    if argv[0] == "--selftest":
        return selftest()
    if argv[0] == "--gate":
        return _cli_gate(argv[1:])
    if argv[0] == "--ft-sample":
        return _cli_ft_sample(argv[1:])
    if argv[0] == "--ft-gate":
        return _cli_ft_gate(argv[1:])
    ap = _ArgParser(prog="rules.py", description="the L1 setup rules (AM-9)")
    ap.add_argument("stl", nargs="?")
    ap.add_argument("--id")
    ap.add_argument("--out-dir")
    ap.add_argument("--flow")
    ap.add_argument("--out")
    ap.add_argument("--case-dir")
    ap.add_argument("--name")
    ap.add_argument("--fingerprint")
    ap.add_argument("--records")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    try:
        if args.id and args.stl:
            sys.stderr.write("rules: give --id GID or an STL path, not both\n")
            return 2
        if args.id:
            return _cmd_id(args)
        if args.stl:
            return _cmd_stl(args)
        sys.stderr.write("rules: give --id GID or an STL path\n")
        return 2
    except (RulesError, schema.SchemaError, OSError) as e:
        sys.stderr.write("rules: %s\n" % e)
        return 2


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    sys.exit(main())
