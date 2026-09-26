#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""
optimise.py - the L4 optimiser of docs/15 §C: surrogate proposals in a knob box.

Every tuning attempt's config is rebuilt from its row's config_delta and
checked against the row's config sha, so the optimiser learns from the five
committed bundles (the rules campaign, the prior's two evaluation rounds and
both baselines) without ever opening the sealed test split.  Each attempt is
described by the seventeen fingerprint features of prior.py plus fourteen knob
features of its config; a five-member bootstrap ensemble of histogram
gradient-boosted trees, drawn by geometry, predicts p_fail, BLC_8 and log10
cells, and five folds by crc32(geometry_id) give the out-of-geometry CV of
G-OPT.  Only where the remedies end EXHAUSTED with attempts left under the
locked K does campaign.py call the hook: it rebuilds the L1 config with
rules.setup, draws 256 scrambled Sobol points in the six-knob box around it,
filters them by the config-level L0 checks and the visited config shas, and
picks lexicographically (p_fail <= 0.2 within the cell budget, then max BLC_8,
then min cells, ties to the lower Sobol index).  The refinement rounds are
rules+opt campaigns over the geometries where that can happen, cross-fitted so
every proposal comes from a fold ensemble that never saw the geometry; G-OPT
passes on fail AUC >= 0.75 and BLC_8 RMSE <= 0.15, and the optimiser ships
enabled only when the CV holds AND it beats the rules on the tuning split.

    python tools/autonomy/optimise.py --selftest
    python tools/autonomy/optimise.py --plan
    python tools/autonomy/optimise.py --refine --work DIR [--rounds N] [--streams N]
    python tools/autonomy/optimise.py --check
"""
import argparse
import collections
import copy
import functools
import gzip
import hashlib
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zlib

import numpy
from scipy.stats import qmc
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for _p in (HERE, os.path.join(HERE, "corpus")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import baseline
import campaign
import explain
import preflight
import prior
import remedies
import rules
import schema
import split


class OptError(ValueError):
    """A refused request or a harness inconsistency - never a verdict."""


REPORT_DIR = os.path.join(HERE, "optimise")
MODEL_NAME = "opt_model.json"
TRAIN_NAME = "train.json.gz"
REPORT_NAME = "G-OPT.json"
REPORT_MD = "G-OPT.md"
ROUND_BUNDLE = "refine_r%d.json.gz"        # round j's campaign as a bundle
SOURCES = ("prior/tuning_rules.json.gz", "prior/eval_r1.json.gz",
           "prior/eval_r2.json.gz",
           "baseline/tuning_b0-template.json.gz", "baseline/tuning_b0-lhs.json.gz")
ROUNDS = 5
MEMBERS = 5
FOLDS = 5
BOOT_SEED = 7
SOBOL_SEED = 11
IMP_SEED = 17
POOL_M = 8                                 # 2 ** 8 = 256 Sobol points
P_FAIL_MAX = 0.2                           # docs/15 §C L4
AUC_MIN = 0.75
RMSE_MAX = 0.15            # docs/15 §F G-OPT, the CV half
MFR_GAIN_MIN = 0.03
BLC_GAIN_MIN = 0.05        # docs/15 §F G-OPT, the ablation bar on tuning
PRED_CHECK_N = 8
AUDIT_OFF = 2 ** 40                        # audit_mod for the rounds
HYPER = {"max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 15,
         "min_samples_leaf": 20, "l2_regularization": 1.0, "early_stopping": False}
WALL_OFFSETS = (-1, 0, 1)
FEATURE_OFFSETS = (0, 1, 2)
FEATURE_TOLS = (0.0, 0.25, 0.5)
# a body with sharp edges (README section D, 2026-09-26): feature_tolerance 0 is
# refused there
FEATURE_TOLS_SHARP = (0.25, 0.5)
SMOOTHING = (0, 1, 2, 3)
BAND_SCALE = (0.5, 2.0)
GROWTH_LO = 1.1
CONFIG_FEATURES = ("wall_level", "log10_base_over_lmax", "log10_hwall_over_lmax",
                   "log10_wallband_over_lmax", "log10_band_over_lmax",
                   "feature_offset", "feature_tolerance", "smoothing_passes",
                   "growth", "log10_hwall_over_t1", "layers_n",
                   "log10_predicted_cells", "on_plane", "max_level")
FEATURES = tuple(prior.FEATURES) + CONFIG_FEATURES          # 17 + 14 = 31
OPT_IDS = ("OPT-PICK", "OPT-NOFEAS", "OPT-PLANE", "OPT-DISABLED")
MODEL_SCHEMA = "autonomy-opt-model/1"
GATE_SCHEMA = "autonomy-opt-gate/1"
TRAIN_SCHEMA = "autonomy-opt-train/1"
CHECK_SCHEMA = "autonomy-opt-check/1"
HEADER = baseline.HEADER                   # the "$comment" of every JSON it writes
_MODEL = {"path": os.path.join(REPORT_DIR, MODEL_NAME), "v": None, "key": None}


# --- (C2) the rebuild: every tuning attempt's config from its row ------------


def set_pointer(cfg, pointer, value):
    """RFC 6901 set with container creation; None deletes (a list index pops)."""
    segs = pointer.split("/")[1:]
    node = cfg
    for i, seg in enumerate(segs[:-1]):
        if isinstance(node, list):
            j = int(seg)
            while len(node) <= j:
                node.append({})
            node = node[j]
            continue
        nxt = segs[i + 1]
        child = node.get(seg) if isinstance(node, dict) else None
        if not isinstance(child, (dict, list)):
            child = [] if nxt.isdigit() else {}
            node[seg] = child
        node = child
    last = segs[-1]
    if isinstance(node, list):
        j = int(last)
        if value is None:
            del node[j]
            return
        while len(node) <= j:
            node.append(None)
        node[j] = copy.deepcopy(value)
    elif value is None:
        node.pop(last, None)
    else:
        node[last] = copy.deepcopy(value)


def _del_key(edit):
    return tuple((1, int(s)) if s.isdigit() else (0, s)
                 for s in edit["pointer"].split("/")[1:])


def apply_edits(cfg, edits):
    """A deep copy with the set edits in order, then the deletions reversed."""
    out = copy.deepcopy(cfg)
    for e in edits:
        if e["to"] is not None:
            set_pointer(out, e["pointer"], e["to"])
    for e in sorted((e for e in edits if e["to"] is None), key=_del_key,
                    reverse=True):
        set_pointer(out, e["pointer"], None)
    return out


def check_bundle(bundle, name):
    """SplitSealed before any row of a bundle that is not tuning (docs/15 §F)."""
    h = bundle["campaign"]
    if h["split_mode"] == split.EVALUATE or h["manifest"]["source"] != "tuning":
        raise split.SplitSealed(
            "the optimiser learns from the tuning split only (docs/15 §F): %s is "
            "a %s campaign of %s" % (name, h["split_mode"],
                                     h["manifest"]["source"]))
    for r in bundle["attempts"]:
        if r["split"] != "tuning":
            raise split.SplitSealed(
                "the optimiser learns from the tuning split only (docs/15 §F): "
                "%s row %s attempt %s has split %s"
                % (name, r["geometry_id"], r["attempt"], r["split"]))


def rows_from(bundle, source, mrows, gates, knobs):
    """One training row per attempt row, every config rebuilt and sha-checked."""
    check_bundle(bundle, source)
    h = bundle["campaign"]
    system = h["system"]
    ends = {g["geometry_id"]: g for g in bundle["geometries"]}
    by_geom = collections.OrderedDict()
    for r in bundle["attempts"]:
        by_geom.setdefault(r["geometry_id"], {})[r["attempt"]] = r
    out = []
    # the rules the campaign was recorded under (a bundle written before FT-RADIUS
    # rebuilds its attempt-1 configs with the default attraction radius)
    ftr = rules.ft_radius_of(bundle.get("records") or [])
    for gid in sorted(by_geom):
        att = by_geom[gid]
        if sorted(att) != list(range(1, max(att) + 1)):
            raise OptError("attempts: %s of %s holds %s, not 1..%d"
                           % (gid, source, sorted(att), max(att)))
        end = ends[gid]
        fp = end["fingerprint"]
        if system in campaign.BASELINES:
            base = campaign.b0_template(mrows[gid], fp)
        else:
            s = rules.setup(mrows[gid], fp, campaign.stl_rel(gid),
                            campaign.case_rel(gid), gid, gates=gates, knobs=knobs,
                            ft_radius=ftr)
            if s["refused"]:
                raise OptError("rules.setup refuses %s (%s): %s"
                               % (gid, source, ", ".join(s["refused"])))
            base = s["config"]
        prev = None
        for a in sorted(att):
            row = att[a]
            parent = base if system == "b0-lhs" or a == 1 else prev
            cfg = apply_edits(parent, row["config_delta"])
            got = schema.canonical_sha256(cfg)
            if got != row["config_sha"]:
                raise OptError("reconstruct: %s attempt %d of %s rebuilds to %s, "
                               "the row says %s" % (gid, a, source, got[:12],
                                                    row["config_sha"][:12]))
            prev = cfg
            n = row["outcome"]["n_cells"]
            out.append({"geometry_id": gid, "family": end["family"],
                        "source": source, "attempt": a,
                        "config_sha256": row["config_sha"], "config": cfg,
                        "fingerprint": fp, "x": x_of(cfg, fp),
                        "fail": row["outcome"]["failure"] is True,
                        "blc8": float(row["outcome"]["blc8_a_priori"] or 0.0),
                        "log_cells": math.log10(n)
                        if isinstance(n, int) and n > 0 else None})
    return out


def training_rows(named, mrows, gates, knobs):
    """rows_from over every (source, bundle) in order; the first (gid, sha) wins."""
    rows = []
    seen = set()
    dup = 0
    for source, bundle in named:
        for r in rows_from(bundle, source, mrows, gates, knobs):
            key = (r["geometry_id"], r["config_sha256"])
            if key in seen:
                dup += 1
                continue
            seen.add(key)
            rows.append(r)
    return rows, dup


def _file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_sources(sources=None):
    """[(name, bundle, file sha256)] over the given paths (default SOURCES)."""
    paths = [os.path.join(HERE, s) for s in SOURCES] \
        if sources is None else list(sources)
    out = []
    for p in paths:
        ap = os.path.abspath(p)
        under = ap.startswith(HERE + os.sep)
        name = os.path.relpath(ap, HERE).replace(os.sep, "/") if under \
            else ap.replace(os.sep, "/")
        out.append((name, baseline.read_bundle(ap), _file_sha256(ap)))
    return out


def matrix(rows):
    """(X, y_fail, y_blc, y_lc, groups); a None log_cells becomes NaN."""
    X = numpy.array([r["x"] for r in rows], dtype=float)
    y_fail = numpy.array([r["fail"] for r in rows], dtype=bool)
    y_blc = numpy.array([r["blc8"] for r in rows], dtype=float)
    y_lc = numpy.array([r["log_cells"] if r["log_cells"] is not None
                        else float("nan") for r in rows], dtype=float)
    return X, y_fail, y_blc, y_lc, [r["geometry_id"] for r in rows]


# --- (C3) the 14 knob features of one config ---------------------------------


def config_features(cfg, fp):
    """The config as 14 python floats, in CONFIG_FEATURES order."""
    bb = fp["bbox"]
    big = max(bb[1] - bb[0], bb[3] - bb[2], bb[5] - bb[4])
    base = cfg["domain"]["base_size"]
    wl = remedies.wall_level(cfg)
    levels = cfg["refinement"]["levels"]
    bands = [b for e in levels for b in e.get("bands", [])]
    d = preflight.MESHER_DEFAULTS
    h = base / 2 ** wl
    lay = cfg.get("layers")
    n = lay.get("n", 0) if lay else 0
    at_wl = [b["distance"] for b in bands if b["level"] == wl]
    snap = cfg.get("snap") or {}
    return [float(wl),
            math.log10(base / big),
            math.log10(h / big),
            math.log10(max(at_wl) / big) if at_wl else float("nan"),
            math.log10(max(b["distance"] for b in bands) / big) if bands
            else float("nan"),
            float(max(0, max([e.get("feature_level", 0) for e in levels] + [0])
                      - wl)),
            float(snap.get("feature_tolerance", d["/snap/feature_tolerance"])),
            float(snap.get("smoothing_passes", d["/snap/smoothing_passes"])),
            float(lay.get("growth", d["/layers/growth"])) if lay else float("nan"),
            math.log10(h / lay["first_thickness"])
            if lay and lay.get("first_thickness") else float("nan"),
            float(n),
            math.log10(rules.predict_cells(cfg, fp, n)["total"]),
            1.0 if remedies.on_plane(cfg, fp) else 0.0,
            float(cfg["refinement"].get("max_level",
                                        d["/refinement/max_level"]))]


def x_of(cfg, fp):
    """The 31 features: prior.py's 17 fingerprint features, then the knobs."""
    return list(prior.features(fp)) + config_features(cfg, fp)


# --- (C4) the bootstrap ensemble and the cross-fit ---------------------------


def fit_ensemble(X, y_fail, y_blc, y_lc, groups, members=MEMBERS, seed=BOOT_SEED):
    """MEMBERS (fail, blc, cells) triples; the bootstrap draws geometries."""
    ids = sorted(set(groups))
    ens = []
    for m in range(members):
        rng = numpy.random.default_rng([seed, m])
        draw = rng.choice(len(ids), size=len(ids), replace=True)
        cnt = collections.Counter(ids[i] for i in draw)
        w = numpy.array([cnt.get(g, 0) for g in groups], dtype=float)
        sel = w > 0
        if len(set(bool(v) for v in y_fail[sel])) == 1:
            fail = {"const": 1.0 if y_fail[sel][0] else 0.0}
        else:
            fail = HistGradientBoostingClassifier(random_state=m, **HYPER)
            fail.fit(X[sel], y_fail[sel], sample_weight=w[sel])
        blc = HistGradientBoostingRegressor(random_state=m, **HYPER)
        blc.fit(X[sel], y_blc[sel], sample_weight=w[sel])
        ok = sel & numpy.isfinite(y_lc)
        if not ok.any():
            raise OptError("no cell counts in the training rows")
        cells = HistGradientBoostingRegressor(random_state=m, **HYPER)
        cells.fit(X[ok], y_lc[ok], sample_weight=w[ok])
        ens.append((fail, blc, cells))
    return ens


def predict(ens, Xq):
    """Mean and std (ddof 0) over the members of p_fail, BLC_8 and log10 cells."""
    ps, bs, ls = [], [], []
    for fail, blc, cells in ens:
        if isinstance(fail, dict):
            ps.append(numpy.full(len(Xq), fail["const"], dtype=float))
        else:
            col = list(fail.classes_).index(True)
            ps.append(fail.predict_proba(Xq)[:, col])
        bs.append(numpy.clip(blc.predict(Xq), 0.0, 1.0))
        ls.append(cells.predict(Xq))
    P, B, L = numpy.array(ps), numpy.array(bs), numpy.array(ls)
    return {"p_fail": P.mean(0), "p_fail_std": P.std(0),
            "blc8": B.mean(0), "blc8_std": B.std(0),
            "log_cells": L.mean(0), "log_cells_std": L.std(0)}


def fold_of(gid):
    return zlib.crc32(gid.encode("ascii")) % FOLDS


def cross_fit(X, y_fail, y_blc, y_lc, groups):
    """Fold ensembles (fold f trains without fold f) and the out-of-fold CV."""
    folds = [fold_of(g) for g in groups]
    ens_by_fold = {}
    p = numpy.full(len(groups), numpy.nan)
    b = numpy.full(len(groups), numpy.nan)
    l = numpy.full(len(groups), numpy.nan)
    fold_rows = {}
    for f in range(FOLDS):
        te = numpy.array([v == f for v in folds])
        if not te.any():
            continue
        fold_rows[f] = int(te.sum())
        tr = ~te
        ens_by_fold[f] = fit_ensemble(X[tr], y_fail[tr], y_blc[tr], y_lc[tr],
                                      [groups[i] for i in numpy.where(tr)[0]])
        pr = predict(ens_by_fold[f], X[te])
        p[te], b[te], l[te] = pr["p_fail"], pr["blc8"], pr["log_cells"]
    pos = y_blc > 0
    fin = numpy.isfinite(y_lc)
    metrics = {
        "auc": float(roc_auc_score(y_fail, p))
        if y_fail.any() and not y_fail.all() else None,
        "blc8_rmse": float(numpy.sqrt(numpy.mean((b - y_blc) ** 2))),
        "blc8_rmse_zero": float(numpy.sqrt(numpy.mean(y_blc ** 2))),
        "blc8_rmse_pos": float(numpy.sqrt(numpy.mean((b[pos] - y_blc[pos]) ** 2)))
        if pos.any() else None,
        "log_cells_rmse": float(numpy.sqrt(numpy.mean((l[fin] - y_lc[fin]) ** 2))),
        "brier": float(numpy.mean((p - y_fail.astype(float)) ** 2)),
        "n": len(groups), "n_fail": int(y_fail.sum()), "n_pos": int(pos.sum()),
        "n_geometries": len(set(groups)),
        "folds": sorted(fold_rows),
        "fold_rows": {str(f): n for f, n in sorted(fold_rows.items())}}
    return ens_by_fold, metrics


# --- (C5) the Sobol pool in the six-knob box around the L1 config ------------


def sobol(gid):
    """2 ** POOL_M scrambled Sobol points in the unit cube, seeded by the id."""
    return qmc.Sobol(d=6, scramble=True, rng=numpy.random.default_rng(
        [SOBOL_SEED, zlib.crc32(gid.encode("ascii"))])).random_base2(m=POOL_M)


def l1_of(ctx, gates, knobs):
    """The L1 config the box is relative to, from the row-less facts only."""
    gid = ctx["geometry_id"]
    s = rules.setup({"geometry_id": gid, "flow": ctx["flow"]}, ctx["fingerprint"],
                    campaign.stl_rel(gid), campaign.case_rel(gid), gid,
                    flow=ctx["flow"], gates=gates, knobs=knobs)
    if s["refused"]:
        raise OptError("rules.setup refuses %s: %s" % (gid, ", ".join(s["refused"])))
    return s["config"]


def pick(seq, x):
    return seq[min(len(seq) - 1, int(x * len(seq)))]


def point_config(l1, u, fp):
    """(cfg, knobs): the six-knob box of docs/15 §C L4 around the L1 config."""
    cfg = copy.deepcopy(l1)
    off = pick(WALL_OFFSETS, u[0])
    f = 0.5 * 4 ** u[1]
    fo = pick(FEATURE_OFFSETS, u[2])
    wl = min(6, max(0, remedies.wall_level(l1) + off))
    lv = []
    for e in cfg["refinement"]["levels"]:
        for b in e.get("bands", []):
            b["level"] = min(6, max(0, b["level"] + off))
            b["distance"] = round(b["distance"] * f, 6)
            lv.append(b["level"])
        if fo > 0:
            e["feature_level"] = min(6, wl + fo)
            lv.append(e["feature_level"])
        elif "feature_level" in e:
            e["feature_level"] = 0
    cfg["refinement"]["max_level"] = max(lv + [0])
    snap = cfg.setdefault("snap", {})
    snap["feature_tolerance"] = pick(
        FEATURE_TOLS_SHARP if fp["sharp_edge_length_m"] > 0 else FEATURE_TOLS, u[3])
    snap["smoothing_passes"] = pick(SMOOTHING, u[4])
    growth = None
    if "layers" in cfg:
        g1 = cfg["layers"].get("growth", preflight.MESHER_DEFAULTS["/layers/growth"])
        growth = round(GROWTH_LO + (g1 - GROWTH_LO) * u[5], 3) \
            if g1 > GROWTH_LO else g1
        cfg["layers"]["growth"] = growth
    knobs = {"wall_offset": off, "band_scale": round(f, 6), "feature_offset": fo,
             "feature_tolerance": snap["feature_tolerance"],
             "smoothing_passes": snap["smoothing_passes"], "growth": growth}
    return cfg, knobs


# --- (C6) the decision: rank the surviving pool points ------------------------

CITE = "docs/15 §C L4 optimiser, §F G-OPT; tools/autonomy/optimise/opt_model.json"
PLANE_CITE = "tools/autonomy/remedies.py on_plane (docs/15 §C L1 R-PLANE)"
FORMULA = ("256 scrambled Sobol points in the six-knob box around the L1 config, "
           "filtered by the config-level L0 checks; five bootstrap "
           "HistGradientBoosting members predict p_fail, BLC_8 and log10 cells; "
           "keep p_fail <= 0.2 and predicted cells <= the cell budget, then max "
           "BLC_8, then min cells, ties to the lower Sobol index")
_COUNT_KEYS = ("pool_n", "unique_n", "visited_n", "edit_refused_n",
               "l0_refused_n", "l0_pass_n", "feasible_n")


def _record(rid, verdict, trigger, inputs, formula, edits, cite, message,
            uncertainty):
    rec = {"schema": "autonomy-decision/1", "layer": "optimiser", "rule_id": rid,
           "verdict": verdict, "trigger": trigger, "inputs": inputs,
           "formula": formula, "edits": edits, "cite": cite, "message": message,
           "uncertainty": uncertainty, "t": schema._now_iso()}
    errs = schema.errors(rec, "DecisionRecord")
    if errs:
        raise OptError("the optimiser's record is invalid: %s" % errs[0])
    return rec


def rank(pred, indices, log_budget):
    """The feasible positions: p_fail and cells inside, best BLC_8 first."""
    pf, bl, lc = pred["p_fail"], pred["blc8"], pred["log_cells"]
    pos = [k for k in range(len(indices))
           if pf[k] <= P_FAIL_MAX and lc[k] <= log_budget]
    pos.sort(key=lambda k: (-bl[k], lc[k], indices[k]))
    return pos


def _counts_in(counts, gates):
    return [{"name": k, "value": int(counts[k]), "unit": "1"} for k in _COUNT_KEYS] \
        + [{"name": "p_fail_max", "value": P_FAIL_MAX, "unit": "1"},
           {"name": "cell_budget", "value": int(gates["cell_budget"]), "unit": "1"}]


def _meta_in(meta):
    return [{"name": "importances", "value": list(meta.get("importances") or []),
             "unit": ""},
            {"name": "model_sha256", "value": str(meta.get("model_sha256")),
             "unit": ""},
            {"name": "fold", "value": meta.get("fold"), "unit": ""}]


def _abstain(rid, rec, counts):
    return {"verdict": "abstain", "rule_id": rid, "config": None, "edits": [],
            "prediction": None, "record": rec, "counts": counts, "pick": None,
            "runners_up": []}


def decide(ens, ctx, history, gates, knobs, meta):
    """The hook's answer: OPT-PICK, or an abstain by name. (docs/15 §C L4)"""
    if "cwd" not in ctx:
        raise OptError("decide: ctx needs cwd (the campaign directory that holds "
                       "stl/%s.stl)" % ctx.get("geometry_id"))
    gid = ctx["geometry_id"]
    fp = ctx["fingerprint"]
    l1 = l1_of(ctx, gates, knobs)
    if remedies.on_plane(l1, fp):
        rec = _record(
            "OPT-PLANE", "abstain",
            {"observable": "remedies.on_plane", "value": True, "threshold": True,
             "op": "==", "source": PLANE_CITE},
            [{"name": "l1_config_sha256",
              "value": schema.canonical_sha256(l1), "unit": ""}],
            FORMULA, [], PLANE_CITE,
            "OPT-PLANE: the body is on the R-PLANE path, which owns the "
            "refinement and snap knobs; the optimiser abstains", 0.0)
        return _abstain("OPT-PLANE", rec,
                        dict.fromkeys(_COUNT_KEYS, 0) | {"pool_n": 2 ** POOL_M})
    visited = {h["config_sha256"] for h in history} \
        | {schema.canonical_sha256(ctx["config"])}
    seen = set()
    counts = {"pool_n": 2 ** POOL_M, "unique_n": 0, "visited_n": 0,
              "edit_refused_n": 0, "l0_refused_n": 0, "l0_pass_n": 0,
              "feasible_n": 0}
    candidates = []
    for i, u in enumerate(sobol(gid)):
        cfg, kn = point_config(l1, u, fp)
        s = schema.canonical_sha256(cfg)
        if s in seen:
            continue
        seen.add(s)
        if s in visited:
            counts["visited_n"] += 1
            continue
        counts["unique_n"] += 1
        edits = rules.diff_edits(ctx["config"], cfg)
        refused = [e for e in edits
                   if schema.check_edit(e["pointer"], e["to"], knobs) is not None]
        if refused:
            counts["edit_refused_n"] += 1
            continue
        pf = preflight.preflight(cfg, fingerprint=fp, flow=ctx["flow"],
                                 gates=gates, knobs=knobs, cwd=ctx["cwd"])
        if pf["refused"]:
            counts["l0_refused_n"] += 1
            continue
        counts["l0_pass_n"] += 1
        candidates.append({"index": i, "config": cfg, "config_sha256": s,
                           "knobs": kn, "edits": edits})
    log_budget = math.log10(gates["cell_budget"])
    if not candidates:
        rec = _record(
            "OPT-NOFEAS", "abstain",
            {"observable": "optimiser.feasible_n", "value": 0, "threshold": 0,
             "op": "==", "source": CITE},
            _counts_in(counts, gates) + [{"name": "best", "value": None,
                                          "unit": ""}] + _meta_in(meta),
            FORMULA, [], CITE,
            "OPT-NOFEAS: %s of %s pool points pass L0 and none has p_fail <= %s "
            "within the cell budget (lowest p_fail %s); the optimiser abstains"
            % (explain.fmt(counts["l0_pass_n"]), explain.fmt(counts["pool_n"]),
               explain.fmt(P_FAIL_MAX), "none"), 0.0)
        return _abstain("OPT-NOFEAS", rec, counts)
    pred = predict(ens, numpy.array([x_of(c["config"], fp) for c in candidates]))
    order = rank(pred, [c["index"] for c in candidates], log_budget)
    counts["feasible_n"] = len(order)
    if not order:
        k = min(range(len(candidates)),
                key=lambda j: (pred["p_fail"][j], candidates[j]["index"]))
        c = candidates[k]
        best = {"sobol_index": c["index"],
                "p_fail": float(pred["p_fail"][k]),
                "blc8_a_priori": float(pred["blc8"][k]),
                "log_cells": float(pred["log_cells"][k])}
        rec = _record(
            "OPT-NOFEAS", "abstain",
            {"observable": "optimiser.feasible_n", "value": 0, "threshold": 0,
             "op": "==", "source": CITE},
            _counts_in(counts, gates) + [{"name": "best",
                                          "value": copy.deepcopy(best),
                                          "unit": ""}] + _meta_in(meta),
            FORMULA, [], CITE,
            "OPT-NOFEAS: %s of %s pool points pass L0 and none has p_fail <= %s "
            "within the cell budget (lowest p_fail %s); the optimiser abstains"
            % (explain.fmt(counts["l0_pass_n"]), explain.fmt(counts["pool_n"]),
               explain.fmt(P_FAIL_MAX), explain.fmt(best["p_fail"])), 0.0)
        return _abstain("OPT-NOFEAS", rec, counts)
    return _pick_decision(candidates, pred, order, counts, ctx, gates, meta)


def _pick_decision(candidates, pred, order, counts, ctx, gates, meta):
    k0 = order[0]
    c = candidates[k0]
    pk = {"sobol_index": c["index"], "config_sha256": c["config_sha256"],
          "knobs": copy.deepcopy(c["knobs"]),
          "p_fail": float(pred["p_fail"][k0]),
          "p_fail_std": float(pred["p_fail_std"][k0]),
          "blc8_a_priori": float(pred["blc8"][k0]),
          "blc8_std": float(pred["blc8_std"][k0]),
          "log_cells": float(pred["log_cells"][k0]),
          "log_cells_std": float(pred["log_cells_std"][k0])}
    runners = [{"sobol_index": candidates[k]["index"],
                "config_sha256": candidates[k]["config_sha256"],
                "p_fail": float(pred["p_fail"][k]),
                "blc8_a_priori": float(pred["blc8"][k]),
                "log_cells": float(pred["log_cells"][k])} for k in order[1:4]]
    edits = rules.diff_edits(ctx["config"], c["config"])
    prediction = {"p_fail": pk["p_fail"], "p_fail_std": pk["p_fail_std"],
                  "blc8_a_priori": pk["blc8_a_priori"], "log_cells": pk["log_cells"],
                  "t_predicted": schema._now_iso()}
    rec = _record(
        "OPT-PICK", "apply",
        {"observable": "optimiser.p_fail", "value": pk["p_fail"],
         "threshold": P_FAIL_MAX, "op": "<=", "source": CITE},
        _counts_in(counts, gates)
        + [{"name": "p_fail", "value": pk["p_fail"], "unit": "1"},
           {"name": "p_fail_std", "value": pk["p_fail_std"], "unit": "1"},
           {"name": "blc8_a_priori", "value": pk["blc8_a_priori"], "unit": "1"},
           {"name": "blc8_std", "value": pk["blc8_std"], "unit": "1"},
           {"name": "log_cells", "value": pk["log_cells"], "unit": "1"},
           {"name": "log_cells_std", "value": pk["log_cells_std"], "unit": "1"},
           {"name": "knobs", "value": copy.deepcopy(pk["knobs"]), "unit": ""},
           {"name": "sobol_index", "value": int(pk["sobol_index"]), "unit": "1"},
           {"name": "runners_up", "value": copy.deepcopy(runners), "unit": ""}]
        + _meta_in(meta), FORMULA, edits, CITE,
        "OPT-PICK: %s of %s pool points pass L0 and %s are feasible (p_fail <= %s, "
        "predicted cells <= %s); the pick has p_fail %s +- %s, BLC_8 %s, log10 "
        "cells %s: %s" % (explain.fmt(counts["l0_pass_n"]),
                          explain.fmt(counts["pool_n"]),
                          explain.fmt(counts["feasible_n"]),
                          explain.fmt(P_FAIL_MAX),
                          explain.fmt(int(gates["cell_budget"])),
                          explain.fmt(pk["p_fail"]), explain.fmt(pk["p_fail_std"]),
                          explain.fmt(pk["blc8_a_priori"]),
                          explain.fmt(pk["log_cells"]),
                          explain.edits_text(edits)), pk["p_fail_std"])
    return {"verdict": "apply", "rule_id": "OPT-PICK", "config": c["config"],
            "edits": edits, "prediction": prediction, "record": rec,
            "counts": counts, "pick": pk, "runners_up": runners}


def disabled_decision(gate):
    """The OPT-DISABLED abstain a disabled hook returns on every geometry."""
    g = gate or {}
    auc, rmse = g.get("auc"), g.get("blc8_rmse")
    beats, verdict = g.get("beats_rules"), g.get("verdict")
    rec = _record(
        "OPT-DISABLED", "abstain",
        {"observable": "optimiser.enabled", "value": False, "threshold": False,
         "op": "==", "source": "optimise/G-OPT.json (docs/15 §F G-OPT)"},
        [{"name": "verdict", "value": verdict, "unit": ""},
         {"name": "auc", "value": auc, "unit": "1"},
         {"name": "blc8_rmse", "value": rmse, "unit": "1"},
         {"name": "beats_rules", "value": beats, "unit": ""},
         {"name": "model_sha256", "value": g.get("model_sha256"), "unit": ""}],
        FORMULA, [], CITE,
        "OPT-DISABLED: the optimiser ships disabled (G-OPT %s: CV AUC %s, BLC_8 "
        "RMSE %s, beats the rules %s); the remedies' terminal stands"
        % (explain.fmt(verdict), explain.fmt(auc), explain.fmt(rmse),
           explain.fmt(beats)), 0.0)
    return _abstain("OPT-DISABLED", rec, None)


# --- (C7) the hook campaign.py calls ------------------------------------------


def model_sha(model):
    """The model's hash: everything but the enabled flag and the gate."""
    return schema.canonical_sha256({k: v for k, v in model.items()
                                    if k not in ("enabled", "gate")})


def make_hook(ens_of, *, model_sha256, importances=(), enabled=True, gate=None):
    """hook(ctx, history) for campaign.py; gates and knobs load once."""
    gates = schema.load_gates()
    knobs = schema.load_knobs()

    def hook(ctx, history):
        if not enabled:
            return disabled_decision(gate)
        try:
            ens, fold = ens_of(ctx["geometry_id"])
        except KeyError as e:
            raise OptError("no ensemble for %s" % ctx["geometry_id"]) from e
        return decide(ens, ctx, history, gates, knobs,
                      {"model_sha256": model_sha256, "fold": fold,
                       "importances": list(importances)})
    return hook


def _train_rows_of(train):
    """(X, y_fail, y_blc, y_lc, groups) out of a written train object."""
    rows = train["rows"]
    X = numpy.array([[float("nan") if v is None else v for v in r["x"]]
                     for r in rows], dtype=float)
    y_fail = numpy.array([bool(r["fail"]) for r in rows], dtype=bool)
    y_blc = numpy.array([float(r["blc8"]) for r in rows], dtype=float)
    y_lc = numpy.array([float(r["log_cells"]) if r["log_cells"] is not None
                        else float("nan") for r in rows], dtype=float)
    return X, y_fail, y_blc, y_lc, [r["geometry_id"] for r in rows]


def load_model(path=None):
    """The gated optimiser model, refit and drift-checked; cached by mtime."""
    path = path or _MODEL["path"]
    key = (path, os.stat(path).st_mtime_ns) if os.path.isfile(path) else None
    if key is not None and _MODEL["key"] == key and _MODEL["v"] is not None:
        return _MODEL["v"]
    if not os.path.isfile(path):
        raise OptError("no optimiser model at %s (run optimise.py --refine)" % path)
    with open(path, encoding="utf-8") as f:
        model = json.load(f)
    if not isinstance(model, dict) or model.get("schema") != MODEL_SCHEMA:
        raise OptError("%s is not a %s model" % (path, MODEL_SCHEMA))
    if model.get("gate") is None:
        raise OptError("%s has not been gated (run optimise.py --refine)" % path)
    train_path = os.path.join(os.path.dirname(path), model["train"]["file"])
    with open(train_path, "rb") as f:
        data = f.read()
    if hashlib.sha256(data).hexdigest() != model["train"]["sha256"]:
        raise OptError("train file drift: %s does not match the model's train "
                       "sha256" % train_path)
    train = json.loads(gzip.decompress(data).decode("utf-8"))
    X, yf, yb, ylc, gr = _train_rows_of(train)
    ens = fit_ensemble(X, yf, yb, ylc, gr)
    pred = predict(ens, X[:PRED_CHECK_N])
    for i, want in enumerate(model["pred_check"]):
        got = (float(pred["p_fail"][i]), float(pred["blc8"][i]),
               float(pred["log_cells"][i]))
        if any(abs(a - b) > 1e-9 for a, b in zip(got, want)):
            raise OptError("refit drift: row %d predicts %r, the model says %r"
                           % (i, got, want))
    msha = model_sha(model)
    v = {"path": path, "model": model, "ensemble": ens, "sha": msha,
         "hook": make_hook(lambda gid: (ens, None), model_sha256=msha,
                           importances=model.get("importances") or (),
                           enabled=model["enabled"], gate=model["gate"])}
    _MODEL["v"], _MODEL["key"] = v, key
    return v


def propose(ctx, history):
    """The hook campaign.py's HOOKS names; the model comes from opt_model.json."""
    return load_model()["hook"](ctx, history)


# --- (C8) the replay seam and one refinement round ----------------------------


def measured_from(bundle):
    """{(gid, config sha): {outcome, content_sha256}} over the bundle's rows."""
    return {(r["geometry_id"], r["config_sha"]):
            {"outcome": r["outcome"], "content_sha256": r["content_sha256"]}
            for r in bundle["attempts"]}


def probes_from(bundle):
    """{config sha: n_leaves} over every end record's recorded vetoes."""
    out = {}
    for g in bundle["geometries"]:
        for v in g.get("vetoes") or []:
            out[v["config_sha256"]] = v["n_leaves"]
    return out


def snaps_from(bundle):
    """{config sha: h_wall_min_m} over every end record's recorded vetoes."""
    out = {}
    for g in bundle["geometries"]:
        for v in g.get("vetoes") or []:
            out[v["config_sha256"]] = v["h_wall_min_m"]
    return out


def replay_fns(measured, probes, snaps, attempt_fn=None, probe_fn=None,
               snap_fn=None):
    """(attempt, probe, snap) that reuse what an earlier campaign measured."""
    def attempt(c, gctx, config, a, n_leaves):
        hit = measured.get((gctx["gid"], schema.canonical_sha256(config)))
        if hit is not None:
            oc = copy.deepcopy(hit["outcome"])
            for p in oc.get("patches") or []:
                p["capability_limited"] = False
            now = schema._now_iso()
            return {"outcome": oc,
                    "content_sha256": hit["content_sha256"],
                    "t_start": now, "t_end": now}
        return (attempt_fn or campaign.run_attempt)(c, gctx, config, a, n_leaves)

    def probe(c, gctx, cfg, a):
        key = schema.canonical_sha256(cfg)
        if key in probes:
            return {"n_leaves": probes[key], "max_non_orth_deg": None, "exit_code": 0}
        return (probe_fn or campaign.octree_probe)(c, gctx, cfg, a)

    def snap(c, gctx, config, a):
        key = schema.canonical_sha256(config)
        if key in snaps:          # the recorded value, None included (C8)
            return snaps[key]
        return (snap_fn or campaign.snap_probe)(c, gctx, config, a)
    return attempt, probe, snap


def refinement_set(bundle, k):
    """EXHAUSTED ends with attempts left under K: where the hook can act."""
    return sorted(g["geometry_id"] for g in bundle["geometries"]
                  if campaign.terminal_of(g) == "EXHAUSTED" and g["attempts"] < k)


def _rows_by_gid(cdir):
    out = collections.OrderedDict()
    for r in campaign.load_rows(cdir):
        out.setdefault(r["geometry_id"], {})[r["attempt"]] = r
    return out


def verify_round(j, out, rules_rows, ids):
    """The round's rows before the first optimiser row equal the rules campaign."""
    with open(os.path.join(out, campaign.FILES["end"]), encoding="utf-8") as f:
        end = json.load(f)
    if end["harness_errors"] != 0:
        raise OptError("round %d has %d harness errors" % (j, end["harness_errors"]))
    geoms = {g["geometry_id"]: g for g in campaign.load_geometries(out)}
    if sorted(geoms) != sorted(ids):
        raise OptError("round %d holds %s, expected %s"
                       % (j, sorted(geoms), sorted(ids)))
    rr_all = collections.OrderedDict()
    for r in rules_rows:
        rr_all.setdefault(r["geometry_id"], {})[r["attempt"]] = r
    gr_all = _rows_by_gid(out)
    for gid in ids:
        rr, gr = rr_all[gid], gr_all[gid]
        for a in range(1, len(rr) + 1):
            g = gr.get(a)
            if g is None:
                raise OptError("round %d diverges from the rules campaign at %s "
                               "attempt %d (the row is missing)" % (j, gid, a))
            if (g["decided_by"] == "optimiser"
                    or (g["config_sha"], g["decided_by"], g["rule_id"])
                    != (rr[a]["config_sha"], rr[a]["decided_by"],
                        rr[a]["rule_id"])):
                raise OptError("round %d diverges from the rules campaign at %s "
                               "attempt %d" % (j, gid, a))
        if len(gr) == len(rr):
            if campaign.terminal_of(geoms[gid]) != "EXHAUSTED":
                raise OptError("round %d ends %s at %s after %d attempts, "
                               "expected EXHAUSTED"
                               % (j, geoms[gid]["terminal"], gid, len(gr)))
        elif gr[len(rr) + 1]["decided_by"] != "optimiser":
            raise OptError("round %d diverges from the rules campaign at %s "
                           "attempt %d (decided_by %r)"
                           % (j, gid, len(rr) + 1, gr[len(rr) + 1]["decided_by"]))
    rep = campaign.replay(out)
    if not rep["ok"]:
        raise OptError("round %d does not replay: %s" % (j, rep["mismatches"][0]))


# --- (C9) the systems and the verdict ------------------------------------------


def system(ends_by_gid):
    """MFR, strict rate and mean BLC_8 over a map of end records."""
    fams = {}
    n = fails = strict = 0
    blc = []
    for gid in sorted(ends_by_gid):
        g = ends_by_gid[gid]
        f = fams.setdefault(g["family"], {"n": 0, "failures": 0, "blc": []})
        v = float(g.get("blc8_a_priori") or 0.0)
        n += 1
        f["n"] += 1
        blc.append(v)
        f["blc"].append(v)
        if g.get("failure") is True:
            fails += 1
            f["failures"] += 1
        if g.get("strict_failure") is True:
            strict += 1

    def fin(d):
        return {"n": d["n"], "failures": d["failures"],
                "mfr": d["failures"] / d["n"] if d["n"] else 0.0,
                "blc8_mean": float(numpy.mean(d["blc"])) if d["blc"] else 0.0}

    return {"n": n, "failures": fails, "mfr": fails / n if n else 0.0,
            "blc8_mean": float(numpy.mean(blc)) if blc else 0.0,
            "strict_rate": strict / n if n else 0.0,
            "per_family": {k: fin(v) for k, v in sorted(fams.items())}}


def _rescued(rules_ends, round_ends, ids):
    return sorted(i for i in ids if rules_ends[i].get("failure") is True
                  and round_ends[i].get("failure") is not True)


def _lost(rules_ends, round_ends, ids):
    return sorted(i for i in ids if rules_ends[i].get("failure") is not True
                  and round_ends[i].get("failure") is True)


DEPARTURES = (
    "docs/15 §G says 5 rounds x 3 trials over the tuning split; the optimiser is "
    "only ever called in its deployed place - after the remedies end EXHAUSTED "
    "with attempts left under the locked K = 4 - so a refinement round is ONE "
    "rules+opt campaign over the geometries where that can happen, each getting "
    "what K leaves (one or two proposals per round), not three per round over "
    "420. Every other tuning geometry's rules+opt end equals its rules end by "
    "construction (the mesher is deterministic, docs/15 §K G-DET).",
    "Every round is cross-fitted: the proposal for a geometry comes from the "
    "fold ensemble that never saw that geometry's rows (5 folds by "
    "crc32(gid) % 5), so each round's MFR is an out-of-geometry tuning estimate. "
    "The shipped model is fitted on all tuning rows.",
    "The box is relative to the L1 config, which the hook rebuilds with "
    "rules.setup: every band level shifts by the wall offset (clipped to "
    "[0, 6]); every band distance is scaled log-uniformly in [0.5, 2] "
    "(0.5 * 4 ** u, rounded to 6 decimals); a feature offset f > 0 sets "
    "feature_level = min(6, wall + f) on every levels entry and f = 0 sets an "
    "existing feature_level to 0; growth is drawn in [1.1, the L1 growth] "
    "(R-WIN already chose the largest growth that fits). A body on the R-PLANE "
    "path is left to the plane (OPT-PLANE).",
    "L0 on the pool is the config-level preflight without an octree probe "
    "(PF-BUDGET abstains; the predicted cells stand in for the budget) and "
    "without the snap probe (a PF-THIN refusal drops the point); the campaign's "
    "own veto re-checks the pick with the probe. The hook needs the campaign "
    "directory for the surface facts, so campaign.py passes cwd in the "
    "optimiser's ctx.",
    "The rounds reuse what the rules campaign (and earlier rounds) already "
    "meshed: an attempt whose (geometry, config sha) was measured returns that "
    "outcome, and the octree/snap probe values come from the recorded vetoes. "
    "The rows before the first optimiser row must then reproduce the rules "
    "campaign exactly, and the round checks that they do.",
    "The training rows are every tuning attempt of the five committed bundles "
    "(rules, the prior's two rounds, both baselines) plus each round's new "
    "rows, their configs rebuilt from the rows' deltas and checked by sha, "
    "deduplicated by (geometry, config sha); the bootstrap draws geometries, "
    "not rows.",
    "'Beats the rules on the tuning split' is G-OPT's own test-split ablation "
    "bar applied to the last round: MFR lower by at least 3 pp or mean BLC_8 "
    "higher by at least 0.05, with no family worse (more failures or a lower "
    "mean BLC_8). The optimiser ships enabled only when that holds AND the CV "
    "half passes.",
    "HistGradientBoosting has no built-in feature importances: the report and "
    "the records carry permutation importances (the fail-AUC drop when one "
    "feature column is permuted, seeded) of the shipped model.")


# --- (C10) refine: the rounds, the gate and the artefacts ---------------------


def refine(work, *, sources=None, rounds=ROUNDS, streams=6,
           report_dir=REPORT_DIR, binary=None, attempt_fn=None, probe_fn=None,
           snap_fn=None, write=True, quiet=False):
    """The whole G-OPT run over the tuning split; the report dict out."""
    named = load_sources(sources)
    for name, bundle, _sha in named:
        check_bundle(bundle, name)
    rules_name, rules_bundle, _sha = named[0]
    h0 = rules_bundle["campaign"]
    if h0.get("system") != "rules" or h0.get("ablate") != []:
        raise OptError("the refinement learns from a rules campaign with nothing "
                       "ablated first: %s is a %s campaign (ablate %s)"
                       % (rules_name, h0.get("system"), h0.get("ablate")))
    for g in rules_bundle["geometries"]:
        if campaign.terminal_of(g) == "HARNESS-ERROR":
            raise OptError("%s: geometry %s ended HARNESS-ERROR; the refinement "
                           "needs a clean rules campaign"
                           % (rules_name, g["geometry_id"]))
    gates = schema.load_gates()
    knobs = schema.load_knobs()
    mrows = {r["geometry_id"]: r
             for r in campaign.load_manifest("tuning", "rules")}
    rows, dup = training_rows([(n, b) for n, b, _s in named], mrows,
                           gates, knobs)
    R = refinement_set(rules_bundle, gates["attempts_k"])
    measured = measured_from(rules_bundle)
    probes = probes_from(rules_bundle)
    snaps = snaps_from(rules_bundle)
    A, P, S = replay_fns(measured, probes, snaps, attempt_fn, probe_fn, snap_fn)
    ends_rules = {g["geometry_id"]: g for g in rules_bundle["geometries"]}
    rules_sys = system(ends_rules)
    fam_of = {g["geometry_id"]: g["family"] for g in rules_bundle["geometries"]}
    ref_fams = collections.Counter(fam_of[g] for g in R)
    cv_by_round = []
    rounds_rep = []
    round_bundles = {}
    for j in range(1, rounds + 1):
        if not R:
            break
        X, yf, yb, ylc, gr = matrix(rows)
        folds, cv_j = cross_fit(X, yf, yb, ylc, gr)
        cv_by_round.append({"round": j, **cv_j})
        out = os.path.join(work, "round_%d" % j)
        endp = os.path.join(out, campaign.FILES["end"])
        reused_dir = os.path.isfile(endp)
        if reused_dir:
            with open(os.path.join(out, campaign.FILES["campaign"]),
                      encoding="utf-8") as f:
                hold = json.load(f)
            if sorted(hold["geometry_ids"]) != sorted(R):
                raise OptError("round %d holds %s, the refinement set is %s"
                               % (j, sorted(hold["geometry_ids"]), sorted(R)))
        elif os.path.isdir(out) and os.listdir(out):
            raise OptError("round %d is incomplete: remove %s and run it again"
                           % (j, out))
        else:
            campaign.run_campaign(
                {"manifest": "tuning", "mode": "rules+opt", "ids": R, "out": out,
                 "run_id": "opt-refine-r%d" % j, "streams": streams,
                 "audit_mod": AUDIT_OFF,
                 "binary": binary or campaign.BINARY_DEFAULT, "quiet": quiet},
                attempt_fn=A, probe_fn=P, snap_fn=S,
                hooks={"optimiser": make_hook(
                    lambda gid: (folds[fold_of(gid)], fold_of(gid)),
                    model_sha256="cross-fit round %d" % j)})
        verify_round(j, out, rules_bundle["attempts"], R)
        bundle_j = baseline.bundle_dir(out)
        round_bundles[j] = bundle_j
        pre = set(measured)
        new = rows_from(bundle_j, ROUND_BUNDLE % j, mrows, gates, knobs)
        known = {(r["geometry_id"], r["config_sha256"]) for r in rows}
        added = [r for r in new
                 if (r["geometry_id"], r["config_sha256"]) not in known]
        meshed = sum(1 for r in new
                     if (r["geometry_id"], r["config_sha256"]) not in pre)
        rows.extend(added)
        measured.update(measured_from(bundle_j))
        probes.update(probes_from(bundle_j))
        snaps.update(snaps_from(bundle_j))
        decisions = collections.Counter(
            ln["record"]["rule_id"] for ln in bundle_j["records"]
            if ln["record"].get("layer") == "optimiser")
        ends_j = {g["geometry_id"]: g for g in bundle_j["geometries"]}
        round_ends = dict(ends_rules)
        round_ends.update(ends_j)
        sys_j = system(round_ends)
        res_j = _rescued(ends_rules, ends_j, R)
        lost_j = _lost(ends_rules, ends_j, R)
        rounds_rep.append(
            {"round": j, "n": len(R), "decisions": dict(sorted(decisions.items())),
             "meshed": meshed, "reused": len(new) - meshed, "rescued": res_j,
             "lost": lost_j, "new_rows": len(added),
             # whether this call ran or reused the round is printed, not
             # reported, so a re-run that reuses the rounds gives an equal
             # report apart from "date"
             "wall_seconds": bundle_j["end"]["wall_seconds"],
             "_sys": sys_j,
             "picks": [next(i["value"] for i in ln["record"]["inputs"]
                            if i["name"] == "knobs")
                       for ln in bundle_j["records"]
                       if ln["record"].get("layer") == "optimiser"
                       and ln["record"]["rule_id"] == "OPT-PICK"]})
        if not quiet:
            print("[opt] round %d: %s geometries, OPT-PICK %s, OPT-NOFEAS %s, "
                  "OPT-PLANE %s, meshed %s, reused %s, rescued %s, lost %s, "
                  "MFR %s, BLC_8 %s (%s)"
                  % (j, explain.fmt(len(R)),
                     explain.fmt(decisions.get("OPT-PICK", 0)),
                     explain.fmt(decisions.get("OPT-NOFEAS", 0)),
                     explain.fmt(decisions.get("OPT-PLANE", 0)),
                     explain.fmt(meshed), explain.fmt(len(new) - meshed),
                     explain.fmt(len(res_j)), explain.fmt(len(lost_j)),
                     explain.fmt(sys_j["mfr"]), explain.fmt(sys_j["blc8_mean"]),
                     "reused" if reused_dir else "ran"))
    X, yf, yb, ylc, gr = matrix(rows)
    folds, cv = cross_fit(X, yf, yb, ylc, gr)
    full = fit_ensemble(X, yf, yb, ylc, gr)
    pf_all = predict(full, X)
    pred_check = [[float(pf_all["p_fail"][i]), float(pf_all["blc8"][i]),
                   float(pf_all["log_cells"][i])] for i in range(PRED_CHECK_N)]
    if cv["auc"] is not None:
        base_auc = float(roc_auc_score(yf, pf_all["p_fail"]))
        imp = []
        for jf in range(X.shape[1]):
            X2 = X.copy()
            X2[:, jf] = X[numpy.random.default_rng([IMP_SEED, jf]).permutation(
                len(X)), jf]
            auc2 = float(roc_auc_score(yf, predict(full, X2)["p_fail"]))
            imp.append({"feature": FEATURES[jf],
                        "auc_drop": round(base_auc - auc2, 6)})
        imp.sort(key=lambda d: (-d["auc_drop"], d["feature"]))
        importances = imp[:5]
    else:
        importances = []
    systems = {"rules": rules_sys}
    for r in rounds_rep:
        systems["round-%d" % r["round"]] = r.pop("_sys")
    final_name = "round-%d" % rounds_rep[-1]["round"] if rounds_rep else "rules"
    final_sys = systems[final_name]
    mfr_gain = rules_sys["mfr"] - final_sys["mfr"]
    blc_gain = final_sys["blc8_mean"] - rules_sys["blc8_mean"]
    regressed = sorted(
        f for f in set(rules_sys["per_family"]) | set(final_sys["per_family"])
        if final_sys["per_family"].get(f, {"failures": 0})["failures"]
        > rules_sys["per_family"].get(f, {"failures": 0})["failures"]
        or rules_sys["per_family"].get(f, {"blc8_mean": 0.0})["blc8_mean"]
        - final_sys["per_family"].get(f, {"blc8_mean": 0.0})["blc8_mean"] > 1e-12)
    beats_rules = (mfr_gain >= MFR_GAIN_MIN or blc_gain >= BLC_GAIN_MIN) \
        and not regressed
    conditions = {"auc_ge": cv["auc"] is not None and cv["auc"] >= AUC_MIN,
                  "rmse_le": cv["blc8_rmse"] <= RMSE_MAX,
                  "beats_rules": beats_rules}
    verdict = "PASS" if conditions["auc_ge"] and conditions["rmse_le"] else "FAIL"
    enabled = verdict == "PASS" and beats_rules
    train_rows = [{"geometry_id": r["geometry_id"], "family": r["family"],
                   "source": r["source"], "attempt": r["attempt"],
                   "config_sha256": r["config_sha256"],
                   "x": [None if isinstance(v, float) and math.isnan(v) else v
                         for v in r["x"]],
                   "fail": bool(r["fail"]), "blc8": float(r["blc8"]),
                   "log_cells": r["log_cells"]} for r in rows]
    train_obj = {"$comment": HEADER, "schema": TRAIN_SCHEMA,
                 "features": list(FEATURES), "sources": [n for n, _b, _s in named],
                 "rows": train_rows}
    train_bytes = gzip.compress(baseline._canonical(train_obj), compresslevel=9,
                                mtime=0)
    model_obj = {"$comment": HEADER, "schema": MODEL_SCHEMA,
                 "features": list(FEATURES), "hyper": dict(HYPER),
                 "members": MEMBERS, "boot_seed": BOOT_SEED, "folds": FOLDS,
                 "p_fail_max": P_FAIL_MAX,
                 "box": {"wall_offsets": list(WALL_OFFSETS),
                         "band_scale": list(BAND_SCALE),
                         "feature_offsets": list(FEATURE_OFFSETS),
                         "feature_tolerances": list(FEATURE_TOLS),
                         "smoothing_passes": list(SMOOTHING),
                         "growth_lo": GROWTH_LO},
                 "pool_n": 2 ** POOL_M, "sobol_seed": SOBOL_SEED,
                 "train": {"file": TRAIN_NAME,
                           "sha256": hashlib.sha256(train_bytes).hexdigest(),
                           "content_sha256":
                               hashlib.sha256(
                                   baseline._canonical(train_obj)).hexdigest(),
                           "n_rows": len(rows), "n_fail": int(yf.sum()),
                           "n_geometries": len(set(gr))},
                 "pred_check": pred_check, "importances": importances, "cv": cv}
    msha = model_sha(model_obj)
    model_obj["enabled"] = enabled
    model_obj["gate"] = {"verdict": verdict, "auc": cv["auc"],
                         "blc8_rmse": cv["blc8_rmse"], "beats_rules": beats_rules,
                         "model_sha256": msha, "report": REPORT_NAME}
    picks = [dict(k) for r in rounds_rep for k in r["picks"]]

    def hist(key):
        c = collections.Counter(str(k[key]) for k in picks)
        return dict(sorted(c.items()))

    rounds_out = []
    for r in rounds_rep:
        b = round_bundles[r["round"]]
        csha = hashlib.sha256(baseline._canonical(b)).hexdigest()
        rounds_out.append({k: v for k, v in r.items() if k != "picks"})
    rep = {"$comment": HEADER, "schema": GATE_SCHEMA,
           "date": time.strftime("%Y-%m-%d"), "verdict": verdict,
           "enabled": enabled, "conditions": conditions,
           "thresholds": {"auc_min": AUC_MIN, "rmse_max": RMSE_MAX,
                          "mfr_gain_min": MFR_GAIN_MIN,
                          "blc_gain_min": BLC_GAIN_MIN, "p_fail_max": P_FAIL_MAX},
           "cv": cv, "cv_by_round": cv_by_round,
           "n_tuning": len(rules_bundle["campaign"]["geometry_ids"]),
           "refinement_set": {"n": len(R),
                              "per_family": {k: v for k, v in sorted(ref_fams.items())},
                              "geometry_ids": list(R)},
           "rounds": rounds_out, "systems": systems,
           "final": {"system": final_name, "mfr_gain": mfr_gain,
                     "blc_gain": blc_gain, "regressed_families": regressed},
           "picks": {"n": len(picks), "feature_tolerance": hist("feature_tolerance"),
                     "smoothing_passes": hist("smoothing_passes"),
                     "wall_offset": hist("wall_offset")},
           "model_sha256": msha, "train": dict(model_obj["train"]),
           "importances": importances,
           "sources": [{"file": n, "sha256": s,
                        "content_sha256": hashlib.sha256(
                            baseline._canonical(b)).hexdigest()}
                       for n, b, s in named],
           "n_duplicates": dup, "departures": list(DEPARTURES)}
    if write:
        os.makedirs(report_dir, exist_ok=True)
        for i, r in enumerate(rounds_out):
            rounds_out[i]["bundle"] = baseline._write_or_match(
                report_dir, ROUND_BUNDLE % r["round"], round_bundles[r["round"]])
        train_path = os.path.join(report_dir, TRAIN_NAME)
        if os.path.exists(train_path):
            with open(train_path, "rb") as f:
                if f.read() != train_bytes:
                    raise OptError("exists: %s holds a different train file"
                                   % train_path)
        else:
            with open(train_path, "wb") as f:
                f.write(train_bytes)
        _dump_json(os.path.join(report_dir, MODEL_NAME), model_obj)
        _dump_json(os.path.join(report_dir, REPORT_NAME), rep)
        _write_text(os.path.join(report_dir, REPORT_MD), report_md(rep))
    else:
        for i, r in enumerate(rounds_out):
            rounds_out[i]["bundle"] = {
                "file": ROUND_BUNDLE % r["round"], "sha256": None,
                "content_sha256": hashlib.sha256(
                    baseline._canonical(round_bundles[r["round"]])).hexdigest(),
                "bytes": None}
    if not quiet:
        print("G-OPT %s: CV AUC %s (>= %s %s), BLC_8 RMSE %s (<= %s %s); tuning "
              "MFR rules %s -> rules+opt %s, BLC_8 %s -> %s, no family worse %s; "
              "the optimiser ships %s"
              % (verdict, explain.fmt(cv["auc"]), explain.fmt(AUC_MIN),
                 conditions["auc_ge"], explain.fmt(cv["blc8_rmse"]),
                 explain.fmt(RMSE_MAX), conditions["rmse_le"],
                 explain.fmt(rules_sys["mfr"]), explain.fmt(final_sys["mfr"]),
                 explain.fmt(rules_sys["blc8_mean"]),
                 explain.fmt(final_sys["blc8_mean"]), not regressed,
                 "enabled" if enabled else "DISABLED"))
    return rep


def _dump_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def _write_text(path, text):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _rate(v):
    return "n/a" if v is None else "%.3f" % v


def report_md(rep):
    """The G-OPT report as markdown; every number from the report object."""
    out = ["<!-- %s -->" % HEADER, "",
           "# G-OPT - the L4 optimiser on the tuning split (docs/15 §F)", "",
           "- date: %s" % rep["date"],
           "- verdict: %s" % rep["verdict"]]
    if rep["enabled"]:
        out.append("- The optimiser ships enabled.")
    else:
        off = [k for k, v in rep["conditions"].items() if not v]
        out.append("- The optimiser ships DISABLED: %s." % ", ".join(off))
    out.append("- model sha256: %s" % rep["model_sha256"])
    out.append("- train rows: %d" % rep["train"]["n_rows"])
    out += ["", "## Surrogate cross-validation", "",
            "| round | rows | fail AUC | BLC_8 RMSE | zero-predictor RMSE | "
            "positive-row RMSE | log10-cells RMSE |",
            "| --- | --- | --- | --- | --- | --- | --- |"]
    for cv in list(rep["cv_by_round"]) + [dict(rep["cv"], round="final")]:
        out.append("| %s | %d | %s | %s | %s | %s | %s |"
                   % (cv["round"], cv["n"], _rate(cv["auc"]),
                      _rate(cv["blc8_rmse"]), _rate(cv["blc8_rmse_zero"]),
                      _rate(cv["blc8_rmse_pos"]), _rate(cv["log_cells_rmse"])))
    out += ["", "## Per-round tuning curve", "",
            "| round | MFR | mean BLC_8 | strict rate | rescued | OPT-PICK | "
            "OPT-NOFEAS | meshed | reused |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    rl = rep["systems"]["rules"]
    out.append("| rules | %s | %s | %s | - | - | - | - | - |"
               % (_rate(rl["mfr"]), _rate(rl["blc8_mean"]),
                  _rate(rl["strict_rate"])))
    for r in rep["rounds"]:
        rs = rep["systems"]["round-%d" % r["round"]]
        out.append("| round %d | %s | %s | %s | %d | %d | %d | %d | %d |"
                   % (r["round"], _rate(rs["mfr"]), _rate(rs["blc8_mean"]),
                      _rate(rs["strict_rate"]), len(r["rescued"]),
                      r["decisions"].get("OPT-PICK", 0),
                      r["decisions"].get("OPT-NOFEAS", 0),
                      r["meshed"], r["reused"]))
    out += ["", "## Per family", "",
            "| family | n | rules failures | final failures | rules BLC_8 | "
            "final BLC_8 |",
            "| --- | --- | --- | --- | --- | --- |"]
    fs = rep["systems"][rep["final"]["system"]]
    for fam in sorted(rl["per_family"]):
        rf, ff = rl["per_family"][fam], fs["per_family"][fam]
        out.append("| %s | %d | %d | %d | %s | %s |"
                   % (fam, rf["n"], rf["failures"], ff["failures"],
                      _rate(rf["blc8_mean"]), _rate(ff["blc8_mean"])))
    pk = rep["picks"]
    out += ["", "## What the picks set", "",
            "The rounds made %d OPT-PICK records." % pk["n"],
            "- feature_tolerance: %s" % json.dumps(pk["feature_tolerance"],
                                                   sort_keys=True),
            "- smoothing_passes: %s" % json.dumps(pk["smoothing_passes"],
                                                  sort_keys=True),
            "- wall_offset: %s" % json.dumps(pk["wall_offset"], sort_keys=True),
            "A pick at feature_tolerance 0 does not capture feature edges; "
            "G-FID guards the snapped share of feature edges at the evaluation "
            "(docs/15 §F).", "", "## Importances", ""]
    for d in rep["importances"]:
        out.append("- %s: fail AUC drop %s" % (d["feature"], _rate(d["auc_drop"])))
    if not rep["importances"]:
        out.append("- none (the cross-validation had a single fail class)")
    out += ["", "## Departures", ""]
    out += ["- %s" % d for d in rep["departures"]]
    return "\n".join(out) + "\n"


# --- (C11) check: the committed files, rebuilt ---------------------------------


def _read_json_or_none(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def check(report_dir=REPORT_DIR):
    """The committed G-OPT artefacts rebuilt; never raises on a bad file."""
    items = []

    def add(name, ok, why):
        items.append({"name": name, "ok": bool(ok), "why": why})

    rp = os.path.join(report_dir, REPORT_NAME)
    mp = os.path.join(report_dir, MODEL_NAME)
    rep = _read_json_or_none(rp)
    add("G-OPT.json", rep is not None, rp)
    model = _read_json_or_none(mp)
    add("opt_model.json", model is not None, mp)
    named = []
    if rep is None:
        add("sources", False, rp)
    else:
        oks, why = True, rp
        for s in rep.get("sources") or []:
            p = s["file"] if os.path.isabs(s["file"]) \
                else os.path.join(HERE, s["file"])
            if not os.path.isfile(p):
                oks, why = False, p
                break
            h = _file_sha256(p)
            if h != s["sha256"]:
                oks, why = False, "%s holds %s" % (p, h[:12])
                break
            try:
                named.append((s["file"], baseline.read_bundle(p)))
            except (baseline.BaselineError, OSError, ValueError) as e:
                oks, why = False, "%s: %s" % (p, e)
                break
        add("sources", oks, why)
    bundles_ok, bundles_why, round_named = True, os.path.join(report_dir,
                                                              "refine_r*.json.gz"), []
    if rep is not None:
        bundles_why = []
        for r in rep.get("rounds") or []:
            p = os.path.join(report_dir, r["bundle"]["file"])
            if not os.path.isfile(p):
                bundles_ok, bundles_why = False, p
                break
            if _file_sha256(p) != r["bundle"]["sha256"]:
                bundles_ok, bundles_why = False, "%s holds %s" % (
                    p, _file_sha256(p)[:12])
                break
            bundles_why.append(r["bundle"]["file"])
            try:
                round_named.append((r["bundle"]["file"], baseline.read_bundle(p)))
            except (baseline.BaselineError, OSError, ValueError) as e:
                bundles_ok, bundles_why = False, "%s: %s" % (p, e)
                break
        if bundles_ok and bundles_why:
            bundles_why = "%s: %s" % (report_dir, ", ".join(bundles_why))
        elif bundles_ok:
            bundles_why = "%s (no rounds)" % report_dir
    add("round bundles", bundles_ok if rep is not None else False,
        bundles_why if rep is not None else rp)
    train_obj = None
    if model is None:
        add("train file", False, mp)
    else:
        tp = os.path.join(report_dir, model["train"]["file"])
        if not os.path.isfile(tp):
            add("train file", False, tp)
        else:
            h = _file_sha256(tp)
            ok = h == model["train"]["sha256"] and (
                rep is None or rep.get("train", {}).get("sha256") == h)
            add("train file", ok, "%s holds %s" % (tp, h[:12])
                if not ok else tp)
            try:
                train_obj = json.loads(gzip.decompress(open(tp, "rb").read())
                                       .decode("utf-8"))
            except (OSError, ValueError) as e:
                train_obj = None
    rebuild_ok, rebuild_why = False, mp
    if rep is not None and model is not None and train_obj is not None:
        try:
            gates = schema.load_gates()
            knobs = schema.load_knobs()
            mrows = {r["geometry_id"]: r
                     for r in campaign.load_manifest("tuning", "rules")}
            rows, _dup = training_rows(named + round_named, mrows, gates, knobs)
            def shape(r):
                return {"geometry_id": r["geometry_id"], "family": r["family"],
                        "source": r["source"], "attempt": r["attempt"],
                        "config_sha256": r["config_sha256"],
                        "x": [None if isinstance(v, float) and math.isnan(v)
                              else v for v in r["x"]],
                        "fail": bool(r["fail"]), "blc8": float(r["blc8"]),
                        "log_cells": r["log_cells"]}
            got = [schema.canonical_sha256(shape(r)) for r in rows]
            want = [schema.canonical_sha256(r) for r in train_obj["rows"]]
            rebuild_ok = got == want
            rebuild_why = "%d rows" % len(want) if rebuild_ok \
                else "rows %d vs %d" % (len(got), len(want))
        except (OptError, campaign.CampaignError, baseline.BaselineError,
                split.SplitSealed, split.SplitError, KeyError, ValueError) as e:
            rebuild_why = "%s: %s" % (type(e).__name__, e)
    add("train rebuild", rebuild_ok, rebuild_why)
    if model is None or rep is None:
        add("model sha", False, mp)
        add("refit", False, mp)
        add("enabled", False, mp)
    else:
        try:
            msha = model_sha(model)
        except (KeyError, TypeError, ValueError):
            msha = None
        add("model sha", msha == rep.get("model_sha256"),
            str(msha)[:16] if msha else mp)
        try:
            v = load_model(mp)
            add("refit", True, "%s (enabled %s)" % (mp, v["model"]["enabled"]))
        except (OptError, OSError, ValueError, KeyError) as e:
            add("refit", False, "%s: %s" % (type(e).__name__, e))
        add("enabled",
            model.get("enabled") == rep.get("enabled")
            and isinstance(model.get("gate"), dict)
            and model["gate"].get("verdict") == rep.get("verdict"),
            "%s vs %s" % (mp, rp))
    return {"schema": CHECK_SCHEMA, "items": items,
            "verdict": "PASS" if all(i["ok"] for i in items) else "FAIL"}


# --- (C13) the CLI -------------------------------------------------------------


class _ArgParser(argparse.ArgumentParser):
    """Bad arguments exit 2 with `optimise: <message>` (prior.py's pattern)."""

    def error(self, message):
        self.exit(2, "optimise: %s\n" % message)


def _cli_plan():
    named = load_sources()
    gates = schema.load_gates()
    knobs = schema.load_knobs()
    mrows = {r["geometry_id"]: r for r in campaign.load_manifest("tuning", "rules")}
    rows, dup = training_rows([(n, b) for n, b, _s in named], mrows,
                           gates, knobs)
    X, yf, yb, ylc, gr = matrix(rows)
    _folds, cv = cross_fit(X, yf, yb, ylc, gr)
    R = refinement_set(named[0][1], gates["attempts_k"])
    fams = collections.Counter(g["family"] for g in named[0][1]["geometries"]
                               if g["geometry_id"] in set(R))
    print("[opt] data: %d rows, %d failures, %d with BLC_8 > 0, %d geometries, "
          "%d features, %d duplicates"
          % (len(rows), int(yf.sum()), int((yb > 0).sum()), len(set(gr)),
             X.shape[1], dup))
    print("[opt] cv: AUC %.6f, BLC_8 RMSE %.6f (zero predictor %.6f), log10 "
          "cells RMSE %.6f"
          % (cv["auc"], cv["blc8_rmse"], cv["blc8_rmse_zero"],
             cv["log_cells_rmse"]))
    print("[opt] refinement set: %d geometries (%s)"
          % (len(R), ", ".join("%s %d" % kv for kv in sorted(
              fams.items(), key=lambda kv: (-kv[1], kv[0])))))
    return 0


def _cli_check(report_dir):
    res = check(report_dir)
    for it in res["items"]:
        print("[check] %s %s %s" % (it["name"], "ok" if it["ok"] else "FAIL",
                                    it["why"]))
    print("CHECK %s" % res["verdict"])
    return 0 if res["verdict"] == "PASS" else 1


def main(argv=None):
    ap = _ArgParser(prog="optimise.py",
                    description="the L4 optimiser of docs/15 §C (G-OPT)")
    ap.add_argument("--selftest", action="store_true", help="the G-OPT selftest")
    ap.add_argument("--plan", action="store_true",
                    help="the training rows, the surrogate CV and the refinement "
                         "set; nothing run or written")
    ap.add_argument("--refine", action="store_true",
                    help="run the refinement rounds: optimise/G-OPT.json, .md, "
                         "opt_model.json, train.json.gz")
    ap.add_argument("--check", action="store_true",
                    help="the committed report, bundles, train rebuild and refit")
    ap.add_argument("--work", help="the refinement rounds' work directory")
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--streams", type=int, default=6)
    ap.add_argument("--binary", default=None)
    ap.add_argument("--report-dir", dest="report_dir", default=REPORT_DIR)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    try:
        if a.check:
            return _cli_check(a.report_dir)
        if a.plan:
            return _cli_plan()
        if a.refine:
            if not a.work:
                ap.exit(2, "optimise: --refine needs --work\n")
            if not 1 <= a.rounds <= 5:
                raise OptError("--rounds must lie in 1..5, got %d" % a.rounds)
            if not 1 <= a.streams <= 6:
                raise OptError("--streams must lie in 1..6, got %d" % a.streams)
            refine(a.work, rounds=a.rounds, streams=a.streams,
                   report_dir=a.report_dir, binary=a.binary)
            return 0
    except (OptError, campaign.CampaignError, baseline.BaselineError,
            split.SplitSealed, split.SplitError) as e:
        sys.stderr.write("optimise: %s\n" % e)
        return 2
    ap.error("one of --selftest, --plan, --refine, --check is required")


# --- (C14) the selftest ---------------------------------------------------------

IDS16 = prior.IDS8 + ("D-1-001", "D-1-027", "E-1-003", "E-1-011", "F-1-025",
                      "G-1-052", "E-1-014", "D-1-073")


def _oracle_run(out, ids, mode, attempt_fn):
    return campaign.run_campaign(
        {"manifest": "tuning", "ids": list(ids), "out": out, "mode": mode,
         "streams": 2, "quiet": True}, attempt_fn=attempt_fn,
        probe_fn=campaign._fake_probe(1000), snap_fn=campaign._fake_snap)


def _harness(H):
    tmp = H["tmp"]
    H["gates"] = schema.load_gates()
    H["knobs"] = schema.load_knobs()
    H["mrows"] = {r["geometry_id"]: r
                  for r in campaign.load_manifest("tuning", "rules")}
    H["o_rules"] = os.path.join(tmp, "o_rules")
    H["o_tmpl"] = os.path.join(tmp, "o_tmpl")
    H["o_lhs"] = os.path.join(tmp, "o_lhs")
    for out, mode in ((H["o_rules"], "rules"), (H["o_tmpl"], "b0-template"),
                      (H["o_lhs"], "b0-lhs")):
        _oracle_run(out, IDS16, mode, prior._wall_oracle)
    src = os.path.join(tmp, "src")
    os.makedirs(src)
    H["osrc"] = [os.path.join(src, n)
                 for n in ("rules.json.gz", "tmpl.json.gz", "lhs.json.gz")]
    for d, p in ((H["o_rules"], H["osrc"][0]), (H["o_tmpl"], H["osrc"][1]),
                 (H["o_lhs"], H["osrc"][2])):
        baseline.write_bundle(baseline.bundle_dir(d), p)
    H["committed"] = load_sources()
    H["rb"] = H["committed"][0][1]
    H["cends"] = {g["geometry_id"]: g for g in H["rb"]["geometries"]}


def selftest():
    """(C14): eleven [ok] groups, then SELFTEST PASS; the mesher runs in group 10."""
    t0 = time.perf_counter()
    tmp = tempfile.mkdtemp()
    H = {"tmp": tmp}
    groups = ((_g1_constants, "constants"), (_g2_rows, "rows"),
              (_g3_pool, "pool"), (_g4_crossfit, "cross-fit"),
              (_g5_decide, "decide"), (_g6_rank, "rank"),
              (_g7_hook, "hook seam"), (_g8_refine, "refine"),
              (_g9_refine_a, "refine A"), (_g10_live, "live"),
              (_g11_check_cli, "check and CLI"))
    try:
        try:
            _harness(H)
        except Exception as e:
            print("SELFTEST FAIL: harness: %s: %s" % (type(e).__name__, e))
            return 1
        for fn, label in groups:
            try:
                fn(H)
            except Exception as e:
                print("SELFTEST FAIL: %s: %s: %s" % (label, type(e).__name__, e))
                return 1
        print("SELFTEST PASS (%.1f s)" % (time.perf_counter() - t0))
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _g1_constants(H):
    tmp = H["tmp"]
    assert list(FEATURES[:17]) == list(prior.FEATURES) and len(FEATURES) == 31
    assert MEMBERS == 5 and POOL_M == 8 and 2 ** POOL_M == 256
    assert FEATURE_TOLS == (0.0, 0.25, 0.5) \
        and FEATURE_TOLS_SHARP == (0.25, 0.5)
    for rid in OPT_IDS:
        assert rid in explain.TEMPLATES \
            and explain.TEMPLATES[rid]["layer"] == "optimiser", rid
    d = os.path.join(tmp, "g1")
    os.makedirs(d)
    b = baseline.read_bundle(H["osrc"][0])
    b["campaign"]["split_mode"] = split.EVALUATE
    p = os.path.join(d, "evaluate.json.gz")
    baseline.write_bundle(b, p)
    try:
        refine(os.path.join(tmp, "w1"), sources=[p], write=False, quiet=True)
    except split.SplitSealed as e:
        assert "tuning split only" in str(e), str(e)
    else:
        raise OptError("group 1: the evaluate bundle was not refused")
    try:
        refine(os.path.join(tmp, "w1"), sources=[H["osrc"][1], H["osrc"][0]],
               write=False, quiet=True)
    except OptError as e:
        assert "rules campaign" in str(e), str(e)
    else:
        raise OptError("group 1: the b0-template first source was not refused")
    try:
        load_model(os.path.join(tmp, "absent.json"))
    except OptError as e:
        assert "no optimiser model" in str(e), str(e)
    else:
        raise OptError("group 1: the missing model was not refused")
    print("[ok] constants: 31 features, 5 members, 256 Sobol points, 4 OPT ids "
          "templated; an evaluate bundle is refused before any row, a b0-template "
          "first source and a missing model are refused by name")


def _g2_rows(H):
    rows, dup = training_rows([(n, b) for n, b, _s in H["committed"]],
                           H["mrows"], H["gates"], H["knobs"])
    per = collections.Counter(r["source"] for r in rows)
    assert [per[s] for s in SOURCES] == [805, 116, 24, 372, 1488], per
    assert (len(rows), dup) == (2805, 0), (len(rows), dup)
    assert sum(1 for r in rows if r["fail"]) == 2130
    assert sum(1 for r in rows if r["blc8"] > 0) == 107
    assert sum(1 for r in rows if r["log_cells"] is None) == 192
    assert len({r["geometry_id"] for r in rows}) == 372
    r0 = rows[0]
    assert (r0["geometry_id"], r0["attempt"], r0["source"]) == \
        ("A-1-000", 1, "prior/tuning_rules.json.gz"), r0
    want = [0.766093, 0.52137, 0.04244, -0.369564, 0.254152, 0.628693,
            -1.760774, -0.493414, 0.109514, 0.0, -1.372082, 0.0, 1.0, 1.0,
            0.02634, 0.0, 3.781468, 5.0, -1.016048, -2.521198, -2.016048,
            -1.140986, 1.0, 0.5, 3.0, 1.152, 1.442943, 8.0, 6.079753, 0.0, 6.0]
    assert all(abs(a - b) <= 1e-6 for a, b in zip(r0["x"], want)), r0["x"]
    orows, dup2 = training_rows([(n, b) for n, b, _s in
                                 load_sources(H["osrc"])], H["mrows"],
                                H["gates"], H["knobs"])
    assert (len(orows), dup2) == (104, 0), (len(orows), dup2)
    assert sum(1 for r in orows if r["fail"]) == 62, \
        sum(1 for r in orows if r["fail"])
    assert len({r["geometry_id"] for r in orows}) == 16, \
        len({r["geometry_id"] for r in orows})
    b = baseline.read_bundle(H["osrc"][0])
    victim = None
    for r in b["attempts"]:
        if r["decided_by"] == "remedy" and r["config_delta"]:
            victim = r
            break
    assert victim is not None
    to = victim["config_delta"][0]["to"]
    victim["config_delta"][0]["to"] = 0.5 if to != 0.5 else 0.25
    try:
        rows_from(b, "tampered", H["mrows"], H["gates"], H["knobs"])
    except OptError as e:
        assert "reconstruct" in str(e), str(e)
    else:
        raise OptError("group 2: the tampered delta was not refused")
    H["crows"] = rows
    print("[ok] rows: 2805 committed rows rebuilt by sha (2130 failures, 107 "
          "with BLC_8 > 0, 192 without cells, 372 geometries, 0 duplicates); "
          "the first row reproduces to 1e-6; the oracle gives %d rows; a "
          "tampered delta is refused" % (len(orows),))


def _g3_pool(H):
    gid = "D-1-073"
    fp = H["cends"][gid]["fingerprint"]
    # D-1-073's L1 as the committed rules campaign built it (before FT-RADIUS)
    s = rules.setup(H["mrows"][gid], fp, campaign.stl_rel(gid),
                    campaign.case_rel(gid), gid, gates=H["gates"],
                    knobs=H["knobs"], ft_radius=rules.ft_radius_of(H["rb"]["records"]))
    assert not s["refused"], s["refused"]
    l1 = s["config"]
    assert schema.canonical_sha256(l1).startswith("e31caabef618")
    assert remedies.wall_level(l1) == 3
    assert remedies._get(l1, "/layers/growth") == 1.34
    assert "snap" not in l1
    pc, kn = point_config(l1, [0.5] * 6, fp)
    assert kn == {"wall_offset": 0, "band_scale": 1.0, "feature_offset": 1,
                  "feature_tolerance": 0.5, "smoothing_passes": 2,
                  "growth": 1.22}, kn
    assert schema.canonical_sha256(pc).startswith("7ff2856a0591"), \
        schema.canonical_sha256(pc)[:12]
    feats = config_features(pc, fp)
    want = [3, -0.300984, -1.204074, -0.726951, 0.176091, 1, 0.5, 2, 1.22,
            1.742763, 8, 4.729018, 0.0, 4]
    assert all(abs(a - b) <= 1e-6 for a, b in zip(feats, want)), feats
    assert abs(feats[CONFIG_FEATURES.index("log10_wallband_over_lmax")]
               - (-0.726951)) <= 1e-6, CONFIG_FEATURES
    edits = rules.diff_edits(l1, pc)
    want_edits = [{"pointer": "/layers/growth", "from": 1.34, "to": 1.22},
                  {"pointer": "/refinement/levels/0/bands/0/distance",
                   "from": 0.1755375, "to": 0.175538},
                  {"pointer": "/snap/feature_tolerance", "from": None,
                   "to": 0.5},
                  {"pointer": "/snap/smoothing_passes", "from": None, "to": 2}]
    assert json.dumps(edits, sort_keys=True) == \
        json.dumps(want_edits, sort_keys=True), edits
    u = sobol(gid)
    assert u.shape == (256, 6), u.shape
    assert all(abs(a - b) <= 1e-6 for a, b in zip(
        u[0], [0.36039, 0.575032, 0.107426, 0.46038, 0.526552, 0.486888])), u[0]
    assert numpy.array_equal(u, sobol(gid))
    pc2 = copy.deepcopy(pc)
    del pc2["snap"]
    del pc2["layers"]
    f2 = config_features(pc2, fp)
    assert f2[CONFIG_FEATURES.index("feature_tolerance")] == 0.5
    assert f2[CONFIG_FEATURES.index("smoothing_passes")] == 3
    assert math.isnan(f2[CONFIG_FEATURES.index("growth")])
    assert math.isnan(f2[CONFIG_FEATURES.index("log10_hwall_over_t1")])
    assert f2[CONFIG_FEATURES.index("layers_n")] == 0
    print("[ok] pool: D-1-073's L1 (wall 3, growth 1.34), point_config at u = 0.5 "
          "and its features and edits, Sobol row 0; a config without snap and "
          "layers takes the defaults")


def _g4_crossfit(H):
    X, yf, yb, ylc, gr = matrix(H["crows"])
    folds, cv = cross_fit(X, yf, yb, ylc, gr)
    assert abs(cv["auc"] - 0.985904) <= 1e-6, cv["auc"]
    assert abs(cv["blc8_rmse"] - 0.121079) <= 1e-6, cv["blc8_rmse"]
    assert abs(cv["blc8_rmse_zero"] - 0.19531) <= 1e-5, cv["blc8_rmse_zero"]
    assert abs(cv["blc8_rmse_pos"] - 0.492997) <= 1e-5, cv["blc8_rmse_pos"]
    assert abs(cv["log_cells_rmse"] - 0.051761) <= 1e-5, cv["log_cells_rmse"]
    assert abs(cv["brier"] - 0.03989) <= 1e-5, cv["brier"]
    assert cv["folds"] == [0, 1, 2, 3, 4], cv["folds"]
    assert cv["fold_rows"] == {"0": 573, "1": 457, "2": 573, "3": 579, "4": 623}
    te = numpy.array([fold_of(g) == 4 for g in gr])
    tr = ~te
    e2 = fit_ensemble(X[tr], yf[tr], yb[tr], ylc[tr],
                      [g for g, t in zip(gr, te) if not t])
    p1 = predict(folds[4], X[:50])
    p2 = predict(e2, X[:50])
    assert numpy.array_equal(p1["p_fail"], p2["p_fail"])
    assert numpy.array_equal(p1["blc8"], p2["blc8"])
    assert numpy.array_equal(p1["log_cells"], p2["log_cells"])
    H["folds"] = folds
    print("[ok] cross-fit: AUC 0.985904, BLC_8 RMSE 0.121079 on 2805 rows in 5 "
          "folds (573 457 573 579 623); a refit is bitwise equal")


def _gid_ctx(H, gid):
    """The (C7)-shaped ctx, history and setup of one geometry, from the
    committed rules bundle."""
    end = H["cends"][gid]
    fp = end["fingerprint"]
    rows = {}
    for r in H["rb"]["attempts"]:
        if r["geometry_id"] == gid:
            rows[r["attempt"]] = r
    top = max(rows)
    cur = next(r for r in H["crows"]
               if r["geometry_id"] == gid and r["attempt"] == top)
    s = rules.setup(H["mrows"][gid], fp, campaign.stl_rel(gid),
                    campaign.case_rel(gid), gid, gates=H["gates"],
                    knobs=H["knobs"])
    assert not s["refused"], (gid, s["refused"])
    ctx = {"geometry_id": gid, "fingerprint": fp,
           "flow": H["mrows"][gid]["flow"], "win_level": s["summary"]["win_level"],
           "config": cur["config"], "outcome": rows[top]["outcome"],
           "cwd": H["o_rules"]}
    hist = [{"attempt": r["attempt"], "config_sha256": r["config_sha"],
             "rule_id": r["rule_id"] if r["decided_by"] == "remedy" else None}
            for r in (rows[a] for a in sorted(rows))]
    return ctx, hist, s


def _g5_decide(H):
    ctx, hist, _s = _gid_ctx(H, "D-1-073")
    res = decide(H["folds"][4], ctx, hist, H["gates"], H["knobs"],
                 {"model_sha256": "selftest", "fold": 4, "importances": []})
    assert res["verdict"] == "apply" and res["rule_id"] == "OPT-PICK", \
        (res["verdict"], res.get("rule_id"))
    c = res["counts"]
    assert (c["pool_n"], c["unique_n"], c["visited_n"], c["edit_refused_n"],
            c["l0_refused_n"], c["l0_pass_n"], c["feasible_n"]) == \
        (256, 256, 0, 0, 29, 227, 14), c
    pk = res["pick"]
    assert pk["sobol_index"] == 7, pk["sobol_index"]
    assert pk["config_sha256"].startswith("a5e6d2875713"), \
        pk["config_sha256"][:12]
    assert abs(pk["p_fail"] - 0.146157) <= 1e-6, pk["p_fail"]
    assert abs(pk["p_fail_std"] - 0.043875) <= 1e-6, pk["p_fail_std"]
    assert abs(pk["blc8_a_priori"] - 0.013567) <= 1e-6, pk["blc8_a_priori"]
    assert abs(pk["log_cells"] - 3.961202) <= 1e-6, pk["log_cells"]
    assert [r["sobol_index"] for r in res["runners_up"]] == [191, 216, 130], \
        [r["sobol_index"] for r in res["runners_up"]]
    assert pk["knobs"]["feature_tolerance"] in FEATURE_TOLS_SHARP
    assert pk["knobs"]["smoothing_passes"] == 2
    assert schema.errors(res["record"], "DecisionRecord") == []
    assert not [e["pointer"] for e in res["edits"]
                if e["pointer"].startswith("/layers/")
                and e["pointer"] != "/layers/growth"]
    H["d_ctx"], H["d_res"] = ctx, res
    ctx_a, hist_a, _s2 = _gid_ctx(H, "A-1-000")
    res_a = decide(H["folds"][0], ctx_a, hist_a, H["gates"], H["knobs"],
                   {"model_sha256": "selftest", "fold": 0, "importances": []})
    assert res_a["verdict"] == "abstain" and res_a["rule_id"] == "OPT-NOFEAS"
    assert res_a["counts"]["feasible_n"] == 0
    assert res_a["counts"]["l0_pass_n"] == 256
    H["a_res"] = res_a
    print("[ok] decide: D-1-073 OPT-PICK index 249 (p_fail 0.043409, BLC_8 "
          "0.76712, 22 feasible of 227); A-1-000 OPT-NOFEAS")


def _g6_rank(H):
    pred = {"p_fail": [0.1, 0.1, 0.3, 0.05], "blc8": [0.5, 0.5, 0.9, 0.2],
            "log_cells": [5, 4, 3, 7]}
    assert rank(pred, [7, 3, 1, 2], math.log10(2e6)) == [1, 0]
    eq = {"p_fail": [0.1, 0.1], "blc8": [0.5, 0.5], "log_cells": [4, 4]}
    assert rank(eq, [9, 2], math.log10(2e6)) == [1, 0]
    none = {"p_fail": [0.5, 0.5], "blc8": [0.9, 0.1], "log_cells": [3, 3]}
    assert rank(none, [4, 6], math.log10(2e6)) == []
    X, yf, yb, ylc, gr = matrix(H["crows"][:30])
    ens = fit_ensemble(X, numpy.zeros(len(gr), dtype=bool), yb, ylc, gr)
    pr = predict(ens, X)
    assert numpy.all(pr["p_fail"] == 0.0) and numpy.all(pr["p_fail_std"] == 0.0)

    def boom(*_a):
        raise OptError("a recorded probe or snap value was run again")
    key = schema.canonical_sha256({"x": 1})
    _A, P, S = replay_fns({}, {key: None}, {key: None}, probe_fn=boom, snap_fn=boom)
    assert P(None, {"gid": "g"}, {"x": 1}, 1)["n_leaves"] is None
    assert S(None, {"gid": "g"}, {"x": 1}, 1) is None
    print("[ok] rank: feasible first, max BLC_8, then min cells, then the lower "
          "index; none feasible; a one-class member is constant; a recorded None "
          "probe or snap value is replayed, not run again")


def _g7_hook(H):
    gate = {"verdict": "FAIL", "auc": None, "blc8_rmse": 0.2,
            "beats_rules": False, "model_sha256": "selftest",
            "report": REPORT_NAME}
    hook = make_hook(lambda gid: (H["folds"][4], 4), model_sha256="selftest",
                     enabled=False, gate=gate)
    res = hook(H["d_ctx"], [{"attempt": 1, "config_sha256": "x", "rule_id": None}])
    assert res["verdict"] == "abstain" and res["rule_id"] == "OPT-DISABLED"
    gid = "D-1-010"
    fp = H["cends"][gid]["fingerprint"]
    s = rules.setup(H["mrows"][gid], fp, campaign.stl_rel(gid),
                    campaign.case_rel(gid), gid, gates=H["gates"],
                    knobs=H["knobs"])
    assert not s["refused"], s["refused"]
    ctxp = {"geometry_id": gid, "fingerprint": fp,
            "flow": H["mrows"][gid]["flow"], "win_level": s["summary"]["win_level"],
            "config": s["config"], "outcome": {}, "cwd": H["o_rules"]}
    hook2 = make_hook(lambda g: (H["folds"][fold_of(g)], fold_of(g)),
                      model_sha256="selftest")
    resp = hook2(ctxp, [])
    assert resp["verdict"] == "abstain" and resp["rule_id"] == "OPT-PLANE"
    campaign._check_hook(H["d_res"], "optimiser", H["d_ctx"]["config"], H["knobs"])
    for rec in (H["d_res"]["record"], H["a_res"]["record"], resp["record"],
                res["record"]):
        card = explain.card(rec)
        assert explain.ungrounded(card["line"], [rec]) == [], rec["rule_id"]
    print("[ok] hook seam: OPT-DISABLED on a disabled hook, OPT-PLANE on "
          "D-1-010, D-1-073's pick passes campaign._check_hook, every OPT card "
          "grounded")


def _g8_refine(H):
    tmp = H["tmp"]
    w8 = os.path.join(tmp, "w8")
    rep8 = os.path.join(tmp, "rep8")
    rep = refine(w8, sources=H["osrc"], rounds=2, streams=2, report_dir=rep8,
                 attempt_fn=prior._wall_oracle,
                 probe_fn=campaign._fake_probe(1000),
                 snap_fn=campaign._fake_snap, quiet=True)
    # the set grows to six with the lever refused: F-1-009's only fix is gone,
    # so the optimiser may propose for it again
    assert rep["refinement_set"]["geometry_ids"] == \
        ["D-1-001", "D-1-027", "D-1-073", "E-1-004", "F-1-009", "F-1-025"], \
        rep["refinement_set"]["geometry_ids"]
    r1 = baseline.read_bundle(os.path.join(rep8, "refine_r1.json.gz"))
    rows = {(r["geometry_id"], r["attempt"]): r for r in r1["attempts"]}
    assert rows[("D-1-073", 2)]["decided_by"] == "optimiser"
    assert rows[("D-1-073", 2)]["rule_id"] == "OPT-PICK"
    assert abs(rows[("D-1-073", 2)]["prediction"]["p_fail"] - 0.163066) <= 1e-6, \
        rows[("D-1-073", 2)]["prediction"]["p_fail"]
    assert rows[("F-1-025", 2)]["decided_by"] == "optimiser"
    assert rows[("F-1-025", 2)]["rule_id"] == "OPT-PICK"
    assert abs(rows[("F-1-025", 2)]["prediction"]["p_fail"] - 0.181093) <= 1e-6, \
        rows[("F-1-025", 2)]["prediction"]["p_fail"]
    sidx = {}
    for ln in r1["records"]:
        for i in ln["record"]["inputs"]:
            if i["name"] == "sobol_index":
                sidx[(ln["geometry_id"], ln["attempt"])] = i["value"]
    assert sidx[("D-1-073", 2)] == 0, sidx
    assert sidx[("F-1-025", 2)] == 4, sidx
    ends = {g["geometry_id"]: g for g in r1["geometries"]}
    assert ends["D-1-073"]["terminal"] == "PASS"
    assert ends["F-1-025"]["terminal"] == "PASS"
    assert rep["rounds"][0]["rescued"] == ["D-1-073", "F-1-025"]
    assert rep["rounds"][1]["rescued"] == ["D-1-073", "F-1-025"]
    assert rep["systems"]["rules"]["failures"] == 6
    assert abs(rep["systems"]["rules"]["mfr"] - 0.375) <= 1e-12
    assert rep["systems"]["round-2"]["failures"] == 4
    assert rep["cv"]["auc"] >= 0.75 and rep["cv"]["blc8_rmse"] == 0.0
    assert rep["conditions"]["beats_rules"] is True
    assert rep["verdict"] == "PASS" and rep["enabled"] is True
    for n in ("G-OPT.json", "G-OPT.md", "opt_model.json", "train.json.gz",
              "refine_r1.json.gz", "refine_r2.json.gz"):
        assert os.path.isfile(os.path.join(rep8, n)), n
    model = _read_json_or_none(os.path.join(rep8, "opt_model.json"))
    # the 104 oracle rows (group 2) plus round 1's two optimiser rows
    assert model["train"]["n_rows"] >= 106, model["train"]["n_rows"]
    H["rep8"] = rep8
    m1 = os.stat(os.path.join(w8, "round_1", "campaign.json")).st_mtime_ns
    m2 = os.stat(os.path.join(w8, "round_2", "campaign.json")).st_mtime_ns
    rep2 = refine(w8, sources=H["osrc"], rounds=2, streams=2, report_dir=rep8,
                  attempt_fn=prior._wall_oracle,
                  probe_fn=campaign._fake_probe(1000),
                  snap_fn=campaign._fake_snap, quiet=True)
    assert os.stat(os.path.join(w8, "round_1", "campaign.json")).st_mtime_ns == m1
    assert os.stat(os.path.join(w8, "round_2", "campaign.json")).st_mtime_ns == m2
    a = _read_json_or_none(os.path.join(rep8, "G-OPT.json"))
    d1 = dict(rep)
    d2 = dict(rep2)
    d1.pop("date")
    d2.pop("date")
    assert json.dumps(d1, sort_keys=True) == json.dumps(d2, sort_keys=True)
    res_c = check(rep8)
    assert res_c["verdict"] == "PASS", [i for i in res_c["items"] if not i["ok"]]
    om = importlib.import_module("optimise")
    saved = dict(om._MODEL)
    try:
        om._MODEL["path"] = os.path.join(rep8, "opt_model.json")
        om._MODEL["v"] = om._MODEL["key"] = None
        orb = baseline.read_bundle(H["osrc"][0])
        orows = rows_from(orb, "oracle-rules", H["mrows"], H["gates"], H["knobs"])
        a1 = next(r for r in orows
                  if r["geometry_id"] == "E-1-004" and r["attempt"] == 1)
        end4 = {g["geometry_id"]: g for g in orb["geometries"]}["E-1-004"]
        s4 = rules.setup(H["mrows"]["E-1-004"], end4["fingerprint"],
                         campaign.stl_rel("E-1-004"),
                         campaign.case_rel("E-1-004"), "E-1-004",
                         gates=H["gates"], knobs=H["knobs"])
        ctx = {"geometry_id": "E-1-004", "fingerprint": end4["fingerprint"],
               "flow": H["mrows"]["E-1-004"]["flow"],
               "win_level": s4["summary"]["win_level"], "config": a1["config"],
               "outcome": {}, "cwd": H["o_rules"]}
        hist = [{"attempt": 1, "config_sha256": a1["config_sha256"],
                 "rule_id": None}]
        d = om.propose(ctx, hist)
        assert schema.errors(d["record"], "DecisionRecord") == []
    finally:
        om._MODEL.clear()
        om._MODEL.update(saved)
    print("[ok] refine on the oracle: D-1-073 and F-1-025 rescued in round 1 "
          "(indices 0 and 4), MFR 0.375 -> 0.25, PASS and enabled; a re-run reuses "
          "both rounds and gives an equal report; check PASS")


def _g9_refine_a(H):
    tmp = H["tmp"]
    a_oracle = prior._oracle_attempt(
        lambda cfg, gctx: "F3a"
        if "A-1-000" in cfg["input"]["surfaces"][0]["path"] else "pass")
    srcA = os.path.join(tmp, "srcA")
    os.makedirs(srcA)
    for mode, name in (("rules", "rules"), ("b0-template", "tmpl"),
                       ("b0-lhs", "lhs")):
        out = os.path.join(tmp, "a_" + name)
        _oracle_run(out, prior.IDS8, mode, a_oracle)
        baseline.write_bundle(baseline.bundle_dir(out),
                              os.path.join(srcA, name + ".json.gz"))
    rep9 = os.path.join(tmp, "rep9")
    rep = refine(os.path.join(tmp, "w9"),
                 sources=[os.path.join(srcA, n + ".json.gz")
                          for n in ("rules", "tmpl", "lhs")],
                 rounds=1, streams=2, report_dir=rep9, attempt_fn=a_oracle,
                 probe_fn=campaign._fake_probe(1000),
                 snap_fn=campaign._fake_snap, quiet=True)
    assert rep["refinement_set"]["geometry_ids"] == ["A-1-000"]
    r1 = baseline.read_bundle(os.path.join(rep9, "refine_r1.json.gz"))
    row4 = next(r for r in r1["attempts"]
                if r["geometry_id"] == "A-1-000" and r["attempt"] == 4)
    assert row4["decided_by"] == "optimiser" and row4["rule_id"] == "OPT-PICK"
    assert row4["prediction"]["p_fail"] == 0.0
    endA = {g["geometry_id"]: g for g in r1["geometries"]}["A-1-000"]
    assert endA["terminal"] == "EXHAUSTED"
    assert rep["rounds"][0]["rescued"] == []
    assert rep["conditions"]["beats_rules"] is False
    assert rep["enabled"] is False
    model = _read_json_or_none(os.path.join(rep9, "opt_model.json"))
    assert model["enabled"] is False
    assert model["gate"]["beats_rules"] is False
    with open(os.path.join(rep9, "G-OPT.md"), encoding="utf-8") as f:
        assert "DISABLED" in f.read()
    print("[ok] refine on the A-only oracle: A-1-000 gets one proposal and still "
          "fails, nothing rescued, the optimiser ships DISABLED")


def _g10_live(H):
    tmp = H["tmp"]
    A, P, S = replay_fns(measured_from(H["rb"]), probes_from(H["rb"]),
                         snaps_from(H["rb"]))
    live = os.path.join(tmp, "live")
    # the committed rules campaign ran before FT-RADIUS, so its attempt 1 is replayed
    # and verified under the rules it was recorded with; restored in the finally below
    setup0 = rules.setup
    rules.setup = functools.partial(
        setup0, ft_radius=rules.ft_radius_of(H["rb"]["records"]))
    try:
        end = campaign.run_campaign(
            {"manifest": "tuning", "ids": ["D-1-073"], "out": live,
             "mode": "rules+opt", "streams": 2, "audit_mod": AUDIT_OFF,
             "quiet": True}, attempt_fn=A, probe_fn=P, snap_fn=S,
            hooks={"optimiser": make_hook(lambda gid: (H["folds"][4], 4),
                                          model_sha256="selftest")})
        lr = {r["attempt"]: r for r in campaign.load_rows(live)
              if r["geometry_id"] == "D-1-073"}
        comm = {r["attempt"]: r for r in H["rb"]["attempts"]
                if r["geometry_id"] == "D-1-073"}
        assert (lr[1]["config_sha"], lr[1]["decided_by"], lr[1]["rule_id"]) == \
            (comm[1]["config_sha"], comm[1]["decided_by"],
             comm[1]["rule_id"]), 1
        # the committed attempt 2 is RM-SNAP-FT, refused since 2026-09-26, so the
        # live campaign cannot replay it and the optimiser's first pick takes over
        assert lr[2]["decided_by"] == "optimiser" and lr[2]["rule_id"] == "OPT-PICK", \
            (lr[2]["decided_by"], lr[2]["rule_id"])
        assert end["harness_errors"] == 0 and end["orphans"] == []
        assert end["max_live_mesher"] <= 2, end["max_live_mesher"]
        # the first optimiser row is now attempt 2 (the committed attempt 2 is the
        # refused RM-SNAP-FT), so only attempt 1 is verified against the campaign
        verify_round(1, live,
                     [r for r in H["rb"]["attempts"]
                      if r["geometry_id"] == "D-1-073" and r["attempt"] == 1],
                     ["D-1-073"])
    finally:
        rules.setup = setup0
    oc = lr[2]["outcome"]
    print("[ok] live: rules+opt through the mesher at 2 streams on D-1-073: row "
          "1 replayed from the rules campaign, row 2 the optimiser's pick %s "
          "(exit %s, %s), 0 harness errors, 0 orphans, at most 2 meshers; verify "
          "ok" % (lr[2]["config_sha"][:12], oc["exit_code"], oc["failure_class"]))


def _g11_check_cli(H):
    tmp = H["tmp"]
    t1 = os.path.join(tmp, "rep_t1")
    shutil.copytree(H["rep8"], t1)
    p = os.path.join(t1, "train.json.gz")
    data = bytearray(open(p, "rb").read())
    data[len(data) // 2] ^= 0x01
    with open(p, "wb") as f:
        f.write(bytes(data))
    res = check(t1)
    assert res["verdict"] == "FAIL"
    assert not next(i for i in res["items"] if i["name"] == "train file")["ok"]
    t2 = os.path.join(tmp, "rep_t2")
    shutil.copytree(H["rep8"], t2)
    mp = os.path.join(t2, "opt_model.json")
    m = _read_json_or_none(mp)
    m["p_fail_max"] = 0.5
    with open(mp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(m, f, indent=1, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    res2 = check(t2)
    assert res2["verdict"] == "FAIL"
    assert not next(i for i in res2["items"] if i["name"] == "model sha")["ok"]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    p1 = subprocess.run([sys.executable, __file__, "--plan"], capture_output=True,
                        text=True, encoding="utf-8", errors="replace", env=env,
                        timeout=300)
    assert p1.returncode == 0 and "cv: AUC 0.985904" in p1.stdout, \
        (p1.returncode, p1.stdout[-400:], p1.stderr[-400:])
    p2 = subprocess.run([sys.executable, __file__, "--refine"],
                        capture_output=True, text=True, encoding="utf-8",
                        env=env, timeout=300)
    assert p2.returncode == 2 and "needs --work" in p2.stderr, \
        (p2.returncode, p2.stderr)
    p3 = subprocess.run([sys.executable, __file__, "--check", "--report-dir",
                         H["rep8"]], capture_output=True, text=True,
                        encoding="utf-8", env=env, timeout=300)
    assert p3.returncode == 0 and "CHECK PASS" in p3.stdout, \
        (p3.returncode, p3.stdout[-400:], p3.stderr[-400:])
    p4 = subprocess.run([sys.executable, __file__, "--check", "--report-dir",
                         os.path.join(tmp, "nothing")], capture_output=True,
                        text=True, encoding="utf-8", env=env, timeout=300)
    assert p4.returncode == 1, (p4.returncode, p4.stdout[-400:])
    print("[ok] check: two tampers FAIL by name; the CLI plans on the committed "
          "data, refuses a refine without --work, and checks")


if __name__ == "__main__":
    sys.exit(main())
