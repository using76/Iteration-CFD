#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""
prior.py - the L3 prior of docs/15 §C: a warm start for attempt one.

A distance-weighted k-NN (k = 3) over standardised fingerprints of the PASSING
tuning geometries.  Every fingerprint becomes seventeen shape features with a
stated fill value and a missing-indicator for every nullable field; the pool
mean and std standardise them; the three nearest passing neighbours vote with
weight 1/(d + eps) over the remedy PATHS they needed before they passed, and
the winning path's refinement and snap remedies are re-applied to this
geometry's L1 config through remedies.py's own functions, guards and _commit -
a wall level never drops below this geometry's y+ floor, the R-PLANE path is
left alone, and the layers block is never touched.  The prior abstains by name
(PR-FAR, PR-KEEP, PR-NOEDIT) when the neighbours are too far, needed nothing,
or the path changes nothing here.  G-PRIOR (docs/15 §F) measures, on the tuning
split and leave-one-geometry-out, whether the real prior's attempt-1 pass rate
beats rules-only's and a shuffled-fingerprint control; if either condition
fails the model ships DISABLED and records PR-DISABLED on every geometry.

    python tools/autonomy/prior.py --selftest
    python tools/autonomy/prior.py --plan --rules DIR
    python tools/autonomy/prior.py --gate --rules DIR --work DIR
    python tools/autonomy/prior.py --check
"""
import argparse
import copy
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

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for _p in (HERE, os.path.join(HERE, "corpus")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import baseline
import campaign
import explain
import preflight
import remedies
import rules
import schema
import split


class PriorError(ValueError):
    """A refused request or a harness inconsistency - never a verdict."""


REPORT_DIR = os.path.join(HERE, "prior")
MODEL_NAME = "prior_model.json"
REPORT_NAME = "G-PRIOR.json"
REPORT_MD = "G-PRIOR.md"
RULES_BUNDLE = "tuning_rules.json.gz"      # the rules campaign the bank is built from
ROUND_BUNDLE = "eval_r%d.json.gz"          # evaluation round j (1-based)
K = 3                                      # docs/15 §C L3
EPS = 1e-6                                 # weight 1 / (d + EPS)
D_ABSTAIN_PCT = 95.0                       # the locked abstention distance: this percentile of (C3)
SHUFFLES = 3
SHUFFLE_SEED = 13
AUDIT_OFF = 2 ** 40                        # audit_mod for the evaluation rounds (baseline.py's R-CURV value)
STD_MIN = 1e-12                            # a feature whose pool std is not above this is dropped
TRANSFER_STAGES = ("octree", "castellate", "snap")
TRANSFER = tuple(r["id"] for r in remedies.REMEDIES if r["stage"] in TRANSFER_STAGES)
PR_IDS = ("PR-KNN", "PR-KEEP", "PR-FAR", "PR-NOEDIT", "PR-DISABLED")
VARIANTS = ("real", "shuffle-0", "shuffle-1", "shuffle-2")
FEATURES = ("log10_lmax", "mid_over_lmax", "min_over_lmax", "log10_area_over_lmax2",
            "volume_over_bbox", "log10_1p_sharp_over_lmax", "log10_r5_over_lmax",
            "log10_r50_over_lmax", "log10_r95_over_lmax", "curv_missing",
            "log10_inner_over_lmax", "inner_missing", "log10_gap_over_lmax",
            "gap_missing", "planar_frac", "commensurate", "log10_n_triangles")
CURV = ("curvature_radius_p5_m", "curvature_radius_p50_m", "curvature_radius_p95_m")
CLIP = {"curv": (-3.0, 2.0), "inner": (-3.0, 0.0), "gap": (-3.0, 1.0)}
FILL = {"curv": 2.0, "inner": 0.0, "gap": 1.0}
MODEL_SCHEMA = "autonomy-prior-model/1"
GATE_SCHEMA = "autonomy-prior-gate/1"
CHECK_SCHEMA = "autonomy-prior-check/1"
DECISION_SCHEMA = "autonomy-prior-decision/1"
HEADER = baseline.HEADER                   # the "$comment" of every JSON it writes
_MODEL = {"path": os.path.join(REPORT_DIR, MODEL_NAME), "v": None, "key": None}
_REMEDY_BY_ID = {r["id"]: r for r in remedies.REMEDIES}


NULL_POLICY = (
    {"fields": list(CURV),
     "features": ["log10_r5_over_lmax", "log10_r50_over_lmax", "log10_r95_over_lmax"],
     "indicator": "curv_missing", "fill": FILL["curv"], "clip": list(CLIP["curv"]),
     "why": "features.py leaves curvature null when no triangle is curved (a box); "
            "a plane's radius is infinite, so it takes the flattest value the clip allows"},
    {"fields": ["inner_thickness_m"], "features": ["log10_inner_over_lmax"],
     "indicator": "inner_missing", "fill": FILL["inner"], "clip": list(CLIP["inner"]),
     "why": "null when no inner tangent ball meets a facing wall: there is no thin "
            "section, so the thickness is taken as the body's own largest extent"},
    {"fields": ["outer_gap_m"], "features": ["log10_gap_over_lmax"],
     "indicator": "gap_missing", "fill": FILL["gap"], "clip": list(CLIP["gap"]),
     "why": "null when no outer tangent ball meets a second wall (a single body): "
            "the gap is taken as ten body extents"},
)

EXCLUDED = {
    "lattice_base_size_m":
        "null on every non-commensurate body; the commensurate feature carries it",
    "patches": "patch names are the generator's; every tuning fingerprint has one patch",
    "feature_angle_deg": "the same 30 degrees on every fingerprint",
    "stl_sha256": "an identity, not a shape",
    "geometry_id": "an identity, not a shape",
}

# docs/15 §F: where the tree departs from the plan's wording, stated on the record.
DEPARTURES = (
    "\"leave-one-group-out\" and \"leave-one-geometry-out\" are the same fold here: each "
    "geometry is one group and contributes at most one bank entry (its earliest attempt "
    "with no F flag)",
    "docs/15 §C L3 says the prior \"transfers only refinement and snap knobs\"; the tree "
    "transfers them as a remedy PATH - the refinement and snap remedies (stages octree, "
    "castellate, snap) the neighbour needed before it passed are re-applied to this "
    "geometry's L1 config through remedies.py's own functions, guards and _commit, so a "
    "wall level never drops below this geometry's y+ floor, the R-PLANE path is left "
    "alone, and the layers block is never touched (L1 always recomputes t1 and the window)",
    "the standardisation (mean, std) and the abstention distance use every fingerprinted "
    "tuning geometry with its outcome unused; the bank holds only the passing ones",
    "the shuffled control is three permutations of the bank's fingerprints inside each "
    "fold; the gate compares their mean. A config the rules campaign already ran for that "
    "geometry (equal config sha256) is not meshed again (the mesher is deterministic, "
    "docs/15 §K G-DET). Evaluation rounds run with the audit sample off",
    "first-attempt pass = the attempt-1 outcome's failure is False (docs/15 §D.1, the MFR "
    "definition); strict passes are reported, not gated. A geometry with no attempt-1 row "
    "(SURFACE-OPEN, SURFACE-REFUSED, REFUSED) is a first-attempt failure in every "
    "variant, as MFR counts it",
)


def _dump(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def _clip(x, lo_hi):
    lo, hi = lo_hi
    return min(max(x, lo), hi)


def features(fp):
    """The fingerprint as 17 floats, in FEATURES order, with the null policy."""
    errs = schema.errors(fp, "Fingerprint")
    if errs:
        raise PriorError("the fingerprint is invalid: %s" % errs[0])
    b = fp["bbox"]
    e = sorted([b[1] - b[0], b[3] - b[2], b[5] - b[4]])
    if any(v <= 0 for v in e):
        raise PriorError("the fingerprint's bbox is degenerate: extents %s" % (e,))
    L = e[2]
    out = [math.log10(L), e[1] / L, e[0] / L,
           math.log10(fp["area_m2"] / L ** 2),
           fp["volume_m3"] / (e[0] * e[1] * e[2]),
           math.log10(1.0 + fp["sharp_edge_length_m"] / L)]
    for f in CURV:
        v = fp[f]
        out.append(FILL["curv"] if v is None else _clip(math.log10(v / L), CLIP["curv"]))
    out.append(1.0 if fp["curvature_radius_p50_m"] is None else 0.0)
    v = fp["inner_thickness_m"]
    out.append(FILL["inner"] if v is None else _clip(math.log10(v / L), CLIP["inner"]))
    out.append(1.0 if v is None else 0.0)
    v = fp["outer_gap_m"]
    out.append(FILL["gap"] if v is None else _clip(math.log10(v / L), CLIP["gap"]))
    out.append(1.0 if v is None else 0.0)
    out.append(float(fp["planar_frac"]))
    out.append(1.0 if fp["commensurate"] else 0.0)
    out.append(math.log10(fp["n_triangles"]))
    if any(not math.isfinite(float(v)) for v in out):
        raise PriorError("the fingerprint features are not finite")
    return [float(v) for v in out]


def fit_scaler(X):
    """The pool's per-feature mean and std (ddof 0); features with std <= STD_MIN drop."""
    X = list(X)
    if len(X) < 2:
        raise PriorError("pool: at least 2 fingerprints are needed, got %d" % len(X))
    A = numpy.asarray(X, dtype=float)
    if A.ndim != 2 or A.shape[0] < 2:
        raise PriorError("pool: the rows must be equal-length float vectors")
    mean = A.mean(axis=0)
    std = A.std(axis=0)
    kept = std > STD_MIN
    return {"mean": mean.tolist(), "std": std.tolist(),
            "kept": [bool(k) for k in kept]}


def standardise(x, scaler):
    """The kept features of one vector, standardised: (x - mean) / std, in order."""
    return numpy.asarray([(x[i] - scaler["mean"][i]) / scaler["std"][i]
                          for i in range(len(x)) if scaler["kept"][i]], dtype=float)


def rms_distance(a, b):
    """The root-mean-square distance over the vector's entries."""
    return float(numpy.sqrt(numpy.mean(
        (numpy.asarray(a, dtype=float) - numpy.asarray(b, dtype=float)) ** 2)))


def d_abstain(Z):
    """The locked abstention distance: the D_ABSTAIN_PCT percentile of the nearest-
    neighbour RMS distances inside the pool (fingerprints only)."""
    Z = [numpy.asarray(z, dtype=float) for z in Z]
    if len(Z) < 2:
        raise PriorError("pool: at least 2 standardised fingerprints are needed, "
                         "got %d" % len(Z))
    nn = [min(rms_distance(Z[i], Z[j]) for j in range(len(Z)) if j != i)
          for i in range(len(Z))]
    return float(numpy.percentile(numpy.asarray(nn), D_ABSTAIN_PCT))


def vote(zq, bank, d_abs, k=K):
    """The distance-weighted k-NN vote over the neighbours' remedy paths."""
    scored = sorted(((rms_distance(zq, e["z"]), e["geometry_id"], e) for e in bank),
                    key=lambda t: (t[0], t[1]))
    out = {"neighbours": [t[1] for t in scored[:k]],
           "distances": [t[0] for t in scored[:k]],
           "paths": [list(t[2]["path"]) for t in scored[:k]],
           "nearest": scored[0][0] if scored else None}
    if not scored:
        out.update({"kind": "few", "weights": [], "winner": None})
        return out
    nb = scored[:k]
    if len(bank) < k:
        out.update({"kind": "few", "weights": [], "winner": None})
        return out
    if nb[0][0] > d_abs:
        out.update({"kind": "far", "weights": [], "winner": None})
        return out
    scores = {}
    for d_, _gid, e in nb:
        key = tuple(e["path"])
        scores[key] = scores.get(key, 0.0) + 1.0 / (d_ + EPS)
    best = None
    winner = None
    for d_, _gid, e in nb:          # ties go to the earliest neighbour in nb order
        key = tuple(e["path"])
        if winner is None or scores[key] > best:
            best, winner = scores[key], key
    out.update({"kind": "keep" if winner == () else "vote",
                "weights": [1.0 / (d_ + EPS) for d_ in out["distances"]],
                "winner": list(winner)})
    return out


def bundle_sha(bundle):
    """sha256 hex over the canonical bytes of the bundle (baseline's content hash)."""
    return hashlib.sha256(json.dumps(bundle, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def bank_from(bundle):
    """One entry per passing geometry: its earliest passing attempt and the
    transferred remedies it needed on the way (layer-stage remedies dropped)."""
    ends = {g["geometry_id"]: g for g in bundle["geometries"]}
    by = {}
    for r in bundle["attempts"]:
        by.setdefault(r["geometry_id"], {})[r["attempt"]] = r
    out = []
    for gid in sorted(by):
        att = by[gid]
        want = list(range(1, max(att) + 1))
        if sorted(att) != want:
            raise PriorError("incomplete: %s carries attempts %s, not 1..%d"
                             % (gid, sorted(att), max(att)))
        p = None
        for a in want:
            if att[a]["outcome"]["failure"] is False:
                p = a
                break
        if p is None:
            continue
        end = ends.get(gid)
        if end is None:
            raise PriorError("incomplete: no end record for %s" % gid)
        out.append({"geometry_id": gid, "family": end["family"], "attempt": p,
                    "path": [att[a]["rule_id"] for a in range(2, p + 1)
                             if att[a]["decided_by"] == "remedy"
                             and att[a]["rule_id"] in TRANSFER],
                    "fingerprint_sha256":
                        schema.canonical_sha256(end["fingerprint"])})
    return out


def build(bundle):
    """The model: the pool's scaler and abstention distance, the bank with z."""
    pool = [g for g in bundle["geometries"] if g.get("fingerprint") is not None]
    pool.sort(key=lambda g: g["geometry_id"])
    if len(pool) < 2:
        raise PriorError("pool: %d fingerprinted geometries is not a pool"
                         % len(pool))
    scaler = fit_scaler([features(g["fingerprint"]) for g in pool])
    d_abs = d_abstain([standardise(features(g["fingerprint"]), scaler)
                       for g in pool])
    bank = []
    for e in bank_from(bundle):
        e = dict(e)
        e["z"] = standardise(features(_pool_fp(pool, e["geometry_id"])),
                             scaler).tolist()
        bank.append(e)
    head = bundle["campaign"]
    return {"$comment": HEADER, "schema": MODEL_SCHEMA, "k": K, "eps": EPS,
            "features": list(FEATURES),
            "kept": [f for f, k in zip(FEATURES, scaler["kept"]) if k],
            "scaler": scaler, "d_abstain": d_abs,
            "d_abstain_pct": D_ABSTAIN_PCT, "transfer": list(TRANSFER),
            "null_policy": list(NULL_POLICY), "excluded": dict(EXCLUDED),
            "pool_n": len(pool), "bank_n": len(bank), "bank": bank,
            "source": {"campaign_id": head["campaign_id"],
                       "binary_sha256": head["binary_sha256"],
                       "git_sha": head["git_sha"],
                       "manifest_sha256": head["manifest"]["sha256"],
                       "n_geometries": len(head["geometry_ids"]),
                       "content_sha256": bundle_sha(bundle)},
            "enabled": False, "gate": None}


def _pool_fp(pool, gid):
    for g in pool:
        if g["geometry_id"] == gid:
            return g["fingerprint"]
    raise PriorError("incomplete: no fingerprinted end record for %s" % gid)


def model_sha(model):
    """canonical_sha256 of the model without enabled/gate - built == shipped."""
    return schema.canonical_sha256({k: v for k, v in model.items()
                                    if k not in ("enabled", "gate")})


PLANE_WHY = "on the R-PLANE path (the plane owns the refinement and snap knobs)"


def apply_path(config, path, ctx, gates, knobs):
    """Re-apply a neighbours' remedy path to the L1 config through remedies.py.

    The ONLY place a config is changed; every step goes through the remedy
    table's own function, its guards and _commit.  Returns (after or None,
    skipped)."""
    if remedies.on_plane(config, ctx["fingerprint"]):
        return None, [{"rule_id": None, "why": PLANE_WHY}]
    cur = copy.deepcopy(config)
    skipped = []
    for rid in path:
        if rid not in TRANSFER:
            raise PriorError("the prior transfers only %s, not %s"
                             % (", ".join(TRANSFER), rid))
        row = _REMEDY_BY_ID[rid]
        after, why, _extra = getattr(remedies, row["fn"])(
            dict(ctx, config=cur), gates, knobs)
        if after is None:
            skipped.append({"rule_id": rid, "why": why})
            continue
        remedies._commit(cur, after, row["stage"], knobs)   # raises on an illegal edit
        cur = after
    if schema.canonical_sha256(cur) == schema.canonical_sha256(config):
        return None, skipped
    return cur, skipped


def _record(rid, verdict, trigger, inputs, formula, edits, cite, message,
            uncertainty):
    rec = {"schema": "autonomy-decision/1", "layer": "prior", "rule_id": rid,
           "verdict": verdict, "trigger": trigger, "inputs": inputs,
           "formula": formula, "edits": edits, "cite": cite, "message": message,
           "uncertainty": uncertainty, "t": schema._now_iso()}
    errs = schema.errors(rec, "DecisionRecord")
    if errs:
        raise PriorError("the prior's record is invalid: %s" % errs[0])
    return rec


FORMULA = ("k-NN (k = 3) on the RMS distance of standardised fingerprint "
           "features; weight 1/(d + 1e-6); the remedy path with the largest summed "
           "weight wins, ties to the nearest; its refinement and snap remedies are "
           "re-applied through remedies.py")
CITE = ("docs/15 §C L3 prior, §F G-PRIOR; tools/autonomy/remedies.py "
        "(the transferred refinement and snap remedies)")


def _inputs(v, skipped, model, bank_n):
    return [{"name": "neighbours", "value": list(v["neighbours"]), "unit": ""},
            {"name": "distances", "value": [float(d) for d in v["distances"]],
             "unit": "1"},
            {"name": "weights", "value": [float(w) for w in v["weights"]],
             "unit": "1"},
            {"name": "paths", "value": [list(p) for p in v["paths"]], "unit": ""},
            {"name": "winner",
             "value": list(v["winner"]) if v["winner"] is not None else None,
             "unit": ""},
            {"name": "skipped", "value": copy.deepcopy(skipped), "unit": ""},
            {"name": "d_abstain", "value": float(model["d_abstain"]), "unit": "1"},
            {"name": "k", "value": int(model["k"]), "unit": "1"},
            {"name": "bank_n", "value": int(bank_n), "unit": "1"},
            {"name": "model_sha256", "value": model_sha(model), "unit": ""}]


def _trigger(v, model):
    near, d_abs = v["nearest"], model["d_abstain"]
    return {"observable": "prior.nearest_distance", "value": near,
            "threshold": d_abs,
            "op": "<=" if near is not None and near <= d_abs else ">",
            "source": "prior_model.json d_abstain: the 95th percentile of the "
                      "tuning fingerprints' nearest-neighbour distance (docs/15 "
                      "§C L3)"}


def _nb_text(v):
    return ", ".join("%s at %s" % (gid, explain.fmt(d))
                     for gid, d in zip(v["neighbours"], v["distances"]))


def decide(zq, bank, model, ctx, gates, knobs):
    """One prior decision on one geometry: the vote, the config, the record."""
    v = vote(zq, bank, model["d_abstain"], model["k"])
    skipped = []
    after = None
    if v["kind"] in ("few", "far"):
        rid = "PR-FAR"
    elif v["kind"] == "keep":
        rid = "PR-KEEP"
    else:
        after, skipped = apply_path(ctx["config"], v["winner"], ctx, gates, knobs)
        rid = "PR-KNN" if after is not None else "PR-NOEDIT"
    verdict = "apply" if after is not None else "abstain"
    edits = rules.diff_edits(ctx["config"], after) if after is not None else []
    if rid == "PR-KNN":
        msg = ("PR-KNN: the %d nearest passing tuning geometries (%s) lie within "
               "d_abstain %s; their path %s won the vote and is re-applied here: %s"
               % (model["k"], _nb_text(v), explain.fmt(model["d_abstain"]),
                  " + ".join(v["winner"]), explain.edits_text(edits)))
    elif rid == "PR-KEEP":
        msg = ("PR-KEEP: the %d nearest passing tuning geometries (%s) passed with "
               "the setup rules' own config; attempt 1 stays the rules' config"
               % (model["k"], _nb_text(v)))
    elif v["kind"] == "few":
        msg = ("PR-FAR: the bank holds %d passing tuning geometries, fewer than "
               "k = %d; the prior abstains" % (len(bank), model["k"]))
    elif v["kind"] == "far":
        msg = ("PR-FAR: the nearest passing tuning geometry %s is at %s > d_abstain "
               "%s; the prior abstains" % (v["neighbours"][0],
                                           explain.fmt(v["nearest"]),
                                           explain.fmt(model["d_abstain"])))
    else:
        msg = ("PR-NOEDIT: the neighbours' path %s changes nothing here (%s); "
               "attempt 1 stays the rules' config"
               % (" + ".join(v["winner"]),
                  "; ".join("%s: %s" % (s["rule_id"] or "R-PLANE", s["why"])
                            for s in skipped)))
    rec = _record(rid, verdict, _trigger(v, model), _inputs(v, skipped, model,
                                                            len(bank)),
                  FORMULA, edits, CITE, msg, v["nearest"] or 0.0)
    return {"schema": DECISION_SCHEMA, "geometry_id": ctx["geometry_id"],
            "verdict": verdict, "rule_id": rid, "config": after, "edits": edits,
            "record": rec, "vote": v, "skipped": skipped}


def disabled_decision(model):
    """The abstain decision recorded on every geometry when the prior ships disabled."""
    g = model["gate"]
    msg = ("PR-DISABLED: the prior ships disabled (G-PRIOR %s: attempt-1 passes "
           "rules %s, real %s, shuffled mean %s of %s); attempt 1 stays the rules' "
           "config" % (g["verdict"], explain.fmt(g["rules_pass"]),
                       explain.fmt(g["real_pass"]),
                       explain.fmt(g["shuffled_mean_pass"]), explain.fmt(g["n"])))
    rec = _record("PR-DISABLED", "abstain",
                  {"observable": "prior.enabled", "value": False, "threshold": False,
                   "op": "==",
                   "source": "prior/G-PRIOR.json (docs/15 §F G-PRIOR)"},
                  [{"name": "verdict", "value": g["verdict"], "unit": ""},
                   {"name": "rules_pass", "value": g["rules_pass"], "unit": ""},
                   {"name": "real_pass", "value": g["real_pass"], "unit": ""},
                   {"name": "shuffled_mean_pass",
                    "value": g["shuffled_mean_pass"], "unit": ""},
                   {"name": "n", "value": g["n"], "unit": ""},
                   {"name": "model_sha256", "value": model_sha(model), "unit": ""}],
                  "the gate did not earn the prior its place, so the setup rules "
                  "decide attempt one", [],
                  "docs/15 §F G-PRIOR; tools/autonomy/prior.py --gate", msg, 0.0)
    return {"schema": DECISION_SCHEMA, "geometry_id": None, "verdict": "abstain",
            "rule_id": "PR-DISABLED", "config": None, "edits": [], "record": rec,
            "vote": None, "skipped": []}


def load_model(path=None):
    """The gated prior model, cached by (path, mtime)."""
    path = path or _MODEL["path"]
    key = (path, os.stat(path).st_mtime_ns) if os.path.isfile(path) else None
    if key is not None and _MODEL["key"] == key and _MODEL["v"] is not None:
        return _MODEL["v"]
    if not os.path.isfile(path):
        raise PriorError("no prior model at %s (run prior.py --gate)" % path)
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    if not isinstance(m, dict) or m.get("schema") != MODEL_SCHEMA:
        raise PriorError("%s is not a %s model" % (path, MODEL_SCHEMA))
    if m.get("gate") is None:
        raise PriorError("%s has not been gated (run prior.py --gate)" % path)
    _MODEL["v"], _MODEL["key"] = m, key
    return m


def make_hook(model):
    """The campaign hook: PR-DISABLED when disabled, decide() otherwise."""
    gates = schema.load_gates()
    knobs = schema.load_knobs()

    def hook(ctx):
        if not model.get("enabled"):
            d = disabled_decision(model)
            return {"verdict": d["verdict"], "record": d["record"]}
        d = decide(standardise(features(ctx["fingerprint"]), model["scaler"]),
                   model["bank"], model, ctx, gates, knobs)
        out = {"verdict": d["verdict"], "record": d["record"]}
        if d["verdict"] == "apply":
            out["config"] = d["config"]
            out["edits"] = d["edits"]
        return out
    return hook


def attempt1(ctx):
    """The hook campaign.py's HOOKS names; the model comes from prior_model.json."""
    return make_hook(load_model())(ctx)


def decide_all(bundle, model, gates, knobs):
    """Every geometry's decision under the real variant and the three shuffles."""
    head = bundle["campaign"]
    mrows = {r["geometry_id"]: r for r in campaign.load_manifest(
        "tuning", "rules", ids=head["geometry_ids"])}
    ends = {g["geometry_id"]: g for g in bundle["geometries"]}
    rows_by = {}
    for r in bundle["attempts"]:
        rows_by.setdefault(r["geometry_id"], {})[r["attempt"]] = r
    out = []
    for gid in sorted(head["geometry_ids"]):
        end, rows = ends[gid], rows_by.get(gid, {})
        base = {"geometry_id": gid, "family": end["family"],
                "terminal": campaign.terminal_of(end)}
        if 1 not in rows:
            out.append(dict(base, eligible=False, variants=None))
            continue
        fp = end["fingerprint"]
        s = rules.setup(mrows[gid], fp, campaign.stl_rel(gid),
                        campaign.case_rel(gid), gid, gates=gates, knobs=knobs)
        if s["refused"]:
            raise PriorError("rules.setup refuses %s: %s"
                             % (gid, ", ".join(s["refused"])))
        if schema.canonical_sha256(s["config"]) != rows[1]["config_sha"]:
            raise PriorError("the rules campaign's attempt 1 is not rules.setup's "
                             "config (%s)" % gid)
        ctx = {"geometry_id": gid, "fingerprint": fp, "flow": mrows[gid]["flow"],
               "win_level": s["summary"]["win_level"], "config": s["config"],
               "rules": s}
        zq = standardise(features(fp), model["scaler"])
        fold = [e for e in model["bank"] if e["geometry_id"] != gid]
        variants = {"real": decide(zq, fold, model, ctx, gates, knobs)}
        for r in range(SHUFFLES):
            rng = numpy.random.default_rng(
                [SHUFFLE_SEED, r, zlib.crc32(gid.encode("ascii"))])
            perm = rng.permutation(len(fold))
            sbank = [dict(fold[i], z=fold[int(perm[i])]["z"])
                     for i in range(len(fold))]
            variants["shuffle-%d" % r] = decide(zq, sbank, model, ctx, gates, knobs)
        out.append(dict(base, eligible=True,
                        l1_sha256=schema.canonical_sha256(s["config"]),
                        fingerprint_sha256=schema.canonical_sha256(fp),
                        variants=variants))
    return out


def plan_rounds(decisions, measured):
    """The evaluation rounds: the distinct (geometry, config) pairs the rules
    campaign has not already measured, in first-seen order, one round per rank."""
    needed, index = [], {}
    for variant in VARIANTS:
        for d in decisions:
            if not d["eligible"]:
                continue
            dec = d["variants"][variant]
            if dec["verdict"] != "apply":
                continue
            gid = d["geometry_id"]
            csha = schema.canonical_sha256(dec["config"])
            if csha in measured[gid]:
                continue
            key = (gid, csha)
            if key not in index:
                index[key] = {"geometry_id": gid, "config": dec["config"],
                              "config_sha256": csha, "l1_sha256": d["l1_sha256"],
                              "fingerprint_sha256": d["fingerprint_sha256"],
                              "record": dec["record"], "variants": []}
                needed.append(index[key])
            if variant not in index[key]["variants"]:
                index[key]["variants"].append(variant)
    rounds = []
    rank = 0
    while True:
        rnd = {}
        for d in decisions:
            items = [it for it in needed if it["geometry_id"] == d["geometry_id"]]
            if rank < len(items):
                rnd[d["geometry_id"]] = items[rank]
        if not rnd:
            return rounds
        rounds.append(rnd)
        rank += 1


def _seal_check(rules_dir, head):
    """The seal FIRST: the prior learns from the tuning split only (docs/15 §F)."""
    if head.get("split_mode") == split.EVALUATE or \
            head.get("manifest", {}).get("source") != "tuning":
        raise split.SplitSealed(
            "the prior learns from the tuning split only (docs/15 §F): %s is a %s "
            "campaign of %s" % (rules_dir, head.get("split_mode"),
                                head.get("manifest", {}).get("source")))
    if head.get("mode") != "rules" or head.get("ablate") != []:
        raise PriorError("the G-PRIOR baseline is a rules campaign with nothing "
                         "ablated: %s is mode %s with ablate %s"
                         % (rules_dir, head.get("mode"), head.get("ablate")))


def _round_lookup(rnd):
    """The round's hook: hand the planned config to the campaign runner."""
    def hook(ctx):
        item = rnd.get(ctx["geometry_id"])
        if item is None:
            raise PriorError("round config drift: no planned config for %s"
                             % ctx["geometry_id"])
        if schema.canonical_sha256(ctx["config"]) != item["l1_sha256"]:
            raise PriorError("round config drift: %s's attempt-1 config is not the "
                             "planned L1 config" % ctx["geometry_id"])
        if schema.canonical_sha256(ctx["fingerprint"]) != item["fingerprint_sha256"]:
            raise PriorError("round config drift: %s's fingerprint differs from the "
                             "plan" % ctx["geometry_id"])
        rec = copy.deepcopy(item["record"])
        rec["t"] = schema._now_iso()
        return {"verdict": "apply", "config": copy.deepcopy(item["config"]),
                "edits": rules.diff_edits(ctx["config"], item["config"]),
                "record": rec}
    return hook


def _run_round(j, rnd, work, *, streams, binary, attempt_fn, probe_fn, snap_fn,
               quiet):
    """One evaluation round: run it, or reuse the completed directory."""
    ids = sorted(rnd)
    out = os.path.join(work, "round_%d" % j)
    end_path = os.path.join(out, campaign.FILES["end"])
    reused = False
    if os.path.isfile(end_path):
        with open(os.path.join(out, campaign.FILES["campaign"]),
                  encoding="utf-8") as f:
            rhead = json.load(f)
        if sorted(rhead["geometry_ids"]) != ids:
            raise PriorError("round %d holds %s, the plan needs %s"
                             % (j, sorted(rhead["geometry_ids"]), ids))
        reused = True
    elif os.path.isdir(out) and os.listdir(out):
        raise PriorError("round %d is incomplete: remove %s and run it again"
                         % (j, out))
    print("[prior] round %d: %d geometries (%s)"
          % (j, len(ids), "reused" if reused else "run"))
    if not reused:
        campaign.run_campaign({"manifest": "tuning", "mode": "rules+prior",
                               "ablate": ("remedies",), "ids": ids, "out": out,
                               "run_id": "prior-eval-r%d" % j, "streams": streams,
                               "audit_mod": AUDIT_OFF,
                               "binary": binary or campaign.BINARY_DEFAULT,
                               "quiet": quiet},
                              attempt_fn=attempt_fn, probe_fn=probe_fn,
                              snap_fn=snap_fn, hooks={"prior": _round_lookup(rnd)})
    return out


def _outcome_of(dec, gid, rows1, measured, round_rows, rounds_of):
    """(C8): one variant's status and outcome for one eligible geometry."""
    entry = {}
    if dec["verdict"] == "abstain":
        entry["status"] = dec["rule_id"]
        return entry, rows1[1]["outcome"]
    csha = schema.canonical_sha256(dec["config"])
    if csha in measured[gid]:
        a = measured[gid][csha]
        entry["status"] = "reused"
        entry["attempt"] = a
        return entry, rows1[a]["outcome"]
    j = rounds_of[(gid, csha)]
    row1 = round_rows[j][gid]
    entry["round"] = j
    if row1["decided_by"] == "prior" and row1["config_sha"] == csha:
        entry["status"] = "ran"
    elif row1["decided_by"] == "rule":
        entry["status"] = "refused"
        entry["refused_by"] = [r["rule_id"] for r in row1["constraint_refusals"]]
    else:
        raise PriorError("round %d's attempt 1 for %s was decided by %s"
                         % (j, gid, row1["decided_by"]))
    return entry, row1["outcome"]


def _bundle_info(fname, bundle, report_dir, write):
    """Write (or keep) one bundle; its stats dict for the report."""
    content = bundle_sha(bundle)
    if not write:
        return {"file": fname, "sha256": None, "content_sha256": content,
                "bytes": None}
    path = os.path.join(report_dir, fname)
    if os.path.exists(path):
        try:
            old = baseline.read_bundle(path)
        except (baseline.BaselineError, OSError, ValueError):
            old = None
        if old is None or bundle_sha(old) != content:
            raise PriorError("exists: %s holds a different bundle" % path)
        with open(path, "rb") as f:
            data = f.read()
        return {"file": fname, "sha256": hashlib.sha256(data).hexdigest(),
                "content_sha256": content, "bytes": len(data)}
    os.makedirs(report_dir, exist_ok=True)
    info = baseline.write_bundle(bundle, path)
    return {"file": fname, "sha256": info["sha256"], "content_sha256": content,
            "bytes": info["bytes"]}


def _prepare(rules_dir):
    """gate's steps 1-4: the seal, the bundle checks, the model, the decisions."""
    cpath = os.path.join(rules_dir, campaign.FILES["campaign"])
    if not os.path.isfile(cpath):
        raise PriorError("no rules campaign at %s: %s is missing"
                         % (rules_dir, cpath))
    with open(cpath, encoding="utf-8") as f:
        head = json.load(f)
    _seal_check(rules_dir, head)
    bundle = baseline.bundle_dir(rules_dir)
    for r in bundle["attempts"]:
        if r["split"] != "tuning":
            raise split.SplitSealed(
                "the prior learns from the tuning split only (docs/15 §F): row %s "
                "of %s carries split %s" % (r["geometry_id"], rules_dir, r["split"]))
    end_ids = [g["geometry_id"] for g in bundle["geometries"]]
    if sorted(end_ids) != sorted(head["geometry_ids"]) or \
            len(set(end_ids)) != len(end_ids):
        raise PriorError("incomplete: the rules campaign's end records do not match "
                         "its header (%d ends, %d header ids)"
                         % (len(end_ids), len(head["geometry_ids"])))
    bad = [g["geometry_id"] for g in bundle["geometries"]
           if campaign.terminal_of(g) == "HARNESS-ERROR"]
    if bad:
        raise PriorError("%d harness error(s) in the rules campaign: %s"
                         % (len(bad), ", ".join(sorted(bad))))
    model = build(bundle)
    gates = schema.load_gates()
    knobs = schema.load_knobs()
    decisions = decide_all(bundle, model, gates, knobs)
    measured = {}
    for r in bundle["attempts"]:
        measured.setdefault(r["geometry_id"], {})[r["config_sha"]] = r["attempt"]
    return head, bundle, model, gates, knobs, decisions, measured


def gate(rules_dir, work, *, streams=6, report_dir=REPORT_DIR, binary=None,
         attempt_fn=None, probe_fn=None, snap_fn=None, write=True, quiet=False):
    """G-PRIOR: the gate on a finished rules campaign of the tuning split."""
    head, bundle, model, gates, knobs, decisions, measured = _prepare(rules_dir)
    rounds = plan_rounds(decisions, measured)
    round_rows, rounds_of = {}, {}
    round_infos = []
    for j, rnd in enumerate(rounds, start=1):
        out = _run_round(j, rnd, work, streams=streams, binary=binary,
                         attempt_fn=attempt_fn, probe_fn=probe_fn, snap_fn=snap_fn,
                         quiet=quiet)
        rb = baseline.bundle_dir(out)
        round_rows[j] = {}
        for r in rb["attempts"]:
            if r["attempt"] == 1:
                round_rows[j][r["geometry_id"]] = r
        for gid, item in rnd.items():
            rounds_of[(gid, item["config_sha256"])] = j
        ends_r = {g["geometry_id"]: g for g in rb["geometries"]}
        for gid in sorted(rnd):
            if campaign.terminal_of(ends_r[gid]) == "HARNESS-ERROR":
                raise PriorError("round %d ends %s in a harness error"
                                 % (j, gid))
            if gid not in round_rows[j]:
                raise PriorError("round %d has no attempt-1 row for %s" % (j, gid))
        round_infos.append({"round": j, "n": len(rnd), "geometry_ids": sorted(rnd),
                            "dir": out})
    return _finish(head, bundle, model, decisions, measured, rounds, round_rows,
                   rounds_of, round_infos, report_dir, write)


def _finish(head, bundle, model, decisions, measured, rounds, round_rows,
            rounds_of, round_infos, report_dir, write):
    """(C8): every variant's status and outcome, then the systems table."""
    n = len(head["geometry_ids"])
    rows_by = {}
    for r in bundle["attempts"]:
        rows_by.setdefault(r["geometry_id"], {})[r["attempt"]] = r
    systems = {}
    for name in ("rules",) + VARIANTS:
        systems[name] = {"pass": 0, "strict_pass": 0, "statuses": {},
                         "decisions": {} if name in VARIANTS else None,
                         "pass_gids": set()}
    geom_out, fams = [], {}
    for d in decisions:
        gid = d["geometry_id"]
        rows = rows_by.get(gid, {})       # a SURFACE-OPEN / REFUSED geometry has no rows
        fams.setdefault(d["family"], []).append(gid)
        grow = {"geometry_id": gid, "family": d["family"],
                "eligible": d["eligible"], "terminal": d["terminal"]}
        for name in ("rules",) + VARIANTS:
            entry, vote_d = {}, None
            if name == "rules":
                status = "rules" if d["eligible"] else "ineligible"
                oc = rows[1]["outcome"] if d["eligible"] else None
            elif not d["eligible"]:
                status, oc = "ineligible", None
            else:
                dec = d["variants"][name]
                entry, oc = _outcome_of(dec, gid, rows, measured, round_rows,
                                        rounds_of)
                status = entry.pop("status")
                vote_d = dec["vote"]
                rid = dec["rule_id"]
            ok = oc is not None and oc["failure"] is False
            strict = oc is not None and oc["strict_failure"] is False
            if name == "rules":
                grow["rules"] = {"pass": ok, "strict_pass": strict}
            else:
                grow[name] = dict(
                    {"rule_id": rid, "status": status, "pass": ok,
                     "strict_pass": strict,
                     "neighbours": vote_d["neighbours"] if vote_d else None,
                     "distances": vote_d["distances"] if vote_d else None,
                     "winner": vote_d["winner"] if vote_d else None}, **entry)
            systems[name]["statuses"][status] =                 systems[name]["statuses"].get(status, 0) + 1
            systems[name]["pass"] += int(ok)
            systems[name]["strict_pass"] += int(strict)
            if ok:
                systems[name]["pass_gids"].add(gid)
            if name in VARIANTS and d["eligible"]:
                vrid = d["variants"][name]["rule_id"]
                systems[name]["decisions"][vrid] =                     systems[name]["decisions"].get(vrid, 0) + 1
        geom_out.append(grow)
    return _systems_and_report(head, bundle, model, systems, geom_out, fams, n,
                               rounds, round_infos, report_dir, write)


def _systems_and_report(head, bundle, model, systems, geom_out, fams, n, rounds,
                        round_infos, report_dir, write):
    """The conditions and verdict, the per-family table, the report, the bundles."""
    for name in ("rules",) + VARIANTS:
        s = systems[name]
        s["rate"] = s["pass"] / n
        s["strict_rate"] = s["strict_pass"] / n
        s["gain"] = len(s["pass_gids"] - systems["rules"]["pass_gids"])
        s["loss"] = len(systems["rules"]["pass_gids"] - s["pass_gids"])
    sh = [systems[v]["pass"] for v in VARIANTS if v.startswith("shuffle-")]
    shuffled_mean_pass = sum(sh) / float(len(sh))
    conditions = {"real_ge_rules": systems["real"]["pass"] >= systems["rules"]["pass"],
                  "shuffled_worse": shuffled_mean_pass < systems["real"]["pass"]}
    verdict = "PASS" if all(conditions.values()) else "FAIL"
    enabled = verdict == "PASS"
    rep_fams = {}
    for fam in sorted(fams):
        gids = set(fams[fam])
        row = {"family": fam, "n": len(fams[fam])}
        for name in ("rules",) + VARIANTS:
            row[name] = len(systems[name]["pass_gids"] & gids)
        row["shuffled_mean"] = sum(row[v] for v in VARIANTS
                                   if v.startswith("shuffle-")) / 3.0
        rep_fams[fam] = row
    gids_all = set(head["geometry_ids"])
    row = {"family": "all", "n": n}
    for name in ("rules",) + VARIANTS:
        row[name] = len(systems[name]["pass_gids"] & gids_all)
    row["shuffled_mean"] = sum(row[v] for v in VARIANTS
                               if v.startswith("shuffle-")) / 3.0
    rep_fams["all"] = row
    pool = [g for g in bundle["geometries"] if g.get("fingerprint") is not None]
    null_counts = {}
    for f in CURV + ("inner_thickness_m", "outer_gap_m", "lattice_base_size_m"):
        null_counts[f] = sum(1 for g in pool if g["fingerprint"].get(f) is None)
    shipped = dict(model)
    shipped["enabled"] = enabled
    shipped["gate"] = {"verdict": verdict, "rules_pass": systems["rules"]["pass"],
                       "real_pass": systems["real"]["pass"],
                       "shuffled_mean_pass": shuffled_mean_pass, "n": n,
                       "report": REPORT_NAME}
    rep = {"$comment": HEADER, "schema": GATE_SCHEMA,
           "date": time.strftime("%Y-%m-%d"), "verdict": verdict,
           "enabled": enabled, "conditions": conditions, "n_tuning": n,
           "n_eligible": sum(1 for g in geom_out if g["eligible"]),
           "pool_n": model["pool_n"], "bank_n": model["bank_n"], "k": model["k"],
           "d_abstain": model["d_abstain"], "model_sha256": model_sha(model),
           "features": list(FEATURES), "kept": model["kept"],
           "null_policy": list(NULL_POLICY), "excluded": dict(EXCLUDED),
           "null_counts": null_counts,
           "systems": {k: {x: v for x, v in systems[k].items()
                           if x != "pass_gids"} for k in systems},
           "shuffled_mean_pass": shuffled_mean_pass,
           "shuffled_mean_rate": shuffled_mean_pass / n,
           "per_family": rep_fams, "rounds": [], "rules_campaign": {},
           "geometries": geom_out, "departures": list(DEPARTURES)}
    return _report_and_bundles(head, bundle, shipped, rep, round_infos,
                               report_dir, write)


def _report_and_bundles(head, bundle, shipped, rep, round_infos, report_dir,
                        write):
    """The bundles, the report JSON and .md, the shipped model."""
    rep["rules_campaign"] = {
        "campaign_id": head["campaign_id"],
        "binary_sha256": head["binary_sha256"], "git_sha": head["git_sha"],
        "n_geometries": len(head["geometry_ids"]),
        "n_rows": len(bundle["attempts"]),
        "bundle": _bundle_info(RULES_BUNDLE, bundle, report_dir, write)}
    for info in round_infos:
        b = baseline.bundle_dir(info["dir"])
        bi = _bundle_info(ROUND_BUNDLE % info["round"], b, report_dir, write)
        rep["rounds"].append({"round": info["round"], "n": info["n"],
                              "geometry_ids": info["geometry_ids"],
                              "bundle": bi})
    if write:
        _dump(os.path.join(report_dir, MODEL_NAME), shipped)
        _dump(os.path.join(report_dir, REPORT_NAME), rep)
        _write_text(os.path.join(report_dir, REPORT_MD), report_md(rep))
    return rep


def _write_text(path, text):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _f3(x):
    return "%.3f" % x


def report_md(rep):
    """The G-PRIOR report as deterministic markdown."""
    a = []
    a.append("<!-- %s -->" % HEADER)
    a.append("")
    a.append("# G-PRIOR - the L3 prior on the tuning split (docs/15 §F)")
    a.append("")
    a.append("- date: %s" % rep["date"])
    a.append("- verdict: %s" % rep["verdict"])
    false = [k for k, v in rep["conditions"].items() if not v]
    if rep["enabled"]:
        a.append("- The prior ships enabled.")
    else:
        a.append("- The prior ships DISABLED: %s." % ", ".join(false))
    a.append("- model sha256: %s" % rep["model_sha256"])
    rc = rep["rules_campaign"]
    a.append("- rules campaign: %s (binary sha256 %s, %d geometries, %d rows)"
             % (rc["campaign_id"], rc["binary_sha256"], rc["n_geometries"],
                rc["n_rows"]))
    a.append("")
    a.append("## Attempt-1 passes")
    a.append("")
    a.append("| system | passes | rate | strict passes | gain vs rules | loss vs rules |")
    a.append("|---|---|---|---|---|---|")
    for name in ("rules",) + VARIANTS:
        s = rep["systems"][name]
        a.append("| %s | %d | %s | %d | %d | %d |"
                 % (name, s["pass"], _f3(s["rate"]), s["strict_pass"],
                    s["gain"], s["loss"]))
    sm = rep["shuffled_mean_pass"]
    a.append("| shuffled mean | %s | %s | | | |"
             % (_f3(sm), _f3(rep["shuffled_mean_rate"])))
    a.append("")
    a.append("## Decisions")
    a.append("")
    a.append("| variant | PR-KNN | PR-KEEP | PR-FAR | PR-NOEDIT | reused | ran | refused |")
    a.append("|---|---|---|---|---|---|---|---|")
    for name in VARIANTS:
        s = rep["systems"][name]
        dec, st = s["decisions"], s["statuses"]
        a.append("| %s | %d | %d | %d | %d | %d | %d | %d |"
                 % (name, dec.get("PR-KNN", 0), dec.get("PR-KEEP", 0),
                    dec.get("PR-FAR", 0), dec.get("PR-NOEDIT", 0),
                    st.get("reused", 0), st.get("ran", 0), st.get("refused", 0)))
    a.append("")
    a.append("## Per family")
    a.append("")
    a.append("| family | n | rules | real | shuffle-0 | shuffle-1 | shuffle-2 | shuffled mean |")
    a.append("|---|---|---|---|---|---|---|---|")
    for fam, row in rep["per_family"].items():
        a.append("| %s | %d | %d | %d | %d | %d | %d | %s |"
                 % (row["family"], row["n"], row["rules"], row["real"],
                    row["shuffle-0"], row["shuffle-1"], row["shuffle-2"],
                    _f3(row["shuffled_mean"])))
    a.append("")
    a.append("## Null policy")
    a.append("")
    for np_ in rep["null_policy"]:
        a.append("- %s -> %s (indicator %s), fill %s, clip [%s, %s]: %s"
                 % (", ".join(np_["fields"]), ", ".join(np_["features"]),
                    np_["indicator"], explain.fmt(float(np_["fill"])),
                    explain.fmt(float(np_["clip"][0])),
                    explain.fmt(float(np_["clip"][1])), np_["why"]))
    a.append("- excluded: %s." % "; ".join("%s (%s)" % (k, v)
                                           for k, v in rep["excluded"].items()))
    a.append("- null counts over the pool: %s."
             % "; ".join("%s %d" % (k, v) for k, v in rep["null_counts"].items()))
    a.append("")
    a.append("## Rounds")
    a.append("")
    if not rep["rounds"]:
        a.append("- none: every apply decision reused a config the rules campaign "
                 "had already measured.")
    for r in rep["rounds"]:
        a.append("- round %d: %d geometries (%s), bundle %s sha256 %s"
                 % (r["round"], r["n"], ", ".join(r["geometry_ids"]),
                    r["bundle"]["file"], (r["bundle"]["sha256"] or "-")[:12]))
    a.append("")
    a.append("## Departures")
    a.append("")
    for d in rep["departures"]:
        a.append("- %s" % d)
    a.append("")
    return "\n".join(a) + "\n"


def _read_json_or_none(path):
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def check(report_dir=REPORT_DIR):
    """The committed files, rebuilt: (C11). Never raises on a missing file."""
    items = []

    def add(name, ok, why):
        items.append({"name": name, "ok": bool(ok), "why": why})

    rp = os.path.join(report_dir, REPORT_NAME)
    mp = os.path.join(report_dir, MODEL_NAME)
    rep = _read_json_or_none(rp)
    add("G-PRIOR.json", rep is not None, rp)
    pm = _read_json_or_none(mp)
    add("prior_model.json", pm is not None, mp)
    bpath = os.path.join(report_dir, RULES_BUNDLE)
    if rep is None or not os.path.isfile(bpath):
        add("rules bundle", False, bpath)
    else:
        with open(bpath, "rb") as f:
            data = f.read()
        want = rep.get("rules_campaign", {}).get("bundle", {}).get("sha256")
        add("rules bundle", hashlib.sha256(data).hexdigest() == want, bpath)
    ok_all, why_all = True, "%s (no rounds)" % os.path.join(report_dir, "eval_r*.json.gz")
    if rep is not None:
        checked = []
        for r in rep.get("rounds") or []:
            fname = r["bundle"]["file"]
            fpath = os.path.join(report_dir, fname)
            if not os.path.isfile(fpath):
                ok_all, why_all = False, fpath
                break
            with open(fpath, "rb") as f:
                h = hashlib.sha256(f.read()).hexdigest()
            if h != r["bundle"]["sha256"]:
                ok_all, why_all = False, "%s holds %s" % (fpath, h[:12])
                break
            checked.append(fname)
        if ok_all and checked:
            why_all = "%s: %s" % (report_dir, ", ".join(checked))
        if not (rep.get("rounds") or []):
            why_all = "%s (no rounds)" % os.path.join(report_dir, "eval_r*.json.gz")
    add("round bundles", ok_all and rep is not None,
        why_all if rep is not None else rp)
    return _check_models(rep, pm, bpath, mp, items, report_dir, rp)


def _check_models(rep, pm, bpath, mp, items, report_dir, rp_path):
    """check's model rebuild, model sha and enabled items; the verdict out."""
    def add(name, ok, why):
        items.append({"name": name, "ok": bool(ok), "why": why})

    msha = model_sha(pm) if isinstance(pm, dict) and pm.get("schema") == MODEL_SCHEMA else None
    if pm is None or rep is None or not os.path.isfile(bpath) or msha is None:
        add("model rebuild", False, bpath if os.path.isfile(bpath) or pm is None
            else mp)
        add("model sha", False, mp)
    else:
        why_r = bpath
        try:
            rebuilt = model_sha(build(baseline.read_bundle(bpath)))
        except (PriorError, baseline.BaselineError, ValueError, KeyError) as e:
            rebuilt, why_r = None, "%s: %s" % (bpath, e)
        add("model rebuild", rebuilt is not None and rebuilt == msha,
            msha if rebuilt is not None else why_r)
        add("model sha", rep.get("model_sha256") == msha, mp)
    if pm is None or rep is None:
        add("enabled", False, mp)
    else:
        add("enabled",
            pm.get("enabled") == (rep.get("verdict") == "PASS")
            and isinstance(pm.get("gate"), dict)
            and pm["gate"].get("verdict") == rep.get("verdict"),
            "%s vs %s" % (mp, rp_path))
    return {"schema": CHECK_SCHEMA, "items": items,
            "verdict": "PASS" if all(i["ok"] for i in items) else "FAIL"}


class _ArgParser(argparse.ArgumentParser):
    """Bad arguments exit 2 with `prior: <message>` (campaign.py's pattern)."""

    def error(self, message):
        self.exit(2, "prior: %s\n" % message)


def _counts_line(variant, decisions, measured):
    dec = {"PR-KNN": 0, "PR-KEEP": 0, "PR-FAR": 0, "PR-NOEDIT": 0}
    m = w = 0
    for d in decisions:
        if not d["eligible"]:
            continue
        v = d["variants"][variant]
        dec[v["rule_id"]] = dec.get(v["rule_id"], 0) + 1
        if v["verdict"] == "apply":
            gid = d["geometry_id"]
            if schema.canonical_sha256(v["config"]) in measured[gid]:
                m += 1
            else:
                w += 1
    return ("[prior] %s: PR-KNN %d, PR-KEEP %d, PR-FAR %d, PR-NOEDIT %d, "
            "measured %d, new %d"
            % (variant, dec["PR-KNN"], dec["PR-KEEP"], dec["PR-FAR"],
               dec["PR-NOEDIT"], m, w))


def _plan_lines(model, decisions, measured, rounds):
    lines = ["[prior] model: pool %d, %d features kept, bank %d, d_abstain %s"
             % (model["pool_n"], len(model["kept"]), model["bank_n"],
                "%.6f" % model["d_abstain"])]
    for variant in VARIANTS:
        lines.append(_counts_line(variant, decisions, measured))
    lines.append("[prior] rounds %d: %s geometries"
                 % (len(rounds), ", ".join(str(len(r)) for r in rounds)
                    if rounds else "0"))
    return lines


def _cli_check(a):
    ck = check(a.report_dir)
    for item in ck["items"]:
        print("[check] %s %s %s" % (item["name"], "ok" if item["ok"] else "FAIL",
                                    item["why"]))
    print("CHECK %s" % ck["verdict"])
    return 0 if ck["verdict"] == "PASS" else 1


def _cli_run(a):
    if not a.rules or (a.gate and not a.work):
        sys.stderr.write("prior: --gate needs --rules and --work\n" if a.gate
                         else "prior: --plan needs --rules\n")
        return 2
    if not 1 <= a.streams <= campaign.MAX_STREAMS:
        sys.stderr.write("prior: --streams %r outside 1..%d\n"
                         % (a.streams, campaign.MAX_STREAMS))
        return 2
    head, bundle, model, gates, knobs, decisions, measured = _prepare(a.rules)
    rounds = plan_rounds(decisions, measured)
    for line in _plan_lines(model, decisions, measured, rounds):
        print(line)
    if a.plan:
        return 0
    rep = gate(a.rules, a.work, streams=a.streams, report_dir=a.report_dir,
               binary=a.binary, quiet=False)
    s = rep["systems"]
    print("G-PRIOR %s: attempt-1 passes rules %d/%d, real %d/%d, shuffled %d, %d, "
          "%d (mean %.3f); the prior ships %s"
          % (rep["verdict"], s["rules"]["pass"], rep["n_tuning"],
             s["real"]["pass"], rep["n_tuning"], s["shuffle-0"]["pass"],
             s["shuffle-1"]["pass"], s["shuffle-2"]["pass"],
             rep["shuffled_mean_pass"], "enabled" if rep["enabled"] else "DISABLED"))
    return 0


def main(argv=None):
    ap = _ArgParser(prog="prior.py",
                    description="the L3 prior of docs/15 §C (k-NN warm start, G-PRIOR)")
    ap.add_argument("--selftest", action="store_true", help="the G-PRIOR selftest")
    ap.add_argument("--plan", action="store_true",
                    help="the decisions and rounds of a rules campaign, nothing run")
    ap.add_argument("--gate", action="store_true",
                    help="run G-PRIOR: the report, the bundles and the model")
    ap.add_argument("--check", action="store_true",
                    help="the committed report, bundles and model rebuild")
    ap.add_argument("--rules", help="the finished rules campaign directory")
    ap.add_argument("--work", help="the evaluation rounds' work directory")
    ap.add_argument("--streams", type=int, default=6)
    ap.add_argument("--binary", default=None)
    ap.add_argument("--report-dir", dest="report_dir", default=REPORT_DIR)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    try:
        if a.check:
            return _cli_check(a)
        if a.plan or a.gate:
            return _cli_run(a)
    except (PriorError, campaign.CampaignError, baseline.BaselineError,
            split.SplitSealed, split.SplitError) as e:
        sys.stderr.write("prior: %s\n" % e)
        return 2
    ap.error("one of --selftest, --plan, --gate, --check is required")



IDS8 = ("D-1-010", "G-1-016", "F-1-009", "D-1-077", "F-1-011", "A-1-000",
        "E-1-004", "E-1-010")


def _oracle_attempt(kind_of):
    def attempt(c, gctx, config, a, n_leaves):
        kind = kind_of(config, gctx)
        t_start = schema._now_iso()
        oc = remedies.synthetic_outcome(kind, config, gctx["fp"], gctx["flow"],
                                        c.gates)
        t_end = schema._now_iso()
        content = None
        if oc["exit_code"] == 0:
            content = hashlib.sha256(
                (schema.canonical_sha256(config) + kind).encode("ascii")).hexdigest()
        return {"outcome": oc, "content_sha256": content, "t_start": t_start,
                "t_end": t_end}
    return attempt


_L1_WALL = {}


def _l1_wall(gctx):
    """The setup rules' wall level for this geometry (the oracles' reference), cached."""
    gid = gctx["gid"]
    if gid not in _L1_WALL:
        s = rules.setup(gctx["mrow"], gctx["fp"], campaign.stl_rel(gid),
                        campaign.case_rel(gid), gid)
        _L1_WALL[gid] = None if s["refused"] else remedies.wall_level(s["config"])
    return _L1_WALL[gid]


def _wall_kind(config, gctx):
    """The selftests' fake mesher since 2026-09-26 (feature_tolerance 0 is no longer a
    lever on a body with sharp edges): the R-PLANE path passes; any other config passes
    one wall level or more below the setup rules' and fails F3a otherwise (docs/15 §K
    G-PILOT: wall level -1 was the only F3-clean knob with the attraction on)."""
    if remedies.on_plane(config, gctx["fp"]):
        return "pass"
    w = _l1_wall(gctx)
    return "pass" if w is not None and remedies.wall_level(config) < w else "F3a"


def _mixed_kind(config, gctx):
    """A second fake mesher for the round mechanics: families A, B and G pass one wall
    level below the setup rules' (F3a otherwise), D, E and F one level above (F3d
    otherwise); the R-PLANE path passes."""
    if remedies.on_plane(config, gctx["fp"]):
        return "pass"
    w = _l1_wall(gctx)
    if w is None:
        return "F3a"
    d = remedies.wall_level(config) - w
    if gctx["gid"][0] in "ABG":
        return "pass" if d < 0 else "F3a"
    return "pass" if d > 0 else "F3d"


_wall_oracle = _oracle_attempt(_wall_kind)
_mixed_oracle = _oracle_attempt(_mixed_kind)
_pass_oracle = _oracle_attempt(lambda config, gctx: "pass")


def _run_oracle(out, attempt_fn, ids=IDS8):
    return campaign.run_campaign(
        {"manifest": "tuning", "ids": list(ids), "out": out, "mode": "rules",
         "streams": 2, "quiet": True},
        attempt_fn=attempt_fn, probe_fn=campaign._fake_probe(1000),
        snap_fn=campaign._fake_snap)


def _setup_of(H, gid):
    """The L1 config and the (C7)-shaped ctx of one geometry, from rules.setup."""
    end, mrow = H["ends"][gid], H["mrows"][gid]
    s = rules.setup(mrow, end["fingerprint"], campaign.stl_rel(gid),
                    campaign.case_rel(gid), gid, gates=H["gates"],
                    knobs=H["knobs"])
    assert not s["refused"], (gid, s["refused"])
    ctx = {"geometry_id": gid, "fingerprint": end["fingerprint"],
           "flow": mrow["flow"], "win_level": s["summary"]["win_level"],
           "config": s["config"], "rules": s}
    return s["config"], ctx


def _g1_constants(H):
    assert len(FEATURES) == 17 and K == 3 and SHUFFLES == 3 and EPS == 1e-6
    assert TRANSFER == ("RM-BUDGET-FAR", "RM-BUDGET-FEAT", "RM-BUDGET-WALL",
                        "RM-TOPO-REFINE", "RM-SNAP-WALL", "RM-SNAP-FT",
                        "RM-SNAP-REFINE"), TRANSFER
    for rid in PR_IDS:
        assert rid in explain.TEMPLATES and \
            explain.TEMPLATES[rid]["layer"] == "prior", rid
    d = os.path.join(H["tmp"], "g1-eval")
    os.makedirs(d)
    _dump(os.path.join(d, campaign.FILES["campaign"]),
          {"split_mode": split.EVALUATE, "manifest": {"source": "test"}})
    try:
        gate(d, os.path.join(H["tmp"], "g1-work"))
    except split.SplitSealed:
        pass
    else:
        raise AssertionError("an evaluate campaign was not refused by the seal")
    d2 = os.path.join(H["tmp"], "g1-b0")
    os.makedirs(d2)
    with open(os.path.join(H["R1"], campaign.FILES["campaign"]),
              encoding="utf-8") as f:
        head = json.load(f)
    head["mode"] = "b0-template"
    _dump(os.path.join(d2, campaign.FILES["campaign"]), head)
    try:
        gate(d2, os.path.join(H["tmp"], "g1-work"))
    except PriorError as e:
        assert "rules campaign" in str(e), str(e)
    else:
        raise AssertionError("a b0-template campaign was not refused")
    try:
        load_model(os.path.join(H["tmp"], "absent.json"))
    except PriorError as e:
        assert "no prior model" in str(e), str(e)
    else:
        raise AssertionError("a missing model was not refused")
    print("[ok] constants: 17 features, k 3, 3 shuffles, 7 transferable remedies, "
          "5 PR ids templated; an evaluate campaign is refused before any other "
          "file, a b0-template campaign and a missing model are refused by name")


_WANT_FEATURES = {
    "D-1-010": [0.173507, 0.387097, 0.354839, 0.245163, 1.0, 0.901335, 2.0, 2.0,
                2.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0],
    "E-1-004": [-0.234647, 0.515297, 0.515297, 0.152886, 0.428823, 0.0, -0.672432,
                -0.595262, -0.541533, 0.0, 0.0, 1.0, -1.291485, 0.0, 0.0, 0.0,
                3.899602],
}


def _g2_features(H):
    for gid, want in _WANT_FEATURES.items():
        got = features(H["ends"][gid]["fingerprint"])
        assert len(got) == 17
        for a, b in zip(got, want):
            assert abs(a - b) <= 1e-6, (gid, a, b)
    fp = copy.deepcopy(H["ends"]["D-1-010"]["fingerprint"])
    for f in CURV:
        fp[f] = 1e-9
    f9 = features(fp)
    assert f9[6:9] == [-3.0, -3.0, -3.0] and f9[9] == 0.0, f9[6:10]
    fp2 = copy.deepcopy(H["ends"]["D-1-010"]["fingerprint"])
    fp2["inner_thickness_m"] = 1e6
    f10 = features(fp2)
    assert f10[10] == 0.0 and f10[11] == 0.0, f10[10:12]
    fp3 = copy.deepcopy(H["ends"]["D-1-010"]["fingerprint"])
    fp3["area_m2"] = 0
    try:
        features(fp3)
    except PriorError as e:
        assert "invalid" in str(e), str(e)
    else:
        raise AssertionError("an invalid fingerprint was not refused")
    for gid in IDS8:
        assert all(math.isfinite(v) for v in features(H["ends"][gid]["fingerprint"]))
    print("[ok] features: D-1-010 and E-1-004 reproduce to 1e-6; a clipped radius "
          "-3, a clipped thickness 0, an invalid fingerprint refused")


def _g3_scaler(H):
    sc = fit_scaler([[0, 0, 5], [1, 0, 5], [0, 2, 5], [1, 2, 5]])
    assert numpy.allclose(sc["mean"], [0.5, 1, 5]), sc["mean"]
    assert numpy.allclose(sc["std"], [0.5, 1, 0]), sc["std"]
    assert sc["kept"] == [True, True, False]
    rows = [standardise(list(x), sc).tolist()
            for x in ([0, 0, 5], [1, 0, 5], [0, 2, 5], [1, 2, 5])]
    assert numpy.allclose(rows, [[-1, -1], [1, -1], [-1, 1], [1, 1]]), rows
    assert abs(rms_distance([-1, -1], [1, -1]) - math.sqrt(2)) <= 1e-12
    assert abs(d_abstain(rows) - math.sqrt(2)) <= 1e-12
    try:
        fit_scaler([[1.0, 2.0]])
    except PriorError as e:
        assert "pool" in str(e), str(e)
    else:
        raise AssertionError("a one-row pool was not refused")
    try:
        d_abstain([[1.0, 2.0]])
    except PriorError as e:
        assert "pool" in str(e), str(e)
    else:
        raise AssertionError("a one-row d_abstain was not refused")
    m = H["model"]
    assert len(m["kept"]) == 17, m["kept"]
    assert abs(m["d_abstain"] - 1.181871658) <= 1e-6, m["d_abstain"]
    assert m["pool_n"] == 8
    print("[ok] scaler: std 0 drops a feature, Z of the unit square, RMS sqrt 2, "
          "d_abstain sqrt 2; the oracle pool keeps 17 with d_abstain 1.181872")


def _g4_vote(H):
    def bank(*specs):
        return [{"geometry_id": g, "z": list(z), "path": list(p)}
                for g, z, p in specs]
    a = ("a", [0.1, 0], ["RM-SNAP-FT"])
    b = ("b", [0.15, 0], ["RM-SNAP-WALL"])
    c = ("c", [0.2, 0], ["RM-SNAP-WALL"])
    e = ("d", [5, 5], [])
    v = vote([0, 0], bank(a, b, c, e), 1.0)
    assert v["kind"] == "vote" and v["neighbours"] == ["a", "b", "c"], v
    for got, want in zip(v["distances"], [0.0707107, 0.1060660, 0.1414214]):
        assert abs(got - want) <= 1e-6, (got, want)
    assert v["winner"] == ["RM-SNAP-WALL"], v["winner"]
    v = vote([0, 0], bank(a, ("b", [-0.1, 0], ["RM-SNAP-WALL"]),
                          ("c", [0, 0.1], [])), 1.0)
    assert v["kind"] == "vote" and len(set(v["distances"])) == 1
    assert v["winner"] == ["RM-SNAP-FT"], v["winner"]
    v = vote([0, 0], bank(("a", [0.1, 0], []), ("b", [0.15, 0], []),
                          ("c", [0.2, 0], ["RM-SNAP-FT"])), 1.0)
    assert v["kind"] == "keep" and v["winner"] == [], v
    v = vote([3, 3], bank(a, b, c), 1.0)
    assert v["kind"] == "far" and v["neighbours"] == ["c", "b", "a"], v
    assert abs(v["nearest"] - 2.9017236) <= 1e-6, v["nearest"]
    assert v["winner"] is None and v["weights"] == []
    v = vote([0, 0], bank(a, b), 1.0)
    assert v["kind"] == "few" and v["neighbours"] == ["a", "b"]
    assert v["winner"] is None and v["weights"] == []
    print("[ok] vote: a 2-1 weighted vote, a three-way tie to the nearest, keep, "
          "far and few")


_WANT_BANK = {
    "A-1-000": (2, ["RM-SNAP-WALL"]),
    "D-1-010": (1, []),
    "D-1-077": (2, ["RM-SNAP-WALL"]),
    "E-1-010": (2, ["RM-SNAP-WALL"]),
    "F-1-011": (2, ["RM-SNAP-WALL"]),
    "G-1-016": (1, []),
}


def _g5_bank(H):
    got = {e["geometry_id"]: (e["attempt"], e["path"])
           for e in bank_from(H["B1"])}
    assert got == _WANT_BANK, got
    m = H["model"]
    assert m["bank_n"] == 6 and all(len(e["z"]) == 17 for e in m["bank"])
    assert m["null_policy"] == list(NULL_POLICY)
    assert m["source"]["content_sha256"] == bundle_sha(H["B1"])
    assert m["enabled"] is False and m["gate"] is None
    b2 = copy.deepcopy(H["B1"])
    # F-1-009 left the bank (its only passing attempt needed the refused lever),
    # so the gap test cuts a banked geometry's middle attempt instead
    b2["attempts"] = [r for r in b2["attempts"]
                      if not (r["geometry_id"] == "A-1-000" and r["attempt"] == 1)]
    try:
        bank_from(b2)
    except PriorError as e:
        assert "incomplete" in str(e), str(e)
    else:
        raise AssertionError("a bundle with a gap in the attempts was not refused")
    print("[ok] bank: 6 passing geometries with their remedy paths, E-1-004 and "
          "F-1-009 absent; the model carries the null policy and the source sha")


def _g6_apply(H):
    cfg, ctx = H["cfg"], H["ctx"]
    after, skipped = apply_path(cfg["F-1-009"], ["RM-SNAP-WALL", "RM-SNAP-FT"],
                                ctx["F-1-009"], H["gates"], H["knobs"])
    assert after is None
    assert [x["rule_id"] for x in skipped] == ["RM-SNAP-WALL", "RM-SNAP-FT"]
    assert "below the y+ floor" in skipped[0]["why"], skipped[0]
    assert skipped[1]["why"] == remedies.FT_FORBIDDEN_WHY, skipped[1]
    after2, skipped2 = apply_path(cfg["E-1-004"], ["RM-SNAP-WALL", "RM-SNAP-FT"],
                                  ctx["E-1-004"], H["gates"], H["knobs"])
    assert after2 is None
    assert [x["rule_id"] for x in skipped2] == ["RM-SNAP-WALL", "RM-SNAP-FT"]
    assert skipped2[1] == {"rule_id": "RM-SNAP-FT",
                           "why": "no sharp edge (features.py)"}, skipped2[1]
    after3, skipped3 = apply_path(cfg["A-1-000"], ["RM-SNAP-WALL", "RM-SNAP-FT"],
                                  ctx["A-1-000"], H["gates"], H["knobs"])
    assert schema.canonical_sha256(after3) == H["att2_sha"]["A-1-000"]
    assert skipped3 == [{"rule_id": "RM-SNAP-FT",
                         "why": remedies.FT_FORBIDDEN_WHY}], skipped3
    after4, skipped4 = apply_path(cfg["A-1-000"], ["RM-SNAP-FT"], ctx["A-1-000"],
                                  H["gates"], H["knobs"])
    assert after4 is None and skipped4 == skipped3, (after4, skipped4)
    after5, skipped5 = apply_path(cfg["D-1-010"], ["RM-SNAP-FT"], ctx["D-1-010"],
                                  H["gates"], H["knobs"])
    assert after5 is None
    assert skipped5 == [{"rule_id": None, "why": PLANE_WHY}], skipped5
    model = H["model"]
    fold = [e for e in model["bank"] if e["geometry_id"] != "A-1-000"]
    zq = standardise(features(ctx["A-1-000"]["fingerprint"]), model["scaler"])
    d = decide(zq, fold, model, ctx["A-1-000"], H["gates"], H["knobs"])
    assert d["rule_id"] == "PR-KNN" and d["verdict"] == "apply", d["rule_id"]
    assert schema.errors(d["record"], "DecisionRecord") == []
    edits = rules.diff_edits(cfg["A-1-000"], after3)
    for es in (edits, d["edits"]):
        for x in es:
            assert not x["pointer"].startswith("/layers")
    try:
        apply_path(cfg["A-1-000"], ["RM-PLANE"], ctx["A-1-000"], H["gates"],
                   H["knobs"])
    except PriorError:
        pass
    else:
        raise AssertionError("a path outside the transfer table was not refused")
    print("[ok] apply: F-1-009's wall step is skipped at the y+ floor and RM-SNAP-FT "
          "by name, so its path changes nothing; E-1-004 and D-1-010 change nothing; "
          "A-1-000's path lands on its attempt-2 config with RM-SNAP-FT skipped by "
          "name; no edit in the layers block")


def _g7_gate(H):
    work = os.path.join(H["tmp"], "work")
    rdir = os.path.join(H["tmp"], "rep")
    rep = gate(H["R1"], work, streams=2, report_dir=rdir, attempt_fn=_wall_oracle,
               probe_fn=campaign._fake_probe(1000), snap_fn=campaign._fake_snap,
               quiet=True)
    s = rep["systems"]
    assert s["rules"]["pass"] == 2 and s["real"]["pass"] == 6, \
        (s["rules"]["pass"], s["real"]["pass"])
    assert s["shuffle-0"]["pass"] == 4 and s["shuffle-1"]["pass"] == 5 \
        and s["shuffle-2"]["pass"] == 4
    assert abs(rep["shuffled_mean_pass"] - 13.0 / 3.0) <= 1e-9
    assert s["real"]["decisions"] == {"PR-KNN": 4, "PR-KEEP": 2, "PR-NOEDIT": 1,
                                      "PR-FAR": 1}, s["real"]["decisions"]
    assert s["real"]["statuses"] == {"reused": 4, "PR-KEEP": 2, "PR-NOEDIT": 1,
                                     "PR-FAR": 1}, s["real"]["statuses"]
    assert s["real"]["gain"] == 4 and s["real"]["loss"] == 0
    assert rep["rounds"] == [], rep["rounds"]
    assert rep["conditions"] == {"real_ge_rules": True, "shuffled_worse": True}
    assert rep["verdict"] == "PASS" and rep["enabled"] is True
    for name in (REPORT_NAME, REPORT_MD, MODEL_NAME, RULES_BUNDLE):
        assert os.path.isfile(os.path.join(rdir, name)), name
    with open(os.path.join(rdir, MODEL_NAME), encoding="utf-8") as f:
        pm = json.load(f)
    assert pm["enabled"] is True and pm["bank_n"] == 6 and pm["pool_n"] == 8
    grow = {g["geometry_id"]: g for g in rep["geometries"]}
    assert grow["F-1-009"]["real"]["rule_id"] == "PR-FAR"
    H["shipped"] = pm
    rep2 = gate(H["R1"], work, streams=2, report_dir=rdir, attempt_fn=_wall_oracle,
                probe_fn=campaign._fake_probe(1000), snap_fn=campaign._fake_snap,
                quiet=True)
    r1 = dict(rep)
    r2 = dict(rep2)
    r1.pop("date")
    r2.pop("date")
    assert r1 == r2, "the re-run's report differs beyond the date"
    # the mixed oracle: a round runs, the control is not worse, the prior ships disabled
    mwork = os.path.join(H["tmp"], "work_mixed")
    mrdir = os.path.join(H["tmp"], "rep_mixed")
    mrules = os.path.join(H["tmp"], "rules_mixed")
    _run_oracle(mrules, _mixed_oracle)
    mrep = gate(mrules, mwork, streams=2, report_dir=mrdir, attempt_fn=_mixed_oracle,
                probe_fn=campaign._fake_probe(1000), snap_fn=campaign._fake_snap,
                quiet=True)
    ms = mrep["systems"]
    assert ms["real"]["pass"] == 2, ms["real"]["pass"]
    assert ms["real"]["statuses"].get("ran") == 1, ms["real"]["statuses"]
    assert len(mrep["rounds"]) == 1 and mrep["rounds"][0]["geometry_ids"] == \
        ["E-1-010", "F-1-011"], mrep["rounds"]
    assert ms["shuffle-0"]["pass"] == 3 and ms["shuffle-1"]["pass"] == 2 \
        and ms["shuffle-2"]["pass"] == 3
    assert abs(mrep["shuffled_mean_pass"] - 8.0 / 3.0) <= 1e-9
    assert mrep["conditions"] == {"real_ge_rules": True, "shuffled_worse": False}
    assert mrep["verdict"] == "FAIL" and mrep["enabled"] is False
    assert os.path.isfile(os.path.join(mrdir, ROUND_BUNDLE % 1)), "round bundle"
    cj = os.path.join(mwork, "round_1", campaign.FILES["campaign"])
    mtime = os.stat(cj).st_mtime
    mrep2 = gate(mrules, mwork, streams=2, report_dir=mrdir,
                 attempt_fn=_mixed_oracle, probe_fn=campaign._fake_probe(1000),
                 snap_fn=campaign._fake_snap, quiet=True)
    assert os.stat(cj).st_mtime == mtime, "the mixed round was not reused"
    m1 = dict(mrep)
    m2 = dict(mrep2)
    m1.pop("date")
    m2.pop("date")
    assert m1 == m2, "the mixed re-run's report differs beyond the date"
    print("[ok] gate on the oracle campaign: rules 2, real 6, shuffled 4 5 4 (mean "
          "4.333), no round (every prior config already meshed), PASS and enabled; "
          "on the mixed oracle one round of two geometries (E-1-010, F-1-011) runs, "
          "the shuffled control is not worse so it FAILs disabled, and a re-run "
          "reuses the round and gives an equal report")


def _g8_disabled(H):
    r2 = os.path.join(H["tmp"], "rules_pass")
    _run_oracle(r2, _pass_oracle, ids=IDS8 + ("G-1-026",))   # + one SURFACE-OPEN
    rep = gate(r2, os.path.join(H["tmp"], "work_pass"), streams=2,
               report_dir=os.path.join(H["tmp"], "rep_fail"),
               attempt_fn=_pass_oracle, probe_fn=campaign._fake_probe(1000),
               snap_fn=campaign._fake_snap, quiet=True)
    for name in ("rules",) + VARIANTS:
        assert rep["systems"][name]["pass"] == 8, (name, rep["systems"][name]["pass"])
    assert rep["rounds"] == []
    assert rep["n_tuning"] == 9 and rep["n_eligible"] == 8,         (rep["n_tuning"], rep["n_eligible"])
    g26 = [g for g in rep["geometries"] if g["geometry_id"] == "G-1-026"][0]
    assert g26["eligible"] is False and g26["rules"]["pass"] is False and         all(g26[v]["status"] == "ineligible" and g26[v]["pass"] is False
            for v in VARIANTS), g26
    assert rep["conditions"]["real_ge_rules"] is True
    assert rep["conditions"]["shuffled_worse"] is False
    assert rep["verdict"] == "FAIL" and rep["enabled"] is False
    with open(os.path.join(H["tmp"], "rep_fail", MODEL_NAME),
              encoding="utf-8") as f:
        pm = json.load(f)
    assert pm["enabled"] is False and pm["gate"]["verdict"] == "FAIL"
    assert all(e["path"] == [] for e in pm["bank"])
    with open(os.path.join(H["tmp"], "rep_fail", REPORT_MD), encoding="utf-8") as f:
        md = f.read()
    assert "DISABLED" in md
    print("[ok] a control that is not worse: rules 8, real 8, shuffled 8 8 8 of 9 (a "
          "SURFACE-OPEN geometry fails everywhere), no round, FAIL, the prior ships "
          "disabled")


def _fake_prior_campaign(out, ids, hooks=None):
    return campaign.run_campaign(
        {"manifest": "tuning", "ids": list(ids), "out": out, "mode": "rules+prior",
         "ablate": ("remedies",), "streams": 2, "quiet": True},
        attempt_fn=_wall_oracle, probe_fn=campaign._fake_probe(1000),
        snap_fn=campaign._fake_snap, hooks=hooks)


def _g9_hook(H):
    rdir = os.path.join(H["tmp"], "rep")
    m = load_model(os.path.join(rdir, MODEL_NAME))
    out = os.path.join(H["tmp"], "hook")
    end = _fake_prior_campaign(out, IDS8, hooks={"prior": make_hook(m)})
    rows = campaign.load_rows(out)
    by1 = {r["geometry_id"]: r for r in rows if r["attempt"] == 1}
    prior_gids = sorted(g for g, r in by1.items() if r["decided_by"] == "prior")
    # F-1-009 left the bank (2026-09-26): its only path held the refused lever,
    # so the model no longer fires there
    assert prior_gids == ["A-1-000", "D-1-077", "E-1-010", "F-1-011"], \
        prior_gids
    assert all(by1[g]["rule_id"] == "PR-KNN" for g in prior_gids)
    assert sum(1 for r in by1.values() if r["outcome"]["failure"] is False) == 6, \
        sum(1 for r in by1.values() if r["outcome"]["failure"] is False)
    assert end["harness_errors"] == 0
    rp = campaign.replay(out)
    assert rp["ok"] and rp["hook_decisions"] == 4, rp
    recs = campaign.load_records(out)
    assert explain.audit(rows, recs)["ok"], "audit failed"
    for gid in sorted(recs):
        for tag in recs[gid]:
            rec = tag["record"]
            if rec["layer"] != "prior":
                continue
            card = explain.card(rec)
            assert explain.ungrounded(card["line"], [rec]) == [], \
                (gid, rec["rule_id"], card["line"])
    res = make_hook(m)(H["ctx"]["F-1-009"])
    campaign._check_hook(res, "prior", H["cfg"]["F-1-009"], H["knobs"])
    pm_mod = importlib.import_module("prior")
    old = dict(pm_mod._MODEL)
    try:
        # F-1-009 no longer takes a prior decision (2026-09-26): the model's
        # bank lost it, so the importlib leg runs F-1-011 instead
        ids2 = ["F-1-011", "D-1-010"]
        pm_mod._MODEL["path"] = os.path.join(rdir, MODEL_NAME)
        pm_mod._MODEL["v"] = pm_mod._MODEL["key"] = None
        out2 = os.path.join(H["tmp"], "hook-import")
        _fake_prior_campaign(out2, ids2)
        by2 = {r["geometry_id"]: r
               for r in campaign.load_rows(out2) if r["attempt"] == 1}
        assert by2["F-1-011"]["decided_by"] == "prior" and \
            by2["F-1-011"]["rule_id"] == "PR-KNN", by2["F-1-011"]
        assert by2["D-1-010"]["decided_by"] == "rule"
        recs2 = campaign.load_records(out2)
        assert any(t["attempt"] == 1 and t["record"]["rule_id"] == "PR-KEEP"
                   for t in recs2["D-1-010"])
        dis = copy.deepcopy(m)
        dis["enabled"] = False
        pdis = os.path.join(H["tmp"], "disabled.json")
        _dump(pdis, dis)
        pm_mod._MODEL["path"] = pdis
        pm_mod._MODEL["v"] = pm_mod._MODEL["key"] = None
        out3 = os.path.join(H["tmp"], "hook-disabled")
        _fake_prior_campaign(out3, ids2)
        by3 = {r["geometry_id"]: r
               for r in campaign.load_rows(out3) if r["attempt"] == 1}
        assert all(r["decided_by"] == "rule" for r in by3.values())
        recs3 = campaign.load_records(out3)
        assert sum(1 for gid in recs3 for t in recs3[gid]
                   if t["record"]["rule_id"] == "PR-DISABLED") == 2
        pm_mod._MODEL["path"] = os.path.join(H["tmp"], "absent.json")
        pm_mod._MODEL["v"] = pm_mod._MODEL["key"] = None
        out4 = os.path.join(H["tmp"], "hook-absent")
        end4 = _fake_prior_campaign(out4, ids2)
        assert end4["harness_errors"] == 2, end4["harness_errors"]
        geoms4 = {g["geometry_id"]: g for g in campaign.load_geometries(out4)}
        assert all("no prior model" in (geoms4[g]["reason"] or "") for g in geoms4)
    finally:
        pm_mod._MODEL.clear()
        pm_mod._MODEL.update(old)
    print("[ok] hook seam: make_hook in a fake rules+prior campaign (4 prior rows, "
          "6 attempt-1 passes), attempt1 through importlib, PR-DISABLED on a "
          "disabled model, a missing model ends each geometry HARNESS-ERROR; every "
          "PR card grounded, audit and replay ok")


def _g10_live(H):
    mlive = copy.deepcopy(H["shipped"])
    mlive["d_abstain"] = 1e9
    for e in mlive["bank"]:
        e["path"] = ["RM-SNAP-WALL"]
    mlive["enabled"] = True
    out = os.path.join(H["tmp"], "live")
    end = campaign.run_campaign(
        {"manifest": "tuning", "ids": ["A-1-000", "D-1-010"], "out": out,
         "mode": "rules+prior", "ablate": ("remedies",), "streams": 2,
         "audit_mod": AUDIT_OFF, "quiet": True},
        hooks={"prior": make_hook(mlive)})
    by1 = {r["geometry_id"]: r
           for r in campaign.load_rows(out) if r["attempt"] == 1}
    f = by1["A-1-000"]
    assert f["decided_by"] == "prior" and f["rule_id"] == "PR-KNN", \
        (f["decided_by"], f["rule_id"])
    after, _sk = apply_path(H["cfg"]["A-1-000"], ["RM-SNAP-WALL"],
                            H["ctx"]["A-1-000"], H["gates"], H["knobs"])
    assert f["config_delta"] == rules.diff_edits(H["cfg"]["A-1-000"], after), \
        f["config_delta"]
    assert by1["D-1-010"]["decided_by"] == "rule"
    recs = campaign.load_records(out)
    assert any(t["attempt"] == 1 and t["record"]["rule_id"] == "PR-NOEDIT"
               for t in recs["D-1-010"])
    assert end["harness_errors"] == 0, end["harness_errors"]
    assert end["orphans"] == [] and end["max_live_mesher"] <= 2
    print("[ok] live: rules+prior through the mesher at 2 streams: A-1-000 attempt "
          "1 decided by the prior (RM-SNAP-WALL, exit %s, %s), D-1-010 by "
          "the rules (PR-NOEDIT on the plane); 0 harness errors, 0 orphans, at "
          "most 2 meshers"
          % (f["outcome"]["exit_code"], f["outcome"]["failure_class"]))


def _copy_dir(src, dst):
    shutil.copytree(src, dst)
    return dst


def _child(argv):
    return subprocess.run([sys.executable, os.path.abspath(__file__)] + argv,
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace",
                          env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                          timeout=600)


def _g11_check_cli(H):
    rdir = os.path.join(H["tmp"], "rep")
    ck = check(rdir)
    assert ck["verdict"] == "PASS", ck
    # the flipped byte needs a round bundle: only the mixed gate (rep_mixed)
    # ran a round, the wall gate's rounds list is empty by design
    c1 = _copy_dir(os.path.join(H["tmp"], "rep_mixed"),
                   os.path.join(H["tmp"], "check-flip"))
    p1 = os.path.join(c1, ROUND_BUNDLE % 1)
    data = bytearray(open(p1, "rb").read())
    data[len(data) // 2] ^= 0x01
    with open(p1, "wb") as f:
        f.write(bytes(data))
    ck1 = check(c1)
    it1 = {i["name"]: i for i in ck1["items"]}
    assert it1["round bundles"]["ok"] is False and ck1["verdict"] == "FAIL", \
        (ck1["verdict"], it1)
    c2 = _copy_dir(rdir, os.path.join(H["tmp"], "check-model"))
    p2 = os.path.join(c2, MODEL_NAME)
    with open(p2, encoding="utf-8") as f:
        mm = json.load(f)
    mm["d_abstain"] = mm["d_abstain"] + 0.5
    _dump(p2, mm)
    ck2 = check(c2)
    it2 = {i["name"]: i for i in ck2["items"]}
    assert it2["model rebuild"]["ok"] is False and it2["model sha"]["ok"] is False
    assert ck2["verdict"] == "FAIL"
    p = _child(["--plan", "--rules", H["R1"]])
    # every prior config is already meshed on this oracle, so the plan has no
    # round left to run (2026-09-26)
    assert p.returncode == 0 and "rounds 0" in p.stdout, \
        (p.returncode, p.stdout[-400:], p.stderr[-400:])
    p = _child(["--gate", "--rules", H["R1"]])
    assert p.returncode == 2 and "needs --rules and --work" in (p.stderr or ""), \
        (p.returncode, p.stderr[-400:])
    p = _child(["--check", "--report-dir", rdir])
    assert p.returncode == 0 and "CHECK PASS" in p.stdout, \
        (p.returncode, p.stdout[-400:])
    p = _child(["--check", "--report-dir", os.path.join(H["tmp"], "nothing")])
    assert p.returncode == 1, (p.returncode, p.stdout[-400:])
    print("[ok] check: PASS on the gate's files, a flipped round-bundle byte and "
          "an edited model FAIL by name; the CLI plans the rounds it has, refuses "
          "a gate without --rules, and checks")


def selftest():
    """(C14): eleven [ok] groups, then SELFTEST PASS; the mesher runs in group 10."""
    t0 = time.perf_counter()
    tmp = tempfile.mkdtemp()
    H = {"tmp": tmp}
    groups = ((_g1_constants, "constants"), (_g2_features, "features"),
              (_g3_scaler, "scaler"), (_g4_vote, "vote"), (_g5_bank, "bank"),
              (_g6_apply, "apply"), (_g7_gate, "gate"), (_g8_disabled, "disabled"),
              (_g9_hook, "hook seam"), (_g10_live, "live"),
              (_g11_check_cli, "check and CLI"))
    try:
        try:
            H["gates"] = schema.load_gates()
            H["knobs"] = schema.load_knobs()
            H["R1"] = os.path.join(tmp, "rules")
            _run_oracle(H["R1"], _wall_oracle)
            H["B1"] = baseline.bundle_dir(H["R1"])
            H["model"] = build(H["B1"])
            H["ends"] = {g["geometry_id"]: g for g in H["B1"]["geometries"]}
            H["mrows"] = {r["geometry_id"]: r for r in campaign.load_manifest(
                "tuning", "rules", ids=list(IDS8))}
            H["cfg"], H["ctx"] = {}, {}
            for gid in IDS8:
                cfg, ctx = _setup_of(H, gid)
                H["cfg"][gid], H["ctx"][gid] = cfg, ctx
            rows_by = {}
            for r in H["B1"]["attempts"]:
                rows_by.setdefault(r["geometry_id"], {})[r["attempt"]] = r
            H["att2_sha"] = {"A-1-000": rows_by["A-1-000"][2]["config_sha"]}
            H["A1_shas"] = {r["config_sha"] for r in rows_by["A-1-000"].values()}
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


if __name__ == "__main__":
    sys.exit(main())
