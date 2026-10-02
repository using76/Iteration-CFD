#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""
aml.py - the AM-L tuning re-measure: rules campaigns on the HEAD binary against the committed rules campaign, family by family.

The subset is 120 tuning geometries stratified by family and stratum (one per
stratum, then the largest remainders, picked by the subset salt); the before
side is the committed rules campaign `prior/tuning_rules.json.gz`, as run and
as re-scored by `rescore/FEAT-CONSTRAINT.json`; the after side is a subset and
a full rules campaign directory run by the supervisor; per family it reports
failures, MFR, strict failure, BLC_8, BLC_full, the CAPABILITY-LIMITED wall
area split by the layer stage's `drop_cause`, and the feature-edge capture
share; the gates are integrity only (no harness error, no orphan, peak <= 60 %
of RAM, at most 6 live meshers, the replay reproduces); a G-BLC-1 target is
PROPOSED, never locked. The seal holds (a campaign that is not a tuning
campaign is refused); the test split is spent and is not read.

    python tools/autonomy/aml.py --plan [--ids-only]
    python tools/autonomy/aml.py --inspect DIR
    python tools/autonomy/aml.py --report --subset DIR --full DIR [--rcurv FILE] [--report-dir DIR] [--bundle]
    python tools/autonomy/aml.py --check [--report-dir DIR]
    python tools/autonomy/aml.py --ledger [--write] [--report-dir DIR] [--doc FILE]
    python tools/autonomy/aml.py --selftest
"""
import argparse
import copy
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for _p in (HERE, os.path.join(HERE, "corpus")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import baseline
import campaign
import explain
import schema
import split

AML_SCHEMA = "autonomy-aml/1"
HEADER = baseline.HEADER
REPORT_DIR = os.path.join(HERE, "aml")
REPORT_NAME = "L5.json"
REPORT_MD = "L5.md"
BUNDLE_NAME = "tuning_rules_L5.json.gz"
BEFORE_BUNDLE = os.path.join(HERE, "prior", "tuning_rules.json.gz")
RESCORE_REPORT = os.path.join(HERE, "rescore", "FEAT-CONSTRAINT.json")
PLAN_DOC = os.path.join(REPO, "docs", "15-autonomous-setup-plan.md")
SUBSET_N = 120
SUBSET_SALT = "aml-L5/1"
FAMILIES = ("A", "B", "D", "E", "F", "G")
LABELS = FAMILIES + ("tier1", "non-plane", "all")
CAUSES = ("inner_gate", "outer_gate", "thin_after_caps", "thin_proposed",
          "zero_disp", "unrecorded")
PEAK_FRAC_MAX = 0.60
MAX_STREAMS = campaign.MAX_STREAMS
TARGET_Z = 1.96
LEDGER_BEGIN = "<!-- BEGIN aml.py --ledger (AM-L L5) -->"
LEDGER_END = "<!-- END aml.py --ledger (AM-L L5) -->"
TARGET_STATUS = ("proposed from the tuning split; the user decides it (D-L9) and the evaluation unit "
                 "locks it before a fresh test seed is opened")

DEPARTURES = (
    "The subset is 120 tuning geometries drawn by a salted hash within each (family, stratum) stratum, one per stratum and the rest by largest remainder; the plan row names a stratified subset without a rule.",
    "The before side is the committed rules campaign (binary 054bba67, scored before the 2026-09-26 rule): its strict failure, BLC and CAPABILITY-LIMITED numbers are as run, and its failures are also given as FEAT-CONSTRAINT re-scored them; its rows carry no drop_cause and no capture share.",
    "CAPABILITY-LIMITED is the wall-area share of baseline.area_split on each geometry's final attempt (a requested patch dropped min_thickness or retreat_snapped on a geometry the R-PLANE predicate does not qualify for), and the terminal count is reported beside it.",
    "R-CURV on and off is baseline.py --rcurv on the same binary (attempt 1 only, no remedies); its F3 column here counts F3a-F3e where baseline's own report counts F3a-F3d.",
)


class AmlError(ValueError):
    """A campaign, a report or a request that is not what aml reports on; never guessed past."""


def _canon(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def _file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _rel(path):
    """REPO-relative with / separators when inside REPO, else the absolute path."""
    apath = os.path.abspath(path)
    try:
        r = os.path.relpath(apath, REPO)
    except ValueError:
        return apath.replace(os.sep, "/")
    if r == ".." or r.startswith(".." + os.sep):
        return apath.replace(os.sep, "/")
    return r.replace(os.sep, "/")


# --- (C2) the subset ---------------------------------------------------------


def subset(salt=SUBSET_SALT, n=SUBSET_N):
    """The stratified tuning subset: one per (family, stratum) stratum, then the
    largest remainders; within a stratum the salted hash picks the ids."""
    rows = campaign.load_manifest("tuning", "rules")
    strata = {}
    for r in rows:
        strata.setdefault((r["family"], r["stratum"]), []).append(r["geometry_id"])
    keys = sorted(strata)
    n_pool = len(rows)
    rest = n - len(keys)
    if rest < 0:
        raise AmlError("subset: %d geometries are fewer than the %d strata"
                       % (n, len(keys)))
    q = {}
    frac = {}
    for k in keys:
        x = rest * len(strata[k]) / n_pool
        q[k] = 1 + math.floor(x)
        frac[k] = x - math.floor(x)
    left = n - sum(q.values())
    for k in sorted(keys, key=lambda k: (-frac[k], k))[:left]:
        q[k] += 1
    picked = []
    for k in keys:
        ids = sorted(strata[k])
        ids.sort(key=lambda g: hashlib.sha256(
            ("%s|%s" % (salt, g)).encode("ascii")).hexdigest())
        picked.extend(ids[:q[k]])
    out = sorted(picked)
    return {"ids": out, "n": len(out), "n_pool": n_pool, "n_strata": len(keys),
            "quotas": {"%s/%s" % k: q[k] for k in keys}, "salt": salt,
            "sha256": hashlib.sha256("\n".join(out).encode("ascii")).hexdigest()}


def plan_line(s):
    return ("aml plan: %d of %d tuning geometries in %d strata, salt %s, "
            "subset sha256 %s"
            % (s["n"], s["n_pool"], s["n_strata"], s["salt"], s["sha256"]))


# --- (C3) views: one campaign, from a directory or a bundle ------------------


def view_from_dir(cdir):
    """A rules tuning campaign directory read as a view; the seal is baseline's."""
    header = baseline._check_seal_dir(cdir)
    if header.get("mode") != "rules":
        raise AmlError("mode: %s is a %s campaign; aml reads rules campaigns"
                       % (_rel(cdir), header.get("mode")))
    epath = os.path.join(cdir, campaign.FILES["end"])
    if not os.path.isfile(epath):
        raise AmlError("campaign: %s has no %s (it has not finished)"
                       % (_rel(cdir), campaign.FILES["end"]))
    geoms = campaign.load_geometries(cdir)
    rows_by = {}
    for r in campaign.load_rows(cdir):
        rows_by.setdefault(r["geometry_id"], {})[r["attempt"]] = r
    return {"source": _rel(cdir), "cdir": cdir, "header": header,
            "end": baseline._load_json(epath, "report"), "geoms": geoms,
            "rows_by": rows_by, "summary": None, "sha256": None}


def view_from_bundle(path):
    """A baseline bundle read as a view; a non-tuning bundle is refused."""
    if not os.path.isfile(path):
        raise AmlError("no such bundle file: %s" % path)
    try:
        b = baseline.read_bundle(path)
    except baseline.BaselineError as e:
        raise AmlError(str(e))
    h = b["campaign"]
    if (h.get("manifest", {}).get("source") != "tuning"
            or h.get("split_mode") == split.EVALUATE
            or any(r.get("split") != "tuning" for r in b.get("attempts") or [])):
        raise split.SplitSealed("the test split is spent: %s is not a tuning "
                                "campaign" % _rel(path))
    if h.get("mode") != "rules":
        raise AmlError("mode: %s is a %s campaign; aml reads rules campaigns"
                       % (_rel(path), h.get("mode")))
    rows_by = {}
    for r in b.get("attempts") or []:
        rows_by.setdefault(r["geometry_id"], {})[r["attempt"]] = r
    return {"source": _rel(path), "cdir": None, "header": h, "end": b["end"],
            "geoms": b["geometries"], "rows_by": rows_by,
            "summary": b.get("summary"), "sha256": _file_sha256(path)}


def restrict(view, ids):
    """A deep copy of the view keeping only the named geometries and rows."""
    have = {g["geometry_id"] for g in view["geoms"]}
    for gid in ids:
        if gid not in have:
            raise AmlError("restrict: %s is not in %s" % (gid, view["source"]))
    keep = set(ids)
    header = copy.deepcopy(view["header"])
    header["geometry_ids"] = sorted(ids)
    return {"source": view["source"], "cdir": view["cdir"], "header": header,
            "end": copy.deepcopy(view["end"]),
            "geoms": [copy.deepcopy(g) for g in view["geoms"]
                      if g["geometry_id"] in keep],
            "rows_by": {gid: copy.deepcopy(r)
                        for gid, r in view["rows_by"].items() if gid in keep},
            "summary": copy.deepcopy(view["summary"]), "sha256": view["sha256"]}


def case_causes(cdir, gid, attempt):
    """{patch name: drop_cause} from the case summary's layers stage, or None."""
    if cdir is None or attempt is None:
        return None
    path = os.path.join(cdir, "cases", "%s_a%d" % (gid, attempt),
                        "%s_a%d_summary.json" % (gid, attempt))
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        rep = json.load(f)
    stages = rep.get("stages")
    if isinstance(stages, list):
        stage = next((s for s in stages
                      if isinstance(s, dict) and s.get("stage") == "layers"), None)
    elif isinstance(stages, dict):
        stage = stages.get("layers")
    else:
        stage = None
    if stage is None:
        return {}
    names = {}
    for p in stage.get("patches") or []:
        if isinstance(p, dict) and p.get("name") is not None:
            names[p["name"]] = p.get("drop_cause")
    for reg in stage.get("regions") or []:
        for p in reg.get("patches") or []:
            if isinstance(p, dict) and p.get("name") is not None:
                names[p["name"]] = p.get("drop_cause")
    return names


# --- (C4) the per-geometry table ----------------------------------------------


def geometry_table(view, gates, knobs):
    """One row per geometry: failure, capture, the area split and its causes."""
    mrows = {r["geometry_id"]: r for r in campaign.load_manifest(
        "tuning", view["header"]["split_mode"],
        ids=view["header"]["geometry_ids"])}
    table = []
    for g in view["geoms"]:
        gid = g["geometry_id"]
        q = baseline.qualifies(mrows[gid], g.get("fingerprint"), gates, knobs)
        rws = view["rows_by"].get(gid, {})
        split_ = baseline.geometry_areas(g, rws, q)["final"]
        fa = g.get("final_attempt")
        oc = rws[fa]["outcome"] if fa is not None else None
        flags = (oc or {}).get("flags") or {}
        capture = oc.get("feature_capture") if oc else None
        f3e = flags.get("F3e") if oc else None
        f3e_only_on = (f3e is True
                       and not any(flags.get(k) is True
                                   for k in baseline.FLAG_KEYS if k != "F3e")
                       and capture is not None and capture > 0)
        causes = {c: 0.0 for c in CAUSES}
        if oc:
            total = sum((p.get("area_m2") or 0.0) for p in oc["patches"])
            dc = case_causes(view["cdir"], gid, fa) or {}
            for p in oc["patches"]:
                if p.get("delivered") is True:
                    continue
                if (p.get("requested") and p.get("layer_class")
                        in baseline.DROP_CLASSES and not q):
                    c = dc.get(p["name"])
                    causes[c if c in CAUSES[:-1] else "unrecorded"] += \
                        (p.get("area_m2") or 0.0) / total
            s = sum(causes.values())
            if abs(s - split_["capability_limited"]) > 1e-12:
                raise AmlError("causes: %s sums to %r, the area split says %r"
                               % (gid, s, split_["capability_limited"]))
        table.append({"geometry_id": gid, "family": g["family"],
                      "stratum": g["stratum"],
                      "terminal": campaign.terminal_of(g),
                      "final_attempt": fa, "attempts": len(rws),
                      "failure": bool(g["failure"]),
                      "strict": bool(g["strict_failure"]),
                      "blc8": g["blc8_a_priori"],
                      "blc_full": g["blc_full_a_priori"], "qualifies": q,
                      "cap_limited": split_["capability_limited"],
                      "capture": capture, "f3e": f3e,
                      "f3e_only_on": f3e_only_on, "causes": causes,
                      "config_shas": [rws[a]["config_sha"] for a in sorted(rws)],
                      "content_shas": [rws[a].get("content_sha256")
                                       for a in sorted(rws)]})
    table.sort(key=lambda r: r["geometry_id"])
    return table


# --- (C5) the groups ----------------------------------------------------------


def groups(table):
    """Per label: failures, MFR and its interval, strict, BLC, the area split,
    the drop causes and the capture facts."""
    out = {}
    for label in LABELS:
        if label == "all":
            ms = table
        elif label == "tier1":
            ms = [r for r in table if r["family"] in baseline.TIER1]
        elif label == "non-plane":
            ms = [r for r in table if r["qualifies"] is False]
        else:
            ms = [r for r in table if r["family"] == label]
        if not ms:
            continue
        n = len(ms)
        fail = sum(1 for r in ms if r["failure"])
        strict = sum(1 for r in ms if r["strict"])
        terminals = {}
        for r in ms:
            terminals[r["terminal"]] = terminals.get(r["terminal"], 0) + 1
        crows = [r["capture"] for r in ms if r["capture"] is not None]
        out[label] = {
            "n": n, "fail": fail, "mfr": fail / n,
            "mfr_ci": list(explain.clopper_pearson(fail, n)),
            "strict": strict, "strict_rate": strict / n,
            "strict_ci": list(explain.clopper_pearson(strict, n)),
            "blc8_mean": sum(r["blc8"] for r in ms) / n,
            "blc_full_mean": sum(r["blc_full"] for r in ms) / n,
            "cap_limited_mean": sum(r["cap_limited"] for r in ms) / n,
            "n_cap_terminal": sum(1 for r in ms
                                  if r["terminal"] == "CAPABILITY-LIMITED"),
            "terminals": {t: terminals[t] for t in sorted(terminals)},
            "capture_n": len(crows),
            "capture_median": statistics.median(crows) if crows else None,
            "capture_min": min(crows) if crows else None,
            "f3e": sum(1 for r in ms if r["f3e"] is True),
            "f3e_only_on": sum(1 for r in ms if r["f3e_only_on"]),
            "causes_mean": {c: sum(r["causes"][c] for r in ms) / n
                            for c in CAUSES}}
    return out


# --- (C6) integrity -----------------------------------------------------------


def integrity(view, replay_fn=None):
    """The campaign's integrity facts and the five gates; replay runs when it can."""
    end = view["end"]
    geoms = view["geoms"]
    harness_errors = sum(1 for g in geoms
                         if campaign.terminal_of(g) == "HARNESS-ERROR")
    if replay_fn is not None:
        rp = replay_fn(view)
    elif view["cdir"]:
        rp = campaign.replay(view["cdir"])
    else:
        rp = None
    if view["cdir"]:
        summary = campaign.summarise_campaign(view["cdir"])
    else:
        summary = view["summary"]
    replay_ok = rp["ok"] if rp else None
    checks = {
        "harness_errors_zero": harness_errors == 0,
        "orphans_zero": len(end["orphans"]) == 0,
        "peak_frac_ok": end["peak_frac"] <= PEAK_FRAC_MAX,
        "max_live_ok": end["max_live_mesher"] <= MAX_STREAMS,
        "replay_ok": replay_ok is True,
    }
    out = {
        "harness_errors": harness_errors,
        "harness_errors_recorded": end["harness_errors"],
        "orphans": len(end["orphans"]),
        "peak_frac": end["peak_frac"],
        "peak_rss_mib": end["peak_rss_mib"],
        "max_live_mesher": end["max_live_mesher"],
        "streams": end["streams"],
        "wall_seconds": end["wall_seconds"],
        "n_geometries": len(geoms),
        "n_rows": sum(len(v) for v in view["rows_by"].values()),
        "replay_ok": replay_ok,
        "replay_decisions": rp["decisions"] if rp else None,
        "replay_mismatches": len(rp.get("mismatches") or []) if rp else None,
        "audit_sample": summary.get("audit_sample") if summary else None,
        "explain_audit_ok": (summary.get("audit") or {}).get("ok")
        if summary else None,
        "checks": checks,
        "ok": all(checks.values()),
    }
    return out


# --- (C7) subset against full, the proposed target, R-CURV --------------------


def _geometry_signature(view, g):
    rws = view["rows_by"].get(g["geometry_id"]) or {}
    return [campaign.terminal_of(g),
            [[a, rws[a]["config_sha"], rws[a].get("content_sha256")]
             for a in sorted(rws)]]


def subset_vs_full(sub, full):
    """How many of the subset's geometries the full campaign reproduced."""
    full_by = {g["geometry_id"]: g for g in full["geoms"]}
    equal = 0
    diffs = []
    for g in sorted(sub["geoms"], key=lambda g: g["geometry_id"]):
        sg = _geometry_signature(sub, g)
        fg = _geometry_signature(full, full_by[g["geometry_id"]]) \
            if g["geometry_id"] in full_by else None
        if sg == fg:
            equal += 1
        else:
            diffs.append({"geometry_id": g["geometry_id"], "subset": sg,
                          "full": fg})
    return {"n": len(sub["geoms"]), "equal": equal, "n_diffs": len(diffs),
            "diffs": diffs[:20]}


def propose_target(table):
    """The tier-1 per-geometry BLC target: mean minus 1.96 standard errors,
    floored to 0.01, never below 0. PROPOSED - the user decides it."""
    ms = [r for r in table if r["family"] in baseline.TIER1]
    out = {"families": list(baseline.TIER1)}
    for key in ("blc8", "blc_full"):
        vals = [r[key] for r in ms]
        n = len(vals)
        if n < 2:
            out[key] = {"n": n, "mean": None, "sd": None, "lower95": None,
                        "target": None}
        else:
            mean = statistics.mean(vals)
            sd = statistics.stdev(vals)
            lower95 = mean - TARGET_Z * sd / math.sqrt(n)
            out[key] = {"n": n, "mean": mean, "sd": sd, "lower95": lower95,
                        "target": max(0.0, math.floor(lower95 * 100) / 100)}
    out["status"] = TARGET_STATUS
    return out


def _median_of(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def rcurv_block(rep):
    """The R-CURV report's paired facts, copied and regrouped (F3 = F3a-F3e)."""
    if rep.get("schema") != baseline.RCURV_SCHEMA:
        raise AmlError("rcurv: not a %s report" % baseline.RCURV_SCHEMA)

    def arm(arms, key):
        return {"failure": sum(1 for x in arms if x["failure"]),
                "strict_failure": sum(1 for x in arms if x["strict_failure"]),
                "f3": sum(1 for x in arms
                          if any((x.get("flags") or {}).get("F3" + s) is True
                                 for s in ("a", "b", "c", "d", "e"))),
                "cells_median": _median_of([x.get("n_cells") for x in arms]),
                "seconds_median": _median_of([x.get("seconds") for x in arms]),
                "cap_limited": sum(x["capability_limited"] for x in arms)
                / len(arms)}

    out = {}
    for label in ("A", "B", "D", "E", "F", "all"):
        pairs = rep["pairs"] if label == "all" \
            else [p for p in rep["pairs"] if p["family"] == label]
        if not pairs:
            continue
        w = arm([p["with"] for p in pairs], "with")
        o = arm([p["without"] for p in pairs], "without")
        out[label] = {"n": len(pairs),
                      "failure_with": w["failure"],
                      "failure_without": o["failure"],
                      "f3_with": w["f3"], "f3_without": o["f3"],
                      "strict_with": w["strict_failure"],
                      "strict_without": o["strict_failure"],
                      "cells_median_with": w["cells_median"],
                      "cells_median_without": o["cells_median"],
                      "seconds_median_with": w["seconds_median"],
                      "seconds_median_without": o["seconds_median"],
                      "cap_limited_with": w["cap_limited"],
                      "cap_limited_without": o["cap_limited"]}
    return {"n_pairs": rep["n_pairs"], "n_rows": rep["n_rows"],
            "skipped": rep["skipped"], "harness_errors": rep["harness_errors"],
            "binary_sha256": rep["binary_sha256"],
            "wall_seconds": rep["wall_seconds"],
            "peak_rss_mib": rep["peak_rss_mib"],
            "max_live_mesher": rep["max_live_mesher"],
            "orphans": rep["orphans"], "groups": out}


# --- (C8) the report ----------------------------------------------------------


def _comparison(before_groups, full_groups, resc):
    """Per family and all: the as-run before, the re-scored before, the after."""
    comp = {}
    for label in FAMILIES + ("all",):
        fg = full_groups[label]
        after = fg["fail"]
        if label == "all":
            comp[label] = {"n": before_groups["all"]["n"],
                           "fail_before_as_run": before_groups["all"]["fail"],
                           "fail_before_rescored": resc["failures_after"],
                           "fail_after": after,
                           "delta_vs_rescored": after - resc["failures_after"],
                           "rose": after > resc["failures_after"]}
        else:
            scored = resc["per_family"][label]
            comp[label] = {"n": fg["n"],
                           "fail_before_as_run": before_groups[label]["fail"],
                           "fail_before_rescored": scored,
                           "fail_after": after,
                           "delta_vs_rescored": after - scored,
                           "rose": after > scored}
    return comp


def build_report(sub, full, *, rcurv=None, before_path=BEFORE_BUNDLE,
                 rescore_path=RESCORE_REPORT, replay_fn=None):
    """The whole report; the three refusals fire before any table is built."""
    plan = subset()
    if sorted(sub["header"]["geometry_ids"]) != plan["ids"]:
        raise AmlError("subset: %s is not the %d-geometry subset (sha256 %s)"
                       % (sub["source"], SUBSET_N, plan["sha256"][:8]))
    full_ids = sorted(full["header"]["geometry_ids"])
    if full_ids != sorted(r["geometry_id"]
                          for r in campaign.load_manifest("tuning", "rules")):
        raise AmlError("full: %s holds %d geometries, not the %d tuning "
                       "geometries" % (full["source"], len(full_ids),
                                       baseline.TUNING_N))
    if sub["header"]["binary_sha256"] != full["header"]["binary_sha256"]:
        raise AmlError("binary: the subset ran %s, the full campaign %s"
                       % (sub["header"]["binary_sha256"][:12],
                          full["header"]["binary_sha256"][:12]))
    gates = schema.load_gates()
    knobs = schema.load_knobs()
    before = view_from_bundle(before_path)
    sub_table = geometry_table(sub, gates, knobs)
    full_table = geometry_table(full, gates, knobs)
    before_table = geometry_table(before, gates, knobs)
    sub_groups = groups(sub_table)
    full_groups = groups(full_table)
    before_groups = groups(before_table)
    sub_int = integrity(sub, replay_fn)
    full_int = integrity(full, replay_fn)
    with open(rescore_path, encoding="utf-8") as f:
        systems = json.load(f)["systems"]["rules"]
    comp = _comparison(before_groups, full_groups,
                       {"failures_after": systems["failures_after"],
                        "per_family": {f: systems["per_family"][f]["failures_after"]
                                       for f in FAMILIES}})
    report = {
        "$comment": HEADER, "schema": AML_SCHEMA, "date": baseline._today(),
        "binary_sha256": full["header"]["binary_sha256"],
        "git_sha": full["header"]["git_sha"], "subset_plan": plan,
        "before": {"source": before["source"], "sha256": before["sha256"],
                   "binary_sha256": before["header"]["binary_sha256"],
                   "groups": before_groups, "integrity": integrity(before),
                   "rescored": {"source": _rel(rescore_path),
                                "sha256": _file_sha256(rescore_path),
                                "failures_after": systems["failures_after"],
                                "per_family": {f: systems["per_family"][f]["failures_after"]
                                               for f in FAMILIES}}},
        "subset": {"source": sub["source"],
                   "campaign_id": sub["header"]["campaign_id"],
                   "groups": sub_groups, "integrity": sub_int,
                   "table": sub_table},
        "full": {"source": full["source"],
                 "campaign_id": full["header"]["campaign_id"],
                 "groups": full_groups, "integrity": full_int,
                 "table": full_table},
        "subset_vs_full": subset_vs_full(sub, full),
        "comparison": comp,
        "rises": [f for f in FAMILIES if comp[f]["rose"]],
        "target": propose_target(full_table),
        "rcurv": rcurv_block(rcurv) if rcurv is not None else None,
        "gates": {"subset_ok": sub_int["ok"], "full_ok": full_int["ok"],
                  "ok": sub_int["ok"] and full_int["ok"]},
        "bundle": None, "departures": list(DEPARTURES)}
    return report


def write_report(report, report_dir, *, full_dir=None, bundle=False):
    """The JSON and the md into report_dir; optionally the full campaign's bundle."""
    if bundle:
        if full_dir is None:
            raise AmlError("bundle: --bundle needs the full campaign directory")
        bpath = os.path.join(report_dir, BUNDLE_NAME)
        try:
            info = baseline.write_bundle(baseline.bundle_dir(full_dir), bpath)
        except baseline.BaselineError as e:
            raise AmlError(str(e))
        report["bundle"] = {"file": BUNDLE_NAME, "sha256": info["sha256"],
                            "content_sha256": info["content_sha256"],
                            "bytes": info["bytes"],
                            "n_rows": report["full"]["integrity"]["n_rows"]}
    os.makedirs(report_dir, exist_ok=True)
    jpath = os.path.join(report_dir, REPORT_NAME)
    with open(jpath, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(report, indent=1, sort_keys=True,
                           ensure_ascii=False) + "\n")
    mpath = os.path.join(report_dir, REPORT_MD)
    with open(mpath, "w", encoding="utf-8", newline="\n") as f:
        f.write(render_md(report))
    return {"json": jpath, "md": mpath}


# --- the markdown -------------------------------------------------------------


def _fmt(v, spec="%.3f"):
    return "-" if v is None else spec % v


def _md_head(report):
    b = report["before"]
    return ["<!-- %s -->" % HEADER, "",
            "# L5 - the AM-L tuning re-measure (rules campaigns, tuning split)",
            "",
            "- date: %s" % report["date"],
            "- binary sha256: %s" % report["binary_sha256"],
            "- git sha: %s" % report["git_sha"],
            "- subset: %s" % plan_line(report["subset_plan"]),
            "- before: %s (binary sha256 %s)" % (b["source"], b["binary_sha256"]),
            "- re-scored by: %s" % b["rescored"]["source"]]


def _md_integrity(report):
    out = ["", "## Integrity (the gates)", "",
           "| campaign | geometries | rows | harness errors | orphans | peak RAM"
           " | max live meshers | replay | audit reruns equal | wall s | ok |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, i in (("subset", report["subset"]["integrity"]),
                    ("full", report["full"]["integrity"]),
                    ("before (reported)", report["before"]["integrity"])):
        replay = "-" if i["replay_ok"] is None else \
            ("ok" if i["replay_ok"] else "MISMATCH")
        s = i["audit_sample"]
        audit = "-" if s is None else "%d/%d" % (s["equal"], s["n"])
        out.append("| %s | %d | %d | %d | %d | %.1f %% | %d | %s | %s | %.0f | %s |"
                   % (name, i["n_geometries"], i["n_rows"], i["harness_errors"],
                      i["orphans"], i["peak_frac"] * 100, i["max_live_mesher"],
                      replay, audit, i["wall_seconds"],
                      "PASS" if i["ok"] else "FAIL"))
    return out


def _md_families(report):
    out = ["", "## Family by family (tuning, a priori)", "",
           "| group | n | failures before (as run) | before (re-scored) | after"
           " | MFR before (re-scored) -> after | strict before -> after | BLC_8"
           " before -> after | BLC_full before -> after | CAPABILITY-LIMITED"
           " area before -> after | capture median after | F3e after | F3e"
           " only, attraction on |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    bg = report["before"]["groups"]
    fg = report["full"]["groups"]
    resc = report["before"]["rescored"]
    for label in LABELS:
        b, f = bg.get(label), fg.get(label)
        if b is None or f is None:
            continue
        if label in FAMILIES:
            r = resc["per_family"][label]
        elif label == "all":
            r = resc["failures_after"]
        else:
            r = None
        mfr = ("- -> %.3f" % f["mfr"]) if r is None else \
            ("%.3f -> %.3f" % (r / f["n"], f["mfr"]))
        out.append("| %s | %d | %d | %s | %d | %s | %.3f -> %.3f"
                   " | %.3f -> %.3f | %.3f -> %.3f | %.3f -> %.3f | %s | %d | %d |"
                   % (label, f["n"], b["fail"], _fmt(r, "%d"), f["fail"], mfr,
                      b["strict_rate"], f["strict_rate"], b["blc8_mean"],
                      f["blc8_mean"], b["blc_full_mean"], f["blc_full_mean"],
                      b["cap_limited_mean"], f["cap_limited_mean"],
                      _fmt(f["capture_median"]), f["f3e"], f["f3e_only_on"]))
    out.append("")
    out.append("MFR rises against FEAT-CONSTRAINT's re-scored rules number: %s"
               % (", ".join(report["rises"]) if report["rises"] else "none"))
    return out


def _md_causes(report):
    out = ["",
           "## Why the lost wall was lost (after; shares of all STL wall area)",
           "",
           "| group | CAPABILITY-LIMITED | inner_gate | outer_gate |"
           " thin_after_caps | thin_proposed | zero_disp | unrecorded |",
           "|---|---|---|---|---|---|---|---|"]
    fg = report["full"]["groups"]
    for label in LABELS:
        f = fg.get(label)
        if f is None:
            continue
        out.append("| %s | %.3f | %s |"
                   % (label, f["cap_limited_mean"],
                      " | ".join("%.3f" % f["causes_mean"][c] for c in CAUSES)))
    return out


def _md_subset(report):
    out = ["", "## The subset (stage a)", "",
           "| group | n | failures | MFR | strict | BLC_8 | BLC_full"
           " | CAPABILITY-LIMITED |",
           "|---|---|---|---|---|---|---|---|"]
    for label in LABELS:
        g = report["subset"]["groups"].get(label)
        if g is None:
            continue
        out.append("| %s | %d | %d | %.3f | %d | %.3f | %.3f | %.3f |"
                   % (label, g["n"], g["fail"], g["mfr"], g["strict"],
                      g["blc8_mean"], g["blc_full_mean"],
                      g["cap_limited_mean"]))
    s = report["subset_vs_full"]
    out.append("")
    out.append("Subset against the full campaign on its %d geometries: "
               "%d equal, %d differ." % (s["n"], s["equal"], s["n_diffs"]))
    return out


def _md_rcurv(report):
    out = ["", "## R-CURV on and off where it fires", ""]
    rc = report["rcurv"]
    if rc is None:
        out.append("Not measured.")
        return out
    out.append("| family | n | failure with | failure without | F3 with |"
               " F3 without | cells median with | cells median without |"
               " s median with | s median without | CAPABILITY-LIMITED with |"
               " CAPABILITY-LIMITED without |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for label in ("A", "B", "D", "E", "F", "all"):
        g = rc["groups"].get(label)
        if g is None:
            continue
        out.append("| %s | %d | %d | %d | %d | %d | %s | %s | %s | %s"
                   " | %.3f | %.3f |"
                   % (label, g["n"], g["failure_with"], g["failure_without"],
                      g["f3_with"], g["f3_without"],
                      _fmt(g["cells_median_with"], "%d"),
                      _fmt(g["cells_median_without"], "%d"),
                      _fmt(g["seconds_median_with"], "%.0f"),
                      _fmt(g["seconds_median_without"], "%.0f"),
                      g["cap_limited_with"], g["cap_limited_without"]))
    return out


def _md_target(report):
    out = ["", "## Proposed G-BLC-1 target (tier 1: A, B, E, F; a priori)", ""]
    for name, key in (("BLC_8", "blc8"), ("BLC_full", "blc_full")):
        m = report["target"][key]
        out.append("%s: mean %s, sd %s, n %d, lower 95 %% bound %s,"
                   " proposed target %s"
                   % (name, _fmt(m["mean"], "%.6f"), _fmt(m["sd"], "%.6f"),
                      m["n"], _fmt(m["lower95"], "%.6f"),
                      _fmt(m["target"], "%.2f")))
    out.append("")
    out.append(TARGET_STATUS)
    return out


def render_md(report):
    """The report as deterministic markdown (the date is the report's)."""
    parts = _md_head(report) + _md_integrity(report) + _md_families(report) \
        + _md_causes(report) + _md_subset(report) + _md_rcurv(report) \
        + _md_target(report)
    parts += ["", "## Departures", ""]
    parts += ["- " + d for d in report["departures"]]
    return "\n".join(parts) + "\n"


# --- the ledger ---------------------------------------------------------------


def _ledger_para(report):
    par = ("Run by `campaign.py --run --manifest tuning --mode rules` on the "
           "%d-geometry subset and then all %d tuning geometries, "
           "`baseline.py --rcurv`, then `aml.py --report`; binary sha256 "
           "`%s…%s`, tree `%s`; the report is `tools/autonomy/aml/L5.json` "
           "(and `.md`)"
           % (report["subset_plan"]["n"],
              report["before"]["groups"]["all"]["n"],
              report["binary_sha256"][:8], report["binary_sha256"][-4:],
              report["git_sha"][:7]))
    b = report.get("bundle")
    if b:
        par += ", the bundle `%s` (sha256 `%s…%s`)" % (b["file"],
                                                       b["sha256"][:8],
                                                       b["sha256"][-4:])
    par += ". These are tuning numbers, not a result; the test split is spent."
    return par


def _ledger_integrity(report):
    out = []
    for name in ("subset", "full"):
        i = report[name]["integrity"]
        out.append("%s %d geometries, %d rows, %d harness errors, %d orphans, "
                   "peak %.1f %%, max live %d, replay %s decisions: %s"
                   % (name, i["n_geometries"], i["n_rows"], i["harness_errors"],
                      i["orphans"], i["peak_frac"] * 100, i["max_live_mesher"],
                      i["replay_decisions"], "PASS" if i["ok"] else "FAIL"))
    return "- **Integrity:** " + "; ".join(out) + "."


def _ledger_families(report):
    bg = report["before"]["groups"]
    fg = report["full"]["groups"]
    resc = report["before"]["rescored"]
    out = ["- **Family by family** (failures as run / re-scored -> after, "
           "strict, BLC_8, BLC_full, CAPABILITY-LIMITED area before -> after, "
           "capture median, F3e, F3e only with the attraction on)"]
    for label in FAMILIES + ("tier1", "all"):
        b, f = bg[label], fg[label]
        if label == "all":
            r = resc["failures_after"]
        elif label in FAMILIES:
            r = resc["per_family"][label]
        else:
            r = None
        out.append("  - %s: %d / %s -> %d, strict %.3f -> %.3f, BLC_8 %.3f -> "
                   "%.3f, BLC_full %.3f -> %.3f, CAPABILITY-LIMITED %.3f -> "
                   "%.3f, capture median %s, F3e %d, F3e only with the "
                   "attraction on %d"
                   % (label, b["fail"], _fmt(r, "%d"), f["fail"],
                      b["strict_rate"], f["strict_rate"],
                      b["blc8_mean"], f["blc8_mean"],
                      b["blc_full_mean"], f["blc_full_mean"],
                      b["cap_limited_mean"], f["cap_limited_mean"],
                      _fmt(f["capture_median"]), f["f3e"], f["f3e_only_on"]))
    return out


def render_ledger(report):
    """The docs/15 §K entry: BEGIN..END, the headline, the run and the bullets."""
    bg = report["before"]["groups"]
    fg = report["full"]["groups"]
    resc = report["before"]["rescored"]
    t = report["target"]
    lines = [LEDGER_BEGIN, ""]
    lines.append("### The tuning re-measure (AM-L L5), %s: MFR %.3f -> %.3f, "
                 "tier-1 BLC_8 %.3f -> %.3f"
                 % (report["date"], resc["failures_after"] / bg["all"]["n"],
                    fg["all"]["mfr"], bg["tier1"]["blc8_mean"],
                    fg["tier1"]["blc8_mean"]))
    lines.append("")
    lines.append(_ledger_para(report))
    lines.append("")
    lines.append(_ledger_integrity(report))
    lines.extend(_ledger_families(report))
    lines.append("- **MFR rises against FEAT-CONSTRAINT's re-scored rules "
                 "number:** %s"
                 % (", ".join(report["rises"]) if report["rises"] else "none"))
    parts = ["CAPABILITY-LIMITED %.3f" % fg["all"]["cap_limited_mean"]]
    for c in CAUSES:
        if fg["all"]["causes_mean"][c] > 0:
            parts.append("%s %.3f" % (c, fg["all"]["causes_mean"][c]))
    lines.append("- **Why the lost wall was lost** (all): %s."
                 % "; ".join(parts))
    s = report["subset_vs_full"]
    g = report["subset"]["groups"]["all"]
    lines.append("- **The subset:** %d geometries, MFR %.3f, %d equal / %d "
                 "differ against the full campaign."
                 % (g["n"], g["mfr"], s["equal"], s["n_diffs"]))
    rc = report["rcurv"]
    if rc is None:
        lines.append("- **R-CURV on and off:** not measured.")
    else:
        a = rc["groups"]["all"]
        lines.append("- **R-CURV on and off:** %d pairs, failure %d -> %d, "
                     "F3 %d -> %d, cells median %s -> %s."
                     % (a["n"], a["failure_with"], a["failure_without"],
                        a["f3_with"], a["f3_without"],
                        _fmt(a["cells_median_with"], "%d"),
                        _fmt(a["cells_median_without"], "%d")))
    lines.append("- **Proposed G-BLC-1 target:** BLC_8 target %s (lower 95 %% "
                 "bound %s), BLC_full target %s (lower 95 %% bound %s); %s."
                 % (_fmt(t["blc8"]["target"], "%.2f"),
                    _fmt(t["blc8"]["lower95"], "%.6f"),
                    _fmt(t["blc_full"]["target"], "%.2f"),
                    _fmt(t["blc_full"]["lower95"], "%.6f"), TARGET_STATUS))
    lines.append("- **Where the run departs from the plan:** %s"
                 % " ".join(report["departures"]))
    lines.append("")
    lines.append(LEDGER_END)
    return "\n".join(lines)


def write_ledger(text, doc=PLAN_DOC):
    """Append the entry to the plan doc, or replace the held one in place."""
    if (not text.startswith(LEDGER_BEGIN + "\n")
            or not text.endswith(LEDGER_END)
            or text.count(LEDGER_BEGIN) != 1
            or text.count(LEDGER_END) != 1):
        raise AmlError("ledger: the rendered text does not carry its begin and "
                       "end markers exactly once, first and last")
    with open(doc, encoding="utf-8") as f:
        content = f.read().replace("\r\n", "\n")
    nb = content.count(LEDGER_BEGIN)
    ne = content.count(LEDGER_END)
    if nb:
        if nb != 1 or ne != 1 or content.index(LEDGER_END) < content.index(LEDGER_BEGIN):
            raise AmlError("ledger: %s holds %d begin and %d end markers"
                           % (doc, nb, ne))
        start = content.index(LEDGER_BEGIN)
        stop = content.index(LEDGER_END) + len(LEDGER_END)
        out = content[:start] + text + content[stop:]
        how = "replaced"
    else:
        out = content.rstrip("\n") + "\n\n" + text + "\n"
        how = "appended"
    with open(doc, "w", encoding="utf-8", newline="\n") as f:
        f.write(out)
    return how


# --- (C9) the check and the CLI ------------------------------------------------


def _check_groups(rep, gates, knobs):
    items = []
    for name, key in (("groups_full", "full"), ("groups_subset", "subset")):
        built = groups(rep[key]["table"])
        ok = _canon(built) == _canon(rep[key]["groups"])
        items.append({"name": name, "ok": ok,
                      "why": "the stored groups equal groups(table)" if ok
                      else "the stored %s groups differ from groups(table)"
                      % key})
    bv = view_from_bundle(BEFORE_BUNDLE)
    ok_sha = bv["sha256"] == rep["before"]["sha256"]
    ok_grp = _canon(groups(geometry_table(bv, gates, knobs))) \
        == _canon(rep["before"]["groups"])
    items.append({"name": "before", "ok": ok_sha and ok_grp,
                  "why": "the bundle sha256 and the rebuilt groups match the "
                  "stored before" if ok_sha and ok_grp else
                  "the stored before sha256 or groups differ from the bundle"})
    return items


def _check_bundle(rep, report_dir):
    b = rep["bundle"]
    bpath = os.path.join(report_dir, b["file"])
    if not os.path.isfile(bpath) or _file_sha256(bpath) != b["sha256"]:
        return {"name": "bundle", "ok": False,
                "why": "the bundle file is missing or its sha256 differs"}
    try:
        bundle = baseline.read_bundle(bpath)
    except baseline.BaselineError as e:
        return {"name": "bundle", "ok": False,
                "why": "the bundle does not read: %s" % e}
    ids = sorted(g["geometry_id"] for g in bundle.get("geometries") or [])
    want = sorted(r["geometry_id"] for r in rep["full"]["table"])
    if len(bundle.get("attempts") or []) != b["n_rows"] or ids != want:
        return {"name": "bundle", "ok": False,
                "why": "the bundle's attempt rows or geometry ids differ from "
                "the report"}
    return {"name": "bundle", "ok": True,
            "why": "%d attempt rows and %d geometries match the report"
            % (b["n_rows"], len(ids))}


def _check_rest(rep, report_dir):
    comp = _comparison(rep["before"]["groups"], rep["full"]["groups"],
                       rep["before"]["rescored"])
    rises = [f for f in FAMILIES if comp[f]["rose"]]
    items = []
    ok = _canon(comp) == _canon(rep["comparison"]) and rises == rep["rises"]
    items.append({"name": "comparison", "ok": ok,
                  "why": "rebuilt from the stored groups and re-scored numbers"
                  if ok else "the stored comparison or rises differ from the "
                  "rebuild"})
    tgt = propose_target(rep["full"]["table"])
    ok = _canon(tgt) == _canon(rep["target"])
    items.append({"name": "target", "ok": ok,
                  "why": "the stored target equals propose_target(full table)"
                  if ok else "the stored target differs from "
                  "propose_target(full table)"})
    if rep.get("bundle") is None:
        items.append({"name": "bundle", "ok": True, "why": "no bundle written"})
    else:
        items.append(_check_bundle(rep, report_dir))
    mpath = os.path.join(report_dir, REPORT_MD)
    if not os.path.isfile(mpath):
        items.append({"name": "md", "ok": False, "why": "the md file is missing"})
    else:
        with open(mpath, encoding="utf-8", newline="") as f:
            stored = f.read()
        ok = render_md(rep) == stored
        items.append({"name": "md", "ok": ok,
                      "why": "render_md(report) equals the stored md" if ok
                      else "the stored md differs from render_md(report)"})
    return items


def check(report_dir=REPORT_DIR):
    """Every stored number against its inputs, equality always by _canon."""
    jpath = os.path.join(report_dir, REPORT_NAME)
    if not os.path.isfile(jpath):
        raise AmlError("check: %s is missing" % jpath)
    with open(jpath, encoding="utf-8") as f:
        rep = json.load(f)
    gates = schema.load_gates()
    knobs = schema.load_knobs()
    ok = _canon(rep.get("subset_plan")) == _canon(subset())
    items = [{"name": "subset_plan", "ok": ok,
              "why": "the stored subset_plan equals subset()" if ok
              else "the stored subset_plan differs from subset()"}]
    items += _check_groups(rep, gates, knobs)
    items += _check_rest(rep, report_dir)
    return {"items": items,
            "verdict": "PASS" if all(i["ok"] for i in items) else "FAIL"}


class _ArgParser(argparse.ArgumentParser):
    """Usage errors exit 2, like campaign.py's."""

    def error(self, message):
        self.print_usage(sys.stderr)
        sys.stderr.write("aml: error: %s\n" % message)
        raise SystemExit(2)


def _cmd_inspect(cdir):
    v = view_from_dir(cdir)
    gates = schema.load_gates()
    knobs = schema.load_knobs()
    g = groups(geometry_table(v, gates, knobs))
    i = integrity(v)
    al = g["all"]
    print("aml inspect: %s: %d geometries, %d rows, fail %d (MFR %.3f), "
          "strict %d, BLC_8 %.3f, BLC_full %.3f, capability-limited %.3f, "
          "harness errors %d, orphans %d, peak %.1f %% RAM, max live %d, "
          "replay %s (%d decisions), integrity %s"
          % (_rel(cdir), i["n_geometries"], i["n_rows"], al["fail"], al["mfr"],
             al["strict"], al["blc8_mean"], al["blc_full_mean"],
             al["cap_limited_mean"], i["harness_errors"], i["orphans"],
             i["peak_frac"] * 100, i["max_live_mesher"],
             "ok" if i["replay_ok"] else "MISMATCH", i["replay_decisions"],
             "PASS" if i["ok"] else "FAIL"))
    for f in FAMILIES:
        if f not in g:
            continue
        fl = g[f]
        causes = " ".join("%s=%.3f" % (c, fl["causes_mean"][c])
                          for c in CAUSES if fl["causes_mean"][c] > 0) or "-"
        print("aml inspect: family %s n %d fail %d strict %d BLC_8 %.3f "
              "BLC_full %.3f capability-limited %.3f causes %s"
              % (f, fl["n"], fl["fail"], fl["strict"], fl["blc8_mean"],
                 fl["blc_full_mean"], fl["cap_limited_mean"], causes))
    return 0 if i["ok"] else 1


def _cmd_report(a):
    rcurv = None
    if a.rcurv:
        with open(a.rcurv, encoding="utf-8") as f:
            rcurv = json.load(f)
    rep = build_report(view_from_dir(a.subset), view_from_dir(a.full),
                       rcurv=rcurv)
    write_report(rep, a.report_dir, full_dir=a.full, bundle=a.bundle)
    fa = rep["full"]["groups"]["all"]
    ft = rep["full"]["groups"]["tier1"]
    print("aml report: full MFR %.3f (%d/%d), strict %.3f, tier-1 BLC_8 %.3f, "
          "rises %s, integrity %s"
          % (fa["mfr"], fa["fail"], fa["n"], fa["strict_rate"],
             ft["blc8_mean"], ",".join(rep["rises"]) or "none",
             "PASS" if rep["gates"]["ok"] else "FAIL"))
    return 0 if rep["gates"]["ok"] else 1


def _cmd_check(a):
    res = check(a.report_dir)
    for it in res["items"]:
        print("[check] %s %s %s" % (it["name"], "ok" if it["ok"] else "FAIL",
                                    it["why"]))
    print("CHECK PASS" if res["verdict"] == "PASS" else "CHECK FAIL")
    return 0 if res["verdict"] == "PASS" else 1


def _cmd_ledger(a):
    jpath = os.path.join(a.report_dir, REPORT_NAME)
    if not os.path.isfile(jpath):
        raise AmlError("ledger: %s is missing" % jpath)
    with open(jpath, encoding="utf-8") as f:
        rep = json.load(f)
    text = render_ledger(rep)
    if a.write:
        print("aml ledger: %s in %s" % (write_ledger(text, a.doc),
                                        _rel(a.doc)))
    else:
        sys.stdout.write(text + "\n")
    return 0


def main(argv=None):
    ap = _ArgParser(description="the AM-L tuning re-measure: rules campaigns "
                                "on the HEAD binary against the committed "
                                "rules campaign, family by family")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--ids-only", action="store_true",
                    help="with --plan: the ids, comma-joined")
    ap.add_argument("--inspect", metavar="DIR")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--subset", metavar="DIR")
    ap.add_argument("--full", metavar="DIR")
    ap.add_argument("--rcurv", metavar="FILE")
    ap.add_argument("--report-dir", metavar="DIR", default=REPORT_DIR)
    ap.add_argument("--bundle", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--ledger", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--doc", metavar="FILE", default=PLAN_DOC)
    a = ap.parse_args(argv)
    chosen = [k for k, v in (("--selftest", a.selftest), ("--plan", a.plan),
                             ("--inspect", a.inspect is not None),
                             ("--report", a.report), ("--check", a.check),
                             ("--ledger", a.ledger)) if v]
    if len(chosen) != 1:
        ap.error("exactly one of --selftest, --plan, --inspect, --report, "
                 "--check, --ledger (got %s)" % (", ".join(chosen) or "none"))
    if a.report and (not a.subset or not a.full):
        ap.error("--report needs --subset DIR and --full DIR")
    try:
        if a.selftest:
            return selftest()
        if a.plan:
            s = subset()
            print(",".join(s["ids"]) if a.ids_only else plan_line(s))
            return 0
        if a.inspect is not None:
            return _cmd_inspect(a.inspect)
        if a.report:
            return _cmd_report(a)
        if a.check:
            return _cmd_check(a)
        return _cmd_ledger(a)
    except (AmlError, baseline.BaselineError, campaign.CampaignError,
            split.SplitSealed) as e:
        sys.stderr.write("aml: error: %s\n" % e)
        return 2


# --- the selftest -------------------------------------------------------------


def _g1_subset(H):
    s = H["subset"]
    assert s["n"] == 120 and s["n_pool"] == 420 and s["n_strata"] == 18
    assert s["quotas"] == {
        "A/easy": 1, "A/hard": 11, "A/medium": 11, "B/easy": 5, "B/hard": 9,
        "B/medium": 10, "D/easy": 6, "D/hard": 2, "D/medium": 15, "E/easy": 7,
        "E/hard": 2, "E/medium": 5, "F/easy": 4, "F/hard": 6, "F/medium": 3,
        "G/easy": 8, "G/hard": 9, "G/medium": 6}
    per = {}
    for gid in s["ids"]:
        per[gid.split("-")[0]] = per.get(gid.split("-")[0], 0) + 1
    assert per == {"A": 23, "B": 24, "D": 23, "E": 14, "F": 13, "G": 23}
    assert s["ids"][:5] == ["A-1-000", "A-1-001", "A-1-002", "A-1-006",
                            "A-1-009"]
    assert s["ids"][-3:] == ["G-1-106", "G-1-110", "G-1-111"]
    assert s["sha256"] == "4379f25151d97be8ee85f8116dd23ff48c4119bfce73c9e538a" \
                          "21061eacaba27"
    assert subset() == s
    x = subset(salt="aml-L5/x")
    assert x["quotas"] == s["quotas"] and x["n"] == s["n"]
    assert x["sha256"] != s["sha256"]
    return ("subset: 120 of 420 tuning geometries in 18 strata, one per "
            "stratum then the largest remainders, sha256 4379f251; "
            "deterministic, and another salt draws other ids")


def _g2_before(H):
    t = H["table"]
    assert len(t) == 420
    g = groups(t)
    fams = {"A": (78, 84), "B": (5, 81), "D": (13, 51), "E": (18, 40),
            "F": (17, 28), "G": (62, 78)}
    for f, (fail, strict) in fams.items():
        assert g[f]["fail"] == fail and g[f]["strict"] == strict
    assert g["tier1"]["fail"] == 118 and g["tier1"]["n"] == 252
    assert g["tier1"]["strict"] == 233
    assert g["non-plane"]["fail"] == 144 and g["non-plane"]["n"] == 335
    assert g["non-plane"]["strict"] == 313
    assert g["all"]["fail"] == 193 and g["all"]["n"] == 420
    assert g["all"]["strict"] == 362
    for v, want in (("blc8_mean", 0.140476), ("blc_full_mean", 0.133533),
                    ("cap_limited_mean", 0.680952)):
        assert abs(g["all"][v] - want) <= 5e-7
    caps = {"A": 0.845238, "B": 0.964286, "D": 0.559524, "E": 0.880952,
            "F": 0.547619, "G": 0.321429}
    terms = {"A": 6, "B": 76, "D": 38, "E": 22, "F": 11, "G": 16}
    for f, want in caps.items():
        assert abs(g[f]["cap_limited_mean"] - want) <= 5e-7
        assert g[f]["n_cap_terminal"] == terms[f]
    assert g["all"]["n_cap_terminal"] == 169
    for label in LABELS:
        assert g[label]["capture_n"] == 0
        assert g[label]["capture_median"] is None
        assert g[label]["capture_min"] is None
        assert g[label]["f3e"] == 0 and g[label]["f3e_only_on"] == 0
        for c in CAUSES[:-1]:
            assert g[label]["causes_mean"][c] == 0.0
        assert abs(g[label]["causes_mean"]["unrecorded"]
                   - g[label]["cap_limited_mean"]) <= 1e-12
    bv = H["before"]
    mrows = {r["geometry_id"]: r for r in campaign.load_manifest(
        "tuning", bv["header"]["split_mode"],
        ids=bv["header"]["geometry_ids"])}
    q = {}
    areas = {}
    for g_ in bv["geoms"]:
        gid = g_["geometry_id"]
        q[gid] = baseline.qualifies(mrows[gid], g_.get("fingerprint"),
                                    H["gates"], H["knobs"])
        areas[gid] = baseline.geometry_areas(g_, bv["rows_by"].get(gid, {}),
                                             q[gid])
    by_key = {(e["family"], e["stratum"]): e
              for e in baseline._build_groups(bv["geoms"], bv["rows_by"], q,
                                              areas, "rules")}
    for label in LABELS:
        e = by_key[(label, "all")]
        mine = g[label]
        assert e["n"] == mine["n"] and e["fail"] == mine["fail"]
        assert e["strict"] == mine["strict"]
        assert abs(e["blc8_mean"] - mine["blc8_mean"]) <= 1e-12
        assert abs(e["blc_full_mean"] - mine["blc_full_mean"]) <= 1e-12
        assert abs(e["area"]["capability_limited"]
                   - mine["cap_limited_mean"]) <= 1e-12
    return ("before: the committed rules campaign gives 420 rows, failures "
            "A 78 B 5 D 13 E 18 F 17 G 62 (193), strict 362, BLC_8 0.140, "
            "CAPABILITY-LIMITED 0.681, all unrecorded; every family, tier1, "
            "non-plane and all agree with baseline's own groups")


def _g3_causes(H):
    with tempfile.TemporaryDirectory() as tmp:
        v = restrict(H["before"], ["B-1-000"])
        v["cdir"] = tmp
        d = os.path.join(tmp, "cases", "B-1-000_a1")
        os.makedirs(d)
        path = os.path.join(d, "B-1-000_a1_summary.json")

        def build():
            t = geometry_table(v, H["gates"], H["knobs"])
            row = t[0]
            assert abs(sum(row["causes"].values()) - row["cap_limited"]) <= 1e-12
            assert abs(row["cap_limited"] - 1.0) <= 1e-12
            return groups(t)

        def write(rep):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(rep, f)

        g = build()
        assert abs(g["B"]["causes_mean"]["unrecorded"] - 1.0) <= 1e-12
        write({"stages": [{"stage": "layers",
                           "patches": [{"name": "body",
                                        "drop_cause": "thin_after_caps"}]}]})
        g = build()
        assert abs(g["B"]["causes_mean"]["thin_after_caps"] - 1.0) <= 1e-12
        assert g["B"]["causes_mean"]["unrecorded"] == 0.0
        write({"stages": {"layers": {"regions": [{"patches": [
            {"name": "body", "drop_cause": "outer_gate"}]}]}}})
        g = build()
        assert abs(g["B"]["causes_mean"]["outer_gate"] - 1.0) <= 1e-12
        write({"stages": [{"stage": "layers",
                           "patches": [{"name": "body",
                                        "drop_cause": "thin_proposed"}]}]})
        g = build()
        assert abs(g["B"]["causes_mean"]["thin_proposed"] - 1.0) <= 1e-12
        write({"stages": [{"stage": "layers",
                           "patches": [{"name": "body",
                                        "drop_cause": "bogus"}]}]})
        g = build()
        assert abs(g["B"]["causes_mean"]["unrecorded"] - 1.0) <= 1e-12
        write({"stages": [{"stage": "layers",
                           "patches": [{"name": "body", "drop_cause": None}]}]})
        g = build()
        assert abs(g["B"]["causes_mean"]["unrecorded"] - 1.0) <= 1e-12
        try:
            restrict(H["before"], ["Z-9-999"])
            raise AssertionError("restrict accepted an unknown id")
        except AmlError as e:
            assert "is not in" in str(e)
    return ("causes: B-1-000's lost wall is unrecorded without a summary and "
            "takes the layer stage's drop_cause from either stages form; an "
            "unknown cause stays unrecorded; restrict refuses an unknown id")


def _g4_f3e(H):
    v = restrict(H["before"], ["A-1-000"])
    gid = "A-1-000"
    fa = v["geoms"][0]["final_attempt"]
    oc = v["rows_by"][gid][fa]["outcome"]

    def build():
        return groups(geometry_table(v, H["gates"], H["knobs"]))

    oc["flags"] = {k: False for k in baseline.FLAG_KEYS}
    oc["flags"]["F3e"] = True
    oc["feature_capture"] = 0.3
    t = geometry_table(v, H["gates"], H["knobs"])
    assert t[0]["f3e_only_on"] is True and t[0]["f3e"] is True
    assert t[0]["capture"] == 0.3
    g = build()
    assert g["A"]["f3e"] == 1 and g["A"]["f3e_only_on"] == 1
    assert g["A"]["capture_n"] == 1
    assert g["A"]["capture_median"] == 0.3 and g["A"]["capture_min"] == 0.3
    oc["feature_capture"] = 0.0
    assert geometry_table(v, H["gates"], H["knobs"])[0]["f3e_only_on"] is False
    oc["feature_capture"] = 0.3
    oc["flags"]["F3a"] = True
    assert geometry_table(v, H["gates"], H["knobs"])[0]["f3e_only_on"] is False
    oc["flags"]["F3a"] = False
    oc["flags"]["F3e"] = None
    t = geometry_table(v, H["gates"], H["knobs"])
    assert t[0]["f3e_only_on"] is False and t[0]["f3e"] is None
    assert build()["A"]["f3e"] == 0
    oc["feature_capture"] = None
    assert build()["A"]["capture_n"] == 0
    return ("f3e: an F3e-only failure with the attraction on counts once; "
            "capture 0, a second flag or no F3e do not; the capture median and "
            "minimum come from the measured rows")


def _g5_integrity(H):
    i = integrity(H["before"], H["stub"])
    assert i["harness_errors"] == 0 and i["orphans"] == 0
    assert i["peak_frac"] == 0.2672383020094763
    assert i["max_live_mesher"] == 4
    assert i["n_geometries"] == 420 and i["n_rows"] == 805
    assert i["audit_sample"] == {"check_failed": 0, "equal": 87, "n": 87}
    assert i["explain_audit_ok"] is True
    assert i["replay_ok"] is True and i["replay_decisions"] == 7
    assert i["replay_mismatches"] == 0
    assert i["ok"] is True

    def mutated(fn):
        v = copy.deepcopy(H["before"])
        fn(v)
        return integrity(v, H["stub"])

    def orphans(v):
        v["end"]["orphans"] = [{"pid": 1}]

    def peak(v):
        v["end"]["peak_frac"] = 0.61

    def live(v):
        v["end"]["max_live_mesher"] = 7

    def harness(v):
        v["geoms"][0]["terminal"] = "HARNESS-ERROR"
        v["geoms"][0]["reason"] = None

    assert mutated(orphans)["checks"]["orphans_zero"] is False
    assert mutated(peak)["checks"]["peak_frac_ok"] is False
    assert mutated(live)["checks"]["max_live_ok"] is False
    assert mutated(harness)["checks"]["harness_errors_zero"] is False

    def bad_stub(w):
        return {"ok": False, "decisions": 7,
                "mismatches": [{"geometry_id": "x"}]}

    bad = copy.deepcopy(H["before"])
    assert integrity(bad, bad_stub)["checks"]["replay_ok"] is False
    i = integrity(H["before"])
    assert i["replay_ok"] is None and i["replay_decisions"] is None
    assert i["ok"] is False
    return ("integrity: the committed campaign reads 0 harness errors, 0 "
            "orphans, peak 26.7 % and 4 live meshers; an orphan, a 61 % peak, "
            "7 live meshers, a harness error, a replay mismatch and an "
            "unreplayed campaign each fail it")


def _g6_compare(H):
    sub = restrict(H["before"], H["subset"]["ids"])
    sv = subset_vs_full(sub, H["before"])
    assert sv["n"] == 120 and sv["equal"] == 120 and sv["n_diffs"] == 0
    v = copy.deepcopy(sub)
    v["rows_by"]["A-1-000"][1]["content_sha256"] = "0" * 64
    sv = subset_vs_full(v, H["before"])
    assert sv["equal"] == 119 and sv["n_diffs"] == 1
    assert sv["diffs"][0]["geometry_id"] == "A-1-000"
    v = copy.deepcopy(sub)
    g9 = next(g for g in v["geoms"] if g["geometry_id"] == "A-1-009")
    assert campaign.terminal_of(g9) == "REFUSED"
    assert "A-1-009" not in sub["rows_by"]
    g9["terminal"] = "SURFACE-OPEN"
    sv = subset_vs_full(v, H["before"])
    assert sv["equal"] == 119 and sv["n_diffs"] == 1
    assert sv["diffs"][0]["geometry_id"] == "A-1-009"
    tgt = propose_target(H["table"])
    for key, mean, sd, lo, t in (("blc8", 0.075397, 0.264556, 0.042733, 0.04),
                                 ("blc_full", 0.070905, 0.254527, 0.039479,
                                  0.03)):
        m = tgt[key]
        assert m["n"] == 252
        assert abs(m["mean"] - mean) <= 5e-7 and abs(m["sd"] - sd) <= 5e-7
        assert abs(m["lower95"] - lo) <= 5e-7 and m["target"] == t
    assert tgt["status"] == TARGET_STATUS
    rows = [{"family": "B", "blc8": 0.0, "blc_full": 0.0},
            {"family": "B", "blc8": 0.0, "blc_full": 0.0},
            {"family": "B", "blc8": 0.0, "blc_full": 0.0}]
    m = propose_target(rows)["blc8"]
    assert m["n"] == 3 and m["sd"] == 0.0 and m["target"] == 0.0
    m = propose_target(rows[:1])["blc8"]
    assert m["n"] == 1 and m["target"] is None and m["mean"] is None
    with open(os.path.join(HERE, "baseline", "R-CURV.json"),
              encoding="utf-8") as f:
        rc = json.load(f)
    blk = rcurv_block(rc)
    assert blk["n_pairs"] == 133
    want = {"A": (33, 33, 33, 33, 33), "B": (70, 45, 52, 45, 52),
            "D": (27, 27, 27, 27, 27), "E": (1, 0, 0, 0, 0),
            "F": (2, 2, 2, 1, 1), "all": (133, 107, 114, 106, 113)}
    for label, (n, fw, wo, f3w, f3o) in want.items():
        g = blk["groups"][label]
        assert g["n"] == n and g["failure_with"] == fw
        assert g["failure_without"] == wo
        assert g["f3_with"] == f3w and g["f3_without"] == f3o
    bad = copy.deepcopy(rc)
    bad["schema"] = "x"
    try:
        rcurv_block(bad)
        raise AssertionError("rcurv_block accepted a foreign schema")
    except AmlError as e:
        assert "not a" in str(e)
    return ("compare and target: the subset matches itself 120 of 120, and a "
            "changed content sha and a changed terminal on a geometry without "
            "rows are each named; tier-1 BLC_8 0.0754 gives a proposed target "
            "0.04 and BLC_full 0.03; R-CURV 133 pairs read failure "
            "107 -> 114 and F3 106 -> 113")


def _g7_force(sub, full, rcurv, stub, tmp):
    with open(RESCORE_REPORT, encoding="utf-8") as f:
        resc = json.load(f)
    resc["systems"]["rules"]["per_family"]["B"]["failures_after"] = 4
    rpath = os.path.join(tmp, "rescore.json")
    with open(rpath, "w", encoding="utf-8") as f:
        json.dump(resc, f)
    rep2 = build_report(sub, full, rcurv=rcurv, replay_fn=stub,
                        rescore_path=rpath)
    assert rep2["comparison"]["B"]["rose"] is True
    assert rep2["rises"] == ["B"]


def _g7_refusals(sub, full, sub_ids, rcurv, stub):
    bad_sub = restrict(full, sub_ids[:119])
    try:
        build_report(bad_sub, full, rcurv=rcurv, replay_fn=stub)
        raise AssertionError("build_report accepted a 119-geometry subset")
    except AmlError as e:
        assert "is not the 120-geometry" in str(e)
    try:
        build_report(sub, restrict(full, sub_ids),
                     rcurv=rcurv, replay_fn=stub)
        raise AssertionError("build_report accepted a subset as full")
    except AmlError as e:
        assert "not the 420 tuning geometries" in str(e)
    bad_bin = copy.deepcopy(sub)
    bad_bin["header"]["binary_sha256"] = "f" * 64
    try:
        build_report(bad_bin, full, rcurv=rcurv, replay_fn=stub)
        raise AssertionError("build_report accepted a binary mismatch")
    except AmlError as e:
        assert "binary:" in str(e)


def _g7_roundtrip(rep, tmp):
    repdir = os.path.join(tmp, "rep")
    w = write_report(rep, repdir)
    assert os.path.isfile(w["json"]) and os.path.isfile(w["md"])
    rep2dir = os.path.join(tmp, "rep2")
    os.makedirs(rep2dir)
    for name in (REPORT_NAME, REPORT_MD):
        with open(os.path.join(repdir, name), "rb") as f:
            data = f.read()
        with open(os.path.join(rep2dir, name), "wb") as f:
            f.write(data)
    assert check(repdir)["verdict"] == "PASS"
    with open(w["json"], encoding="utf-8") as f:
        stored = json.load(f)
    stored["full"]["table"][0]["failure"] = \
        not stored["full"]["table"][0]["failure"]
    with open(w["json"], "w", encoding="utf-8", newline="\n") as f:
        json.dump(stored, f, indent=1, sort_keys=True, ensure_ascii=False)
    res = check(repdir)
    assert res["verdict"] == "FAIL"
    assert any(i["name"] == "groups_full" and not i["ok"]
               for i in res["items"])
    with open(w["md"], encoding="utf-8", newline="") as f:
        md = f.read()
    with open(w["md"], "w", encoding="utf-8", newline="\n") as f:
        f.write(md.replace("## Departures", "## Departure ", 1))
    res = check(repdir)
    assert res["verdict"] == "FAIL"
    assert any(i["name"] == "md" and not i["ok"] for i in res["items"])
    return repdir


def _g7_ledger(rep, tmp):
    text = render_ledger(rep)
    assert text.startswith(LEDGER_BEGIN) and text.endswith(LEDGER_END)
    assert text.count(LEDGER_BEGIN) == 1 and text.count(LEDGER_END) == 1
    assert "### The tuning re-measure (AM-L L5)" in text
    doc = os.path.join(tmp, "plan.md")
    with open(doc, "w", encoding="utf-8", newline="\n") as f:
        f.write("# x\n\n## K. Ledger\n\nold\n")
    assert write_ledger(text, doc) == "appended"
    with open(doc, encoding="utf-8") as f:
        assert f.read().endswith(LEDGER_END + "\n")
    rep3 = copy.deepcopy(rep)
    rep3["date"] = "2000-01-01"
    assert write_ledger(render_ledger(rep3), doc) == "replaced"
    with open(doc, encoding="utf-8") as f:
        stored = f.read()
    assert stored.count(LEDGER_BEGIN) == 1 and "2000-01-01" in stored
    with open(doc, "w", encoding="utf-8", newline="\n") as f:
        f.write(LEDGER_BEGIN + "\n\n" + LEDGER_BEGIN + "\n\nx "
                + LEDGER_END + "\n")
    try:
        write_ledger(text, doc)
        raise AssertionError("write_ledger accepted two begin markers")
    except AmlError:
        pass


def _g7_cli(H, rep2dir, tmp):
    exe = sys.executable
    me = os.path.abspath(__file__)
    r = subprocess.run([exe, me, "--plan"], capture_output=True,
                       encoding="utf-8", errors="replace", timeout=240)
    assert r.returncode == 0
    assert r.stdout.splitlines()[0] == plan_line(H["subset"])
    r = subprocess.run([exe, me, "--plan", "--ids-only"], capture_output=True,
                       encoding="utf-8", errors="replace", timeout=240)
    assert r.returncode == 0
    assert r.stdout.strip() == ",".join(H["subset"]["ids"])
    r = subprocess.run([exe, me, "--report", "--subset", tmp],
                       capture_output=True, encoding="utf-8",
                       errors="replace", timeout=240)
    assert r.returncode == 2
    r = subprocess.run([exe, me, "--check", "--report-dir", rep2dir],
                       capture_output=True, encoding="utf-8",
                       errors="replace", timeout=240)
    assert r.returncode == 0 and "CHECK PASS" in r.stdout


def _g7_report(H):
    with open(os.path.join(HERE, "baseline", "R-CURV.json"),
              encoding="utf-8") as f:
        rcurv = json.load(f)
    with tempfile.TemporaryDirectory() as tmp:
        sub = restrict(H["before"], H["subset"]["ids"])
        rep = build_report(sub, H["before"], rcurv=rcurv,
                           replay_fn=H["stub"])
        c = rep["comparison"]
        assert c["A"] == {"n": 84, "fail_before_as_run": 78,
                          "fail_before_rescored": 84, "fail_after": 78,
                          "delta_vs_rescored": -6, "rose": False}
        assert (c["B"]["fail_before_as_run"], c["B"]["fail_before_rescored"],
                c["B"]["fail_after"]) == (5, 55, 5)
        assert (c["all"]["fail_before_as_run"],
                c["all"]["fail_before_rescored"],
                c["all"]["fail_after"]) == (193, 340, 193)
        assert rep["rises"] == []
        assert rep["gates"] == {"subset_ok": True, "full_ok": True,
                                "ok": True}
        _g7_force(sub, H["before"], rcurv, H["stub"], tmp)
        _g7_refusals(sub, H["before"], H["subset"]["ids"], rcurv, H["stub"])
        _g7_roundtrip(rep, tmp)
        _g7_ledger(rep, tmp)
        _g7_cli(H, os.path.join(tmp, "rep2"), tmp)
    return ("report: before 193 / re-scored 340 against a stand-in after of "
            "193 names no rise and a forced one names B; the report "
            "round-trips through check, a flipped row and an edited md fail "
            "it; three refusals; the ledger appends then replaces; the CLI "
            "plans and checks")


def selftest():
    H = {"gates": schema.load_gates(), "knobs": schema.load_knobs(),
         "before": view_from_bundle(BEFORE_BUNDLE)}
    H["table"] = geometry_table(H["before"], H["gates"], H["knobs"])
    H["subset"] = subset()
    H["stub"] = lambda v: {"ok": True, "decisions": 7, "mismatches": []}
    groups_ = (("subset", _g1_subset), ("before", _g2_before),
               ("causes", _g3_causes), ("f3e", _g4_f3e),
               ("integrity", _g5_integrity),
               ("compare and target", _g6_compare), ("report", _g7_report))
    bad = 0
    for name, fn in groups_:
        try:
            text = fn(H)
        except Exception as e:  # noqa: BLE001 - the failure IS the report
            bad += 1
            print("SELFTEST FAIL: %s: %s: %s" % (name, type(e).__name__, e))
        else:
            print("[ok] " + text)
    print("SELFTEST PASS" if not bad else "SELFTEST FAIL")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
