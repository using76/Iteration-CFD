#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""
baseline.py - the two baselines of docs/15 §F (AM-12, chain M).

B0-template (one naive config per geometry) and B0-LHS (four Latin-hypercube
configs around it, best of 4) run through campaign.py with ONE binary on both
splits.  The tuning split is reported per family and stratum: failure and
strict failure with Clopper-Pearson intervals, BLC_8, BLC_full, BLC_beta,
cells, seconds, the F flags, and the share of each geometry's STL wall area
whose layers were given up on a snapped wall (CAPABILITY-LIMITED, docs/15
§I-1).  Each test campaign is sealed into one deterministic gzip bundle whose
sha256 goes into a write-once lock; the bundle opens only in mode evaluate.
A paired run measures R-CURV's F3 cost on the tuning rows where it fires.

    python tools/autonomy/baseline.py --selftest
    python tools/autonomy/baseline.py --run --split tuning --system b0-template --out DIR
    python tools/autonomy/baseline.py --report --template DIR --lhs DIR
    python tools/autonomy/baseline.py --rcurv --out DIR
    python tools/autonomy/baseline.py --check
"""

import argparse
import copy
import datetime
import gzip
import hashlib
import io
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for _p in (HERE, os.path.join(HERE, "corpus")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import campaign  # noqa: E402
import explain  # noqa: E402
import preflight  # noqa: E402
import remedies  # noqa: E402
import rules  # noqa: E402
import schema  # noqa: E402
import split  # noqa: E402

REPORT_DIR = os.path.join(HERE, "baseline")
SEALED_SUBDIR = "sealed"
LOCK_NAME = "sealed.lock"
SYSTEMS = ("b0-template", "b0-lhs")
SPLITS = ("tuning", "test")
TIER1 = ("A", "B", "E", "F")   # docs/15 §F G-BLC-1's families (tier 1, snapped walls)
CATEGORIES = ("delivered", "capability_limited", "plane_fixable", "no_full_stack",
              "other", "no_mesh")
DROP_CLASSES = ("min_thickness", "retreat_snapped")
RCURV_FAMILIES = ("A", "B", "D", "E", "F")
RCURV_AUDIT_OFF = 2 ** 40      # int(sha[:8], 16) % 2**40 == 0 only for 00000000
TUNING_N = 420                 # the tuning manifest's rows (split.lock)
BUNDLE_SCHEMA = "autonomy-baseline-bundle/1"
REPORT_SCHEMA = "autonomy-baseline-report/1"
SEAL_SCHEMA = "autonomy-baseline-seal/1"
RCURV_SCHEMA = "autonomy-baseline-rcurv/1"
CHECK_SCHEMA = "autonomy-baseline-check/1"
HEADER = ("meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). Source-available, "
          "not Open Source. No GPL-licensed source was consulted.")
RCURV_CITE = ("docs/15 §C L1 R-CURV; DECISIONS 2026-09-24 "
              "(its F3 cost is measured with the rule off)")
RCURV_MSG = "R-CURV: off for the paired F3-cost run (the rule is measured, not changed)"
FLAG_KEYS = ("F1", "F2", "F3a", "F3b", "F3c", "F3d", "F4", "F5")
BETA_KEYS = ("0.5", "0.8", "0.95")
DEPARTURES = (
    "docs/15 §F runs the baselines at 12 streams; the house caps campaigns at 6 "
    "while the solver workflow runs (campaign.MAX_STREAMS).",
    "docs/15 §F's \"sealed file\" is sealed per system here: one gzip bundle per "
    "baseline system (baseline/sealed/test_<system>.json.gz) whose sha256 goes "
    "into a write-once baseline/sealed.lock; the plaintext campaign directory is "
    "removed once the bundle is verified, and only load_sealed(system, "
    "\"evaluate\") opens a bundle.",
    "CAPABILITY-LIMITED is a remedies terminal; a baseline runs no remedies, so "
    "the report reads it from the layer rows with remedies' own predicate: a "
    "requested patch dropped min_thickness or retreat_snapped (the two classes "
    "remedies names CAPABILITY-LIMITED) on a geometry R-PLANE does not qualify "
    "for. It is counted whatever the F flags say; a second column counts it only "
    "on geometries with no F flag (fclean), where remedies would end "
    "CAPABILITY-LIMITED at once.",
    "DECISIONS 2026-09-24: R-CURV is kept \"until AM-12 measures its F3 cost\": "
    "the paired run of this module is that measurement (attempt 1 only, no "
    "remedies, reported not gated).",
    "a thin body lost in castellation exits 1 at layers and scores config (F1), "
    "not F4.",
)


class BaselineError(ValueError):
    """A refused request or a harness inconsistency - never a verdict."""


def _today():
    return datetime.date.today().isoformat()


def _sha256_of_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _dump_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def _write_text(path, text):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


# --- (C2) bundles: one campaign directory as one deterministic gzip file ------

def bundle_dir(cdir):
    cpath = os.path.join(cdir, campaign.FILES["campaign"])
    epath = os.path.join(cdir, campaign.FILES["end"])
    if not os.path.isfile(cpath):
        raise BaselineError("bundle: %s is missing" % cpath)
    if not os.path.isfile(epath):
        raise BaselineError("bundle: %s is missing" % epath)
    with open(cpath, encoding="utf-8") as f:
        head = json.load(f)
    with open(epath, encoding="utf-8") as f:
        end = json.load(f)
    return {"$comment": HEADER, "schema": BUNDLE_SCHEMA, "campaign": head,
            "end": end, "attempts": campaign.load_rows(cdir),
            "geometries": campaign.load_geometries(cdir),
            "records": campaign._read_jsonl(
                os.path.join(cdir, campaign.FILES["records"])),
            "jobs": campaign._read_jsonl(os.path.join(cdir, campaign.FILES["jobs"])),
            "samples": campaign._read_jsonl(
                os.path.join(cdir, campaign.FILES["samples"])),
            "summary": campaign.summarise_campaign(cdir)}


def _bundle_file_bytes(bundle):
    raw = _canonical(bundle)
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, compresslevel=9,
                       mtime=0) as gz:
        gz.write(raw)
    return buf.getvalue()


def write_bundle(bundle, path):
    if os.path.exists(path):
        raise BaselineError("exists: %s" % path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    data = _bundle_file_bytes(bundle)
    with open(path, "wb") as f:
        f.write(data)
    return {"sha256": hashlib.sha256(data).hexdigest(),
            "content_sha256": hashlib.sha256(_canonical(bundle)).hexdigest(),
            "bytes": len(data)}


def read_bundle(path):
    with open(path, "rb") as f:
        data = f.read()
    try:
        bundle = json.loads(gzip.decompress(data).decode("utf-8"))
    except (OSError, ValueError) as e:
        raise BaselineError("bundle: %s does not decompress: %s" % (path, e))
    if not isinstance(bundle, dict) or bundle.get("schema") != BUNDLE_SCHEMA:
        raise BaselineError("bundle: %s is not a %s bundle" % (path, BUNDLE_SCHEMA))
    return bundle


def _write_or_match(report_dir, fname, bundle):
    """An existing file is replaced ONLY when its content_sha256 is equal."""
    os.makedirs(report_dir, exist_ok=True)
    path = os.path.join(report_dir, fname)
    data = _bundle_file_bytes(bundle)
    info = {"file": fname, "sha256": hashlib.sha256(data).hexdigest(),
            "content_sha256": hashlib.sha256(_canonical(bundle)).hexdigest(),
            "bytes": len(data)}
    if os.path.exists(path):
        try:
            old = read_bundle(path)
        except (BaselineError, OSError, ValueError):
            old = None
        if old is None or _canonical(old) != _canonical(bundle):
            raise BaselineError("exists: %s holds a different bundle" % path)
        if _sha256_of_file(path) == info["sha256"]:
            return info
        os.remove(path)
    with open(path, "wb") as f:
        f.write(data)
    return info


# --- (C3) the wall-area split of one outcome (docs/15 §D.2) -------------------

def area_split(outcome, qualifies):
    """Each patch of the STL goes to exactly ONE category, first match wins;
    the denominator is ALL STL wall area."""
    acc = {c: 0.0 for c in CATEGORIES}
    total = 0.0
    for p in outcome.get("patches") or []:
        a = p.get("area_m2") or 0.0
        total += a
        if p.get("delivered") is True:
            cat = "delivered"
        elif p.get("requested") and p.get("layer_class") in DROP_CLASSES:
            cat = "plane_fixable" if qualifies else "capability_limited"
        elif p.get("requested") and p.get("layer_class") == "no_full_stack":
            cat = "no_full_stack"
        else:
            cat = "other"
        acc[cat] += a
    if total <= 0:
        raise BaselineError("area_split: the total STL wall area is %r" % total)
    return {c: acc[c] / total for c in CATEGORIES}


def _empty_areas():
    out = {c: 0.0 for c in CATEGORIES}
    out["no_mesh"] = 1.0
    return out


def geometry_areas(g, rows, qualifies):
    fa = g.get("final_attempt")
    if fa is None:
        return {"final": _empty_areas(), "mean4": _empty_areas(), "fclean": 0.0}
    ordered = [rows[a] for a in sorted(rows)]
    final_row = rows[fa]
    final = area_split(final_row["outcome"], qualifies)
    cats = {c: [] for c in CATEGORIES}
    for r in ordered:
        sp = area_split(r["outcome"], qualifies)
        for c in CATEGORIES:
            cats[c].append(sp[c])
    mean4 = {c: sum(cats[c]) / len(cats[c]) for c in CATEGORIES}
    fclean = final["capability_limited"] \
        if not final_row["outcome"]["failure"] else 0.0
    return {"final": final, "mean4": mean4, "fclean": fclean}


# --- (C4) the R-PLANE predicate ------------------------------------------------

def qualifies(mrow, fp, gates, knobs):
    """True when remedies' R-PLANE predicate fires on the B0-template config;
    False when it does not; None without a fingerprint (neither)."""
    if fp is None:
        return None
    ctx = {"geometry_id": mrow["geometry_id"],
           "config": campaign.b0_template(mrow, fp),
           "fingerprint": fp, "flow": mrow["flow"]}
    return remedies._qualifies(ctx, gates, knobs)[0] is not None


# --- (C5) the tuning report ----------------------------------------------------

def _load_json(path, what):
    if not os.path.isfile(path):
        raise BaselineError("%s: %s is missing" % (what, path))
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _check_seal_dir(cdir):
    """docs/15 §F: the tuning report never touches a test campaign."""
    header = _load_json(os.path.join(cdir, campaign.FILES["campaign"]), "report")
    smode = header.get("split_mode")
    source = (header.get("manifest") or {}).get("source")
    if smode == split.EVALUATE or source != "tuning":
        raise split.SplitSealed(
            "the test split's baselines are sealed until the evaluation "
            "(docs/15 §F): %s is a %s campaign of %s" % (cdir, smode, source))
    for r in campaign.load_rows(cdir):
        if r.get("split") != "tuning":
            raise split.SplitSealed(
                "the test split's baselines are sealed until the evaluation "
                "(docs/15 §F): %s holds a %s row (%s)" % (cdir, r.get("split"),
                                                          r.get("geometry_id")))
    return header


def _group_stats(family, stratum, ms, rows_by, q, areas, system):
    n = len(ms)
    fails = sum(1 for g in ms if g["failure"])
    stricts = sum(1 for g in ms if g["strict_failure"])
    grp = {"family": family, "stratum": stratum, "n": n, "fail": fails,
           "mfr": fails / n, "mfr_ci": list(explain.clopper_pearson(fails, n)),
           "strict": stricts, "strict_rate": stricts / n,
           "strict_ci": list(explain.clopper_pearson(stricts, n)),
           "blc8_mean": sum(g["blc8_a_priori"] for g in ms) / n,
           "blc_full_mean": sum(g["blc_full_a_priori"] for g in ms) / n}
    beta_lo = {k: [] for k in BETA_KEYS}
    inexact = 0
    cells = []
    mesher_s = []
    wall_s = []
    flags = {k: 0 for k in FLAG_KEYS}
    flags["F3"] = 0
    fclasses = {}
    terminals = {}
    for g in ms:
        t = campaign.terminal_of(g)
        terminals[t] = terminals.get(t, 0) + 1
        gid = g["geometry_id"]
        row = rows_by.get(gid, {}).get(g.get("final_attempt"))
        if row is None:
            flags["F1"] += 1
            fclasses[t] = fclasses.get(t, 0) + 1
            for k in BETA_KEYS:
                beta_lo[k].append(0.0)
            wall_s.append(g["seconds"])
            continue
        oc = row["outcome"]
        if oc.get("failure_class") is None:
            fclasses["pass"] = fclasses.get("pass", 0) + 1
        else:
            fclasses[oc["failure_class"]] = fclasses.get(oc["failure_class"], 0) + 1
        for k in FLAG_KEYS:
            if oc.get("flags", {}).get(k) is True:
                flags[k] += 1
        if any(oc.get("flags", {}).get("F3" + s) is True for s in ("a", "b", "c", "d")):
            flags["F3"] += 1
        bad_beta = False
        for b in oc.get("blc_beta_a_priori") or []:
            key = str(b["beta"])
            if key in beta_lo:
                beta_lo[key].append(b["lo"])
            if b.get("exact") is False:
                bad_beta = True
        if bad_beta:
            inexact += 1
        if oc.get("n_cells") is not None:
            cells.append(oc["n_cells"])
        mesher_s.append(sum((r["outcome"].get("seconds") or 0.0)
                            for r in rows_by[gid].values()))
        wall_s.append(g["seconds"])
    grp["blc_beta_mean"] = {k: (sum(v) / len(v) if v else 0.0)
                            for k, v in beta_lo.items()}
    grp["blc_beta_inexact"] = inexact
    grp["cells_median"] = statistics.median(cells) if cells else None
    grp["cells_max"] = max(cells) if cells else None
    grp["mesher_seconds_median"] = statistics.median(mesher_s) if mesher_s else None
    grp["wall_seconds_median"] = statistics.median(wall_s) if wall_s else None
    grp["flags"] = flags
    grp["failure_classes"] = fclasses
    grp["terminals"] = terminals
    area_acc = {c: 0.0 for c in CATEGORIES}
    mean4_acc = {c: 0.0 for c in CATEGORIES}
    fclean_acc = 0.0
    n_cap = 0
    m4_fail = []
    m4_strict = []
    m4_blc8 = []
    for g in ms:
        gid = g["geometry_id"]
        ar = areas[gid]
        for c in CATEGORIES:
            area_acc[c] += ar["final"][c]
            mean4_acc[c] += ar["mean4"][c]
        fclean_acc += ar["fclean"]
        if ar["final"]["capability_limited"] > 0:
            n_cap += 1
        rws = rows_by.get(gid, {})
        if system == "b0-lhs":
            if not rws:
                m4_fail.append(1.0)
                m4_strict.append(1.0)
                m4_blc8.append(0.0)
            else:
                m4_fail.append(1.0 if g.get("lhs_fail_frac") is None
                               else g["lhs_fail_frac"])
                m4_strict.append(sum(1 for r in rws.values()
                                     if r["outcome"]["strict_failure"]) / len(rws))
                m4_blc8.append(sum(r["outcome"]["blc8_a_priori"]
                                   for r in rws.values()) / len(rws))
    grp["area"] = {c: area_acc[c] / n for c in CATEGORIES}
    grp["area_fclean"] = fclean_acc / n
    grp["n_capability_limited"] = n_cap
    if system == "b0-lhs":
        grp["mean4"] = {"fail_frac": sum(m4_fail) / n,
                        "strict_frac": sum(m4_strict) / n,
                        "blc8": sum(m4_blc8) / n,
                        "area": {c: mean4_acc[c] / n for c in CATEGORIES}}
    return grp


def _group_keys(geoms):
    pairs = []
    for g in geoms:
        p = (g["family"], g["stratum"])
        if p not in pairs:
            pairs.append(p)
    fams = sorted({f for f, _ in pairs})
    return sorted(pairs) + [(f, "all") for f in fams], fams


def _group_members(family, stratum, geoms, q):
    out = []
    for g in geoms:
        if family not in ("all", "tier1", "non-plane") and g["family"] != family:
            continue
        if family == "tier1" and g["family"] not in TIER1:
            continue
        if family == "non-plane" and q.get(g["geometry_id"]) is not False:
            continue
        if stratum != "all" and g["stratum"] != stratum:
            continue
        out.append(g)
    return out


def _build_groups(geoms, rows_by, q, areas, system):
    keys, _fams = _group_keys(geoms)
    out = []
    for family, stratum in keys:
        ms = _group_members(family, stratum, geoms, q)
        if ms:
            out.append(_group_stats(family, stratum, ms, rows_by, q, areas, system))
    for family, label in (("tier1", "tier1"), ("non-plane", "non-plane"),
                          ("all", "all")):
        ms = _group_members(label, "all", geoms, q)
        if ms:
            out.append(_group_stats(family, "all", ms, rows_by, q, areas, system))
    return out


def _system_report(cdir, header, gates, knobs, expect_n):
    end = _load_json(os.path.join(cdir, campaign.FILES["end"]), "report")
    geoms = campaign.load_geometries(cdir)
    terms = [campaign.terminal_of(g) for g in geoms]
    recorded = sum(1 for g in geoms if g["terminal"] == "HARNESS-ERROR")
    if recorded != end["harness_errors"]:
        raise BaselineError(
            "report: %s: campaign_end.json counts %d harness errors, the end "
            "records %d" % (cdir, end["harness_errors"], recorded))
    rows_by = {}
    for r in campaign.load_rows(cdir):
        rows_by.setdefault(r["geometry_id"], {})[r["attempt"]] = r
    replay = campaign.replay(cdir)
    summary = campaign.summarise_campaign(cdir)
    mrows = {r["geometry_id"]: r for r in campaign.load_manifest(
        "tuning", header["split_mode"], ids=header["geometry_ids"])}
    q = {}
    areas = {}
    for g in geoms:
        gid = g["geometry_id"]
        fp = g.get("fingerprint")
        q[gid] = qualifies(mrows[gid], fp, gates, knobs)
        areas[gid] = geometry_areas(g, rows_by.get(gid, {}), q[gid])
    groups = _build_groups(geoms, rows_by, q, areas, header["system"])
    audit = summary.get("audit")
    sample = summary["audit_sample"]
    blk = {"campaign_id": header["campaign_id"], "n_geometries": len(geoms),
           "n_expected": expect_n, "n_rows": sum(len(v) for v in rows_by.values()),
           "wall_seconds": end["wall_seconds"], "peak_rss_mib": end["peak_rss_mib"],
           "peak_frac": end["peak_frac"], "max_live_mesher": end["max_live_mesher"],
           "orphans": len(end["orphans"]),
           "harness_errors": terms.count("HARNESS-ERROR"),
           "harness_errors_recorded": end["harness_errors"],
           "surface_refused": sorted(g["geometry_id"] for g, t in
                                     zip(geoms, terms)
                                     if t == "SURFACE-REFUSED"),
           "binary_sha256": header["binary_sha256"], "git_sha": header["git_sha"],
           "manifest_sha256": header["manifest"]["sha256"],
           "replay_ok": bool(replay["ok"]),
           "replay_decisions": replay["decisions"],
           "audit_ok": bool(audit and audit["ok"]),
           "audit_reruns": sample["n"], "audit_reruns_equal": sample["equal"],
           "audit_check_failed": sample["check_failed"]}
    return {"cdir": cdir, "geoms": geoms, "groups": groups, "blk": blk}


def _seal_lock_path(report_dir):
    return os.path.join(report_dir, LOCK_NAME)


def _read_seal_lock(report_dir):
    path = _seal_lock_path(report_dir)
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def report(template_dir, lhs_dir, *, report_dir=REPORT_DIR, expect_n=TUNING_N,
           allow_partial=False, write=True):
    header_t = _check_seal_dir(template_dir)
    header_l = _check_seal_dir(lhs_dir)
    if header_t.get("system") != "b0-template" or header_l.get("system") != "b0-lhs":
        raise BaselineError(
            "report: %s is a %s campaign and %s is a %s campaign "
            "(--template needs b0-template, --lhs b0-lhs)"
            % (template_dir, header_t.get("system"), lhs_dir, header_l.get("system")))
    gates = schema.load_gates()
    knobs = schema.load_knobs()
    sysrep = {"b0-template": _system_report(template_dir, header_t, gates, knobs,
                                            expect_n),
              "b0-lhs": _system_report(lhs_dir, header_l, gates, knobs, expect_n)}
    checks = {}
    for system in SYSTEMS:
        blk = sysrep[system]["blk"]
        geoms = sysrep[system]["geoms"]
        hdr = header_t if system == "b0-template" else header_l
        ids_end = [g["geometry_id"] for g in geoms]
        ids_hdr = hdr["geometry_ids"]
        complete = sorted(ids_end) == sorted(ids_hdr) \
            and len(ids_end) == len(set(ids_end))
        if expect_n is not None:
            complete = complete and blk["n_geometries"] == expect_n
        checks["complete_" + system] = complete
    checks["harness_errors_zero"] = all(
        sysrep[s]["blk"]["harness_errors"] == 0 for s in SYSTEMS)
    checks["orphans_zero"] = all(sysrep[s]["blk"]["orphans"] == 0 for s in SYSTEMS)
    checks["peak_ok"] = all(sysrep[s]["blk"]["peak_frac"] <= 0.60 for s in SYSTEMS)
    checks["mesher_cap_ok"] = all(
        sysrep[s]["blk"]["max_live_mesher"] <= campaign.MAX_STREAMS for s in SYSTEMS)
    checks["replay_ok"] = all(sysrep[s]["blk"]["replay_ok"] for s in SYSTEMS)
    checks["audit_ok"] = all(sysrep[s]["blk"]["audit_ok"] for s in SYSTEMS)
    checks["audit_reruns_equal"] = all(
        sysrep[s]["blk"]["audit_reruns_equal"] == sysrep[s]["blk"]["audit_reruns"]
        for s in SYSTEMS)
    entries = _read_seal_lock(report_dir).get("entries") or {}
    checks["sealed_present"] = all(s in entries for s in SYSTEMS)
    checks["sealed_files_match"] = checks["sealed_present"] and all(
        os.path.isfile(os.path.join(report_dir, entries[s]["file"]))
        and _sha256_of_file(os.path.join(report_dir, entries[s]["file"]))
        == entries[s]["sha256"] for s in SYSTEMS)
    checks["one_binary"] = all(
        sysrep["b0-template"]["blk"]["binary_sha256"]
        == sysrep[s]["blk"]["binary_sha256"] for s in SYSTEMS) \
        and checks["sealed_present"] and all(
            entries[s]["binary_sha256"]
            == sysrep["b0-template"]["blk"]["binary_sha256"] for s in SYSTEMS)
    verdict = "PASS" if all(checks.values()) else \
        ("PARTIAL" if allow_partial else "FAIL")
    return _report_result(sysrep, checks, verdict, entries, report_dir, write)


def _report_result(sysrep, checks, verdict, entries, report_dir, write):
    fams = sorted({f for f, _ in _group_keys(sysrep["b0-template"]["geoms"])[0]}
                  | {f for f, _ in _group_keys(sysrep["b0-lhs"]["geoms"])[0]})
    labels = fams + ["tier1", "non-plane", "all"]
    headline = {"capability_limited": {}, "capability_limited_fclean": {},
                "blc8": {}, "mfr": {}}
    m4 = {}
    for system in SYSTEMS:
        by_key = {(g["family"], g["stratum"]): g
                  for g in sysrep[system]["groups"]}
        hc, hf, hb, hm = {}, {}, {}, {}
        for label in labels:
            g = by_key.get((label, "all"))
            if g is None:
                continue
            hc[label] = g["area"]["capability_limited"]
            hf[label] = g["area_fclean"]
            hb[label] = g["blc8_mean"]
            hm[label] = g["mfr"]
            if system == "b0-lhs":
                m4[label] = g["mean4"]["area"]["capability_limited"]
        headline["capability_limited"][system] = hc
        headline["capability_limited_fclean"][system] = hf
        headline["blc8"][system] = hb
        headline["mfr"][system] = hm
    if m4:
        headline["capability_limited_mean4"] = {"b0-lhs": m4}
    result = {"$comment": HEADER, "schema": REPORT_SCHEMA, "date": _today(),
              "verdict": verdict, "checks": checks,
              "binary_sha256": sysrep["b0-template"]["blk"]["binary_sha256"],
              "systems": {}, "headline": headline, "sealed": entries,
              "departures": list(DEPARTURES)}
    for system in SYSTEMS:
        if write:
            info = _write_or_match(report_dir, "tuning_%s.json.gz" % system,
                                   bundle_dir(sysrep[system]["cdir"]))
        else:
            info = {"file": "tuning_%s.json.gz" % system, "sha256": None,
                    "content_sha256": None, "bytes": None}
        result["systems"][system] = {"campaign": dict(sysrep[system]["blk"]),
                                     "groups": sysrep[system]["groups"],
                                     "bundle": info}
    if write:
        _dump_json(os.path.join(report_dir, "B0.json"), result)
        _write_text(os.path.join(report_dir, "B0.md"), report_md(result))
    return result


def _pct(x):
    return "-" if x is None else "%.1f%%" % (100.0 * x)


def _rate(x):
    return "-" if x is None else "%.3f" % x


def _ci(ci):
    return "-" if ci is None or ci[0] is None else "[%.3f, %.3f]" % (ci[0], ci[1])


def _cells(x):
    return "-" if x is None else "%.0f" % x


def _sec(x):
    return "-" if x is None else "%.1f" % x


def _labels_of(rep):
    fams = sorted(set(rep["headline"]["capability_limited"]["b0-template"])
                  | set(rep["headline"]["capability_limited"]["b0-lhs"]))
    known = [f for f in fams if f not in ("tier1", "non-plane", "all")]
    return known + ["tier1", "non-plane", "all"]


def _report_md_lines(rep):
    L = []
    a = L.append
    a("<!-- %s -->" % HEADER)
    a("# B0 - the baselines on the tuning split (docs/15 §F)")
    a("")
    bad = [k for k in sorted(rep["checks"]) if not rep["checks"][k]]
    a("- date: %s" % rep["date"])
    a("- binary: %s" % rep["binary_sha256"])
    a("- verdict: %s" % rep["verdict"])
    a("- failing checks: %s" % (", ".join(bad) if bad else "none"))
    a("")
    a("## CAPABILITY-LIMITED wall area (the AM-L decision)")
    a("")
    a("| label | B0-template | B0-template F-clean | B0-LHS best | B0-LHS mean of 4 |")
    a("|---|---|---|---|---|")
    m4 = (rep["headline"].get("capability_limited_mean4") or {}).get("b0-lhs") or {}
    for label in _labels_of(rep):
        hc = rep["headline"]["capability_limited"]
        a("| %s | %s | %s | %s | %s |" % (
            label,
            _pct(hc["b0-template"].get(label)),
            _pct(rep["headline"]["capability_limited_fclean"]["b0-template"].get(label)),
            _pct(hc["b0-lhs"].get(label)),
            _pct(m4.get(label))))
    a("")
    a("CAPABILITY-LIMITED wall area is the mean share of each geometry's STL wall "
      "area, a priori, whose requested layer patch was dropped min_thickness or "
      "retreat_snapped on a geometry the R-PLANE predicate does not qualify for "
      "(docs/15 §D.2: the denominator is all STL wall area), counted whatever the "
      "F flags say; the F-clean column counts it only on geometries with no F "
      "flag, where remedies would end CAPABILITY-LIMITED at once.")
    a("")

    def grp_rows(system):
        out = []
        for g in rep["systems"][system]["groups"]:
            label = "%s/%s" % (g["family"], g["stratum"])
            f = g["flags"]
            cells = "| %s | %d | %s %s | %s %s | %s | %s | %s | %s | %s | %d | %d | %d | %d |"
            vals = [label, g["n"], _rate(g["mfr"]), _ci(g["mfr_ci"]),
                    _rate(g["strict_rate"]), _ci(g["strict_ci"]),
                    _rate(g["blc8_mean"]), _rate(g["blc_full_mean"]),
                    _rate(g["blc_beta_mean"].get("0.8")),
                    _cells(g["cells_median"]), _sec(g["mesher_seconds_median"]),
                    f["F1"], f["F3"], f["F4"], f["F5"]]
            if system == "b0-lhs":
                cells += " %s | %s |"
                vals += [_rate(g["mean4"]["fail_frac"]), _rate(g["mean4"]["blc8"])]
            out.append(cells % tuple(vals))
        return out
    for system, title in (("b0-template", "## B0-template"),
                          ("b0-lhs", "## B0-LHS (best of 4)")):
        a(title)
        a("")
        head = ("| group | n | MFR [95 % CI] | strict [95 % CI] | BLC_8 | BLC_full "
                "| BLC_0.8 | cells median | mesher s median | F1 | F3 | F4 | F5 |")
        head += " MFR mean of 4 | BLC_8 mean of 4 |" if system == "b0-lhs" else ""
        a(head)
        a("|" + "---|" * (15 if system == "b0-lhs" else 13))
        L.extend(grp_rows(system))
        a("")
    a("## Where the wall area went")
    a("")
    for system in SYSTEMS:
        a("### %s" % system)
        a("")
        a("| label | delivered | capability_limited | plane_fixable | "
          "no_full_stack | other | no_mesh |")
        a("|---|---|---|---|---|---|---|")
        for g in rep["systems"][system]["groups"]:
            label = "%s/%s" % (g["family"], g["stratum"])
            if g["stratum"] != "all" or label.count("/") != 1:
                continue
            a("| %s | %s |" % (label, " | ".join(
                _pct(g["area"][c]) for c in CATEGORIES)))
        a("")
    a("## Failure classes")
    a("")
    for system in SYSTEMS:
        g = next(g for g in rep["systems"][system]["groups"]
                 if (g["family"], g["stratum"]) == ("all", "all"))
        classes = ", ".join("%s %d" % (k, g["failure_classes"][k])
                            for k in sorted(g["failure_classes"]))
        a("- %s: %s" % (system, classes))
    a("")
    a("## The sealed test baselines")
    a("")
    sealed = rep.get("sealed") or {}
    if not sealed:
        a("- none yet: the supervisor's test runs are not sealed")
    for system in sorted(sealed):
        e = sealed[system]
        a("- %s: %s sha256 %s n %d binary %s" % (
            system, e["file"], e["sha256"], e["n_geometries"],
            e["binary_sha256"]))
    a("")
    a("Not opened before the evaluation unit (docs/15 §F).")
    a("")
    a("## Departures")
    a("")
    for d in rep["departures"]:
        a("- %s" % d)
    return L


def report_md(rep):
    return "\n".join(_report_md_lines(rep)) + "\n"


# --- (C6) the seal -------------------------------------------------------------

def _split_lock():
    return split.read_lock(os.path.join(split.MANIFEST_DIR, "split.lock"))


def seal(system, out, *, report_dir=REPORT_DIR, require_test=True, remove=True):
    def refuse(why):
        raise BaselineError("seal: %s" % why)
    if system not in SYSTEMS:
        refuse("system %r is not one of %s" % (system, ", ".join(SYSTEMS)))
    cpath = os.path.join(out, campaign.FILES["campaign"])
    if not os.path.isfile(cpath):
        refuse("%s is missing" % cpath)
    header = _load_json(cpath, "seal")
    if header.get("system") != system:
        refuse("%s is a %s campaign, not %s" % (cpath, header.get("system"), system))
    if header.get("split_mode") != split.EVALUATE:
        refuse("%s is a %s campaign; only mode evaluate seals the test baseline"
               % (cpath, header.get("split_mode")))
    if require_test:
        manifest = header.get("manifest") or {}
        slock = _split_lock()
        if manifest.get("source") != "test":
            refuse("the campaign's manifest source is %r, not the test split"
                   % (manifest.get("source"),))
        if manifest.get("sha256") != slock["test"].split(" ")[-1]:
            refuse("the campaign's manifest sha256 is not the sealed test "
                   "manifest's %s" % (slock["test"].split(" ")[-1],))
        ids = sorted(header.get("geometry_ids") or [])
        if ids != sorted(slock["test_ids"].split()):
            refuse("the campaign covers %d geometries, the sealed test split "
                   "has %d" % (len(ids), len(slock["test_ids"].split())))
    epath = os.path.join(out, campaign.FILES["end"])
    if not os.path.isfile(epath):
        refuse("not finished: %s is missing" % epath)
    end = _load_json(epath, "seal")
    geoms = campaign.load_geometries(out)
    ids_end = [g["geometry_id"] for g in geoms]
    ids_hdr = header["geometry_ids"]
    if sorted(ids_end) != sorted(ids_hdr) or len(ids_end) != len(set(ids_end)):
        refuse("the end records do not match the campaign's geometries exactly "
               "once each")
    if end["harness_errors"] > 0:
        refuse("%d harness error(s): fix the harness and run the campaign again"
               % end["harness_errors"])
    if end["orphans"]:
        refuse("%d orphaned mesher process(es)" % len(end["orphans"]))
    rep = campaign.replay(out)
    if not rep["ok"]:
        refuse("replay does not reproduce the campaign's decisions")
    summary = campaign.summarise_campaign(out)
    if not summary.get("audit") or not summary["audit"]["ok"]:
        refuse("the campaign audit is not ok")
    lock = _read_seal_lock(report_dir)
    if not lock:
        lock = {"$comment": HEADER, "schema": SEAL_SCHEMA, "entries": {}}
    entries = lock.setdefault("entries", {})
    if system in entries:
        refuse("%s is already sealed" % system)
    bpath = os.path.join(report_dir, SEALED_SUBDIR, "test_%s.json.gz" % system)
    if os.path.exists(bpath):
        refuse("%s exists" % bpath)
    w = write_bundle(bundle_dir(out), bpath)
    if _sha256_of_file(bpath) != w["sha256"]:
        os.remove(bpath)
        refuse("the written bundle's bytes do not hash to %s" % w["sha256"])
    rb = read_bundle(bpath)
    if hashlib.sha256(_canonical(rb)).hexdigest() != w["content_sha256"]:
        os.remove(bpath)
        refuse("the written bundle does not round-trip to the same content")
    sample = summary["audit_sample"]
    entry = {"file": SEALED_SUBDIR + "/test_%s.json.gz" % system,
             "sha256": w["sha256"], "content_sha256": w["content_sha256"],
             "bytes": w["bytes"], "n_geometries": len(ids_hdr),
             "campaign_id": header["campaign_id"],
             "binary_sha256": header["binary_sha256"], "git_sha": header["git_sha"],
             "manifest_sha256": header["manifest"]["sha256"],
             "gates_sha256": header["gates_sha256"],
             "knobs_sha256": header["knobs_sha256"], "replay_ok": True,
             "audit_ok": True, "audit_reruns": sample["n"],
             "audit_reruns_equal": sample["equal"], "harness_errors": 0,
             "orphans": 0, "wall_seconds": end["wall_seconds"],
             "peak_rss_mib": end["peak_rss_mib"], "peak_frac": end["peak_frac"],
             "max_live_mesher": end["max_live_mesher"],
             "sealed_at": schema._now_iso()}
    entries[system] = entry
    _dump_json(_seal_lock_path(report_dir), lock)
    if remove:
        shutil.rmtree(out)
    return entry


def load_sealed(system, mode, *, report_dir=REPORT_DIR):
    """The ONLY reader of a sealed bundle; mode evaluate or nothing."""
    if mode != split.EVALUATE:
        raise split.SplitSealed(
            "the sealed baselines open only in mode evaluate (docs/15 §F)")
    entries = _read_seal_lock(report_dir).get("entries") or {}
    if system not in entries:
        raise BaselineError("load_sealed: %s is not sealed in %s"
                            % (system, _seal_lock_path(report_dir)))
    entry = entries[system]
    path = os.path.join(report_dir, entry["file"])
    if not os.path.isfile(path):
        raise BaselineError("load_sealed: %s is missing" % path)
    if _sha256_of_file(path) != entry["sha256"]:
        raise BaselineError("load_sealed: sha256 mismatch for %s" % path)
    return read_bundle(path)


# --- (C7) run ------------------------------------------------------------------

def run(split_name, system, out, *, streams=6, resume=False, binary=None,
        ids=None, limit=None, report_dir=REPORT_DIR):
    if split_name not in SPLITS:
        raise BaselineError("split: %r is not one of %s"
                            % (split_name, ", ".join(SPLITS)))
    if system not in SYSTEMS:
        raise BaselineError("system: %s is not a baseline system (b0-template, "
                            "b0-lhs)" % system)
    binary = binary or campaign.BINARY_DEFAULT
    if split_name == "tuning":
        return campaign.run_campaign(
            {"manifest": "tuning", "mode": system, "out": out,
             "run_id": system + "-tuning", "streams": streams, "resume": resume,
             "binary": binary, "ids": ids, "limit": limit, "quiet": False})
    if ids or limit:
        raise BaselineError("the sealed test baseline covers the whole test "
                            "split (no --ids / --limit)")
    stop = threading.Event()

    def tick():
        while not stop.wait(60.0):
            try:
                with open(os.path.join(out, campaign.FILES["progress"]),
                          encoding="utf-8") as f:
                    pr = json.load(f)
            except (ValueError, OSError):
                continue
            print("[baseline test %s] %s/%s geometries, %s running, peak %.0f MiB"
                  % (system, pr.get("n_done", 0), pr.get("n_total", 0),
                     pr.get("n_live", 0), pr.get("peak_rss_mib", 0.0)), flush=True)
    th = threading.Thread(target=tick, daemon=True)
    th.start()
    try:
        campaign.run_campaign({"manifest": "test", "mode": split.EVALUATE,
                               "system": system, "out": out,
                               "run_id": system + "-test", "streams": streams,
                               "resume": resume, "binary": binary, "quiet": True})
    finally:
        stop.set()
        th.join(5.0)
    entry = seal(system, out, report_dir=report_dir)
    print("[baseline test %s] sealed %s sha256 %s (%d geometries, binary %s)"
          % (system, entry["file"], entry["sha256"], entry["n_geometries"],
             entry["binary_sha256"][:12]), flush=True)
    return entry


# --- (C8) R-CURV's F3 cost ------------------------------------------------------

RCURV_LOCK = threading.Lock()


def setup_without_rcurv(mrow, fp, stl, case, name, gates, knobs):
    """rules.setup with R-CURV swapped for an abstention; the swap never
    outlives this call (a module lock keeps other setup callers out)."""
    with RCURV_LOCK:
        orig = rules.RULE_FN["R-CURV"]

        def off(state):
            return rules._rec("R-CURV", "abstain", None, [], "", [], RCURV_CITE,
                              RCURV_MSG)
        rules.RULE_FN["R-CURV"] = off
        try:
            res = rules.setup(mrow, fp, stl, case, name, gates=gates, knobs=knobs)
        finally:
            rules.RULE_FN["R-CURV"] = orig
    assert rules.RULE_FN["R-CURV"] is orig
    return res


def _rcurv_arm(setup, probe, pf, res):
    oc = res["outcome"]
    return {"wall_level": setup["summary"]["wall_level"],
            "h_over_t1": setup["summary"]["h_over_t1"],
            "config_sha256": setup["config_sha256"],
            "n_leaves": probe.get("n_leaves"),
            "preflight_refused": list(pf["refused"]),
            "failure": oc["failure"], "strict_failure": oc["strict_failure"],
            "failure_class": oc["failure_class"], "flags": oc["flags"],
            "f3": any(oc.get("flags", {}).get("F3" + s) is True
                      for s in ("a", "b", "c", "d")),
            "pinned_frac": oc["pinned_frac"], "p99_over_hf": oc["p99_over_hf"],
            "max_over_hf": oc["max_over_hf"], "n_cells": oc["n_cells"],
            "seconds": oc["seconds"], "blc8_a_priori": oc["blc8_a_priori"],
            "capability_limited": area_split(oc, False)["capability_limited"]}


def rcurv(out, *, streams=6, binary=None, ids=None, limit=None,
          report_dir=REPORT_DIR, write=True):
    if os.path.isdir(out) and os.listdir(out):
        raise BaselineError("not empty: %s" % out)
    os.makedirs(out, exist_ok=True)
    all_rows = campaign.load_manifest("tuning", "rules", ids=ids, limit=limit)
    rows = [r for r in all_rows if r["family"] in RCURV_FAMILIES]
    skips = []
    skipped = {}

    def skip(gid, reason, kind):
        skipped[kind] = skipped.get(kind, 0) + 1
        skips.append({"geometry_id": gid, "reason": reason})
    for r in all_rows:
        if r["family"] not in RCURV_FAMILIES:
            skip(r["geometry_id"], "family", "family")
    c = campaign.Campaign(out, campaign_id="rcurv", mode="rules", system="rules",
                          ablate=(), split_mode="rules",
                          binary=binary or campaign.BINARY_DEFAULT,
                          streams=streams, timeout_s=campaign.TIMEOUT_S,
                          probe_timeout_s=campaign.PROBE_TIMEOUT_S,
                          audit_mod=RCURV_AUDIT_OFF, quiet=False)
    gates = schema.load_gates()
    knobs = schema.load_knobs()
    c.start()
    t0 = time.perf_counter()
    pairs = []
    try:
        def _obs(mrow):
            return campaign.observe(c, mrow)
        with ThreadPoolExecutor(max_workers=streams) as ex:
            obs_by = {r["geometry_id"]: o
                      for r, o in zip(rows, ex.map(_obs, rows))}
        for mrow in rows:
            gid = mrow["geometry_id"]
            obs = obs_by.get(gid) or {}
            if obs.get("terminal") is not None:
                skip(gid, "surface", "surface")
                continue
            fp = obs["fingerprint"]
            a = rules.setup(mrow, fp, campaign.stl_rel(gid),
                            campaign.case_rel(gid), gid, gates=gates, knobs=knobs)
            b = setup_without_rcurv(mrow, fp, campaign.stl_rel(gid),
                                    campaign.case_rel(gid), gid, gates, knobs)
            ra = next(r for r in a["records"] if r["rule_id"] == "R-CURV")
            if ra["verdict"] != "apply":
                skip(gid, "R-CURV " + ra["verdict"], "rcurv_" + ra["verdict"])
                continue
            if a["verdict"] == "refuse" or b["verdict"] == "refuse":
                skip(gid, "refused", "refused")
                continue
            if a["config_sha256"] == b["config_sha256"]:
                skip(gid, "same config", "same config")
                continue
            pairs.append({"geometry_id": gid, "family": mrow["family"],
                          "stratum": mrow["stratum"], "mrow": mrow, "fp": fp,
                          "obs": obs, "cfg_with": a["config"],
                          "cfg_without": b["config"], "st_with": a, "st_without": b})

        def arm_of(pair, which, k):
            cfg = pair["cfg_" + which]
            gctx = {"gid": pair["geometry_id"], "areas": pair["obs"]["areas"],
                    "flow": pair["mrow"]["flow"], "audit": []}
            probe = campaign.octree_probe(c, gctx, cfg, k)
            pf = preflight.preflight(cfg, fingerprint=pair["fp"],
                                     flow=pair["mrow"]["flow"],
                                     octree_probe=probe, gates=gates,
                                     knobs=knobs, cwd=c.dir)
            campaign.write_config(c, pair["geometry_id"], k, cfg)
            res = campaign.run_attempt(c, gctx, cfg, k, probe.get("n_leaves"))
            return _rcurv_arm(pair["st_" + which], probe, pf, res)

        def one(pair):
            try:
                pair["arm_with"] = arm_of(pair, "with", 1)
                pair["arm_without"] = arm_of(pair, "without", 2)
            except Exception as e:
                pair["error"] = "%s: %s" % (type(e).__name__, e)
        with ThreadPoolExecutor(max_workers=streams) as ex:
            list(ex.map(one, pairs))
    finally:
        c.kill_all()
        c.stop()
    wall = time.perf_counter() - t0
    for p in pairs:
        if "error" in p:
            skip(p["geometry_id"], "harness error: " + p["error"],
                 "harness_error")
    out_pairs = [{"geometry_id": p["geometry_id"], "family": p["family"],
                  "stratum": p["stratum"], "with": p["arm_with"],
                  "without": p["arm_without"]} for p in pairs
                 if "error" not in p]
    rep = {"$comment": HEADER, "schema": RCURV_SCHEMA, "date": _today(),
           "binary_sha256": c.binary_sha, "git_sha": c.git_sha,
           "n_rows": len(all_rows), "n_pairs": len(out_pairs),
           "skipped": skipped, "harness_errors": skipped.get("harness_error", 0),
           "groups": rcurv_summary(out_pairs),
           "pairs": out_pairs, "skips": skips, "peak_rss_mib": c.peak_rss_mib,
           "max_live_mesher": c.max_live_mesher, "orphans": len(c.orphans()),
           "wall_seconds": wall}
    if write:
        os.makedirs(report_dir, exist_ok=True)
        _dump_json(os.path.join(report_dir, "R-CURV.json"), rep)
        _write_text(os.path.join(report_dir, "R-CURV.md"), rcurv_md(rep))
    return rep


def _rcurv_side(arms):
    def med(key):
        vals = [a[key] for a in arms if a.get(key) is not None]
        return statistics.median(vals) if vals else None
    n = len(arms)
    return {"wall_level_median": med("wall_level"),
            "f3": sum(1 for x in arms if x["f3"]),
            "failure": sum(1 for x in arms if x["failure"]),
            "strict": sum(1 for x in arms if x["strict_failure"]),
            "f1": sum(1 for x in arms
                      if (x.get("flags") or {}).get("F1") is True),
            "preflight_refused": sum(1 for x in arms if x.get("preflight_refused")),
            "cells_median": med("n_cells"),
            "seconds_median": med("seconds"),
            "blc8_mean": (sum(x["blc8_a_priori"] for x in arms) / n) if n else None,
            "capability_limited_mean":
                (sum(x["capability_limited"] for x in arms) / n) if n else None}


def rcurv_summary(pairs):
    fams = sorted({p["family"] for p in pairs})
    out = []
    for fam in fams + ["all"]:
        ms = [p for p in pairs if fam == "all" or p["family"] == fam]
        if not ms:
            continue
        g = {"family": fam, "n": len(ms),
             "with": _rcurv_side([p["with"] for p in ms]),
             "without": _rcurv_side([p["without"] for p in ms])}

        def count(k, w, wo):
            return sum(1 for p in ms
                       if bool(p["with"][k]) == w and bool(p["without"][k]) == wo)
        for pre, key in (("f3", "f3"), ("fail", "failure")):
            g[pre + "_both"] = count(key, True, True)
            g[pre + "_with_only"] = count(key, True, False)
            g[pre + "_without_only"] = count(key, False, True)
            g[pre + "_neither"] = count(key, False, False)
        out.append(g)
    return out


def rcurv_md(rep):
    L = []
    a = L.append
    a("<!-- %s -->" % HEADER)
    a("# R-CURV - its F3 cost on the tuning split")
    a("")
    a("- date: %s" % rep["date"])
    a("- binary: %s" % rep["binary_sha256"])
    a("- pairs run: %d of %d rows; skipped: %s"
      % (rep["n_pairs"], rep["n_rows"],
         ", ".join("%s %d" % (k, v) for k, v in sorted(rep["skipped"].items()))
         or "none"))
    a("- harness errors %d" % rep.get("harness_errors", 0))
    a("- peak rss %.0f MiB, max live mesher %d, orphans %d, wall %.1f s"
      % (rep["peak_rss_mib"], rep["max_live_mesher"], rep["orphans"],
         rep["wall_seconds"]))
    a("")
    a("| family | n | level with | level without | F3 with | F3 without | "
      "F3 with-only | F3 without-only | failure with | failure without | "
      "strict with | strict without | refusals with | refusals without | "
      "cells median with | cells median without | s median with | "
      "s median without | capability-limited area with | "
      "capability-limited area without |")
    a("|" + "---|" * 20)
    for g in rep["groups"]:
        w, o = g["with"], g["without"]
        a("| %s | %d | %s | %s | %d | %d | %d | %d | %d | %d | %d | %d | %d | "
          "%d | %s | %s | %s | %s | %.3f | %.3f |" % (
              g["family"], g["n"],
              _cells(w["wall_level_median"]), _cells(o["wall_level_median"]),
              w["f3"], o["f3"], g["f3_with_only"], g["f3_without_only"],
              w["failure"], o["failure"], w["strict"], o["strict"],
              w["preflight_refused"], o["preflight_refused"],
              _cells(w["cells_median"]), _cells(o["cells_median"]),
              _sec(w["seconds_median"]), _sec(o["seconds_median"]),
              w["capability_limited_mean"], o["capability_limited_mean"]))
    a("")
    a("Reported, not gated: the user decides keep / retune / drop "
      "(DECISIONS 2026-09-24).")
    return "\n".join(L) + "\n"


# --- (C9) check -----------------------------------------------------------------

def check(report_dir=REPORT_DIR):
    """The committed files, without opening a sealed bundle."""
    items = []

    def add(name, ok, why):
        items.append({"name": name, "ok": bool(ok), "why": why})
    b0p = os.path.join(report_dir, "B0.json")
    rep = None
    if not os.path.isfile(b0p):
        add("B0.json", False, b0p)
    else:
        try:
            with open(b0p, encoding="utf-8") as f:
                rep = json.load(f)
            add("B0.json", True, b0p)
        except (ValueError, OSError) as e:
            add("B0.json", False, "%s: %s" % (b0p, e))
    add("B0 verdict", rep is not None and rep.get("verdict") == "PASS",
        b0p if rep is None else "verdict %s" % rep.get("verdict"))
    ok = rep is not None
    why = []
    for system in SYSTEMS:
        fname = "tuning_%s.json.gz" % system
        path = os.path.join(report_dir, fname)
        want = (((rep or {}).get("systems") or {}).get(system)
                or {}).get("bundle", {}).get("sha256")
        if not os.path.isfile(path):
            ok = False
            why.append("%s is missing" % path)
        elif want is None:
            ok = False
            why.append("%s: no sha256 recorded in B0.json" % fname)
        else:
            got = _sha256_of_file(path)
            if got != want:
                ok = False
                why.append("%s sha256 %s != %s" % (fname, got, want))
    add("tuning bundles", ok, "; ".join(why)
        or os.path.join(report_dir, "tuning_b0-*.json.gz"))
    lock = {}
    lockp = _seal_lock_path(report_dir)
    if not os.path.isfile(lockp):
        add("sealed.lock", False, lockp)
    else:
        try:
            with open(lockp, encoding="utf-8") as f:
                lock = json.load(f)
        except ValueError as e:
            add("sealed.lock", False, "%s: %s" % (lockp, e))
        else:
            n = len(lock.get("entries") or {})
            add("sealed.lock", all(s in (lock.get("entries") or {})
                                   for s in SYSTEMS),
                "%s (%d entries)" % (lockp, n))
    entries = lock.get("entries") or {}
    ok = bool(entries)
    why = []
    for system in sorted(entries):
        entry = entries[system]
        path = os.path.join(report_dir, entry.get("file", ""))
        if not os.path.isfile(path):
            ok = False
            why.append("%s is missing" % path)
            continue
        got = _sha256_of_file(path)
        if got != entry.get("sha256"):
            ok = False
            why.append("%s sha256 %s != %s" % (entry.get("file"), got,
                                               entry.get("sha256")))
    add("sealed files", ok, "; ".join(why) or lockp)
    bin_sha = (rep or {}).get("binary_sha256")
    add("one binary", bool(bin_sha) and all(
        entries.get(s, {}).get("binary_sha256") == bin_sha for s in SYSTEMS),
        bin_sha or "no binary_sha256 in B0.json")
    rcp = os.path.join(report_dir, "R-CURV.json")
    if not os.path.isfile(rcp):
        add("R-CURV.json", False, rcp)
    else:
        try:
            with open(rcp, encoding="utf-8") as f:
                rrep = json.load(f)
        except (ValueError, OSError) as e:
            add("R-CURV.json", False, "%s: %s" % (rcp, e))
        else:
            add("R-CURV.json", bool(bin_sha) and rrep.get("binary_sha256") == bin_sha,
                rcp if rrep.get("binary_sha256") == bin_sha
                else "%s: binary_sha256 differs from B0.json" % rcp)
    good = all(i["ok"] for i in items)
    return {"$comment": HEADER, "schema": CHECK_SCHEMA, "items": items,
            "verdict": "PASS" if good else "FAIL"}


# --- (C10) the selftest ---------------------------------------------------------

_IDS5 = ("D-1-010", "F-1-009", "G-1-016", "F-1-005", "G-1-026")
_SCRIPT_T = {"D-1-010": ["retreat_snapped"], "F-1-009": ["min_thickness"],
             "G-1-016": ["F3a"], "F-1-005": ["pass"], "G-1-026": []}
_SCRIPT_L = {"D-1-010": ["F3a", "retreat_snapped", "pass", "F3a"],
             "F-1-009": ["min_thickness"]}
_SCRIPT_E = {"D-1-010": ["retreat_snapped"], "F-1-009": ["pass"]}


def _run_group(n, fn, *args):
    try:
        line = fn(*args)
    except Exception as e:
        raise AssertionError("group %d failed: %s: %s"
                             % (n, type(e).__name__, e)) from e
    print("[ok] %s" % line, flush=True)


def _g1(tmp):
    x = os.path.join(tmp, "g1refused")
    cases = [(("tuning", "rules", x), {}, "system"),
             (("nope", "b0-template", x), {}, "split"),
             (("test", "b0-template", x), {"ids": ["A-1-008"]}, "whole test split"),
             (("test", "b0-template", x), {"limit": 3}, "whole test split")]
    for args, kwargs, needle in cases:
        try:
            run(*args, **kwargs)
        except BaselineError as e:
            assert needle in str(e), str(e)
        else:
            raise AssertionError("run(%r, %r) was not refused" % (args, kwargs))
    assert not os.path.exists(x)
    try:
        load_sealed("b0-template", "rules", report_dir=os.path.join(tmp, "absent"))
    except split.SplitSealed:
        pass
    else:
        raise AssertionError("load_sealed outside evaluate was not refused")
    assert len(SYSTEMS) == 2 and len(CATEGORIES) == 6
    assert TIER1 == ("A", "B", "E", "F")
    assert DROP_CLASSES == tuple(k.split(":", 1)[1] for k in remedies.DROP_KEYS)
    return ("constants and refusals: 2 systems, 6 categories, tier1 A B E F; run "
            "refuses a non-baseline system, an unknown split and a partial test "
            "run before any directory; load_sealed refuses outside evaluate "
            "before any file")


def _g2(geoms_by, rows_by, q, mrows, fps):
    for gid, want in (("D-1-010", "plane_fixable"), ("F-1-009", "capability_limited"),
                      ("G-1-016", "delivered"), ("F-1-005", "delivered"),
                      ("G-1-026", "no_mesh")):
        g = geoms_by[gid]
        ar = geometry_areas(g, rows_by.get(gid, {}), q[gid])
        assert ar["final"][want] > 1 - 1e-12, (gid, ar["final"])
        assert abs(sum(ar["final"].values()) - 1.0) <= 1e-12
        assert abs(sum(ar["mean4"].values()) - 1.0) <= 1e-12

    def patch(name, area, **kw):
        row = {"name": name, "area_m2": area, "requested": False, "n_layers": 0,
               "dropped": None, "full_area_frac": None, "t1_requested_m": None,
               "yplus_a_priori": None, "delivered": False,
               "capability_limited": False, "layer_class": None,
               "mean_frac": None, "t1_min_m": None}
        row.update(kw)
        return row
    oc = copy.deepcopy(rows_by["F-1-009"][1]["outcome"])
    oc["patches"] = [patch("a", 1.0, delivered=True, n_layers=8,
                           full_area_frac=1.0),
                     patch("b", 3.0, requested=True, dropped="x",
                           layer_class="retreat_snapped")]
    s = area_split(oc, False)
    assert abs(s["delivered"] - 0.25) <= 1e-12 and abs(s["capability_limited"] - 0.75) <= 1e-12
    s = area_split(oc, True)
    assert abs(s["plane_fixable"] - 0.75) <= 1e-12 and abs(s["delivered"] - 0.25) <= 1e-12
    oc["patches"][1]["layer_class"] = "no_full_stack"
    s = area_split(oc, False)
    assert abs(s["no_full_stack"] - 0.75) <= 1e-12
    oc2 = remedies.synthetic_outcome(
        "config", campaign.b0_template(mrows["F-1-009"], fps["F-1-009"]),
        fps["F-1-009"], mrows["F-1-009"]["flow"])
    assert area_split(oc2, False)["other"] > 1 - 1e-12
    try:
        area_split({"patches": [patch("a", 0.0)]}, False)
    except BaselineError:
        pass
    else:
        raise AssertionError("area_split accepted a zero total area")
    return "area split: 5 fake geometries and 3 hand-built outcomes, every split sums to 1"


def _g3(q, mrows, gates, knobs):
    assert q["D-1-010"] is True and q["G-1-016"] is True
    assert q["F-1-009"] is False and q["F-1-005"] is False
    assert qualifies(mrows["D-1-010"], None, gates, knobs) is None
    return ("R-PLANE predicate: D-1-010 True, G-1-016 True, F-1-009 False, "
            "F-1-005 False, no fingerprint None")


def _g4(tmp):
    repd = os.path.join(tmp, "rep")
    rep = report(os.path.join(tmp, "t"), os.path.join(tmp, "l"),
                 report_dir=repd, expect_n=None, allow_partial=True)
    tg = {(g["family"], g["stratum"]): g
          for g in rep["systems"]["b0-template"]["groups"]}
    a_ = tg[("all", "all")]
    assert a_["n"] == 5, a_
    assert a_["fail"] == 2, a_
    assert abs(a_["mfr"] - 0.4) <= 1e-12, a_
    assert a_["strict"] == 4, a_
    assert abs(a_["area"]["delivered"] - 0.4) <= 1e-12, a_["area"]
    assert abs(a_["area"]["capability_limited"] - 0.2) <= 1e-12, a_["area"]
    assert abs(a_["area"]["plane_fixable"] - 0.2) <= 1e-12, a_["area"]
    assert abs(a_["area"]["no_mesh"] - 0.2) <= 1e-12, a_["area"]
    assert abs(a_["area_fclean"] - 0.2) <= 1e-12, a_
    assert a_["n_capability_limited"] == 1, a_
    assert a_["flags"]["F1"] == 1, a_["flags"]
    assert a_["flags"]["F3a"] == 1, a_["flags"]
    assert a_["failure_classes"] == {"SURFACE-OPEN": 1,
                                     "layer_dropped:min_thickness": 1,
                                     "layer_dropped:retreat_snapped": 1,
                                     "pass": 2}, a_["failure_classes"]
    for key in (("tier1", "all"), ("non-plane", "all")):
        g = tg[key]
        assert g["n"] == 2, (key, g["n"])
        assert abs(g["area"]["capability_limited"] - 0.5) <= 1e-12, (key, g["area"])
        assert abs(g["area"]["delivered"] - 0.5) <= 1e-12, (key, g["area"])
    lg = {(g["family"], g["stratum"]): g
          for g in rep["systems"]["b0-lhs"]["groups"]}[("all", "all")]
    assert lg["n"] == 2, lg
    assert lg["fail"] == 0, lg
    assert lg["strict"] == 1, lg
    assert abs(lg["area"]["delivered"] - 0.5) <= 1e-12, lg["area"]
    assert abs(lg["area"]["capability_limited"] - 0.5) <= 1e-12, lg["area"]
    m4 = lg["mean4"]
    assert abs(m4["fail_frac"] - 0.25) <= 1e-12, m4
    assert abs(m4["strict_frac"] - 0.875) <= 1e-12, m4
    assert abs(m4["area"]["delivered"] - 0.375) <= 1e-12, m4
    assert abs(m4["area"]["plane_fixable"] - 0.125) <= 1e-12, m4
    assert abs(m4["area"]["capability_limited"] - 0.5) <= 1e-12, m4
    ck = rep["checks"]
    for key in ("replay_ok", "audit_ok", "harness_errors_zero", "orphans_zero"):
        assert ck[key] is True, key
    assert ck["sealed_present"] is False and ck["one_binary"] is False, ck
    assert rep["verdict"] == "PARTIAL", rep["verdict"]
    assert os.path.isfile(os.path.join(repd, "B0.json")), repd
    assert os.path.isfile(os.path.join(repd, "B0.md")), repd
    for s in SYSTEMS:
        assert os.path.isfile(os.path.join(repd, "tuning_%s.json.gz" % s)), repd
    with open(os.path.join(repd, "B0.md"), encoding="utf-8") as f:
        md = f.read()
    assert "CAPABILITY-LIMITED" in md and "20.0" in md, md[-2000:]
    with open(os.path.join(repd, "B0.json"), "rb") as f:
        first = f.read()
    report(os.path.join(tmp, "t"), os.path.join(tmp, "l"),
           report_dir=repd, expect_n=None, allow_partial=True)
    with open(os.path.join(repd, "B0.json"), "rb") as f:
        assert f.read() == first, "the second report wrote a different B0.json"
    rep3 = report(os.path.join(tmp, "t"), os.path.join(tmp, "l"),
                  report_dir=repd, expect_n=None, allow_partial=False,
                  write=False)
    assert rep3["verdict"] == "FAIL", rep3["verdict"]
    t2 = os.path.join(tmp, "t2")
    shutil.copytree(os.path.join(tmp, "t"), t2)
    gpath = os.path.join(t2, campaign.FILES["geometries"])
    with open(gpath, encoding="utf-8") as f:
        recs = [json.loads(ln) for ln in f if ln.strip()]
    hit = [g for g in recs if g["terminal"] == "SURFACE-OPEN"]
    assert len(hit) == 1, [g["terminal"] for g in recs]

    def rewrite(recs, reason):
        hit[0]["terminal"] = "HARNESS-ERROR"
        hit[0]["reason"] = reason
        with open(gpath, "w", encoding="utf-8", newline=chr(10)) as f:
            for g in recs:
                f.write(json.dumps(g, sort_keys=True,
                                   ensure_ascii=False) + chr(10))
    rewrite(recs, "features.py: surface/degenerate: triangle 1 has zero area")
    epath = os.path.join(t2, campaign.FILES["end"])

    def write_end(harness_errors):
        with open(epath, encoding="utf-8") as f:
            e2 = json.load(f)
        e2["harness_errors"] = harness_errors
        e2["terminals"]["HARNESS-ERROR"] = (
            e2["terminals"].get("HARNESS-ERROR", 0) + 1)
        e2["terminals"]["SURFACE-OPEN"] -= 1
        with open(epath, "w", encoding="utf-8", newline=chr(10)) as f:
            json.dump(e2, f, indent=1, sort_keys=True,
                      ensure_ascii=False)
            f.write(chr(10))
    write_end(1)
    repd2 = os.path.join(tmp, "rep2")
    rep2 = report(t2, os.path.join(tmp, "l"), report_dir=repd2,
                  expect_n=None, allow_partial=True, write=False)
    blk2 = rep2["systems"]["b0-template"]["campaign"]
    assert rep2["checks"]["harness_errors_zero"] is True, rep2["checks"]
    assert (blk2["harness_errors"] == 0 and
            blk2["harness_errors_recorded"] == 1), blk2
    assert blk2["surface_refused"] == [hit[0]["geometry_id"]], blk2
    fc2 = {(g["family"], g["stratum"]): g for g in
           rep2["systems"]["b0-template"]["groups"]}[("all", "all")]
    fc2 = fc2["failure_classes"]
    assert fc2.get("SURFACE-REFUSED") == 1 and "SURFACE-OPEN" not in fc2, fc2
    rewrite(recs, "RuntimeError: boom")
    rep4 = report(t2, os.path.join(tmp, "l"), report_dir=repd2,
                  expect_n=None, allow_partial=True, write=False)
    assert rep4["checks"]["harness_errors_zero"] is False, rep4["checks"]
    with open(epath, encoding="utf-8") as f:
        e3 = json.load(f)
    e3["harness_errors"] = 0
    with open(epath, "w", encoding="utf-8", newline=chr(10)) as f:
        json.dump(e3, f, indent=1, sort_keys=True, ensure_ascii=False)
        f.write(chr(10))
    try:
        report(t2, os.path.join(tmp, "l"), report_dir=repd2,
               expect_n=None, allow_partial=True, write=False)
    except BaselineError as e:
        assert "harness errors" in str(e), str(e)
    else:
        raise BaselineError("the harness-error count mismatch was not refused")
    return ("report on fake campaigns: template MFR 0.400 strict 0.800 "
            "capability-limited 0.200, LHS MFR 0.000 mean-of-4 fail 0.250, "
            "verdict PARTIAL, a features.py surface refusal re-read as "
            "SURFACE-REFUSED keeps harness_errors_zero strict")


def _g5(tmp):
    rows = campaign.load_manifest("tuning", "selftest",
                                  ids=["D-1-010", "F-1-009"])
    fdir = os.path.join(tmp, "rows.jsonl")
    with open(fdir, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")
    E = os.path.join(tmp, "e")
    R = os.path.join(tmp, "rep5")

    def run_file(out, mode="evaluate"):
        return campaign.run_campaign(
            {"manifest": fdir, "mode": mode, "system": "b0-template"
             if mode == "evaluate" else None, "out": out, "streams": 2,
             "quiet": True},
            attempt_fn=campaign._fake_attempt(_SCRIPT_E),
            probe_fn=campaign._fake_probe(), snap_fn=campaign._fake_snap)
    run_file(E)
    try:
        report(E, os.path.join(tmp, "l"), report_dir=R, expect_n=None,
               allow_partial=True)
    except split.SplitSealed as e:
        assert "evaluate" in str(e)
    else:
        raise AssertionError("report accepted an evaluate campaign")
    try:
        seal("b0-template", E, report_dir=R, require_test=True)
    except BaselineError as e:
        assert str(e).startswith("seal:"), str(e)
    else:
        raise AssertionError("seal accepted a non-test campaign with require_test")
    assert os.path.isdir(E)
    entry = seal("b0-template", E, report_dir=R, require_test=False)
    assert entry["n_geometries"] == 2
    assert not os.path.exists(E)
    lock = _read_seal_lock(R)
    assert "b0-template" in lock["entries"]
    b = load_sealed("b0-template", "evaluate", report_dir=R)
    assert len(b["geometries"]) == 2
    E2 = os.path.join(tmp, "e2")
    run_file(E2)
    try:
        seal("b0-template", E2, report_dir=R, require_test=False)
    except BaselineError as e:
        assert "already sealed" in str(e)
    else:
        raise AssertionError("seal wrote twice")
    E3 = os.path.join(tmp, "e3")
    run_file(E3, mode="b0-template")
    try:
        seal("b0-template", E3, report_dir=R, require_test=False)
    except BaselineError as e:
        assert "evaluate" in str(e), str(e)
    else:
        raise AssertionError("seal accepted a b0-template campaign")
    E4 = os.path.join(tmp, "e4")
    shutil.copytree(E2, E4)
    os.remove(os.path.join(E4, campaign.FILES["end"]))
    try:
        seal("b0-template", E4, report_dir=R, require_test=False)
    except BaselineError as e:
        assert "not finished" in str(e), str(e)
    else:
        raise AssertionError("seal accepted an unfinished campaign")
    spath = os.path.join(R, SEALED_SUBDIR, "test_b0-template.json.gz")
    with open(spath, "rb") as f:
        data = f.read()
    flipped = bytearray(data)
    flipped[len(flipped) // 2] ^= 0x01
    with open(spath, "wb") as f:
        f.write(bytes(flipped))
    try:
        load_sealed("b0-template", "evaluate", report_dir=R)
    except BaselineError as e:
        assert "sha256" in str(e), str(e)
    else:
        raise AssertionError("load_sealed accepted a flipped byte")
    with open(spath, "wb") as f:
        f.write(data)
    return ("seal: write-once, verified, directory removed, opened only in "
            "evaluate, a flipped byte refused, a test bundle refused by the report")


def _g6(tmp):
    b = bundle_dir(os.path.join(tmp, "t"))
    p1 = os.path.join(tmp, "b1.json.gz")
    p2 = os.path.join(tmp, "b2.json.gz")
    w1 = write_bundle(b, p1)
    w2 = write_bundle(b, p2)
    with open(p1, "rb") as f:
        d1 = f.read()
    with open(p2, "rb") as f:
        assert f.read() == d1
    assert d1[4:8] == b"\x00\x00\x00\x00", "gzip mtime is not 0"
    assert read_bundle(p1) == b
    assert hashlib.sha256(_canonical(read_bundle(p1))).hexdigest() \
        == w1["content_sha256"] == w2["content_sha256"]
    try:
        write_bundle(b, p1)
    except BaselineError as e:
        assert "exists" in str(e)
    else:
        raise AssertionError("write_bundle overwrote an existing path")
    return ("bundles: two writes byte-identical, gzip mtime 0, round trip equal, "
            "an existing path refused")


def _g7(mrows, obs_b, obs_d, gates, knobs):
    gid = "B-1-011"
    fp = obs_b["fingerprint"]
    stl = campaign.stl_rel(gid)
    case = campaign.case_rel(gid)
    a = rules.setup(mrows[gid], fp, stl, case, gid, gates=gates, knobs=knobs)
    b = setup_without_rcurv(mrows[gid], fp, stl, case, gid, gates, knobs)
    assert a["summary"]["wall_level"] == 6, a["summary"]["wall_level"]
    assert b["summary"]["wall_level"] == 4, b["summary"]["wall_level"]
    rb = next(r for r in b["records"] if r["rule_id"] == "R-CURV")
    assert rb["verdict"] == "abstain"
    assert rb["message"].startswith("R-CURV: ")
    assert rules.RULE_FN["R-CURV"] is rules.r_curv
    dgid = "D-1-010"
    ad = rules.setup(mrows[dgid], obs_d["fingerprint"], campaign.stl_rel(dgid),
                     campaign.case_rel(dgid), dgid, gates=gates, knobs=knobs)
    rd = next(r for r in ad["records"] if r["rule_id"] == "R-CURV")
    assert rd["verdict"] == "abstain"

    def pair(gid2, fam, fw, wo):
        def arm(f3):
            return {"wall_level": 4, "h_over_t1": 2.0,
                    "config_sha256": "0" * 64, "n_leaves": 10,
                    "preflight_refused": [], "failure": False,
                    "strict_failure": False, "failure_class": None,
                    "flags": {"F1": False, "F2": False, "F3a": f3,
                              "F3b": False, "F3c": False, "F3d": False,
                              "F4": False, "F5": False},
                    "f3": f3, "pinned_frac": 0.0, "p99_over_hf": 0.0,
                    "max_over_hf": 0.0, "n_cells": 100, "seconds": 1.0,
                    "blc8_a_priori": 0.5, "capability_limited": 0.25}
        return {"geometry_id": gid2, "family": fam, "stratum": "1",
                "with": arm(fw), "without": arm(wo)}
    pairs = [pair("B-1-001", "B", True, True), pair("B-1-002", "B", True, False),
             pair("D-1-001", "D", False, True), pair("D-1-002", "D", False, False)]
    gs = rcurv_summary(pairs)
    ga = next(g for g in gs if g["family"] == "all")
    assert ga["f3_both"] == 1 and ga["f3_with_only"] == 1
    assert ga["f3_without_only"] == 1 and ga["f3_neither"] == 1
    gb = next(g for g in gs if g["family"] == "B")
    assert gb["n"] == 2
    return ("R-CURV ablation: B-1-011 wall level 6 with, 4 without, the rule "
            "restored; D-1-010 abstains; paired counts on 4 hand-built pairs")


def _g8(tmp):
    repd = os.path.join(tmp, "rep")
    R = os.path.join(tmp, "rep5")
    shutil.copytree(os.path.join(R, SEALED_SUBDIR),
                    os.path.join(repd, SEALED_SUBDIR))
    shutil.copy2(_seal_lock_path(R), _seal_lock_path(repd))
    ck = check(repd)
    names = {i["name"]: i for i in ck["items"]}
    assert ck["verdict"] == "FAIL"
    assert not names["B0 verdict"]["ok"]
    assert names["tuning bundles"]["ok"]
    spath = os.path.join(repd, SEALED_SUBDIR, "test_b0-template.json.gz")
    with open(spath, "rb") as f:
        data = f.read()
    flipped = bytearray(data)
    flipped[len(flipped) // 2] ^= 0x01
    with open(spath, "wb") as f:
        f.write(bytes(flipped))
    ck2 = check(repd)
    names2 = {i["name"]: i for i in ck2["items"]}
    assert not names2["sealed files"]["ok"], names2["sealed files"]
    return ('check: a partial report FAILs on its verdict, a flipped sealed byte '
            'FAILs "sealed files"')


def _g9(tmp):
    A = os.path.join(tmp, "live_t")
    B = os.path.join(tmp, "live_l")
    R2 = os.path.join(tmp, "rep_live")
    X = os.path.join(tmp, "refused_out")

    def py(*args):
        return subprocess.run([sys.executable, __file__] + list(args),
                              capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
    q1 = py("--run", "--split", "tuning", "--system", "b0-template", "--ids",
            "D-1-010,F-1-009", "--out", A, "--streams", "2")
    assert q1.returncode == 0, (q1.stdout + q1.stderr)[-2000:]
    q2 = py("--run", "--split", "tuning", "--system", "b0-lhs", "--ids",
            "B-1-011", "--out", B, "--streams", "2")
    assert q2.returncode == 0, (q2.stdout + q2.stderr)[-2000:]
    q3 = py("--report", "--template", A, "--lhs", B, "--report-dir", R2,
            "--allow-partial")
    assert q3.returncode == 0, (q3.stdout + q3.stderr)[-2000:]
    with open(os.path.join(R2, "B0.json"), encoding="utf-8") as f:
        rj = json.load(f)
    assert rj["verdict"] == "PARTIAL"
    for key in ("replay_ok", "audit_ok", "harness_errors_zero", "orphans_zero"):
        assert rj["checks"][key] is True, key
    q4 = py("--run", "--split", "test", "--system", "b0-template", "--ids",
            "A-1-008", "--out", X)
    assert q4.returncode == 2, q4.returncode
    assert "whole test split" in q4.stderr, q4.stderr[-500:]
    assert not os.path.exists(X)
    q5 = py("--run", "--split", "tuning", "--system", "rules", "--out", X)
    assert q5.returncode == 2, q5.returncode
    assert "system" in q5.stderr, q5.stderr[-500:]
    return ("live: b0-template on D-1-010 and F-1-009, b0-lhs on B-1-011, "
            "through the CLI at 2 streams; the report PARTIAL with replay, "
            "audit, 0 harness errors and 0 orphans; the CLI refuses a partial "
            "test run and a non-baseline system")


def _selftest_body(tmp):
    H = {"tmp": tmp}
    gates = schema.load_gates()
    knobs = schema.load_knobs()
    campaign._fake_campaign(H, "t", list(_IDS5), _SCRIPT_T, mode="b0-template")
    campaign._fake_campaign(H, "l", ["D-1-010", "F-1-009"], _SCRIPT_L,
                            mode="b0-lhs")
    mrows = {r["geometry_id"]: r for r in campaign.load_manifest(
        "tuning", "selftest", ids=list(_IDS5) + ["B-1-011"])}
    geoms_by = {g["geometry_id"]: g
                for g in campaign.load_geometries(os.path.join(tmp, "t"))}
    rows_by = {}
    for r in campaign.load_rows(os.path.join(tmp, "t")):
        rows_by.setdefault(r["geometry_id"], {})[r["attempt"]] = r
    fps = {gid: g.get("fingerprint") for gid, g in geoms_by.items()}
    q = {gid: qualifies(mrows[gid], fps.get(gid), gates, knobs)
         for gid in _IDS5}
    os.makedirs(os.path.join(tmp, "obs"), exist_ok=True)
    c = campaign.Campaign(os.path.join(tmp, "obs"), campaign_id="obs",
                          mode="rules", system="rules", ablate=(),
                          split_mode="rules", binary=campaign.BINARY_DEFAULT,
                          streams=2, timeout_s=campaign.TIMEOUT_S,
                          probe_timeout_s=campaign.PROBE_TIMEOUT_S, audit_mod=1,
                          quiet=True)
    c.start()
    try:
        obs_b = campaign.observe(c, mrows["B-1-011"])
        obs_d = campaign.observe(c, mrows["D-1-010"])
    finally:
        c.kill_all()
        c.stop()
    assert obs_b["terminal"] is None and obs_d["terminal"] is None
    groups = (
        (1, lambda: _g1(tmp)),
        (2, lambda: _g2(geoms_by, rows_by, q, mrows, fps)),
        (3, lambda: _g3(q, mrows, gates, knobs)),
        (4, lambda: _g4(tmp)),
        (5, lambda: _g5(tmp)),
        (6, lambda: _g6(tmp)),
        (7, lambda: _g7(mrows, obs_b, obs_d, gates, knobs)),
        (8, lambda: _g8(tmp)),
        (9, lambda: _g9(tmp)))
    for n, fn in groups:
        _run_group(n, fn)


def selftest():
    t0 = time.perf_counter()
    tmp = tempfile.mkdtemp(prefix="baseline-selftest-")
    try:
        _selftest_body(tmp)
    except AssertionError as e:
        sys.stderr.write("SELFTEST FAIL: %s\n" % e)
        return 1
    except Exception as e:
        sys.stderr.write("SELFTEST FAIL: %s: %s\n" % (type(e).__name__, e))
        return 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("SELFTEST PASS (%.1f s)" % (time.perf_counter() - t0), flush=True)
    return 0


# --- (C11) the CLI ---------------------------------------------------------------

class _ArgParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write("baseline: %s\n" % message)
        sys.exit(2)


def _print_sealed(entry, system):
    print("[baseline test %s] sealed %s sha256 %s (%d geometries, binary %s)"
          % (system, entry["file"], entry["sha256"], entry["n_geometries"],
             entry["binary_sha256"][:12]), flush=True)


def main(argv=None):
    ap = _ArgParser(prog="baseline.py")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--seal", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--rcurv", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--split")
    ap.add_argument("--system")
    ap.add_argument("--out")
    ap.add_argument("--streams", type=int, default=6)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--binary")
    ap.add_argument("--ids")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--report-dir", default=REPORT_DIR)
    ap.add_argument("--template")
    ap.add_argument("--lhs")
    ap.add_argument("--allow-partial", action="store_true")
    a = ap.parse_args(argv)
    try:
        if a.selftest:
            return selftest()
        if a.run:
            ids = [s for s in a.ids.split(",") if s] if a.ids else None
            run(a.split, a.system, a.out, streams=a.streams,
                resume=a.resume, binary=a.binary, ids=ids,
                limit=a.limit, report_dir=a.report_dir)
            return 0
        if a.seal:
            _print_sealed(seal(a.system, a.out, report_dir=a.report_dir), a.system)
            return 0
        if a.report:
            return _main_report(a)
        if a.rcurv:
            return _main_rcurv(a)
        if a.check:
            return _main_check(a)
        ap.error("one of --selftest, --run, --seal, --report, --rcurv, --check "
                 "is required")
    except (BaselineError, campaign.CampaignError, split.SplitSealed,
            split.SplitError) as e:
        sys.stderr.write("baseline: %s\n" % e)
        return 2


def _main_report(a):
    rep = report(a.template, a.lhs, report_dir=a.report_dir,
                 allow_partial=a.allow_partial)
    tg = next(g for g in rep["systems"]["b0-template"]["groups"]
              if (g["family"], g["stratum"]) == ("all", "all"))
    lg = next(g for g in rep["systems"]["b0-lhs"]["groups"]
              if (g["family"], g["stratum"]) == ("all", "all"))
    hl = rep["headline"]["capability_limited"]["b0-template"]
    print("B0 %s: template n %d MFR %.3f strict %.3f blc8 %.3f; lhs n %d MFR "
          "%.3f strict %.3f blc8 %.3f; capability-limited tier1 %.3f all %.3f"
          % (rep["verdict"], tg["n"], tg["mfr"], tg["strict_rate"],
             tg["blc8_mean"], lg["n"], lg["mfr"], lg["strict_rate"],
             lg["blc8_mean"], hl.get("tier1", 0.0), hl.get("all", 0.0)),
          flush=True)
    bad = [k for k in sorted(rep["checks"]) if not rep["checks"][k]]
    print("failing checks: %s" % (", ".join(bad) if bad else "none"), flush=True)
    if rep["verdict"] == "PASS" or (rep["verdict"] == "PARTIAL"
                                    and a.allow_partial):
        return 0
    return 1


def _main_rcurv(a):
    ids = [s for s in a.ids.split(",") if s] if a.ids else None
    rep = rcurv(a.out, streams=a.streams, binary=a.binary, ids=ids,
                limit=a.limit, report_dir=a.report_dir)
    for g in rep["groups"]:
        w, o = g["with"], g["without"]
        print("R-CURV %s: n %d wall level %s/%s F3 %d/%d failure %d/%d "
              "capability-limited %.3f/%.3f"
              % (g["family"], g["n"], _cells(w["wall_level_median"]),
                 _cells(o["wall_level_median"]), w["f3"], o["f3"],
                 w["failure"], o["failure"], w["capability_limited_mean"],
                 o["capability_limited_mean"]), flush=True)
    print("R-CURV done: %d pairs, %d skipped"
          % (rep["n_pairs"], sum(rep["skipped"].values())), flush=True)
    return 0


def _main_check(a):
    ck = check(a.report_dir)
    for item in ck["items"]:
        print("[check] %s %s %s" % (item["name"], "ok" if item["ok"] else "FAIL",
                                    item["why"]), flush=True)
    print("CHECK %s" % ck["verdict"], flush=True)
    return 0 if ck["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
