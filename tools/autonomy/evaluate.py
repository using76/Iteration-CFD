#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""
evaluate.py - the held-out evaluation of docs/15 §F (AM-16).

It verifies the two sealed baselines by hash, writes a write-once plan lock,
opens the test split ONCE (mode evaluate), and runs nine campaigns as
`evaluate`-mode campaigns: the full system, a fresh B0-template re-measure, two
G-DET runs on the 20 G-DET geometries, then the four ablations, rules only and
no-preflight, which reuse what the earlier campaigns meshed (the mesher is
deterministic, so a (geometry, config sha) already measured returns that
outcome).  An octree probe predicting more than 8192 MiB of peak memory is not
run and is scored as an F1 no-mesh failure (the RAM guard).  From the committed
bundles it decides G-FAIL, G-BLC-0, G-QUAL, G-FID, G-COST, G-DET, G-EXPL and
G-OPT, reports G-ABL, and writes the results page EVAL.md beside EVAL.json.

The real evaluation (hours of CPU on the held-out split) is the supervisor's
`--run`; nothing in this module ever reads a sealed bundle's contents except
through the hash check, and once the lock records a plan the split is spent.

    python tools/autonomy/evaluate.py --selftest
    python tools/autonomy/evaluate.py --plan
    python tools/autonomy/evaluate.py --run --work DIR
    python tools/autonomy/evaluate.py --report
    python tools/autonomy/evaluate.py --check
"""
import argparse
import ast
import collections
import copy
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for _p in (HERE, os.path.join(HERE, "corpus")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import baseline
import campaign
import explain
import optimise
import preflight
import prior
import remedies
import schema
import score
import split


class EvalError(ValueError):
    """A refused request or a harness inconsistency - never a verdict."""


REPORT_DIR = os.path.join(HERE, "evaluate")
REPORT_NAME = "EVAL.json"
REPORT_MD = "EVAL.md"
RUNS_NAME = "runs.json"
LOCK_NAME = "opened.lock"
CONTEXT_NAME = "tuning_context.json"
CONTEXT_SCHEMA = "autonomy-eval-context/1"
COUNTS_NAME = "eval_counts.json"            # inside each campaign directory
BUNDLE_NAME = "eval_%s.json.gz"             # one per campaign name, in the report dir
BASELINE_COPY = "baseline_%s.json.gz"       # rehearsal only: the stand-in baselines
SEALED = {"b0-template": "7685d908ada56902576840d03baef07476408bb7e969a4739fb8d9a3fac79a59",
          "b0-lhs": "03ee0ac451a0474bd3339028f1364e18745336ba890b2e46a06b324285d79a59"}
# (name, system, ablate, reuse, ids) in run order; ids "all" = the manifest, "gdet" = the G-DET ids
CAMPAIGNS = (("full", "full", (), False, "all"),
             ("b0-template", "b0-template", (), False, "all"),
             ("gdet-1", "full", (), False, "gdet"),
             ("gdet-2", "full", (), False, "gdet"),
             ("no-remedies", "full", ("remedies",), True, "all"),
             ("no-optimiser", "rules+prior", (), True, "all"),
             ("rules", "rules", (), True, "all"),
             ("no-prior", "rules+opt", (), True, "all"),
             ("no-preflight", "full", ("preflight",), True, "all"))
ABL_ORDER = ("full", "no-preflight", "no-remedies", "no-prior", "no-optimiser", "rules",
             "b0-template", "b0-lhs")
LABELS = {"full": "full", "no-preflight": "-preflight", "no-remedies": "-remedies",
          "no-prior": "-prior", "no-optimiser": "-optimiser",
          "rules": "rules + remedies only", "b0-template": "B0-template (sealed)",
          "b0-lhs": "B0-LHS best of 4 (sealed)"}
FAMILIES = ("A", "B", "D", "E", "F", "G")
TIER0_FAMILIES = ("D", "F")
TEST_N = 180
GDET_N = 20
GDET_SALT = "gdet:"
RAM_GUARD_MIB = 8192.0
AUDIT_OFF = 2 ** 40         # audit_mod of the reusing campaigns (optimise.py's value)
MFR_RATIO_MAX = 0.25
MFR_UPPER_MAX = 0.10
MCNEMAR_P_MAX = 0.01        # docs/15 §F G-FAIL
BLC8_MIN = 0.90
BLCF_MIN = 0.80             # docs/15 §F G-BLC-0
CELLS_RATIO_MAX = 1.5
WALL_MAX_S = 10800.0
STREAMS_DOCS = 12
RAM_FRAC_MAX = 0.60         # docs/15 §F G-COST
MFR_GAIN_MIN = 0.03
BLC_GAIN_MIN = 0.05         # docs/15 §F G-OPT (optimise.py's values)
EPS = 1e-12
DOI = "10.3389/frobt.2025.1566623"
GUARD_LINE = ("evaluate.py: RAM guard: the octree probe's %d leaves predict "
              "%.1f MiB > %.1f MiB; not run")
REPORT_SCHEMA = "autonomy-eval/1"
RUNS_SCHEMA = "autonomy-eval-runs/1"
LOCK_SCHEMA = "autonomy-eval-lock/1"
CHECK_SCHEMA = "autonomy-eval-check/1"
PASS, FAIL, UNDECIDED, REPORTED = "PASS", "FAIL", "UNDECIDED", "REPORTED"
HEADER = baseline.HEADER                    # the "$comment" of every JSON it writes
DEPARTURES = (
    "docs/15 §F measures G-COST's campaign wall time at 12 streams; the house caps campaigns at 6 while the solver "
    "workflow runs, so the wall item passes when the 6-stream time is at most 3 h, fails when even perfect 12-stream "
    "scaling (half the 6-stream time) or the longest single geometry exceeds 3 h, and is undecided in between.",
    "The ablations reuse what earlier campaigns of this evaluation measured: an attempt whose (geometry, config sha) "
    "was meshed returns that outcome (the mesher is deterministic, §F G-DET). The full system, the B0-template "
    "re-measure and both G-DET runs are meshed fresh.",
    "The sealed B0-template rows carry no feature-edge counts, so B0-template is re-meshed once on the test split "
    "for G-FID; its rows must equal the sealed rows apart from time fields and git_sha, and every other B0 number "
    "comes from the seal.",
    "G-FID's feature-edge share is n_snapped_to_edge / n_feature_edges; a mesh snapped with feature_tolerance 0 "
    "extracts no edges and reports 0 of them, so on a body whose fingerprint has sharp edges its share is 0.0, not "
    "undefined. A body without sharp edges is left out.",
    "The -preflight ablation meshes configs L0 would refuse. On this shared machine an attempt whose octree probe "
    "predicts more than 8192 MiB of peak memory is not run and is scored as an F1 no-mesh failure (class crash); the "
    "count is reported, and it is 0 wherever L0 runs, whose budget check keeps a probe under 2 M leaves (4.9 GiB).",
    "G-OPT as written compares the full system with rules+remedies only (mode rules); the optimiser's own marginal "
    "(full against -optimiser) and rules+opt against rules are reported beside it.",
    "-remedies runs K = 1, so the optimiser, which acts only after the remedies are spent with attempts left, never "
    "acts in it either.",
    "The test manifest is opened once by this evaluation (split.load('test', 'evaluate')) after a write-once lock "
    "records the plan; each campaign re-reads the same lock-checked rows by name. A second evaluation with any other "
    "plan is refused: the split is spent (§F).",
    "The 20 G-DET geometries are fixed from the lock's test ids before any outcome exists: within each family by "
    "sha256('gdet:' + id), drawn round-robin A, B, D, E, F, G.",
    "The tier-0 stratum of G-BLC-0 is the D and F rows whose manifest field commensurate is true.",
    "A geometry that ends SURFACE-OPEN, SURFACE-REFUSED, REFUSED or HARNESS-ERROR is a failure in every system "
    "(§D.1 F1), the sealed baselines included (read through campaign.terminal_of).",
)
FID_METRICS = (("p99_over_hf", "le"), ("pinned_frac", "le"), ("feature_share", "ge"))


def _dump_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def _write_text(path, text):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _sha256_of_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# --- (C2) the seal and the G-DET ids ----------------------------------------

def verify_seal(report_dir=baseline.REPORT_DIR):
    """Both sealed files hash to the committed constants; never decompressed."""
    entries = baseline._read_seal_lock(report_dir).get("entries") or {}
    out = {}
    for system in baseline.SYSTEMS:
        if system not in entries or "file" not in entries[system] \
                or "sha256" not in entries[system]:
            raise EvalError("seal: %s is not sealed in %s"
                            % (system, os.path.join(report_dir, baseline.LOCK_NAME)))
        entry = entries[system]
        path = os.path.join(report_dir, entry["file"])
        if not os.path.isfile(path):
            raise EvalError("seal: %s is missing" % path)
        h = _sha256_of_file(path)
        if h != SEALED[system] or h != entry["sha256"]:
            raise EvalError("seal: %s hashes to %s, not the sealed %s or the "
                            "lock's %s" % (path, h, SEALED[system], entry["sha256"]))
        out[system] = {"file": path, "sha256": h, "ok": True}
    return out


def lock_test_ids():
    lock = split.read_lock(os.path.join(split.MANIFEST_DIR, "split.lock"))
    return lock["test_ids"].split()


def gdet_ids(ids, n=GDET_N):
    """20 test ids fixed before any outcome exists: within each family by
    sha256('gdet:' + id), drawn round-robin A, B, D, E, F, G (docs/15 §F)."""
    by = {}
    for gid in ids:
        by.setdefault(gid[0], []).append(gid)
    for fam in by:
        by[fam].sort(key=lambda g: hashlib.sha256(
            (GDET_SALT + g).encode("ascii")).hexdigest())
    out = []
    while len(out) < n:
        drew = False
        for fam in FAMILIES:
            if by.get(fam):
                out.append(by[fam].pop(0))
                drew = True
                if len(out) >= n:
                    break
        if not drew:
            break
    return sorted(out)


# --- (C3) the split, opened once, and the baselines --------------------------

def open_split(manifest):
    """The evaluation's only manifest readers; "test" opens the sealed split."""
    if manifest == "test":
        rows = split.load("test", split.EVALUATE)
        if len(rows) != TEST_N:
            raise EvalError("open_split: the test split holds %d rows, not %d"
                            % (len(rows), TEST_N))
        return {"source": "test", "sha256": campaign.manifest_sha("test"),
                "rows": rows, "rehearsal": False}
    rows = campaign.load_manifest(manifest, split.EVALUATE)
    for r in rows:
        split.refuse_test(r["geometry_id"], "tuning")
    return {"source": os.path.abspath(manifest).replace(os.sep, "/"),
            "sha256": campaign.manifest_sha(manifest), "rows": rows,
            "rehearsal": True}


def meta_of(opened):
    rows = opened["rows"]
    fams = dict((f, 0) for f in FAMILIES)
    tier0 = []
    for r in rows:
        if r["family"] in fams:
            fams[r["family"]] += 1
        if r["family"] in TIER0_FAMILIES and r.get("commensurate") is True:
            tier0.append(r["geometry_id"])
    return {"source": opened["source"], "sha256": opened["sha256"], "n": len(rows),
            "ids": [r["geometry_id"] for r in rows], "families": fams,
            "tier0_ids": tier0, "rehearsal": opened["rehearsal"]}


def subset_bundle(bundle, ids):
    b = copy.deepcopy(bundle)
    want = set(ids)
    for key in ("attempts", "geometries", "records", "jobs"):
        b[key] = [e for e in (b.get(key) or [])
                  if e.get("geometry_id") in want]
    have = [g["geometry_id"] for g in b["geometries"]]
    missing = [gid for gid in ids if gid not in have]
    if missing:
        raise EvalError("baseline %s lacks %s"
                        % (b.get("campaign", {}).get("system", "?"),
                           ", ".join(missing)))
    b["samples"] = []
    head = dict(b["campaign"])
    head["geometry_ids"] = [gid for gid in ids]
    b["campaign"] = head
    return b


def load_baselines(meta, report_dir, sources=None):
    if not meta["rehearsal"]:
        # the seal lives in baseline/, whatever this evaluation's report dir is
        verify_seal()
        return dict((s, baseline.load_sealed(s, split.EVALUATE))
                    for s in baseline.SYSTEMS)
    if sources is None:
        return dict((s, baseline.read_bundle(os.path.join(report_dir,
                                                          BASELINE_COPY % s)))
                    for s in baseline.SYSTEMS)
    if set(sources) != set(baseline.SYSTEMS) or len(sources) != len(baseline.SYSTEMS):
        raise EvalError("a rehearsal needs --baselines")
    out = {}
    for s in baseline.SYSTEMS:
        b = subset_bundle(baseline.read_bundle(sources[s]), meta["ids"])
        baseline._write_or_match(report_dir, BASELINE_COPY % s, b)
        out[s] = b
    return out


def bundle_sha(b):
    return hashlib.sha256(baseline._canonical(b)).hexdigest()


# --- (C4) the plan and the write-once lock -----------------------------------

def plan_of(manifest_info, rehearsal, gdet, streams, binary, baselines):
    return {"manifest": {"source": manifest_info["source"],
                         "sha256": manifest_info["sha256"],
                         "n": manifest_info["n"]},
            "rehearsal": rehearsal,
            "campaigns": [[name, system, list(ablate), reuse, which]
                          for name, system, ablate, reuse, which in CAMPAIGNS],
            "gdet_ids": list(gdet), "streams": streams,
            "binary_sha256": _sha256_of_file(binary),
            "gates_sha256": campaign.sha(schema.load_gates()),
            "knobs_sha256": campaign.sha(schema.load_knobs()),
            "models": {"prior": _sha256_of_file(os.path.join(prior.REPORT_DIR,
                                                             prior.MODEL_NAME)),
                       "optimiser": _sha256_of_file(os.path.join(
                           optimise.SHIP_DIR, optimise.MODEL_NAME)),
                       "optimiser_train": _sha256_of_file(os.path.join(
                           optimise.SHIP_DIR, optimise.TRAIN_NAME))},
            "baselines": dict(baselines),
            "ram_guard_mib": RAM_GUARD_MIB, "audit_mod": campaign.AUDIT_MOD}


def write_lock(report_dir, plan):
    os.makedirs(report_dir, exist_ok=True)
    path = os.path.join(report_dir, LOCK_NAME)
    new_sha = campaign.sha(plan)
    if os.path.isfile(path):
        old = _read_json(path)
        if old.get("plan_sha256") == new_sha:
            return old
        raise EvalError("the test split is spent: %s records plan %s and this "
                        "run's plan is %s; a second claim needs a fresh test "
                        "seed and a new lock (docs/15 §F)"
                        % (path, str(old.get("plan_sha256"))[:12], new_sha[:12]))
    lock = {"$comment": HEADER, "schema": LOCK_SCHEMA, "plan": plan,
            "plan_sha256": new_sha, "opened_at": schema._now_iso(),
            "git_sha": campaign.Campaign._git_sha()}
    _dump_json(path, lock)
    return lock


# --- (C5) the reuse and RAM-guard seam ---------------------------------------

def new_cache():
    return {"measured": {}, "probes": {}, "snaps": {}, "guarded": set()}


def new_counts():
    return {"meshed": 0, "reused": 0, "ram_guarded": 0, "guard_probes": 0,
            "guarded": []}


def eval_fns(cache, counts, *, reuse, attempt_fn=None, probe_fn=None, snap_fn=None):
    """(attempt, probe, snap) that reuse earlier measurements and RAM-guard:
    a campaign runs up to 6 threads, so one lock guards every counts update."""
    lock = threading.Lock()

    def probe(c, gctx, cfg, a):
        key = campaign.sha(cfg)
        if reuse and key in cache["probes"]:
            return {"n_leaves": cache["probes"][key], "max_non_orth_deg": None,
                    "exit_code": 0}
        return (probe_fn or campaign.octree_probe)(c, gctx, cfg, a)

    def snap(c, gctx, config, a):
        key = campaign.sha(config)
        if reuse and key in cache["snaps"]:
            return cache["snaps"][key]      # the recorded value, None included
        return (snap_fn or campaign.snap_probe)(c, gctx, config, a)

    def attempt(c, gctx, config, a, n_leaves):
        key = (gctx["gid"], campaign.sha(config))
        now = schema._now_iso()
        with lock:
            hit = bool(reuse) and key in cache["measured"]
        if hit:
            with lock:
                counts["reused"] += 1
            stored = cache["measured"][key]
            oc = copy.deepcopy(stored["outcome"])
            for p in oc.get("patches") or []:
                p["capability_limited"] = False
            return {"outcome": oc, "content_sha256": stored["content_sha256"],
                    "t_start": now, "t_end": now}
        nl = n_leaves
        if nl is None and "preflight" in c.ablate:
            nl = probe(c, gctx, config, a).get("n_leaves")
            with lock:
                counts["guard_probes"] += 1
        if isinstance(nl, int) and not isinstance(nl, bool) \
                and campaign.expected_mib(nl) > RAM_GUARD_MIB:
            exp = campaign.expected_mib(nl)
            with lock:
                counts["ram_guarded"] += 1
                counts["guarded"].append([key[0], key[1]])
            return {"outcome": score.score_run(
                        exit_code=None, stdout="",
                        stderr=GUARD_LINE % (nl, exp, RAM_GUARD_MIB), summary=None,
                        config=config, patch_areas_m2=gctx["areas"],
                        flow=gctx["flow"], wall_seconds=0.0,
                        gates=c.gates)["outcome"],
                    "content_sha256": None, "t_start": now, "t_end": now}
        with lock:
            counts["meshed"] += 1
        return (attempt_fn or campaign.run_attempt)(c, gctx, config, a, nl)

    return attempt, probe, snap


def update_cache(cache, bundle, guarded):
    cache["guarded"] |= {tuple(k) for k in guarded}
    m = optimise.measured_from(bundle)
    for k in list(m):
        if k in cache["guarded"]:
            del m[k]
    cache["measured"].update(m)
    cache["probes"].update(optimise.probes_from(bundle))
    cache["snaps"].update(optimise.snaps_from(bundle))


# --- (C6) statistics ---------------------------------------------------------

def mcnemar(b, c):
    """The exact two-sided McNemar p over b/c discordant pairs (integers)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
    return min(1.0, 2 * tail / 2 ** n)


def _mean(vals):
    vals = list(vals)
    return sum(vals) / len(vals) if vals else 0.0


def _median(vals):
    return statistics.median(vals) if vals else None


def _stats(ms):
    n = len(ms)
    fails = sum(1 for g in ms if g["failure"] is True)
    stricts = sum(1 for g in ms if g["strict_failure"] is True)
    cells = [g["n_cells"] for g in ms
             if g["failure"] is False and g["n_cells"] is not None]
    return {"n": n, "failures": fails, "mfr": fails / n if n else 0.0,
            "mfr_ci": list(explain.clopper_pearson(fails, n)),
            "strict": stricts, "strict_rate": stricts / n if n else 0.0,
            "blc8_mean": _mean(g["blc8_a_priori"] for g in ms),
            "blc_full_mean": _mean(g["blc_full_a_priori"] for g in ms),
            "cells_median": _median(cells) if cells else None}


def system_stats(ends, ids):
    missing = [i for i in ids if i not in ends]
    if missing:
        raise EvalError("system lacks %s" % missing[0])
    ms = [ends[i] for i in ids]
    out = _stats(ms)
    terminals = collections.Counter()
    for g in ms:
        terminals[campaign.terminal_of(g)] += 1
    out["terminals"] = dict(terminals)
    per = {}
    for f in FAMILIES:
        fids = [i for i in ids if ends[i].get("family") == f]
        if fids:
            per[f] = _stats([ends[i] for i in fids])
    out["per_family"] = per
    return out


def family_worse(a, b):
    return sorted(f for f in b["per_family"]
                  if a["per_family"][f]["failures"] > b["per_family"][f]["failures"])


def regressed(new, old):
    out = []
    for f in sorted(old["per_family"]):
        on = new["per_family"].get(f)
        oo = old["per_family"][f]
        if on is None:
            continue
        if on["failures"] > oo["failures"] or oo["blc8_mean"] - on["blc8_mean"] > EPS:
            out.append(f)
    return out


# --- (C7) one campaign, and what is read from its directory -------------------

def run_one(spec, work, meta, gdet, cache, *, streams, binary, attempt_fn=None,
            probe_fn=None, snap_fn=None, hooks=None, quiet=False):
    name, system, ablate, reuse, which = spec
    ids = list(meta["ids"]) if which == "all" else list(gdet)
    out = os.path.join(work, name)
    end_path = os.path.join(out, campaign.FILES["end"])
    ran = not os.path.isfile(end_path)
    if not ran:
        head = _read_json(os.path.join(out, campaign.FILES["campaign"]))
        want = {"system": system, "ablate": sorted(ablate),
                "split_mode": split.EVALUATE,
                "manifest": {"source": meta["source"], "sha256": meta["sha256"],
                             "n": len(ids)},
                "geometry_ids": ids, "binary_sha256": _sha256_of_file(binary),
                "streams": streams}
        for k, v in want.items():
            if head.get(k) != v:
                raise EvalError("campaign %s in %s was run with another plan "
                                "(%s differs)" % (name, work, k))
        cpath = os.path.join(out, COUNTS_NAME)
        if not os.path.isfile(cpath):
            raise EvalError("campaign %s in %s has no %s" % (name, work, COUNTS_NAME))
        counts = _read_json(cpath)
    elif os.path.isdir(out) and os.listdir(out):
        raise EvalError("campaign %s is incomplete: remove %s and run it again"
                        % (name, out))
    else:
        counts = new_counts()
        A, P, S = eval_fns(cache, counts, reuse=reuse, attempt_fn=attempt_fn,
                           probe_fn=probe_fn, snap_fn=snap_fn)
        campaign.run_campaign({"manifest": meta["source"], "mode": "evaluate",
                               "system": system, "ablate": tuple(ablate),
                               "ids": list(ids), "out": out, "run_id": "eval-" + name,
                               "streams": streams,
                               "audit_mod": AUDIT_OFF if reuse else campaign.AUDIT_MOD,
                               "binary": binary, "quiet": quiet},
                              attempt_fn=A, probe_fn=P, snap_fn=S, hooks=hooks)
        counts["guarded"] = sorted(counts["guarded"])
        _dump_json(os.path.join(out, COUNTS_NAME), counts)
    bundle = baseline.bundle_dir(out)
    update_cache(cache, bundle, counts["guarded"])
    if not quiet:
        end = bundle["end"]
        print("[eval] %s (%s, ablate %s): %d geometries, %d rows, meshed %d, "
              "reused %d, RAM-guarded %d, harness errors %d, %.1f s (%s)"
              % (name, system, ", ".join(sorted(ablate)) or "none",
                 end["n_geometries"], end["n_rows"], counts["meshed"],
                 counts["reused"], counts["ram_guarded"], end["harness_errors"],
                 end["wall_seconds"], "ran" if ran else "reused"))
    return out, bundle, counts


def qual_of(out, rows, knobs):
    sha_bad = quality_bad = edits = edits_bad = 0
    for row in rows:
        cpath = os.path.join(out, campaign.config_rel(row["geometry_id"],
                                                      row["attempt"]))
        if not os.path.isfile(cpath):
            raise EvalError("qual_of: %s is missing" % cpath)
        cfg = _read_json(cpath)
        if campaign.sha(cfg) != row["config_sha"]:
            sha_bad += 1
        _, eff, raw = preflight._quality_block(cfg)
        if preflight._pf_quality(eff, raw, None)["verdict"] != "pass":
            quality_bad += 1
        delta = row.get("config_delta") or []
        edits += len(delta)
        edits_bad += sum(1 for e in delta
                         if schema.check_edit(e["pointer"], e["to"], knobs) is not None)
    return {"configs": len(rows), "sha_bad": sha_bad, "quality_bad": quality_bad,
            "edits": edits, "edits_bad": edits_bad}


def feature_share(nfe, nse, ft, sharp):
    if nfe is None:
        return None
    if nfe > 0:
        return nse / nfe
    if nfe == 0 and ft == 0 and sharp is not None and sharp > 0:
        return 0.0
    return None


def fid_rows(out, bundle):
    rows = []
    default_ft = preflight.MESHER_DEFAULTS["/snap/feature_tolerance"]
    for g in bundle["geometries"]:
        if g["failure"] is not False or not g.get("final_attempt"):
            continue
        k = g["final_attempt"]
        gid = g["geometry_id"]
        row = next((r for r in bundle["attempts"]
                    if r["geometry_id"] == gid and r["attempt"] == k), None)
        if row is None:
            continue
        cfg = _read_json(os.path.join(out, campaign.config_rel(gid, k)))
        ft = (cfg.get("snap") or {}).get("feature_tolerance", default_ft)
        sharp = g["fingerprint"]["sharp_edge_length_m"]
        sdir = os.path.join(out, "cases", "%s_a%d" % (gid, k))
        spath = os.path.join(sdir, "%s_a%d_summary.json" % (gid, k))
        has_summary = os.path.isfile(spath)
        nfe = nse = None
        if has_summary:
            summ = _read_json(spath)
            st = next((s for s in (summ.get("stages") or [])
                       if s.get("stage") == "snap"), None)
            if st is not None:
                nfe, nse = st.get("n_feature_edges"), st.get("n_snapped_to_edge")
        rows.append({"geometry_id": gid, "family": g["family"], "attempt": k,
                     "pinned_frac": row["outcome"]["pinned_frac"],
                     "p99_over_hf": row["outcome"]["p99_over_hf"],
                     "n_feature_edges": nfe, "n_snapped_to_edge": nse,
                     "feature_tolerance": ft, "sharp_edge_length_m": sharp,
                     "feature_share": feature_share(nfe, nse, ft, sharp),
                     "summary": has_summary})
    return rows


def seal_compare(re_bundle, sealed_bundle):
    def keyed(bundle):
        out = {}
        for r in bundle["attempts"]:
            x = campaign._strip_row(r)
            x.pop("git_sha", None)
            out[(r["geometry_id"], r["attempt"])] = x
        return out

    a, b = keyed(re_bundle), keyed(sealed_bundle)
    row_diffs = []
    for key in sorted(set(a) | set(b)):
        if key not in a or key not in b:
            row_diffs.append({"key": list(key), "path": "/"})
            continue
        p = campaign._first_diff(a[key], b[key])
        if p is not None:
            row_diffs.append({"key": list(key), "path": p})
    content_compared = content_equal = 0
    for key in sorted(set(a) & set(b)):
        ca, cb = a[key].get("content_sha256"), b[key].get("content_sha256")
        if ca is not None and cb is not None:
            content_compared += 1
            content_equal += 1 if ca == cb else 0

    def terms(bundle):
        return dict((g["geometry_id"], campaign.terminal_of(g))
                    for g in bundle["geometries"])

    ta, tb = terms(re_bundle), terms(sealed_bundle)
    terminal_diffs = sorted(gid for gid in set(ta) | set(tb)
                            if ta.get(gid) != tb.get(gid))
    equal = not row_diffs and not terminal_diffs and content_compared == content_equal
    return {"rows_a": len(a), "rows_b": len(b), "n_row_diffs": len(row_diffs),
            "row_diffs": row_diffs[:20], "content_compared": content_compared,
            "content_equal": content_equal, "terminal_diffs": terminal_diffs,
            "equal": equal}


def compare_subset(dir_a, dir_b, ids):
    """campaign.compare's row part restricted to the rows of `ids` (reported)."""
    want = set(ids)

    def keyed(d):
        return {(r["geometry_id"], r["attempt"]): campaign._strip_row(r)
                for r in campaign.load_rows(d) if r["geometry_id"] in want}

    a, b = keyed(dir_a), keyed(dir_b)
    row_diffs = []
    for key in sorted(set(a) | set(b)):
        if key not in a or key not in b:
            row_diffs.append({"key": list(key), "path": "/"})
            continue
        p = campaign._first_diff(a[key], b[key])
        if p is not None:
            row_diffs.append({"key": list(key), "path": p})
    content_compared = content_equal = 0
    for key in sorted(set(a) & set(b)):
        ca, cb = a[key].get("content_sha256"), b[key].get("content_sha256")
        if ca is not None and cb is not None:
            content_compared += 1
            content_equal += 1 if ca == cb else 0
    return {"rows": len(a), "n_row_diffs": len(row_diffs),
            "row_diffs": row_diffs[:20], "content_compared": content_compared,
            "content_equal": content_equal,
            "ok": len(a) == len(b) and not row_diffs and content_compared == content_equal}


def flag_literals(knobs):
    """"<file>:<lineno>" for every whole-string constant equal to a forbidden
    flag - evaluate.py itself never spells one (it reads them from knobs)."""
    flags = [f["flag"] for f in knobs.get("forbidden_flags") or []]
    out = set()
    for fname in (os.path.join(HERE, "campaign.py"), os.path.abspath(__file__)):
        with open(fname, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value in flags:
                out.add("%s:%d" % (os.path.basename(fname), node.lineno))
    return sorted(out)


# --- (C8) the gates ----------------------------------------------------------

def g_fail(full, b0, pairs):
    b = sum(1 for p, f in pairs if p is True and f is False)
    c = sum(1 for p, f in pairs if p is False and f is True)
    p = mcnemar(b, c)
    worse = family_worse(full, b0)
    conditions = {"ratio_le_quarter": 4 * full["failures"] <= b0["failures"],
                  "upper_le_10pct": full["mfr_ci"][1] <= MFR_UPPER_MAX,
                  "mcnemar_p_lt_001": p < MCNEMAR_P_MAX,
                  "no_family_worse": not worse}
    return {"verdict": PASS if all(conditions.values()) else FAIL,
            "conditions": conditions, "mfr_full": full["mfr"], "mfr_b0": b0["mfr"],
            "ratio": (full["mfr"] / b0["mfr"]) if b0["mfr"] else None,
            "upper": full["mfr_ci"][1], "mcnemar": {"b": b, "c": c, "p": p},
            "worse_families": worse}


def _blc_other(ends_full, other_ids):
    per = {}
    for f in FAMILIES:
        fids = [i for i in other_ids if ends_full[i].get("family") == f]
        if fids:
            per[f] = {"n": len(fids),
                      "blc8_mean": _mean(ends_full[i]["blc8_a_priori"] for i in fids),
                      "blc_full_mean": _mean(ends_full[i]["blc_full_a_priori"]
                                             for i in fids),
                      "n_capability_limited": sum(1 for i in fids
                                                  if ends_full[i]["capability_limited"])}
    cl = [{"geometry_id": i, "patches": list(ends_full[i]["capability_limited"])}
          for i in sorted(other_ids) if ends_full[i]["capability_limited"]]
    return {"per_family": per, "capability_limited": cl}


def g_blc0(ends_full, ends_b0, tier0_ids, other_ids):
    out = {"verdict": UNDECIDED, "n": len(tier0_ids), "ids": list(tier0_ids),
           "blc8_mean": None, "blc_full_mean": None,
           "b0": {"blc8_mean": None, "blc_full_mean": None},
           "conditions": {}, "other": _blc_other(ends_full, other_ids)}
    if not tier0_ids:
        return out
    m8 = _mean(ends_full[i]["blc8_a_priori"] for i in tier0_ids)
    mf = _mean(ends_full[i]["blc_full_a_priori"] for i in tier0_ids)
    conditions = {"blc8_ge_090": m8 >= BLC8_MIN, "blc_full_ge_080": mf >= BLCF_MIN}
    out.update({"blc8_mean": m8, "blc_full_mean": mf, "conditions": conditions,
                "b0": {"blc8_mean": _mean(ends_b0[i]["blc8_a_priori"]
                                          for i in tier0_ids),
                       "blc_full_mean": _mean(ends_b0[i]["blc_full_a_priori"]
                                              for i in tier0_ids)},
                "verdict": PASS if all(conditions.values()) else FAIL})
    return out


def g_qual(quals, static):
    totals = {"configs": 0, "sha_bad": 0, "quality_bad": 0, "edits": 0, "edits_bad": 0}
    for q in quals.values():
        for k in totals:
            totals[k] += q.get(k, 0)
    ok = (all(q["sha_bad"] == 0 and q["quality_bad"] == 0 and q["edits_bad"] == 0
              for q in quals.values())
          and bool(static["remedies_scan_ok"]) and not static["flag_literals"])
    return {"verdict": PASS if ok else FAIL, "per_campaign": quals,
            "static": static, "totals": totals}


def g_fid(fid_full, fid_b0, seal_equal):
    present = [f for f in FAMILIES
               if any(r["family"] == f for r in fid_full)
               or any(r["family"] == f for r in fid_b0)]
    per = {}
    worse = []
    compared = 0
    for f in present:
        ff = [r for r in fid_full if r["family"] == f]
        fb = [r for r in fid_b0 if r["family"] == f]
        metrics = {}
        for m, op in FID_METRICS:
            vf = _median([r[m] for r in ff if r.get(m) is not None])
            vb = _median([r[m] for r in fb if r.get(m) is not None])
            comp = vf is not None and vb is not None
            if comp:
                ok = vf <= vb + EPS if op == "le" else vf >= vb - EPS
                compared += 1
                if not ok:
                    worse.append([f, m])
            else:
                ok = None
            metrics[m] = {"full": vf, "b0": vb, "comparable": comp, "ok": ok}
        per[f] = {"n_full": len(ff), "n_b0": len(fb),
                  "n_ft0_sharp": sum(1 for r in ff if r.get("feature_tolerance") == 0
                                     and r.get("sharp_edge_length_m") is not None
                                     and r["sharp_edge_length_m"] > 0),
                  "metrics": metrics}
    if not seal_equal:
        verdict = UNDECIDED
    elif worse:
        verdict = FAIL
    elif compared == 0:
        verdict = UNDECIDED
    else:
        verdict = PASS
    return {"verdict": verdict, "per_family": per, "compared": compared,
            "worse": worse, "seal_equal": seal_equal}


def wall_verdict(wall_s, streams, longest_s):
    if wall_s <= WALL_MAX_S:
        return PASS
    bound = max(wall_s * streams / STREAMS_DOCS, longest_s)
    if bound > WALL_MAX_S:
        return FAIL
    return UNDECIDED


def g_cost(ends_full, ends_b0, ids, end_full, header_full, gates):
    both = [i for i in ids if ends_b0.get(i) is not None
            and ends_full[i]["failure"] is False
            and ends_b0[i]["failure"] is False
            and ends_full[i]["n_cells"] is not None
            and ends_b0[i]["n_cells"] is not None]
    if not both:
        cells = {"verdict": UNDECIDED, "n_both": 0, "median_full": None,
                 "median_b0": None, "ratio": None}
    else:
        mf = _median([ends_full[i]["n_cells"] for i in both])
        mb = _median([ends_b0[i]["n_cells"] for i in both])
        ratio = mf / mb if mb else None
        cells = {"verdict": PASS if ratio is not None and ratio <= CELLS_RATIO_MAX
                 else FAIL, "n_both": len(both), "median_full": mf,
                 "median_b0": mb, "ratio": ratio}
    n_over = sum(1 for i in ids if ends_full[i]["n_cells"] is not None
                 and ends_full[i]["n_cells"] > gates["cell_budget"])
    budget = {"verdict": PASS if n_over == 0 else FAIL, "n_over": n_over}
    longest = max([ends_full[i]["seconds"] for i in ids
                   if ends_full[i].get("seconds") is not None] or [0.0])
    wv = wall_verdict(end_full["wall_seconds"], header_full["streams"], longest)
    wall = {"verdict": wv, "seconds": end_full["wall_seconds"],
            "streams": header_full["streams"], "longest_geometry_s": longest,
            "bound_12": max(end_full["wall_seconds"] * header_full["streams"]
                            / STREAMS_DOCS, longest)}
    ram = {"verdict": PASS if end_full["peak_frac"] <= RAM_FRAC_MAX else FAIL,
           "peak_frac": end_full["peak_frac"],
           "peak_rss_mib": end_full["peak_rss_mib"]}
    orphans = {"verdict": PASS if not end_full["orphans"] else FAIL,
               "n": len(end_full["orphans"])}
    items = {"cells": cells, "budget": budget, "wall": wall, "ram": ram,
             "orphans": orphans}
    vs = [it["verdict"] for it in items.values()]
    verdict = FAIL if FAIL in vs else (PASS if UNDECIDED not in vs else UNDECIDED)
    return {"verdict": verdict, "items": items}


def g_det(compare, replay_1, replay_2, vs_full):
    ok = bool(compare.get("ok")) and bool(replay_1.get("ok")) \
        and bool(replay_2.get("ok"))
    return {"verdict": PASS if ok else FAIL, "compare": compare,
            "replay_1": replay_1, "replay_2": replay_2, "vs_full": vs_full}


def g_expl(summaries):
    per = {}
    for name in sorted(summaries):
        au = summaries[name]
        good = au is None or bool(au.get("ok"))
        per[name] = {"ok": good,
                     "n_rows": au.get("n_rows", 0) if au else 0,
                     "untemplated": au.get("untemplated", 0) if au else 0}
    verdict = PASS if all(v["ok"] for v in per.values()) else FAIL
    return {"verdict": verdict, "per_campaign": per}


def opt_compare(new, old):
    mfr_gain = old["mfr"] - new["mfr"]
    blc_gain = new["blc8_mean"] - old["blc8_mean"]
    reg = regressed(new, old)
    return {"mfr_old": old["mfr"], "mfr_new": new["mfr"], "mfr_gain": mfr_gain,
            "blc8_old": old["blc8_mean"], "blc8_new": new["blc8_mean"],
            "blc_gain": blc_gain, "regressed": reg,
            "beats": (mfr_gain >= MFR_GAIN_MIN or blc_gain >= BLC_GAIN_MIN)
            and not reg}


def g_opt(systems):
    as_written = opt_compare(systems["full"], systems["rules"])
    return {"verdict": PASS if as_written["beats"] else FAIL,
            "as_written": as_written,
            "optimiser_marginal": opt_compare(systems["full"],
                                              systems["no-optimiser"]),
            "rules_opt_vs_rules": opt_compare(systems["no-prior"],
                                              systems["rules"])}


def g_abl(systems, tuning):
    table = []
    for name in ABL_ORDER:
        st = systems[name]
        row = {"name": name, "label": LABELS[name]}
        row.update(dict((k, v) for k, v in st.items() if k != "per_family"))
        row["per_family"] = dict((f, {"failures": st["per_family"][f]["failures"],
                                      "mfr": st["per_family"][f]["mfr"],
                                      "blc8_mean": st["per_family"][f]["blc8_mean"]})
                                 for f in sorted(st["per_family"]))
        table.append(row)
    return {"verdict": REPORTED, "table": table, "tuning_curve": tuning["curve"]}


def load_context(report_dir):
    """The tuning context frozen when this report was first written, or None."""
    p = os.path.join(report_dir, CONTEXT_NAME)
    if not os.path.isfile(p):
        return None
    obj = _read_json(p)
    if obj.get("schema") != CONTEXT_SCHEMA or "context" not in obj:
        raise EvalError("%s is not a %s file" % (p, CONTEXT_SCHEMA))
    return obj["context"]


def tuning_context():
    p = _read_json(os.path.join(prior.REPORT_DIR, prior.REPORT_NAME))
    o = _read_json(os.path.join(HERE, "optimise", "G-OPT.json"))
    curve = []
    for k in ("rules", "round-1", "round-2", "round-3", "round-4", "round-5"):
        st = (o.get("systems") or {}).get(k)
        if st is None:
            continue
        curve.append({"system": k, "failures": st["failures"], "mfr": st["mfr"],
                      "blc8_mean": st["blc8_mean"]})
    return {"prior": {"verdict": p["verdict"], "n": p["n_tuning"],
                      "rules_pass": p["systems"]["rules"]["pass"],
                      "real_pass": p["systems"]["real"]["pass"],
                      "shuffled_mean_pass": p["shuffled_mean_pass"],
                      "gain": p["systems"]["real"]["pass"]
                      - p["systems"]["rules"]["pass"],
                      "control_share": p["shuffled_mean_pass"]
                      - p["systems"]["rules"]["pass"],
                      "fingerprint_share": p["systems"]["real"]["pass"]
                      - p["shuffled_mean_pass"]},
            "optimiser": {"verdict": o["verdict"], "enabled": o["enabled"],
                          "final": o["final"]},
            "curve": curve}


# --- (C9) run and report ------------------------------------------------------

def run(work, *, manifest="test", baselines=None, gdet_n=None, streams=6,
        report_dir=REPORT_DIR, binary=None, attempt_fn=None, probe_fn=None,
        snap_fn=None, hooks=None, quiet=False):
    binary = binary or campaign.BINARY_DEFAULT
    if not isinstance(streams, int) or isinstance(streams, bool) \
            or not 1 <= streams <= campaign.MAX_STREAMS:
        raise EvalError("streams: %r is not an int in 1..%d"
                        % (streams, campaign.MAX_STREAMS))
    if manifest == "test":
        if attempt_fn or probe_fn or snap_fn or hooks or baselines \
                or gdet_n is not None:
            raise EvalError("the evaluation runs the real system on the sealed "
                            "baselines: no fakes, no --baselines, no --gdet-n")
        verify_seal()
        gdet = gdet_ids(lock_test_ids())
        base_sha = dict(SEALED)
        plan = plan_of({"source": "test", "sha256": campaign.manifest_sha("test"),
                        "n": len(lock_test_ids())}, False, gdet, streams, binary,
                       base_sha)
        write_lock(report_dir, plan)
        opened = open_split("test")
        meta = meta_of(opened)
        if meta["sha256"] != plan["manifest"]["sha256"] \
                or meta["n"] != plan["manifest"]["n"]:
            raise EvalError("the opened test split (%s, n %d) differs from the "
                            "locked plan (%s, n %d)" % (meta["sha256"], meta["n"],
                                                        plan["manifest"]["sha256"],
                                                        plan["manifest"]["n"]))
        base = load_baselines(meta, report_dir)
    else:
        if os.path.abspath(report_dir) == os.path.abspath(REPORT_DIR):
            raise EvalError("a rehearsal writes its report outside evaluate/ "
                            "(--report-dir)")
        if set(baselines or ()) != set(baseline.SYSTEMS) \
                or not (isinstance(gdet_n, int) and not isinstance(gdet_n, bool)
                        and gdet_n >= 1):
            raise EvalError("a rehearsal needs --baselines and --gdet-n")
        opened = open_split(manifest)
        meta = meta_of(opened)
        gdet = sorted(meta["ids"][:gdet_n])
        base_sha = dict((s, _sha256_of_file(baselines[s])) for s in baseline.SYSTEMS)
        plan = plan_of({"source": meta["source"], "sha256": meta["sha256"],
                        "n": meta["n"]}, True, gdet, streams, binary, base_sha)
        write_lock(report_dir, plan)
        base = load_baselines(meta, report_dir, sources=baselines)
    os.makedirs(work, exist_ok=True)
    cache = new_cache()
    outs = {}
    bundles = {}
    counts = {}
    infos = []
    for spec in CAMPAIGNS:
        name = spec[0]
        outs[name], bundles[name], counts[name] = run_one(
            spec, work, meta, gdet, cache, streams=streams, binary=binary,
            attempt_fn=attempt_fn, probe_fn=probe_fn, snap_fn=snap_fn,
            hooks=hooks, quiet=quiet)
    knobs = schema.load_knobs()
    for spec in CAMPAIGNS:
        name = spec[0]
        rp = campaign.replay(outs[name])
        info = baseline._write_or_match(report_dir, BUNDLE_NAME % name, bundles[name])
        infos.append({"name": name, "system": spec[1], "ablate": list(spec[2]),
                      "counts": counts[name],
                      "replay": {"ok": rp["ok"], "decisions": rp["decisions"],
                                 "hook_decisions": rp["hook_decisions"],
                                 "mismatches": rp["mismatches"][:20]},
                      "qual": qual_of(outs[name], bundles[name]["attempts"], knobs),
                      "bundle": info})
    fid = {"full": fid_rows(outs["full"], bundles["full"]),
           "b0-template": fid_rows(outs["b0-template"], bundles["b0-template"])}
    gdet_facts = {"compare": campaign.compare(outs["gdet-1"], outs["gdet-2"]),
                  "vs_full": compare_subset(outs["full"], outs["gdet-1"], gdet)}
    sc = seal_compare(bundles["b0-template"], base["b0-template"])
    s = remedies.static_scan()
    static = {"remedies_scan_ok": s["ok"], "violations": s["violations"],
              "flag_literals": flag_literals(knobs)}
    runs = {"$comment": HEADER, "schema": RUNS_SCHEMA,
            "plan_sha256": campaign.sha(plan),
            "meta": dict((k, v) for k, v in meta.items() if k != "rows"),
            "gdet_ids": gdet, "campaigns": infos, "fid": fid, "gdet": gdet_facts,
            "seal_compare": sc, "static": static,
            "baselines": dict((s, {"source": "sealed" if not meta["rehearsal"]
                                   else BASELINE_COPY % s,
                                   "file_sha256": base_sha[s]})
                              for s in baseline.SYSTEMS)}
    _dump_json(os.path.join(report_dir, RUNS_NAME), runs)
    return report(report_dir, quiet=quiet)


def report(report_dir=REPORT_DIR, *, write=True, quiet=False):
    rpath = os.path.join(report_dir, RUNS_NAME)
    if not os.path.isfile(rpath):
        raise EvalError("report: %s is missing" % rpath)
    runs = _read_json(rpath)
    lpath = os.path.join(report_dir, LOCK_NAME)
    if not os.path.isfile(lpath):
        raise EvalError("report: %s is missing" % lpath)
    lock = _read_json(lpath)
    if lock.get("plan_sha256") != runs.get("plan_sha256"):
        raise EvalError("report: the lock records plan %s, runs.json %s"
                        % (str(lock.get("plan_sha256"))[:12],
                           str(runs.get("plan_sha256"))[:12]))
    bundles = {}
    for c in runs["campaigns"]:
        bpath = os.path.join(report_dir, c["bundle"]["file"])
        if _sha256_of_file(bpath) != c["bundle"]["sha256"]:
            raise EvalError("bundle drift: %s" % bpath)
        bundles[c["name"]] = baseline.read_bundle(bpath)
    base = load_baselines(runs["meta"], report_dir)
    ends = dict((name, dict((g["geometry_id"], g) for g in b["geometries"]))
                for name, b in bundles.items())
    ends_b0t = dict((g["geometry_id"], g) for g in base["b0-template"]["geometries"])
    ends_b0l = dict((g["geometry_id"], g) for g in base["b0-lhs"]["geometries"])
    ids = runs["meta"]["ids"]
    systems = {}
    for name in ("full", "no-preflight", "no-remedies", "no-prior", "no-optimiser",
                 "rules"):
        systems[name] = system_stats(ends[name], ids)
    systems["b0-template"] = system_stats(ends_b0t, ids)
    systems["b0-lhs"] = system_stats(ends_b0l, ids)
    tier0 = runs["meta"]["tier0_ids"]
    replays = dict((c["name"], c["replay"]) for c in runs["campaigns"])
    gates_const = schema.load_gates()
    tuning = load_context(report_dir)
    if tuning is None:
        tuning = tuning_context()
        if write:
            _dump_json(os.path.join(report_dir, CONTEXT_NAME),
                       {"$comment": HEADER, "schema": CONTEXT_SCHEMA,
                        "note": "the tuning context (prior/aml/G-PRIOR.json and "
                                "optimise/G-OPT.json) as it stood when this "
                                "evaluation's report was first written",
                        "context": tuning})
    gate_reps = {
        "G-FAIL": g_fail(systems["full"], systems["b0-template"],
                         [(ends_b0t[i]["failure"] is True,
                           ends["full"][i]["failure"] is True) for i in ids]),
        "G-BLC-0": g_blc0(ends["full"], ends_b0t, tier0,
                          [i for i in ids if i not in set(tier0)]),
        "G-QUAL": g_qual(dict((c["name"], c["qual"]) for c in runs["campaigns"]),
                         runs["static"]),
        "G-FID": g_fid(runs["fid"]["full"], runs["fid"]["b0-template"],
                       runs["seal_compare"]["equal"]),
        "G-COST": g_cost(ends["full"], ends_b0t, ids, bundles["full"]["end"],
                         bundles["full"]["campaign"], gates_const),
        "G-DET": g_det(runs["gdet"]["compare"], replays["gdet-1"], replays["gdet-2"],
                       runs["gdet"]["vs_full"]),
        "G-EXPL": g_expl(dict((c["name"], bundles[c["name"]]["summary"]["audit"])
                              for c in runs["campaigns"])),
        "G-OPT": g_opt(systems),
        "G-ABL": g_abl(systems, tuning)}
    decisions = {}
    for name in ("full", "no-preflight", "no-remedies", "no-prior", "no-optimiser"):
        cnt = {}
        for line in bundles[name]["records"]:
            rec = line.get("record") or {}
            if rec.get("layer") in ("prior", "optimiser"):
                key = str(rec.get("rule_id", ""))
                cnt[key] = cnt.get(key, 0) + 1
        decisions[name] = cnt
    camps = []
    for c in runs["campaigns"]:
        e = bundles[c["name"]]["end"]
        camps.append({"name": c["name"], "system": c["system"], "ablate": c["ablate"],
                      "n_geometries": e["n_geometries"], "n_rows": e["n_rows"],
                      "terminals": e["terminals"], "harness_errors": e["harness_errors"],
                      "wall_seconds": e["wall_seconds"], "peak_rss_mib": e["peak_rss_mib"],
                      "peak_frac": e["peak_frac"], "max_live_mesher": e["max_live_mesher"],
                      "orphans": len(e["orphans"]), "streams": e["streams"],
                      "meshed": c["counts"]["meshed"], "reused": c["counts"]["reused"],
                      "ram_guarded": c["counts"]["ram_guarded"],
                      "guard_probes": c["counts"]["guard_probes"],
                      "replay_ok": c["replay"]["ok"],
                      "audit_ok": gate_reps["G-EXPL"]["per_campaign"][c["name"]]["ok"],
                      "audit_sample": bundles[c["name"]]["summary"]["audit_sample"],
                      "bundle": c["bundle"]})
    rep = {"$comment": HEADER, "schema": REPORT_SCHEMA,
           "date": time.strftime("%Y-%m-%d"),
           "rehearsal": runs["meta"]["rehearsal"],
           "plan_sha256": runs["plan_sha256"], "opened_at": lock["opened_at"],
           "git_sha": bundles["full"]["campaign"]["git_sha"],
           "binary_sha256": bundles["full"]["campaign"]["binary_sha256"],
           "split": {"source": runs["meta"]["source"], "sha256": runs["meta"]["sha256"],
                     "n": runs["meta"]["n"], "families": runs["meta"]["families"],
                     "tier0_n": len(tier0)},
           "seal": runs["baselines"], "gdet_ids": runs["gdet_ids"],
           "campaigns": camps, "systems": systems, "gates": gate_reps,
           "headline": {"G-FAIL": gate_reps["G-FAIL"]["verdict"],
                        "G-BLC-0": gate_reps["G-BLC-0"]["verdict"]},
           "decisions": decisions, "tuning": tuning,
           "departures": list(DEPARTURES), "doi": DOI}
    if write:
        _dump_json(os.path.join(report_dir, REPORT_NAME), rep)
        _write_text(os.path.join(report_dir, REPORT_MD), report_md(rep))
    if not quiet:
        for ln in verdict_lines(rep):
            print(ln)
    return rep


def _fmt_list(vals):
    return ",".join(str(v) for v in vals) if vals else "none"


def _b(x):
    return str(bool(x))


def verdict_lines(rep):
    """One line per gate, in gate order (numbers by explain.fmt)."""
    g = rep["gates"]
    out = []
    f = g["G-FAIL"]
    b0t = rep["systems"]["b0-template"]
    out.append("G-FAIL %s: MFR full %s %s vs B0-template %s (4 x %s <= %s %s); "
               "95 %% upper %s (<= 0.10 %s); McNemar b %s c %s p %s (< 0.01 %s); "
               "families worse %s"
               % (f["verdict"], explain.fmt(float(f["mfr_full"])),
                  explain.fmt([float(v) for v in rep["systems"]["full"]["mfr_ci"]]),
                  explain.fmt(float(f["mfr_b0"])),
                  str(rep["systems"]["full"]["failures"]),
                  str(b0t["failures"]),
                  _b(f["conditions"]["ratio_le_quarter"]),
                  explain.fmt(float(f["upper"])),
                  _b(f["conditions"]["upper_le_10pct"]),
                  str(f["mcnemar"]["b"]), str(f["mcnemar"]["c"]),
                  explain.fmt(float(f["mcnemar"]["p"])),
                  _b(f["conditions"]["mcnemar_p_lt_001"]),
                  _fmt_list(f["worse_families"])))
    bl = g["G-BLC-0"]
    if bl["verdict"] == UNDECIDED and bl["n"] == 0:
        out.append("G-BLC-0 UNDECIDED: no tier-0 geometry")
    else:
        out.append("G-BLC-0 %s: tier 0 (%s D/F commensurate) BLC_8 %s (>= 0.90 %s), "
                   "BLC_full %s (>= 0.80 %s), a priori; B0-template %s / %s"
                   % (bl["verdict"], str(bl["n"]),
                      explain.fmt(float(bl["blc8_mean"])),
                      _b(bl["conditions"]["blc8_ge_090"]),
                      explain.fmt(float(bl["blc_full_mean"])),
                      _b(bl["conditions"]["blc_full_ge_080"]),
                      explain.fmt(float(bl["b0"]["blc8_mean"])),
                      explain.fmt(float(bl["b0"]["blc_full_mean"]))))
    q = g["G-QUAL"]
    out.append("G-QUAL %s: %s configs, %s off the reference quality block, "
               "%s of %s edits outside the whitelist, %s config sha mismatches, "
               "%s forbidden-flag literals, remedies scan ok %s"
               % (q["verdict"], str(q["totals"]["configs"]),
                  str(q["totals"]["quality_bad"]), str(q["totals"]["edits_bad"]),
                  str(q["totals"]["edits"]), str(q["totals"]["sha_bad"]),
                  str(len(q["static"]["flag_literals"])),
                  _b(q["static"]["remedies_scan_ok"])))
    fd = g["G-FID"]
    out.append("G-FID %s: %s family metrics compared, worse %s, re-measured "
               "B0-template equal to the seal %s"
               % (fd["verdict"], str(fd["compared"]),
                  _fmt_list(["%s:%s" % (w[0], w[1]) for w in fd["worse"]]),
                  _b(fd["seal_equal"])))
    co = g["G-COST"]
    items = co["items"]
    out.append("G-COST %s: cells %s (%s), over budget %s (%s), wall %s s at %s "
               "streams (%s), peak %s of RAM (%s), orphans %s (%s)"
               % (co["verdict"], explain.fmt(items["cells"]["ratio"])
                  if items["cells"]["ratio"] is not None else "absent",
                  items["cells"]["verdict"], str(items["budget"]["n_over"]),
                  items["budget"]["verdict"],
                  explain.fmt(float(items["wall"]["seconds"])),
                  str(items["wall"]["streams"]), items["wall"]["verdict"],
                  explain.fmt(float(items["ram"]["peak_frac"])),
                  items["ram"]["verdict"], str(items["orphans"]["n"]),
                  items["orphans"]["verdict"]))
    dt = g["G-DET"]
    out.append("G-DET %s: %s/%s rows, %s row diffs, content %s/%s, replays %s %s"
               % (dt["verdict"], str(dt["compare"]["rows_a"]),
                  str(dt["compare"]["rows_b"]), str(dt["compare"]["n_row_diffs"]),
                  str(dt["compare"]["content_equal"]),
                  str(dt["compare"]["content_compared"]),
                  _b(dt["replay_1"]["ok"]), _b(dt["replay_2"]["ok"])))
    ex = g["G-EXPL"]
    out.append("G-EXPL %s: %s campaigns audited, %s not ok"
               % (ex["verdict"], str(len(ex["per_campaign"])),
                  str(sum(1 for v in ex["per_campaign"].values() if not v["ok"]))))
    op = g["G-OPT"]
    aw = op["as_written"]
    out.append("G-OPT %s: full vs rules MFR %s -> %s (gain %s), BLC_8 %s -> %s "
               "(gain %s), regressed %s; optimiser marginal MFR %s -> %s"
               % (op["verdict"], explain.fmt(float(aw["mfr_old"])),
                  explain.fmt(float(aw["mfr_new"])), explain.fmt(float(aw["mfr_gain"])),
                  explain.fmt(float(aw["blc8_old"])), explain.fmt(float(aw["blc8_new"])),
                  explain.fmt(float(aw["blc_gain"])), _fmt_list(aw["regressed"]),
                  explain.fmt(float(op["optimiser_marginal"]["mfr_old"])),
                  explain.fmt(float(op["optimiser_marginal"]["mfr_new"]))))
    ab = g["G-ABL"]
    out.append("G-ABL REPORTED: %s"
               % ", ".join("%s MFR %s BLC_8 %s"
                           % (row["label"], explain.fmt(float(row["mfr"])),
                              explain.fmt(float(row["blc8_mean"])))
                           for row in ab["table"]))
    return out


# --- (C10) the results page ---------------------------------------------------

def _r3(v):
    return "%.3f" % v if v is not None else "absent"


def report_md(rep):
    """THE results page - a pure function of the report dict."""
    L = []
    L.append("# Autonomous mesh setup: the held-out evaluation (docs/15 §F)")
    L.append("")
    if rep["rehearsal"]:
        L.append("**REHEARSAL on tuning geometries - not a result.**")
        L.append("")
    L.append("- date: %s" % rep["date"])
    L.append("- split: %s, sha256 %s, %d geometries"
             % (rep["split"]["source"], rep["split"]["sha256"], rep["split"]["n"]))
    L.append("- binary sha256: %s" % rep["binary_sha256"])
    L.append("- git sha: %s" % rep["git_sha"])
    L.append("- plan sha256: %s" % rep["plan_sha256"])
    L.append("- opened at: %s" % rep["opened_at"])
    L.append("")
    L.append("The evaluation is in the style of the ablation tables of DOI "
             "10.3389/frobt.2025.1566623; it reproduces nothing of that paper, and "
             "its numbers are not comparable with ours.")
    L.append("")
    L.append("Every BLC number is a priori (docs/15 §D.3) until the solved y+ check "
             "(G-YPLUS) exists.")
    L.append("")
    L.append("Every rule on this page was fixed before the test split was opened. "
             "A missed gate is reported as missed and is not re-run with other "
             "settings; the test split is now spent, and a second claim needs a "
             "fresh test seed and a new lock (docs/15 §F).")
    L.append("")
    g = rep["gates"]
    lines = dict(zip(("G-FAIL", "G-BLC-0", "G-QUAL", "G-FID", "G-COST", "G-DET",
                      "G-EXPL", "G-OPT", "G-ABL"), verdict_lines(rep)))
    L.append("## Headlines")
    L.append("")
    L.append(lines["G-FAIL"])
    L.append("")
    L.append(lines["G-BLC-0"])
    L.append("")
    L.append("## Gates")
    L.append("")
    L.append("| gate | verdict | numbers |")
    L.append("|---|---|---|")
    for name in ("G-FAIL", "G-BLC-0", "G-QUAL", "G-FID", "G-COST", "G-DET",
                 "G-EXPL", "G-OPT", "G-ABL"):
        ln = lines[name]
        v = ln.split(": ", 1)[0].split(" ", 1)[1]
        nums = ln.split(": ", 1)[1] if ": " in ln else ""
        L.append("| %s | %s | %s |" % (name, v, nums))
    L.append("")
    L.append("## Mesh failure per family (G-FAIL)")
    L.append("")
    L.append("| family | n | B0-template | B0-LHS best | full | full MFR [95 % CI] | full strict |")
    L.append("|---|---|---|---|---|---|---|")
    fams = [f for f in FAMILIES if f in rep["systems"]["full"]["per_family"]]
    for f in fams:
        pf = rep["systems"]["full"]["per_family"][f]
        b0 = rep["systems"]["b0-template"]["per_family"].get(f, {})
        bl = rep["systems"]["b0-lhs"]["per_family"].get(f, {})
        L.append("| %s | %d | %d | %d | %d | %s [%s, %s] | %d |"
                 % (f, pf["n"], b0.get("failures", 0), bl.get("failures", 0),
                    pf["failures"], _r3(pf["mfr"]), _r3(pf["mfr_ci"][0]),
                    _r3(pf["mfr_ci"][1]), pf["strict"]))
    allf = rep["systems"]["full"]
    L.append("| all | %d | %d | %d | %d | %s [%s, %s] | %d |"
             % (allf["n"], rep["systems"]["b0-template"]["failures"],
                rep["systems"]["b0-lhs"]["failures"], allf["failures"],
                _r3(allf["mfr"]), _r3(allf["mfr_ci"][0]), _r3(allf["mfr_ci"][1]),
                allf["strict"]))
    L.append("")
    bl0 = g["G-BLC-0"]
    L.append("## Boundary-layer capture, a priori (G-BLC-0)")
    L.append("")
    if bl0["n"]:
        L.append("Tier 0 (%d D/F commensurate geometries): full BLC_8 %s, "
                 "BLC_full %s; B0-template %s / %s; a priori."
                 % (bl0["n"], _r3(bl0["blc8_mean"]), _r3(bl0["blc_full_mean"]),
                    _r3(bl0["b0"]["blc8_mean"]), _r3(bl0["b0"]["blc_full_mean"])))
    else:
        L.append("No tier-0 geometry.")
    L.append("")
    L.append("| family | n | BLC_8 | BLC_full | with CAPABILITY-LIMITED patches |")
    L.append("|---|---|---|---|---|")
    for f, st in bl0["other"]["per_family"].items():
        L.append("| %s | %d | %s | %s | %d |"
                 % (f, st["n"], _r3(st["blc8_mean"]), _r3(st["blc_full_mean"]),
                    st["n_capability_limited"]))
    L.append("")
    L.append("CAPABILITY-LIMITED patches lose their layers on a snapped curved "
             "wall, which this mesher cannot grow (docs/15 §I-1); they are "
             "reported, never gated.")
    L.append("")
    fd = g["G-FID"]
    L.append("## Fidelity (G-FID)")
    L.append("")
    L.append("| family | full meshes | B0 meshes | feature_tolerance 0 on sharp bodies | metric | full median | B0 median | no worse |")
    L.append("|---|---|---|---|---|---|---|---|")
    for f, st in fd["per_family"].items():
        for m, _op in FID_METRICS:
            mt = st["metrics"][m]
            L.append("| %s | %d | %d | %d | %s | %s | %s | %s |"
                     % (f, st["n_full"], st["n_b0"], st["n_ft0_sharp"], m,
                        _r3(mt["full"]), _r3(mt["b0"]),
                        "-" if mt["ok"] is None else _b(mt["ok"])))
    L.append("")
    L.append("A mesh snapped with feature_tolerance 0 captures no feature edge "
             "and scores a share of 0.")
    L.append("")
    co = g["G-COST"]
    it = co["items"]
    L.append("## Cost (G-COST)")
    L.append("")
    if it["cells"]["median_full"] is not None:
        L.append("- cells: median full %s / median B0-template %s on %d "
                 "both-pass geometries, ratio %s (%s, <= %.2f)"
                 % (str(it["cells"]["median_full"]), str(it["cells"]["median_b0"]),
                    it["cells"]["n_both"], _r3(it["cells"]["ratio"]),
                    it["cells"]["verdict"], CELLS_RATIO_MAX))
    else:
        L.append("- cells: no geometry passed in both systems (%s)"
                 % it["cells"]["verdict"])
    L.append("- over budget: %d geometries over the cell budget (%s)"
             % (it["budget"]["n_over"], it["budget"]["verdict"]))
    L.append("- wall: %.1f s at %d streams, the best a 12-stream run could do is %.1f s "
             "(%s, <= %.0f s)" % (it["wall"]["seconds"], it["wall"]["streams"],
                                  it["wall"]["bound_12"], it["wall"]["verdict"],
                                  WALL_MAX_S))
    L.append("- RAM: peak %.3f of total (%s, <= %.2f)" % (it["ram"]["peak_frac"],
                                                          it["ram"]["verdict"],
                                                          RAM_FRAC_MAX))
    L.append("- orphans: %d (%s)" % (it["orphans"]["n"], it["orphans"]["verdict"]))
    L.append("")
    dt = g["G-DET"]
    L.append("## Determinism (G-DET)")
    L.append("")
    L.append("The 20 G-DET geometries, fixed before any outcome: %s."
             % " ".join(rep["gdet_ids"]))
    L.append("")
    L.append("gdet-1 against gdet-2: %d/%d rows, %d row diffs, content %d/%d equal; "
             "verdict %s." % (dt["compare"]["rows_a"], dt["compare"]["rows_b"],
                              dt["compare"]["n_row_diffs"],
                              dt["compare"]["content_equal"],
                              dt["compare"]["content_compared"], dt["verdict"]))
    L.append("")
    L.append("Replay through rules.py and remedies.py: gdet-1 ok %s (%s decisions), "
             "gdet-2 ok %s (%s decisions)."
             % (_b(dt["replay_1"]["ok"]), str(dt["replay_1"].get("decisions", 0)),
                _b(dt["replay_2"]["ok"]), str(dt["replay_2"].get("decisions", 0))))
    L.append("")
    vs = dt["vs_full"]
    L.append("The full campaign against gdet-1 on these ids: %d rows, %d row diffs, "
             "content %d/%d equal; ok %s."
             % (vs["rows"], vs["n_row_diffs"], vs["content_equal"],
                vs["content_compared"], _b(vs["ok"])))
    L.append("")
    q = g["G-QUAL"]
    ex = g["G-EXPL"]
    L.append("## Guards (G-QUAL, G-EXPL)")
    L.append("")
    L.append("%d configs, %d off the reference quality block, %d of %d edits "
             "outside the whitelist, %d config sha mismatches, %d forbidden-flag "
             "literals, remedies scan ok %s; %d campaigns audited, %d not ok."
             % (q["totals"]["configs"], q["totals"]["quality_bad"],
                q["totals"]["edits_bad"], q["totals"]["edits"],
                q["totals"]["sha_bad"], len(q["static"]["flag_literals"]),
                _b(q["static"]["remedies_scan_ok"]), len(ex["per_campaign"]),
                sum(1 for v in ex["per_campaign"].values() if not v["ok"])))
    L.append("")
    for name in sorted(ex["per_campaign"]):
        st = ex["per_campaign"][name]
        n_un = len(st["untemplated"]) if isinstance(st["untemplated"], list) \
            else st["untemplated"]
        L.append("- %s: audit ok %s, %d rows, %d untemplated."
                 % (name, _b(st["ok"]), st["n_rows"], n_un))
    L.append("")
    ab = g["G-ABL"]
    L.append("## Ablation (G-ABL)")
    L.append("")
    L.append("| system | MFR [95 % CI] | strict | BLC_8 | BLC_full | cells median |")
    L.append("|---|---|---|---|---|---|")
    for row in ab["table"]:
        L.append("| %s | %s [%s, %s] | %d | %s | %s | %s |"
                 % (row["label"], _r3(row["mfr"]), _r3(row["mfr_ci"][0]),
                    _r3(row["mfr_ci"][1]), row["strict"], _r3(row["blc8_mean"]),
                    _r3(row["blc_full_mean"]), str(row["cells_median"])))
    L.append("")
    L.append("| system | A | B | D | E | F | G |")
    L.append("|---|---|---|---|---|---|---|")
    for row in ab["table"]:
        L.append("| %s | %s |"
                 % (row["label"],
                    " | ".join(_r3(row["per_family"][f]["mfr"])
                               if f in row["per_family"] else "-" for f in FAMILIES)))
    L.append("")
    L.append("### G-OPT")
    L.append("")

    def _opt_line(tag, oc):
        return "%s: MFR %s -> %s (gain %s), BLC_8 %s -> %s (gain %s), regressed %s" \
            % (tag, _r3(oc["mfr_old"]), _r3(oc["mfr_new"]), _r3(oc["mfr_gain"]),
               _r3(oc["blc8_old"]), _r3(oc["blc8_new"]), _r3(oc["blc_gain"]),
               _fmt_list(oc["regressed"]))

    L.append("- " + _opt_line("as written (full vs rules + remedies only)",
                              g["G-OPT"]["as_written"]) + ".")
    L.append("- " + _opt_line("optimiser marginal (full vs -optimiser)",
                              g["G-OPT"]["optimiser_marginal"]) + ".")
    L.append("- " + _opt_line("rules+opt vs rules", g["G-OPT"]["rules_opt_vs_rules"]) + ".")
    L.append("")
    L.append("### The per-round tuning curve (tuning split, not the result)")
    L.append("")
    L.append("| system | failures | MFR | BLC_8 |")
    L.append("|---|---|---|---|")
    for c in ab["tuning_curve"]:
        L.append("| %s | %d | %s | %s |" % (c["system"], c["failures"], _r3(c["mfr"]),
                                            _r3(c["blc8_mean"])))
    L.append("")
    L.append("## What the prior and the optimiser decided on the test split")
    L.append("")
    for name in sorted(rep["decisions"]):
        cnt = rep["decisions"][name]
        L.append("- %s: %s" % (name, ", ".join("%s %d" % (k, v)
                                               for k, v in sorted(cnt.items()))
                               or "no prior or optimiser decisions"))
    L.append("")
    L.append("## The campaigns")
    L.append("")
    L.append("| campaign | system | ablate | geometries | rows | meshed | reused | RAM-guarded | wall s | peak RSS MiB | harness errors | replay | audit |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in rep["campaigns"]:
        L.append("| %s | %s | %s | %d | %d | %d | %d | %d | %.1f | %.1f | %d | %s | %s |"
                 % (c["name"], c["system"], ", ".join(c["ablate"]) or "none",
                    c["n_geometries"], c["n_rows"], c["meshed"], c["reused"],
                    c["ram_guarded"], c["wall_seconds"], c["peak_rss_mib"],
                    c["harness_errors"], _b(c["replay_ok"]), _b(c["audit_ok"])))
    L.append("")
    t = rep["tuning"]
    L.append("## Tuning context (not the result)")
    L.append("")
    pr = t["prior"]
    L.append("On the tuning split: rules-only %s of %s tuning geometries pass at "
             "attempt 1, the prior %s, the shuffled control %s on average: of the "
             "%s-geometry gain, %s is reached with shuffled fingerprints and %s is "
             "the fingerprint's own."
             % (explain.fmt(float(pr["rules_pass"])), explain.fmt(float(pr["n"])),
                explain.fmt(float(pr["real_pass"])),
                explain.fmt(float(pr["shuffled_mean_pass"])),
                explain.fmt(float(pr["gain"])),
                explain.fmt(float(pr["control_share"])),
                explain.fmt(float(pr["fingerprint_share"]))))
    L.append("")
    L.append("The optimiser's tuning verdict: %s (enabled %s); final MFR gain %s, "
             "BLC_8 gain %s, regressed %s."
             % (t["optimiser"]["verdict"], _b(t["optimiser"]["enabled"]),
                _r3(t["optimiser"]["final"].get("mfr_gain")),
                _r3(t["optimiser"]["final"].get("blc_gain")),
                _fmt_list(t["optimiser"]["final"].get("regressed_families") or [])))
    L.append("")
    L.append("## Departures")
    L.append("")
    for s in rep["departures"]:
        L.append("- %s" % s)
    return "\n".join(L) + "\n"


# --- (C11) check and the CLI --------------------------------------------------

def check(report_dir=REPORT_DIR):
    """EVAL.json, runs.json, the lock, the bundles, the baselines, the report
    and the results page agree; a broken item is recorded, never raised."""
    items = []

    def item(name, fn):
        try:
            fn()
            items.append({"name": name, "ok": True, "why": ""})
        except Exception as e:  # noqa: BLE001 - the failure IS the result here
            items.append({"name": name, "ok": False, "why": "%s: %s"
                          % (type(e).__name__, e)})

    def _runs():
        p = os.path.join(report_dir, RUNS_NAME)
        if not os.path.isfile(p):
            raise EvalError("%s is missing" % p)
        return _read_json(p)

    def _eval_json():
        p = os.path.join(report_dir, REPORT_NAME)
        if not os.path.isfile(p):
            raise EvalError("%s is missing" % p)
        _read_json(p)

    def _lock():
        runs = _runs()
        p = os.path.join(report_dir, LOCK_NAME)
        if not os.path.isfile(p):
            raise EvalError("%s is missing" % p)
        lock = _read_json(p)
        if lock.get("plan_sha256") != runs.get("plan_sha256"):
            raise EvalError("the lock records plan %s, runs.json %s"
                            % (str(lock.get("plan_sha256"))[:12],
                               str(runs.get("plan_sha256"))[:12]))

    def _bundles():
        for c in _runs()["campaigns"]:
            p = os.path.join(report_dir, c["bundle"]["file"])
            if not os.path.isfile(p):
                raise EvalError("%s is missing" % p)
            if _sha256_of_file(p) != c["bundle"]["sha256"]:
                raise EvalError("bundle drift: %s" % p)

    def _baselines():
        runs = _runs()
        if runs["meta"]["rehearsal"]:
            for s in baseline.SYSTEMS:
                p = os.path.join(report_dir, BASELINE_COPY % s)
                if not os.path.isfile(p):
                    raise EvalError("%s is missing" % p)
                baseline.read_bundle(p)
        else:
            verify_seal()

    def _report():
        p = os.path.join(report_dir, REPORT_NAME)
        if not os.path.isfile(p):
            raise EvalError("%s is missing" % p)
        want = _read_json(p)
        got = report(report_dir, write=False, quiet=True)
        a, b = dict(got), dict(want)
        a.pop("date", None)
        b.pop("date", None)
        if campaign.sha(a) != campaign.sha(b):
            raise EvalError("the rebuilt report differs from %s" % REPORT_NAME)

    def _md():
        p = os.path.join(report_dir, REPORT_NAME)
        mp = os.path.join(report_dir, REPORT_MD)
        if not os.path.isfile(p) or not os.path.isfile(mp):
            raise EvalError("%s or %s is missing" % (p, mp))
        rep = _read_json(p)
        with open(mp, encoding="utf-8") as f:
            text = f.read()
        if report_md(rep) != text:
            raise EvalError("the results page differs from report_md(%s)"
                            % REPORT_NAME)

    item("EVAL.json", _eval_json)
    item("runs.json", _runs)
    item("lock", _lock)
    item("bundles", _bundles)
    item("baselines", _baselines)
    item("report", _report)
    item("results page", _md)
    verdict = PASS if all(i["ok"] for i in items) else FAIL
    return {"schema": CHECK_SCHEMA, "items": items, "verdict": verdict}


class _ArgParser(argparse.ArgumentParser):
    """Bad arguments exit 2 with `evaluate: <message>` (prior.py's pattern)."""

    def error(self, message):
        self.exit(2, "evaluate: %s\n" % message)


def _cli_plan(report_dir):
    seal = verify_seal()
    for s in baseline.SYSTEMS:
        print("[eval] seal %s ok %s" % (s, seal[s]["sha256"][:12]))
    n = len(lock_test_ids())
    print("[eval] test split: %d geometries in the lock (manifest sha %s), "
          "not opened" % (n, campaign.manifest_sha("test")[:12]))
    print("[eval] campaigns: %s" % ", ".join(c[0] for c in CAMPAIGNS))
    print("[eval] G-DET ids: %s" % " ".join(gdet_ids(lock_test_ids())))
    lpath = os.path.join(report_dir, LOCK_NAME)
    if not os.path.isfile(lpath):
        print("[eval] lock: absent")
    else:
        lock = _read_json(lpath)
        print("[eval] lock: present, plan %s, opened %s"
              % (str(lock.get("plan_sha256"))[:12], lock.get("opened_at")))
    return 0


def main(argv=None):
    ap = _ArgParser(prog="evaluate.py")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--work")
    ap.add_argument("--streams", type=int, default=6)
    ap.add_argument("--binary")
    ap.add_argument("--report-dir")
    ap.add_argument("--manifest", default="test")
    ap.add_argument("--baselines")
    ap.add_argument("--gdet-n", type=int, default=None)
    a = ap.parse_args(argv)
    rdir = a.report_dir or REPORT_DIR
    try:
        if a.selftest:
            return _selftest()
        if a.plan:
            return _cli_plan(rdir)
        if a.run:
            if not a.work:
                print("evaluate: --run needs --work", file=sys.stderr)
                return 2
            sources = None
            if a.baselines:
                parts = a.baselines.split(",")
                if len(parts) != 2:
                    raise EvalError("--baselines is two comma-separated paths "
                                    "(b0-template first)")
                sources = {"b0-template": parts[0], "b0-lhs": parts[1]}
            elif a.manifest != "test":
                raise EvalError("a rehearsal needs --baselines and --gdet-n")
            run(a.work, manifest=a.manifest, baselines=sources, gdet_n=a.gdet_n,
                streams=a.streams, report_dir=rdir, binary=a.binary)
            return 0
        if a.report:
            report(rdir)
            return 0
        if a.check:
            res = check(rdir)
            for it in res["items"]:
                print("[check] %s ok" % it["name"] if it["ok"]
                      else "[check] %s FAIL %s" % (it["name"], it["why"]))
            print("CHECK %s" % res["verdict"])
            return 0 if res["verdict"] == PASS else 1
    except (EvalError, campaign.CampaignError, baseline.BaselineError,
            optimise.OptError, split.SplitSealed, split.SplitError) as e:
        print("evaluate: %s" % e, file=sys.stderr)
        return 2
    print("evaluate: nothing to do (pass --selftest, --plan, --run, --report "
          "or --check)", file=sys.stderr)
    return 2


# --- (C13) the selftest --------------------------------------------------------

def _hand_end(gid, failure, blc8=0.0, blcf=0.0, cells=None, terminal=None,
              cl=(), seconds=1.0):
    return {"geometry_id": gid, "family": gid[0], "failure": failure,
            "strict_failure": failure, "blc8_a_priori": blc8,
            "blc_full_a_priori": blcf, "n_cells": cells,
            "terminal": terminal or ("EXHAUSTED" if failure else "PASS"),
            "reason": None, "capability_limited": list(cl), "seconds": seconds,
            "final_attempt": 1,
            "fingerprint": {"sharp_edge_length_m": 0.0}}


def _selftest():
    gates = schema.load_gates()
    knobs = schema.load_knobs()
    rows = dict((r["geometry_id"], r)
                for r in campaign.load_manifest("tuning", "rules"))
    F = {"attempt_fn": prior._wall_oracle, "probe_fn": campaign._fake_probe(1000),
         "snap_fn": campaign._fake_snap}
    IDS = ("D-1-010", "F-1-009", "E-1-004", "D-1-073")
    tmp = tempfile.mkdtemp()
    t0 = time.perf_counter()

    def group(name, fn):
        try:
            fn()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("SELFTEST FAIL: %s: %s: %s"
                  % (name, type(e).__name__, e), file=sys.stderr)
            shutil.rmtree(tmp, ignore_errors=True)
            return False
        print("[ok] %s" % name)
        return True

    def g1():
        names = [c[0] for c in CAMPAIGNS]
        assert names == ["full", "b0-template", "gdet-1", "gdet-2", "no-remedies",
                         "no-optimiser", "rules", "no-prior", "no-preflight"], names
        assert [c[3] for c in CAMPAIGNS]             == [False, False, False, False, True, True, True, True, True]
        assert all(c[1] in campaign.SYSTEMS for c in CAMPAIGNS)
        assert all(a in campaign.ABLATABLE for c in CAMPAIGNS for a in c[2])
        assert len(DEPARTURES) == 11
        seal = verify_seal()
        assert all(seal[s]["ok"] and seal[s]["sha256"] == SEALED[s]
                   for s in baseline.SYSTEMS)
        sdir = os.path.join(tmp, "seal")
        os.makedirs(sdir)
        shutil.copyfile(os.path.join(baseline.REPORT_DIR, baseline.LOCK_NAME),
                        os.path.join(sdir, baseline.LOCK_NAME))
        for s in baseline.SYSTEMS:
            src = os.path.join(baseline.REPORT_DIR, "sealed",
                               "test_%s.json.gz" % s)
            os.makedirs(os.path.join(sdir, "sealed"), exist_ok=True)
            shutil.copyfile(src, os.path.join(sdir, "sealed", "test_%s.json.gz" % s))
        tam = os.path.join(sdir, "sealed", "test_b0-lhs.json.gz")
        with open(tam, "rb") as f:
            data = bytearray(f.read())
        data[-1] ^= 0xFF
        with open(tam, "wb") as f:
            f.write(bytes(data))
        try:
            verify_seal(sdir)
            raise AssertionError("the tampered seal was accepted")
        except EvalError as e:
            assert "seal" in str(e), str(e)
        assert len(lock_test_ids()) == 180
        want = ("A-1-043 A-1-045 A-1-059 A-1-084 B-1-101 B-1-103 B-1-115 B-1-117 "
                "D-1-009 D-1-021 D-1-028 E-1-034 E-1-044 E-1-047 F-1-013 F-1-052 "
                "F-1-058 G-1-006 G-1-062 G-1-080").split()
        assert gdet_ids(lock_test_ids()) == want
        assert gdet_ids(["B-1-000", "A-1-000"], 5) == ["A-1-000", "B-1-000"]
        # the evaluation reads the seal where baseline.py keeps it, whatever the
        # report dir (load_sealed stubbed: the selftest never opens a sealed bundle)
        orig = baseline.load_sealed
        seen = []
        baseline.load_sealed = lambda s, m, report_dir=baseline.REPORT_DIR:             seen.append((s, m, report_dir)) or {"system": s}
        try:
            got = load_baselines({"rehearsal": False}, os.path.join(tmp, "elsewhere"))
        finally:
            baseline.load_sealed = orig
        assert sorted(got) == sorted(baseline.SYSTEMS), got
        assert [r for _, _, r in seen] == [baseline.REPORT_DIR] * 2, seen
        cdir = os.path.join(tmp, "c1")
        _dump_json(os.path.join(cdir, RUNS_NAME), {"meta": {"rehearsal": False},
                                                   "campaigns": [], "plan_sha256": "x"})
        items = dict((i["name"], i) for i in check(cdir)["items"])
        assert items["baselines"]["ok"] is True, items["baselines"]

    def g2():
        assert mcnemar(10, 0) == 0.001953125
        assert mcnemar(12, 3) == 0.03515625
        assert mcnemar(0, 0) == 1.0
        assert mcnemar(5, 5) == 1.0
        assert mcnemar(3, 1) == 0.625
        assert mcnemar(99, 0) == 3.1554436208840472e-30
        assert mcnemar(50, 2) == 6.123990203832363e-13
        ends = {"A-1-001": _hand_end("A-1-001", True),
                "A-1-002": _hand_end("A-1-002", False, blc8=1.0, cells=100),
                "D-1-001": _hand_end("D-1-001", False, blc8=0.5, cells=300),
                "D-1-002": _hand_end("D-1-002", True)}
        st = system_stats(ends, ["A-1-001", "A-1-002", "D-1-001", "D-1-002"])
        assert st["n"] == 4 and st["failures"] == 2 and st["mfr"] == 0.5
        assert st["mfr_ci"] == list(explain.clopper_pearson(2, 4))
        assert st["blc8_mean"] == 0.375 and st["cells_median"] == 200.0
        assert st["per_family"]["A"]["failures"] == 1
        assert st["per_family"]["D"]["failures"] == 1
        assert st["terminals"] == {"PASS": 2, "EXHAUSTED": 2}
        worse = dict(ends)
        worse["D-1-001"] = _hand_end("D-1-001", True, blc8=0.5, cells=300)
        assert family_worse(system_stats(worse, list(worse)), st) == ["D"]
        assert family_worse(st, system_stats(worse, list(worse))) == []
        lower = dict(ends)
        lower["A-1-002"] = _hand_end("A-1-002", False, blc8=0.5, cells=100)
        assert regressed(system_stats(lower, list(lower)), st) == ["A"]
        try:
            system_stats(ends, ["A-1-001", "Z-1-999"])
            raise AssertionError("a missing id was accepted")
        except EvalError as e:
            assert "system lacks" in str(e), str(e)

    def g3():
        b0 = dict(("A-1-%03d" % i, _hand_end("A-1-%03d" % i, True))
                  for i in range(50))
        b0.update(dict(("D-1-%03d" % i, _hand_end("D-1-%03d" % i, True))
                       for i in range(50)))
        full = dict(("A-1-%03d" % i, _hand_end("A-1-%03d" % i, i == 0))
                    for i in range(50))
        full.update(dict(("D-1-%03d" % i, _hand_end("D-1-%03d" % i, False, blc8=1.0))
                         for i in range(50)))
        ids = sorted(b0)
        s_b0 = system_stats(b0, ids)
        s_full = system_stats(full, ids)
        res = g_fail(s_full, s_b0, [(b0[i]["failure"] is True, full[i]["failure"] is True)
                                    for i in ids])
        assert res["verdict"] == PASS, res["conditions"]
        assert res["mcnemar"]["b"] == 99 and res["mcnemar"]["c"] == 0
        assert res["mcnemar"]["p"] == 3.1554436208840472e-30
        assert res["upper"] == 0.054459385392080624
        assert abs(res["ratio"] - 0.01) < 1e-12 and res["worse_families"] == []
        b0b = dict(("A-1-%03d" % i, _hand_end("A-1-%03d" % i, True))
                   for i in range(50))
        b0b.update(dict(("D-1-%03d" % i, _hand_end("D-1-%03d" % i, False, blc8=1.0))
                        for i in range(50)))
        fullb = dict(("A-1-%03d" % i, _hand_end("A-1-%03d" % i, False, blc8=1.0))
                     for i in range(50))
        fullb.update(dict(("D-1-%03d" % i, _hand_end("D-1-%03d" % i, i < 2))
                          for i in range(50)))
        ids_b = sorted(b0b)
        res_b = g_fail(system_stats(fullb, ids_b), system_stats(b0b, ids_b),
                       [(b0b[i]["failure"] is True, fullb[i]["failure"] is True)
                        for i in ids_b])
        assert res_b["conditions"]["ratio_le_quarter"] is True
        assert res_b["upper"] == 0.07038393247107012
        assert res_b["mcnemar"]["b"] == 50 and res_b["mcnemar"]["c"] == 2
        assert res_b["mcnemar"]["p"] == 6.123990203832363e-13
        assert res_b["conditions"]["mcnemar_p_lt_001"] is True
        assert res_b["conditions"]["no_family_worse"] is False
        assert res_b["worse_families"] == ["D"] and res_b["verdict"] == FAIL
        ef = {"D-1-000": _hand_end("D-1-000", False, blc8=1.0, blcf=1.0),
              "D-1-001": _hand_end("D-1-001", False, blc8=1.0, blcf=0.9),
              "D-1-002": _hand_end("D-1-002", False, blc8=0.8, blcf=0.6),
              "A-1-000": _hand_end("A-1-000", False, cl=("body",))}
        eb = dict((k, _hand_end(k, False)) for k in ef)
        res_c = g_blc0(ef, eb, ["D-1-000", "D-1-001", "D-1-002"], ["A-1-000"])
        assert res_c["verdict"] == PASS
        assert abs(res_c["blc8_mean"] - 0.9333333333333332) < 1e-15
        assert abs(res_c["blc_full_mean"] - 0.8333333333333334) < 1e-15
        assert res_c["other"]["per_family"]["A"]["n_capability_limited"] == 1
        assert res_c["other"]["capability_limited"] == [{"geometry_id": "A-1-000",
                                                         "patches": ["body"]}]
        assert g_blc0(ef, eb, [], ["A-1-000"])["verdict"] == UNDECIDED
        rules_e = dict(("A-1-%03d" % i, _hand_end("A-1-%03d" % i, i < 5,
                                                  blc8=0.5 if i == 0 else 0.0))
                       for i in range(10))
        rules_e.update(dict(("D-1-%03d" % i, _hand_end("D-1-%03d" % i, i < 3))
                            for i in range(10)))
        full_e = dict(("A-1-%03d" % i, _hand_end("A-1-%03d" % i, i < 5,
                                                 blc8=0.5 if i == 0 else 0.0))
                      for i in range(10))
        full_e.update(dict(("D-1-%03d" % i, _hand_end("D-1-%03d" % i, i < 2))
                           for i in range(10)))
        systems = {"full": system_stats(full_e, sorted(full_e)),
                   "rules": system_stats(rules_e, sorted(rules_e)),
                   "no-optimiser": system_stats(rules_e, sorted(rules_e)),
                   "no-prior": system_stats(rules_e, sorted(rules_e))}
        res_d = g_opt(systems)
        assert abs(res_d["as_written"]["mfr_gain"] - 0.05) < 1e-12
        assert res_d["as_written"]["beats"] is True and res_d["verdict"] == PASS
        assert res_d["rules_opt_vs_rules"]["beats"] is False
        full_e2 = dict(("A-1-%03d" % i, _hand_end("A-1-%03d" % i, i < 5, blc8=0.0))
                       for i in range(10))
        full_e2.update(dict(("D-1-%03d" % i, _hand_end("D-1-%03d" % i, i < 2))
                            for i in range(10)))
        systems2 = dict(systems)
        systems2["full"] = system_stats(full_e2, sorted(full_e2))
        res_d2 = g_opt(systems2)
        assert res_d2["as_written"]["regressed"] == ["A"]
        assert res_d2["verdict"] == FAIL
        assert wall_verdict(10000.0, 6, 500.0) == PASS
        assert wall_verdict(15000.0, 6, 500.0) == UNDECIDED
        assert wall_verdict(25000.0, 6, 500.0) == FAIL
        assert wall_verdict(15000.0, 6, 11000.0) == FAIL
        ff = [{"family": "B", "p99_over_hf": 0.01, "pinned_frac": 0.0,
               "feature_share": 0.0, "feature_tolerance": 0.0,
               "sharp_edge_length_m": 1.0}]
        fb = [{"family": "B", "p99_over_hf": 0.02, "pinned_frac": 0.01,
               "feature_share": 0.5, "feature_tolerance": 0.5,
               "sharp_edge_length_m": 1.0}]
        res_f = g_fid(ff, fb, True)
        assert res_f["verdict"] == FAIL
        assert res_f["worse"] == [["B", "feature_share"]]
        assert res_f["compared"] == 3
        assert res_f["per_family"]["B"]["n_ft0_sharp"] == 1
        ef2 = [{"family": "E", "p99_over_hf": 0.01, "pinned_frac": 0.0,
                "feature_share": None} for _ in range(2)]
        eb2 = [{"family": "E", "p99_over_hf": 0.01, "pinned_frac": 0.0,
                "feature_share": None} for _ in range(2)]
        res_f2 = g_fid(ef2, eb2, True)
        assert res_f2["verdict"] == PASS and res_f2["compared"] == 2
        assert g_fid(ff, [], True)["verdict"] == UNDECIDED
        assert g_fid(ff, fb, False)["verdict"] == UNDECIDED
        ce = {"A-1-000": _hand_end("A-1-000", False, cells=150),
              "A-1-001": _hand_end("A-1-001", False, cells=250)}
        cb = {"A-1-000": _hand_end("A-1-000", False, cells=100),
              "A-1-001": _hand_end("A-1-001", False, cells=200)}
        end_full = {"wall_seconds": 10000.0, "peak_frac": 0.2,
                    "peak_rss_mib": 5000.0, "orphans": []}
        header_full = {"streams": 6}
        ids_g = ["A-1-000", "A-1-001"]
        res_g = g_cost(ce, cb, ids_g, end_full, header_full, gates)
        assert abs(res_g["items"]["cells"]["ratio"] - 1.3333333333333333) < 1e-12
        assert res_g["verdict"] == PASS, res_g["items"]
        ce2 = dict(ce)
        ce2["A-1-002"] = _hand_end("A-1-002", False, cells=3000000)
        res_g2 = g_cost(ce2, cb, ids_g + ["A-1-002"], end_full, header_full, gates)
        assert res_g2["items"]["budget"]["verdict"] == FAIL
        assert res_g2["verdict"] == FAIL
        clean = {"full": {"configs": 4, "sha_bad": 0, "quality_bad": 0,
                          "edits": 3, "edits_bad": 0}}
        ok_static = {"remedies_scan_ok": True, "violations": [], "flag_literals": []}
        assert g_qual(clean, ok_static)["verdict"] == PASS
        bad_static = {"remedies_scan_ok": True, "violations": [],
                      "flag_literals": ["campaign.py:1"]}
        assert g_qual(clean, bad_static)["verdict"] == FAIL
        assert g_det({"ok": True}, {"ok": True}, {"ok": True}, {"ok": True})["verdict"] == PASS
        assert g_det({"ok": True}, {"ok": True}, {"ok": False}, {"ok": True})["verdict"] == FAIL
        assert g_expl({"full": {"ok": True, "n_rows": 4, "untemplated": 0}})["verdict"] == PASS
        assert g_expl({"full": {"ok": False, "n_rows": 4, "untemplated": 1}})["verdict"] == FAIL
        assert g_expl({"b0": None})["per_campaign"]["b0"]["ok"] is True

    def g4():
        assert feature_share(10, 4, 0.5, 1.0) == 0.4
        assert feature_share(0, 0, 0.0, 1.2) == 0.0
        assert feature_share(0, 0, 0.0, 0.0) is None
        assert feature_share(0, 0, 0.5, 1.2) is None
        assert feature_share(None, None, 0.0, 1.0) is None
        d = os.path.join(tmp, "d4")
        os.makedirs(os.path.join(d, "configs"))
        with open(os.path.join(d, "configs", "D-1-010_a2.json"), "w",
                  encoding="utf-8", newline="\n") as f:
            json.dump({"snap": {"feature_tolerance": 0.0}}, f)
            f.write("\n")
        with open(os.path.join(d, "configs", "F-1-009_a1.json"), "w",
                  encoding="utf-8", newline="\n") as f:
            json.dump({}, f)
            f.write("\n")
        os.makedirs(os.path.join(d, "cases", "D-1-010_a2"))
        with open(os.path.join(d, "cases", "D-1-010_a2",
                               "D-1-010_a2_summary.json"), "w",
                  encoding="utf-8", newline="\n") as f:
            json.dump({"stages": [{"stage": "snap", "n_feature_edges": 0,
                                   "n_snapped_to_edge": 0}]}, f)
            f.write("\n")
        bundle = {"geometries": [
            {"geometry_id": "D-1-010", "family": "D", "failure": False,
             "final_attempt": 2, "fingerprint": {"sharp_edge_length_m": 1.5}},
            {"geometry_id": "F-1-009", "family": "F", "failure": False,
             "final_attempt": 1, "fingerprint": {"sharp_edge_length_m": 0.0}},
            {"geometry_id": "D-1-077", "family": "D", "failure": True,
             "final_attempt": 1, "fingerprint": {"sharp_edge_length_m": 1.0}}],
            "attempts": [
            {"geometry_id": "D-1-010", "attempt": 1, "outcome":
             {"pinned_frac": 0.1, "p99_over_hf": 0.02}},
            {"geometry_id": "D-1-010", "attempt": 2, "outcome":
             {"pinned_frac": 0.0, "p99_over_hf": 0.002}},
            {"geometry_id": "F-1-009", "attempt": 1, "outcome":
             {"pinned_frac": 0.0, "p99_over_hf": 0.002}}]}
        fr = fid_rows(d, bundle)
        assert len(fr) == 2, fr
        r1 = fr[0]
        assert r1["geometry_id"] == "D-1-010" and r1["attempt"] == 2
        assert r1["feature_tolerance"] == 0.0 and r1["feature_share"] == 0.0
        assert r1["summary"] is True and r1["p99_over_hf"] == 0.002
        assert r1["sharp_edge_length_m"] == 1.5
        r2 = fr[1]
        assert r2["geometry_id"] == "F-1-009" and r2["feature_tolerance"] == 0.5
        assert r2["feature_share"] is None and r2["summary"] is False
        qcfgs = [{}, {"quality": {"max_non_orth_deg": 85.0}}, {},
                 {"quality": {"max_non_orth_deg": 70.0}}]
        for i, cfg in enumerate(qcfgs, 1):
            with open(os.path.join(d, "configs", "D-1-010_a%d.json" % i), "w",
                      encoding="utf-8", newline="\n") as f:
                json.dump(cfg, f)
                f.write("\n")
        qrows = [{"geometry_id": "D-1-010", "attempt": 1,
                  "config_sha": campaign.sha(qcfgs[0]), "config_delta": []},
                 {"geometry_id": "D-1-010", "attempt": 2,
                  "config_sha": campaign.sha(qcfgs[1]), "config_delta": []},
                 {"geometry_id": "D-1-010", "attempt": 3,
                  "config_sha": campaign.sha(qcfgs[2]),
                  "config_delta": [{"pointer": "/quality/max_non_orth_deg",
                                    "from": 70.0, "to": 85.0}]},
                 {"geometry_id": "D-1-010", "attempt": 4,
                  "config_sha": "0" * 64, "config_delta": []}]
        q = qual_of(d, qrows, knobs)
        assert q == {"configs": 4, "sha_bad": 1, "quality_bad": 1, "edits": 1,
                     "edits_bad": 1}, q
        try:
            qual_of(d, [{"geometry_id": "D-1-099", "attempt": 1,
                         "config_sha": "x", "config_delta": []}], knobs)
            raise AssertionError("a missing config was accepted")
        except EvalError as e:
            assert "missing" in str(e), str(e)
        assert flag_literals(knobs) == []

    def g5():
        c = types.SimpleNamespace(ablate=("preflight",), gates=gates)
        gctx = {"gid": "D-1-010", "areas": {"body": 2.0},
                "flow": rows["D-1-010"]["flow"]}
        cfgs = [{"layers": {"n": 8, "patches": ["body"]}, "k": i}
                for i in (1, 2, 3)]
        seen = []

        def stub_attempt(cc, gctx_, config, a, n_leaves):
            seen.append(n_leaves)
            t = schema._now_iso()
            return {"outcome": {"failure": False}, "content_sha256": "s",
                    "t_start": t, "t_end": t}

        probe_calls = []

        def stub_probe(cc, gctx_, config, a):
            probe_calls.append(1)
            return {"n_leaves": 5000000, "max_non_orth_deg": None, "exit_code": 0}

        counts5 = new_counts()
        cache5 = new_cache()
        cache5["measured"][("D-1-010", campaign.sha(cfgs[0]))] = {
            "outcome": {"failure": False, "patches":
                        [{"name": "body", "capability_limited": True}]},
            "content_sha256": "x"}
        A, P, S = eval_fns(cache5, counts5, reuse=True,
                           attempt_fn=stub_attempt, probe_fn=stub_probe,
                           snap_fn=campaign._fake_snap)
        res1 = A(c, gctx, cfgs[0], 1, 1000)
        assert counts5["reused"] == 1 and seen == [] and probe_calls == []
        assert res1["outcome"]["patches"][0]["capability_limited"] is False
        assert cache5["measured"][("D-1-010", campaign.sha(cfgs[0]))] \
            ["outcome"]["patches"][0]["capability_limited"] is True
        res2 = A(c, gctx, cfgs[1], 1, None)
        assert counts5["guard_probes"] == 1 and counts5["ram_guarded"] == 1
        assert counts5["meshed"] == 0 and seen == []
        assert res2["outcome"]["failure_class"] == "crash"
        assert res2["outcome"]["flags"]["F1"] is True
        assert res2["content_sha256"] is None
        c2 = types.SimpleNamespace(ablate=(), gates=gates)
        A(c2, gctx, cfgs[1], 1, 1000)
        assert counts5["meshed"] == 1 and seen == [1000]
        cache5["probes"][campaign.sha(cfgs[2])] = 42
        pv = P(c, gctx, cfgs[2], 1)
        assert pv == {"n_leaves": 42, "max_non_orth_deg": None, "exit_code": 0}
        assert probe_calls == [1]      # the one guard probe; the hit called none
        b5 = {"geometries": [], "jobs": [], "attempts": [
            {"geometry_id": "D-1-010", "config_sha": campaign.sha(cfgs[1]),
             "outcome": {"failure": False}, "content_sha256": "y"}]}
        update_cache(cache5, b5, counts5["guarded"])
        assert ("D-1-010", campaign.sha(cfgs[1])) not in cache5["measured"]
        assert ("D-1-010", campaign.sha(cfgs[1])) in cache5["guarded"]

    def g6():
        plan = plan_of({"source": "x", "sha256": "y", "n": 4}, True,
                       ["D-1-010"], 2, campaign.BINARY_DEFAULT,
                       {"b0-template": "a", "b0-lhs": "b"})
        l6 = os.path.join(tmp, "l6")
        lock = write_lock(l6, plan)
        assert lock["plan_sha256"] == campaign.sha(plan)
        assert lock["schema"] == LOCK_SCHEMA and lock["plan"] == plan
        again = write_lock(l6, plan)
        assert again == lock and again["opened_at"] == lock["opened_at"]
        other = dict(plan)
        other["streams"] = 1
        try:
            write_lock(l6, other)
            raise AssertionError("another plan was accepted")
        except EvalError as e:
            assert "spent" in str(e), str(e)

    def g7():
        rep_dir = os.path.join(tmp, "rep7")
        src_dir = os.path.join(tmp, "src")
        work7 = os.path.join(tmp, "w7")
        base_paths = {}
        for system in baseline.SYSTEMS:
            d = os.path.join(tmp, system)
            campaign.run_campaign({"manifest": "tuning", "mode": system,
                                   "ids": list(IDS), "out": d, "streams": 2,
                                   "quiet": True}, **F)
            base_paths[system] = os.path.join(src_dir, "%s.json.gz" % system)
            baseline.write_bundle(baseline.bundle_dir(d), base_paths[system])
        m7 = os.path.join(tmp, "m7.jsonl")
        with open(m7, "w", encoding="utf-8", newline="\n") as f:
            for gid in IDS:
                f.write(json.dumps(rows[gid], sort_keys=True) + "\n")
        rep = run(work7, manifest=m7, baselines=base_paths, gdet_n=2, streams=2,
                  report_dir=rep_dir, quiet=True, **F)
        assert rep["rehearsal"] is True
        assert set(rep["gates"]) == {"G-FAIL", "G-BLC-0", "G-QUAL", "G-FID",
                                     "G-COST", "G-DET", "G-EXPL", "G-OPT", "G-ABL"}
        for name in [c[0] for c in CAMPAIGNS]:
            d = os.path.join(work7, name)
            assert os.path.isfile(os.path.join(d, campaign.FILES["end"]))
            assert os.path.isfile(os.path.join(d, COUNTS_NAME))
        for fname in [LOCK_NAME, RUNS_NAME, REPORT_NAME, REPORT_MD] \
                + [BUNDLE_NAME % c[0] for c in CAMPAIGNS] \
                + [BASELINE_COPY % s for s in baseline.SYSTEMS]:
            assert os.path.isfile(os.path.join(rep_dir, fname)), fname
        assert os.path.isfile(os.path.join(rep_dir, CONTEXT_NAME))
        assert load_context(rep_dir) == rep["tuning"]
        runs7 = _read_json(os.path.join(rep_dir, RUNS_NAME))
        by_name = dict((c["name"], c) for c in runs7["campaigns"])
        assert by_name["full"]["counts"]["meshed"] > 0
        assert by_name["full"]["counts"]["reused"] == 0
        assert by_name["no-remedies"]["counts"]["meshed"] == 0
        nr = next(c for c in rep["campaigns"] if c["name"] == "no-remedies")
        assert by_name["no-remedies"]["counts"]["reused"] == nr["n_rows"]
        assert all(c["counts"]["ram_guarded"] == 0 for c in runs7["campaigns"])
        assert runs7["seal_compare"]["equal"] is True
        assert runs7["gdet"]["compare"]["ok"] is True
        assert all(c["replay"]["ok"] for c in runs7["campaigns"])
        assert rep["gates"]["G-QUAL"]["verdict"] == PASS
        assert rep["gates"]["G-DET"]["verdict"] == PASS
        # EXPL-FIX: the optimiser records written after the remedies' terminal
        # on a geometry's last attempt end the geometry (explain.ends_geometry),
        # so every campaign's audit is clean and G-EXPL passes
        assert rep["gates"]["G-EXPL"]["verdict"] == PASS, rep["gates"]["G-EXPL"]
        assert all(rep["gates"]["G-EXPL"]["per_campaign"][c["name"]]["ok"]
                   for c in runs7["campaigns"]), rep["gates"]["G-EXPL"]
        cd = os.path.join(work7, "full")
        rows_f = campaign.load_rows(cd)
        recs_f = campaign.load_records(cd)
        au = explain.audit(rows_f, recs_f)
        assert au["ok"] and au["record_order"]["bad"] == [], au["record_order"]
        last = max(r["attempt"] for r in rows_f if r["geometry_id"] == "F-1-009")
        tags = recs_f["F-1-009"]
        # GLB-CONFIG: R-WIN's margin puts attempt 1 at wall level 4 while the
        # floor stays the plain edge's 3, so RM-SNAP-WALL's coarsening to 3
        # lands and its attempt-2 mesh passes the preflight: the terminal
        # RM-PASS ends the geometry (no RM-SNAP-TAU run, no RM-EXHAUSTED)
        assert any(e and t["record"]["rule_id"] == "RM-PASS" for t, e in
                   zip(tags, explain.ends_geometry(tags, last))), tags
        assert rep["systems"]["b0-template"]["failures"] == 4
        assert rep["gates"]["G-FID"]["verdict"] == UNDECIDED
        assert rep["systems"]["full"]["n"] == 4
        assert rep["gdet_ids"] == ["D-1-010", "F-1-009"]
        assert rep["decisions"]["full"] and all(
            k in explain.TEMPLATES for k in rep["decisions"]["full"]), rep["decisions"]
        with open(os.path.join(rep_dir, REPORT_MD), encoding="utf-8") as f:
            md = f.read()
        for marker in ("REHEARSAL", "## Headlines", "## Ablation (G-ABL)",
                       "## Tuning context (not the result)", "## Departures",
                       "in the style of", DOI):
            assert marker in md, marker
        mtimes = {}
        for name in [c[0] for c in CAMPAIGNS]:
            mtimes[name] = os.path.getmtime(
                os.path.join(work7, name, campaign.FILES["campaign"]))
        rep2 = run(work7, manifest=m7, baselines=base_paths, gdet_n=2, streams=2,
                   report_dir=rep_dir, quiet=True, **F)
        for name, mt in mtimes.items():
            assert os.path.getmtime(
                os.path.join(work7, name, campaign.FILES["campaign"])) == mt, name
        a1, a2 = dict(rep), dict(rep2)
        a1.pop("date"), a2.pop("date")
        assert campaign.sha(a1) == campaign.sha(a2)
        other = dict(_read_json(os.path.join(rep_dir, LOCK_NAME))["plan"])
        other["streams"] = 1
        try:
            run(work7, manifest=m7, baselines=base_paths, gdet_n=2, streams=1,
                report_dir=rep_dir, quiet=True, **F)
            raise AssertionError("a different plan was accepted")
        except EvalError as e:
            assert "spent" in str(e), str(e)
        res7 = check(rep_dir)
        assert res7["verdict"] == PASS, [(i["name"], i["why"][:300])
                                         for i in res7["items"] if not i["ok"]]

    def g8():
        rep8a = os.path.join(tmp, "rep8a")
        shutil.copytree(os.path.join(tmp, "rep7"), rep8a)
        bpath = os.path.join(rep8a, BUNDLE_NAME % "full")
        with open(bpath, "rb") as f:
            data = bytearray(f.read())
        data[-3] ^= 0xFF
        with open(bpath, "wb") as f:
            f.write(bytes(data))
        res_a = check(rep8a)
        bundles_item = next(i for i in res_a["items"] if i["name"] == "bundles")
        assert bundles_item["ok"] is False and res_a["verdict"] == FAIL
        rep8b = os.path.join(tmp, "rep8b")
        shutil.copytree(os.path.join(tmp, "rep7"), rep8b)
        epath = os.path.join(rep8b, REPORT_NAME)
        rep_b = _read_json(epath)
        rep_b["gates"]["G-DET"]["verdict"] = FAIL
        _dump_json(epath, rep_b)
        res_b = check(rep8b)
        report_item = next(i for i in res_b["items"] if i["name"] == "report")
        assert report_item["ok"] is False
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        p1 = subprocess.run([sys.executable, os.path.abspath(__file__), "--plan"],
                            capture_output=True, text=True, encoding="utf-8",
                            errors="replace", env=env, timeout=300)
        assert p1.returncode == 0, (p1.stdout + p1.stderr)[-500:]
        assert "seal b0-template ok 7685d908ada5" in p1.stdout
        assert "not opened" in p1.stdout
        p2 = subprocess.run([sys.executable, os.path.abspath(__file__), "--run"],
                            capture_output=True, text=True, encoding="utf-8",
                            errors="replace", env=env, timeout=300)
        assert p2.returncode == 2 and "needs --work" in p2.stderr
        p3 = subprocess.run([sys.executable, os.path.abspath(__file__), "--check",
                             "--report-dir", os.path.join(tmp, "rep7")],
                            capture_output=True, text=True, encoding="utf-8",
                            errors="replace", env=env, timeout=300)
        assert p3.returncode == 0 and "CHECK PASS" in p3.stdout
        p4 = subprocess.run([sys.executable, os.path.abspath(__file__), "--check",
                             "--report-dir", os.path.join(tmp, "nothing")],
                            capture_output=True, text=True, encoding="utf-8",
                            errors="replace", env=env, timeout=300)
        assert p4.returncode == 1
    for name, fn in (("constants, the seal, the G-DET ids", g1),
                     ("statistics", g2),
                     ("gates", g3),
                     ("rows", g4),
                     ("seam", g5),
                     ("lock", g6),
                     ("run", g7),
                     ("check and CLI", g8)):
        if not group(name, fn):
            return 1
    shutil.rmtree(tmp, ignore_errors=True)
    print("SELFTEST PASS (%.1f s)" % (time.perf_counter() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
