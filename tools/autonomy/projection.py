#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""
projection.py - the pre-registered go / no-go for a fresh seed (docs/17 §12
row AUT-12, docs/15 §F, D-9).

From a tuning re-measure (a `rules` or `full` campaign bundle on the tuning
split) and the B0-template bundle on the same ids, it projects every
headline and guard of docs/15 §F as the held-out test bar would read them,
and says GO only when every projected verdict is PASS.  A fresh seed is
spent only on a GO (AUT-13): D-9's recommendation is not to spend one on a
no-go.  Nothing here changes a rule, a gate or a constant; every threshold
is read from evaluate.py.

PJ-1 k*(N) = the largest k in [0, N] whose two-sided Clopper-Pearson upper
bound clopper_pearson(k, N)[1] <= evaluate.MFR_UPPER_MAX; N = evaluate.TEST_N
(180), so k*(180) = 10.  PJ-2 the tuning bar bar(n) = floor(n k*(N) / N) in
integers (bar(420) = 23): G-FAIL's 95 % upper-bound condition projects as
held iff the tuning failures are at most bar(n).  PJ-3 the ratio condition
4 F_fail <= B_fail is scale-free.  PJ-4 McNemar: b = #(B0 fails, system
passes), c = #(B0 passes, system fails) on the tuning ids; each is scaled to
N by round_half_up(x N / n) and evaluate.mcnemar(b', c') < 0.01 is the
condition.  PJ-5 no family has more system failures than B0 failures.
PJ-6 G-BLC-0, G-BLC-1, G-FID and G-COST are evaluate.py's own functions on
the tuning ends: tier 0 is D and F with the manifest's commensurate flag,
tier 1 is baseline.TIER1; G-FID reads p99_over_hf, pinned_frac and
feature_capture from the bundle rows (feature_share needs the cases'
summaries, which a bundle does not carry, so it is not projected); G-COST's
campaign wall time is scaled to N (wall x N / n).  PJ-7 G-QUAL projects the
rows' edits against the knob whitelist (the quality block needs the configs,
which a bundle does not carry); G-EXPL is the bundle's summary audit.
G-DET (two runs) and G-OPT (the test ablation) are NOT-PROJECTED.  PJ-8 GO
iff all of G-FAIL, G-BLC-0, G-BLC-1, G-QUAL, G-FID, G-COST and G-EXPL
project PASS; otherwise NO-GO with every reason, G-FAIL's first.

    python tools/autonomy/projection.py --selftest
    python tools/autonomy/projection.py --project --full BUNDLE --b0 BUNDLE --name NAME
                                        [--out-dir DIR] [--note TEXT]
    python tools/autonomy/projection.py --check [--out-dir DIR]
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "corpus")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import baseline
import evaluate
import explain
import schema
import split

OUT_DIR = os.path.join(HERE, "projection")
SCHEMA = "autonomy-projection/1"
HEADER = baseline.HEADER
TEST_N = evaluate.TEST_N
GO, NO_GO = "GO", "NO-GO"
PASS, FAIL = evaluate.PASS, evaluate.FAIL
NOT_PROJECTED = "NOT-PROJECTED"
DECIDING = ("G-FAIL", "G-BLC-0", "G-BLC-1", "G-QUAL", "G-FID", "G-COST", "G-EXPL")
FID_METRICS_BUNDLE = (("p99_over_hf", "le"), ("pinned_frac", "le"),
                      ("feature_capture", "ge"))
L5_FULL = os.path.join(HERE, "aml", "tuning_rules_L5.json.gz")
L5_B0 = os.path.join(HERE, "baseline", "tuning_b0-template.json.gz")


class ProjectionError(RuntimeError):
    pass


def k_star(n_test=TEST_N, upper_max=evaluate.MFR_UPPER_MAX):
    """PJ-1: the most failures of n_test whose CP upper bound is <= upper_max."""
    best = -1
    for k in range(n_test + 1):
        if explain.clopper_pearson(k, n_test)[1] <= upper_max:
            best = k
        else:
            break
    return best


def fail_bar(n, n_test=TEST_N, k=None):
    """PJ-2: floor(n k*(N) / N), integers only."""
    k = k_star(n_test) if k is None else k
    return (n * k) // n_test if k >= 0 else -1


def round_half_up(num, den):
    """round(num / den) with halves up, for non-negative integers."""
    return (2 * num + den) // (2 * den)


def _ends(bundle):
    return dict((g["geometry_id"], g) for g in bundle["geometries"])


def _row_of(bundle, gid, k):
    return next((r for r in bundle["attempts"]
                 if r["geometry_id"] == gid and r["attempt"] == k), None)


def fid_rows_bundle(bundle):
    """G-FID's rows from a bundle alone: the final attempt of each passing geometry."""
    rows = []
    for g in bundle["geometries"]:
        if g["failure"] is not False or not g.get("final_attempt"):
            continue
        row = _row_of(bundle, g["geometry_id"], g["final_attempt"])
        if row is None:
            continue
        o = row["outcome"]
        rows.append({"geometry_id": g["geometry_id"], "family": g["family"],
                     "attempt": g["final_attempt"], "p99_over_hf": o.get("p99_over_hf"),
                     "pinned_frac": o.get("pinned_frac"),
                     "feature_capture": o.get("feature_capture")})
    return rows


def qual_rows_bundle(bundle, knobs):
    edits = bad = 0
    for row in bundle["attempts"]:
        delta = row.get("config_delta") or []
        edits += len(delta)
        bad += sum(1 for e in delta
                   if schema.check_edit(e["pointer"], e["to"], knobs) is not None)
    return {"verdict": PASS if bad == 0 else FAIL, "rows": len(bundle["attempts"]),
            "edits": edits, "edits_bad": bad,
            "not_projected": "the configs' quality block (a bundle carries no configs)"}


def g_fail_projected(full, b0, pairs, n, n_test=TEST_N):
    """PJ-2..PJ-5 on the tuning stats (evaluate.system_stats) and (B0, system) pairs."""
    k = k_star(n_test)
    bar = fail_bar(n, n_test, k)
    b = sum(1 for p, f in pairs if p is True and f is False)
    c = sum(1 for p, f in pairs if p is False and f is True)
    bs, cs = round_half_up(b * n_test, n), round_half_up(c * n_test, n)
    p = evaluate.mcnemar(bs, cs)
    worse = evaluate.family_worse(full, b0)
    conditions = {"ratio_le_quarter": 4 * full["failures"] <= b0["failures"],
                  "upper_le_10pct": full["failures"] <= bar,
                  "mcnemar_p_lt_001": p < evaluate.MCNEMAR_P_MAX,
                  "no_family_worse": not worse}
    return {"verdict": PASS if all(conditions.values()) else FAIL,
            "conditions": conditions, "n": n, "n_test": n_test,
            "failures": full["failures"], "failures_b0": b0["failures"],
            "k_star": k, "bar": bar, "ratio_bar": b0["failures"] // 4,
            "projected_test_failures": full["failures"] * n_test / n,
            "mfr": full["mfr"], "mfr_b0": b0["mfr"],
            "mcnemar": {"b": b, "c": c, "b_test": bs, "c_test": cs, "p": p},
            "worse_families": worse}


def project(full_b, b0_b, manifest_rows, gates, knobs, n_test=TEST_N):
    """PJ-1..PJ-8: the projected report (a dict) of a tuning bundle against B0."""
    ends_f, ends_b = _ends(full_b), _ends(b0_b)
    ids = sorted(ends_f)
    missing = [i for i in ids if i not in ends_b]
    if missing:
        raise ProjectionError("B0 lacks %d of the system's ids (first %s)"
                              % (len(missing), missing[0]))
    meta = dict((r["geometry_id"], r) for r in manifest_rows)
    lost = [i for i in ids if i not in meta]
    if lost:
        raise ProjectionError("the manifest lacks %s" % lost[0])
    n = len(ids)
    sf, sb = evaluate.system_stats(ends_f, ids), evaluate.system_stats(ends_b, ids)
    pairs = [(ends_b[i]["failure"] is True, ends_f[i]["failure"] is True) for i in ids]
    tier0 = [i for i in ids if meta[i]["family"] in evaluate.TIER0_FAMILIES
             and meta[i].get("commensurate") is True]
    t0 = set(tier0)
    tier1 = [i for i in ids if ends_f[i].get("family") in baseline.TIER1]
    end = dict(full_b["end"])
    end["wall_seconds"] = full_b["end"]["wall_seconds"] * n_test / n
    gates_rep = {
        "G-FAIL": g_fail_projected(sf, sb, pairs, n, n_test),
        "G-BLC-0": evaluate.g_blc0(ends_f, ends_b, tier0, [i for i in ids if i not in t0]),
        "G-BLC-1": evaluate.g_blc1(ends_f, ends_b, tier1),
        "G-QUAL": qual_rows_bundle(full_b, knobs),
        "G-FID": evaluate.g_fid(fid_rows_bundle(full_b), fid_rows_bundle(b0_b), True,
                                metrics=FID_METRICS_BUNDLE),
        "G-COST": evaluate.g_cost(ends_f, ends_b, ids, end, full_b["campaign"], gates),
        "G-EXPL": evaluate.g_expl({"system": full_b["summary"]["audit"]}),
        "G-DET": {"verdict": NOT_PROJECTED, "why": "needs two runs of the same ids"},
        "G-OPT": {"verdict": NOT_PROJECTED, "why": "decided on the test ablation"}}
    gates_rep["G-COST"]["wall_scaled_from_s"] = full_b["end"]["wall_seconds"]
    reasons = [reason_of(g, gates_rep[g]) for g in DECIDING
               if gates_rep[g]["verdict"] != PASS]
    return {"schema": SCHEMA, "verdict": GO if not reasons else NO_GO,
            "reasons": reasons, "n": n, "n_test": n_test,
            "system": {"campaign_id": full_b["campaign"]["campaign_id"],
                       "mode": full_b["campaign"].get("mode"),
                       "binary_sha256": full_b["campaign"].get("binary_sha256")},
            "b0": {"campaign_id": b0_b["campaign"]["campaign_id"],
                   "binary_sha256": b0_b["campaign"].get("binary_sha256")},
            "systems": {"system": sf, "b0-template": sb},
            "tier0_ids": tier0, "gates": gates_rep}


def _f(x):
    return explain.fmt(float(x)) if x is not None else "-"


def reason_of(name, rep):
    """One line per failed gate, the numbers first."""
    v = rep["verdict"]
    if name == "G-FAIL":
        bad = [k for k, ok in rep["conditions"].items() if not ok]
        parts = []
        if "upper_le_10pct" in bad:
            parts.append("%d/%d > %d (the 95 %% upper bound on %d test bodies holds at <= %d failures)"
                         % (rep["failures"], rep["n"], rep["bar"], rep["n_test"], rep["k_star"]))
        if "ratio_le_quarter" in bad:
            parts.append("4 x %d > B0-template %d" % (rep["failures"], rep["failures_b0"]))
        if "mcnemar_p_lt_001" in bad:
            m = rep["mcnemar"]
            parts.append("McNemar b %d c %d on %d bodies, p %s >= 0.01"
                         % (m["b_test"], m["c_test"], rep["n_test"], _f(m["p"])))
        if "no_family_worse" in bad:
            parts.append("families worse than B0-template: %s" % ", ".join(rep["worse_families"]))
        return "G-FAIL %s: %s" % (v, "; ".join(parts))
    if name in ("G-BLC-0", "G-BLC-1"):
        return "%s %s: n %d, BLC_8 %s, BLC_full %s" % (name, v, rep["n"], _f(rep["blc8_mean"]),
                                                       _f(rep["blc_full_mean"]))
    if name == "G-FID":
        return "G-FID %s: worse %s" % (v, ", ".join("%s %s" % tuple(w) for w in rep["worse"])
                                      or "(nothing comparable)")
    if name == "G-COST":
        bad = ["%s %s" % (k, it["verdict"]) for k, it in rep["items"].items()
               if it["verdict"] != PASS]
        c = rep["items"]["cells"]
        return "G-COST %s: %s; cells median %s vs B0 %s (ratio %s > 1.5)" % (
            v, ", ".join(bad), _f(c["median_full"]), _f(c["median_b0"]), _f(c["ratio"]))
    if name == "G-QUAL":
        return "G-QUAL %s: %d of %d edits off the whitelist" % (v, rep["edits_bad"], rep["edits"])
    return "%s %s" % (name, v)


def report_md(rep, note=None):
    g = rep["gates"]
    lines = ["# The go / no-go projection (AUT-12): %s" % rep["verdict"], "",
             "System `%s` (%s, binary `%s`) against B0-template `%s` on %d tuning geometries, "
             "projected to a %d-body test split (projection.py, PJ-1..PJ-8)."
             % (rep["system"]["campaign_id"], rep["system"]["mode"],
                str(rep["system"]["binary_sha256"])[:12], rep["b0"]["campaign_id"],
                rep["n"], rep["n_test"]), ""]
    if note:
        lines += [note, ""]
    lines += ["| gate | projected | numbers |", "|---|---|---|"]
    f = g["G-FAIL"]
    lines.append("| G-FAIL | %s | failures %d/%d (bar %d = floor(%d x %d / %d)); B0-template %d; "
                 "McNemar b %d c %d -> %d/%d, p %s; families worse: %s |"
                 % (f["verdict"], f["failures"], f["n"], f["bar"], f["n"], f["k_star"],
                    f["n_test"], f["failures_b0"], f["mcnemar"]["b"], f["mcnemar"]["c"],
                    f["mcnemar"]["b_test"], f["mcnemar"]["c_test"], _f(f["mcnemar"]["p"]),
                    ", ".join(f["worse_families"]) or "none"))
    for k in ("G-BLC-0", "G-BLC-1"):
        r = g[k]
        lines.append("| %s | %s | n %d, BLC_8 %s, BLC_full %s (B0 %s, %s) |"
                     % (k, r["verdict"], r["n"], _f(r["blc8_mean"]), _f(r["blc_full_mean"]),
                        _f(r["b0"]["blc8_mean"]), _f(r["b0"]["blc_full_mean"])))
    q = g["G-QUAL"]
    lines.append("| G-QUAL | %s | %d edits, %d off the whitelist (not projected: %s) |"
                 % (q["verdict"], q["edits"], q["edits_bad"], q["not_projected"]))
    lines.append("| G-FID | %s | compared %d, worse %s |"
                 % (g["G-FID"]["verdict"], g["G-FID"]["compared"],
                    ", ".join("%s %s" % tuple(w) for w in g["G-FID"]["worse"]) or "none"))
    c = g["G-COST"]["items"]
    lines.append("| G-COST | %s | cells %s (median %s vs %s, ratio %s on %d both-pass); budget %s "
                 "(%d over); wall %s (%s s scaled from %s s); RAM %s; orphans %s |"
                 % (g["G-COST"]["verdict"], c["cells"]["verdict"], _f(c["cells"]["median_full"]),
                    _f(c["cells"]["median_b0"]), _f(c["cells"]["ratio"]), c["cells"]["n_both"],
                    c["budget"]["verdict"], c["budget"]["n_over"], c["wall"]["verdict"],
                    _f(c["wall"]["seconds"]), _f(g["G-COST"]["wall_scaled_from_s"]),
                    c["ram"]["verdict"], c["orphans"]["verdict"]))
    lines.append("| G-EXPL | %s | the bundle's summary audit |" % g["G-EXPL"]["verdict"])
    for k in ("G-DET", "G-OPT"):
        lines.append("| %s | %s | %s |" % (k, g[k]["verdict"], g[k]["why"]))
    lines += ["", "**Per family** (failures system / B0-template, BLC_8 system):", ""]
    sf, sb = rep["systems"]["system"]["per_family"], rep["systems"]["b0-template"]["per_family"]
    for fam in sorted(sf):
        lines.append("- %s: %d / %d of %d, BLC_8 %s" % (fam, sf[fam]["failures"],
                                                       sb[fam]["failures"], sf[fam]["n"],
                                                       _f(sf[fam]["blc8_mean"])))
    lines += ["", "**Verdict: %s.**" % rep["verdict"]]
    lines += ["- %s" % r for r in rep["reasons"]]
    return "\n".join(lines) + "\n"


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _rel(path):
    return os.path.relpath(os.path.abspath(path), evaluate.REPO).replace(os.sep, "/")


def run_project(full_path, b0_path, name, out_dir=OUT_DIR, note=None, write=True):
    full_b, b0_b = baseline.read_bundle(full_path), baseline.read_bundle(b0_path)
    rep = project(full_b, b0_b, split.load("tuning", "projection"),
                  schema.load_gates(), schema.load_knobs())
    rep["inputs"] = {"full": {"file": _rel(full_path), "sha256": _sha(full_path)},
                     "b0": {"file": _rel(b0_path), "sha256": _sha(b0_path)}}
    rep["note"] = note
    if write:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, name + ".json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(dict({"$comment": HEADER}, **rep), f, indent=1, sort_keys=True,
                      ensure_ascii=False)
            f.write("\n")
        with open(os.path.join(out_dir, name + ".md"), "w", encoding="utf-8", newline="\n") as f:
            f.write(report_md(rep, note))
    return rep


def check(out_dir=OUT_DIR):
    """Every committed report rebuilds byte-equal from its recorded inputs."""
    names = sorted(f[:-5] for f in os.listdir(out_dir) if f.endswith(".json"))
    if not names:
        raise ProjectionError("check: no report in %s" % out_dir)
    for nm in names:
        with open(os.path.join(out_dir, nm + ".json"), "r", encoding="utf-8") as f:
            old = json.load(f)
        if old.get("schema") != SCHEMA:
            continue
        paths = {}
        for k in ("full", "b0"):
            p = os.path.join(evaluate.REPO, old["inputs"][k]["file"])
            if _sha(p) != old["inputs"][k]["sha256"]:
                raise ProjectionError("check: %s input %s drifted" % (nm, k))
            paths[k] = p
        with tempfile.TemporaryDirectory() as td:
            run_project(paths["full"], paths["b0"], nm, out_dir=td, note=old.get("note"))
            for ext in (".json", ".md"):
                if _sha(os.path.join(td, nm + ext)) != _sha(os.path.join(out_dir, nm + ext)):
                    raise ProjectionError("check: %s%s does not rebuild" % (nm, ext))
        print("projection check: %s %s rebuilds (%s)" % (nm, old["verdict"],
                                                         old["reasons"][0] if old["reasons"]
                                                         else "every gate PASS"))
    return 0


# --- selftest -----------------------------------------------------------------


def _hand_bundle(ends, cid="hand", wall=100.0):
    geoms, atts = [], []
    for gid, fam, fail, blc8, cells in ends:
        geoms.append({"geometry_id": gid, "family": fam, "failure": fail,
                      "strict_failure": fail, "blc8_a_priori": blc8,
                      "blc_full_a_priori": blc8, "n_cells": cells, "seconds": 1.0,
                      "capability_limited": [], "final_attempt": 1, "terminal": "PASS"
                      if not fail else "EXHAUSTED", "attempts": 1})
        atts.append({"geometry_id": gid, "attempt": 1, "config_delta": [],
                     "outcome": {"p99_over_hf": 0.01, "pinned_frac": 0.0,
                                 "feature_capture": 1.0}})
    return {"geometries": geoms, "attempts": atts,
            "campaign": {"campaign_id": cid, "mode": "rules", "streams": 6,
                         "binary_sha256": "0" * 64},
            "end": {"wall_seconds": wall, "peak_frac": 0.1, "peak_rss_mib": 10.0,
                    "orphans": []},
            "summary": {"audit": {"ok": True, "n_rows": len(atts), "untemplated": []}}}


def _ok(msg):
    print("[ok] %s" % msg)


def _selftest():
    import io
    import contextlib
    # 1. PJ-1, PJ-2
    assert k_star(180) == 10
    assert explain.clopper_pearson(10, 180)[1] <= 0.10 < explain.clopper_pearson(11, 180)[1]
    assert fail_bar(420) == 23 and fail_bar(180) == 10 and fail_bar(18) == 1
    _ok("PJ-1/PJ-2: k*(180) = 10, bar(420) = 23, bar(180) = 10")
    # 2. rounding and McNemar scaling
    assert [round_half_up(x, 2) for x in (0, 1, 2, 3)] == [0, 1, 1, 2]
    assert round_half_up(7 * 180, 420) == 3 and round_half_up(1 * 180, 420) == 0
    sf = {"failures": 0, "mfr": 0.0, "per_family": {"A": {"failures": 0}}}
    sb = {"failures": 40, "mfr": 1.0, "per_family": {"A": {"failures": 40}}}
    r = g_fail_projected(sf, sb, [(True, False)] * 40, 40, 180)
    assert r["mcnemar"]["b_test"] == 180 and r["mcnemar"]["c_test"] == 0
    assert r["verdict"] == PASS and r["bar"] == 2
    _ok("PJ-3/PJ-4: half-up scaling, McNemar on the scaled pairs")
    # 3. hand ends: GO, then NO-GO naming G-FAIL first
    # 18 tier-0 D rows and 18 tier-1 A rows; B0 passes rows 0-2 of each family
    ids = [("%s-%02d" % (fam, i), fam) for fam in ("A", "D") for i in range(18)]
    rows = [{"geometry_id": g, "family": fam, "commensurate": fam == "D"} for g, fam in ids]
    gates, knobs = schema.load_gates(), schema.load_knobs()
    full = _hand_bundle([(g, fam, False, 1.0, 100) for g, fam in ids])
    b0 = _hand_bundle([(g, fam, int(g[2:]) > 2, 0.0, 100) for g, fam in ids], cid="b0")
    rep = project(full, b0, rows, gates, knobs)
    assert rep["verdict"] == GO, rep["reasons"]
    assert rep["gates"]["G-FAIL"]["bar"] == 2 and rep["gates"]["G-COST"]["verdict"] == PASS
    for g in full["geometries"][10:13]:          # A-10..A-12: 3 failures > bar(36) = 2
        g["failure"] = g["strict_failure"] = True
    rep = project(full, b0, rows, gates, knobs)
    assert rep["verdict"] == NO_GO and rep["reasons"][0].startswith("G-FAIL FAIL: 3/36 > 2 "), \
        rep["reasons"]
    _ok("PJ-8: an all-pass hand case is GO; 3 failures over bar(36) = 2 is NO-GO, G-FAIL first")
    # 4. the L5 bundle: no-go with its numbers
    rep5 = run_project(L5_FULL, L5_B0, "L5", write=False)
    f = rep5["gates"]["G-FAIL"]
    assert rep5["verdict"] == NO_GO
    assert (f["failures"], f["n"], f["bar"], f["k_star"]) == (177, 420, 23, 10)
    assert rep5["reasons"][0].startswith("G-FAIL FAIL: 177/420 > 23 "), rep5["reasons"][0]
    assert f["conditions"]["upper_le_10pct"] is False
    _ok("L5: NO-GO, %s" % rep5["reasons"][0])
    # 5. determinism and the CLI
    with tempfile.TemporaryDirectory() as td:
        run_project(L5_FULL, L5_B0, "a", out_dir=td)
        run_project(L5_FULL, L5_B0, "b", out_dir=td)
        assert _sha(os.path.join(td, "a.md")) == _sha(os.path.join(td, "b.md"))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert main(["--project", "--full", L5_FULL, "--b0", L5_B0, "--name", "c",
                         "--out-dir", td]) == 0
            assert main(["--check", "--out-dir", td]) == 0
        assert "NO-GO" in buf.getvalue()
        with open(os.path.join(td, "a.json"), encoding="utf-8") as fa, \
                open(os.path.join(td, "c.json"), encoding="utf-8") as fc:
            assert json.load(fa)["gates"] == json.load(fc)["gates"]
    _ok("determinism, --project and --check")
    print("SELFTEST PASS")
    return 0


class _ArgParser(argparse.ArgumentParser):
    def error(self, message):
        self.exit(2, "projection.py: %s\n" % message)


def main(argv=None):
    ap = _ArgParser(prog="projection.py",
                    description="the pre-registered go / no-go for a fresh seed (AUT-12)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--project", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--full")
    ap.add_argument("--b0")
    ap.add_argument("--name")
    ap.add_argument("--note")
    ap.add_argument("--out-dir", dest="out_dir", default=OUT_DIR)
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    if a.check:
        return check(a.out_dir)
    if a.project:
        if not (a.full and a.b0 and a.name):
            ap.error("--project needs --full, --b0 and --name")
        rep = run_project(a.full, a.b0, a.name, out_dir=a.out_dir, note=a.note)
        print("projection %s: %s" % (a.name, rep["verdict"]))
        for r in rep["reasons"]:
            print("  " + r)
        return 0
    ap.error("one of --selftest, --project, --check")


if __name__ == "__main__":
    sys.exit(main())
