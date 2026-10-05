#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""
rescore.py - the tuning campaigns re-scored under the feature-edge rule of
2026-09-26.

`rescore.py --run` re-scores the committed TUNING bundles under the feature-edge rule of section D, with no re-meshing: every attempt's config is rebuilt from its row (`optimise.rows_from`, sha-checked), F3e is decided by `score.feature_capture` from the fingerprint, `preflight.plane_path` and, where a campaign directory is given with `--cases`, the attempt's own `stages[snap]` report, and a geometry passes when one of its attempts that ran still passes.

It is the MFR of the attempts actually run: a geometry whose passing mesh now fails ended there, so the attempts a re-run would have made instead never ran, and a re-run can only rescue.

The seal holds (a bundle that is not a tuning campaign is refused); the test split is spent and is not read.

    python tools/autonomy/rescore.py --run [--cases SOURCE=DIR ...] [--out DIR]
    python tools/autonomy/rescore.py --selftest
"""
import argparse
import collections
import copy
import hashlib
import io
import json
import os
import sys

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
import schema
import score
import split

REPORT_DIR = os.path.join(HERE, "rescore")
REPORT_NAME = "FEAT-CONSTRAINT.json"
REPORT_MD = "FEAT-CONSTRAINT.md"
SCHEMA = "autonomy-rescore/1"
HEADER = baseline.HEADER
ORDER = ("B0-template", "B0-LHS", "rules", "rules+opt")
SOURCES = {"B0-template": ("baseline/tuning_b0-template.json.gz",),
           "B0-LHS": ("baseline/tuning_b0-lhs.json.gz",),
           "rules": ("prior/tuning_rules.json.gz",),
           "rules+opt": ("prior/tuning_rules.json.gz", "optimise/refine_r5.json.gz")}
CAPTURE_KINDS = ("smooth", "plane", "zero", "unmeasured", "no_summary")


DEPARTURES = (
    "`rescore.py --run` re-scores the committed TUNING bundles under the feature-edge rule of section D, with no re-meshing: every attempt's config is rebuilt from its row (`optimise.rows_from`, sha-checked), F3e is decided by `score.feature_capture` from the fingerprint, `preflight.plane_path` and, where a campaign directory is given with `--cases`, the attempt's own `stages[snap]` report, and a geometry passes when one of its attempts that ran still passes.",
    'It is the MFR of the attempts actually run: a geometry whose passing mesh now fails ended there, so the attempts a re-run would have made instead never ran, and a re-run can only rescue.',
    'The seal holds (a bundle that is not a tuning campaign is refused); the test split is spent and is not read.',
)


class RescoreError(ValueError):
    """A bundle that does not say what the re-score reads; never guessed past."""


def classify(train_row, cases_dir):
    """Which of CAPTURE_KINDS the rule reads this PASSING attempt as."""
    fp = train_row["fingerprint"]
    cfg = train_row["config"]
    ft = (cfg.get("snap") or {}).get(
        "feature_tolerance", preflight.MESHER_DEFAULTS["/snap/feature_tolerance"])
    snap = None
    if cases_dir is not None:
        gid, k = train_row["geometry_id"], train_row["attempt"]
        p = os.path.join(cases_dir, "cases", "%s_a%d" % (gid, k),
                         "%s_a%d_summary.json" % (gid, k))
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                summ = json.load(f)
            snap = next((st for st in summ.get("stages") or []
                         if st.get("stage") == "snap"), None)
    gid = train_row["geometry_id"]
    try:
        fc, f3e, note = score.feature_capture(
            fp["sharp_edge_length_m"], preflight.plane_path(cfg, fp), ft, snap)
    except score.ScoreParseError as e:
        raise RescoreError("classify: %s attempt %d: %s"
                           % (gid, train_row["attempt"], e)) from e
    if f3e is False and fc is None:
        return "smooth"
    if fc == 1.0:
        return "plane"
    if f3e is True:
        return "zero"
    if note == score.M_FCAP:
        return "unmeasured"
    if note == score.M_FCAP_NOSNAP:
        return "no_summary"
    raise RescoreError("classify: %s attempt %d reads (%r, %r, %r)"
                       % (gid, train_row["attempt"], fc, f3e, note))


def _cut(mrows, gids):
    rows = mrows.values() if isinstance(mrows, dict) else mrows
    return {r["geometry_id"]: r for r in rows if r["geometry_id"] in gids}


def rescore_system(named, mrows, gates, knobs, cases):
    """One system's failure counts before and after the rule, from its bundles.

    `named` is [(source, bundle)] in SOURCES order; a later source replaces a
    geometry wholly.  `cases` maps a source path to its campaign directory (or
    is None); there the attempt's own snap report is read, never guessed.
    """
    merged = {}
    for source, bundle in named:
        rows = optimise.rows_from(bundle, source, mrows, gates, knobs)
        ends = {g["geometry_id"]: g for g in bundle["geometries"]}
        train = collections.OrderedDict()
        for r in rows:
            train.setdefault(r["geometry_id"], {})[r["attempt"]] = r
        raw = collections.OrderedDict()
        for r in bundle["attempts"]:
            raw.setdefault(r["geometry_id"], {})[r["attempt"]] = r
        for gid, g in ends.items():
            merged[gid] = {"end": g, "train": train.get(gid, {}),
                           "raw": raw.get(gid, {})}
    cdir = None
    kinds = {}
    cases_read = 0
    for gid, e in merged.items():
        e["before"] = e["end"]["failure"] is True
        ran = [r for r in e["raw"].values() if r["outcome"]["failure"] is False]
        if e["train"]:
            if e["before"] != (not ran):
                raise RescoreError("identity: %s says failure %r but %d of its "
                                   "%d attempts pass" % (gid, e["end"]["failure"],
                                                         len(ran), len(e["raw"])))
        e["after"] = e["before"]
        e["after_strict"] = e["before"]
        newpass = False
        newpass_strict = False
        for a, r in e["train"].items():
            if e["raw"][a]["outcome"]["failure"] is not False:
                continue
            cdir = (cases or {}).get(r["source"])
            if cdir is not None:
                p = os.path.join(cdir, "cases", "%s_a%d" % (gid, a),
                                 "%s_a%d_summary.json" % (gid, a))
                if os.path.isfile(p):
                    cases_read += 1
            kind = classify(r, cdir)
            kinds[kind] = kinds.get(kind, 0) + 1
            if kind != "zero":
                newpass = True
            if kind not in ("zero", "plane"):
                newpass_strict = True
        e["after"] = e["before"] or not newpass
        e["after_strict"] = e["before"] or not newpass_strict
    n = len(mrows)
    fb = sum(1 for e in merged.values() if e["before"])
    fa = sum(1 for e in merged.values() if e["after"])
    fs = sum(1 for e in merged.values() if e["after_strict"])
    flipped = sum(1 for e in merged.values() if e["after"] and not e["before"])
    flipped_strict = sum(1 for e in merged.values()
                         if e["after_strict"] and not e["before"])
    gids = set(merged)
    fams = sorted(set(r["family"] for r in mrows.values()))
    per_family = {}
    for fam in fams:
        ids = set(r["geometry_id"] for r in mrows.values()
                  if r["family"] == fam)
        per_family[fam] = {
            "n": len(ids),
            "failures_before": sum(1 for g in ids if g in gids
                                   and merged[g]["before"]),
            "failures_after": sum(1 for g in ids if g in gids
                                  and merged[g]["after"]),
            "failures_after_strict": sum(1 for g in ids if g in gids
                                         and merged[g]["after_strict"])}
    return {"n": n, "failures_before": fb, "failures_after": fa,
            "failures_after_strict": fs,
            "mfr_before": fb / n if n else 0.0,
            "mfr_after": fa / n if n else 0.0,
            "mfr_after_strict": fs / n if n else 0.0,
            "mfr_ci_before": explain.clopper_pearson(fb, n) if n else (0.0, 0.0),
            "mfr_ci_after": explain.clopper_pearson(fa, n) if n else (0.0, 0.0),
            "flipped": fa - fb, "flipped_strict": fs - fb,
            "pass_rows": {k: kinds.get(k, 0) for k in CAPTURE_KINDS},
            "per_family": per_family, "cases_read": cases_read}


def _file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cases=None, report_dir=REPORT_DIR):
    """Re-score every system of ORDER, write both reports, print the lines."""
    gates = schema.load_gates()
    knobs = schema.load_knobs()
    mrows = {r["geometry_id"]: r
             for r in campaign.load_manifest("tuning", "rules")}
    bundles = collections.OrderedDict()
    for name in ORDER:
        for p in SOURCES[name]:
            if p not in bundles:
                bundles[p] = baseline.read_bundle(os.path.join(HERE, p))
    inputs = [{"source": p, "sha256": _file_sha(os.path.join(HERE, p))}
              for p in bundles]
    systems = collections.OrderedDict()
    for name in ORDER:
        named = [(p, bundles[p]) for p in SOURCES[name]]
        scases = None if not cases else \
            {p: cases[p] for p in SOURCES[name] if p in cases}
        systems[name] = rescore_system(named, mrows, gates, knobs, scases)
    report = {"$comment": HEADER, "schema": SCHEMA,
              "rule": "README section D (2026-09-26)", "inputs": inputs,
              "departures": list(DEPARTURES), "systems": systems}
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(report_dir, REPORT_NAME), "w", encoding="utf-8",
              newline="\n") as f:
        f.write(json.dumps(report, indent=1, sort_keys=True,
                           ensure_ascii=False) + "\n")
    _write_md(report, report_dir)
    for name in ORDER:
        s = systems[name]
        print("[rescore] %s: %d/%d -> %d/%d (plane path forbidden too: %d/%d)" %
              (name, s["failures_before"], s["n"], s["failures_after"], s["n"],
               s["failures_after_strict"], s["n"]))
    return report


def _write_md(report, report_dir):
    s4 = report["systems"]
    fams = sorted(next(iter(s4.values()))["per_family"])
    L = ["# rescore.py - the tuning campaigns under the 2026-09-26 rule", ""]
    for d in report["departures"]:
        L += [d, ""]
    L += ["| system (tuning, %d geometries) | failures before | after | "
          "MFR before -> after | after, plane path forbidden too |" % s4["rules"]["n"],
          "|---|---|---|---|---|"]
    for name in ORDER:
        s = s4[name]
        L.append("| %s | %d | %d | %.3f -> %.3f | %d (%.3f) |" %
                 (name, s["failures_before"], s["failures_after"],
                  s["mfr_before"], s["mfr_after"], s["failures_after_strict"],
                  s["mfr_after_strict"]))
    L += ["", "Family by family (failures), before -> after:", "",
          "| system | " + " | ".join(fams) + " |",
          "|---|" + "---|" * len(fams)]
    for name in ORDER:
        pf = s4[name]["per_family"]
        L.append("| %s | " % name + " | ".join(
            "%d -> %d" % (pf[f]["failures_before"], pf[f]["failures_after"])
            for f in fams) + " |")
    with open(os.path.join(report_dir, REPORT_MD), "w", encoding="utf-8",
              newline="\n") as f:
        f.write("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="the tuning campaigns re-scored under the 2026-09-26 rule")
    ap.add_argument("--run", action="store_true",
                    help="re-score the committed tuning bundles and report")
    ap.add_argument("--cases", action="append", default=[], metavar="SOURCE=DIR",
                    help="a campaign directory for a source's case reports")
    ap.add_argument("--out", metavar="DIR", help="report directory")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.run:
        ap.error("--run or --selftest")
    known = [p for srcs in SOURCES.values() for p in srcs]
    cases = {}
    for c in a.cases:
        src, _, d = c.partition("=")
        if not _ or src not in known:
            sys.stderr.write("rescore: --cases SOURCE=DIR ... with SOURCE one "
                             "of: %s\n" % ", ".join(known))
            return 2
        cases[src] = d
    run(cases, report_dir=a.out or REPORT_DIR)
    return 0



# --- the selftest ------------------------------------------------------------


def _g1_kinds(H):
    by = {}
    for r in H["rows"]:
        by[(r["geometry_id"], r["attempt"])] = r
    assert classify(by[("B-1-000", 1)], None) == "smooth"
    assert classify(by[("D-1-010", 1)], None) == "plane"
    assert classify(by[("A-1-049", 2)], None) == "zero"
    r4 = copy.deepcopy(by[("A-1-049", 2)])
    r4["config"]["snap"]["feature_tolerance"] = 0.5
    assert classify(r4, None) == "no_summary"


def _g2_tiny(H):
    gids = {"A-1-000", "A-1-049", "B-1-000", "D-1-010"}
    s = rescore_system([(H["src"], H["cut"](gids))], H["mcut"](gids),
                       H["gates"], H["knobs"], None)
    assert s["n"] == 4 and s["cases_read"] == 0, s
    assert (s["failures_before"], s["failures_after"],
            s["failures_after_strict"]) == (1, 2, 3), s
    assert s["pass_rows"] == {"smooth": 1, "plane": 1, "zero": 1,
                              "unmeasured": 0, "no_summary": 0}, s["pass_rows"]
    assert (s["flipped"], s["flipped_strict"]) == (1, 2), s
    H["tiny_cut"] = (gids,)


def _g3_identity(H):
    gids = {"A-1-000", "A-1-049", "B-1-000", "D-1-010"}
    b3 = H["cut"](gids)
    for g in b3["geometries"]:
        if g["geometry_id"] == "A-1-049":
            g["failure"] = True
    try:
        rescore_system([(H["src"], b3)], H["mcut"](gids),
                       H["gates"], H["knobs"], None)
    except RescoreError as e:
        assert "identity" in str(e) and "A-1-049" in str(e), str(e)
    else:
        raise AssertionError("an end that contradicts its rows was not refused")


def _g4_seal(H):
    gids = {"A-1-000", "A-1-049", "B-1-000", "D-1-010"}
    b4 = H["cut"](gids)
    b4["campaign"]["split_mode"] = split.EVALUATE
    try:
        rescore_system([(H["src"], b4)], H["mcut"](gids),
                       H["gates"], H["knobs"], None)
    except split.SplitSealed:
        pass
    else:
        raise AssertionError("an evaluate-mode bundle was not refused")


def _g5_determinism(H):
    gids = {"A-1-000", "A-1-049", "B-1-000", "D-1-010"}
    args = ([(H["src"], H["cut"](gids))], H["mcut"](gids), H["gates"], H["knobs"])
    a = json.dumps(rescore_system(*args, cases=None), sort_keys=True)
    b = json.dumps(rescore_system(*args, cases=None), sort_keys=True)
    assert a == b


def _g6_cli(H):
    for bad in ("nope", "prior/x.json.gz=C:/"):
        err = io.StringIO()
        old = sys.stderr
        sys.stderr = err
        try:
            code = main(["--run", "--cases", bad])
        except SystemExit as e:
            code = e.code
        finally:
            sys.stderr = old
        assert code == 2, (bad, code)
        assert "--cases" in err.getvalue(), (bad, err.getvalue())


def selftest():
    gates = schema.load_gates()
    knobs = schema.load_knobs()
    mrows = {r["geometry_id"]: r
             for r in campaign.load_manifest("tuning", "rules")}
    src = "prior/tuning_rules.json.gz"
    bundle = baseline.read_bundle(os.path.join(HERE, src))
    rows = optimise.rows_from(bundle, src, mrows, gates, knobs)
    H = {"gates": gates, "knobs": knobs, "mrows": mrows, "src": src,
         "cut": lambda gids: _cut_bundle(bundle, gids),
         "mcut": lambda gids: _cut(mrows, gids), "rows": rows}
    groups = ((_g1_kinds, "capture kinds"), (_g2_tiny, "tiny re-score"),
              (_g3_identity, "identity"), (_g4_seal, "seal"),
              (_g5_determinism, "determinism"), (_g6_cli, "cli"))
    bad = 0
    for fn, name in groups:
        try:
            fn(H)
        except Exception as e:  # noqa: BLE001 - the failure IS the report
            bad += 1
            print("SELFTEST FAIL: %s: %s: %s" % (name, type(e).__name__, e))
        else:
            print("[ok] %s" % name)
    print("SELFTEST PASS" if not bad else "SELFTEST FAIL")
    return 1 if bad else 0


def _cut_bundle(bundle, gids):
    b = copy.deepcopy(bundle)
    b["geometries"] = [g for g in b["geometries"] if g["geometry_id"] in gids]
    b["attempts"] = [r for r in b["attempts"] if r["geometry_id"] in gids]
    return b


if __name__ == "__main__":
    sys.exit(main())
