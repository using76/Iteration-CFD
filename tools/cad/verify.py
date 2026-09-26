#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""verify.py - stages S4 and S10 of the CAD loop (docs/16 §D, §E.4): judge each compiled check against its measurement as pass, fail or not_evaluable from the margin against the measurement uncertainty u, and derive the design verdict from the hard rows.

The tri-state verdict with a deterministically derived overall verdict is an idea taken from ai-cad (Apache-2.0,
dfma_evaluator.py, ADR 0009 #5), reimplemented here from docs/16 §E.4; no ai-cad code is copied.

Usage:
  python verify.py --selftest
  python verify.py judge CHECKS_JSON MEASUREMENTS_JSON OUT_JSON EVAL_KEY [CFD_U_JSON]
"""

import copy
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common
import schema

REASONS = ("NE-UNCERTAIN", "NE-MISSING", "NE-ERROR")
RAISED_ID = "MEAS-RAISED"
SHA_RE = re.compile("^[0-9a-f]{64}$")
CFD_KEYS = ("gci_fine", "repeat_band")
CASES = os.path.join(HERE, "fixtures", "verify", "cases.json")
GOLDEN_V1 = os.path.join(HERE, "fixtures", "reqs", "golden", "v1_nominal.json")
USAGE = ("usage: python verify.py --selftest" + chr(10)
         + "       python verify.py judge CHECKS_JSON MEASUREMENTS_JSON OUT_JSON EVAL_KEY [CFD_U_JSON]")


def _num(x) -> bool:
    """A finite non-bool number: the only kind that can carry m, u, tol or a bound."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def check_shape(check) -> None:
    """Refuse a compiled check the rule of docs/16 §E.4 cannot decide, by VERIFY-CHECK."""
    rid = check["req_id"]
    tol = check["tol"]
    if not _num(tol) or tol < 0:
        raise ValueError("VERIFY-CHECK: %s: tol %r is not a finite number >= 0" % (rid, tol))
    u = check["u"]
    if u is not None and (not _num(u) or u < 0):
        raise ValueError("VERIFY-CHECK: %s: u %r is not None or a finite number >= 0" % (rid, u))
    op, lo, hi = check["op"], check["lo"], check["hi"]
    if check["hardness"] == "objective":
        if op not in ("<=", ">="):
            raise ValueError("VERIFY-CHECK: %s: objective op %r is not <= or >=" % (rid, op))
        if lo is not None or hi is not None:
            raise ValueError("VERIFY-CHECK: %s: objective has bounds lo %r hi %r" % (rid, lo, hi))
        return
    if op == "is_true":
        if lo is not None or hi is not None:
            raise ValueError("VERIFY-CHECK: %s: is_true has bounds lo %r hi %r" % (rid, lo, hi))
    elif op == "<=":
        if lo is not None or not _num(hi):
            raise ValueError("VERIFY-CHECK: %s: <= needs lo None and a finite hi, got %r %r" % (rid, lo, hi))
    elif op == ">=":
        if hi is not None or not _num(lo):
            raise ValueError("VERIFY-CHECK: %s: >= needs hi None and a finite lo, got %r %r" % (rid, lo, hi))
    elif op == "==":
        if not _num(lo) or hi != lo:
            raise ValueError("VERIFY-CHECK: %s: == needs finite lo == hi, got %r %r" % (rid, lo, hi))
    elif op == "in":
        if not _num(lo) or not _num(hi) or lo > hi:
            raise ValueError("VERIFY-CHECK: %s: in needs finite lo <= hi, got %r %r" % (rid, lo, hi))
    else:
        raise ValueError("VERIFY-CHECK: %s: op %r is not one of <= >= == in is_true" % (rid, op))


def decide(op, m, u, lo, hi, tol):
    """The margin rule of docs/16 §E.4: pass, fail, or None (the uncertainty straddles the bound)."""
    if op == "<=":
        b = hi
        if m + u <= b + tol:
            return "pass"
        if m - u > b + tol:
            return "fail"
        return None
    if op == ">=":
        b = lo
        if m - u >= b - tol:
            return "pass"
        if m + u < b - tol:
            return "fail"
        return None
    if op == "==":
        d = abs(m - lo)
        if d + u <= tol:
            return "pass"
        if d - u > tol:
            return "fail"
        return None
    if op == "in":
        if m - u >= lo - tol and m + u <= hi + tol:
            return "pass"
        if m + u < lo - tol or m - u > hi + tol:
            return "fail"
        return None
    raise ValueError("VERIFY-CHECK: op %r is not one of <= >= == in" % (op,))


def resolve_u(check, record, cfd):
    """The row's u of docs/16 §E.4: the compiled u, else max of the CFD entry, lifted by the record's u_meas.

    A check with u null and no CFD u leaves u unknown (None), never 0 (docs/16 §E.4); is_true has no u.
    """
    if check["op"] == "is_true":
        return 0.0
    base = check["u"]
    if base is None and cfd is not None:
        vals = []
        for key in CFD_KEYS:
            if key in cfd and cfd[key] is not None:
                v = cfd[key]
                if not _num(v) or v < 0:
                    raise ValueError("VERIFY-CFDU: %s: %s %r is not None or a finite number >= 0"
                                     % (check["req_id"], key, v))
                vals.append(float(v))
        if vals:
            base = max(vals)
    if base is None:
        return None
    if isinstance(record, dict) and record.get("u_meas") is not None:
        return float(max(base, record["u_meas"]))
    return float(base)


def judge(check, record, cfd=None, evidence=None):
    """One cad-verdict/1 row: docs/16 §E.4's rule applied to one check and its measurement."""
    check_shape(check)
    if evidence is not None:
        if not isinstance(evidence, dict) or not isinstance(evidence.get("path"), str) \
                or not evidence["path"] or not isinstance(evidence.get("sha"), str) \
                or not SHA_RE.match(evidence["sha"]):
            raise ValueError("VERIFY-EVIDENCE: %s: evidence must be a dict with a non-empty path "
                             "and a 64-hex sha" % (check["req_id"],))
    rid = check["req_id"]
    row = {"req_id": rid, "primitive": check["primitive"], "feature": check["args"]["feature"],
           "hardness": check["hardness"], "m": None, "u": None, "verdict": "not_evaluable",
           "reason_id": None, "evidence_path": None, "evidence_sha": None}
    if isinstance(record, dict) and evidence is not None:
        row["evidence_path"] = evidence["path"]
        row["evidence_sha"] = evidence["sha"]

    def not_ev(reason):
        row["verdict"] = "not_evaluable"
        row["reason_id"] = reason
        return row

    if record is None:
        return not_ev("NE-MISSING")
    if isinstance(record, BaseException):
        return not_ev("NE-ERROR")
    if not isinstance(record, dict):
        raise ValueError("VERIFY-RECORD: %s: record is %s, not a cad-measure/1 dict"
                         % (rid, type(record).__name__))
    errs = schema.errors(record, "cad-measure/1")
    if errs:
        raise ValueError("VERIFY-RECORD: %s: %s" % (rid, errs[0]))
    if record["primitive"] != check["primitive"]:
        raise ValueError("VERIFY-RECORD: %s: primitive %r is not %r"
                         % (rid, record["primitive"], check["primitive"]))
    if record["status"] == "error":
        return not_ev("NE-ERROR")
    if record["status"] == "refused":
        return not_ev("NE-MISSING")
    if not _num(record["value"]):
        return not_ev("NE-ERROR")
    if record["u_meas"] is not None and not _num(record["u_meas"]):
        return not_ev("NE-ERROR")                       # a NaN u_meas would vanish inside max()
    m = float(record["value"])
    if check["op"] == "is_true":
        if m == 1.0 or m == 0.0:
            row["m"] = m
            row["u"] = 0.0
            row["verdict"] = "pass" if m == 1.0 else "fail"
            return row
        return not_ev("NE-ERROR")
    u = resolve_u(check, record, cfd)
    row["m"] = m
    row["u"] = u
    if u is None:
        return not_ev("NE-UNCERTAIN")
    if check["hardness"] == "objective":
        row["verdict"] = "pass"
        return row
    got = decide(check["op"], m, u, check["lo"], check["hi"], check["tol"])
    if got is None:
        return not_ev("NE-UNCERTAIN")
    row["verdict"] = got
    return row


def design_verdict(rows) -> str:
    """The design verdict of docs/16 §E.4, from the hard rows only: any fail infeasible, all pass feasible."""
    hard = [r for r in rows if r["hardness"] == "hard"]
    if not hard:
        raise ValueError("VERIFY-NOHARD: no hard row")
    if any(r["verdict"] == "fail" for r in hard):
        return "infeasible"
    if all(r["verdict"] == "pass" for r in hard):
        return "feasible"
    return "not_evaluable"


def evaluate(checks_doc, measurements, eval_key, cfd_u=None, evidence=None) -> dict:
    """One evaluation (docs/16 §D S4/S10): a cad-checks/1 doc plus measurements into a cad-verdict/1 doc."""
    errs = schema.errors(checks_doc, "cad-checks/1")
    if errs:
        raise ValueError("VERIFY-CHECKS: %s" % (errs[0],))
    checks = checks_doc["checks"]
    seen = set()
    for c in checks:
        if c["req_id"] in seen:
            raise ValueError("VERIFY-CHECKS: %s: duplicate req_id" % (c["req_id"],))
        seen.add(c["req_id"])
    if not isinstance(eval_key, str) or not SHA_RE.match(eval_key):
        raise ValueError("VERIFY-KEY: eval_key %r is not a 64-hex sha" % (eval_key,))
    if not isinstance(measurements, dict):
        raise ValueError("VERIFY-MEAS: measurements is %s, not a dict" % type(measurements).__name__)
    strays = sorted(k for k in measurements if k not in seen)
    if strays:
        raise ValueError("VERIFY-STRAY: %s" % (strays[0],))
    if cfd_u is not None:
        if not isinstance(cfd_u, dict):
            raise ValueError("VERIFY-CFDU: cfd_u is %s, not a dict" % type(cfd_u).__name__)
        for k, v in cfd_u.items():
            if k not in seen:
                raise ValueError("VERIFY-STRAY: %s" % (k,))
            if not isinstance(v, dict) or any(kk not in CFD_KEYS for kk in v):
                raise ValueError("VERIFY-CFDU: %s: %r is not a dict of %s keys"
                                 % (k, v, "/".join(CFD_KEYS)))
    rows = [judge(c, measurements.get(c["req_id"]), (cfd_u or {}).get(c["req_id"]), evidence)
            for c in checks]
    doc = {"schema": "cad-verdict/1", "study_id": checks_doc["study_id"], "eval_key": eval_key,
           "design_verdict": design_verdict(rows), "verdicts": rows}
    errs = schema.errors(doc, "cad-verdict/1")
    if errs:
        raise ValueError("VERIFY-DOC: %s" % (errs[0],))
    return doc


def tally(doc) -> dict:
    """The cad-iteration/1 tally of one verdict doc: hard counts and the objective's m and u."""
    n_pass = n_fail = n_ne = n_obj = 0
    obj_m = obj_u = None
    for r in doc["verdicts"]:
        if r["hardness"] == "hard":
            if r["verdict"] == "pass":
                n_pass += 1
            elif r["verdict"] == "fail":
                n_fail += 1
            else:
                n_ne += 1
        elif r["hardness"] == "objective":
            n_obj += 1
            if n_obj > 1:
                raise ValueError("VERIFY-OBJ: more than one objective row")
            obj_m, obj_u = r["m"], r["u"]
    return {"n_hard_pass": n_pass, "n_hard_fail": n_fail, "n_hard_ne": n_ne,
            "objective": obj_m, "objective_u": obj_u}


def guarded(primitive, unit, fn, *args, **kwargs) -> dict:
    """Run a measurement call; an Exception becomes a MEAS-RAISED error record, SystemExit still dies."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        return {"schema": "cad-measure/1", "primitive": primitive, "feature": None, "where": [],
                "value": None, "unit": unit, "u_meas": None, "method": "guarded call", "status": "error",
                "reason_id": RAISED_ID, "detail": ("%s: %s" % (type(e).__name__, e))[:300]}


def _rec(primitive, value, u_meas, unit="m", status="ok", reason_id=None) -> dict:
    """A fixture cad-measure/1 record: feature None, where [], method "fixture", detail ""."""
    return {"schema": "cad-measure/1", "primitive": primitive, "feature": None, "where": [],
            "value": value, "unit": unit, "u_meas": u_meas, "method": "fixture", "status": status,
            "reason_id": reason_id, "detail": ""}


def _raises(fn) -> None:
    """Assert fn raised ValueError - not another exception type, and not nothing."""
    try:
        fn()
    except ValueError:
        return
    except Exception as e:
        raise AssertionError("expected ValueError, got %s: %s" % (type(e).__name__, e))
    raise AssertionError("expected ValueError, nothing raised")


def selftest() -> None:
    cases_doc = common.read_json(CASES)
    cases = cases_doc["cases"]
    ev = cases_doc["evidence"]
    assert len(cases) == 30, "expected 30 table cases, got %d" % len(cases)
    names = [c["name"] for c in cases]
    assert len(set(names)) == 30, "the case names are not unique"
    n_pass = n_fail = 0
    nes = {"NE-UNCERTAIN": 0, "NE-MISSING": 0, "NE-ERROR": 0}
    rows = []
    for case in cases:
        check = case["check"]
        rec = RuntimeError("fixture raise") if case["record"] == "RAISE" else copy.deepcopy(case["record"])
        has_rec = isinstance(case["record"], dict)
        snap = (common.canonical_json(check),
                common.canonical_json(rec) if isinstance(rec, dict) else None,
                common.canonical_json(case["cfd"]) if case["cfd"] is not None else None)
        got = judge(check, rec, case["cfd"], ev)
        want = case["want"]
        wanted = {"req_id": check["req_id"], "primitive": check["primitive"],
                  "feature": check["args"]["feature"], "hardness": check["hardness"],
                  "m": want["m"], "u": want["u"], "verdict": want["verdict"],
                  "reason_id": want["reason_id"],
                  "evidence_path": ev["path"] if has_rec else None,
                  "evidence_sha": ev["sha"] if has_rec else None}
        assert got == wanted, case["name"]
        assert common.canonical_json(got) == common.canonical_json(wanted), case["name"]
        assert snap == (common.canonical_json(check),
                        common.canonical_json(rec) if isinstance(rec, dict) else None,
                        common.canonical_json(case["cfd"]) if case["cfd"] is not None else None), case["name"]
        rows.append(got)
        if got["verdict"] == "pass":
            n_pass += 1
        elif got["verdict"] == "fail":
            n_fail += 1
        else:
            nes[got["reason_id"]] += 1
    print("[ok] 30 of 30 table cases judged right: %d pass, %d fail, %d NE-UNCERTAIN, %d NE-MISSING, %d NE-ERROR"
          % (n_pass, n_fail, nes["NE-UNCERTAIN"], nes["NE-MISSING"], nes["NE-ERROR"]))
    doc = {"schema": "cad-verdict/1", "study_id": "verify_cases", "eval_key": "0" * 64,
           "design_verdict": design_verdict(rows), "verdicts": rows}
    assert schema.errors(doc, "cad-verdict/1") == []
    assert doc["design_verdict"] == "infeasible"
    print("[ok] the 30 rows validate as cad-verdict/1; their design verdict is infeasible")

    def dv(*pairs):
        return design_verdict([{"hardness": h, "verdict": v} for h, v in pairs])

    assert dv(("hard", "pass"), ("hard", "pass"), ("soft", "fail"),
              ("objective", "not_evaluable")) == "feasible"
    assert dv(("hard", "pass"), ("hard", "fail"), ("hard", "not_evaluable")) == "infeasible"
    assert dv(("hard", "pass"), ("hard", "not_evaluable")) == "not_evaluable"
    assert dv(("hard", "not_evaluable"), ("hard", "fail")) == "infeasible"
    assert dv(("hard", "pass"), ("soft", "not_evaluable")) == "feasible"
    assert dv(("hard", "not_evaluable"), ("hard", "not_evaluable")) == "not_evaluable"
    _raises(lambda: dv(("soft", "pass"), ("objective", "pass")))
    print("[ok] 7 of 7 design derivations right")

    C = common.read_json(GOLDEN_V1)["checks"]
    M = {"REQ-001": _rec("diameter_at_plane", 0.06, 1e-9),
         "REQ-002": _rec("area_ratio", 9.0, 9e-9, unit="1"),
         "REQ-003": _rec("extent_along_axis", 0.07, 1e-9),
         "REQ-004": _rec("meridian_min_wall", 0.003, 1e-8),
         "REQ-005": _rec("slope_max", 0.5, 5e-7, unit="rad"),
         "REQ-006": _rec("extent_along_axis", 0.07, 1e-9),
         "SYS-SOLID": _rec("n_solids", 1.0, 0.0, unit="1"),
         "SYS-VALID": _rec("valid", 1.0, 0.0, unit="1"),
         "SYS-WATERTIGHT": _rec("watertight", 1.0, 0.0, unit="1"),
         "SYS-AXIS": _rec("axis_x", 1.0, 0.0, unit="1"),
         "SYS-UNITS": _rec("units_m", 1.0, 0.0, unit="1")}
    K = "a" * 64
    snap_c, snap_m = common.canonical_json(C), common.canonical_json(M)

    def by_id(d, rid):
        return [r for r in d["verdicts"] if r["req_id"] == rid][0]

    doc_a = evaluate(C, M, K, None, ev)
    assert doc_a["design_verdict"] == "not_evaluable"
    mach = by_id(doc_a, "SYS-MACH")
    assert mach["verdict"] == "not_evaluable" and mach["reason_id"] == "NE-MISSING"
    assert mach["m"] is None and mach["u"] is None
    assert tally(doc_a) == {"n_hard_pass": 9, "n_hard_fail": 0, "n_hard_ne": 1,
                            "objective": 0.07, "objective_u": 1e-9}
    assert [r["req_id"] for r in doc_a["verdicts"]] == [c["req_id"] for c in C["checks"]]
    assert schema.errors(doc_a, "cad-verdict/1") == []

    M_b = dict(M)
    M_b["SYS-MACH"] = _rec("mach_max", 0.1, None, unit="1")
    doc_b = evaluate(C, M_b, K, None, ev)
    assert doc_b["design_verdict"] == "not_evaluable"
    mach = by_id(doc_b, "SYS-MACH")
    assert mach["verdict"] == "not_evaluable" and mach["reason_id"] == "NE-UNCERTAIN"
    assert mach["m"] == 0.1 and mach["u"] is None

    cfd = {"SYS-MACH": {"gci_fine": 0.01, "repeat_band": 0.02}}
    doc_c = evaluate(C, M_b, K, cfd, ev)
    assert doc_c["design_verdict"] == "feasible"
    mach = by_id(doc_c, "SYS-MACH")
    assert mach["verdict"] == "pass" and mach["u"] == 0.02
    assert tally(doc_c) == {"n_hard_pass": 10, "n_hard_fail": 0, "n_hard_ne": 0,
                            "objective": 0.07, "objective_u": 1e-9}

    M_d = dict(M_b)
    M_d["REQ-004"] = _rec("meridian_min_wall", 0.0018759, 1e-8)
    doc_d = evaluate(C, M_d, K, cfd, ev)
    assert doc_d["design_verdict"] == "infeasible"
    assert by_id(doc_d, "REQ-004")["verdict"] == "fail"
    assert tally(doc_d) == {"n_hard_pass": 9, "n_hard_fail": 1, "n_hard_ne": 0,
                            "objective": 0.07, "objective_u": 1e-9}

    M_e = dict(M_b)
    M_e["REQ-005"] = _rec("slope_max", 0.7, 5e-7, unit="rad")
    doc_e = evaluate(C, M_e, K, cfd, ev)
    assert by_id(doc_e, "REQ-005")["verdict"] == "fail"
    assert doc_e["design_verdict"] == "feasible"

    assert common.canonical_json(C) == snap_c and common.canonical_json(M) == snap_m
    print("[ok] v1 golden end-to-end: not_evaluable without Mach, NE-UNCERTAIN without CFD u, "
          "feasible with u 0.02, infeasible at wall 0.0018759, a soft fail leaves it feasible")

    chk_le = [c for c in cases if c["name"] == "le_pass_clear"][0]["check"]
    chk_true = [c for c in cases if c["name"] == "is_true_pass"][0]["check"]
    chk_mach = [c for c in cases if c["name"] == "cfd_u_absent_is_ne_not_zero"][0]["check"]

    def _chk(op, lo, hi, tol, u, hardness="hard"):
        return {"req_id": "REQ-090", "primitive": "extent_along_axis",
                "args": {"feature": None, "where": [], "Re": None, "level": None},
                "op": op, "lo": lo, "hi": hi, "tol": tol, "u": u, "hardness": hardness}

    _raises(lambda: judge(_chk("<=", None, 1.0, 0.0, 1e-9, hardness="objective"), None))
    _raises(lambda: judge(_chk("==", 1.0, 2.0, 0.0, 0.0), None))
    _raises(lambda: judge(_chk("in", 4.0, 2.0, 0.0, 0.0), None))
    _raises(lambda: judge(_chk("<=", None, 4.0, 0.0, -1.0), None))
    _raises(lambda: judge(chk_le, _rec("n_solids", 3.0, 0.0, unit="1")))
    _raises(lambda: judge(chk_le, _rec(chk_le["primitive"], 3.0, 0.5, status="okay")))
    _raises(lambda: judge(chk_le, "3.0"))
    _raises(lambda: judge(chk_le, _rec(chk_le["primitive"], 3.0, 0.5), None,
                          {"path": "p", "sha": "xyz"}))
    _raises(lambda: evaluate(C, {"REQ-099": None}, K))
    _raises(lambda: evaluate(C, dict(M_b), K, {"SYS-MACH": {"gci_fine": -1.0}}))
    _raises(lambda: evaluate(C, dict(M_b), "abc"))
    bad_doc = copy.deepcopy(C)
    bad_doc["schema"] = "cad-checks/2"
    _raises(lambda: evaluate(bad_doc, dict(M_b), K))
    print("[ok] 12 of 12 malformed inputs refused with ValueError")

    for r in (judge(chk_le, _rec(chk_le["primitive"], None, 0.5)),
              judge(chk_le, _rec(chk_le["primitive"], float("nan"), 0.5)),
              judge(chk_le, _rec(chk_le["primitive"], float("inf"), 0.5)),
              judge(chk_true, _rec(chk_true["primitive"], 0.5, 0.0, unit="1")),
              judge(chk_true, _rec(chk_true["primitive"], 2.0, 0.0, unit="1")),
              judge(chk_le, _rec(chk_le["primitive"], 3.0, float("nan")))):
        assert r["verdict"] == "not_evaluable" and r["reason_id"] == "NE-ERROR", r
        assert r["m"] is None and r["u"] is None
    r_zero = judge(chk_mach, _rec(chk_mach["primitive"], 0.125, None, unit="1"),
                   {"gci_fine": 0.0, "repeat_band": None})
    assert r_zero["verdict"] == "pass" and r_zero["u"] == 0.0
    print("[ok] 7 of 7 edge values: None, NaN, inf, non-boolean 0.5 and 2.0 and a NaN u_meas are NE-ERROR; "
          "an explicit CFD u of 0 is used")

    def _boom():
        raise RuntimeError("boom")

    rec = guarded("extent_along_axis", "m", _boom)
    assert rec["status"] == "error" and rec["reason_id"] == RAISED_ID
    assert rec["detail"] == "RuntimeError: boom"
    assert schema.errors(rec, "cad-measure/1") == []
    got = judge(chk_le, rec)
    assert got["verdict"] == "not_evaluable" and got["reason_id"] == "NE-ERROR"

    r_fix = _rec(chk_le["primitive"], 3.0, 0.5)
    assert guarded("extent_along_axis", "m", lambda: r_fix) is r_fix

    def _die():
        raise SystemExit(3)

    try:
        guarded("extent_along_axis", "m", _die)
        raise AssertionError("SystemExit did not propagate")
    except SystemExit:
        pass
    print("[ok] guarded: a raise becomes a MEAS-RAISED error record judged NE-ERROR; SystemExit propagates")

    with tempfile.TemporaryDirectory() as td:
        checks_p = os.path.join(td, "checks.json")
        meas_p = os.path.join(td, "meas.json")
        cfd_p = os.path.join(td, "cfd.json")
        out_p = os.path.join(td, "out.json")
        common.write_json(checks_p, C)
        common.write_json(meas_p, M_b)
        common.write_json(cfd_p, cfd)
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        cmd = [sys.executable, os.path.abspath(__file__), "judge",
               checks_p, meas_p, out_p, "a" * 64, cfd_p]
        p1 = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env, timeout=120)
        with open(out_p, "rb") as f:
            b1 = f.read()
        p2 = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env, timeout=120)
        with open(out_p, "rb") as f:
            b2 = f.read()
        assert p1.returncode == 0 and p2.returncode == 0, (p1.returncode, p1.stderr, p2.stderr)
        assert b1 == b2
        ref = (common.canonical_json(evaluate(C, M_b, "a" * 64, cfd,
                                              {"path": "meas.json", "sha": common.sha256_file(meas_p)}))
               + chr(10)).encode("utf-8")
        assert b1 == ref
        assert '"design_verdict":"feasible"' in p1.stdout

        out_ne = os.path.join(td, "out_ne.json")
        p3 = subprocess.run([sys.executable, os.path.abspath(__file__), "judge",
                             checks_p, meas_p, out_ne, "a" * 64],
                            capture_output=True, text=True, encoding="utf-8", env=env, timeout=120)
        assert p3.returncode == 1, p3.returncode
        with open(out_ne, "rb") as f:
            d_ne = json.loads(f.read().decode("utf-8"))
        assert d_ne["design_verdict"] == "not_evaluable"

        out_bad = os.path.join(td, "out_bad.json")
        p4 = subprocess.run([sys.executable, os.path.abspath(__file__), "judge",
                             checks_p, meas_p, out_bad, "abc"],
                            capture_output=True, text=True, encoding="utf-8", env=env, timeout=120)
        assert p4.returncode == 2 and not os.path.exists(out_bad)
        p5 = subprocess.run([sys.executable, os.path.abspath(__file__), "judge", checks_p],
                            capture_output=True, text=True, encoding="utf-8", env=env, timeout=120)
        assert p5.returncode == 2
        assert "cadquery" not in sys.modules and "OCP" not in sys.modules and "measure" not in sys.modules
        print("[ok] CLI: two runs byte-identical and equal to evaluate(); exits 0 feasible, "
              "1 not_evaluable, 2 bad key, 2 usage; no CAD module loaded")
    print("SELFTEST PASS")


def main(argv) -> int:
    """The CLI of docs/16 §I CAD-08: --selftest, judge."""
    if argv == ["--selftest"]:
        try:
            selftest()
        except Exception:
            traceback.print_exc()
            return 1
        return 0
    try:
        if len(argv) in (5, 6) and argv[0] == "judge":
            checks_doc = common.read_json(argv[1])
            measurements = common.read_json(argv[2])
            evidence = {"path": os.path.basename(argv[2]), "sha": common.sha256_file(argv[2])}
            cfd_u = common.read_json(argv[5]) if len(argv) == 6 else None
            doc = evaluate(checks_doc, measurements, argv[4], cfd_u, evidence)
            common.atomic_write(argv[3], common.canonical_json(doc) + chr(10))
            t = tally(doc)
            print(common.canonical_json({"design_verdict": doc["design_verdict"],
                                         "n_hard_pass": t["n_hard_pass"],
                                         "n_hard_fail": t["n_hard_fail"],
                                         "n_hard_ne": t["n_hard_ne"]}))
            return 0 if doc["design_verdict"] == "feasible" else 1
    except (ValueError, OSError) as e:
        print("verify: %s" % (e,), file=sys.stderr)
        return 2
    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
