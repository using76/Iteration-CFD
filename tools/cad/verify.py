#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""verify.py - stages S4 and S10 of the CAD loop (docs/16 §D, §E.4): judge each compiled check against its measurement as pass, fail or not_evaluable from the margin against the measurement uncertainty u, and derive the design verdict from the hard rows.

The tri-state verdict with a deterministically derived overall verdict is an idea taken from ai-cad (Apache-2.0,
dfma_evaluator.py, ADR 0009 #5), reimplemented here from docs/16 §E.4; no ai-cad code is copied.

AMG-8 (docs/16a §B.1, §E, §F) reimplements, from reading only (no code copied), Amagine3D's coverage equality
(skills/text-a3d/scene_contract.py `validate`, build_manifest.py `bind_inputs`): the verdict, check and locked-row id
sets are equal and each check comes from exactly one locked row, or VER-COVER; every verdict doc names the
requirements lock and the checks sha it was judged under. m is only a cad-measure/1 record's value, and u and tol
are only the check's own, compiled into checks.json for its representation (repr); a CFD u applies to repr cfd
only; there is no tolerance option.

Usage:
  python verify.py --selftest
  python verify.py --help
  python verify.py judge CHECKS_JSON REQUIREMENTS_DIR MEASUREMENTS_JSON OUT_JSON EVAL_KEY [CFD_U_JSON]
"""

import ast
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
import reqs
import schema

REASONS = ("NE-UNCERTAIN", "NE-MISSING", "NE-ERROR")
RAISED_ID = "MEAS-RAISED"
SHA_RE = re.compile("^[0-9a-f]{64}$")
CFD_KEYS = ("gci_fine", "repeat_band")
CASES = os.path.join(HERE, "fixtures", "verify", "cases.json")
GOLDEN_V1 = os.path.join(HERE, "fixtures", "reqs", "golden", "v1_nominal.json")
USAGE = ("usage: python verify.py --selftest" + chr(10)
         + "       python verify.py --help" + chr(10)
         + "       python verify.py judge CHECKS_JSON REQUIREMENTS_DIR MEASUREMENTS_JSON OUT_JSON EVAL_KEY"
         + " [CFD_U_JSON]")


def _num(x) -> bool:
    """A finite non-bool number: the only kind that can carry m, u, tol or a bound."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def check_shape(check) -> None:
    """Refuse a compiled check the rule of docs/16 §E.4 cannot decide, by VERIFY-CHECK."""
    rid = check["req_id"]
    if check.get("repr") not in reqs.REPRS:
        raise ValueError("VERIFY-CHECK: %s: repr %r is not one of %s"
                         % (rid, check.get("repr"), " ".join(reqs.REPRS)))
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

    A check with u null and no CFD u leaves u unknown (None), never 0 (docs/16 §E.4); is_true has no u. The compiled
    u is the check's own representation's (docs/16a §B.1): the record's u_meas can only raise it, never stand in
    for it, so a BRep u_meas never judges an STL, mesh or CFD comparison.
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
    if cfd is not None and check["repr"] != "cfd":
        raise ValueError("VERIFY-CFDU: %s: a CFD u applies to repr cfd only, this check is repr %s"
                         % (check["req_id"], check["repr"]))
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


def cover(checks_doc, requirements_doc) -> None:
    """VER-COVER of docs/16a §E: the checks are exactly the locked rows', one check per row, each from one row.

    The locked document must match its lock (GATE-LOCK); the checks must name that lock, its declaration and its
    study; no check id twice; no check without a locked row, no locked row without a check; no check id carried
    by two locked rows; each check's hardness and op are its row's."""
    if not reqs.lock_ok(requirements_doc):
        raise ValueError("GATE-LOCK: the requirements lock does not match its document")
    for key, want in (("requirements_lock", requirements_doc["lock_sha"]),
                      ("declaration_sha", requirements_doc["declaration_sha"]),
                      ("study_id", requirements_doc["study_id"])):
        if checks_doc[key] != want:
            raise ValueError("VER-COVER: the checks carry %s %r, the locked set %r" % (key, checks_doc[key], want))
    rows, checks = requirements_doc["rows"], checks_doc["checks"]
    n_rows = {}
    for r in rows:
        n_rows[r["id"]] = n_rows.get(r["id"], 0) + 1
    seen = set()
    for c in checks:
        if c["req_id"] in seen:
            raise ValueError("VER-COVER: check %s appears twice" % (c["req_id"],))
        seen.add(c["req_id"])
    for c in checks:
        if c["req_id"] not in n_rows:
            raise ValueError("VER-COVER: extra check %s has no locked row" % (c["req_id"],))
    for r in rows:
        if r["id"] not in seen:
            raise ValueError("VER-COVER: locked row %s has no check" % (r["id"],))
    for c in checks:
        if n_rows[c["req_id"]] != 1:
            raise ValueError("VER-COVER: check %s comes from %d locked rows" % (c["req_id"], n_rows[c["req_id"]]))
    by_id = dict((r["id"], r) for r in rows)
    for c in checks:
        for key in ("hardness", "op"):
            if c[key] != by_id[c["req_id"]][key]:
                raise ValueError("VER-COVER: check %s %s %r is not its row's %r"
                                 % (c["req_id"], key, c[key], by_id[c["req_id"]][key]))


def evaluate(checks_doc, requirements_doc, measurements, eval_key, cfd_u=None, evidence=None) -> dict:
    """One evaluation (docs/16 §D S4/S10): a cad-checks/1 doc, its locked cad-requirements/1 doc and measurements
    into a cad-verdict/1 doc bound to the lock and the checks sha (docs/16a §F)."""
    errs = schema.errors(checks_doc, "cad-checks/1")
    if errs:
        raise ValueError("VERIFY-CHECKS: %s" % (errs[0],))
    errs = schema.errors(requirements_doc, "cad-requirements/1")
    if errs:
        raise ValueError("VERIFY-REQS: %s" % (errs[0],))
    cover(checks_doc, requirements_doc)
    checks = checks_doc["checks"]
    seen = set(c["req_id"] for c in checks)
    reprs = dict((c["req_id"], c["repr"]) for c in checks)
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
            if reprs[k] != "cfd":
                raise ValueError("VERIFY-CFDU: %s: a CFD u applies to repr cfd only, this check is repr %s"
                                 % (k, reprs[k]))
    rows = [judge(c, measurements.get(c["req_id"]), (cfd_u or {}).get(c["req_id"]), evidence)
            for c in checks]
    if set(r["req_id"] for r in rows) != seen or seen != set(r["id"] for r in requirements_doc["rows"]):
        raise ValueError("VER-COVER: the verdict, check and locked-row id sets differ")
    doc = {"schema": "cad-verdict/1", "study_id": checks_doc["study_id"],
           "requirements_lock": requirements_doc["lock_sha"], "checks_sha": common.sha256_of(checks_doc),
           "eval_key": eval_key, "design_verdict": design_verdict(rows), "verdicts": rows}
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


def _refused(fn, prefix) -> str:
    """Assert fn raised ValueError whose message starts with prefix; return the message."""
    try:
        fn()
    except ValueError as e:
        msg = str(e)
        assert msg.startswith(prefix), "expected %r, got %r" % (prefix, msg)
        return msg
    raise AssertionError("expected ValueError %r, nothing raised" % (prefix,))


def _no_tolerance_option(path) -> list:
    """The AST scan of docs/16a §G AMG-8 outside selftest code: no option parser, no environment read, no flag-like
    string naming a tolerance, no module-level name holding one. Returns the offending findings (empty is clean)."""
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    flag = re.compile("(?i)(^|[ =])-{1,2}[a-z_-]*tol")
    bad = []
    for top in tree.body:
        if isinstance(top, ast.FunctionDef) and top.name.startswith(("selftest", "_selftest", "_no_tolerance")):
            continue
        if isinstance(top, ast.Assign):
            bad += ["name %s" % t.id for t in top.targets if isinstance(t, ast.Name) and "TOL" in t.id.upper()]
        for node in ast.walk(top):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] + ([node.module] if isinstance(node, ast.ImportFrom) else [])
                bad += ["import %s" % n for n in names if n and n.split(".")[0] in ("argparse", "getopt",
                                                                                   "optparse", "click")]
            if isinstance(node, ast.Attribute) and node.attr in ("environ", "getenv", "add_argument"):
                bad.append("attribute %s" % node.attr)
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and flag.search(node.value):
                bad.append("string %r" % node.value[:40])
    return bad


def _selftest_cover(C, R, M, M_b, cfd, ev, K) -> None:
    """The six AMG-8 cases of docs/16a §G and the binding arms: VER-COVER, evidence, repr, no tolerance option."""
    snap = (common.canonical_json(C), common.canonical_json(R))

    def drop(doc, rid):
        out = copy.deepcopy(doc)
        out["checks"] = [c for c in out["checks"] if c["req_id"] != rid]
        return out

    def without(meas, rid):
        return dict((k, v) for k, v in meas.items() if k != rid)

    c1 = drop(C, "REQ-003")
    _refused(lambda: cover(c1, R), "VER-COVER: locked row REQ-003 has no check")
    _refused(lambda: evaluate(c1, R, without(M, "REQ-003"), K, None, ev),
             "VER-COVER: locked row REQ-003 has no check")
    print("[ok] VER-COVER: a missing check (REQ-003 dropped from checks.json) is refused, never a silent subset")

    c2 = copy.deepcopy(C)
    c2["checks"].append(dict(copy.deepcopy([c for c in C["checks"] if c["req_id"] == "REQ-003"][0]),
                             req_id="REQ-099"))
    _refused(lambda: cover(c2, R), "VER-COVER: extra check REQ-099 has no locked row")
    _refused(lambda: evaluate(c2, R, M, K, None, ev), "VER-COVER: extra check REQ-099 has no locked row")
    print("[ok] VER-COVER: an extra check (REQ-099, a copy of REQ-003 with no locked row) is refused")

    r3 = copy.deepcopy(R)
    [r for r in r3["rows"] if r["id"] == "REQ-002"][0]["id"] = "REQ-001"
    r3["lock_sha"] = reqs.lock_sha_of(r3)
    assert reqs.lock_ok(r3) and schema.errors(r3, "cad-requirements/1") == []
    c3 = dict(drop(C, "REQ-002"), requirements_lock=r3["lock_sha"])
    _refused(lambda: cover(c3, r3), "VER-COVER: check REQ-001 comes from 2 locked rows")
    _refused(lambda: evaluate(c3, r3, without(M, "REQ-002"), K, None, ev),
             "VER-COVER: check REQ-001 comes from 2 locked rows")
    print("[ok] VER-COVER: a check from two rows (REQ-002 re-id'd REQ-001, relocked, its check dropped) is refused")

    bad_lock = copy.deepcopy(R)
    bad_lock["rows"][0]["value"] = 0.061
    _refused(lambda: cover(C, bad_lock), "GATE-LOCK")
    _refused(lambda: cover(dict(C, requirements_lock="b" * 64), R), "VER-COVER: the checks carry requirements_lock")
    _refused(lambda: cover(dict(C, declaration_sha="c" * 64), R), "VER-COVER: the checks carry declaration_sha")
    dup = copy.deepcopy(C)
    dup["checks"].append(copy.deepcopy([c for c in C["checks"] if c["req_id"] == "REQ-003"][0]))
    _refused(lambda: cover(dup, R), "VER-COVER: check REQ-003 appears twice")
    hard5 = copy.deepcopy(C)
    [c for c in hard5["checks"] if c["req_id"] == "REQ-005"][0]["hardness"] = "hard"
    _refused(lambda: cover(hard5, R), "VER-COVER: check REQ-005 hardness")
    op4 = copy.deepcopy(C)
    c4 = [c for c in op4["checks"] if c["req_id"] == "REQ-004"][0]
    c4["op"], c4["lo"], c4["hi"] = "<=", None, c4["lo"]
    _refused(lambda: cover(op4, R), "VER-COVER: check REQ-004 op")
    _refused(lambda: evaluate(C, dict((k, v) for k, v in R.items() if k != "rows"), M_b, K, cfd, ev),
             "VERIFY-REQS: ")
    doc = evaluate(C, R, M_b, K, cfd, ev)
    assert doc["requirements_lock"] == R["lock_sha"] == C["requirements_lock"]
    assert doc["checks_sha"] == common.sha256_of(C) and schema.errors(doc, "cad-verdict/1") == []
    assert set(r["req_id"] for r in doc["verdicts"]) == set(c["req_id"] for c in C["checks"]) \
        == set(r["id"] for r in R["rows"])
    tol3 = copy.deepcopy(C)
    [c for c in tol3["checks"] if c["req_id"] == "REQ-003"][0]["tol"] = 1e-9
    assert evaluate(tol3, R, M_b, K, cfd, ev)["checks_sha"] != doc["checks_sha"]
    print("[ok] binding: an unsealed set GATE-LOCK; checks of another lock or declaration, a duplicate check and a "
          "check whose hardness or op is not its row's VER-COVER; a non-document VERIFY-REQS; the verdict carries "
          "the lock and a checks_sha that moves with a tol")

    row1 = [r for r in R["rows"] if r["id"] == "REQ-001"][0]
    chk1 = [c for c in C["checks"] if c["req_id"] == "REQ-001"][0]
    assert row1["value"] == chk1["lo"] == chk1["hi"]           # the card row carries the target as its value
    card = dict(_rec("diameter_at_plane", 0.06, 1e-9), ears=row1["ears"])
    msgs = (_refused(lambda: judge(chk1, copy.deepcopy(row1)), "VERIFY-RECORD: REQ-001: cad-measure/1: "),
            _refused(lambda: judge(chk1, row1["ears"]), "VERIFY-RECORD: REQ-001: record is str"),
            _refused(lambda: judge(chk1, card), "VERIFY-RECORD: REQ-001: cad-measure/1: "),
            _refused(lambda: evaluate(C, R, dict(M, **{"REQ-001": copy.deepcopy(row1)}), K, cfd, ev),
                     "VERIFY-RECORD: REQ-001: cad-measure/1: "))
    _refused(lambda: evaluate(C, R, copy.deepcopy(R), K, cfd, ev), "VERIFY-STRAY: ")
    print("[ok] evidence: the card row, its EARS sentence and a record carrying ears passed as REQ-001's measurement "
          "are refused by cad-measure/1 (%s); the locked document as the measurements is VERIFY-STRAY"
          % (msgs[0].split(": ", 3)[3],))

    chk_stl = {"req_id": "REQ-091", "primitive": "extent_along_axis",
               "args": {"feature": None, "where": ["body"], "Re": None, "level": None},
               "op": "<=", "lo": None, "hi": 0.25, "tol": 0.0, "u": 2.0 ** -12, "hardness": "hard", "repr": "stl"}
    rec_brep = _rec("extent_along_axis", 0.25 - 2.0 ** -14, 2.0 ** -30)     # the BRep's u_meas rides the record
    a = judge(chk_stl, rec_brep)
    assert (a["verdict"], a["reason_id"], a["u"]) == ("not_evaluable", "NE-UNCERTAIN", 2.0 ** -12), a
    b = judge(dict(chk_stl, repr="brep", u=rec_brep["u_meas"]), rec_brep)
    assert (b["verdict"], b["u"]) == ("pass", 2.0 ** -30), b
    c = judge(chk_stl, _rec("extent_along_axis", 0.25 - 2.0 ** -11, 2.0 ** -30))
    assert (c["verdict"], c["u"]) == ("pass", 2.0 ** -12), c
    d = judge(chk_stl, _rec("extent_along_axis", 0.25 - 2.0 ** -11, 2.0 ** -10))
    assert (d["verdict"], d["reason_id"], d["u"]) == ("not_evaluable", "NE-UNCERTAIN", 2.0 ** -10), d
    e = judge(dict(chk_stl, u=None), rec_brep)
    assert (e["verdict"], e["reason_id"], e["u"]) == ("not_evaluable", "NE-UNCERTAIN", None), e
    _refused(lambda: judge(chk_stl, rec_brep, {"gci_fine": 2.0 ** -30, "repeat_band": None}),
             "VERIFY-CFDU: REQ-091: a CFD u applies to repr cfd only, this check is repr stl")
    _refused(lambda: evaluate(C, R, M_b, K, {"REQ-003": {"gci_fine": 0.0}}, ev),
             "VERIFY-CFDU: REQ-003: a CFD u applies to repr cfd only, this check is repr brep")
    _refused(lambda: judge(dict(chk_stl, repr="step"), rec_brep), "VERIFY-CHECK: REQ-091: repr 'step'")
    _refused(lambda: judge(dict((k, v) for k, v in chk_stl.items() if k != "repr"), rec_brep),
             "VERIFY-CHECK: REQ-091: repr None")
    print("[ok] repr: an STL check with its own u 2^-12 is NE-UNCERTAIN at m = b - 2^-14 where the BRep u_meas "
          "2^-30 would pass it; it passes at b - 2^-11, a larger record u_meas lifts u, u null stays NE-UNCERTAIN; "
          "a CFD u on a non-cfd check is VERIFY-CFDU; a missing or unknown repr is VERIFY-CHECK")

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    ph = subprocess.run([sys.executable, os.path.abspath(__file__), "--help"],
                        capture_output=True, text=True, encoding="utf-8", env=env, timeout=120)
    assert ph.returncode == 0 and "judge CHECKS_JSON REQUIREMENTS_DIR" in ph.stdout, (ph.returncode, ph.stdout)
    assert not re.search("(?i)tol", ph.stdout), ph.stdout
    assert _no_tolerance_option(os.path.abspath(__file__)) == []
    with tempfile.TemporaryDirectory() as td:
        planted = os.path.join(td, "planted.py")
        with open(planted, "w", encoding="utf-8") as f:
            f.write("import argparse" + chr(10) + "TOL_M = 1e-6" + chr(10)
                    + "argparse.ArgumentParser().add_argument('--tol')" + chr(10))
        found = _no_tolerance_option(planted)
        assert found == ["import argparse", "name TOL_M", "attribute add_argument", "string '--tol'"], found
        out_t = os.path.join(td, "out.json")
        pt = subprocess.run([sys.executable, os.path.abspath(__file__), "judge", "c.json", td, "m.json",
                             out_t, "a" * 64, "--tol", "1"],
                            capture_output=True, text=True, encoding="utf-8", env=env, timeout=120)
        assert pt.returncode == 2 and "usage:" in pt.stderr and not os.path.exists(out_t), pt.returncode
    assert (common.canonical_json(C), common.canonical_json(R)) == snap
    print("[ok] no tolerance option: --help exits 0 naming no tol, the AST scan of verify.py finds nothing while it "
          "finds all 4 planted in a probe file, and judge ... --tol 1 is a usage error writing nothing")


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
    doc = {"schema": "cad-verdict/1", "study_id": "verify_cases", "requirements_lock": "0" * 64,
           "checks_sha": "0" * 64, "eval_key": "0" * 64, "design_verdict": design_verdict(rows), "verdicts": rows}
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
    R = common.read_json(GOLDEN_V1)["requirements"]
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

    doc_a = evaluate(C, R, M, K, None, ev)
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
    doc_b = evaluate(C, R, M_b, K, None, ev)
    assert doc_b["design_verdict"] == "not_evaluable"
    mach = by_id(doc_b, "SYS-MACH")
    assert mach["verdict"] == "not_evaluable" and mach["reason_id"] == "NE-UNCERTAIN"
    assert mach["m"] == 0.1 and mach["u"] is None

    cfd = {"SYS-MACH": {"gci_fine": 0.01, "repeat_band": 0.02}}
    doc_c = evaluate(C, R, M_b, K, cfd, ev)
    assert doc_c["design_verdict"] == "feasible"
    mach = by_id(doc_c, "SYS-MACH")
    assert mach["verdict"] == "pass" and mach["u"] == 0.02
    assert tally(doc_c) == {"n_hard_pass": 10, "n_hard_fail": 0, "n_hard_ne": 0,
                            "objective": 0.07, "objective_u": 1e-9}

    M_d = dict(M_b)
    M_d["REQ-004"] = _rec("meridian_min_wall", 0.0018759, 1e-8)
    doc_d = evaluate(C, R, M_d, K, cfd, ev)
    assert doc_d["design_verdict"] == "infeasible"
    assert by_id(doc_d, "REQ-004")["verdict"] == "fail"
    assert tally(doc_d) == {"n_hard_pass": 9, "n_hard_fail": 1, "n_hard_ne": 0,
                            "objective": 0.07, "objective_u": 1e-9}

    M_e = dict(M_b)
    M_e["REQ-005"] = _rec("slope_max", 0.7, 5e-7, unit="rad")
    doc_e = evaluate(C, R, M_e, K, cfd, ev)
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
    _raises(lambda: evaluate(C, R, {"REQ-099": None}, K))
    _raises(lambda: evaluate(C, R, dict(M_b), K, {"SYS-MACH": {"gci_fine": -1.0}}))
    _raises(lambda: evaluate(C, R, dict(M_b), "abc"))
    bad_doc = copy.deepcopy(C)
    bad_doc["schema"] = "cad-checks/2"
    _raises(lambda: evaluate(bad_doc, R, dict(M_b), K))
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
        req_d = os.path.join(td, "study")
        os.makedirs(req_d)
        reqs.write_locked(req_d, R)
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        cmd = [sys.executable, os.path.abspath(__file__), "judge",
               checks_p, req_d, meas_p, out_p, "a" * 64, cfd_p]
        p1 = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env, timeout=120)
        with open(out_p, "rb") as f:
            b1 = f.read()
        p2 = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env, timeout=120)
        with open(out_p, "rb") as f:
            b2 = f.read()
        assert p1.returncode == 0 and p2.returncode == 0, (p1.returncode, p1.stderr, p2.stderr)
        assert b1 == b2
        ref = (common.canonical_json(evaluate(C, R, M_b, "a" * 64, cfd,
                                              {"path": "meas.json", "sha": common.sha256_file(meas_p)}))
               + chr(10)).encode("utf-8")
        assert b1 == ref
        assert '"design_verdict":"feasible"' in p1.stdout

        out_ne = os.path.join(td, "out_ne.json")
        p3 = subprocess.run([sys.executable, os.path.abspath(__file__), "judge",
                             checks_p, req_d, meas_p, out_ne, "a" * 64],
                            capture_output=True, text=True, encoding="utf-8", env=env, timeout=120)
        assert p3.returncode == 1, p3.returncode
        with open(out_ne, "rb") as f:
            d_ne = json.loads(f.read().decode("utf-8"))
        assert d_ne["design_verdict"] == "not_evaluable"

        out_bad = os.path.join(td, "out_bad.json")
        p4 = subprocess.run([sys.executable, os.path.abspath(__file__), "judge",
                             checks_p, req_d, meas_p, out_bad, "abc"],
                            capture_output=True, text=True, encoding="utf-8", env=env, timeout=120)
        assert p4.returncode == 2 and not os.path.exists(out_bad)
        p5 = subprocess.run([sys.executable, os.path.abspath(__file__), "judge", checks_p],
                            capture_output=True, text=True, encoding="utf-8", env=env, timeout=120)
        assert p5.returncode == 2
        assert "cadquery" not in sys.modules and "OCP" not in sys.modules and "measure" not in sys.modules
        print("[ok] CLI: two runs byte-identical and equal to evaluate(); exits 0 feasible, "
              "1 not_evaluable, 2 bad key, 2 usage; no CAD module loaded")
    _selftest_cover(C, R, M, M_b, cfd, ev, K)
    print("SELFTEST PASS")


def main(argv) -> int:
    """The CLI of docs/16 §I CAD-08: --selftest, --help, judge; no option sets a tolerance (docs/16a §G AMG-8)."""
    if argv == ["--selftest"]:
        try:
            selftest()
        except Exception:
            traceback.print_exc()
            return 1
        return 0
    if argv in (["--help"], ["-h"]):
        print(USAGE)
        return 0
    try:
        if len(argv) in (6, 7) and argv[0] == "judge":
            checks_doc = common.read_json(argv[1])
            requirements_doc = reqs.read_locked(argv[2])
            measurements = common.read_json(argv[3])
            evidence = {"path": os.path.basename(argv[3]), "sha": common.sha256_file(argv[3])}
            cfd_u = common.read_json(argv[6]) if len(argv) == 7 else None
            doc = evaluate(checks_doc, requirements_doc, measurements, argv[5], cfd_u, evidence)
            common.atomic_write(argv[4], common.canonical_json(doc) + chr(10))
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
