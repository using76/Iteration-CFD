#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""tg4.py - the TG4 turbulent nozzle study (docs/16 §E.2, §E.4, §E.7, §H.5 item 5 and TG4, §I
CAD-30), the turbulent twin of studies/g4_nozzle: the locked brief's contraction through the real
"cfd_turb" evaluator on the REDUCED path - an evaluation outside the walk that lands in the same
cache a later walk reuses with zero work - plus the judge, the report and the cad-tg4/1 record.
The full run (the walk's own prefilter of 4 x 256 candidates, at most 24 L1 evaluations, the L2
confirmation at 6000 iterations and the deep replay) is the supervisor's and stays deferred until
it has run; and TG0 must PASS first: no turbulent nozzle number counts while TG0 is OPEN.

Usage:
  python tg4.py --selftest
  python tg4.py init STUDY_DIR [--registry R]
  python tg4.py evaluate STUDY_DIR NAME LEVEL [PARAMS_JSON] [--registry R]
  python tg4.py record STUDY_DIR OUT_JSON [--registry R]
  python tg4.py report STUDY_DIR OUT_MD
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAD = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, CAD)
import common  # noqa: E402
import gate  # noqa: E402
import loop  # noqa: E402
import optimise_cad  # noqa: E402
import reqs  # noqa: E402
import verify  # noqa: E402

VERSION = "cad-tg4/1"
STUDY = "tg4_nozzle"
TG4_IDS = ("TG4-TG0", "TG4-START", "TG4-NOT-RUN", "TG4-EVALS", "TG4-CONFIRM", "TG4-REGRESS", "TG4-REPLAY",
           "TG4-CFDU", "TG4-KCFD")
MAX_EVALS = 24                       # docs/16 §E.7, = gates.json max_evals (assert it)
REDUCED_JSONL = "reduced.jsonl"
REFUSAL_IDS = ("TG4-NAME", "TG4-PARAMS")
TG0_SOURCE = "tools/cad/cases/pipe_turb/tg0_record.json"
CFD_U_SOURCE = "tools/cad/cases/nozzle_turb/tg123_record.json"
K_START_PLAN = 4.97e-6               # docs/16 §H.5 TG4: the start's a priori K_max
NAME_RE = re.compile("^[a-z0-9_]{1,40}$")
USAGE = ("usage: python tg4.py --selftest" + chr(10)
         + "       python tg4.py init STUDY_DIR [--registry R]" + chr(10)
         + "       python tg4.py evaluate STUDY_DIR NAME LEVEL [PARAMS_JSON] [--registry R]" + chr(10)
         + "       python tg4.py record STUDY_DIR OUT_JSON [--registry R]" + chr(10)
         + "       python tg4.py report STUDY_DIR OUT_MD")
REDUCED_KEYS = ("schema", "name", "level", "eval_key", "params_sha", "params")
TALLY_KEYS = ("n_hard_pass", "n_hard_fail", "n_hard_ne", "objective")
K_REQ = "REQ-005"                    # the a priori k_max_apriori row of this study's requirements


def init(study_dir, registry_path) -> dict:
    """loop.init of this study directory's start.json into STUDY_DIR (idempotent, one genesis row)."""
    start = common.read_json(os.path.join(HERE, "start.json"))
    return loop.init(study_dir, HERE, start, registry_path)


def _reduced_path(study_dir) -> str:
    return os.path.join(study_dir, REDUCED_JSONL)


def evaluate(study_dir, name, level, params=None, registry_path=gate.REGISTRY,
             eval_kwargs=None) -> dict:
    """One REDUCED evaluation of docs/16 §I CAD-30: the study's locked inputs through
    loop.evaluate_one with the study's "cfd_turb" evaluator, appended to reduced.jsonl, into the
    same cache a later walk reuses with zero work. eval_kwargs carries the selftest's fakes only;
    the live path (no kwargs) runs the GPU and is the supervisor's, never this repo's selftest."""
    _doc, decl, template_sha, declaration_sha, checks, study, _gates, gates_lock = \
        loop._pre_walk(study_dir, registry_path)
    if params is None:
        if name != "start":
            raise ValueError("TG4-NAME: the study's start params evaluate under the name 'start',"
                             " not %r" % (name,))
        params = study["start"]["params"]
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise ValueError("TG4-NAME: name %r is not ^[a-z0-9_]{1,40}$" % (name,))
    for row in common.read_jsonl(_reduced_path(study_dir)):
        if isinstance(row, dict) and row.get("name") == name:
            raise ValueError("TG4-NAME: name %r is already in %s" % (name, REDUCED_JSONL))
    names = sorted(p["name"] for p in decl["params"])
    if not isinstance(params, dict) or sorted(params.keys()) != names:
        raise ValueError("TG4-PARAMS: the parameters %s are not exactly the declared %s"
                         % (sorted(params.keys()) if isinstance(params, dict) else params, names))
    verdict, ev, ek = loop.evaluate_one(study_dir, params, level, checks, _doc, template_sha,
                                        declaration_sha, gates_lock, True, "cfd_turb", eval_kwargs)
    common.jsonl_append(_reduced_path(study_dir),
                        {"schema": "cad-tg4-reduced/1", "name": name, "level": level, "eval_key": ek,
                         "params_sha": optimise_cad.params_sha(params), "params": params})
    return {"name": name, "eval_key": ek, "design_verdict": verdict["design_verdict"],
            "stage_reached": ev["stage_reached"], "solve_class": ev["solve_class"]}


def judge_tg4(start, it_rows, dec_rows, replay_ok, cfd_u_status, tg0_verdict, k_cfd_ok) -> dict:
    """The TG4 verdict (docs/16 §H.5 TG4): reasons in TG4_IDS order, PASS iff none. TG4-TG0 fires
    first while the TG0 record's verdict is not PASS. start is the start row (None or a dict
    carrying n_hard_fail); it_rows and dec_rows are the walk's cad-iteration/1 and cad-decision/1
    rows; replay_ok None means the deep replay has not run; k_cfd_ok None means no confirmation
    has run, so the confirmed design's CFD K re-check is still owed."""
    reasons = []
    if tg0_verdict != "PASS":
        reasons.append("TG4-TG0")
    if start is None or start.get("n_hard_fail") == 0:
        reasons.append("TG4-START")
    confirmed = [r for r in dec_rows if isinstance(r, dict)
                 and r.get("decision") in ("confirm_pass", "confirm_fail")]
    stopped = [r for r in dec_rows if isinstance(r, dict)
               and r.get("decision") in ("stop", "abstain")]
    if not confirmed and not stopped:
        reasons.append("TG4-NOT-RUN")
    evals = [r for r in it_rows if isinstance(r, dict) and r.get("kind") == "eval"
             and r.get("origin") != "confirmation"]
    if len(evals) > MAX_EVALS or not any(r.get("design_verdict") == "feasible" for r in evals):
        reasons.append("TG4-EVALS")
    if not (dec_rows and isinstance(dec_rows[-1], dict)
            and dec_rows[-1].get("decision") == "confirm_pass"):
        reasons.append("TG4-CONFIRM")
    if any(isinstance(r, dict) and r.get("decision") == "promote"
           and r.get("regressed") for r in dec_rows):
        reasons.append("TG4-REGRESS")
    if replay_ok is not True:
        reasons.append("TG4-REPLAY")
    if cfd_u_status != "CLOSED":
        reasons.append("TG4-CFDU")
    if k_cfd_ok is not True:
        reasons.append("TG4-KCFD")
    assert reasons == [i for i in TG4_IDS if i in reasons]
    return {"verdict": "PASS" if not reasons else "OPEN", "reasons": reasons}


TABLE_HEAD = ("| n / 번호 | origin / 출처 | level / 격자 | stage / 단계 | solve / 해석"
              " | verdict / 판정 | hard pass-fail-NE / 필수 통과-실패-미판정"
              " | objective m / 목적함수 m | cache hit / 캐시 | eval_key |")


def report(iterations_path) -> str:
    """The bilingual markdown report from ONLY iterations.jsonl (common.read_jsonl): the genesis
    ids, one table line per eval row in file order, then the bilingual summary. Deterministic."""
    rows = common.read_jsonl(iterations_path)
    genesis = rows[0] if rows and isinstance(rows[0], dict) and rows[0].get("kind") == "genesis" else {}
    evals = [r for r in rows if isinstance(r, dict) and r.get("kind") == "eval"]
    L = []
    L.append("# TG4 turbulent nozzle study / TG4 난류 노즐 연구")
    L.append("")
    L.append("- study_id / 연구: %s" % (genesis.get("study_id"),))
    L.append("- lock_sha / 요구 잠금: %s" % (genesis.get("lock_sha"),))
    L.append("- template_sha / 템플릿: %s" % (genesis.get("template_sha"),))
    L.append("- declaration_sha / 선언: %s" % (genesis.get("declaration_sha"),))
    L.append("- gates_lock / 게이트 잠금: %s" % (genesis.get("gates_lock"),))
    L.append("")
    L.append(TABLE_HEAD)
    L.append("| " + " | ".join(["---"] * 10) + " |")
    for r in evals:
        cells = [str(r.get("n")), str(r.get("origin")), str(r.get("level")),
                 str(r.get("stage_reached")), str(r.get("solve_class")),
                 str(r.get("design_verdict")),
                 "%d-%d-%d" % (r.get("n_hard_pass", 0), r.get("n_hard_fail", 0),
                               r.get("n_hard_ne", 0)),
                 repr(r.get("objective")), str(r.get("cache_hit")),
                 str(r.get("eval_key"))[:12]]
        L.append("| " + " | ".join(cells) + " |")
    L.append("")
    by_level = {}
    for r in evals:
        by_level[r.get("level")] = by_level.get(r.get("level"), 0) + 1
    feasible = [r for r in evals if r.get("design_verdict") == "feasible"]
    l1 = [r for r in feasible if r.get("level") == "L1"]
    best = min(l1, key=lambda r: r.get("objective")) if l1 else None
    start_rows = [r for r in evals if r.get("origin") == "start"]
    confirms = [r for r in evals if r.get("origin") == "confirmation"]
    L.append("## Summary / 요약")
    L.append("")
    L.append("- eval rows by level / 격자별 평가 행: "
             + ", ".join("%s %d" % (lv, n) for lv, n in sorted(by_level.items())) + ";"
             " feasible / 실현가능 %d" % (len(feasible),))
    L.append("- best feasible objective at L1 / L1 최고 실현가능 목적함수: "
             + ("none / 없음" if best is None else "%r (n %d)" % (best.get("objective"), best.get("n"))))
    for r in start_rows:
        L.append("- start / 시작: verdict %s, n_hard_fail %d, fails a hard row / 필수 행 실패: %s"
                 % (r.get("design_verdict"), r.get("n_hard_fail", 0),
                    "yes / 예" if r.get("n_hard_fail", 0) > 0 else "no / 아니오"))
    L.append("- confirmation rows / 확인 행: "
             + ("none / 없음" if not confirms else
                ", ".join("n %d %s %s" % (r.get("n"), r.get("level"), r.get("design_verdict"))
                          for r in confirms)))
    return chr(10).join(L) + chr(10)


TG4_DOC_KEYS = ("version", "reduced", "study_id", "lock_sha", "binary", "tg0", "cfd_u", "start",
                "precondition", "reduced_evals", "k_recheck", "loop", "replay", "tg4", "deferred")
ENTRY_FIELDS = ("stage_reached", "solve_class", "reason_id", "message", "cells", "mesh", "solve",
                "wall_s", "gpu_shared", "k_recheck", "reported")
VERDICT_FIELDS = ("req_id", "verdict", "reason_id", "m", "u")
DEFERRED = ("The walk's CAD prefilter of 4 x 256 candidates has not run.",
            "At most 24 L1 evaluations in the walk (docs/16 §E.7) have not run.",
            "The L2 confirmation at 6000 iterations has not run.",
            "The deep replay re-deriving every row has not run.",
            "TG0 must PASS first: no turbulent nozzle number counts while TG0 is OPEN (docs/16 §H.5).")


def _reduced_entry(row, study_dir) -> dict:
    """One reduced.jsonl row joined with its cache entry's stage.json and verdict.json; the stage's
    k_recheck dict and reported report travel with the entry."""
    entry = {"name": row["name"], "level": row["level"], "eval_key": row["eval_key"],
             "params": row["params"]}
    d = os.path.join(study_dir, "cache", row["eval_key"])
    st = common.read_json(os.path.join(d, "stage.json"))
    for k in ENTRY_FIELDS:
        if k == "gpu_shared":
            entry[k] = (st.get("gpu") or {}).get("shared")
        else:
            entry[k] = st.get(k)
    vd = common.read_json(os.path.join(d, "verdict.json"))
    t = verify.tally(vd)
    entry["design_verdict"] = vd["design_verdict"]
    for k in TALLY_KEYS:
        entry[k] = t[k]
    entry["verdicts"] = [dict((k, r[k]) for k in VERDICT_FIELDS) for r in vd["verdicts"]]
    return entry


def _m_of(entry, req_id) -> float:
    """The entry's verdict m for one req id, or None."""
    for r in entry.get("verdicts", []):
        if r.get("req_id") == req_id:
            return r.get("m")
    return None


def record(study_dir, registry_path=gate.REGISTRY) -> dict:
    """The cad-tg4/1 record: the reduced evaluations with their k_recheck and reported, the start
    row and its a priori K precondition, the per-eval CFD K re-check, the loop status, the TG0 and
    CFD u states and the TG4 judge; replay NOT_RUN and the full run's steps named in deferred. No
    absolute path anywhere in the canonical form."""
    doc = reqs.read_locked(study_dir)
    it_rows = common.read_jsonl(os.path.join(study_dir, loop.IT_JSON))
    dec_rows = common.read_jsonl(os.path.join(study_dir, loop.DEC_JSON))
    reduced_rows = common.read_jsonl(_reduced_path(study_dir))
    bg = common.read_json(os.path.join(CAD, "bin_gpu.json"))
    entry = bg["binaries"]["ofgpu-lowmach"]
    binary = {"name": "ofgpu-lowmach", "path": entry["path"], "sha256": entry["sha256"]}
    tg0_path = os.path.join(CAD, "cases", "pipe_turb", "tg0_record.json")
    tg0_rec = common.read_json(tg0_path)
    tg0 = {"source": TG0_SOURCE, "sha256": common.sha256_file(tg0_path),
           "verdict": tg0_rec["tg0"]["verdict"], "reasons": tg0_rec["tg0"]["reasons"]}
    cu_path = os.path.join(CAD, "cases", "nozzle_turb", "tg123_record.json")
    cu_rec = common.read_json(cu_path)
    cfd_u = {"source": CFD_U_SOURCE, "sha256": common.sha256_file(cu_path),
             "status": "CLOSED" if cu_rec["tg3"]["verdict"] == "PASS" else "OPEN",
             "fields_bit_identical": cu_rec["tg_repeat"]["fields_bit_identical"],
             "gci_fine_known": False}
    reduced_evals = [_reduced_entry(r, study_dir) for r in reduced_rows]
    walk_start = [r for r in it_rows if isinstance(r, dict) and r.get("origin") == "start"]
    red_start = [e for e in reduced_evals if e["name"] == "start"]
    if walk_start:
        start = dict(walk_start[-1], source="walk")
    elif red_start:
        start = dict(red_start[-1], source="reduced")
    else:
        start = None
    nhf = None if start is None else start.get("n_hard_fail", 0)
    fails, k_start = [], None
    if start is not None:
        for r in start.get("verdicts", []):
            if r.get("verdict") == "fail":
                fails.append(r["req_id"])
            if r.get("req_id") == K_REQ:
                k_start = r.get("m")
    agrees = k_start is not None and abs(k_start / K_START_PLAN - 1.0) <= 1e-3
    precondition = {"start_n_hard_fail": nhf, "start_failing_rows": fails, "K_start": k_start,
                    "K_start_plan": K_START_PLAN, "K_start_agrees": agrees,
                    "met": nhf is not None and nhf >= 1}
    k_recheck = [{"name": e["name"], "K_apriori": _m_of(e, K_REQ),
                  "K_case": (e.get("k_recheck") or {}).get("K_apriori_case"),
                  "K_cfd": (e.get("k_recheck") or {}).get("K_cfd"),
                  "pass": (e.get("k_recheck") or {}).get("pass")}
                 for e in reduced_evals]
    st = loop.status(study_dir, registry_path)
    st["n_iterations"] = len(it_rows)
    st["n_decisions"] = len(dec_rows)
    doc_out = {"version": VERSION, "reduced": True, "study_id": doc["study_id"],
               "lock_sha": doc["lock_sha"], "binary": binary, "tg0": tg0, "cfd_u": cfd_u,
               "start": start, "precondition": precondition, "reduced_evals": reduced_evals,
               "k_recheck": k_recheck, "loop": st,
               "replay": {"status": "NOT_RUN", "ok": None},
               "tg4": judge_tg4(start, it_rows, dec_rows, None, cfd_u["status"], tg0["verdict"],
                                None),
               "deferred": list(DEFERRED)}
    assert tuple(sorted(doc_out.keys())) == tuple(sorted(TG4_DOC_KEYS)), sorted(doc_out)
    text = common.canonical_json(doc_out)
    assert "C:" not in text and os.path.abspath(study_dir) not in text \
        and os.path.abspath(study_dir).replace(os.sep, "/") not in text, "an absolute path leaked"
    return doc_out


# ---- the CAD-30 selftest: T1..T5, one [ok] line each; fakes only, the GPU is never launched ----

def _fake_solve(case_dir, geom_dir, run_dir, iters):
    raise AssertionError("the fake solve must never be called")


def _expect_checks(checks) -> None:
    """The compiled checks of the committed requirements, exactly as docs/16 §H.5 fixed them."""
    def row(rid, prim, op, lo, hi, tol, u, hardness, repr_, re_):
        c = [x for x in checks["checks"] if x["req_id"] == rid]
        assert len(c) == 1, (rid, [x["req_id"] for x in checks["checks"]])
        c = c[0]
        assert (c["primitive"], c["op"], c["lo"], c["hi"], c["tol"], c["u"], c["hardness"],
                c["repr"], c["args"]["Re"]) == (prim, op, lo, hi, tol, u, hardness, repr_, re_), \
            (rid, c)
    row("REQ-001", "diameter_at_plane", "==", 0.3, 0.3, 5e-05, 1e-09, "hard", "brep", None)
    row("REQ-002", "diameter_at_plane", "==", 0.21209999999999998, 0.21209999999999998, 5e-05,
        1e-09, "hard", "brep", None)
    row("REQ-003", "separation_free", "is_true", None, None, 0.0, None, "hard", "cfd", None)
    row("REQ-004", "exit_nonuniformity", "<=", None, 0.01, 0.0, None, "hard", "cfd", None)
    row("REQ-005", "k_max_1d", "<=", None, 3e-06, 0.0, 3.0000000000000002e-15, "hard", "brep",
        630000.0)
    row("REQ-006", "meridian_min_wall", ">=", 0.003, None, 0.0, 1e-08, "hard", "brep", None)
    row("REQ-007", "extent_along_axis", "<=", None, None, 0.0, 1e-09, "objective", "brep", None)
    row("SYS-SOLID", "n_solids", "==", 1.0, 1.0, 0.0, 0.0, "hard", "brep", None)
    row("SYS-VALID", "valid", "is_true", None, None, 0.0, 0.0, "hard", "brep", None)
    row("SYS-WATERTIGHT", "watertight", "is_true", None, None, 0.0, 0.0, "hard", "stl", None)
    row("SYS-AXIS", "axis_x", "is_true", None, None, 0.0, 0.0, "hard", "brep", None)
    row("SYS-UNITS", "units_m", "is_true", None, None, 0.0, 0.0, "hard", "brep", None)
    row("SYS-MACH", "mach_max", "<=", None, 0.3, 0.0, None, "hard", "cfd", None)


def selftest():
    import shutil
    import tempfile
    import time
    import unicodedata
    t0 = time.time()
    gates, _gl = gate.load_gates()
    assert gates["max_evals"] == MAX_EVALS, (gates["max_evals"], MAX_EVALS)
    with tempfile.TemporaryDirectory(prefix="cad30-tg4-") as td:
        # (T1) the committed brief, proposal, report and requirements are exactly docs/16 §H.5's
        brief = common.read_json(os.path.join(HERE, "brief.json"))
        proposal = common.read_json(os.path.join(HERE, "proposal.json"))
        report_doc = common.read_json(os.path.join(HERE, "report.json"))
        text = brief["text"]
        assert unicodedata.normalize("NFC", text) == text, "the brief is not NFC"
        paras = text.split(chr(10) + chr(10))
        assert len(paras) == 2, paras
        assert any(chr(0xAC00) <= ch <= chr(0xD7A3) for ch in paras[1]), "the second paragraph"
        decl, tsha, dsha = reqs.load_template(reqs.NOZZLE_DIR)
        rep = reqs.check(proposal, brief, decl, tsha, dsha)
        assert rep["status"] == "ok" and rep["refusals"] == [], rep["refusals"]
        locked = reqs.lock(rep, "docs16-H5")
        committed = open(os.path.join(HERE, "requirements.json"), "rb").read()
        assert committed == common.canonical_bytes(locked) + chr(10).encode("utf-8"), \
            "requirements.json is not lock(report, docs16-H5)"
        committed_lock = open(os.path.join(HERE, "requirements.lock"), "rb").read()
        assert committed_lock == (locked["lock_sha"] + chr(10)).encode("utf-8")
        checks = reqs.compile_checks(locked, decl, dsha)
        _expect_checks(checks)
        print("[ok] study files: NFC brief with two paragraphs, check ok with no refusals,"
              " lock(report, docs16-H5) byte-equals requirements.json, the lock file is the sha,"
              " and the 13 compiled checks match the table exactly")

        # (T2) start.json is a CADOPT-DOC-accepted start; init is idempotent with one genesis row
        start_doc = common.read_json(os.path.join(HERE, "start.json"))
        pdoc, _row = optimise_cad.start(locked["study_id"], 0, decl, tsha, _gl, checks,
                                        start_doc["start"]["params"], start_doc["start"]["provenance"])
        vals = {v["name"]: (v["choice"] if v["choice"] is not None else v["real"])
                for v in pdoc["values"]}
        assert vals["t_wall"] == 0.004 and vals["law"] == "poly7" and vals["L_over_Di"] == 0.5, vals
        assert vals["D_i"] == 0.30 and vals["CR"] == 2.0 and vals["Lu_over_Di"] == 2.0, vals
        prov = {v["name"]: v["provenance"] for v in pdoc["values"]}
        assert prov["t_wall"] == "llm_choice" and prov["law"] == "llm_choice", prov
        sdir = os.path.join(td, "study")
        reg = os.path.join(td, "studies.jsonl")
        study = init(sdir, reg)
        assert study["evaluator"] == "cfd_turb" and study["repeat_band"] == 0.0, study
        it = common.read_jsonl(os.path.join(sdir, "iterations.jsonl"))
        assert len(it) == 1 and it[0]["kind"] == "genesis" and it[0]["study_id"] == STUDY, it
        def _tree(root):
            out = {}
            for dp, _d, fns in os.walk(root):
                for fn in fns:
                    p = os.path.join(dp, fn)
                    out[os.path.relpath(p, root).replace(os.sep, "/")] = common.sha256_file(p)
            return out
        before = _tree(td)
        init(sdir, reg)
        assert _tree(td) == before, "the second init changed bytes"
        print("[ok] start: optimise_cad.start accepts the poly7 D_i 0.30 start, init writes the"
              " genesis row with evaluator cfd_turb, and re-init changes no byte")

        # (T3) the report reads only iterations.jsonl and is deterministic
        it_rows = [it[0],
                   {"kind": "eval", "n": 0, "origin": "start", "level": "L1", "eval_key": "a" * 64,
                    "stage_reached": "checks", "solve_class": "not_run", "design_verdict":
                    "infeasible", "n_hard_pass": 5, "n_hard_fail": 1, "n_hard_ne": 2,
                    "objective": 0.106, "cache_hit": False},
                   {"kind": "eval", "n": 1, "origin": "walk", "level": "L1", "eval_key": "b" * 64,
                    "stage_reached": "judge", "solve_class": "steady", "design_verdict":
                    "feasible", "n_hard_pass": 8, "n_hard_fail": 0, "n_hard_ne": 0,
                    "objective": 0.0912, "cache_hit": False},
                   {"kind": "eval", "n": 2, "origin": "confirmation", "level": "L2",
                    "eval_key": "c" * 64, "stage_reached": "judge", "solve_class": "steady",
                    "design_verdict": "feasible", "n_hard_pass": 8, "n_hard_fail": 0,
                    "n_hard_ne": 0, "objective": 0.0913, "cache_hit": True}]
        pit = os.path.join(td, "it.jsonl")
        for r in it_rows:
            common.jsonl_append(pit, r)
        r1 = report(pit)
        r2 = report(pit)
        assert r1 == r2, "the report is not deterministic"
        assert "TG4 turbulent nozzle study" in r1 and "TG4 난류 노즐 연구" in r1
        assert r1.endswith(chr(10)) and chr(13) not in r1
        n_table = sum(1 for l in r1.splitlines() if l.startswith("| "))
        assert n_table == 5, n_table          # header + separator (starts "| ---") + 3 eval rows
        assert "0.0912" in r1
        solo = os.path.join(td, "solo")
        os.makedirs(solo)
        common.jsonl_append(os.path.join(solo, "iterations.jsonl"), it[0])
        assert "# TG4 turbulent nozzle study" in report(os.path.join(solo, "iterations.jsonl"))
        print("[ok] report: two calls byte-equal, both titles, 3 table lines, 0.0912 present,"
              " and a directory holding only iterations.jsonl reports fine")

        # (T4) the judge on planted states, reasons in TG4_IDS order
        def start_row(nhf):
            return {"kind": "eval", "n": 0, "origin": "start", "level": "L1",
                    "design_verdict": "not_evaluable", "n_hard_fail": nhf}
        def walk_row(n, dv):
            return {"kind": "eval", "n": n, "origin": "walk", "level": "L1",
                    "design_verdict": dv, "n_hard_fail": 0}
        def confirm_row(n, dv):
            return {"kind": "eval", "n": n, "origin": "confirmation", "level": "L2",
                    "design_verdict": dv, "n_hard_fail": 0}
        def dec(n, d, regressed=None):
            return {"n": n, "decision": d, "regressed": regressed or []}
        j = judge_tg4(None, [], [], True, "CLOSED", "OPEN", True)
        assert j["verdict"] == "OPEN" and j["reasons"][:2] == ["TG4-TG0", "TG4-START"], j
        good_it = [start_row(1), walk_row(1, "feasible"), confirm_row(2, "feasible")]
        good_dec = [dec(1, "promote"), dec(2, "confirm_pass")]
        j = judge_tg4(start_row(1), good_it, good_dec, True, "CLOSED", "PASS", True)
        assert j == {"verdict": "PASS", "reasons": []}, j
        j = judge_tg4(start_row(1), good_it, good_dec, True, "CLOSED", "OPEN", True)
        assert j["reasons"] == ["TG4-TG0"], j
        j = judge_tg4(start_row(1), good_it, good_dec, True, "CLOSED", "PASS", None)
        assert j["reasons"] == ["TG4-KCFD"], j
        j = judge_tg4(start_row(1), good_it, [dec(1, "promote", regressed=["REQ-004"]),
                                              dec(2, "confirm_pass")], True, "CLOSED", "PASS", True)
        assert j["reasons"] == ["TG4-REGRESS"], j
        many = [start_row(1)] + [walk_row(i, "feasible") for i in range(1, 26)]
        j = judge_tg4(start_row(1), many, [dec(26, "promote"), dec(27, "confirm_pass")],
                      True, "CLOSED", "PASS", True)
        assert j["reasons"] == ["TG4-EVALS"], j
        j = judge_tg4(start_row(1), good_it, [dec(1, "promote"), dec(2, "confirm_fail")],
                      True, "CLOSED", "PASS", True)
        assert j["reasons"] == ["TG4-CONFIRM"], j
        print("[ok] judge: start None with TG0 OPEN opens [TG4-TG0, TG4-START] first, the complete"
              " history PASSES, tg0 OPEN gives [TG4-TG0], k_cfd_ok None [TG4-KCFD], a regressed"
              " promote [TG4-REGRESS], 25 evals [TG4-EVALS], confirm_fail last [TG4-CONFIRM]")

        # (T5) the record of one reduced "start" evaluation on the evaluate_cfd E10 path (the fake
        # never solves): the start stops at checks on the a priori K row
        s5 = os.path.join(td, "study5")
        reg5 = os.path.join(td, "studies5.jsonl")
        init(s5, reg5)
        res = evaluate(s5, "start", "L1", None, reg5, eval_kwargs={"solve_fn": _fake_solve})
        assert res["name"] == "start" and res["stage_reached"] == "checks", res
        rec = record(s5, reg5)
        assert tuple(sorted(rec.keys())) == tuple(sorted(TG4_DOC_KEYS)), sorted(rec)
        assert rec["version"] == VERSION and rec["reduced"] is True and rec["study_id"] == STUDY
        assert rec["start"]["source"] == "reduced" and rec["start"]["name"] == "start"
        pc = rec["precondition"]
        assert pc["start_n_hard_fail"] == 1 and pc["start_failing_rows"] == ["REQ-005"], pc
        assert pc["K_start_agrees"] is True and pc["met"] is True, pc
        assert abs(pc["K_start"] / K_START_PLAN - 1.0) <= 1e-3, pc["K_start"]
        assert rec["tg4"]["verdict"] == "OPEN"
        rs = rec["tg4"]["reasons"]
        assert rs == ["TG4-TG0", "TG4-NOT-RUN", "TG4-EVALS", "TG4-CONFIRM", "TG4-REPLAY",
                      "TG4-CFDU", "TG4-KCFD"], rs
        assert rec["replay"] == {"status": "NOT_RUN", "ok": None}
        assert rec["tg0"]["verdict"] == "OPEN" and rec["tg0"]["reasons"] == common.read_json(
            os.path.join(CAD, "cases", "pipe_turb", "tg0_record.json"))["tg0"]["reasons"], rec["tg0"]
        assert rec["cfd_u"]["status"] == "OPEN" and rec["cfd_u"]["gci_fine_known"] is False
        assert rec["cfd_u"]["fields_bit_identical"] is True
        assert len(rec["reduced_evals"]) == 1 and rec["reduced_evals"][0]["name"] == "start"
        e0 = rec["reduced_evals"][0]
        assert e0["stage_reached"] == "checks" and e0["solve_class"] == "not_run"
        assert e0["k_recheck"]["K_lim"] == 3e-6 and e0["reported"] is None, e0["k_recheck"]
        kr = rec["k_recheck"]
        assert len(kr) == 1 and kr[0]["name"] == "start" and kr[0]["pass"] is False, kr
        assert abs(kr[0]["K_apriori"] / 4.966618287407307e-06 - 1.0) <= 1e-9, kr[0]
        assert kr[0]["K_case"] is None and kr[0]["K_cfd"] is None, kr[0]
        assert rec["binary"]["sha256"] == "50471caaa54125e0c2eee4fb34eebdfc2da44727d2b818a2b96bb4234c02103c"
        text = common.canonical_json(rec)
        assert "C:" not in text and td not in text, "an absolute path in the record"
        reds = common.read_jsonl(os.path.join(s5, REDUCED_JSONL))
        assert len(reds) == 1 and tuple(sorted(reds[0].keys())) == tuple(sorted(REDUCED_KEYS))
        print("[ok] record: cad-tg4/1 keys exact, start.source reduced, the a priori precondition"
              " met on REQ-005, tg4 OPEN from TG4-TG0 to TG4-KCFD without TG4-START, tg0 OPEN,"
              " replay NOT_RUN, and no absolute path anywhere")

    shutil.rmtree(td, ignore_errors=True)
    print("selftest wall %.1f s" % (time.time() - t0))
    print("SELFTEST PASS")
    return 0


def main(argv) -> int:
    """The CLI: each verb prints its result as one canonical JSON line (report writes its file and
    prints the path doc); a refusal prints "<ID>: <message>" to stderr and exits 1."""
    if argv == ["--selftest"]:
        try:
            return selftest()
        except Exception:
            import traceback
            traceback.print_exc()
            return 1
    if argv in (["--help"], ["-h"]):
        print(USAGE)
        return 0
    want = {"init": (1, 1), "evaluate": (3, 4), "record": (2, 2), "report": (2, 2)}
    if not argv or argv[0] not in want:
        print(USAGE, file=sys.stderr)
        return 2
    verb, rest = argv[0], argv[1:]
    pos, registry = [], gate.REGISTRY
    i = 0
    while i < len(rest):
        if rest[i] == "--registry" and i + 1 < len(rest):
            registry = rest[i + 1]
            i += 2
        else:
            pos.append(rest[i])
            i += 1
    lo, hi = want[verb]
    if not lo <= len(pos) <= hi:
        print(USAGE, file=sys.stderr)
        return 2
    try:
        if verb == "init":
            res = init(pos[0], registry)
        elif verb == "evaluate":
            params = None if len(pos) < 4 else common.read_json(pos[3])
            res = evaluate(pos[0], pos[1], pos[2], params, registry)
        elif verb == "record":
            res = record(pos[0], registry)
            common.write_json(pos[1], res)
        else:
            res = {"report": os.path.abspath(pos[1]),
                   "bytes": len(report(os.path.join(pos[0], "iterations.jsonl")))}
            with open(pos[1], "wb") as f:
                f.write(report(os.path.join(pos[0], "iterations.jsonl")).encode("utf-8"))
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1
    print(common.canonical_json(res))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
