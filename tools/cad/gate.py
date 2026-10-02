#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""gate.py - stage S11's promotion gate (docs/16 §E.6), the ladder and the stop (§E.7) and the lock
anchor (§E.8, docs/16a §B.1): a candidate replaces the stable design only by the hard-row arithmetic,
ties never promote, and every decision is a cad-decision/1 row carrying the requirement deltas.

The proposal/stable split with backups is an idea from ai-cad (Apache-2.0, ADR 0006), whose
`judge_proposal` promotes a tie; here ties never promote. Coverage equality, the re-hash of bound
evidence before and after judgement, the stale-base refusal (their 409) and the delta vocabulary are
ideas from Amagine3D (Apache-2.0, e608dc6: scene_contract.py `validate`, build_manifest.py
`semantic_evidence_errors`, server/model-parameters.ts, cad_compile.py `_write_repair_state`),
reimplemented from reading only; no code copied.

Usage:
  python gate.py --selftest
  python gate.py --help
"""

import copy
import json
import math
import os
import re
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common
import reqs
import schema
import verify

GATES = os.path.join(HERE, "gates.json")
GATES_LOCK = os.path.join(HERE, "gates.lock")
REGISTRY = os.path.join(HERE, "studies.jsonl")
SCENARIOS = os.path.join(HERE, "fixtures", "gate", "scenarios.json")
GOLDEN_V1 = os.path.join(HERE, "fixtures", "reqs", "golden", "v1_nominal.json")
GATES_SHA = "8c59f024d1241e16ce959c09487a79f229caa0ec805ab573794dd328e3cb103c"
REFUSAL_IDS = ("GATE-LOCK", "GATE-COVER", "GATE-EVIDENCE", "GATE-STALE", "GATE-DOC")
RULE_IDS = ("GATE-NE", "GATE-FIRST", "GATE-REGRESS", "GATE-MORE", "GATE-TIE", "GATE-OBJNE", "GATE-OBJ", "GATE-NOISE")
LADDER_IDS = ("GATE-MAXEVALS", "GATE-FEASIBLE", "GATE-LLMOFF", "GATE-STALL")
CONFIRM_ID = "GATE-CONFIRM"
DELTA_KEYS = ("resolved", "regressed", "new_fail", "remaining", "not_reevaluated")
GENESIS_KEYS = ("kind", "study_id", "lock_sha", "declaration_sha", "template_sha", "gates_lock")
REGISTRY_KEYS = ("study_id", "lock_sha", "declaration_sha", "template_sha", "gates_lock", "supersedes_study", "supersedes_lock")
CTX_KEYS = ("n", "after_iteration", "params_sha", "law", "sobol_index", "ei")
STATE_KEYS = ("n_evals", "since_promote", "llm_rejected", "llm_off", "stable_feasible", "best_ei", "band")
LOCK_LINE_RE = re.compile(r"^[0-9a-f]{64}  gates\.json$")
USAGE = ("usage: python gate.py --selftest" + chr(10)
         + "       python gate.py --help")


def _num(x) -> bool:
    """A finite non-bool number: the only kind that can carry repeat_band, best_ei or band."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def load_gates(gates_path=GATES, lock_path=GATES_LOCK) -> tuple:
    """The gates doc and the sha256 of its bytes, bound by gates.lock (docs/16 §E.6, §E.8).

    The file's bytes are read once, hashed through a stable-file snapshot and parsed from those
    same bytes - the file is never opened again after hashing: a file that changed while read,
    a missing or unreadable lock, a lock line outside sha256sum format, or a lock sha over other
    bytes is GATE-LOCK; a gates doc that is not JSON or outside cad-gates/1 is GATE-DOC."""
    try:
        with open(gates_path, "rb") as f:
            data = f.read()
    except OSError as e:
        raise ValueError("GATE-LOCK: gates file is unreadable: %s" % (e,))
    snap = common.stable_file_snapshot(gates_path)
    if snap["stable"] is not True:
        raise ValueError("GATE-LOCK: gates file is not a stable regular file: %s" % (gates_path,))
    sha = common.sha256_bytes(data)
    if snap["sha256"] != sha:
        raise ValueError("GATE-LOCK: gates file changed while read: %s" % (gates_path,))
    try:
        with open(lock_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.read().splitlines()]
    except OSError as e:
        raise ValueError("GATE-LOCK: gates lock is unreadable: %s" % (e,))
    lines = [ln for ln in lines if ln and not ln.startswith("#")]
    if len(lines) != 1 or not LOCK_LINE_RE.match(lines[0]):
        raise ValueError("GATE-LOCK: gates lock is not one sha256sum line naming gates.json")
    if lines[0].split("  ", 1)[0] != sha:
        raise ValueError("GATE-LOCK: gates lock %s does not stamp these gates bytes %s"
                         % (lines[0].split("  ", 1)[0], sha))
    try:
        doc = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise ValueError("GATE-DOC: gates file is not JSON: %s" % (e,))
    errs = schema.errors(doc, "cad-gates/1")
    if errs:
        raise ValueError("GATE-DOC: %s" % (errs[0],))
    return (doc, sha)


def lock_anchor(study_dir, gates_sha, registry_path=REGISTRY) -> dict:
    """The locked requirements doc, anchored outside the study directory (docs/16a §B.1 GATE-LOCK).

    Rewriting requirements.json together with requirements.lock fools reqs.read_locked, so the lock
    is compared with the anchors: the genesis row of the study's iterations.jsonl and the append-only
    registry row. Any disagreement, a missing or doubled genesis row, or a missing or doubled
    registry row is GATE-LOCK."""
    try:
        doc = reqs.read_locked(study_dir)
    except ValueError as e:
        raise ValueError("GATE-LOCK: %s" % (e,))
    rows = common.read_jsonl(os.path.join(study_dir, "iterations.jsonl"))
    if not rows:
        raise ValueError("GATE-LOCK: no genesis row in iterations.jsonl")
    first = rows[0]
    if not isinstance(first, dict) or tuple(sorted(first.keys())) != tuple(sorted(GENESIS_KEYS)) \
            or first.get("kind") != "genesis":
        raise ValueError("GATE-LOCK: iterations.jsonl row 0 is not a genesis row with the keys %s"
                         % (", ".join(GENESIS_KEYS),))
    for row in rows[1:]:
        if isinstance(row, dict) and row.get("kind") == "genesis":
            raise ValueError("GATE-LOCK: a second genesis row in iterations.jsonl")
    for field, want in (("study_id", doc["study_id"]), ("lock_sha", doc["lock_sha"]),
                        ("declaration_sha", doc["declaration_sha"]), ("template_sha", doc["template_sha"]),
                        ("gates_lock", gates_sha)):
        if first.get(field) != want:
            raise ValueError("GATE-LOCK: the genesis row's %s is %r, the locked document's %r"
                             % (field, first.get(field), want))
    reg = [r for r in common.read_jsonl(registry_path)
           if isinstance(r, dict) and r.get("study_id") == doc["study_id"]]
    if len(reg) != 1:
        raise ValueError("GATE-LOCK: studies.jsonl holds %d rows for study %s, exactly one expected"
                         % (len(reg), doc["study_id"]))
    row = reg[0]
    if tuple(sorted(row.keys())) != tuple(sorted(REGISTRY_KEYS)):
        raise ValueError("GATE-LOCK: the studies.jsonl row does not carry exactly the keys %s"
                         % (", ".join(REGISTRY_KEYS),))
    for field in ("lock_sha", "declaration_sha", "template_sha", "gates_lock"):
        if row.get(field) != first.get(field):
            raise ValueError("GATE-LOCK: the studies.jsonl row's %s is %r, the genesis row's %r"
                             % (field, row.get(field), first.get(field)))
    for field in ("supersedes_study", "supersedes_lock"):
        if row.get(field) != doc.get(field):
            raise ValueError("GATE-LOCK: the studies.jsonl row's %s is %r, the locked document's %r"
                             % (field, row.get(field), doc.get(field)))
    return doc


def cover(verdict_doc, checks_doc, requirements_doc) -> None:
    """GATE-COVER (docs/16a §B.1/§E): one cad-verdict/1 doc covers exactly one checks doc.

    Both docs must be schema-valid (GATE-DOC otherwise); verify.cover must accept the pair; the
    verdict must name the same study, the requirements lock and the checks sha; no req_id twice; the
    check and verdict id sets equal, in their own orders for the messages; every verdict's hardness
    and primitive its check's; and the design verdict the derived one."""
    errs = schema.errors(verdict_doc, "cad-verdict/1") or schema.errors(checks_doc, "cad-checks/1")
    if errs:
        raise ValueError("GATE-DOC: %s" % (errs[0],))
    try:
        verify.cover(checks_doc, requirements_doc)
    except ValueError as e:
        raise ValueError("GATE-COVER: %s" % (e,))
    checks = checks_doc["checks"]
    for field, want in (("study_id", requirements_doc["study_id"]),
                        ("requirements_lock", requirements_doc["lock_sha"]),
                        ("checks_sha", common.sha256_of(checks_doc))):
        if verdict_doc[field] != want:
            raise ValueError("GATE-COVER: the verdict carries %s %r, the locked set %r"
                             % (field, verdict_doc[field], want))
    verdicts = verdict_doc["verdicts"]
    seen = set()
    for v in verdicts:
        if v["req_id"] in seen:
            raise ValueError("GATE-COVER: verdict %s appears twice" % (v["req_id"],))
        seen.add(v["req_id"])
    for c in checks:
        if c["req_id"] not in seen:
            raise ValueError("GATE-COVER: missing verdict %s" % (c["req_id"],))
    check_ids = set(c["req_id"] for c in checks)
    for v in verdicts:
        if v["req_id"] not in check_ids:
            raise ValueError("GATE-COVER: extra verdict %s" % (v["req_id"],))
    by_id = dict((c["req_id"], c) for c in checks)
    for v in verdicts:
        for field in ("hardness", "primitive"):
            if v[field] != by_id[v["req_id"]][field]:
                raise ValueError("GATE-COVER: verdict %s %s %r is not its check's %r"
                                 % (v["req_id"], field, v[field], by_id[v["req_id"]][field]))
    if verdict_doc["design_verdict"] != verify.design_verdict(verdicts):
        raise ValueError("GATE-COVER: the design verdict %r is not the derived %r"
                         % (verdict_doc["design_verdict"], verify.design_verdict(verdicts)))


def evidence_check(study_dir, verdict_doc, phase) -> None:
    """GATE-EVIDENCE (docs/16a §B.1): every verdict row's bound evidence file re-hashed.

    evidence_path and evidence_sha are set together or not at all; a set path carries no drive and
    resolves strictly below the study directory (a common-path test); the file must hash to its
    bound sha, before or after the comparison."""
    for v in verdict_doc["verdicts"]:
        path, sha = v.get("evidence_path"), v.get("evidence_sha")
        if (path is None) != (sha is None):
            raise ValueError("GATE-EVIDENCE: %s: %s: evidence_path and evidence_sha are set together"
                             " or not at all" % (phase, v["req_id"]))
        if path is None:
            continue
        if os.path.isabs(path) or os.path.splitdrive(path)[0]:
            raise ValueError("GATE-EVIDENCE: %s: %s: evidence path %r leaves the study"
                             % (phase, v["req_id"], path))
        abs_study = os.path.normpath(os.path.abspath(study_dir))
        abs_path = os.path.normpath(os.path.join(abs_study, path))
        if abs_path == abs_study or os.path.commonpath((abs_path, abs_study)) != abs_study:
            raise ValueError("GATE-EVIDENCE: %s: %s: evidence path %r leaves the study"
                             % (phase, v["req_id"], path))
        snap = common.stable_file_snapshot(os.path.join(study_dir, path))
        if snap["stable"] is not True:
            raise ValueError("GATE-EVIDENCE: %s: %s: %s is missing or not a stable regular file"
                             % (phase, v["req_id"], path))
        if snap["sha256"] != sha:
            raise ValueError("GATE-EVIDENCE: %s: %s: %s carries sha %s, the verdict bound %s"
                             % (phase, v["req_id"], path, snap["sha256"], sha))


_DELTA_OF = {
    ("pass", "pass"): None, ("pass", "fail"): "regressed",
    ("pass", "not_evaluable"): "not_reevaluated",
    ("fail", "pass"): "resolved", ("fail", "fail"): "remaining",
    ("fail", "not_evaluable"): "not_reevaluated",
    ("not_evaluable", "pass"): "resolved", ("not_evaluable", "fail"): "new_fail",
    ("not_evaluable", "not_evaluable"): "remaining",
    (None, "pass"): None, (None, "fail"): "new_fail",
    (None, "not_evaluable"): "remaining",
}


def deltas(candidate, stable) -> dict:
    """The requirement deltas of docs/16a §B.1 over the hard rows: five sorted id lists.

    A hard row NE in the candidate but decided in the stable is not_reevaluated, never resolved; a
    soft or objective row never enters."""
    out = dict((k, []) for k in DELTA_KEYS)
    decided = None if stable is None \
        else dict((v["req_id"], v["verdict"]) for v in stable["verdicts"] if v["hardness"] == "hard")
    for v in candidate["verdicts"]:
        if v["hardness"] != "hard":
            continue
        s = decided.get(v["req_id"]) if decided is not None else None
        key = _DELTA_OF[(s, v["verdict"])]
        if key is not None:
            out[key].append(v["req_id"])
    return dict((k, sorted(out[k])) for k in DELTA_KEYS)


def _hard_verdicts(doc) -> dict:
    """The hard rows of a verdict doc as {req_id: verdict} (the objective row never enters)."""
    return dict((v["req_id"], v["verdict"]) for v in doc["verdicts"] if v["hardness"] == "hard")


def compare(candidate, stable, objective, repeat_band) -> dict:
    """The promotion rule of docs/16 §E.6 on two already covered verdict docs.

    (a) no hard not_evaluable row in the candidate; (b) every hard row that passes in the stable
    passes in the candidate; (c) strictly more hard passes, or every hard row passes in both and the
    objective improves by more than max(u, repeat band). Ties never promote. Soft rows never enter.
    Returns {"decision", "rule_id", "reason"}."""
    cand, stab = _hard_verdicts(candidate), None if stable is None else _hard_verdicts(stable)
    n_cand = sum(1 for s in cand.values() if s == "pass")
    n_stab = 0 if stable is None else sum(1 for s in stab.values() if s == "pass")
    ne = sorted(r for r, s in cand.items() if s == "not_evaluable")
    if ne:
        return {"decision": "reject", "rule_id": "GATE-NE",
                "reason": "the candidate is not evaluable on hard row %s, so its %d hard passes cannot"
                          " be judged against the stable design" % (ne[0], n_cand)}
    if stable is None:
        return {"decision": "promote", "rule_id": "GATE-FIRST",
                "reason": "no stable design exists yet, so the candidate with %d hard passes and no"
                          " not_evaluable row becomes the first stable" % (n_cand,)}
    regressed = sorted(r for r, s in stab.items() if s == "pass" and cand[r] != "pass")
    if regressed:
        return {"decision": "reject", "rule_id": "GATE-REGRESS",
                "reason": "the candidate passes %d hard rows against the stable's %d but loses the"
                          " passing row %s" % (n_cand, n_stab, regressed[0])}
    if n_cand > n_stab:
        return {"decision": "promote", "rule_id": "GATE-MORE",
                "reason": "the candidate passes %d hard rows against the stable's %d, strictly more"
                          " with no regression" % (n_cand, n_stab)}
    if n_cand != n_stab or len(cand) != n_cand:
        return {"decision": "reject", "rule_id": "GATE-TIE",
                "reason": "the candidate's %d hard passes do not exceed the stable's %d and not every"
                          " hard row passes in both: ties never promote" % (n_cand, n_stab)}
    if objective is None:
        return {"decision": "reject", "rule_id": "GATE-TIE",
                "reason": "all %d hard rows pass in both but the study states no objective: ties never"
                          " promote" % (n_cand,)}
    def _obj(doc):
        for v in doc["verdicts"]:
            if v["hardness"] == "objective":
                return v
        return None
    o_cand, o_stab = _obj(candidate), _obj(stable)
    if o_cand is None or o_stab is None or o_cand["verdict"] == "not_evaluable" \
            or o_stab["verdict"] == "not_evaluable" \
            or o_cand["m"] is None or o_cand["u"] is None or o_stab["m"] is None or o_stab["u"] is None:
        return {"decision": "reject", "rule_id": "GATE-OBJNE",
                "reason": "all %d hard rows pass in both but the objective is not a measured number in"
                          " both designs, so no gain can be proven" % (n_cand,)}
    sense = objective.get("sense")
    gain = (o_stab["m"] - o_cand["m"]) if sense == "min" else (o_cand["m"] - o_stab["m"])
    noise = max(o_stab["u"], o_cand["u"], repeat_band)
    if gain > noise:
        return {"decision": "promote", "rule_id": "GATE-OBJ",
                "reason": "all %d hard rows pass in both and the objective improves by %r, more than the"
                          " noise %r" % (n_cand, gain, noise)}
    if gain > 0:
        return {"decision": "reject", "rule_id": "GATE-NOISE",
                "reason": "all %d hard rows pass in both and the objective improves by %r, at or below"
                          " the noise %r" % (n_cand, gain, noise)}
    return {"decision": "reject", "rule_id": "GATE-TIE",
            "reason": "all %d hard rows pass in both and the objective gain %r is not positive: ties"
                      " never promote" % (n_cand, gain)}


def promote(study_dir, candidate, stable, checks_doc, base_stable_eval_key, ctx, repeat_band=0.0,
            gates_path=GATES, gates_lock_path=GATES_LOCK, registry_path=REGISTRY, between_hook=None):
    """The whole gate of docs/16 §E.6 (docs/16a §B.1): locks, anchors, coverage, evidence, compare.

    Reads only, writes nothing: the caller owns the proposal/stable swap (CAD-18). The optional
    between_hook(study_dir), a selftest hook, runs after the decision and before the evidence is
    hashed again, so a file changed during the comparison is caught. Returns one cad-decision/1 row."""
    gates, gates_sha = load_gates(gates_path, gates_lock_path)
    doc = lock_anchor(study_dir, gates_sha, registry_path)
    if not isinstance(ctx, dict) or tuple(sorted(ctx.keys())) != tuple(sorted(CTX_KEYS)):
        raise ValueError("GATE-DOC: ctx does not carry exactly the keys %s" % (", ".join(CTX_KEYS),))
    if not _num(repeat_band) or repeat_band < 0:
        raise ValueError("GATE-DOC: repeat_band %r is not a finite number >= 0" % (repeat_band,))
    cover(candidate, checks_doc, doc)
    if stable is not None:
        cover(stable, checks_doc, doc)
    if stable is None:
        if base_stable_eval_key is not None:
            raise ValueError("GATE-STALE: no stable design yet, so base_stable_eval_key must be null,"
                             " got %r" % (base_stable_eval_key,))
    elif base_stable_eval_key != stable["eval_key"]:
        raise ValueError("GATE-STALE: base_stable_eval_key %r is not the stable design's eval key %r"
                         % (base_stable_eval_key, stable["eval_key"]))
    evidence_check(study_dir, candidate, "before")
    if stable is not None:
        evidence_check(study_dir, stable, "before")
    res = compare(candidate, stable, doc["objective"], repeat_band)
    d = deltas(candidate, stable)
    if between_hook is not None:
        between_hook(study_dir)
    evidence_check(study_dir, candidate, "after")
    if stable is not None:
        evidence_check(study_dir, stable, "after")
    row = {"schema": "cad-decision/1", "study_id": doc["study_id"], "n": ctx["n"],
           "after_iteration": ctx["after_iteration"], "decision": res["decision"],
           "rule_id": res["rule_id"], "stable_eval_key": None if stable is None else stable["eval_key"],
           "candidate_eval_key": candidate["eval_key"], "params_sha": ctx["params_sha"],
           "law": ctx["law"], "sobol_index": ctx["sobol_index"], "ei": ctx["ei"],
           "reason": res["reason"], "gates_lock": gates_sha, "requirements_lock": doc["lock_sha"],
           "base_stable_eval_key": base_stable_eval_key}
    row.update(d)
    errs = schema.errors(row, "cad-decision/1")
    if errs:
        raise ValueError("GATE-DOC: %s" % (errs[0],))
    return row


def _non_neg_int(x) -> bool:
    """A non-bool integer >= 0: the only kind that may carry n_evals, since_promote or llm_rejected."""
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


def ladder(state, gates):
    """The ladder and the stop of docs/16 §E.7: (decision, rule_id) or None to keep iterating.

    Stop as soon as max_evals is reached or the stable design is feasible with the best expected
    improvement below the band; the LLM goes off after llm_reject_k rejected edits and is consulted
    after stall_k non-promotions past the initial design."""
    if not isinstance(state, dict) or tuple(sorted(state.keys())) != tuple(sorted(STATE_KEYS)):
        raise ValueError("GATE-DOC: state does not carry exactly the keys %s" % (", ".join(STATE_KEYS),))
    for key in ("n_evals", "since_promote", "llm_rejected"):
        if not _non_neg_int(state[key]):
            raise ValueError("GATE-DOC: state %s %r is not a non-negative integer" % (key, state[key]))
    for key in ("llm_off", "stable_feasible"):
        if not isinstance(state[key], bool):
            raise ValueError("GATE-DOC: state %s %r is not a bool" % (key, state[key]))
    if state["best_ei"] is not None and not _num(state["best_ei"]):
        raise ValueError("GATE-DOC: state best_ei %r is neither null nor a finite number"
                         % (state["best_ei"],))
    if not _num(state["band"]) or state["band"] < 0:
        raise ValueError("GATE-DOC: state band %r is not a finite number >= 0" % (state["band"],))
    if state["n_evals"] >= gates["max_evals"]:
        return ("stop", "GATE-MAXEVALS")
    if state["stable_feasible"] and state["best_ei"] is not None and state["best_ei"] < state["band"]:
        return ("stop", "GATE-FEASIBLE")
    if not state["llm_off"] and state["llm_rejected"] >= gates["llm_reject_k"]:
        return ("llm_off", "GATE-LLMOFF")
    if not state["llm_off"] and state["n_evals"] >= gates["init_design"] \
            and state["since_promote"] >= gates["stall_k"]:
        return ("llm_consult", "GATE-STALL")
    return None


def _margin(check, m) -> float:
    """The margin of one check at its measured m: positive means passing (docs/16 §E.7)."""
    if check["op"] == "<=":
        return check["hi"] + check["tol"] - m
    if check["op"] == ">=":
        return m - check["lo"] + check["tol"]
    if check["op"] == "==":
        return check["tol"] - abs(m - check["lo"])
    return min(m - check["lo"] + check["tol"], check["hi"] + check["tol"] - m)


def confirm(l2_verdict, checks_doc, requirements_doc, cfd_u) -> dict:
    """The L2 confirmation of docs/16 §E.7: every performance margin must exceed the noise.

    First the coverage gate; then every hard row fails the confirmation whose verdict is not a pass,
    and every hard repr-cfd row (op not is_true) whose margin is not strictly greater than the largest
    non-None noise of its req_id in cfd_u - a missing entry or no non-None value fails too."""
    cover(l2_verdict, checks_doc, requirements_doc)
    by_id = dict((v["req_id"], v) for v in l2_verdict["verdicts"])
    failing = []
    for check in checks_doc["checks"]:
        if check["hardness"] != "hard":
            continue
        rid = check["req_id"]
        v = by_id[rid]
        if v["verdict"] != "pass" or v["m"] is None:
            failing.append(rid)
            continue
        if check["repr"] == "cfd" and check["op"] != "is_true":
            entry = cfd_u.get(rid)
            noises = [x for x in entry.values() if x is not None] if isinstance(entry, dict) else []
            if not noises or not (_margin(check, v["m"]) > max(noises)):
                failing.append(rid)
    return {"decision": "confirm_pass" if not failing else "confirm_fail", "rule_id": CONFIRM_ID,
            "failing": sorted(failing)}


# ---- selftest fixtures (temp directories only; nothing here runs in production) ----

ALL_PASS = {"fail": [], "ne": [], "soft": "pass", "obj_m": 0.0625, "obj_u": 2.0 ** -30}


def _fx_study(td) -> dict:
    """A locked v1_nominal study in td: requirements, genesis row, registry row, gates copies."""
    g = common.read_json(GOLDEN_V1)
    R, C = g["requirements"], g["checks"]
    study = os.path.join(td, "study")
    reqs.write_locked(study, R)
    genesis = {"kind": "genesis", "study_id": R["study_id"], "lock_sha": R["lock_sha"],
               "declaration_sha": R["declaration_sha"], "template_sha": R["template_sha"],
               "gates_lock": GATES_SHA}
    common.jsonl_append(os.path.join(study, "iterations.jsonl"), genesis)
    reg = dict((k, v) for k, v in genesis.items() if k != "kind")
    reg["supersedes_study"] = R["supersedes_study"]
    reg["supersedes_lock"] = R["supersedes_lock"]
    registry = os.path.join(td, "studies.jsonl")
    common.jsonl_append(registry, reg)
    for name, src in (("gates.json", GATES), ("gates.lock", GATES_LOCK)):
        with open(src, "rb") as f:
            common.atomic_write(os.path.join(td, name), f.read())
    return {"study": study, "registry": registry, "gates": os.path.join(td, "gates.json"),
            "gates_lock": os.path.join(td, "gates.lock"), "R": R, "C": C}


def _fx_verdict(study, R, C, name, side_name, side) -> dict:
    """A cad-verdict/1 doc for one scenario side, with its evidence file bound by sha."""
    ev_rel = "evidence/%s-%s.json" % (name, side_name)
    ev_bytes = common.canonical_bytes({"scenario": name, "side": side_name}) + b"\n"
    common.atomic_write(os.path.join(study, ev_rel), ev_bytes)
    ev_sha = common.sha256_bytes(ev_bytes)
    rows = []
    for c in C["checks"]:
        rid, hardness = c["req_id"], c["hardness"]
        if hardness == "objective":
            m, u = side["obj_m"], side["obj_u"]
            if m is not None and u is not None:
                verdict, reason = "pass", None
            else:
                verdict, reason = "not_evaluable", "NE-UNCERTAIN"
        elif rid in side["ne"]:
            m, u, verdict, reason = None, None, "not_evaluable", "NE-UNCERTAIN"
        elif rid in side["fail"] or (hardness == "soft" and side["soft"] == "fail"):
            m, u, verdict, reason = 0.0, 0.0, "fail", None
        else:
            m, u, verdict, reason = 1.0, 0.0, "pass", None
        rows.append({"req_id": rid, "primitive": c["primitive"], "feature": c["args"]["feature"],
                     "hardness": hardness, "m": m, "u": u, "verdict": verdict, "reason_id": reason,
                     "evidence_path": ev_rel if m is not None else None,
                     "evidence_sha": ev_sha if m is not None else None})
    doc = {"schema": "cad-verdict/1", "study_id": R["study_id"], "requirements_lock": R["lock_sha"],
           "checks_sha": common.sha256_of(C),
           "eval_key": common.sha256_bytes((side_name + ":" + name).encode("ascii")),
           "design_verdict": verify.design_verdict(rows), "verdicts": rows}
    errs = schema.errors(doc, "cad-verdict/1")
    assert not errs, (name, side_name, errs)
    return doc


def _fx_ctx(i, name) -> dict:
    """The loop context of one decision, keyed by the scenario name."""
    return {"n": i, "after_iteration": i, "params_sha": common.sha256_bytes(("params:" + name).encode("ascii")),
            "law": "poly5", "sobol_index": None, "ei": None}


def _fx_refused(fn, prefix, needle=None) -> str:
    """Assert fn raised ValueError starting with prefix + ':' (containing needle); return the text."""
    try:
        fn()
    except ValueError as e:
        msg = str(e)
        assert msg.startswith(prefix + ":"), "expected %r, got %r" % (prefix + ":", msg)
        if needle is not None:
            assert needle in msg, "expected %r in %r" % (needle, msg)
        return msg
    raise AssertionError("expected ValueError %s, nothing raised" % (prefix,))


# ---- the selftest: T1-T12, one [ok] line each ----

def _t1() -> None:
    """gates.lock binds gates.json by bytes sha; tampering, a wrong lock and a missing lock refuse."""
    doc, sha = load_gates()
    assert doc == common.read_json(GATES) and sha == GATES_SHA, sha
    with tempfile.TemporaryDirectory() as td:
        fx = _fx_study(td)
        stable = _fx_verdict(fx["study"], fx["R"], fx["C"], "T1", "stable", ALL_PASS)
        cand = _fx_verdict(fx["study"], fx["R"], fx["C"], "T1", "candidate", ALL_PASS)
        common.write_json(fx["gates"], dict(copy.deepcopy(doc), stall_k=5))
        _fx_refused(lambda: load_gates(fx["gates"], fx["gates_lock"]), "GATE-LOCK")
        args = (fx["study"], cand, stable, fx["C"], stable["eval_key"], _fx_ctx(0, "T1"))
        _fx_refused(lambda: promote(*args, 0.0, fx["gates"], fx["gates_lock"], fx["registry"]),
                    "GATE-LOCK")
        with open(GATES, "rb") as f:
            common.atomic_write(fx["gates"], f.read())
        common.atomic_write(fx["gates_lock"], "# c\n" + "0" * 64 + "  gates.json\n")
        _fx_refused(lambda: load_gates(fx["gates"], fx["gates_lock"]), "GATE-LOCK", "does not stamp")
        os.remove(fx["gates_lock"])
        _fx_refused(lambda: load_gates(fx["gates"], fx["gates_lock"]), "GATE-LOCK", "unreadable")
    real_read_json = common.read_json

    def _third_read(path):
        """A read_json gone bad: the gates doc with stall_k 99, not the hashed bytes."""
        res = real_read_json(path)
        return dict(res, stall_k=99) if str(path).endswith("gates.json") else res

    common.read_json = _third_read
    try:
        doc2, sha2 = load_gates()
    finally:
        common.read_json = real_read_json
    assert doc2["stall_k"] == 3, \
        "load_gates returned a doc it never hashed: stall_k %r" % (doc2["stall_k"],)
    print("[ok] gates.lock binds gates.json by bytes sha %s...; tampered gates.json, wrong lock,"
          " missing lock refused GATE-LOCK" % (GATES_SHA[:8],))


def _scenario_rows(fx, cases) -> list:
    """The 16 promote rows of the scenario table, in file order (T2, T12)."""
    rows = []
    for i, case in enumerate(cases):
        name = case["name"]
        stable = _fx_verdict(fx["study"], fx["R"], fx["C"], name, "stable", case["stable"]) \
            if case["stable"] is not None else None
        cand = _fx_verdict(fx["study"], fx["R"], fx["C"], name, "candidate", case["candidate"])
        rows.append(promote(fx["study"], cand, stable, fx["C"],
                            None if stable is None else stable["eval_key"],
                            _fx_ctx(i, name), case["repeat_band"],
                            fx["gates"], fx["gates_lock"], fx["registry"]))
    return rows


def _t2() -> list:
    """All 16 scenario tables judged exactly as their want: decision, rule_id, five delta arrays."""
    cases = common.read_json(SCENARIOS)["cases"]
    assert len(cases) == 16, len(cases)
    with tempfile.TemporaryDirectory() as td:
        rows = _scenario_rows(_fx_study(td), cases)
    n_promote = 0
    for case, row in zip(cases, rows):
        want = case["want"]
        assert row["decision"] == want["decision"] and row["rule_id"] == want["rule_id"], \
            (case["name"], row["decision"], row["rule_id"])
        for k in DELTA_KEYS:
            assert row[k] == want[k], (case["name"], k, row[k], want[k])
        n_promote += row["decision"] == "promote"
    print("[ok] 16 scenario tables judged as specified: %d promote, %d reject"
          % (n_promote, len(cases) - n_promote))
    return rows


def _t3(rows) -> None:
    """Every promote row is a valid cad-decision/1 row bound to the lock, the gates and both keys."""
    cases = common.read_json(SCENARIOS)["cases"]
    fx_R_lock = common.read_json(GOLDEN_V1)["requirements"]["lock_sha"]
    for case, row in zip(cases, rows):
        name, cand_key = case["name"], common.sha256_bytes(("candidate:" + case["name"]).encode("ascii"))
        assert not schema.errors(row, "cad-decision/1"), (name, schema.errors(row, "cad-decision/1"))
        assert row["requirements_lock"] == fx_R_lock and row["gates_lock"] == GATES_SHA, name
        assert row["base_stable_eval_key"] == (None if case["stable"] is None else
                                               common.sha256_bytes(("stable:" + name).encode("ascii"))), name
        assert row["candidate_eval_key"] == cand_key, name
        assert row["study_id"] == "v1_nominal", name
    print("[ok] every promote row is a valid cad-decision/1 row carrying requirements_lock, both"
          " eval keys and the gates lock")


def _t4() -> None:
    """A candidate whose verdicts miss one check id is refused GATE-COVER (extra 1)."""
    with tempfile.TemporaryDirectory() as td:
        fx = _fx_study(td)
        cand = _fx_verdict(fx["study"], fx["R"], fx["C"], "T4", "candidate", ALL_PASS)
        cand["verdicts"] = [v for v in cand["verdicts"] if v["req_id"] != "SYS-MACH"]
        msg = _fx_refused(lambda: promote(fx["study"], cand, None, fx["C"], None, _fx_ctx(0, "T4"),
                                          0.0, fx["gates"], fx["gates_lock"], fx["registry"]),
                          "GATE-COVER", "missing verdict SYS-MACH")
    print("[ok] coverage: a candidate with no SYS-MACH verdict is refused (%s)" % (msg,))


def _t5() -> None:
    """An extra verdict, a flipped hardness, a wrong checks sha and a tampered design verdict refuse."""
    with tempfile.TemporaryDirectory() as td:
        fx = _fx_study(td)
        R, C, study = fx["R"], fx["C"], fx["study"]
        cand = _fx_verdict(study, R, C, "T5", "candidate", ALL_PASS)
        extra = copy.deepcopy([v for v in cand["verdicts"] if v["req_id"] == "REQ-003"][0])
        extra["req_id"] = "REQ-007"
        cand["verdicts"].append(extra)
        _fx_refused(lambda: promote(study, cand, None, C, None, _fx_ctx(1, "T5"),
                                    0.0, fx["gates"], fx["gates_lock"], fx["registry"]),
                    "GATE-COVER", "extra verdict REQ-007")
        flip = _fx_verdict(study, R, C, "T5f", "candidate", dict(ALL_PASS, fail=["SYS-MACH"]))
        [v for v in flip["verdicts"] if v["req_id"] == "SYS-MACH"][0]["hardness"] = "soft"
        _fx_refused(lambda: promote(study, flip, None, C, None, _fx_ctx(2, "T5f"),
                                    0.0, fx["gates"], fx["gates_lock"], fx["registry"]),
                    "GATE-COVER", "SYS-MACH")
        sha = _fx_verdict(study, R, C, "T5s", "candidate", ALL_PASS)
        sha["checks_sha"] = "0" * 64
        _fx_refused(lambda: promote(study, sha, None, C, None, _fx_ctx(3, "T5s"),
                                    0.0, fx["gates"], fx["gates_lock"], fx["registry"]),
                    "GATE-COVER", "checks_sha")
        dv = _fx_verdict(study, R, C, "T5d", "candidate", ALL_PASS)
        dv["design_verdict"] = "infeasible"
        _fx_refused(lambda: promote(study, dv, None, C, None, _fx_ctx(4, "T5d"),
                                    0.0, fx["gates"], fx["gates_lock"], fx["registry"]),
                    "GATE-COVER", "design verdict")
    print("[ok] coverage: an extra verdict REQ-007, a hardness flipped to soft, a zeroed checks_sha"
          " and a hand-set design_verdict are all refused GATE-COVER")


def _t6_leaves(fx) -> None:
    """A drive-relative evidence path and a ..-climbing one leave the study: GATE-EVIDENCE."""
    for bad in ("Z:evidence/x.json", "evidence/../../x.json"):
        cand = _fx_verdict(fx["study"], fx["R"], fx["C"], "T6p", "candidate", ALL_PASS)
        cand["verdicts"][0]["evidence_path"] = bad
        _fx_refused(lambda: evidence_check(fx["study"], cand, "before"),
                    "GATE-EVIDENCE", "leaves the study")


def _t6() -> None:
    """A tampered evidence file is refused GATE-EVIDENCE before the comparison (extra 3)."""
    with tempfile.TemporaryDirectory() as td:
        fx = _fx_study(td)
        cand = _fx_verdict(fx["study"], fx["R"], fx["C"], "T6", "candidate", ALL_PASS)
        common.atomic_write(os.path.join(fx["study"], "evidence", "T6-candidate.json"),
                            b'{"scenario": "T6", "side": "tampered"}\n')
        _fx_refused(lambda: promote(fx["study"], cand, None, fx["C"], None, _fx_ctx(5, "T6"),
                                    0.0, fx["gates"], fx["gates_lock"], fx["registry"]),
                    "GATE-EVIDENCE", "before")
        _t6_leaves(fx)
    print("[ok] evidence: overwriting the candidate's evidence file before promote is refused"
          " GATE-EVIDENCE in the before phase")


def _t7() -> None:
    """Evidence changed during the comparison refuses after the decision; an identical rewrite passes."""
    with tempfile.TemporaryDirectory() as td:
        fx = _fx_study(td)
        name = "T7"
        cand = _fx_verdict(fx["study"], fx["R"], fx["C"], name, "candidate", ALL_PASS)
        evp = os.path.join(fx["study"], "evidence", name + "-candidate.json")

        def hook_append(sd):
            with open(os.path.join(sd, "evidence", name + "-candidate.json"), "ab") as f:
                f.write(b"x")

        _fx_refused(lambda: promote(fx["study"], cand, None, fx["C"], None, _fx_ctx(6, name),
                                    0.0, fx["gates"], fx["gates_lock"], fx["registry"],
                                    between_hook=hook_append),
                    "GATE-EVIDENCE", "after")
        common.atomic_write(evp, common.canonical_bytes({"scenario": name, "side": "candidate"}) + b"\n")
        runs = []

        def hook_same(sd):
            runs.append(1)
            common.atomic_write(evp, common.canonical_bytes({"scenario": name, "side": "candidate"}) + b"\n")

        promote(fx["study"], cand, None, fx["C"], None, _fx_ctx(6, name),
                0.0, fx["gates"], fx["gates_lock"], fx["registry"], between_hook=hook_same)
        assert runs == [1], runs
    print("[ok] evidence: a byte appended by the between hook refuses GATE-EVIDENCE in the after"
          " phase; rewriting the identical bytes passes, and the hook ran exactly once")


def _t8() -> None:
    """A stale base_stable_eval_key refuses GATE-STALE, both with and without a stable design (extra 5)."""
    with tempfile.TemporaryDirectory() as td:
        fx = _fx_study(td)
        R, C, study = fx["R"], fx["C"], fx["study"]
        stable = _fx_verdict(study, R, C, "T8", "stable", ALL_PASS)
        cand = _fx_verdict(study, R, C, "T8", "candidate", ALL_PASS)
        _fx_refused(lambda: promote(study, cand, stable, C, "e" * 64, _fx_ctx(7, "T8"),
                                    0.0, fx["gates"], fx["gates_lock"], fx["registry"]),
                    "GATE-STALE")
        first = _fx_verdict(study, R, C, "T8f", "candidate", ALL_PASS)
        _fx_refused(lambda: promote(study, first, None, C, first["eval_key"], _fx_ctx(8, "T8f"),
                                    0.0, fx["gates"], fx["gates_lock"], fx["registry"]),
                    "GATE-STALE")
    print("[ok] stale base: a base_stable_eval_key that is not the stable design's eval key, and one"
          " named with no stable design at all, are refused GATE-STALE")


def _t9() -> None:
    """Both lock files rewritten consistently is caught by the genesis row, then by studies.jsonl."""
    with tempfile.TemporaryDirectory() as td:
        fx = _fx_study(td)
        study, R, C = fx["study"], fx["R"], fx["C"]
        R2 = copy.deepcopy(R)
        R2["rows"][2]["value"] = 0.09
        R2["lock_sha"] = reqs.lock_sha_of(R2)
        common.atomic_write(os.path.join(study, "requirements.json"), common.canonical_json(R2) + "\n")
        common.atomic_write(os.path.join(study, "requirements.lock"), R2["lock_sha"] + "\n")
        assert reqs.read_locked(study) == R2, "the sibling lock check is fooled"
        C2 = reqs.compile_checks(R2, *reqs.load_template(reqs.NOZZLE_DIR)[0::2])
        stable = _fx_verdict(study, R2, C2, "T9", "stable", ALL_PASS)
        cand = _fx_verdict(study, R2, C2, "T9", "candidate", ALL_PASS)
        args = (study, cand, stable, C2, stable["eval_key"], _fx_ctx(9, "T9"))
        tail = (0.0, fx["gates"], fx["gates_lock"], fx["registry"])
        _fx_refused(lambda: promote(*args, *tail), "GATE-LOCK", "genesis")
        it = os.path.join(study, "iterations.jsonl")
        lines = lambda rs: b"".join(common.canonical_bytes(r) + b"\n" for r in rs)
        rows = common.read_jsonl(it)
        rows[0]["lock_sha"] = R2["lock_sha"]
        common.atomic_write(it, lines(rows))
        _fx_refused(lambda: promote(*args, *tail), "GATE-LOCK", "studies.jsonl")
        os.remove(it)
        _fx_refused(lambda: promote(*args, *tail), "GATE-LOCK", "no genesis row")
        common.atomic_write(it, lines(rows))
        common.jsonl_append(it, dict(rows[0]))
        _fx_refused(lambda: promote(*args, *tail), "GATE-LOCK", "a second genesis row")
        common.atomic_write(it, lines(rows[:1]))
        reg_rows = common.read_jsonl(fx["registry"])
        common.jsonl_append(fx["registry"], dict(reg_rows[0]))
        _fx_refused(lambda: promote(*args, *tail), "GATE-LOCK", "studies.jsonl")
    print("[ok] anchors: requirements.json and requirements.lock rewritten consistently is refused"
          " GATE-LOCK by the genesis row, then by studies.jsonl; a missing iterations.jsonl, a second"
          " genesis row and a doubled registry row refuse too")


def _t10() -> None:
    """The §E.7 ladder judged on the tree's gates, plus its state shape refusals."""
    gates, _ = load_gates()
    base = {"n_evals": 10, "since_promote": 0, "llm_rejected": 0, "llm_off": False,
            "stable_feasible": False, "best_ei": None, "band": 0.0}
    assert ladder(base, gates) is None, "the base state must keep iterating"
    table = ((dict(n_evals=24), ("stop", "GATE-MAXEVALS")),
             (dict(n_evals=23), None),
             (dict(stable_feasible=True, best_ei=0.0009765625, band=0.001953125), ("stop", "GATE-FEASIBLE")),
             (dict(stable_feasible=True, best_ei=0.001953125, band=0.001953125), None),
             (dict(since_promote=3), ("llm_consult", "GATE-STALL")),
             (dict(since_promote=2), None),
             (dict(since_promote=3, n_evals=5), None),
             (dict(llm_rejected=2), ("llm_off", "GATE-LLMOFF")),
             (dict(llm_rejected=2, llm_off=True, since_promote=3), None),
             (dict(n_evals=24, llm_rejected=2), ("stop", "GATE-MAXEVALS")))
    for patch, want in table:
        state = dict(base, **patch)
        got = ladder(state, gates)
        assert got == want, (patch, got, want)
    no_band = dict(base)
    del no_band["band"]
    _fx_refused(lambda: ladder(no_band, gates), "GATE-DOC")
    _fx_refused(lambda: ladder(dict(base, n_evals=True), gates), "GATE-DOC")
    print("[ok] ladder: 10 state changes judged in rule order - max_evals, feasible-then-stop,"
          " llm_off at 2 rejects, llm_consult at 3 stalls - and 2 malformed states GATE-DOC")


def _t11() -> None:
    """The L2 confirmation: every hard repr-cfd margin must exceed the largest non-None noise."""
    with tempfile.TemporaryDirectory() as td:
        fx = _fx_study(td)
        R, C, study = fx["R"], fx["C"], fx["study"]
        cand = _fx_verdict(study, R, C, "T11", "candidate", ALL_PASS)
        [v for v in cand["verdicts"] if v["req_id"] == "SYS-MACH"][0]["m"] = 0.25
        first = {"SYS-MACH": {"gci_fine": 0.03125, "repeat_band": 0.015625}}
        assert confirm(cand, C, R, first) == {"decision": "confirm_pass", "rule_id": CONFIRM_ID,
                                              "failing": []}
        for cfd_u in ({"SYS-MACH": {"gci_fine": 0.0625, "repeat_band": None}},
                      {},
                      {"SYS-MACH": {"gci_fine": None, "repeat_band": 0.0625}}):
            res = confirm(cand, C, R, cfd_u)
            assert res["decision"] == "confirm_fail" and res["failing"] == ["SYS-MACH"] \
                and res["rule_id"] == CONFIRM_ID, (cfd_u, res)
        failing = _fx_verdict(study, R, C, "T11f", "candidate", dict(ALL_PASS, fail=["REQ-003"]))
        [v for v in failing["verdicts"] if v["req_id"] == "SYS-MACH"][0]["m"] = 0.25
        assert confirm(failing, C, R, first) == {"decision": "confirm_fail", "rule_id": CONFIRM_ID,
                                                 "failing": ["REQ-003"]}
    print("[ok] confirm: the SYS-MACH margin 0.05 passes over noise 0.03125 and fails over 0.0625, a"
          " missing or all-null cfd_u entry fails, a failed REQ-003 fails, rule_id GATE-CONFIRM")


def _snap_tree(root) -> dict:
    """{relative posix path: sha256 of bytes} for every file under root."""
    out = {}
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            p = os.path.join(dirpath, fn)
            out[os.path.relpath(p, root).replace(os.sep, "/")] = common.sha256_file(p)
    return out


def _t12() -> None:
    """Two runs of the 16 scenarios in two fresh dirs give byte-identical rows; promote writes nothing."""
    cases = common.read_json(SCENARIOS)["cases"]
    digests = []
    for _run in range(2):
        with tempfile.TemporaryDirectory() as td:
            fx = _fx_study(td)
            built = []
            for i, case in enumerate(cases):
                name = case["name"]
                stable = _fx_verdict(fx["study"], fx["R"], fx["C"], name, "stable", case["stable"]) \
                    if case["stable"] is not None else None
                cand = _fx_verdict(fx["study"], fx["R"], fx["C"], name, "candidate", case["candidate"])
                built.append((stable, cand))
            before = _snap_tree(td)
            rows = []
            for i, ((stable, cand), case) in enumerate(zip(built, cases)):
                rows.append(promote(fx["study"], cand, stable, fx["C"],
                                    None if stable is None else stable["eval_key"],
                                    _fx_ctx(i, case["name"]), case["repeat_band"],
                                    fx["gates"], fx["gates_lock"], fx["registry"]))
            assert before == _snap_tree(td), "promote wrote to the study or the registry"
            digests.append(common.sha256_of(rows))
    assert digests[0] == digests[1], digests
    print("[ok] determinism: two runs in two fresh temp dirs give byte-identical rows (sha %s...)"
          " and promote writes nothing anywhere" % (digests[0][:8],))


def selftest() -> None:
    """The CAD-16 gate: T1-T12, one [ok] line each, SELFTEST PASS at the end."""
    _t1()
    rows = _t2()
    _t3(rows)
    _t4()
    _t5()
    _t6()
    _t7()
    _t8()
    _t9()
    _t10()
    _t11()
    _t12()
    print("SELFTEST PASS")


def main(argv) -> int:
    """The CLI: --selftest, --help; gate.py reads only, so it has no other verb."""
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
    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
