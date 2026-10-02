#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""loop.py - the study loop of stage S11 (docs/16 §D, §I CAD-18): from locked requirements to a
confirmed stable design, every evaluation built in a temp dir and moved whole into cache/<eval_key>,
promotion one atomic replace of the params/stable.json pointer, resume by one deterministic walk.

The walk is the ONLY writer: a row already on disk must equal the row the walk re-derives
(LOOP-RESUME), the frontier is where the walk appends, and replay is the same walk in verify-only
mode. The candidate/commit split with backups and the filesystem resume are ideas from ai-cad
(Apache-2.0, ADR 0006, ADR 0012) and Amagine3D (Apache-2.0, e608dc6: server/model-parameters.ts
`rebuildModelWithParameters`/`promoteFiles`, intent_revision.py write-once lineage), reimplemented
from reading only; no code copied (docs/16 §E.6-§E.8, §F, docs/16a §B.1, §F, §H).

Usage:
  python loop.py --selftest
  python loop.py --help
  python loop.py init STUDY_DIR REQUIREMENTS_DIR START_JSON [--registry PATH]
  python loop.py run STUDY_DIR [--registry PATH] [--unattended]
  python loop.py status STUDY_DIR [--registry PATH]
  python loop.py replay STUDY_DIR [--registry PATH]
  python loop.py intake STUDY_DIR EDIT_JSON [--registry PATH]
  python loop.py launch STUDY_DIR [--registry PATH] [--unattended]
"""

import copy
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402
import gate  # noqa: E402
import optimise_cad  # noqa: E402
import reqs  # noqa: E402
import schema  # noqa: E402
import turb_integral  # noqa: E402
import verify  # noqa: E402

REFUSAL_IDS = ("LOOP-STUDY", "LOOP-TEMPLATE", "LOOP-EVALUATOR", "LOOP-IMMUTABLE", "LOOP-LINEAGE", "LOOP-CHECKS",
               "LOOP-PREFILTER", "LOOP-CACHE", "LOOP-RESUME", "LOOP-BUSY", "LOOP-CLOSED", "LOOP-MAXEVALS")
INTAKE_IDS = ("CAD-EDIT", "CAD-LOCKED", "GATE-STALE", "CAD-INTENT", "CAD-RANGE")   # checked in this order
KILL_POINTS = ("after_propose_row", "after_build", "after_replace", "after_iteration_row", "after_gate_row",
               "after_history")
KILL_EXIT = 77
STATUSES = ("new", "running", "paused", "confirmed", "confirmation_failed", "infeasible")
EVALUATORS = ("stub",)
STUB_VERSION = "1"
STUB_ENV = {"stub": STUB_VERSION}
STUB_CASE_WRITER = "stub/1"
STUB_REPEAT_BAND = 5e-4          # m of objective: the gate's repeat band and the ladder's band for the stub study
STUB_CFD_U = {"gci_fine": 1e-3, "repeat_band": 5e-4}   # given for every check with repr "cfd"
STUB_L2_REL = 1e-3               # mach_max at L2 is the L1 value times (1 + STUB_L2_REL)
STUB_SLOPE_K = {"poly3": 1.5, "poly5": 1.875, "poly7": 2.1875}   # cubic_matched: 0.75 / min(x_m, 1 - x_m)
STUDY_KEYS = ("schema", "study_id", "evaluator", "repeat_band", "start")          # study.json, schema "cad-study/1"
START_KEYS = ("params", "provenance")
EVAL_KEYS = ("schema", "eval_key", "parts", "level", "evaluator", "stage_reached", "solve_class", "files")
                                 # cache/<ek>/eval.json, schema "cad-eval/1"
POINTER_KEYS = ("schema", "study_id", "eval_key", "params_sha", "decision_n", "iteration_n")
                                 # params/stable.json and history/<n>/stable.json, schema "cad-stable/1"
EDIT_KEYS = ("schema", "study_id", "base_stable_eval_key", "edits", "reason", "card")   # schema "cad-edit/1"
CARD_KEYS = ("approved_by", "params")
_KILL = None                     # selftest hook only: (point name, k) - the k-th live hit of that point calls os._exit(KILL_EXIT)
EVALUATOR_CALLS = [0]            # process-wide count of evaluator invocations (the "work" counter)
INTAKE_RE = re.compile("^intake edit ([0-9a-f]{64})")
SHA_RE = reqs.SHA_RE
PF_JSON = "prefilter.jsonl"
DEC_JSON = "decisions.jsonl"
IT_JSON = "iterations.jsonl"
GENESIS_KEYS = gate.GENESIS_KEYS
REGISTRY_KEYS = gate.REGISTRY_KEYS
USAGE = ("usage: python loop.py --selftest" + chr(10)
         + "       python loop.py --help" + chr(10)
         + "       python loop.py init STUDY_DIR REQUIREMENTS_DIR START_JSON [--registry PATH]" + chr(10)
         + "       python loop.py run STUDY_DIR [--registry PATH] [--unattended]" + chr(10)
         + "       python loop.py status STUDY_DIR [--registry PATH]" + chr(10)
         + "       python loop.py replay STUDY_DIR [--registry PATH]" + chr(10)
         + "       python loop.py intake STUDY_DIR EDIT_JSON [--registry PATH]" + chr(10)
         + "       python loop.py launch STUDY_DIR [--registry PATH] [--unattended]")


class _Frontier(Exception):
    """The walk reached a step that needs a live action (verify-only mode)."""

    def __init__(self, last_kind, state=None):
        super().__init__("frontier")
        self.last_kind = last_kind
        self.state = state or {}


class _CacheMiss(Exception):
    """A cache entry an existing eval row names is not on disk (verify-only mode)."""

    def __init__(self, ek):
        super().__init__("cache missing")
        self.ek = ek


class _Mismatch(Exception):
    """A row on disk does not equal the row the walk re-derived."""

    def __init__(self, where):
        super().__init__(where)
        self.where = where


_KILL_HITS = {}


def _kill(point) -> None:
    """The selftest kill point: the KILL-th live hit of POINT exits hard, mid-action."""
    if _KILL is not None and _KILL[0] == point:
        _KILL_HITS[point] = _KILL_HITS.get(point, 0) + 1
        if _KILL_HITS[point] == _KILL[1]:
            sys.stdout.flush()
            os._exit(KILL_EXIT)


def _write_once(path, blob) -> None:
    """WRITE-ONCE (docs/16a §B.1): identical bytes accepted and left untouched, anything else
    existing on the path is LOOP-IMMUTABLE."""
    if os.path.lexists(path):
        snap = common.stable_file_snapshot(path)
        data = blob.encode("utf-8") if isinstance(blob, str) else blob
        if snap["stable"] is not True or snap["sha256"] != common.sha256_bytes(data):
            raise ValueError("LOOP-IMMUTABLE: %s already exists with other bytes"
                             % (os.path.relpath(path, os.path.dirname(os.path.dirname(path))),))
        return
    common.atomic_write(path, blob)


def _write_once_json(path, obj) -> None:
    _write_once(path, json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + chr(10))


def _drop_torn(path) -> None:
    """A torn tail line was never a row: truncate after the last newline (run/intake mode only)."""
    if not os.path.exists(path):
        return
    with open(path, "rb") as f:
        raw = f.read()
    if not raw or raw.endswith(chr(10).encode("utf-8")):
        return
    common.atomic_write(path, raw[:raw.rfind(chr(10).encode("utf-8")) + 1])


class _Log:
    """One emitter of the walk: rows already on disk are verified against what the walk re-derives,
    the frontier is where the walk appends (docs/16 §I CAD-18)."""

    def __init__(self, path, name, live):
        self.path = path
        self.name = name
        self.live = live
        self.disk = common.read_jsonl(path)
        self.pos = 0

    def emit(self, row, kill_point=None):
        """(row, appended) for the row the walk produced at its current position."""
        i = self.pos
        if i < len(self.disk):
            self.pos = i + 1
            if common.canonical_json(self.disk[i]) != common.canonical_json(row):
                raise _Mismatch("%s:%d" % (self.name, i))
            return self.disk[i], False
        if not self.live:
            raise _Frontier("pending")
        common.jsonl_append(self.path, row)
        self.pos = i + 1
        if kill_point is not None:
            _kill(kill_point)
        return row, True


def eval_parts(study, params, level, template_sha, declaration_sha, lock_sha, gates_lock) -> dict:
    """Exactly reqs.EVAL_KEY_PARTS for the stub evaluator (docs/16 §D): the level lives in the mesh
    recipe version, so an L1 and an L2 evaluation of the same params never share a key."""
    del study
    return {"template_sha": template_sha, "declaration_sha": declaration_sha, "params": params,
            "requirements_lock": lock_sha, "gates_lock": gates_lock, "env": dict(STUB_ENV),
            "mesh_recipe_version": "stub/1@" + level, "case_writer_version": STUB_CASE_WRITER,
            "bin_sha": None}


def stub_records(params, checks_doc, level) -> dict:
    """One cad-measure/1 record per check the stub knows, keyed by req_id: a documented stand-in
    (docs/16 §I CAD-18), NOT a physics model. mach_max = 0.05 K_max Re_De makes SYS-MACH <= 0.3 the
    same inequality as CAD-17's stand-in guard kre <= 6."""
    di, cr = float(params["D_i"]), float(params["CR"])
    de = di / math.sqrt(cr)
    law = params["law"]
    length = float(params["L_over_Di"]) * di
    xm = None if params["x_m"] is None else float(params["x_m"])
    out = {}
    for c in checks_doc["checks"]:
        prim, where = c["primitive"], list(c["args"]["where"])
        if prim == "diameter_at_plane":
            if where == ["contraction_start"]:
                v, unit, u = di, "m", 1e-9
            elif where == ["exit_plane"]:
                v, unit, u = de, "m", 1e-9
            else:
                continue
        elif prim == "area_ratio":
            v, unit, u = cr, "1", 1e-9
        elif prim == "extent_along_axis":
            v, unit, u = length + float(params["Lx_over_De"]) * de, "m", 1e-9
        elif prim == "plane_distance":
            v, unit, u = length, "m", 1e-9
        elif prim == "meridian_min_wall":
            v, unit, u = float(params["t_wall"]), "m", 1e-8
        elif prim == "slope_max":
            if law in STUB_SLOPE_K:
                k = STUB_SLOPE_K[law]
            else:
                k = 0.75 / min(xm, 1.0 - xm)
            v, unit, u = math.atan(k * (di - de) / (2.0 * length)), "rad", 1e-9
        elif prim in ("n_solids", "valid", "watertight", "axis_x", "units_m"):
            v, unit, u = 1.0, "1", None
        elif prim == "mach_max":
            kre = turb_integral.kre_max(law, cr, float(params["L_over_Di"]), xm)[0]
            v = 0.05 * kre * ((1.0 + STUB_L2_REL) if level == "L2" else 1.0)
            unit, u = "1", None
        else:
            continue
        out[c["req_id"]] = {"schema": "cad-measure/1", "primitive": prim,
                            "feature": c["args"]["feature"], "where": where, "value": v,
                            "unit": unit, "u_meas": u, "method": "stub/1", "status": "ok",
                            "reason_id": None, "detail": "CAD-18 stub evaluator"}
    return out


def stub_prefilter(params, checks_doc) -> dict:
    """The stub's CAD-only prefilter record, the same shape as optimise_cad.cad_prefilter without
    the build: the hard brep/stl rows plus the objective, judged with no CFD u and no evidence."""
    records = stub_records(params, checks_doc, "L1")
    rows, first_fail, obj_val = {}, None, None
    for c in checks_doc["checks"]:
        if not ((c["hardness"] == "hard" and c["repr"] in optimise_cad.PREFILTER_REPRS)
                or c["hardness"] == "objective"):
            continue
        v = verify.judge(c, records.get(c["req_id"]))
        if c["hardness"] == "objective":
            raw = None if v["m"] is None else v["m"]
            obj_val = float(raw) if raw is not None else None
        else:
            rows[c["req_id"]] = v
            if v["verdict"] != "pass" and first_fail is None:
                first_fail = (c["req_id"], v["verdict"])
    if first_fail is not None:
        return _pf_record(params, "refused", first_fail[0], obj_val, rows,
                          "hard row %s is %s" % (first_fail[0], first_fail[1]))
    return _pf_record(params, "pass", None, obj_val, rows,
                      "the stub's %d hard brep/stl rows pass" % (len(rows),))


def _pf_record(params, status, rule_id, objective, rows, reason) -> dict:
    """One prefilter record: exactly optimise_cad.PREFILTER_KEYS."""
    return {"params_sha": optimise_cad.params_sha(params), "status": status, "rule_id": rule_id,
            "objective": objective, "rows": rows, "reason": reason}


def stub_evaluate(params, level, checks_doc, out_dir) -> dict:
    """One stub evaluation's files into OUT_DIR: params.json, measurements.json, cfd_u.json and
    geom.json; returns {"stage_reached", "solve_class"} and counts one evaluator invocation."""
    os.makedirs(out_dir, exist_ok=True)
    records = stub_records(params, checks_doc, level)
    common.write_json(os.path.join(out_dir, "params.json"), params)
    common.write_json(os.path.join(out_dir, "measurements.json"), records)
    common.write_json(os.path.join(out_dir, "cfd_u.json"),
                      dict((c["req_id"], dict(STUB_CFD_U)) for c in checks_doc["checks"]
                           if c["repr"] == "cfd"))
    files = dict((name, common.sha256_file(os.path.join(out_dir, name)))
                 for name in ("cfd_u.json", "measurements.json", "params.json"))
    common.write_json(os.path.join(out_dir, "geom.json"),
                      {"version": "stub/1", "params_sha": optimise_cad.params_sha(params),
                       "files": files})
    EVALUATOR_CALLS[0] += 1
    return {"stage_reached": "judge", "solve_class": "not_run"}


def _cache_message(eval_key, msg) -> str:
    return "LOOP-CACHE: cache/%s: %s" % (eval_key, msg)


def _validate_full(study_dir, eval_key, checks_doc, requirements_doc) -> tuple:
    """(verdict doc, eval doc) of one cache entry, every check of docs/16a §H re-run (LOOP-CACHE):
    eval.json's keys and schema, the key identity, the exact file set, every listed file re-hashed
    through a stable snapshot, geom.json's files present in eval.json's map, and the verdict
    recomputed from the raw measurements equal to verdict.json. Never writes."""
    d = os.path.join(study_dir, "cache", eval_key)
    p_eval = os.path.join(d, "eval.json")
    if not os.path.isfile(p_eval):
        raise ValueError(_cache_message(eval_key, "eval.json is missing"))
    ev = common.read_json(p_eval)
    if not isinstance(ev, dict) or tuple(sorted(ev.keys())) != tuple(sorted(EVAL_KEYS)) \
            or ev.get("schema") != "cad-eval/1":
        raise ValueError(_cache_message(eval_key, "eval.json does not carry exactly %s with schema"
                                                " cad-eval/1" % (", ".join(EVAL_KEYS),)))
    if ev["eval_key"] != eval_key:
        raise ValueError(_cache_message(eval_key, "eval.json names eval_key %r" % (ev["eval_key"],)))
    parts = ev["parts"]
    if not isinstance(parts, dict) or reqs.eval_key(parts) != eval_key:
        raise ValueError(_cache_message(eval_key, "the eval key is not reqs.eval_key of eval.json's"
                                                  " parts"))
    files = ev["files"]
    if not isinstance(files, dict) or not files:
        raise ValueError(_cache_message(eval_key, "eval.json lists no files"))
    want = set(files) | {"eval.json"}
    have = set(n for n in os.listdir(d) if os.path.isfile(os.path.join(d, n)))
    if have != want:
        raise ValueError(_cache_message(eval_key, "the entry holds %s, want exactly %s"
                                                % (sorted(have - want) or sorted(want - have), sorted(want))))
    for name, sha in sorted(files.items()):
        snap = common.stable_file_snapshot(os.path.join(d, name))
        if snap["stable"] is not True:
            raise ValueError(_cache_message(eval_key, "%s is missing or not a stable regular file"
                                                      % (name,)))
        if snap["sha256"] != sha:
            raise ValueError(_cache_message(eval_key, "%s carries sha %s, eval.json bound %s"
                                                      % (name, snap["sha256"], sha)))
    geom = common.read_json(os.path.join(d, "geom.json"))
    if not isinstance(geom, dict) or not isinstance(geom.get("files"), dict):
        raise ValueError(_cache_message(eval_key, "geom.json has no files map"))
    for name, sha in sorted(geom["files"].items()):
        if files.get(name) != sha:
            raise ValueError(_cache_message(eval_key, "geom.json binds %s %s, eval.json %s"
                                                      % (name, sha, files.get(name))))
    measurements = common.read_json(os.path.join(d, "measurements.json"))
    cfd_u = common.read_json(os.path.join(d, "cfd_u.json"))
    evidence = {"path": "cache/" + eval_key + "/measurements.json", "sha": files["measurements.json"]}
    again = verify.evaluate(checks_doc, requirements_doc, measurements, eval_key, cfd_u, evidence)
    on_disk = common.read_json(os.path.join(d, "verdict.json"))
    if common.canonical_json(again) != common.canonical_json(on_disk):
        raise ValueError(_cache_message(eval_key, "the verdict recomputed from measurements.json and"
                                                  " cfd_u.json differs from verdict.json"))
    return on_disk, ev


def validate_cache(study_dir, eval_key, checks_doc, requirements_doc) -> dict:
    """The verdict doc of one cache entry, re-validated on every use (run, replay alike);
    any miss is LOOP-CACHE naming the entry and the check."""
    return _validate_full(study_dir, eval_key, checks_doc, requirements_doc)[0]


def _replace_dir(src, dst) -> None:
    """ONE os.replace of the built temp dir into cache/, retried like common.atomic_write."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    retries = 0
    while True:
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            retries += 1
            if retries > 5:
                raise
            time.sleep(0.05)


def evaluate_one(study_dir, params, level, checks_doc, requirements_doc, template_sha,
                 declaration_sha, gates_lock, live) -> tuple:
    """(verdict doc, eval doc, eval_key) of one evaluation (docs/16a §H): a cache hit validates and
    never touches the entry; a miss is built in tmp/<ek>/ and moved whole by ONE os.replace."""
    parts = eval_parts(None, params, level, template_sha, declaration_sha,
                       requirements_doc["lock_sha"], gates_lock)
    ek = reqs.eval_key(parts)
    cache = os.path.join(study_dir, "cache", ek)
    if os.path.isdir(cache):
        return _validate_full(study_dir, ek, checks_doc, requirements_doc) + (ek,)
    if not live:
        raise _CacheMiss(ek)
    tmp = os.path.join(study_dir, "tmp", ek)
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    stage = stub_evaluate(params, level, checks_doc, tmp)
    measurements = common.read_json(os.path.join(tmp, "measurements.json"))
    cfd_u = common.read_json(os.path.join(tmp, "cfd_u.json"))
    files = dict((name, common.sha256_file(os.path.join(tmp, name)))
                 for name in sorted(n for n in os.listdir(tmp)
                                    if os.path.isfile(os.path.join(tmp, n))))
    m_sha = files["measurements.json"]
    verdict = verify.evaluate(checks_doc, requirements_doc, measurements, ek, cfd_u,
                              {"path": "cache/" + ek + "/measurements.json", "sha": m_sha})
    common.write_json(os.path.join(tmp, "verdict.json"), verdict)
    files = dict((name, common.sha256_file(os.path.join(tmp, name)))
                 for name in sorted(n for n in os.listdir(tmp)
                                    if os.path.isfile(os.path.join(tmp, n))))
    common.write_json(os.path.join(tmp, "eval.json"),
                      {"schema": "cad-eval/1", "eval_key": ek, "parts": parts, "level": level,
                       "evaluator": "stub", "stage_reached": stage["stage_reached"],
                       "solve_class": stage["solve_class"], "files": files})
    _kill("after_build")
    _replace_dir(tmp, cache)
    _kill("after_replace")
    return verdict, common.read_json(os.path.join(cache, "eval.json")), ek


def init(study_dir, requirements_dir, start_doc, registry_path=gate.REGISTRY) -> dict:
    """The study layout of docs/16 §I CAD-18 (docs/16a §B.1): requirements + lock write-once,
    checks.json, study.json, the genesis row, the registry row, then the gate anchor. Re-init with
    identical inputs changes no byte; a changed input is refused."""
    gates, gates_lock = gate.load_gates()
    doc = reqs.read_locked(requirements_dir)                     # GATE-LOCK passes through
    decl, template_sha, declaration_sha = reqs.load_template(reqs.NOZZLE_DIR)
    if doc["template_sha"] != template_sha:
        raise ValueError("LOOP-TEMPLATE: the locked set names template_sha %r but today's"
                         " template.py hashes to %r" % (doc["template_sha"], template_sha))
    checks = reqs.compile_checks(doc, decl, declaration_sha)     # GATE-LOCK passes through
    if not isinstance(start_doc, dict) or tuple(sorted(start_doc.keys())) != \
            ("evaluator", "repeat_band", "start"):
        raise ValueError("LOOP-STUDY: the start doc must be a dict with exactly the keys"
                         " evaluator, repeat_band, start")
    evaluator, band, start = start_doc["evaluator"], start_doc["repeat_band"], start_doc["start"]
    if evaluator not in EVALUATORS:
        raise ValueError("LOOP-EVALUATOR: evaluator %r is not one of %s"
                         % (evaluator, ", ".join(EVALUATORS)))
    if not isinstance(band, (int, float)) or isinstance(band, bool) or not math.isfinite(band) \
            or band < 0:
        raise ValueError("LOOP-STUDY: repeat_band %r is not a finite number >= 0" % (band,))
    if not isinstance(start, dict) or tuple(sorted(start.keys())) != tuple(sorted(START_KEYS)) \
            or not isinstance(start["params"], dict) or not isinstance(start["provenance"], dict):
        raise ValueError("LOOP-STUDY: the start must carry exactly %s"
                         % (", ".join(START_KEYS),))
    optimise_cad.start(doc["study_id"], 0, decl, template_sha, gates_lock, checks,
                       start["params"], start["provenance"])     # CADOPT-DOC passes through
    if doc["supersedes_study"] is not None:
        old = [r for r in common.read_jsonl(registry_path)
               if isinstance(r, dict) and r.get("study_id") == doc["supersedes_study"]]
        if len(old) != 1 or old[0]["lock_sha"] != doc["supersedes_lock"]:
            raise ValueError("LOOP-LINEAGE: the registry holds %d rows for %s with its lock %r,"
                             " exactly one matching row expected"
                             % (len(old), doc["supersedes_study"], doc["supersedes_lock"]))
    os.makedirs(study_dir, exist_ok=True)
    reqs.write_locked(study_dir, doc)                            # REQ-IMMUTABLE passes through
    _write_once_json(os.path.join(study_dir, "checks.json"), checks)
    study_doc = {"schema": "cad-study/1", "study_id": doc["study_id"], "evaluator": evaluator,
                 "repeat_band": band, "start": {"params": start["params"],
                                                "provenance": start["provenance"]}}
    _write_once_json(os.path.join(study_dir, "study.json"), study_doc)
    genesis = {"kind": "genesis", "study_id": doc["study_id"], "lock_sha": doc["lock_sha"],
               "declaration_sha": doc["declaration_sha"], "template_sha": doc["template_sha"],
               "gates_lock": gates_lock}
    p_it = os.path.join(study_dir, IT_JSON)
    if not os.path.exists(p_it):
        common.jsonl_append(p_it, genesis)
    elif common.read_jsonl(p_it)[0] != genesis:
        raise ValueError("LOOP-IMMUTABLE: iterations.jsonl row 0 is not this study's genesis row")
    reg = dict((k, v) for k, v in genesis.items() if k != "kind")
    reg["supersedes_study"] = doc["supersedes_study"]
    reg["supersedes_lock"] = doc["supersedes_lock"]
    rows = common.read_jsonl(registry_path)
    same = [r for r in rows if isinstance(r, dict) and r.get("study_id") == doc["study_id"]]
    if not same:
        common.jsonl_append(registry_path, reg)
    elif len(same) == 1 and same[0] == reg:
        pass
    else:
        raise ValueError("GATE-LOCK: studies.jsonl holds %d rows for study %s"
                         % (len(same), doc["study_id"]))
    gate.lock_anchor(study_dir, gates_lock, registry_path)
    return study_doc


def _pre_walk(study_dir, registry_path):
    """Everything run, intake and replay check BEFORE the walk: the anchors, the checks file, the
    study file. Returns (doc, decl, template_sha, declaration_sha, checks, study, gates, gates_lock)."""
    gates, gates_lock = gate.load_gates()
    gate.lock_anchor(study_dir, gates_lock, registry_path)       # GATE-LOCK passes through
    doc = reqs.read_locked(study_dir)
    decl, template_sha, declaration_sha = reqs.load_template(reqs.NOZZLE_DIR)
    checks = reqs.compile_checks(doc, decl, declaration_sha)     # GATE-LOCK passes through
    on_disk = common.read_json(os.path.join(study_dir, "checks.json"))
    if common.canonical_json(on_disk) != common.canonical_json(checks):
        raise ValueError("LOOP-CHECKS: checks.json is not reqs.compile_checks of the locked"
                         " document")
    study = common.read_json(os.path.join(study_dir, "study.json"))
    if not isinstance(study, dict) or tuple(sorted(study.keys())) != tuple(sorted(STUDY_KEYS)) \
            or study["evaluator"] not in EVALUATORS \
            or study["schema"] != "cad-study/1" or study["study_id"] != doc["study_id"]:
        raise ValueError("LOOP-STUDY: study.json does not carry exactly %s with an evaluator of %s"
                         % (", ".join(STUDY_KEYS), "/".join(EVALUATORS)))
    return doc, decl, template_sha, declaration_sha, checks, study, gates, gates_lock
def _check_edit(edit, checks_doc, requirements_doc, decl, gates, stable_params, stable_key) -> None:
    """The intake verification of docs/16 §I CAD-18, first failure wins in INTAKE_IDS order; raises
    ValueError("<ID>: <message>") with the message naming what was wrong."""
    rid0 = INTAKE_IDS[0]
    if not isinstance(edit, dict):
        raise ValueError("%s: the edit is %s, not a JSON object"
                         % (rid0, type(edit).__name__))
    if tuple(sorted(edit.keys())) != tuple(sorted(EDIT_KEYS)):
        raise ValueError("%s: the edit must carry exactly %s"
                         % (rid0, ", ".join(sorted(EDIT_KEYS))))
    if edit["schema"] != "cad-edit/1":
        raise ValueError("%s: schema %r is not cad-edit/1" % (rid0, edit["schema"]))
    if edit["study_id"] != requirements_doc["study_id"]:
        raise ValueError("%s: the edit names study %r, this study is %r"
                         % (rid0, edit["study_id"], requirements_doc["study_id"]))
    base = edit["base_stable_eval_key"]
    if not isinstance(base, str) or not SHA_RE.match(base):
        raise ValueError("%s: base_stable_eval_key %r is not a 64-hex string" % (rid0, base))
    edits = edit["edits"]
    if not isinstance(edits, list) or not edits \
            or any(not isinstance(e, dict) or tuple(sorted(e.keys())) != ("pointer", "value")
                   or not isinstance(e["pointer"], str) for e in edits):
        raise ValueError("%s: edits must be a non-empty list of {pointer, value} objects with a"
                         " string pointer" % (rid0,))
    seen = set()
    for e in edits:
        if e["pointer"] in seen:
            raise ValueError("%s: the pointer %r appears twice" % (rid0, e["pointer"]))
        seen.add(e["pointer"])
    if not isinstance(edit["reason"], str) or not edit["reason"]:
        raise ValueError("%s: reason must be a non-empty string" % (rid0,))
    card = edit["card"]
    if card is not None and (not isinstance(card, dict)
                             or tuple(sorted(card.keys())) != tuple(sorted(CARD_KEYS))
                             or not isinstance(card["approved_by"], str)
                             or not card["approved_by"]
                             or not isinstance(card["params"], list)
                             or any(not isinstance(p, str) for p in card["params"])):
        raise ValueError("%s: card must be null or an object with exactly %s"
                         % (rid0, ", ".join(CARD_KEYS)))
    names = set()
    locked = set()
    for row in requirements_doc["rows"]:
        if row["hardness"] == "hard":
            locked.update(row.get("locks_params") or [])
    for e in edits:
        p = e["pointer"]
        if not p.startswith("/params/") or p == "/params/" or "/" in p[len("/params/"):]:
            raise ValueError("CAD-LOCKED: the pointer %r is not exactly /params/<name>; pointers"
                             " into requirements, checks, templates or gates are refused" % (p,))
        names.add(p[len("/params/"):])
    by_name = dict((p["name"], p) for p in decl["params"])
    changed = []
    for e in edits:
        name = e["pointer"][len("/params/"):]
        if name not in by_name:
            raise ValueError("CAD-EDIT: %r is not a declared parameter of this template" % (name,))
        if e["value"] != stable_params[name]:
            changed.append((name, e["value"]))
    if not changed:
        raise ValueError("CAD-EDIT: none of the edited values differs from the stable design's"
                         " parameters")
    if base != stable_key:
        raise ValueError("GATE-STALE: base_stable_eval_key %r is not the stable design's eval key"
                         " %r (a late edit on a moved stable)" % (base, stable_key))
    carded = set(card["params"]) if card is not None else set()
    for name, _val in changed:
        if name in locked or (name == "law" and gates["llm_law_change_needs_card"]):
            if name not in carded:
                raise ValueError("CAD-INTENT: %s is intent-locked and needs a card listing it"
                                 " (a law change always needs a card)" % (name,))
    vec = dict(stable_params)
    for name, val in changed:
        vec[name] = val
    _check_range(vec, decl)


def _check_range(vec, decl) -> None:
    """CAD-RANGE (docs/16 §F item 3): the edited vector inside the template's box - the law among
    the choices, an active searched real a finite number in its [min, max], an inactive one None, a
    fixed design real exactly its min == max, an intent real finite and > 0."""
    ds = optimise_cad.design_space(decl)
    if vec["law"] not in ds["laws"]:
        raise ValueError("CAD-RANGE: law %r is not one of %s" % (vec["law"], ", ".join(ds["laws"])))
    for s in ds["searched"]:
        active = s["only_when"] is None or s["only_when"] == "law=" + vec["law"]
        v = vec[s["name"]]
        if active:
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) \
                    or not (s["min"] <= v <= s["max"]):
                raise ValueError("CAD-RANGE: %s %r is not a finite number in [%r, %r] for law %s"
                                 % (s["name"], v, s["min"], s["max"], vec["law"]))
        elif v is not None:
            raise ValueError("CAD-RANGE: %s must be null for law %s, got %r"
                             % (s["name"], vec["law"], v))
    for p in decl["params"]:
        if p["kind"] != "real":
            continue
        if p["min"] is not None and p["min"] == p["max"]:
            if vec[p["name"]] != p["min"]:
                raise ValueError("CAD-RANGE: the fixed %s must be %r, got %r"
                                 % (p["name"], p["min"], vec[p["name"]]))
        elif p["role"] == "intent":
            v = vec[p["name"]]
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) \
                    or not v > 0:
                raise ValueError("CAD-RANGE: the intent %s must be a finite number > 0, got %r"
                                 % (p["name"], v))




def _walk(study_dir, registry_path, mode, unattended, progress, intake_edit=None):
    """ONE deterministic walk from the study's start (docs/16 §I CAD-18). mode: "run" (appends),
    "verify" (intake's pre-walk; stops at the frontier), "intake" (live), "replay" (verify + record
    re-derivation). intake_edit is (sha, edit doc) for the live intake. Returns the run-style
    result dict; raises refusals."""
    live = mode in ("run", "intake")
    deep = mode == "replay"
    calls0 = EVALUATOR_CALLS[0]
    (doc, decl, template_sha, declaration_sha, checks, study, gates,
     gates_lock) = _pre_walk(study_dir, registry_path)
    if live:
        shutil.rmtree(os.path.join(study_dir, "tmp"), ignore_errors=True)
        for name in (DEC_JSON, IT_JSON, PF_JSON):
            _drop_torn(os.path.join(study_dir, name))
    it = _Log(os.path.join(study_dir, IT_JSON), IT_JSON, live)
    dec = _Log(os.path.join(study_dir, DEC_JSON), DEC_JSON, live)
    band = study["repeat_band"]
    if not it.disk or it.disk[0].get("kind") != "genesis":
        raise ValueError("GATE-LOCK: no genesis row in iterations.jsonl")
    it.pos = 1
    rec = {"history": [], "seen_params": set(), "eval_keys": [], "stable": None, "pointer": None,
           "since_promote": 0, "llm_rejected": 0, "llm_off": False, "consulted": False,
           "n_evals": 0, "last_candidate": None, "it_eval_n": None, "n_eval_rows": 0,
           "last_kind": "not_rest", "closed": False, "hist_writes": [], "reconciled": False,
           "stop": None}
    r = rec

    def progress_line(kind, row, appended):
        if progress is None or not appended:
            return
        if kind == "eval":
            progress("eval %d %s %s %s objective %r" % (row["n"], row["origin"], row["level"],
                                                        row["design_verdict"], row["objective"]))
        else:
            progress("decision %d %s %s" % (row["n"], row["decision"], row["rule_id"]))

    def _emit(log, row, kill_point=None):
        """log.emit with the walk's rest-point marker attached to a frontier stop."""
        try:
            return log.emit(row, kill_point)
        except _Frontier:
            raise _Frontier(r["last_kind"], {"n_evals": r["n_evals"],
                                             "llm_rejected": r["llm_rejected"],
                                             "llm_off": r["llm_off"]})

    def gated_evaluation(params, origin, level, base_key, pdoc, law, sobol_index=None, ei=None):
        """One gated evaluation of a proposal (docs/16 §I CAD-18): proposal.json, evaluate_one, the
        iteration row, gate.promote and the pointer swap. Confirmation origin skips the gate. The
        proposal write, the history write, the after_history kill point and the pointer replace are
        live only where the row they precede is new (appended in this call) - a row verified from
        disk causes no write and hits no kill point."""
        params_sha = optimise_cad.params_sha(params)
        new_eval = it.pos >= len(it.disk)    # the iteration row is at the frontier: this is new
        if not live and new_eval:
            raise _Frontier(r["last_kind"], {"n_evals": r["n_evals"],
                                             "llm_rejected": r["llm_rejected"],
                                             "llm_off": r["llm_off"]})
                                             # a new evaluation: proposal write + build are live
        if pdoc is not None and live and new_eval:
            common.write_json(os.path.join(study_dir, "params", "proposal.json"), pdoc)
        verdict, ev, ek = evaluate_one(study_dir, params, level, checks, doc, template_sha,
                                       declaration_sha, gates_lock, live)
        t = verify.tally(verdict)
        cache_hit = ek in r["eval_keys"]
        row = {"schema": "cad-iteration/1", "kind": "eval", "study_id": doc["study_id"],
               "n": r["n_eval_rows"], "eval_key": ek, "params_sha": params_sha,
               "template_id": doc["template_id"], "template_sha": doc["template_sha"],
               "requirements_lock": doc["lock_sha"], "gates_lock": gates_lock,
               "env_sha": common.sha256_of(eval_parts(None, params, level, template_sha,
                                                      declaration_sha, doc["lock_sha"],
                                                      gates_lock)["env"]),
               "level": level, "origin": origin,
               "stage_reached": ev["stage_reached"], "solve_class": ev["solve_class"],
               "design_verdict": verdict["design_verdict"], "n_hard_pass": t["n_hard_pass"],
               "n_hard_fail": t["n_hard_fail"], "n_hard_ne": t["n_hard_ne"],
               "objective": t["objective"], "objective_u": t["objective_u"],
               "verdict_sha": common.sha256_file(os.path.join(study_dir, "cache", ek,
                                                              "verdict.json")),
               "cache_hit": cache_hit}
        errs = schema.errors(row, "cad-iteration/1")
        if errs:
            raise RuntimeError("loop: the walk built an invalid cad-iteration/1 row: %s" % (errs[0],))
        got, appended = _emit(it, row, "after_iteration_row")
        r["n_eval_rows"] += 1
        r["it_eval_n"] = got["n"]
        r["eval_keys"].append(ek)
        r["last_candidate"] = verdict
        if origin != "confirmation":
            r["n_evals"] += 1
            if params_sha not in r["seen_params"]:
                r["seen_params"].add(params_sha)
                r["history"].append({"params_sha": params_sha, "params": params,
                                     "verdict": verdict})
        progress_line("eval", got, appended)
        if origin == "confirmation":
            return verdict, ek, got
        ctx = {"n": dec.pos, "after_iteration": got["n"], "params_sha": params_sha, "law": law,
               "sobol_index": sobol_index, "ei": ei}
        g = gate.promote(study_dir, verdict, r["stable"], checks, base_key, ctx, band,
                         gate.GATES, gate.GATES_LOCK, registry_path)
        got, appended = _emit(dec, g, "after_gate_row")
        r["since_promote"] += 1
        if g["decision"] == "promote":
            r["since_promote"] = 0
            r["consulted"] = False
            new_ptr = {"schema": "cad-stable/1", "study_id": doc["study_id"], "eval_key": ek,
                       "params_sha": params_sha, "decision_n": g["n"], "iteration_n": got["n"]}
            if r["pointer"] is not None:
                r["hist_writes"].append((g["n"], r["pointer"]))
                if appended:                 # live only: an appended promote row writes and kills
                    _write_once_json(os.path.join(study_dir, "history", str(g["n"]),
                                                  "stable.json"), r["pointer"])
                    _kill("after_history")
            if appended:                     # the pointer replace, never a re-walk side effect
                common.write_json(os.path.join(study_dir, "params", "stable.json"), new_ptr)
            r["pointer"] = new_ptr
            r["stable"] = verdict
        elif g["decision"] == "reject" and origin == "llm_edit":
            r["llm_rejected"] += 1
        r["last_kind"] = "rest" if origin == "llm_edit" else "not_rest"
        progress_line("decision", got, appended)
        return verdict, ek, got

    def dec_row(decision, rule_id, reason, cand_eval_key=None, params_sha=None, law=None,
                sobol_index=None, ei=None, deltas=None, base_stable_override=None):
        """One cad-decision/1 row the loop builds itself, schema-checked (a schema error is a bug)."""
        stable_ek = None if r["pointer"] is None else r["pointer"]["eval_key"]
        row = {"schema": "cad-decision/1", "study_id": doc["study_id"], "n": dec.pos,
               "after_iteration": r["it_eval_n"], "decision": decision, "rule_id": rule_id,
               "stable_eval_key": stable_ek, "candidate_eval_key": cand_eval_key,
               "params_sha": params_sha, "law": law, "sobol_index": sobol_index, "ei": ei,
               "reason": reason, "gates_lock": gates_lock, "requirements_lock": doc["lock_sha"],
               "base_stable_eval_key": stable_ek if base_stable_override is None
               else base_stable_override}
        for k in gate.DELTA_KEYS:
            row[k] = (deltas or {}).get(k, [])
        errs = schema.errors(row, "cad-decision/1")
        if errs:
            raise RuntimeError("loop: the walk built an invalid cad-decision/1 row: %s" % (errs[0],))
        return row

    def reconcile():
        """Make history/ and params/stable.json match the rows (run and intake only): history files
        write-once, the current pointer rewritten if it differs or is missing."""
        if not live:
            return
        for n_hist, prev_ptr in r["hist_writes"]:
            _write_once_json(os.path.join(study_dir, "history", str(n_hist), "stable.json"), prev_ptr)
        if r["pointer"] is not None:
            p_cur = os.path.join(study_dir, "params", "stable.json")
            want = (json.dumps(r["pointer"], indent=2, ensure_ascii=False, allow_nan=False)
                    + chr(10)).encode("utf-8")
            need = True
            if os.path.isfile(p_cur):
                snap = common.stable_file_snapshot(p_cur)
                need = not (snap["stable"] is True and snap["sha256"] == common.sha256_bytes(want))
            if need:
                common.write_json(p_cur, r["pointer"])
        r["reconciled"] = True

    def first_frontier():
        """The pointer reconcile the walk owes when it first reaches the frontier."""
        if not r["reconciled"]:
            reconcile()

    def on_appended(appended):
        """The walk owes its pointer reconcile at the first append: every promote row passed so
        far - verified from disk as well as appended - is then covered, repairing a kill between
        a gate row and its pointer replace. A re-walk that appends nothing reconciles only at the
        frontier (finalise), where stable.json == the current pointer is the invariant."""
        if live and appended:
            first_frontier()

    # ---- the prefilter rows (docs/16 §I CAD-18): pool order, CAND_KEYS equal, replay re-derives
    fixed = dict((name, study["start"]["params"][name])
                 for name in optimise_cad.design_space(decl)["fixed"])
    cands = optimise_cad.pool(study["study_id"], decl, fixed, gates["sobol_pool"])
    pf_path = os.path.join(study_dir, PF_JSON)
    pf_disk = common.read_jsonl(pf_path)
    if len(pf_disk) > len(cands):
        raise ValueError("LOOP-PREFILTER: prefilter.jsonl holds %d rows against a pool of %d"
                         % (len(pf_disk), len(cands)))
    for i, cand in enumerate(cands):
        rec_i = None
        if i < len(pf_disk):
            row = pf_disk[i]
            if not isinstance(row, dict) or tuple(sorted(row.keys())) != \
                    tuple(sorted(optimise_cad.CAND_KEYS + ("prefilter",))) \
                    or any(row[k] != cand[k] for k in optimise_cad.CAND_KEYS):
                raise ValueError("LOOP-PREFILTER: prefilter.jsonl:%d is not pool candidate %d"
                                 % (i, i))
            if deep:
                rec_i = stub_prefilter(cand["params"], checks)
                if common.canonical_json(rec_i) != common.canonical_json(row["prefilter"]):
                    raise _Mismatch("%s:%d" % (PF_JSON, i))
            rec_i = rec_i or row["prefilter"]
        else:
            if not live:
                raise _Frontier(r["last_kind"])
            rec_i = stub_prefilter(cand["params"], checks)
            common.jsonl_append(pf_path, dict([(k, cand[k]) for k in optimise_cad.CAND_KEYS]
                                              + [("prefilter", rec_i)]))
            on_appended(True)
    pf_map = {}
    for i, cand in enumerate(cands):
        if i < len(pf_disk):
            pf_map[cand["params_sha"]] = pf_disk[i]["prefilter"]
        elif mode == "run":
            pf_map[cand["params_sha"]] = stub_prefilter(cand["params"], checks)

    def _stable_params():
        if r["pointer"] is None:
            return None
        sha = r["pointer"]["params_sha"]
        for h in r["history"]:
            if h["params_sha"] == sha:
                return h["params"]
        raise RuntimeError("loop: the stable pointer names %s, absent from the history" % (sha,))

    def apply_edit(edit, sha):
        """The intake of docs/16 §D S11 and §E.8: verify in INTAKE_IDS order, record a reject row or
        run the edit as an llm_edit evaluation. Returns True when the edit was accepted."""
        base = edit.get("base_stable_eval_key") if isinstance(edit, dict) else None
        base_ok = isinstance(base, str) and SHA_RE.match(base) and len(base) == 64
        try:
            _check_edit(edit, checks, doc, decl, gates, _stable_params(),
                        None if r["pointer"] is None else r["pointer"]["eval_key"])
        except ValueError as e:
            msg = str(e)
            rid = msg.split(":", 1)[0]
            body = msg.split(":", 1)[1].strip()
            row = dec_row("reject", rid, "intake edit %s: %s" % (sha, body),
                          base_stable_override=base if base_ok else None)
            got, appended = _emit(dec, row)
            r["llm_rejected"] += 1
            r["last_kind"] = "rest"
            r["status_after_intake"] = "paused"
            progress_line("decision", got, appended)
            return False
        pairs = [(e["pointer"][len("/params/"):], e["value"]) for e in edit["edits"]]
        changed = sorted(name for name, val in pairs if val != _stable_params()[name])
        vec = dict(_stable_params())
        for name, val in pairs:
            vec[name] = val
        prov = optimise_cad.optimiser_provenance(decl)
        for name in changed:
            prov[name] = "llm_choice"
        pdoc = optimise_cad.params_doc(vec, decl, template_sha, doc["lock_sha"],
                                       edit["base_stable_eval_key"], prov)
        row = dec_row("propose", "CAD-INTAKE",
                      "intake edit %s: sets %s" % (sha, ", ".join(changed)),
                      params_sha=optimise_cad.params_sha(vec), law=vec["law"],
                      base_stable_override=edit["base_stable_eval_key"])
        got, appended = _emit(dec, row, "after_propose_row")
        r["last_kind"] = "not_rest"
        progress_line("decision", got, appended)
        on_appended(appended)
        gated_evaluation(vec, "llm_edit", gates["loop_level"], edit["base_stable_eval_key"], pdoc, vec["law"])
        r["status_after_intake"] = "paused"
        return True

    # ---- (a) the intake replay of docs/16 §I CAD-18: re-apply stored intakes at rest points
    def reapply():
        while dec.pos < len(dec.disk):
            nxt = dec.disk[dec.pos]
            m = INTAKE_RE.match(nxt.get("reason", "")) if isinstance(nxt, dict) else None
            if not m:
                return
            sha = m.group(1)
            p_edit = os.path.join(study_dir, "intake", sha + ".json")
            if not os.path.isfile(p_edit):
                raise ValueError("LOOP-RESUME: decisions.jsonl:%d names intake edit %s but"
                                 " intake/%s.json is missing" % (dec.pos, sha, sha))
            apply_edit(common.read_json(p_edit), sha)


    def _ladder_state(best_ei):
        feasible = r["stable"] is not None and r["stable"]["design_verdict"] == "feasible"
        return {"n_evals": r["n_evals"], "since_promote": r["since_promote"],
                "llm_rejected": r["llm_rejected"], "llm_off": r["llm_off"],
                "stable_feasible": feasible, "best_ei": best_ei, "band": band}

    def emit_stop(rule_id, ei_value):
        if rule_id == "GATE-FEASIBLE":
            reason = ("the stable design is feasible and the proposal's expected improvement %r is"
                      " below the band %r" % (ei_value, band))
        else:
            reason = ("the study reached its %d evaluations at %s"
                      % (gates["max_evals"], gates["loop_level"]))
        got, appended = _emit(dec, dec_row("stop", rule_id, reason, ei=ei_value))
        r["last_kind"] = "not_rest"
        progress_line("decision", got, appended)
        on_appended(appended)

    def finalise():
        """docs/16 §E.7: the stable design feasible means the L2 confirmation, else infeasible."""
        first_frontier()
        r["final_status"] = "infeasible"
        if r["stable"] is not None and r["stable"]["design_verdict"] == "feasible":
            params_s = _stable_params()
            verdict2, ek2, _row2 = gated_evaluation(params_s, "confirmation", gates["confirm_level"],
                                           r["pointer"]["eval_key"], None, params_s["law"])
            cfd_u = common.read_json(os.path.join(study_dir, "cache", ek2, "cfd_u.json"))
            conf = gate.confirm(verdict2, checks, doc, cfd_u)
            if conf["decision"] == "confirm_pass":
                reason = "every hard row passes at L2 with its margin above the CFD noise"
            else:
                reason = "the L2 confirmation fails on %s" % (", ".join(conf["failing"]),)
            stable_ek = r["pointer"]["eval_key"]
            row = dec_row(conf["decision"], gate.CONFIRM_ID, reason, cand_eval_key=ek2,
                          params_sha=r["pointer"]["params_sha"], law=params_s["law"],
                          deltas=gate.deltas(verdict2, r["stable"]),
                          base_stable_override=stable_ek)
            row["stable_eval_key"] = stable_ek
            got, appended = _emit(dec, row)
            r["closed"] = True
            r["final_status"] = "confirmed" if conf["decision"] == "confirm_pass" \
                else "confirmation_failed"
            progress_line("decision", got, appended)
            return conf["decision"]
        r["closed"] = True
        return "infeasible"

    def result(status):
        return {"status": status, "n_evals": r["n_evals"], "n_decisions": dec.pos,
                "stable_eval_key": None if r["pointer"] is None else r["pointer"]["eval_key"],
                "evaluator_calls": EVALUATOR_CALLS[0] - calls0}

    # ---- the walk itself (docs/16 §I CAD-18): the start, then the ladder / propose / gate loop
    pdoc0, row0 = optimise_cad.start(doc["study_id"], dec.pos, decl, template_sha, gates_lock,
                                     checks, study["start"]["params"], study["start"]["provenance"])
    got, appended = _emit(dec, row0, "after_propose_row")
    r["last_kind"] = "not_rest"
    progress_line("decision", got, appended)
    on_appended(appended)
    gated_evaluation(study["start"]["params"], "start", gates["loop_level"], None, pdoc0, row0["law"])
    while True:
        reapply()
        if r["closed"]:
            break
        lad = gate.ladder(_ladder_state(None), gates)
        if lad is not None and lad[0] == "stop":
            emit_stop(lad[1], None)
            finalise()
            return result(r["final_status"])
        if lad is not None and lad[0] == "llm_off":
            got, appended = _emit(dec, dec_row("llm_off", lad[1],
                                             "the LLM is off for this study after %d rejected"
                                             " edits" % (r["llm_rejected"],)))
            r["llm_off"] = True
            r["last_kind"] = "not_rest"
            progress_line("decision", got, appended)
            on_appended(appended)
            continue
        if lad is not None and lad[0] == "llm_consult" and not r["consulted"]:
            got, appended = _emit(dec, dec_row("llm_consult", lad[1],
                                             "the optimiser has stalled %d evaluations since the"
                                             " last promotion: consult the LLM with the requirement"
                                             " deltas" % (r["since_promote"],),
                                             deltas=gate.deltas(r["last_candidate"], r["stable"])))
            r["consulted"] = True
            r["last_kind"] = "rest"
            progress_line("decision", got, appended)
            if appended and not unattended:
                first_frontier()
                return result("paused")
            continue                       # a rest point: the intake replay runs at the loop top
        if intake_edit is not None and dec.pos >= len(dec.disk):
            # the live intake: the walk re-derived the whole rest point, the frontier check
            # agrees with the verify walk, and the edit is the next thing the study records
            if r["last_kind"] != "rest":
                raise ValueError("LOOP-BUSY: the study's frontier is not a rest point; run the"
                                 " study to its pause first")
            apply_edit(intake_edit[1], intake_edit[0])
            reconcile()
            return result(r["status_after_intake"])
        ctx = {"study_id": doc["study_id"], "n": dec.pos, "decl": decl, "template_sha": template_sha,
               "gates": gates, "gates_lock": gates_lock, "checks_doc": checks,
               "objective": doc["objective"], "pool": cands, "prefilter": pf_map,
               "history": r["history"], "stable": r["stable"]}
        pdoc, row = optimise_cad.propose(ctx)
        got, appended = _emit(dec, row, "after_propose_row")
        r["last_kind"] = "not_rest"
        progress_line("decision", got, appended)
        on_appended(appended)
        if row["decision"] == "abstain":
            finalise()
            return result(r["final_status"])
        lad2 = gate.ladder(_ladder_state(row["ei"]), gates)
        if lad2 == ("stop", "GATE-FEASIBLE"):
            emit_stop(lad2[1], row["ei"])
            finalise()
            return result(r["final_status"])
        cand = [c for c in cands if c["params_sha"] == row["params_sha"]][0]
        base = None if r["pointer"] is None else r["pointer"]["eval_key"]
        gated_evaluation(cand["params"], "init_design" if row["rule_id"] == "CADOPT-INIT" else "optimiser",
                gates["loop_level"], base, pdoc, row["law"], row["sobol_index"], row["ei"])
    reconcile()
    return result(r.get("final_status", "infeasible"))


def _walk_resume(study_dir, registry_path, mode, unattended, progress, intake_edit=None) -> dict:
    """_walk for the live verbs (run, run_unattended, intake): a row on disk that the walk
    re-derives differently is the refusal LOOP-RESUME <file>:<index> (docs/16 §I CAD-18); only
    replay reports the position as first_mismatch."""
    try:
        return _walk(study_dir, registry_path, mode, unattended, progress, intake_edit)
    except _Mismatch as m:
        raise ValueError("LOOP-RESUME: %s" % (m.where,))


def run(study_dir, registry_path=gate.REGISTRY, unattended=False, progress=None) -> dict:
    """Walk the study to a rest point or the end (docs/16 §I CAD-18); resume is simply run again."""
    return _walk_resume(study_dir, registry_path, "run", unattended, progress)


def run_unattended(study_dir, registry_path=gate.REGISTRY, progress=None) -> dict:
    """run(..., unattended=True): the ladder's llm_consult pause does not stop the walk."""
    return _walk_resume(study_dir, registry_path, "run", True, progress)


def intake(study_dir, edit_path, registry_path=gate.REGISTRY, progress=None) -> dict:
    """The LLM edit intake (docs/16 §D S11, §E.8): verify-only walk to the frontier, the rest-point
    and LLM-off checks, the edit file stored write-once by content sha, then the live intake; a
    refusal is a recorded reject row, not an exception."""
    with open(edit_path, "rb") as f:
        raw = f.read()
    sha = common.sha256_bytes(raw)
    try:
        res = _walk_resume(study_dir, registry_path, "verify", True, None)
    except _Frontier as f:
        if f.last_kind != "rest":
            raise ValueError("LOOP-BUSY: the study's frontier is not a rest point; run the study"
                             " to its pause first")
        gates, _gl = gate.load_gates()
        if f.state["llm_off"] or f.state["llm_rejected"] >= gates["llm_reject_k"]:
            raise ValueError("GATE-LLMOFF: the LLM is off for this study after %d rejected edits"
                             % (f.state["llm_rejected"],))
        if f.state["n_evals"] >= gates["max_evals"]:
            raise ValueError("LOOP-MAXEVALS: the study has spent its %d L1 evaluations"
                             % (gates["max_evals"],))
    except _CacheMiss as m:
        raise ValueError(_cache_message(m.ek, "the cache entry of an existing eval row is missing"))
    else:
        raise ValueError("LOOP-CLOSED: the study is closed with status %r" % (res["status"],))
    try:
        edit = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        edit = None                          # recorded as a CAD-EDIT reject row by the live walk
    _write_once(os.path.join(study_dir, "intake", sha + ".json"), raw)
    return _walk_resume(study_dir, registry_path, "intake", True, progress,
                        intake_edit=(sha, edit))


def status(study_dir, registry_path=gate.REGISTRY) -> dict:
    """Read rows and files only (no walk, no evaluator) and name the study's status."""
    gates, _gl = gate.load_gates()
    doc = reqs.read_locked(study_dir)
    dec = common.read_jsonl(os.path.join(study_dir, DEC_JSON))
    it = common.read_jsonl(os.path.join(study_dir, IT_JSON))
    n_evals = sum(1 for row in it if isinstance(row, dict) and row.get("kind") == "eval"
                  and row.get("level") == gates["loop_level"])
    stable_key = stable_dv = stable_obj = None
    for row in dec:
        if isinstance(row, dict) and row.get("decision") == "promote":
            stable_key = row.get("candidate_eval_key")
    if stable_key is not None:
        p_verdict = os.path.join(study_dir, "cache", stable_key, "verdict.json")
        if os.path.isfile(p_verdict):
            v = common.read_json(p_verdict)
            stable_dv = v["design_verdict"]
            stable_obj = verify.tally(v)["objective"]
    last = dec[-1] if dec else None
    last_decision = None if last is None else {"decision": last["decision"],
                                               "rule_id": last["rule_id"]}
    st = "running"
    if last is None:
        st = "new"
    elif last["decision"] == "confirm_pass":
        st = "confirmed"
    elif last["decision"] == "confirm_fail":
        st = "confirmation_failed"
    elif last["decision"] in ("stop", "abstain"):
        st = "infeasible" if stable_dv in (None, "infeasible", "not_evaluable") else "running"
    elif last["decision"] == "llm_consult" \
            or (last["decision"] == "reject" and INTAKE_RE.match(last.get("reason", ""))):
        st = "paused"
    elif last["decision"] in ("promote", "reject") and len(dec) >= 2:
        prev_eval = [x for x in it if isinstance(x, dict) and x.get("kind") == "eval"
                     and x.get("n") == last.get("after_iteration")]
        if prev_eval and prev_eval[-1].get("origin") == "llm_edit":
            st = "paused"
    return {"study_id": doc["study_id"], "status": st, "n_evals": n_evals, "n_decisions": len(dec),
            "stable_eval_key": stable_key, "stable_design_verdict": stable_dv,
            "stable_objective": stable_obj, "last_decision": last_decision}


def _n_rows(study_dir, name) -> int:
    return len(common.read_jsonl(os.path.join(study_dir, name)))


def replay(study_dir, registry_path=gate.REGISTRY) -> dict:
    """Re-derive every row from the study's own inputs (docs/16 §I CAD-17's discipline): a cache
    failure is first_mismatch "cache:<eval_key>", a row mismatch "<file>:<index>"; every other
    refusal raises."""
    try:
        res = _walk(study_dir, registry_path, "replay", True, None)
    except _CacheMiss as m:
        return {"ok": False, "n_iterations": _n_rows(study_dir, IT_JSON),
                "n_decisions": _n_rows(study_dir, DEC_JSON), "first_mismatch": "cache:" + m.ek}
    except _Mismatch as m:
        return {"ok": False, "n_iterations": _n_rows(study_dir, IT_JSON),
                "n_decisions": _n_rows(study_dir, DEC_JSON), "first_mismatch": m.where}
    except _Frontier:
        return {"ok": True, "n_iterations": _n_rows(study_dir, IT_JSON),
                "n_decisions": _n_rows(study_dir, DEC_JSON), "first_mismatch": None}
    except ValueError as e:
        msg = str(e)
        if msg.startswith("LOOP-CACHE: cache/"):
            return {"ok": False, "n_iterations": _n_rows(study_dir, IT_JSON),
                    "n_decisions": _n_rows(study_dir, DEC_JSON),
                    "first_mismatch": "cache:" + msg.split("cache/", 1)[1].split(":", 1)[0]}
        raise
    del res
    return {"ok": True, "n_iterations": _n_rows(study_dir, IT_JSON),
            "n_decisions": _n_rows(study_dir, DEC_JSON), "first_mismatch": None}


def console_argv(study_dir, registry_path=gate.REGISTRY, unattended=False) -> list:
    """The visible-console command of docs/16 §I CAD-18: python loop.py run <abs study> --registry
    <abs registry> [--unattended]."""
    argv = [sys.executable, os.path.abspath(__file__), "run", os.path.abspath(study_dir),
            "--registry", os.path.abspath(registry_path)]
    if unattended:
        argv.append("--unattended")
    return argv


def main(argv) -> int:
    """The CLI: each verb prints its result dict as one canonical JSON line and exits 0; a refusal
    prints "<ID>: <message>" to stderr and exits 1; bad usage exits 2."""
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
    want = {"init": 3, "run": 1, "status": 1, "replay": 1, "intake": 2, "launch": 1}
    if not argv or argv[0] not in want:
        print(USAGE, file=sys.stderr)
        return 2
    verb, rest = argv[0], argv[1:]
    pos, registry, unattended = [], gate.REGISTRY, False
    i = 0
    while i < len(rest):
        if rest[i] == "--registry" and i + 1 < len(rest):
            registry = rest[i + 1]
            i += 2
        elif rest[i] == "--unattended":
            unattended = True
            i += 1
        else:
            pos.append(rest[i])
            i += 1
    if len(pos) != want[verb]:
        print(USAGE, file=sys.stderr)
        return 2
    try:
        if verb == "init":
            res = init(pos[0], pos[1], common.read_json(pos[2]), registry)
        elif verb == "run":
            res = run(pos[0], registry, unattended, lambda line: print(line, flush=True))
        elif verb == "status":
            res = status(pos[0], registry)
        elif verb == "replay":
            res = replay(pos[0], registry)
            if not res["ok"]:
                print(common.canonical_json(res))
                return 1
        elif verb == "intake":
            res = intake(pos[0], pos[1], registry, lambda line: print(line, flush=True))
        else:
            proc = subprocess.Popen(console_argv(pos[0], registry, unattended),
                                    creationflags=subprocess.CREATE_NEW_CONSOLE
                                    if os.name == "nt" else 0)
            print(common.canonical_json({"pid": proc.pid}))
            return 0
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1
    print(common.canonical_json(res))
    return 0


def _fx_doc() -> dict:
    """The v1 golden's requirements relocked against today's template module (a fixture relock)."""
    doc = copy.deepcopy(common.read_json(gate.GOLDEN_V1)["requirements"])
    doc["template_sha"] = reqs.load_template(reqs.NOZZLE_DIR)[1]
    doc["lock_sha"] = reqs.lock_sha_of(doc)
    return doc


START_DOC = {"evaluator": "stub", "repeat_band": STUB_REPEAT_BAND,
             "start": {"params": dict(optimise_cad.STANDIN_START),
                       "provenance": dict(optimise_cad.STANDIN_START_PROVENANCE)}}


def _fx_study(td) -> str:
    """A locked v1_nominal study with its own registry inside td; returns the study directory."""
    req_dir = os.path.join(td, "req")
    reqs.write_locked(req_dir, _fx_doc())
    return init(os.path.join(td, "study"), req_dir, dict(START_DOC), os.path.join(td, "studies.jsonl")) \
        and os.path.join(td, "study")


# ---- the CAD-18 selftest: T1-T12, one [ok] line each (docs/16 I CAD-18) ----

def _tree(root) -> dict:
    """Rel path (forward slashes) -> sha256 of every file under root; the no-byte-changed oracle."""
    out = {}
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            p = os.path.join(dirpath, fn)
            out[os.path.relpath(p, root).replace(os.sep, "/")] = common.sha256_file(p)
    return out


def _mtimes(root) -> dict:
    """Rel path -> st_mtime_ns of every file under root (nothing rewritten after validation)."""
    out = {}
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            p = os.path.join(dirpath, fn)
            out[os.path.relpath(p, root).replace(os.sep, "/")] = os.stat(p).st_mtime_ns
    return out


_REF = None                                     # (tempdir, study, registry) of the closed study
_PAUSED = None                                  # (tempdir, study, registry) of the first-run pause
_TEMPS = []                                     # every tempdir the selftest made, for the cleanup


def _mktemp_dir() -> str:
    """tempfile.mkdtemp registered for removal before selftest() returns, on success or failure."""
    td = tempfile.mkdtemp()
    _TEMPS.append(td)
    return td


def _cleanup_temps() -> None:
    """Remove every directory the selftest created and drop the module caches, so a failed run
    leaves nothing in %TEMP% either."""
    global _REF, _PAUSED, _KILLED_TD
    while _TEMPS:
        shutil.rmtree(_TEMPS.pop(), ignore_errors=True)
    _REF = _PAUSED = _KILLED_TD = None


def _reference() -> tuple:
    """The uninterrupted reference study, built once per process (the selftest's time budget)."""
    global _REF
    if _REF is None:
        td = _mktemp_dir()
        study = _fx_study(td)
        reg = os.path.join(td, "studies.jsonl")
        r1 = run(study, reg)
        assert r1["status"] == "paused", r1
        r2 = run_unattended(study, reg)
        assert r2["status"] == "confirmed", r2
        _REF = (td, study, reg)
    return _REF


def _paused() -> tuple:
    """A study stopped at its llm_consult rest point, built once per process."""
    global _PAUSED
    if _PAUSED is None:
        td = _mktemp_dir()
        study = _fx_study(td)
        reg = os.path.join(td, "studies.jsonl")
        r1 = run(study, reg)
        assert r1["status"] == "paused", r1
        _PAUSED = (td, study, reg)
    return _PAUSED


def _copy_ref(tag) -> tuple:
    """A fresh copytree of the reference study's whole temp dir (its own registry comes along)."""
    dst = os.path.join(_mktemp_dir(), tag)
    shutil.copytree(_reference()[0], dst)
    return dst, os.path.join(dst, "study"), os.path.join(dst, "studies.jsonl")


def _copy_paused(tag) -> tuple:
    dst = os.path.join(_mktemp_dir(), tag)
    shutil.copytree(_paused()[0], dst)
    return dst, os.path.join(dst, "study"), os.path.join(dst, "studies.jsonl")


def _edit_file(td, name, doc) -> str:
    p = os.path.join(td, name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")
    return p


def _edit_doc(edits, base, reason="fixture edit", card=None) -> dict:
    return {"schema": "cad-edit/1", "study_id": "v1_nominal", "base_stable_eval_key": base,
            "edits": [{"pointer": "/params/" + n, "value": v} for n, v in edits],
            "reason": reason, "card": card}


def _mach(meas) -> float:
    for row in meas.values():
        if isinstance(row, dict) and row.get("primitive") == "mach_max":
            return row["value"]
    raise AssertionError("no mach_max row in measurements")


def _refuse(fn, *args) -> ValueError:
    """Call fn(*args), expect ValueError; return the exception for the caller's id check."""
    try:
        fn(*args)
    except ValueError as e:
        return e
    raise AssertionError("%s was accepted, a refusal was expected" % (getattr(fn, "__name__", fn),))


def _t1() -> None:
    td = _mktemp_dir()
    study = _fx_study(td)
    for rel in ("checks.json", "study.json", "iterations.jsonl",
                "requirements.json", "requirements.lock"):
        assert os.path.exists(os.path.join(study, rel)), "missing layout file %s" % (rel,)
    assert os.path.isfile(os.path.join(td, "studies.jsonl"))
    it = common.read_jsonl(os.path.join(study, "iterations.jsonl"))
    assert len(it) == 1 and tuple(sorted(it[0].keys())) == tuple(sorted(gate.GENESIS_KEYS))
    reg = common.read_jsonl(os.path.join(td, "studies.jsonl"))
    assert len(reg) == 1 and tuple(sorted(reg[0].keys())) == tuple(sorted(gate.REGISTRY_KEYS))
    assert reg[0]["supersedes_study"] is None and reg[0]["supersedes_lock"] is None
    before = _tree(td)
    _fx_study(td)                                # the identical second init
    assert _tree(td) == before, "the second init changed bytes"
    sd = {"evaluator": "stub", "repeat_band": 1e-3, "start": dict(START_DOC["start"])}
    e = _refuse(init, study, os.path.join(td, "req"), sd, os.path.join(td, "studies.jsonl"))
    assert str(e).startswith("LOOP-IMMUTABLE"), str(e)
    td2 = _mktemp_dir()
    reqs.write_locked(os.path.join(td2, "req"), common.read_json(gate.GOLDEN_V1)["requirements"])
    e = _refuse(init, os.path.join(td2, "study"), os.path.join(td2, "req"), dict(START_DOC),
                os.path.join(td2, "studies.jsonl"))
    assert str(e).startswith("LOOP-TEMPLATE"), str(e)
    td3 = _mktemp_dir()
    reqs.write_locked(os.path.join(td3, "req"), _fx_doc())
    bad = {"evaluator": "cfd", "repeat_band": STUB_REPEAT_BAND, "start": dict(START_DOC["start"])}
    e = _refuse(init, os.path.join(td3, "study"), os.path.join(td3, "req"), bad,
                os.path.join(td3, "studies.jsonl"))
    assert str(e).startswith("LOOP-EVALUATOR"), str(e)
    gates, gates_lock = gate.load_gates()
    gate.lock_anchor(study, gates_lock, os.path.join(td, "studies.jsonl"))
    print("[ok] T1 init: layout, genesis/registry keys, idempotent re-init, refusals, anchor")


_T2_DECISIONS = (("propose", "CADOPT-START"), ("promote", "GATE-FIRST"), ("propose", "CADOPT-INIT"),
                 ("promote", "GATE-MORE"), ("propose", "CADOPT-INIT"), ("reject", "GATE-REGRESS"),
                 ("propose", "CADOPT-INIT"), ("reject", "GATE-TIE"), ("propose", "CADOPT-INIT"),
                 ("reject", "GATE-REGRESS"), ("propose", "CADOPT-INIT"), ("reject", "GATE-TIE"),
                 ("llm_consult", "GATE-STALL"), ("propose", "CADOPT-EI"), ("promote", "GATE-OBJ"),
                 ("propose", "CADOPT-EI"), ("promote", "GATE-OBJ"), ("propose", "CADOPT-EI"),
                 ("stop", "GATE-FEASIBLE"), ("confirm_pass", "GATE-CONFIRM"))

_T2_EVALS = (("start", "L1", "infeasible", 9, 1, 0, 0.040000, 0.6323),
             ("init_design", "L1", "feasible", 10, 0, 0, 0.069604, 0.2446),
             ("init_design", "L1", "infeasible", 9, 1, 0, 0.048705, 0.4795),
             ("init_design", "L1", "feasible", 10, 0, 0, 0.072741, 0.2965),
             ("init_design", "L1", "infeasible", 9, 1, 0, 0.065825, 0.3271),
             ("init_design", "L1", "feasible", 10, 0, 0, 0.072364, 0.2188),
             ("optimiser", "L1", "feasible", 10, 0, 0, 0.055047, 0.2700),
             ("optimiser", "L1", "feasible", 10, 0, 0, 0.050420, 0.2975),
             ("confirmation", "L2", "feasible", 10, 0, 0, 0.050420, None))


def _t2() -> None:
    td = _mktemp_dir()
    study = _fx_study(td)
    reg = os.path.join(td, "studies.jsonl")
    r1 = run(study, reg)
    assert r1["status"] == "paused" and r1["n_decisions"] == 13 and r1["n_evals"] == 6 \
        and r1["evaluator_calls"] == 6, r1
    it = common.read_jsonl(os.path.join(study, "iterations.jsonl"))
    assert r1["stable_eval_key"] == it[2]["eval_key"], "the pause's stable is not eval 1's"
    r2 = run_unattended(study, reg)
    assert r2["status"] == "confirmed" and r2["evaluator_calls"] == 3, r2
    assert r1["evaluator_calls"] + r2["evaluator_calls"] == 9
    it = common.read_jsonl(os.path.join(study, "iterations.jsonl"))
    dec = common.read_jsonl(os.path.join(study, "decisions.jsonl"))
    assert len(dec) == 20 and len(it) == 10
    assert tuple((r["decision"], r["rule_id"]) for r in dec) == _T2_DECISIONS
    for row, (origin, level, verdict, hp, hf, hne, obj, mach) in zip(it[1:], _T2_EVALS):
        assert row["kind"] == "eval" and not schema.errors(row, "cad-iteration/1"), row["n"]
        assert (row["origin"], row["level"], row["design_verdict"]) == (origin, level, verdict)
        assert (row["n_hard_pass"], row["n_hard_fail"], row["n_hard_ne"]) == (hp, hf, hne)
        assert abs(row["objective"] - obj) < 1e-6, (row["n"], row["objective"])
        meas = common.read_json(os.path.join(study, "cache", row["eval_key"], "measurements.json"))
        if mach is not None:                     # the table prints mach to 4 decimals
            assert round(_mach(meas), 4) == mach, (row["n"], _mach(meas))
    for row in dec:
        assert not schema.errors(row, "cad-decision/1"), row["n"]
    assert r2["stable_eval_key"] == it[8]["eval_key"], "stable.json is not eval 7"
    p7 = common.read_json(os.path.join(study, "cache", it[8]["eval_key"], "params.json"))
    assert p7["law"] == "poly3"
    assert abs(p7["L_over_Di"] - 0.7462969906628132) < 1e-12
    assert abs(p7["Lx_over_De"] - 0.2820980343967676) < 1e-12
    assert abs(p7["t_wall"] - 0.006150376803241671) < 1e-12
    hist = sorted(os.listdir(os.path.join(study, "history")))
    assert hist == ["14", "16", "3"], hist
    for n, replaced in zip(("3", "14", "16"), (it[1], it[2], it[7])):
        old = common.read_json(os.path.join(study, "history", n, "stable.json"))
        assert old["eval_key"] == replaced["eval_key"], (n, old["eval_key"])
    want = {"cfd_u.json", "eval.json", "geom.json", "measurements.json", "params.json", "verdict.json"}
    for row in it[1:]:
        assert set(os.listdir(os.path.join(study, "cache", row["eval_key"]))) == want, row["n"]
    assert not os.path.exists(os.path.join(study, "tmp")) or os.listdir(os.path.join(study, "tmp")) == []
    ts, ds = reqs.load_template(reqs.NOZZLE_DIR)[1], reqs.load_template(reqs.NOZZLE_DIR)[2]
    start_key = reqs.eval_key(eval_parts(None, dict(optimise_cad.STANDIN_START), "L1", ts, ds,
                                         _fx_doc()["lock_sha"], gate.load_gates()[1]))
    print("[ok] T2 oracle: 20 decisions, 9 evals, pause 13/6, confirmed 9 calls, start L1 key %s"
          % (start_key,))


def _t3() -> None:
    td = _mktemp_dir()
    study = _fx_study(td)
    reg = os.path.join(td, "studies.jsonl")
    run(study, reg)
    run_unattended(study, reg)
    rtd, rstudy, rreg = _reference()
    for name in ("iterations.jsonl", "decisions.jsonl", "params/stable.json"):
        a = common.sha256_file(os.path.join(study, name.replace("/", os.sep)))
        b = common.sha256_file(os.path.join(rstudy, name.replace("/", os.sep)))
        assert a == b, "the second study diverges on %s" % (name,)
    rr = replay(rstudy, rreg)
    assert rr == {"ok": True, "n_iterations": 10, "n_decisions": 20, "first_mismatch": None}, rr
    before = _tree(rtd)
    replay(rstudy, rreg)
    assert _tree(rtd) == before, "the replay wrote bytes"
    cd, cs, cr = _copy_ref("t3_dec")
    p = os.path.join(cs, "decisions.jsonl")
    rows = common.read_jsonl(p)
    rows[7]["reason"] = "tampered by T3"
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(common.canonical_json(r) + "\n" for r in rows))
    rr = replay(cs, cr)
    assert rr["ok"] is False and rr["first_mismatch"] == "decisions.jsonl:7", rr
    e = _refuse(run, cs, cr)                     # the same mismatch is the refusal LOOP-RESUME
    assert str(e).startswith("LOOP-RESUME: decisions.jsonl:7"), str(e)
    ep = _edit_file(cd, "e_t3.json", _edit_doc([("t_wall", 0.004)], "0" * 64))
    e = _refuse(intake, cs, ep, cr)              # the verify walk refuses before the edit is judged
    assert str(e).startswith("LOOP-RESUME: decisions.jsonl:7"), str(e)
    cd2, cs2, cr2 = _copy_ref("t3_pf")
    p = os.path.join(cs2, "prefilter.jsonl")
    rows = common.read_jsonl(p)
    rows[0]["prefilter"]["objective"] = 999.0
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(common.canonical_json(r) + "\n" for r in rows))
    rr = replay(cs2, cr2)
    assert rr["ok"] is False and rr["first_mismatch"] == "prefilter.jsonl:0", rr
    print("[ok] T3 determinism: second study identical, replay ok 10/20 silent, tamper mismatches,"
          " LOOP-RESUME on run and intake")


def _child_kill(kill, study, reg) -> int:
    """A child that imports loop, sets _KILL and walks; returns its exit code."""
    code = ("import sys; sys.path.insert(0, %r); import loop; loop._KILL = %r; "
            "loop.run_unattended(%r, %r)" % (HERE, kill, study, reg))
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    return p.returncode


_KILLED_TD = None                                # the T4 killed tempdir copytree, kept for T7/T11


def _t4() -> None:
    global _KILLED_TD
    td = _mktemp_dir()
    study = _fx_study(td)
    reg = os.path.join(td, "studies.jsonl")
    assert _child_kill(("after_build", 3), study, reg) == KILL_EXIT, "the child did not exit 77"
    cache = sorted(os.listdir(os.path.join(study, "cache")))
    assert len(cache) == 2, cache
    doc = reqs.read_locked(study)
    checks = common.read_json(os.path.join(study, "checks.json"))
    for ek in cache:
        validate_cache(study, ek, checks, doc)
    tmp = os.listdir(os.path.join(study, "tmp"))
    assert len(tmp) == 1 and tmp[0] not in cache, (tmp, cache)
    _KILLED_TD = os.path.join(_mktemp_dir(), "t4_killed")
    shutil.copytree(td, _KILLED_TD)
    rtd, rstudy, rreg = _reference()
    r = run_unattended(study, reg)
    assert r["status"] == "confirmed", r
    assert not os.path.exists(os.path.join(study, "tmp")) or os.listdir(os.path.join(study, "tmp")) == []
    for name in ("iterations.jsonl", "decisions.jsonl", "params/stable.json"):
        a = common.sha256_file(os.path.join(study, name.replace("/", os.sep)))
        b = common.sha256_file(os.path.join(rstudy, name.replace("/", os.sep)))
        assert a == b, "the resumed study diverges on %s" % (name,)
    td2 = _mktemp_dir()
    study2 = _fx_study(td2)
    reg2 = os.path.join(td2, "studies.jsonl")
    assert _child_kill(("after_gate_row", 2), study2, reg2) == KILL_EXIT, \
        "the gate-row child did not exit 77"
    r2 = run_unattended(study2, reg2)            # the reconcile repairs gate row then kill
    assert r2["status"] == "confirmed", r2
    for name in ("iterations.jsonl", "decisions.jsonl", "params/stable.json"):
        a = common.sha256_file(os.path.join(study2, name.replace("/", os.sep)))
        b = common.sha256_file(os.path.join(rstudy, name.replace("/", os.sep)))
        assert a == b, "the gate-row-killed study diverges on %s" % (name,)
    assert _tree(os.path.join(study2, "history")) == _tree(os.path.join(rstudy, "history")), \
        "history/ diverges"
    print("[ok] T4 kills before the replace: exit 77, cache validates, build- and gate-row resumes"
          " equal the reference")


def _t5() -> None:
    td = _mktemp_dir()
    study = _fx_study(td)
    reg = os.path.join(td, "studies.jsonl")
    for kill in (("after_replace", 2), ("after_iteration_row", 4), ("after_history", 2)):
        assert _child_kill(kill, study, reg) == KILL_EXIT, "the child for %r did not exit 77" % (kill,)
    rtd, rstudy, rreg = _reference()
    r = run_unattended(study, reg)
    assert r["status"] == "confirmed", r
    assert r["evaluator_calls"] == 1, r          # only the L2 confirmation is left after child 3
    for name in ("iterations.jsonl", "decisions.jsonl", "params/stable.json"):
        a = common.sha256_file(os.path.join(study, name.replace("/", os.sep)))
        b = common.sha256_file(os.path.join(rstudy, name.replace("/", os.sep)))
        assert a == b, "the thrice-killed study diverges on %s" % (name,)
    assert sorted(os.listdir(os.path.join(study, "cache"))) == sorted(os.listdir(os.path.join(rstudy, "cache")))
    for rel, sha in _tree(os.path.join(rstudy, "history")).items():
        assert common.sha256_file(os.path.join(study, "history", rel.replace("/", os.sep))) == sha, rel
    for name in KILL_POINTS:                     # a closed study's re-walk hits no kill point
        cd, cs, cr = _copy_ref("t5_" + name)
        assert _child_kill((name, 1), cs, cr) == 0, \
            "the closed re-walk child for %s did not exit 0" % (name,)
        for rel in (IT_JSON, DEC_JSON, "params/stable.json"):
            assert common.sha256_file(os.path.join(cs, rel.replace("/", os.sep))) == \
                common.sha256_file(os.path.join(rstudy, rel.replace("/", os.sep))), (name, rel)
    print("[ok] T5 three kills: each child exits 77, final resume equals the reference (1 call),"
          " closed re-walk exits 0 on every kill point")


def _t6() -> None:
    _td, study, reg = _copy_paused("t6")
    it = common.read_jsonl(os.path.join(study, "iterations.jsonl"))
    ek5 = it[6]["eval_key"]
    p5 = common.read_json(os.path.join(study, "cache", ek5, "params.json"))
    base = status(study, reg)["stable_eval_key"]
    edits = [("L_over_Di", p5["L_over_Di"]), ("Lx_over_De", p5["Lx_over_De"]),
             ("t_wall", p5["t_wall"])]
    ep = _edit_file(_td, "e_repeat.json", _edit_doc(edits, base, reason="repeat eval 5"))
    n_cache = len(os.listdir(os.path.join(study, "cache")))
    calls0 = EVALUATOR_CALLS[0]
    ri = intake(study, ep, reg)
    assert ri["evaluator_calls"] == 0, ri
    assert EVALUATOR_CALLS[0] == calls0, "the zero-work edit ran the evaluator"
    it = common.read_jsonl(os.path.join(study, "iterations.jsonl"))
    dec = common.read_jsonl(os.path.join(study, "decisions.jsonl"))
    new = it[-1]
    assert new["cache_hit"] is True and new["origin"] == "llm_edit" and new["eval_key"] == ek5
    assert dec[-1]["decision"] == "reject" and dec[-1]["rule_id"] == "GATE-TIE"
    assert len(os.listdir(os.path.join(study, "cache"))) == n_cache
    assert status(study, reg)["stable_eval_key"] == base, "the rejected edit moved the stable"
    print("[ok] T6 zero work: E_REPEAT replays eval 5 from cache, evaluator silent, GATE-TIE reject")


def _t7() -> None:
    base = status(_paused()[1], _paused()[2])["stable_eval_key"]
    it0 = len(common.read_jsonl(os.path.join(_paused()[1], "iterations.jsonl")))
    dec0 = len(common.read_jsonl(os.path.join(_paused()[1], "decisions.jsonl")))
    start_key = reqs.eval_key(eval_parts(None, dict(optimise_cad.STANDIN_START), "L1",
                                         reqs.load_template(reqs.NOZZLE_DIR)[1],
                                         reqs.load_template(reqs.NOZZLE_DIR)[2],
                                         _fx_doc()["lock_sha"], gate.load_gates()[1]))
    doc_locked = _edit_doc([("t_wall", 0.004)], base, reason="try the locked path")
    doc_locked["edits"] = [{"pointer": "/requirements/rows/2/value", "value": 0.07}]
    cases = (
        ("intent", _edit_doc([("D_i", 0.07)], base, reason="tighten the inlet"), "CAD-INTENT"),
        ("locked", doc_locked, "CAD-LOCKED"),
        ("stale", _edit_doc([("t_wall", 0.004)], start_key, reason="stale base"), "GATE-STALE"),
        ("range", _edit_doc([("L_over_Di", 2.0)], base, reason="out of range"), "CAD-RANGE"),
    )
    for tag, doc, rid in cases:
        td_i, study_i, reg_i = _copy_paused("t7_" + tag)
        ep = _edit_file(td_i, tag + ".json", doc)
        intake(study_i, ep, reg_i)
        dec = common.read_jsonl(os.path.join(study_i, "decisions.jsonl"))
        assert len(dec) == dec0 + 1 \
            and len(common.read_jsonl(os.path.join(study_i, "iterations.jsonl"))) == it0, tag
        assert dec[-1]["decision"] == "reject" and dec[-1]["rule_id"] == rid, (tag, dec[-1]["rule_id"])
    _td2, study2, reg2 = _copy_paused("t7_notjson")
    base2 = status(study2, reg2)["stable_eval_key"]
    p = os.path.join(_td2, "e_notjson.json")
    with open(p, "w", encoding="utf-8") as f:
        f.write("not json")
    intake(study2, p, reg2)
    dec = common.read_jsonl(os.path.join(study2, "decisions.jsonl"))
    assert dec[-1]["decision"] == "reject" and dec[-1]["rule_id"] == "CAD-EDIT"
    _td3, study3, reg3 = _copy_paused("t7_law")
    base3 = status(study3, reg3)["stable_eval_key"]
    ep = _edit_file(_td3, "e_law.json", _edit_doc([("law", "poly5")], base3, reason="try poly5"))
    intake(study3, ep, reg3)
    dec = common.read_jsonl(os.path.join(study3, "decisions.jsonl"))
    assert dec[-1]["decision"] == "reject" and dec[-1]["rule_id"] == "CAD-INTENT"
    _td4, study4, reg4 = _copy_paused("t7_lawcard")
    ep = _edit_file(_td4, "e_law_ok.json",
                    _edit_doc([("law", "poly5")], base3, reason="poly5, approved",
                              card={"approved_by": "fixture", "params": ["law"]}))
    intake(study4, ep, reg4)
    dec = common.read_jsonl(os.path.join(study4, "decisions.jsonl"))
    assert any(r["decision"] == "propose" and r["rule_id"] == "CAD-INTAKE" for r in dec), dec[-1]
    assert common.read_jsonl(os.path.join(study4, "iterations.jsonl"))[-1]["origin"] == "llm_edit"
    _td5, study5, reg5 = _copy_paused("t7_llmoff")
    base5 = status(study5, reg5)["stable_eval_key"]
    intake(study5, _edit_file(_td5, "r1.json", _edit_doc([("D_i", 0.07)], base5)), reg5)
    intake(study5, _edit_file(_td5, "r2.json", _edit_doc([("L_over_Di", 2.0)], base5)), reg5)
    n = len(common.read_jsonl(os.path.join(study5, "decisions.jsonl")))
    ep = _edit_file(_td5, "e_ok3.json",
                    _edit_doc([("L_over_Di", 0.8), ("Lx_over_De", 0.3), ("t_wall", 0.005)], base5))
    e = _refuse(intake, study5, ep, reg5)
    assert str(e).startswith("GATE-LLMOFF"), str(e)
    assert len(common.read_jsonl(os.path.join(study5, "decisions.jsonl"))) == n
    _td6, study6, reg6 = _copy_ref("t7_closed")
    e = _refuse(intake, study6, _edit_file(_td6, "x.json", _edit_doc([("t_wall", 0.004)], None)), reg6)
    assert str(e).startswith("LOOP-CLOSED"), str(e)
    _td7 = os.path.join(_mktemp_dir(), "t7_busy")
    shutil.copytree(_KILLED_TD, _td7)
    study7, reg7 = os.path.join(_td7, "study"), os.path.join(_td7, "studies.jsonl")
    e = _refuse(intake, study7, _edit_file(_td7, "x.json", _edit_doc([("t_wall", 0.004)], None)), reg7)
    assert str(e).startswith("LOOP-BUSY"), str(e)
    print("[ok] T7 intake refusals: CAD-INTENT/CAD-LOCKED/GATE-STALE/CAD-RANGE/CAD-EDIT,"
          " card accepts, GATE-LLMOFF, LOOP-CLOSED, LOOP-BUSY")


def _t8() -> None:
    _td, study, reg = _copy_paused("t8")
    base = status(study, reg)["stable_eval_key"]
    ep = _edit_file(_td, "e_ok.json",
                    _edit_doc([("L_over_Di", 0.8), ("Lx_over_De", 0.3), ("t_wall", 0.005)], base,
                              reason="settle the diffuser"))
    intake(study, ep, reg)
    it = common.read_jsonl(os.path.join(study, "iterations.jsonl"))
    new = it[-1]
    assert new["origin"] == "llm_edit" and new["design_verdict"] == "feasible"
    assert abs(new["objective"] - 0.054) < 1e-6, new["objective"]
    meas = common.read_json(os.path.join(study, "cache", new["eval_key"], "measurements.json"))
    assert abs(_mach(meas) - 0.27756908089749505) < 1e-9, _mach(meas)
    dec = common.read_jsonl(os.path.join(study, "decisions.jsonl"))
    assert dec[-1]["decision"] == "promote" and dec[-1]["rule_id"] == "GATE-OBJ"
    assert common.read_json(os.path.join(study, "params", "stable.json"))["eval_key"] == new["eval_key"]
    r = run_unattended(study, reg)
    rr = replay(study, reg)
    assert rr["ok"] is True, rr
    _td2, study2, reg2 = _copy_paused("t8b")
    ep2 = _edit_file(_td2, "e_ok.json",
                     _edit_doc([("L_over_Di", 0.8), ("Lx_over_De", 0.3), ("t_wall", 0.005)], base,
                               reason="settle the diffuser"))
    with open(ep, "rb") as a, open(ep2, "rb") as b:
        assert a.read() == b.read(), "the two edit files differ in bytes"
    intake(study2, ep2, reg2)
    run_unattended(study2, reg2)
    for name in ("iterations.jsonl", "decisions.jsonl"):
        assert common.sha256_file(os.path.join(study, name)) == \
            common.sha256_file(os.path.join(study2, name)), name
    st = status(study, reg)
    print("[ok] T8 accepted edit: %s after %d evals, replay ok, twin copies byte-identical"
          % (st["status"], st["n_evals"]))


def _t9() -> None:
    _td, study, reg = _copy_ref("t9_relock")
    doc = common.read_json(os.path.join(study, "requirements.json"))
    for row in doc["rows"]:
        if row["id"] == "REQ-003":
            row["value"] = 0.07
    doc["lock_sha"] = reqs.lock_sha_of(doc)
    common.write_json(os.path.join(study, "requirements.json"), doc)
    with open(os.path.join(study, "requirements.lock"), "w", encoding="utf-8", newline="\n") as f:
        f.write(doc["lock_sha"] + "\n")
    e = _refuse(run, study, reg)
    assert str(e).startswith("GATE-LOCK"), str(e)
    e = _refuse(replay, study, reg)
    assert str(e).startswith("GATE-LOCK"), str(e)
    _td2, study2, reg2 = _copy_ref("t9_checks")
    checks = common.read_json(os.path.join(study2, "checks.json"))
    checks["checks"][0]["tol"] = checks["checks"][0]["tol"] * 2.0
    common.write_json(os.path.join(study2, "checks.json"), checks)
    e = _refuse(run, study2, reg2)
    assert str(e).startswith("LOOP-CHECKS"), str(e)
    _td3, study3, reg3 = _copy_ref("t9_registry")
    p = os.path.join(_td3, "studies.jsonl")
    rows = common.read_jsonl(p)
    rows[0]["lock_sha"] = "0" * 64
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(common.canonical_json(r) + "\n" for r in rows))
    e = _refuse(run, study3, reg3)
    assert str(e).startswith("GATE-LOCK"), str(e)
    print("[ok] T9 locks: relock GATE-LOCK on run and replay, tampered checks LOOP-CHECKS,"
          " tampered registry GATE-LOCK")


def _t10() -> None:
    _td, study, reg = _copy_ref("t10_meas")
    it = common.read_jsonl(os.path.join(study, "iterations.jsonl"))
    ek1, ek2, ek3 = it[2]["eval_key"], it[3]["eval_key"], it[4]["eval_key"]
    p = os.path.join(study, "cache", ek1, "measurements.json")
    m = common.read_json(p)
    k0 = sorted(m)[0]
    m[k0]["value"] = m[k0]["value"] * (1.0 + 1e-9)
    common.write_json(p, m)
    rr = replay(study, reg)
    assert rr["ok"] is False and rr["first_mismatch"] == "cache:" + ek1, rr
    e = _refuse(run, study, reg)
    assert str(e).startswith("LOOP-CACHE") and ek1 in str(e), str(e)
    _td2, study2, reg2 = _copy_ref("t10_swap")
    c2 = os.path.join(study2, "cache")
    shutil.rmtree(os.path.join(c2, ek3))
    shutil.copytree(os.path.join(c2, ek2), os.path.join(c2, ek3))
    e = _refuse(run, study2, reg2)
    assert str(e).startswith("LOOP-CACHE") and ek3 in str(e), str(e)
    _td3, study3, reg3 = _copy_ref("t10_delete")
    os.remove(os.path.join(study3, "cache", ek1, "params.json"))
    e = _refuse(run, study3, reg3)
    assert str(e).startswith("LOOP-CACHE") and ek1 in str(e), str(e)
    _td4, study4, reg4 = _copy_ref("t10_untouched")
    d0, m0 = _tree(study4), _mtimes(study4)      # every file under the study directory
    n0 = (len(common.read_jsonl(os.path.join(study4, "decisions.jsonl"))),
          len(common.read_jsonl(os.path.join(study4, "iterations.jsonl"))))
    run(study4, reg4)
    assert _tree(study4) == d0, "the clean run changed bytes"
    assert _mtimes(study4) == m0, "the clean run touched a file"
    assert (len(common.read_jsonl(os.path.join(study4, "decisions.jsonl"))),
            len(common.read_jsonl(os.path.join(study4, "iterations.jsonl")))) == n0
    print("[ok] T10 cache: tampered/missing/neighbour entries refused, a clean run rewrites"
          " nothing (bytes and mtimes, whole study dir)")


def _t11() -> None:
    _td = os.path.join(_mktemp_dir(), "t11_torn")
    shutil.copytree(_KILLED_TD, _td)
    study, reg = os.path.join(_td, "study"), os.path.join(_td, "studies.jsonl")
    with open(os.path.join(study, "decisions.jsonl"), "ab") as f:
        f.write(b'{"schema": "cad-dec')
    rtd, rstudy, rreg = _reference()
    r = run_unattended(study, reg)
    assert r["status"] == "confirmed", r
    for name in ("iterations.jsonl", "decisions.jsonl", "params/stable.json"):
        a = common.sha256_file(os.path.join(study, name.replace("/", os.sep)))
        b = common.sha256_file(os.path.join(rstudy, name.replace("/", os.sep)))
        assert a == b, "the torn-tail resume diverges on %s" % (name,)
    doc_b = _fx_doc()
    doc_b["study_id"] = "v1_nominal_b"
    doc_b["supersedes_study"] = "v1_nominal"
    doc_b["supersedes_lock"] = _fx_doc()["lock_sha"]
    doc_b["change_kind"] = "target_change"
    doc_b["change_reason"] = "REQ-003 tightened to 0.07 m"
    for row in doc_b["rows"]:
        if row["id"] == "REQ-003":
            row["value"] = 0.07
    doc_b["lock_sha"] = reqs.lock_sha_of(doc_b)
    td2 = _mktemp_dir()
    reg2 = os.path.join(td2, "studies.jsonl")
    _fx_study(td2)                                   # the fresh v1 study, same registry
    reqs.write_locked(os.path.join(td2, "req_b"), doc_b)
    init(os.path.join(td2, "study_b"), os.path.join(td2, "req_b"), dict(START_DOC), reg2)
    rows = common.read_jsonl(reg2)
    assert len(rows) == 2 and rows[1]["study_id"] == "v1_nominal_b"
    assert rows[1]["supersedes_study"] == "v1_nominal" and rows[1]["supersedes_lock"] == _fx_doc()["lock_sha"]
    gates, gates_lock = gate.load_gates()
    gate.lock_anchor(os.path.join(td2, "study_b"), gates_lock, reg2)
    doc_c = dict(doc_b)
    doc_c["supersedes_lock"] = "0" * 64
    doc_c["lock_sha"] = reqs.lock_sha_of(doc_c)
    td3 = _mktemp_dir()
    reg3 = os.path.join(td3, "studies.jsonl")
    _fx_study(td3)
    reqs.write_locked(os.path.join(td3, "req_c"), doc_c)
    e = _refuse(init, os.path.join(td3, "study_c"), os.path.join(td3, "req_c"), dict(START_DOC), reg3)
    assert str(e).startswith("LOOP-LINEAGE"), str(e)
    doc_d = dict(doc_b)
    doc_d["supersedes_study"] = "no_such_study"
    doc_d["lock_sha"] = reqs.lock_sha_of(doc_d)
    td4 = _mktemp_dir()
    reg4 = os.path.join(td4, "studies.jsonl")
    _fx_study(td4)
    reqs.write_locked(os.path.join(td4, "req_d"), doc_d)
    e = _refuse(init, os.path.join(td4, "study_d"), os.path.join(td4, "req_d"), dict(START_DOC), reg4)
    assert str(e).startswith("LOOP-LINEAGE"), str(e)
    print("[ok] T11 torn tail and lineage: half-line resumes clean, superseding study anchors,"
          " two LOOP-LINEAGE refusals")


def _t12() -> None:
    td = _mktemp_dir()
    reqs.write_locked(os.path.join(td, "req"), _fx_doc())
    sp = os.path.join(td, "start.json")
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(START_DOC, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    reg = os.path.join(td, "studies.jsonl")
    study = os.path.join(td, "study")
    lp = os.path.join(HERE, "loop.py")

    def cli(*args):
        return subprocess.run([sys.executable, lp] + list(args), capture_output=True, text=True,
                              env=dict(os.environ, PYTHONIOENCODING="utf-8"))

    p = cli("init", study, os.path.join(td, "req"), sp, "--registry", reg)
    assert p.returncode == 0, (p.returncode, p.stderr[-300:])
    p = cli("run", study, "--registry", reg, "--unattended")
    assert p.returncode == 0, (p.returncode, p.stderr[-300:])
    p = cli("status", study, "--registry", reg)
    assert p.returncode == 0, (p.returncode, p.stderr[-300:])
    st = json.loads(p.stdout.strip().splitlines()[-1])
    assert st["status"] == "confirmed" and st["n_evals"] == 8, st
    p = cli("replay", study, "--registry", reg)
    assert p.returncode == 0, (p.returncode, p.stderr[-300:])
    assert json.loads(p.stdout.strip().splitlines()[-1])["ok"] is True
    _td2, study2, reg2 = _copy_ref("t12_checks")
    checks = common.read_json(os.path.join(study2, "checks.json"))
    checks["checks"][0]["tol"] = checks["checks"][0]["tol"] * 2.0
    common.write_json(os.path.join(study2, "checks.json"), checks)
    p = cli("run", study2, "--registry", reg2)
    assert p.returncode == 1 and p.stderr.startswith("LOOP-CHECKS:"), (p.returncode, p.stderr[:120])
    av = console_argv(study, reg)
    assert av[:2] == [sys.executable, os.path.abspath(lp)]
    assert av[2:] == ["run", os.path.abspath(study), "--registry", os.path.abspath(reg)]
    assert cli("--help").returncode == 0
    assert not os.path.exists(os.path.join(HERE, "studies.jsonl")), "tools/cad/studies.jsonl exists"
    print("[ok] T12 CLI: init/run/status/replay exit 0, confirmed 8 evals, LOOP-CHECKS exit 1,"
          " --help, no studies.jsonl")


def selftest() -> None:
    """The CAD-18 gate: T1-T12, one [ok] line each, SELFTEST PASS at the end."""
    try:
        _t1()
        _t2()
        _t3()
        _t4()
        _t5()
        _t6()
        _t7()
        _t8()
        _t9()
        _t10()
        _t11()
        _t12()
    finally:
        _cleanup_temps()
    print("SELFTEST PASS")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
