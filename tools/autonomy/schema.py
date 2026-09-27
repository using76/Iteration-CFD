#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""
schema.py - the autonomy package's validator: a stdlib JSON-Schema draft
2020-12 subset written from the keyword list in docs/15 (no third-party
import on the validation path), the canonical-sha256 gate lock (docs/15 §F,
§I-3), the knob whitelist and legal-range table (docs/15 §C L0 (d), §I-5),
the seventeen semantic checks on an attempt row (docs/15 §D), and the a priori
y+ helpers (docs/15 §D.3, §I-4).

    python tools/autonomy/schema.py --selftest              # the package gate
    python tools/autonomy/schema.py --print-lock            # lock lines, no comments
    python tools/autonomy/schema.py --write-lock            # once, before tuning
    python tools/autonomy/schema.py --validate KIND FILE    # valid / errors
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SCHEMA_DIR = os.path.join(HERE, 'schema')
GATES_PATH = os.path.join(HERE, 'gates.json')
LOCK_PATH = os.path.join(HERE, 'gates.lock')
KNOBS_PATH = os.path.join(HERE, 'schema', 'knobs.json')
FIXTURES_PATH = os.path.join(HERE, 'fixtures', 'good.json')

SCHEMAS = {"FlowSpec": "flow_spec.schema.json",
           "ManifestRow": "manifest_row.schema.json",
           "Fingerprint": "fingerprint.schema.json",
           "DecisionRecord": "decision_record.schema.json",
           "AttemptRow": "attempt_row.schema.json",
           "GateConstants": "gate_constants.schema.json",
           "Knobs": "knobs.schema.json"}
FIXTURE_KINDS = {"flow": "FlowSpec", "manifest": "ManifestRow",
                 "fingerprint": "Fingerprint", "decision": "DecisionRecord",
                 "attempt": "AttemptRow"}
YPLUS_NAME = re.compile(r"(?i)(yplus|y_plus|blc|u_tau|(^|_)cf(_|$))")
F1_CLASSES = ("config", "surface_closed", "layer_t1_G5", "timeout", "crash", "io")
FLAT_PLATE_CITE = ("SPEC-LIT §32.5.6 (solver tree, feat/core-2 d272391, rust/SPEC-LIT.md:3898-3899); "
                   "Schlichting & Gersten, Boundary-Layer Theory, 8th ed., Springer (2000)")


class SchemaError(ValueError):
    """A schema file itself is malformed: an unknown keyword or a broken $ref."""


class LockError(ValueError):
    """gates.json / schema/knobs.json disagree with the locked hashes."""


# --- schema loading: the keyword subset of docs/15, no others --------------

ALLOWED_KEYWORDS = frozenset([
    "$schema", "$id", "$comment", "$defs", "$ref", "title", "description",
    "type", "properties", "required", "additionalProperties", "enum", "const",
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "minLength",
    "maxLength", "pattern", "items", "minItems", "maxItems"])

_DOCS = {}


def _check_keywords(node, fname, loc):
    """Raise SchemaError when a schema NODE carries a keyword outside the subset."""
    if not isinstance(node, dict):
        return
    for k in node:
        if k not in ALLOWED_KEYWORDS:
            raise SchemaError("%s: %s: unknown schema keyword %r" % (fname, loc or "$", k))
    for k, v in node.items():
        if k == "properties":
            for name, psch in v.items():
                _check_keywords(psch, fname, loc + "/properties/" + name)
        elif k == "$defs":
            for name, dsch in v.items():
                _check_keywords(dsch, fname, loc + "/" + name)
        elif k == "items":
            _check_keywords(v, fname, loc + "/items")


def _load_doc(path):
    """Read one schema file (cached), keyword-checking it exactly once."""
    path = os.path.normpath(path)
    if path in _DOCS:
        return _DOCS[path]
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    _check_keywords(doc, os.path.basename(path), "")
    _DOCS[path] = doc
    return doc


def load_schema(kind: str) -> dict:
    """The parsed schema for a kind in SCHEMAS, keyword-checked; SchemaError if bad."""
    if kind not in SCHEMAS:
        raise SchemaError("unknown schema kind %r (have %s)" % (kind, ", ".join(sorted(SCHEMAS))))
    return _load_doc(os.path.join(SCHEMA_DIR, SCHEMAS[kind]))


def _resolve(sch, root):
    """Follow a $ref; returns (resolved node, the document that now owns it)."""
    if not (isinstance(sch, dict) and "$ref" in sch):
        return sch, root
    ref = sch["$ref"]
    if ref.startswith("#"):
        cur = root
        for part in ref[1:].split("/")[1:]:
            cur = cur[part.replace("~1", "/").replace("~0", "~")]
        return cur, root
    fname, _, frag = ref.partition("#")
    doc = _load_doc(os.path.join(SCHEMA_DIR, fname))
    if not frag:
        return doc, doc
    cur = doc
    for part in frag.split("/")[1:]:
        cur = cur[part.replace("~1", "/").replace("~0", "~")]
    return cur, doc


def _json_type(v):
    """The JSON-Schema type name of a Python value: bool is boolean, never a number."""
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int):
        return "integer"
    if isinstance(v, float):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "array"
    if isinstance(v, dict):
        return "object"
    if v is None:
        return "null"
    return type(v).__name__


def _type_ok(v, types):
    jt = _json_type(v)
    if jt == "boolean":
        return "boolean" in types
    if jt == "integer":
        return "integer" in types or "number" in types
    if jt == "number" and "integer" in types and math.isfinite(v) and v.is_integer():
        return True  # draft 2020-12: a number with a zero fractional part is an integer
    return jt in types


def _same(a, b):
    """enum/const equality: 1 == 1.0 passes, True != 1."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    num = isinstance(a, (int, float)) and isinstance(b, (int, float))
    return (a == b) if num else (type(a) is type(b) and a == b)


def _child(path, key):
    """The path string of a child field: the root '$' itself is not shown."""
    return key if path == "$" else path + "." + key


def _item(path, i):
    return path + "[%d]" % i if path != "$" else "$[%d]" % i


def _validate(v, sch, root, path, kind, errs):
    """Collect every violation of `sch` by `v` into errs as "<kind>: <path>: <reason>"."""
    sch, root = _resolve(sch, root)
    types = sch.get("type")
    if types is not None:
        tl = [types] if isinstance(types, str) else list(types)
        if not _type_ok(v, tl):
            errs.append("%s: %s: expected %s, got %s"
                        % (kind, path, ", ".join(tl), _json_type(v)))
            return
    if "enum" in sch and not any(_same(v, e) for e in sch["enum"]):
        errs.append("%s: %s: not one of %s" % (kind, path, json.dumps(sch["enum"])))
    if "const" in sch and not _same(v, sch["const"]):
        errs.append("%s: %s: must equal %s" % (kind, path, json.dumps(sch["const"])))
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if "minimum" in sch and v < sch["minimum"]:
            errs.append("%s: %s: below minimum %s" % (kind, path, json.dumps(sch["minimum"])))
        if "exclusiveMinimum" in sch and v <= sch["exclusiveMinimum"]:
            errs.append("%s: %s: at or below exclusive minimum %s"
                        % (kind, path, json.dumps(sch["exclusiveMinimum"])))
        if "maximum" in sch and v > sch["maximum"]:
            errs.append("%s: %s: above maximum %s" % (kind, path, json.dumps(sch["maximum"])))
        if "exclusiveMaximum" in sch and v >= sch["exclusiveMaximum"]:
            errs.append("%s: %s: at or above exclusive maximum %s"
                        % (kind, path, json.dumps(sch["exclusiveMaximum"])))
    if isinstance(v, str):
        if "minLength" in sch and len(v) < sch["minLength"]:
            errs.append("%s: %s: shorter than %d" % (kind, path, sch["minLength"]))
        if "maxLength" in sch and len(v) > sch["maxLength"]:
            errs.append("%s: %s: longer than %d" % (kind, path, sch["maxLength"]))
        if "pattern" in sch and not re.search(sch["pattern"], v):
            errs.append("%s: %s: does not match %s" % (kind, path, sch["pattern"]))
    if isinstance(v, list):
        if "minItems" in sch and len(v) < sch["minItems"]:
            errs.append("%s: %s: fewer than %d items" % (kind, path, sch["minItems"]))
        if "maxItems" in sch and len(v) > sch["maxItems"]:
            errs.append("%s: %s: more than %d items" % (kind, path, sch["maxItems"]))
        if "items" in sch:
            isch, iroot = _resolve(sch["items"], root)
            for i, el in enumerate(v):
                _validate(el, isch, iroot, _item(path, i), kind, errs)
    if isinstance(v, dict):
        props = sch.get("properties", {})
        for name, psch in props.items():
            if name in v:
                _validate(v[name], psch, root, _child(path, name), kind, errs)
        if sch.get("additionalProperties", None) is False:
            for k in v:
                if k not in props:
                    errs.append("%s: %s: unknown field" % (kind, _child(path, k)))
        for name in sch.get("required", []):
            if name not in v:
                errs.append("%s: %s: required field missing" % (kind, _child(path, name)))


def errors(instance, kind: str) -> list[str]:
    """Every violation of schema `kind` by `instance`, as "<kind>: <path>: <reason>"."""
    root = load_schema(kind)
    errs: list[str] = []
    _validate(instance, root, root, "$", kind, errs)
    return errs


def validate(instance, kind: str) -> None:
    """Raise SchemaError(errors[0]) when `instance` violates schema `kind`."""
    errs = errors(instance, kind)
    if errs:
        raise SchemaError(errs[0])


# --- the gate lock (docs/15 §F, §I-3) --------------------------------------

def canonical_sha256(obj) -> str:
    """sha256 hex over canonical JSON: sort_keys, tight separators, ascii, no NaN."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(blob).hexdigest()


def lock_lines(gates: dict, knobs: dict) -> list[str]:
    """The whole-gates hash, one hash per gates field, and the knobs hash."""
    lines = ["%s  gates.json" % canonical_sha256(gates)]
    for k in sorted(gates):
        lines.append("%s  gates.json#%s" % (canonical_sha256(gates[k]), k))
    lines.append("%s  schema/knobs.json" % canonical_sha256(knobs))
    return lines


def check_gates_against_lock(gates: dict, knobs: dict, lock_text: str) -> list[str]:
    """One refusal per mismatch between the files as parsed and gates.lock."""
    locked = {}
    for line in lock_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        h, _, name = line.partition("  ")
        locked[name.strip()] = h.strip()
    out = []
    for k in sorted(gates):
        name = "gates.json#" + k
        if name not in locked:
            out.append("GATES-LOCK: %s: not in the lock" % k)
        elif canonical_sha256(gates[k]) != locked[name]:
            out.append("GATES-LOCK: %s: %r does not match the lock" % (k, gates[k]))
    for name in locked:
        if name.startswith("gates.json#") and name[len("gates.json#"):] not in gates:
            out.append("GATES-LOCK: %s: missing from gates.json" % name[len("gates.json#"):])
    if "gates.json" in locked and canonical_sha256(gates) != locked["gates.json"]:
        out.append("GATES-LOCK: gates.json: whole-file hash does not match the lock")
    if locked.get("schema/knobs.json") != canonical_sha256(knobs):
        out.append("KNOBS-LOCK: schema/knobs.json does not match the lock")
    return out


def _read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_gates(path: str = GATES_PATH, lock: str = LOCK_PATH) -> dict:
    """Parse, schema-check and lock-check gates.json; LockError on the first refusal."""
    gates = _read_json(path)
    validate(gates, "GateConstants")
    knobs = _read_json(KNOBS_PATH)
    validate(knobs, "Knobs")
    refusals = check_gates_against_lock(gates, knobs, open(lock, encoding="utf-8").read())
    if refusals:
        raise LockError(refusals[0])
    return gates


def load_knobs(path: str = KNOBS_PATH, lock: str = LOCK_PATH) -> dict:
    """Parse, schema-check and lock-check schema/knobs.json; LockError on refusal."""
    knobs = _read_json(path)
    validate(knobs, "Knobs")
    gates = _read_json(GATES_PATH)
    validate(gates, "GateConstants")
    refusals = check_gates_against_lock(gates, knobs, open(lock, encoding="utf-8").read())
    if refusals:
        raise LockError(refusals[0])
    return knobs


# --- the knob whitelist (docs/15 §C L0 (d), §I-5) ---------------------------

_POINTER_RE = re.compile(r"^(/[^/]+)+$")


def _now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _refusal(rule_id, cite, message, trigger, edits=None):
    """A DecisionRecord-shaped refusal (valid under the DecisionRecord schema).

    `edits` is omitted when the pointer itself is malformed (WL-POINTER) or is
    a command-line flag (WL-FLAG), because an Edit's `pointer` must be a valid
    RFC 6901 pointer for the refusal to validate under the DecisionRecord schema.
    """
    return {"schema": "autonomy-decision/1", "layer": "preflight", "rule_id": rule_id,
            "verdict": "refuse", "trigger": trigger, "inputs": [], "formula": "",
            "edits": edits or [], "cite": cite, "message": message, "uncertainty": 0,
            "t": _now_iso()}


def _knob_row(pointer, knobs):
    """The single whitelist row matching `pointer` (a `*` matches one all-digits segment)."""
    segs = pointer.split("/")[1:]
    for row in knobs.get("whitelist", []):
        rsegs = row["pointer"].split("/")[1:]
        if len(segs) != len(rsegs):
            continue
        if all(rs == "*" and s.isdigit() or s == rs for s, rs in zip(segs, rsegs)):
            return row
    return None


def _type_bad(value, typ):
    """True when `value` fails the row's declared type."""
    if typ == "int":
        return not (isinstance(value, int) and not isinstance(value, bool))
    if typ == "float":
        return not (isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value))
    if typ == "str":
        return not (isinstance(value, str) and value != "")
    if typ == "str_list":
        return not (isinstance(value, list) and len(value) > 0
                    and all(isinstance(x, str) and x != "" for x in value))
    if typ == "extent":
        return not (isinstance(value, list) and len(value) == 6
                    and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                            and math.isfinite(x) for x in value))
    return True


def check_edit(pointer, value, knobs: dict) -> dict | None:
    """The first (WL-*) refusal of one knob edit against the whitelist, or None."""
    ok_pointer = isinstance(pointer, str) and _POINTER_RE.match(pointer)
    trigger = lambda th, op: {"observable": pointer if isinstance(pointer, str) else repr(pointer),
                              "value": value, "threshold": th, "op": op,
                              "source": "tools/autonomy/schema/knobs.json"}
    if not ok_pointer:
        return _refusal("WL-POINTER", "RFC 6901 JSON Pointer",
                        "WL-POINTER: %r is not a JSON Pointer" % (pointer,),
                        trigger(pointer if isinstance(pointer, str) else None, "=="))
    row = _knob_row(pointer, knobs)
    for f in knobs.get("forbidden", []):
        if pointer == f["pointer"] or \
           (f["match"] == "prefix" and pointer.startswith(f["pointer"] + "/")):
            return _refusal("WL-FORBIDDEN", f["cite"],
                            "WL-FORBIDDEN: %s is out of the action space (%s)"
                            % (pointer, f["cite"]),
                            trigger(f["pointer"], "=="),
                            edits=[{"pointer": pointer, "from": None, "to": value}])
    if row is None:
        return _refusal("WL-UNLISTED", "docs/15 §C L0 (d)",
                        "WL-UNLISTED: %s is not a whitelisted knob" % pointer,
                        trigger(pointer, "not_in"),
                        edits=[{"pointer": pointer, "from": None, "to": value}])
    if _type_bad(value, row["type"]):
        return _refusal("WL-TYPE", row["field"],
                        "WL-TYPE: %s needs a %s, got %r" % (pointer, row["type"], value),
                        trigger(row["type"], "=="),
                        edits=[{"pointer": pointer, "from": None, "to": value}])
    return _finish_check_edit(pointer, value, row)


def check_flags(argv: list[str], knobs: dict) -> list[dict]:
    """One WL-FLAG refusal per argv item equal to a forbidden flag."""
    out = []
    for arg in argv:
        for f in knobs.get("forbidden_flags", []):
            if arg == f["flag"]:
                out.append(_refusal("WL-FLAG", f["cite"],
                                    "WL-FLAG: %s is forbidden (%s)" % (arg, f["cite"]),
                                    {"observable": arg, "value": arg, "threshold": arg,
                                     "op": "==", "source": "tools/autonomy/schema/knobs.json"}))
    return out


def _finish_check_edit(pointer, value, row):
    """WL-RANGE: the numeric bounds, or the extent's lo >= hi per axis."""
    lo, hi, excl = row["min"], row["max"], row["min_excl"]
    cite = row["validator"] or row["declared"]
    trig = {"observable": pointer, "value": value, "threshold": [lo, hi], "op": "<=",
            "source": "tools/autonomy/schema/knobs.json"}
    edit = [{"pointer": pointer, "from": None, "to": value}]
    if row["type"] in ("int", "float") and isinstance(value, (int, float)) \
            and not isinstance(value, bool):
        if lo is not None and (value < lo or (excl and value == lo)):
            return _refusal("WL-RANGE", cite,
                            "WL-RANGE: %s = %r is below its %s bound %r (cite %s)"
                            % (pointer, value, "exclusive" if excl else "minimum", lo, cite),
                            trig, edits=edit)
        if hi is not None and value > hi:
            trig = dict(trig, op=">=")
            return _refusal("WL-RANGE", cite,
                            "WL-RANGE: %s = %r is above its maximum %r (cite %s)"
                            % (pointer, value, hi, cite), trig, edits=edit)
    if row["type"] == "extent":
        for j, axis in enumerate("xyz"):
            if value[2 * j] >= value[2 * j + 1]:
                return _refusal("WL-RANGE", cite,
                                "WL-RANGE: %s: the %s axis is empty or reversed (%r >= %r)"
                                % (pointer, axis, value[2 * j], value[2 * j + 1]),
                                trig, edits=edit)
    return None


# --- the seventeen semantic checks on an attempt row (docs/15 §D) -----------

def _parse_iso(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def check_attempt(row: dict, gates: dict, knobs: dict) -> list[str]:
    """S1..S12 (docs/15 §D), run AFTER schema validation; empty list = consistent."""
    out = []
    oc = row["outcome"]
    fl = oc["flags"]
    if row["attempt"] > gates["attempts_k"]:
        out.append("attempt: attempt: attempt %d exceeds the locked K = %d (docs/15 §I-3)"
                   % (row["attempt"], gates["attempts_k"]))
    for i, e in enumerate(row["config_delta"]):
        ref = check_edit(e["pointer"], e["to"], knobs)
        if ref is not None:
            out.append("attempt: config_delta[%d].pointer: %s" % (i, ref["message"]))
    if row["decided_by"] == "default" and row["rule_id"] is not None:
        out.append("attempt: rule_id: a default decision carries no rule id")
    if row["decided_by"] in ("rule", "remedy") and row["rule_id"] is None:
        out.append("attempt: rule_id: a rule or remedy decision must carry its rule id")
    if row["decided_by"] == "optimiser" and row["prediction"] is None:
        out.append("attempt: prediction: an optimiser decision must carry its prediction")
    if row["prediction"] is not None:
        if _parse_iso(row["prediction"]["t_predicted"]) >= _parse_iso(row["t_start"]):
            out.append("attempt: prediction.t_predicted: the prediction is not from "
                       "before the attempt started")
    if _parse_iso(row["t_end"]) < _parse_iso(row["t_start"]):
        out.append("attempt: t_end: the attempt ended before it started")
    if bool(oc["failure"]) != any(f is True for f in fl.values()):
        out.append("attempt: outcome.failure: failure is true iff some F flag of "
                   "docs/15 §D.1 is true")
    dropped_req = any(p["requested"] and p["dropped"] is not None for p in oc["patches"])
    if bool(oc["strict_failure"]) != (bool(oc["failure"]) or dropped_req):
        out.append("attempt: outcome.strict_failure: strict failure is failure OR any "
                   "requested layer patch dropped (docs/15 §D.1)")
    want_verdict = "fail" if oc["failure"] else "pass"
    if oc["verdict"] != want_verdict:
        out.append("attempt: outcome.verdict: verdict must be %r when failure is %r"
                   % (want_verdict, oc["failure"]))
    for i, p in enumerate(oc["patches"]):
        want = p["dropped"] is None and p["n_layers"] >= gates["delivered_min_layers"]
        if bool(p["delivered"]) != want:
            out.append("attempt: outcome.patches[%d].delivered: delivered is (dropped is "
                       "null and n_layers >= %d) (docs/15 §D.2)" % (i, gates["delivered_min_layers"]))
    if row["fingerprint"]["geometry_id"] != row["geometry_id"]:
        out.append("attempt: fingerprint.geometry_id: the embedded fingerprint names a "
                   "different geometry")
    if fl["F3a"] != (oc["pinned_frac"] is not None
                     and oc["pinned_frac"] > gates["pinned_frac_max"]):
        out.append("attempt: outcome.flags.F3a: F3a is pinned_frac > pinned_frac_max "
                   "(docs/15 §D.1)")
    if fl["F3b"] != (oc["p99_over_hf"] is not None
                     and oc["p99_over_hf"] > gates["p99_residual_over_hf_max"]):
        out.append("attempt: outcome.flags.F3b: F3b is p99_over_hf > "
                   "p99_residual_over_hf_max (docs/15 §D.1)")
    if fl["F3c"] != (oc["max_over_hf"] is not None
                     and oc["max_over_hf"] > gates["max_residual_over_hf_max"]):
        out.append("attempt: outcome.flags.F3c: F3c is max_over_hf > "
                   "max_residual_over_hf_max (docs/15 §D.1)")
    if ("F3e" in fl) != ("feature_capture" in oc):
        out.append("attempt: outcome.feature_capture: F3e and feature_capture are written "
                   "together (the user's decision of 2026-09-26)")
    elif "F3e" in fl:
        fc = oc["feature_capture"]
        if (fl["F3e"] is True) != (fc is not None and fc == 0):
            out.append("attempt: outcome.flags.F3e: F3e is feature_capture == 0 on a body "
                       "with sharp edges (the user's decision of 2026-09-26)")
        elif fc is not None and fl["F3e"] is None:
            out.append("attempt: outcome.flags.F3e: a measured feature_capture decides F3e")
    if fl["F5"] != (oc["n_cells"] is not None and oc["n_cells"] > gates["cell_budget"]):
        out.append("attempt: outcome.flags.F5: F5 is n_cells > cell_budget (docs/15 §D.1)")
    if fl["F1"] != (oc["exit_code"] != 0 or oc["failure_class"] in F1_CLASSES
                    or (oc["failure_class"] or "").startswith("gate_")):
        out.append("attempt: outcome.flags.F1: F1 is exit != 0, timeout, crash, io or a "
                   "refusal class (docs/15 §D.1)")
    return out


# --- the a priori y+ helpers (docs/15 §D.3, §I-4) ---------------------------

def _walk_ap(sch, root, where, out, seen):
    """Collect y+-bearing property names missing _a_priori, across $defs, items, nesting."""
    sch, root = _resolve(sch, root)
    if id(sch) in seen:
        return
    seen.add(id(sch))
    for name, psch in sch.get("properties", {}).items():
        if YPLUS_NAME.search(name) and "_a_priori" not in name:
            out.append("%s.%s: y+-bearing field must carry _a_priori (docs/15 §I-4)"
                       % (where, name))
        _walk_ap(psch, root, where + "." + name, out, seen)
    for name, dsch in sch.get("$defs", {}).items():
        _walk_ap(dsch, root, where + "." + name, out, seen)
    if "items" in sch and isinstance(sch["items"], dict):
        _walk_ap(sch["items"], root, where + "[]", out, seen)


def a_priori_name_violations(schema_obj: dict, where: str = "$") -> list[str]:
    """Every y+-bearing property name that does not carry _a_priori (docs/15 §I-4)."""
    out: list[str] = []
    _walk_ap(schema_obj, schema_obj, where, out, set())
    return out


def _count_ap(sch, root, seen):
    """How many real property names match YPLUS_NAME (they all carry _a_priori)."""
    sch, root = _resolve(sch, root)
    if id(sch) in seen:
        return 0
    seen.add(id(sch))
    n = 0
    for name, psch in sch.get("properties", {}).items():
        if YPLUS_NAME.search(name):
            n += 1
        n += _count_ap(psch, root, seen)
    for dsch in sch.get("$defs", {}).values():
        n += _count_ap(dsch, root, seen)
    if "items" in sch and isinstance(sch["items"], dict):
        n += _count_ap(sch["items"], root, seen)
    return n


def count_yplus_names(schema_obj: dict) -> int:
    """The YPLUS_NAME match count over a whole schema's properties, each node once."""
    return _count_ap(schema_obj, schema_obj, set())


def flat_plate_cf(re_l: float) -> float | None:
    """The flat-plate skin-friction correlation; None when Re_L <= 0 or not finite.

    Cite: SPEC-LIT §32.5.6 (solver tree, feat/core-2 d272391, rust/SPEC-LIT.md:3898-3899);
    Schlichting & Gersten, Boundary-Layer Theory, 8th ed., Springer (2000) - FLAT_PLATE_CITE.
    """
    if not isinstance(re_l, (int, float)) or isinstance(re_l, bool):
        return None
    if not math.isfinite(re_l) or re_l <= 0:
        return None
    if re_l < 5e5:
        return 1.328 / math.sqrt(re_l)
    return 0.455 / math.log10(re_l) ** 2.58


def a_priori_wall(flow: dict, yplus: float = 1.0) -> dict:
    """The a priori wall quantities (docs/15 §D.3): Re_L, C_F, u_tau, t1 - each named _a_priori."""
    u, l, nu = flow["u_ref_m_s"], flow["l_ref_m"], flow["nu_m2_s"]
    re_l = u * l / nu
    cf = flat_plate_cf(re_l)
    if cf is None:
        return {"re_l": re_l, "cf_a_priori": None, "u_tau_a_priori_m_s": None,
                "t1_a_priori_m": None, "yplus_target_a_priori": yplus,
                "cite": FLAT_PLATE_CITE}
    u_tau = u * math.sqrt(cf / 2)
    return {"re_l": re_l, "cf_a_priori": cf, "u_tau_a_priori_m_s": u_tau,
            "t1_a_priori_m": yplus * nu / u_tau, "yplus_target_a_priori": yplus,
            "cite": FLAT_PLATE_CITE}


def yplus_a_priori(t1_m: float, flow: dict) -> float:
    """y+ = t1 * u_tau / nu (docs/15 §D.2's y+_p), with u_tau from a_priori_wall."""
    wall = a_priori_wall(flow)
    return t1_m * wall["u_tau_a_priori_m_s"] / flow["nu_m2_s"]


# --- the package selftest (schema.py --selftest) ----------------------------

def _disp(segs):
    """The error-string path of a segment list: the root '$' itself is not shown."""
    out = None
    for kind_, k in segs[1:]:
        if kind_ == "key":
            out = k if out is None else out + "." + k
        else:
            out = ("$[%d]" % k) if out is None else out + ("[%d]" % k)
    return out if out is not None else "$"


def _node_at(inst, segs):
    cur = inst
    for _, k in segs[1:]:
        cur = cur[k]
    return cur


def _set_at(inst, segs, val):
    cur = inst
    for _, k in segs[1:-1]:
        cur = cur[k]
    if isinstance(cur, (str, int, float, bool)) or cur is None:
        raise AssertionError("walker descended into a scalar: segs=%r" % (segs,))
    old = cur[segs[-1][1]]
    cur[segs[-1][1]] = val
    return old


def _del_at(inst, segs, key):
    cur = _node_at(inst, segs)
    old = cur[key]
    del cur[key]
    return old


def _first_wrong(types):
    """The first candidate whose JSON type is outside `types` (docs/15 AM-1 walker)."""
    for cand in ("__bad__", 12345, 1.5, True, [], {}):
        if not _type_ok(cand, types):
            return cand
    return None


def _expect_refused(inst, kind, path, needle, counter, counters, oracle, why):
    errs = errors(inst, kind)
    hit = any(e.startswith("%s: %s: " % (kind, path)) and needle in e for e in errs)
    assert hit, "the %s mutation at %s was not refused naming it (got %s)" % (why, path, errs)
    if oracle is not None:
        assert oracle(inst, kind), "the jsonschema oracle did not refuse the %s at %s" % (why, path)
    counters[counter] += 1


def _walk_bad(inst, sch, root, segs, kind, counters, oracle):
    """Mutate every field of one good fixture: wrong type, deleted, unknown key."""
    sch, root = _resolve(sch, root)
    types = sch.get("type")
    closed = "enum" in sch or "const" in sch
    if types is None and not closed:
        counters["any"] += 1
        return
    disp = _disp(segs)
    is_root = segs == [("$", "")]
    if types is not None:
        tl = [types] if isinstance(types, str) else list(types)
        bad = None if is_root else _first_wrong(tl)
        if bad is not None:
            old = _set_at(inst, segs, bad)
            _expect_refused(inst, kind, disp, "expected ", "type", counters, oracle,
                            "wrong-type %r" % (bad,))
            _set_at(inst, segs, old)
    if closed and not is_root and (types is None or _type_ok("__bad__", tl)):
        old = _set_at(inst, segs, "__bad__")
        _expect_refused(inst, kind, disp, "not one of" if "enum" in sch else "must equal",
                        "value", counters, oracle, "wrong-value")
        _set_at(inst, segs, old)
    node = _node_at(inst, segs)
    if isinstance(node, dict):
        props = sch.get("properties", {})
        if sch.get("additionalProperties", None) is False:
            base = "" if disp == "$" else disp + "."
            node["__unknown__"] = 1
            _expect_refused(inst, kind, base + "__unknown__", "unknown field",
                            "unknown", counters, oracle, "unknown-key")
            del node["__unknown__"]
        for k in list(sch.get("required", [])):
            if k in node:
                base = "" if disp == "$" else disp + "."
                old = _del_at(inst, segs, k)
                _expect_refused(inst, kind, base + k, "required field missing", "missing",
                                counters, oracle, "missing-field %s" % k)
                node[k] = old
        for k in list(node):
            if k in props:
                _walk_bad(inst, props[k], root, segs + [("key", k)], kind, counters, oracle)
    if isinstance(node, list) and "items" in sch and node:
        isch, iroot = _resolve(sch["items"], root)
        _walk_bad(inst, isch, iroot, segs + [("idx", 0)], kind, counters, oracle)


EDIT_REFUSALS = [
    (("/layers/cell_frac", 0.3), "WL-FORBIDDEN"),
    (("/layers/medial_frac", 0.4), "WL-FORBIDDEN"),
    (("/quality/max_non_orth_deg", 80.0), "WL-FORBIDDEN"),
    (("/quality", {}), "WL-FORBIDDEN"),
    (("/layers/min_thickness", 0.01), "WL-UNLISTED"),
    (("/snap/max_area_ratio", 8.0), "WL-UNLISTED"),
    (("/refinement/levels/x/feature_level", 1), "WL-UNLISTED"),
    (("layers/n", 8), "WL-POINTER"),
    (("/layers/n", 8.5), "WL-TYPE"),
    (("/layers/n", True), "WL-TYPE"),
    (("/snap/iterations", 1000), "WL-RANGE"),
    (("/layers/growth", 0.5), "WL-RANGE"),
    (("/domain/base_size", 0.0), "WL-RANGE"),
    (("/domain/extent", [0, 1, 0, 1, 1, 0]), "WL-RANGE"),
    (("/refinement/levels/0/bands/1/level", 7), "WL-RANGE"),
    (("/layers/patches", []), "WL-TYPE"),
]
EDIT_PASSES = [("/refinement/levels/0/bands/1/level", 4),
               ("/snap/feature_tolerance", 0),
               ("/layers/patches", ["wing"])]


def _oracle_factory():
    """The jsonschema Draft 2020-12 oracle, or None when it is not installed."""
    try:
        import jsonschema
        from referencing import Registry, Resource
    except ImportError:
        return None, "(jsonschema oracle: not installed)"
    registry = Registry()
    for fname in SCHEMAS.values():
        doc = _load_doc(os.path.join(SCHEMA_DIR, fname))
        registry = registry.with_resource(doc["$id"], Resource.from_contents(doc))

    def oracle(inst, kind):
        return list(jsonschema.Draft202012Validator(
            load_schema(kind), registry=registry).iter_errors(inst))

    return oracle, "(jsonschema oracle agrees)"


def _check_refs(node, root, kind):
    if isinstance(node, dict):
        if "$ref" in node:
            _resolve(node, root)  # resolves or raises; the result itself is not walked
        for k, v in node.items():
            if k == "properties":
                for ps in v.values():
                    _check_refs(ps, root, kind)
            elif k == "$defs":
                for ds in v.values():
                    _check_refs(ds, root, kind)
            elif k == "items":
                _check_refs(v, root, kind)


def selftest() -> int:
    """The eight checks behind `schema.py --selftest`; 1 and SELFTEST FAIL on any."""
    lines = []
    try:
        for kind in SCHEMAS:
            root = load_schema(kind)
            assert root.get("type") == "object" and root.get("additionalProperties") is False, kind
            _check_refs(root, root, kind)
        lines.append("[ok] schemas: %d files load, every keyword in the subset, every $ref resolves"
                     % len(SCHEMAS))

        fixtures = _read_json(FIXTURES_PATH)
        gates = _read_json(GATES_PATH)
        knobs = _read_json(KNOBS_PATH)
        oracle, note = _oracle_factory()
        n_good = 0
        for key, kind in FIXTURE_KINDS.items():
            validate(fixtures[key], kind)
            if oracle is not None:
                assert not oracle(fixtures[key], kind), "oracle rejects good %s" % key
            n_good += 1
        for inst, kind in ((gates, "GateConstants"), (knobs, "Knobs")):
            validate(inst, kind)
            if oracle is not None:
                assert not oracle(inst, kind), "oracle rejects %s" % kind
            n_good += 1
        m2 = dict(fixtures["manifest"], seed=7.0)  # draft 2020-12: 7.0 is an integer
        assert errors(m2, "ManifestRow") == [], errors(m2, "ManifestRow")
        if oracle is not None:
            assert not oracle(m2, "ManifestRow"), "oracle rejects seed 7.0"
        lines.append("[ok] good fixtures: %d valid %s" % (n_good, note))
        assert oracle is not None or note.startswith("(jsonschema oracle: not installed)")

        counters = {"type": 0, "value": 0, "missing": 0, "unknown": 0, "any": 0}
        for key, kind in FIXTURE_KINDS.items():
            _walk_bad(fixtures[key], load_schema(kind), load_schema(kind),
                      [("$", "")], kind, counters, oracle)
        for inst, kind in ((gates, "GateConstants"), (knobs, "Knobs")):
            _walk_bad(inst, load_schema(kind), load_schema(kind), [("$", "")], kind,
                      counters, oracle)
        n_bad = counters["type"] + counters["value"] + counters["missing"] + counters["unknown"]
        lines.append("[ok] bad fixtures: %d refused by field name (%d wrong type, %d wrong enum/const "
                     "value, %d missing, %d unknown field; %d any-typed skipped), oracle refuses all %d"
                     % (n_bad, counters["type"], counters["value"], counters["missing"],
                        counters["unknown"], counters["any"], n_bad))

        row0 = fixtures["attempt"]
        assert check_attempt(row0, gates, knobs) == [], "the good attempt row fails a semantic check"
        paths = ["attempt", "config_delta[0].pointer", "rule_id", "rule_id", "prediction",
                 "prediction.t_predicted", "t_end", "outcome.failure",
                 "outcome.strict_failure", "outcome.verdict", "outcome.patches[0].delivered",
                 "fingerprint.geometry_id", "outcome.flags.F3a", "outcome.flags.F3b",
                 "outcome.flags.F3c", "outcome.flags.F5", "outcome.flags.F1",
                 "outcome.feature_capture", "outcome.flags.F3e", "outcome.flags.F3e"]
        bads = []
        b = dict(row0); b["attempt"] = gates["attempts_k"] + 1; bads.append(b)
        b = dict(row0); b["config_delta"] = [dict(row0["config_delta"][0],
                pointer="/layers/min_thickness")]; bads.append(b)
        b = dict(row0); b["decided_by"] = "default"; bads.append(b)
        b = dict(row0); b["decided_by"] = "rule"; b["rule_id"] = None; bads.append(b)
        b = dict(row0); b["decided_by"] = "optimiser"; bads.append(b)
        b = dict(row0); b["prediction"] = {"p_fail": 0.1, "p_fail_std": 0.02,
                "blc8_a_priori": 0.5, "log_cells": 4.9, "t_predicted": row0["t_start"]}
        bads.append(b)
        b = dict(row0); b["t_end"] = "2026-09-23T08:00:00Z"; bads.append(b)
        b = dict(row0); b["outcome"] = dict(row0["outcome"], failure=False,
                verdict="pass"); bads.append(b)
        b = dict(row0); b["outcome"] = dict(row0["outcome"], strict_failure=False); bads.append(b)
        b = dict(row0); b["outcome"] = dict(row0["outcome"], verdict="pass"); bads.append(b)
        b = dict(row0); b["outcome"] = dict(row0["outcome"], patches=[
                dict(row0["outcome"]["patches"][0], delivered=True)]); bads.append(b)
        b = dict(row0); b["fingerprint"] = dict(row0["fingerprint"],
                geometry_id="other-geom"); bads.append(b)
        b = dict(row0); b["outcome"] = dict(row0["outcome"], flags=dict(
                row0["outcome"]["flags"], F3a=False)); bads.append(b)
        b = dict(row0); b["outcome"] = dict(row0["outcome"], flags=dict(
                row0["outcome"]["flags"], F3b=False)); bads.append(b)
        b = dict(row0); b["outcome"] = dict(row0["outcome"], flags=dict(
                row0["outcome"]["flags"], F3c=False)); bads.append(b)
        b = dict(row0); b["outcome"] = dict(row0["outcome"], flags=dict(
                row0["outcome"]["flags"], F5=True)); bads.append(b)
        b = dict(row0); b["outcome"] = dict(row0["outcome"], flags=dict(
                row0["outcome"]["flags"], F1=True)); bads.append(b)
        b = dict(row0); b["outcome"] = dict(row0["outcome"], feature_capture=0.0,
                flags=dict(row0["outcome"]["flags"])); bads.append(b)
        b = dict(row0); b["outcome"] = dict(row0["outcome"], feature_capture=0.0,
                flags=dict(row0["outcome"]["flags"], F3e=False)); bads.append(b)
        b = dict(row0); b["outcome"] = dict(row0["outcome"], feature_capture=1.0,
                flags=dict(row0["outcome"]["flags"], F3e=None)); bads.append(b)
        for b, path in zip(bads, paths):
            msgs = check_attempt(b, gates, knobs)
            assert any(m.startswith("attempt: %s: " % path) for m in msgs),                 "S-check for %s did not fire: %s" % (path, msgs)
        lines.append("[ok] semantic refusals: 20 by name")
        good = dict(row0); good["outcome"] = dict(row0["outcome"], feature_capture=None,
                flags=dict(row0["outcome"]["flags"], F3e=False))
        assert check_attempt(good, gates, knobs) == [], "the F3e-positive row fails"

        k_names = 0
        for kind in SCHEMAS:
            doc = load_schema(kind)
            assert a_priori_name_violations(doc) == [], "a real schema violates the naming rule"
            k_names += count_yplus_names(doc)
        planted = {"properties": {"yplus_mean": {"type": "number"}}}
        viol = a_priori_name_violations(planted)
        assert len(viol) == 1 and "yplus_mean" in viol[0], viol
        assert k_names >= 5, k_names
        lines.append("[ok] a-priori naming: %d y+-bearing fields carry _a_priori; "
                     "a planted yplus_mean is refused" % k_names)

        flow = fixtures["flow"]
        wall = a_priori_wall(flow, 1.0)
        t1 = wall["t1_a_priori_m"]
        assert abs(t1 - 7.29699e-4) / 7.29699e-4 < 0.01, t1
        assert abs(wall["cf_a_priori"] - 1.328 / math.sqrt(2e4)) < 1e-12
        assert abs(yplus_a_priori(t1, flow) - 1.0) < 1e-9
        cf6 = flat_plate_cf(1e6)
        assert abs(cf6 - 4.4708e-3) < 1e-6, cf6
        assert flat_plate_cf(5e5) == 0.455 / math.log10(5e5) ** 2.58
        for bad_re in (0, -1.0, float("nan"), float("inf"), None):
            assert flat_plate_cf(bad_re) is None, bad_re
        lines.append("[ok] flat-plate C_F (SPEC-LIT §32.5.6): Re_L 2e4 -> t1 = %.3e m; "
                     "Re_L 1e6 -> C_F = %.3e; 5e5 takes the turbulent branch" % (t1, cf6))

        wl = [r["pointer"] for r in knobs["whitelist"]]
        assert len(wl) == 21 and len(set(wl)) == 21, wl
        forb = {(f["pointer"], f["match"]) for f in knobs["forbidden"]}
        assert forb == {("/quality", "prefix"), ("/layers/cell_frac", "exact"),
                        ("/layers/medial_frac", "exact")}, forb
        assert [f["flag"] for f in knobs["forbidden_flags"]] == ["-permissive"]
        for ptr in wl:  # docs/15 §C, §I-5: no limiter and no gate threshold is a knob
            assert not (ptr == "/quality" or ptr.startswith("/quality/")
                        or ptr in ("/layers/cell_frac", "/layers/medial_frac")), ptr
        for (ptr, val), rid in EDIT_REFUSALS:
            ref = check_edit(ptr, val, knobs)
            assert ref is not None and ref["rule_id"] == rid, (ptr, rid, ref)
            assert ref["message"].startswith(rid), ref["message"]
            validate(ref, "DecisionRecord")
            if oracle is not None:
                assert not oracle(ref, "DecisionRecord"), "oracle rejects the %s refusal" % rid
        for ptr, val in EDIT_PASSES:
            assert check_edit(ptr, val, knobs) is None, ptr
        fl = check_flags(["-permissive"], knobs)
        assert len(fl) == 1 and fl[0]["rule_id"] == "WL-FLAG", fl
        assert fl[0]["message"].startswith("WL-FLAG")
        validate(fl[0], "DecisionRecord")
        assert check_flags(["-tag", "x"], knobs) == []
        lines.append("[ok] knobs: %d whitelisted, forbidden /quality /layers/cell_frac "
                     "/layers/medial_frac -permissive; %d edit refusals by rule id, %d passes"
                     % (len(wl), len(EDIT_REFUSALS), len(EDIT_PASSES)))

        lock_text = open(LOCK_PATH, encoding="utf-8").read()
        want = lock_lines(gates, knobs)
        got = [l for l in lock_text.splitlines() if l and not l.startswith("#")]
        assert got == want, "gates.lock disagrees with the files it locks"
        whole = want[0]
        for seed in ("1", "987654321"):
            env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONIOENCODING="utf-8")
            p = subprocess.run([sys.executable, os.path.join(HERE, "schema.py"), "--print-lock"],
                               capture_output=True, text=True, encoding="utf-8", env=env,
                               timeout=120)
            assert p.returncode == 0, p.stderr
            assert p.stdout.splitlines() == want, \
                "child with PYTHONHASHSEED=%s printed a different lock" % seed
        F = 0
        for key in gates:
            v = gates[key]
            g2 = dict(gates)
            if isinstance(v, bool):
                g2[key] = not v
            elif isinstance(v, int):
                g2[key] = v + 1
            elif isinstance(v, float):
                g2[key] = v * 1.5
            else:
                g2[key] = v + "x"
            refus = check_gates_against_lock(g2, knobs, lock_text)
            assert any(r.startswith("GATES-LOCK: %s:" % key) for r in refus), (key, refus)
            F += 1
        k2 = json.loads(json.dumps(knobs))
        k2["schema"] = "autonomy-knobs/2"
        refus = check_gates_against_lock(gates, k2, lock_text)
        assert any(r.startswith("KNOBS-LOCK:") for r in refus), refus
        before = open(LOCK_PATH, "rb").read()
        pw = subprocess.run([sys.executable, os.path.join(HERE, "schema.py"), "--write-lock"],
                            capture_output=True, text=True, encoding="utf-8",
                            env=dict(os.environ, PYTHONIOENCODING="utf-8"), timeout=120)
        after = open(LOCK_PATH, "rb").read()
        assert pw.returncode == 1 and before == after, (pw.returncode, pw.stdout)
        lines.append("[ok] gates lock: %s stable across 2 processes and equal to gates.lock; "
                     "%d per-field tampers refused by name; --write-lock refuses an existing lock"
                     % (whole[:16], F))
    except (AssertionError, SchemaError, LockError, OSError, KeyError) as e:
        print("SELFTEST FAIL: %s" % e)
        return 1
    for l in lines:
        print(l)
    print("SELFTEST PASS")
    return 0


_LOCK_HEADER = [
    "# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). No GPL-licensed source was consulted.",
    '# sha256 over canonical JSON (sort_keys, separators (",", ":"), ensure_ascii, allow_nan=False) of the parsed file / field',
    "# locked 2026-09-23 before any tuning (docs/15 §F, §I-3); relocking is the user's decision - never rewrite this file to make a number pass",
]


def main(argv: list[str] | None = None) -> int:
    """--selftest | --print-lock | --write-lock | --validate KIND FILE."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--selftest" in argv:
        return selftest()
    if "--print-lock" in argv:
        try:  # the lock is LF-terminated; do not let Windows translate the newlines
            sys.stdout.reconfigure(newline="")
        except (AttributeError, ValueError):
            pass
        for line in lock_lines(_read_json(GATES_PATH), _read_json(KNOBS_PATH)):
            print(line)
        return 0
    if "--write-lock" in argv:
        if os.path.exists(LOCK_PATH):
            print("GATES-LOCK: gates.lock exists; relocking is the user's decision (docs/15 §F)")
            return 1
        text = "\n".join(_LOCK_HEADER + lock_lines(_read_json(GATES_PATH),
                                                   _read_json(KNOBS_PATH))) + "\n"
        with open(LOCK_PATH, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        print("locked: %s" % os.path.basename(LOCK_PATH))
        return 0
    if len(argv) == 3 and argv[0] == "--validate":
        errs = errors(_read_json(argv[2]), argv[1])
        if not errs:
            print("valid")
            return 0
        for e in errs:
            print(e)
        return 1
    print("usage: schema.py --selftest | --print-lock | --write-lock | --validate KIND FILE\n"
          "kinds: " + ", ".join(sorted(SCHEMAS)))
    return 2


if __name__ == "__main__":
    sys.exit(main())
