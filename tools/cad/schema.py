#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""schema.py - a stdlib JSON-Schema draft 2020-12 subset validator for the CAD loop (docs/16 §D), on the pattern of this repository's tools/autonomy/schema.py.

Usage:
  python schema.py --selftest
  python schema.py --validate SCHEMA INSTANCE_FILE
"""

import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA_DIR = os.path.join(HERE, "schema")   # CAD-02 fills it; may not exist yet
ALLOWED_KEYWORDS = frozenset([              # the same 23 as tools/autonomy/schema.py
    "$schema", "$id", "$comment", "$defs", "$ref", "title", "description",
    "type", "properties", "required", "additionalProperties", "enum", "const",
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "minLength",
    "maxLength", "pattern", "items", "minItems", "maxItems"])
JSON_TYPES = frozenset(["null", "boolean", "integer", "number", "string", "array", "object"])

_USAGE = ("usage: python schema.py --selftest | --validate SCHEMA INSTANCE_FILE" + chr(10))

_CACHE = {}


class SchemaError(ValueError):
    """The SCHEMA is malformed: an unknown keyword, a broken $ref, a bad type name, a missing file."""


class ValidationError(ValueError):
    """An INSTANCE violates a schema; str() is the first error string."""


def schema_path(kind: str) -> str:
    """Where the schema of a kind lives: cad-template/1 -> schema/cad-template-1.schema.json."""
    return os.path.join(SCHEMA_DIR, kind.replace("/", "-") + ".schema.json")


def check_schema(sch: dict, label: str = "schema") -> None:
    """Refuse any schema outside the keyword subset, naming the keyword and its location."""
    _check_node(sch, label, "$", sch)


def _check_node(node, label, loc, root):
    if not isinstance(node, dict):
        raise SchemaError("%s: %s: items is not a schema" % (label, loc))
    for k in node:
        if k not in ALLOWED_KEYWORDS:
            raise SchemaError("%s: %s: unknown schema keyword %r" % (label, loc, k))
    t = node.get("type")
    if t is not None:
        for name in ([t] if isinstance(t, str) else list(t)):
            if name not in JSON_TYPES:
                raise SchemaError("%s: %s: unknown type %r" % (label, loc, name))
    ref = node.get("$ref")
    if ref is not None:
        if not re.match(r"^#/[$]defs/[A-Za-z0-9_]+$", ref):
            raise SchemaError("%s: %s: $ref %r is not #/$defs/<Name>" % (label, loc, ref))
        if ref.split("/")[-1] not in root.get("$defs", {}):
            raise SchemaError("%s: %s: $ref %r names a missing def" % (label, loc, ref))
    base = "" if loc == "$" else loc
    for k, v in node.items():
        if k == "properties":
            for pname, psch in v.items():
                _check_node(psch, label, base + "/properties/" + pname, root)
        elif k == "$defs":
            for dname, dsch in v.items():
                _check_node(dsch, label, base + "/$defs/" + dname, root)
        elif k == "items":
            _check_node(v, label, base + "/items", root)


def load_schema(kind_or_path: str) -> dict:
    """Load (and cache) one schema by path or by kind like cad-template/1."""
    if kind_or_path.endswith(".json"):
        kind, path = None, kind_or_path
    else:
        kind, path = kind_or_path, schema_path(kind_or_path)
    key = os.path.normpath(os.path.abspath(path))
    sch = _CACHE.get(key)
    if sch is None:
        if not os.path.exists(path):
            raise SchemaError("no schema file for %r: %s" % (kind_or_path, path))
        with open(path, "r", encoding="utf-8") as f:
            sch = json.load(f)
        check_schema(sch, sch.get("$id") or os.path.basename(path))
        _CACHE[key] = sch
    if kind is not None and sch.get("$id") != kind:   # checked on a cache hit too
        raise SchemaError("schema $id %r does not match the requested kind %r"
                          % (sch.get("$id"), kind))
    return sch


def _resolve(sch, root):
    """Local $ref only: #/$defs/Name -> root["$defs"]["Name"] (check_schema vetted the form)."""
    hops = 0
    while isinstance(sch, dict) and "$ref" in sch:
        sch = root["$defs"][sch["$ref"].split("/")[-1]]
        hops += 1
        if hops > 32:
            raise SchemaError("$ref chain does not terminate")
    return sch, root


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


def errors(instance, sch) -> list:
    """Every violation of `sch` (a dict, or a str for load_schema) by `instance`, in order."""
    if isinstance(sch, str):
        sch = load_schema(sch)
    else:
        check_schema(sch, sch.get("$id") or "schema")   # an inline dict gets the same subset gate
    errs = []
    _validate(instance, sch, sch, "$", sch.get("$id") or "schema", errs)
    return errs


def is_valid(instance, sch) -> bool:
    """True when errors() is empty."""
    return not errors(instance, sch)


def validate(instance, sch) -> None:
    """Raise ValidationError(errors[0]) when the instance violates the schema."""
    errs = errors(instance, sch)
    if errs:
        raise ValidationError(errs[0])


_PROBE = os.path.join(HERE, "fixtures", "validator", "probe.schema.json")
_CASES = os.path.join(HERE, "fixtures", "validator", "cases.json")
_GOOD = {"schema": "cad-probe/1", "name": "nozzle_a", "kind": "nozzle",
         "count": 1, "ratio": 0.5, "tags": [], "items": [{"id": "r1", "value": 1.0}]}


def selftest() -> int:
    """Prove the validator: probe schema, 12 fixtures, the oracle, semantics, refusals, CLI."""
    sch = load_schema(_PROBE)
    assert sch["$id"] == "cad-probe/1", sch.get("$id")
    assert errors(_GOOD, sch) == [], errors(_GOOD, sch)
    bad_row = dict(_GOOD, items=[{"id": "r1", "value": "x"}])
    assert errors(bad_row, sch) == \
        ["cad-probe/1: items[0].value: expected number, got string"], errors(bad_row, sch)
    print("[ok] probe schema: keyword subset loads and #/$defs/Row resolves")

    with open(_CASES, "r", encoding="utf-8") as f:
        cases = json.load(f)
    n_v = n_i = 0
    for c in cases["cases"]:
        got = errors(c["instance"], sch)
        if c["valid"]:
            assert got == [], (c["name"], got)
            n_v += 1
        else:
            assert got == c["errors"], (c["name"], got, c["errors"])
            n_i += 1
    assert (n_v, n_i) == (4, 8), (n_v, n_i)
    print("[ok] fixtures: 12 judged right (4 valid, 8 invalid)")

    try:
        import jsonschema
    except ImportError:
        print("[ok] oracle: skipped, jsonschema not installed")
    else:
        v = jsonschema.Draft202012Validator(sch)
        for c in cases["cases"]:
            assert bool(v.is_valid(c["instance"])) is c["valid"], \
                "oracle disagrees on %s" % c["name"]
        print("[ok] oracle: jsonschema %s agrees on 12"
              % importlib.metadata.version("jsonschema"))

    assert errors(True, {"type": "integer"}) == \
        ["schema: $: expected integer, got boolean"], errors(True, {"type": "integer"})
    assert errors(True, {"type": "number"}), "bool must not be a number"
    assert errors(True, {"enum": [1]}), "enum [1] must refuse True"
    assert errors(True, {"enum": [True]}) == [], errors(True, {"enum": [True]})
    assert errors(1.0, {"const": 1}) == [], "const 1 must accept 1.0"
    assert errors(1.0, {"const": True}), "const True must refuse 1.0"
    assert errors(3.0, {"type": "integer"}) == [], "an integral float is an integer"
    assert errors(3.5, {"type": "integer"}), "3.5 is not an integer"
    assert errors(0, {"exclusiveMinimum": 0}) == \
        ["schema: $: at or below exclusive minimum 0"], errors(0, {"exclusiveMinimum": 0})
    assert errors(0, {"minimum": 1}) == ["schema: $: below minimum 1"]
    assert errors(0, {"exclusiveMinimum": 1}) == \
        ["schema: $: at or below exclusive minimum 1"]
    assert errors(2, {"maximum": 1}) == ["schema: $: above maximum 1"]
    assert errors(2, {"exclusiveMaximum": 2}) == \
        ["schema: $: at or above exclusive maximum 2"]
    for kw, bound, inst, msg in (
            ("minLength", 2, "a", "shorter than 2"),
            ("maxLength", 3, "abcd", "longer than 3"),
            ("minItems", 2, [], "fewer than 2 items"),
            ("maxItems", 2, [1, 2, 3], "more than 2 items")):
        e = errors(inst, {kw: bound})
        assert e == ["schema: $: " + msg], (kw, e)
        field = "s" if isinstance(inst, str) else "xs"
        sch2 = {"type": "object", "properties": {field: {kw: bound}}}
        e = errors({field: inst}, sch2)
        assert e == ["schema: %s: %s" % (field, msg)], (kw, e)
    assert errors("ABC", {"pattern": "^[a-z]+$"}) == \
        ["schema: $: does not match ^[a-z]+$"], errors("ABC", {"pattern": "^[a-z]+$"})
    print("[ok] semantics: bool is no number, 1 == 1.0, True != 1, every message names its field")

    for kw in ("oneOf", "anyOf", "allOf", "if", "frobnicate"):
        spots = (({kw: {}}, "$"),
                 ({"properties": {"x": {kw: {}}}}, "/properties/x"),
                 ({"items": {kw: {}}}, "/items"),
                 ({"$defs": {"T": {kw: {}}}}, "/$defs/T"))
        for bad, loc in spots:
            try:
                check_schema(bad, "t")
            except SchemaError as e:
                assert kw in str(e) and loc in str(e), (loc, str(e))
            else:
                raise AssertionError("check_schema accepted %r at %s" % (kw, loc))
    try:
        check_schema({"$ref": "other.json#/x"}, "t")
    except SchemaError as e:
        assert "is not #/$defs/<Name>" in str(e) and "other.json#/x" in str(e), str(e)
    else:
        raise AssertionError("accepted a foreign $ref")
    try:
        check_schema({"$ref": "#/$defs/Gone"}, "t")
    except SchemaError as e:
        assert "names a missing def" in str(e), str(e)
    else:
        raise AssertionError("accepted a $ref to a missing def")
    try:
        check_schema({"type": "float"}, "t")
    except SchemaError as e:
        assert "unknown type" in str(e) and "'float'" in str(e), str(e)
    else:
        raise AssertionError("accepted a type name outside the 7 JSON types")
    try:
        check_schema({"type": ["string", "float"]}, "t")
    except SchemaError as e:
        assert "unknown type" in str(e), str(e)
    else:
        raise AssertionError("accepted a bad type name inside a list")
    try:
        check_schema({"items": []}, "t")
    except SchemaError as e:
        assert "items is not a schema" in str(e), str(e)
    else:
        raise AssertionError("accepted a list where items must be a schema")
    try:
        load_schema("cad-missing/1")
    except SchemaError as e:
        assert "no schema file for" in str(e) and "cad-missing-1.schema.json" in str(e), str(e)
    else:
        raise AssertionError("load_schema invented a schema")
    for bad in ({"oneOf": [{"type": "string"}]}, {"properties": {"x": {"anyOf": []}}},
                {"$ref": "other.json#/x"}):
        try:
            errors(5, bad)
        except SchemaError:
            pass
        else:
            raise AssertionError("errors() validated against an unchecked dict schema %r" % bad)
    global SCHEMA_DIR
    real_dir = SCHEMA_DIR
    with tempfile.TemporaryDirectory() as td:
        SCHEMA_DIR = td
        try:
            wrong = os.path.join(td, "cad-x-1.schema.json")
            with open(wrong, "w", encoding="utf-8") as f:
                json.dump({"$id": "cad-y/1", "type": "object"}, f)
            load_schema(wrong)
            try:
                load_schema("cad-x/1")
            except SchemaError as e:
                assert "cad-y/1" in str(e) and "cad-x/1" in str(e), str(e)
            else:
                raise AssertionError("a cached path load let a kind lookup skip the $id check")
        finally:
            SCHEMA_DIR = real_dir
            _CACHE.clear()
    print("[ok] refusals: oneOf/anyOf/allOf/if/unknown at 4 spots, bad $refs, bad types, missing file")

    with tempfile.TemporaryDirectory() as td:
        for idx, want_rc in ((0, 0), (4, 1)):
            case = cases["cases"][idx]
            ipath = os.path.join(td, case["name"] + ".json")
            with open(ipath, "w", encoding="utf-8") as f:
                json.dump(case["instance"], f)
            p = subprocess.run(
                [sys.executable, os.path.join(HERE, "schema.py"), "--validate", _PROBE, ipath],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                env=dict(os.environ, PYTHONIOENCODING="utf-8"), timeout=600)
            assert p.returncode == want_rc, \
                (case["name"], p.returncode, p.stdout, p.stderr)
            if want_rc == 0:
                assert p.stdout.strip() == "valid", p.stdout
            else:
                assert p.stdout.strip().splitlines() == case["errors"], p.stdout
        p = subprocess.run(
            [sys.executable, os.path.join(HERE, "schema.py"), "--validate", _PROBE, _PROBE],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            env=dict(os.environ, PYTHONIOENCODING="utf-8"), timeout=600)
        assert p.returncode == 1 and "required field missing" in p.stdout, p.stdout
    print("[ok] cli: --validate prints valid / exits 1 with the errors")
    print("SELFTEST PASS")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ["--selftest"]:
        return selftest()
    if len(argv) == 3 and argv[0] == "--validate":
        sch = load_schema(argv[1])
        with open(argv[2], "r", encoding="utf-8") as f:
            inst = json.load(f)
        errs = errors(inst, sch)
        if not errs:
            print("valid")
            return 0
        for e in errs:
            print(e)
        return 1
    sys.stderr.write(_USAGE)
    return 2


if __name__ == "__main__":
    sys.exit(main())
