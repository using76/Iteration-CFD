#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""schema_fixtures.py - the CAD-02 gate over tools/cad/schema/ and tools/cad/fixtures/schema/.

The nine flat CAD schemas of docs/16 (§E.1, §E.4, §E.6-§E.7, §G, §H.1) are judged against the
supervisor's case files: every valid instance passes the CAD-01 validator with no error, and
every invalid case is a single-field mutation of the first valid instance that the validator
refuses with exactly the case's own error list, whose first string names the mutated field.
A flatness check refuses any schema that is not flat: every object node closed and fully
required, nullable pairs only, object depth at most 3, shared $defs only, every def
referenced, no union keyword anywhere. The flatness rule is proved against five planted
non-flat schemas. `jsonschema` (MIT) is an optional oracle, imported only inside selftest().

Usage:
  python schema_fixtures.py --selftest
"""

import copy
import importlib.metadata
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common
import schema

CASES_DIR = os.path.join(HERE, "fixtures", "schema")
KINDS = ("cad-template/1", "cad-params/1", "cad-requirements/1", "cad-checks/1", "cad-measure/1",
         "cad-verdict/1", "cad-iteration/1", "cad-decision/1", "cad-gates/1")
DRAFT = "https://json-schema.org/draft/2020-12/schema"
UNION_WORDS = ("oneOf", "anyOf", "allOf")
MAX_DEPTH = 3
SHARED_DEFS = {
    "Name": {"type": "string", "pattern": "^[A-Za-z][A-Za-z0-9_]{0,63}$"},
    "NameOrNull": {"type": ["string", "null"], "pattern": "^[A-Za-z][A-Za-z0-9_]{0,63}$"},
    "Sha": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
    "ShaOrNull": {"type": ["string", "null"], "pattern": "^[0-9a-f]{64}$"},
    "TemplateId": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,63}/[1-9][0-9]*$"},
    "StudyId": {"type": "string", "pattern": "^[a-z0-9][a-z0-9_-]{0,63}$"},
    "ReqId": {"type": "string", "pattern": "^(REQ-[0-9]{3}|SYS-[A-Z]+)$"},
    "RuleId": {"type": "string", "pattern": "^[A-Z][A-Z0-9]*(-[A-Z0-9]+)*$"},
    "RuleIdOrNull": {"type": ["string", "null"], "pattern": "^[A-Z][A-Z0-9]*(-[A-Z0-9]+)*$"},
    "Level": {"type": "string", "pattern": "^L[0-9]$"},
    "LevelOrNull": {"type": ["string", "null"], "pattern": "^L[0-9]$"},
    "PrfId": {"type": "string", "pattern": "^PRF-[A-Z0-9]+$"},
    "OnlyWhen": {"type": ["string", "null"], "pattern": "^[A-Za-z][A-Za-z0-9_]*=[A-Za-z0-9_]+$"},
    "Unit": {"enum": ["m", "m2", "m3", "rad", "m/s", "m3/s", "kg/s", "Pa", "K", "1"]},
    "Hardness": {"enum": ["hard", "soft", "objective"]},
    "Op": {"enum": ["<=", ">=", "==", "in", "is_true"]},
}


def slug(kind: str) -> str:
    """The file stem of a kind: cad-template/1 -> cad-template-1."""
    return kind.replace("/", "-")


def _is_object_node(node) -> bool:
    """True when the schema node describes an object (type object, or a nullable pair)."""
    t = node.get("type")
    return t == "object" or (isinstance(t, list) and "object" in t)


def _collect_refs(node, out: list) -> None:
    """Every $defs name any $ref in the file names, found by a full recursive scan."""
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str):
            out.append(ref.split("/")[-1])
        for v in node.values():
            _collect_refs(v, out)
    elif isinstance(node, list):
        for v in node:
            _collect_refs(v, out)


def _walk(node, loc, depth, problems, refs) -> None:
    """Collect flatness problems at one schema node, then descend like schema._check_node."""
    if not isinstance(node, dict):
        return
    props = node.get("properties")
    if not isinstance(props, dict):
        props = None
    is_obj = _is_object_node(node)
    if is_obj:
        if node.get("additionalProperties") is not False:
            problems.append(loc + ": additionalProperties is not false")
        if node.get("required") != list(props or {}):
            problems.append(loc + ": required is not the property list")
        if depth > MAX_DEPTH:
            problems.append("%s: object depth %d above %d" % (loc, depth, MAX_DEPTH))
    t = node.get("type")
    if isinstance(t, list) and (len(t) != 2 or t[1] != "null" or t[0] == "null"):
        problems.append(loc + ": type list is not [T, null]")
    for word in UNION_WORDS:
        if word in node:
            problems.append("%s: union keyword %s" % (loc, word))
    base = "" if loc == "$" else loc
    down = depth + 1 if is_obj else depth
    if props:
        for pname, psch in props.items():
            _walk(psch, base + "/properties/" + pname, down, problems, refs)
    if "items" in node:
        _walk(node["items"], base + "/items", down, problems, refs)
    defs = node.get("$defs")
    if isinstance(defs, dict):
        for dname, dsch in defs.items():
            dloc = base + "/$defs/" + dname
            if dname not in SHARED_DEFS or dsch != SHARED_DEFS[dname]:
                problems.append("%s: def %s is not a shared def" % (dloc, dname))
            if dname not in refs:
                problems.append("%s: def %s is never referenced" % (dloc, dname))
            _walk(dsch, dloc, down, problems, refs)


def flat_problems(sch: dict) -> list:
    """[] when the schema is flat; else one string per problem, each naming its location."""
    problems = []
    refs = []
    _collect_refs(sch, refs)
    _walk(sch, "$", 1, problems, refs)
    return problems


def apply_mutation(instance, case: dict):
    """A NEW deep copy of the instance with the case's one mutation applied; never edits in place."""
    mut = copy.deepcopy(instance)
    steps = []
    for seg in str(case["path"]).split("."):
        pieces = seg.split("[")
        steps.append(pieces[0])
        for extra in pieces[1:]:
            steps.append(int(extra[:-1]))
    node = mut
    for step in steps[:-1]:
        node = node[step]
    last = steps[-1]
    if case["op"] == "set":
        node[last] = case["value"]
    elif case["op"] == "delete":
        del node[last]
    else:
        raise AssertionError("case %r: unknown mutation op %r" % (case.get("name"), case.get("op")))
    return mut


def judge_kind(kind: str) -> tuple:
    """Judge one kind against its case file; (n_valid, n_invalid); AssertionError on any miss."""
    sch = schema.load_schema(kind)
    cases = common.read_json(os.path.join(CASES_DIR, slug(kind) + ".cases.json"))
    assert cases["kind"] == kind, "%s: case file declares kind %r" % (kind, cases.get("kind"))
    valid = cases["valid"]
    invalid = cases["invalid"]
    assert len(valid) >= 1, "%s: no valid case" % kind
    assert len(invalid) >= 3, "%s: only %d invalid cases" % (kind, len(invalid))
    for case in valid:
        errs = schema.errors(case["instance"], sch)
        assert errs == [], "%s: valid case %s judged invalid: %s" % (
            kind, case["name"], "; ".join(errs))
    for case in invalid:
        mut = apply_mutation(valid[0]["instance"], case)
        assert mut != valid[0]["instance"], "%s: case %s changed nothing" % (kind, case["name"])
        errs = schema.errors(mut, sch)
        assert errs == case["errors"], "%s: case %s: got %r, want %r" % (
            kind, case["name"], errs, case["errors"])
        head = ": " + case["path"] + ": "
        assert head in case["errors"][0], "%s: case %s: first error does not name the path" % (
            kind, case["name"])
    return (len(valid), len(invalid))


def _inventory() -> dict:
    """The inventory and top-level checks of the selftest; returns {kind: (n_valid, n_invalid)}."""
    names = sorted(os.listdir(schema.SCHEMA_DIR))
    want = sorted(slug(k) + ".schema.json" for k in KINDS)
    assert names == want, "tools/cad/schema holds %r, want %r" % (names, want)
    cnames = sorted(os.listdir(CASES_DIR))
    cwant = sorted(slug(k) + ".cases.json" for k in KINDS)
    assert cnames == cwant, "fixtures/schema holds %r, want %r" % (cnames, cwant)
    for k in KINDS:
        with open(schema.schema_path(k), "r", encoding="utf-8") as f:
            raw = f.read()
        for word in UNION_WORDS:
            assert word not in raw, "%s: raw text contains %r" % (k, word)
        sch = schema.load_schema(k)
        keys = list(sch)
        assert keys[:9] == ["$comment", "$schema", "$id", "title", "description", "type",
                            "additionalProperties", "required", "properties"], \
            "%s: top-level key order is %r" % (k, keys)
        assert keys[9:] == ["$defs"], "%s: trailing top-level keys are %r" % (k, keys[9:])
        assert sch["$schema"] == DRAFT, "%s: $schema is %r" % (k, sch.get("$schema"))
        assert sch["$id"] == k, "%s: $id is %r" % (k, sch.get("$id"))
        assert sch["$comment"] == common.HEADER_COMMENT, "%s: $comment is not the header" % k
        assert isinstance(sch["title"], str) and sch["title"], "%s: empty title" % k
        assert isinstance(sch["description"], str) and sch["description"], "%s: no description" % k
    assert schema.__file__.endswith("schema.py"), "import schema resolved to %r" % (schema.__file__,)
    counts = {}
    for k in KINDS:
        counts[k] = judge_kind(k)
    return counts


def _planted() -> tuple:
    """Five non-flat schemas; flat_problems must name each one's reason word."""
    deep = {"type": "object", "additionalProperties": False, "required": ["s"],
            "properties": {"s": {"type": "string"}}}
    for _ in range(3):
        deep = {"type": "object", "additionalProperties": False, "required": ["p"],
                "properties": {"p": deep}}
    return (
        ("open-root",
         {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]},
         "additionalProperties is not false"),
        ("empty-required",
         {"type": "object", "properties": {"a": {"type": "string"}},
          "additionalProperties": False, "required": []},
         "required is not the property list"),
        ("wide-type-pair",
         {"type": "object", "properties": {"x": {"type": ["string", "number"]}},
          "additionalProperties": False, "required": ["x"]},
         "type list is not [T, null]"),
        ("too-deep", deep, "object depth 4 above 3"),
        ("orphan-def",
         {"type": "object", "properties": {}, "additionalProperties": False, "required": [],
          "$defs": {"Sha": SHARED_DEFS["Sha"]}},
         "def Sha is never referenced"),
    )


def selftest() -> int:
    """The 13 [ok] lines of docs/16 §I CAD-02, then SELFTEST PASS; 13 counted by selftest.py."""
    counts = _inventory()
    n_valid = sum(c[0] for c in counts.values())
    n_invalid = sum(c[1] for c in counts.values())
    print("[ok] inventory: 9 schemas, 9 case files, %d valid, %d invalid" % (n_valid, n_invalid))
    for k in KINDS:
        problems = flat_problems(schema.load_schema(k))
        assert problems == [], "%s: not flat: %s" % (k, "; ".join(problems))
    print("[ok] flat: 9 schemas")
    for name, planted, word in _planted():
        problems = flat_problems(planted)
        assert any(word in p for p in problems), \
            "planted %s: no %r in %r" % (name, word, problems)
    print("[ok] flat refuses 5 planted non-flat schemas")
    for k in KINDS:
        n_v, n_i = counts[k]
        print("[ok] %s: %d valid, %d invalid judged right" % (k, n_v, n_i))
    try:
        import jsonschema
        from jsonschema import Draft202012Validator
    except ImportError:
        print("[ok] oracle: skipped, jsonschema not installed")
    else:
        version = importlib.metadata.version("jsonschema")
        n = 0
        for k in KINDS:
            sch = schema.load_schema(k)
            Draft202012Validator.check_schema(sch)
            oracle = Draft202012Validator(sch)
            cases = common.read_json(os.path.join(CASES_DIR, slug(k) + ".cases.json"))
            for case in cases["valid"]:
                assert oracle.is_valid(case["instance"]) is True, \
                    "%s: oracle refuses valid case %s" % (k, case["name"])
                n += 1
            for case in cases["invalid"]:
                mut = apply_mutation(cases["valid"][0]["instance"], case)
                assert oracle.is_valid(mut) is False, \
                    "%s: oracle accepts invalid case %s" % (k, case["name"])
                n += 1
        print("[ok] oracle: jsonschema %s agrees on %d instances" % (version, n))
    print("SELFTEST PASS")
    return 0


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if list(argv) == ["--selftest"]:
        return selftest()
    sys.stderr.write("usage: python schema_fixtures.py --selftest" + common.NL)
    return 2


if __name__ == "__main__":
    sys.exit(main())
