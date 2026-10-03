#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""reqs.py - stage S1 of the CAD loop (docs/16 §D, §E.1-§E.4, §E.8): check an LLM's flat requirement rows against the brief and the frozen template, lock the approved set, and compile it into typed assertions.

EARS sentence forms: Mavin et al., "Easy Approach to Requirements Syntax (EARS)", RE 2009, DOI 10.1109/RE.2009.9.

Usage:
  python reqs.py --selftest
  python reqs.py check PROPOSAL_JSON BRIEF_JSON TEMPLATE_DIR OUT_JSON
  python reqs.py lock REPORT_JSON APPROVED_BY OUT_DIR
  python reqs.py compile REQUIREMENTS_JSON TEMPLATE_DIR OUT_JSON
  python reqs.py vocab TEMPLATE_DIR OUT_JSON
  python reqs.py diff OLD_DIR NEW_DIR OLD_TEMPLATE_DIR NEW_TEMPLATE_DIR OUT_JSON
  python reqs.py supersede REPORT_JSON APPROVED_BY OLD_DIR OLD_TEMPLATE_DIR NEW_TEMPLATE_DIR CHANGE_KIND CHANGE_REASON OUT_DIR [EVIDENCE]

AMG-6 (docs/16a §B.1, §E, §F) reimplements, from reading only (no code copied), four ideas of Amagine3D
(https://github.com/amagine-ai/Amagine3D, commit e608dc6, Apache-2.0): per-value provenance with a derived
confidence (skills/text-a3d/intent_contract.py `validate`, scene_contract.py `INTENT_ONLY_FIELDS`), a write-once
intent file (authoring.py `_write_json(immutable=True)`, intent_revision.py `load_history`), and an LLM vocabulary
generated from the validator's live constants and fingerprinted (capability_manifest.py `_intent_input_constraints`,
`build_manifest`).

AMG-7 (docs/16a §B.1, §D.7, §E) reimplements, from reading only (no code copied), Amagine3D's revision diff and
lineage (skills/text-a3d/intent_revision.py `semantic_diff`, `validate_revision`) without their two traps: rows are
matched by what they measure (quantity, the catalogue where as a set, condition, objective or not), never by REQ id
or position, and the ordered where is compared as a field, so a swapped area_ratio reads as a target change; and a
generated file is refused as evidence by its CONTENT sha against the superseded study's cache/ and iterations.jsonl,
not by a schema prefix it can drop.
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
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common
import schema

# reqs.py never runs the template and never measures: it judges the LLM's rows against the brief's own words and
# the template's DECLARATION (template.json), and it compiles an approved, locked set into cad-checks/1. Every
# refusal names one rule id of docs/16 §E.3; a row gets at most one refusal, the first in the order of (C5).
VERSION = 1
CAD07_REFUSAL_IDS = ("REQ-QTY", "REQ-UNIT", "REQ-QUOTE", "REQ-GROUND", "REQ-OP", "REQ-TOL", "REQ-DUP",
                     "REQ-CONFLICT", "REQ-OUTSIDE", "REQ-PIXEL", "REQ-OBJ", "REQ-DEFAULT-HARD")
REFUSAL_IDS = CAD07_REFUSAL_IDS + ("REQ-STD", "REQ-VOCAB")    # docs/16a §E; REQ-IMMUTABLE is write_locked's
REPORT_KEYS = ("version", "status", "refusals", "questions", "requirements", "derived")
PROPOSAL_KEYS = ("study_id", "template_id", "vocab_sha", "operating_point", "rows", "created_by")
OPPOINT_KEYS = ("fluid", "T_K", "p0_Pa", "flow")
OPPOINT_OPTIONAL = ("fluid_source", "T_K_source", "p0_Pa_source")   # absent -> "assumed" (docs/16a §F)
OPPOINT_SOURCES = ("brief", "sketch_label", "default", "assumed")
FLOW_KEYS = ("field", "value", "unit", "source", "quote")
FLOW_FIELDS = {"U_exit_m_s": "m/s", "Q_m3_s": "m3/s", "mdot_kg_s": "kg/s"}
FLOW_SOURCES = ("brief", "sketch_label", "assumed")
ROW_IN_KEYS = ("quantity", "feature", "op", "value", "upper", "tol_abs", "tol_rel", "unit", "condition",
               "hardness", "source", "quote")
ROW_IN_OPTIONAL = ("ears", "ticked", "standard_ref")   # ears: the LLM's draft, discarded; ticked: the card's tick
ROW_KEYS = ("id", "ears", "quantity", "feature", "op", "value", "upper", "tol_abs", "tol_rel", "unit", "kind",
            "method", "condition", "hardness", "source", "quote", "locks_params", "ticked", "confidence",
            "standard_ref")
OPS = ("<=", ">=", "==", "in", "is_true")
HARDNESS = ("hard", "soft", "objective")
SOURCES = ("brief", "sketch_label", "default", "assumed", "standard")
TICK_SOURCES = ("default", "assumed")     # a hard row from these needs the card's tick (REQ-DEFAULT-HARD)
CONFIDENCE = {"brief": "high", "sketch_label": "medium", "default": "low", "assumed": "low", "standard": "high",
              "system": "high"}           # docs/16a §F: derived from the source, never taken from the LLM
CHANGE_KINDS = ("new", "target_change", "evidence_correction")
REPRS = ("brep", "stl", "mesh", "cfd")
REPR_BY_METHOD = {"geometry": "brep", "mesh": "mesh", "cfd": "cfd"}
REPR_BY_PRIMITIVE = {"watertight": "stl"}  # judged on the named STL's stl_repair report, not on the BREP
VOCAB_VERSION = 1
EVAL_KEY_PARTS = ("template_sha", "declaration_sha", "params", "requirements_lock", "gates_lock", "env",
                  "mesh_recipe_version", "case_writer_version", "bin_sha")    # docs/16 §D, docs/16a §F
DIFF_VERSION = 1
DIFF_ROW_FIELDS = tuple(k for k in ROW_KEYS if k != "id") + ("where",)    # compared per matched row, in this order
DIFF_CLASSES = ("target_change", "added", "removed", "none")
DIFF_CONTEXT = ("template_id", "template_sha", "declaration_sha", "vocab_sha", "brief_sha", "attachments")
DIFF_HEAD = ("study_id", "lock_sha", "declaration_sha", "template_sha")
DIFF_KEYS = ("version", "old", "new", "change", "counts", "rows", "operating_point", "context_changed")
SUPERSEDE_KINDS = ("target_change", "evidence_correction")
SHA_TOKEN_RE = re.compile("(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])")
BOOLEAN_PRIMITIVES = ("valid", "watertight", "axis_x", "units_m", "separation_free")
LEVEL_RE = re.compile("^L[0-9]$")
STUDY_RE = re.compile("^[a-z0-9][a-z0-9_-]{0,63}$")
SHA_RE = re.compile("^[0-9a-f]{64}$")
PENDING = "pending"                      # approved_by before a person approves the card
GROUND_REL = 1e-12                       # a quoted number grounds a value when they agree to this relative gap
UNIT_TABLE = {  # written unit -> (SI unit, divisor); deg and ° are multiplied by pi/180 instead (divisor None)
    "mm": ("m", 1000.0), "m": ("m", 1.0), "deg": ("rad", None), "°": ("rad", None), "rad": ("rad", 1.0),
    "L/s": ("m3/s", 1000.0), "m³/s": ("m3/s", 1.0), "m3/s": ("m3/s", 1.0), "kg/s": ("kg/s", 1.0),
    "m/s": ("m/s", 1.0), "Pa": ("Pa", 1.0), "%": ("1", 100.0), "−": ("1", 1.0), "-": ("1", 1.0), "1": ("1", 1.0),
}
QUOTE_UNITS = ("m³/s", "m3/s", "m/s", "mm", "m", "kg/s", "L/s", "Pa", "deg", "°", "rad", "%")
NUM_RE = re.compile("(?<![A-Za-z0-9_.])([0-9]+(?:[.][0-9]+)?(?:[eE][-+]?[0-9]+)?)[ ]?("
                    + "|".join(re.escape(u) for u in QUOTE_UNITS) + ")?(?![A-Za-z0-9])")
# Fixed densities at 293.15 K and 101 325 Pa, the operating point of docs/16 §H.2; not a function of T_K or p0.
# water: IAPWS-95 as tabulated by the NIST Chemistry WebBook (https://webbook.nist.gov/chemistry/fluid/);
# air: ideal gas p/(R T) with R = 287.05 J/(kg K). lowmach reads massFlowRate as a volume flux, so every flow
# becomes Q in m3/s here (docs/16 §E.1).
RHO_TABLE = {"air": 1.2041, "water": 998.21}
SYS_ROWS = (  # (id, quantity, op, value, kind, method, primitive, where, u, ears) - docs/16 §E.1
    ("SYS-SOLID", "n_solids", "==", 1.0, "geometric", "geometry", "n_solids", ["fluid"], 0.0,
     "The design shall be exactly one solid."),
    ("SYS-VALID", "valid", "is_true", None, "geometric", "geometry", "valid", ["fluid"], 0.0,
     "The design shall be a valid BREP at every stage."),
    ("SYS-WATERTIGHT", "watertight", "is_true", None, "geometric", "geometry", "watertight", ["fluid"], 0.0,
     "The design shall be watertight."),
    ("SYS-AXIS", "axis", "is_true", None, "geometric", "geometry", "axis_x", ["fluid"], 0.0,
     "The design shall be revolved about the +x axis."),
    ("SYS-UNITS", "units", "is_true", None, "geometric", "geometry", "units_m", ["fluid"], 0.0,
     "The design shall be in metres at scale 1."),
    ("SYS-MACH", "mach_max", "<=", 0.3, "performance", "cfd", "mach_max", ["fluid"], None,
     "The design shall keep the Mach number at or below 0.3."),
)
QUESTIONS = {  # id -> (param, text), asked in this order
    "Q-D_i": ("D_i", "What is the inlet diameter? The template fixes D_i by requirement and has no default."),
    "Q-CR": ("CR", "What is the contraction ratio, or the exit diameter? The template fixes CR by requirement and has no default."),
    "Q-FLOW": (None, "What is the flow: the exit velocity, the volume flow or the mass flow?"),
}
NOZZLE = "nozzle_contraction/1"
LOCKS = {  # nozzle: the template parameters a HARD "==" row on this quantity fixes (sorted)
    "inlet_diameter": ["D_i"], "exit_diameter": ["CR", "D_i"], "contraction_ratio": ["CR"],
    "contraction_length": ["D_i", "L_over_Di"], "total_length": ["CR", "D_i", "L_over_Di", "Lx_over_De"],
    "min_wall_normal": ["t_wall"],
}
EARS_RE = re.compile("^(The design|While [^,]+, the design) (shall|should) [a-z].*[.]$")
FIXTURES = os.path.join(HERE, "fixtures", "reqs")
CASES = os.path.join(FIXTURES, "cases.json")
GOLDEN = os.path.join(FIXTURES, "golden")
LINEAGE = os.path.join(FIXTURES, "lineage.json")
DIFF_PAIRS = os.path.join(FIXTURES, "diff", "pairs.json")
NOZZLE_DIR = os.path.join(HERE, "templates", "nozzle_contraction")
USAGE = ("usage: python reqs.py --selftest" + chr(10)
         + "       python reqs.py check PROPOSAL_JSON BRIEF_JSON TEMPLATE_DIR OUT_JSON" + chr(10)
         + "       python reqs.py lock REPORT_JSON APPROVED_BY OUT_DIR" + chr(10)
         + "       python reqs.py compile REQUIREMENTS_JSON TEMPLATE_DIR OUT_JSON" + chr(10)
         + "       python reqs.py vocab TEMPLATE_DIR OUT_JSON" + chr(10)
         + "       python reqs.py diff OLD_DIR NEW_DIR OLD_TEMPLATE_DIR NEW_TEMPLATE_DIR OUT_JSON" + chr(10)
         + "       python reqs.py supersede REPORT_JSON APPROVED_BY OLD_DIR OLD_TEMPLATE_DIR NEW_TEMPLATE_DIR"
         + " CHANGE_KIND CHANGE_REASON OUT_DIR [EVIDENCE]")


def nfc(text) -> str:
    """NFC-normalised text: quotes and briefs are compared in this form (docs/16 §E.3 REQ-QUOTE)."""
    return unicodedata.normalize("NFC", text)


def to_si(value, unit) -> tuple:
    """(float, si_unit): the written value in the written unit, into the template's SI units (docs/16 §E.1)."""
    if not isinstance(unit, str) or unit not in UNIT_TABLE:
        raise ValueError("unit %r is not in the unit table" % (unit,))
    si, div = UNIT_TABLE[unit]
    if div is None:                                   # deg and °: multiply by pi/180
        return (float(value) * math.pi / 180.0, si)
    return (float(value) / div, si)


def quote_numbers(quote) -> list:
    """[(float, unit-or-None)] over the NFC quote: a bare number is unitless and read in the row's own unit."""
    return [(float(m.group(1)), m.group(2)) for m in NUM_RE.finditer(nfc(quote))]


def _is_num(x) -> bool:
    """A finite JSON number that is not a bool."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _is_pos(x) -> bool:
    """A finite positive JSON number that is not a bool."""
    return _is_num(x) and x > 0


def _refuse(row, rid, chk, detail) -> dict:
    """One refusal record of docs/16 §E.3."""
    return {"row": row, "id": rid, "check": chk, "detail": detail}


def render_ears(row) -> str:
    """The server-rendered EARS sentence of an OUTPUT row (SI values); the LLM's draft is never used (docs/16 §E.1)."""
    def fmt(x):
        return repr(float(x))
    q = row["quantity"].replace("_", " ")
    feat = " at " + row["feature"] if row["feature"] else ""
    us = "" if row["unit"] == "1" else " " + row["unit"]
    modal = "shall" if row["hardness"] == "hard" else "should"
    parts = []
    if row["condition"]["Re"] is not None:
        parts.append("Re is " + fmt(row["condition"]["Re"]))
    if row["condition"]["level"] is not None:
        parts.append("the mesh level is " + row["condition"]["level"])
    head = "While " + " and ".join(parts) + ", the design " if parts else "The design "
    if row["tol_abs"] is not None:
        tp = " within " + fmt(row["tol_abs"]) + us
    elif row["tol_rel"] is not None:
        tp = " within a relative " + fmt(row["tol_rel"])
    else:
        tp = ""
    op = row["op"]
    if row["hardness"] == "objective":
        body = ("minimise " if op == "<=" else "maximise ") + q + feat
    elif op == "is_true":
        body = "satisfy " + q + feat
    elif op == "<=":
        body = "have " + q + feat + " of at most " + fmt(row["value"]) + us + tp
    elif op == ">=":
        body = "have " + q + feat + " of at least " + fmt(row["value"]) + us + tp
    elif op == "==":
        body = "have " + q + feat + " of " + fmt(row["value"]) + us + tp
    else:
        body = ("have " + q + feat + " between " + fmt(row["value"]) + " and " + fmt(row["upper"]) + us + tp)
    return head + modal + " " + body + "."


def _bounds(row) -> tuple:
    """(lo, hi) of a normalised row: the compile bounds of docs/16 §E.4; objective and is_true bound nothing."""
    if row["hardness"] == "objective" or row["op"] == "is_true":
        return (None, None)
    op = row["op"]
    if op == "<=":
        return (None, row["value"])
    if op == ">=":
        return (row["value"], None)
    if op == "==":
        return (row["value"], row["value"])
    return (row["value"], row["upper"])                     # in


def _ref_of(lo, hi):
    """max(abs(x) for the set bounds), or None when both are None (docs/16 §E.4)."""
    xs = [abs(x) for x in (lo, hi) if x is not None]
    return max(xs) if xs else None


def _tol(row, ref):
    """The row's tolerance in SI: tol_abs, else tol_rel*ref, else 0.0 (docs/16 §E.4)."""
    if row.get("tol_abs") is not None:
        return row["tol_abs"]
    if row.get("tol_rel") is not None and ref is not None:
        return row["tol_rel"] * ref
    return 0.0


def _u_of(cat, ref):
    """The primitive's uncertainty u for a catalogue row: abs/rel/exact, None for cfd or a null u_meas."""
    kind = cat.get("u_kind")
    if kind == "exact":
        return 0.0
    um = cat.get("u_meas")
    if um is None:
        return None
    if kind == "abs":
        return um
    if kind == "rel":
        return None if ref is None else um * ref
    return None


def _check_brief(brief) -> None:
    """The stored brief (docs/16 §D brief.json): exactly schema, text, attachments (<= 4), provider."""
    if not isinstance(brief, dict):
        raise ValueError("brief: not an object")
    if set(brief.keys()) != {"schema", "text", "attachments", "provider"}:
        raise ValueError("brief: keys must be exactly schema, text, attachments, provider, got %s"
                         % (sorted(str(k) for k in brief.keys()),))
    if brief["schema"] != "cad-brief/1":
        raise ValueError("brief: schema must be cad-brief/1, got %r" % (brief["schema"],))
    if not isinstance(brief["text"], str):
        raise ValueError("brief: text must be a string")
    atts = brief["attachments"]
    if not isinstance(atts, list) or len(atts) > 4:
        raise ValueError("brief: attachments must be a list of at most 4 sketches")
    for a in atts:
        if not isinstance(a, dict) or set(a.keys()) != {"sha256", "labels"}:
            raise ValueError("brief: each attachment must have exactly sha256 and labels")
        if not isinstance(a["sha256"], str) or not SHA_RE.match(a["sha256"]):
            raise ValueError("brief: attachment sha256 must be 64 lowercase hex")
        if not isinstance(a["labels"], list) or not all(isinstance(l, str) for l in a["labels"]):
            raise ValueError("brief: attachment labels must be a list of strings")
    if not isinstance(brief["provider"], str) or not brief["provider"]:
        raise ValueError("brief: provider must be a non-empty string")


def _check_envelope(proposal, declaration) -> None:
    """The proposal envelope: errors here are exceptions; errors inside rows are refusals (docs/16 §D)."""
    if not isinstance(proposal, dict):
        raise ValueError("proposal: not an object")
    if set(proposal.keys()) != set(PROPOSAL_KEYS):
        raise ValueError("proposal: keys must be exactly %s, got %s"
                         % (", ".join(PROPOSAL_KEYS), sorted(str(k) for k in proposal.keys())))
    if not isinstance(proposal["study_id"], str) or not STUDY_RE.match(proposal["study_id"]):
        raise ValueError("proposal: study_id %r must match %s" % (proposal["study_id"], STUDY_RE.pattern))
    if declaration.get("template_id") != NOZZLE or proposal["template_id"] != NOZZLE:
        raise ValueError("proposal: reqs.py knows the direct functions of %r only, got template_id %r for a"
                         " declaration of %r" % (NOZZLE, proposal["template_id"], declaration.get("template_id")))
    if not isinstance(proposal["rows"], list) or not proposal["rows"]:
        raise ValueError("proposal: rows must be a non-empty list")
    cb = proposal["created_by"]
    if not isinstance(cb, dict) or set(cb.keys()) != {"provider", "model"} \
            or not isinstance(cb["provider"], str) or not cb["provider"] \
            or not isinstance(cb["model"], str) or not cb["model"]:
        raise ValueError("proposal: created_by must be an object with non-empty provider and model")


def _quote_pass(source, quote, brief, labels):
    """Step 5 of docs/16 §E.3: the quote rule. None when it passes, else (id, check, detail)."""
    if source == "brief":
        if not isinstance(quote, str) or not quote:
            return ("REQ-QUOTE", "missing", "the row cites the brief but carries no quote")
        if nfc(quote) not in nfc(brief["text"]):
            return ("REQ-QUOTE", "not_in_brief",
                    "the quote is not a verbatim span of the NFC brief: %r" % (quote,))
        return None
    if source == "sketch_label":
        if not isinstance(quote, str) or not quote:
            return ("REQ-PIXEL", "not_a_label", "the row cites a sketch label but carries no quote")
        nq = nfc(quote)
        if not any(nq in nfc(lab) for lab in labels):
            return ("REQ-PIXEL", "not_a_label",
                    "the quote is not a printed label of any attachment: %r" % (quote,))
        return None
    if quote is not None:                                    # default or assumed
        return ("REQ-QUOTE", "quote_on_default",
                "a %s row must carry no quote, got %r" % (source, quote))
    return None


def _std_pass(row, refs):
    """REQ-STD (docs/16a §E): a standard row names a ref of the template's frozen standards table; no other row
    names one. None when it passes, else (id, check, detail)."""
    ref = row.get("standard_ref")
    if row["source"] == "standard":
        if ref is None:
            return ("REQ-STD", "missing_ref", "the row is sourced standard but carries no standard_ref")
        if ref not in refs:
            return ("REQ-STD", "not_in_table", "the standard_ref %r is not in the template's standards table [%s]"
                    % (ref, ", ".join(refs)))
        return None
    if ref is not None:
        return ("REQ-STD", "ref_on_non_standard", "a %s row carries the standard_ref %r" % (row["source"], ref))
    return None


def _ground_pass(unit, quote, values):
    """Step 6 of docs/16 §E.3: every cited magnitude must appear in its quote after conversion.
    values: (name, si_value) pairs. None when it passes, else (id, check, detail)."""
    si_unit = UNIT_TABLE[unit][0]
    for name, x in values:
        if x is None:
            continue
        grounded = False
        for (n, u) in quote_numbers(quote):
            n_si, n_unit = to_si(n, u if u is not None else unit)
            if n_unit == si_unit and abs(n_si - x) <= GROUND_REL * max(abs(n_si), abs(x)):
                grounded = True
                break
        if not grounded:
            return ("REQ-GROUND", name,
                    "the %s %g in SI units is grounded in no number of the quote %r" % (name, x, quote))
    return None


def _type_pass(row):
    """Step 1 types of docs/16 §E.3. None when they pass, else the name of the first bad key."""
    if not isinstance(row["quantity"], str):
        return "quantity"
    if row["feature"] is not None and not isinstance(row["feature"], str):
        return "feature"
    if not isinstance(row["op"], str):
        return "op"
    for k in ("value", "upper", "tol_abs", "tol_rel"):
        if row[k] is not None and not _is_num(row[k]):
            return k
        if k in ("tol_abs", "tol_rel") and row[k] is not None and row[k] < 0:
            return k
    if not isinstance(row["unit"], str):
        return "unit"
    cond = row["condition"]
    if not isinstance(cond, dict) or set(cond.keys()) != {"Re", "level"}:
        return "condition"
    if cond["Re"] is not None and not _is_pos(cond["Re"]):
        return "condition Re"
    if cond["level"] is not None and not (isinstance(cond["level"], str) and LEVEL_RE.match(cond["level"])):
        return "condition level"
    if row["hardness"] not in HARDNESS:
        return "hardness"
    if row["source"] not in SOURCES:
        return "source"
    if row["quote"] is not None and not isinstance(row["quote"], str):
        return "quote"
    if "ears" in row and not isinstance(row["ears"], str):
        return "ears"
    if "ticked" in row and not isinstance(row["ticked"], bool):
        return "ticked"
    if "standard_ref" in row and row["standard_ref"] is not None and not isinstance(row["standard_ref"], str):
        return "standard_ref"
    return None


def _shape_pass(row, cat):
    """Step 4 of docs/16 §E.3: the op/value shape against the catalogue row. None or (id, check, detail)."""
    op = row["op"]
    quad = (row["value"], row["upper"], row["tol_abs"], row["tol_rel"])
    if op not in OPS:
        return ("REQ-OP", "op", "op %r is not one of %s" % (op, ", ".join(OPS)))
    if cat["primitive"] in BOOLEAN_PRIMITIVES:
        if op != "is_true":
            return ("REQ-OP", "op", "the boolean quantity %s only takes is_true, got %r" % (cat["quantity"], op))
        if row["hardness"] == "objective":
            return ("REQ-OP", "objective", "the boolean quantity %s cannot be the objective" % (cat["quantity"],))
        if any(x is not None for x in quad):
            return ("REQ-OP", "shape", "the boolean quantity %s takes no value, upper or tolerance" % (cat["quantity"],))
        return None
    if op == "is_true":
        return ("REQ-OP", "op", "op is_true only fits boolean quantities, not %s" % (cat["quantity"],))
    if row["hardness"] == "objective":
        if op in ("<=", ">=") and all(x is None for x in quad):
            return None
        return ("REQ-OP", "objective",
                "an objective row on %s needs op <= or >= and no value, upper or tolerance" % (cat["quantity"],))
    if op in ("==", "<=", ">="):
        if row["value"] is None or row["upper"] is not None:
            return ("REQ-OP", "shape",
                    "op %s on %s needs a value and no upper" % (op, cat["quantity"]))
    elif op == "in":
        if row["value"] is None or row["upper"] is None or not row["value"] < row["upper"]:
            return ("REQ-OP", "shape",
                    "op in on %s needs a value below its upper" % (cat["quantity"],))
    if row["tol_abs"] is not None and row["tol_rel"] is not None:
        return ("REQ-OP", "tol_both",
                "row on %s carries both tol_abs and tol_rel" % (cat["quantity"],))
    return None


def _row_pass(row, cat, brief, labels, refs):
    """Steps 1-8 of docs/16 §E.3 for one input row; the first failing step wins. None or a refusal record."""
    allowed = set(ROW_IN_KEYS) | set(ROW_IN_OPTIONAL)
    if not isinstance(row, dict):
        return _refuse(0, "REQ-OP", "unknown_key", "a row must be an object")
    unknown = sorted(str(k) for k in row.keys() if k not in allowed)
    if unknown:
        return _refuse(0, "REQ-OP", "unknown_key",
                       "the row has keys outside the row schema: %s" % (", ".join(unknown),))
    missing = [k for k in ROW_IN_KEYS if k not in row]
    if missing:
        return _refuse(0, "REQ-OP", "missing_key",
                       "the row misses the keys %s" % (", ".join(missing),))
    bad = _type_pass(row)
    if bad is not None:
        return _refuse(0, "REQ-OP", "type", "the row's %s has the wrong type" % (bad,))
    if cat is None:
        return _refuse(0, "REQ-QTY", "quantity",
                       "the quantity %r is not in the template catalogue" % (row["quantity"],))
    if row["feature"] is not None and row["feature"] not in cat["where"]:
        return _refuse(0, "REQ-QTY", "feature",
                       "the feature %r is not a where of %s (%s)" % (row["feature"], cat["quantity"],
                                                                     ", ".join(cat["where"])))
    if row["unit"] not in UNIT_TABLE:
        return _refuse(0, "REQ-UNIT", "unknown_unit", "the unit %r is not in the unit table" % (row["unit"],))
    if UNIT_TABLE[row["unit"]][0] != cat["unit"]:
        return _refuse(0, "REQ-UNIT", "dimension",
                       "the unit %r measures %s, but %s is measured in %s"
                       % (row["unit"], UNIT_TABLE[row["unit"]][0], cat["quantity"], cat["unit"]))
    shape = _shape_pass(row, cat)
    if shape is not None:
        return _refuse(0, shape[0], shape[1], shape[2])
    q = _quote_pass(row["source"], row["quote"], brief, labels)
    if q is not None:
        return _refuse(0, q[0], q[1], q[2])
    s = _std_pass(row, refs)
    if s is not None:
        return _refuse(0, s[0], s[1], s[2])
    if row["source"] in ("brief", "sketch_label"):
        si_row = _si_row(row)
        g = _ground_pass(row["unit"], row["quote"],
                         (("value", si_row["value"]), ("upper", si_row["upper"])))
        if g is not None:
            return _refuse(0, g[0], g[1], g[2])
    si_row = _si_row(row)
    lo, hi = _bounds(si_row)
    ref = _ref_of(lo, hi)
    tol = _tol(si_row, ref)
    u = _u_of(cat, ref)
    if row["op"] == "==" and u is not None and tol < u:
        return _refuse(0, "REQ-TOL", "below_u",
                       "the tolerance %g is below the primitive uncertainty u = %g of %s, so the row can"
                       " never be decided" % (tol, u, cat["quantity"]))
    if row["op"] == "in" and u is not None and (hi - lo) + 2 * tol < 2 * u:
        return _refuse(0, "REQ-TOL", "band_below_u",
                       "the band [%g, %g] plus twice the tolerance %g is narrower than twice the primitive"
                       " uncertainty u = %g of %s" % (lo, hi, tol, u, cat["quantity"]))
    if row["hardness"] == "hard" and row["source"] in TICK_SOURCES and row.get("ticked") is not True:
        return _refuse(0, "REQ-DEFAULT-HARD", "not_ticked",
                       "the %s row on %s is hard but carries no tick of approval on the card"
                       % (row["source"], cat["quantity"]))
    return None


def _si_row(row) -> dict:
    """The input row with value, upper, tol_abs and tol_rel in SI (docs/16 §E.1)."""
    unit = row["unit"]
    return {"op": row["op"], "hardness": row["hardness"],
            "value": None if row["value"] is None else to_si(row["value"], unit)[0],
            "upper": None if row["upper"] is None else to_si(row["upper"], unit)[0],
            "tol_abs": None if row["tol_abs"] is None else to_si(row["tol_abs"], unit)[0],
            "tol_rel": None if row["tol_rel"] is None else float(row["tol_rel"])}


def _norm_rec(index, row, cat) -> dict:
    """A user row normalised for the set passes: SI values, its interval, tol and u (docs/16 §E.3)."""
    rec = _si_row(row)
    rec.update({"index": index, "sys": False, "quantity": row["quantity"], "feature": row["feature"],
                "Re": row["condition"]["Re"], "level": row["condition"]["level"], "source": row["source"],
                "where": list(cat["where"]), "input": row})
    rec["lo"], rec["hi"] = _bounds(rec)
    rec["ref"] = _ref_of(rec["lo"], rec["hi"])
    rec["tol"] = _tol(rec, rec["ref"])
    rec["u"] = _u_of(cat, rec["ref"])
    return rec


def _sys_rec(t) -> dict:
    """One SYS row as a set-pass record that comes before every user row (docs/16 §E.1)."""
    rid, quantity, op, value, kind, method, primitive, where, u, ears = t
    rec = {"index": None, "sys": True, "id": rid, "quantity": quantity, "feature": None, "Re": None,
           "level": None, "op": op, "hardness": "hard", "value": value, "upper": None, "tol_abs": None,
           "tol_rel": None, "u": u, "kind": kind, "method": method, "primitive": primitive,
           "where": list(where), "ears": ears, "source": "system"}
    rec["lo"], rec["hi"] = _bounds(rec)
    rec["ref"] = _ref_of(rec["lo"], rec["hi"])
    rec["tol"] = _tol(rec, rec["ref"])
    return rec


def _interval(rec) -> tuple:
    """The row's acceptance interval with its tolerance folded in (docs/16 §E.3 REQ-CONFLICT)."""
    op, tol = rec["op"], rec["tol"]
    if op == "<=":
        return (-math.inf, rec["value"] + tol)
    if op == ">=":
        return (rec["value"] - tol, math.inf)
    if op == "==":
        return (rec["value"] - tol, rec["value"] + tol)
    return (rec["value"] - tol, rec["upper"] + tol)          # in


def _meets(iv, reach) -> bool:
    """Two closed intervals meet iff each starts before the other ends."""
    return iv[0] <= reach[1] and reach[0] <= iv[1]


def _dup_pass(entries) -> list:
    """REQ-DUP over the SYS rows then the surviving user rows, in order; only user rows are refused.
    The key takes the catalogue where, not the row's feature: the primitive measures a quantity at its where
    whatever the feature names, so a row with feature null and one naming that plane are one measurement."""
    refusals = []
    seen = set()
    for rec in entries:
        key = (rec["quantity"], tuple(rec["where"]), rec["Re"], rec["level"], rec["op"],
               rec["hardness"] == "objective")
        if rec["sys"]:
            seen.add(key)
        elif key in seen:
            refusals.append(_refuse(rec["index"], "REQ-DUP", "duplicate",
                                    "the row repeats %s at %s with the same op, condition and hardness"
                                    % (rec["quantity"], ", ".join(rec["where"]))))
        else:
            seen.add(key)
    return refusals


def _obj_pass(entries) -> list:
    """REQ-OBJ: every objective row after the first is refused."""
    refusals = []
    first = True
    for rec in entries:
        if rec["sys"] or rec["hardness"] != "objective":
            continue
        if first:
            first = False
            continue
        refusals.append(_refuse(rec["index"], "REQ-OBJ", "second_objective",
                                "the row makes a second objective; the study already minimises or maximises"
                                " %s" % (rec["quantity"],)))
    return refusals


def _conflict_pass(entries) -> list:
    """REQ-CONFLICT: per (quantity, catalogue where, Re, level) group, a row whose interval empties the running one."""
    refusals = []
    groups = {}
    order = []
    for rec in entries:
        if rec["hardness"] == "objective" or rec["op"] == "is_true":
            continue
        key = (rec["quantity"], tuple(rec["where"]), rec["Re"], rec["level"])
        if key not in groups:
            groups[key] = [-math.inf, math.inf]
            order.append(key)
        iv = _interval(rec)
        lo, hi = groups[key]
        if not rec["sys"] and (iv[0] > hi or iv[1] < lo):
            refusals.append(_refuse(rec["index"], "REQ-CONFLICT", "empty_intersection",
                                    "the row's interval [%g, %g] empties the running interval [%g, %g]"
                                    " of the earlier %s rows" % (iv[0], iv[1], lo, hi, rec["quantity"])))
        else:
            groups[key] = [max(lo, iv[0]), min(hi, iv[1])]
    refusals.sort(key=lambda r: r["row"])
    return refusals


def _outside_pass(recs, params) -> list:
    """REQ-OUTSIDE (docs/16 §E.3, decided analytically): a LOCKS row whose interval cannot meet its reach
    inside the template box. The reach is a superset, so only certainly unreachable rows are refused."""
    def box(name, default_lo, default_hi):
        p = params.get(name, {})
        lo = p.get("min")
        hi = p.get("max")
        return (default_lo if lo is None else float(lo), default_hi if hi is None else float(hi))

    d0 = box("D_i", 0.0, math.inf)
    c0 = box("CR", 1.0, math.inf)
    a1, a2 = box("L_over_Di", 0.0, math.inf)
    b1, b2 = box("Lx_over_De", 0.0, math.inf)
    t1, t2 = box("t_wall", 0.0, math.inf)
    hits = []

    def reach_of(quantity):
        if quantity == "inlet_diameter":
            return d
        if quantity == "contraction_ratio":
            return c
        if quantity == "exit_diameter":
            return (d[0] / math.sqrt(c[1]), d[1] / math.sqrt(c[0]))
        if quantity == "contraction_length":
            return (a1 * d[0], a2 * d[1])
        if quantity == "total_length":
            return (d[0] * (a1 + b1 / math.sqrt(c[1])), d[1] * (a2 + b2 / math.sqrt(c[0])))
        return (t1, t2)                                   # min_wall_normal

    d, c = d0, c0
    for rec in recs:
        if rec["sys"] or rec["quantity"] not in LOCKS or rec["hardness"] == "objective" \
                or rec["op"] == "is_true":
            continue
        if rec["hardness"] == "hard" and rec["op"] == "==" \
                and rec["quantity"] in ("inlet_diameter", "contraction_ratio"):
            base = d0 if rec["quantity"] == "inlet_diameter" else c0
            iv = _interval(rec)
            if not _meets(iv, base):
                hits.append(_refuse(rec["index"], "REQ-OUTSIDE", "unreachable",
                                    "the fixed %s = %g is outside the template box [%g, %g]"
                                    % (rec["quantity"], rec["value"], base[0], base[1])))
            elif rec["quantity"] == "inlet_diameter":
                d = (max(d[0], iv[0]), min(d[1], iv[1]))
            else:
                c = (max(c[0], iv[0]), min(c[1], iv[1]))
    for rec in recs:
        if rec["sys"] or rec["quantity"] not in LOCKS or rec["hardness"] == "objective" \
                or rec["op"] == "is_true":
            continue
        if rec["hardness"] == "hard" and rec["op"] == "==" \
                and rec["quantity"] in ("inlet_diameter", "contraction_ratio"):
            continue                                      # already tested against D0 / C0 above
        iv = _interval(rec)
        reach = reach_of(rec["quantity"])
        if not _meets(iv, reach):
            hits.append(_refuse(rec["index"], "REQ-OUTSIDE", "unreachable",
                                "the row's interval [%g, %g] cannot meet the reach [%g, %g] of %s inside"
                                " the template box" % (iv[0], iv[1], reach[0], reach[1], rec["quantity"])))
    hits.sort(key=lambda r: r["row"])
    return hits


def _questions(user_recs, flow_present) -> list:
    """The clarifying questions of docs/16 §F 2, computed on the surviving rows, in QUESTIONS order."""
    def hard_eq(quantity):
        return any(r["quantity"] == quantity and r["hardness"] == "hard" and r["op"] == "=="
                   for r in user_recs)
    di_known = hard_eq("inlet_diameter")
    questions = []
    if not di_known:
        questions.append({"id": "Q-D_i", "param": QUESTIONS["Q-D_i"][0], "text": QUESTIONS["Q-D_i"][1]})
    if not hard_eq("contraction_ratio") and not (hard_eq("exit_diameter") and di_known):
        questions.append({"id": "Q-CR", "param": QUESTIONS["Q-CR"][0], "text": QUESTIONS["Q-CR"][1]})
    if not flow_present:
        questions.append({"id": "Q-FLOW", "param": QUESTIONS["Q-FLOW"][0], "text": QUESTIONS["Q-FLOW"][1]})
    return questions


def _sys_out_row(t) -> dict:
    """The OUTPUT form of one SYS row: exactly ROW_KEYS (docs/16 §E.1)."""
    rid, quantity, op, value, kind, method, primitive, where, u, ears = t
    return {"id": rid, "ears": ears, "quantity": quantity, "feature": None, "op": op, "value": value,
            "upper": None, "tol_abs": None, "tol_rel": None, "unit": "1", "kind": kind, "method": method,
            "condition": {"Re": None, "level": None}, "hardness": "hard", "source": "system", "quote": None,
            "locks_params": [], "ticked": False, "confidence": CONFIDENCE["system"], "standard_ref": None}


def _user_out_row(index, row, cat) -> dict:
    """The OUTPUT form of one user row: SI values, catalogue unit/kind/method, server ears and id."""
    unit = row["unit"]
    locks = list(LOCKS[row["quantity"]]) \
        if row["hardness"] == "hard" and row["op"] == "==" and row["quantity"] in LOCKS else []
    out = {"id": "REQ-%03d" % (index + 1), "ears": "", "quantity": row["quantity"],
           "feature": row["feature"], "op": row["op"],
           "value": None if row["value"] is None else to_si(row["value"], unit)[0],
           "upper": None if row["upper"] is None else to_si(row["upper"], unit)[0],
           "tol_abs": None if row["tol_abs"] is None else to_si(row["tol_abs"], unit)[0],
           "tol_rel": None if row["tol_rel"] is None else float(row["tol_rel"]),
           "unit": cat["unit"], "kind": cat["kind"], "method": cat["method"],
           "condition": {"Re": None if row["condition"]["Re"] is None else float(row["condition"]["Re"]),
                         "level": row["condition"]["level"]},
           "hardness": row["hardness"], "source": row["source"], "quote": row["quote"],
           "locks_params": locks, "ticked": row.get("ticked") is True, "confidence": CONFIDENCE[row["source"]],
           "standard_ref": row.get("standard_ref")}
    out["ears"] = render_ears(out)
    return out


def _build_document(proposal, brief, template_sha, declaration_sha, user_recs, catalogue) -> dict:
    """The cad-requirements/1 document: approved_by PENDING, lock_sha None (docs/16 §E.1, §E.8)."""
    rows = []
    for r in user_recs:
        row = r["input"]
        rows.append(_user_out_row(r["index"], row, catalogue[row["quantity"]]))
    rows.extend(_sys_out_row(t) for t in SYS_ROWS)
    opp = proposal["operating_point"]
    flow_fields = {"U_exit_m_s": None, "Q_m3_s": None, "mdot_kg_s": None}
    flow = opp["flow"]
    if flow is not None:                       # only a refusal-free report builds a document,
        flow_fields[flow["field"]] = to_si(flow["value"], flow["unit"])[0]   # so flow is fully checked
    objective = None
    for r in user_recs:
        if r["hardness"] == "objective":
            objective = {"quantity": r["quantity"], "sense": "min" if r["op"] == "<=" else "max"}
            break
    return {"schema": "cad-requirements/1", "study_id": proposal["study_id"],
            "template_id": proposal["template_id"], "template_sha": template_sha,
            "declaration_sha": declaration_sha, "vocab_sha": proposal["vocab_sha"],
            "brief_sha": common.sha256_of(brief),
            "attachments": [a["sha256"] for a in brief["attachments"]],
            "operating_point": {"fluid": opp["fluid"], "T_K": float(opp["T_K"]), "p0_Pa": float(opp["p0_Pa"]),
                                "U_exit_m_s": flow_fields["U_exit_m_s"], "Q_m3_s": flow_fields["Q_m3_s"],
                                "mdot_kg_s": flow_fields["mdot_kg_s"],
                                "flow_source": None if flow is None else flow["source"],
                                "flow_quote": None if flow is None else flow["quote"],
                                "fluid_source": opp.get("fluid_source", "assumed"),
                                "T_K_source": opp.get("T_K_source", "assumed"),
                                "p0_Pa_source": opp.get("p0_Pa_source", "assumed")},
            "rows": rows, "objective": objective,
            "created_by": copy.deepcopy(proposal["created_by"]),
            "approved_by": PENDING, "lock_sha": None,
            "supersedes_study": None, "supersedes_lock": None, "change_kind": "new", "change_reason": None,
            "evidence_sha": None}


def check(proposal, brief, declaration, template_sha, declaration_sha) -> dict:
    """The S1 gate (docs/16 §D, §E.3): pure; refuses the LLM's rows by rule id, asks for missing drivers,
    or returns the report with its cad-requirements/1 document. Never mutates an argument."""
    proposal = copy.deepcopy(proposal)
    brief = copy.deepcopy(brief)
    _check_brief(brief)
    _check_envelope(proposal, declaration)
    if not isinstance(template_sha, str) or not SHA_RE.match(template_sha):
        raise ValueError("template_sha: %r is not a 64 lowercase hex sha256" % (template_sha,))
    if not isinstance(declaration_sha, str) or not SHA_RE.match(declaration_sha):
        raise ValueError("declaration_sha: %r is not a 64 lowercase hex sha256" % (declaration_sha,))
    if not isinstance(declaration.get("standards"), list):
        raise ValueError("declaration: the frozen standards table is missing (docs/16a §F)")
    refs = [s["ref"] for s in declaration["standards"]]
    catalogue = {c["quantity"]: c for c in declaration["catalogue"]}
    params = {p["name"]: p for p in declaration["params"]}
    labels = [lab for a in brief["attachments"] for lab in a["labels"]]
    refusals = []
    if proposal["vocab_sha"] != vocab_sha(declaration):
        refusals.append(_refuse("proposal", "REQ-VOCAB", "stale",
                                "the proposal's vocab_sha %r is not the vocabulary reqs.py vocab gives now"
                                % (proposal["vocab_sha"],)))
    rows = proposal["rows"]
    opp = proposal["operating_point"]
    flow = opp.get("flow") if isinstance(opp, dict) else None

    opp_ok = _check_operating_point(opp)
    if opp_ok is not None:
        refusals.append(opp_ok)
    elif flow is not None:
        fr = _check_flow(flow, brief, labels)
        if fr is not None:
            refusals.append(fr)
            flow = None

    recs = []
    dead = set()
    for i, row in enumerate(rows):
        q = row.get("quantity") if isinstance(row, dict) else None
        cat = catalogue.get(q) if isinstance(q, str) else None
        ref = _row_pass(row, cat, brief, labels, refs)
        if ref is not None:
            ref["row"] = i
            refusals.append(ref)
            dead.add(i)
        else:
            recs.append(_norm_rec(i, row, cat))

    entries = [_sys_rec(t) for t in SYS_ROWS] + recs
    for ref in _dup_pass(entries):
        dead.add(ref["row"])
        refusals.append(ref)
    live = [r for r in recs if r["index"] not in dead]
    for ref in _obj_pass([_sys_rec(t) for t in SYS_ROWS] + live):
        dead.add(ref["row"])
        refusals.append(ref)
    live = [r for r in recs if r["index"] not in dead]
    for ref in _conflict_pass([_sys_rec(t) for t in SYS_ROWS] + live):
        dead.add(ref["row"])
        refusals.append(ref)
    live = [r for r in recs if r["index"] not in dead]
    for ref in _outside_pass([_sys_rec(t) for t in SYS_ROWS] + live, params):
        dead.add(ref["row"])
        refusals.append(ref)
    live = [r for r in recs if r["index"] not in dead]

    questions = _questions(live, flow is not None)
    status = "refused" if refusals else ("questions" if questions else "ok")
    document = None
    derived = None
    if status != "refused":
        document = _build_document(proposal, brief, template_sha, declaration_sha, live, catalogue)
        errs = schema.errors(document, "cad-requirements/1")
        if errs:
            raise ValueError("reqs: the built document violates cad-requirements/1: %s" % (errs[0],))
    if status == "ok":
        derived = _derived(document)
    return {"version": VERSION, "status": status, "refusals": refusals, "questions": questions,
            "requirements": document, "derived": derived}


def _check_operating_point(opp):
    """The operating-point shape rule of docs/16 §E.3; None when it passes, else its refusal record."""
    def bad(why):
        return _refuse("operating_point", "REQ-OP", "shape", "the operating point is malformed: %s" % (why,))
    if not isinstance(opp, dict):
        return bad("not an object")
    keys = set(opp.keys())
    if not set(OPPOINT_KEYS) <= keys <= set(OPPOINT_KEYS) | set(OPPOINT_OPTIONAL):
        return bad("keys must be %s, optionally with %s" % (", ".join(OPPOINT_KEYS), ", ".join(OPPOINT_OPTIONAL)))
    for k in OPPOINT_OPTIONAL:
        if k in opp and opp[k] not in OPPOINT_SOURCES:
            return bad("%s %r is not one of %s" % (k, opp[k], ", ".join(OPPOINT_SOURCES)))
    if opp["fluid"] not in RHO_TABLE:
        return bad("fluid %r is neither air nor water" % (opp["fluid"],))
    if not _is_pos(opp["T_K"]):
        return bad("T_K %r is not a positive number" % (opp["T_K"],))
    if not _is_pos(opp["p0_Pa"]):
        return bad("p0_Pa %r is not a positive number" % (opp["p0_Pa"],))
    if opp["flow"] is not None and not isinstance(opp["flow"], dict):
        return bad("flow must be null or an object")
    return None


def _check_flow(flow, brief, labels):
    """The flow rules of docs/16 §E.3: shape, unit, dimension, quote and grounding; on the row "flow"."""
    def bad(rid, chk, why):
        return _refuse("flow", rid, chk, why)
    if not isinstance(flow, dict):
        return bad("REQ-OP", "shape", "the flow must be an object")
    if set(flow.keys()) != set(FLOW_KEYS):
        return bad("REQ-OP", "shape", "flow keys must be exactly %s" % (", ".join(FLOW_KEYS),))
    if flow["field"] not in FLOW_FIELDS:
        return bad("REQ-OP", "shape", "flow field %r is not one of %s" % (flow["field"], ", ".join(FLOW_FIELDS)))
    if not _is_pos(flow["value"]):
        return bad("REQ-OP", "shape", "flow value %r is not a positive number" % (flow["value"],))
    if flow["source"] not in FLOW_SOURCES:
        return bad("REQ-OP", "shape", "flow source %r is not one of %s" % (flow["source"], ", ".join(FLOW_SOURCES)))
    if flow["quote"] is not None and not isinstance(flow["quote"], str):
        return bad("REQ-OP", "shape", "flow quote must be null or a string")
    if flow["unit"] not in UNIT_TABLE:
        return bad("REQ-UNIT", "unknown_unit", "the flow unit %r is not in the unit table" % (flow["unit"],))
    if UNIT_TABLE[flow["unit"]][0] != FLOW_FIELDS[flow["field"]]:
        return bad("REQ-UNIT", "dimension",
                   "the flow unit %r measures %s, but the field %s is measured in %s"
                   % (flow["unit"], UNIT_TABLE[flow["unit"]][0], flow["field"], FLOW_FIELDS[flow["field"]]))
    q = _quote_pass(flow["source"], flow["quote"], brief, labels)
    if q is not None:
        return bad(q[0], q[1], q[2])
    if flow["source"] in ("brief", "sketch_label"):
        g = _ground_pass(flow["unit"], flow["quote"], (("value", to_si(flow["value"], flow["unit"])[0]),))
        if g is not None:
            return bad(g[0], g[1], g[2])
    return None


def _doc_D_e(rows) -> float:
    """D_e from a document's rows: the hard == exit_diameter, else D_i/sqrt(CR) from the hard == rows."""
    di = cr = None
    for r in rows:
        if r["id"].startswith("SYS-"):
            continue
        if r["quantity"] == "exit_diameter" and r["hardness"] == "hard" and r["op"] == "==":
            return r["value"]
        if r["quantity"] == "inlet_diameter" and r["hardness"] == "hard" and r["op"] == "==":
            di = r["value"]
        if r["quantity"] == "contraction_ratio" and r["hardness"] == "hard" and r["op"] == "==":
            cr = r["value"]
    if di is None or cr is None:
        raise ValueError("flow_Q: the document fixes neither the exit diameter nor the inlet diameter and"
                         " contraction ratio, so no D_e follows")
    return di / math.sqrt(cr)


def flow_Q(doc) -> float:
    """Q in m3/s from a cad-requirements/1 document, by the derived rules of docs/16 §E.1."""
    op = doc["operating_point"]
    if op.get("Q_m3_s") is not None:
        return op["Q_m3_s"]
    rho = RHO_TABLE[op["fluid"]]
    if op.get("mdot_kg_s") is not None:
        return op["mdot_kg_s"] / rho
    if op.get("U_exit_m_s") is not None:
        d = _doc_D_e(doc["rows"])
        return op["U_exit_m_s"] * (math.pi * d * d / 4.0)
    raise ValueError("flow_Q: the document states no flow")


def _derived(document) -> dict:
    """The report's derived block: rho, D_e and the flow as a volume flux (docs/16 §E.1)."""
    return {"rho_kg_m3": RHO_TABLE[document["operating_point"]["fluid"]],
            "D_e_m": _doc_D_e(document["rows"]),
            "Q_m3_s": flow_Q(document)}


def lock_sha_of(doc) -> str:
    """The lock sha: sha256 over the canonical JSON of the document with lock_sha null (docs/16 §E.8)."""
    return common.sha256_of(dict(doc, lock_sha=None))


def lock_ok(doc) -> bool:
    """True when the document carries a lock sha that matches its content."""
    return isinstance(doc, dict) and isinstance(doc.get("lock_sha"), str) \
        and doc["lock_sha"] == lock_sha_of(doc)


def _check_lineage(doc) -> None:
    """The lineage fields of docs/16a §F agree with change_kind, or REQ-LOCK: a new study names no predecessor; a
    superseding one names another study, its lock sha and a reason, and an evidence_correction the evidence sha."""
    kind = doc.get("change_kind")
    study, lsha = doc.get("supersedes_study"), doc.get("supersedes_lock")
    reason, ev = doc.get("change_reason"), doc.get("evidence_sha")
    if kind == "new":
        if (study, lsha, reason, ev) != (None, None, None, None):
            raise ValueError("REQ-LOCK: change_kind new carries no supersedes_study, supersedes_lock,"
                             " change_reason or evidence_sha")
        return
    if kind not in SUPERSEDE_KINDS:
        raise ValueError("REQ-LOCK: change_kind %r is not one of %s" % (kind, ", ".join(CHANGE_KINDS)))
    if not isinstance(study, str) or not STUDY_RE.match(study) or study == doc.get("study_id"):
        raise ValueError("REQ-LOCK: a %s names another study it supersedes, got %r" % (kind, study))
    if not isinstance(lsha, str) or not SHA_RE.match(lsha):
        raise ValueError("REQ-LOCK: a %s names the superseded lock sha, got %r" % (kind, lsha))
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("REQ-LOCK: a %s needs a change_reason" % (kind,))
    if kind == "evidence_correction" and not (isinstance(ev, str) and SHA_RE.match(ev)):
        raise ValueError("REQ-LOCK: an evidence_correction needs the evidence file's sha, got %r" % (ev,))
    if kind == "target_change" and ev is not None:
        raise ValueError("REQ-LOCK: a target_change carries no evidence_sha, got %r" % (ev,))


def lock(report, approved_by) -> dict:
    """Seal an accepted report into a locked cad-requirements/1 document (docs/16 §E.8, §F 1)."""
    if not isinstance(report, dict) or report.get("status") != "ok":
        raise ValueError("REQ-LOCK: the report status is %r, not an accepted requirement set"
                         % (report.get("status") if isinstance(report, dict) else None,))
    if not isinstance(approved_by, str) or not approved_by or approved_by == PENDING:
        raise ValueError("REQ-LOCK: approved_by must be a non-empty name other than %r, got %r"
                         % (PENDING, approved_by))
    doc = copy.deepcopy(report["requirements"])
    _check_lineage(doc)
    doc["approved_by"] = approved_by
    doc["lock_sha"] = lock_sha_of(doc)
    errs = schema.errors(doc, "cad-requirements/1")
    if errs:
        raise ValueError("REQ-LOCK: the locked document violates cad-requirements/1: %s" % (errs[0],))
    return doc


def _repr_of(primitive, method) -> str:
    """The representation a check is judged on (docs/16a §F cad-checks/1): stl for the watertight report, else by
    the row's method."""
    if primitive in REPR_BY_PRIMITIVE:
        return REPR_BY_PRIMITIVE[primitive]
    if method not in REPR_BY_METHOD:
        raise ValueError("reqs: method %r has no representation in %s" % (method, ", ".join(REPRS)))
    return REPR_BY_METHOD[method]


def compile_checks(doc, declaration, declaration_sha) -> dict:
    """The pure compile of docs/16 §E.4: a locked document into cad-checks/1, one check per row in row order."""
    if not lock_ok(doc):
        raise ValueError("GATE-LOCK: the requirements lock does not match its document")
    if doc.get("declaration_sha") != declaration_sha:
        raise ValueError("GATE-LOCK: the document binds the declaration %r, not this template's %r"
                         % (doc.get("declaration_sha"), declaration_sha))
    if doc["template_id"] != declaration["template_id"]:
        raise ValueError("GATE-LOCK: the document is for template %r but the declaration is %r"
                         % (doc["template_id"], declaration["template_id"]))
    catalogue = {c["quantity"]: c for c in declaration["catalogue"]}
    sys_by_id = {t[0]: t for t in SYS_ROWS}
    checks = []
    for row in doc["rows"]:
        rid = row["id"]
        if rid in sys_by_id:
            t = sys_by_id[rid]
            primitive, where, u = t[6], list(t[7]), t[8]
            feature, re_, level = None, None, None
            cat = None
        else:
            cat = catalogue[row["quantity"]]
            primitive, where = cat["primitive"], list(cat["where"])
            feature, re_, level = row["feature"], row["condition"]["Re"], row["condition"]["level"]
            u = None
        lo, hi = _bounds(row)
        ref = _ref_of(lo, hi)
        if cat is not None:
            u = _u_of(cat, ref)
        checks.append({"req_id": rid, "primitive": primitive,
                       "args": {"feature": feature, "where": where, "Re": re_, "level": level},
                       "op": row["op"], "lo": lo, "hi": hi, "tol": _tol(row, ref), "u": u,
                       "hardness": row["hardness"], "repr": _repr_of(primitive, row["method"])})
    out = {"schema": "cad-checks/1", "study_id": doc["study_id"], "requirements_lock": doc["lock_sha"],
           "declaration_sha": doc["declaration_sha"], "checks": checks}
    errs = schema.errors(out, "cad-checks/1")
    if errs:
        raise ValueError("reqs: the compiled checks violate cad-checks/1: %s" % (errs[0],))
    return out


def write_canonical(path, obj) -> None:
    """The only writer reqs.py uses: canonical JSON plus one newline, atomically (docs/16 §D)."""
    common.atomic_write(path, common.canonical_json(obj) + chr(10))


def write_locked(out_dir, doc) -> tuple:
    """requirements.json plus requirements.lock, written once (docs/16 §E.8, docs/16a §E REQ-IMMUTABLE): an
    existing file with identical bytes is accepted and left untouched; any other existing path is refused."""
    if not lock_ok(doc):
        raise ValueError("GATE-LOCK: the requirements lock does not match its document")
    p_doc = os.path.join(out_dir, "requirements.json")
    p_lock = os.path.join(out_dir, "requirements.lock")
    pairs = ((p_doc, (common.canonical_json(doc) + chr(10)).encode("utf-8")),
             (p_lock, (doc["lock_sha"] + chr(10)).encode("utf-8")))
    for path, blob in pairs:
        if not os.path.lexists(path):
            continue
        snap = common.stable_file_snapshot(path)
        if snap["stable"] is not True or snap["sha256"] != common.sha256_bytes(blob):
            raise ValueError("REQ-IMMUTABLE: %s already exists with other bytes; a changed requirement set is"
                             " a new study" % (os.path.basename(path),))
    for path, blob in pairs:
        if not os.path.lexists(path):
            common.atomic_write(path, blob)
    return (p_doc, p_lock)


def read_locked(out_dir) -> dict:
    """Read a locked requirement set back; refuses a lock mismatch with GATE-LOCK (docs/16 §E.8)."""
    doc = common.read_json(os.path.join(out_dir, "requirements.json"))
    with open(os.path.join(out_dir, "requirements.lock"), "r", encoding="utf-8") as f:
        stamped = f.read().strip()
    if stamped != doc.get("lock_sha") or not lock_ok(doc):
        raise ValueError("GATE-LOCK: the requirements lock does not match its document")
    return doc


def load_template(template_dir) -> tuple:
    """(declaration, template_sha, declaration_sha): the template's DECLARATION, the sha of its module and the sha
    of template.json's bytes (docs/16 §D; docs/16a §D.13: the catalogue, u_meas and LOCKS live in template.json)."""
    p_decl = os.path.join(template_dir, "template.json")
    return (common.read_json(p_decl), common.sha256_file(os.path.join(template_dir, "template.py")),
            common.sha256_file(p_decl))


def vocab(declaration) -> dict:
    """The cad_requirements_propose vocabulary (docs/16a §B.1): generated from template.json and this module's
    constants only, so any change to either moves vocab_sha."""
    quantities = []
    features = []
    for c in declaration["catalogue"]:
        ops = ["is_true"] if c["primitive"] in BOOLEAN_PRIMITIVES else ["<=", ">=", "==", "in"]
        quantities.append({"quantity": c["quantity"], "unit": c["unit"], "kind": c["kind"], "method": c["method"],
                           "where": list(c["where"]), "ops": ops,
                           "units": [u for u in UNIT_TABLE if UNIT_TABLE[u][0] == c["unit"]]})
        for w in c["where"]:
            if w not in features:
                features.append(w)
    return {"version": VOCAB_VERSION, "template_id": declaration["template_id"], "quantities": quantities,
            "features": features, "planes": [p["name"] for p in declaration["planes"]],
            "tags": [t["name"] for t in declaration["tags"]],
            "standards": [s["ref"] for s in declaration["standards"]],
            "units": list(UNIT_TABLE), "ops": list(OPS), "hardness": list(HARDNESS), "sources": list(SOURCES),
            "flow_fields": list(FLOW_FIELDS), "flow_sources": list(FLOW_SOURCES),
            "oppoint_sources": list(OPPOINT_SOURCES), "fluids": sorted(RHO_TABLE),
            "proposal_keys": list(PROPOSAL_KEYS), "oppoint_keys": list(OPPOINT_KEYS) + list(OPPOINT_OPTIONAL),
            "flow_keys": list(FLOW_KEYS), "row_keys": list(ROW_IN_KEYS), "row_optional": list(ROW_IN_OPTIONAL)}


def vocab_sha(declaration) -> str:
    """sha256 of the canonical vocabulary: the proposal's vocab_sha must equal it (REQ-VOCAB)."""
    return common.sha256_of(vocab(declaration))


def eval_key(parts) -> str:
    """The evaluation key of docs/16 §D with declaration_sha beside template_sha (docs/16a §F): sha256 of the
    canonical dict of exactly EVAL_KEY_PARTS."""
    if not isinstance(parts, dict) or set(parts.keys()) != set(EVAL_KEY_PARTS):
        raise ValueError("EVAL-KEY: the parts must be exactly %s" % (", ".join(EVAL_KEY_PARTS),))
    return common.sha256_of({k: parts[k] for k in EVAL_KEY_PARTS})


def _where_of(row, declaration) -> list:
    """Where a locked row is measured: the SYS table's where for a SYS row, else its catalogue row's (ordered)."""
    for t in SYS_ROWS:
        if t[0] == row["id"]:
            return list(t[7])
    for c in declaration["catalogue"]:
        if c["quantity"] == row["quantity"]:
            return list(c["where"])
    raise ValueError("GATE-LOCK: row %s measures %r, which the declaration's catalogue does not hold"
                     % (row["id"], row["quantity"]))


def _diff_view(doc, declaration) -> list:
    """Per row: its id and op, its match key and its compared fields (every row key but id, plus the ordered where).
    The key is REQ-DUP's without the op: quantity, the where as a SET, condition Re and level, objective or not."""
    out = []
    for row in doc["rows"]:
        where = _where_of(row, declaration)
        fields = {k: row[k] for k in DIFF_ROW_FIELDS if k != "where"}
        fields["where"] = where
        key = (row["quantity"], tuple(sorted(where)), row["condition"]["Re"], row["condition"]["level"],
               row["hardness"] == "objective")
        out.append({"id": row["id"], "op": row["op"], "key": key, "fields": fields})
    return out


def _key_dict(key) -> dict:
    """A match key as the diff reports it."""
    return {"quantity": key[0], "where": list(key[1]), "Re": key[2], "level": key[3], "objective": key[4]}


def _diff_docs(old, new, old_template, new_template) -> dict:
    """The row diff of two requirement documents (docs/16a §B.1, §D.7). A template is load_template's tuple; each
    document must bind its template's declaration_sha (GATE-LOCK). Rows pair first on key and op, then the one row
    left on each side of a key pairs (an op change); the rest are added or removed. A pair is target_change when
    any compared field differs, else none."""
    for side, doc, tpl in (("old", old, old_template), ("new", new, new_template)):
        if doc.get("declaration_sha") != tpl[2]:
            raise ValueError("GATE-LOCK: the %s set binds the declaration %r, not the given template's %r"
                             % (side, doc.get("declaration_sha"), tpl[2]))
    olds = _diff_view(old, old_template[0])
    news = _diff_view(new, new_template[0])
    pair = {}                                   # new index -> old index
    used = set()
    for j, n in enumerate(news):                # pass 1: the same key and the same op
        for i, o in enumerate(olds):
            if i not in used and o["key"] == n["key"] and o["op"] == n["op"]:
                pair[j] = i
                used.add(i)
                break
    for j, n in enumerate(news):                # pass 2: the one row left on each side of a key
        if j in pair:
            continue
        o_left = [i for i, o in enumerate(olds) if i not in used and o["key"] == n["key"]]
        n_left = [k for k, m in enumerate(news) if k not in pair and m["key"] == n["key"]]
        if len(o_left) == 1 and len(n_left) == 1:
            pair[j] = o_left[0]
            used.add(o_left[0])
    rows = []
    for j, n in enumerate(news):                # new order: pairs and added rows, then removed rows in old order
        if j in pair:
            o = olds[pair[j]]
            changed = [k for k in DIFF_ROW_FIELDS if o["fields"][k] != n["fields"][k]]
            rows.append({"class": "target_change" if changed else "none", "old_id": o["id"], "new_id": n["id"],
                         "key": _key_dict(n["key"]), "fields": changed})
        else:
            rows.append({"class": "added", "old_id": None, "new_id": n["id"], "key": _key_dict(n["key"]),
                         "fields": []})
    for i, o in enumerate(olds):
        if i not in used:
            rows.append({"class": "removed", "old_id": o["id"], "new_id": None, "key": _key_dict(o["key"]),
                         "fields": []})
    op_old, op_new = old["operating_point"], new["operating_point"]
    op_fields = sorted(k for k in set(op_old) | set(op_new) if op_old.get(k) != op_new.get(k))
    counts = {c: sum(1 for r in rows if r["class"] == c) for c in DIFF_CLASSES}
    change = "none" if counts["none"] == len(rows) and not op_fields else "target_change"
    return {"version": DIFF_VERSION, "old": {k: old.get(k) for k in DIFF_HEAD},
            "new": {k: new.get(k) for k in DIFF_HEAD}, "change": change, "counts": counts, "rows": rows,
            "operating_point": {"class": "target_change" if op_fields else "none", "fields": op_fields},
            "context_changed": [k for k in DIFF_CONTEXT if old.get(k) != new.get(k)]}


def diff(old_doc, new_doc, old_template, new_template) -> dict:
    """`reqs.py diff` (docs/16a §B.1): the row diff of two LOCKED requirement sets; refuses an unlocked one
    GATE-LOCK. Pure: never mutates an argument."""
    for side, doc in (("old", old_doc), ("new", new_doc)):
        if not lock_ok(doc):
            raise ValueError("GATE-LOCK: the %s requirement set's lock does not match its document" % (side,))
    return _diff_docs(copy.deepcopy(old_doc), copy.deepcopy(new_doc), old_template, new_template)


def generated_shas(study_dir) -> set:
    """Every sha a study generated (docs/16a §E REQ-EVIDENCE-GENERATED): each file under cache/, the
    iterations.jsonl file itself and every 64-hex token written in it. A missing cache/ or log adds nothing."""
    shas = set()
    cache = os.path.join(study_dir, "cache")
    if os.path.isdir(cache):
        for root, dirs, files in os.walk(cache):
            dirs.sort()
            for name in sorted(files):
                path = os.path.join(root, name)
                if os.path.isfile(path):
                    shas.add(common.sha256_file(path))
    log = os.path.join(study_dir, "iterations.jsonl")
    if os.path.isfile(log):
        shas.add(common.sha256_file(log))
        with open(log, "r", encoding="utf-8", errors="replace") as f:
            shas.update(SHA_TOKEN_RE.findall(f.read()))
    return shas


def supersede(report, approved_by, old_dir, old_template, new_template, change_kind, change_reason,
              evidence_path=None) -> dict:
    """Lock an accepted report as a study that supersedes the locked set in old_dir (docs/16a §B.1, §E, §F): it
    records supersedes_study, supersedes_lock, change_kind, change_reason and evidence_sha. A target_change must
    change a target (the diff is not none) and cites no file; an evidence_correction cites one stable file whose
    content sha the old study never generated (REQ-EVIDENCE-GENERATED). Any other inconsistency is REQ-LOCK."""
    old = read_locked(old_dir)
    if change_kind not in SUPERSEDE_KINDS:
        raise ValueError("REQ-LOCK: a superseding study is one of %s, got %r"
                         % (", ".join(SUPERSEDE_KINDS), change_kind))
    if not isinstance(change_reason, str) or not change_reason.strip():
        raise ValueError("REQ-LOCK: a superseding study needs a change_reason")
    if not isinstance(report, dict) or report.get("status") != "ok":
        raise ValueError("REQ-LOCK: the report status is %r, not an accepted requirement set"
                         % (report.get("status") if isinstance(report, dict) else None,))
    doc = copy.deepcopy(report["requirements"])
    if doc["study_id"] == old["study_id"]:
        raise ValueError("REQ-LOCK: the superseding study needs a new study_id, not %r" % (old["study_id"],))
    if doc["template_id"] != old["template_id"]:
        raise ValueError("REQ-LOCK: the study %s is for template %r, the superseded one for %r"
                         % (doc["study_id"], doc["template_id"], old["template_id"]))
    evidence_sha = None
    if change_kind == "evidence_correction":
        if evidence_path is None:
            raise ValueError("REQ-LOCK: an evidence_correction cites an evidence file")
        name = os.path.basename(str(evidence_path))
        snap = common.stable_file_snapshot(evidence_path)
        if snap["stable"] is not True:
            raise ValueError("REQ-LOCK: the evidence %s is not a stable regular file" % (name,))
        if snap["sha256"] in generated_shas(old_dir):
            raise ValueError("REQ-EVIDENCE-GENERATED: the evidence %s has the sha %s.. of a file the superseded"
                             " study %s generated (cache/ or iterations.jsonl); generated output cannot correct a"
                             " requirement" % (name, snap["sha256"][:12], old["study_id"]))
        evidence_sha = snap["sha256"]
    elif evidence_path is not None:
        raise ValueError("REQ-LOCK: a target_change cites no evidence file")
    doc.update({"supersedes_study": old["study_id"], "supersedes_lock": old["lock_sha"],
                "change_kind": change_kind, "change_reason": change_reason, "evidence_sha": evidence_sha})
    delta = _diff_docs(old, doc, old_template, new_template)
    if change_kind == "target_change" and delta["change"] == "none":
        raise ValueError("REQ-LOCK: the new set changes no target of %s; a target_change must change one"
                         % (old["study_id"],))
    return lock(dict(report, requirements=doc), approved_by)


def _walk(node, piece):
    """One path piece of the fixture grammar: `name` or `name[i]`."""
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\[([0-9]+)\])?$", piece)
    if m is None:
        raise ValueError("fixture patch: bad path piece %r" % (piece,))
    child = node[m.group(1)]
    return child if m.group(2) is None else child[int(m.group(2))]


def _fixture_proposal(cases, s) -> dict:
    """A fixture set's proposal: its own, or its base's with every patch step applied (docs/16 §I CAD-07)."""
    if s.get("proposal") is not None:
        node = copy.deepcopy(s["proposal"])
        node.setdefault("vocab_sha", cases["vocab_sha"])      # the vocabulary the fixture was recorded with
        return node
    base = None
    for other in cases["sets"]:
        if other["name"] == s["base"]:
            base = other
            break
    if base is None:
        raise ValueError("fixture patch: base %r not found" % (s["base"],))
    node = copy.deepcopy(base["proposal"])
    node.setdefault("vocab_sha", cases["vocab_sha"])
    for step in s["patch"]:
        pieces = step["path"].split(".")
        parent = node
        for piece in pieces[:-1]:
            parent = _walk(parent, piece)
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\[([0-9]+)\])?$", pieces[-1])
        if m is None:
            raise ValueError("fixture patch: bad path piece %r" % (pieces[-1],))
        if step["op"] == "set":
            if m.group(2) is None:
                parent[m.group(1)] = copy.deepcopy(step["value"])
            else:
                _walk(parent, m.group(1))[int(m.group(2))] = copy.deepcopy(step["value"])
        elif step["op"] == "append":
            _walk(parent, pieces[-1]).append(copy.deepcopy(step["value"]))
        else:
            raise ValueError("fixture patch: unknown op %r" % (step["op"],))
    return node


def _patch_doc(node, steps) -> None:
    """The diff fixtures' patch grammar on a document: set, append, delete (one list element), permute (a list)."""
    for step in steps:
        pieces = step["path"].split(".")
        parent = node
        for piece in pieces[:-1]:
            parent = _walk(parent, piece)
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\[([0-9]+)\])?$", pieces[-1])
        if m is None:
            raise ValueError("diff fixture patch: bad path piece %r" % (pieces[-1],))
        name, index = m.group(1), m.group(2)
        if step["op"] == "set" and index is None:
            parent[name] = copy.deepcopy(step["value"])
        elif step["op"] == "set":
            parent[name][int(index)] = copy.deepcopy(step["value"])
        elif step["op"] == "append" and index is None:
            parent[name].append(copy.deepcopy(step["value"]))
        elif step["op"] == "delete" and index is not None:
            del parent[name][int(index)]
        elif step["op"] == "permute" and index is None:
            items = parent[name]
            if sorted(step["value"]) != list(range(len(items))):
                raise ValueError("diff fixture patch: %r is not a permutation of %d items"
                                 % (step["value"], len(items)))
            parent[name] = [items[i] for i in step["value"]]
        else:
            raise ValueError("diff fixture patch: bad step %r" % (step,))


def _diff_side(side, tmp) -> tuple:
    """One side of a diff fixture pair: a golden requirements document with the patch applied, the user rows'
    locks_params, confidence and ears re-derived, the catalogue where of the named quantities replaced in a
    temporary template.json, and the lock re-sealed. Returns (document, template tuple)."""
    doc = common.read_json(os.path.join(GOLDEN, side["golden"] + ".json"))["requirements"]
    tpl = load_template(NOZZLE_DIR)
    if not side["patch"] and side["where"] is None:
        return doc, tpl
    _patch_doc(doc, side["patch"])
    for row in doc["rows"]:
        if row["id"].startswith("SYS-"):
            continue
        row["locks_params"] = list(LOCKS[row["quantity"]]) \
            if row["hardness"] == "hard" and row["op"] == "==" and row["quantity"] in LOCKS else []
        row["confidence"] = CONFIDENCE[row["source"]]
        row["ears"] = render_ears(row)
    if side["where"] is not None:
        decl2 = copy.deepcopy(tpl[0])
        for c in decl2["catalogue"]:
            if c["quantity"] in side["where"]:
                c["where"] = list(side["where"][c["quantity"]])
        tdir = os.path.join(tmp, "template")
        os.makedirs(tdir)
        with open(os.path.join(NOZZLE_DIR, "template.py"), "rb") as f:
            blob = f.read()
        with open(os.path.join(tdir, "template.py"), "wb") as f:
            f.write(blob)
        with open(os.path.join(tdir, "template.json"), "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(decl2, indent=2, ensure_ascii=False) + chr(10))
        tpl = load_template(tdir)
        doc["declaration_sha"] = tpl[2]
        doc["vocab_sha"] = vocab_sha(tpl[0])
    doc["lock_sha"] = lock_sha_of(doc)
    errs = schema.errors(doc, "cad-requirements/1")
    if errs:
        raise ValueError("diff fixture %s: %s" % (side["golden"], errs[0]))
    return doc, tpl


def _refused(fn, prefix, needle=None) -> str:
    """Run fn; it must raise ValueError whose text starts with prefix (and holds needle). Returns the text."""
    try:
        fn()
    except ValueError as e:
        text = str(e)
        assert text.startswith(prefix), text
        assert needle is None or needle in text, (needle, text)
        return text
    raise AssertionError("expected a %s refusal" % (prefix,))


DIFF_PAIR_NAMES = ("d01_value_moved", "d02_bound_widened", "d03_tolerance_tightened", "d04_op_changed",
                   "d05_hardness_changed", "d06_source_changed", "d07_ticked_changed", "d08_where_swapped",
                   "d09_rows_reordered", "d10_ids_renumbered", "d11_row_added", "d12_row_removed")


def _selftest_diff_pairs(td) -> dict:
    """(D1-D12) the 12 fixture pairs of fixtures/reqs/diff/pairs.json, each classified exactly. Returns the built
    (old, new, old_template, new_template) per pair name."""
    pairs = common.read_json(DIFF_PAIRS)["pairs"]
    assert tuple(p["name"] for p in pairs) == DIFF_PAIR_NAMES, [p["name"] for p in pairs]
    built = {}
    for p in pairs:
        old, old_t = _diff_side(p["old"], os.path.join(td, p["name"], "old"))
        new, new_t = _diff_side(p["new"], os.path.join(td, p["name"], "new"))
        assert lock_ok(old) and lock_ok(new), p["name"]
        before = common.canonical_json([old, new])
        d = diff(old, new, old_t, new_t)
        assert common.canonical_json([old, new]) == before, "diff mutated %s" % (p["name"],)
        assert common.canonical_json(d) == common.canonical_json(diff(old, new, old_t, new_t)), p["name"]
        e = p["expect"]
        changed = [{k: r[k] for k in ("class", "old_id", "new_id", "fields")} for r in d["rows"]
                   if r["class"] != "none"]
        assert d["change"] == e["change"], (p["name"], d["change"])
        assert d["counts"] == e["counts"], (p["name"], d["counts"])
        assert changed == e["changed"], (p["name"], changed)
        assert d["context_changed"] == e["context_changed"], (p["name"], d["context_changed"])
        assert d["operating_point"] == e["operating_point"], (p["name"], d["operating_point"])
        assert sum(d["counts"].values()) == len(d["rows"]) and tuple(d) == DIFF_KEYS, p["name"]
        text = "; ".join("%s %s->%s [%s]" % (r["class"], r["old_id"], r["new_id"], ", ".join(r["fields"]))
                         for r in changed) or "all %d rows none" % (len(d["rows"]),)
        if "pairs" in e:
            got = [[r["old_id"], r["new_id"]] for r in d["rows"]]
            assert [x for x in got if x in e["pairs"]] == e["pairs"], (p["name"], got)
            text += "; paired " + ", ".join("%s->%s" % (a, b) for a, b in e["pairs"])
        if e["context_changed"]:
            text += "; context " + ", ".join(e["context_changed"])
        built[p["name"]] = (old, new, old_t, new_t)
        print("[ok] %s: %s; %s" % (p["name"], d["change"], text))
    return built


def _selftest_diff_refusals(built) -> None:
    """(D13) an identical pair is none, a reversed pair mirrors, an unsealed set and a foreign declaration are
    GATE-LOCK."""
    v1, _new, tpl, _t = built["d01_value_moved"]
    same = diff(v1, v1, tpl, tpl)
    assert same["change"] == "none" and same["counts"] == {"target_change": 0, "added": 0, "removed": 0,
                                                           "none": 12}
    back = diff(built["d11_row_added"][1], built["d11_row_added"][0], tpl, tpl)
    assert back["counts"]["removed"] == 1 and [r["old_id"] for r in back["rows"] if r["class"] == "removed"] \
        == ["REQ-007"]
    rev = diff(built["d01_value_moved"][1], v1, tpl, tpl)
    assert [(r["old_id"], r["fields"]) for r in rev["rows"] if r["class"] != "none"] == [("REQ-003",
                                                                                          ["ears", "value"])]
    unsealed = copy.deepcopy(v1)
    unsealed["rows"][2]["value"] = 0.09
    _refused(lambda: diff(v1, unsealed, tpl, tpl), "GATE-LOCK", "new requirement set")
    where_t = built["d08_where_swapped"][3]
    _refused(lambda: diff(v1, v1, tpl, where_t), "GATE-LOCK", "the new set binds the declaration")
    wrong = copy.deepcopy(v1)
    wrong["rows"][0]["quantity"] = "throat_length"
    wrong["lock_sha"] = lock_sha_of(wrong)
    _refused(lambda: diff(v1, wrong, tpl, tpl), "GATE-LOCK", "'throat_length'")
    print("[ok] diff refusals: an identical pair is none on 12 rows, d11 and d01 reversed read removed REQ-007 and"
          " REQ-003 [ears, value], an unsealed set, a foreign declaration and an uncatalogued quantity are"
          " GATE-LOCK")


def _fixture_study(td, v1) -> tuple:
    """A superseded study on disk: v1 locked, two cache files (one nested), and an iterations.jsonl whose eval row
    names a report sha that is in no file any more. Returns (old_dir, eval key, the bytes by name)."""
    old_dir = os.path.join(td, "v1_nominal")
    write_locked(old_dir, v1)
    key = common.sha256_bytes(b"fixture eval key")
    blobs = {"geom": common.canonical_bytes({"schema": "fixture-geom/1", "volume_m3": 1.4e-05}) + b"\n",
             "boundary": b"FoamFile boundary fixture\n",
             "report": b"verdict fixture: a report the loop generated and later deleted\n"}
    common.atomic_write(os.path.join(old_dir, "cache", key, "geom.json"), blobs["geom"])
    common.atomic_write(os.path.join(old_dir, "cache", key, "mesh", "boundary"), blobs["boundary"])
    log = os.path.join(old_dir, "iterations.jsonl")
    common.jsonl_append(log, {"kind": "genesis", "lock_sha": v1["lock_sha"]})
    common.jsonl_append(log, {"kind": "eval", "eval_key": key, "report_sha": common.sha256_bytes(blobs["report"])})
    return old_dir, key, blobs


def _report_of(doc, study_id) -> dict:
    """An accepted report carrying a copy of doc under another study id."""
    return {"status": "ok", "requirements": dict(copy.deepcopy(doc), study_id=study_id)}


def _copy_bytes(src, dst) -> None:
    """Copy one file's bytes to a new name."""
    with open(src, "rb") as f:
        blob = f.read()
    with open(dst, "wb") as f:
        f.write(blob)


def _selftest_lineage(td, built) -> dict:
    """(E1-E6) the lineage of a superseding study and REQ-EVIDENCE-GENERATED by content sha. Returns what the CLI
    test reuses."""
    v1, moved, tpl, _t = built["d01_value_moved"]
    old_dir, key, blobs = _fixture_study(td, v1)
    ext = os.path.join(td, "external")
    os.makedirs(ext)
    gen = generated_shas(old_dir)
    assert len(gen) == 6 and v1["lock_sha"] in gen and key in gen, sorted(gen)
    rep = _report_of(v1, "v1_nominal_r2")
    reason = "the supplier corrected the wall thickness in writing"
    ev = "evidence_correction"
    for src, name in ((os.path.join(old_dir, "cache", key, "geom.json"), "vendor_datasheet.json"),
                      (os.path.join(old_dir, "cache", key, "mesh", "boundary"), "boundary_from_supplier.txt")):
        dst = os.path.join(ext, name)
        _copy_bytes(src, dst)
        _refused(lambda: supersede(rep, "reviewer", old_dir, tpl, tpl, ev, reason, dst),
                 "REQ-EVIDENCE-GENERATED", "v1_nominal generated")
    print("[ok] e01 generated evidence: cache/ geom.json copied out as vendor_datasheet.json and the nested"
          " mesh/boundary renamed are refused REQ-EVIDENCE-GENERATED by content sha; 6 generated shas")
    report_file = os.path.join(ext, "loop_report.txt")
    with open(report_file, "wb") as f:
        f.write(blobs["report"])
    _refused(lambda: supersede(rep, "reviewer", old_dir, tpl, tpl, ev, reason, report_file),
             "REQ-EVIDENCE-GENERATED", "iterations.jsonl")
    log_copy = os.path.join(ext, "history.txt")
    _copy_bytes(os.path.join(old_dir, "iterations.jsonl"), log_copy)
    _refused(lambda: supersede(rep, "reviewer", old_dir, tpl, tpl, ev, reason, log_copy),
             "REQ-EVIDENCE-GENERATED")
    print("[ok] e02 generated evidence: a deleted report whose sha iterations.jsonl records and a copy of"
          " iterations.jsonl itself are refused REQ-EVIDENCE-GENERATED")
    letter = os.path.join(ext, "supplier_letter.txt")
    with open(letter, "wb") as f:
        f.write(b"Supplier letter 2026-09-26: the wall is 2.5 mm, not 2 mm.\n")
    old_bytes = common.sha256_file(os.path.join(old_dir, "requirements.json"))
    doc = supersede(rep, "reviewer", old_dir, tpl, tpl, ev, reason, letter)
    assert (doc["supersedes_study"], doc["supersedes_lock"], doc["change_kind"], doc["change_reason"],
            doc["evidence_sha"]) == ("v1_nominal", v1["lock_sha"], ev, reason, common.sha256_file(letter))
    assert lock_ok(doc) and doc["approved_by"] == "reviewer" and not schema.errors(doc, "cad-requirements/1")
    new_dir = os.path.join(td, "v1_nominal_r2")
    write_locked(new_dir, doc)
    assert read_locked(new_dir) == doc and common.sha256_file(os.path.join(old_dir, "requirements.json")) \
        == old_bytes
    _refused(lambda: write_locked(old_dir, doc), "REQ-IMMUTABLE")
    print("[ok] e03 external evidence: supplier_letter.txt passes as an evidence_correction of v1_nominal with its"
          " sha, locked, written once and read back; writing it over the old study is REQ-IMMUTABLE")
    tc = supersede(_report_of(moved, "v1_nominal_r3"), "reviewer", old_dir, tpl, tpl, "target_change",
                   "the user asked for 90 mm")
    assert (tc["change_kind"], tc["evidence_sha"], tc["supersedes_lock"]) == ("target_change", None,
                                                                             v1["lock_sha"])
    _refused(lambda: supersede(rep, "reviewer", old_dir, tpl, tpl, "target_change", reason), "REQ-LOCK",
             "changes no target")
    _refused(lambda: supersede(_report_of(moved, "v1_nominal_r3"), "reviewer", old_dir, tpl, tpl,
                               "target_change", reason, letter), "REQ-LOCK", "cites no evidence")
    swapped, where_t = built["d08_where_swapped"][1], built["d08_where_swapped"][3]
    ws = supersede(_report_of(swapped, "v1_nominal_r4"), "reviewer", old_dir, tpl, where_t, "target_change",
                   "the ratio is taken exit over inlet")
    assert ws["declaration_sha"] == where_t[2]
    _refused(lambda: supersede(_report_of(swapped, "v1_nominal_r4"), "reviewer", old_dir, tpl, tpl,
                               "target_change", reason), "GATE-LOCK")
    print("[ok] e04 target_change: a moved value supersedes with no evidence, an unchanged set and a cited file"
          " are REQ-LOCK, a swapped where supersedes under its own declaration and is GATE-LOCK under the old one")
    same_id = _report_of(v1, "v1_nominal")
    for fn, needle in (
            (lambda: supersede(same_id, "reviewer", old_dir, tpl, tpl, ev, reason, letter), "new study_id"),
            (lambda: supersede(rep, "reviewer", old_dir, tpl, tpl, "new", reason, letter), "one of"),
            (lambda: supersede(rep, "reviewer", old_dir, tpl, tpl, "rename", reason, letter), "one of"),
            (lambda: supersede(rep, "reviewer", old_dir, tpl, tpl, ev, "  ", letter), "change_reason"),
            (lambda: supersede(rep, "reviewer", old_dir, tpl, tpl, ev, reason), "cites an evidence file"),
            (lambda: supersede(rep, "reviewer", old_dir, tpl, tpl, ev, reason, os.path.join(ext, "none.txt")),
             "not a stable regular file"),
            (lambda: supersede({"status": "refused"}, "reviewer", old_dir, tpl, tpl, ev, reason, letter),
             "not an accepted")):
        _refused(fn, "REQ-LOCK", needle)
    linked = os.path.join(ext, "letter_copy.txt")
    _copy_bytes(letter, linked)
    os.link(linked, os.path.join(ext, "letter_link.txt"))
    _refused(lambda: supersede(rep, "reviewer", old_dir, tpl, tpl, ev, reason, linked), "REQ-LOCK",
             "not a stable regular file")
    print("[ok] e05 supersede refusals: the same study_id, change_kind new and rename, a blank reason, no evidence,"
          " a missing file, a refused report and a hard-linked evidence file are REQ-LOCK")
    base = copy.deepcopy(rep["requirements"])
    lin_ok = {"supersedes_study": "v1_nominal", "supersedes_lock": v1["lock_sha"], "change_kind": "target_change",
              "change_reason": reason, "evidence_sha": None}
    for patch, needle in (({"change_kind": "new", "supersedes_study": "v1_nominal"}, "change_kind new"),
                          ({"supersedes_study": None}, "names another study"),
                          ({"supersedes_study": "v1_nominal_r2"}, "names another study"),
                          ({"supersedes_lock": "abc"}, "lock sha"),
                          ({"change_reason": ""}, "change_reason"),
                          ({"change_kind": "evidence_correction"}, "evidence file's sha"),
                          ({"evidence_sha": "0" * 64}, "carries no evidence_sha")):
        bad = dict(base, **lin_ok)
        bad.update(patch)
        _refused(lambda: lock({"status": "ok", "requirements": bad}, "reviewer"), "REQ-LOCK", needle)
    assert lock({"status": "ok", "requirements": dict(base, **lin_ok)}, "reviewer")["change_kind"] == "target_change"
    print("[ok] e06 lock lineage: 7 inconsistent lineages refused REQ-LOCK (new with a predecessor, no or own"
          " predecessor, a bad lock sha, no reason, a correction without and a change with an evidence sha)")
    return {"old_dir": old_dir, "rep": rep, "reason": reason, "letter": letter, "doc": doc,
            "datasheet": os.path.join(ext, "vendor_datasheet.json")}


def _selftest_diff_cli(td, built, lin) -> None:
    """(D14) the diff and supersede verbs in fresh processes, byte-equal to the in-process results."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    me = os.path.abspath(__file__)
    old, new, old_t, new_t = built["d10_ids_renumbered"]
    dirs = []
    for name, doc in (("cli_old", old), ("cli_new", new)):
        dirs.append(os.path.join(td, name))
        write_locked(dirs[-1], doc)
    out = os.path.join(td, "diff.json")
    pr = subprocess.run([sys.executable, me, "diff", dirs[0], dirs[1], NOZZLE_DIR, NOZZLE_DIR, out],
                        capture_output=True, encoding="utf-8", errors="replace", env=env, timeout=120)
    assert pr.returncode == 0, (pr.returncode, pr.stderr[-500:])
    d = diff(old, new, old_t, new_t)
    with open(out, "rb") as f:
        assert f.read() == common.canonical_bytes(d) + b"\n"
    assert pr.stdout.strip() == common.canonical_json({"change": "none", "counts": d["counts"]}), pr.stdout
    rep_file = os.path.join(td, "report_r2.json")
    write_canonical(rep_file, lin["rep"])
    new_dir = os.path.join(td, "cli_r2")
    base = [sys.executable, me, "supersede", rep_file, "reviewer", lin["old_dir"], NOZZLE_DIR, NOZZLE_DIR,
            "evidence_correction", lin["reason"], new_dir]
    pr = subprocess.run(base + [lin["letter"]], capture_output=True, encoding="utf-8", errors="replace", env=env,
                        timeout=120)
    assert pr.returncode == 0, (pr.returncode, pr.stderr[-500:])
    assert pr.stdout.strip() == lin["doc"]["lock_sha"] and read_locked(new_dir) == lin["doc"]
    bad_dir = os.path.join(td, "cli_bad")
    pr = subprocess.run(base[:-1] + [bad_dir, lin["datasheet"]], capture_output=True, encoding="utf-8",
                        errors="replace", env=env, timeout=120)
    assert pr.returncode == 1 and "REQ-EVIDENCE-GENERATED" in pr.stderr and not os.path.exists(bad_dir), \
        (pr.returncode, pr.stderr[-300:])
    print("[ok] CLI diff and supersede in fresh processes: d10 diff bytes equal, the letter's lock sha %s.. equal,"
          " the datasheet refused REQ-EVIDENCE-GENERATED with exit 1 and nothing written"
          % (lin["doc"]["lock_sha"][:12],))


def _selftest_diff() -> None:
    """The AMG-7 gates of docs/16a §G.1: D1-D12, D13, E1-E6, D14 (20 [ok] lines)."""
    with tempfile.TemporaryDirectory() as td:
        built = _selftest_diff_pairs(os.path.join(td, "pairs"))
        _selftest_diff_refusals(built)
        lin = _selftest_lineage(os.path.join(td, "lineage"), built)
        _selftest_diff_cli(os.path.join(td, "cli"), built, lin)


def selftest():
    """The thirteen fixture gates of docs/16 §I CAD-07: Z1..Z13 in order, then SELFTEST PASS."""
    cases = common.read_json(CASES)
    decl = common.read_json(os.path.join(NOZZLE_DIR, "template.json"))
    sets = {s["name"]: s for s in cases["sets"]}
    DSHA = cases["declaration_sha"]

    def run(name):
        s = sets[name]
        return check(_fixture_proposal(cases, s), cases["briefs"][s["brief"]], decl, cases["template_sha"], DSHA)

    def judged_right(s, rep):
        got = (rep["status"], [{"row": r["row"], "id": r["id"]} for r in rep["refusals"]])
        want = (s["expect"]["status"], [{"row": r["row"], "id": r["id"]} for r in s["expect"]["refusals"]])
        return got == want

    # (Z1) the unit table converts exactly as measured, and 7.07 kg/s of water becomes 7.07/998.21 m3/s.
    assert to_si(60, "mm") == (0.06, "m")
    assert to_si(35, "deg") == (35 * math.pi / 180.0, "rad")
    assert to_si(30, "°")[0] == 30 * math.pi / 180.0
    assert to_si(7.07, "L/s") == (0.00707, "m3/s")
    assert to_si(5, "%") == (0.05, "1")
    assert to_si(9, "−") == (9.0, "1")
    assert to_si(0.00707, "m³/s") == (0.00707, "m3/s")
    try:
        to_si(1, "furlong")
        raise AssertionError("to_si accepted furlong")
    except ValueError:
        pass
    r2 = run("v2_water_mdot")
    want_q = 7.07 / 998.21
    got_q = r2["derived"]["Q_m3_s"]
    rel = abs(got_q - want_q) / want_q
    assert rel <= 1e-15, rel
    fq = flow_Q(r2["requirements"])
    assert abs(fq - want_q) / want_q <= 1e-15
    assert r2["derived"]["rho_kg_m3"] == 998.21
    r3 = run("v3_exit_velocity")
    assert r3["derived"]["Q_m3_s"] == 22.5 * (math.pi * 0.02 * 0.02 / 4.0)
    assert r3["derived"]["D_e_m"] == 0.02
    r1 = run("v1_nominal")
    assert r1["derived"]["Q_m3_s"] == 0.00707
    print("[ok] unit table: 7 conversions and the furlong refusal ; 7.07 kg/s of water -> %r m3/s"
          " (rel err %g)" % (got_q, rel))

    # (Z2) quote numbers parse as measured; NFC is applied to brief and quote.
    measured = [
        ("The inlet diameter is 60 mm", [(60.0, "mm")]),
        ("the contraction ratio is 9:1", [(9.0, None), (1.0, None)]),
        ("air at 20 °C", [(20.0, None)]),
        ("wall slope under 30°", [(30.0, "°")]),
        ("contraction between 50 and 70 mm long", [(50.0, None), (70.0, "mm")]),
        ("mesh level L1 and Re 30000", [(30000.0, None)]),
        ("5 mmHg", [(5.0, None)]),
        ("0.00707 m³/s", [(0.00707, "m³/s")]),
        ("101,325 Pa", [(101.0, None), (325.0, "Pa")]),
    ]
    for text, want in measured:
        got = quote_numbers(text)
        assert got == want, (text, got, want)
    pre = "R" + "e" + chr(0x301) + "sume: inlet 60 mm"
    qpre = "R" + chr(0xe9) + "sume: inlet 60 mm"
    assert qpre not in pre
    assert nfc(qpre) in nfc(pre)
    p = _fixture_proposal(cases, sets["v1_nominal"])
    p["rows"][0]["quote"] = qpre
    b1 = copy.deepcopy(cases["briefs"]["B1"])
    b1["text"] = pre + ". " + b1["text"]
    rep = check(p, b1, decl, cases["template_sha"], DSHA)
    assert rep["status"] == "ok", rep["status"]
    print("[ok] quote numbers: 9 measured parses, NFC quote-in-brief, v1 accepted with the precomposed quote")

    # (Z3) the EARS sentences are server-rendered exactly; the LLM's draft is discarded.
    v1_ears = [
        "The design shall have inlet diameter at contraction_start of 0.06 m within 1e-06 m.",
        "The design shall have contraction ratio of 9.0 within a relative 1e-06.",
        "The design shall have total length of at most 0.08 m.",
        "The design shall have min wall normal of at least 0.002 m.",
        "The design should have max wall slope of at most 0.6108652381980153 rad.",
        "The design should minimise total length.",
    ]
    doc1 = r1["requirements"]
    assert [r["ears"] for r in doc1["rows"][:6]] == v1_ears
    assert doc1["rows"][0]["ears"] != "The nozzle inlet shall be 60 mm."
    doc7 = run("v7_level_relative_max")["requirements"]
    assert doc7["rows"][3]["ears"] == \
        "While the mesh level is L1, the design should have max wall slope of at most 0.6 rad."
    doc5 = run("v5_band_default_assumed")["requirements"]
    assert doc5["rows"][2]["ears"] == "The design shall have contraction length between 0.05 and 0.07 m."
    for name in sorted(n for n in sets if sets[n]["expect"]["status"] == "ok"):
        for r in run(name)["requirements"]["rows"]:
            assert EARS_RE.match(r["ears"]), (name, r["id"], r["ears"])
    print("[ok] EARS: v1's six sentences verbatim, the v7 level and v5 band lines, all rows match EARS_RE")

    # (Z4) the 8 valid sets are accepted: schema-valid, ids and SYS rows in order, pending lock fields.
    valid_names = sorted(n for n in sets if sets[n]["expect"]["status"] == "ok")
    for name in valid_names:
        rep = run(name)
        s = sets[name]
        assert judged_right(s, rep), name
        assert rep["questions"] == [], name
        d = rep["requirements"]
        assert schema.errors(d, "cad-requirements/1") == [], name
        n_user = len(d["rows"]) - len(SYS_ROWS)
        assert [r["id"] for r in d["rows"]] == ["REQ-%03d" % (i + 1) for i in range(n_user)] \
            + [t[0] for t in SYS_ROWS], name
        assert d["approved_by"] == PENDING and d["lock_sha"] is None, name
    assert [r["locks_params"] for r in doc1["rows"][:6]] == [["D_i"], ["CR"], [], [], [], []]
    assert doc1["objective"] == {"quantity": "total_length", "sense": "min"}
    assert run("v4_sketch_labels")["requirements"]["attachments"] == ["c3" * 32]
    op8 = run("v8_assumed_flow_degree_sign")["requirements"]["operating_point"]
    assert op8["Q_m3_s"] == 0.00707 and op8["U_exit_m_s"] is None and op8["mdot_kg_s"] is None
    print("[ok] 8 of 8 accepted: schema-valid, REQ-001.. ids then the six SYS rows, pending card fields")

    # (Z5) the 12 refusal sets are each refused by exactly one id at the expected row; the ids cover the table.
    refused_names = sorted(n for n in sets if sets[n]["expect"]["status"] == "refused")
    expected_ids = set()
    for name in refused_names:
        rep = run(name)
        s = sets[name]
        assert judged_right(s, rep), (name, rep["refusals"])
        assert len(rep["refusals"]) == 1, name
        assert rep["requirements"] is None and rep["derived"] is None, name
        expected_ids.add(rep["refusals"][0]["id"])
    assert expected_ids == set(CAD07_REFUSAL_IDS), expected_ids ^ set(CAD07_REFUSAL_IDS)
    print("[ok] 12 of 12 refusal sets each refused by exactly one id: %s" % (", ".join(
        [r["id"] for s2 in refused_names for r in run(s2)["refusals"]]),))

    # (Z6) v1's compile equals the hand-derived literal of docs/16 §E.4 and validates as cad-checks/1.
    def _a(feature, where):
        return {"feature": feature, "where": where, "Re": None, "level": None}

    _V1_CHECKS = [
        {"req_id": "REQ-001", "primitive": "diameter_at_plane", "args": _a("contraction_start", ["contraction_start"]),
         "op": "==", "lo": 0.06, "hi": 0.06, "tol": 1e-06, "u": 1e-09, "hardness": "hard"},
        {"req_id": "REQ-002", "primitive": "area_ratio", "args": _a(None, ["contraction_start", "exit_plane"]),
         "op": "==", "lo": 9.0, "hi": 9.0, "tol": 9e-06, "u": 9.000000000000001e-09, "hardness": "hard"},
        {"req_id": "REQ-003", "primitive": "extent_along_axis", "args": _a(None, ["body"]),
         "op": "<=", "lo": None, "hi": 0.08, "tol": 0.0, "u": 1e-09, "hardness": "hard"},
        {"req_id": "REQ-004", "primitive": "meridian_min_wall", "args": _a(None, ["wetted", "outer"]),
         "op": ">=", "lo": 0.002, "hi": None, "tol": 0.0, "u": 1e-08, "hardness": "hard"},
        {"req_id": "REQ-005", "primitive": "slope_max", "args": _a(None, ["wall_contraction"]),
         "op": "<=", "lo": None, "hi": 0.6108652381980153, "tol": 0.0, "u": 6.108652381980153e-07,
         "hardness": "soft"},
        {"req_id": "REQ-006", "primitive": "extent_along_axis", "args": _a(None, ["body"]),
         "op": "<=", "lo": None, "hi": None, "tol": 0.0, "u": 1e-09, "hardness": "objective"},
        {"req_id": "SYS-SOLID", "primitive": "n_solids", "args": _a(None, ["fluid"]),
         "op": "==", "lo": 1.0, "hi": 1.0, "tol": 0.0, "u": 0.0, "hardness": "hard"},
        {"req_id": "SYS-VALID", "primitive": "valid", "args": _a(None, ["fluid"]),
         "op": "is_true", "lo": None, "hi": None, "tol": 0.0, "u": 0.0, "hardness": "hard"},
        {"req_id": "SYS-WATERTIGHT", "primitive": "watertight", "args": _a(None, ["fluid"]),
         "op": "is_true", "lo": None, "hi": None, "tol": 0.0, "u": 0.0, "hardness": "hard"},
        {"req_id": "SYS-AXIS", "primitive": "axis_x", "args": _a(None, ["fluid"]),
         "op": "is_true", "lo": None, "hi": None, "tol": 0.0, "u": 0.0, "hardness": "hard"},
        {"req_id": "SYS-UNITS", "primitive": "units_m", "args": _a(None, ["fluid"]),
         "op": "is_true", "lo": None, "hi": None, "tol": 0.0, "u": 0.0, "hardness": "hard"},
        {"req_id": "SYS-MACH", "primitive": "mach_max", "args": _a(None, ["fluid"]),
         "op": "<=", "lo": None, "hi": 0.3, "tol": 0.0, "u": None, "hardness": "hard"},
    ]
    for c, rp in zip(_V1_CHECKS, ["brep"] * 8 + ["stl", "brep", "brep", "cfd"]):
        c["repr"] = rp                        # docs/16a §F: SYS-WATERTIGHT is judged on the STL, SYS-MACH by CFD
    locked1 = lock(r1, "reviewer")
    k1 = compile_checks(locked1, decl, DSHA)
    assert k1["checks"] == _V1_CHECKS, "v1 compile differs from the hand-derived checks"
    assert k1["declaration_sha"] == locked1["declaration_sha"] == DSHA
    assert k1["requirements_lock"] == locked1["lock_sha"]
    assert schema.errors(k1, "cad-checks/1") == []
    print("[ok] v1 compile: 12 checks byte-equal to the hand-derived literal, cad-checks/1 valid")

    # (Z7) the golden compiles: the bytes of (C11), recomputed now, equal the committed golden files.
    for name in valid_names:
        rep = run(name)
        locked = lock(rep, cases["approved_by"])
        checks = compile_checks(locked, decl, DSHA)
        blob = {"$comment": common.HEADER_COMMENT, "requirements": locked, "checks": checks}
        want = common.canonical_bytes(blob) + b"\n"
        with open(os.path.join(GOLDEN, name + ".json"), "rb") as f:
            got = f.read()
        assert got == want, "golden %s differs from the recomputed bytes" % (name,)
    print("[ok] 8 of 8 golden compiles byte-identical: %s..%s" % (valid_names[0], valid_names[-1]))

    # (Z8) the lock: sha rule, tamper detection, GATE-LOCK in compile and read_locked, refusal to lock.
    locked1 = lock(r1, "reviewer")
    assert locked1["lock_sha"] == common.sha256_of(dict(locked1, lock_sha=None))
    assert lock_ok(locked1)
    tampered = copy.deepcopy(locked1)
    tampered["rows"][2]["value"] = 0.09
    assert not lock_ok(tampered)
    try:
        compile_checks(tampered, decl, DSHA)
        raise AssertionError("compile_checks accepted a tampered lock")
    except ValueError as e:
        assert str(e).startswith("GATE-LOCK"), e
    with tempfile.TemporaryDirectory() as td:
        write_locked(td, locked1)
        assert read_locked(td) == locked1
        common.atomic_write(os.path.join(td, "requirements.lock"), "0" * 64 + chr(10))
        try:
            read_locked(td)
            raise AssertionError("read_locked accepted a zeroed lock file")
        except ValueError as e:
            assert str(e).startswith("GATE-LOCK"), e
    for bad_report, bad_by in ((run("x01_qty"), "reviewer"), (r1, ""), (r1, PENDING)):
        try:
            lock(bad_report, bad_by)
            raise AssertionError("lock accepted %r" % (bad_by,))
        except ValueError as e:
            assert str(e).startswith("REQ-LOCK"), e
    print("[ok] lock: sha seal, tamper and zeroed lock file refused with GATE-LOCK, unapproved refused")

    # (Z9) missing drivers become questions, not refusals, and such a document cannot be locked.
    def without_row(name, idx):
        p = _fixture_proposal(cases, sets[name])
        del p["rows"][idx]
        return check(p, cases["briefs"][sets[name]["brief"]], decl, cases["template_sha"], DSHA)

    rep = without_row("v1_nominal", 1)
    assert rep["status"] == "questions" and [q["id"] for q in rep["questions"]] == ["Q-CR"], rep["questions"]
    assert rep["requirements"]["lock_sha"] is None and rep["derived"] is None
    assert schema.errors(rep["requirements"], "cad-requirements/1") == []
    try:
        lock(rep, "reviewer")
        raise AssertionError("lock accepted a questions report")
    except ValueError as e:
        assert str(e).startswith("REQ-LOCK"), e
    rep = without_row("v1_nominal", 0)
    assert [q["id"] for q in rep["questions"]] == ["Q-D_i"], rep["questions"]
    rep = without_row("v3_exit_velocity", 0)
    assert [q["id"] for q in rep["questions"]] == ["Q-D_i", "Q-CR"], rep["questions"]
    p = _fixture_proposal(cases, sets["v1_nominal"])
    p["operating_point"]["flow"] = None
    rep = check(p, cases["briefs"]["B1"], decl, cases["template_sha"], DSHA)
    assert [q["id"] for q in rep["questions"]] == ["Q-FLOW"], rep["questions"]
    op = rep["requirements"]["operating_point"]
    assert op["U_exit_m_s"] is None and op["Q_m3_s"] is None and op["mdot_kg_s"] is None
    print("[ok] questions: Q-CR, Q-D_i, Q-D_i+Q-CR and Q-FLOW asked, documents stay lockable-later only")

    # (Z10) operating-point, flow and row-shape errors are refusals by id, one each, at the exact check.
    def one_ref(mutate):
        p = _fixture_proposal(cases, sets["v1_nominal"])
        mutate(p)
        rep = check(p, cases["briefs"]["B1"], decl, cases["template_sha"], DSHA)
        assert len(rep["refusals"]) == 1, rep["refusals"]
        r = rep["refusals"][0]
        return (r["row"], r["id"], r["check"])

    def setk(p, idx, key, value):
        p["rows"][idx][key] = value

    cases10 = [
        (lambda p: p["operating_point"].__setitem__("fluid", "oil"),
         ("operating_point", "REQ-OP", "shape")),
        (lambda p: p["operating_point"]["flow"].__setitem__("field", "Q_l_s"),
         ("flow", "REQ-OP", "shape")),
        (lambda p: p["operating_point"]["flow"].__setitem__("unit", "kg/s"),
         ("flow", "REQ-UNIT", "dimension")),
        (lambda p: p["operating_point"]["flow"].__setitem__("value", 7.1),
         ("flow", "REQ-GROUND", "value")),
        (lambda p: setk(p, 2, "id", "REQ-999"), (2, "REQ-OP", "unknown_key")),
        (lambda p: p["rows"][2].pop("quote"), (2, "REQ-OP", "missing_key")),
        (lambda p: setk(p, 2, "value", True), (2, "REQ-OP", "type")),
        (lambda p: setk(p, 0, "feature", "outlet"), (0, "REQ-QTY", "feature")),
        (lambda p: (setk(p, 2, "op", "in"), setk(p, 2, "upper", 70)), (2, "REQ-OP", "shape")),
        (lambda p: setk(p, 0, "tol_rel", 1e-06), (0, "REQ-OP", "tol_both")),
    ]
    for mutate, want in cases10:
        got = one_ref(mutate)
        assert got == want, (want, got)
    print("[ok] 10 single refusals at exact (row, id, check): op point, flow, keys, types, qty, shape, tol")

    # (Z11) the set passes against the SYS rows and the analytic box, each exactly one refusal.
    def appended_row(p, row):
        p["rows"].append(row)

    def assumed(quantity, op, value, unit, hardness, upper=None, tol_rel=None):
        return {"quantity": quantity, "feature": None, "op": op, "value": value, "upper": upper,
                "tol_abs": None, "tol_rel": tol_rel, "unit": unit, "condition": {"Re": None, "level": None},
                "hardness": hardness, "source": "assumed", "quote": None,
                "ticked": True}                    # REQ-DEFAULT-HARD covers assumed rows; the card ticked these

    p = _fixture_proposal(cases, sets["v1_nominal"])
    appended_row(p, assumed("watertight", "is_true", None, "-", "hard"))
    rep = check(p, cases["briefs"]["B1"], decl, cases["template_sha"], DSHA)
    assert [(r["row"], r["id"]) for r in rep["refusals"]] == [(6, "REQ-DUP")], rep["refusals"]
    p = _fixture_proposal(cases, sets["v1_nominal"])
    appended_row(p, assumed("n_solids", ">=", 2, "-", "hard"))
    rep = check(p, cases["briefs"]["B1"], decl, cases["template_sha"], DSHA)
    assert [(r["row"], r["id"]) for r in rep["refusals"]] == [(6, "REQ-CONFLICT")], rep["refusals"]
    # the feature names WHERE the row is measured, and the primitive measures a quantity at its catalogue where
    # whatever the feature says: rows 0 (feature contraction_start) and these (feature null) are one measurement,
    # so a second "==" on it is a duplicate and a ">=" beyond it a conflict (both were accepted when the key
    # held the raw feature)
    p = _fixture_proposal(cases, sets["v1_nominal"])
    r61 = assumed("inlet_diameter", "==", 61, "mm", "hard")
    r61["tol_abs"] = 0.001
    appended_row(p, r61)
    rep = check(p, cases["briefs"]["B1"], decl, cases["template_sha"], DSHA)
    assert [(r["row"], r["id"]) for r in rep["refusals"]] == [(6, "REQ-DUP")], rep["refusals"]
    p = _fixture_proposal(cases, sets["v1_nominal"])
    appended_row(p, assumed("inlet_diameter", ">=", 61, "mm", "hard"))
    rep = check(p, cases["briefs"]["B1"], decl, cases["template_sha"], DSHA)
    assert [(r["row"], r["id"]) for r in rep["refusals"]] == [(6, "REQ-CONFLICT")], rep["refusals"]
    p = _fixture_proposal(cases, sets["v1_nominal"])
    appended_row(p, assumed("contraction_length", ">=", 100, "mm", "soft"))
    rep = check(p, cases["briefs"]["B1"], decl, cases["template_sha"], DSHA)
    assert [(r["row"], r["id"]) for r in rep["refusals"]] == [(6, "REQ-OUTSIDE")], rep["refusals"]
    p = _fixture_proposal(cases, sets["v1_nominal"])
    p["rows"][3] = assumed("min_wall_normal", ">=", 12, "mm", "soft")
    rep = check(p, cases["briefs"]["B1"], decl, cases["template_sha"], DSHA)
    assert [(r["row"], r["id"]) for r in rep["refusals"]] == [(3, "REQ-OUTSIDE")], rep["refusals"]
    p = _fixture_proposal(cases, sets["v1_nominal"])
    p["rows"][1] = assumed("contraction_ratio", "==", 0.5, "-", "hard", tol_rel=1e-06)
    rep = check(p, cases["briefs"]["B1"], decl, cases["template_sha"], DSHA)
    assert [(r["row"], r["id"]) for r in rep["refusals"]] == [(1, "REQ-OUTSIDE")], rep["refusals"]
    print("[ok] set passes: DUP vs SYS-WATERTIGHT, CONFLICT vs SYS-SOLID, DUP and CONFLICT across features, OUTSIDE from the analytic box")

    # (Z12) check is pure and deterministic; the CLI's verbs work in fresh processes.
    p0 = _fixture_proposal(cases, sets["v1_nominal"])
    b0 = copy.deepcopy(cases["briefs"]["B1"])
    d0 = copy.deepcopy(decl)
    sha0 = cases["template_sha"]
    ra = check(p0, b0, d0, sha0, DSHA)
    assert p0 == _fixture_proposal(cases, sets["v1_nominal"]) and b0 == cases["briefs"]["B1"] and d0 == decl
    rb = check(p0, b0, d0, sha0, DSHA)
    assert common.canonical_json(ra) == common.canonical_json(rb)
    with tempfile.TemporaryDirectory() as td:
        pj = os.path.join(td, "proposal.json")
        bj = os.path.join(td, "brief.json")
        write_canonical(pj, p0)
        write_canonical(bj, b0)
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        outs = []
        for i in (1, 2):
            out_i = os.path.join(td, "report%d.json" % (i,))
            pr = subprocess.run([sys.executable, os.path.abspath(__file__), "check", pj, bj, NOZZLE_DIR, out_i],
                                capture_output=True, encoding="utf-8", errors="replace", env=env, timeout=120)
            assert pr.returncode == 0, (pr.returncode, pr.stderr[-500:])
            outs.append(out_i)
        with open(outs[0], "rb") as f:
            blob1 = f.read()
        with open(outs[1], "rb") as f:
            blob2 = f.read()
        assert blob1 == blob2, "the two CLI check runs differ"
        rep_doc = common.read_json(outs[0])["requirements"]
        assert rep_doc["template_sha"] == common.sha256_file(os.path.join(NOZZLE_DIR, "template.py"))
        study = os.path.join(td, "study")
        pr = subprocess.run([sys.executable, os.path.abspath(__file__), "lock", outs[0], "reviewer", study],
                            capture_output=True, encoding="utf-8", errors="replace", env=env, timeout=120)
        assert pr.returncode == 0, (pr.returncode, pr.stderr[-500:])
        assert read_locked(study) == lock(common.read_json(outs[0]), "reviewer")
        gold1 = common.read_json(os.path.join(GOLDEN, "v1_nominal.json"))
        req_file = os.path.join(td, "reqs.json")
        write_canonical(req_file, gold1["requirements"])
        comp = os.path.join(td, "checks.json")
        pr = subprocess.run([sys.executable, os.path.abspath(__file__), "compile", req_file, NOZZLE_DIR, comp],
                            capture_output=True, encoding="utf-8", errors="replace", env=env, timeout=120)
        assert pr.returncode == 0, (pr.returncode, pr.stderr[-500:])
        with open(comp, "rb") as f:
            assert f.read() == common.canonical_bytes(gold1["checks"]) + b"\n"
        bad_file = os.path.join(td, "tampered.json")
        write_canonical(bad_file, tampered)
        comp2 = os.path.join(td, "checks2.json")
        pr = subprocess.run([sys.executable, os.path.abspath(__file__), "compile", bad_file, NOZZLE_DIR, comp2],
                            capture_output=True, encoding="utf-8", errors="replace", env=env, timeout=120)
        assert pr.returncode == 1 and "GATE-LOCK" in pr.stderr, (pr.returncode, pr.stderr[-300:])
        x01p = os.path.join(td, "x01.json")
        write_canonical(x01p, _fixture_proposal(cases, sets["x01_qty"]))
        outx = os.path.join(td, "x01_report.json")
        pr = subprocess.run([sys.executable, os.path.abspath(__file__), "check", x01p, bj, NOZZLE_DIR, outx],
                            capture_output=True, encoding="utf-8", errors="replace", env=env, timeout=120)
        assert pr.returncode == 1, (pr.returncode, pr.stderr[-300:])
    print("[ok] purity: args untouched, deterministic bytes, CLI check/lock/compile in fresh processes")

    # (Z13) the fixture inventory: 20 sets, 8 ok and 12 refused, 8 golden files.
    names = [s["name"] for s in cases["sets"]]
    assert len(names) == len(set(names)) == 20
    assert sorted(s["expect"]["status"] for s in cases["sets"]).count("ok") == 8
    assert sorted(s["expect"]["status"] for s in cases["sets"]).count("refused") == 12
    assert sorted(os.listdir(GOLDEN)) == sorted(n + ".json" for n in valid_names)
    print("[ok] inventory: 20 unique sets, 8 ok + 12 refused, %d golden files" % (len(os.listdir(GOLDEN)),))

    # (A0) the fixtures record the declaration and the vocabulary they were judged with; both are today's.
    B1 = cases["briefs"]["B1"]
    TSHA = cases["template_sha"]
    assert DSHA == common.sha256_file(os.path.join(NOZZLE_DIR, "template.json")), "re-record declaration_sha"
    assert cases["vocab_sha"] == vocab_sha(decl), "re-record vocab_sha"
    assert decl["standards"] == []
    lin = common.read_json(LINEAGE)
    lin_names = sorted([s["name"] for s in lin["sets"]] + [p["name"] for p in lin["procedures"]])
    assert len(lin_names) == 10 and [n[:3] for n in lin_names] == ["a%02d" % i for i in range(1, 11)], lin_names
    print("[ok] recorded shas: declaration %s.. and vocab %s.. are today's; standards table empty; 10 lineage"
          " fixtures" % (DSHA[:12], cases["vocab_sha"][:12]))

    # (A1) the flow's source and quote survive check, lock, write_locked and read_locked byte for byte.
    with tempfile.TemporaryDirectory() as td:
        for name in ("v1_nominal", "v2_water_mdot", "v8_assumed_flow_degree_sign"):
            flow = _fixture_proposal(cases, sets[name])["operating_point"]["flow"]
            d = os.path.join(td, name)
            write_locked(d, lock(run(name), "reviewer"))
            op = read_locked(d)["operating_point"]
            assert (op["flow_source"], op["flow_quote"]) == (flow["source"], flow["quote"]), name
            assert common.canonical_json(op["flow_quote"]) == common.canonical_json(flow["quote"]), name
            assert (op["fluid_source"], op["T_K_source"], op["p0_Pa_source"]) == ("assumed",) * 3, name
    print("[ok] a01 flow provenance: the brief, brief and assumed flows of v1, v2 and v8 keep source and quote"
          " byte for byte through lock, write_locked and read_locked")

    # (A2) a template.json edit moves declaration_sha, the lock and the eval key; compile refuses the old lock.
    with tempfile.TemporaryDirectory() as td:
        tdir = os.path.join(td, "nozzle_contraction")
        os.makedirs(tdir)
        for fn in ("template.json", "template.py"):
            with open(os.path.join(NOZZLE_DIR, fn), "rb") as f:
                blob = f.read()
            with open(os.path.join(tdir, fn), "wb") as f:
                f.write(blob)
        d2 = copy.deepcopy(decl)
        d2["title"] = d2["title"] + " (edited)"
        with open(os.path.join(tdir, "template.json"), "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(d2, indent=2, ensure_ascii=False) + chr(10))
        decl2, sha2, dsha2 = load_template(tdir)
        decl1, sha1, dsha1 = load_template(NOZZLE_DIR)
    assert sha2 == sha1 and dsha1 == DSHA and dsha2 != dsha1 and decl2 == d2
    assert vocab_sha(decl2) == vocab_sha(decl1)             # the title is not vocabulary
    pv = _fixture_proposal(cases, sets["v1_nominal"])
    l1 = lock(check(pv, B1, decl1, TSHA, dsha1), "reviewer")
    l2 = lock(check(pv, B1, decl2, TSHA, dsha2), "reviewer")
    assert (l1["declaration_sha"], l2["declaration_sha"]) == (dsha1, dsha2) and l1["lock_sha"] != l2["lock_sha"]
    assert compile_checks(l2, decl2, dsha2)["declaration_sha"] == dsha2
    try:
        compile_checks(l1, decl2, dsha2)
        raise AssertionError("compile_checks accepted a lock bound to the old declaration")
    except ValueError as e:
        assert str(e).startswith("GATE-LOCK"), e
    parts = {"template_sha": sha1, "declaration_sha": dsha1, "params": {"D_i": 0.06, "CR": 9.0},
             "requirements_lock": l1["lock_sha"], "gates_lock": "0" * 64, "env": {"python": "fixture"},
             "mesh_recipe_version": 1, "case_writer_version": 1, "bin_sha": "1" * 64}
    k_1 = eval_key(parts)
    k_2 = eval_key(dict(parts, declaration_sha=dsha2))
    assert k_1 != k_2 and k_1 == eval_key(copy.deepcopy(parts))
    assert eval_key(dict(parts, declaration_sha=dsha2, requirements_lock=l2["lock_sha"])) not in (k_1, k_2)
    for bad in (dict(parts, extra=1), {k: parts[k] for k in EVAL_KEY_PARTS[1:]}):
        try:
            eval_key(bad)
            raise AssertionError("eval_key accepted the parts %r" % (sorted(bad),))
        except ValueError as e:
            assert str(e).startswith("EVAL-KEY"), e
    print("[ok] a02 declaration edit: declaration_sha %s.. -> %s.., lock and eval key move, the vocabulary does"
          " not, compile refuses the old lock GATE-LOCK" % (dsha1[:8], dsha2[:8]))

    # (A3..A6, A9) the proposal fixtures: each refused by exactly its (row, id, check), naming its field.
    new_ids = set()
    for s in lin["sets"]:
        rep = check(_fixture_proposal(cases, s), cases["briefs"][s["brief"]], decl, TSHA, DSHA)
        got = [(r["row"], r["id"], r["check"]) for r in rep["refusals"]]
        want = [(r["row"], r["id"], r["check"]) for r in s["expect"]["refusals"]]
        assert rep["status"] == s["expect"]["status"] and got == want, (s["name"], got)
        assert s["expect"]["detail_has"] in rep["refusals"][0]["detail"], (s["name"], rep["refusals"][0])
        new_ids.add(got[0][1])
        print("[ok] %s: refused %s %s at row %s (%r in the detail)"
              % (s["name"], got[0][1], got[0][2], got[0][0], s["expect"]["detail_has"]))
    assert new_ids == {"REQ-DEFAULT-HARD", "REQ-OP", "REQ-STD", "REQ-VOCAB"}, new_ids

    # (A7, A8) requirements.json and .lock are write-once: other bytes REQ-IMMUTABLE, identical bytes accepted.
    with tempfile.TemporaryDirectory() as td:
        la = lock(r1, "reviewer")
        pd, pl = write_locked(td, la)
        before = [(os.stat(q).st_mtime_ns, common.sha256_file(q)) for q in (pd, pl)]
        for bad_dir, doc_b, name_b in ((td, lock(r1, "another-reviewer"), "requirements.json"),
                                       (os.path.join(td, "lone_lock"), la, "requirements.lock"),
                                       (os.path.join(td, "dir_in_place"), la, "requirements.json")):
            if name_b == "requirements.lock":
                common.atomic_write(os.path.join(bad_dir, name_b), "0" * 64 + chr(10))
            elif bad_dir != td:
                os.makedirs(os.path.join(bad_dir, name_b))
            try:
                write_locked(bad_dir, doc_b)
                raise AssertionError("write_locked rewrote %s in %s" % (name_b, bad_dir))
            except ValueError as e:
                assert str(e).startswith("REQ-IMMUTABLE: " + name_b), e
        assert not os.path.exists(os.path.join(td, "lone_lock", "requirements.json"))
        assert [(os.stat(q).st_mtime_ns, common.sha256_file(q)) for q in (pd, pl)] == before
        print("[ok] a07 immutable: another approver's bytes, a lone rewritten lock and a directory in place are"
              " refused REQ-IMMUTABLE; nothing written")
        assert write_locked(td, copy.deepcopy(la)) == (pd, pl)
        assert [(os.stat(q).st_mtime_ns, common.sha256_file(q)) for q in (pd, pl)] == before
        assert read_locked(td) == la
        print("[ok] a08 immutable: identical bytes accepted, both files untouched (mtime and sha)")

    # (A10) reqs.py vocab is byte-identical in two fresh processes and moves with one catalogue quantity.
    with tempfile.TemporaryDirectory() as td:
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        blobs = []
        for i in (1, 2):
            out_i = os.path.join(td, "vocab%d.json" % (i,))
            pr = subprocess.run([sys.executable, os.path.abspath(__file__), "vocab", NOZZLE_DIR, out_i],
                                capture_output=True, encoding="utf-8", errors="replace", env=env, timeout=120)
            assert pr.returncode == 0 and pr.stdout.strip() == vocab_sha(decl), (pr.returncode, pr.stderr[-300:])
            with open(out_i, "rb") as f:
                blobs.append(f.read())
    assert blobs[0] == blobs[1], "two fresh vocab runs differ"
    vj = json.loads(blobs[0].decode("utf-8"))
    assert vj["vocab_sha"] == vocab_sha(decl) == cases["vocab_sha"] and vj["vocab"] == vocab(decl)
    assert [q["quantity"] for q in vj["vocab"]["quantities"]] == [c["quantity"] for c in decl["catalogue"]]
    d3 = copy.deepcopy(decl)
    d3["catalogue"].append({"quantity": "throat_area", "primitive": "section_at_plane", "where": ["exit_plane"],
                            "kind": "geometric", "method": "geometry", "unit": "m2", "u_kind": "rel",
                            "u_meas": 1e-09})
    assert vocab_sha(d3) != vocab_sha(decl) and vocab(d3)["quantities"][-1]["quantity"] == "throat_area"
    rep = check(_fixture_proposal(cases, sets["v1_nominal"]), B1, d3, TSHA, DSHA)
    assert [(r["row"], r["id"]) for r in rep["refusals"]] == [("proposal", "REQ-VOCAB")], rep["refusals"]
    print("[ok] a10 vocab: %d bytes byte-identical in two fresh processes, sha %s..; one more catalogue quantity"
          " moves it and the old proposal is refused REQ-VOCAB" % (len(blobs[0]), vj["vocab_sha"][:12]))

    # (A11) confidence is derived from the source; a table standard is admitted; a ref on a brief row is not.
    assert CONFIDENCE == {"brief": "high", "sketch_label": "medium", "default": "low", "assumed": "low",
                          "standard": "high", "system": "high"}
    seen = set()
    for name in valid_names:
        for r in run(name)["requirements"]["rows"]:
            assert r["confidence"] == CONFIDENCE[r["source"]] and r["standard_ref"] is None, (name, r["id"])
            assert r["ticked"] is (r["source"] == "default"), (name, r["id"])
            seen.add((r["source"], r["confidence"]))
    assert seen == {("brief", "high"), ("sketch_label", "medium"), ("default", "low"), ("assumed", "low"),
                    ("system", "high")}, seen
    d4 = copy.deepcopy(decl)
    d4["standards"] = [{"ref": "FIXTURE-STD-1", "title": "a fixture standard", "url": "https://example.org/std1"}]
    p = _fixture_proposal(cases, sets["v1_nominal"])
    p["vocab_sha"] = vocab_sha(d4)
    p["rows"].append({"quantity": "min_curvature_radius", "feature": None, "op": ">=", "value": 5, "upper": None,
                      "tol_abs": None, "tol_rel": None, "unit": "mm", "condition": {"Re": None, "level": None},
                      "hardness": "soft", "source": "standard", "quote": None, "standard_ref": "FIXTURE-STD-1"})
    rep = check(p, B1, d4, TSHA, DSHA)
    row6 = rep["requirements"]["rows"][6]
    assert rep["status"] == "ok" and (row6["source"], row6["confidence"], row6["standard_ref"]) == \
        ("standard", "high", "FIXTURE-STD-1")
    p["rows"][0]["standard_ref"] = "FIXTURE-STD-1"
    rep = check(p, B1, d4, TSHA, DSHA)
    assert [(r["row"], r["id"], r["check"]) for r in rep["refusals"]] == [(0, "REQ-STD", "ref_on_non_standard")]
    p = _fixture_proposal(cases, sets["v1_nominal"])
    p["operating_point"]["fluid_source"] = "brief"
    rep = check(p, B1, decl, TSHA, DSHA)
    assert rep["status"] == "ok" and rep["requirements"]["operating_point"]["fluid_source"] == "brief"
    p["operating_point"]["T_K_source"] = "guess"
    rep = check(p, B1, decl, TSHA, DSHA)
    assert [(r["row"], r["id"], r["check"]) for r in rep["refusals"]] == [("operating_point", "REQ-OP", "shape")]
    print("[ok] confidence derived per source on all 8 valid sets; a standard row from a 1-entry table admitted"
          " high; a standard_ref on a brief row refused REQ-STD; fluid_source brief kept, T_K_source guess refused")
    _selftest_diff()
    print("SELFTEST PASS")


def main(argv) -> int:
    """The CLI of docs/16 §I CAD-07 and docs/16a AMG-6, AMG-7: --selftest, check, lock, compile, vocab, diff,
    supersede."""
    if argv and argv[0] == "--selftest":
        try:
            selftest()
        except Exception:
            traceback.print_exc()
            return 1
        return 0
    try:
        if len(argv) == 5 and argv[0] == "check":
            decl, sha, dsha = load_template(argv[3])
            report = check(common.read_json(argv[1]), common.read_json(argv[2]), decl, sha, dsha)
            write_canonical(argv[4], report)
            print(common.canonical_json({
                "status": report["status"],
                "refusals": [r["id"] for r in report["refusals"]],
                "questions": [q["id"] for q in report["questions"]]}))
            return 0 if report["status"] == "ok" else 1
        if len(argv) == 4 and argv[0] == "lock":
            doc = lock(common.read_json(argv[1]), argv[2])
            write_locked(argv[3], doc)
            print(doc["lock_sha"])
            return 0
        if len(argv) == 4 and argv[0] == "compile":
            decl, _sha, dsha = load_template(argv[2])
            write_canonical(argv[3], compile_checks(common.read_json(argv[1]), decl, dsha))
            return 0
        if len(argv) == 3 and argv[0] == "vocab":
            v = vocab(load_template(argv[1])[0])
            write_canonical(argv[2], {"vocab": v, "vocab_sha": common.sha256_of(v)})
            print(common.sha256_of(v))
            return 0
        if len(argv) == 6 and argv[0] == "diff":
            d = diff(read_locked(argv[1]), read_locked(argv[2]), load_template(argv[3]), load_template(argv[4]))
            write_canonical(argv[5], d)
            print(common.canonical_json({"change": d["change"], "counts": d["counts"]}))
            return 0
        if len(argv) in (9, 10) and argv[0] == "supersede":
            doc = supersede(common.read_json(argv[1]), argv[2], argv[3], load_template(argv[4]),
                            load_template(argv[5]), argv[6], argv[7], argv[9] if len(argv) == 10 else None)
            write_locked(argv[8], doc)
            print(doc["lock_sha"])
            return 0
    except ValueError as e:
        print("reqs: %s" % (e,), file=sys.stderr)
        return 1
    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
