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
REFUSAL_IDS = ("REQ-QTY", "REQ-UNIT", "REQ-QUOTE", "REQ-GROUND", "REQ-OP", "REQ-TOL", "REQ-DUP", "REQ-CONFLICT",
               "REQ-OUTSIDE", "REQ-PIXEL", "REQ-OBJ", "REQ-DEFAULT-HARD")
REPORT_KEYS = ("version", "status", "refusals", "questions", "requirements", "derived")
PROPOSAL_KEYS = ("study_id", "template_id", "operating_point", "rows", "created_by")
OPPOINT_KEYS = ("fluid", "T_K", "p0_Pa", "flow")
FLOW_KEYS = ("field", "value", "unit", "source", "quote")
FLOW_FIELDS = {"U_exit_m_s": "m/s", "Q_m3_s": "m3/s", "mdot_kg_s": "kg/s"}
FLOW_SOURCES = ("brief", "sketch_label", "assumed")
ROW_IN_KEYS = ("quantity", "feature", "op", "value", "upper", "tol_abs", "tol_rel", "unit", "condition",
               "hardness", "source", "quote")
ROW_IN_OPTIONAL = ("ears", "ticked")     # ears: the LLM's draft, discarded; ticked: the card's tick on a default row
ROW_KEYS = ("id", "ears", "quantity", "feature", "op", "value", "upper", "tol_abs", "tol_rel", "unit", "kind",
            "method", "condition", "hardness", "source", "quote", "locks_params")
OPS = ("<=", ">=", "==", "in", "is_true")
HARDNESS = ("hard", "soft", "objective")
SOURCES = ("brief", "sketch_label", "default", "assumed")
BOOLEAN_PRIMITIVES = ("valid", "watertight", "axis_x", "units_m")
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
NOZZLE_DIR = os.path.join(HERE, "templates", "nozzle_contraction")
USAGE = ("usage: python reqs.py --selftest" + chr(10)
         + "       python reqs.py check PROPOSAL_JSON BRIEF_JSON TEMPLATE_DIR OUT_JSON" + chr(10)
         + "       python reqs.py lock REPORT_JSON APPROVED_BY OUT_DIR" + chr(10)
         + "       python reqs.py compile REQUIREMENTS_JSON TEMPLATE_DIR OUT_JSON")


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


def _row_pass(row, cat, brief, labels):
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
    if row["hardness"] == "hard" and row["source"] == "default" and row.get("ticked") is not True:
        return _refuse(0, "REQ-DEFAULT-HARD", "not_ticked",
                       "the default row on %s is hard but carries no tick of approval on the card"
                       % (cat["quantity"],))
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
            "locks_params": []}


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
           "locks_params": locks}
    out["ears"] = render_ears(out)
    return out


def _build_document(proposal, brief, template_sha, user_recs, catalogue) -> dict:
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
            "brief_sha": common.sha256_of(brief),
            "attachments": [a["sha256"] for a in brief["attachments"]],
            "operating_point": {"fluid": opp["fluid"], "T_K": float(opp["T_K"]), "p0_Pa": float(opp["p0_Pa"]),
                                "U_exit_m_s": flow_fields["U_exit_m_s"], "Q_m3_s": flow_fields["Q_m3_s"],
                                "mdot_kg_s": flow_fields["mdot_kg_s"]},
            "rows": rows, "objective": objective,
            "created_by": copy.deepcopy(proposal["created_by"]),
            "approved_by": PENDING, "lock_sha": None}


def check(proposal, brief, declaration, template_sha) -> dict:
    """The S1 gate (docs/16 §D, §E.3): pure; refuses the LLM's rows by rule id, asks for missing drivers,
    or returns the report with its cad-requirements/1 document. Never mutates an argument."""
    proposal = copy.deepcopy(proposal)
    brief = copy.deepcopy(brief)
    _check_brief(brief)
    _check_envelope(proposal, declaration)
    if not isinstance(template_sha, str) or not SHA_RE.match(template_sha):
        raise ValueError("template_sha: %r is not a 64 lowercase hex sha256" % (template_sha,))
    catalogue = {c["quantity"]: c for c in declaration["catalogue"]}
    params = {p["name"]: p for p in declaration["params"]}
    labels = [lab for a in brief["attachments"] for lab in a["labels"]]
    refusals = []
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
        ref = _row_pass(row, cat, brief, labels)
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
        document = _build_document(proposal, brief, template_sha, live, catalogue)
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
    if set(opp.keys()) != set(OPPOINT_KEYS):
        return bad("keys must be exactly %s" % (", ".join(OPPOINT_KEYS),))
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


def lock(report, approved_by) -> dict:
    """Seal an accepted report into a locked cad-requirements/1 document (docs/16 §E.8, §F 1)."""
    if not isinstance(report, dict) or report.get("status") != "ok":
        raise ValueError("REQ-LOCK: the report status is %r, not an accepted requirement set"
                         % (report.get("status") if isinstance(report, dict) else None,))
    if not isinstance(approved_by, str) or not approved_by or approved_by == PENDING:
        raise ValueError("REQ-LOCK: approved_by must be a non-empty name other than %r, got %r"
                         % (PENDING, approved_by))
    doc = copy.deepcopy(report["requirements"])
    doc["approved_by"] = approved_by
    doc["lock_sha"] = lock_sha_of(doc)
    errs = schema.errors(doc, "cad-requirements/1")
    if errs:
        raise ValueError("REQ-LOCK: the locked document violates cad-requirements/1: %s" % (errs[0],))
    return doc


def compile_checks(doc, declaration) -> dict:
    """The pure compile of docs/16 §E.4: a locked document into cad-checks/1, one check per row in row order."""
    if not lock_ok(doc):
        raise ValueError("GATE-LOCK: the requirements lock does not match its document")
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
                       "hardness": row["hardness"]})
    out = {"schema": "cad-checks/1", "study_id": doc["study_id"], "requirements_lock": doc["lock_sha"],
           "checks": checks}
    errs = schema.errors(out, "cad-checks/1")
    if errs:
        raise ValueError("reqs: the compiled checks violate cad-checks/1: %s" % (errs[0],))
    return out


def write_canonical(path, obj) -> None:
    """The only writer reqs.py uses: canonical JSON plus one newline, atomically (docs/16 §D)."""
    common.atomic_write(path, common.canonical_json(obj) + chr(10))


def write_locked(out_dir, doc) -> tuple:
    """requirements.json plus requirements.lock; refuses an unlocked document (docs/16 §E.8)."""
    if not lock_ok(doc):
        raise ValueError("GATE-LOCK: the requirements lock does not match its document")
    p_doc = os.path.join(out_dir, "requirements.json")
    p_lock = os.path.join(out_dir, "requirements.lock")
    write_canonical(p_doc, doc)
    common.atomic_write(p_lock, doc["lock_sha"] + chr(10))
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
    """(declaration, template_sha): the template's DECLARATION and the sha of its module (docs/16 §D)."""
    return (common.read_json(os.path.join(template_dir, "template.json")),
            common.sha256_file(os.path.join(template_dir, "template.py")))


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
        return copy.deepcopy(s["proposal"])
    base = None
    for other in cases["sets"]:
        if other["name"] == s["base"]:
            base = other
            break
    if base is None:
        raise ValueError("fixture patch: base %r not found" % (s["base"],))
    node = copy.deepcopy(base["proposal"])
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


def selftest():
    """The thirteen fixture gates of docs/16 §I CAD-07: Z1..Z13 in order, then SELFTEST PASS."""
    cases = common.read_json(CASES)
    decl = common.read_json(os.path.join(NOZZLE_DIR, "template.json"))
    sets = {s["name"]: s for s in cases["sets"]}

    def run(name):
        s = sets[name]
        return check(_fixture_proposal(cases, s), cases["briefs"][s["brief"]], decl, cases["template_sha"])

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
    rep = check(p, b1, decl, cases["template_sha"])
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
    assert expected_ids == set(REFUSAL_IDS), expected_ids ^ set(REFUSAL_IDS)
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
    locked1 = lock(r1, "reviewer")
    k1 = compile_checks(locked1, decl)
    assert k1["checks"] == _V1_CHECKS, "v1 compile differs from the hand-derived checks"
    assert k1["requirements_lock"] == locked1["lock_sha"]
    assert schema.errors(k1, "cad-checks/1") == []
    print("[ok] v1 compile: 12 checks byte-equal to the hand-derived literal, cad-checks/1 valid")

    # (Z7) the golden compiles: the bytes of (C11), recomputed now, equal the committed golden files.
    for name in valid_names:
        rep = run(name)
        locked = lock(rep, cases["approved_by"])
        checks = compile_checks(locked, decl)
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
        compile_checks(tampered, decl)
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
        return check(p, cases["briefs"][sets[name]["brief"]], decl, cases["template_sha"])

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
    rep = check(p, cases["briefs"]["B1"], decl, cases["template_sha"])
    assert [q["id"] for q in rep["questions"]] == ["Q-FLOW"], rep["questions"]
    op = rep["requirements"]["operating_point"]
    assert op["U_exit_m_s"] is None and op["Q_m3_s"] is None and op["mdot_kg_s"] is None
    print("[ok] questions: Q-CR, Q-D_i, Q-D_i+Q-CR and Q-FLOW asked, documents stay lockable-later only")

    # (Z10) operating-point, flow and row-shape errors are refusals by id, one each, at the exact check.
    def one_ref(mutate):
        p = _fixture_proposal(cases, sets["v1_nominal"])
        mutate(p)
        rep = check(p, cases["briefs"]["B1"], decl, cases["template_sha"])
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
                "hardness": hardness, "source": "assumed", "quote": None}

    p = _fixture_proposal(cases, sets["v1_nominal"])
    appended_row(p, assumed("watertight", "is_true", None, "-", "hard"))
    rep = check(p, cases["briefs"]["B1"], decl, cases["template_sha"])
    assert [(r["row"], r["id"]) for r in rep["refusals"]] == [(6, "REQ-DUP")], rep["refusals"]
    p = _fixture_proposal(cases, sets["v1_nominal"])
    appended_row(p, assumed("n_solids", ">=", 2, "-", "hard"))
    rep = check(p, cases["briefs"]["B1"], decl, cases["template_sha"])
    assert [(r["row"], r["id"]) for r in rep["refusals"]] == [(6, "REQ-CONFLICT")], rep["refusals"]
    # the feature names WHERE the row is measured, and the primitive measures a quantity at its catalogue where
    # whatever the feature says: rows 0 (feature contraction_start) and these (feature null) are one measurement,
    # so a second "==" on it is a duplicate and a ">=" beyond it a conflict (both were accepted when the key
    # held the raw feature)
    p = _fixture_proposal(cases, sets["v1_nominal"])
    r61 = assumed("inlet_diameter", "==", 61, "mm", "hard")
    r61["tol_abs"] = 0.001
    appended_row(p, r61)
    rep = check(p, cases["briefs"]["B1"], decl, cases["template_sha"])
    assert [(r["row"], r["id"]) for r in rep["refusals"]] == [(6, "REQ-DUP")], rep["refusals"]
    p = _fixture_proposal(cases, sets["v1_nominal"])
    appended_row(p, assumed("inlet_diameter", ">=", 61, "mm", "hard"))
    rep = check(p, cases["briefs"]["B1"], decl, cases["template_sha"])
    assert [(r["row"], r["id"]) for r in rep["refusals"]] == [(6, "REQ-CONFLICT")], rep["refusals"]
    p = _fixture_proposal(cases, sets["v1_nominal"])
    appended_row(p, assumed("contraction_length", ">=", 100, "mm", "soft"))
    rep = check(p, cases["briefs"]["B1"], decl, cases["template_sha"])
    assert [(r["row"], r["id"]) for r in rep["refusals"]] == [(6, "REQ-OUTSIDE")], rep["refusals"]
    p = _fixture_proposal(cases, sets["v1_nominal"])
    p["rows"][3] = assumed("min_wall_normal", ">=", 12, "mm", "soft")
    rep = check(p, cases["briefs"]["B1"], decl, cases["template_sha"])
    assert [(r["row"], r["id"]) for r in rep["refusals"]] == [(3, "REQ-OUTSIDE")], rep["refusals"]
    p = _fixture_proposal(cases, sets["v1_nominal"])
    p["rows"][1] = assumed("contraction_ratio", "==", 0.5, "-", "hard", tol_rel=1e-06)
    rep = check(p, cases["briefs"]["B1"], decl, cases["template_sha"])
    assert [(r["row"], r["id"]) for r in rep["refusals"]] == [(1, "REQ-OUTSIDE")], rep["refusals"]
    print("[ok] set passes: DUP vs SYS-WATERTIGHT, CONFLICT vs SYS-SOLID, DUP and CONFLICT across features, OUTSIDE from the analytic box")

    # (Z12) check is pure and deterministic; the CLI's verbs work in fresh processes.
    p0 = _fixture_proposal(cases, sets["v1_nominal"])
    b0 = copy.deepcopy(cases["briefs"]["B1"])
    d0 = copy.deepcopy(decl)
    sha0 = cases["template_sha"]
    ra = check(p0, b0, d0, sha0)
    assert p0 == _fixture_proposal(cases, sets["v1_nominal"]) and b0 == cases["briefs"]["B1"] and d0 == decl
    rb = check(p0, b0, d0, sha0)
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
    print("SELFTEST PASS")


def main(argv) -> int:
    """The CLI of docs/16 §I CAD-07: --selftest, check, lock, compile."""
    if argv and argv[0] == "--selftest":
        try:
            selftest()
        except Exception:
            traceback.print_exc()
            return 1
        return 0
    try:
        if len(argv) == 5 and argv[0] == "check":
            decl, sha = load_template(argv[3])
            report = check(common.read_json(argv[1]), common.read_json(argv[2]), decl, sha)
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
            write_canonical(argv[3], compile_checks(common.read_json(argv[1]), load_template(argv[2])[0]))
            return 0
    except ValueError as e:
        print("reqs: %s" % (e,), file=sys.stderr)
        return 1
    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
