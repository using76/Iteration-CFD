#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""
explain.py - deterministic explanations and the campaign summary (docs/15
§C's explain row, AM-15; docs/15 §F G-EXPL).

One fixed template per rule_id - what the rule decides and why it exists,
with no number in it - turns every DecisionRecord into a card, and every
autonomy-attempt/1 row into deterministic text: the decision, its trigger
against its threshold, the edits before and after, prediction against
observation, the observed mesh, the layer rows and the measurement the edit
moved.  Every number in the text is grounded in its rows (the docs/15 §C L5
lint, `ungrounded`).  `summarise` gives the failure and strict-failure rates
with Clopper-Pearson intervals per family and stratum.  `gate` writes the
G-EXPL report: it scans the package for rule ids without a template, validates
every row, checks that predictions and decisions precede their runs, and
compares three golden texts.  No mesher, no solver, no clock in any text.

    python tools/autonomy/explain.py --selftest
    python tools/autonomy/explain.py --gate
    python tools/autonomy/explain.py --write-golden
    python tools/autonomy/explain.py --rows R.jsonl [--records REC.json] [--geometry ID] [--json]
    python tools/autonomy/explain.py --summary --rows R.jsonl (--meta M.json | --manifest tuning|test) [--json]
    python tools/autonomy/explain.py --audit --rows R.jsonl [--records REC.json]
"""
from __future__ import annotations

import ast
import copy
import json
import math
import os
import random
import re
import statistics
import sys
import tempfile
from datetime import timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import schema  # noqa: E402
import remedies  # noqa: E402
import preflight  # noqa: E402
import rules  # noqa: E402

FIXTURE_DIR = os.path.join(HERE, "fixtures", "explain")
GOLDEN_DIR = os.path.join(FIXTURE_DIR, "golden")
REPORT_DIR = os.path.join(HERE, "explain")
EXPLAIN_SCHEMA = "autonomy-explain/1"
SUMMARY_SCHEMA = "autonomy-explain-summary/1"
AUDIT_SCHEMA = "autonomy-explain-audit/1"
GATE_SCHEMA = "autonomy-explain-gate/1"
ID_RE = re.compile(r"(PF|WL|R|RM|PR|OPT|LLM)-[A-Z0-9]+(-[A-Z0-9]+)*")
LAYER_OF_PREFIX = {"PF": "preflight", "WL": "preflight", "R": "rule", "RM": "remedy",
                   "PR": "prior", "OPT": "optimiser", "LLM": "llm"}
EXEMPT_IDS = {"PF-TEST": ("remedies.py",)}     # test scaffolding: allowed only in these files
TERMINAL_IDS = tuple(remedies.TERMINAL_ID.values())
TERMINAL_OF = {v: k for k, v in remedies.TERMINAL_ID.items()}
FLAG_ORDER = ("F1", "F2", "F3a", "F3b", "F3c", "F3d", "F3e", "F4", "F5")
NUM_RE = re.compile(r"(?<![A-Za-z0-9_.])\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
GOLDEN_IDS = ("box_sphere", "wing_a_L3", "D-1-002")


class ExplainError(ValueError):
    """A caller error (mixed geometries, a gap in the attempts, an unknown id)."""


TEMPLATES = {
    "WL-POINTER": {"layer": "preflight",
                   "title": "refuse an edit whose target is not a JSON pointer",
                   "because": "every edit names one config leaf by an RFC pointer"},
    "WL-FORBIDDEN": {"layer": "preflight",
                     "title": "refuse an edit to a knob outside the action space",
                     "because": "the quality gate, the permissive flag and the layer limiters are "
                                "never tuned: the loop improves meshes, it does not loosen the gate "
                                "that judges them"},
    "WL-UNLISTED": {"layer": "preflight",
                    "title": "refuse an edit to a knob that is not on the whitelist",
                    "because": "only knobs in the locked legal-range table may be changed"},
    "WL-TYPE": {"layer": "preflight",
                "title": "refuse an edit whose value has the wrong type",
                "because": "the knob table declares each knob's type, and the mesher's parser would "
                           "reject another"},
    "WL-RANGE": {"layer": "preflight",
                 "title": "refuse an edit whose value lies outside the knob's legal range",
                 "because": "the knob table bounds every knob, including the ones the mesher's "
                            "validator leaves unchecked"},
    "WL-FLAG": {"layer": "preflight",
                "title": "refuse a forbidden command-line flag",
                "because": "the permissive flag would let a mesh past the quality gate, and it is "
                           "never passed"},
    "WL-SHARP-FT0": {"layer": "preflight",
                     "title": "refuse switching the feature attraction off on a body with sharp edges",
                     "because": "the user decided that a body with sharp edges must have its edges "
                                "captured, and a zero feature tolerance stops the snap pulling any point "
                                "onto an edge; only the cell-plane path, whose edges lie on lattice "
                                "lines, keeps it at zero"},
    "PF-SURFACE": {"layer": "preflight",
                   "title": "check that every surface is closed and consistently wound",
                   "because": "an open or inverted surface cannot be meshed, so it is refused before "
                              "any run"},
    "PF-QUALITY": {"layer": "preflight",
                   "title": "check that the quality block equals the reference",
                   "because": "the gate that judges a mesh is fixed; a config that alters it is "
                              "refused"},
    "PF-FLAGS": {"layer": "preflight",
                 "title": "check the command line for forbidden flags",
                 "because": "no run may pass the permissive flag"},
    "PF-KNOBS": {"layer": "preflight",
                 "title": "check every settings leaf against the whitelist, its range and the "
                          "mesher's parser",
                 "because": "a config the mesher would reject, or a knob outside the table, is "
                            "refused before any run"},
    "PF-PATCH": {"layer": "preflight",
                 "title": "check that every named patch is one of the surface's own",
                 "because": "a layer or band on a patch the surface does not have is refused by the "
                            "mesher"},
    "PF-CONFIG": {"layer": "preflight",
                  "title": "check that the mesher's own parser and validator accept the config",
                  "because": "the mirror of the parser refuses what the mesher would refuse, on the "
                             "same field"},
    "PF-NONORTH": {"layer": "preflight",
                   "title": "check that the non-orthogonality ceiling clears the octree floor",
                   "because": "a two-to-one octree transition sets a non-orthogonality floor that no "
                              "refined mesh can go below"},
    "PF-YPLUS": {"layer": "preflight",
                 "title": "check that the a priori y+ window is non-empty and reachable",
                 "because": "the first layer must meet y+ at most one while the stack fits under the "
                            "cell limiter, at a level the octree can reach"},
    "PF-THIN": {"layer": "preflight",
                "title": "check the first layer against the thickness-ratio gate at the wall edge",
                "because": "a first layer too thin for its wall cells makes every layer cell fail the "
                           "quality gate, so the mesher drops them"},
    "PF-DOMAIN": {"layer": "preflight",
                  "title": "check that the domain holds the surface with margin",
                  "because": "a surface that touches the domain boundary without spanning it cannot "
                             "be meshed as a body"},
    "PF-BUDGET": {"layer": "preflight",
                  "title": "check the octree probe's leaf count against the cell budget",
                  "because": "a config whose octree alone exceeds the budget would fail on cost"},
    "R-YP": {"layer": "rule",
             "title": "set the first layer thickness from the a priori wall unit",
             "because": "the flat-plate skin-friction correlation gives the friction velocity, and "
                        "the first cell height follows from y+ at most one"},
    "R-DOM": {"layer": "rule",
              "title": "set the base cell size and the domain extent from the reference length",
              "because": "the domain keeps fixed multiples of the reference length upstream, "
                         "downstream and on each side"},
    "R-PLANE": {"layer": "rule",
                "title": "put a commensurate body's faces on cell planes",
                "because": "when the faces lie on the lattice the walls need no snapping, and the "
                           "layers are delivered"},
    "R-WIN": {"layer": "rule",
              "title": "pick the wall level and the growth inside the feasible window",
              "because": "the wall cell must be thick enough for the stack under the limiter and thin "
                         "enough for the thickness-ratio gate"},
    "R-CURV": {"layer": "rule",
               "title": "refine the wall where the curvature radius is small",
               "because": "a wall cell larger than a fraction of the curvature radius cannot follow "
                          "the surface"},
    "R-GAP": {"layer": "rule",
              "title": "refine the wall where two surfaces face each other across a small gap",
              "because": "a gap needs several cells across it to be resolved"},
    "R-FEAT": {"layer": "rule",
               "title": "raise the feature level where the body has sharp edges",
               "because": "sharp edges need finer cells to be captured by the snap"},
    "R-BUDGET": {"layer": "rule",
                 "title": "coarsen the ladder until the predicted cells fit the budget",
                 "because": "far-field bands are coarsened first and the wall last, so the cells fit "
                            "with the least loss at the wall"},
    "RM-BUDGET-FAR": {"layer": "remedy",
                      "title": "halve the far-field band distances",
                      "because": "a run over the cell budget or out of time sheds far-field cells "
                                 "first"},
    "RM-BUDGET-FEAT": {"layer": "remedy",
                       "title": "drop the feature bump to the wall level",
                       "because": "after the far field, the feature refinement is the next cost to "
                                  "shed"},
    "RM-BUDGET-WALL": {"layer": "remedy",
                       "title": "coarsen the whole refinement ladder one level, not below the y+ floor",
                       "because": "the wall level is the last cost to shed, and never below the level "
                                  "that y+ needs"},
    "RM-TOPO-REFINE": {"layer": "remedy",
                       "title": "refine the whole refinement ladder one level",
                       "because": "a missing wall patch or a split fluid region means the lattice is "
                                  "too coarse for the body's topology"},
    "RM-SNAP-WALL": {"layer": "remedy",
                     "title": "coarsen the whole refinement ladder one level, not below the y+ floor",
                     "because": "the pilot sweep found one level coarser the only snap-clean change "
                                "with the feature attraction on"},
    "RM-SNAP-FT": {"layer": "remedy",
                   "title": "switch the feature attraction off",
                   "because": "the pilot sweep found that it unpins the snap on every feature-bearing "
                              "body, at the cost of the edges not being captured"},
    "RM-SNAP-REFINE": {"layer": "remedy",
                       "title": "refine the whole refinement ladder one level",
                       "because": "the snapped surface misses area, so the lattice is too coarse for "
                                  "the body"},
    "RM-T1-RAISE": {"layer": "remedy",
                    "title": "raise the first layer to its y+ bound",
                    "because": "the first layer is thinner than the thickness-ratio gate allows, and "
                               "a thicker one still meets y+"},
    "RM-T1-REFINE": {"layer": "remedy",
                     "title": "refine the whole refinement ladder one level",
                     "because": "smaller wall cells let the same first layer pass the thickness-ratio "
                                "gate"},
    "RM-PLANE": {"layer": "remedy",
                 "title": "put the body on cell planes with the plane rule",
                 "because": "a layer drop on a snapped wall goes to the plane rule when the body is "
                            "commensurate"},
    "RM-PLANE-FINER": {"layer": "remedy",
                       "title": "take the next lattice divisor on the plane path",
                       "because": "a finer divisor shrinks the wall cell so that the stack meets the "
                                  "corner bound"},
    "RM-LAYER-FIT": {"layer": "remedy",
                     "title": "fit the stack under the cell limiter",
                     "because": "the stack is trimmed by the limiter, so the growth or the first "
                                "layer is lowered until it fits"},
    "RM-PASS": {"layer": "remedy",
                "title": "end the geometry: the attempt passes",
                "because": "no failure flag is set and no requested layer patch is dropped or "
                           "trimmed"},
    "RM-CAPABILITY-LIMITED": {"layer": "remedy",
                              "title": "end the geometry: the layers are limited by the mesher's "
                                       "capability",
                              "because": "the mesher drops layers on a snapped wall by name, so no "
                                         "further trial is spent on them"},
    "RM-EXHAUSTED": {"layer": "remedy",
                     "title": "end the geometry: the attempts or the remedies are spent",
                     "because": "each remedy fires at most twice, no config is revisited, and the "
                                "attempt count is capped"},
    "RM-NO-REMEDY": {"layer": "remedy",
                     "title": "end the geometry: the failure is outside the remedy table",
                     "because": "a surface defect, a config the mesher rejects, a harness fault or a "
                                "quality gate is not something a bounded edit may change"},
    "PR-KNN": {"layer": "prior",
               "title": "warm-start attempt one from the nearest passing tuning geometries",
               "because": "similar geometries needed the same refinement and snap remedies to pass, "
                          "so attempt one starts from their path"},
    "PR-KEEP": {"layer": "prior",
                "title": "keep the rules' attempt: the nearest passing geometries needed no remedy",
                "because": "the neighbours passed with the setup rules' own config"},
    "PR-FAR": {"layer": "prior",
               "title": "abstain: no passing tuning geometry is near enough",
               "because": "a neighbour beyond the locked distance says nothing about this geometry"},
    "PR-NOEDIT": {"layer": "prior",
                  "title": "abstain: the neighbours' remedies change nothing here",
                  "because": "the transferred remedies are refused by their own guards on this "
                             "config, or the body is on the plane path"},
    "PR-DISABLED": {"layer": "prior",
                    "title": "abstain: the prior ships disabled",
                    "because": "the prior did not earn its place on the tuning split, so the setup "
                               "rules decide attempt one"},
    "OPT-PICK": {"layer": "optimiser",
                 "title": "propose the surrogate's pick from the Sobol pool",
                 "because": "the remedies are spent, and the surrogate predicts this config passes "
                            "with the most boundary-layer capture for its cells"},
    "OPT-NOFEAS": {"layer": "optimiser",
                   "title": "abstain: no pool config is predicted to pass within the budget",
                   "because": "a proposal the surrogate expects to fail would spend an attempt "
                              "for nothing"},
    "OPT-PLANE": {"layer": "optimiser",
                  "title": "abstain: the body is on the plane path",
                  "because": "the plane rule owns the refinement and snap knobs of a commensurate "
                             "body"},
    "OPT-DISABLED": {"layer": "optimiser",
                     "title": "abstain: the optimiser ships disabled",
                     "because": "the optimiser did not earn its place on the tuning split, so the "
                                "remedies' terminal stands"},
}


# --- the formatting of one value (docs/15 §C explain row) --------------------


def fmt(v) -> str:
    """remedies._fmt re-implemented: float %.6g, list joined, None absent, else json."""
    if isinstance(v, float):
        return "%.6g" % v
    if isinstance(v, list):
        return "[" + ", ".join(fmt(x) for x in v) + "]"
    if v is None:
        return "absent"
    return json.dumps(v)


def trigger_text(t) -> str:
    """The trigger as one line: observable = value op threshold (source)."""
    if t is None:
        return "no trigger"
    return "%s = %s %s %s (%s)" % (t["observable"], fmt(t["value"]), t["op"],
                                   fmt(t["threshold"]), t["source"])


def edits_text(edits) -> str:
    """The edits as one line: pointer from -> to, in order; none when empty."""
    if not edits:
        return "none"
    return "; ".join("%s %s -> %s" % (e["pointer"], fmt(e["from"]), fmt(e["to"]))
                     for e in edits)


def template(rule_id: str) -> dict:
    """The fixed template of one rule id; ExplainError when it has none."""
    tp = TEMPLATES.get(rule_id)
    if tp is None:
        raise ExplainError("no template for rule id %r (TEMPLATES has %d ids)"
                           % (rule_id, len(TEMPLATES)))
    return tp


# --- the static scan (docs/15 §F G-EXPL: every rule_id has a template) -------


def static_scan(extra_sources=()) -> dict:
    """Every rule id string in the package sources has a template (or is exempt)."""
    base = []
    for d in (HERE, os.path.join(HERE, "corpus")):
        for name in sorted(os.listdir(d)):
            if name.endswith(".py"):
                p = os.path.join(d, name)
                if os.path.abspath(p) != os.path.abspath(__file__):
                    base.append(p)
    files = sorted(base) + list(extra_sources)
    ids = {}
    for path in files:
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        base_name = os.path.basename(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if ID_RE.fullmatch(node.value):
                    ids.setdefault(node.value, [])
                    if base_name not in ids[node.value]:
                        ids[node.value].append(base_name)
    for rid in ids:
        ids[rid] = sorted(ids[rid])
    missing = sorted(set(ids) - set(TEMPLATES) - set(EXEMPT_IDS))
    dead = sorted(set(TEMPLATES) - set(ids))
    exempt_bad = []
    for rid, allowed_files in EXEMPT_IDS.items():
        for fname in ids.get(rid, []):
            if fname not in allowed_files:
                exempt_bad.append("%s in %s" % (rid, fname))
    table = (list(preflight.REFUSAL_IDS) + [c[1] for c in preflight.CHECKS]
             + list(rules.RULES) + [r["id"] for r in remedies.REMEDIES]
             + list(TERMINAL_IDS))
    tables_missing = sorted(set(table) - set(TEMPLATES))
    return {"ids": ids, "files": len(files), "missing": missing, "dead": dead,
            "exempt_bad": sorted(exempt_bad), "tables_missing": tables_missing,
            "ok": not (missing or dead or exempt_bad or tables_missing)}


# --- the card (docs/15 §C: the studio's AttemptCard feeds on this) -----------


def card(rec: dict) -> dict:
    """One DecisionRecord rendered against its rule's template."""
    errs = schema.errors(rec, "DecisionRecord")
    if errs:
        raise ExplainError("not a DecisionRecord: %s" % errs[0])
    tp = template(rec["rule_id"])
    if tp["layer"] != rec["layer"]:
        raise ExplainError("rule id %s is a %s template but the record's layer is %s"
                           % (rec["rule_id"], tp["layer"], rec["layer"]))
    line = ("%s [%s %s]: %s; because %s. %s (cite: %s)"
            % (rec["rule_id"], rec["layer"], rec["verdict"], tp["title"], tp["because"],
               rec["message"], rec["cite"]))
    return {"rule_id": rec["rule_id"], "layer": rec["layer"], "verdict": rec["verdict"],
            "title": tp["title"], "because": tp["because"], "trigger": rec["trigger"],
            "trigger_text": trigger_text(rec["trigger"]) if rec["trigger"] else None,
            "edits": rec["edits"], "inputs": rec["inputs"], "message": rec["message"],
            "cite": rec["cite"], "t": rec["t"], "line": line}


# --- the observables (docs/15 §C L2: the next card shows the after-value) ----

_PATCH_FIELD_RE = re.compile(r"patches\[(.+)\]\.(\w+)")
_BARE_KEY_RE = re.compile(r"\w+")


def resolve(outcome: dict, observable: str) -> tuple:
    """(found, value) of one observable in a score.py outcome block."""
    if not isinstance(observable, str) or not observable.startswith("outcome."):
        return (False, None)
    r = observable[len("outcome."):]
    if r.startswith("flags."):
        flags = outcome.get("flags")
        f = r[len("flags."):]
        if isinstance(flags, dict) and f in flags:
            return (True, flags[f])
        return (False, None)
    m = _PATCH_FIELD_RE.fullmatch(r)
    if m:
        for p in outcome.get("patches", []):
            if p.get("name") == m.group(1):
                if m.group(2) in p:
                    return (True, p[m.group(2)])
                return (False, None)
        return (False, None)
    if _BARE_KEY_RE.fullmatch(r) and r in outcome:
        return (True, outcome[r])
    return (False, None)


# --- the row validator (G-EXPL: 100 % of rows validate) ----------------------


def validate_row(row: dict, gates=None, knobs=None) -> list:
    """Schema + S1..S12 + every rule id templated; empty list = valid."""
    gates = gates or schema.load_gates()
    knobs = knobs or schema.load_knobs()
    errs = schema.errors(row, "AttemptRow")
    if errs:
        return errs
    errs = schema.check_attempt(row, gates, knobs)
    rid = row["rule_id"]
    if rid is not None and rid not in TEMPLATES:
        errs.append("explain: rule_id: no template for %s" % rid)
    for i, r in enumerate(row["constraint_refusals"]):
        if r["rule_id"] not in TEMPLATES:
            errs.append("explain: rule_id: no template for %s" % r["rule_id"])
    return errs


# --- the per-geometry text and JSON (docs/15 §C explain row) -----------------


def explain_geometry(rows: list, records=None) -> dict:
    """One geometry's rows and tagged records as deterministic text and JSON."""
    if not rows:
        raise ExplainError("no rows to explain")
    rows = sorted(rows, key=lambda r: r["attempt"])
    gid = rows[0]["geometry_id"]
    campaign = rows[0]["campaign_id"]
    split_name = rows[0]["split"]
    for r in rows:
        if r["geometry_id"] != gid or r["campaign_id"] != campaign:
            raise ExplainError("mixed geometry_id or campaign_id in one geometry: %s / %s"
                               % (r["geometry_id"], r["campaign_id"]))
    attempts = [r["attempt"] for r in rows]
    if attempts != list(range(1, len(rows) + 1)):
        raise ExplainError("the attempts of %s are %s, not 1..%d"
                           % (gid, attempts, len(rows)))
    tags = list(records) if records else []
    for tag in tags:
        if tag["attempt"] not in attempts:
            raise ExplainError("a record is tagged attempt %s, which %s has no row for"
                               % (tag["attempt"], gid))
    by_attempt = {a: {"pre": [], "end": []} for a in attempts}
    for tag in tags:
        c = card(tag["record"])
        tgt = by_attempt[tag["attempt"]]
        (tgt["end"] if c["rule_id"] in TERMINAL_IDS else tgt["pre"]).append(c)
    terminal = "none recorded"
    if by_attempt[rows[-1]["attempt"]]["end"]:
        terminal = TERMINAL_OF[by_attempt[rows[-1]["attempt"]]["end"][-1]["rule_id"]]
    lines = ["# %s: %d attempt(s), split %s, campaign %s, terminal %s"
             % (gid, len(rows), split_name, campaign, terminal)]
    out_attempts = []
    for r in rows:
        a = r["attempt"]
        oc = r["outcome"]
        rid = r["rule_id"]
        pre = by_attempt[a]["pre"]
        end = by_attempt[a]["end"]
        refused = [card(x) for x in r["constraint_refusals"]]
        if rid:
            decided = template(rid)["title"]
        elif r["decided_by"] == "default":
            decided = "the config as given: no rule chose it"
        else:
            decided = "the config the %s layer proposed" % r["decided_by"]
        pred = r["prediction"]
        if pred is None:
            pred_text = "none"
        else:
            pred_text = ("p_fail %s +- %s, BLC_8 %s, log10 cells %s, at %s; observed verdict %s, "
                         "BLC_8 %s, cells %s"
                         % (fmt(pred["p_fail"]), fmt(pred["p_fail_std"]), fmt(pred["blc8_a_priori"]),
                            fmt(pred["log_cells"]), pred["t_predicted"], oc["verdict"],
                            fmt(oc["blc8_a_priori"]), fmt(oc["n_cells"])))
        flags_true = [f for f in FLAG_ORDER if oc["flags"].get(f) is True]
        obs_text = ("verdict %s, failure class %s, flags %s, cells %s, pinned %s, p99/h_f %s, "
                    "max/h_f %s, BLC_8 %s, BLC_full %s, %s s"
                    % (oc["verdict"], oc["failure_class"] or "none", " ".join(flags_true) or "none",
                       fmt(oc["n_cells"]), fmt(oc["pinned_frac"]), fmt(oc["p99_over_hf"]),
                       fmt(oc["max_over_hf"]), fmt(oc["blc8_a_priori"]),
                       fmt(oc["blc_full_a_priori"]), fmt(oc["seconds"])))
        if not oc["patches"]:
            layers_text = "no patch rows"
        else:
            parts = []
            for p in oc["patches"]:
                if p["delivered"]:
                    status = "delivered"
                elif not p["requested"]:
                    status = "not requested"
                elif p.get("layer_class"):
                    status = "not delivered (%s)" % p["layer_class"]
                else:
                    status = "not delivered"
                s = "%s %d layers, full %s, %s" % (p["name"], p["n_layers"],
                                                   fmt(p["full_area_frac"]), status)
                if p.get("capability_limited") is True:
                    s += ", capability-limited"
                parts.append(s)
            layers_text = "; ".join(parts)
        moved = None
        moved_text = None
        if a >= 2 and r["decided_by"] == "remedy" and r["trigger"] is not None:
            found, after = resolve(oc, r["trigger"]["observable"])
            if found:
                moved = {"observable": r["trigger"]["observable"],
                         "before": r["trigger"]["value"], "after": after,
                         "from_attempt": a - 1, "to_attempt": a,
                         "unchanged": r["trigger"]["value"] == after}
                moved_text = ("%s %s -> %s (attempt %d -> %d)"
                              % (r["trigger"]["observable"], fmt(r["trigger"]["value"]),
                                 fmt(after), a - 1, a))
                if moved["unchanged"]:
                    moved_text += "; unchanged"
        lines.append("")
        lines.append("## attempt %d: %s%s, stage focus %s"
                     % (a, r["decided_by"], (" " + rid) if rid else "",
                        r["stage_focus"] or "none"))
        lines.append("decided: %s" % decided)
        lines.append("why: %s" % trigger_text(r["trigger"]))
        lines.append("edits: %s" % edits_text(r["config_delta"]))
        for c in refused:
            lines.append("refused: %s" % c["line"])
        for c in pre:
            lines.append("record %s" % c["line"])
        lines.append("predicted: %s" % pred_text)
        lines.append("observed: %s" % obs_text)
        lines.append("layers: %s" % layers_text)
        if moved_text is not None:
            lines.append("moved: %s" % moved_text)
        for c in end:
            lines.append("end %s" % c["line"])
        out_attempts.append({"attempt": a, "decided_by": r["decided_by"], "rule_id": rid,
                             "stage_focus": r["stage_focus"], "decided": decided,
                             "why": r["trigger"], "why_text": trigger_text(r["trigger"]),
                             "edits": r["config_delta"],
                             "edits_text": edits_text(r["config_delta"]),
                             "refused": refused, "records": pre, "prediction": pred,
                             "prediction_text": pred_text,
                             "observed": {"verdict": oc["verdict"],
                                          "failure_class": oc["failure_class"],
                                          "flags_true": flags_true, "n_cells": oc["n_cells"],
                                          "pinned_frac": oc["pinned_frac"],
                                          "p99_over_hf": oc["p99_over_hf"],
                                          "max_over_hf": oc["max_over_hf"],
                                          "blc8_a_priori": oc["blc8_a_priori"],
                                          "blc_full_a_priori": oc["blc_full_a_priori"],
                                          "seconds": oc["seconds"]},
                             "observed_text": obs_text, "layers_text": layers_text,
                             "moved": moved, "moved_text": moved_text, "end": end})
    return {"schema": EXPLAIN_SCHEMA, "geometry_id": gid, "split": split_name,
            "campaign_id": campaign, "n_attempts": len(rows), "terminal": terminal,
            "attempts": out_attempts, "text": "\n".join(lines) + "\n"}


# --- the audit (G-EXPL on any set of rows; AM-16 reuses it) ------------------


def audit(rows: list, records=None, gates=None, knobs=None) -> dict:
    """Validate every row and check prediction, record and trigger order."""
    gates = gates or schema.load_gates()
    knobs = knobs or schema.load_knobs()
    records = records or {}
    invalid = []
    rule_ids = set()
    n_predictions = 0
    prediction_late = []
    time_reversed = []
    for i, r in enumerate(rows):
        if r["rule_id"]:
            rule_ids.add(r["rule_id"])
        for x in r["constraint_refusals"]:
            rule_ids.add(x["rule_id"])
        errs = validate_row(r, gates, knobs)
        if errs:
            invalid.append({"index": i, "geometry_id": r["geometry_id"],
                            "attempt": r["attempt"], "errors": errs})
        if r["prediction"] is not None:
            n_predictions += 1
            if schema._parse_iso(r["prediction"]["t_predicted"]) >= schema._parse_iso(r["t_start"]):
                prediction_late.append({"geometry_id": r["geometry_id"],
                                        "attempt": r["attempt"]})
        if schema._parse_iso(r["t_end"]) < schema._parse_iso(r["t_start"]):
            time_reversed.append({"geometry_id": r["geometry_id"], "attempt": r["attempt"]})
    by_geom = {}
    for r in rows:
        by_geom.setdefault(r["geometry_id"], {})[r["attempt"]] = r
    checked = 0
    bad = []
    for gid2 in sorted(records):
        for tag in records[gid2]:
            rec = tag["record"]
            rule_ids.add(rec["rule_id"])
            row = by_geom.get(gid2, {}).get(tag["attempt"])
            if row is None:
                bad.append({"geometry_id": gid2, "attempt": tag["attempt"],
                            "rule_id": rec["rule_id"], "why": "no row"})
                continue
            checked += 1
            if rec["rule_id"] in TERMINAL_IDS:
                if schema._parse_iso(rec["t"]) < schema._parse_iso(row["t_end"]):
                    bad.append({"geometry_id": gid2, "attempt": tag["attempt"],
                                "rule_id": rec["rule_id"],
                                "why": "a terminal record before its run ended"})
            elif schema._parse_iso(rec["t"]) > schema._parse_iso(row["t_start"]):
                bad.append({"geometry_id": gid2, "attempt": tag["attempt"],
                            "rule_id": rec["rule_id"],
                            "why": "a decision record after its run started"})
    trigger_checked = 0
    trigger_mismatch = []
    for r in rows:
        if r["decided_by"] == "remedy" and r["attempt"] >= 2 and r["trigger"] is not None:
            trigger_checked += 1
            prev = by_geom.get(r["geometry_id"], {}).get(r["attempt"] - 1)
            if prev is None:
                trigger_mismatch.append({"geometry_id": r["geometry_id"],
                                         "attempt": r["attempt"],
                                         "observable": r["trigger"]["observable"],
                                         "trigger": r["trigger"]["value"], "previous": None})
                continue
            found, v = resolve(prev["outcome"], r["trigger"]["observable"])
            if not found or json.dumps(v) != json.dumps(r["trigger"]["value"]):
                trigger_mismatch.append({"geometry_id": r["geometry_id"],
                                         "attempt": r["attempt"],
                                         "observable": r["trigger"]["observable"],
                                         "trigger": r["trigger"]["value"], "previous": v})
    untemplated = sorted(rid for rid in rule_ids if rid not in TEMPLATES)
    return {"schema": AUDIT_SCHEMA, "n_rows": len(rows),
            "n_valid": len(rows) - len(invalid), "invalid": invalid,
            "rule_ids": sorted(rule_ids), "untemplated": untemplated,
            "n_predictions": n_predictions, "prediction_late": prediction_late,
            "time_reversed": time_reversed,
            "record_order": {"checked": checked, "bad": bad},
            "trigger_checked": trigger_checked, "trigger_mismatch": trigger_mismatch,
            "ok": (not invalid and not untemplated and not prediction_late
                   and not time_reversed and not bad and not trigger_mismatch)}


# --- the exact binomial interval (Clopper-Pearson 1934) ----------------------


def clopper_pearson(x: int, n: int, alpha: float = 0.05) -> tuple:
    """The two-sided exact (Clopper-Pearson) interval for x of n, as (lo, hi)."""
    if not (isinstance(x, int) and isinstance(n, int)) or not (0 <= x <= n):
        raise ExplainError("clopper_pearson needs ints 0 <= x <= n, got x=%r n=%r" % (x, n))
    if n == 0:
        return (None, None)
    from scipy.stats import beta
    lo = 0.0 if x == 0 else float(beta.ppf(alpha / 2, x, n - x + 1))
    hi = 1.0 if x == n else float(beta.ppf(1 - alpha / 2, x + 1, n - x))
    return (lo, hi)


# --- the campaign summary (docs/15 §C: remember, explain; AM-16 reads it) ----


def meta_from_manifest(manifest_rows: list) -> dict:
    """{geometry_id: {family, stratum}} from a split.load manifest."""
    return {r["geometry_id"]: {"family": r["family"], "stratum": r["stratum"]}
            for r in manifest_rows}


def summarise(rows: list, meta: dict, alpha: float = 0.05) -> dict:
    """Final attempt per geometry; MFR and strict rates with CP intervals."""
    if not rows:
        raise ExplainError("no rows to summarise")
    campaign = rows[0]["campaign_id"]
    finals = {}
    for r in rows:
        if r["campaign_id"] != campaign:
            raise ExplainError("mixed campaigns in one summary: %s and %s"
                               % (campaign, r["campaign_id"]))
        gid = r["geometry_id"]
        if gid not in meta:
            raise ExplainError("no meta entry for geometry %s" % gid)
        if gid not in finals or r["attempt"] > finals[gid]["attempt"]:
            finals[gid] = r
    geo = {}
    for gid, r in finals.items():
        oc = r["outcome"]
        geo[gid] = {"family": meta[gid]["family"], "stratum": meta[gid]["stratum"],
                    "fail": oc["verdict"] == "fail", "strict": oc["strict_failure"] is True,
                    "blc8": oc["blc8_a_priori"], "blcf": oc["blc_full_a_priori"],
                    "cells": oc["n_cells"], "attempts": r["attempt"],
                    "cap": any(p.get("capability_limited") is True for p in oc["patches"])}
    keys = sorted(set((g["family"], g["stratum"]) for g in geo.values()))
    groups = []
    for family in sorted(set(k[0] for k in keys)):
        strata = sorted(k[1] for k in keys if k[0] == family)
        for sel in [(family, s) for s in strata] + [(family, "all")]:
            members = [g for g in geo.values()
                       if sel[1] == "all" and g["family"] == sel[0]
                       or sel[1] != "all" and (g["family"], g["stratum"]) == sel]
            groups.append(_group(sel[0], sel[1], members, alpha))
    groups.append(_group("all", "all", list(geo.values()), alpha))
    return {"schema": SUMMARY_SCHEMA, "campaign_id": campaign, "alpha": alpha,
            "confidence": 1 - alpha, "n_rows": len(rows), "n_geometries": len(geo),
            "groups": groups}


def _group(family, stratum, members, alpha):
    n = len(members)
    fail = sum(1 for g in members if g["fail"])
    strict = sum(1 for g in members if g["strict"])
    cells = [g["cells"] for g in members if g["cells"] is not None]
    return {"family": family, "stratum": stratum, "n": n, "fail": fail, "mfr": fail / n,
            "mfr_ci": list(clopper_pearson(fail, n, alpha)), "strict": strict,
            "strict_rate": strict / n, "strict_ci": list(clopper_pearson(strict, n, alpha)),
            "blc8_mean": statistics.fmean(g["blc8"] for g in members),
            "blc_full_mean": statistics.fmean(g["blcf"] for g in members),
            "cells_median": statistics.median(cells) if cells else None,
            "attempts_mean": statistics.fmean(g["attempts"] for g in members),
            "capability_limited": sum(1 for g in members if g["cap"])}


def summary_text(summary: dict) -> str:
    """The summary as deterministic text, one line per group."""
    lines = ["# campaign %s: %d geometries, %d rows, Clopper-Pearson intervals at confidence %s"
             % (summary["campaign_id"], summary["n_geometries"], summary["n_rows"],
                fmt(summary["confidence"]))]
    for g in summary["groups"]:
        lines.append("%s/%s: n %d, MFR %s [%s, %s], strict %s [%s, %s], BLC_8 %s, BLC_full %s, "
                     "median cells %s, mean attempts %s, capability-limited %d"
                     % (g["family"], g["stratum"], g["n"], fmt(g["mfr"]), fmt(g["mfr_ci"][0]),
                        fmt(g["mfr_ci"][1]), fmt(g["strict_rate"]), fmt(g["strict_ci"][0]),
                        fmt(g["strict_ci"][1]), fmt(g["blc8_mean"]), fmt(g["blc_full_mean"]),
                        fmt(g["cells_median"]), fmt(g["attempts_mean"]), g["capability_limited"]))
    return "\n".join(lines) + "\n"


# --- the grounding lint (docs/15 §C L5: every number is in its source) -------


def ungrounded(text: str, sources: list) -> list:
    """The numbers in `text` that no leaf of any source contains."""
    allowed = set()

    def walk(v):
        if isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        elif isinstance(v, bool):
            pass
        elif isinstance(v, (int, float)):
            for s in (fmt(v), json.dumps(v)):
                allowed.update(NUM_RE.findall(s))
        elif isinstance(v, str):
            for s in (v, json.dumps(v)):
                allowed.update(NUM_RE.findall(s))

    for src in sources:
        walk(src)
    return sorted(set(NUM_RE.findall(text)) - allowed)


# --- the fixtures and the golden files (G-EXPL part 4) -----------------------


def load_fixtures() -> dict:
    """The supervisor's six rows, tagged records, records by id, meta, labels."""
    def _json(name):
        with open(os.path.join(FIXTURE_DIR, name), encoding="utf-8") as f:
            return json.load(f)

    with open(os.path.join(FIXTURE_DIR, "rows.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(ln) for ln in f.read().splitlines() if ln.strip()]
    return {"rows": rows, "records": _json("records.json")["records"],
            "records_by_id": _json("records_by_id.json")["records"],
            "meta": _json("meta.json")["meta"], "expect": _json("expect.json")}


def _fixture_geometry(gid: str, fx: dict) -> dict:
    """One fixture geometry through explain_geometry."""
    return explain_geometry([r for r in fx["rows"] if r["geometry_id"] == gid],
                            fx["records"].get(gid))


def _fixture_summary(fx: dict) -> dict:
    """The fixture campaign summary."""
    return summarise(fx["rows"], fx["meta"])


def _write_text(path: str, s: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)


def _dump(obj) -> str:
    return json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def write_golden() -> int:
    """Write the eight golden files; the only writer they ever get."""
    fx = load_fixtures()
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    for gid in GOLDEN_IDS:
        obj = _fixture_geometry(gid, fx)
        _write_text(os.path.join(GOLDEN_DIR, gid + ".txt"), obj["text"])
        _write_text(os.path.join(GOLDEN_DIR, gid + ".json"), _dump(obj))
        print(os.path.join(GOLDEN_DIR, gid + ".txt"))
        print(os.path.join(GOLDEN_DIR, gid + ".json"))
    summ = _fixture_summary(fx)
    _write_text(os.path.join(GOLDEN_DIR, "summary.txt"), summary_text(summ))
    _write_text(os.path.join(GOLDEN_DIR, "summary.json"), _dump(summ))
    print(os.path.join(GOLDEN_DIR, "summary.txt"))
    print(os.path.join(GOLDEN_DIR, "summary.json"))
    return 0


def _read_text(path: str) -> str:
    with open(path, encoding="utf-8", newline="") as f:
        return f.read()


def _golden_report(fx: dict) -> dict:
    """The golden comparison and the expect labels, as data (the gate reads it)."""
    rep = {"geometries": {}, "summary": {}, "ok": True}
    for gid in GOLDEN_IDS:
        obj = _fixture_geometry(gid, fx)
        text = obj["text"]
        exp = fx["expect"]["golden"][gid]
        src = ([r for r in fx["rows"] if r["geometry_id"] == gid]
               + [t["record"] for t in fx["records"].get(gid, [])]
               + [len([r for r in fx["rows"] if r["geometry_id"] == gid])])
        lines = text.split("\n")
        counts = {k: sum(1 for ln in lines if ln.startswith(k)) for k in exp["counts"]}
        labels = sum(1 for s in exp["must_contain"] if s in text)
        labels += sum(1 for s in exp["must_not_contain"] if s not in text)
        labels += sum(1 for k, n in exp["counts"].items() if counts[k] == n)
        checks = {"text": text == _read_text(os.path.join(GOLDEN_DIR, gid + ".txt")),
                  "json": _dump(obj) == _read_text(os.path.join(GOLDEN_DIR, gid + ".json")),
                  "labels": labels,
                  "n_labels": (len(exp["must_contain"]) + len(exp["must_not_contain"])
                               + len(exp["counts"])),
                  "ungrounded": len(ungrounded(text, src))}
        checks["ok"] = (checks["text"] and checks["json"]
                        and checks["labels"] == checks["n_labels"]
                        and checks["ungrounded"] == 0)
        rep["geometries"][gid] = {"ok": checks["ok"], "n_attempts": obj["n_attempts"],
                                  "terminal": obj["terminal"],
                                  "rule_ids": sorted({r["rule_id"] for r in fx["rows"]
                                                      if r["geometry_id"] == gid
                                                      and r["rule_id"]}),
                                  **checks}
        rep["ok"] = rep["ok"] and checks["ok"]
    summ = _fixture_summary(fx)
    stxt = summary_text(summ)
    sexp = fx["expect"]["summary"]
    schecks = {"text": stxt == _read_text(os.path.join(GOLDEN_DIR, "summary.txt")),
               "json": _dump(summ) == _read_text(os.path.join(GOLDEN_DIR, "summary.json")),
               "labels": sum(1 for s in sexp["must_contain"] if s in stxt),
               "n_labels": len(sexp["must_contain"]),
               "ungrounded": len(ungrounded(stxt, [summ]))}
    schecks["ok"] = (schecks["text"] and schecks["json"]
                     and schecks["labels"] == schecks["n_labels"]
                     and schecks["ungrounded"] == 0)
    rep["summary"] = {"ok": schecks["ok"], **schecks}
    rep["ok"] = rep["ok"] and schecks["ok"]
    return rep


# --- the selftest (G-EXPL as code, 13 groups) --------------------------------

_DECISION_ID_RE = re.compile(r"^[A-Z][A-Z0-9]*(-[A-Z0-9]+)+$")


def _st_binom_cdf(x: int, n: int, p: float) -> float:
    total = 0.0
    for k in range(x + 1):
        total += math.comb(n, k) * (p ** k) * ((1.0 - p) ** (n - k))
    return total


def _g1_templates():
    assert len(TEMPLATES) == 51, len(TEMPLATES)
    counts = {}
    for rid, tp in TEMPLATES.items():
        assert _DECISION_ID_RE.fullmatch(rid) and ID_RE.fullmatch(rid), rid
        prefix = rid.split("-")[0]
        assert tp["layer"] == LAYER_OF_PREFIX[prefix], rid
        assert len(tp) == 3, rid
        assert re.search(r"\d", tp["title"] + tp["because"]) is None, rid
        counts[prefix] = counts.get(prefix, 0) + 1
    assert counts == {"PF": 11, "WL": 7, "R": 8, "RM": 16, "PR": 5,
                      "OPT": 4}, counts
    print("[ok] templates: 51 rule ids (PF 11, WL 7, R 8, RM 16, PR 5, OPT 4), "
          "each with its "
          "prefix's layer, no digit in any template")


def _g2_static_scan():
    sc = static_scan()
    assert sc["ok"], (sc["missing"], sc["dead"], sc["exempt_bad"], sc["tables_missing"])
    assert set(sc["ids"]) == set(TEMPLATES) | {"PF-TEST"}, set(sc["ids"]) ^ set(TEMPLATES)
    assert sc["ids"]["PF-TEST"] == ["remedies.py"], sc["ids"]["PF-TEST"]
    tmp = tempfile.mkdtemp()
    try:
        planted = os.path.join(tmp, "planted.py")
        with open(planted, "w", encoding="utf-8", newline="\n") as f:
            f.write('X = "RM-NEW-THING"\n')
        sc2 = static_scan([planted])
        assert sc2["missing"] == ["RM-NEW-THING"], sc2["missing"]
        assert not sc2["ok"]
    finally:
        os.remove(planted)
        os.rmdir(tmp)
    print("[ok] static scan: 51 ids in %d source files, all templated, none dead, "
          "PF-TEST only in remedies.py; module tables templated; a planted "
          "RM-NEW-THING is reported missing" % sc["files"])


def _g3_records():
    fx = load_fixtures()
    recs = fx["records_by_id"]
    assert len(recs) == 60, len(recs)
    ids = set()
    for rec in recs:
        c = card(rec)
        assert c["line"].startswith(rec["rule_id"] + " "), rec["rule_id"]
        ids.add(rec["rule_id"])
        assert ungrounded(c["line"], [rec]) == [], rec["rule_id"]
    fx_ids = {r for r in TEMPLATES if not r.startswith(("PR-", "OPT-"))}
    assert ids == fx_ids, ids ^ fx_ids
    for v in [0.0, 1.0, 0.1334231805929919, 3, True, None, "a", [1.5, None], {"k": 1}]:
        assert fmt(v) == remedies._fmt(v), (v, fmt(v), remedies._fmt(v))
    print("[ok] records: 60 fixture records of 42 ids render, every line starts with "
          "its rule id, 0 ungrounded; fmt == remedies._fmt on 9 values")


def _g4_rows():
    fx = load_fixtures()
    for r in fx["rows"]:
        assert validate_row(r) == [], (r["geometry_id"], r["attempt"])
    w2 = next(r for r in fx["rows"]
              if r["geometry_id"] == "wing_a_L3" and r["attempt"] == 2)

    def broken(mutate, want):
        r = copy.deepcopy(w2)
        mutate(r)
        errs = validate_row(r)
        assert any(want in e for e in errs), (want, errs)
        return len(errs)

    def drop_sha(r):
        del r["binary_sha"]

    def bad_id(r):
        r["rule_id"] = "bad id"

    def none_id(r):
        r["rule_id"] = None

    def early_end(r):
        r["t_end"] = (schema._parse_iso(r["t_start"])
                      - timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")

    def late_pred(r):
        r["prediction"] = {"p_fail": 0.1, "p_fail_std": 0.01, "blc8_a_priori": 0.5,
                           "log_cells": 4.0, "t_predicted": r["t_start"]}

    def unknown_rule(r):
        r["rule_id"] = "RM-NOT-A-" + "RULE"

    def flip_verdict(r):
        r["outcome"]["verdict"] = "pass"

    assert broken(drop_sha, "binary_sha") >= 1
    assert broken(bad_id, "rule_id") >= 1
    assert broken(none_id, "must carry its rule id") >= 1
    assert broken(early_end, "t_end") >= 1
    assert broken(late_pred, "prediction.t_predicted") >= 1
    assert broken(unknown_rule, "no template") >= 1
    assert broken(flip_verdict, "outcome.verdict") >= 1
    print("[ok] rows: 6/6 fixture rows valid; 7 broken copies refused by name")


def _g5_timestamps():
    fx = load_fixtures()
    rows = fx["rows"]
    assert sum(1 for r in rows if r["prediction"] is not None) == 0
    for r in rows:
        for d, want_ok in ((-1, True), (0, False), (1, False)):
            r2 = copy.deepcopy(r)
            t = (schema._parse_iso(r["t_start"])
                 + timedelta(seconds=d)).strftime("%Y-%m-%dT%H:%M:%SZ")
            r2["prediction"] = {"p_fail": 0.1, "p_fail_std": 0.01, "blc8_a_priori": 0.5,
                                "log_cells": 4.0, "t_predicted": t}
            errs = validate_row(r2)
            if want_ok:
                assert errs == [], errs
            else:
                assert any("prediction.t_predicted" in e for e in errs), errs
    au = audit(rows, fx["records"])
    assert au["record_order"]["checked"] == 23, au["record_order"]
    assert au["record_order"]["bad"] == [], au["record_order"]["bad"]
    print("[ok] timestamps: 0 fixture predictions; on 6 rows a prediction 1 s before "
          "t_start passes and at or 1 s after it is refused; 23/23 records ordered "
          "around their runs")


def _g6_moved():
    fx = load_fixtures()
    au = audit(fx["rows"], fx["records"])
    assert au["trigger_checked"] == 3, au["trigger_checked"]
    assert au["trigger_mismatch"] == [], au["trigger_mismatch"]
    wing1 = next(r for r in fx["rows"]
                 if r["geometry_id"] == "wing_a_L3" and r["attempt"] == 1)["outcome"]
    found, v = resolve(wing1, "outcome.pinned_frac")
    assert found and v == wing1["pinned_frac"] and abs(v - 0.1334231805929919) < 1e-12, v
    assert resolve(wing1, "outcome.flags.F3a") == (True, True)
    found, v = resolve(wing1, "outcome.patches[wing].layer_class")
    assert found and v == "retreat_snapped", v
    assert resolve(wing1, "outcome.patches[nosuch].n_layers") == (False, None)
    assert resolve(wing1, "h_wall / t1") == (False, None)
    print("[ok] moved: 3/3 remedy triggers equal the previous attempt's measurement; "
          "resolve on 5 observable forms")


def _g7_golden():
    fx = load_fixtures()
    labels = 0
    for gid in GOLDEN_IDS:
        obj = _fixture_geometry(gid, fx)
        text = obj["text"]
        assert text == _read_text(os.path.join(GOLDEN_DIR, gid + ".txt")), gid
        assert _dump(obj) == _read_text(os.path.join(GOLDEN_DIR, gid + ".json")), gid
        exp = fx["expect"]["golden"][gid]
        for s in exp["must_contain"]:
            assert s in text, (gid, s)
        for s in exp["must_not_contain"]:
            assert s not in text, (gid, s)
        lines = text.split("\n")
        for k, n in exp["counts"].items():
            got = sum(1 for ln in lines if ln.startswith(k))
            assert got == n, (gid, k, got, n)
        labels += len(exp["must_contain"]) + len(exp["must_not_contain"]) + len(exp["counts"])
        src = ([r for r in fx["rows"] if r["geometry_id"] == gid]
               + [t["record"] for t in fx["records"].get(gid, [])]
               + [len([r for r in fx["rows"] if r["geometry_id"] == gid])])
        assert ungrounded(text, src) == [], (gid, ungrounded(text, src))
    print("[ok] golden: box_sphere, wing_a_L3, D-1-002 equal their golden text and JSON, "
          "%d labels present, 0 ungrounded" % labels)


def _g8_summary():
    fx = load_fixtures()
    summ = _fixture_summary(fx)
    stxt = summary_text(summ)
    assert stxt == _read_text(os.path.join(GOLDEN_DIR, "summary.txt"))
    assert _dump(summ) == _read_text(os.path.join(GOLDEN_DIR, "summary.json"))
    for s in fx["expect"]["summary"]["must_contain"]:
        assert s in stxt, s
    want = fx["expect"]["summary"]["groups"]
    assert [(g["family"], g["stratum"]) for g in summ["groups"]] == \
        [(g["family"], g["stratum"]) for g in want], [g["family"] + "/" + g["stratum"] for g in summ["groups"]]
    mine = {(g["family"], g["stratum"]): g for g in summ["groups"]}
    for eg in want:
        g = mine[(eg["family"], eg["stratum"])]
        assert (g["n"], g["fail"], g["strict"]) == (eg["n"], eg["fail"], eg["strict"]), eg
        for key in ("mfr_ci", "strict_ci"):
            assert abs(g[key][0] - eg[key][0]) <= 1e-12 \
                and abs(g[key][1] - eg[key][1]) <= 1e-12, (eg["family"], key)
    assert ungrounded(stxt, [summ]) == [], ungrounded(stxt, [summ])
    print("[ok] summary: the fixture summary equals its golden, %d labels, %d groups "
          "as expected" % (len(fx["expect"]["summary"]["must_contain"]), len(summ["groups"])))


def _g9_clopper_pearson():
    fx = load_fixtures()
    for x, n, lo, hi in fx["expect"]["clopper_pearson"]:
        l2, h2 = clopper_pearson(x, n)
        assert abs(l2 - lo) <= 1e-12 and abs(h2 - hi) <= 1e-12, (x, n, l2, h2)
    m = 0
    for n in list(range(1, 61)) + [180, 420]:
        for x in range(0, n + 1):
            lo, hi = clopper_pearson(x, n)
            if x < n:
                assert abs(_st_binom_cdf(x, n, hi) - 0.025) <= 1e-9, (x, n)
            if x > 0:
                assert abs(1.0 - _st_binom_cdf(x - 1, n, lo) - 0.025) <= 1e-9, (x, n)
            if x == 0:
                assert lo == 0.0 and abs(hi - (1.0 - 0.025 ** (1.0 / n))) <= 1e-12, (n, hi)
            if x == n:
                assert hi == 1.0 and abs(lo - 0.025 ** (1.0 / n)) <= 1e-12, (n, lo)
            m += 1
    assert clopper_pearson(0, 0) == (None, None)
    for bad in ((3, 2), (-1, 5)):
        try:
            clopper_pearson(*bad)
            raise AssertionError("clopper_pearson%s did not raise ExplainError" % (bad,))
        except ExplainError:
            pass
    print("[ok] clopper-pearson: 9 references within 1e-12; binomial tails within "
          "1e-9 on %d (x, n); closed forms at x = 0 and x = n" % m)


def _g10_property():
    sys.path.insert(0, os.path.join(HERE, "corpus"))
    import split
    meta = meta_from_manifest(split.load("tuning", "explain"))
    assert len(meta) == 420, len(meta)
    rng = random.Random(15)
    for _trial in range(200):
        ids = rng.sample(sorted(meta), rng.randint(1, 60))
        rows = []
        for gid in ids:
            k = rng.randint(1, 4)
            for a in range(1, k + 1):
                rows.append({"campaign_id": "prop", "geometry_id": gid, "attempt": a,
                             "outcome": {"verdict": rng.choice(["pass", "fail"]),
                                         "strict_failure": rng.random() < 0.6,
                                         "blc8_a_priori": rng.random(),
                                         "blc_full_a_priori": rng.random(),
                                         "n_cells": rng.choice([None,
                                                                rng.randint(100, 2000000)]),
                                         "patches": [{"capability_limited":
                                                      rng.random() < 0.3}]}})
        rng.shuffle(rows)
        summ = summarise(rows, meta)
        finals = {}
        for r in rows:
            gid = r["geometry_id"]
            if gid not in finals or r["attempt"] > finals[gid]["attempt"]:
                finals[gid] = r
        buckets = {}
        for gid, r in finals.items():
            oc = r["outcome"]
            g = {"fail": oc["verdict"] == "fail", "strict": oc["strict_failure"] is True,
                 "blc8": oc["blc8_a_priori"], "blcf": oc["blc_full_a_priori"],
                 "cells": oc["n_cells"], "attempts": r["attempt"],
                 "cap": any(p["capability_limited"] is True for p in oc["patches"])}
            key = (meta[gid]["family"], meta[gid]["stratum"])
            buckets.setdefault(key, []).append(g)
            buckets.setdefault((key[0], "all"), []).append(g)
            buckets.setdefault(("all", "all"), []).append(g)
        order = []
        for fam in sorted(set(k[0] for k in buckets if k[0] != "all")):
            for strat in sorted(set(k[1] for k in buckets
                                    if k[0] == fam and k[1] != "all")):
                order.append((fam, strat))
            order.append((fam, "all"))
        order.append(("all", "all"))
        assert [(g["family"], g["stratum"]) for g in summ["groups"]] == order, order
        for sel in order:
            mem = buckets[sel]
            got = next(g for g in summ["groups"]
                       if (g["family"], g["stratum"]) == sel)
            n = len(mem)
            fail = sum(1 for g in mem if g["fail"])
            strict = sum(1 for g in mem if g["strict"])
            cap = sum(1 for g in mem if g["cap"])
            assert (got["n"], got["fail"], got["strict"],
                    got["capability_limited"]) == (n, fail, strict, cap), sel
            cells = [g["cells"] for g in mem if g["cells"] is not None]
            assert got["cells_median"] == (statistics.median(cells) if cells else None), sel
            for key, val in (("blc8_mean", statistics.fmean(g["blc8"] for g in mem)),
                             ("blc_full_mean", statistics.fmean(g["blcf"] for g in mem)),
                             ("attempts_mean", statistics.fmean(g["attempts"] for g in mem))):
                assert abs(got[key] - val) <= 1e-12, (sel, key)
            assert got["mfr_ci"] == list(clopper_pearson(fail, n)), sel
            assert got["strict_ci"] == list(clopper_pearson(strict, n)), sel
    print("[ok] summary property: 200 random campaigns over the 420 tuning geometries "
          "equal an independent recomputation")


def _g11_audit_faults():
    fx = load_fixtures()
    au = audit(fx["rows"], fx["records"])
    assert au["ok"] and au["n_rows"] == 6 and au["n_valid"] == 6, au["ok"]
    assert au["record_order"]["checked"] == 23 and au["trigger_checked"] == 3, au
    assert au["prediction_late"] == [] and au["trigger_mismatch"] == [], au

    rows_c = copy.deepcopy(fx["rows"])
    del rows_c[0]["binary_sha"]
    au = audit(rows_c, fx["records"])
    assert len(au["invalid"]) == 1 and au["n_valid"] == 5 and not au["ok"], au["invalid"]

    recs_c = copy.deepcopy(fx["records"])
    recs_c["box_sphere"][0]["record"]["rule_id"] = "RM-NEW"
    au = audit(fx["rows"], recs_c)
    assert au["untemplated"] == ["RM-NEW"] and not au["ok"], au["untemplated"]

    rows_c = copy.deepcopy(fx["rows"])
    t = (schema._parse_iso(rows_c[0]["t_start"])
         + timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows_c[0]["prediction"] = {"p_fail": 0.1, "p_fail_std": 0.01, "blc8_a_priori": 0.5,
                               "log_cells": 4.0, "t_predicted": t}
    au = audit(rows_c, fx["records"])
    assert len(au["prediction_late"]) == 1 and not au["ok"], au["prediction_late"]

    recs_c = copy.deepcopy(fx["records"])
    w2 = next(r for r in fx["rows"]
              if r["geometry_id"] == "wing_a_L3" and r["attempt"] == 2)
    target = next(t2 for t2 in recs_c["wing_a_L3"]
                  if t2["attempt"] == 2 and t2["record"]["rule_id"] == "RM-SNAP-FT")
    target["record"]["t"] = (schema._parse_iso(w2["t_start"])
                             + timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    au = audit(fx["rows"], recs_c)
    assert len(au["record_order"]["bad"]) == 1 and not au["ok"], au["record_order"]["bad"]

    rows_c = copy.deepcopy(fx["rows"])
    next(r for r in rows_c
         if r["geometry_id"] == "wing_a_L3" and r["attempt"] == 2)["trigger"]["value"] = 0.5
    au = audit(rows_c, fx["records"])
    assert len(au["trigger_mismatch"]) == 1 and not au["ok"], au["trigger_mismatch"]
    print("[ok] audit: the fixture audit is ok (6 rows, 23 records, 3 triggers); "
          "5 planted faults each reported")


def _g12_determinism():
    fx = load_fixtures()
    for gid in GOLDEN_IDS:
        a = _dump(_fixture_geometry(gid, fx))
        b = _dump(_fixture_geometry(gid, fx))
        assert a == b, gid
    a = _dump(_fixture_summary(fx))
    b = _dump(_fixture_summary(fx))
    assert a == b
    print("[ok] determinism: 3 golden geometries and the summary rendered twice, "
          "byte-equal")


def _g13_cli():
    import subprocess
    rows_p = os.path.join(FIXTURE_DIR, "rows.jsonl")
    recs_p = os.path.join(FIXTURE_DIR, "records.json")
    q = subprocess.run([sys.executable, __file__, "--rows", rows_p, "--records", recs_p,
                        "--geometry", "wing_a_L3"], capture_output=True,
                       encoding="utf-8", errors="replace", timeout=120)
    assert q.returncode == 0, (q.returncode, q.stderr[-500:])
    assert q.stdout == _read_text(os.path.join(GOLDEN_DIR, "wing_a_L3.txt")), \
        "the --rows text differs from the golden"
    q = subprocess.run([sys.executable, __file__, "--summary", "--rows", rows_p,
                        "--manifest", "test"], capture_output=True,
                       encoding="utf-8", errors="replace", timeout=120)
    assert q.returncode == 2 and "sealed" in q.stderr, (q.returncode, q.stderr[-500:])
    q = subprocess.run([sys.executable, __file__, "--rows", rows_p, "--geometry", "NOPE"],
                       capture_output=True, encoding="utf-8", errors="replace",
                       timeout=120)
    assert q.returncode == 2, (q.returncode, q.stderr[-500:])
    print("[ok] cli: --rows prints the golden wing_a_L3 text; --summary --manifest test "
          "exits 2 on the seal; --geometry NOPE exits 2")


def selftest() -> int:
    """The 13 G-EXPL groups; [ok] per group, SELFTEST PASS at the end."""
    groups = [("templates", _g1_templates), ("static scan", _g2_static_scan),
              ("records", _g3_records), ("rows", _g4_rows),
              ("timestamps", _g5_timestamps), ("moved", _g6_moved),
              ("golden", _g7_golden), ("summary", _g8_summary),
              ("clopper-pearson", _g9_clopper_pearson),
              ("summary property", _g10_property), ("audit", _g11_audit_faults),
              ("determinism", _g12_determinism), ("cli", _g13_cli)]
    for name, fn in groups:
        try:
            fn()
        except Exception as e:  # a failure names its group; never a silent skip
            print("SELFTEST FAIL: %s: %r" % (name, e))
            return 1
    print("SELFTEST PASS")
    return 0


# --- the gate (G-EXPL, docs/15 §F) -------------------------------------------


def _gate_numbers(p: dict) -> str:
    if p["part"] == 1:
        return "%d/%d rows valid, audit %s, row rule ids %d"
    if p["part"] == 2:
        return "%d ids in %d files, %d missing, %d dead, %d records, %d ungrounded"
    if p["part"] == 3:
        return ("%d fixture predictions, before-ok %d/6, late refused %d/12, "
                "records %d/%d")
    return "%d/%d geometries, %d labels, summary %s" % (
        sum(1 for g in p["geometries"].values() if g["ok"]), len(p["geometries"]),
        p["labels"], "ok" if p["summary_ok"] else "BAD")


def _gate_line(p: dict) -> str:
    if p["part"] == 4:
        return _gate_numbers(p)
    if p["part"] == 1:
        args = (p["n_valid"], p["n_rows"], "ok" if p["audit_ok"] else "BAD",
                len(p["rule_ids"]))
    elif p["part"] == 2:
        args = (p["ids"], p["files"], len(p["missing"]), len(p["dead"]),
                p["n_records"], p["n_ungrounded"])
    else:
        args = (p["n_predictions"], p["before_ok"], p["late_refused"],
                p["records_checked"], p["records_checked"] + p["records_bad"])
    return _gate_numbers(p) % args


def _gate_md(doc: dict) -> str:
    lines = ["<!-- meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). "
             "Source-available, not Open Source. No GPL-licensed source was consulted. -->",
             "", "# G-EXPL — explain.py (docs/15 §F)", "",
             "git head: %s" % doc["git_head"], ""]
    p1, p2, p3, p4 = doc["parts"]
    lines.append("- Part 1 rows: %s; rule ids: %s."
                 % (_gate_line(p1), ", ".join(p1["rule_ids"]) or "none"))
    lines.append("- Part 2 templates: %s; missing: %s; dead: %s."
                 % (_gate_line(p2), ", ".join(p2["missing"]) or "none",
                    ", ".join(p2["dead"]) or "none"))
    lines.append("- Part 3 timestamps: %s." % _gate_line(p3))
    lines.append("- Part 4 golden: %s." % _gate_line(p4))
    for gid in GOLDEN_IDS:
        g = p4["geometries"][gid]
        lines.append("  - %s: %d attempt(s), terminal %s; rule ids %s; labels %d/%d; "
                     "ungrounded %d." % (gid, g["n_attempts"], g["terminal"],
                                         ", ".join(g["rule_ids"]) or "none",
                                         g["labels"], g["n_labels"], g["ungrounded"]))
    s = p4["summary"]
    lines.append("  - summary: labels %d/%d; ungrounded %d."
                 % (s["labels"], s["n_labels"], s["ungrounded"]))
    lines += ["", "G-EXPL PASS" if doc["ok"] else "G-EXPL FAIL"]
    return "\n".join(lines) + "\n"


def gate() -> int:
    """The four G-EXPL parts; writes REPORT_DIR/G-EXPL.json and G-EXPL.md."""
    import subprocess
    fx = load_fixtures()
    rows = fx["rows"]
    parts = []

    n_valid = sum(1 for r in rows if not validate_row(r))
    au = audit(rows, fx["records"])
    parts.append({"part": 1, "name": "rows",
                  "ok": n_valid == len(rows) and au["ok"],
                  "n_rows": len(rows), "n_valid": n_valid, "audit_ok": au["ok"],
                  "rule_ids": sorted({r["rule_id"] for r in rows if r["rule_id"]})})

    sc = static_scan()
    un = sum(len(ungrounded(card(rec)["line"], [rec])) for rec in fx["records_by_id"])
    parts.append({"part": 2, "name": "templates", "ok": sc["ok"] and un == 0,
                  "ids": len(sc["ids"]), "files": sc["files"], "missing": sc["missing"],
                  "dead": sc["dead"], "n_records": len(fx["records_by_id"]),
                  "n_ungrounded": un})

    n_pred = sum(1 for r in rows if r["prediction"] is not None)
    before_ok = late_refused = 0
    for r in rows:
        for d in (-1, 0, 1):
            r2 = copy.deepcopy(r)
            t = (schema._parse_iso(r["t_start"])
                 + timedelta(seconds=d)).strftime("%Y-%m-%dT%H:%M:%SZ")
            r2["prediction"] = {"p_fail": 0.1, "p_fail_std": 0.01, "blc8_a_priori": 0.5,
                                "log_cells": 4.0, "t_predicted": t}
            errs = validate_row(r2)
            if d == -1 and not errs:
                before_ok += 1
            if d >= 0 and any("prediction.t_predicted" in e for e in errs):
                late_refused += 1
    ro = au["record_order"]
    parts.append({"part": 3, "name": "timestamps",
                  "ok": n_pred == 0 and before_ok == 6 and late_refused == 12
                        and ro["checked"] == 23 and not ro["bad"],
                  "n_predictions": n_pred, "before_ok": before_ok,
                  "late_refused": late_refused, "records_checked": ro["checked"],
                  "records_bad": len(ro["bad"])})

    rep = _golden_report(fx)
    labels = sum(g["labels"] for g in rep["geometries"].values()) + rep["summary"]["labels"]
    parts.append({"part": 4, "name": "golden", "ok": rep["ok"], "labels": labels,
                  "geometries": {gid: {k: rep["geometries"][gid][k]
                                       for k in ("ok", "n_attempts", "terminal",
                                                 "rule_ids", "labels", "n_labels",
                                                 "ungrounded")}
                                 for gid in GOLDEN_IDS},
                  "summary": {k: rep["summary"][k]
                              for k in ("ok", "labels", "n_labels", "ungrounded")},
                  "summary_ok": rep["summary"]["ok"]})

    q = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                       text=True)
    doc = {"schema": GATE_SCHEMA,
           "git_head": q.stdout.strip() if q.returncode == 0 else "unknown",
           "ok": all(p["ok"] for p in parts), "parts": parts}
    os.makedirs(REPORT_DIR, exist_ok=True)
    _write_text(os.path.join(REPORT_DIR, "G-EXPL.json"), _dump(doc))
    _write_text(os.path.join(REPORT_DIR, "G-EXPL.md"), _gate_md(doc))
    for p in parts:
        print("[gate] part %d %s: %s" % (p["part"], "PASS" if p["ok"] else "FAIL",
                                         _gate_line(p)))
    if doc["ok"]:
        print("G-EXPL PASS")
        return 0
    print("G-EXPL FAIL: %s" % " ".join(str(p["part"]) for p in parts if not p["ok"]))
    return 1


# --- the CLI -----------------------------------------------------------------


def _usage() -> str:
    return ("usage: explain.py --selftest | --gate | --write-golden | "
            "--rows FILE [--records FILE] [--geometry ID] [--json] | "
            "--summary --rows FILE (--meta FILE | --manifest tuning|test) [--json] | "
            "--audit --rows FILE [--records FILE]")


def _load_rows(path: str) -> list:
    if not os.path.isfile(path):
        raise ExplainError("no such rows file: %s" % path)
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f.read().splitlines() if ln.strip()]


def _load_records_maybe(path) -> dict:
    if path is None:
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)["records"]


def _parse_opts(rest: list) -> tuple:
    pos, vals, flags = [], {}, set()
    i = 0
    while i < len(rest):
        a = rest[i]
        if a == "--json":
            flags.add(a)
            i += 1
        elif a in ("--rows", "--records", "--geometry", "--meta", "--manifest"):
            if i + 1 >= len(rest):
                raise ExplainError("%s needs a value" % a)
            vals[a] = rest[i + 1]
            i += 2
        elif a.startswith("-"):
            raise ExplainError("unknown option %r; %s" % (a, _usage()))
        else:
            pos.append(a)
            i += 1
    return pos, vals, flags


def _rows_path(pos: list, vals: dict) -> str:
    """The rows FILE: positional after the mode, or the value of a --rows option."""
    got = ([vals["--rows"]] if "--rows" in vals else []) + pos
    if len(got) != 1:
        raise ExplainError("exactly one rows FILE is needed; %s" % _usage())
    return got[0]


def _cli_rows(rest: list) -> int:
    pos, vals, flags = _parse_opts(rest)
    rows = _load_rows(_rows_path(pos, vals))
    records = _load_records_maybe(vals.get("--records"))
    gids = []
    for r in rows:
        if r["geometry_id"] not in gids:
            gids.append(r["geometry_id"])
    if vals.get("--geometry") is not None:
        gids = [vals["--geometry"]]
    objs = [explain_geometry([r for r in rows if r["geometry_id"] == gid],
                             records.get(gid) or None) for gid in gids]
    if "--json" in flags:
        sys.stdout.write(_dump(objs))
    else:
        sys.stdout.write("\n".join(o["text"] for o in objs))
    return 0


def _cli_summary(rest: list) -> int:
    pos, vals, flags = _parse_opts(rest)
    rows = _load_rows(_rows_path(pos, vals))
    if ("--meta" in vals) == ("--manifest" in vals):
        raise ExplainError("--summary needs exactly one of --meta FILE or "
                           "--manifest tuning|test")
    if "--meta" in vals:
        with open(vals["--meta"], encoding="utf-8") as f:
            meta = json.load(f)["meta"]
    else:
        sys.path.insert(0, os.path.join(HERE, "corpus"))
        import split
        try:
            meta = meta_from_manifest(split.load(vals["--manifest"], "explain"))
        except (split.SplitSealed, split.SplitError) as e:
            raise ExplainError(str(e))
    summ = summarise(rows, meta)
    if "--json" in flags:
        sys.stdout.write(_dump(summ))
    else:
        sys.stdout.write(summary_text(summ))
    return 0


def _cli_audit(rest: list) -> int:
    pos, vals, _flags = _parse_opts(rest)
    au = audit(_load_rows(_rows_path(pos, vals)),
               _load_records_maybe(vals.get("--records")) or None)
    sys.stdout.write(_dump(au))
    return 0 if au["ok"] else 1


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except AttributeError:
            pass
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if not argv:
            raise ExplainError("no mode given; %s" % _usage())
        mode, rest = argv[0], argv[1:]
        if mode == "--selftest":
            return selftest()
        if mode == "--gate":
            return gate()
        if mode == "--write-golden":
            return write_golden()
        if mode == "--rows":
            return _cli_rows(rest)
        if mode == "--summary":
            return _cli_summary(rest)
        if mode == "--audit":
            return _cli_audit(rest)
        raise ExplainError("unknown mode %r; %s" % (mode, _usage()))
    except ExplainError as e:
        sys.stderr.write("explain: %s\n" % (e,))
        return 2


if __name__ == "__main__":
    sys.exit(main())
