#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""admit.py - admits or refuses a candidate template by the first failing ADM rule, and freezes the
admitted ones into templates.lock (docs/16 §G, docs/16a lines 69-70 and 254, CAD-23).

The rule order, decided here: ADM-AST-LOCK, ADM-AST-IMPORT, ADM-AST-NAME, ADM-AST-FALLBACK,
ADM-CONTRACT, ADM-BUILD, ADM-STAGE, ADM-DETERM, ADM-TAGS, ADM-INSENSITIVE; a candidate is refused
by the first failing id. ADM-AST-LOCK comes before ADM-AST-IMPORT so a template importing `reqs`
is named for the lock breach, not for an unlisted import. ADM-AST-FALLBACK (docs/16a line 69)
refuses a handler that returns the receiver of a fillet/chamfer/shell/offset2D/cut/union/
intersect/fuse/loft/sweep/revolve call unchanged, unless it re-raises or rebuilds it; its stated
limit holds here too - a handler that passes and falls through to a later return of that name is
not caught statically, and is left to the geometric checks. ADM-STAGE also refuses a boolean whose
measure change relative to the body is below NOOP_REL (docs/16a line 254), the convention every
template follows from AMG-0 on.

The parent NEVER imports the candidate: the four static rules are pure `ast`, and every dynamic
probe runs in a runner child through admit_probe.py (entries declare and sweep). The AST whitelist
plus a child process is not a security sandbox on Windows: admission runs only on a user-initiated
authoring turn, never inside the GUI server process, and is decision D-6 (docs/16 §G).

Usage:
  python admit.py --selftest
  python admit.py check SOURCE_PY RECORD_JSON
  python admit.py freeze SOURCE_PY RECORD_JSON --by NAME [--lock LOCK]
  python admit.py frozen TEMPLATE_DIR [--lock LOCK]
"""

import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402
import hints  # noqa: E402
import schema  # noqa: E402

RULES = ("ADM-AST-LOCK", "ADM-AST-IMPORT", "ADM-AST-NAME", "ADM-AST-FALLBACK", "ADM-CONTRACT",
         "ADM-BUILD", "ADM-STAGE", "ADM-DETERM", "ADM-TAGS", "ADM-INSENSITIVE")
FREEZE_IDS = ("FREEZE-BY", "FREEZE-UNADMITTED", "FREEZE-IMMUTABLE")
RETRY_CLASS = {"ADM-AST-LOCK": "execution", "ADM-AST-IMPORT": "execution", "ADM-AST-NAME": "execution",
               "ADM-AST-FALLBACK": "execution", "ADM-CONTRACT": "execution", "ADM-BUILD": "execution",
               "ADM-DETERM": "execution", "ADM-STAGE": "geometry", "ADM-TAGS": "geometry",
               "ADM-INSENSITIVE": "geometry"}
SOBOL_N_LOG2 = 6            # 64 points, scipy.stats.qmc.Sobol(d, scramble=True, seed=SOBOL_SEED).random_base2(6)
SOBOL_SEED = 0
CORNER_CAP = 32
PERTURB = 0.05              # +/-5 % (docs/16 §G ADM-INSENSITIVE)
NOOP_REL = 1e-9             # docs/16a line 254: a boolean whose |dV|/V_before is below this is a no-op
TAG_AREA_REL = 1e-12        # a face tag whose area <= this * the fluid's total face area is zero-area
EDGE_LEN_REL = 1e-12        # an edge tag whose length <= this * (x_outlet - x_inlet) is zero-length
MOVE_REL = 1e-12            # moved iff |q - q0| > max(u0, MOVE_REL * |q0|), u0 the nominal record's u_meas
DECLARE_TIMEOUT_S = 120
SWEEP_TIMEOUT_S = 900
TRACE_TAIL = 4096
ADMIT_JOBS = 2              # selftest: at most 2 `check` children at a time
TEMPLATES_LOCK = os.path.join(HERE, "templates.lock")
ADMISSIBLE_PRIMITIVES = ("diameter_at_plane", "area_ratio", "extent_along_axis", "plane_distance",
                         "meridian_min_wall", "slope_max", "curvature_radius_min", "n_solids", "valid",
                         "axis_x", "watertight", "units_m")     # what export.measure_catalogue dispatches
LOCK_NAMES = ("reqs", "write_locked", "apply", "loop", "gate", "admit")
BANNED_NAMES = {"open", "exec", "eval", "compile", "__import__", "getattr", "setattr", "delattr",
                "globals", "locals", "vars", "os", "sys", "subprocess", "pathlib", "socket",
                "ctypes", "importlib", "__name__", "__file__", "__builtins__", "breakpoint", "input"}
FALLBACK_METHODS = ("fillet", "chamfer", "shell", "offset2D", "cut", "union", "intersect", "fuse",
                    "loft", "sweep", "revolve")
OCP_MODULES = ("gp", "Geom", "Geom2d", "GeomAbs", "GeomAPI", "GeomConvert", "TColgp", "TColStd",
               "BRepBuilderAPI", "BRepPrimAPI", "BRepExtrema", "BRepCheck", "BRepGProp", "GProp",
               "BRepAdaptor", "GCPnts", "ShapeAnalysis", "TopExp", "TopAbs", "TopoDS", "TopTools",
               "Precision", "BRepAlgoAPI")
OCP_ONLY = {"BRepAlgoAPI": ("BRepAlgoAPI_Check",)}         # booleans must go through cadquery, which the probe records
PROBE = os.path.join(HERE, "admit_probe.py")
NOZZLE_DIR = os.path.join(HERE, "templates", "nozzle_contraction")
ADMISSIONS_DIR = os.path.join(HERE, "admissions")
ADMISSION_KEYS = ("schema", "source", "source_sha256", "template_id", "status", "rule", "detail",
                  "retry_class", "point", "traceback", "hint", "drivers_mode", "counts", "sensitivity",
                  "env", "rules")
LOCK_ENTRY_KEYS = ("template_id", "source", "source_sha256", "declaration_sha256", "cadquery", "occt",
                   "frozen_by", "admission_sha256")
CONTRACT_NAMES = ("TEMPLATE_ID", "PARAMS", "PLANES", "TAGS", "CATALOGUE")
USAGE = ("usage: python admit.py --selftest" + chr(10)
         + "       python admit.py check SOURCE_PY RECORD_JSON" + chr(10)
         + "       python admit.py freeze SOURCE_PY RECORD_JSON --by NAME [--lock LOCK]" + chr(10)
         + "       python admit.py frozen TEMPLATE_DIR [--lock LOCK]" + chr(10))


# ---------------------------------------------------------------- the static rules
def _dotted(name):
    """The dotted components of a module name ("tools.cad.reqs" -> tools, cad, reqs)."""
    return [p for p in (name or "").split(".") if p]


def _is_dunder(name):
    return bool(name) and name.startswith("__") and name.endswith("__")


def _parse(source_path):
    """(tree, None) or (None, (ADM-CONTRACT, detail)): the file must be readable UTF-8 that parses."""
    try:
        with open(source_path, "rb") as f:
            raw = f.read()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as e:
        return None, ("ADM-CONTRACT", "the source is not a readable UTF-8 file: %s" % (e,))
    try:
        return ast.parse(text), None
    except SyntaxError as e:
        return None, ("ADM-CONTRACT", "the source does not parse as a module: %s" % (e,))


def _lock_refusal(tree):
    """ADM-AST-LOCK: nothing under templates/ imports reqs, write_locked or an apply path."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                for part in _dotted(a.name) + ([a.asname] if a.asname else []):
                    if part in LOCK_NAMES:
                        return ("ADM-AST-LOCK", "line %d: import %s names %r"
                                % (node.lineno, a.name, part))
        elif isinstance(node, ast.ImportFrom):
            parts = _dotted(node.module)
            parts += [a.name for a in node.names if a.name != "*"]
            parts += [a.asname for a in node.names if a.asname]
            for part in parts:
                if part in LOCK_NAMES:
                    return ("ADM-AST-LOCK", "line %d: from %s import names %r"
                            % (node.lineno, node.module, part))
        elif isinstance(node, ast.Name):
            if node.id == "write_locked":
                return ("ADM-AST-LOCK", "line %d: use of the name write_locked" % (node.lineno,))
        elif isinstance(node, ast.Attribute):
            if node.attr == "write_locked":
                return ("ADM-AST-LOCK", "line %d: use of the attribute write_locked" % (node.lineno,))
    return None


def _import_refusal(tree):
    """ADM-AST-IMPORT: exactly math, cadquery (with an alias), from math, and from OCP.<listed>."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if len(node.names) != 1:
                return ("ADM-AST-IMPORT", "line %d: an import names one module, got %r"
                        % (node.lineno, ", ".join(a.name for a in node.names)))
            a = node.names[0]
            if a.name == "math" and a.asname is None:
                continue
            if a.name == "cadquery":
                continue
            return ("ADM-AST-IMPORT", "line %d: import %s is not on the whitelist"
                    % (node.lineno, a.name))
        elif isinstance(node, ast.ImportFrom):
            names = [a.name for a in node.names]
            if node.level:
                return ("ADM-AST-IMPORT", "line %d: a relative import is refused" % (node.lineno,))
            if "*" in names:
                return ("ADM-AST-IMPORT", "line %d: a star import from %s is refused"
                        % (node.lineno, node.module))
            if node.module == "math":
                continue
            mods = _dotted(node.module)
            if len(mods) == 2 and mods[0] == "OCP" and mods[1] in OCP_MODULES:
                m = mods[1]
                if m in OCP_ONLY:
                    bad = [n for n in names if n not in OCP_ONLY[m]]
                    if bad:
                        return ("ADM-AST-IMPORT", "line %d: OCP.%s may import only %s, got %s"
                                % (node.lineno, m, ", ".join(OCP_ONLY[m]), ", ".join(bad)))
                continue
            return ("ADM-AST-IMPORT", "line %d: from %s import %s is not on the whitelist"
                    % (node.lineno, node.module, ", ".join(names)))
    return None


def _name_refusal(tree):
    """ADM-AST-NAME: no banned or dunder name, definition, argument, alias, global or attribute."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id in BANNED_NAMES:
                return ("ADM-AST-NAME", "line %d: use of the banned name %r" % (node.lineno, node.id))
            if _is_dunder(node.id):
                return ("ADM-AST-NAME", "line %d: use of the dunder name %r" % (node.lineno, node.id))
        elif isinstance(node, ast.Attribute):
            if _is_dunder(node.attr):
                return ("ADM-AST-NAME", "line %d: use of the dunder attribute %r"
                        % (node.lineno, node.attr))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in BANNED_NAMES or _is_dunder(node.name):
                return ("ADM-AST-NAME", "line %d: the definition name %r is banned or a dunder"
                        % (node.lineno, node.name))
        elif isinstance(node, ast.arg):
            if node.arg in BANNED_NAMES or _is_dunder(node.arg):
                return ("ADM-AST-NAME", "line %d: the argument name %r is banned or a dunder"
                        % (node.lineno, node.arg))
        elif isinstance(node, ast.alias):
            nm = node.asname or node.name
            if nm in BANNED_NAMES or _is_dunder(nm):
                return ("ADM-AST-NAME", "line %d: the import alias %r is banned or a dunder"
                        % (node.lineno, nm))
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            for nm in node.names:
                if nm in BANNED_NAMES or _is_dunder(nm):
                    return ("ADM-AST-NAME", "line %d: the %s name %r is banned or a dunder"
                            % (node.lineno, "global" if isinstance(node, ast.Global) else "nonlocal", nm))
    return None


def _root_name(call):
    """The root Name of a call chain: body.faces(">X").fillet(r) -> body; None when there is none."""
    node = call
    while True:
        if isinstance(node, ast.Call):
            node = node.func
        elif isinstance(node, ast.Attribute):
            node = node.value
        elif isinstance(node, ast.Name):
            return node.id
        else:
            return None


def _assigns_to(handler_nodes, name):
    for x in handler_nodes:
        if isinstance(x, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == name for t in x.targets):
                return True
        elif isinstance(x, (ast.AugAssign, ast.AnnAssign, ast.NamedExpr)):
            if isinstance(x.target, ast.Name) and x.target.id == name:
                return True
    return False


def _fallback_refusal(tree):
    """ADM-AST-FALLBACK (docs/16a line 69): a handler that returns a boolean/feature receiver
    unchanged, unless it re-raises or rebuilds it. A handler that passes and falls through to a
    later return of the name is not caught here; that is left to the geometric checks."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        roots = {}
        for stmt in node.body:
            for sub in ast.walk(stmt):
                if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                        and sub.func.attr in FALLBACK_METHODS):
                    nm = _root_name(sub)
                    if nm is not None:
                        roots.setdefault(nm, set()).add(sub.func.attr)
        if not roots:
            continue
        for handler in node.handlers:
            inner = list(ast.walk(handler))
            has_raise = any(isinstance(x, ast.Raise) for x in inner)
            for nm in sorted(roots):
                returns_n = any(isinstance(x, ast.Return) and isinstance(x.value, ast.Name)
                                and x.value.id == nm for x in inner)
                if returns_n and not has_raise and not _assigns_to(inner, nm):
                    return ("ADM-AST-FALLBACK",
                            "line %d: the handler returns %s unchanged from a %s call instead of"
                            " re-raising or rebuilding it"
                            % (handler.lineno, nm, ", ".join(sorted(roots[nm]))))
    return None


def _ast_refusal(tree):
    """The four static rules in order: LOCK, IMPORT, NAME, FALLBACK; (rule, detail) or None."""
    return _lock_refusal(tree) or _import_refusal(tree) or _name_refusal(tree) or _fallback_refusal(tree)


# ---------------------------------------------------------------- the sweep
def nominal_params(params_rows):
    """{name: default_real for real rows, default_choice for choice rows}, in PARAMS order."""
    out = {}
    for row in params_rows:
        if row["kind"] == "real":
            out[row["name"]] = row["default_real"]
        else:
            out[row["name"]] = row["default_choice"]
    return out


def sweep_points(params_rows):
    """[nominal] + 64 Sobol points + box corners (capped) -> (points, n_sobol, n_corners).

    Sobol dims are the bounded reals in PARAMS order then the choice rows in order; a real gets
    min + u * (max - min), a choice gets choices[min(int(u * n), n - 1)], every other param keeps
    its nominal. Corner k sets bit j to the max of bounded row j, else the min. Two calls return
    equal lists (scramble seed SOBOL_SEED).
    """
    bounded = [r for r in params_rows
               if r["kind"] == "real" and r["min"] is not None and r["max"] is not None]
    choices = [r for r in params_rows if r["kind"] == "choice"]
    from scipy.stats import qmc
    sampler = qmc.Sobol(len(bounded) + len(choices), scramble=True, seed=SOBOL_SEED)
    units = sampler.random_base2(SOBOL_N_LOG2)
    points = [dict(nominal_params(params_rows))]
    for row_u in units:
        p = dict(points[0])
        k = 0
        for r in bounded:
            p[r["name"]] = float(r["min"] + float(row_u[k]) * (r["max"] - r["min"]))
            k += 1
        for r in choices:
            cs = list(r["choices"])
            p[r["name"]] = cs[min(int(float(row_u[k]) * len(cs)), len(cs) - 1)]
            k += 1
        points.append(p)
    n_corners = min(2 ** len(bounded), CORNER_CAP)
    for k in range(n_corners):
        p = dict(points[0])
        for j, r in enumerate(bounded):
            p[r["name"]] = float(r["max"] if (k >> j) & 1 else r["min"])
        points.append(p)
    return points, len(units), n_corners


# ---------------------------------------------------------------- the record helpers
def _env():
    fp = common.env_fingerprint()
    return {"cadquery": fp["packages"]["cadquery"], "occt": fp["occt"]}


def _rel_source(path):
    """The path relative to common.REPO with / separators when inside the repo, else the basename."""
    ap = os.path.normpath(os.path.abspath(path))
    repo = os.path.normpath(os.path.abspath(common.REPO))
    try:
        rel = os.path.relpath(ap, repo)
    except ValueError:
        return os.path.basename(ap)
    if rel.startswith(".."):
        return os.path.basename(ap)
    return rel.replace(os.sep, "/")


def _new_record(source_path):
    rec = {}
    for k in ADMISSION_KEYS:
        rec[k] = None
    rec["schema"] = "cad-admission/1"
    rec["source"] = _rel_source(source_path)
    rec["source_sha256"] = common.sha256_file(source_path)
    rec["counts"] = {"sweep": 0, "sobol": 0, "corners": 0, "accepted": 0, "pipelines": 0}
    rec["sensitivity"] = {}
    rec["env"] = _env()
    rec["rules"] = list(RULES)
    return rec


def _refuse(rec, rule, detail, point=None, traceback=None, hint=None):
    rec["status"] = "refused"
    rec["rule"] = rule
    rec["detail"] = detail
    rec["retry_class"] = RETRY_CLASS[rule]
    rec["point"] = point
    rec["traceback"] = traceback
    if traceback:
        rec["hint"] = hint if hint is not None else hints.match_hint(traceback or detail)
    return rec


def _module_assigns(tree):
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                and node.value is not None:
            names.add(node.target.id)
    return names


def _module_defs(tree):
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.setdefault(node.name, node)
    return out


def _positional(fn):
    a = fn.args
    return len(a.posonlyargs) + len(a.args) + (1 if a.vararg else 0)


def _contract_static(source_path, tree):
    """The static half of ADM-CONTRACT: the five module names, the two entries, DRIVERS."""
    assigns = _module_assigns(tree)
    for nm in CONTRACT_NAMES:
        if nm not in assigns:
            return ("ADM-CONTRACT", "no module-level assignment to %s" % (nm,))
    defs = _module_defs(tree)
    dr = defs.get("domain_rules")
    if dr is None or _positional(dr) != 1:
        return ("ADM-CONTRACT", "no module-level def domain_rules with exactly 1 positional argument")
    bd = defs.get("build")
    if bd is None or _positional(bd) != 2:
        return ("ADM-CONTRACT", "no module-level def build with exactly 2 positional arguments")
    tj = os.path.join(os.path.dirname(os.path.abspath(source_path)), "template.json")
    if "DRIVERS" not in assigns and not os.path.isfile(tj):
        return ("ADM-CONTRACT", "no module-level DRIVERS and no template.json beside the source:"
                                " a written candidate must declare which parameter drives which quantity")
    return None


# ---------------------------------------------------------------- ADM-CONTRACT
def _declare(source_path, tmp):
    """The declare child; (value, None) or (None, (rule, detail, traceback, hint))."""
    import runner
    job = {"candidate": os.path.abspath(source_path)}
    r = runner.run_job(PROBE, job, os.path.join(tmp, "declare"), entry="declare",
                       timeout_s=DECLARE_TIMEOUT_S)
    if r["status"] != "ok":
        return None, ("ADM-CONTRACT", r["message"] or ("the declare child %s" % (r["rule"],)),
                      r["stderr_tail"], r["hint"])
    v = r["value"]
    if not isinstance(v, dict) or sorted(v) != sorted(
            ("template_id", "params", "planes", "tags", "catalogue", "drivers", "profile_rules",
             "standards", "nominal_rule")):
        return None, ("ADM-CONTRACT", "the declare child returned an unexpected value shape", None, None)
    return v, None


def _contract_checks(source_path, v):
    """The five dynamic contract checks, first failure wins; (rule, detail) or None."""
    decl = {"schema": "cad-template/1", "template_id": v["template_id"], "title": v["template_id"],
            "axis": "+x", "units": "m", "params": v["params"], "planes": v["planes"],
            "tags": v["tags"], "catalogue": v["catalogue"], "profile_rules": v["profile_rules"],
            "standards": v["standards"]}
    errs = schema.errors(decl, "cad-template/1")
    if errs:
        return ("ADM-CONTRACT", "the declaration fails cad-template/1: %s" % (errs[0],))
    import measure
    quantities = []
    for row in v["catalogue"]:
        if row["primitive"] not in ADMISSIBLE_PRIMITIVES or row["primitive"] not in measure.PRIMITIVES:
            return ("ADM-CONTRACT", "catalogue row %s measures through primitive %r, which is"
                                    " neither admissible nor in measure.py"
                    % (row["quantity"], row["primitive"]))
        if row["quantity"] in quantities:
            return ("ADM-CONTRACT", "catalogue row %s repeats an earlier quantity" % (row["quantity"],))
        quantities.append(row["quantity"])
    drivers = v["drivers"]
    if drivers is not None:
        real_names = set(row["name"] for row in v["params"] if row["kind"] == "real")
        if not isinstance(drivers, dict) or set(drivers) != set(quantities):
            return ("ADM-CONTRACT", "DRIVERS is not a dict keyed by exactly the catalogue quantities")
        for q, lst in drivers.items():
            if not isinstance(lst, list) or any(n not in real_names for n in lst) \
                    or len(set(lst)) != len(lst):
                return ("ADM-CONTRACT", "DRIVERS[%s] is not a list of distinct real parameter names"
                        % (q,))
    tj = os.path.join(os.path.dirname(os.path.abspath(source_path)), "template.json")
    if os.path.isfile(tj):
        on_disk = common.read_json(tj)
        for field in ("params", "planes", "tags", "catalogue"):
            if common.canonical_json(on_disk.get(field)) != common.canonical_json(v[field]):
                return ("ADM-CONTRACT", "the template.json beside the source has %s that differ"
                                        " from the module's" % (field,))
    if v["nominal_rule"] is not None:
        return ("ADM-CONTRACT", "domain_rules refuses the defaults: %s: %s"
                % (v["nominal_rule"][0], v["nominal_rule"][1]))
    return None


# ---------------------------------------------------------------- the staged copy
def _stage_copy(source_path, declaration, tmp):
    """A temp directory holding the source's bytes as template.py and the generated declaration as
    template.json; every export.run_pipeline judges this copy, each run in its own out dir."""
    stage = os.path.join(tmp, "stage")
    os.makedirs(stage, exist_ok=True)
    with open(source_path, "rb") as f:
        raw = f.read()
    with open(os.path.join(stage, "template.py"), "wb") as f:
        f.write(raw)
    common.write_json(os.path.join(stage, "template.json"), declaration)
    return stage


def _pipeline(stage, params, out_dir):
    """One export.run_pipeline of the staged copy; (status, rule, message, out_dir)."""
    import export
    r = export.run_pipeline(os.path.join(stage, "template.py"), dict(params), out_dir)
    return r


def _probes_rows(out_dir):
    return common.read_json(os.path.join(out_dir, "probes.json"))["rows"]


def _probes_bad(rows):
    for row in rows:
        if row["record"]["status"] != "ok":
            return row
    return None


# ---------------------------------------------------------------- ADM-INSENSITIVE
def _only_when_holds(row, nom):
    ow = row.get("only_when")
    if ow is None:
        return True
    key, want = ow.split("=", 1)
    return nom.get(key) == want


def _checked_params(params_rows, nom):
    """The real rows in PARAMS order whose only_when holds at the nominal."""
    return [row for row in params_rows if row["kind"] == "real" and _only_when_holds(row, nom)]


def _sides_of(row, p0):
    """The kept +/-5 % sides of one param: (side values, kept count)."""
    if p0 == 0:
        if row["min"] is None or row["max"] is None:
            return []
        cands = [p0 - PERTURB * (row["max"] - row["min"]), p0 + PERTURB * (row["max"] - row["min"])]
    else:
        cands = [p0 * (1.0 + PERTURB), p0 * (1.0 - PERTURB)]
    kept = []
    for v in cands:
        if row["min"] is not None and v < row["min"]:
            continue
        if row["max"] is not None and v > row["max"]:
            continue
        kept.append(float(v))
    return kept


def _insensitive(rec, v, stage, nom, q0, u0, tmp):
    """ADM-INSENSITIVE: declared mode with DRIVERS, inferred mode without (hand-reviewed only)."""
    rec["drivers_mode"] = "declared" if v["drivers"] is not None else "inferred"
    params_rows = v["params"]
    checked = _checked_params(params_rows, nom)
    sides = {}
    for row in checked:
        sides[row["name"]] = _sides_of(row, nom[row["name"]])
    quantities = [row["quantity"] for row in v["catalogue"]]
    u_kinds = dict((row["quantity"], row["u_kind"]) for row in v["catalogue"])
    moved = dict((q, dict((row["name"], False) for row in checked)) for q in quantities)
    n = 0
    for row in checked:
        p = row["name"]
        for value_p in sides[p]:
            sp = dict(nom)
            sp[p] = value_p
            out = os.path.join(tmp, "side_%s_%d" % (p, n))
            n += 1
            rec["counts"]["pipelines"] += 1
            r = _pipeline(stage, sp, out)
            if r["status"] != "ok":
                continue                      # a refused or failed side is skipped
            bad = _probes_bad(_probes_rows(out))
            if bad is not None:
                return _refuse(rec, "ADM-INSENSITIVE",
                               "the perturbation of %s measured %s with status %r"
                               % (p, bad["quantity"], bad["record"]["status"]), traceback=None)
            for prow in _probes_rows(out):
                q = prow["quantity"]
                if q in q0 and abs(prow["record"]["value"] - q0[q]) > max(u0[q], MOVE_REL * abs(q0[q])):
                    moved[q][p] = True
    rec["sensitivity"] = moved
    drivers = v["drivers"]
    if drivers is not None:
        for q in quantities:
            for row in checked:
                p = row["name"]
                if p in drivers[q] and not sides[p]:
                    return _refuse(rec, "ADM-INSENSITIVE", "%s: driver %s could not be perturbed" % (q, p))
                if p in drivers[q] and not moved[q][p]:
                    return _refuse(rec, "ADM-INSENSITIVE",
                                   "%s does not move when its driver %s moves +/-5 %%" % (q, p))
                if p not in drivers[q] and moved[q][p]:
                    return _refuse(rec, "ADM-INSENSITIVE", "%s moves when the unrelated %s moves" % (q, p))
    else:
        for q in quantities:
            if u_kinds[q] != "exact" and not any(moved[q].values()):
                return _refuse(rec, "ADM-INSENSITIVE", "%s moves under no parameter" % (q,))
            if u_kinds[q] == "exact":
                for row in checked:
                    if moved[q][row["name"]]:
                        return _refuse(rec, "ADM-INSENSITIVE",
                                       "the exact quantity %s moves when %s moves" % (q, row["name"]))
    return None


# ---------------------------------------------------------------- ADM-TAGS
def _tags_refusal(rec, declared, accepted):
    """ADM-TAGS, per accepted ok point in order, then across the sweep; a refusal record or None."""
    face_declared = [t["name"] for t in declared["tags"] if t["kind"] == "face"]
    edge_declared = [t["name"] for t in declared["tags"] if t["kind"] == "edge"]
    plane_declared = sorted(p["name"] for p in declared["planes"])
    seen = set()
    for s in accepted:
        if s["status"] != "ok":
            continue
        where = "point %d" % (s["i"],)
        for tag in s["face_tags"]:
            if tag not in face_declared:
                return _refuse(rec, "ADM-TAGS", "%s: face tag %s is not declared" % (where, tag),
                               point=None)
        if s["untagged"]:
            return _refuse(rec, "ADM-TAGS", "%s: %d fluid faces carry no tag: %s"
                           % (where, len(s["untagged"]), s["untagged"][:8]))
        if s["duplicated"]:
            return _refuse(rec, "ADM-TAGS", "%s: fluid faces in more than one tag: %s"
                           % (where, s["duplicated"][:8]))
        for tag in sorted(s["face_tags"]):
            row = s["face_tags"][tag]
            if row["area"] <= TAG_AREA_REL * s["fluid_area"]:
                return _refuse(rec, "ADM-TAGS", "%s: face tag %s measures %r m2, at or below the"
                                " zero area %r" % (where, tag, row["area"], TAG_AREA_REL * s["fluid_area"]))
        for tag in edge_declared:
            in_m, in_w = tag in s["meridian_edges"], tag in s["wall_edges"]
            if not in_m and not in_w:
                return _refuse(rec, "ADM-TAGS", "%s: declared edge tag %s is missing" % (where, tag))
            if in_m and in_w:
                return _refuse(rec, "ADM-TAGS", "%s: declared edge tag %s is duplicated" % (where, tag))
            length = s["meridian_edges"].get(tag, 0.0) + s["wall_edges"].get(tag, 0.0)
            if length <= EDGE_LEN_REL * s["span"]:
                return _refuse(rec, "ADM-TAGS", "%s: edge tag %s measures %r m, at or below the"
                                " zero length %r" % (where, tag, length, EDGE_LEN_REL * s["span"]))
        if sorted(s["planes"]) != plane_declared:
            return _refuse(rec, "ADM-TAGS", "%s: the plane names %s are not the declared %s"
                           % (where, sorted(s["planes"]), plane_declared))
        seen.update(s["face_tags"])
    for tag in face_declared:
        if tag not in seen:
            return _refuse(rec, "ADM-TAGS", "declared face tag %s appears on no sweep point" % (tag,))
    return None


# ---------------------------------------------------------------- admit
def admit(source_path) -> dict:
    """Judge one candidate source by the RULES in order; the cad-admission/1 record."""
    rec = _new_record(source_path)
    tmp = tempfile.mkdtemp(prefix="admit_")
    try:
        return _admit_in(source_path, rec, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _admit_in(source_path, rec, tmp):
    tree, ref = _parse(source_path)
    if ref:
        return _refuse(rec, ref[0], ref[1])
    ref = _ast_refusal(tree)
    if ref:
        return _refuse(rec, ref[0], ref[1])
    ref = _contract_static(source_path, tree)
    if ref:
        return _refuse(rec, ref[0], ref[1])
    v, ref = _declare(source_path, tmp)
    if ref:
        return _refuse(rec, ref[0], ref[1], traceback=ref[2], hint=ref[3])
    rec["template_id"] = v["template_id"]
    ref = _contract_checks(source_path, v)
    if ref:
        return _refuse(rec, ref[0], ref[1])
    declaration = {"schema": "cad-template/1", "template_id": v["template_id"],
                   "title": v["template_id"], "axis": "+x", "units": "m", "params": v["params"],
                   "planes": v["planes"], "tags": v["tags"], "catalogue": v["catalogue"],
                   "profile_rules": v["profile_rules"], "standards": v["standards"]}
    stage = _stage_copy(source_path, declaration, tmp)
    nom = nominal_params(v["params"])
    points, n_sobol, n_corners = sweep_points(v["params"])
    rec["counts"]["sweep"] = len(points)
    rec["counts"]["sobol"] = n_sobol
    rec["counts"]["corners"] = n_corners
    import runner
    job = {"candidate": os.path.abspath(source_path), "points": points}
    r = runner.run_job(PROBE, job, os.path.join(tmp, "sweep"), entry="sweep",
                       timeout_s=SWEEP_TIMEOUT_S)
    if r["status"] != "ok":
        return _refuse(rec, "ADM-BUILD", r["message"] or ("the sweep child %s" % (r["rule"],)),
                       traceback=r["stderr_tail"], hint=r["hint"])
    srecs = r["value"]["points"]
    accepted = [s for s in srecs if s["accepted"]]
    rec["counts"]["accepted"] = len(accepted)
    if not accepted:
        return _refuse(rec, "ADM-BUILD", "domain_rules accepts no sweep point")
    for s in accepted:
        if s["status"] in ("error", "malformed"):
            return _refuse(rec, "ADM-BUILD", "point %d: %s" % (s["i"], s["error"] or s["detail"]),
                           point=points[s["i"]], traceback=s["traceback"])
    # ADM-STAGE: the sweep's refused points, stage failures and no-op booleans
    for s in accepted:
        if s["status"] == "refused":
            return _refuse(rec, "ADM-STAGE", "point %d: the template refused %s: %s"
                           % (s["i"], s["rule"], s["detail"]), point=points[s["i"]])
        if s["stage"]:
            return _refuse(rec, "ADM-STAGE", "point %d: %s" % (s["i"], "; ".join(s["stage"])),
                           point=points[s["i"]])
        for b in s["booleans"]:
            if b["rel"] < NOOP_REL:
                return _refuse(rec, "ADM-STAGE",
                               "point %d: %s changed the measure by rel %r < %r (a no-op boolean)"
                               % (s["i"], b["op"], b["rel"], NOOP_REL), point=points[s["i"]])
    # ADM-STAGE: the nominal pipeline run 1
    out1 = os.path.join(tmp, "nominal1")
    rec["counts"]["pipelines"] += 1
    p1 = _pipeline(stage, nom, out1)
    if p1["status"] != "ok":
        return _refuse(rec, "ADM-STAGE", "the nominal pipeline run 1 is %s: %s"
                       % (p1["status"], p1["message"] or p1["rule"]))
    bad = _probes_bad(_probes_rows(out1))
    if bad is not None:
        return _refuse(rec, "ADM-STAGE", "the nominal pipeline run 1 measured %s with status %r"
                       % (bad["quantity"], bad["record"]["status"]))
    # ADM-DETERM: a second fresh nominal run, byte-equal probes.json
    out2 = os.path.join(tmp, "nominal2")
    rec["counts"]["pipelines"] += 1
    p2 = _pipeline(stage, nom, out2)
    if p2["status"] != "ok":
        return _refuse(rec, "ADM-DETERM", "the nominal pipeline run 2 is %s: %s"
                       % (p2["status"], p2["message"] or p2["rule"]))
    with open(os.path.join(out1, "probes.json"), "rb") as f:
        b1 = f.read()
    with open(os.path.join(out2, "probes.json"), "rb") as f:
        b2 = f.read()
    if b2 != b1:
        return _refuse(rec, "ADM-DETERM", "two fresh nominal pipeline runs wrote different"
                                          " probes.json bytes")
    ref = _tags_refusal(rec, declaration, accepted)
    if ref:
        return ref
    q0, u0 = {}, {}
    for row in _probes_rows(out1):
        q0[row["quantity"]] = row["record"]["value"]
        u0[row["quantity"]] = row["record"]["u_meas"]
    ref = _insensitive(rec, v, stage, nom, q0, u0, tmp)
    if ref:
        return ref
    rec["status"] = "admitted"
    return rec


# ---------------------------------------------------------------- freeze and the lock
def freeze(source_path, record_path, by, lock_path=TEMPLATES_LOCK) -> dict:
    """Write the lock entry for one admitted record; ValueError("<ID>: ..."), first failure wins."""
    if by is None or not str(by).strip():
        raise ValueError("FREEZE-BY: freeze needs the person who froze the template (--by NAME)")
    try:
        rec = common.read_json(record_path)
    except (OSError, ValueError) as e:
        raise ValueError("FREEZE-UNADMITTED: the admission record %s cannot be read: %s"
                         % (record_path, e))
    if not isinstance(rec, dict) or rec.get("status") != "admitted":
        raise ValueError("FREEZE-UNADMITTED: %s is not an admitted admission record" % (record_path,))
    src_sha = common.sha256_file(source_path)
    if rec.get("source_sha256") != src_sha:
        raise ValueError("FREEZE-UNADMITTED: the record's source_sha256 %r differs from the"
                         " source's sha256 today %r" % (rec.get("source_sha256"), src_sha))
    env = _env()
    if rec.get("env") != env:
        raise ValueError("FREEZE-UNADMITTED: the record was admitted under env %r, today %r"
                         % (rec.get("env"), env))
    if os.path.isfile(lock_path):
        lock = common.read_json(lock_path)
    else:
        lock = {"schema": "cad-templates-lock/1", "templates": []}
    tid = rec.get("template_id")
    for e in lock["templates"]:
        if e["template_id"] == tid and e["source_sha256"] != src_sha:
            raise ValueError("FREEZE-IMMUTABLE: the lock holds %s at source_sha256 %r, refusing %r"
                             % (tid, e["source_sha256"], src_sha))
    tj = os.path.join(os.path.dirname(os.path.abspath(source_path)), "template.json")
    decl_sha = common.sha256_file(tj) if os.path.isfile(tj) else None
    entry = {"template_id": tid, "source": rec.get("source"), "source_sha256": src_sha,
             "declaration_sha256": decl_sha, "cadquery": env["cadquery"], "occt": env["occt"],
             "frozen_by": by, "admission_sha256": common.sha256_of(rec)}
    for k in LOCK_ENTRY_KEYS:
        if k not in entry:
            raise ValueError("FREEZE-IMMUTABLE: the entry lacks %s" % (k,))
    lock["templates"] = sorted([e for e in lock["templates"] if e["template_id"] != tid] + [entry],
                               key=lambda e: e["template_id"])
    common.write_json(lock_path, lock)
    return entry


def check_frozen(template_dir, lock_path=TEMPLATES_LOCK) -> dict:
    """The lock entry answering template_dir, or ValueError("TPL-UNFROZEN: ...")."""
    if not os.path.isfile(lock_path):
        raise ValueError("TPL-UNFROZEN: the templates lock %s does not exist" % (lock_path,))
    try:
        lock = common.read_json(lock_path)
    except (OSError, ValueError) as e:
        raise ValueError("TPL-UNFROZEN: the templates lock %s cannot be read: %s" % (lock_path, e))
    src = os.path.join(template_dir, "template.py")
    if not os.path.isfile(src):
        raise ValueError("TPL-UNFROZEN: %s holds no template.py" % (template_dir,))
    sha = common.sha256_file(src)
    entries = [e for e in lock.get("templates", [])
               if isinstance(e, dict) and e.get("source_sha256") == sha]
    if not entries:
        raise ValueError("TPL-UNFROZEN: the lock holds no entry for %s at source_sha256 %r"
                         % (template_dir, sha))
    entry = entries[0]
    tj = os.path.join(template_dir, "template.json")
    decl_sha = common.sha256_file(tj) if os.path.isfile(tj) else None
    if entry.get("declaration_sha256") != decl_sha:
        raise ValueError("TPL-UNFROZEN: the entry's declaration_sha256 %r differs from today's %r"
                         % (entry.get("declaration_sha256"), decl_sha))
    env = _env()
    if entry.get("cadquery") != env["cadquery"] or entry.get("occt") != env["occt"]:
        raise ValueError("TPL-UNFROZEN: the entry was frozen under cadquery %r occt %r, today"
                         " %r / %r" % (entry.get("cadquery"), entry.get("occt"), env["cadquery"],
                                       env["occt"]))
    return entry


# ---------------------------------------------------------------- the selftest
FIXTURE_DIR = os.path.join(HERE, "fixtures", "admit")
CHECK_TIMEOUT_S = 1740

T3_FIXTURES = (
    ("F01", None, "admitted", None, "inferred", ()),
    ("F02", "good_nozzle_candidate.py", "admitted", None, "declared", ()),
    ("F03", "good_pipe.py", "admitted", None, "declared", ()),
    ("F04", "good_pipe_cut.py", "admitted", None, "declared", ()),
    ("F05", "bad_lock_reqs.py", "refused", "ADM-AST-LOCK", None, ()),
    ("F06", "bad_import_json.py", "refused", "ADM-AST-IMPORT", None, ()),
    ("F07", "bad_name_open.py", "refused", "ADM-AST-NAME", None, ()),
    ("F08", "bad_fallback_fillet.py", "refused", "ADM-AST-FALLBACK", None, ()),
    ("F09", "bad_contract_primitive.py", "refused", "ADM-CONTRACT", None, ("throat_area",)),
    ("F10", "bad_build_raise.py", "refused", "ADM-BUILD", None, ()),
    ("F11", "bad_stage_shell.py", "refused", "ADM-STAGE", None, ("body.brep",)),
    ("F12", "bad_stage_cutter_miss.py", "refused", "ADM-STAGE", None, ("no-op", "cut")),
    ("F13", "bad_determ_outdir.py", "refused", "ADM-DETERM", None, ()),
    ("F14", "bad_tags_never.py", "refused", "ADM-TAGS", None, ("wall_lip",)),
    ("F15", "bad_insensitive_length.py", "refused", "ADM-INSENSITIVE", "declared",
     ("total_length", "L_over_D")),
)


def _row_real(name, mn, mx, dv):
    return {"name": name, "kind": "real", "unit": "m", "min": mn, "max": mx, "default_real": dv,
            "choices": [], "default_choice": None, "role": "design", "only_when": None}


def _fixture_path(rel_or_none):
    if rel_or_none is None:
        return os.path.join(NOZZLE_DIR, "template.py")
    return os.path.join(FIXTURE_DIR, rel_or_none)


def _one_line(rec):
    if rec.get("status") == "admitted":
        return "admitted (%s drivers)" % (rec.get("drivers_mode"),)
    return "refused %s: %s" % (rec.get("rule"), (rec.get("detail") or "")[:110])


def _check_child(source, out_path, timeout_s=CHECK_TIMEOUT_S):
    return subprocess.run([sys.executable, os.path.abspath(__file__), "check", source, out_path],
                          capture_output=True, text=True, encoding="utf-8", errors="replace",
                          env=dict(os.environ, PYTHONIOENCODING="utf-8"), timeout=timeout_s)


def selftest() -> int:
    t0 = time.monotonic()
    misses = []

    def ok(msg):
        print("[ok] " + msg)

    def fail(msg):
        misses.append(msg)
        print("[FAIL] " + msg)

    with tempfile.TemporaryDirectory(prefix="admit_selftest_", ignore_cleanup_errors=True) as td:
        try:    # T1
            assert RULES == ("ADM-AST-LOCK", "ADM-AST-IMPORT", "ADM-AST-NAME", "ADM-AST-FALLBACK",
                             "ADM-CONTRACT", "ADM-BUILD", "ADM-STAGE", "ADM-DETERM", "ADM-TAGS",
                             "ADM-INSENSITIVE"), RULES
            assert FREEZE_IDS == ("FREEZE-BY", "FREEZE-UNADMITTED", "FREEZE-IMMUTABLE"), FREEZE_IDS
            import measure
            bad = [p for p in ADMISSIBLE_PRIMITIVES if p not in measure.PRIMITIVES]
            assert not bad, bad
            assert set(RETRY_CLASS) == set(RULES), "RETRY_CLASS does not cover RULES"
            ok("T1 constants: RULES and FREEZE_IDS as decided, %d ADMISSIBLE_PRIMITIVES all in"
               " measure.PRIMITIVES, RETRY_CLASS covers RULES" % (len(ADMISSIBLE_PRIMITIVES),))
        except AssertionError as e:
            fail("T1 constants: %r" % (e,))

        try:    # T2
            decl = common.read_json(os.path.join(NOZZLE_DIR, "template.json"))
            pts, n_sob, n_cor = sweep_points(decl["params"])
            pts2, n_sob2, n_cor2 = sweep_points(decl["params"])
            assert (len(pts), n_sob, n_cor) == (97, 64, 32), (len(pts), n_sob, n_cor)
            assert pts[0] == nominal_params(decl["params"]), "the first point is not the nominal"
            assert pts == pts2 and (n_sob, n_cor) == (n_sob2, n_cor2), "two calls disagree"
            rows3 = [_row_real("a", 0.0, 1.0, 0.5), _row_real("b", 1.0, 3.0, 2.0),
                     _row_real("c", -1.0, 1.0, 0.0)]
            p3, s3, c3 = sweep_points(rows3)
            assert (len(p3), s3, c3) == (73, 64, 8), (len(p3), s3, c3)
            rows6 = [_row_real("r%d" % i, 0.0, 1.0, 0.5) for i in range(6)]
            p6, s6, c6 = sweep_points(rows6)
            assert (len(p6), s6, c6) == (97, 64, 32) and c6 == CORNER_CAP, (len(p6), s6, c6)
            ok("T2 sweep: the nozzle's box gives 97 = 1 + 64 + 32 points, first the nominal, two"
               " calls equal; 3 bounded reals 73; 6 bounded reals capped at 32 corners")
        except AssertionError as e:
            fail("T2 sweep: %r" % (e,))

        try:    # T6
            cases = (
                ("import os" + chr(10), "ADM-AST-IMPORT"),
                ("from tools.cad import reqs" + chr(10), "ADM-AST-LOCK"),
                ("x.__class__" + chr(10), "ADM-AST-NAME"),
                ("try:" + chr(10) + "    body = body.fillet(0.01)" + chr(10)
                 + "except Exception:" + chr(10) + "    raise" + chr(10), None),
                ("try:" + chr(10) + "    body = body.fillet(0.01)" + chr(10)
                 + "except Exception:" + chr(10) + "    body = rebuild(body)" + chr(10)
                 + "    return body" + chr(10), None),
                ("from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut" + chr(10), "ADM-AST-IMPORT"),
            )
            for text, want in cases:
                got = _ast_refusal(ast.parse(text))
                assert (got[0] if got else None) == want, (text, got, want)
            ok("T6 static rules on inline sources: import os and an OCP.BRepAlgoAPI cut are"
               " ADM-AST-IMPORT, tools.cad.reqs is ADM-AST-LOCK, a dunder attribute is"
               " ADM-AST-NAME, and a re-raising or rebuilding handler is not ADM-AST-FALLBACK")
        except AssertionError as e:
            fail("T6 static rules: %r" % (e,))

        try:    # T3
            order = [(tag, _fixture_path(rel), ws, wr, wm, words)
                     for tag, rel, ws, wr, wm, words in T3_FIXTURES]
            from concurrent.futures import ThreadPoolExecutor

            def _run_one(item):
                tag, src, _ws, _wr, _wm, _words = item
                out = os.path.join(td, tag + ".json")
                try:
                    _check_child(src, out)
                    return tag, None, common.read_json(out)
                except Exception as e:
                    return tag, repr(e), {}

            with ThreadPoolExecutor(max_workers=ADMIT_JOBS) as pool:
                got = dict((tag, (err, rec))
                           for tag, err, rec in pool.map(_run_one, order))
            for tag, src, ws, wr, wm, words in order:
                try:
                    err, rec = got[tag]
                    assert err is None, err
                    assert rec.get("status") == ws, "status %r, want %r" % (rec.get("status"), ws)
                    assert rec.get("rule") == wr, "rule %r, want %r" % (rec.get("rule"), wr)
                    assert rec.get("drivers_mode") == wm, \
                        "drivers_mode %r, want %r" % (rec.get("drivers_mode"), wm)
                    for w in words:
                        assert w in (rec.get("detail") or ""), "%r not in %r" % (w, rec.get("detail"))
                    if tag == "F10":
                        assert (rec.get("point") or {}).get("t_wall", 0.0) > 0.008, rec.get("point")
                    if ws == "admitted":
                        assert rec.get("detail") is None and rec.get("rule") is None, rec.get("detail")
                    ok("T3 %s %s %s" % (tag, os.path.basename(src), _one_line(rec)))
                except AssertionError as e:
                    fail("T3 %s: %r" % (tag, e))
        except Exception as e:
            fail("T3 pool: %r" % (e,))

        try:    # T4
            f03 = os.path.join(FIXTURE_DIR, "good_pipe.py")
            lock_a = os.path.join(td, "lock_a.json")
            p = subprocess.run([sys.executable, os.path.abspath(__file__), "freeze", f03,
                                os.path.join(td, "F03.json"), "--lock", lock_a],
                               capture_output=True, text=True, encoding="utf-8", errors="replace",
                               env=dict(os.environ, PYTHONIOENCODING="utf-8"), timeout=300)
            assert p.returncode == 1 and p.stderr.startswith("FREEZE-BY"), \
                (p.returncode, p.stderr[:80])
            ok("T4 freeze: the CLI without --by exits 1 with FREEZE-BY on stderr")
            lock_b = os.path.join(td, "lock_b.json")
            try:
                freeze(os.path.join(FIXTURE_DIR, "bad_import_json.py"),
                       os.path.join(td, "F06.json"), "tester", lock_b)
            except ValueError as e:
                assert str(e).startswith("FREEZE-UNADMITTED"), str(e)
            else:
                raise AssertionError("a refused record was frozen")
            ok("T4 freeze: F06's refused record is FREEZE-UNADMITTED")
            cp = os.path.join(td, "good_pipe_plus.py")
            shutil.copyfile(f03, cp)
            with open(cp, "a", encoding="utf-8") as f:
                f.write(chr(10))
            try:
                freeze(cp, os.path.join(td, "F03.json"), "tester", lock_b)
            except ValueError as e:
                assert str(e).startswith("FREEZE-UNADMITTED"), str(e)
            else:
                raise AssertionError("a moved source was frozen")
            ok("T4 freeze: one appended byte on a temp copy moves the sha, FREEZE-UNADMITTED")
            lock_d = os.path.join(td, "lock_d.json")
            rec_f03 = common.read_json(os.path.join(td, "F03.json"))
            entry = freeze(f03, os.path.join(td, "F03.json"), "tester", lock_d)
            assert tuple(entry) == LOCK_ENTRY_KEYS, tuple(entry)
            assert entry["template_id"] == "pipe_straight/1", entry["template_id"]
            assert entry["source_sha256"] == common.sha256_file(f03), entry["source_sha256"]
            assert entry["cadquery"] == _env()["cadquery"] and entry["occt"] == _env()["occt"]
            assert entry["frozen_by"] == "tester", entry["frozen_by"]
            assert entry["declaration_sha256"] is None, entry["declaration_sha256"]
            assert entry["admission_sha256"] == common.sha256_of(rec_f03)
            ok("T4 freeze: F03's entry carries template_id, source_sha256, cadquery %r, occt %r,"
               " frozen_by and admission_sha256" % (entry["cadquery"], entry["occt"]))
            with open(lock_d, "rb") as f:
                before = f.read()
            freeze(f03, os.path.join(td, "F03.json"), "tester", lock_d)
            with open(lock_d, "rb") as f:
                after = f.read()
            assert before == after, "an identical freeze changed the lock bytes"
            ok("T4 freeze: an identical freeze leaves the lock bytes unchanged")
            f04 = os.path.join(FIXTURE_DIR, "good_pipe_cut.py")
            with open(f04, "r", encoding="utf-8") as f:
                text = f.read()
            claim = text.replace('TEMPLATE_ID = "pipe_cut/1"', 'TEMPLATE_ID = "pipe_straight/1"')
            assert claim != text, "the F04 TEMPLATE_ID edit did not take"
            f04c = os.path.join(td, "pipe_straight_claim.py")
            with open(f04c, "w", encoding="utf-8", newline="") as f:
                f.write(claim)
            rec4 = os.path.join(td, "F04_claim.json")
            _check_child(f04c, rec4)
            got4 = common.read_json(rec4)
            assert got4["status"] == "admitted", _one_line(got4)
            try:
                freeze(f04c, rec4, "tester", lock_d)
            except ValueError as e:
                assert str(e).startswith("FREEZE-IMMUTABLE"), str(e)
            else:
                raise AssertionError("a second source at the same id was frozen")
            ok("T4 freeze: an F04 copy claiming pipe_straight/1 is admitted, then FREEZE-IMMUTABLE")
        except AssertionError as e:
            fail("T4 freeze: %r" % (e,))

        try:    # T5
            entry = check_frozen(NOZZLE_DIR)
            assert entry["template_id"] == "nozzle_contraction/1", entry["template_id"]
            lock_e = os.path.join(td, "lock_empty.json")
            common.write_json(lock_e, {"schema": "cad-templates-lock/1", "templates": []})
            try:
                check_frozen(NOZZLE_DIR, lock_e)
            except ValueError as e:
                assert str(e).startswith("TPL-UNFROZEN:"), str(e)
            else:
                raise AssertionError("an empty lock answered the nozzle directory")
            committed = common.read_json(TEMPLATES_LOCK)
            wrong = {"schema": "cad-templates-lock/1",
                     "templates": [dict(e, source_sha256="0" * 64)
                                   if e["template_id"] == "nozzle_contraction/1" else dict(e)
                                   for e in committed["templates"]]}
            lock_f = os.path.join(td, "lock_wrong_sha.json")
            common.write_json(lock_f, wrong)
            try:
                check_frozen(NOZZLE_DIR, lock_f)
            except ValueError as e:
                assert str(e).startswith("TPL-UNFROZEN:"), str(e)
            else:
                raise AssertionError("a changed source_sha256 answered")
            ok("T5 check_frozen: the committed templates.lock answers %s; an empty lock and a"
               " changed source_sha256 both refuse TPL-UNFROZEN"
               % os.path.basename(NOZZLE_DIR))
        except (AssertionError, OSError, ValueError) as e:
            fail("T5 check_frozen: %r (the committed lock must exist before finishing)" % (e,))

        try:    # T7
            f01_path = os.path.join(td, "F01.json")
            rec_n = common.read_json(f01_path)
            assert tuple(rec_n) == ADMISSION_KEYS, tuple(rec_n)
            with open(f01_path, "r", encoding="utf-8") as f:
                text = f.read()
            assert common.REPO not in text, "the repo absolute path leaked into the record"
            assert not re.search(r"[A-Za-z]:[\\/]", text), "a drive-letter path leaked"
            _check_child(os.path.join(FIXTURE_DIR, "good_pipe.py"), os.path.join(td, "F03b.json"))
            with open(os.path.join(td, "F03.json"), "rb") as f:
                a = f.read()
            with open(os.path.join(td, "F03b.json"), "rb") as f:
                b = f.read()
            assert a == b, "two checks of F03 wrote different record bytes"
            ok("T7 record: the nozzle's record has exactly ADMISSION_KEYS, no absolute path in its"
               " JSON text, and a second check of F03 writes byte-identical record bytes")
        except AssertionError as e:
            fail("T7 record: %r" % (e,))

        try:    # T8
            f03_path = os.path.join(FIXTURE_DIR, "good_pipe.py")
            with open(f03_path, "r", encoding="utf-8") as f:
                text = f.read()
            needle = "def domain_rules(p):" + chr(10)
            claim = text.replace(needle, needle
                                 + '    if p.get("t_wall", 0.0) > 0.008:' + chr(10)
                                 + '        raise RuntimeError("probe: domain_rules raises")'
                                 + chr(10))
            assert claim != text, "the domain_rules insertion did not take"
            src8 = os.path.join(td, "good_pipe_raise.py")
            with open(src8, "w", encoding="utf-8", newline="") as f:
                f.write(claim)
            rec8 = os.path.join(td, "T8.json")
            _check_child(src8, rec8)
            got8 = common.read_json(rec8)
            assert got8["status"] == "refused", _one_line(got8)
            assert got8["rule"] == "ADM-BUILD", got8["rule"]
            m = re.match(r"point (\d+): ", got8.get("detail") or "")
            assert m, got8.get("detail")
            assert "domain_rules raises" in (got8.get("detail") or ""), got8["detail"]
            assert (got8.get("point") or {}).get("t_wall", 0.0) > 0.008, got8.get("point")
            assert got8.get("traceback"), "no traceback in the record"
            ok("T8 a domain_rules that raises at a sweep point is ADM-BUILD at point %s"
               " (t_wall %g)" % (m.group(1), got8["point"]["t_wall"]))
        except AssertionError as e:
            fail("T8 raise point: %r" % (e,))

    print("selftest wall %.1f s" % (time.monotonic() - t0,))
    if misses:
        print("SELFTEST FAIL (%d miss(es))" % (len(misses),))
        return 1
    print("SELFTEST PASS")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ["--selftest"]:
        return selftest()
    if not argv:
        sys.stderr.write(USAGE)
        return 2
    verb, args = argv[0], argv[1:]
    if verb == "check":
        if len(args) != 2:
            sys.stderr.write(USAGE)
            return 2
        rec = admit(args[0])
        common.write_json(args[1], rec)
        print(common.canonical_json({"status": rec["status"], "rule": rec["rule"],
                                     "detail": rec["detail"]}))
        return 0 if rec["status"] == "admitted" else 1
    if verb == "freeze":
        by = None
        lock = TEMPLATES_LOCK
        pos = []
        i = 0
        while i < len(args):
            if args[i] in ("--by", "--lock"):
                if i + 1 >= len(args):
                    sys.stderr.write(USAGE)
                    return 2
                if args[i] == "--by":
                    by = args[i + 1]
                else:
                    lock = args[i + 1]
                i += 2
            else:
                pos.append(args[i])
                i += 1
        if len(pos) != 2:
            sys.stderr.write(USAGE)
            return 2
        try:
            entry = freeze(pos[0], pos[1], by, lock)
        except ValueError as e:
            sys.stderr.write("%s%s" % (e, chr(10)))
            return 1
        print(common.canonical_json(entry))
        return 0
    if verb == "frozen":
        lock = TEMPLATES_LOCK
        pos = []
        i = 0
        while i < len(args):
            if args[i] == "--lock":
                if i + 1 >= len(args):
                    sys.stderr.write(USAGE)
                    return 2
                lock = args[i + 1]
                i += 2
            else:
                pos.append(args[i])
                i += 1
        if len(pos) != 1:
            sys.stderr.write(USAGE)
            return 2
        try:
            entry = check_frozen(pos[0], lock)
        except ValueError as e:
            sys.stderr.write("%s%s" % (e, chr(10)))
            return 1
        print(common.canonical_json(entry))
        return 0
    sys.stderr.write(USAGE)
    return 2


if __name__ == "__main__":
    sys.exit(main())
