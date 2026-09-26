#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""
preflight.py - L0: refuse a config before any run (AM-8, docs/15 §C).

Nine checks (a)-(h) in a fixed order, each refusal a DecisionRecord naming
its rule id, the value against its limit and a cite:

  (a) PF-SURFACE  the STL exists, stl_repair reads it, it is closed and
                  consistently wound (outward orientation is NOT required:
                  the mesher classifies castellation by parity).
  (b) PF-QUALITY  the quality block is byte-equal to REFERENCE_QUALITY.
  (c) PF-FLAGS    no forbidden command-line flag, through schema.check_flags.
  (d) PF-KNOBS    every knob inside schema/knobs.json, through
                  schema.check_edit; the patch names against the STL; and
                  PF-CONFIG, a mirror of the mesher's own argument parser,
                  serde parser (document order, deny_unknown_fields) and
                  validate (mod.rs:526) plus its surface read - so a config
                  the mirror refuses is refused here before any run.
                  WL-SHARP-FT0 (2026-09-26) refuses feature_tolerance 0 on a
                  body with sharp edges off the R-PLANE path.
  (e) PF-NONORTH  no quality ceiling under the 25.2394 deg floor a 2:1
                  octree transition sets (atan(sqrt(2)/3)).
  (f) PF-YPLUS    the §D.3 y+ window non-empty at some level 0..6, and
      PF-THIN     the (92.51) edge 3*t1/h >= min_thickness_ratio on the
                  config's own first layer.
  (g) PF-DOMAIN   stage 0's domain margin (automesher.rs:509), which
                  -dryRun does not run.
  (h) PF-BUDGET   the octree probe's n_leaves within gates.json's budget.

G-PREFLIGHT (docs/15 §F): --gate compares the mirror with the binary's
-dryRun on 10,000 random configs, reproduces the thin_t1 refusal bit for
bit, and reports the castellated h prediction against 48 real stage-5
outcomes. stdlib only. No GPL-licensed source was consulted.
"""

import copy
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import schema  # noqa: E402
import score  # noqa: E402

BINARY_DEFAULT = score.BINARY_DEFAULT
STL_REPAIR = os.path.join(REPO, "tools", "geom", "stl_repair.py")
BOX_SPHERE_GEN = os.path.join(REPO, "tools", "automesher", "examples",
                              "make_box_sphere_stl.py")
PROBES_DIR = os.path.join(HERE, "fixtures", "probes")
CUBEP_STL = os.path.join(HERE, "fixtures", "stl", "cubep.stl")
REPORT_DIR = os.path.join(HERE, "preflight")
RESULT_SCHEMA = "autonomy-preflight/1"
CHECKS = (("a", "PF-SURFACE"), ("b", "PF-QUALITY"), ("c", "PF-FLAGS"), ("d", "PF-KNOBS"),
          ("e", "PF-NONORTH"), ("f", "PF-YPLUS"), ("f", "PF-THIN"), ("g", "PF-DOMAIN"),
          ("h", "PF-BUDGET"))
REFUSAL_IDS = ("PF-SURFACE", "PF-QUALITY", "WL-FLAG", "WL-POINTER", "WL-FORBIDDEN", "WL-UNLISTED",
               "WL-TYPE", "WL-RANGE", "PF-PATCH", "PF-CONFIG", "PF-NONORTH", "PF-YPLUS",
               "PF-THIN", "PF-DOMAIN", "PF-BUDGET", "WL-SHARP-FT0")
CHECK_OF = {"WL-FLAG": "PF-FLAGS", "WL-POINTER": "PF-KNOBS", "WL-FORBIDDEN": "PF-KNOBS",
            "WL-UNLISTED": "PF-KNOBS", "WL-TYPE": "PF-KNOBS", "WL-RANGE": "PF-KNOBS",
            "PF-PATCH": "PF-KNOBS", "PF-CONFIG": "PF-KNOBS", "WL-SHARP-FT0": "PF-KNOBS"}
NON_ORTH_FLOOR_DEG = math.degrees(math.atan(math.sqrt(2.0) / 3.0))   # 25.239401820678918
G5_FACTOR = 3.0                                                      # layers.rs:1306
REFERENCE_QUALITY = {"max_closure": 1e-10, "max_non_orth_deg": 70.0,
                     "report_non_orth_deg": 60.0, "min_thickness_ratio": 0.05,
                     "max_cond": 10000.0}
G5_LINE_RE = re.compile(r"3 \* (\S+) / (\S+) = (\S+) < min_thickness_ratio = (\S+) \(92\.51\)")
ENUM_VALUES = ("largest", "seed")
TEMPLATE_FREE = ("/castellation/keep_region", "/castellation/seed_point")
BOX_PATCH_NAMES = ("xMin", "xMax", "yMin", "yMax", "zMin", "zMax")
FLOW_OK = {"u_ref_m_s": 1.0, "l_ref_m": 1.0, "nu_m2_s": 1.5e-5}
FLOW_FAST = {"u_ref_m_s": 10.0, "l_ref_m": 1.0, "nu_m2_s": 1.5e-5}
_SURFACE_CACHE = {}


class PreflightError(ValueError):
    """A caller or harness error (bad argument, unreadable report) - never a verdict."""


class _ParseErr(Exception):
    """The mirror's serde: the first field the binary's parser would stop at."""

    def __init__(self, field, message):
        super().__init__("%s: %s" % (field, message))
        self.field = field
        self.message = message


# (C2) the mesher's config tree, cross-checked against the binary's -schema.
# Each struct: (field, kind, required) in DECLARATION order - the order serde
# checks missing fields in. 13 structs, 55 fields.
STRUCTS = {
 "AutomeshConfig": [("$schema", "opt_str", False), ("input", "InputSpec", True),
     ("domain", "DomainSpec", True), ("refinement", "RefinementSpec", False),
     ("castellation", "CastellationSpec", False), ("snap", "SnapSpec", False),
     ("layers", "LayerSpec", False), ("quality", "QualitySpec", False),
     ("output", "OutputSpec", True)],
 "InputSpec": [("surfaces", "list:SurfaceInput", True)],
 "SurfaceInput": [("path", "str", True), ("name", "opt_str", False)],
 "DomainSpec": [("extent", "f64x6", True), ("base_size", "f64", True),
     ("grading", "f64x3", False)],
 "RefinementSpec": [("levels", "list:RefinementBand", False), ("feature_angle_deg", "f64", False),
     ("max_level", "u32", False)],
 "RefinementBand": [("patch", "str", True), ("bands", "list:DistanceBand", True),
     ("feature_level", "u32", False)],
 "DistanceBand": [("distance", "f64", True), ("level", "u32", True)],
 "CastellationSpec": [("keep_region", "enum", False), ("seed_point", "opt_f64x3", False),
     ("min_faces", "usize", False), ("bodies", "list:BodySpec", False)],
 "BodySpec": [("name", "str", True), ("patches", "list:str", True)],
 "SnapSpec": [("iterations", "usize", False), ("tolerance", "f64", False),
     ("smoothing_passes", "usize", False), ("smoothing", "f64", False),
     ("undo_limit", "usize", False), ("max_area_ratio", "f64", False),
     ("feature_tolerance", "f64", False)],
 "LayerSpec": [("patches", "list:str", False), ("n", "usize", False),
     ("first_thickness", "f64", False), ("growth", "f64", False),
     ("min_thickness", "f64", False), ("medial_frac", "f64", False),
     ("normal_passes", "usize", False), ("cell_frac", "f64", False),
     ("smoothing", "f64", False), ("smoothing_passes", "usize", False),
     ("retreat_limit", "usize", False)],
 "QualitySpec": [("max_closure", "f64", False), ("max_non_orth_deg", "f64", False),
     ("report_non_orth_deg", "f64", False), ("min_thickness_ratio", "f64", False),
     ("max_cond", "f64", False)],
 "OutputSpec": [("case_dir", "str", True), ("name", "str", True),
     ("patch_names", "map:str", False)],
}


# (C3) MESHER_DEFAULTS - the binary's own defaults, cross-checked in
# selftest group 1 against -schema (31 comparisons).
MESHER_DEFAULTS = {
 "/domain/grading": [1.0, 1.0, 1.0],
 "/refinement/levels": [], "/refinement/feature_angle_deg": 30.0, "/refinement/max_level": 2,
 "/refinement/levels/*/feature_level": 0,
 "/castellation/keep_region": "largest", "/castellation/seed_point": None,
 "/castellation/min_faces": 4,
 "/castellation/bodies": [],
 "/snap/iterations": 30, "/snap/tolerance": 0.001, "/snap/smoothing_passes": 3,
 "/snap/smoothing": 0.5,
 "/snap/undo_limit": 4, "/snap/max_area_ratio": 4.0, "/snap/feature_tolerance": 0.5,
 "/layers/patches": [], "/layers/n": 0, "/layers/first_thickness": 0.05, "/layers/growth": 1.3,
 "/layers/min_thickness": 0.1, "/layers/medial_frac": 0.5, "/layers/normal_passes": 3,
 "/layers/cell_frac": 0.5,
 "/layers/smoothing": 0.5, "/layers/smoothing_passes": 4, "/layers/retreat_limit": 4,
 "/quality/max_closure": 1e-10, "/quality/max_non_orth_deg": 70.0,
 "/quality/report_non_orth_deg": 60.0, "/quality/min_thickness_ratio": 0.05,
 "/quality/max_cond": 10000.0,
}


def _last_line(text):
    """The last non-empty line of a child's output, or None."""
    out = None
    for l in (text or "").replace("\r\n", "\n").split("\n"):
        if l.strip():
            out = l.strip()
    return out


def _last_error_line(text):
    """The content of the last line starting `error: `, without the prefix."""
    out = None
    for l in (text or "").replace("\r\n", "\n").split("\n"):
        if l.startswith("error: "):
            out = l[len("error: "):].rstrip()
    return out


def _stl_report(path):
    """One stl_repair --json run as the surface oracle, cached by file stamp."""
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_size, st.st_mtime_ns)
    hit = _SURFACE_CACHE.get(key)
    if hit is not None:
        return hit
    tmp = tempfile.mkdtemp(prefix="preflight_stl_")
    try:
        out = os.path.join(tmp, "report.json")
        try:
            p = subprocess.run([sys.executable, STL_REPAIR, path, "--json", out],
                               capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=300)
            rc, errtext = p.returncode, p.stderr or ""
        except (OSError, subprocess.TimeoutExpired) as e:
            rc, errtext = -1, "stl_repair failed: %s" % e
        rep = None
        if rc == 0 and os.path.isfile(out):
            try:
                with open(out, encoding="utf-8") as fh:
                    rep = json.load(fh)
            except (OSError, ValueError):
                rep = None
        entry = _stl_entry(path, rep, rc, errtext)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    _SURFACE_CACHE[key] = entry
    return entry


def _stl_entry(path, rep, rc, errtext):
    """The per-surface fact dict of (C6); report fields None when not ok."""
    base = {"path": path, "abspath": os.path.abspath(path), "exists": True,
            "ok": rep is not None, "error": None, "open_edges": None,
            "non_manifold_edges": None, "closed": None, "reoriented_triangles": None,
            "flipped_components": None, "patches": None, "bbox": None, "triangles": None}
    if rep is None:
        base["error"] = _last_line(errtext) or "stl_repair rc=%s" % rc
        return base
    before = rep.get("before") or {}
    ori = rep.get("orientation") or {}
    base.update(open_edges=before.get("open_edges"),
                non_manifold_edges=before.get("non_manifold_edges"),
                closed=before.get("closed"),
                reoriented_triangles=ori.get("reoriented_triangles"),
                flipped_components=ori.get("flipped_components"),
                patches=rep.get("patches"), bbox=rep.get("bbox"),
                triangles=rep.get("triangles_in"))
    return base


def surface_facts(config, cwd=None):
    """(C6): stl_repair's report per input.surfaces entry, merged (cached)."""
    cwd = cwd or os.getcwd()
    inp = config.get("input") if isinstance(config, dict) else None
    surfs = inp.get("surfaces") if isinstance(inp, dict) else None
    wanted = []
    if isinstance(surfs, list):
        for s in surfs:
            if not (isinstance(s, dict) and isinstance(s.get("path"), str)):
                wanted = []
                break
            wanted.append(s)
    entries, names = [], []
    for s in wanted:
        p = s["path"]
        ap = p if os.path.isabs(p) else os.path.join(cwd, p)
        if not os.path.isfile(ap):
            miss = _stl_entry(p, None, -1, "the file does not exist")
            miss["exists"] = False
            entries.append(miss)
            continue
        rep = _stl_report(ap)
        entries.append(rep)
        if isinstance(s.get("name"), str):
            names.append(s["name"])
        elif rep["ok"]:
            names.extend(rep["patches"] or [])
    ok = [e for e in entries if e["ok"]]
    bbox = None
    if ok:
        lo = [min(e["bbox"][i] for e in ok) for i in range(3)]
        hi = [max(e["bbox"][3 + i] for e in ok) for i in range(3)]
        bbox = lo + hi
    return {"surfaces": entries, "patches": names, "bbox": bbox,
            "open_edges": sum(e["open_edges"] or 0 for e in ok),
            "non_manifold_edges": sum(e["non_manifold_edges"] or 0 for e in ok),
            "readable": all(e["exists"] and e["ok"] for e in entries)}


def _type_desc(v):
    """serde's name for a JSON value's type, as its error lines carry it."""
    if isinstance(v, bool):
        return "boolean `%s`" % ("true" if v else "false")
    if isinstance(v, float):
        return "floating point `%r`" % v
    if isinstance(v, int):
        return "integer `%d`" % v
    if v is None:
        return "null"
    if isinstance(v, str):
        return "string %r" % v
    if isinstance(v, list):
        return "sequence"
    return "map"


def _parse(value, kind, path):
    """(C4): serde's parse of `value` as `kind` at field `path`; _ParseErr first."""
    if kind == "f64":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _ParseErr(path, "invalid type: %s, expected a number" % _type_desc(value))
        return
    if kind in ("usize", "u32"):
        top = 2 ** 64 - 1 if kind == "usize" else 2 ** 32 - 1
        if isinstance(value, bool) or isinstance(value, float) or not isinstance(value, int):
            raise _ParseErr(path, "invalid type: %s, expected %s" % (_type_desc(value), kind))
        if not 0 <= value <= top:
            raise _ParseErr(path, "invalid value: integer `%d`, expected %s" % (value, kind))
        return
    if kind in ("str", "opt_str"):
        if not (isinstance(value, str) or (kind == "opt_str" and value is None)):
            raise _ParseErr(path, "invalid type: %s, expected a string"
                            % _type_desc(value))
        return
    if kind == "enum":
        if not (isinstance(value, str) and value in ENUM_VALUES):
            raise _ParseErr(path, "invalid type: %s, expected %r or %r"
                            % (_type_desc(value), ENUM_VALUES[0], ENUM_VALUES[1]))
        return
    if kind in ("f64x3", "f64x6"):
        n = int(kind[4:])
        if not isinstance(value, list):
            raise _ParseErr(path, "invalid type: %s, expected a sequence of %d"
                            % (_type_desc(value), n))
        for i, v in enumerate(value[:n]):
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise _ParseErr(path + "[%d]" % i,
                                "invalid type: %s, expected a number" % _type_desc(v))
        if len(value) != n:
            raise _ParseErr(path, "invalid length %d, expected %d" % (len(value), n))
        return
    if kind == "opt_f64x3":
        if value is None:
            return
        return _parse(value, "f64x3", path)
    if kind == "list:str":
        if not isinstance(value, list):
            raise _ParseErr(path, "invalid type: %s, expected a sequence" % _type_desc(value))
        for i, v in enumerate(value):
            if not isinstance(v, str):
                raise _ParseErr(path + "[%d]" % i,
                                "invalid type: %s, expected a string" % _type_desc(v))
        return
    if kind == "map:str":
        if not isinstance(value, dict):
            raise _ParseErr(path, "invalid type: %s, expected a map" % _type_desc(value))
        for k in sorted(value):
            v = value[k]
            if not isinstance(v, str):
                raise _ParseErr(path + "." + k,
                                "invalid type: %s, expected a string" % _type_desc(v))
        return
    if kind.startswith("list:"):
        item = kind[5:]
        if not isinstance(value, list):
            raise _ParseErr(path, "invalid type: %s, expected a sequence" % _type_desc(value))
        for i, v in enumerate(value):
            _parse(v, item, path + "[%d]" % i)
        return
    # a struct: serde_json::Value is a BTreeMap, so keys are visited in
    # lexicographic order (case_json.rs routes the document through a Value
    # before serde sees it), then the first REQUIRED field absent, in
    # declaration order.
    if not isinstance(value, dict):
        raise _ParseErr(path or ".", "invalid type: %s, expected struct %s"
                        % (_type_desc(value), kind))
    fields = dict((f[0], f[1]) for f in STRUCTS[kind])
    decl = STRUCTS[kind]
    for key in sorted(value):
        where = key if path == "" else path + "." + key
        if key not in fields:
            raise _ParseErr(where, "unknown field `%s`" % key)
        _parse(value[key], fields[key], where)
    for fname, _, req in decl:
        if req and fname not in value:
            where = fname if path == "" else path + "." + fname
            raise _ParseErr(where, "missing field `%s`" % fname)
    return


_RUNID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def _mirror_argv(argv):
    """(C4 stage 1): the tokens after CONFIG, as automesher.rs 111-208 reads them."""
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "-dryRun":
            pass
        elif a == "-tag":
            i += 1
            if i >= len(argv):
                return "-tag needs a NAME argument"
            v = argv[i]
            if v == "" or "/" in v or "\\" in v:
                return ("-tag: '%s' is a NAME - a suffix, not a path; an empty NAME "
                        "or one containing / or \\ is refused" % v)
        elif a == "-runId":
            i += 1
            if i >= len(argv):
                return "-runId needs an ID argument"
            if not _RUNID_RE.match(argv[i]):
                return ("-runId: '%s' is not a run id - 1 to 64 characters from "
                        "[A-Za-z0-9._-]" % argv[i])
        elif a == "-stopAfter":
            i += 1
            if i >= len(argv):
                return "-stopAfter needs a STAGE argument"
        elif a in ("-check", "-schema"):
            raise PreflightError("%s is another mode, never a meshing command line" % a)
        elif a.startswith("-"):
            return "unknown argument '%s'" % a
        else:
            return "one <config.json> positional is taken, got a second one '%s'" % a
        i += 1
    return None


def _patch_name_bad(s):
    """check_patch_name (polymesh.rs 893-917): why `s` cannot be a patch name."""
    if s == "":
        return "empty body name - a patch the boundary file carries has to be named"
    for c in s:
        if c.isspace() or unicodedata.category(c) == "Cc":
            return "body name %r carries a whitespace or control character" % s
    for c in s:
        if c in "{}();\"'":
            return ("body name %r carries %r, which the boundary file's patch entry "
                    "grammar reserves" % (s, c))
    return None


def _vget(cfg, struct, field, default):
    """The config's field, or the mesher default when the field is absent."""
    node = cfg.get(struct)
    if isinstance(node, dict) and field in node:
        return node[field]
    return default


def _mirror_validate(cfg):
    """(C4 stage 3): mod.rs 526-667's validate, in its own order."""
    e = _vget(cfg, "domain", "extent", None)
    for a, nm in ((0, "x"), (1, "y"), (2, "z")):
        if not (e[2 * a + 1] > e[2 * a]):
            return ("domain.extent", "%s-axis is empty or reversed (%slo = %r, %shi = %r)"
                    % (nm, nm, e[2 * a], nm, e[2 * a + 1]))
    bs = _vget(cfg, "domain", "base_size", None)
    if not bs > 0:
        return ("domain.base_size", "must be > 0, got %r" % (bs,))
    g = _vget(cfg, "domain", "grading", MESHER_DEFAULTS["/domain/grading"])
    if not (g[0] > 0 and g[1] > 0 and g[2] > 0):
        return ("domain.grading", "every axis must be > 0, got [%r, %r, %r]" % tuple(g))
    ml = _vget(cfg, "refinement", "max_level", 2)
    if ml > 6:
        return ("refinement.max_level",
                "%r exceeds the 6 levels SPEC-LIT §74.2 caps the octree at" % (ml,))
    mar = _vget(cfg, "snap", "max_area_ratio", 4.0)
    if not mar >= 1:
        return ("snap.max_area_ratio", "must be >= 1, got %r" % (mar,))
    sm = _vget(cfg, "snap", "smoothing", 0.5)
    if not (0 <= sm <= 1):
        return ("snap.smoothing", "must lie in [0, 1], got %r" % (sm,))
    ft = _vget(cfg, "snap", "feature_tolerance", 0.5)
    if not (math.isfinite(ft) and ft >= 0):
        return ("snap.feature_tolerance", "must be >= 0 and finite, got %r" % (ft,))
    for f in ("max_closure", "max_non_orth_deg", "report_non_orth_deg",
              "min_thickness_ratio", "max_cond"):
        v = _vget(cfg, "quality", f, MESHER_DEFAULTS["/quality/" + f])
        if not v > 0:
            return ("quality." + f, "threshold must be > 0, got %r" % (v,))
    if not _vget(cfg, "input", "surfaces", None):
        return ("input.surfaces", "at least one STL is required, got none")
    gr = _vget(cfg, "layers", "growth", 1.3)
    if not gr > 0:
        return ("layers.growth", "must be > 0, got %r" % (gr,))
    if _vget(cfg, "castellation", "keep_region", "largest") == "seed" \
            and _vget(cfg, "castellation", "seed_point", None) is None:
        return ("castellation.seed_point",
                "keep_region = \"seed\" needs a seed_point, got none")
    bodies = _vget(cfg, "castellation", "bodies", [])
    seen = []
    for i, b in enumerate(bodies):
        fld = "castellation.bodies[%d].name" % i
        why = _patch_name_bad(b["name"])
        if why:
            return (fld, why)
        if b["name"] == "fluid":
            return (fld, "\"fluid\" is the fluid region's own name - a body cannot take it")
        if b["name"] in BOX_PATCH_NAMES:
            return (fld, "\"%s\" is a domain patch name - the six box patches are "
                         "xMin..zMax" % b["name"])
        if b["name"] in seen:
            return (fld, "\"%s\" is declared twice" % b["name"])
        seen.append(b["name"])
        if not b["patches"]:
            return ("castellation.bodies[%d].patches" % i,
                    "empty - a body is a set of STL patch names, got none")
        for other in bodies[:i]:
            for p in b["patches"]:
                if p in other["patches"]:
                    return ("castellation.bodies[%d].patches" % i,
                            "\"%s\" is also a patch of body \"%s\" - a patch belongs "
                            "to one body" % (p, other["name"]))
    return None


def mirror_dryrun(config, argv=(), surface=None):
    """(C4): what `-dryRun` will do - None, or the first error it stops at."""
    why = _mirror_argv(list(argv))
    if why is not None:
        return {"stage": "argv", "field": "argv", "message": why}
    try:
        _parse(config, "AutomeshConfig", "")
    except _ParseErr as e:
        return {"stage": "parse", "field": e.field, "message": e.message}
    bad = _mirror_validate(config)
    if bad is not None:
        return {"stage": "validate", "field": bad[0], "message": bad[1]}
    if surface is None:
        surface = surface_facts(config)
    for e in surface["surfaces"]:
        if not e["exists"]:
            return {"stage": "surface", "field": "io",
                    "message": "io error on %s: the file does not exist" % e["path"]}
        if not e["ok"]:
            return {"stage": "surface", "field": "stl",
                    "message": "%s: %s" % (e["path"], e["error"] or "stl_repair refused")}
    if surface["open_edges"] + surface["non_manifold_edges"] > 0:
        return {"stage": "surface", "field": "surface/closed",
                "message": "surface/closed: \"%d open edge(s), %d non-manifold edge(s)\" "
                           "is not supported by ofgpu"
                           % (surface["open_edges"], surface["non_manifold_edges"])}
    return None


_MISSING_FIELD_RE = re.compile(r"missing field `(\w[^`]*)`")


def dryrun_field(stderr, config_path, surface_paths=()):
    """(C5): the field of a real -dryRun refusal; None when it printed none."""
    lines = [l for l in (stderr or "").replace("\r\n", "\n").split("\n")
             if l.startswith("error: ")]
    if not lines:
        return None
    t = lines[-1][len("error: "):].rstrip()
    if t.startswith(("unknown argument", "-tag", "-runId", "-stopAfter")) \
            or "positional" in t:
        return "argv"
    if t.startswith("io error on "):
        return "io"
    if t.startswith(config_path + ": "):
        rest = t[len(config_path) + 2:]
        parts = rest.split(": ", 1)
        if len(parts) == 2:
            path, msg = parts
            m = _MISSING_FIELD_RE.match(msg)
            if m:
                return m.group(1) if path == "." else path + "." + m.group(1)
            return path
        return rest
    if t.startswith("surface/closed:"):
        return "surface/closed"
    for p in surface_paths:
        if t.startswith(p + ": "):
            return "stl"
    return t.split(": ", 1)[0]


def field_to_pointer(field):
    """`a.b[0].c` -> `/a/b/0/c`; `$schema` -> `/$schema`."""
    if field.startswith("$"):
        return "/" + field
    out = []
    for seg in field.split("."):
        bits = []
        while seg.endswith("]"):
            i = seg.rindex("[")
            bits.append(seg[i + 1:-1])
            seg = seg[:i]
        bits.append(seg)
        out.extend(bits)
    return "/" + "/".join(out)


def run_dryrun(binary, config_path, argv=(), timeout_s=60.0, surface_paths=()):
    """One live `-dryRun`; the field via dryrun_field, the last error line kept."""
    try:
        p = subprocess.run([binary, config_path, "-dryRun", *list(argv)],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout_s)
        rc, err = p.returncode, p.stderr
    except subprocess.TimeoutExpired as e:
        rc = None
        err = e.stderr if isinstance(e.stderr, str) else (e.stderr or b"").decode(
            "utf-8", "replace")
    except OSError as e:
        rc, err = -1, "error: %s" % e
    return {"exit_code": rc,
            "field": dryrun_field(err, config_path, surface_paths) if rc else None,
            "error": _last_error_line(err) if rc else None}


_LEAF_KINDS = ("f64", "usize", "u32", "str", "opt_str", "enum", "f64x3", "f64x6",
               "opt_f64x3", "list:str", "map:str")


def config_leaves(config):
    """(C7 d): every leaf-kind field present, in document order, as (pointer, value)."""
    out = []

    def walk(struct_name, node, ptr):
        if not isinstance(node, dict):
            return
        kinds = dict((f[0], f[1]) for f in STRUCTS[struct_name])
        for key in node:
            kind = kinds.get(key)
            if kind is None:
                continue
            if ptr == "" and key in ("$schema", "input", "output", "quality"):
                continue
            p = ptr + "/" + key
            if struct_name == "CastellationSpec" and key == "bodies":
                out.append((p, node[key]))
                continue
            if kind in _LEAF_KINDS:
                out.append((p, node[key]))
            elif kind.startswith("list:"):
                item = kind[5:]
                if item in STRUCTS and isinstance(node[key], list):
                    for i, it in enumerate(node[key]):
                        if isinstance(it, dict):
                            walk(item, it, p + "/%d" % i)
            elif kind in STRUCTS and isinstance(node[key], dict):
                walk(kind, node[key], p)

    walk("AutomeshConfig", config, "")
    return out


def _pointer_pattern(ptr):
    """The pointer with every all-digit segment replaced by `*`."""
    return "/".join("*" if s.isdigit() else s for s in ptr.split("/"))


def _eq_default(a, b):
    """Recursive default equality; a bool never equals a number."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_eq_default(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_eq_default(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


def _is_default(ptr, pat, value):
    if pat not in MESHER_DEFAULTS:
        return False
    return _eq_default(MESHER_DEFAULTS[pat], value)


def _forbidden_row(ptr, knobs):
    for f in knobs.get("forbidden", []):
        if ptr == f["pointer"] or (f.get("match") == "prefix"
                                   and ptr.startswith(f["pointer"] + "/")):
            return f
    return None


_KNOBS_CACHE = {}


def _knobs():
    """schema.load_knobs() once per process; the lock check runs once."""
    if not _KNOBS_CACHE:
        _KNOBS_CACHE.update(schema.load_knobs())
    return _KNOBS_CACHE


def h_wall_predicted(config, fingerprint=None):
    """(C7 f): the castellated prediction base_size/2**L for the layer patches."""
    out = {"h_m": None, "level": None, "patch": None, "features": None}
    try:
        dom = config["domain"]
        base = dom["base_size"]
        if isinstance(base, bool) or not isinstance(base, (int, float)) or not base > 0:
            return out
        ref = config.get("refinement")
        ref = ref if isinstance(ref, dict) else {}
        ml = ref.get("max_level", MESHER_DEFAULTS["/refinement/max_level"])
        if isinstance(ml, bool) or not isinstance(ml, int) or ml < 0:
            return out
        if fingerprint is None:
            features, fsrc = True, "assumed"
        else:
            sel = fingerprint["sharp_edge_length_m"]
            if isinstance(sel, bool) or not isinstance(sel, (int, float)):
                return out
            features, fsrc = sel > 0, "fingerprint"
        lay = config.get("layers")
        lay = lay if isinstance(lay, dict) else {}
        patches = lay.get("patches", [])
        levels = ref.get("levels", [])
        if not isinstance(patches, list) or not isinstance(levels, list):
            return out
        best = None
        for p in patches:
            ls = []
            for entry in levels:
                if not isinstance(entry, dict) or entry.get("patch") != p:
                    if not isinstance(entry, dict):
                        return out
                    continue
                for b in entry.get("bands", []) or []:
                    if not isinstance(b, dict):
                        return out
                    lv = b.get("level", 0)
                    if isinstance(lv, bool) or not isinstance(lv, int):
                        return out
                    ls.append(lv)
                fl = entry.get("feature_level", 0)
                if isinstance(fl, bool) or not isinstance(fl, int):
                    return out
                if features and fl > 0:
                    ls.append(fl)
            l_p = min(max(ls, default=0), ml)
            if best is None or l_p > best[0]:
                best = (l_p, p)
        if best is None:
            best = (0, None)
        # a huge (parse-legal u32) max_level must not build a 2**L integer: past
        # double range the predicted spacing simply underflows to zero.
        h = base / (2.0 ** best[0] if best[0] < 1024 else math.inf)
        out.update(h_m=h, level=best[0], patch=best[1], features=fsrc)
        return out
    except (TypeError, KeyError, AttributeError):
        return out


def yplus_window(t1_m, n, growth, cell_frac, ratio):
    """(C7 f): the §D.3 window (lo, hi) - the stack limiter below, (92.51) above."""
    s = float(n) if growth == 1.0 else (growth ** n - 1.0) / (growth - 1.0)
    return (t1_m * s / cell_frac, G5_FACTOR * t1_m / ratio)


def _rec(rule_id, verdict, message, cite, trigger, inputs=None, formula=""):
    """One autonomy-decision/1 record; abstains carry trigger None."""
    return {"schema": "autonomy-decision/1", "layer": "preflight", "rule_id": rule_id,
            "verdict": verdict, "trigger": trigger, "inputs": inputs or [],
            "formula": formula, "edits": [], "cite": cite, "message": message,
            "uncertainty": 0, "t": schema._now_iso()}


def _value_at(config, field):
    """The offending value at a mirror field, or None when unplaceable."""
    if not field or field in ("argv", "."):
        return None
    node = config
    for seg in field_to_pointer(field).split("/")[1:]:
        if isinstance(node, dict) and seg in node:
            node = node[seg]
        elif isinstance(node, list):
            try:
                node = node[int(seg)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return node


def _covers(records, field):
    """(C7 d 4): does an existing refusal already cover the mirror's field?"""
    if field == "argv":
        return any(r["rule_id"] == "WL-FLAG" for r in records)
    if field == "io" or field == "stl" or field == "surface/closed" \
            or field.startswith("input"):
        return any(r["rule_id"] == "PF-SURFACE" for r in records)
    if field == "quality" or field.startswith("quality."):
        return any(r["rule_id"] == "PF-QUALITY" for r in records)
    tp = field_to_pointer(field)
    for r in records:
        rid = r["rule_id"]
        if not (rid.startswith("WL-") or rid == "PF-PATCH"):
            continue
        obs = (r.get("trigger") or {}).get("observable")
        if obs == tp or (isinstance(obs, str) and tp.startswith(obs + "/")):
            return True
    return False


def _levels_max(config):
    """The (C7 e) L = min(max_level, every band level and feature_level); None malformed."""
    ref = config.get("refinement") if isinstance(config, dict) else None
    if ref is None:
        return 0
    if not isinstance(ref, dict):
        return None
    ml = ref.get("max_level", MESHER_DEFAULTS["/refinement/max_level"])
    if isinstance(ml, bool) or not isinstance(ml, int) or ml < 0:
        return None
    levels = ref.get("levels", MESHER_DEFAULTS["/refinement/levels"])
    if not isinstance(levels, list):
        return None
    vals = []
    for entry in levels:
        if not isinstance(entry, dict):
            return None
        fl = entry.get("feature_level", 0)
        if isinstance(fl, bool) or not isinstance(fl, int) or fl < 0:
            return None
        vals.append(fl)
        bands = entry.get("bands", [])
        if not isinstance(bands, list):
            return None
        for b in bands:
            if not isinstance(b, dict):
                return None
            bl = b.get("level", 0)
            if isinstance(bl, bool) or not isinstance(bl, int) or bl < 0:
                return None
            vals.append(bl)
    return min(ml, max(vals, default=0))


_CITE_A = "docs/15 §C L0 (a); tools/geom/stl_repair.py before/orientation; automesher.rs:433"
_CITE_B = "docs/15 §C L0 (b), §H: the loop never loosens the gate that judges it (SPEC-LIT §92.3)"
_CITE_E = "docs/15 §C L0 (e); docs/15 §B fact 4"
_CITE_Y = "docs/15 §D.3 (R-WIN); schema.a_priori_wall (SPEC-LIT §32.5.6)"
_CITE_T = "layers.rs:1292-1316 (92.51); docs/15 §D.3 caution 1"
_CITE_G = "automesher.rs:509 (stage 0); SPEC-LIT §92.2"
_CITE_H = "gates.json cell_budget; docs/15 §D.1 F5"


def _quality_block(config):
    """(q_eff, effective, raw) for checks (b)/(e)/(f): q_eff is None unless the
    raw block is absent or a dict whose values are all numbers."""
    q = config.get("quality") if isinstance(config, dict) else None
    d = dict((k[len("/quality/"):], v) for k, v in MESHER_DEFAULTS.items()
              if k.startswith("/quality/"))
    if q is None:
        return d, d, None
    if isinstance(q, dict):
        eff = dict(d)
        eff.update(q)
        numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool)
                      for v in q.values())
        return (eff if numeric else None), eff, q
    return None, None, q


def _pf_surface_refuse_or_pass(surface):
    """(a) the first failing per-surface condition, else the pass record."""
    for i, e in enumerate(surface["surfaces"]):
        obs = "/input/surfaces/%d/path" % i
        if not e["exists"]:
            return _rec("PF-SURFACE", "refuse",
                        "PF-SURFACE: %s: the STL is missing" % e["path"], _CITE_A,
                        {"observable": obs, "value": False, "threshold": True,
                         "op": "==", "source": "os.path.isfile"})
        if not e["ok"]:
            return _rec("PF-SURFACE", "refuse",
                        "PF-SURFACE: %s: stl_repair could not read it: %s"
                        % (e["path"], e["error"]), _CITE_A,
                        {"observable": obs, "value": False, "threshold": True,
                         "op": "==", "source": "tools/geom/stl_repair.py --json rc"})
        if e["closed"] is False:
            return _rec("PF-SURFACE", "refuse",
                        "PF-SURFACE: %s: %d open edge(s), %d non-manifold edge(s) before "
                        "any repair (the mesher refuses surface/closed; castellation "
                        "classifies by parity, SPEC-LIT §23.3)"
                        % (e["path"], e["open_edges"], e["non_manifold_edges"]), _CITE_A,
                        {"observable": obs,
                         "value": (e["open_edges"] or 0) + (e["non_manifold_edges"] or 0),
                         "threshold": 0, "op": "<=", "source": "tools/geom/stl_repair.py before"})
        if (e["reoriented_triangles"] or 0) > 0:
            return _rec("PF-SURFACE", "refuse",
                        "PF-SURFACE: %s: %d triangle(s) wound against their neighbours"
                        % (e["path"], e["reoriented_triangles"]), _CITE_A,
                        {"observable": obs, "value": e["reoriented_triangles"],
                         "threshold": 0, "op": "<=",
                         "source": "tools/geom/stl_repair.py orientation"})
    return _pf_surface_pass(surface)




def _pf_surface_pass(surface):
    """The (a) pass record; flipped_components is an input here, never a refusal."""
    ins = [{"name": "surface %d %s" % (i, e["path"]),
            "value": {"open_edges": e["open_edges"], "non_manifold_edges": e["non_manifold_edges"],
                      "reoriented_triangles": e["reoriented_triangles"],
                      "flipped_components": e["flipped_components"],
                      "triangles": e["triangles"]}, "unit": ""}
           for i, e in enumerate(surface["surfaces"])]
    if surface["bbox"] is not None:
        ins.append({"name": "bbox", "value": surface["bbox"], "unit": "m"})
    return _rec("PF-SURFACE", "pass",
                "PF-SURFACE: %d surface(s) closed and consistently wound"
                % len(surface["surfaces"]), _CITE_A,
                {"observable": "/input/surfaces", "value": len(surface["surfaces"]),
                 "threshold": len(surface["surfaces"]), "op": "==",
                 "source": "tools/geom/stl_repair.py --json"}, ins)


def _pf_quality(effective, q_raw, reference_quality):
    """(b) PF-QUALITY: the block byte-equal to the reference (canonical JSON)."""
    ref = reference_quality if reference_quality is not None else dict(REFERENCE_QUALITY)
    if effective is None:
        return _rec("PF-QUALITY", "refuse",
                    "PF-QUALITY: quality is %s, not a mapping of numbers" % _type_desc(q_raw),
                    _CITE_B, {"observable": "/quality", "value": q_raw,
                              "threshold": schema.canonical_sha256(ref), "op": "==",
                              "source": "the config against REFERENCE_QUALITY"})
    bad = {}
    for k in sorted(set(effective) | set(ref)):
        ev, rv = effective.get(k), ref.get(k)
        if schema.canonical_sha256(ev) != schema.canonical_sha256(rv):
            bad[k] = ev
    if bad:
        msg = "; ".join("%s %s != reference %s" % (k, json.dumps(bad[k], sort_keys=True),
                         json.dumps(ref[k], sort_keys=True) if k in ref else "absent")
                        for k in sorted(bad))
        return _rec("PF-QUALITY", "refuse", "PF-QUALITY: " + msg, _CITE_B,
                    {"observable": "/quality", "value": bad, "threshold": ref, "op": "==",
                     "source": "the config against REFERENCE_QUALITY"})
    return _rec("PF-QUALITY", "pass",
                "PF-QUALITY: the quality block is byte-equal to the reference", _CITE_B,
                {"observable": "/quality", "value": schema.canonical_sha256(effective),
                 "threshold": schema.canonical_sha256(ref), "op": "==",
                 "source": "schema.canonical_sha256"})


def preflight_checks(config, *, argv, records, q_eff, q_eff_b, q_raw, surface, gates,
                     knobs, edits, fingerprint, flow, octree_probe, h_wall_min_m,
                     reference_quality, mirror=None):
    """The checks after (a), in (C7)'s order; preflight() owns the assembly."""
    records.append(_pf_quality(q_eff_b, q_raw, reference_quality))
    flag_records = schema.check_flags(list(argv), knobs)
    if flag_records:
        records.extend(flag_records)
    else:
        records.append(_rec("PF-FLAGS", "pass",
                            "PF-FLAGS: no forbidden flag on the command line",
                            "docs/15 §C L0 (c); tools/autonomy/schema/knobs.json",
                            {"observable": "argv", "value": list(argv),
                             "threshold": [f["flag"] for f in knobs.get("forbidden_flags", [])],
                             "op": "not_in", "source": "tools/autonomy/schema/knobs.json"}))
    for e in (edits or []):
        if isinstance(e, dict):
            r = schema.check_edit(e.get("pointer"), e.get("to"), knobs)
        else:
            r = schema.check_edit(None, None, knobs)
        if r is not None:
            records.append(r)
    n0 = len(records)
    patch_leaves = _pf_knob_leaves(config, knobs, records)
    if surface.get("readable"):
        _pf_patch(surface, patch_leaves, records)
    r = _pf_sharp_ft0(config, fingerprint)
    if r is not None:
        records.append(r)
    if len(records) == n0:
        records.append(_rec(
            "PF-KNOBS", "pass",
            "PF-KNOBS: every settings leaf is whitelisted and in range, the patch "
            "names are the STL's own, and the mesher's own parser takes the config",
            "docs/15 §C L0 (d); tools/autonomy/schema/knobs.json",
            {"observable": "/", "value": "the config's settings leaves",
             "threshold": "schema/knobs.json whitelist", "op": "in",
             "source": "tools/autonomy/schema/knobs.json"}))
    _pf_nonorth(q_eff, config, records)
    _pf_layers(config, q_eff, surface, fingerprint, flow, h_wall_min_m, gates, knobs, records)
    _pf_domain(config, surface, records)
    _pf_budget(gates, octree_probe, records)
    m = mirror if mirror is not None else mirror_dryrun(config, argv, surface)
    if m is not None and not _covers(records, m["field"]):
        records.append(_rec(
            "PF-CONFIG", "refuse",
            "PF-CONFIG: the mesher refuses this config: %s: %s" % (m["field"], m["message"]),
            "automesher.rs:111-253, mod.rs:526 (validate), serde deny_unknown_fields",
            {"observable": "argv" if m["field"] == "argv" else field_to_pointer(m["field"]),
             "value": _value_at(config, m["field"]),
             "threshold": "the mesher's parser and validator", "op": "==",
             "source": "ofgpu-automesher -dryRun, mirrored"}))
    return m


def _ptr_get(config, pointer):
    """The value at a JSON pointer, or the mesher's default when a key is missing."""
    node = config
    for seg in pointer.split("/")[1:]:
        if isinstance(node, dict) and seg in node:
            node = node[seg]
        else:
            return MESHER_DEFAULTS.get(pointer)
    return node


def _ptr_wall_level(config):
    """The largest band level over every refinement entry; 0 when there is no band."""
    top = 0
    for e in _ptr_get(config, "/refinement/levels") or []:
        for b in e.get("bands") or []:
            top = max(top, b["level"])
    return top


def plane_path(config, fingerprint):
    """True when the config puts the body on the R-PLANE path (docs/15 §C L1 R-PLANE): a commensurate body, snap.feature_tolerance and snap.smoothing_passes 0, the lattice spacing a whole multiple of the wall cell, the extent on the body's faces - the one config on which a body with sharp edges keeps feature_tolerance 0 (README section D, 2026-09-26). False on a malformed config."""
    try:
        if fingerprint.get("commensurate") is not True:
            return False
        s = fingerprint.get("lattice_base_size_m")
        if not (isinstance(s, (int, float)) and not isinstance(s, bool) and s > 0):
            return False
        if _ptr_get(config, "/snap/feature_tolerance") != 0:
            return False
        if _ptr_get(config, "/snap/smoothing_passes") != 0:
            return False
        h = config["domain"]["base_size"] / 2 ** _ptr_wall_level(config)
        q = s / h
        if abs(q - round(q)) > 1e-9 or round(q) < 1:
            return False
        bb = fingerprint["bbox"]
        ext = config["domain"]["extent"]
        for a in range(3):
            q = (bb[2 * a] - ext[2 * a]) / h
            if abs(q - round(q)) > 1e-9:
                return False
        return True
    except (TypeError, KeyError, AttributeError, ValueError, IndexError,
            ZeroDivisionError, OverflowError):
        return False


_CITE_FT0 = ("tools/autonomy/README.md section D (the user's decision of 2026-09-26); "
             "docs/15 §K G-PILOT caution 1; SPEC-LIT (92.38)")


def _pf_sharp_ft0(config, fingerprint):
    """WL-SHARP-FT0: feature_tolerance 0 on a body with sharp edges, off the R-PLANE
    path, is refused (2026-09-26); None when it does not apply or no fingerprint is given."""
    if not isinstance(config, dict) or not isinstance(fingerprint, dict):
        return None
    sel = fingerprint.get("sharp_edge_length_m")
    if isinstance(sel, bool) or not isinstance(sel, (int, float)) or not sel > 0:
        return None
    ft = _vget(config, "snap", "feature_tolerance", 0.5)
    if isinstance(ft, bool) or not isinstance(ft, (int, float)) or ft != 0:
        return None
    if plane_path(config, fingerprint):
        return None
    fa = fingerprint.get("feature_angle_deg", 30.0)
    return _rec("WL-SHARP-FT0", "refuse",
                "WL-SHARP-FT0: snap.feature_tolerance = 0 on a body with %.6g m of sharp edge "
                "at %g deg switches the feature attraction off, so its edges are not captured; "
                "the user's decision of 2026-09-26 forbids it off the R-PLANE path" % (sel, fa),
                _CITE_FT0,
                {"observable": "/snap/feature_tolerance", "value": ft, "threshold": 0.0,
                 "op": ">", "source": "features.py sharp_edge_length_m; preflight.plane_path"},
                [{"name": "sharp_edge_length_m", "value": sel, "unit": "m"},
                 {"name": "feature_angle_deg", "value": fa, "unit": "deg"},
                 {"name": "plane_path", "value": False, "unit": "1"}],
                "refuse when sharp_edge_length_m > 0 and feature_tolerance == 0 "
                "and not plane_path")


def _pf_knob_leaves(config, knobs, records):
    """(d) 1-2: the edit refusals and the leaf walk; the patch leaves kept for PF-PATCH."""
    patch_leaves = {}
    for ptr, val in config_leaves(config):
        pat = _pointer_pattern(ptr)
        if pat in TEMPLATE_FREE:
            continue
        r = None
        if _forbidden_row(ptr, knobs) is not None:
            if not _is_default(ptr, pat, val):
                r = schema.check_edit(ptr, val, knobs)
        elif schema._knob_row(ptr, knobs) is not None:
            r = schema.check_edit(ptr, val, knobs)
            if r is None and pat == "/refinement/levels/*/patch" and isinstance(val, str):
                patch_leaves[ptr] = [val]
            elif r is None and pat == "/layers/patches" and isinstance(val, list):
                patch_leaves[ptr] = [v for v in val if isinstance(v, str)]
        elif not _is_default(ptr, pat, val):
            r = schema.check_edit(ptr, val, knobs)
        if r is not None:
            records.append(r)
    return patch_leaves


def _pf_patch(surface, patch_leaves, records):
    """(d) 3 PF-PATCH: the layer and band patch names against the STL's own."""
    names = surface.get("patches") or []
    for ptr in sorted(patch_leaves):
        unknown = [v for v in patch_leaves[ptr] if v not in names]
        if unknown:
            records.append(_rec(
                "PF-PATCH", "refuse",
                "PF-PATCH: %s: %s not in the STL's patches (%s)"
                % (ptr, json.dumps(unknown, ensure_ascii=False), ", ".join(names)),
                "docs/15 §C L0 (d); preflight-only: -dryRun does not check patch names "
                "(automesher.rs:433)",
                {"observable": ptr, "value": unknown, "threshold": names, "op": "in",
                 "source": "tools/geom/stl_repair.py patches"}))


def _pf_nonorth(q_eff, config, records):
    """(e) PF-NONORTH: no ceiling under the 25.2394 deg floor of a 2:1 transition."""
    if q_eff is None:
        records.append(_rec("PF-NONORTH", "abstain",
                            "PF-NONORTH: the quality block is not a mapping of numbers; "
                            "no floor check", _CITE_E, None))
        return
    lv = _levels_max(config)
    if lv is None:
        records.append(_rec("PF-NONORTH", "abstain",
                            "PF-NONORTH: the refinement block is malformed; no floor check",
                            _CITE_E, None))
        return
    v = q_eff["max_non_orth_deg"]
    if lv <= 0:
        records.append(_rec("PF-NONORTH", "pass",
                            "PF-NONORTH: no refinement transition (level 0)", _CITE_E,
                            {"observable": "/refinement/levels", "value": 0,
                             "threshold": 0, "op": "==", "source": "the config"}))
    elif not v >= NON_ORTH_FLOOR_DEG:
        records.append(_rec("PF-NONORTH", "refuse",
                            "PF-NONORTH: quality.max_non_orth_deg = %r is below the %.4f deg "
                            "floor a 2:1 octree transition sets (atan(sqrt(2)/3); docs/15 §B "
                            "fact 4): every refined mesh would fail G4"
                            % (v, NON_ORTH_FLOOR_DEG), _CITE_E,
                            {"observable": "/quality/max_non_orth_deg", "value": v,
                             "threshold": NON_ORTH_FLOOR_DEG, "op": ">=",
                             "source": "docs/15 §B fact 4"},
                            formula="max_non_orth_deg >= atan(sqrt(2)/3)"))
    else:
        records.append(_rec("PF-NONORTH", "pass",
                            "PF-NONORTH: max_non_orth_deg %r clears the %.4f deg floor "
                            "at refinement level %d" % (v, NON_ORTH_FLOOR_DEG, lv), _CITE_E,
                            {"observable": "/quality/max_non_orth_deg", "value": v,
                             "threshold": NON_ORTH_FLOOR_DEG, "op": ">=",
                             "source": "docs/15 §B fact 4"}))


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _layers_state(config):
    """(None, None, abstain?) or (n, patches, layers dict) for check (f)."""
    lay = config.get("layers") if isinstance(config, dict) else None
    if lay is None:
        return 0, [], None
    if not isinstance(lay, dict):
        return None, None, True
    n = lay.get("n", 0)
    patches = lay.get("patches", [])
    if isinstance(n, bool) or not isinstance(n, int) or not isinstance(patches, list):
        return None, None, True
    return n, patches, lay


def _pf_layers(config, q_eff, surface, fingerprint, flow, h_wall_min_m, gates, knobs,
               records):
    """(f) PF-YPLUS and PF-THIN, in that order."""
    n, patches, lay = _layers_state(config)
    if n is None:
        records.append(_rec("PF-YPLUS", "abstain",
                            "PF-YPLUS: layers.n or layers.patches is malformed", _CITE_Y, None))
        records.append(_rec("PF-THIN", "abstain",
                            "PF-THIN: layers.n or layers.patches is malformed", _CITE_T, None))
        return
    if n <= 0 or not patches:
        for rid, cite in (("PF-YPLUS", _CITE_Y), ("PF-THIN", _CITE_T)):
            records.append(_rec(rid, "pass", "%s: no layers requested" % rid, cite,
                                {"observable": "/layers/n", "value": n, "threshold": 0,
                                 "op": "==", "source": "the config"}))
        return
    _pf_yplus(config, n, lay, q_eff, flow, gates, knobs, records)
    _pf_thin(config, lay, q_eff, fingerprint, h_wall_min_m, records)


def _pf_yplus(config, n, lay, q_eff, flow, gates, knobs, records):
    t1a = None
    if flow is not None and not schema.errors(flow, "FlowSpec"):
        t1a = schema.a_priori_wall(flow, gates["yplus_max_a_priori"])["t1_a_priori_m"]
    g = lay.get("growth", MESHER_DEFAULTS["/layers/growth"])
    cf = lay.get("cell_frac", MESHER_DEFAULTS["/layers/cell_frac"])
    r = q_eff["min_thickness_ratio"] if q_eff else None
    dom = config.get("domain")
    base = dom.get("base_size") if isinstance(dom, dict) else None
    if t1a is None or not _num(g) or g <= 0 or not _num(cf) or cf <= 0 \
            or r is None or not _num(base) or base <= 0:
        records.append(_rec("PF-YPLUS", "abstain",
                            "PF-YPLUS: no usable flow spec, growth, cell_frac, "
                            "min_thickness_ratio or base_size", _CITE_Y, None))
        return
    lo, hi = yplus_window(t1a, n, g, cf, r)
    cap = schema._knob_row("/refinement/max_level", knobs)["max"]
    landing = [L for L in range(cap + 1) if lo <= base / 2 ** L <= hi]
    s = float(n) if g == 1.0 else (g ** n - 1.0) / (g - 1.0)
    ins = [{"name": "t1_m", "value": t1a, "unit": "m"},
           {"name": "S", "value": s, "unit": ""},
           {"name": "lo", "value": lo, "unit": "m"},
           {"name": "hi", "value": hi, "unit": "m"},
           {"name": "landing", "value": landing, "unit": "level"}]
    if lo > hi:
        records.append(_rec(
            "PF-YPLUS", "refuse",
            "PF-YPLUS: the §D.3 window is empty at n = %d, g = %r: t1*S/cell_frac = %r m > "
            "3*t1/min_thickness_ratio = %r m (y+ = 1 a priori, t1 = %r m)" % (n, g, lo, hi, t1a),
            _CITE_Y, {"observable": "/layers/growth", "value": lo, "threshold": hi,
                      "op": ">", "source": "docs/15 §D.3 (R-WIN)"}, ins,
            formula="t1*S/cell_frac <= h <= 3*t1/min_thickness_ratio"))
    elif not landing and base / 2 ** cap > hi:
        records.append(_rec(
            "PF-YPLUS", "refuse",
            "PF-YPLUS: y+ <= 1 needs h_wall <= %.3g mm, finest reachable %.3g mm at "
            "max_level %d" % (hi * 1e3, base / 2 ** cap * 1e3, cap), _CITE_Y,
            {"observable": "/domain/base_size", "value": base, "threshold": hi * 2 ** cap,
             "op": "<=", "source": "docs/15 §D.3 (R-WIN)"}, ins,
            formula="base/2**cap <= hi"))
    elif not landing and base < lo:
        records.append(_rec(
            "PF-YPLUS", "refuse",
            "PF-YPLUS: y+ <= 1 needs h_wall >= %.3g mm, coarsest reachable %.3g mm at "
            "level 0" % (lo * 1e3, base * 1e3), _CITE_Y,
            {"observable": "/domain/base_size", "value": base, "threshold": lo,
             "op": ">=", "source": "docs/15 §D.3 (R-WIN)"}, ins,
            formula="base >= lo"))
    elif not landing:
        records.append(_rec(
            "PF-YPLUS", "refuse",
            "PF-YPLUS: y+ <= 1 needs h_wall in [%.3g, %.3g] mm, no level 0..%d lands in "
            "[%.3g, %.3g] mm" % (lo * 1e3, hi * 1e3, cap, lo * 1e3, hi * 1e3), _CITE_Y,
            {"observable": "/domain/base_size", "value": base, "threshold": [lo, hi],
             "op": "not_in", "source": "docs/15 §D.3 (R-WIN)"}, ins,
            formula="lo <= base/2**L <= hi for some L"))
    else:
        records.append(_rec("PF-YPLUS", "pass",
                            "PF-YPLUS: the window [%r, %r] m is landed by level(s) %s"
                            % (lo, hi, landing), _CITE_Y,
                            {"observable": "landing levels", "value": landing,
                             "threshold": 1, "op": ">=",
                             "source": "docs/15 §D.3 (R-WIN)"}, ins,
                            formula="lo <= base/2**L <= hi for some L"))


def _pf_thin(config, lay, q_eff, fingerprint, h_wall_min_m, records):
    """(f) PF-THIN: the (92.51) edge on the config's own first layer."""
    r = q_eff["min_thickness_ratio"] if q_eff else None
    t1c = lay.get("first_thickness", MESHER_DEFAULTS["/layers/first_thickness"])
    if not _num(t1c) or t1c <= 0 or r is None:
        records.append(_rec("PF-THIN", "abstain",
                            "PF-THIN: first_thickness or min_thickness_ratio is not a "
                            "positive number", _CITE_T, None))
        return
    pred = h_wall_predicted(config, fingerprint)
    if h_wall_min_m is not None:
        h, src = h_wall_min_m, "measured"
    else:
        h, src = pred["h_m"], "predicted"
    if h is None:
        records.append(_rec("PF-THIN", "abstain",
                            "PF-THIN: no h_wall available (no measured h and the prediction "
                            "abstains)", _CITE_T, None))
        return
    ratio = G5_FACTOR * t1c / h if h > 0 else math.inf
    ins = [{"name": "t1_m", "value": t1c, "unit": "m"},
           {"name": "h_m", "value": h, "unit": "m"},
           {"name": "ratio", "value": ratio, "unit": ""},
           {"name": "source", "value": src, "unit": ""},
           {"name": "level", "value": pred["level"], "unit": ""},
           {"name": "patch", "value": pred["patch"], "unit": ""}]
    if not ratio >= r:
        why = ("the measured post-snap shortest wall edge" if src == "measured"
               else "base_size/2**L on a castellated wall - a prediction on a snapped one")
        records.append(_rec(
            "PF-THIN", "refuse",
            "PF-THIN: 3 * %r / %r = %r < min_thickness_ratio = %r (SPEC-LIT §92.51); "
            "every layer cell would fail G5 (h: %s)" % (t1c, h, ratio, r, why), _CITE_T,
            {"observable": "/layers/first_thickness", "value": t1c, "threshold": r,
             "op": ">=", "source": "the config against min_thickness_ratio"}, ins,
            formula="3 * t1 / h >= min_thickness_ratio"))
    else:
        records.append(_rec("PF-THIN", "pass",
                            "PF-THIN: 3 * %r / %r = %r >= min_thickness_ratio = %r"
                            % (t1c, h, ratio, r), _CITE_T,
                            {"observable": "/layers/first_thickness", "value": t1c,
                             "threshold": r, "op": ">=",
                             "source": "the config against min_thickness_ratio"}, ins,
                            formula="3 * t1 / h >= min_thickness_ratio"))


def _pf_domain(config, surface, records):
    """(g) PF-DOMAIN: stage 0's margin, which -dryRun does not run."""
    dom = config.get("domain") if isinstance(config, dict) else None
    ext = dom.get("extent") if isinstance(dom, dict) else None
    bs = dom.get("base_size") if isinstance(dom, dict) else None
    ext_ok = (isinstance(ext, list) and len(ext) == 6
              and all(_num(v) for v in ext)
              and ext[1] > ext[0] and ext[3] > ext[2] and ext[5] > ext[4])
    if not (ext_ok and _num(bs) and bs > 0) or surface.get("bbox") is None:
        records.append(_rec("PF-DOMAIN", "abstain",
                            "PF-DOMAIN: the extent, base_size or the surface bbox is "
                            "missing or malformed", _CITE_G, None))
        return
    bbox, margin = surface["bbox"], float(bs)
    hit = None
    for a, nm in enumerate("xyz"):
        s_lo, s_hi = bbox[a], bbox[a + 3]
        d_lo, d_hi = ext[2 * a], ext[2 * a + 1]
        if (s_lo <= d_lo and s_hi >= d_hi) or \
                (s_lo >= d_lo + margin and s_hi <= d_hi - margin):
            continue
        hit = (nm, s_lo, s_hi, d_lo, d_hi)
        break
    if hit is not None:
        nm, s_lo, s_hi, d_lo, d_hi = hit
        records.append(_rec(
            "PF-DOMAIN", "refuse",
            "PF-DOMAIN: on %s the surface [%.3f, %.3f] neither sits inside the domain "
            "[%.3f, %.3f] with one base_size (%.3f) of margin nor spans past it on both "
            "sides (automesher.rs:509, SPEC-LIT §92.2 stage 0)"
            % (nm, s_lo, s_hi, d_lo, d_hi, margin), _CITE_G,
            {"observable": "/domain/extent", "value": [d_lo, d_hi],
             "threshold": [s_lo - margin, s_hi + margin], "op": "<=",
             "source": "automesher.rs:509 (stage 0)"}, formula="span or margin per axis"))
    else:
        records.append(_rec("PF-DOMAIN", "pass",
                            "PF-DOMAIN: the surface sits inside the domain with margin or "
                            "spans it on every axis", _CITE_G,
                            {"observable": "/domain/extent", "value": list(ext),
                             "threshold": "span or one base_size of margin, per axis",
                             "op": "==", "source": "automesher.rs:509 (stage 0)"},
                            formula="span or margin per axis"))


def _pf_budget(gates, octree_probe, records):
    """(h) PF-BUDGET: the octree probe's leaves against gates.json."""
    n_leaves = None
    if isinstance(octree_probe, int) and not isinstance(octree_probe, bool):
        n_leaves = octree_probe
    elif isinstance(octree_probe, dict):
        nl = octree_probe.get("n_leaves")
        if isinstance(nl, int) and not isinstance(nl, bool):
            n_leaves = nl
    budget = gates["cell_budget"]
    if n_leaves is None:
        records.append(_rec("PF-BUDGET", "abstain",
                            "PF-BUDGET: no octree probe was run", _CITE_H, None))
    elif n_leaves > budget:
        records.append(_rec("PF-BUDGET", "refuse",
                            "PF-BUDGET: the octree probe has %d leaves > cell_budget %d "
                            "(gates.json; docs/15 §D.1 F5)" % (n_leaves, budget), _CITE_H,
                            {"observable": "octree.n_leaves", "value": n_leaves,
                             "threshold": budget, "op": "<=", "source": "gates.json"},
                            formula="n_leaves <= cell_budget"))
    else:
        records.append(_rec("PF-BUDGET", "pass",
                            "PF-BUDGET: the octree probe has %d leaves <= cell_budget %d"
                            % (n_leaves, budget), _CITE_H,
                            {"observable": "octree.n_leaves", "value": n_leaves,
                             "threshold": budget, "op": "<=", "source": "gates.json"},
                            formula="n_leaves <= cell_budget"))


def _assemble(records):
    """(C1): in CHECKS order, each check's refusals, or else its one pass/abstain
    record. PF-CONFIG (decided last) lands in PF-KNOBS's group, and a check's
    pass record never stands beside a refusal of its own group."""
    out = []
    for _, cid in CHECKS:
        mine = [r for r in records if CHECK_OF.get(r["rule_id"], r["rule_id"]) == cid]
        refs = [r for r in mine if r["verdict"] == "refuse"]
        out.extend(refs if refs else [r for r in mine if r["verdict"] != "refuse"][:1])
    n_ref = sum(1 for r in records if r["verdict"] == "refuse")
    if sum(1 for r in out if r["verdict"] == "refuse") != n_ref:
        raise PreflightError("a refusal record belongs to no check: %r"
                             % sorted(set(r["rule_id"] for r in records)))
    return out


def preflight(config, *, argv=(), edits=None, surface=None, fingerprint=None, flow=None,
              octree_probe=None, h_wall_min_m=None, reference_quality=None, gates=None,
              knobs=None, cwd=None):
    """(C7): the nine L0 checks on one config; the autonomy-preflight/1 result out."""
    gates = gates if gates is not None else schema.load_gates()
    knobs = knobs if knobs is not None else schema.load_knobs()
    if surface is None:
        surface = surface_facts(config, cwd)
    q_eff, q_eff_b, q_raw = _quality_block(config)
    records = []

    # (a) PF-SURFACE
    inp = config.get("input") if isinstance(config, dict) else None
    surfs = inp.get("surfaces") if isinstance(inp, dict) else None
    well = (isinstance(surfs, list) and len(surfs) > 0
            and all(isinstance(s, dict) and isinstance(s.get("path"), str) for s in surfs))
    if not well:
        records.append(_rec(
            "PF-SURFACE", "refuse",
            "PF-SURFACE: input.surfaces: at least one STL is required", _CITE_A,
            {"observable": "/input/surfaces", "value": surfs, "threshold": "a non-empty "
             "list of {path: str}", "op": "not_in", "source": "the config"}))
    else:
        records.append(_pf_surface_refuse_or_pass(surface))

    m = preflight_checks(config, argv=argv, records=records, q_eff=q_eff, q_eff_b=q_eff_b,
                         q_raw=q_raw, surface=surface, gates=gates, knobs=knobs,
                         edits=edits, fingerprint=fingerprint, flow=flow,
                         octree_probe=octree_probe, h_wall_min_m=h_wall_min_m,
                         reference_quality=reference_quality, mirror=None)
    records = _assemble(records)
    for r in records:
        errs = schema.errors(r, "DecisionRecord")
        if errs:
            raise PreflightError("preflight produced an invalid DecisionRecord (%s): %s"
                                 % (errs[0], r.get("rule_id")))
    refused = [r["rule_id"] for r in records if r["verdict"] == "refuse"]
    return {"schema": RESULT_SCHEMA, "verdict": "refuse" if refused else "pass",
            "refused": refused, "records": records,
            "dryrun": {"refuses": m is not None, "field": m["field"] if m else None,
                       "message": m["message"] if m else None}}


def _foam_list_body(path):
    """(count, lines) of an ASCII FoamFile list: the count line, then `(`...`)`."""
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().replace("\r\n", "\n").split("\n")
    for i, l in enumerate(lines):
        if l.strip() == "(":
            count = int(lines[i - 1].strip())
            body = []
            for l in lines[i + 1:]:
                s = l.strip()
                if s == ")":
                    break
                if s:
                    body.append(s)
            return count, body
    raise PreflightError("no list found in %s" % path)


def _foam_points(path):
    count, body = _foam_list_body(path)
    vals = " ".join(body).replace("(", " ").replace(")", " ").split()
    pts = [float(v) for v in vals]
    if len(pts) != 3 * count:
        raise PreflightError("%s: %d floats, expected 3 * %d" % (path, len(pts), count))
    return [pts[i:i + 3] for i in range(0, len(pts), 3)]


def _foam_faces(path):
    count, body = _foam_list_body(path)
    faces = []
    for s in body:
        m = re.match(r"^(\d+)\((.*)\)\s*$", s)
        if not m:
            raise PreflightError("%s: bad face line %r" % (path, s[:60]))
        ids = [int(v) for v in m.group(2).split()]
        if len(ids) != int(m.group(1)):
            raise PreflightError("%s: face declares %s, carries %d"
                                 % (path, m.group(1), len(ids)))
        faces.append(ids)
    if len(faces) != count:
        raise PreflightError("%s: %d faces, expected %d" % (path, len(faces), count))
    return faces


def _foam_boundary(path):
    count, body = _foam_list_body(path)
    text = "\n".join(body)
    out = []
    for m in re.finditer(r"(\S+)\s*\{([^}]*)\}", text, re.S):
        nf = re.search(r"nFaces\s+(\d+)\s*;", m.group(2))
        sf = re.search(r"startFace\s+(\d+)\s*;", m.group(2))
        if not (nf and sf):
            raise PreflightError("%s: patch %s carries no nFaces/startFace"
                                 % (path, m.group(1)))
        out.append((m.group(1), int(nf.group(1)), int(sf.group(1))))
    if len(out) != count:
        raise PreflightError("%s: %d boundary entries, expected %d"
                             % (path, len(out), count))
    return out


def wall_edge_min(case_dir, patches):
    """(C9): layers.rs 1294-1305's loop - the shortest polygon edge over the patches."""
    pm = os.path.join(case_dir, "constant", "polyMesh")
    pts = _foam_points(os.path.join(pm, "points"))
    faces = _foam_faces(os.path.join(pm, "faces"))
    bnd = dict((name, (nf, sf)) for name, nf, sf in
               _foam_boundary(os.path.join(pm, "boundary")))
    best = float("inf")
    for p in patches:
        if p not in bnd:
            raise PreflightError("patch %r is not in %s" % (p, os.path.join(pm, "boundary")))
        nf, sf = bnd[p]
        for f in faces[sf:sf + nf]:
            n = len(f)
            for k in range(n):
                a, b = pts[f[k]], pts[f[(k + 1) % n]]
                dx, dy, dz = a[0] - b[0], a[1] - b[1], a[2] - b[2]
                d = math.sqrt(dx * dx + dy * dy + dz * dz)
                if d < best:
                    best = d
    if best == float("inf"):
        raise PreflightError("no faces found for %r in %s" % (patches, pm))
    return best


def _run_stage(binary, config, workdir, stage, timeout_s):
    """One `-stopAfter STAGE` run; (rc, stdout, stderr, case dir, config path)."""
    cfg = copy.deepcopy(config)
    tag = "%s_%s" % (stage, cfg["output"].get("name", "probe"))
    case = os.path.join(workdir, tag + "_probe")
    cfg["output"]["case_dir"] = case
    os.makedirs(workdir, exist_ok=True)
    cfg_path = os.path.join(workdir, tag + "_config.json")
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=1)
    try:
        p = subprocess.run([binary, cfg_path, "-stopAfter", stage], capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=timeout_s)
        return p.returncode, p.stdout, p.stderr, case
    except subprocess.TimeoutExpired as e:
        err = e.stderr if isinstance(e.stderr, str) else (e.stderr or b"").decode(
            "utf-8", "replace")
        return None, "", err, case


def snap_probe(binary, config, workdir, timeout_s=900.0):
    """(C9): -stopAfter snap, then the shortest wall edge over the layer patches."""
    patches = config["layers"]["patches"]
    rc, _, err, case = _run_stage(binary, config, workdir, "snap", timeout_s)
    try:
        if rc != 0:
            raise PreflightError("snap_probe refused (rc=%s): %s"
                                 % (rc, _last_error_line(err) or "no error line"))
        return wall_edge_min(case, patches)
    finally:
        shutil.rmtree(case, ignore_errors=True)


def octree_probe(binary, config, workdir, timeout_s=900.0):
    """(C9): -stopAfter octree; n_leaves and the live max non-orthogonality."""
    name = config["output"]["name"]
    rc, _, err, case = _run_stage(binary, config, workdir, "octree", timeout_s)
    try:
        if rc != 0:
            return {"n_leaves": None, "max_non_orth_deg": None, "exit_code": rc,
                    "error": _last_error_line(err)}
        sp = os.path.join(case, "%s_summary.json" % name)
        with open(sp, encoding="utf-8") as fh:
            st = json.load(fh)["stages"][0]
        return {"n_leaves": st["n_leaves"], "max_non_orth_deg": st.get("max_non_orth_deg"),
                "exit_code": 0}
    finally:
        shutil.rmtree(case, ignore_errors=True)


def open_variant(stl_path, out_path):
    """(C10): the ASCII STL minus its last facet block - 3 open edges, measured."""
    try:
        with open(stl_path, encoding="utf-8") as fh:
            text = fh.read()
    except UnicodeDecodeError:
        raise PreflightError("open_variant: %s is not an ASCII STL" % stl_path)
    ms = list(re.finditer(r"[ \t]*facet normal.*?endfacet[ \t]*\r?\n", text, re.S))
    if not ms:
        raise PreflightError("open_variant: %s carries no facet block" % stl_path)
    m = ms[-1]
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text[:m.start()] + text[m.end():])
    return out_path


_BS = chr(92)  # a single backslash, kept out of the literals below


def _node(c, key):
    """c[key] as a dict: created when absent, passed through when broken."""
    if key not in c:
        c[key] = {}
    return c[key]


def _set_pointer(c, ptr, value):
    segs = ptr.strip("/").split("/")
    node = c
    for s in segs[:-1]:
        if isinstance(node, list):
            i = int(s) if s.isdigit() else -1
            if not 0 <= i < len(node):
                raise IndexError("pointer %s leaves the template" % ptr)
            node = node[i]
            continue
        nxt = node.get(s)
        if not isinstance(nxt, (dict, list)):
            nxt = {}
            node[s] = nxt
        node = nxt
    if isinstance(node, list):
        i = int(segs[-1]) if segs[-1].isdigit() else -1
        if not 0 <= i < len(node):
            raise IndexError("pointer %s leaves the template" % ptr)
        node[i] = value
    else:
        node[segs[-1]] = value


def _random_band(c, rng):
    ref = c.get("refinement")
    if not isinstance(ref, dict):
        raise TypeError("refinement is broken")
    bands = []
    for i, lv in enumerate(ref.get("levels", []) or []):
        for j, b in enumerate((lv.get("bands") or []) if isinstance(lv, dict) else []):
            bands.append((i, j))
    if not bands:
        raise IndexError("no bands")
    return rng.choice(bands)


def _band(c, i, j):
    return c["refinement"]["levels"][i]["bands"][j]


# (C10) the 60 defect kinds of random_config; uniform/randint/choice are the
# rng instance's. The brief's prose says 58; its own table carries 60 rows and
# the table is what random_config applies.
def _k_table_snap_iterations(c, rng, ctx):
    _node(c, "snap")["iterations"] = rng.randint(201, 1000)


def _k_table_snap_tolerance(c, rng, ctx):
    _node(c, "snap")["tolerance"] = rng.choice([0.02, 0.5, 1e-10])


def _k_table_snap_undo_limit(c, rng, ctx):
    _node(c, "snap")["undo_limit"] = rng.randint(11, 50)


def _k_table_snap_feature_tol(c, rng, ctx):
    _node(c, "snap")["feature_tolerance"] = rng.uniform(1.01, 5)


def _k_table_layers_n(c, rng, ctx):
    _node(c, "layers")["n"] = rng.randint(17, 64)


def _k_table_layers_growth(c, rng, ctx):
    _node(c, "layers")["growth"] = rng.choice([rng.uniform(0.1, 0.99), rng.uniform(2.01, 3)])


def _k_table_layers_smoothing(c, rng, ctx):
    _node(c, "layers")["smoothing"] = rng.uniform(1.01, 2)


def _k_table_layers_retreat(c, rng, ctx):
    _node(c, "layers")["retreat_limit"] = rng.randint(9, 30)


def _k_table_band_level(c, rng, ctx):
    i, j = _random_band(c, rng)
    _band(c, i, j)["level"] = rng.randint(7, 12)


def _k_table_feature_level(c, rng, ctx):
    _node(c, "refinement")["levels"][0]["feature_level"] = rng.randint(7, 12)


def _k_table_band_distance(c, rng, ctx):
    i, j = _random_band(c, rng)
    _band(c, i, j)["distance"] = rng.choice([0.0, -rng.uniform(0.01, 1)])


def _k_table_first_thickness(c, rng, ctx):
    _node(c, "layers")["first_thickness"] = rng.choice([0.0, -1e-3])


def _k_unlisted_min_thickness(c, rng, ctx):
    _node(c, "layers")["min_thickness"] = rng.uniform(0.01, 0.09)


def _k_unlisted_feature_angle(c, rng, ctx):
    _node(c, "refinement")["feature_angle_deg"] = rng.uniform(10, 25)


def _k_unlisted_min_faces(c, rng, ctx):
    _node(c, "castellation")["min_faces"] = rng.choice([1, 2, 3, 5, 8])


def _k_unlisted_grading(c, rng, ctx):
    _node(c, "domain")["grading"] = [1.0, 1.1, 1.0]


def _k_unlisted_body_ok(c, rng, ctx):
    _node(c, "castellation")["bodies"] = [{"name": "solid1", "patches": [ctx["patches"][0]]}]


def _k_forbidden_cell_frac(c, rng, ctx):
    _node(c, "layers")["cell_frac"] = rng.uniform(0.1, 0.45)


def _k_forbidden_medial_frac(c, rng, ctx):
    _node(c, "layers")["medial_frac"] = rng.uniform(0.1, 0.45)


def _k_quality_altered(c, rng, ctx):
    c["quality"] = dict(REFERENCE_QUALITY, max_non_orth_deg=rng.uniform(26, 80))


def _k_quality_nonpositive(c, rng, ctx):
    nm = rng.choice(("max_closure", "max_non_orth_deg", "report_non_orth_deg",
                     "min_thickness_ratio", "max_cond"))
    c["quality"] = dict(REFERENCE_QUALITY, **{nm: rng.choice([0.0, -1.0])})


def _k_quality_unknown_key(c, rng, ctx):
    c["quality"] = dict(REFERENCE_QUALITY, foo=1)


def _k_validate_extent_axis(c, rng, ctx):
    a = rng.randint(0, 2)
    ext = c["domain"]["extent"]
    ext[2 * a + 1] = ext[2 * a] - rng.uniform(0, 1)


def _k_validate_base_size(c, rng, ctx):
    _node(c, "domain")["base_size"] = rng.choice([0.0, -0.5])


def _k_validate_grading(c, rng, ctx):
    _node(c, "domain")["grading"] = [1.0, rng.choice([0.0, -1.0]), 1.0]


def _k_validate_max_level(c, rng, ctx):
    _node(c, "refinement")["max_level"] = rng.randint(7, 40)


def _k_validate_max_area_ratio(c, rng, ctx):
    _node(c, "snap")["max_area_ratio"] = rng.uniform(0.0, 0.99)


def _k_validate_snap_smoothing(c, rng, ctx):
    _node(c, "snap")["smoothing"] = rng.choice([-0.5, rng.uniform(1.01, 2)])


def _k_validate_feature_tol(c, rng, ctx):
    _node(c, "snap")["feature_tolerance"] = -rng.uniform(0.01, 1)


def _k_validate_growth(c, rng, ctx):
    _node(c, "layers")["growth"] = rng.choice([0.0, -1.3])


def _k_validate_seed_point(c, rng, ctx):
    cc = _node(c, "castellation")
    cc["keep_region"] = "seed"
    cc["seed_point"] = None


def _k_validate_no_surfaces(c, rng, ctx):
    _node(c, "input")["surfaces"] = []


def _k_validate_body_name(c, rng, ctx):
    _node(c, "castellation")["bodies"] = [
        {"name": rng.choice(["fluid", "xMax", "a b", "a;b", ""]),
         "patches": [ctx["patches"][0]]}]


def _k_validate_body_dup(c, rng, ctx):
    _node(c, "castellation")["bodies"] = [
        {"name": "s1", "patches": [ctx["patches"][0]]},
        {"name": "s1", "patches": ["other"]}]


def _k_validate_body_empty(c, rng, ctx):
    _node(c, "castellation")["bodies"] = [{"name": "s1", "patches": []}]


def _k_validate_body_shared(c, rng, ctx):
    _node(c, "castellation")["bodies"] = [
        {"name": "s1", "patches": [ctx["patches"][0]]},
        {"name": "s2", "patches": [ctx["patches"][0]]}]


_UINT_FIELDS = (("snap", "iterations"), ("snap", "smoothing_passes"), ("layers", "n"),
                ("castellation", "min_faces"), ("refinement", "max_level"))
_NUM_FIELDS = (("snap", "tolerance"), ("layers", "first_thickness"),
               ("domain", "base_size"), ("snap", "iterations"))


def _k_parse_float_for_uint(c, rng, ctx):
    s, f = rng.choice(_UINT_FIELDS)
    _node(c, s)[f] = rng.choice([2.5, 3.0])


def _k_parse_negative_uint(c, rng, ctx):
    s, f = rng.choice(_UINT_FIELDS)
    _node(c, s)[f] = -1


def _k_parse_big_u32(c, rng, ctx):
    w = rng.randrange(3)
    if w == 0:
        _node(c, "refinement")["max_level"] = 2 ** 32
    elif w == 1:
        i, j = _random_band(c, rng)
        _band(c, i, j)["level"] = 2 ** 32
    else:
        _node(c, "refinement")["levels"][0]["feature_level"] = 2 ** 32


def _k_parse_string_for_number(c, rng, ctx):
    s, f = rng.choice(_NUM_FIELDS)
    _node(c, s)[f] = "1"


def _k_parse_bool_for_number(c, rng, ctx):
    s, f = rng.choice(_NUM_FIELDS)
    _node(c, s)[f] = True


def _k_parse_null_for_number(c, rng, ctx):
    s, f = rng.choice(_NUM_FIELDS)
    _node(c, s)[f] = None


def _k_parse_unknown_key(c, rng, ctx):
    w = rng.randrange(9)
    if w == 0:
        c["zz"] = 1
    elif w in (1, 2, 3, 4, 5, 6):
        _node(c, ("domain", "snap", "layers", "refinement", "castellation",
                  "output")[w - 1])["zz"] = 1
    elif w == 7:
        _node(c, "refinement")["levels"][0]["zz"] = 1
    else:
        i, j = _random_band(c, rng)
        _band(c, i, j)["zz"] = 1


def _k_parse_missing_required(c, rng, ctx):
    w = rng.randrange(13)
    if w == 0:
        del c["domain"]["extent"]
    elif w == 1:
        del c["domain"]["base_size"]
    elif w == 2:
        del c["output"]["name"]
    elif w == 3:
        del c["output"]["case_dir"]
    elif w == 4:
        del c["input"]["surfaces"]
    elif w == 5:
        del c["input"]["surfaces"][0]["path"]
    elif w == 6:
        del c["refinement"]["levels"][0]["bands"]
    elif w == 7:
        del c["refinement"]["levels"][0]["patch"]
    elif w == 8:
        i, j = _random_band(c, rng)
        del _band(c, i, j)["distance"]
    elif w == 9:
        i, j = _random_band(c, rng)
        del _band(c, i, j)["level"]
    elif w == 10:
        del c["input"]
    elif w == 11:
        del c["domain"]
    else:
        del c["output"]


def _k_parse_extent_length(c, rng, ctx):
    ext = c["domain"]["extent"]
    c["domain"]["extent"] = rng.choice([ext[:5], list(ext) + [1.0]])


def _k_parse_extent_element(c, rng, ctx):
    c["domain"]["extent"][rng.randint(0, 5)] = "a"


def _k_parse_grading_length(c, rng, ctx):
    _node(c, "domain")["grading"] = [1.0, 1.0]


def _k_parse_seed_length(c, rng, ctx):
    _node(c, "castellation")["seed_point"] = [0.5, 0.5]


def _k_parse_keep_variant(c, rng, ctx):
    _node(c, "castellation")["keep_region"] = rng.choice(["Largest", "outside", 1])


def _k_parse_container(c, rng, ctx):
    w = rng.randrange(5)
    if w == 0:
        c["snap"] = 5
    elif w == 1:
        _node(c, "refinement")["levels"] = {}
    elif w == 2:
        _node(c, "output")["patch_names"] = []
    elif w == 3:
        c["domain"] = None
    else:
        _node(c, "layers")["patches"] = "x"


def _k_parse_patch_names_value(c, rng, ctx):
    _node(c, "output")["patch_names"] = {"xMin": 3}


def _k_surface_missing(c, rng, ctx):
    _node(c, "input")["surfaces"][0]["path"] = ctx["missing_stl"]


def _k_surface_open(c, rng, ctx):
    _node(c, "input")["surfaces"][0]["path"] = ctx["open_stl"]


def _k_patch_unknown_layer(c, rng, ctx):
    _node(c, "layers")["patches"] = list(ctx["patches"]) + ["nosuch"]


def _k_patch_unknown_band(c, rng, ctx):
    _node(c, "refinement")["levels"][0]["patch"] = "nosuch"


def _k_domain_no_margin(c, rng, ctx):
    a = rng.randint(0, 2)
    bs = c["domain"]["base_size"]
    c["domain"]["extent"][2 * a] = ctx["bbox"][a] - rng.uniform(0.0, 0.99) * bs


def _k_argv_permissive(c, rng, ctx):
    ctx["argv"].append("-permissive")


def _k_argv_tag_ok(c, rng, ctx):
    ctx["argv"] += ["-tag", "g%d" % rng.randint(0, 99)]


def _k_argv_tag_bad(c, rng, ctx):
    ctx["argv"] += ["-tag", rng.choice(["a/b", "a" + _BS + "b"])]


def _k_argv_unknown_flag(c, rng, ctx):
    ctx["argv"].append(rng.choice(["-force", "-quiet", "-v"]))


KINDS = (
 ("table:snap_iterations", _k_table_snap_iterations),
 ("table:snap_tolerance", _k_table_snap_tolerance),
 ("table:snap_undo_limit", _k_table_snap_undo_limit),
 ("table:snap_feature_tol", _k_table_snap_feature_tol),
 ("table:layers_n", _k_table_layers_n),
 ("table:layers_growth", _k_table_layers_growth),
 ("table:layers_smoothing", _k_table_layers_smoothing),
 ("table:layers_retreat", _k_table_layers_retreat),
 ("table:band_level", _k_table_band_level),
 ("table:feature_level", _k_table_feature_level),
 ("table:band_distance", _k_table_band_distance),
 ("table:first_thickness", _k_table_first_thickness),
 ("unlisted:min_thickness", _k_unlisted_min_thickness),
 ("unlisted:feature_angle", _k_unlisted_feature_angle),
 ("unlisted:min_faces", _k_unlisted_min_faces),
 ("unlisted:grading", _k_unlisted_grading),
 ("unlisted:body_ok", _k_unlisted_body_ok),
 ("forbidden:cell_frac", _k_forbidden_cell_frac),
 ("forbidden:medial_frac", _k_forbidden_medial_frac),
 ("quality:altered", _k_quality_altered),
 ("quality:nonpositive", _k_quality_nonpositive),
 ("quality:unknown_key", _k_quality_unknown_key),
 ("validate:extent_axis", _k_validate_extent_axis),
 ("validate:base_size", _k_validate_base_size),
 ("validate:grading", _k_validate_grading),
 ("validate:max_level", _k_validate_max_level),
 ("validate:max_area_ratio", _k_validate_max_area_ratio),
 ("validate:snap_smoothing", _k_validate_snap_smoothing),
 ("validate:feature_tol", _k_validate_feature_tol),
 ("validate:growth", _k_validate_growth),
 ("validate:seed_point", _k_validate_seed_point),
 ("validate:no_surfaces", _k_validate_no_surfaces),
 ("validate:body_name", _k_validate_body_name),
 ("validate:body_dup", _k_validate_body_dup),
 ("validate:body_empty", _k_validate_body_empty),
 ("validate:body_shared", _k_validate_body_shared),
 ("parse:float_for_uint", _k_parse_float_for_uint),
 ("parse:negative_uint", _k_parse_negative_uint),
 ("parse:big_u32", _k_parse_big_u32),
 ("parse:string_for_number", _k_parse_string_for_number),
 ("parse:bool_for_number", _k_parse_bool_for_number),
 ("parse:null_for_number", _k_parse_null_for_number),
 ("parse:unknown_key", _k_parse_unknown_key),
 ("parse:missing_required", _k_parse_missing_required),
 ("parse:extent_length", _k_parse_extent_length),
 ("parse:extent_element", _k_parse_extent_element),
 ("parse:grading_length", _k_parse_grading_length),
 ("parse:seed_length", _k_parse_seed_length),
 ("parse:keep_variant", _k_parse_keep_variant),
 ("parse:container", _k_parse_container),
 ("parse:patch_names_value", _k_parse_patch_names_value),
 ("surface:missing", _k_surface_missing),
 ("surface:open", _k_surface_open),
 ("patch:unknown_layer", _k_patch_unknown_layer),
 ("patch:unknown_band", _k_patch_unknown_band),
 ("domain:no_margin", _k_domain_no_margin),
 ("argv:permissive", _k_argv_permissive),
 ("argv:tag_ok", _k_argv_tag_ok),
 ("argv:tag_bad", _k_argv_tag_bad),
 ("argv:unknown_flag", _k_argv_unknown_flag),
)


def _expanded_rows(c):
    """The whitelist rows in order, wildcards over the template's own indices."""
    rows = []
    ref = c.get("refinement") if isinstance(c.get("refinement"), dict) else {}
    levels = ref.get("levels") if isinstance(ref.get("levels"), list) else []
    for row in _knobs()["whitelist"]:
        p = row["pointer"]
        if p in ("/refinement/levels/*/patch", "/layers/patches"):
            continue
        if p.endswith("/bands/*/distance") or p.endswith("/bands/*/level"):
            leaf = "distance" if p.endswith("/distance") else "level"
            for i, e in enumerate(levels):
                bands = e.get("bands") if isinstance(e, dict) and isinstance(e.get("bands"),
                                                                            list) else []
                for j in range(len(bands)):
                    rows.append((row, "/refinement/levels/%d/bands/%d/%s" % (i, j, leaf),
                                 (i, j)))
            continue
        if p == "/refinement/levels/*/feature_level":
            for i in range(len(levels)):
                rows.append((row, "/refinement/levels/%d/feature_level" % i, None))
            continue
        rows.append((row, p, None))
    return rows


def _legal_value(c, rng, row, idx):
    """A LEGAL draw for one whitelist row (step 2 of random_config)."""
    p, typ, lo, hi = row["pointer"], row["type"], row["min"], row["max"]
    if p == "/domain/base_size":
        cur = c.get("domain", {}).get("base_size") if isinstance(c.get("domain"), dict) \
            else None
        cur = cur if _num(cur) else 1.0
        return cur * rng.uniform(0.5, 2)
    if p == "/layers/first_thickness":
        return 10 ** rng.uniform(-5, -1.3)
    if p.endswith("/bands/*/distance"):
        i, j = idx
        cur = _band(c, i, j).get("distance") if isinstance(_band(c, i, j), dict) else None
        cur = cur if _num(cur) else 0.25
        return cur * rng.uniform(0.5, 2)
    if p == "/domain/extent":
        ext = c.get("domain", {}).get("extent") if isinstance(c.get("domain"), dict) else None
        out = []
        for a in range(3):
            lo_, hi_ = float(a), float(a) + 4.0
            if isinstance(ext, list) and len(ext) == 6 and _num(ext[2 * a]) and \
                    _num(ext[2 * a + 1]):
                lo_, hi_ = float(ext[2 * a]), float(ext[2 * a + 1])
            cen, wid = (lo_ + hi_) / 2.0, (hi_ - lo_) / 2.0
            out.extend([cen - wid * rng.uniform(0.8, 1.5), cen + wid * rng.uniform(0.8, 1.5)])
        return out
    if typ == "int":
        return rng.randint(int(lo), int(hi))
    return rng.uniform(float(lo), float(hi))


def random_config(rng, template, patches, bbox, open_stl, missing_stl):
    """(C10): (config, argv, kinds) - a legal knob draw, then 0-3 defect kinds."""
    c = copy.deepcopy(template)
    argv = []
    kinds = []
    ctx = {"patches": patches, "bbox": bbox, "open_stl": open_stl,
           "missing_stl": missing_stl, "argv": argv}
    for row, ptr, idx in _expanded_rows(c):
        if rng.random() < 0.5:
            _set_pointer(c, ptr, _legal_value(c, rng, row, idx))
    if rng.random() < 0.5:
        for _ in range(rng.choice([1, 1, 2, 3])):
            name, fn = rng.choice(KINDS)
            try:
                fn(c, rng, ctx)
                kinds.append(name)
            except (TypeError, KeyError, IndexError, AttributeError):
                kinds.append(name + "!skipped")
    return c, argv, kinds


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head():
    try:
        p = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=30)
        return (p.stdout or "").strip() or "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def _gate_template(name, out_dir, stl_path):
    """(C10): the box_sphere / wing_b probe config, pointed at the gate's STL."""
    src = os.path.join(PROBES_DIR, "box_sphere" if name == "box_sphere" else "wing_b_L4",
                       "config.json")
    with open(src, encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg["input"]["surfaces"][0]["path"] = stl_path
    cfg["output"]["case_dir"] = os.path.join(out_dir, "case_" + name)
    if name == "wing_b":
        cfg.get("layers", {}).pop("min_thickness", None)
    return cfg


def _gate_part1(stls, n, seed, streams, out_dir, gates, knobs, binary):
    """(C10 part 1): setting-level checks against -dryRun on n random configs."""
    names = list(stls)
    cfg_dir = os.path.join(out_dir, "cfg")
    os.makedirs(cfg_dir, exist_ok=True)
    jobs, extras = [], []
    i = 0
    for k, nm in enumerate(names):
        per = n // len(names) + (1 if k < n % len(names) else 0)
        rng = random.Random(seed * 1000 + k)
        template = _gate_template(nm, out_dir, stls[nm])
        sf = surface_facts(template)
        open_stl = open_variant(stls[nm], os.path.join(out_dir, nm + "_open.stl"))
        missing = os.path.join(out_dir, nm + "_missing.stl")
        if os.path.exists(missing):
            raise PreflightError("the missing STL %s exists" % missing)
        extras.append((open_stl, missing))
        for _ in range(per):
            c, argv, kinds = random_config(rng, template, sf["patches"], sf["bbox"],
                                           open_stl, missing)
            cfg_path = os.path.join(cfg_dir, "%s_%05d.json" % (nm, i))
            with open(cfg_path, "w", encoding="utf-8") as fh:
                json.dump(c, fh, indent=1)
            jobs.append({"i": i, "stl": nm, "kinds": kinds, "argv": list(argv),
                         "cfg_path": cfg_path, "config": c, "open": open_stl,
                         "missing": missing})
            i += 1

    done, lock, t0 = [0], threading.Lock(), time.time()

    def one(job):
        res = preflight(job["config"], argv=job["argv"], gates=gates, knobs=knobs,
                        cwd=out_dir)
        d = run_dryrun(binary, job["cfg_path"], job["argv"],
                       surface_paths=[stls[job["stl"]], job["open"], job["missing"]])
        with lock:
            done[0] += 1
            if done[0] % 250 == 0 or done[0] == len(jobs):
                print("[gate] part 1 %d/%d rows (%.0f s)" % (done[0], len(jobs),
                       time.time() - t0), flush=True)
        return job, res, d

    with ThreadPoolExecutor(max_workers=min(streams, 6)) as ex:
        outs = list(ex.map(one, jobs))
    rows_path = os.path.join(out_dir, "g_preflight_rows.jsonl")
    m_ref_dry_pass = m_pass_dry_ref = field_dis = bad_exit = undetected = 0
    n_dry_refuse = n_dry_pass = 0
    per_stl = dict((nm, {"refused": 0, "passed": 0}) for nm in names)
    kind_count, hist, disagree = {}, {}, []
    with open(rows_path, "w", encoding="utf-8") as fh:
        for job, res, d in outs:
            rc, dfield = d["exit_code"], d["field"]
            row = {"i": job["i"], "stl": job["stl"], "kinds": job["kinds"],
                   "argv": job["argv"], "dryrun_exit": rc, "dryrun_field": dfield,
                   "mirror_refuses": res["dryrun"]["refuses"],
                   "mirror_field": res["dryrun"]["field"], "verdict": res["verdict"],
                   "refused": res["refused"]}
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            keep = False
            if res["dryrun"]["refuses"] and rc == 0:
                m_ref_dry_pass += 1
                keep = True
            if not res["dryrun"]["refuses"] and rc not in (0, None):
                m_pass_dry_ref += 1
                keep = True
            if rc != 0:
                n_dry_refuse += 1
                per_stl[job["stl"]]["refused"] += 1
                if res["dryrun"]["field"] != dfield:
                    field_dis += 1
                    keep = True
                if res["verdict"] != "refuse":
                    undetected += 1
                    keep = True
            elif rc == 0:
                n_dry_pass += 1
                per_stl[job["stl"]]["passed"] += 1
                if res["verdict"] == "refuse":
                    for r in res["records"]:
                        if r["verdict"] == "refuse":
                            hist[r["rule_id"]] = hist.get(r["rule_id"], 0) + 1
            if rc not in (0, 1):
                bad_exit += 1
            for k in job["kinds"]:
                if not k.endswith("!skipped"):
                    kind_count[k] = kind_count.get(k, 0) + 1
            if keep:
                disagree.append(dict(row, cfg=job["cfg_path"]))
            else:
                os.remove(job["cfg_path"])
    return _part1_verdict(n, len(outs), m_ref_dry_pass, m_pass_dry_ref, field_dis,
                          bad_exit, undetected, n_dry_refuse, n_dry_pass, per_stl,
                          kind_count, hist, disagree)


def _part1_verdict(n, n_rows, m_ref_dry_pass, m_pass_dry_ref, field_dis, bad_exit,
                   undetected, n_dry_refuse, n_dry_pass, per_stl, kind_count, hist,
                   disagree):
    """The part-1 PASS decision and its report block."""
    kinds_missing = [k for k, _ in KINDS if kind_count.get(k, 0) < 20]
    min_kind = min([kind_count.get(k, 0) for k, _ in KINDS], default=0)
    ok = (n_rows == n and m_ref_dry_pass == 0 and m_pass_dry_ref == 0 and field_dis == 0
          and bad_exit == 0 and undetected == 0 and not kinds_missing
          and n_dry_refuse >= 0.2 * n and n_dry_pass >= 0.2 * n)
    line = ("%d rows, %d mirror-refused/dryRun-passed, %d mirror-passed/dryRun-refused, "
            "%d field mismatches, %d bad exits, %d undetected, %d refused / %d passed by "
            "-dryRun, min kind %d of %d"
            % (n_rows, m_ref_dry_pass, m_pass_dry_ref, field_dis, bad_exit, undetected,
               n_dry_refuse, n_dry_pass, min_kind, len(KINDS)))
    return {"verdict": "PASS" if ok else "FAIL", "line": line, "n_rows": n_rows,
            "mirror_refused_dryrun_passed": m_ref_dry_pass,
            "mirror_passed_dryrun_refused": m_pass_dry_ref,
            "field_mismatches": field_dis, "bad_exits": bad_exit,
            "undetected": undetected, "dryrun_refused": n_dry_refuse,
            "dryrun_passed": n_dry_pass, "per_stl": per_stl, "kinds": kind_count,
            "kinds_missing": kinds_missing, "preflight_only": hist,
            "disagreeing_rows": disagree[:20]}


def c_thin_check(binary, stl_path, workdir):
    """(C10 part 2): the thin_t1 refusal reproduced to the digit, four ways."""
    with open(os.path.join(PROBES_DIR, "thin_t1", "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg["input"]["surfaces"][0]["path"] = stl_path
    cfg["output"]["case_dir"] = os.path.join(workdir, "case_thin_t1")
    os.makedirs(workdir, exist_ok=True)
    out = {"h": None, "t1": None, "ratio": None, "r": None, "exit_code": None,
           "mesher_line": None, "bit_equal": False, "six": False, "predicted": None,
           "refused": None, "ok": False}
    h = snap_probe(binary, cfg, workdir)
    out["h"] = h
    res = preflight(cfg, h_wall_min_m=h)
    out["refused"] = res["refused"]
    refs = [x for x in res["records"]
            if x["rule_id"] == "PF-THIN" and x["verdict"] == "refuse"]
    if res["refused"] != ["PF-THIN"] or not refs:
        return out
    ins = dict((i["name"], i["value"]) for i in refs[0]["inputs"])
    t1c, ratio, r = ins["t1_m"], ins["ratio"], refs[0]["trigger"]["threshold"]
    out.update(t1=t1c, ratio=ratio, r=r)
    cfg_path = os.path.join(workdir, "thin_t1_config.json")
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=1)
    p = subprocess.run([binary, cfg_path], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=900.0)
    line = _last_error_line(p.stderr)
    out["exit_code"], out["mesher_line"] = p.returncode, line
    m = G5_LINE_RE.search(line or "")
    if p.returncode == 1 and m:
        nums = [float(m.group(k)) for k in (1, 2, 3, 4)]
        out["bit_equal"] = (nums[0] == t1c and nums[1] == h
                            and nums[2] == ratio and nums[3] == r)
        out["six"] = ("%.6f" % ratio == "0.007984")
    res2 = preflight(cfg)
    refs2 = [x for x in res2["records"]
             if x["rule_id"] == "PF-THIN" and x["verdict"] == "refuse"]
    out["predicted"] = bool(
        res2["refused"] == ["PF-THIN"] and refs2
        and any(i["name"] == "ratio"
                and i["value"] == G5_FACTOR * 0.0001 / 0.125
                for x in refs2 for i in x["inputs"]))
    out["ok"] = bool(res["refused"] == ["PF-THIN"] and out["bit_equal"]
                     and out["six"] and out["predicted"])
    return out


def _part3_rows():
    """The 48 (stl, L, wall kind, multiplier) rows of part 3."""
    rows = []
    for L in (2, 3, 4):
        for m in (0.25, 0.5, 0.8, 1.25, 2.0, 4.0, 8.0, 16.0):
            rows.append(("box_sphere", L, "snapped", m))
        for m in (0.5, 0.8, 1.25, 2.0):
            rows.append(("box_sphere", L, "castellated", m))
    for L in (3, 4):
        for m in (0.25, 0.5, 1.25, 2.0, 4.0, 8.0):
            rows.append(("wing_b", L, "snapped", m))
    return rows


def _part3_config(template, stl_name, L, wall, m, work):
    """One row's config: bands and layers per (C10), t1 from the prediction."""
    c = copy.deepcopy(template)
    ref = c.get("refinement") if isinstance(c.get("refinement"), dict) else {}
    if stl_name == "box_sphere":
        ref["levels"] = [{"patch": "sphere",
                          "bands": [{"distance": 0.5, "level": L},
                                    {"distance": 2.0, "level": max(L - 1, 1)}]}]
        ref["max_level"] = L
        c["refinement"] = ref
        c["layers"] = {"patches": ["sphere"], "n": 3, "first_thickness": 0.0,
                       "growth": 1.2}
        if wall == "castellated":
            snap = c.get("snap") if isinstance(c.get("snap"), dict) else {}
            snap["iterations"] = 0
            c["snap"] = snap
    else:
        ref["levels"] = [{"patch": "wing",
                          "bands": [{"distance": 0.1, "level": L},
                                    {"distance": 0.5, "level": max(L - 1, 1)},
                                    {"distance": 1.5, "level": 2}],
                          "feature_level": L}]
        ref["max_level"] = L
        c["refinement"] = ref
        c["layers"] = {"patches": ["wing"], "n": 3, "first_thickness": 0.0,
                       "growth": 1.2}
    pred = h_wall_predicted(c)
    hp = pred["h_m"]
    c["layers"]["first_thickness"] = hp * 0.05 / 3.0 * m
    tag = "L%d_%s_%s" % (L, wall, str(m).replace(".", "p"))
    c["output"]["case_dir"] = os.path.join(work, "case_%s_%s" % (stl_name, tag))
    return c, hp


def _gate_part3(stls, out_dir, gates, knobs, binary, streams):
    """(C10 part 3): the prediction against the real stage 5, 48 full runs."""
    work = os.path.join(out_dir, "p3")
    os.makedirs(work, exist_ok=True)
    specs = []
    for k, (nm, L, wall, m) in enumerate(_part3_rows()):
        cfg, hp = _part3_config(_gate_template(nm, out_dir, stls[nm]), nm, L, wall, m, work)
        pre = preflight(cfg, gates=gates, knobs=knobs)
        specs.append({"i": k, "stl": nm, "L": L, "wall": wall, "m": m, "h_pred": hp,
                      "t1": cfg["layers"]["first_thickness"],
                      "predicted": "refuse" if "PF-THIN" in pre["refused"] else "pass",
                      "config": cfg})

    def run(spec):
        cfg_path = os.path.join(work, "cfg_%03d.json" % spec["i"])
        with open(cfg_path, "w", encoding="utf-8") as fh:
            json.dump(spec["config"], fh, indent=1)
        t0 = time.perf_counter()
        try:
            p = subprocess.run([binary, cfg_path], capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=900.0)
            rc, so, se, timed_out = p.returncode, p.stdout, p.stderr, False
        except subprocess.TimeoutExpired as e:
            rc, timed_out = None, True
            so = e.stdout if isinstance(e.stdout, str) else (e.stdout or b"").decode(
                "utf-8", "replace")
            se = e.stderr if isinstance(e.stderr, str) else (e.stderr or b"").decode(
                "utf-8", "replace")
        row = dict((k2, spec[k2]) for k2 in ("i", "stl", "L", "wall", "m", "t1",
                                             "h_pred", "predicted"))
        row["exit_code"], row["timed_out"] = rc, timed_out
        row["seconds"] = time.perf_counter() - t0
        line = _last_error_line(se)
        row["error"] = line
        gm = G5_LINE_RE.search(line) if line else None
        if timed_out or rc not in (0, 1):
            row["actual"] = "crash"
        elif rc == 1 and gm:
            row["actual"] = "refused"
            row["h_mesher"] = float(gm.group(2))
            row["ratio_mesher"] = float(gm.group(3))
        elif rc == 0 or score.last_stage(so) == "layers":
            row["actual"] = "not_refused"
        else:
            row["actual"] = "not_reached"
        shutil.rmtree(spec["config"]["output"]["case_dir"], ignore_errors=True)
        print("[gate] part 3 case %02d/%02d %s L%d %s -> %s (%.0f s)"
              % (spec["i"] + 1, len(specs), spec["stl"], spec["L"], spec["wall"],
                 row["actual"], row["seconds"]), flush=True)
        return row

    with ThreadPoolExecutor(max_workers=min(streams, 6)) as ex:
        rows = list(ex.map(run, specs))
    return {"verdict": None, "rows": rows, **_part3_summary(rows)}


def _part3_summary(rows):
    """Per-wall-kind counts, rates over reached rows, and the h ratio spread."""
    summary = {}
    for wall in ("castellated", "snapped"):
        rs = [r for r in rows if r["wall"] == wall]
        refd = [r for r in rs if r["actual"] == "refused"]
        nref = [r for r in rs if r["actual"] == "not_refused"]
        nreach = [r for r in rs if r["actual"] in ("refused", "not_refused")]
        fp = [r for r in rs if r["predicted"] == "pass" and r["actual"] == "refused"]
        fr = [r for r in rs if r["predicted"] == "refuse" and r["actual"] == "not_refused"]
        ratios = sorted(r["h_pred"] / r["h_mesher"] for r in refd)
        summary[wall] = {
            "n": len(rs), "refused": len(refd), "not_refused": len(nref),
            "not_reached": len([r for r in rs if r["actual"] == "not_reached"]),
            "crash": len([r for r in rs if r["actual"] == "crash"]),
            "false_pass": len(fp), "false_refuse": len(fr),
            "false_pass_rate": (len(fp) / len(nreach)) if nreach else None,
            "false_refuse_rate": (len(fr) / len(nreach)) if nreach else None,
            "h_pred_over_h_mesher": ({"min": ratios[0], "median": ratios[len(ratios) // 2],
                                      "max": ratios[-1]} if ratios else None)}
    ok = (len(rows) == 48 and not any(r["actual"] == "crash" for r in rows)
          and summary["castellated"]["false_pass"] == 0
          and summary["castellated"]["refused"] >= 4
          and all(r["h_mesher"] == r["h_pred"] for r in rows
                  if r["wall"] == "castellated" and r["actual"] == "refused"))
    line = ("48 rows, %d crashes; castellated n 12, refused %d, false_pass %d, "
            "false_refuse %d, h_pred/h_mesher %s; snapped n 36, refused %d, "
            "false_pass %d, false_refuse %d (reported, not gated)"
            % (summary["castellated"]["crash"] + summary["snapped"]["crash"],
               summary["castellated"]["refused"], summary["castellated"]["false_pass"],
               summary["castellated"]["false_refuse"],
               summary["castellated"]["h_pred_over_h_mesher"],
               summary["snapped"]["refused"], summary["snapped"]["false_pass"],
               summary["snapped"]["false_refuse"]))
    return {"verdict": "PASS" if ok else "FAIL", "line": line,
            "castellated": summary["castellated"], "snapped": summary["snapped"]}


def _report_md(report):
    """G-PREFLIGHT.md, every number read from the merged JSON (<= 60 lines)."""
    p1 = report["parts"].get("1")
    p2 = report["parts"].get("2")
    p3 = report["parts"].get("3")
    hist = "none" if not (p1 and p1.get("preflight_only")) else ", ".join(
        "%s %d" % (k, v) for k, v in sorted(p1["preflight_only"].items()))
    L = []
    L.append("<!-- meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). "
             "Source-available, not Open Source. No GPL-licensed source was consulted. -->")
    L.append("")
    L.append("# G-PREFLIGHT (docs/15 §F) - preflight.py's gate")
    L.append("")
    L.append("date %s; binary sha256 %s; git HEAD %s; seed %s; n %s; streams %s"
             % (report.get("date"), report.get("binary_sha256"), report.get("git_head"),
                report.get("seed"), report.get("n"), report.get("streams")))
    L.append("")
    L.append("verdict: %s" % report.get("verdict"))
    L.append("")
    if p1:
        L.append("## part 1 - the mirror against -dryRun (%s configs) - %s"
                 % (p1["n_rows"], p1["verdict"]))
        L.append("")
        L.append(p1["line"] + ".")
        per = ", ".join("%s: %d refused / %d passed" % (nm, v["refused"], v["passed"])
                        for nm, v in sorted((p1.get("per_stl") or {}).items()))
        if per:
            L.append("")
            L.append("per STL: " + per + ".")
        L.append("")
        L.append("preflight-only refusals (dryRun passed, preflight refused): " + hist + ".")
        L.append("")
    if p2:
        L.append("## part 2 - C-THIN to the digit - %s" % p2.get("verdict",
                  "PASS" if p2.get("ok") else "FAIL"))
        L.append("")
        L.append("h = %r; preflight refuses 3 * %r / %r = %r < %r; the live refusal line is "
                 "bit-equal (%s); \"%%.6f\" %% ratio = %s; the castellated prediction refuses "
                 "3 * 0.0001 / 0.125 = 0.0024 (%s)."
                 % (p2["h"], p2.get("t1"), p2["h"], p2.get("ratio"), p2.get("r"),
                    p2["bit_equal"], p2.get("six"), p2.get("predicted")))
        L.append("")
        L.append("mesher: " + str(p2.get("mesher_line")))
        L.append("")
    if p3:
        c3, s3 = p3["castellated"], p3["snapped"]
        L.append("## part 3 - the castellated h prediction against %d real stage-5 outcomes - %s"
                 % (c3["n"] + s3["n"], p3["verdict"]))
        L.append("")
        L.append("castellated: n %d, refused %d, not_refused %d, not_reached %d, crashes %d, "
                 "false_pass %d (rate %s), false_refuse %d (rate %s), h_pred/h_mesher %s."
                 % (c3["n"], c3["refused"], c3["not_refused"], c3["not_reached"], c3["crash"],
                    c3["false_pass"], c3["false_pass_rate"], c3["false_refuse"],
                    c3["false_refuse_rate"], c3["h_pred_over_h_mesher"]))
        L.append("")
        L.append("snapped: n %d, refused %d, not_refused %d, not_reached %d, crashes %d, "
                 "false_pass %d (rate %s), false_refuse %d (rate %s), h_pred/h_mesher %s."
                 % (s3["n"], s3["refused"], s3["not_refused"], s3["not_reached"], s3["crash"],
                    s3["false_pass"], s3["false_pass_rate"], s3["false_refuse"],
                    s3["false_refuse_rate"], s3["h_pred_over_h_mesher"]))
        L.append("")
        L.append("on snapped walls the castellated h is a prediction; these rates are "
                 "reported, not gated (docs/15 §F).")
        L.append("")
    return "\n".join(L)


def _write_reports(report):
    """The only in-tree writes of the gate: the JSON and the md in REPORT_DIR."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, "G-PREFLIGHT.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, sort_keys=True, ensure_ascii=False)
    with open(os.path.join(REPORT_DIR, "G-PREFLIGHT.md"), "w", encoding="utf-8",
              newline="\n") as fh:
        fh.write(_report_md(report))


def g_preflight(stls, n, seed, streams, out_dir, parts, binary):
    """G-PREFLIGHT: run the requested parts, merge them, rewrite the report files."""
    os.makedirs(out_dir, exist_ok=True)
    rep_path = os.path.join(out_dir, "report.json")
    report = {"schema": RESULT_SCHEMA, "binary_sha256": _sha256_file(binary),
              "git_head": _git_head(), "seed": seed, "n": n, "streams": streams,
              "date": schema._now_iso(), "parts": {}}
    if os.path.isfile(rep_path):
        with open(rep_path, encoding="utf-8") as fh:
            report["parts"] = json.load(fh).get("parts", {})
    gates = schema.load_gates()
    knobs = schema.load_knobs()
    for part in parts:
        print("[gate] part %s start (%s)" % (part, time.strftime("%H:%M:%S")), flush=True)
        if part == "1":
            report["parts"]["1"] = _gate_part1(stls, n, seed, streams, out_dir, gates,
                                               knobs, binary)
        elif part == "2":
            report["parts"]["2"] = _gate_part2(stls, out_dir, binary)
        elif part == "3":
            report["parts"]["3"] = _gate_part3(stls, out_dir, gates, knobs, binary, streams)
        else:
            raise PreflightError("--parts: unknown part %r" % part)
        present = report["parts"]
        report["verdict"] = "PASS" if all(
            k in present and present[k].get("verdict") == "PASS"
            for k in ("1", "2", "3")) else "FAIL"
        with open(rep_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1, sort_keys=True, ensure_ascii=False)
        _write_reports(report)
        p = report["parts"][part]
        print("[gate] part %s %s: %s" % (part, p["verdict"], p.get("line", "")))
    missing = [k for k in ("1", "2", "3") if k not in report["parts"]]
    failed = [k for k, v in sorted(report["parts"].items()) if v.get("verdict") != "PASS"]
    if missing or failed:
        print("G-PREFLIGHT FAIL: %s" % ", ".join(missing + failed))
        return 1
    print("G-PREFLIGHT PASS")
    return 0


def _gate_part2(stls, out_dir, binary):
    """(C10 part 2) wrapper: the verdict and the numbers from c_thin_check."""
    work = os.path.join(out_dir, "p2")
    r = c_thin_check(binary, stls["box_sphere"], work)
    line = ("h %r, 3 * %r / h = %r < %r, bit_equal %s, six %s, predicted %s"
            % (r["h"], r["t1"], r["ratio"], r["r"], r["bit_equal"], r["six"],
               r["predicted"]))
    return dict(r, line=line, verdict="PASS" if r["ok"] else "FAIL")


def _clean_config(tmp):
    """(C8): the clean config - -dryRun 0, octree n_leaves 1310, 8 layers."""
    return {"input": {"surfaces": [{"path": CUBEP_STL}]},
            "domain": {"extent": [0.0, 4.0, 0.0, 4.0, 0.0, 4.0], "base_size": 1.0},
            "refinement": {"levels": [{"patch": "cube",
                                       "bands": [{"distance": 0.25, "level": 2}]}],
                           "max_level": 2},
            "snap": {"feature_tolerance": 0.0, "smoothing_passes": 0},
            "layers": {"patches": ["cube"], "n": 8, "first_thickness": 0.01,
                       "growth": 1.2},
            "output": {"case_dir": os.path.join(tmp, "case_clean"), "name": "clean"}}


def _selftest_schema(binary, tmp, shared):
    """Group 1: the (C2) struct table and (C3) defaults against -schema."""
    p = subprocess.run([binary, "-schema"], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=60)
    if p.returncode != 0:
        raise AssertionError("-schema exited %d: %s" % (p.returncode, p.stderr[-300:]))
    sch = json.loads(p.stdout)
    props, defs = sch["properties"], sch["$defs"]
    n_fields = 0
    for sname, fields in STRUCTS.items():
        node = sch if sname == "AutomeshConfig" else defs.get(sname)
        if node is None:
            raise AssertionError("no $defs.%s in -schema" % sname)
        want = [f for f, _, _ in fields]
        if sorted(node.get("properties", {})) != sorted(want):
            raise AssertionError("%s: fields %r != %r" % (sname, sorted(node["properties"]),
                                                          sorted(want)))
        req = [f for f, _, r in fields if r]
        if sorted(node.get("required", [])) != sorted(req):
            raise AssertionError("%s: required %r != %r" % (sname, node.get("required"),
                                                            req))
        for f, kind, _ in fields:
            if not _schema_kind_ok(node["properties"][f], kind):
                raise AssertionError("%s.%s: %r does not look like %s"
                                     % (sname, f, node["properties"][f], kind))
            n_fields += 1
    consts = sorted(c["const"] for c in defs["KeepRegion"]["oneOf"])
    if consts != sorted(ENUM_VALUES):
        raise AssertionError("KeepRegion consts %r != %r" % (consts, sorted(ENUM_VALUES)))
    n_defaults = _selftest_defaults(props, defs)
    return ["[ok] mesher schema: %d structs, %d fields, %d defaults equal -schema"
            % (len(STRUCTS), n_fields, n_defaults)]


def _schema_kind_ok(s, kind):
    """The (C2) kind vocabulary as -schema renders it."""
    if kind == "f64":
        return s.get("type") == "number"
    if kind == "usize":
        return s.get("type") == "integer" and s.get("format") == "uint"
    if kind == "u32":
        return s.get("type") == "integer" and s.get("format") == "uint32"
    if kind == "str":
        return s.get("type") == "string"
    if kind == "opt_str":
        return s.get("type") == ["string", "null"]
    if kind in ("f64x3", "f64x6"):
        n = int(kind[4:])
        return (s.get("type") == "array" and s.get("minItems") == n
                and s.get("maxItems") == n and (s.get("items") or {}).get("type") == "number")
    if kind == "opt_f64x3":
        return (s.get("type") == ["array", "null"] and s.get("minItems") == 3
                and s.get("maxItems") == 3)
    if kind == "enum":
        return s.get("$ref") == "#/$defs/KeepRegion"
    if kind == "map:str":
        return (s.get("type") == "object"
                and (s.get("additionalProperties") or {}).get("type") == "string")
    if kind == "list:str":
        return s.get("type") == "array" and (s.get("items") or {}).get("type") == "string"
    if kind.startswith("list:"):
        return (s.get("type") == "array"
                and (s.get("items") or {}).get("$ref") == "#/$defs/" + kind[5:])
    return s.get("$ref") == "#/$defs/" + kind


def _selftest_defaults(props, defs):
    """Group 1's 31 default comparisons (29 root default keys + 2 $defs fields)."""
    skip = {"/domain/grading", "/refinement/levels/*/feature_level", "/castellation/bodies"}
    got_keys = set()
    n = 0
    for sname in ("refinement", "castellation", "snap", "layers", "quality"):
        d = props[sname].get("default")
        if not isinstance(d, dict):
            raise AssertionError("properties.%s carries no default object" % sname)
        for k, v in d.items():
            key = "/%s/%s" % (sname, k)
            if key not in MESHER_DEFAULTS:
                raise AssertionError("-schema default %s is not in MESHER_DEFAULTS" % key)
            if not _eq_default(MESHER_DEFAULTS[key], v):
                raise AssertionError("default %s: %r != table %r"
                                     % (key, v, MESHER_DEFAULTS[key]))
            got_keys.add(key)
            n += 1
    want_keys = set(k for k in MESHER_DEFAULTS if k not in skip)
    if got_keys != want_keys:
        raise AssertionError("default keys differ: %r" % sorted(want_keys ^ got_keys))
    if not _eq_default(defs["DomainSpec"]["properties"]["grading"].get("default"),
                       MESHER_DEFAULTS["/domain/grading"]):
        raise AssertionError("$defs DomainSpec.grading default differs")
    if not _eq_default(defs["RefinementBand"]["properties"]["feature_level"].get("default"),
                       MESHER_DEFAULTS["/refinement/levels/*/feature_level"]):
        raise AssertionError("$defs RefinementBand.feature_level default differs")
    n += 2
    if "bodies" not in defs["CastellationSpec"]["properties"]:
        raise AssertionError("CastellationSpec.properties has no bodies")
    if "default" in defs["CastellationSpec"]["properties"]["bodies"]:
        raise AssertionError("CastellationSpec.bodies must carry no default")
    if n != 31:
        raise AssertionError("compared %d defaults, expected 31" % n)
    return n


def _selftest_constants(binary, tmp, shared):
    """Group 2: REFERENCE_QUALITY, the floor against the live octree, NO25's log."""
    p = subprocess.run([binary, "-schema"], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=60)
    sch = json.loads(p.stdout)
    if schema.canonical_sha256(sch["properties"]["quality"]["default"]) != \
            schema.canonical_sha256(REFERENCE_QUALITY):
        raise AssertionError("REFERENCE_QUALITY is not the -schema quality default")
    probe = octree_probe(binary, _clean_config(tmp), tmp)
    if probe["n_leaves"] is None:
        raise AssertionError("the clean octree probe refused: %r" % probe.get("error"))
    live = probe["max_non_orth_deg"]
    if live is None or abs(NON_ORTH_FLOOR_DEG - live) >= 1e-9:
        raise AssertionError("floor %r vs live octree %r" % (NON_ORTH_FLOOR_DEG, live))
    with open(os.path.join(PROBES_DIR, "NO25", "log.txt"), encoding="utf-8") as fh:
        log = fh.read()
    if "max 25.239 deg" not in log:
        raise AssertionError("NO25's log carries no 'max 25.239 deg' line")
    shared["clean_probe"] = probe
    return ["[ok] constants: REFERENCE_QUALITY is the -schema default; floor 25.2394 = the "
            "clean octree's %r (|d| < 1e-9) and NO25's log line \"max 25.239 deg\"" % live]


def _selftest_cases(tmp):
    """Group 3: the 54 hand cases of (C5) as (name, mutator, expected field)."""
    big64, big32 = 2 ** 64, 2 ** 32

    def setk(c, path, value):
        node = c
        for seg in path[:-1]:
            node = node[seg]
        node[path[-1]] = value

    def delk(c, path):
        node = c
        for seg in path[:-1]:
            node = node[seg]
        del node[path[-1]]

    return (
        ("1", lambda c: setk(c, ("snap", "iterations"), 2.5), "snap.iterations"),
        ("2", lambda c: setk(c, ("snap", "iterations"), 30.0), "snap.iterations"),
        ("3", lambda c: setk(c, ("layers", "n"), -1), "layers.n"),
        ("4", lambda c: setk(c, ("snap", "foo"), 1), "snap.foo"),
        ("5", lambda c: delk(c, ("domain", "base_size")), "domain.base_size"),
        ("6", lambda c: setk(c, ("refinement", "levels", 0, "bands", 0, "level"), big32),
         "refinement.levels[0].bands[0].level"),
        ("7", lambda c: setk(c, ("layers", "growth"), True), "layers.growth"),
        ("8", lambda c: setk(c, ("domain", "extent"), [0, 4, 3, 2, 0, 4]), "domain.extent"),
        ("9", lambda c: (setk(c, ("snap", "iterations"), 2.5),
                         delk(c, ("domain", "base_size"))), "domain.base_size"),
        ("10", lambda c: setk(c, ("quality",), {"max_cond": 0}), "quality.max_cond"),
        ("11", lambda c: setk(c, ("input", "surfaces", 0, "path"),
                              os.path.join(tmp, "nope.stl")), "io"),
        ("12", lambda c: setk(c, ("castellation",), {"keep_region": "Largest"}),
         "castellation.keep_region"),
        ("13", lambda c: setk(c, ("layers", "growth"), 1), "pass"),
        ("14", lambda c: setk(c, ("input", "surfaces"), []), "input.surfaces"),
        ("15", lambda c: delk(c, ("output",)), "output"),
        ("16", lambda c: setk(c, ("zzz",), 1), "zzz"),
        ("17", lambda c: setk(c, ("castellation",), {"keep_region": "seed"}),
         "castellation.seed_point"),
        ("18", lambda c: setk(c, ("domain", "extent"), [0, 4, 0, 4, 0]), "domain.extent"),
        ("19", lambda c: delk(c, ("refinement", "levels", 0, "bands")),
         "refinement.levels[0].bands"),
        ("20", lambda c: setk(c, ("snap", "iterations"), "30"), "snap.iterations"),
        ("21", lambda c: setk(c, ("snap", "iterations"), big64), "snap.iterations"),
        ("22", lambda c: setk(c, ("$schema",), None), "pass"),
        ("23", lambda c: setk(c, ("output", "patch_names"), {"xMin": 3}),
         "output.patch_names.xMin"),
    )


def _selftest_cases2(tmp):
    """Group 3 continued: hand cases 24-54."""
    def setk(c, path, value):
        node = c
        for seg in path[:-1]:
            node = node[seg]
        node[path[-1]] = value

    def delk(c, path):
        node = c
        for seg in path[:-1]:
            node = node[seg]
        del node[path[-1]]

    return (
        ("24", lambda c: setk(c, ("refinement", "max_level"), 7), "refinement.max_level"),
        ("25", lambda c: setk(c, ("domain", "grading"), [1, 0, 1]), "domain.grading"),
        ("26", lambda c: setk(c, ("snap", "max_area_ratio"), 0.5), "snap.max_area_ratio"),
        ("27", lambda c: setk(c, ("snap", "feature_tolerance"), -0.1),
         "snap.feature_tolerance"),
        ("28", lambda c: setk(c, ("snap", "smoothing"), 1.5), "snap.smoothing"),
        ("29", lambda c: setk(c, ("layers", "growth"), 0), "layers.growth"),
        ("30", lambda c: setk(c, ("domain", "extent"), [0, 4, 0, 4, 0, 4, 3]),
         "domain.extent"),
        ("31", lambda c: setk(c, ("domain", "extent"), [0, 4, "a", 4, 0, 4]),
         "domain.extent[2]"),
        ("32", lambda c: setk(c, ("domain", "grading"), [1, 1]), "domain.grading"),
        ("33", lambda c: setk(c, ("castellation",),
                              {"keep_region": "seed", "seed_point": [1, 1]}),
         "castellation.seed_point"),
        ("34", lambda c: setk(c, ("refinement", "levels"), {}), "refinement.levels"),
        ("35", lambda c: setk(c, ("refinement", "levels", 0, "bands"), []), "pass"),
        ("36", lambda c: setk(c, ("output", "patch_names"), []), "output.patch_names"),
        ("37", lambda c: setk(c, ("domain",), None), "domain"),
        ("38", lambda c: setk(c, ("$schema",), 3), "$schema"),
        ("39", lambda c: setk(c, ("snap",), 5), "snap"),
        ("40", lambda c: (setk(c, ("domain", "zz"), 1),
                          delk(c, ("domain", "extent"))), "domain.zz"),
        ("41", lambda c: (delk(c, ("domain", "extent")),
                          setk(c, ("domain", "zz"), 1)), "domain.zz"),
        ("42", lambda c: (delk(c, ("domain", "extent")),
                          delk(c, ("domain", "base_size"))), "domain.extent"),
        ("43", lambda c: setk(c, ("layers", "patches"), [1]), "layers.patches[0]"),
        ("44", lambda c: setk(c, ("castellation",),
                              {"bodies": [{"name": "fluid", "patches": ["cube"]}]}),
         "castellation.bodies[0].name"),
        ("45", lambda c: setk(c, ("castellation",),
                              {"bodies": [{"name": "a b", "patches": ["cube"]}]}),
         "castellation.bodies[0].name"),
        ("46", lambda c: setk(c, ("castellation",),
                              {"bodies": [{"name": "solid1", "patches": ["cube"]}]}),
         "pass"),
        ("47", lambda c: setk(c, ("refinement", "max_level"), -1.5),
         "refinement.max_level"),
        ("48", lambda c: setk(c, ("snap", "feature_tolerance"), 0), "pass"),
        ("49", lambda c: setk(c, ("refinement", "levels", 0, "bands", 0, "distance"), -1),
         "pass"),
        ("50", lambda c: setk(c, ("refinement", "max_level"), 4294967295),
         "refinement.max_level"),
        ("51", lambda c: None, "argv"),
        ("52", lambda c: (setk(c, ("zz",), 1), delk(c, ("input",))), "zz"),
        ("53", lambda c: setk(c, ("castellation",), {"keep_region": 1}),
         "castellation.keep_region"),
        ("54", lambda c: setk(c, ("domain", "extent"), None), "domain.extent"),
    )


def _selftest_mirror_cases(binary, tmp, shared):
    """Group 3: all 54 cases through the mirror AND the live -dryRun."""
    gates_knobs = (schema.load_gates(), schema.load_knobs())
    n_ok = 0
    for cases in (_selftest_cases(tmp), _selftest_cases2(tmp)):
        for name, mut, want in cases:
            c = copy.deepcopy(_clean_config(tmp))
            mut(c)
            cp = os.path.join(tmp, "case_%s.json" % name)
            with open(cp, "w", encoding="utf-8") as fh:
                json.dump(c, fh, indent=1)
            m = mirror_dryrun(c, ["-permissive"] if name == "51" else ())
            d = run_dryrun(binary, cp, ["-permissive"] if name == "51" else ())
            got_m = m["field"] if m else None
            want_f = None if want == "pass" else want
            if got_m != want_f or d["field"] != want_f:
                raise AssertionError("case %s: mirror %r / dryRun %r, expected %r"
                                     % (name, got_m, d["field"], want))
            n_ok += 1
            os.remove(cp)
    shared["gates_knobs"] = gates_knobs
    return ["[ok] mirror cases: %d of 54 (mirror and -dryRun)" % n_ok]


def _selftest_random(binary, tmp, shared):
    """Group 4: 300 random configs on cubep against -dryRun."""
    gates, knobs = schema.load_gates(), schema.load_knobs()
    template = _clean_config(tmp)
    sf = surface_facts(template)
    open_stl = open_variant(CUBEP_STL, os.path.join(tmp, "cubep_open.stl"))
    missing = os.path.join(tmp, "cubep_missing.stl")
    rng = random.Random(7)
    drawn = [random_config(rng, template, sf["patches"], sf["bbox"], open_stl, missing)
             for _ in range(300)]
    jobs = []
    for i, (c, argv, kinds) in enumerate(drawn):
        cp = os.path.join(tmp, "g4_%03d.json" % i)
        with open(cp, "w", encoding="utf-8") as fh:
            json.dump(c, fh, indent=1)
        jobs.append((c, argv, cp))

    def one(job):
        c, argv, cp = job
        res = preflight(c, argv=argv, gates=gates, knobs=knobs)
        d = run_dryrun(binary, cp, argv, surface_paths=[CUBEP_STL, open_stl, missing])
        return res, d

    with ThreadPoolExecutor(max_workers=4) as ex:
        outs = list(ex.map(one, jobs))
    n_ref = n_pass = disagree = undetected = 0
    for (res, d), (c, argv, cp) in zip(outs, jobs):
        if res["dryrun"]["refuses"] != (d["exit_code"] != 0) \
                or res["dryrun"]["field"] != d["field"]:
            disagree += 1
        if d["exit_code"] != 0:
            n_ref += 1
            if res["verdict"] != "refuse":
                undetected += 1
        else:
            n_pass += 1
        os.remove(cp)
    if disagree or undetected:
        raise AssertionError("%d disagreements, %d dryRun-refused configs preflight passed"
                             % (disagree, undetected))
    if n_ref < 90 or n_pass < 90:
        raise AssertionError("only %d refused / %d passed of 300" % (n_ref, n_pass))
    shared["open_cubep"] = open_stl
    return ["[ok] mirror vs -dryRun: 300 configs on cubep, 0 disagreements, %d refused, "
            "%d passed" % (n_ref, n_pass)]


def _survey_cfg(probe_id, box_stl, tmp):
    """A survey probe config with its STL path and case_dir replaced."""
    with open(os.path.join(PROBES_DIR, probe_id, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg["input"]["surfaces"][0]["path"] = box_stl
    cfg["output"]["case_dir"] = os.path.join(tmp, "case_" + probe_id)
    return cfg


def _selftest_fixtures(binary, tmp, shared):
    """Group 5: one fixture per refusal id, refused naming exactly that id."""
    gates, knobs = schema.load_gates(), schema.load_knobs()
    box_stl = shared["box_stl"]
    with open(os.path.join(PROBES_DIR, "NO25", "config.json"), encoding="utf-8") as fh:
        no25q = json.load(fh)["quality"]
    fx = [
        ("PF-SURFACE", lambda c, t: c["input"]["surfaces"][0].update(
            path=open_variant(CUBEP_STL, os.path.join(t, "cubep_open5.stl"))), {}),
        ("PF-QUALITY", lambda c, t: c.update(
            quality=dict(REFERENCE_QUALITY, max_non_orth_deg=26.0)), {}),
        ("WL-FLAG", lambda c, t: None, {"argv": ["-permissive"]}),
        ("WL-POINTER", lambda c, t: None,
         {"edits": [{"pointer": "layers/n", "from": 8, "to": 8}]}),
        ("WL-FORBIDDEN", lambda c, t: c["layers"].update(cell_frac=0.3), {}),
        ("WL-UNLISTED", lambda c, t: c["layers"].update(min_thickness=0.01), {}),
        ("WL-TYPE", lambda c, t: c["layers"].update(n=8.5), {}),
        ("WL-RANGE", lambda c, t: c["snap"].update(iterations=1000), {}),
        ("PF-PATCH", lambda c, t: c["layers"].update(patches=["cube", "nosuch"]), {}),
        ("PF-CONFIG", lambda c, t: c["snap"].update(foo=1), {}),
        ("PF-NONORTH", lambda c, t: c.update(quality=dict(no25q)),
         {"reference_quality": dict(no25q)}),
        ("PF-YPLUS", lambda c, t: None, {"flow": FLOW_FAST}),
        ("PF-THIN", lambda c, t: c["layers"].update(first_thickness=1e-4), {}),
        ("PF-DOMAIN", lambda c, t: c["domain"].update(
            extent=[0.5, 4.0, 0.0, 4.0, 0.0, 4.0]), {}),
        ("PF-BUDGET", lambda c, t: None, {"octree_probe": {"n_leaves": 2000001}}),
    ]
    records, n_ok = [], 0
    for rid, mut, kw in fx:
        c = _clean_config(tmp)
        mut(c, tmp)
        res = preflight(c, gates=gates, knobs=knobs, **kw)
        if res["refused"] != [rid]:
            raise AssertionError("fixture %s: refused %r" % (rid, res["refused"]))
        # (C1): in CHECKS order, each check holds its refusals OR one pass/abstain
        # record - never a pass record beside a refusal of its own group.
        grp = CHECK_OF.get(rid, rid)
        want_ids = [rid if cid == grp else cid for _, cid in CHECKS]
        got = [(r["rule_id"], r["verdict"]) for r in res["records"]]
        if [g[0] for g in got] != want_ids \
                or any(v == "refuse" for i, v in got if i != rid):
            raise AssertionError("fixture %s: records %r, expected ids %r"
                                 % (rid, got, want_ids))
        records.extend(res["records"])
        n_ok += 1
    shared["records"] = shared.get("records", []) + records
    return ["[ok] fixtures: %d of 15 refused naming exactly their rule" % n_ok]


def _selftest_clean(binary, tmp, shared):
    """Group 6: the clean config, 9 pass records, with a flow and a live octree probe."""
    gates, knobs = schema.load_gates(), schema.load_knobs()
    probe = shared.get("clean_probe") or octree_probe(binary, _clean_config(tmp), tmp)
    res = preflight(_clean_config(tmp), flow=dict(FLOW_OK), octree_probe=probe,
                    gates=gates, knobs=knobs)
    want = [rid for _, rid in CHECKS]
    ids = [r["rule_id"] for r in res["records"]]
    if res["verdict"] != "pass" or ids != want or len(res["records"]) != 9 \
            or any(r["verdict"] != "pass" for r in res["records"]):
        raise AssertionError("clean: verdict %r, ids %r" % (res["verdict"], ids))
    shared["records"] = shared.get("records", []) + res["records"]
    return ["[ok] clean: 9 pass records, octree probe %s leaves" % probe["n_leaves"]]


def _selftest_survey(binary, tmp, shared):
    """Group 7: the survey configs refused before any run (no mesher here)."""
    gates, knobs = schema.load_gates(), schema.load_knobs()
    box_stl = shared["box_stl"]
    want = {"NO25": ["PF-QUALITY", "PF-NONORTH"], "NO15": ["PF-QUALITY", "PF-NONORTH"],
            "NO26": ["PF-QUALITY"], "thin_t1": ["PF-THIN"]}
    records = []
    for probe_id in sorted(want):
        res = preflight(_survey_cfg(probe_id, box_stl, tmp), gates=gates, knobs=knobs)
        if res["refused"] != want[probe_id]:
            raise AssertionError("%s: refused %r, expected %r"
                                 % (probe_id, res["refused"], want[probe_id]))
        records.extend(res["records"])
    shared["records"] = shared.get("records", []) + records
    thin = [x for x in records if x["rule_id"] == "PF-THIN"]
    ratio = next((i["value"] for x in thin for i in x["inputs"] if i["name"] == "ratio"),
                 None)
    return ["[ok] survey configs: NO25, NO15 -> PF-QUALITY, PF-NONORTH; NO26 -> PF-QUALITY; "
            "thin_t1 -> PF-THIN (%r)" % ratio]


def _selftest_cthin(binary, tmp, shared):
    """Group 8: C-THIN bit for bit, against the live refusal."""
    r = c_thin_check(binary, shared["box_stl"], tmp)
    if not r["ok"]:
        raise AssertionError("c_thin_check: %r" % {k: v for k, v in r.items()
                                                   if k != "mesher_line"})
    return ["[ok] C-THIN: h %r, 3 * %r / h = %r < %r, bit-equal to the live refusal"
            % (r["h"], r["t1"], r["ratio"], r["r"])]


def _selftest_domain(binary, tmp, shared):
    """Group 9: PF-DOMAIN against the live stage 0 on 5 extents."""
    gates, knobs = schema.load_gates(), schema.load_knobs()
    extents = [([0, 4, 0, 4, 0, 4], None), ([0.5, 4, 0, 4, 0, 4], "x"),
               ([0, 4, 0, 4, 0, 3.2], "z"), ([1.2, 2.3, 0, 4, 0, 4], None),
               ([0, 3.5, 0, 4, 0, 4], None)]

    def one(item):
        k, (ext, axis) = item
        c = _clean_config(tmp)
        c["output"]["name"] = "dom%d" % k
        c["domain"]["extent"] = ext
        res = preflight(c, gates=gates, knobs=knobs)
        rec = next(r for r in res["records"] if r["rule_id"] == "PF-DOMAIN")
        probe = octree_probe(binary, c, tmp)
        return ext, axis, rec, probe

    with ThreadPoolExecutor(max_workers=4) as ex:
        outs = list(ex.map(one, enumerate(extents)))
    records = []
    for ext, axis, rec, probe in outs:
        live_pass = probe["exit_code"] == 0 and probe["n_leaves"] is not None
        if (rec["verdict"] == "pass") != live_pass:
            raise AssertionError("extent %r: preflight %s, live exit %s (%r)"
                                 % (ext, rec["verdict"], probe["exit_code"],
                                    probe.get("error")))
        if axis is not None:
            if ("on %s" % axis) not in rec["message"] \
                    or ("on %s" % axis) not in (probe.get("error") or ""):
                raise AssertionError("extent %r: axis %s not named by both"
                                     % (ext, axis))
        records.append(rec)
    shared["records"] = shared.get("records", []) + records
    return ["[ok] domain: 5 of 5 agree with the live stage 0"]


def _selftest_records(binary, tmp, shared):
    """Group 10: every record a valid DecisionRecord; refusals name their rule."""
    allowed = set(rid for _, rid in CHECKS) | set(REFUSAL_IDS)
    n = 0
    for r in shared.get("records", []):
        errs = schema.errors(r, "DecisionRecord")
        if errs:
            raise AssertionError("record %s invalid: %s" % (r.get("rule_id"), errs[0]))
        if r["rule_id"] not in allowed:
            raise AssertionError("rule id %r outside the vocabulary" % r["rule_id"])
        if r["verdict"] == "refuse" and not r["message"].startswith(r["rule_id"] + ":"):
            raise AssertionError("refusal %s does not name itself: %r"
                                 % (r["rule_id"], r["message"][:40]))
        n += 1
    if n < 20:
        raise AssertionError("only %d records collected" % n)
    return ["[ok] records: %d records valid, every refusal message starts with its rule id"
            % n]


def _selftest_cli(binary, tmp, shared):
    """Group 11: the CLI - exit 0 and 9 records clean, exit 3 PF-THIN thin_t1."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    me = os.path.abspath(__file__)
    cp = os.path.join(tmp, "cli_clean.json")
    with open(cp, "w", encoding="utf-8") as fh:
        json.dump(_clean_config(tmp), fh, indent=1)
    p = subprocess.run([sys.executable, me, cp], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env, timeout=300)
    n_rec = sum(1 for l in p.stdout.splitlines() if re.match(r"^\[(pass|abstain)\]", l))
    if p.returncode != 0 or n_rec != 9 or "PREFLIGHT PASS" not in p.stdout:
        raise AssertionError("clean CLI: exit %d, %d records: %s"
                             % (p.returncode, n_rec, (p.stdout + p.stderr)[-400:]))
    tp = os.path.join(tmp, "cli_thin_t1.json")
    with open(tp, "w", encoding="utf-8") as fh:
        json.dump(_survey_cfg("thin_t1", shared["box_stl"], tmp), fh, indent=1)
    q = subprocess.run([sys.executable, me, tp], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env, timeout=300)
    if q.returncode != 3 or "PREFLIGHT REFUSED: PF-THIN" not in q.stdout:
        raise AssertionError("thin_t1 CLI: exit %d: %s"
                             % (q.returncode, (q.stdout + q.stderr)[-400:]))
    rp = os.path.join(tmp, "records.jsonl")
    for _ in range(2):
        subprocess.run([sys.executable, me, cp, "--records", rp], capture_output=True,
                       text=True, encoding="utf-8", env=env, timeout=300)
    with open(rp, encoding="utf-8") as fh:
        n_lines = sum(1 for _ in fh)
    if n_lines != 18:
        raise AssertionError("--records appended %d lines, expected 18" % n_lines)
    return ["[ok] cli: clean exit 0 with 9 records, thin_t1 exit 3 naming PF-THIN, "
            "--records appended 9 + 9"]


def _selftest_determinism(binary, tmp, shared):
    """Group 12: the same seed gives the same configs; two preflights equal apart from t."""
    gates, knobs = schema.load_gates(), schema.load_knobs()
    template = _clean_config(tmp)
    sf = surface_facts(template)
    open_stl = shared.get("open_cubep") or open_variant(
        CUBEP_STL, os.path.join(tmp, "cubep_open12.stl"))
    missing = os.path.join(tmp, "cubep_missing12.stl")
    a = [random_config(random.Random(11), template, sf["patches"], sf["bbox"],
                       open_stl, missing) for _ in range(50)]
    b = [random_config(random.Random(11), template, sf["patches"], sf["bbox"],
                       open_stl, missing) for _ in range(50)]
    if a != b:
        raise AssertionError("the same seed drew different configs")
    c, argv, _ = a[0]
    r1 = preflight(c, argv=argv, gates=gates, knobs=knobs)
    r2 = preflight(c, argv=argv, gates=gates, knobs=knobs)
    strip = lambda res: dict(res, records=[dict(r, t="") for r in res["records"]])
    if strip(r1) != strip(r2):
        raise AssertionError("two preflights differ beyond t")
    return ["[ok] determinism: two draws of 50 configs equal, two preflights equal "
            "apart from t"]


def _selftest_sharp_ft0(binary, tmp, shared):
    """Group 13: WL-SHARP-FT0 (the user's decision of 2026-09-26) by case."""
    gates, knobs = schema.load_gates(), schema.load_knobs()
    with open(os.path.join(PROBES_DIR, "cubep_nofeat", "config.json"),
              encoding="utf-8") as fh:
        base = json.load(fh)
    with open(os.path.join(HERE, "fixtures", "remedies", "fingerprints.json"),
              encoding="utf-8") as fh:
        fp = json.load(fh)["fingerprints"]["cubep.stl"]

    def make(mut=None, fpm=None):
        c = copy.deepcopy(base)
        c["input"]["surfaces"][0]["path"] = CUBEP_STL
        c["output"]["case_dir"] = os.path.join(tmp, "ft0case").replace(os.sep, "/")
        if mut:
            mut(c)
        f = copy.deepcopy(fp)
        if fpm:
            fpm(f)
        return c, f

    def refused_by(res, rid="WL-SHARP-FT0"):
        for r in res["records"]:
            if r["rule_id"] == rid and r["verdict"] == "refuse":
                return r
        return None

    cfg, fp1 = make()
    res = preflight(cfg, fingerprint=fp1, flow=FLOW_OK, gates=gates,
                    knobs=knobs)
    refused_rec = refused_by(res)
    if refused_rec is None or not refused_rec["message"].startswith("WL-SHARP-FT0: "):
        raise AssertionError("case 1: WL-SHARP-FT0 not refused: %r" % res["refused"])

    if any(r["rule_id"] == "PF-KNOBS" and r["verdict"] == "pass" for r in res["records"]):
        raise AssertionError("case 1: a PF-KNOBS pass record stands beside the refusal")
    cfg2, fp2 = make(lambda c: c["snap"].update(smoothing_passes=0))
    res2 = preflight(cfg2, fingerprint=fp2, flow=FLOW_OK, gates=gates, knobs=knobs)
    if not plane_path(cfg2, fp2):
        raise AssertionError("case 2: the R-PLANE variant is not a plane path")
    if refused_by(res2) is not None:
        raise AssertionError("case 2: WL-SHARP-FT0 refused on the R-PLANE path")
    cfg3, fp3 = make(lambda c: c["snap"].update(feature_tolerance=0.5))
    res3 = preflight(cfg3, fingerprint=fp3, flow=FLOW_OK, gates=gates, knobs=knobs)
    if refused_by(res3) is not None:
        raise AssertionError("case 3: WL-SHARP-FT0 refused at feature_tolerance 0.5")
    cfg4, fp4 = make(fpm=lambda f: f.update(sharp_edge_length_m=0.0))
    res4 = preflight(cfg4, fingerprint=fp4, flow=FLOW_OK, gates=gates, knobs=knobs)
    if refused_by(res4) is not None:
        raise AssertionError("case 4: WL-SHARP-FT0 refused without a sharp edge")
    cfg5 = make()[0]
    res5 = preflight(cfg5, fingerprint=None, flow=FLOW_OK, gates=gates, knobs=knobs)
    if refused_by(res5) is not None:
        raise AssertionError("case 5: WL-SHARP-FT0 refused without a fingerprint")
    if plane_path({}, fp) or plane_path(cfg, {}):
        raise AssertionError("case 6: plane_path true on a malformed input")
    if plane_path(cfg, fp1) is not False:
        raise AssertionError("case 7: the off-plane base config is a plane path")
    return ["[ok] WL-SHARP-FT0: feature_tolerance 0 on cubep (18 m of sharp edge) "
            "refused by name in the PF-KNOBS group; the R-PLANE variant, "
            "feature_tolerance 0.5, a body without a sharp edge and no fingerprint "
            "are not; plane_path is False on a malformed config"]


def selftest() -> int:
    """(C11): the 13 groups; [ok] per group, SELFTEST PASS, or FAIL naming it."""
    binary = BINARY_DEFAULT
    if not os.path.isfile(binary):
        print("SELFTEST FAIL: no automesher binary at %s" % binary)
        return 1
    tmp = tempfile.mkdtemp(prefix="preflight_selftest_")
    shared = {}
    groups = (_selftest_schema, _selftest_constants, _selftest_mirror_cases,
              _selftest_random, _selftest_fixtures, _selftest_clean, _selftest_survey,
              _selftest_cthin, _selftest_domain, _selftest_records, _selftest_cli,
              _selftest_determinism, _selftest_sharp_ft0)
    try:
        box_stl = os.path.join(tmp, "box_sphere.stl")
        p = subprocess.run([sys.executable, BOX_SPHERE_GEN, box_stl], capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=300)
        if p.returncode != 0 or not os.path.isfile(box_stl):
            raise AssertionError("make_box_sphere_stl failed: %s" % (p.stderr or p.stdout))
        shared["box_stl"] = box_stl
        lines = []
        for k, fn in enumerate(groups):
            try:
                lines.extend(fn(binary, tmp, shared))
            except Exception as e:  # noqa: BLE001 - a group failure names the group
                print("SELFTEST FAIL: group %d (%s): %s" % (k + 1, fn.__name__, e))
                return 1
        for l in lines:
            print(l)
        print("SELFTEST PASS")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _take(args, i, name):
    """The value of args[i], either joined by '=' or the next token."""
    a = args[i]
    if a.startswith(name + "="):
        return i, a[len(name) + 1:]
    if a == name:
        i += 1
        if i >= len(args):
            raise PreflightError("%s needs a value" % name)
        return i, args[i]
    return i, None


def _read_json_or(path, what):
    if not path:
        raise PreflightError("%s needs a path" % what)
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as e:
        raise PreflightError("%s %s is unreadable: %s" % (what, path, e))


def _main_config(args):
    """preflight.py CONFIG [options] - the L0 verdict on one config."""
    config_path, argv, edits, flow, fingerprint = None, [], None, None, None
    do_octree = do_snap = as_json = False
    h_min, binary, records_path = None, BINARY_DEFAULT, None
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--arg" or a.startswith("--arg="):
            i, v = _take(args, i, "--arg")
            argv.append(v)
        elif a == "--edits" or a.startswith("--edits="):
            i, v = _take(args, i, "--edits")
            edits = _read_json_or(v, "--edits")
        elif a == "--flow" or a.startswith("--flow="):
            i, v = _take(args, i, "--flow")
            flow = _read_json_or(v, "--flow")
        elif a == "--fingerprint" or a.startswith("--fingerprint="):
            i, v = _take(args, i, "--fingerprint")
            fingerprint = _read_json_or(v, "--fingerprint")
        elif a == "--h-wall-min" or a.startswith("--h-wall-min="):
            i, v = _take(args, i, "--h-wall-min")
            try:
                h_min = float(v)
            except ValueError:
                raise PreflightError("--h-wall-min needs a number, got %r" % v)
        elif a == "--binary" or a.startswith("--binary="):
            i, v = _take(args, i, "--binary")
            binary = v
        elif a == "--records" or a.startswith("--records="):
            i, v = _take(args, i, "--records")
            records_path = v
        elif a == "--octree-probe":
            do_octree = True
        elif a == "--snap-probe":
            do_snap = True
        elif a == "--json":
            as_json = True
        elif a.startswith("-"):
            raise PreflightError("unknown argument %r" % a)
        elif config_path is None:
            config_path = a
        else:
            raise PreflightError("one CONFIG positional is taken, got a second one %r" % a)
        i += 1
    if config_path is None:
        raise PreflightError("a CONFIG positional is required")
    config = _read_json_or(config_path, "CONFIG")
    if not isinstance(config, dict):
        raise PreflightError("CONFIG %s is not a JSON object" % config_path)
    probe, tmp = None, None
    try:
        if do_octree or do_snap:
            tmp = tempfile.mkdtemp(prefix="preflight_cli_")
            if do_snap:
                h_min = snap_probe(binary, config, tmp)
            if do_octree:
                probe = octree_probe(binary, config, tmp)
        res = preflight(config, argv=argv, edits=edits, fingerprint=fingerprint, flow=flow,
                        octree_probe=probe, h_wall_min_m=h_min)
        if records_path:
            with open(records_path, "a", encoding="utf-8") as fh:
                for r in res["records"]:
                    fh.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")
        if as_json:
            print(json.dumps(res, indent=1, ensure_ascii=False))
        else:
            for r in res["records"]:
                print("[%s] %s" % (r["verdict"], r["message"]))
        if res["verdict"] == "pass":
            print("PREFLIGHT PASS")
            return 0
        print("PREFLIGHT REFUSED: %s" % ", ".join(res["refused"]))
        return 3
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


def _main_gate(args):
    """preflight.py --gate ... - G-PREFLIGHT into --out and REPORT_DIR."""
    stls, out_dir = {}, None
    n, seed, streams = 10000, 1, 6
    parts, binary = ["1", "2", "3"], BINARY_DEFAULT
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--stl" or a.startswith("--stl="):
            i, v = _take(args, i, "--stl")
            nm, sep, p = v.partition("=")
            if not nm or not sep or not p:
                raise PreflightError("--stl needs NAME=PATH")
            stls[nm] = p
        elif a == "--out" or a.startswith("--out="):
            i, v = _take(args, i, "--out")
            out_dir = v
        elif a == "--n" or a.startswith("--n="):
            i, v = _take(args, i, "--n")
            n = int(v)
        elif a == "--seed" or a.startswith("--seed="):
            i, v = _take(args, i, "--seed")
            seed = int(v)
        elif a == "--streams" or a.startswith("--streams="):
            i, v = _take(args, i, "--streams")
            streams = int(v)
        elif a == "--parts" or a.startswith("--parts="):
            i, v = _take(args, i, "--parts")
            parts = [s.strip() for s in v.split(",") if s.strip()]
        elif a == "--binary" or a.startswith("--binary="):
            i, v = _take(args, i, "--binary")
            binary = v
        else:
            raise PreflightError("unknown argument %r" % a)
        i += 1
    if not out_dir:
        raise PreflightError("--out DIR is required")
    if not os.path.isfile(binary):
        raise PreflightError("no automesher binary at %s" % binary)
    if ("1" in parts or "3" in parts) and not ("box_sphere" in stls and "wing_b" in stls):
        raise PreflightError("--parts 1/3 need --stl box_sphere=P and --stl wing_b=P")
    if "2" in parts and "box_sphere" not in stls:
        raise PreflightError("--parts 2 needs --stl box_sphere=P")
    return g_preflight(stls, n, seed, streams, out_dir, parts, binary)


def main(argv=None):
    """The CLI: 0 pass, 3 refused, 2 a caller error, 1 a failed selftest or gate."""
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        if "--selftest" in args:
            if len(args) != 1:
                raise PreflightError("--selftest takes no other argument")
            return selftest()
        if "--gate" in args:
            args.remove("--gate")
            return _main_gate(args)
        return _main_config(args)
    except PreflightError as e:
        print("preflight: %s" % e, file=sys.stderr)
        return 2
    except (schema.LockError, schema.SchemaError) as e:
        print("preflight: %s" % e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    sys.exit(main())
