#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""admit_probe.py - the runner child of admit.py (CAD-23): loads ONE candidate template by path and
answers the `declare` and `sweep` entries. The admission parent never imports a candidate; only this
child does, under the module name cad_admission_candidate. The sweep entry wraps the cadquery
boolean entry points so each build's cut/fuse/intersect calls record the measure before and after,
which is what ADM-STAGE's no-op rule judges; the wrappers call and return the originals unchanged.

Usage (spawned by runner.py only, through child.py):
  entry declare, job {"candidate": <abs path>} -> the declaration value of admit.py's ADM-CONTRACT
  entry sweep,   job {"candidate": <abs path>, "points": [params dicts]} -> {"points": [records]}
"""

import importlib.util
import math
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402

MODULE_NAME = "cad_admission_candidate"
TRACE_TAIL = 4096
POINT_KEYS = ("i", "accepted", "status", "rule", "detail", "error", "traceback", "stage", "booleans",
              "face_tags", "meridian_edges", "wall_edges", "untagged", "duplicated", "n_fluid_faces",
              "fluid_area", "planes", "span")
RESULT_REQUIRED = ("template_id", "status", "rule", "detail", "params", "derived", "planes", "files",
                   "face_tags", "meridian_edges", "wall_edges")
STAGE_FILES = (("fluid.brep", "solid"), ("body.brep", "solid"),
               ("meridian.brep", "face"), ("wall_meridian.brep", "face"))
_STATE = {"booleans": []}


def load_candidate(path):
    """Import the candidate by path, in THIS child only, under cad_admission_candidate."""
    spec = importlib.util.spec_from_file_location(MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load the candidate from %s" % (path,))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = mod
    spec.loader.exec_module(mod)
    return mod


def _tail(text):
    return (text or "")[-TRACE_TAIL:]


def _measure(shp):
    """Sum of solid volumes when the shape has solids, else sum of face areas, else 0.0."""
    try:
        solids = shp.Solids()
    except Exception:
        solids = []
    if solids:
        return math.fsum(s.Volume() for s in solids)
    try:
        faces = shp.Faces()
    except Exception:
        faces = []
    if faces:
        return math.fsum(f.Area() for f in faces)
    return 0.0


def _install_recorder():
    """Wrap the cadquery boolean entry points for the whole child; originals pass through."""
    import cadquery as cq
    from cadquery.occ_impl import shapes as occ_impl_shapes

    def wrap(owner, name):
        original = getattr(owner, name)

        def recorder(*args, **kwargs):
            result = original(*args, **kwargs)
            before = _measure(args[0]) if args else 0.0
            after = _measure(result)
            if before > 0:
                rel = abs(after - before) / before
            elif after == before:
                rel = 0.0
            else:
                rel = 1.0
            _STATE["booleans"].append({"op": name, "before": before, "after": after, "rel": rel})
            return result

        setattr(owner, name, recorder)

    for name in ("cut", "fuse", "intersect"):
        wrap(cq.Shape, name)
        wrap(cq.Compound, name)
        wrap(occ_impl_shapes, name)


def _nominal_of(params_rows):
    out = {}
    for row in params_rows:
        if row["kind"] == "real":
            out[row["name"]] = row["default_real"]
        else:
            out[row["name"]] = row["default_choice"]
    return out


def declare(job, out_dir):
    """The declaration value ADM-CONTRACT judges: the five module tables, DRIVERS, the nominal rule."""
    mod = load_candidate(job["candidate"])
    rule = mod.domain_rules(_nominal_of(mod.PARAMS))
    value = {"template_id": mod.TEMPLATE_ID, "params": mod.PARAMS, "planes": mod.PLANES,
             "tags": mod.TAGS, "catalogue": mod.CATALOGUE,
             "drivers": getattr(mod, "DRIVERS", None),
             "profile_rules": list(getattr(mod, "PROFILE_RULES", []) or []),
             "standards": list(getattr(mod, "STANDARDS", []) or []),
             "nominal_rule": None if rule is None else [rule[0], rule[1]]}
    common.canonical_json(value)    # refuses NaN and non-JSON before anything is returned
    return value


def _blank(i):
    rec = {}
    for k in POINT_KEYS:
        rec[k] = None
    rec.update({"i": i, "accepted": False, "status": None, "rule": None, "detail": None,
                "error": None, "traceback": None, "stage": [], "booleans": [], "face_tags": {},
                "meridian_edges": {}, "wall_edges": {}, "untagged": [], "duplicated": [],
                "n_fluid_faces": 0, "fluid_area": 0.0, "planes": [], "span": None})
    return rec


def _bad_index(value, tag, j, n, what):
    if isinstance(j, bool) or not isinstance(j, int) or not 0 <= j < n:
        return "%s tag %s indexes %r out of %d %s" % (what, tag, j, n, what)
    return None


def _fill(rec, value, point_dir):
    """Stage failures, tag and edge tables on one ok point; a detail string when malformed."""
    import cadquery as cq
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Check
    from OCP.BRepCheck import BRepCheck_Analyzer
    p = lambda name: os.path.join(point_dir, name)
    stage = []
    for _logical, fname in sorted((value.get("files") or {}).items()):
        if not os.path.isfile(p(fname)):
            stage.append("%s: missing" % (fname,))
    for fname, kind in STAGE_FILES:
        if not os.path.isfile(p(fname)) and not any(s.startswith(fname + ":") for s in stage):
            stage.append("%s: missing" % (fname,))
    if any(s.endswith(": missing") for s in stage):
        rec["stage"] = stage
        return None
    fluid = cq.Shape.importBrep(p("fluid.brep"))
    body = cq.Shape.importBrep(p("body.brep"))
    meridian = cq.Shape.importBrep(p("meridian.brep"))
    wall_m = cq.Shape.importBrep(p("wall_meridian.brep"))
    n_fluid = len(fluid.Faces())
    n_mer = len(meridian.Edges())
    n_wall = len(wall_m.Edges())
    for tag, idxs in (value.get("face_tags") or {}).items():
        for j in idxs:
            bad = _bad_index(value, tag, j, n_fluid, "face")
            if bad:
                return bad
    for tag, idxs in (value.get("meridian_edges") or {}).items():
        for j in idxs:
            bad = _bad_index(value, tag, j, n_mer, "meridian edge")
            if bad:
                return bad
    for tag, idxs in (value.get("wall_edges") or {}).items():
        for j in idxs:
            bad = _bad_index(value, tag, j, n_wall, "wall edge")
            if bad:
                return bad
    for fname, kind in STAGE_FILES:
        shp = fluid if fname == "fluid.brep" else body if fname == "body.brep" \
            else meridian if fname == "meridian.brep" else wall_m
        if not BRepCheck_Analyzer(shp.wrapped).IsValid():
            stage.append("%s: BRepCheck invalid" % (fname,))
        if kind == "solid":
            sols = shp.Solids()
            if len(sols) != 1:
                stage.append("%s: %d solids, want 1" % (fname, len(sols)))
            elif not BRepAlgoAPI_Check(shp.wrapped).IsValid():
                stage.append("%s: BRepAlgoAPI invalid" % (fname,))
        else:
            fs = shp.Faces()
            if len(fs) != 1:
                stage.append("%s: %d faces, want 1" % (fname, len(fs)))
    face_tags = {}
    for tag, idxs in (value.get("face_tags") or {}).items():
        face_tags[tag] = {"n": len(idxs),
                          "area": math.fsum(fluid.Faces()[j].Area() for j in idxs)}
    meridian_edges = dict((tag, math.fsum(meridian.Edges()[j].Length() for j in idxs))
                          for tag, idxs in (value.get("meridian_edges") or {}).items())
    wall_edges = dict((tag, math.fsum(wall_m.Edges()[j].Length() for j in idxs))
                      for tag, idxs in (value.get("wall_edges") or {}).items())
    owner = {}
    for tag, idxs in (value.get("face_tags") or {}).items():
        for j in idxs:
            owner.setdefault(j, []).append(tag)
    untagged = [j for j in range(n_fluid) if j not in owner]
    duplicated = sorted(j for j, ts in owner.items() if len(ts) > 1)
    rec.update({"stage": stage, "face_tags": face_tags, "meridian_edges": meridian_edges,
                "wall_edges": wall_edges, "untagged": untagged, "duplicated": duplicated,
                "n_fluid_faces": n_fluid,
                "fluid_area": math.fsum(f.Area() for f in fluid.Faces()),
                "planes": [q["name"] for q in value["planes"]],
                "span": value["derived"]["x_outlet"] - value["derived"]["x_inlet"]})
    return None


def _run_point(mod, i, params, point_dir):
    rec = _blank(i)
    try:
        ref = mod.domain_rules(dict(params))
    except Exception as e:
        rec["accepted"] = True    # a raising domain_rules is the child failing on the point,
        rec["status"] = "error"   # so the ADM-BUILD judging in admit.py must see it
        rec["error"] = "%s: %s" % (type(e).__name__, e)
        rec["traceback"] = _tail(traceback.format_exc())
        return rec
    if ref is not None:
        rec["rule"], rec["detail"] = ref[0], ref[1]
        return rec
    rec["accepted"] = True
    _STATE["booleans"] = []
    os.makedirs(point_dir, exist_ok=True)
    try:
        value = mod.build(dict(params), point_dir)
    except Exception as e:
        rec["status"] = "error"
        rec["error"] = "%s: %s" % (type(e).__name__, e)
        rec["traceback"] = _tail(traceback.format_exc())
        rec["booleans"] = list(_STATE["booleans"])
        return rec
    rec["booleans"] = list(_STATE["booleans"])
    if not isinstance(value, dict):
        rec["status"] = "malformed"
        rec["detail"] = "the build result is %s, not a dict" % (type(value).__name__,)
        return rec
    for k in RESULT_REQUIRED:
        if k not in value:
            rec["status"] = "malformed"
            rec["detail"] = "the build result lacks %r" % (k,)
            return rec
    if value["status"] not in ("ok", "refused"):
        rec["status"] = "malformed"
        rec["detail"] = "the build status is %r, not ok or refused" % (value["status"],)
        return rec
    if value["status"] == "refused":
        rec["status"] = "refused"
        rec["rule"], rec["detail"] = value["rule"], value["detail"]
        return rec
    d = value["derived"]
    if not isinstance(d, dict) or any(k not in d for k in ("x_inlet", "x_outlet", "D_e")):
        rec["status"] = "malformed"
        rec["detail"] = "derived lacks x_inlet, x_outlet or D_e"
        return rec
    rec["status"] = "ok"
    bad = _fill(rec, value, point_dir)
    if bad:
        rec["status"] = "malformed"
        rec["detail"] = bad
        rec["stage"] = []
        rec["booleans"] = list(rec["booleans"])
    return rec


def sweep(job, out_dir):
    """One record per point; the boolean recorder is installed for the whole child."""
    mod = load_candidate(job["candidate"])
    _install_recorder()
    points = []
    for i, p in enumerate(job["points"]):
        points.append(_run_point(mod, i, p, os.path.join(out_dir, "p%03d" % i)))
    return {"points": points}
