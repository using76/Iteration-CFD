#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""Mutation fixture (docs/16 §E.5): the runner-child entry that builds the nominal nozzle through the template,
then plants ONE deterministic defect and re-tags. mutate.py is its only caller; no production path imports it."""
import copy
import importlib.util
import os

import cadquery as cq

V = cq.Vector
EXIT_SCALE = 1.05      # M01: exit diameter +5 % with D_i kept, so CR / 1.05^2
L_SCALE = 0.9          # M03: contraction length -10 %
LIP_M = 5e-5           # M07: the 0.05 mm land at the outlet end of the body
MATCH_TOL = 1e-12      # m: a reloaded wall edge is matched to its construction edge by midpoint
NAMES = {"M00": "identity", "M01": "exit_d_plus5", "M02": "wall_radial", "M03": "contraction_l_minus10",
         "M04": "outlet_face_missing", "M05": "bow_tie", "M06": "axis_crossing", "M07": "lip_0p05mm",
         "M08": "axis_on_z", "M09": "stray_second_solid", "M10": "mm_as_m"}


def load_template(path):
    spec = importlib.util.spec_from_file_location("nozzle_template_under_mutation", path)
    tpl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tpl)
    return tpl


def _p(out_dir, name):
    return os.path.join(out_dir, name)


def _load(out_dir, name):
    return cq.Shape.importBrep(_p(out_dir, name))


def _save(out_dir, name, shape):
    if not shape.exportBrep(_p(out_dir, name)):
        raise RuntimeError("exportBrep returned False for %s" % (name,))


def _line(a, b):
    return cq.Edge.makeLine(V(a[0], a[1], 0), V(b[0], b[1], 0))


def _face(edges):
    return cq.Face.makeFromWires(cq.Wire.assembleEdges(edges))


def _revolve(face):
    return cq.Solid.revolve(face, 360, V(0, 0, 0), V(1, 0, 0))


def _by_x(edges):
    return sorted(edges, key=lambda e: e.positionAt(0.5).x)


def _edges(shape, idx):
    es = shape.Edges()
    return [es[i] for i in idx]

def _meridian_mutant(tpl, out_dir, value, edges):
    """M05 / M06: a new fluid meridian from `edges`, revolved; tags from the template's own rules."""
    _save(out_dir, "meridian.brep", _face(edges))
    _save(out_dir, "fluid.brep", _revolve(_load(out_dir, "meridian.brep")))
    d = value["derived"]
    value["face_tags"] = tpl.classify_fluid(_load(out_dir, "fluid.brep"), d)
    value["meridian_edges"] = tpl.classify_meridian(_load(out_dir, "meridian.brep"), d)
    return value


def _wall_mutant(out_dir, value, outer, outer_end):
    """M02 / M07: the body meridian rebuilt from the nominal wetted edges and a new outer wall; tags by construction."""
    d = value["derived"]
    wet = _by_x(_edges(_load(out_dir, "wall_meridian.brep"), value["wall_edges"]["wetted"]))
    start = _line((0.0, d["R_i"]), (0.0, d["R_i"] + d["t"]))
    end = _line(outer_end, (d["x_outlet"], d["R_e"]))
    _save(out_dir, "wall_meridian.brep", _face([start] + outer + [end] + list(reversed(wet))))
    _save(out_dir, "body.brep", _revolve(_load(out_dir, "wall_meridian.brep")))
    mids = lambda es: [e.positionAt(0.5) for e in es]
    near = lambda m, ms: any((m - q).Length <= MATCH_TOL for q in ms)
    tags = {"wetted": [], "outer": [], "ends": []}
    for i, e in enumerate(_load(out_dir, "wall_meridian.brep").Edges()):
        m = e.positionAt(0.5)
        key = "wetted" if near(m, mids(wet)) else ("outer" if near(m, mids(outer)) else "ends")
        tags[key].append(i)
    if len(tags["ends"]) != 2:
        raise RuntimeError("mutant wall meridian has %d end edges, want 2" % (len(tags["ends"]),))
    value["wall_edges"] = tags
    return value

def build(params, out_dir):
    """The runner entry: params {template, params, mutant}; the nominal build, then mutant `mutant` planted."""
    tpl = load_template(params["template"])
    mid = params["mutant"]
    if mid not in NAMES:
        raise ValueError("unknown mutant %r" % (mid,))
    p = dict(params["params"])
    if mid == "M01":
        p["CR"] = p["CR"] / EXIT_SCALE ** 2
    if mid == "M03":
        p["L_over_Di"] = p["L_over_Di"] * L_SCALE
    value = tpl.build(p, out_dir)
    if value["status"] != "ok":
        raise RuntimeError("the mutant's template build was refused: %s %s" % (value["rule"], value["detail"]))
    value = copy.deepcopy(value)
    d = value["derived"]
    if mid == "M02":
        wet = _by_x(_edges(_load(out_dir, "wall_meridian.brep"), value["wall_edges"]["wetted"]))
        outer = [e.translate(V(0, d["t"], 0)) for e in wet]
        value = _wall_mutant(out_dir, value, outer, (d["x_outlet"], d["R_e"] + d["t"]))
    elif mid == "M04":
        fl = _load(out_dir, "fluid.brep")
        keep = [f for i, f in enumerate(fl.Faces()) if i not in value["face_tags"]["outlet"]]
        _save(out_dir, "fluid.brep", cq.Shell.makeShell(keep))
        value["face_tags"] = tpl.classify_fluid(_load(out_dir, "fluid.brep"), d)
    elif mid == "M05":
        me = _load(out_dir, "meridian.brep")
        t = value["meridian_edges"]
        head = _edges(me, t["inlet"]) + _edges(me, t["slip_upstream"]) + _by_x(_edges(me, t["wall_contraction"]))
        L, Re, xo = d["L"], d["R_e"], d["x_outlet"]
        tail = [_line((L, Re), (xo, 0.0)), _line((xo, 0.0), (xo, Re)), _line((xo, Re), (L, 0.0)),
                _line((L, 0.0), (d["x_inlet"], 0.0))]
        value = _meridian_mutant(tpl, out_dir, value, head + tail)

    elif mid == "M06":
        me = _load(out_dir, "meridian.brep")
        t = value["meridian_edges"]
        keep = []
        for tag in ("inlet", "slip_upstream", "wall_contraction", "wall_exit", "outlet"):
            keep += _by_x(_edges(me, t[tag]))
        xm = 0.5 * (d["x_inlet"] + d["x_outlet"])
        sag = cq.Edge.makeThreePointArc(V(d["x_outlet"], 0, 0), V(xm, -0.5 * d["R_e"], 0), V(d["x_inlet"], 0, 0))
        value = _meridian_mutant(tpl, out_dir, value, keep + [sag])
    elif mid == "M07":
        wm = _load(out_dir, "wall_meridian.brep")
        outs = _by_x(_edges(wm, value["wall_edges"]["outer"]))
        tube = outs[-1]
        a = min((tube.startPoint(), tube.endPoint()), key=lambda q: q.x)
        xc = d["x_outlet"] - 0.5 * d["Lx"]
        outer = outs[:-1] + [_line((a.x, a.y), (xc, a.y)), _line((xc, a.y), (d["x_outlet"], d["R_e"] + LIP_M))]
        value = _wall_mutant(out_dir, value, outer, (d["x_outlet"], d["R_e"] + LIP_M))
    elif mid == "M08":
        for name in ("fluid.brep", "body.brep", "meridian.brep", "wall_meridian.brep"):
            _save(out_dir, name, _load(out_dir, name).rotate(V(0, 0, 0), V(0, 1, 0), -90))
    elif mid == "M09":
        _save(out_dir, "fluid.brep", cq.Compound.makeCompound([_load(out_dir, "fluid.brep"),
                                                                _load(out_dir, "body.brep")]))
    value["mutant"] = {"id": mid, "name": NAMES[mid]}
    return value
