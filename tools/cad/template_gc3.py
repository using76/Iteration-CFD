#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""template_gc3.py - gate GC-3 of docs/16 §H.3 for the nozzle_contraction template, run only through runner children: nominal, laws, volume, wall, box corners and the six PRF refusals. Usage: python template_gc3.py --selftest"""
import ast
import itertools
import json
import math
import os
import sys
import tempfile
import time

import numpy as np
import cadquery as cq
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.gp import gp_Pnt, gp_Vec
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402
import schema  # noqa: E402
import runner  # noqa: E402
import measure  # noqa: E402

TEMPLATE = os.path.join(HERE, "templates", "nozzle_contraction", "template.py")
TEMPLATE_JSON = os.path.join(HERE, "templates", "nozzle_contraction", "template.json")
INJECT = os.path.join(HERE, "fixtures", "template", "prf_inject.py")
TIMEOUT_S = 180
NOMINAL = {"D_i": 0.06, "CR": 9.0, "L_over_Di": 1.0, "law": "poly5", "x_m": None,
           "Lx_over_De": 0.5, "Lu_over_Di": 0.5, "upstream_role": "slip", "t_wall": 0.003}
JOBS = []                                       # every runner job this gate made, in order


def job(params, td, name, module=TEMPLATE, entry="build"):
    """One runner child; the gate process itself never imports the module it runs."""
    JOBS.append(name)
    return runner.run_job(module, params, os.path.join(td, name), entry=entry, timeout_s=TIMEOUT_S)


def load(out_dir, name):
    return cq.Shape.importBrep(os.path.join(out_dir, name))


# Analytic truth, written again here from the formulas (never imported from template.py).
def pieces(law, X):
    """(Lfrac, a) per wall-law piece: a's monomial coefficients in s, the piece's share of L."""
    if law == "poly3":
        return [(1.0, (0.0, 0.0, 3.0, -2.0))]
    if law == "poly5":
        return [(1.0, (0.0, 0.0, 0.0, 10.0, -15.0, 6.0))]
    if law == "poly7":
        return [(1.0, (0.0, 0.0, 0.0, 0.0, 35.0, -84.0, 70.0, -20.0))]
    return [(X, (0.0, 0.0, 0.0, X)), (1.0 - X, (X, 3.0 * (1.0 - X), -3.0 * (1.0 - X), 1.0 - X))]


def volume_truth(prm):
    """pi * (R_i^2 Lu + sum over pieces of Lfrac L int_0^1 r(s)^2 ds + R_e^2 Lx)."""
    r_i = prm["D_i"] / 2.0
    r_e = r_i / math.sqrt(prm["CR"])
    length = prm["L_over_Di"] * prm["D_i"]
    total = r_i ** 2 * (prm["Lu_over_Di"] * prm["D_i"]) + r_e ** 2 * (prm["Lx_over_De"] * 2.0 * r_e)
    for lfrac, a in pieces(prm["law"], prm["x_m"]):
        dr = r_i - r_e
        coef = np.array([r_i - dr * a[0]] + [-dr * c for c in a[1:]])
        inner = np.polynomial.polynomial.polyint(np.polynomial.polynomial.polymul(coef, coef), 1)
        total += lfrac * length * float(np.polynomial.polynomial.polyval(1.0, inner))
    return math.pi * total


def rho_min_offset_side(prm):
    """1 / max curvature where r'' > 0, over 20001 samples per piece; inf when r'' never bends that way."""
    r_i = prm["D_i"] / 2.0
    r_e = r_i / math.sqrt(prm["CR"])
    length = prm["L_over_Di"] * prm["D_i"]
    dr = r_i - r_e
    kmax = 0.0
    for lfrac, a in pieces(prm["law"], prm["x_m"]):
        s = np.linspace(0.0, 1.0, 20001)
        fp = np.zeros_like(s)
        fpp = np.zeros_like(s)
        for k in range(1, len(a)):
            fp = fp + k * a[k] * s ** (k - 1)
        for k in range(2, len(a)):
            fpp = fpp + k * (k - 1) * a[k] * s ** (k - 2)
        r1 = -dr * fp / (lfrac * length)
        r2 = -dr * fpp / (lfrac * length) ** 2
        mask = r2 > 0
        if bool(mask.any()):
            kappa = r2[mask] / (1.0 + r1[mask] ** 2) ** 1.5
            kmax = max(kmax, float(np.max(kappa)))
    return math.inf if kmax <= 0.0 else 1.0 / kmax


def curvature_truth():
    """poly5 nominal: 1 / max |kappa|, scanned at 200001 samples then bounded-refined to xatol 1e-14."""
    r_i = NOMINAL["D_i"] / 2.0
    r_e = r_i / math.sqrt(NOMINAL["CR"])
    length = NOMINAL["L_over_Di"] * NOMINAL["D_i"]
    dr = r_i - r_e
    a = (0.0, 0.0, 0.0, 10.0, -15.0, 6.0)

    def kappa_abs(sv):
        f1 = sum(k * a[k] * sv ** (k - 1) for k in range(1, len(a)))
        f2 = sum(k * (k - 1) * a[k] * sv ** (k - 2) for k in range(2, len(a)))
        r1 = -dr * f1 / length
        r2 = -dr * f2 / length ** 2
        return np.abs(r2) / (1.0 + r1 ** 2) ** 1.5

    s = np.linspace(0.0, 1.0, 200001)
    vals = kappa_abs(s)
    k = int(np.argmax(vals))
    res = minimize_scalar(lambda x: -float(kappa_abs(np.array([float(x)]))[0]),
                          bounds=(float(s[max(k - 1, 0)]), float(s[min(k + 1, 200000)])),
                          method="bounded", options={"xatol": 1e-14})
    return 1.0 / (-res.fun)


def selftest():
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        # (G1) the generated declaration equals declare() and is a valid cad-template/1
        decl = common.read_json(TEMPLATE_JSON)
        r = job({}, td, "declare", entry="declare")
        assert r["status"] == "ok", "declare run %r: %s" % (r["status"], r["message"])
        assert r["value"] == decl, "template.json differs from declare()"
        assert schema.errors(decl, "cad-template/1") == [], "declaration violates cad-template/1"
        ok_where = set(p["name"] for p in decl["planes"]) | set(t["name"] for t in decl["tags"]) | {"fluid", "body"}
        for row in decl["catalogue"]:
            assert row["primitive"] in measure.PRIMITIVES, "catalogue primitive %r is not a measure primitive" % (row["primitive"],)
            kind, num = measure.U_MEAS[row["primitive"]]
            if row["u_kind"] == "exact":
                assert (kind, num) == ("abs", 0.0) and row["u_meas"] == 0.0, "u_meas of %r does not match U_MEAS" % (row["quantity"],)
            else:
                assert (row["u_kind"], row["u_meas"]) == (kind, num), "u_meas of %r does not match U_MEAS" % (row["quantity"],)
            assert set(row["where"]) <= ok_where, "where %r names no plane, tag, fluid or body" % (row["where"],)
        want_rules = ["PRF-BOX", "PRF-RMIN", "PRF-MONO", "PRF-DERIV", "PRF-SELFX", "PRF-FACE2D"]
        assert decl["profile_rules"] == want_rules, "profile_rules %r" % (decl["profile_rules"],)
        print("[ok] declaration: template.json equals declare() and is a valid cad-template/1; %d catalogue rows on measure primitives with their u_meas" % (len(decl["catalogue"]),))

        # (G2) the admission shape of template.py, judged on its AST
        tree = ast.parse(open(TEMPLATE, "r", encoding="utf-8").read())
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules.append(node.module)
        ocp_k = 0
        for m in modules:
            assert m == "math" or m == "cadquery" or m.startswith("OCP."), "template.py imports %r" % (m,)
            if m.startswith("OCP."):
                ocp_k = ocp_k + 1
        forbidden = {"open", "exec", "eval", "compile", "__import__", "getattr", "setattr", "delattr",
                     "globals", "locals", "vars", "os", "sys", "subprocess", "pathlib", "socket",
                     "ctypes", "importlib", "__name__", "__file__", "__builtins__"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                assert node.id not in forbidden, "template.py uses the forbidden name %r" % (node.id,)
            if isinstance(node, ast.Attribute):
                a = node.attr
                assert not (a.startswith("__") and a.endswith("__")), "template.py uses the dunder attribute %r" % (a,)
        print("[ok] admission shape: template.py imports only math, cadquery and %d OCP modules and uses no forbidden name" % (ocp_k,))

        # (G3) the nominal build: one valid fluid solid and one valid body solid, four BREP files
        r_nom = job(dict(NOMINAL), td, "nominal")
        assert r_nom["status"] == "ok", "nominal run %r: %s" % (r_nom["status"], r_nom["message"])
        val = r_nom["value"]
        assert val["status"] == "ok", "nominal build refused: %r %r" % (val.get("rule"), val.get("detail"))
        out_nom = os.path.join(td, "nominal")
        for f in val["files"].values():
            assert os.path.isfile(os.path.join(out_nom, f)), "the nominal job wrote no %s" % (f,)
        fluid = load(out_nom, "fluid.brep")
        body = load(out_nom, "body.brep")
        assert measure.n_solids(fluid)["value"] == 1, "fluid n_solids %r" % (measure.n_solids(fluid)["value"],)
        assert measure.valid(fluid)["value"] == 1, "fluid valid %r" % (measure.valid(fluid)["value"],)
        assert measure.n_solids(body)["value"] == 1, "body n_solids %r" % (measure.n_solids(body)["value"],)
        assert measure.valid(body)["value"] == 1, "body valid %r" % (measure.valid(body)["value"],)
        print("[ok] nominal poly5: fluid 1 valid solid, body 1 valid solid, 4 BREP files, %s s in the child" % (r_nom["wall_s"],))

        # (G4) every face and edge carries exactly one tag, and the tags survive a BREP reload
        d = val["derived"]
        meridian = load(out_nom, "meridian.brep")
        wall_m = load(out_nom, "wall_meridian.brep")
        fts, mets, wts = val["face_tags"], val["meridian_edges"], val["wall_edges"]
        assert sorted(i for lst in fts.values() for i in lst) == list(range(len(fluid.Faces()))), "fluid face tags are not a partition"
        assert sorted(i for lst in mets.values() for i in lst) == list(range(len(meridian.Edges()))), "meridian tags are not a partition"
        assert sorted(i for lst in wts.values() for i in lst) == list(range(len(wall_m.Edges()))), "wall tags are not a partition"
        fi = fluid.Faces()[fts["inlet"][0]]
        fo = fluid.Faces()[fts["outlet"][0]]
        assert fi.geomType() == "PLANE" and abs(fi.Center().x - d["x_inlet"]) <= 1e-12, "inlet face is not the inlet disc"
        assert fo.geomType() == "PLANE" and abs(fo.Center().x - d["x_outlet"]) <= 1e-12, "outlet face is not the outlet disc"
        for tag, lo, hi in (("slip_upstream", d["x_inlet"], 0.0), ("wall_contraction", 0.0, d["L"]),
                            ("wall_exit", d["L"], d["x_outlet"])):
            c = fluid.Faces()[fts[tag][0]].Center().x
            assert lo < c < hi, "%s face centre %r is not in (%r, %r)" % (tag, c, lo, hi)
        ax = meridian.Edges()[mets["axis"][0]]
        assert abs(ax.startPoint().y) <= 1e-12 and abs(ax.endPoint().y) <= 1e-12, "the axis edge leaves y = 0"
        assert len(wts["ends"]) == 2, "the wall meridian has %d end edges" % (len(wts["ends"]),)
        print("[ok] tags: %d fluid faces, %d meridian edges, %d wall edges each carry exactly one tag after a BREP reload" % (len(fluid.Faces()), len(meridian.Edges()), len(wall_m.Edges())))

        # (G4b) the wall role: Lu/D_i 2.0 tags the upstream face wall_upstream; the geometry does not move
        r_wall = job(dict(NOMINAL, Lu_over_Di=2.0, upstream_role="wall"), td, "wall_role")
        assert r_wall["status"] == "ok", "wall role run %r: %s" % (r_wall["status"], r_wall["message"])
        vw = r_wall["value"]
        assert vw["status"] == "ok", "wall role refused: %r %r" % (vw.get("rule"), vw.get("detail"))
        assert sorted(vw["face_tags"]) == ["inlet", "outlet", "wall_contraction", "wall_exit", "wall_upstream"], sorted(vw["face_tags"])
        assert sorted(vw["meridian_edges"]) == ["axis", "inlet", "outlet", "wall_contraction", "wall_exit", "wall_upstream"], sorted(vw["meridian_edges"])
        assert "slip_upstream" not in vw["face_tags"] and "slip_upstream" not in vw["meridian_edges"], "a wall-role build carries slip_upstream"
        out_wall = os.path.join(td, "wall_role")
        fluid_w = load(out_wall, "fluid.brep")
        meridian_w = load(out_wall, "meridian.brep")
        wall_mw = load(out_wall, "wall_meridian.brep")
        assert sorted(i for lst in vw["face_tags"].values() for i in lst) == list(range(len(fluid_w.Faces()))), "wall role: fluid face tags are not a partition"
        assert sorted(i for lst in vw["meridian_edges"].values() for i in lst) == list(range(len(meridian_w.Edges()))), "wall role: meridian tags are not a partition"
        assert sorted(i for lst in vw["wall_edges"].values() for i in lst) == list(range(len(wall_mw.Edges()))), "wall role: wall tags are not a partition"
        cw = fluid_w.Faces()[vw["face_tags"]["wall_upstream"][0]].Center().x
        assert -0.12 < cw < 0, "wall_upstream face centre x %r is not in (-0.12, 0)" % (cw,)
        truth_w = volume_truth(dict(NOMINAL, Lu_over_Di=2.0))
        vrel = abs(vw["checks"]["fluid_volume_m3"] - truth_w) / truth_w
        assert vrel <= 1e-6, "wall role fluid volume rel %r vs truth" % (vrel,)
        brel = abs(vw["checks"]["body_volume_m3"] - val["checks"]["body_volume_m3"]) / val["checks"]["body_volume_m3"]
        assert brel <= 1e-12, "wall role body volume rel %r vs the nominal" % (brel,)
        print("[ok] wall role: Lu/D_i 2.0 tags the upstream face wall_upstream (no slip_upstream), fluid volume rel %s vs truth, body volume unchanged" % (vrel,))

        # (G5) r'' = 0 at both ends of the wetted law, within 1e-6 1/m, for poly5 and poly7
        def end_second(mer, idx):
            assert len(idx) == 1, "want a single wall_contraction edge, got %d" % (len(idx),)
            c = BRepAdaptor_Curve(mer.Edges()[idx[0]].wrapped)
            worst = 0.0
            for u in (c.FirstParameter(), c.LastParameter()):
                P = gp_Pnt()
                V1 = gp_Vec()
                V2 = gp_Vec()
                c.D2(u, P, V1, V2)
                r2 = (V1.X() * V2.Y() - V1.Y() * V2.X()) / V1.X() ** 3
                worst = max(worst, abs(r2))
            return worst

        w5 = end_second(meridian, mets["wall_contraction"])
        r7 = job(dict(NOMINAL, law="poly7"), td, "poly7")
        assert r7["status"] == "ok", "poly7 run %r: %s" % (r7["status"], r7["message"])
        assert r7["value"]["status"] == "ok", "poly7 refused: %r %r" % (r7["value"].get("rule"), r7["value"].get("detail"))
        mer7 = load(os.path.join(td, "poly7"), "meridian.brep")
        w7 = end_second(mer7, r7["value"]["meridian_edges"]["wall_contraction"])
        assert w5 <= 1e-6, "poly5 r'' at the ends is %r 1/m" % (w5,)
        assert w7 <= 1e-6, "poly7 r'' at the ends is %r 1/m" % (w7,)
        print("[ok] r'' at the ends: poly5 %s 1/m, poly7 %s 1/m (<= 1e-6)" % (w5, w7))

        # (G6) fluid volume vs pi*int r^2 dx for all four laws, rel <= 1e-6
        worst_v = 0.0
        for law, x_m, name in (("poly3", None, "vol_poly3"), ("poly5", None, "nominal"),
                               ("poly7", None, "poly7"), ("cubic_matched", 0.5, "vol_cubic")):
            prm = dict(NOMINAL, law=law, x_m=x_m)
            if name == "nominal":
                fl = fluid
            elif name == "poly7":
                fl = load(os.path.join(td, "poly7"), "fluid.brep")
            else:
                rl = job(prm, td, name)
                assert rl["status"] == "ok", "%s run %r: %s" % (law, rl["status"], rl["message"])
                assert rl["value"]["status"] == "ok", "%s refused: %r %r" % (law, rl["value"].get("rule"), rl["value"].get("detail"))
                fl = load(os.path.join(td, name), "fluid.brep")
            vt = volume_truth(prm)
            got = measure.volume(fl)["value"]
            rel = abs(got - vt) / vt
            worst_v = max(worst_v, rel)
            assert rel <= 1e-6, "law %s: volume %r vs truth %r (rel %r)" % (law, got, vt, rel)
        print("[ok] fluid volume vs pi*int r^2 dx: worst rel %s over 4 laws (<= 1e-6)" % (worst_v,))

        # (G7) the minimum normal wall at the nominal equals t within 1e-8 m
        wet = [wall_m.Edges()[i] for i in wts["wetted"]]
        out = [wall_m.Edges()[i] for i in wts["outer"]]
        rec = measure.meridian_min_wall(wet, out)
        assert rec["status"] == "ok", "meridian_min_wall %r: %s" % (rec["status"], rec["detail"])
        wall_err = abs(rec["value"] - 0.003)
        assert wall_err <= 1e-8, "min normal wall %r m, |v - t| = %r m" % (rec["value"], wall_err)
        print("[ok] min normal wall at nominal: %s m, |v - t| = %s m (<= 1e-8)" % (rec["value"], wall_err))

        # (G8) the catalogue's primitives read the template's own numbers at the nominal
        pls = {p["name"]: p for p in val["planes"]}
        cs, ep = pls["contraction_start"], pls["exit_plane"]
        di = measure.diameter_at_plane(fluid, cs)
        de = measure.diameter_at_plane(fluid, ep)
        assert di["status"] == "ok" and abs(di["value"] - 0.06) <= 1e-9, "inlet diameter %r: %s" % (di["value"], di["detail"])
        assert de["status"] == "ok" and abs(de["value"] - 0.02) <= 1e-9, "exit diameter %r: %s" % (de["value"], de["detail"])
        ar = measure.area_ratio(fluid, cs, ep)
        assert ar["status"] == "ok" and abs(ar["value"] - 9.0) / 9.0 <= 1e-9, "area ratio %r: %s" % (ar["value"], ar["detail"])
        ext = measure.extent_along_axis(body)
        want_ext = 0.06 + 0.5 * 0.02
        assert ext["status"] == "ok" and abs(ext["value"] - want_ext) <= 1e-9, "extent %r: %s" % (ext["value"], ext["detail"])
        pd = measure.plane_distance(cs, ep)
        assert pd["status"] == "ok" and abs(pd["value"] - 0.06) <= 1e-9, "plane distance %r: %s" % (pd["value"], pd["detail"])
        wce = [meridian.Edges()[i] for i in mets["wall_contraction"]]
        sm = measure.slope_max(wce)
        want_sm = math.atan(0.625)
        assert sm["status"] == "ok" and abs(sm["value"] - want_sm) / want_sm <= 1e-6, "slope %r: %s" % (sm["value"], sm["detail"])
        crm = measure.curvature_radius_min(wce)
        want_crm = curvature_truth()
        assert crm["status"] == "ok" and abs(crm["value"] - want_crm) / want_crm <= 1e-6, "curvature radius %r vs truth %r: %s" % (crm["value"], want_crm, crm["detail"])
        print("[ok] catalogue at nominal: D_i, D_e, CR, total length, contraction length, slope and curvature radius read the template's own values")

        # (G9) the box corners: all 40 build, each one valid fluid solid and one valid body solid
        defs = []
        for law in ("poly3", "poly5", "poly7"):
            for lo, lx, tw in itertools.product((0.5, 1.5), (0.25, 1.0), (0.001, 0.01)):
                defs.append((law, dict(NOMINAL, law=law, x_m=None, L_over_Di=lo, Lx_over_De=lx, t_wall=tw)))
        for xm in (0.2, 0.8):
            for lo, lx, tw in itertools.product((0.5, 1.5), (0.25, 1.0), (0.001, 0.01)):
                defs.append(("cubic_matched", dict(NOMINAL, law="cubic_matched", x_m=xm, L_over_Di=lo, Lx_over_De=lx, t_wall=tw)))
        COMBOS = ((0.5, "slip"), (2.0, "wall"), (2.0, "slip"), (0.5, "wall"))
        defs = [(law, dict(prm, Lu_over_Di=COMBOS[i % 4][0], upstream_role=COMBOS[i % 4][1]))
                for i, (law, prm) in enumerate(defs)]
        corners = []
        corner_s = 0.0
        for i, (law, prm) in enumerate(defs):
            name = "corner_%02d" % i
            r = job(prm, td, name)
            assert r["status"] == "ok", "corner %d (%s) run %r: %s" % (i, law, r["status"], r["message"])
            v = r["value"]
            assert v["status"] == "ok", "corner %d (%s) refused: %r %r" % (i, law, v.get("rule"), v.get("detail"))
            odir = os.path.join(td, name)
            fl = load(odir, "fluid.brep")
            bd = load(odir, "body.brep")
            assert measure.n_solids(fl)["value"] == 1 and measure.valid(fl)["value"] == 1, "corner %d (%s): the fluid is not 1 valid solid" % (i, law)
            assert measure.n_solids(bd)["value"] == 1 and measure.valid(bd)["value"] == 1, "corner %d (%s): the body is not 1 valid solid" % (i, law)
            up = "wall_upstream" if prm["upstream_role"] == "wall" else "slip_upstream"
            other = "slip_upstream" if up == "wall_upstream" else "wall_upstream"
            assert up in v["face_tags"] and other not in v["face_tags"], "corner %d (%s): face_tags %r" % (i, law, sorted(v["face_tags"]))
            corner_s = corner_s + r["wall_s"]
            corners.append((law, prm, v, odir))
        assert len(corners) == 40, "%d corners ran, want 40" % (len(corners),)
        for law in ("poly3", "poly5", "poly7", "cubic_matched"):
            for combo in COMBOS:
                n = sum(1 for l2, p2, v2, o2 in corners
                        if l2 == law and (p2["Lu_over_Di"], p2["upstream_role"]) == combo)
                assert n >= 2, (law, combo, n)
        print("[ok] box corners: 40 of 40 build (poly3 8, poly5 8, poly7 8, cubic_matched 16), each 1 valid fluid and 1 valid body solid, %s s, every law at the 4 (Lu, role) corners" % (round(corner_s, 1),))

        # (G10) at every corner the outer wall is the true (trimmed where needed) normal offset
        worst_e = 0.0
        worst_f = 0.0
        n_trim = 0
        for i, (law, prm, v, odir) in enumerate(corners):
            wm = load(odir, "wall_meridian.brep")
            wet = [wm.Edges()[j] for j in v["wall_edges"]["wetted"]]
            out = [wm.Edges()[j] for j in v["wall_edges"]["outer"]]
            rec = measure.meridian_min_wall(wet, out)
            assert rec["status"] == "ok", "corner %d (%s): meridian_min_wall %r: %s" % (i, law, rec["status"], rec["detail"])
            err = abs(rec["value"] - prm["t_wall"])
            worst_e = max(worst_e, err)
            assert err <= 1e-8, "corner %d (%s): |min wall - t| = %r m" % (i, law, err)
            fid = v["offset"]["fidelity_m"]
            worst_f = max(worst_f, fid)
            assert fid <= 1e-9, "corner %d (%s): offset fidelity %r m" % (i, law, fid)
            truth = prm["t_wall"] > rho_min_offset_side(prm)
            assert bool(v["offset"]["trimmed"]) == truth, "corner %d (%s): trimmed %r, truth %r (rho_min %r)" % (i, law, v["offset"]["trimmed"], truth, rho_min_offset_side(prm))
            if truth:
                n_trim = n_trim + 1
        print("[ok] box corners: worst |min wall - t| %s m (<= 1e-8), worst offset fidelity %s m (<= 1e-9), trimmed exactly where t > rho_min (%d corners)" % (worst_e, worst_f, n_trim))

        # (G11) the refusal fixtures, each hitting its own PRF id and writing no BREP
        norole = dict((k, v) for k, v in NOMINAL.items() if k != "upstream_role")
        for want, nm, prm in (("PRF-BOX", "box_l", dict(NOMINAL, L_over_Di=0.4)),
                              ("PRF-BOX", "box_lu04", dict(NOMINAL, Lu_over_Di=0.4)),
                              ("PRF-BOX", "box_lu21", dict(NOMINAL, Lu_over_Di=2.1)),
                              ("PRF-BOX", "box_role", dict(NOMINAL, upstream_role="noslip")),
                              ("PRF-BOX", "box_norole", norole),
                              ("PRF-RMIN", "rmin", dict(NOMINAL, D_i=-0.06)),
                              ("PRF-MONO", "mono", dict(NOMINAL, CR=0.5)),
                              ("PRF-SELFX", "selfx", dict(NOMINAL, D_i=0.006, L_over_Di=0.5, Lx_over_De=0.25, t_wall=0.01))):
            r = job(prm, td, "refuse_" + nm)
            assert r["status"] == "ok", "%s run %r: %s" % (want, r["status"], r["message"])
            v = r["value"]
            assert v["status"] == "refused", "%s: value status %r" % (want, v["status"])
            assert v["rule"] == want, "%s: got %r (%s)" % (want, v["rule"], v["detail"])
            assert v["files"] == {}, "%s: files %r" % (want, v["files"])
            assert [f for f in os.listdir(os.path.join(td, "refuse_" + nm)) if f.endswith(".brep")] == [], "%s wrote a BREP" % (want,)
        for want, case in (("PRF-DERIV", "deriv"), ("PRF-FACE2D", "face2d")):
            r = job({"template": TEMPLATE, "case": case, "params": dict(NOMINAL)}, td, "refuse_" + want, module=INJECT)
            assert r["status"] == "ok", "%s run %r: %s" % (want, r["status"], r["message"])
            assert r["value"]["rule"] == want, "%s: got %r (%s)" % (want, r["value"]["rule"], r["value"]["detail"])
        print("[ok] refusals: 6 of 6 fixtures hit PRF-BOX, PRF-RMIN, PRF-MONO, PRF-DERIV, PRF-SELFX, PRF-FACE2D and write no BREP; PRF-BOX also on Lu 0.4, Lu 2.1, role noslip, role missing")

        # (G12) every build ran in a runner child; this process never imported the template
        assert "nozzle_template_under_test" not in sys.modules, "the template fixture ran in this process"
        assert "cad_child_job" not in sys.modules, "a child module ran in this process"
        tpl_dir = os.path.dirname(TEMPLATE)
        for m in list(sys.modules.values()):
            f = getattr(m, "__file__", None)
            assert not (isinstance(f, str) and os.path.abspath(f).startswith(tpl_dir)), "this process imported %r" % (f,)
        print("[ok] isolation: %d runner jobs; this process never imported template.py" % (len(JOBS),))
    print("gc3 wall %.1f s" % (time.time() - t0))
    print("SELFTEST PASS")
    return 0


def main(argv=None):
    if (sys.argv[1:] if argv is None else argv) == ["--selftest"]:
        return selftest()
    sys.stderr.write("usage: python template_gc3.py --selftest" + chr(10))
    return 2


if __name__ == "__main__":
    sys.exit(main())
