#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""gen_thin - docs/15 §E family F: 60 thin and sharp bodies.

Six-slot cycle: lattice-commensurate plates (plate_c, tier-0 stratum with
box_c), incommensurate plates (plate_n, nudged until the docs/15 §C lattice
rule says so), swept tapered fins with constant-length wedge edges, and
L-section angles in a commensurate and an incommensurate flavour.  The
plate thickness lies along z (a flat plate at zero incidence), the fin's
span along z, chord along x, thickness along y.  The wedge length e is the
same at every fin station, so every face is planar and the closed form is
exact.  Every quad is split by meshkit's diagonal rule, so the two big
faces of a plate are mirror-triangulated and features.py's inner-thickness
sample faces its opposite point; plate thickness is gated at 2 %, fin
thickness is only reported.  Wound outward by construction and refused,
never repaired, when stl_io.check_closed does not pass.

    python tools/autonomy/corpus/gen_thin.py --seed 1 --n 60 --out DIR
    python tools/autonomy/corpus/gen_thin.py --selftest
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import meshkit  # noqa: E402
import schema  # noqa: E402
import stl_io  # noqa: E402

FAMILY = "F"
GENERATOR = "corpus/gen_thin.py"
SOLID = "body"
SALT = 15
NU = 1.5e-5
RE_RANGE = (1.0e4, 1.0e5)
DIV = 16
VOLUME_FORM = ("plate_c n_x*n_y*n_t*s^3; plate_n length*width*thickness; fin "
               "t*span*((c_root + c_tip)/2 - e); lcorner W*t*(A + B - t)")
SHAPE_CYCLE = ("plate_c", "plate_n", "fin", "lcorner_c", "lcorner_n", "fin")

_KEYS = {
    "plate_c": ("s", "n_x", "n_y", "n_t"),
    "plate_n": ("length", "width", "thickness"),
    "fin": ("c_root", "taper", "span", "t_frac", "sweep_deg", "wedge_frac",
            "n_wedge"),
    "lcorner_c": ("s", "n_a", "n_b", "n_t", "n_w"),
    "lcorner_n": ("leg_a", "leg_b", "thickness", "width"),
}


def sample_params(seed: int, index: int) -> dict:
    """One meshkit.uniforms call, every draw in the fixed table order."""
    U = meshkit.uniforms(SALT, seed, index)
    shape = SHAPE_CYCLE[index % len(SHAPE_CYCLE)]
    p = {"shape": shape}
    if shape == "plate_c":
        p["s"] = meshkit.fl(meshkit.lin(U, 0, 0.012, 0.02), 4)
        p["n_x"] = meshkit.ik(U, 1, 48, 64)
        p["n_y"] = max(2, int(round(meshkit.lin(U, 2, 0.4, 1.0) * p["n_x"])))
        p["n_t"] = meshkit.ik(U, 3, 1, 2)
    elif shape == "plate_n":
        p["length"] = meshkit.fl(meshkit.lin(U, 0, 0.6, 1.2), 4)
        p["width"] = meshkit.fl(meshkit.lin(U, 2, 0.4, 1.0) * p["length"], 4)
        p["thickness"] = meshkit.fl(
            meshkit.lin(U, 3, 0.004, 0.03) * p["length"], 5)
        p = _nudge(p, "thickness", 0.00001)
    elif shape == "fin":
        p["c_root"] = meshkit.fl(meshkit.lin(U, 0, 0.5, 1.0), 4)
        p["taper"] = meshkit.fl(meshkit.lin(U, 1, 0.4, 1.0), 4)
        p["span"] = meshkit.fl(meshkit.lin(U, 2, 0.5, 1.5) * p["c_root"], 4)
        p["t_frac"] = meshkit.fl(meshkit.lin(U, 3, 0.01, 0.04), 4)
        p["sweep_deg"] = meshkit.fl(meshkit.lin(U, 4, 0.0, 40.0), 3)
        p["wedge_frac"] = meshkit.fl(meshkit.lin(U, 5, 0.1, 0.3), 4)
        p["n_wedge"] = 2
    elif shape == "lcorner_c":
        p["s"] = meshkit.fl(meshkit.lin(U, 0, 0.012, 0.02), 4)
        p["n_a"] = meshkit.ik(U, 1, 40, 64)
        p["n_b"] = max(8, int(round(meshkit.lin(U, 2, 0.3, 1.0) * p["n_a"])))
        p["n_t"] = meshkit.ik(U, 3, 1, 3)
        p["n_w"] = max(4, int(round(meshkit.lin(U, 4, 0.3, 1.0) * p["n_a"])))
    else:
        p["leg_a"] = meshkit.fl(meshkit.lin(U, 0, 0.6, 1.2), 4)
        p["leg_b"] = meshkit.fl(meshkit.lin(U, 2, 0.3, 1.0) * p["leg_a"], 4)
        p["thickness"] = meshkit.fl(
            meshkit.lin(U, 3, 0.004, 0.04) * p["leg_a"], 5)
        p["width"] = meshkit.fl(meshkit.lin(U, 4, 0.3, 1.0) * p["leg_a"], 4)
        p = _nudge(p, "thickness", 0.00001)
    p["re_l"] = float("%.4g" % 10 ** meshkit.lin(U, 11, 4.0, 5.0))
    p["div"] = DIV
    return p


def _nudge(p: dict, key: str, eps: float) -> dict:
    """Nudge <key> up until the docs/15 §C lattice rule says incommensurate."""
    for _ in range(1000):
        if meshkit.lattice_spacing(planes(p)) is None:
            return p
        p[key] = meshkit.fl(p[key] + eps, 5)
    raise RuntimeError("%s: no incommensurate %s in 1000 nudges"
                       % (p["shape"], key))


def _fin(v, name, lo, hi):
    v = float(v)
    if not math.isfinite(v):
        raise ValueError("%s: %r is not finite" % (name, v))
    if v < lo or v > hi:
        raise ValueError("%s: %g outside %g..%g" % (name, v, lo, hi))
    return v


def _int(params, key, lo, hi):
    v = params[key]
    if not isinstance(v, int) or isinstance(v, bool):
        raise ValueError("%s: %r is not an int" % (key, v))
    if v < lo or v > hi:
        raise ValueError("%s: %d outside %d..%d" % (key, v, lo, hi))
    return v


def validate_params(params: dict) -> None:
    """Refuse every broken dict by name: ValueError("<key>: <why>")."""
    shape = params.get("shape")
    if shape not in _KEYS:
        raise ValueError("shape: %r is not one of %s"
                         % (shape, ", ".join(SHAPE_CYCLE)))
    want = set(_KEYS[shape]) | {"shape", "re_l", "div"}
    for k in sorted(set(params) - want):
        raise ValueError("%s: unknown key for %s" % (k, shape))
    for k in sorted(want - set(params)):
        raise ValueError("%s: missing key for %s" % (k, shape))
    _fin(params["re_l"], "re_l", RE_RANGE[0], RE_RANGE[1])
    _int(params, "div", 4, 64)
    if shape == "plate_c":
        _fin(params["s"], "s", 0.002, 0.1)
        _int(params, "n_x", 4, 64)
        _int(params, "n_y", 2, 64)
        n_t = _int(params, "n_t", 1, 8)
        if n_t >= params["n_y"]:
            raise ValueError("n_t: %d is not < n_y %d" % (n_t, params["n_y"]))
    elif shape == "plate_n":
        length = _fin(params["length"], "length", 0.2, 5.0)
        _fin(params["width"], "width", 0.05 * length, 2.0 * length)
        _fin(params["thickness"], "thickness", 0.002 * length, 0.1 * length)
        if meshkit.lattice_spacing(planes(params)) is not None:
            raise ValueError("thickness: plate_n dims are lattice-commensurate")
    elif shape == "fin":
        c_root = _fin(params["c_root"], "c_root", 0.1, 5.0)
        _fin(params["taper"], "taper", 0.2, 1.0)
        _fin(params["span"], "span", 0.2 * c_root, 5.0 * c_root)
        _fin(params["t_frac"], "t_frac", 0.002, 0.1)
        _fin(params["sweep_deg"], "sweep_deg", 0.0, 60.0)
        _fin(params["wedge_frac"], "wedge_frac", 0.05, 0.4)
        _int(params, "n_wedge", 1, 16)
    else:
        s = params.get("s")
        if shape == "lcorner_c":
            _fin(params["s"], "s", 0.002, 0.1)
            _int(params, "n_a", 1, 64)
            _int(params, "n_b", 1, 64)
            n_t = _int(params, "n_t", 1, 8)
            _int(params, "n_w", 1, 64)
            if 2 * n_t >= params["n_a"]:
                raise ValueError("n_t: 2*%d is not < n_a %d"
                                 % (n_t, params["n_a"]))
            if 2 * n_t >= params["n_b"]:
                raise ValueError("n_t: 2*%d is not < n_b %d"
                                 % (n_t, params["n_b"]))
        else:
            leg_a = _fin(params["leg_a"], "leg_a", 0.2, 5.0)
            _fin(params["leg_b"], "leg_b", 0.05 * leg_a, 2.0 * leg_a)
            _fin(params["width"], "width", 0.05 * leg_a, 2.0 * leg_a)
            t = _fin(params["thickness"], "thickness", 0.002 * leg_a,
                     float("inf"))
            if not 2.0 * t < min(leg_a, params["leg_b"]):
                raise ValueError("thickness: 2*%g is not < min(leg_a, leg_b)"
                                 % t)
            if meshkit.lattice_spacing(planes(params)) is not None:
                raise ValueError("thickness: lcorner_n dims are "
                                 "lattice-commensurate")


def l_ref(params: dict) -> float:
    shape = params["shape"]
    if shape == "plate_c":
        return params["n_x"] * params["s"]
    if shape == "plate_n":
        return params["length"]
    if shape == "fin":
        return params["c_root"] * (1.0 + params["taper"]) / 2.0
    if shape == "lcorner_c":
        return params["n_a"] * params["s"]
    return params["leg_a"]


def _lcorner_dims(params: dict) -> tuple:
    shape = params["shape"]
    if shape == "lcorner_c":
        s = params["s"]
        return (params["n_a"] * s, params["n_b"] * s, params["n_t"] * s,
                params["n_w"] * s)
    return (params["leg_a"], params["leg_b"], params["thickness"],
            params["width"])


def planes(params: dict):
    """Axis-aligned plane coordinates for the planar shapes, else None."""
    shape = params["shape"]
    if shape == "plate_c":
        s = params["s"]
        return [[0.0, params["n_x"] * s],
                [-(params["n_y"] * s) / 2.0, (params["n_y"] * s) / 2.0],
                [0.0, params["n_t"] * s]]
    if shape == "plate_n":
        return [[0.0, params["length"]],
                [-params["width"] / 2.0, params["width"] / 2.0],
                [0.0, params["thickness"]]]
    if shape in ("lcorner_c", "lcorner_n"):
        A, B, t, W = _lcorner_dims(params)
        return [[0.0, t, A], [-W / 2.0, W / 2.0], [0.0, t, B]]
    return None


def closed_form_volume(params: dict) -> float:
    shape = params["shape"]
    if shape == "plate_c":
        return params["n_x"] * params["n_y"] * params["n_t"] * params["s"] ** 3
    if shape == "plate_n":
        return params["length"] * params["width"] * params["thickness"]
    if shape == "fin":
        c_tip = params["taper"] * params["c_root"]
        e = params["wedge_frac"] * c_tip
        t = params["t_frac"] * params["c_root"]
        return t * params["span"] * ((params["c_root"] + c_tip) / 2.0 - e)
    A, B, t, W = _lcorner_dims(params)
    return W * t * (A + B - t)


def stratum(params: dict) -> str:
    shape = params["shape"]
    if shape in ("plate_c", "lcorner_c"):
        return "easy"
    if shape == "fin":
        return "hard"
    t = params["thickness"]
    return "hard" if t < l_ref(params) / 64.0 else "medium"


def n_bodies(params: dict) -> int:
    return 1


def expected_features(params: dict) -> dict:
    shape = params["shape"]
    if shape == "plate_c":
        return {"commensurate": True,
                "lattice_base_size_m": meshkit.lattice_spacing(planes(params)),
                "planar_one": True,
                "inner_thickness_m": params["n_t"] * params["s"]}
    if shape == "plate_n":
        return {"commensurate": False, "lattice_base_size_m": None,
                "planar_one": True,
                "inner_thickness_m": params["thickness"]}
    if shape == "fin":
        return {"commensurate": False, "lattice_base_size_m": None,
                "planar_one": False,
                "inner_thickness_reported_m":
                    params["t_frac"] * params["c_root"]}
    if shape == "lcorner_c":
        return {"commensurate": True,
                "lattice_base_size_m": meshkit.lattice_spacing(planes(params)),
                "planar_one": True}
    return {"commensurate": False, "lattice_base_size_m": None,
            "planar_one": True}


def flow(params: dict) -> dict:
    return {"u_ref_m_s": params["re_l"] * NU / l_ref(params),
            "l_ref_m": l_ref(params), "nu_m2_s": NU,
            "note": "a priori target Re_L = %.4g (docs/15 §D.3 window)"
                    % params["re_l"]}


def geometry_id(seed: int, index: int) -> str:
    return "%s-%d-%03d" % (FAMILY, seed, index)


def _ring_fin(params: dict, k: int, n_st: int, n_c: int) -> list:
    """The swept tapered hexagon of station k, counter-clockwise in (x, y)."""
    c_tip = params["taper"] * params["c_root"]
    e = params["wedge_frac"] * c_tip
    t = params["t_frac"] * params["c_root"]
    x0 = math.tan(math.radians(params["sweep_deg"])) * params["span"] * k / n_st
    c = params["c_root"] + (c_tip - params["c_root"]) * k / n_st
    nw = params["n_wedge"]
    return (meshkit.line((x0, 0.0), (x0 + e, -t / 2.0), nw)
            + meshkit.line((x0 + e, -t / 2.0), (x0 + c - e, -t / 2.0), n_c)
            + meshkit.line((x0 + c - e, -t / 2.0), (x0 + c, 0.0), nw)
            + meshkit.line((x0 + c, 0.0), (x0 + c - e, t / 2.0), nw)
            + meshkit.line((x0 + c - e, t / 2.0), (x0 + e, t / 2.0), n_c)
            + meshkit.line((x0 + e, t / 2.0), (x0, 0.0), nw))


def build(params: dict):
    """Validate, then wind the body outward; check_closed inside refuses."""
    validate_params(params)
    shape = params["shape"]
    step = l_ref(params) / DIV
    if shape in ("plate_c", "plate_n"):
        if shape == "plate_c":
            s = params["s"]
            xs = meshkit.axis_coords(0.0, params["n_x"] * s,
                                     meshkit.divs(params["n_x"] * s, step))
            ys = meshkit.axis_coords(-(params["n_y"] * s) / 2.0,
                                     (params["n_y"] * s) / 2.0,
                                     meshkit.divs(params["n_y"] * s, step))
            zs = meshkit.axis_coords(0.0, params["n_t"] * s,
                                     meshkit.divs(params["n_t"] * s, step))
        else:
            xs = meshkit.axis_coords(0.0, params["length"],
                                     meshkit.divs(params["length"], step))
            ys = meshkit.axis_coords(-params["width"] / 2.0,
                                     params["width"] / 2.0,
                                     meshkit.divs(params["width"], step))
            zs = meshkit.axis_coords(0.0, params["thickness"],
                                     meshkit.divs(params["thickness"], step))
        return meshkit.grid_box(xs, ys, zs)
    if shape == "fin":
        c_tip = params["taper"] * params["c_root"]
        e = params["wedge_frac"] * c_tip
        span = params["span"]
        n_st = max(2, meshkit.divs(span, step))
        n_c = max(2, meshkit.divs(params["c_root"] - 2.0 * e, step))
        rings = [_ring_fin(params, k, n_st, n_c) for k in range(n_st + 1)]
        ws = [span * k / n_st for k in range(n_st + 1)]
        centres = []
        for k in (0, n_st):
            x0 = math.tan(math.radians(params["sweep_deg"])) * span * k / n_st
            c = params["c_root"] + (c_tip - params["c_root"]) * k / n_st
            centres.append((x0 + c / 2.0, 0.0))
        return meshkit.loft(rings, ws, (0, 1, 2), (centres[0], centres[1]))
    A, B, t, W = _lcorner_dims(params)

    def ln(p, q, ln_len):
        return meshkit.line(p, q, meshkit.divs(ln_len, step))

    ring = (ln((0.0, 0.0), (A, 0.0), A) + ln((A, 0.0), (A, t), t)
            + ln((A, t), (t, t), A - t) + ln((t, t), (t, B), B - t)
            + ln((t, B), (0.0, B), t) + ln((0.0, B), (0.0, 0.0), B))
    return meshkit.prism(ring,
                         meshkit.axis_coords(-W / 2.0, W / 2.0,
                                             meshkit.divs(W, step)),
                         (0, 2, 1), (t / 2.0, t / 2.0))


def make_row(seed: int, index: int):
    return meshkit.row_make(sys.modules[__name__], seed, index)


def write_row(row: dict, out_dir: str) -> str:
    return meshkit.row_write(sys.modules[__name__], row, out_dir)


def generate(seed: int, n: int, out_dir: str) -> list:
    return meshkit.row_generate(sys.modules[__name__], seed, n, out_dir)


def main(argv=None) -> int:
    return meshkit.row_cli(sys.modules[__name__], argv, _selftest)


# --- the selftest ---------------------------------------------------------


def _facing_sets(P: np.ndarray, axis: int, value: float, tol: float) -> set:
    sel = P[np.abs(P[:, axis] - value) <= tol]
    others = [a for a in range(3) if a != axis]
    return set(map(tuple, np.round(sel[:, others] / tol).astype(np.int64)))


def _selftest() -> int:
    def group_closed_forms():
        pc = {"shape": "plate_c", "s": 0.015, "n_x": 50, "n_y": 30, "n_t": 2,
              "re_l": 2.0e4, "div": 16}
        assert abs(closed_form_volume(pc) - 50 * 30 * 2 * 0.015 ** 3) <= 1e-18
        fin = {"shape": "fin", "c_root": 1.0, "taper": 0.5, "span": 0.8,
               "t_frac": 0.03, "sweep_deg": 25.0, "wedge_frac": 0.2,
               "n_wedge": 2, "re_l": 2.0e4, "div": 16}
        want = 0.03 * 0.8 * ((1.0 + 0.5) / 2.0 - 0.2 * 0.5)
        P, T = build(fin)
        got = stl_io.signed_volume(P, T)
        assert abs(got / want - 1.0) <= 1e-12, "fin %r != %r" % (got, want)
        lc = {"shape": "lcorner_n", "leg_a": 1.0, "leg_b": 0.6,
              "thickness": 0.1, "width": 0.5, "re_l": 2.0e4, "div": 16}
        assert abs(closed_form_volume(lc) - 0.5 * 0.1 * 1.5) <= 1e-12
        return "plate_c, fin (built, 1e-12) and lcorner closed forms exact"

    def group_closed():
        tris = []
        for i in range(12):
            p = sample_params(7, i)
            P, T = build(p)
            tris.append(len(T))
        return "12 bodies of seed 7 build closed, %d..%d triangles" \
            % (min(tris), max(tris))

    def group_volume():
        worst = {}
        for i in range(12):
            p = sample_params(7, i)
            P, T = build(p)
            dev = abs(stl_io.signed_volume(P, T) / closed_form_volume(p) - 1.0)
            assert dev <= 1e-12, "%s dev %g" % (p["shape"], dev)
            worst[p["shape"]] = max(worst.get(p["shape"], 0.0), dev)
        return "12 bodies within 1e-12 of the closed form: %s" % ", ".join(
            "%s %.2e" % kv for kv in sorted(worst.items()))

    def group_faces():
        for i in range(12):
            p = sample_params(7, i)
            P, T = build(p)
            shape = p["shape"]
            if shape in ("plate_c", "plate_n"):
                zs = P[:, 2]
                tol = 1e-9 * (zs.max() - zs.min())
                top = _facing_sets(P, 2, float(zs.max()), tol)
                bot = _facing_sets(P, 2, float(zs.min()), tol)
                assert top == bot and top, "%s: big faces differ" % shape
                ct = _plate_centroids(P, T, float(zs.max()))
                cb = _plate_centroids(P, T, float(zs.min()))
                assert len(ct) == len(cb) and np.allclose(
                    np.asarray(ct), np.asarray(cb), rtol=0.0, atol=1e-12), \
                    "%s: big faces' centroids differ" % shape
            elif shape == "fin":
                ys = P[:, 1]
                tol = 1e-9 * (P[:, 0].max() - P[:, 0].min())
                hi = _facing_sets(P, 1, float(ys.max()), tol)
                lo = _facing_sets(P, 1, float(ys.min()), tol)
                assert hi == lo and hi, "fin: side faces differ"
        return "plates: top and bottom share vertex (x, y) and centroid sets; fins: y = +/-t/2 share (x, z)"

    def _plate_centroids(P, T, z):
        others = [0, 1]
        cs = []
        for t in T:
            if all(abs(P[t[k], 2] - z) <= 1e-12 for k in range(3)):
                cs.append((float(P[t][:, 0].mean()), float(P[t][:, 1].mean())))
        return sorted(cs)

    def group_lattice():
        for i in range(60):
            p = sample_params(7, i)
            shape = p["shape"]
            if shape == "plate_c":
                want = math.gcd(p["n_x"], p["n_y"], p["n_t"]) * p["s"]
                got = meshkit.lattice_spacing(planes(p))
                assert got is not None and abs(got / want - 1.0) <= 1e-12, \
                    "plate_c %d: %r != %r" % (i, got, want)
            elif shape == "lcorner_c":
                want = math.gcd(p["n_a"], p["n_b"], p["n_t"],
                                p["n_w"]) * p["s"]
                got = meshkit.lattice_spacing(planes(p))
                assert got is not None and abs(got / want - 1.0) <= 1e-12, \
                    "lcorner_c %d: %r != %r" % (i, got, want)
            elif shape in ("plate_n", "lcorner_n"):
                assert meshkit.lattice_spacing(planes(p)) is None, \
                    "%s %d is commensurate" % (shape, i)
            else:
                assert planes(p) is None
        return "60 draws: *_c == gcd*s, *_n None, fins no planes"

    def group_sampler():
        counts = {}
        strata = {"easy": 0, "medium": 0, "hard": 0}
        for i in range(60):
            p = sample_params(7, i)
            counts[p["shape"]] = counts.get(p["shape"], 0) + 1
            validate_params(p)
            for v in p.values():
                if isinstance(v, float) and v == 0.0:
                    assert math.copysign(1.0, v) > 0.0, "-0.0 in %r" % p
            strata[stratum(p)] += 1
        assert counts == {"plate_c": 10, "plate_n": 10, "fin": 20,
                          "lcorner_c": 10, "lcorner_n": 10}, counts
        return ("10 plate_c, 10 plate_n, 20 fin, 10 lcorner_c, 10 lcorner_n; "
                "strata %s" % strata)

    def group_refusals():
        def refuses(params, frag):
            try:
                validate_params(params)
            except ValueError as e:
                assert frag in str(e), "wrong refusal: %s" % e
                return
            raise AssertionError("no ValueError naming %r" % frag)
        pc = {"shape": "plate_c", "s": 0.015, "n_x": 50, "n_y": 30, "n_t": 2,
              "re_l": 2.0e4, "div": 16}
        pn = {"shape": "plate_n", "length": 1.0, "width": 0.6,
              "thickness": 0.01337, "re_l": 2.0e4, "div": 16}
        refuses(dict(pc, s=0.001), "s: 0.001 outside")
        refuses(dict(pc, n_x=3), "n_x: 3 outside")
        refuses(dict(pc, n_t=2, n_y=2), "n_t: 2 is not < n_y 2")
        refuses(dict(pn, thickness=0.25, length=1.0, width=0.6),
                "thickness: 0.25 outside")
        refuses(dict(pn, thickness=0.0001), "thickness: 0.0001 outside")
        refuses({"shape": "fin", "c_root": 1.0, "taper": 1.5, "span": 0.8,
                 "t_frac": 0.03, "sweep_deg": 10.0, "wedge_frac": 0.2,
                 "n_wedge": 2, "re_l": 2.0e4, "div": 16},
                "taper: 1.5 outside")
        refuses({"shape": "fin", "c_root": 1.0, "taper": 0.5, "span": 0.8,
                 "t_frac": 0.03, "sweep_deg": 70.0, "wedge_frac": 0.2,
                 "n_wedge": 2, "re_l": 2.0e4, "div": 16},
                "sweep_deg: 70 outside")
        refuses({"shape": "lcorner_c", "s": 0.015, "n_a": 16, "n_b": 20,
                 "n_t": 8, "n_w": 8, "re_l": 2.0e4, "div": 16},
                "n_t: 2*8 is not < n_a 16")
        refuses({"shape": "lcorner_n", "leg_a": 1.0, "leg_b": 0.6,
                 "thickness": 0.51, "width": 0.5, "re_l": 2.0e4, "div": 16},
                "thickness: 2*0.51 is not < min(leg_a, leg_b)")
        refuses(dict(pn, thickness=0.02), "lattice-commensurate")
        bad = dict(pc)
        bad["extra"] = 1
        refuses(bad, "extra: unknown key for plate_c")
        short = dict(pn)
        del short["thickness"]
        refuses(short, "thickness: missing key for plate_n")
        refuses(dict(pc, shape="foil"), "shape: 'foil' is not one of")
        return "13 by name"

    groups = [("[ok] closed forms", group_closed_forms),
              ("[ok] closed", group_closed),
              ("[ok] volume", group_volume),
              ("[ok] faces aligned", group_faces),
              ("[ok] lattice", group_lattice),
              ("[ok] sampler", group_sampler),
              ("[ok] determinism", lambda: meshkit.check_determinism(
                  sys.modules[__name__], 7, (0, 1, 2, 3))),
              ("[ok] rows", lambda: meshkit.check_rows(
                  sys.modules[__name__], 7, (0, 1, 2, 3))),
              ("[ok] refusals", group_refusals)]
    for name, fn in groups:
        try:
            note = fn()
        except AssertionError as e:
            print("SELFTEST FAIL: %s: %s" % (name, e))
            return 1
        print("%s: %s" % (name, note))
    print("SELFTEST PASS")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    sys.exit(main())
