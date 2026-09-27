#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""gen_gap - docs/15 §E family E: 60 two-body pairs at a small gap.

Three shapes, 20 each: box pairs, sphere pairs, cross-flow cylinder pairs.
The gap is gap_over_h * h_ref with h_ref = l_ref / 32 - docs/15 §E's
"0.5-5 h" with h taken as the wall cell of the §B / AM-5 L4 template
(base_size 0.5*l_ref at max_level 4).  Body a sits on the -y side, body b
on the +y side, both with the same x and z discretisation, so the
closest-approach vertices sit exactly at y = -gap/2 and y = +gap/2 with
equal (x, z) and their normals along the line of centres - which is what
makes features.py's outer_gap exact.  One written STL holds both bodies:
two closed components, V - E + F = 4.  Wound outward by construction and
refused, never repaired, when stl_io.check_closed does not pass.

    python tools/autonomy/corpus/gen_gap.py --seed 1 --n 60 --out DIR
    python tools/autonomy/corpus/gen_gap.py --selftest
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

FAMILY = "E"
GENERATOR = "corpus/gen_gap.py"
SOLID = "body"
SALT = 14
NU = 1.5e-5
RE_RANGE = (1.0e4, 1.0e5)
DIV = 16
H_DIV = 32
VOLUME_FORM = ("boxes length*height*(width_a + width_b); spheres "
               "4/3*pi*(radius_a^3 + radius_b^3); cylinders "
               "pi*(radius_a^2 + radius_b^2)*height")
SHAPE_CYCLE = ("boxes", "spheres", "cylinders")

_KEYS = {
    "boxes": ("length", "height", "width_a", "width_b"),
    "spheres": ("radius_a", "radius_b", "n_theta", "n_phi"),
    "cylinders": ("radius_a", "radius_b", "height", "n_theta"),
}


def sample_params(seed: int, index: int) -> dict:
    """One meshkit.uniforms call, every draw in the fixed table order."""
    U = meshkit.uniforms(SALT, seed, index)
    shape = SHAPE_CYCLE[index % len(SHAPE_CYCLE)]
    p = {"shape": shape}
    if shape == "boxes":
        p["length"] = meshkit.fl(meshkit.lin(U, 0, 0.6, 1.2), 4)
        p["height"] = meshkit.fl(meshkit.lin(U, 1, 0.2, 0.6) * p["length"], 4)
        p["width_a"] = meshkit.fl(meshkit.lin(U, 2, 0.2, 0.6) * p["length"], 4)
        p["width_b"] = meshkit.fl(meshkit.lin(U, 3, 0.2, 0.6) * p["length"], 4)
    elif shape == "spheres":
        p["radius_a"] = meshkit.fl(meshkit.lin(U, 0, 0.15, 0.4), 4)
        p["radius_b"] = meshkit.fl(
            meshkit.lin(U, 1, 0.4, 1.0) * p["radius_a"], 4)
        p["n_theta"] = 64
        p["n_phi"] = 32
    else:
        p["radius_a"] = meshkit.fl(meshkit.lin(U, 0, 0.15, 0.4), 4)
        p["radius_b"] = meshkit.fl(
            meshkit.lin(U, 1, 0.4, 1.0) * p["radius_a"], 4)
        p["height"] = meshkit.fl(meshkit.lin(U, 2, 1.0, 3.0) * p["radius_a"], 4)
        p["n_theta"] = 64
    p["gap_over_h"] = meshkit.fl(meshkit.lin(U, 4, 0.5, 5.0), 3)
    p["re_l"] = float("%.4g" % 10 ** meshkit.lin(U, 11, 4.0, 5.0))
    p["div"] = DIV
    return p


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


def _n_theta(params: dict) -> None:
    n = _int(params, "n_theta", 8, 512)
    if n % 4 != 0:
        raise ValueError("n_theta: %d is not a multiple of 4" % n)


def validate_params(params: dict) -> None:
    """Refuse every broken dict by name: ValueError("<key>: <why>")."""
    shape = params.get("shape")
    if shape not in _KEYS:
        raise ValueError("shape: %r is not one of %s"
                         % (shape, ", ".join(SHAPE_CYCLE)))
    want = set(_KEYS[shape]) | {"shape", "gap_over_h", "re_l", "div"}
    for k in sorted(set(params) - want):
        raise ValueError("%s: unknown key for %s" % (k, shape))
    for k in sorted(want - set(params)):
        raise ValueError("%s: missing key for %s" % (k, shape))
    _fin(params["re_l"], "re_l", RE_RANGE[0], RE_RANGE[1])
    _int(params, "div", 4, 64)
    _fin(params["gap_over_h"], "gap_over_h", 0.5, 5.0)
    if shape == "boxes":
        length = _fin(params["length"], "length", 0.2, 5.0)
        for k in ("height", "width_a", "width_b"):
            _fin(params[k], k, 0.05 * length, 2.0 * length)
    else:
        ra = _fin(params["radius_a"], "radius_a", 0.02, 2.0)
        _fin(params["radius_b"], "radius_b", 0.3 * ra, ra)
        if shape == "spheres":
            _n_theta(params)
            _int(params, "n_phi", 4, 256)
        else:
            _fin(params["height"], "height", 0.5 * ra, 10.0 * ra)
            _n_theta(params)


def l_ref(params: dict) -> float:
    if params["shape"] == "boxes":
        return params["length"]
    return 2.0 * params["radius_a"]


def gap(params: dict) -> float:
    """gap_over_h * h_ref, h_ref = l_ref / 32 (the L4-template wall cell)."""
    return params["gap_over_h"] * l_ref(params) / H_DIV


def planes(params: dict):
    if params["shape"] != "boxes":
        return None
    g = gap(params) / 2.0
    return [[0.0, params["length"]],
            [-g - params["width_a"], -g, g, g + params["width_b"]],
            [0.0, params["height"]]]


def build_bodies(params: dict) -> list:
    """[body a, body b]: a on the -y side, b on the +y side of the gap."""
    validate_params(params)
    shape = params["shape"]
    step = l_ref(params) / DIV
    g = gap(params) / 2.0
    if shape == "boxes":
        X = meshkit.axis_coords(0.0, params["length"],
                                meshkit.divs(params["length"], step))
        Z = meshkit.axis_coords(0.0, params["height"],
                                meshkit.divs(params["height"], step))
        a = meshkit.grid_box(
            X, meshkit.axis_coords(-g - params["width_a"], -g,
                                   meshkit.divs(params["width_a"], step)), Z)
        b = meshkit.grid_box(
            X, meshkit.axis_coords(g, g + params["width_b"],
                                   meshkit.divs(params["width_b"], step)), Z)
    elif shape == "spheres":
        a = meshkit.uv_sphere(params["radius_a"], (0.0, -g - params["radius_a"],
                                                   0.0), 1,
                              params["n_theta"], params["n_phi"])
        b = meshkit.uv_sphere(params["radius_b"], (0.0, g + params["radius_b"],
                                                   0.0), 1,
                              params["n_theta"], params["n_phi"])
    else:
        Zs = meshkit.axis_coords(0.0, params["height"],
                                 meshkit.divs(params["height"], step))
        a = meshkit.prism(
            meshkit.circle(0.0, -g - params["radius_a"], params["radius_a"],
                           params["n_theta"]), Zs, (0, 1, 2),
            (0.0, -g - params["radius_a"]))
        b = meshkit.prism(
            meshkit.circle(0.0, g + params["radius_b"], params["radius_b"],
                           params["n_theta"]), Zs, (0, 1, 2),
            (0.0, g + params["radius_b"]))
    return [a, b]


def build(params: dict):
    """The pair as ONE surface: two closed components, V - E + F = 4."""
    return meshkit.combine(build_bodies(params))


def n_bodies(params: dict) -> int:
    return 2


def closed_form_volume(params: dict) -> float:
    if params["shape"] == "boxes":
        return params["length"] * params["height"] * (params["width_a"]
                                                      + params["width_b"])
    if params["shape"] == "spheres":
        return 4.0 / 3.0 * math.pi * (params["radius_a"] ** 3
                                      + params["radius_b"] ** 3)
    return math.pi * (params["radius_a"] ** 2 + params["radius_b"] ** 2) \
        * params["height"]


def stratum(params: dict) -> str:
    if params["gap_over_h"] >= 3.0:
        return "easy"
    if params["gap_over_h"] >= 1.5:
        return "medium"
    return "hard"


def expected_features(params: dict) -> dict:
    if params["shape"] == "boxes":
        sp = meshkit.lattice_spacing(planes(params))
        return {"commensurate": sp is not None, "lattice_base_size_m": sp,
                "planar_one": True, "outer_gap_m": gap(params)}
    return {"commensurate": False, "lattice_base_size_m": None,
            "planar_one": False, "outer_gap_m": gap(params)}


def flow(params: dict) -> dict:
    return {"u_ref_m_s": params["re_l"] * NU / l_ref(params),
            "l_ref_m": l_ref(params), "nu_m2_s": NU,
            "note": "a priori target Re_L = %.4g (docs/15 §D.3 window)"
                    % params["re_l"]}


def geometry_id(seed: int, index: int) -> str:
    return "%s-%d-%03d" % (FAMILY, seed, index)


def make_row(seed: int, index: int):
    return meshkit.row_make(sys.modules[__name__], seed, index)


def write_row(row: dict, out_dir: str) -> str:
    return meshkit.row_write(sys.modules[__name__], row, out_dir)


def generate(seed: int, n: int, out_dir: str) -> list:
    return meshkit.row_generate(sys.modules[__name__], seed, n, out_dir)


def main(argv=None) -> int:
    return meshkit.row_cli(sys.modules[__name__], argv, _selftest)


# --- the selftest ---------------------------------------------------------


def _selftest() -> int:
    def group_closed_forms():
        boxes = {"shape": "boxes", "length": 1.0, "height": 0.4,
                 "width_a": 0.3, "width_b": 0.5, "gap_over_h": 2.0,
                 "re_l": 2.0e4, "div": 16}
        assert abs(closed_form_volume(boxes) - 1.0 * 0.4 * 0.8) <= 1e-12
        spheres = {"shape": "spheres", "radius_a": 0.3, "radius_b": 0.2,
                   "n_theta": 64, "n_phi": 32, "gap_over_h": 2.0,
                   "re_l": 2.0e4, "div": 16}
        want = 4.0 / 3.0 * math.pi * (0.027 + 0.008)
        assert abs(closed_form_volume(spheres) - want) <= 1e-12
        cyl = {"shape": "cylinders", "radius_a": 0.3, "radius_b": 0.2,
               "height": 0.7, "n_theta": 64, "gap_over_h": 2.0,
               "re_l": 2.0e4, "div": 16}
        want = math.pi * (0.09 + 0.04) * 0.7
        assert abs(closed_form_volume(cyl) - want) <= 1e-12
        return "boxes/spheres/cylinders closed forms exact at hand values"

    def group_closed():
        devs = []
        for i in range(12):
            p = sample_params(7, i)
            P, T = build(p)
            assert stl_io.euler_characteristic(P, T) == 4, \
                "%s: V-E+F = %d" % (p["shape"],
                                    stl_io.euler_characteristic(P, T))
            lref = l_ref(p)
            Pa, _Ta = build_bodies(p)[0]
            Pb, _Tb = build_bodies(p)[1]
            dev = abs((Pb[:, 1].min() - Pa[:, 1].max()) - gap(p))
            assert dev <= 1e-12 * lref, "y separation %g != gap" % dev
            devs.append(dev / lref)
        return "12 pairs: two bodies, V-E+F = 4, y separation = gap"

    def group_closest():
        for i in range(12):
            p = sample_params(7, i)
            g = gap(p)
            tol = 1e-12 * l_ref(p)
            Pa, _Ta = build_bodies(p)[0]
            Pb, _Tb = build_bodies(p)[1]
            fa = Pa[np.abs(Pa[:, 1] + g / 2.0) <= tol]
            fb = Pb[np.abs(Pb[:, 1] - g / 2.0) <= tol]
            assert len(fa) and len(fb), "%s: no facing vertices" % p["shape"]
            da = np.abs(fa[:, [0, 2]][:, None, :] - fb[:, [0, 2]][None, :, :])
            assert da.max(axis=2).min() <= tol, \
                "%s: facing vertices do not share (x, z)" % p["shape"]
        return "every pair: vertices at y = -g/2 and +g/2 with equal (x, z)"

    def group_volume():
        worst = {}
        for i in range(12):
            p = sample_params(7, i)
            P, T = build(p)
            dev = abs(stl_io.signed_volume(P, T) / closed_form_volume(p) - 1.0)
            assert dev <= 0.01, "%s dev %g" % (p["shape"], dev)
            worst[p["shape"]] = max(worst.get(p["shape"], 0.0), dev)
        return "12 pairs within 1 %%: %s" % ", ".join(
            "%s %.2e" % kv for kv in sorted(worst.items()))

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
            assert 0.5 <= p["gap_over_h"] <= 5.0
            strata[stratum(p)] += 1
        assert counts == {"boxes": 20, "spheres": 20, "cylinders": 20}, counts
        return ("20 boxes, 20 spheres, 20 cylinders; strata %s" % strata)

    def group_refusals():
        def refuses(params, frag):
            try:
                validate_params(params)
            except ValueError as e:
                assert frag in str(e), "wrong refusal: %s" % e
                return
            raise AssertionError("no ValueError naming %r" % frag)
        boxes = {"shape": "boxes", "length": 1.0, "height": 0.4,
                 "width_a": 0.3, "width_b": 0.5, "gap_over_h": 2.0,
                 "re_l": 2.0e4, "div": 16}
        spheres = {"shape": "spheres", "radius_a": 0.3, "radius_b": 0.2,
                   "n_theta": 64, "n_phi": 32, "gap_over_h": 2.0,
                   "re_l": 2.0e4, "div": 16}
        refuses(dict(boxes, gap_over_h=0.4), "gap_over_h: 0.4 outside")
        refuses(dict(boxes, gap_over_h=5.1), "gap_over_h: 5.1 outside")
        refuses(dict(spheres, radius_b=0.35), "radius_b: 0.35 outside")
        refuses(dict(spheres, n_theta=30), "n_theta: 30 is not a multiple")
        refuses(dict(spheres, n_phi=300), "n_phi: 300 outside")
        refuses({"shape": "cylinders", "radius_a": 0.3, "radius_b": 0.2,
                 "height": 0.03, "n_theta": 64, "gap_over_h": 2.0,
                 "re_l": 2.0e4, "div": 16}, "height: 0.03 outside")
        bad = dict(boxes)
        bad["extra"] = 1
        refuses(bad, "extra: unknown key for boxes")
        short = dict(boxes)
        del short["width_b"]
        refuses(short, "width_b: missing key for boxes")
        refuses(dict(boxes, re_l=2.0e5), "re_l: 200000 outside")
        refuses(dict(boxes, shape="cones"), "shape: 'cones' is not one of")
        return "10 by name"

    groups = [("[ok] closed forms", group_closed_forms),
              ("[ok] closed", group_closed),
              ("[ok] closest approach", group_closest),
              ("[ok] volume", group_volume),
              ("[ok] sampler", group_sampler),
              ("[ok] determinism", lambda: meshkit.check_determinism(
                  sys.modules[__name__], 7, (0, 1, 2))),
              ("[ok] rows", lambda: meshkit.check_rows(
                  sys.modules[__name__], 7, (0, 1, 2))),
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
