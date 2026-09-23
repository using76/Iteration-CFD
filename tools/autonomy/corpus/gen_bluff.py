#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""gen_bluff - docs/15 §E family D: 120 bluff bodies from one seed.

Eight-slot cycle: lattice-commensurate boxes (box_c, the G-BLC-0 tier-0
stratum), incommensurate boxes (box_n, nudged until the docs/15 §C lattice
rule says so), rounded boxes, cross-flow cylinders, and Ahmed-type bodies
rounded in the side view only, no stilts: front radius, rear slant 0-40 deg.
The proportions follow Ahmed, S. R., Ramm, G. & Faltin, G. (1984), "Some
Salient Features Of The Time-Averaged Ground Vehicle Wake", SAE Technical
Paper 840300, DOI 10.4271/840300 (resolved 2026-09-24 through doi.org and
Crossref) - only the published model proportions, as the centre of the draw
ranges.  Bodies are wound outward by construction and refused, never
repaired, when stl_io.check_closed does not pass.

    python tools/autonomy/corpus/gen_bluff.py --seed 1 --n 120 --out DIR
    python tools/autonomy/corpus/gen_bluff.py --selftest
"""
from __future__ import annotations

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import meshkit  # noqa: E402
import schema  # noqa: E402
import stl_io  # noqa: E402

FAMILY = "D"
GENERATOR = "corpus/gen_bluff.py"
SOLID = "body"
SALT = 13
NU = 1.5e-5
RE_RANGE = (1.0e4, 1.0e5)
DIV = 16
VOLUME_FORM = ("box_c n_x*n_y*n_z*s^3; box_n length*width*height; "
               "rounded_box height*(length*width - (4 - pi)*r^2); "
               "cylinder pi*(diameter/2)^2*height; ahmed width*(length*height "
               "- (4 - pi)/2*r_front^2 - 0.5*slant_len^2*sin*cos)")
SHAPE_CYCLE = ("box_c", "box_n", "box_c", "box_n", "rounded_box", "cylinder",
               "ahmed", "ahmed")

_KEYS = {
    "box_c": ("s", "n_x", "n_y", "n_z"),
    "box_n": ("length", "width", "height"),
    "rounded_box": ("length", "width", "height", "r_frac", "n_arc"),
    "cylinder": ("diameter", "height", "n_theta"),
    "ahmed": ("length", "width", "height", "r_front", "slant_deg",
              "slant_len", "n_arc"),
}
_INT_KEYS = ("n_x", "n_y", "n_z", "n_arc", "n_theta", "div")


def sample_params(seed: int, index: int) -> dict:
    """One meshkit.uniforms call, every draw in the fixed table order."""
    U = meshkit.uniforms(SALT, seed, index)
    shape = SHAPE_CYCLE[index % len(SHAPE_CYCLE)]
    p = {"shape": shape}
    if shape == "box_c":
        p["s"] = meshkit.fl(meshkit.lin(U, 0, 0.025, 0.05), 4)
        p["n_x"] = meshkit.ik(U, 1, 16, 32)
        p["n_y"] = max(2, int(round(meshkit.lin(U, 2, 0.3, 0.8) * p["n_x"])))
        p["n_z"] = max(2, int(round(meshkit.lin(U, 3, 0.25, 0.7) * p["n_x"])))
    elif shape == "box_n":
        p["length"] = meshkit.fl(meshkit.lin(U, 0, 0.6, 1.2), 4)
        p["width"] = meshkit.fl(meshkit.lin(U, 2, 0.3, 0.8) * p["length"], 4)
        p["height"] = meshkit.fl(meshkit.lin(U, 3, 0.25, 0.7) * p["length"], 4)
        for _ in range(1000):
            if meshkit.lattice_spacing(planes(p)) is None:
                break
            p["height"] = meshkit.fl(p["height"] + 0.0001, 4)
        else:
            raise RuntimeError("box_n %d: no incommensurate height in 1000 "
                               "nudges" % index)
    elif shape == "rounded_box":
        p["length"] = meshkit.fl(meshkit.lin(U, 0, 0.6, 1.2), 4)
        p["width"] = meshkit.fl(meshkit.lin(U, 2, 0.3, 0.8) * p["length"], 4)
        p["height"] = meshkit.fl(meshkit.lin(U, 3, 0.25, 0.7) * p["length"], 4)
        p["r_frac"] = meshkit.fl(meshkit.lin(U, 4, 0.05, 0.25), 4)
        p["n_arc"] = 8
    elif shape == "cylinder":
        p["diameter"] = meshkit.fl(meshkit.lin(U, 0, 0.2, 0.6), 4)
        p["height"] = meshkit.fl(meshkit.lin(U, 3, 1.0, 4.0) * p["diameter"], 4)
        p["n_theta"] = 64
    else:
        p["length"] = meshkit.fl(meshkit.lin(U, 0, 0.6, 1.2), 4)
        p["width"] = meshkit.fl(meshkit.lin(U, 2, 0.30, 0.45) * p["length"], 4)
        p["height"] = meshkit.fl(meshkit.lin(U, 3, 0.22, 0.33) * p["length"], 4)
        p["r_front"] = meshkit.fl(meshkit.lin(U, 4, 0.25, 0.40) * p["height"], 4)
        p["slant_deg"] = meshkit.fl(meshkit.lin(U, 5, 0.0, 40.0), 3)
        p["slant_len"] = meshkit.fl(meshkit.lin(U, 6, 0.15, 0.25) * p["length"],
                                    4)
        sin = math.sin(math.radians(p["slant_deg"]))
        if p["slant_len"] * sin > 0.6 * p["height"]:
            p["slant_len"] = meshkit.fl(
                0.6 * p["height"] / sin - 0.0001, 4)
        p["n_arc"] = 8
    p["re_l"] = float("%.4g" % 10 ** meshkit.lin(U, 11, 4.0, 5.0))
    p["div"] = DIV
    return p


def _fin(v: float, name: str, lo: float, hi: float) -> float:
    v = float(v)
    if not math.isfinite(v):
        raise ValueError("%s: %r is not finite" % (name, v))
    if v < lo or v > hi:
        raise ValueError("%s: %g outside %g..%g" % (name, v, lo, hi))
    return v


def _int(params: dict, key: str, lo: int, hi: int) -> int:
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
    if shape == "box_c":
        _fin(params["s"], "s", 0.005, 0.2)
        _int(params, "n_x", 4, 64)
        _int(params, "n_y", 2, 64)
        _int(params, "n_z", 2, 64)
    elif shape == "box_n":
        _box_dims(params, shape)
    elif shape == "rounded_box":
        _box_dims(params, shape)
        rf = _fin(params["r_frac"], "r_frac", 0.0, 0.45)
        if not rf > 0.0:
            raise ValueError("r_frac: %g is not > 0" % rf)
        _int(params, "n_arc", 2, 64)
    elif shape == "cylinder":
        d = _fin(params["diameter"], "diameter", 0.05, 2.0)
        _fin(params["height"], "height", 0.5 * d, 10.0 * d)
        n = _int(params, "n_theta", 8, 512)
        if n % 4 != 0:
            raise ValueError("n_theta: %d is not a multiple of 4" % n)
    else:
        _ahmed_dims(params)


def _box_dims(params: dict, shape: str) -> None:
    length = _fin(params["length"], "length", 0.2, 5.0)
    _fin(params["width"], "width", 0.05 * length, 2.0 * length)
    _fin(params["height"], "height", 0.05 * length, 2.0 * length)
    if shape == "box_n" and meshkit.lattice_spacing(planes(params)) is not None:
        raise ValueError("height: box_n dims are lattice-commensurate")


def _ahmed_dims(params: dict) -> None:
    length = _fin(params["length"], "length", 0.2, 5.0)
    _fin(params["width"], "width", 0.05 * length, 2.0 * length)
    height = _fin(params["height"], "height", 0.05 * length, 2.0 * length)
    _fin(params["r_front"], "r_front", 0.05 * height, 0.45 * height)
    _fin(params["slant_deg"], "slant_deg", 0.0, 40.0)
    slant = _fin(params["slant_len"], "slant_len", 0.05 * length,
                 0.3 * length)
    sin = math.sin(math.radians(params["slant_deg"]))
    if slant * sin > 0.6 * height + 1e-12:
        raise ValueError("slant_len: slant_len*sin(slant_deg) = %g above "
                         "0.6*height = %g" % (slant * sin, 0.6 * height))
    if slant * math.cos(math.radians(params["slant_deg"])) \
            + params["r_front"] > 0.9 * length:
        raise ValueError("slant_len: slant_len*cos + r_front above 0.9*length")


def l_ref(params: dict) -> float:
    if params["shape"] == "box_c":
        return params["n_x"] * params["s"]
    if params["shape"] == "cylinder":
        return params["diameter"]
    return params["length"]


def planes(params: dict):
    """The axis-aligned plane coordinates (x, y, z lists), or None."""
    if params["shape"] == "box_c":
        s = params["s"]
        return [[0.0, params["n_x"] * s],
                [-(params["n_y"] * s) / 2.0, (params["n_y"] * s) / 2.0],
                [0.0, params["n_z"] * s]]
    if params["shape"] == "box_n":
        return [[0.0, params["length"]],
                [-params["width"] / 2.0, params["width"] / 2.0],
                [0.0, params["height"]]]
    return None


def closed_form_volume(params: dict) -> float:
    shape = params["shape"]
    if shape == "box_c":
        return params["n_x"] * params["n_y"] * params["n_z"] * params["s"] ** 3
    if shape == "box_n":
        return params["length"] * params["width"] * params["height"]
    if shape == "rounded_box":
        r = params["r_frac"] * min(params["length"], params["width"])
        return params["height"] * (params["length"] * params["width"]
                                   - (4.0 - math.pi) * r * r)
    if shape == "cylinder":
        return math.pi * (params["diameter"] / 2.0) ** 2 * params["height"]
    phi = math.radians(params["slant_deg"])
    return params["width"] * (params["length"] * params["height"]
                              - (4.0 - math.pi) / 2.0 * params["r_front"] ** 2
                              - 0.5 * params["slant_len"] ** 2
                              * math.sin(phi) * math.cos(phi))


def stratum(params: dict) -> str:
    shape = params["shape"]
    if shape == "box_c":
        return "easy"
    if shape == "ahmed":
        return "hard" if params["slant_deg"] >= 25.0 else "medium"
    return "medium"


def n_bodies(params: dict) -> int:
    return 1


def expected_features(params: dict) -> dict:
    shape = params["shape"]
    return {"commensurate": shape == "box_c",
            "lattice_base_size_m": (meshkit.lattice_spacing(planes(params))
                                    if shape == "box_c" else None),
            "planar_one": shape in ("box_c", "box_n")}


def flow(params: dict) -> dict:
    return {"u_ref_m_s": params["re_l"] * NU / l_ref(params),
            "l_ref_m": l_ref(params), "nu_m2_s": NU,
            "note": "a priori target Re_L = %.4g (docs/15 §D.3 window)"
                    % params["re_l"]}


def geometry_id(seed: int, index: int) -> str:
    return "%s-%d-%03d" % (FAMILY, seed, index)


def _ring_rbox(params: dict) -> list:
    """The top-view rounded rectangle, counter-clockwise in (x, y)."""
    L = params["length"]
    W2 = params["width"] / 2.0
    r = params["r_frac"] * min(L, params["width"])
    n_arc = params["n_arc"]
    step = l_ref(params) / DIV
    return (meshkit.line((r, -W2), (L - r, -W2), meshkit.divs(L - 2.0 * r, step))
            + meshkit.arc((L - r, -W2), (L - r, -W2 + r), r, 270.0, 360.0, n_arc)
            + meshkit.line((L, -W2 + r), (L, W2 - r),
                           meshkit.divs(2.0 * W2 - 2.0 * r, step))
            + meshkit.arc((L, W2 - r), (L - r, W2 - r), r, 0.0, 90.0, n_arc)
            + meshkit.line((L - r, W2), (r, W2),
                           meshkit.divs(L - 2.0 * r, step))
            + meshkit.arc((r, W2), (r, W2 - r), r, 90.0, 180.0, n_arc)
            + meshkit.line((0.0, W2 - r), (0.0, -W2 + r),
                           meshkit.divs(2.0 * W2 - 2.0 * r, step))
            + meshkit.arc((0.0, -W2 + r), (r, -W2 + r), r, 180.0, 270.0, n_arc))


def _ring_ahmed(params: dict) -> list:
    """The side-view profile, counter-clockwise in (x, z)."""
    L = params["length"]
    H = params["height"]
    rf = params["r_front"]
    phi = math.radians(params["slant_deg"])
    d = params["slant_len"] * math.sin(phi)
    a = params["slant_len"] * math.cos(phi)
    n_arc = params["n_arc"]
    step = l_ref(params) / DIV

    def ln(p, q, ln_len):
        return meshkit.line(p, q, meshkit.divs(ln_len, step))

    return (ln((rf, 0.0), (L, 0.0), L - rf)
            + ln((L, 0.0), (L, H - d), H - d)
            + ln((L, H - d), (L - a, H), math.hypot(a, d))
            + ln((L - a, H), (rf, H), L - a - rf)
            + meshkit.arc((rf, H), (rf, H - rf), rf, 90.0, 180.0, n_arc)
            + ln((0.0, H - rf), (0.0, rf), H - 2.0 * rf)
            + meshkit.arc((0.0, rf), (rf, rf), rf, 180.0, 270.0, n_arc))


def build(params: dict):
    """Validate, then wind the body outward; check_closed inside refuses."""
    validate_params(params)
    shape = params["shape"]
    step = l_ref(params) / DIV
    if shape == "box_c":
        s = params["s"]
        P, T = meshkit.grid_box(
            meshkit.axis_coords(0.0, params["n_x"] * s,
                                meshkit.divs(params["n_x"] * s, step)),
            meshkit.axis_coords(-(params["n_y"] * s) / 2.0,
                                (params["n_y"] * s) / 2.0,
                                meshkit.divs(params["n_y"] * s, step)),
            meshkit.axis_coords(0.0, params["n_z"] * s,
                                meshkit.divs(params["n_z"] * s, step)))
    elif shape == "box_n":
        P, T = meshkit.grid_box(
            meshkit.axis_coords(0.0, params["length"],
                                meshkit.divs(params["length"], step)),
            meshkit.axis_coords(-params["width"] / 2.0, params["width"] / 2.0,
                                meshkit.divs(params["width"], step)),
            meshkit.axis_coords(0.0, params["height"],
                                meshkit.divs(params["height"], step)))
    elif shape == "rounded_box":
        P, T = meshkit.prism(_ring_rbox(params),
                             meshkit.axis_coords(0.0, params["height"],
                                                 meshkit.divs(params["height"],
                                                              step)),
                             (0, 1, 2),
                             (params["length"] / 2.0, 0.0))
    elif shape == "cylinder":
        P, T = meshkit.prism(
            meshkit.circle(params["diameter"] / 2.0, 0.0,
                           params["diameter"] / 2.0, params["n_theta"]),
            meshkit.axis_coords(0.0, params["height"],
                                meshkit.divs(params["height"], step)),
            (0, 1, 2), (params["diameter"] / 2.0, 0.0))
    else:
        P, T = meshkit.prism(_ring_ahmed(params),
                             meshkit.axis_coords(-params["width"] / 2.0,
                                                 params["width"] / 2.0,
                                                 meshkit.divs(params["width"],
                                                              step)),
                             (0, 2, 1),
                             (params["length"] / 2.0, params["height"] / 2.0))
    return P, T


def make_row(seed: int, index: int):
    return meshkit.row_make(sys.modules[__name__], seed, index)


def write_row(row: dict, out_dir: str) -> str:
    return meshkit.row_write(sys.modules[__name__], row, out_dir)


def generate(seed: int, n: int, out_dir: str) -> list:
    return meshkit.row_generate(sys.modules[__name__], seed, n, out_dir)


def main(argv=None) -> int:
    return meshkit.row_cli(sys.modules[__name__], argv, _selftest)


# --- the selftest ---------------------------------------------------------


def _shape_counts(seed: int, n: int) -> dict:
    counts = {}
    for index in range(n):
        p = sample_params(seed, index)
        counts[p["shape"]] = counts.get(p["shape"], 0) + 1
    return counts


def _selftest() -> int:
    def group_closed_forms():
        box = {"shape": "box_n", "length": 1.0, "width": 0.5, "height": 0.4,
               "re_l": 1.0e4, "div": 16}
        rbox = dict(box, shape="rounded_box", r_frac=1.0e-8)
        assert abs(closed_form_volume(rbox) - closed_form_volume(box)) \
            <= 1e-12 * closed_form_volume(box)
        ah0 = dict(box, shape="ahmed", width=0.4, height=0.3, r_front=1.0e-8,
                   slant_deg=0.0, slant_len=0.2, n_arc=8)
        assert abs(closed_form_volume(ah0) - 1.0 * 0.4 * 0.3) \
            <= 1e-12 * 0.12
        bc = {"shape": "box_c", "s": 0.03, "n_x": 20, "n_y": 10, "n_z": 8,
              "re_l": 1.0e4, "div": 16}
        assert abs(closed_form_volume(bc) - 20 * 10 * 8 * 0.03 ** 3) <= 1e-18
        return ("rounded_box -> 0 and ahmed at slant 0, r_front -> 0 equal "
                "the box; box_c equals n_x*n_y*n_z*s^3")

    def group_closed():
        tris = []
        for i in range(16):
            p = sample_params(7, i)
            P, T = build(p)
            tris.append(len(T))
        return "16 bodies of seed 7 build closed, %d..%d triangles" \
            % (min(tris), max(tris))

    def group_volume():
        worst = {}
        for i in range(16):
            p = sample_params(7, i)
            P, T = build(p)
            dev = abs(stl_io.signed_volume(P, T) / closed_form_volume(p) - 1.0)
            lim = 1e-12 if p["shape"] in ("box_c", "box_n") else 0.01
            assert dev <= lim, "%s dev %g" % (p["shape"], dev)
            worst[p["shape"]] = max(worst.get(p["shape"], 0.0), dev)
        return "16 bodies within 1 %% (planar 1e-12): %s" % ", ".join(
            "%s %.2e" % kv for kv in sorted(worst.items()))

    def group_lattice():
        for i in range(120):
            p = sample_params(7, i)
            if p["shape"] == "box_c":
                want = math.gcd(p["n_x"], p["n_y"], p["n_z"]) * p["s"]
                got = meshkit.lattice_spacing(planes(p))
                assert got is not None and abs(got / want - 1.0) <= 1e-12, \
                    "box_c %d: %r != %r" % (i, got, want)
            elif p["shape"] == "box_n":
                assert meshkit.lattice_spacing(planes(p)) is None, \
                    "box_n %d is commensurate" % i
            else:
                assert planes(p) is None
        return "120 draws: box_c == gcd(n_x,n_y,n_z)*s, box_n None, others no planes"

    def group_sampler():
        counts = _shape_counts(7, 120)
        assert counts == {"box_c": 30, "box_n": 30, "rounded_box": 15,
                          "cylinder": 15, "ahmed": 30}, counts
        strata = {"easy": 0, "medium": 0, "hard": 0}
        for i in range(120):
            p = sample_params(7, i)
            validate_params(p)
            for v in p.values():
                if isinstance(v, float) and v == 0.0:
                    assert math.copysign(1.0, v) > 0.0, "-0.0 in params %r" % p
            if p["shape"] == "ahmed":
                assert 0.0 <= p["slant_deg"] <= 40.0
                sin = math.sin(math.radians(p["slant_deg"]))
                assert p["slant_len"] * sin <= 0.6 * p["height"] + 1e-12
            strata[stratum(p)] += 1
        return ("30 box_c, 30 box_n, 15 rounded_box, 15 cylinder, 30 ahmed; "
                "strata %s" % strata)

    def group_refusals():
        def refuses(params, frag):
            try:
                validate_params(params)
            except ValueError as e:
                assert frag in str(e), "wrong refusal: %s" % e
                return
            raise AssertionError("no ValueError naming %r" % frag)
        base = {"shape": "box_c", "s": 0.03, "n_x": 20, "n_y": 10, "n_z": 8,
                "re_l": 2.0e4, "div": 16}
        refuses(dict(base, s=0.001), "s: 0.001 outside")
        refuses(dict(base, n_x=3), "n_x: 3 outside")
        refuses({"shape": "box_n", "length": 1.0, "width": 0.5, "height": 0.25,
                 "re_l": 2.0e4, "div": 16},
                "height: box_n dims are lattice-commensurate")
        refuses({"shape": "box_n", "length": 1.0, "width": 0.01, "height": 0.3,
                 "re_l": 2.0e4, "div": 16}, "width: 0.01 outside")
        refuses({"shape": "rounded_box", "length": 1.0, "width": 0.5,
                 "height": 0.3, "r_frac": 0.5, "n_arc": 8, "re_l": 2.0e4,
                 "div": 16}, "r_frac: 0.5 outside")
        refuses({"shape": "cylinder", "diameter": 0.4, "height": 0.6,
                 "n_theta": 30, "re_l": 2.0e4, "div": 16},
                "n_theta: 30 is not a multiple of 4")
        refuses({"shape": "ahmed", "length": 1.0, "width": 0.4, "height": 0.3,
                 "r_front": 0.09, "slant_deg": 45.0, "slant_len": 0.2,
                 "n_arc": 8, "re_l": 2.0e4, "div": 16},
                "slant_deg: 45 outside")
        refuses({"shape": "ahmed", "length": 1.0, "width": 0.4, "height": 0.3,
                 "r_front": 0.09, "slant_deg": 40.0, "slant_len": 0.29,
                 "n_arc": 8, "re_l": 2.0e4, "div": 16},
                "slant_len: slant_len*sin(slant_deg)")
        bad = dict(base)
        bad["extra"] = 1
        refuses(bad, "extra: unknown key for box_c")
        short = dict(base)
        del short["n_z"]
        refuses(short, "n_z: missing key for box_c")
        refuses(dict(base, re_l=5.0e3), "re_l: 5000 outside")
        refuses(dict(base, shape="tetra"), "shape: 'tetra' is not one of")
        refuses({"shape": "rounded_box", "length": 1.0, "width": 0.01,
                 "height": 0.3, "r_frac": 0.1, "n_arc": 8, "re_l": 2.0e4,
                 "div": 16}, "width: 0.01 outside")
        return "13 by name"

    groups = [("[ok] closed forms", group_closed_forms),
              ("[ok] closed", group_closed),
              ("[ok] volume", group_volume),
              ("[ok] lattice", group_lattice),
              ("[ok] sampler", group_sampler),
              ("[ok] determinism", lambda: meshkit.check_determinism(
                  sys.modules[__name__], 7, (0, 1, 4, 5, 6))),
              ("[ok] rows", lambda: meshkit.check_rows(
                  sys.modules[__name__], 7, (0, 1, 4, 5, 6))),
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
