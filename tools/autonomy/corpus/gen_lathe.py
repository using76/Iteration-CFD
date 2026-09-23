#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""gen_lathe - docs/15 §E family B: 120 bodies of revolution about x.

Three shapes, 40 each: prolate ellipsoids, tangent-ogive + cylinder bodies
and tangent-ogive + cylinder + conical-boat-tail bodies, all with flat
bases.  Each pole (r == 0 node) is ONE vertex; every ring node is n_theta
vertices; consecutive nodes are stitched with the outward winding derived
and checked below.  The nose volume pi*(rho^2*Ln - Ln^3/3
- (rho-R)*rho^2*asin(Ln/rho)) is integrated from pi*r^2 (textbook solid
geometry); the closed forms anchor the 1 % volume gate of §F G-CORPUS.
The mesh is wound outward by construction and refused, never repaired,
when stl_io.check_closed does not pass.

    python tools/autonomy/corpus/gen_lathe.py --seed 1 --n 120 --out DIR
    python tools/autonomy/corpus/gen_lathe.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import stl_io  # noqa: E402
import schema  # noqa: E402

FAMILY = "B"
GENERATOR = "corpus/gen_lathe.py"
SOLID = "body"
SALT = 11
NU = 1.5e-5
RE_RANGE = (1.0e4, 1.0e5)
VOLUME_FORM = ("ellipsoid 4/3*pi*a*b^2; tangent-ogive nose "
               "pi*(rho^2*Ln - Ln^3/3 - (rho-R)*rho^2*asin(Ln/rho)) "
               "+ cylinder pi*R^2*Lc + cone frustum pi*Lb*(R^2+R*r_b+r_b^2)/3")

SHAPES = ("ellipsoid", "ogive", "boattail")
BASE_KEYS = ("shape", "length", "fineness", "re_l", "n_theta")
SHAPE_KEYS = {"ellipsoid": ("n_phi",),
              "ogive": ("nose_frac", "n_nose", "n_base"),
              "boattail": ("nose_frac", "tail_frac", "beta_deg", "n_nose", "n_base")}
_INT_KEYS = ("n_theta", "n_phi", "n_nose", "n_base")


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate_params(params: dict) -> None:
    """Refuse a broken params dict as ValueError("<key>: <why>")."""
    if not isinstance(params, dict):
        raise ValueError("params: expected a dict, got %s" % type(params).__name__)
    shape = params.get("shape")
    if shape not in SHAPES:
        raise ValueError("shape: %r is not one of %s" % (shape, ", ".join(SHAPES)))
    want = set(BASE_KEYS) | set(SHAPE_KEYS[shape])
    unknown = sorted(set(params) - want)
    if unknown:
        raise ValueError("%s: unknown key for shape %s" % (unknown[0], shape))
    missing = sorted(want - set(params))
    if missing:
        raise ValueError("%s: missing key for shape %s" % (missing[0], shape))
    for k in _INT_KEYS:
        if k in params and not _is_int(params[k]):
            raise ValueError("%s: expected an int, got %r" % (k, params[k]))
    if not _is_num(params["length"]) or not 0.2 <= params["length"] <= 5.0:
        raise ValueError("length: %r outside 0.2..5.0" % params["length"])
    if not _is_num(params["fineness"]) or not 1.5 <= params["fineness"] <= 12:
        raise ValueError("fineness: %r outside 1.5..12" % params["fineness"])
    R = params["length"] / (2.0 * params["fineness"])
    if "nose_frac" in params:
        nf = params["nose_frac"]
        if not _is_num(nf) or not 0.05 <= nf <= 0.5:
            raise ValueError("nose_frac: %r outside 0.05..0.5" % nf)
        if nf * params["length"] < R:
            raise ValueError("nose_frac: Ln = %.6g < R = %.6g (a tangent ogive "
                             "is pointed only when Ln >= R)" % (nf * params["length"], R))
    if "tail_frac" in params:
        tf = params["tail_frac"]
        if not _is_num(tf) or not 0.05 <= tf <= 0.3:
            raise ValueError("tail_frac: %r outside 0.05..0.3" % tf)
        Lc = params["length"] * (1.0 - params["nose_frac"] - tf)
        if Lc < 0.1 * params["length"]:
            raise ValueError("tail_frac: Lc = %.6g < 0.1*length" % Lc)
    if "beta_deg" in params:
        bd = params["beta_deg"]
        if not _is_num(bd) or not 0.0 < bd <= 20.0:
            raise ValueError("beta_deg: %r outside 0..20" % bd)
        Lb = params["tail_frac"] * params["length"]
        r_b = R - Lb * math.tan(math.radians(bd))
        if r_b < 0.3 * R:
            raise ValueError("beta_deg: r_b = %.6g < 0.3*R = %.6g" % (r_b, 0.3 * R))
    if not _is_num(params["re_l"]) or not RE_RANGE[0] <= params["re_l"] <= RE_RANGE[1]:
        raise ValueError("re_l: %r outside %s..%s"
                         % (params["re_l"], RE_RANGE[0], RE_RANGE[1]))
    ranges = {"n_theta": (8, 512), "n_phi": (8, 512), "n_nose": (4, 256),
              "n_base": (2, 64)}
    for k in _INT_KEYS:
        if k in params and not ranges[k][0] <= params[k] <= ranges[k][1]:
            raise ValueError("%s: %d outside %d..%d"
                             % (k, params[k], ranges[k][0], ranges[k][1]))


def sample_params(seed: int, index: int) -> dict:
    """One RNG, every draw made in the fixed table order for every shape."""
    rng = np.random.default_rng([SALT, int(seed), int(index)])
    shape = SHAPES[index % 3]
    length = round(float(rng.uniform(0.8, 1.5)), 4) + 0.0
    fineness = round(float(rng.uniform(2.5, 8.0)), 3) + 0.0
    R = length / (2.0 * fineness)
    nose_frac = round(float(rng.uniform(0.15, 0.35)), 4) + 0.0
    if nose_frac * length < 1.05 * R:
        nose_frac = round(1.05 * R / length + 0.0001, 4)
    tail_frac = round(float(rng.uniform(0.10, 0.25)), 4) + 0.0
    beta_deg = round(float(rng.uniform(4.0, 12.0)), 3) + 0.0
    Lb = tail_frac * length
    r_b = R - Lb * math.tan(math.radians(beta_deg))
    if r_b < 0.3 * R:
        beta_deg = round(math.degrees(math.atan(0.7 * R / Lb)) - 0.001, 3)
    re_l = float("%.4g" % 10 ** float(rng.uniform(4.0, 5.0)))
    params = {"shape": shape, "length": length, "fineness": fineness,
              "re_l": re_l, "n_theta": 64}
    if shape == "ellipsoid":
        params["n_phi"] = 64
    elif shape == "ogive":
        params.update(nose_frac=nose_frac, n_nose=24, n_base=4)
    else:
        params.update(nose_frac=nose_frac, tail_frac=tail_frac,
                      beta_deg=beta_deg, n_nose=24, n_base=4)
    return params


def _v_nose(Ln: float, R: float) -> float:
    """The tangent-ogive nose volume, integrated from pi*r^2."""
    rho = (R * R + Ln * Ln) / (2.0 * R)
    return math.pi * (rho * rho * Ln - Ln ** 3 / 3.0
                      - (rho - R) * rho * rho * math.asin(Ln / rho))


def _nodes(params: dict) -> list:
    """Profile nodes (x, r), nose -> tail -> base centre; r == 0 is a pole."""
    shape = params["shape"]
    length = params["length"]
    R = length / (2.0 * params["fineness"])
    nodes = [(0.0, 0.0)]
    if shape == "ellipsoid":
        a, b = length / 2.0, R
        n_phi = params["n_phi"]
        for k in range(1, n_phi):
            nodes.append((a * (1.0 - math.cos(math.pi * k / n_phi)),
                          b * math.sin(math.pi * k / n_phi)))
        nodes.append((length, 0.0))
        return nodes
    Ln = params["nose_frac"] * length
    n_nose = params["n_nose"]
    for k in range(1, n_nose + 1):
        x = Ln * (1.0 - math.cos(math.pi * k / (2 * n_nose)))
        r = math.sqrt(_rho(Ln, R) ** 2 - (Ln - x) ** 2) - (_rho(Ln, R) - R)
        if k == n_nose:
            x, r = Ln, R
        nodes.append((x, r))
    if shape == "ogive":
        Lc = length - Ln
        Nc = min(48, max(2, math.ceil(Lc / (0.5 * R))))
        for j in range(1, Nc + 1):
            nodes.append((Ln + Lc * j / Nc if j < Nc else length, R))
        n_base = params["n_base"]
        for i in range(1, n_base):
            nodes.append((length, R * (n_base - i) / n_base))
        nodes.append((length, 0.0))
        return nodes
    Lb = params["tail_frac"] * length
    Lc = length - Ln - Lb
    r_b = R - Lb * math.tan(math.radians(params["beta_deg"]))
    Nc = min(48, max(2, math.ceil(Lc / (0.5 * R))))
    for j in range(1, Nc + 1):
        nodes.append((Ln + Lc * j / Nc, R))
    Nb = min(24, max(2, math.ceil(Lb / (0.5 * R))))
    for j in range(1, Nb + 1):
        x = Ln + Lc + Lb * j / Nb
        r = R - (R - r_b) * j / Nb
        if j == Nb:
            x, r = length, r_b
        nodes.append((x, r))
    n_base = params["n_base"]
    for i in range(1, n_base):
        nodes.append((length, r_b * (n_base - i) / n_base))
    nodes.append((length, 0.0))
    return nodes


def _rho(Ln: float, R: float) -> float:
    return (R * R + Ln * Ln) / (2.0 * R)


def _vertices(nodes: list, n_theta: int):
    cosr = [math.cos(2.0 * math.pi * j / n_theta) for j in range(n_theta)]
    sinr = [math.sin(2.0 * math.pi * j / n_theta) for j in range(n_theta)]
    vs = []
    starts = []
    for x, r in nodes:
        starts.append(len(vs))
        if r == 0.0:
            vs.append((x, 0.0, 0.0))          # a pole is ONE vertex
        else:
            vs.extend((x, r * c, r * s) for c, s in zip(cosr, sinr))
    return np.array(vs, dtype=np.float64), starts


def _triangles(nodes: list, starts: list, n_theta: int) -> np.ndarray:
    tris = []
    for k in range(len(nodes) - 1):
        a, b = starts[k], starts[k + 1]
        ra, rb = nodes[k][1], nodes[k + 1][1]
        if ra == 0.0:                          # pole -> ring
            for j in range(n_theta):
                tris.append((a, b + (j + 1) % n_theta, b + j))
        elif rb == 0.0:                        # ring -> pole
            for j in range(n_theta):
                tris.append((a + j, a + (j + 1) % n_theta, b))
        else:                                  # ring -> ring
            for j in range(n_theta):
                j1 = (j + 1) % n_theta
                tris.append((a + j, a + j1, b + j1))
                tris.append((a + j, b + j1, b + j))
    return np.array(tris, dtype=np.int64)


def build(params: dict):
    """validate, then (P, T) outward by construction; check_closed raises."""
    validate_params(params)
    nodes = _nodes(params)
    n_theta = params["n_theta"]
    P, starts = _vertices(nodes, n_theta)
    T = _triangles(nodes, starts, n_theta)
    stl_io.check_closed(P, T)
    return P, T


def closed_form_volume(params: dict) -> float:
    shape = params["shape"]
    length = params["length"]
    R = length / (2.0 * params["fineness"])
    if shape == "ellipsoid":
        a, b = length / 2.0, R
        return 4.0 / 3.0 * math.pi * a * b * b
    Ln = params["nose_frac"] * length
    v_nose = _v_nose(Ln, R)
    if shape == "ogive":
        return v_nose + math.pi * R * R * (length - Ln)
    Lb = params["tail_frac"] * length
    Lc = length - Ln - Lb
    r_b = R - Lb * math.tan(math.radians(params["beta_deg"]))
    return (v_nose + math.pi * R * R * Lc
            + math.pi * Lb * (R * R + R * r_b + r_b * r_b) / 3.0)


def l_ref(params: dict) -> float:
    return params["length"]


def stratum(params: dict) -> str:
    if params["shape"] == "ellipsoid":
        return "easy" if params["fineness"] <= 5 else "medium"
    if params["shape"] == "ogive":
        return "hard" if params["fineness"] > 6.5 else "medium"
    return "hard"


def flow(params: dict) -> dict:
    return {"u_ref_m_s": params["re_l"] * NU / l_ref(params),
            "l_ref_m": l_ref(params), "nu_m2_s": NU,
            "note": "a priori target Re_L = %.4g (docs/15 §D.3 window)"
                    % params["re_l"]}


def geometry_id(seed: int, index: int) -> str:
    return "%s-%d-%03d" % (FAMILY, seed, index)


def make_row(seed: int, index: int):
    params = sample_params(seed, index)
    P, T = build(params)
    data = stl_io.stl_bytes(P, T, SOLID)
    row = {"schema": "autonomy-manifest/1", "geometry_id": geometry_id(seed, index),
           "family": FAMILY, "generator": GENERATOR, "params": params,
           "seed": int(seed), "stratum": stratum(params), "commensurate": False,
           "stl_sha256": hashlib.sha256(data).hexdigest(), "flow": flow(params)}
    return row, data


def write_row(row: dict, out_dir: str) -> str:
    P, T = build(row["params"])
    path = os.path.join(out_dir, row["geometry_id"] + ".stl")
    with open(path, "wb") as f:
        f.write(stl_io.stl_bytes(P, T, SOLID))
    with open(path, "rb") as f:
        got = hashlib.sha256(f.read()).hexdigest()
    if got != row["stl_sha256"]:
        raise ValueError("%s: rebuilt sha256 %s != row %s"
                         % (row["geometry_id"], got, row["stl_sha256"]))
    return path


def generate(seed: int, n: int, out_dir: str) -> list:
    if os.path.isdir(out_dir):
        if os.listdir(out_dir):
            raise ValueError("out: %s is not empty" % out_dir)
    elif os.path.exists(out_dir):
        raise ValueError("out: %s exists and is not a directory" % out_dir)
    else:
        os.makedirs(out_dir)
    rows = []
    for index in range(n):
        row, data = make_row(seed, index)
        with open(os.path.join(out_dir, row["geometry_id"] + ".stl"), "wb") as f:
            f.write(data)
        rows.append(row)
    man = os.path.join(out_dir, "manifest_%s.jsonl" % FAMILY)
    with open(man, "wb") as f:
        for row in rows:
            f.write((json.dumps(row, sort_keys=True, separators=(",", ":"),
                                ensure_ascii=True) + "\n").encode("ascii"))
    return rows


def _poison_global_rng() -> None:
    importlib.import_module("numpy").random.seed(999)
    importlib.import_module("random").seed(999)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="gen_lathe", description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()
    if args.seed is None or args.n is None or not args.out:
        ap.error("--seed, --n and --out are required (or --selftest)")
    rows = generate(args.seed, args.n, args.out)
    print("wrote %d STL(s) and manifest_%s.jsonl to %s" % (len(rows), FAMILY, args.out))
    return 0


# --- the selftest ---------------------------------------------------------

def _selftest():
    def group_forms():
        got = _v_nose(1.0, 1.0)
        want = 2.0 / 3.0 * math.pi
        assert abs(got / want - 1.0) <= 1e-12, "V_nose(Ln=R) %.15g" % got
        cf = closed_form_volume({"shape": "ellipsoid", "length": 2.0,
                                 "fineness": 1.0})
        assert abs(cf / (4.0 / 3.0 * math.pi) - 1.0) <= 1e-12, cf
        R, Lb = 1.0, 2.0
        frustum = math.pi * Lb * (R * R + R * R + R * R) / 3.0
        assert abs(frustum / (math.pi * R * R * Lb) - 1.0) <= 1e-12
        return ("V_nose(Ln=R) = 2/3*pi*R^3 to 1e-12; ellipsoid at fineness 1 "
                "= 4/3*pi*a^3; frustum at r_b = R = pi*R^2*Lb")

    def group_closed():
        seen = {"ellipsoid": 0, "ogive": 0, "boattail": 0}
        for index in range(24):
            params = sample_params(7, index)
            seen[params["shape"]] += 1
            P, T = build(params)
            axis = (P[:, 1] == 0.0) & (P[:, 2] == 0.0)
            assert int(axis.sum()) == 2, \
                "%s: %d axis vertices" % (geometry_id(7, index), int(axis.sum()))
            if params["shape"] != "ellipsoid":
                n = int((P[:, 0] == params["length"]).sum())
                want = params["n_base"] * params["n_theta"] + 1
                assert n == want, "%s: %d base vertices != %d" \
                    % (geometry_id(7, index), n, want)
        assert seen == {"ellipsoid": 8, "ogive": 8, "boattail": 8}, seen
        return ("seed 7, 24 bodies (8 of each shape): check_closed, exactly "
                "2 axis vertices (the poles), base x=length count "
                "n_base*n_theta + 1")

    def group_volume():
        worst = 0.0
        for index in range(24):
            params = sample_params(7, index)
            P, T = build(params)
            dev = abs(stl_io.signed_volume(P, T)
                      / closed_form_volume(params) - 1.0)
            worst = max(worst, dev)
            assert dev <= 0.01, "%s: %.4f off" % (geometry_id(7, index), dev)
        return ("24 bodies within 1 %% of the closed form; max dev %.3f %%"
                % (worst * 100))

    def group_sampler():
        counts = {"ellipsoid": 0, "ogive": 0, "boattail": 0}
        for index in range(120):
            params = sample_params(7, index)
            counts[params["shape"]] += 1
            validate_params(params)
            if params["shape"] != "ellipsoid":
                R = params["length"] / (2.0 * params["fineness"])
                assert params["nose_frac"] * params["length"] >= R, index
            if params["shape"] == "boattail":
                R = params["length"] / (2.0 * params["fineness"])
                Lb = params["tail_frac"] * params["length"]
                r_b = R - Lb * math.tan(math.radians(params["beta_deg"]))
                assert r_b >= 0.3 * R, index
                assert params["length"] - params["nose_frac"] * params["length"] \
                    - Lb >= 0.1 * params["length"], index
        assert counts == {"ellipsoid": 40, "ogive": 40, "boattail": 40}, counts
        return ("120 samples: shapes 40/40/40, every ogive Ln >= R, every "
                "boattail r_b >= 0.3*R and Lc >= 0.1*length, all valid")

    def group_determinism():
        row1, data1 = make_row(7, 5)
        P2, T2 = build(row1["params"])
        assert stl_io.stl_bytes(P2, T2, SOLID) == data1, "two builds differ"
        rows = [make_row(7, i)[0] for i in range(24)]
        assert rows[5]["stl_sha256"] == row1["stl_sha256"], \
            "the same row differs inside a batch"
        _poison_global_rng()
        assert make_row(7, 5)[1] == data1, "global RNG state leaked into a build"
        import re
        src = open(__file__, encoding="utf-8").read()
        hits = sorted(set(re.findall(r"np\.random\.\w+", src)))
        assert hits == ["np.random.default_rng"], hits
        imported = set(re.findall(r"(?m)^\s*import\s+(\w+)", src))
        assert "ran" + "dom" not in imported, "global random imported"
        P4, T4 = build(json.loads(json.dumps(row1["params"])))
        assert hashlib.sha256(stl_io.stl_bytes(P4, T4, SOLID)).hexdigest() \
            == row1["stl_sha256"], "JSON round trip changed the sha"
        return ("two builds identical, batch == alone, global RNG poison is a "
                "no-op, source clean, JSON round trip == row sha")

    def group_rows():
        for index in range(24):
            row, _data = make_row(7, index)
            assert "split" not in row
            for s in ("tuning", "test"):
                errs = schema.errors(dict(row, split=s), "ManifestRow")
                assert errs == [], "%s (%s): %s" % (row["geometry_id"], s, errs)
        return "24 rows validate as ManifestRow with split tuning and test"

    def group_refusals():
        ell = sample_params(7, 0)
        ogv = sample_params(7, 1)
        boa = sample_params(7, 2)
        assert (ell["shape"], ogv["shape"], boa["shape"]) == \
            ("ellipsoid", "ogive", "boattail")
        small_R_ogive = dict(ogv, length=1.0, fineness=8.0, nose_frac=0.05)
        cases = [
            ("shape", dict(ell, shape="sphere")),
            ("length", dict(ell, length=0.1)),
            ("fineness", dict(ell, fineness=1.0)),
            ("nose_frac", small_R_ogive),
            ("tail_frac", dict(boa, tail_frac=0.4)),
            ("beta_deg", dict(boa, beta_deg=25.0)),
            ("beta_deg", dict(boa, length=1.0, fineness=12.0, nose_frac=0.05,
                              tail_frac=0.25, beta_deg=20.0)),
            ("nose_frac", dict(ell, nose_frac=0.2)),
            ("beta_deg", {k: v for k, v in boa.items() if k != "beta_deg"}),
            ("n_theta", dict(ell, n_theta=4)),
            ("re_l", dict(ell, re_l=2e5)),
        ]
        for key, bad in cases:
            try:
                validate_params(bad)
            except ValueError as e:
                assert str(e).startswith(key + ":"), \
                    "%r does not name %s: %s" % (str(e), key, e)
                continue
            raise AssertionError("%s=%r was not refused" % (key, bad.get(key)))
        return "%d by name" % len(cases)

    groups = [("closed forms", group_forms), ("closed", group_closed),
              ("volume", group_volume), ("sampler", group_sampler),
              ("determinism", group_determinism), ("rows", group_rows),
              ("refusals", group_refusals)]
    n_ok = 0
    for name, fn in groups:
        try:
            note = fn()
        except AssertionError as e:
            print("SELFTEST FAIL: %s: %s" % (name, e))
            return 1
        n_ok += 1
        print("[ok] %s: %s" % (name, note))
    print("SELFTEST PASS")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    sys.exit(main())
