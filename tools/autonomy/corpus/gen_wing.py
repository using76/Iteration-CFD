#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""gen_wing - docs/15 §E family A: 120 NACA 4-digit wings from one seed.

The section is NACA Report 460's 4-digit formula, verbatim with its open
trailing edge (-0.1015); the closed-trailing-edge modification is NOT used,
because Report 460 does not contain it.  Citation: Jacobs, E. N., Ward, K. E.
& Pinkerton, R. M. (1933), The characteristics of 78 related airfoil sections
from tests in the variable-density wind tunnel, NACA Report No. 460
(NACA-TR-460), NTRS 19930091108, https://ntrs.nasa.gov/citations/19930091108
(no DOI; US government work; resolved 2026-09-24).

Full-span planform along y, chord along x, thickness along z, symmetric
about y = 0: quarter-chord sweep, linear taper, linear twist, planar tip
caps.  The mesh is wound outward by construction and refused, never
repaired, when stl_io.check_closed does not pass.

    python tools/autonomy/corpus/gen_wing.py --seed 1 --n 120 --out DIR
    python tools/autonomy/corpus/gen_wing.py --selftest
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

FAMILY = "A"
GENERATOR = "corpus/gen_wing.py"
SOLID = "wing"
SALT = 10
NU = 1.5e-5
RE_RANGE = (1.0e4, 1.0e5)
VOLUME_FORM = ("0.685*t*span*(c_root^2 + c_root*c_tip + c_tip^2)/3 "
               "(NACA Report 460 section area 0.685*t*c^2)")

PARAM_KEYS = ("m", "p", "tt", "code", "c_root", "span", "taper", "sweep_deg",
              "twist_deg", "re_l", "nc", "ns_half")
_INT_KEYS = ("m", "p", "tt", "nc", "ns_half")


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate_params(params: dict) -> None:
    """Refuse a broken params dict as ValueError("<key>: <why>")."""
    if not isinstance(params, dict):
        raise ValueError("params: expected a dict, got %s" % type(params).__name__)
    unknown = sorted(set(params) - set(PARAM_KEYS))
    if unknown:
        raise ValueError("%s: unknown key" % unknown[0])
    missing = sorted(set(PARAM_KEYS) - set(params))
    if missing:
        raise ValueError("%s: missing key" % missing[0])
    for k in _INT_KEYS:
        if not _is_int(params[k]):
            raise ValueError("%s: expected an int, got %r" % (k, params[k]))
    m, p, tt = params["m"], params["p"], params["tt"]
    if not 0 <= m <= 4:
        raise ValueError("m: %d outside 0..4 (camber capped so the Report 460 "
                         "section area stays within 1 %% of 0.685*t)" % m)
    if m == 0:
        if p != 0:
            raise ValueError("p: must be 0 when m == 0 (got %d)" % p)
    elif not 2 <= p <= 6:
        raise ValueError("p: %d outside 2..6" % p)
    if not 6 <= tt <= 24:
        raise ValueError("tt: %d outside 6..24" % tt)
    want = "%d%d%02d" % (m, p, tt)
    if params["code"] != want:
        raise ValueError("code: %r does not match the code %r of (m, p, tt)"
                         % (params["code"], want))
    for k, lo, hi in (("c_root", 0.3, 2.0), ("taper", 0.2, 1.0),
                      ("sweep_deg", 0.0, 45.0), ("twist_deg", -10.0, 10.0)):
        if not _is_num(params[k]) or not lo <= params[k] <= hi:
            raise ValueError("%s: %r outside %s..%s" % (k, params[k], lo, hi))
    cr, span = params["c_root"], params["span"]
    if not _is_num(span) or not cr * 1.0 <= span <= cr * 10.0:
        raise ValueError("span: %r outside %s..%s (1..10 x c_root)"
                         % (span, cr * 1.0, cr * 10.0))
    if not _is_num(params["re_l"]) or not RE_RANGE[0] <= params["re_l"] <= RE_RANGE[1]:
        raise ValueError("re_l: %r outside %s..%s"
                         % (params["re_l"], RE_RANGE[0], RE_RANGE[1]))
    if not 8 <= params["nc"] <= 200:
        raise ValueError("nc: %d outside 8..200" % params["nc"])
    if not 2 <= params["ns_half"] <= 100:
        raise ValueError("ns_half: %d outside 2..100" % params["ns_half"])


def sample_params(seed: int, index: int) -> dict:
    """The one RNG: default_rng([SALT, seed, index]), fixed draw order."""
    rng = np.random.default_rng([SALT, int(seed), int(index)])
    m = int(rng.integers(0, 5))
    p = int(rng.integers(2, 7))
    tt = int(rng.integers(6, 25))
    if m == 0:
        p = 0
    code = "%d%d%02d" % (m, p, tt)
    c_root = round(float(rng.uniform(0.6, 1.2)), 4) + 0.0
    span = round(float(rng.uniform(2.5, 6.0)) * c_root, 4) + 0.0
    taper = round(float(rng.uniform(0.3, 1.0)), 4) + 0.0
    if index % 4 == 0:
        taper = 1.0
    sweep_deg = round(float(rng.uniform(0.0, 35.0)), 3) + 0.0
    twist_deg = round(float(rng.uniform(-6.0, 0.0)), 3) + 0.0
    if index % 3 == 0:
        twist_deg = 0.0
    re_l = float("%.4g" % 10 ** float(rng.uniform(4.0, 5.0)))
    return {"m": m, "p": p, "tt": tt, "code": code, "c_root": c_root,
            "span": span, "taper": taper, "sweep_deg": sweep_deg,
            "twist_deg": twist_deg, "re_l": re_l, "nc": 60, "ns_half": 12}


def _section(m: int, p: int, tt: int, nc: int):
    """The Report 460 ring (sx, sz), R = 2*nc+1 points, LE first."""
    m_ = m / 100.0
    p_ = p / 10.0
    t = tt / 100.0
    i = np.arange(nc + 1, dtype=np.float64)
    x = 0.5 * (1.0 - np.cos(np.pi * i / nc))
    yt = 5.0 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2
                    + 0.2843 * x ** 3 - 0.1015 * x ** 4)
    if m == 0:
        yc = np.zeros(nc + 1)
        dyc = np.zeros(nc + 1)
    else:
        yc = np.where(x < p_, m_ / p_ ** 2 * (2.0 * p_ * x - x ** 2),
                      m_ / (1.0 - p_) ** 2 * (1.0 - 2.0 * p_ + 2.0 * p_ * x - x ** 2))
        dyc = np.where(x < p_, 2.0 * m_ / p_ ** 2 * (p_ - x),
                       2.0 * m_ / (1.0 - p_) ** 2 * (p_ - x))
    th = np.arctan(dyc)
    xu = x - yt * np.sin(th)
    zu = yc + yt * np.cos(th)
    xl = x + yt * np.sin(th)
    zl = yc - yt * np.cos(th)
    sx = np.empty(2 * nc + 1)
    sz = np.empty(2 * nc + 1)
    sx[0], sz[0] = 0.0, 0.0            # the leading edge is ONE point
    sx[1:nc + 1], sz[1:nc + 1] = xu[1:], zu[1:]
    sx[nc + 1:], sz[nc + 1:] = xl[nc:0:-1], zl[nc:0:-1]   # L_nc .. L_1
    return sx, sz


def _ring_area(sx, sz) -> float:
    """|shoelace| of the closed section ring."""
    return 0.5 * float(np.abs(np.sum(sx * np.roll(sz, -1) - np.roll(sx, -1) * sz)))


def _triangles(nc: int, ns: int) -> np.ndarray:
    R = 2 * nc + 1
    tris = []
    for k in range(2 * ns):
        for r in range(R):
            r1 = (r + 1) % R
            a, b = k * R + r, k * R + r1
            c, d = (k + 1) * R + r1, (k + 1) * R + r
            tris.append((a, b, c))
            tris.append((a, c, d))
    hi = 2 * ns * R                       # tip cap at k = 2*ns, normal +y
    tris.append((hi, hi + 1, hi + R - 1))
    for i in range(1, nc):
        tris.append((hi + i, hi + i + 1, hi + R - (i + 1)))
        tris.append((hi + i, hi + R - (i + 1), hi + R - i))
    tris.append((0, R - 1, 1))            # cap at k = 0, normal -y: 2nd/3rd swapped
    for i in range(1, nc):
        tris.append((i, R - (i + 1), i + 1))
        tris.append((i, R - i, R - (i + 1)))
    return np.array(tris, dtype=np.int64)


def _mesh(params: dict):
    m, p, tt = params["m"], params["p"], params["tt"]
    nc, ns = params["nc"], params["ns_half"]
    sx, sz = _section(m, p, tt, nc)
    R = 2 * nc + 1
    c_root, span = params["c_root"], params["span"]
    taper, sweep_deg, twist_deg = params["taper"], params["sweep_deg"], params["twist_deg"]
    ks = np.arange(2 * ns + 1, dtype=np.float64)
    eta = np.abs(ks - ns) / ns
    y = (span / 2.0) * ((ks - ns) / ns)
    ck = c_root * (1.0 - (1.0 - taper) * eta)
    tau = math.radians(twist_deg) * eta
    xqc = 0.25 * c_root + math.tan(math.radians(sweep_deg)) * (span / 2.0) * eta
    ct, st = np.cos(tau), np.sin(tau)
    P = np.empty(((2 * ns + 1) * R, 3))
    for k in range(2 * ns + 1):
        xs = (sx - 0.25) * ck[k]
        zs = sz * ck[k]
        i0 = k * R
        P[i0:i0 + R, 0] = xs * ct[k] + zs * st[k] + xqc[k]
        P[i0:i0 + R, 1] = y[k]
        P[i0:i0 + R, 2] = -xs * st[k] + zs * ct[k]
    return P, _triangles(nc, ns)


def build(params: dict):
    """validate, then (P, T) outward by construction; check_closed raises."""
    validate_params(params)
    P, T = _mesh(params)
    stl_io.check_closed(P, T)
    return P, T


def closed_form_volume(params: dict) -> float:
    t = params["tt"] / 100.0
    cr, c_tip = params["c_root"], params["taper"] * params["c_root"]
    return 0.685 * t * params["span"] * (cr * cr + cr * c_tip + c_tip * c_tip) / 3.0


def l_ref(params: dict) -> float:
    return params["c_root"] * (1.0 + params["taper"]) / 2.0


def stratum(params: dict) -> str:
    if (params["tt"] <= 9 or params["taper"] < 0.45
            or params["sweep_deg"] > 25):
        return "hard"
    if (params["taper"] == 1.0 and params["sweep_deg"] < 10
            and params["tt"] >= 12 and params["twist_deg"] == 0.0):
        return "easy"
    return "medium"


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
    ap = argparse.ArgumentParser(prog="gen_wing", description=__doc__.splitlines()[0])
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
    def group_section():
        nc = 400
        sx, sz = _section(0, 0, 12, nc)
        upper = sz[1:nc + 1]
        dev = abs(float(upper.max()) / 0.06 - 1.0)
        assert dev <= 1e-3, "max half-thickness %.6f, %.3g off 0.06" % (upper.max(), dev)
        assert (sx[0], sz[0]) == (0.0, 0.0), "the LE is not one point (0, 0)"
        t = 0.12
        assert abs((sz[nc] - sz[nc + 1]) - 0.021 * t) <= 1e-12, \
            "TE base thickness %.9g" % (sz[nc] - sz[nc + 1])
        return ("NACA 0012 at nc 400: max z %.6f (0.06 within %.3g %%), "
                "LE (0, 0), TE base 0.021*t exact"
                % (float(upper.max()), dev * 100))

    def group_areas():
        worst, worst_sym, n = 0.0, 0.0, 0
        for m in range(5):
            for p in ((0,) if m == 0 else tuple(range(2, 7))):
                for tt in range(6, 25):
                    sx, sz = _section(m, p, tt, 60)
                    a = _ring_area(sx, sz)
                    target = 0.685 * tt / 100.0
                    dev = abs(a / target - 1.0)
                    n += 1
                    worst = max(worst, dev)
                    if m == 0:
                        worst_sym = max(worst_sym, dev)
                    assert dev <= 0.01, "code %d%d%02d: %.4f off" % (m, p, tt, dev)
                    if m == 0:
                        assert dev <= 1e-3, "symmetric %d%02d: %.4f" % (m, tt, dev)
        return ("%d codes at nc 60 within 1 %% of 0.685*t (symmetric within "
                "0.1 %%); max dev %.3f %%, symmetric %.3f %%"
                % (n, worst * 100, worst_sym * 100))

    def group_closed():
        span_w = None
        for index in range(24):
            params = sample_params(7, index)
            P, T = build(params)
            assert (len(P), len(T)) == (3025, 6046), \
                "%s: (nP, nT) = (%d, %d)" % (geometry_id(7, index), len(P), len(T))
            n_side = len(T) - 2 * (2 * params["nc"] - 1)
            caps = ((T[n_side:n_side + 2 * params["nc"] - 1], +1),
                    (T[len(T) - (2 * params["nc"] - 1):], -1))
            span_w = params["span"]
            for cap, sign in caps:
                ny = stl_io.unit_normals(P, cap)[:, 1]
                assert (ny > 0).all() if sign > 0 else (ny < 0).all(), \
                    "%s: cap normal not %+.0fy" % (geometry_id(7, index), sign)
                ys = P[cap.reshape(-1), 1]
                assert ((ys == sign * span_w / 2.0).all()), "%s: cap not planar" \
                    % geometry_id(7, index)
        return ("seed 7, 24 wings: check_closed, nP 3025, nT 6046, both caps "
                "planar with normals +y / -y")

    def group_volume():
        params = sample_params(7, 0)
        assert params["taper"] == 1.0 and params["twist_deg"] == 0.0, params
        sx, sz = _section(params["m"], params["p"], params["tt"], params["nc"])
        P, T = build(params)
        v = stl_io.signed_volume(P, T)
        target = _ring_area(sx, sz) * params["c_root"] ** 2 * params["span"]
        dev1 = abs(v / target - 1.0)
        cf = closed_form_volume(params)
        dev2 = abs(v / cf - 1.0)
        assert dev1 <= 1e-9, "V != A*c^2*span: %.3g" % dev1
        assert dev2 <= 0.01, "V off closed form: %.4f" % dev2
        return ("V = %.6g = area*c^2*span to %.2g rel; off 0.685*t*c^2*span by "
                "%.4f %%" % (v, dev1, dev2 * 100))

    def group_sampler():
        n_taper, n_twist, strata = 0, 0, {"easy": 0, "medium": 0, "hard": 0}
        for index in range(120):
            params = sample_params(7, index)
            assert params["m"] <= 4
            assert params["taper"] == 1.0 if index % 4 == 0 else True
            n_taper += params["taper"] == 1.0
            n_twist += params["twist_deg"] == 0.0
            for k in ("c_root", "span", "taper", "sweep_deg", "twist_deg", "re_l"):
                v = params[k]
                if v == 0.0:
                    assert math.copysign(1, v) > 0, "%s carries -0.0" % k
            validate_params(params)
            strata[stratum(params)] += 1
        assert n_taper >= 30, n_taper
        assert n_twist >= 40, n_twist
        return ("120 samples: taper==1 %d, twist==0 %d, no -0.0, all valid; "
                "strata easy %d / medium %d / hard %d"
                % (n_taper, n_twist, strata["easy"], strata["medium"], strata["hard"]))

    def group_determinism():
        row1, data1 = make_row(7, 5)
        P2, T2 = build(row1["params"])
        assert stl_io.stl_bytes(P2, T2, SOLID) == data1, "two builds differ"
        rows = [make_row(7, i)[0] for i in range(24)]
        assert rows[5]["stl_sha256"] == row1["stl_sha256"], \
            "the same row differs inside a batch"
        _poison_global_rng()
        row3, data3 = make_row(7, 5)
        assert data3 == data1, "global RNG state leaked into a build"
        src = open(__file__, encoding="utf-8").read()
        import re
        hits = sorted(set(re.findall(r"np\.random\.\w+", src)))
        assert hits == ["np.random.default_rng"], hits
        imported = set(re.findall(r"(?m)^\s*import\s+(\w+)", src))
        assert "ran" + "dom" not in imported, "global random imported"
        import json as _json
        P4, T4 = build(_json.loads(_json.dumps(row1["params"])))
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
        good = sample_params(7, 1)
        cases = [("m", dict(good, m=5)), ("p", dict(good, p=0)),
                 ("p", dict(good, m=0, p=3)), ("tt", dict(good, tt=5)),
                 ("code", dict(good, code="999")),
                 ("c_root", dict(good, c_root=0.1)),
                 ("span", dict(good, span=0.5 * good["c_root"])),
                 ("taper", dict(good, taper=0.1)),
                 ("sweep_deg", dict(good, sweep_deg=50.0)),
                 ("twist_deg", dict(good, twist_deg=11.0)),
                 ("re_l", dict(good, re_l=5e3)), ("nc", dict(good, nc=7))]
        no_tt = dict(good)
        del no_tt["tt"]
        cases.append(("tt", no_tt))
        cases.append(("zz", dict(good, zz=1)))
        for key, bad in cases:
            try:
                validate_params(bad)
            except ValueError as e:
                assert str(e).startswith(key + ":"), \
                    "%r does not name %s: %s" % (str(e), key, e)
                continue
            raise AssertionError("%s=%r was not refused" % (key, bad.get(key)))
        return "%d by name" % len(cases)

    groups = [("section", group_section), ("section areas", group_areas),
              ("closed", group_closed), ("volume", group_volume),
              ("sampler", group_sampler), ("determinism", group_determinism),
              ("rows", group_rows), ("refusals", group_refusals)]
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
