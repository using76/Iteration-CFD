#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""thwaites.py - the laminar boundary-layer reference of the CAD loop (docs/16 §D S9, §H.4 G2): Thwaites' integral method made axisymmetric in the Rott-Crabtree way, fed any edge velocity U(x) and wall radius r(x).

Thwaites, B. (1949) Aeronaut. Q. 1(3):245-280, DOI 10.1017/S0001925900000184 (the 0.45 quadrature and lambda).
Rott, N. & Crabtree, L. F. (1952) J. Aeronaut. Sci. 19:553-565, DOI 10.2514/8.2381 (the r^2 axisymmetric form).
H(lambda), l(lambda): the Cebeci & Bradshaw (1977) fits as restated in White, Viscous Fluid Flow, sec. 4-6.
dU/dx: Savitzky & Golay (1964), DOI 10.1021/ac60214a047, as a local least-squares fit in x with a fixed window.
The formulas are reimplemented from these citations; no code copied.

Usage:
  python thwaites.py --selftest
  python thwaites.py solve IN_JSON OUT_JSON
"""

import json
import math
import os
import subprocess
import sys
import tempfile
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common

METHOD = "thwaites-rott-crabtree/1"
THWAITES_A = 0.45          # theta^2 r^2 U^6 = A nu integral(r^2 U^5 dx)  (Thwaites 1949)
THWAITES_B = 6.0           # the exponent of U; the integrand carries U^(B-1)
LAM_MIN = -0.09            # docs/16 §I: lambda clamped to [-0.09, 0.25]
LAM_MAX = 0.25
CF_SEP = 0.0005            # separation flag: Cf <= CF_SEP
SG_WINDOW = 11             # fixed Savitzky-Golay window, in stations
SG_ORDER = 3
IDS = ("THW-SHAPE", "THW-FINITE", "THW-ORDER", "THW-NU", "THW-U", "THW-R", "THW-THETA0", "THW-INPUT")
IN_KEYS = ("x", "U", "nu", "r", "theta0")     # r and theta0 optional in IN_JSON
USAGE = ("usage: python thwaites.py --selftest" + chr(10)
         + "       python thwaites.py solve IN_JSON OUT_JSON")


def fits(lam):
    """The Cebeci & Bradshaw (1977) H(lambda), l(lambda) fits, as restated in White sec. 4-6.

    No clamping here: solve clamps lambda before calling. At 0.25: H 2.000, l 0.500
    (Thwaites' own table); at -0.09 l < 0, so a clamped adverse station carries Cf < 0.
    """
    lam = float(lam)
    if lam >= 0.0:
        H = 2.61 - 3.75 * lam + 5.24 * lam ** 2
        lf = 0.22 + 1.57 * lam - 1.8 * lam ** 2
    else:
        H = 2.088 + 0.0731 / (lam + 0.14)
        lf = 0.22 + 1.402 * lam + 0.018 * lam / (lam + 0.107)
    return H, lf


def sg_derivative(x, U):
    """Savitzky-Golay first derivative generalised to a nonuniform grid (docs/16 §I CAD-12).

    A fixed SG_WINDOW-station, SG_ORDER-degree least-squares polynomial fit in the scaled
    coordinate t = (x - x_i)/scale at every station: centred in the interior, shifted
    inward at the ends. On a uniform grid this equals scipy savgol_filter deriv=1 mode interp.
    """
    x = np.asarray(x, dtype=float)
    U = np.asarray(U, dtype=float)
    n = x.shape[0]
    h = SG_WINDOW // 2
    d = np.zeros(n)
    for i in range(n):
        a = min(max(i - h, 0), n - SG_WINDOW)
        s = slice(a, a + SG_WINDOW)
        scale = x[a + SG_WINDOW - 1] - x[a]
        t = (x[s] - x[i]) / scale
        c = np.polynomial.polynomial.polyfit(t, U[s], SG_ORDER)
        d[i] = c[1] / scale
    return d


def solve(x, U, nu, r=None, theta0=0.0):
    """The axisymmetric Thwaites / Rott-Crabtree solution on given stations (docs/16 §D S9).

    Validates in a fixed order and refuses bad input with one THW-* id each, then
    integrates theta^2 r^2 U^B = theta0^2 r0^2 U0^B + A nu integral(r^2 U^(B-1) dx)
    by a cumulative trapezoid, clamps lambda, and returns theta, delta*, Cf and a
    separation flag. Reads THWAITES_A, THWAITES_B, LAM_MIN, LAM_MAX at call time.
    """
    xa = np.asarray(x)
    ua = np.asarray(U)
    if xa.ndim != 1 or ua.ndim != 1 or xa.shape[0] != ua.shape[0]:
        raise ValueError("THW-SHAPE: x and U must be 1-D of equal length")
    n = int(xa.shape[0])
    ra = None
    if r is not None:
        ra = np.asarray(r)
        if ra.ndim != 1 or ra.shape[0] != n:
            raise ValueError("THW-SHAPE: r must be 1-D of the same length as x")
    if n < SG_WINDOW:
        raise ValueError("THW-SHAPE: need at least %d stations, got %d" % (SG_WINDOW, n))
    xa = np.asarray(x, dtype=float)
    ua = np.asarray(U, dtype=float)
    if ra is not None:
        ra = np.asarray(ra, dtype=float)
    for name, arr in (("x", xa), ("U", ua), ("r", ra)):
        if arr is not None and not bool(np.all(np.isfinite(arr))):
            raise ValueError("THW-FINITE: %s carries a non-finite entry" % name)
    if isinstance(nu, bool) or not isinstance(nu, (int, float)) or not math.isfinite(nu):
        raise ValueError("THW-FINITE: nu must be a finite real number")
    if isinstance(theta0, bool) or not isinstance(theta0, (int, float)) or not math.isfinite(theta0):
        raise ValueError("THW-FINITE: theta0 must be a finite real number")
    if np.any(np.diff(xa) <= 0.0):
        raise ValueError("THW-ORDER: x must be strictly increasing")
    if not nu > 0:
        raise ValueError("THW-NU: nu must be > 0")
    if np.any(ua < 0.0) or np.any(ua[1:] <= 0.0):
        raise ValueError("THW-U: U must be >= 0 everywhere and > 0 past the first station")
    if ra is not None and (np.any(ra < 0.0) or np.any(ra[1:] <= 0.0)):
        raise ValueError("THW-R: r must be >= 0 everywhere and > 0 past the first station")
    singular = bool(ua[0] == 0.0 or (ra is not None and ra[0] == 0.0))
    if theta0 < 0 or (singular and theta0 != 0):
        raise ValueError("THW-THETA0: theta0 must be >= 0, and 0 at a singular start")
    nu = float(nu)
    theta0 = float(theta0)
    planar = ra is None
    r_used = np.ones(n) if planar else ra
    with np.errstate(divide="ignore", invalid="ignore"):
        g = r_used ** 2 * ua ** (THWAITES_B - 1.0)
        I = np.zeros(n)
        I[1:] = np.cumsum(np.diff(xa) * (g[1:] + g[:-1]) * 0.5)
        num0 = theta0 ** 2 * r_used[0] ** 2 * ua[0] ** THWAITES_B
        theta = np.sqrt((num0 + THWAITES_A * nu * I) / (r_used ** 2 * ua ** THWAITES_B))
        dUdx = sg_derivative(xa, ua)
        lam_raw = theta ** 2 * dUdx / nu
        lam = np.minimum(np.maximum(lam_raw, LAM_MIN), LAM_MAX)
        clamped = (lam_raw < LAM_MIN) | (lam_raw > LAM_MAX)
        H = np.zeros(n)
        lf = np.zeros(n)
        for i in range(n):
            H[i], lf[i] = fits(lam[i])
        dstar = H * theta
        Re_theta = ua * theta / nu
        Cf = 2.0 * nu * lf / (ua * theta)
        Cf = np.where(theta == 0.0, np.nan, Cf)
        if singular:
            for arr in (theta, dstar, H, lf, Cf, Re_theta, lam, lam_raw):
                arr[0] = np.nan
            clamped[0] = False
        sep = np.less_equal(Cf, CF_SEP)
    separated = bool(np.any(sep))
    x_sep = float(xa[int(np.argmax(sep))]) if separated else None
    return {"method": METHOD, "planar": planar, "singular_start": singular,
            "x": xa, "U": ua, "r": r_used, "dUdx": dUdx, "theta": theta, "dstar": dstar,
            "H": H, "l": lf, "Cf": Cf, "Re_theta": Re_theta, "lam": lam, "lam_raw": lam_raw,
            "clamped": np.asarray(clamped, dtype=bool), "sep": np.asarray(sep, dtype=bool),
            "separated": separated, "x_sep": x_sep,
            "constants": {"A": THWAITES_A, "B": THWAITES_B, "lam_min": LAM_MIN, "lam_max": LAM_MAX,
                          "cf_sep": CF_SEP, "sg_window": SG_WINDOW, "sg_order": SG_ORDER}}


def to_json(res):
    """Make a solve result pass common.canonical_json: NaN and +-inf become null."""
    out = {}
    for key, val in res.items():
        if isinstance(val, np.ndarray):
            if val.dtype == np.bool_:
                out[key] = [bool(v) for v in val]
            else:
                out[key] = [float(v) if np.isfinite(v) else None for v in val]
        elif isinstance(val, (np.floating, np.integer, np.bool_)):
            out[key] = val.item()
        else:
            out[key] = val
    return out


def solve_doc(doc):
    """Run solve on an IN_JSON document: keys a subset of IN_KEYS holding x, U and nu."""
    if not isinstance(doc, dict):
        raise ValueError("THW-INPUT: the input must be a JSON object with keys x, U and nu")
    if not set(doc.keys()) <= set(IN_KEYS):
        raise ValueError("THW-INPUT: unknown keys %s; allowed are %s"
                         % (sorted(str(k) for k in set(doc.keys()) - set(IN_KEYS)), list(IN_KEYS)))
    for need in ("x", "U", "nu"):
        if need not in doc:
            raise ValueError("THW-INPUT: missing required key %s" % need)
    return to_json(solve(doc["x"], doc["U"], doc["nu"], doc.get("r"), doc.get("theta0", 0.0)))


def nominal_case(Re_De, nu=1.5e-5, De=0.02, CR=9.0, L_over_Di=1.0, Lx_over_De=0.5, n=6000):
    """The plan's nominal contraction (docs/16 §H.2): Bell & Mehta poly5, area rule, no feedback."""
    Re_ = De / 2; Ri = Re_ * math.sqrt(CR); Di = 2 * Ri; Lc = L_over_Di * Di; Lx = Lx_over_De * De
    Ue = Re_De * nu / De; Q = Ue * math.pi * Re_ ** 2
    x = np.linspace(0.0, Lc + Lx, n)
    xi = np.clip(x / Lc, 0.0, 1.0)
    r = Ri - (Ri - Re_) * (10 * xi ** 3 - 15 * xi ** 4 + 6 * xi ** 5)      # Bell & Mehta poly5
    U = Q / (math.pi * r ** 2)                                           # 1-D area rule, no feedback
    ie = int(np.argmin(np.abs(x - Lc)))                                  # the exit plane station
    return x, U, r, ie, Re_


def main(argv):
    """The CLI of docs/16 §I CAD-12: --selftest, solve IN OUT."""
    if argv == ["--selftest"]:
        try:
            selftest()
        except Exception:
            traceback.print_exc()
            return 1
        return 0
    try:
        if len(argv) == 3 and argv[0] == "solve":
            doc = common.read_json(argv[1])
            out = solve_doc(doc)
            common.atomic_write(argv[2], common.canonical_json(out) + chr(10))
            print(common.canonical_json({"separated": out["separated"], "x_sep": out["x_sep"],
                                         "n": len(out["x"])}))
            return 0
    except (ValueError, OSError) as e:
        print("thwaites: %s" % (e,), file=sys.stderr)
        return 2
    print(USAGE, file=sys.stderr)
    return 2


NU0 = 1.5e-5


def close(a, b, rtol):
    """abs(a-b) <= rtol|b| + 1e-15 elementwise over the finite entries, NaN positions equal."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    fa = np.isfinite(a)
    fb = np.isfinite(b)
    if not np.array_equal(fa, fb):
        return False
    return bool(np.all(np.abs(a[fa] - b[fa]) <= rtol * np.abs(b[fa]) + 1e-15))


def _flat_plate_err():
    x = np.linspace(0.0, 1.0, 2001)
    res = solve(x, np.full(x.shape, 3.0), NU0)
    err = np.abs(res["theta"][1:] * np.sqrt(3.0 / (NU0 * x[1:])) - math.sqrt(0.45))
    return float(np.max(err))


def _pred_t2():
    return _flat_plate_err() <= 1e-4


def _hiemenz_input():
    x = np.linspace(0.0, 1.0, 4001)
    return x, 2.0 * x


def _homann_err(rarr):
    x, U = _hiemenz_input()
    res = solve(x, U, NU0, rarr)
    m = x >= 0.1
    return float(np.max(np.abs(res["lam"][m] - 0.05625)))


def _pred_t4():
    x, _ = _hiemenz_input()
    return _homann_err(x) <= 1e-4


def _clamp_pred():
    x = np.linspace(0.0, 1.0, 101)
    res = solve(x, 1.0 + x, NU0, theta0=0.01)
    return bool(res["lam_raw"][0] > LAM_MAX and res["lam"][0] == LAM_MAX
                and bool(res["clamped"][0]))


def _howarth():
    L = 1.0
    x = np.linspace(0.0, 0.2, 4001)
    U0 = 1e5 * NU0 / L
    return x, solve(x, U0 * (1.0 - x / L), NU0)


def _nominal_solve(Re_De):
    x, U, r, ie, Re_ = nominal_case(Re_De)
    return solve(x, U, NU0, r), ie, Re_


def _t1():
    H0, l0 = fits(0.0)
    assert abs(H0 - 2.61) <= 1e-12 and abs(l0 - 0.22) <= 1e-12
    Hq, lq = fits(0.25)
    assert abs(Hq - 2.0) <= 1e-12 and abs(lq - 0.5) <= 1e-12
    Hm, lm = fits(-1e-12)
    assert abs(Hm - 2.61) < 2e-4 and abs(lm - 0.22) < 1e-9
    assert fits(-0.09)[1] < 0.0
    print("[ok] fits: H 2.61 l 0.22 at 0, Thwaites' 2.00 and 0.500 at 0.25, continuous at 0, l < 0 at -0.09")


def _t2():
    err = _flat_plate_err()
    assert err <= 1e-4
    x = np.linspace(0.0, 1.0, 2001)
    res = solve(x, np.full(x.shape, 3.0), NU0)
    assert not res["singular_start"]
    assert np.isnan(res["Cf"][0])
    # dU/dx of a constant is zero to polyfit roundoff only (measured 4.3e-13 here),
    # so lam vanishes to roundoff, not bitwise (deviation from the brief noted in the report)
    assert float(np.max(np.abs(res["lam"]))) <= 1e-12
    print("[ok] flat plate: theta sqrt(U/(nu x)) = sqrt(0.45) within %.1e over 2000 stations" % err)


def _t3():
    x, U = _hiemenz_input()
    res = solve(x, U, NU0)
    assert res["singular_start"]
    for key in ("theta", "lam", "Cf"):
        assert np.isnan(res[key][0])
    assert not bool(res["sep"][0])
    m = x >= 0.1
    err = float(np.max(np.abs(res["lam"][m] - 0.075)))
    assert err <= 1e-4
    print("[ok] Hiemenz: lambda = 0.075 within %.1e for x >= 0.1; the stagnation station is NaN" % err)


def _t4():
    x, _ = _hiemenz_input()
    err = _homann_err(x)
    assert err <= 1e-4
    print("[ok] Homann: lambda = 0.05625 within %.1e with r = x, so r enters squared" % err)


def _t5():
    x, U, r, ie, Re_ = nominal_case(3e4)
    resp = solve(x, U, NU0)
    resc = solve(x, U, NU0, np.full(x.shape, 0.03))
    for key in ("theta", "dstar", "Cf", "lam"):
        assert close(resp[key], resc[key], 1e-12), key
    refr = solve(x, U, NU0, r)
    res7 = solve(x, U, NU0, 7.0 * r)
    for key in ("theta", "dstar", "Cf", "lam"):
        assert close(refr[key], res7[key], 1e-12), key
    print("[ok] constant radius reduces to planar and r scale drops out, both to 1e-12")


def _t6():
    x, U, r, ie, Re_ = nominal_case(3e4)
    full = solve(x, U, NU0, r)
    k = 3000
    a = solve(x[:k + 1], U[:k + 1], NU0, r[:k + 1])
    b = solve(x[k:], U[k:], NU0, r[k:], theta0=float(a["theta"][k]))
    joined = np.concatenate([a["theta"][1:], b["theta"][1:]])
    assert close(joined, full["theta"][1:], 1e-12)
    print("[ok] restart from theta0 at station 3000 reproduces the one-pass theta")


def _t7():
    rng = np.random.default_rng(1)
    xs = np.sort(rng.uniform(0, 1, 300))
    U = 1 + 2 * xs - 3 * xs ** 2 + 0.7 * xs ** 3
    ea = float(np.max(np.abs(sg_derivative(xs, U) - (2 - 6 * xs + 2.1 * xs ** 2))))
    assert ea <= 1e-9, ea
    import scipy.signal
    xu = np.linspace(0, 2, 200)
    Uu = np.sin(3 * xu) + 0.1 * np.cos(17 * xu)
    ref = scipy.signal.savgol_filter(Uu, SG_WINDOW, SG_ORDER, deriv=1,
                                     delta=xu[1] - xu[0], mode="interp")
    eb = float(np.max(np.abs(sg_derivative(xu, Uu) - ref)))
    assert eb <= 1e-9, eb
    xw = np.linspace(0, 1, 201)
    U0 = 1.0 + xw
    U1 = U0.copy()
    U1[100] += 0.01
    # the centre station sees no change: in a centred odd window its derivative weight is
    # exactly zero (w_k proportional to t_k, t_centre = 0), so the set is 95..105 minus 100
    idx = np.where(np.abs(sg_derivative(xw, U1) - sg_derivative(xw, U0)) > 1e-14)[0]
    assert list(idx) == [i for i in range(95, 106) if i != 100], idx
    print("[ok] Savitzky-Golay: exact for a cubic on 300 random stations, equal to scipy on a uniform grid, local to 11 stations")


def _t8():
    assert _clamp_pred()
    xh, resh = _howarth()
    low = resh["lam_raw"] < LAM_MIN
    cnt = int(np.sum(low))
    assert cnt >= 1
    assert bool(np.all(resh["lam"][low] == LAM_MIN))
    assert bool(np.all(resh["clamped"][low]))
    print("[ok] lambda clamped to [-0.09, 0.25] with the flag set; %d Howarth stations clamped at -0.09" % cnt)


def _t9():
    xh, resh = _howarth()
    raw = resh["lam_raw"]
    hit = np.where(raw <= LAM_MIN)[0]
    assert hit.size >= 1
    k = int(hit[0])
    xk = float(xh[k])
    assert 0.120 <= xk <= 0.126, xk
    assert resh["separated"]
    xsep = float(resh["x_sep"])
    assert 0.110 <= xsep <= xk, xsep
    assert not bool(np.any(resh["sep"][xh < 0.10]))
    print("[ok] Howarth: lambda reaches -0.09 at x/L %.4f, the Cf flag first at %.4f; Thwaites gives 0.123"
          % (xk, xsep))


def _t10():
    res, ie, Re_ = _nominal_solve(3e4)
    v = float(res["theta"][ie] / Re_)
    assert abs(v / 0.0059 - 1.0) <= 0.03, v
    print("[ok] nominal Re_De 3e4: theta_e/R_e %.5f vs plan 0.0059 (%+.2f %%)"
          % (v, (v / 0.0059 - 1.0) * 100.0))


def _t11():
    res, ie, Re_ = _nominal_solve(1e4)
    v = float(res["theta"][ie] / Re_)
    assert abs(v / 0.0102 - 1.0) <= 0.03, v
    print("[ok] nominal Re_De 1e4: theta_e/R_e %.5f vs plan 0.0102 (%+.2f %%)"
          % (v, (v / 0.0102 - 1.0) * 100.0))


def _t12():
    r3, ie3, _ = _nominal_solve(3e4)
    r1, ie1, _ = _nominal_solve(1e4)
    slope_d = math.log(float(r3["dstar"][ie3]) / float(r1["dstar"][ie1])) / math.log(3.0)
    slope_t = math.log(float(r3["theta"][ie3]) / float(r1["theta"][ie1])) / math.log(3.0)
    assert abs(slope_d + 0.503) <= 0.01, slope_d
    assert abs(slope_t + 0.5) <= 0.01, slope_t
    assert not r3["separated"] and not r1["separated"]
    print("[ok] slope d ln dstar_e / d ln Re_De %.4f (plan -0.503 +- 0.01), theta_e %.4f; the nominal never separates"
          % (slope_d, slope_t))


def _expect(fn, thw_id):
    try:
        fn()
    except ValueError as e:
        assert str(e).startswith(thw_id), "%s: %s" % (thw_id, e)
        return
    raise AssertionError("no ValueError raised for %s" % thw_id)


def _t13():
    xb = np.linspace(0.0, 1.0, 2001)
    Ub = np.full(xb.shape, 3.0)
    xh, Uh = _hiemenz_input()
    r9 = np.full(xb.shape, 0.03)
    r9[3] = 0.0
    u = Ub.copy(); u[7] = np.nan
    xw = xb.copy(); xw[5], xw[6] = xw[6], xw[5]
    u5 = Ub.copy(); u5[5] = 0.0
    um = Ub.copy(); um[0] = -1.0
    _expect(lambda: solve(xb, Ub[:-1], NU0), "THW-SHAPE")
    _expect(lambda: solve(xb[:10], Ub[:10], NU0), "THW-SHAPE")
    _expect(lambda: solve(xb, u, NU0), "THW-FINITE")
    _expect(lambda: solve(xb, Ub, float("inf")), "THW-FINITE")
    _expect(lambda: solve(xw, Ub, NU0), "THW-ORDER")
    _expect(lambda: solve(xb, Ub, 0.0), "THW-NU")
    _expect(lambda: solve(xb, u5, NU0), "THW-U")
    _expect(lambda: solve(xb, um, NU0), "THW-U")
    _expect(lambda: solve(xb, Ub, NU0, r9), "THW-R")
    _expect(lambda: solve(xb, Ub, NU0, None, -1e-6), "THW-THETA0")
    _expect(lambda: solve(xh, Uh, NU0, None, 1e-4), "THW-THETA0")
    _expect(lambda: solve_doc({"x": xb.tolist(), "U": Ub.tolist(), "nu": NU0, "extra": 1}), "THW-INPUT")
    _expect(lambda: solve_doc({"x": xb.tolist(), "U": Ub.tolist()}), "THW-INPUT")
    print("[ok] 13 bad inputs refused by the 8 THW ids, each by exactly its id")


def _t14():
    global THWAITES_A, LAM_MIN, LAM_MAX
    saved = (THWAITES_A, LAM_MIN, LAM_MAX)
    ok = []
    try:
        THWAITES_A = 0.44
        ok.append(not _pred_t2())
        THWAITES_A = saved[0]
        xh, _ = _hiemenz_input()
        ok.append(not (_homann_err(np.sqrt(xh)) <= 1e-4))
        LAM_MIN = float("-inf")
        LAM_MAX = float("inf")
        ok.append(not _clamp_pred())
    finally:
        THWAITES_A, LAM_MIN, LAM_MAX = saved
    assert len(ok) == 3 and all(ok), ok
    assert _pred_t2() and _pred_t4() and _clamp_pred()
    print("[ok] 3 of 3 planted mutants (A 0.44, r to the first power, no clamp) fail their gate")


def _t15():
    with tempfile.TemporaryDirectory() as td:
        x, U = _hiemenz_input()
        doc = {"x": [float(v) for v in x], "U": [float(v) for v in U], "nu": NU0, "r": None}
        pin = os.path.join(td, "in.json")
        common.atomic_write(pin, common.canonical_json(doc) + chr(10))
        pout = os.path.join(td, "out.json")
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        pout2 = os.path.join(td, "out2.json")
        cmd = [sys.executable, os.path.abspath(__file__), "solve", pin]
        r1 = subprocess.run(cmd + [pout], capture_output=True, text=True, encoding="utf-8",
                            env=env, timeout=120)
        r2 = subprocess.run(cmd + [pout2], capture_output=True, text=True, encoding="utf-8",
                            env=env, timeout=120)
        assert r1.returncode == 0 and r2.returncode == 0, (r1.returncode, r1.stderr)
        with open(pout, "rb") as f:
            b1 = f.read()
        with open(pout2, "rb") as f:
            b2 = f.read()
        assert b1 == b2
        expect = (common.canonical_json(to_json(solve(x, U, NU0))) + chr(10)).encode("utf-8")
        assert b1 == expect
        got = common.read_json(pout)
        assert got["theta"][0] is None
        xs = [float(v) for v in x]
        xs[5], xs[6] = xs[6], xs[5]
        pbad = os.path.join(td, "bad.json")
        pbadout = os.path.join(td, "bad_out.json")
        common.atomic_write(pbad, common.canonical_json(
            {"x": xs, "U": [float(v) for v in U], "nu": NU0}) + chr(10))
        rb = subprocess.run([sys.executable, os.path.abspath(__file__), "solve", pbad, pbadout],
                            capture_output=True, text=True, encoding="utf-8", env=env, timeout=120)
        assert rb.returncode == 2, rb.returncode
        assert not os.path.exists(pbadout)
        assert rb.stderr.startswith("thwaites: THW-ORDER"), rb.stderr
        ru = subprocess.run([sys.executable, os.path.abspath(__file__), "solve", pbad],
                            capture_output=True, text=True, encoding="utf-8", env=env, timeout=120)
        assert ru.returncode == 2, ru.returncode
    assert "cadquery" not in sys.modules and "OCP" not in sys.modules and "gmsh" not in sys.modules
    print("[ok] CLI: two runs byte-identical and equal to solve(); NaN written as null; exit 2 on THW-ORDER with nothing written, 2 on usage")


def selftest():
    for t in (_t1, _t2, _t3, _t4, _t5, _t6, _t7, _t8, _t9,
              _t10, _t11, _t12, _t13, _t14, _t15):
        t()
    print("SELFTEST PASS")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
