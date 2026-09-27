#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""turb_integral.py - the turbulent references of the CAD loop (docs/16 §H.5, gate GC-8): the axisymmetric 1/7-power momentum integral, the smooth-pipe laws, the acceleration parameters K and p, the 1-D area-rule K_max, and Head's entrainment method as a report-only comparison.

Schlichting, NACA TM 1218 (1949), https://ntrs.nasa.gov/citations/20050040758: eqs. 16.4 (Blasius), 16.14 (log law),
16.22 (core defect), 16.26 (Prandtl's universal law), 17.3-17.11 (the 1/7-power plate method; the 0.074 plate law).
Nikuradse, NASA TT F-10,359 (1966), https://archive.org/details/nasa_techdoc_19670004508: the smooth-pipe measurements
behind 16.14, 16.22 and 16.26.
Rott & Crabtree 1952, DOI 10.2514/8.2381: the axisymmetric form (the wall radius r in the momentum integral).
Head, ARC R&M 3152 (1958), https://reports.aerade.cranfield.ac.uk/handle/1826.2/3720: the entrainment equation and its
H1(H) and F(H1) curves (Figs. 2 and 1), digitised from the scan (page 13, 400 dpi, gridline-calibrated trace).
Ludwieg & Tillmann, NACA TM 1285 (1950), https://ntrs.nasa.gov/citations/19930093945: the skin friction used with Head.
Cebeci & Bradshaw (1977) curve fits of Head's two relations: cited, used only as a comparison with the digitised curves.
Kline, Reynolds, Schraub & Runstadler 1967, DOI 10.1017/S0022112067001740 (cited), restated in Prakash, Balin, Evans &
Jansen, arXiv:2306.05972 §3.1: the relaminarisation threshold K 3e-6 and the laminarescent p -0.005.
The formulas are reimplemented from these citations; no code copied.

Usage:
  python turb_integral.py --selftest
  python turb_integral.py solve IN_JSON OUT_JSON
  python turb_integral.py kmax LAW CR L_OVER_DI RE_DE [X_M]
"""

import ast
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
import thwaites

METHOD = "power17-axisymmetric/1"
HEAD_METHOD = "head-entrainment/1"
PLATE_THETA = 0.037        # theta = 0.037 x Re_x^(-1/5): C_F = 0.074 Re_x^(-1/5) (TM 1218 eq. 17.10, 5e5 < Re_x < 1e7)
H17 = 9.0 / 7.0            # the 1/7 profile: delta* = delta/8, theta = 7 delta/72 (TM 1218 eqs. 17.3-17.5)
DELTA_OVER_THETA = 72.0 / 7.0
EXP17 = 1.25               # Theta = theta^(5/4), from the Re_theta^(-1/4) wall law
C17 = PLATE_THETA ** EXP17 / EXP17         # tau_w/(rho U^2) = C17 Re_theta^(-1/4); plate limit = PLATE_THETA exactly
C17_PIPE = 0.0225 * (7.0 / 72.0) ** 0.25   # the same law straight from TM 1218 eq. 17.7 (its plate C_F is 0.072)
BLASIUS_A = 0.3164         # f = 0.3164 Re_D^(-1/4) (TM 1218 eq. 16.4)
BLASIUS_RE_MAX = 1.0e5     # its stated range
PN_A = 2.0                 # 1/sqrt(f) = 2.0 log10(Re sqrt(f)) - 0.8 (TM 1218 eq. 16.26, Nikuradse)
PN_B = -0.8
LOG_A = 2.5                # u+ = 2.5 ln y+ + 5.5 (TM 1218 eq. 16.14; kappa 0.4)
LOG_B = 5.5
CORE_DEFECT = 4.07         # (U_axis - U_b)/u_tau (TM 1218 eq. 16.22, Nikuradse)
K_RELAM = 3.0e-6           # relaminarisation onset (Kline et al. 1967; arXiv:2306.05972 §3.1)
P_LAMINARESCENT = -0.005   # laminarescent onset of p (arXiv:2306.05972 §3.1)
LT_A = 0.246               # Cf = 0.246 10^(-0.678 H) Re_theta^(-0.268) (Ludwieg & Tillmann, NACA TM 1285)
LT_B = -0.678
LT_C = -0.268
LAWS = ("poly3", "poly5", "poly7", "cubic_matched")
POLY = {"poly3": (0.0, 0.0, 3.0, -2.0),
        "poly5": (0.0, 0.0, 0.0, 10.0, -15.0, 6.0),
        "poly7": (0.0, 0.0, 0.0, 0.0, 35.0, -84.0, 70.0, -20.0)}   # f(xi) monomials, r = R_i - (R_i - R_e) f
X_M_BOX = (0.2, 0.8)       # the template's x_m box (docs/16 §H.1)

# The two tables below were digitised by the supervisor on 2026-09-26 from the Cranfield scan of R&M 3152 (sha256
# 5317c6fc6fe4b382b345eb4fb65521cfb0b582876c14a89a2c8b1fc2714c1eb0), page 13, rendered at 400 dpi: gridlines found by
# long-run detection, the drawn curve traced column- and row-wise, calibrated piecewise-linearly between gridlines,
# sampled at these abscissae. One pixel is 0.0025 in H and 0.0127 in H1 (Fig. 2), 0.0126 in H1 and 6.4e-5 in F (Fig. 1).
HEAD_G_TABLE = ((1.28, 9.826), (1.30, 9.292), (1.33, 8.587), (1.36, 7.982), (1.40, 7.280), (1.45, 6.585),
                (1.50, 6.067), (1.55, 5.652), (1.60, 5.324), (1.70, 4.829), (1.80, 4.484), (1.90, 4.228),
                (2.00, 4.061), (2.20, 3.789), (2.40, 3.668), (2.60, 3.581), (2.80, 3.527), (2.95, 3.509))   # (H, H1), Fig. 2
HEAD_F_TABLE = ((3.40, 0.06162), (3.50, 0.04869), (3.60, 0.04170), (3.80, 0.03405), (4.00, 0.02951),
                (4.50, 0.02297), (5.00, 0.01904), (5.50, 0.01663), (6.00, 0.01478), (7.00, 0.01259),
                (8.00, 0.01060), (9.00, 0.00966), (10.00, 0.00888), (11.00, 0.00804), (11.80, 0.00753))  # (H1, F), Fig. 1

IDS = ("TI-SHAPE", "TI-FINITE", "TI-ORDER", "TI-NU", "TI-U", "TI-R", "TI-THETA0", "TI-INPUT",
       "TI-LAW", "TI-GEOM", "TI-RE", "TI-YPLUS", "TI-H0", "TI-CURVES")
IN_KEYS = ("x", "U", "nu", "r", "theta0")     # r and theta0 optional in IN_JSON
USAGE = ("usage: python turb_integral.py --selftest" + chr(10)
         + "       python turb_integral.py solve IN_JSON OUT_JSON" + chr(10)
         + "       python turb_integral.py kmax LAW CR L_OVER_DI RE_DE [X_M]")
P = np.polynomial.polynomial


def _stations(x, U, nu, r, theta0):
    """Validate the stations in the fixed order of docs/16 §I CAD-24, one TI-* id each."""
    xa = np.asarray(x)
    ua = np.asarray(U)
    if xa.ndim != 1 or ua.ndim != 1 or xa.shape[0] != ua.shape[0]:
        raise ValueError("TI-SHAPE: x and U must be 1-D of equal length")
    n = int(xa.shape[0])
    ra = None
    if r is not None:
        ra = np.asarray(r)
        if ra.ndim != 1 or ra.shape[0] != n:
            raise ValueError("TI-SHAPE: r must be 1-D of the same length as x")
    if n < thwaites.SG_WINDOW:
        raise ValueError("TI-SHAPE: need at least %d stations, got %d" % (thwaites.SG_WINDOW, n))
    xa = np.asarray(x, dtype=float)
    ua = np.asarray(U, dtype=float)
    if ra is not None:
        ra = np.asarray(ra, dtype=float)
    for name, arr in (("x", xa), ("U", ua), ("r", ra)):
        if arr is not None and not bool(np.all(np.isfinite(arr))):
            raise ValueError("TI-FINITE: %s carries a non-finite entry" % name)
    if isinstance(nu, bool) or not isinstance(nu, (int, float)) or not math.isfinite(nu):
        raise ValueError("TI-FINITE: nu must be a finite real number")
    if isinstance(theta0, bool) or not isinstance(theta0, (int, float)) or not math.isfinite(theta0):
        raise ValueError("TI-FINITE: theta0 must be a finite real number")
    if np.any(np.diff(xa) <= 0.0):
        raise ValueError("TI-ORDER: x must be strictly increasing")
    if not nu > 0:
        raise ValueError("TI-NU: nu must be > 0")
    if np.any(ua <= 0.0):
        raise ValueError("TI-U: every U must be > 0 (a turbulent layer has no stagnation start here)")
    if ra is not None and np.any(ra <= 0.0):
        raise ValueError("TI-R: every r must be > 0")
    if theta0 < 0.0:
        raise ValueError("TI-THETA0: theta0 must be >= 0")
    return xa, ua, ra, float(nu), float(theta0)


def solve(x, U, nu, r=None, theta0=0.0):
    """The axisymmetric 1/7-power momentum integral on given stations (docs/16 lines 524-528).

    d(theta^(5/4))/dx + (5/4) theta^(5/4) [(2 + H) U'/U + r'/r] = (5/4) c (nu/U)^(1/4) with H = 9/7,
    one quadrature with the integrating factor (U^(2+H) r)^(5/4), normalised at the LAST station so a
    constant radius is planar bit for bit. c is set so the plate limit is theta = 0.037 x Re_x^(-1/5).
    Reads PLATE_THETA, H17, EXP17, DELTA_OVER_THETA at call time.
    """
    xa, ua, ra, nu, theta0 = _stations(x, U, nu, r, theta0)
    n = len(xa)
    planar = ra is None
    ru = np.ones(n) if planar else ra
    e = EXP17
    Hs = H17
    c = PLATE_THETA ** e / e
    F = ((ua / ua[-1]) ** (2.0 + Hs) * (ru / ru[-1])) ** e
    g = F * (nu / ua) ** 0.25
    I = np.zeros(n)
    I[1:] = np.cumsum(np.diff(xa) * (g[1:] + g[:-1]) * 0.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        Theta = (theta0 ** e * F[0] + e * c * I) / F
        theta = Theta ** (1.0 / e)
        H = np.full(n, Hs)
        dstar = Hs * theta
        delta = DELTA_OVER_THETA * theta
        Re_theta = ua * theta / nu
        pos = theta > 0.0
        Cf = np.where(pos, 2.0 * c * Re_theta ** -0.25, np.nan)
        u_tau = np.where(pos, ua * np.sqrt(c) * Re_theta ** -0.125, np.nan)
        dUdx = thwaites.sg_derivative(xa, ua)
        K = nu * dUdx / ua ** 2
        p = -nu * ua * dUdx / u_tau ** 3
    return {"method": METHOD, "planar": planar, "x": xa, "U": ua, "r": ru, "dUdx": dUdx,
            "theta": theta, "dstar": dstar, "delta": delta, "H": H, "Re_theta": Re_theta,
            "Cf": Cf, "u_tau": u_tau, "K": K, "p": p,
            "constants": {"plate_theta": PLATE_THETA, "H": H17, "exp": EXP17, "c": c, "c_pipe": C17_PIPE,
                          "k_relam": K_RELAM, "p_laminarescent": P_LAMINARESCENT,
                          "sg_window": thwaites.SG_WINDOW, "sg_order": thwaites.SG_ORDER}}


def solve_doc(doc):
    """Run solve on an IN_JSON document: keys a subset of IN_KEYS holding x, U and nu."""
    if not isinstance(doc, dict):
        raise ValueError("TI-INPUT: the input must be a JSON object with keys x, U and nu")
    if not set(doc.keys()) <= set(IN_KEYS):
        raise ValueError("TI-INPUT: unknown keys %s; allowed are %s"
                         % (sorted(str(k) for k in set(doc.keys()) - set(IN_KEYS)), list(IN_KEYS)))
    for need in ("x", "U", "nu"):
        if need not in doc:
            raise ValueError("TI-INPUT: missing required key %s" % need)
    return thwaites.to_json(solve(doc["x"], doc["U"], doc["nu"], doc.get("r"), doc.get("theta0", 0.0)))


def _pos_real(val, msg):
    """Refuse a bool or anything that is not a finite real number > 0, with the given TI id."""
    if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or not val > 0:
        raise ValueError(msg)
    return float(val)


def f_blasius(re_d):
    """f = 0.3164 Re_D^(-1/4) (TM 1218 eq. 16.4)."""
    return BLASIUS_A * _pos_real(re_d, "TI-RE: re_d must be a finite real number > 0") ** -0.25


def f_prandtl(re_tau):
    """1/sqrt(f) = 2.0 log10(Re sqrt(f)) - 0.8 (TM 1218 eq. 16.26) in closed form from Re_tau.

    Re sqrt(f) = 2 sqrt(8) Re_tau, since f = 8 (u_tau/U_b)^2 and Re sqrt(f) = (U_b D/nu) sqrt(f)
    = (U_b D/nu) sqrt(8) u_tau/U_b = 2 sqrt(8) Re_tau.
    """
    rt = _pos_real(re_tau, "TI-RE: re_tau must be a finite real number > 0")
    return (PN_A * math.log10(2.0 * math.sqrt(8.0) * rt) + PN_B) ** -2


def re_d_from_re_tau(re_tau):
    """Re_D implied by eq. 16.26 at the given Re_tau."""
    return 2.0 * _pos_real(re_tau, "TI-RE: re_tau must be a finite real number > 0") * math.sqrt(8.0 / f_prandtl(re_tau))


def u_plus_log(y_plus):
    """u+ = 2.5 ln y+ + 5.5 (TM 1218 eq. 16.14, Nikuradse's kappa 0.4, B 5.5)."""
    y = np.asarray(y_plus, dtype=float)
    if not bool(np.all(np.isfinite(y))) or np.any(y <= 0.0):
        raise ValueError("TI-YPLUS: y_plus must be finite and > 0")
    out = LOG_A * np.log(y) + LOG_B
    if np.ndim(y_plus) == 0:
        return float(out)
    return out


def u_axis_from_bulk(u_b, u_tau):
    """The core defect (TM 1218 eq. 16.22): (U_axis - U_b)/u_tau = 4.07."""
    return u_b + CORE_DEFECT * u_tau


def pipe_drive(re_tau, R, nu):
    """The TG0 body force: tau_w 2 pi R L = rho g_x pi R^2 L gives g_x = 2 u_tau^2 / R."""
    rt = _pos_real(re_tau, "TI-RE: re_tau must be a finite real number > 0")
    if isinstance(R, bool) or not isinstance(R, (int, float)) or not math.isfinite(R) or not R > 0:
        raise ValueError("TI-GEOM: R must be a finite real number > 0")
    if isinstance(nu, bool) or not isinstance(nu, (int, float)) or not math.isfinite(nu) or not nu > 0:
        raise ValueError("TI-NU: nu must be a finite real number > 0")
    u_tau = rt * float(nu) / float(R)
    return {"u_tau": u_tau, "g_x": 2.0 * u_tau ** 2 / float(R)}


def accel_K(x, U, nu):
    """K = (nu/U^2) dU/dx on the given stations (the relaminarisation parameter)."""
    xa, ua, _, nuf, _ = _stations(x, U, nu, None, 0.0)
    return nuf * thwaites.sg_derivative(xa, ua) / ua ** 2


def p_param(x, U, nu, u_tau):
    """p = (nu/(rho u_tau^3)) dp/dx = -nu U U'/u_tau^3 (Bernoulli), laminarescent onset -0.005."""
    xa, ua, _, nuf, _ = _stations(x, U, nu, None, 0.0)
    ut = np.asarray(u_tau, dtype=float)
    if ut.ndim > 1 or (ut.ndim == 1 and ut.shape[0] != len(xa)):
        raise ValueError("TI-SHAPE: u_tau must be a scalar or 1-D of the length of x")
    good = np.isfinite(ut) & (ut > 0.0)
    with np.errstate(invalid="ignore"):
        uts = np.where(good, ut, np.nan)
        dUdx = thwaites.sg_derivative(xa, ua)
        p = -nuf * ua * dUdx / uts ** 3
    return p


def law_pieces(law, x_m=None):
    """The wall law as (xi0, xi1, a) pieces: f = sum a_k s^k on s in [0, 1], xi = xi0 + (xi1 - xi0) s."""
    if law == "cubic_matched":
        xm = float(x_m)
        return [(0.0, xm, (0.0, 0.0, 0.0, xm)),
                (xm, 1.0, (xm, 3.0 * (1.0 - xm), -3.0 * (1.0 - xm), 1.0 - xm))]
    return [(0.0, 1.0, POLY[law])]


def _law_check(law, x_m):
    if law not in LAWS:
        raise ValueError("TI-LAW: unknown law %r; allowed are %s" % (law, list(LAWS)))
    if law == "cubic_matched":
        if isinstance(x_m, bool) or not isinstance(x_m, (int, float)) or not math.isfinite(x_m) \
                or not X_M_BOX[0] <= x_m <= X_M_BOX[1]:
            raise ValueError("TI-LAW: cubic_matched needs a real finite x_m in %s" % (X_M_BOX,))
    elif x_m is not None:
        raise ValueError("TI-LAW: x_m must be None for the polynomial laws")


def _geom_cr(CR, L_over_Di):
    if isinstance(CR, bool) or not isinstance(CR, (int, float)) or not math.isfinite(CR) or not CR > 1.0:
        raise ValueError("TI-GEOM: CR must be a finite real number > 1")
    if isinstance(L_over_Di, bool) or not isinstance(L_over_Di, (int, float)) \
            or not math.isfinite(L_over_Di) or not L_over_Di > 0:
        raise ValueError("TI-GEOM: L_over_Di must be a finite real number > 0")
    return float(CR), float(L_over_Di)


def wall_f(xi, law, x_m=None):
    """f(xi) and df/dxi on the wall law; a later piece overwrites the junction."""
    _law_check(law, x_m)
    s_arr = np.asarray(xi, dtype=float)
    if not bool(np.all(np.isfinite(s_arr))) or np.any(s_arr < 0.0) or np.any(s_arr > 1.0):
        raise ValueError("TI-GEOM: xi must be finite and within [0, 1]")
    f = np.zeros(s_arr.shape, dtype=float)
    df = np.zeros(s_arr.shape, dtype=float)
    for xi0, xi1, a in law_pieces(law, x_m):
        w = xi1 - xi0
        m = (s_arr >= xi0) & (s_arr <= xi1)
        ss = (s_arr[m] - xi0) / w
        f[m] = P.polyval(ss, a)
        df[m] = P.polyval(ss, P.polyder(a)) / w
    return f, df


def kre_curve(xi, law, CR, L_over_Di, x_m=None):
    """K Re_De along xi on the 1-D area rule U/U_e = (R_e/r)^2."""
    _law_check(law, x_m)
    q = math.sqrt(_geom_cr(CR, L_over_Di)[0])
    f, fp = wall_f(xi, law, x_m)
    return 2.0 * (q - (q - 1.0) * f) * (q - 1.0) * fp / (L_over_Di * q)


def kre_max(law, CR, L_over_Di, x_m=None):
    """(max of K Re_De, xi at it), EXACT from the wall law's stationary points.

    K Re_De = 2 (q - (q-1) f) (q-1) f'(xi) / (L_over_Di q); on a piece with s = (xi - xi0)/w the
    stationary points come from d/ds [(q - (q-1) f(s)) df/ds] = 0, a polynomial solved by polyroots.
    """
    _law_check(law, x_m)
    cr, lodi = _geom_cr(CR, L_over_Di)
    q = math.sqrt(cr)
    best = (-math.inf, None)
    for xi0, xi1, a in law_pieces(law, x_m):
        w = xi1 - xi0
        g = P.polymul(P.polysub([q], P.polymul([q - 1.0], a)), P.polyder(a))
        cands = [0.0, 1.0] + [float(z.real) for z in P.polyroots(P.polyder(g))
                              if abs(z.imag) <= 1e-9 and 0.0 <= z.real <= 1.0]
        for s in cands:
            v = 2.0 * (q - 1.0) * float(P.polyval(s, g)) / (w * lodi * q)
            if v > best[0]:
                best = (v, xi0 + w * s)
    return best


def k_max_1d(law, CR, L_over_Di, Re_De, x_m=None):
    """The a priori K_max record: kre_max, K_max and the closed-form shortest L/D_i at K_lim.

    K Re_De is exactly proportional to 1/(L/D_i), so the shortest L/D_i with K_max <= K_RELAM
    is kre_max(law, CR, 1.0) / (K_RELAM Re_De).
    """
    kre, xi_at = kre_max(law, CR, L_over_Di, x_m)
    re_de = _pos_real(Re_De, "TI-RE: Re_De must be a finite real number > 0")
    kmax = kre / re_de
    return {"law": law, "CR": float(CR), "L_over_Di": float(L_over_Di),
            "x_m": (None if x_m is None else float(x_m)), "Re_De": re_de,
            "kre_max": float(kre), "xi_at_max": float(xi_at), "K_max": kmax,
            "K_lim": K_RELAM, "guard_ok": bool(kmax <= K_RELAM),
            "shortest_L_over_Di": float(kre_max(law, CR, 1.0, x_m)[0] / (K_RELAM * re_de))}


def contraction_1d(Di, CR, L_over_Di, law, Lx_over_De, Ue, n=6001, x_m=None):
    """The 1-D nozzle: x, r, U on the area rule, and the exit_plane station ie (docs/16 line 514)."""
    Ri = Di / 2.0
    Rex = Ri / math.sqrt(CR)
    L = L_over_Di * Di
    Lx = Lx_over_De * 2.0 * Rex
    x = np.linspace(0.0, L + Lx, n)
    f, _ = wall_f(np.clip(x / L, 0.0, 1.0), law, x_m)
    r = Ri - (Ri - Rex) * f
    U = Ue * (Rex / r) ** 2
    ie = int(np.argmin(np.abs(x - L)))
    return x, r, U, ie


def head_curves(curves):
    """(G, Ginv, F): scalar float versions of Head's H1(H) and F(H1) relations."""
    if curves == "digitised":
        gh = np.array([float(v[0]) for v in HEAD_G_TABLE])
        gv = np.array([float(v[1]) for v in HEAD_G_TABLE])
        fa = np.array([float(v[0]) for v in HEAD_F_TABLE])
        fv = np.array([float(v[1]) for v in HEAD_F_TABLE])
        def gfun(h):
            return float(np.interp(h, gh, gv))
        def ginv(h1):
            return float(np.interp(h1, gv[::-1], gh[::-1]))
        def ffun(h1):
            return float(np.interp(h1, fa, fv))
        return gfun, ginv, ffun
    if curves == "fit":
        def gfun(h):
            h = float(h)
            with np.errstate(all="ignore"):
                if h <= 1.6:
                    return float(np.float64(3.3 + 0.8234 * np.float64(h - 1.1) ** -1.287))
                return float(np.float64(3.3 + 1.5501 * np.float64(h - 0.6778) ** -3.064))
        def ginv(h1):
            h1 = float(h1)
            with np.errstate(all="ignore"):
                if h1 >= 5.3:
                    return float(np.float64(1.1 + np.float64((h1 - 3.3) / 0.8234) ** (-1.0 / 1.287)))
                return float(np.float64(0.6778 + np.float64((h1 - 3.3) / 1.5501) ** (-1.0 / 3.064)))
        def ffun(h1):
            h1 = float(h1)
            with np.errstate(all="ignore"):
                return float(np.float64(0.0306 * np.float64(h1 - 3.0) ** -0.6169))
        return gfun, ginv, ffun
    raise ValueError("TI-CURVES: curves must be one of digitised, fit")


def head(x, U, nu, r=None, theta0=None, H0=1.3, curves="digitised"):
    """Head's entrainment method (R&M 3152) with Ludwieg-Tillmann friction: REPORT-ONLY.

    Heun on the given stations for (theta, q = theta H1); never a gated reference because its two
    relations are curves digitised from the report's figures. Counts the stations whose H or H1
    falls outside the digitised curves.
    """
    th0 = 0.0 if theta0 is None else theta0
    xa, ua, ra, nuf, _ = _stations(x, U, nu, r, th0)
    if isinstance(theta0, bool) or not isinstance(theta0, (int, float)) \
            or not math.isfinite(theta0) or not theta0 > 0:
        raise ValueError("TI-THETA0: theta0 must be a finite real number > 0 for Head's method")
    if isinstance(H0, bool) or not isinstance(H0, (int, float)) or not math.isfinite(H0) \
            or not HEAD_G_TABLE[0][0] <= H0 <= HEAD_G_TABLE[-1][0]:
        raise ValueError("TI-H0: H0 must be a finite real number in [%r, %r]"
                         % (HEAD_G_TABLE[0][0], HEAD_G_TABLE[-1][0]))
    if curves not in ("digitised", "fit"):
        raise ValueError("TI-CURVES: curves must be one of digitised, fit")
    G, Ginv, F = head_curves(curves)
    n = len(xa)
    ru = np.ones(n) if ra is None else ra
    dU = thwaites.sg_derivative(xa, ua)
    dr = thwaites.sg_derivative(xa, ru)

    def rhs(i, th, qv):
        h1 = qv / th
        h = Ginv(h1)
        cf = LT_A * 10.0 ** (LT_B * h) * (ua[i] * th / nuf) ** LT_C
        a = dU[i] / ua[i]
        b = dr[i] / ru[i]
        return (cf / 2.0 - th * ((2.0 + h) * a + b), F(h1) - qv * (a + b))

    th = np.zeros(n)
    qv = np.zeros(n)
    th[0] = float(theta0)
    qv[0] = float(theta0) * G(float(H0))
    with np.errstate(all="ignore"):
        for i in range(n - 1):
            h = xa[i + 1] - xa[i]
            k1 = rhs(i, th[i], qv[i])
            k2 = rhs(i + 1, th[i] + h * k1[0], qv[i] + h * k1[1])
            th[i + 1] = th[i] + h * (k1[0] + k2[0]) / 2.0
            qv[i + 1] = qv[i] + h * (k1[1] + k2[1]) / 2.0
        h1s = qv / th
        hs = np.array([Ginv(v) for v in h1s])
        cf = LT_A * 10.0 ** (LT_B * hs) * (ua * th / nuf) ** LT_C
        extrapolated = (hs < 1.28) | (hs > 2.95) | (h1s < 3.40) | (h1s > 11.80)
    return {"method": HEAD_METHOD, "curves": curves, "report_only": True,
            "x": xa, "theta": th, "H": hs, "H1": h1s, "Cf": cf,
            "extrapolated": extrapolated.astype(bool),
            "n_extrapolated": int(np.count_nonzero(extrapolated)),
            "finite": bool(np.all(np.isfinite(th)) and np.all(np.isfinite(hs))),
            "range": {"H": [1.28, 2.95], "H1": [3.40, 11.80]}}


def head_residual():
    """The digitised curves against the Cebeci-Bradshaw fits, rel = fit/digitised - 1 (REPORT-ONLY)."""
    G, _, F = head_curves("fit")
    rel_g = np.array([G(h) / v - 1.0 for h, v in HEAD_G_TABLE])
    rel_f = np.array([F(h1) / v - 1.0 for h1, v in HEAD_F_TABLE])
    gh = np.array([float(v[0]) for v in HEAD_G_TABLE])
    fa = np.array([float(v[0]) for v in HEAD_F_TABLE])
    return {"G_max_rel": float(np.max(np.abs(rel_g))),
            "G_max_rel_H_ge_1.36": float(np.max(np.abs(rel_g[gh >= 1.36]))),
            "F_max_rel": float(np.max(np.abs(rel_f))),
            "F_max_rel_H1_ge_3.6": float(np.max(np.abs(rel_f[fa >= 3.6]))),
            "n_G": len(HEAD_G_TABLE), "n_F": len(HEAD_F_TABLE)}


def main(argv):
    """The CLI of docs/16 §I CAD-24: --selftest, solve IN OUT, kmax LAW CR L_OVER_DI RE_DE [X_M]."""
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
            print(common.canonical_json({"n": len(out["x"]), "theta_last": out["theta"][-1]}))
            return 0
        if len(argv) in (5, 6) and argv[0] == "kmax":
            vals = [float(v) for v in argv[2:]]
            res = k_max_1d(argv[1], vals[0], vals[1], vals[2], (vals[3] if len(vals) > 3 else None))
            print(common.canonical_json(res))
            return 0
    except (ValueError, OSError) as e:
        print("turb_integral: %s" % (e,), file=sys.stderr)
        return 2
    print(USAGE, file=sys.stderr)
    return 2


NU = 1.516e-5


def _expect(fn, tid):
    """The refusal-test helper: the callable must raise ValueError starting with the TI id."""
    try:
        fn()
    except ValueError as e:
        assert str(e).startswith(tid), "%s: %s" % (tid, e)
        return
    raise AssertionError("no ValueError raised for %s" % tid)


def _pred_plate():
    """T1: the flat-plate limit of the 1/7-power method, against 0.037 and 0.0592 literally."""
    x = np.linspace(0.0, 1.0, 2001)
    uu = np.full(x.shape, 15.0)
    res = solve(x, uu, NU)
    rex = 15.0 * x / NU
    th = res["theta"]
    cf = res["Cf"]
    e1 = float(np.max(np.abs(th[1:] * rex[1:] ** 0.2 / x[1:] / 0.037 - 1.0)))
    e2 = float(np.max(np.abs(cf[1:] / (0.0592 * rex[1:] ** -0.2) - 1.0)))
    ok = e1 <= 1e-6 and e2 <= 1e-6
    ok = ok and bool(np.isnan(res["Cf"][0])) and bool(np.isnan(res["u_tau"][0])) and bool(np.isnan(res["p"][0]))
    return ok, e1, e2


def _pred_cone(rfun):
    """T3: r = 1 + x against the closed form with r^(5/4) in the integrating factor."""
    x = np.linspace(0.0, 1.0, 4001)
    uu = np.full(x.shape, 10.0)
    res = solve(x, uu, NU, rfun(x), 0.0)
    c = 0.037 ** 1.25 / 1.25
    theta_exact = (1.25 * c * (NU / 10.0) ** 0.25 * (4.0 / 9.0)
                   * ((1.0 + x) ** 2.25 - 1.0) / (1.0 + x) ** 1.25) ** 0.8
    err = float(np.max(np.abs(res["theta"][1:] / theta_exact[1:] - 1.0)))
    return err <= 1e-6, err


def _pred_ode():
    """T5 (the ODE part): the quadrature satisfies the plan's ODE, literals not globals."""
    x, r, uu, ie = contraction_1d(0.3, 2.0, 1.5, "poly5", 0.5, 30.0)
    res = solve(x, uu, NU, r, 1.5547e-3)
    sg = thwaites.sg_derivative
    h9 = 9.0 / 7.0
    c = 0.037 ** 1.25 / 1.25
    big_th = res["theta"] ** 1.25
    lhs = sg(x, big_th) + 1.25 * big_th * ((2.0 + h9) * sg(x, uu) / uu + sg(x, r) / r)
    rhs = 1.25 * c * (NU / uu) ** 0.25
    err = float(np.max(np.abs(lhs / rhs - 1.0)))
    return err <= 1e-5, err


def _pred_pn():
    """T6 (the PN part): eq. 16.26 in closed form gives f 0.02579 and 0.01802 at the two Re_tau."""
    e1 = abs(f_prandtl(576.69) - 0.02579)
    e2 = abs(f_prandtl(2358.00) - 0.01802)
    return e1 <= 1e-5 and e2 <= 1e-5, e1, e2


def _t1():
    ok, e1, e2 = _pred_plate()
    assert ok, "plate %r %r" % (e1, e2)
    print("[ok] flat plate: theta Re_x^(1/5)/x = 0.037 within %.1e and local Cf = 0.0592 Re_x^(-1/5) within %.1e"
          % (e1, e2))


def _t2():
    x, r, uu, ie = contraction_1d(0.3, 2.0, 1.5, "poly5", 0.5, 30.0)
    a = solve(x, uu, NU)
    b = solve(x, uu, NU, np.full(x.shape, 0.03))
    assert np.array_equal(a["theta"], b["theta"])
    assert np.array_equal(a["dstar"], b["dstar"])
    assert np.array_equal(a["Cf"], b["Cf"], equal_nan=True)
    assert np.array_equal(a["u_tau"], b["u_tau"], equal_nan=True)
    assert np.array_equal(a["K"], b["K"])
    assert np.array_equal(a["p"], b["p"], equal_nan=True)
    cr = solve(x, uu, NU, r)
    c7 = solve(x, uu, NU, 7.0 * r)
    assert thwaites.close(cr["theta"], c7["theta"], 1e-12)
    assert thwaites.close(cr["Cf"], c7["Cf"], 1e-12)
    assert thwaites.close(cr["p"], c7["p"], 1e-12)
    print("[ok] constant radius reduces to planar bit for bit and r scale drops out to 1e-12")


def _t3():
    ok, err = _pred_cone(lambda t: 1.0 + t)
    assert ok, "cone %r" % (err,)
    print("[ok] axisymmetric r = 1 + x: theta matches the closed form within %.1e, so r enters as r^(5/4) in the factor"
          % err)


def _t4():
    x, r, u30, ie = contraction_1d(0.3, 2.0, 1.5, "poly5", 0.5, 30.0)
    _, _, u60, _ = contraction_1d(0.3, 2.0, 1.5, "poly5", 0.5, 60.0)
    a30 = solve(x, u30, NU, r)
    a60 = solve(x, u60, NU, r)
    s = math.log(a60["theta"][ie] / a30["theta"][ie]) / math.log(2.0)
    assert abs(s + 0.2) <= 1e-6, "slope %r" % (s,)
    print("[ok] Reynolds slope d ln theta_e / d ln Re_De = %.7f on the turbulent nominal (-0.2 within 1e-6)" % s)


def _t5():
    ok, err = _pred_ode()
    assert ok, "ode %r" % (err,)
    k = 3000
    x, r, uu, ie = contraction_1d(0.3, 2.0, 1.5, "poly5", 0.5, 30.0)
    one = solve(x, uu, NU, r, 1.5547e-3)
    pa = solve(x[:k + 1], uu[:k + 1], NU, r[:k + 1], 1.5547e-3)
    pb = solve(x[k:], uu[k:], NU, r[k:], pa["theta"][k])
    assert thwaites.close(pa["theta"], one["theta"][:k + 1], 1e-12)
    assert thwaites.close(pb["theta"][1:], one["theta"][k + 1:], 1e-12)
    print("[ok] the quadrature solves d(theta^(5/4))/dx + (5/4) theta^(5/4) [(2 + H) U'/U + r'/r] = (5/4) c (nu/U)^(1/4)"
          " within %.1e, and a restart at station 3000 reproduces it" % err)


def _t6():
    ok, e1, e2 = _pred_pn()
    assert ok, "pn %r %r" % (e1, e2)
    fp1 = f_prandtl(576.69)
    fp2 = f_prandtl(2358.00)
    for fp, rt in ((fp1, 576.69), (fp2, 2358.00)):
        re_d = re_d_from_re_tau(rt)
        assert abs(1.0 / math.sqrt(fp) - (2.0 * math.log10(re_d * math.sqrt(fp)) - 0.8)) <= 1e-12
    fb = f_blasius(2.0e4)
    assert abs(fb - 0.026606) <= 1e-6
    d1 = 100.0 * (f_blasius(re_d_from_re_tau(576.69)) / fp1 - 1.0)
    d2 = 100.0 * (f_blasius(re_d_from_re_tau(2358.00)) / fp2 - 1.0)
    assert abs(d1 - 2.77) <= 0.006 and abs(d2 + 1.09) <= 0.006, "blasius vs pn %r %r" % (d1, d2)
    print("[ok] pipe laws: Prandtl-Nikuradse f %.5f at Re_tau 576.69 and %.5f at 2358.00, Blasius %.6f at Re_D 2e4,"
          " Blasius vs PN %+.2f %% and %+.2f %%" % (fp1, fp2, fb, d1, d2))


def _t7():
    v1 = u_plus_log(1.0)
    assert isinstance(v1, float) and v1 == 5.5
    assert abs(u_plus_log(math.e ** 2) - 10.5) <= 1e-12
    arr = u_plus_log(np.array([1.0, math.e ** 2]))
    assert isinstance(arr, np.ndarray) and arr.shape == (2,)
    y = np.geomspace(30.0, 300.0, 101)
    gap = u_plus_log(y) - np.log(9.8 * y) / 0.41
    gmin = float(np.min(gap))
    gmax = float(np.max(gap))
    assert abs(gmin - 0.14) <= 0.005 and abs(gmax - 0.28) <= 0.005, "gap %r %r" % (gmin, gmax)
    assert abs(u_axis_from_bulk(10.0, 0.5) - 12.035) <= 1e-12
    d1 = pipe_drive(576.69, 0.025, NU)
    d2 = pipe_drive(2358.00, 0.025, NU)
    assert d1["g_x"] == 2.0 * d1["u_tau"] ** 2 / 0.025 and d2["g_x"] == 2.0 * d2["u_tau"] ** 2 / 0.025
    assert thwaites.close([d1["u_tau"]], [0.34971], 5e-5) and thwaites.close([d2["u_tau"]], [1.42989], 5e-5)
    assert thwaites.close([d1["g_x"]], [9.78357], 5e-5) and thwaites.close([d2["g_x"]], [163.567], 5e-5)
    print("[ok] log law 2.5 ln y+ + 5.5 sits %.3f to %.3f above the solver's ln(9.8 y+)/0.41 over y+ 30-300;"
          " core defect 4.07; g_x %.5f and %.3f m/s2" % (gmin, gmax, d1["g_x"], d2["g_x"]))


def _t8():
    x = np.linspace(0.0, 0.5, 2001)
    m = 2.0
    uu = m / (1.0 - x)
    k = accel_K(x, uu, NU)
    ek = float(np.max(np.abs(k / (NU / m) - 1.0)))
    assert ek <= 1e-6, "K %r" % (ek,)
    res = solve(x, uu, NU)
    pp = p_param(x, uu, NU, 0.05)
    exact = -NU * uu * (m / (1.0 - x) ** 2) / 0.05 ** 3
    ep = float(np.max(np.abs(pp / exact - 1.0)))
    assert ep <= 1e-6, "p %r" % (ep,)
    assert np.array_equal(res["K"], k)
    assert np.array_equal(res["p"], p_param(x, uu, NU, res["u_tau"]), equal_nan=True)
    print("[ok] sink flow: K = nu/m within %.1e and p = -nu U U'/u_tau^3 within %.1e" % (ek, ep))


def _t9():
    v9 = kre_max("poly5", 9.0, 1.0)[0]
    v8 = kre_max("poly5", 2.0, 1.5)[0]
    assert thwaites.close([v9], [5.454], 1e-3), "kre CR9 %r" % (v9,)
    assert thwaites.close([v8], [0.895], 1e-3), "kre CR2 %r" % (v8,)
    xi = np.linspace(0.0, 1.0, 40001)
    f, _ = wall_f(xi, "poly5")
    for cr, lodi, val in ((9.0, 1.0, v9), (2.0, 1.5, v8)):
        q = math.sqrt(cr)
        u = 1.0 / (q - (q - 1.0) * f) ** 2
        kfd = 2.0 * np.gradient(u, xi * lodi * 2.0 * q) / u ** 2
        fd = float(np.max(kfd))
        assert abs(fd / val - 1.0) <= 1e-6, "fd %r vs %r" % (fd, val)
        assert float(np.max(kre_curve(xi, "poly5", cr, lodi))) <= val + 1e-12
    re45 = 45.0 * (0.3 / math.sqrt(2.0)) / NU
    shorts = []
    for law, want in (("poly3", 0.5704), ("poly5", 0.7107), ("poly7", 0.8282)):
        rec = k_max_1d(law, 2.0, 1.0, re45)
        assert thwaites.close([rec["shortest_L_over_Di"]], [want], 1e-3), "%s %r" % (law, rec)
        rec2 = k_max_1d(law, 2.0, rec["shortest_L_over_Di"], re45)
        assert abs(rec2["K_max"] / K_RELAM - 1.0) <= 1e-12
        shorts.append(rec["shortest_L_over_Di"])
    de = 0.3 / math.sqrt(2.0)
    assert "%.2e" % k_max_1d("poly5", 2.0, 1.5, 30.0 * de / NU)["K_max"] == "2.13e-06"
    assert "%.2e" % k_max_1d("poly5", 2.0, 1.5, 60.0 * de / NU)["K_max"] == "1.07e-06"
    kp7 = k_max_1d("poly7", 2.0, 0.5, re45)
    assert "%.2e" % kp7["K_max"] == "4.97e-06" and kp7["guard_ok"] is False
    vc, xic = kre_max("cubic_matched", 9.0, 1.0, 0.5)
    assert abs(vc - 8.0) <= 1e-12 and abs(xic - 0.5) <= 1e-12
    print("[ok] K_max Re_De %.4f (CR 9, L/D_i 1) and %.4f (CR 2, L/D_i 1.5); shortest L/D_i %.4f / %.4f / %.4f"
          " at U_e 45; a priori K_max 2.13e-6, 1.07e-6, poly7 4.97e-6" % (v9, v8, shorts[0], shorts[1], shorts[2]))


def _t10():
    with open(os.path.join(HERE, "templates", "nozzle_contraction", "template.py"), "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    poly_node = None
    laws_node = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == "POLY":
                poly_node = node.value
            elif node.targets[0].id == "LAWS":
                laws_node = node.value

    def num(nd):
        if isinstance(nd, ast.Constant) and isinstance(nd.value, (int, float)) and not isinstance(nd.value, bool):
            return float(nd.value)
        if isinstance(nd, ast.UnaryOp) and isinstance(nd.op, ast.USub) and isinstance(nd.operand, ast.Constant):
            return -float(nd.operand.value)
        raise AssertionError("unexpected AST node in the template tables: %r" % (nd,))

    built = {}
    for key, val in zip(poly_node.keys, poly_node.values):
        built[key.value] = tuple(num(e) for e in val.elts)
    assert built == POLY
    assert tuple(nd.value for nd in laws_node.elts) == LAWS
    for xm in (0.2, 0.5, 0.8):
        f0, d0 = wall_f(np.array([0.0]), "cubic_matched", xm)
        f1v, d1v = wall_f(np.array([1.0]), "cubic_matched", xm)
        assert f0[0] == 0.0 and f1v[0] == 1.0 and d0[0] == 0.0 and d1v[0] == 0.0
        pieces = law_pieces("cubic_matched", xm)
        w0 = pieces[0][1] - pieces[0][0]
        w1 = pieces[1][1] - pieces[1][0]
        d_left = float(P.polyval(1.0, P.polyder(pieces[0][2])) / w0)
        d_right = float(P.polyval(0.0, P.polyder(pieces[1][2])) / w1)
        f_left = float(P.polyval(1.0, pieces[0][2]))
        fm, dm = wall_f(np.array([xm]), "cubic_matched", xm)
        assert abs(d_left - d_right) <= 1e-12
        assert abs(fm[0] - f_left) <= 1e-12 and abs(dm[0] - d_right) <= 1e-12
    print("[ok] wall laws equal the template's POLY and LAWS and the matched cubic is C1 at x_m")


def _t11():
    hr = head_residual()
    for key, want in (("G_max_rel", 0.097394), ("G_max_rel_H_ge_1.36", 0.025184),
                      ("F_max_rel", 0.126045), ("F_max_rel_H1_ge_3.6", 0.069599)):
        assert abs(hr[key] - want) <= 1e-6, "%s %r" % (key, hr[key])
    assert hr["n_G"] == 18 and hr["n_F"] == 15
    x0 = 1.0e6 * NU / 15.0
    x1 = 1.0e7 * NU / 15.0
    xs = np.linspace(x0, x1, 4001)
    th0 = 0.037 * x0 * (1.0e6) ** -0.2
    ratios = []
    hlast = []
    for cv in ("digitised", "fit"):
        hr2 = head(xs, np.full(xs.shape, 15.0), NU, None, th0, 1.42, cv)
        assert hr2["finite"] and hr2["n_extrapolated"] == 0
        ratios.append(hr2["theta"][-1] / (0.037 * x1 * (1.0e7) ** -0.2))
        hlast.append(float(hr2["H"][-1]))
    assert 0.9 < ratios[0] < 1.1 and 0.9 < ratios[1] < 1.1
    assert 1.25 < hlast[0] < 1.45 and 1.25 < hlast[1] < 1.45
    print("[ok] Head curves digitised from R&M 3152 Figs. 1-2: the Cebeci-Bradshaw fits differ by %.3f (H1(H);"
          " %.3f for H >= 1.36) and %.3f (F(H1); %.3f for H1 >= 3.6); flat plate to Re_x 1e7:"
          " theta/(0.037 x Re_x^-1/5) %.4f digitised, %.4f fit, H %.3f / %.3f"
          % (hr["G_max_rel"], hr["G_max_rel_H_ge_1.36"], hr["F_max_rel"], hr["F_max_rel_H1_ge_3.6"],
             ratios[0], ratios[1], hlast[0], hlast[1]))


def _t12():
    x, r, u30, ie = contraction_1d(0.3, 2.0, 1.5, "poly5", 0.5, 30.0)
    _, _, u60, _ = contraction_1d(0.3, 2.0, 1.5, "poly5", 0.5, 60.0)
    runs = []
    for uu, th0 in ((u30, 1.5547e-3), (u60, 1.3534e-3)):
        hd = head(x, uu, NU, r, th0, 1.30, "digitised")
        hf = head(x, uu, NU, r, th0, 1.30, "fit")
        p17 = solve(x, uu, NU, r, th0)
        assert hd["finite"] and hf["finite"] and bool(np.all(np.isfinite(p17["theta"])))
        runs.append((hd, hf, p17))
    (hd30, hf30, p17_30) = runs[0]
    (hd60, hf60, p17_60) = runs[1]
    nd30 = int(np.count_nonzero(hd30["extrapolated"][:ie + 1]))
    nd60 = int(np.count_nonzero(hd60["extrapolated"][:ie + 1]))
    print("[ok] REPORT-ONLY Head on the turbulent nominal: U_e 30 theta_e %.4f mm H_e %.3f (%d of %d stations"
          " outside the digitised curves; fit %.4f mm), U_e 60 %.4f mm H_e %.3f (%d of %d; fit %.4f mm);"
          " the 1/7-power method gives %.4f and %.4f mm"
          % (hd30["theta"][ie] * 1e3, hd30["H"][ie], nd30, ie + 1, hf30["theta"][ie] * 1e3,
             hd60["theta"][ie] * 1e3, hd60["H"][ie], nd60, ie + 1, hf60["theta"][ie] * 1e3,
             p17_30["theta"][ie] * 1e3, p17_60["theta"][ie] * 1e3))


def _t13():
    xs = np.linspace(0.0, 1.0, 2001)
    us = np.full(xs.shape, 15.0)
    x10 = np.linspace(0.0, 1.0, 10)
    ubad = np.full(xs.shape, 15.0)
    ubad[7] = np.nan
    xsw = np.array(xs)
    xsw[3], xsw[4] = xs[4], xs[3]
    rr = 1.0 + xs
    rr[3] = 0.0
    _expect(lambda: solve(xs, us[:-1], NU), "TI-SHAPE")
    _expect(lambda: solve(x10, np.full(x10.shape, 15.0), NU), "TI-SHAPE")
    _expect(lambda: solve(xs, ubad, NU), "TI-FINITE")
    _expect(lambda: solve(xs, us, float("inf")), "TI-FINITE")
    _expect(lambda: solve(xsw, us, NU), "TI-ORDER")
    _expect(lambda: solve(xs, us, 0.0), "TI-NU")
    _expect(lambda: solve(xs, np.concatenate(([0.0], us[1:])), NU), "TI-U")
    _expect(lambda: solve(xs, us, NU, rr), "TI-R")
    _expect(lambda: solve(xs, us, NU, None, -1e-6), "TI-THETA0")
    _expect(lambda: solve_doc({"x": list(xs), "U": list(us), "nu": NU, "extra": 1}), "TI-INPUT")
    _expect(lambda: solve_doc({"x": list(xs), "U": list(us)}), "TI-INPUT")
    _expect(lambda: k_max_1d("poly4", 2.0, 1.5, 4.0e5), "TI-LAW")
    _expect(lambda: k_max_1d("cubic_matched", 2.0, 1.5, 4.0e5), "TI-LAW")
    _expect(lambda: k_max_1d("poly5", 1.0, 1.5, 4.0e5), "TI-GEOM")
    _expect(lambda: k_max_1d("poly5", 2.0, 0.0, 4.0e5), "TI-GEOM")
    _expect(lambda: k_max_1d("poly5", 2.0, 1.5, 0.0), "TI-RE")
    _expect(lambda: f_blasius(-1.0), "TI-RE")
    _expect(lambda: f_prandtl(float("nan")), "TI-RE")
    _expect(lambda: u_plus_log(0.0), "TI-YPLUS")
    _expect(lambda: head(xs, us, NU, None, 0.0), "TI-THETA0")
    _expect(lambda: head(xs, us, NU, None, 1.0e-3, 1.0), "TI-H0")
    _expect(lambda: head(xs, us, NU, None, 1.0e-3, 1.4, "spline"), "TI-CURVES")
    print("[ok] 22 bad inputs refused by the 14 TI ids, each by exactly its id")


def _t14():
    def mutate(name, val, fn):
        saved = globals()[name]
        try:
            globals()[name] = val
            return fn()
        finally:
            globals()[name] = saved
    ok_a = not mutate("PLATE_THETA", 0.036, lambda: _pred_plate()[0])
    ok_b = not _pred_cone(lambda t: (1.0 + t) ** 0.8)[0]
    ok_c = not mutate("H17", 1.4, lambda: _pred_ode()[0])
    ok_d = not mutate("PN_B", -0.91, lambda: _pred_pn()[0])
    assert ok_a and ok_b and ok_c and ok_d, "mutants %r" % ((ok_a, ok_b, ok_c, ok_d),)
    assert _pred_plate()[0]
    assert _pred_cone(lambda t: 1.0 + t)[0]
    assert _pred_ode()[0]
    assert _pred_pn()[0]
    print("[ok] 4 of 4 planted mutants (plate 0.036, r to the first power, H 1.4, the theoretical -0.91 of eq. 16.25)"
          " fail their gate")


def _t15():
    import json
    with tempfile.TemporaryDirectory() as td:
        xin = np.linspace(0.0, 1.0, 2001)
        uin = np.full(xin.shape, 15.0)
        doc = {"x": [float(v) for v in xin], "U": [float(v) for v in uin], "nu": NU, "r": None}
        pin = os.path.join(td, "in.json")
        common.atomic_write(pin, common.canonical_json(doc) + chr(10))
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        script = os.path.abspath(__file__)
        outs = []
        for name in ("out1.json", "out2.json"):
            po = os.path.join(td, name)
            pr = subprocess.run([sys.executable, script, "solve", pin, po], capture_output=True, text=True,
                                encoding="utf-8", env=env, timeout=120)
            assert pr.returncode == 0, pr.stderr
            outs.append(po)
        with open(outs[0], "rb") as fh:
            b1 = fh.read()
        with open(outs[1], "rb") as fh:
            b2 = fh.read()
        assert b1 == b2
        want = (common.canonical_json(thwaites.to_json(solve(xin, uin, NU))) + chr(10)).encode("utf-8")
        assert b1 == want
        parsed = json.loads(b1.decode("utf-8"))
        assert parsed["Cf"][0] is None and parsed["theta"][0] == 0.0
        xb = [float(v) for v in xin]
        xb[3], xb[4] = xb[4], xb[3]
        pin2 = os.path.join(td, "bad.json")
        common.atomic_write(pin2, common.canonical_json({"x": xb, "U": [float(v) for v in uin], "nu": NU}) + chr(10))
        pout2 = os.path.join(td, "never.json")
        pr2 = subprocess.run([sys.executable, script, "solve", pin2, pout2], capture_output=True, text=True,
                             encoding="utf-8", env=env, timeout=120)
        assert pr2.returncode == 2
        assert not os.path.exists(pout2)
        assert pr2.stderr.startswith("turb_integral: TI-ORDER")
        pr3 = subprocess.run([sys.executable, script, "kmax", "poly5", "2", "1.5", "420000"], capture_output=True,
                             text=True, encoding="utf-8", env=env, timeout=120)
        assert pr3.returncode == 0, pr3.stderr
        km = json.loads(pr3.stdout)
        assert abs(km["kre_max"] / k_max_1d("poly5", 2.0, 1.5, 420000.0)["kre_max"] - 1.0) <= 1e-12
        pr4 = subprocess.run([sys.executable, script, "kmax", "poly4", "2", "1.5", "420000"], capture_output=True,
                             text=True, encoding="utf-8", env=env, timeout=120)
        assert pr4.returncode == 2
        assert pr4.stderr.startswith("turb_integral: TI-LAW")
        pr5 = subprocess.run([sys.executable, script, "solve", pin], capture_output=True, text=True,
                             encoding="utf-8", env=env, timeout=120)
        assert pr5.returncode == 2
    assert not any(m in sys.modules for m in ("cadquery", "OCP", "gmsh", "scipy"))
    print("[ok] CLI: two solve runs byte-identical and equal to solve(); Cf null at a zero-theta start;"
          " kmax prints the K_max record; exit 2 on TI-ORDER and TI-LAW with nothing written, 2 on usage")


def selftest():
    _t1()
    _t2()
    _t3()
    _t4()
    _t5()
    _t6()
    _t7()
    _t8()
    _t9()
    _t10()
    _t11()
    _t12()
    _t13()
    _t14()
    _t15()
    print("SELFTEST PASS")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
