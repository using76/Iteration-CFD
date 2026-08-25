#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
"""Horizontal-slice contour frames for the `room` case.

Reads the OpenFOAM ASCII output of `ofgpu-buoyant` on a CARVED `room` case,
rebuilds the (i,j) plane at a chosen z from the fluid-only cell ordering
(i fastest, solids skipped - SPEC-LIT §23.4's renumbering), and writes one
PNG per output time for temperature and for velocity magnitude.

The solid mask is re-derived analytically from the same baffle boxes the STL
was generated from, so the script needs no mesh parsing at all.

    python room_slices.py <caseDir> <outDir> --z 2.8125
"""

import argparse
import os
import re
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------
#  Case constants - must match blockgen's `room` kind and make_baffles.py
# --------------------------------------------------------------------------
NX, NY, NZ = 80, 80, 24
LX, LY, LZ = 10.0, 10.0, 3.0
H = LX / NX  # 0.125

# (x0, x1, y0, y1) of each baffle, full height
BAFFLES = [
    (2.375, 2.625, 0.0, 7.5),
    (4.875, 5.125, 2.5, 10.0),
    (7.375, 7.625, 0.0, 7.5),
]

T_MIN, T_MAX = 293.15, 573.15  # ambient .. inlet

# The fire palette of render_plume.py (kept in one place there; duplicated
# here only as data, with the same stops).
_STOPS = [
    (0.00, (10, 10, 40)),
    (0.25, (40, 40, 140)),
    (0.45, (30, 140, 180)),
    (0.62, (60, 200, 90)),
    (0.78, (250, 220, 60)),
    (0.90, (250, 120, 30)),
    (1.00, (255, 240, 200)),
]


def colormap(x):
    x = np.clip(np.nan_to_num(x, nan=0.0), 0.0, 1.0)
    pos = np.array([s[0] for s in _STOPS])
    cols = np.array([s[1] for s in _STOPS], dtype=float)
    out = np.empty(x.shape + (3,), dtype=float)
    for c in range(3):
        out[..., c] = np.interp(x, pos, cols[:, c])
    return out.astype(np.uint8)


# --------------------------------------------------------------------------
#  OpenFOAM ASCII readers
# --------------------------------------------------------------------------

def read_scalar(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    m = re.search(
        r"internalField\s+nonuniform\s+List<scalar>\s*\n?(\d+)\s*\n\((.*?)\n\)",
        text, re.S)
    if m:
        return np.fromstring(m.group(2), sep=" ")
    m = re.search(r"internalField\s+uniform\s+([-\d.eE+]+)\s*;", text)
    if m:
        return None  # caller substitutes the uniform value
    raise ValueError(f"{path}: no internalField")


def read_vector_mag(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    m = re.search(
        r"internalField\s+nonuniform\s+List<vector>\s*\n?(\d+)\s*\n\((.*?)\n\)",
        text, re.S)
    if not m:
        raise ValueError(f"{path}: no vector internalField")
    flat = np.fromstring(m.group(2).replace("(", " ").replace(")", " "), sep=" ")
    v = flat.reshape(-1, 3)
    return np.sqrt((v * v).sum(axis=1))


# --------------------------------------------------------------------------
#  The carved-grid mapping
# --------------------------------------------------------------------------

def solid_mask():
    """(NY, NX) bool, True where a cell centre lies inside any baffle."""
    xc = (np.arange(NX) + 0.5) * H
    yc = (np.arange(NY) + 0.5) * H
    X, Y = np.meshgrid(xc, yc)          # (NY, NX)
    m = np.zeros((NY, NX), dtype=bool)
    for x0, x1, y0, y1 in BAFFLES:
        m |= (X > x0) & (X < x1) & (Y > y0) & (Y < y1)
    return m


def fluid_index_of_slice(k):
    """Index into the FLUID-ONLY cell array for each (j,i) of plane k.

    The carve renumbers fluid cells in (i fastest, j, k) order with solid
    cells skipped; baffles are z-uniform, so the per-plane fluid count is
    constant and the mask is the same at every k.
    """
    m = solid_mask()                    # (NY, NX), True = solid
    fluid_per_plane = int((~m).sum())
    order = np.full((NY, NX), -1, dtype=np.int64)
    seq = 0
    for j in range(NY):
        for i in range(NX):
            if not m[j, i]:
                order[j, i] = seq
                seq += 1
    idx = np.where(order >= 0, order + k * fluid_per_plane, -1)
    return idx, m


def plane(values, idx, mask, fill=np.nan):
    out = np.full(idx.shape, fill, dtype=float)
    ok = idx >= 0
    out[ok] = values[idx[ok]]
    out[mask] = np.nan
    return out


# --------------------------------------------------------------------------
#  Rendering
# --------------------------------------------------------------------------

def render(field2d, lo, hi, title, px=9):
    norm = (field2d - lo) / max(hi - lo, 1e-30)
    rgb = colormap(norm)
    rgb[np.isnan(field2d)] = (52, 52, 56)          # baffles: dark grey
    rgb = rgb[::-1]                                 # y up
    img = Image.fromarray(rgb, "RGB").resize((NX * px, NY * px), Image.NEAREST)

    # colour bar + labels
    barw, pad, footer = 26, 8, 30
    W = img.width + barw + 3 * pad + 64
    Hh = img.height + footer + 2 * pad
    canvas = Image.new("RGB", (W, Hh), (18, 18, 22))
    canvas.paste(img, (pad, pad))

    grad = colormap(np.linspace(1, 0, img.height).reshape(-1, 1))
    grad = np.repeat(grad, barw, axis=1)
    canvas.paste(Image.fromarray(grad, "RGB"), (img.width + 2 * pad, pad))

    d = ImageDraw.Draw(canvas)
    f = ImageFont.load_default()
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        yv = pad + int((1 - frac) * (img.height - 1))
        d.text((img.width + 2 * pad + barw + 4, yv - 5),
               f"{lo + frac * (hi - lo):.0f}", fill=(230, 230, 230), font=f)
    d.text((pad, img.height + pad + 8), title, fill=(240, 240, 240), font=f)
    # inlet / door markers
    d.text((pad + 2, pad + img.height // 2 - 6), "IN>", fill=(255, 255, 255), font=f)
    door0 = pad + int((1 - 6.0 / LY) * img.height)
    door1 = pad + int((1 - 4.0 / LY) * img.height)
    d.line([(pad + img.width - 2, door0), (pad + img.width - 2, door1)],
           fill=(255, 255, 255), width=3)
    d.text((pad + img.width - 40, door1 + 2), "DOOR", fill=(255, 255, 255), font=f)
    return canvas


def save(img, path):
    img.convert("P", palette=Image.ADAPTIVE, colors=256).save(path, optimize=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("out")
    ap.add_argument("--z", type=float, default=2.8125)
    args = ap.parse_args()

    k = int(args.z / (LZ / NZ))
    zc = (k + 0.5) * (LZ / NZ)
    idx, mask = fluid_index_of_slice(k)
    os.makedirs(args.out, exist_ok=True)

    times = sorted(
        (float(d), d) for d in os.listdir(args.case)
        if re.fullmatch(r"\d+(\.\d+)?", d) and float(d) > 0
        and os.path.isfile(os.path.join(args.case, d, "T")))

    # one consistent |U| scale across all frames
    umax = 0.0
    for _, d in times:
        um = read_vector_mag(os.path.join(args.case, d, "U"))
        umax = max(umax, float(np.nanmax(plane(um, idx, mask))))
    umax = float(np.ceil(umax * 2) / 2)

    frames = []
    for t, d in times:
        Tv = read_scalar(os.path.join(args.case, d, "T"))
        Um = read_vector_mag(os.path.join(args.case, d, "U"))
        Tp = plane(Tv, idx, mask)
        Up = plane(Um, idx, mask)
        ti = render(Tp, T_MIN, T_MAX, f"T [K]   z = {zc:.3f} m   t = {t:g} s")
        ui = render(Up, 0.0, umax, f"|U| [m/s]   z = {zc:.3f} m   t = {t:g} s")
        tp = os.path.join(args.out, f"T_{t:05.1f}.png")
        up = os.path.join(args.out, f"U_{t:05.1f}.png")
        save(ti, tp)
        save(ui, up)
        frames.append({
            "t": t, "T": tp, "U": up,
            "Tmax": float(np.nanmax(Tp)), "Tmean": float(np.nanmean(Tp)),
            "Umax": float(np.nanmax(Up)),
        })
        print(f"t={t:5.1f}  T[{np.nanmin(Tp):6.1f},{np.nanmax(Tp):6.1f}] K"
              f"  |U|max {np.nanmax(Up):5.2f} m/s")

    import json
    with open(os.path.join(args.out, "frames.json"), "w") as fh:
        json.dump({"umax": umax, "z": zc, "frames": frames}, fh, indent=1)
    print(f"{len(frames)} frames, |U| scale 0..{umax}")


if __name__ == "__main__":
    sys.exit(main())
