#!/usr/bin/env python3
"""
Render the plume temperature field from an ofgpu case.

matplotlib is not installed on this machine, so everything here is numpy + PIL.
That turns out to be a feature: the 3-D view is a real volume render rather
than a scatter plot, because compositing a structured grid slab by slab is
about ten lines of numpy.

    python tools/render_plume.py <caseDir> <outDir> [--field T]

Writes one PNG per time directory plus frames.json with the per-frame stats
the report needs.
"""

import json
import os
import re
import sys

import numpy as np
from PIL import Image

# --------------------------------------------------------------------------
#  Case reading
# --------------------------------------------------------------------------

def read_internal_field(path):
    """The internalField of an OpenFOAM ASCII volScalarField, as a flat array."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()

    m = re.search(
        r"internalField\s+nonuniform\s+List<scalar>\s*\n?(\d+)\s*\n\((.*?)\n\)",
        text,
        re.S,
    )
    if m:
        return np.fromstring(m.group(2), sep=" ")

    m = re.search(r"internalField\s+uniform\s+([-\d.eE+]+)\s*;", text)
    if m:
        return np.full(1, float(m.group(1)))

    raise ValueError(f"{path}: no internalField found")


def grid_shape(case_dir):
    """(nx, ny, nz) from the blockgen banner the mesh was written with.

    blockgen orders cells i fastest, then j, then k, so a plain reshape to
    (nz, ny, nx) is the structured field - no scatter/gather needed.
    """
    owner = os.path.join(case_dir, "constant", "polyMesh", "owner")
    with open(owner, encoding="utf-8", errors="replace") as fh:
        head = fh.read(4000)
    m = re.search(r"nCells:\s*(\d+)", head)
    n_cells = int(m.group(1)) if m else None

    # The plume case is the only structured case this script is used for, and
    # its shape is fixed by the geometry spec.
    for shape in [(98, 42, 20)]:
        if n_cells is None or shape[0] * shape[1] * shape[2] == n_cells:
            return shape

    raise ValueError(f"unrecognised cell count {n_cells}")


def time_dirs(case_dir):
    """Numeric time directories, ascending, excluding 0."""
    out = []
    for name in os.listdir(case_dir):
        p = os.path.join(case_dir, name)
        if not os.path.isdir(p):
            continue
        try:
            t = float(name)
        except ValueError:
            continue
        if t > 0 and os.path.exists(os.path.join(p, "T")):
            out.append((t, name))
    return sorted(out)


# --------------------------------------------------------------------------
#  Colour
# --------------------------------------------------------------------------

# A black-body-ish ramp. Deliberately NOT a rainbow: rainbow maps invent
# banding that reads as structure the solution does not have.
_STOPS = [
    (0.00, (10, 16, 34)),      # ambient - near black, cool bias
    (0.14, (38, 26, 96)),      # deep indigo
    (0.30, (99, 30, 128)),     # violet
    (0.47, (168, 44, 105)),    # magenta
    (0.63, (222, 78, 58)),     # ember
    (0.78, (245, 141, 30)),    # orange
    (0.90, (252, 202, 62)),    # amber
    (1.00, (255, 248, 214)),   # white hot
]


def colormap(x):
    """x in [0,1] -> uint8 RGB, shape (..., 3)."""
    x = np.clip(x, 0.0, 1.0)
    pos = np.array([s[0] for s in _STOPS])
    cols = np.array([s[1] for s in _STOPS], dtype=float)
    out = np.empty(x.shape + (3,), dtype=float)
    for c in range(3):
        out[..., c] = np.interp(x, pos, cols[:, c])
    return out.astype(np.uint8)


# --------------------------------------------------------------------------
#  Views
# --------------------------------------------------------------------------

def slice_image(field2d, lo, hi, px_per_cell, flip_y=True):
    """A 2-D slice, nearest-neighbour upscaled so cell boundaries stay visible."""
    norm = (field2d - lo) / max(hi - lo, 1e-30)
    rgb = colormap(norm)
    if flip_y:
        rgb = rgb[::-1]
    img = Image.fromarray(rgb, "RGB")
    return img.resize(
        (img.width * px_per_cell, img.height * px_per_cell), Image.NEAREST
    )


def volume_render(vol, lo, hi, scale=8.0, alpha_gain=0.35, gamma=0.75):
    """Isometric volume render of a structured (nz, ny, nx) field.

    Back-to-front compositing along y, but by INVERSE MAPPING rather than by
    splatting voxels forward. Splatting looks obvious and is wrong here: several
    voxels land on one pixel, and numpy fancy-index `+=` applies such a
    duplicate only once, so a forward splat leaves a moire of holes. Walking the
    output pixels instead means every pixel is written exactly once per slab.

    Opacity rises with how far above ambient a cell is, so still air is
    transparent and only the plume accumulates.
    """
    nz, ny, nx = vol.shape
    cos30, sin30 = np.cos(np.pi / 6), np.sin(np.pi / 6)
    S = scale

    # Screen extent of the whole box, so every slab shares one canvas.
    corners = [((x - y) * cos30 * S, (x + y) * sin30 * S - z * S)
               for x in (0, nx) for y in (0, ny) for z in (0, nz)]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    pad = 22
    W = int(max(xs) - min(xs)) + 2 * pad
    H = int(max(ys) - min(ys)) + 2 * pad
    ox, oy = -min(xs) + pad, -min(ys) + pad

    Xg, Yg = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))

    norm_all = np.clip((vol - lo) / max(hi - lo, 1e-30), 0.0, 1.0)

    acc = np.zeros((H, W, 3), dtype=float)
    acc_a = np.zeros((H, W), dtype=float)

    A = cos30 * S            # dX/du
    B = sin30 * S            # dY/du

    # Far slab first: increasing y moves left and DOWN here, so the largest y
    # is furthest from the eye.
    for j in range(ny - 1, -1, -1):
        plane = norm_all[:, j, :]
        if plane.max() <= 0.002:
            continue

        C = ox - j * cos30 * S
        F = oy + j * sin30 * S

        u = (Xg - C) / A                       # x index, fractional
        v = (B * u - Yg + F) / S               # z index, fractional

        inside = (u >= 0) & (u <= nx - 1) & (v >= 0) & (v <= nz - 1)
        if not inside.any():
            continue

        ui = np.clip(u, 0, nx - 1)
        vi = np.clip(v, 0, nz - 1)

        # Bilinear sample so the sheared lattice does not alias into stripes.
        u0 = np.floor(ui).astype(int); u1 = np.minimum(u0 + 1, nx - 1)
        v0 = np.floor(vi).astype(int); v1 = np.minimum(v0 + 1, nz - 1)
        fu = ui - u0; fv = vi - v0

        val = (plane[v0, u0] * (1 - fu) * (1 - fv)
             + plane[v0, u1] * fu * (1 - fv)
             + plane[v1, u0] * (1 - fu) * fv
             + plane[v1, u1] * fu * fv)
        val = np.where(inside, val, 0.0)

        a = (val ** gamma) * alpha_gain
        contrib = a * (1.0 - acc_a)
        rgb = colormap(val).astype(float)
        acc += rgb * contrib[..., None]
        acc_a += contrib

    bg = np.array([13, 17, 22], dtype=float)
    out = acc + bg * (1.0 - acc_a)[..., None]
    img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")

    # Domain wireframe, so the plume is read against the box it sits in.
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)

    def P(x, y, z):
        return ((x - y) * cos30 * S + ox, (x + y) * sin30 * S - z * S + oy)

    edge = (86, 100, 112)
    for (p0, p1) in [
        # floor
        ((0,0,0),(nx,0,0)), ((nx,0,0),(nx,ny,0)), ((nx,ny,0),(0,ny,0)), ((0,ny,0),(0,0,0)),
        # ceiling
        ((0,0,nz),(nx,0,nz)), ((nx,0,nz),(nx,ny,nz)), ((nx,ny,nz),(0,ny,nz)), ((0,ny,nz),(0,0,nz)),
        # verticals
        ((0,0,0),(0,0,nz)), ((nx,0,0),(nx,0,nz)), ((nx,ny,0),(nx,ny,nz)), ((0,ny,0),(0,ny,nz)),
    ]:
        d.line([P(*p0), P(*p1)], fill=edge, width=1)

    # inlet footprint on the floor, 8x8 cells centred
    i0, i1 = nx // 2 - 4, nx // 2 + 4
    j0, j1 = ny // 2 - 4, ny // 2 + 4
    d.line([P(i0,j0,0), P(i1,j0,0), P(i1,j1,0), P(i0,j1,0), P(i0,j0,0)],
           fill=(214, 96, 48), width=2)

    return img


def save(img, path):
    """Palette-quantise before writing.

    Every image here is a colormap lookup, so 256 colours is visually lossless
    and cuts the PNG to roughly a third - which matters because all of them end
    up base64-inlined in a single self-contained report.
    """
    img.convert("P", palette=Image.ADAPTIVE, colors=256).save(path, optimize=True)


# --------------------------------------------------------------------------
#  Main
# --------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1

    case_dir, out_dir = sys.argv[1], sys.argv[2]
    field = "T"
    if "--field" in sys.argv:
        field = sys.argv[sys.argv.index("--field") + 1]

    os.makedirs(out_dir, exist_ok=True)
    nx, ny, nz = grid_shape(case_dir)
    times = time_dirs(case_dir)
    if not times:
        print(f"no time directories with {field} in {case_dir}", file=sys.stderr)
        return 1

    # One colour scale for every frame, or the animation lies about growth.
    lo, hi = None, None
    vols = []
    for t, name in times:
        v = read_internal_field(os.path.join(case_dir, name, field))
        if v.size == 1:
            v = np.full(nx * ny * nz, v[0])
        v = v.reshape(nz, ny, nx)
        vols.append((t, name, v))
        lo = v.min() if lo is None else min(lo, v.min())
        hi = v.max() if hi is None else max(hi, v.max())

    print(f"{len(vols)} frames, {field} in [{lo:.2f}, {hi:.2f}]")

    # z index of the two report planes and the inlet centre-line
    k_ceiling = int(round(2.8 / 3.0 * nz)) - 1          # z = 2.8 m
    k_breath = int(round(1.5 / 3.0 * nz)) - 1           # z = 1.5 m
    j_centre = ny // 2

    manifest = {
        "field": field,
        "range": [float(lo), float(hi)],
        "shape": [nx, ny, nz],
        "planes": {
            "ceiling_z": 2.8,
            "breathing_z": 1.5,
            "section_y": 0.0,
        },
        "frames": [],
    }

    for t, name, v in vols:
        stem = f"{field}_{name.replace('.', 'p')}"

        save(volume_render(v, lo, hi), os.path.join(out_dir, stem + "_3d.png"))
        save(slice_image(v[k_ceiling], lo, hi, 8),
             os.path.join(out_dir, stem + "_zceiling.png"))
        save(slice_image(v[k_breath], lo, hi, 8),
             os.path.join(out_dir, stem + "_zbreath.png"))
        save(slice_image(v[:, j_centre, :], lo, hi, 8),
             os.path.join(out_dir, stem + "_section.png"))

        manifest["frames"].append({
            "t": t,
            "dir": name,
            "stem": stem,
            "min": float(v.min()),
            "max": float(v.max()),
            "mean": float(v.mean()),
            "ceiling_max": float(v[k_ceiling].max()),
            "breath_max": float(v[k_breath].max()),
        })
        print(f"  t = {t:6.2f} s   min {v.min():8.2f}  max {v.max():8.2f}  "
              f"mean {v.mean():8.2f}")

    with open(os.path.join(out_dir, "frames.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)

    print(f"\nwrote {len(vols)*4} images + frames.json to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
