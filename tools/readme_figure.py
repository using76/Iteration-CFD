#!/usr/bin/env python
"""
The figure at the top of the README: a centre-plane slice of a finished
`ofgpu-fire` run, temperature beside velocity magnitude.

    python tools/readme_figure.py <caseDir> <out.png> [--time T]

Reads the OpenFOAM ASCII fields `ofgpu-fire -output foam` writes, so it needs
no VTK and no ParaView. The mesh shape comes from the case's own cell count
rather than being hard-coded, which is the difference between this and
`render_plume.py` — that one is pinned to the 98x42x20 plume and stays that way
because the numbers in `docs/` were made with it.

Structured meshes only. `blockgen` orders cells i fastest then j then k, so the
field reshapes to (nz, ny, nx) with no gather.
"""

import argparse
import math
import os
import re
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402


def read_field(path):
    """internalField of an OpenFOAM ASCII volScalar/volVectorField."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()

    m = re.search(
        r"internalField\s+nonuniform\s+List<vector>\s*\n?(\d+)\s*\n\((.*?)\n\)\s*;",
        text,
        re.S,
    )
    if m:
        rows = re.findall(r"\(([^)]*)\)", m.group(2))
        return np.array([[float(v) for v in r.split()] for r in rows])

    m = re.search(
        r"internalField\s+nonuniform\s+List<scalar>\s*\n?(\d+)\s*\n\((.*?)\n\)\s*;",
        text,
        re.S,
    )
    if m:
        return np.fromstring(m.group(2), sep=" ")

    m = re.search(r"internalField\s+uniform\s+\(([^)]*)\)\s*;", text)
    if m:
        return np.array([[float(v) for v in m.group(1).split()]])

    m = re.search(r"internalField\s+uniform\s+([-\d.eE+]+)\s*;", text)
    if m:
        return np.full(1, float(m.group(1)))

    raise ValueError(f"{path}: no internalField found")


def case_geometry(case_dir, jsonc=None):
    """(nx, ny, nz) and the bounding box.

    A JSONC case builds its mesh in memory and writes no `constant/polyMesh`,
    so the shape comes from the case file when one is given and from the
    written mesh otherwise. Both are read rather than assumed - a figure whose
    aspect ratio is a guess is a figure that lies about the geometry.
    """
    if jsonc:
        with open(jsonc, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        text = re.sub(r"//[^\n]*", "", text)
        cells = re.search(r'"cells"\s*:\s*\[([^\]]*)\]', text)
        bmin = re.search(r'"min"\s*:\s*\[([^\]]*)\]', text)
        bmax = re.search(r'"max"\s*:\s*\[([^\]]*)\]', text)
        if not (cells and bmin and bmax):
            raise SystemExit(f"{jsonc}: no mesh.cells / mesh.bounds found")
        nx, ny, nz = (int(float(v)) for v in cells.group(1).split(","))
        lo = np.array([float(v) for v in bmin.group(1).split(",")])
        hi = np.array([float(v) for v in bmax.group(1).split(",")])
        return (nx, ny, nz), lo, hi

    pts_path = os.path.join(case_dir, "constant", "polyMesh", "points")
    with open(pts_path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    body = re.search(r"\n(\d+)\s*\n\((.*?)\n\)\s*;", text, re.S)
    rows = re.findall(r"\(([^)]*)\)", body.group(2))
    pts = np.array([[float(v) for v in r.split()] for r in rows])

    # A blockgen mesh is a tensor grid, so the distinct coordinate values along
    # each axis give the shape directly - one more point than cells.
    nx = len(np.unique(np.round(pts[:, 0], 9))) - 1
    ny = len(np.unique(np.round(pts[:, 1], 9))) - 1
    nz = len(np.unique(np.round(pts[:, 2], 9))) - 1
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    return (nx, ny, nz), lo, hi


def latest_time(case_dir):
    times = []
    for name in os.listdir(case_dir):
        p = os.path.join(case_dir, name)
        if not os.path.isdir(p):
            continue
        try:
            t = float(name)
        except ValueError:
            continue
        if t > 0 and os.path.exists(os.path.join(p, "T")):
            times.append((t, name))
    if not times:
        raise SystemExit(f"{case_dir}: no time directory with a T field")
    return max(times)[1]


# A flame ramp that stays legible in both themes: deep blue-grey through
# ember to a pale core. Not a rainbow - the eye reads ordered brightness,
# and a perceptually reversing map invents structure that is not there.
FLAME = LinearSegmentedColormap.from_list(
    "flame",
    ["#141A21", "#26303A", "#5A3A66", "#A8342F", "#E07A18", "#F5C542", "#FDF3D0"],
)
SPEED = LinearSegmentedColormap.from_list(
    "speed",
    ["#141A21", "#173A4A", "#0E7490", "#3FA9C4", "#9FD8E6", "#F2FAFD"],
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("out")
    ap.add_argument("--time", default=None)
    ap.add_argument("--jsonc", default=None,
                    help="the .jsonc case, when the mesh was built in memory")
    args = ap.parse_args()

    t_name = args.time or latest_time(args.case)
    (nx, ny, nz), lo, hi = case_geometry(args.case, args.jsonc)
    n = nx * ny * nz

    temp = read_field(os.path.join(args.case, t_name, "T"))
    vel = read_field(os.path.join(args.case, t_name, "U"))
    if temp.size != n:
        raise SystemExit(f"T has {temp.size} values, mesh has {n} cells")

    temp = temp.reshape(nz, ny, nx)
    speed = np.linalg.norm(vel, axis=1).reshape(nz, ny, nx)

    j = ny // 2  # the vertical centre plane
    t_slice = temp[:, j, :]
    u_slice = speed[:, j, :]

    x = np.linspace(lo[0], hi[0], nx)
    z = np.linspace(lo[2], hi[2], nz)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), constrained_layout=True)
    fig.patch.set_facecolor("#FFFFFF")

    panels = [
        (axes[0], t_slice, FLAME, "temperature", "K", "%.0f"),
        (axes[1], u_slice, SPEED, "velocity magnitude", "m/s", "%.1f"),
    ]
    for ax, data, cmap, title, unit, fmt in panels:
        levels = np.linspace(float(data.min()), float(data.max()), 24)
        if levels[0] == levels[-1]:
            levels = None
        cs = ax.contourf(x, z, data, levels=levels, cmap=cmap)
        ax.set_aspect("equal")
        ax.set_title(f"{title}  [{unit}]", fontsize=10.5, color="#33404F", pad=8)
        ax.set_xlabel("x  [m]", fontsize=9, color="#606C7B")
        ax.set_ylabel("z  [m]", fontsize=9, color="#606C7B")
        ax.tick_params(labelsize=8.5, colors="#606C7B")
        for s in ax.spines.values():
            s.set_color("#DFE5EB")
        cb = fig.colorbar(cs, ax=ax, shrink=0.88, format=fmt)
        cb.ax.tick_params(labelsize=8, colors="#606C7B")
        cb.outline.set_edgecolor("#DFE5EB")

    fig.savefig(args.out, dpi=150, facecolor=fig.get_facecolor())
    print(f"{args.out}  t = {t_name} s  mesh {nx}x{ny}x{nz}")
    print(f"  T    {temp.min():.2f} .. {temp.max():.2f} K")
    print(f"  |U|  {speed.min():.4f} .. {speed.max():.4f} m/s")


if __name__ == "__main__":
    sys.exit(main())
