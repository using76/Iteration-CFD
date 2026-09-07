#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""
A parametric open-wheel race car, written to a binary STL.

Why generate one rather than download one. Every open-wheel geometry that is
easy to find is either a GPL tutorial asset, a CAD-site model with no licence
at all, or a benchmark whose terms forbid redistribution. This repository
asserts, and tests, that nothing in it was taken from a GPL source, and a
demonstration case that ships on a public website has to be redistributable
without an asterisk. So the shape here is ours: a stylised 2020s-era open-wheel
car built from primitives, dimensionally in the right place (5.6 m long, 2.0 m
wide, 0.95 m tall, 3.6 m wheelbase) so the Reynolds number and the wake are
those of a real car, without being anyone's actual car.

It is deliberately blunt. A cut-cell mesher wants a closed, non-degenerate
surface far more than it wants fillets, and the flow features this case is
meant to show - the front-wing vortex, the wheel wakes, the underbody
acceleration, the rear-wing downwash - come from where the surfaces are, not
from how smoothly they meet.

    python tools/racecar_stl.py cases/racecar.stl
"""
from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

Vec = tuple[float, float, float]

# x is downstream, y is up, z is to the left. The car sits nose-first into the
# wind with its axle line on x = 0.
LENGTH = 5.60
WIDTH = 2.00
WHEELBASE = 3.60
RIDE = 0.055


def _tri(a: Vec, b: Vec, c: Vec) -> tuple[Vec, Vec, Vec]:
    return (a, b, c)


def quad(a: Vec, b: Vec, c: Vec, d: Vec) -> list[tuple[Vec, Vec, Vec]]:
    """Two triangles, wound so the normal follows the right-hand rule a->b->c."""
    return [_tri(a, b, c), _tri(a, c, d)]


def box(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float) -> list[tuple[Vec, Vec, Vec]]:
    """A closed axis-aligned box with outward normals."""
    p = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]
    return (
        quad(p[0], p[3], p[2], p[1])  # -z
        + quad(p[4], p[5], p[6], p[7])  # +z
        + quad(p[0], p[1], p[5], p[4])  # -y
        + quad(p[3], p[7], p[6], p[2])  # +y
        + quad(p[0], p[4], p[7], p[3])  # -x
        + quad(p[1], p[2], p[6], p[5])  # +x
    )


def loft(sections: list[list[Vec]], cap_start: bool = True, cap_end: bool = True) -> list[tuple[Vec, Vec, Vec]]:
    """
    Skin a stack of equally-sized closed rings. Every ring must be wound the
    same way; the caps are fans from the first vertex, which is watertight for
    the convex rings this file builds.
    """
    tris: list[tuple[Vec, Vec, Vec]] = []
    n = len(sections[0])
    for s0, s1 in zip(sections, sections[1:]):
        for i in range(n):
            j = (i + 1) % n
            tris += quad(s0[i], s0[j], s1[j], s1[i])
    if cap_start:
        r = sections[0]
        for i in range(1, n - 1):
            tris.append(_tri(r[0], r[i + 1], r[i]))
    if cap_end:
        r = sections[-1]
        for i in range(1, n - 1):
            tris.append(_tri(r[0], r[i], r[i + 1]))
    return tris


def ring(x: float, half_w: float, y_low: float, y_high: float, corner: float, n: int = 16) -> list[Vec]:
    """A rounded rectangle in the yz plane at station x — the body's cross-section."""
    cz = 0.0
    cy = (y_low + y_high) / 2
    hz = half_w
    hy = (y_high - y_low) / 2
    r = min(corner, hz * 0.9, hy * 0.9)
    pts: list[Vec] = []
    for i in range(n):
        a = 2 * math.pi * i / n
        # Superellipse: exponent 4 gives a section that is boxy at the sides and
        # rounded at the corners, which is what a monocoque actually looks like.
        e = 2.0 + 2.0 * (r / max(hz, 1e-6))
        cs, sn = math.cos(a), math.sin(a)
        z = cz + hz * math.copysign(abs(cs) ** (2 / e), cs)
        y = cy + hy * math.copysign(abs(sn) ** (2 / e), sn)
        pts.append((x, y, z))
    return pts


def wheel(cx: float, cz: float, radius: float, width: float, n: int = 28) -> list[tuple[Vec, Vec, Vec]]:
    """A rolling wheel: a cylinder about the z axis, flattened where it meets the road."""
    y_axis = radius + RIDE - 0.012  # a hair of contact patch, so it is not tangent
    z0, z1 = cz - width / 2, cz + width / 2
    rings: list[list[Vec]] = []
    for z in (z0, z1):
        pts: list[Vec] = []
        for i in range(n):
            a = 2 * math.pi * i / n
            y = y_axis + radius * math.sin(a)
            x = cx + radius * math.cos(a)
            pts.append((x, max(y, RIDE * 0.5), z))
        rings.append(pts)
    return loft(rings)


def body() -> list[tuple[Vec, Vec, Vec]]:
    """Nose, monocoque, engine cover and airbox as one lofted solid."""
    #      x,     half width, floor y, top y, corner
    stations = [
        (-2.80, 0.045, 0.075, 0.180, 0.03),   # nose tip
        (-2.35, 0.090, 0.070, 0.235, 0.05),
        (-1.90, 0.140, 0.065, 0.300, 0.07),
        (-1.35, 0.215, 0.060, 0.430, 0.09),   # front bulkhead
        (-0.75, 0.310, 0.058, 0.520, 0.10),
        (-0.20, 0.360, 0.058, 0.560, 0.11),   # cockpit
        (0.30, 0.370, 0.058, 0.640, 0.12),    # roll hoop root
        (0.85, 0.330, 0.058, 0.700, 0.12),    # airbox
        (1.45, 0.255, 0.060, 0.610, 0.10),
        (2.05, 0.175, 0.065, 0.470, 0.08),
        (2.55, 0.105, 0.070, 0.330, 0.05),    # tail
    ]
    rings = [ring(x, hw, y0, y1, c) for x, hw, y0, y1, c in stations]
    return loft(rings)


def sidepod(sign: float) -> list[tuple[Vec, Vec, Vec]]:
    """One sidepod: a duct-shaped mass that sets up the underbody and the wheel wake."""
    stations = [
        (-0.55, 0.16, 0.10, 0.44),
        (-0.10, 0.34, 0.09, 0.52),
        (0.45, 0.36, 0.08, 0.50),
        (1.05, 0.28, 0.08, 0.42),
        (1.65, 0.16, 0.09, 0.32),
    ]
    rings: list[list[Vec]] = []
    for x, half, y0, y1 in stations:
        cz = sign * (0.34 + half * 0.55)
        pts = [
            (x, y0, cz - half * 0.5),
            (x, y0, cz + half * 0.5),
            (x, y1, cz + half * 0.42),
            (x, y1, cz - half * 0.42),
        ]
        rings.append(pts)
    return loft(rings)


def wing(x0: float, x1: float, y0: float, y1: float, half_span: float, endplate: float) -> list[tuple[Vec, Vec, Vec]]:
    """A wing element with two endplates. Flat plates: the wake is set by the span and the gap."""
    tris = box(x0, x1, y0, y1, -half_span, half_span)
    t = 0.022
    # The endplates overlap the element rather than butting against it. A face
    # that lands exactly on another face gives the mesher an edge shared by four
    # triangles, and it refuses that surface as non-manifold - correctly, since
    # inside/outside classification has no answer for it.
    bite = 0.018
    for s in (-1.0, 1.0):
        z = s * half_span
        inner = z - s * bite
        outer = inner + s * (t + bite)
        tris += box(x0 - 0.10, x1 + 0.14, y0 - endplate * 0.35, y1 + endplate * 0.65,
                    min(inner, outer), max(inner, outer))
    return tris


def floor_and_diffuser() -> list[tuple[Vec, Vec, Vec]]:
    """A flat floor that ramps up at the back — where most of the downforce is."""
    tris: list[tuple[Vec, Vec, Vec]] = []
    z = 0.66
    tris += box(-1.60, 1.90, RIDE, RIDE + 0.028, -z, z)
    # The ramp starts inside the floor for the same reason the endplates bite
    # into the wing: coincident faces are not a surface.
    stations = [(1.84, RIDE), (2.35, RIDE + 0.10), (2.80, RIDE + 0.26)]
    rings = []
    for x, y in stations:
        rings.append([(x, y, -z), (x, y, z), (x, y + 0.028, z), (x, y + 0.028, -z)])
    tris += loft(rings)
    return tris


def build() -> list[tuple[Vec, Vec, Vec]]:
    tris: list[tuple[Vec, Vec, Vec]] = []
    tris += body()
    tris += sidepod(+1.0)
    tris += sidepod(-1.0)
    tris += floor_and_diffuser()

    # Front wing: low, wide, ahead of the axle.
    tris += wing(-3.05, -2.55, RIDE + 0.055, RIDE + 0.085, WIDTH / 2, 0.30)
    tris += wing(-2.62, -2.30, RIDE + 0.115, RIDE + 0.145, WIDTH / 2 - 0.06, 0.26)
    # Rear wing: high, narrower, behind the rear axle.
    tris += wing(2.40, 2.92, 0.780, 0.815, 0.52, 0.36)
    tris += wing(2.62, 2.96, 0.895, 0.925, 0.50, 0.30)
    # Rear wing pylon.
    tris += box(2.33, 2.78, 0.34, 0.802, -0.05, 0.05)

    front_x = -WHEELBASE / 2
    rear_x = WHEELBASE / 2
    for sign in (-1.0, 1.0):
        tris += wheel(front_x, sign * (WIDTH / 2 - 0.18), 0.330, 0.305)
        tris += wheel(rear_x, sign * (WIDTH / 2 - 0.16), 0.360, 0.395)
    return tris


def normal(t: tuple[Vec, Vec, Vec]) -> Vec:
    (ax, ay, az), (bx, by, bz), (cx, cy, cz) = t
    ux, uy, uz = bx - ax, by - ay, bz - az
    vx, vy, vz = cx - ax, cy - ay, cz - az
    nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    n = math.sqrt(nx * nx + ny * ny + nz * nz)
    return (0.0, 0.0, 0.0) if n == 0 else (nx / n, ny / n, nz / n)


def write_stl(path: Path, tris: list[tuple[Vec, Vec, Vec]], name: str = "iterations racecar") -> None:
    header = name.encode("ascii", "replace")[:80].ljust(80, b"\0")
    with path.open("wb") as f:
        f.write(header)
        f.write(struct.pack("<I", len(tris)))
        for t in tris:
            f.write(struct.pack("<3f", *normal(t)))
            for v in t:
                f.write(struct.pack("<3f", *v))
            f.write(struct.pack("<H", 0))


def fit(tris: list[tuple[Vec, Vec, Vec]], length: float, nose_x: float, centre_z: float, ground_y: float) -> list[tuple[Vec, Vec, Vec]]:
    """
    Scale and place the car inside a wind tunnel of a given size.

    The `big` mesh preset is a unit cube with inlet, outlet, floor and roof
    patches - a tunnel, but a one-metre one. A car has to be scaled into it,
    and where it sits decides what the case can show: too far forward and the
    inlet profile is still developing over the nose, too far back and the wake
    leaves through the outlet before it has done anything.
    """
    xs = [v[0] for t in tris for v in t]
    ys = [v[1] for t in tris for v in t]
    zs = [v[2] for t in tris for v in t]
    s = length / (max(xs) - min(xs))
    dx = nose_x - min(xs) * s
    dy = ground_y - min(ys) * s
    dz = centre_z - ((min(zs) + max(zs)) / 2) * s
    return [tuple((v[0] * s + dx, v[1] * s + dy, v[2] * s + dz) for v in t) for t in tris]  # type: ignore[misc]


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "racecar.stl")
    tris = build()
    # `racecar_stl.py <out> [length nose_x centre_z ground_y]`
    if len(sys.argv) > 2:
        length, nose_x, centre_z, ground_y = (float(a) for a in sys.argv[2:6])
        tris = fit(tris, length, nose_x, centre_z, ground_y)
    degenerate = sum(1 for t in tris if normal(t) == (0.0, 0.0, 0.0))
    out.parent.mkdir(parents=True, exist_ok=True)
    write_stl(out, tris)

    xs = [v[0] for t in tris for v in t]
    ys = [v[1] for t in tris for v in t]
    zs = [v[2] for t in tris for v in t]
    print(f"{out}: {len(tris)} triangles, {degenerate} degenerate")
    print(f"  bounds  x [{min(xs):.3f}, {max(xs):.3f}]  y [{min(ys):.3f}, {max(ys):.3f}]  z [{min(zs):.3f}, {max(zs):.3f}]")
    print(f"  length {max(xs) - min(xs):.2f} m   width {max(zs) - min(zs):.2f} m   height {max(ys) - min(ys):.2f} m")
    return 1 if degenerate else 0


if __name__ == "__main__":
    raise SystemExit(main())
