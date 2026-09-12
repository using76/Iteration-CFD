# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
#
# make_box_sphere_stl.py [out.stl] [radius] [cx cy cz] [n_lat]
#
# Writes the ONE closed surface box_sphere.json meshes: an icosphere-free
# latitude/longitude sphere, ASCII STL, solid name `sphere`. A generator rather
# than a checked-in binary asset, so the geometry is readable and the repository
# carries no opaque file. Default: radius 1 m at (5, 5, 5), 48 latitude bands
# (9024 triangles), which is fine enough that a level-3 leaf (0.125 m at
# base_size 1.0) never spans two of its triangles.
import math
import sys

out = sys.argv[1] if len(sys.argv) > 1 else "box_sphere.stl"
r = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
cx = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
cy = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0
cz = float(sys.argv[5]) if len(sys.argv) > 5 else 5.0
nlat = int(sys.argv[6]) if len(sys.argv) > 6 else 48
nlon = 2 * nlat


def p(i, j):
    """Point on band i (0 = north pole, nlat = south pole), meridian j."""
    th = math.pi * i / nlat
    ph = 2.0 * math.pi * (j % nlon) / nlon
    return (cx + r * math.sin(th) * math.cos(ph),
            cy + r * math.sin(th) * math.sin(ph),
            cz + r * math.cos(th))


def tri(f, a, b, c):
    """One facet, wound so the normal points OUT of the sphere."""
    ux, uy, uz = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    vx, vy, vz = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    nx, ny, nz = (uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)
    m = math.sqrt(nx * nx + ny * ny + nz * nz)
    if m < 1e-300:
        return 0
    f.write("  facet normal %.9e %.9e %.9e\n    outer loop\n" % (nx / m, ny / m, nz / m))
    for q in (a, b, c):
        f.write("      vertex %.9e %.9e %.9e\n" % q)
    f.write("    endloop\n  endfacet\n")
    return 1


n = 0
with open(out, "w", encoding="ascii") as f:
    f.write("solid sphere\n")
    for i in range(nlat):
        for j in range(nlon):
            a, b = p(i, j), p(i, j + 1)
            c, d = p(i + 1, j + 1), p(i + 1, j)
            if i == 0:                      # the north cap is a triangle fan
                n += tri(f, a, c, d)
            elif i == nlat - 1:             # ... and so is the south cap
                n += tri(f, a, b, c)
            else:
                n += tri(f, a, b, c)
                n += tri(f, a, c, d)
    f.write("endsolid sphere\n")
print("%s: %d triangles, sphere r=%g at (%g, %g, %g)" % (out, n, r, cx, cy, cz))
