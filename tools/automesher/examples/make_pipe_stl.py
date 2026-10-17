# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
#
# make_pipe_stl.py [out.stl] [R] [L] [n_theta] [n_x]
#
# Writes the ONE closed surface pipe_tet.json meshes: a straight pipe of
# radius R and length L along +x, ASCII STL, three solids - `wall`, `inlet`
# and `outlet` - which stay separate patches in the mesh. A generator rather
# than a checked-in binary asset, so the geometry is readable and the
# repository carries no opaque file. Default: R 0.5 m, L 12 m, 64 facets
# round, 96 stations.
"""The shell is wound INWARD: every normal points into the fluid and the
enclosed volume is -(n_theta/2) R^2 sin(2 pi/n_theta) L. SPEC-LIT 116.9's
order 5 refuses such a cavity shell, EXCEPT when the carve keeps the fluid
from a seed (SPEC-LIT 116.12, castellation.keep_region "seed", TET-14): the
cavity IS the internal flow, so pipe_tet.json sets that and its refusal is
lifted. A seed on the solid side is still refused, by name, at the carve."""
import math
import sys

out = sys.argv[1] if len(sys.argv) > 1 else "pipe_tet.stl"
r = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
length = float(sys.argv[3]) if len(sys.argv) > 3 else 12.0
ntheta = int(sys.argv[4]) if len(sys.argv) > 4 else 64
nst = int(sys.argv[5]) if len(sys.argv) > 5 else 96


def p(i, j):
    """Station i along the axis, facet j round (taken mod ntheta)."""
    ph = 2.0 * math.pi * (j % ntheta) / ntheta
    return (length * i / nst, r * math.cos(ph), r * math.sin(ph))


def tri(f, a, b, c):
    """One facet (a, b, c); the facet normal is the right-hand normal of
    the order written, which here points INTO the fluid."""
    ux, uy, uz = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    vx, vy, vz = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    qx, qy, qz = (uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)
    m = math.sqrt(qx * qx + qy * qy + qz * qz)
    if m < 1e-300:
        return 0
    f.write("  facet normal %.9e %.9e %.9e\n    outer loop\n" % (qx / m, qy / m, qz / m))
    for q in (a, b, c):
        f.write("      vertex %.9e %.9e %.9e\n" % q)
    f.write("    endloop\n  endfacet\n")
    return 1


n = 0
with open(out, "w", encoding="ascii") as f:
    f.write("solid wall\n")
    for i in range(nst):
        for j in range(ntheta):
            a, b = p(i, j), p(i + 1, j)
            c, d = p(i + 1, j + 1), p(i, j + 1)
            n += tri(f, a, b, c)
            n += tri(f, a, c, d)
    f.write("endsolid wall\n")
    f.write("solid inlet\n")
    o = (0.0, 0.0, 0.0)
    for j in range(ntheta):
        n += tri(f, o, p(0, j), p(0, j + 1))
    f.write("endsolid inlet\n")
    f.write("solid outlet\n")
    e = (length, 0.0, 0.0)
    for j in range(ntheta):
        n += tri(f, e, p(nst, j + 1), p(nst, j))
    f.write("endsolid outlet\n")
v = -(ntheta / 2.0) * r * r * math.sin(2.0 * math.pi / ntheta) * length
print("%s: %d triangles, pipe r=%g l=%g, %d around x %d stations, inward, volume %g"
      % (out, n, r, length, ntheta, nst, v))
