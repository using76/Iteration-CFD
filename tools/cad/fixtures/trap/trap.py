#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""trap.py - the trap body of docs/16a §G, in metres: a poly5 contraction (R_i 30, R_e 10, L 30 mm) interpolated through 201 points with +x end tangents, a 3 mm RADIAL wall and a 10 mm exit tube, revolved 360 degrees about +x; its Pappus volume is 4560*pi mm^3 because the integral of r over the contraction is (R_i + R_e) L / 2.
"""
import math

import cadquery as cq

V = cq.Vector
RI, RE, L, W, LX = 0.030, 0.010, 0.030, 0.003, 0.010    # m
N = 200                                                  # spline spans; N + 1 interpolation points
VOLUME_M3 = 4560.0 * math.pi * 1e-9                      # Pappus truth


def r(x):
    xi = x / L
    return RI - (RI - RE) * (10 * xi ** 3 - 15 * xi ** 4 + 6 * xi ** 5)


def dr(x):
    xi = x / L
    return -(RI - RE) * (30 * xi ** 2 - 60 * xi ** 3 + 30 * xi ** 4) / L


def spline(off):
    return cq.Edge.makeSpline([V(L * k / N, r(L * k / N) + off, 0) for k in range(N + 1)],
                              tangents=[V(1, 0, 0), V(1, 0, 0)])


def meridian():
    wet = [spline(0.0), cq.Edge.makeLine(V(L, RE, 0), V(L + LX, RE, 0))]
    out = [spline(W), cq.Edge.makeLine(V(L, RE + W, 0), V(L + LX, RE + W, 0))]
    return wet, out


def build():
    wet, out = meridian()
    edges = [wet[0], wet[1], cq.Edge.makeLine(V(L + LX, RE, 0), V(L + LX, RE + W, 0)),
             out[1], out[0], cq.Edge.makeLine(V(0, RI + W, 0), V(0, RI, 0))]
    face = cq.Face.makeFromWires(cq.Wire.assembleEdges(edges))
    return cq.Solid.revolve(face, 360, V(0, 0, 0), V(1, 0, 0))


def wetted_face(body):
    fs = [f for f in body.Faces()
          if f.geomType() == "REVOLUTION" and f.BoundingBox().ymax < RI + W / 2]
    if len(fs) != 1:
        raise ValueError("wetted_face: %d REVOLUTION faces under y %g, expected 1" % (len(fs), RI + W / 2))
    return fs[0]
