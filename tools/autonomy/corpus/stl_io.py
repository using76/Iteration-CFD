#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""stl_io - the corpus's pure-numpy ASCII STL writer and in-memory topology
checks.  The writers of docs/15 §E families A-F share this module so every
generated file has one byte format: "%.9e" numbers, "\n" line ends, one
space between fields, shared corners printed from one point array so they
are bit-identical, and +0.0 canonical everywhere (IEEE -0.0 + 0.0 = +0.0),
so no "-0." can reach a file.  It also carries the topology oracles the
generators self-judge with: per-edge open / non-manifold / same-direction
counts, the Euler characteristic, the signed volume, the shortest edge,
and check_closed, which raises instead of repairing.

    python tools/autonomy/corpus/stl_io.py --selftest
"""
from __future__ import annotations

import hashlib
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))

SOLID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def canonical(a):
    """-0.0 + 0.0 is +0.0: the one place signed zeros are removed."""
    return np.asarray(a, dtype=np.float64) + 0.0


def unit_normals(P, T):
    """(nT, 3) outward unit normals from the corner order of each triangle."""
    P = canonical(P)
    T = np.asarray(T, dtype=np.int64)
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    n = np.cross(b - a, c - a)
    norm = np.linalg.norm(n, axis=1)
    for i, z in enumerate(norm):
        if z == 0.0:
            raise ValueError("degenerate triangle %d: zero area" % i)
    return canonical(n / norm[:, None])


def stl_bytes(P, T, solid: str) -> bytes:
    """The whole ASCII STL as one str joined and ascii-encoded.

    solid <solid> / facet normal / outer loop / 3 x vertex / endloop /
    endfacet / endsolid, numbers "%.9e", "\n" ends, corners in T's order.
    """
    P = canonical(P)
    T = np.asarray(T, dtype=np.int64)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("points: expected (nP, 3), got %r" % (P.shape,))
    if not np.isfinite(P).all():
        bad = int(np.nonzero(~np.isfinite(P).all(axis=1))[0][0])
        raise ValueError("points: non-finite coordinate at point %d" % bad)
    nP = len(P)
    if T.ndim != 2 or T.shape[1] != 3:
        raise ValueError("triangles: expected (nT, 3), got %r" % (T.shape,))
    if len(T) and (int(T.min()) < 0 or int(T.max()) >= nP):
        raise ValueError("triangles: index %d outside [0, %d)"
                         % (int(T.max() if T.min() >= 0 else T.min()), nP))
    if not SOLID_RE.match(solid):
        raise ValueError("solid: %r does not match ^[A-Za-z][A-Za-z0-9_]*$" % solid)
    N = unit_normals(P, T)
    parts = ["solid %s\n" % solid]
    for i in range(len(T)):
        parts.append("facet normal %.9e %.9e %.9e\n" % (N[i, 0], N[i, 1], N[i, 2]))
        parts.append("outer loop\n")
        for j in range(3):
            p = P[T[i, j]]
            parts.append("vertex %.9e %.9e %.9e\n" % (p[0], p[1], p[2]))
        parts.append("endloop\n")
        parts.append("endfacet\n")
    parts.append("endsolid %s\n" % solid)
    data = "".join(parts).encode("ascii")
    assert b"-0." not in data, "a signed zero escaped canonical()"
    return data


def write_stl(path: str, P, T, solid: str) -> str:
    """Write stl_bytes to path; return the sha256 hex of what is on disk."""
    data = stl_bytes(P, T, solid)
    with open(path, "wb") as f:
        f.write(data)
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def edge_report(T, nP: int) -> dict:
    """{"open", "non_manifold", "same_direction"} per undirected edge."""
    T = np.asarray(T, dtype=np.int64)
    if len(T) == 0:
        return {"open": 0, "non_manifold": 0, "same_direction": 0}
    a = T[:, [0, 1, 2]].reshape(-1)
    b = T[:, [1, 2, 0]].reshape(-1)
    key = np.minimum(a, b) * np.int64(nP) + np.maximum(a, b)
    _uniq, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    fwd = np.bincount(inv[a < b], minlength=len(counts))
    open_e = int(np.count_nonzero(counts == 1))
    same = int(np.count_nonzero((counts == 2) & ((fwd == 2) | (fwd == 0))))
    nm = int(np.count_nonzero(counts >= 3))
    return {"open": open_e, "non_manifold": nm, "same_direction": same}


def signed_volume(P, T) -> float:
    """Sum P[a] . (P[b] x P[c]) / 6: positive when wound outward."""
    P = canonical(P)
    T = np.asarray(T, dtype=np.int64)
    pa, pb, pc = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    return float(np.einsum("ij,ij->i", pa, np.cross(pb, pc)).sum() / 6.0)


def _undirected_count(T, nP: int) -> int:
    T = np.asarray(T, dtype=np.int64)
    a = T[:, [0, 1, 2]].reshape(-1)
    b = T[:, [1, 2, 0]].reshape(-1)
    key = np.minimum(a, b) * np.int64(nP) + np.maximum(a, b)
    return int(len(np.unique(key)))


def euler_characteristic(P, T) -> int:
    """V - E + F from the undirected edge count."""
    P = np.asarray(P, dtype=np.float64)
    T = np.asarray(T, dtype=np.int64)
    return int(len(P) - _undirected_count(T, len(P)) + len(T))


def min_edge_length(P, T) -> float:
    """The shortest of all triangle edges."""
    P = canonical(P)
    T = np.asarray(T, dtype=np.int64)
    ds = np.stack([P[T[:, 1]] - P[T[:, 0]],
                   P[T[:, 2]] - P[T[:, 1]],
                   P[T[:, 0]] - P[T[:, 2]]])
    return float(np.linalg.norm(ds, axis=2).min())


def check_closed(P, T) -> None:
    """Raise RuntimeError naming the first failure; never repair."""
    P = np.asarray(P, dtype=np.float64)
    T = np.asarray(T, dtype=np.int64)
    nP = len(P)
    used = np.unique(T)
    if len(used) != nP or int(used[0]) != 0 or int(used[-1]) != nP - 1:
        raise RuntimeError("surface not closed: %d of %d point(s) used"
                           % (len(used), nP))
    er = edge_report(T, nP)
    if er["open"] or er["non_manifold"] or er["same_direction"]:
        raise RuntimeError("surface not closed: edge_report %s" % er)
    chi = euler_characteristic(P, T)
    if chi != 2:
        raise RuntimeError("surface not closed: euler characteristic %d != 2" % chi)
    vol = signed_volume(P, T)
    if not vol > 0.0:
        raise RuntimeError("surface not closed: signed volume %.6g <= 0" % vol)


# --- the selftest ---------------------------------------------------------

def _selftest():
    tet_P = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    tet_T = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)

    def group_writer():
        data = stl_bytes(tet_P, tet_T, "tet")
        lines = data.decode("ascii").split("\n")[:-1]
        assert len(lines) == 30, "expected 30 lines, got %d" % len(lines)
        assert lines[0] == "solid tet", lines[0]
        assert lines[1] == ("facet normal 0.000000000e+00 0.000000000e+00 "
                            "-1.000000000e+00"), lines[1]
        assert lines[2] == "outer loop", lines[2]
        assert lines[3] == ("vertex 0.000000000e+00 0.000000000e+00 "
                            "0.000000000e+00"), lines[3]
        assert lines[4] == ("vertex 0.000000000e+00 1.000000000e+00 "
                            "0.000000000e+00"), lines[4]
        assert lines[-1] == "endsolid tet", lines[-1]
        assert b"\r" not in data, "a carriage return reached the file"
        import tempfile
        d = tempfile.mkdtemp(prefix="stlio_")
        try:
            path = os.path.join(d, "tet.stl")
            got = write_stl(path, tet_P, tet_T, "tet")
            with open(path, "rb") as f:
                assert hashlib.sha256(f.read()).hexdigest() == got
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)
        return "30 lines, exact bytes, sha256 of the file on disk"

    def group_canonical():
        assert "%.9e" % -0.0 == "-0.000000000e+00", "why the rule exists"
        assert not bool(np.signbit(float(canonical([-0.0])[0]))), "-0.0 survived"
        pneg = np.where(tet_P == 0.0, -0.0, tet_P)
        assert stl_bytes(pneg, tet_T, "tet") == stl_bytes(tet_P, tet_T, "tet"), \
            "-0.0 input changed the bytes"
        return "+0.0 canonical: -0.0 corners write as the +0.0 bytes"

    def group_refusals():
        bad = []
        t0 = tet_T.copy()
        t0[0] = [0, 0, 0]
        for name, P2, T2, why in (
                ("zero area", tet_P, t0, "zero area"),
                ("NaN coordinate", None, tet_T, "non-finite"),
                ("index out of range", tet_P, np.array([[0, 1, 4], [0, 1, 2],
                                                        [0, 2, 3], [1, 2, 3]]),
                 "outside"),
                ("bad solid name", tet_P, tet_T, "does not match")):
            if name == "NaN coordinate":
                P2 = tet_P.copy()
                P2[0, 0] = float("nan")
            try:
                stl_bytes(P2, T2, "tet" if name != "bad solid name" else "1bad")
            except ValueError as e:
                assert why in str(e), "%s: %s" % (name, e)
                bad.append(name)
                continue
            raise AssertionError("%s: stl_bytes did not refuse" % name)
        return "4 by name: %s" % ", ".join(bad)

    def group_topology():
        er = edge_report(tet_T, 4)
        assert er == {"open": 0, "non_manifold": 0, "same_direction": 0}, er
        assert euler_characteristic(tet_P, tet_T) == 2
        assert abs(signed_volume(tet_P, tet_T) - 1.0 / 6.0) <= 1e-15
        removed = tet_T[1:]
        assert edge_report(removed, 4)["open"] == 3, "one face removed"
        flipped = tet_T.copy()
        flipped[3] = [1, 3, 2]
        assert edge_report(flipped, 4)["same_direction"] == 3, "one face flipped"
        dup = np.vstack([tet_T, tet_T[3]])
        assert edge_report(dup, 4)["non_manifold"] == 3, "one face duplicated"
        sys.path.insert(0, os.path.join(REPO, "tools", "geom"))
        import stl_repair
        for T2, want in ((removed, (3, 0)), (flipped, (0, 3)), (dup, (0, 3))):
            got = stl_repair._defects(np.asarray(T2, dtype=np.int64), 4)
            assert got == want, "_defects %r != %r" % (got, want)
        return ("closed tet 0/0/0, euler 2, V = 1/6; removed open 3, "
                "flipped same-direction 3, duplicated non-manifold 3; "
                "stl_repair._defects agrees on all three")

    groups = [("writer", group_writer), ("canonical zero", group_canonical),
              ("refusals", group_refusals), ("topology", group_topology)]
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


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="stl_io", description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()
    ap.error("only --selftest is implemented")
    return 2


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    sys.exit(main())
