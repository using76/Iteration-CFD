#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""vtu_read.py - check an `io::vtu` point file against the caller's counts.

`vtk` (BSD-3) is used as an installed library when present, through its
published Python API only; none of its source was read. The builtin parser
reads the same appended-binary layout straight from the published format
description, so the file checks out on machines without `vtk` too; when both
are available both run and the two must agree.

No GPL-licensed source was consulted.
"""

import argparse
import json
import struct
import sys
import xml.etree.ElementTree as ET

try:
    import vtk
except ImportError:  # the builtin parser below covers this machine
    vtk = None

MARKER = b'<AppendedData encoding="raw">\n_'


def read_builtin(path):
    """Parse an appended-binary `.vtu` with the stdlib alone.

    Finds the `<AppendedData encoding="raw">` marker, parses the XML head
    (everything before it, plus the closing `</VTKFile>`) with ElementTree,
    and for every `format="appended"` DataArray reads the little-endian
    UInt64 length and the payload that follows. Raises ValueError on
    anything malformed.
    """
    with open(path, "rb") as f:
        blob = f.read()
    i = blob.find(MARKER)
    if i < 0:
        raise ValueError("no <AppendedData encoding=\"raw\"> marker")
    root = ET.fromstring(blob[:i] + b"</VTKFile>")
    if root.get("header_type") != "UInt64":
        raise ValueError(f"header_type {root.get('header_type')!r} is not UInt64")
    end = blob.find(b"\n  </AppendedData>", i)
    if end < 0:
        raise ValueError("no closing </AppendedData>")
    appended = blob[i + len(MARKER):end]

    def block(offset):
        (n,) = struct.unpack_from("<Q", appended, offset)
        return appended[offset + 8:offset + 8 + n]

    piece = root.find(".//Piece")
    if piece is None:
        raise ValueError("no <Piece> in the XML head")

    def section(tag, into):
        node = piece.find(tag)
        if node is None:
            return
        for da in node.findall("DataArray"):
            name = da.get("Name")
            if da.get("format") != "appended":
                raise ValueError(f"DataArray {name} is not format=appended")
            width = {"Float64": 8, "Int64": 8, "UInt8": 1}.get(da.get("type"))
            if width is None:
                raise ValueError(f"DataArray {name}: unsupported type {da.get('type')}")
            ncomp = int(da.get("NumberOfComponents", "1"))
            payload = block(int(da.get("offset")))
            if len(payload) % (width * ncomp):
                raise ValueError(f"DataArray {name}: payload is not whole tuples")
            into[name] = (ncomp, da.get("type"), payload)

    cell_data, point_data = {}, {}
    section("CellData", cell_data)
    section("PointData", point_data)
    return (
        int(piece.get("NumberOfPoints")),
        int(piece.get("NumberOfCells")),
        cell_data,
        point_data,
    )


def read_vtk(path):
    """Read the same file through the installed `vtk` library."""
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    grid = reader.GetOutput()

    def arrays(container):
        out = {}
        for k in range(container.GetNumberOfArrays()):
            a = container.GetArray(k)
            vals = []
            for t in range(a.GetNumberOfTuples()):
                vals.extend(a.GetTuple(t))
            out[a.GetName()] = (a.GetNumberOfComponents(), "vtk", vals)
        return out

    return (
        grid.GetNumberOfPoints(),
        grid.GetNumberOfCells(),
        arrays(grid.GetCellData()),
        arrays(grid.GetPointData()),
    )


def asymmetry(entry):
    """`max |T_ij - T_ji|` and `max |T_ij|` over every 9-component tuple."""
    ncomp, kind, payload = entry
    if ncomp != 9:
        raise ValueError(f"expected 9 components, got {ncomp}")
    if kind == "vtk":
        vals = payload
    elif kind == "Float64":
        vals = struct.unpack(f"<{len(payload) // 8}d", payload)
    else:
        raise ValueError(f"needs a Float64 array, got {kind}")
    worst, scale = 0.0, 0.0
    for t in range(len(vals) // 9):
        nine = vals[9 * t:9 * t + 9]
        for r in range(3):
            for c in range(3):
                worst = max(worst, abs(nine[3 * r + c] - nine[3 * c + r]))
                scale = max(scale, abs(nine[3 * r + c]))
    return worst, scale


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="check a `.vtu` against expected counts")
    ap.add_argument("file")
    ap.add_argument("--points", type=int, default=None, metavar="N")
    ap.add_argument("--cells", type=int, default=None, metavar="N")
    ap.add_argument("--cell", action="append", default=[], metavar="NAME:NCOMP")
    ap.add_argument("--point", action="append", default=[], metavar="NAME:NCOMP")
    ap.add_argument("--symmetric", action="append", default=[], metavar="NAME")
    args = ap.parse_args()

    try:
        points, cells, cell_data, point_data = read_builtin(args.file)
    except (ValueError, OSError, ET.ParseError, struct.error) as e:
        print(f"malformed file: {e}", file=sys.stderr)
        return 2

    used = "builtin"
    if vtk is not None:
        try:
            vpoints, vcells, vcell, vpoint = read_vtk(args.file)
        except Exception as e:
            print(f"vtk reader failed: {e}", file=sys.stderr)
            return 2
        used = "vtk+builtin"
        if (vpoints, vcells) != (points, cells):
            print(
                f"readers disagree on counts: vtk says {(vpoints, vcells)}, "
                f"builtin says {(points, cells)}",
                file=sys.stderr,
            )
            return 1

    problems = []
    if args.points is not None and points != args.points:
        problems.append(f"--points {args.points}: file has {points}")
    if args.cells is not None and cells != args.cells:
        problems.append(f"--cells {args.cells}: file has {cells}")

    for wanted, got, label in (
        (args.cell, cell_data, "--cell"),
        (args.point, point_data, "--point"),
    ):
        for spec in wanted:
            name, _, comp = spec.partition(":")
            ncomp = int(comp)
            if name not in got:
                problems.append(f"{label} {name}: no such array (has {sorted(got)})")
            elif got[name][0] != ncomp:
                problems.append(
                    f"{label} {name}:{ncomp}: file has {got[name][0]} component(s)"
                )

    asymmetry_map = {}
    for name in args.symmetric:
        entry = cell_data.get(name) or point_data.get(name)
        if entry is None:
            problems.append(f"--symmetric {name}: no such array")
            continue
        try:
            worst, scale = asymmetry(entry)
        except ValueError as e:
            problems.append(f"--symmetric {name}: {e}")
            continue
        asymmetry_map[name] = worst
        if worst > 1e-12 * scale:
            problems.append(
                f"--symmetric {name}: max |T - T^T| = {worst:e} against max |T| = {scale:e}"
            )

    print(json.dumps({
        "reader": used,
        "points": points,
        "cells": cells,
        "cell_data": {k: v[0] for k, v in cell_data.items()},
        "point_data": {k: v[0] for k, v in point_data.items()},
        "asymmetry": asymmetry_map,
    }))
    if problems:
        print("; ".join(problems), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
