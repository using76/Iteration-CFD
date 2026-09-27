#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""usage: python tools/aero/drag_post.py <caseDir> <timeDir> <U_inf> [--rho R] [--aref A] [--band YMAX ZMIN] [--body NAME] [--outlet NAME] [--json OUT]

Two independent drag estimates from the written final-time fields of an
OpenFOAM ASCII case (moved from the F1 aero session's cases/ scripts,
docs/14 T3; the layout parsed here is the one this crate's own writers
produce - src/io/polymesh.rs and src/io/fields.rs):

  1. Pressure integration over the --body wall patch:
       F_x = sum_faces p_f * Sf_x
     Force on the body = +sum p*Sf: Sf points out of the fluid cell, i.e.
     into the body, so at the stagnation face Sf_x > 0 and p > p_inf and
     drag is +x. p_f is the patch's own value when the field carries one,
     else the owner cell's p (first order; the owner hugs the wall).
  2. Momentum-deficit (wake survey) over the --outlet patch:
       D = sum_faces rho * u_x * (U_inf - u_x) * |Sf_x|
     over the faces whose centre lies in the wake band |y| <= y_max,
     z >= z_min - the no-slip ground develops its own boundary layer, which
     is not the body's doing - skipping faces already at free stream
     (u_x >= 0.999 U_inf). u_x is the outlet's own value when the field
     carries one, else the owner cell's U.

With --json OUT the same numbers are written to a file as one JSON
document, never stdout.
"""
import argparse
import json
import re
import sys
from array import array
from pathlib import Path


def die(msg, code=1):
    sys.stderr.write('drag_post: %s\n' % msg)
    sys.stderr.flush()
    raise SystemExit(code)


# The point coordinates area_vector and momentum_deficit read; read_points
# fills it once and everything after measures against it.
pts = array("d")


def read_boundary(path):
    """boundary file: patch -> (startFace, nFaces, type)."""
    text = path.read_text()
    patches = {}
    for m in re.finditer(r"(\w+)\s*\{\s*type\s+(\w+);\s*nFaces\s+(\d+);\s*startFace\s+(\d+);", text):
        patches[m.group(1)] = (int(m.group(4)), int(m.group(3)), m.group(2))
    return patches


def stream_faces(path: Path, targets):
    """Faces are `N(p1 p2 ... pN)` and may wrap lines - token-parse complete
    `count(...)` groups through a chunked regex instead of trusting lines."""
    out = {name: [] for name, _, _ in targets}
    end_after = max(sf + nf for _, sf, nf in targets)
    idx = 0
    tail = ""
    pat = re.compile(r"(\d+)\(([^)]*)\)")
    with path.open() as fh:
        while True:
            chunk = fh.read(16 * 1024 * 1024)
            if not chunk:
                break
            data = tail + chunk
            last = 0
            for m in pat.finditer(data):
                idx += 1
                for name, sf, nf in targets:
                    if sf <= idx - 1 < sf + nf:
                        out[name].append(m.group(2))
                        break
                last = m.end()
                if idx >= end_after:
                    break
            if idx >= end_after:
                break
            tail = data[last:]
    return out


def stream_per_line(path: Path, targets):
    """owner/neighbour: one label per line, in face order."""
    out = {name: [] for name, _, _ in targets}
    n = 0
    in_list = False
    end_after = max(sf + nf for _, sf, nf in targets)
    with path.open() as fh:
        for line in fh:
            s = line.strip()
            if not in_list:
                if s == "(":
                    in_list = True
                continue
            for name, sf, nf in targets:
                if sf <= n < sf + nf:
                    out[name].append(int(s))
                    break
            n += 1
            if n >= end_after:
                break
    return out


def read_points(path):
    """points: `N\n(\n(x y z)\n...)`, one tuple per line."""
    ptext = path.read_text()
    m = re.search(r"\n\d+\s*\n\s*\(", ptext)
    for tup in re.finditer(r"\(([^)]+)\)", ptext[m.end():]):
        for v in tup.group(1).split():
            pts.append(float(v))
    return len(pts) // 3


def area_vector(point_ids):
    sx = sy = sz = 0.0
    n = len(point_ids)
    for i in range(n):
        a = point_ids[i] * 3
        b = point_ids[(i + 1) % n] * 3
        sx += (pts[a + 1] - pts[b + 1]) * (pts[a + 2] + pts[b + 2])
        sy += (pts[a + 2] - pts[b + 2]) * (pts[a] + pts[b])
        sz += (pts[a] - pts[b]) * (pts[a + 1] + pts[b + 1])
    return sx * 0.5, sy * 0.5, sz * 0.5


def read_field(path: Path, n_comp, body, outlet, nfaces):
    """internalField plus the two patches' `value` entries, every entry
    normalised on the way in: a `uniform` value becomes ('uniform', v) - a
    float, or one flat [x, y, z] for a vector - and a nonuniform list
    becomes ('list', vals) with one value per cell/face. value_at answers
    every later read, so no caller tests what kind of entry it got."""
    text = path.read_text()
    out = {"internal": None, body: None, outlet: None}
    mi = re.search(r"internalField\s+uniform\s+([^;]+);", text)
    if mi:
        tok = mi.group(1).strip()
        v = [float(v) for v in tok.strip("()").split()] if n_comp == 3 else float(tok)
        out["internal"] = ("uniform", v)
    else:
        mn = re.search(r"internalField\s+nonuniform\s+List<[^>]+>\s*\d+\s*\n\(", text)
        start = mn.end()
        end = text.index("\n)", start)
        vals = []
        if n_comp == 3:
            for tup in re.finditer(r"\(([^)]+)\)", text[start:end]):
                vals.append([float(v) for v in tup.group(1).split()])
        else:
            for line in text[start:end].splitlines():
                s = line.strip().strip("()")
                if s:
                    vals.append(float(s))
        out["internal"] = ("list", vals)
    bf = text[text.index("boundaryField"):]
    for name in (body, outlet):
        mm = re.search(name + r"\s*\{(.*?)\n\s*\}", bf, re.S)
        if not mm:
            continue
        block = mm.group(1)
        vu = re.search(r"value\s+uniform\s+([^;]+);", block)
        vn = re.search(r"value\s+nonuniform\s+List<[^>]+>\s*\d+\s*\n\((.*?)\n\)", block, re.S)
        if vn:
            vals = []
            for line in vn.group(1).splitlines():
                s = line.strip().strip("()")
                if s:
                    vals.append([float(v) for v in s.split()] if n_comp == 3 else float(s))
            entry = ("list", vals)
        elif vu:
            tok = vu.group(1).strip()
            entry = ("uniform", [float(v) for v in tok.strip("()").split()] if n_comp == 3 else float(tok))
        else:
            continue
        if entry[0] == "list" and len(entry[1]) != nfaces[name]:
            die("%s: patch %s carries %d value(s), the patch has %d faces"
                % (path, name, len(entry[1]), nfaces[name]))
        out[name] = entry
    return out


def value_at(entry, i):
    """The i-th value of a normalised entry: the value itself when the entry
    is uniform, the i-th list item otherwise."""
    kind, v = entry
    return v if kind == "uniform" else v[i]


def pressure_drag(faces_list, owners, p_int, p_patch):
    """Force on the body = +sum p*Sf (Sf points out of the fluid cell, i.e.
    into the body). At the stagnation face Sf_x > 0 and p > p_inf, so drag
    is +x. The face pressure is the patch's own value when the field
    carries one, else the owner cell's p."""
    Fx = 0.0
    for i, fids in enumerate(faces_list):
        sx, _, _ = area_vector(fids)
        if p_patch is not None:
            pf = value_at(p_patch, i)
        else:
            pf = value_at(p_int, owners[i])
        Fx += pf * sx
    return Fx


def momentum_deficit(faces_list, owners, u_int, u_out, u_inf, rho, y_max, z_min):
    """The no-slip ground develops its own boundary layer through the domain,
    which is NOT the body's doing - restrict the survey to the wake band
    |y| <= y_max, z >= z_min behind the body, and skip a face already at
    free stream. u_x is the outlet's own value when the field carries one,
    else the owner cell's U."""
    D_mom = 0.0
    used = 0
    for i, fids in enumerate(faces_list):
        sx, sy, sz = area_vector(fids)
        ax = abs(sx)
        # face centre y/z
        cy = cz = 0.0
        for pid in fids:
            cy += pts[pid * 3 + 1]
            cz += pts[pid * 3 + 2]
        cy /= len(fids)
        cz /= len(fids)
        if abs(cy) > y_max or cz < z_min:
            continue
        if u_out is not None:
            ux = value_at(u_out, i)[0]
        else:
            ux = value_at(u_int, owners[i])[0]
        if ux >= u_inf * 0.999:
            continue
        D_mom += rho * ux * (u_inf - ux) * ax
        used += 1
    return D_mom, used


def parse_args(argv):
    p = argparse.ArgumentParser(
        prog='drag_post',
        description="Two drag estimates from a written OpenFOAM ASCII case: "
                    "the pressure integral over a wall patch and the "
                    "momentum-deficit survey over an outlet patch.")
    p.add_argument('case_dir', help='the case directory (holds constant/polyMesh)')
    p.add_argument('time_dir', help='a directory path holding p and U')
    p.add_argument('u_inf', type=float, help='free-stream velocity [m/s]')
    p.add_argument('--rho', type=float, default=1.2041, help='density [kg/m^3]')
    p.add_argument('--aref', type=float, default=1.6022, help='reference area [m^2]')
    p.add_argument('--band', nargs=2, type=float, default=[1.8, 0.3], metavar=('YMAX', 'ZMIN'),
                   help='wake band: |y| <= YMAX and z >= ZMIN')
    p.add_argument('--body', default='car', help='the wall patch the pressure integral covers')
    p.add_argument('--outlet', default='outlet', help='the patch the wake survey covers')
    p.add_argument('--json', metavar='OUT',
                   help='write the drag document to this file, never stdout')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    case = Path(args.case_dir)
    time_dir = Path(args.time_dir)
    u_inf = args.u_inf
    body, outlet = args.body, args.outlet
    y_max, z_min = args.band
    pm = case / "constant" / "polyMesh"

    for f in ("boundary", "faces", "owner", "points"):
        if not (pm / f).is_file():
            die("missing file %s" % (pm / f))
    for f in ("p", "U"):
        if not (time_dir / f).is_file():
            die("missing file %s" % (time_dir / f))

    patches = read_boundary(pm / "boundary")
    print("patches:", {k: v[:2] for k, v in patches.items()})
    if body not in patches:
        die("the boundary file has no patch named %s (patches: %s)" % (body, ", ".join(patches)))
    if outlet not in patches:
        die("the boundary file has no patch named %s (patches: %s)" % (outlet, ", ".join(patches)))
    body_sf, body_nf, _ = patches[body]
    out_sf, out_nf, _ = patches[outlet]
    targets = [(body, body_sf, body_nf), (outlet, out_sf, out_nf)]

    faces_raw = stream_faces(pm / "faces", targets)
    faces = {name: [[int(x) for x in re.findall(r"\d+", f)] for f in faces_raw[name]] for name, _, _ in targets}
    print(f"{body} faces: {len(faces[body])}, {outlet} faces: {len(faces[outlet])}")

    owner = {name: stream_per_line(pm / "owner", targets)[name] for name, _, _ in targets}

    print(f"points: {read_points(pm / 'points')}")

    print("reading p...")
    pfield = read_field(time_dir / "p", 1, body, outlet, {body: body_nf, outlet: out_nf})
    print("reading U...")
    ufield = read_field(time_dir / "U", 3, body, outlet, {body: body_nf, outlet: out_nf})

    Fx = pressure_drag(faces[body], owner[body], pfield["internal"], pfield[body])
    print(f"pressure drag Fx = {Fx:.2f} N")

    D_mom, used = momentum_deficit(faces[outlet], owner[outlet], ufield["internal"],
                                   ufield[outlet], u_inf, args.rho, y_max, z_min)
    print(f"momentum-deficit drag = {D_mom:.2f} N (wake faces: {used})")

    q = 0.5 * args.rho * u_inf ** 2
    print(f"\nq = 0.5*rho*U^2 = {q:.1f} Pa | A_ref = {args.aref} m^2 | rho = {args.rho} kg/m^3 | U = {u_inf} m/s")
    print(f"pressure drag  : {Fx:10.1f} N   Cd {Fx / (q * args.aref):.4f}")
    print(f"momentum drag  : {D_mom:10.1f} N   Cd {D_mom / (q * args.aref):.4f}")

    if args.json:
        doc = {
            "tool": "drag_post",
            "case": args.case_dir,
            "time_dir": args.time_dir,
            "u_inf": u_inf,
            "rho": args.rho,
            "a_ref": args.aref,
            "q_pa": q,
            "band": {"y_max": y_max, "z_min": z_min},
            "body": {"patch": body, "faces": len(faces[body]),
                     "source": "value" if pfield[body] is not None else "owner",
                     "fx_n": Fx, "cd": Fx / (q * args.aref)},
            "wake": {"patch": outlet, "faces": len(faces[outlet]), "used": used,
                     "source": "value" if ufield[outlet] is not None else "owner",
                     "drag_n": D_mom, "cd": D_mom / (q * args.aref)},
        }
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=1)
    return 0


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.exit(main())
