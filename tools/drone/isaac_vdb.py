#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.

"""Write the drone flow .vdb in plain Python - numpy plus an openvdb module,
Isaac Sim's own when the interpreter has no other.

Run by tools/drone/isaac_export.py as a separate program:

  python isaac_vdb.py --npz <lattice.npz> --out <file.vdb> --manifest <json>

The write/read-back/manifest contract is DRONE-ISAAC run 1's vdb mode,
written fresh here so this file imports no bpy and keeps the repository's
Prosperity licence: every grid goes out dense (copyFromArray with tolerance
-1 activates every voxel, the all-zero grid too - Isaac's RTX volume path
wants a dense grid), under the linear transform whose row-vector matrix has
the voxel on the diagonal and lo as the translation row, so index (0, 0, 0)
lands on world lo; all grids go out in one openvdb.write; then every grid
is read back and measured into the manifest (active_voxels, source min/max,
roundtrip_max_abs, world_first/world_last) together with the openvdb
module's identity and library version and the written file's format
version. The version is why this writer exists: Isaac Sim 6.0 refuses file
format 225 - what Blender 5.1's bundled openvdb writes - and hangs; its own
omni.volume module writes 224 (both measured 2026-10-06).

The module search (load_openvdb): a plain `import openvdb` first; if the
interpreter has none, Isaac Sim's omni.volume extension openvdb, which only
loads after its DLL directories are set up; without either the tool refuses
DI-VDB and exits 2. At interpreter exit the omni.volume nanobind prints
"leaked function" lines on stderr - harmless, the exit code stays 0.

--selftest runs V1-V2. V1 needs an openvdb module: on a plain Python
without one the tool refuses, which is the point.
"""

import argparse
import glob
import json
import os
import shutil
import struct
import sys
import tempfile

import numpy as np

ISAAC_EXTSCACHE = "C:/iss/env/Lib/site-packages/isaacsim/extscache"
VDB_MAGIC = 0x56444220                                # " VDB" as a little-endian int64


def _die(msg):
    print("[isaac_vdb] ERROR: " + str(msg), flush=True)
    raise SystemExit(1)


def load_openvdb():
    """-> (module, source): the interpreter's own openvdb ("python-env")
    first, then Isaac Sim's omni.volume extension openvdb
    ("isaac-omni.volume", file format 224 - the one Isaac Sim 6.0 reads),
    which loads only after os.add_dll_directory on the extension's bin and
    bin/deps, then the sys.path insert. Neither -> refused DI-VDB, exit 2."""
    try:
        import openvdb
        return openvdb, "python-env"
    except ImportError:
        pass
    for ext in sorted(glob.glob(os.path.join(ISAAC_EXTSCACHE, "omni.volume-*"))):
        if not os.path.isdir(ext):
            continue
        for sub in ("bin", os.path.join("bin", "deps")):
            p = os.path.join(ext, sub)
            if os.path.isdir(p):
                os.add_dll_directory(p)
        sys.path.insert(0, ext)
        try:
            import openvdb
            return openvdb, "isaac-omni.volume"
        except ImportError:
            sys.path.remove(ext)
    print("refused: DI-VDB: no openvdb module", flush=True)
    raise SystemExit(2)


def vdb_file_version(path):
    """The little-endian uint32 at bytes 8..11 of a .vdb - the file format
    version Isaac's reader compares (bytes 0..7 are the magic as an int64).
    -1 when the file is shorter than 12 bytes or the magic does not match."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(12)
    except OSError:
        return -1
    if len(head) < 12:
        return -1
    magic, version = struct.unpack("<qI", head)
    if magic != VDB_MAGIC:
        return -1
    return int(version)


def _read_npz(npz_path):
    """lattice.npz -> (lo (3,) float64, voxel float, dims [3] int, names,
    {name: (dims,) float32 contiguous})."""
    data = np.load(npz_path, allow_pickle=False)
    lo = np.asarray(data["lo"], dtype=np.float64).reshape(3)
    voxel = float(data["voxel"])
    dims = [int(v) for v in np.asarray(data["dims"]).reshape(3)]
    names = [str(v) for v in np.asarray(data["grids"]).reshape(-1)]
    arrays = {}
    for name in names:
        arr = np.ascontiguousarray(data[name], dtype=np.float32)
        if list(arr.shape) != dims:
            _die(name + ": shape " + repr(arr.shape) + " != dims " + repr(dims))
        arrays[name] = arr
    return lo, voxel, dims, names, arrays


def write_vdb(npz_path, out_path, manifest_path):
    """Every grid of the lattice dense into one .vdb, then read back and
    measured into the manifest with file_version, openvdb, library_version."""
    vdb, source = load_openvdb()
    lo, voxel, dims, names, arrays = _read_npz(npz_path)
    grids = []
    for name in names:
        g = vdb.FloatGrid()
        g.copyFromArray(arrays[name], ijk=(0, 0, 0), tolerance=-1)
        m = [[0.0] * 4 for _ in range(4)]
        for i in range(3):
            m[i][i] = voxel
        m[3][0], m[3][1], m[3][2] = float(lo[0]), float(lo[1]), float(lo[2])
        m[3][3] = 1.0
        g.transform = vdb.createLinearTransform(matrix=m)
        g.name = name
        grids.append(g)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    vdb.write(out_path, grids=grids)
    loaded, _meta = vdb.readAll(out_path)
    byname = {str(g.name): g for g in loaded}
    per = []
    for name in names:
        g = byname.get(name)
        if g is None:
            _die("grid " + name + " did not read back from " + out_path)
        back = np.zeros(tuple(dims), dtype=np.float32)
        g.copyToArray(back, ijk=(0, 0, 0))
        src = arrays[name]
        per.append({
            "name": name,
            "active_voxels": int(g.activeVoxelCount()),
            "min": float(src.min()),
            "max": float(src.max()),
            "roundtrip_max_abs": float(np.abs(back - src).max()),
            "world_first": [float(v) for v in g.transform.indexToWorld((0, 0, 0))],
            "world_last": [float(v) for v in g.transform.indexToWorld(
                (dims[0] - 1, dims[1] - 1, dims[2] - 1))],
        })
    try:
        lib = [int(v) for v in vdb.LIBRARY_VERSION]
    except (AttributeError, TypeError):
        lib = None
    man = {"file": out_path.replace(os.sep, "/"),
           "bytes": int(os.path.getsize(out_path)),
           "dims": dims, "voxel": voxel, "lo": [float(v) for v in lo],
           "openvdb": source, "library_version": lib,
           "file_version": vdb_file_version(out_path),
           "grids": per}
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=2)
    print("[isaac_vdb] wrote " + out_path + " (" + str(len(grids)) + " grids, openvdb "
          + source + ", file format " + str(man["file_version"]) + ")", flush=True)


# ---------------------------------------------------------------- selftest

def _v1():
    """The tiny lattice through write_vdb in the current interpreter: active
    8, round trip exact, the written file's format version a real number."""
    tmp = tempfile.mkdtemp(prefix="isaac-vdb-v1-")
    try:
        npz = os.path.join(tmp, "tiny.npz")
        np.savez(npz, lo=np.zeros(3, dtype=np.float64), voxel=np.float64(1.0),
                 dims=np.array([2, 2, 2], dtype=np.int64), grids=np.array(["a"]),
                 a=np.arange(8, dtype=np.float32).reshape(2, 2, 2))
        man = os.path.join(tmp, "m.json")
        write_vdb(npz, os.path.join(tmp, "tiny.vdb"), man)
        with open(man, encoding="utf-8") as fh:
            d = json.load(fh)
        assert [g["name"] for g in d["grids"]] == ["a"], d
        assert int(d["grids"][0]["active_voxels"]) == 8, d
        assert float(d["grids"][0]["roundtrip_max_abs"]) == 0.0, d
        assert isinstance(d["file_version"], int) and d["file_version"] > 0, d
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _v2():
    """vdb_file_version: the 12-byte header read, a short file and a wrong
    magic refused with -1."""
    tmp = tempfile.mkdtemp(prefix="isaac-vdb-v2-")
    try:
        p = os.path.join(tmp, "v224.vdb")
        with open(p, "wb") as fh:
            fh.write(struct.pack("<qI", VDB_MAGIC, 224))
        assert vdb_file_version(p) == 224, vdb_file_version(p)
        p = os.path.join(tmp, "short.vdb")
        with open(p, "wb") as fh:
            fh.write(b"VDB ")
        assert vdb_file_version(p) == -1, vdb_file_version(p)
        p = os.path.join(tmp, "magic.vdb")
        with open(p, "wb") as fh:
            fh.write(struct.pack("<qI", 0x12345678, 224))
        assert vdb_file_version(p) == -1, vdb_file_version(p)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def selftest():
    """V1-V2; V1 loads an openvdb module or the tool refuses (exit 2)."""
    tests = [("V1", _v1), ("V2", _v2)]
    npass = 0
    for name, fn in tests:
        try:
            fn()
            print("[ok] " + name, flush=True)
            npass += 1
        except Exception as exc:
            print("[FAIL] " + name + ": " + str(exc), flush=True)
    if npass == len(tests):
        print("SELFTEST PASS " + str(npass) + "/" + str(len(tests)), flush=True)
        return 0
    print("SELFTEST FAIL " + str(npass) + "/" + str(len(tests)), flush=True)
    return 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog="isaac_vdb.py",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("--npz")
    ap.add_argument("--out")
    ap.add_argument("--manifest")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    missing = [f for f in ("npz", "out", "manifest") if not getattr(args, f)]
    if missing:
        _die("needs --npz --out --manifest, missing: " + ", ".join(missing))
    write_vdb(args.npz, args.out, args.manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
