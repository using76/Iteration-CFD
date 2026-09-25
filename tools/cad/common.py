#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""common.py - canonical JSON, sha256, atomic writes, fsynced jsonl and the env fingerprint of the CAD loop (docs/16 §D Reproducibility).

Usage:
  python common.py --selftest
  python common.py --digest FILE
  python common.py --env
"""

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))       # tools/cad
REPO = os.path.dirname(os.path.dirname(HERE))
FIXTURES = os.path.join(HERE, "fixtures")
NL = "\n"
NB = b"\n"
HEADER_COMMENT = ("meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). "
                  "Source-available, not Open Source. No GPL-licensed source was consulted.")
DIGEST20 = "181ba7663373e3a7dfe47e1af96624dea42039939c30e678d55dc38847a4b143"
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

_USAGE = ("usage: python common.py --selftest | --digest FILE | --env"
          + NL)


def _check(obj, path="$"):
    """Refuse anything json.dumps(allow_nan=False) would mis-serialise, naming the path."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                raise TypeError("canonical_json: %s: dict key %r is not a str" % (path, k))
            _check(v, "$." + k if path == "$" else path + "." + k)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _check(v, "%s[%d]" % (path, i))
    elif isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError("canonical_json: %s: non-finite number %r" % (path, obj))
    elif not isinstance(obj, (str, int, bool)) and obj is not None:
        raise TypeError("canonical_json: %s: %s is not JSON" % (path, type(obj).__name__))


def canonical_json(obj) -> str:
    """The one canonical form: sort_keys, tight separators, ascii, no NaN (docs/16 §D)."""
    _check(obj)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def canonical_bytes(obj) -> bytes:
    """canonical_json(obj).encode("ascii"): ensure_ascii keeps it ascii-safe."""
    return canonical_json(obj).encode("ascii")


def sha256_bytes(data: bytes) -> str:
    """Lowercase hex sha256 of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_of(obj) -> str:
    """sha256 over canonical_bytes(obj): the reproducibility key of docs/16 §D."""
    return sha256_bytes(canonical_bytes(obj))


def sha256_file(path: str) -> str:
    """sha256 of a file's bytes, read in 1 MiB chunks so big meshes stream."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def _package_version(name):
    """importlib.metadata.version(name), or None when the package is absent."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def env_fingerprint() -> dict:
    """The environment half of the docs/16 §D reproducibility key.

    Package versions come from importlib.metadata only - no package is imported.
    `occt` is DERIVED from the cadquery-ocp version (its first three dot-separated
    components name the OCCT release that OCP wraps), not read from OCCT itself.
    """
    ocp = _package_version("cadquery-ocp")
    occt = None
    if ocp is not None:
        parts = ocp.split(".")
        if len(parts) >= 3:
            occt = ".".join(parts[:3])
    return {
        "schema": "cad-env/1",
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": sys.platform,
        "machine": platform.machine(),
        "packages": {
            "cadquery": _package_version("cadquery"),
            "cadquery-ocp": ocp,
            "gmsh": _package_version("gmsh"),
            "numpy": _package_version("numpy"),
            "scikit-learn": _package_version("scikit-learn"),
            "scipy": _package_version("scipy"),
        },
        "occt": occt,
    }


def atomic_write(path: str, data) -> None:
    """Write bytes or str (str as UTF-8, no newline translation) through a same-directory temp file and os.replace."""
    if isinstance(data, str):
        blob = data.encode("utf-8")
    elif isinstance(data, (bytes, bytearray)):
        blob = bytes(data)
    else:
        raise TypeError("atomic_write: data is %s, not bytes or str" % type(data).__name__)
    target = os.path.abspath(path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + os.path.basename(target) + ".",
                               suffix=".tmp", dir=os.path.dirname(target))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(blob)
            f.flush()
            os.fsync(f.fileno())
        retries = 0
        while True:
            try:
                os.replace(tmp, target)
                break
            except PermissionError:
                retries += 1          # Windows scanners hold the new file briefly
                if retries > 5:
                    raise
                time.sleep(0.05)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def write_json(path: str, obj) -> None:
    """Serialise FIRST (a NaN raises before any file is touched), then atomic_write."""
    atomic_write(path, json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + NL)


def read_json(path: str):
    """json.load with UTF-8."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def jsonl_append(path: str, row: dict) -> None:
    """Append one canonical line + LF and fsync, so a crashed run leaves whole lines only."""
    if not isinstance(row, dict):
        raise TypeError("jsonl_append: row is %s, not a dict" % type(row).__name__)
    target = os.path.abspath(path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    line = canonical_bytes(row) + NB
    with open(target, "ab") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: str) -> list:
    """Every row of a jsonl file; a missing file reads as []; a broken line names it."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    rows = []
    lines = raw.split(NL)
    if lines and lines[-1] == "":
        lines.pop()                      # the final empty string after the last LF
    for i, line in enumerate(lines, start=1):
        try:
            rows.append(json.loads(line))
        except ValueError as e:
            raise ValueError("%s:%d: %s" % (path, i, e))
    return rows


def selftest() -> int:
    """Prove rows 1-6 of the unit: canonical, digest20, sha256, atomic_write, jsonl, env."""
    a = canonical_json({"b": 2, "a": 1})
    assert a == canonical_json({"a": 1, "b": 2}), "key order changed the canonical form"
    assert a == '{"a":1,"b":2}', a
    uj = canonical_json({"a": [3, 1, 2], "u": "노즐"})
    assert uj == '{"a":[3,1,2],"u":"' + chr(92) + 'ub178' + chr(92) + 'uc990"}', uj
    for bad in (float("nan"), float("inf"), float("-inf")):
        try:
            canonical_json({"x": bad})
        except ValueError as e:
            assert "$.x" in str(e), str(e)
        else:
            raise AssertionError("canonical_json accepted %r" % bad)
    for bad, tname in ((set(), "set"), (b"x", "bytes"), (object(), "object")):
        try:
            canonical_json({"s": bad})
        except TypeError as e:
            assert "$.s" in str(e) and tname in str(e), str(e)
        else:
            raise AssertionError("canonical_json accepted a %s" % tname)
    try:
        canonical_json({"d": {1: "x"}})
    except TypeError as e:
        assert "$.d" in str(e), str(e)
    else:
        raise AssertionError("canonical_json accepted a non-str dict key")
    try:
        canonical_json([1, {"d": {True: "x"}}])
    except TypeError as e:
        assert "$[1].d" in str(e), str(e)
    else:
        raise AssertionError("canonical_json accepted a non-str key at an index")
    assert canonical_json({"z": -0.0}) == '{"z":-0.0}', canonical_json({"z": -0.0})
    assert canonical_json({"t": 2.0}) == '{"t":2.0}', canonical_json({"t": 2.0})
    print("[ok] canonical: sorted, tight, ascii, NaN and type refusals name the path")

    dig = os.path.join(FIXTURES, "digest20.json")
    obj = read_json(dig)
    assert isinstance(obj, dict) and len(obj) == 20, \
        "digest20.json must have exactly 20 top-level keys, got %d" % len(obj)
    assert sha256_of(obj) == DIGEST20, \
        "in-process digest %s disagrees with the pinned constant" % sha256_of(obj)
    for seed in ("1", "987654321"):
        env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONIOENCODING="utf-8")
        p = subprocess.run([sys.executable, os.path.join(HERE, "common.py"), "--digest", dig],
                           capture_output=True, text=True, encoding="utf-8", errors="replace",
                           env=env, timeout=600)
        assert p.returncode == 0, "child with PYTHONHASHSEED=%s failed: %s" % (seed, p.stderr)
        assert p.stdout.strip() == DIGEST20, \
            "child with PYTHONHASHSEED=%s printed %s" % (seed, p.stdout.strip())
    print("[ok] digest20: in-process and two hash-seeded children agree with the pinned digest")

    assert sha256_bytes(b"") == EMPTY_SHA256, "the empty digest drifted"
    with tempfile.TemporaryDirectory() as td:
        big = os.path.join(td, "big.bin")
        payload = bytes(range(256)) * 10240          # 2.5 MiB: more than one 1 MiB chunk
        with open(big, "wb") as f:
            f.write(payload)
        assert len(payload) > 1048576
        assert sha256_file(big) == hashlib.sha256(payload).hexdigest(), \
            "chunked sha256_file disagrees with hashlib"
    print("[ok] sha256: chunked file digest equals hashlib; the empty digest matches")

    with tempfile.TemporaryDirectory() as td:
        pb = os.path.join(td, "blob.bin")
        atomic_write(pb, b"\x00\x01\xff")
        with open(pb, "rb") as f:
            assert f.read() == b"\x00\x01\xff", "bytes were not written verbatim"
        pt = os.path.join(td, "text.txt")
        atomic_write(pt, "a" + NL + "b")
        with open(pt, "rb") as f:
            assert f.read() == b"a" + NB + b"b", "newline translation happened"
        try:
            atomic_write(os.path.join(td, "int.bin"), 123)
        except TypeError:
            pass
        else:
            raise AssertionError("atomic_write accepted an int")
        assert not os.path.exists(os.path.join(td, "int.bin")), "TypeError touched the disk"
        atomic_write(pt, "c" + NL + "d")             # overwrite: the old temp must not linger
        leftovers = [n for n in os.listdir(td) if n.endswith(".tmp")]
        assert not leftovers, "temp files left behind: %r" % leftovers
        keep = os.path.join(td, "keep.json")
        atomic_write(keep, b"keep")
        try:
            write_json(keep, {"x": float("nan")})
        except ValueError:
            pass
        else:
            raise AssertionError("write_json accepted NaN")
        with open(keep, "rb") as f:
            assert f.read() == b"keep", "the refused write damaged the old file"
        real_replace = os.replace
        def broken_replace(src, dst):
            raise OSError("simulated replace failure")
        try:
            os.replace = broken_replace
            try:
                atomic_write(keep, b"new")
            except OSError:
                pass
            else:
                raise AssertionError("atomic_write swallowed the replace failure")
        finally:
            os.replace = real_replace
        with open(keep, "rb") as f:
            assert f.read() == b"keep", "the failed replace damaged the old file"
        leftovers = [n for n in os.listdir(td) if n.endswith(".tmp")]
        assert not leftovers, "temp files left behind after failure: %r" % leftovers
    print("[ok] atomic_write: bytes and str verbatim, no temp left, failures keep the old file")

    with tempfile.TemporaryDirectory() as td:
        jf = os.path.join(td, "log.jsonl")
        rows = [{"a": 1, "b": "x"}, {"n": [1, 2], "z": None}, {"s": "노즐"}]
        real_fsync = os.fsync
        calls = [0]
        def counting_fsync(fd):
            calls[0] += 1
            return real_fsync(fd)
        os.fsync = counting_fsync
        try:
            for r in rows:
                jsonl_append(jf, r)
        finally:
            os.fsync = real_fsync
        assert calls[0] == 3, "expected one fsync per append, got %d" % calls[0]
        with open(jf, "rb") as f:
            raw = f.read()
        want = b"".join(canonical_bytes(r) + NB for r in rows)
        assert raw == want, "the file is not one canonical line + LF per call"
        assert read_jsonl(jf) == rows, "read_jsonl lost or reordered rows"
        assert read_jsonl(os.path.join(td, "missing.jsonl")) == [], "a missing file must read as []"
        with open(jf, "ab") as f:
            f.write(b'{"a": 1')                    # a truncated tail, no LF
        try:
            read_jsonl(jf)
        except ValueError as e:
            assert ":4:" in str(e), str(e)
        else:
            raise AssertionError("a truncated last line was accepted")
        try:
            jsonl_append(jf, [1, 2])
        except TypeError:
            pass
        else:
            raise AssertionError("jsonl_append accepted a list")
    print("[ok] jsonl: one fsync per append, canonical lines, truncation names the line")

    fp = env_fingerprint()
    assert list(fp) == ["schema", "python", "implementation", "platform", "machine",
                        "packages", "occt"], list(fp)
    assert fp["schema"] == "cad-env/1", fp["schema"]
    assert list(fp["packages"]) == ["cadquery", "cadquery-ocp", "gmsh", "numpy",
                                    "scikit-learn", "scipy"], list(fp["packages"])
    for name, v in fp["packages"].items():
        want = _package_version(name)
        assert v == want, "%s: %r != importlib.metadata %r" % (name, v, want)
    ocp = fp["packages"]["cadquery-ocp"]
    if ocp is None:
        assert fp["occt"] is None, "occt must be None without cadquery-ocp"
    else:
        assert fp["occt"] == ".".join(ocp.split(".")[:3]), \
            "occt %r is not the first three components of %r" % (fp["occt"], ocp)
    assert env_fingerprint() == fp, "two calls disagree"
    blob = canonical_json(fp)
    assert '"cad-env/1"' in blob, blob
    print("[ok] env_fingerprint: %s" % blob[:120])
    print("SELFTEST PASS")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ["--selftest"]:
        return selftest()
    if len(argv) == 2 and argv[0] == "--digest":
        print(sha256_of(read_json(argv[1])))
        return 0
    if len(argv) == 1 and argv[0] == "--env":
        print(canonical_json(env_fingerprint()))
        return 0
    sys.stderr.write(_USAGE)
    return 2


if __name__ == "__main__":
    sys.exit(main())
