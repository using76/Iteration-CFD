#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""common.py - canonical JSON, sha256, atomic writes, fsynced jsonl and the env fingerprint of the CAD loop (docs/16 §D Reproducibility).

It also holds, in one VERBATIM block, Amagine3D's four file-snapshot functions (Apache-2.0, docs/16a §J, D-10).

Usage:
  python common.py --selftest
  python common.py --digest FILE
  python common.py --env
  python common.py --upstream-diff FRESHNESS_CHECK_PY
"""

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import stat
import subprocess
import sys
import tempfile
import time
from hashlib import sha256
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))       # tools/cad
REPO = os.path.dirname(os.path.dirname(HERE))
FIXTURES = os.path.join(HERE, "fixtures")
NL = "\n"
NB = b"\n"
HEADER_COMMENT = ("meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). "
                  "Source-available, not Open Source. No GPL-licensed source was consulted.")
DIGEST20 = "181ba7663373e3a7dfe47e1af96624dea42039939c30e678d55dc38847a4b143"
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
SNAPSHOT_BEGIN_MARK = ("# ---- BEGIN VERBATIM: Amagine3D e608dc6 skills/text-a3d/freshness_check.py:14-107"
                       " (Apache-2.0) ----")
SNAPSHOT_END_MARK = "# ---- END VERBATIM: Amagine3D freshness_check.py ----"
MODIFIED_MARK = "# modified by Iteration-CFD"
UPSTREAM_SNAPSHOT_SHA256 = "14b336f8c1a5090b1c026dc36ad796d399e9dae0b1732acec9d739b2f67d7306"
UPSTREAM_SNAPSHOT_LINES = (14, 107)             # 1-based, inclusive, in freshness_check.py at e608dc6
SNAPSHOT_HEADER = (
    "# Portions copied from Amagine3D (https://github.com/amagine-ai/Amagine3D, e608dc6, skills/text-a3d/freshness_check.py),",
    "# Copyright 2026 amagine-ai, licensed under the Apache License 2.0 (LICENSE-APACHE-2.0.amagine3d).",
    "# Modified by Iteration-CFD: stable_file_snapshot also accepts a str path (one inserted line, marked).",
)

_USAGE = ("usage: python common.py --selftest | --digest FILE | --env | --upstream-diff FILE"
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


# ---- BEGIN VERBATIM: Amagine3D e608dc6 skills/text-a3d/freshness_check.py:14-107 (Apache-2.0) ----
# Portions copied from Amagine3D (https://github.com/amagine-ai/Amagine3D, e608dc6, skills/text-a3d/freshness_check.py),
# Copyright 2026 amagine-ai, licensed under the Apache License 2.0 (LICENSE-APACHE-2.0.amagine3d).
# Modified by Iteration-CFD: stable_file_snapshot also accepts a str path (one inserted line, marked).
def _missing_snapshot() -> dict[str, object]:
    return {
        "exists": False,
        "mtime_ns": None,
        "sha256": None,
        "size": None,
        "stable": False,
    }


def _same_file_state(
    left: os.stat_result,
    right: os.stat_result,
    *,
    compare_change_time: bool = True,
) -> bool:
    same_change_time = (
        left.st_ctime_ns == right.st_ctime_ns
        if compare_change_time
        else True
    )
    return (
        stat.S_ISREG(right.st_mode)
        and right.st_nlink == 1
        and left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
        and same_change_time
    )


def _hash_descriptor(descriptor: int) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = sha256()
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def stable_file_snapshot(path: Path) -> dict[str, object]:
    """Hash one regular file through one descriptor and bind it to its path."""

    path = path if isinstance(path, Path) else Path(path)  # modified by Iteration-CFD
    descriptor: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            return _missing_snapshot()
        digest = _hash_descriptor(descriptor)
        first_after = os.fstat(descriptor)
        descriptor_stable = _same_file_state(before, first_after)
        if os.name == "nt":
            # Windows st_ctime is creation time, so it cannot reveal a
            # same-size rewrite whose mtime was restored. A second read does.
            verification_digest = _hash_descriptor(descriptor)
            after = os.fstat(descriptor)
            descriptor_stable = (
                descriptor_stable
                and digest == verification_digest
                and _same_file_state(first_after, after)
            )
        else:
            after = first_after
        current = path.lstat()
    except OSError:
        return _missing_snapshot()
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
    stable = descriptor_stable and _same_file_state(
        before,
        current,
        # CPython 3.12 deprecated Windows st_ctime as a creation-time alias.
        # Keep ctime protection between the two descriptor snapshots, but do
        # not compare that value across Windows fstat/lstat implementations.
        compare_change_time=os.name != "nt",
    )
    return {
        "exists": True,
        "mtime_ns": after.st_mtime_ns,
        "sha256": digest if stable else None,
        "size": after.st_size,
        "stable": stable,
    }


# ---- END VERBATIM: Amagine3D freshness_check.py ----


def snapshot_block(source_text: str) -> tuple:
    """(header_lines, content_lines) of the one VERBATIM block (docs/16a §J): the leading lines after BEGIN that
    start with '#' are the header, the rest up to END is the content. ValueError unless exactly one line equals
    each mark and BEGIN comes first."""
    lines = source_text.split("\n")
    if lines.count(SNAPSHOT_BEGIN_MARK) != 1 or lines.count(SNAPSHOT_END_MARK) != 1:
        raise ValueError("common.py: the SNAPSHOT BEGIN/END marks are not each there exactly once")
    begin = lines.index(SNAPSHOT_BEGIN_MARK)
    end = lines.index(SNAPSHOT_END_MARK)
    if begin > end:
        raise ValueError("common.py: the SNAPSHOT BEGIN mark does not come first")
    rest = lines[begin + 1:end]
    header = []
    for line in rest:
        if line.startswith("#"):
            header.append(line)
        else:
            break
    return header, rest[len(header):]


def unmarked_sha256(content_lines: list) -> str:
    """sha256 of the content lines that do NOT end with MODIFIED_MARK, each followed by LF, utf-8."""
    kept = [l for l in content_lines if not l.rstrip().endswith(MODIFIED_MARK)]
    return sha256(("\n".join(kept) + "\n").encode("utf-8")).hexdigest()


def verbatim_diff(content_lines: list, upstream_lines: list) -> tuple:
    """(ok, n_marked, diff_text). difflib.SequenceMatcher(None, upstream_lines, content_lines, autojunk=False):
    ok iff every 'replace' and 'insert' opcode's content-side lines all end with MODIFIED_MARK and there is no
    'delete' opcode. n_marked counts content lines ending with MODIFIED_MARK. diff_text is
    difflib.unified_diff(upstream_lines, content_lines, 'amagine3d e608dc6 freshness_check.py:14-107',
    'tools/cad/common.py VERBATIM block', lineterm='') joined with LF."""
    import difflib
    up = [l.rstrip("\r") for l in upstream_lines]
    con = [l.rstrip("\r") for l in content_lines]
    ok = True
    for tag, _i1, _i2, j1, j2 in difflib.SequenceMatcher(None, up, con, autojunk=False).get_opcodes():
        if tag == "delete" or (tag in ("replace", "insert") and
                               any(not l.rstrip().endswith(MODIFIED_MARK) for l in con[j1:j2])):
            ok = False
    n_marked = sum(1 for l in con if l.rstrip().endswith(MODIFIED_MARK))
    diff_text = "\n".join(difflib.unified_diff(
        up, con, "amagine3d e608dc6 freshness_check.py:14-107",
        "tools/cad/common.py VERBATIM block", lineterm=""))
    return ok, n_marked, diff_text


def upstream_diff(path: str) -> int:
    """--upstream-diff FILE: pin the file to e608dc6 by its unmarked sha256, then show the block's diff."""
    raw = open(path, "rb").read().decode("utf-8")
    upstream = raw.split("\n")[13:107]
    if unmarked_sha256(upstream) != UPSTREAM_SNAPSHOT_SHA256:
        print("upstream-diff: %s is not freshness_check.py at e608dc6" % path)
        return 1
    header, content = snapshot_block(open(os.path.abspath(__file__), "rb").read().decode("utf-8"))
    ok, n_marked, diff_text = verbatim_diff(content, upstream)
    print(diff_text)
    if ok and header == list(SNAPSHOT_HEADER):
        print("upstream-diff: ok, %d marked line(s), header 3 line(s)" % n_marked)
        return 0
    print("upstream-diff: FAIL")
    return 1


def snapshot_selftest() -> None:
    """The AMG-3 proofs (docs/16a §J, AMG-3): the snapshot behaviour, with Amagine3D's two passing tests
    ported as behaviour, not text, and the VERBATIM block's pin and diff."""
    import types
    from unittest import mock
    with tempfile.TemporaryDirectory() as td:
        left = types.SimpleNamespace(st_mode=stat.S_IFREG, st_nlink=1, st_dev=1, st_ino=2,
                                     st_size=3, st_mtime_ns=4, st_ctime_ns=5)
        right = types.SimpleNamespace(st_mode=stat.S_IFREG, st_nlink=1, st_dev=1, st_ino=2,
                                      st_size=3, st_mtime_ns=4, st_ctime_ns=6)
        assert not _same_file_state(left, right)
        assert _same_file_state(left, right, compare_change_time=False)
        right.st_nlink = 2
        assert not _same_file_state(left, right, compare_change_time=False)
        print("[ok] snapshot: _same_file_state compares change time unless told not to, and refuses nlink 2")

        plain = os.path.join(td, "artifact.bin")
        with open(plain, "wb") as f:
            f.write(b"stable artifact")
        snap = stable_file_snapshot(Path(plain))
        assert snap == stable_file_snapshot(str(plain)), "the str path changed the snapshot"
        assert snap["exists"] is True and snap["stable"] is True
        pst = Path(plain).stat()
        assert snap["size"] == pst.st_size and snap["mtime_ns"] == pst.st_mtime_ns
        assert snap["sha256"] == sha256_file(str(plain))
        print("[ok] snapshot: a regular file binds size, mtime and digest; a str path gives the same snapshot")

        def rewriter(target, replacement):
            st0 = os.stat(target)
            original_read = os.read
            state = {"n": 0}

            def side_effect(fd, n):
                chunk = original_read(fd, n)
                if chunk and state["n"] == 0:
                    state["n"] += 1
                    with open(target, "r+b") as g:
                        g.write(replacement)
                    os.utime(target, ns=(st0.st_atime_ns, st0.st_mtime_ns))
                return chunk
            return st0, side_effect

        mutated = os.path.join(td, "mutated.bin")
        with open(mutated, "wb") as f:
            f.write(b"original payload")
        replacement = b"changed! payload"
        assert len(replacement) == len(b"original payload"), "lengths differ"
        st3, effect3 = rewriter(mutated, replacement)
        with mock.patch.object(os, "read", side_effect=effect3):
            snap3 = stable_file_snapshot(Path(mutated))
        assert open(mutated, "rb").read() == replacement, "the rewrite did not happen"
        assert snap3["exists"] is True and snap3["stable"] is False and snap3["sha256"] is None
        assert Path(mutated).stat().st_mtime_ns == st3.st_mtime_ns, "the mtime was not restored"
        print("[ok] snapshot: a same-size in-place rewrite with the mtime restored is detected (stable False, no digest)")

        with open(mutated, "wb") as f:
            f.write(b"original payload")
        st4, effect4 = rewriter(mutated, replacement)
        with mock.patch.object(os, "name", "nt"), mock.patch.object(os, "read", side_effect=effect4):
            snap4 = stable_file_snapshot(Path(mutated))
        assert open(mutated, "rb").read() == replacement, "the rewrite did not happen"
        assert snap4["exists"] is True and snap4["stable"] is False and snap4["sha256"] is None
        assert Path(mutated).stat().st_mtime_ns == st4.st_mtime_ns, "the mtime was not restored"
        print("[ok] snapshot: the Windows branch re-hashes and refuses a digest that changed between the two reads")

        hard_src = os.path.join(td, "hard_source.bin")
        with open(hard_src, "wb") as f:
            f.write(b"hard link body")
        hard_link = os.path.join(td, "hard_link.bin")
        os.link(hard_src, hard_link)
        assert stable_file_snapshot(Path(hard_src)) == _missing_snapshot()
        assert stable_file_snapshot(Path(hard_link)) == _missing_snapshot()
        os.remove(hard_link)
        snap5 = stable_file_snapshot(Path(hard_src))
        assert snap5["stable"] is True and snap5["sha256"] == sha256_file(hard_src)
        print("[ok] snapshot: a hard link is refused on both names (nlink 2); removing it restores the source")

        sym_target = Path(os.path.join(td, "sym_target.bin"))
        sym_target.write_bytes(b"symlink body")
        sym_link = Path(os.path.join(td, "sym_link.bin"))
        try:
            sym_link.symlink_to(sym_target)
        except OSError as e:
            print("[skip] snapshot: symlink not tested, this account cannot create one (%s)"
                  % (getattr(e, "winerror", None) or e.errno))
        else:
            snap6 = stable_file_snapshot(sym_link)
            assert snap6["stable"] is False and snap6["sha256"] is None
            print("[ok] snapshot: a symlink is refused (stable False, no digest)")

        missing = _missing_snapshot()
        assert sorted(missing.keys()) == ["exists", "mtime_ns", "sha256", "size", "stable"]
        assert stable_file_snapshot(Path(os.path.join(td, "no_such_file.bin"))) == missing
        assert stable_file_snapshot(Path(td)) == missing
        print("[ok] snapshot: a missing file and a directory give the missing snapshot")

        source = open(os.path.abspath(__file__), "rb").read().decode("utf-8")
        header, content = snapshot_block(source)
        assert header == list(SNAPSHOT_HEADER), header
        assert len(content) == 95, len(content)
        assert unmarked_sha256(content) == UPSTREAM_SNAPSHOT_SHA256
        assert sum(1 for l in content if l.rstrip().endswith(MODIFIED_MARK)) == 1
        names = [l[4:].split("(")[0] for l in content if l.startswith("def ")]
        assert names == ["_missing_snapshot", "_same_file_state", "_hash_descriptor", "stable_file_snapshot"], names
        every_line = source.split("\n")
        assert every_line.count(SNAPSHOT_BEGIN_MARK) == 1 and every_line.count(SNAPSHOT_END_MARK) == 1
        print("[ok] verbatim: the block's unmarked lines hash to upstream e608dc6 14-107, 3 header lines, 1 marked line")

        upstream = [l for l in content if not l.rstrip().endswith(MODIFIED_MARK)]
        assert len(upstream) == 94, len(upstream)
        ok9, n9, _ = verbatim_diff(content, upstream)
        assert ok9 and n9 == 1, (ok9, n9)
        hits = [k for k, l in enumerate(content) if "1024 * 1024" in l]
        assert len(hits) == 1, hits
        k = hits[0]
        mut_a = list(content)
        mut_a[k] = content[k].replace("1024 * 1024", "1024 * 512")
        assert not verbatim_diff(mut_a, upstream)[0], "an unmarked change was accepted"
        first_stable = next(j for j, l in enumerate(content) if l == '        "stable": False,')
        mut_b = list(content)
        del mut_b[first_stable]
        assert not verbatim_diff(mut_b, upstream)[0], "a deletion was accepted"
        mut_c = list(content)
        mut_c.insert(1, "    pass")
        assert not verbatim_diff(mut_c, upstream)[0], "an unmarked insertion was accepted"
        mut_d = list(content)
        mut_d[k] = content[k].replace("1024 * 1024", "1024 * 512") + "  # modified by Iteration-CFD"
        okd, nd, _ = verbatim_diff(mut_d, upstream)
        assert okd and nd == 2, (okd, nd)
        print("[ok] verbatim: the diff refuses an unmarked change, a deletion and an unmarked insertion, and accepts a marked change")


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
    snapshot_selftest()
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
    if len(argv) == 2 and argv[0] == "--upstream-diff":
        return upstream_diff(argv[1])
    sys.stderr.write(_USAGE)
    return 2


if __name__ == "__main__":
    sys.exit(main())
