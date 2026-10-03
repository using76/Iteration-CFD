#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""split - the docs/15 §E split of the generated corpus, written once.

One pool per family at corpus seed 1, drawn ONCE into a tuning manifest
(420 rows) and a held-out test manifest (180 rows), stratified per
(family, stratum) with Hamilton's largest-remainder quotas, the ten spent
G-PILOT geometries always in tuning.  docs/15 §E says the tuning is drawn
at "seed S_tune" and the test at "seed S_test"; the tree disagrees and the
tree wins: G-CORPUS ran on each generator at seed 1 with §E's n and
G-PILOT's ten tuning geometries are seed-1 ids, so the split is one pool
per family at corpus seed 1 divided by a stratified draw.  A second claim
needs a fresh corpus seed for its test pool and a NEW lock file.

Nobody reads a test row's outcome before the evaluation unit (docs/15 §F):
split.py refuses to load test.jsonl, or to pass a test row through its
guard, in any mode but "evaluate" - and it refuses BEFORE the file is
opened.  The lock carries the test manifest's sha256 and ids, is
write-once, and is checked against the git history.

    python tools/autonomy/corpus/split.py --write    # ONCE; then sealed
    python tools/autonomy/corpus/split.py --check
    python tools/autonomy/corpus/split.py --guard ROWS.jsonl --mode MODE
    python tools/autonomy/corpus/split.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import schema  # noqa: E402

CORPUS_SEED = 1
SPLIT_SALT = 17
STRATA = ("easy", "medium", "hard")
FAMILY_TABLE = (("A", "gen_wing", 120, 36), ("B", "gen_lathe", 120, 36),
                ("D", "gen_bluff", 120, 36), ("E", "gen_gap", 60, 18),
                ("F", "gen_thin", 60, 18), ("G", "inject", 120, 36))
SPENT = ("A-1-000", "A-1-001", "A-1-002", "A-1-003", "A-1-012",
         "B-1-000", "B-1-001", "B-1-002", "B-1-003", "B-1-004")
EVALUATE = "evaluate"
MANIFEST_DIR = os.path.join(_HERE, "manifests")
LOCK_REL = "tools/autonomy/corpus/manifests/split.lock"
REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
FRESH_SEEDS = (2,)           # corpus seeds that may carry a fresh held-out split
FRESH_SPLITS = {"test2": 2}  # split name -> its corpus seed


class SplitError(ValueError):
    """A split inconsistency (bad quota, sha drift, unknown id, ...)."""


class SplitSealed(PermissionError):
    """A test-split access outside mode evaluate - refused before reading."""


# --- the pool and the draw --------------------------------------------------


def quotas(counts: dict, n_test: int) -> dict:
    """Hamilton's largest remainder: per stratum, floor of the exact share,
    the leftover units to the largest fractional remainders (STRATA order
    breaks ties)."""
    total = sum(counts.get(s, 0) for s in STRATA)
    exact = {s: n_test * counts[s] / total for s in STRATA if counts.get(s, 0)}
    q = {s: int(math.floor(exact[s])) for s in exact}
    rest = n_test - sum(q.values())
    order = sorted(exact, key=lambda s: (-(exact[s] - q[s]), STRATA.index(s)))
    for s in order[:rest]:
        q[s] += 1
    return {s: q.get(s, 0) for s in STRATA}


def pool(sizes=None, seed=CORPUS_SEED) -> dict:
    """{family: [rows]} from gen.make_row(seed, i) - no bytes, no STLs,
    just the manifest rows.  sizes overrides {family: (n, n_test)}; the
    default seed is the corpus seed, a fresh seed passes its own."""
    if sizes is None:
        sizes = {fam: (n, t) for fam, _m, n, t in FAMILY_TABLE}
    out = {}
    for fam, mod, _n, _t in FAMILY_TABLE:
        if fam not in sizes:
            continue
        gen = importlib.import_module(mod)
        out[fam] = [gen.make_row(seed, i)[0] for i in range(sizes[fam][0])]
    return out


def assign(pooldict: dict, sizes: dict, seed=CORPUS_SEED,
           spent=SPENT) -> tuple:
    """(tuning_rows, test_rows): per family and stratum, the test quota is
    drawn with default_rng([SPLIT_SALT, seed, family ordinal,
    stratum ordinal]) over the sorted non-spent candidate ids; everything
    else is tuning.  Both lists sorted by (family, geometry_id)."""
    tuning, test = [], []
    for fam_ord, (fam, _mod, _n, _t) in enumerate(FAMILY_TABLE):
        if fam not in pooldict:
            continue
        rows = pooldict[fam]
        counts = {}
        for r in rows:
            counts[r["stratum"]] = counts.get(r["stratum"], 0) + 1
        q = quotas(counts, sizes[fam][1])
        by_s = {}
        for r in rows:
            by_s.setdefault(r["stratum"], []).append(r["geometry_id"])
        test_ids = set()
        for s in STRATA:
            cands = sorted(i for i in by_s.get(s, []) if i not in spent)
            if len(cands) < q[s]:
                raise SplitError("%s %s: %d candidates < test quota %d"
                                 % (fam, s, len(cands), q[s]))
            rng = np.random.default_rng([SPLIT_SALT, seed, fam_ord,
                                         STRATA.index(s)])
            perm = rng.permutation(len(cands))
            test_ids.update(cands[int(i)] for i in perm[:q[s]])
        for r in rows:
            if r["geometry_id"] in test_ids:
                test.append(dict(r, split="test"))
            else:
                tuning.append(dict(r, split="tuning"))
    key = lambda r: (r["family"], r["geometry_id"])
    return sorted(tuning, key=key), sorted(test, key=key)


def manifest_bytes(rows: list) -> bytes:
    return "".join(json.dumps(r, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=True) + "\n"
                   for r in rows).encode("ascii")


def lock_bytes(tuning_b: bytes, test_b: bytes, test_ids: list,
               sizes: dict) -> bytes:
    """The write-once lock: ASCII, LF, no time, no path - the test
    manifest's sha256 and ids are sealed here."""
    fam_line = " ".join("%s %d/%d" % (fam, sizes[fam][0], sizes[fam][1])
                        for fam, _m, _n, _t in FAMILY_TABLE if fam in sizes)
    lines = [
        "# autonomy-split/1 - the docs/15 section E split, written once by "
        "tools/autonomy/corpus/split.py --write.",
        "# The test manifest is held out: only mode evaluate reads it "
        "(docs/15 sections E and F), and split.py",
        "# refuses to overwrite this file. A second claim needs a fresh test "
        "seed and a NEW lock file.",
        "schema autonomy-split/1",
        "corpus_seed %d" % CORPUS_SEED,
        "split_salt %d" % SPLIT_SALT,
        "families " + fam_line,
        "spent " + " ".join(SPENT),
        "tuning tuning.jsonl rows %d sha256 %s"
        % (tuning_b.count(b"\n"), hashlib.sha256(tuning_b).hexdigest()),
        "test test.jsonl rows %d sha256 %s"
        % (len(test_ids), hashlib.sha256(test_b).hexdigest()),
        "test_ids " + " ".join(sorted(test_ids)),
    ]
    return ("\n".join(lines) + "\n").encode("ascii")


def read_lock(path: str) -> dict:
    """The lock's non-comment lines as {key: rest}."""
    out = {}
    with open(path, "r", encoding="ascii") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            key, _, rest = ln.partition(" ")
            out[key] = rest
    return out


def build_files(sizes=None) -> dict:
    """The three files, purely in memory."""
    if sizes is None:
        sizes = {fam: (n, t) for fam, _m, n, t in FAMILY_TABLE}
    tuning_rows, test_rows = assign(pool(sizes), sizes)
    tb, sb = manifest_bytes(tuning_rows), manifest_bytes(test_rows)
    return {"tuning.jsonl": tb, "test.jsonl": sb,
            "split.lock": lock_bytes(tb, sb,
                                     [r["geometry_id"] for r in test_rows],
                                     sizes)}


def write(out_dir=MANIFEST_DIR, sizes=None) -> dict:
    """Write the three files ONCE.  A lock seals the directory; manifests
    without a lock are refused too - only --write may create them."""
    lock = os.path.join(out_dir, "split.lock")
    if os.path.exists(lock):
        raise SplitError("split.lock exists in %s: the test split is sealed "
                         "(docs/15 §E); a second claim needs a fresh test "
                         "seed and a new lock" % out_dir)
    for name in ("tuning.jsonl", "test.jsonl"):
        if os.path.exists(os.path.join(out_dir, name)):
            raise SplitError("%s exists in %s without a lock; refusing to "
                             "write" % (name, out_dir))
    files = build_files(sizes)
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(files):
        with open(os.path.join(out_dir, name), "wb") as f:
            f.write(files[name])
    return files



def fresh_dir(seed: int, out_dir=MANIFEST_DIR) -> str:
    """The fresh seed's own directory under the manifests: seed2/."""
    return os.path.join(out_dir, "seed%d" % seed)


def fresh_lock_bytes(seed, test_b: bytes, test_ids: list,
                     sizes: dict) -> bytes:
    """The fresh write-once lock: ASCII, LF, a trailing newline, no time,
    no path - the fresh test manifest's sha256 and ids are sealed here."""
    fam_line = " ".join("%s %d/%d" % (fam, sizes[fam][0], sizes[fam][1])
                        for fam, _m, _n, _t in FAMILY_TABLE if fam in sizes)
    lines = [
        "# autonomy-split/1 - a FRESH test seed for a second claim (docs/15 "
        "section F), written once by",
        "# tools/autonomy/corpus/split.py --write-fresh. Only mode evaluate "
        "reads its test manifest; the",
        "# seed-1 split and its lock are untouched. split.py refuses to "
        "overwrite this file.",
        "schema autonomy-split/1",
        "fresh_test_seed %d" % seed,
        "corpus_seed %d" % seed,
        "split_salt %d" % SPLIT_SALT,
        "families " + fam_line,
        "test test.jsonl rows %d sha256 %s"
        % (len(test_ids), hashlib.sha256(test_b).hexdigest()),
        "test_ids " + " ".join(sorted(test_ids)),
    ]
    return ("\n".join(lines) + "\n").encode("ascii")


def build_fresh_files(seed, sizes=None) -> dict:
    """The two fresh files, purely in memory: the seed-`seed` pool drawn
    with no spent ids, ONLY its test rows kept (the tuning remainder is
    never written)."""
    if sizes is None:
        sizes = {fam: (n, t) for fam, _m, n, t in FAMILY_TABLE}
    _tuning, test_rows = assign(pool(sizes, seed), sizes, seed, spent=())
    tb = manifest_bytes(test_rows)
    return {"test.jsonl": tb,
            "split.lock": fresh_lock_bytes(
                seed, tb, [r["geometry_id"] for r in test_rows], sizes)}


def write_fresh(seed, out_dir=MANIFEST_DIR, sizes=None) -> dict:
    """Write the two fresh files ONCE into fresh_dir(seed, out_dir),
    refused before anything is written."""
    if seed not in FRESH_SEEDS:
        raise SplitError("fresh seed %d is not one of %s"
                         % (seed, ", ".join(str(s) for s in FRESH_SEEDS)))
    if seed == CORPUS_SEED:
        raise SplitError("fresh seed %d is the corpus seed itself" % seed)
    d = fresh_dir(seed, out_dir)
    for name in ("split.lock", "test.jsonl"):
        if os.path.exists(os.path.join(d, name)):
            raise SplitError("%s exists in %s: a fresh seed is written once"
                             % (name, d))
    files = build_fresh_files(seed, sizes)
    os.makedirs(d, exist_ok=True)
    for name in sorted(files):
        with open(os.path.join(d, name), "wb") as f:
            f.write(files[name])
    return files


def fresh_test_ids(seed, out_dir=MANIFEST_DIR) -> list:
    """The fresh lock's test ids, [] when no fresh lock is present."""
    p = os.path.join(fresh_dir(seed, out_dir), "split.lock")
    if not os.path.isfile(p):
        return []
    return (read_lock(p).get("test_ids") or "").split()


# --- the sealed guard -------------------------------------------------------


def _read_checked(name: str, out_dir: str) -> bytes:
    """The manifest's bytes, sha-checked against split.lock."""
    lock_path = os.path.join(out_dir, "split.lock")
    if not os.path.isfile(lock_path):
        raise SplitError("no split.lock in %s" % out_dir)
    want = (read_lock(lock_path).get(name) or "").split(" ")
    want_sha = want[-1] if want else ""
    with open(os.path.join(out_dir, name + ".jsonl"), "rb") as f:
        data = f.read()
    got = hashlib.sha256(data).hexdigest()
    if got != want_sha:
        raise SplitError("%s.jsonl sha256 %s != split.lock %s"
                         % (name, got, want_sha))
    return data


def load(split: str, mode: str, out_dir=MANIFEST_DIR) -> list:
    """The rows of one manifest.  mode "evaluate" is the ONLY mode that may
    read the held-out test split, and the seal is checked BEFORE any file
    is opened (docs/15 §E, §F)."""
    if split not in ("tuning", "test") and split not in FRESH_SPLITS:
        raise SplitError("split: %r is not tuning or test" % (split,))
    if not isinstance(mode, str) or not mode:
        raise SplitError("mode: %r is not a non-empty string" % (mode,))
    if split == "test" and mode != EVALUATE:
        raise SplitSealed(
            "the test split is sealed: mode %r may not read it (docs/15 §E: "
            "only mode evaluate reads the held-out split, in the evaluation "
            "unit)" % (mode,))
    if split in FRESH_SPLITS:
        if mode != EVALUATE:
            raise SplitSealed(
                "the fresh test split %s is sealed: mode %r may not read it "
                "(docs/15 §F: only mode evaluate reads a held-out split, "
                "in the evaluation unit)" % (split, mode))
        return [json.loads(ln)
                for ln in _read_checked("test",
                                        fresh_dir(FRESH_SPLITS[split],
                                                  out_dir)
                                        ).decode("ascii").splitlines()]
    return [json.loads(ln)
            for ln in _read_checked(split, out_dir).decode("ascii").splitlines()]


def _manifest_ids(out_dir: str) -> tuple:
    """(test ids from the lock, tuning ids from the sha-checked manifest)."""
    tdata = _read_checked("tuning", out_dir)
    lock = read_lock(os.path.join(out_dir, "split.lock"))
    test_ids = set((lock.get("test_ids") or "").split())
    for s in FRESH_SEEDS:
        test_ids.update(fresh_test_ids(s, out_dir))
    tuning_ids = {json.loads(ln)["geometry_id"]
                  for ln in tdata.decode("ascii").splitlines()}
    return test_ids, tuning_ids


def _refuse_ids(geometry_id: str, mode: str, test_ids: set,
                tuning_ids: set) -> str:
    if geometry_id in test_ids:
        if mode != EVALUATE:
            raise SplitSealed(
                "geometry %s is in the held-out test split: mode %r may not "
                "read or score it (docs/15 §E)" % (geometry_id, mode))
        return "test"
    if geometry_id in tuning_ids:
        return "tuning"
    raise SplitError("geometry %s is not in the corpus manifests"
                     % geometry_id)


def refuse_test(geometry_id: str, mode: str, out_dir=MANIFEST_DIR) -> str:
    """Reads ONLY split.lock (test ids) and tuning.jsonl (sha-checked, ids
    only) - never test.jsonl.  Returns "tuning" or "test"."""
    test_ids, tuning_ids = _manifest_ids(out_dir)
    return _refuse_ids(geometry_id, mode, test_ids, tuning_ids)


def filter_rows(rows: list, mode: str, out_dir=MANIFEST_DIR) -> list:
    """Every row must pass refuse_test; the lock is read once.  Rows are
    returned unchanged - no field of a row is printed or inspected."""
    test_ids, tuning_ids = _manifest_ids(out_dir)
    for i, row in enumerate(rows):
        gid = row.get("geometry_id") if isinstance(row, dict) else None
        if not gid:
            raise SplitError("row %d has no geometry_id" % i)
    for row in rows:
        _refuse_ids(row["geometry_id"], mode, test_ids, tuning_ids)
    return rows


# --- the check --------------------------------------------------------------


def _raw(name: str, out_dir: str):
    path = os.path.join(out_dir, name)
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        return f.read()


def check(out_dir=MANIFEST_DIR) -> list:
    """The 8 [split] lines over the manifests on disk; raises SplitError
    naming the first failure."""
    sizes = {fam: (n, t) for fam, _m, n, t in FAMILY_TABLE}
    tun = load("tuning", "check", out_dir)
    test_rows = [json.loads(ln)
                 for ln in _read_checked("test", out_dir).decode("ascii").splitlines()]
    lines = ["[split] files: tuning.jsonl %d rows, test.jsonl %d rows, "
             "sha256 equal to split.lock" % (len(tun), len(test_rows))]

    def per(rows, key):
        c = {}
        for r in rows:
            c[r[key]] = c.get(r[key], 0) + 1
        return c

    tc, sc = per(tun, "family"), per(test_rows, "family")
    fmt = lambda c: ", ".join("%s %d" % (f, c.get(f, 0))
                              for f, _m, _n, _t in FAMILY_TABLE)
    lines.append("[split] counts: tuning %d (%s), test %d (%s)"
                 % (len(tun), fmt(tc), len(test_rows), fmt(sc)))

    cells, bad = 0, []
    for fam, _m, _n, _t in FAMILY_TABLE:
        p_rows = [r for r in tun + test_rows if r["family"] == fam]
        pc = per(p_rows, "stratum")
        n_test = sc.get(fam, 0)
        if n_test != sizes[fam][1]:
            bad.append("%s: %d test rows != %d" % (fam, n_test, sizes[fam][1]))
        q = quotas(pc, n_test)
        tcc = per([r for r in test_rows if r["family"] == fam], "stratum")
        for s in STRATA:
            if pc.get(s, 0):
                cells += 1
            if tcc.get(s, 0) != q.get(s, 0):
                bad.append("%s %s: test %d != quota %d"
                           % (fam, s, tcc.get(s, 0), q.get(s, 0)))
    if bad:
        raise SplitError("stratified: " + "; ".join(bad))
    lines.append("[split] stratified: %d (family, stratum) cells, every "
                 "test count equal to its largest-remainder quota" % cells)

    t_ids = [r["geometry_id"] for r in tun]
    s_ids = [r["geometry_id"] for r in test_rows]
    t_sha = [r["stl_sha256"] for r in tun]
    s_sha = [r["stl_sha256"] for r in test_rows]
    shared_ids = len(set(t_ids) & set(s_ids))
    shared_sha = len(set(t_sha) & set(s_sha))
    all_sha = set(t_sha) | set(s_sha)
    if shared_ids or shared_sha or len(all_sha) != len(tun) + len(test_rows):
        raise SplitError("disjoint: %d shared ids, %d shared sha, %d "
                         "distinct sha of %d" % (shared_ids, shared_sha,
                                                 len(all_sha),
                                                 len(tun) + len(test_rows)))
    g_parents = [r["params"]["parent_id"]
                 for r in tun + test_rows if r["family"] == "G"]
    if len(set(g_parents)) != len(g_parents):
        raise SplitError("G parents repeat")
    if set(g_parents) & (set(t_ids) | set(s_ids)):
        raise SplitError("a G parent is a corpus id")
    lines.append("[split] disjoint: %d shared geometry_id, %d shared "
                 "stl_sha256, %d distinct stl_sha256; G parents %d distinct, "
                 "none a corpus id"
                 % (shared_ids, shared_sha, len(all_sha),
                    len(set(g_parents))))

    missing = [i for i in SPENT
               if not any(r["geometry_id"] == i and r["split"] == "tuning"
                          for r in tun)]
    if missing:
        raise SplitError("spent: %s not in tuning" % missing)
    lines.append("[split] spent: %d of %d G-PILOT ids in tuning"
                 % (len(SPENT) - len(missing), len(SPENT)))
    return _check_rest(out_dir, tun, test_rows, s_ids, lines)


def _check_rest(out_dir, tun, test_rows, s_ids, lines) -> list:
    """The rows, regen and lock-history lines of check()."""
    errs = []
    labelled = ([(r, "tuning") for r in tun]
                + [(r, "test") for r in test_rows])
    for r, want in labelled:
        if r["split"] != want:
            raise SplitError("row %s: split field %r does not match its file"
                             % (r["geometry_id"], r["split"]))
        errs += schema.errors(r, "ManifestRow")
    if errs:
        raise SplitError("rows: %s" % errs[0])
    lock = read_lock(os.path.join(out_dir, "split.lock"))
    if (lock.get("test_ids") or "").split() != sorted(s_ids):
        raise SplitError("split.lock test_ids != test.jsonl ids")
    lines.append("[split] rows: %d valid ManifestRow, each split field equal "
                 "to its file; lock test_ids equal test.jsonl ids"
                 % (len(tun) + len(test_rows)))

    files = build_files()
    same = all(files[name] == _raw(name, out_dir) for name in sorted(files))
    if not same:
        raise SplitError("regen: the in-process rebuild differs from the "
                         "files on disk")
    tmp = tempfile.mkdtemp(prefix="split_regen_")
    try:
        env = dict(os.environ, PYTHONHASHSEED="12345",
                   PYTHONIOENCODING="utf-8")
        child = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--write", "--out",
             tmp], capture_output=True, env=env, timeout=600)
        if child.returncode != 0:
            raise SplitError("regen child exited %d: %s"
                             % (child.returncode,
                                child.stderr.decode("utf-8", "replace")[-300:]))
        same = all(files[name] == _raw(name, tmp) for name in sorted(files))
        if not same:
            raise SplitError("regen: the child rebuild under PYTHONHASHSEED="
                             "12345 differs")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    lines.append("[split] regen: in-process byte-identical (3 files); child "
                 "under PYTHONHASHSEED=12345 byte-identical (3 files)")

    if os.path.abspath(out_dir) != os.path.abspath(MANIFEST_DIR):
        lines.append("[split] lock history: not in the tree")
        return lines
    r = subprocess.run(["git", "-C", REPO, "log", "--format=%H", "--",
                        LOCK_REL], capture_output=True, timeout=120)
    if r.returncode != 0:
        raise SplitError("git log failed: %s"
                         % r.stderr.decode("utf-8", "replace").strip())
    commits = r.stdout.decode("ascii").split()
    if len(commits) >= 2:
        raise SplitError("split.lock was rewritten: %d commits touch it"
                         % len(commits))
    if len(commits) == 1:
        r2 = subprocess.run(["git", "-C", REPO, "show", "%s:%s"
                             % (commits[0], LOCK_REL)], capture_output=True,
                            timeout=120)
        working = _raw("split.lock", out_dir)
        if r2.returncode != 0 or r2.stdout != working:
            raise SplitError("split.lock differs from its commit %s"
                             % commits[0])
        lines.append("[split] lock history: 1 commit(s) touch split.lock; "
                     "working copy equals %s" % commits[0][:8])
    else:
        lines.append("[split] lock history: 0 commit(s) touch split.lock; "
                     "not yet committed")
    return lines



def check_fresh(seed, out_dir=MANIFEST_DIR, sizes=None) -> list:
    """The 6 [split] fresh lines over the fresh seed's two files; raises
    SplitError naming the first failure (sizes None = FAMILY_TABLE's)."""
    if sizes is None:
        sizes = {fam: (n, t) for fam, _m, n, t in FAMILY_TABLE}
    d = fresh_dir(seed, out_dir)
    test_rows = [json.loads(ln)
                 for ln in _read_checked("test", d).decode("ascii").splitlines()]
    lock = read_lock(os.path.join(d, "split.lock"))
    lines = ["[split] fresh seed %d: test.jsonl %d rows, sha256 equal to "
             "seed%d/split.lock" % (seed, len(test_rows), seed)]

    def per(rows, key):
        c = {}
        for r in rows:
            c[r[key]] = c.get(r[key], 0) + 1
        return c

    pd = pool(sizes, seed)
    bad = []
    for fam, _m, _n, _t in FAMILY_TABLE:
        if fam not in sizes:
            continue
        tcc = per([r for r in test_rows if r["family"] == fam], "stratum")
        q = quotas(per(pd[fam], "stratum"), sizes[fam][1])
        for s in STRATA:
            if tcc.get(s, 0) != q.get(s, 0):
                bad.append("%s %s: test %d != quota %d"
                           % (fam, s, tcc.get(s, 0), q.get(s, 0)))
    if bad:
        raise SplitError("fresh stratified: " + "; ".join(bad))
    counts = ", ".join("%s %d" % (f, per(test_rows, "family").get(f, 0))
                       for f, _m, _n, _t in FAMILY_TABLE if f in sizes)
    lines.append("[split] fresh counts: %s, every (family, stratum) count "
                 "equal to its quota in the seed-%d pool" % (counts, seed))
    tun = [json.loads(ln) for ln in
           _read_checked("tuning", out_dir).decode("ascii").splitlines()]
    s_ids = set(r["geometry_id"] for r in test_rows)
    s_sha = set(r["stl_sha256"] for r in test_rows)
    sh_ids = len(s_ids & set(r["geometry_id"] for r in tun))
    sh_sha = len(s_sha & set(r["stl_sha256"] for r in tun))
    lock1 = read_lock(os.path.join(out_dir, "split.lock"))
    sh_test1 = len(s_ids & set((lock1.get("test_ids") or "").split()))
    if sh_ids or sh_sha or sh_test1:
        raise SplitError("fresh disjoint: %d ids and %d stl_sha256 shared "
                         "with tuning.jsonl, %d ids shared with the seed-1 "
                         "test lock" % (sh_ids, sh_sha, sh_test1))
    lines.append("[split] fresh disjoint: %d ids and %d stl_sha256 shared "
                 "with tuning.jsonl, %d ids shared with the seed-1 test lock"
                 % (sh_ids, sh_sha, sh_test1))
    errs = []
    for r in test_rows:
        if r["split"] != "test":
            raise SplitError("fresh row %s: split field %r is not \"test\""
                             % (r["geometry_id"], r["split"]))
        errs += schema.errors(r, "ManifestRow")
    if errs:
        raise SplitError("fresh rows: %s" % errs[0])
    if (lock.get("test_ids") or "").split() != sorted(r["geometry_id"]
                                                      for r in test_rows):
        raise SplitError("fresh lock test_ids != test.jsonl ids")
    lines.append("[split] fresh rows: %d valid ManifestRow, split \"test\", "
                 "lock test_ids equal test.jsonl ids" % len(test_rows))
    files = build_fresh_files(seed, sizes)
    if any(files[n] != _raw(n, d) for n in ("test.jsonl", "split.lock")):
        raise SplitError("fresh regen: the in-process rebuild differs from "
                         "the files in %s" % d)
    lines.append("[split] fresh regen: in-process byte-identical (2 files)")
    if os.path.abspath(out_dir) != os.path.abspath(MANIFEST_DIR):
        lines.append("[split] fresh lock history: not in the tree")
        return lines
    rel = "tools/autonomy/corpus/manifests/seed%d/split.lock" % seed
    r = subprocess.run(["git", "-C", REPO, "log", "--format=%H", "--", rel],
                       capture_output=True, timeout=120)
    if r.returncode != 0:
        raise SplitError("git log failed: %s"
                         % r.stderr.decode("utf-8", "replace").strip())
    commits = r.stdout.decode("ascii").split()
    if len(commits) >= 2:
        raise SplitError("seed%d/split.lock was rewritten: %d commits touch it"
                         % (seed, len(commits)))
    if len(commits) == 1:
        r2 = subprocess.run(["git", "-C", REPO, "show", "%s:%s"
                             % (commits[0], rel)], capture_output=True,
                            timeout=120)
        if r2.returncode != 0 or r2.stdout != _raw("split.lock", d):
            raise SplitError("seed%d/split.lock differs from its commit %s"
                             % (seed, commits[0]))
        lines.append("[split] fresh lock history: %d commit(s) touch "
                     "seed%d/split.lock; working copy equals %s"
                     % (len(commits), seed, commits[0][:8]))
    else:
        lines.append("[split] fresh lock history: %d commit(s) touch "
                     "seed%d/split.lock; not yet committed"
                     % (len(commits), seed))
    return lines


# --- the selftest -----------------------------------------------------------


def _selftest() -> int:
    SMALL = {"A": (12, 4), "B": (12, 4), "D": (8, 3), "E": (6, 2),
             "F": (6, 2), "G": (12, 4)}

    def group_quotas():
        cases = [({"easy": 1, "medium": 58, "hard": 61}, 36,
                  {"easy": 0, "medium": 18, "hard": 18}),
                 ({"easy": 30, "medium": 83, "hard": 7}, 36,
                  {"easy": 9, "medium": 25, "hard": 2}),
                 ({"easy": 33, "medium": 21, "hard": 6}, 18,
                  {"easy": 10, "medium": 6, "hard": 2}),
                 ({"easy": 20, "medium": 11, "hard": 29}, 18,
                  {"easy": 6, "medium": 3, "hard": 9}),
                 ({"easy": 10}, 4, {"easy": 4, "medium": 0, "hard": 0})]
        for counts, n_test, want in cases:
            got = quotas(counts, n_test)
            assert got == want, (counts, n_test, got, want)
            assert sum(got.values()) == n_test
        tiny = {"A": [{"geometry_id": "A-1-000", "stratum": "easy",
                       "family": "A"},
                      {"geometry_id": "A-1-001", "stratum": "easy",
                       "family": "A"}]}
        try:
            assign(tiny, {"A": (2, 2)})
        except SplitError as e:
            assert "candidates < test quota" in str(e), e
        else:
            raise AssertionError("the tiny pool was not refused")
        return "5 unit cases exact, sums == n_test, a tiny pool refused"

    def group_check():
        lines = check()
        for ln in lines:
            print("  %s" % ln)
        assert len(lines) == 8, len(lines)
        assert all(ln.startswith("[split] ") for ln in lines)
        return "the 8 [split] lines above on the in-tree manifests"


    def group_sealed():
        for mode in ("tuning", "rules", "b0-template"):
            try:
                load("test", mode)
            except SplitSealed as e:
                assert "sealed" in str(e) and mode in str(e), e
            else:
                raise AssertionError("mode %r read the test split" % mode)
        d = tempfile.mkdtemp(prefix="split_seal_")
        try:
            for name in ("split.lock", "tuning.jsonl"):
                shutil.copy(os.path.join(MANIFEST_DIR, name),
                            os.path.join(d, name))
            try:
                load("test", "tuning", d)
            except SplitSealed:
                pass
            else:
                raise AssertionError("the seal did not refuse before reading")
            assert not os.path.exists(os.path.join(d, "test.jsonl"))
        finally:
            shutil.rmtree(d, ignore_errors=True)
        assert len(load("test", EVALUATE)) == 180
        test_id = read_lock(os.path.join(MANIFEST_DIR, "split.lock"))
        test_id = test_id["test_ids"].split()[0]
        tun_rows = load("tuning", "check")
        poisoned = [dict(tun_rows[0], geometry_id=test_id)] + tun_rows[1:3]
        try:
            filter_rows(poisoned, "b0-template")
        except SplitSealed as e:
            assert test_id in str(e), e
        else:
            raise AssertionError("a test id passed the guard outside evaluate")
        assert filter_rows(poisoned, EVALUATE) == poisoned
        try:
            filter_rows([{"geometry_id": "A-1-999"}], EVALUATE)
        except SplitError as e:
            assert "not in the corpus manifests" in str(e), e
        else:
            raise AssertionError("an unknown id passed the guard")
        d2 = tempfile.mkdtemp(prefix="split_guard_")
        try:
            rows_file = os.path.join(d2, "rows.jsonl")
            exe = [sys.executable, os.path.abspath(__file__), "--guard",
                   rows_file]
            with open(rows_file, "w", encoding="ascii") as f:
                f.write(json.dumps({"geometry_id": test_id}) + "\n")
            r = subprocess.run(exe + ["--mode", "b0-template"],
                               capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=120)
            assert r.returncode == 3 and "REFUSED:" in r.stderr, \
                (r.returncode, r.stderr[:200])
            with open(rows_file, "w", encoding="ascii") as f:
                for rrow in tun_rows[:3]:
                    f.write(json.dumps({"geometry_id":
                                        rrow["geometry_id"]}) + "\n")
            r = subprocess.run(exe + ["--mode", "b0-template"],
                               capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=120)
            assert r.returncode == 0, (r.returncode, r.stdout[:200])
            assert "guard: 3 row(s) pass in mode b0-template" in r.stdout
        finally:
            shutil.rmtree(d2, ignore_errors=True)
        return "test sealed outside evaluate (even with the file deleted), " \
            "--guard exits 3 on a test row and 0 on 3 tuning rows"


    def group_tamper():
        d = tempfile.mkdtemp(prefix="split_tamper_")
        try:
            for name in ("split.lock", "tuning.jsonl", "test.jsonl"):
                shutil.copy(os.path.join(MANIFEST_DIR, name),
                            os.path.join(d, name))
            tp = os.path.join(d, "test.jsonl")
            data = bytearray(open(tp, "rb").read())
            i = data.index(b'"stl_sha256":"')
            data[i + 14] = ord("0") if data[i + 14] != ord("0") else ord("1")
            open(tp, "wb").write(bytes(data))
            try:
                load("test", EVALUATE, d)
            except SplitError as e:
                assert "test.jsonl sha256" in str(e), e
            else:
                raise AssertionError("a tampered test.jsonl was accepted")
            shutil.copy(os.path.join(MANIFEST_DIR, "test.jsonl"), tp)
            lp = os.path.join(d, "split.lock")
            ldata = bytearray(open(lp, "rb").read())
            j = ldata.index(b"test test.jsonl rows")
            k = ldata.index(b"sha256 ", j) + 7
            ldata[k] = ord("0") if ldata[k] != ord("0") else ord("1")
            open(lp, "wb").write(bytes(ldata))
            try:
                load("test", EVALUATE, d)
            except SplitError as e:
                assert "test.jsonl sha256" in str(e), e
            else:
                raise AssertionError("an edited lock sha was accepted")
        finally:
            shutil.rmtree(d, ignore_errors=True)
        return "a flipped sha byte in test.jsonl and in split.lock are refused"

    def group_write_once():
        d = tempfile.mkdtemp(prefix="split_once_")
        try:
            files = write(d, SMALL)
            assert len(files) == 3, sorted(files)
            for name in files:
                assert os.path.isfile(os.path.join(d, name))
            try:
                write(d, SMALL)
            except SplitError as e:
                assert "split.lock exists" in str(e), e
            else:
                raise AssertionError("the second write was not refused")
            os.remove(os.path.join(d, "split.lock"))
            try:
                write(d, SMALL)
            except SplitError as e:
                assert "without a lock" in str(e), e
            else:
                raise AssertionError("a manifest write without a lock passed")
        finally:
            shutil.rmtree(d, ignore_errors=True)
        return "one write works; the second and the lock-less rewrite refused"

    def group_spent():
        path = os.path.join(os.path.dirname(_HERE), "pilot",
                            "pilot_rows.jsonl")
        pat = re.compile(r"^[A-G]-[0-9]+-[0-9]{3}$")
        ids = set()
        with open(path, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                gid = json.loads(ln).get("geometry_id", "")
                if pat.match(gid):
                    ids.add(gid)
        assert ids == set(SPENT), sorted(ids ^ set(SPENT))
        tun_ids = {r["geometry_id"] for r in load("tuning", "check")}
        assert all(i in tun_ids for i in SPENT), "a spent id is not in tuning"
        return "SPENT == the corpus ids of pilot_rows.jsonl, all in tuning"

    def group_rows():
        rows = load("tuning", "check") + load("test", EVALUATE)
        for r in rows:
            errs = schema.errors(r, "ManifestRow")
            assert not errs, (r["geometry_id"], errs[:1])
            if r["family"] == "G":
                assert r["params"]["parent_index"] >= 1000, r["geometry_id"]
        return "%d in-tree rows validate as ManifestRow; G parents at " \
            "index >= 1000" % len(rows)

    def group_fresh():
        SMALL_F = {"D": (8, 3), "F": (8, 3)}
        WANT_IDS = ["D-2-002", "D-2-003", "D-2-005",
                    "F-2-002", "F-2-003", "F-2-004"]
        WANT_SHA = ("a99d8e8d235316e8779f71f1e71ef9d253f17946a56cd4e931a"
                    "45a915e9e44bf")
        d = tempfile.mkdtemp(prefix="split_fresh_")
        try:
            for name in ("split.lock", "tuning.jsonl"):
                shutil.copy(os.path.join(MANIFEST_DIR, name),
                            os.path.join(d, name))
            files = write_fresh(2, d, SMALL_F)
            ids = sorted(json.loads(ln)["geometry_id"] for ln in
                         files["test.jsonl"].decode("ascii").splitlines())
            assert ids == WANT_IDS, ids
            assert hashlib.sha256(files["test.jsonl"]).hexdigest() == WANT_SHA
            try:
                write_fresh(2, d, SMALL_F)
            except SplitError as e:
                assert "written once" in str(e), e
            else:
                raise AssertionError("the second fresh write was not refused")
            try:
                write_fresh(1, d, SMALL_F)
            except SplitError:
                pass
            else:
                raise AssertionError("fresh seed 1 was not refused")
            tp = os.path.join(d, "seed2", "test.jsonl")
            os.remove(tp)
            try:
                load("test2", "rules", d)
            except SplitSealed:
                pass
            else:
                raise AssertionError("the fresh seal did not refuse before "
                                     "opening")
            with open(tp, "wb") as f:
                f.write(files["test.jsonl"])
            assert len(load("test2", EVALUATE, d)) == 6
            try:
                refuse_test("D-2-002", "rules", d)
            except SplitSealed:
                pass
            else:
                raise AssertionError("a fresh test id passed the guard")
            assert refuse_test("D-2-002", EVALUATE, d) == "test"
            lines = check_fresh(2, d, SMALL_F)
            assert len(lines) == 6, lines
            assert lines[0].startswith("[split] fresh seed 2"), lines[0]
            assert lines[-1] == "[split] fresh lock history: not in the tree"
            data = bytearray(open(tp, "rb").read())
            data[0] ^= 0xFF
            open(tp, "wb").write(bytes(data))
            try:
                check_fresh(2, d, SMALL_F)
            except SplitError as e:
                assert "sha256" in str(e), e
            else:
                raise AssertionError("a flipped fresh byte was accepted")
            f1 = build_fresh_files(2, SMALL_F)
            f2 = build_fresh_files(2, SMALL_F)
            assert f1 == f2 and sorted(f1) == ["split.lock", "test.jsonl"]
        finally:
            shutil.rmtree(d, ignore_errors=True)
        return "fresh seed 2: the 6 ids and sha of the oracle, write-once, "             "sealed outside evaluate, check_fresh 6 lines"

    groups = [("[ok] quotas", group_quotas),
              ("[ok] check", group_check),
              ("[ok] sealed", group_sealed),
              ("[ok] tamper", group_tamper),
              ("[ok] write-once", group_write_once),
              ("[ok] spent", group_spent),
              ("[ok] rows", group_rows),
              ("[ok] fresh seed", group_fresh)]
    for name, fn in groups:
        try:
            note = fn()
        except AssertionError as e:
            print("SELFTEST FAIL: %s: %s" % (name, e))
            return 1
        print("%s: %s" % (name, note))
    print("SELFTEST PASS")
    return 0


# --- the CLI ----------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="split",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--write-fresh", type=int, default=None)
    ap.add_argument("--check-fresh", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--guard", default=None)
    ap.add_argument("--mode", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()
    if args.write:
        out = args.out or MANIFEST_DIR
        files = write(out)
        print("wrote tuning.jsonl (%d rows), test.jsonl (%d rows), split.lock "
              "to %s" % (files["tuning.jsonl"].count(b"\n"),
                         files["test.jsonl"].count(b"\n"), out))
        print("test.jsonl sha256 %s"
              % hashlib.sha256(files["test.jsonl"]).hexdigest())
        return 0
    if args.write_fresh is not None:
        out = args.out or MANIFEST_DIR
        files = write_fresh(args.write_fresh, out)
        print("wrote seed%d/test.jsonl (%d rows), seed%d/split.lock"
              % (args.write_fresh, files["test.jsonl"].count(b"\n"),
                 args.write_fresh))
        print("test.jsonl sha256 %s"
              % hashlib.sha256(files["test.jsonl"]).hexdigest())
        return 0
    if args.check_fresh is not None:
        try:
            for ln in check_fresh(args.check_fresh, args.out or MANIFEST_DIR):
                print(ln)
        except (SplitError, SplitSealed) as e:
            print("FRESH CHECK FAIL: %s" % e)
            return 1
        print("FRESH CHECK PASS")
        return 0
    if args.check:
        try:
            for ln in check(args.out or MANIFEST_DIR):
                print(ln)
        except (SplitError, SplitSealed) as e:
            print("SPLIT CHECK FAIL: %s" % e)
            return 1
        print("SPLIT CHECK PASS")
        return 0
    if args.guard:
        if not args.mode:
            ap.error("--mode is required with --guard")
        try:
            with open(args.guard, "r", encoding="ascii") as f:
                rows = [json.loads(ln) for ln in f if ln.strip()]
            kept = filter_rows(rows, args.mode)
        except (SplitSealed, SplitError) as e:
            print("REFUSED: %s" % e, file=sys.stderr)
            return 3
        print("guard: %d row(s) pass in mode %s" % (len(kept), args.mode))
        return 0
    ap.error("one of --write, --check, --guard/--mode or --selftest is "
             "required")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    sys.exit(main())
