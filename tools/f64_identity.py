#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""
f64_identity.py - prove the f64 build did not move.

`sass` hashes the SASS of every cubin of a build through `cuobjdump -sass`
into one JSON document; `sass-diff` diffs two such documents (changed /
removed / added, IDENTICAL or DIFFERS); `rows` diffs two `ofgpu-validate
-json` documents row by row (never `seq`), with `--subset` proving that a
`-sections` run's rows are exactly an ordered subset of the full run's; and
`selftest` covers all three in memory, with no GPU and no cuobjdump.

    python tools/f64_identity.py sass [--target-dir DIR | --cubin-dir DIR] -o OUT.json
    python tools/f64_identity.py sass-diff BEFORE.json AFTER.json
    python tools/f64_identity.py rows BEFORE.json AFTER.json [--subset] [--max-diffs N]
    python tools/f64_identity.py selftest

`sass` searches `<target>/release/build/ofgpu-*/` for the freshest build
(the `output` file's mtime) holding `out/*.cubin`; for the f32 build pass
`--target-dir rust/target/single`. Exit 0 means identical / subset, 1 means
it moved, 2 means the tool could not do its work at all.
"""
import argparse
import contextlib
import glob
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile

SASS_SCHEMA = "f64-identity-sass/1"
ROWS_SCHEMA = "ofgpu-validate/1"
SECTION_SKIP_WHY = "not selected by -sections "

# The row fields `rows` compares. `seq` is never one of them: a -sections run
# renumbers its rows, and a diff that compared seq would call every filtered
# run a difference.
ROW_FIELDS = ("kind", "what", "err", "tol", "ok", "replayed", "why", "nonFinite", "gate")


def die(message: str) -> int:
    print(message, file=sys.stderr)
    return 2


def load_json(path: str, want_schema: str) -> "dict | int":
    """The parsed document, or the exit code (2) when it is unusable."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return die(f"cannot read {path}: {e}")
    if not isinstance(doc, dict) or doc.get("schema") != want_schema:
        return die(f"{path}: schema is not {want_schema}")
    return doc


# ---------------------------------------------------------------- sass


def find_cuobjdump() -> "str | None":
    exe = shutil.which("cuobjdump")
    if exe:
        return exe
    root = os.environ.get("CUDA_PATH")
    if root:
        for name in ("cuobjdump.exe", "cuobjdump"):
            cand = os.path.join(root, "bin", name)
            if os.path.isfile(cand):
                return cand
    return None


def resolve_cubin_dir(args) -> "str | None":
    """`--cubin-dir` if given, else the freshest ofgpu build dir with cubins."""
    if args.cubin_dir:
        return os.path.abspath(args.cubin_dir)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target = args.target_dir or os.path.join(repo, "rust", "target")
    best, best_mtime = None, -1.0
    for d in sorted(glob.glob(os.path.join(target, "release", "build", "ofgpu-*"))):
        output = os.path.join(d, "output")
        if not os.path.isfile(output):
            continue
        out_dir = os.path.join(d, "out")
        if not glob.glob(os.path.join(out_dir, "*.cubin")):
            continue
        mtime = os.path.getmtime(output)
        if mtime > best_mtime:
            best, best_mtime = out_dir, mtime
    return best


def cmd_sass(args) -> int:
    cubin_dir = resolve_cubin_dir(args)
    if cubin_dir is None:
        return die("sass: no build dir with out/*.cubin found (pass --target-dir or --cubin-dir)")
    cuobjdump = find_cuobjdump()
    if cuobjdump is None:
        return die("sass: cuobjdump not found (set CUDA_PATH or put it on PATH)")

    cubins = {}
    for path in sorted(glob.glob(os.path.join(cubin_dir, "*.cubin"))):
        name = os.path.basename(path)
        proc = subprocess.run(
            [cuobjdump, "-sass", path],
            capture_output=True,
        )
        if proc.returncode != 0:
            detail = proc.stderr.decode("utf-8", "replace").strip()
            return die(f"sass: cuobjdump -sass failed on {name}: {detail}")
        cubins[name] = hashlib.sha256(proc.stdout).hexdigest()

    if not cubins:
        return die(f"sass: no cubin found under {cubin_dir}")

    doc = {
        "schema": SASS_SCHEMA,
        "cubinDir": cubin_dir,
        "cuobjdump": cuobjdump,
        "cubins": dict(sorted(cubins.items())),
    }
    parent = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(parent, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    for name, digest in cubins.items():
        print(f"{digest[:16]}  {name}")
    print(f"SASS {len(cubins)} cubins hashed from {cubin_dir}")
    return 0


def cmd_sass_diff(args) -> int:
    before = load_json(args.before, SASS_SCHEMA)
    after = load_json(args.after, SASS_SCHEMA)
    if isinstance(before, int) or isinstance(after, int):
        return 2
    return sass_diff_docs(before, after)


def sass_diff_docs(before: dict, after: dict) -> int:
    """The whole of `sass-diff` on two already-loaded documents."""
    old, new = before["cubins"], after["cubins"]
    changed = removed = added = 0
    for name in sorted(set(old) | set(new)):
        if name in old and name in new:
            if old[name] != new[name]:
                print(f"changed {name}")
                changed += 1
        elif name in old:
            print(f"removed {name}")
            removed += 1
        else:
            print(f"added {name}")
            added += 1
    if changed == 0 and removed == 0:
        line = f"SASS IDENTICAL {len(set(old) & set(new))}"
        if added:
            line += f" ({added} added)"
        print(line)
        return 0
    print(f"SASS DIFFERS changed={changed} removed={removed} added={added}")
    return 1


# ---------------------------------------------------------------- rows


def row_view(row: dict) -> tuple:
    """The comparable fields of one row, in a fixed order."""
    return tuple(row.get(f) for f in ROW_FIELDS)


def fmt(value) -> str:
    return json.dumps(value, separators=(", ", ": "))


def is_section_skip(row: dict) -> bool:
    return (
        row.get("kind") == "skip"
        and isinstance(row.get("why"), str)
        and row["why"].startswith(SECTION_SKIP_WHY)
    )


def cmd_rows(args) -> int:
    before = load_json(args.before, ROWS_SCHEMA)
    after = load_json(args.after, ROWS_SCHEMA)
    if isinstance(before, int) or isinstance(after, int):
        return 2
    if args.subset:
        return rows_subset(before, after, args.max_diffs)
    return rows_equal(before, after, args.max_diffs)


def cap(lines: "list[str]", max_diffs: int) -> int:
    """Print at most max_diffs of the differences; return the full count."""
    for line in lines[:max_diffs]:
        print(line)
    return len(lines)


def rows_equal(before: dict, after: dict, max_diffs: int) -> int:
    """Two runs of the same shape, position by position."""
    old_rows, new_rows = before["rows"], after["rows"]
    diffs = []
    if len(old_rows) != len(new_rows):
        diffs.append(f"row count: {len(old_rows)} != {len(new_rows)}")
    for i, (a, b) in enumerate(zip(old_rows, new_rows)):
        for f, va, vb in zip(ROW_FIELDS, row_view(a), row_view(b)):
            if va != vb:
                diffs.append(f"row {i}: {f}: {fmt(va)} != {fmt(vb)}")
    if before.get("totals") != after.get("totals"):
        diffs.append(f"totals: {fmt(before.get('totals'))} != {fmt(after.get('totals'))}")
    if before.get("exitCode") != after.get("exitCode"):
        diffs.append(f"exitCode: {fmt(before.get('exitCode'))} != {fmt(after.get('exitCode'))}")
    old_gates, new_gates = before.get("gates") or [], after.get("gates") or []
    for i in range(max(len(old_gates), len(new_gates))):
        ga = old_gates[i] if i < len(old_gates) else None
        gb = new_gates[i] if i < len(new_gates) else None
        if ga != gb:
            diffs.append(f"gates[{i}]: {fmt(ga)} != {fmt(gb)}")
    if diffs:
        cap(diffs, max_diffs)
        print(f"ROWS DIFFER {len(diffs)}")
        return 1
    print(f"ROWS IDENTICAL {len(old_rows)}")
    return 0


def rows_subset(before: dict, after: dict, max_diffs: int) -> int:
    """AFTER is a -sections run of the SAME build as full BEFORE.

    Its section skip rows are dropped - they are the filter talking, not the
    build - and every remaining row must equal a distinct BEFORE row, matched
    greedily in order (each match strictly after the previous one), which is
    what makes the filtered run a subsequence of the full one.
    """
    old_rows = before["rows"]
    kept = [r for r in after["rows"] if not is_section_skip(r)]
    unmatched = []
    prev = -1
    for i, row in enumerate(kept):
        want = row_view(row)
        j = prev + 1
        while j < len(old_rows) and row_view(old_rows[j]) != want:
            j += 1
        if j >= len(old_rows):
            unmatched.append((i, row))
        else:
            prev = j
    if unmatched:
        lines = [f"unmatched after row {i}: {fmt(r.get('what'))}" for i, r in unmatched]
        cap(lines, max_diffs)
        print(f"ROWS NOT A SUBSET {len(unmatched)}")
        return 1
    print(f"ROWS SUBSET {len(kept)} of {len(old_rows)}")
    return 0


# ---------------------------------------------------------------- selftest


def check_row(seq: int, what: str, err: float = 0.0, tol: float = 1e-9, ok: bool = True,
              replayed: bool = False, gate: "str | None" = None) -> dict:
    return {
        "seq": seq, "kind": "check", "what": what, "err": err, "tol": tol, "ok": ok,
        "replayed": replayed, "why": None, "nonFinite": None, "gate": gate,
    }


def section_skip_row(seq: int, what: str, spec: str) -> dict:
    return {
        "seq": seq, "kind": "skip", "what": what, "err": None, "tol": None, "ok": None,
        "replayed": None, "why": SECTION_SKIP_WHY + spec, "nonFinite": None, "gate": None,
    }


def rows_doc(rows: "list[dict]", totals: "dict | None" = None, exit_code: int = 0,
             gates: "list | None" = None) -> dict:
    return {
        "schema": ROWS_SCHEMA,
        "totals": totals or {"total": len(rows), "failures": 0, "skipped": 0, "replayed": 0},
        "exitCode": exit_code,
        "gates": gates or [],
        "rows": rows,
    }


def sass_doc(cubins: "dict[str, str]") -> dict:
    return {
        "schema": SASS_SCHEMA,
        "cubinDir": "<memory>",
        "cuobjdump": "<memory>",
        "cubins": dict(cubins),
    }


def run_captured(fn, ns) -> "tuple[int, str]":
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = fn(ns)
    return code, buf.getvalue()


def cmd_selftest(_args) -> int:
    cases = []

    def case(name: str, ok: bool, detail: str = "") -> None:
        cases.append((name, ok, detail))

    base_rows = [
        check_row(1, "first", gate="Gate 99-Z"),
        check_row(2, "second", err=1e-7, tol=1e-6),
        section_skip_row(3, "device row", "cuFFT absent on this card"),
        check_row(4, "third", replayed=True, gate="Gate 99-Y"),
        check_row(5, "fourth", err=0.5, tol=0.25, ok=False),
    ]
    gates = [{"gate": "Gate 99-Z", "verdict": "OPEN"}]

    with tempfile.TemporaryDirectory(prefix="f64-identity-selftest-") as tmp:
        def write(name: str, doc) -> str:
            path = os.path.join(tmp, name)
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(doc, f)
            return path

        a = write("a.json", rows_doc(base_rows, gates=gates))

        # identical -> 0
        b = write("b.json", rows_doc([dict(r) for r in base_rows], gates=[dict(g) for g in gates]))
        code, out = run_captured(cmd_rows, argparse.Namespace(before=a, after=b, subset=False, max_diffs=20))
        case("identical", code == 0 and "ROWS IDENTICAL 5" in out, out)

        # one changed err -> 1 (totals and gates carried over so only it differs)
        rows = [dict(r) for r in base_rows]
        rows[1] = dict(rows[1], err=2e-7)
        code, out = run_captured(cmd_rows, argparse.Namespace(before=a, after=write("c.json", rows_doc(rows, gates=[dict(g) for g in gates])), subset=False, max_diffs=20))
        case("changed-err", code == 1 and "ROWS DIFFER 1" in out and "row 1: err" in out, out)

        # a different row count -> 1
        code, out = run_captured(cmd_rows, argparse.Namespace(before=a, after=write("d.json", rows_doc(base_rows[:4])), subset=False, max_diffs=20))
        case("row-count", code == 1 and "row count: 5 != 4" in out, out)

        # a -sections AFTER whose kept rows are a subsequence -> --subset 0
        sub_rows = [
            section_skip_row(1, "section 1: one", "1"),
            check_row(2, "second", err=1e-7, tol=1e-6),
            section_skip_row(3, "section 3: three", "2"),
            check_row(4, "fourth", err=0.5, tol=0.25, ok=False),
        ]
        sub = write("sub.json", rows_doc(sub_rows))
        code, out = run_captured(cmd_rows, argparse.Namespace(before=a, after=sub, subset=True, max_diffs=20))
        case("subset", code == 0 and "ROWS SUBSET 2 of 5" in out, out)

        # one altered row in it -> --subset 1
        rows = [dict(r) for r in sub_rows]
        rows[3] = dict(rows[3], what="fourth, but moved")
        code, out = run_captured(cmd_rows, argparse.Namespace(before=a, after=write("sub-bad.json", rows_doc(rows)), subset=True, max_diffs=20))
        case("subset-altered", code == 1 and "ROWS NOT A SUBSET 1" in out, out)

        # an out-of-order subset -> --subset 1 (matches must go forward)
        rows = [sub_rows[3], sub_rows[1]]
        code, out = run_captured(cmd_rows, argparse.Namespace(before=a, after=write("sub-ooo.json", rows_doc(rows)), subset=True, max_diffs=20))
        case("subset-order", code == 1 and "ROWS NOT A SUBSET 1" in out, out)

        # wrong schema -> 2
        bad = write("bad.json", {"schema": "nope/9"})
        code, _ = run_captured(cmd_rows, argparse.Namespace(before=bad, after=a, subset=False, max_diffs=20))
        case("rows-schema", code == 2)
        code, _ = run_captured(cmd_sass_diff, argparse.Namespace(before=bad, after=bad))
        case("sass-schema", code == 2)

        sass_a = write("sass-a.json", sass_doc({"one.cubin": "aa", "two.cubin": "bb", "three.cubin": "cc"}))

        code, out = run_captured(cmd_sass_diff, argparse.Namespace(
            before=sass_a, after=write("sass-b.json", sass_doc({"one.cubin": "aa", "two.cubin": "bb", "three.cubin": "cc"}))))
        case("sass-identical", code == 0 and "SASS IDENTICAL 3" in out, out)

        code, out = run_captured(cmd_sass_diff, argparse.Namespace(
            before=sass_a, after=write("sass-c.json", sass_doc({"one.cubin": "aa", "two.cubin": "XX", "three.cubin": "cc"}))))
        case("sass-changed", code == 1 and "changed two.cubin" in out and "SASS DIFFERS changed=1 removed=0 added=0" in out, out)

        code, out = run_captured(cmd_sass_diff, argparse.Namespace(
            before=sass_a, after=write("sass-d.json", sass_doc({"one.cubin": "aa", "two.cubin": "bb", "three.cubin": "cc", "four.cubin": "dd"}))))
        case("sass-added", code == 0 and "added four.cubin" in out and "SASS IDENTICAL 3 (1 added)" in out, out)

        code, out = run_captured(cmd_sass_diff, argparse.Namespace(
            before=sass_a, after=write("sass-e.json", sass_doc({"one.cubin": "aa", "three.cubin": "cc"}))))
        case("sass-removed", code == 1 and "removed two.cubin" in out and "SASS DIFFERS changed=0 removed=1 added=0" in out, out)

    passed = sum(1 for _, ok, _ in cases if ok)
    for name, ok, detail in cases:
        if not ok:
            print(f"SELFTEST FAIL {name}")
            if detail:
                for line in detail.strip().splitlines()[:4]:
                    print(f"    {line}")
    if passed == len(cases):
        print(f"SELFTEST PASS {passed}/{passed}")
        return 0
    print(f"SELFTEST PASS {passed}/{len(cases)}")
    return 1


# ---------------------------------------------------------------- main


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="f64_identity.py",
        description="Prove the f64 build did not move: SASS hashes and validate rows.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("sass", help="hash every cubin of a build")
    p.add_argument("--target-dir", default=None, help="a cargo target dir (default <repo>/rust/target)")
    p.add_argument("--cubin-dir", default=None, help="a directory of .cubin files, searched directly")
    p.add_argument("-o", "--out", required=True, help="where the hash document goes")
    p.set_defaults(fn=cmd_sass)

    p = sub.add_parser("sass-diff", help="diff two sass hash documents")
    p.add_argument("before")
    p.add_argument("after")
    p.set_defaults(fn=cmd_sass_diff)

    p = sub.add_parser("rows", help="diff two ofgpu-validate -json documents row by row")
    p.add_argument("before")
    p.add_argument("after")
    p.add_argument("--subset", action="store_true",
                   help="AFTER is a -sections run of BEFORE's build: its kept rows must be an ordered subset")
    p.add_argument("--max-diffs", type=int, default=20, help="differences printed before the tally (default 20)")
    p.set_defaults(fn=cmd_rows)

    p = sub.add_parser("selftest", help="cover all three in memory, no GPU, no cuobjdump")
    p.set_defaults(fn=cmd_selftest)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
