#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""fetch_answer_keys.py - place and check the SPEC-LIT 110 and 144.6 answer keys.

usage:
    python tools/keys/fetch_answer_keys.py write
    python tools/keys/fetch_answer_keys.py fetch [--from DIR]
    python tools/keys/fetch_answer_keys.py check | --check

``write`` extracts the two tracked transcriptions from the fenced blocks of
rust/SPEC-LIT.md (110.3 Driver & Seegmiller, 110.4 McCaffrey) and writes them
byte for byte under reference/. ``fetch`` downloads the keys that are not
redistributed (so they are git-ignored) - the three MKM channel DNS files from
turbulence.oden.utexas.edu and the ERCOFTAC T3A summary table t3ay.dat from
the ERCOFTAC Classic Collection (SPEC-LIT 144.6, docs/17 decision D-VAL-2) -
or takes them from a local copy with ``--from DIR``; a file whose SHA-256 is
not the manifest's pinned digest is never kept. When a download fails, the
manual placement is printed: the page to download the file from by hand and
the ``--from`` command that checks and places it. ``check`` recomputes every
pin - the fenced blocks, the manifest rows and every key file present - and
prints ``check: <ok> ok, <missing> missing, <failed> failed``, exiting 0 iff
nothing failed. SPEC-LIT 110.1, 110.5 and 144.6 are the contract this script
serves.
"""
import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

# (id, path under reference/, how, pinned sha256, bytes)
KEYS = [
    ("driver-seegmiller1985", "driver_seegmiller_1985/reattachment.txt", "write",
     "f4d9523fa69ccd69b04ed2c86c312aa9d81a1d2f751f32d4cfdaeb57eb5f106a", 514),
    ("mccaffrey1979-table1", "mccaffrey_1979/centreline_table1.txt", "write",
     "c8db8376b8fd17d032dfdeff6a8f05c8d0d51ab53516d081beed73ab1f872122", 658),
    ("mkm99-chan180", "mkm99/chan180.means", "fetch",
     "f54c86e30a869798f140aa004c1e167cd13a6280e988d2794d72ca2be8b5fc82", 7044),
    ("mkm99-chan395", "mkm99/chan395.means", "fetch",
     "211c6b8149cc7ef5cd20323716b47a45dd98cca55486a9554acfd4a7975d737a", 12932),
    ("mkm99-chan590", "mkm99/chan590.means", "fetch",
     "878bc951794516b6136912f8f098cd6f2f96c4bbe7f21a8cea657ffb270d419d", 12932),
    ("ercoftac-t3a", "ercoftac_t3a/t3ay.dat", "fetch",
     "61b15b27511fb940409bab20b67c4776c170f2a1679e4a369a6536ab9043e8cc", 2181),
]

MKM_URL = "https://turbulence.oden.utexas.edu/data/MKM/chan{N}/profiles/chan{N}.means"
MKM_PAGE = "https://turbulence.oden.utexas.edu/MKM_1999.html"
# The ERCOFTAC case page links its files over http, the URL the manifest row
# cites; the bytes are pinned by digest, so the transport changes nothing kept.
T3A_URL = ("http://cfd.mace.manchester.ac.uk/ercoftac/lib/exe/fetch.php"
           "?media=cdata:case020:t3ay.dat")
T3A_PAGE = "http://cfd.mace.manchester.ac.uk/ercoftac/doku.php?id=cases:case020"


def fetch_source(key_id, rel):
    """(download URL, the page a user downloads it from by hand, the local
    file names ``--from DIR`` looks for, relative to DIR) for a fetch key."""
    if key_id == "ercoftac-t3a":
        return T3A_URL, T3A_PAGE, [Path("t3ay.dat")]
    n = Path(rel).stem[len("chan"):]
    name = "chan{}.means".format(n)
    return (MKM_URL.format(N=n), MKM_PAGE,
            [Path(name), Path("chan{}".format(n)) / "profiles" / name])


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


class FenceError(Exception):
    """The fenced block a write key pins cannot be extracted from SPEC-LIT."""


def fenced_block(rel):
    """The bytes of the ```text fence SPEC-LIT pins for reference/<rel>."""
    text = (ROOT / "rust" / "SPEC-LIT.md").read_bytes()
    anchor = "reference/{}`:".format(rel).encode()
    at = text.find(anchor)
    if at < 0:
        raise FenceError("answer key {0}: no 'reference/{0}`:' anchor in "
                         "rust/SPEC-LIT.md".format(rel))
    opening = text.find(b"```text\n", at)
    if opening < 0:
        raise FenceError("answer key {0}: no ```text fence follows the "
                         "anchor".format(rel))
    start = opening + len(b"```text\n")
    end = text.find(b"```", start)
    if end < 0:
        raise FenceError("answer key {0}: the fence never closes".format(rel))
    data = text[start:end]
    if b"\r" in data:
        raise FenceError("answer key {0}: the fenced block holds CR "
                         "bytes".format(rel))
    return data


def place(rel, data):
    path = ROOT / "reference" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def cmd_write():
    staged = []
    for key_id, rel, how, pin, _size in KEYS:
        if how != "write":
            continue
        try:
            data = fenced_block(rel)
        except FenceError as e:
            print(e)
            return 1
        got = sha256_hex(data)
        if got != pin:
            print("answer key {0}: rust/SPEC-LIT.md's fenced block has "
                  "sha256 {1}, not the pinned {2}".format(key_id, got, pin))
            return 1
        staged.append((rel, pin, data))
    for rel, pin, data in staged:
        place(rel, data)
        print("wrote reference/{0} sha256 {1}".format(rel, pin))
    return 0

def cmd_fetch(from_dir):
    rc = 0
    for key_id, rel, how, pin, _size in KEYS:
        if how != "fetch":
            continue
        url, page, names = fetch_source(key_id, rel)
        if from_dir is not None:
            local = next((Path(from_dir) / p for p in names
                          if (Path(from_dir) / p).exists()), None)
            if local is None:
                print("answer key {0}: no {1} under {2}"
                      .format(key_id, names[0], from_dir))
                rc = 1
                continue
            data = local.read_bytes()
        else:
            try:
                with urllib.request.urlopen(url, timeout=120) as r:
                    data = r.read()
            except Exception as e:  # the host, not the tree
                print("answer key {0}: the fetch failed: {1}".format(key_id, e))
                print("answer key {0}: to place it by hand, download {1} from "
                      "{2} into a folder DIR and run: python "
                      "tools/keys/fetch_answer_keys.py fetch --from DIR"
                      .format(key_id, names[0], page))
                rc = 1
                continue
        got = sha256_hex(data)
        if got != pin:
            print("answer key {0}: reference/{1} has sha256 {2}, not the "
                  "pinned {3}".format(key_id, rel, got, pin))
            rc = 1
            continue
        place(rel, data)
        print("fetched reference/{0} sha256 {1}".format(rel, pin))
    return rc

def manifest_rows():
    """The table's rows of reference/PROVENANCE.md, 7 stripped cells each."""
    rows = []
    text = (ROOT / "reference" / "PROVENANCE.md").read_text(encoding="utf-8")
    for line in text.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = [c.strip() for c in line.split("|")]
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]
        if len(cells) != 7:
            continue
        if cells[0] == "id" or cells[0].startswith("-"):
            continue
        rows.append(cells)
    return rows


def cmd_check():
    ok = missing = failed = 0
    rows = manifest_rows()
    for key_id, rel, how, pin, _size in KEYS:
        want = "file" if how == "write" else "fetched"
        bad = []
        row = next((r for r in rows if r[0] == key_id), None)
        if row is None:
            bad.append("answer key {0}: reference/PROVENANCE.md has no row "
                       "naming it".format(key_id))
        elif (row[1], row[2], row[6]) != (want, rel, pin):
            bad.append("answer key {0}: reference/PROVENANCE.md's row reads "
                       "kind {1}, file {2}, sha256 {3}, not {4}, {5}, {6}"
                       .format(key_id, row[1], row[2], row[6], want, rel, pin))
        if how == "write":
            try:
                got = sha256_hex(fenced_block(rel))
            except FenceError as e:
                bad.append(str(e))
            else:
                if got != pin:
                    bad.append("answer key {0}: rust/SPEC-LIT.md's fenced "
                               "block has sha256 {1}, not the pinned {2}"
                               .format(key_id, got, pin))
        if bad:
            for line in bad:
                print(line)
            failed += 1
            continue
        path = ROOT / "reference" / rel
        if not path.exists():
            print("answer key {0} missing: reference/{1} (run: {2})"
                  .format(key_id, rel, how))
            if how == "fetch":
                missing += 1
            else:
                failed += 1
            continue
        got = sha256_hex(path.read_bytes())
        if got != pin:
            print("answer key {0}: reference/{1} has sha256 {2}, not the "
                  "pinned {3}".format(key_id, rel, got, pin))
            failed += 1
        else:
            ok += 1
    print("check: {0} ok, {1} missing, {2} failed".format(ok, missing, failed))
    return 0 if failed == 0 else 1

def main():
    ap = argparse.ArgumentParser(
        description="place and check the SPEC-LIT 110 and 144.6 answer keys")
    ap.add_argument("command", nargs="?", default="check",
                    choices=("write", "fetch", "check"))
    ap.add_argument("--check", action="store_const", dest="command",
                    const="check", help="alias of the check subcommand")
    ap.add_argument("--from", dest="from_dir", default=None, metavar="DIR",
                    help="fetch: take chan<N>.means and t3ay.dat from DIR, "
                    "not the host")
    args = ap.parse_args()
    if args.command == "write":
        return cmd_write()
    if args.command == "fetch":
        return cmd_fetch(args.from_dir)
    return cmd_check()


if __name__ == "__main__":
    sys.exit(main())
