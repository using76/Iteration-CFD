#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""validate_record.py - the record of one full ofgpu-validate run (SPEC-LIT §110.5, docs/17 VAL-82).

`sections` splits a run's `-json` document into its sections by the order of the run's
stdout (banners, 'not selected' skip rows, row lines); `merge` joins one build's runs;
`render` prints the status, gate and section blocks; and `check` (`--write`) holds the
README and guidebook blocks between their markers equal to what the records render.
Python standard library only.

Usage:
  python tools/validate_record.py sections RUN.json STDOUT.txt -o OUT.json
  python tools/validate_record.py merge A.json [B.json ...] [--not-run N=REASON ...] -o BUILD.json
  python tools/validate_record.py render {status|gates|sections} {en|ko} [--records DIR]
  python tools/validate_record.py check [--write] [--records DIR] FILE [FILE ...]
  python tools/validate_record.py selftest
"""
import argparse
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = os.path.join(ROOT, "docs", "validate-run")
SCHEMA = "ofgpu-validate-record/1"
BANNER_RE = re.compile(r"^=== (.*) ===\s*$")
UNSEL_RE = re.compile(r"^  skip  section (\d+): ")
SUMMARY_RE = re.compile(r"^(\d+)/(\d+) checks passed\s*$")
STAMP_RE = re.compile(r"^\s*\d+\.\d\d ")
STAMP_HEAD_RE = re.compile(r"^\s*(\d+\.\d\d) (.*)$")
ROW_KINDS = ("  ok   ", "  FAIL ", "  skip  ")
AUDIT_RE = re.compile(r"\((S69\.[234]|SPEC-LIT 94\.3)\)")
VERDICT_RE = re.compile(r"^\s+\*\* (.*) (MISSES|OPEN) \*\* \(")
BLOCKS = ("status", "gates", "sections")
MARK_RE = re.compile(r"^<!-- validate-run:([A-Za-z]+):([a-z]+) (begin|end) -->\s*$")
def _unstamp(line, stamped):
    m = STAMP_HEAD_RE.match(line) if stamped else None
    return m.group(2) if m else line


def _stamp(line, stamped):
    m = STAMP_HEAD_RE.match(line) if stamped else None
    return float(m.group(1)) if m else None


def _row_word(s):
    return s[2:].split(" ", 1)[0]


def _want_word(row):
    if row.get("kind") == "skip":
        return "skip"
    return "ok" if row.get("ok") else "FAIL"


def _match_row(s, row, ri):
    word, want = _row_word(s), _want_word(row)
    if word != want:
        raise ValueError("stdout rows do not match the run's rows: row %d (seq %r) line says %r, run says %r"
                         % (ri, row.get("seq"), word, want))


def sections_spec(run):
    argv = run.get("argv") or []
    if "-sections" in argv and argv.index("-sections") + 1 < len(argv):
        return argv[argv.index("-sections") + 1]
    return None


def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=1) + "\n")
def split_sections(run, stdout_lines):
    """The section record of one run, from its document and its stdout (raw or stamped)."""
    if not stdout_lines:
        raise ValueError("stdout is empty")
    stamped = bool(STAMP_RE.match(stdout_lines[0]))
    rows = run.get("rows") or []
    k = ri = 0
    titles = {}
    all_secs = []          # every banner or unselected line, in run order
    cur = None             # the selected section now open
    audit = {"checks": 0, "passed": 0}
    found = []             # one (gate, verdict, section) per stdout verdict line, in order
    summary_stamp = None
    for raw in stdout_lines:
        body = raw.rstrip("\r\n")
        s = _unstamp(body, stamped)
        st = _stamp(body, stamped)
        bm = BANNER_RE.match(s)
        if bm:
            k += 1
            titles[k] = bm.group(1)
            cur = {"n": k, "title": bm.group(1), "_rows": [], "_skips": 0, "_start": st}
            all_secs.append(cur)
            continue
        um = UNSEL_RE.match(s)
        if um and "not selected by -sections" in s:
            k += 1
            n = int(um.group(1))
            if n != k:
                raise ValueError("stdout announces section %d where section %d was expected" % (n, k))
            if ri >= len(rows):
                raise ValueError("stdout rows do not match the run's rows: extra row line %r" % s[:40])
            _match_row(s, rows[ri], ri)
            what = rows[ri].get("what") or ""
            lead = "section %d: " % n
            if what.startswith(lead):
                titles[k] = what[len(lead):]
            else:
                tail = s[7:]
                rest = tail[len(lead):] if tail.startswith(lead) else tail
                titles[k] = re.split(r"\s{2,}|\s+not selected by -sections", rest)[0].rstrip()
            ri += 1
            cur = None
            all_secs.append({"n": k, "title": titles[k], "_start": st})
            continue
        sm = SUMMARY_RE.match(s)
        if sm:
            summary_stamp = st
            continue
        vm = VERDICT_RE.match(s)
        if vm:
            if cur is None:
                raise ValueError("verdict line outside any selected section: %r" % s[:60])
            found.append((vm.group(1), vm.group(2), cur["n"]))
            continue
        if not s.startswith(ROW_KINDS):
            continue
        if ri >= len(rows):
            raise ValueError("stdout rows do not match the run's rows: extra row line %r" % s[:40])
        _match_row(s, rows[ri], ri)
        row = rows[ri]
        ri += 1
        if row.get("kind") == "check" and AUDIT_RE.search(row.get("what") or ""):
            audit["checks"] += 1
            if row.get("ok"):
                audit["passed"] += 1
            continue
        if cur is None:
            raise ValueError("row line before the first banner / outside any selected section: %r" % s[:60])
        if row.get("kind") == "skip":
            cur["_skips"] += 1
        else:
            cur["_rows"].append(row)
    if ri != len(rows):
        raise ValueError("stdout rows do not match the run's rows: %d row line(s) vs %d row(s) in the run"
                         % (ri, len(rows)))
    gates = run.get("gates") or []
    if len(found) != len(gates):
        raise ValueError("stdout verdict lines do not match the run's gates: %d verdict line(s) vs %d gate(s) in the run"
                         % (len(found), len(gates)))
    for i, ((name, verdict, _sec), g) in enumerate(zip(found, gates)):
        if (name, verdict) != (g.get("gate"), g.get("verdict")):
            raise ValueError("stdout verdict lines do not match the run's gates: verdict line %d says %r %r, gate %d says %r %r"
                             % (i + 1, name, verdict, i + 1, g.get("gate"), g.get("verdict")))
    return _finish(run, all_secs, titles, audit, summary_stamp, stamped, k, found)


def _finish(run, all_secs, titles, audit, summary_stamp, stamped, k, verdicts):
    """Count and time each selected section and assemble the record."""
    selected = [sec for sec in all_secs if "_rows" in sec]
    out_secs = []
    for i, sec in enumerate(all_secs):
        if "_rows" not in sec:
            continue
        checks = sec["_rows"]
        n_checks = len(checks)
        passed = sum(1 for r in checks if r.get("ok"))
        failed = n_checks - passed
        status = "misses" if failed > 0 else ("holds" if n_checks > 0 else "no rows")
        end = all_secs[i + 1].get("_start") if i + 1 < len(all_secs) else summary_stamp
        if stamped and sec["_start"] is not None and end is not None:
            seconds = round(end - sec["_start"], 1)
        elif len(selected) == 1:
            seconds = round(run.get("wallSeconds") or 0.0, 1)
        else:
            seconds = None
        out_secs.append({"n": sec["n"], "title": sec["title"], "status": status,
                         "checks": n_checks, "passed": passed, "failed": failed,
                         "skips": sec["_skips"], "seconds": seconds})
    out_secs.sort(key=lambda sec: sec["n"])
    machine = run.get("machine") or {}
    source = {"runId": run.get("runId"), "startedAt": run.get("startedAt"),
              "gitSha": run.get("gitSha"), "sectionsSpec": sections_spec(run),
              "wallSeconds": round(run.get("wallSeconds") or 0.0, 1),
              "exitCode": run.get("exitCode"), "aborted": run.get("aborted"),
              "totals": run.get("totals")}
    return {"schema": SCHEMA,
            "build": "f64" if machine.get("precision") == "double" else "single",
            "sources": [source], "sectionCount": k, "sections": out_secs,
            "titles": titles, "audit": audit,
            "gates": [{"gate": g.get("gate"), "verdict": g.get("verdict"),
                       "how": g.get("how"), "against": g.get("against"),
                       "headline": g.get("headline"), "section": sec}
                      for g, (_name, _verdict, sec) in zip(run.get("gates") or [], verdicts)]}


def _merge_gates(gates):
    """All registrations of all records, sorted by section (stable, so a section's own order is kept)."""
    return sorted(gates, key=lambda g: g.get("section") if g.get("section") is not None else 10 ** 9)
def merge_records(records, not_run):
    """Join one build's records: each section from exactly one record, or NOT RUN with its reason."""
    if not records:
        raise ValueError("merge needs at least one record")
    builds = sorted({r.get("build") for r in records})
    if len(builds) > 1:
        raise ValueError("merge mixes builds: %s" % ", ".join(map(str, builds)))
    counts = sorted({r.get("sectionCount") for r in records})
    if len(counts) > 1:
        raise ValueError("merge mixes sectionCounts: %s" % ", ".join(map(str, counts)))
    total = records[0]["sectionCount"]
    by_n = {}
    for r in records:
        for sec in r.get("sections") or []:
            if sec["n"] in by_n:
                raise ValueError("section %d appears in two records" % sec["n"])
            by_n[sec["n"]] = sec
    for n in not_run:
        if n in by_n:
            raise ValueError("section %d is in a record but is marked not-run" % n)
    titles = {}
    for r in records:
        for key, val in (r.get("titles") or {}).items():
            try:
                titles[int(key)] = val
            except (TypeError, ValueError):
                pass
    sections = []
    for n in range(1, total + 1):
        if n in by_n:
            sections.append(by_n[n])
        elif n in not_run:
            sections.append({"n": n, "title": titles.get(n), "status": "NOT RUN",
                             "reason": not_run[n], "checks": 0, "passed": 0,
                             "failed": 0, "skips": 0, "seconds": None})
        else:
            raise ValueError("section %d is in no record" % n)
    return {"schema": SCHEMA, "build": records[0]["build"],
            "sources": [src for r in records for src in r.get("sources") or []],
            "sectionCount": total, "sections": sections, "titles": titles,
            "audit": {"checks": sum((r.get("audit") or {}).get("checks", 0) for r in records),
                      "passed": sum((r.get("audit") or {}).get("passed", 0) for r in records)},
            "gates": _merge_gates([g for r in records for g in r.get("gates") or []])}
def _comma(x):
    return "{:,}".format(int(round(x)))


def _stats(rec):
    """Counts for render: sections by status, checks and passes, wall seconds, gates by verdict."""
    st = {"holds": 0, "misses": 0, "norows": 0, "notrun": 0}
    secs = (rec or {}).get("sections") or []
    for sec in secs:
        status = sec.get("status")
        if status == "holds":
            st["holds"] += 1
        elif status == "misses":
            st["misses"] += 1
        elif status == "NOT RUN":
            st["notrun"] += 1
        else:
            st["norows"] += 1
    audit = (rec or {}).get("audit") or {}
    st["checks"] = sum(sec.get("checks", 0) for sec in secs) + (audit.get("checks") or 0)
    st["passed"] = sum(sec.get("passed", 0) for sec in secs) + (audit.get("passed") or 0)
    srcs = (rec or {}).get("sources") or []
    st["wall"] = sum(src.get("wallSeconds") or 0 for src in srcs)
    st["S"] = (rec or {}).get("sectionCount", 0)
    st["date"] = (srcs[0].get("startedAt") or "")[:10] if srcs else "?"
    unfiltered = sum(1 for src in srcs if src.get("sectionsSpec") is None)
    if len(srcs) == 1 and unfiltered == 1:
        st["how"] = ("one unfiltered run", "필터 없는 실행 1회")
    elif len(srcs) == 1:
        st["how"] = ("one run", "실행 1회")
    else:
        st["how"] = ("%d runs" % len(srcs), "실행 %d회" % len(srcs))
    gates = (rec or {}).get("gates") or []
    st["m"] = sum(1 for g in gates if g.get("verdict") == "MISSES")
    st["o"] = sum(1 for g in gates if g.get("verdict") == "OPEN")
    return st
def render_status(F, S, lang):
    """The fenced status block: a three-line stanza per build."""
    lines = ["```"]
    for rec, tag in ((F, "f64"), (S, "f32")):
        label = ("ofgpu-validate (%s)" % tag).ljust(23)
        if rec is None:
            lines.append(label + "not recorded")
            continue
        st = _stats(rec)
        pad = " " * 23
        lines.append("%s%d / %d checks passed in %d sections (%s, %s, %s s; SPEC-LIT §110.5)"
                     % (label, st["passed"], st["checks"], st["S"], st["date"],
                        st["how"][0 if lang == "en" else 1], _comma(st["wall"])))
        if lang == "en":
            lines.append(pad + "%d sections hold, %d miss, %d have no rows, %d NOT RUN"
                         % (st["holds"], st["misses"], st["norows"], st["notrun"]))
            lines.append(pad + "then %d gates registered MISSES and %d OPEN, named below"
                         % (st["m"], st["o"]))
        else:
            lines.append(pad + "section %d개 중 %d개 통과, %d개 미달, 행 없음 %d개, NOT RUN %d개"
                         % (st["S"], st["holds"], st["misses"], st["norows"], st["notrun"]))
            lines.append(pad + "이어서 판정이 MISSES로 등록된 게이트 %d개와 OPEN %d개 (아래 표)"
                         % (st["m"], st["o"]))
    lines.append("```")
    return "\n".join(lines)


def _esc(cell):
    return str(cell).replace("|", "\\|")
def render_gates(F, S, lang):
    """The gate table: one row per f64 registration, MISSES first then OPEN (record order), then
    f32 registrations with no f64 pair; F and S pair by (gate name, occurrence index from 0)."""
    lines = (["| # | Gate | f64 | f32 | Compared against: the verdict |"] if lang == "en"
             else ["| # | 게이트 | f64 | f32 | 비교 대상: 판정 |"])
    lines.append("|---|---|---|---|---|")

    def keyed(gates):
        seen, out = {}, []
        for g in gates:
            name = g.get("gate")
            occ = seen.get(name, 0)
            seen[name] = occ + 1
            out.append((name, occ, g))
        return out

    fkeys = keyed((F or {}).get("gates") or [])
    skeys = keyed((S or {}).get("gates") or [])
    sat = {(name, occ): g for name, occ, g in skeys}
    frows = [(g, sat.get((name, occ))) for name, occ, g in fkeys]
    paired = {(name, occ) for name, occ, _g in fkeys if (name, occ) in sat}
    ordered = ([r for r in frows if r[0].get("verdict") == "MISSES"]
               + [r for r in frows if r[0].get("verdict") != "MISSES"]
               + [(None, g) for name, occ, g in skeys if (name, occ) not in paired])
    for i, (fgate, sgate) in enumerate(ordered, 1):
        own = fgate if fgate is not None else sgate
        lines.append("| %d | %s | %s | %s | %s |" % (
            i, _esc(own.get("gate")),
            fgate.get("verdict") if fgate is not None else "-",
            sgate.get("verdict") if sgate is not None else "-",
            _esc("%s: %s" % (own.get("against"), own.get("headline")))))
    return "\n".join(lines)


def _sec_cells(sec):
    if sec.get("status") == "NOT RUN":
        return "NOT RUN", "-"
    if sec.get("checks", 0) == 0:
        rows = "-"
    else:
        rows = "%d/%d" % (sec.get("passed", 0), sec.get("checks", 0))
        if sec.get("status") == "misses":
            rows = "**%s**" % rows
    return rows, ("-" if sec.get("seconds") is None else "%.1f" % sec["seconds"])


def render_sections(F, S, lang):
    """The section table: one row per f64 section, its f32 cells from the f32 section of the same number."""
    lines = (["| # | section | f64 rows | f64 s | f32 rows | f32 s |"] if lang == "en"
             else ["| # | 절 | f64 행 | f64 s | f32 행 | f32 s |"])
    lines.append("|---|---|---|---|---|---|")
    smap = {sec["n"]: sec for sec in (S or {}).get("sections") or []}
    for sec in (F or {}).get("sections") or []:
        f_rows, f_secs = _sec_cells(sec)
        other = smap.get(sec["n"])
        s_rows, s_secs = _sec_cells(other) if other else ("-", "-")
        lines.append("| %d | %s | %s | %s | %s | %s |" % (
            sec["n"], _esc(sec.get("title") or "-"), f_rows, f_secs, s_rows, s_secs))
    return "\n".join(lines)
def render(block, lang, F, S):
    return {"status": render_status, "gates": render_gates, "sections": render_sections}[block](F, S, lang)


def _markers(lines, path):
    """The marker pairs as (block, lang, begin line, end line); nested or unmatched markers raise ValueError."""
    stack, blocks = [], []
    for i, line in enumerate(lines):
        m = MARK_RE.match(line)
        if not m:
            continue
        block, lang, which = m.group(1).lower(), m.group(2), m.group(3)
        if block not in BLOCKS or lang not in ("en", "ko"):
            raise ValueError("%s:%d: unknown validate-run marker %s:%s" % (path, i + 1, block, lang))
        if which == "begin":
            if stack:
                raise ValueError("%s:%d: nested marker %s:%s inside open %s:%s"
                                 % (path, i + 1, block, lang, stack[-1][0], stack[-1][1]))
            stack.append((block, lang, i))
        elif not stack:
            raise ValueError("%s:%d: unmatched end marker %s:%s" % (path, i + 1, block, lang))
        else:
            b, l, bi = stack.pop()
            blocks.append((b, l, bi, i))
    if stack:
        b, l, bi = stack[-1]
        raise ValueError("%s:%d: unclosed marker %s:%s" % (path, bi + 1, b, l))
    return blocks


def check_text(text, F, S, path="<text>", write=False):
    """Compare each marked block with its render; with write, return the text with every block re-rendered."""
    lines = text.split("\n")
    blocks = _markers(lines, path)
    if not blocks:
        return None, ["none %s" % path]
    reports, out, shift = [], list(lines), 0
    for block, lang, bi, ei in blocks:
        want = render(block, lang, F, S)
        tag = "%s:%s" % (block, lang)
        if "\n".join(lines[bi + 1:ei]) == want:
            reports.append("ok   %s %s" % (path, tag))
        elif write:
            new = want.split("\n")
            out[bi + 1 + shift:ei + shift] = new
            shift += len(new) - (ei - bi - 1)
            reports.append("wrote %s %s" % (path, tag))
        else:
            reports.append("DIFF %s %s" % (path, tag))
    return (None if not write else "\n".join(out)), reports
def _run1_out1():
    """Case 1: an unfiltered stamped run, two sections and the four audit rows."""
    spec = [("check", "a1", True), ("check", "a2", False), ("check", "b1", True),
            ("skip", "b2", None), ("check", "x (S69.2)", True), ("check", "y (S69.3)", True),
            ("check", "z (SPEC-LIT 94.3)", True), ("check", "w (S69.4)", True)]
    run = {"runId": "r1", "startedAt": "1970-01-01T00:00:00.000Z", "wallSeconds": 4.25,
           "exitCode": 1, "aborted": False, "argv": ["ofgpu-validate", "-json"],
           "gitSha": "0" * 40, "machine": {"precision": "double"},
           "totals": {"total": 10, "failures": 1, "skipped": 1, "replayed": 0},
           "gates": [{"gate": "DST-X", "verdict": "MISSES", "how": "Live",
                      "against": "analytic", "headline": "h|pipe"}],
           "rows": [{"seq": i, "kind": kind, "what": what, "ok": ok}
                    for i, (kind, what, ok) in enumerate(spec, 1)]}
    body = [(0.0, "ofgpu validation | X | double"), (1.0, "=== A ==="),
            (1.1, "  ok   a1 fine"), (1.2, "  FAIL a2 off"),
            (1.15, "        ** DST-X MISSES ** (run live above): h|pipe"), (3.5, "=== B ==="),
            (3.6, "  ok   b1 fine"), (3.7, "  skip  b2 because"), (3.8, "  ok   x (S69.2) fine"),
            (3.9, "  ok   y (S69.3) fine"), (4.0, "  ok   z (SPEC-LIT 94.3) fine"),
            (4.1, "  ok   w (S69.4) fine"), (4.25, "5/6 checks passed")]
    return run, ["%9.2f %s" % pair for pair in body]


def _case1():
    rec = split_sections(*_run1_out1())
    a, b = rec["sections"]
    assert (a["checks"], a["passed"], a["failed"], a["skips"], a["seconds"], a["status"]) \
        == (2, 1, 1, 0, 2.5, "misses")
    assert (b["checks"], b["passed"], b["failed"], b["skips"], b["seconds"], b["status"]) \
        == (1, 1, 0, 1, 0.8, "holds")
    assert rec["audit"] == {"checks": 4, "passed": 4} and rec["sectionCount"] == 2
    assert rec["build"] == "f64" and rec["titles"] == {1: "A", 2: "B"}
    assert rec["sources"][0]["sectionsSpec"] is None
    assert rec["gates"][0]["section"] == 1
def _run2_out2():
    """Case 2: a raw run of -sections 2, one of three sections selected."""
    run = {"runId": "r2", "startedAt": "1970-01-02T00:00:00.000Z", "wallSeconds": 7.25,
           "exitCode": 0, "aborted": False, "argv": ["ofgpu-validate", "-sections", "2", "-json"],
           "gitSha": "0" * 40, "machine": {"precision": "single"},
           "totals": {"total": 1, "failures": 0, "skipped": 2, "replayed": 0}, "gates": [],
           "rows": [{"seq": 1, "kind": "skip", "what": "section 1: geometry setup"},
                    {"seq": 2, "kind": "check", "what": "c1", "ok": True},
                    {"seq": 3, "kind": "skip", "what": "section 3: turbulence models"}]}
    out = ["ofgpu validation | X | single",
           "  skip  section 1: geometry setup    not selected by -sections 2",
           "=== solver cookies ===",
           "  ok   c1 fine",
           "  skip  section 3: turbulence models    not selected by -sections 2",
           "1/1 checks passed"]
    return run, out


def _case2():
    rec = split_sections(*_run2_out2())
    assert rec["build"] == "single" and rec["sectionCount"] == 3
    assert rec["sources"][0]["sectionsSpec"] == "2"
    s2 = rec["sections"][0]
    assert s2["n"] == 2 and (s2["checks"], s2["passed"], s2["seconds"]) == (1, 1, 7.2)
    assert rec["titles"] == {1: "geometry setup", 2: "solver cookies", 3: "turbulence models"}


def _case3():
    run, out = _run1_out1()
    run["rows"][0] = dict(run["rows"][0], ok=False)
    try:
        split_sections(run, out)
    except ValueError as e:
        assert "do not match" in str(e)
        return
    raise AssertionError("row mismatch did not raise ValueError")
def _case4():
    rec2 = split_sections(*_run2_out2())
    rec_b = {"schema": SCHEMA, "build": "single", "sources": [], "sectionCount": 3,
             "sections": [{"n": 1, "title": "geometry setup", "status": "holds", "checks": 5,
                           "passed": 5, "failed": 0, "skips": 0, "seconds": 11.0}],
             "titles": {1: "geometry setup"}, "audit": {"checks": 2, "passed": 2}, "gates": []}
    merged = merge_records([rec2, rec_b], {3: "over 600 s"})
    assert merged["sectionCount"] == 3 and [s["n"] for s in merged["sections"]] == [1, 2, 3]
    nr = merged["sections"][2]
    assert nr["status"] == "NOT RUN" and nr["reason"] == "over 600 s"
    assert nr["title"] == "turbulence models"
    assert merged["audit"] == {"checks": 2, "passed": 2}
    try:
        merge_records([rec2, rec2], {})
    except ValueError as e:
        assert "two records" in str(e)
        return
    raise AssertionError("duplicate section did not raise ValueError")


def _case5():
    rec1 = split_sections(*_run1_out1())
    lines = render_status(rec1, None, "en").split("\n")
    assert lines[1] == ("ofgpu-validate (f64)   6 / 7 checks passed in 2 sections "
                        "(1970-01-01, one unfiltered run, 4 s; SPEC-LIT §110.5)"), lines[1]
    assert lines[2] == " " * 23 + "1 sections hold, 1 miss, 0 have no rows, 0 NOT RUN"
    assert lines[3] == " " * 23 + "then 1 gates registered MISSES and 0 OPEN, named below"
    assert lines[4] == "ofgpu-validate (f32)   not recorded"
    assert lines[0] == "```" and lines[-1] == "```"
    rec2 = split_sections(*_run2_out2())
    assert ", one run, " in render_status(rec2, None, "en").split(chr(10))[1]
def _case6():
    rec1 = split_sections(*_run1_out1())
    text = ("intro\n<!-- validate-run:status:en begin -->\nstale\n"
            "<!-- validate-run:status:en end -->\ntail\n")
    _, reports = check_text(text, rec1, None)
    assert any(r.startswith("DIFF") for r in reports), reports
    new, reports = check_text(text, rec1, None, write=True)
    assert any(r.startswith("wrote") for r in reports), reports
    rows = new.split("\n")
    i0 = rows.index("<!-- validate-run:status:en begin -->")
    i1 = rows.index("<!-- validate-run:status:en end -->")
    assert "\n".join(rows[i0 + 1:i1]) == render_status(rec1, None, "en")
    _, reports = check_text(new, rec1, None)
    assert any(r.startswith("ok   ") for r in reports), reports
    try:
        check_text("<!-- validate-run:status:en begin -->\nx\n", rec1, None)
    except ValueError:
        return
    raise AssertionError("unclosed marker did not raise ValueError")


def _run7_out7():
    """Case 7: one section; the name G is registered twice (MISSES, OPEN) and H once (OPEN)."""
    run = {"runId": "r7", "startedAt": "1970-01-07T00:00:00.000Z", "wallSeconds": 2.0,
           "exitCode": 1, "aborted": False, "argv": ["ofgpu-validate", "-json"],
           "gitSha": "0" * 40, "machine": {"precision": "double"},
           "totals": {"total": 3, "failures": 1, "skipped": 0, "replayed": 0},
           "gates": [{"gate": "G", "verdict": "MISSES", "how": "Live",
                      "against": "a|1", "headline": "first G"},
                     {"gate": "G", "verdict": "OPEN", "how": "Live",
                      "against": "a|2", "headline": "second G"},
                     {"gate": "H", "verdict": "OPEN", "how": "Live",
                      "against": "a3", "headline": "only H"}],
           "rows": [{"seq": i, "kind": "check", "what": "c%d" % i, "ok": True}
                    for i in (1, 2, 3)]}
    out = ["ofgpu validation | X | double",
           "=== S ===",
           "  ok   c1 fine", "  ok   c2 fine", "  ok   c3 fine",
           "    ** G MISSES ** (run live above): a|1: first G",
           "    ** G OPEN ** (run live above): a|2: second G",
           "    ** H OPEN ** (run live above): a3: only H",
           "2/3 checks passed"]
    return run, out


def _case7():
    rec = split_sections(*_run7_out7())
    assert [(g["gate"], g["verdict"], g["section"]) for g in rec["gates"]] \
        == [("G", "MISSES", 1), ("G", "OPEN", 1), ("H", "OPEN", 1)]
    body = render_gates(rec, rec, "en").split("\n")[2:]
    assert [r.split("|")[2].strip() for r in body] == ["G", "G", "H"]
    assert [(r.split("|")[3].strip(), r.split("|")[4].strip()) for r in body] \
        == [("MISSES", "MISSES"), ("OPEN", "OPEN"), ("OPEN", "OPEN")]
    assert (_stats(rec)["m"], _stats(rec)["o"]) == (1, 2)


def _case8():
    run, out = _run7_out7()
    out[6], out[7] = out[7], out[6]
    try:
        split_sections(run, out)
    except ValueError as e:
        assert "do not match the run's gates" in str(e)
        return
    raise AssertionError("verdict order mismatch did not raise ValueError")


def _case9():
    """Merge of two records of one build: gates keep every registration, sorted by section."""
    def hold(n, t):
        return {"n": n, "title": t, "status": "holds", "checks": 1, "passed": 1,
                "failed": 0, "skips": 0, "seconds": None}

    def rec(secs, titles, gates):
        return {"schema": SCHEMA, "build": "f64", "sources": [], "sectionCount": 3,
                "sections": secs, "titles": titles,
                "audit": {"checks": len(secs), "passed": len(secs)}, "gates": gates}

    x = rec([hold(2, "two")], {2: "two"}, [{"gate": "P", "verdict": "MISSES", "section": 2}])
    y = rec([hold(1, "one"), hold(3, "three")], {1: "one", 3: "three"},
            [{"gate": "Q", "verdict": "OPEN", "section": 1},
             {"gate": "R", "verdict": "OPEN", "section": 3}])
    merged = merge_records([x, y], {})
    assert [g["gate"] for g in merged["gates"]] == ["Q", "P", "R"]
    assert [s["n"] for s in merged["sections"]] == [1, 2, 3]


def _selftest():
    """Synthetic runs in memory: no file, no GPU."""
    cases = (("split stamped unfiltered 2-section run", _case1),
             ("split filtered raw run -sections 2", _case2),
             ("row mismatch raises ValueError", _case3),
             ("merge not-run and duplicate rejection", _case4),
             ("render status en", _case5),
             ("check/write round trip on a string", _case6),
             ("repeated gate names keep every registration", _case7),
             ("verdict line order mismatch raises ValueError", _case8),
             ("merge sorts gates by section (stable)", _case9))
    for name, fn in cases:
        fn()
        print("[ok] %s" % name)
    print("selftest: %d passed" % len(cases))
def _load_records(directory):
    f_path = os.path.join(directory, "f64.json")
    s_path = os.path.join(directory, "single.json")
    if not os.path.exists(f_path):
        raise SystemExit("f64 record not found: %s" % f_path)
    return _load_json(f_path), (_load_json(s_path) if os.path.exists(s_path) else None)


def _cmd_sections(args):
    with open(args.stdout, encoding="utf-8") as f:
        lines = f.read().splitlines()
    rec = split_sections(_load_json(args.run), lines)
    _write_json(args.out, rec)
    print("wrote %s" % args.out)
    print("build %s, sources %d, sectionCount %d, selected %d"
          % (rec["build"], len(rec["sources"]), rec["sectionCount"], len(rec["sections"])))
    for sec in rec["sections"]:
        print('section %d "%s": checks %d passed %d failed %d skips %d seconds %s %s'
              % (sec["n"], (sec["title"] or "")[:60], sec["checks"], sec["passed"],
                 sec["failed"], sec["skips"], sec["seconds"], sec["status"]))
    print("audit checks %d passed %d" % (rec["audit"]["checks"], rec["audit"]["passed"]))


def _cmd_merge(args):
    not_run = {}
    for item in args.not_run:
        n, sep, reason = item.partition("=")
        if not sep:
            raise SystemExit("--not-run expects N=REASON, got %r" % item)
        not_run[int(n)] = reason
    merged = merge_records([_load_json(p) for p in args.records], not_run)
    _write_json(args.out, merged)
    st = _stats(merged)
    print("wrote %s" % args.out)
    print("build %s, sources %d, sectionCount %d: %d hold, %d miss, %d no rows, %d NOT RUN"
          % (merged["build"], len(merged["sources"]), merged["sectionCount"],
             st["holds"], st["misses"], st["norows"], st["notrun"]))
def _cmd_render(args):
    F, S = _load_records(args.records)
    print(render(args.block, args.lang, F, S))


def _cmd_check(args):
    F, S = _load_records(args.records)
    bad = False
    for path in args.files:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        try:
            new, reports = check_text(text, F, S, path=path, write=args.write)
        except ValueError as e:
            print(str(e))
            bad = True
            continue
        for r in reports:
            print(r)
            bad = bad or r.startswith("DIFF")
        if new is not None and new != text:
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(new)
    if bad:
        raise SystemExit(1)


def main(argv=None):
    p = argparse.ArgumentParser(description="The record of a full ofgpu-validate run (SPEC-LIT §110.5).")
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("sections", help="a run document and its stdout -> its section record")
    sp.add_argument("run")
    sp.add_argument("stdout")
    sp.add_argument("-o", "--out", required=True)
    sp.set_defaults(fn=_cmd_sections)
    mp = sub.add_parser("merge", help="join one build's records")
    mp.add_argument("records", nargs="+")
    mp.add_argument("--not-run", action="append", metavar="N=REASON", default=[])
    mp.add_argument("-o", "--out", required=True)
    mp.set_defaults(fn=_cmd_merge)
    rp = sub.add_parser("render", help="print the status, gates or sections block")
    rp.add_argument("block", choices=BLOCKS)
    rp.add_argument("lang", choices=("en", "ko"))
    rp.add_argument("--records", default=DEFAULT_DIR)
    rp.set_defaults(fn=_cmd_render)
    cp = sub.add_parser("check", help="check the blocks between markers (--write rewrites them)")
    cp.add_argument("--write", action="store_true")
    cp.add_argument("--records", default=DEFAULT_DIR)
    cp.add_argument("files", nargs="+")
    cp.set_defaults(fn=_cmd_check)
    stp = sub.add_parser("selftest", help="self-test on synthetic runs in memory")
    stp.set_defaults(fn=lambda a: _selftest())
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
