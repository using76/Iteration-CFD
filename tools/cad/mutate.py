#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""mutate.py - gate GC-2 of docs/16 §E.5 / §H.3: ten deterministic mutants of the nominal nozzle, each measured by the production export and judged by verify.evaluate; every mutant must be killed by a failing hard geometric row, every hard geometric row must kill one, the nominal must pass them all. Selftest only: no production path has a mutation flag.

Usage:
  python mutate.py --selftest
  python mutate.py run OUT_DIR
"""
import os
import sys
import tempfile
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common
import export
import readiness
import reqs
import runner
import verify

FIXTURE = os.path.join(HERE, "fixtures", "mutate", "nominal_set.json")
MUTANT = os.path.join(HERE, "fixtures", "mutate", "mutant.py")
MATRIX_FILE = "mutation.json"
VERSION = 1
CONTROL = ("M00", "identity", None, "the nominal through the mutation child with nothing planted")
MUTANTS = (  # (id, name, CADReview class, what is planted) - docs/16 §E.5, in the plan's order
    ("M01", "exit_d_plus5", "Size", "exit diameter +5 %: CR divided by 1.05^2, D_i kept"),
    ("M02", "wall_radial", "Logic", "the outer wall is the RADIAL offset of the wetted curve by t, not the normal one"),
    ("M03", "contraction_l_minus10", "Size", "contraction length -10 %: L_over_Di times 0.9"),
    ("M04", "outlet_face_missing", "Missing", "the fluid is the open shell without its outlet face"),
    ("M05", "bow_tie", "Logic", "the exit tube and outlet re-wired into two crossing diagonals"),
    ("M06", "axis_crossing", "Position", "the axis edge replaced by an arc dipping R_e/2 below the axis"),
    ("M07", "lip_0p05mm", "Primitive", "the body's outer exit wall chamfered to a 0.05 mm land at the outlet"),
    ("M08", "axis_on_z", "Rotation", "every BREP rotated -90 deg about +y, so the axis is +z"),
    ("M09", "stray_second_solid", "Redundant", "the wall body left inside the fluid export as a second solid"),
    ("M10", "mm_as_m", "Constant", "the STEP files written with millimetre numbers under a METRE header"),
)
EXPECTED_KILLS = {
    "M01": ["REQ-002", "REQ-003", "REQ-005"],
    "M02": ["REQ-006"],
    "M03": ["REQ-004", "REQ-005"],
    "M04": ["SYS-SOLID", "SYS-WATERTIGHT"],
    "M05": ["SYS-VALID", "SYS-WATERTIGHT"],
    "M06": ["SYS-VALID", "SYS-WATERTIGHT"],
    "M07": ["REQ-006"],
    "M08": ["REQ-005", "SYS-AXIS", "SYS-UNITS"],
    "M09": ["REQ-001", "REQ-002", "REQ-003", "SYS-SOLID", "SYS-VALID"],
    "M10": ["SYS-UNITS"],
}
USAGE = ("usage: python mutate.py --selftest" + chr(10)
         + "       python mutate.py run OUT_DIR")


def load_checks():
    """The GC-2 set through the production reqs path: accept, lock, compile (docs/16 §E.3, §E.4)."""
    fx = common.read_json(FIXTURE)
    decl, sha, dsha = reqs.load_template(reqs.NOZZLE_DIR)
    proposal = dict(fx["proposal"], vocab_sha=reqs.vocab_sha(decl))    # stamped at call time, as GUI-1 will
    rep = reqs.check(proposal, fx["brief"], decl, sha, dsha)
    if rep["status"] != "ok":
        raise RuntimeError("the GC-2 requirement set is %s with refusals %r"
                           % (rep["status"], rep.get("refusals")))
    doc = reqs.lock(rep, fx["approved_by"])
    checks_doc = reqs.compile_checks(doc, decl, dsha)
    return doc, checks_doc, decl


def split_checks(checks_doc, decl):
    """(geometric checks, excluded req ids): geometric iff a catalogue row shares primitive, where and the geometry method."""
    geo, excluded = [], []
    for c in checks_doc["checks"]:
        hit = any(row["primitive"] == c["primitive"] and list(row["where"]) == list(c["args"]["where"])
                  and row["method"] == "geometry" for row in decl["catalogue"])
        if hit:
            geo.append(c)
        else:
            excluded.append(c["req_id"])
    return geo, excluded


def measurements(out_dir, geo):
    """(req_id -> record) for each geometric check's ONE probes.json row, with the rows beside it."""
    rows = common.read_json(os.path.join(out_dir, "probes.json"))["rows"]
    meas = {}
    for c in geo:
        hit = [r for r in rows if r["primitive"] == c["primitive"]
               and list(r["where"]) == list(c["args"]["where"])]
        if len(hit) != 1:
            raise RuntimeError("probes.json has %d rows for %s (%s), want exactly 1"
                               % (len(hit), c["req_id"], c["primitive"]))
        meas[c["req_id"]] = hit[0]["record"]
    return meas, rows


def judge_dir(out_dir, doc, checks_doc, geo):
    """verify.evaluate over the whole cad-checks/1 doc and its locked set; write verdict.json; return the doc.

    SYS-MACH is absent from the measurements on purpose: with no CFD it is NE-MISSING, never a kill."""
    meas, rows = measurements(out_dir, geo)
    key = common.sha256_of(rows)
    ev = {"path": "probes.json", "sha": common.sha256_file(os.path.join(out_dir, "probes.json"))}
    vdoc = verify.evaluate(checks_doc, doc, meas, key, evidence=ev)
    reqs.write_canonical(os.path.join(out_dir, "verdict.json"), vdoc)
    return vdoc


def kills(doc, geo):
    """The req ids, in check order, of hard geometric rows whose verdict is fail; a NE row never enters."""
    ids = set(c["req_id"] for c in geo)
    return [r["req_id"] for r in doc["verdicts"]
            if r["hardness"] == "hard" and r["verdict"] == "fail" and r["req_id"] in ids]


def coverage(kill_lists, hard_geo_ids):
    """The hard geometric rows in NO kill list; empty means every hard geometric row kills at least one mutant."""
    hit = set()
    for ks in kill_lists:
        hit.update(ks)
    return [rid for rid in hard_geo_ids if rid not in hit]


def build_mutant(mid, out_dir):
    """(value, None) or (None, message): the mutant's build in a runner child; the template runs only there."""
    r = runner.run_job(MUTANT, {"template": export.TEMPLATE, "params": dict(export.NOMINAL),
                                "mutant": mid}, out_dir, entry="build",
                       timeout_s=export.BUILD_TIMEOUT_S)
    if r["status"] == "ok":
        return (r["value"], None)
    return (None, "%s: %s" % (r["rule"], r["message"][:200]))


def export_mutant(mid, out_dir, value):
    """The production export of one mutant; M10 writes millimetre numbers under the METRE header, once."""
    if mid == "M10":
        orig = export.write_step
        export.write_step = lambda shape, path, unit=export.STEP_UNIT: orig(shape.scale(1000.0), path, unit)
        try:
            export.export_build(out_dir, value, export.TEMPLATE)
        finally:
            export.write_step = orig
        return
    export.export_build(out_dir, value, export.TEMPLATE)


def row_of(out_dir, name, klass, planted, doc, geo, error):
    """One matrix row: kill list, NE map and geo verdicts from the verdict doc, readiness recorded beside."""
    ident = os.path.basename(os.path.normpath(out_dir)).split("_")[0]
    if error is not None:
        return {"id": ident, "name": name, "class": klass, "planted": planted, "status": "error",
                "error": error, "killed_by": [], "not_evaluable": {}, "verdicts": {},
                "readiness": None, "eval_key": None}
    geo_ids = set(c["req_id"] for c in geo)
    not_ev = dict((r["req_id"], r["reason_id"]) for r in doc["verdicts"]
                  if r["verdict"] == "not_evaluable" and r["req_id"] in geo_ids)
    verdicts = dict((r["req_id"], r["verdict"]) for r in doc["verdicts"] if r["req_id"] in geo_ids)
    rd = readiness.check(out_dir, readiness.H_SELFTEST_M)
    return {"id": ident, "name": name, "class": klass, "planted": planted, "status": "ok",
            "error": None, "killed_by": kills(doc, geo), "not_evaluable": not_ev,
            "verdicts": verdicts, "readiness": {"status": rd["status"], "rule": rd["rule"]},
            "eval_key": doc["eval_key"]}


def run_matrix(out_root):
    """The GC-2 matrix: nominal, control, ten mutants; mutation.json written canonically; returns the dict."""
    out_root = os.path.abspath(out_root)
    os.makedirs(out_root, exist_ok=True)
    doc, checks_doc, decl = load_checks()
    reqs.write_locked(out_root, doc)
    reqs.write_canonical(os.path.join(out_root, "checks.json"), checks_doc)
    geo, excluded = split_checks(checks_doc, decl)
    hard_geo = [c["req_id"] for c in geo if c["hardness"] == "hard"]

    def one(mid, name, klass, planted):
        out_dir = os.path.join(out_root, "%s_%s" % (mid, name))
        value, error = build_mutant(mid, out_dir)
        jdoc = None
        if error is None:
            try:
                export_mutant(mid, out_dir, value)
            except Exception as e:              # an export failure is a row status, never a kill
                error = "%s: %s" % (type(e).__name__, str(e)[:200])
        if error is None:
            jdoc = judge_dir(out_dir, doc, checks_doc, geo)
        return row_of(out_dir, name, klass, planted, jdoc, geo, error)

    nom_dir = os.path.join(out_root, "nominal")
    res = export.run_pipeline(export.TEMPLATE, dict(export.NOMINAL), nom_dir)
    if res["status"] != "ok":
        raise RuntimeError("the nominal export is %s (%s)" % (res["status"], res["rule"]))
    nominal = row_of(nom_dir, "nominal", None, "nothing", judge_dir(nom_dir, doc, checks_doc, geo), geo, None)
    control = one(CONTROL[0], CONTROL[1], CONTROL[2], CONTROL[3])
    mutants = [one(mid, name, klass, planted) for (mid, name, klass, planted) in MUTANTS]
    killed = sum(1 for m in mutants if m["killed_by"])
    geo_ids = [c["req_id"] for c in geo]
    uncovered = coverage([m["killed_by"] for m in mutants], hard_geo)
    nominal_all_pass = all(nominal["verdicts"][rid] == "pass" for rid in geo_ids)
    matrix = {
        "version": VERSION,
        "template_id": decl["template_id"],
        "template_sha": common.sha256_file(export.TEMPLATE),
        "requirements_lock": doc["lock_sha"],
        "checks_sha": common.sha256_of(checks_doc),
        "geometric": geo_ids,
        "hard_geometric": hard_geo,
        "excluded": excluded,
        "nominal": nominal,
        "control": control,
        "mutants": mutants,
        "killed": killed,
        "total": 10,
        "uncovered": uncovered,
        "nominal_all_pass": nominal_all_pass,
        "gc2": killed == 10 and not uncovered and nominal_all_pass and control["killed_by"] == [],
    }
    reqs.write_canonical(os.path.join(out_root, MATRIX_FILE), matrix)
    return matrix


def selftest():
    """The GC-2 proof: one matrix run in a temp dir, then the assertions on the files it wrote."""
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "matrix")
        assert main(["run", out]) == 0, "the matrix run did not return 0"
        matrix = common.read_json(os.path.join(out, MATRIX_FILE))
        by_id = dict((m["id"], m) for m in matrix["mutants"])

        # T1: the fixture is byte-identical and the production path compiles it to the GC-2 split.
        assert common.sha256_file(FIXTURE) == "14f5ec5003427a4c9e0444ce251dfacab820b4b433cd2cf5c6e4e4b485bb7a5d"
        doc, checks_doc, decl = load_checks()
        geo, excluded = split_checks(checks_doc, decl)
        geo_ids = [c["req_id"] for c in geo]
        hard = [c["req_id"] for c in geo if c["hardness"] == "hard"]
        assert len(checks_doc["checks"]) == 14, len(checks_doc["checks"])
        assert geo_ids == ["REQ-001", "REQ-002", "REQ-003", "REQ-004", "REQ-005", "REQ-006", "REQ-007",
                           "REQ-008", "SYS-SOLID", "SYS-VALID", "SYS-WATERTIGHT", "SYS-AXIS", "SYS-UNITS"], geo_ids
        assert hard == ["REQ-001", "REQ-002", "REQ-003", "REQ-004", "REQ-005", "REQ-006", "SYS-SOLID",
                        "SYS-VALID", "SYS-WATERTIGHT", "SYS-AXIS", "SYS-UNITS"], hard
        assert excluded == ["SYS-MACH"], excluded
        print("[ok] GC-2 set: 14 checks compiled, 13 geometric (11 hard), excluded %r" % (excluded,))

        # T2: the nominal passes every geometric row; SYS-MACH is NE-MISSING; readiness is ready.
        nom = matrix["nominal"]
        assert all(nom["verdicts"][rid] == "pass" for rid in matrix["geometric"]), nom["verdicts"]
        vd = common.read_json(os.path.join(out, "nominal", "verdict.json"))
        mach = [r for r in vd["verdicts"] if r["req_id"] == "SYS-MACH"][0]
        assert mach["verdict"] == "not_evaluable" and mach["reason_id"] == "NE-MISSING" and mach["u"] is None, mach
        assert nom["readiness"]["status"] == "ready", nom["readiness"]
        print("[ok] nominal passes 13/13 geometric rows; SYS-MACH NE-MISSING (u None); readiness ready")

        # T3: the identity control plants nothing and kills nothing.
        ctl = matrix["control"]
        assert ctl["status"] == "ok" and ctl["killed_by"] == [], (ctl["status"], ctl["killed_by"])
        a = common.read_json(os.path.join(out, "M00_identity", "probes.json"))["rows"]
        b = common.read_json(os.path.join(out, "nominal", "probes.json"))["rows"]
        assert common.canonical_json(a) == common.canonical_json(b), "the control probes differ from the nominal"
        print("[ok] control M00: probes identical to the nominal, 0 kills")

        # T4: every mutant is built, exported and killed by exactly its expected rows.
        for m in matrix["mutants"]:
            exp = EXPECTED_KILLS[m["id"]]
            assert m["status"] == "ok", "%s status %s: %s" % (m["id"], m["status"], m["error"])
            assert m["killed_by"] == exp, "%s killed_by %r, expected %r" % (m["id"], m["killed_by"], exp)
            print("[ok] %s %s (%s) killed by %s" % (m["id"], m["name"], m["class"], " ".join(m["killed_by"])))

        # T5: the kill matrix is full, and coverage can fail when a row's only killers are removed.
        assert matrix["killed"] == 10 and matrix["uncovered"] == [], (matrix["killed"], matrix["uncovered"])
        print("[ok] mutation matrix: 10/10 killed, uncovered []")
        no_m08_m10 = [m["killed_by"] for m in matrix["mutants"] if m["id"] not in ("M08", "M10")]
        no_m09 = [m["killed_by"] for m in matrix["mutants"] if m["id"] != "M09"]
        cov1 = coverage(no_m08_m10, matrix["hard_geometric"])
        cov2 = coverage(no_m09, matrix["hard_geometric"])
        assert cov1 == ["SYS-AXIS", "SYS-UNITS"] and cov2 == ["REQ-001"], (cov1, cov2)
        print("[ok] every hard geometric row kills >= 1 mutant; coverage without M08+M10 -> %r, without M09 -> %r"
              % (cov1, cov2))

        # T6: NE is never a kill.
        m08, m06 = by_id["M08"], by_id["M06"]
        assert m08["verdicts"]["REQ-001"] == "not_evaluable" and "REQ-001" not in m08["killed_by"], m08
        assert all(r in m06["not_evaluable"] for r in ("REQ-001", "REQ-002", "REQ-003")), m06["not_evaluable"]
        print("[ok] NE is never a kill: M08 REQ-001 not_evaluable (MEAS-NOTCIRCLE -> NE-MISSING), M06 diameters NE")

        # T7: the process is clean - the writer is the original and no template or fixture module was loaded.
        wf = export.__dict__["write_step"]
        assert wf.__module__ == "export" and wf.__name__ == "write_step", (wf.__module__, wf.__name__)
        tdir = os.path.dirname(export.TEMPLATE)
        fdir = os.path.join(HERE, "fixtures", "mutate")
        bad = sorted(n for n, mod in sys.modules.items() if getattr(mod, "__file__", None)
                     and os.path.normcase(os.path.dirname(os.path.abspath(mod.__file__)))
                     in (os.path.normcase(tdir), os.path.normcase(fdir)))
        assert not bad, bad
        print("[ok] isolation: write_step restored; this process never imported template.py or mutant.py")

        # T8: the matrix is canonical and path-free, gc2 holds, 12 distinct eval keys, the usage exits 2.
        assert matrix["gc2"] is True, matrix["gc2"]
        cj = common.canonical_json(matrix)
        for s in (td, os.path.abspath(td), td.replace(os.sep, "/"), td.replace(os.sep, os.sep + os.sep), "C:"):
            assert s not in cj, s
        keys = [matrix["nominal"]["eval_key"], matrix["control"]["eval_key"]] \
            + [m["eval_key"] for m in matrix["mutants"]]
        assert len(keys) == 12 and all(len(k) == 64
                                       and all(ch in "0123456789abcdef" for ch in k) for k in keys)
        assert keys[0] == keys[1], "the control's eval key differs from the nominal's"
        mkeys = set(keys[2:])
        assert len(mkeys) == 10 and keys[0] not in mkeys, sorted(mkeys)
        assert main([]) == 2 and main(["run"]) == 2
        print("[ok] mutation.json canonical and path-free, gc2 true, 12 eval keys, 11 contents"
              " (control == nominal); usage exits 2")

    print("selftest wall %.1f s" % (time.monotonic() - t0,))
    print("SELFTEST PASS")
    return 0


def main(argv):
    """--selftest, or run OUT_DIR; anything else is the usage on stderr with exit 2."""
    if argv == ["--selftest"]:
        try:
            return selftest()
        except Exception:
            traceback.print_exc()
            return 1
    if len(argv) == 2 and argv[0] == "run":
        matrix = run_matrix(os.path.abspath(argv[1]))
        for m in matrix["mutants"]:
            print("%s %s killed_by %s" % (m["id"], m["name"], " ".join(m["killed_by"])))
        print("GC-2 %s: %d/%d killed, uncovered %s"
              % ("PASS" if matrix["gc2"] else "FAIL", matrix["killed"], matrix["total"], matrix["uncovered"]))
        return 0 if matrix["gc2"] else 1
    sys.stderr.write(USAGE + chr(10))
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
