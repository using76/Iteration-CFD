#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""evaluate_cfd.py - the real S3-S10 evaluator "cfd" of docs/16 §E.2/§E.4 and §I CAD-21: one
design at one mesh level through build, checks, readiness, wedge mesh, case, solve, post and
judge, stopping at the first refused stage with every measurement a refused record.

B1  eval_parts is exactly reqs.EVAL_KEY_PARTS; the level lives in mesh_recipe_version, so an L1
    and an L2 evaluation of the same params never share an eval key.
B2  cfd_u per docs/16 §E.4 read from the CAD-20 nominal record: gci_fine when known, the repeat
    band 0.0 for every cfd row while the repeat's fields are bit-identical.
B3  post.py metrics become cad-measure/1 records by primitive name, units from post.METRICS.
B4  a design that fails the mesh gate stops at stage "mesh", its cfd rows refused EVAL-MESH.
B5  a meshable design runs build -> checks -> readiness -> mesh -> case -> solve -> post with
    injected fakes; the cache entry's top-level files are exactly TOP_FILES.
B6  an unsteady solve refuses the cfd rows with the solve's reason id; a failing hard geometric
    row stops at "checks" and no mesh is built.
B7  loop.evaluate_one with evaluator "cfd" writes a cache entry loop.validate_cache accepts; a
    second call is a cache hit and does zero work.
B8  the stub path's bytes are unchanged: loop.py --selftest still prints its 12 [ok].

Usage:
  python evaluate_cfd.py --selftest
"""
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "cases", "poiseuille"))
import case_writer  # noqa: E402
import common  # noqa: E402
import export  # noqa: E402
import optimise_cad  # noqa: E402
import poiseuille  # noqa: E402
import post  # noqa: E402
import readiness  # noqa: E402
import reqs  # noqa: E402
import schema  # noqa: E402
import solve  # noqa: E402
import verify  # noqa: E402
import wedge_mesh  # noqa: E402

VERSION = "cfd/1"
LEVELS = ("L0", "L1", "L2")
ITERS = {"L0": 1000, "L1": 1200, "L2": 6000}      # = nozzle_nominal.ITERS, fixed before any run (CAD-20)
H_READY_M = readiness.H_SELFTEST_M                # m: the readiness h, about the nominal's L1 core cell
NOMINAL_RECORD = os.path.join(HERE, "cases", "nozzle_nominal", "nominal_record.json")
BIN_GPU = os.path.join(HERE, "bin_gpu.json")
BIN_CPU = os.path.join(HERE, "bin.json")
STAGES = ("build", "checks", "readiness", "mesh", "case", "solve", "post", "judge")
REASON_IDS = ("EVAL-LEVEL", "EVAL-BUILD", "EVAL-CHECKS", "EVAL-MESH", "EVAL-POST")
TOP_FILES = ("cfd_u.json", "geom.json", "measurements.json", "params.json", "stage.json")
SUBDIRS = ("geom", "mesh", "case", "run")
GEOM_REPRS = ("brep", "stl")
_UNITS = dict((m[0], m[1]) for m in post.METRICS)
_DECL_UNITS = None        # lazy primitive -> unit from the template declaration, for refused records


def level_index(level) -> int:
    """0, 1 or 2 for "L0", "L1", "L2"; anything else raises EVAL-LEVEL."""
    if level not in LEVELS:
        raise ValueError("EVAL-LEVEL: level %r is not one of %s" % (level, ", ".join(LEVELS)))
    return LEVELS.index(level)


def eval_parts(params, level, template_sha, declaration_sha, lock_sha, gates_lock) -> dict:
    """Exactly reqs.EVAL_KEY_PARTS (docs/16 §D): the level lives in mesh_recipe_version, so an L1
    and an L2 evaluation of the same params never share a key. env carries this module's own
    inputs: the evaluator version, the nominal record and the CPU bin file the key depends on,
    the solve and post versions and the level's fixed iteration budget."""
    level_index(level)
    return {"template_sha": template_sha, "declaration_sha": declaration_sha, "params": params,
            "requirements_lock": lock_sha, "gates_lock": gates_lock,
            "env": dict(common.env_fingerprint(), evaluator=VERSION,
                        cfd_u_source_sha256=common.sha256_file(NOMINAL_RECORD),
                        bin_cpu_sha256=common.sha256_file(BIN_CPU),
                        solve_version=solve.VERSION, post_version=post.VERSION,
                        iters=ITERS[level]),
            "mesh_recipe_version": wedge_mesh.RECIPE["version"] + "@" + level,
            "case_writer_version": case_writer.CASE_VERSION + "/" + case_writer.CASE_WRITER_VERSION,
            "bin_sha": common.read_json(BIN_GPU)["binaries"]["ofgpu-lowmach"]["sha256"]}


def cfd_u_of(checks_doc, record) -> dict:
    """B2: one {"gci_fine", "repeat_band"} entry per repr-cfd check, keyed by req_id, in check
    order, from the CAD-20 nominal record. gci_fine when the record knows it, else None. While
    the repeat's fields are bit-identical every quantity computed from them repeats exactly, so
    the band is 0.0 for EVERY cfd check; otherwise the record's per-quantity band, else None."""
    rec = record if isinstance(record, dict) else {}
    rep = rec.get("g_repeat") if isinstance(rec.get("g_repeat"), dict) else {}
    ident = rep.get("fields_bit_identical") is True
    band = rep.get("band") if isinstance(rep.get("band"), dict) else {}
    cfd_u = rec.get("cfd_u") if isinstance(rec.get("cfd_u"), dict) else {}
    out = {}
    for c in checks_doc["checks"]:
        if c["repr"] != "cfd":
            continue
        prim = c["primitive"]
        row = cfd_u.get(prim) if isinstance(cfd_u.get(prim), dict) else {}
        g = row.get("gci_fine")
        if ident:
            b = 0.0
        else:
            b = band.get(prim)
        out[c["req_id"]] = {"gci_fine": g, "repeat_band": b}
    return out


def _decl_units():
    """primitive -> unit from the template declaration (lazy; the refused records' unit)."""
    global _DECL_UNITS
    if _DECL_UNITS is None:
        decl = common.read_json(os.path.join(HERE, "templates", "nozzle_contraction", "template.json"))
        _DECL_UNITS = {}
        for row in decl["catalogue"]:
            _DECL_UNITS.setdefault(row["primitive"], row["unit"])
    return _DECL_UNITS


def _record(check, value, status, reason_id, detail, unit=None) -> dict:
    """One cad-measure/1 record of this module: feature and where from the check, unit from
    post.METRICS for a cfd primitive else from the declaration."""
    prim = check["primitive"]
    rec = {"schema": "cad-measure/1", "primitive": prim, "feature": check["args"]["feature"],
           "where": list(check["args"]["where"]), "value": value,
           "unit": unit if unit is not None else _decl_units().get(prim, "1"), "u_meas": None,
           "method": "post/" + post.VERSION if check["repr"] == "cfd" else VERSION,
           "status": status, "reason_id": reason_id, "detail": detail[:300]}
    errs = schema.errors(rec, "cad-measure/1")
    if errs:
        raise RuntimeError("evaluate_cfd: record fails cad-measure/1: %r" % (errs[:1],))
    return rec


def cfd_records(checks_doc, post_doc) -> dict:
    """B3: one cad-measure/1 record per repr-cfd check from the cad-post/1 doc, keyed by req_id,
    units from post.METRICS, method "post/" + post.VERSION. A refused post refuses every record
    EVAL-POST with post's reason named; an absent or None metric is an error record EVAL-POST."""
    ok = isinstance(post_doc, dict) and post_doc.get("status") == "ok"
    metrics = post_doc.get("metrics") if isinstance(post_doc, dict) else None
    preason = None if not isinstance(post_doc, dict) else post_doc.get("reason_id")
    out = {}
    for c in checks_doc["checks"]:
        if c["repr"] != "cfd":
            continue
        prim = c["primitive"]
        unit = _UNITS.get(prim, "1")
        if not ok:
            out[c["req_id"]] = _record(c, None, "refused", "EVAL-POST",
                                       "post refused with %s" % (preason,), unit=unit)
        elif not isinstance(metrics, dict) or metrics.get(prim) is None \
                or metrics[prim].get("value") is None:
            out[c["req_id"]] = _record(c, None, "error", "EVAL-POST",
                                       "metric %s is absent or None" % (prim,), unit=unit)
        else:
            out[c["req_id"]] = _record(c, float(metrics[prim]["value"]), "ok", None, "", unit=unit)
    return out


def refused_records(checks_doc, reprs, reason_id, detail) -> dict:
    """The status-refused record (value None) of every check whose repr is in REPRS, for a stage
    that stops the evaluation."""
    out = {}
    for c in checks_doc["checks"]:
        if c["repr"] in reprs:
            out[c["req_id"]] = _record(c, None, "refused", reason_id, detail)
    return out


def geometry_records(checks_doc, geom_dir) -> dict:
    """The probes.json record of every check with repr brep or stl (optimise_cad._probes_row);
    a check with no probes row gets no record."""
    out = {}
    for c in checks_doc["checks"]:
        if c["repr"] not in GEOM_REPRS:
            continue
        try:
            out[c["req_id"]] = optimise_cad._probes_row(geom_dir, c)
        except RuntimeError:
            continue
    return out


def prefilter(params, checks_doc) -> dict:
    """optimise_cad.cad_prefilter in a fresh temp dir removed in a finally, so run and replay
    never write into the study."""
    work = tempfile.mkdtemp(prefix="cad21-pf-")
    try:
        return optimise_cad.cad_prefilter(params, checks_doc, work, H_READY_M)
    finally:
        import shutil
        shutil.rmtree(work, ignore_errors=True)


def _scrub(text, out_dir) -> str:
    """Replace every spelling of out_dir by "<eval_dir>": stage.json never holds an absolute path."""
    for form in (out_dir, os.path.abspath(out_dir), os.path.abspath(out_dir).replace(os.sep, "/")):
        if form:
            text = str(text).replace(form, "<eval_dir>")
    return text


def _solve_row(s) -> dict:
    """The stage.json solve row (the nozzle_nominal _solve_row shape) of one solve doc."""
    if s is None:
        return None
    res = s.get("result") or {}
    return {"class": s.get("class"), "reason_id": s.get("reason_id"), "failed": res.get("failed"),
            "n_iter_lines": len(res.get("iterations") or []),
            "log_sha256": (s.get("log") or {}).get("sha256"),
            "binary_sha256": (s.get("binary") or {}).get("sha256")}


def evaluate(params, level, checks_doc, out_dir, req_dir, solve_fn=None, post_fn=None,
             snapshot_fn=None) -> dict:
    """One design at one level through the stages of STAGES in order, into out_dir (exists and is
    empty); stops at the first refused stage with the not-yet-measured reprs refused by id, writes
    the TOP_FILES and returns {"stage_reached", "solve_class"}. Never raises on a stage refusal;
    an unexpected exception propagates. No absolute path lands in stage.json."""
    level_index(level)                            # validates; ITERS keys are "L0".."L2"
    iters = ITERS[level]
    out_dir = os.path.abspath(out_dir)
    geom_dir = os.path.join(out_dir, "geom")
    mesh_dir = os.path.join(out_dir, "mesh")
    case_dir = os.path.join(out_dir, "case")
    run_dir = os.path.join(out_dir, "run")
    stage = {"version": VERSION, "level": level, "iters": iters, "stage_reached": None,
             "solve_class": "not_run", "reason_id": None, "message": "", "cells": None,
             "mesh": None, "solve": None, "wall_s": None, "gpu": None}
    records = {}

    def refuse(stage_name, reprs, rid, message):
        stage["stage_reached"] = stage_name
        stage["reason_id"] = rid
        stage["message"] = _scrub(message, out_dir)
        records.update(refused_records(checks_doc, reprs, rid, stage["message"]))

    def finish() -> dict:
        common.write_json(os.path.join(out_dir, "params.json"), params)
        common.write_json(os.path.join(out_dir, "measurements.json"), records)
        nominal = common.read_json(NOMINAL_RECORD)
        common.write_json(os.path.join(out_dir, "cfd_u.json"), cfd_u_of(checks_doc, nominal))
        common.write_json(os.path.join(out_dir, "stage.json"), stage)
        files = dict((name, common.sha256_file(os.path.join(out_dir, name)))
                     for name in TOP_FILES if name != "geom.json")
        common.write_json(os.path.join(out_dir, "geom.json"),
                          {"version": VERSION, "params_sha": optimise_cad.params_sha(params),
                           "files": files})
        return {"stage_reached": stage["stage_reached"], "solve_class": stage["solve_class"]}

    # 1. build
    pipe = export.run_pipeline(export.TEMPLATE, params, geom_dir)
    if pipe["status"] != "ok":
        refuse("build", reqs.REPRS, "EVAL-BUILD",
               "the build stage refused with %s: %s" % (pipe["rule"], pipe["message"]))
        return finish()
    # 2. checks: the hard brep/stl rows decide; the objective and soft rows are only recorded
    records.update(geometry_records(checks_doc, geom_dir))
    first_bad = None
    for c in checks_doc["checks"]:
        if c["hardness"] != "hard" or c["repr"] not in GEOM_REPRS:
            continue
        v = verify.judge(c, records.get(c["req_id"]))
        if v["verdict"] != "pass" and first_bad is None:
            first_bad = (c["req_id"], v["verdict"], v["reason_id"])
    if first_bad is not None:
        refuse("checks", ("cfd",), "EVAL-CHECKS",
               "hard row %s is %s (%s)" % first_bad)
        return finish()
    # 3. readiness
    ready = readiness.check(geom_dir, H_READY_M)
    if ready["status"] != "ready":
        refuse("readiness", ("cfd",), ready["rule"],
               "readiness refused with %s: %s" % (ready["rule"], ready["detail"]))
        return finish()
    # 4. mesh
    mesh = wedge_mesh.run(geom_dir, mesh_dir, levels=(level_index(level),))
    if mesh["status"] != "ok":
        refuse("mesh", ("cfd",), mesh["rule"],
               "wedge_mesh refused with %s: %s" % (mesh["rule"], mesh["message"]))
        return finish()
    lv0 = mesh["report"]["levels"][0]
    stage["cells"] = lv0["cells"]
    stage["mesh"] = {"gc6": lv0["gc6"], "tau_min": lv0["check"]["tau_min"],
                     "non_orth_max_deg": lv0["check"]["non_orth_max_deg"]}
    if mesh["report"]["gc6_pass"] is not True:
        bad = sorted(k for k, v in lv0["gc6"].items() if v is not True and k != "pass")
        refuse("mesh", ("cfd",), "EVAL-MESH",
               "gc6 false on %s; tau_min %r, non_orth_max_deg %r"
               % (", ".join(bad), stage["mesh"]["tau_min"], stage["mesh"]["non_orth_max_deg"]))
        return finish()
    # 5. case
    case = case_writer.write_case(mesh_dir, level_index(level), geom_dir, req_dir, case_dir)
    if case["status"] != "ok":
        refuse("case", ("cfd",), case["rule"],
               "case_writer refused with %s: %s" % (case["rule"], case["message"]))
        return finish()
    # 6. solve
    if snapshot_fn is None:
        snapshot_fn = poiseuille.gpu_snapshot
    before = snapshot_fn()
    t0 = time.monotonic()
    if solve_fn is None:
        sdoc = solve.launch(case_dir, geom_dir, run_dir, iters, bin_json=BIN_GPU, visible=True)
    else:
        sdoc = solve_fn(case_dir, geom_dir, run_dir, iters)
    stage["wall_s"] = time.monotonic() - t0
    after = snapshot_fn()
    stage["gpu"] = {"before": before, "after": after,
                    "shared": poiseuille.shared_flag(before, after), "note": poiseuille.GPU_NOTE}
    stage["solve"] = _solve_row(sdoc)
    stage["solve_class"] = sdoc.get("class")
    if sdoc.get("class") != "steady":
        rid = sdoc.get("reason_id") or "SOLVE-" + str(sdoc.get("class")).upper()
        refuse("solve", ("cfd",), rid,
               "the solve class is %s: %s" % (sdoc.get("class"), sdoc.get("message")))
        return finish()
    # 7. post
    if post_fn is None:
        pdoc = post.post(case_dir, str(iters), geom_dir)
    else:
        pdoc = post_fn(case_dir, str(iters), geom_dir)
    records.update(cfd_records(checks_doc, pdoc))
    # 8. judge
    stage["stage_reached"] = "judge"
    return finish()


# ---- the CAD-21 selftest: E1..E7, one [ok] line each; fakes only, the GPU is never launched ----

def _no_solve(case_dir, geom_dir, run_dir, iters):
    raise AssertionError("the fake solve must never be called")


def _steady_doc():
    return {"version": "fake", "class": "steady", "reason_id": None, "message": "", "result": {},
            "log": {"sha256": "0" * 64}, "binary": {"sha256": "1" * 64}}


def _post_doc():
    metrics = {"Q_in": 0.00707, "Q_out": 0.00707, "mass_imbalance": 1e-9, "dp": 118.0, "Cd": 0.97,
               "dp_loss": 121.0, "p0_loss_axis": 0.004, "exit_nonuniformity": 0.005,
               "theta_exit": 8.2e-5, "dstar_exit": 1.6e-5, "H_exit": 1.32, "u_axis_ratio_exit": 1.81,
               "mach_max": 0.07, "momentum_closure": 0.003, "reversal_fraction": 0.0,
               "separation_free": 1.0}
    return {"version": post.VERSION, "status": "ok", "reason_id": None, "message": "",
            "metrics": dict((k, {"value": v, "unit": _UNITS[k], "repr": "cfd", "where": [],
                                 "reason_id": None, "definition": "planted"}) for k, v in metrics.items())}


def _gates_lock() -> str:
    """The sha256 first token of gates.lock (the sha256sum line gate.load_gates binds)."""
    with open(os.path.join(HERE, "gates.lock"), "r", encoding="utf-8") as f:
        return f.read().split()[0]


def _g4_inputs():
    doc = reqs.read_locked(os.path.join(HERE, "studies", "g4_nozzle"))
    decl, tsha, dsha = reqs.load_template(reqs.NOZZLE_DIR)
    checks = reqs.compile_checks(doc, decl, dsha)
    return doc, decl, tsha, dsha, checks


def _start_params():
    st = common.read_json(os.path.join(HERE, "studies", "g4_nozzle", "start.json"))
    return st["start"]["params"]


def _parts(params, level, tsha, dsha, lock, gl):
    return eval_parts(params, level, tsha, dsha, lock, gl)


def selftest():
    import copy
    import shutil
    import tempfile
    import time
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="cad21-ecfd-") as td:
        doc, _decl, tsha, dsha, checks = _g4_inputs()
        gl = _gates_lock()
        lock = doc["lock_sha"]
        params = _start_params()

        # (E1) the eval key: exactly EVAL_KEY_PARTS, the level inside mesh_recipe_version
        parts = _parts(params, "L1", tsha, dsha, lock, gl)
        assert tuple(sorted(parts.keys())) == tuple(sorted(reqs.EVAL_KEY_PARTS)), sorted(parts)
        assert parts["mesh_recipe_version"] == "cad-wedge/1@L1", parts["mesh_recipe_version"]
        assert parts["case_writer_version"] == "cad-case/1/1", parts["case_writer_version"]
        k1 = reqs.eval_key(parts)
        k2 = reqs.eval_key(_parts(params, "L2", tsha, dsha, lock, gl))
        assert k1 != k2, "an L1 and an L2 evaluation of the same params share a key"
        bg = common.read_json(BIN_GPU)
        assert parts["bin_sha"] == bg["binaries"]["ofgpu-lowmach"]["sha256"], "bin_sha is not bin_gpu's"
        assert parts["bin_sha"] == "50471caaa54125e0c2eee4fb34eebdfc2da44727d2b818a2b96bb4234c02103c"
        planted = copy.deepcopy(parts)
        planted["env"]["cfd_u_source_sha256"] = "0" * 64
        assert reqs.eval_key(planted) != k1, "a changed cfd_u_source_sha256 did not move the key"
        try:
            level_index("L3")
            raise AssertionError("level_index accepted L3")
        except ValueError as e:
            assert str(e).startswith("EVAL-LEVEL"), e
        print("[ok] eval parts: exactly reqs.EVAL_KEY_PARTS, cad-wedge/1@L1, L1 != L2 key,"
              " bin_sha = bin_gpu's 50471caa, a changed cfd_u_source moves the key, L3 raises EVAL-LEVEL")

        # (E2) cfd_u from the real nominal record: bit-identical repeat fields give band 0.0 everywhere
        nominal = common.read_json(NOMINAL_RECORD)
        u = cfd_u_of(checks, nominal)
        cfd_ids = [c["req_id"] for c in checks["checks"] if c["repr"] == "cfd"]
        assert cfd_ids == ["REQ-003", "REQ-004", "REQ-005", "SYS-MACH"], cfd_ids
        assert all(u[i] == {"gci_fine": None, "repeat_band": 0.0} for i in cfd_ids), u
        planted = dict(nominal, g_repeat=dict(nominal["g_repeat"], fields_bit_identical=False,
                                              band={"Cd": 0.002}))
        u2 = cfd_u_of(checks, planted)
        assert u2["REQ-005"]["repeat_band"] == 0.002, u2["REQ-005"]
        assert u2["REQ-004"]["repeat_band"] is None, u2["REQ-004"]
        assert u2["REQ-005"]["gci_fine"] is None
        print("[ok] cfd_u: the nominal record gives gci_fine None and band 0.0 for every cfd check;"
              " planted band Cd 0.002 hits REQ-005 only, REQ-004 stays None")

        # (E3) cfd_records: a planted post doc, a refused post, a None metric
        pdoc = _post_doc()
        recs = cfd_records(checks, pdoc)
        assert recs["REQ-005"]["value"] == 0.97 and recs["REQ-004"]["value"] == 0.005
        assert recs["REQ-003"]["value"] == 1.0 and recs["SYS-MACH"]["value"] == 0.07
        assert all(recs[i]["unit"] == "1" and recs[i]["status"] == "ok" for i in cfd_ids)
        assert all(schema.errors(r, "cad-measure/1") == [] for r in recs.values())
        rrecs = cfd_records(checks, {"status": "refused", "reason_id": "POST-BIND", "metrics": {}})
        assert all(rrecs[i]["status"] == "refused" and rrecs[i]["reason_id"] == "EVAL-POST"
                   and "POST-BIND" in rrecs[i]["detail"] for i in cfd_ids)
        nul = _post_doc()
        nul["metrics"]["Cd"]["value"] = None
        erecs = cfd_records(checks, nul)
        assert erecs["REQ-005"]["status"] == "error"
        v = verify.judge([c for c in checks["checks"] if c["req_id"] == "REQ-005"][0], erecs["REQ-005"])
        assert v["reason_id"] == "NE-ERROR" and v["verdict"] == "not_evaluable", v
        print("[ok] cfd records: planted post values exact with unit 1 and cad-measure/1 valid;"
              " a refused post refuses all EVAL-POST naming POST-BIND; a None Cd is NE-ERROR")

        # (E4) the REAL start (poly7, L/D 0.5, t_wall 0.004) at L1: the mesh gate refuses EVAL-MESH
        out4 = os.path.join(td, "e4")
        os.makedirs(out4)
        stage = evaluate(params, "L1", checks, out4, os.path.join(HERE, "studies", "g4_nozzle"),
                         solve_fn=_no_solve)
        assert stage == {"stage_reached": "mesh", "solve_class": "not_run"}, stage
        st = common.read_json(os.path.join(out4, "stage.json"))
        assert st["reason_id"] == "EVAL-MESH" and "tau" in st["message"], st["message"]
        assert 0.0 < st["mesh"]["tau_min"] < 0.05, st["mesh"]["tau_min"]
        assert sorted(n for n in os.listdir(out4) if os.path.isfile(os.path.join(out4, n))) \
            == sorted(TOP_FILES), sorted(os.listdir(out4))
        ek = reqs.eval_key(_parts(params, "L1", tsha, dsha, lock, gl))
        ms = common.read_json(os.path.join(out4, "measurements.json"))
        cu = common.read_json(os.path.join(out4, "cfd_u.json"))
        verdict = verify.evaluate(checks, doc, ms, ek, cu)
        rows = dict((r["req_id"], r) for r in verdict["verdicts"])
        for rid in ("REQ-001", "REQ-002", "REQ-006"):
            assert rows[rid]["verdict"] == "pass", (rid, rows[rid])
        for rid in ("REQ-003", "REQ-004", "REQ-005", "SYS-MACH"):
            assert rows[rid]["verdict"] == "not_evaluable" and rows[rid]["reason_id"] == "NE-MISSING", \
                (rid, rows[rid])
        assert verdict["design_verdict"] == "not_evaluable", verdict["design_verdict"]
        print("[ok] mesh gate: the real start at L1 stops at stage mesh with EVAL-MESH (tau_min %.6f),"
              " the entry holds exactly the five top files, REQ-001/002/006 pass and every cfd row"
              " is NE-MISSING (design not_evaluable)" % (st["mesh"]["tau_min"],))

        # (E5) the meshable design (poly7, L/D 0.75, t_wall 0.004) at L1 with fakes reaches judge
        p5 = dict(params, L_over_Di=0.75)
        out5 = os.path.join(td, "e5")
        os.makedirs(out5)
        seen = {}
        def solve5(case_dir, geom_dir, run_dir, iters):
            seen["solve"] = (case_dir, geom_dir, run_dir, iters)
            return _steady_doc()
        def post5(case_dir, time_name, geom_dir):
            seen["post"] = (time_name,)
            return _post_doc()
        def snap5():
            return {"status": "unavailable", "gpu": None, "compute_apps": [], "detail": "fake"}
        stage5 = evaluate(p5, "L1", checks, out5, os.path.join(HERE, "studies", "g4_nozzle"),
                          solve_fn=solve5, post_fn=post5, snapshot_fn=snap5)
        assert stage5 == {"stage_reached": "judge", "solve_class": "steady"}, stage5
        assert os.path.isfile(os.path.join(out5, "case", "case.json"))
        assert seen["solve"][3] == ITERS["L1"] and seen["post"][0] == str(ITERS["L1"])
        st5 = common.read_json(os.path.join(out5, "stage.json"))
        text = open(os.path.join(out5, "stage.json"), "r", encoding="utf-8").read()
        assert td not in text and "C:" not in text, "stage.json holds an absolute path"
        assert st5["gpu"]["shared"] is None and st5["gpu"]["note"] == poiseuille.GPU_NOTE
        assert st5["solve"]["class"] == "steady" and st5["wall_s"] >= 0.0
        ek5 = reqs.eval_key(_parts(p5, "L1", tsha, dsha, lock, gl))
        ms5 = common.read_json(os.path.join(out5, "measurements.json"))
        cu5 = common.read_json(os.path.join(out5, "cfd_u.json"))
        v5 = verify.evaluate(checks, doc, ms5, ek5, cu5)
        assert v5["design_verdict"] == "feasible", v5["design_verdict"]
        print("[ok] full path: the real L/D 0.75 design at L1 (gc6 tau %.6f) reaches judge through"
              " the injected fakes, case.json exists, and the planted post makes the design feasible"
              % (st5["mesh"]["tau_min"],))

        # (E6) an unsteady solve refuses at stage solve; a failing wall stops at checks, no mesh
        out6 = os.path.join(td, "e6")
        os.makedirs(out6)
        def solve6(case_dir, geom_dir, run_dir, iters):
            return dict(_steady_doc(), **{"class": "unsteady", "reason_id": "SOLVE-UNSTEADY",
                                          "message": "planted"})
        stage6 = evaluate(p5, "L1", checks, out6, os.path.join(HERE, "studies", "g4_nozzle"),
                          solve_fn=solve6, post_fn=post5, snapshot_fn=snap5)
        assert stage6 == {"stage_reached": "solve", "solve_class": "unsteady"}, stage6
        st6 = common.read_json(os.path.join(out6, "stage.json"))
        assert st6["reason_id"] == "SOLVE-UNSTEADY" and st6["solve"]["reason_id"] == "SOLVE-UNSTEADY"
        ms6 = common.read_json(os.path.join(out6, "measurements.json"))
        assert all(ms6[i]["status"] == "refused" and ms6[i]["reason_id"] == "SOLVE-UNSTEADY"
                   for i in ("REQ-003", "REQ-004", "REQ-005", "SYS-MACH"))
        out6b = os.path.join(td, "e6b")
        os.makedirs(out6b)
        stage6b = evaluate(dict(params, t_wall=0.002), "L1", checks, out6b,
                           os.path.join(HERE, "studies", "g4_nozzle"), solve_fn=_no_solve)
        assert stage6b == {"stage_reached": "checks", "solve_class": "not_run"}, stage6b
        assert not os.path.isdir(os.path.join(out6b, "mesh"))
        st6b = common.read_json(os.path.join(out6b, "stage.json"))
        assert st6b["reason_id"] == "EVAL-CHECKS" and "REQ-006" in st6b["message"], st6b["message"]
        ek6 = reqs.eval_key(_parts(dict(params, t_wall=0.002), "L1", tsha, dsha, lock, gl))
        v6 = verify.evaluate(checks, doc, common.read_json(os.path.join(out6b, "measurements.json")),
                             ek6, common.read_json(os.path.join(out6b, "cfd_u.json")))
        assert v6["design_verdict"] == "infeasible", v6["design_verdict"]
        print("[ok] refusals: an unsteady fake solve stops at stage solve with SOLVE-UNSTEADY and the"
              " cfd rows NE-MISSING; t_wall 0.002 stops at checks EVAL-CHECKS naming REQ-006, builds"
              " no mesh and is infeasible")

        # (E7) a temp g4 study: loop.evaluate_one with evaluator "cfd" caches; the second call is
        # a cache hit and does zero work. Test-only module lookups - this module's import graph
        # never reaches loop (loop imports this module).
        sys.path.insert(0, os.path.join(HERE, "studies", "g4_nozzle"))
        import g4  # noqa: E402
        loop_mod = sys.modules.get("loop")
        assert loop_mod is not None, "g4 did not load loop"
        reg = os.path.join(td, "studies.jsonl")
        study_dir = os.path.join(td, "study")
        g4.init(study_dir, reg)
        calls0 = loop_mod.EVALUATOR_CALLS[0]
        verdict, ev, ek7 = loop_mod.evaluate_one(study_dir, params, "L1", checks, doc, tsha, dsha,
                                                 gl, True, evaluator="cfd",
                                                 eval_kwargs={"solve_fn": _no_solve})
        assert loop_mod.EVALUATOR_CALLS[0] - calls0 == 1, "evaluate_one did not count one call"
        assert ev["evaluator"] == "cfd" and ev["stage_reached"] == "mesh"
        assert verdict["design_verdict"] == "not_evaluable"
        again = loop_mod.validate_cache(study_dir, ek7, checks, doc)
        assert common.canonical_json(again) == common.canonical_json(verdict)
        n0 = loop_mod.EVALUATOR_CALLS[0]
        verdict2, ev2, ek7b = loop_mod.evaluate_one(study_dir, params, "L1", checks, doc, tsha,
                                                    dsha, gl, True, evaluator="cfd",
                                                    eval_kwargs={"solve_fn": _no_solve})
        assert ek7b == ek7 and loop_mod.EVALUATOR_CALLS[0] == n0, "the cache hit did work"
        print("[ok] loop cache: g4.init + evaluate_one(cfd) writes a cache/<ek> entry"
              " validate_cache accepts, counts exactly one call, and the second call is a hit"
              " doing zero work")

    shutil.rmtree(td, ignore_errors=True)
    print("selftest wall %.1f s" % (time.time() - t0))
    print("SELFTEST PASS")
    return 0


def main(argv) -> int:
    if argv == ["--selftest"]:
        try:
            return selftest()
        except Exception:
            import traceback
            traceback.print_exc()
            return 1
    print("usage: python evaluate_cfd.py --selftest", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
