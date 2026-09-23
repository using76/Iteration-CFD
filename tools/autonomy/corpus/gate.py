#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""gate - docs/15 §F G-CORPUS for the generated corpus families A, B, D, E, F.

Every written STL is re-read from disk by two independent oracles:
tools/geom/stl_repair.py (before.open_edges 0, non_manifold 0, nothing
welded, nothing reoriented, n_bodies closed components, positive re-read
volume) and ofgpu-automesher -dryRun (exit 0 and the bit-exact weld's point
count).  Checks, each by its name: closed, dryrun, no_negzero, sha, regen,
row, volume, min_edge - and for families D, E and F the features check,
which re-computes tools/autonomy/features.py's fingerprint on the written
file and refuses any disagreement with the generator's expected_features
(commensurability and lattice spacing, the planar fraction, the outer gap
and the plate thickness, each within its own tolerance).  Regeneration
runs in a child process under a different PYTHONHASHSEED; rows re-validate
as ManifestRow with split added; volumes compare against the generator's
closed form (1 %).  Negative controls in --selftest prove the gate is not
vacuous.

    python tools/autonomy/corpus/gate.py --family A --family B --n 120 --seed 1
    python tools/autonomy/corpus/gate.py --selftest
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(_HERE))), "tools", "geom"))

import stl_io  # noqa: E402
import schema  # noqa: E402
import stl_repair  # noqa: E402

FAMILIES = {"A": "gen_wing", "B": "gen_lathe", "D": "gen_bluff",
           "E": "gen_gap", "F": "gen_thin"}
REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
BINARY = os.path.join(REPO, "rust", "target", "release",
                      "ofgpu-automesher" + (".exe" if os.name == "nt" else ""))
CHECKS = ("closed", "dryrun", "no_negzero", "sha", "regen", "row", "volume",
          "min_edge")


def checks_for(gen) -> tuple:
    """A and B print today's checks; D, E and F add the features agreement."""
    return (CHECKS + ("features",) if hasattr(gen, "expected_features")
            else CHECKS)


def _load(family: str):
    return importlib.import_module(FAMILIES[family])


def repair_report(stl_path: str) -> dict:
    """stl_repair's report for a WRITTEN file, judged on the re-read bytes."""
    buf = io.StringIO()
    tmp_json = stl_path + ".repair.json"
    try:
        with contextlib.redirect_stdout(buf):
            rc = stl_repair.main([stl_path, "--json", tmp_json])
    except SystemExit as e:
        return {"_error": "stl_repair exited %s" % (e.code,)}
    if rc != 0:
        return {"_error": "stl_repair returned %d" % rc}
    with open(tmp_json, encoding="utf-8") as f:
        return json.load(f)


def dry_run(stl_path: str, case_root: str, binary: str) -> dict:
    """The automesher's -dryRun on the written file; writes only the config."""
    stem = os.path.splitext(os.path.basename(stl_path))[0]
    tris, _fmt = stl_repair.read_stl(stl_path, stem)
    pts = np.array([v for _p, t in tris for v in t], dtype=np.float64)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    diag = float(np.linalg.norm(hi - lo))
    cfg = {"input": {"surfaces": [{"path":
                                   os.path.abspath(stl_path).replace(os.sep, "/")}]},
           "domain": {"extent": [float(lo[0] - diag), float(hi[0] + diag),
                                 float(lo[1] - diag), float(hi[1] + diag),
                                 float(lo[2] - diag), float(hi[2] + diag)],
                      "base_size": diag / 4.0},
           "output": {"case_dir": os.path.join(case_root, "case_" + stem)
                      .replace(os.sep, "/"), "name": stem}}
    cfg_path = os.path.join(case_root, stem + ".dry.json")
    with open(cfg_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(cfg, f, indent=1)
        f.write("\n")
    r = subprocess.run([binary, cfg_path, "-dryRun"], capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=60)
    return {"exit": r.returncode, "stdout": r.stdout, "stderr": r.stderr}


def _surface_line(stdout: str) -> str:
    for ln in stdout.splitlines():
        if ln.startswith("surface: "):
            return ln.strip()
    return ""


def check_row(row: dict, stl_path: str, binary: str, scratch: dict) -> dict:
    """The C6 checks, each by name: "" when it passes, else the why."""
    gen = _load(row["family"])
    chks = checks_for(gen)
    res = {c: "" for c in chks}
    res["_vol_dev"] = None
    P, T = gen.build(row["params"])
    nP, nT = len(P), len(T)
    rep = repair_report(stl_path)
    if "_error" in rep:
        res["closed"] = rep["_error"]
    else:
        b, w, o, aft = rep["before"], rep["weld"], rep["orientation"], rep["after"]
        bad = []
        if b["open_edges"] != 0:
            bad.append("before.open_edges = %d" % b["open_edges"])
        if b["non_manifold_edges"] != 0:
            bad.append("before.non_manifold_edges = %d" % b["non_manifold_edges"])
        if rep["degenerate_dropped"] != 0:
            bad.append("degenerate_dropped = %d" % rep["degenerate_dropped"])
        if w["merged"] != 0:
            bad.append("weld.merged = %d" % w["merged"])
        if w["points_bit_exact"] != nP:
            bad.append("weld.points_bit_exact = %d != %d" % (w["points_bit_exact"], nP))
        if o["reoriented_triangles"] != 0:
            bad.append("orientation.reoriented_triangles = %d"
                       % o["reoriented_triangles"])
        if o["flipped_components"] != 0:
            bad.append("orientation.flipped_components = %d"
                       % o["flipped_components"])
        nb = gen.n_bodies(row["params"]) if hasattr(gen, "n_bodies") else 1
        if rep["n_components"] != nb:
            bad.append("n_components = %d != %d" % (rep["n_components"], nb))
        if aft["closed"] is not True:
            bad.append("after.closed is not True")
        if not (isinstance(aft["volume"], float) and aft["volume"] > 0.0):
            bad.append("after.volume %r is not > 0" % (aft["volume"],))
        res["closed"] = "; ".join(bad)

    stem = os.path.splitext(os.path.basename(stl_path))[0]
    if not os.path.isfile(binary):
        res["dryrun"] = "binary not found: %s" % binary
    else:
        d = dry_run(stl_path, scratch["case"], binary)
        why = []
        if d["exit"] != 0:
            first_err = ""
            for ln in (d["stderr"] or "").splitlines():
                if ln.strip():
                    first_err = ln.strip()
                    break
            why.append("exit %d: %s" % (d["exit"], first_err[:160]))
        if "dry run - the surface stages ran" not in d["stdout"]:
            why.append("no dry-run line in stdout")
        want = "surface: %d triangle(s), %d point(s)" % (nT, nP)
        got = _surface_line(d["stdout"])
        if got != want:
            why.append("surface line %r != %r" % (got, want))
        case_dir = os.path.join(scratch["case"], "case_" + stem)
        if os.path.exists(case_dir):
            why.append("case dir left behind: %s" % case_dir)
        res["dryrun"] = "; ".join(why)

    with open(stl_path, "rb") as f:
        data = f.read()
    if b"-0." in data:
        res["no_negzero"] = "the file contains -0."

    why = []
    if hashlib.sha256(data).hexdigest() != row["stl_sha256"]:
        why.append("file sha256 != row stl_sha256")
    p2 = gen.write_row(json.loads(json.dumps(row)), scratch["rw"])
    with open(p2, "rb") as f:
        if f.read() != data:
            why.append("row rebuilt from params alone gives other bytes")
    res["sha"] = "; ".join(why)

    errs = []
    for s in ("tuning", "test"):
        errs += schema.errors(dict(row, split=s), "ManifestRow")
    if errs:
        res["row"] = errs[0]

    if isinstance(rep.get("after", {}).get("volume"), float):
        vc = gen.closed_form_volume(row["params"])
        dev = abs(rep["after"]["volume"] / vc - 1.0)
        res["_vol_dev"] = dev
        if dev > 0.01:
            res["volume"] = "|V_file/V_closed - 1| = %.4f" % dev
    else:
        res["volume"] = "no re-read volume to compare"

    if "_error" not in rep:
        me = stl_io.min_edge_length(P, T)
        tol = rep["weld"]["tol_abs"]
        if not me >= 10.0 * tol:
            res["min_edge"] = "min edge %.6g < 10 x weld tol %.6g" % (me, tol)
    else:
        res["min_edge"] = rep["_error"]
    if "features" in chks:
        res["features"], res["_feat"] = features_check(gen, row, stl_path)
    res["_nT"] = nT
    return res


def features_check(gen, row: dict, stl_path: str) -> tuple:
    """features.py's fingerprint must agree with expected_features."""
    import features

    info = {"planar_frac": None, "commensurate": None,
            "gap_dev": None, "thick_dev": None, "thick_rep_dev": None}
    try:
        fp, _diag = features.fingerprint(stl_path, row["geometry_id"])
    except ValueError as e:
        return "features refused: %s" % e, info
    exp = gen.expected_features(row["params"])
    why = []
    if (fp["commensurate"] != exp["commensurate"]
            or row["commensurate"] != exp["commensurate"]):
        why.append("commensurate %s != expected %s"
                   % (fp["commensurate"], exp["commensurate"]))
    el = exp.get("lattice_base_size_m")
    fl_ = fp["lattice_base_size_m"]
    if el is None:
        if fl_ is not None:
            why.append("lattice_base_size_m %r != %r" % (fl_, el))
    elif fl_ is None or abs(fl_ / el - 1.0) > 1e-8:
        why.append("lattice_base_size_m %r != %r" % (fl_, el))
    po = exp.get("planar_one")
    if po is True and fp["planar_frac"] != 1.0:
        why.append("planar_frac %r != 1.0" % (fp["planar_frac"],))
    if po is False and fp["planar_frac"] >= 1.0:
        why.append("planar_frac %r is >= 1.0" % (fp["planar_frac"],))
    info["planar_frac"] = fp["planar_frac"]
    info["commensurate"] = bool(fp["commensurate"])

    def dev(key, limit, tag):
        want = exp.get(key)
        if want is None:
            return
        got = fp.get(key)
        if got is None:
            why.append("%s %r vs %r (none measured)" % (key, got, want))
            info[tag] = None
            return
        d = abs(got / want - 1.0)
        info[tag] = d
        if d > limit:
            why.append("%s %r vs %r (%.3f %%)" % (key, got, want, 100.0 * d))

    dev("outer_gap_m", 0.02, "gap_dev")
    dev("inner_thickness_m", 0.02, "thick_dev")
    rep = exp.get("inner_thickness_reported_m")
    if rep is not None and fp.get("inner_thickness_m") is not None:
        info["thick_rep_dev"] = abs(fp["inner_thickness_m"] / rep - 1.0)
    return "; ".join(why), info


def run_gate(family: str, n: int, seed: int, binary: str,
             keep: str | None = None) -> dict:
    """Generate, then check every row; the printed lines carry the numbers."""
    gen = _load(family)
    chks = checks_for(gen)
    tmp_root = keep
    if keep:
        if os.path.isdir(keep):
            if os.listdir(keep):
                raise ValueError("keep: %s is not empty" % keep)
        elif os.path.exists(keep):
            raise ValueError("keep: %s exists and is not a directory" % keep)
        else:
            os.makedirs(keep)
    else:
        tmp_root = tempfile.mkdtemp(prefix="gcorpus_%s_" % family)
    try:
        dir1 = os.path.join(tmp_root, "gen")
        dir2 = os.path.join(tmp_root, "regen")
        dir3 = os.path.join(tmp_root, "rewrite")
        case = os.path.join(tmp_root, "case")
        os.makedirs(dir3)
        os.makedirs(case)
        rows = gen.generate(seed, n, dir1)

        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONHASHSEED="12345")
        child = subprocess.run(
            [sys.executable, gen.__file__, "--seed", str(seed), "--n", str(n),
             "--out", dir2], capture_output=True, text=True, encoding="utf-8",
            errors="replace", env=env, timeout=600)
        man1 = os.path.join(dir1, "manifest_%s.jsonl" % family)
        man2 = os.path.join(dir2, "manifest_%s.jsonl" % family)
        man_same = os.path.isfile(man2) and open(man1, "rb").read() \
            == open(man2, "rb").read()
        regen_why = {}
        for row in rows:
            why = ""
            if child.returncode != 0:
                why = "child exited %d" % child.returncode
            else:
                p = os.path.join(dir2, row["geometry_id"] + ".stl")
                with open(p, "rb") as f:
                    if hashlib.sha256(f.read()).hexdigest() != row["stl_sha256"]:
                        why = "regenerated sha256 differs"
            if not man_same:
                why = (why + "; " if why else "") + "manifest is not byte-identical"
            regen_why[row["geometry_id"]] = why

        results, fails = [], []
        for row in rows:
            stl_path = os.path.join(dir1, row["geometry_id"] + ".stl")
            res = check_row(row, stl_path, binary,
                            {"case": case, "rw": dir3})
            res["regen"] = regen_why[row["geometry_id"]]
            results.append((row, res))
            for c in chks:
                if res[c]:
                    fails.append("%s %s: %s" % (row["geometry_id"], c, res[c]))
        counts = {c: sum(1 for _r, res in results if res[c] == "")
                  for c in chks}
        devs = [res["_vol_dev"] for _r, res in results
                if res["_vol_dev"] is not None]
        worst = "%.3f %%" % (100.0 * max(devs)) if devs else "n/a"
        nts = [res["_nT"] for _r, res in results]
        print("G-CORPUS %s seed %d n %d: %s" % (family, seed, n,
              ", ".join("%s %d/%d" % (c, counts[c], n) for c in chks)))
        print("G-CORPUS %s volume: max |V_file/V_closed - 1| = %s over %d "
              "measured of %d (limit 1 %%); %s"
              % (family, worst, len(devs), n, gen.VOLUME_FORM))
        if family == "A":
            # pair each untapered row with ITS OWN deviation (a row whose
            # re-read failed has none and is counted, not skipped)
            ut = [res["_vol_dev"] for r, res in results
                  if r["params"]["taper"] == 1.0]
            ut_dev = [x for x in ut if x is not None]
            print("G-CORPUS %s untapered: %d rows (%d measured), max "
                  "|V_file/(0.685*t*c^2*span) - 1| = %s (limit 1 %%)"
                  % (family, len(ut), len(ut_dev),
                     "%.3f %%" % (100.0 * max(ut_dev)) if ut_dev else "n/a"))
        strata = {"easy": 0, "medium": 0, "hard": 0}
        for r, _res in results:
            strata[r["stratum"]] += 1
        shapes = ""
        if any("shape" in r["params"] for r, _res in results):
            sc = {}
            for r, _res in results:
                sc[r["params"]["shape"]] = sc.get(r["params"]["shape"], 0) + 1
            shapes = "; shapes " + ", ".join("%s %d" % kv for kv in sorted(sc.items()))
        print("G-CORPUS %s strata: easy %d, medium %d, hard %d; triangles "
              "%d..%d%s" % (family, strata["easy"], strata["medium"],
                            strata["hard"], min(nts), max(nts), shapes))
        if hasattr(gen, "expected_features"):
            ck = sum(1 for r, _ in results
                     if gen.expected_features(r["params"])["commensurate"])
            a = sum(1 for r, res in results
                    if gen.expected_features(r["params"])["commensurate"]
                    and res["_feat"]["commensurate"]
                    and res["_feat"]["planar_frac"] == 1.0)
            m = len(results) - ck
            b = sum(1 for r, res in results
                    if not gen.expected_features(r["params"])["commensurate"]
                    and res["_feat"]["commensurate"] is False)
            print("G-CORPUS %s features: commensurate rows %d (planar_frac "
                  "1.0 and commensurate True on %d/%d); other rows %d "
                  "(commensurate False on %d/%d)"
                  % (family, ck, a, ck, m, b, m))
            if family == "E":
                gd = [res["_feat"]["gap_dev"] for r, res in results
                      if res["_feat"]["gap_dev"] is not None]
                gh = [r["params"]["gap_over_h"] for r, _ in results]
                worst = "%.3f %%" % (100.0 * max(gd)) if gd else "n/a"
                print("G-CORPUS E gap: gap/h %s..%s, max |gap_features/gap "
                      "- 1| = %s over %d (limit 2 %%)"
                      % ("%.3f" % min(gh), "%.3f" % max(gh), worst, len(gd)))
            if family == "F":
                pd = [res["_feat"]["thick_dev"] for r, res in results
                      if r["params"]["shape"] in ("plate_c", "plate_n")
                      and res["_feat"]["thick_dev"] is not None]
                fd = [res["_feat"]["thick_rep_dev"] for r, res in results
                      if r["params"]["shape"] == "fin"
                      and res["_feat"]["thick_rep_dev"] is not None]
                print("G-CORPUS F thickness: plates max |t_features/t - 1| = "
                      "%s over %d (limit 2 %%); fins max %s over %d "
                      "(reported, not gated)"
                      % ("%.3f %%" % (100.0 * max(pd)) if pd else "n/a",
                         len(pd),
                         "%.3f %%" % (100.0 * max(fd)) if fd else "n/a",
                         len(fd)))
        ok = all(counts[c] == n for c in chks)
        if ok:
            print("G-CORPUS PASS")
        else:
            print("G-CORPUS FAIL: " + " | ".join(fails[:10]))
        return {"pass": ok, "counts": counts, "fails": fails}
    finally:
        if not keep:
            shutil.rmtree(tmp_root, ignore_errors=True)


def _selftest():
    def group_a():
        out = run_gate("A", 24, 7, BINARY)
        assert out["pass"], out["fails"][:5]
        return "every check 24/24 (counts line above)"

    def group_b():
        out = run_gate("B", 24, 7, BINARY)
        assert out["pass"], out["fails"][:5]
        return "every check 24/24 (counts line above)"

    def group_d():
        out = run_gate("D", 16, 7, BINARY)
        assert out["pass"], out["fails"][:5]
        return "every check 16/16 incl. features (counts line above)"

    def group_e():
        out = run_gate("E", 16, 7, BINARY)
        assert out["pass"], out["fails"][:5]
        return "every check 16/16 incl. features (counts line above)"

    def group_f():
        out = run_gate("F", 16, 7, BINARY)
        assert out["pass"], out["fails"][:5]
        return "every check 16/16 incl. features (counts line above)"

    def _ctrl_bytes(row, data, name):
        d = tempfile.mkdtemp(prefix="gctrl_")
        try:
            scratch = {"case": os.path.join(d, "case"), "rw": os.path.join(d, "rw")}
            os.makedirs(scratch["case"])
            os.makedirs(scratch["rw"])
            path = os.path.join(d, name + ".stl")
            with open(path, "wb") as f:
                f.write(data)
            return check_row(row, path, BINARY, scratch)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def group_controls():
        wing = importlib.import_module("gen_wing")
        lathe = importlib.import_module("gen_lathe")
        bluff = importlib.import_module("gen_bluff")
        gap_gen = importlib.import_module("gen_gap")
        thin = importlib.import_module("gen_thin")
        verdicts = []

        # (1) an open lathe: the ogive's nose-pole fan removed
        row, _ = lathe.make_row(7, 1)
        P, T = lathe.build(row["params"])
        n_theta = row["params"]["n_theta"]
        res = _ctrl_bytes(row, stl_io.stl_bytes(P, T[n_theta:], lathe.SOLID),
                          "open_ogive")
        assert "before.open_edges = %d" % n_theta in res["closed"], res["closed"]
        assert res["dryrun"] and "surface/closed" in res["dryrun"], res["dryrun"]
        extra1 = sorted(c for c in CHECKS if res[c] and c not in
                        ("closed", "dryrun"))
        verdicts.append("open ogive: closed (open_edges = %d), dryrun (%s); "
                        "also failed %s" % (n_theta, "exit 1 surface/closed",
                                            ", ".join(extra1) or "nothing"))

        # (2) a planted -0.000000000e+00 in ONE vertex line
        row, data = wing.make_row(7, 1)
        lines = data.split(b"\n")
        hit = -1
        for i, ln in enumerate(lines):
            if ln.startswith(b"vertex ") and b" 0.000000000e+00" in ln:
                lines[i] = ln.replace(b" 0.000000000e+00",
                                      b" -0.000000000e+00", 1)
                hit = i
                break
        assert hit >= 0, "no zero coordinate found to plant -0. into"
        data2 = b"\n".join(lines)
        assert data2 != data and b"-0." in data2
        res = _ctrl_bytes(row, data2, "planted_negzero")
        assert res["no_negzero"], "the planted -0. passed no_negzero"
        extra2 = sorted(c for c in CHECKS if res[c] and c != "no_negzero")
        verdicts.append("planted -0.: no_negzero; also failed %s (the bit-exact "
                        "weld sees -0.0 and +0.0 as two points)"
                        % (", ".join(extra2) or "nothing"))

        # (3) a flipped triangle
        row, _ = wing.make_row(7, 3)
        P, T = wing.build(row["params"])
        T2 = T.copy()
        T2[-1] = T2[-1][[0, 2, 1]]
        res = _ctrl_bytes(row, stl_io.stl_bytes(P, T2, wing.SOLID), "flipped")
        assert res["closed"], res["closed"]
        assert res["dryrun"], res["dryrun"]
        verdicts.append("flipped triangle: closed (%s), dryrun"
                        % res["closed"][:40])

        # (4) every x multiplied by 1.02
        row, _ = wing.make_row(7, 5)
        P, T = wing.build(row["params"])
        P2 = P.copy()
        P2[:, 0] *= 1.02
        res = _ctrl_bytes(row, stl_io.stl_bytes(P2, T, wing.SOLID), "scaled")
        assert res["volume"], res["volume"]
        verdicts.append("volume x1.02: volume (%s)" % res["volume"])

        # (5) a commensurate box_c made incommensurate (every z x 1.0003)
        row, _ = bluff.make_row(7, 0)
        assert row["params"]["shape"] == "box_c", row["params"]["shape"]
        P, T = bluff.build(row["params"])
        P2 = P.copy()
        P2[:, 2] *= 1.0003
        res = _ctrl_bytes(row, stl_io.stl_bytes(P2, T, bluff.SOLID),
                          "box_incommensurate")
        assert res["features"], res["features"]
        assert "commensurate False != expected True" in res["features"], \
            res["features"]
        verdicts.append("box_c z x1.0003: features (%s)" % res["features"])

        # (6) an E pair's gap closed by 5 per cent (body b moved in -y)
        row, _ = gap_gen.make_row(7, 0)
        assert row["params"]["shape"] == "boxes", row["params"]["shape"]
        P, T = gap_gen.build(row["params"])
        P2 = P.copy()
        P2[P2[:, 1] > 0.0, 1] -= 0.05 * gap_gen.gap(row["params"])
        res = _ctrl_bytes(row, stl_io.stl_bytes(P2, T, gap_gen.SOLID),
                          "gap_closed")
        assert res["features"], res["features"]
        assert "outer_gap_m" in res["features"], res["features"]
        verdicts.append("E gap -5%%: features (%s)" % res["features"])

        # (7) an E pair with body b removed: one closed component, not two
        row, _ = gap_gen.make_row(7, 0)
        P, T = gap_gen.build_bodies(row["params"])[0]
        res = _ctrl_bytes(row, stl_io.stl_bytes(P, T, gap_gen.SOLID),
                          "body_removed")
        assert "n_components = 1 != 2" in res["closed"], res["closed"]
        assert res["volume"], res["volume"]
        verdicts.append("E body removed: closed (n_components = 1 != 2), "
                        "volume (%s)" % res["volume"])

        # (8) a plate thickened by 5 per cent (its z-max face lifted)
        row, _ = thin.make_row(7, 0)
        assert row["params"]["shape"] == "plate_c", row["params"]["shape"]
        P, T = thin.build(row["params"])
        P2 = P.copy()
        P2[P2[:, 2] == P2[:, 2].max(), 2] = 1.05 * P2[:, 2].max()
        res = _ctrl_bytes(row, stl_io.stl_bytes(P2, T, thin.SOLID),
                          "plate_thickened")
        assert res["features"], res["features"]
        assert "inner_thickness_m" in res["features"], res["features"]
        verdicts.append("plate z x1.05: features (%s)" % res["features"])
        for v in verdicts:
            print("  [gate] %s" % v)
        return "8 of 8 failed by name"

    if not os.path.isfile(BINARY):
        print("SELFTEST FAIL: binary not found: %s" % BINARY)
        return 1
    groups = [("G-CORPUS A seed 7 n 24", group_a),
              ("G-CORPUS B seed 7 n 24", group_b),
              ("G-CORPUS D seed 7 n 16", group_d),
              ("G-CORPUS E seed 7 n 16", group_e),
              ("G-CORPUS F seed 7 n 16", group_f),
              ("negative controls", group_controls)]
    n_ok = 0
    for name, fn in groups:
        try:
            note = fn()
        except AssertionError as e:
            print("SELFTEST FAIL: %s: %s" % (name, e))
            return 1
        n_ok += 1
        print("[ok] %s: %s" % (name, note))
    print("SELFTEST PASS")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="gate", description=__doc__.splitlines()[0])
    ap.add_argument("--family", action="append", default=None,
                    choices=sorted(FAMILIES))
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--binary", default=BINARY)
    ap.add_argument("--keep", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()
    if not args.family or args.n is None or args.seed is None:
        ap.error("--family, --n and --seed are required (or --selftest)")
    rc = 0
    for fam in args.family:
        # one subdirectory per family, so --keep DIR works with several --family
        keep = os.path.join(args.keep, fam) if args.keep else None
        if not run_gate(fam, args.n, args.seed, args.binary, keep)["pass"]:
            rc = 1
    return rc


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    sys.exit(main())
