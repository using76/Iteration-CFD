#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""
score.py - one automesher run in, docs/15 §D's Outcome out (AM-2, G-SCORER).

Reads only what `ofgpu-automesher` itself prints and writes: the stage
banners and elapsed lines of src/automesher/driver.rs, the `error:` refusal
of src/bin/automesher.rs's main in SPEC-LIT §92.3's fixed grammar, and
<name>_summary.json (§92.14.3). Applies docs/15 §D: the closed failure
enum, F1-F5, strict failure, BLC_8 / BLC_full with schema.yplus_a_priori,
BLC_beta bounded per patch, and the polyMesh content sha256.

    python tools/autonomy/score.py --selftest
    python tools/autonomy/score.py --fixtures
    python tools/autonomy/score.py --hash-gate CONFIG [--pointer P] [--to V]
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import schema  # noqa: E402  (the package's own module, same directory)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
BINARY_DEFAULT = os.path.join(REPO, "rust", "target", "release",
                              "ofgpu-automesher" + (".exe" if os.name == "nt" else ""))
PROBES_DIR = os.path.join(HERE, "fixtures", "probes")
STAGES = ("octree", "castellate", "snap", "split", "layers")
LAYER_CLASSES = ("min_thickness", "retreat_snapped", "no_full_stack")
DEFAULT_BETAS = (0.5, 0.8, 0.95)
POLYMESH_FILES = ("points", "faces", "owner", "neighbour", "boundary")
BANNER_RE = re.compile(r"(?m)^=== stage (\d+)/(\d+)\s+(\w+) ===[ \t]*$")
ELAPSED_RE = re.compile(r"(?m)^--- (\w+): ([0-9.]+) s[ \t]*$")
ERROR_RE = re.compile(r"(?m)^error: (.*)$")
GATE_RE = re.compile(r"quality gate (G[1-7]) \(")
M_F2 = "F2: -check was not run on this attempt (the campaign runs it on its audit sample); F2 is false"
M_F3D = "F3d: stages[snap] carries no per-patch area_ratio (AM-R2 adds it); F3d is null"
M_OCT = "octree: stages[octree] carries no gate_passed or max_non_orth_deg (AM-R2 adds them)"
M_BETA = ("BLC_beta: layer rows carry no per-face tau or row area (AM-R1 adds them); "
          "BLC_beta is bounded from t1_min, mean_frac and full_area_frac")


class ScoreParseError(ValueError):
    """The run's reports are not what this scorer reads; never guessed past."""


def _norm(text: str) -> str:
    """(C1): every search runs on \n, whatever the console wrote."""
    return (text or "").replace("\r\n", "\n")


def last_stage(text: str) -> str | None:
    """The group(3) of the LAST stage banner in the run's output, or None."""
    m = None
    for m in BANNER_RE.finditer(text):
        pass
    if m is None:
        return None
    name = m.group(3)
    if name not in STAGES:
        raise ScoreParseError("last stage banner names %r, not one of %s"
                              % (name, ", ".join(STAGES)))
    return name


def parse_refusal(stdout: str, stderr: str, stage: str | None) -> dict | None:
    """The `error:` line of SPEC-LIT §92.3's fixed grammar, classed by name.

    automesher.rs's main prints it on stderr; the fixtures store both
    streams as one text, so stderr is searched first, then stdout.
    """
    g = None
    for src in (stderr, stdout):
        m = None
        for m in ERROR_RE.finditer(src):
            pass
        if m is None:
            continue
        line = m.group(1).rstrip()
        text = src[m.start():].rstrip()
        if line.startswith("surface/closed:"):
            cls = "surface_closed"
        else:
            g = GATE_RE.search(line)
            if g is not None:
                if stage is None:
                    raise ScoreParseError("a quality-gate refusal needs the stage "
                                          "banner and the log carries none: %r" % line)
                cls = "gate_%s@%s" % (g.group(1), stage)
            elif "the first layer is thinner than the quality gate allows" in line:
                cls = "layer_t1_G5"
            elif line.startswith("io error on "):
                cls = "io"
            else:
                cls = "config"
        return {"line": line, "text": text, "class": cls, "gate": g.group(1) if g else None}
    return None


def classify_drop(reason: str) -> str:
    """The binary's own drop text (layers.rs) to a class; an unknown one refuses."""
    if "the thickness fell below min_thickness * T" in reason:
        return "min_thickness"
    if re.search(r"the gate still failed after \d+ retreat\(s\)", reason):
        return "retreat_snapped"
    if ("the gate failed on cells no layer point reaches" in reason
            or "the applied displacement is zero at a layer point" in reason
            or "the patch carries no layer face" in reason):
        return "retreat_snapped"
    raise ScoreParseError("unrecognised layer drop reason: %r" % (reason,))


def h_f(config: dict) -> float | None:
    """The finest cell size, base_size / 2 ** max_level; None if either is missing."""
    dom = config.get("domain") or {}
    ref = config.get("refinement") or {}
    base = dom.get("base_size")
    max_level = ref.get("max_level")
    if base is None or max_level is None:
        return None
    return base / 2 ** max_level


def layer_rows(summary: dict) -> dict[str, dict]:
    """{patch name: layer row} from summary["stages"], per-region rows concatenated."""
    st = {s["stage"]: s for s in summary["stages"]}
    lay = st.get("layers")
    if lay is None or lay.get("skipped"):
        return {}
    rows: dict[str, dict] = {}
    if "regions" in lay:
        lists = [r["patches"] for r in lay["regions"]]
    else:
        lists = [lay.get("patches", [])]
    for lst in lists:
        for row in lst:
            if row["name"] in rows:
                raise ScoreParseError("layer rows name patch %r twice" % row["name"])
            rows[row["name"]] = row
    return rows


def beta_share_bounds(tau_min: float, m: float, f: float, b: float) -> tuple[float, float]:
    """Bounds on the area share with tau >= b, from tau_min, mean tau and full share.

    Two Markov-type inequalities on tau in [tau_min, 1] with mean m
    (docs/15 §D.2's BLC_beta; exact per face only after AM-R1).
    """
    if tau_min >= b:
        return (1.0, 1.0)
    lo = max(f, (m - b) / (1.0 - b), 0.0)
    hi = min(1.0, max(lo, (m - tau_min) / (b - tau_min)))
    return (min(lo, 1.0), hi)


def content_sha256(case_dir: str) -> str | None:
    """sha256 over the five polyMesh files, name- and length-prefixed; None without."""
    pm = os.path.join(case_dir, "constant", "polyMesh")
    if not os.path.isdir(pm):
        return None
    h = hashlib.sha256()
    for name in POLYMESH_FILES:
        path = os.path.join(pm, name)
        if not os.path.isfile(path):
            raise ScoreParseError("content_sha256: %s is missing from %s"
                                  % (name, pm))
        data = open(path, "rb").read()
        h.update(name.encode("ascii") + b"\0" + str(len(data)).encode("ascii") + b"\0")
        h.update(data)
    return h.hexdigest()


def score_run(*, exit_code, stdout, stderr, summary, config, patch_areas_m2, flow,
              timed_out=False, wall_seconds=None, check_exit=None, case_dir=None,
              gates=None, betas=DEFAULT_BETAS) -> dict:
    """One run in, {"outcome": ..., "content_sha256": ...} out (docs/15 §D)."""
    gates = gates or schema.load_gates()
    for b in betas:
        if not (isinstance(b, (int, float)) and not isinstance(b, bool) and 0 < b < 1):
            raise ValueError("beta %r is not in (0, 1)" % (b,))
    stdout, stderr = _norm(stdout), _norm(stderr)
    text = stdout + "\n" + stderr
    stage = last_stage(text)
    if wall_seconds is not None:
        seconds = float(wall_seconds)
    elif exit_code == 0 and summary is not None:
        seconds = float(summary["total_seconds"])
    else:
        seconds = sum(float(m.group(2)) for m in ELAPSED_RE.finditer(text))
    lay = config.get("layers") or {}

    def requested_patch(p):
        return p in (lay.get("patches") or []) and (lay.get("n") or 0) > 0

    def empty_patch(p):
        return {"name": p, "area_m2": patch_areas_m2[p], "requested": requested_patch(p),
                "n_layers": 0, "dropped": None, "full_area_frac": None,
                "t1_requested_m": None, "yplus_a_priori": None, "delivered": False,
                "capability_limited": False, "layer_class": None, "mean_frac": None,
                "t1_min_m": None}

    # ---- the F1 branch (§D.1: no mesh): a refused run's summary is stale
    f1 = False
    ref = None
    if timed_out:
        cls, f1 = "timeout", True
    elif exit_code is None:
        cls, f1 = "crash", True
    elif exit_code == 1:
        ref = parse_refusal(stdout, stderr, stage)
        cls, f1 = (ref["class"] if ref else "crash"), True
    elif exit_code != 0:
        cls, f1 = "crash", True
    elif summary is None:
        cls, f1 = "io", True
    if f1:
        flags = {"F1": True, "F2": False, "F3a": False, "F3b": False, "F3c": False,
                 "F3d": None, "F4": False, "F5": False}
        outcome = {
            "verdict": "fail", "failure_class": cls, "flags": flags,
            "failure": True, "strict_failure": True, "exit_code": exit_code,
            "last_stage": stage, "n_cells": None, "seconds": seconds,
            "pinned_frac": None, "p99_over_hf": None, "max_over_hf": None,
            "blc8_a_priori": 0.0, "blc_full_a_priori": 0.0,
            "h_f_m": None, "n_pinned": None, "n_boundary_points": None,
            "refusal_line": ref["line"] if ref else None,
            "blc_beta_a_priori": [{"beta": b, "lo": 0.0, "hi": 0.0, "exact": False}
                                  for b in betas],
            "missing_signals": [],
            "patches": [empty_patch(p) for p in patch_areas_m2],
        }
        return {"outcome": outcome, "content_sha256": None}

    # ---- a written mesh (exit 0, summary given): §92.14.3's summary is read
    if summary.get("stopped_after") is not None:
        raise ScoreParseError("score_run: a stopped run (-stopAfter %s) is a probe, "
                              "not an attempt" % (summary["stopped_after"],))
    cfg = summary["config"]
    st = {s["stage"]: s for s in summary["stages"]}
    for need in ("castellate", "snap", "layers"):
        if need not in st:
            raise ScoreParseError("score_run: summary stages carry no %s report" % need)
    sp = summary["surface"]["patches"]
    if set(sp) != set(patch_areas_m2):
        raise ScoreParseError("score_run: patch areas given for %s but the summary's "
                              "surface carries %s" % (sorted(patch_areas_m2), sorted(sp)))
    total = sum(patch_areas_m2[p] for p in sp)
    if total <= 0:
        raise ScoreParseError("score_run: the STL wall area sums to %r, not positive"
                              % (total,))
    requested = set(cfg["layers"]["patches"]) if cfg["layers"]["n"] > 0 else set()
    hf = h_f(cfg)
    snap = st["snap"]
    nb = snap["n_boundary_points"]
    npin = snap["n_pinned"]
    pinned_frac = npin / nb if nb > 0 else None
    p99_over_hf = snap["p99_residual"] / hf if hf is not None else None
    max_over_hf = snap["max_residual"] / hf if hf is not None else None
    flags = {
        "F1": False,
        "F2": check_exit is not None and check_exit != 0,
        "F3a": pinned_frac is not None and pinned_frac > gates["pinned_frac_max"],
        "F3b": p99_over_hf > gates["p99_residual_over_hf_max"],
        "F3c": max_over_hf > gates["max_residual_over_hf_max"],
        "F3d": None,
        "F4": any(st["castellate"]["wall_patches"].get(p, 0) == 0 for p in sp)
              or summary["quality"]["n_regions"] != 1,
        "F5": summary["mesh"]["n_cells"] > gates["cell_budget"],
    }
    rows = layer_rows(summary)
    for p in sp:
        if p in requested and p not in rows:
            raise ScoreParseError("score_run: layers.patches requests %r but the layers "
                                  "stage writes no row for it" % p)
    blc8 = blcf = 0.0
    acc = {b: [0.0, 0.0] for b in betas}
    patches = []
    first_dropped = None
    for p in sp:
        a = patch_areas_m2[p]
        r = rows.get(p)
        if r is not None:
            n_layers = r["n_layers"]
            dropped = r["dropped"]
            full = r["full_area_frac"]
            t1r = r["t1_requested"] if r["t1_requested"] > 0 else None
            mean = r["mean_frac"]
            t1min = r["t1_min"]
            if dropped is not None:
                layer_class = classify_drop(dropped)
            elif n_layers > 0 and full == 0.0:
                layer_class = "no_full_stack"
            else:
                layer_class = None
        else:
            n_layers, dropped, full, t1r, mean, t1min, layer_class = 0, None, None, \
                None, None, None, None
        yplus = schema.yplus_a_priori(t1r, flow) if t1r else None
        y_ok = yplus is not None and yplus <= gates["yplus_max_a_priori"]
        delivered = r is not None and dropped is None \
            and n_layers >= gates["delivered_min_layers"]
        blc8 += a * (delivered and y_ok)
        blcf += a * (full or 0.0) * y_ok
        for b in betas:
            if (r is not None and dropped is None and n_layers > 0 and y_ok and t1r):
                lo, hi = beta_share_bounds(t1min / t1r, mean, full, b)
            else:
                lo, hi = 0.0, 0.0
            acc[b][0] += a * lo
            acc[b][1] += a * hi
        patches.append({"name": p, "area_m2": a, "requested": p in requested,
                        "n_layers": n_layers, "dropped": dropped,
                        "full_area_frac": full, "t1_requested_m": t1r,
                        "yplus_a_priori": yplus, "delivered": delivered,
                        "capability_limited": False, "layer_class": layer_class,
                        "mean_frac": mean, "t1_min_m": t1min})
        if first_dropped is None and p in requested and layer_class is not None:
            first_dropped = layer_class
    failure_class = "layer_dropped:" + first_dropped if first_dropped else None
    strict = any(v is True for v in flags.values()) or \
        any(q["requested"] and q["dropped"] is not None for q in patches)
    missing = []
    if check_exit is None:
        missing.append(M_F2)
    missing.append(M_F3D)
    if "gate_passed" not in st.get("octree", {}):
        missing.append(M_OCT)
    missing.append(M_BETA)
    outcome = {
        "verdict": "fail" if any(v is True for v in flags.values()) else "pass",
        "failure_class": failure_class, "flags": flags,
        "failure": any(v is True for v in flags.values()),
        "strict_failure": strict, "exit_code": 0, "last_stage": stage,
        "n_cells": summary["mesh"]["n_cells"], "seconds": seconds,
        "pinned_frac": pinned_frac, "p99_over_hf": p99_over_hf,
        "max_over_hf": max_over_hf,
        "blc8_a_priori": min(1.0, blc8 / total),
        "blc_full_a_priori": min(1.0, blcf / total),
        "h_f_m": hf, "n_pinned": npin, "n_boundary_points": nb,
        "refusal_line": None,
        "blc_beta_a_priori": [{"beta": b, "lo": min(1.0, acc[b][0] / total),
                               "hi": min(1.0, acc[b][1] / total), "exact": False}
                              for b in betas],
        "missing_signals": missing, "patches": patches,
    }
    return {"outcome": outcome,
            "content_sha256": content_sha256(case_dir) if case_dir else None}


def outcome_errors(outcome: dict, gates: dict | None = None,
                   knobs: dict | None = None) -> list[str]:
    """The outcome pasted into the good attempt row; every schema + S-refusal."""
    gates = gates if gates is not None else schema.load_gates()
    knobs = knobs if knobs is not None else schema.load_knobs()
    good = copy.deepcopy(schema._read_json(schema.FIXTURES_PATH)["attempt"])
    good["outcome"] = outcome
    good["content_sha256"] = None
    errs = schema.errors(good, "AttemptRow")
    return errs + (schema.check_attempt(good, gates, knobs) if not errs else [])


def score_probe(probe_id: str, labels: dict, flow: dict | None = None,
                betas=None) -> dict:
    """Score one frozen probe directory; the labels carry its run facts."""
    d = os.path.join(PROBES_DIR, probe_id)
    with open(os.path.join(d, "log.txt"), encoding="utf-8") as fh:
        log = _norm(fh.read())
    with open(os.path.join(d, "config.json"), encoding="utf-8") as fh:
        config = json.load(fh)
    spath = os.path.join(d, "summary.json")
    summary = json.load(open(spath, encoding="utf-8")) if os.path.isfile(spath) else None
    row = next(r for r in labels["probes"] if r["id"] == probe_id)
    return score_run(exit_code=row["exit_code"], stdout=log, stderr="",
                     summary=summary, config=config,
                     patch_areas_m2=row["patch_areas_m2"],
                     flow=flow or labels["flow"],
                     wall_seconds=row["wall_seconds"],
                     betas=tuple(betas or labels["betas"]))


def run_automesher(binary: str, config_path: str, args: list[str], cwd: str,
                   timeout_s: float) -> dict:
    """One automesher process; the timeout kills its own child by handle."""
    t0 = time.perf_counter()
    try:
        p = subprocess.run([binary, config_path, *args], cwd=cwd,
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout_s)
        return {"exit_code": p.returncode, "stdout": p.stdout, "stderr": p.stderr,
                "seconds": time.perf_counter() - t0, "timed_out": False}
    except subprocess.TimeoutExpired as e:
        out = e.stdout if isinstance(e.stdout, str) else (e.stdout or b"").decode(
            "utf-8", "replace")
        err = e.stderr if isinstance(e.stderr, str) else (e.stderr or b"").decode(
            "utf-8", "replace")
        return {"exit_code": None, "stdout": out, "stderr": err,
                "seconds": time.perf_counter() - t0, "timed_out": True}


def run_check(binary: str, config_path: str, case_dir: str,
              timeout_s: float = 600.0) -> dict:
    """§92.14.5 `-check` on a written mesh; the refusal line and its gate named."""
    res = run_automesher(binary, config_path, ["-check", case_dir],
                         cwd=os.path.dirname(os.path.abspath(config_path)),
                         timeout_s=timeout_s)
    line = None
    for src in (_norm(res["stderr"]), _norm(res["stdout"])):
        m = None
        for m in ERROR_RE.finditer(src):
            pass
        if m is not None:
            line = m.group(1).rstrip()
            break
    g = GATE_RE.search(line) if line else None
    res["refusal_line"] = line
    res["gate"] = g.group(1) if g else None
    return res


def _fmt(x, spec="%.4f") -> str:
    return spec % x if isinstance(x, (int, float)) and not isinstance(x, bool) else "-"


def print_fixtures(labels: dict) -> int:
    """One line per frozen probe: id, class, verdict, strict, flags, numbers."""
    for row in labels["probes"]:
        oc = score_probe(row["id"], labels)["outcome"]
        hot = ",".join(k for k, v in oc["flags"].items() if v is True) or "-"
        print("%-13s %-32s %-4s strict=%-5s pinned=%-9s p99/hf=%-9s max/hf=%-9s "
              "BLC8=%.3f BLCfull=%.3f flags=%s"
              % (row["id"], oc["failure_class"] or "-", oc["verdict"],
                 oc["strict_failure"], _fmt(oc["pinned_frac"]),
                 _fmt(oc["p99_over_hf"]), _fmt(oc["max_over_hf"]),
                 oc["blc8_a_priori"], oc["blc_full_a_priori"], hot))
    return len(labels["probes"])


def _leaf_eq(a, b) -> bool:
    """G-SCORER's equality: None only to None, bools/strings exact, numbers close."""
    if a is None or b is None:
        return a is b
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def _compare_expect(expect: dict, oc: dict, pid: str, skip=("seconds",)) -> tuple[int, list[str]]:
    """Every expect leaf against the outcome; (leaf count, disagreements)."""
    n = 0
    bad = []
    for k, v in expect.items():
        if k in ("patches", "blc_beta_a_priori") or k in skip:
            continue
        n += 1
        if not _leaf_eq(v, oc.get(k)):
            bad.append("%s.%s: expected %r, got %r" % (pid, k, v, oc.get(k)))
    if "blc_beta_a_priori" in expect:
        n += 1
        eb, gb = expect["blc_beta_a_priori"], oc["blc_beta_a_priori"]
        if len(eb) != len(gb):
            bad.append("%s.blc_beta_a_priori: expected %d bounds, got %d"
                       % (pid, len(eb), len(gb)))
        else:
            for j, (e, g) in enumerate(zip(eb, gb)):
                for f in ("beta", "lo", "hi", "exact"):
                    n += 1
                    if not _leaf_eq(e[f], g[f]):
                        bad.append("%s.blc_beta_a_priori[%d].%s: expected %r, got %r"
                                   % (pid, j, f, e[f], g[f]))
    if "patches" in expect:
        by_name = {p["name"]: p for p in oc["patches"]}
        for e in expect["patches"]:
            g = by_name.get(e["name"])
            if g is None:
                bad.append("%s.patches[%s]: no PatchOutcome named that"
                           % (pid, e["name"]))
                continue
            for f in ("requested", "n_layers", "full_area_frac",
                      "yplus_a_priori", "delivered"):
                if f in e:
                    n += 1
                    if not _leaf_eq(e[f], g[f]):
                        bad.append("%s.patches[%s].%s: expected %r, got %r"
                                   % (pid, e["name"], f, e[f], g[f]))
            if "dropped" in e:
                n += 1
                want, got = e["dropped"], g["dropped"] is not None
                if not _leaf_eq(want, got):
                    bad.append("%s.patches[%s].dropped: expected %r, got %r"
                               % (pid, e["name"], want, got))
            if "t1_min_over_requested" in e:
                n += 1
                got = None if g["t1_requested_m"] is None else \
                    g["t1_min_m"] / g["t1_requested_m"]
                if not _leaf_eq(e["t1_min_over_requested"], got):
                    bad.append("%s.patches[%s].t1_min_over_requested: expected %r, "
                               "got %r" % (pid, e["name"], e["t1_min_over_requested"], got))
            if "row" in e:
                n += 1
                got = g["n_layers"] != 0 or g["dropped"] is not None
                if not _leaf_eq(e["row"], got):
                    bad.append("%s.patches[%s].row: expected %r, got %r"
                               % (pid, e["name"], e["row"], got))
    return n, bad


NINE_LAYER_PROBES = ("box_sphere", "rev2_L4", "box_L4", "wing_a_L3", "wing_a_L4",
                     "wing_a_L5", "wing_b_L4", "wing_c_L4", "wb_L4")


def _g_scorer(labels: dict, results: dict, lines: list) -> None:
    """(C6): every expected number, bool, string and null reproduces."""
    bad = []
    n = 0
    valid = 0
    for row in labels["probes"]:
        pid = row["id"]
        oc = results[pid]
        k, b = _compare_expect(row["expect"], oc, pid, skip=())
        n += k
        bad += b
        errs = outcome_errors(oc)
        n += 1
        if errs:
            bad.append("%s.outcome: %d refusal(s): %s" % (pid, len(errs), errs[0]))
        else:
            valid += 1
    if bad:
        raise AssertionError("G-SCORER: %d disagreement(s):\n  %s"
                             % (len(bad), "\n  ".join(bad)))
    lines.append("[ok] G-SCORER: %d of %d probes reproduce %d numbers, bools and "
                 "classes; %d outcomes valid; 0 disagreements"
                 % (len(labels["probes"]), len(labels["probes"]), n, valid))


def _nine_layer_probes(results: dict, lines: list) -> None:
    """(R6): §B's nine exit 0 with zero layers and are strict failures by name."""
    classes = {}
    f3a = 0
    for pid in NINE_LAYER_PROBES:
        oc = results[pid]
        assert oc["exit_code"] == 0, (pid, "exit", oc["exit_code"])
        assert oc["strict_failure"], (pid, "not a strict failure")
        assert oc["failure_class"].startswith("layer_dropped:"), \
            (pid, oc["failure_class"])
        assert oc["blc8_a_priori"] == 0.0 and oc["blc_full_a_priori"] == 0.0, pid
        classes[oc["failure_class"]] = classes.get(oc["failure_class"], 0) + 1
        f3a += 1 if oc["flags"]["F3a"] else 0
    assert classes == {"layer_dropped:min_thickness": 1,
                       "layer_dropped:retreat_snapped": 8}, classes
    assert f3a == 7, f3a
    lines.append("[ok] the nine layer probes: 9 of 9 exit 0 with zero layers and are "
                 "strict failures by name (1 layer_dropped:min_thickness, 8 "
                 "layer_dropped:retreat_snapped); 7 of 9 also fail F3a")


def _section_f(results: dict, lines: list) -> None:
    """(R7): the five labels docs/15 §F names for G-SCORER."""
    bs = results["box_sphere"]
    assert bs["failure_class"] == "layer_dropped:min_thickness", bs["failure_class"]
    assert bs["blc8_a_priori"] == 0.0 and bs["blc_full_a_priori"] == 0.0
    cn = results["cubep_nofeat"]
    assert cn["verdict"] == "pass" and cn["failure_class"] is None
    assert cn["blc_full_a_priori"] == 1.0
    cc = results["cubep_nofeat_cf"]
    assert cc["blc_full_a_priori"] == 0.0
    p = cc["patches"][0]
    assert p["dropped"] is None and p["n_layers"] == 3
    assert round(p["t1_min_m"] / p["t1_requested_m"], 4) == 0.2506, \
        p["t1_min_m"] / p["t1_requested_m"]
    assert results["NO25"]["failure_class"] == "gate_G4@castellate"
    wa = results["wing_a_L4"]
    assert wa["flags"]["F3a"] is True
    assert round(wa["pinned_frac"], 4) == 0.4039, wa["pinned_frac"]
    lines.append("[ok] section F labels: 5 of 5 (box_sphere, cubep_nofeat, "
                 "cubep_nofeat_cf, NO25, wing_a_L4)")


def _flow_variants(labels: dict, lines: list) -> None:
    """(R8): the y+ indicator and the BLC_beta bounds move with the flow."""
    parts = []
    for v in labels["variants"]:
        vid = v["id"]
        oc = score_probe(vid, labels, flow=v["flow"], betas=v["betas"])["outcome"]
        e = v["expect"]
        assert _leaf_eq(e["yplus_a_priori"], oc["patches"][0]["yplus_a_priori"]), \
            (vid, "yplus", e["yplus_a_priori"], oc["patches"][0]["yplus_a_priori"])
        for k in ("blc8_a_priori", "blc_full_a_priori", "failure_class", "verdict"):
            if k in e:
                assert _leaf_eq(e[k], oc[k]), (vid, k, e[k], oc[k])
        if "blc_beta_a_priori" in e:
            assert len(e["blc_beta_a_priori"]) == len(oc["blc_beta_a_priori"]), vid
            for j, eb in enumerate(e["blc_beta_a_priori"]):
                gb = oc["blc_beta_a_priori"][j]
                for f in ("beta", "lo", "hi", "exact"):
                    assert _leaf_eq(eb[f], gb[f]), (vid, j, f, eb[f], gb[f])
        hot = [b for b in e["blc_beta_a_priori"] if b["lo"] or b["hi"]]
        if hot:
            desc = "BLC_beta " + ", ".join("%g [%g, %g]" % (b["beta"], b["lo"], b["hi"])
                                           for b in e["blc_beta_a_priori"])
        else:
            desc = "BLC_full %g" % e["blc_full_a_priori"]
        parts.append("%s y+ %.4g -> %s" % (vid, e["yplus_a_priori"], desc))
    lines.append("[ok] flow variants: %d of %d (%s)"
                 % (len(labels["variants"]), len(labels["variants"]), "; ".join(parts)))


def _beta_bounds(lines: list) -> None:
    """(R9): the closed-form pairs beta_share_bounds must reproduce."""
    cases = [((0.2, 0.6, 0.3, 0.5), (0.3, 1.0)),
             ((0.2, 0.6, 0.3, 0.8), (0.3, 0.6666666666666666)),
             ((0.2, 0.6, 0.3, 0.95), (0.3, 0.5333333333333333)),
             ((1.0, 1.0, 1.0, 0.95), (1.0, 1.0)),
             ((0.25062656641604008, 0.2506265664160399, 0.0, 0.5), (0.0, 0.0))]
    for args, want in cases:
        got = beta_share_bounds(*args)
        ok = all(math.isclose(g, w, rel_tol=1e-12, abs_tol=1e-15)
                 for g, w in zip(got, want)) and len(got) == 2
        assert ok, (args, got, want)
    lines.append("[ok] BLC_beta bounds: %d of %d (tau_min, mean, full, beta) pairs "
                 "match the Markov bounds" % (len(cases), len(cases)))


def _probe_json(*parts) -> dict:
    with open(os.path.join(PROBES_DIR, *parts), encoding="utf-8") as fh:
        return json.load(fh)


def _refusal_classes(labels: dict, lines: list) -> None:
    """(R10): io, config, crash x3, timeout, and a stale summary ignored."""
    cfg = _probe_json("cubep_nofeat", "config.json")
    summary = _probe_json("cubep_nofeat", "summary.json")
    row = next(r for r in labels["probes"] if r["id"] == "cubep_nofeat")
    no25_log = open(os.path.join(PROBES_DIR, "NO25", "log.txt"),
                    encoding="utf-8").read()

    def run(**kw):
        base = dict(exit_code=1, stdout="", stderr="", summary=None, config=cfg,
                    patch_areas_m2=row["patch_areas_m2"], flow=labels["flow"])
        base.update(kw)
        oc = score_run(**base)["outcome"]
        errs = outcome_errors(oc)
        assert errs == [], (kw, errs)
        return oc

    oc = run(stderr="\nerror: io error on x.stl: The system cannot find the file")
    assert oc["failure_class"] == "io", oc["failure_class"]
    oc = run(stderr="\nerror: layers.growth: 0.9 is not in [1, 3]")
    assert oc["failure_class"] == "config", oc["failure_class"]
    assert run()["failure_class"] == "crash"
    assert run(exit_code=101)["failure_class"] == "crash"
    assert run(exit_code=None)["failure_class"] == "crash"
    assert run(timed_out=True, exit_code=None)["failure_class"] == "timeout"
    oc = run(stdout=_norm(no25_log), summary=copy.deepcopy(summary))
    assert oc["failure_class"] == "gate_G4@castellate", oc["failure_class"]
    assert oc["n_cells"] is None
    lines.append("[ok] refusal classes: 7 synthetic runs (io, config, crash x3, "
                 "timeout, stale summary ignored)")


def _drop_reasons(lines: list) -> None:
    """(R11): the six binary texts map; an unknown one refuses by name."""
    long_form = _probe_json("L2", "summary.json")["stages"][4]["patches"][0]["dropped"]
    cases = [
        ('patch "a": the thickness fell below min_thickness * T = 4.600e-3',
         "min_thickness"),
        ('patch "a": the gate still failed after 4 retreat(s)', "retreat_snapped"),
        (long_form, "retreat_snapped"),
        ('patch "a": the gate failed on cells no layer point reaches',
         "retreat_snapped"),
        ('patch "a": the applied displacement is zero at a layer point - x',
         "retreat_snapped"),
        ('patch "a": the patch carries no layer face', "retreat_snapped"),
    ]
    for reason, want in cases:
        got = classify_drop(reason)
        assert got == want, (reason, got, want)
    try:
        classify_drop('patch "a": something new')
    except ScoreParseError as e:
        assert "something new" in str(e), str(e)
    else:
        raise AssertionError("an unknown drop reason scored instead of refusing")
    lines.append("[ok] drop reasons: 6 mapped, 1 refused")


def _parse_errors(labels: dict, lines: list) -> None:
    """(R12): a stopped run, an area mismatch, an unknown drop, a missing row."""
    cfg = _probe_json("cubep_nofeat", "config.json")
    summary = _probe_json("cubep_nofeat", "summary.json")
    row = next(r for r in labels["probes"] if r["id"] == "cubep_nofeat")
    log = _norm(open(os.path.join(PROBES_DIR, "cubep_nofeat", "log.txt"),
                     encoding="utf-8").read())

    def run(summary, areas=None):
        return score_run(exit_code=0, stdout=log, stderr="", summary=summary,
                         config=cfg, patch_areas_m2=areas or row["patch_areas_m2"],
                         flow=labels["flow"])

    def must_fail(s, areas=None, why=""):
        try:
            run(s, areas)
        except ScoreParseError as e:
            return e
        raise AssertionError("no ScoreParseError for %s" % why)

    s = copy.deepcopy(summary)
    s["stopped_after"] = "octree"
    e = must_fail(s, why="stopped_after")
    assert "stopped" in str(e), str(e)
    e = must_fail(summary, areas={"other": 1.0}, why="area mismatch")
    assert "other" in str(e) and "cube" in str(e), str(e)
    s = copy.deepcopy(summary)
    s["stages"][4]["patches"][0]["dropped"] = 'patch "cube": weird'
    e = must_fail(s, why="unknown drop reason")
    assert "weird" in str(e), str(e)
    s = copy.deepcopy(summary)
    s["stages"][4]["patches"] = []
    e = must_fail(s, why="requested patch with no row")
    assert "cube" in str(e), str(e)
    lines.append("[ok] parse errors: 4 by name")


def _content_hash(lines: list) -> None:
    """(R13): equal for a copy, different after one byte, named refusals."""
    with tempfile.TemporaryDirectory() as tmp:
        case = os.path.join(tmp, "case")
        pm = os.path.join(case, "constant", "polyMesh")
        os.makedirs(pm)
        for name in POLYMESH_FILES:
            with open(os.path.join(pm, name), "wb") as fh:
                fh.write(b"polyMesh content of " + name.encode("ascii") + b"\n")
        h0 = content_sha256(case)
        assert h0 and len(h0) == 64
        two = os.path.join(tmp, "case2")
        shutil.copytree(case, two)
        assert content_sha256(two) == h0
        with open(os.path.join(two, "constant", "polyMesh", "owner"), "ab") as fh:
            fh.write(b"x")
        assert content_sha256(two) != h0
        os.remove(os.path.join(two, "constant", "polyMesh", "boundary"))
        try:
            content_sha256(two)
        except ScoreParseError as e:
            assert "boundary" in str(e), str(e)
        else:
            raise AssertionError("a missing polyMesh file hashed")
        empty = os.path.join(tmp, "empty")
        os.makedirs(empty)
        assert content_sha256(empty) is None
    lines.append("[ok] content hash: copy equal, one byte differs, missing file "
                 "refused, no mesh -> None")


def _live(lines: list) -> None:
    """(R14): two runs hash equal, a knob move changes the hash, -check gates."""
    binary = BINARY_DEFAULT
    if not os.path.isfile(binary):
        print("[skip] live automesher: no binary at rust/target/release")
        return
    knobs = schema.load_knobs()
    labels = _probe_json("labels.json")
    row = next(r for r in labels["probes"] if r["id"] == "cubep_nofeat")
    tmp = tempfile.mkdtemp()
    try:
        cfg = copy.deepcopy(_probe_json("cubep_nofeat", "config.json"))
        cfg["input"]["surfaces"][0]["path"] = \
            os.path.join(HERE, "fixtures", "stl", "cubep.stl").replace("\\", "/")
        cfg["output"]["case_dir"] = os.path.join(tmp, "case").replace("\\", "/")
        cfg["output"]["name"] = "live"
        live = os.path.join(tmp, "live.json")
        with open(live, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=1)
        assert schema.check_edit("/snap/feature_tolerance", 0.5, knobs) is None
        cfg_c = copy.deepcopy(cfg)
        cfg_c["snap"]["feature_tolerance"] = 0.5
        live_c = os.path.join(tmp, "live_c.json")
        with open(live_c, "w", encoding="utf-8") as fh:
            json.dump(cfg_c, fh, indent=1)
        runs = {}
        for tag, cfgp in (("a", live), ("b", live), ("c", live_c)):
            r = run_automesher(binary, cfgp, ["-tag", tag], cwd=tmp, timeout_s=600.0)
            assert r["exit_code"] == 0, (tag, r["stderr"][-400:])
            case = os.path.join(tmp, "case_" + tag)
            runs[tag] = (r, case, content_sha256(case))
        assert runs["a"][2] and runs["a"][2] == runs["b"][2], "two runs, one mesh"
        assert runs["c"][2] != runs["a"][2], "feature_tolerance must change the mesh"
        with open(os.path.join(tmp, "case_a", "live_a_summary.json"),
                  encoding="utf-8") as fh:
            summary_a = json.load(fh)
        res = score_run(exit_code=runs["a"][0]["exit_code"],
                        stdout=runs["a"][0]["stdout"], stderr=runs["a"][0]["stderr"],
                        summary=summary_a, config=cfg,
                        patch_areas_m2=row["patch_areas_m2"], flow=labels["flow"],
                        case_dir=os.path.join(tmp, "case_a"))
        # The frozen probe predates AM-R2: its missing_signals still names M_OCT. A live
        # binary that writes stages[octree].gate_passed legitimately drops that note.
        expect = copy.deepcopy(row["expect"])
        octree = {s["stage"]: s for s in summary_a["stages"]}.get("octree", {})
        if "gate_passed" in octree:
            assert M_OCT in expect.get("missing_signals", []), "frozen probe lost M_OCT"
            expect["missing_signals"].remove(M_OCT)
        k, bad = _compare_expect(expect, res["outcome"], "live-a")
        assert not bad, "live score differs from the frozen one:\n  %s" % "\n  ".join(bad)
        assert k > 0 and res["content_sha256"] == runs["a"][2]
        chk = run_check(binary, live, os.path.join(tmp, "case_a"))
        assert chk["exit_code"] == 0 and chk["refusal_line"] is None, chk
        cfg25 = copy.deepcopy(cfg)
        cfg25["quality"] = {"max_non_orth_deg": 25.0, "report_non_orth_deg": 20.0}
        no25 = os.path.join(tmp, "live_no25.json")
        with open(no25, "w", encoding="utf-8") as fh:
            json.dump(cfg25, fh, indent=1)
        chk25 = run_check(binary, no25, os.path.join(tmp, "case_a"))
        assert chk25["exit_code"] == 1 and chk25["gate"] == "G4", chk25
        res2 = score_run(exit_code=runs["a"][0]["exit_code"],
                         stdout=runs["a"][0]["stdout"], stderr=runs["a"][0]["stderr"],
                         summary=summary_a, config=cfg,
                         patch_areas_m2=row["patch_areas_m2"], flow=labels["flow"],
                         check_exit=1)
        assert res2["outcome"]["flags"]["F2"] is True
        assert res2["outcome"]["verdict"] == "fail"
        lines.append("[ok] live automesher: cubep_nofeat twice -> equal content sha256 "
                     "%s, feature_tolerance 0.5 -> different; live score equals the "
                     "frozen one; -check exit 0, and exit 1 (G4) under a 25 deg ceiling"
                     % runs["a"][2][:16])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def selftest() -> int:
    """The ten checks behind `score.py --selftest`; 1 and SELFTEST FAIL on any."""
    lines = []
    try:
        labels = _probe_json("labels.json")
        results = {r["id"]: score_probe(r["id"], labels)["outcome"]
                   for r in labels["probes"]}
        groups = [
            ("G-SCORER", lambda: _g_scorer(labels, results, lines)),
            ("the nine layer probes", lambda: _nine_layer_probes(results, lines)),
            ("section F labels", lambda: _section_f(results, lines)),
            ("flow variants", lambda: _flow_variants(labels, lines)),
            ("BLC_beta bounds", lambda: _beta_bounds(lines)),
            ("refusal classes", lambda: _refusal_classes(labels, lines)),
            ("drop reasons", lambda: _drop_reasons(lines)),
            ("parse errors", lambda: _parse_errors(labels, lines)),
            ("content hash", lambda: _content_hash(lines)),
            ("live automesher", lambda: _live(lines)),
        ]
        for name, fn in groups:
            try:
                fn()
            except (AssertionError, ValueError) as e:
                print("SELFTEST FAIL: %s: %s" % (name, e))
                return 1
    except (AssertionError, ValueError, OSError, KeyError) as e:
        print("SELFTEST FAIL: %s: %s" % (type(e).__name__, e))
        return 1
    for l in lines:
        print(l)
    print("SELFTEST PASS")
    return 0


def hash_gate(config_path: str, binary: str, pointer: str, to) -> int:
    """One config, three tagged runs: A == B by content sha256, C != A."""
    cfg_path = os.path.abspath(config_path)
    cwd = os.path.dirname(cfg_path)
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = json.load(fh)  # the gate needs a plain JSON config: no comments
    case_dir = cfg["output"]["case_dir"]
    if not os.path.isabs(case_dir):
        case_dir = os.path.join(cwd, case_dir)
    tags = {t: case_dir + "_hash" + t for t in ("A", "B", "C")}
    for t, d in tags.items():
        if os.path.exists(d):
            print("HASH GATE FAIL: %s exists; remove it first" % d)
            return 2
    ref = schema.check_edit(pointer, to, schema.load_knobs())
    if ref is not None:
        print("HASH GATE FAIL: %s" % ref["message"])
        return 2
    cfg_c = copy.deepcopy(cfg)
    node = cfg_c
    parts = [p for p in pointer.split("/") if p]
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = to
    cfg_c_path = os.path.join(cwd, os.path.splitext(os.path.basename(cfg_path))[0]
                              + ".hashC.json")
    with open(cfg_c_path, "w", encoding="utf-8") as fh:
        json.dump(cfg_c, fh, indent=1)
    os.makedirs(os.path.dirname(case_dir) or ".", exist_ok=True)
    results = {}
    try:
        for t, cfgp in (("A", cfg_path), ("B", cfg_path), ("C", cfg_c_path)):
            r = run_automesher(binary, cfgp, ["-tag", "hash" + t], cwd=cwd,
                               timeout_s=900.0)
            sha = content_sha256(tags[t]) if r["exit_code"] == 0 else None
            n_cells = None
            if r["exit_code"] == 0:
                spath = os.path.join(tags[t], "%s_hash%s_summary.json"
                                     % (cfg["output"]["name"], t))
                if os.path.isfile(spath):
                    with open(spath, encoding="utf-8") as fh:
                        n_cells = json.load(fh)["mesh"]["n_cells"]
            results[t] = (r["exit_code"], sha, r["timed_out"])
            print("hash%s exit %s %6.1f s n_cells %s sha256 %s"
                  % (t, r["exit_code"], r["seconds"], n_cells, sha or "-"))
        a, b, c = (results[t][1] for t in ("A", "B", "C"))
        if a and b and c and a == b and c != a:
            print("HASH GATE PASS: A == B and C != A")
            return 0
        print("HASH GATE FAIL: A %s B %s C %s" % (a or "-", b or "-", c or "-"))
        return 1
    finally:
        for d in tags.values():
            shutil.rmtree(d, ignore_errors=True)
        if os.path.isfile(cfg_c_path):
            os.remove(cfg_c_path)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--selftest":
        return selftest()
    if argv and argv[0] == "--fixtures":
        print_fixtures(_probe_json("labels.json"))
        return 0
    if argv and argv[0] == "--hash-gate":
        cfg, pointer, to, binary = argv[1], "/snap/smoothing_passes", 2, BINARY_DEFAULT
        i = 2
        while i < len(argv):
            if argv[i] == "--binary" and i + 1 < len(argv):
                binary = argv[i + 1]
            elif argv[i] == "--pointer" and i + 1 < len(argv):
                pointer = argv[i + 1]
            elif argv[i] == "--to" and i + 1 < len(argv):
                to = json.loads(argv[i + 1])
            else:
                print("unknown or incomplete argument %r" % argv[i])
                return 2
            i += 2
        return hash_gate(cfg, binary, pointer, to)
    print(__doc__.strip())
    return 2


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    sys.exit(main())
