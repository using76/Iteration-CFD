#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""
campaign.py - the campaign runner (AM-11, docs/15 §C act / verify / remember).

One campaign takes manifest rows (tuning, test or a file), and for every
geometry: regenerates the STL, gates it on stl_repair's closure (SURFACE-OPEN
is a named end with no mesher run), fingerprints it, builds attempt 1 by the
named mode and runs the remedy loop:

  b0-template  the naive template of docs/15 §F, one attempt, decided_by default
  b0-lhs       four Latin-hypercube configs around the template, best of 4
  rules        rules.setup builds attempt 1 under the preflight veto (the
               octree probe is the cost signal and PF-THIN is re-checked with
               snap_probe's measured post-snap wall edge), then remedies
  rules+prior  the prior hook may replace attempt 1 (AM-13's prior.attempt1)
  rules+opt    the optimiser hook may propose after EXHAUSTED (AM-14)
  full         prior + optimiser
  evaluate     a system name on the held-out split - the ONLY mode that reads
               it; every other access is refused before the file is opened

The runner pools at most MAX_STREAMS mesher jobs (the house cap while the
solver workflow owns the machine; docs/15 §C says 12), each with a hard
timeout killed through the child's own Popen handle, RAM admission from the
octree probe's n_leaves, a sampled peak, a 10 % audit sample that re-gates
(-check) and re-runs the mesh for an equal content hash, and deletes every
polyMesh after scoring.  One validated autonomy-attempt/1 row per attempt is
appended under the campaign lock; replay and compare reproduce every decision
from the rows alone.  G-DET is docs/15 §F: two runs of one campaign give
identical rows apart from the time fields.

    python tools/autonomy/campaign.py --selftest
    python tools/autonomy/campaign.py --run --manifest tuning --mode rules --out DIR
    python tools/autonomy/campaign.py --gate --out DIR [--parts smoke,1,2]
"""

import argparse
import copy
import hashlib
import importlib
import json
import math
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for _p in (HERE, os.path.join(HERE, "corpus")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import schema  # noqa: E402
import score  # noqa: E402
import preflight  # noqa: E402
import rules  # noqa: E402
import remedies  # noqa: E402
import explain  # noqa: E402
import split  # noqa: E402

BINARY_DEFAULT = score.BINARY_DEFAULT
STL_REPAIR = preflight.STL_REPAIR
REPORT_DIR = os.path.join(HERE, "campaign")
FIXTURE_DIR = os.path.join(HERE, "fixtures", "campaign")
CAMPAIGN_SCHEMA = "autonomy-campaign/1"
GEOMETRY_SCHEMA = "autonomy-campaign-geometry/1"
JOB_SCHEMA = "autonomy-campaign-job/1"
SAMPLE_SCHEMA = "autonomy-campaign-sample/1"
RECORD_LINE_SCHEMA = "autonomy-campaign-record/1"
END_SCHEMA = "autonomy-campaign-end/1"
SUMMARY_SCHEMA = "autonomy-campaign-summary/1"
REPLAY_SCHEMA = "autonomy-campaign-replay/1"
COMPARE_SCHEMA = "autonomy-campaign-compare/1"
GATE_SCHEMA = "autonomy-campaign-gate/1"
MODES = ("b0-template", "b0-lhs", "rules", "rules+prior", "rules+opt", "full",
         "evaluate")
SYSTEMS = MODES[:-1]
BASELINES = ("b0-template", "b0-lhs")
LAYERS = {"b0-template": (), "b0-lhs": (),
          "rules": ("preflight", "rules", "remedies"),
          "rules+prior": ("preflight", "rules", "remedies", "prior"),
          "rules+opt": ("preflight", "rules", "remedies", "optimiser"),
          "full": ("preflight", "rules", "remedies", "prior", "optimiser")}
ABLATABLE = ("preflight", "remedies")
HOOKS = {"prior": ("prior", "attempt1", "AM-13"),
         "optimiser": ("optimise", "propose", "AM-14")}
MAX_STREAMS = 6        # the house cap while the solver workflow runs (docs/15 §C says 12)
STREAMS_DOCS = 12      # docs/15 §C act: min(12, (free RAM - 4 GB) / measured peak)
AUDIT_MOD = 10         # docs/15 §C verify: a 10 % audit sample
TIMEOUT_S = 3600.0     # one full attempt
PROBE_TIMEOUT_S = 900.0
REPAIR_TIMEOUT_S = 300.0
SAMPLE_S = 5.0         # docs/15 §F G-COST: sampled every 5 s
HEARTBEAT_S = 60.0
RAM_FRACTION = 0.60    # docs/15 §F G-COST: peak RAM <= 60 %
RAM_RESERVE_MIB = 4096.0
MIB_BASE = 64.0
MIB_PER_LEAF = 2.5 / 1024.0   # measured 2.05-2.39 KiB of peak working set per probe leaf
B0_WALL_LEVEL = 4
B0_BAND_FRAC = 0.1
B0_BASE_FRAC = 0.5
B0_N = 8
B0_MARGINS = rules.DOMAIN_MARGINS            # (3, 6, 2.5, 2.5, 2.5, 2.5) x L
LHS_N = 4
LHS_GROWTH = (1.1, 1.3)
LHS_WALL = (-1, 0, 1)
LHS_FEATURE = (0, 1, 2)
LHS_FT = (0.0, 0.25, 0.5)
LHS_SP = (0, 1, 2, 3)
GEOMETRY_TERMINALS = remedies.TERMINALS + ("BASELINE", "REFUSED", "SURFACE-OPEN",
                                           "HARNESS-ERROR")
SMOKE_IDS = ("D-1-010", "F-1-009", "G-1-016", "G-1-026")
LIVE_IDS = ("D-1-010", "F-1-005", "G-1-026")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
FILES = {"campaign": "campaign.json", "attempts": "attempts.jsonl",
         "records": "records.jsonl", "geometries": "geometries.jsonl",
         "jobs": "jobs.jsonl", "samples": "samples.jsonl",
         "progress": "progress.json", "end": "campaign_end.json",
         "summary": "summary.json", "records_json": "records.json"}
TIME_KEYS = ("campaign_id", "t_start", "t_end")


class CampaignError(ValueError):
    """A refused campaign or a harness inconsistency - never a verdict."""


def sha(x):
    """schema.canonical_sha256 - the config / row / record hash used everywhere."""
    return schema.canonical_sha256(x)


def _sha256_of_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def stl_rel(gid):
    return "stl/%s.stl" % gid


def case_rel(gid):
    return "cases/%s" % gid


def config_rel(gid, a):
    return "configs/%s_a%d.json" % (gid, a)


def _read_jsonl(path):
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def load_manifest(source, mode, ids=None, limit=None):
    """The manifest rows, split-sealed and equal to the official corpus rows."""
    if source in ("tuning", "test"):
        rows = split.load(source, mode)
    else:
        with open(source, "r", encoding="utf-8") as f:
            rows = [json.loads(ln) for ln in f if ln.strip()]
        split.filter_rows(rows, mode)
        official = {r["geometry_id"]: r for r in split.load("tuning", mode)}
        if any(r["geometry_id"] not in official for r in rows):
            official.update({r["geometry_id"]: r for r in split.load("test", mode)})
        for r in rows:
            if sha(r) != sha(official[r["geometry_id"]]):
                raise CampaignError("row %s differs from the %s manifest"
                                    % (r["geometry_id"], source))
    seen = set()
    for r in rows:
        gid = r.get("geometry_id")
        if not gid:
            raise CampaignError("a row of manifest %s has no geometry_id" % source)
        if gid in seen:
            raise CampaignError("duplicate geometry %s in manifest %s" % (gid, source))
        seen.add(gid)
    if ids is not None:
        by = {r["geometry_id"]: r for r in rows}
        for gid in ids:
            if gid not in by:
                raise CampaignError("geometry %s is not in manifest %s" % (gid, source))
        rows = [by[gid] for gid in ids]
    if limit is not None:
        rows = rows[:limit]
    return rows


def manifest_sha(source):
    """The lock's sha for tuning/test, the file bytes' sha for a FILE."""
    if source in ("tuning", "test"):
        lock = split.read_lock(os.path.join(split.MANIFEST_DIR, "split.lock"))
        return lock[source].split(" ")[-1]
    return _sha256_of_file(source)


def b0_template(mrow, fp):
    """The B0-template config of docs/15 §F: bbox + margins on the base lattice,
    one uniform wall band, layers n 8 at the a priori t1, defaults elsewhere."""
    gid = mrow["geometry_id"]
    bb = fp["bbox"]
    big = max(bb[1] - bb[0], bb[3] - bb[2], bb[5] - bb[4])
    base = round(B0_BASE_FRAC * big, 4) + 0.0
    if base <= 0:
        raise CampaignError("b0_template: base_size %r is not positive for %s"
                            % (base, gid))
    m = [f * big for f in B0_MARGINS]
    k = [math.floor((bb[0] - m[0]) / base), math.ceil((bb[1] + m[1]) / base),
         math.floor((bb[2] - m[2]) / base), math.ceil((bb[3] + m[3]) / base),
         math.floor((bb[4] - m[4]) / base), math.ceil((bb[5] + m[5]) / base)]
    extent = [round(ki * base, 10) + 0.0 for ki in k]
    patches = []
    for p in fp["patches"]:
        if p["name"] not in patches:
            patches.append(p["name"])
    cfg = {"input": {"surfaces": [{"path": stl_rel(gid)}]},
           "domain": {"extent": extent, "base_size": base},
           "refinement": {"levels": [{"patch": p, "bands": [
               {"distance": round(B0_BAND_FRAC * big, 6), "level": B0_WALL_LEVEL}]}
               for p in patches], "max_level": B0_WALL_LEVEL},
           "output": {"case_dir": case_rel(gid), "name": gid}}
    w = schema.a_priori_wall(mrow["flow"])
    if w["t1_a_priori_m"] is not None:
        cfg["layers"] = {"patches": patches, "n": B0_N,
                         "first_thickness": rules.t1_floor(w["t1_a_priori_m"])}
    return cfg


def lhs_unit(geometry_id):
    """LHS_N x 6 Latin-hypercube unit draws, seeded by the geometry id alone."""
    rng = random.Random(int(hashlib.sha256(
        ("b0-lhs/" + geometry_id).encode()).hexdigest()[:16], 16))
    u = [[0.0] * 6 for _ in range(LHS_N)]
    for d in range(6):
        perm = list(range(LHS_N))
        rng.shuffle(perm)
        for i in range(LHS_N):
            u[i][d] = (perm[i] + rng.random()) / LHS_N
    return u


def b0_lhs(mrow, fp):
    """Four LHS configs around the B0-template, inside the L4 box of docs/15 §F."""
    gid = mrow["geometry_id"]
    t = b0_template(mrow, fp)
    bb = fp["bbox"]
    big = max(bb[1] - bb[0], bb[3] - bb[2], bb[5] - bb[4])

    def pick(seq, x):
        return seq[min(len(seq) - 1, int(x * len(seq)))]

    out = []
    for u in lhs_unit(gid):
        c = copy.deepcopy(t)
        wl = B0_WALL_LEVEL + pick(LHS_WALL, u[0])
        f = 0.5 * 4 ** u[1]
        fo = pick(LHS_FEATURE, u[2])
        feat = []
        for e in c["refinement"]["levels"]:
            for b in e["bands"]:
                b["level"] = wl
                b["distance"] = round(B0_BAND_FRAC * big * f, 6)
            if fo > 0:
                e["feature_level"] = min(6, wl + fo)
                feat.append(e["feature_level"])
        c["refinement"]["max_level"] = max([wl] + feat)
        c["snap"] = {"feature_tolerance": pick(LHS_FT, u[3]),
                     "smoothing_passes": pick(LHS_SP, u[4])}
        if "layers" in c:
            c["layers"]["growth"] = round(
                LHS_GROWTH[0] + (LHS_GROWTH[1] - LHS_GROWTH[0]) * u[5], 3)
        out.append(c)
    return out


def audit_selected(config_sha, mod=AUDIT_MOD):
    """The 10 % audit sample, keyed on the config alone so two runs agree."""
    return int(config_sha[:8], 16) % mod == 0


def expected_mib(n_leaves):
    """The G-COST admission estimate: 64 MiB + 2.5 KiB per octree probe leaf."""
    if isinstance(n_leaves, int) and not isinstance(n_leaves, bool) and n_leaves > 0:
        return MIB_BASE + MIB_PER_LEAF * n_leaves
    return MIB_BASE


class Ram:
    """The G-COST admission: expected peak MiB against the RAM budget.  A job
    larger than the budget runs alone rather than never."""

    def __init__(self, budget_mib):
        self.budget = float(budget_mib)
        self.used = 0.0
        self.n = 0
        self.cv = threading.Condition()

    def acquire(self, mib):
        with self.cv:
            while self.n > 0 and self.used + mib > self.budget:
                self.cv.wait(1.0)
            self.used += mib
            self.n += 1

    def release(self, mib):
        with self.cv:
            self.used -= mib
            self.n -= 1
            self.cv.notify_all()


class Campaign:
    """One campaign directory: append-only rows, job lines, samples, the PID
    runner and the heartbeat.  Also built directly by the selftest harness."""

    MESHER_KINDS = ("octree", "snap", "full", "check", "rerun")

    def __init__(self, out, *, campaign_id, mode, system, ablate, split_mode,
                 binary, streams, timeout_s, probe_timeout_s, audit_mod,
                 attempt_fn=None, probe_fn=None, snap_fn=None, hooks=None,
                 gates=None, knobs=None, quiet=False):
        self.out = out
        self.dir = os.path.abspath(out)
        self.campaign_id = campaign_id
        self.mode = mode
        self.system = system
        self.ablate = tuple(ablate)
        self.split_mode = split_mode
        self.binary = binary
        self.streams = streams
        self.timeout_s = timeout_s
        self.probe_timeout_s = probe_timeout_s
        self.audit_mod = audit_mod
        self.quiet = quiet
        self.layers = tuple(l for l in LAYERS[system] if l not in ablate)
        self.gates = gates if gates is not None else schema.load_gates()
        self.knobs = knobs if knobs is not None else schema.load_knobs()
        self.binary_sha = _sha256_of_file(binary)
        self.git_sha = self._git_sha()
        self.attempt_fn = attempt_fn or run_attempt
        self.probe_fn = probe_fn or octree_probe
        self.snap_fn = snap_fn or snap_probe
        self.hooks = dict(hooks or {})
        self.lock = threading.Lock()
        self.fp_lock = threading.Lock()
        self.progress_lock = threading.Lock()
        self.total_mib = psutil.virtual_memory().total / 2 ** 20
        self.ram = Ram(RAM_FRACTION * self.total_mib - 512.0)
        self.live = {}
        self.launched = []
        self.max_live_mesher = 0
        self.peak_rss_mib = 0.0
        self.peak_used_frac = 0.0
        self.n_done = 0
        self.n_rows = 0
        self.n_total = 0
        self._stop = threading.Event()
        self._thread = None

    @staticmethod
    def _git_sha():
        try:
            p = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                               capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=30)
            out = (p.stdout or "").strip()
        except (OSError, subprocess.SubprocessError) as e:
            raise CampaignError("git rev-parse HEAD failed: %s" % e)
        if p.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}", out):
            raise CampaignError("git rev-parse HEAD: %s"
                                % ((p.stderr or out or "no output").strip()))
        return out

    def append(self, key, obj):
        """One JSON line under the campaign's lock, flushed and fsynced."""
        with self.lock:
            path = os.path.join(self.dir, FILES[key])
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(obj, sort_keys=True, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())

    def append_row(self, row):
        """Refuse a test row and an invalid row BEFORE anything is written."""
        got = split.refuse_test(row["geometry_id"], self.split_mode)
        if got != row["split"]:
            raise CampaignError("row %s says split %r, the corpus says %r"
                                % (row["geometry_id"], row["split"], got))
        errs = explain.validate_row(row, self.gates, self.knobs)
        if errs:
            raise CampaignError("row %s/%d is invalid: %s"
                                % (row["geometry_id"], row["attempt"], errs[0]))
        self.append("attempts", row)
        with self.lock:
            self.n_rows += 1

    def append_record(self, gid, attempt, rec):
        errs = schema.errors(rec, "DecisionRecord")
        if errs:
            raise CampaignError("record %s/%s is invalid: %s" % (gid, attempt, errs[0]))
        self.append("records", {"schema": RECORD_LINE_SCHEMA,
                                "campaign_id": self.campaign_id,
                                "geometry_id": gid, "attempt": attempt,
                                "record": rec})

    def launch(self, argv, *, cwd, timeout_s, kind, gid, attempt, tag,
               expected=None, meta=None):
        """One child: peak working set sampled, timeout killed through its own
        Popen handle, one JOB_SCHEMA line, stdout/stderr read back for scoring."""
        jdir = os.path.join(self.dir, "jobs", gid)
        os.makedirs(jdir, exist_ok=True)
        so_path = os.path.join(jdir, tag + ".stdout")
        se_path = os.path.join(jdir, tag + ".stderr")
        t_start = schema._now_iso()
        t0 = time.perf_counter()
        timed_out = False
        peak = 0
        with open(so_path, "wb") as fo, open(se_path, "wb") as fe:
            proc = subprocess.Popen(argv, cwd=cwd, stdout=fo, stderr=fe)
            watcher = None
            ct = None
            try:
                watcher = psutil.Process(proc.pid)
                ct = watcher.create_time()
            except psutil.Error:
                pass
            with self.lock:
                self.live[proc.pid] = (proc, kind)
                self.launched.append({"pid": proc.pid, "kind": kind,
                                      "create_time": ct})
                if kind in self.MESHER_KINDS:
                    nmes = sum(1 for _, k in self.live.values()
                               if k in self.MESHER_KINDS)
                    self.max_live_mesher = max(self.max_live_mesher, nmes)
            try:
                while proc.poll() is None:
                    if watcher is not None:
                        try:
                            mi = watcher.memory_info()
                            peak = max(peak, getattr(mi, "peak_wset", None) or mi.rss)
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    if time.perf_counter() - t0 > timeout_s:
                        proc.kill()
                        proc.wait()
                        timed_out = True
                        break
                    time.sleep(0.2)
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
                with self.lock:
                    self.live.pop(proc.pid, None)
        seconds = time.perf_counter() - t0
        job = {"schema": JOB_SCHEMA, "campaign_id": self.campaign_id,
               "geometry_id": gid, "attempt": attempt, "kind": kind, "tag": tag,
               "pid": proc.pid,
               "exit_code": None if timed_out else proc.returncode,
               "timed_out": timed_out, "seconds": seconds,
               "peak_rss_mib": peak / 2 ** 20, "expected_mib": expected,
               "t_start": t_start, "t_end": schema._now_iso()}
        if meta:
            job.update(meta)
        self.append("jobs", job)
        out = dict(job)
        with open(so_path, "r", encoding="utf-8", errors="replace") as f:
            out["stdout"] = f.read()
        with open(se_path, "r", encoding="utf-8", errors="replace") as f:
            out["stderr"] = f.read()
        return out

    def sample(self):
        """Own RSS plus every live child, against the whole-system pressure."""
        with self.lock:
            procs = [p for p, _ in self.live.values()]
            nmes = sum(1 for _, k in self.live.values() if k in self.MESHER_KINDS)
        tot = psutil.Process().memory_info().rss
        for p in procs:
            try:
                tot += psutil.Process(p.pid).memory_info().rss
            except psutil.Error:
                pass
        rss = tot / 2 ** 20
        vm = psutil.virtual_memory()
        self.peak_rss_mib = max(self.peak_rss_mib, rss)
        self.peak_used_frac = max(self.peak_used_frac, vm.percent / 100.0)
        self.append("samples", {"schema": SAMPLE_SCHEMA, "t": schema._now_iso(),
                                "rss_mib": rss, "n_live": len(procs),
                                "n_mesher": nmes,
                                "system_used_frac": vm.percent / 100.0,
                                "available_mib": vm.available / 2 ** 20})
        return rss

    def _write_progress(self):
        with self.lock:
            nlive = len(self.live)
        rec = {"schema": "autonomy-campaign-progress/1",
               "campaign_id": self.campaign_id, "mode": self.mode,
               "system": self.system, "n_done": self.n_done,
               "n_total": self.n_total, "n_rows": self.n_rows, "n_live": nlive,
               "peak_rss_mib": self.peak_rss_mib, "t": schema._now_iso()}
        path = os.path.join(self.dir, FILES["progress"])
        with self.progress_lock:
            # One writer at a time (they shared one .tmp). A Windows reader
            # can hold progress.json open and fail os.replace with
            # PermissionError. progress.json is advisory: a dropped heartbeat
            # copy must never end a geometry as HARNESS-ERROR, so 20 tries.
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8", newline="\n") as f:
                json.dump(rec, f, indent=1, sort_keys=True, ensure_ascii=False)
                f.write("\n")
            for _ in range(20):
                try:
                    os.replace(tmp, path)
                    return
                except PermissionError:
                    time.sleep(0.05)

    def _loop(self):
        last = time.perf_counter()
        while not self._stop.wait(SAMPLE_S):
            rss = self.sample()
            if time.perf_counter() - last >= HEARTBEAT_S:
                last = time.perf_counter()
                with self.lock:
                    nlive = len(self.live)
                self.log("[campaign %s] heartbeat: %d/%d geometries, %d rows, "
                         "%d running, rss %.1f MiB, peak %.1f MiB"
                         % (self.campaign_id, self.n_done, self.n_total,
                            self.n_rows, nlive, rss, self.peak_rss_mib))
                self._write_progress()

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(10.0)
        self.sample()

    def kill_all(self):
        """Kill every live child through its own handle - never by name."""
        with self.lock:
            procs = [p for p, _ in self.live.values()]
        for p in procs:
            try:
                p.kill()
            except (OSError, psutil.Error):
                pass

    def orphans(self):
        """Launched pids still alive with the same create time."""
        out = []
        for e in self.launched:
            if e["create_time"] is None:
                continue
            try:
                if psutil.pid_exists(e["pid"]) and \
                        psutil.Process(e["pid"]).create_time() == e["create_time"]:
                    out.append(dict(e))
            except psutil.Error:
                continue
        return out

    def log(self, text):
        if not self.quiet:
            print(text, flush=True)


def observe(c, mrow):
    """docs/15 §C observe: regenerate, stl_repair (gated on after.closed),
    fingerprint.  Common to every mode - a baseline meshes the same surface."""
    gid = mrow["geometry_id"]
    raw_dir = os.path.join(c.dir, "stl", "raw")
    try:
        mod = importlib.import_module("inject" if mrow["family"] == "G"
                                      else rules.GEN_OF[mrow["family"]])
        os.makedirs(raw_dir, exist_ok=True)
        raw = mod.write_row(mrow, raw_dir)
    except (KeyError, ValueError, OSError) as e:
        return {"terminal": "HARNESS-ERROR", "reason": "the generator: %s" % e,
                "surface": None}
    out = os.path.join(c.dir, "stl", gid + ".stl")
    rep = os.path.join(c.dir, "stl", gid + "_repair.json")
    job = c.launch([sys.executable, STL_REPAIR, raw, "--out", out, "--json", rep],
                   cwd=c.dir, timeout_s=REPAIR_TIMEOUT_S, kind="repair", gid=gid,
                   attempt=0, tag="repair")
    report = None
    if job["exit_code"] == 0 and os.path.isfile(rep):
        with open(rep, encoding="utf-8") as f:
            report = json.load(f)
    if report is None:
        lines = [ln for ln in job["stderr"].splitlines() if ln.strip()]
        why = lines[-1] if lines else "no output"
        return {"terminal": "HARNESS-ERROR",
                "reason": "stl_repair failed (exit %s): %s" % (job["exit_code"], why),
                "surface": None}
    before = report["before"]
    after = report["after"]
    o = report.get("orientation") or {}
    surface = {"closed_before": bool(before["closed"]),
               "open_edges_before": before["open_edges"],
               "non_manifold_before": before["non_manifold_edges"],
               "closed_after": bool(after["closed"]),
               "open_edges_after": after["open_edges"],
               "non_manifold_after": after["non_manifold_edges"],
               "reoriented_triangles": o.get("reoriented_triangles", 0),
               "flipped_components": o.get("flipped_components", 0),
               "repaired": False}
    if surface["closed_before"] and not surface["reoriented_triangles"] \
            and not surface["flipped_components"]:
        shutil.copyfile(raw, out)
    elif surface["closed_after"]:
        surface["repaired"] = True
    else:
        return {"terminal": "SURFACE-OPEN",
                "reason": "stl_repair leaves the surface open: %d open edge(s), %d "
                          "non-manifold edge(s) after repair (docs/15 §C observe is "
                          "gated on after.closed)" % (surface["open_edges_after"],
                                                      surface["non_manifold_after"]),
                "surface": surface}
    with c.fp_lock:
        try:
            fp = rules._fp_of(out, gid)
        except Exception as e:
            return {"terminal": "HARNESS-ERROR", "reason": "features.py: %s" % e,
                    "surface": surface}
    errs = schema.errors(fp, "Fingerprint")
    if errs:
        return {"terminal": "HARNESS-ERROR",
                "reason": "features.py: the fingerprint is invalid: %s" % errs[0],
                "surface": surface}
    areas = {}
    for p in fp["patches"]:
        areas[p["name"]] = areas.get(p["name"], 0.0) + p["area_m2"]
    return {"terminal": None, "fingerprint": fp, "areas": areas, "surface": surface}


def write_config(c, gid, a, cfg):
    path = os.path.join(c.dir, config_rel(gid, a))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(cfg, f, indent=1, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def _last_error_line(text):
    lines = [ln for ln in text.splitlines() if ln.startswith("error:")]
    return lines[-1] if lines else None


def octree_probe(c, gctx, config, a):
    """-stopAfter octree through the runner: n_leaves is the cost signal."""
    gid = gctx["gid"]
    tag = "probe_a%d_%s" % (a, sha(config)[:8])
    pdir = os.path.join(c.dir, "probes", gid)
    os.makedirs(pdir, exist_ok=True)
    cfg = copy.deepcopy(config)
    cfg["output"] = {"case_dir": "probes/%s/%s" % (gid, tag), "name": tag}
    with open(os.path.join(pdir, tag + ".json"), "w", encoding="utf-8",
              newline="\n") as f:
        json.dump(cfg, f, indent=1, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    case = os.path.join(c.dir, "probes", gid, tag)
    try:
        job = c.launch([c.binary, "probes/%s/%s.json" % (gid, tag), "-stopAfter",
                        "octree"], cwd=c.dir, timeout_s=c.probe_timeout_s,
                       kind="octree", gid=gid, attempt=a, tag=tag)
        if job["exit_code"] != 0:
            return {"n_leaves": None, "max_non_orth_deg": None,
                    "exit_code": job["exit_code"],
                    "error": _last_error_line(job["stderr"])}
        with open(os.path.join(case, tag + "_summary.json"),
                  encoding="utf-8") as f:
            st = json.load(f)["stages"][0]
        return {"n_leaves": st["n_leaves"],
                "max_non_orth_deg": st.get("max_non_orth_deg"), "exit_code": 0}
    finally:
        shutil.rmtree(case, ignore_errors=True)


def snap_probe(c, gctx, config, a):
    """-stopAfter snap, then the shortest wall edge over the layer patches."""
    gid = gctx["gid"]
    tag = "snap_a%d_%s" % (a, sha(config)[:8])
    pdir = os.path.join(c.dir, "probes", gid)
    os.makedirs(pdir, exist_ok=True)
    cfg = copy.deepcopy(config)
    cfg["output"] = {"case_dir": "probes/%s/%s" % (gid, tag), "name": tag}
    with open(os.path.join(pdir, tag + ".json"), "w", encoding="utf-8",
              newline="\n") as f:
        json.dump(cfg, f, indent=1, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    case = os.path.join(c.dir, "probes", gid, tag)
    try:
        job = c.launch([c.binary, "probes/%s/%s.json" % (gid, tag), "-stopAfter",
                        "snap"], cwd=c.dir, timeout_s=c.probe_timeout_s,
                       kind="snap", gid=gid, attempt=a, tag=tag)
        if job["exit_code"] != 0:
            return None
        return preflight.wall_edge_min(case, config["layers"]["patches"])
    finally:
        shutil.rmtree(case, ignore_errors=True)


def run_attempt(c, gctx, config, a, n_leaves):
    """One mesher run plus its 10 % audit (re-gate and a rerun for the same
    content hash); the polyMesh is deleted after scoring."""
    gid = gctx["gid"]
    rel = config_rel(gid, a)
    tag = "a%d" % a
    case = os.path.join(c.dir, "cases", "%s_%s" % (gid, tag))
    exp = expected_mib(n_leaves)
    meta = {"wall_level": remedies.wall_level(config), "n_leaves": n_leaves}
    c.ram.acquire(exp)
    try:
        t_start = schema._now_iso()
        job = c.launch([c.binary, rel, "-tag", tag, "-runId", c.campaign_id],
                       cwd=c.dir, timeout_s=c.timeout_s, kind="full", gid=gid,
                       attempt=a, tag=tag, expected=exp, meta=meta)
        t_end = schema._now_iso()
        summary = None
        spath = os.path.join(case, "%s_%s_summary.json" % (gid, tag))
        if job["exit_code"] == 0 and os.path.isfile(spath):
            with open(spath, encoding="utf-8") as f:
                summary = json.load(f)
        audited = job["exit_code"] == 0 and audit_selected(sha(config), c.audit_mod)
        check_exit = None
        sha2 = None
        rerun_exit = None
        if audited:
            chk = c.launch([c.binary, rel, "-tag", tag, "-check"], cwd=c.dir,
                           timeout_s=c.timeout_s, kind="check", gid=gid, attempt=a,
                           tag=tag + "_check")
            check_exit = 1 if chk["timed_out"] else chk["exit_code"]
            rcase = os.path.join(c.dir, "cases", "%s_%sr" % (gid, tag))
            # the rerun runs under the first job's reservation, still held and
            # now free (the first job has ended): a second acquire here would
            # wait on itself whenever the job is more than half the budget
            rjob = c.launch([c.binary, rel, "-tag", tag + "r", "-runId",
                             c.campaign_id], cwd=c.dir, timeout_s=c.timeout_s,
                            kind="rerun", gid=gid, attempt=a, tag=tag + "r",
                            expected=exp, meta=meta)
            rerun_exit = rjob["exit_code"]
            if rjob["exit_code"] == 0:
                sha2 = score.content_sha256(rcase)
            shutil.rmtree(rcase, ignore_errors=True)
        sc = score.score_run(exit_code=job["exit_code"], stdout=job["stdout"],
                             stderr=job["stderr"], summary=summary, config=config,
                             patch_areas_m2=gctx["areas"], flow=gctx["flow"],
                             timed_out=job["timed_out"], wall_seconds=job["seconds"],
                             check_exit=check_exit,
                             case_dir=case if job["exit_code"] == 0 else None,
                             gates=c.gates)
        if audited:
            gctx["audit"].append({"attempt": a, "check_exit": check_exit,
                                  "content_sha256": sc["content_sha256"],
                                  "rerun_content_sha256": sha2,
                                  "rerun_exit": rerun_exit,
                                  "equal": sha2 is not None
                                  and sha2 == sc["content_sha256"]})
        shutil.rmtree(os.path.join(case, "constant", "polyMesh"), ignore_errors=True)
        shutil.rmtree(os.path.join(case, "mesh"), ignore_errors=True)
        return {"outcome": sc["outcome"], "content_sha256": sc["content_sha256"],
                "t_start": t_start, "t_end": t_end}
    finally:
        c.ram.release(exp)


def run_preflight(c, gctx, cand, a):
    """The L0 checks with the octree probe; PF-THIN is re-checked with the
    measured post-snap wall edge (docs/15 §K G-PREFLIGHT part 3)."""
    probe = c.probe_fn(c, gctx, cand, a)
    pf = preflight.preflight(cand, fingerprint=gctx["fp"], flow=gctx["flow"],
                             octree_probe=probe, gates=c.gates, knobs=c.knobs,
                             cwd=c.dir)
    h = None
    if pf["refused"] == ["PF-THIN"] and remedies._requested(cand):
        h = c.snap_fn(c, gctx, cand, a)
        if h is not None:
            pf = preflight.preflight(cand, fingerprint=gctx["fp"], flow=gctx["flow"],
                                     octree_probe=probe, h_wall_min_m=h,
                                     gates=c.gates, knobs=c.knobs, cwd=c.dir)
    gctx["vetoes"].append({"attempt": a, "config_sha256": sha(cand),
                           "refused": list(pf["refused"]),
                           "n_leaves": probe.get("n_leaves"), "h_wall_min_m": h})
    gctx["leaves"][sha(cand)] = probe.get("n_leaves")
    return pf


def make_veto(c, gctx, a_next):
    """The veto remedies.propose consults: preflight on each candidate."""
    def veto(cand):
        pf = run_preflight(c, gctx, cand, a_next)
        if pf["refused"]:
            gctx["pending_refusals"].extend(
                r for r in pf["records"] if r["verdict"] == "refuse")
        else:
            gctx["pending_pass"] = list(pf["records"])
        return pf["refused"]
    return veto


def _check_hook(res, layer, before, knobs):
    if not isinstance(res, dict) or res.get("verdict") not in ("apply", "abstain"):
        raise CampaignError("the %s hook returned verdict %r, not apply/abstain"
                            % (layer, (res or {}).get("verdict")))
    rec = res.get("record")
    errs = schema.errors(rec, "DecisionRecord")
    if errs:
        raise CampaignError("the %s hook's record is invalid: %s" % (layer, errs[0]))
    if rec["layer"] != layer:
        raise CampaignError("the %s hook's record is layer %r" % (layer, rec["layer"]))
    if res["verdict"] != "apply":
        return
    cfg = res.get("config")
    if not isinstance(cfg, dict) or sha(cfg) == sha(before):
        raise CampaignError("the %s hook applied without changing the config" % layer)
    if res.get("edits") != rules.diff_edits(before, cfg):
        raise CampaignError("the %s hook's edits do not match its config diff" % layer)
    for e in res["edits"]:
        ref = schema.check_edit(e["pointer"], e["to"], knobs)
        if ref is not None:
            raise CampaignError("the %s hook's edit %s is refused: %s"
                                % (layer, e["pointer"], ref["message"]))
    if layer == "optimiser":
        pred = res.get("prediction")
        need = ("p_fail", "p_fail_std", "blc8_a_priori", "log_cells", "t_predicted")
        if not isinstance(pred, dict) or any(k not in pred for k in need):
            raise CampaignError("the optimiser hook's prediction needs %s" % (need,))


def _row(c, gctx, a, cfg, d, refusals, res):
    return {"schema": "autonomy-attempt/1", "campaign_id": c.campaign_id,
            "split": gctx["split"], "geometry_id": gctx["gid"],
            "fingerprint": gctx["fp"], "attempt": a,
            "stage_focus": d["stage_focus"], "config_sha": sha(cfg),
            "config_delta": d["config_delta"], "decided_by": d["decided_by"],
            "rule_id": d["rule_id"], "trigger": d["trigger"],
            "prediction": d["prediction"], "constraint_refusals": list(refusals),
            "outcome": res["outcome"], "content_sha256": res["content_sha256"],
            "binary_sha": c.binary_sha, "git_sha": c.git_sha,
            "t_start": res["t_start"], "t_end": res["t_end"]}


def _attempt_line(c, gctx, row):
    oc = row["outcome"]
    return ("[campaign %s] %s a%d %s %s exit %s %s cells %s %.1f s"
            % (c.campaign_id, gctx["gid"], row["attempt"], row["decided_by"],
               row["rule_id"] or "-", oc["exit_code"], oc["failure_class"] or "pass",
               oc["n_cells"], oc["seconds"] if oc["seconds"] is not None else 0.0))


def _wait_prediction(t_predicted):
    """A prediction must precede its run's second (schema.check_attempt)."""
    deadline = time.perf_counter() + 5.0
    while schema._now_iso() <= t_predicted:
        if time.perf_counter() > deadline:
            return
        time.sleep(0.2)


def _end(c, gctx, terminal, reason=None, refused=(), capability_limited=()):
    if terminal not in GEOMETRY_TERMINALS:
        raise CampaignError("terminal %r is not one of %s"
                            % (terminal, ", ".join(GEOMETRY_TERMINALS)))
    rows = gctx["rows"]
    rec = {"schema": GEOMETRY_SCHEMA, "campaign_id": c.campaign_id,
           "split": gctx["split"], "geometry_id": gctx["gid"],
           "family": gctx["mrow"]["family"], "stratum": gctx["mrow"]["stratum"],
           "mode": c.mode, "system": c.system, "ablate": sorted(c.ablate),
           "terminal": terminal, "reason": reason, "refused": list(refused),
           "capability_limited": list(capability_limited), "attempts": len(rows),
           "final_attempt": None, "failure": True, "strict_failure": True,
           "blc8_a_priori": 0.0, "blc_full_a_priori": 0.0, "n_cells": None,
           "lhs_fail_frac": None, "surface": gctx["surface"],
           "fingerprint": gctx["fp"], "vetoes": gctx["vetoes"],
           "audit": gctx["audit"],
           "records": [{"attempt": a, "record": r} for a, r in gctx["norow"]],
           "t_start": gctx["t_start"], "t_end": schema._now_iso(),
           "seconds": time.perf_counter() - gctx["t0"]}
    if rows:
        if c.system == "b0-lhs":
            fails = [bool(r["outcome"]["failure"]) for r in rows]
            rec["failure"] = all(fails)
            rec["strict_failure"] = all(bool(r["outcome"]["strict_failure"])
                                        for r in rows)
            final = next((r for r in rows if not r["outcome"]["failure"]
                          and not r["outcome"]["strict_failure"]), None)
            if final is None:
                final = next((r for r in rows if not r["outcome"]["failure"]), rows[0])
            rec["lhs_fail_frac"] = sum(fails) / float(len(rows))
        else:
            final = rows[-1]
            rec["failure"] = bool(final["outcome"]["failure"])
            rec["strict_failure"] = bool(final["outcome"]["strict_failure"])
        if terminal == "HARNESS-ERROR":
            rec["failure"] = True
            rec["strict_failure"] = True
        rec["final_attempt"] = final["attempt"]
        rec["blc8_a_priori"] = final["outcome"]["blc8_a_priori"]
        rec["blc_full_a_priori"] = final["outcome"]["blc_full_a_priori"]
        rec["n_cells"] = final["outcome"]["n_cells"]
    c.append("geometries", rec)
    with c.lock:
        c.n_done += 1
    c._write_progress()
    with c.lock:
        procs = [p for p, _ in c.live.values()]
    rss = psutil.Process().memory_info().rss / 2 ** 20
    for p in procs:
        try:
            rss += psutil.Process(p.pid).memory_info().rss / 2 ** 20
        except psutil.Error:
            pass
    c.log("[campaign %s] %d/%d %s %s attempts %d failure %s %.1f s | live %d, "
          "rss %.1f MiB (%.1f %% of RAM)"
          % (c.campaign_id, c.n_done, c.n_total, gctx["gid"], terminal, len(rows),
             rec["failure"], rec["seconds"], len(procs), rss,
             rss / c.total_mib * 100.0))


def run_geometry(c, mrow):
    """One geometry, observe to end record; the rows already written stay."""
    gid = mrow["geometry_id"]
    gctx = {"gid": gid, "mrow": mrow, "split": split.refuse_test(gid, c.split_mode),
            "fp": None, "areas": None, "flow": mrow["flow"], "vetoes": [],
            "audit": [], "leaves": {}, "pending_refusals": [], "pending_pass": [],
            "norow": [], "rows": [], "t_start": schema._now_iso(),
            "t0": time.perf_counter(), "surface": None}
    try:
        obs = observe(c, mrow)
        gctx["surface"] = obs["surface"]
        if obs["terminal"] is not None:
            _end(c, gctx, obs["terminal"], reason=obs["reason"])
            return
        gctx["fp"] = obs["fingerprint"]
        gctx["areas"] = obs["areas"]
        if c.system in BASELINES:
            _run_baseline(c, gctx, mrow)
        else:
            _run_system(c, gctx, mrow)
    except split.SplitSealed:
        raise
    except Exception as e:
        _end(c, gctx, "HARNESS-ERROR", reason="%s: %s" % (type(e).__name__, e))


def _run_baseline(c, gctx, mrow):
    t = b0_template(mrow, gctx["fp"])
    cfgs = [t] if c.system == "b0-template" else b0_lhs(mrow, gctx["fp"])
    for a, cfg in enumerate(cfgs, 1):
        probe = c.probe_fn(c, gctx, cfg, a)
        write_config(c, gctx["gid"], a, cfg)
        res = c.attempt_fn(c, gctx, cfg, a, probe.get("n_leaves"))
        d = {"decided_by": "default", "rule_id": None, "trigger": None,
             "config_delta": [] if a == 1 and c.system == "b0-template"
             else rules.diff_edits(t, cfg), "stage_focus": None, "prediction": None}
        row = _row(c, gctx, a, cfg, d, [], res)
        c.append_row(row)
        gctx["rows"].append(row)
        c.log(_attempt_line(c, gctx, row))
    _end(c, gctx, "BASELINE")


def _run_system(c, gctx, mrow):
    gid = gctx["gid"]
    k = 1 if "remedies" in c.ablate else c.gates["attempts_k"]
    setup = rules.setup(mrow, gctx["fp"], stl_rel(gid), case_rel(gid), gid,
                        gates=c.gates, knobs=c.knobs)
    recs = [(1, r) for r in setup["records"]]
    if setup["refused"]:
        gctx["norow"] = list(recs)
        _end(c, gctx, "REFUSED", refused=setup["refused"],
             reason="rules.setup refuses: " + ", ".join(setup["refused"]))
        return
    cfg = setup["config"]
    last = [r for r in setup["records"] if r["verdict"] == "apply" and r["edits"]][-1]
    d = {"decided_by": "rule", "rule_id": last["rule_id"], "trigger": last["trigger"],
         "config_delta": [], "stage_focus": None, "prediction": None}
    refusals = []
    ctx0 = {"geometry_id": gid, "fingerprint": gctx["fp"], "flow": gctx["flow"],
            "win_level": setup["summary"]["win_level"]}
    just_pf = False
    if "prior" in c.layers:
        pr = c.hooks["prior"](dict(ctx0, config=cfg, rules=setup))
        _check_hook(pr, "prior", cfg, c.knobs)
        recs.append((1, pr["record"]))
        if pr["verdict"] == "apply":
            if "preflight" in c.layers:
                pf = run_preflight(c, gctx, pr["config"], 1)
                if pf["refused"]:
                    refusals.extend(r for r in pf["records"]
                                    if r["verdict"] == "refuse")
                else:
                    cfg = pr["config"]
                    d = {"decided_by": "prior",
                         "rule_id": pr["record"]["rule_id"],
                         "trigger": pr["record"]["trigger"],
                         "config_delta": pr["edits"], "stage_focus": None,
                         "prediction": None}
                    recs.extend((1, r) for r in pf["records"])
                    just_pf = True
            else:
                cfg = pr["config"]
                d = {"decided_by": "prior", "rule_id": pr["record"]["rule_id"],
                     "trigger": pr["record"]["trigger"], "config_delta": pr["edits"],
                     "stage_focus": None, "prediction": None}
    if not just_pf:
        pf = run_preflight(c, gctx, cfg, 1)
        if pf["refused"] and "preflight" in c.layers:
            gctx["norow"] = list(recs) + [(1, r) for r in pf["records"]]
            _end(c, gctx, "REFUSED", refused=pf["refused"],
                 reason="preflight refuses attempt 1: " + ", ".join(pf["refused"]))
            return
        if pf["refused"]:
            refusals.extend(r for r in pf["records"] if r["verdict"] == "refuse")
            recs.extend((1, r) for r in pf["records"] if r["verdict"] != "refuse")
        else:
            recs.extend((1, r) for r in pf["records"])
    for a1, r in recs:
        c.append_record(gid, a1, r)
    history = []
    a = 1
    while True:
        if d["prediction"] is not None:
            _wait_prediction(d["prediction"]["t_predicted"])
        write_config(c, gid, a, cfg)
        res = c.attempt_fn(c, gctx, cfg, a, gctx["leaves"].get(sha(cfg)))
        row = _row(c, gctx, a, cfg, d, refusals, res)
        history.append({"attempt": a, "config_sha256": sha(cfg),
                        "rule_id": d["rule_id"] if d["decided_by"] == "remedy"
                        else None})
        ctx = dict(ctx0, config=cfg, outcome=res["outcome"])
        gctx["pending_refusals"] = []
        gctx["pending_pass"] = []
        veto = make_veto(c, gctx, a + 1) if "preflight" in c.layers else None
        prop = remedies.propose(ctx, history, k=k, gates=c.gates, knobs=c.knobs,
                                veto=veto)
        nxt = None
        op = None
        rm_written = False
        if prop["verdict"] == "apply":
            nxt = ("remedy", prop)
        elif "optimiser" in c.layers and prop["terminal"] == "EXHAUSTED" \
                and len(history) < k:
            c.append_record(gid, a, prop["record"])
            rm_written = True
            op = c.hooks["optimiser"](ctx, history)
            _check_hook(op, "optimiser", cfg, c.knobs)
            if op["verdict"] == "apply" and (veto is None or veto(op["config"]) == []):
                nxt = ("optimiser", op)
        if nxt is None:
            names = list(prop["capability_limited"])
            for p in row["outcome"]["patches"]:
                if p["name"] in names:
                    p["capability_limited"] = True
            c.append_row(row)
            gctx["rows"].append(row)
            c.log(_attempt_line(c, gctx, row))
            if not rm_written:
                c.append_record(gid, a, prop["record"])
            if op is not None:
                c.append_record(gid, a, op["record"])
            gctx["norow"].extend((a + 1, r) for r in gctx["pending_refusals"])
            _end(c, gctx, prop["terminal"], capability_limited=names,
                 reason=prop["record"]["message"])
            return
        c.append_row(row)
        gctx["rows"].append(row)
        c.log(_attempt_line(c, gctx, row))
        kind, nres = nxt
        c.append_record(gid, a + 1, nres["record"])
        for r in gctx["pending_pass"]:
            c.append_record(gid, a + 1, r)
        cfg = nres["config"]
        d = {"decided_by": kind, "rule_id": nres["record"]["rule_id"],
             "trigger": nres["record"]["trigger"], "config_delta": nres["edits"],
             "stage_focus": prop["stage_focus"] if kind == "remedy" else None,
             "prediction": nres.get("prediction") if kind == "optimiser" else None}
        refusals = list(gctx["pending_refusals"])
        a += 1


_RUN_OPTS = {"system": None, "ablate": (), "run_id": None, "tag": None,
             "streams": MAX_STREAMS, "timeout_s": TIMEOUT_S,
             "probe_timeout_s": PROBE_TIMEOUT_S, "audit_mod": AUDIT_MOD,
             "ids": None, "limit": None, "binary": BINARY_DEFAULT,
             "resume": False, "quiet": False}


def _check_opts(o):
    if o["mode"] not in MODES:
        raise CampaignError("mode: %r is not one of %s" % (o["mode"], ", ".join(MODES)))
    if o["mode"] == "evaluate":
        if o["system"] not in SYSTEMS:
            raise CampaignError("--system: mode evaluate needs one of %s"
                                % ", ".join(SYSTEMS))
    else:
        if o["system"] is not None:
            raise CampaignError("--system: only mode evaluate takes a system "
                                "(%r given)" % (o["system"],))
        o["system"] = o["mode"]
    for item in o["ablate"]:
        if item not in ABLATABLE:
            raise CampaignError("ablate: %r is not one of %s"
                                % (item, ", ".join(ABLATABLE)))
    if o["ablate"] and o["system"] in BASELINES:
        raise CampaignError("baseline: no ablation on a baseline system")
    if not isinstance(o["streams"], int) or isinstance(o["streams"], bool) \
            or not 1 <= o["streams"] <= MAX_STREAMS:
        raise CampaignError("streams: %r outside 1..%d (the house cap while the "
                            "solver workflow runs; docs/15 §C says 12)"
                            % (o["streams"], MAX_STREAMS))
    if not isinstance(o["audit_mod"], int) or isinstance(o["audit_mod"], bool) \
            or o["audit_mod"] < 1:
        raise CampaignError("audit_mod: %r is not an int >= 1" % (o["audit_mod"],))
    if not o["timeout_s"] > 0 or not o["probe_timeout_s"] > 0:
        raise CampaignError("timeout: the timeouts must be positive")
    o["run_id"] = o["run_id"] or os.path.basename(os.path.normpath(o["out"]))
    if not RUN_ID_RE.fullmatch(o["run_id"]):
        raise CampaignError("run id: %r does not match %s"
                            % (o["run_id"], RUN_ID_RE.pattern))


def run_campaign(opts, *, attempt_fn=None, probe_fn=None, snap_fn=None, hooks=None):
    """The whole campaign; the END record out (docs/15 §C act/verify/remember)."""
    o = dict(_RUN_OPTS)
    o.update(opts)
    for key in ("manifest", "mode", "out"):
        if not o.get(key):
            raise CampaignError("%s: this option is required" % key)
    _check_opts(o)
    split_mode = split.EVALUATE if o["mode"] == "evaluate" else o["mode"]
    rows = load_manifest(o["manifest"], split_mode, ids=o["ids"], limit=o["limit"])
    hook_fns = {}
    for layer in LAYERS[o["system"]]:
        if layer in HOOKS:
            fn = (hooks or {}).get(layer)
            if fn is None:
                mod_name, fn_name = HOOKS[layer][0], HOOKS[layer][1]
                try:
                    fn = getattr(importlib.import_module(mod_name), fn_name)
                except (ImportError, AttributeError):
                    raise CampaignError("mode %s needs %s.py (%s), which this tree "
                                        "does not have" % (o["mode"], mod_name,
                                                           fn_name))
            hook_fns[layer] = fn
    if not os.path.isfile(o["binary"]):
        raise CampaignError("binary: %s is not a file" % o["binary"])
    gates = schema.load_gates()
    knobs = schema.load_knobs()
    vm = psutil.virtual_memory()
    header = {"schema": CAMPAIGN_SCHEMA, "campaign_id": o["run_id"],
              "tag": o["tag"], "mode": o["mode"], "system": o["system"],
              "ablate": sorted(o["ablate"]), "split_mode": split_mode,
              "manifest": {"source": o["manifest"] if o["manifest"] in ("tuning",
                           "test") else os.path.abspath(o["manifest"]).replace(
                               os.sep, "/"),
                           "sha256": manifest_sha(o["manifest"]), "n": len(rows)},
              "geometry_ids": [r["geometry_id"] for r in rows],
              "binary": os.path.basename(o["binary"]),
              "binary_sha256": _sha256_of_file(o["binary"]),
              "git_sha": Campaign._git_sha(), "streams": o["streams"],
              "timeout_s": o["timeout_s"], "probe_timeout_s": o["probe_timeout_s"],
              "audit_mod": o["audit_mod"], "gates_sha256": sha(gates),
              "knobs_sha256": sha(knobs), "ram_total_mib": vm.total / 2 ** 20,
              "ram_budget_mib": RAM_FRACTION * vm.total / 2 ** 20 - 512.0,
              "max_streams": MAX_STREAMS, "t_start": schema._now_iso()}
    out = o["out"]
    ended = set()
    if os.path.isdir(out) and os.listdir(out):
        if not o["resume"]:
            raise CampaignError("%s is not empty (pass --resume to continue it)" % out)
        cpath = os.path.join(out, FILES["campaign"])
        if not os.path.isfile(cpath):
            raise CampaignError("resume: no campaign.json in %s" % out)
        with open(cpath, encoding="utf-8") as f:
            old = json.load(f)
        for field, want in (("campaign_id", o["run_id"]), ("mode", o["mode"]),
                            ("system", o["system"]), ("ablate", sorted(o["ablate"])),
                            ("split_mode", split_mode), ("manifest", header["manifest"]),
                            ("geometry_ids", header["geometry_ids"]),
                            ("binary_sha256", header["binary_sha256"]),
                            ("git_sha", header["git_sha"]),
                            ("audit_mod", o["audit_mod"])):
            if old.get(field) != want:
                raise CampaignError("resume: campaign.json %s differs" % field)
        ended = {g["geometry_id"] for g in load_geometries(out)}
        partial = sorted({r["geometry_id"] for r in load_rows(out)} - ended)
        if partial:
            raise CampaignError("resume: geometries with rows but no end record: %s "
                                "(start a fresh --out)" % ", ".join(partial))
    else:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, FILES["campaign"]), "w", encoding="utf-8",
                  newline="\n") as f:
            json.dump(header, f, indent=1, sort_keys=True, ensure_ascii=False)
            f.write("\n")
    todo = [r for r in rows if r["geometry_id"] not in ended]
    c = Campaign(out, campaign_id=o["run_id"], mode=o["mode"], system=o["system"],
                 ablate=o["ablate"], split_mode=split_mode, binary=o["binary"],
                 streams=o["streams"], timeout_s=o["timeout_s"],
                 probe_timeout_s=o["probe_timeout_s"], audit_mod=o["audit_mod"],
                 attempt_fn=attempt_fn, probe_fn=probe_fn, snap_fn=snap_fn,
                 hooks=hook_fns, gates=gates, knobs=knobs, quiet=o["quiet"])
    c.n_total = len(rows)
    c.n_done = len(ended)
    t0 = time.perf_counter()
    c.start()
    try:
        with ThreadPoolExecutor(max_workers=o["streams"]) as ex:
            list(ex.map(lambda r: run_geometry(c, r), todo))
    finally:
        c.kill_all()
        c.stop()
    wall = time.perf_counter() - t0
    rows_all = load_rows(out)
    geoms = load_geometries(out)
    terminals = {}
    for g in geoms:
        terminals[g["terminal"]] = terminals.get(g["terminal"], 0) + 1
    end = {"schema": END_SCHEMA, "campaign_id": o["run_id"],
           "n_geometries": len(rows), "n_rows": len(rows_all),
           "terminals": terminals, "harness_errors": terminals.get("HARNESS-ERROR", 0),
           "wall_seconds": wall, "peak_rss_mib": c.peak_rss_mib,
           "peak_frac": c.peak_rss_mib / c.total_mib,
           "peak_used_frac": c.peak_used_frac, "ram_total_mib": c.total_mib,
           "max_live_mesher": c.max_live_mesher, "streams": o["streams"],
           "orphans": c.orphans(), "t_end": schema._now_iso()}
    with open(os.path.join(out, FILES["end"]), "w", encoding="utf-8",
              newline="\n") as f:
        json.dump(end, f, indent=1, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    recs = {}
    for line in _read_jsonl(os.path.join(out, FILES["records"])):
        recs.setdefault(line["geometry_id"], []).append(
            {"attempt": line["attempt"], "record": line["record"]})
    _dump_json(os.path.join(out, FILES["records_json"]),
               {"schema": "autonomy-explain-records/1", "records": recs})
    _dump_json(os.path.join(out, FILES["summary"]), summarise_campaign(out))
    c.log("[campaign %s] done: %d geometries, %d rows, terminals %s, peak rss %.1f "
          "MiB (%.1f %% of RAM), max %d mesher processes, orphans %d, %.1f s"
          % (o["run_id"], len(rows), len(rows_all), json.dumps(terminals,
             sort_keys=True), c.peak_rss_mib, c.peak_rss_mib / c.total_mib * 100.0,
             c.max_live_mesher, len(end["orphans"]), wall))
    return end


def _dump_json(path, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def load_rows(cdir):
    return _read_jsonl(os.path.join(cdir, FILES["attempts"]))


def load_geometries(cdir):
    return _read_jsonl(os.path.join(cdir, FILES["geometries"]))


def load_records(cdir):
    """{gid: [{"attempt", "record"}, ...]} from records.jsonl, in file order."""
    out = {}
    for line in _read_jsonl(os.path.join(cdir, FILES["records"])):
        out.setdefault(line["geometry_id"], []).append(
            {"attempt": line["attempt"], "record": line["record"]})
    return out


def summarise_campaign(cdir):
    """The campaign summary from the end records alone (no manifest is read)."""
    geoms = load_geometries(cdir)
    rows = load_rows(cdir)
    with open(os.path.join(cdir, FILES["campaign"]), encoding="utf-8") as f:
        header = json.load(f)
    keys = []
    for g in geoms:
        pair = (g["family"], g["stratum"])
        if pair not in keys:
            keys.append(pair)
    fams = sorted({f for f, _ in keys})
    ordered = sorted(keys) + [(f, "all") for f in fams] + [("all", "all")]

    def members(family, stratum):
        return [g for g in geoms
                if (family == "all" or g["family"] == family)
                and (stratum == "all" or g["stratum"] == stratum)]

    groups = []
    for family, stratum in ordered:
        ms = members(family, stratum)
        if not ms:
            continue
        n = len(ms)
        fails = sum(1 for g in ms if g["failure"])
        stricts = sum(1 for g in ms if g["strict_failure"])
        cells = sorted(g["n_cells"] for g in ms if g["n_cells"] is not None)
        lhs = [g["lhs_fail_frac"] for g in ms if g["lhs_fail_frac"] is not None]
        terms = {}
        for g in ms:
            terms[g["terminal"]] = terms.get(g["terminal"], 0) + 1
        groups.append({"family": family, "stratum": stratum, "n": n, "fail": fails,
                       "mfr": fails / n,
                       "mfr_ci": list(explain.clopper_pearson(fails, n)),
                       "strict": stricts, "strict_rate": stricts / n,
                       "strict_ci": list(explain.clopper_pearson(stricts, n)),
                       "blc8_mean": sum(g["blc8_a_priori"] for g in ms) / n,
                       "blc_full_mean": sum(g["blc_full_a_priori"] for g in ms) / n,
                       "cells_median": statistics.median(cells) if cells else None,
                       "attempts_mean": sum(g["attempts"] for g in ms) / n,
                       "terminals": terms,
                       "capability_limited": sum(1 for g in ms
                                                 if g["capability_limited"]),
                       "lhs_fail_frac_mean": sum(lhs) / len(lhs) if lhs else None})
    audit = None
    if rows:
        a = explain.audit(rows, load_records(cdir))
        audit = {"ok": a["ok"], "n_rows": a["n_rows"], "n_valid": a["n_valid"],
                 "untemplated": a["untemplated"],
                 "record_order_bad": len(a["record_order"]["bad"]),
                 "trigger_mismatch": len(a["trigger_mismatch"]),
                 "time_reversed": len(a["time_reversed"]),
                 "prediction_late": len(a["prediction_late"])}
    sample = {"n": 0, "equal": 0, "check_failed": 0}
    for g in geoms:
        for e in g["audit"]:
            sample["n"] += 1
            if e["equal"]:
                sample["equal"] += 1
            if e["check_exit"] != 0:
                sample["check_failed"] += 1
    return {"schema": SUMMARY_SCHEMA, "campaign_id": header["campaign_id"],
            "mode": header["mode"], "system": header["system"],
            "ablate": header["ablate"], "n_geometries": len(geoms),
            "n_rows": len(rows), "terminals": {t: sum(1 for g in geoms
                                                      if g["terminal"] == t)
                                               for t in sorted({g["terminal"]
                                                                for g in geoms})},
            "groups": groups, "audit": audit, "audit_sample": sample}


def summary_text(s):
    lines = ["campaign %s (%s/%s): %d geometries, %d rows, terminals %s"
             % (s["campaign_id"], s["mode"], s["system"], s["n_geometries"],
                s["n_rows"], json.dumps(s["terminals"], sort_keys=True))]
    for g in s["groups"]:
        cells = "None" if g["cells_median"] is None else "%.0f" % g["cells_median"]
        lines.append("%s %s n %d MFR %.3f [%.3f, %.3f] strict %.3f [%.3f, %.3f] "
                     "BLC_8 %.3f cells %s attempts %.2f"
                     % (g["family"], g["stratum"], g["n"], g["mfr"],
                        g["mfr_ci"][0] or 0.0, g["mfr_ci"][1] or 0.0,
                        g["strict_rate"], g["strict_ci"][0] or 0.0,
                        g["strict_ci"][1] or 0.0, g["blc8_mean"], cells,
                        g["attempts_mean"]))
    return "\n".join(lines)


class _ReplayStop(Exception):
    """One geometry's replay is done (a mismatch was recorded)."""


def replay(cdir, hooks=None):
    """docs/15 §F G-DET: the rows through rules.py and remedies.py again."""
    with open(os.path.join(cdir, FILES["campaign"]), encoding="utf-8") as f:
        header = json.load(f)
    by_gid = {}
    for r in load_rows(cdir):
        by_gid.setdefault(r["geometry_id"], {})[r["attempt"]] = r
    geoms = {g["geometry_id"]: g for g in load_geometries(cdir)}
    mrows = {r["geometry_id"]: r for r in
             load_manifest(header["manifest"]["source"], header["split_mode"],
                           ids=header["geometry_ids"])}
    gates = schema.load_gates()
    knobs = schema.load_knobs()
    layers = tuple(l for l in LAYERS[header["system"]] if l not in header["ablate"])
    k = 1 if "remedies" in header["ablate"] else gates["attempts_k"]
    st = {"decisions": 0, "hook_decisions": 0, "mismatches": []}

    def bad(gid, attempt, why):
        st["mismatches"].append({"geometry_id": gid, "attempt": attempt,
                                 "why": why})
        raise _ReplayStop()

    def read_cfg(gid, a):
        with open(os.path.join(cdir, config_rel(gid, a)), encoding="utf-8") as f:
            return json.load(f)

    for gid in header["geometry_ids"]:
        try:
            _replay_one(gid, header, layers, k, by_gid.get(gid, {}), geoms.get(gid),
                        mrows[gid], cdir, gates, knobs, st, bad, read_cfg)
        except _ReplayStop:
            continue
    return {"schema": REPLAY_SCHEMA, "campaign_id": header["campaign_id"],
            "geometries": len(header["geometry_ids"]), "decisions": st["decisions"],
            "hook_decisions": st["hook_decisions"],
            "mismatches": st["mismatches"], "ok": not st["mismatches"]}


def _replay_one(gid, header, layers, k, rows_by, end, mrow, cdir, gates, knobs,
                st, bad, read_cfg):
    if end is None:
        bad(gid, 0, "no end record")
    ordered = [rows_by[a] for a in sorted(rows_by)]
    for r in ordered:
        rel = config_rel(gid, r["attempt"])
        if not os.path.isfile(os.path.join(cdir, rel)):
            bad(gid, r["attempt"], "config file %s is missing" % rel)
        if sha(read_cfg(gid, r["attempt"])) != r["config_sha"]:
            bad(gid, r["attempt"], "config file %s differs from the row" % rel)
    if end["terminal"] in ("SURFACE-OPEN", "HARNESS-ERROR"):
        if ordered:
            bad(gid, 0, "a %s geometry carries %d rows"
                % (end["terminal"], len(ordered)))
        return
    fp = end["fingerprint"]
    if header["system"] in BASELINES:
        t = b0_template(mrow, fp)
        want = [t] if header["system"] == "b0-template" else b0_lhs(mrow, fp)
        if len(ordered) != len(want):
            bad(gid, 0, "%d rows for %d baseline configs"
                % (len(ordered), len(want)))
        for r, cfg in zip(ordered, want):
            if r["config_sha"] != sha(cfg):
                bad(gid, r["attempt"], "row %d carries %s, the rebuilt baseline is %s"
                    % (r["attempt"], r["config_sha"][:12], sha(cfg)[:12]))
            st["decisions"] += 1
        return
    setup = rules.setup(mrow, fp, stl_rel(gid), case_rel(gid), gid, gates=gates,
                        knobs=knobs)
    st["decisions"] += 1
    lookup = {(v["attempt"], v["config_sha256"]): v["refused"]
              for v in end["vetoes"]}
    if setup["refused"]:
        if end["terminal"] != "REFUSED" or sorted(end["refused"]) != \
                sorted(setup["refused"]):
            bad(gid, 1, "rules.setup refuses %s but the end record says %s / %s"
                % (setup["refused"], end["terminal"], end["refused"]))
        return
    v1 = lookup.get((1, sha(setup["config"]))) if "preflight" in layers else None
    if "preflight" in layers and v1 is None:
        bad(gid, 1, "no recorded attempt-1 verdict for the setup config")
    if end["terminal"] == "REFUSED":
        if ordered:
            bad(gid, 1, "REFUSED end with %d rows" % len(ordered))
        if not v1:
            bad(gid, 1, "REFUSED end but the recorded attempt-1 verdict passes")
        if sorted(v1) != sorted(end["refused"]):
            bad(gid, 1, "the recorded refusals %s differ from the end record %s"
                % (sorted(v1), sorted(end["refused"])))
        return
    if v1:
        bad(gid, 1, "the recorded attempt-1 verdict refuses %s but the geometry ran"
            % v1)
    if not ordered:
        bad(gid, 1, "no rows after a passing setup")
    appliers = [r for r in setup["records"] if r["verdict"] == "apply" and r["edits"]]
    last_rule = appliers[-1]["rule_id"] if appliers else None
    row1 = ordered[0]
    if row1["decided_by"] == "prior":
        st["hook_decisions"] += 1
    else:
        if row1["config_sha"] != sha(setup["config"]):
            bad(gid, 1, "row 1 carries %s, the setup config is %s"
                % (row1["config_sha"][:12], sha(setup["config"])[:12]))
        if row1["rule_id"] != last_rule:
            bad(gid, 1, "row 1 rule_id %r, the last applying setup rule is %r"
                % (row1["rule_id"], last_rule))
    for i, r in enumerate(ordered):
        cfg = read_cfg(gid, r["attempt"])
        oc = copy.deepcopy(r["outcome"])
        for p in oc.get("patches") or []:
            p["capability_limited"] = False
        hist = [{"attempt": x["attempt"], "config_sha256": x["config_sha"],
                 "rule_id": x["rule_id"] if x["decided_by"] == "remedy" else None}
                for x in ordered[:i + 1]]
        ctx = {"geometry_id": gid, "fingerprint": fp, "flow": mrow["flow"],
               "win_level": setup["summary"]["win_level"], "config": cfg,
               "outcome": oc}
        stop = {"sha": None}

        def veto(cand, a_now=r["attempt"], stop=stop):
            key = (a_now + 1, sha(cand))
            if key not in lookup:
                stop["sha"] = sha(cand)
                raise _ReplayStop()
            return lookup[key]

        try:
            res = remedies.propose(ctx, hist, k=k, gates=gates, knobs=knobs,
                                   veto=veto if "preflight" in layers else None)
        except _ReplayStop:
            bad(gid, r["attempt"] + 1,
                "a veto for an unrecorded candidate %s at attempt %d"
                % ((stop["sha"] or "?")[:12], r["attempt"] + 1))
        st["decisions"] += 1
        nxt = ordered[i + 1] if i + 1 < len(ordered) else None
        if nxt is not None:
            _replay_next(gid, r, nxt, res, st, bad)
        else:
            if res.get("terminal") != end["terminal"]:
                bad(gid, r["attempt"], "the replayed terminal is %s, the end record "
                    "says %s" % (res.get("terminal"), end["terminal"]))
            if sorted(res.get("capability_limited") or []) != \
                    sorted(end["capability_limited"]):
                bad(gid, r["attempt"], "the replayed capability-limited patches %s "
                    "differ from the end record %s"
                    % (sorted(res.get("capability_limited") or []),
                       sorted(end["capability_limited"])))


def _replay_next(gid, r, nxt, res, st, bad):
    if nxt["decided_by"] == "remedy":
        if res["verdict"] != "apply":
            bad(gid, nxt["attempt"], "expected a remedy apply, got %s"
                % res["verdict"])
        for field, got, want in (
                ("rule_id", res["rule_id"], nxt["rule_id"]),
                ("config_sha256", res["config_sha256"], nxt["config_sha"]),
                ("edits", res["edits"], nxt["config_delta"]),
                ("stage_focus", res["stage_focus"], nxt["stage_focus"]),
                ("trigger", res["record"]["trigger"], nxt["trigger"])):
            if json.dumps(got, sort_keys=True) != json.dumps(want, sort_keys=True):
                bad(gid, nxt["attempt"], "the replayed %s differs: %s != %s"
                    % (field, json.dumps(got, sort_keys=True)[:200],
                       json.dumps(want, sort_keys=True)[:200]))
    elif nxt["decided_by"] == "optimiser":
        if res.get("terminal") != "EXHAUSTED":
            bad(gid, nxt["attempt"], "expected EXHAUSTED before the optimiser, got "
                "%s" % res.get("terminal"))
        st["hook_decisions"] += 1
    else:
        bad(gid, nxt["attempt"], "row %d decided_by %r"
            % (nxt["attempt"], nxt["decided_by"]))


def _strip_row(r):
    x = copy.deepcopy(r)
    for k in TIME_KEYS:
        x.pop(k, None)
    if isinstance(x.get("outcome"), dict):
        x["outcome"]["seconds"] = None
    if x.get("prediction") is not None:
        x["prediction"]["t_predicted"] = None
    for e in x.get("constraint_refusals") or []:
        if isinstance(e, dict) and "t" in e:
            e["t"] = None
    return x


def _strip_geom(g):
    x = copy.deepcopy(g)
    for k in TIME_KEYS:
        x.pop(k, None)
    x.pop("seconds", None)
    for tag in x.get("records") or []:
        if isinstance(tag, dict) and isinstance(tag.get("record"), dict) \
                and "t" in tag["record"]:
            tag["record"]["t"] = None
    return x


def _first_diff(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return path + "/" + k
            d = _first_diff(a[k], b[k], path + "/" + k)
            if d is not None:
                return d
        return None
    if isinstance(a, list) and isinstance(b, list):
        for i in range(max(len(a), len(b))):
            if i >= len(a) or i >= len(b):
                return "%s/%d" % (path, i)
            d = _first_diff(a[i], b[i], "%s/%d" % (path, i))
            if d is not None:
                return d
        return None
    if json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True):
        return path or "/"
    return None


def compare(dir_a, dir_b):
    """Two runs of one campaign: equal apart from the time fields (G-DET)."""
    rows_a = {(r["geometry_id"], r["attempt"]): r for r in load_rows(dir_a)}
    rows_b = {(r["geometry_id"], r["attempt"]): r for r in load_rows(dir_b)}
    row_diffs = []
    for key in sorted(set(rows_a) | set(rows_b)):
        if key not in rows_a or key not in rows_b:
            row_diffs.append({"key": list(key), "path": "/"})
            continue
        p = _first_diff(_strip_row(rows_a[key]), _strip_row(rows_b[key]))
        if p is not None:
            row_diffs.append({"key": list(key), "path": p})
    geoms_a = {g["geometry_id"]: g for g in load_geometries(dir_a)}
    geoms_b = {g["geometry_id"]: g for g in load_geometries(dir_b)}
    geom_diffs = []
    for gid in sorted(set(geoms_a) | set(geoms_b)):
        if gid not in geoms_a or gid not in geoms_b:
            geom_diffs.append({"key": [gid], "path": "/"})
            continue
        p = _first_diff(_strip_geom(geoms_a[gid]), _strip_geom(geoms_b[gid]))
        if p is not None:
            geom_diffs.append({"key": [gid], "path": p})
    content_compared = 0
    content_equal = 0
    for key in sorted(set(rows_a) & set(rows_b)):
        ca, cb = rows_a[key].get("content_sha256"), rows_b[key].get("content_sha256")
        if ca is not None and cb is not None:
            content_compared += 1
            content_equal += 1 if ca == cb else 0

    def audit_of(d):
        out = {"n": 0, "equal": 0}
        for g in load_geometries(d):
            for e in g["audit"]:
                out["n"] += 1
                if e["equal"]:
                    out["equal"] += 1
        return out

    audit_a = audit_of(dir_a)
    audit_b = audit_of(dir_b)
    ok = (len(rows_a) == len(rows_b) and not row_diffs and not geom_diffs
          and content_compared == content_equal and audit_a == audit_b)
    return {"schema": COMPARE_SCHEMA, "rows_a": len(rows_a), "rows_b": len(rows_b),
            "rows_equal": len(rows_a) == len(rows_b),
            "n_row_diffs": len(row_diffs), "row_diffs": row_diffs[:20],
            "content_compared": content_compared, "content_equal": content_equal,
            "n_geometry_diffs": len(geom_diffs), "geometry_diffs": geom_diffs[:20],
            "audit_a": audit_a, "audit_b": audit_b, "ok": ok}


def _audit_counts(out_dir):
    n = eq = 0
    for g in load_geometries(out_dir):
        for e in g["audit"]:
            n += 1
            if e["equal"]:
                eq += 1
    return n, eq


def _gate_smoke(out_dir, streams, binary):
    a = os.path.join(out_dir, "smoke_a")
    b = os.path.join(out_dir, "smoke_b")
    for d in (a, b):
        if os.path.exists(d):
            raise CampaignError("gate: %s exists - use a fresh transient directory"
                                % d)
    opts = {"manifest": "tuning", "mode": "rules", "ids": list(SMOKE_IDS),
            "streams": min(streams, 4), "audit_mod": 1, "binary": binary}
    end_a = run_campaign(dict(opts, out=a, run_id="smoke_a"))
    end_b = run_campaign(dict(opts, out=b, run_id="smoke_b"))
    cmp_res = compare(a, b)
    rp_a = replay(a)
    rp_b = replay(b)
    na, ea = _audit_counts(a)
    nb, eb = _audit_counts(b)
    ok = (cmp_res["ok"] and rp_a["ok"] and rp_b["ok"]
          and end_a["harness_errors"] == 0 and end_b["harness_errors"] == 0
          and na >= 1 and ea == na and nb >= 1 and eb == nb
          and summarise_campaign(a)["audit"]["ok"]
          and summarise_campaign(b)["audit"]["ok"]
          and end_a["orphans"] == [] and end_b["orphans"] == []
          and end_a["max_live_mesher"] <= streams
          and end_b["max_live_mesher"] <= streams)
    for d, want in ((a, "PASS"), (b, "PASS")):
        geoms = {g["geometry_id"]: g for g in load_geometries(d)}
        ok = ok and geoms.get("D-1-010", {}).get("terminal") == want \
            and geoms.get("G-1-026", {}).get("terminal") == "SURFACE-OPEN"
    nums = ("rows %d, terminals %s, decisions %d, audited %d, peak rss %.1f MiB, "
            "wall %.1f s"
            % (end_a["n_rows"], json.dumps(end_a["terminals"], sort_keys=True),
               rp_a["decisions"] + rp_b["decisions"], na + nb,
               max(end_a["peak_rss_mib"], end_b["peak_rss_mib"]),
               end_a["wall_seconds"] + end_b["wall_seconds"]))
    return {"verdict": "PASS" if ok else "FAIL", "numbers": nums}


def _level_table(dirs):
    lv = {}
    for d in dirs:
        for j in _read_jsonl(os.path.join(d, FILES["jobs"])):
            if j["kind"] != "full" or "wall_level" not in j:
                continue
            lv.setdefault(j["wall_level"], []).append(j)
    table = {}
    for level in sorted(lv):
        js = lv[level]
        peaks = sorted(j["peak_rss_mib"] for j in js)
        secs = sorted(j["seconds"] for j in js)
        table[str(level)] = {"n": len(js), "peak_mib_max": peaks[-1],
                             "peak_mib_median": statistics.median(peaks),
                             "seconds_median": statistics.median(secs)}
    return table


def _gate_part1(out_dir, streams, binary):
    with open(os.path.join(FIXTURE_DIR, "gdet_ids.json"), encoding="utf-8") as f:
        ids = json.load(f)["ids"]
    a = os.path.join(out_dir, "det_a")
    b = os.path.join(out_dir, "det_b")
    for d in (a, b):
        if os.path.exists(d):
            raise CampaignError("gate: %s exists - use a fresh transient directory"
                                % d)
    streams = min(streams, MAX_STREAMS)
    opts = {"manifest": "tuning", "mode": "rules", "ids": ids, "streams": streams,
            "audit_mod": AUDIT_MOD, "binary": binary}
    end_a = run_campaign(dict(opts, out=a, run_id="det_a"))
    end_b = run_campaign(dict(opts, out=b, run_id="det_b"))
    cmp_res = compare(a, b)
    rp_a = replay(a)
    rp_b = replay(b)
    na, ea = _audit_counts(a)
    nb, eb = _audit_counts(b)
    table = _level_table([a, b])
    why = None
    l5_peak = None
    l5_cap = None
    for level, row in table.items():
        if int(level) == 5:
            l5_peak = row["peak_mib_max"]
    if l5_peak is None:
        why = "no level-5 job"
    else:
        l5_cap = min(STREAMS_DOCS, int((end_a["ram_total_mib"] - RAM_RESERVE_MIB)
                                       // l5_peak))
    ok = (cmp_res["ok"] and rp_a["ok"] and rp_b["ok"]
          and end_a["harness_errors"] == 0 and end_b["harness_errors"] == 0
          and na >= 1 and ea == na and nb >= 1 and eb == nb
          and end_a["peak_frac"] <= RAM_FRACTION
          and end_b["peak_frac"] <= RAM_FRACTION
          and end_a["orphans"] == [] and end_b["orphans"] == []
          and end_a["max_live_mesher"] <= streams
          and end_b["max_live_mesher"] <= streams and why is None)
    if why is not None:
        ok = False
    nums = ("rows %d, terminals %s, decisions %d, audited %d, peak rss %.1f/%.1f MiB "
            "(%.1f%%/%.1f%% of RAM), max mesher %d, l5 peak %.1f MiB, l5 cap %d, "
            "wall %.1f s"
            % (end_a["n_rows"], json.dumps(end_a["terminals"], sort_keys=True),
               rp_a["decisions"] + rp_b["decisions"], na + nb,
               end_a["peak_rss_mib"], end_b["peak_rss_mib"],
               end_a["peak_frac"] * 100.0, end_b["peak_frac"] * 100.0,
               max(end_a["max_live_mesher"], end_b["max_live_mesher"]),
               l5_peak if l5_peak is not None else 0.0,
               l5_cap if l5_cap is not None else 0,
               end_a["wall_seconds"] + end_b["wall_seconds"]))
    out = {"verdict": "PASS" if ok else "FAIL", "numbers": nums, "levels": table}
    if why is not None:
        out["why"] = why
    return out


def _seal_child(args, timeout=600):
    return subprocess.run([sys.executable, os.path.abspath(__file__), *args],
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout)


def _gate_part2(out_dir, streams, binary):
    lock = split.read_lock(os.path.join(split.MANIFEST_DIR, "split.lock"))
    tid = sorted(lock["test_ids"].split())[0]
    checks = {}
    sa = os.path.join(out_dir, "sealed_a")
    p = _seal_child(["--run", "--manifest", "test", "--mode", "rules",
                     "--out", sa])
    checks["a"] = (p.returncode == 2 and "sealed" in (p.stderr or "")
                   and not os.path.exists(sa))
    rows_path = os.path.join(out_dir, "sealed_rows.jsonl")
    with open(rows_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"geometry_id": tid}, sort_keys=True) + "\n")
    sb = os.path.join(out_dir, "sealed_b")
    p = _seal_child(["--run", "--manifest", rows_path, "--mode", "rules",
                     "--out", sb])
    checks["b"] = (p.returncode == 2 and "held-out" in (p.stderr or "")
                   and not os.path.exists(sb))
    with open(os.path.join(HERE, "fixtures", "explain", "rows.jsonl"),
              encoding="utf-8") as f:
        last = [json.loads(ln) for ln in f if ln.strip()][-1]
    row = copy.deepcopy(last)
    row["geometry_id"] = tid
    row["fingerprint"]["geometry_id"] = tid
    row["split"] = "test"
    sc = os.path.join(out_dir, "sealed_c")
    camp = Campaign(sc, campaign_id="sealc", mode="rules", system="rules",
                    ablate=(), split_mode="rules", binary=binary, streams=1,
                    timeout_s=TIMEOUT_S, probe_timeout_s=PROBE_TIMEOUT_S,
                    audit_mod=1, quiet=True)
    raised = False
    try:
        camp.append_row(row)
    except split.SplitSealed:
        raised = True
    checks["c"] = raised and not os.path.exists(os.path.join(sc, FILES["attempts"]))
    ok = all(checks.values())
    nums = ("seal a %s, b %s, c %s (first test id %s)"
            % ("ok" if checks["a"] else "BAD", "ok" if checks["b"] else "BAD",
               "ok" if checks["c"] else "BAD", tid))
    return {"verdict": "PASS" if ok else "FAIL", "numbers": nums, "checks": checks}


_GATE_PARTS = {"smoke": _gate_smoke, "1": _gate_part1, "2": _gate_part2}


def _gate_write_md(report):
    with open(os.path.join(HERE, "README.md"), encoding="utf-8") as f:
        first = f.read().splitlines()[0]
    lines = [first, "", "# G-DET - the campaign runner gate (AM-11, docs/15 §F)", "",
             "date: %s" % report["date"],
             "binary sha256: %s" % report["binary_sha256"],
             "git head: %s" % report["git_head"],
             "verdict: %s" % report["verdict"], ""]
    for key in sorted(report["parts"]):
        part = report["parts"][key]
        lines.append("## part %s - %s" % (key, part["verdict"]))
        lines.append("")
        lines.append("%s" % part["numbers"])
        for k in sorted(part):
            if k in ("levels", "checks") and part[k]:
                lines.append("")
                for kk in sorted(part[k]):
                    lines.append("- %s: %s" % (kk, json.dumps(part[k][kk],
                                                              sort_keys=True)))
        lines.append("")
    with open(os.path.join(REPORT_DIR, "G-DET.md"), "w", encoding="utf-8",
              newline="\n") as f:
        f.write("\n".join(lines).rstrip("\n") + "\n")


def gate(parts, out_dir, streams=6, binary=None):
    """G-DET (docs/15 §F): the report lands in tools/autonomy/campaign/."""
    binary = binary or BINARY_DEFAULT
    if isinstance(parts, str):
        parts = [p.strip() for p in parts.split(",") if p.strip()]
    for part in parts:
        if part not in _GATE_PARTS:
            raise CampaignError("gate: part %r is not one of %s"
                                % (part, ", ".join(sorted(_GATE_PARTS))))
    os.makedirs(REPORT_DIR, exist_ok=True)
    report = None
    report_path = os.path.join(REPORT_DIR, "G-DET.json")
    if os.path.isfile(report_path):
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
    if not isinstance(report, dict) or report.get("schema") != GATE_SCHEMA:
        report = {"schema": GATE_SCHEMA}
    report["binary_sha256"] = _sha256_of_file(binary)
    report["git_head"] = Campaign._git_sha()
    report["date"] = time.strftime("%Y-%m-%d")
    report.setdefault("parts", {})
    failed = []
    for part in parts:
        res = _GATE_PARTS[part](out_dir, streams, binary)
        report["parts"][part] = res
        if res["verdict"] != "PASS":
            failed.append(part)
        print("[gate] part %s %s: %s" % (part, res["verdict"], res["numbers"]),
              flush=True)
    have = sorted(report["parts"])
    if failed:
        report["verdict"] = "FAIL"
    elif "1" in report["parts"] and "2" in report["parts"] \
            and all(report["parts"][p]["verdict"] == "PASS"
                    for p in ("1", "2")) \
            and report["parts"].get("smoke", {}).get("verdict", "PASS") == "PASS":
        report["verdict"] = "PASS"
    else:
        report["verdict"] = "PARTIAL"
    _dump_json(report_path, report)
    _gate_write_md(report)
    if failed:
        print("G-DET FAIL: %s" % ", ".join(sorted(failed)), flush=True)
        return 1
    if report["verdict"] == "PASS":
        print("G-DET PASS", flush=True)
        return 0
    print("G-DET PARTIAL: have %s" % have, flush=True)
    return 0


# --- the selftest (C15) -------------------------------------------------------

OBS_IDS = ("D-1-010", "F-1-009", "G-1-016", "G-1-026", "F-1-005")


def _fake_probe(leaves=1000):
    def probe(c, gctx, cfg, a):
        return {"n_leaves": leaves, "max_non_orth_deg": None, "exit_code": 0}
    return probe


def _fake_snap(c, gctx, config, a):
    return None


def _fake_attempt(script):
    def attempt(c, gctx, config, a, n_leaves):
        seq = script[gctx["gid"]]
        kind = seq[min(a, len(seq)) - 1]
        t_start = schema._now_iso()
        oc = remedies.synthetic_outcome(kind, config, gctx["fp"], gctx["flow"],
                                        c.gates)
        t_end = schema._now_iso()
        content = None
        if oc["exit_code"] == 0:
            content = hashlib.sha256(
                (schema.canonical_sha256(config) + kind).encode("ascii")).hexdigest()
        return {"outcome": oc, "content_sha256": content, "t_start": t_start,
                "t_end": t_end}
    return attempt


def _fake_campaign(H, name, ids, script, *, mode="rules", system=None, ablate=(),
                   probe=1000, hooks=None):
    return run_campaign({"manifest": "tuning", "ids": list(ids),
                         "out": os.path.join(H["tmp"], name), "mode": mode,
                         "system": system, "ablate": ablate, "streams": 2,
                         "quiet": True},
                        attempt_fn=_fake_attempt(script), probe_fn=_fake_probe(probe),
                        snap_fn=_fake_snap, hooks=hooks)


def _g1_constants(H):
    assert MODES == ("b0-template", "b0-lhs", "rules", "rules+prior", "rules+opt",
                     "full", "evaluate") and len(SYSTEMS) == 6
    assert LAYERS["full"] == ("preflight", "rules", "remedies", "prior", "optimiser")
    assert MAX_STREAMS == 6 and STREAMS_DOCS == 12 and AUDIT_MOD == 10
    assert len(GEOMETRY_TERMINALS) == 8
    base = {"manifest": "tuning", "mode": "rules", "out": "x/out"}
    cases = [("mode", dict(base, mode="nope"), "mode"),
             ("--system", dict(base, mode="evaluate"), "--system"),
             ("--system", dict(base, system="rules"), "--system"),
             ("ablate", dict(base, ablate=("prior",)), "ablate"),
             ("baseline", dict(base, mode="b0-template", ablate=("preflight",)),
              "baseline"),
             ("solver", dict(base, streams=0), "solver"),
             ("solver", dict(base, streams=7), "solver"),
             ("audit", dict(base, audit_mod=0), "audit"),
             ("timeout", dict(base, timeout_s=0), "timeout"),
             ("run id", dict(base, run_id="a/b"), "run id")]
    for name, opts, want in cases:
        o = dict(_RUN_OPTS)
        o.update(opts)
        try:
            _check_opts(o)
        except CampaignError as e:
            assert want in str(e), (name, str(e))
            continue
        raise CampaignError("group 1: %s was not refused" % name)


def _leaf_check(cfg, knobs):
    for ptr, val in preflight.config_leaves(cfg):
        if schema._knob_row(ptr, knobs) is not None:
            ref = schema.check_edit(ptr, val, knobs)
            assert ref is None, (ptr, val, ref)


def _g2_b0_template(H):
    knobs = schema.load_knobs()
    for gid in ("D-1-010", "F-1-009", "G-1-016"):
        row = H["rows"][gid]
        cfg = b0_template(row, H["obs"][gid]["fingerprint"])
        assert sha(cfg) == sha(b0_template(row, H["obs"][gid]["fingerprint"]))
        fp = H["obs"][gid]["fingerprint"]
        bb = fp["bbox"]
        big = max(bb[1] - bb[0], bb[3] - bb[2], bb[5] - bb[4])
        lay = cfg["refinement"]["levels"]
        assert cfg["layers"]["n"] == B0_N and len(lay) == len(fp["patches"])
        assert all(len(e["bands"]) == 1 and e["bands"][0]["level"] == B0_WALL_LEVEL
                   and e["bands"][0]["distance"] == round(0.1 * big, 6) for e in lay)
        assert cfg["refinement"]["max_level"] == B0_WALL_LEVEL
        assert all("feature_level" not in e for e in lay) and "snap" not in cfg
        want_t1 = rules.t1_floor(schema.a_priori_wall(row["flow"])["t1_a_priori_m"])
        assert cfg["layers"]["first_thickness"] == want_t1
        assert cfg["domain"]["base_size"] == round(0.5 * big, 4)
        ext = cfg["domain"]["extent"]
        assert all(abs(v / cfg["domain"]["base_size"]
                       - round(v / cfg["domain"]["base_size"])) < 1e-9 for v in ext)
        assert ext[0] <= bb[0] - 3 * big and ext[1] >= bb[1] + 6 * big
        assert ext[2] <= bb[2] - 2.5 * big and ext[3] >= bb[3] + 2.5 * big
        assert ext[4] <= bb[4] - 2.5 * big and ext[5] >= bb[5] + 2.5 * big
        _leaf_check(cfg, knobs)
    end = _fake_campaign(H, "g2-b0", ("D-1-010",), {"D-1-010": ["pass"]},
                         mode="b0-template")
    d = os.path.join(H["tmp"], "g2-b0")
    rows = load_rows(d)
    assert len(rows) == 1 and rows[0]["decided_by"] == "default" \
        and explain.validate_row(rows[0]) == []
    geoms = load_geometries(d)
    assert len(geoms) == 1 and geoms[0]["terminal"] == "BASELINE"
    assert end["terminals"] == {"BASELINE": 1}


def _g3_b0_lhs(H):
    knobs = schema.load_knobs()
    for gid in ("D-1-010", "F-1-009", "G-1-016"):
        u = lhs_unit(gid)
        assert u == lhs_unit(gid) and lhs_unit("D-1-010") != lhs_unit("G-1-016")
        for d in range(6):
            assert sorted(int(u[i][d] * 4) for i in range(LHS_N)) == [0, 1, 2, 3]
        row = H["rows"][gid]
        t = b0_template(row, H["obs"][gid]["fingerprint"])
        cfgs = b0_lhs(row, H["obs"][gid]["fingerprint"])
        assert len(cfgs) == LHS_N and all(
            rules.diff_edits(t, c) for c in cfgs)
        for c in cfgs:
            for e in rules.diff_edits(t, c):
                assert schema.check_edit(e["pointer"], e["to"], knobs) is None, e
            wl = max(b["level"] for lv in c["refinement"]["levels"]
                     for b in lv["bands"])
            feats = [lv["feature_level"] for lv in c["refinement"]["levels"]
                     if "feature_level" in lv]
            assert c["refinement"]["max_level"] >= wl
            assert all(0 <= lv <= 6 for lv in
                       [c["refinement"]["max_level"], wl] + feats)
            assert 1.1 <= c["layers"]["growth"] <= 1.3
            assert c["snap"]["feature_tolerance"] in LHS_FT
            assert c["snap"]["smoothing_passes"] in LHS_SP
    script = {"F-1-009": ["F3a", "pass", "F3a", "F3a"]}
    _fake_campaign(H, "g3-lhs", ("F-1-009",), script, mode="b0-lhs")
    d = os.path.join(H["tmp"], "g3-lhs")
    rows = load_rows(d)
    assert len(rows) == 4 and all(r["decided_by"] == "default" for r in rows)
    assert all(explain.validate_row(r) == [] for r in rows)
    geoms = load_geometries(d)
    g = geoms[0]
    assert g["terminal"] == "BASELINE" and g["failure"] is False
    assert g["final_attempt"] == 2 and abs(g["lhs_fail_frac"] - 0.75) < 1e-12


def _g4_seal(H):
    lock = split.read_lock(os.path.join(split.MANIFEST_DIR, "split.lock"))
    tid = sorted(lock["test_ids"].split())[0]
    try:
        load_manifest("test", "rules")
    except split.SplitSealed:
        pass
    else:
        raise CampaignError("group 4: the test manifest was not sealed")
    fdir = os.path.join(H["tmp"], "g4-file.json")
    with open(fdir, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"geometry_id": tid}, sort_keys=True) + "\n")
    out = os.path.join(H["tmp"], "g4-sealed-out")
    try:
        run_campaign({"manifest": fdir, "mode": "rules", "out": out,
                      "streams": 2, "quiet": True})
    except split.SplitSealed:
        pass
    else:
        raise CampaignError("group 4: a test id in a FILE was not refused")
    assert not os.path.exists(out)
    row = copy.deepcopy(H["fixture_row"])
    row["geometry_id"] = tid
    row["fingerprint"]["geometry_id"] = tid
    row["split"] = "test"
    cdir = os.path.join(H["tmp"], "g4-rowwriter")
    camp = Campaign(cdir, campaign_id="g4", mode="rules", system="rules",
                    ablate=(), split_mode="rules", binary=H["binary"], streams=1,
                    timeout_s=TIMEOUT_S, probe_timeout_s=PROBE_TIMEOUT_S,
                    audit_mod=1, quiet=True)
    try:
        camp.append_row(row)
    except split.SplitSealed:
        pass
    else:
        raise CampaignError("group 4: the row writer accepted a test row")
    assert not os.path.exists(os.path.join(cdir, FILES["attempts"]))
    changed = copy.deepcopy(H["rows"]["D-1-010"])
    key = sorted(changed["params"])[0]
    v = changed["params"][key]
    changed["params"][key] = v + 1 if isinstance(v, (int, float)) \
        and not isinstance(v, bool) else "tampered"
    tdir = os.path.join(H["tmp"], "g4-tampered.json")
    with open(tdir, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(changed, sort_keys=True, ensure_ascii=False) + "\n")
    try:
        load_manifest(tdir, "rules")
    except CampaignError as e:
        assert "differs" in str(e), str(e)
    else:
        raise CampaignError("group 4: an altered FILE row was not refused")
    out2 = os.path.join(H["tmp"], "g4-evaluate")
    run_campaign({"manifest": H["tuning_file"], "mode": "evaluate",
                  "system": "rules", "ids": ["D-1-010"], "out": out2,
                  "streams": 2, "quiet": True},
                 attempt_fn=_fake_attempt({"D-1-010": ["pass"]}),
                 probe_fn=_fake_probe(), snap_fn=_fake_snap)
    rows = load_rows(out2)
    assert len(rows) == 1 and rows[0]["split"] == "tuning"
    with open(os.path.join(out2, FILES["campaign"]), encoding="utf-8") as f:
        assert json.load(f)["mode"] == "evaluate"


def _g5_observe(H):
    c = H["camp"]
    raw = open(os.path.join(c.dir, "stl", "raw", "D-1-010.stl"), "rb").read()
    out = open(os.path.join(c.dir, "stl", "D-1-010.stl"), "rb").read()
    assert raw == out and H["obs"]["D-1-010"]["surface"]["repaired"] is False
    s16 = H["obs"]["G-1-016"]["surface"]
    assert s16["repaired"] is True and s16["open_edges_before"] == 24
    s26 = H["obs"]["G-1-026"]["surface"]
    assert H["obs"]["G-1-026"]["terminal"] == "SURFACE-OPEN"
    assert s26["non_manifold_after"] == 12
    for gid in ("D-1-010", "F-1-009", "G-1-016", "F-1-005"):
        errs = schema.errors(H["obs"][gid]["fingerprint"], "Fingerprint")
        assert errs == [], (gid, errs)


_G6_IDS = ("D-1-010", "F-1-009", "G-1-016", "G-1-026", "F-1-005")
_G6_SCRIPT = {"D-1-010": ["pass"], "F-1-009": ["F3a"], "G-1-016": ["F3a"],
              "F-1-005": ["retreat_snapped"]}


def _g6_fake_rules(H):
    end = _fake_campaign(H, "g6", _G6_IDS, _G6_SCRIPT)
    d = os.path.join(H["tmp"], "g6")
    rows = load_rows(d)
    geoms = {g["geometry_id"]: g for g in load_geometries(d)}
    by = {}
    for r in rows:
        by.setdefault(r["geometry_id"], []).append(r)
    assert geoms["D-1-010"]["terminal"] == "PASS" and len(by["D-1-010"]) == 1
    assert by["D-1-010"][0]["decided_by"] == "rule"
    mrow = H["rows"]["D-1-010"]
    setup = rules.setup(mrow, H["obs"]["D-1-010"]["fingerprint"], stl_rel("D-1-010"),
                        case_rel("D-1-010"), "D-1-010")
    last = [r for r in setup["records"] if r["verdict"] == "apply" and r["edits"]][-1]
    assert by["D-1-010"][0]["rule_id"] == last["rule_id"]
    assert len(by["F-1-009"]) >= 2
    assert all(r["decided_by"] == "remedy" for r in by["F-1-009"][1:])
    assert geoms["G-1-016"]["terminal"] == "EXHAUSTED" and len(by["G-1-016"]) == 1
    assert geoms["F-1-005"]["terminal"] == "CAPABILITY-LIMITED"
    assert len(by["F-1-005"]) == 1
    assert geoms["F-1-005"]["capability_limited"] == ["body"]
    patch = [p for p in by["F-1-005"][-1]["outcome"]["patches"] if p["name"] == "body"]
    assert patch and patch[0]["capability_limited"] is True
    assert geoms["G-1-026"]["terminal"] == "SURFACE-OPEN"
    assert not by.get("G-1-026") and geoms["G-1-026"]["failure"] is True
    assert end["harness_errors"] == 0 and end["orphans"] == []
    H["g6_dir"] = d
    recs = load_records(d)
    _g6_assert_common(H, d)
    return {"rows": len(rows), "recs": sum(len(v) for v in recs.values()),
            "terms": json.dumps(end["terminals"], sort_keys=True)}


def _g6_assert_common(H, d):
    rows = load_rows(d)
    recs = load_records(d)
    for r in rows:
        assert explain.validate_row(r) == [], (r["geometry_id"], r["attempt"])
        cfgp = os.path.join(d, config_rel(r["geometry_id"], r["attempt"]))
        with open(cfgp, encoding="utf-8") as f:
            assert sha(json.load(f)) == r["config_sha"]
    a = explain.audit(rows, recs)
    assert a["ok"], json.dumps({"invalid": a["invalid"], "bad":
                                a["record_order"]["bad"], "trig":
                                a["trigger_mismatch"]})[:2000]
    for gid, tags in recs.items():
        for tag in tags:
            assert any(r["attempt"] == tag["attempt"] and r["geometry_id"] == gid
                       for r in rows), (gid, tag["attempt"])
    geoms = load_geometries(d)
    assert len(geoms) == 5
    with open(os.path.join(d, FILES["records_json"]), encoding="utf-8") as f:
        rj = json.load(f)
    assert rj["schema"] == "autonomy-explain-records/1"
    assert set(rj["records"]) <= {g["geometry_id"] for g in geoms}
    s = summarise_campaign(d)
    allg = [g for g in s["groups"] if g["family"] == "all" and g["stratum"] == "all"]
    assert allg and allg[0]["n"] == 5


def _g7_replay(H):
    d = H["g6_dir"]
    rp = replay(d)
    assert rp["ok"], json.dumps(rp["mismatches"])[:2000]
    assert rp["decisions"] >= 5, rp["decisions"]
    others = ("RM-SNAP-WALL", "RM-SNAP-FT", "RM-SNAP-REFINE")
    for name, mutate in (
            ("rule id", _tamper_rule_id(others)),
            ("config file", _tamper_config),
            ("veto", _tamper_vetoes)):
        copy_dir = os.path.join(H["tmp"], "g7-" + name.replace(" ", "-"))
        shutil.copytree(d, copy_dir)
        mutate(copy_dir)
        rp2 = replay(copy_dir)
        assert not rp2["ok"], name
        assert any(m["geometry_id"] == "F-1-009" for m in rp2["mismatches"]), name
    return {"decisions": rp["decisions"]}


def _rewrite_rows(d, fn):
    path = os.path.join(d, FILES["attempts"])
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(ln) for ln in f if ln.strip()]
    rows = [fn(r) for r in rows]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")


def _tamper_rule_id(others):
    def mutate(d):
        def fn(r):
            if r["geometry_id"] == "F-1-009" and r["attempt"] == 2:
                pick = [o for o in others if o != r["rule_id"]]
                r["rule_id"] = pick[0]
            return r
        _rewrite_rows(d, fn)
    return mutate


def _tamper_config(d):
    path = os.path.join(d, "configs", "F-1-009_a2.json")
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    assert "growth" in cfg["layers"], "the fixture has no /layers/growth"
    cfg["layers"]["growth"] = round(cfg["layers"]["growth"] + 0.01, 3)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(cfg, f, indent=1, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def _tamper_vetoes(d):
    path = os.path.join(d, FILES["geometries"])
    with open(path, encoding="utf-8") as f:
        geoms = [json.loads(ln) for ln in f if ln.strip()]
    for g in geoms:
        if g["geometry_id"] == "F-1-009":
            g["vetoes"] = []
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for g in geoms:
            f.write(json.dumps(g, sort_keys=True, ensure_ascii=False) + "\n")


def _g8_veto_ablation(H):
    end = _fake_campaign(H, "g8-refused", ("D-1-010",), {"D-1-010": ["pass"]},
                         probe=3000000)
    d = os.path.join(H["tmp"], "g8-refused")
    g = load_geometries(d)[0]
    assert g["terminal"] == "REFUSED" and "PF-BUDGET" in g["refused"]
    assert not load_rows(d) and g["attempts"] == 0
    assert any(r["record"]["rule_id"] == "PF-BUDGET"
               for r in g["records"] if r["attempt"] == 1)
    end = _fake_campaign(H, "g8-nopf", ("D-1-010",), {"D-1-010": ["pass"]},
                         ablate=("preflight",), probe=3000000)
    d = os.path.join(H["tmp"], "g8-nopf")
    g = load_geometries(d)[0]
    rows = load_rows(d)
    assert len(rows) == 1 and g["terminal"] == "PASS"
    assert [r["rule_id"] for r in rows[0]["constraint_refusals"]] == ["PF-BUDGET"]
    assert explain.validate_row(rows[0]) == []
    end = _fake_campaign(H, "g8-norel", ("F-1-009",), {"F-1-009": ["F3a"]},
                         ablate=("remedies",))
    d = os.path.join(H["tmp"], "g8-norel")
    g = load_geometries(d)[0]
    assert len(load_rows(d)) == 1 and g["terminal"] == "EXHAUSTED"
    assert "K = 1" in (g["reason"] or ""), g["reason"]


def _hook_record(layer, rid, verdict, edits, message):
    return {"schema": "autonomy-decision/1", "layer": layer, "rule_id": rid,
            "verdict": verdict,
            "trigger": {"observable": "hook", "value": 0, "threshold": 0, "op": "==",
                        "source": "the selftest hook"},
            "inputs": [], "formula": "the selftest hook", "edits": edits,
            "cite": "tools/autonomy/campaign.py --selftest", "message": message,
            "uncertainty": 0.0, "t": schema._now_iso()}


def _g9_hooks(H):
    pid = "-".join(("PR", "FAKE"))
    oid = "-".join(("OPT", "FAKE"))
    for mode, want in (("rules+prior", "prior.py"), ("rules+opt", "optimise.py")):
        out = os.path.join(H["tmp"], "g9-missing-" + mode.replace("+", "-"))
        try:
            run_campaign({"manifest": "tuning", "mode": mode, "out": out,
                          "ids": ["D-1-010"], "streams": 2, "quiet": True})
        except CampaignError as e:
            assert want in str(e), str(e)
        else:
            raise CampaignError("group 9: %s was not refused" % mode)
        assert not os.path.exists(out)
    explain.TEMPLATES[pid] = {"layer": "prior", "title": "the selftest prior",
                              "because": "the selftest exercises the hook seam"}
    explain.TEMPLATES[oid] = {"layer": "optimiser",
                              "title": "the selftest optimiser",
                              "because": "the selftest exercises the hook seam"}

    def prior(ctx):
        cfg = ctx["config"]
        if ctx["geometry_id"] != "D-1-010":
            return {"verdict": "abstain",
                    "record": _hook_record("prior", pid, "abstain", [],
                                           "the selftest prior abstains")}
        after = copy.deepcopy(cfg)
        after.setdefault("snap", {})["smoothing_passes"] = 1
        edits = rules.diff_edits(cfg, after)
        return {"verdict": "apply", "config": after, "edits": edits,
                "record": _hook_record("prior", pid, "apply", edits,
                                       "the selftest prior applies")}

    def optimise(ctx, history):
        cfg = ctx["config"]
        after = copy.deepcopy(cfg)
        after.setdefault("layers", {})["growth"] = 1.1
        edits = rules.diff_edits(cfg, after)
        pred = {"p_fail": 0.1, "p_fail_std": 0.05, "blc8_a_priori": 0.5,
                "log_cells": 4.0, "t_predicted": schema._now_iso()}
        return {"verdict": "apply", "config": after, "edits": edits,
                "prediction": pred,
                "record": _hook_record("optimiser", oid, "apply", edits,
                                       "the selftest optimiser proposes")}
    try:
        _g9_hooks_run(H, pid, oid, prior, optimise)
    finally:
        explain.TEMPLATES.pop(pid, None)
        explain.TEMPLATES.pop(oid, None)


def _g9_hooks_run(H, pid, oid, prior, optimise):
    end = _fake_campaign(H, "g9-full", ("D-1-010", "G-1-016"),
                         {"D-1-010": ["pass"], "G-1-016": ["F3a", "pass"]},
                         mode="full", hooks={"prior": prior,
                                             "optimiser": optimise})
    d = os.path.join(H["tmp"], "g9-full")
    rows = load_rows(d)
    by = {}
    for r in rows:
        by.setdefault(r["geometry_id"], []).append(r)
    r1 = by["D-1-010"][0]
    assert r1["decided_by"] == "prior" and r1["rule_id"] == pid
    g2 = by["G-1-016"][1]
    assert g2["decided_by"] == "optimiser" and g2["rule_id"] == oid
    assert g2["prediction"] is not None
    assert schema._parse_iso(g2["prediction"]["t_predicted"]) < \
        schema._parse_iso(g2["t_start"])
    geoms = {g["geometry_id"]: g for g in load_geometries(d)}
    assert geoms["D-1-010"]["terminal"] == "PASS"
    assert geoms["G-1-016"]["terminal"] == "PASS"
    assert end["harness_errors"] == 0
    assert all(explain.validate_row(r) == [] for r in rows)
    a = explain.audit(rows, load_records(d))
    assert a["ok"], json.dumps(a["record_order"]["bad"])[:1000]


def _g10_runner(H):
    c = H["camp"]
    jobs_path = os.path.join(c.dir, FILES["jobs"])
    n_jobs_before = len(_read_jsonl(jobs_path))
    job = c.launch([sys.executable, "-c", "import time; time.sleep(30)"],
                   cwd=H["tmp"], timeout_s=1.5, kind="repair", gid="g10",
                   attempt=0, tag="sleepy")
    assert job["timed_out"] is True and job["exit_code"] is None
    assert job["seconds"] < 10
    assert not psutil.pid_exists(job["pid"])
    job = c.launch([sys.executable, "-c",
                    "b = bytearray(50 * 2**20); import time; time.sleep(1.0)"],
                   cwd=H["tmp"], timeout_s=60.0, kind="repair", gid="g10",
                   attempt=0, tag="eater")
    assert job["peak_rss_mib"] >= 45, job["peak_rss_mib"]
    peak = job["peak_rss_mib"]
    assert len(_read_jsonl(jobs_path)) - n_jobs_before == 2
    assert c.orphans() == [] and c.live == {}
    ram = Ram(100.0)
    ram.acquire(80)
    order = []
    def second():
        t0 = time.perf_counter()
        ram.acquire(80)
        order.append(time.perf_counter() - t0)
    th = threading.Thread(target=second)
    th.start()
    time.sleep(0.5)
    ram.release(80)
    th.join(5)
    assert order and order[0] >= 0.4, order
    ram.release(80)
    t0s = time.perf_counter()
    ram.acquire(500)
    assert time.perf_counter() - t0s < 0.5, "acquire(500) waited"
    ram.release(500)
    _g10_rerun_reservation(H)
    _g10_progress_race(c)
    return {"peak": "%.1f" % peak}


def _g10_progress_race(c):
    """Concurrent _write_progress against a reader holding the file open."""
    c._write_progress()
    errs = []

    def writer():
        for _ in range(50):
            try:
                c._write_progress()
            except OSError as e:
                errs.append("%s: %s" % (type(e).__name__, e))

    def reader():
        path = os.path.join(c.dir, FILES["progress"])
        for _ in range(20):
            f = open(path, encoding="utf-8")
            f.read()
            time.sleep(0.01)
            f.close()

    ws = [threading.Thread(target=writer) for _ in range(4)]
    rd = threading.Thread(target=reader)
    rd.start()
    for t in ws:
        t.start()
    for t in ws:
        t.join(60)
    rd.join(60)
    assert not errs, "%d progress-write errors, first: %s" % (len(errs), errs[0])
    with open(os.path.join(c.dir, FILES["progress"]), encoding="utf-8") as f:
        assert json.load(f)["campaign_id"] == c.campaign_id


def _g10_rerun_reservation(H):
    """An audited attempt larger than half the budget still reaches its rerun
    (the rerun reuses the first job's reservation, never a second acquire)."""
    cdir = os.path.join(H["tmp"], "g10-rerun")
    c = Campaign(cdir, campaign_id="g10", mode="rules", system="rules", ablate=(),
                 split_mode="rules", binary=H["binary"], streams=1, timeout_s=10.0,
                 probe_timeout_s=10.0, audit_mod=1, quiet=True)
    c.ram = Ram(100.0)
    kinds = []

    def fake_launch(argv, **kw):
        kinds.append(kw["kind"])
        raise CampaignError("stop after the launch sequence")

    def fake_first(argv, **kw):
        kinds.append(kw["kind"])
        c.launch = fake_launch if kw["kind"] == "check" else fake_first
        return {"exit_code": 0, "timed_out": False, "stdout": "", "stderr": "",
                "seconds": 0.0}
    c.launch = fake_first
    n = int((80.0 - MIB_BASE) / MIB_PER_LEAF) + 1
    gctx = {"gid": "D-1-010", "areas": {}, "flow": {}, "audit": []}
    err = []

    def go():
        try:
            run_attempt(c, gctx, {"selftest": 1}, 1, n)
        except CampaignError as e:
            err.append(str(e))
    th = threading.Thread(target=go, daemon=True)
    th.start()
    th.join(5.0)
    assert not th.is_alive(), "the audit rerun waited on its own reservation"
    assert kinds == ["full", "check", "rerun"], kinds
    assert c.ram.n == 0 and abs(c.ram.used) < 1e-9, (c.ram.n, c.ram.used)


def _g11_sample_compare(H):
    n = sum(1 for i in range(1000)
            if audit_selected(hashlib.sha256(str(i).encode()).hexdigest()))
    assert 70 <= n <= 130, n
    assert all(audit_selected(hashlib.sha256(str(i).encode()).hexdigest(), 1)
               for i in range(50))
    d = H["g6_dir"]
    a = os.path.join(H["tmp"], "g11-a")
    b = os.path.join(H["tmp"], "g11-b")
    shutil.copytree(d, a)
    shutil.copytree(d, b)
    def shift(r):
        r["campaign_id"] = "shifted"
        r["t_start"] = "2000-01-01T00:00:00Z"
        r["t_end"] = "2000-01-01T00:00:01Z"
        r["outcome"]["seconds"] = None
        return r
    _rewrite_rows(b, shift)
    cmp_res = compare(a, b)
    assert cmp_res["ok"], json.dumps(cmp_res["row_diffs"])[:1000]
    def zero_sha(r):
        if r["geometry_id"] == "D-1-010":
            r["content_sha256"] = "0" * 64
        return r
    shutil.rmtree(b)
    shutil.copytree(d, b)
    _rewrite_rows(b, zero_sha)
    cmp_res = compare(a, b)
    assert not cmp_res["ok"] and cmp_res["n_row_diffs"] >= 1
    assert cmp_res["content_compared"] > cmp_res["content_equal"]
    assert cmp_res["row_diffs"][0]["path"] == "/content_sha256"
    shutil.rmtree(b)
    shutil.copytree(d, b)
    path = os.path.join(b, FILES["attempts"])
    with open(path, encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip()]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(lines[1:])
    cmp_res = compare(a, b)
    assert not cmp_res["ok"] and cmp_res["rows_a"] != cmp_res["rows_b"]
    return {"k": n}


def _g12_resume(H):
    ids = ("D-1-010", "F-1-005")
    script = {"D-1-010": ["pass"], "F-1-005": ["retreat_snapped"]}
    opts = {"manifest": "tuning", "ids": list(ids),
            "out": os.path.join(H["tmp"], "g12"), "mode": "rules", "streams": 2,
            "quiet": True}
    run_campaign(opts, attempt_fn=_fake_attempt(script), probe_fn=_fake_probe(),
                 snap_fn=_fake_snap)
    n_rows = len(load_rows(opts["out"]))
    assert n_rows == 2
    run_campaign(dict(opts, resume=True), attempt_fn=_fake_attempt(script),
                 probe_fn=_fake_probe(), snap_fn=_fake_snap)
    assert len(load_rows(opts["out"])) == n_rows
    path = os.path.join(opts["out"], FILES["geometries"])
    with open(path, encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip()]
    kept = [ln for ln in lines if json.loads(ln)["geometry_id"] != "F-1-005"]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(kept)
    try:
        run_campaign(dict(opts, resume=True), attempt_fn=_fake_attempt(script),
                     probe_fn=_fake_probe(), snap_fn=_fake_snap)
    except CampaignError as e:
        assert "F-1-005" in str(e), str(e)
    else:
        raise CampaignError("group 12: a partial geometry was not refused")
    try:
        run_campaign(dict(opts, resume=True, mode="b0-template"),
                     attempt_fn=_fake_attempt(script), probe_fn=_fake_probe(),
                     snap_fn=_fake_snap)
    except CampaignError as e:
        assert "differs" in str(e), str(e)
    else:
        raise CampaignError("group 12: a changed mode was not refused")
    try:
        run_campaign(opts, attempt_fn=_fake_attempt(script),
                     probe_fn=_fake_probe(), snap_fn=_fake_snap)
    except CampaignError as e:
        assert "not empty" in str(e), str(e)
    else:
        raise CampaignError("group 12: a non-empty directory was not refused")


def _g13_live(H):
    tmp = H["tmp"]
    outs = []
    for tag in ("a", "b"):
        d = os.path.join(tmp, "g13-" + tag)
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--run",
                            "--manifest", "tuning", "--ids", ",".join(LIVE_IDS),
                            "--mode", "rules", "--out", d, "--streams", "2",
                            "--audit-mod", "1", "--quiet"],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=900, env=_child_env())
        assert p.returncode == 0, (p.stdout or "")[-2000:] + (p.stderr or "")[-2000:]
        outs.append(d)
    cmp_res = compare(outs[0], outs[1])
    assert cmp_res["ok"], json.dumps(cmp_res["row_diffs"])[:1000]
    rp = [replay(d) for d in outs]
    assert rp[0]["ok"] and rp[1]["ok"], json.dumps(rp[0]["mismatches"])[:1000]
    for d in outs:
        geoms = {g["geometry_id"]: g for g in load_geometries(d)}
        assert geoms["D-1-010"]["terminal"] == "PASS"
        assert geoms["F-1-005"]["terminal"] == "NO-REMEDY"
        assert geoms["G-1-026"]["terminal"] == "SURFACE-OPEN"
        n, eq = _audit_counts(d)
        assert n >= 1 and eq == n, (n, eq)
        cases = os.path.join(d, "cases")
        for case in os.listdir(cases):
            assert not os.path.isdir(os.path.join(cases, case, "constant",
                                                  "polyMesh")), case
        with open(os.path.join(d, FILES["end"]), encoding="utf-8") as f:
            e = json.load(f)
        assert e["peak_rss_mib"] > 0 and e["orphans"] == []
    sealed = os.path.join(tmp, "g13-sealed")
    p = _seal_child(["--run", "--manifest", "test", "--mode", "rules",
                     "--out", sealed])
    assert p.returncode == 2 and "sealed" in (p.stderr or "")
    assert not os.path.exists(sealed)
    p = _seal_child(["--run", "--manifest", "tuning", "--mode", "rules",
                     "--out", os.path.join(tmp, "g13-7str"), "--streams", "7"])
    assert p.returncode == 2 and "solver" in (p.stderr or "")
    assert not os.path.exists(os.path.join(tmp, "g13-7str"))
    p = _seal_child(["--run", "--manifest", "tuning", "--mode", "rules",
                     "--out", os.path.join(tmp, "g13-0str"), "--streams", "0"])
    assert p.returncode == 2 and "solver" in (p.stderr or ""), p.stderr
    assert not os.path.exists(os.path.join(tmp, "g13-0str"))
    for argv in ([], ["--summary"]):
        p = _seal_child(argv)
        assert p.returncode == 2 and (p.stderr or "").startswith("campaign: "),             (argv, p.returncode, (p.stderr or "")[-300:])
    rp0 = rp[0]
    n_a = sum(len([r for r in load_rows(d) if r["outcome"]["exit_code"] == 0])
              for d in outs)
    peak = 0.0
    for d in outs:
        with open(os.path.join(d, FILES["end"]), encoding="utf-8") as f:
            peak = max(peak, json.load(f)["peak_rss_mib"])
    return {"d": rp0["decisions"], "a": n_a, "peak": "%.1f" % peak}


def _child_env():
    return dict(os.environ, PYTHONIOENCODING="utf-8")




def selftest():
    """Thirteen [ok] groups, then SELFTEST PASS; mesher runs only in group 13."""
    tmp = tempfile.mkdtemp(prefix="campaign-selftest-")
    H = {"tmp": tmp, "binary": BINARY_DEFAULT}
    try:
        rows = load_manifest("tuning", "selftest", ids=OBS_IDS)
        H["rows"] = {r["geometry_id"]: r for r in rows}
        with open(os.path.join(HERE, "fixtures", "explain", "rows.jsonl"),
                  encoding="utf-8") as f:
            H["fixture_row"] = [json.loads(ln) for ln in f if ln.strip()][-1]
        tfile = os.path.join(tmp, "tuning-D-1-010.jsonl")
        with open(tfile, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(H["rows"]["D-1-010"], sort_keys=True,
                               ensure_ascii=False) + "\n")
        H["tuning_file"] = tfile
        hdir = os.path.join(tmp, "harness")
        os.makedirs(hdir)
        H["camp"] = Campaign(hdir, campaign_id="selftest-harness", mode="rules",
                             system="rules", ablate=(), split_mode="rules",
                             binary=BINARY_DEFAULT, streams=2, timeout_s=TIMEOUT_S,
                             probe_timeout_s=PROBE_TIMEOUT_S, audit_mod=1,
                             quiet=True)
        H["obs"] = {gid: observe(H["camp"], H["rows"][gid]) for gid in OBS_IDS}
        _group("constants: 7 modes, 6 systems, layers per system, MAX_STREAMS 6 "
               "(docs/15 says 12), audit 1 in 10, 8 geometry terminals, 10 refusals "
               "by name", _g1_constants, H)
        _group("b0-template: 3 geometries, one band per patch at level 4, t1 = the "
               "a priori first layer, extent on the base lattice around bbox + "
               "margins, every knob leaf whitelisted", _g2_b0_template, H)
        _group("b0-lhs: 4 configs x 3 geometries, each dim one value per quarter, "
               "deterministic, every edit whitelisted, levels in 0..6",
               _g3_b0_lhs, H)
        _group("seal: test manifest sealed outside evaluate, a test id in a FILE "
               "refused before any directory, the row writer refuses a test row, "
               "an altered row refused, evaluate runs a tuning row", _g4_seal, H)
        _group("observe: D-1-010 raw bytes kept (closed), G-1-016 repaired (24 open "
               "edges -> closed), G-1-026 SURFACE-OPEN (12 non-manifold), "
               "4 fingerprints valid", _g5_observe, H)
        _group("fake rules campaign: 5 geometries, {rows} rows valid, audit ok, "
               "{recs} records tagged, terminals {terms}", _g6_fake_rules, H)
        _group("replay: {decisions} decisions over 5 geometries reproduced; a "
               "tampered rule id, config file and veto are each reported",
               _g7_replay, H)
        _group("veto and ablation: PF-BUDGET refuses attempt 1 (REFUSED, 0 rows); "
               "-preflight runs with the refusal recorded; -remedies ends after 1 "
               "attempt (EXHAUSTED, K = 1)", _g8_veto_ablation, H)
        _group("hooks: rules+prior and rules+opt refused without prior.py / "
               "optimise.py; a fake prior decides attempt 1; a fake optimiser runs "
               "after EXHAUSTED with its prediction before t_start", _g9_hooks, H)
        _group("runner: a 30 s child killed by its PID after 1.5 s, a 50 MiB child "
               "peaks at {peak} MiB, RAM admission ordered, 0 orphans",
               _g10_runner, H)
        _group("audit sample and comparator: 1000 shas -> {k} audited, mod 1 always; "
               "equal apart from time fields; a content sha change and a missing row "
               "are reported", _g11_sample_compare, H)
        _group("resume: 2 ended geometries skipped (0 new rows), a partial geometry "
               "refused by name, a changed mode refused, a non-empty directory "
               "refused without --resume", _g12_resume, H)
        _group("live: D-1-010 PASS, F-1-005 NO-REMEDY, G-1-026 SURFACE-OPEN through "
               "the CLI twice at 2 streams, rows and content sha identical, "
               "replay {d} decisions, {a} audited reruns equal, no polyMesh left, "
               "peak {peak} MiB; the CLI refuses the sealed test manifest and "
               "7 streams", _g13_live, H)
        print("SELFTEST PASS", flush=True)
        return 0
    except CampaignError as e:
        sys.stderr.write("SELFTEST FAIL: %s\n" % e)
        return 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _group(name, fn, H):
    try:
        subs = fn(H)
    except (CampaignError, split.SplitError, split.SplitSealed, AssertionError) as e:
        raise CampaignError("[%s] %s: %s" % (name.split(":")[0], type(e).__name__,
                                             str(e)[:500])) from e
    print("[ok] %s" % (name.format(**subs) if subs else name), flush=True)


# --- the CLI (C16) -------------------------------------------------------------

class _ArgParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write("campaign: %s\n" % message)
        sys.exit(2)


def _csv(text):
    return [x.strip() for x in text.split(",") if x.strip()] if text else None


def _dispatch(args, ap):
    if args.selftest:
        return selftest()
    if args.run:
        opts = {"manifest": args.manifest, "mode": args.mode, "out": args.out,
                "ablate": tuple(_csv(args.ablate) or ()),
                "binary": args.binary or BINARY_DEFAULT, "resume": args.resume,
                "quiet": args.quiet}
        if args.system:
            opts["system"] = args.system
        if args.run_id:
            opts["run_id"] = args.run_id
        if args.tag:
            opts["tag"] = args.tag
        if args.streams is not None:
            opts["streams"] = args.streams
        if args.timeout is not None:
            opts["timeout_s"] = args.timeout
        if args.probe_timeout is not None:
            opts["probe_timeout_s"] = args.probe_timeout
        if args.audit_mod is not None:
            opts["audit_mod"] = args.audit_mod
        if args.ids:
            opts["ids"] = _csv(args.ids)
        if args.limit is not None:
            opts["limit"] = args.limit
        run_campaign(opts)
        return 0
    if args.summary:
        s = summarise_campaign(_need_out(args, ap))
        if args.json:
            print(json.dumps(s, indent=1, sort_keys=True, ensure_ascii=False))
        else:
            print(summary_text(s))
        return 0
    if args.replay:
        r = replay(_need_out(args, ap))
        if args.json:
            print(json.dumps(r, indent=1, sort_keys=True, ensure_ascii=False))
        else:
            print("replay %s: %d geometries, %d decisions, %d hook decisions, "
                  "%d mismatches" % ("ok" if r["ok"] else "MISMATCH",
                                     r["geometries"], r["decisions"],
                                     r["hook_decisions"], len(r["mismatches"])))
            for m in r["mismatches"][:20]:
                print("  %s/%s: %s" % (m["geometry_id"], m["attempt"], m["why"]))
        return 0 if r["ok"] else 1
    if args.compare:
        r = compare(args.compare[0], args.compare[1])
        print(json.dumps(r, indent=1, sort_keys=True, ensure_ascii=False)
              if args.json else "compare %s: %d/%d rows, %d row diffs, content "
              "%d/%d, %d geometry diffs, audit %s/%s"
              % ("ok" if r["ok"] else "DIFF", r["rows_a"], r["rows_b"],
                 r["n_row_diffs"], r["content_equal"], r["content_compared"],
                 r["n_geometry_diffs"], r["audit_a"], r["audit_b"]))
        return 0 if r["ok"] else 1
    if args.status:
        out = _need_out(args, ap)
        done = os.path.join(out, FILES["end"])
        path = done if os.path.isfile(done) else os.path.join(out, FILES["progress"])
        with open(path, encoding="utf-8") as f:
            print(json.dumps(json.load(f), indent=1, sort_keys=True,
                             ensure_ascii=False))
        return 0
    if args.gate:
        return gate(args.parts or "smoke,1,2", _need_out(args, ap),
                    streams=args.streams if args.streams is not None else MAX_STREAMS,
                    binary=args.binary or BINARY_DEFAULT)
    ap.error("one of --selftest --run --summary --replay --compare "
             "--status --gate is required")


def _need_out(args, ap):
    if not args.out:
        ap.error("--out is required")
    return args.out


def main(argv=None):
    ap = _ArgParser(prog="campaign.py", description="the campaign runner (AM-11)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--manifest")
    ap.add_argument("--mode")
    ap.add_argument("--out")
    ap.add_argument("--system")
    ap.add_argument("--ablate")
    ap.add_argument("--run-id")
    ap.add_argument("--tag")
    ap.add_argument("--streams", type=int)
    ap.add_argument("--timeout", type=float)
    ap.add_argument("--probe-timeout", type=float)
    ap.add_argument("--audit-mod", type=int)
    ap.add_argument("--ids")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--binary")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--compare", nargs=2, metavar=("DIR_A", "DIR_B"))
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--parts", default="smoke,1,2")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    try:
        return _dispatch(args, ap)
    except (CampaignError, split.SplitError, split.SplitSealed) as e:
        sys.stderr.write("campaign: %s\n" % e)
        return 2


if __name__ == "__main__":
    sys.exit(main())
