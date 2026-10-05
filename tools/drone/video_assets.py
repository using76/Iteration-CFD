#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.

"""The drone showreel's asset stager: compositions, videos, stills, credits.

`stage` copies tools/drone/video/index.html and the nine scene compositions
into the HyperFrames build project, writes the staged compositions/title.html
from the promo title with four exact text substitutions (nothing generated is
kept in the tree), encodes the three film videos through the promo stager's
own encode_job, copies the 59 stills/brand/font/licence files, draws the
residual chart as a static SVG from the solve's residuals.csv, writes
CREDITS.txt, and records everything in <project>/assets/assets.json. `check`
re-judges a staged project from that assets.json. `--selftest` runs T1-T6 in
a temp directory, under 120 s.

The drone renders derive from the PX4 x500 model (BSD-3-Clause, (c) 2022
Rudis Laboratories / PX4 Autopilot for Drones), so NOTHING made from them is
written inside the repository: every media output goes under --project, which
must lie outside it, with the model's licence files copied beside them.
It imports the promo stager tools/promo/video_assets.py (as promo_video_assets,
via importlib, not copied) and reuses encode_job, expected_frames, sha256_file,
seq_files and probe_video from it. ffmpeg/ffprobe run as separate programs.
No GPL-licensed source was consulted. Licence: the repository's Prosperity
licence - it does not import bpy.
"""

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

import importlib.util

TOOL = "tools/drone/video_assets.py"
VERSION = "drone-video/1"
FPS = 30
SCRIPT = os.path.abspath(__file__)
DRONE_DIR = os.path.dirname(SCRIPT)
SRC_DIR = os.path.join(DRONE_DIR, "video")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT)))
PROMO_TITLE = os.path.join(REPO_ROOT, "tools", "promo", "video", "compositions", "title.html")
_PA_PATH = os.path.join(REPO_ROOT, "tools", "promo", "video_assets.py")
_spec = importlib.util.spec_from_file_location("promo_video_assets", _PA_PATH)
PA = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PA)

JOBS = [
    {"out": "assets/turntable_clean.mp4", "source": "{render}/turntable_clean/frame_%04d.png",
     "kind": "seq", "files": 240, "ss": None, "t": None, "crop": None, "scale": None,
     "speed": 1.0, "hold": 0.0},
    {"out": "assets/turntable_cp.mp4", "source": "{render}/turntable_cp/frame_%04d.png",
     "kind": "seq", "files": 240, "ss": None, "t": None, "crop": None, "scale": None,
     "speed": 1.0, "hold": 0.0},
    {"out": "assets/reveal.mp4", "source": "{render}/reveal/frame_%04d.png",
     "kind": "seq", "files": 150, "ss": None, "t": None, "crop": None, "scale": None,
     "speed": 1.0, "hold": 1.5},
]  # frames 240, 240, 195 -> 8.0, 8.0, 6.5 s

STILLS = [
    ("assets/stills/hero_clean.png", "{render}/hero_clean.png"),
    ("assets/stills/hero_cp.png", "{render}/hero_cp.png"),
    ("assets/stills/downwash.png", "{render}/downwash.png"),
    ("assets/stills/slice.png", "{render}/slice.png"),
    ("assets/stills/before.png", "{render}/before.png"),
    ("assets/stills/after.png", "{render}/after.png"),
    ("assets/multiverse/grid_clean.png", "{render}/multiverse/grid_clean.png"),
]
for _nn in range(36):
    STILLS.append(("assets/multiverse/variant_%02d_clean.png" % _nn,
                   "{render}/multiverse/variant_%02d_clean.png" % _nn))
STILLS += [
    ("assets/isaac/isaac_front34.png", "{isaac}/isaac/isaac_front34.png"),
    ("assets/isaac/isaac_side.png", "{isaac}/isaac/isaac_side.png"),
    ("assets/isaac/isaac_wake34.png", "{isaac}/isaac/isaac_wake34.png"),
    ("assets/bulc/smoke_1.png", "{smoke}/over_00012.9.png"),
    ("assets/bulc/smoke_2.png", "{smoke}/over_00021.8.png"),
    ("assets/bulc/smoke_3.png", "{smoke}/over_00030.5.png"),
    ("assets/bulc/smoke_4.png", "{smoke}/over_00038.7.png"),
    ("assets/bulc/smoke_5.png", "{smoke}/over_00046.9.png"),
    ("assets/bulc/smoke_6.png", "{smoke}/over_00055.1.png"),
    ("assets/bulc/smoke_7.png", "{smoke}/over_00063.2.png"),
    ("assets/brand/meteo_simulation_logo.png", "{brand}/meteor_simulation_logo_917x225.png"),
    ("assets/brand/bulc_mark.png", "{brand}/bulc_mark_flat_1796.png"),
    ("assets/fonts/NotoSansKR-VF.ttf", "{font}"),
    ("assets/fonts/BricolageGrotesque-VF.ttf",
     REPO_ROOT + "/tools/promo/brand/fonts/BricolageGrotesque-VF.ttf"),
    ("assets/LICENSE.txt", "{render}/LICENSE.txt"),
    ("assets/LICENSE", "{render}/LICENSE"),
]

TITLE_SUBS = [
    ('src="assets/stills/streamlines.png"', 'src="assets/stills/downwash.png"'),
    ('<div id="title-legal">(주)이터레이션즈 · Iterations Co., Ltd.</div>',
     '<div id="title-legal">제공 메테오시뮬레이션 · Meteo Simulation</div>'),
    ('<div id="title-sub-ko">F1 레이싱카 공력해석 테스트</div>',
     '<div id="title-sub-ko">국방·드론을 위한 AI 기반 CFD</div>'),
    ('<div id="title-sub-en">F1 race-car aerodynamics test</div>',
     '<div id="title-sub-en">AI-driven CFD for defence and drones</div>'),
]

RES_VIEW = (560, 300)
RES_PLOT = (56, 20, 540, 260)
RES_DEC = (-9.0, -1.0)
RES_EVERY = 5
RES_SERIES = [("U_res", "#14B8A6"), ("p_res", "#f5a524"), ("cont_err", "#a7afb8")]
CREDITS_SHA_MUSIC = "dca6366ae16930a12a33bf2f4281ee4c39d8ff15dffe99fc2b1bf6781d37586b"


class Refusal(Exception):
    def __init__(self, code, text):
        super().__init__(code + ": " + text)
        self.code = code
        self.text = text


def refuse(code, text):
    raise Refusal(code, text)


def promo_call(fn, *args):
    """Run one promo-stager call; its Refusal (VA-FFMPEG ...) is re-raised as
    the drone's VD-FFMPEG so main prints one refused line and exits 2."""
    try:
        return fn(*args)
    except PA.Refusal as e:
        raise Refusal("VD-FFMPEG", e.text)


def read_file(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def sha256_file(path):
    return PA.sha256_file(path)


def validate_project(project):
    rp = os.path.normcase(os.path.realpath(project))
    rr = os.path.normcase(os.path.realpath(REPO_ROOT))
    if rp == rr or rp.startswith(rr + os.sep):
        refuse("VD-OUT", "--project is inside the repository (" + project
               + "); nothing made from the PX4 renders is written inside it")
    if not os.path.isfile(os.path.join(project, "hyperframes.json")):
        refuse("VD-PROJECT", "--project has no hyperframes.json: " + project)


def validate_sources():
    idx = os.path.join(SRC_DIR, "index.html")
    comp = os.path.join(SRC_DIR, "compositions")
    want = ["problem.html", "setup.html", "solve.html", "review.html", "multiverse.html",
            "isaac.html", "bulc.html", "close.html", "captions.html"]
    for w in want:
        if not os.path.isfile(os.path.join(comp, w)):
            refuse("VD-SRC", "composition source missing: " + w)
    if not os.path.isfile(idx):
        refuse("VD-SRC", "index source missing: " + idx)
    if not os.path.isfile(os.path.join(SRC_DIR, "audio", "soundtrack.json")):
        refuse("VD-SRC", "audio/soundtrack.json missing under " + SRC_DIR)
    if not os.path.isfile(PROMO_TITLE):
        refuse("VD-SRC", "promo title.html missing: " + PROMO_TITLE)
    title = read_file(PROMO_TITLE)
    for old, _new in TITLE_SUBS:
        if title.count(old) != 1:
            refuse("VD-SRC", "title substitution old string count "
                   + str(title.count(old)) + " != 1: " + old[:80])
    return [idx] + [os.path.join(comp, w) for w in want]


def make_title(promo_html):
    for old, new in TITLE_SUBS:
        n = promo_html.count(old)
        if n != 1:
            refuse("VD-SRC", "title substitution old string count " + str(n)
                   + " != 1: " + old[:80])
        promo_html = promo_html.replace(old, new)
    return promo_html


def resolve_source(src, dirs):
    return src.format(render=dirs["render"], isaac=dirs["isaac"], smoke=dirs["smoke"],
                      brand=dirs["brand"], font=dirs["font"])


def _res_y(log_val):
    x0, y0, x1, y1 = RES_PLOT
    return y0 + (y1 - y0) * (RES_DEC[1] - log_val) / (RES_DEC[1] - RES_DEC[0])


def _res_kept(rows):
    keep = [r for r in rows if r[0] % RES_EVERY == 0]
    if rows and (not keep or keep[-1][0] != rows[-1][0]):
        keep.append(rows[-1])
    return keep


def parse_res_rows(csv_text):
    rows = []
    lines = [ln.strip() for ln in csv_text.strip().splitlines() if ln.strip()]
    header = lines[0].split(",")
    col = {name: i for i, name in enumerate(header)}
    for ln in lines[1:]:
        parts = ln.split(",")
        rows.append((int(parts[col["step"]]),
                     {name: float(parts[col[name]]) for name, _c in RES_SERIES}))
    return rows


def residual_svg(csv_text):
    rows = parse_res_rows(csv_text)
    if not rows:
        refuse("VD-INPUT", "residuals.csv has no data rows")
    kept = _res_kept(rows)
    x0, y0, x1, y1 = RES_PLOT
    last_step = rows[-1][0]
    out = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d">'
           % (RES_VIEW[0], RES_VIEW[1], RES_VIEW[0], RES_VIEW[1])]
    for dec in (-1, -5, -9):
        gy = _res_y(float(dec))
        out.append('<line x1="%g" y1="%.1f" x2="%g" y2="%.1f" stroke="rgba(255,255,255,0.12)" stroke-width="1"/>'
                   % (x0, gy, x1, gy))
    for name, colour in RES_SERIES:
        pts = []
        for step, vals in kept:
            lv = math.log10(max(vals[name], 1e-30))
            lv = min(max(lv, RES_DEC[0]), RES_DEC[1])
            px = x0 + (x1 - x0) * step / last_step
            pts.append("%.1f,%.1f" % (px, _res_y(lv)))
        out.append('<polyline fill="none" stroke="%s" stroke-width="3" stroke-linejoin="round"'
                   " stroke-linecap=\"round\" points=\"%s\"/>" % (colour, " ".join(pts)))
    out.append("</svg>")
    return "\n".join(out) + "\n"


def credits_text(sources):
    return "\n".join([
        "Iterations - defence drone showreel (provided by Meteo Simulation / 제공 메테오시뮬레이션)",
        "",
        "Drone model: PX4 x500, (c) 2022 Rudis Laboratories / PX4 Autopilot for Drones, "
        "BSD-3-Clause (LICENSE.txt / LICENSE beside this file). No endorsement by the "
        "copyright holders is implied.",
        "Music: HeyGen audio library 295267cb9a224310b0d170e9194db151 \"Astral Generated "
        "Music: 295267cb\" (driving synth-heavy tension, cinematic military build), "
        "sha256 " + CREDITS_SHA_MUSIC + ".",
        "Sound effects: licensed HeyGen audio library SFX (assets/sfx/SFX_INDEX.json).",
        "Narration: Kokoro-82M (Apache-2.0), voice am_michael, local.",
        "Smoke frames: Isaac Sim 6.0 path-traced smoke from a GPU fire-solver run, "
        "isaacsim_cuFFT (2026-09-21, gui_0921_114721).",
        "Isaac Sim frames: NVIDIA Isaac Sim 6.0, headless render of the exported USD scene.",
        "Fonts: Noto Sans KR, Bricolage Grotesque (SIL Open Font License 1.1).",
        "Demonstration numbers from an unsteady, unvalidated run: no validated aerodynamic "
        "coefficients. Multiverse variants are concept visualisation, not solved.",
    ]) + "\n"


def parse_argv(argv):
    if not argv:
        refuse("VD-ARGS", "no subcommand; use stage, check or --selftest")
    if argv[0] == "--selftest":
        if len(argv) != 1:
            refuse("VD-ARGS", "--selftest takes no other arguments")
        return {"cmd": "selftest"}
    if argv[0] == "stage":
        opts = {"cmd": "stage", "render": None, "isaac": None, "solve": None,
                "smoke": None, "brand": None, "font": None, "project": None}
        keys = {"--render": "render", "--isaac": "isaac", "--solve": "solve",
                "--smoke": "smoke", "--brand": "brand", "--font": "font",
                "--project": "project"}
        i = 1
        while i < len(argv):
            a = argv[i]
            if a in keys:
                if i + 1 >= len(argv):
                    refuse("VD-ARGS", a + " needs a value")
                opts[keys[a]] = argv[i + 1]
                i += 2
            else:
                refuse("VD-ARGS", "unknown argument: " + a)
        for k in ("render", "isaac", "solve", "smoke", "brand", "font", "project"):
            if not opts[k]:
                refuse("VD-ARGS", "stage needs --" + k)
        return opts
    if argv[0] == "check":
        if len(argv) != 3 or argv[1] != "--project":
            refuse("VD-ARGS", "check needs exactly --project DIR")
        return {"cmd": "check", "project": argv[2]}
    refuse("VD-ARGS", "unknown subcommand: " + argv[0])


def _copy_one(src, rel, project):
    dst = os.path.join(project, rel.replace("/", os.sep))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    return {"out": rel, "source": src, "sha256": sha256_file(dst)}


def validate_inputs(dirs):
    job_sources = []
    for job in JOBS:
        src = resolve_source(job["source"], dirs)
        files = promo_call(PA.seq_files, src)
        if len(files) != int(job["files"]):
            refuse("VD-INPUT", "sequence " + src + " has " + str(len(files))
                   + " files, expected " + str(job["files"]))
        job_sources.append((job, src))
    stills = [(rel, resolve_source(src, dirs)) for rel, src in STILLS]
    for rel, p in stills:
        if not os.path.isfile(p):
            refuse("VD-INPUT", "missing input file for " + rel + ": " + p)
    csv_path = os.path.join(dirs["solve"], "residuals.csv")
    if not os.path.isfile(csv_path):
        refuse("VD-INPUT", "missing residuals.csv: " + csv_path)
    n_rows = len([ln for ln in read_file(csv_path).strip().splitlines()
                  if ln.strip()]) - 1
    if n_rows < 2:
        refuse("VD-INPUT", "residuals.csv has " + str(max(0, n_rows)) + " data rows, need >= 2")
    return job_sources, stills, csv_path


def run_stage(opts):
    t0 = time.time()
    project = opts["project"]
    validate_project(project)
    sources = validate_sources()
    job_sources, stills, csv_path = validate_inputs(opts)
    promo_call(PA.ensure_ffmpeg)
    os.makedirs(os.path.join(project, "assets"), exist_ok=True)
    videos = []
    for job, src in job_sources:
        out = os.path.join(project, job["out"].replace("/", os.sep))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        frames, w, h = promo_call(PA.encode_job, job, src, out)
        videos.append({"out": job["out"], "source": src, "kind": job["kind"],
                       "frames": frames, "seconds": frames / FPS,
                       "width": w, "height": h, "sha256": sha256_file(out)})
        print("[stage] " + job["out"] + " " + str(frames) + " frames", flush=True)
    comps_dir = os.path.join(project, "compositions")
    os.makedirs(comps_dir, exist_ok=True)
    shutil.copyfile(sources[0], os.path.join(project, "index.html"))
    comp_names = []
    for path in sources[1:]:
        name = os.path.basename(path)
        shutil.copyfile(path, os.path.join(comps_dir, name))
        comp_names.append(name[:-5])
    title = make_title(read_file(PROMO_TITLE))
    title_path = os.path.join(comps_dir, "title.html")
    with open(title_path, "w", encoding="utf-8", newline="") as f:
        f.write(title)
    copies = [_copy_one(p, rel, project) for rel, p in stills]
    svg = residual_svg(read_file(csv_path))
    with open(os.path.join(project, "assets", "residuals.svg"), "w", encoding="utf-8",
              newline="") as f:
        f.write(svg)
    cred = credits_text({k: opts[k] for k in ("render", "isaac", "smoke", "brand", "font")})
    with open(os.path.join(project, "assets", "CREDITS.txt"), "w", encoding="utf-8",
              newline="") as f:
        f.write(cred)
    man = {
        "tool": TOOL, "version": VERSION,
        "staged": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "videos": videos, "copies": copies,
        "title": {"source": PROMO_TITLE, "source_sha256": sha256_file(PROMO_TITLE),
                  "subs": len(TITLE_SUBS), "sha256": sha256_file(title_path)},
        "residuals": {"source": csv_path,
                      "rows_kept": len(_res_kept(parse_res_rows(read_file(csv_path)))),
                      "sha256": sha256_file(os.path.join(project, "assets", "residuals.svg"))},
        "credits": {"sha256": sha256_file(os.path.join(project, "assets", "CREDITS.txt"))},
        "compositions": sorted(comp_names),
    }
    with open(os.path.join(project, "assets", "assets.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, indent=2)
    print("STAGE OK " + str(len(videos)) + " videos " + str(len(copies)) + " copies "
          "title residuals credits " + ("%.1f" % (time.time() - t0)) + " s", flush=True)
    return man


def run_check(project):
    man_path = os.path.join(project, "assets", "assets.json")
    if not os.path.isfile(man_path):
        refuse("VD-INPUT", "no assets/assets.json under " + project)
    with open(man_path, encoding="utf-8") as f:
        man = json.load(f)
    results = []
    promo_call(PA.ensure_ffmpeg)
    for v in man.get("videos", []):
        why = None
        p = os.path.join(project, v["out"].replace("/", os.sep))
        if not os.path.isfile(p):
            why = "missing"
        elif sha256_file(p) != v["sha256"]:
            why = "sha256 mismatch"
        else:
            frames, w, h = promo_call(PA.probe_video, p)
            if frames != v["frames"] or (w, h) != (v["width"], v["height"]):
                why = ("probe " + str(frames) + " frames " + str(w) + "x" + str(h)
                       + " != recorded " + str(v["frames"]) + " "
                       + str(v["width"]) + "x" + str(v["height"]))
        results.append((v["out"], why))
    for c in man.get("copies", []):
        why = None
        p = os.path.join(project, c["out"].replace("/", os.sep))
        if not os.path.isfile(p):
            why = "missing"
        elif sha256_file(p) != c["sha256"]:
            why = "sha256 mismatch"
        results.append((c["out"], why))
    key_paths = {"title": os.path.join(project, "compositions", "title.html"),
                 "residuals": os.path.join(project, "assets", "residuals.svg"),
                 "credits": os.path.join(project, "assets", "CREDITS.txt")}
    for key in ("title", "residuals", "credits"):
        rec = man.get(key, {})
        p = key_paths[key]
        why = None
        if not os.path.isfile(p):
            why = "missing"
        elif sha256_file(p) != rec.get("sha256"):
            why = "sha256 mismatch"
        results.append((key, why))
    for name in man.get("compositions", []):
        why = None if os.path.isfile(os.path.join(project, "compositions",
                                                  name + ".html")) else "missing"
        results.append(("composition " + name, why))
    for out, why in results:
        if why is None:
            print("[ok] " + out)
        else:
            print("[FAIL] " + out + ": " + why)
    k = sum(1 for _, why in results if why is None)
    n = len(results)
    print(("CHECK PASS " if k == n else "CHECK FAIL ") + str(k) + "/" + str(n))
    return k == n


def _t1():
    got = [PA.expected_frames(j) for j in JOBS]
    assert got == [240, 240, 195], got
    assert [f / FPS for f in got] == [8.0, 8.0, 6.5], got
    outs = [rel for rel, _s in STILLS]
    assert len(STILLS) == 59, len(STILLS)
    assert len(set(outs)) == 59
    print("[ok] T1 jobs and stills table")


def _t2():
    csv = ("step,U_res,p_res,cont_err,M_max,M_cell\n"
           "0,0.1,0.01,1e-9,1,1\n5,0.001,1e-5,1e-12,1,1\n10,1e-5,1e-9,1e-9,1,1\n")
    svg = residual_svg(csv)
    polys = re.findall(r'<polyline[^>]*points="([^"]*)"', svg)
    want = ["56.0,20.0 298.0,80.0 540.0,140.0",
            "56.0,50.0 298.0,140.0 540.0,260.0",
            "56.0,260.0 298.0,260.0 540.0,260.0"]
    assert polys == want, polys
    assert "<text" not in svg
    assert svg.startswith('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 560 300"')
    print("[ok] T2 residual_svg points")


def _t3():
    title = read_file(PROMO_TITLE)
    got = make_title(title)
    for _old, new in TITLE_SUBS:
        assert got.count(new) == 1, new[:60]
    for old, _new in TITLE_SUBS:
        assert got.count(old) == 0, old[:60]
    d_len = sum(len(new) - len(old) for old, new in TITLE_SUBS)
    assert len(got) == len(title) + d_len, (len(got), len(title), d_len)
    print("[ok] T3 make_title substitutions")


def _t4():
    title = read_file(PROMO_TITLE)
    cut = title.replace('<div id="title-legal">(주)이터레이션즈 · Iterations Co., Ltd.</div>', "")
    try:
        make_title(cut)
        raise AssertionError("expected VD-SRC")
    except Refusal as e:
        assert e.code == "VD-SRC", e.code
    print("[ok] T4 make_title refuses VD-SRC")


def _t5(tmp):
    from PIL import Image
    d = os.path.join(tmp, "t5")
    os.makedirs(d)
    for i in range(1, 4):
        Image.new("RGB", (16, 16), (i * 40, 60, 90)).save(
            os.path.join(d, "frame_%04d.png" % i))
    job = {"out": "assets/t5.mp4", "source": os.path.join(d, "frame_%04d.png"),
           "kind": "seq", "files": 3, "ss": None, "t": None, "crop": None,
           "scale": None, "speed": 1.0, "hold": 0.1}
    assert PA.expected_frames(job) == 6
    frames, w, h = promo_call(PA.encode_job, job, job["source"], os.path.join(d, "t5.mp4"))
    assert (frames, w, h) == (6, 16, 16), (frames, w, h)
    bad = os.path.join(tmp, "t5zero")
    os.makedirs(bad)
    for i in range(1, 4):
        with open(os.path.join(bad, "frame_%04d.png" % i), "wb"):
            pass
    job2 = dict(job, out="assets/t5zero.mp4", source=os.path.join(bad, "frame_%04d.png"))
    try:
        promo_call(PA.encode_job, job2, job2["source"], os.path.join(bad, "t5zero.mp4"))
        raise AssertionError("expected VD-FFMPEG")
    except Refusal as e:
        assert e.code == "VD-FFMPEG", e.code
    print("[ok] T5 seq encode 3+3=6 frames 16x16; 0-byte seq refuses VD-FFMPEG")


def _walk(root):
    got = {}
    for rd, _dirs, fs in os.walk(root):
        for fn in fs:
            p = os.path.join(rd, fn)
            got[os.path.relpath(p, root)] = os.path.getsize(p)
    return got


def _t6(tmp):
    empty = os.path.join(tmp, "t6empty")
    os.makedirs(empty)
    try:
        validate_project(empty)
        raise AssertionError("expected VD-PROJECT")
    except Refusal as e:
        assert e.code == "VD-PROJECT", e.code
    before = _walk(tmp)
    for args, code in (
            (["stage", "--render", "r", "--isaac", "i", "--solve", "s", "--smoke", "m",
              "--brand", "b", "--font", "f", "--project", REPO_ROOT], "VD-OUT"),
            (["stage", "--render", "r", "--isaac", "i", "--solve", "s", "--smoke", "m",
              "--brand", "b", "--font", "f", "--project", empty], "VD-PROJECT")):
        r = subprocess.run([sys.executable, SCRIPT] + args, capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
        assert r.returncode == 2 and code in r.stdout, (code, r.returncode, r.stdout[-200:])
    assert _walk(tmp) == before
    print("[ok] T6 refusals VD-OUT VD-PROJECT write nothing")


def selftest():
    t0 = time.time()
    tmp = tempfile.mkdtemp(prefix="vd-selftest-")
    try:
        _t1()
        _t2()
        _t3()
        _t4()
        _t5(tmp)
        _t6(tmp)
    except AssertionError as e:
        print("SELFTEST FAIL: " + str(e))
        return 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("selftest %.1f s" % (time.time() - t0))
    print("SELFTEST PASS 6/6")
    return 0


def main(argv):
    try:
        opts = parse_argv(argv)
        if opts["cmd"] == "selftest":
            return selftest()
        if opts["cmd"] == "stage":
            run_stage(opts)
            return 0
        return 0 if run_check(opts["project"]) else 1
    except Refusal as e:
        print("refused: " + e.code + ": " + e.text)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
