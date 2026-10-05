#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.

"""The F1 promo video asset stager.

`stage` copies tools/promo/video/index.html and every tools/promo/video/
compositions/*.html into a HyperFrames project (--project, which must carry a
hyperframes.json and lie outside the repository), then encodes the six film
videos, copies the five stills, the UI font, the brand wordmark font and the
CC BY 4.0 model's credits, and writes <project>/assets/assets.json recording
every output. `check`
re-judges a staged project from that assets.json: each video's file present,
sha256 equal, and a fresh ffprobe giving the recorded frames, width and
height; each copy present and sha256-equal; each composition source present.
`--selftest` runs T1-T6 in a temp directory, under 120 s.

JOBS, the six film videos in film order:

    | out                      | source                           | kind | ss    | t     | crop             | scale     | speed | hold | frames |
    |--------------------------|----------------------------------|------|-------|-------|------------------|-----------|-------|------|--------|
    | assets/cad_turntable.mp4 | <render>/turntable_clean/%04d    | seq  | -     | -     | -                | -         | -     | 0    | 240    |
    | assets/cp_turntable.mp4  | <render>/turntable_cp/%04d       | seq  | -     | -     | -                | -         | -     | 0    | 240    |
    | assets/reveal.mp4        | <render>/reveal/%04d             | seq  | -     | -     | -                | -         | -     | 1.5  | 195    |
    | assets/chat.mp4          | <capture>/clips/s1_geometry.mp4  | clip | 0     | 130.5 | -                | -         | 9.0   | 1.5  | 480    |
    | assets/mesh.mp4          | <capture-v1>/clips/s2_mesh.mp4   | clip | 348.5 | 9.0   | 1040:585:460:180 | 1920:1080 | 1.0   | 0    | 270    |
    | assets/residuals.mp4     | <capture>/clips/s5_residuals.mp4 | clip | 0     | 8.3   | 1170:640:330:84  | 1756:960  | 1.0   | 1.7  | 300    |

expected_frames: seq = file count + round(hold * FPS); clip =
round(t / speed * FPS) + round(hold * FPS). Seconds = frames / FPS.

Outputs under --project: the six videos, assets/stills/{hero,side,top_rear,
streamlines,slice}.png, assets/fonts/NotoSansKR-VF.ttf,
assets/fonts/BricolageGrotesque-VF.ttf (the wordmark font, OFL, from
tools/promo/brand/fonts/), assets/CREDITS.txt, assets/LICENSE.txt and
assets/assets.json.

Refusals (exit 2, before anything is written): VA-ARGS (bad CLI), VA-OUT
(--project inside the repository), VA-PROJECT (--project has no
hyperframes.json), VA-SRC (the composition sources are missing), VA-INPUT (a
source missing, a sequence with the wrong file count, a clip shorter than
ss + t, a still/font/credits file missing), VA-FFMPEG (ffmpeg/ffprobe not on
PATH, an encode failing - its last five stderr lines are printed - or a probe
mismatch).

The renders and captures derive from a CC BY 4.0 model ("F1 2026 concept" by
Qvist_Designs, via Sketchfab), so NOTHING made from them is written inside
the repository: every media output goes under --project, which must lie
outside the repository, with the model's credits and licence copied beside
them. ffmpeg and ffprobe are run as separate programs; no GPL-licensed source
was consulted.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

TOOL = "tools/promo/video_assets.py"
VERSION = "promo-video/2"
FPS = 30
CRF = 14
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "video")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STILLS = ("hero.png", "side.png", "top_rear.png", "streamlines.png", "slice.png")
FONT_NAME = "NotoSansKR-VF.ttf"
BRAND_FONT_NAME = "BricolageGrotesque-VF.ttf"
BRAND_FONT_SRC = os.path.join(REPO_ROOT, "tools", "promo", "brand", "fonts", BRAND_FONT_NAME)

JOBS = [
    {"out": "assets/cad_turntable.mp4", "source": "{render}/turntable_clean/frame_%04d.png",
     "kind": "seq", "files": 240, "ss": None, "t": None, "crop": None, "scale": None,
     "speed": 1.0, "hold": 0.0},
    {"out": "assets/cp_turntable.mp4", "source": "{render}/turntable_cp/frame_%04d.png",
     "kind": "seq", "files": 240, "ss": None, "t": None, "crop": None, "scale": None,
     "speed": 1.0, "hold": 0.0},
    {"out": "assets/reveal.mp4", "source": "{render}/reveal/frame_%04d.png",
     "kind": "seq", "files": 150, "ss": None, "t": None, "crop": None, "scale": None,
     "speed": 1.0, "hold": 1.5},
    {"out": "assets/chat.mp4", "source": "{capture}/clips/s1_geometry.mp4",
     "kind": "clip", "files": None, "ss": 0, "t": 130.5, "crop": None, "scale": None,
     "speed": 9.0, "hold": 1.5},
    {"out": "assets/mesh.mp4", "source": "{capture_v1}/clips/s2_mesh.mp4",
     "kind": "clip", "files": None, "ss": 348.5, "t": 9.0, "crop": "1040:585:460:180",
     "scale": "1920:1080", "speed": 1.0, "hold": 0.0},
    {"out": "assets/residuals.mp4", "source": "{capture}/clips/s5_residuals.mp4",
     "kind": "clip", "files": None, "ss": 0, "t": 8.3, "crop": "1170:640:330:84",
     "scale": "1756:960", "speed": 1.0, "hold": 1.7},
]


class Refusal(Exception):
    def __init__(self, code, text):
        super().__init__(code + ": " + text)
        self.code = code
        self.text = text


def refuse(code, text):
    raise Refusal(code, text)


def _num(x):
    f = float(x)
    return str(int(f)) if f.is_integer() else repr(f)


def expected_frames(job):
    if job["kind"] == "seq":
        return int(job["files"]) + round(float(job["hold"]) * FPS)
    return round(float(job["t"]) / float(job["speed"]) * FPS) + round(float(job["hold"]) * FPS)


def ffmpeg_args(job, src, out):
    args = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if job["kind"] == "seq":
        args += ["-framerate", str(FPS), "-start_number", "1", "-i", src]
    else:
        args += ["-ss", _num(job["ss"]), "-t", _num(job["t"]), "-i", src]
    vf = []
    if job.get("crop"):
        vf.append("crop=" + job["crop"])
    if job.get("scale"):
        vf.append("scale=" + job["scale"] + ":flags=lanczos")
    if float(job.get("speed", 1.0)) != 1.0:
        vf.append("setpts=PTS/" + repr(float(job["speed"])))
    vf.append("fps=" + str(FPS))
    hold = float(job.get("hold", 0.0))
    if hold > 0:
        vf.append("tpad=stop_mode=clone:stop_duration=" + repr(hold))
    args += ["-vf", ",".join(vf)]
    args += ["-an", "-c:v", "libx264", "-preset", "slow", "-crf", str(CRF),
             "-pix_fmt", "yuv420p", "-r", str(FPS), "-movflags", "+faststart",
             "-frames:v", str(expected_frames(job)), "-y", out]
    return args


def _tail(stderr):
    lines = [ln for ln in (stderr or "").splitlines() if ln.strip()]
    return " | ".join(lines[-5:]) if lines else "(no stderr)"


def _run(args):
    return subprocess.run(args, capture_output=True, text=True, errors="replace")


def ensure_ffmpeg():
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        refuse("VA-FFMPEG", "ffmpeg/ffprobe not on PATH")


def probe_video(path):
    args = ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames,width,height", "-of", "json", path]
    proc = _run(args)
    if proc.returncode != 0:
        refuse("VA-FFMPEG", "ffprobe failed on " + path + ": " + _tail(proc.stderr))
    try:
        st = json.loads(proc.stdout)["streams"][0]
        return int(st["nb_read_frames"]), int(st["width"]), int(st["height"])
    except (KeyError, IndexError, ValueError):
        refuse("VA-FFMPEG", "ffprobe gave no video stream for " + path)


def probe_size(path):
    args = ["ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "json", path]
    proc = _run(args)
    if proc.returncode != 0:
        refuse("VA-FFMPEG", "ffprobe failed on " + path + ": " + _tail(proc.stderr))
    try:
        st = json.loads(proc.stdout)["streams"][0]
        return int(st["width"]), int(st["height"])
    except (KeyError, IndexError, ValueError):
        refuse("VA-FFMPEG", "ffprobe gave no video stream for " + path)


def probe_duration(path):
    args = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "json", path]
    proc = _run(args)
    if proc.returncode != 0:
        refuse("VA-FFMPEG", "ffprobe failed on " + path + ": " + _tail(proc.stderr))
    try:
        return float(json.loads(proc.stdout)["format"]["duration"])
    except (KeyError, ValueError):
        refuse("VA-FFMPEG", "ffprobe gave no duration for " + path)


def seq_files(pattern):
    name = os.path.basename(pattern)
    m = re.match(r"^(.*)(%0(\d+)d)(.*)$", name)
    if not m:
        refuse("VA-INPUT", "sequence pattern has no %0Nd field: " + pattern)
    rx = re.compile(re.escape(m.group(1)) + r"\d{" + m.group(3) + "}" + re.escape(m.group(4)))
    d = os.path.dirname(pattern) or "."
    if not os.path.isdir(d):
        refuse("VA-INPUT", "missing sequence directory: " + d)
    return sorted(n for n in os.listdir(d) if rx.fullmatch(n))


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_project(project):
    rp = os.path.normcase(os.path.realpath(project))
    rr = os.path.normcase(os.path.realpath(REPO_ROOT))
    if rp == rr or rp.startswith(rr + os.sep):
        refuse("VA-OUT", "--project is inside the repository (" + project
               + "); nothing made from the CC BY 4.0 model is written inside it")
    if not os.path.isfile(os.path.join(project, "hyperframes.json")):
        refuse("VA-PROJECT", "--project has no hyperframes.json: " + project)


def validate_sources():
    idx = os.path.join(SRC_DIR, "index.html")
    comp = os.path.join(SRC_DIR, "compositions")
    if not os.path.isfile(idx) or not os.path.isdir(comp):
        refuse("VA-SRC", "composition sources missing under " + SRC_DIR)
    out = [(idx, "index.html")]
    for n in sorted(os.listdir(comp)):
        if n.endswith(".html"):
            out.append((os.path.join(comp, n), "compositions/" + n))
    return out


def resolve_source(job, dirs):
    return job["source"].format(render=dirs["render"], capture=dirs["capture"],
                                capture_v1=dirs["capture_v1"])


def validate_inputs(job, src):
    if job["kind"] == "seq":
        files = seq_files(src)
        if len(files) != int(job["files"]):
            refuse("VA-INPUT", "sequence " + src + " has " + str(len(files))
                   + " files, expected " + str(job["files"]))
        return files
    if not os.path.isfile(src):
        refuse("VA-INPUT", "missing clip source: " + src)
    dur = probe_duration(src)
    need = float(job["ss"]) + float(job["t"])
    if dur < need:
        refuse("VA-INPUT", "clip " + src + " is " + ("%.3f" % dur)
               + " s, shorter than ss + t = " + ("%.3f" % need) + " s")
    return None


def encode_job(job, src, out):
    ensure_ffmpeg()
    proc = _run(ffmpeg_args(job, src, out))
    if proc.returncode != 0:
        refuse("VA-FFMPEG", "encode failed for " + out + ": " + _tail(proc.stderr))
    frames, w, h = probe_video(out)
    exp = expected_frames(job)
    if frames != exp:
        refuse("VA-FFMPEG", out + ": probe frames " + str(frames)
               + " != expected " + str(exp))
    if job.get("scale"):
        sw, sh = job["scale"].split(":")
        ew, eh = int(sw), int(sh)
    elif job["kind"] == "seq":
        files = seq_files(src)
        ew, eh = probe_size(os.path.join(os.path.dirname(src), files[0]))
    else:
        ew, eh = probe_size(src)
    if (w, h) != (ew, eh):
        refuse("VA-FFMPEG", out + ": probe size " + str(w) + "x" + str(h)
               + " != expected " + str(ew) + "x" + str(eh))
    return frames, w, h


def _copy_one(src, dst, project):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    return {"out": os.path.relpath(dst, project).replace(os.sep, "/"),
            "source": src, "sha256": sha256_file(dst)}


def copy_html(project, htmls):
    sources = []
    for src, rel in htmls:
        dst = os.path.join(project, rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        sources.append(rel)
    return sources


def run_stage(project, job_sources, stills, font, credits, sources):
    assets = os.path.join(project, "assets")
    os.makedirs(assets, exist_ok=True)
    videos = []
    for job, src in job_sources:
        out = os.path.join(project, job["out"].replace("/", os.sep))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        frames, w, h = encode_job(job, src, out)
        videos.append({"out": job["out"], "source": src, "kind": job["kind"],
                       "frames": frames, "seconds": frames / FPS,
                       "width": w, "height": h, "sha256": sha256_file(out),
                       "ffmpeg_args": ffmpeg_args(job, src, out)})
        print("[stage] " + job["out"] + " " + str(frames) + " frames "
              + ("%.1f" % (frames / FPS)) + " s", flush=True)
    copies = []
    for src in stills:
        copies.append(_copy_one(src, os.path.join(assets, "stills", os.path.basename(src)), project))
    if font:
        copies.append(_copy_one(font, os.path.join(assets, "fonts", FONT_NAME), project))
    copies.append(_copy_one(BRAND_FONT_SRC, os.path.join(assets, "fonts", BRAND_FONT_NAME), project))
    for src in credits:
        copies.append(_copy_one(src, os.path.join(assets, os.path.basename(src)), project))
    man = {"tool": TOOL, "version": VERSION, "fps": FPS, "crf": CRF,
           "videos": videos, "copies": copies, "sources": list(sources)}
    with open(os.path.join(assets, "assets.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, indent=2)
    return man


def cmd_stage(opts):
    t0 = time.time()
    project = opts["project"]
    validate_project(project)
    htmls = validate_sources()
    if opts["sources_only"]:
        sources = copy_html(project, htmls)
        print("STAGE OK sources-only " + str(len(sources)) + " sources")
        return 0
    dirs = {"render": opts["render"], "capture": opts["capture"],
            "capture_v1": opts["capture_v1"]}
    ensure_ffmpeg()
    job_sources = []
    for job in JOBS:
        src = resolve_source(job, dirs)
        validate_inputs(job, src)
        job_sources.append((job, src))
    stills = [os.path.join(opts["render"], n) for n in STILLS]
    credits = [os.path.join(opts["render"], "CREDITS.txt"),
               os.path.join(opts["render"], "LICENSE.txt")]
    for p in stills + credits + [opts["font"], BRAND_FONT_SRC]:
        if not os.path.isfile(p):
            refuse("VA-INPUT", "missing input file: " + p)
    sources = copy_html(project, htmls)
    run_stage(project, job_sources, stills, opts["font"], credits, sources)
    print("STAGE OK " + str(len(JOBS)) + " videos " + str(len(STILLS)) + " stills "
          + str(len(sources)) + " sources " + ("%.1f" % (time.time() - t0)) + " s")
    return 0


def run_check(project):
    man_path = os.path.join(project, "assets", "assets.json")
    if not os.path.isfile(man_path):
        refuse("VA-INPUT", "no assets/assets.json under " + project)
    with open(man_path, encoding="utf-8") as f:
        man = json.load(f)
    results = []
    for v in man.get("videos", []):
        why = None
        p = os.path.join(project, v["out"].replace("/", os.sep))
        if not os.path.isfile(p):
            why = "missing"
        elif sha256_file(p) != v["sha256"]:
            why = "sha256 mismatch"
        else:
            try:
                frames, w, h = probe_video(p)
                if frames != v["frames"] or (w, h) != (v["width"], v["height"]):
                    why = ("probe " + str(frames) + " frames " + str(w) + "x" + str(h)
                           + " != recorded " + str(v["frames"]) + " "
                           + str(v["width"]) + "x" + str(v["height"]))
            except Refusal as e:
                why = "probe failed: " + e.text
        results.append((v["out"], why))
    for c in man.get("copies", []):
        why = None
        p = os.path.join(project, c["out"].replace("/", os.sep))
        if not os.path.isfile(p):
            why = "missing"
        elif sha256_file(p) != c["sha256"]:
            why = "sha256 mismatch"
        results.append((c["out"], why))
    for s in man.get("sources", []):
        why = None if os.path.isfile(os.path.join(project, s.replace("/", os.sep))) else "missing"
        results.append((s, why))
    for out, why in results:
        if why is None:
            print("[ok] " + out)
        else:
            print("[FAIL] " + out + ": " + why)
    k = sum(1 for _, why in results if why is None)
    n = len(results)
    print(("CHECK PASS " if k == n else "CHECK FAIL ") + str(k) + "/" + str(n))
    return k == n


def cmd_check(opts):
    return 0 if run_check(opts["project"]) else 1


def _job_seq(out, files, hold):
    return {"out": out, "kind": "seq", "files": files, "ss": None, "t": None,
            "crop": None, "scale": None, "speed": 1.0, "hold": hold}


def _t1():
    assert [j["out"] for j in JOBS] == [
        "assets/cad_turntable.mp4", "assets/cp_turntable.mp4", "assets/reveal.mp4",
        "assets/chat.mp4", "assets/mesh.mp4", "assets/residuals.mp4"]
    assert [j["kind"] for j in JOBS] == ["seq", "seq", "seq", "clip", "clip", "clip"]
    got = [expected_frames(j) for j in JOBS]
    assert got == [240, 240, 195, 480, 270, 300], got
    secs = [f / FPS for f in got]
    assert secs == [8.0, 8.0, 6.5, 16.0, 9.0, 10.0], secs
    print("[ok] T1 jobs table and expected_frames")


def _t2():
    a = ffmpeg_args(JOBS[3], "src.mp4", "out.mp4")
    i_ss = a.index("-ss")
    assert a[i_ss + 1] == "0" and i_ss < a.index("-i")
    assert a[a.index("-vf") + 1] == "setpts=PTS/9.0,fps=30,tpad=stop_mode=clone:stop_duration=1.5"
    assert a[a.index("-frames:v") + 1] == "480"
    m = ffmpeg_args(JOBS[4], "src.mp4", "out.mp4")
    assert m[m.index("-vf") + 1] == "crop=1040:585:460:180,scale=1920:1080:flags=lanczos,fps=30"
    r = ffmpeg_args(JOBS[2], "pat/frame_%04d.png", "out.mp4")
    assert r[r.index("-framerate") + 1] == "30"
    assert r[r.index("-start_number") + 1] == "1"
    assert r[r.index("-vf") + 1] == "fps=30,tpad=stop_mode=clone:stop_duration=1.5"
    print("[ok] T2 ffmpeg_args")


def _t3(tmp):
    d = os.path.join(tmp, "t3")
    os.makedirs(d)
    pat = os.path.join(d, "frame_%04d.png")
    proc = _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                 "-i", "testsrc=size=64x36:rate=30", "-frames:v", "12", pat])
    assert proc.returncode == 0, proc.stderr
    job = _job_seq("assets/t3.mp4", 12, 0.5)
    frames, w, h = encode_job(job, pat, os.path.join(d, "t3.mp4"))
    assert (frames, w, h) == (27, 64, 36), (frames, w, h)
    print("[ok] T3 seq encode 12+15=27 frames 64x36")


def _t4(tmp):
    d = os.path.join(tmp, "t4")
    os.makedirs(d)
    src = os.path.join(d, "src.mp4")
    proc = _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                 "-i", "testsrc=size=128x72:rate=30", "-t", "3", "-c:v", "libx264",
                 "-pix_fmt", "yuv420p", "-y", src])
    assert proc.returncode == 0, proc.stderr
    job = {"out": "assets/t4.mp4", "kind": "clip", "files": None, "ss": 1.0,
           "t": 1.5, "crop": "64:36:16:8", "scale": "128:72", "speed": 1.5, "hold": 0.5}
    frames, w, h = encode_job(job, src, os.path.join(d, "t4.mp4"))
    assert (frames, w, h) == (45, 128, 72), (frames, w, h)
    print("[ok] T4 clip encode ss/t/crop/scale/speed/hold 45 frames 128x72")


def _t5(tmp, t3_dir, t4_src):
    proj = os.path.join(tmp, "proj5")
    os.makedirs(proj)
    try:
        validate_project(proj)
        raise AssertionError("expected VA-PROJECT")
    except Refusal as e:
        assert e.code == "VA-PROJECT", e.code
    inside = os.path.join(REPO_ROOT, "tools", "promo", "_va_out_tmp")
    os.makedirs(inside, exist_ok=True)
    try:
        try:
            validate_project(inside)
            raise AssertionError("expected VA-OUT")
        except Refusal as e:
            assert e.code == "VA-OUT", e.code
    finally:
        os.rmdir(inside)
    d11 = os.path.join(tmp, "t5a")
    os.makedirs(d11)
    for i in range(1, 12):
        shutil.copyfile(os.path.join(t3_dir, "frame_%04d.png" % i),
                        os.path.join(d11, "frame_%04d.png" % i))
    try:
        validate_inputs(_job_seq("x", 12, 0.0), os.path.join(d11, "frame_%04d.png"))
        raise AssertionError("expected VA-INPUT (seq count)")
    except Refusal as e:
        assert e.code == "VA-INPUT", e.code
    try:
        validate_inputs({"kind": "clip", "ss": 2.0, "t": 1.5}, t4_src)
        raise AssertionError("expected VA-INPUT (clip short)")
    except Refusal as e:
        assert e.code == "VA-INPUT", e.code
    assert not os.path.isdir(os.path.join(proj, "assets"))
    print("[ok] T5 refusals VA-PROJECT VA-OUT VA-INPUT write nothing")


def _t6(tmp, t3_dir, t4_src):
    proj = os.path.join(tmp, "proj6")
    os.makedirs(proj)
    with open(os.path.join(proj, "hyperframes.json"), "w", encoding="utf-8") as f:
        f.write("{}")
    jobs = [(_job_seq("assets/t3.mp4", 12, 0.5), os.path.join(t3_dir, "frame_%04d.png")),
            ({"out": "assets/t4.mp4", "kind": "clip", "files": None, "ss": 1.0,
              "t": 1.5, "crop": "64:36:16:8", "scale": "128:72", "speed": 1.5,
              "hold": 0.5}, t4_src)]
    run_stage(proj, jobs, [], None, [], [])
    assert run_check(proj)
    p = os.path.join(proj, "assets", "t3.mp4")
    with open(p, "rb") as f:
        data = f.read()
    with open(p, "wb") as f:
        f.write(data[:-100])
    assert not run_check(proj)
    print("[ok] T6 stage + check pass, tamper caught")


def selftest():
    tmp = tempfile.mkdtemp(prefix="video-selftest-")
    try:
        _t1()
        _t2()
        _t3(tmp)
        _t4(tmp)
        _t5(tmp, os.path.join(tmp, "t3"), os.path.join(tmp, "t4", "src.mp4"))
        _t6(tmp, os.path.join(tmp, "t3"), os.path.join(tmp, "t4", "src.mp4"))
    except AssertionError as e:
        print("SELFTEST FAIL: " + str(e))
        return 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("SELFTEST PASS 6/6")
    return 0


def parse_argv(argv):
    if not argv:
        refuse("VA-ARGS", "no subcommand; use stage, check or --selftest")
    if argv[0] == "--selftest":
        if len(argv) != 1:
            refuse("VA-ARGS", "--selftest takes no other arguments")
        return {"cmd": "selftest", "project": None}
    if argv[0] == "stage":
        opts = {"cmd": "stage", "render": None, "capture": None, "capture_v1": None,
                "font": None, "project": None, "sources_only": False}
        keys = {"--render": "render", "--capture": "capture",
                "--capture-v1": "capture_v1", "--font": "font", "--project": "project"}
        i = 1
        while i < len(argv):
            a = argv[i]
            if a == "--sources-only":
                opts["sources_only"] = True
                i += 1
            elif a in keys:
                if i + 1 >= len(argv):
                    refuse("VA-ARGS", a + " needs a value")
                opts[keys[a]] = argv[i + 1]
                i += 2
            else:
                refuse("VA-ARGS", "unknown argument: " + a)
        for k in ("render", "capture", "capture_v1", "font", "project"):
            if not opts[k]:
                refuse("VA-ARGS", "stage needs --" + k.replace("_", "-"))
        return opts
    if argv[0] == "check":
        if len(argv) != 3 or argv[1] != "--project":
            refuse("VA-ARGS", "check needs exactly --project DIR")
        return {"cmd": "check", "project": argv[2]}
    refuse("VA-ARGS", "unknown subcommand: " + argv[0])


def main(argv):
    try:
        opts = parse_argv(argv)
        if opts["cmd"] == "selftest":
            return selftest()
        if opts["cmd"] == "stage":
            return cmd_stage(opts)
        return cmd_check(opts)
    except Refusal as e:
        print("refused: " + e.code + ": " + e.text)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
