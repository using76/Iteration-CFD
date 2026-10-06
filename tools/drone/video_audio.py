#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.

"""The drone showreel's soundtrack: the promo chain, rewired by import.

`voice`, `build` and `asr` are the promo commands: this module loads
tools/promo/video_audio.py once through importlib (module name
promo_video_audio), points its DATA_FILE, TOOL and VIDEO_DIR at the drone
files, and calls its cmd_voice / cmd_build / cmd_asr unchanged - so the
Kokoro VO, the music bed with VO carving and ducking, the SFX bank, the
-14 LUFS mastering with the own BS.1770-4 meter and true-peak limiter, the
ffmpeg ebur128 gate loop and the faster-whisper ASR judging are exactly the
promo code (the asr _whisper subprocess runs the promo's own script path).
`check` is the drone's own: it re-judges a built, staged project against the
C3 item list - soundtrack, the 17 voice clips, the fit, the master format/
sha/LUFS/TP, the index slots, the nine wrap fades, the composition videos
against assets.json seconds, the captions times and texts and the honesty
chip, the light-SFX and honesty and brand rules, and the staged title's
sha256. `--selftest` runs T1-T6 in a temp directory, under 120 s.

Command line (P = the staged HyperFrames project, outside the repository):

    python tools/drone/video_audio.py voice --project P --tts-python TTS
    python tools/drone/video_audio.py build --project P
    python tools/drone/video_audio.py asr --project P --asr-python ASR
    python tools/drone/video_audio.py check --project P
    python tools/drone/video_audio.py --selftest

The data file is always tools/drone/video/audio/soundtrack.json (the
supervisor's contract copy), found relative to this script. Light SFX only:
no "sub", no "impact" kind, every cue gain <= -11 dB. Branding: the only
(주)Iterations mention is the close card's "Collaboration by (주)Iterations".
Refusals are the promo's VAU-* codes; outputs live under --project only.
No GPL-licensed source was consulted. Licence: the repository's Prosperity
licence - it does not import bpy.
"""

import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile

TOOL = "tools/drone/video_audio.py"
VERSION = "drone-video-audio/1"
SCRIPT = os.path.abspath(__file__)
DRONE_DIR = os.path.dirname(SCRIPT)
DATA_FILE = os.path.join(DRONE_DIR, "video", "audio", "soundtrack.json")
VIDEO_DIR = os.path.join(DRONE_DIR, "video")
CHIP = (25.7, 49.4)
LIGHT_KINDS_BANNED = ("sub", "impact")
LIGHT_MAX_DB = -11.0
COLLAB = "Collaboration by (주)Iterations"
MAIN_CREDIT_KO = "제공 메테오시뮬레이션"
PX4_CREDIT = "(c) 2022 Rudis Laboratories / PX4 Autopilot for Drones, BSD-3-Clause"
BANNED_BRAND = ("이터레이션즈", "Iterations Co., Ltd.")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT)))

_PROMO_PATH = os.path.join(REPO_ROOT, "tools", "promo", "video_audio.py")
_spec = importlib.util.spec_from_file_location("promo_video_audio", _PROMO_PATH)
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)


def _wire_promo():
    """the promo functions read their module globals at call time, so point
    DATA_FILE (its load_data), TOOL (its audio_meta.json record) and VIDEO_DIR
    (its honesty walk) at the drone files."""
    A.DATA_FILE = DATA_FILE
    A.TOOL = TOOL
    A.VIDEO_DIR = VIDEO_DIR
    return A


_wire_promo()


class Refusal(Exception):
    def __init__(self, code, text):
        super().__init__(code + ": " + text)
        self.code = code
        self.text = text


HONESTY_DRONE = re.compile(
    A.HONESTY.pattern
    + r"|(?<![A-Za-z0-9_])lift(?![A-Za-z0-9_])|(?<![A-Za-z0-9_])nvdb",
    re.I)


def load_promo():
    return _wire_promo()


def visible_text(html):
    txt = re.sub(r"<!--.*?-->", " ", html, flags=re.S)
    txt = re.sub(r"<script\b.*?</script>", " ", txt, flags=re.S | re.I)
    txt = re.sub(r"<style\b.*?</style>", " ", txt, flags=re.S | re.I)
    txt = re.sub(r"<[^>]+>", " ", txt)
    return re.sub(r"\s+", " ", txt).strip()


def light_sfx(data):
    bad = []
    for cue in data["sfx"]["cues"]:
        t, kind, gain = cue[0], cue[1], cue[2]
        if kind in LIGHT_KINDS_BANNED:
            bad.append((t, kind, gain, "banned kind " + kind))
        elif gain > LIGHT_MAX_DB:
            bad.append((t, kind, gain, "gain %.1f dB above %.1f" % (gain, LIGHT_MAX_DB)))
    return bad


def brand_scan(htmls):
    collab = 0
    collab_in_close = False
    banned = []
    main_credit_files = []
    for name, text in sorted(htmls.items()):
        n = text.count(COLLAB)
        collab += n
        if name == "close.html" and n:
            collab_in_close = True
        for s in BANNED_BRAND:
            if s in text:
                banned.append((name, s))
        if MAIN_CREDIT_KO in text:
            main_credit_files.append(name)
    return {"collab_count": collab, "collab_in_close": collab_in_close,
            "banned": banned, "main_credit_files": main_credit_files}


def cmd_voice(args):
    return A.cmd_voice(args)


def cmd_build(args):
    return A.cmd_build(args)


def cmd_asr(args):
    return A.cmd_asr(args)


def _composition_texts(proj):
    comp_dir = os.path.join(proj, "compositions")
    texts = {}
    for name in sorted(os.listdir(comp_dir)):
        if name.endswith(".html"):
            texts[name] = A.read_file(os.path.join(comp_dir, name))
    return texts


def cmd_check(args):
    proj = A.resolve_project(args.project)
    A.which("ffmpeg")
    A.which("ffprobe")
    data, ids = A.load_data()
    items = []

    def add(name, ok, why=""):
        items.append((name, bool(ok), why))

    try:
        A.validate_data(json.loads(A.read_file(DATA_FILE)))
        add("soundtrack", True)
    except A.Refusal as e:
        add("soundtrack", False, e.text)
    clips = {}
    try:
        clips = A.read_clips(proj)
    except A.Refusal as e:
        add("voice", False, e.text)
    vrows = {l["id"]: l for l in A.load_voice_json(proj).get("lines", [])}
    for line in data["vo"]["lines"]:
        wav = os.path.join(proj, "assets", "voice", line["id"] + ".wav")
        row = vrows.get(line["id"])
        if not os.path.isfile(wav) or row is None:
            add("voice " + line["id"], False, "file or voice.json row missing")
        elif row.get("sha256") != A.sha256_file(wav):
            add("voice " + line["id"], False, "sha256 differs from voice.json")
        else:
            add("voice " + line["id"], True)
    if clips:
        try:
            A.check_fit(data, ids, clips)
            add("fit", True)
        except A.Refusal as e:
            add("fit", False, e.text)
    else:
        add("fit", False, "no voice.json")
    master_path = os.path.join(proj, "assets", "audio", "master.wav")
    meta_path = os.path.join(proj, "assets", "audio", "audio_meta.json")
    meta = None
    if os.path.isfile(meta_path):
        try:
            meta = json.loads(A.read_file(meta_path))
        except ValueError:
            meta = None
    if not os.path.isfile(master_path):
        for what in ("format", "sha", "lufs", "tp"):
            add("master " + what, False, "master.wav missing")
    else:
        codec, sr, ch, samples = A.ffprobe_format(master_path)
        n_exp = int(round(data["duration"] * A.SR))
        ok = (codec, sr, ch, samples) == ("pcm_s24le", A.SR, 2, n_exp)
        add("master format", ok, "got %s %d Hz %d ch %d samples" % (codec, sr, ch, samples))
        sha = A.sha256_file(master_path)
        add("master sha", bool(meta and meta.get("master", {}).get("sha256") == sha),
            "differs from audio_meta.json" if meta else "audio_meta.json missing")
        i, tp, _lra = A.ffmpeg_measure(master_path)
        gl = data["master"]["gate_lufs"]
        add("master lufs", gl[0] <= i <= gl[1], "I %.2f LUFS outside %s" % (i, gl))
        add("master tp", tp <= data["master"]["gate_tp_dbtp"],
            "TP %.2f dBTP above gate" % tp)
    index_path = os.path.join(proj, "index.html")
    if not os.path.isfile(index_path):
        for nm in (["index duration"] + ["index " + s["id"] for s in data["scenes"]]
                   + ["index captions", "index audio"]):
            add(nm, False, "index.html missing (run stage first)")
        return _finish(items, proj, data, ids, clips, None)
    index = A.read_file(index_path)
    root = re.search(r'<div[^>]*id="root"[^>]*>', index)
    dm = re.search(r'data-duration="(-?[\d.]+)"', root.group(0)) if root else None
    add("index duration", bool(dm and abs(float(dm.group(1)) - data["duration"]) < 0.001),
        "root data-duration " + (dm.group(1) if dm else "missing"))
    for sc in data["scenes"]:
        s, d = A.slot_times(index, sc["id"])
        ok = s is not None and abs(s - sc["start"]) < 0.001 and abs(d - sc["dur"]) < 0.001
        add("index " + sc["id"], ok, "slot %s/%s, want %s/%s" % (s, d, sc["start"], sc["dur"]))
    cs, cd = A.slot_times(index, "captions")
    add("index captions", cs is not None and abs(cs) < 0.001 and abs(cd - data["duration"]) < 0.001,
        "captions slot %s/%s" % (cs, cd))
    add("index audio", A.AUDIO_EL in index, "audio element not found verbatim")
    return _finish(items, proj, data, ids, clips, index)


def _finish(items, proj, data, ids, clips, index):
    def add(name, ok, why=""):
        items.append((name, bool(ok), why))

    comp_dir = os.path.join(proj, "compositions")
    staged = {}
    if os.path.isdir(comp_dir):
        staged = _composition_texts(proj)
    index_ok = index is not None
    for sc in data["scenes"]:
        html = staged.get(sc["id"] + ".html")
        if html is None:
            add("fade " + sc["id"], False, "composition missing")
            continue
        d, x = A.wrap_fade(html, sc["id"])
        ok = d is not None and abs((x or 0) + d - sc["dur"]) < 0.001
        add("fade " + sc["id"], ok, "fade %s+%s != %s" % (x, d, sc["dur"]))
    assets = {}
    man_path = os.path.join(proj, "assets", "assets.json")
    if os.path.isfile(man_path):
        try:
            assets = json.loads(A.read_file(man_path))
        except ValueError:
            assets = {}
    secs = {v["out"]: v["frames"] / 30.0 for v in assets.get("videos", [])}
    for sc in data["scenes"]:
        html = staged.get(sc["id"] + ".html")
        if html is None:
            add("video " + sc["id"], False, "composition missing")
            continue
        bad = []
        for vm in re.finditer(r"<video\b[^>]*>", html):
            tag = vm.group(0)
            dm = re.search(r'data-duration="([\d.]+)"', tag)
            ds = re.search(r'data-start="(-?[\d.]+)"', tag)
            sm = re.search(r'src="([^"]+)"', tag)
            vd = float(dm.group(1)) if dm else None
            vs = float(ds.group(1)) if ds else None
            src = sm.group(1) if sm else ""
            why = None
            if vd is None or vs is None:
                why = "video without data-start/data-duration"
            elif vs + vd > sc["dur"] + 0.001:
                why = "data-start %s + data-duration %s over scene %s" % (vs, vd, sc["dur"])
            elif src not in secs:
                why = "src " + src + " not in assets.json"
            elif vd > secs[src] + 0.001:
                why = "data-duration %s over asset %s s" % (vd, secs[src])
            if why:
                bad.append(why)
        add("video " + sc["id"], not bad, "; ".join(bad))
    cap_path = os.path.join(comp_dir, "captions.html")
    cap_html = staged.get("captions.html")
    want_rows = {}
    if clips:
        want_rows = A.caption_times(data, {lid: c["clip_s"] for lid, c in clips.items()})
    want = [(lid, t[0], t[1]) for lid, t in sorted(want_rows.items()) if lid != "V01"]
    if cap_html is None:
        add("captions times", False, "captions.html missing")
        for lid, _ti, _to in want:
            add("captions text " + lid, False, "captions.html missing")
        add("captions chip", False, "captions.html missing")
    else:
        got = A.caps_block(cap_html)
        ok = len(got) == len(want) and all(
            g[0] == w[0] and abs(g[1] - w[1]) <= 0.02 and abs(g[2] - w[2]) <= 0.02
            for g, w in zip(got, want))
        add("captions times", ok, "CAPS %s want %s" % (got, want))
        kos = {l["id"]: (l["ko"] or "").replace("\n", "<br />") for l in data["vo"]["lines"]}
        for lid, _ti, _to in want:
            m = re.search(r'id="captions-' + re.escape(lid) + r'-k">(.*?)</div>',
                          cap_html, re.S)
            text = m.group(1) if m else None
            add("captions text " + lid, text == kos.get(lid),
                "%r != %r" % (text, kos.get(lid)))
        cin = re.search(r'tl\.fromTo\("#captions-chip".*?,\s*([\d.]+)\s*\)\s*;',
                        cap_html, re.S)
        cout = re.findall(r'tl\.to\("#captions-chip",\s*\{([^}]*)\}\s*,\s*([\d.]+)\s*\)',
                          cap_html)
        c_ok = bool(cin and abs(float(cin.group(1)) - CHIP[0]) < 0.001
                    and cout and abs(float(cout[-1][1]) - CHIP[1]) < 0.001
                    and "opacity: 0" in cout[-1][0].replace("opacity:0", "opacity: 0"))
        add("captions chip", c_ok,
            "chip in %s, out %s" % (cin.group(1) if cin else "?",
                                    cout[-1][1] if cout else "?"))
    add("sfx light", not light_sfx(data), str(light_sfx(data)[:3]))
    bad = []
    for name, text in sorted(list(staged.items()) + [("index.html", index or "")]):
        m2 = HONESTY_DRONE.search(text)
        if m2:
            bad.append(name + ":" + m2.group(0))
    m3 = HONESTY_DRONE.search(A.read_file(DATA_FILE))
    if m3:
        bad.append("soundtrack.json:" + m3.group(0))
    add("honesty", not bad, "; ".join(bad[:5]))
    vis = {name: visible_text(text) for name, text in staged.items()}
    scan = brand_scan(vis)
    add("brand collab", scan["collab_count"] == 1 and scan["collab_in_close"],
        "collab count %d, in close %s" % (scan["collab_count"], scan["collab_in_close"]))
    add("brand banned", not scan["banned"], str(scan["banned"][:3]))
    close_vis = vis.get("close.html", "")
    title_vis = vis.get("title.html", "")
    mc = MAIN_CREDIT_KO in title_vis and MAIN_CREDIT_KO in close_vis
    img = re.search(r'<img[^>]*src="assets/brand/meteo_simulation_logo\.png"', staged.get("close.html", ""))
    add("brand main credit", mc and bool(img),
        "main credit in title %s close %s, logo img %s"
        % (MAIN_CREDIT_KO in title_vis, MAIN_CREDIT_KO in close_vis, bool(img)))
    add("brand px4", PX4_CREDIT in close_vis, "PX4 credit missing from close.html")
    mv = vis.get("multiverse.html", "")
    add("label multiverse", "Concept visualisation" in mv and "not solved" in mv,
        "multiverse labels missing")
    isa = vis.get("isaac.html", "")
    add("label isaac", "Blender render" in isa and "Isaac Sim 6.0" in isa,
        "isaac labels missing")
    rev = vis.get("review.html", "")
    add("label reduced", "REDUCED" in rev, "review REDUCED label missing")
    bl = vis.get("bulc.html", "")
    add("label smoke", "isaacsim_cuFFT" in bl, "bulc isaacsim_cuFFT label missing")
    trec = assets.get("title", {})
    tpath = os.path.join(comp_dir, "title.html")
    ok = os.path.isfile(tpath) and trec and trec.get("sha256") == A.sha256_file(tpath)
    add("title subs", ok, "staged title sha256 != assets.json")
    for name, okf, why in items:
        print(("[ok] " if okf else "[FAIL] ") + name + (": " + str(why) if why and not okf else ""))
    n = len(items)
    k = sum(1 for _x, okf, _y in items if okf)
    print(("CHECK PASS" if k == n else "CHECK FAIL") + " " + str(k) + "/" + str(n))
    return 0 if k == n else 1


PROBE_CLIPS = {"V01": 5.035, "V02": 3.157, "V03": 3.179, "V04": 4.992, "V05": 5.355,
               "V06": 5.227, "V07": 2.901, "V08": 4.971, "V09": 4.117, "V10": 3.371,
               "V11": 5.824, "V12": 5.376, "V13": 3.712, "V14": 5.0, "V15": 3.605,
               "V16": 4.629, "V17": 1.6}


def selftest():
    import time
    t0 = time.time()
    tmp = tempfile.mkdtemp(prefix="vau_dronetest_")
    results = []

    def ok(name, cond, why=""):
        results.append((name.split()[0], bool(cond)))
        print(("[ok] " if cond else "[FAIL] ") + name + (" -- " + why if why and not cond else ""))

    data, ids = A.load_data()
    sc = data["scenes"]
    ok("T1 soundtrack validates", len(sc) == 9 and len(data["vo"]["lines"]) == 17
       and len(data["sfx"]["cues"]) == 35
       and sum(len(l["asr_keys"]) for l in data["vo"]["lines"]) == 62
       and abs(sum(s["dur"] for s in sc) - 90.0) < 1e-9
       and abs(sc[0]["start"]) < 1e-12
       and all(abs(sc[i]["start"] - (sc[i - 1]["start"] + sc[i - 1]["dur"])) < 1e-9
               for i in range(1, len(sc))))
    probe = {k: {"clip_s": v} for k, v in PROBE_CLIPS.items()}
    caps = A.caption_times(data, {k: v["clip_s"] for k, v in probe.items()})
    want_caps = {"V02": (6.45, 9.96), "V05": (19.65, 25.36), "V10": (46.15, 49.87),
                 "V13": (65.95, 69.9), "V17": (85.95, 87.9)}
    ok("T2 caption rule", set(caps) == set(PROBE_CLIPS) - {"V01"}
       and all(caps[k] == w for k, w in want_caps.items()),
       "got %s" % sorted(caps.items()))
    try:
        A.check_fit(data, ids, probe)
        ok("T2 fit passes", True)
    except A.Refusal as e:
        ok("T2 fit passes", False, e.text)
    ok("T3 light_sfx clean", light_sfx(data) == [])
    d2 = json.loads(json.dumps(data))
    d2["sfx"]["cues"] = d2["sfx"]["cues"] + [[1.0, "impact", -4.0, "x"]]
    d2["sfx"]["files"]["impact"] = "assets/sfx/src/impact.mp3"
    r2 = light_sfx(d2)
    ok("T3 light_sfx impact", len(r2) == 1 and r2[0][1] == "impact",
       "got %s" % r2)
    d3 = json.loads(json.dumps(data))
    d3["sfx"]["cues"] = d3["sfx"]["cues"] + [[2.0, "tick", -9.0, "y"]]
    r3 = light_sfx(d3)
    ok("T3 light_sfx hot tick", len(r3) == 1 and r3[0][2] == -9.0, "got %s" % r3)
    t4a = brand_scan({"close.html": "top ... " + COLLAB + " ... end"})
    ok("T4 brand_scan one", t4a["collab_count"] == 1 and t4a["collab_in_close"]
       and t4a["banned"] == [], "got %s" % t4a)
    t4b = brand_scan({"close.html": COLLAB,
                      "title.html": "(주)이터레이션즈 · Iterations Co., Ltd."})
    ok("T4 brand_scan banned", len(t4b["banned"]) == 2 and t4b["collab_count"] == 1,
       "got %s" % t4b)
    vis = visible_text("<!-- 이터레이션즈 -->keep<script>var a=\"lift\";</script>"
                       "<style>.x{}</style><p>end</p>")
    ok("T4 visible_text", vis == "keep end", "got %r" % vis)
    vis2 = visible_text('<div class="p"><span>제공</span><img src="x.png"></div>'
                        "<div>메테오시뮬레이션 · Meteo Simulation</div>")
    ok("T4 visible_text credit", vis2 == "제공 메테오시뮬레이션 · Meteo Simulation",
       "got %r" % vis2)
    for good in ("drag", "dragon", "Cl", "lift", "양력", "x.nvdb"):
        ok("T5 matches " + good, bool(HONESTY_DRONE.search(good)))
    for bad_word in ("liftoff", "Clean", "Cp", "OpenVDB"):
        ok("T5 no match " + bad_word, not HONESTY_DRONE.search(bad_word))

    def walk(root):
        got = {}
        for rd, _d, fs in os.walk(root):
            for fn in fs:
                p = os.path.join(rd, fn)
                got[os.path.relpath(p, root)] = os.path.getsize(p)
        return got

    empty = os.path.join(tmp, "empty")
    os.makedirs(empty)
    before = walk(tmp)
    for args, code in ((["check", "--project", REPO_ROOT], "VAU-OUT"),
                       (["check", "--project", empty], "VAU-PROJECT")):
        r = subprocess.run([sys.executable, SCRIPT] + args, capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
        ok("T6 " + code, r.returncode == 2 and code + ":" in r.stdout,
           "rc %d out %r" % (r.returncode, r.stdout[-160:]))
    ok("T6 nothing written", walk(tmp) == before)
    import shutil as _sh
    _sh.rmtree(tmp, ignore_errors=True)
    by = {}
    for prefix, passed in results:
        by[prefix] = by.get(prefix, True) and passed
    n = len(by)
    k = sum(1 for v in by.values() if v)
    print("selftest %.1f s" % (time.time() - t0))
    print(("SELFTEST PASS" if k == n else "SELFTEST FAIL") + " %d/%d" % (k, n))
    return 0 if k == n else 1


def main(argv):
    if len(argv) >= 2 and argv[1] == "_whisper":
        return A.main(argv[1:])
    if len(argv) >= 2 and argv[1] == "--selftest":
        return selftest()
    import argparse
    ap = argparse.ArgumentParser(prog=TOOL, add_help=True)
    sub = ap.add_subparsers(dest="cmd")
    pv = sub.add_parser("voice")
    pv.add_argument("--project", required=True)
    pv.add_argument("--tts-python", required=True)
    pv.add_argument("--only", nargs="*")
    pb = sub.add_parser("build")
    pb.add_argument("--project", required=True)
    pa = sub.add_parser("asr")
    pa.add_argument("--project", required=True)
    pa.add_argument("--asr-python", required=True)
    pc = sub.add_parser("check")
    pc.add_argument("--project", required=True)
    a = ap.parse_args(argv[1:])
    if a.cmd == "voice":
        return cmd_voice(a)
    if a.cmd == "build":
        return cmd_build(a)
    if a.cmd == "asr":
        return cmd_asr(a)
    if a.cmd == "check":
        return cmd_check(a)
    raise Refusal("VAU-ARGS", "unknown command")


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (A.Refusal, Refusal) as e:
        print("%s: %s" % (e.code, e.text))
        sys.exit(2)
