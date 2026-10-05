#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.

"""The F1 promo v2 soundtrack: voice, mix, master and the v2 composition checks.

`voice` voices the 12 soundtrack lines with local Kokoro-82M (am_michael)
through `npx --yes hyperframes@0.8.122 tts` into <project>/assets/voice/<ID>.wav
and writes voice.json (engine, voice, per-line clip seconds, speed and sha256).
`build` mixes the film soundtrack: the licensed music bed (gained, VO-carved
and ducked), the placed VO clips, and 33 timed SFX cues (audio v3; the
synthesised "sub" kind has no cue since then), then masters to -14 LUFS /
<= -2.0 dBTP with its own
BS.1770-4 integrated-loudness meter and a 4x-oversampled true-peak limiter,
verifies the result with ffmpeg ebur128 against the gates and writes
<project>/assets/audio/master.wav (48 kHz, 24-bit, stereo, exactly 4320000
samples), the three pre-master stems and audio_meta.json.
`asr` transcribes stems/vo.wav and master.wav with faster-whisper (small,
CPU, in its own venv python given as --asr-python) and judges every line's
asr_keys. Both whole files and, per line, the [start - 0.3, end + 0.5]
excerpt cut from each of them into a temp 16 kHz mono WAV (26 wav inputs)
go through one _whisper call, and each line is judged on all the words of
its own excerpt with no time filter: faster-whisper's word timestamps drift
across a pause at segment boundaries (run 1's V07 "pressure" was stamped
0.56 s into the preceding silence, though it is spoken inside the clip), so
the whole-file window filter judged the transcript's clock instead of the
audio. asr.json keeps the whole-file transcripts under "full" and the
per-line excerpts under "lines". `check` re-judges a built, staged project
without rebuilding. `--selftest` runs T1-T9 in a temp directory, under 120 s.

Command line (P = the HyperFrames build project, outside the repository):

    python tools/promo/video_audio.py voice --project P --tts-python TTS [--only V03 V10 ...]
    python tools/promo/video_audio.py build --project P
    python tools/promo/video_audio.py asr --project P --asr-python ASR
    python tools/promo/video_audio.py check --project P
    python tools/promo/video_audio.py --selftest
    python tools/promo/video_audio.py _whisper <out.json> <wav>...   (internal, under ASR)

The data file is always tools/promo/video/audio/soundtrack.json, found
relative to this script; all build/VO times are film seconds. The signal
chain of `build`: music bed (decode -> src slice -> bed_gain_db) + VO track
(place at round(start*SR), vo_gain_db) with a presence envelope p(t) in
[0,1] (20 ms non-overlapping RMS, gate -45 dBFS, hold 0.20 s, slew limits
1/0.12 s up and 1/0.45 s down) -> the bed is carved -2 dB at 180-420 Hz and
-4.5 dB at 900-4200 Hz (Butterworth order 4, zero-phase sosfiltfilt) and
ducked -6 dB by 10^(duck_db*p/20) -> SFX bank from the cue list (kind "sub"
synthesised: an exponential 78 -> 40 Hz sine sweep over 0.9 s with envelope
exp(-t/0.35) plus a 60 Hz, 0.12 s raised-cosine thump, peak 0.9 before gain)
-> master: gain to target_lufs, then a true-peak limiter (4x resample_poly
oversample, per-sample required gain, maximum-filter hold over the 0.005 s
lookahead applied early, one-pole release 0.08 s), gain+limit repeated up to
6 times until own |I - target| < 0.05 and own TP <= ceiling; the written
master is measured with ffmpeg ebur128 and, while I misses gate_lufs or TP
misses gate_tp_dbtp, the ceiling drops 0.2 dB and step 6 redoes (at most 4).

Refusals (exit 2, message "<CODE>: <text>", nothing written by that
command): VAU-ARGS bad command line, VAU-OUT --project inside the
repository, VAU-PROJECT --project has no hyperframes.json, VAU-DATA an
invalid soundtrack.json, VAU-INPUT a missing voice clip / music / SFX file
or a music sha256 mismatch, VAU-FIT a VO clip that does not fit its slot,
VAU-TOOL ffmpeg/ffprobe/npx missing, a failing tts call, or a bad
--tts-python / --asr-python, VAU-MASTER the master still missing the gates
after the retries.

Outputs live under --project only: assets/voice/{V01..V12}.wav + voice.json,
assets/audio/master.wav, assets/audio/stems/{vo,music,sfx}.wav,
assets/audio/audio_meta.json, assets/audio/asr.json. Nothing is written
inside the repository by any command. numpy (BSD-3) and scipy (BSD-3) are
imported; ffmpeg, ffprobe, npx hyperframes (its tts runs Kokoro-82M) and
faster-whisper (in its own venv) run as separate programs. The approach
follows the user's own Shinhwa showreel audio scripts (read-only reference
copies in the build project), which are not GPL. No GPL-licensed source was
consulted. Licence: the repository's Prosperity licence - it does not
import `bpy`.
"""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

# the _whisper subcommand runs inside the ASR venv, which has faster-whisper (and
# numpy) but no scipy: skip the heavy imports there, nothing else touches them
if not (len(sys.argv) > 1 and sys.argv[1] == "_whisper"):
    import numpy as np
    from scipy.ndimage import maximum_filter1d
    from scipy.signal import butter, lfilter, resample_poly, sosfiltfilt

TOOL = "tools/promo/video_audio.py"
VERSION = "promo-video-audio/1"
SR = 48000
SCRIPT = os.path.abspath(__file__)
PROMO_DIR = os.path.dirname(SCRIPT)
DATA_FILE = os.path.join(PROMO_DIR, "video", "audio", "soundtrack.json")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT)))
VIDEO_DIR = os.path.join(PROMO_DIR, "video")


class Refusal(Exception):
    def __init__(self, code, text):
        super().__init__(f"{code}: {text}")
        self.code, self.text = code, text


def db(v):
    return 10.0 ** (v / 20.0)


def log(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------ project and data

def which(tool):
    p = shutil.which(tool)
    if not p:
        raise Refusal("VAU-TOOL", f"{tool} is not on PATH")
    return p


def resolve_project(raw):
    proj = os.path.realpath(os.path.abspath(raw))
    if proj == os.path.realpath(REPO_ROOT) or proj.startswith(os.path.realpath(REPO_ROOT) + os.sep):
        raise Refusal("VAU-OUT", "--project is inside the repository: " + raw)
    if not os.path.isfile(os.path.join(proj, "hyperframes.json")):
        raise Refusal("VAU-PROJECT", "--project has no hyperframes.json: " + raw)
    return proj


def validate_data(data):
    dur = data["duration"]
    scenes = data["scenes"]
    if not scenes or abs(scenes[0]["start"]) > 1e-9:
        raise Refusal("VAU-DATA", "scenes do not start at 0")
    t = 0.0
    kinds = set(data["sfx"]["files"])
    for s in scenes:
        if abs(s["start"] - t) > 1e-6:
            raise Refusal("VAU-DATA", f"scene {s['id']} start {s['start']} is not contiguous (expected {t})")
        if s["dur"] <= 0:
            raise Refusal("VAU-DATA", f"scene {s['id']} has non-positive dur")
        t = s["start"] + s["dur"]
    if abs(t - dur) > 1e-6:
        raise Refusal("VAU-DATA", f"scenes sum to {t}, duration is {dur}")
    ids = {s["id"]: s for s in scenes}
    prev = -1e18
    for line in data["vo"]["lines"]:
        sc = ids.get(line["scene"])
        if sc is None:
            raise Refusal("VAU-DATA", f"line {line['id']} has unknown scene {line['scene']}")
        if not (sc["start"] + 0.10 <= line["start"] <= sc["start"] + sc["dur"] - 0.10):
            raise Refusal("VAU-DATA", f"line {line['id']} start {line['start']} outside its scene {sc['id']}")
        if line["start"] <= prev:
            raise Refusal("VAU-DATA", f"line {line['id']} start {line['start']} not after previous start {prev}")
        prev = line["start"]
    for cue in data["sfx"]["cues"]:
        if not (0.0 <= cue[0] < dur):
            raise Refusal("VAU-DATA", f"cue at {cue[0]} outside [0, {dur})")
        if cue[1] not in kinds:
            raise Refusal("VAU-DATA", f"cue at {cue[0]} has unknown kind {cue[1]}")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", data["music"]["sha256"]):
        raise Refusal("VAU-DATA", "music sha256 is not 64 hex chars")
    return ids


def load_data():
    try:
        with open(DATA_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        raise Refusal("VAU-DATA", f"cannot read {DATA_FILE}: {e}")
    return data, validate_data(data)


def scene_end(ids, sid):
    s = ids[sid]
    return s["start"] + s["dur"]


# ------------------------------------------------------------------ ffmpeg / ffprobe / sha256

def run_cmd(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, **kw)


def ffprobe_duration(path):
    r = run_cmd(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "json", path])
    if r.returncode != 0:
        raise Refusal("VAU-TOOL", f"ffprobe failed on {path}: {r.stderr.decode(errors='replace')[-300:]}")
    try:
        return float(json.loads(r.stdout.decode())["format"]["duration"])
    except (KeyError, ValueError):
        raise Refusal("VAU-TOOL", f"ffprobe gives no duration for {path}")


def decode(path, ch=2):
    r = run_cmd(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le",
                 "-ac", str(ch), "-ar", str(SR), "-"])
    if r.returncode != 0 or not r.stdout:
        raise Refusal("VAU-INPUT", f"cannot decode {path}: {r.stderr.decode(errors='replace')[-300:]}")
    x = np.frombuffer(r.stdout, dtype=np.float32).astype(np.float64)
    return x.reshape(-1, ch) if ch == 2 else x


def save_wav(path, x, bits=24):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    r = run_cmd(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ac", "2", "-ar", str(SR), "-i", "-",
                 "-c:a", f"pcm_s{bits}le", path], input=x.tobytes())
    if r.returncode != 0:
        raise Refusal("VAU-TOOL", f"ffmpeg write failed for {path}: {r.stderr.decode(errors='replace')[-300:]}")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ffmpeg_measure(path):
    r = run_cmd(["ffmpeg", "-hide_banner", "-nostats", "-i", path,
                 "-af", "ebur128=peak=true:framelog=quiet", "-f", "null", "-"])
    txt = r.stderr.decode(errors="replace")
    try:
        i = float(txt.split("I:")[-1].split("LUFS")[0])
        tp = float(txt.split("Peak:")[-1].split("dBFS")[0])
        lra = float(txt.split("LRA:")[-1].split("LU")[0])
    except ValueError:
        raise Refusal("VAU-TOOL", f"cannot parse ebur128 summary for {path}")
    return i, tp, lra


# ------------------------------------------------------------------ loudness (BS.1770-4) and true peak

def _kw_filters():
    """ITU BS.1770-4 K-weighting at 48 kHz: the high-shelf then the RLB high-pass.
    The Annex 2 fixed 48 kHz coefficients (identical to what ffmpeg ebur128 applies)."""
    b1 = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285])
    a1 = np.array([1.0, -1.69065929318241, 0.73248077421585])
    b2 = np.array([1.0, -2.0, 1.0])
    a2 = np.array([1.0, -1.99004745483398, 0.99007225036621])
    return (b1, a1), (b2, a2)


(_KB1, _KA1), (_KB2, _KA2) = (None, None), (None, None)


def _ensure_kw():
    global _KB1, _KA1, _KB2, _KA2
    if _KB1 is None:
        (_KB1, _KA1), (_KB2, _KA2) = _kw_filters()



def k_weight(x):
    x = np.atleast_2d(x.T).T if x.ndim == 1 else x
    _ensure_kw()
    return lfilter(_KB2, _KA2, lfilter(_KB1, _KA1, x, axis=0), axis=0)


def lufs_int(x):
    """BS.1770-4 gated integrated loudness of stereo x at 48 kHz."""
    p = (k_weight(np.asarray(x, dtype=np.float64)) ** 2).sum(axis=1)
    blk, hop = int(0.4 * SR), int(0.1 * SR)
    if len(p) < blk:
        return -70.0
    c = np.concatenate([[0.0], np.cumsum(p)])
    starts = np.arange(0, len(p) - blk + 1, hop)
    z = (c[starts + blk] - c[starts]) / blk
    l = -0.691 + 10 * np.log10(z + 1e-20)
    z = z[l > -70.0]
    if len(z) == 0:
        return -70.0
    rel = -0.691 + 10 * np.log10(z.mean()) - 10.0
    keep = (-0.691 + 10 * np.log10(z)) > rel
    if not keep.any():
        keep = np.ones(len(z), dtype=bool)
    return float(-0.691 + 10 * np.log10(z[keep].mean()))


def true_peak_db(x):
    """4x-oversampled true peak of stereo x in dBTP."""
    m = 0.0
    for c in range(x.shape[1]):
        up = np.abs(resample_poly(x[:, c], 4, 1))
        m = max(m, float(up.max()))
    return -200.0 if m <= 0 else 20 * math.log10(m)


# ------------------------------------------------------------------ true-peak limiter and master

def _forward_max(a, window):
    ap = np.concatenate([a, np.zeros(window - 1)])
    return np.lib.stride_tricks.sliding_window_view(ap, window).max(axis=1)[: len(a)]


def limiter_env(x, ceiling_dbtp, lookahead_s, release_s):
    """Gain envelope that keeps the 4x-oversampled true peak at ceiling_dbtp:
    per-sample required gain, held (forward maximum) over the lookahead window
    so the gain is already down before the peak, one-pole released."""
    n = x.shape[0]
    ceiling = db(ceiling_dbtp)
    block = None
    for c in range(x.shape[1]):
        up = np.abs(resample_poly(x[:, c], 4, 1))[: 4 * n]
        block = up if block is None else np.maximum(block, up)
    req = np.minimum(1.0, ceiling / np.maximum(block, 1e-12))
    m = len(req) // 4
    req_n = req[: m * 4].reshape(m, 4).min(axis=1)
    if m < n:
        req_n = np.concatenate([req_n, np.ones(n - m)])
    att = 1.0 - req_n
    hold = max(1, int(round(lookahead_s * SR)))
    if hold > 1:
        att = _forward_max(att, hold)
    g = 1.0 - att
    step = 48
    a_rel = math.exp(-step / (release_s * SR))
    # block-minimum decimation: a gain dip shorter than the control step must still
    # be applied (g[::step] skips dips narrower than the step, and the skipped
    # inter-sample true-peak then survives the limiter)
    m2 = len(g) // step
    gs = g[: m2 * step].reshape(m2, step).min(axis=1)
    out = np.empty_like(gs)
    cur = 1.0
    for i, v in enumerate(gs):
        cur = v if v < cur else cur + (v - cur) * (1.0 - a_rel)
        out[i] = cur
    env = np.repeat(out, step)[:n]
    return np.minimum(env, g)


def master_chain(mix, master_cfg, ceiling_dbtp):
    """One master pass: gain to target_lufs, then up to 6 gain+limit rounds until
    own |I - target| < 0.05 and own TP <= ceiling. One round re-gains to the target
    and then limits repeatedly: gain modulation creates new inter-sample peaks in the
    4x oversampled signal, so a single limiting pass does not pin the true peak."""
    target = master_cfg["target_lufs"]
    g = target - lufs_int(mix)
    x = mix * db(g)
    iterations = 0
    i_own, tp_own = lufs_int(x), true_peak_db(x)
    for it in range(6):
        if abs(i_own - target) < 0.05 and tp_own <= ceiling_dbtp:
            break
        x = x * db(target - i_own)
        for _ in range(3):
            env = limiter_env(x, ceiling_dbtp, master_cfg["lookahead_s"], master_cfg["release_s"])
            if env.min() >= 1.0 - 1e-4:
                break
            x = x * env[:, None]
        i_own, tp_own = lufs_int(x), true_peak_db(x)
        iterations = it + 1
    return x, i_own, tp_own, g, iterations


# ------------------------------------------------------------------ presence, carve, duck

def presence(vo_stereo, gate_dbfs, hold_s, attack_s, release_s):
    """p(t) in [0,1]: 20 ms non-overlapping RMS of the VO track, sample-and-hold,
    gate runs extended by hold_s, then slew-limited (rise 1/attack_s, fall 1/release_s)."""
    mono = vo_stereo[:, 0]
    win = int(0.02 * SR)
    n_win = len(mono) // win
    rms = np.sqrt((mono[: n_win * win].reshape(n_win, win) ** 2).mean(axis=1))
    gate = (rms > db(gate_dbfs)).astype(np.float64)
    hold = int(round(hold_s / 0.02))
    if hold > 0:
        idx = np.nonzero(gate)[0]
        for i in idx:
            gate[i + 1: i + 1 + hold] = np.maximum(gate[i + 1: i + 1 + hold], 1.0)
    t = np.repeat(gate, win)
    if len(t) < len(mono):
        t = np.concatenate([t, np.full(len(mono) - len(t), t[-1] if len(t) else 0.0)])
    t = t[: len(mono)]
    step_up = 1.0 / (attack_s * SR)
    step_dn = 1.0 / (release_s * SR)
    idx = np.arange(len(t), dtype=np.float64)
    p = np.minimum.accumulate(t - step_up * idx) + step_up * idx
    p = np.maximum.accumulate(p + step_dn * idx) - step_dn * idx
    return np.clip(p, 0.0, 1.0)


def carve_duck(bed, p, bands, duck_db):
    """bed - sum_k (1 - 10^(g_k p/20)) BP_k(bed), then * 10^(duck_db p/20);
    the band-passes are Butterworth order 4, zero-phase (sosfiltfilt)."""
    out = bed.copy()
    for lo, hi, gd in bands:
        sos = butter(4, [float(lo), float(hi)], btype="band", fs=SR, output="sos")
        band = sosfiltfilt(sos, bed, axis=0)
        out -= (1.0 - db(gd) ** p)[:, None] * band
    return out * (db(duck_db) ** p)[:, None]


def synth_sub():
    """kind "sub": a sine falling exponentially 78 -> 40 Hz over 0.9 s with envelope
    exp(-t/0.35), plus a 60 Hz, 0.12 s burst with 5 ms raised-cosine fades, peak 0.9."""
    dur = 0.9
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = 78.0 * (40.0 / 78.0) ** (t / dur)
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.35)
    nb = int(0.12 * SR)
    env = np.ones(nb)
    k = int(0.005 * SR)
    env[:k] = 0.5 * (1 - np.cos(np.pi * np.arange(k) / k))
    env[-k:] = env[:k][::-1]
    y[:nb] += env * np.sin(2 * np.pi * 60.0 * np.arange(nb) / SR)
    y *= 0.9 / np.max(np.abs(y))
    return np.stack([y, y], axis=1)


# ------------------------------------------------------------------ tracks

def place(dst, src, t, gain_db):
    """add src (stereo) into dst at film time t, clipped at the film end."""
    s = np.asarray(src, dtype=np.float64) * db(gain_db)
    i = int(round(t * SR))
    if i >= len(dst) or i + len(s) <= 0:
        return
    if i < 0:
        s, i = s[-i:], 0
    j = min(len(dst), i + len(s))
    dst[i:j] += s[: j - i]


def vo_track(data, proj):
    """each clip decoded to 48 kHz mono, placed at round(start*SR), gained, mono to both."""
    n = int(round(data["duration"] * SR))
    vo = np.zeros((n, 2))
    for line in data["vo"]["lines"]:
        clip = os.path.join(proj, "assets", "voice", line["id"] + ".wav")
        x = decode(clip, ch=1) * db(data["mix"]["vo_gain_db"])
        i = int(round(line["start"] * SR))
        j = min(n, i + len(x))
        if j > i:
            vo[i:j, 0] += x[: j - i]
    vo[:, 1] = vo[:, 0]
    return vo


def music_bed(data, proj):
    """decode the music, take src [src_start, src_start+seconds), place at film_start,
    apply bed_gain_db; zero-padded to the film length."""
    m = data["music"]
    path = os.path.join(proj, m["file"])
    if not os.path.isfile(path):
        raise Refusal("VAU-INPUT", f"music file missing: {m['file']}")
    if sha256_file(path) != m["sha256"]:
        raise Refusal("VAU-INPUT", f"music sha256 differs from soundtrack.json: {m['file']}")
    x = decode(path, ch=2)
    n = int(round(data["duration"] * SR))
    b = int(round(m["src_start"] * SR))
    bed = x[b: b + n]
    if len(bed) < n:
        bed = np.concatenate([bed, np.zeros((n - len(bed), 2))])
    return bed * db(m["bed_gain_db"])


def sfx_bank(data, proj):
    """place every cue (or the synthesised "sub") into a silent stereo bank."""
    n = int(round(data["duration"] * SR))
    bank = np.zeros((n, 2))
    subs = {"sub"}
    files = {}
    for cue in data["sfx"]["cues"]:
        t, kind, gain_db = cue[0], cue[1], cue[2]
        if kind in subs:
            x = synth_sub()
        else:
            if kind not in files:
                rel = data["sfx"]["files"][kind]
                path = os.path.join(proj, rel)
                if not os.path.isfile(path):
                    raise Refusal("VAU-INPUT", f"sfx file missing for kind {kind}: {rel}")
                files[kind] = decode(path, ch=2)
            x = files[kind]
        place(bank, x, t, gain_db)
    return bank


# ------------------------------------------------------------------ voice

def load_voice_json(proj):
    path = os.path.join(proj, "assets", "voice", "voice.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"engine": "", "voice": "", "lines": []}


def read_clips(proj):
    """clip seconds per line id from voice.json (VAU-INPUT when missing)."""
    vj = load_voice_json(proj)
    clips = {l["id"]: l for l in vj.get("lines", [])}
    if not clips:
        raise Refusal("VAU-INPUT", "voice.json missing or empty (run the voice command first)")
    for line_id in clips:
        wav = os.path.join(proj, "assets", "voice", line_id + ".wav")
        if not os.path.isfile(wav):
            raise Refusal("VAU-INPUT", f"voice clip missing: {line_id}.wav")
    return clips


def check_fit(data, ids, clips):
    """C3 VAU-FIT: every clip must end before the next line - 0.05 and its scene end - 0.10."""
    lines = data["vo"]["lines"]
    for i, line in enumerate(lines):
        cs = clips.get(line["id"])
        if cs is None:
            raise Refusal("VAU-INPUT", f"voice.json has no clip for line {line['id']}")
        end = line["start"] + cs["clip_s"]
        limit = scene_end(ids, line["scene"]) - 0.10
        if i + 1 < len(lines):
            limit = min(limit, lines[i + 1]["start"] - 0.05)
        if end > limit + 1e-9:
            raise Refusal("VAU-FIT", f"{line['id']} ends {end:.3f} but must end by {limit:.3f}")


def run_tts(proj, text, voice, speed, out, tts_python):
    if not os.path.isfile(tts_python):
        raise Refusal("VAU-TOOL", "--tts-python is not a file: " + tts_python)
    cmd = (f'npx --yes hyperframes@0.8.122 tts "{text}" --voice {voice} '
           f'--speed {speed} -o "{out}"')
    env = dict(os.environ)
    env["HYPERFRAMES_PYTHON"] = tts_python
    r = subprocess.run(cmd, shell=True, cwd=proj, stdin=subprocess.DEVNULL,
                       capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    if r.returncode != 0:
        tail_lines = [l for l in (r.stderr or "").strip().splitlines() if l.strip()][-3:]
        raise Refusal("VAU-TOOL", f"tts failed for {os.path.basename(out)}: " + " | ".join(tail_lines))
    return r


def cmd_voice(args):
    proj = resolve_project(args.project)
    data, ids = load_data()
    which("npx")
    tts_python = os.path.abspath(args.tts_python)
    if not os.path.isfile(tts_python):
        raise Refusal("VAU-TOOL", "--tts-python is not a file: " + args.tts_python)
    outdir = os.path.join(proj, "assets", "voice")
    os.makedirs(outdir, exist_ok=True)
    engine = data["vo"]["engine"]
    voice = data["vo"]["voice"]
    want = set(args.only or [])
    old = {l["id"]: l for l in load_voice_json(proj).get("lines", [])}
    rows = []
    for line in data["vo"]["lines"]:
        if want and line["id"] not in want:
            if line["id"] in old:
                rows.append(old[line["id"]])
            continue
        speed = line.get("speed", data["vo"]["speed"])
        out = os.path.join(outdir, line["id"] + ".wav")
        run_tts(proj, line["tts"], voice, speed, out, tts_python)
        clip_s = ffprobe_duration(out)
        row = {"id": line["id"], "file": f"assets/voice/{line['id']}.wav", "clip_s": round(clip_s, 3),
               "speed": speed, "tts": line["tts"], "sha256": sha256_file(out)}
        rows.append(row)
        log(f"[voice] {line['id']} clip_s {clip_s:.3f} speed {speed}")
    rows.sort(key=lambda r: r["id"])
    vj = {"engine": engine, "voice": voice, "lines": rows}
    with open(os.path.join(outdir, "voice.json"), "w", encoding="utf-8") as f:
        json.dump(vj, f, indent=1, ensure_ascii=False)
    log(f"VOICE OK {len(rows)} lines")


# ------------------------------------------------------------------ build

def cmd_build(args):
    t0 = datetime.now(timezone.utc)
    proj = resolve_project(args.project)
    which("ffmpeg")
    which("ffprobe")
    data, ids = load_data()
    clips = read_clips(proj)
    check_fit(data, ids, clips)
    n = int(round(data["duration"] * SR))
    log("[build] decoding music bed")
    bed = music_bed(data, proj)
    log("[build] placing voice clips")
    vo = vo_track(data, proj)
    mix_cfg = data["mix"]
    log("[build] presence, carve and duck")
    p = presence(vo, mix_cfg["gate_dbfs"], mix_cfg["hold_s"], mix_cfg["attack_s"], mix_cfg["release_s"])
    bed_d = carve_duck(bed, p, mix_cfg["carve"], mix_cfg["duck_db"])
    log("[build] placing sfx cues")
    sfx = sfx_bank(data, proj)
    mix = bed_d + vo + sfx
    log("[build] mastering")
    out_dir = os.path.join(proj, "assets", "audio")
    os.makedirs(out_dir, exist_ok=True)
    master_cfg = data["master"]
    ceiling = master_cfg["ceiling_dbtp"]
    audio_dir = os.path.join(proj, "assets", "audio")
    tmp_wav = os.path.join(audio_dir, ".tmp_master.wav")
    lufs_i = tp_i = lra = None
    x = i_own = tp_own = g = iters = None
    for attempt in range(4):
        x, i_own, tp_own, g, iters = master_chain(mix, master_cfg, ceiling)
        save_wav(tmp_wav, x, bits=master_cfg["bits"])
        lufs_i, tp_i, lra = ffmpeg_measure(tmp_wav)
        log(f"[build] master attempt {attempt + 1}: I {lufs_i:.2f} LUFS TP {tp_i:.2f} dBTP "
            f"(gates {master_cfg['gate_lufs']} / {master_cfg['gate_tp_dbtp']})")
        if master_cfg["gate_lufs"][0] <= lufs_i <= master_cfg["gate_lufs"][1] and tp_i <= master_cfg["gate_tp_dbtp"]:
            break
        ceiling -= 0.2
    else:
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)
        raise Refusal("VAU-MASTER", f"master misses the gates after 4 attempts (last I {lufs_i:.2f} TP {tp_i:.2f})")
    master_path = os.path.join(audio_dir, "master.wav")
    if os.path.exists(master_path):
        os.remove(master_path)
    os.replace(tmp_wav, master_path)
    stems_dir = os.path.join(audio_dir, "stems")
    os.makedirs(stems_dir, exist_ok=True)
    save_wav(os.path.join(stems_dir, "vo.wav"), vo)
    save_wav(os.path.join(stems_dir, "music.wav"), bed_d)
    save_wav(os.path.join(stems_dir, "sfx.wav"), sfx)
    caps = caption_times(data, {lid: c["clip_s"] for lid, c in clips.items()})
    meta = {
        "tool": TOOL, "built": t0.isoformat(), "duration": data["duration"],
        "sample_rate": SR, "samples": n,
        "voice": {"engine": data["vo"]["engine"], "voice": data["vo"]["voice"]},
        "lines": [{"id": line["id"], "scene": line["scene"], "start": line["start"],
                   "end": round(line["start"] + clips[line["id"]]["clip_s"], 3),
                   "clip_s": clips[line["id"]]["clip_s"],
                   "speed": clips[line["id"]]["speed"], "text": line["text"], "ko": line["ko"],
                   "cap_in": caps[line["id"]][0] if line["id"] in caps else None,
                   "cap_out": caps[line["id"]][1] if line["id"] in caps else None}
                  for line in data["vo"]["lines"]],
        "music": {k: data["music"][k] for k in
                  ("file", "sha256", "heygen_audio_id", "src_start", "film_start", "seconds", "bed_gain_db")},
        "sfx": [{"t": c[0], "kind": c[1], "file": (None if c[1] == "sub" else data["sfx"]["files"][c[1]]),
                 "gain_db": c[2], "note": c[3]} for c in data["sfx"]["cues"]],
        "mix": mix_cfg,
        "master": {"file": "assets/audio/master.wav", "sha256": sha256_file(master_path),
                   "lufs_i": lufs_i, "true_peak_dbtp": tp_i, "lra": lra,
                   "own_lufs_i": i_own, "own_tp_dbtp": tp_own, "gain_db": g,
                   "ceiling_dbtp": ceiling, "iterations": iters},
        "stems": {k: sha256_file(os.path.join(stems_dir, k + ".wav")) for k in ("vo", "music", "sfx")},
    }
    with open(os.path.join(audio_dir, "audio_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    wall = (datetime.now(timezone.utc) - t0).total_seconds()
    log(f"BUILD OK master I {lufs_i:.2f} LUFS TP {tp_i:.2f} dBTP {wall:.1f} s")


# ------------------------------------------------------------------ captions (C5) and asr

def caption_times(data, clips):
    """C5: for every line with ko, cap_in = max(start-0.05, scene.start+0.10, prev.cap_out),
    cap_out = min(end+0.30, scene.end-0.10, next.start-0.10), rounded to 0.01."""
    ids = {s["id"]: s for s in data["scenes"]}
    lines = [l for l in data["vo"]["lines"] if l["ko"]]
    caps = {}
    prev_out = 0.0
    for i, line in enumerate(lines):
        sc = ids[line["scene"]]
        end = line["start"] + clips[line["id"]]
        cap_in = max(line["start"] - 0.05, sc["start"] + 0.10, prev_out)
        limit_next = lines[i + 1]["start"] - 0.10 if i + 1 < len(lines) else float("inf")
        cap_out = min(end + 0.30, sc["start"] + sc["dur"] - 0.10, limit_next)
        caps[line["id"]] = (round(cap_in, 2), round(cap_out, 2))
        prev_out = cap_out
    return caps


def _norm_word(w):
    return "".join(c for c in w.lower() if c.isalnum())


def cmd_whisper(argv):
    """internal: run under the ASR venv python; writes {basename: {text, words[]}}."""
    if len(argv) < 3:
        print("usage: _whisper <out.json> <wav>...", file=sys.stderr)
        sys.exit(2)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from faster_whisper import WhisperModel  # noqa: E402
    m = WhisperModel("small", device="cpu", compute_type="int8")
    out = {}
    for f in argv[2:]:
        segs, _ = m.transcribe(f, language="en", beam_size=5, word_timestamps=True,
                               vad_filter=False, condition_on_previous_text=False)
        words, text = [], []
        for seg in segs:
            text.append(seg.text.strip())
            for w in (seg.words or []):
                words.append({"text": w.word.strip(), "start": round(w.start, 3),
                              "end": round(w.end, 3), "p": round(w.probability, 3)})
        out[os.path.basename(f)] = {"text": " ".join(text), "words": words}
        log(f"{os.path.basename(f)}: {' '.join(text)}")
    with open(argv[1], "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)


def judge_asr(data, asr):
    """C5 bars on the per-line excerpts, each line judged on ALL the words of its
    own excerpt (no time filter): stem needs every key, master >= ceil(2/3) of
    them; the 90 % aggregate is over the master's keys. Returns (stem_ok_count,
    master_ok_count, master_keys_found, master_keys_total, report_lines)."""
    stem_n = master_n = 0
    m_found = m_keys = 0
    out = []
    for line in data["vo"]["lines"]:
        keys = line["asr_keys"]
        pair = asr.get("lines", {}).get(line["id"], {})
        for tag in ("stem", "master"):
            words = [_norm_word(w["text"]) for w in pair.get(tag, {}).get("words", [])]
            found = [k for k in keys if any(k == wn or k in wn for wn in words)]
            if tag == "stem":
                ok = len(found) == len(keys)
                stem_n += 1 if ok else 0
            else:
                ok = len(found) >= math.ceil(2 * len(keys) / 3)
                master_n += 1 if ok else 0
                m_found += len(found)
                m_keys += len(keys)
            if ok:
                out.append(f"[ok] {tag} {line['id']}")
            else:
                missing = [k for k in keys if k not in found]
                out.append(f"[FAIL] {tag} {line['id']}: missing {missing} (found {found})")
    return stem_n, master_n, m_found, m_keys, out


def cmd_asr(args):
    proj = resolve_project(args.project)
    data, ids = load_data()
    asr_python = os.path.abspath(args.asr_python)
    if not os.path.isfile(asr_python):
        raise Refusal("VAU-TOOL", "--asr-python is not a file: " + args.asr_python)
    clips = read_clips(proj)
    stem = os.path.join(proj, "assets", "audio", "stems", "vo.wav")
    master = os.path.join(proj, "assets", "audio", "master.wav")
    for path in (stem, master):
        if not os.path.isfile(path):
            raise Refusal("VAU-INPUT", f"missing {path} (run build first)")
    out_json = os.path.join(proj, "assets", "audio", "asr.json")
    which("ffmpeg")
    tmpd = tempfile.mkdtemp(prefix="vau_asr_")
    try:
        wants = []
        for line in data["vo"]["lines"]:
            end = line["start"] + clips[line["id"]]["clip_s"]
            a = max(0.0, line["start"] - 0.3)
            b = min(data["duration"], end + 0.5)
            for tag, src in (("stem", stem), ("master", master)):
                ex = os.path.join(tmpd, line["id"] + "_" + tag + ".wav")
                r = run_cmd(["ffmpeg", "-v", "error", "-y", "-ss", f"{a:.3f}", "-i", src,
                             "-t", f"{b - a:.3f}", "-ac", "1", "-ar", "16000",
                             "-c:a", "pcm_s16le", ex])
                if r.returncode != 0 or not os.path.isfile(ex):
                    raise Refusal("VAU-TOOL", f"cannot cut the {line['id']} {tag} excerpt "
                                  f"[{a:.3f}, {b:.3f}]: " + r.stderr.decode(errors="replace")[-300:])
                wants.append((line["id"], tag, ex))
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ""
        r = subprocess.run([asr_python, SCRIPT, "_whisper", out_json, stem, master]
                           + [w[2] for w in wants],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
        if r.returncode != 0:
            raise Refusal("VAU-TOOL", "whisper failed: " + (r.stderr or "")[-400:])
        with open(out_json, encoding="utf-8") as f:
            raw = json.load(f)
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)
    asr = {"full": {}, "lines": {}}
    for base in ("vo.wav", "master.wav"):
        asr["full"][base] = raw.get(base, {"text": "", "words": []})
    for lid, tag, ex in wants:
        asr["lines"].setdefault(lid, {})[tag] = raw.get(os.path.basename(ex), {"text": "", "words": []})
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(asr, f, indent=1, ensure_ascii=False)
    stem_n, master_n, found, n_keys, lines = judge_asr(data, asr)
    for l in lines:
        log(l)
    n_lines = len(data["vo"]["lines"])
    if stem_n == n_lines and master_n == n_lines and n_keys and found / n_keys >= 0.9:
        log(f"ASR PASS stem {stem_n}/{n_lines} master {master_n}/{n_lines}")
    else:
        log(f"ASR FAIL stem {stem_n}/{n_lines} master {master_n}/{n_lines} "
            f"(master keys found {found}/{n_keys} = {found / max(1, n_keys):.0%})")


# ------------------------------------------------------------------ check

HONESTY = re.compile(r"(?<![A-Za-z0-9_])C[dl](?![A-Za-z0-9_])|drag|downforce|항력|양력|다운포스", re.I)
AUDIO_EL = ('<audio id="el-master" src="assets/audio/master.wav" data-start="0" '
            'data-duration="90" data-track-index="5" data-volume="1"></audio>')


def read_file(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def slot_times(html, sid):
    m = re.search(r'<div[^>]*id="el-' + re.escape(sid) + r'"[^>]*>', html)
    if not m:
        return None, None
    tag = m.group(0)
    s = re.search(r'data-start="(-?[\d.]+)"', tag)
    d = re.search(r'data-duration="(-?[\d.]+)"', tag)
    return (float(s.group(1)) if s else None, float(d.group(1)) if d else None)


def wrap_fade(html, sid):
    """the last tl.to("#<sid>-wrap", {..opacity: 0..}, X): returns (duration, position)."""
    pat = re.compile(r'tl\.to\("#' + re.escape(sid) + r'-wrap",\s*\{([^}]*)\}\s*,\s*([\d.]+)\s*\)')
    hits = [h for h in pat.finditer(html) if re.search(r'opacity:\s*0', h.group(1))]
    if not hits:
        return None, None
    h = hits[-1]
    d = re.search(r'duration:\s*([\d.]+)', h.group(1))
    return (float(d.group(1)) if d else None), float(h.group(2))


def video_duration(html):
    m = re.search(r'<video[^>]*data-duration="([\d.]+)"', html)
    return float(m.group(1)) if m else None


def caps_block(html):
    m = re.search(r'var\s+CAPS\s*=\s*\[(.*?)\];', html, re.S)
    rows = re.findall(r'\[\s*"([A-Za-z0-9]+)"\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]', m.group(1)) if m else []
    return [(r[0], float(r[1]), float(r[2])) for r in rows]


def ffprobe_format(path):
    r = run_cmd(["ffprobe", "-v", "error", "-show_entries",
                 "stream=codec_name,sample_rate,channels,nb_samples:format=duration",
                 "-of", "json", path])
    if r.returncode != 0:
        raise Refusal("VAU-TOOL", f"ffprobe failed on {path}")
    j = json.loads(r.stdout.decode())
    st = j["streams"][0]
    n = int(st.get("nb_samples") or 0)
    if not n:
        try:
            n = int(round(float(j["format"]["duration"]) * SR))
        except (KeyError, ValueError):
            n = 0
    return (st.get("codec_name"), int(st.get("sample_rate", 0)), int(st.get("channels", 0)), n)


def cmd_check(args):
    proj = resolve_project(args.project)
    which("ffmpeg")
    which("ffprobe")
    data, ids = load_data()
    items = []

    def add(name, ok, why=""):
        items.append((name, bool(ok), why))

    try:
        validate_data(json.loads(read_file(DATA_FILE)))
        add("soundtrack", True)
    except Refusal as e:
        add("soundtrack", False, e.text)
    clips = {}
    try:
        clips = read_clips(proj)
    except Refusal as e:
        add("voice", False, e.text)
    vj = load_voice_json(proj)
    vrows = {l["id"]: l for l in vj.get("lines", [])}
    for line in data["vo"]["lines"]:
        wav = os.path.join(proj, "assets", "voice", line["id"] + ".wav")
        row = vrows.get(line["id"])
        if not os.path.isfile(wav) or row is None:
            add(f"voice {line['id']}", False, "file or voice.json row missing")
        elif row.get("sha256") != sha256_file(wav):
            add(f"voice {line['id']}", False, "sha256 differs from voice.json")
        else:
            add(f"voice {line['id']}", True)
    if clips:
        try:
            check_fit(data, ids, clips)
            add("fit", True)
        except Refusal as e:
            add("fit", False, e.text)
    else:
        add("fit", False, "no voice.json")
    master_path = os.path.join(proj, "assets", "audio", "master.wav")
    meta_path = os.path.join(proj, "assets", "audio", "audio_meta.json")
    meta = None
    if os.path.isfile(meta_path):
        try:
            meta = json.loads(read_file(meta_path))
        except ValueError:
            meta = None
    if not os.path.isfile(master_path):
        add("master format", False, "master.wav missing")
        add("master sha", False, "master.wav missing")
        add("master lufs", False, "master.wav missing")
        add("master tp", False, "master.wav missing")
    else:
        codec, sr, ch, samples = ffprobe_format(master_path)
        n_exp = int(round(data["duration"] * SR))
        ok = (codec, sr, ch, samples) == ("pcm_s24le", SR, 2, n_exp)
        add("master format", ok, f"got {codec} {sr} Hz {ch} ch {samples} samples")
        sha = sha256_file(master_path)
        add("master sha", bool(meta and meta.get("master", {}).get("sha256") == sha),
            "differs from audio_meta.json" if meta else "audio_meta.json missing")
        i, tp, _ = ffmpeg_measure(master_path)
        gl = data["master"]["gate_lufs"]
        add("master lufs", gl[0] <= i <= gl[1], f"I {i:.2f} LUFS outside {gl}")
        add("master tp", tp <= data["master"]["gate_tp_dbtp"], f"TP {tp:.2f} dBTP above gate")
    index_path = os.path.join(proj, "index.html")
    if not os.path.isfile(index_path):
        add("index duration", False, "index.html missing (run stage first)")
    else:
        index = read_file(index_path)
        root = re.search(r'<div[^>]*id="root"[^>]*>', index)
        dm = re.search(r'data-duration="(-?[\d.]+)"', root.group(0)) if root else None
        add("index duration", bool(dm and abs(float(dm.group(1)) - data["duration"]) < 0.001),
            f"root data-duration {dm.group(1) if dm else 'missing'}")
        for sc in data["scenes"]:
            s, d = slot_times(index, sc["id"])
            ok = s is not None and abs(s - sc["start"]) < 0.001 and abs(d - sc["dur"]) < 0.001
            add(f"index {sc['id']}", ok, f"slot {s}/{d}, want {sc['start']}/{sc['dur']}")
        cs, cd = slot_times(index, "captions")
        add("index captions", abs(cs) < 0.001 and abs(cd - data["duration"]) < 0.001,
            f"captions slot {cs}/{cd}")
        add("index audio", AUDIO_EL in index, "audio element not found verbatim")
        comp_dir = os.path.join(proj, "compositions")
        for sc in data["scenes"]:
            path = os.path.join(comp_dir, sc["id"] + ".html")
            if not os.path.isfile(path):
                add(f"fade {sc['id']}", False, "composition missing")
                continue
            d, x = wrap_fade(read_file(path), sc["id"])
            ok = d is not None and abs((x or 0) + d - sc["dur"]) < 0.001
            add(f"fade {sc['id']}", ok, f"fade {x}+{d} != {sc['dur']}")
        for fname in sorted(os.listdir(comp_dir)):
            if not fname.endswith(".html"):
                continue
            html = read_file(os.path.join(comp_dir, fname))
            vd = video_duration(html)
            if vd is None:
                continue
            sid = fname[:-5]
            sc = next((s for s in data["scenes"] if s["id"] == sid), None)
            if sc is None:
                add(f"video {sid}", False, "no such scene")
            else:
                add(f"video {sid}", abs(vd - sc["dur"]) < 0.001, f"video {vd} != {sc['dur']}")
        cap_html = read_file(os.path.join(comp_dir, "captions.html"))
        got = caps_block(cap_html)
        want_rows = caption_times(data, {lid: c["clip_s"] for lid, c in clips.items()})
        want = [(lid, t[0], t[1]) for lid, t in sorted(want_rows.items()) if lid != "V01"]
        ok = len(got) == len(want) and all(
            g[0] == w[0] and abs(g[1] - w[1]) <= 0.02 and abs(g[2] - w[2]) <= 0.02
            for g, w in zip(got, want))
        add("captions times", ok, f"CAPS {got} want {want}")
        for lid, t_in, t_out in want:
            m = re.search(r'id="captions-' + re.escape(lid) + r'-k">(.*?)</div>', cap_html, re.S)
            text = m.group(1) if m else None
            ko_html = None
            for line in data["vo"]["lines"]:
                if line["id"] == lid:
                    ko_html = (line["ko"] or "").replace("\n", "<br />")
            add(f"captions text {lid}", text == ko_html, f"{text!r} != {ko_html!r}")
        cin = re.search(r'tl\.fromTo\("#captions-chip".*?,\s*([\d.]+)\s*\)\s*;', cap_html, re.S)
        cout = re.findall(r'tl\.to\("#captions-chip",\s*\{([^}]*)\}\s*,\s*([\d.]+)\s*\)', cap_html)
        c_ok = bool(cin and abs(float(cin.group(1)) - 47.7) < 0.001
                    and cout and abs(float(cout[-1][1]) - 71.4) < 0.001
                    and 'opacity: 0' in cout[-1][0].replace('opacity:0', 'opacity: 0'))
        add("captions chip", c_ok, f"chip in {cin.group(1) if cin else '?'}, out {cout[-1][1] if cout else '?'}")
        bad = []
        for root_dir, _, files in os.walk(VIDEO_DIR):
            for fn in files:
                if fn.endswith(".html") or fn == "soundtrack.json":
                    p = os.path.join(root_dir, fn)
                    m2 = HONESTY.search(read_file(p))
                    if m2:
                        bad.append(f"{os.path.relpath(p, REPO_ROOT)}:{m2.group(0)}")
        add("honesty", not bad, "; ".join(bad[:5]))
    for name, ok, why in items:
        log(f"[{'ok' if ok else 'FAIL'}] {name}" + (f": {why}" if why and not ok else ""))
    n = len(items)
    k = sum(1 for _, ok, _ in items if ok)
    log(("CHECK PASS" if k == n else "CHECK FAIL") + f" {k}/{n}")
    return 0 if k == n else 1


# ------------------------------------------------------------------ selftest

def selftest():
    import time
    t0 = time.time()
    tmp = tempfile.mkdtemp(prefix="vau_selftest_")
    results = []

    def ok(name, cond, why=""):
        results.append((name.split()[0], bool(cond)))
        log(f"[{'ok' if cond else 'FAIL'}] {name}" + (f" -- {why}" if why and not cond else ""))

    data, ids = load_data()
    sc = data["scenes"]
    ok("T1 soundtrack validates", len(sc) == 10 and len(data["vo"]["lines"]) == 12
       and len(data["sfx"]["cues"]) == 33
       and sum(len(l["asr_keys"]) for l in data["vo"]["lines"]) == 41
       and abs(sum(s["dur"] for s in sc) - 90.0) < 1e-9
       and abs(sc[0]["start"]) < 1e-12
       and all(abs(sc[i]["start"] - (sc[i - 1]["start"] + sc[i - 1]["dur"])) < 1e-9
               for i in range(1, len(sc))))

    t = np.arange(10 * SR) / SR
    sine = 0.1 * np.sin(2 * np.pi * 1000.0 * t)
    x = np.stack([sine, sine], axis=1)
    own = lufs_int(x)
    ok("T2 own lufs -20.00", abs(own - (-20.0)) <= 0.10, f"own {own:.3f} LUFS")
    p2 = os.path.join(tmp, "sine24.wav")
    save_wav(p2, x)
    i_ff, _, _ = ffmpeg_measure(p2)
    ok("T2 ffmpeg agrees", abs(i_ff - own) <= 0.2, f"ffmpeg {i_ff:.3f} vs own {own:.3f}")

    t = np.arange(2 * SR) / SR
    sine4 = 0.5 * np.sin(np.pi * np.arange(2 * SR) / 2 + np.pi / 4)
    x4 = np.stack([sine4, sine4], axis=1)
    spk = 20 * math.log10(np.abs(x4).max())
    tpk = true_peak_db(x4)
    ok("T3 sample peak", abs(spk - (-9.03)) <= 0.05, f"{spk:.3f} dBFS")
    ok("T3 own true peak", abs(tpk - (-6.02)) <= 0.3, f"{tpk:.3f} dBTP")

    rng = np.random.default_rng(20261005)
    n4 = 20 * SR
    noise = rng.standard_normal(n4)
    noise *= db(-30.0) / noise.std()
    y = np.repeat(noise, 2).reshape(n4, 2).copy()
    for k in range(4, 20, 4):
        i0 = k * SR
        seg = rng.standard_normal(int(0.5 * SR))
        seg *= db(-6.0) / seg.std()
        y[i0: i0 + int(0.5 * SR)] = np.stack([seg, seg], axis=1)
    mx, i_o, tp_o, g_db, iters = master_chain(y, data["master"], data["master"]["ceiling_dbtp"])
    p4 = os.path.join(tmp, "noise_master.wav")
    save_wav(p4, mx)
    i_ff, tp_ff, _ = ffmpeg_measure(p4)
    ok("T4 master lufs", -14.3 <= i_ff <= -13.7, f"I {i_ff:.3f} LUFS (own {i_o:.3f})")
    ok("T4 master tp", tp_ff <= -1.9, f"TP {tp_ff:.3f} dBTP (own {tp_o:.3f}, iters {iters})")

    n5 = 6 * SR
    t = np.arange(n5) / SR
    bed = (0.05 * np.sin(2 * np.pi * 300.0 * t) + 0.05 * np.sin(2 * np.pi * 2000.0 * t)
           + 0.05 * np.sin(2 * np.pi * 8000.0 * t))
    bed = np.stack([bed, bed], axis=1)
    vo = np.zeros((n5, 2))
    i0, i1 = int(2.0 * SR), int(4.0 * SR)
    tv = np.arange(i1 - i0) / SR
    tone = 0.1 * np.sin(2 * np.pi * 440.0 * tv)
    vo[i0:i1, 0] = tone
    vo[i0:i1, 1] = tone
    p = presence(vo, -45.0, 0.2, 0.12, 0.45)
    out = carve_duck(bed, p, [[180, 420, -2.0], [900, 4200, -4.5]], -6.0)

    def proj(arr, f, a, b):
        seg = arr[int(a * SR): int(b * SR), 0]
        tt = np.arange(len(seg)) / SR
        return 2 * abs(np.mean(seg * np.exp(-2j * np.pi * f * tt)))
    for f, want, a, b, tol in ((300.0, -8.0, 2.5, 3.5, 0.3), (2000.0, -10.5, 2.5, 3.5, 0.3),
                               (8000.0, -6.0, 2.5, 3.5, 0.3), (300.0, 0.0, 0.2, 1.2, 0.2),
                               (2000.0, 0.0, 0.2, 1.2, 0.2), (8000.0, 0.0, 0.2, 1.2, 0.2)):
        got = 20 * math.log10(proj(out, f, a, b) / proj(bed, f, a, b))
        ok(f"T5 carve {f:.0f} Hz {a}-{b}", abs(got - want) <= tol, f"{got:.2f} dB, want {want} +- {tol}")

    bank = np.zeros((2 * SR, 2))
    click = np.zeros(100)
    click[0] = 1.0
    place(bank, np.stack([click, click], axis=1), 1.2345, 0.0)
    nz = np.nonzero(bank[:, 0])[0]
    ok("T6 click placement", len(nz) and nz[0] == 59256, f"first sample {nz[0] if len(nz) else None}")

    def refusal_case(name, script, cmd_list, files_before, root):
        r = subprocess.run([sys.executable, script] + cmd_list, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        files_after = walk_files(root)
        ok(name, r.returncode == 2 and files_after == files_before,
           f"rc {r.returncode} files_same {files_after == files_before} err {r.stderr[-160:]}")

    def walk_files(root):
        got = {}
        for rd, _, fs in os.walk(root):
            for fn in fs:
                p = os.path.join(rd, fn)
                got[os.path.relpath(p, root)] = os.path.getsize(p)
        return got

    empty = os.path.join(tmp, "empty"); os.makedirs(empty, exist_ok=True)
    refusal_case("T7 VAU-OUT", SCRIPT, ["check", "--project", REPO_ROOT], walk_files(tmp), tmp)
    refusal_case("T7 VAU-PROJECT", SCRIPT, ["check", "--project", empty], walk_files(tmp), tmp)
    t2 = os.path.join(tmp, "vau_data")
    os.makedirs(os.path.join(t2, "tools", "promo", "video", "audio"))
    os.makedirs(os.path.join(t2, "proj"))
    shutil.copy(SCRIPT, os.path.join(t2, "tools", "promo", "video_audio.py"))
    d2 = json.loads(read_file(DATA_FILE))
    v04 = next(l for l in d2["vo"]["lines"] if l["id"] == "V04")
    v05 = next(l for l in d2["vo"]["lines"] if l["id"] == "V05")
    v05["start"] = v04["start"]
    with open(os.path.join(t2, "tools", "promo", "video", "audio", "soundtrack.json"), "w", encoding="utf-8") as f:
        json.dump(d2, f)
    with open(os.path.join(t2, "proj", "hyperframes.json"), "w", encoding="utf-8") as f:
        f.write("{}")
    before = walk_files(tmp)
    refusal_case("T7 VAU-DATA", os.path.join(t2, "tools", "promo", "video_audio.py"),
                 ["check", "--project", os.path.join(t2, "proj")], before, tmp)

    probe = {"V01": 3.904, "V02": 6.101, "V03": 6.187, "V04": 5.760, "V05": 5.739,
             "V06": 9.045, "V07": 2.923, "V08": 4.800, "V09": 4.885, "V10": 4.096,
             "V11": 3.051, "V12": 1.579}
    caps = caption_times(data, probe)
    want_caps = {"V03": (14.05, 20.59), "V10": (72.45, 76.60),
                 "V11": (76.65, 79.80), "V12": (79.85, 81.78)}
    ok("T8 caption rule", set(caps) == {f"V{i:02d}" for i in range(2, 13)} and all(
        caps[k] == w for k, w in want_caps.items()),
       f"got {sorted(caps.items())}")

    # T9: the pure excerpt judge (no whisper) - the excerpt words ARE the judged words.
    def synth_words(keys):
        return {"words": [{"text": k, "start": 0.0, "end": 0.1, "p": 1.0} for k in keys]}

    full = {"lines": {l["id"]: {"stem": synth_words(l["asr_keys"]),
                                "master": synth_words(l["asr_keys"])}
                      for l in data["vo"]["lines"]}}
    s1, m1, f1, k1, _ = judge_asr(data, full)
    ok("T9 judge all keys", (s1, m1, f1, k1) == (12, 12, 41, 41),
       f"stem {s1} master {m1} keys {f1}/{k1}")
    drop = json.loads(json.dumps(full))
    drop["lines"]["V07"]["stem"]["words"] = [w for w in drop["lines"]["V07"]["stem"]["words"]
                                             if w["text"] != "pressure"]
    s2, m2, _, _, _ = judge_asr(data, drop)
    ok("T9 judge one missing", (s2, m2) == (11, 12), f"stem {s2} master {m2} (2/3 remain -> master ok)")
    sub = json.loads(json.dumps(full))
    sub["lines"]["V10"]["stem"]["words"] = [{"text": "meteor"}, {"text": "aidriven"},
                                            {"text": "solver"}]
    s3, _, _, _, r3 = judge_asr(data, sub)
    ok("T9 judge substring", s3 == 12 and "[ok] stem V10" in r3, f"stem {s3}")
    shutil.rmtree(tmp, ignore_errors=True)
    by = {}
    for prefix, passed in results:
        by[prefix] = by.get(prefix, True) and passed
    n = len(by)
    k = sum(1 for v in by.values() if v)
    log(f"selftest {time.time() - t0:.1f} s")
    log(("SELFTEST PASS" if k == n else "SELFTEST FAIL") + f" {k}/{n}")
    return 0 if k == n else 1


def main(argv):
    if len(argv) >= 2 and argv[1] == "_whisper":
        cmd_whisper(argv[1:])
        return 0
    if len(argv) >= 2 and argv[1] == "--selftest":
        return selftest()
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
    except Refusal as e:
        log(f"{e.code}: {e.text}")
        sys.exit(2)
