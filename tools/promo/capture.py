#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.

"""The F1 promo studio capture: `run` boots the GUI studio on a workspace
built by tools/promo/studio_ws.py, drives its chat with Korean requests in a
headless real-GPU Chromium, and records 1920x1080 clips, stills and a
shots.json/capture.json pair.

Subcommands:
  run         --ws WS_DIR --gui-tree DIR --out DIR [--llm zai|mock]
               [--api-port 8799] [--web-port 5183] [--reduced]
               [--turn-timeout 360] [--script FILE.json]
  --selftest  runs T7-T14 (no browser, no server), < 30 s.

Layout under --out: frames/<shot id>/f_%05d.jpg screencast frames (a seed
screenshot at t = 0, then at most one kept frame per 1/15 s, every CDP frame
acked), frames/<shot id>/frames.csv and list.txt (the ffmpeg concat list),
clips/<shot id>.mp4 (1920x1080 H.264 yuv420p at 30 fps), stills/<shot id>.png,
server.log and web.log, shots.json + capture.json, and the CC BY 4.0 model's
LICENSE.txt copied beside them.

Before the first shot the capture preselects the finished run and reloads the
page (dev React StrictMode drops the run subscription, so a click on a
finished run alone leaves the residual chart and log empty), then clears a
restored stored session so the chat starts empty; either step failing is
recorded in capture.json, never refused.

Refusals (exit 2, before any process starts unless said otherwise): CAP-PORT
(a forbidden or already-listening port), CAP-WS (the workspace is not a
studio_ws.py output), CAP-OUT (--out inside the repository or the gui tree),
CAP-SCRIPT (an invalid shot list), CAP-NODE (the gui tree's tsx/vite are
missing - nothing is ever installed), CAP-HEALTH (the two servers are not
ready within 180 s), CAP-LLM (hello.llm is not the requested one). The two
servers are stopped by their own PIDs on every exit path once started, and
the refusal is printed after the stop.

The mesh and geometry shown derive from a CC BY 4.0 model ("F1 2026
concept" by Qvist_Designs, via Sketchfab), so NOTHING made from them is
written inside the repository: every output goes under --out, which must
lie outside the repository and the gui tree, with the model's LICENSE.txt
copied beside it. Playwright (Apache-2.0) drives the browser as a library;
ffmpeg is run as a separate program.
"""

import argparse
import base64
import json
import os
import re
import shutil
import socket
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tunnel_mesh import refuse  # noqa: E402

from playwright.sync_api import sync_playwright  # noqa: E402

TOOL = "tools/promo/capture.py"
VERSION = "promo-capture/1"
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VIEWPORT = (1920, 1080)
CHROME_ARGS = ["--enable-gpu", "--ignore-gpu-blocklist", "--use-angle=d3d11"]
FPS_CAP = 15            # screencast frames kept at most every 1/15 s
CLIP_FPS = 30
JPEG_QUALITY = 92
TYPE_DELAY_MS = 45
SETTLE_S = 3.0
ORBIT_PX = 360
ORBIT_STEPS = 90
ORBIT_S = 6.0
HOLD_S = 2.0
FORBIDDEN_PORTS = (8787, 5173, 5180)


SHOTS = [
    {"id": "s0_studio", "kind": "hold", "seconds": 4,
     "shows": "the studio opened in Korean, empty chat"},
    {"id": "s1_geometry", "kind": "prompt", "orbit": True,
     "text": "geometry/f1_sim.stl 에 있는 F1 레이스카 형상을 열어서 보여줘.",
     "shows": "chat answering; the F1 sim surface in the 3-D view"},
    {"id": "s2_mesh", "kind": "prompt", "orbit": True,
     "text": "cases/f1-promo 해석 격자를 3D 뷰어에 띄워서 차량 표면의 격자선(Surface + Edges)을 보여줘. 터널 벽은 클립 박스로 잘라내 줘.",
     "shows": "chat answering; the 3.2 M-cell castellated mesh surface of the car with its edges"},
    {"id": "s3_cp", "kind": "prompt", "orbit": True,
     "text": "차량 표면을 압력계수 Cp 로 색칠해줘. coolwarm 색상, 범위 -1 ~ 1, 격자선 없이 Surface 로 보여줘.",
     "shows": "chat answering; surface Cp colouring on the car in the result view"},
    {"id": "s4_drag", "kind": "prompt", "orbit": False,
     "text": "잔차 수렴 그래프를 보여주고, cases/f1-promo/post/numbers.json 에서 항력과 Cd 를 알려줘.",
     "shows": "chat answering with the drag numbers; the residual chart"},
    {"id": "s5_residuals", "kind": "click_run", "run_id": "promo-solve-full", "seconds": 8,
     "shows": "the residual chart of the imported 1000-step solve (a scripted click, not the assistant)"},
]
REDUCED_SHOTS = [
    {"id": "r0_studio", "kind": "hold", "seconds": 3, "shows": "the studio opened in Korean"},
    {"id": "r1_hello", "kind": "prompt", "orbit": False, "turn_timeout": 60,
     "text": "안녕하세요", "shows": "the mock assistant answering"},
    {"id": "r2_residuals", "kind": "click_run", "run_id": "promo-solve-full", "seconds": 4,
     "shows": "the residual chart of the imported solve"},
]


# --------------------------------------------------------- pure: C9/C7/C6

def keep_frame(t, last_t, fps_cap):
    """The FPS cap: keep the first frame, then one per 1/fps_cap s."""
    if last_t is None:
        return True
    return (t - last_t) >= 1.0 / fps_cap


def turn_done(stop_visible, send_visible):
    """C13: a prompt turn is over once the stop button is gone and the send
    button is back (a turn may end before stop-btn is ever seen)."""
    return (not stop_visible) and bool(send_visible)


def concat_list(times, names, t_end):
    """The ffmpeg concat list: file/duration pairs with the last file
    repeated once, every duration clamped to >= 1/CLIP_FPS."""
    out = []
    mind = 1.0 / CLIP_FPS
    for i in range(len(times)):
        d = (times[i + 1] - times[i]) if i + 1 < len(times) else (t_end - times[i])
        if d < mind:
            d = mind
        out.append("file '%s'" % names[i])
        out.append("duration %.6f" % d)
    out.append("file '%s'" % names[-1])
    return "\n".join(out) + "\n"


def ffmpeg_args(list_name, out_path, t_end):
    """The C9 ffmpeg command for one clip, cut to t_end: this ffmpeg (9.0.2)
    honours the last duration line AND shows the repeated last file line
    again for it, so the output is trimmed with -t."""
    return ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f",
            "concat", "-safe", "0", "-i", list_name, "-vf",
            "fps=30,format=yuv420p", "-c:v", "libx264", "-crf", "16",
            "-preset", "medium", "-movflags", "+faststart",
            "-t", "%.6f" % t_end, out_path]


def png_size(path):
    """(w, h) from a PNG's IHDR, without PIL."""
    with open(path, "rb") as f:
        d = f.read(33)
    if d[:8] != b"\x89PNG\r\n\x1a\n" or d[12:16] != b"IHDR":
        raise ValueError(path + ": not a PNG")
    return int.from_bytes(d[16:20], "big"), int.from_bytes(d[20:24], "big")


def jpeg_size(path):
    """(w, h) from the first SOF0/SOF2 marker."""
    with open(path, "rb") as f:
        d = f.read()
    if d[:2] != b"\xff\xd8":
        raise ValueError(path + ": not a JPEG")
    i = 2
    while i + 9 < len(d):
        if d[i] != 0xFF:
            i += 1
            continue
        m = d[i + 1]
        if m == 0x01 or m == 0xD8 or 0xD0 <= m <= 0xD7:
            i += 2
            continue
        seg = int.from_bytes(d[i + 2:i + 4], "big")
        if m in (0xC0, 0xC2):
            h = int.from_bytes(d[i + 5:i + 7], "big")
            w = int.from_bytes(d[i + 7:i + 9], "big")
            return w, h
        i += 2 + seg
    raise ValueError(path + ": no SOF0/SOF2 marker")


def validate_shots(shots):
    """Problems of C7's shot shape; empty means valid."""
    if not isinstance(shots, list):
        return ["the shot list is not a JSON list"]
    problems = []
    seen = {}
    for i, s in enumerate(shots):
        where = "shot %d" % i
        if not isinstance(s, dict):
            problems.append(where + " is not an object")
            continue
        sid = s.get("id")
        if not isinstance(sid, str) or not re.fullmatch(r"[a-z0-9_]+", sid):
            problems.append(where + ": id must match [a-z0-9_]+")
        else:
            seen[sid] = seen.get(sid, 0) + 1
        kind = s.get("kind")
        if kind not in ("hold", "prompt", "click_run"):
            problems.append(where + ": unknown kind %r" % (kind,))
            continue
        if not isinstance(s.get("shows"), str) or not s.get("shows"):
            problems.append(where + ": shows must be non-empty text")
        if kind == "hold":
            if not _pos_num(s.get("seconds")):
                problems.append(where + ": hold needs seconds > 0")
        elif kind == "prompt":
            if not isinstance(s.get("text"), str) or not s.get("text"):
                problems.append(where + ": prompt needs non-empty text")
            if "orbit" in s and not isinstance(s["orbit"], bool):
                problems.append(where + ": orbit must be a bool")
            if "turn_timeout" in s and not _pos_num(s["turn_timeout"]):
                problems.append(where + ": turn_timeout must be > 0")
        else:
            if not isinstance(s.get("run_id"), str) or not s.get("run_id"):
                problems.append(where + ": click_run needs run_id")
            if not _pos_num(s.get("seconds")):
                problems.append(where + ": click_run needs seconds > 0")
    for sid, n in seen.items():
        if n > 1:
            problems.append("duplicate id " + sid)
    return problems


def _pos_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0


def server_env(base_env, ws, api_port, llm):
    """The server process env: no ZAI_API_KEY, the workspace pointed at the
    built tree, the runs state at <ws>/state."""
    env = dict(base_env)
    env.pop("ZAI_API_KEY", None)
    env["CFD_PORT"] = str(api_port)
    env["CFD_WORKSPACE"] = os.path.join(ws, "ws")
    env["CFD_GUI_DIR"] = os.path.join(ws, "state")
    env["CFD_LLM"] = llm
    env["CFD_PYTHON"] = "python"
    return env


# ------------------------------------------------------- checks and servers

class CaptureRefusal(Exception):
    """A refusal raised after the servers started; printed after the stop."""

    def __init__(self, code, text):
        super().__init__(code + ": " + text)
        self.code = code
        self.text = text


def _posix(path):
    return os.path.abspath(path).replace(os.sep, "/")


def is_inside(child, parent):
    c = os.path.normcase(os.path.abspath(child))
    p = os.path.normcase(os.path.abspath(parent))
    if c == p:
        return True
    return c.startswith(p.rstrip("/" + os.sep) + os.sep)


def tcp_open(port):
    s = socket.socket()
    s.settimeout(0.5)
    try:
        s.connect(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def _get_json(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _post_json(url, obj, timeout=30):
    req = urllib.request.Request(url, data=json.dumps(obj).encode("utf-8"),
                                 headers={"content-type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def check_ports(api_port, web_port):
    for p in (api_port, web_port):
        if p in FORBIDDEN_PORTS:
            refuse("CAP-PORT", "port %d is forbidden" % p)
        if tcp_open(p):
            refuse("CAP-PORT", "port %d already has a listener" % p)


def check_ws(ws):
    for rel in ("ws/docs/schema/case-1.json", "ws/cases/f1-promo",
                "state/runs", "studio_ws.json"):
        p = os.path.join(ws, rel)
        if not os.path.exists(p):
            refuse("CAP-WS", p + " is missing (is --ws a studio_ws.py out?)")


def check_out(out, gui_tree):
    if is_inside(out, REPO) or is_inside(out, gui_tree):
        refuse("CAP-OUT", _posix(out) + " lies inside the repository or the gui tree")


def check_script(shots):
    problems = validate_shots(shots)
    if problems:
        refuse("CAP-SCRIPT", "; ".join(problems))


def check_node(gui_tree):
    for f in ("gui/node_modules/.bin/tsx.cmd", "gui/node_modules/.bin/vite.cmd"):
        p = os.path.join(gui_tree, f)
        if not os.path.isfile(p):
            refuse("CAP-NODE", p + " is missing (nothing is ever installed)")


def start_servers(ws, gui_tree, out, api_port, web_port, llm):
    flags = subprocess.CREATE_NEW_PROCESS_GROUP
    cwd = os.path.join(gui_tree, "gui")
    log_s = open(os.path.join(out, "server.log"), "ab")
    log_w = open(os.path.join(out, "web.log"), "ab")
    server = subprocess.Popen(
        ["npx.cmd", "tsx", "server/src/main.ts"], cwd=cwd,
        stdout=log_s, stderr=subprocess.STDOUT,
        env=server_env(os.environ, ws, api_port, llm), creationflags=flags)
    web = subprocess.Popen(
        ["npm.cmd", "run", "dev", "-w", "web"], cwd=cwd,
        stdout=log_w, stderr=subprocess.STDOUT,
        env={**os.environ, "CFD_PORT": str(api_port),
             "CFD_WEB_PORT": str(web_port)}, creationflags=flags)
    return {"server": server, "web": web}


def wait_ready(api_port, web_port, deadline_s=180):
    """Both ready -> the seconds it took; else raise CAP-HEALTH."""
    t0 = time.time()
    api_ok = web_ok = False
    while time.time() - t0 < deadline_s:
        if not api_ok:
            try:
                api_ok = bool(_get_json(
                    "http://127.0.0.1:%d/api/health" % api_port, 3).get("ok"))
            except Exception:
                pass
        if api_ok and not web_ok:
            try:
                with urllib.request.urlopen(
                        "http://127.0.0.1:%d/" % web_port, timeout=3) as r:
                    web_ok = (r.status == 200)
            except Exception:
                pass
        if api_ok and web_ok:
            return time.time() - t0
        time.sleep(1.0)
    raise CaptureRefusal("CAP-HEALTH",
                         "the servers were not ready within %d s (api %s, web %s)"
                         % (deadline_s, api_ok, web_ok))


def prewarm(api_port):
    """Open the dataset and the geometry before the browser opens; a
    failure is recorded, not refused."""
    info = {"dataset_status": None, "dataset_seconds": None,
            "geometry_ok": False, "error": None}
    base = "http://127.0.0.1:%d" % api_port
    try:
        t0 = time.time()
        r = None
        for attempt in (0, 1):
            try:
                r = _post_json(base + "/api/datasets/open",
                               {"path": "cases/f1-promo"}, 60)
                break
            except urllib.error.HTTPError:
                if attempt == 0:
                    time.sleep(2.0)
                else:
                    raise
        ds_id = (r or {}).get("datasetId")
        if not ds_id:
            raise RuntimeError("no datasetId in the open response")
        while time.time() - t0 < 300:
            # GET /api/datasets/:id returns the manifest (no status field)
            # once ready and 404s while the manifest is still null.
            try:
                d = _get_json(base + "/api/datasets/" + ds_id, 10)
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    time.sleep(2.0)
                    continue
                raise
            st = d.get("status")
            if st is None:
                st = "ready" if (d.get("source") or d.get("cellCount")) else "loading"
            if st != "loading":
                info["dataset_status"] = st
                info["dataset_seconds"] = round(time.time() - t0, 1)
                break
            time.sleep(2.0)
        if info["dataset_status"] is None:
            raise RuntimeError("the dataset was still loading after 300 s")
        _post_json(base + "/api/geometry/open",
                   {"path": "geometry/f1_sim.stl"}, 120)
        info["geometry_ok"] = True
    except Exception as e:
        info["error"] = repr(e)
        body = ""
        if isinstance(e, urllib.error.HTTPError):
            try:
                body = e.read().decode("utf-8", "replace")[:300]
            except Exception:
                pass
        if body:
            info["error"] += " | body: " + body
    return info


def nvidia_smi():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        return (r.stdout or "").strip() or (r.stderr or "").strip()
    except Exception as e:
        return "unavailable: " + repr(e)


def stop_servers(procs):
    """taskkill by the PIDs this script started, never by image name."""
    rcs = {}
    for name, p in procs.items():
        rcs[name] = subprocess.run(
            ["taskkill", "/PID", str(p.pid), "/T", "/F"],
            capture_output=True, text=True).returncode
    return rcs


def wait_ports_free(ports, deadline_s=20):
    t0 = time.time()
    while time.time() - t0 < deadline_s:
        if not any(tcp_open(p) for p in ports):
            return True
        time.sleep(0.5)
    return False


# ------------------------------------------------------------ the shots

def _inner(page, test_id, first=True, limit=2000):
    loc = page.get_by_test_id(test_id)
    loc = loc.first if first else loc.last
    try:
        if page.get_by_test_id(test_id).count() == 0:
            return None
        return loc.inner_text()[:limit]
    except Exception:
        return None


def _cards(page):
    try:
        loc = page.get_by_test_id("tool-card")
        return [loc.nth(i).inner_text()[:200] for i in range(loc.count())]
    except Exception:
        return []


def run_shot(page, cdp, shot, out, opts, first_epoch):
    sid, kind = shot["id"], shot["kind"]
    fdir = os.path.join(out, "frames", sid)
    os.makedirs(fdir, exist_ok=True)
    t0 = time.time()
    kept = []                      # (frame index, t seconds from the shot start)
    ctr = {"n": 1, "last": 0.0}
    pr = {"t": time.time()}

    def on_frame(params):
        try:
            cdp.send("Page.screencastFrameAck",
                     {"sessionId": params.get("sessionId")})
        except Exception:
            pass
        meta = params.get("metadata") or {}
        data = params.get("data")
        if meta.get("timestamp") is None or not data:
            return
        t = float(meta["timestamp"]) - t0
        if keep_frame(t, ctr["last"], FPS_CAP):
            with open(os.path.join(fdir, "f_%05d.jpg" % ctr["n"]), "wb") as f:
                f.write(base64.b64decode(data))
            kept.append((ctr["n"], t))
            ctr["n"] += 1
            ctr["last"] = t

    def progress(extra=""):
        if time.time() - pr["t"] >= 10.0:
            pr["t"] = time.time()
            print("[capture] %s %s t=%.1f frames=%d %s"
                  % (sid, kind, time.time() - t0, len(kept) + 1, extra), flush=True)

    def hold(seconds):
        w0 = time.time()
        while time.time() - w0 < seconds:
            page.wait_for_timeout(200)
            progress()

    print("[capture] %s %s t=0.0 frames=1" % (sid, kind), flush=True)
    with open(os.path.join(fdir, "f_00000.jpg"), "wb") as f:
        f.write(page.screenshot(type="jpeg", quality=JPEG_QUALITY))
    cdp.on("Page.screencastFrame", on_frame)
    cdp.send("Page.startScreencast", {"format": "jpeg", "quality": JPEG_QUALITY,
                                      "maxWidth": 1920, "maxHeight": 1080,
                                      "everyNthFrame": 1})
    rec = {"id": sid, "kind": kind, "shows": shot.get("shows"),
           "prompt": shot.get("text"), "turn_seconds": None,
           "timed_out": None, "approvals": 0,
           "assistant_text": None, "tool_cards": []}
    start_cards = set(_cards(page))
    if kind == "hold":
        hold(float(shot["seconds"]))
    elif kind == "prompt":
        _run_prompt(page, shot, rec, opts, progress, hold, start_cards)
    else:
        _run_click_run(page, shot, opts, progress, hold)
    end_cards = _cards(page)
    rec["tool_cards"] = [t for t in end_cards if t not in start_cards]
    t_end = time.time() - t0
    scroll_chat(page)
    page.screenshot(path=os.path.join(out, "stills", sid + ".png"))
    cdp.send("Page.stopScreencast")
    cdp.remove_listener("Page.screencastFrame", on_frame)
    rec.update(_finalize(fdir, out, kept, t_end, t0, page, rec))
    rec["start_s"] = round(t0 - first_epoch, 3)
    rec["seconds"] = round(t_end, 3)
    rec["viewer_visible"] = _visible(page, "viewer3d")
    rec["legend_text"] = _inner(page, "viewer-legend")
    rec["residuals_visible"] = _visible(page, "residuals-chart")
    rec["residuals_data"] = _visible(page, "residuals-legend")
    rec["viewer_backend"] = _inner(page, "viewer-backend")
    try:
        rec["user_messages"] = page.get_by_test_id("user-message").count()
    except Exception:
        rec["user_messages"] = None
    return rec


def scroll_chat(page):
    """C16: keep the chat list scrolled to its newest message; absent list
    is fine (the stills of hold shots have nothing to follow)."""
    try:
        page.evaluate(
            "() => { const el = document.querySelector("
            "'[data-testid=\"message-list\"]');"
            " if (el) el.scrollTop = el.scrollHeight; }")
    except Exception:
        pass


def _visible(page, test_id):
    try:
        return bool(page.get_by_test_id(test_id).first.is_visible())
    except Exception:
        return False


def _run_prompt(page, shot, rec, opts, progress, hold, start_cards):
    page.get_by_test_id("composer-input").first.click()
    page.keyboard.type(shot["text"], delay=TYPE_DELAY_MS)
    page.keyboard.press("Enter")
    try:
        # C13: a fast answer can end the turn before stop-btn is ever seen;
        # the poll loop below still decides when the turn is done.
        page.get_by_test_id("stop-btn").first.wait_for(
            state="visible", timeout=15000)
    except Exception:
        pass
    stop_btn = page.get_by_test_id("stop-btn").first
    send_btn = page.get_by_test_id("send-btn").first
    appr = page.get_by_test_id("approve-btn")
    turn_timeout = float(shot.get("turn_timeout") or opts.turn_timeout)
    turn0 = time.time()
    approvals, timed_out = 0, False
    while True:
        page.wait_for_timeout(500)
        scroll_chat(page)
        try:
            for i in range(appr.count()):
                b = appr.nth(i)
                if b.is_visible():
                    b.click()
                    approvals += 1
                    break
        except Exception:
            pass
        try:
            stop_vis = stop_btn.is_visible()
        except Exception:
            stop_vis = False
        try:
            send_vis = send_btn.is_visible()
        except Exception:
            send_vis = False
        progress("turn=" + ("answering" if stop_vis else "idle"))
        if turn_done(stop_vis, send_vis):
            break
        if time.time() - turn0 > turn_timeout:
            try:
                stop_btn.click(timeout=2000)
            except Exception:
                pass
            timed_out = True
            break
    rec["turn_seconds"] = round(time.time() - turn0, 3)
    rec["timed_out"] = timed_out
    rec["approvals"] = approvals
    page.wait_for_timeout(int(SETTLE_S * 1000))
    scroll_chat(page)
    if shot.get("orbit") and _visible(page, "viewer3d"):
        _orbit(page, progress)
    hold(HOLD_S)
    rec["assistant_text"] = _inner(page, "assistant-message", first=False)


def _orbit(page, progress):
    bb = page.get_by_test_id("viewer3d").first.bounding_box()
    if not bb:
        return
    cx, cy = bb["x"] + bb["width"] / 2.0, bb["y"] + bb["height"] / 2.0
    page.mouse.move(cx, cy)
    page.mouse.down()
    step = ORBIT_PX / float(ORBIT_STEPS)
    dt = int(1000 * ORBIT_S / ORBIT_STEPS)
    for i in range(ORBIT_STEPS):
        page.mouse.move(cx + step * (i + 1), cy)
        page.wait_for_timeout(dt)
        progress("orbit")
    page.mouse.up()


def _preselect_target(shots, ws):
    """The run the capture preselects: the first click_run shot's run_id,
    else promo-solve-full, with that run's label."""
    run_id = "promo-solve-full"
    for s in shots:
        if s.get("kind") == "click_run" and s.get("run_id"):
            run_id = s["run_id"]
            break
    label = ""
    try:
        rj = os.path.join(ws, "state", "runs", run_id, "run.json")
        label = json.load(open(rj, encoding="utf-8")).get("label") or ""
    except Exception:
        pass
    return run_id, label


def preselect_run(page, run_id, label):
    """C11: select the finished run once, then reload. Dev StrictMode runs
    the App effect connect -> cleanup -> connect, and only the first connect
    registers the ui watcher, so clicking a finished run never sends
    run.subscribe; the reload re-hellos with activeRunId restored from
    localStorage and fills the residual chart and the terminal log."""
    res = {"run_id": run_id, "ok": False, "seconds": None}
    t0 = time.time()
    try:
        page.get_by_test_id("activity-run").first.click()
        rows = page.get_by_test_id("run-row")
        target = None
        try:
            for i in range(rows.count()):
                r = rows.nth(i)
                txt = r.inner_text()
                if (label and label in txt) or run_id in txt:
                    target = r
                    break
        except Exception:
            pass
        if target is None:
            target = rows.first
        target.click()
        page.wait_for_timeout(1500)
        page.get_by_test_id("activity-explorer").first.click()
        page.reload(wait_until="domcontentloaded")
        page.get_by_test_id("composer-input").first.wait_for(
            state="visible", timeout=60000)
        page.get_by_test_id("residuals-legend").first.wait_for(
            state="visible", timeout=20000)
        res["ok"] = True
    except Exception as e:
        res["error"] = repr(e)
    res["seconds"] = round(time.time() - t0, 2)
    return res


def _empty_chat(page, errors):
    """C12: the browser reopens the server's latest stored session; clear it
    so the capture starts from an empty chat. True when new-chat was used."""
    try:
        if page.get_by_test_id("user-message").count() == 0:
            return False
    except Exception:
        return False
    clicked = False
    try:
        page.get_by_test_id("new-chat-btn").first.click()
        clicked = True
    except Exception as e:
        errors.append("new-chat-btn click failed: " + repr(e))
        return False
    t0 = time.time()
    while time.time() - t0 < 10.0:
        try:
            if page.get_by_test_id("user-message").count() == 0:
                return clicked
        except Exception:
            pass
        page.wait_for_timeout(250)
    errors.append("user-message still present 10 s after new-chat-btn")
    return clicked


def set_session_locale(page, api_port, errors, locale="ko"):
    """C15: the session's settings.locale defaults to 'en' and only the
    settings popover's language <select> sends settings.set, and selecting
    the value already shown fires no change - so go en -> ko through the
    popover, then read the session back over the API. Returns the read
    locale (None on failure; recorded in errors, never refused)."""
    try:
        page.get_by_test_id("activity-settings").first.click()
        sel = page.locator('select:has(option[value="ko"])').first
        sel.select_option("en")
        page.wait_for_timeout(500)
        sel.select_option(locale)
        page.wait_for_timeout(500)
        page.keyboard.press("Escape")
        page.wait_for_timeout(300)
        if sel.is_visible():
            page.get_by_test_id("activity-settings").first.click()
            page.wait_for_timeout(300)
    except Exception as e:
        errors.append("set_session_locale ui failed: " + repr(e))
        return None
    try:
        base = "http://127.0.0.1:%d" % api_port
        sessions = _get_json(base + "/api/sessions", 10)
        if not isinstance(sessions, list) or not sessions:
            raise RuntimeError("no sessions in GET /api/sessions")
        newest = max(sessions, key=lambda s: s.get("updatedAt") or "")
        detail = _get_json(base + "/api/sessions/" + str(newest.get("id")), 10)
        val = ((detail.get("settings") or {}).get("locale"))
        return str(val) if val is not None else None
    except Exception as e:
        errors.append("set_session_locale read failed: " + repr(e))
        return None


def _run_click_run(page, shot, opts, progress, hold):
    run_id = shot["run_id"]
    _, label = _preselect_target([shot], opts.ws)
    page.get_by_test_id("activity-run").first.click()
    rows = page.get_by_test_id("run-row")
    target = None
    try:
        for i in range(rows.count()):
            r = rows.nth(i)
            txt = r.inner_text()
            if run_id in txt or (label and label in txt):
                target = r
                break
    except Exception:
        pass
    if target is None:
        target = rows.first
    target.click()
    try:
        # C14: the data signal is the legend; a missing one is recorded as
        # residuals_data false, the shot itself still stands.
        page.get_by_test_id("residuals-legend").first.wait_for(
            state="visible", timeout=20000)
    except Exception:
        pass
    hold(float(shot["seconds"]))


def _finalize(fdir, out, kept, t_end, t0, page, rec):
    """frames.csv, the concat list, the clip and the size checks."""
    times = [0.0] + [t for _, t in kept]
    names = ["f_%05d.jpg" % i for i in range(len(times))]
    with open(os.path.join(fdir, "frames.csv"), "w", encoding="utf-8",
              newline="\n") as f:
        f.write("index,t\n")
        for i, t in enumerate(times):
            f.write("%d,%.6f\n" % (i, t))
    with open(os.path.join(fdir, "list.txt"), "w", encoding="utf-8",
              newline="\n") as f:
        f.write(concat_list(times, names, t_end))
    clip_rel = "clips/%s.mp4" % rec["id"]
    clip_path = os.path.join(out, clip_rel.replace("/", os.sep))
    clip_seconds, clip_error = None, None
    try:
        r = subprocess.run(
            ffmpeg_args("list.txt", os.path.abspath(clip_path), t_end),
            cwd=fdir, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            raise RuntimeError("ffmpeg rc %d: %s" % (r.returncode,
                                                     (r.stderr or "")[-400:]))
        pr = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", os.path.abspath(clip_path)],
            capture_output=True, text=True, timeout=60)
        clip_seconds = float(pr.stdout.strip())
    except Exception as e:
        clip_error = repr(e)
        clip_rel = None
        clip_seconds = None
    sizes_ok, size_notes = True, []
    checked = [(os.path.join(out, "stills", rec["id"] + ".png"), png_size),
               (os.path.join(fdir, names[0]), jpeg_size),
               (os.path.join(fdir, names[-1]), jpeg_size)]
    for path, fn in checked:
        try:
            w, h = fn(path)
            if (w, h) != VIEWPORT:
                sizes_ok = False
                size_notes.append("%s is %dx%d" % (os.path.basename(path), w, h))
        except Exception as e:
            sizes_ok = False
            size_notes.append(repr(e))
    return {"file": clip_rel, "frames_dir": "frames/" + rec["id"],
            "still": "stills/" + rec["id"] + ".png",
            "clip_seconds": clip_seconds, "clip_error": clip_error,
            "frames": len(times), "sizes_ok": sizes_ok,
            "size_notes": size_notes}


# ------------------------------------------------------------------- run

def load_shots(a):
    if a.script:
        shots = json.load(open(a.script, encoding="utf-8"))
        check_script(shots)
        return shots
    shots = REDUCED_SHOTS if a.reduced else SHOTS
    check_script(shots)
    return shots


def cmd_run(a):
    t0 = time.time()
    shots = load_shots(a)
    check_ports(a.api_port, a.web_port)
    check_ws(a.ws)
    out = os.path.abspath(a.out)
    check_out(out, a.gui_tree)
    check_node(a.gui_tree)
    os.makedirs(out, exist_ok=True)
    for sub in ("frames", "clips", "stills"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    st = {"procs": {}, "shots": [], "refusal": None, "errors": [],
          "hello": None, "ready": None, "prewarm": None,
          "run_preselect": None, "new_chat_clicked": None,
          "session_locale": None,
          "backend": None, "nvidia_before": None, "nvidia_after": None,
          "stopped": {}, "ports_free": None, "browser": None, "pw": None,
          "interrupted": False}
    try:
        st["procs"] = start_servers(a.ws, a.gui_tree, out, a.api_port,
                                    a.web_port, a.llm)
        st["ready"] = wait_ready(a.api_port, a.web_port)
        hello = _get_json("http://127.0.0.1:%d/api/hello" % a.api_port, 10)
        st["hello"] = {"llm": hello.get("llm"), "model": hello.get("model"),
                       "mode": hello.get("mode")}
        if hello.get("llm") != a.llm:
            raise CaptureRefusal("CAP-LLM", "hello.llm %r != --llm %r"
                                 % (hello.get("llm"), a.llm))
        st["prewarm"] = prewarm(a.api_port)
        st["nvidia_before"] = nvidia_smi()
        _drive_browser(st, a, out, shots)
    except CaptureRefusal as r:
        st["refusal"] = r
        st["errors"].append("refused: " + r.code + ": " + r.text)
    except KeyboardInterrupt:
        st["interrupted"] = True
        st["errors"].append("interrupted by the user")
    except Exception as e:
        import traceback
        st["errors"].append(traceback.format_exc()[-2000:])
    finally:
        try:
            if st["browser"]:
                st["browser"].close()
        except Exception:
            pass
        try:
            if st["pw"]:
                st["pw"].stop()
        except Exception:
            pass
        st["nvidia_after"] = nvidia_smi()
        if st["procs"]:
            st["stopped"] = stop_servers(st["procs"])
            st["ports_free"] = wait_ports_free([a.api_port, a.web_port])
        _write_outputs(st, a, out, shots, t0)
    if st["refusal"]:
        print("refused: " + st["refusal"].code + ": " + st["refusal"].text)
        sys.exit(2)
    if st["interrupted"]:
        sys.exit(130)
    print("CAPTURE DONE %d shots %.1f s" % (len(st["shots"]), time.time() - t0))


def _drive_browser(st, a, out, shots):
    st["pw"] = sync_playwright().start()
    browser = st["pw"].chromium.launch(headless=True, args=CHROME_ARGS)
    st["browser"] = browser
    ctx = browser.new_context(viewport={"width": VIEWPORT[0],
                                        "height": VIEWPORT[1]}, locale="ko-KR")
    page = ctx.new_page()
    page.goto("http://127.0.0.1:%d/" % a.web_port, wait_until="domcontentloaded")
    page.get_by_test_id("composer-input").first.wait_for(
        state="visible", timeout=60000)
    run_id, label = _preselect_target(shots, a.ws)
    st["run_preselect"] = preselect_run(page, run_id, label)
    if not st["run_preselect"].get("ok"):
        st["errors"].append("run_preselect failed: " +
                            json.dumps(st["run_preselect"], ensure_ascii=False))
    st["new_chat_clicked"] = _empty_chat(page, st["errors"])
    st["session_locale"] = set_session_locale(page, a.api_port, st["errors"])
    cdp = ctx.new_cdp_session(page)
    first_epoch = time.time()
    for shot in shots:
        st["shots"].append(run_shot(page, cdp, shot, out, a, first_epoch))
    st["backend"] = _inner(page, "viewer-backend")


def _last_backend(st):
    """C14: the last non-null per-shot viewer_backend, else the end-of-run
    read."""
    for r in reversed(st.get("shots") or []):
        if r.get("viewer_backend") is not None:
            return r["viewer_backend"]
    return st.get("backend")


def _write_outputs(st, a, out, shots, t0):
    hello = st.get("hello") or {}
    ws_man_path = os.path.join(a.ws, "studio_ws.json")
    attribution = None
    try:
        attribution = json.load(open(ws_man_path, encoding="utf-8")).get("attribution")
    except Exception:
        pass
    shots_doc = {"version": VERSION, "viewport": list(VIEWPORT),
                 "fps_cap": FPS_CAP, "clip_fps": CLIP_FPS,
                 "llm": hello.get("llm"), "model": hello.get("model"),
                 "attribution": attribution, "shots": st["shots"]}
    with open(os.path.join(out, "shots.json"), "w", encoding="utf-8",
              newline="\n") as f:
        json.dump(shots_doc, f, indent=1, ensure_ascii=False)
        f.write("\n")
    lic = os.path.join(a.ws, "LICENSE.txt")
    if os.path.isfile(lic):
        shutil.copyfile(lic, os.path.join(out, "LICENSE.txt"))
    procs = st.get("procs") or {}
    cap = {"tool": TOOL, "version": VERSION, "ws": _posix(a.ws),
           "gui_tree": _posix(a.gui_tree), "out": _posix(out),
           "api_port": a.api_port, "web_port": a.web_port, "llm": a.llm,
           "reduced": bool(a.reduced), "hello": hello,
           "server_pid": procs["server"].pid if "server" in procs else None,
           "web_pid": procs["web"].pid if "web" in procs else None,
           "ready_seconds": st.get("ready"),
           "prewarm": st.get("prewarm"),
           "run_preselect": st.get("run_preselect"),
           "new_chat_clicked": st.get("new_chat_clicked"),
           "session_locale": st.get("session_locale"),
           "viewer_backend": _last_backend(st),
           "nvidia_smi_before": st.get("nvidia_before"),
           "nvidia_smi_after": st.get("nvidia_after"), "gpu_shared": True,
           "stopped": st.get("stopped") or {},
           "ports_free_after": st.get("ports_free"),
           "seconds": round(time.time() - t0, 1), "errors": st["errors"]}
    with open(os.path.join(out, "capture.json"), "w", encoding="utf-8",
              newline="\n") as f:
        json.dump(cap, f, indent=1, ensure_ascii=False)
        f.write("\n")


# --------------------------------------------------------------- selftest

def _selftest():
    import subprocess
    import tempfile
    t0 = time.time()
    ok = []

    # T7 concat_list and keep_frame
    got = concat_list([0.0, 0.5, 1.25], ["a.jpg", "b.jpg", "c.jpg"], 2.0)
    exp = ("file 'a.jpg'\nduration 0.500000\nfile 'b.jpg'\n"
           "duration 0.750000\nfile 'c.jpg'\nduration 0.750000\nfile 'c.jpg'\n")
    assert got == exp, repr(got)
    assert keep_frame(0.05, 0.0, 15) is False
    assert keep_frame(0.07, 0.0, 15) is True
    assert keep_frame(0.0, None, 15) is True
    ok.append("T7")

    # T8 png_size / jpeg_size on tiny images written here
    with tempfile.TemporaryDirectory() as root:
        def chunk(typ, data):
            return (struct.pack(">I", len(data)) + typ + data +
                    struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF))
        raw = b"".join(b"\x00" + b"\x40\x80\xc0" * 3 for _ in range(2))
        png = (b"\x89PNG\r\n\x1a\n" +
               chunk(b"IHDR", struct.pack(">IIBBBBB", 3, 2, 8, 2, 0, 0, 0)) +
               chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
        pp = os.path.join(root, "tiny.png")
        open(pp, "wb").write(png)
        assert png_size(pp) == (3, 2), png_size(pp)
        jp = os.path.join(root, "tiny.jpg")
        r = subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel",
                            "error", "-f", "lavfi", "-i", "color=c=red:s=4x6",
                            "-frames:v", "1", jp], capture_output=True, text=True)
        if r.returncode == 0 and os.path.isfile(jp):
            assert jpeg_size(jp) == (4, 6), jpeg_size(jp)
        else:
            print("[capture] T8: ffmpeg missing, jpeg half skipped")
    ok.append("T8")

    # T9 validate_shots
    assert validate_shots(SHOTS) == []
    assert validate_shots(REDUCED_SHOTS) == []
    dup = [dict(SHOTS[0]), dict(SHOTS[0])]
    p = validate_shots(dup)
    assert len(p) == 1 and "duplicate id" in p[0], p
    p = validate_shots([{"id": "x", "kind": "zoom", "shows": "s"}])
    assert len(p) == 1 and "unknown kind" in p[0], p
    p = validate_shots([{"id": "x", "kind": "prompt", "text": "", "shows": "s"}])
    assert len(p) == 1 and "non-empty text" in p[0], p
    p = validate_shots([{"id": "x", "kind": "hold", "shows": "s"}])
    assert len(p) == 1 and "seconds > 0" in p[0], p
    ok.append("T9")

    # T10 server_env
    env = server_env({"ZAI_API_KEY": "x", "PATH": "p"}, "C:/ws", 8799, "zai")
    assert "ZAI_API_KEY" not in env and env["PATH"] == "p"
    assert env["CFD_PORT"] == "8799"
    assert env["CFD_WORKSPACE"] == os.path.join("C:/ws", "ws")
    assert env["CFD_GUI_DIR"] == os.path.join("C:/ws", "state")
    assert env["CFD_LLM"] == "zai" and env["CFD_PYTHON"] == "python"
    ok.append("T10")
    print("[capture] selftest T7-T10 %.1f s" % (time.time() - t0))

    this = os.path.abspath(__file__)

    def run_cli(args):
        return subprocess.run([sys.executable, this, "run"] + args,
                              capture_output=True, text=True)

    def make_ws(root, broken=False):
        ws = os.path.join(root, "ws")
        os.makedirs(os.path.join(ws, "ws", "docs", "schema"), exist_ok=True)
        os.makedirs(os.path.join(ws, "ws", "cases", "f1-promo"), exist_ok=True)
        os.makedirs(os.path.join(ws, "state", "runs"), exist_ok=True)
        open(os.path.join(ws, "ws", "docs", "schema", "case-1.json"), "w").write("{}")
        if not broken:
            open(os.path.join(ws, "studio_ws.json"), "w").write("{}")
        return ws

    # T11 refusals before any process starts
    with tempfile.TemporaryDirectory() as root:
        gt = os.path.join(root, "gui")
        os.makedirs(gt, exist_ok=True)
        ws = make_ws(root)
        out = os.path.join(root, "out")
        r = run_cli(["--ws", ws, "--gui-tree", gt, "--out", out,
                     "--llm", "mock", "--reduced", "--api-port", "8787"])
        assert r.returncode == 2 and "refused: CAP-PORT" in r.stdout, r.stdout + r.stderr
        r = run_cli(["--ws", ws, "--gui-tree", gt, "--out", os.path.join(REPO, "cap_selftest_out"),
                     "--llm", "mock", "--reduced"])
        assert r.returncode == 2 and "refused: CAP-OUT" in r.stdout, r.stdout + r.stderr
        bad_ws = make_ws(os.path.join(root, "b"), broken=True)
        r = run_cli(["--ws", bad_ws, "--gui-tree", gt, "--out", out,
                     "--llm", "mock", "--reduced"])
        assert r.returncode == 2 and "refused: CAP-WS" in r.stdout, r.stdout + r.stderr
    ok.append("T11")
    print("[capture] selftest T11 %.1f s" % (time.time() - t0))

    # T12 a --script with an unknown kind refuses before any process starts
    with tempfile.TemporaryDirectory() as root:
        gt = os.path.join(root, "gui")
        os.makedirs(gt, exist_ok=True)
        ws = make_ws(root)
        script = os.path.join(root, "shots.json")
        open(script, "w", encoding="utf-8").write(
            json.dumps([{"id": "x", "kind": "spin", "shows": "s"}]))
        r = run_cli(["--ws", ws, "--gui-tree", gt, "--out", os.path.join(root, "out"),
                     "--llm", "mock", "--reduced", "--script", script])
        assert r.returncode == 2 and "refused: CAP-SCRIPT" in r.stdout, r.stdout + r.stderr
    ok.append("T12")
    print("[capture] selftest T12 %.1f s" % (time.time() - t0))

    # T13 the turn's end condition and the shot lists still validate
    assert turn_done(False, True) is True
    assert turn_done(True, True) is False
    assert turn_done(False, False) is False
    assert validate_shots(SHOTS) == [] and validate_shots(REDUCED_SHOTS) == []
    ok.append("T13")

    # T14 the clip's ffmpeg args cut to t_end; the C18 s2/s3 texts verbatim
    args = ffmpeg_args("list.txt", "clips/x.mp4", 3.5)
    i = args.index("-t")
    assert args[i + 1] == "%.6f" % 3.5, args
    assert args[-1] == "clips/x.mp4", args
    assert "%.6f" % 0 == "0.000000"
    s2 = [s for s in SHOTS if s["id"] == "s2_mesh"][0]
    s3 = [s for s in SHOTS if s["id"] == "s3_cp"][0]
    assert s2["text"] == ("cases/f1-promo 해석 격자를 3D 뷰어에 띄워서 차량 표면의 "
                          "격자선(Surface + Edges)을 보여줘. 터널 벽은 클립 박스로 "
                          "잘라내 줘."), s2["text"]
    assert s3["text"] == ("차량 표면을 압력계수 Cp 로 색칠해줘. coolwarm 색상, "
                          "범위 -1 ~ 1, 격자선 없이 Surface 로 보여줘."), s3["text"]
    ok.append("T14")
    print("SELFTEST PASS %d/%d (%.1f s)" % (len(ok), 8, time.time() - t0))


def main():
    ap = argparse.ArgumentParser(description=TOOL)
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run")
    r.add_argument("--ws", required=True)
    r.add_argument("--gui-tree", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--llm", default="zai", choices=["zai", "mock"])
    r.add_argument("--api-port", type=int, default=8799)
    r.add_argument("--web-port", type=int, default=5183)
    r.add_argument("--reduced", action="store_true")
    r.add_argument("--turn-timeout", type=float, default=360)
    r.add_argument("--script", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    elif a.cmd == "run":
        cmd_run(a)
    else:
        ap.print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()
