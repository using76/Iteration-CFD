#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.

"""Try the DRONE-ISAAC scene in Isaac Sim, headless, in memory only.

Opens tools/drone/isaac_export.py's drone_flow.usda through SimulationApp
(headless), waits for the stage to load, adds a DomeLight and a
DistantLight, hides every /World/Flow volume except the requested field's
(wake by default) and adds the `density` field relationship Isaac's volume
path shades, then renders the first --frames views (front34 / side /
wake34 cameras from look_at matrices, clipping 0.01-50, --warmup
rep.orchestrator.step each) through one render_product + rgb annotator,
saving isaac_<view>.png and, for the first view, two volume-hidden control
frames (isaac_<view>_novolume.png and _novolume2.png, b and c grabbed back
to back with the volume still hidden) plus diff = diff_verdict(a, b) and
noise = diff_verdict(b, c) (the threshold rule below), volume_visible =
volume_effect(diff, noise): the volume counts as visible only when diff
says so AND its changed_px is more than twice the noise's, because a and b
are separate RaytracedLighting accumulations whose speckle (run 3: 32284
changed px, all on the drone's own surfaces, the empty background
untouched) is re-render noise, not the volume - in diff the volume is
visible iff the pair differs on more than DIFF_MIN_FRAC of the frame by
more than DIFF_THR/255 per pixel.
Writes probe.json and prints "ISAAC-PROBE <json>" as its summary line; the
stage is never saved. [probe] <what> <t> s progress lines run on stdout
(SimulationApp up, every 50 load updates, the stage, each view's render
product, each grab), and a daemon watchdog Timer armed right after
SimulationApp starts fires at --max-seconds: the in-loop checks only run
between calls, so when one blocking call hangs the watchdog itself records
timed_out, writes probe.json, prints the summary line and exits 3.

Run by the SUPERVISOR with Isaac's own Python
(C:/iss/env/Scripts/python.exe), never by this repository's tooling and
never here: isaacsim, omni and carb are imported only inside main, after
SimulationApp starts. The scene's flow values are demonstration numbers
from an unsteady run, not validated; wake is |U - U_inf| / |U_inf|,
derived from U. The drone model: (c) 2022 Rudis Laboratories / PX4
Autopilot for Drones, BSD-3-Clause.

--selftest runs P1-P8 in plain Python (no Isaac needed).
"""

import argparse
import json
import os
import shutil
import struct
import sys
import tempfile
import threading
import time
import zlib

import numpy as np

TOOL = "tools/drone/isaac_probe.py"
VERSION = "drone-isaac-probe/1"
TARGET = (-0.15, 0.0, -0.08)
VIEWS = [("front34", (1.2, 0.9, 0.6)), ("side", (0.0, -1.6, 0.05)), ("wake34", (-1.4, 0.7, 0.7))]   # eye positions (m)
DIFF_THR = 4                      # /255, per pixel = max over RGB of |a - b|
DIFF_MIN_FRAC = 0.0005
_MAIN_T0 = None                   # process start, set in main; the watchdog's wall_s


def safe_prim_name(field):
    """The isaac_export rule, re-implemented: [^A-Za-z0-9_] -> "_", digit first gets "_". """
    s = "".join(c if (c.isascii() and c.isalnum()) or c == "_" else "_" for c in str(field))
    return ("_" + s) if s[:1].isdigit() else s


def diff_verdict(a, b, thr=DIFF_THR):
    """R7 of the probe: is the volume visible in a, given the control b?"""
    zero = {"changed_px": 0, "changed_frac": 0.0, "maxdiff": 0, "volume_visible": False}
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return zero
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2)
    changed = int((d > thr).sum())
    frac = float(changed) / float(d.size) if d.size else 0.0
    maxdiff = int(d.max()) if d.size else 0
    return {"changed_px": changed, "changed_frac": frac, "maxdiff": maxdiff,
            "volume_visible": bool(frac > DIFF_MIN_FRAC and maxdiff > thr)}


def volume_effect(diff, noise):
    """True only when diff calls the volume visible AND its changed_px is
    more than twice the b/c re-render noise: run 3's 32284 changed px all
    sat on the drone's own specular surfaces while the empty background -
    where the wake is - never moved, so the two frames are separate
    RaytracedLighting accumulations whose speckle is noise, not the volume."""
    if diff is None or noise is None:
        return False
    return bool(diff["volume_visible"]
                and diff["changed_px"] > 2 * noise["changed_px"])


def look_at(eye, target):
    """16 floats, row-major Gf.Matrix4d rows: right, up, -forward, eye."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up0 = np.array([0.0, 0.0, 1.0])
    fwd = target - eye
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, up0)
    right = right / np.linalg.norm(right)
    up = np.cross(right, fwd)
    m = [right[0], right[1], right[2], 0.0,
         up[0], up[1], up[2], 0.0,
         -fwd[0], -fwd[1], -fwd[2], 0.0,
         eye[0], eye[1], eye[2], 1.0]
    return [float(v) for v in m]


def _save_png(arr, path):
    """One minimal RGB8 PNG (zlib + struct, no PIL)."""
    body = np.ascontiguousarray(arr, dtype=np.uint8)
    h, w = int(body.shape[0]), int(body.shape[1])
    rows = bytearray()
    for y in range(h):
        rows += bytes(1)
        rows += body[y].tobytes()

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    sig = bytes([137, 80, 78, 71, 13, 10, 26, 10])
    with open(path, "wb") as fh:
        fh.write(sig)
        fh.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
        fh.write(chunk(b"IDAT", zlib.compress(bytes(rows), 6)))
        fh.write(chunk(b"IEND", b""))


def _isaac_version():
    try:
        import isaacsim
        p = os.path.join(os.path.dirname(os.path.abspath(isaacsim.__file__)), "VERSION")
        with open(p, encoding="utf-8") as fh:
            return fh.read().strip() or None
    except Exception:
        return None


def _grab(ann, width, height):
    data = ann.get_data()
    a = np.asarray(data)
    if a.size == 0:
        return None
    if a.ndim == 1:
        a = a.reshape((int(height), int(width), 4))
    return np.ascontiguousarray(a[..., :3], dtype=np.uint8)


def _progress(what, t0):
    """One [probe] heartbeat line, flushed - what the supervisor reads when
    the outer timeout has to be interpreted."""
    print("[probe] " + str(what) + " " + ("%.1f" % (time.time() - t0)) + " s",
          flush=True)


def arm_watchdog(seconds, out_dir, result, exit_fn=os._exit):
    """The daemon Timer main arms right after SimulationApp starts. The
    in-loop --max-seconds checks only run between calls, so a hang inside
    one blocking call never returns: when the Timer fires it records
    timed_out, writes probe.json, prints the ISAAC-PROBE line and exits 3.
    Returns the Timer (main cancels it after a normal finish)."""
    armed_at = time.time()

    def _fire():
        result["timed_out"] = True
        result["error"] = "watchdog: " + str(seconds) + " s"
        result["wall_s"] = round(time.time()
                                 - (_MAIN_T0 if _MAIN_T0 is not None else armed_at), 1)
        try:
            write_probe_json(out_dir, result)
        finally:
            print("ISAAC-PROBE " + json.dumps(result, default=str), flush=True)
            exit_fn(3)

    t = threading.Timer(float(seconds), _fire)
    t.daemon = True
    t.start()
    return t


def _run_probe(sim, args, result, t0):
    import omni.replicator.core as rep
    import omni.usd
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdVol

    ctx = omni.usd.get_context()
    ctx.open_stage(args.scene)
    loaded = False
    for i in range(100000):
        sim.update()
        if (i + 1) % 50 == 0:
            _progress(str(i + 1) + " load updates", t0)
        _files_loaded, total_files = ctx.get_stage_loading_status()[1:3]
        if total_files == 0 and i > 10:
            loaded = True
            break
        if time.time() - t0 > args.max_seconds:
            result["timed_out"] = True
            break
    result["stage_loaded"] = bool(loaded)
    _progress("stage loaded" if loaded else "stage load unfinished", t0)
    stage = ctx.get_stage()
    result["isaac_version"] = _isaac_version()
    if stage is None:
        return
    UsdLux.DomeLight.Define(stage, "/World/ProbeDome").CreateIntensityAttr(1000.0)
    UsdLux.DistantLight.Define(stage, "/World/ProbeSun").CreateIntensityAttr(3000.0)
    result["mesh_prims"] = sum(1 for p in stage.Traverse() if p.IsA(UsdGeom.Mesh))
    flow = stage.GetPrimAtPath("/World/Flow")
    target = None
    want = safe_prim_name(args.field)
    if flow.IsValid():
        result["volume_prims"] = sum(1 for p in flow.GetChildren() if p.IsA(UsdVol.Volume))
        for child in flow.GetChildren():
            if not child.IsA(UsdVol.Volume):
                continue
            if child.GetName() == want:
                target = child
            else:
                UsdGeom.Imageable(child).MakeInvisible()
    if target is not None:
        field_path = Sdf.Path(str(target.GetPath()) + "/" + want)
        UsdVol.Volume(target).CreateFieldRelationship("density", field_path)
        UsdGeom.Imageable(target).MakeVisible()
    nviews = max(1, min(int(args.frames), len(VIEWS)))
    for vi in range(nviews):
        if time.time() - t0 > args.max_seconds:
            result["timed_out"] = True
            break
        vname, eye = VIEWS[vi]
        cam_path = "/World/ProbeCam_" + vname
        cam = UsdGeom.Camera.Define(stage, cam_path)
        camx = UsdGeom.Xformable(cam.GetPrim())
        camx.ClearXformOpOrder()
        camx.AddTransformOp().Set(Gf.Matrix4d(*look_at(eye, TARGET)))
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 50.0))
        rp = rep.create.render_product(cam_path, (args.width, args.height))
        _progress("render product " + vname, t0)
        ann = rep.AnnotatorRegistry.get_annotator("rgb")
        ann.attach([rp])
        for _ in range(int(args.warmup)):
            rep.orchestrator.step()
        a = _grab(ann, args.width, args.height)
        _progress("grab " + vname, t0)
        png = "isaac_" + vname + ".png"
        if a is not None:
            _save_png(a, os.path.join(args.out_dir, png))
            result["frames"].append({"view": vname, "png": png, "mean": float(a.mean()),
                                     "min": int(a.min()), "max": int(a.max())})
        else:
            result["frames"].append({"view": vname, "png": "", "mean": 0.0,
                                     "min": 0, "max": 0})
        if vi == 0:
            b = None
            c = None
            if target is not None:
                UsdGeom.Imageable(target).MakeInvisible()
                for _ in range(int(args.warmup)):
                    rep.orchestrator.step()
                b = _grab(ann, args.width, args.height)
                _progress("grab " + vname + " no-volume", t0)
                if b is not None:
                    _save_png(b, os.path.join(args.out_dir,
                                              "isaac_" + vname + "_novolume.png"))
                for _ in range(int(args.warmup)):
                    rep.orchestrator.step()
                c = _grab(ann, args.width, args.height)
                _progress("grab " + vname + " no-volume2", t0)
                if c is not None:
                    _save_png(c, os.path.join(args.out_dir,
                                              "isaac_" + vname + "_novolume2.png"))
                UsdGeom.Imageable(target).MakeVisible()
            result["diff"] = (diff_verdict(a, b)
                              if a is not None and b is not None else None)
            result["noise"] = (diff_verdict(b, c)
                               if b is not None and c is not None else None)
            result["volume_visible"] = volume_effect(result["diff"],
                                                     result["noise"])


def write_probe_json(out_dir, result):
    """One probe.json: the same dict the ISAAC-PROBE line prints, indent 2;
    returns the file path."""
    path = os.path.join(str(out_dir), "probe.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
        fh.write("\n")
    return path


def main(argv=None):
    global _MAIN_T0
    t0 = time.time()
    _MAIN_T0 = t0
    ap = argparse.ArgumentParser(prog="isaac_probe.py",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--field", default="wake")
    ap.add_argument("--renderer", default="RaytracedLighting")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--warmup", type=int, default=120)
    ap.add_argument("--max-seconds", type=float, default=540.0)
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    result = {"tool": TOOL, "version": VERSION, "isaac_version": None,
              "renderer": args.renderer, "scene": str(args.scene).replace(os.sep, "/"),
              "stage_loaded": False, "mesh_prims": 0, "volume_prims": 0,
              "field": args.field, "frames": [], "diff": None,
              "noise": None, "volume_visible": False, "wall_s": 0.0,
              "timed_out": False, "error": None}
    from isaacsim import SimulationApp
    sim = SimulationApp({"headless": True, "renderer": args.renderer,
                         "width": args.width, "height": args.height,
                         "anti_aliasing": 1})
    _progress("SimulationApp up", t0)
    watchdog = arm_watchdog(float(args.max_seconds), args.out_dir, result)
    err = None
    try:
        _run_probe(sim, args, result, t0)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        err = exc
    watchdog.cancel()
    result["wall_s"] = round(time.time() - t0, 1)
    result["error"] = None if err is None else type(err).__name__ + ": " + str(err)
    write_probe_json(args.out_dir, result)
    print("ISAAC-PROBE " + json.dumps(result, default=str), flush=True)
    sim.close()
    return 1 if err is not None else 0


# ---------------------------------------------------------------- selftest

def _p1():
    a = np.zeros((100, 100, 3), dtype=np.uint8)
    v = diff_verdict(a, a.copy())
    assert v["changed_px"] == 0 and v["maxdiff"] == 0, v
    assert v["volume_visible"] is False and abs(v["changed_frac"]) == 0.0, v


def _p2():
    a = np.zeros((100, 100, 3), dtype=np.uint8)
    b = a.copy()
    b[10:20, 10:20, :] = 50
    v = diff_verdict(a, b)
    assert v["changed_px"] == 100, v
    assert abs(v["changed_frac"] - 0.01) < 1e-12, v
    assert v["maxdiff"] == 50 and v["volume_visible"] is True, v


def _p3():
    a = np.zeros((100, 100, 3), dtype=np.uint8)
    b = a.copy()
    b[10:20, 10:20, :] = 3
    v = diff_verdict(a, b)
    assert v["changed_px"] == 0 and v["volume_visible"] is False, v
    v = diff_verdict(a, np.zeros((50, 100, 3), dtype=np.uint8))
    assert v["changed_px"] == 0 and v["volume_visible"] is False, v


def _p4():
    m = look_at((0, -2, 0), (0, 0, 0))
    assert len(m) == 16, len(m)
    assert tuple(m[8:11]) == (0.0, -1.0, 0.0), m
    assert tuple(m[12:15]) == (0.0, -2.0, 0.0), m
    assert m[15] == 1.0, m


def _p5():
    assert [v[0] for v in VIEWS] == ["front34", "side", "wake34"], VIEWS
    assert "isaacsim" not in sys.modules, "isaacsim leaked into module scope"


def _p6():
    """write_probe_json round-trips the 15 result keys plus error, and the
    module stays isaacsim-free."""
    d = {"tool": TOOL, "version": VERSION, "isaac_version": None,
         "renderer": "RaytracedLighting", "scene": "drone_flow.usda",
         "stage_loaded": True, "mesh_prims": 25, "volume_prims": 6,
         "field": "wake", "frames": [{"view": "front34", "png": "isaac_front34.png",
                                      "mean": 12.5, "min": 0, "max": 255}],
         "diff": {"changed_px": 100, "changed_frac": 0.01, "maxdiff": 50,
                  "volume_visible": True},
         "noise": {"changed_px": 300, "changed_frac": 0.0001, "maxdiff": 9,
                   "volume_visible": False},
         "volume_visible": True,
         "wall_s": 74.5, "timed_out": False, "error": None}
    tmp = tempfile.mkdtemp(prefix="isaac-probe-p6-")
    try:
        p = write_probe_json(tmp, d)
        assert os.path.basename(p) == "probe.json", p
        with open(p, encoding="utf-8") as fh:
            back = json.load(fh)
        assert back == d, back
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert "isaacsim" not in sys.modules, "isaacsim leaked into module scope"


def _p7():
    """arm_watchdog fires: exit code 3 through exit_fn, probe.json on disk
    with timed_out true."""
    tmp = tempfile.mkdtemp(prefix="isaac-probe-p7-")
    try:
        result = {"tool": TOOL, "version": VERSION, "frames": [],
                  "wall_s": 0.0, "timed_out": False, "error": None}
        codes = []
        t = arm_watchdog(0.2, tmp, result, exit_fn=codes.append)
        deadline = time.time() + 3.0
        while time.time() < deadline and not codes:
            time.sleep(0.02)
        t.join(timeout=1.0)
        assert codes == [3], codes
        with open(os.path.join(tmp, "probe.json"), encoding="utf-8") as fh:
            back = json.load(fh)
        assert back["timed_out"] is True, back
        assert back["error"] == "watchdog: 0.2 s", back
        assert isinstance(back["wall_s"], float) and back["wall_s"] > 0.0, back
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert "isaacsim" not in sys.modules, "isaacsim leaked into module scope"


def _p8():
    """volume_effect: None on either side is False; the run-3 numbers are
    False against noise 30000 and True against 1000; a not-visible diff is
    False even against zero noise."""
    x = {"changed_px": 32284, "volume_visible": True}
    assert volume_effect(None, x) is False
    assert volume_effect(x, None) is False
    assert volume_effect(x, {"changed_px": 30000}) is False
    assert volume_effect(x, {"changed_px": 1000}) is True
    assert volume_effect({"changed_px": 5, "volume_visible": False},
                         {"changed_px": 0}) is False


def selftest():
    """R10: P1-P8 in plain Python - no isaacsim/omni/carb import anywhere."""
    tests = [("P1", _p1), ("P2", _p2), ("P3", _p3), ("P4", _p4), ("P5", _p5),
             ("P6", _p6), ("P7", _p7), ("P8", _p8)]
    npass = 0
    for name, fn in tests:
        try:
            fn()
            print("[ok] " + name, flush=True)
            npass += 1
        except Exception as exc:
            print("[FAIL] " + name + ": " + str(exc), flush=True)
    if npass == len(tests):
        print("SELFTEST PASS " + str(npass) + "/" + str(len(tests)), flush=True)
        return 0
    print("SELFTEST FAIL " + str(npass) + "/" + str(len(tests)), flush=True)
    return 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    sys.exit(main())
