#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.

"""The generic headless Isaac visibility probe (SPEC-LIT 117.10).

Opens a .usda scene through Isaac Sim's SimulationApp (headless), finds the
volume prim, frames a box around it, renders one frame with the volume shown
and two control frames with it hidden (b, c grabbed back to back), then
decides from the pixels whether the volume is actually visible:

    changed(a, b) = #{ p : d(p) > 4 }     d(p) = largest abs RGB diff / 255
    shows(a, b) <=> changed(a, b)/N_px > 0.0005 and max_p d(p) > 4   (117.15)
    visible    <=> shows(a, b) and changed(a, b) > 2 changed(b, c)   (117.16)

a is the shown frame; b, c are two volume-hidden frames whose diff is the
re-render noise, so a surface-speckle diff is not mistaken for the volume.
The framing rule (117.17) of SPEC-LIT 117.10 puts every eye at
R = 1.1 r / sin(theta_v / 2) from the box centre (r = the box half-diagonal,
theta_v = the vertical field of view) so every box corner stays inside the
frustum, with the clipping range [0.01 R, 10 (R + r)] and, on the USD
camera, f = V / (2 tan(theta_v / 2)) with V = 20.955 * height / width.

Only the SUPERVISOR runs this, with Isaac's own Python
(C:/iss/env/Scripts/python.exe), never this repository's tooling and never
here: isaacsim, omni, carb and pxr are imported only inside the Isaac-only
functions, after SimulationApp starts; the module scope stays stdlib +
numpy only.  --selftest runs P1-P10 in plain Python (no Isaac) and is what
this unit executes; it must print SELFTEST PASS 10/10.

tools/drone/isaac_probe.py is a thin caller of this module; usda_check.py
(VIO-09) is the scene checker beside it.  See README.md in this directory.
"""

import argparse
import json
import math
import os
import re
import shutil
import struct
import sys
import tempfile
import threading
import time
import zlib

import numpy as np

TOOL = "tools/isaac/probe.py"
VERSION = "isaac-probe/1"
DIFF_THR = 4                      # /255, per pixel = max over RGB of |a - b|
DIFF_MIN_FRAC = 0.0005            # (117.15): the pair must differ on more than this
NOISE_FACTOR = 2                  # (117.16): diff must beat NOISE_FACTOR * noise, strict
MARGIN = 1.1                      # (117.17): R = MARGIN r / sin(theta_v / 2)
DEFAULT_VFOV_DEG = 60.0
H_APERTURE = 20.955               # UsdGeom.Camera's default horizontalAperture
VIEW_DIRS = [("front34", (1.0, -1.0, 0.6)),   # box centre -> eye, Z-up, not normalised
             ("side", (0.0, -1.0, 0.0)),
             ("back34", (-1.0, 1.0, 0.6))]
_MAIN_T0 = None                   # process start, set in run; the watchdog's wall_s


def _progress(what, t0):
    """One [probe] heartbeat line, flushed - what the supervisor reads when
    the outer timeout has to be interpreted."""
    print("[probe] " + str(what) + " " + ("%.1f" % (time.time() - t0)) + " s",
          flush=True)


def diff_verdict(a, b, thr=DIFF_THR):
    """(117.15): does a show the volume, given the control b? Keys
    changed_px, changed_frac, maxdiff, volume_visible; a shape mismatch
    counts as no difference at all."""
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
    """(117.16): diff must call the volume visible AND its changed_px must
    beat NOISE_FACTOR * the b/c re-render noise, strict - a and b are
    separate accumulations whose speckle is noise, not the volume."""
    if diff is None or noise is None:
        return False
    return bool(diff["volume_visible"]
                and diff["changed_px"] > NOISE_FACTOR * noise["changed_px"])


def up_vector(up_axis):
    """The world up the stage's upAxis names, as a plain tuple."""
    if up_axis == "Z":
        return (0.0, 0.0, 1.0)
    if up_axis == "Y":
        return (0.0, 1.0, 0.0)
    raise ValueError("up_vector: unknown up axis %r (want \"Z\" or \"Y\")"
                     % (up_axis,))


def to_stage_frame(d, up_axis):
    """A Z-up view direction into the stage's frame: a Z stage keeps it, a Y
    stage turns (dx, dy, dz) into (dx, dz, -dy)."""
    if up_axis == "Z":
        return (float(d[0]), float(d[1]), float(d[2]))
    if up_axis == "Y":
        return (float(d[0]), float(d[2]), -float(d[1]))
    raise ValueError("to_stage_frame: unknown up axis %r (want \"Z\" or \"Y\")"
                     % (up_axis,))


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    """16 floats, row-major Gf.Matrix4d rows: right, up, -forward, eye."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up0 = np.asarray(up, dtype=np.float64)
    fwd = target - eye
    if float(np.linalg.norm(fwd)) == 0.0:
        raise ValueError("look_at: eye == target")
    fwd = fwd / np.linalg.norm(fwd)
    if float(np.linalg.norm(np.cross(fwd, up0))) < 1e-12:
        raise ValueError("look_at: looking along the up axis %r"
                         % (tuple(float(v) for v in up0),))
    right = np.cross(fwd, up0)
    right = right / np.linalg.norm(right)
    uprow = np.cross(right, fwd)
    m = [right[0], right[1], right[2], 0.0,
         uprow[0], uprow[1], uprow[2], 0.0,
         -fwd[0], -fwd[1], -fwd[2], 0.0,
         eye[0], eye[1], eye[2], 1.0]
    return [float(v) for v in m]


def focal_length(vfov_deg, width, height):
    """The UsdGeom.Camera focalLength for theta_v: with the apertures in the
    same unit, theta_v = 2 atan(V / (2 f)), so f = V / (2 tan(theta_v / 2))
    with V = H_APERTURE * height / width.  theta_v binds only for a
    landscape frame: width < height is refused."""
    if int(width) < int(height):
        raise ValueError("focal_length: width %s < height %s: the theta_v "
                         "framing binds only for a landscape frame"
                         % (width, height))
    v = H_APERTURE * float(height) / float(width)
    return v / (2.0 * math.tan(math.radians(float(vfov_deg)) / 2.0))


def frame_box(lo, hi, vfov_deg=DEFAULT_VFOV_DEG, up_axis="Z", dirs=VIEW_DIRS):
    """(117.17): frame the axis-aligned box lo..hi - centre, half-diagonal r,
    eye distance R = MARGIN r / sin(theta_v / 2), the clipping range
    [0.01 R, 10 (R + r)] and one (name, eye) per direction in dirs (box
    centre -> eye, taken in the stage's frame via to_stage_frame)."""
    lo = tuple(float(v) for v in lo)
    hi = tuple(float(v) for v in hi)
    if any(hi[i] < lo[i] for i in range(3)):
        raise ValueError("frame_box: hi %r < lo %r on some axis" % (hi, lo))
    if not 0.0 < float(vfov_deg) < 180.0:
        raise ValueError("frame_box: vfov_deg %r not in (0, 180)" % (vfov_deg,))
    centre = tuple((lo[i] + hi[i]) / 2.0 for i in range(3))
    r = 0.5 * math.sqrt(sum((hi[i] - lo[i]) ** 2 for i in range(3)))
    if r == 0.0:
        raise ValueError("frame_box: degenerate box, r == 0")
    R = MARGIN * r / math.sin(math.radians(float(vfov_deg)) / 2.0)
    views = []
    for name, d in dirs:
        ds = to_stage_frame(d, up_axis)
        n = math.sqrt(ds[0] ** 2 + ds[1] ** 2 + ds[2] ** 2)
        views.append((name, (centre[0] + R * ds[0] / n,
                             centre[1] + R * ds[1] / n,
                             centre[2] + R * ds[2] / n)))
    return {"centre": centre, "r": r, "R": R,
            "clip": (0.01 * R, 10.0 * (R + r)), "views": views}


def renderer_for(profile):
    """(renderer, [(carb setting, value), ...]) per profile: plain
    RaytracedLighting for index; PathTracing with the volume-shading
    settings for pathtrace.  Anything else (auto unresolved included) is
    refused."""
    if profile == "index":
        return ("RaytracedLighting", [])
    if profile == "pathtrace":
        return ("PathTracing",
                [("/rtx/raytracing/indexdirect/enabled", False),
                 ("/rtx/index/compositeEnabled", False),
                 ("/rtx/pathtracing/ptvol/enabled", True),
                 ("/rtx/pathtracing/nonUniformVolumes", True)])
    raise ValueError("renderer_for: unknown profile %r (want \"index\" or "
                     "\"pathtrace\"; resolve auto first)" % (profile,))


def scene_info(text):
    """Parse .usda TEXT (no pxr): the stage's upAxis and metersPerUnit with
    USD's Y / 0.01 fallbacks, and the profile - pathtrace when a prim
    declares primvars:isVolume, index when a `def Volume "` survives, None
    when the scene says neither."""
    up_axis = "Y"
    mpu = 0.01
    profile = None
    for raw in str(text).splitlines():
        line = raw.strip()
        m = re.match(r'upAxis\s*=\s*"(Z|Y)"', line)
        if m:
            up_axis = m.group(1)
        m = re.match(r"metersPerUnit\s*=\s*([-+0-9.eE]+)", line)
        if m:
            try:
                mpu = float(m.group(1))
            except ValueError:
                pass
        if re.search(r"primvars:isVolume\s*=\s*(1|true)\b", line):
            profile = "pathtrace"
        elif profile is None and 'def Volume "' in line:
            profile = "index"
    return {"profile": profile, "up_axis": up_axis, "meters_per_unit": mpu}


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


def write_probe_json(out_dir, result):
    """One probe.json: the same dict the ISAAC-PROBE line prints, indent 2;
    returns the file path."""
    path = os.path.join(str(out_dir), "probe.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
        fh.write("\n")
    return path


def arm_watchdog(seconds, out_dir, result, exit_fn=os._exit):
    """The daemon Timer run arms right after SimulationApp starts. The
    in-loop --max-seconds checks only run between calls, so a hang inside
    one blocking call never returns: when the Timer fires it records
    timed_out, writes probe.json, prints the ISAAC-PROBE line and exits 3.
    Returns the Timer (run cancels it after a normal finish)."""
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


def _is_volume_prim(prim, usdvol, usdgeom):
    """A UsdVol.Volume, or a UsdGeom.Mesh flagged primvars:isVolume (the
    pathtrace fixture's box-with-material stand-in for a volume)."""
    if prim.IsA(usdvol.Volume):
        return True
    if prim.IsA(usdgeom.Mesh):
        attr = prim.GetAttribute("primvars:isVolume")
        if attr is not None and attr.Get() in (True, 1):
            return True
    return False


def build_parser(prog):
    """The generic command line; callers may add their own options (the
    drone caller adds --field and pins --renderer) before parse_args."""
    ap = argparse.ArgumentParser(prog=prog,
                                 description=__doc__.splitlines()[0])
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--profile", choices=["auto", "index", "pathtrace"],
                    default="auto",
                    help="auto = scene_info of the scene file's text")
    ap.add_argument("--renderer", default=None,
                    help="default: the profile's renderer")
    ap.add_argument("--volume", default=None,
                    help="the volume prim path (default: the first found)")
    ap.add_argument("--box", type=float, nargs=6, default=None,
                    metavar=("LO_X", "LO_Y", "LO_Z", "HI_X", "HI_Y", "HI_Z"),
                    help="the box to frame (default: the volume prim's bound)")
    ap.add_argument("--vfov-deg", type=float, default=DEFAULT_VFOV_DEG)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--warmup", type=int, default=120)
    ap.add_argument("--max-seconds", type=float, default=540.0)
    return ap


def run(args, prepare=None, views=None, clip=None, tool=TOOL, version=VERSION,
        extra=None):
    """The Isaac part (the supervisor only): SimulationApp, the profile's
    renderer settings, the stage, the volume prim, the cameras and the a/b/c
    frames.  `prepare` (drone-style callers) does the scene-specific staging
    and returns the target prim; `views` (a list of (name, eye, target))
    replaces the (117.17) framing and pairs with `clip`, leaving the lens at
    the USD default.  Writes probe.json, prints the ISAAC-PROBE line,
    returns 1 on an exception else 0."""
    global _MAIN_T0
    t0 = time.time()
    _MAIN_T0 = t0
    os.makedirs(str(args.out_dir), exist_ok=True)
    result = {"tool": tool, "version": version, "isaac_version": None,
              "profile": None, "renderer": None,
              "scene": str(args.scene).replace(os.sep, "/"),
              "stage_loaded": False, "mesh_prims": 0, "volume_prims": 0,
              "volume_path": None, "box": None,
              "frames": [], "diff": None, "noise": None,
              "volume_visible": False, "wall_s": 0.0, "timed_out": False,
              "error": None}
    if extra:
        result.update(extra)
    sim = None
    watchdog = None
    err = None

    try:
        profile = args.profile
        if profile == "auto":
            with open(args.scene, encoding="utf-8") as fh:
                profile = scene_info(fh.read())["profile"]
        settings = renderer_for(profile)[1]
        renderer = (args.renderer if args.renderer is not None
                    else renderer_for(profile)[0])
        result["profile"] = profile
        result["renderer"] = renderer
        from isaacsim import SimulationApp
        sim = SimulationApp({"headless": True, "renderer": renderer,
                             "width": args.width, "height": args.height,
                             "anti_aliasing": 1})
        _progress("SimulationApp up", t0)
        watchdog = arm_watchdog(float(args.max_seconds), args.out_dir, result)
        import carb.settings
        cset = carb.settings.get_settings()
        for key, val in settings:
            cset.set(key, val)
        _run_scene(sim, args, result, t0, prepare, views, clip)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        err = exc
    if watchdog is not None:
        watchdog.cancel()
    result["wall_s"] = round(time.time() - t0, 1)
    result["error"] = None if err is None else type(err).__name__ + ": " + str(err)
    write_probe_json(args.out_dir, result)
    print("ISAAC-PROBE " + json.dumps(result, default=str), flush=True)
    if sim is not None:
        sim.close()
    return 1 if err is not None else 0


def _run_scene(sim, args, result, t0, prepare, views, clip):
    """The stage work run() calls inside its try: open the scene, wait for
    the load, light it, pick the volume prim, build the camera plan (the
    (117.17) framing when views is None, the caller's (name, eye, target)
    triples otherwise) and render the a/b/c frames like the drone probe."""
    import omni.replicator.core as rep
    import omni.usd
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdVol

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
    vol_prims = [p for p in stage.Traverse()
                 if _is_volume_prim(p, UsdVol, UsdGeom)]
    result["volume_prims"] = len(vol_prims)
    target = None
    if prepare is not None:
        target = prepare(stage, args, result)
    elif args.volume:
        prim = stage.GetPrimAtPath(args.volume)
        if prim.IsValid():
            target = prim
    elif vol_prims:
        target = vol_prims[0]
    if target is not None:
        result["volume_path"] = str(target.GetPath())
    cam_up = (0.0, 0.0, 1.0)
    lens = None
    if views is None:
        if args.box is not None:
            lo = tuple(float(v) for v in args.box[0:3])
            hi = tuple(float(v) for v in args.box[3:6])
        elif target is not None:
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                      ["default", "render"])
            rng = cache.ComputeWorldBound(target).ComputeAlignedRange()
            lo = tuple(float(v) for v in rng.GetMin())
            hi = tuple(float(v) for v in rng.GetMax())
        else:
            raise ValueError("run: no volume prim found and no --box: "
                             "nothing to frame")
        result["box"] = [lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]]
        up_axis = UsdGeom.GetStageUpAxis(stage) or "Y"
        frame = frame_box(lo, hi, args.vfov_deg, up_axis)
        cam_up = up_vector(up_axis)
        plan = [(name, eye, frame["centre"]) for name, eye in frame["views"]]
        clip = frame["clip"]
        lens = (focal_length(args.vfov_deg, args.width, args.height),
                H_APERTURE,
                H_APERTURE * float(args.height) / float(args.width))
    else:
        plan = [(str(name), tuple(eye), tuple(tgt))
                for name, eye, tgt in views]
        if clip is None:
            clip = (0.01, 50.0)

    nviews = max(1, min(int(args.frames), len(plan)))
    for vi in range(nviews):
        if time.time() - t0 > args.max_seconds:
            result["timed_out"] = True
            break
        vname, eye, tgt = plan[vi]
        cam_path = "/World/ProbeCam_" + vname
        cam = UsdGeom.Camera.Define(stage, cam_path)
        camx = UsdGeom.Xformable(cam.GetPrim())
        camx.ClearXformOpOrder()
        camx.AddTransformOp().Set(Gf.Matrix4d(*look_at(eye, tgt, cam_up)))
        cam.CreateClippingRangeAttr(Gf.Vec2f(float(clip[0]), float(clip[1])))
        if lens is not None:
            cam.CreateFocalLengthAttr(float(lens[0]))
            cam.CreateHorizontalApertureAttr(float(lens[1]))
            cam.CreateVerticalApertureAttr(float(lens[2]))
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
            result["frames"].append({"view": vname, "png": png,
                                     "mean": float(a.mean()),
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


def main(argv=None):
    """parse_args + run: the supervisor's entry, never executed in this unit."""
    args = build_parser("isaac_probe.py").parse_args(argv)
    return run(args)


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
    """volume_effect (117.16): None on either side is False; the run-3
    numbers are False against noise 30000 and True against 1000; the strict
    inequality (2000 vs 2*1000) is False; a not-visible diff is False even
    against zero noise."""
    x = {"changed_px": 32284, "volume_visible": True}
    assert volume_effect(None, x) is False
    assert volume_effect(x, None) is False
    assert volume_effect(x, {"changed_px": 30000}) is False
    assert volume_effect(x, {"changed_px": 1000}) is True
    assert volume_effect({"changed_px": 2000, "volume_visible": True},
                         {"changed_px": 1000}) is False
    assert volume_effect({"changed_px": 2000, "volume_visible": True},
                         {"changed_px": 999}) is True
    assert volume_effect({"changed_px": 5, "volume_visible": False},
                         {"changed_px": 0}) is False


def _p5():
    """frame_box (117.17) on the unit cube: centre, r, R, clip, the three
    view names and eyes, every eye exactly R from the centre."""
    f = frame_box((0, 0, 0), (2, 2, 2), 60.0, "Z")
    for i in range(3):
        assert abs(f["centre"][i] - 1.0) < 1e-12, f["centre"]
    assert abs(f["r"] - 1.7320508075688772) < 1e-12, f["r"]
    assert abs(f["R"] - 3.8105117766515306) < 1e-12, f["R"]
    assert abs(f["clip"][0] - 0.03810511776651531) < 1e-12, f["clip"]
    assert abs(f["clip"][1] - 55.42562584220407) < 1e-12, f["clip"]
    assert [v[0] for v in f["views"]] == ["front34", "side", "back34"], f["views"]
    eyes = dict(f["views"])
    want = {"front34": (3.4804318924093356, -1.4804318924093356, 2.4882591354456016),
            "side": (1.0, -2.8105117766515306, 1.0),
            "back34": (-1.4804318924093356, 3.4804318924093356, 2.4882591354456016)}
    for name, w in want.items():
        for a, b in zip(w, eyes[name]):
            assert abs(a - b) < 1e-12, (name, eyes[name])
    for _, eye in f["views"]:
        d = math.sqrt(sum((eye[i] - f["centre"][i]) ** 2 for i in range(3)))
        assert abs(d - f["R"]) < 1e-12, (eye, d, f["R"])


def _p6():
    """The (117.17) framing keeps the drone lattice box inside the frustum:
    for every view and all 8 corners the angle between the view axis and
    corner - eye stays within asin(sin 30 deg / 1.1) = 27.035691789412287
    deg, hence under 30 deg."""
    lo = (-1.2, -0.6, -0.8)
    hi = (0.6, 0.6, 0.4)
    f = frame_box(lo, hi, 60.0, "Z")
    centre = np.array(f["centre"])
    corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                        for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    limit = math.degrees(math.asin(math.sin(math.radians(30.0)) / MARGIN))
    worst = 0.0
    for _, eye in f["views"]:
        eye = np.asarray(eye, dtype=np.float64)
        axis = centre - eye
        axis = axis / np.linalg.norm(axis)
        for corner in corners:
            v = corner - eye
            v = v / np.linalg.norm(v)
            ang = math.degrees(math.acos(
                float(np.clip(float(np.dot(axis, v)), -1.0, 1.0))))
            worst = max(worst, ang)
    print("[p6] largest corner angle " + repr(worst)
          + " deg (limit " + repr(limit) + ")", flush=True)
    assert worst <= limit + 1e-9 and worst < 30.0, worst


def _p7():
    """look_at stays the drone's formula (the default Z up gives the same 16
    floats), a Y-up stage frames through to_stage_frame with the Y up row,
    and the guards raise."""
    m = look_at((0, -2, 0), (0, 0, 0))
    assert len(m) == 16 and all(isinstance(v, float) for v in m), m
    assert tuple(m[8:11]) == (0.0, -1.0, 0.0), m
    assert tuple(m[12:15]) == (0.0, -2.0, 0.0), m
    assert m[15] == 1.0, m
    f = frame_box((0, 0, 0), (2, 2, 2), 60.0, "Y")
    side = dict(f["views"])["side"]
    for a, b in zip((1.0, 1.0, 1.0 + 3.8105117766515306), side):
        assert abs(a - b) < 1e-12, side
    m2 = look_at(side, f["centre"], up_vector("Y"))
    for a, b in zip((0.0, 1.0, 0.0), m2[4:7]):
        assert abs(a - b) < 1e-12, m2
    for fn in (lambda: look_at((0, 0, 5), (0, 0, 0)),
               lambda: up_vector("X"),
               lambda: frame_box((0, 0, 0), (1, -1, 1))):
        try:
            fn()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def _p8():
    """The lens and the renderer table: f closes theta_v, portrait frames
    are refused, renderer_for is exactly the contract's table."""
    f = focal_length(60.0, 1280, 720)
    assert abs(f - 10.208003814170388) < 1e-12, f
    v = H_APERTURE * 720 / 1280
    assert abs(v - 11.787187499999998) < 1e-12, v
    ang = math.degrees(2.0 * math.atan(v / (2.0 * f)))
    assert abs(ang - 60.0) < 1e-12, ang
    try:
        focal_length(60.0, 720, 1280)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    assert renderer_for("index") == ("RaytracedLighting", [])
    assert renderer_for("pathtrace") == (
        "PathTracing",
        [("/rtx/raytracing/indexdirect/enabled", False),
         ("/rtx/index/compositeEnabled", False),
         ("/rtx/pathtracing/ptvol/enabled", True),
         ("/rtx/pathtracing/nonUniformVolumes", True)])
    try:
        renderer_for("nvdb")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def _p9():
    """scene_info reads the two fixtures and the empty stage, and
    write_probe_json round-trips a full result dict."""
    here = os.path.dirname(os.path.abspath(__file__))
    for prof, want in [("index", ("index", "Z", 1.0)),
                       ("pathtrace", ("pathtrace", "Z", 1.0))]:
        path = os.path.join(here, "fixtures", prof, "isaac.usda")
        with open(path, encoding="utf-8") as fh:
            info = scene_info(fh.read())
        got = (info["profile"], info["up_axis"], info["meters_per_unit"])
        assert got == want, (path, info)
    info = scene_info('#usda 1.0\ndef Xform "World"\n{\n}\n')
    assert info == {"profile": None, "up_axis": "Y", "meters_per_unit": 0.01}, info
    d = {"tool": TOOL, "version": VERSION, "isaac_version": None,
         "profile": "index", "renderer": "RaytracedLighting",
         "scene": "fixtures/index/isaac.usda",
         "stage_loaded": True, "mesh_prims": 25, "volume_prims": 1,
         "volume_path": "/World/density",
         "box": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
         "frames": [{"view": "front34", "png": "isaac_front34.png",
                     "mean": 12.5, "min": 0, "max": 255}],
         "diff": {"changed_px": 100, "changed_frac": 0.01, "maxdiff": 50,
                  "volume_visible": True},
         "noise": {"changed_px": 300, "changed_frac": 0.0001, "maxdiff": 9,
                   "volume_visible": False},
         "volume_visible": True, "wall_s": 74.5, "timed_out": False,
         "error": None, "field": "wake"}
    tmp = tempfile.mkdtemp(prefix="isaac-probe-p9-")
    try:
        p = write_probe_json(tmp, d)
        assert os.path.basename(p) == "probe.json", p
        with open(p, encoding="utf-8") as fh:
            assert json.load(fh) == d, "probe.json round-trip changed the dict"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _p10():
    """arm_watchdog fires: exit code 3 through exit_fn, probe.json on disk
    with timed_out true; the module stays isaac-free."""
    tmp = tempfile.mkdtemp(prefix="isaac-probe-p10-")
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
    for mod in ("isaacsim", "omni", "carb", "pxr"):
        assert mod not in sys.modules, mod + " leaked into module scope"


def selftest():
    """P1-P10 in plain Python - no isaacsim/omni/carb/pxr import anywhere."""
    tests = [("P1", _p1), ("P2", _p2), ("P3", _p3), ("P4", _p4), ("P5", _p5),
             ("P6", _p6), ("P7", _p7), ("P8", _p8), ("P9", _p9),
             ("P10", _p10)]
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
