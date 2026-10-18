#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.

"""Try the DRONE-ISAAC scene in Isaac Sim, headless, in memory only.

Thin caller of tools/isaac/probe.py, the generic headless visibility probe
(SPEC-LIT 117.10): this file only wires the drone scene - the /World/Flow
staging in prepare() below and the drone's front34 / side / wake34 camera
triples - and hands everything else (SimulationApp, the renderer settings,
the lights, the render loop, the a/b/c grabs and PNGs, diff_verdict /
volume_effect, the watchdog, probe.json and the ISAAC-PROBE line) to the
generic module, loaded from ../isaac/probe.py by path.  It opens
tools/drone/isaac_export.py's drone_flow.usda, hides every /World/Flow
volume except the requested field's (wake by default), adds the `density`
field relationship Isaac's volume path shades, then renders the first
--frames views (clipping 0.01-50, --warmup rep.orchestrator.step each)
through one render_product + rgb annotator per view, saving
isaac_<view>.png and, for the first view, the two volume-hidden control
frames (b and c grabbed back to back) plus diff, noise and
volume_visible = volume_effect(diff, noise) ((117.15)-(117.16) of
SPEC-LIT 117.10).

The scene's flow values are demonstration numbers from an unsteady run,
not validated; wake is |U - U_inf| / |U_inf|, derived from U.  The drone
model: (c) 2022 Rudis Laboratories / PX4 Autopilot for Drones,
BSD-3-Clause.

Run by the SUPERVISOR with Isaac's own Python
(C:/iss/env/Scripts/python.exe), never by this repository's tooling and
never here: isaacsim, omni and carb are imported only inside the probe's
run, after SimulationApp starts.

--selftest runs P1-P8 in plain Python (no Isaac needed).
"""

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "isaac_probe_generic",
    Path(__file__).resolve().parents[1] / "isaac" / "probe.py")
isaac_probe_generic = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("isaac_probe_generic", isaac_probe_generic)
_SPEC.loader.exec_module(isaac_probe_generic)

TOOL = "tools/drone/isaac_probe.py"
VERSION = "drone-isaac-probe/1"
TARGET = (-0.15, 0.0, -0.08)
VIEWS = [("front34", (1.2, 0.9, 0.6)), ("side", (0.0, -1.6, 0.05)), ("wake34", (-1.4, 0.7, 0.7))]   # eye positions (m)

# the names the selftest uses, re-exported from the generic module
DIFF_THR = isaac_probe_generic.DIFF_THR
DIFF_MIN_FRAC = isaac_probe_generic.DIFF_MIN_FRAC
diff_verdict = isaac_probe_generic.diff_verdict
volume_effect = isaac_probe_generic.volume_effect
look_at = isaac_probe_generic.look_at
write_probe_json = isaac_probe_generic.write_probe_json
arm_watchdog = isaac_probe_generic.arm_watchdog


def safe_prim_name(field):
    """The isaac_export rule, re-implemented: [^A-Za-z0-9_] -> "_", digit first gets "_". """
    s = "".join(c if (c.isascii() and c.isalnum()) or c == "_" else "_" for c in str(field))
    return ("_" + s) if s[:1].isdigit() else s


def prepare(stage, args, result):
    """The drone scene's staging, which the generic probe calls after the
    load: hide every /World/Flow volume except safe_prim_name(args.field)'s,
    give the target its `density` field relationship, make it visible, and
    record the /World/Flow volume count.  Returns the target prim, or None."""
    from pxr import Sdf, UsdGeom, UsdVol

    target = None
    want = safe_prim_name(args.field)
    flow = stage.GetPrimAtPath("/World/Flow")
    if flow.IsValid():
        result["volume_prims"] = sum(1 for p in flow.GetChildren()
                                     if p.IsA(UsdVol.Volume))
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
    return target


def main(argv=None):
    """The generic probe's command line plus --field, with the drone's
    cameras, clip range and pinned renderer."""
    parser = isaac_probe_generic.build_parser("isaac_probe.py")
    parser.add_argument("--field", default="wake")
    parser.set_defaults(renderer="RaytracedLighting")
    args = parser.parse_args(argv)
    nviews = max(1, min(int(args.frames), len(VIEWS)))
    return isaac_probe_generic.run(
        args, prepare=prepare,
        views=[(name, eye, TARGET) for name, eye in VIEWS[:nviews]],
        clip=(0.01, 50.0), tool=TOOL, version=VERSION,
        extra={"field": args.field})


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
