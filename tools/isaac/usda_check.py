#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.

"""Check an Isaac scene (SPEC-LIT 117.9, VIO-09) opens and is what it claims.

usda_check.py SCENE.usda [--profile index|pathtrace]   check one scene
usda_check.py --selftest                               the fixtures + mutants

usd-core (pxr, Apache-2.0) does the reading; the volume it references is
NEVER opened - the check is of the stage text. Every check prints one
`[ok] ...` line; the script exits 0 when all pass, 1 otherwise, in well
under a minute. An `.nvdb` reference fails on sight: an Isaac scene never
references one (SPEC-LIT 117.9).
"""

import argparse
import re
import shutil
import sys
import tempfile
from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdShade, UsdVol

try:
    import pxr.UsdValidation as UsdValidation
except ImportError:  # pragma: no cover - usd-core without the module
    UsdValidation = None

FIXTURES = Path(__file__).resolve().parent / "fixtures"

_failures: list[str] = []


def ok(name: str, detail: str = "") -> None:
    print(f"[ok] {name}" + (f" - {detail}" if detail else ""))


def fail(name: str, why: str) -> None:
    print(f"[FAIL] {name} - {why}")
    _failures.append(name)


def check(cond: bool, name: str, why: str, detail: str = "") -> bool:
    if cond:
        ok(name, detail)
    else:
        fail(name, why)
    return cond


def asset_refs(attr) -> list[str]:
    """Every asset value of `attr`: the default, then each time sample."""
    refs = [v.path for v in ([attr.Get()] if attr.Get() is not None else [])]
    refs += [v.path for v in (attr.Get(t) for t in attr.GetTimeSamples()) if v is not None]
    return refs


def validate_topology(mesh: UsdGeom.Mesh, name: str) -> bool:
    """usd-core's own UsdGeom.Mesh.ValidateTopology on the mesh's counts,
    indices and point count, on a non-empty point list."""
    counts = mesh.GetFaceVertexCountsAttr().Get() or []
    indices = mesh.GetFaceVertexIndicesAttr().Get() or []
    points = mesh.GetPointsAttr().Get() or []
    valid, why = UsdGeom.Mesh.ValidateTopology(indices, counts, len(points))
    return check(
        bool(points) and valid, f"{name}: topology valid",
        why or f"{len(points)} points",
        f"{len(counts)} faces, {len(points)} points",
    )


def points_box(mesh: UsdGeom.Mesh):
    pts = mesh.GetPointsAttr().Get() or []
    if not pts:
        return None
    lo = Gf.Vec3f(*[min(p[i] for p in pts) for i in range(3)])
    hi = Gf.Vec3f(*[max(p[i] for p in pts) for i in range(3)])
    return lo, hi


def check_common(stage: Usd.Stage, text: str) -> None:
    """The checks every profile shares: the stage metadata, the default
    prim, the boundary surface when present, and the never-`.nvdb` rule."""
    # GetMetadata returns the schema fallback when nothing is authored, so
    # "set" is HasAuthoredMetadata.
    up = stage.GetMetadata("upAxis")
    check(stage.HasAuthoredMetadata("upAxis") and up in ("Y", "Z"), "stage: upAxis set",
          f"upAxis is {up!r} (authored: {stage.HasAuthoredMetadata('upAxis')})", str(up))
    mpu = stage.GetMetadata("metersPerUnit")
    check(stage.HasAuthoredMetadata("metersPerUnit") and mpu is not None and mpu > 0,
          "stage: metersPerUnit set",
          f"metersPerUnit is {mpu!r} (authored: {stage.HasAuthoredMetadata('metersPerUnit')})")
    dp = stage.GetDefaultPrim()
    check(bool(dp) and dp.GetName() == "World", "stage: defaultPrim World",
          f"defaultPrim is {dp.GetName() if bool(dp) else None!r}")
    check(".nvdb" not in text, "scene: no .nvdb reference",
          "the layer text names .nvdb - an Isaac scene never references one")
    bnd = stage.GetPrimAtPath("/World/boundary")
    if bool(bnd) and bnd.IsActive():
        validate_topology(UsdGeom.Mesh(bnd), "/World/boundary")


def check_index(stage: Usd.Stage) -> None:
    """The `index` profile: /World/density is a UsdVol.Volume whose field
    `density` is an OpenVDBAsset named `density`, referencing `.vdb` only."""
    prim = stage.GetPrimAtPath("/World/density")
    vol = UsdVol.Volume(prim) if bool(prim) else None
    check(vol and vol.GetPrim().IsDefined(), "index: /World/density is a UsdVol.Volume",
          "the prim is missing or not a Volume")
    if not vol:
        return
    paths = vol.GetFieldPaths()
    check("density" in paths, "index: field density is named",
          f"GetFieldPaths() has {sorted(paths)}", str(paths.get("density", "")))
    if "density" not in paths:
        return
    field = UsdVol.OpenVDBAsset(stage.GetPrimAtPath(str(paths["density"])))
    check(field and field.GetPrim().IsDefined(), "index: the field is an OpenVDBAsset",
          f"{paths['density']} is not an OpenVDBAsset")
    if not field:
        return
    check(field.GetFieldNameAttr().Get() == "density", "index: fieldName density",
          f"fieldName is {field.GetFieldNameAttr().Get()!r}")
    refs = asset_refs(field.GetFilePathAttr())
    check(bool(refs) and all(r.endswith(".vdb") for r in refs),
          "index: every filePath ends .vdb", f"filePath values {refs!r}", f"{len(refs)} ref(s)")


def check_pathtrace(stage: Usd.Stage) -> None:
    """The `pathtrace` profile: /World/density is a volume Mesh bound to an
    OmniVolumeDensity material whose texture asset(s) are `.vdb`, with the
    extent matching the points' box."""
    prim = stage.GetPrimAtPath("/World/density")
    mesh = UsdGeom.Mesh(prim) if bool(prim) else None
    check(mesh and mesh.GetPrim().IsDefined(), "pathtrace: /World/density is a UsdGeom.Mesh",
          "the prim is missing or not a Mesh")
    if not mesh:
        return
    validate_topology(mesh, "/World/density")
    check(prim.GetAttribute("primvars:isVolume").Get() is True,
          "pathtrace: primvars:isVolume True",
          f"isVolume is {prim.GetAttribute('primvars:isVolume').Get()!r}")
    check("MaterialBindingAPI" in prim.GetAppliedSchemas(),
          "pathtrace: MaterialBindingAPI applied", f"applied: {prim.GetAppliedSchemas()}")
    bound = UsdShade.MaterialBindingAPI(prim)
    mat, _ = bound.ComputeBoundMaterial()
    check(bool(mat), "pathtrace: a material is bound", "ComputeBoundMaterial found none")
    if not mat:
        return
    out = mat.GetVolumeOutput("mdl")
    conns = out.GetAttr().GetConnections()
    check(bool(conns), "pathtrace: the material's mdl volume output connects",
          f"{out.GetFullName()} has no connection")
    if not conns:
        return
    shader = UsdShade.Shader(stage.GetPrimAtPath(str(conns[0].GetPrimPath())))
    src = shader.GetSourceAsset("mdl")
    check(src.path == "OmniVolumeDensity.mdl", "pathtrace: mdl sourceAsset OmniVolumeDensity.mdl",
          f"sourceAsset is {src.path!r}")
    check(shader.GetSourceAssetSubIdentifier("mdl") == "OmniVolumeDensity",
          "pathtrace: mdl subIdentifier OmniVolumeDensity",
          f"subIdentifier is {shader.GetSourceAssetSubIdentifier('mdl')!r}")
    tex = shader.GetInput("volume_density_texture").GetAttr()
    refs = asset_refs(tex)
    check(bool(refs) and all(r.endswith(".vdb") for r in refs),
          "pathtrace: volume_density_texture ends .vdb", f"texture values {refs!r}",
          f"{len(refs)} ref(s)")
    box = points_box(mesh)
    extent = list(mesh.GetExtentAttr().Get() or [])
    check(bool(extent) and box is not None and tuple(extent[0]) == tuple(box[0])
          and tuple(extent[1]) == tuple(box[1]),
          "pathtrace: extent is the points' box",
          f"extent {extent} vs points box {box}")


def check_validation(stage: Usd.Stage) -> None:
    """UsdValidation with every registered validator. The only errors an
    Isaac scene may carry are the two usd-core cannot help: the MDL source
    asset (no MDL plugin here) and the 'mdl' source type missing from its
    sdr registry. Anything else fails."""
    if UsdValidation is None:
        fail("UsdValidation: registry unavailable", "pxr.UsdValidation did not import")
        return
    registry = UsdValidation.ValidationRegistry()
    ctx = UsdValidation.ValidationContext(registry.GetOrLoadAllValidators())
    errors = ctx.Validate(stage)
    allowed = ("unresolvable external dependency 'omnivolumedensity.mdl'",
               "sourcetype 'mdl' specified on shader prim")
    bad = [e.GetMessage() for e in errors
           if not any(a in e.GetMessage().lower() for a in allowed)]
    check(not bad, "UsdValidation: no unexpected errors",
          "; ".join(b[:160] for b in bad) or "all errors were allowed",
          f"{len(errors)} error(s), all allowed" if not bad else "")


def detect_profile(stage: Usd.Stage) -> str:
    prim = stage.GetPrimAtPath("/World/density")
    if bool(prim) and prim.IsA(UsdVol.Volume):
        return "index"
    if bool(prim) and prim.IsA(UsdGeom.Mesh):
        return "pathtrace"
    return "index"


def run(path: Path, profile: str | None) -> bool:
    _failures.clear()
    try:
        stage = Usd.Stage.Open(str(path))
        text = path.read_text(encoding="utf-8")
    except Exception as e:  # noqa: BLE001 - any Open failure is a failure
        fail("stage: Usd.Stage.Open", str(e))
        return False
    ok("stage: Usd.Stage.Open", str(path))
    check_common(stage, text)
    got = profile or detect_profile(stage)
    if got == "index":
        check_index(stage)
    else:
        check_pathtrace(stage)
    check_validation(stage)
    return not _failures


MUTANTS = [
    ("a .nvdb reference", "index", lambda t: t.replace(".vdb@", ".nvdb@")),
    ("fieldName density -> rho", "index",
     lambda t: t.replace('token fieldName = "density"', 'token fieldName = "rho"')),
    ("the upAxis line removed", "index", lambda t: t.replace('    upAxis = "Z"\n', "")),
    ("primvars:isVolume removed", "pathtrace",
     lambda t: "\n".join(l for l in t.splitlines()
                         if "primvars:isVolume" not in l) + "\n"),
]


VDB_REF = re.compile(r"@([^@]+[.]vdb)@")


def stage_copy(text: str, home: Path) -> Path:
    """`text` as home/isaac.usda, with an empty placeholder at every `.vdb`
    path it references, so the MissingReference validator resolves them:
    the fixtures carry no binary volume, and no check here reads one."""
    home.mkdir(parents=True)
    for ref in set(VDB_REF.findall(text)):
        target = home / ref
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    dst = home / "isaac.usda"
    dst.write_text(text, encoding="utf-8")
    return dst


def selftest() -> bool:
    """Both fixtures pass; every mutated copy fails. Each scene is checked
    as a copy staged by `stage_copy`."""
    good = True
    tmp = Path(tempfile.mkdtemp(prefix="usda_check_selftest_"))
    try:
        for profile in ("index", "pathtrace"):
            text = (FIXTURES / profile / "isaac.usda").read_text(encoding="utf-8")
            good &= run(stage_copy(text, tmp / profile), profile)
            print()
        for n, (label, profile, mutate) in enumerate(MUTANTS):
            text = (FIXTURES / profile / "isaac.usda").read_text(encoding="utf-8")
            passed = run(stage_copy(mutate(text), tmp / f"mutant{n}"), profile)
            if passed:
                fail(f"selftest: mutant FAILS - {label}", "the mutated scene passed")
                good = False
            else:
                ok(f"selftest: mutant fails as it should - {label}",
                   f"{len(_failures)} check(s) failed")
            print()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return good


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("scene", nargs="?", help="the SCENE.usda to check")
    ap.add_argument("--profile", choices=("index", "pathtrace"),
                    help="the profile to check (default: detected from the scene)")
    ap.add_argument("--selftest", action="store_true",
                    help="the fixtures pass, the mutants fail")
    args = ap.parse_args()
    if args.selftest:
        good = selftest()
    elif args.scene:
        good = run(Path(args.scene), args.profile)
    else:
        ap.error("give a SCENE.usda or --selftest")
    print("usda_check: PASS" if good else "usda_check: FAIL")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
