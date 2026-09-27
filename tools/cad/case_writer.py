#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""case_writer.py - the S7 case of the CAD loop (docs/16 §D S7, §H.2, §I; docs/16a §E CASE-BIND, §H): one level of
the wedge mesh, its geom.json and a locked requirement set become a cold-start laminar ofgpu-lowmach case in the
OpenFOAM layout. The numerics are transcribed from the solver tree at bceb799 (turek_hron.rs steady_controls() and
turek_rules(), blockgen.rs write_system() for T), quoted with their sha in fixtures/case/sources.json. Every patch
of the boundary gets an explicit condition in U, p and T. Every input is hashed through common.stable_file_snapshot
before and after the write (CASE-BIND).

Usage:
  python case_writer.py --selftest
  python case_writer.py write WEDGE_DIR LEVEL GEOM_DIR REQUIREMENTS_DIR OUT_DIR [ROLES_JSON]
  python case_writer.py scan CASE_DIR
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
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common
import reqs
sys.path.insert(0, os.path.join(common.REPO, "tools", "mesh"))
import polymesh_write

CASE_VERSION = "cad-case/1"
CASE_WRITER_VERSION = "1"          # enters the eval key (docs/16 §D) as case_writer_version
MACH_LIMIT = 0.25                  # docs/16 §E.1: the case writer refuses a predicted U_e/c > 0.25
NU_AIR = 1.5e-5                    # m2/s, blockgen.rs:3604 at bceb799; docs/16 §H.2 (Re_De 3e4 at 22.5 m/s, 20 mm)
GAS = {"R_universal": 8.314462618, "W": 0.0289647, "gamma": 1.4}   # energy.rs:311-329 GasProperties::default()
STATE = {"fluid": "air", "T_K": 293.15, "p0_Pa": 101325.0}         # docs/16 §H.2; the one state NU_AIR holds
SPAN_REL = 1e-9                    # mesh x span against geom.json x_span_m
SWIRL_TOL = 1e-12                  # a normalised inlet direction's y or z component above this is swirl
POLYMESH_FILES = ("boundary", "faces", "neighbour", "owner", "points")
FIELDS = ("U", "p", "T")
ROLES = ("velocity_inlet", "pressure_outlet", "wall", "slip", "wedge")
ROLE_TYPES = {"velocity_inlet": "patch", "pressure_outlet": "patch", "wall": "wall", "slip": "patch",
              "wedge": "wedge"}
NOZZLE_ROLES = {"version": "cad-roles/1",
                "patches": {"inlet": {"role": "velocity_inlet", "direction": [1.0, 0.0, 0.0]},
                            "outlet": {"role": "pressure_outlet"}, "wall_nozzle": {"role": "wall"},
                            "slip_upstream": {"role": "slip"}, "wedge_front": {"role": "wedge"},
                            "wedge_back": {"role": "wedge"}}}
REFUSAL_IDS = ("CASE-OUT", "CASE-BIND", "GATE-LOCK", "CASE-UNITS", "CASE-FLUID", "CASE-NOBC", "CASE-SWIRL",
               "CASE-MACH")      # in the order the rules run; the first failing rule is the refusal
INPUT_KEYS = ("wedge_mesh.json", "polyMesh/boundary", "polyMesh/faces", "polyMesh/neighbour", "polyMesh/owner",
              "polyMesh/points", "geom.json", "requirements.json", "requirements.lock")
CASE_KEYS = ("version", "case_writer_version", "status", "level", "inputs", "wedge", "operating_point", "patches",
             "fields", "numerics", "sources", "cold_start", "files")
OP_KEYS = ("fluid", "T_K", "p0_Pa", "requirements_lock", "Q_m3_s", "A_inlet_m2", "A_outlet_m2", "U_inlet_m_s",
           "U_exit_m_s", "nu_m2_s", "R_s", "gamma", "c_m_s", "mach_exit", "mach_limit")
WEDGE_KEYS = ("recipe_sha", "geom_sha256", "template_sha", "params_sha", "msh_sha256", "cells", "gc6_pass")
SOURCES = os.path.join(common.FIXTURES, "case", "sources.json")
GOLDEN = os.path.join(common.FIXTURES, "case", "golden_nominal.json")
USAGE = ("usage: python case_writer.py --selftest" + chr(10)
         + "       python case_writer.py write WEDGE_DIR LEVEL GEOM_DIR REQUIREMENTS_DIR OUT_DIR [ROLES_JSON]"
         + chr(10) + "       python case_writer.py scan CASE_DIR")

# The two quoted spans of the solver tree at bceb799 live in fixtures/case/sources.json; DIFFERENCES records what
# the case intentionally does NOT copy, and every TRANSCRIBED row is (source id, quote substring, case file,
# written substring) - T1 proves both halves of every row.
DIFFERENCES = (
    "nu: turek_hron.rs NU 1e-3 is the Turek-Hron fluid; the case writes air, NU_AIR 1.5e-5 (blockgen.rs:3604)",
    "variable_viscosity_stress: steady_controls() sets false; lowmach has no case key and fixes it itself",
    "report_continuity: not a case setting; lowmach prints continuity by default",
    "inlet: turek_rules() Inlet is the Turek-Hron parabola; docs/16 section H.2 asks for a uniform fixedValue",
    "outlet U: turek_rules() ZeroGradient; docs/16 section H.2 asks for inletOutlet",
    "T: steady_controls() has no energy equation; the T entries are blockgen.rs write_system()'s",
)
TRANSCRIBED = (
    ("turek_hron", "solver: LinearSolverKind::PCG", "system/fvSolution", "solver          PCG;"),
    ("turek_hron", "precon: Preconditioner::Dic", "system/fvSolution", "preconditioner  DIC;"),
    ("turek_hron", "rel_tol: 0.05", "system/fvSolution", "relTol          0.05;"),
    ("turek_hron", "max_iter: 2000", "system/fvSolution", "maxIter         2000;"),
    ("turek_hron", "tolerance: 1e-10", "system/fvSolution", "tolerance       1e-10;"),
    ("turek_hron", "check_interval: 5", "system/fvSolution", "checkInterval   5;"),
    ("turek_hron", "solver: LinearSolverKind::PBiCGStab", "system/fvSolution", "solver          PBiCGStab;"),
    ("turek_hron", "precon: Preconditioner::Dilu", "system/fvSolution", "preconditioner  DILU;"),
    ("turek_hron", "rel_tol: 0.1", "system/fvSolution", "relTol          0.1;"),
    ("turek_hron", "max_iter: 200", "system/fvSolution", "maxIter         200;"),
    ("turek_hron", "u_relax: 0.7", "system/fvSolution", "U               0.7;"),
    ("turek_hron", "p_relax: 0.3", "system/fvSolution", "p               0.3;"),
    ("turek_hron", "n_non_orth_correctors: 1", "system/fvSolution", "nNonOrthogonalCorrectors 1;"),
    ("turek_hron", "n_correctors: 1", "system/fvSolution", "nCorrectors     1;"),
    ("turek_hron", "n_outer_correctors: 1", "system/fvSolution", "nOuterCorrectors 1;"),
    ("turek_hron", "momentum_predictor: true", "system/fvSolution", "momentumPredictor yes;"),
    ("turek_hron", "div_scheme: DivScheme::LinearUpwind", "system/fvSchemes", "Gauss linearUpwind grad(U);"),
    ("turek_hron", "bounded_convection: true", "system/fvSchemes", "div(phi,U)      bounded Gauss"),
    ("turek_hron", "sn_grad: crate::fv::SnGradScheme::Corrected", "system/fvSchemes", "default         corrected;"),
    ("turek_hron", '("outlet", VelocityBc::ZeroGradient, PressureBc::Fixed(0.0))', "0/p",
     "type            fixedValue;"),
    ("blockgen", "div(phi,T)       bounded Gauss upwind;", "system/fvSchemes",
     "div(phi,T)      bounded Gauss upwind;"),
    ("blockgen", "default         steadyState;", "system/fvSchemes", "default         steadyState;"),
    ("blockgen", "preconditioner  diagonal;", "system/fvSolution", "preconditioner  diagonal;"),
    ("blockgen", "T               0.7;", "system/fvSolution", "T               0.7;"),
)

_DASH = "-" * 75
BANNER = ("/*" + _DASH + "*" + chr(92) + chr(10)                       # the solver's own io/fields.rs banner
          + "| ofgpu  --  GPU-native finite volume CFD".ljust(78) + "|" + chr(10)
          + "|" + " " * 77 + "|" + chr(10)
          + "| Written in the OpenFOAM ASCII case format so that existing pre- and".ljust(78) + "|" + chr(10)
          + "| post-processing tools can read it. A file format is not a work: ofgpu is".ljust(78) + "|" + chr(10)
          + "| an independent implementation, neither derived from nor affiliated with".ljust(78) + "|" + chr(10)
          + "| OpenFOAM.".ljust(78) + "|" + chr(10)
          + chr(92) + "*" + _DASH + "*/")
SEPARATOR = "//" + " *" * 37 + " //"
FOOTER_RULE = "// " + "*" * 73 + " //"


class Refused(Exception):
    """A refusal by id (wedge_mesh.Refused's shape): rule and detail."""

    def __init__(self, rule, detail):
        super().__init__("%s: %s" % (rule, detail))
        self.rule = rule
        self.detail = detail


def fmt(x) -> str:
    """repr(float(x)): 2.5 -> '2.5', 0 -> '0.0', 1.5e-5 -> '1.5e-05'."""
    return repr(float(x))


def _uniform(v) -> str:
    """A uniform value: a vector as (a b c), a scalar plain through fmt."""
    if isinstance(v, (list, tuple)):
        return "(" + " ".join(fmt(c) for c in v) + ")"
    return fmt(v)


def bc_table(role, u_in, t_k) -> dict:
    """{"U": bc, "p": bc, "T": bc} for one role: turek_rules()'s table with docs/16 section H.2's changes."""
    if role == "velocity_inlet":
        return {"U": {"type": "fixedValue", "value": [float(v) for v in u_in]},
                "p": {"type": "zeroGradient"},
                "T": {"type": "fixedValue", "value": t_k}}
    if role == "pressure_outlet":
        return {"U": {"type": "inletOutlet", "inletValue": [0.0, 0.0, 0.0], "value": [0.0, 0.0, 0.0]},
                "p": {"type": "fixedValue", "value": 0.0},
                "T": {"type": "inletOutlet", "inletValue": t_k, "value": t_k}}
    if role == "wall":
        return {"U": {"type": "noSlip"}, "p": {"type": "zeroGradient"}, "T": {"type": "zeroGradient"}}
    if role == "slip":
        return {"U": {"type": "slip"}, "p": {"type": "slip"}, "T": {"type": "slip"}}
    if role == "wedge":
        return {"U": {"type": "wedge"}, "p": {"type": "wedge"}, "T": {"type": "wedge"}}
    raise ValueError("bc_table: unknown role %r" % (role,))


def foam_file(cls, location, obj, body) -> str:
    """The solver's banner, the FoamFile header, the writer's mark, then the body between the two rules."""
    return (BANNER + chr(10) + "FoamFile" + chr(10) + "{" + chr(10)
            + "    version     2.0;" + chr(10)
            + "    format      ascii;" + chr(10)
            + "    class       %s;" % cls + chr(10)
            + '    location    "%s";' % location + chr(10)
            + "    object      %s;" % obj + chr(10)
            + "}" + chr(10)
            + "// written by tools/cad/case_writer.py, " + CASE_VERSION + chr(10)
            + SEPARATOR + chr(10) + chr(10)
            + body + chr(10) + chr(10) + FOOTER_RULE + chr(10))


_CONTROL_DICT = (
    "application     foamRun;", "startFrom       startTime;", "startTime       0;", "stopAt          endTime;",
    "endTime         1;", "deltaT          1;", "writeControl    timeStep;", "writeInterval   1;",
    "purgeWrite      0;", "writeFormat     ascii;", "writePrecision  6;", "writeCompression off;",
    "timeFormat      general;", "timePrecision   6;", "runTimeModifiable true;")

_FV_SCHEMES = (
    "ddtSchemes", "{", "    default         steadyState;", "}", "",
    "gradSchemes", "{", "    default         Gauss linear;", "}", "",
    "divSchemes", "{", "    default         none;",
    "    div(phi,U)      bounded Gauss linearUpwind grad(U);",
    "    div(phi,T)      bounded Gauss upwind;", "}", "",
    "laplacianSchemes", "{", "    default         Gauss linear corrected;", "}", "",
    "interpolationSchemes", "{", "    default         linear;", "}", "",
    "snGradSchemes", "{", "    default         corrected;", "}")

_FV_SOLUTION = (
    "solvers", "{",
    "    p", "    {",
    "        solver          PCG;", "        preconditioner  DIC;", "        tolerance       1e-10;",
    "        relTol          0.05;", "        maxIter         2000;", "        checkInterval   5;", "    }", "",
    "    U", "    {",
    "        solver          PBiCGStab;", "        preconditioner  DILU;", "        tolerance       1e-10;",
    "        relTol          0.1;", "        maxIter         200;", "        checkInterval   5;", "    }", "",
    "    T", "    {",
    "        solver          PBiCGStab;", "        preconditioner  diagonal;", "        tolerance       1e-08;",
    "        relTol          0.01;", "        maxIter         200;", "    }", "}",
    "",
    "SIMPLE", "{",
    "    nNonOrthogonalCorrectors 1;", "    nCorrectors     1;", "    nOuterCorrectors 1;",
    "    momentumPredictor yes;", "}",
    "",
    "relaxationFactors", "{",
    "    fields", "    {", "        p               0.3;", "    }", "",
    "    equations", "    {", "        U               0.7;", "        T               0.7;", "    }", "}")


def system_files() -> dict:
    """The three system files: blockgen.rs write_system()'s controlDict, Gate 105-C's fvSchemes/fvSolution."""
    return {"system/controlDict": foam_file("dictionary", "system", "controlDict", chr(10).join(_CONTROL_DICT)),
            "system/fvSchemes": foam_file("dictionary", "system", "fvSchemes", chr(10).join(_FV_SCHEMES)),
            "system/fvSolution": foam_file("dictionary", "system", "fvSolution", chr(10).join(_FV_SOLUTION))}


def constant_files() -> dict:
    """physicalProperties (air nu at the one STATE) and the laminar momentumTransport."""
    phys = chr(10).join(["viscosityModel  constant;", "",
                         "nu              [0 2 -1 0 0 0 0] " + fmt(NU_AIR) + ";"])
    return {"constant/physicalProperties": foam_file("dictionary", "constant", "physicalProperties", phys),
            "constant/momentumTransport": foam_file("dictionary", "constant", "momentumTransport",
                                                    "simulationType  laminar;")}


_DIMS = {"U": "[0 1 -1 0 0 0 0]", "p": "[0 2 -2 0 0 0 0]", "T": "[0 0 0 1 0 0 0]"}


def field_files(patches, t_k) -> dict:
    """0/U, 0/p and 0/T: dimensions, a uniform internal field, and one explicit block per boundary patch in
    boundary-file order; keywords padded to 16 columns."""
    internal = {"U": "(0.0 0.0 0.0)", "p": "0.0", "T": fmt(t_k)}
    out = {}
    for f in FIELDS:
        lines = ["dimensions      " + _DIMS[f] + ";", "",
                 "internalField   uniform " + internal[f] + ";", "", "boundaryField", "{"]
        for row in patches:
            bc = row[f]
            lines.append("    " + row["name"])
            lines.append("    {")
            lines.append("        type            %s;" % bc["type"])
            if "inletValue" in bc:
                lines.append("        inletValue      uniform %s;" % _uniform(bc["inletValue"]))
            if "value" in bc:
                lines.append("        value           uniform %s;" % _uniform(bc["value"]))
            lines.append("    }")
        lines.append("}")
        cls = "volVectorField" if f == "U" else "volScalarField"
        out["0/" + f] = foam_file(cls, "0", f, chr(10).join(lines))
    return out


def _bound(path, want_sha, name):
    """(bytes, sha) of one input through common.stable_file_snapshot; CASE-BIND when unbound, wrong or changing."""
    snap = common.stable_file_snapshot(path)
    if snap["stable"] is not True:
        raise Refused("CASE-BIND", "%s: not a stable regular file (unbound)" % name)
    if want_sha is not None and snap["sha256"] != want_sha:
        raise Refused("CASE-BIND", "%s: sha %s differs from the wedge-mesh record's %s"
                      % (name, snap["sha256"], want_sha))
    with open(path, "rb") as f:
        data = f.read()
    if common.sha256_bytes(data) != snap["sha256"]:
        raise Refused("CASE-BIND", "%s: changed while read" % name)
    return data, snap["sha256"]


def read_bound(path, want_sha, name) -> bytes:
    """One input's bytes, bound to its stable snapshot and, when given, to want_sha."""
    return _bound(path, want_sha, name)[0]


def _input_paths(wedge_dir, level, geom_dir, req_dir) -> dict:
    """The nine inputs by INPUT_KEYS name; no path ever enters case.json."""
    pm = os.path.join(wedge_dir, "L%d" % level, "case", "constant", "polyMesh")
    return {"wedge_mesh.json": os.path.join(wedge_dir, "wedge_mesh.json"),
            "polyMesh/boundary": os.path.join(pm, "boundary"),
            "polyMesh/faces": os.path.join(pm, "faces"),
            "polyMesh/neighbour": os.path.join(pm, "neighbour"),
            "polyMesh/owner": os.path.join(pm, "owner"),
            "polyMesh/points": os.path.join(pm, "points"),
            "geom.json": os.path.join(geom_dir, "geom.json"),
            "requirements.json": os.path.join(req_dir, "requirements.json"),
            "requirements.lock": os.path.join(req_dir, "requirements.lock")}


def write_case(wedge_dir, level, geom_dir, req_dir, out_dir, roles=None, between_hook=None) -> dict:
    """The rules of docs/16 section I CAD-13 in their order; never raises Refused - the refusal or the case comes
    back. Every file lands in a temp directory that becomes out_dir only on success, so a refusal leaves nothing."""
    level = int(level)
    out_dir = os.path.abspath(out_dir)
    if os.path.lexists(out_dir):
        return {"status": "refused", "rule": "CASE-OUT", "message": "%s exists" % out_dir, "case": None}
    parent = os.path.dirname(out_dir)
    os.makedirs(parent, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=".case-", dir=parent)
    try:
        doc = _write_case(wedge_dir, level, geom_dir, req_dir, tmp, roles)
        if between_hook is not None:
            between_hook()
        for key, path in _input_paths(wedge_dir, level, geom_dir, req_dir).items():
            snap = common.stable_file_snapshot(path)
            if snap["sha256"] != doc["inputs"][key]:
                raise Refused("CASE-BIND", "%s changed while the case was written" % key)
    except Refused as r:
        shutil.rmtree(tmp, ignore_errors=True)
        return {"status": "refused", "rule": r.rule, "message": r.detail, "case": None}
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    retries = 0
    while True:
        try:
            os.replace(tmp, out_dir)
            break
        except PermissionError:
            retries += 1                      # Windows scanners hold the new directory briefly
            if retries > 5:
                raise
            time.sleep(0.05)
    return {"status": "ok", "rule": None, "message": "", "case": doc}


def _write_case(wedge_dir, level, geom_dir, req_dir, tmp, roles) -> dict:
    """write_case's body: CASE-BIND, GATE-LOCK, CASE-UNITS, CASE-FLUID, CASE-NOBC, CASE-SWIRL, CASE-MACH, then
    the sixteen files (the five polyMesh copies, the eight case files, case.json last)."""
    paths = _input_paths(wedge_dir, level, geom_dir, req_dir)
    inputs = {}
    wbytes, inputs["wedge_mesh.json"] = _bound(paths["wedge_mesh.json"], None, "wedge_mesh.json")
    record = json.loads(wbytes.decode("utf-8"))
    rows = [lv for lv in record["levels"] if lv["level"] == level]
    if not rows:
        raise Refused("CASE-BIND", "level %d is not in the wedge-mesh record" % level)
    row = rows[0]
    pm_bytes = {}
    for name in POLYMESH_FILES:
        want = row.get("polymesh_sha256", {}).get(name)
        if want is None:
            raise Refused("CASE-BIND", "polyMesh/%s is not recorded in the wedge-mesh level row" % name)
        pm_bytes[name], inputs["polyMesh/" + name] = _bound(paths["polyMesh/" + name], want,
                                                            "polyMesh/" + name)
    gbytes, inputs["geom.json"] = _bound(paths["geom.json"], record["geom_sha256"], "geom.json")
    doc = lock_text = None
    for key in ("requirements.json", "requirements.lock"):
        blob, inputs[key] = _bound(paths[key], None, key)
        if key == "requirements.json":
            doc = json.loads(blob.decode("utf-8"))
        else:
            lock_text = blob.decode("utf-8")
    if not reqs.lock_ok(doc) or lock_text.strip() != doc["lock_sha"]:
        raise Refused("GATE-LOCK", "the requirements lock does not match its document")
    geom = json.loads(gbytes.decode("utf-8"))
    os.makedirs(os.path.join(tmp, "constant", "polyMesh"))
    for name in POLYMESH_FILES:
        with open(os.path.join(tmp, "constant", "polyMesh", name), "wb") as f:
            f.write(pm_bytes[name])
    for key, want in (("units", "m"), ("scale", 1), ("axis", "+x"), ("step_length_unit", "METRE")):
        if geom.get(key) != want:
            raise Refused("CASE-UNITS", "geom.json %s is %r, want %r" % (key, geom.get(key), want))
    pm = polymesh_write.read_polymesh(os.path.join(tmp, "constant", "polyMesh"))
    span = float(np.max(pm["points"][:, 0]) - np.min(pm["points"][:, 0]))
    if abs(span - geom["x_span_m"]) > SPAN_REL * geom["x_span_m"]:
        raise Refused("CASE-UNITS", "mesh x span %r m against geom.json x_span_m %r m"
                      % (span, geom["x_span_m"]))
    op = doc["operating_point"]
    for key in ("fluid", "T_K", "p0_Pa"):
        if op.get(key) != STATE[key]:
            raise Refused("CASE-FLUID", "operating_point %s is %r, want %r; NU_AIR holds only there"
                          % (key, op.get(key), STATE[key]))
    if roles is None:
        roles = NOZZLE_ROLES
    if not isinstance(roles, dict) or roles.get("version") != "cad-roles/1" \
            or not isinstance(roles.get("patches"), dict):
        raise Refused("CASE-NOBC", "the roles document must be cad-roles/1 with a patches dict")
    rmap = roles["patches"]
    patches = pm["patches"]
    names = [p["name"] for p in patches]
    for n in names:
        if n not in rmap:
            raise Refused("CASE-NOBC", "boundary patch %s has no role" % n)
    for n in rmap:
        if n not in names:
            raise Refused("CASE-NOBC", "role patch %s is not in the boundary" % n)
    for p in patches:
        role = rmap[p["name"]].get("role")
        if role not in ROLES:
            raise Refused("CASE-NOBC", "patch %s has role %r, want one of %s"
                          % (p["name"], role, ", ".join(ROLES)))
        if p["type"] != ROLE_TYPES[role]:
            raise Refused("CASE-NOBC", "patch %s (role %s) has boundary type %s, want %s"
                          % (p["name"], role, p["type"], ROLE_TYPES[role]))
    inlets = [n for n in names if rmap[n].get("role") == "velocity_inlet"]
    outlets = [n for n in names if rmap[n].get("role") == "pressure_outlet"]
    if len(inlets) != 1 or len(outlets) != 1:
        raise Refused("CASE-NOBC", "want exactly one velocity_inlet and one pressure_outlet, got %d and %d"
                      % (len(inlets), len(outlets)))
    tags = geom.get("tags", {}).get("fluid_faces", {})
    for n in (inlets[0], outlets[0]):
        if n not in tags:
            raise Refused("CASE-NOBC", "patch %s is not a fluid face of geom.json's tags" % n)
    a_inlet, a_outlet = tags[inlets[0]]["area_m2"], tags[outlets[0]]["area_m2"]
    d = rmap[inlets[0]].get("direction")
    if not isinstance(d, (list, tuple)) or len(d) != 3 \
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in d):
        raise Refused("CASE-SWIRL", "the velocity_inlet's direction must be a list of 3 finite numbers")
    norm = math.sqrt(float(d[0]) ** 2 + float(d[1]) ** 2 + float(d[2]) ** 2)
    if not norm > 0.0:
        raise Refused("CASE-SWIRL", "the inlet direction's norm must be positive")
    n = [float(v) / norm for v in d]
    if abs(n[1]) > SWIRL_TOL or abs(n[2]) > SWIRL_TOL or n[0] <= 0:
        raise Refused("CASE-SWIRL", "the wedge's symmetry alias cannot carry swirl or a non-axial inflow")
    q = reqs.flow_Q(doc)
    u_inlet, u_exit = q / a_inlet, q / a_outlet
    r_s = GAS["R_universal"] / GAS["W"]
    c = math.sqrt(GAS["gamma"] * r_s * op["T_K"])
    mach = u_exit / c
    if mach > MACH_LIMIT:
        raise Refused("CASE-MACH", "predicted U_e/c is %r, over %r" % (mach, MACH_LIMIT))
    u_in = [u_inlet * n[0], u_inlet * n[1], u_inlet * n[2]]
    patch_rows = []
    for p in patches:
        role = rmap[p["name"]]["role"]
        bcs = bc_table(role, u_in, op["T_K"])
        patch_rows.append({"name": p["name"], "type": p["type"], "n_faces": p["nFaces"],
                           "start_face": p["startFace"], "role": role,
                           "U": bcs["U"], "p": bcs["p"], "T": bcs["T"]})
    files = dict(("constant/polyMesh/" + nm, common.sha256_bytes(pm_bytes[nm])) for nm in POLYMESH_FILES)

    def put(rel, text):
        blob = text.encode("utf-8")
        target = os.path.join(tmp, rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "wb") as f:
            f.write(blob)
        files[rel] = common.sha256_bytes(blob)

    for rel, text in system_files().items():
        put(rel, text)
    for rel, text in constant_files().items():
        put(rel, text)
    for rel, text in sorted(field_files(patch_rows, op["T_K"]).items()):
        put(rel, text)
    case = {"version": CASE_VERSION, "case_writer_version": CASE_WRITER_VERSION, "status": "ok",
            "level": level, "inputs": dict((k, inputs[k]) for k in INPUT_KEYS),
            "wedge": {"recipe_sha": record["recipe_sha"], "geom_sha256": record["geom_sha256"],
                      "template_sha": record["template_sha"], "params_sha": record["params_sha"],
                      "msh_sha256": row["msh_sha256"], "cells": row["cells"], "gc6_pass": row["gc6"]["pass"]},
            "operating_point": {"fluid": op["fluid"], "T_K": op["T_K"], "p0_Pa": op["p0_Pa"],
                                "requirements_lock": doc["lock_sha"], "Q_m3_s": q,
                                "A_inlet_m2": a_inlet, "A_outlet_m2": a_outlet, "U_inlet_m_s": u_inlet,
                                "U_exit_m_s": u_exit, "nu_m2_s": NU_AIR, "R_s": r_s, "gamma": GAS["gamma"],
                                "c_m_s": c, "mach_exit": mach, "mach_limit": MACH_LIMIT},
            "patches": patch_rows,
            "fields": {"U": {"dimensions": "[0 1 -1 0 0 0 0]", "internal": [0.0, 0.0, 0.0]},
                       "p": {"dimensions": "[0 2 -2 0 0 0 0]", "internal": 0.0},
                       "T": {"dimensions": "[0 0 0 1 0 0 0]", "internal": op["T_K"]}},
            "numerics": {"transcribed": [list(r) for r in TRANSCRIBED], "differences": list(DIFFERENCES)},
            "sources": [{"id": s["id"], "file": s["file"], "commit": s["commit"], "blob": s["blob"],
                         "lines": s["lines"], "text_sha256": s["text_sha256"]}
                        for s in common.read_json(SOURCES)["sources"]],
            "cold_start": True, "files": files}
    with open(os.path.join(tmp, "case.json"), "wb") as f:
        f.write((common.canonical_json(case) + chr(10)).encode("utf-8"))
    return case


def scan_bcs(case_dir) -> dict:
    """{"U": [missing], "p": [...], "T": [...]}: every boundary patch missing from a field's boundaryField, or
    present with no type line - a patch with no entry would silently hold its cell value (field_setup.rs:3046)."""
    names = [p["name"] for p in
             polymesh_write.read_polymesh(os.path.join(case_dir, "constant", "polyMesh"))["patches"]]
    out = {}
    for f in FIELDS:
        with open(os.path.join(case_dir, "0", f), "r", encoding="utf-8") as fh:
            text = fh.read()
        seg = text[text.index("boundaryField"):]
        entries = dict((m.group(1), m.group(2)) for m in re.finditer(
            r"    ([A-Za-z_][A-Za-z0-9_]*)\n    \{\n(.*?)\n    \}", seg, re.S))
        out[f] = sorted(n for n in names
                        if n not in entries or not re.search(r"^\s*type\s+\S+;", entries[n], re.M))
    return out


def main(argv) -> int:
    """--selftest; write (6-7 args); scan (2 args). ValueError/OSError print and exit 2, anything else usage."""
    if argv == ["--selftest"]:
        try:
            selftest()
        except Exception:
            traceback.print_exc()
            return 1
        return 0
    try:
        if len(argv) in (6, 7) and argv[0] == "write":
            roles_json = common.read_json(argv[6]) if len(argv) == 7 else None
            res = write_case(argv[1], argv[2], argv[3], argv[4], argv[5], roles=roles_json)
            if res["status"] == "ok":
                print(common.canonical_json({"status": "ok", "files": len(res["case"]["files"]) + 1}))
                return 0
            print(common.canonical_json({"refused": res["rule"], "detail": res["message"]}))
            return 1
        if len(argv) == 2 and argv[0] == "scan":
            scan = scan_bcs(argv[1])
            print(common.canonical_json(scan))
            return 0 if not (scan["U"] or scan["p"] or scan["T"]) else 1
    except (ValueError, OSError) as e:
        print("case_writer: %s" % (e,), file=sys.stderr)
        return 2
    print(USAGE, file=sys.stderr)
    return 2


def selftest() -> None:
    """T1..T9 of docs/16 section I CAD-13: the golden, the bindings, the scan and every refusal id."""
    import time
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        import export
        import wedge_mesh

        # setup: the nominal geometry, its L0 wedge and the locked exit-velocity study (air, 293.15 K, 101325 Pa)
        gdir = os.path.join(td, "geom")
        res = export.run_pipeline(export.TEMPLATE, dict(export.NOMINAL), gdir)
        assert res["status"] == "ok", (res["status"], res["rule"], res["message"])
        wdir = os.path.join(td, "wedge")
        res = wedge_mesh.run(gdir, wdir, levels=(0,))
        assert res["status"] == "ok", (res["status"], res["rule"], res["message"])
        sdir = os.path.join(td, "study")
        study_doc = common.read_json(os.path.join(common.FIXTURES, "reqs", "golden",
                                                  "v3_exit_velocity.json"))["requirements"]
        reqs.write_locked(sdir, study_doc)

        def lock_variant(name, **op_changes):
            doc = common.read_json(os.path.join(common.FIXTURES, "reqs", "golden",
                                                "v3_exit_velocity.json"))["requirements"]
            doc["operating_point"].update(op_changes)
            doc["lock_sha"] = reqs.lock_sha_of(doc)
            reqs.write_locked(os.path.join(td, name), doc)
            return os.path.join(td, name)

        def wedge_copy(name):
            return shutil.copytree(wdir, os.path.join(td, name))

        def rebind(wd, level=0, geom_dir=None):
            path = os.path.join(wd, "wedge_mesh.json")
            rep = common.read_json(path)
            row = [lv for lv in rep["levels"] if lv["level"] == level][0]
            pd = os.path.join(wd, "L%d" % level, "case", "constant", "polyMesh")
            row["polymesh_sha256"] = dict((nm, common.sha256_file(os.path.join(pd, nm)))
                                          for nm in POLYMESH_FILES)
            if geom_dir is not None:
                rep["geom_sha256"] = common.sha256_file(os.path.join(geom_dir, "geom.json"))
            common.atomic_write(path, common.canonical_json(rep) + chr(10))

        def refused(res, rule, out):
            assert res["status"] == "refused", (rule, res)
            assert res["rule"] == rule, (rule, res["rule"], res["message"])
            assert res["case"] is None, rule
            assert not os.path.lexists(out), out
            parent = os.path.dirname(os.path.abspath(out))
            assert not [n for n in os.listdir(parent) if n.startswith(".case-")], parent

        # T1: the two quotes hash to their recorded sha, and all 24 transcribed settings occur in the quote
        # and in the written file (the nominal write of T2 serves here)
        src = common.read_json(SOURCES)
        assert len(src["sources"]) == 2, len(src["sources"])
        for s in src["sources"]:
            assert common.sha256_bytes(s["text"].encode("utf-8")) == s["text_sha256"], s["id"]
        c0 = os.path.join(td, "case0")
        res = write_case(wdir, 0, gdir, sdir, c0)
        assert res["status"] == "ok", (res["status"], res["rule"], res["message"])
        case_texts = {}
        for rel in ("0/T", "0/U", "0/p", "constant/physicalProperties", "constant/momentumTransport",
                    "system/controlDict", "system/fvSchemes", "system/fvSolution"):
            with open(os.path.join(c0, rel.replace("/", os.sep)), "rb") as f:
                case_texts[rel] = f.read().decode("utf-8")
        quote_text = dict((s["id"], s["text"]) for s in src["sources"])
        for sid, q_sub, c_file, w_sub in TRANSCRIBED:
            assert q_sub in quote_text[sid], (sid, q_sub)
            assert w_sub in case_texts[c_file], (c_file, w_sub)
        print("[ok] 2 quotes at bceb799 hash to their recorded sha; 24 transcribed settings occur in their "
              "quote and in the written case")

        # T2: the nominal operating point, the cold start and the canonical, path-free case.json
        case = res["case"]
        opi = case["operating_point"]
        assert abs(opi["U_inlet_m_s"] - 2.5) <= 1e-12 * 2.5, opi["U_inlet_m_s"]
        assert abs(opi["U_exit_m_s"] - 22.5) <= 1e-12 * 22.5, opi["U_exit_m_s"]
        assert abs(opi["mach_exit"] - 0.0655527591094952) <= 1e-12, opi["mach_exit"]
        assert abs(opi["c_m_s"] - 343.23498058132736) <= 1e-9, opi["c_m_s"]
        assert sorted(os.listdir(c0)) == ["0", "case.json", "constant", "system"], os.listdir(c0)
        assert sorted(os.listdir(os.path.join(c0, "0"))) == ["T", "U", "p"], os.listdir(os.path.join(c0, "0"))
        p_json = os.path.join(c0, "case.json")
        with open(p_json, "rb") as f:
            blob = f.read()
        assert blob == (common.canonical_json(case) + chr(10)).encode("utf-8"), "case.json is not canonical"
        assert sorted(case) == sorted(CASE_KEYS), sorted(case)
        assert sorted(case["operating_point"]) == sorted(OP_KEYS), sorted(case["operating_point"])
        assert sorted(case["wedge"]) == sorted(WEDGE_KEYS), sorted(case["wedge"])
        assert case["cold_start"] is True
        text = blob.decode("ascii")
        for poison in (td, td.replace(chr(92), "/"), "C:"):
            assert poison not in text, poison
        print("[ok] nominal: U_inlet %r m/s, U_exit %r m/s, mach_exit %r"
              % (opi["U_inlet_m_s"], opi["U_exit_m_s"], opi["mach_exit"]))

        # T3: the fresh nominal case is byte-identical to the recorded golden, all 14 files
        gold = common.read_json(GOLDEN)
        for rel in sorted(gold["texts"]):
            with open(os.path.join(c0, rel.replace("/", os.sep)), "rb") as f:
                assert f.read().decode("utf-8") == gold["texts"][rel], rel

        def sha_map(case_dir):
            m = {}
            for root, dirs, names in os.walk(case_dir):
                for nm in names:
                    p = os.path.join(root, nm)
                    m[os.path.relpath(p, case_dir).replace(os.sep, "/")] = common.sha256_file(p)
            return m

        assert sha_map(c0) == gold["sha256"], sorted(set(sha_map(c0).items()) ^ set(gold["sha256"].items()))
        print("[ok] nominal case byte-identical to golden (14 files)")

        # T4: the inputs map holds the nine snapshot shas; the copied polyMesh equals the wedge record's shas
        for key, path in _input_paths(wdir, 0, gdir, sdir).items():
            snap = common.stable_file_snapshot(path)
            assert snap["stable"] is True and case["inputs"][key] == snap["sha256"], key
        rec = common.read_json(os.path.join(wdir, "wedge_mesh.json"))
        row0 = [lv for lv in rec["levels"] if lv["level"] == 0][0]
        for name in POLYMESH_FILES:
            assert case["files"]["constant/polyMesh/" + name] == row0["polymesh_sha256"][name], name
        print("[ok] inputs: 9 stable shas bound; the copied polyMesh equals the wedge record's shas")

        # T5: the BC scan over the 6 nominal patches, then 7 with a zero-face patch, then a planted hole
        assert scan_bcs(c0) == {"U": [], "p": [], "T": []}, scan_bcs(c0)
        wz = wedge_copy("w_zero")
        pm = polymesh_write.read_polymesh(os.path.join(wz, "L0", "case", "constant", "polyMesh"))
        n_faces = len(pm["faces"])
        bpath = os.path.join(wz, "L0", "case", "constant", "polyMesh", "boundary")
        with open(bpath, "r", encoding="utf-8") as f:
            btxt = f.read()
        assert btxt.count(chr(10) + "6" + chr(10) + "(") == 1, "the count line 6 is not where T5 expects it"
        btxt = btxt.replace(chr(10) + "6" + chr(10) + "(", chr(10) + "7" + chr(10) + "(", 1)
        block = ("    stray" + chr(10) + "    {" + chr(10) + "        type            wall;" + chr(10)
                 + "        nFaces          0;" + chr(10) + "        startFace       " + str(n_faces) + ";"
                 + chr(10) + "    }" + chr(10))
        j = btxt.rindex(")")
        common.atomic_write(bpath, btxt[:j] + block + btxt[j:])
        rebind(wz)
        roles7 = {"version": "cad-roles/1",
                  "patches": dict(NOZZLE_ROLES["patches"], stray={"role": "wall"})}
        cz = os.path.join(td, "case_z")
        res = write_case(wz, 0, gdir, sdir, cz, roles=roles7)
        assert res["status"] == "ok", (res["rule"], res["message"])
        assert scan_bcs(cz) == {"U": [], "p": [], "T": []}, scan_bcs(cz)
        zrow = [p for p in res["case"]["patches"] if p["name"] == "stray"][0]
        assert zrow["U"]["type"] == "noSlip" and zrow["p"]["type"] == "zeroGradient" \
            and zrow["T"]["type"] == "zeroGradient", zrow
        ch = os.path.join(td, "case_holed")
        shutil.copytree(c0, ch)
        upath = os.path.join(ch, "0", "U")
        with open(upath, "r", encoding="utf-8") as f:
            utxt = f.read()
        utxt2 = re.sub(r"    inlet\n    \{\n.*?\n    \}\n", "", utxt, count=1, flags=re.S)
        assert utxt2 != utxt, "the inlet block was not found in 0/U"
        common.atomic_write(upath, utxt2)
        assert scan_bcs(ch) == {"T": [], "U": ["inlet"], "p": []}, scan_bcs(ch)
        print("[ok] every patch has a BC in U, p and T (6 nominal, 7 with a zero-face patch); "
              "a deleted entry is found")

        # T6: the plan's four refusals, each by its exact id (CASE-MACH at 86, passing at 85.8)
        res = write_case(wz, 0, gdir, sdir, os.path.join(td, "case_norole"))
        refused(res, "CASE-NOBC", os.path.join(td, "case_norole"))
        assert "stray" in res["message"], res["message"]
        s86 = lock_variant("m86", U_exit_m_s=86.0, flow_quote="at 86 m/s")
        res = write_case(wdir, 0, gdir, s86, os.path.join(td, "case_m86"))
        refused(res, "CASE-MACH", os.path.join(td, "case_m86"))
        s858 = lock_variant("m858", U_exit_m_s=85.8, flow_quote="at 85.8 m/s")
        res = write_case(wdir, 0, gdir, s858, os.path.join(td, "case_858"))
        assert res["status"] == "ok", (res["rule"], res["message"])
        assert 0.2499 < res["case"]["operating_point"]["mach_exit"] < 0.25, \
            res["case"]["operating_point"]["mach_exit"]
        swirl = {"version": "cad-roles/1",
                 "patches": dict(NOZZLE_ROLES["patches"],
                                 inlet=dict(NOZZLE_ROLES["patches"]["inlet"],
                                            direction=[1.0, 0.0, 1e-3]))}
        res = write_case(wdir, 0, gdir, sdir, os.path.join(td, "case_swirl"), roles=swirl)
        refused(res, "CASE-SWIRL", os.path.join(td, "case_swirl"))
        gmm = os.path.join(td, "g_mm")
        shutil.copytree(gdir, gmm)
        gdoc = common.read_json(os.path.join(gmm, "geom.json"))
        gdoc["units"] = "mm"
        common.write_json(os.path.join(gmm, "geom.json"), gdoc)
        wmm = wedge_copy("w_mm")
        rebind(wmm, geom_dir=gmm)
        res = write_case(wmm, 0, gmm, sdir, os.path.join(td, "case_mm"))
        refused(res, "CASE-UNITS", os.path.join(td, "case_mm"))
        wx = wedge_copy("w_x1000")
        ppath = os.path.join(wx, "L0", "case", "constant", "polyMesh", "points")
        pmx = polymesh_write.read_polymesh(os.path.dirname(ppath))
        with open(ppath, "r", encoding="utf-8") as f:
            old = f.read()
        i = old.index("(")
        lines = ["(" + " ".join(repr(float(v)) for v in row) + ")" for row in (pmx["points"] * 1000.0)]
        common.atomic_write(ppath, old[:i + 1] + chr(10) + chr(10).join(lines) + chr(10) + ")" + chr(10))
        rebind(wx)
        res = write_case(wx, 0, gdir, sdir, os.path.join(td, "case_x1000"))
        refused(res, "CASE-UNITS", os.path.join(td, "case_x1000"))
        print("[ok] refusals: CASE-NOBC, CASE-MACH (86 refused, 85.8 passes), CASE-SWIRL, "
              "CASE-UNITS (units mm), CASE-UNITS (points x1000)")

        # T7: CASE-BIND five ways - points, boundary, geom.json, a hard link, a mid-write rewrite
        bp = wedge_copy("b_pts")
        with open(os.path.join(bp, "L0", "case", "constant", "polyMesh", "points"), "ab") as f:
            f.write(b"// x" + common.NB)
        res = write_case(bp, 0, gdir, sdir, os.path.join(td, "case_bp"))
        refused(res, "CASE-BIND", os.path.join(td, "case_bp"))
        assert "polyMesh/points" in res["message"], res["message"]
        bb = wedge_copy("b_bnd")
        with open(os.path.join(bb, "L0", "case", "constant", "polyMesh", "boundary"), "ab") as f:
            f.write(b"// x" + common.NB)
        res = write_case(bb, 0, gdir, sdir, os.path.join(td, "case_bb"))
        refused(res, "CASE-BIND", os.path.join(td, "case_bb"))
        assert "polyMesh/boundary" in res["message"], res["message"]
        gb = os.path.join(td, "g_b")
        shutil.copytree(gdir, gb)
        with open(os.path.join(gb, "geom.json"), "ab") as f:
            f.write(common.NB)
        res = write_case(wdir, 0, gb, sdir, os.path.join(td, "case_gb"))
        refused(res, "CASE-BIND", os.path.join(td, "case_gb"))
        assert "geom.json" in res["message"], res["message"]
        bl = wedge_copy("b_link")
        os.link(os.path.join(bl, "L0", "case", "constant", "polyMesh", "points"),
                os.path.join(td, "points_link"))
        res = write_case(bl, 0, gdir, sdir, os.path.join(td, "case_bl"))
        refused(res, "CASE-BIND", os.path.join(td, "case_bl"))
        assert "unbound" in res["message"], res["message"]
        bh = wedge_copy("b_hook")

        def hook():
            with open(os.path.join(bh, "L0", "case", "constant", "polyMesh", "points"), "ab") as f:
                f.write(b"// y" + common.NB)

        res = write_case(bh, 0, gdir, sdir, os.path.join(td, "case_bh"), between_hook=hook)
        refused(res, "CASE-BIND", os.path.join(td, "case_bh"))
        assert "changed while the case was written" in res["message"], res["message"]
        print("[ok] CASE-BIND: points, boundary, geom.json, a hard link (unbound) and a mid-write rewrite refused")

        # T8: CASE-OUT, GATE-LOCK and CASE-FLUID (water; T_K 300)
        res = write_case(wdir, 0, gdir, sdir, c0)
        assert res["status"] == "refused" and res["rule"] == "CASE-OUT" and res["case"] is None, res
        assert not [n for n in os.listdir(td) if n.startswith(".case-")], td
        sb = os.path.join(td, "s_bad")
        shutil.copytree(sdir, sb)
        common.atomic_write(os.path.join(sb, "requirements.lock"), "0" * 64 + chr(10))
        res = write_case(wdir, 0, gdir, sb, os.path.join(td, "case_sbad"))
        refused(res, "GATE-LOCK", os.path.join(td, "case_sbad"))
        sw_ = lock_variant("s_water", fluid="water")
        res = write_case(wdir, 0, gdir, sw_, os.path.join(td, "case_water"))
        refused(res, "CASE-FLUID", os.path.join(td, "case_water"))
        assert "fluid" in res["message"], res["message"]
        s300 = lock_variant("s_300", T_K=300.0)
        res = write_case(wdir, 0, gdir, s300, os.path.join(td, "case_300"))
        refused(res, "CASE-FLUID", os.path.join(td, "case_300"))
        assert "T_K" in res["message"], res["message"]
        print("[ok] CASE-OUT, GATE-LOCK, CASE-FLUID (water; T_K 300) each refused")

        # T9: the CLI - two fresh processes byte-identical to the golden, a refusal exits 1 with its id,
        # usage exits 2, scan exits 0 and prints canonical JSON
        for name in ("cli_a", "cli_b"):
            r = subprocess.run([sys.executable, __file__, "write", wdir, "0", gdir, sdir,
                                os.path.join(td, name)],
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
            assert r.returncode == 0, (name, r.returncode, r.stdout, r.stderr)
            assert json.loads(r.stdout.strip().splitlines()[-1])["files"] == 14, r.stdout
        ma, mb = sha_map(os.path.join(td, "cli_a")), sha_map(os.path.join(td, "cli_b"))
        assert ma == mb == gold["sha256"], "the CLI writes differ from each other or from the golden"
        r = subprocess.run([sys.executable, __file__, "write", wdir, "0", gdir,
                            os.path.join(td, "s_water"), os.path.join(td, "cli_c")],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        assert r.returncode == 1, (r.returncode, r.stdout, r.stderr)
        assert json.loads(r.stdout.strip().splitlines()[-1])["refused"] == "CASE-FLUID", r.stdout
        assert main([]) == 2 and main(["write"]) == 2, "usage must exit 2"
        r = subprocess.run([sys.executable, __file__, "scan", os.path.join(td, "cli_a")],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
        assert json.loads(r.stdout.strip().splitlines()[-1]) == {"U": [], "p": [], "T": []}, r.stdout
        print("[ok] CLI: two fresh processes byte-identical to the golden; a refusal exits 1 with its id; "
              "usage exits 2; scan exits 0")

    print("selftest wall %.1f s" % (time.time() - t0))
    print("SELFTEST PASS")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
