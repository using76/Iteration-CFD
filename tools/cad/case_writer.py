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

The turbulent writer (docs/16 §H.5, §I CAD-26) builds the kOmegaSST cases the same way, its BC rows and
turbulent numerics quoted from the solver at 08e10bb in fixtures/case/sources_turb.json: write_pipe_case the
TG0 periodic pipe driven by a constant/fvSources momentumSource at g_x = 2 u_tau^2/R, write_turb_case the
turbulent nozzle (I = 1 %, l = 3 mm inlet); k, omega and nut carry an explicit condition on every patch, and
CASE-TURB-NOBC, CASE-YPLUS, CASE-RELAM and CASE-MODEL join the refusals.

Usage:
  python case_writer.py --selftest
  python case_writer.py write WEDGE_DIR LEVEL GEOM_DIR REQUIREMENTS_DIR OUT_DIR [ROLES_JSON]
  python case_writer.py write-pipe PIPE_DIR LEVEL RE_TAU OUT_DIR [TURB_JSON]
  python case_writer.py write-nozzle-turb WEDGE_DIR LEVEL GEOM_DIR REQUIREMENTS_DIR OUT_DIR [TURB_JSON [ROLES_JSON]]
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
import turb_integral
import wedge_mesh
import pipe_mesh

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
# slip is typed symmetry: the solver then prescribes the patch's boundary flux to zero
# (momentum.cu momFluxIsPrescribed), so the slip section of docs/16 §H.2 carries no mass.
ROLE_TYPES = {"velocity_inlet": "patch", "pressure_outlet": "patch", "wall": "wall", "slip": "symmetry",
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
         + chr(10) + "       python case_writer.py write-pipe PIPE_DIR LEVEL RE_TAU OUT_DIR [TURB_JSON]"
         + chr(10)
         + "       python case_writer.py write-nozzle-turb WEDGE_DIR LEVEL GEOM_DIR REQUIREMENTS_DIR OUT_DIR"
         + " [TURB_JSON [ROLES_JSON]]"
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
# The two blockgen lines the laminar fvSchemes carries so ofgpu-lowmach's reader (rust/src/io/case.rs
# read_fv_schemes asks for div(phi,k) and div(phi,epsilon) on EVERY model) does not refuse the case;
# both are inert for simulationType laminar - no turbulence transport reads them.
TRANSCRIBED_LAMINAR = (
    ("blockgen", "div(phi,k)       bounded Gauss upwind;", "system/fvSchemes",
     "div(phi,k)      bounded Gauss upwind;"),
    ("blockgen", "div(phi,epsilon) bounded Gauss upwind;", "system/fvSchemes",
     "div(phi,epsilon) bounded Gauss upwind;"),
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
        return {"U": {"type": "symmetry"}, "p": {"type": "symmetry"}, "T": {"type": "symmetry"}}
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
    """The three system files: blockgen.rs write_system()'s controlDict, Gate 105-C's fvSchemes/fvSolution.
    The laminar fvSchemes also carries div(phi,k) and div(phi,epsilon): the two lines are blockgen.rs
    write_system()'s, inert for laminar, written because ofgpu-lowmach's fvSchemes reader asks for
    div(phi,k) and div(phi,epsilon) on every model."""
    sch = list(_FV_SCHEMES)
    j = sch.index("    div(phi,T)      bounded Gauss upwind;") + 1
    sch[j:j] = ["    div(phi,k)      bounded Gauss upwind;", "    div(phi,epsilon) bounded Gauss upwind;"]
    return {"system/controlDict": foam_file("dictionary", "system", "controlDict", chr(10).join(_CONTROL_DICT)),
            "system/fvSchemes": foam_file("dictionary", "system", "fvSchemes", chr(10).join(sch)),
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
            "numerics": {"transcribed": [list(r) for r in TRANSCRIBED]
                         + [list(r) for r in TRANSCRIBED_LAMINAR], "differences": list(DIFFERENCES)},
            "sources": [{"id": s["id"], "file": s["file"], "commit": s["commit"], "blob": s["blob"],
                         "lines": s["lines"], "text_sha256": s["text_sha256"]}
                        for s in common.read_json(SOURCES)["sources"]],
            "cold_start": True, "files": files}
    with open(os.path.join(tmp, "case.json"), "wb") as f:
        f.write((common.canonical_json(case) + chr(10)).encode("utf-8"))
    return case


def scan_bcs(case_dir) -> dict:
    """{"U": [missing], ...} for every field of FIELDS_TURB that exists under 0/ (a laminar case has
    only U, p and T, and scans to those three keys): every boundary patch missing from a field's
    boundaryField, or present with no type line - a patch with no entry would silently hold its cell
    value (field_setup.rs:3046)."""
    names = [p["name"] for p in
             polymesh_write.read_polymesh(os.path.join(case_dir, "constant", "polyMesh"))["patches"]]
    out = {}
    for f in FIELDS_TURB:
        path = os.path.join(case_dir, "0", f)
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        seg = text[text.index("boundaryField"):]
        entries = dict((m.group(1), m.group(2)) for m in re.finditer(
            r"    ([A-Za-z_][A-Za-z0-9_]*)\n    \{\n(.*?)\n    \}", seg, re.S))
        out[f] = sorted(n for n in names
                        if n not in entries or not re.search(r"^\s*type\s+\S+;", entries[n], re.M))
    return out



# --------------------------------------------------------------------------- the turbulent cases
# (docs/16 section H.5 and section I CAD-26): the TG0 pipe driven by a constant/fvSources momentumSource
# and the turbulent nozzle, kOmegaSST, with k, omega and nut on every patch. The BC rows and the
# turbulent numerics lines are quoted read-only from the solver at 08e10bb in
# fixtures/case/sources_turb.json; T10 proves every transcription row against those quotes.
CASE_TURB_VERSION = "cad-case-turb/1"
TURB_VERSION = "cad-turb/1"
FIELDS_TURB = ("U", "p", "T", "k", "omega", "nut")
ROLES_TURB = ("velocity_inlet", "pressure_outlet", "wall", "slip", "wedge", "cyclic")
ROLE_TYPES_TURB = dict(ROLE_TYPES, cyclic="cyclic")
WALLS = ("resolved", "wall_function")
C_MU = 0.09                      # the solver's k-omega C_mu (betaStar); the inlet formulas' constant
YPLUS_LIMIT = 1.0                # docs/16 section H.5 item 3: y+1 <= 1 on the gated (resolved) path
REGISTRY_MODELS = ("laminar", "kEpsilon", "KEpsilon", "LaunderSharmaKE", "realizableKE", "RealizableKE",
                   "RNGkEpsilon", "RNGKEpsilon", "kOmega", "KOmega", "kOmegaSST", "kOmegaSSTLM",
                   "kOmegaSSTGamma", "KOmegaSST", "SpalartAllmaras", "SpalartAllmarras")
WRITER_MODELS = ("kOmegaSST",)
TURB_DEFAULT = {"version": "cad-turb/1", "model": "kOmegaSST", "wall": "resolved", "bl_gate": True,
                "intensity": 0.01, "mixing_length_m": 0.003}
TURB_SPEC_KEYS = ("version", "model", "wall", "bl_gate", "intensity", "mixing_length_m")
PIPE_ROLES = {"version": "cad-roles/1",
              "patches": {"periodic_a": {"role": "cyclic"}, "periodic_b": {"role": "cyclic"},
                          "wall": {"role": "wall"}, "wedge_front": {"role": "wedge"},
                          "wedge_back": {"role": "wedge"}}}
TURB_NOZZLE_ROLES = {"version": "cad-roles/1",
                     "patches": {"inlet": {"role": "velocity_inlet", "direction": [1.0, 0.0, 0.0]},
                                 "outlet": {"role": "pressure_outlet"}, "wall_nozzle": {"role": "wall"},
                                 "wall_upstream": {"role": "wall"}, "wedge_front": {"role": "wedge"},
                                 "wedge_back": {"role": "wedge"}}}
INPUT_KEYS_PIPE = ("pipe_mesh.json", "polyMesh/boundary", "polyMesh/faces", "polyMesh/neighbour",
                   "polyMesh/owner", "polyMesh/points")
INPUT_KEYS_TURB = ("wedge_turb.json", "polyMesh/boundary", "polyMesh/faces", "polyMesh/neighbour",
                   "polyMesh/owner", "polyMesh/points", "geom.json", "requirements.json",
                   "requirements.lock")
CASE_TURB_KEYS = ("version", "case_writer_version", "status", "kind", "level", "inputs", "mesh",
                  "operating_point", "turbulence", "patches", "fields", "numerics", "sources",
                  "cold_start", "files")
MESH_KEYS = ("record", "recipe_sha", "geom_sha256", "template_sha", "params_sha", "msh_sha256", "cells",
             "h1_max_m", "gc7_pass")
OP_TURB_KEYS = ("fluid", "T_K", "p0_Pa", "requirements_lock", "Q_m3_s", "A_inlet_m2", "A_outlet_m2",
                "U_inlet_m_s", "U_exit_m_s", "U_ref_m_s", "nu_m2_s", "R_s", "gamma", "c_m_s", "mach",
                "mach_limit", "re_tau", "u_tau_m_s", "g_x_m_s2", "U_b_m_s")
TURB_KEYS = ("model", "wall", "bl_gate", "intensity", "mixing_length_m", "C_mu", "k_ref_m2_s2",
             "omega_ref_1_s", "u_tau_apriori_m_s", "apriori_method", "yplus1_apriori", "yplus_limit",
             "K_max_apriori", "K_lim", "Re_De")
REFUSAL_IDS_TURB = ("CASE-OUT", "CASE-TURB-SPEC", "CASE-MODEL", "CASE-BIND", "GATE-LOCK", "CASE-UNITS",
                    "CASE-FLUID", "CASE-TURB-NOBC", "CASE-SWIRL", "CASE-MACH", "CASE-RELAM",
                    "CASE-YPLUS")   # the order they run
SOURCES_TURB = os.path.join(common.FIXTURES, "case", "sources_turb.json")
GOLDEN_PIPE = os.path.join(common.FIXTURES, "case", "golden_pipe_turb.json")
GOLDEN_NOZZLE_TURB = os.path.join(common.FIXTURES, "case", "golden_nozzle_turb.json")
# The fresh-process export's geom.json (the selftest setup exports TURB_NOMINAL in a child process):
# OCCT numbers every STEP product with a process-global counter, so a geom.json sha would otherwise
# depend on how many exports ran before it in the same process; a fresh child process always pins
# the first-export bytes below.
TURB_GEOM_SHA = "ddf95fa803ba1636aca0c7355145a656eb1398b7a1a7ff6a66ad6febeda6eab6"

DIFFERENCES_TURB = (
    "wall row (resolved): nut fixedValue 0, k fixedValue 0, omega omegaWallFunction - SPEC-LIT 15.5 calls"
    " a fixedValue 0 on nut the correct low-Re setup, SPEC-LIT 33.2 sets k = 0 at the wall, and"
    " omega_wall()'s root-sum-square of the log branch and Wilcox's 6 nu/(beta_1 y^2) takes its viscous"
    " branch at y+ <= 1 (SPEC-LIT 15.2); the solver's lowRe preset row (nutLowReWallFunction /"
    " kLowReWallFunction / omega zeroGradient) is accepted with LaunderSharmaKE only, and"
    " nutLowReWallFunction next to omegaWallFunction is a mixed row the wall-row check refuses",
    "wall row (wall_function): the solver's standard row, nut nutkWallFunction, k kqRWallFunction,"
    " omega omegaWallFunction (TB7, report only, never refused on y+)",
    "nu: the turbulent cases write wedge_mesh.NU_TURB 1.516e-5 (docs/16 section H.5), the value the meshes"
    " and the a priori u_tau were built at; NU_AIR 1.5e-5 stays for the laminar case",
    "T at the pipe wall: fixedValue T_K - the pipe is a closed domain (cyclic + wall + wedge), ofgpu pins"
    " the pressure itself, and a closed steady domain with no Dirichlet T has a singular T equation; the"
    " nozzle's walls keep the laminar zeroGradient",
    "pipe drive: constant/fvSources momentumSource bodyForce g_x = 2 u_tau^2 / R with u_tau = Re_tau nu / R"
    " (turb_integral.pipe_drive), so the wall shear fixes itself exactly (docs/16 section H.5, TG0)",
    "inlet turbulence: k_ref = 3/2 (I U_ref)^2 and omega_ref = k_ref^(1/2) / (C_mu^(1/4) l), the solver's"
    " own inlet_turb formulas; U_ref is U_inlet for the nozzle and U_b for the pipe; the values are written"
    " at the inlet, at the outlet and as internal fields (a cold start)",
    "nut: calculated 0 at the inlet and the outlet",
)

TRANSCRIBED_TURB = (
    ("constant_ras", "simulationType  RAS;", "constant/momentumTransport", "simulationType  RAS;", "both"),
    ("constant_ras", "model           {model};", "constant/momentumTransport",
     "model           kOmegaSST;", "both"),
    ("constant_ras", "turbulence      on;", "constant/momentumTransport", "turbulence      on;", "both"),
    ("constant_ras", "printCoeffs     on;", "constant/momentumTransport", "printCoeffs     on;", "both"),
    ("blockgen", "div(phi,k)       bounded Gauss upwind;", "system/fvSchemes",
     "div(phi,k)      bounded Gauss upwind;", "both"),
    ("blockgen", "div(phi,omega)   bounded Gauss upwind;", "system/fvSchemes",
     "div(phi,omega)  bounded Gauss upwind;", "both"),
    ("blockgen", "k               0.7;", "system/fvSolution", "k               0.7;", "both"),
    ("blockgen", "omega           0.7;", "system/fvSolution", "omega           0.7;", "both"),
    ("bc_names", "turbulentIntensityKineticEnergyInlet", "0/k",
     "type            turbulentIntensityKineticEnergyInlet;", "nozzle"),
    ("inlet_turb", "turbulentIntensity", "0/k", "turbulentIntensity", "nozzle"),
    ("bc_names", "turbulentMixingLengthFrequencyInlet", "0/omega",
     "type            turbulentMixingLengthFrequencyInlet;", "nozzle"),
    ("inlet_turb", "mixingLength", "0/omega", "mixingLength", "nozzle"),
    ("bc_names", "omegaWallFunction", "0/omega", "type            omegaWallFunction;", "both"),
    ("bc_names", "calculated", "0/nut", "type            calculated;", "nozzle"),
    ("bc_names", "cyclic", "0/k", "type            cyclic;", "pipe"),
    ("momentum_source", '"momentumSource" => {', "constant/fvSources",
     "type            momentumSource;", "pipe"),
    ("momentum_source", "bodyForce", "constant/fvSources", "bodyForce", "pipe"),
    ("selector", "all", "constant/fvSources", "selection       all;", "pipe"),
)

TRANSCRIBED_TURB_NOZZLE = tuple(r for r in TRANSCRIBED_TURB
                                if not (r[0] in ("bc_names", "inlet_turb")
                                        and r[1] in ("turbulentIntensityKineticEnergyInlet",
                                                     "turbulentIntensity",
                                                     "turbulentMixingLengthFrequencyInlet",
                                                     "mixingLength")))
DIFFERENCES_NOZZLE_TURB = (
    "inlet k and omega (nozzle): fixedValue k_ref and omega_ref - the values turbulentIntensityKineticEnergyInlet"
    " and turbulentMixingLengthFrequencyInlet give a uniform inflow by the solver's inlet_turb formulas; the"
    " pinned 90510fc ofgpu-lowmach builds the turbulence fields with no U and no k to evaluate those two"
    " conditions from and stops at set-up, so the case names the fixed values; I and l stay in turbulence",
    "initial U (nozzle): uniform (U_inlet 0 0), still a cold start (no earlier solve); from rest the pinned"
    " binary's first step reaches M 0.526 at the outlet axis at U_e 60 and the SPEC-LIT 93.6 Mach guard stops"
    " the run; the pipe keeps its rest state",
)

def turb_refs(u_ref, intensity, mixing_length_m):
    """(k_ref, omega_ref) by the solver's own inlet formulas (field_setup.rs inlet_turb):
    k = 3/2 (I |U|)^2 and omega = k^(1/2) / (C_mu^(1/4) l), with C_mu 0.09 (betaStar)."""
    k = 1.5 * (float(intensity) * float(u_ref)) ** 2
    return k, math.sqrt(k) / (C_MU ** 0.25 * float(mixing_length_m))


def check_turb_spec(turb):
    """The validated spec (rules 2 then 3): CASE-TURB-SPEC for a malformed document, CASE-MODEL for a
    model the solver's registry does not name or this writer does not write."""
    if not isinstance(turb, dict) or sorted(turb) != sorted(TURB_SPEC_KEYS):
        raise Refused("CASE-TURB-SPEC", "turb must be a dict with exactly the keys %s"
                      % ", ".join(TURB_SPEC_KEYS))
    if turb["version"] != TURB_VERSION:
        raise Refused("CASE-TURB-SPEC", "turb version %r is not %r" % (turb["version"], TURB_VERSION))
    if not isinstance(turb["model"], str):
        raise Refused("CASE-TURB-SPEC", "turb model %r is not a string" % (turb["model"],))
    if turb["wall"] not in WALLS:
        raise Refused("CASE-TURB-SPEC", "turb wall %r is not one of %s" % (turb["wall"], ", ".join(WALLS)))
    if not isinstance(turb["bl_gate"], bool):
        raise Refused("CASE-TURB-SPEC", "turb bl_gate %r is not a bool" % (turb["bl_gate"],))
    i = turb["intensity"]
    if isinstance(i, bool) or not isinstance(i, (int, float)) or not math.isfinite(i) or not 0.0 < i <= 0.2:
        raise Refused("CASE-TURB-SPEC", "turb intensity %r is not a finite real in (0, 0.2]" % (i,))
    l = turb["mixing_length_m"]
    if isinstance(l, bool) or not isinstance(l, (int, float)) or not math.isfinite(l) or not l > 0.0:
        raise Refused("CASE-TURB-SPEC", "turb mixing_length_m %r is not a finite real > 0" % (l,))
    if turb["model"] not in REGISTRY_MODELS:
        raise Refused("CASE-MODEL", "model %r is not in the solver's registry" % (turb["model"],))
    if turb["model"] not in WRITER_MODELS:
        raise Refused("CASE-MODEL", "model %r is in the solver's registry, but this writer writes only"
                      " kOmegaSST" % (turb["model"],))
    return dict(turb)


def _check_re_tau(re_tau):
    """The pipe's rule-2 tail: re_tau must be a finite real > 0."""
    if isinstance(re_tau, bool) or not isinstance(re_tau, (int, float)) or not math.isfinite(re_tau) \
            or not re_tau > 0.0:
        raise Refused("CASE-TURB-SPEC", "re_tau %r is not a finite real > 0" % (re_tau,))
    return float(re_tau)


def bc_table_turb(role, u_in, t_k, k_ref, omega_ref, wall, isothermal_wall, intensity,
                  mixing_length_m):
    """{"U":..,"p":..,"T":..,"k":..,"omega":..,"nut":..} for one role. U, p and T are the laminar
    bc_table's, the wall T going fixedValue when isothermal_wall; k, omega and nut are the quoted rows -
    resolved: k/nut fixedValue 0 with omegaWallFunction; wall_function: kqRWallFunction / omegaWallFunction
    / nutkWallFunction. The inlet entry's turbulentIntensity and mixingLength are the spec's I and l
    exactly, written as given (no arithmetic on them)."""
    if role == "cyclic":
        base = dict((f, {"type": "cyclic"}) for f in ("U", "p", "T"))
    else:
        base = bc_table(role, u_in, t_k)
    if role == "wall":
        if isothermal_wall:
            base["T"] = {"type": "fixedValue", "value": t_k}
        omega_row = {"type": "omegaWallFunction", "value": omega_ref}
        if wall == "wall_function":
            k_row = {"type": "kqRWallFunction", "value": k_ref}
            nut_row = {"type": "nutkWallFunction", "value": 0.0}
        else:
            k_row = {"type": "fixedValue", "value": 0.0}
            nut_row = {"type": "fixedValue", "value": 0.0}
    elif role == "velocity_inlet":
        k_row = {"type": "fixedValue", "value": k_ref}
        omega_row = {"type": "fixedValue", "value": omega_ref}
        nut_row = {"type": "calculated", "value": 0.0}
    elif role == "pressure_outlet":
        k_row = {"type": "inletOutlet", "inletValue": k_ref, "value": k_ref}
        omega_row = {"type": "inletOutlet", "inletValue": omega_ref, "value": omega_ref}
        nut_row = {"type": "calculated", "value": 0.0}
    else:
        k_row = omega_row = nut_row = {"type": base["U"]["type"]}
    base["k"], base["omega"], base["nut"] = k_row, omega_row, nut_row
    return dict((f, base[f]) for f in FIELDS_TURB)


def turb_constant_files(model, nu, g_x=None):
    """physicalProperties at the turbulent nu, the RAS momentumTransport (constant_ras), and fvSources
    with the momentumSource body force when g_x is given (the pipe; fv_sources, momentum_source)."""
    phys = chr(10).join(["viscosityModel  constant;", "",
                         "nu              [0 2 -1 0 0 0 0] " + fmt(nu) + ";"])
    ras = chr(10).join(["simulationType  RAS;", "", "RAS", "{", "    model           %s;" % model,
                        "    turbulence      on;", "    printCoeffs     on;", "}"])
    out = {"constant/physicalProperties": foam_file("dictionary", "constant", "physicalProperties", phys),
           "constant/momentumTransport": foam_file("dictionary", "constant", "momentumTransport", ras)}
    if g_x is not None:
        src = chr(10).join(["drive", "{", "    type            momentumSource;", "    field           U;",
                            "    selection       all;",
                            "    bodyForce       " + _uniform([g_x, 0.0, 0.0]) + ";", "}"])
        out["constant/fvSources"] = foam_file("dictionary", "constant", "fvSources", src)
    return out


def turb_system_files():
    """controlDict as the laminar one; fvSchemes with div(phi,k) and div(phi,omega) right after div(phi,T);
    fvSolution with k and omega solver blocks after the T block and k/omega 0.7 after the T relaxation -
    blockgen.rs write_system()'s turbulent entries."""
    sch = list(_FV_SCHEMES)
    j = sch.index("    div(phi,T)      bounded Gauss upwind;") + 1
    sch[j:j] = ["    div(phi,k)      bounded Gauss upwind;", "    div(phi,omega)  bounded Gauss upwind;"]
    sol = list(_FV_SOLUTION)
    j = sol.index("    }", sol.index("    T"))
    sol[j + 1:j + 1] = ["", "    k", "    {", "        solver          PBiCGStab;",
                        "        preconditioner  diagonal;", "        tolerance       1e-08;",
                        "        relTol          0.01;", "        maxIter         200;", "    }", "",
                        "    omega", "    {", "        solver          PBiCGStab;",
                        "        preconditioner  diagonal;", "        tolerance       1e-08;",
                        "        relTol          0.01;", "        maxIter         200;", "    }"]
    e = sol.index("        T               0.7;") + 1
    sol[e:e] = ["        k               0.7;", "        omega           0.7;"]
    return {"system/controlDict": foam_file("dictionary", "system", "controlDict",
                                            chr(10).join(_CONTROL_DICT)),
            "system/fvSchemes": foam_file("dictionary", "system", "fvSchemes", chr(10).join(sch)),
            "system/fvSolution": foam_file("dictionary", "system", "fvSolution", chr(10).join(sol))}


_DIMS_TURB = dict(_DIMS, k="[0 2 -2 0 0 0 0]", omega="[0 0 -1 0 0 0 0]", nut="[0 2 -1 0 0 0 0]")
_KEY_ORDER = ("type", "turbulentIntensity", "mixingLength", "inletValue", "value")


def _bc_lines(bc):
    """One patch block's lines in entry order: the keyword padded to 16 columns, ONE space from 16 on
    (so turbulentIntensity 0.01;), values behind uniform through _uniform."""
    lines = []
    for key in _KEY_ORDER:
        if key not in bc:
            continue
        if key == "type":
            val = bc[key]
        elif key in ("turbulentIntensity", "mixingLength"):
            val = fmt(bc[key])
        else:
            val = "uniform " + _uniform(bc[key])
        lines.append("        " + (key.ljust(16) if len(key) < 16 else key + " ") + val + ";")
    return lines


def turb_field_files(patches, internal):
    """0/U, 0/p, 0/T, 0/k, 0/omega, 0/nut: dimensions, a uniform cold internal field, one explicit block
    per boundary patch in boundary-file order - every patch, zero-face ones included."""
    out = {}
    for f in FIELDS_TURB:
        lines = ["dimensions      " + _DIMS_TURB[f] + ";", "",
                 "internalField   uniform " + internal[f] + ";", "", "boundaryField", "{"]
        for row in patches:
            lines.append("    " + row["name"])
            lines.append("    {")
            lines.extend(_bc_lines(row[f]))
            lines.append("    }")
        lines.append("}")
        cls = "volVectorField" if f == "U" else "volScalarField"
        out["0/" + f] = foam_file(cls, "0", f, chr(10).join(lines))
    return out


def _turb_fields_doc(t_k, k_ref, omega_ref):
    """The fields block of case.json: the dimensions and the cold internal value of each of the six."""
    return {"U": {"dimensions": _DIMS_TURB["U"], "internal": [0.0, 0.0, 0.0]},
            "p": {"dimensions": _DIMS_TURB["p"], "internal": 0.0},
            "T": {"dimensions": _DIMS_TURB["T"], "internal": t_k},
            "k": {"dimensions": _DIMS_TURB["k"], "internal": k_ref},
            "omega": {"dimensions": _DIMS_TURB["omega"], "internal": omega_ref},
            "nut": {"dimensions": _DIMS_TURB["nut"], "internal": 0.0}}


def _roles_ok(patches, roles, kind, fluid_tags):
    """CASE-TURB-NOBC, rule 8: cad-roles/1 with a patches dict; a role for every boundary patch and no
    extra; the role types matching the boundary; at least one wall; cyclic pairs naming each other back;
    the nozzle also exactly one velocity_inlet and one pressure_outlet, both fluid faces of geom.json;
    the pipe neither of the two. Returns (inlet, outlet) for the nozzle, None for the pipe."""
    if not isinstance(roles, dict) or roles.get("version") != "cad-roles/1" \
            or not isinstance(roles.get("patches"), dict):
        raise Refused("CASE-TURB-NOBC", "the roles document must be cad-roles/1 with a patches dict")
    rmap = roles["patches"]
    names = [p["name"] for p in patches]
    for n in names:
        if n not in rmap:
            raise Refused("CASE-TURB-NOBC", "boundary patch %s has no role" % n)
    for n in rmap:
        if n not in names:
            raise Refused("CASE-TURB-NOBC", "role patch %s is not in the boundary" % n)
    for p in patches:
        role = rmap[p["name"]].get("role")
        if role not in ROLES_TURB:
            raise Refused("CASE-TURB-NOBC", "patch %s has role %r, want one of %s"
                          % (p["name"], role, ", ".join(ROLES_TURB)))
        if p["type"] != ROLE_TYPES_TURB[role]:
            raise Refused("CASE-TURB-NOBC", "patch %s (role %s) has boundary type %s, want %s"
                          % (p["name"], role, p["type"], ROLE_TYPES_TURB[role]))
    if not [n for n in names if rmap[n].get("role") == "wall"]:
        raise Refused("CASE-TURB-NOBC", "no wall-role patch: k, omega and nut need a wall row")
    pby = dict((p["name"], p) for p in patches)
    for n in names:
        if rmap[n].get("role") != "cyclic":
            continue
        other = pby[n].get("neighbourPatch")
        if other is None:
            raise Refused("CASE-TURB-NOBC", "cyclic patch %s has no neighbourPatch" % n)
        if other not in pby or rmap[other].get("role") != "cyclic":
            raise Refused("CASE-TURB-NOBC", "cyclic patch %s's neighbourPatch %s is not a cyclic-role"
                          " patch of the boundary" % (n, other))
        if pby[other].get("neighbourPatch") != n:
            raise Refused("CASE-TURB-NOBC", "cyclic patch %s's neighbourPatch %s does not name it back"
                          % (n, other))
    inlets = [n for n in names if rmap[n].get("role") == "velocity_inlet"]
    outlets = [n for n in names if rmap[n].get("role") == "pressure_outlet"]
    if kind == "pipe":
        if inlets or outlets:
            raise Refused("CASE-TURB-NOBC", "the pipe is driven by its fvSources body force, not by an"
                          " inlet or an outlet: %s" % ", ".join(inlets + outlets))
        return None
    if len(inlets) != 1 or len(outlets) != 1:
        raise Refused("CASE-TURB-NOBC", "want exactly one velocity_inlet and one pressure_outlet, got"
                      " %d and %d" % (len(inlets), len(outlets)))
    for n in (inlets[0], outlets[0]):
        if n not in fluid_tags:
            raise Refused("CASE-TURB-NOBC", "patch %s is not a fluid face of geom.json's tags" % n)
    return inlets[0], outlets[0]


def _sources_all():
    """The 2 quotes of sources.json then the 16 of sources_turb.json, path-free rows for case.json."""
    rows = []
    for path in (SOURCES, SOURCES_TURB):
        for s in common.read_json(path)["sources"]:
            rows.append({"id": s["id"], "file": s["file"], "commit": s["commit"], "blob": s["blob"],
                         "lines": s["lines"], "text_sha256": s["text_sha256"]})
    return rows


def _emit_turb(tmp, case, written):
    """The written files, then case.json last; case's files map ends up holding every sha but
    case.json's own."""
    for rel, text in written:
        blob = text.encode("utf-8")
        target = os.path.join(tmp, rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "wb") as f:
            f.write(blob)
        case["files"][rel] = common.sha256_bytes(blob)
    with open(os.path.join(tmp, "case.json"), "wb") as f:
        f.write((common.canonical_json(case) + chr(10)).encode("utf-8"))


def _pipe_input_paths(pipe_dir, level):
    """The six pipe inputs by INPUT_KEYS_PIPE name; no path ever enters case.json."""
    pm = os.path.join(pipe_dir, "L%d" % level, "case", "constant", "polyMesh")
    return dict([("pipe_mesh.json", os.path.join(pipe_dir, "pipe_mesh.json"))]
                + [("polyMesh/" + nm, os.path.join(pm, nm)) for nm in POLYMESH_FILES])


def write_pipe_case(pipe_dir, level, re_tau, out_dir, turb=None, roles=None, between_hook=None):
    """The TG0 pipe case (docs/16 section H.5): the same contract as write_case - never raises Refused,
    builds in a .case- temp directory that becomes out_dir only on success, re-checks every input after
    between_hook (CASE-BIND), and leaves nothing behind on a refusal."""
    level = int(level)
    out_dir = os.path.abspath(out_dir)
    if os.path.lexists(out_dir):
        return {"status": "refused", "rule": "CASE-OUT", "message": "%s exists" % out_dir, "case": None}
    parent = os.path.dirname(out_dir)
    os.makedirs(parent, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=".case-", dir=parent)
    try:
        doc = _write_pipe_body(pipe_dir, level, re_tau, tmp, turb, roles)
        if between_hook is not None:
            between_hook()
        for key, path in _pipe_input_paths(pipe_dir, level).items():
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


def _write_pipe_body(pipe_dir, level, re_tau, tmp, turb, roles):
    """write_pipe_case's body: CASE-TURB-SPEC, CASE-MODEL, CASE-BIND, CASE-UNITS, CASE-TURB-NOBC,
    CASE-MACH, CASE-YPLUS, then the eighteen files with case.json last."""
    spec = check_turb_spec(TURB_DEFAULT if turb is None else turb)
    re_tau = _check_re_tau(re_tau)
    paths = _pipe_input_paths(pipe_dir, level)
    inputs = {}
    pbytes, inputs["pipe_mesh.json"] = _bound(paths["pipe_mesh.json"], None, "pipe_mesh.json")
    record = json.loads(pbytes.decode("utf-8"))
    rows = [lv for lv in record["levels"] if lv["level"] == level]
    if not rows:
        raise Refused("CASE-BIND", "level %d is not in the pipe-mesh record" % level)
    row = rows[0]
    pm_bytes = {}
    for name in POLYMESH_FILES:
        want = row.get("polymesh_sha256", {}).get(name)
        if want is None:
            raise Refused("CASE-BIND", "polyMesh/%s is not recorded in the pipe-mesh level row" % name)
        pm_bytes[name], inputs["polyMesh/" + name] = _bound(paths["polyMesh/" + name], want,
                                                            "polyMesh/" + name)
    r_m, l_m = float(record["recipe"]["R_m"]), float(record["recipe"]["L_m"])
    os.makedirs(os.path.join(tmp, "constant", "polyMesh"))
    for name in POLYMESH_FILES:
        with open(os.path.join(tmp, "constant", "polyMesh", name), "wb") as f:
            f.write(pm_bytes[name])
    pm = polymesh_write.read_polymesh(os.path.join(tmp, "constant", "polyMesh"))
    span = float(np.max(pm["points"][:, 0]) - np.min(pm["points"][:, 0]))
    if abs(span - l_m) > SPAN_REL * l_m:
        raise Refused("CASE-UNITS", "mesh x span %r m against the pipe recipe's L_m %r m" % (span, l_m))
    if roles is None:
        roles = PIPE_ROLES
    _roles_ok(pm["patches"], roles, "pipe", None)
    t_k = STATE["T_K"]
    drive = turb_integral.pipe_drive(re_tau, r_m, wedge_mesh.NU_TURB)
    u_tau, g_x = drive["u_tau"], drive["g_x"]
    u_b = u_tau * math.sqrt(8.0 / turb_integral.f_prandtl(re_tau))
    k_ref, omega_ref = turb_refs(u_b, spec["intensity"], spec["mixing_length_m"])
    r_s = GAS["R_universal"] / GAS["W"]
    c = math.sqrt(GAS["gamma"] * r_s * t_k)
    mach = u_b / c
    if mach > MACH_LIMIT:
        raise Refused("CASE-MACH", "predicted U_b/c is %r, over %r" % (mach, MACH_LIMIT))
    yplus1 = row["h1_max_m"] * u_tau / wedge_mesh.NU_TURB
    if spec["wall"] == "resolved" and yplus1 > YPLUS_LIMIT:
        raise Refused("CASE-YPLUS", "the resolved wall's a priori y+1 is %r, over %r (wall_function is"
                      " the report-only TB7 leg)" % (yplus1, YPLUS_LIMIT))
    patch_rows = []
    for p in pm["patches"]:
        role = roles["patches"][p["name"]]["role"]
        bcs = bc_table_turb(role, None, t_k, k_ref, omega_ref, spec["wall"], True,
                            spec["intensity"], spec["mixing_length_m"])
        patch_rows.append(dict({"name": p["name"], "type": p["type"], "n_faces": p["nFaces"],
                                "start_face": p["startFace"], "role": role},
                               **dict((f, bcs[f]) for f in FIELDS_TURB)))
    files = dict(("constant/polyMesh/" + nm, common.sha256_bytes(pm_bytes[nm])) for nm in POLYMESH_FILES)
    internal = {"U": "(0.0 0.0 0.0)", "p": "0.0", "T": fmt(t_k), "k": fmt(k_ref),
                "omega": fmt(omega_ref), "nut": "0.0"}
    written = sorted(turb_constant_files(spec["model"], wedge_mesh.NU_TURB, g_x=g_x).items()) \
        + sorted(turb_system_files().items()) + sorted(turb_field_files(patch_rows, internal).items())
    case = {"version": CASE_TURB_VERSION, "case_writer_version": CASE_WRITER_VERSION, "status": "ok",
            "kind": "pipe", "level": level, "inputs": dict((k, inputs[k]) for k in INPUT_KEYS_PIPE),
            "files": files,
            "mesh": {"record": "pipe_mesh.json", "recipe_sha": record["recipe_sha"], "geom_sha256": None,
                     "template_sha": None, "params_sha": None, "msh_sha256": row["msh_sha256"],
                     "cells": row["cells"], "h1_max_m": row["h1_max_m"],
                     "gc7_pass": row["gc7"]["pass"]},
            "operating_point": {"fluid": STATE["fluid"], "T_K": t_k, "p0_Pa": STATE["p0_Pa"],
                                "requirements_lock": None, "Q_m3_s": None, "A_inlet_m2": None,
                                "A_outlet_m2": None, "U_inlet_m_s": None, "U_exit_m_s": None,
                                "U_ref_m_s": u_b, "nu_m2_s": wedge_mesh.NU_TURB, "R_s": r_s,
                                "gamma": GAS["gamma"], "c_m_s": c, "mach": mach,
                                "mach_limit": MACH_LIMIT, "re_tau": re_tau, "u_tau_m_s": u_tau,
                                "g_x_m_s2": g_x, "U_b_m_s": u_b},
            "turbulence": {"model": spec["model"], "wall": spec["wall"], "bl_gate": spec["bl_gate"],
                           "intensity": spec["intensity"], "mixing_length_m": spec["mixing_length_m"],
                           "C_mu": C_MU, "k_ref_m2_s2": k_ref, "omega_ref_1_s": omega_ref,
                           "u_tau_apriori_m_s": u_tau, "apriori_method": "pipe-exact/1",
                           "yplus1_apriori": yplus1, "yplus_limit": YPLUS_LIMIT, "K_max_apriori": None,
                           "K_lim": None, "Re_De": None},
            "patches": patch_rows, "fields": _turb_fields_doc(t_k, k_ref, omega_ref),
            "numerics": {"transcribed": [list(r) for r in TRANSCRIBED]
                         + [list(r) for r in TRANSCRIBED_TURB],
                         "differences": list(DIFFERENCES) + list(DIFFERENCES_TURB)},
            "sources": _sources_all(), "cold_start": True}
    _emit_turb(tmp, case, written)
    return case


def _turb_input_paths(wedge_dir, level, geom_dir, req_dir):
    """The nine turbulent inputs by INPUT_KEYS_TURB name; no path ever enters case.json."""
    paths = _input_paths(wedge_dir, level, geom_dir, req_dir)
    paths["wedge_turb.json"] = os.path.join(wedge_dir, "wedge_turb.json")
    del paths["wedge_mesh.json"]
    return paths


def write_turb_case(wedge_dir, level, geom_dir, req_dir, out_dir, turb=None, roles=None,
                    between_hook=None):
    """The turbulent nozzle case (docs/16 section H.5): the same contract as write_case - never raises
    Refused, builds in a .case- temp directory that becomes out_dir only on success, re-checks every
    input after between_hook (CASE-BIND), and leaves nothing behind on a refusal."""
    level = int(level)
    out_dir = os.path.abspath(out_dir)
    if os.path.lexists(out_dir):
        return {"status": "refused", "rule": "CASE-OUT", "message": "%s exists" % out_dir, "case": None}
    parent = os.path.dirname(out_dir)
    os.makedirs(parent, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=".case-", dir=parent)
    try:
        doc = _write_turb_body(wedge_dir, level, geom_dir, req_dir, tmp, turb, roles)
        if between_hook is not None:
            between_hook()
        for key, path in _turb_input_paths(wedge_dir, level, geom_dir, req_dir).items():
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


def _write_turb_body(wedge_dir, level, geom_dir, req_dir, tmp, turb, roles):
    """write_turb_case's body: CASE-TURB-SPEC, CASE-MODEL, CASE-BIND, GATE-LOCK, CASE-UNITS, CASE-FLUID,
    CASE-TURB-NOBC, CASE-SWIRL, CASE-MACH, CASE-RELAM, CASE-YPLUS, then the seventeen files."""
    spec = check_turb_spec(TURB_DEFAULT if turb is None else turb)
    paths = _turb_input_paths(wedge_dir, level, geom_dir, req_dir)
    inputs = {}
    wbytes, inputs["wedge_turb.json"] = _bound(paths["wedge_turb.json"], None, "wedge_turb.json")
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
            raise Refused("CASE-FLUID", "operating_point %s is %r, want %r; NU_TURB holds only there"
                          % (key, op.get(key), STATE[key]))
    if roles is None:
        roles = TURB_NOZZLE_ROLES
    inlet_name, outlet_name = _roles_ok(pm["patches"], roles, "nozzle",
                                        geom.get("tags", {}).get("fluid_faces", {}))
    tags = geom["tags"]["fluid_faces"]
    a_inlet, a_outlet = tags[inlet_name]["area_m2"], tags[outlet_name]["area_m2"]
    d = roles["patches"][inlet_name].get("direction")
    if not isinstance(d, (list, tuple)) or len(d) != 3 \
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                       for v in d):
        raise Refused("CASE-SWIRL", "the velocity_inlet's direction must be a list of 3 finite numbers")
    norm = math.sqrt(float(d[0]) ** 2 + float(d[1]) ** 2 + float(d[2]) ** 2)
    if not norm > 0.0:
        raise Refused("CASE-SWIRL", "the inlet direction's norm must be positive")
    n = [float(v) / norm for v in d]
    if abs(n[1]) > SWIRL_TOL or abs(n[2]) > SWIRL_TOL or n[0] <= 0:
        raise Refused("CASE-SWIRL", "the wedge's symmetry alias cannot carry swirl or a non-axial inflow")
    t_k = STATE["T_K"]
    q = reqs.flow_Q(doc)
    u_inlet, u_exit = q / a_inlet, q / a_outlet
    r_s = GAS["R_universal"] / GAS["W"]
    c = math.sqrt(GAS["gamma"] * r_s * t_k)
    mach = u_exit / c
    if mach > MACH_LIMIT:
        raise Refused("CASE-MACH", "predicted U_e/c is %r, over %r" % (mach, MACH_LIMIT))
    params = geom["params"]
    re_de = u_exit * (params["D_i"] / math.sqrt(params["CR"])) / wedge_mesh.NU_TURB
    kmax = turb_integral.k_max_1d(params["law"], params["CR"], params["L_over_Di"], re_de,
                                  params.get("x_m"))["K_max"]
    if spec["bl_gate"] and kmax > turb_integral.K_RELAM:
        raise Refused("CASE-RELAM", "a turbulent BL gate is asked where the a priori K_max is %r, over"
                      " %r (relaminarisation, Kline et al. 1967)" % (kmax, turb_integral.K_RELAM))
    ap = wedge_mesh.a_priori_utau(params, u_exit, wedge_mesh.NU_TURB,
                                  n=wedge_mesh.RECIPE_TURB["n_apriori"])
    u_tau = ap["u_tau_max_m_s"]
    k_ref, omega_ref = turb_refs(u_inlet, spec["intensity"], spec["mixing_length_m"])
    yplus1 = row["h1_max_m"] * u_tau / wedge_mesh.NU_TURB
    if spec["wall"] == "resolved" and yplus1 > YPLUS_LIMIT:
        raise Refused("CASE-YPLUS", "the resolved wall's a priori y+1 is %r, over %r (wall_function is"
                      " the report-only TB7 leg)" % (yplus1, YPLUS_LIMIT))
    u_in = [u_inlet * n[0], u_inlet * n[1], u_inlet * n[2]]
    patch_rows = []
    for p in pm["patches"]:
        role = roles["patches"][p["name"]]["role"]
        bcs = bc_table_turb(role, u_in, t_k, k_ref, omega_ref, spec["wall"], False,
                            spec["intensity"], spec["mixing_length_m"])
        patch_rows.append(dict({"name": p["name"], "type": p["type"], "n_faces": p["nFaces"],
                                "start_face": p["startFace"], "role": role},
                               **dict((f, bcs[f]) for f in FIELDS_TURB)))
    files = dict(("constant/polyMesh/" + nm, common.sha256_bytes(pm_bytes[nm])) for nm in POLYMESH_FILES)
    internal = {"U": _uniform(u_in), "p": "0.0", "T": fmt(t_k), "k": fmt(k_ref),
                "omega": fmt(omega_ref), "nut": "0.0"}
    written = sorted(turb_constant_files(spec["model"], wedge_mesh.NU_TURB).items()) \
        + sorted(turb_system_files().items()) + sorted(turb_field_files(patch_rows, internal).items())
    case = {"version": CASE_TURB_VERSION, "case_writer_version": CASE_WRITER_VERSION, "status": "ok",
            "kind": "nozzle", "level": level, "inputs": dict((k, inputs[k]) for k in INPUT_KEYS_TURB),
            "files": files,
            "mesh": {"record": "wedge_turb.json", "recipe_sha": record["recipe_sha"],
                     "geom_sha256": record["geom_sha256"], "template_sha": record["template_sha"],
                     "params_sha": record["params_sha"], "msh_sha256": row["msh_sha256"],
                     "cells": row["cells"], "h1_max_m": row["h1_max_m"],
                     "gc7_pass": row["gc7"]["pass"]},
            "operating_point": {"fluid": op["fluid"], "T_K": op["T_K"], "p0_Pa": op["p0_Pa"],
                                "requirements_lock": doc["lock_sha"], "Q_m3_s": q,
                                "A_inlet_m2": a_inlet, "A_outlet_m2": a_outlet,
                                "U_inlet_m_s": u_inlet, "U_exit_m_s": u_exit, "U_ref_m_s": u_inlet,
                                "nu_m2_s": wedge_mesh.NU_TURB, "R_s": r_s, "gamma": GAS["gamma"],
                                "c_m_s": c, "mach": mach, "mach_limit": MACH_LIMIT, "re_tau": None,
                                "u_tau_m_s": u_tau, "g_x_m_s2": None, "U_b_m_s": None},
            "turbulence": {"model": spec["model"], "wall": spec["wall"], "bl_gate": spec["bl_gate"],
                           "intensity": spec["intensity"], "mixing_length_m": spec["mixing_length_m"],
                           "C_mu": C_MU, "k_ref_m2_s2": k_ref, "omega_ref_1_s": omega_ref,
                           "u_tau_apriori_m_s": u_tau, "apriori_method": turb_integral.METHOD,
                           "yplus1_apriori": yplus1, "yplus_limit": YPLUS_LIMIT,
                           "K_max_apriori": kmax, "K_lim": turb_integral.K_RELAM, "Re_De": re_de},
            "patches": patch_rows,
            "fields": dict(_turb_fields_doc(t_k, k_ref, omega_ref),
                           U={"dimensions": _DIMS_TURB["U"],
                              "internal": [u_in[0], u_in[1], u_in[2]]}),
            "numerics": {"transcribed": [list(r) for r in TRANSCRIBED]
                         + [list(r) for r in TRANSCRIBED_TURB_NOZZLE],
                         "differences": list(DIFFERENCES) + list(DIFFERENCES_TURB)
                         + list(DIFFERENCES_NOZZLE_TURB)},
            "sources": _sources_all(), "cold_start": True}
    _emit_turb(tmp, case, written)
    return case


def main(argv) -> int:
    """--selftest; write (6-7 args); write-pipe (5-6 args); write-nozzle-turb (6-8 args); scan (2 args).
    ValueError/OSError print and exit 2, anything else usage."""
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
        if len(argv) in (5, 6) and argv[0] == "write-pipe":
            turb = common.read_json(argv[5]) if len(argv) == 6 else None
            res = write_pipe_case(argv[1], int(argv[2]), float(argv[3]), argv[4], turb=turb)
            if res["status"] == "ok":
                print(common.canonical_json({"status": "ok", "files": len(res["case"]["files"]) + 1}))
                return 0
            print(common.canonical_json({"refused": res["rule"], "detail": res["message"]}))
            return 1
        if len(argv) in (6, 7, 8) and argv[0] == "write-nozzle-turb":
            turb = common.read_json(argv[6]) if len(argv) >= 7 else None
            roles_json = common.read_json(argv[7]) if len(argv) == 8 else None
            res = write_turb_case(argv[1], int(argv[2]), argv[3], argv[4], argv[5], turb=turb,
                                  roles=roles_json)
            if res["status"] == "ok":
                print(common.canonical_json({"status": "ok", "files": len(res["case"]["files"]) + 1}))
                return 0
            print(common.canonical_json({"refused": res["rule"], "detail": res["message"]}))
            return 1
        if len(argv) == 2 and argv[0] == "scan":
            scan = scan_bcs(argv[1])
            print(common.canonical_json(scan))
            return 0 if not any(scan.values()) else 1
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

        # T1: the two quotes hash to their recorded sha, and all 26 transcribed settings occur in the quote
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
        for sid, q_sub, c_file, w_sub in TRANSCRIBED + TRANSCRIBED_LAMINAR:
            assert q_sub in quote_text[sid], (sid, q_sub)
            assert w_sub in case_texts[c_file], (c_file, w_sub)
        lam_sch = system_files()["system/fvSchemes"]
        assert "    div(phi,k)      bounded Gauss upwind;" in lam_sch, "laminar fvSchemes lacks div(phi,k)"
        assert "    div(phi,epsilon) bounded Gauss upwind;" in lam_sch, "laminar fvSchemes lacks div(phi,epsilon)"
        turb_sch = turb_system_files()["system/fvSchemes"]
        assert turb_sch.count("div(phi,k)") == 1, turb_sch.count("div(phi,k)")
        assert "epsilon" not in turb_sch, "turbulent fvSchemes carries epsilon"
        print("[ok] 2 quotes at bceb799 hash to their recorded sha; 26 transcribed settings occur in their "
              "quote and in the written case; the laminar fvSchemes carries the inert div(phi,k)/"
              "div(phi,epsilon) lines, the turbulent one div(phi,k) once and no epsilon")

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

        # T3b: the slip section is written type symmetry (the solver prescribes its flux to zero, docs/16
        # §H.2), and a boundary still carrying patch there refuses CASE-NOBC
        for rel in ("0/U", "0/p", "0/T"):
            with open(os.path.join(c0, rel.replace("/", os.sep)), "r", encoding="utf-8") as f:
                blk = re.search(r"slip_upstream\s*\{[^}]*\}", f.read()).group(0)
            assert "type            symmetry;" in blk, rel
        wp = wedge_copy("w_slip_patch")
        bpath = os.path.join(wp, "L0", "case", "constant", "polyMesh", "boundary")
        with open(bpath, "r", encoding="utf-8") as f:
            btxt = f.read()
        assert btxt.count("symmetry;") == 1, "slip_upstream is not the only symmetry in the boundary"
        common.atomic_write(bpath, btxt.replace("symmetry;", "patch;"))
        rebind(wp)
        res = write_case(wp, 0, gdir, sdir, os.path.join(td, "case_slip_patch"))
        refused(res, "CASE-NOBC", os.path.join(td, "case_slip_patch"))
        assert res["message"] == "patch slip_upstream (role slip) has boundary type patch, want symmetry", \
            res["message"]
        print("[ok] slip_upstream carries type symmetry in 0/U, 0/p and 0/T; the patch-typed boundary"
              " refuses CASE-NOBC")

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

        # ---- the turbulent cases (docs/16 section H.5, section I CAD-26): T10..T17; setup: the L0 pipe,
        # the TURB_NOMINAL geometry and its L0 turbulent wedge at U_e 60
        pdir = os.path.join(td, "pipe")
        res = pipe_mesh.run(pdir, levels=(0,))
        assert res["status"] == "ok", (res["status"], res["rule"], res["message"])
        # The turbulent geometry is exported by export.py's own CLI in a FRESH child process: OCCT
        # numbers each STEP product with a process-global counter, so an in-process export order would
        # shift geom.json's sha; the child pins the first-export bytes TURB_GEOM_SHA records.
        gtdir = os.path.join(td, "geom_turb")
        pparams = os.path.join(td, "turb_params.json")
        common.write_json(pparams, dict(export.TURB_NOMINAL))
        r = subprocess.run([sys.executable, os.path.join(HERE, "export.py"), "run",
                            export.TEMPLATE, pparams, gtdir],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
        gsha = common.sha256_file(os.path.join(gtdir, "geom.json"))
        assert gsha == TURB_GEOM_SHA, gsha
        wtdir = os.path.join(td, "wedge_turb")
        res = wedge_mesh.run_turb(gtdir, wtdir, U_e=60.0, levels=(0,))
        assert res["status"] == "ok", (res["status"], res["rule"], res["message"])

        def turb_study(name, U_e, **op_changes):
            doc = common.read_json(os.path.join(common.FIXTURES, "reqs", "golden",
                                                "v3_exit_velocity.json"))["requirements"]
            for row_ in doc["rows"]:
                if row_["id"] == "REQ-001":
                    row_["value"] = 0.30
                if row_["id"] == "REQ-002":
                    row_["value"] = 0.30 / math.sqrt(2.0)
            doc["operating_point"]["U_exit_m_s"] = U_e
            doc["operating_point"]["flow_quote"] = "at %r m/s" % U_e
            doc["operating_point"].update(op_changes)
            doc["lock_sha"] = reqs.lock_sha_of(doc)
            reqs.write_locked(os.path.join(td, name), doc)
            return os.path.join(td, name)

        def pipe_copy(name):
            return shutil.copytree(pdir, os.path.join(td, name))

        def wedge_t_copy(name):
            return shutil.copytree(wtdir, os.path.join(td, name))

        def rebind_t(wd, level=0):
            path = os.path.join(wd, "wedge_turb.json")
            rep = common.read_json(path)
            row_ = [lv for lv in rep["levels"] if lv["level"] == level][0]
            pd = os.path.join(wd, "L%d" % level, "case", "constant", "polyMesh")
            row_["polymesh_sha256"] = dict((nm, common.sha256_file(os.path.join(pd, nm)))
                                           for nm in POLYMESH_FILES)
            common.atomic_write(path, common.canonical_json(rep) + chr(10))

        sturb = turb_study("study_t", 60.0)
        turb_cases = []                   # (label, case dir) - every turbulent case this selftest writes

        # T10: the 16 quotes hash to their recorded sha; REGISTRY_MODELS is the parsed registry; every
        # BC type bc_table_turb can emit is in the bc_names quote; every TRANSCRIBED_TURB row hits its
        # quote and the matching golden case file
        src_t = common.read_json(SOURCES_TURB)
        assert len(src_t["sources"]) == 16, len(src_t["sources"])
        quote_t = {}
        for s in src_t["sources"]:
            assert common.sha256_bytes(s["text"].encode("utf-8")) == s["text_sha256"], s["id"]
            assert isinstance(s["commit"], str) and s["commit"].startswith("08e10bb"), s["id"]
            assert isinstance(s["blob"], str) and len(s["blob"]) == 40, s["id"]
            quote_t[s["id"]] = s["text"]
        parsed = tuple(m.group(1) for m in
                       re.finditer(r'"([A-Za-z]+)", RasModel::', quote_t["registry"]))
        assert parsed == REGISTRY_MODELS, parsed
        emitted = set()
        for role in ROLES_TURB:
            for wl in WALLS:
                tab = bc_table_turb(role, [30.0, 0.0, 0.0], 293.15, 0.135, 223.606797749979, wl,
                                    False, 0.01, 0.003)
                emitted.update(v["type"] for v in tab.values())
        assert not emitted - set(re.findall(r'"([A-Za-z]+)"', quote_t["bc_names"])), sorted(emitted)
        ctp = os.path.join(td, "case_pipe")
        res = write_pipe_case(pdir, 0, 576.69, ctp, turb=dict(TURB_DEFAULT))
        assert res["status"] == "ok", (res["status"], res["rule"], res["message"])
        ctn = os.path.join(td, "case_nozz")
        res = write_turb_case(wtdir, 0, gtdir, sturb, ctn, turb=dict(TURB_DEFAULT))
        assert res["status"] == "ok", (res["status"], res["rule"], res["message"])
        turb_cases.append(("pipe 576.69", ctp))
        turb_cases.append(("nozzle 60", ctn))
        gold_pipe = common.read_json(GOLDEN_PIPE)
        gold_nozz = common.read_json(GOLDEN_NOZZLE_TURB)
        gtext = {"pipe": gold_pipe["texts"], "nozzle": gold_nozz["texts"]}
        quote_all = dict(quote_t)
        quote_all.update(dict((s["id"], s["text"]) for s in common.read_json(SOURCES)["sources"]))
        for kind, rows_t in (("pipe", TRANSCRIBED_TURB), ("nozzle", TRANSCRIBED_TURB_NOZZLE)):
            for sid, q_sub, c_file, w_sub, applies in rows_t:
                assert q_sub in quote_all[sid], (sid, q_sub)
                if applies in (kind, "both"):
                    assert w_sub in gtext[kind][c_file], (kind, c_file, w_sub)
        print("[ok] 16 quotes at 08e10bb hash to their recorded sha; REGISTRY_MODELS is the parsed "
              "registry; every emitted BC type is in the bc_names quote; %d pipe and %d nozzle "
              "transcription rows hit quote and case"
              % (len(TRANSCRIBED_TURB), len(TRANSCRIBED_TURB_NOZZLE)))

        # T11: the TG0 pipe case at Re_tau 576.69 - the oracle numbers, the drive line, the six field
        # files, the canonical path-free case.json, byte-identity with the golden (18 files); Re_tau
        # 2358.0 writes ok
        with open(os.path.join(ctp, "case.json"), "rb") as f:
            pj = json.loads(f.read().decode("utf-8"))
        opj, tj = pj["operating_point"], pj["turbulence"]
        for key, want in (("u_tau_m_s", 0.34970481600000003), ("g_x_m_s2", 9.78347666668751),
                          ("U_b_m_s", 6.159250975292723), ("mach", 0.01794470646570164)):
            assert abs(opj[key] - want) <= 1e-12 * abs(want), (key, opj[key], want)
        for key, want in (("k_ref_m2_s2", 0.005690455886496654),
                          ("omega_ref_1_s", 45.908346237454694)):
            assert abs(tj[key] - want) <= 1e-12 * abs(want), (key, tj[key], want)
        assert abs(tj["yplus1_apriori"] - 0.2443347243164723) <= 1e-9, tj["yplus1_apriori"]
        with open(os.path.join(ctp, "constant", "fvSources"), "r", encoding="utf-8") as f:
            assert "bodyForce       (9.78347666668751 0.0 0.0);" in f.read()
        assert sorted(os.listdir(os.path.join(ctp, "0"))) == ["T", "U", "k", "nut", "omega", "p"]
        with open(os.path.join(ctp, "case.json"), "rb") as f:
            blob = f.read()
        assert blob == (common.canonical_json(pj) + chr(10)).encode("utf-8"), "case.json is not canonical"
        assert sorted(pj) == sorted(CASE_TURB_KEYS), sorted(pj)
        assert sorted(pj["mesh"]) == sorted(MESH_KEYS), sorted(pj["mesh"])
        assert sorted(pj["operating_point"]) == sorted(OP_TURB_KEYS), sorted(pj["operating_point"])
        assert sorted(pj["turbulence"]) == sorted(TURB_KEYS), sorted(pj["turbulence"])
        assert pj["mesh"]["gc7_pass"] is True
        text = blob.decode("ascii")
        for poison in (td, td.replace(chr(92), "/"), "C:"):
            assert poison not in text, poison
        for rel in sorted(gold_pipe["texts"]):
            with open(os.path.join(ctp, rel.replace("/", os.sep)), "rb") as f:
                assert f.read().decode("utf-8") == gold_pipe["texts"][rel], rel
        assert sha_map(ctp) == gold_pipe["sha256"], \
            sorted(set(sha_map(ctp).items()) ^ set(gold_pipe["sha256"].items()))
        p2 = os.path.join(td, "case_pipe2")
        res = write_pipe_case(pdir, 0, 2358.0, p2, turb=dict(TURB_DEFAULT))
        assert res["status"] == "ok", (res["rule"], res["message"])
        with open(os.path.join(p2, "case.json"), "rb") as f:
            j2 = json.loads(f.read().decode("utf-8"))
        assert abs(j2["operating_point"]["g_x_m_s2"] - 163.56710750699514) <= 1e-12 * 163.56710750699514
        assert abs(j2["turbulence"]["yplus1_apriori"] - 0.999048500820617) <= 1e-9
        turb_cases.append(("pipe 2358", p2))
        print("[ok] TG0 pipe at Re_tau 576.69: oracle numbers, the drive line, six field files, canonical "
              "path-free case.json, byte-identical to the golden (18 files); Re_tau 2358 writes ok")

        # T12: the turbulent nozzle at U_e 60 - the oracle numbers, no fvSources anywhere, byte-identity
        # with the golden (17 files)
        with open(os.path.join(ctn, "case.json"), "rb") as f:
            nj = json.loads(f.read().decode("utf-8"))
        opn, tn = nj["operating_point"], nj["turbulence"]
        assert abs(opn["U_inlet_m_s"] - 30.0) <= 1e-12 * 30.0, opn["U_inlet_m_s"]
        assert abs(opn["U_exit_m_s"] - 60.0) <= 1e-12 * 60.0, opn["U_exit_m_s"]
        assert abs(tn["k_ref_m2_s2"] - 0.135) <= 1e-12 * 0.135, tn["k_ref_m2_s2"]
        assert abs(tn["omega_ref_1_s"] - 223.606797749979) <= 1e-12 * 223.606797749979
        assert abs(opn["mach"] - 0.17480735762532038) <= 1e-12 * 0.17480735762532038, opn["mach"]
        assert abs(tn["K_max_apriori"] - 1.065999154190022e-06) <= 1e-9 * 1.065999154190022e-06
        assert abs(tn["yplus1_apriori"] - 0.9999036429627939) <= 1e-9, tn["yplus1_apriori"]
        assert not os.path.lexists(os.path.join(ctn, "constant", "fvSources"))
        assert not os.path.lexists(os.path.join(ctn, "0", "fvSources"))
        assert nj["mesh"]["gc7_pass"] is True
        with open(os.path.join(ctn, "0", "k"), "r", encoding="utf-8") as f:
            ktxt = f.read()
        kblock = ("    inlet" + chr(10) + "    {" + chr(10)
                  + "        type            fixedValue;" + chr(10)
                  + "        value           uniform 0.13499999999999993;" + chr(10)
                  + "    }")
        assert kblock in ktxt, ktxt[:400]
        assert "turbulentIntensity" not in ktxt and "mixingLength" not in ktxt
        with open(os.path.join(ctn, "0", "omega"), "r", encoding="utf-8") as f:
            otxt = f.read()
        assert "turbulentIntensity" not in otxt and "mixingLength" not in otxt
        with open(os.path.join(ctn, "0", "U"), "r", encoding="utf-8") as f:
            assert ("internalField   uniform " + _uniform(nj["fields"]["U"]["internal"])
                    + ";") in f.read()
        for rel in sorted(gold_nozz["texts"]):
            with open(os.path.join(ctn, rel.replace("/", os.sep)), "rb") as f:
                assert f.read().decode("utf-8") == gold_nozz["texts"][rel], rel
        assert sha_map(ctn) == gold_nozz["sha256"], \
            sorted(set(sha_map(ctn).items()) ^ set(gold_nozz["sha256"].items()))
        print("[ok] turbulent nozzle at U_e 60: oracle numbers, no fvSources anywhere, byte-identical "
              "to the golden (17 files)")

        # T13: the scan covers the six fields; a zero-face stray wall patch writes the resolved row and
        # scans clean in all six; a deleted inlet block in 0/omega is found
        assert scan_bcs(ctp) == dict((f, []) for f in FIELDS_TURB), scan_bcs(ctp)
        assert scan_bcs(ctn) == dict((f, []) for f in FIELDS_TURB), scan_bcs(ctn)
        wz2 = wedge_t_copy("wt_zero")
        pmz = polymesh_write.read_polymesh(os.path.join(wz2, "L0", "case", "constant", "polyMesh"))
        n_faces = len(pmz["faces"])
        bpath = os.path.join(wz2, "L0", "case", "constant", "polyMesh", "boundary")
        with open(bpath, "r", encoding="utf-8") as f:
            btxt = f.read()
        assert btxt.count(chr(10) + "6" + chr(10) + "(") == 1, "the count line 6 is not where T13 expects"
        btxt = btxt.replace(chr(10) + "6" + chr(10) + "(", chr(10) + "7" + chr(10) + "(", 1)
        block = ("    stray" + chr(10) + "    {" + chr(10) + "        type            wall;" + chr(10)
                 + "        nFaces          0;" + chr(10) + "        startFace       " + str(n_faces)
                 + ";" + chr(10) + "    }" + chr(10))
        j = btxt.rindex(")")
        common.atomic_write(bpath, btxt[:j] + block + btxt[j:])
        rebind_t(wz2)
        roles7b = {"version": "cad-roles/1",
                   "patches": dict(TURB_NOZZLE_ROLES["patches"], stray={"role": "wall"})}
        cz2 = os.path.join(td, "case_tz")
        res = write_turb_case(wz2, 0, gtdir, sturb, cz2, roles=roles7b)
        assert res["status"] == "ok", (res["rule"], res["message"])
        assert scan_bcs(cz2) == dict((f, []) for f in FIELDS_TURB), scan_bcs(cz2)
        zrow = [p for p in res["case"]["patches"] if p["name"] == "stray"][0]
        assert zrow["U"]["type"] == "noSlip" and zrow["nut"]["type"] == "fixedValue" \
            and zrow["k"]["type"] == "fixedValue" and zrow["omega"]["type"] == "omegaWallFunction", zrow
        turb_cases.append(("nozzle stray 7", cz2))
        ch2 = os.path.join(td, "case_t_holed")
        shutil.copytree(ctn, ch2)
        opath = os.path.join(ch2, "0", "omega")
        with open(opath, "r", encoding="utf-8") as f:
            otxt = f.read()
        otxt2 = re.sub(r"    inlet\n    \{\n.*?\n    \}\n", "", otxt, count=1, flags=re.S)
        assert otxt2 != otxt, "the inlet block was not found in 0/omega"
        common.atomic_write(opath, otxt2)
        assert scan_bcs(ch2) == {"U": [], "p": [], "T": [], "k": [], "omega": ["inlet"], "nut": []}, \
            scan_bcs(ch2)
        print("[ok] scan over the six fields (both goldens clean); a zero-face stray wall patch gets "
              "the resolved row and scans clean; a deleted 0/omega entry is found")

        # T14: the plan's four refusals hit their ids, each with its passing neighbour
        res = write_turb_case(wz2, 0, gtdir, sturb, os.path.join(td, "case_t_norole"))
        refused(res, "CASE-TURB-NOBC", os.path.join(td, "case_t_norole"))
        assert "stray" in res["message"], res["message"]
        rwall = {"version": "cad-roles/1",
                 "patches": dict(PIPE_ROLES["patches"], periodic_a={"role": "wall"})}
        res = write_pipe_case(pdir, 0, 576.69, os.path.join(td, "case_p_rw"), roles=rwall)
        refused(res, "CASE-TURB-NOBC", os.path.join(td, "case_p_rw"))
        res = write_pipe_case(pdir, 0, 2400.0, os.path.join(td, "case_p2400"))
        refused(res, "CASE-YPLUS", os.path.join(td, "case_p2400"))
        assert "1.0168432" in res["message"], res["message"]
        res = write_pipe_case(pdir, 0, 2400.0, os.path.join(td, "case_p2400wf"),
                              turb=dict(TURB_DEFAULT, wall="wall_function"))
        assert res["status"] == "ok", (res["rule"], res["message"])
        with open(os.path.join(td, "case_p2400wf", "case.json"), "rb") as f:
            jw = json.loads(f.read().decode("utf-8"))
        assert abs(jw["turbulence"]["yplus1_apriori"] - 1.0168432578326891) <= 1e-9
        turb_cases.append(("pipe 2400 wf", os.path.join(td, "case_p2400wf")))
        s75 = turb_study("study_75", 75.0)
        res = write_turb_case(wtdir, 0, gtdir, s75, os.path.join(td, "case_n75"))
        refused(res, "CASE-YPLUS", os.path.join(td, "case_n75"))
        assert "1.222298" in res["message"], res["message"]
        s20 = turb_study("study_20", 20.0)
        res = write_turb_case(wtdir, 0, gtdir, s20, os.path.join(td, "case_n20"))
        refused(res, "CASE-RELAM", os.path.join(td, "case_n20"))
        assert "K_max" in res["message"] and "3.197997" in res["message"], res["message"]
        s20b = turb_study("study_20b", 20.0)
        res = write_turb_case(wtdir, 0, gtdir, s20b, os.path.join(td, "case_n20b"),
                              turb=dict(TURB_DEFAULT, bl_gate=False))
        assert res["status"] == "ok", (res["rule"], res["message"])
        with open(os.path.join(td, "case_n20b", "case.json"), "rb") as f:
            jb = json.loads(f.read().decode("utf-8"))
        assert abs(jb["turbulence"]["K_max_apriori"] - 3.1979974625700663e-06) <= 1e-9 \
            * 3.1979974625700663e-06, jb["turbulence"]["K_max_apriori"]
        turb_cases.append(("nozzle 20 nogate", os.path.join(td, "case_n20b")))
        for model, sub in (("kOmegaSSTSAS", "not in the solver's registry"),
                           ("kEpsilon", "writes only kOmegaSST"), ("laminar", "writes only kOmegaSST")):
            res = write_turb_case(wtdir, 0, gtdir, sturb, os.path.join(td, "case_m_" + model),
                                  turb=dict(TURB_DEFAULT, model=model))
            refused(res, "CASE-MODEL", os.path.join(td, "case_m_" + model))
            assert sub in res["message"], (model, res["message"])
        print("[ok] CASE-TURB-NOBC (stray unrole'd, periodic_a as wall), CASE-YPLUS (pipe 2400 refused, "
              "wall_function passes; nozzle 75 refused), CASE-RELAM (U_e 20 refused, bl_gate false "
              "passes), CASE-MODEL (SAS, kEpsilon, laminar) each hit their id")

        # T15: the TB7 wall-function rows on pipe and nozzle; every wall row of every case this selftest
        # wrote is single-family under the quoted wall-row rule, and no case pairs nutLowReWallFunction
        # with omegaWallFunction; the I sensitivity scales k_ref by 1/4 and 4
        res = write_pipe_case(pdir, 0, 2358.0, os.path.join(td, "case_p2358wf"),
                              turb=dict(TURB_DEFAULT, wall="wall_function"))
        assert res["status"] == "ok", (res["rule"], res["message"])
        turb_cases.append(("pipe 2358 wf", os.path.join(td, "case_p2358wf")))
        res = write_turb_case(wtdir, 0, gtdir, sturb, os.path.join(td, "case_n60wf"),
                              turb=dict(TURB_DEFAULT, wall="wall_function"))
        assert res["status"] == "ok", (res["rule"], res["message"])
        turb_cases.append(("nozzle 60 wf", os.path.join(td, "case_n60wf")))
        for i, want in ((0.005, 0.03375), (0.02, 0.54)):
            ci = os.path.join(td, "case_i_%r" % i)
            res = write_turb_case(wtdir, 0, gtdir, sturb, ci, turb=dict(TURB_DEFAULT, intensity=i))
            assert res["status"] == "ok", (res["rule"], res["message"])
            with open(os.path.join(ci, "case.json"), "rb") as f:
                ji = json.loads(f.read().decode("utf-8"))
            got = ji["turbulence"]["k_ref_m2_s2"]
            assert abs(got - want) <= 1e-12 * want, (i, got, want)
            if i == 0.02:
                with open(os.path.join(ci, "0", "k"), "r", encoding="utf-8") as f:
                    assert ("        value           uniform "
                            + fmt(ji["turbulence"]["k_ref_m2_s2"]) + ";") in f.read()
            turb_cases.append(("nozzle I %r" % i, ci))
        wf_dirs = (os.path.join(td, "case_p2358wf"), os.path.join(td, "case_n60wf"))
        for wd_ in wf_dirs:
            with open(os.path.join(wd_, "case.json"), "rb") as f:
                jwf = json.loads(f.read().decode("utf-8"))
            kref = jwf["turbulence"]["k_ref_m2_s2"]
            for prow in jwf["patches"]:
                if prow["role"] != "wall":
                    continue
                assert prow["nut"]["type"] == "nutkWallFunction" \
                    and prow["k"]["type"] == "kqRWallFunction" \
                    and prow["omega"]["type"] == "omegaWallFunction", (wd_, prow)
                assert prow["k"]["value"] == kref, (wd_, prow["k"])

        def nut_fam(t):
            return "Full" if t in ("nutkWallFunction", "nutUWallFunction", "nutkRoughWallFunction",
                                   "nutURoughWallFunction") else ("LowRe" if t == "nutLowReWallFunction"
                                                                  else None)

        def k_fam(t):
            return {"kqRWallFunction": "Full", "kLowReWallFunction": "LowRe"}.get(t)

        def eo_fam(t):
            return "Full" if t == "omegaWallFunction" else None

        fams = []
        for _, cdir in turb_cases:
            with open(os.path.join(cdir, "case.json"), "rb") as f:
                jd = json.loads(f.read().decode("utf-8"))
            for prow in jd["patches"]:
                if prow["role"] != "wall":
                    continue
                row_fams = [fam for fam in (nut_fam(prow["nut"]["type"]), k_fam(prow["k"]["type"]),
                                            eo_fam(prow["omega"]["type"])) if fam is not None]
                assert row_fams and len(set(row_fams)) == 1, (cdir, prow["name"], row_fams)
                fams.extend(row_fams)
                assert not (prow["nut"]["type"] == "nutLowReWallFunction"
                            and prow["omega"]["type"] == "omegaWallFunction"), (cdir, prow["name"])
        assert fams and set(fams) == {"Full"}, fams
        print("[ok] TB7 wall-function rows on pipe and nozzle (nutkWallFunction / kqRWallFunction at "
              "k_ref / omegaWallFunction); all %d cases' wall rows single-family, none pairs "
              "nutLowReWallFunction with omegaWallFunction; I 0.5 %%, 2 %% scale k_ref 1/4 and 4"
              % len(turb_cases))

        # T16: the remaining turbulent refusals, each leaving nothing behind
        for name, turbv in (("v2", dict(TURB_DEFAULT, version="cad-turb/2")),
                            ("ineg", dict(TURB_DEFAULT, intensity=-0.01)),
                            ("ibool", dict(TURB_DEFAULT, intensity=True)),
                            ("lowre", dict(TURB_DEFAULT, wall="lowRe")),
                            ("extra", dict(TURB_DEFAULT, extra=1))):
            res = write_pipe_case(pdir, 0, 576.69, os.path.join(td, "case_spec_" + name), turb=turbv)
            refused(res, "CASE-TURB-SPEC", os.path.join(td, "case_spec_" + name))
        res = write_pipe_case(pdir, 0, 0.0, os.path.join(td, "case_rt0"))
        refused(res, "CASE-TURB-SPEC", os.path.join(td, "case_rt0"))
        res = write_pipe_case(pdir, 0, float("nan"), os.path.join(td, "case_rtnan"))
        refused(res, "CASE-TURB-SPEC", os.path.join(td, "case_rtnan"))
        res = write_pipe_case(pdir, 0, 576.69, ctp)
        assert res["status"] == "refused" and res["rule"] == "CASE-OUT" and res["case"] is None, res
        assert not [n for n in os.listdir(td) if n.startswith(".case-")], td
        bp2 = pipe_copy("b_pts2")
        with open(os.path.join(bp2, "L0", "case", "constant", "polyMesh", "points"), "ab") as f:
            f.write(b"// x" + common.NB)
        res = write_pipe_case(bp2, 0, 576.69, os.path.join(td, "case_pbp"))
        refused(res, "CASE-BIND", os.path.join(td, "case_pbp"))
        assert "polyMesh/points" in res["message"], res["message"]
        bph = pipe_copy("b_hook2")

        def phook():
            with open(os.path.join(bph, "L0", "case", "constant", "polyMesh", "points"), "ab") as f:
                f.write(b"// y" + common.NB)

        res = write_pipe_case(bph, 0, 576.69, os.path.join(td, "case_pbh"), between_hook=phook)
        refused(res, "CASE-BIND", os.path.join(td, "case_pbh"))
        assert "changed while the case was written" in res["message"], res["message"]
        sb2 = os.path.join(td, "s_bad_t")
        shutil.copytree(sturb, sb2)
        common.atomic_write(os.path.join(sb2, "requirements.lock"), "0" * 64 + chr(10))
        res = write_turb_case(wtdir, 0, gtdir, sb2, os.path.join(td, "case_glock"))
        refused(res, "GATE-LOCK", os.path.join(td, "case_glock"))
        sw2 = turb_study("s_water_t", 60.0, fluid="water")
        res = write_turb_case(wtdir, 0, gtdir, sw2, os.path.join(td, "case_water_t"))
        refused(res, "CASE-FLUID", os.path.join(td, "case_water_t"))
        assert "fluid" in res["message"], res["message"]
        s86 = turb_study("s_m86_t", 86.0)
        res = write_turb_case(wtdir, 0, gtdir, s86, os.path.join(td, "case_m86_t"))
        refused(res, "CASE-MACH", os.path.join(td, "case_m86_t"))
        print("[ok] CASE-TURB-SPEC (version, intensity -0.01 and True, wall lowRe, an extra key, re_tau "
              "0 and nan), CASE-OUT, CASE-BIND (points appended; a mid-write rewrite), GATE-LOCK, "
              "CASE-FLUID (water), CASE-MACH (U_e 86) each refused with nothing left")

        # T17: the CLI - two fresh processes per writer, byte-identical to each other and to the goldens;
        # a refusal exits 1 with its id; usage exits 2; scan of a turbulent case exits 0 with the 6 keys
        for cmd, args, n_files, gold_shas in (
                ("write-pipe", [pdir, "0", "576.69"], 18, gold_pipe["sha256"]),
                ("write-nozzle-turb", [wtdir, "0", gtdir, sturb], 17, gold_nozz["sha256"])):
            outs = []
            for k in (0, 1):
                out_k = os.path.join(td, "cli_t_%s_%d" % (cmd, k))
                r = subprocess.run([sys.executable, __file__, cmd] + args + [out_k],
                                   capture_output=True, text=True, encoding="utf-8", errors="replace")
                assert r.returncode == 0, (cmd, r.returncode, r.stdout, r.stderr)
                assert json.loads(r.stdout.strip().splitlines()[-1])["files"] == n_files, r.stdout
                outs.append(out_k)
            assert sha_map(outs[0]) == sha_map(outs[1]) == gold_shas, cmd
        r = subprocess.run([sys.executable, __file__, "write-pipe", pdir, "0", "2400.0",
                            os.path.join(td, "cli_p2400")], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        assert r.returncode == 1, (r.returncode, r.stdout, r.stderr)
        assert json.loads(r.stdout.strip().splitlines()[-1])["refused"] == "CASE-YPLUS", r.stdout
        assert main(["write-pipe"]) == 2 and main(["write-nozzle-turb"]) == 2, "usage must exit 2"
        r = subprocess.run([sys.executable, __file__, "scan", os.path.join(td, "cli_t_write-pipe_0")],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
        assert json.loads(r.stdout.strip().splitlines()[-1]) == dict((f, []) for f in FIELDS_TURB), \
            r.stdout
        print("[ok] CLI: write-pipe and write-nozzle-turb twice each in fresh processes, byte-identical "
              "to each other and to the goldens (18 and 17 files); CASE-YPLUS exits 1; usage exits 2; "
              "scan exits 0 with the six keys")

    print("selftest wall %.1f s" % (time.time() - t0))
    print("SELFTEST PASS")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
