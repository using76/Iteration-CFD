#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
# Writes the 0/, constant/ and system/ files of the NH3 dispersion case on top of a converted
# polyMesh (constant/polyMesh from ofgpu-convert-mesh).  python make_case.py <caseDir> [iters]
#
# Physics, as agreed with the user on 2026-09-08:
#   - wind: 10 m/s blowing towards -x, entering through the +x face ('east'); the other four outer
#     faces ('west', 'south', 'north', 'top') are pressure outlets at 0 Pa with backflow allowed
#   - release: ammonia gas at 25 C leaving the roof of the warehouse at the ship's bow
#     (patch 'nh3_source') straight up at 2 m/s
#   - ambient: air at 25 C, steady state
# Solver: ofgpu-buoyant, which solves U, p, T with b = g*(TRef/T - 1) and the k-epsilon model.
# This tree transports no species (rust/src/bin/lowmach.rs:119 says so; ofgpu::species is only
# reached by the validator), so the ammonia is carried by the TEMPERATURE ANALOGY: pure NH3 at
# 25 C has the density of air at T_src = TRef * W_air / W_NH3 = 298.15 * 28.97 / 17.03 = 507.2 K,
# and T elsewhere is a stand-in for the mixture; nh3_from_T.py turns the T field back into the
# NH3 mass fraction, mole fraction and ppm. Heat transfer to walls is off (zeroGradient), so T is
# a passive scalar with the right buoyancy. Pr_t = 0.85 plays the role of Sc_t.
import os, sys, re

case = sys.argv[1]
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
T_AMB = 298.15                      # 25 C
T_SRC = T_AMB                       # air at ambient temperature: a mesh test, no buoyancy
U_WIND = 10.0                       # m/s, towards -x
U_SRC = 1.0                         # m/s, +z
I_WIND, L_WIND = 0.10, 50.0         # turbulence intensity and length scale of the wind
I_SRC, L_SRC = 0.10, 3.5            # ... of the release (0.07 x the 50 m building)
CMU = 0.09


def k_eps(u, i, l):
    k = 1.5 * (u * i) ** 2
    return k, CMU ** 0.75 * k ** 1.5 / l


K_WIND, E_WIND = k_eps(U_WIND, I_WIND, L_WIND)
K_SRC, E_SRC = k_eps(U_SRC, I_SRC, L_SRC)

BANNER = """/*---------------------------------------------------------------------------*\\
| NH3 dispersion at the terminal - written by make_case_air.py (air 1 m/s from the ring, mesh test)                    |
| OpenFOAM ASCII case format, read by ofgpu-buoyant (an independent solver).  |
\\*---------------------------------------------------------------------------*/
"""


def header(cls, obj, loc):
    return BANNER + "FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       %s;\n    location    \"%s\";\n    object      %s;\n}\n\n" % (cls, loc, obj)


# ---- the patches, from the converted mesh
btxt = open(os.path.join(case, 'constant', 'polyMesh', 'boundary'), encoding='utf-8').read()
patches = re.findall(r'\n\s*([A-Za-z_][\w]*)\s*\{\s*type\s+(\w+);', btxt)
assert patches, 'no patches found in constant/polyMesh/boundary'
names = [n for n, t in patches]
kinds = dict(patches)
INLET, OUTLETS = 'east', ['west', 'south', 'north', 'top']
for n in [INLET] + OUTLETS:
    assert n in kinds, 'patch %s missing (have %s)' % (n, names)
# the sources: the patch named like the case directory (pool_<case>) and every patch whose name
# starts with 'inlet' (a geometry with several sources, classification.pool_prefix ""); all of
# them release at U_SRC. Other pool_* patches are closed (walls).
CASE_NAME = os.path.basename(os.path.dirname(os.path.normpath(case)))
SOURCES = [n for n in names if n == CASE_NAME or n.lower().startswith('inlet')]
walls = [n for n, t in patches if t == 'wall'] + [n for n, t in patches if n.startswith('pool_') and n not in SOURCES]   # other pools closed
others = [n for n in names if n not in walls + [INLET] + SOURCES + OUTLETS]
assert not others, 'patches with no rule: %s (type them as walls in ofgpu-convert-mesh)' % others


def field(obj, cls, dims, internal, rules):
    s = header(cls, obj, '0') + 'dimensions      %s;\n\ninternalField   %s;\n\nboundaryField\n{\n' % (dims, internal)
    for n in names:
        body = rules(n)
        s += '    %s\n    {\n' % n + ''.join('        %-15s %s;\n' % kv for kv in body) + '    }\n'
    return s + '}\n'


def U(n):
    if n == INLET: return [('type', 'fixedValue'), ('value', 'uniform (%g 0 0)' % -U_WIND)]
    if n in SOURCES: return [('type', 'fixedValue'), ('value', 'uniform (0 0 %g)' % U_SRC)]
    if n in OUTLETS: return [('type', 'inletOutlet'), ('inletValue', 'uniform (0 0 0)'), ('value', 'uniform (0 0 0)')]
    return [('type', 'noSlip')]


def p(n):
    if n in OUTLETS: return [('type', 'fixedValue'), ('value', 'uniform 0')]
    return [('type', 'zeroGradient')]


def T(n):
    if n == INLET: return [('type', 'fixedValue'), ('value', 'uniform %g' % T_AMB)]
    if n in SOURCES: return [('type', 'fixedValue'), ('value', 'uniform %.2f' % T_SRC)]
    if n in OUTLETS: return [('type', 'inletOutlet'), ('inletValue', 'uniform %g' % T_AMB), ('value', 'uniform %g' % T_AMB)]
    return [('type', 'zeroGradient')]


def k(n):
    if n == INLET: return [('type', 'fixedValue'), ('value', 'uniform %.4g' % K_WIND)]
    if n in SOURCES: return [('type', 'fixedValue'), ('value', 'uniform %.4g' % K_SRC)]
    if n in OUTLETS: return [('type', 'inletOutlet'), ('inletValue', 'uniform %.4g' % K_WIND), ('value', 'uniform %.4g' % K_WIND)]
    return [('type', 'kqRWallFunction'), ('value', 'uniform %.4g' % K_WIND)]


def eps(n):
    if n == INLET: return [('type', 'fixedValue'), ('value', 'uniform %.4g' % E_WIND)]
    if n in SOURCES: return [('type', 'fixedValue'), ('value', 'uniform %.4g' % E_SRC)]
    if n in OUTLETS: return [('type', 'inletOutlet'), ('inletValue', 'uniform %.4g' % E_WIND), ('value', 'uniform %.4g' % E_WIND)]
    return [('type', 'epsilonWallFunction'), ('value', 'uniform %.4g' % E_WIND)]


def nut(n):
    if n in walls: return [('type', 'nutkWallFunction'), ('value', 'uniform 0')]
    return [('type', 'calculated'), ('value', 'uniform 0')]


files = {
    '0/U': field('U', 'volVectorField', '[0 1 -1 0 0 0 0]', 'uniform (%g 0 0)' % -U_WIND, U),
    '0/p': field('p', 'volScalarField', '[0 2 -2 0 0 0 0]', 'uniform 0', p),
    '0/T': field('T', 'volScalarField', '[0 0 0 1 0 0 0]', 'uniform %g' % T_AMB, T),
    '0/k': field('k', 'volScalarField', '[0 2 -2 0 0 0 0]', 'uniform %.4g' % K_WIND, k),
    '0/epsilon': field('epsilon', 'volScalarField', '[0 2 -3 0 0 0 0]', 'uniform %.4g' % E_WIND, eps),
    '0/nut': field('nut', 'volScalarField', '[0 2 -1 0 0 0 0]', 'uniform 0', nut),
    'constant/g': header('uniformDimensionedVectorField', 'g', 'constant') + 'dimensions      [0 1 -2 0 0 0 0];\ng               [0 1 -2 0 0 0 0] (0 0 -9.81);\n',
    'constant/physicalProperties': header('dictionary', 'physicalProperties', 'constant') +
        'viscosityModel  constant;\nnu              [0 2 -1 0 0 0 0] 1.5e-05;\nPr              0.71;\nPrt             0.85;\nTRef            %g;\n' % T_AMB,
    'constant/momentumTransport': header('dictionary', 'momentumTransport', 'constant') +
        'simulationType  RAS;\n\nRAS\n{\n    model           kEpsilon;\n    turbulence      on;\n    printCoeffs     on;\n}\n',
    'system/controlDict': header('dictionary', 'controlDict', 'system') +
        'application     foamRun;\nstartFrom       startTime;\nstartTime       0;\nstopAt          endTime;\nendTime         %d;\ndeltaT          1;\n'
        'writeControl    timeStep;\nwriteInterval   %d;\npurgeWrite      0;\nwriteFormat     ascii;\nwritePrecision  6;\nwriteCompression off;\n'
        'timeFormat      general;\ntimePrecision   6;\nrunTimeModifiable true;\n' % (ITERS, ITERS),
    'system/fvSchemes': header('dictionary', 'fvSchemes', 'system') +
        'ddtSchemes\n{\n    default         steadyState;\n}\n\ngradSchemes\n{\n    default         Gauss linear;\n}\n\n'
        'divSchemes\n{\n    default         none;\n    div(phi,U)       bounded Gauss upwind;\n    div(phi,T)       bounded Gauss upwind;\n'
        '    div(phi,k)       bounded Gauss upwind;\n    div(phi,epsilon) bounded Gauss upwind;\n}\n\n'
        'laplacianSchemes\n{\n    default         Gauss linear uncorrected;\n}\n\ninterpolationSchemes\n{\n    default         linear;\n}\n\n'
        'snGradSchemes\n{\n    default         uncorrected;\n}\n',
    'system/fvSolution': header('dictionary', 'fvSolution', 'system') +
        'solvers\n{\n'
        '    p\n    {\n        solver          PBiCGStab;\n        preconditioner  DIC;\n        tolerance       1e-08;\n        relTol          0.01;\n        maxIter         1000;\n    }\n'
        '    Phi\n    {\n        solver          PBiCGStab;\n        preconditioner  DIC;\n        tolerance       1e-10;\n        relTol          0;\n        maxIter         5000;\n    }\n'
        + ''.join('    %s\n    {\n        solver          PBiCGStab;\n        preconditioner  diagonal;\n        tolerance       1e-08;\n        relTol          0.1;\n        maxIter         200;\n    }\n' % f for f in ('U', 'T', 'k', 'epsilon'))
        + '}\n\nSIMPLE\n{\n    nNonOrthogonalCorrectors 0;\n    residualControl\n    {\n        p               1e-4;\n        U               1e-4;\n        T               1e-4;\n        k               1e-4;\n        epsilon         1e-4;\n    }\n}\n\n'
        'relaxationFactors\n{\n    fields\n    {\n        p               0.1;\n    }\n    equations\n    {\n        U               0.3;\n        T               0.7;\n        k               0.7;\n        epsilon         0.7;\n    }\n}\n',
}
for rel, text in files.items():
    path = os.path.join(case, *rel.split('/'))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, 'w', encoding='utf-8', newline='\n').write(text)
print('case written to %s: patches %s' % (case, names))
print('  walls: %s' % walls)
print('  wind %g m/s -x through %s (k %.3g eps %.3g); outlets %s at p = 0; source %s: (0 0 %g) m/s, T %.1f K (pure NH3 analogy, k %.3g eps %.3g)' % (
    U_WIND, INLET, K_WIND, E_WIND, OUTLETS, SOURCES, U_SRC, T_SRC, K_SRC, E_SRC))
print('  %d SIMPLE iterations, write at the end' % ITERS)
