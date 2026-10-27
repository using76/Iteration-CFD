# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
"""Case definitions for the MMS generator (tools/mms/gen.py).

Data only: the symbols, the exact parameter values, the manufactured phi
expressions and the manifest texts. gen.py differentiates PHI_2D / PHI_3D
and prints the generated Rust module and the formula manifests.
"""

from sympy import cos, pi, sin, symbols

X, Y, Z, T = symbols("x y z t")
(L, PHI0, PHI_X, PHI_Y, PHI_Z, PHI_XY, PHI_YZ,
 A_X, A_Y, A_Z, A_XY, A_YZ, EPS, OMEGA) = symbols(
    "L PHI0 PHI_X PHI_Y PHI_Z PHI_XY PHI_YZ A_X A_Y A_Z A_XY A_YZ EPS OMEGA")
UX, UY, UZ, GAMMA = symbols("U_x U_y U_z Gamma")

PHI_2D = (PHI0 + PHI_X * sin(A_X * pi * X / L) + PHI_Y * cos(A_Y * pi * Y / L)
          + PHI_XY * cos(A_XY * pi * X * Y / L**2))
PHI_3D = (PHI_2D + PHI_Z * sin(A_Z * pi * Z / L)
          + PHI_YZ * cos(A_YZ * pi * Y * Z / L**2))

# The four generated cases: (rust module name, case id, dim, steady, u0).
CASES = [
    ("ver01_2d", "VER-01-2D", 2, True, (1.0, 0.6, 0.0)),
    ("ver01_3d", "VER-01-3D", 3, True, (1.0, 0.6, 0.4)),
    ("ver02_2d", "VER-02-2D", 2, False, (1.0, 0.6, 0.0)),
    ("ver02_3d", "VER-02-3D", 3, False, (1.0, 0.6, 0.4)),
]

# The Rust constants of the generated module, name and literal value.
CONSTS = [
    ("L", "1.0"), ("PHI0", "1.0"), ("PHI_X", "0.15"), ("PHI_Y", "-0.1"),
    ("PHI_Z", "0.12"), ("PHI_XY", "0.08"), ("PHI_YZ", "0.05"),
    ("A_X", "0.75"), ("A_Y", "1.25"), ("A_Z", "0.6"), ("A_XY", "0.9"),
    ("A_YZ", "0.7"), ("EPS", "0.5"), ("OMEGA", "2.0 * PI"), ("GAMMA0", "0.05"),
]

# Manifest texts (docs/19 section 8 format).
MANIFESTS = {
    "VER-01": {
        "title": "VER-01: 정상 상태 제조해 스칼라 대류-확산 (2-D / 3-D)",
        "intro": ("Roy 등(2004)의 삼각함수형 제조해로 스칼라 대류-확산 솔버의 공간 수렴 "
                  "차수를 검증한다. 2-D 변형은 z에 독립이고, 3-D 변형은 z 삼각항과 yz "
                  "교차항을 추가한다. 소스항은 셀 중심 점값이 아니라 4점 가우스-르장드르 "
                  "텐서 규칙으로 구한 셀 평균으로 주입한다."),
    },
    "VER-02": {
        "title": "VER-02: 비정상 제조해 스칼라 대류-확산 (2-D / 3-D)",
        "intro": ("VER-01의 제조해에 (1 + eps sin(omega t)) 인자를 곱해 비정상 소스를 "
                  "만들고, 시간 차분 스킴(Euler, backward, Crank-Nicolson)의 시간 수렴 "
                  "차수를 검증한다. 2-D/3-D 변형과 소스의 셀 평균 주입은 VER-01과 같다."),
    },
}

MANIFEST_PARAMS = {
    "VER-01": {
        "L": 1.0, "phi0": 1.0, "phi_x": 0.15, "phi_y": -0.1, "phi_z": 0.12,
        "phi_xy": 0.08, "phi_yz": 0.05, "a_x": 0.75, "a_y": 1.25, "a_z": 0.6,
        "a_xy": 0.9, "a_yz": 0.7, "Gamma0": 0.05,
        "u_2d": 1.0, "v_2d": 0.6, "u_3d": 1.0, "v_3d": 0.6, "w_3d": 0.4,
    },
    "VER-02": {
        "L": 1.0, "phi0": 1.0, "phi_x": 0.15, "phi_y": -0.1, "phi_z": 0.12,
        "phi_xy": 0.08, "phi_yz": 0.05, "a_x": 0.75, "a_y": 1.25, "a_z": 0.6,
        "a_xy": 0.9, "a_yz": 0.7, "eps": 0.5, "omega": 6.283185307179586,
        "Gamma0": 0.05, "u_2d": 1.0, "v_2d": 0.6, "u_3d": 1.0, "v_3d": 0.6,
        "w_3d": 0.4,
    },
}

SOURCES = [
    {"cite": 'P. J. Roache, "Code Verification by the Method of Manufactured '
             'Solutions", J. Fluids Eng. 124 (2002) 4-10',
     "doi_or_url": "10.1115/1.1436090"},
    {"cite": 'K. Salari, P. Knupp, "Code Verification by the Method of '
             'Manufactured Solutions", Sandia report SAND2000-1444 (2000)',
     "doi_or_url": "10.2172/759450"},
    {"cite": 'C. J. Roy, C. C. Nelson, T. M. Smith, C. C. Ober, "Verification '
             'of Euler and Navier-Stokes Codes by the Method of Manufactured '
             'Solutions", Int. J. Numer. Meth. Fluids 44 (2004) 599-620',
     "doi_or_url": "10.1002/fld.660"},
    {"cite": 'A. Meurer et al., "SymPy: symbolic computing in Python", '
             'PeerJ Comput. Sci. 3 (2017) e103',
     "doi_or_url": "10.7717/peerj-cs.103"},
]
