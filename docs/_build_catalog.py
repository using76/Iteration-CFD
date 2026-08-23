#!/usr/bin/env python3
"""
Regenerate docs/01-model-catalog.md and docs/02-gpu-portability.md from the
structured output the cataloguing agents produced.

The agents returned validated JSON (one object per subsystem, ~1100 components
in total). Handing that to a writing model and asking for tables loses rows -
it summarises. This script does not: every component in the data becomes
exactly one row.

    python docs/_build_catalog.py <journal.jsonl> [more.jsonl ...]
"""

import html
import json
import os
import sys
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))

TIER_ORDER = ["A-trivial", "B-sparse-solve", "C-fft", "D-hard", "E-cpu-only"]

TIER_BLURB = {
    "A-trivial": "Elementwise or face-stencil work. Pure CUDA kernels, no library needed, "
                 "fully device resident today.",
    "B-sparse-solve": "Cost is dominated by solving a sparse system. AMGX (multigrid) or "
                      "cuDSS (direct) replaces the OpenFOAM solver; the assembly around it "
                      "is tier A.",
    "C-fft": "Spectral or convolution work - cuFFT.",
    "D-hard": "Irregular, serial, topology-mutating or search-heavy. A device-resident port "
              "is a research project, not a transcription.",
    "E-cpu-only": "I/O, dictionaries, runtime selection, decomposition setup. Belongs on the "
                  "host and should stay there.",
}


def esc(s):
    """Make a value safe inside a markdown table cell."""
    if s is None:
        return ""
    s = str(s).replace("\r", " ").replace("\n", " ").strip()
    s = s.replace("|", "\\|")
    while "  " in s:
        s = s.replace("  ", " ")
    return s


def load(paths):
    catalogs = OrderedDict()
    verdicts = OrderedDict()

    for p in paths:
        if not os.path.exists(p):
            print("skip (missing):", p, file=sys.stderr)
            continue
        with open(p, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if d.get("type") != "result":
                    continue
                r = d.get("result")
                if not isinstance(r, dict):
                    continue
                key = r.get("subsystem") or r.get("area") or "?"
                if "components" in r:
                    catalogs[key] = r
                if "verdicts" in r:
                    verdicts[key] = r

    return catalogs, verdicts


def short_subsystem(name):
    """The agents returned long descriptive subsystem strings; trim to a heading."""
    for sep in [" (", " - ", ", OpenFOAM"]:
        if sep in name:
            name = name.split(sep)[0]
    return name.strip()


def write_catalog(catalogs, out):
    n_rows = 0
    L = []
    L.append("# OpenFOAM model catalogue")
    L.append("")
    L.append("OpenFOAM-12 (OpenFOAM Foundation) 소스 트리에 들어 있는 모든 "
             "런타임 선택 가능 모델·스킴·솔버·메쉬 구성요소의 전수 목록입니다.")
    L.append("")
    L.append("Every entry was read out of `upstream/OpenFOAM-Foundation-12/src`. "
             "The *Keyword* column is what you write in a dictionary to select the "
             "component; an empty keyword means it is a base class, a helper, or "
             "selected implicitly.")
    L.append("")
    L.append("ESI(v2606)에만 있는 모델은 `03-esi-vs-foundation.md`를 보세요. "
             "GPU 이식성 등급은 `02-gpu-portability.md`에 있습니다.")
    L.append("")

    # Summary table
    L.append("## Contents")
    L.append("")
    L.append("| Subsystem | Components |")
    L.append("|---|---:|")
    for name, cat in catalogs.items():
        L.append("| [%s](#%s) | %d |"
                 % (esc(short_subsystem(name)),
                    short_subsystem(name).lower().replace(" ", "-").replace("/", "").replace(":", ""),
                    len(cat["components"])))
    L.append("| **Total** | **%d** |" % sum(len(c["components"]) for c in catalogs.values()))
    L.append("")

    for name, cat in catalogs.items():
        L.append("---")
        L.append("")
        L.append("## " + short_subsystem(name))
        L.append("")
        if cat.get("notes"):
            # The agents' notes run to whole essays. Break them at sentence
            # ends so the blockquote is readable instead of one huge line.
            note = " ".join(str(cat["notes"]).split())
            L.append("> **Subsystem notes**")
            L.append(">")
            buf = ""
            for sentence in note.replace(". ", ".\x00").split("\x00"):
                if len(buf) + len(sentence) > 300 and buf:
                    L.append("> " + buf.strip())
                    buf = ""
                buf += sentence + " "
            if buf.strip():
                L.append("> " + buf.strip())
            L.append("")

        # group by category
        by_cat = OrderedDict()
        for c in cat["components"]:
            by_cat.setdefault(c.get("category", "other"), []).append(c)

        for catname in sorted(by_cat):
            items = by_cat[catname]
            L.append("### " + esc(catname) + "  <sub>(%d)</sub>" % len(items))
            L.append("")
            L.append("| Name | Keyword | Path | What it computes | Equations |")
            L.append("|---|---|---|---|---|")
            for c in sorted(items, key=lambda x: x.get("name", "")):
                L.append("| `%s` | %s | `%s` | %s | %s |" % (
                    esc(c.get("name")),
                    ("`%s`" % esc(c["selectable"])) if c.get("selectable") else "",
                    esc(c.get("path")),
                    esc(c.get("purpose")),
                    esc(c.get("equations")),
                ))
                n_rows += 1
            L.append("")

    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")

    return n_rows


def write_portability(verdicts, out):
    n_rows = 0
    L = []
    L.append("# GPU portability of the OpenFOAM model set")
    L.append("")
    L.append("각 구성요소를 **완전히 device에 상주하는 시간 루프** 기준으로 분류했습니다. "
             "대상 하드웨어는 NVIDIA RTX 5070 Ti (Blackwell, sm_120, 16 GB), CUDA 13.3.")
    L.append("")
    L.append("The bar is not \"can a GPU help here\" - almost anything can be accelerated "
             "somehow. The bar is: **can this run for the whole time loop without a host "
             "round-trip?** That is what decides whether a component can join a "
             "device-resident solver or whether it forces a synchronisation every step.")
    L.append("")

    L.append("## Tiers")
    L.append("")
    L.append("| Tier | Meaning |")
    L.append("|---|---|")
    for t in TIER_ORDER:
        L.append("| **%s** | %s |" % (t, TIER_BLURB[t]))
    L.append("")

    # Tier census
    census = OrderedDict()
    for name, v in verdicts.items():
        row = OrderedDict((t, 0) for t in TIER_ORDER)
        row["other"] = 0
        for d in v["verdicts"]:
            t = d.get("gpuTier", "")
            hit = None
            for known in TIER_ORDER:
                if known.split("-")[0] == t.split("-")[0] and t:
                    hit = known
                    break
            if hit:
                row[hit] += 1
            else:
                row["other"] += 1
        census[name] = row

    L.append("## Census")
    L.append("")
    L.append("| Subsystem | " + " | ".join(TIER_ORDER) + " | other | total |")
    L.append("|---|" + "---:|"*(len(TIER_ORDER)+2))
    totals = OrderedDict((t, 0) for t in TIER_ORDER)
    totals["other"] = 0
    for name, row in census.items():
        tot = sum(row.values())
        for k in row:
            totals[k] += row[k]
        L.append("| %s | %s | %d | %d |" % (
            esc(short_subsystem(name)),
            " | ".join(str(row[t]) for t in TIER_ORDER),
            row["other"], tot))
    L.append("| **Total** | %s | **%d** | **%d** |" % (
        " | ".join("**%d**" % totals[t] for t in TIER_ORDER),
        totals["other"], sum(totals.values())))
    L.append("")

    L.append("## Recommended porting order")
    L.append("")
    L.append("속도 향상 대비 노력이 큰 순서입니다.")
    L.append("")
    L.append("1. **Mesh + LDU addressing on the device** (tier A). 모든 것의 전제조건입니다. "
             "`lduAddressing`의 `lowerAddr`/`upperAddr`, `V`, `Sf`, `magSf`, `deltaCoeffs`, "
             "`weights`, `nonOrthCorrectionVectors`를 한 번 올리고 끝냅니다. "
             "이미 `gpu/common/src/ofgpu_mesh.cu`에 있습니다.")
    L.append("2. **`fvm::` / `fvc::` Gauss operators** (tier A). 면 루프 하나가 커널 하나입니다. "
             "OpenFOAM의 scatter를 cell->face CSR gather로 뒤집는 것이 핵심입니다.")
    L.append("3. **The linear solver** (tier B). 난류 모델은 반복당 두 번의 sparse solve로 "
             "시간을 씁니다. 자체 PBiCGStab로도 충분하고, 압력 방정식이 들어오면 AMGX가 필요합니다.")
    L.append("4. **Eddy-viscosity RAS models** (tier A + B). k-epsilon, k-omega, "
             "kOmegaSST, realizableKE, RNGkEpsilon, SpalartAllmaras - 전부 같은 골격입니다. "
             "k-epsilon과 k-omega는 `gpu/kEpsilon`, `gpu/kOmega`에 구현되어 있습니다.")
    L.append("5. **Wall functions** (tier A). 벽면당 커널 하나. 벽셀 평균만 "
             "atomic 없이 gather로 바꾸면 됩니다.")
    L.append("6. **The SIMPLE/PIMPLE pressure equation** (tier B). 대칭 행렬이라 "
             "AMGX의 classical AMG가 가장 잘 맞습니다. `GAMG` + `DIC`를 대체합니다.")
    L.append("7. **LES SGS models** (tier A). 필터 연산이 face 스텐실이라 RAS와 같습니다. "
             "`dynamicLagrangian`만 시간 평균 필드가 추가로 필요합니다.")
    L.append("8. **Thermophysical property evaluation** (tier A). "
             "JANAF/NASA 다항식은 셀당 독립이라 이상적인 커널입니다.")
    L.append("9. **Chemistry ODE integration** (tier A-ish, but stiff). "
             "셀당 독립이라 병렬성은 완벽하지만, 셀마다 스텝 수가 달라 warp divergence가 심합니다. "
             "셀을 강성도로 정렬하거나 배치 solver를 쓰는 것이 정석입니다.")
    L.append("10. **Lagrangian particle tracking** (tier D). 셀 탐색, 동적 리스트, "
             "충돌 - GPU 재설계가 필요합니다.")
    L.append("11. **Mesh generation and topology change** (tier D/E). "
             "`snappyHexMesh`, `polyTopoChange`, AMR은 host에 두는 것이 맞습니다.")
    L.append("")

    L.append("## Mesh data structures on the GPU")
    L.append("")
    L.append("한 번 올리면 정적 메쉬에서는 다시 건드리지 않는 배열입니다.")
    L.append("")
    L.append("| OpenFOAM | Size | Why the GPU needs it |")
    L.append("|---|---|---|")
    L.append("| `lduAddr().lowerAddr()` (= `owner`) | nInternalFaces | 행렬 그래프의 절반 |")
    L.append("| `lduAddr().upperAddr()` (= `neighbour`) | nInternalFaces | 나머지 절반 |")
    L.append("| `mesh.V()` | nCells | 모든 체적 적분 |")
    L.append("| `mesh.C()` | nCells | 기울기, 비직교 보정 |")
    L.append("| `mesh.Sf()`, `mesh.magSf()` | nInternalFaces + nBoundaryFaces | 모든 면 적분 |")
    L.append("| `mesh.Cf()` | 면 전체 | 보간 가중치 |")
    L.append("| `surfaceInterpolation::weights()` | nInternalFaces | 선형 보간 |")
    L.append("| `nonOrthDeltaCoeffs()` | 면 전체 | laplacian, snGrad |")
    L.append("| `nonOrthCorrectionVectors()` | nInternalFaces | 비직교 명시적 보정 |")
    L.append("| patch `faceCells()` | 패치별 | 경계 조립 |")
    L.append("| `nearWallDist` / `y` | 벽 패치 | 벽함수 |")
    L.append("")
    L.append("여기에 더해, OpenFOAM에는 없지만 GPU에는 **반드시 필요한** 것이 하나 있습니다: "
             "**cell -> face CSR**. OpenFOAM은 대각 성분을 `diag[owner[f]] -= ...` 처럼 "
             "면 루프에서 흩뿌립니다. GPU에서 그러려면 double atomic이 필요하고 합산 순서가 "
             "비결정적이 됩니다. 역맵을 만들어 두면 셀 하나가 스레드 하나로 자기 면들을 "
             "모으기만 하면 되고, 결과가 비트 단위로 재현됩니다.")
    L.append("")
    L.append("**절대 device에 상주할 수 없는 메쉬 연산:**")
    L.append("")
    L.append("- `polyTopoChange` 계열 (셀 분할/병합, layer addition, sliding interface) - "
             "주소 배열 자체를 다시 만듭니다")
    L.append("- `snappyHexMesh`, `blockMesh` - 생성 단계이고 트리 탐색과 동적 자료구조를 씁니다")
    L.append("- `decomposePar` / `renumberMesh` - 그래프 분할 (METIS/Scotch)")
    L.append("- `meshSearch`, `treeBoundBox` 계열 - 재귀적 트리 탐색")
    L.append("- `fvMeshDistributor` - MPI 재분배")
    L.append("")
    L.append("움직이지만 위상은 안 바뀌는 메쉬(`fvMeshMovers`, `motionSolvers`)는 중간입니다: "
             "점 이동은 tier A, 기하 재계산도 tier A, 그런데 `displacementLaplacian` 같은 "
             "모션 솔버는 자체 sparse solve가 있으므로 tier B입니다.")
    L.append("")

    L.append("## Library fit")
    L.append("")
    L.append("### AMGX")
    L.append("")
    L.append("대체 대상: `GAMG` (`src/OpenFOAM/matrices/lduMatrix/solvers/GAMG`)와 그 "
             "agglomeration (`pairGAMGAgglomeration`, `algebraicPairGAMGAgglomeration`, "
             "`faceAreaPairGAMGAgglomeration`), 그리고 `PCG`/`PBiCG`/`PBiCGStab` + "
             "`DIC`/`DILU`/`FDIC` 조합.")
    L.append("")
    L.append("압력 방정식이 진짜 표적입니다. 비압축성 솔버 실행시간의 60~80 %가 여기 있고, "
             "대칭 M-행렬이라 classical AMG가 가장 잘 듣습니다. AMGX는 setup과 solve가 "
             "모두 device에 있어서 host 왕복이 없습니다. LDU를 CSR로 한 번 순열해 두면 "
             "값 채우기는 gather 한 번입니다 (`GpuCsrMatrix`가 그렇게 되어 있습니다).")
    L.append("")
    L.append("### cuDSS")
    L.append("")
    L.append("대체 대상: 직접법이 필요한 작고 조밀한 계 - `simpleMatrix`, `scalarMatrices` "
             "(`LUDecompose`/`LUsolve`), 화학 반응 Jacobian, `EulerImplicit`/`ode` "
             "chemistry solver 내부의 dense solve.")
    L.append("")
    L.append("메인 압력/속도 계에는 맞지 않습니다 - 3-D FV 행렬은 fill-in이 심해서 "
             "직접 분해가 반복법보다 훨씬 비쌉니다. 반면 화학은 종 개수만큼(보통 10~100) "
             "작은 dense 계를 셀마다 풀어야 해서 배치 직접법이 정확히 맞습니다.")
    L.append("")
    L.append("### cuFFT")
    L.append("")
    L.append("대체 대상: `src/randomProcesses` (`fft`, `Kmesh`, `UOprocess`, "
             "`noiseModels`), `turbulentDFSEMInlet` / 합성 난류 유입 생성, "
             "그리고 균일 격자 위의 스펙트럼 후처리 (`energySpectrum`).")
    L.append("")
    L.append("범위가 좁습니다. OpenFOAM의 핵심 반복 루프에는 FFT가 없습니다 - "
             "비정렬 격자 FV 코드이기 때문입니다. cuFFT가 값을 하는 곳은 "
             "LES 유입 조건 생성과 음향 후처리입니다.")
    L.append("")
    L.append("### cuSPARSE / cuBLAS / CUB")
    L.append("")
    L.append("직접적인 OpenFOAM 대응물은 없지만 자체 솔버의 재료입니다. "
             "cuSPARSE는 CSR SpMV, cuBLAS는 dot/axpy, CUB는 리덕션입니다. "
             "이 프로젝트는 LDU gather SpMV를 직접 쓰고 리덕션만 CUB에 맡깁니다 - "
             "LDU 구조가 이미 최적이라 CSR로 바꿀 이유가 없기 때문입니다.")
    L.append("")

    for name, v in verdicts.items():
        L.append("---")
        L.append("")
        L.append("## " + short_subsystem(name))
        L.append("")
        rows = v["verdicts"]

        by_tier = OrderedDict()
        for d in rows:
            by_tier.setdefault(d.get("gpuTier", "unknown"), []).append(d)

        def tier_key(t):
            for i, k in enumerate(TIER_ORDER):
                if t.split("-")[0] == k.split("-")[0]:
                    return i
            return 99

        for tier in sorted(by_tier, key=tier_key):
            items = by_tier[tier]
            L.append("### %s  <sub>(%d)</sub>" % (esc(tier), len(items)))
            L.append("")
            L.append("| Component | Library | Why | Blockers | Effort | Path |")
            L.append("|---|---|---|---|---|---|")
            for d in sorted(items, key=lambda x: x.get("name", "")):
                L.append("| `%s` | %s | %s | %s | %s | `%s` |" % (
                    esc(d.get("name")),
                    esc(d.get("gpuLibrary")),
                    esc(d.get("rationale")),
                    esc(d.get("blockers")) or "—",
                    esc(d.get("effort")),
                    esc(d.get("path")),
                ))
                n_rows += 1
            L.append("")

    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")

    return n_rows


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    catalogs, verdicts = load(sys.argv[1:])

    print("subsystems: %d catalogued, %d classified"
          % (len(catalogs), len(verdicts)))

    n1 = write_catalog(catalogs, os.path.join(HERE, "01-model-catalog.md"))
    n2 = write_portability(verdicts, os.path.join(HERE, "02-gpu-portability.md"))

    print("01-model-catalog.md    : %d component rows" % n1)
    print("02-gpu-portability.md  : %d verdict rows" % n2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
