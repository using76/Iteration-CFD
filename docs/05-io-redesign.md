# I/O 재설계 계획

**meteor-cfd — OpenFOAM 케이스 형식에서 자체 형식으로**

주식회사 메테오시뮬레이션 · 개정 2 · 2026-08-24

조사 4건(NanoVDB/OpenVDB, Isaac Sim/USD, 메쉬 교환, 케이스 형식)의 결과를
반영한 판입니다. 초안의 전제 세 가지가 사실 확인으로 뒤집혔고, 각 절에
"정정"으로 표시했습니다.

---

## 0. 한 장 요약

| | 지금 | 바꾼 뒤 |
|---|---|---|
| 케이스 정의 | 13개 파일 705줄 | **JSONC 1개 파일 ~93줄** + JSON Schema |
| 지오메트리 | polyMesh 13 MB | **표면(OBJ/USD/STL) + 격자 명세 6줄** |
| 체적 메쉬 | ASCII 파일 왕복 | **메모리에서 직접 생성**; 디스크 형식은 polyMesh 유지 |
| 정밀 결과 | OpenFOAM ASCII | **VTU** (다면체 지원, 자체 구현) |
| 시각화 결과 | 없음 | **`.vdb`** — GPU에서 NanoVDB로 구축 |
| Omniverse/Isaac | 없음 | OpenFOAM·VTU를 NVIDIA 플러그인이 직접 읽음 + USD 씬 |
| 재시작 | 결과 파일 겸용 | **자체 바이너리** (분리) |
| OpenFOAM 출력 | 유일한 형식 | **유지** — Omniverse 진입 경로이기도 함 |

핵심 판단:

1. **"I/O"는 다섯 가지 서로 다른 일이다** — 케이스 정의, 지오메트리, 체적
   메쉬, 시각화, 재시작. 지금은 한 형식에 뭉쳐 있어 어느 쪽에도 최적이 아니다.
2. **표면 입력, 복셀+정밀 이중 출력** — Blender·Isaac Sim은 표면 도구다.
   체적 메쉬를 주고받는 길은 없고, 이것은 전제로 삼아야 할 사실이다.
3. **OpenFOAM 출력은 버리는 짐이 아니라 자산이다** (정정 — §4.5).

---

## 1. "I/O"는 다섯 가지 다른 일이다

| 하는 일 | 요구사항 | 형식 |
|---|---|---|
| **케이스 정의** | 사람이 쓰고 읽고 diff, 검증 가능 | JSONC + JSON Schema |
| **지오메트리** | Blender·Isaac Sim·CAD에서 옴 | OBJ / USD / 다중 STL |
| **체적 메쉬** | 정확, 대용량, 기계 전용 | 내부 생성; 파일이 필요하면 **polyMesh** |
| **정밀 결과·교환** | 다면체 보존, 후처리 도구 호환 | VTU (+ OpenFOAM 선택) |
| **시각화 결과** | 렌더러 호환, 손실 허용 | `.vdb` (GPU에서 NanoVDB) |
| **재시작** | 정확, 메쉬 정합, 비트 재현 | 자체 바이너리 |

OpenFOAM은 결과 시각화와 재시작을 같은 시간 디렉터리에 담고, 우리는 그 설계를
물려받았습니다. 이 분리가 전체 계획의 뼈대입니다 — 분리하면 VDB는 시각화만
맡으면 되므로 복셀 재샘플링의 손실이 문제가 되지 않습니다.

**전제가 되는 제약**: NanoVDB는 희소 복셀 격자, 이 솔버는 임의 다면체
유한체적입니다. 균일 직교 격자에서는 1:1로 대응해 VDB 출력이 정확하고,
비정렬 격자에서는 재샘플링이라 손실입니다. NVIDIA 자신도 같은 결론입니다 —
Kit-CAE 문서: *"NanoVDB voxelizes the data onto a regular grid... datasets
with polyhedral faces must use NanoVDB [voxelisation]."* 재샘플링은 우리가
발명한 타협이 아니라 업계의 현재 상태입니다.

---

## 2. 지금 코드가 말해주는 것 (실측)

500,000셀 균일 직교 격자, RTX 5070 Ti:

| | 시간 | 솔브 반복 환산 |
|---|---|---|
| 솔브 1회 | 5.75 ms | 1 |
| 메쉬 읽기 + 기하 계산 | 1.52 s | **264회** |
| 결과 1스텝 쓰기 (6필드, 30 MB) | 0.87 s | **151회** |

- 메쉬 파일 **72 MB** — 경계상자와 정수 3개로 완전히 기술되는 격자
- `cases/plumeB`: 압력 선택기가 `uniform cartesian (98,42,20)`으로 판정하는
  완벽한 직교 격자를 **13 MB 비정렬 polyMesh**로 저장 중
- 설정 705줄 13개 파일 → JSONC 93줄 1개 파일 (`docs/case-example.json`)
- `blockgen`이 메쉬를 ASCII로 쓰고 솔버가 되읽음. **검증 바이너리조차**
  `write_block_mesh` 다음 줄이 `read_poly_mesh` (`validate.rs:317`)
- `PatchWindow` 구조체는 순전히 OpenFOAM이 패치를 `startFace`+`nFaces`로만
  저장해서 존재하는 우회 설계

다행인 것: `io::` import의 약 85 %가 형식이 아니라 **설정 타입**이고,
`FoamDict`는 이미 경로→값 트리입니다. JSON이 정확히 같은 모양이라 추상화는
이미 존재합니다. 이것이 이 계획의 위험을 크게 낮춥니다.

---

## 3. 구조

```
  Blender ──── OBJ (usemtl=패치) ──┐
  Isaac Sim ── OBJ/glTF/USD ───────┤          ┌── case.jsonc ──┐
  CAD ──────── STL(다중 파일) ─────┤          │ 격자·물성·BC·  │
                                   ▼          │ 수치·출력 명세 │
                            표면 리더          └───────┬───────┘
                                   │                   │
                                   ▼                   ▼
                         내부 메쉬 생성기 (직교 배경격자 + 옥트리 + 컷셀)
                                   │
                                   ▼   HostMesh (메모리 — 디스크 왕복 없음)
                            ┌────────────┐
                            │    솔버    │   수치 코어 무변경
                            └──────┬─────┘
             ┌─────────────────────┼──────────────────────────┐
             ▼                     ▼                          ▼
   시각화 (손실 허용)        정밀·교환 (무손실)            재시작 (정확)
   ┌──────────────────┐    ┌──────────────────────┐    ┌──────────────┐
   │ GPU: NanoVDB     │    │ results.vtu          │    │ restart.mcr  │
   │  IndexGrid, 상주 │    │  (다면체 보존)       │    │ 자체 바이너리│
   │ 파일: .vdb       │    │ OpenFOAM ASCII(선택) │    │ 메쉬 해시    │
   │ scene.usda       │    └──────────┬───────────┘    └──────────────┘
   └────────┬─────────┘               │
            ▼                         ▼
   Blender · RTX Scientific   ParaView · Kit-CAE(NVIDIA 플러그인이
   · omni.vdb                 OpenFOAM/VTU를 직접 읽음) · PhysicsNeMo
```

배경 격자가 직교이면 구조가 스스로를 강화합니다: 메쉬가 케이스 파일에
함축되고, 셀=복셀이라 VDB가 정확하며, **cuFFT 압력 solver(25배)의 적용 범위가
넓어지고**, Blender/Isaac Sim의 표면이 정확히 필요한 입력입니다.

**비정렬 경로는 유지합니다.** 메쉬 출처가 둘(내부 생성기 / 외부 비정렬),
내부 표현 `HostMesh`는 하나입니다.

---

## 4. 부분별 설계 — 조사 결과 반영

### 4.1 케이스 정의 — JSONC + 생성된 스키마

**결정: JSON을 JSONC 방언(주석 + 후행 쉼표, 그 외 없음)으로 제한해 사용.**
첫 키는 `$schema`, 스키마는 **Rust 타입에서 `schemars`로 생성**합니다.

의존성 4개, 전부 허용 라이선스(MIT / MIT+Apache-2.0):

| 크레이트 | 역할 | 버전 (2026-08) |
|---|---|---|
| `serde_json` | 파싱·직렬화 | 1.0.151 |
| `jsonc-parser` | 주석·후행쉼표 제거 | 0.33.1 |
| `schemars` | Rust 타입 → JSON Schema | 1.2.2 |
| `serde_path_to_error` | `patches[3].kind: unknown variant …` 급 진단 | 0.1.20 |

**정정 — 자체 JSON 파서 우선 방침 철회.** 파서 자체는 쉽지만, 가치는
파서가 아니라 **스키마를 타입에서 생성하는 것**에 있습니다. SU2의 고질병이
정확히 "코드에는 있는데 문서에는 없는 옵션"(config_template.cfg와
option_structure.hpp의 불일치)이고, schemars는 이 부류의 오류를 구조적으로
없앱니다. `schemars`는 Rust 내부 태그 enum을 정확히 `oneOf` + `const`
판별자로 방출합니다 — 우리 `BcKind`·`DivScheme`·`LinearSolverKind`가 이미
그 모양입니다.

설계 세부 세 가지:

1. **패치 규칙은 객체가 아니라 순서 있는 배열.** `serde_json::Map`은 기본이
   BTreeMap(정렬 순서)이라 "파일 순서대로 패턴 매칭"이라는 기존 의미론을 JSON
   객체로는 표현할 수 없습니다. `[{ "match": "wall.*", ... }, ...]` 형태로.
2. **수식 부속 언어는 문자열 값 안에 직접 구현.**
   `"U": "(0, 0, 1.0 + 0.1*sin(2*pi*t))"` (변수 x, y, z, t). 주의: 가장 유명한
   크레이트 `evalexpr`는 **12.0.0부터 AGPL-3.0**으로 재라이선스됐습니다 —
   배제. 이 프로젝트는 이미 정규식 엔진과 토크나이저를 직접 썼으므로 작은
   수식 평가기는 자체 구현합니다.
3. **단위는 7지수 SI 차원 벡터를 검사형으로 승격.** 현재 `dimensions [0 2 -2
   0 0 0 0]`를 불투명 문자열로 왕복시키는데, `[i64; 7]`로 승격하면 차원 검사가
   공짜입니다. UCUM은 라이선스가 철회 가능·파생 금지라 부적합.

기각한 대안: YAML(성숙한 Rust 바인딩 부재 + PyYAML이 `1e-5`를 **문자열**로
읽는 수치 위험 — CFD 입력에 치명적), TOML(4단 중첩 가독성), RON(Python
생태계 부재), 자체 DSL(파서가 아니라 TextMate 문법 + 언어 서버 + 에디터
확장을 영구 유지하는 비용).

**기존 §13.4 계약(지원 안 되면 시끄럽게 실패)은 그대로 계승**하고, JSON
Schema가 그 계약의 기계 판독 형태가 됩니다.

### 4.2 지오메트리 — 표면, 규약은 "재질 이름 = 패치 이름"

**정정 아닌 확정**: 조사로 형식별 패치 정체성 보존이 판명됐습니다.

| 형식 | 패치 정체성 | 판정 |
|---|---|---|
| **OBJ** | `usemtl <재질>` 면 구간 + `g` 그룹 — Blender가 실제로 씀 | **1순위** |
| **USD** | `UsdGeomSubset`, `familyName=materialBind`, `familyType=partition` | **2순위** |
| **STL 다중 파일** | 파일명 = 패치명 (OpenFOAM triSurface와 같은 규약) | 3순위 |
| STL 단일 파일 | Blender가 `solid \n`(빈 이름)로 씀 — **정체성 없음** | 지원 안 함 |
| PLY | 면 그룹 개념 없음 | 지원 안 함 |
| glTF | 재질별 primitive | 필요시 |

**하나의 규약이 모든 경로를 관통합니다: 재질(material) 이름이 곧 패치
이름입니다.** Blender→OBJ, Blender→USD, Blender→glTF 모두 이 규약으로 면
그룹이 살아남고, USD의 `familyType = partition`은 문자 그대로 "모든 경계면은
정확히 하나의 패치에 속한다"는 CFD 불변식입니다.

Isaac Sim의 Asset Converter는 **.fbx / .obj / .gltf만** 받습니다(STL·PLY
불가). Isaac Sim에서 지오메트리를 꺼내는 현실적 경로도 OBJ/glTF입니다.

**Blender 애드온은 별도 GPL 컴포넌트여야 합니다.** `import bpy`를 하는 순간
Blender 파생물이라 GPL — 본체에 넣을 수 없고, 파일만 쓰는 독립 애드온으로
분리합니다(BlenderFDS가 같은 구조). 애드온 없이도 표준 OBJ 내보내기만으로
동작하도록 규약을 설계했으므로 애드온은 편의 기능입니다.

### 4.3 체적 메쉬 — 내부 생성

직교 배경 격자 + 옥트리 세분화 + 경계 컷셀. 문헌 기반이 확보됐습니다:

- 컷셀 생성: Aftosmis, Berger & Melton, *AIAA J.* 36(6) (1998) 952 — 25년
  검증된 방법
- 점성 벽: Berger & Aftosmis, AIAA 2012-1301 — **벽 모델 필수**가 문헌의 결론
  (우리 벽함수 자산과 부합)
- 작은 셀 — **주의, 조사 권고를 우리 솔버에 맞게 수정**: state
  redistribution(Giuliani et al.)은 **명시적** 압축성 코드(AMReX 계열)의
  해법입니다. 우리 솔버는 암시적 SIMPLE/PISO라 작은 셀의 해악이 CFL이 아니라
  **행렬 조건수**로 나타나며, 1차 대응은 **셀 병합(agglomeration)** 입니다 —
  병합된 셀은 그냥 다면체이고 우리 gather-CSR 조립이 이미 임의 다면체를
  처리하므로 구조가 잘 맞습니다. state redistribution이 필요한 곳은 **명시적
  서브사이클을 도는 VOF의 alpha 방정식**뿐입니다. 참조 구현 AMReX는 BSD-3
- 내외 판정: 일반화 권선수(winding number), Barill et al., SIGGRAPH 2018 —
  자기교차·구멍에 강건. SideFX 참조 구현이 MIT
- 공공 영역 선례: **FDS가 정확히 이 구조** — `&GEOM` 삼각 표면 입력 + 직교
  격자. NIST가 컷셀 경로의 미완성 지점(작은 셀, 모서리 효과)도 공개 문서화

snappyHexMesh의 3단계 전략(castellate→snap→layer)은 공개 문서로 알고리즘이
알려져 있으나 코드는 GPL — **독립 구현만 가능**하고, 층 추가 단계가 가장
취약하다는 것도 알려진 사실. 규모가 커서 **별도 설계 문서로 분리**합니다.

### 4.3.1 결정 — 체적 메쉬의 디스크 형식은 polyMesh (2026-08-24 확정)

새 체적 메쉬 형식을 만들지 않습니다. 근거:

1. **형식에 라이선스 부담이 없습니다.** 파일 형식은 저작물이 아니고, 우리
   리더·라이터는 데이터 파일에서 관찰한 형식으로부터 작성했습니다
   (`LICENSING.md` §3c). GPL인 것은 OpenFOAM 소프트웨어이지 형식이 아닙니다.
2. **이미 완비돼 있습니다** — 리더(`io/polymesh.rs`), 라이터(blockgen),
   기하 검증, 시험. 새 형식은 이 전부를 다시 만드는 비용입니다.
3. **생태계가 읽습니다** — ParaView(vtkOpenFOAMReader), NVIDIA
   cae-openusd-plugins(→Omniverse), 기존 OpenFOAM 도구 전부.
4. 형식의 알려진 한계(패치 = startFace+nFaces 연속 구간, `PatchWindow` 우회)는
   직교 경로에서는 메쉬가 케이스에 함축되어 **파일 자체가 없으므로** 문제가
   되지 않고, 파일이 필요한 경우(외부 비정렬 메쉬, castellated 메쉬 저장)에는
   감수할 만한 제약입니다.

역할 분담이 최종 확정됩니다: **케이스는 JSONC, 표면은 STL/OBJ/USD, 체적
메쉬 파일은 polyMesh, 정밀 결과는 VTU, 시각화는 .vdb, 재시작은 자체
바이너리.** 발명하는 형식은 재시작 하나뿐입니다.

### 4.4 시각화 출력 — GPU에서 NanoVDB, 파일은 `.vdb`

**정정 1 — 라이선스 장애물 없음.** OpenVDB·NanoVDB는 **12.0.0(2024-10-31)부터
Apache-2.0** (openvdb.org의 MPL-2.0 표기는 낡은 것 — 저장소가 기준).
NanoVDB는 헤더 온리, 외부 의존성 0.

**정정 2 — GPU 직접 생성 가능, 그리고 "읽기 전용"은 부정확.**
`nanovdb::tools::cuda::voxelsToGrid`(v12+ 네임스페이스)가 장치 포인터에서
그리드를 만들고, **토폴로지는 불변이지만 복셀 값은 제자리 변경 가능**합니다.
설계 패턴:

```
케이스 설정 시  IndexGrid(ValueOnIndex) 토폴로지 1회 구축
시간 루프      값 = 평평한 선형 장치 배열, 트리 무접촉
출력 시점      indexToGrid로 실체화 → deviceDownload 1회(선형 DMA) → 파일
```

**정정 3 — `.nvdb`는 배포 형식이 아님.** SideFX가 장기 저장 형식이 아니라고
명시, 메이저 버전이 ABI=파일 형식 변경을 의미(현재 32.9). **Blender는
`.nvdb`를 못 읽고**(OpenVDB 임포트뿐) `UsdVolOpenVDBAsset`도 `.vdb`용.
ParaView는 5.10부터 `.vdb`. → **배포 파일은 `.vdb`**, `.nvdb`는 내부
체크포인트(선택).

라이터는 자체 구현: `.nvdb`는 헤더에 바이트 단위로 완전 규격화(FileHeader
16 B, FileMetaData 176 B, `PNanoVDB.h` 3,590줄이 사실상 기계 판독 규격서).
`.vdb`는 정식 규격이 없어 **최대 구현 위험** — 예산(2~3주) 초과 시 Apache-2.0
OpenVDB를 링크하는 탈출구를 미리 정합니다.

**정직한 평가**: CFD 필드는 거의 모든 셀이 0이 아니라 VDB 희소성 이득은 거의
없습니다. 실제 이득은 텍스트 포매팅 제거와 fp16이고, 진짜 이유는 **연동**
입니다.

### 4.5 Omniverse / Isaac Sim — 조사가 뒤집은 가장 큰 전제

**정정 4 — Isaac Sim은 UsdVol 볼륨을 네이티브로 렌더링하지 않습니다.**
Isaac Sim 6.0의 렌더 모드 3종(RTX Real-Time 2.0 / Interactive / Minimal)
어디에도 `UsdVolVolume` 렌더링이 없습니다. NVIDIA 자체 문서가 명시하고,
NVIDIA 엔지니어의 공식 답변(2025-05-21)도 "RTX Scientific을 설치하라"입니다.
Isaac Sim은 로봇 시뮬레이터입니다. CFD 볼륨 렌더링의 Omniverse 경로는:

- **Kit-CAE** (엔지니어링 데이터용 Kit 앱) + **RTX Scientific / IndeX**
- 기본 RTX에서는 큐브 메쉬 + `OmniVolumeDensity` MDL 재질 우회로(밀도 1필드,
  최대 4볼륨)

**정정 5 — 우리의 OpenFOAM 라이터가 이미 Omniverse 진입 경로입니다.**
NVIDIA가 **Apache-2.0**으로 공개한 `cae-openusd-plugins`(v0.1.1, 2026-08-04)가
USD SdfFileFormat 플러그인으로 **OpenFOAM polyMesh를 직접 읽습니다** —
ASCII·바이너리 모두, 패치 이름 보존, 셀 중심 필드를 USD 타임샘플로. VTU(다면체
완전 지원), CGNS, EnSight, `.nvdb`도 읽습니다.

**따라서 초안의 "OpenFOAM을 내보내기 전용으로 강등" 방침을 철회합니다.**
OpenFOAM ASCII 출력은 현존 최고 충실도의 Omniverse 진입 경로이고, 유지 비용이
0입니다. 다만 알려진 구멍은 문서화합니다: 그 플러그인은 boundary 필드 값을
읽지 않고(internalField만), 텐서 필드를 건너뜁니다.

**신규 — VTU 라이터가 최고 가치의 단일 결정입니다.** `.vtu`는 XML +
base64/appended 바이너리로 Kitware가 완전 공개 문서화했고, `VTK_POLYHEDRON`
(타입 42)으로 임의 다면체를 보존합니다. 읽는 곳: ParaView, NVIDIA
cae-openusd-plugins(→Omniverse), 그리고 **NVIDIA PhysicsNeMo의 기본 출력이
VTK**입니다 — VTU가 사실상 NVIDIA CFD 교환 형식입니다. 자체 구현 가능, 형식
자체는 라이선스 부담 없음.

**USD 씬은 손으로 씁니다.** 최소 USDA는 라이브러리 없이 방출 가능합니다:

```usda
#usda 1.0
def Volume "plume" {
    float3[] extent = [(-7.3,-3.1,0), (7.3,3.1,3)]
    rel field:temperature = </plume/T>
    def OpenVDBAsset "T" {
        asset filePath = @./plume_0040.vdb@
        token fieldName = "temperature"
    }
}
```

경계 표면도 `def Mesh` 5개 속성 + `UsdGeomSubset`(패치)으로 방출 — Rust
`format!`으로 충분합니다. 순수 Rust `openusd` 크레이트(MIT, 0.6.0,
.usda/.usdc/.usdz 읽기·쓰기)가 필요해지면 있습니다. OpenUSD 본체는
Tomorrow OST License 1.0(상표 조항만 다른 Apache-2.0) — 사용 가능하나 링크할
필요가 없습니다.

미검증 가정 1건: cae-openusd-plugins가 Kit-CAE 밖(Isaac Sim 안)에서도
로드되는가 — 표준 `PXR_PLUGINPATH_NAME` 등록이라 구조적으로는 되어야 하지만,
**1시간짜리 실증 테스트를 5단계 시작 전에** 수행합니다.

### 4.5.1 추가 형식의 우선순위 (조사 권고 그대로)

| 형식 | 판정 | 근거 |
|---|---|---|
| **Gmsh `.msh` 4.1** | **입력 전용 지원** | 학술 사용자(무료 등급)가 들고 오는 형식. 공개 매뉴얼 기술로부터 구현 — Gmsh 소스는 GPL이므로 읽지 않음. `$PhysicalNames`가 패치 이름 |
| CGNS | 유료 고객이 이름을 대며 요구할 때까지 보류 | zlib 라이선스로 문제없으나 HDF5 빌드 무게 |
| Exodus II | 지원 안 함 | Sandia 생태계 한정 |
| MED | **지원 안 함** | LGPL — Windows 동적 링크 문제를 다시 들여옴 |
| VTKHDF | VTU 크기 불만이 실제로 나올 때만 | HDF5 대가로 얻는 것이 적음 |

**USD 출력의 정확한 성격**: 체적 메쉬를 USD 지오메트리로 번역하지 **않습니다.**
NVIDIA 자신의 구조가 정답을 확인해 줍니다 — *"네이티브 파일이 진실의 원천으로
남고, USD는 그것을 참조한다."* 우리 USD 출력은 (a) `.vtu`/`.vdb`를 가리키는
얇은 씬 그래프, (b) 렌더링 가능한 경계 표면(`UsdGeomMesh` + 패치별
`UsdGeomSubset`, 벽 물리량을 primvar로) 두 가지입니다. `UsdGeomTetMesh`는
사면체 전용 그래픽스 프리미티브라 CFD 컨테이너로 쓰지 않습니다.

### 4.6 재시작 — 자체 바이너리

헤더(버전·셀 수·필드 목록·**메쉬 해시**) + 필드별 f64 블록. 메쉬 해시 불일치
시 거부. `ofgpu-buoyant`가 재시작 시 `phi`를 쓰지 않아 potential flow로
되돌아가는 기존 문제도 여기서 해결.

---

## 5. 무엇을 포기하는가 — 그리고 누구를 위한 구조인가

**1차 사용자는 건물 규모의 부력·환경 유동 해석입니다** (이 프로젝트의 출발점이자
회사의 본업). 컷셀 직교 경로·VDB·Omniverse 연동이 이들을 위한 것입니다.
공력·기계 해석은 비정렬 경로 + 외부 체적 정합 메쉬로 받칩니다 — 포기가
아니라 우선순위입니다.


- **`.nvdb`를 기본 배포 형식으로 쓰려던 원안** — Blender가 못 읽고 저장
  형식이 아니므로. NanoVDB는 GPU 내부 표현으로서 원안의 의도(장치 상주,
  Isaac 생태계)를 그대로 달성합니다.
- **컷셀은 체적 정합 경계층의 대체가 아닙니다.** 벽 전단이 결과를 좌우하는
  문제는 비정렬 경로 + 벽함수로.
- **Isaac Sim 안에서의 볼륨 렌더링 품질은 우리 손 밖입니다** — RTX
  Scientific/Kit-CAE는 NVIDIA 독점 컴포넌트라 우리는 형식만 맞춰 줄 수
  있습니다.
- **Blender 애드온은 GPL로 분리** — 본체에 못 넣습니다.

## 6. 단계별 계획

각 단계가 독립적으로 유용하고, 빅뱅 전환이 없습니다.

| 단계 | 내용 | 완료 판정 | 위험 |
|---|---|---|---|
| **1. 분리** | 중립 설정 트리 + JSONC 파서(4크레이트) + schemars 스키마 | 같은 케이스 두 형식 → **비트 동일 결과** | 낮음 |
| **2. 출력 seam** | `ResultWriter` 트레이트 (드라이버 51곳 → 5곳) | 기존 출력과 바이트 동일 | 낮음 |
| **3. 재시작** | 자체 바이너리, `phi` 포함, 메쉬 해시 | 재시작 후 첫 압력 잔차 = 연속 실행 | 낮음 |
| **4. 메쉬 인메모리** | blockgen → HostMesh 직접 | 500k셀 기동 1.68 s → 0.3 s | 낮음 |
| **5. VTU 라이터** | 자체 구현, VTK_POLYHEDRON 포함 | ParaView + cae-openusd-plugins 로드 확인 | 낮음 |
| **5′. 실증 테스트** | cae-openusd-plugins를 Isaac Sim에서 로드 (1시간) | 판정 기록 | — |
| **6. NanoVDB GPU 경로** | IndexGrid 상주 + `.nvdb` 체크포인트 | 왕복 무손실 | 낮음 |
| **7. `.vdb` 라이터** | 배포 형식. **예산 2~3주, 초과 시 OpenVDB 링크** | Blender·ParaView에서 열림 | **높음** |
| **8. USD 씬 방출** | 볼륨 + 경계 표면 + GeomSubset, 손으로 | Omniverse에서 열림 | 중 |
| **9. 표면 입력** | OBJ(usemtl) → 다중 STL → USD(GeomSubset) | Blender 표면으로 케이스 구성 | 중 |
| **10. 컷셀 생성기** | 별도 설계 문서 (Aftosmis 1998 + state redistribution) | 표면 하나로 MMS 2차 유지 | **높음** |

1~5단계는 형식 논쟁과 무관하게 지금 이득입니다. OpenFOAM 라이터는 어느
단계에서도 제거되지 않습니다.

## 7. 위험과 미결정 — 조사 후 상태

| 항목 | 상태 |
|---|---|
| NanoVDB GPU 직접 생성 | **해결 — 가능** (`tools::cuda::voxelsToGrid`, 값 제자리 변경) |
| OpenVDB/NanoVDB 라이선스 | **해결 — Apache-2.0** (12.0.0+) |
| Isaac Sim이 읽는 것 | **해결 — UsdVol 네이티브 렌더 없음.** 경로는 Kit-CAE/RTX Scientific, 그리고 그 플러그인은 OpenFOAM·VTU를 직접 읽음 |
| USDA 수기 작성 | **해결 — 가능**, 최소 스키마 확보. 순수 Rust 크레이트도 존재(MIT) |
| 표면 패치 정체성 | **해결 — "재질=패치" 규약** (OBJ usemtl / USD GeomSubset partition) |
| 케이스 형식 | **해결 — JSONC + schemars**, 패치는 순서 배열, 수식은 자체 평가기 |
| `.vdb` 라이터 (규격 없음) | **최대 구현 위험** — 예산 초과 시 OpenVDB 링크 탈출구 |
| cae-openusd-plugins가 Isaac Sim에서 로드되는가 | **미검증** — 5′ 실증 테스트 (1시간) |
| cae-openusd-plugins 의존 위험 | v0.1.1 신생 — USD 방출(8단계)을 1급 결과물로 유지해 헤지 |
| 컷셀 벽 정확도·작은 셀 비용 | 문헌 확보(벽 모델 필수, state redistribution) — 별도 설계 문서 |
| VS Code의 2020-12 스키마 부분 구현 | 알려진 제약 — 깊은 oneOf에서 자동완성 저하 가능, draft-07 호환 방출 검토 |

라이선스 함정 기록(재발 방지): `evalexpr` 12.0.0+는 **AGPL**, `TetGen`
1.5.0+는 **AGPL**, Gmsh·Blender 코드·FluidX3D·Kit-CAE 본체는 사용 불가.
UCUM 명세는 철회 가능 라이선스. 모두 배제했습니다.

## 8. 반론 검토

외부 비판 에이전트가 서버 사정으로 세 차례 실패해, 같은 브리프의 7개 질문을
내부에서 수행한 결과입니다. 계획을 바꾼 항목은 표시했습니다.

**Q1. 이 구조가 오늘 가능한 것 중 무엇을 불가능하게 만드는가?**
기능적으로는 없음 — 비정렬 경로가 남습니다. 실제 비용은 **시험 행렬이
두 배**가 되는 것입니다: 메쉬 출처가 둘이면 새 기능마다 두 경로에서 검증해야
합니다. 완화: 내부 표현이 `HostMesh` 하나이므로 연산자·모델 시험은 공유되고,
경로별로 갈리는 것은 생성기 시험뿐입니다.

**Q2. 컷셀에서 경계층은?** 문제 부류로 갈라집니다. 플룸·건물 규모
유동은 벽이 거칠고 y+가 수백~수천이라 직교 셀 + 벽함수가 표준 관행입니다
(직교 격자 코드들이 이 구조로 오래 검증해 왔습니다). 반면 공력 해석은 y+
제어가 필요한데 컷셀에서는 첫 셀 높이가 체적 격자 해상도에 종속되어
**제어가 불가능**합니다.
→ **계획 수정: 컷셀 경로의 대상을 AEC·플룸으로 명시**하고, 공력류는
비정렬 경로 + 외부 체적 정합 메쉬로 보냅니다. 절충이 아니라 분업입니다.

**Q3. 작은 셀의 실제 비용은?** 조사 권고(state redistribution)를 §4.3에서
수정했습니다 — 그것은 명시적 코드의 해법이고, 암시적 솔버의 1차 대응은 셀
병합입니다. 병합 셀은 다면체일 뿐이라 기존 조립이 그대로 처리합니다.

**Q4. 조밀 필드에 VDB가 정직한가?** §4.4에 이미 명시(희소성 이득 거의 없음,
가치는 연동). 추가 발견 하나: 조사에 인용된 NVIDIA 포럼 사례가 정확히
**Vec3 그리드가 Omniverse에서 깨지고 Blender에서만 되는** 건이었습니다.
→ **계획 수정: 벡터 필드는 성분별 스칼라 그리드 3개 + 크기 그리드로
출력합니다** (`U.x`, `U.y`, `U.z`, `U.mag`). fp16의 해상도(T 293–1173 K에서
약 0.5 K)는 시각화에 충분하지만 필드별 정밀도 선택을 케이스에 둡니다.

**Q5. 재시작은 한 형식으로 되는가?** 안 됩니다 — 이 계획의 출발점이며,
분리(§4.6)가 답입니다. 확인만.

**Q6. 누가 사용자인가?** 이 계획이 가장 답해야 했던 질문. 이 프로젝트의
대표 케이스는 900 °C 플룸이고, 회사는 시뮬레이션 기반 디지털 트윈을
지향합니다. 즉 **건물 규모의 부력·환경 유동 해석이 본업**이고,
"표면 입력 → 직교 격자 → VDB/Omniverse 출력"은 정확히 그
사용자를 위한 구조입니다. 공력 엔지니어는 이 구조의 1차 고객이 아니며,
비정렬 경로가 그들을 받치는 2차 경로입니다. 이 우선순위를 §5에 명시합니다.

**Q7. 조사끼리 합의했지만 의심스러운 것.**
(a) "재질=패치" 규약의 실패 모드: 시각용 재질(외관)과 기능용 재질(패치)을
섞어 쓰는 모델. → 접두사 규약 `bc_<이름>`을 문서화하고, 접두사 없는 재질은
기본 벽으로. (b) VS Code의 2020-12 부분 구현: 깊은 oneOf에서 자동완성이
저하될 수 있음 → schemars의 draft-07 호환 방출을 시험. (c) cae-openusd-plugins
v0.1.1 단일 의존 위험 → USD 방출(9단계)을 1급 결과물로 유지(기존 헤지 유효).

**단계 순서에 대한 자기 반론**: 실측 고통이 가장 큰 것은 메쉬 I/O(솔브
264회분)이므로, 최대 ROI 순서는 4→2→3→1입니다. 다만 1단계(JSONC)가 이후 모든
단계의 설정 통로라 착수 순서는 팀 사정으로 정해도 됩니다 — 서로 독립입니다.

---

## 9. 다음 행동

1. ~~조사 반영~~ ~~반론 검토~~ (이 문서로 완결)
2. 1~5단계 착수 — 형식 논쟁과 무관하게 즉시 이득 (권장 순서는 팀 재량,
   최대 ROI는 4→2→3→1→5)
3. 5′ 실증 테스트: cae-openusd-plugins를 Isaac Sim에서 로드 (1시간)
4. 컷셀 생성기(10단계) 별도 설계 문서 — 대상은 AEC·플룸 유동으로 한정
