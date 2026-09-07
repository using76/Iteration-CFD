# racecar — 외부 공력 샘플 / external-aerodynamics sample

이 폴더에 있는 것은 **형상 하나(`racecar.stl`)와 명령 두 줄**입니다. 메쉬와 결과는
없습니다 — 합쳐서 1 GB가 넘고, 그것을 만드는 것이 이 프로그램이 하는 일이기
때문입니다. 아래를 그대로 실행하면 210만 셀짜리 컷셀 격자와 수렴한 해가
여러분의 GPU 위에서 만들어집니다.

What ships here is **one geometry (`racecar.stl`) and two commands**. There is no
mesh and no result: together they are over a gigabyte, and producing them is what
this program is for. Run the two lines below and a 2.1-million-cell cut-cell mesh
and a converged solution are built on your own GPU.

## 딸깍 / One click

설치 프로그램이 `bin`을 PATH에 넣었다면:

```
racecar.cmd
```

이 스크립트는 아래 두 단계를 순서대로 실행합니다.

## 두 단계 / The two steps

```powershell
# 1. 단위 풍동(1 m 정육면체)에 차체를 컷셀로 새깁니다.
ofgpu-generate-mesh big racecar_case 128 -stl car=racecar.stl -cutcell

# 2. k-omega 로 2,500회 풉니다. 결과는 racecar_case\1\ 에 OpenFOAM ASCII 로 쓰입니다.
ofgpu-k-omega racecar_case -iters 2500 -check 250 -output foam
```

1단계는 형상 분류가 CPU에서 도는 구간이라 **10–20분** 걸립니다. 2단계는 GPU에
상주하는 루프라 RTX 5070 Ti 기준 **약 40초** — 초당 1억 3천만 셀-반복입니다.

Step 1 takes **10–20 minutes**: the inside/outside classification runs on the CPU.
Step 2 is the GPU-resident loop and takes **about 40 seconds** on an RTX 5070 Ti —
130 million cell-iterations a second.

`momentumTransport`의 `model`은 생성 직후 `kEpsilon`입니다. k-omega로 풀려면
`kOmega`로 바꾸거나, `ofgpu-k-epsilon`을 대신 쓰십시오.

The generated `momentumTransport` says `kEpsilon`. Change it to `kOmega` to run the
k-omega driver, or run `ofgpu-k-epsilon` instead.

## 무엇을 보게 되는가 / What you get to look at

Studio의 3D 뷰어에서 `racecar_case`를 열면 컷셀 격자 속에 숨어 있던 블록이
복원되어 절단면·유선·글리프가 모두 살아납니다. 대칭면(z 중앙) 속도 컨투어,
차체를 감는 유선, 앞바퀴 뒤 횡단면의 속도 벡터가 이 케이스가 보여주려는
세 가지입니다.

Open `racecar_case` in the Studio's 3D viewer and the block hiding inside the
cut-cell mesh is recovered, so slices, streamlines and glyphs all work. The three
things this case is for: a velocity contour on the symmetry plane, streamlines
wrapping the body, and velocity vectors on a cross-section behind the front axle.

## 형상은 어디서 왔는가 / Where the geometry came from

`racecar.stl`은 내려받은 것이 아니라 `tools/racecar_stl.py`가 매개변수로 만든
자체 형상입니다 — 삼각형 1,040개, 축퇴 0개, 닫힌 다양체. 이 저장소가
"GPL 소스를 참조하지 않았다"는 주장을 시험 가능하게 유지하려면 형상도 출처가
분명해야 하기 때문입니다. 크기를 바꾸려면:

`racecar.stl` was not downloaded. `tools/racecar_stl.py` generates it
parametrically — 1,040 triangles, no degenerate ones, a closed manifold — because
a geometry of unclear provenance would undo the point of a repository whose
licence claim is testable. To change the size:

```powershell
python tools\racecar_stl.py cases\racecar.stl 0.66 0.10 0.5 0.0015
#                                             길이 노즈x  중심z  지면y
```

컷셀 분류(SPEC-LIT §23.2)는 **닫힌 다양체**를 요구합니다. 직접 만든 STL을 쓸
때 `non-manifold edge(s)`로 거절당한다면, 맞닿은 부품이 면을 공유하고 있다는
뜻이므로 서로 겹치게 만드십시오.

Cut-cell classification (SPEC-LIT §23.2) needs a **closed manifold**. If your own
STL is rejected with `non-manifold edge(s)`, two parts are meeting face-to-face;
overlap them instead.
