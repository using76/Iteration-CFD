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

# 2. 운동량까지 풉니다. 결과는 racecar_case\1\ 에 OpenFOAM ASCII 로 쓰입니다.
ofgpu-lowmach racecar_case -iters 3000 -check 250 -output foam
```

1단계의 형상 분류는 모든 코어에서 병렬로 돕니다 — 32코어에서 측정한 경주차
96³ 조각이 559초에서 48초로 줄었습니다(약 11.6배). 이 샘플 크기인 128³은 수
분입니다. 2단계는 GPU에 상주하는 루프이고, 압력 방정식까지 푸느라 **1–2분**
입니다.

Step 1's inside/outside classification now runs on every core: measured on 32
cores, a 96-cube carve of this car went from 559 s to 48 s — about 11.6x. The
128-cube this sample builds takes minutes. Step 2 is the GPU-resident loop and
takes **a minute or two** — it solves a pressure equation as well as momentum.

### `0/p`와 `0/T`는 생성기가 쓴다 / The generator writes `0/p` and `0/T`

메쉬 생성기가 `p`와 `T`를 `0/`에 직접 쓰므로 예전의 복사 단계는 필요 없어졌습니다.
생성되는 내용은 `racecar.fields/`에 담아 두던 것과 같습니다 — 둘 다 균일장이고,
내용이라 할 것은 경계조건뿐입니다. 출구에서 `p = 0`으로 압력을 고정하고 나머지는
zeroGradient, 온도는 293.15 K 등온입니다. `racecar.fields/`는 그 시절 원본이
참고용으로 남아 있을 뿐, 레시피에는 들어가지 않습니다.

The mesh generator writes `p` and `T` into `0/` itself, so the copy step is gone.
What it writes is what `racecar.fields/` used to hold — both uniform, and their
boundary conditions are the whole content: `p = 0` fixed at the outlet to
reference the pressure, zero-gradient elsewhere, and an isothermal 293.15 K.
`racecar.fields/` stays in the folder as that era's originals, for reference
only; the recipe no longer uses it.

어떤 드라이버로 풀지는 여전히 중요합니다. `ofgpu-k-epsilon`과 `ofgpu-k-omega`는
이름 그대로 난류 두 방정식만 풀고 **속도장은 건드리지 않습니다**(얼린 `U` 위에서
돕니다). 그러니 그 둘로 이 케이스를 돌리면 유선은 직선이고 컨투어는 균일합니다 —
차체 주위 유동이 아니라 초기장을 보고 있는 것입니다. 차체 주위 유동을 보려면
`U`와 `p`를 함께 푸는 `ofgpu-lowmach`로 돌리십시오.

Which driver solves what still matters. `ofgpu-k-epsilon` and `ofgpu-k-omega` are
exactly what their names say — two turbulence equations on a velocity field they
never touch. Run this case with either and the streamlines come out straight and
the contour flat, because you are looking at the initial field, not at flow around
a car. To see flow around the car, run `ofgpu-lowmach`, which solves `U` and `p`
together.

## 무엇을 보게 되는가 / What you get to look at

Studio의 3D 뷰어에서 `racecar_case`를 열면 컷셀 격자 속에 숨어 있던 블록이
복원되어 절단면·유선·글리프가 모두 살아납니다. 대칭면(z 중앙) 속도 컨투어,
차체를 감는 유선, 앞바퀴 뒤 횡단면의 속도 벡터가 이 케이스가 보여주려는
세 가지입니다. 그 셋은 모두 `U`를 읽으므로, 2단계를 `ofgpu-lowmach`로
돌렸을 때에만 의미가 있습니다.

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
