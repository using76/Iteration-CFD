# Tranche 18 — the follow-ups plan (docs/18, adopted 2026-10-25)

Planned by Opus 5.5 on 2026-10-25 on branch `feat/t18` at `120eaff` (made from `feat/t17`; `origin/main` is
`7770bb8` and holds all of tranche 17). Every open row of sections D and E of `docs/17-review-later.md` was read and
classified against the tree with `grep`, by reading the code, and in one case by running the failing test. The
rows that were already true are marked 해결 in that file in the same commit as this plan.

---

## 1. The user's order, verbatim

> 좋아 그럼 이제 이폴더에서 작업을 이어가자 opus 5.5 로 리뷰및 계획 glm 5.3flash로 코딩하자

For the scope, the user chose **"D 후속 작업부터 (권장)"**: start with section D (연결·기능이 남은 것) of
`docs/17-review-later.md`.

Working rule from the same order: Opus 5.5 plans, writes contract-only briefs and reviews; GLM-5.3-Flash writes every
implementation through `tools/glm-code.sh`. Only the one unit marked `coder: opus-only` in §6 is written by Opus,
and its row says why.

## 2. Scope

| in scope now | later tranches | out of scope |
|---|---|---|
| Every open row of sections D and E of `docs/17-review-later.md` (§3). The CODE rows become units (§6, §7); the USER rows go to the user in one list (§8); the LONG rows wait for their run or input | Sections A (misses), B (deferred checks) and C (model choices) stay in `docs/17-review-later.md` for later tranches. A unit that closes an A/B/C row on the way marks it, but no unit is planned for them here | The rigid-body code (`src/rigid_body*`, `cuda/rigidbody.cu`, the `-rigidBody` paths of `src/bin/lowmach.rs`) and branch `feat/rigid` (not merged); FSI and the CAD loop (tranche-17 exclusions); MS-MPI and a second GPU (deferred by the user) |

## 3. Classification of every open D and E row

Each row is referenced by its section, its line number in `docs/17-review-later.md` at `120eaff`, and its first
words (D bullets have no ids). Line 379 holds three asks and is split into 379a/b/c. The classes:

- **CODE**: a coder can build it now, with no user decision and no numerics change, and verify it with commands
  under 10 minutes.
- **USER**: it needs a user decision (numerics, a gate or tolerance, licence, a verdict-changing download, hardware,
  deletion, scope).
- **LONG**: it needs a long GPU run, MS-MPI, multi-GPU or a missing external input.
- **DONE**: it is already true in the tree. The evidence is cited, and the row is marked 해결 in place.
- **DUP**: it is the same ask as another row, which is named.
- **RIGID**: it touches the rigid-body code.

Rows already 해결 at `120eaff` are not listed: D 335, 338, 345, 348, 355, 362, 363, 366, 372, 373, 377, 382, 383, 387,
389, 392, 396, 397, 407 and E 423, 424, 425.

| ref (section:line at 120eaff) | first words | class | unit, evidence or reason |
|---|---|---|---|
| D:318 | BC-06, BC-09, BC-14, BC-15, BC-17의 경계 이름: 모든 드라이버가 | **CODE** | T18-SOLV-01 (BC-14 turbulentInlet, BC-16 synthetic, BC-17 jumps), T18-SOLV-02 (BC-06 timed Function1, BC-15 tvMapped), T18-SOLV-03 (BC-09 advective) |
| D:319 | BC-14 mapped는 lowmach/buoyant의 U, p만 연결돼 있다. | **CODE** | T18-SOLV-03 (mapped on T and the turbulence fields) |
| D:320 | turbulentInlet은 transient 경로가 필요하다. | **DUP** | of D:343 (the same transient turbulentInlet path), T18-SOLV-01 |
| D:321 | BC-10, BC-11: 드라이버 연결이 제한적이다. | **DONE** | lowmach.rs:2443-2465 and buoyant.rs:2126-2227 pass pressure_inlet/patch_means/mapped (BC-10) and buoyant p_rgh prgh (BC-11); vof.rs:575 prgh; no other driver runs `Simple::correct_outer` on user patch names |
| D:322 | BC-12: lowmach end-to-end 테스트가 없다 (BC-22 몫). | **CODE** | T18-SOLV-04 |
| D:323 | AUT-15: y+를 계산하는 드라이버가 ofgpu-lowmach 하나뿐이다. | **CODE** | T18-SOLV-05 |
| D:324 | DST-05 → DST-06: FlowState에 μ(T)가 필요하다. | **USER** | a temperature-dependent viscosity in `FlowState` changes the DST gate results (model and numerics choice) |
| D:325 | AMG-04c/07b: 케이스에서 FCG, K-cycle, SPMD GAMG를 고를 수 없다. | **CODE** | T18-SOLV-07 (case keys: FCG, K-cycle) and T18-PAR-02 (DistGamg in ofgpu-par) |
| D:326 | AMG-07a: build_spmd 안에서 한 rank만 SpmdHalo::from_level에 실패하면 나머지 rank가 | **CODE** | T18-PAR-01 |
| D:327 | PAR-17: ofgpu-par가 rank별 잔차를 출력하지 않으므로, 실행 패널은 rank별 | **CODE** | T18-PAR-01 (the per-rank lines) and T18-GUI-11 (the run panel) |
| D:328 | STU-06b: 메시 초안에서 텍스트 필드를 비울 수 없다. 한 | **CODE** | T18-GUI-01 |
| D:329 | STU-07a: gui의 patchRules 매칭은 ^...$로 고정되지만, 솔버(resolve_patch_rule)는 고정하… | **CODE** | T18-GUI-02 (the gui follows the solver, TAKEN) |
| D:330 | STU-07b: set_patch는 명령 한 번에 필드 하나만 쓴다. T를 | **CODE** | T18-GUI-02 |
| D:331 | STU-13: 볼록하지 않거나 뒤틀린 셀은 nearest-centre로 답한다. snappy 계열 | **CODE** | T18-GUI-09 |
| D:332 | STU-17: boolean 결과 파일을 덮어써도, 이미 열린 탭은 다시 | **CODE** | T18-GUI-10 |
| D:333 | VAL-70: README와 GUIDEBOOK이 splash 기준을 "실험과 비교하지 않았다"고 적고 | **CODE** | T18-DOC-01 (opus-only); README.md:425, README.en.md:435, GUIDEBOOK.en.md:1100/1113 still say 'not transcribed' |
| D:334 | VAL-30: JSONC cartesian 메시는 축마다 GradedAxis가 하나라서 내부의 앞전 | **CODE** | T18-MESH-01 |
| D:336 | TET-07a: 매 pass가 Delaunay를 처음부터 다시 만든다 (Delaunay::insert가 비공개였다). | **USER** | recover.rs:513/1278 still rebuild per pass; incremental insertion may change the mesh at degenerate (cospherical) ties - a mesher numerics decision |
| D:337 | AUT-05: gui 카드 템플릿(templates.json)에 R-SNAP 항목이 없어 AUT-05 이후 | **CODE** | T18-GUI-07 (R-ECON and R-TE are missing too) |
| D:339 | DST-04: dust_progress가 공유 GPU에서 빌드마다 5~6분 걸린다 (계획 예산 | **CODE** | T18-DUST-01 |
| D:340 | F32-07: validate의 max_abs_diff가 NaN을 버리므로, single에서 다른 행들도 NaN으로 | **CODE** | T18-F32-01 (the NaN half is already true: validate.rs:916-955 `nan_max`; marked in place) |
| D:341 | AMG-04b: PbicgstabBackend 밖의 방정식(ofgpu-vof p_rgh, 압축성 p, 에너지, 난류, | **CODE** | T18-SOLV-06 |
| D:342 | AMX-02: 이 기계에는 AMGX_DLL/AMGX_DIR/PATH 설정이 없다. 모든 AMGX 실행은 | **USER** | machine configuration: AMGX_DLL system-wide or a copied DLL |
| D:343 | BC-16: 어떤 드라이버도 BcInputs::turbulent_inlet을 넘기지 않아 turbulentInlet, turb… | **CODE** | T18-SOLV-01 |
| D:344 | BC-16, BC-21: gui pick-list(registry)에 새 BC 이름 6개가 없다 | **DONE** | BC-22 7c0054f: bcCatalogue.data.ts:839-914 carries all six names; registry/sync holds it to the JSON |
| D:346 | AUT-07: optimise.py의 L4 상자가 R-ECON 이전 L1(l1_of(r_econ=False))을 중심으로 한다… | **LONG** | needs the R-ECON campaign (AUT-12 L-run, about 4 h of GPU) before the surrogate can be refitted |
| D:347 | AUT-11: AM-14 기록 optimise/(PASS, AUC 0.986)는 AUT-11 이전에 gate돼 | **LONG** | follows the AUT-12 L-run (a new gate directory); the optimiser ships disabled |
| D:349 | STU-15: tet 100 000개 넘게 지나거나 부피 0인 tet(중심에 | **CODE** | T18-GUI-09 |
| D:350 | gui registry: MESH_KINDS에 VAL-30의 plate가 없어 studio에서 평판 preset을 | **CODE** | T18-GUI-01 (registry.ts:122 has no 'plate') |
| D:351 | F32-14: gate verdict 줄(SPEC-LIT 99 보고)은 single에서도 f64 조건을 | **CODE** | T18-VAL-02 (the adopted D-F32-1 class rule, `precision::roundoff_tol`, identity in f64) |
| D:352 | F32-13/14: lane M의 W4 VAL 단위가 F32-13/14 이전 트리에서 | **CODE** | T18-VAL-01 |
| D:353 | AMG-06b: JSONC의 JsonGamg(deny_unknown_fields)에 precision이 없어서 JSONC ga… | **CODE** | T18-SOLV-07 |
| D:354 | AMG-06b: f32 GAMG 사본은 V-cycle만 돈다 (set_cycle(K)는 이름으로 거부). | **USER** | new numerics scope for the opt-in f32 preconditioner (K-cycle, DistGamg) |
| D:356 | TET-11/12: §92.3을 HostGeom64에서 평가하도록 바꾸기 전까지 single에서는 tet 경로가 | **USER** | A28/A87/A132: evaluating the section 92.3 sums in `HostGeom64` is a gate-evaluation change the row itself gives to the user |
| D:357 | VAL-23: cyclic 패치에 wall function 이름이 붙는 버그(A39)는 faces_where가 | **USER** | A39: fixing `faces_where` changes existing results; held by the row |
| D:358 | VAL-33: Gamma onset과 LM 난류 C_f는 네 번째(더 고운) | **LONG** | a fourth, finer level or a fine run that meets the stop rule (long GPU run) |
| D:359 | VAL-52: gui registry의 ofgpu-generate-mesh에 -inflow 플래그를 오케스트레이터가 병합 때 | **DUP** | of D:350, T18-GUI-01 |
| D:360 | CMP-29: Sod shock-tube 템플릿이 없다 (JSONC initial이 균일만 표현하므로 | **CODE** | T18-GUI-08 (the Sod part is already 해결 in the row; Mach preset and residualStyle remain) |
| D:361 | validate section 번호: M 레인의 Gate 131-P(69)가 들어오면서 G | **DONE** | docs/validate-run/f64.json: sectionCount 95, titles 70 CMP-SOD .. 95 DST-REMBE; one tree now, no lane renumbering |
| D:364 | CMP-25: SpalartAllmaras::correct_conservative(src/models/spalart_allma… | **CODE** | T18-SOLV-08 |
| D:365 | CMP-25: ofgpu-compressible은 JSONC 케이스의 SA를 physics.turbulence, Spalart… | **CODE** | T18-SOLV-09 |
| D:367 | TET-13b, TET-13c: run_tet의 emit stage JSON에 n_prisms, n_layer_points,… | **CODE** | T18-GUI-10 |
| D:368 | VAL-23 (TET-14가 발견): validate_gates/pipe.rs가 '143.5 MISSED:'로 시작하는 not… | **CODE** | T18-VAL-02 (pipe.rs:518 still writes '143.5 MISSED:') |
| D:369 | CHR-12b (기존 문제를 발견, 고치지 않음): ofgpu-lowmach가 열린 경계가 | **USER** | the row names it a numerics decision (refuse by name or point to -sealed), SPEC-LIT 133.11 |
| D:370 | CMP-18 → CMP-23/24/27: characteristic far-field (126.8)은 물리적 c를 쓰므로, | **USER** | A64's 1.186 c_0 is a numerics finding; any cure changes the far-field |
| D:371 | CMP-28: physics.compressible 케이스를 다른 드라이버(예: ofgpu-lowmach)에 주면 거부하지 않… | **CODE** | T18-SOLV-06 (common/mod.rs has no refuse_compressible) |
| D:374 | F32-16 병합 규칙: §112.3의 굵은 숫자는 카운트 테스트(types::tests::the_f32_failure_cou… | **DONE** | one tree, no more lane merges; the count test types.rs:688 holds SPEC-LIT 112.3 (library 254, binary 9, equal to a grep of the attribute) |
| D:375 | CHR-14/CHR-15/CHR-16: ofgpu-lowmach에서 TCI 닫힘 중 split step에 연결된 것은 | **USER** | the wiring has to choose the EDM and PaSR forms (model and numerics choice) |
| D:376 | CHR-28 → G 레인: gui/server/src/prompts/system.ts의 반응 줄(88행)이 아직 'EDM, | **CODE** | T18-GUI-06 |
| D:378 | CHR-23 → CHR-23b (제안): ofgpu-buoyant의 P1/fvDOM 연결. ScalarTransport에 im… | **USER** | a new physics path in the Boussinesq driver (implicit sink in ScalarTransport); today refused with a pointer to ofgpu-lowmach, SPEC-LIT 137.8 |
| D:379a | CHR-26 후속 (만들지 않음, SPEC-LIT 137.10): ofgpu-lowmach의 constant/cloudProp… | **CODE** | T18-CHEM-01 (the constant/cloudProperties dictionary route) |
| D:379b | CHR-26 후속 (만들지 않음, SPEC-LIT 137.10): ofgpu-lowmach의 constant/cloudProp… | **USER** | scope: parcels in steady, restart and moving-mesh runs, and evaporation into a case without species |
| D:379c | CHR-26 후속 (만들지 않음, SPEC-LIT 137.10): ofgpu-lowmach의 constant/cloudProp… | **USER** | numerics: the (77.2) mixed vapour c_p and the (77.1) dilution of A116 change CHR-26's results |
| D:380 | CHR-13 → 이후 반응 작업: ofgpu-lowmach가 split step 뒤 | **DONE** | a record of behaviour already in the tree: lowmach_chem/mod.rs:708 `refused_cells` |
| D:381 | CHR-16 → S, M 레인 (다음 병합): case_json.rs의 ScalarBc와 | **DONE** | API notice for lane merges; all lanes merged (fa150be, main 7770bb8) and there is one tree |
| D:384 | F32-15 ↔ F32-16: 이 병합으로 M 레인 F32-16 72281f9가 | **DONE** | a record: the ALE single failures carry the K attribute and are inside 112.3's count; tests/f32_unfinished.rs keeps the values |
| D:385 | F32-17 → rigid-body 브랜치: single에서 ofgpu-lowmach의 rigid-body v2 테스트 | **RIGID** | the rigid-body branch's tests; out of scope (feat/rigid not merged) |
| D:386 | F32-17 (기존 race, 두 빌드): ofgpu-datacentre의 the_shipped_dc_schema_is_the… | **CODE** | T18-TEST-01 |
| D:388 | F32-17 → 사용자: 문턱이 f32 반올림보다 낮은 production 거부 | **USER** | A108: loosening a production threshold is the user's |
| D:390 | F32-18: validate section 80과 81은 validate 실행 파일 옆의 | **CODE** | T18-DOC-01 (a GUIDEBOOK sentence; the 'not built' skip is by design) |
| D:391 | PAR-10c → 이후 PAR 단위: decomposed Simple은 upwind/central이 아닌 | **USER** | scope: extending the decomposed SIMPLE beyond its refusal list (multi-GPU deferred) |
| D:393 | PAR-11, PAR-12 → 다른 레인 (API, 다음 병합): distflow::FlowExchange에 | **DONE** | API notice for lane merges; one tree now |
| D:394 | PAR-15 후속 (만들지 않음): restart 집합은 층류 비압축 SIMPLE | **USER** | scope: decomposed restart of turbulence, thermal and transient levels (multi-GPU deferred) |
| D:395 | PAR-15 (기존 출력, 고치지 않음): ofgpu-par -solve restart -quiet가 | **CODE** | T18-PAR-01 |
| D:398 | G W9 → S, M 레인 (API, 다음 병합; | **DONE** | API notice for lane merges; one tree now |
| D:399 | G W9 → G 레인 gui: docs/schema/case-1.json에 dust.cloud.igniter (DST-17), | **CODE** | T18-GUI-06 |
| D:400 | DST-15, DST-16 후속 (만들지 않음, SPEC-LIT 159.10, 159.11): Phase | **USER** | both halves change numerics (device gather order, the char step end) |
| D:401 | 병합 bf30278이 발견 (SPEC-LIT, 기존 상태): '## 109.' (block-coupled | **CODE** | T18-DOC-02 |
| D:402 | 병합 bf30278이 발견 (기존 경고, W9 파일 아님): single의 | **CODE** | T18-TEST-01 |
| D:403 | PAR-13a, PAR-13b, PAR-14 → 다른 레인 (API, 다음 병합; | **DONE** | API notice for lane merges; one tree now |
| D:404 | PAR-13a, PAR-13b 후속 (만들지 않음, SPEC-LIT 139.8, 139.9): least-squares와 | **USER** | scope: gradients, compressible/vof face sums and a turbulence bitwise gate on the decomposed path (multi-GPU deferred) |
| D:405 | DST-11 → 다른 레인 (API, 다음 병합; 병합 fa150be에서는 | **DONE** | API notice for lane merges; one tree now |
| D:406 | DST-11 → 다른 드라이버: ofgpu-compressible만 64 MiB worker thread | **DONE** | only ofgpu-compressible opens a vent (compressible.rs:2005 DRIVER_STACK); every other driver refuses it (common/mod.rs:163) |
| D:408 | G W10 (GUI-BC-FP 7988c19) 후속: web gui/web/src/components/side/Boundary… | **CODE** | T18-GUI-03 |
| D:409 | G W10 (F32-19b f2f01f6) 후속 정리: gui/server/src/runs/dispatch.ts의 runHea… | **CODE** | T18-GUI-04 |
| D:410 | G W11 (GUI-PAR-REG cdfb1bd) 후속 (만들지 않음): gui는 ofgpu-par의 | **CODE** | T18-GUI-04 |
| D:411 | G W11 (GUI-POLYMESH 349d34e) 후속 (만들지 않음): polymesh .jsonc | **CODE** | T18-GUI-05 |
| D:412 | G W11 (GUI-POLYMESH 349d34e) 후속: polymesh 케이스의 gui case | **CODE** | T18-GUI-05 |
| D:413 | G W11 (GUI-POLYMESH 349d34e, 기존 동작, 고치지 않음): mock | **CODE** | T18-GUI-05 |
| D:414 | G W11 (GUI-DCSEED 9642c6c) 후속: gui/shared/src/ontology/seed/dc.capabil… | **CODE** | T18-TOOL-01 |
| D:415 | VAL-82 (02201f2) → validate.rs (Rust 변경, 만들지 않음): gate | **CODE** | T18-VAL-02 |
| D:416 | VAL-82 (02201f2) → README 내용 정리: README.en.md의 한계 bullet | **CODE** | T18-DOC-01 (opus-only) |
| D:417 | VAL-82 (02201f2) → tools: tools/validate_record.py selftest에 한 파일에 mar… | **CODE** | T18-TOOL-02 |
| D:418 | VAL-82 (02201f2)가 발견, brief 작성 규칙: 한국어 굵은 글씨를 | **DONE** | the rule is in this plan's section 5 |
| D:419 | coder 파이프라인 (VAL-82가 발견): tools/glm-code.sh로 띄운 GLM-5.3-Flash child가 2… | **DONE** | 120eaff: tools/glm-code.sh passes --tools; probe 2026-10-25 on CLI 2.1.310 gets all six tools |
| E:426 | import.test의 221-runs 케이스 (2026-10-20, W9 병합 bf30278과 S W6 | **USER** | the failure is now 'Mesh inserted 4, expected 0' (import.test.ts:315) from ignored runtime .meshSummary.json files under cases/; deleting them is the user's |

Notes on the classification:

- **D:318 is split by family.** The tree has every driver hook these names need. `Simple::begin_time_step`
  (simple.rs:874-881) advances turbulentInlet and the synthetic inlets. `Simple::correct_outer` refreshes mapped
  inlets (simple.rs:1073) and assembles cyclic jumps (simple.rs:1166-1174, 1336-1341). `setup_*_field_timed`
  (field_setup.rs:5690, 5726) returns the `TimedBcs` a driver applies after `F1Clock::advance`. What is missing is
  the opt-ins: no driver passes `BcInputs::turbulent_inlet`, `jumps` or `advective`, and none calls the `_timed`
  setup (SPEC-LIT 120.1 says "Wiring the drivers is a follow-up unit"). Cases that do not use these names keep
  every number. The work is wiring, not numerics.
- **D:329 is decided by recommendation (TAKEN): the gui follows the solver.** The solver's `resolve_patch_rule`
  (case_json.rs:3074) is what runs, and it searches without anchors. The gui (`casejsonc.ts:114`) anchors with
  `^(?:...)$`. Anchoring the solver instead would change how existing cases lower.
- **E:426 has a new cause.** Run on 2026-10-25, the failing assertion is `Mesh inserted toBe(0)`, which received 4
  (import.test.ts:315). Three of the four are ignored runtime files written by studio runs on 2026-10-15:
  `cases/channel/constant/polyMesh/.meshSummary.json` and `.meshSummary.json` under
  `cases/e2e-control/mesh-cartesian` and `cases/e2e-control/mesh-sphere-in-box`. The `runs/` tree is present (267
  entries), so the old cause ("no runs/ dir") no longer holds. Removing the files is a deletion, so it is the user's
  call (§8).
- **USER rows the tree confirmed:**
  - D:336: `recover.rs:513` and `:1278` still rebuild the Delaunay each pass. Only `refine.rs` uses `insert_point`.
  - D:356: A28, A87 and A132 give the §92.3 `HostGeom64` change to the user.
  - D:378: `ofgpu-buoyant` refuses a participating medium and points to `ofgpu-lowmach` (buoyant.rs:2805-2812,
    SPEC-LIT 137.8).

## 4. Conventions

### 4.1 Unit ids and the tree

- Unit ids are `T18-<AREA>-NN`. AREA is one of GUI, TOOL, TEST, DOC, VAL, SOLV, PAR, MESH, F32, CHEM, DUST.
- **One tree.** Every unit runs in `C:/Users/sdd32/Documents/GitHub/Iteration-CFD` on `feat/t18`, one unit after
  another. The -S, -M and -R worktrees were removed on the user's order (disk space), and no new worktree is made.
  A wave is one workflow run (`t18-wave.js`). Its units run in the order §7 lists them, each from the tree the
  previous unit left clean.
- No lane merges. The tranche-17 merge rules (§7.3 of docs/17: SPEC-LIT order, unions, recounts) apply only in the
  sense that each unit leaves the tree consistent: it recomputes the counts it changes through their audit tests,
  regenerates `docs/schema/*.json` through the solver's own write test, and keeps SPEC-LIT sections in number order.
- **SPEC-LIT.** A unit that must state something writes it in place in the section that owns the subject:
  - the next free subsection, e.g. a "which drivers advance the clock" sentence in 120.1, or the case keys of
    T18-SOLV-07 as a new 141.x;
  - or a sentence in an existing paragraph that is now stale.
  - No new top-level section is planned. §162 stays spare, and §163 onward is free if a unit needs one; it says so.
- **dc.seed anchors.** Any Rust edit that moves an anchored line breaks the shared `dc.seed.test`. From T18-TOOL-01
  on (the first unit of W2), every Rust unit runs its re-anchor tool before committing. Before that, no Rust unit runs
  (W1 has none).

### 4.2 The review-later file

- A unit marks the `docs/17-review-later.md` rows it closes, in place and in that file's Korean prose:
  `**해결 (<date>, T18 <ID> <commit>)**: <what is now true>`. It never renumbers or deletes a row.
- A row a unit closes only in part gets a dated note instead of 해결.
- New findings are returned as `REVIEW-LATER:` lines. The **orchestrator** writes them as new rows, continuing
  **A137, B54, C96** and new D bullets. Units mark only their own rows.
- This plan's commit marks the DONE rows of §3.

### 4.3 Verify blocks (each command under 10 minutes; judge by test COUNTS against HEAD)

**V-RUST**, in Bash. Run the builds one after another, never in parallel.

```
export PATH="$HOME/.cargo/bin:$PATH"; cd C:/Users/sdd32/Documents/GitHub/Iteration-CFD/rust; export CARGO_BUILD_JOBS=2
cargo build --release                                                          # f64, full output kept (no `| tail` on a no-warning gate)
cargo build --release --lib --features single --target-dir target/single       # single
cargo test --release --lib --no-run                                            # BUILD step, not a verification command: after a SPEC-LIT edit it takes ~11 min; run it in the background with visible progress
cargo test --release --lib -- <the unit's filters>                             # the unit's tests, counts stated
cargo test --release --features single --target-dir target/single --lib -- <the unit's filters>
cargo test --release --lib -- provenance_audit notice xref types::tests capture::registry spec_lit
```

Bin-target tests need `--bin ofgpu-<driver>`, and integration tests need `--test <name>`. A lib filter that names
a bin test runs 0 tests and passes. The single lib test target compiles (since `115fb5a`). The schema write and
drift test pairs race when run in parallel (D:386): until T18-TEST-01 lands, run the drift tests alone. A unit that
touches a `.cu` file, or claims f64 bitwise identity across a kernel path, also runs `tools/f64_identity.py`. For
a kernel change that is `sass` before and after, then `sass-diff`. For validate rows it is `rows` on the touched
`-sections`.

**V-GUI**:

```
cd C:/Users/sdd32/Documents/GitHub/Iteration-CFD/gui/shared && npx vitest run      # baseline 120eaff: 156 passed (14 files)
cd ../server && npx vitest run                                                       # baseline: 1008 passed, 1 failed, 1 skipped (1010)
cd ../web && npx vitest run                                                          # baseline: 296 passed (46 files)
cd .. && npm run typecheck
```

The one allowed failure is server `src/ontology/import.test.ts` "the real tree folds to at least 221 runs …", by
name (E:426). Any other failure is the unit's to explain. A unit that moves a solver CLI flag also runs server
`registry/sync.test`.

**V-PY**: `python tools/validate_record.py selftest` and `python tools/validate_record.py check`. Both must pass,
with the selftest case count stated.

## 5. Rules

Tranche 18 inherits `docs/17-completion-plan.md` §5 in full:

- §5.1, single precision IMPORTANT: both builds compile, f64 bitwise unless a unit says otherwise, every new row
  and tolerance has a precision class P/A/D/R, and f64 is used where round-off is the subject.
- §5.2, BCs: a §4 triple rewrite, Function1 through BC-06, names refused with a reason.
- §5.3, gates: never loosened. Gate code lives in `validate_gates/`. A gate with no reference is NAMED and NOT RUN.
  A numerics change is the user's.
- §5.4, licence and provenance: no GPL/LGPL/AGPL source. House headers and PROVENANCE rows for new files, and
  NOTICE/PROVENANCE quote the real source-file count.
- §5.5, process: a bug is proven by a failing test first, and verification is ≤ 10 minutes per command.
- §5.6, the DUST clean-room rule: equations only from the open papers of SPEC-LIT 151-160 and
  `docs/17-dust/RESEARCH.md`. Never open FSCDustFoam or OpenFOAM source.

The standing memory rules hold:

- A missed gate or an unconverged solve does not stop a unit: it is asserted and printed as MISSED, the unit lands,
  and the miss is logged.
- Decide by recommendation and record it as TAKEN.
- Decided items get no more tests.
- Push, merge, rebase, deletion, gate loosening and numerics changes are the user's.
- Never `git add -A`: stage by name, never `.claude/skills/typesafe-ai/`, `.zcode/`,
  `f1_2026_concept_polygon_model.glb` or `_archive/`.
- Never touch ports 8787/5173/5180. Kill by PID or exact name only. Never write under `C:/Users/sdd32/.claude`.

What tranche 18 adds:

1. **The GLM tool fix and its probe.** Since about 17:02 on 2026-10-20, the GLM child started by `claude -p` had
   no Bash, Read, Edit, Write, Glob or Grep (D:419). `tools/glm-code.sh` now passes `--tools "$GLM_TOOLS"` (default
   `Bash,Read,Edit,Write,Glob,Grep`; `120eaff`), and a probe on 2026-10-25 with CLI 2.1.310 got all six back.
   Before the first unit of each wave, the supervisor runs a one-line probe ("run `git log -1 --oneline` with
   Bash and print it"). If the probe or a unit reports missing tools, the unit stops as failed with
   `coder_unavailable`, and the orchestrator re-codes it with Sonnet 5.5 (the user's 2026-10-19 rule). Opus never
   writes it instead.
2. **Dispatch.** From the tree, with `env -u ZAI_API_KEY CLAUDE_CODE_DISABLE_AUTO_MEMORY=1
   GLM_PERMISSION_MODE=acceptEdits`. The run goes to the background, and the supervisor waits in the foreground for
   the log's `exit` line. Tool inputs are at most 50 lines per call. If a log shows 429 / 1310 / 1311, the wave
   stops (quota).
3. **Brief-writing rule (D:418).** In Korean Markdown, never close bold right after a code span and before Hangul
   (`` **`X`**인 ``), because CommonMark does not close it there. Put the particle inside the bold:
   `` **`X`인** ``.
4. **The gui follows the solver.** Where the studio re-implements a solver rule (patch matching, refusals, flag
   order), the solver's behaviour is the contract. A gui unit cites the solver line it mirrors and changes no Rust.
5. **Generated files stay generated.** `docs/schema/*.json`, `gui/shared/src/bcCatalogue.data.ts`, and the README
   and GUIDEBOOK blocks between `tools/validate_record.py` markers are rewritten only by their generators.
6. **Single-tree hygiene.** A unit leaves `git status` showing only the three untracked items above. A unit that
   cannot land saves its work on `wip/<ID>` and returns `feat/t18` to a clean tree.

## 6. The units

### 6.1 Summary

| id | what is TRUE after it | closes | size | coder | deps |
|---|---|---|---|---|---|
| T18-GUI-01 | Mesh dialog text fields can be cleared (`-check` goes away); studio offers the `plate` preset | D:328, D:350 (D:359) | S | glm | — |
| T18-GUI-02 | studio patch-rule matching is the solver's; `set_patch` writes several fields in one edit | D:329, D:330 | M | glm | — |
| T18-GUI-03 | the web Boundary editor edits fixedProfile (direction, origin, profile) | D:408 | S | glm | — |
| T18-GUI-04 | studio launches `ofgpu-par -solve` / `-bench` modes; the dead preview flag is gone | D:409, D:410 | S | glm | — |
| T18-GUI-06 | the studio prompt and case-block list match case-1.json (dust, vent, initial.regions, parcels, EDC/EDM/PaSR) | D:376, D:399 | S | glm | — |
| T18-GUI-07 | every autonomy rule id has an explanation template (R-SNAP, R-ECON, R-TE added) | D:337 | S | glm | — |
| T18-TOOL-02 | `validate_record.py selftest` renders two blocks into one text and checks it | D:417 | S | glm | — |
| T18-TOOL-01 | one command re-anchors the dc.seed capabilities and their gold copies from their probe symbols | D:414 | S | glm | — |
| T18-TEST-01 | schema-writing tests write atomically (no race); two single-build test warnings gone | D:386, D:402 | S | glm | TOOL-01 |
| T18-DOC-02 | SPEC-LIT §109 sits in number order | D:401 | S | glm | — |
| T18-VAL-02 | single verdict lines follow their rows' class; the gate registry comment is true; the pipe note's verdict word is lower case | D:351, D:368, D:415 | S | glm | TOOL-01 |
| T18-SOLV-06 | every non-compressible driver refuses `physics.compressible`/`initial.regions`; GAMG on an unsupported equation is refused at read time | D:341, D:371 | M | glm | TOOL-01 |
| T18-PAR-01 | a failed SPMD halo build fails on every rank at once; `ofgpu-par` prints per-rank residuals; `-quiet` is quiet | D:326, D:327 (solver), D:395 | M | glm | TOOL-01 |
| T18-SOLV-07 | cases select FCG, the K-cycle and the f32 hierarchy copy (JSONC `gamg.precision`) | D:325 (part), D:353 | M | glm | TOOL-01 |
| T18-SOLV-01 | transient lowmach/buoyant run turbulentInlet, the digital-filter and DFSEM inlets, and cyclic jumps | D:318 (part), D:343 (D:320) | M | glm | W2 |
| T18-SOLV-02 | transient lowmach/buoyant advance time-varying Function1 and timeVaryingMappedFixedValue patches | D:318 (part) | M | glm | SOLV-01 |
| T18-SOLV-03 | lowmach/buoyant run the advective outlet; mapped inlets refresh T and the turbulence fields | D:318 (part), D:319 | M | glm | SOLV-02 |
| T18-SOLV-04 | an ofgpu-lowmach end-to-end test holds the thermal family against closed forms | D:322 | S | glm | — |
| T18-SOLV-05 | ofgpu-buoyant writes yPlus.json from the shared y+ helper | D:323 | M | glm | — |
| T18-F32-01 | CHT region residuals are formed in f64 under single | D:340 (rest) | S | glm | — |
| T18-SOLV-08 | SA's conservative correction calls one RasCore entry instead of copying it | D:364 | S | glm | — |
| T18-SOLV-09 | ofgpu-compressible runs a JSONC Spalart-Allmaras case (`initial.nuTilda`) | D:365 | M | glm | SOLV-08 |
| T18-MESH-01 | JSONC cartesian axes can be graded in segments | D:334 | M | glm | — |
| T18-CHEM-01 | ofgpu-lowmach reads parcels from `constant/cloudProperties` | D:379a | M | glm | — |
| T18-GUI-05 | polymesh .jsonc cases list their patches read-only; the gui checks the file-level refusals; the mock sizes its box from the mesh | D:411, D:412, D:413 | M | glm | — |
| T18-GUI-08 | compressible runs plot Mach monitors apart from residuals; results open coloured by Mach | D:360 | M | glm | — |
| T18-GUI-11 | the run panel shows ofgpu-par's per-rank residuals | D:327 (gui) | S | glm | PAR-01 |
| T18-VAL-01 | every validate row carries a precision class; worst-of folds propagate NaN | D:352 | M | glm | VAL-02 |
| T18-PAR-02 | ofgpu-par preconditions with the distributed GAMG when the case asks for GAMG | D:325 (part) | M | glm | SOLV-07, PAR-01 |
| T18-GUI-09 | probes locate points in non-convex cells; streamlines survive an exit and re-entry | D:331, D:349 | M | glm | — |
| T18-GUI-10 | an open tab reloads an overwritten boolean result; the automesher stage counters show | D:332, D:367 | S | glm | — |
| T18-DUST-01 | dust_progress shares its host transfers and runs faster, with the same asserts | D:339 | S | glm | — |
| T18-DOC-01 | README and GUIDEBOOK prose match the tree (limits, Gate 78-D, validate siblings) | D:333, D:390, D:416 | M | **opus-only** | all others |

### 6.2 The rows in full

Every row states purpose, closes, owned files, contract, Verify, deps and size. The supervisor re-anchors every
line number with `grep -n` before writing the brief. Where the tree disagrees with a row, the tree wins and the
supervisor says so.

#### T18-GUI-01 — Mesh drafts can be cleared; the plate preset

- **TRUE after:** emptying a text field of the Mesh dialog (`outputDir`, `config`, `layoutDir`, `check`) clears
  that draft field, and the automesher argv then has no `-check`. `MESH_KINDS` holds every kind
  `ofgpu-generate-mesh` takes, `plate` among them.
- **Closes:** D:328, D:350, and D:359 as its duplicate.
- **Files:**
  - `gui/web/src/components/side/MeshDialog.tsx` (+ `MeshDialog.test.tsx`)
  - `gui/shared/src/protocol.ts` (the `open_mesh_dialog` command, `MeshDraftSchema`)
  - `gui/server/src/tools/gui.ts` (the draft merge)
  - `gui/shared/src/registry.ts` (`MESH_KINDS`, line 122)
  - a registry test
- **Contract:**
  - `open_mesh_dialog` accepts `null` for each of the four text fields, meaning "clear". An omitted key keeps
    today's "unchanged". `MeshDialog`'s commit of `''` on a field whose draft holds a value sends `null` for it
    (today `''` returns no command, MeshDialog.tsx:58).
  - `MESH_KINDS` = `['channel', 'cavity', 'step', 'big', 'plume', 'room', 'damBreak', 'plate']`. These are the names
    of `CaseKind::from_name` (rust/src/blockgen.rs:3651-3658; `plate` is the usage name). A test reads that match
    block and requires every primary name to be in `MESH_KINDS`.
- **Tests:**
  - a draft `{check: '25'}` cleared in the dialog produces `{type: 'open_mesh_dialog', check: null}`;
  - the server merge of `check: null` leaves a draft without `check`, and the argv built from it has no `-check`;
  - `MESH_KINDS.includes('plate')`.
- **Verify:** V-GUI. Counts are the baseline plus the new tests.
- **Deps:** none. **Size:** S. **Coder:** glm.

#### T18-GUI-02 — Patch rules match as the solver matches; multi-field set_patch

- **TRUE after:**
  - `casejsonc.ts`'s `patchRules` builds `new RegExp(r.match)` without anchors, first rule wins, exactly as
    `resolve_patch_rule` does (rust/src/io/case_json.rs:3074, `Regex::is_match`). Decision TAKEN: the gui follows
    the solver (§3 note).
  - `set_patch` takes `fields: { <field>: { value } | { bc } }` with one or more fields, applied as ONE case_edit.
    So turning a wall into a velocity-inlet on a case that solves T writes U and T together. The single-field form
    `{field, value|bc}` keeps working unchanged.
- **Closes:** D:329, D:330.
- **Files:**
  - `gui/server/src/formats/casejsonc.ts` (+ test)
  - `gui/server/src/tools/guiBoundary.ts` (+ `guiBoundary.test.ts`)
  - `gui/shared/src/protocol.ts` (`set_patch` schema, around line 651)
  - `gui/server/src/agent/prompt.ts` (the one set_patch line)
- **Tests:**
  - rules `[{match: 'wall', kind: 'wall'}, {match: '.*', kind: 'patch'}]` give patch `wallXMin` the type `wall`
    (today `patch`);
  - a rule `set_patch` wrote as `^inlet$` still matches only `inlet`;
  - `set_patch {patch: 'inlet', kind: 'velocity-inlet', fields: {U: {value: [1,0,0]}, T: {value: 300}}}` on a case
    with T is accepted with one approval card that writes both;
  - `fields` naming a field the case does not solve is refused `PATCH_FIELD`, naming it;
  - both `fields` and `field` in one command are refused `PATCH_FIELD`.
- **Verify:** V-GUI. **Deps:** none. **Size:** M. **Coder:** glm.

#### T18-GUI-03 — fixedProfile in the web Boundary editor

- **TRUE after:** when a field's type is `fixedProfile`, `BoundaryEditor.tsx` shows:
  - `direction` (three numbers);
  - `origin` (a number, default 0);
  - `profile` (rows of `[s, value]`; a value is three numbers for U).

  Saving sends the payload `POST /api/case/patch` takes (`PatchBcSchema`: `direction`, `origin`, `profile`). A
  refusal from the server (`refuse_bad_fixed_profile`, SPEC-LIT 120.1) is shown with its code and message.
- **Closes:** D:408.
- **Files:** `gui/web/src/components/side/BoundaryEditor.tsx` (+ its test).
- **Tests:**
  - the three inputs render only for `fixedProfile`;
  - direction `[0,0,1]` and profile `[[0,0],[1,2]]` on T produce exactly
    `{type: 'fixedProfile', direction: [0,0,1], origin: 0, profile: [[0,0],[1,2]]}`;
  - a mocked server refusal is shown by code.
- **Verify:** V-GUI (web + typecheck). **Deps:** none. **Size:** S. **Coder:** glm.

#### T18-GUI-04 — ofgpu-par modes from the studio; the dead preview flag removed

- **TRUE after:**
  - When the args hold `-solve` or `-bench`, `buildArgv` (gui/server/src/runs/dispatch.ts:44) puts that flag and
    its value FIRST and omits the case positional, because par.rs reads a mode only as the first argument and
    neither mode reads a case (par.rs:20-44).
  - Every other binary's argv is unchanged: `[case, ...positionals, ...flags]`.
  - `SINGLE_PRECISION_PREVIEW` (gui/shared/src/protocol.ts) and `runHeaderLine`'s dead branch (dispatch.ts:~112)
    are deleted. The flag has been false since F32-19b `f2f01f6`.
- **Closes:** D:409, D:410.
- **Files:**
  - `gui/server/src/runs/dispatch.ts` (+ `dispatch.test.ts`)
  - `gui/shared/src/protocol.ts`
  - `gui/shared/src/registry.ts` (only the two flag descriptions, if they still say "must come first")
- **Tests:**
  - `buildArgv({casePath: 'c.jsonc', positionals: [], args: [{flag: '-solve', value: 'simple'}, {flag: '-n', value: 4}]})`
    → `['-solve', 'simple', '-n', '4']`;
  - the same with `-bench strong` → `['-bench', 'strong', ...]`;
  - without a mode → today's argv;
  - `grep -r SINGLE_PRECISION_PREVIEW gui/*/src` is empty, and the dispatch.test header expectations are unchanged.
- **Verify:** V-GUI plus server `registry/sync.test`. **Deps:** none. **Size:** S. **Coder:** glm.

#### T18-GUI-06 — The prompt and the case-block list match the schema

- **TRUE after:**
  - `CASE_BLOCK_NAMES` (gui/server/src/registry/schema.ts:175) also lists `dust`, `vent` and `initial`, each with
    the enum choices that `docs/schema/case-1.json` gives its members.
  - `system.ts` (gui/server/src/prompts) says EDC runs on `ofgpu-lowmach` and EDM and PaSR are refused by name
    (SPEC-LIT 137.11). Today line 88 says all three are refused.
  - Its parcels line names `landing` and `injectors[].size`.
  - A dust line names FSC/TFC, the Phase B `dust.cloud` on `ofgpu-lowmach` (laminar combustion only, P1 only,
    SPEC-LIT 159.10) and that `ofgpu-compressible` runs the vented case.
  - A vent line names `vent.probes`.
  - The schema test's `EXPECTED_CHOICES` / `CASE_BLOCK_CHOICES` follow the schema.
- **Closes:** D:376, D:399.
- **Files:**
  - `gui/server/src/prompts/system.ts`
  - `gui/server/src/registry/schema.ts` (+ `schema.test.ts`)
- **Tests:**
  - schema.test "case blocks" passes with the three new blocks, and each choice list equals the schema's enum;
  - a prompt test asserts the EDC/EDM/PaSR sentence and the presence of `vent.probes` and `dust.cloud`.
- **Verify:** V-GUI. **Deps:** none. **Size:** S. **Coder:** glm.

#### T18-GUI-07 — Explanation templates for every autonomy rule

- **TRUE after:**
  - `gui/web/src/assistant/autonomy/templates.json` has a template for each rule id that
    `tools/autonomy/rules.py` defines: R-BUDGET, R-CURV, R-DOM, R-ECON, R-FEAT, R-GAP, R-PLANE, R-SNAP, R-TE,
    R-WIN, R-YP. Today R-ECON, R-SNAP and R-TE are missing.
  - Template wording comes from rules.py's and remedies.py's own descriptions, with no new claims.
- **Closes:** D:337.
- **Files:** `templates.json` and the explain test that loads it.
- **Tests:**
  - a test extracts the `R-[A-Z0-9-]+` ids from rules.py and requires a template for each;
  - explaining the records in `tools/autonomy/fixtures/explain/records_by_id.json` returns no "no template for rule
    id" string.
- **Verify:** V-GUI (web). **Deps:** none. **Size:** S. **Coder:** glm.

#### T18-TOOL-02 — validate_record selftest covers two blocks in one file

- **TRUE after:** `tools/validate_record.py selftest` has a case that renders the status block and the gates block
  into ONE text, runs `check` on it, and fails when a later block's offset is shifted. That is the C95 bug, in
  which `check_text` overwrote prose after the second block.
- **Proof the case bites:** the supervisor runs it against a scratch copy with the C95 fix reverted, where it must
  fail, and records that.
- **Closes:** D:417.
- **Files:** `tools/validate_record.py`.
- **Verify:** V-PY. The selftest count rises by one, and `check` passes on the five real documents.
- **Deps:** none. **Size:** S. **Coder:** glm.

#### T18-TOOL-01 — dc.seed anchors re-anchor themselves

- **TRUE after:** `node gui/shared/scripts/reanchor-dc-seed.mjs [--check]` does the following.
  1. It reads `DC_CAPABILITY_ANCHORS` (gui/shared/src/ontology/seed/dc.types.ts:173: `{file, symbol}` per
     capability).
  2. It finds the one line of `file` that holds `symbol`.
  3. It rewrites the `file:line` anchor in `dc.capabilities.ts` and in the copies in
     `gui/server/src/corpus/gold/spec-lit-52-55.json`.
  4. With `--check` it rewrites nothing and exits 1, listing the stale anchors.
  5. A symbol found on 0 or more than 1 lines is an error naming it, as `dc.seed.test` demands (line 91).
  Run at HEAD, it changes nothing: all 23 anchors are exact (D:414).
- **Closes:** D:414.
- **Files:** the new script (house header), `gui/shared/package.json` (a `reanchor` script entry).
- **Tests:**
  - a vitest case runs the script's function on a temp copy with one anchor shifted by +3 lines and gets it back;
  - `--check` at HEAD exits 0.
- **Verify:** V-GUI (shared, server gold.test).
- **Deps:** none. Runs FIRST in W2, so every later Rust unit can re-anchor. **Size:** S. **Coder:** glm.

#### T18-TEST-01 — Schema tests write atomically; two single-build warnings

- **TRUE after:**
  - Every test or tool that rewrites a shipped schema (`docs/schema/case-1.json`, `cht-1.json`, `dc-1.json`,
    `bc-catalogue-1.json` and the gui `bcCatalogue.data.ts`) writes a sibling temp file and renames it over the
    target, through one helper `write_atomically(path, bytes)`. A drift test running in parallel then reads the
    old or the new file, never an empty one. The drift tests themselves are unchanged.
  - The single lib-test build no longer warns about the unused `r` (wallfunctions.rs ~5038) or the needless `mut`
    (tetmesh/sizefield.rs ~1904).
- **Closes:** D:386, D:402.
- **Files:**
  - the helper (in an existing io module, or a new file with a house header and PROVENANCE row)
  - `src/io/case_json.rs` (~6296)
  - `src/io/case_cht/tests.rs` (~2572)
  - `src/bin/datacentre.rs` (~2540)
  - `src/field/catalogue_json.rs`
  - `src/wallfunctions.rs`, `src/tetmesh/sizefield.rs`
- **Proof first:** a test where a reader thread polls a file 10 000 times while a writer rewrites it 200 times
  with `std::fs::write` sees a 0-byte read (it fails today), and with `write_atomically` it sees none. If the
  plain-write race cannot be shown on this machine in under a minute, the supervisor records that, and the test
  asserts the helper's property only.
- **Verify:**
  - V-RUST with the filters `schema catalogue write_atomically`, run with `--test-threads=8` three times: 0
    failures, counts stated;
  - `--bin ofgpu-datacentre -- schema`;
  - a single lib-test build with no warning in the two files (full output kept).
- **Deps:** T18-TOOL-01. **Size:** S. **Coder:** glm.

#### T18-DOC-02 — SPEC-LIT §109 in number order

- **TRUE after:** the whole `## 109.` section (today after `## 113.`, rust/SPEC-LIT.md ~35513) is moved verbatim
  to between `## 108.` and `## 110.`, with its `---` separator. No text changes.
- **Closes:** D:401.
- **Files:** `rust/SPEC-LIT.md`.
- **Verify:**
  - V-RUST: `spec_lit xref` pass with counts unchanged;
  - `git diff --stat` shows equal insertions and deletions;
  - a section-order check (`^## N.` strictly increasing) passes.
- **Deps:** none. **Size:** S. **Coder:** glm.

#### T18-VAL-02 — Single verdicts by class; the registry comment; the pipe note

- **TRUE after:**
  - Under single, the verdict conditions of Gates 98-B, 98-C, 100-A and 100-B, and any other verdict line the
    supervisor finds that re-tests an R-class row's quantity, use `precision::roundoff_tol(tol64)` exactly where
    the row itself is class R. That is the adopted D-F32-1 rule (§5.1.4 of docs/17); no new tolerance is made.
    `roundoff_tol` is the identity in f64 (precision.rs:131-137), so f64 verdicts are bitwise unchanged.
  - The `GateReport.gate` comment (validate.rs:~205) says the summary counts registrations and a name may repeat
    (95-A four times, 110-B and 110-C twice), as SPEC-LIT 110.5 counts them.
  - `validate_gates/pipe.rs:518` writes `143.5, missed: …` (TET-14's lower-case form) instead of `143.5 MISSED:`.
- **Closes:** D:351, D:368, D:415.
- **Files:** `rust/src/bin/validate.rs` and the gate files whose verdicts change, `rust/src/bin/validate_gates/pipe.rs`.
- **Verify:**
  - V-RUST;
  - `ofgpu-validate -sections <98-B, 98-C, 100-A, 100-B and pipe's numbers>` in f64: `tools/f64_identity.py rows`
    against HEAD is identical;
  - under single, each touched verdict line agrees with its R rows;
  - `cargo test --release --bin ofgpu-validate -- class verdict` counts stated.
- **Deps:** T18-TOOL-01. **Size:** S. **Coder:** glm.

#### T18-SOLV-06 — Load-time refusals

- **TRUE after:**
  - (a) `LoweredCase::refuse_compressible` refuses `physics.compressible` and `initial.regions` by name ("runs on
    ofgpu-compressible", SPEC-LIT 127/158). `common::load_case_inner` calls it for every driver except
    `ofgpu-compressible`, beside `refuse_parcels`, `refuse_dust` and `refuse_vent` (common/mod.rs:129-163). The
    dust+compressible message at common/mod.rs:149 is unchanged.
  - (b) A case that selects GAMG (solver or preconditioner) for an equation the driver does not solve through
    `PbicgstabBackend` is refused when the case is read. Today it is refused at the first solve, by the internal
    name `set_gamg`. Those equations are the `ofgpu-vof` p_rgh, the compressible p, energy, turbulence, U and
    `dist_solve`. The refusal names the equation, GAMG and SPEC-LIT 141.7.
- **Closes:** D:341, D:371.
- **Files:**
  - `rust/src/io/case_json.rs` (`refuse_compressible` beside `refuse_vent`, ~1954)
  - `rust/src/bin/common/mod.rs`
  - `rust/src/io/case.rs` (the equation check)
  - the drivers' tests
- **Tests:**
  - `ofgpu-lowmach` on `cases/sodShockTube.jsonc` is refused, naming `physics.compressible` and
    `ofgpu-compressible`;
  - `ofgpu-compressible` on it is unchanged (its existing tests);
  - a lowmach JSONC case with GAMG on U is refused at load, naming U, before any GPU work;
  - GAMG on lowmach p is still accepted;
  - every shipped `cases/*.jsonc` still loads in the drivers whose tests load it (no test count drops).
- **Verify:**
  - V-RUST, plus `--bin ofgpu-lowmach -- refuse` and `--bin ofgpu-compressible -- case`;
  - `ofgpu-validate -sections <the CMP and DST sections that load JSONC cases>` rows unchanged (f64_identity rows).
- **Deps:** T18-TOOL-01. **Size:** M. **Coder:** glm.

#### T18-PAR-01 — SPMD failures vote; per-rank residual lines; a quiet -quiet

- **TRUE after:**
  - (a) `build_spmd` (rust/src/amg/dist.rs:671) and the `DistGamg` setup (distcycle.rs:~814) call `vote`
    (dist.rs:714) on `SpmdHalo::from_level`'s result before the next collective. A failure on one rank then
    returns the same refusal on every rank at once, not a ThreadComm timeout.
  - (b) After each distributed solve, unless `-quiet`, `ofgpu-par` prints one line per rank:
    `rank <r>: <label> iterations <n> residual <x:.3e>`. Here `x` is that rank's owned-row part of the normalised
    L1 residual, `sum_{owned} |b - A psi| / normFactor_global`, so the ranks' `x` sum to the printed global residual
    (class A: 1e-12 relative in f64, R in single).
  - (c) `ofgpu-par -solve restart -quiet` prints no `0/ fields: …` line.
- **Closes:** D:326, D:327 (solver half), D:395.
- **Files:**
  - `rust/src/amg/dist.rs`, `rust/src/amg/distcycle.rs`
  - `rust/src/bin/par.rs` (and the cavity mesh path that prints the line)
  - tests
- **Proof first:**
  - (a) a ThreadComm P = 2 test with a short timeout, in which rank 1's level halo is made invalid through a
    `#[cfg(test)]` hook: it fails by timeout today and returns the same refusal on both ranks after;
  - (c) a test that captures `-quiet` stdout and finds the line today.
- **Verify:**
  - V-RUST, plus `--test` for the distflow, amg_dist and par_driver integration tests the supervisor names, counts
    stated;
  - golden bitwise hashes of PAR-14 (139.10) unchanged.
- **Deps:** T18-TOOL-01. **Size:** M. **Coder:** glm.

#### T18-SOLV-07 — Case keys for FCG, the K-cycle and the f32 hierarchy

- **TRUE after:**
  - `LinearSolverKind::Fcg` (`FCG` in `from_name`, rust/src/io/case.rs:139-146; dictionary and JSONC) runs
    `solver::solve_fcg` (solver.rs:2509/2525, both builds). It is symmetric-only like PCG: the `needs_symmetry`
    check in `solver::solve` (:3558) covers it.
  - The GAMG keys `cycle` (`V` default | `K`) and `kThreshold` (default `amg::cycle::DEFAULT_K_THRESHOLD` = 0.25,
    finite in [0, 1)) are read by `read_gamg_controls` (case.rs:293) and `JsonGamg::lower` (case_json.rs:1270) and
    reach `Gamg::set_cycle` (amg/cycle.rs:326).
  - `K` is accepted only when the GAMG is the preconditioner of FCG. Anywhere else it is refused by name, citing
    SPEC-LIT 141.6 ("PCG and PBiCGStab refuse it").
  - `JsonGamg` gains `precision` (`native` | `single`, through `gamg_precision_from_name`, case.rs:206), so the
    JSONC route equals the dictionary route.
  - `docs/schema/case-1.json` is regenerated by its write test, and SPEC-LIT 141 gets a subsection with the keys.
  - Every existing case is bitwise unchanged.
- **Closes:** D:325 (the case half), D:353.
- **Files:**
  - `rust/src/io/case.rs`, `rust/src/io/case_json.rs`
  - `rust/src/solver.rs` (one dispatch arm)
  - the code that turns `GamgControls` into a `Gamg` (supervisor greps `set_gamg`, solver.rs:602)
  - `docs/schema/case-1.json`, `rust/SPEC-LIT.md` (141), and the gui schema tests if they pin the list
- **Tests:**
  - the dictionary `p { solver FCG; preconditioner { preconditioner GAMG; cycle K; kThreshold 0.3; } }` and the
    equivalent JSONC lower to equal `SolverControls` (Fcg, GAMG precon, cycle K 0.3);
  - `solver PCG` with a K-cycle GAMG is refused, naming FCG;
  - `kThreshold 1.0` is refused;
  - JSONC `gamg: {precision: 'single'}` lowers to `GamgPrecision::Single` (today: unknown field);
  - the smallest shipped lowmach cavity with p FCG+GAMG(K) meets the same residualControl as PCG+GAMG(V) within
    its iteration budget (class P; no bitwise claim).
- **Verify:** V-RUST (filters `case:: case_json gamg fcg`), plus server `schema.test`, `registry/sync.test`.
- **Deps:** T18-TOOL-01. **Size:** M. **Coder:** glm.

#### T18-SOLV-01 — turbulentInlet, the synthetic inlets and cyclic jumps in transient lowmach/buoyant

- **TRUE after:**
  - `ofgpu-lowmach` and `ofgpu-buoyant` pass `BcInputs::turbulent_inlet = true` on U exactly when the run is
    transient (`-endTime`/`-deltaT`). turbulentInlet, turbulentDigitalFilterInlet and turbulentDFSEMInlet then
    advance through `Simple::begin_time_step` (simple.rs:874-881), from the first step on, as SPEC-LIT 120.10 and
    120.12 define. If step 0 needs a draw (lowmach.rs `open_time_step` skips step 0), the unit adds it without
    moving any other step.
  - A steady run keeps refusing the three names by name, now with "transient runs only".
  - Both drivers pass `BcInputs::jumps = true` on p (simple.rs:1166-1174, 1336-1341 already refresh and assemble
    the jump), so fixedJump, a constant uniformJump, fan and porousBafflePressure on a cyclic pair run. A
    time-varying uniformJump waits for T18-SOLV-02.
  - The `BcInputs` doc comments that say "every driver passes today" (field_setup.rs:~305, ~84) are corrected.
  - A case without these names is bitwise unchanged.
  - The `-rigidBody` paths are not touched.
- **Closes:** D:343, D:320 (duplicate), and the BC-14 turbulentInlet, BC-16 and BC-17 parts of D:318.
- **Files:**
  - `rust/src/bin/lowmach.rs` (setup only), `rust/src/bin/buoyant.rs`, `rust/src/field_setup.rs` (comments)
  - their tests
  - SPEC-LIT 120.2, 120.10, 120.12 (one sentence each on which drivers opt in)
- **Tests:**
  - a 3-step transient lowmach channel (blockgen, 8x4x4) with turbulentInlet on the inlet: the inlet faces' U
    differs between steps 1 and 2; the face mean stays within SPEC-LIT 120.10's bound of `referenceField`; two
    runs are bitwise equal;
  - the same case steady is refused, naming turbulentInlet;
  - a cyclic pair with fixedJump Δp on p: after the solve the pressure jump across the pair is Δp to 1e-12 (f64;
    class R, tol32 1e-5, under single);
  - the same accept/refuse pair on `ofgpu-buoyant`.
- **Verify:** V-RUST (`--bin ofgpu-lowmach`, `--bin ofgpu-buoyant` filters), plus `ofgpu-validate -sections`
  covering lowmach/buoyant gates unchanged (f64_identity rows).
- **Deps:** W2 landed. **Size:** M. **Coder:** glm.

#### T18-SOLV-02 — Timed Function1 and timeVaryingMappedFixedValue in transient lowmach/buoyant

- **TRUE after:**
  - In a transient run, both drivers set U, p, T (and the turbulence scalars where present) up with
    `setup_*_field_timed` (field_setup.rs:5690/5726) over the run's `[t_start, t_end]`.
  - They keep one `F1Clock` (function1/device.rs:92), advanced once per time step, and apply each field's
    `TimedBcs` (`apply_scalar`/`apply_vector`, field_setup.rs:613/635) after the advance and before the step's
    solve.
  - A steady run keeps refusing time-varying entries by name.
  - Constant entries and cases without timed patches are bitwise unchanged.
  - SPEC-LIT 120.1's "Which runs advance the clock" paragraph names the two drivers.
- **Closes:** the BC-06 and BC-15 parts of D:318.
- **Files:** `rust/src/bin/lowmach.rs` (non-rigid paths), `rust/src/bin/buoyant.rs`, tests, `rust/SPEC-LIT.md`
  (120.1, 120.11).
- **Tests:**
  - **Ramp:** SPEC-LIT 120.1's closed form through the driver. A transient lowmach run (t in [0, 1],
    dt = 1/64) has a `flowRateInletVelocity` patch with rate `Q0·linearRamp(start 0.25, duration 0.5)`. The
    trapezoid integral of the measured inflow is 0.5·Q0 to 1e-12 relative in f64 (class R; tol32 = 1e-5 under
    single).
  - **Inflow database:** a `timeVaryingMappedFixedValue` patch has two time folders, value `a` at t = 0 and `b` at
    t = 1, uniform. At t = 0.5 the face value is (a+b)/2 to 1e-12 (R).
  - **Steady refusal:** a steady run with a `table` entry is refused, naming the patch and the entry.
- **Verify:** V-RUST (bin filters), plus `ofgpu-validate` lowmach/buoyant sections unchanged.
- **Deps:** T18-SOLV-01. **Size:** M. **Coder:** glm.

#### T18-SOLV-03 — The advective outlet; mapped T and turbulence inlets

- **TRUE after:**
  - In a transient run, both drivers pass `BcInputs::advective = true` and refresh each advective face once per
    step through `field::advective` (its `store_old_time` and triple, field.rs:389-396, field_ops.rs:1085-1090).
    The wave speed is what SPEC-LIT 120.4 prescribes for an incompressible outlet. If 120.4 leaves it to the
    driver, the unit writes the face-normal velocity `phi_f/|S_f|` into 120.4 as TAKEN.
  - `mapped`/`mappedFixedValue` on T (the energy field) and on k, epsilon, omega and nut refresh every outer
    iteration, as U and p do (`field_ops::update_mapped_scalar`), so those setups pass `mapped: true`.
  - Cases without these names are bitwise unchanged.
- **Closes:** the BC-09 part of D:318, and D:319.
- **Files:** `rust/src/bin/lowmach.rs` (non-rigid), `rust/src/bin/buoyant.rs`, tests, SPEC-LIT 120.4 and 120.10.
- **Tests:**
  - a mapped T inlet sampling an offset plane equals the plane's cell values bitwise after one outer iteration
    (class D);
  - an advective outlet on a uniform field stays bitwise uniform over 10 steps (D);
  - on a convected step profile, the outlet face after each step equals the (120.4) update of its stored old value,
    computed on the host in f64 (class A, 1e-12 f64 / R single).
- **Verify:** V-RUST (bin filters), plus validate lowmach/buoyant sections unchanged.
- **Deps:** T18-SOLV-02. **Size:** M. **Coder:** glm.

#### T18-SOLV-04 — The thermal family end to end in ofgpu-lowmach

- **TRUE after:** a new integration test runs `ofgpu-lowmach` as the binary on a still, laminar slab (U = 0,
  gravity off), 1-D in x, n = 16, length L, conductivity k.
  - One wall is fixedValue T1. The other is `externalWallHeatFluxTemperature` in coefficient mode (h, Ta, no
    layers).
  - The steady cell temperatures equal the closed form `T(x) = T1 + (Ta − T1)·h·x/(k + h·L)` to 1e-10 (class P
    in f64; R under single).
  - The wall heat flux equals `h·(Ta − T_w)` (class A).
  - A failure is a bug found: the test stays, asserting and printing MISSED with the measured value, and the unit
    returns a REVIEW-LATER line. No solver change.
- **Closes:** D:322.
- **Files:** `rust/tests/lowmach_thermal_bc.rs` (new; house header, PROVENANCE row, source count in NOTICE and
  PROVENANCE).
- **Verify:** V-RUST (`--test lowmach_thermal_bc`, both builds).
- **Deps:** none. **Size:** S. **Coder:** glm.

#### T18-SOLV-05 — ofgpu-buoyant writes yPlus.json

- **TRUE after:**
  - The y+ block of `ofgpu-lowmach` (lowmach.rs:~3873-3960) moves into one helper in `rust/src/bin/common`: the
    per-patch first-cell `y+ = C_mu^{1/4} y sqrt(k_P)/nu` on wall patches and `yPlus.json` (SPEC-LIT 29 note,
    AUT-15).
  - lowmach calls it, and its output is byte-identical: `tests/lowmach_yplus_json.rs` passes unchanged.
  - `ofgpu-buoyant` calls it at the end of a run whose model carries k. It prints the same lines and writes
    `yPlus.json` with `"solver": "ofgpu-buoyant"`, and removes a stale one when there is no k.
  - The SPEC-LIT note names both drivers.
- **Closes:** D:323.
- **Files:**
  - `rust/src/bin/common/` (new helper module or an existing file)
  - `rust/src/bin/lowmach.rs` (non-rigid output path), `rust/src/bin/buoyant.rs`
  - `rust/tests/buoyant_yplus_json.rs` (new)
  - SPEC-LIT 29
- **Tests:** `tests/buoyant_yplus_json.rs` runs `ofgpu-buoyant` for 20 iterations on its smallest shipped RANS case
  and requires `%g` of each JSON value to equal its printed token, as the lowmach gate does.
- **Verify:** V-RUST (`--test lowmach_yplus_json`, `--test buoyant_yplus_json`).
- **Deps:** none. **Size:** M. **Coder:** glm.

#### T18-F32-01 — CHT region residuals in f64 under single

- **TRUE after:**
  - Under `--features single`, `solver::residual_ranged` (solver.rs:1823) forms the ranged residual and its norm
    factor through the f64 accumulation path. That path is the one §118's `residual64_terms` /
    `norm_factor64_terms` use for the whole system.
  - So cht.rs:2225/2245 `region_met` no longer sits at the f32 floor.
  - The change is `cfg(feature = "single")` only. f64 is bitwise: `tools/f64_identity.py sass` + `sass-diff`
    shows identical hashes if a `.cu` is touched.
  - This extends the adopted §118 mixed-precision rule; it is not a new numerics choice.
- **Closes:** the rest of D:340 (its NaN half is already true and is noted in place).
- **Files:** `rust/src/solver.rs`, `rust/tests/f32_cht.rs`, SPEC-LIT 118 (one sentence).
- **Proof first:** a two-region conduction case in `tests/f32_cht.rs` whose region residual reaches 1e-9 relative
  in f64. Under single it stalls near 1e-7 today (fails), and after the change it meets 1e-9 within the same
  iteration budget (class A).
- **Verify:** V-RUST (`--test f32_cht` both builds; lib filter `residual_ranged cht`).
- **Deps:** none. **Size:** S. **Coder:** glm.

#### T18-SOLV-08 — One RasCore entry for SA's conservative correction

- **TRUE after:**
  - `RasCore` (rust/src/turbulence.rs) exposes the after-diffusivity assembly as one `pub(crate)` entry.
  - `SpalartAllmaras::correct_conservative` (rust/src/models/spalart_allmaras/conservative.rs) calls it, and its
    step-by-step copy is deleted.
  - `rho_one_is_bitwise_kinematic` and every SA and compressible SA test pass bitwise in both builds.
  - No kernel changes.
- **Closes:** D:364.
- **Files:** `rust/src/turbulence.rs`, `rust/src/models/spalart_allmaras/conservative.rs`.
- **Verify:** V-RUST (filters `spalart turbulence`), plus `--bin ofgpu-compressible -- sa`, counts unchanged.
- **Deps:** none. **Size:** S. **Coder:** glm.

#### T18-SOLV-09 — JSONC Spalart-Allmaras in ofgpu-compressible

- **TRUE after:**
  - JSONC `initial.nuTilda` (a number ≥ 0) and per-patch `nuTilda` conditions lower into `LoweredCase`, and the
    schema is regenerated.
  - `ofgpu-compressible` accepts `physics.turbulence: 'SpalartAllmaras'` on a JSONC case (today refused, naming
    nuTilda) and sets nu_tilda/nut up from it, as the in-process CMP-27 path does.
  - Laminar JSONC cases and every existing case are bitwise unchanged.
- **Closes:** D:365.
- **Files:**
  - `rust/src/io/case_json.rs`, `rust/src/bin/compressible.rs`
  - `docs/schema/case-1.json`, SPEC-LIT 127 (one sentence)
  - gui schema tests if pinned
- **Tests:** a JSONC copy of an existing in-process compressible SA test case gives fields bitwise equal to the
  in-process path after N iterations (class D).
- **Verify:** V-RUST (`--bin ofgpu-compressible`), plus server `schema.test`.
- **Deps:** T18-SOLV-08. **Size:** M. **Coder:** glm.

#### T18-MESH-01 — Segment-graded JSONC cartesian axes

- **TRUE after:**
  - `mesh.grading.<axis>` accepts `segments: [{to, cells, expansion}]` as an alternative to
    `expansion`/`twoSided`; both together are refused by name.
  - The axis is the concatenation of single-graded sub-axes `[lo, to1], [to1, to2], …`, each graded by
    blockgen's existing one-sided formula (`GradedAxis`, blockgen.rs:102).
  - The cells must sum to `mesh.cells[axis]` and the last `to` must equal the bounds' `hi`; both are refused by
    name otherwise.
  - A case without segments lowers to the same mesh bitwise
    (`a_case_without_grading_lowers_to_the_same_mesh_as_before`).
  - `cases/flatPlateLaminar.jsonc` is NOT changed: its gate keeps its mesh. A new example is optional.
  - Existing `GradedAxis` literals are not touched.
- **Closes:** D:334.
- **Files:**
  - `rust/src/io/case_json.rs`
  - `rust/src/blockgen.rs` (only if a points helper must become pub)
  - `docs/schema/case-1.json`, the SPEC-LIT JSONC mesh section (supervisor finds it)
  - gui schema test if pinned
- **Tests:**
  - x = `[{to: 0.2, cells: 8, expansion: 0.25}, {to: 1.0, cells: 24, expansion: 4}]` has node 8 at exactly 0.2,
    and its point list equals two separate graded axes concatenated, bitwise (D);
  - a cell sum of 31 is refused, and so is a last `to` of 0.9.
- **Verify:** V-RUST (filter `case_json grading blockgen`), plus server `schema.test`.
- **Deps:** none. **Size:** M. **Coder:** glm.

#### T18-CHEM-01 — Parcels from constant/cloudProperties

- **TRUE after:**
  - A dictionary case with `constant/cloudProperties` is lowered onto the same `CaseParcels` the JSONC `parcels`
    block gives. Its entry names are taken by name only from the public user guides, as §120's practice is, and
    map one to one onto the JSONC fields listed in SPEC-LIT 137.10.
  - `ofgpu-lowmach` runs it. Every other driver still refuses it by name.
  - An entry with no JSONC counterpart is refused by name.
- **Closes:** D:379a.
- **Files:**
  - a reader in `rust/src/io/` (new file: house header, PROVENANCE row, counts)
  - `rust/src/bin/lowmach.rs` (the non-rigid case-load path)
  - tests, SPEC-LIT 137.10
- **Tests:** `lowmach_parcels::tests`' evaporation case, written as a dictionary case, gives a `RunEnd.parcels`
  bitwise equal to the JSONC route (D). An unknown entry is refused.
- **Verify:** V-RUST (`--bin ofgpu-lowmach -- parcels`, lib filter `cloud_dict case_parcels`).
- **Deps:** none. **Size:** M. **Coder:** glm.

#### T18-GUI-05 — Polymesh cases in the studio

- **TRUE after:**
  - For a .jsonc case with `mesh.kind: 'polymesh'`, `case_read` and the Boundary editor read
    `<mesh.path>/constant/polyMesh/boundary` (relative to the case file's directory) with the parser
    `foamBoundaryTable` already uses, and list the patches read-only. set_patch stays refused with `MESH_KIND`.
  - `case_validate` adds the four file-level refusals of `polymesh_of`, with the solver's messages (SPEC-LIT
    158.10):
    - no polyMesh directory;
    - a cyclic patch in the file;
    - a patch the wedge rule governs;
    - a polyMesh patch no rule governs.
  - The mock solver (`gui/server/src/mock/solver.ts`) uses the polyMesh points' bounding box for a polymesh case.
- **Closes:** D:411, D:412, D:413.
- **Files:** `gui/server/src/tools/case.ts`, `gui/server/src/tools/guiBoundary.ts`, `gui/server/src/mock/solver.ts`,
  tests (fixture dirs made in a temp folder).
- **Tests:** one per refusal, plus a listed-patches test and a mock box test.
- **Verify:** V-GUI. **Deps:** none. **Size:** M. **Coder:** glm.

#### T18-GUI-08 — Compressible monitors and the Mach view

- **TRUE after:**
  - `ofgpu-compressible`'s `residualStyle` is `'compressible'` (gui/shared/src/registry.ts:406; the union at :49
    gains it).
  - Its parser (`gui/shared/src/residuals.ts`) puts the solver's residual tokens on the residual chart and
    `M_max`, `M_mean` on a monitor series. The tokens follow the binary's actual println lines, which the
    supervisor reads in compressible.rs.
  - A binary may name a default colour field (`defaultField`). `ofgpu-compressible` names `Mach`, and its results
    open coloured by Mach when the field exists; otherwise as today.
- **Closes:** D:360 (the Mach preset and residualStyle parts).
- **Files:** `gui/shared/src/registry.ts`, `gui/shared/src/residuals.ts` (+ test), the web viewer's
  default-field selection.
- **Tests:** a captured 5-line compressible log parses into the expected residual and monitor names; the registry
  sync test; a viewer test of the default field.
- **Verify:** V-GUI plus `registry/sync.test`. **Deps:** none. **Size:** M. **Coder:** glm.

#### T18-GUI-11 — Per-rank residuals in the run panel

- **TRUE after:**
  - `gui/shared/src/parReport.ts` parses T18-PAR-01's `rank <r>: <label> iterations <n> residual <x>` lines into
    per-rank series.
  - `ParReport.tsx` shows them beside the Krylov iteration counts.
  - The mock (`gui/server/src/mock/par.ts`) prints the same lines.
- **Closes:** D:327 (the gui half).
- **Files:** `gui/shared/src/parReport.ts` (+ test), `gui/web/src/components/side/ParReport.tsx`,
  `gui/server/src/mock/par.ts`.
- **Verify:** V-GUI. **Deps:** T18-PAR-01. **Size:** S. **Coder:** glm.

#### T18-VAL-01 — Every row classed; NaN-propagating folds

- **TRUE after:**
  - A full f64 run's class line prints `0 rows carry no class yet` (today 53; 38 in single's 90-section command,
    D:352).
  - Each plain `c.check` row becomes `check_p` / `check_a` / `check_d` / `check_roundoff` by SPEC-LIT 118.14's
    rule. The supervisor's brief lists every row with its class; the classes are its judgement.
  - The sections the class line names (20, 66, 67, 68, 69, 80 at the last count) wrap in `c.ran`.
  - The 19 `worst = worst.max(..)` folds use `nan_max` (validate.rs:916), so a NaN fails its row.
  - f64 pass/fail of every row is unchanged: `tools/f64_identity.py rows` against HEAD on the touched sections
    shows class columns only. Under single only R rows move, by D-F32-1.
- **Closes:** D:352.
- **Files:** `rust/src/bin/validate.rs`, `rust/src/bin/validate_gates/*.rs` (the touched gates).
- **Proof first:** a unit test feeds one converted fold a NaN. The row fails after the change and passes today.
- **Verify:** V-RUST, plus `ofgpu-validate -sections <the touched sections>` in both builds. A section group over
  10 minutes is split into separate commands.
- **Deps:** T18-VAL-02 (same file, earlier wave). **Size:** M. **Coder:** glm.

#### T18-PAR-02 — Distributed GAMG in ofgpu-par

- **TRUE after:**
  - When the case's p solver or preconditioner is GAMG, `ofgpu-par` builds one `DistGamg` per rank on its part
    (amg/distcycle.rs:716, setup :753) and preconditions the distributed Krylov with it (`precondition`, :1058):
    a V-cycle, or a K-cycle only under FCG, per T18-SOLV-07.
  - `-check` reports it, and the SPEC-LIT 139.4 refusal list no longer names GAMG.
  - Without GAMG, every run is bitwise unchanged.
- **Closes:** D:325 (the ofgpu-par half).
- **Files:** `rust/src/bin/par.rs`, possibly `rust/src/amg/distcycle.rs` (no numerics change in the cycle), tests,
  SPEC-LIT 139.
- **Tests:**
  - a ThreadComm P = 2 decomposed cavity (`BITWISE_CAVITY_N`) with p GAMG meets the same residualControl as the
    P = 1 serial GAMG (class P);
  - if SPEC-LIT 139.10's bitwise path covers it, P = 1 SPMD equals serial bitwise (D). The supervisor decides
    after reading 139.10 and says which.
- **Verify:** V-RUST (par_driver, distflow tests), plus PAR-14's golden hashes unchanged.
- **Deps:** T18-SOLV-07, T18-PAR-01. **Size:** M. **Coder:** glm.

#### T18-GUI-09 — Probes and streamlines on non-convex cells

- **TRUE after:**
  - `gui/server/src/formats/pointLocate.ts` decides containment in a cell by its centre-fan tet decomposition:
    each face is triangulated about its centroid, and the triangles make tets with the cell centre. A point is
    inside when it is inside one of them, with an orientation tolerance of 1e-12 times the cell size. So a
    star-shaped cell answers inside, not `nearest-centre`. `nearest-centre` stays the answer only for a point in
    no cell.
  - `polystream.ts` continues a streamline that leaves through a boundary face and comes back within one step
    length, by re-locating the next point. A true exit ends `exited`, and a zero-volume tet ends the line only when
    no neighbour holds the point.
- **Closes:** D:331, D:349.
- **Files:** `gui/server/src/formats/pointLocate.ts`, `gui/server/src/formats/polystream.ts`, their tests.
- **Tests:**
  - an L-shaped non-convex polyhedron: a point in the notch is outside, the others inside;
  - a snappy split-face hex: inside;
  - a streamline across a U-channel's mouth continues.
- **Verify:** V-GUI. **Deps:** none. **Size:** M. **Coder:** glm.

#### T18-GUI-10 — Reloaded boolean results; automesher stage counters

- **TRUE after:**
  - An open tab of a boolean result reloads when the file is overwritten. The tab keys its dataset on path plus
    mtime and size, or the server sends a file-changed event; the supervisor picks the smaller change.
  - The automesher run's emit-stage row shows n_prisms, n_layer_points, inflate_rounds, n_cut, n_collapsed,
    n_tapered and coverage when they are nonzero (the TET-13b/c stage JSON).
- **Closes:** D:332, D:367.
- **Files:** the geometry-tab dataset cache and the run stage view (the supervisor names them by grep), plus their
  tests.
- **Tests:** open a tab, overwrite its file, and the dataset updates without reopening; a stage JSON with
  `n_prisms: 12` renders the counter.
- **Verify:** V-GUI. **Deps:** none. **Size:** S. **Coder:** glm.

#### T18-DUST-01 — dust_progress shares its host transfers

- **TRUE after:** the `rust/tests/dust_progress.rs` harness uploads the mesh and the constant fields once per test
  binary, as a shared fixture, instead of once per case. The asserts, tolerances and test count are the same. The
  per-build wall time before and after is reported, with a target under 3 min on the shared GPU (D:339 measured
  5-6 min against an 8-min plan budget).
- **Closes:** D:339.
- **Files:** `rust/tests/dust_progress.rs` (the DUST clean-room rule applies: no equations change).
- **Verify:** V-RUST (`--test dust_progress`, both builds, counts unchanged).
- **Deps:** none. **Size:** S. **Coder:** glm.

#### T18-DOC-01 — README and GUIDEBOOK prose match the tree

- **TRUE after:**
  - `README.md` / `README.en.md` limitation bullets no longer claim 'No finite-rate (Arrhenius) chemistry',
    'Radiation here is surface-to-surface only' or 'S2S, species transport and Lagrangian sprays have no case
    format'. CHR-07..28 added them. Instead the bullets state what the tree does and refuses.
  - The Mundo / Gate 78-D reference lines (README.md:425, README.en.md:435, GUIDEBOOK.en.md:1100, 1113) say the
    measured leg is compared with Riboux & Gordillo 2014 (SPEC-LIT 146.2) and MISSES.
  - GUIDEBOOK §11 says sections 80, 81, 92, 93 and 95 start sibling executables (`ofgpu-lowmach`,
    `ofgpu-compressible`, `ofgpu-automesher`) next to `ofgpu-validate`, so validate runs from
    `target/<dir>/release`.
  - The tranche-18 features (BC wiring, FCG and K-cycle keys, JSONC SA, segments, cloudProperties) are named where
    the documents list capabilities.
  - The generated blocks are not hand-edited, `python tools/validate_record.py check` passes, and the Korean and
    English documents stay parallel.
- **Closes:** D:333, D:390, D:416.
- **Files:** `README.md`, `README.en.md`, `docs/GUIDEBOOK.md`, `docs/GUIDEBOOK.en.md`.
- **Verify:** V-PY `check`, plus `cargo test --release --lib -- xref` if a SPEC-LIT citation is added, plus a
  rendered-Markdown look at each changed table. The D:418 bold rule applies.
- **Deps:** every other unit (it describes them). **Size:** M.
- **Coder:** **opus-only**. It is judgement text: deciding which bilingual claims are stale and how to state the
  current limits honestly. There is no code, and a contract-only brief would have to contain the prose itself.

## 7. The waves

A wave is one `t18-wave.js` run, and its units run one after another in the order listed. The order puts the most
valuable and lowest-risk units first, keeps gui units that share files together, and orders the solver units so
that the ones sharing `lowmach.rs`/`buoyant.rs` run back to back. Before the first unit of each wave the
supervisor runs the GLM tool probe (§5 rule 1). After each wave the orchestrator does three things:

- writes the new review-later rows;
- runs V-GUI and the audit filters once on the wave's tip;
- reports the push SHA. Pushing is the user's.

| wave | units in order | why this order |
|---|---|---|
| **W1 — studio and tools** | T18-GUI-01, T18-GUI-02, T18-GUI-03, T18-GUI-04, T18-GUI-06, T18-GUI-07, T18-TOOL-02 | No Rust, so it carries no dc.seed or build risk. These are small user-visible fixes. GUI-02 and GUI-03 both touch the Boundary editor path and run back to back |
| **W2 — Rust hygiene, refusals and solver keys** | T18-TOOL-01, T18-TEST-01, T18-DOC-02, T18-VAL-02, T18-SOLV-06, T18-PAR-01, T18-SOLV-07 | TOOL-01 first, so every later Rust unit can re-anchor dc.seed. TEST-01 next removes the schema race that would flake every later V-RUST run. Then the small, f64-bitwise units, and then the two M units |
| **W3 — BC driver wiring and outputs** | T18-SOLV-01, T18-SOLV-02, T18-SOLV-03, T18-SOLV-04, T18-SOLV-05, T18-F32-01 | The three wiring units share `lowmach.rs`/`buoyant.rs` and depend in that order. SOLV-04 and SOLV-05 then exercise the same drivers. F32-01 is independent |
| **W4 — models, formats and studio follow-ups** | T18-SOLV-08, T18-SOLV-09, T18-MESH-01, T18-CHEM-01, T18-GUI-05, T18-GUI-08, T18-GUI-11 | SA refactor before JSONC SA. The three schema-touching units (SOLV-09, MESH-01, CHEM-01) run back to back, then the gui units (GUI-11 needs W2's PAR-01) |
| **W5 — classes, distributed GAMG, post-processing, prose** | T18-VAL-01, T18-PAR-02, T18-GUI-09, T18-GUI-10, T18-DUST-01, T18-DOC-01 | The larger or lower-value units. DOC-01 is last because it describes everything before it |

## 8. The USER list (one decision each)

Every item is open. The tranche does not wait on any of them, and none is built without the user's answer.

1. **D:324 (DST-06):** add a temperature-dependent viscosity μ(T) to `FlowState` for the ρ-weighted SST of dust
   cases? This changes the DST gate results.
2. **D:336 (TET-07a):** allow incremental Delaunay insertion in boundary recovery? It is faster on large surfaces,
   but the mesh may differ at degenerate ties. The other choice is to keep the per-pass rebuild.
3. **D:342 (AMX-02):** set AMGX_DLL system-wide, or copy `amgxsh.dll` into this project? Today runs borrow
   FDS_GPU_RUST_AmgX2's copy.
4. **D:354 (AMG-06b):** extend the opt-in f32 GAMG copy to the K-cycle and to `DistGamg`? This is new numerics
   scope.
5. **D:356 (TET-11/12, A28, A87, A132):** evaluate the §92.3 G1-G6 sums in `HostGeom64` under single, so the tet
   and automesher paths write meshes in single? This is a gate-evaluation change. Recommended: yes.
6. **D:357 (VAL-23, A39):** should `faces_where` check the patch kind, so cyclic patches stop getting wall-function
   names? Existing results change.
7. **D:369 (CHR-12b):** an `ofgpu-lowmach` 'open' domain with no open patch should either be refused by name or
   be pointed to `-sealed`. Which one?
8. **D:370 (CMP-18, A64):** accept the far-field's acoustic transmissivity inheriting the 1.186 c_0 mismatch, or
   open a numerics change?
9. **D:375 (CHR-14/15/16):** wire EDM and PaSR into `ofgpu-lowmach`'s split step? If yes, choose:
   - EDM: integrate ω over dt, or an explicit source;
   - PaSR: κ times the reactor increment, or κ times the laminar rate.
10. **D:378 (CHR-23b):** build participating radiation (P1/fvDOM) into `ofgpu-buoyant`? It needs an implicit sink
    in ScalarTransport. The other choice is to keep today's refusal, which points to `ofgpu-lowmach` (SPEC-LIT
    137.8).
11. **D:379b (CHR-26):** add parcels in steady, restart and moving-mesh lowmach runs, and evaporation into a case
    without species? This is a scope decision.
12. **D:379c (CHR-26, A116):** the (77.2) mixed vapour c_p and the (77.1) dilution of evaporated vapour are
    numerics changes to CHR-26's results. Make them?
13. **D:388 (F32-17, A108):** relax the 1e-12 thresholds of `DdtCoeffs::is_consistent` and `Wobble::new` to a
    `roundoff_tol` under single?
14. **D:391 (PAR-10c):** extend the decomposed SIMPLE beyond its refusal list? That covers non-upwind schemes,
    non-orthogonal snGrad, variable viscosity and halo'd gradients. This is a scope decision while multi-GPU is
    deferred.
15. **D:394 (PAR-15):** add decomposed restart of the turbulence, thermal and transient levels? This is scope.
16. **D:400 (DST-15/16):** add a device gather for Phase B, and change the char-only step end that can go
    subnormal? Both are numerics changes.
17. **D:404 (PAR-13a/b):** add least-squares and skew-corrected gradients and the compressible/vof face-flux sums
    to the decomposed path, plus a turbulence bitwise gate? This is scope.
18. **E:426 (gui import.test):** the failure is now 'Mesh inserted 4, expected 0'. It comes from ignored runtime
    `.meshSummary.json` files under `cases/` (channel, e2e-control/mesh-cartesian,
    e2e-control/mesh-sphere-in-box; 2026-10-15). Delete them (a deletion), or keep the known failure?

LONG rows, waiting for a run or an input rather than a decision:

- D:346 (AUT-07): the R-ECON campaign, about 4 h of GPU.
- D:347 (AUT-11): after the AUT-12 L-run.
- D:358 (VAL-33): a fourth level or a fine run.

The RIGID row D:385 waits until the rigid-body branch is reopened.

## 9. Counts

- **Open D/E rows classified: 86.** That is 85 in D, counting line 379 as three asks, and 1 in E.

  | class | rows |
  |---|---|
  | CODE | 48 |
  | USER | 18 |
  | DONE | 14 |
  | LONG | 3 |
  | DUP | 2 |
  | RIGID | 1 |

- **Units: 33 in 5 waves.**
  - W1: 7
  - W2: 7
  - W3: 6
  - W4: 7
  - W5: 6
- **Units by area:**
  - GUI 11
  - SOLV 9
  - PAR 2
  - TOOL 2
  - DOC 2
  - VAL 2
  - TEST 1
  - F32 1
  - MESH 1
  - CHEM 1
  - DUST 1
- **Units by size:** 16 S and 17 M. No unit is L.
- **Coders:** 32 units glm, 1 opus-only (T18-DOC-01).
- **Rows marked 해결 by this plan's commit: 14.** These are D 321, 344, 361, 374, 380, 381, 384, 393, 398, 403,
  405, 406, 418 and 419. D:340 also gets a dated note for its already-true NaN half.
- **Baselines at 120eaff:**
  - gui shared: 156 / 156
  - gui server: 1008 passed, 1 failed (E:426), 1 skipped of 1010
  - gui web: 296 / 296
  - SPEC-LIT 112.3: library **254**, binary **9**
  - validate: 95 sections (docs/validate-run/f64.json)
