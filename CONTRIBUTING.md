# Contributing to meteor-cfd

[한국어 요약](#한국어-요약) · [Contributor commercial grants](CONTRIBUTOR-GRANTS.md)

meteor-cfd is a GPU-resident finite volume CFD solver by 주식회사 이터레이션즈
(Iterations Co., Ltd.), developed in collaboration with 주식회사 메테오시뮬레이션
(Meteo Simulation Co., Ltd.). It is **source-available under the Prosperity Public
License 3.0.0**, not open source; see [`LICENSE`](LICENSE) and [`LICENSING.md`](LICENSING.md).

Contributions are welcome: bug reports, fixes, new features, validation cases, and
upgrades to physical models. People who make significant contributions can earn
**commercial-use rights** to meteor-cfd. See [`CONTRIBUTOR-GRANTS.md`](CONTRIBUTOR-GRANTS.md).

This guide is written so that a person or an AI coding agent can follow it step by
step. Every rule here is checkable.

---

## 1. Before you start

1. **Search the issues** for the bug or feature. Comment there instead of opening a duplicate.
2. **Open an issue first** for anything larger than a small fix. Use the templates:
   - **Bug report**: something gives a wrong number, crashes, or refuses valid input.
   - **Feature request**: a new capability.
   - **Model upgrade**: a change to a turbulence, viscosity/rheology, heat-transfer or
     vibration/structural-dynamics model. This is the route to the perpetual grant.
3. **Changes to solver numerics need an issue that a maintainer has accepted.** This means
   discretisation, solution algorithms, coefficients and model constants. It also covers
   any change that moves a published validation number. A pull request that changes
   numerics without an accepted issue will be closed.

## 2. Repository map

| Path | What it holds |
|---|---|
| `rust/src/` | the solver library (crate `ofgpu`), CUDA kernels, the automesher (`automesher/`) |
| `rust/src/bin/` | the `ofgpu-*` programs (see `[[bin]]` in `rust/Cargo.toml`) |
| `rust/SPEC-LIT.md` | the specification: every equation with its literature source, and every validation gate |
| `rust/PROVENANCE.md` | one row per source file, recording what it was written from |
| `tools/geom/`, `tools/mesh/`, `tools/cad/`, `tools/autonomy/` | Python tools: geometry, meshing pipelines, parametric CAD, autonomous meshing |
| `gui/` | the web studio (server, shared, web) and its AI assistant |
| `docs/` | design plans and records |
| `cases/` | example cases |

## 3. Build and test

```bash
# Rust (needs CUDA 13.x and an NVIDIA GPU for the GPU tests)
cd rust
cargo build --release
cargo test --release --lib -- <module filter>      # run the tests of what you touched
cargo test --release --lib -- xref provenance_audit # always: spec cross-references and provenance

# Validation suite (long; run the sections your change touches)
./target/release/ofgpu-validate

# Python tools (Python 3.13)
python tools/geom/selftest.py
python tools/mesh/selftest.py
python tools/cad/selftest.py
python tools/autonomy/selftest.py

# Studio
cd gui && npm ci && npm run typecheck && npm run test -w shared && npm run test -w web && npm run test -w server
```

Report test results by **name and count** (before and after), not by exit code alone.
Some suites have known, named failures. They are listed in the pull-request template.
Any other failure is a finding.

## 4. Rules for every change

1. **Tests first for bugs.** Add a test that fails on the current code, then fix it.
   Say in the PR that you saw it fail.
2. **Never loosen a gate or a tolerance to make a number pass.** If a gate cannot pass,
   say so and leave it failing (or OPEN) with the measured value.
3. **Never delete or weaken an existing assertion** to make a suite green.
4. **Cite sources.** A new equation, model or constant goes into `rust/SPEC-LIT.md` with
   its open reference (DOI or URL). A validation case names its reference data.
5. **Provenance.** Every new source file gets the repository header (below) and a row
   in `rust/PROVENANCE.md`.
6. **No GPL-licensed source.** Do not read or copy GPL, LGPL or AGPL code into this
   repository. That includes OpenFOAM, FreeCAD and OpenSCAD. Libraries may be called as
   separate programs where their licence allows. Code you copy under a permissive licence
   (MIT, BSD, Apache-2.0) keeps its notice and goes into `NOTICE`.
7. **One concern per pull request.** Keep refactors apart from behaviour changes.
8. **Byte-identity where promised.** A change described as "output only" or
   "performance only" must leave every mesh and field byte-identical; show the comparison.

New source file header (Rust; use `#` comments for Python):

```
// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.
```

## 5. Commits and pull requests

- **Commit subject:** one sentence stating what is now true, e.g. *"The STL reader
  classifies a binary file whose header starts with `solid` by its size and never
  panics"*. The body gives the gate or test numbers.
- **Pull request:** fill in every section of the template. A PR that leaves the
  verification section empty will not be reviewed.
- **Sign-off:** add `Signed-off-by: Your Name <email>` to each commit (`git commit -s`).
  It certifies the [Developer Certificate of Origin 1.1](https://developercertificate.org/).

## 6. How your contribution is licensed

By opening a pull request you agree that your contribution is licensed to
주식회사 이터레이션즈 (Iterations Co., Ltd.) under the **Apache License 2.0**.
This is one of the "standardized public software licenses" named in the Prosperity
license's *Contributions Back* section, so making the contribution is not commercial
use. Iterations Co., Ltd. may then distribute it as part of meteor-cfd under the
Prosperity Public License and under commercial licences.

You confirm that you have the right to contribute the work. If you make it as part of
your job, confirm that your employer allows it, and name the employer in the PR if you
want a grant to go to the employer (see [`CONTRIBUTOR-GRANTS.md`](CONTRIBUTOR-GRANTS.md)).

## 7. Working with AI coding agents

AI-assisted contributions are welcome. The person or organisation who opens the PR is
the contributor and is responsible for the code. Tell the agent to read this file and
`rust/SPEC-LIT.md` (the sections it touches), and give it these rules:

- Read only what the task needs. `rust/SPEC-LIT.md` is very long, so search it by
  section number (`grep -n "^## 92" rust/SPEC-LIT.md`).
- Re-check every line number by searching for quoted text before editing; never trust
  a number copied from an old note.
- Write a failing test before the fix, and report the test name, the panic text and the
  counts before and after.
- Do not change numerics, gates, `tools/autonomy/gates.lock`, `split.lock`,
  `tools/cad/templates.lock` or requirement lock files unless the accepted issue says so.
- Keep each change small. List every file the agent touched in the PR.
- State in the PR which agent and model was used, and what a human checked.

## 8. Reporting security problems

Do not open a public issue for a security problem. Write to **simul@msimul.com**
with the subject `meteor-cfd security`.

---

## 한국어 요약

- meteor-cfd는 (주)이터레이션즈의 소스 공개(Prosperity 3.0.0) 소프트웨어입니다.
  오픈소스는 아닙니다.
- **기여 방법:** 먼저 이슈를 찾아보고, 없으면 템플릿으로 새로 엽니다(버그·기능·모델 업그레이드).
  솔버 수치를 바꾸는 변경은 관리자가 수락한 이슈가 있어야 합니다.
- **규칙:**
  - 버그는 먼저 실패하는 테스트를 추가하고 고칩니다.
  - 검증 게이트나 허용 오차를 완화하지 않고, 기존 단언(assertion)도 지우지 않습니다.
  - 새 수식과 모델은 공개 출처와 함께 `rust/SPEC-LIT.md`에 적습니다.
  - 새 파일에는 헤더를 붙이고 `rust/PROVENANCE.md`에 행을 추가합니다.
  - GPL 계열 코드는 참조하지 않습니다.
- **기여물 라이선스:** PR을 올리면 기여물을 Apache License 2.0으로 (주)이터레이션즈에 제공하는 데 동의하는 것입니다.
  커밋마다 `git commit -s`로 서명합니다.
- **AI 에이전트:** AI로 작업해도 됩니다. 이 문서의 7절 규칙을 에이전트에게 주고,
  PR에 사용한 에이전트·모델과 사람이 확인한 내용을 적어 주세요.
- **상업 이용 권한 보상:** 중요한 기여는 1년간, 모델 업그레이드는 무기한 상업 이용 권한을 받을 수 있습니다.
  [`CONTRIBUTOR-GRANTS.md`](CONTRIBUTOR-GRANTS.md)를 보세요.
- **보안 문제:** 공개 이슈 대신 simul@msimul.com으로 알려 주세요.
