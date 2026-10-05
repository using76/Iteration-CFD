# Contributor commercial grants

[한국어](#한국어)

meteor-cfd is licensed under the Prosperity Public License 3.0.0. Under that licence,
commercial use is limited to a thirty-day trial. To thank the people who improve it,
**주식회사 이터레이션즈 (Iterations Co., Ltd.)**, the licensor, grants commercial-use
licences to contributors whose work is merged. There are two tiers.

| Tier | What qualifies | What you receive |
|---|---|---|
| **Significant contribution** | a merged bug fix or feature that a maintainer labels `grant:significant` | an **unlimited commercial licence for 12 months** from the merge date |
| **Model upgrade** | a merged upgrade of a physical model in **turbulence, viscosity/rheology, heat transfer or vibration/structural dynamics** that a maintainer labels `grant:model-upgrade` | an **unlimited, perpetual commercial licence** |

These grants are offered by Iterations Co., Ltd. in addition to the Prosperity Public
License. They do not change the licence for anyone else.

## 1. Significant contribution: 12 months

A pull request qualifies when all of the following hold:

1. It is merged into `main`.
2. It is one of:
   - a **bug fix** that comes with a test that failed before the fix; or
   - a **feature** with tests, documentation, and a `rust/SPEC-LIT.md` entry where the
     feature has equations.
3. A maintainer labels it `grant:significant`. Typo fixes, formatting, dependency bumps
   and changes of a few lines without a test do not qualify.

**Grant:** an unlimited commercial licence to use meteor-cfd, for the 12 months
starting on the merge date.
- *Unlimited* means any number of users, machines and projects inside the grantee's
  organisation, for any commercial purpose.
- A later qualifying PR restarts the 12 months from its own merge date.

## 2. Model upgrade: perpetual

A pull request qualifies when all of the following hold:

1. It is merged into `main`, through an issue opened with the **Model upgrade** template
   and accepted by a maintainer.
2. It upgrades or adds a physical model in one of these areas:
   - **turbulence**: RANS, transition, LES, hybrid models and wall treatment;
   - **viscosity / rheology**: Newtonian and non-Newtonian models, temperature dependence;
   - **heat transfer**: the energy equation, conjugate heat transfer, radiation, thermal
     wall treatment;
   - **vibration / structural dynamics**: solid dynamics, modal analysis,
     flow-induced vibration.
3. It brings **evidence**: a validation gate in `ofgpu-validate` (documented in
   `rust/SPEC-LIT.md`) against an open, cited reference. The gate either passes where
   it did not before or shows a measured accuracy gain, and no existing validation row
   gets worse.
4. A maintainer labels it `grant:model-upgrade`.

**Grant:** an unlimited, perpetual, worldwide commercial licence to use meteor-cfd,
covering current and future versions.

## 3. Terms common to both tiers

- **Who receives the grant.** The grantee is the contributor named in the pull request.
  If the PR states that the work was done for an employer or another legal entity, that
  entity is the grantee instead.
  - One PR gives one grant, to one grantee.
  - Co-authored PRs give a grant to each listed co-author whom the maintainers accept as
    a substantial author.
- **Not transferable.** A grant cannot be sold, assigned or sublicensed. It may pass to
  a successor that acquires the grantee's whole business.
- **What it covers.** meteor-cfd as distributed by Iterations Co., Ltd.
  - Third-party parts keep their own licences: the GPL-2.0-or-later Blender scripts in
    `tools/promo/`, the Apache-2.0 files listed in `NOTICE`, and external tools such as
    gmsh.
  - The grant gives no right to relicense or redistribute meteor-cfd under other terms.
- **Record.** Every grant is recorded in [`GRANTS.md`](GRANTS.md) with the grantee,
  the pull request, the tier and the start date. That record is the evidence of the
  licence.
- **Decision.** The maintainers of Iterations Co., Ltd. decide whether a pull request
  qualifies and which tier applies. They state the reason on the pull request.
- **AI-assisted work** qualifies on the same terms. The human or organisation who
  submits the PR, and who confirms the right to contribute it, is the contributor.
- **No warranty.** The software comes as is, as stated in `LICENSE`.
- **Questions:** simul@msimul.com.

---

## 한국어

meteor-cfd는 Prosperity Public License 3.0.0을 따르므로, 상업 이용은 30일 체험까지만
허용됩니다. 라이선스 제공자인 **(주)이터레이션즈**는 기여해 주신 분께 감사의 뜻으로,
병합된 기여에 대해 다음 상업 이용 권한을 드립니다.

| 등급 | 대상 | 권한 |
|---|---|---|
| **중요 기여** | 병합된 버그 수정·기능 추가 중 관리자가 `grant:significant`를 붙인 것 | 병합일부터 **1년간 무제한 상업 이용** |
| **모델 업그레이드** | **난류, 점성(유변학), 열전달, 진동(구조 동역학)** 모델을 개선·추가한 병합 PR 중 관리자가 `grant:model-upgrade`를 붙인 것 | **무제한·무기한 상업 이용** |

- **중요 기여 조건:**
  - 버그 수정은 수정 전에 실패하던 테스트가 함께 있어야 합니다.
  - 기능 추가는 테스트와 문서가 있어야 하고, 수식이 있으면 SPEC-LIT 항목도 있어야 합니다.
  - 오타, 서식 정리, 의존성 갱신, 테스트 없는 몇 줄짜리 변경은 해당하지 않습니다.
  - 새로 자격을 얻은 PR이 병합되면 그 날부터 다시 1년이 시작됩니다.
- **모델 업그레이드 조건:**
  - "모델 업그레이드" 템플릿으로 연 이슈가 수락되어 있어야 합니다.
  - 공개된 출처의 기준 데이터로 `ofgpu-validate` 검증 게이트를 추가해야 합니다.
  - 그 게이트가 새로 통과하거나 정확도가 측정 가능하게 좋아져야 하고, 기존 검증 항목은 하나도 나빠지면 안 됩니다.
- **무제한의 범위:** 권한을 받은 조직 안에서 사용자 수·장비 수·프로젝트 수에 제한이 없습니다.
- **권한을 받는 사람:** PR 작성자입니다. 회사 업무로 기여했다고 PR에 적으면 그 회사가 받습니다.
  권한은 양도하거나 재허락할 수 없습니다.
- **적용 범위:** (주)이터레이션즈가 배포하는 meteor-cfd입니다.
  - 제3자 구성요소는 각자의 라이선스를 따릅니다(`tools/promo`의 GPL Blender 스크립트, `NOTICE`의 Apache-2.0 파일 등).
  - 다른 조건으로 재배포하거나 재라이선스할 권리는 포함되지 않습니다.
- **기록:** 모든 권한은 [`GRANTS.md`](GRANTS.md)에 기록되며, 이 기록이 권한의 근거가 됩니다.
- **AI 활용 기여:** AI를 써서 만든 기여도 같은 조건으로 인정합니다.
- **문의:** simul@msimul.com
