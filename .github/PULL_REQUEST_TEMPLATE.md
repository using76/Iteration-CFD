<!--
Fill in every section. An AI agent can fill this in too; a human must check it before submitting.
모든 항목을 채워 주세요. AI 에이전트가 작성해도 되지만, 제출 전에 사람이 확인해야 합니다.
-->

## What is now true / 무엇이 바뀌었나

<!-- One sentence stating what the code now does, e.g. "The STL reader classifies a binary file whose header starts with `solid` by its size and never panics." -->

Closes #

## Kind / 종류

- [ ] Bug fix (with a test that failed before the fix) / 버그 수정
- [ ] Feature / 기능 추가
- [ ] Model upgrade (turbulence, viscosity/rheology, heat transfer, vibration/structural dynamics; accepted issue required) / 모델 업그레이드
- [ ] Validation case / 검증 사례
- [ ] Performance only (outputs byte-identical) / 성능 개선
- [ ] Documentation or tooling / 문서·도구

## Files changed / 바뀐 파일

<!-- Every file, one line each, with what changed in it. -->

## Verification / 검증

<!-- Tests by NAME and COUNT, before and after. Paste the result lines. -->

| Suite / command | Before | After |
|---|---|---|
| `cargo test --release --lib -- <filter>` | | |
| `cargo test --release --lib -- xref provenance_audit` | | |
| `ofgpu-validate` sections touched | | |
| Python selftests touched (`tools/*/selftest.py`) | | |
| Studio tests touched (`npm run test -w ...`) | | |

- [ ] For a bug fix: I saw the new test fail before the fix. Failure text:
- [ ] For "output only" or "performance only": every mesh and field is byte-identical (show the comparison).

Known failures that may appear and are not caused by this PR: `ofgpu-validate` Gate 95-A 5:1 and 10:1 (4 rows) and the Gate 95-G ring order row; studio `src/ontology/import.test.ts` "at least 221 runs". Any other failure must be explained.

## Numerics / 수치

- [ ] This PR changes no solver numerics and no published number.
- [ ] This PR changes numerics or published numbers, as accepted in the linked issue. Every number that moved, old and new:

## Rules / 규칙

- [ ] No gate, tolerance or existing assertion was loosened or removed.
- [ ] New equations, models and constants are in `rust/SPEC-LIT.md` with open references.
- [ ] New source files have the repository header and a row in `rust/PROVENANCE.md`.
- [ ] No GPL, LGPL or AGPL source was read or copied. Permissive code that was copied is listed in `NOTICE`.
- [ ] Every commit is signed off (`git commit -s`), and I agree to license this contribution to Iterations Co., Ltd. under the Apache License 2.0 (CONTRIBUTING.md §6).

## AI assistance / AI 사용

<!-- Which agent and model wrote or reviewed code (or "none"), and what a human checked by hand. -->

## Contributor grant / 기여자 권한

<!-- Optional. See CONTRIBUTOR-GRANTS.md. Maintainers decide the tier and record it in GRANTS.md. -->

- Tier requested: none / significant (12 months) / model upgrade (perpetual)
- Grantee (your name or the legal entity):
