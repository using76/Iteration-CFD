<!-- meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). Source-available, not Open Source. No GPL-licensed source was consulted. -->

# Seed-2 full scope: withdrawn before running (AUT-01, 2026-10-11)

The seed-2 full scope (180 rows) is withdrawn before running: the binary changed after its lock, and `evaluate.write_lock` refuses the plan under today's binary. This file is the withdrawal record; it re-opens nothing.

- **The lock is unchanged.** `opened.lock` (this directory), file sha256 `55db4a0bc8927fdba37e75091e9dbe6689ad848df4826594af0288729f7c882f`, plan sha256 `60a05a97624b0875fbaa536804748d063bfea28a7a83ad90f63702256bdd1063`, written 2026-10-03 at tree `1b62804` on binary sha256 `3d90ce9156119b16ee6fdd1359c5913152ebc29be3e218ebb903fe70900d923b`.
- **Today's binary differs.** `rust/target/release/ofgpu-automesher.exe` (feat/t17 at `7fd775c`) is sha256 `38a04df1b75a3b74b4f07610ec6f9f98a4d7c19a31ad68674f1f9d1b5cc51ebd`; gates sha256 `8d5fe956…5d13` and knobs sha256 `60a02160…0ba` are unchanged. The lock's plan with today's binary hashes to `ec7eed2dae3f622b9c05a5647c1f4e785d90b85311c75fb82098f8adc816e32d`, so `evaluate.write_lock` refuses a full-scope run ("the test split is spent").
- **What changed after the lock:** the automesher moved on — `57a94fe` (snap freezes only offending points, 92.68), `5423076`, `22b5d3b`, `e4a77c8`, `da0023c`, `07865a4` (face-mode layer stacks) and `8ab1c4a` (the single build), all after `1b62804`.
- **The decision:** the full scope will never run under this lock and seed 2 is not re-opened. The REDUCED page `reduced/EVAL.md` (sha256 `ef7a1cd1be92968c2f249d339501089e8bd3f0f04aaa3b4265738a9362614290`) stays published as measured; six bodies decide nothing and are not a headline. A headline needs a fresh seed under a new lock, spent only on AUT-12's pre-registered go (AUT-13, AUT-14).
