#!/usr/bin/env bash
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
# Steady mesh test of pool_source_QCDC_outer_to_R22p1: air at 1 m/s straight up from the ring
# inlet, ambient temperature (no buoyancy), wind 10 m/s from the east; the numerics that the
# tet meshes tolerated (upwind, uncorrected, no non-orthogonal correction, U 0.3 / p 0.1).
#   run_air.sh [iters] [write_every]
set -uo pipefail
ITERS="${1:-400}"
WRITE="${2:-$ITERS}"
CASE="$(cd "$(dirname "$0")" && pwd)"
SOLVER="/c/Users/sdd32/Documents/GitHub/Iteration-CFD/rust/target/release/ofgpu-buoyant.exe"
cd "$CASE"
echo "case  : $CASE"
echo "solver: $SOLVER"
echo "iters : $ITERS  (foam fields written every $WRITE iterations)"
start=$(date +%s)
"$SOLVER" "$CASE" -iters "$ITERS" -check 1 -nCorrectors 1 -output foam -restartWrite "$WRITE" 2>&1 | tee "$CASE/run.log"
rc=${PIPESTATUS[0]}
echo "solver exit $rc after $(( $(date +%s) - start )) s" | tee -a "$CASE/run.log"
echo "solver exit $rc" > "$CASE/run.exit"
echo "(this window stays open; close it when done)"
exec bash
