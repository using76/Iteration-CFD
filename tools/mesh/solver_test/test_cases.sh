#!/usr/bin/env bash
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
#
# Steady air tests on several converted cases, one after another (each run holds the GPU):
#   test_cases.sh <iters> <write_every> <caseDir> [<caseDir> ...]
# <caseDir> holds constant/polyMesh (from ofgpu-convert-mesh). For each case make_case_air.py
# writes 0/ and system/ (air 1 m/s from the source patch, wind 10 m/s, no buoyancy), the solver
# runs <iters> SIMPLE iterations writing foam fields every <write_every>, and the outcome lands in
# <caseDir>/run.log and run.exit; tests.done next to the first case marks the end. Judge each run
# with diag/field_extremes.py <caseDir> <iters> <pool_x> <pool_y>: |U| max within tens of m/s,
# few |p| > 1000 cells, and the extremes away from the source.
set -uo pipefail
ITERS="${1:?iters}"; WRITE="${2:?write_every}"; shift 2
[ $# -ge 1 ] || { echo "test_cases.sh <iters> <write_every> <caseDir>..."; exit 2; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
SOLVER="${OFGPU_SOLVER:-$REPO/rust/target/release/ofgpu-buoyant.exe}"
[ -x "$SOLVER" ] || { echo "solver not found: $SOLVER"; exit 2; }
DONE="$(cd "$(dirname "$1")" && pwd)/tests.done"
rm -f "$DONE"
for C in "$@"; do
  C="$(cd "$C" && pwd)"
  echo "=== $C: $ITERS iterations (fields every $WRITE)"
  rm -rf "$C/0" "$C/system" "$C"/[1-9]*
  python "$HERE/make_case_air.py" "$C" "$ITERS" | tail -1
  sed -i "s/^writeInterval   $ITERS;/writeInterval   $WRITE;/" "$C/system/controlDict"
  rm -f "$C/run.exit" "$C/run.log"
  start=$(date +%s)
  ( cd "$C" && "$SOLVER" "$C" -iters "$ITERS" -check 1 -nCorrectors 1 -output foam -restartWrite "$WRITE" 2>&1 | tee "$C/run.log"
    echo "solver exit ${PIPESTATUS[0]} after $(( $(date +%s) - start )) s" | tee "$C/run.exit" )
done
echo done > "$DONE"
echo "=== all tests finished"
