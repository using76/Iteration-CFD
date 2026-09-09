#!/usr/bin/env bash
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
# Remesh one pool case from its checkpoint, convert it (polyMesh + Fluent), then run the
# 6-iteration steady air test on it (fields written every 3 iterations).
#   remesh_and_test.sh <case> [iters] [write_every]
set -uo pipefail
N="${1:?case name}"; ITERS="${2:-6}"; WRITE="${3:-3}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
TOOL="C:/Users/sdd32/Documents/GitHub/Iteration-CFD/tools/mesh/run_step_mesh.cmd"
CONV="$ROOT/ofgpu-convert-mesh.exe"
SOLVER="/c/Users/sdd32/Documents/GitHub/Iteration-CFD/rust/target/release/ofgpu-buoyant.exe"
ZONES="-fluentType $N=velocity-inlet"
[ "$N" = "pool_source_QCDC_outer_to_R22p1" ] && ZONES="-fluentType pool_source_QCDC_inner_R10p5=velocity-inlet -fluentType $N=velocity-inlet"
cd "$ROOT"
echo "=== remesh $N from its checkpoint"
cmd //c "$TOOL" "$ROOT/$N/$N.json" --from-checkpoint
rc=$?; echo "step_mesh exit $rc" > "$N/mesh.exit"
if [ $rc -ne 0 ]; then echo "mesh failed ($rc)"; exec bash; fi
echo "=== convert"
rm -rf "$N/case"
"$CONV" "$N/mesh/$N.msh" "$N/case" -fluent "$N/${N}_fluent.msh" $ZONES 2>&1 | tee "$N/convert.log"
echo "convert exit ${PIPESTATUS[0]}" >> "$N/mesh.exit"
echo "=== steady air test: $ITERS iterations"
C="$ROOT/$N/case"
cp "$ROOT/make_case_air.py" "$ROOT/run_air.sh" "$C/" 2>/dev/null
python "$C/make_case_air.py" "$C" "$ITERS" | tail -1
sed -i "s/^writeInterval   $ITERS;/writeInterval   $WRITE;/" "$C/system/controlDict"
rm -f "$C/run.exit" "$C/run.log"
cd "$C"
"$SOLVER" "$C" -iters "$ITERS" -check 1 -nCorrectors 1 -output foam -restartWrite "$WRITE" 2>&1 | tee "$C/run.log"
echo "solver exit ${PIPESTATUS[0]}" | tee "$C/run.exit"
echo "(done; this window stays open)"
exec bash
