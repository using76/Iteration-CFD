#!/usr/bin/env bash
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
#
# One site case from its config to the solver and to Fluent:
#   STEP -> cut -> pools -> checkpoint -> trim -> mesh (run_step_mesh.cmd), the fluid solid of
#   the meshed domain exported as STEP as soon as it exists, then polyMesh + Fluent mesh with
#   every point's inlet patch typed velocity-inlet.
#
#   mesh_case.sh <case.json> [--from-checkpoint] [--no-geometry] [--no-fluent]
#
# Layout (the case directory is the config's directory; out_dir is normally <case dir>/mesh):
#   <case dir>/<name>.json              the recipe (tools/mesh/examples/pool_three_inlets.json)
#   <out_dir>/<name>.msh .vtk _summary.json, work/ (checkpoints, run.log)
#   <case dir>/geometry/fluid_<name>.step        the fluid solid as meshed (after the trim), mm
#   <case dir>/geometry/fluid_<name>_full.step   the same before the trim (from the checkpoint), mm
#   <case dir>/case/constant/polyMesh            for the in-house solver (ofgpu-buoyant)
#   <case dir>/<name>_fluent.msh                 Fluent: File > Read > Mesh (inlets velocity-inlet,
#                                                wall* wall, the rest pressure-outlet; east velocity-inlet)
#   <case dir>/convert.log, mesh.exit            the converter's report; "step_mesh exit N" + "convert exit N"
#
# Launch it in its own console from Bash (cmd //c start "" bash mesh_case.sh <case.json>) so the
# stage banners stay visible; keep every path ASCII (Fluent reads no other). OFGPU_CONVERT can
# point at another converter binary.
set -uo pipefail
CFG="${1:?usage: mesh_case.sh <case.json> [--from-checkpoint] [--no-geometry] [--no-fluent]}"; shift
FROM_CK=""; GEOM=1; FLUENT=1
for a in "$@"; do
  case "$a" in
    --from-checkpoint) FROM_CK="--from-checkpoint" ;;
    --no-geometry) GEOM=0 ;;
    --no-fluent) FLUENT=0 ;;
    *) echo "mesh_case.sh: unknown option $a"; exit 2 ;;
  esac
done
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
TOOL="$HERE/run_step_mesh.cmd"
EXPORT="$HERE/diag/export_fluid.py"
CONV="${OFGPU_CONVERT:-$REPO/rust/target/release/ofgpu-convert-mesh.exe}"
[ -x "$CONV" ] || { echo "mesh_case.sh: converter not found: $CONV (cargo build --release -p ofgpu --bin ofgpu-convert-mesh)"; exit 2; }
CASEDIR="$(cd "$(dirname "$CFG")" && pwd)"
CFGW="$(cygpath -w "$CFG" 2>/dev/null || echo "$CFG")"

cfg() { python -c "import json,sys; d=json.load(open(sys.argv[1],encoding='utf-8-sig')); print($1)" "$CFG"; }
NAME="$(cfg "d['name']")"
OUT="$(cfg "d['out_dir']")"
# every point's inlet patch (classification.pool_prefix + point name) is a velocity inlet
ZONES="$(cfg "' '.join('-fluentType %s%s=velocity-inlet' % (d.get('classification',{}).get('pool_prefix','pool_'), p) for p in d.get('points',{}))")"
OUTU="$(cygpath -u "$OUT" 2>/dev/null || echo "$OUT")"
W="$OUTU/work"
mkdir -p "$W"
rm -f "$CASEDIR/mesh.exit"
# a run from the STEP rewrites the checkpoint and the trim; drop the old files so the geometry
# export waits for this run's (a stale brep would look "stable" at once)
[ -z "$FROM_CK" ] && rm -f "$W/${NAME}_pools.brep" "$W/${NAME}_pools.json" "$W/${NAME}_cut.brep"
[ $GEOM = 1 ] && rm -f "$W/${NAME}_trimmed.brep"

echo "=== step_mesh $NAME ${FROM_CK:-from the STEP}"
cmd //c "$(cygpath -w "$TOOL" 2>/dev/null || echo "$TOOL")" "$CFGW" $FROM_CK &
MESH=$!

# wait until a brep exists and has stopped growing (the mesher writes it in one go, but the
# export must not read a half-written file); give up when the mesher is gone
wait_stable() {
  local f="$1" a=-1 b
  while [ ! -f "$f" ]; do kill -0 "$MESH" 2>/dev/null || return 1; sleep 10; done
  while true; do
    b=$(stat -c %s "$f"); [ "$a" = "$b" ] && return 0; a=$b; sleep 15
  done
}
if [ $GEOM = 1 ]; then
  mkdir -p "$CASEDIR/geometry"
  if [ -z "$FROM_CK" ] && wait_stable "$W/${NAME}_pools.brep"; then
    python "$EXPORT" "$W/${NAME}_pools.brep" "$CASEDIR/geometry/fluid_${NAME}_full" --no-stl
  fi
  if wait_stable "$W/${NAME}_trimmed.brep"; then
    python "$EXPORT" "$W/${NAME}_trimmed.brep" "$CASEDIR/geometry/fluid_${NAME}" --no-stl
    echo "=== geometry: $CASEDIR/geometry/fluid_${NAME}.step"
  fi
fi

wait "$MESH"; rc=$?
echo "step_mesh exit $rc" > "$CASEDIR/mesh.exit"
MSH="$OUTU/$NAME.msh"
if [ $rc -ne 0 ] || [ ! -f "$MSH" ]; then echo "=== mesh failed (exit $rc); see $W/run.log"; exit 1; fi

echo "=== convert $NAME -> polyMesh${FLUENT:+ + Fluent}"
rm -rf "$CASEDIR/case"
FL=""; [ $FLUENT = 1 ] && FL="-fluent $CASEDIR/${NAME}_fluent.msh"
"$CONV" "$MSH" "$CASEDIR/case" $FL $ZONES 2>&1 | tee "$CASEDIR/convert.log"
crc=${PIPESTATUS[0]}
echo "convert exit $crc" >> "$CASEDIR/mesh.exit"
echo "=== done: $(tr '\n' ' ' < "$CASEDIR/mesh.exit")"
exit "$crc"
