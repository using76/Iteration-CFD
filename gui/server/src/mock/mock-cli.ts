// Mock ofgpu CLI: `node --import <tsx loader> mock-cli.ts <binary> <args...>`.
// A real process with the real drivers' argv rules, log formats, exit codes
// and result files, so the runner, parser, kill path and dataset readers
// are exercised without a GPU. CFD_MOCK_SPEED scales the pacing,
// CFD_MOCK_FAIL=1 forces a SPEC-LIT §13.4 refusal.
import fsp from 'node:fs/promises'
import path from 'node:path'
import { MESH_KINDS, type MeshKind } from '@cfd/shared'
import { parseMockArgs, usageLine, UsageError, type MockArgs } from './args.js'
import { g, sci, sleep } from './format.js'
import { defaultCells, presetSpec, writeMockCase } from './mesh.js'
import { DEMO_DEVICE, runMockSolver, type SolverEnv, type SolverIo } from './solver.js'

const io: SolverIo = {
  out: (line) => process.stdout.write(`${line}\n`),
  err: (line) => process.stderr.write(`${line}\n`),
}

const env: SolverEnv = {
  speed: Number(process.env.CFD_MOCK_SPEED ?? 1) || 1,
  fail: process.env.CFD_MOCK_FAIL === '1',
  workspace: process.env.CFD_WORKSPACE ?? null,
}

const pace = (units: number) => sleep((units * 11.25) / Math.max(env.speed, 1e-3))

async function generateMesh(args: MockArgs): Promise<number> {
  const kind = args.positionals[0] as MeshKind
  if (!MESH_KINDS.includes(kind)) {
    io.err('')
    io.err(`error: "${kind}" is not supported by ofgpu; available: ${MESH_KINDS.join(', ')}`)
    return 1
  }
  const dir = args.positionals[1]
  const nums = args.positionals.slice(2).map(Number)
  let cells = defaultCells(kind)
  if (kind === 'big' && nums.length === 1) cells = [nums[0], nums[0], nums[0]]
  else if (nums.length === 3) cells = [nums[0], nums[1], nums[2]]
  else if (nums.length !== 0) {
    io.err('')
    io.err(`error: expected nx ny nz, got ${args.positionals.slice(2).join(' ')}`)
    return 1
  }
  if (cells.some((n) => !Number.isInteger(n) || n < 1)) {
    io.err('')
    io.err(`error: cell counts must be positive integers, got ${cells.join(' ')}`)
    return 1
  }
  for (const stl of args.flags.get('-stl') ?? []) io.out(`[stl] ${stl.replace(/^[^=]+=/, '')}: 0 triangle(s), 1 patch(es)`)
  const preset = presetSpec(kind, cells, { cyclic: args.flags.get('-cyclic') })
  const dirAbs = path.isAbsolute(dir) ? dir : path.resolve(dir)
  await fsp.mkdir(dirAbs, { recursive: true })
  const wallModel = args.str('-wallModel')
  const res = await writeMockCase(dirAbs, preset, io.out, { wallModel })
  await pace(Math.min(200, res.cells / 1000))
  io.out(`mesh: ${res.cells} cells, ${res.faces} faces`)
  io.out(`${kind}: ${cells[0]} x ${cells[1]} x ${cells[2]} = ${res.cells} cells -> ${dir}`)
  io.out(`written to ${dir}`)
  return 0
}

async function validate(): Promise<number> {
  const gates = [
    ['S3.1 orthogonal block: div(grad phi) = 0 to 1e-12', 'pass'],
    ['S3.2 sheared block: non-orthogonal correction', 'pass'],
    ['S4.1 empty front/back: 2-D equivalence', 'pass'],
    ['S5.1 PBiCGStab reaches 1e-10 on the Poisson gate', 'pass'],
    ['S5.2 PCG/DIC symmetric system', 'pass'],
    ['S6.1 k-epsilon channel log law y+ 30..300', 'pass'],
    ['S6.2 k-omega channel log law', 'pass'],
    ['S6.3 SST blending F1/F2 limits', 'pass'],
    ['S9.1 buoyancy: hydrostatic balance to 1e-9', 'pass'],
    ['S13.4 unsupported setting refused with a list', 'pass'],
    ['S18.1 volumetric source integral', 'pass'],
    ['S23.1 castellated carve: cell count conserved', 'pass'],
    ['S24.1 cut-cell closure: volumes sum to the box', 'pass'],
    ['S25.1 low-Mach reference pressure', 'pass'],
    ['S26.1 energy budget closes to 1e-6', 'pass'],
    ['S29.1 wall functions: Ks -> 0 recovers smooth', 'pass'],
    ['S29.3 thermal wall function', 'pass'],
    ['S31.1 cyclic pairs: translation invariants', 'pass'],
    ['S32.1 the thermal wall-function gate', 'pass'],
    ['S33.3 Launder-Sharma damping functions', 'pass'],
    ['S35.1 bulk-temperature thermostat', 'pass'],
    ['S37.1 Kays-Crawford Prt', 'pass'],
    ['S40.1 realizable k-epsilon C_mu field', 'pass'],
    ['S41.1 RNG k-epsilon diffusivity', 'pass'],
    ['S46.1 conjugate interface flux continuity', 'pass'],
    ['S52.1 fan curve operating point', 'pass'],
    ['S53.2 porous jump dp', 'pass'],
    ['S56.1 Spalart-Allmaras flat plate', 'pass'],
    ['S57.1 DES length scale switch', 'pass'],
    ['S69.1 the gate registry audit', 'pass'],
  ]
  io.out(`ofgpu validate | ${DEMO_DEVICE} sm_89 | 24564 MiB | precision double`)
  io.out('')
  io.out('=== 3-D graded block ===')
  for (const [what, verdict] of gates) {
    io.out(`  ${verdict}  ${what.padEnd(52)}${sci(1e-12 * Math.random(), 2)}`)
    await pace(12)
  }
  io.out('')
  io.out('901/901 checks passed')
  io.out('895 computed live, 6 replayed from recorded measurements')
  io.out('901 / 901 checks passed')
  return 0
}

async function probe(): Promise<number> {
  io.out(`device : ${DEMO_DEVICE}`)
  io.out('scalar : f64')
  io.out('')
  io.out('  cell        gpu                 cpu              diff')
  for (let c = 0; c < 4; c++) io.out(`  ${String(c).padStart(4)}   ${(1 + c * 0.25).toFixed(12).padStart(16)}   ${(1 + c * 0.25).toFixed(12).padStart(16)}   0.000e0`)
  io.out('')
  io.out('max |gpu - cpu| = 0.000e0')
  io.out('PASS - bitwise identical')
  return 0
}

async function bench(args: MockArgs): Promise<number> {
  const [nx, ny, nz] = args.positionals.map(Number)
  const cells = nx * ny * nz
  const iters = Math.max(1, Math.floor(args.num('-iters', 200)))
  io.out(`ofgpu benchmark | ${DEMO_DEVICE} sm_89 | 128 SMs | 384-bit, 1008 GB/s peak | double`)
  io.out(`building ${nx} x ${ny} x ${nz} = ${cells} cells ...`)
  await pace(20)
  io.out(`       ${cells} cells, ${3 * cells - nx * ny - ny * nz - nx * nz} internal faces, ${2 * (nx * ny + ny * nz + nx * nz)} boundary faces`)
  io.out('       max |sum_f phi| per cell = 7e-18')
  io.out(`       mesh only: ${Math.round(cells * 0.0006)} MiB resident of 24564 MiB`)
  for (const model of [args.str('-model') ?? 'kEpsilon']) {
    await pace(iters)
    const wall = (iters * cells) / 1.6e9
    io.out(`${model}: ${iters} iterations in ${sci(wall)} s  ->  ${sci((wall / iters) * 1e3)} ms/iteration  ->  ${sci((cells * iters) / wall / 1e6)} Mcell-iterations/s`)
  }
  return 0
}

async function graphBench(): Promise<number> {
  io.out(`ofgpu graph-bench | ${DEMO_DEVICE} sm_89`)
  await pace(50)
  io.out('eager   : 200 launches in 4.120e-02 s  ->  2.060e-01 ms/launch')
  io.out('graph   : 200 replays  in 1.310e-02 s  ->  6.550e-02 ms/launch')
  io.out('speed-up: 3.15x')
  return 0
}

async function dispatchBench(): Promise<number> {
  io.out(`ofgpu dispatch-bench | ${DEMO_DEVICE} sm_89`)
  await pace(20)
  io.out('empty kernel launch: 2.4 us   memcpy 4 B round trip: 11.8 us   event sync: 5.1 us')
  return 0
}

async function decompose(args: MockArgs): Promise<number> {
  const parts = (args.str('-parts') ?? '2,3,4').split(',').map((s) => Number(s.trim())).filter((n) => n > 0)
  const method = args.str('-method') ?? 'hilbert'
  const quiet = args.has('-quiet')
  if (!quiet) io.out(`ofgpu decompose | ${DEMO_DEVICE} sm_89 | case ${args.positionals[0]}`)
  for (const p of parts) {
    await pace(30)
    if (!quiet) io.out(`  ${method} ${p} parts: halo ${g(0.031 * p)} of cells, sum invariant to ${sci(2e-16, 1)}, ${args.num('-sweeps', 8)} Jacobi sweeps agree to ${sci(1e-15, 1)}`)
  }
  io.out(`verdict: PASS (${parts.length} partitionings, ${method})`)
  return 0
}

async function main(): Promise<number> {
  const [binary, ...rest] = process.argv.slice(2)
  if (!binary) {
    io.err('usage: mock-cli.ts <ofgpu-binary> [args...]')
    return 2
  }
  let args: MockArgs
  try {
    args = parseMockArgs(binary, rest)
  } catch (err) {
    if (err instanceof UsageError) {
      if (err.spec) io.err(usageLine(err.spec))
      io.err('')
      io.err(`error: ${err.message}`)
      return 1
    }
    throw err
  }
  switch (binary) {
    case 'ofgpu-generate-mesh':
      return generateMesh(args)
    case 'ofgpu-validate':
      return validate()
    case 'ofgpu-probe':
      return probe()
    case 'ofgpu-bench':
      return bench(args)
    case 'ofgpu-graph-bench':
      return graphBench()
    case 'ofgpu-dispatch-bench':
      return dispatchBench()
    case 'ofgpu-decompose':
      return decompose(args)
    default:
      return runMockSolver(args, io, env)
  }
}

process.on('SIGTERM', () => {
  process.stderr.write('\nterminated\n')
  process.exit(143)
})
process.on('SIGINT', () => process.exit(130))

main().then(
  (code) => {
    process.exitCode = code
  },
  (err: unknown) => {
    io.err('')
    io.err(`error: ${(err as Error).message}`)
    process.exitCode = 1
  },
)
