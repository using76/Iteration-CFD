// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// The mock ofgpu-par: the lines rust/src/bin/par.rs `run()` prints, in its
// exact shapes (SPEC-LIT 138.8), for demo mode. Every gate passes and every
// rank stops on the whole mesh's iteration count, as the real driver does on
// cases/plume.jsonc; the hashes are made up, so a demo log never passes for a
// measurement (the device line says "(demo)").
import type { MockArgs } from './args.js'
import { DEMO_DEVICE } from './solver.js'

const MESH = { cells: 82320, internalFaces: 240044, boundaryFaces: 13832 }
const KRYLOV: Array<{ label: string; iterations: number; residual: string }> = [
  { label: 'PCG', iterations: 125, residual: '9.268e-11' },
  { label: 'PBiCGStab', iterations: 79, residual: '8.914e-11' },
]
const METHODS = ['hilbert', 'linear', 'roundrobin'] as const

const pad = (s: string, n: number): string => s.padEnd(n, ' ')

/** FNV-1a over the world's name, 64 bits, as 16 hex digits: a stand-in for the partition-map hash. */
function demoHash(key: string): string {
  let h = 0xcbf29ce484222325n
  for (const c of key) h = BigInt.asUintN(64, (h ^ BigInt(c.charCodeAt(0))) * 0x100000001b3n)
  return h.toString(16).padStart(16, '0')
}

/** par.rs `refuse_unbuilt_transport`'s refusal for a -comm other than threads, word for word; null for threads. */
export function mockParRefusal(args: MockArgs): string | null {
  const comm = args.str('-comm') ?? 'threads'
  if (comm === 'threads') return null
  if (comm === 'mpi') return '-comm mpi: the MPI transport is deferred (PAR-06) and is not built. Use -comm threads'
  return `-comm '${comm}': unknown transport; known: threads (mpi is deferred, PAR-06)`
}

/** The whole log of one mock ofgpu-par run (a -comm refusal is mockParRefusal's). */
export function mockParLines(args: MockArgs): string[] {
  const nps = (args.str('-np') ?? '2,3,4')
    .split(',')
    .map((s) => Number(s.trim()))
    .filter((n) => Number.isInteger(n) && n > 0)
  const method = args.str('-method') ?? 'all'
  const methods = method === 'all' ? [...METHODS] : [method]
  const sweeps = args.num('-sweeps', 8)
  const quiet = args.has('-quiet')
  const out: string[] = []
  if (!quiet) {
    out.push(`case   : ${args.positionals[0]}`)
    out.push(`device : ${DEMO_DEVICE} (placement [0])`)
    out.push(`mesh   : ${MESH.cells} cells, ${MESH.internalFaces} internal faces, ${MESH.boundaryFaces} boundary faces`)
    out.push(`comm   : threads   ranks [${nps.join(', ')}]   sweeps ${sweeps}`)
    out.push('')
    for (const k of KRYLOV) out.push(`${pad(k.label, 10)} whole mesh: ${k.iterations} iterations to 1e-10, final residual ${k.residual}`)
    out.push('')
  }
  let worlds = 0
  let rotations = 0
  for (const np of nps) {
    for (const name of methods) {
      worlds++
      // P = 2's real ghost count on cases/plume.jsonc (1680) scaled by P - 1: a stand-in, not a measurement.
      const ghosts = 1680 * (np - 1)
      for (let shift = 0; shift < np; shift++) {
        rotations++
        const at = `P = ${np}  ${pad(name, 10)} rot ${shift}`
        out.push(`${at}  MAP       PASS - ${np} rank hashes ${demoHash(`${np}/${name}/${shift}`)}`)
        out.push(`${at}  HALO      PASS - 4 payloads on ${MESH.cells} owned + ${ghosts} ghost cells, 0 differ; ${sweeps} sweeps: 0 of ${MESH.cells} cells differ, worst |d| 0e0`)
        out.push(`${at}  REDUCTION PASS - exact sum and dot of 2 fields: 0 of ${np} ranks x 4 values moved`)
        for (const k of KRYLOV) {
          const iters = Array.from({ length: np }, () => k.iterations).join(', ')
          out.push(`${at}  ${pad(k.label, 9)} PASS - 0 of ${MESH.cells} cells differ, worst |d| 0e0; iterations [${iters}] (whole mesh ${k.iterations})`)
        }
      }
      if (!quiet) {
        out.push(`P = ${np}  ${pad(name, 10)} ${np} rotation(s) in ${(0.2 * np).toFixed(1)} s`)
        out.push('')
      }
    }
  }
  out.push(`ofgpu-par: PASS - ${worlds} worlds, ${rotations} rotations, every gate bit for bit (worst == 0) with the whole mesh's iteration counts, ${(0.4 * rotations).toFixed(1)} s`)
  return out
}
