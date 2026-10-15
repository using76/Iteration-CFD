// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// The decomposition report of an ofgpu-par run, read from its log. Every
// pattern below is lifted from a `println!` in rust/src/bin/par.rs `run()`
// (SPEC-LIT 138.8). ofgpu-par is a gate driver, not a flow solver: per rank it
// prints the Krylov iteration count beside the whole mesh's, and the worst
// |d| over the ranks, and those are what this report carries. Unknown lines
// are ignored, so the parser never throws on a log.

export interface ParMesh {
  cells: number
  internalFaces: number
  boundaryFaces: number
}

/** One whole-mesh Krylov solve: the reference every rank's solve must equal bit for bit. */
export interface ParReference {
  label: string
  iterations: number
  tolerance: number
  finalResidual: number
}

/** One Krylov gate line of one rotation. */
export interface ParKrylov {
  label: string
  ok: boolean
  /** Owned cells whose solution differs from the whole mesh's. */
  differ: number
  /** The worst |d| over every rank. */
  worst: number
  /** Each rank's iteration count, in rank order. */
  iterations: number[]
  /** The whole mesh's iteration count. */
  whole: number
}

/** One label rotation of one (rank count, partitioner) world. */
export interface ParRotation {
  np: number
  method: string
  rot: number
  mapOk: boolean
  /** The partition-map hash every rank allgathered. */
  hash: string
  owned: number | null
  ghosts: number | null
  haloOk: boolean | null
  reductionOk: boolean | null
  krylov: ParKrylov[]
}

export interface ParReport {
  mesh: ParMesh | null
  /** The rank counts the run was asked for (`comm` line). */
  ranks: number[]
  transport: string | null
  references: ParReference[]
  rotations: ParRotation[]
  /** The driver's last line; null while the run is still going. */
  verdict: { ok: boolean; text: string } | null
}

const NUM = String.raw`[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?`
const RE_MESH = /^mesh\s+:\s+(\d+) cells, (\d+) internal faces, (\d+) boundary faces/
const RE_COMM = /^comm\s+:\s+(\S+)\s+ranks \[([\d,\s]*)\]/
const RE_REF = new RegExp(String.raw`^(\S+)\s+whole mesh: (\d+) iterations to (${NUM}), final residual (${NUM})`)
const RE_ROT = /^P = (\d+)\s+(\S+)\s+rot (\d+)\s+(\S+)\s+(PASS|FAIL) - (.*)$/
const RE_MAP = /rank hashes ([0-9a-f]+)/
const RE_HALO = /on (\d+) owned \+ (\d+) ghost cells/
const RE_KRYLOV = new RegExp(String.raw`^(\d+) of \d+ cells differ, worst \|d\| (${NUM}); iterations \[([\d,\s]*)\] \(whole mesh (\d+)\)`)
const RE_VERDICT = /^ofgpu-par: (PASS|FAIL) - (.*)$/

const ints = (s: string): number[] =>
  s
    .split(',')
    .map((x) => x.trim())
    .filter((x) => x.length > 0)
    .map(Number)

export function parseParReport(lines: Iterable<string>): ParReport {
  const report: ParReport = { mesh: null, ranks: [], transport: null, references: [], rotations: [], verdict: null }
  const byKey = new Map<string, ParRotation>()
  for (const raw of lines) {
    const line = raw.replace(/\r$/, '').trim()
    if (!line) continue
    let m: RegExpMatchArray | null
    if ((m = line.match(RE_MESH))) {
      report.mesh = { cells: Number(m[1]), internalFaces: Number(m[2]), boundaryFaces: Number(m[3]) }
      continue
    }
    if ((m = line.match(RE_COMM))) {
      report.transport = m[1]
      report.ranks = ints(m[2])
      continue
    }
    if ((m = line.match(RE_REF))) {
      report.references.push({ label: m[1], iterations: Number(m[2]), tolerance: Number(m[3]), finalResidual: Number(m[4]) })
      continue
    }
    if ((m = line.match(RE_VERDICT))) {
      report.verdict = { ok: m[1] === 'PASS', text: m[2] }
      continue
    }
    if (!(m = line.match(RE_ROT))) continue
    const [, np, method, rot, gate, verdict, rest] = m
    const key = `${np}/${method}/${rot}`
    let r = byKey.get(key)
    if (!r) {
      r = { np: Number(np), method, rot: Number(rot), mapOk: false, hash: '', owned: null, ghosts: null, haloOk: null, reductionOk: null, krylov: [] }
      byKey.set(key, r)
      report.rotations.push(r)
    }
    const ok = verdict === 'PASS'
    if (gate === 'MAP') {
      r.mapOk = ok
      r.hash = rest.match(RE_MAP)?.[1] ?? ''
    } else if (gate === 'HALO') {
      r.haloOk = ok
      const h = rest.match(RE_HALO)
      if (h) {
        r.owned = Number(h[1])
        r.ghosts = Number(h[2])
      }
    } else if (gate === 'REDUCTION') {
      r.reductionOk = ok
    } else {
      const k = rest.match(RE_KRYLOV)
      if (k) r.krylov.push({ label: gate, ok, differ: Number(k[1]), worst: Number(k[2]), iterations: ints(k[3]), whole: Number(k[4]) })
    }
  }
  return report
}

/** Whether every gate of a rotation passed (a gate not yet printed counts as not passed). */
export function parRotationOk(r: ParRotation): boolean {
  return r.mapOk && r.haloOk === true && r.reductionOk === true && r.krylov.length > 0 && r.krylov.every((k) => k.ok)
}
