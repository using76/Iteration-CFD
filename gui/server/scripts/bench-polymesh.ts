// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// Reported benchmark (STU-11), not a CI bound: how long the studio takes to load a large ASCII
// polyMesh, and the peak resident memory it needs. For each requested cell count a uniform hex
// box is written once by writePolyMesh (the format rust/src/io/polymesh.rs writes), then a fresh
// child process runs the dataset worker's polyMesh task (readPolyMesh, cell centres, boundary
// surface, lattice detection) and reports the load seconds and its peak RSS.
//
// Run from gui/: npx tsx server/scripts/bench-polymesh.ts [cells ...]   (default 1000000 3000000)
// BENCH_POLYMESH_DIR picks the scratch folder (default <os tmp>/cfd-bench-polymesh); a mesh already
// there is reused, and --keep leaves the meshes on disk afterwards.
import { fork } from 'node:child_process'
import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { buildCartesianGrid, cartesianToPolyMesh, type CartesianSpec } from '../src/formats/cartesian.js'
import { writePolyMesh } from '../src/formats/polymesh.js'
import { runTask } from '../src/datasets/worker.js'

interface LoadReport {
  nCells: number
  nPoints: number
  surfaceTriangles: number
  lattice: boolean
  seconds: number
  rssBeforeMiB: number
  peakRssMiB: number
}

/** Axis counts whose product is close to `cells`: n x n x round(cells / n^2), n = round(cbrt(cells)). */
function dimsFor(cells: number): [number, number, number] {
  const n = Math.max(1, Math.round(Math.cbrt(cells)))
  return [n, n, Math.max(1, Math.round(cells / (n * n)))]
}

function specFor(dims: [number, number, number]): CartesianSpec {
  return {
    bounds: { min: [0, 0, 0], max: [1, 1, 1] },
    cells: dims,
    grading: null,
    boundaries: { xmin: 'inlet', xmax: 'outlet', ymin: 'floor', ymax: 'ceiling', zmin: 'sideMin', zmax: 'sideMax' },
    regions: [],
    cyclic: [],
    patchTypes: { inlet: 'patch', outlet: 'patch', floor: 'wall', ceiling: 'wall', sideMin: 'wall', sideMax: 'wall' },
  }
}

async function exists(p: string): Promise<boolean> {
  return fs.stat(p).then(
    () => true,
    () => false,
  )
}

async function ensureMesh(root: string, dims: [number, number, number]): Promise<string> {
  const dir = path.join(root, `box-${dims.join('x')}`, 'constant', 'polyMesh')
  if (await exists(path.join(dir, 'neighbour'))) return dir
  const t0 = performance.now()
  const mesh = cartesianToPolyMesh(buildCartesianGrid(specFor(dims)), specFor(dims))
  await writePolyMesh(dir, mesh)
  console.log(`  wrote ${dims.join('x')} (${mesh.nCells} cells, ${mesh.nFaces} faces) in ${((performance.now() - t0) / 1000).toFixed(1)} s`)
  return dir
}

/** Child side: one load, measured in a process that has done nothing else. */
async function loadOnce(dir: string): Promise<void> {
  const mib = (kib: number): number => Math.round(kib / 1024)
  const rssBefore = process.memoryUsage().rss / 1024
  const t0 = performance.now()
  const r = await runTask({ op: 'polyMesh', dir })
  const seconds = (performance.now() - t0) / 1000
  if (r.op !== 'polyMesh') throw new Error(`unexpected result ${r.op}`)
  const report: LoadReport = {
    nCells: r.nCells,
    nPoints: r.nPoints,
    surfaceTriangles: r.surface.indices.length / 3,
    lattice: r.lattice !== null,
    seconds,
    rssBeforeMiB: mib(rssBefore),
    // resourceUsage().maxRSS is in kibibytes on every platform Node supports.
    peakRssMiB: mib(process.resourceUsage().maxRSS),
  }
  // Disconnect after the send, or the IPC channel keeps the child alive.
  process.send?.(report, () => process.disconnect())
}

function measure(dir: string): Promise<LoadReport> {
  return new Promise((resolve, reject) => {
    const child = fork(fileURLToPath(import.meta.url), ['--load', dir], { stdio: 'inherit' })
    let report: LoadReport | null = null
    child.on('message', (m) => (report = m as LoadReport))
    child.on('error', reject)
    child.on('exit', (code) => (report && code === 0 ? resolve(report) : reject(new Error(`load child exited ${code}`))))
  })
}

async function main(argv: string[]): Promise<void> {
  if (argv[0] === '--load') return loadOnce(argv[1])
  const keep = argv.includes('--keep')
  const sizes = argv.filter((a) => !a.startsWith('--')).map(Number)
  const cells = sizes.length ? sizes : [1_000_000, 3_000_000]
  if (cells.some((c) => !Number.isInteger(c) || c < 1)) throw new Error(`cell counts must be positive integers, got ${sizes.join(' ')}`)
  const root = process.env.BENCH_POLYMESH_DIR ?? path.join(os.tmpdir(), 'cfd-bench-polymesh')
  console.log(`bench-polymesh: node ${process.version}, ${os.cpus().length} CPUs, ${Math.round(os.totalmem() / 2 ** 30)} GiB, scratch ${root}`)
  const rows: string[] = []
  for (const c of cells) {
    const dims = dimsFor(c)
    const dir = await ensureMesh(root, dims)
    const r = await measure(dir)
    const row = `${dims.join('x')}  cells ${r.nCells}  points ${r.nPoints}  surface tris ${r.surfaceTriangles}  lattice ${r.lattice}  load ${r.seconds.toFixed(2)} s  peak RSS ${r.peakRssMiB} MiB (${r.rssBeforeMiB} MiB before)`
    console.log(row)
    rows.push(row)
    if (!keep) await fs.rm(path.join(root, `box-${dims.join('x')}`), { recursive: true, force: true })
  }
  console.log(`\nsummary\n${rows.join('\n')}`)
}

main(process.argv.slice(2)).catch((e: unknown) => {
  console.error(e instanceof Error ? e.message : e)
  process.exit(1)
})
