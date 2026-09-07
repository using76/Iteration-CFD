// Parsing of big files runs off the main thread: this module is both the
// worker entry (when loaded by `new Worker`) and the pool the service uses.
// `runTask` is the single implementation, so the pool can fall back to
// running a task inline when no worker can be started.
import { Worker, isMainThread, parentPort } from 'node:worker_threads'
import type { CartesianGrid } from '../formats/cartesian.js'
import { readFoamField } from '../formats/foam.js'
import type { Bounds, SurfaceGeometry } from '../formats/geometry.js'
import { detectLattice, polyMeshBoundarySurface, polyMeshCellCenters, readPolyMesh, type PolyMeshPatch } from '../formats/polymesh.js'
import { readVtuCellData, readVtuInfo, vtuBoundarySurface, type VtuArrayInfo } from '../formats/vtu.js'

export type WorkerTask =
  | { op: 'foamField'; path: string; nCells: number | null }
  | { op: 'polyMesh'; dir: string }
  | { op: 'vtuGeometry'; path: string }
  | { op: 'vtuCellData'; path: string; name: string }

export interface FoamFieldResult {
  op: 'foamField'
  name: string
  class: string
  components: 1 | 3
  count: number
  data: Float32Array
  uniform: boolean
}

export interface PolyMeshResult {
  op: 'polyMesh'
  nCells: number
  surface: SurfaceGeometry
  cellCenters: Float32Array
  lattice: CartesianGrid | null
  patches: PolyMeshPatch[]
}

export interface VtuGeometryResult {
  op: 'vtuGeometry'
  nCells: number
  surface: SurfaceGeometry
  cellCenters: Float32Array
  /** Extent of the boundary face centroids (the true box, not the proxy quads). */
  domainBounds: Bounds
  time: number | null
  arrays: VtuArrayInfo[]
}

export interface VtuCellDataResult {
  op: 'vtuCellData'
  components: number
  data: Float32Array
}

export type WorkerResult = FoamFieldResult | PolyMeshResult | VtuGeometryResult | VtuCellDataResult
export type ResultOf<T extends WorkerTask> = Extract<WorkerResult, { op: T['op'] }>

export async function runTask(task: WorkerTask): Promise<WorkerResult> {
  switch (task.op) {
    case 'foamField': {
      const f = await readFoamField(task.path, { nCells: task.nCells })
      return { op: 'foamField', name: f.name, class: f.header.class, components: f.components, count: f.count, data: f.data, uniform: f.uniform }
    }
    case 'polyMesh': {
      const mesh = await readPolyMesh(task.dir)
      const cellCenters = polyMeshCellCenters(mesh)
      const surface = polyMeshBoundarySurface(mesh, cellCenters)
      const lattice = detectLattice(mesh, cellCenters)
      return { op: 'polyMesh', nCells: mesh.nCells, surface, cellCenters, lattice, patches: mesh.boundary }
    }
    case 'vtuGeometry': {
      const info = await readVtuInfo(task.path)
      const { surface, cellCenters, domainBounds } = await vtuBoundarySurface(info)
      return { op: 'vtuGeometry', nCells: info.nCells, surface, cellCenters, domainBounds, time: info.time, arrays: info.arrays }
    }
    case 'vtuCellData': {
      const info = await readVtuInfo(task.path)
      const { components, data } = await readVtuCellData(info, task.name)
      return { op: 'vtuCellData', components, data }
    }
  }
}

/** ArrayBuffers of every typed array in a result, for zero-copy transfer. */
function transferables(result: WorkerResult): ArrayBuffer[] {
  const out = new Set<ArrayBuffer>()
  const add = (a: ArrayBufferView | undefined | null): void => {
    if (a && a.buffer instanceof ArrayBuffer) out.add(a.buffer)
  }
  switch (result.op) {
    case 'foamField':
    case 'vtuCellData':
      add(result.data)
      break
    case 'polyMesh':
    case 'vtuGeometry':
      add(result.surface.positions)
      add(result.surface.normals)
      add(result.surface.indices)
      add(result.surface.cellOfTri)
      add(result.cellCenters)
      if (result.op === 'polyMesh' && result.lattice) {
        add(result.lattice.nodes.x)
        add(result.lattice.nodes.y)
        add(result.lattice.nodes.z)
      }
      break
  }
  return [...out]
}

interface Envelope {
  id: number
  task: WorkerTask
}

interface Reply {
  id: number
  ok: boolean
  result?: WorkerResult
  error?: string
}

if (!isMainThread && parentPort) {
  const port = parentPort
  port.on('message', (msg: Envelope) => {
    runTask(msg.task).then(
      (result) => port.postMessage({ id: msg.id, ok: true, result } satisfies Reply, transferables(result)),
      (e: unknown) => port.postMessage({ id: msg.id, ok: false, error: e instanceof Error ? e.message : String(e) } satisfies Reply),
    )
  })
}

export interface WorkerPool {
  run<T extends WorkerTask>(task: T): Promise<ResultOf<T>>
  close(): Promise<void>
  /** True once the pool gave up on threads and runs every task inline. */
  readonly inline: boolean
}

interface Pending {
  id: number
  task: WorkerTask
  resolve: (r: WorkerResult) => void
  reject: (e: Error) => void
}

interface Slot {
  worker: Worker
  busy: Pending | null
  ready: boolean
}

/**
 * execArgv for a worker: the parent's flags minus debugger and tsx loader
 * entries, plus (when running from `.ts` sources) an `--import` of the
 * bootstrap that registers tsx inside the thread — tsx itself only registers
 * on the main thread, and Node's native type stripping cannot resolve the
 * `.js` -> `.ts` imports this code base uses.
 */
function workerExecArgv(): string[] {
  const out: string[] = []
  const argv = process.execArgv
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]
    if (a.startsWith('--inspect')) continue
    const takesValue = (a === '--require' || a === '-r' || a === '--import' || a === '--loader' || a === '--experimental-loader') && i + 1 < argv.length
    const value = takesValue ? argv[i + 1] : a
    if (/[/\\]tsx[/\\]|^tsx(\/|$)/.test(value)) {
      if (takesValue) i++
      continue
    }
    out.push(a)
  }
  if (import.meta.url.endsWith('.ts')) out.push('--import', new URL('./worker-bootstrap.mjs', import.meta.url).href)
  return out
}

export function createWorkerPool(opts: { size?: number; inline?: boolean } = {}): WorkerPool {
  const size = Math.max(1, Math.min(2, opts.size ?? 2))
  let inline = opts.inline ?? false
  const slots: Slot[] = []
  const queue: Pending[] = []
  let nextId = 1
  let closed = false

  const pump = (): void => {
    if (closed) return
    while (queue.length) {
      if (inline) {
        const p = queue.shift()!
        runTask(p.task).then(p.resolve, p.reject)
        continue
      }
      const idle = slots.find((s) => !s.busy)
      if (idle) {
        const p = queue.shift()!
        idle.busy = p
        idle.worker.postMessage({ id: p.id, task: p.task } satisfies Envelope)
        continue
      }
      if (slots.length < size) {
        spawn()
        continue
      }
      return
    }
  }

  const spawn = (): void => {
    const worker = new Worker(new URL(import.meta.url), { execArgv: workerExecArgv() })
    const slot: Slot = { worker, busy: null, ready: false }
    slots.push(slot)
    worker.on('message', (reply: Reply) => {
      slot.ready = true
      const p = slot.busy
      slot.busy = null
      if (p && p.id === reply.id) {
        if (reply.ok && reply.result) p.resolve(reply.result)
        else p.reject(new Error(reply.error ?? 'worker task failed'))
      }
      pump()
    })
    const fail = (err: Error): void => {
      const idx = slots.indexOf(slot)
      if (idx >= 0) slots.splice(idx, 1)
      const p = slot.busy
      slot.busy = null
      if (p) {
        if (!slot.ready) {
          // The thread never came up (loader missing, resource limits): run inline from now on.
          inline = true
          queue.unshift(p)
        } else {
          p.reject(err)
        }
      }
      pump()
    }
    worker.on('error', fail)
    worker.on('exit', (code) => {
      if (slot.busy || slots.includes(slot)) fail(new Error(`worker exited with code ${code}`))
    })
    worker.unref()
  }

  return {
    get inline() {
      return inline
    },
    run<T extends WorkerTask>(task: T): Promise<ResultOf<T>> {
      if (closed) return Promise.reject(new Error('worker pool is closed'))
      return new Promise<WorkerResult>((resolve, reject) => {
        queue.push({ id: nextId++, task, resolve, reject })
        pump()
      }) as Promise<ResultOf<T>>
    },
    async close(): Promise<void> {
      closed = true
      for (const p of queue.splice(0)) p.reject(new Error('worker pool is closed'))
      await Promise.all(slots.splice(0).map((s) => s.worker.terminate()))
    },
  }
}
