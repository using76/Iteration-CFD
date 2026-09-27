// GPU state for the status bar and the `gpu_info` tool. Real machines are
// polled with nvidia-smi (once at boot, every 5 s while a GPU run is
// active, every 30 s otherwise); demo mode fabricates a card; no
// nvidia-smi means 'absent'.
import { execFile } from 'node:child_process'
import { scrubbedEnv } from '../env.js'
import type { GpuProcess, GpuState } from '@cfd/shared'

export interface GpuQuery {
  name: string
  memUsedMB: number
  memTotalMB: number
  /** Compute apps `nvidia-smi --query-compute-apps` listed, or null when that query errored. */
  processes?: GpuProcess[] | null
}

export interface GpuMonitor {
  state(): GpuState
  /** Tell the monitor a GPU run started/ended (drives 'busy' and the polling). */
  setActive(active: boolean): void
  onChange(handler: (gpu: GpuState) => void): () => void
  start(): Promise<void>
  stop(): void
}

export const DEMO_GPU: GpuQuery = { name: 'NVIDIA GeForce RTX 4090 (demo)', memUsedMB: 12400, memTotalMB: 24576 }
export const GPU_POLL_MS = 5000
export const GPU_IDLE_POLL_MS = 30_000
export const GPU_PROCESS_LIST_CAP = 64

export function parseNvidiaSmi(output: string): GpuQuery | null {
  const line = output.split(/\r?\n/).map((l) => l.trim()).find((l) => l.length > 0)
  if (!line) return null
  const parts = line.split(',').map((s) => s.trim())
  if (parts.length < 3) return null
  const used = Number(parts[parts.length - 2])
  const total = Number(parts[parts.length - 1])
  if (!Number.isFinite(used) || !Number.isFinite(total)) return null
  return { name: parts.slice(0, parts.length - 2).join(','), memUsedMB: used, memTotalMB: total }
}

/** Parse `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits`.
 *  Tolerant by design: a bad pid skips the line, a non-numeric memory ([N/A] under WDDM) is null,
 *  and the name is reduced to its basename so no user path ever leaves the machine. */
export function parseComputeApps(output: string): GpuProcess[] {
  const out: GpuProcess[] = []
  for (const raw of output.split(/\r?\n/)) {
    const line = raw.trim()
    if (!line) continue
    const parts = line.split(',').map((s) => s.trim())
    const pid = Number(parts[0])
    if (!Number.isFinite(pid)) continue
    const mem = Number(parts[parts.length - 1])
    const name = parts.slice(1, parts.length - 1).join(',')
    const slash = Math.max(name.lastIndexOf('\\'), name.lastIndexOf('/'))
    out.push({ pid, name: slash >= 0 ? name.slice(slash + 1) : name, memUsedMB: Number.isFinite(mem) ? mem : null })
  }
  return out
}

function queryComputeApps(): Promise<GpuProcess[] | null> {
  return new Promise((resolve) => {
    execFile(
      'nvidia-smi',
      ['--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'],
      { timeout: 4000, windowsHide: true, env: scrubbedEnv() },
      (err, stdout) => resolve(err ? null : parseComputeApps(String(stdout))),
    )
  })
}

export async function queryNvidiaSmi(): Promise<GpuQuery | null> {
  const first = await new Promise<GpuQuery | null>((resolve) => {
    execFile(
      'nvidia-smi',
      ['--query-gpu=name,memory.used,memory.total', '--format=csv,noheader,nounits'],
      { timeout: 4000, windowsHide: true, env: scrubbedEnv() },
      (err, stdout) => resolve(err ? null : parseNvidiaSmi(String(stdout))),
    )
  })
  if (!first) return null
  const processes = await queryComputeApps()
  return { ...first, processes }
}

export interface GpuMonitorOptions {
  demo: boolean
  pollMs?: number
  /** Poll rate while no GPU run is active but a card is present. */
  idlePollMs?: number
  query?: () => Promise<GpuQuery | null>
}

export function createGpuMonitor(opts: GpuMonitorOptions): GpuMonitor {
  const query = opts.query ?? queryNvidiaSmi
  const pollMs = opts.pollMs ?? GPU_POLL_MS
  const idlePollMs = opts.idlePollMs ?? GPU_IDLE_POLL_MS
  const handlers = new Set<(gpu: GpuState) => void>()
  let active = false
  let last: GpuQuery | null = opts.demo ? DEMO_GPU : null
  let probed = opts.demo
  let current: GpuState = compute()
  let timer: NodeJS.Timeout | null = null

  function compute(): GpuState {
    if (opts.demo) {
      return { state: active ? 'busy' : 'demo', name: DEMO_GPU.name, memUsedMB: DEMO_GPU.memUsedMB, memTotalMB: DEMO_GPU.memTotalMB, source: 'demo', processes: [], processCount: 0 }
    }
    if (!last) return { state: 'absent', name: null, memUsedMB: null, memTotalMB: null, source: probed ? 'none' : 'none', processes: null, processCount: null }
    // The list is nvidia-smi's view of the card, not this server's claim on it: it names every
    // process with a context (desktop apps too, under WDDM). Biggest memory first, nulls last.
    const procs = last.processes ?? null
    if (!procs) {
      return { state: active ? 'busy' : 'ready', name: last.name, memUsedMB: last.memUsedMB, memTotalMB: last.memTotalMB, source: 'nvidia-smi', processes: null, processCount: null }
    }
    const sorted = [...procs].sort((a, b) => {
      const am = a.memUsedMB ?? -1
      const bm = b.memUsedMB ?? -1
      return am !== bm ? bm - am : a.pid - b.pid
    })
    return { state: active ? 'busy' : 'ready', name: last.name, memUsedMB: last.memUsedMB, memTotalMB: last.memTotalMB, source: 'nvidia-smi', processes: sorted.slice(0, GPU_PROCESS_LIST_CAP), processCount: procs.length }
  }

  function publish() {
    const next = compute()
    if (JSON.stringify(next) === JSON.stringify(current)) return
    current = next
    for (const h of handlers) {
      try {
        h(current)
      } catch {
        // a listener's failure is not the monitor's problem
      }
    }
  }

  async function refresh() {
    if (opts.demo) return
    last = await query()
    probed = true
    publish()
  }

  function schedule() {
    if (opts.demo) return
    if (timer) {
      clearInterval(timer)
      timer = null
    }
    // An absent card is not polled for at idle: without nvidia-smi there is nothing to learn.
    if (!active && last === null) return
    timer = setInterval(() => void refresh(), active ? pollMs : idlePollMs)
    timer.unref()
  }

  function unschedule() {
    if (!timer) return
    clearInterval(timer)
    timer = null
  }

  return {
    state: () => current,
    setActive(a) {
      if (active === a) return
      active = a
      publish()
      schedule()
    },
    onChange(h) {
      handlers.add(h)
      return () => handlers.delete(h)
    },
    async start() {
      await refresh()
      schedule()
    },
    stop: unschedule,
  }
}
