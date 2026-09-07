// GPU state for the status bar and the `gpu_info` tool. Real machines are
// polled with nvidia-smi (once at boot, every 5 s while a GPU run is
// active); demo mode fabricates a card; no nvidia-smi means 'absent'.
import { execFile } from 'node:child_process'
import type { GpuState } from '@cfd/shared'

export interface GpuQuery {
  name: string
  memUsedMB: number
  memTotalMB: number
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

export function queryNvidiaSmi(): Promise<GpuQuery | null> {
  return new Promise((resolve) => {
    execFile(
      'nvidia-smi',
      ['--query-gpu=name,memory.used,memory.total', '--format=csv,noheader,nounits'],
      { timeout: 4000, windowsHide: true },
      (err, stdout) => resolve(err ? null : parseNvidiaSmi(String(stdout))),
    )
  })
}

export interface GpuMonitorOptions {
  demo: boolean
  pollMs?: number
  query?: () => Promise<GpuQuery | null>
}

export function createGpuMonitor(opts: GpuMonitorOptions): GpuMonitor {
  const query = opts.query ?? queryNvidiaSmi
  const pollMs = opts.pollMs ?? GPU_POLL_MS
  const handlers = new Set<(gpu: GpuState) => void>()
  let active = false
  let last: GpuQuery | null = opts.demo ? DEMO_GPU : null
  let probed = opts.demo
  let current: GpuState = compute()
  let timer: NodeJS.Timeout | null = null

  function compute(): GpuState {
    if (opts.demo) {
      return { state: active ? 'busy' : 'demo', name: DEMO_GPU.name, memUsedMB: DEMO_GPU.memUsedMB, memTotalMB: DEMO_GPU.memTotalMB, source: 'demo' }
    }
    if (!last) return { state: 'absent', name: null, memUsedMB: null, memTotalMB: null, source: probed ? 'none' : 'none' }
    return { state: active ? 'busy' : 'ready', name: last.name, memUsedMB: last.memUsedMB, memTotalMB: last.memTotalMB, source: 'nvidia-smi' }
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
    if (timer || opts.demo) return
    timer = setInterval(() => void refresh(), pollMs)
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
      if (a) schedule()
      else unschedule()
    },
    onChange(h) {
      handlers.add(h)
      return () => handlers.delete(h)
    },
    async start() {
      await refresh()
    },
    stop: unschedule,
  }
}
