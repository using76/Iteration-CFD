// Whole-host CPU and memory sampler. Every couple of seconds it broadcasts a
// `host` frame to every connected client, so the UI can show how loaded the
// machine is while a solver runs. CPU % is the delta of the os.cpus() tick
// counters between samples; the counters run since boot, so taking the first
// reading at start makes the first broadcast (one interval later) a real
// delta over that interval.
import os from 'node:os'
import type { HostState } from '@cfd/shared'
import { silentLogger, type Logger } from './log.js'
import type { Hub } from './ws/types.js'

export const HOST_SAMPLE_MS = 2_000

export interface HostSampler {
  stop(): void
}

interface CpuTicks {
  idle: number
  total: number
}

function cpuTicks(): CpuTicks {
  let idle = 0
  let total = 0
  for (const cpu of os.cpus()) {
    idle += cpu.times.idle
    total += cpu.times.user + cpu.times.nice + cpu.times.sys + cpu.times.idle + cpu.times.irq
  }
  return { idle, total }
}

function sampleHost(prev: CpuTicks, now: CpuTicks): { cpu: number; memUsedGb: number; memTotalGb: number } {
  const idle = now.idle - prev.idle
  const total = now.total - prev.total
  const cpu = total > 0 ? 100 * (1 - idle / total) : 0
  const memTotalGb = os.totalmem() / 1024 ** 3
  const memUsedGb = (os.totalmem() - os.freemem()) / 1024 ** 3
  return { cpu: Math.min(100, Math.max(0, Math.round(cpu * 10) / 10)), memUsedGb: Math.round(memUsedGb * 100) / 100, memTotalGb: Math.round(memTotalGb * 100) / 100 }
}

export function startHostSampler(hub: Pick<Hub, 'broadcast'>, log: Logger = silentLogger, intervalMs = HOST_SAMPLE_MS): HostSampler {
  let prev = cpuTicks()
  const timer = setInterval(() => {
    try {
      const now = cpuTicks()
      const { cpu, memUsedGb, memTotalGb } = sampleHost(prev, now)
      prev = now
      const host: HostState = { cpu, memUsedGb, memTotalGb, ts: Date.now() }
      hub.broadcast({ t: 'host', host })
    } catch (err) {
      log.warn(`host sampler: ${(err as Error).message}`)
    }
  }, intervalMs)
  timer.unref()
  return { stop: () => clearInterval(timer) }
}
