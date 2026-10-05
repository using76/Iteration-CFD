// The GPU badge's process count and tooltip, pure so a test can read them.
import type { GpuState, RunInfo } from '@cfd/shared'
import type { Translate } from '../../app/hooks'

export const TOOLTIP_LINES = 16

/** Pids of runs this server started that are still running. */
export function ownRunPids(runs: Record<string, RunInfo>): Set<number> {
  return new Set(
    Object.values(runs)
      .filter((r) => r.status === 'running' && r.pid !== null)
      .map((r) => r.pid as number),
  )
}

/** Count and tooltip body for the badge: this server's own processes first, then the server's order. */
export function gpuProcessView(
  gpu: GpuState | null,
  own: ReadonlySet<number>,
  t: Translate,
): { count: string | null; tooltip: string } {
  if (gpu === null || gpu.processes == null) return { count: null, tooltip: t('gpu.processesUnknown') }
  if (gpu.processes.length === 0) return { count: null, tooltip: t('gpu.processesNone') }
  const n = gpu.processCount ?? gpu.processes.length
  const shown = [...gpu.processes]
    .sort((a, b) => (own.has(a.pid) ? 0 : 1) - (own.has(b.pid) ? 0 : 1))
    .slice(0, TOOLTIP_LINES)
  const lines = [t('gpu.processesHint', { n })]
  for (const p of shown) {
    lines.push(`${own.has(p.pid) ? '★ ' : ''}${p.pid}  ${p.name}${p.memUsedMB !== null ? `  ${p.memUsedMB} MB` : ''}`)
  }
  if (n > shown.length) lines.push(t('gpu.processesMore', { n: n - shown.length }))
  return { count: t('gpu.processes', { n }), tooltip: lines.join('\n') }
}
