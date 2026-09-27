import { useMemo } from 'react'
import { useT, formatGB } from '../../app/hooks'
import { useSessionStore } from '../../state/sessionStore'
import { Icon } from '../common/Icon'
import { gpuProcessView, ownRunPids } from './gpuProcesses'

export function GpuBadge() {
  const t = useT()
  const gpu = useSessionStore((s) => s.gpu)
  const runs = useSessionStore((s) => s.runs)
  const mode = useSessionStore((s) => s.hello?.mode ?? null)
  const state = gpu?.state ?? 'absent'
  const view = useMemo(() => gpuProcessView(gpu, ownRunPids(runs), t), [gpu, runs, t])
  const dot = state === 'ready' ? 'dot-ok' : state === 'busy' ? 'dot-warn' : state === 'demo' ? 'dot-accent' : 'dot-danger'
  const name = mode === 'demo' ? `${t('assistant.demo')} · ${gpu?.name ?? t('gpu.demoName')}` : (gpu?.name ?? t('gpu.absent'))
  const mem = gpu && gpu.memTotalMB !== null ? `${formatGB(gpu.memUsedMB)} / ${formatGB(gpu.memTotalMB)} GB` : null
  return (
    <div className="gpu-badge" data-testid="gpu-badge" title={`${state}${gpu?.source ? ` (${gpu.source})` : ''}\n${view.tooltip}`}>
      <span className={`dot ${dot}`} />
      <b>GPU</b>
      <span className="name truncate" style={{ maxWidth: 200 }}>
        {name}
      </span>
      {mem ? (
        <>
          <span className="sep">|</span>
          <span className="mono">{mem}</span>
        </>
      ) : null}
      {view.count !== null ? (
        <>
          <span className="sep">|</span>
          <span className="mono" data-testid="gpu-procs">
            {view.count}
          </span>
        </>
      ) : null}
      <Icon name="chip" size={14} className="muted" />
    </div>
  )
}
