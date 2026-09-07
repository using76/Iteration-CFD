import { useT, formatGB } from '../../app/hooks'
import { useSessionStore } from '../../state/sessionStore'
import { Icon } from '../common/Icon'

export function GpuBadge() {
  const t = useT()
  const gpu = useSessionStore((s) => s.gpu)
  const mode = useSessionStore((s) => s.hello?.mode ?? null)
  const state = gpu?.state ?? 'absent'
  const dot = state === 'ready' ? 'dot-ok' : state === 'busy' ? 'dot-warn' : state === 'demo' ? 'dot-accent' : 'dot-danger'
  const name = mode === 'demo' ? `${t('assistant.demo')} · ${gpu?.name ?? t('gpu.demoName')}` : (gpu?.name ?? t('gpu.absent'))
  const mem = gpu && gpu.memTotalMB !== null ? `${formatGB(gpu.memUsedMB)} / ${formatGB(gpu.memTotalMB)} GB` : null
  return (
    <div className="gpu-badge" data-testid="gpu-badge" title={`${state}${gpu?.source ? ` (${gpu.source})` : ''}`}>
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
      <Icon name="chip" size={14} className="muted" />
    </div>
  )
}
