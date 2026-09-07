import type { QuickAction } from '@cfd/shared'
import { useT } from '../../app/hooks'
import { useSessionStore } from '../../state/sessionStore'
import { actions } from '../../ws/actions'
import { Icon, type IconName } from '../common/Icon'

const TASKS: Array<{ action: QuickAction; icon: IconName; key: 'task.mesh' | 'task.run' | 'task.post' | 'task.validate' | 'task.export' }> = [
  { action: 'mesh', icon: 'mesh', key: 'task.mesh' },
  { action: 'run', icon: 'play', key: 'task.run' },
  { action: 'postprocess', icon: 'post', key: 'task.post' },
  { action: 'validate', icon: 'validate', key: 'task.validate' },
  { action: 'export', icon: 'exportIcon', key: 'task.export' },
]

export function SimulationTasks() {
  const t = useT()
  const online = useSessionStore((s) => s.connection === 'online' && s.currentSessionId !== null)
  return (
    <div data-testid="simulation-tasks">
      {TASKS.map((task) => (
        <button key={task.action} className="task-row" disabled={!online} onClick={() => actions.quick(task.action)} title={online ? t(task.key) : t('common.offlineDisabled')} data-testid={`task-${task.action}`}>
          <span className="ico">
            <Icon name={task.icon} size={15} />
          </span>
          <span>{t(task.key)}</span>
        </button>
      ))}
    </div>
  )
}
