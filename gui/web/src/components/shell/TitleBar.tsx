import { isMac, useT } from '../../app/hooks'
import { useUiStore } from '../../state/uiStore'
import { Icon, Logo } from '../common/Icon'
import { GpuBadge } from './GpuBadge'
import { SettingsPopover } from './SettingsPopover'

export function TitleBar() {
  const t = useT()
  const setPaletteOpen = useUiStore((s) => s.setPaletteOpen)
  const settingsOpen = useUiStore((s) => s.settingsOpen)
  const setSettingsOpen = useUiStore((s) => s.setSettingsOpen)
  return (
    <header className="titlebar" style={{ position: 'relative' }}>
      <div className="titlebar-left">
        <div className="traffic" aria-hidden>
          <i style={{ background: '#ff5f57' }} />
          <i style={{ background: '#febc2e' }} />
          <i style={{ background: '#28c840' }} />
        </div>
        <div className="brand">
          <Logo size={26} className="brand-logo" />
          <span>{t('app.title')}</span>
        </div>
        <span className="tagline truncate">
          <b>{t('app.tagline')}</b>
        </span>
      </div>
      <button className="cmd-search" onClick={() => setPaletteOpen(true)} data-testid="command-search" aria-label={t('search.placeholder')}>
        <Icon name="search" size={15} />
        <span className="grow truncate">{t('search.placeholder')}</span>
        <span className="kbd">{isMac() ? '⌘K' : 'Ctrl+K'}</span>
      </button>
      <div className="titlebar-right">
        <GpuBadge />
        <div className="profile-chip" data-testid="profile-chip">
          <span className="activity-avatar" style={{ width: 22, height: 22, margin: 0 }}>
            <Icon name="user" size={13} />
          </span>
          <span>{t('common.profile')}</span>
          <Icon name="chevronDown" size={13} className="muted" />
        </div>
        <button className={`icon-btn${settingsOpen ? ' active' : ''}`} onClick={() => setSettingsOpen(!settingsOpen)} title={t('common.settings')} aria-label={t('common.settings')} data-testid="settings-btn">
          <Icon name="gear" size={18} />
        </button>
      </div>
      <SettingsPopover />
    </header>
  )
}
