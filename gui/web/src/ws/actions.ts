// Component-facing helpers that turn UI intents into client frames. Every
// helper reads the stores directly so components stay presentational.
import type { QuickAction, SessionSettings } from '@cfd/shared'
import { pickActiveRun, sortRuns, useSessionStore } from '../state/sessionStore'
import { selectActiveFile, useUiStore } from '../state/uiStore'
import { getWsClient } from './client'

function sessionId(): string | null {
  return useSessionStore.getState().currentSessionId
}

/** Active case path for quick actions: the active JSONC/case file, else the active run's case. */
export function activeCasePath(): string | null {
  const ui = useUiStore.getState()
  const file = selectActiveFile(ui)
  if (file && /\.jsonc?$/i.test(file)) return file
  const s = useSessionStore.getState()
  const run = pickActiveRun(sortRuns(s.runs), ui.activeRunId)
  return run?.casePath ?? file
}

export function activeRunId(): string | null {
  const s = useSessionStore.getState()
  return pickActiveRun(sortRuns(s.runs), useUiStore.getState().activeRunId)?.id ?? null
}

export const actions = {
  sendUserMessage(text: string, attachments: string[] = [], selection: string | null = null): boolean {
    const sid = sessionId()
    const trimmed = text.trim()
    if (!sid || !trimmed) return false
    const ui = useUiStore.getState()
    const ok = getWsClient().send({
      t: 'user.message',
      sessionId: sid,
      text: trimmed,
      context: { activeFile: selectActiveFile(ui), activeRun: activeRunId(), attachments, selection },
    })
    if (ok) {
      useSessionStore.getState().setLastUserText(trimmed)
      useSessionStore.getState().clearTurnMarkers()
    }
    return ok
  },
  retryLast(): boolean {
    const text = useSessionStore.getState().lastUserText
    return text ? actions.sendUserMessage(text) : false
  },
  cancelTurn(): boolean {
    const sid = sessionId()
    return sid ? getWsClient().send({ t: 'turn.cancel', sessionId: sid }) : false
  },
  approve(toolUseIds: string[], remember: 'none' | 'session'): boolean {
    const sid = sessionId()
    return sid ? getWsClient().send({ t: 'tool.approve', sessionId: sid, toolUseIds, remember }) : false
  },
  deny(toolUseIds: string[], reason: string | null = null): boolean {
    const sid = sessionId()
    return sid ? getWsClient().send({ t: 'tool.deny', sessionId: sid, toolUseIds, reason }) : false
  },
  quick(action: QuickAction, casePath: string | null = activeCasePath(), runId: string | null = activeRunId()): boolean {
    const sid = sessionId()
    return sid ? getWsClient().send({ t: 'quick', sessionId: sid, action, casePath, runId }) : false
  },
  newSession(): boolean {
    return getWsClient().send({ t: 'session.new' })
  },
  openSession(id: string): boolean {
    useSessionStore.getState().setCurrentSessionId(id)
    return getWsClient().send({ t: 'session.open', sessionId: id })
  },
  listSessions(): boolean {
    return getWsClient().send({ t: 'session.list' })
  },
  deleteSession(id: string): boolean {
    return getWsClient().send({ t: 'session.delete', sessionId: id })
  },
  renameSession(id: string, title: string): boolean {
    return getWsClient().send({ t: 'session.rename', sessionId: id, title })
  },
  stopRun(runId: string): boolean {
    return getWsClient().send({ t: 'run.stop', runId })
  },
  setSettings(patch: Partial<SessionSettings>): boolean {
    const sid = sessionId()
    return sid ? getWsClient().send({ t: 'settings.set', sessionId: sid, patch }) : false
  },
  /** Open a result path in the 3D viewer tab (explorer clicks, run cards). */
  openInViewer(path: string, runId: string | null = null): void {
    void getWsClient().viewer.openPath(path, runId).then((r) => {
      if (!r.ok && r.error) useSessionStore.getState().addNote('warning', `viewer: ${r.error.code}: ${r.error.message}`)
    })
  },
}
