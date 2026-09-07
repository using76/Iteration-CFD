// Browser-safe shortcuts. Ctrl on Windows/Linux, Cmd on macOS for the
// "Ctrl/Cmd" ones; Ctrl+B / Ctrl+J stay Ctrl everywhere (they are free in browsers).
import { useEffect } from 'react'
import { useEditorStore } from '../state/editorStore'
import { useSessionStore } from '../state/sessionStore'
import { selectActiveFile, useUiStore } from '../state/uiStore'
import { actions } from '../ws/actions'

function isEditable(el: EventTarget | null): boolean {
  if (!(el instanceof HTMLElement)) return false
  return el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable || !!el.closest('.monaco-editor')
}

export function useHotkeys(): void {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const mod = e.ctrlKey || e.metaKey
      const ui = useUiStore.getState()
      if (mod && !e.shiftKey && !e.altKey && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        ui.setPaletteOpen(!ui.paletteOpen)
        return
      }
      if (mod && e.shiftKey && e.key.toLowerCase() === 'n') {
        e.preventDefault()
        actions.newSession()
        return
      }
      if (e.ctrlKey && !e.shiftKey && !e.altKey && e.key.toLowerCase() === 'b') {
        e.preventDefault()
        ui.toggleSide()
        return
      }
      if (e.ctrlKey && !e.shiftKey && !e.altKey && e.key.toLowerCase() === 'j') {
        e.preventDefault()
        ui.toggleBottom()
        return
      }
      if (mod && !e.shiftKey && e.key.toLowerCase() === 's') {
        e.preventDefault()
        const file = selectActiveFile(ui)
        if (file) void useEditorStore.getState().save(file)
        return
      }
      if (mod && e.key === 'Enter' && isEditable(e.target) && (e.target as HTMLElement).dataset.testid === 'composer-input') {
        e.preventDefault()
        const text = ui.composerDraft
        if (actions.sendUserMessage(text)) ui.setComposerDraft('')
        return
      }
      if (e.altKey && !mod && /^[1-4]$/.test(e.key)) {
        e.preventDefault()
        ui.activateTabIndex(Number(e.key) - 1)
        return
      }
      if (e.key === 'Escape') {
        if (ui.paletteOpen) {
          ui.setPaletteOpen(false)
          return
        }
        if (ui.settingsOpen) {
          ui.setSettingsOpen(false)
          return
        }
        if (useSessionStore.getState().turn.active && !isEditable(e.target)) actions.cancelTurn()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])
}
