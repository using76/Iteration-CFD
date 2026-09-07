// Small hooks shared by every component.
import { useCallback, useEffect, useRef, useState } from 'react'
import type { Locale } from '@cfd/shared'
import { tx, type UiKey } from '../i18n/extra'
import { useUiStore } from '../state/uiStore'

export type Translate = (key: UiKey, vars?: Record<string, string | number>) => string

export function useLocale(): Locale {
  return useUiStore((s) => s.locale)
}

export function useT(): Translate {
  const locale = useLocale()
  return useCallback((key: UiKey, vars?: Record<string, string | number>) => tx(locale, key, vars), [locale])
}

/** Applies the theme to <html data-theme> and the document language. */
export function useThemeEffect(): void {
  const theme = useUiStore((s) => s.theme)
  const locale = useUiStore((s) => s.locale)
  useEffect(() => {
    document.documentElement.dataset.theme = theme
    document.documentElement.lang = locale
  }, [theme, locale])
}

/** A ticking clock for elapsed-time displays; `null` interval stops it. */
export function useNow(intervalMs: number | null): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    if (intervalMs === null) return
    const id = setInterval(() => setNow(Date.now()), intervalMs)
    return () => clearInterval(id)
  }, [intervalMs])
  return now
}

/** Close-on-outside-click for popovers and menus. */
export function useDismiss(open: boolean, onClose: () => void): React.RefObject<HTMLDivElement | null> {
  const ref = useRef<HTMLDivElement | null>(null)
  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose()
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      // The Escape that closes this popover is spent here. Letting it reach the
      // window handler cancels the running turn as well.
      e.stopPropagation()
      e.preventDefault()
      onClose()
    }
    document.addEventListener('mousedown', onDown, true)
    document.addEventListener('keydown', onKey, true)
    return () => {
      document.removeEventListener('mousedown', onDown, true)
      document.removeEventListener('keydown', onKey, true)
    }
  }, [open, onClose])
  return ref
}

export function formatDuration(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) ms = 0
  const s = Math.floor(ms / 1000)
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = s % 60
  const two = (n: number) => String(n).padStart(2, '0')
  return `${two(h)}:${two(m)}:${two(sec)}`
}

export function formatGB(mb: number | null): string {
  return mb === null ? '?' : (mb / 1024).toFixed(1)
}

export function basename(path: string): string {
  const i = path.lastIndexOf('/')
  return i >= 0 ? path.slice(i + 1) : path
}

export function isMac(): boolean {
  return typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform)
}

export async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text)
    return true
  } catch {
    return false
  }
}
