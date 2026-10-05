// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// actions.newSession must carry the UI's locale so the server opens the
// session in the language the screen is showing.
import { beforeEach, describe, expect, it, vi } from 'vitest'

const memory = new Map<string, string>()
if (typeof globalThis.localStorage === 'undefined') {
  Object.defineProperty(globalThis, 'localStorage', {
    configurable: true,
    value: {
      getItem: (k: string) => (memory.has(k) ? (memory.get(k) as string) : null),
      setItem: (k: string, v: string) => void memory.set(k, String(v)),
      removeItem: (k: string) => void memory.delete(k),
      clear: () => void memory.clear(),
    },
  })
}
// zustand's default persist storage reads window.localStorage.
if (typeof globalThis.window === 'undefined') Object.defineProperty(globalThis, 'window', { configurable: true, value: globalThis })

const { send } = vi.hoisted(() => ({ send: vi.fn(() => true) }))
vi.mock('./client', () => ({ getWsClient: () => ({ send }) }))

// The localStorage shim above must exist before the stores evaluate.
const { actions } = await import('./actions')
const { useUiStore } = await import('../state/uiStore')

beforeEach(() => {
  send.mockClear()
  useUiStore.setState(useUiStore.getInitialState())
})

describe('newSession', () => {
  it('newSession sends the UI locale', () => {
    useUiStore.getState().setLocale('ko')
    expect(actions.newSession()).toBe(true)
    expect(send).toHaveBeenCalledWith({ t: 'session.new', locale: 'ko' })
    useUiStore.getState().setLocale('en')
    expect(actions.newSession()).toBe(true)
    expect(send).toHaveBeenLastCalledWith({ t: 'session.new', locale: 'en' })
  })
})
