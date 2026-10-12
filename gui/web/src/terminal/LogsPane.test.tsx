// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// STU-03: the Log tab's filter is pure (filterLogLines) and the pane renders
// the ui store's logFilter - text, stream chips and the follow chip.
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'
import type { LogLine } from '@cfd/shared'
import type { LogFilter, LogStream } from '../state/uiStore'
import { filterLogLines, LogsPane } from './LogsPane'

// The real store cannot drive a static server render: zustand v5's hook reads
// getInitialState() on the server. The component tests' precedent
// (Viewer3D.overlays.test.tsx) mocks the store instead, with the fields and
// actions LogsPane touches.
const { useUiStore, useSessionStore, L } = vi.hoisted(() => {
  const state = {
    locale: 'en' as string,
    terminalRunId: 'r1' as string | null,
    logFilter: { text: 'warn', streams: ['stdout', 'stderr'] as string[], follow: false },
    setTerminalRun(): void {},
    setLogFilter(patch: { text?: string; streams?: string[]; follow?: boolean }): void {
      if (patch.follow !== undefined) state.logFilter = { ...state.logFilter, follow: patch.follow }
    },
  }
  const useUiStore = ((selector: (s: typeof state) => unknown) => selector(state)) as {
    (selector: (s: typeof state) => unknown): unknown
    getState(): typeof state
    getInitialState(): typeof state
    setState(patch: Partial<typeof state>): void
    subscribe(listener: () => void): () => void
  }
  useUiStore.getState = () => state
  useUiStore.getInitialState = () => state
  useUiStore.setState = (patch: Partial<typeof state>) => Object.assign(state, patch)
  useUiStore.subscribe = () => () => {}

  const L: LogLine[] = [
    { seq: 1, stream: 'stdout', text: 'iter 1 ok', ts: 0 },
    { seq: 2, stream: 'stderr', text: 'WARNING: low y+', ts: 0 },
    { seq: 3, stream: 'system', text: 'run started', ts: 0 },
    { seq: 4, stream: 'stdout', text: 'warning in stdout', ts: 0 },
  ]
  const session = {
    runs: { r1: { id: 'r1', binary: 'ofgpu-k-epsilon', label: null, status: 'running', startedAt: '2026-10-11T00:00:00Z' } },
    runData: { r1: { logs: L } },
  }
  const useSessionStore = (selector: (s: typeof session) => unknown) => selector(session)
  return { useUiStore, useSessionStore, L }
})

vi.mock('../state/uiStore', () => ({ useUiStore }))
vi.mock('../state/sessionStore', async (orig) => ({ ...(await orig()), useSessionStore }))

const render = (): string => renderToStaticMarkup(<LogsPane />)

describe('filterLogLines (STU-03)', () => {
  it('keeps a line when its stream is shown and the trimmed text matches', () => {
    const f: LogFilter = { text: ' warn ', streams: ['stderr', 'system'], follow: true }
    expect(filterLogLines(L, f).map((l) => l.seq)).toEqual([2])
    expect(
      filterLogLines(L, { text: '', streams: ['stdout', 'stderr', 'system'], follow: true }).map((l) => l.seq),
    ).toEqual([1, 2, 3, 4])
    expect(filterLogLines(L, { text: 'RUN', streams: ['system'], follow: false }).map((l) => l.seq)).toEqual([3])
  })
})

describe('LogsPane render (STU-03)', () => {
  beforeEach(() => {
    useUiStore.setState({ logFilter: { text: 'warn', streams: ['stdout', 'stderr'] as LogStream[], follow: false } })
  })

  it('renders only the filtered rows and mirrors the filter controls', () => {
    const html = render()
    expect(html).toContain('WARNING: low y+')
    expect(html).toContain('warning in stdout')
    expect(html).not.toContain('iter 1 ok')
    expect(html).not.toContain('run started')
    expect(html).toContain('value="warn"')
    expect(html).toMatch(/<button[^>]*class="chip on"[^>]*data-testid="logs-stream-stdout"/)
    expect(html).toMatch(/<button[^>]*class="chip on"[^>]*data-testid="logs-stream-stderr"/)
    expect(html).toMatch(/<button[^>]*class="chip"[^>]*data-testid="logs-stream-system"/)
    expect(html).toMatch(/<button[^>]*class="chip"[^>]*data-testid="logs-follow"/)
  })

  it('the follow chip turns on when the store follows again', () => {
    useUiStore.setState({ logFilter: { text: 'warn', streams: ['stdout', 'stderr'] as LogStream[], follow: true } })
    expect(render()).toMatch(/<button[^>]*class="chip on"[^>]*data-testid="logs-follow"/)
  })
})
