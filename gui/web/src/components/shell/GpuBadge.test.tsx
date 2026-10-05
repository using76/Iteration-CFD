// The GPU badge's tooltip proven as rendered markup: no DOM and no testing library, the state
// arriving through module mocks because a server render reads the stores' initial state.
import { describe, expect, it, vi } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'
import type { GpuState, RunInfo } from '@cfd/shared'

const h = vi.hoisted(() => ({ gpu: null as GpuState | null, runs: {} as Record<string, RunInfo>, hello: null as unknown }))
vi.mock('../../state/sessionStore', () => ({
  useSessionStore: (sel: (s: unknown) => unknown) => sel({ gpu: h.gpu, runs: h.runs, hello: h.hello }),
}))
vi.mock('../../state/uiStore', () => ({ useUiStore: (sel: (s: unknown) => unknown) => sel({ locale: 'en' }) }))

import { RUN_1 } from '../../state/fixtures'
import { TOOLTIP_LINES } from './gpuProcesses'
import { GpuBadge } from './GpuBadge'

// gui/server/src/gpu/index.ts:29 GPU_PROCESS_LIST_CAP — the web workspace cannot import from the server workspace
const SERVER_LIST_CAP = 64

function renderTitle(): { html: string; lines: string[] } {
  const html = renderToStaticMarkup(<GpuBadge />)
  const title = /title="([^"]*)"/.exec(html)?.[1] ?? ''
  return { html, lines: title.split('\n') }
}

describe('GpuBadge rendered markup', () => {
  it('shows the busy tooltip with the own pid starred first, the uncut count and the cap cut', () => {
    const filler = Array.from({ length: SERVER_LIST_CAP - 2 }, (_, i) => ({ pid: 5000 + i, name: `p${i}.exe`, memUsedMB: null }))
    h.gpu = {
      state: 'busy', name: 'RTX 4090', memUsedMB: 9000, memTotalMB: 16303, source: 'nvidia-smi',
      processes: [
        { pid: 3092, name: '[Insufficient Permissions]', memUsedMB: null },
        { pid: 4242, name: 'ofgpu-k-epsilon.exe', memUsedMB: 512 },
        ...filler,
      ],
      processCount: 70,
    }
    h.runs = { r_1: { ...RUN_1, status: 'running', pid: 4242 } }
    expect(h.gpu!.processes).toHaveLength(SERVER_LIST_CAP)
    const { html, lines } = renderTitle()
    expect(lines[0]).toBe('busy (nvidia-smi)')
    expect(lines[1]).toBe('70 processes nvidia-smi sees on the card (★ = a run this server started; per-process memory is unknown under WDDM)')
    expect(lines[2]).toBe('★ 4242  ofgpu-k-epsilon.exe  512 MB')
    expect(lines[3]).toBe('3092  [Insufficient Permissions]')
    expect(lines[3]).not.toContain('MB')
    expect(lines.length).toBe(1 + 1 + TOOLTIP_LINES + 1)
    expect(lines[lines.length - 1]).toBe('… and 54 more')
    expect(html).toContain('data-testid="gpu-badge"')
    expect(html).toContain('RTX 4090')
    expect(html).toContain('8.8 / 15.9 GB')
    expect(html).toContain('data-testid="gpu-procs">70 proc<')
  })
  it('shows the absent badge with no process list when the card is unknown', () => {
    h.gpu = null
    h.runs = {}
    const { html, lines } = renderTitle()
    expect(lines[0]).toBe('absent')
    expect(lines[1]).toBe('process list unavailable (not queried)')
    expect(html).not.toContain('gpu-procs')
  })
})
