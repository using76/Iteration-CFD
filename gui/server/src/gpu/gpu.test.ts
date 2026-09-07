import { describe, expect, it } from 'vitest'
import { createGpuMonitor, DEMO_GPU, parseNvidiaSmi } from './index.js'

describe('gpu monitor', () => {
  it('parses nvidia-smi csv output', () => {
    expect(parseNvidiaSmi('NVIDIA GeForce RTX 5070 Ti, 1234, 16303\n')).toEqual({ name: 'NVIDIA GeForce RTX 5070 Ti', memUsedMB: 1234, memTotalMB: 16303 })
    expect(parseNvidiaSmi('')).toBeNull()
    expect(parseNvidiaSmi('garbage')).toBeNull()
  })

  it('fabricates the demo card and flips to busy while a GPU run is active', async () => {
    const m = createGpuMonitor({ demo: true })
    await m.start()
    expect(m.state()).toEqual({ state: 'demo', name: DEMO_GPU.name, memUsedMB: 12400, memTotalMB: 24576, source: 'demo' })
    const seen: string[] = []
    m.onChange((g) => seen.push(g.state))
    m.setActive(true)
    m.setActive(true)
    expect(m.state().state).toBe('busy')
    m.setActive(false)
    expect(m.state().state).toBe('demo')
    expect(seen).toEqual(['busy', 'demo'])
    m.stop()
  })

  it('reports absent without nvidia-smi and ready/busy with it', async () => {
    const absent = createGpuMonitor({ demo: false, query: async () => null })
    await absent.start()
    expect(absent.state()).toMatchObject({ state: 'absent', name: null, source: 'none' })
    absent.setActive(true)
    expect(absent.state().state).toBe('absent')
    absent.stop()

    let calls = 0
    const present = createGpuMonitor({ demo: false, pollMs: 20, query: async () => ({ name: 'RTX', memUsedMB: 100 + calls++, memTotalMB: 16000 }) })
    await present.start()
    expect(present.state()).toMatchObject({ state: 'ready', name: 'RTX', memUsedMB: 100, source: 'nvidia-smi' })
    present.setActive(true)
    expect(present.state().state).toBe('busy')
    await new Promise((r) => setTimeout(r, 70))
    expect(calls).toBeGreaterThan(1)
    expect(present.state().memUsedMB).toBeGreaterThan(100)
    present.setActive(false)
    const after = calls
    await new Promise((r) => setTimeout(r, 50))
    expect(calls).toBe(after)
    expect(present.state().state).toBe('ready')
    present.stop()
  })
})
