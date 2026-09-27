import { describe, expect, it } from 'vitest'
import { createGpuMonitor, DEMO_GPU, parseComputeApps, parseNvidiaSmi } from './index.js'

// A blank line is line 3: nvidia-smi emits one, and the parser must skip it.
const APPS_FIXTURE = '3092, [Insufficient Permissions], [N/A]\n34184, C:\\Program Files\\Python312\\python.exe, [N/A]\n\n41000, D:\\solver\\ofgpu-k-epsilon.exe, 512\n'

describe('gpu monitor', () => {
  it('parses nvidia-smi csv output', () => {
    expect(parseNvidiaSmi('NVIDIA GeForce RTX 5070 Ti, 1234, 16303\n')).toEqual({ name: 'NVIDIA GeForce RTX 5070 Ti', memUsedMB: 1234, memTotalMB: 16303 })
    expect(parseNvidiaSmi('')).toBeNull()
    expect(parseNvidiaSmi('garbage')).toBeNull()
  })

  it('fabricates the demo card and flips to busy while a GPU run is active', async () => {
    const m = createGpuMonitor({ demo: true })
    await m.start()
    expect(m.state()).toEqual({ state: 'demo', name: DEMO_GPU.name, memUsedMB: 12400, memTotalMB: 24576, source: 'demo', processes: [], processCount: 0 })
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

  it('parses compute-apps csv tolerantly', () => {
    const procs = parseComputeApps(APPS_FIXTURE)
    expect(procs.map((p) => p.pid)).toEqual([3092, 34184, 41000])
    expect(procs.map((p) => p.name)).toEqual(['[Insufficient Permissions]', 'python.exe', 'ofgpu-k-epsilon.exe'])
    expect(procs.map((p) => p.memUsedMB)).toEqual([null, null, 512])
    expect(parseComputeApps('')).toEqual([])
  })

  it('carries the process list and keeps state ready when this server started nothing', async () => {
    const m = createGpuMonitor({ demo: false, query: async () => ({ name: 'RTX', memUsedMB: 100, memTotalMB: 16000, processes: parseComputeApps(APPS_FIXTURE) }) })
    await m.start()
    expect(m.state().processes?.length).toBe(3)
    expect(m.state().processCount).toBe(3)
    expect(m.state().state).toBe('ready')
    expect(m.state().processes?.[0]?.pid).toBe(41000)
    m.stop()

    const many = Array.from({ length: 70 }, (_, i) => ({ pid: i + 1, name: `p${i}.exe`, memUsedMB: null }))
    const big = createGpuMonitor({ demo: false, query: async () => ({ name: 'RTX', memUsedMB: 100, memTotalMB: 16000, processes: many }) })
    await big.start()
    expect(big.state().processes?.length).toBe(64)
    expect(big.state().processCount).toBe(70)
    big.stop()
  })

  it('broadcasts only when the process list changes', async () => {
    let calls = 0
    const listA = [{ pid: 1, name: 'a.exe', memUsedMB: null }]
    const listB = [{ pid: 2, name: 'b.exe', memUsedMB: 8 }]
    const m = createGpuMonitor({
      demo: false,
      idlePollMs: 20,
      query: async () => ({ name: 'RTX', memUsedMB: 100, memTotalMB: 16000, processes: ++calls <= 2 ? listA : listB }),
    })
    await m.start()
    let fires = 0
    m.onChange(() => {
      fires++
    })
    await new Promise((r) => setTimeout(r, 70))
    expect(fires).toBe(1)
    m.stop()
  })

  it('polls at the idle rate when a card is present', async () => {
    let calls = 0
    const m = createGpuMonitor({
      demo: false,
      idlePollMs: 20,
      query: async () => {
        calls++
        return { name: 'RTX', memUsedMB: 100, memTotalMB: 16000, processes: [] }
      },
    })
    await m.start()
    expect(m.state().state).toBe('ready')
    await new Promise((r) => setTimeout(r, 70))
    expect(calls).toBeGreaterThan(1)
    expect(m.state().state).toBe('ready')
    m.stop()
  })
})
