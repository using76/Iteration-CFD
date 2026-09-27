// Pure-function tests for the GPU badge's process count and tooltip.
import { describe, expect, it } from 'vitest'
import type { GpuState } from '@cfd/shared'
import { RUN_1 } from '../../state/fixtures'
import { tx } from '../../i18n/extra'
import type { Translate } from '../../app/hooks'
import { gpuProcessView, ownRunPids, TOOLTIP_LINES } from './gpuProcesses'

const t: Translate = (key, vars) => tx('en', key, vars)
const base: GpuState = { state: 'busy', name: 'RTX', memUsedMB: 9000, memTotalMB: 16303, source: 'nvidia-smi' }

describe('gpuProcessView', () => {
  it('says the list is unavailable when gpu is null', () => {
    expect(gpuProcessView(null, new Set(), t)).toEqual({ count: null, tooltip: 'process list unavailable (not queried)' })
  })
  it('says the list is unavailable when the server sent null fields', () => {
    const gpu: GpuState = { ...base, processes: null, processCount: null }
    expect(gpuProcessView(gpu, new Set(), t)).toEqual({ count: null, tooltip: 'process list unavailable (not queried)' })
  })
  it('says the card is empty when the list is empty', () => {
    const gpu: GpuState = { ...base, processes: [], processCount: 0 }
    expect(gpuProcessView(gpu, new Set(), t)).toEqual({ count: null, tooltip: 'no other process on the card' })
  })
  it('stars own processes and omits null memory', () => {
    const gpu: GpuState = {
      ...base,
      processes: [
        { pid: 41000, name: 'ofgpu-k-epsilon.exe', memUsedMB: 512 },
        { pid: 3092, name: '[Insufficient Permissions]', memUsedMB: null },
        { pid: 34184, name: 'python.exe', memUsedMB: null },
      ],
      processCount: 3,
    }
    const view = gpuProcessView(gpu, new Set([41000]), t)
    expect(view.count).toBe('3 proc')
    const lines = view.tooltip.split('\n')
    expect(lines[0]).toBe('3 processes nvidia-smi sees on the card (★ = a run this server started; per-process memory is unknown under WDDM)')
    expect(lines[1]).toBe('★ 41000  ofgpu-k-epsilon.exe  512 MB')
    expect(lines[2]).toBe('3092  [Insufficient Permissions]')
    expect(lines).toHaveLength(4)
  })
  it('cuts the tooltip at the line limit and reports the remainder of the uncut count', () => {
    const many = Array.from({ length: 20 }, (_, i) => ({ pid: 40000 + i, name: `p${i}.exe`, memUsedMB: null }))
    const gpu: GpuState = { ...base, processes: many, processCount: 70 }
    const lines = gpuProcessView(gpu, new Set(), t).tooltip.split('\n')
    expect(lines).toHaveLength(1 + TOOLTIP_LINES + 1)
    expect(lines[lines.length - 1]).toBe('… and 54 more')
  })
})

describe('ownRunPids', () => {
  it('keeps only running runs with a pid', () => {
    expect(
      ownRunPids({
        a: { ...RUN_1, status: 'running', pid: 5 },
        b: { ...RUN_1, status: 'done', pid: 6 },
        c: { ...RUN_1, status: 'running', pid: null },
      }),
    ).toEqual(new Set([5]))
  })
})
