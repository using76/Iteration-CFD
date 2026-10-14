// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// save_case (STU-05): the editor store validates the open buffer on the server
// before it writes, refuses CASE_INVALID with the finding count unless force,
// and writes exactly the buffer's bytes; the bridge answers save_case from it
// and brings the Problems list forward. The REST client is a fake whose
// validateCase reports one schema error for the BAD text (the server half,
// guiCase.test.ts, proves that text has exactly one).
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { ClientMsg } from '@cfd/shared'

vi.mock('../editor/monacoSetup', () => ({ disposeModel: () => {} }))
vi.mock('../api/rest', () => ({ api: { readFile: vi.fn(), writeFile: vi.fn(), validateCase: vi.fn() }, ApiError: class extends Error {} }))
const casePath = vi.hoisted(() => ({ current: null as string | null }))
vi.mock('../ws/actions', () => ({ actions: { setSettings: () => true, openSession: () => true }, activeCasePath: () => casePath.current }))
vi.mock('../viewer', () => ({
  getViewerApi: () => ({ getState: () => ({ camera: { position: [1, 1, 1], target: [0, 0, 0], projection: 'perspective' }, datasetId: null, datasetName: null, field: null, time: null, colormap: 'viridis', range: null, representation: 'surface', layers: [] }),
    isMounted: () => false,
    getTool: () => 'select',
    viewport: () => null,
  }),
  setViewerApi: () => {},
}))

import { api } from '../api/rest'
import { createUiBridge } from '../ws/uiBridge'
import { useEditorStore } from './editorStore'

const PATH = 'cases/bad.jsonc'
const GOOD = '{\n  // a comment the write must keep\n  "name": "bad",\n  "mesh": { "cells": [8, 8, 8] }\n}\n'
const BAD = '{\n  // a comment the write must keep\n  "name": "bad",\n  "mesh": { "cells": "eight" }\n}\n'
const ONE_ERROR = { pointer: '/mesh/cells', message: 'must be array' }

async function openBuffer(content: string) {
  vi.mocked(api.readFile).mockResolvedValue({ path: PATH, content: GOOD, hash: 'h0' } as never)
  await useEditorStore.getState().loadBuffer(PATH)
  useEditorStore.getState().setContent(PATH, content)
}

beforeEach(() => {
  useEditorStore.setState({ buffers: {}, diffs: {}, reveal: null })
  vi.mocked(api.readFile).mockReset()
  vi.mocked(api.writeFile).mockReset()
  vi.mocked(api.validateCase).mockReset()
  vi.mocked(api.validateCase).mockImplementation(async (path: string, content: string) => ({
    path,
    ok: content !== BAD,
    errors: content === BAD ? [ONE_ERROR] : [],
    warnings: [],
  }))
  vi.mocked(api.writeFile).mockImplementation(async (path: string, content: string) => ({ path, content, hash: 'h1' }) as never)
  casePath.current = PATH
})

describe('editorStore.saveCase', () => {
  it('a buffer with one schema error is refused CASE_INVALID with the count, and nothing is written', async () => {
    await openBuffer(BAD)
    const r = await useEditorStore.getState().saveCase(PATH)
    expect(r).toMatchObject({ ok: false, code: 'CASE_INVALID', errors: 1 })
    expect(vi.mocked(api.validateCase)).toHaveBeenCalledWith(PATH, BAD)
    expect(vi.mocked(api.writeFile)).not.toHaveBeenCalled()
    expect(useEditorStore.getState().buffers[PATH]?.dirty).toBe(true)
  })

  it('force writes anyway, and the bytes written are the buffer', async () => {
    await openBuffer(BAD)
    const r = await useEditorStore.getState().saveCase(PATH, { force: true })
    expect(r).toEqual({ ok: true, path: PATH, errors: 1, warnings: 0 })
    expect(vi.mocked(api.writeFile)).toHaveBeenCalledTimes(1)
    const [p, bytes, baseHash] = vi.mocked(api.writeFile).mock.calls[0]
    expect(p).toBe(PATH)
    expect(bytes).toBe(BAD)
    expect(baseHash).toBe('h0')
    expect(useEditorStore.getState().buffers[PATH]).toMatchObject({ savedContent: BAD, dirty: false })
  })

  it('a valid buffer is written without force', async () => {
    await openBuffer(GOOD.replace('[8, 8, 8]', '[9, 9, 9]'))
    const r = await useEditorStore.getState().saveCase(PATH)
    expect(r).toMatchObject({ ok: true, errors: 0 })
    expect(vi.mocked(api.writeFile).mock.calls[0][1]).toBe(GOOD.replace('[8, 8, 8]', '[9, 9, 9]'))
  })

  it('no open buffer answers NO_BUFFER and validates nothing', async () => {
    const r = await useEditorStore.getState().saveCase(PATH)
    expect(r).toMatchObject({ ok: false, code: 'NO_BUFFER' })
    expect(vi.mocked(api.validateCase)).not.toHaveBeenCalled()
  })
})

describe('the bridge answers save_case from the editor store', () => {
  function bridge() {
    const sent: ClientMsg[] = []
    const ui = { tabs: [], activeRunId: null, showAxes: true, showColorBars: true, setBottomVisible: vi.fn(), setBottomTab: vi.fn() }
    const session = { problems: {}, connection: 'online', addNote: vi.fn(), runs: {}, runData: {} }
    const b = createUiBridge({
      send: (m) => {
        sent.push(m)
        return true
      },
      ui: { getState: () => ui as never },
      session: { getState: () => session as never },
      subscribeRun: vi.fn(),
      editor: {
        isDirty: (p) => !!useEditorStore.getState().buffers[p]?.dirty,
        closeBuffer: () => {},
        removeDiff: () => {},
        saveCase: (p, force) => useEditorStore.getState().saveCase(p, { force }),
      },
    })
    const result = (id: string) => sent.find((m): m is Extract<ClientMsg, { t: 'ui.result' }> => m.t === 'ui.result' && m.requestId === id)
    return { b, ui, result }
  }

  it('refuses CASE_INVALID naming the count and force, and opens Problems', async () => {
    await openBuffer(BAD)
    const { b, ui, result } = bridge()
    await b.handleCommand('s1', { type: 'save_case' })
    expect(result('s1')?.ok).toBe(false)
    expect(result('s1')?.error).toBe(
      `CASE_INVALID (save_case): validation found 1 error in ${PATH}; nothing was written. The findings are in the Problems list: fix them, or send save_case {force: true} to save anyway`,
    )
    expect(ui.setBottomTab).toHaveBeenCalledWith('problems')
    expect(vi.mocked(api.writeFile)).not.toHaveBeenCalled()
  })

  it('force: true saves the buffer', async () => {
    await openBuffer(BAD)
    const { b, result } = bridge()
    await b.handleCommand('s2', { type: 'save_case', force: true })
    expect(result('s2')?.ok).toBe(true)
    expect(vi.mocked(api.writeFile).mock.calls[0][1]).toBe(BAD)
  })

  it('no case on screen answers NO_CASE', async () => {
    casePath.current = null
    const { b, result } = bridge()
    await b.handleCommand('s3', { type: 'save_case' })
    expect(result('s3')?.error).toBe('NO_CASE (save_case): no case is open in the editor; open_case first')
  })
})
