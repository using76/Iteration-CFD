import { TOOL_NAMES } from '@cfd/shared'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { fakeDatasets, fakeHub, fakeRuns, makeWorkspace, type TempWorkspace } from '../agent/test-fakes.js'
import type { ToolContext } from './context.js'
import { RESULT_CAP_BYTES, runTool, sanitizeSchema, toolDefinitions, toolResultBlock, TOOLS } from './index.js'

function walk(node: unknown, visit: (obj: Record<string, unknown>) => void): void {
  if (Array.isArray(node)) return node.forEach((n) => walk(n, visit))
  if (typeof node !== 'object' || node === null) return
  visit(node as Record<string, unknown>)
  for (const v of Object.values(node as Record<string, unknown>)) walk(v, visit)
}

let ws: TempWorkspace
beforeAll(async () => {
  ws = await makeWorkspace()
})
afterAll(() => ws.cleanup())

function ctx(over: Partial<ToolContext> = {}): ToolContext {
  return { config: ws.config, hub: fakeHub(), runs: fakeRuns(), datasets: fakeDatasets(), sessionId: 's1', signal: new AbortController().signal, workspaceRoot: ws.root, settings: { autoApprove: 'reads', effort: 'high', notifyOnRunEnd: true, locale: 'en' }, toolUseId: 'toolu_1', ...over }
}

describe('tool registry', () => {
  it('has a definition for every TOOL_NAMES entry, in the shared order', () => {
    expect(TOOLS.map((t) => t.name)).toEqual([...TOOL_NAMES])
    const defs = toolDefinitions()
    expect(defs.map((d) => d.name)).toEqual([...TOOL_NAMES])
    for (const d of defs) expect(d.description?.length ?? 0).toBeGreaterThan(20)
  })

  it('sanitised schemas carry no $schema and close every object', () => {
    for (const d of toolDefinitions()) {
      walk(d.input_schema, (obj) => {
        expect(obj).not.toHaveProperty('$schema')
        if (obj.type === 'object' && obj.properties) expect(obj.additionalProperties).toBe(false)
        expect(obj.minimum).not.toBe(-Number.MAX_SAFE_INTEGER)
      })
      expect((d as { strict?: boolean }).strict).toBeUndefined()
    }
    expect(sanitizeSchema({ $schema: 'x', type: 'object', properties: { a: { type: 'integer', minimum: -Number.MAX_SAFE_INTEGER } } })).toEqual({ type: 'object', properties: { a: { type: 'integer' } }, additionalProperties: false })
  })

  it('rejects invalid input with INVALID_INPUT before running', async () => {
    const r = await runTool('run_wait', { runId: 'x' }, ctx())
    expect(r.ok).toBe(false)
    expect(r.error?.code).toBe('INVALID_INPUT')
    expect(r.error?.message).toMatch(/maxSeconds/)
  })

  it('reports unknown tools', async () => {
    const r = await runTool('nope', {}, ctx())
    expect(r.error?.code).toBe('UNKNOWN_TOOL')
  })

  it('caps oversized results and points at the full file', async () => {
    const big = 'x'.repeat(RESULT_CAP_BYTES * 2)
    const r = await runTool('suggest_followups', { items: [big.slice(0, 100)] }, ctx())
    expect(r.ok).toBe(true)
    // Force the cap through file_read on a large file.
    const { default: fsp } = await import('node:fs/promises')
    await fsp.writeFile(`${ws.root}/big.txt`, Array.from({ length: 3000 }, (_, i) => `line ${i} ${'y'.repeat(20)}`).join('\n'))
    const read = await runTool('file_read', { path: 'big.txt', startLine: null, endLine: null }, ctx())
    expect(read.ok).toBe(true)
    const text = JSON.stringify(read.data)
    expect(Buffer.byteLength(text)).toBeLessThanOrEqual(RESULT_CAP_BYTES + 64)
    expect((read.data as { truncated: boolean }).truncated).toBe(true)
  })

  it('builds image tool_results as content block arrays', () => {
    const parts = toolResultBlock('toolu_9', { ok: true, data: { ok: true }, images: [{ base64: 'AAAA', mime: 'image/png' }] })
    expect(Array.isArray(parts.block.content)).toBe(true)
    const content = parts.block.content as Array<{ type: string }>
    expect(content[0].type).toBe('image')
    expect(content[1].type).toBe('text')
    expect(parts.block.is_error).toBe(false)
  })

  it('honours cancellation', async () => {
    const controller = new AbortController()
    const runs = fakeRuns({ finishAfterMs: null })
    const c = ctx({ runs, signal: controller.signal })
    const run = await runs.start({ binary: 'ofgpu-k-epsilon', casePath: 'cases/plume.jsonc', args: [], positionals: [], label: null, sessionId: null })
    const p = runTool('run_wait', { runId: run.id, maxSeconds: 30, untilIter: null, untilStatus: null, untilWritten: null }, c)
    setTimeout(() => controller.abort(), 5)
    const r = await p
    expect(r.ok).toBe(false)
    expect(r.error?.code).toBe('CANCELLED')
  })
})
