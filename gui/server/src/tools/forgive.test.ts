import { describe, expect, it } from 'vitest'
import { UiCommandSchema } from '@cfd/shared'
import { forgive } from './forgive.js'
import { forgiveToolInput, getTool } from './index.js'

/** The four payloads GLM-5.3-Flash sent in the live sessions, each refused or neutered before forgive.ts. */
describe('forgiveToolInput', () => {
  const parse = (name: string, input: unknown) => getTool(name)!.schema.safeParse(forgiveToolInput(name, input))

  it('run_status {"runId":"null"} lists every run instead of failing on run "null"', () => {
    const r = parse('run_status', { runId: 'null' })
    expect(r.success).toBe(true)
    expect(r.data).toEqual({ runId: null })
  })

  it('mesh_generate {wallModel:"null"} is the preset form with no wall model', () => {
    const r = parse('mesh_generate', { kind: 'cavity', outputDir: 'cases/x', cells: [8, 8, 1], stl: 'null', cutcell: 'null', grading: 'null', wallModel: 'null', Ks: '', Cs: 'undefined', cyclic: 'null', permissive: 'null', config: 'null' })
    expect(r.success, JSON.stringify(r.error?.issues)).toBe(true)
    const d = r.data as Record<string, unknown>
    expect(d.wallModel).toBeNull()
    expect(d.stl).toBeNull()
    expect(d.config).toBeUndefined()
    expect(d.kind).toBe('cavity')
  })

  it('run_log {grep:"null"} is an unfiltered read', () => {
    const r = parse('run_log', { runId: 'r_1', fromSeq: 0, maxLines: 50, grep: 'null' })
    expect(r.success).toBe(true)
    expect((r.data as { grep: unknown }).grep).toBeNull()
  })

  it('gui_control open_result {timeIndex:"null"} shows the last time', () => {
    const r = parse('gui_control', { type: 'open_result', path: 'cases/x_jsonc', timeIndex: 'null' })
    expect(r.success).toBe(true)
    expect(r.data).toEqual({ type: 'open_result', path: 'cases/x_jsonc' })
    // and the discriminated branch decides what is optional: set_post's own leaves
    const p = parse('gui_control', { type: 'set_post', colormap: 'viridis', range: 'null', component: '', opacity: 'undefined' })
    expect(p.success).toBe(true)
    expect(p.data).toEqual({ type: 'set_post', colormap: 'viridis' })
  })

  it('leaves a required string alone, so a wrong path still fails by name', () => {
    const r = parse('gui_control', { type: 'open_case', path: '' })
    expect(r.success).toBe(true) // open_case wants a string; the empty one is the model's problem, not a parse error
    expect(r.data).toEqual({ type: 'open_case', path: '' })
    expect(UiCommandSchema.safeParse(forgive({ type: 'open_case', path: 'null' }, { type: 'object', properties: { path: { type: 'string' } }, required: ['path'] })).data).toEqual({ type: 'open_case', path: 'null' })
  })

  it('walks arrays and nested objects with the schema, and passes unknown shapes through', () => {
    const spec = { type: 'object', properties: { items: { type: 'array', items: { type: 'object', properties: { a: { type: 'string' }, b: { anyOf: [{ type: 'number' }, { type: 'null' }] } }, required: ['a', 'b'] } } } }
    expect(forgive({ items: [{ a: 'x', b: 'null' }, { a: 'null', b: 1 }] }, spec)).toEqual({ items: [{ a: 'x', b: null }, { a: 'null', b: 1 }] })
    expect(forgive({ zzz: 'null' }, spec)).toEqual({ zzz: 'null' })
    expect(forgive('null', spec)).toBe('null')
    expect(forgiveToolInput('no_such_tool', { a: 'null' })).toEqual({ a: 'null' })
  })
})
