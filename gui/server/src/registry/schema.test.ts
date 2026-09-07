import fs from 'node:fs'
import path from 'node:path'
import { parse as parseJsonc } from 'jsonc-parser'
import { describe, expect, it } from 'vitest'
import { PICK_LISTS } from '@cfd/shared'
import { loadCaseSchema, mergePickLists } from './schema.js'
import { REPO_ROOT } from '../runs/test-helpers.js'

const schema = loadCaseSchema([path.join(REPO_ROOT, 'docs', 'schema', 'case-1.json')])

describe('case schema', () => {
  it('accepts cases/plume.jsonc', () => {
    const json = parseJsonc(fs.readFileSync(path.join(REPO_ROOT, 'cases', 'plume.jsonc'), 'utf8'), [], { allowTrailingComma: true })
    expect(schema.validate(json)).toEqual([])
  })

  it('accepts every shipped JSONC case', () => {
    const dir = path.join(REPO_ROOT, 'cases')
    for (const f of fs.readdirSync(dir).filter((n) => n.endsWith('.jsonc') && !n.includes('.cht.') && !n.includes('.dc.'))) {
      const json = parseJsonc(fs.readFileSync(path.join(dir, f), 'utf8'), [], { allowTrailingComma: true })
      expect(schema.validate(json), f).toEqual([])
    }
  })

  it('reports JSON pointers for a broken case', () => {
    const errors = schema.validate({ name: 42, mesh: { kind: 'polar', bounds: { min: [0, 0, 0], max: [1, 1, 1] }, cells: [1, 2] }, physics: {}, patches: [], initial: {}, numerics: {}, run: { endTime: 'soon' }, bogus: true })
    const pointers = errors.map((e) => e.pointer)
    expect(pointers).toContain('/name')
    expect(pointers).toContain('/mesh/kind')
    expect(pointers).toContain('/mesh/cells')
    expect(pointers).toContain('/run/endTime')
    expect(pointers).toContain('/bogus')
    expect(errors.every((e) => typeof e.message === 'string' && e.keyword.length > 0)).toBe(true)
  })

  it('merges the enum pick-lists from $defs, including oneOf enum + const shapes', () => {
    const lists = mergePickLists(schema.schema)
    expect(lists.algorithms).toEqual(['SIMPLE', 'PISO', 'PIMPLE'])
    expect(lists.patchKinds).toEqual(['wall', 'inlet', 'open', 'empty', 'symmetry'])
    expect(lists.wallTreatments).toEqual(['standard', 'spalding', 'rough', 'lowRe'])
    expect(lists.turbulenceKinds).toEqual(['RAS', 'LES'])
    expect(lists.meshKinds).toEqual(['cartesian'])
    expect(lists.buoyancy).toEqual(['densityRatio', 'boussinesq'])
    expect(lists.divSchemes).toEqual(PICK_LISTS.divSchemes)
    expect(schema.pickLists).toEqual(lists)
  })

  it('throws a clear error when the schema file is missing', () => {
    expect(() => loadCaseSchema(['/nowhere/case-1.json'])).toThrow(/case schema not found/)
  })
})
