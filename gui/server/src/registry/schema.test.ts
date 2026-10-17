import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { parse as parseJsonc } from 'jsonc-parser'
import { describe, expect, it } from 'vitest'
import { PICK_LISTS } from '@cfd/shared'
import { CASE_BLOCK_CHOICES, STATIC_SYSTEM } from '../prompts/system.js'
import { caseBlocks, choicesOf, getChtSchema, loadCaseSchema, loadOptionalCaseSchema, mergePickLists, schemaCandidatesFor, schemaFileFor, setChtSchema } from './schema.js'
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

  it('takes boussinesq, beta and a stated gas on plume.jsonc (CMP-29)', () => {
    const json = parseJsonc(fs.readFileSync(path.join(REPO_ROOT, 'cases', 'plume.jsonc'), 'utf8'), [], { allowTrailingComma: true }) as { physics: Record<string, unknown> & { fluid: Record<string, unknown> } }
    json.physics.buoyancy = 'boussinesq'
    json.physics.fluid.beta = 3.4e-3
    expect(schema.validate(json)).toEqual([])
    json.physics.compressible = { gamma: 1.4, W: 0.0289647, viscosity: { model: 'const', mu: 1.8e-5 }, pOperating: 1e5 }
    expect(schema.validate(json)).toEqual([])
    json.physics.buoyancy = 'bogus'
    expect(schema.validate(json).map((e) => e.pointer)).toContain('/physics/buoyancy')
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
    expect(lists.patchKinds).toEqual(['wall', 'inlet', 'open', 'empty', 'symmetry', 'wedge', 'farfield', 'supersonicInlet'])
    // the static fallback the system prompt prints is the generated list, not an older one
    expect(PICK_LISTS.patchKinds).toEqual(lists.patchKinds)
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

describe('cht schema', () => {
  it('schemaFileFor and the optional cht schema', () => {
    expect(schemaFileFor('cases/dieStack.cht.jsonc')).toBe('cht-1.json')
    expect(schemaFileFor('x/y.CHT.JSON')).toBe('cht-1.json')
    expect(schemaFileFor('cases/plume.jsonc')).toBe('case-1.json')
    expect(loadOptionalCaseSchema([path.join(os.tmpdir(), 'no-such-dir', 'cht-1.json')])).toBeNull()
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'cht-schema-'))
    try {
      const file = path.join(dir, 'cht-1.json')
      fs.writeFileSync(
        file,
        '{"$schema":"https://json-schema.org/draft/2020-12/schema","title":"ChtCase","type":"object","properties":{"name":{"type":"string"},"regions":{"type":"array"}},"required":["name","regions"]}',
      )
      const cht = loadOptionalCaseSchema([file])
      expect(cht?.schema.title).toBe('ChtCase')
      const errors = cht!.validate({ regions: [] })
      expect(errors[0].pointer).toBe('/name')
      expect(errors[0].message).toContain('name')
      setChtSchema(cht)
      expect(getChtSchema()).not.toBeNull()
      setChtSchema(null)
      expect(getChtSchema()).toBeNull()
    } finally {
      fs.rmSync(dir, { recursive: true, force: true })
    }
    expect(schemaCandidatesFor('/ws', '/gui', 'cht-1.json').every((p) => p.endsWith(path.join('docs', 'schema', 'cht-1.json')))).toBe(true)
  })
})

describe('case blocks (CHR-28)', () => {
  // The oracle over today's docs/schema/case-1.json: every dotted choice
  // path the five blocks' descriptions offer, nothing else.
  const EXPECTED_CHOICES: Record<string, string[]> = {
    'species.transport.model': ['lewis'],
    'chemistry.integrator': ['rodas3', 'ros3'],
    'chemistry.splitting': ['lie', 'strang'],
    'combustion.model': ['laminar', 'EDM', 'EDC', 'PaSR'],
    'combustion.variant': ['1989', '1996', '2005'],
    'radiation.absorption.model': ['constant', 'wsgg', 'wsggGrey'],
    'radiation.model': ['P1', 'fvDOM', 's2s', 'viewFactor'],
    'radiation.quadrature': ['S2', 'S4', 'S6', 'S8'],
    'radiation.scatter.model': ['none', 'isotropic', 'linearAnisotropic'],
    'parcels.coupling.energy': ['off', 'explicit', 'semiImplicit'],
    'parcels.coupling.mass': ['none', 'evaporation'],
    'parcels.coupling.momentum': ['off', 'explicit', 'semiImplicit'],
    'parcels.drag': ['none', 'stokes', 'schillerNaumann'],
    'parcels.evaporation.saturation': ['clausiusClapeyron', 'hylandWexler'],
    'parcels.evaporation.transfer': ['ranzMarshall', 'spalding', 'abramzonSirignano'],
    'parcels.impact.splash': ['mundo', 'baiGosman'],
    'parcels.impact.tension': ['constant', 'iapwsR1-76'],
    'parcels.physics': ['inert', 'heating', 'evaporating'],
    'parcels.wall': ['remove', 'rebound', 'stick', 'weber'],
  }

  it('reads the alternatives out of a description clause (choicesOf)', () => {
    expect(choicesOf('`a` | `b`.')).toEqual(['a', 'b'])
    expect(choicesOf('`lewis` (the default). `x` is refused.')).toEqual(['lewis'])
    expect(choicesOf('fvDOM only: `S2` | `S4`; `S4` by default.')).toEqual(['S2', 'S4'])
    expect(choicesOf('Path to a `.dat`; required with `a`/`b`.')).toBeNull()
    expect(choicesOf('The species closed by `1 - sum` (SPEC-LIT §19). No default.')).toBeNull()
    expect(choicesOf(undefined)).toBeNull()
    expect(choicesOf('')).toBeNull()
  })

  it('names the five blocks with their def, keys, required and choices (caseBlocks)', () => {
    const blocks = caseBlocks(schema.schema)
    expect(Object.keys(blocks).sort()).toEqual(['chemistry', 'combustion', 'parcels', 'radiation', 'species'])
    expect([blocks.species.def, blocks.chemistry.def, blocks.combustion.def, blocks.radiation.def, blocks.parcels.def]).toEqual(['JsonSpecies', 'JsonChemistry', 'JsonCombustion', 'JsonRadiation', 'JsonParcels'])
    expect(blocks.species.required).toEqual(['names'])
    expect(blocks.chemistry.required).toEqual([])
    expect(blocks.combustion.required).toEqual([])
    expect(blocks.radiation.required).toEqual(['model'])
    expect(blocks.parcels.required).toEqual(['capacity', 'injectors'])
    expect(blocks.radiation.keys).toEqual(['absorption', 'agglomerate', 'ambientTemperature', 'every', 'maxScatterIterations', 'model', 'open', 'quadrature', 'radiationRelaxation', 'radiositySweeps', 'scatter', 'scatterTol', 'viewFactorQuadrature', 'walls'])
    expect(blocks.species.keys).toEqual(['inert', 'initial', 'names', 'patches', 'transport'])
    expect(blocks.chemistry.keys).toEqual(['atol', 'integrator', 'mechanism', 'rtol', 'splitting', 'thermo', 'transport'])
    const union: Record<string, string[]> = {}
    for (const b of Object.values(blocks)) for (const [k, v] of Object.entries(b.choices)) union[k] = v
    expect(union).toEqual(EXPECTED_CHOICES)
    expect(schema.blocks).toEqual(blocks)
  })

  it('prints the block choices into the static prompt', () => {
    expect(CASE_BLOCK_CHOICES).toEqual(EXPECTED_CHOICES)
    expect(STATIC_SYSTEM).toContain('radiation.model: P1|fvDOM|s2s|viewFactor')
    for (const word of ['species', 'chemistry', 'combustion', 'radiation', 'parcels']) expect(STATIC_SYSTEM).toContain(word)
  })

  it('validates a mock-LLM case with a radiation block', () => {
    const json = parseJsonc(fs.readFileSync(path.join(REPO_ROOT, 'cases', 'plume.jsonc'), 'utf8'), [], { allowTrailingComma: true }) as { radiation?: Record<string, unknown> }
    json.radiation = { model: 'P1', absorption: { model: 'constant', kappa: 0.5 }, walls: { floor: { emissivity: 0.8 } }, every: 1 }
    expect(schema.validate(json)).toEqual([])
    json.radiation = { model: 'P1', absorbtion: {} }
    expect(schema.validate(json).map((e) => e.pointer)).toContain('/radiation/absorbtion')
    json.radiation = { every: 2 }
    expect(schema.validate(json).map((e) => e.pointer)).toContain('/radiation/model')
  })
})
