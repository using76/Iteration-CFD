import { describe, expect, it } from 'vitest'
import { ACTION_TYPES } from './actions.js'
import { LINK_TYPES } from './links.js'
import { DC_OBJECT_TYPES } from './objects.dc.js'
import { OBJECT_TYPES } from './objects.js'
import { ONTOLOGY_VERSION, validateOntology } from './registry.js'

describe('DC object types', () => {
  it('declares nineteen data-centre object types in the order docs/13 section 3 lists them', () => {
    expect(DC_OBJECT_TYPES.length).toBe(19)
    expect(DC_OBJECT_TYPES.map((t) => t.apiName)).toEqual(['Concept', 'Equation', 'Model', 'Capability', 'DcCase', 'DcFan', 'DcTile', 'DcRack', 'DcMetricReport', 'FanOperatingPoint', 'PatchFlowBalance', 'RackInletTemperature', 'ModelCaveat', 'MetricDef', 'Standard', 'StandardClause', 'AcceptanceCriterion', 'AcceptanceVerdict', 'ConvergenceCriterion'])
    for (const t of DC_OBJECT_TYPES) expect(t.ontologyVersion).toBe('0.1.0')
  })

  it('adds nineteen types to the shipped twelve with not one problem, of either severity', () => {
    const problems = validateOntology({ version: ONTOLOGY_VERSION, objects: [...OBJECT_TYPES, ...DC_OBJECT_TYPES], links: LINK_TYPES, actions: ACTION_TYPES })
    expect(problems).toEqual([])
  })

  it('collides with none of the shipped object type names or projections', () => {
    const shippedNames = new Set(OBJECT_TYPES.map((t) => t.apiName))
    const shippedProjections = new Set(OBJECT_TYPES.map((t) => t.source.projection).filter((x) => x !== ''))
    for (const t of DC_OBJECT_TYPES) expect(shippedNames.has(t.apiName)).toBe(false)
    for (const projection of DC_OBJECT_TYPES.map((t) => t.source.projection).filter((x) => x !== ''))
      expect(shippedProjections.has(projection)).toBe(false)
  })

  it('gives exactly AcceptanceCriterion and AcceptanceVerdict an empty projection, and every other type a path', () => {
    const actionOnly = DC_OBJECT_TYPES.filter((t) => t.actionCreatedOnly === true).map((t) => t.apiName).sort()
    expect(actionOnly).toEqual(['AcceptanceCriterion', 'AcceptanceVerdict'])
    for (const t of DC_OBJECT_TYPES) {
      if (t.actionCreatedOnly === true) {
        expect(t.source.projection).toBe('')
        expect(t.source.paths.length).toBe(0)
      } else {
        expect(t.source.projection).not.toBe('')
        expect(t.source.paths.length).toBeGreaterThanOrEqual(1)
      }
    }
  })

  it('declares no text on StandardClause and no nullable gitSha on a result', () => {
    const clause = DC_OBJECT_TYPES.find((t) => t.apiName === 'StandardClause')
    expect(clause).toBeDefined()
    expect(clause!.properties.some((pr) => pr.apiName === 'text')).toBe(false)
    const report = DC_OBJECT_TYPES.find((t) => t.apiName === 'DcMetricReport')
    expect(report).toBeDefined()
    expect(report!.properties.find((pr) => pr.apiName === 'gitSha')?.nullable).toBe(false)
  })
})
