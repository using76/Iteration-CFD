// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// docs/17 BC-22: the catalogue's lookups. server/src/registry/sync.test.ts holds the data to the solver.
import { describe, expect, it } from 'vitest'
import { BC_CATALOGUE, bcCatalogueEntry, describeBcType, IMPLEMENTED_BC_TYPES } from './bcCatalogue'

describe('the BC catalogue', () => {
  it('finds a name exactly before its refused prefix family', () => {
    expect(bcCatalogueEntry('fixedValue')).toMatchObject({ family: 'basic', status: 'implemented', field: 'psi' })
    expect(bcCatalogueEntry('waveTransmissive')?.name).toBe('waveTransmissive')
    expect(bcCatalogueEntry('waveAlpha')).toMatchObject({ name: 'wave*', status: 'refused', reason: 'no wave model' })
    expect(bcCatalogueEntry('codedMixed')).toMatchObject({ name: 'coded*', status: 'refused' })
    expect(bcCatalogueEntry('nutUSpaldingWallFunction')).toMatchObject({ status: 'alias', to: 'nutUWallFunction' })
    expect(bcCatalogueEntry('bogusCondition')).toBeNull()
  })

  it('says what the solver does with a name', () => {
    expect(describeBcType('slip')).toBe('slip is implemented by the solver')
    expect(describeBcType('nutUSpaldingWallFunction')).toBe('nutUSpaldingWallFunction is read by the solver as nutUWallFunction')
    expect(describeBcType('v2WallFunction')).toBe('v2WallFunction is refused by the solver (v2-f turbulence model)')
    expect(describeBcType('bogusCondition')).toBe('bogusCondition is not a boundary-condition type the solver knows')
    const planned = BC_CATALOGUE.entries.find((e) => e.status === 'planned' || e.status === 'owned')
    if (planned) expect(describeBcType(planned.name)).toBe(`${planned.name} is not implemented yet (planned in ${planned.unit})`)
  })

  it('lists every implemented row once, and the counts are the rows', () => {
    const rows = BC_CATALOGUE.entries.filter((e) => e.status === 'implemented').map((e) => e.name)
    expect([...rows].sort()).toEqual([...IMPLEMENTED_BC_TYPES].sort())
    expect(new Set(IMPLEMENTED_BC_TYPES).size).toBe(IMPLEMENTED_BC_TYPES.length)
    const sum = Object.values(BC_CATALOGUE.counts).reduce((a, b) => a + b, 0)
    expect(sum).toBe(BC_CATALOGUE.entries.length)
  })
})
