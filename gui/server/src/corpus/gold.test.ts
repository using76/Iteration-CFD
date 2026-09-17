// Tests for the C4 gold set and its scorer: the gold set is exactly O2's in-scope seed
// carrying verbatim evidence, the loader refuses a bad row by name, the scorer scores.
import { readFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'
import { DC_ONTOLOGY, DC_SEED, buildRegistry, toLinkInputs } from '@cfd/shared'
import type { PropertyDef } from '@cfd/shared'
import { gate } from './gate.js'
import type { GateCandidateLink, GateCandidateObject, GateValue, GateWorld } from './gate.js'
import { checkGoldSet, formatScoreTable, goldText, GoldError, loadGoldSet, resolveEvidence, scoreAgainstGold } from './gold.js'
import type { GoldLink, GoldObject, GoldSet, PropertyScore } from './gold.js'

const REPO = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', '..', '..', '..')
const KEY = (a: string, b: string): string => a + ':' + b
const must = <T>(value: T | undefined, what: string): T => { if (value === undefined) throw new Error('expected ' + what); return value }
const countOccurrences = (text: string, needle: string): number => { let n = 0; for (let at = text.indexOf(needle); at !== -1; at = text.indexOf(needle, at + 1)) n += 1; return n }
const thrown = (run: () => unknown): GoldError | null => { try { run(); return null } catch (e) { return e as GoldError } }

/** The passages of rust/SPEC-LIT.md sections 52 to 55, keyed by subsection number. */
function specLitPassages(): Map<string, string> {
  const lines = readFileSync(path.join(REPO, 'rust', 'SPEC-LIT.md'), 'utf8').split('\n').slice(8280, 9904)
  const out = new Map<string, string>()
  let locator: string | null = null, buf: string[] = []
  const flush = (): void => { if (locator !== null) out.set(locator, buf.join('\n')) }
  for (const line of lines) {
    const m = /^### (5[2-5]\.\d{1,2}) /.exec(line)
    if (m) { flush(); locator = m[1]; buf = [line] } else if (locator !== null) buf.push(line)
  }
  flush(); return out
}

/** The shipped registry and the passages, wired into a read-only GateWorld. chunkId === locator here. */
function goldWorld(passages: Map<string, string>): GateWorld {
  return { ontology: DC_ONTOLOGY, chunkText: (id) => passages.get(id) ?? null, objectExists: () => false, objectsByNormalisedTitle: () => [], existingLinkCount: () => 0 }
}

/** A perfect extractor: one candidate per gold row, every value stringified, every span resolved. */
function candidateFromGold(g: GoldObject, passages: Map<string, string>): GateCandidateObject {
  const span = resolveEvidence(g, passages.get(g.locator) ?? '') ?? { charStart: 0, charEnd: 0 }
  const props: Record<string, GateValue> = {}
  for (const [key, value] of Object.entries(g.properties)) {
    props[key] = { value: goldText(value), quote: g.evidence, charStart: span.charStart, charEnd: span.charEnd }
  }
  return { candidateId: KEY(g.objectType, g.primaryKey), objectType: g.objectType, primaryKey: g.primaryKey, chunkId: g.locator, props }
}

/** The same for links. */
function linkFromGold(l: GoldLink): GateCandidateLink {
  return { candidateLinkId: l.linkType + ':' + KEY(l.fromId, l.toId), linkType: l.linkType, fromObjectType: l.fromType, fromPrimaryKey: l.fromId, toObjectType: l.toType, toPrimaryKey: l.toId, chunkId: l.locator }
}

const PASSAGES = specLitPassages()
const GOLD = loadGoldSet('spec-lit-52-55', (id) => PASSAGES.get(id) ?? null)
const PERFECT = gate(GOLD.objects.map((g) => candidateFromGold(g, PASSAGES)), GOLD.links.map(linkFromGold), goldWorld(PASSAGES))
const FOREIGN: GoldSet = { goldSetId: 'spec-lit-52-55', ontologyVersion: '0.1.0', documentId: 'doc:ashrae-tc99', sourcePath: 'rust/SPEC-LIT.md', seedSource: 'gui/shared/src/ontology/seed/dc.ts', objects: [], links: [] }
// A one-row Widget gold, for the per-cell tallies the real gold's shared properties cannot show.
const p = (apiName: string, nullable = true): PropertyDef => ({ apiName, displayName: apiName, baseType: 'string', nullable, description: apiName })
const WIDGET = buildRegistry({ version: '0.1.0', actions: [], links: [], objects: [{ apiName: 'Widget', displayName: 'Widget', pluralName: 'Widgets', description: 'fixture', icon: '', source: { projection: 'goldFixtureWidgets', paths: ['test://gold'] }, primaryKey: 'widgetId', titleKey: 'name', ontologyVersion: '0.1.0', properties: [p('widgetId', false), p('name', false), p('weight'), p('colour')] }] })
const MINI_PASSAGES = new Map([['52.1', 'one widget']])
const tally = (cells: Map<string, PropertyScore>, key: string): [number, number, number] => { const x = must(cells.get(key), 'cell ' + key); return [x.tp, x.fp, x.fn] }

describe('the gold set and its scorer', () => {
  it("the gold set is exactly O2's seed for SPEC-LIT sections 52 to 55", () => {
    const scoped = DC_SEED.objects.filter((row) => typeof row.props.specSection === 'string' && /^S5[2-5](\.\d{1,2})?$/.test(String(row.props.specSection)))
    const metrics = DC_SEED.objects.filter((row) => row.type === 'MetricDef' && scoped.some((s) => s.id === String(row.props.equationId)))
    const all = [...scoped, ...metrics]
    const inScope = new Set(all.map((row) => KEY(row.type, row.id)))
    const goldIds = new Set(GOLD.objects.map((g) => KEY(g.objectType, g.primaryKey)))
    expect(all.filter((row) => !goldIds.has(KEY(row.type, row.id))).map((row) => KEY(row.type, row.id)), 'missingFromGold').toEqual([])
    expect(GOLD.objects.filter((g) => !inScope.has(KEY(g.objectType, g.primaryKey))).map((g) => KEY(g.objectType, g.primaryKey)), 'extraInGold').toEqual([])
    const linkId = (l: { linkType: string; fromId: string; toId: string }): string => KEY(l.linkType, KEY(l.fromId, l.toId))
    const inLinks = toLinkInputs(DC_ONTOLOGY, DC_SEED).filter((l) => inScope.has(KEY(l.fromType, l.fromId)) && inScope.has(KEY(l.toType, l.toId)))
    const goldLinkIds = new Set(GOLD.links.map(linkId))
    const scopeLinkIds = new Set(inLinks.map(linkId))
    expect(inLinks.filter((l) => !goldLinkIds.has(linkId(l))).map(linkId), 'missingLinks').toEqual([])
    expect(GOLD.links.filter((l) => !scopeLinkIds.has(linkId(l))).map(linkId), 'extraLinks').toEqual([])
    expect(GOLD.objects.length).toBe(43)
    expect(GOLD.links.length).toBe(21)
  })
  it('every gold property value equals the value O2 seeded', () => {
    const byKey = new Map(DC_SEED.objects.map((row) => [KEY(row.type, row.id), row]))
    for (const g of GOLD.objects) {
      const row = must(byKey.get(KEY(g.objectType, g.primaryKey)), KEY(g.objectType, g.primaryKey))
      for (const [key, value] of Object.entries(g.properties)) {
        expect(value, KEY(g.objectType, g.primaryKey) + '.' + key).toEqual(row.props[key])
      }
      for (const [key, value] of Object.entries(row.props)) {
        expect(Object.keys(g.properties), KEY(g.objectType, g.primaryKey) + ' ' + key)[value === null ? 'not' : 'to'].toContain(key)
      }
    }
  })
  it('every gold evidence occurs exactly once in the chunk its locator names', () => {
    for (const row of GOLD.objects) {
      const passage = must(PASSAGES.get(row.locator), 'passage ' + row.locator)
      expect(countOccurrences(passage, row.evidence), KEY(row.objectType, row.primaryKey)).toBe(1)
      const span = resolveEvidence(row, passage)
      if (span === null) throw new Error('evidence not unique: ' + KEY(row.objectType, row.primaryKey))
      expect(passage.slice(span.charStart, span.charEnd)).toBe(row.evidence)
    }
  })
  it('refuses a gold entry whose evidence is not in its chunk, quoting the evidence', () => {
    const row = must(GOLD.objects.find((g) => g.primaryKey === 'EQ-D-RTI'), 'EQ-D-RTI')
    const mutilated = must(PASSAGES.get(row.locator), 'passage ' + row.locator).replace(row.evidence, '')
    const threw = thrown(() => loadGoldSet('spec-lit-52-55', (id) => (id === row.locator ? mutilated : PASSAGES.get(id) ?? null)))
    expect(threw?.code).toBe('GOLD_EVIDENCE_ABSENT')
    expect(threw?.message).toContain(row.locator)
    expect(threw?.message).toContain(row.evidence)
  })
  it('refuses a gold entry whose document is not our own SPEC-LIT', () => {
    const threw = thrown(() => checkGoldSet(FOREIGN, 'spec-lit-52-55', null))
    expect(threw?.code).toBe('GOLD_FOREIGN_DOCUMENT')
    expect(threw?.message).toContain('doc:ashrae-tc99')
    expect(threw?.message).toContain('doc:spec-lit')
  })
  it('types every gold row against the shipped ontology', () => {
    for (const row of GOLD.objects) {
      const def = DC_ONTOLOGY.objectType(row.objectType)
      if (def === null) throw new Error('undeclared object type ' + row.objectType)
      for (const key of Object.keys(row.properties)) {
        expect(def.properties.some((p) => p.apiName === key), row.objectType + '.' + key).toBe(true)
      }
      expect(row.properties[def.primaryKey], row.objectType + ' primary key').toBe(row.primaryKey)
      for (const p of def.properties) if (!p.nullable) expect(Object.keys(row.properties), KEY(row.objectType, row.primaryKey) + ' omits ' + p.apiName).toContain(p.apiName)
    }
  })
  it('a perfect extractor scores one on every property', () => {
    expect(PERFECT.refusals, JSON.stringify(PERFECT.refusals.map((r) => r.message))).toEqual([])
    const score = scoreAgainstGold(GOLD, PERFECT)
    for (const p of score.properties) expect([p.precision, p.recall, p.f1], p.key).toEqual([1, 1, 1])
    expect(score.micro.precision).toBe(1)
    expect(score.objectsMatched).toBe(score.objectsGold)
  })
  it('counts a missing value as a false negative and a wrong value as both', () => {
    // The real gold has no property only one row carries, so the per-cell tally of one
    // dropped value is only visible on a one-row gold: a mini Widget with four cells.
    const mini: GoldSet = { ...FOREIGN, goldSetId: 'mini', documentId: 'doc:spec-lit', objects: [
      { objectType: 'Widget', primaryKey: 'w1', locator: '52.1', evidence: 'one widget', properties: { widgetId: 'w1', name: 'one', weight: '2 kg', colour: 'red' } }] }
    const world: GateWorld = { ontology: WIDGET, chunkText: (id) => MINI_PASSAGES.get(id) ?? null, objectExists: () => false, objectsByNormalisedTitle: () => [], existingLinkCount: () => 0 }
    const row = must(mini.objects[0], 'the mini row')
    const perfect = new Map(scoreAgainstGold(mini, gate([candidateFromGold(row, MINI_PASSAGES)], [], world)).properties.map((q) => [q.key, q]))
    const wronged = candidateFromGold(row, MINI_PASSAGES)
    wronged.props.weight = { value: null, quote: null, charStart: null, charEnd: null }
    wronged.props.colour = { value: 'WRONG', quote: null, charStart: 0, charEnd: 0 }
    const report = gate([wronged], [], world)
    expect(report.refusals).toEqual([])
    const cells = new Map(scoreAgainstGold(mini, report).properties.map((q) => [q.key, q]))
    expect(tally(cells, 'Widget.weight')).toEqual([0, 0, 1])
    expect(tally(cells, 'Widget.colour')).toEqual([0, 1, 1])
    for (const [key, q] of cells) {
      if (key === 'Widget.weight' || key === 'Widget.colour') continue
      expect([q.tp, q.fp, q.fn], key).toEqual(tally(perfect, key))
    }
  })
  it('gives a property nobody proposed no precision at all, not a precision of zero', () => {
    // `equationId` is nullable, so leaving it off every candidate passes the gate - and
    // its cell is proposed by nobody: precision must be null, recall zero, f1 null.
    const candidates = GOLD.objects.map((g) => { const c = candidateFromGold(g, PASSAGES); delete c.props.equationId; return c })
    const report = gate(candidates, GOLD.links.map(linkFromGold), goldWorld(PASSAGES))
    expect(report.refusals).toEqual([])
    const score = scoreAgainstGold(GOLD, report)
    const cell = must(score.properties.find((p) => p.key === 'MetricDef.equationId'), 'cell MetricDef.equationId')
    expect([cell.tp, cell.fp, cell.fn]).toEqual([0, 0, 6])
    expect([cell.precision, cell.recall, cell.f1]).toEqual([null, 0, null])
    expect(score.properties.some((p) => p.key === 'MetricDef.rank')).toBe(false)
  })
  it('keeps refused rows out of the score and counts them beside it', () => {
    const target = must(GOLD.objects.find((g) => g.primaryKey === 'EQ-M4'), 'EQ-M4')
    const candidates = GOLD.objects.map((g) => {
      const c = candidateFromGold(g, PASSAGES)
      if (g === target) c.props.colour = { value: 'red', quote: null, charStart: null, charEnd: null }
      return c
    })
    const report = gate(candidates, GOLD.links.map(linkFromGold), goldWorld(PASSAGES))
    expect(report.passedObjects.length).toBe(GOLD.objects.length - 1)
    const score = scoreAgainstGold(GOLD, report)
    expect(score.refusedByType).toEqual(report.counts.refusedByType)
    for (const key of Object.keys(target.properties)) {
      const cell = must(score.properties.find((p) => p.key === target.objectType + '.' + key), 'cell ' + key)
      expect(cell.fp, cell.key).toBe(0)
      expect(cell.fn, cell.key).toBeGreaterThan(0)
    }
  })
  it('prints one row of the table per property', () => {
    const score = scoreAgainstGold(GOLD, PERFECT)
    const table = formatScoreTable(score)
    console.log(table)
    const lines = table.split('\n')
    expect(lines.length).toBe(score.properties.length + score.links.length + 3)
    expect(lines[0]).toBe('property | tp | fp | fn | precision | recall | f1')
    for (const p of score.properties) expect(lines.some((l) => l.startsWith(p.key + ' | ')), p.key).toBe(true)
    expect(must(lines[lines.length - 2], 'the penultimate line').startsWith('micro | ')).toBe(true)
    expect(must(lines[lines.length - 1], 'the last line').startsWith('refused | ')).toBe(true)
  })
  it('scores a link by its identity triple, not by a property', () => {
    const computes = GOLD.links.filter((l) => l.linkType === 'computes')
    const dropped = must(computes.find((l) => computes.filter((x) => x.fromId === l.fromId).length === 1), 'a computes link whose source carries no other')
    const extra: GateCandidateLink = { ...linkFromGold(dropped), toPrimaryKey: 'NO_SUCH_METRIC' }
    const kept = GOLD.links.filter((l) => l !== dropped).map(linkFromGold)
    const report = gate(GOLD.objects.map((g) => candidateFromGold(g, PASSAGES)), [...kept, extra], goldWorld(PASSAGES))
    expect(report.refusals).toEqual([])
    const score = scoreAgainstGold(GOLD, report)
    const cell = must(score.links.find((l) => l.linkType === 'computes'), 'cell link:computes')
    expect([cell.tp, cell.fp, cell.fn]).toEqual([computes.length - 1, 1, 1])
    expect(score.properties.some((p) => p.key.startsWith('link:'))).toBe(false)
  })
  it('the SPEC-LIT slice this gold set is written against still starts at section 52 and ends before section 56', () => {
    const lines = readFileSync(path.join(REPO, 'rust', 'SPEC-LIT.md'), 'utf8').split('\n').slice(8280, 9904)
    expect(lines[0]?.startsWith('## 52.')).toBe(true)
    expect(lines.some((l) => l.startsWith('## 56.'))).toBe(false)
    expect(PASSAGES.size).toBe(36)
    for (const key of ['55.1', '55.2', '55.3', '55.4']) expect(PASSAGES.has(key), key).toBe(true)
  })
  it('loads the gold set with no chunk text at all, and still refuses a foreign document', () => {
    const bare = loadGoldSet('spec-lit-52-55', null)
    expect(bare.objects.length).toBeGreaterThanOrEqual(12)
    const threw = thrown(() => checkGoldSet(FOREIGN, 'spec-lit-52-55', null))
    expect(threw?.code).toBe('GOLD_FOREIGN_DOCUMENT')
  })
})
