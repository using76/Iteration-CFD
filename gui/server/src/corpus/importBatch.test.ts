// C5 e2e: a JRC 5.3.1 chunk, one candidate StandardClause, approve, and the mirror holds the row
// with its object_chunk provenance; deny leaves nothing. Real file-backed SQLite, no server, no port.
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { ONTOLOGY, buildRegistry } from '@cfd/shared'
import type { BaseType, LinkTypeDef, ObjectTypeDef, Principal, PropertyDef } from '@cfd/shared'
import type { DatabaseSync } from 'node:sqlite'
import { extractionRunId } from './extract.js'
import { candidateProperties, installProposeImport } from './importBatch.js'
import { EDIT_FUNCTIONS } from '../ontology/editset.js'
import { EngineError, createActionEngine, type OntologyActionEngine } from '../ontology/engine.js'
import { openOntologyStore, type OntologyStore } from '../ontology/store.js'
import type { ActionStore } from '../ontology/engine.js'
import type { CandidateLinkInput, CandidateObjectInput, CandidateValue } from './schema.js'

const p = (apiName: string, baseType: BaseType, displayName: string, description: string, nullable = false, extra: Partial<PropertyDef> = {}): PropertyDef =>
  ({ apiName, displayName, baseType, nullable, description, ...extra })
const en = (values: string[]): Partial<PropertyDef> => ({ valueType: 'enum', enumValues: values })

const STANDARD_CLAUSE: ObjectTypeDef = {
  apiName: 'StandardClause', displayName: 'Standard clause', pluralName: 'Standard clauses',
  description: 'A locator inside an external document, our one-line claim about it, and a public value only when a public source states it.',
  icon: 'file-text', source: { projection: '', paths: [] }, actionCreatedOnly: true,
  primaryKey: 'clauseId', titleKey: 'locator', ontologyVersion: '0.1.0',
  properties: [
    p('clauseId', 'string', 'Clause id', 'standardId + " / " + locator'),
    p('standardId', 'string', 'Standard', 'the Standard this clause is in'),
    p('locator', 'string', 'Locator', 'the clause number or table name, as the document prints it'),
    p('claim', 'string', 'Claim', 'our own one line about what the clause governs. Never the clause text.'),
    p('publicValue', 'string', 'Public value', 'the value, only when a public source states it', true),
    p('publicSource', 'string', 'Public source', 'the document id of the public source', true),
    p('verification', 'string', 'Verification', 'how well we know this', false, en(['VERIFIED', 'PARTIAL', 'UNVERIFIED'])),
  ],
}
const STANDARD: ObjectTypeDef = {
  apiName: 'Standard', displayName: 'Standard', pluralName: 'Standards', description: 'An external standard document.',
  icon: 'book', source: { projection: '', paths: [] }, actionCreatedOnly: true,
  primaryKey: 'standardId', titleKey: 'title', ontologyVersion: '0.1.0',
  properties: [
    p('standardId', 'string', 'Standard id', 'issuer:number:edition'), p('issuer', 'string', 'Issuer', 'who publishes it'),
    p('number', 'string', 'Number', 'the committee number'), p('edition', 'string', 'Edition', 'the edition'),
    p('title', 'string', 'Title', 'the title as printed'), p('licence', 'string', 'Licence', 'how it is licensed'),
    p('isPaywalled', 'boolean', 'Paywalled', 'whether buying it is the only way'),
  ],
}
const EQUIVALENT_TO: LinkTypeDef = {
  apiName: 'equivalentTo', displayName: 'Equivalent to',
  from: { apiName: 'equivalentTo', displayName: 'Equivalent to', objectType: 'StandardClause' },
  to: { apiName: 'equivalentClauses', displayName: 'Equivalent clauses', objectType: 'Standard' },
  cardinality: 'MANY_TO_ONE', backing: { kind: 'joinTable', projection: 'equivalentTo' }, ontologyVersion: '0.1.0',
}
const PROPOSE_IMPORT = ONTOLOGY.actionType('proposeImport')!
const REG = buildRegistry({ version: '0.1.0', objects: [STANDARD_CLAUSE, STANDARD], links: [EQUIVALENT_TO], actions: [PROPOSE_IMPORT] })
const BATCH = extractionRunId('doc:jrc-coc-2024', 'StandardClause', 'test-model')
const DOC = 'doc:jrc-coc-2024'
const CHUNK = 'doc:jrc-coc-2024#5.3.1'
const WHO: Principal = { kind: 'user', id: 'local', sessionId: null }

const JRC_531 =
  '5.3.1 Equipment environmental operating ranges. Data Centres should be designed and operated at ' +
  'their highest efficiency to deliver intake air to the IT equipment within the ASHRAE Class A2 ' +
  'allowable range, that is 10–35 °C at the ITE inlet. Operations in this range enable energy ' +
  'savings by reducing or eliminating overcooling.'
const span = (text: string, needle: string) => { const i = text.indexOf(needle); if (i < 0) throw new Error(`fixture: ${needle} not in chunk`); return { charStart: i, charEnd: i + needle.length } }
const val = (value: string | null, needle?: string): CandidateValue =>
  needle === undefined ? { value, quote: null, charStart: null, charEnd: null } : { value, quote: value, ...span(JRC_531, needle) }
const clauseRow = (candidateId: string, num: string, opts: { pv?: string | null; ps?: string | null; ver?: string } = {}, locate = false): CandidateObjectInput => {
  const props: Record<string, CandidateValue> = {
    clauseId: val(`JRC:CoC:2024 / ${num}`),
    locator: locate ? val(num, num) : val(num),
    claim: locate
      ? val('requires the ITE intake air to be within the ASHRAE class A2 allowable range', 'within the ASHRAE Class A2 allowable range')
      : val(`governs ${num}`),
    publicValue: opts.pv === undefined ? val(null) : locate && opts.pv !== null ? val(opts.pv, opts.pv) : val(opts.pv),
    verification: val(opts.ver ?? 'VERIFIED'),
    standardId: val('JRC:CoC:2024'),
    publicSource: opts.ps === undefined ? val(null) : val(opts.ps),
  }
  const spans = Object.values(props).filter((v) => v.charStart !== null)
  const cs = spans.length ? Math.min(...spans.map((v) => v.charStart!)) : null
  const ce = spans.length ? Math.max(...spans.map((v) => v.charEnd!)) : null
  return { candidateId, objectType: 'StandardClause', primaryKey: `JRC:CoC:2024 / ${num}`, documentId: DOC, chunkId: CHUNK, charStart: cs, charEnd: ce, props, extractedBy: 'test-model', extractedAt: 1 }
}
const L1: CandidateLinkInput = { candidateLinkId: 'l1', linkType: 'equivalentTo', fromObjectType: 'StandardClause', fromPrimaryKey: 'JRC:CoC:2024 / 5.3.1', toObjectType: 'Standard', toPrimaryKey: 'ASHRAE:TC9.9:5', documentId: DOC, chunkId: CHUNK, charStart: null, charEnd: null, extractedBy: 'test-model', extractedAt: 1 }

const cleanups: Array<() => void> = []
afterEach(() => { for (const c of cleanups.splice(0)) { try { c() } catch { /* already gone */ } } })

function makeEngine(extra: CandidateObjectInput[] = [], links: CandidateLinkInput[] = [L1]): { store: OntologyStore; adapter: ActionStore; engine: OntologyActionEngine; db: DatabaseSync } {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'cfd-import-'))
  const store = openOntologyStore({ path: path.join(tmp, 'ontology.db'), ontology: REG })
  cleanups.push(() => { store.close(); fs.rmSync(tmp, { recursive: true, force: true }) })
  store.corpus.putDocument({ documentId: DOC, title: 'JRC 2024 Best Practice Guidelines for the EU Code of Conduct on Data Centre Energy Efficiency', tier: 'A', licence: 'CC BY 4.0', isPaywalled: false, accessNote: null, sourcePath: 'corpus/jrc-coc-2024.md', sourceUrl: null, sha256: null, importedAt: 1 })
  store.corpus.putChunks([{ chunkId: CHUNK, documentId: DOC, locator: '5.3.1', part: 1, heading: '', text: JRC_531, charStart: 0, charEnd: JRC_531.length, ordinal: 0, importedAt: 1 }])
  store.corpus.putCandidateObjects([clauseRow('c1', '5.3.1', { pv: '10–35 °C', ps: DOC }, true), clauseRow('c2', '5.3.2'), clauseRow('c3', '5.1.4'), clauseRow('c4', '5.3.3', { pv: 'beyond A2' }), ...extra])
  store.corpus.putCandidateLinks(links)
  store.put({ type: 'Standard', id: 'ASHRAE:TC9.9:5', props: { standardId: 'ASHRAE:TC9.9:5', issuer: 'ASHRAE', number: 'TC9.9', edition: '5', title: 'Thermal Guidelines for Data Processing Environments', licence: 'ASHRAE publication', isPaywalled: false }, sourcePath: 'fixture', importedAt: 1 })
  const adapter = installProposeImport(store, REG)
  const engine = createActionEngine({
    actions: [PROPOSE_IMPORT], registry: REG, store: adapter,
    server: { workspaceRoot: tmp, now: () => Date.now(), gitHead: async () => ({ sha: null, dirty: null }), runs: { list: () => [], get: () => undefined, start: async () => { throw new Error('unused') } } },
  })
  return { store, adapter, engine, db: store.raw() }
}
const propose = (E: ReturnType<typeof makeEngine>) => E.engine.propose('proposeImport', { batchId: BATCH, maxObjects: null, includeLinks: null, note: null }, WHO)
const approveAndApply = async (engine: OntologyActionEngine, id: string, who: Principal) => {
  if (typeof (engine as { authorise?: unknown }).authorise === 'function') engine.authorise(id, who)
  return engine.apply(id, who)
}
const n = (db: DatabaseSync, sql: string): number => (db.prepare(sql).get() as { n: number }).n

describe('proposeImport', () => {
  it('declares proposeImport as a function-backed action with four parameters and no side effects', () => {
    const def = ONTOLOGY.actionType('proposeImport')
    expect(def).not.toBeNull()
    expect(def!.rules).toBeNull()
    expect(def!.functionRule).toEqual({ function: 'proposeImport' })
    expect(def!.maxEdits).toBe(400)
    expect(def!.parameters.map((x) => x.apiName)).toEqual(['batchId', 'maxObjects', 'includeLinks', 'note'])
    expect(def!.criteria.map((c) => c.id)).toEqual(['batchNotEmpty', 'batchWithinCap', 'noStoredClauseText', 'candidatesRefused'])
    expect(def!.permission.policy).toBe('ask')
    expect(def!.sideEffects).toEqual([])
  })

  it('a JRC 5.3.1 candidate becomes one StandardClause create carrying its public value and its public source', async () => {
    const E = makeEngine()
    const pr = await propose(E)
    expect(pr.state).toBe('rendered')
    expect(pr.edits!.objects.length).toBe(3)
    const c1 = pr.edits!.objects.find((o) => o.id === 'JRC:CoC:2024 / 5.3.1')!
    expect(c1.op).toBe('create')
    expect(c1.objectType).toBe('StandardClause')
    expect(c1.before).toBeNull()
    expect(c1.after!.publicValue).toBe('10–35 °C')
    expect(c1.after!.publicSource).toBe('doc:jrc-coc-2024')
    expect(c1.after!.verification).toBe('VERIFIED')
    expect(pr.edits!.links.length).toBe(1)
    expect(pr.edits!.summary.split('\n')[0]).toBe(`import batch ${BATCH} from doc:jrc-coc-2024: 3 object(s), 1 link(s), 1 refused`)
  })

  it('approving it writes the row, its object_chunk provenance and one edit-log row in one transaction', async () => {
    const E = makeEngine()
    const pr = await propose(E)
    await approveAndApply(E.engine, pr.proposalId, WHO)
    expect(E.store.get('StandardClause', 'JRC:CoC:2024 / 5.3.1')!.props.publicValue).toBe('10–35 °C')
    expect(E.store.links({ linkType: 'equivalentTo' }).length).toBe(1)
    expect(n(E.db, `SELECT count(*) AS n FROM object_chunk WHERE object_id = 'JRC:CoC:2024 / 5.3.1'`)).toBe(4)
    const pv = E.db.prepare(`SELECT * FROM object_chunk WHERE object_id = 'JRC:CoC:2024 / 5.3.1' AND property = 'publicValue'`).get() as Record<string, unknown>
    expect(pv.chunk_id).toBe(CHUNK)
    expect(pv.matched_by).toBe('human')
    expect(pv.extracted_by).toBe('action:proposeImport')
    expect(pv.document_id).toBe(DOC)
    expect(JRC_531.slice(Number(pv.char_start), Number(pv.char_end))).toBe('10–35 °C')
    const log = E.adapter.listEditLog()
    expect(log.length).toBe(1)
    expect(log[0].action).toBe('proposeImport')
  })

  it('denying the proposal leaves no row, no span and no edit-log entry', async () => {
    const E = makeEngine()
    const pr = await propose(E)
    await E.engine.reject(pr.proposalId, WHO, 'not reviewed')
    expect(E.store.get('StandardClause', 'JRC:CoC:2024 / 5.3.1')).toBeNull()
    expect(n(E.db, 'SELECT count(*) AS n FROM object_chunk')).toBe(0)
    expect(E.adapter.listEditLog().length).toBe(0)
    await expect(E.engine.apply(pr.proposalId, WHO)).rejects.toMatchObject({ code: 'NOT_AUTHORISED' })
  })

  it('a candidate the gate refuses is dropped, listed by the cell that refused it, and counted by one warning', async () => {
    const E = makeEngine([clauseRow('c5', '5.3.5', { ver: 'MAYBE' })])
    const pr = await propose(E)
    expect(pr.edits!.objects.length).toBe(3)
    expect(pr.edits!.summary).toContain('\nrefused: M4 StandardClause[JRC:CoC:2024 / 5.3.5].verification')
    expect(pr.warnings.length).toBe(1)
    expect(pr.warnings[0].id).toBe('candidatesRefused')
    expect(pr.warnings[0].message).toContain('2 of 5')
    expect(pr.blocking).toEqual([])
  })

  it('a publicValue with no publicSource is refused by name and never becomes a row', async () => {
    const E = makeEngine()
    const pr = await propose(E)
    expect(pr.edits!.summary).toContain(
      'refused: PUBLICVALUE StandardClause[JRC:CoC:2024 / 5.3.3].publicValue has no publicSource; a public value must name the public source that states it')
    expect(pr.edits!.objects.find((o) => o.id === 'JRC:CoC:2024 / 5.3.3')).toBeUndefined()
  })

  it('a candidate carrying a property named text blocks the whole proposal and computes no edit set', async () => {
    const row = clauseRow('c_text', '5.3.9')
    const props: Record<string, CandidateValue> = { ...row.props, text: { value: 'clause text', quote: null, charStart: null, charEnd: null } }
    const E2 = makeEngine([{ ...row, candidateId: 'c_t', primaryKey: 'JRC:CoC:2024 / 5.3.9', props }])
    const pr = await propose(E2)
    expect(pr.state).toBe('rejected')
    expect(pr.edits).toBeNull()
    expect(pr.blocking.length).toBe(1)
    expect(pr.blocking[0].id).toBe('noStoredClauseText')
    expect(pr.blocking[0].message).toContain('carries a property named "text"')
    expect(pr.blocking[0].message).toContain('c_t')
  })

  it('a batch over its cap is rejected naming the count and the cap, and imports nothing', async () => {
    const extra = Array.from({ length: 198 }, (_, i) => clauseRow(`c_b${i}`, `9.${i}`))
    const E = makeEngine(extra, [])
    const pr = await propose(E)
    expect(pr.state).toBe('rejected')
    expect(pr.blocking[0].id).toBe('batchWithinCap')
    expect(pr.blocking[0].message).toContain('201')
    expect(pr.blocking[0].message).toContain('200')
    expect(E.store.count('StandardClause')).toBe(0)
  })

  it('an unknown batch is rejected naming the batch', async () => {
    const E = makeEngine()
    const pr = await E.engine.propose('proposeImport', { batchId: 'b_nope', maxObjects: null, includeLinks: null, note: null }, WHO)
    expect(pr.state).toBe('rejected')
    expect(pr.blocking[0].id).toBe('batchNotEmpty')
    expect(pr.blocking[0].message).toContain('b_nope')
  })

  it('re-proposing an applied batch yields modify edits with the same values and leaves one row', async () => {
    const E = makeEngine()
    const first = await propose(E)
    await approveAndApply(E.engine, first.proposalId, WHO)
    const second = await propose(E)
    expect(second.edits!.objects.length).toBe(3)
    for (const o of second.edits!.objects) expect(o.op).toBe('modify')
    const c1 = second.edits!.objects.find((o) => o.id === 'JRC:CoC:2024 / 5.3.1')!
    expect(c1.before!.publicValue).toBe(c1.after!.publicValue)
    await approveAndApply(E.engine, second.proposalId, WHO)
    expect(E.store.count('StandardClause')).toBe(3)
    expect(E.adapter.listEditLog().length).toBe(2)
  })

  it('the summary quotes an id containing a space and caps its rows at forty', async () => {
    const extra = Array.from({ length: 42 }, (_, i) => clauseRow(`c_q${i}`, `9.${i + 10}`))
    const E = makeEngine(extra, [])
    const pr = await propose(E)
    expect(pr.edits!.summary).toContain('create StandardClause "JRC:CoC:2024 / 5.3.1"')
    const rowLines = pr.edits!.summary.split('\n').filter((l) => /^(create|modify|delete) |^\+ link /.test(l))
    expect(rowLines.length).toBe(40)
    expect(pr.edits!.summary).toContain('\n… and 5 more')
    expect(pr.edits!.objects.length).toBe(45)
  })

  it('the import writes nothing to candidate_object, candidate_link or unmapped_span', async () => {
    const E = makeEngine()
    const before = {
      co: n(E.db, 'SELECT count(*) AS n FROM candidate_object'),
      cl: n(E.db, 'SELECT count(*) AS n FROM candidate_link'),
      us: n(E.db, 'SELECT count(*) AS n FROM unmapped_span'),
      by: n(E.db, "SELECT count(*) AS n FROM candidate_object WHERE extracted_by <> 'test-model'"),
    }
    const pr = await propose(E)
    await approveAndApply(E.engine, pr.proposalId, WHO)
    const after = {
      co: n(E.db, 'SELECT count(*) AS n FROM candidate_object'),
      cl: n(E.db, 'SELECT count(*) AS n FROM candidate_link'),
      us: n(E.db, 'SELECT count(*) AS n FROM unmapped_span'),
      by: n(E.db, "SELECT count(*) AS n FROM candidate_object WHERE extracted_by <> 'test-model'"),
    }
    expect(after).toEqual(before)
    expect(after.co).toBe(4)
    expect(after.cl).toBe(1)
    expect(after.us).toBe(0)
    expect(after.by).toBe(0)
  })

  it('the edit function is registered under exactly one name', () => {
    const E = makeEngine()
    expect(Object.keys(EDIT_FUNCTIONS)).toEqual(['proposeImport'])
    expect(EDIT_FUNCTIONS.proposeImport).toBeTypeOf('function')
    expect(EDIT_FUNCTIONS.proposeImport!.length).toBe(2)
  })

  it('candidateProperties reads both stored shapes and refuses what is not an object', () => {
    const a = candidateProperties('{"a":{"value":"1","quote":"1","charStart":0,"charEnd":1},"b":{"value":null,"charStart":null}}')
    expect(a.a).toEqual({ value: '1', quote: '1', charStart: 0, charEnd: 1 })
    expect(a.b).toEqual({ value: null, quote: null, charStart: null, charEnd: null })
    const b = candidateProperties('{"c":"flat"}')
    expect(b.c).toEqual({ value: 'flat', quote: null, charStart: null, charEnd: null })
    expect(() => candidateProperties('[1,2]')).toThrow()
  })
})
