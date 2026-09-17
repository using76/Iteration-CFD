// gui/server/src/ontology/store.test.ts — the mirror's own tests. Tests 1, 2 and
// 14b run against the real ONTOLOGY; every other test builds its own two-type
// fixture registry (D9), because what the store owes the registry is structure,
// and the twelve real types will grow under N3-N6.
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { ONTOLOGY, buildRegistry } from '@cfd/shared'
import type { LinkTypeDef, ObjectTypeDef, OntologyRegistry, PropertyDef } from '@cfd/shared'
import { loadConfig } from '../config.js'
import { silentLogger, type Logger } from '../log.js'
import { OntologyStoreError, debugSql, openOntologyStore, ontologyDbPath, resolveLinkSide, type OntologyStore } from './store.js'
import { corpusTableNames } from '../corpus/schema.js'

let dir: string
beforeAll(() => {
  dir = fs.mkdtempSync(path.join(os.tmpdir(), 'cfd-ontstore-'))
})
afterAll(() => {
  fs.rmSync(dir, { recursive: true, force: true })
})

// ---- the fixture registry (every one of the eleven BaseTypes) ---------------

const P = (apiName: string, baseType: PropertyDef['baseType'], nullable = false, extra: Partial<PropertyDef> = {}): PropertyDef =>
  ({ apiName, displayName: apiName, baseType, nullable, description: `fixture property ${apiName}`, ...extra })

function testRunDef(v: string): ObjectTypeDef {
  return {
    apiName: 'TestRun', displayName: 'Test run', pluralName: 'Test runs',
    description: 'fixture run', icon: 'run',
    source: { projection: 'testRuns', paths: ['gui/runs/*/run.json'] },
    primaryKey: 'runId', titleKey: 'label',
    properties: [
      P('runId', 'string'),
      P('label', 'string', true),
      P('iter', 'integer'),
      P('logLines', 'long'),
      P('converged', 'boolean'),
      P('startedAt', 'timestamp'),
      P('day', 'date'),
      P('score', 'double', true),
      P('tags', 'array', false, { items: 'string' }),
      P('quality', 'struct', false, { fields: { min: { baseType: 'double', main: true }, max: { baseType: 'double' } } }),
      P('payload', 'json', false),
      P('mode', 'string', false, { valueType: 'enum', enumValues: ['real', 'demo'] }),
      P('blobRef', 'attachmentRef', true),
      P('sha', 'string', true),
    ],
    ontologyVersion: v,
  }
}

function testCommitDef(v: string): ObjectTypeDef {
  return {
    apiName: 'TestCommit', displayName: 'Test commit', pluralName: 'Test commits',
    description: 'fixture commit', icon: 'commit',
    source: { projection: 'testCommits', paths: ['.git'] },
    primaryKey: 'sha', titleKey: 'subject',
    properties: [P('sha', 'string'), P('subject', 'string')],
    ontologyVersion: v,
  }
}

function testNoteDef(v: string): ObjectTypeDef {
  return {
    apiName: 'TestNote', displayName: 'Test note', pluralName: 'Test notes',
    description: 'fixture note', icon: 'note',
    source: { projection: 'testNotes', paths: ['notes/*.md'] },
    primaryKey: 'noteId', titleKey: 'text',
    properties: [P('noteId', 'string'), P('text', 'string')],
    ontologyVersion: v,
  }
}

function testAtCommitDef(v: string): LinkTypeDef {
  return {
    apiName: 'testAtCommit', displayName: 'Test at commit',
    from: { apiName: 'atCommit', displayName: 'Commit', objectType: 'TestRun' },
    to: { apiName: 'runs', displayName: 'Runs', objectType: 'TestCommit' },
    cardinality: 'MANY_TO_ONE',
    backing: { kind: 'foreignKey', onType: 'TestRun', property: 'sha' },
    ontologyVersion: v,
  }
}

function fixtureOntology(version = '1.2.0', extraTypes: ObjectTypeDef[] = []): OntologyRegistry {
  const objects = [testRunDef(version), testCommitDef(version), ...extraTypes.map((t) => ({ ...t, ontologyVersion: version }))]
  const links = [testAtCommitDef(version)]
  return buildRegistry({ version, objects, links, actions: [] })
}

function runProps(id: string, over: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    runId: id, label: `run ${id}`, iter: 1, logLines: 0, converged: false,
    startedAt: '2026-09-15T00:00:00.000Z', day: '2026-09-15', score: null,
    tags: [], quality: { min: 0, max: 1 }, payload: {}, mode: 'real',
    blobRef: null, sha: null, ...over,
  }
}

const putRun = (s: OntologyStore, id: string, over: Record<string, unknown> = {}, sourcePath = 'gui/runs/x.json'): void =>
  s.put({ type: 'TestRun', id, props: runProps(id, over), sourcePath })

const putCommit = (s: OntologyStore, sha: string, sourcePath = 'gui/commits.json'): void =>
  s.put({ type: 'TestCommit', id: sha, props: { sha, subject: `commit ${sha}` }, sourcePath })

const memStore = (ont: OntologyRegistry, log: Logger = silentLogger): OntologyStore =>
  openOntologyStore({ path: ':memory:', ontology: ont, log })

function recordingLogger(): { log: Logger; warns: string[] } {
  const warns: string[] = []
  const log: Logger = {
    level: 'error',
    debug: () => {},
    info: () => {},
    warn: (m) => { warns.push(String(m)) },
    error: () => {},
    child: () => log,
  }
  return { log, warns }
}

describe('ontology store', () => {
  it('generates_one_table_per_object_type_from_the_real_registry', () => {
    const s = memStore(ONTOLOGY)
    const names = debugSql(s, "SELECT name FROM sqlite_master WHERE type = 'table'").map((r) => String(r.name)).sort()
    // The L1 corpus storey is migrated into the same file on open and is not
    // generated from the registry; this test is about the generated half.
    const generated = names.filter((n) => !corpusTableNames().includes(n))
    expect(generated).toEqual(['links', 'meta', ...ONTOLOGY.objectTypes.map((t) => `obj_${t.apiName}`)].sort())
    expect(generated.length).toBe(ONTOLOGY.objectTypes.length + 2)
    s.close()
  })

  it('every_object_table_has_the_five_DM261_columns', () => {
    const s = memStore(ONTOLOGY)
    for (const t of ONTOLOGY.objectTypes) {
      const cols = debugSql(s, `PRAGMA table_info("obj_${t.apiName}")`)
      expect(cols.map((c) => c.name)).toEqual(['id', 'title', 'props', 'source_path', 'imported_at'])
      expect(cols.map((c) => Number(c.notnull))).toEqual([0, 0, 1, 1, 1])
      expect(cols.map((c) => Number(c.pk))).toEqual([1, 0, 0, 0, 0])
    }
    s.close()
  })

  it('put_then_get_roundtrips_every_base_type', () => {
    const s = memStore(fixtureOntology())
    s.put({
      type: 'TestRun', id: 'r_full', sourcePath: 'gui/runs/r_full/run.json',
      props: runProps('r_full', {
        iter: 12, logLines: 9007199254740991, converged: true,
        startedAt: '2026-09-15T04:05:06.000Z', day: '2026-09-15',
        score: 0.0123456789, tags: ['a', 'b'], quality: { min: 0.5, max: 2.5 },
        payload: { nested: { a: 1 }, list: [1, 'x', null] },
        mode: 'demo', blobRef: 'att_1', sha: 'abc',
      }),
    })
    const row = s.get('TestRun', 'r_full')
    expect(row).not.toBeNull()
    expect(row!.props.runId).toBe('r_full')
    expect(row!.props.sha).toBe('abc')
    expect(row!.props.label).toBe('run r_full')
    expect(row!.props.iter).toBe(12)
    expect(row!.props.logLines).toBe(9007199254740991)
    expect(typeof row!.props.logLines).toBe('number')
    expect(row!.props.converged).toBe(true)
    expect(row!.props.startedAt).toBe(Date.parse('2026-09-15T04:05:06.000Z'))
    expect(Number.isInteger(row!.props.startedAt)).toBe(true)
    expect(row!.props.day).toBe(Date.parse('2026-09-15'))
    expect(row!.props.score).toBeCloseTo(0.0123456789, 12)
    expect(row!.props.tags).toEqual(['a', 'b'])
    expect(row!.props.quality).toEqual({ min: 0.5, max: 2.5 })
    expect(row!.props.payload).toEqual({ nested: { a: 1 }, list: [1, 'x', null] })
    expect(row!.props.blobRef).toBe('att_1')
    expect(row!.props.mode).toBe('demo')
    expect(Object.keys(row!.props).length).toBe(14)
    s.close()
  })

  it('title_is_taken_from_the_title_key', () => {
    const s = memStore(fixtureOntology())
    // ObjectInput has no title field: this literal typechecks without one.
    s.put({ type: 'TestRun', id: 'r_t', props: runProps('r_t', { label: 'my label' }), sourcePath: 'x' })
    expect(s.get('TestRun', 'r_t')!.title).toBe('my label')
    s.put({ type: 'TestRun', id: 'r_n', props: runProps('r_n', { label: null }), sourcePath: 'x' })
    expect(s.get('TestRun', 'r_n')!.title).toBeNull()
    s.close()
  })

  it('put_refuses_a_pk_that_disagrees_with_the_primary_key_property', () => {
    const s = memStore(fixtureOntology())
    try {
      s.put({ type: 'TestRun', id: 'r_1', props: runProps('r_2'), sourcePath: 'x' })
      expect.unreachable('put should have thrown')
    } catch (e) {
      const err = e as OntologyStoreError
      expect(err.code).toBe('PK_MISMATCH')
      expect(err.message).toContain('r_1')
      expect(err.message).toContain('r_2')
    }
    s.close()
  })

  it('put_fills_a_missing_nullable_property_with_null_and_refuses_a_missing_non_nullable_one', () => {
    const s = memStore(fixtureOntology())
    s.put({ type: 'TestRun', id: 'r_null', props: runProps('r_null', { score: undefined }), sourcePath: 'x' })
    const row = s.get('TestRun', 'r_null')!
    expect('score' in row.props).toBe(true)
    expect(row.props.score).toBeNull()
    try {
      s.put({ type: 'TestRun', id: 'r_i', props: runProps('r_i', { iter: undefined }), sourcePath: 'x' })
      expect.unreachable('put should have thrown')
    } catch (e) {
      const err = e as OntologyStoreError
      expect(err.code).toBe('MISSING_PROPERTY')
      expect(err.message).toContain('iter')
    }
    s.close()
  })

  it('put_refuses_an_undeclared_property_and_an_empty_source_path', () => {
    const s = memStore(fixtureOntology())
    try {
      s.put({ type: 'TestRun', id: 'r_u', props: { ...runProps('r_u'), nope: 1 }, sourcePath: 'x' })
      expect.unreachable('put should have thrown')
    } catch (e) {
      const err = e as OntologyStoreError
      expect(err.code).toBe('UNKNOWN_PROPERTY')
      expect(err.message).toContain('nope')
      expect(err.message).toContain('TestRun')
    }
    try {
      s.put({ type: 'TestRun', id: 'r_s', props: runProps('r_s'), sourcePath: '   ' })
      expect.unreachable('put should have thrown')
    } catch (e) {
      expect((e as OntologyStoreError).code).toBe('NO_SOURCE')
    }
    s.close()
  })

  it('timestamps_are_normalised_to_epoch_ms_and_a_non_date_is_refused', () => {
    const s = memStore(fixtureOntology())
    const iso = '2026-09-15T04:05:06.000Z'
    s.put({ type: 'TestRun', id: 'r_ts', props: runProps('r_ts', { startedAt: iso }), sourcePath: 'x' })
    expect(s.get('TestRun', 'r_ts')!.props.startedAt).toBe(Date.parse(iso))
    s.put({ type: 'TestRun', id: 'r_n', props: runProps('r_n', { startedAt: Date.parse(iso) }), sourcePath: 'x' })
    const again = s.get('TestRun', 'r_n')!.props.startedAt
    expect(again).toBe(Date.parse(iso))
    expect(Number.isInteger(again)).toBe(true)
    try {
      s.put({ type: 'TestRun', id: 'r_bad', props: runProps('r_bad', { startedAt: 'not a date' }), sourcePath: 'x' })
      expect.unreachable('put should have thrown')
    } catch (e) {
      const err = e as OntologyStoreError
      expect(err.code).toBe('BAD_TIMESTAMP')
      expect(err.message).toContain('startedAt')
    }
    s.close()
  })

  it('put_is_an_upsert_so_a_second_import_leaves_one_row', () => {
    const s = memStore(fixtureOntology())
    s.put({ type: 'TestRun', id: 'r_1', props: runProps('r_1', { label: 'first' }), sourcePath: 'a.json', importedAt: 1000 })
    s.put({ type: 'TestRun', id: 'r_1', props: runProps('r_1', { label: 'second' }), sourcePath: 'a.json', importedAt: 2000 })
    expect(s.count('TestRun')).toBe(1)
    const row = s.get('TestRun', 'r_1')!
    expect(row.props.label).toBe('second')
    expect(row.importedAt).toBe(2000)
    s.close()
  })

  it('query_filters_by_equality_orders_and_pages', () => {
    const s = memStore(fixtureOntology())
    const rows: [string, number, string][] = [['a', 1, 'demo'], ['b', 2, 'real'], ['c', 3, 'demo'], ['d', 4, 'real'], ['e', 5, 'demo']]
    for (const [id, iter, mode] of rows) putRun(s, id, { iter, mode })
    const demo = s.query({ type: 'TestRun', where: [{ property: 'mode', equals: 'demo' }] })
    expect(demo.map((r) => r.id).sort()).toEqual(['a', 'c', 'e'])
    const desc = s.query({ type: 'TestRun', orderBy: { property: 'iter', direction: 'desc' } })
    expect(desc.map((r) => r.props.iter)).toEqual([5, 4, 3, 2, 1])
    const page = s.query({ type: 'TestRun', orderBy: { property: 'iter', direction: 'desc' }, limit: 2, offset: 1 })
    expect(page.map((r) => r.id)).toEqual(['d', 'c'])
    s.close()
  })

  it('query_matches_a_null_with_is_null', () => {
    const s = memStore(fixtureOntology())
    putRun(s, 'r_null', { score: null })
    putRun(s, 'r_one', { score: 1 })
    const hits = s.query({ type: 'TestRun', where: [{ property: 'score', equals: null }] })
    expect(hits.length).toBe(1)
    expect(hits[0].id).toBe('r_null')
    s.close()
  })

  it('query_refuses_an_undeclared_filter_property_and_an_over_large_limit', () => {
    const s = memStore(fixtureOntology())
    putRun(s, 'r_1')
    try {
      s.query({ type: 'TestRun', where: [{ property: 'nope', equals: 1 }] })
      expect.unreachable('query should have thrown')
    } catch (e) {
      expect((e as OntologyStoreError).code).toBe('UNKNOWN_PROPERTY')
    }
    try {
      s.query({ type: 'TestRun', limit: 1001 })
      expect.unreachable('query should have thrown')
    } catch (e) {
      const err = e as OntologyStoreError
      expect(err.code).toBe('BAD_LIMIT')
      expect(err.message).toContain('1001')
    }
    try {
      s.query({ type: 'TestRun', offset: -1 })
      expect.unreachable('query should have thrown')
    } catch (e) {
      expect((e as OntologyStoreError).code).toBe('BAD_LIMIT')
    }
    putManyRuns(s, 101)
    expect(s.query({ type: 'TestRun' }).length).toBe(100)
    s.close()
  })

  function putManyRuns(s: OntologyStore, n: number): void {
    const rows = Array.from({ length: n }, (_, i) => ({
      type: 'TestRun', id: `bulk_${String(i).padStart(3, '0')}`, props: runProps(`bulk_${String(i).padStart(3, '0')}`), sourcePath: 'bulk.json',
    }))
    s.putMany(rows)
  }

  it('links_live_in_one_table_and_are_read_from_both_ends', () => {
    const s = memStore(fixtureOntology())
    putRun(s, 'r_1', { sha: 'abc' })
    putCommit(s, 'abc')
    s.putLink({ type: 'testAtCommit', fromId: 'r_1', toId: 'abc', sourcePath: 'gui/x.json' })
    const raw = debugSql(s, 'SELECT type, from_type, to_type FROM links')
    expect(raw.length).toBe(1)
    expect(raw[0].type).toBe('testAtCommit')
    expect(raw[0].from_type).toBe('TestRun')
    expect(raw[0].to_type).toBe('TestCommit')
    expect(s.links({ from: { type: 'TestRun', id: 'r_1' } }).length).toBe(1)
    expect(s.links({ to: { type: 'TestCommit', id: 'abc' } }).length).toBe(1)
    s.putLink({ type: 'testAtCommit', fromId: 'r_1', toId: 'abc', sourcePath: 'gui/x.json' })
    expect(debugSql(s, 'SELECT type FROM links').length).toBe(1)
    s.close()
  })

  it('traverse_walks_a_link_by_the_side_api_name_in_both_directions', () => {
    const s = memStore(fixtureOntology())
    putRun(s, 'r_1', { sha: 'abc' })
    putRun(s, 'r_2', { sha: 'abc' })
    putCommit(s, 'abc')
    s.putLink({ type: 'testAtCommit', fromId: 'r_1', toId: 'abc', sourcePath: 'x' })
    s.putLink({ type: 'testAtCommit', fromId: 'r_2', toId: 'abc', sourcePath: 'x' })
    expect(s.traverse({ type: 'TestRun', id: 'r_1' }, 'atCommit').map((r) => r.id)).toEqual(['abc'])
    expect(s.traverse({ type: 'TestCommit', id: 'abc' }, 'runs').map((r) => r.id).sort()).toEqual(['r_1', 'r_2'])
    try {
      s.traverse({ type: 'TestRun', id: 'r_1' }, 'nope')
      expect.unreachable('traverse should have thrown')
    } catch (e) {
      expect((e as OntologyStoreError).code).toBe('UNKNOWN_LINK')
    }
    s.close()
  })

  it('the_real_registry_traverses_atCommit_from_a_run', () => {
    const lt = ONTOLOGY.linkType('atCommit')
    expect(lt).not.toBeNull()
    expect(lt!.from.objectType).toBe('Run')
    expect(lt!.from.apiName).toBe('atCommit')
    expect(lt!.to.objectType).toBe('Commit')
    expect(lt!.to.apiName).toBe('runs')
    const sha = 'a'.repeat(40)
    const s = memStore(ONTOLOGY)
    s.put({
      type: 'Commit', id: sha, sourcePath: 'test',
      props: { sha, shortSha: sha.slice(0, 7), subject: 'subject', author: 't', authoredAt: '2026-09-15T00:00:00.000Z', committedAt: '2026-09-15T00:00:00.000Z', nFiles: 1, branch: null },
    })
    s.put({
      type: 'Run', id: 'r_real', sourcePath: 'test',
      props: {
        runId: 'r_real', binary: 'solver', argv: ['solver'], status: 'done',
        startedAt: '2026-09-15T00:00:00.000Z', iter: 1, written: [], converged: true,
        logLines: 0, mode: 'demo', gitSha: sha,
      },
    })
    s.putLink({ type: 'atCommit', fromId: 'r_real', toId: sha, sourcePath: 'test' })
    expect(s.traverse({ type: 'Run', id: 'r_real' }, 'atCommit').map((r) => r.id)).toEqual([sha])
    expect(s.traverse({ type: 'Commit', id: sha }, 'runs').map((r) => r.id)).toEqual(['r_real'])
    expect(resolveLinkSide(ONTOLOGY, 'Run', 'atCommit').def.apiName).toBe('atCommit')
    expect(resolveLinkSide(ONTOLOGY, 'Run', 'atCommit').direction).toBe('forward')
    expect(resolveLinkSide(ONTOLOGY, 'Commit', 'runs').direction).toBe('reverse')
    s.close()
  })

  it('traverse_skips_a_dangling_target_and_warns_once', () => {
    const { log, warns } = recordingLogger()
    const s = memStore(fixtureOntology(), log)
    putRun(s, 'r_1', { sha: 'ghost' })
    s.putLink({ type: 'testAtCommit', fromId: 'r_1', toId: 'ghost', sourcePath: 'x' })
    expect(s.traverse({ type: 'TestRun', id: 'r_1' }, 'atCommit')).toEqual([])
    expect(warns.length).toBe(1)
    expect(warns[0]).toContain('1')
    expect(warns[0]).toContain('testAtCommit')
    s.close()
  })

  it('a_transaction_rolls_back_every_edit_when_the_body_throws', () => {
    const s = memStore(fixtureOntology())
    expect(s.count('TestRun')).toBe(0)
    expect(() => s.tx(() => {
      putRun(s, 'a')
      putRun(s, 'b')
      throw new Error('boom')
    })).toThrow('boom')
    expect(s.count('TestRun')).toBe(0)
    putRun(s, 'c')
    expect(s.count('TestRun')).toBe(1)
    s.close()
  })

  it('a_nested_transaction_joins_the_outer_one', () => {
    const s = memStore(fixtureOntology())
    expect(() => s.tx(() => {
      putRun(s, 'a')
      s.tx(() => { putRun(s, 'b') })
      throw new Error('boom')
    })).toThrow('boom')
    expect(s.count('TestRun')).toBe(0)
    expect(() => s.tx(() => {
      putRun(s, 'a')
      s.tx(() => {
        putRun(s, 'b')
        throw new Error('inner')
      })
    })).toThrow('inner')
    expect(s.count('TestRun')).toBe(0)
    s.close()
  })

  it('tx_refuses_an_async_body_instead_of_committing_early', () => {
    const s = memStore(fixtureOntology())
    putRun(s, 'seed')
    const before = s.count('TestRun')
    expect(() => s.tx((() => Promise.resolve(1)) as unknown as () => number))
      .toThrowError(expect.objectContaining({ code: 'ASYNC_TX' }))
    expect(s.count('TestRun')).toBe(before)
    s.tx(() => putRun(s, 'after'))
    expect(s.count('TestRun')).toBe(before + 1)
    s.close()
  })

  it('the_rows_survive_close_and_reopen_of_a_real_file', () => {
    const file = path.join(dir, 'o.db')
    const s = openOntologyStore({ path: file, ontology: fixtureOntology() })
    putRun(s, 'r_1', { sha: 'abc' })
    putRun(s, 'r_2')
    putRun(s, 'r_3')
    putCommit(s, 'abc')
    s.putLink({ type: 'testAtCommit', fromId: 'r_1', toId: 'abc', sourcePath: 'x' })
    s.setMeta('probe', '1')
    s.close()
    expect(fs.readdirSync(dir)).toEqual(['o.db'])
    const s2 = openOntologyStore({ path: file, ontology: fixtureOntology() })
    expect(s2.opened).toBe('opened')
    expect(s2.count('TestRun')).toBe(3)
    expect(s2.links({}).length).toBe(1)
    expect(s2.meta('probe')).toBe('1')
    s2.close()
  })

  it('a_minor_version_bump_re_imports_and_a_major_bump_refuses_by_name', () => {
    const file = path.join(dir, 'versions.db')
    const s = openOntologyStore({ path: file, ontology: fixtureOntology('1.2.0') })
    expect(s.opened).toBe('created')
    putRun(s, 'r_1')
    putRun(s, 'r_2')
    s.close()
    const minor = openOntologyStore({ path: file, ontology: fixtureOntology('1.3.0') })
    expect(minor.opened).toBe('reimport')
    expect(minor.count('TestRun')).toBe(0)
    minor.close()
    expect(() => openOntologyStore({ path: file, ontology: fixtureOntology('2.0.0') }))
      .toThrowError(expect.objectContaining({ code: 'MIRROR_MAJOR_BEHIND' }))
    let msg = ''
    try {
      openOntologyStore({ path: file, ontology: fixtureOntology('2.0.0') })
    } catch (e) {
      msg = (e as Error).message
    }
    expect(msg).toContain('MIGRATIONS')
    expect(() => openOntologyStore({ path: file, ontology: fixtureOntology('1.1.0') }))
      .toThrowError(expect.objectContaining({ code: 'MIRROR_AHEAD' }))
  })

  it('schema_drift_at_an_unchanged_version_is_refused_by_name', () => {
    const file = path.join(dir, 'drift.db')
    openOntologyStore({ path: file, ontology: fixtureOntology('1.2.0') }).close()
    try {
      openOntologyStore({ path: file, ontology: fixtureOntology('1.2.0', [testNoteDef('1.2.0')]) })
      expect.unreachable('open should have thrown')
    } catch (e) {
      const err = e as OntologyStoreError
      expect(err.code).toBe('MIRROR_SCHEMA_DRIFT')
      expect(err.detail.version).toBe('1.2.0')
      expect(err.detail.expected).not.toBe(err.detail.found)
      expect(String(err.detail.expected)).toMatch(/^[0-9a-f]{16}$/)
      expect(String(err.detail.found)).toMatch(/^[0-9a-f]{16}$/)
      expect(err.message).toContain('1.2.0')
      expect(err.message).toContain(String(err.detail.expected))
      expect(err.message).toContain(String(err.detail.found))
    }
  })

  it('delete_by_source_removes_the_objects_and_their_links', () => {
    const s = memStore(fixtureOntology())
    putRun(s, 'r_1', { sha: 'abc' }, 'gui/runs/r_1/run.json')
    putRun(s, 'r_2', { sha: 'abc' }, 'gui/runs/r_1/run.json')
    putRun(s, 'r_3', {}, 'other.json')
    putCommit(s, 'abc')
    s.putLink({ type: 'testAtCommit', fromId: 'r_1', toId: 'abc', sourcePath: 'gui/runs/r_1/run.json' })
    s.putLink({ type: 'testAtCommit', fromId: 'r_3', toId: 'abc', sourcePath: 'other.json' })
    expect(s.deleteBySource('gui/runs/r_1/run.json')).toEqual({ objects: 2, links: 1 })
    expect(s.count('TestRun')).toBe(1)
    expect(s.get('TestRun', 'r_3')).not.toBeNull()
    expect(s.links({}).length).toBe(1)
    expect(s.deleteBySource('gui/runs/r_1/run.json')).toEqual({ objects: 0, links: 0 })
    s.close()
  })

  it('the_db_path_comes_from_the_config', () => {
    const t = fs.mkdtempSync(path.join(os.tmpdir(), 'cfd-ontcfg-'))
    try {
      const cfg = loadConfig({ CFD_GUI_DIR: t })
      expect(cfg.ontologyDir).toBe(path.join(t, 'ontology'))
      expect(ontologyDbPath(cfg)).toBe(path.join(t, 'ontology', 'ontology.db'))
      const other = path.join(t, 'elsewhere')
      const overridden = loadConfig({ CFD_GUI_DIR: t, CFD_ONTOLOGY_DIR: other })
      expect(overridden.ontologyDir).toBe(path.resolve(other))
      const s = openOntologyStore({ path: ontologyDbPath(cfg), ontology: fixtureOntology() })
      expect(fs.existsSync(ontologyDbPath(cfg))).toBe(true)
      s.close()
    } finally {
      fs.rmSync(t, { recursive: true, force: true })
    }
  })
})
