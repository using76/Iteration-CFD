// gui/server/src/ontology/store.ts — the mirror: one SQLite file whose schema is
// generated from the ontology registry, and a read surface narrow enough that no
// caller ever writes SQL (docs/12 §B.7, AIP DOMAIN-MODEL.md:261, PLAN.md O3).
//
// The driver is node:sqlite (Node 22's own driver — D1, no new dependency).
// node:sqlite is experimental: Node prints one ExperimentalWarning line on first
// require. It is accepted, not silenced (D2) — hiding it would hide every other
// Node warning in the run.
//
// The mirror has the opposite duty of runs/store.ts: a run's file sink degrades
// to a no-op because a failed log write must not kill a solve; here every row
// must be traceable to a file, so every refusal throws BY NAME (D6). A silently
// wrong mirror is worse than no mirror.
import { createHash } from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'
import { DatabaseSync, type StatementSync } from 'node:sqlite'
import type { BaseType, LinkTypeDef, ObjectTypeDef, OntologyRegistry, PropertyDef } from '@cfd/shared'
import { silentLogger, type Logger } from '../log.js'
import { applyCorpusMigrations, makeCorpusStore, type CorpusStore } from '../corpus/schema.js'

export type OntologyStoreErrorCode =
  | 'UNKNOWN_TYPE' | 'UNKNOWN_PROPERTY' | 'MISSING_PROPERTY' | 'PK_MISMATCH'
  | 'BAD_TIMESTAMP' | 'BAD_VALUE' | 'NO_SOURCE' | 'BAD_LIMIT'
  | 'UNKNOWN_LINK' | 'AMBIGUOUS_LINK_SIDE' | 'LINK_PROPERTY_REFUSED'
  | 'MIRROR_MAJOR_BEHIND' | 'MIRROR_AHEAD' | 'MIRROR_SCHEMA_DRIFT'
  | 'ASYNC_TX'

export class OntologyStoreError extends Error {
  readonly code: OntologyStoreErrorCode
  readonly detail: Record<string, string | number>
  constructor(code: OntologyStoreErrorCode, message: string, detail: Record<string, string | number> = {}) {
    super(message)
    this.name = 'OntologyStoreError'
    this.code = code
    this.detail = detail
  }
}

/** What open() did to the file it found. */
export type OpenOutcome = 'created' | 'opened' | 'reimport'

export interface ObjectRef { type: string; id: string }

export interface ObjectInput {
  type: string
  id: string
  props: Record<string, unknown>
  /** Workspace-relative file the fact came from, or `action:<apiName>`. Never empty. */
  sourcePath: string
  /** Epoch ms; defaults to the store's clock. */
  importedAt?: number
}

export interface ObjectRow {
  type: string
  id: string
  title: string | null
  props: Record<string, unknown>
  sourcePath: string
  importedAt: number
}

export interface LinkInput {
  /** LinkTypeDef.apiName; the two endpoint types come from the definition, never from the caller. */
  type: string
  fromId: string
  toId: string
  props?: Record<string, unknown>
  sourcePath: string
  importedAt?: number
}

export interface LinkRow {
  type: string
  fromType: string
  fromId: string
  toType: string
  toId: string
  props: Record<string, unknown>
  sourcePath: string
  importedAt: number
}

export interface Filter { property: string; equals: string | number | boolean | null }
export interface Page { limit?: number; offset?: number }
export interface QuerySpec extends Page {
  type: string
  /** ANDed. */
  where?: Filter[]
  orderBy?: { property: string; direction: 'asc' | 'desc' }
}
export interface LinkQuery extends Page {
  linkType?: string
  from?: ObjectRef
  to?: ObjectRef
}

export interface OntologyStoreOptions {
  /** ':memory:' or an absolute file path; parent directories are created. */
  path: string
  /** N1's built registry instance. */
  ontology: OntologyRegistry
  log?: Logger
  /** Injectable clock so a test can pin imported_at. Defaults to Date.now. */
  now?: () => number
}

export const ONTOLOGY_DB_FILE = 'ontology.db'

/** The one place the mirror's file name is spelled (D10). */
export function ontologyDbPath(cfg: { ontologyDir: string }): string {
  return path.join(cfg.ontologyDir, ONTOLOGY_DB_FILE)
}

// ---- generated DDL (C2) --------------------------------------------------

const META_DDL = `CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
)`

const LINKS_DDL = `CREATE TABLE IF NOT EXISTS links (
  type        TEXT    NOT NULL,
  from_type   TEXT    NOT NULL,
  from_id     TEXT    NOT NULL,
  to_type     TEXT    NOT NULL,
  to_id       TEXT    NOT NULL,
  props       TEXT    NOT NULL,
  source_path TEXT    NOT NULL,
  imported_at INTEGER NOT NULL,
  PRIMARY KEY (type, from_type, from_id, to_type, to_id)
) WITHOUT ROWID`

const LINK_IX_FROM_DDL = 'CREATE INDEX IF NOT EXISTS ix_links_from ON links (from_type, from_id, type)'
const LINK_IX_TO_DDL = 'CREATE INDEX IF NOT EXISTS ix_links_to ON links (to_type, to_id, type)'

const objTableDdl = (apiName: string): string => `CREATE TABLE IF NOT EXISTS ${q(`obj_${apiName}`)} (
  id          TEXT    PRIMARY KEY,
  title       TEXT,
  props       TEXT    NOT NULL,
  source_path TEXT    NOT NULL,
  imported_at INTEGER NOT NULL
)`

const objSourceIndexDdl = (apiName: string): string =>
  `CREATE INDEX IF NOT EXISTS ${q(`ix_${apiName}_source`)} ON ${q(`obj_${apiName}`)} (source_path)`

const fkIndexDdl = (apiName: string, property: string): string =>
  `CREATE INDEX IF NOT EXISTS ${q(`ix_${apiName}_${property}`)} ON ${q(`obj_${apiName}`)} (json_extract(props, '${jsonPath(property)}'))`

function q(name: string): string {
  return `"${name.replaceAll('"', '""')}"`
}
function jsonPath(property: string): string {
  return `$.${property}`
}

// ---- value checking (C7 step 5) ------------------------------------------

/** ISO-8601 on the wire, epoch ms in SQLite (D7). A finite number passes through
 *  (already ms); a string is parsed; anything else is refused by name. */
function toEpochMs(typeName: string, prop: string, v: unknown): number {
  if (typeof v === 'number' && Number.isFinite(v)) return v
  if (typeof v === 'string') {
    const t = Date.parse(v)
    if (Number.isFinite(t)) return t
  }
  throw new OntologyStoreError('BAD_TIMESTAMP', `put ${typeName}.${prop}: expected a timestamp (epoch-ms number or ISO-8601 string), got ${typeof v === 'string' ? JSON.stringify(v) : typeof v}`, { type: typeName, property: prop })
}

type StructField = { baseType: BaseType; main?: true }

function checkBase(typeName: string, label: string, base: BaseType, v: unknown, shape: { items?: BaseType; fields?: Record<string, StructField> }): unknown {
  const bad = (why: string): OntologyStoreError =>
    new OntologyStoreError('BAD_VALUE', `put ${typeName}.${label}: expected ${base}, got ${Array.isArray(v) ? 'array' : v === null ? 'null' : typeof v} (${why})`, { type: typeName, property: label })
  switch (base) {
    case 'string':
    case 'attachmentRef':
      if (typeof v !== 'string') throw bad('must be a string')
      return v
    case 'integer':
    case 'long':
      if (typeof v !== 'number' || !Number.isFinite(v) || !Number.isInteger(v)) throw bad('must be a finite integer')
      return v
    case 'double':
      if (typeof v !== 'number' || !Number.isFinite(v)) throw bad('must be a finite number')
      return v
    case 'boolean':
      if (typeof v !== 'boolean') throw bad('must be a boolean')
      return v
    case 'timestamp':
    case 'date':
      return toEpochMs(typeName, label, v)
    case 'json':
      try { JSON.stringify(v) } catch { throw bad('value is not JSON-serialisable') }
      return v
    case 'struct': {
      if (v === null || typeof v !== 'object' || Array.isArray(v)) throw bad('must be a non-null, non-array object')
      const fields = shape.fields ?? {}
      const out: Record<string, unknown> = {}
      for (const [k, val] of Object.entries(v as Record<string, unknown>)) {
        const f = fields[k]
        if (f === undefined) throw new OntologyStoreError('UNKNOWN_PROPERTY', `put ${typeName}.${label}: field ${k} is not declared in the struct`, { type: typeName, property: `${label}.${k}` })
        out[k] = checkBase(typeName, `${label}.${k}`, f.baseType, val, {})
      }
      return out
    }
    case 'array': {
      if (!Array.isArray(v)) throw bad('must be an array')
      const items = shape.items ?? 'json'
      return v.map((el, i) => checkBase(typeName, `${label}[${i}]`, items, el, {}))
    }
  }
}

/** One property of one declared type: base-type check first, then the enum
 *  narrowing (C7 step 5, last bullet). */
function checkProp(typeName: string, pd: PropertyDef, v: unknown): unknown {
  const out = checkBase(typeName, pd.apiName, pd.baseType, v, pd)
  if (pd.valueType === 'enum') {
    const allowed = pd.enumValues ?? []
    if (!allowed.includes(out as string))
      throw new OntologyStoreError('BAD_VALUE', `put ${typeName}.${pd.apiName}: value ${JSON.stringify(out)} is not one of ${allowed.map((s) => `'${s}'`).join(', ')}`, { type: typeName, property: pd.apiName, value: String(out) })
  }
  return out
}

function pageArgs(p: Page): { limit: number; offset: number } {
  const limit = p.limit ?? DEFAULT_LIMIT
  if (!Number.isInteger(limit) || limit < 0 || limit > MAX_LIMIT)
    throw new OntologyStoreError('BAD_LIMIT', `limit must be an integer between 0 and ${MAX_LIMIT}, got ${limit}`, { limit })
  const offset = p.offset ?? 0
  if (!Number.isInteger(offset) || offset < 0)
    throw new OntologyStoreError('BAD_LIMIT', `offset must be a non-negative integer, got ${offset}`, { offset })
  return { limit, offset }
}

const DEFAULT_LIMIT = 100
const MAX_LIMIT = 1000
type SQLValue = string | number | null

interface ObjectRowRaw { id: SQLOut; title: SQLOut; props: SQLOut; source_path: SQLOut; imported_at: SQLOut }
interface LinkRowRaw { type: SQLOut; from_type: SQLOut; from_id: SQLOut; to_type: SQLOut; to_id: SQLOut; props: SQLOut; source_path: SQLOut; imported_at: SQLOut }
type SQLOut = null | number | bigint | string

function objectRowOf(typeName: string, r: ObjectRowRaw): ObjectRow {
  return {
    type: typeName,
    id: String(r.id),
    title: r.title === null ? null : String(r.title),
    props: JSON.parse(String(r.props)) as Record<string, unknown>,
    sourcePath: String(r.source_path),
    importedAt: Number(r.imported_at),
  }
}

function linkRowOf(r: LinkRowRaw): LinkRow {
  return {
    type: String(r.type),
    fromType: String(r.from_type),
    fromId: String(r.from_id),
    toType: String(r.to_type),
    toId: String(r.to_id),
    props: JSON.parse(String(r.props)) as Record<string, unknown>,
    sourcePath: String(r.source_path),
    importedAt: Number(r.imported_at),
  }
}

const OBJECT_SELECT = 'id, title, props, source_path, imported_at'

const OBJ_UPSERT = (apiName: string): string =>
  `INSERT INTO ${q(`obj_${apiName}`)} (id,title,props,source_path,imported_at) VALUES (?,?,?,?,?)
   ON CONFLICT(id) DO UPDATE SET title=excluded.title, props=excluded.props,
   source_path=excluded.source_path, imported_at=excluded.imported_at`

const LINK_UPSERT = `INSERT INTO links (type,from_type,from_id,to_type,to_id,props,source_path,imported_at)
   VALUES (?,?,?,?,?,?,?,?)
   ON CONFLICT(type,from_type,from_id,to_type,to_id) DO UPDATE SET props=excluded.props,
   source_path=excluded.source_path, imported_at=excluded.imported_at`

const META_UPSERT = 'INSERT INTO meta (key,value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value = excluded.value'

const RAW = new WeakMap<object, DatabaseSync>()

/** Test hook: read-only SQL against the store's own connection, so store.test.ts
 *  can read sqlite_master / PRAGMA table_info of a ':memory:' store. Not part of
 *  the C6 surface; nothing outside the tests should call it. */
export function debugSql(store: OntologyStore, sql: string): Record<string, SQLOut>[] {
  const db = RAW.get(store)
  if (db === undefined) throw new Error('debugSql: not an open OntologyStore')
  return db.prepare(sql).all() as Record<string, SQLOut>[]
}

// ---- the store -------------------------------------------------------------

export function resolveLinkSide(
  ontology: OntologyRegistry, fromType: string, side: string
): { def: LinkTypeDef; direction: 'forward' | 'reverse' } {
  const hits: { def: LinkTypeDef; direction: 'forward' | 'reverse' }[] = []
  for (const def of ontology.linkTypes) {
    if (def.from.objectType === fromType && def.from.apiName === side) hits.push({ def, direction: 'forward' })
    else if (def.to.objectType === fromType && def.to.apiName === side) hits.push({ def, direction: 'reverse' })
  }
  if (hits.length === 1) return hits[0]
  if (hits.length === 0)
    throw new OntologyStoreError('UNKNOWN_LINK', `no link side named ${side} hangs on object type ${fromType}`, { type: fromType, side })
  throw new OntologyStoreError('AMBIGUOUS_LINK_SIDE', `link side ${side} on ${fromType} matches more than one link: ${hits.map((h) => h.def.apiName).join(', ')}`, { type: fromType, side })
}

export interface OntologyStore {
  readonly path: string
  readonly ontologyVersion: string
  readonly opened: OpenOutcome

  /** The L1 corpus on the same connection. It can write `document`, `chunk`,
   *  `object_chunk` and the `candidate_*` / `unmapped_span` staging tables, and
   *  nothing else. */
  readonly corpus: CorpusStore

  put(row: ObjectInput): void
  putMany(rows: ObjectInput[]): number
  get(type: string, id: string): ObjectRow | null
  list(type: string, page?: Page): ObjectRow[]
  query(spec: QuerySpec): ObjectRow[]
  count(type: string, where?: Filter[]): number

  putLink(link: LinkInput): void
  putLinks(links: LinkInput[]): number
  links(spec: LinkQuery): LinkRow[]
  traverse(from: ObjectRef, side: string, page?: Page): ObjectRow[]

  deleteObject(type: string, id: string): boolean
  deleteBySource(sourcePath: string): { objects: number; links: number }
  reset(): void

  tx<T>(body: () => T): T

  meta(key: string): string | null
  setMeta(key: string, value: string): void
  close(): void
}

class OntologyStoreImpl implements OntologyStore {
  readonly path: string
  readonly ontologyVersion: string
  readonly corpus: CorpusStore
  private readonly db: DatabaseSync
  private readonly ont: OntologyRegistry
  private readonly log: Logger
  private readonly nowFn: () => number
  private readonly stmts = new Map<string, StatementSync>()
  private readonly hash: string
  private _opened: OpenOutcome = 'created'
  private depth = 0
  private closed = false

  constructor(opts: OntologyStoreOptions) {
    this.path = opts.path
    this.ont = opts.ontology
    this.ontologyVersion = opts.ontology.version
    this.log = opts.log ?? silentLogger
    this.nowFn = opts.now ?? Date.now
    this.db = new DatabaseSync(opts.path, { timeout: 5000 })
    RAW.set(this, this.db)
    this.hash = createHash('sha256').update(this.ddlStatements().join('\n')).digest('hex').slice(0, 16)
    try {
      this.migrate()
      applyCorpusMigrations(this.db, this.log)
      this.corpus = makeCorpusStore(this.db, (body) => this.tx(body))
    } catch (e) {
      // A refused open (MIRROR_* failures) must not leave the file locked.
      RAW.delete(this)
      this.db.close()
      throw e
    }
  }

  get opened(): OpenOutcome {
    return this._opened
  }

  private stmt(sql: string): StatementSync {
    let s = this.stmts.get(sql)
    if (s === undefined) {
      s = this.db.prepare(sql)
      this.stmts.set(sql, s)
    }
    return s
  }

  /** All generated DDL, in execution order; the schema hash is taken over this
   *  exact text joined by \n, so a registry change at an unchanged version is
   *  caught even when the version stamp cannot see it (C9). */
  private ddlStatements(): string[] {
    const out = [META_DDL, LINKS_DDL, LINK_IX_FROM_DDL, LINK_IX_TO_DDL]
    const fk = new Map<string, string[]>()
    for (const l of this.ont.linkTypes) {
      if (l.backing.kind !== 'foreignKey') continue
      const list = fk.get(l.backing.onType) ?? []
      if (!list.includes(l.backing.property)) list.push(l.backing.property)
      fk.set(l.backing.onType, list)
    }
    for (const t of this.ont.objectTypes) {
      out.push(objTableDdl(t.apiName), objSourceIndexDdl(t.apiName))
      for (const p of fk.get(t.apiName) ?? []) out.push(fkIndexDdl(t.apiName, p))
    }
    return out
  }

  private dropGenerated(): void {
    // Only what the registry generated, plus links — N4's append-only edit_log
    // and anything else in the file survives a re-import (C9).
    this.db.exec(`DROP TABLE IF EXISTS ${q('links')}`)
    for (const t of this.ont.objectTypes) this.db.exec(`DROP TABLE IF EXISTS ${q(`obj_${t.apiName}`)}`)
  }

  private migrate(): void {
    // Pragmas, once, in this order, before any DDL (C2). On ':memory:' WAL
    // reports 'memory' — that is not an error.
    this.db.exec('PRAGMA journal_mode = WAL;')
    this.db.exec('PRAGMA synchronous = NORMAL;')
    this.db.exec('PRAGMA foreign_keys = ON;')
    const statements = this.ddlStatements()
    const dbVersion = this.tableExists('meta') ? this.meta('ontologyVersion') : null
    if (dbVersion === null) {
      this.db.exec(statements.join(';\n'))
      this.setMeta('ontologyVersion', this.ont.version)
      this.setMeta('schemaHash', this.hash)
      this.setMeta('createdAt', String(this.nowFn()))
      this._opened = 'created'
      return
    }
    const cmp = compareVersions(dbVersion, this.ont.version)
    if (cmp > 0)
      throw new OntologyStoreError('MIRROR_AHEAD', `the mirror is at ontology version ${dbVersion}, ahead of the registry at ${this.ont.version}; the registry is the source of truth - delete the mirror or move the registry forward`, { dbVersion, registryVersion: this.ont.version })
    if (cmp < 0) {
      if (majorOf(dbVersion) < majorOf(this.ont.version))
        throw new OntologyStoreError('MIRROR_MAJOR_BEHIND', `the mirror is at ontology version ${dbVersion}, a major version behind the registry at ${this.ont.version}; write a forward-only migration gui/ontology/MIGRATIONS/${this.ont.version}-<slug>.sql first, then re-import`, { dbVersion, registryVersion: this.ont.version, migration: `gui/ontology/MIGRATIONS/${this.ont.version}-*.sql` })
      // MINOR/PATCH behind: the import is a pure, idempotent fold over files
      // that already exist, so the wipe is the migration (C9).
      this.dropGenerated()
      this.db.exec(statements.join(';\n'))
      this.setMeta('ontologyVersion', this.ont.version)
      this.setMeta('schemaHash', this.hash)
      this._opened = 'reimport'
      return
    }
    const found = this.meta('schemaHash')
    if (found !== this.hash)
      throw new OntologyStoreError('MIRROR_SCHEMA_DRIFT', `the ontology is still at version ${this.ont.version} but the generated schema changed (registry ${this.hash}, mirror ${found ?? 'null'}); bump \`ontology.version\` or delete the mirror`, { version: this.ont.version, expected: this.hash, found: found ?? '' })
    this._opened = 'opened'
  }

  private tableExists(name: string): boolean {
    return this.stmt("SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?").get(name) !== undefined
  }

  // ---- the write path (C7) -------------------------------------------------

  private requireType(type: string): ObjectTypeDef {
    const def = this.ont.objectType(type)
    if (def === null) throw new OntologyStoreError('UNKNOWN_TYPE', `no object type named ${type} in the ontology`, { type })
    return def
  }

  private requireSource(typeName: string, sourcePath: unknown): string {
    if (typeof sourcePath !== 'string' || sourcePath.trim() === '')
      throw new OntologyStoreError('NO_SOURCE', `put ${typeName}: sourcePath must be a non-empty workspace-relative file (or action:<apiName>) - a row nobody can trace is a row the mirror would have invented`, { type: typeName })
    return sourcePath
  }

  put(row: ObjectInput): void {
    this.putMany([row])
  }

  putMany(rows: ObjectInput[]): number {
    return this.tx(() => {
      for (const r of rows) this.putOne(r)
      return rows.length
    })
  }

  private putOne(row: ObjectInput): void {
    const def = this.requireType(row.type)
    this.requireSource(def.apiName, row.sourcePath)
    const declared = new Map(def.properties.map((p) => [p.apiName, p]))
    for (const key of Object.keys(row.props)) {
      const pd = declared.get(key)
      // Derived properties are computed on read and never stored (C7 step 4).
      if (pd === undefined || pd.derived !== undefined)
        throw new OntologyStoreError('UNKNOWN_PROPERTY', `put ${def.apiName}: property ${key} is not a stored, declared property of this object type`, { type: def.apiName, property: key })
    }
    const stored: Record<string, unknown> = {}
    for (const pd of def.properties) {
      if (pd.derived !== undefined) continue
      const v = row.props[pd.apiName]
      // Absent, undefined and - on a nullable property - explicit null all
      // store as null; every property is present, possibly null (types.ts).
      if (v === undefined || v === null) {
        if (!pd.nullable)
          throw new OntologyStoreError('MISSING_PROPERTY', `put ${def.apiName}: property ${pd.apiName} is not nullable and was ${v === undefined ? 'not given' : 'given as null'}`, { type: def.apiName, property: pd.apiName })
        stored[pd.apiName] = null
      } else {
        stored[pd.apiName] = checkProp(def.apiName, pd, v)
      }
    }
    const id = String(stored[def.primaryKey])
    if (id !== row.id)
      throw new OntologyStoreError('PK_MISMATCH', `put ${def.apiName}: id ${row.id} does not match primary key ${def.primaryKey} = ${id}`, { type: def.apiName, id: row.id, primaryKey: id })
    const title = stored[def.titleKey] == null ? null : String(stored[def.titleKey])
    this.stmt(OBJ_UPSERT(def.apiName)).run(row.id, title, JSON.stringify(stored), row.sourcePath, row.importedAt ?? this.nowFn())
  }

  putLink(link: LinkInput): void {
    this.putLinks([link])
  }

  putLinks(links: LinkInput[]): number {
    return this.tx(() => {
      for (const l of links) this.putLinkOne(l)
      return links.length
    })
  }

  private putLinkOne(link: LinkInput): void {
    const def = this.ont.linkType(link.type)
    if (def === null) throw new OntologyStoreError('UNKNOWN_LINK', `putLink: no link type named ${link.type}`, { link: link.type })
    this.requireSource(def.apiName, link.sourcePath)
    const declared = new Map((def.properties ?? []).map((p) => [p.apiName, p]))
    const given = link.props ?? {}
    for (const key of Object.keys(given))
      if (!declared.has(key))
        throw new OntologyStoreError('LINK_PROPERTY_REFUSED', `putLink ${def.apiName}: property ${key} is not declared on this link type`, { link: def.apiName, property: key })
    const props: Record<string, unknown> = {}
    for (const [k, v] of Object.entries(given)) {
      const pd = declared.get(k) as PropertyDef
      props[k] = v === null && pd.nullable ? null : checkProp(def.apiName, pd, v)
    }
    if (def.ordered === true && !(typeof props['index'] === 'number' && Number.isInteger(props['index'])))
      throw new OntologyStoreError('MISSING_PROPERTY', `putLink ${def.apiName}: an ordered link needs an integer index on every row`, { link: def.apiName, property: 'index' })
    // The endpoints come from the definition, never from the caller (C7), so an
    // edge can never be mislabelled. The target need not exist: N3 imports out
    // of order and traverse is where dangling is reported.
    this.stmt(LINK_UPSERT).run(def.apiName, def.from.objectType, link.fromId, def.to.objectType, link.toId, JSON.stringify(props), link.sourcePath, link.importedAt ?? this.nowFn())
  }

  // ---- the read surface (C8): type + ANDed equality + one orderBy + paging --

  private requireStoredProperty(def: ObjectTypeDef, property: string): PropertyDef {
    const pd = def.properties.find((p) => p.apiName === property)
    // Derived properties are never stored, so a filter on one would silently
    // match nothing - refused like an undeclared one.
    if (pd === undefined || pd.derived !== undefined)
      throw new OntologyStoreError('UNKNOWN_PROPERTY', `query ${def.apiName}: property ${property} is not a stored, declared property of this object type`, { type: def.apiName, property })
    return pd
  }

  /** SQLite compares json_extract results in its own type order:
   *  NULL < numbers < text < blob. The caller orders on one declared property. */
  query(spec: QuerySpec): ObjectRow[] {
    const def = this.requireType(spec.type)
    const where: string[] = []
    const args: SQLValue[] = []
    for (const f of spec.where ?? []) {
      this.requireStoredProperty(def, f.property)
      const path = jsonPath(f.property)
      if (f.equals === null) {
        where.push(`json_extract(props, '${path}') IS NULL`)
      } else {
        where.push(`json_extract(props, '${path}') = ?`)
        // node:sqlite cannot bind a boolean and JSON true extracts as 1.
        args.push(typeof f.equals === 'boolean' ? (f.equals ? 1 : 0) : f.equals)
      }
    }
    let order = ''
    if (spec.orderBy !== undefined) {
      this.requireStoredProperty(def, spec.orderBy.property)
      order = ` ORDER BY json_extract(props, '${jsonPath(spec.orderBy.property)}') ${spec.orderBy.direction === 'desc' ? 'DESC' : 'ASC'}`
    }
    const { limit, offset } = pageArgs(spec)
    const sql = `SELECT ${OBJECT_SELECT} FROM ${q(`obj_${def.apiName}`)}${where.length > 0 ? ` WHERE ${where.join(' AND ')}` : ''}${order} LIMIT ? OFFSET ?`
    return this.stmt(sql).all(...args, limit, offset).map((r) => objectRowOf(def.apiName, r as unknown as ObjectRowRaw))
  }

  list(type: string, page?: Page): ObjectRow[] {
    return this.query({ type, ...page })
  }

  count(type: string, where?: Filter[]): number {
    const def = this.requireType(type)
    const clauses: string[] = []
    const args: SQLValue[] = []
    for (const f of where ?? []) {
      this.requireStoredProperty(def, f.property)
      const path = jsonPath(f.property)
      if (f.equals === null) clauses.push(`json_extract(props, '${path}') IS NULL`)
      else {
        clauses.push(`json_extract(props, '${path}') = ?`)
        args.push(typeof f.equals === 'boolean' ? (f.equals ? 1 : 0) : f.equals)
      }
    }
    const sql = `SELECT count(*) AS n FROM ${q(`obj_${def.apiName}`)}${clauses.length > 0 ? ` WHERE ${clauses.join(' AND ')}` : ''}`
    return Number((this.stmt(sql).get(...args) as { n: SQLOut }).n)
  }

  get(type: string, id: string): ObjectRow | null {
    const def = this.requireType(type)
    const r = this.stmt(`SELECT ${OBJECT_SELECT} FROM ${q(`obj_${def.apiName}`)} WHERE id = ?`).get(id) as ObjectRowRaw | undefined
    return r === undefined ? null : objectRowOf(def.apiName, r)
  }

  // ---- links and traversal ---------------------------------------------------

  links(spec: LinkQuery): LinkRow[] {
    const where: string[] = []
    const args: SQLValue[] = []
    if (spec.linkType !== undefined) { where.push('type = ?'); args.push(spec.linkType) }
    if (spec.from !== undefined) { where.push('from_type = ?'); args.push(spec.from.type); where.push('from_id = ?'); args.push(spec.from.id) }
    if (spec.to !== undefined) { where.push('to_type = ?'); args.push(spec.to.type); where.push('to_id = ?'); args.push(spec.to.id) }
    const { limit, offset } = pageArgs(spec)
    const sql = `SELECT type, from_type, from_id, to_type, to_id, props, source_path, imported_at FROM links${where.length > 0 ? ` WHERE ${where.join(' AND ')}` : ''} ORDER BY from_type, from_id, to_type, to_id LIMIT ? OFFSET ?`
    return this.stmt(sql).all(...args, limit, offset).map((r) => linkRowOf(r as unknown as LinkRowRaw))
  }

  traverse(from: ObjectRef, side: string, page?: Page): ObjectRow[] {
    const { def, direction } = resolveLinkSide(this.ont, from.type, side)
    const { limit, offset } = pageArgs(page ?? {})
    const forward = direction === 'forward'
    const farType = forward ? def.to.objectType : def.from.objectType
    const sql = forward
      ? `SELECT to_id FROM links WHERE type = ? AND from_type = ? AND from_id = ? ORDER BY to_id LIMIT ? OFFSET ?`
      : `SELECT from_id FROM links WHERE type = ? AND to_type = ? AND to_id = ? ORDER BY from_id LIMIT ? OFFSET ?`
    const ids = (this.stmt(sql).all(def.apiName, from.type, from.id, limit, offset) as { to_id?: SQLOut; from_id?: SQLOut }[])
      .map((r) => String(forward ? r.to_id : r.from_id))
    const out: ObjectRow[] = []
    const missing: string[] = []
    for (const id of ids) {
      const obj = this.get(farType, id)
      if (obj === null) missing.push(id)
      else out.push(obj)
    }
    // A dangling edge is a fact about the data, not a bug in the store - but it
    // is said out loud, once per call.
    if (missing.length > 0)
      this.log.warn(`traverse ${def.apiName}: skipped ${missing.length} dangling ${farType} target(s): ${missing.join(', ')}`)
    return out
  }

  // ---- deletion ---------------------------------------------------------------

  deleteObject(type: string, id: string): boolean {
    this.requireType(type)
    return this.tx(() => Number(this.stmt(`DELETE FROM ${q(`obj_${type}`)} WHERE id = ?`).run(id).changes) > 0)
  }

  deleteBySource(sourcePath: string): { objects: number; links: number } {
    return this.tx(() => {
      let objects = 0
      for (const t of this.ont.objectTypes)
        objects += Number(this.stmt(`DELETE FROM ${q(`obj_${t.apiName}`)} WHERE source_path = ?`).run(sourcePath).changes)
      const links = Number(this.stmt('DELETE FROM links WHERE source_path = ?').run(sourcePath).changes)
      return { objects, links }
    })
  }

  reset(): void {
    this.tx(() => {
      this.dropGenerated()
      this.db.exec(this.ddlStatements().join(';\n'))
      this.setMeta('schemaHash', this.hash)
    })
  }

  // ---- transactions (C10) ------------------------------------------------------

  /** Strictly synchronous. A nested tx joins the outer one (no savepoints: an
   *  inner failure must roll the whole outer back, so the edit-log row N4
   *  writes can never commit apart from the edits). A body that returns a
   *  thenable is refused with ASYNC_TX instead of committing before its writes
   *  land. Every await happens OUTSIDE tx; the synchronous store calls go in. */
  tx<T>(body: () => T): T {
    this.depth++
    try {
      if (this.depth === 1) this.db.exec('BEGIN IMMEDIATE')
      let r: T
      try {
        r = body()
      } catch (e) {
        if (this.depth === 1) this.rollbackQuietly()
        throw e
      }
      if (this.depth === 1) {
        if (r !== null && (typeof r === 'object' || typeof r === 'function') && typeof (r as { then?: unknown }).then === 'function') {
          this.rollbackQuietly()
          throw new OntologyStoreError('ASYNC_TX', 'tx(body) is synchronous: body returned a promise, so its writes would land after COMMIT; await outside tx and pass a synchronous body')
        }
        this.db.exec('COMMIT')
      }
      return r
    } finally {
      this.depth--
    }
  }

  private rollbackQuietly(): void {
    try { this.db.exec('ROLLBACK') } catch { /* the transaction was already undone */ }
  }

  // ---- meta and lifecycle -------------------------------------------------------

  meta(key: string): string | null {
    const r = this.stmt('SELECT value FROM meta WHERE key = ?').get(key) as { value: SQLOut } | undefined
    return r === undefined ? null : String(r.value)
  }

  setMeta(key: string, value: string): void {
    this.stmt(META_UPSERT).run(key, value)
  }

  close(): void {
    if (this.closed) return
    this.closed = true
    RAW.delete(this)
    this.stmts.clear()
    this.db.close()
  }
}

/** Componentwise semver compare: negative when a < b, positive when a > b. */
function compareVersions(a: string, b: string): number {
  const pa = a.split('.').map(Number)
  const pb = b.split('.').map(Number)
  for (let i = 0; i < 3; i++) {
    const d = (pa[i] ?? 0) - (pb[i] ?? 0)
    if (d !== 0) return d
  }
  return 0
}

const majorOf = (v: string): number => Number(v.split('.')[0] ?? 0)

export function openOntologyStore(opts: OntologyStoreOptions): OntologyStore {
  if (opts.path !== ':memory:') fs.mkdirSync(path.dirname(opts.path), { recursive: true })
  return new OntologyStoreImpl(opts)
}
