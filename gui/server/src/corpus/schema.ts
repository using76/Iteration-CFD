// gui/server/src/corpus/schema.ts — the L1 corpus storey of the ontology mirror:
// one forward-only migration that adds the corpus tables, an external-content
// FTS5 index over chunk text, and the one typed door (CorpusStore) through which
// every corpus unit reads and writes them. docs/13 §5-§6.
//
// The separation from the typed ontology (L2) is mechanical, not a promise:
// no corpus table carries a foreign key into a typed-ontology table or the
// link table, and no member of CorpusStore names an object type table. A row
// enters L2 only by an action a human approved; the corpus holds chunks and
// the staging of what an extractor found, and nothing else.
//
// The migration runs inside openOntologyStore, after the ontology's own open
// step and outside any transaction, so a mirror is never half-corpus. It is
// forward-only and never destructive: it drops nothing and touches no table it
// did not create, which is why the corpus survives a MINOR re-import and a
// reset() — both drop only what the registry generated.
import { DatabaseSync } from 'node:sqlite'
import { silentLogger, type Logger } from '../log.js'

export const CORPUS_SCHEMA_VERSION = 1
export const CORPUS_SCHEMA_VERSION_KEY = 'corpusSchemaVersion'

// Byte-identical to the mirror's own meta DDL, with IF NOT EXISTS, so
// applyCorpusMigrations also works on a bare connection in a test.
const META_DDL = `CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);`

const META_UPSERT =
  'INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value'

// Migration 1: the six corpus tables, their indexes, the external-content FTS5
// index over chunk.text, and the five triggers that keep the index in step and
// hold the paywall. External content stores the text once — there is no
// chunk_fts_content shadow table — and cannot drift, because every write path
// is SQL and the triggers live on the table, not in a caller's discipline.
const MIGRATION_1 = `
CREATE TABLE IF NOT EXISTS document (
  document_id  TEXT    PRIMARY KEY,
  title        TEXT    NOT NULL,
  tier         TEXT    NOT NULL CHECK (tier IN ('A','B')),
  licence      TEXT    NOT NULL CHECK (trim(licence) <> ''),
  is_paywalled INTEGER NOT NULL CHECK (is_paywalled IN (0,1)),
  access_note  TEXT,
  source_path  TEXT    NOT NULL CHECK (trim(source_path) <> ''),
  source_url   TEXT,
  sha256       TEXT,
  imported_at  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS chunk (
  seq         INTEGER PRIMARY KEY,
  chunk_id    TEXT    NOT NULL UNIQUE,
  document_id TEXT    NOT NULL REFERENCES document(document_id) ON DELETE CASCADE,
  locator     TEXT    NOT NULL CHECK (trim(locator) <> ''),
  part        INTEGER NOT NULL DEFAULT 1 CHECK (part >= 1),
  heading     TEXT    NOT NULL DEFAULT '',
  text        TEXT    NOT NULL,
  char_start  INTEGER NOT NULL CHECK (char_start >= 0),
  char_end    INTEGER NOT NULL CHECK (char_end > char_start),
  ordinal     INTEGER NOT NULL,
  imported_at INTEGER NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_chunk_locator ON chunk (document_id, locator, part);
CREATE INDEX IF NOT EXISTS ix_chunk_ordinal ON chunk (document_id, ordinal);

CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5 (
  text,
  content='chunk',
  content_rowid='seq',
  tokenize='unicode61 remove_diacritics 2'
);

CREATE TABLE IF NOT EXISTS object_chunk (
  object_type  TEXT    NOT NULL,
  object_id    TEXT    NOT NULL,
  property     TEXT    NOT NULL DEFAULT '',
  document_id  TEXT    NOT NULL REFERENCES document(document_id) ON DELETE CASCADE,
  chunk_id     TEXT    NOT NULL REFERENCES chunk(chunk_id) ON DELETE CASCADE,
  char_start   INTEGER,
  char_end     INTEGER,
  matched_by   TEXT    NOT NULL CHECK (matched_by IN ('exactPrimaryKey','human')),
  extracted_by TEXT    NOT NULL,
  extracted_at INTEGER NOT NULL,
  PRIMARY KEY (object_type, object_id, property, chunk_id)
);
CREATE INDEX IF NOT EXISTS ix_object_chunk_chunk ON object_chunk (chunk_id);

CREATE TABLE IF NOT EXISTS candidate_object (
  candidate_id TEXT    PRIMARY KEY,
  object_type  TEXT    NOT NULL,
  primary_key  TEXT    NOT NULL CHECK (trim(primary_key) <> ''),
  document_id  TEXT    NOT NULL REFERENCES document(document_id) ON DELETE CASCADE,
  chunk_id     TEXT    NOT NULL REFERENCES chunk(chunk_id) ON DELETE CASCADE,
  char_start   INTEGER,
  char_end     INTEGER,
  props        TEXT    NOT NULL CHECK (json_valid(props)),
  extracted_by TEXT    NOT NULL,
  extracted_at INTEGER NOT NULL,
  CHECK (char_start IS NULL OR char_end IS NULL OR char_end >= char_start)
);
CREATE INDEX IF NOT EXISTS ix_candidate_object_type ON candidate_object (object_type, primary_key);
CREATE INDEX IF NOT EXISTS ix_candidate_object_chunk ON candidate_object (chunk_id);

CREATE TABLE IF NOT EXISTS candidate_link (
  candidate_link_id TEXT NOT NULL PRIMARY KEY,
  link_type         TEXT NOT NULL,
  from_object_type  TEXT NOT NULL,
  from_primary_key  TEXT NOT NULL CHECK (trim(from_primary_key) <> ''),
  to_object_type    TEXT NOT NULL,
  to_primary_key    TEXT NOT NULL CHECK (trim(to_primary_key) <> ''),
  document_id  TEXT    NOT NULL REFERENCES document(document_id) ON DELETE CASCADE,
  chunk_id     TEXT    NOT NULL REFERENCES chunk(chunk_id) ON DELETE CASCADE,
  char_start   INTEGER,
  char_end     INTEGER,
  extracted_by TEXT    NOT NULL,
  extracted_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_candidate_link_chunk ON candidate_link (chunk_id);

CREATE TABLE IF NOT EXISTS unmapped_span (
  span_id      TEXT    PRIMARY KEY,
  document_id  TEXT    NOT NULL REFERENCES document(document_id) ON DELETE CASCADE,
  chunk_id     TEXT    NOT NULL REFERENCES chunk(chunk_id) ON DELETE CASCADE,
  char_start   INTEGER,
  char_end     INTEGER,
  text         TEXT    NOT NULL CHECK (trim(text) <> ''),
  reason       TEXT    NOT NULL,
  object_type  TEXT,
  extracted_by TEXT    NOT NULL,
  extracted_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_unmapped_span_chunk ON unmapped_span (chunk_id);

CREATE TRIGGER IF NOT EXISTS chunk_fts_ai AFTER INSERT ON chunk BEGIN
  INSERT INTO chunk_fts (rowid, text) VALUES (new.seq, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunk_fts_ad AFTER DELETE ON chunk BEGIN
  INSERT INTO chunk_fts (chunk_fts, rowid, text) VALUES ('delete', old.seq, old.text);
END;
CREATE TRIGGER IF NOT EXISTS chunk_fts_au AFTER UPDATE ON chunk BEGIN
  INSERT INTO chunk_fts (chunk_fts, rowid, text) VALUES ('delete', old.seq, old.text);
  INSERT INTO chunk_fts (rowid, text) VALUES (new.seq, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunk_refuse_paywalled BEFORE INSERT ON chunk
WHEN (SELECT is_paywalled FROM document WHERE document_id = new.document_id) = 1 BEGIN
  SELECT RAISE(ABORT, 'CORPUS_PAYWALLED: chunk.text may not be stored for a document whose document.is_paywalled = 1');
END;
CREATE TRIGGER IF NOT EXISTS document_refuse_paywall_flip BEFORE UPDATE OF is_paywalled ON document
WHEN new.is_paywalled = 1 AND EXISTS (SELECT 1 FROM chunk WHERE document_id = new.document_id) BEGIN
  SELECT RAISE(ABORT, 'CORPUS_PAYWALLED: document.is_paywalled may not be set to 1 while chunk rows exist for it');
END;
`

export const CORPUS_MIGRATIONS: ReadonlyArray<{ version: number; sql: string }> = [
  { version: 1, sql: MIGRATION_1 },
]

const CORPUS_TABLES: readonly string[] = [
  'document', 'chunk', 'chunk_fts', 'chunk_fts_config', 'chunk_fts_data',
  'chunk_fts_docsize', 'chunk_fts_idx', 'object_chunk', 'candidate_object',
  'candidate_link', 'unmapped_span',
]

/** The eleven corpus tables in creation order. The test asserts its table list
 *  against this, so the list and the DDL cannot drift apart. */
export function corpusTableNames(): readonly string[] {
  return CORPUS_TABLES
}

export type CorpusSchemaErrorCode = 'CORPUS_AHEAD' | 'CORPUS_NESTED_TX'

export class CorpusSchemaError extends Error {
  readonly code: CorpusSchemaErrorCode
  readonly detail: Record<string, string | number>
  constructor(code: CorpusSchemaErrorCode, message: string, detail?: Record<string, string | number>) {
    super(message)
    this.name = 'CorpusSchemaError'
    this.code = code
    this.detail = detail ?? {}
  }
}

/** Forward-only. Applies every migration above the db's stamp, each in its own
 *  transaction, then re-stamps. Never destructive: it drops nothing and it
 *  touches no table it did not create. */
export function applyCorpusMigrations(
  db: DatabaseSync, log: Logger = silentLogger,
): { from: number; to: number; applied: number[] } {
  if (db.isTransaction)
    throw new CorpusSchemaError('CORPUS_NESTED_TX', 'applyCorpusMigrations must not run inside a transaction: each migration commits on its own')
  db.exec(META_DDL)
  const stamp = db.prepare('SELECT value FROM meta WHERE key = ?').get(CORPUS_SCHEMA_VERSION_KEY)?.value
  const parsed = typeof stamp === 'string' || typeof stamp === 'number' || typeof stamp === 'bigint' ? Number(stamp) : NaN
  const from = Number.isInteger(parsed) && parsed >= 0 ? parsed : 0
  if (from > CORPUS_SCHEMA_VERSION)
    throw new CorpusSchemaError(
      'CORPUS_AHEAD',
      `the corpus mirror is at schema version ${from} but this build only knows version ${CORPUS_SCHEMA_VERSION}; update the server or delete the mirror`,
      { found: from, expected: CORPUS_SCHEMA_VERSION },
    )
  const applied: number[] = []
  for (const m of CORPUS_MIGRATIONS) {
    if (m.version <= from) continue
    db.exec('BEGIN IMMEDIATE')
    try {
      db.exec(m.sql)
      db.prepare(META_UPSERT).run(CORPUS_SCHEMA_VERSION_KEY, String(m.version))
      db.exec('COMMIT')
    } catch (e) {
      try { db.exec('ROLLBACK') } catch { /* the transaction was already undone */ }
      throw e
    }
    applied.push(m.version)
    log.debug(`corpus schema: applied migration ${m.version}`)
  }
  return { from, to: CORPUS_SCHEMA_VERSION, applied }
}

/** Wrap a term as an FTS5 string literal: `5.3.1` -> `"5.3.1"`, `a"b` -> `"a""b"`.
 *  An unquoted dotted locator is a MATCH syntax error, so every caller that
 *  searches for an identifier goes through this. */
export function ftsQuote(term: string): string {
  return `"${term.replace(/"/g, '""')}"`
}

export interface DocumentRow {
  documentId: string
  title: string
  tier: 'A' | 'B'
  licence: string
  isPaywalled: boolean
  accessNote: string | null
  sourcePath: string
  sourceUrl: string | null
  sha256: string | null
  importedAt: number
}

export interface ChunkRow {
  chunkId: string
  documentId: string
  /** The bare clause id, e.g. '55.4'. */
  locator: string
  /** 1 unless a clause had to be split. */
  part: number
  /** '' when the source had none. */
  heading: string
  text: string
  /** Offset inside the DOCUMENT. */
  charStart: number
  charEnd: number
  ordinal: number
  importedAt: number
}

/** A chunk with its BM25 score. More negative is better (that is FTS5's convention). */
export interface ChunkHit extends ChunkRow { rank: number }

/** One property an extractor filled. charStart/charEnd are CHUNK-local and may
 *  be null when the model could not locate its own quote. */
export interface CandidateValue {
  value: string | null
  quote: string | null
  charStart: number | null
  charEnd: number | null
}

export interface CandidateObjectInput {
  /** Opaque, minted by the caller; this is the upsert key. The extractor sends
   *  21 characters ('cand_' plus a 16-hex sha256 prefix). Never parsed here. */
  candidateId: string
  objectType: string
  primaryKey: string
  documentId: string
  chunkId: string
  /** Min over located property quotes; null when none located. */
  charStart: number | null
  /** Max over located property quotes; null when none located. */
  charEnd: number | null
  props: Record<string, CandidateValue>
  /** The model id, e.g. 'glm-5.3-flash'. */
  extractedBy: string
  extractedAt: number
}

export interface CandidateLinkInput {
  /** Opaque, minted by the caller; the upsert key. */
  candidateLinkId: string
  linkType: string
  fromObjectType: string
  fromPrimaryKey: string
  toObjectType: string
  toPrimaryKey: string
  documentId: string
  chunkId: string
  charStart: number | null
  charEnd: number | null
  extractedBy: string
  extractedAt: number
}

export interface UnmappedSpanInput {
  /** Opaque, minted by the caller; the upsert key. */
  spanId: string
  documentId: string
  chunkId: string
  charStart: number | null
  charEnd: number | null
  /** The model's quote, VERBATIM. */
  text: string
  /** The extractor's reason word; not constrained here. */
  reason: string
  /** The pass that found it. */
  objectType: string | null
  extractedBy: string
  extractedAt: number
}

export interface ObjectChunkInput {
  objectType: string
  /** The L2 primary key that matched EXACTLY. */
  objectId: string
  /** null means the whole object; stored as ''. */
  property: string | null
  documentId: string
  chunkId: string
  charStart: number | null
  charEnd: number | null
  matchedBy: 'exactPrimaryKey' | 'human'
  extractedBy: string
  extractedAt: number
}

/**
 * The only door into L1. Eleven members, none of which names an object type
 * table: this interface is what makes "an extractor cannot write the typed
 * ontology" a compile-time fact rather than a promise. Every put* upserts on
 * the row's own id, so a re-ingest or a re-extraction is idempotent.
 */
export interface CorpusStore {
  document(documentId: string): DocumentRow | null
  chunks(spec: { documentId: string; locators?: string[]; limit?: number; offset?: number }): ChunkRow[]
  searchChunks(spec: { match: string; limit?: number }): ChunkHit[]
  putDocument(row: DocumentRow): void
  putChunks(rows: ChunkRow[]): number
  putCandidateObjects(rows: CandidateObjectInput[]): number
  putCandidateLinks(rows: CandidateLinkInput[]): number
  putUnmappedSpans(rows: UnmappedSpanInput[]): number
  putObjectChunks(rows: ObjectChunkInput[]): number
  /** Re-ingest: chunks first (so the FTS delete trigger keeps BM25 in step),
   *  then the document, in one tx. Deletes L1 rows only — the document cascade
   *  reaches no table outside this schema. */
  deleteDocument(documentId: string): { document: number; chunks: number }
  tx<T>(body: () => T): T
}

type SqlRow = Record<string, unknown>

const clamp = (n: number | undefined, dflt: number, lo: number, hi: number): number =>
  n === undefined || !Number.isFinite(n) ? dflt : Math.min(hi, Math.max(lo, Math.trunc(n)))

function mapDocument(r: SqlRow): DocumentRow {
  return {
    documentId: String(r.document_id),
    title: String(r.title),
    tier: r.tier === 'B' ? 'B' : 'A',
    licence: String(r.licence),
    isPaywalled: r.is_paywalled === 1,
    accessNote: r.access_note === null ? null : String(r.access_note),
    sourcePath: String(r.source_path),
    sourceUrl: r.source_url === null ? null : String(r.source_url),
    sha256: r.sha256 === null ? null : String(r.sha256),
    importedAt: Number(r.imported_at),
  }
}

function mapChunk(r: SqlRow): ChunkRow {
  return {
    chunkId: String(r.chunk_id),
    documentId: String(r.document_id),
    locator: String(r.locator),
    part: Number(r.part),
    heading: String(r.heading),
    text: String(r.text),
    charStart: Number(r.char_start),
    charEnd: Number(r.char_end),
    ordinal: Number(r.ordinal),
    importedAt: Number(r.imported_at),
  }
}

/** The tx fallback for a bare connection: a body that runs inside the caller's
 *  own transaction just runs; otherwise BEGIN IMMEDIATE / COMMIT / ROLLBACK. */
function ownTx<T>(db: DatabaseSync, body: () => T): T {
  if (db.isTransaction) return body()
  db.exec('BEGIN IMMEDIATE')
  try {
    const r = body()
    db.exec('COMMIT')
    return r
  } catch (e) {
    try { db.exec('ROLLBACK') } catch { /* the transaction was already undone */ }
    throw e
  }
}

/** `tx` is the mirror store's transaction helper, passed in so the corpus joins
 *  the caller's transaction instead of opening a second one on the same
 *  connection. When it is omitted (a bare-connection test), the fallback is
 *  `db.isTransaction ? body() : BEGIN IMMEDIATE / COMMIT / ROLLBACK`. */
export function makeCorpusStore(db: DatabaseSync, tx?: <T>(body: () => T) => T): CorpusStore {
  const inTx = <T,>(body: () => T): T => (tx !== undefined ? tx(body) : ownTx(db, body))
  const store: CorpusStore = {
    document(documentId) {
      const r = db.prepare('SELECT document_id, title, tier, licence, is_paywalled, access_note, source_path, source_url, sha256, imported_at FROM document WHERE document_id = ?').get(documentId)
      return r === undefined ? null : mapDocument(r)
    },
    chunks(spec) {
      const limit = clamp(spec.limit, 200, 1, 1000)
      const params: (string | number)[] = [spec.documentId]
      let sql = 'SELECT chunk_id, document_id, locator, part, heading, text, char_start, char_end, ordinal, imported_at FROM chunk WHERE document_id = ?'
      if (spec.locators !== undefined) {
        if (spec.locators.length === 0) return []
        sql += ` AND locator IN (${spec.locators.map(() => '?').join(', ')})`
        for (const l of spec.locators) params.push(l)
      }
      sql += ' ORDER BY ordinal, part LIMIT ? OFFSET ?'
      params.push(limit, spec.offset ?? 0)
      return db.prepare(sql).all(...params).map(mapChunk)
    },
    searchChunks(spec) {
      const limit = clamp(spec.limit, 20, 1, 200)
      return db.prepare(
        `SELECT c.chunk_id, c.document_id, c.locator, c.part, c.heading, c.text,
              c.char_start, c.char_end, c.ordinal, c.imported_at, bm25(chunk_fts) AS rank
         FROM chunk_fts JOIN chunk c ON c.seq = chunk_fts.rowid
        WHERE chunk_fts MATCH ? ORDER BY rank LIMIT ?`,
      ).all(spec.match, limit).map((r) => ({ ...mapChunk(r), rank: Number(r.rank) }))
    },
    putDocument(row) {
      db.prepare(
        `INSERT INTO document (document_id, title, tier, licence, is_paywalled, access_note, source_path, source_url, sha256, imported_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT(document_id) DO UPDATE SET
           title = excluded.title, tier = excluded.tier, licence = excluded.licence,
           is_paywalled = excluded.is_paywalled, access_note = excluded.access_note,
           source_path = excluded.source_path, source_url = excluded.source_url,
           sha256 = excluded.sha256, imported_at = excluded.imported_at`,
      ).run(row.documentId, row.title, row.tier, row.licence, row.isPaywalled ? 1 : 0,
        row.accessNote, row.sourcePath, row.sourceUrl, row.sha256, row.importedAt)
    },
    putChunks(rows) {
      return inTx(() => {
        const stmt = db.prepare(
          `INSERT INTO chunk (chunk_id, document_id, locator, part, heading, text, char_start, char_end, ordinal, imported_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(chunk_id) DO UPDATE SET
             document_id = excluded.document_id, locator = excluded.locator, part = excluded.part,
             heading = excluded.heading, text = excluded.text, char_start = excluded.char_start,
             char_end = excluded.char_end, ordinal = excluded.ordinal, imported_at = excluded.imported_at`,
        )
        for (const r of rows)
          stmt.run(r.chunkId, r.documentId, r.locator, r.part, r.heading, r.text, r.charStart, r.charEnd, r.ordinal, r.importedAt)
        return rows.length
      })
    },
    putCandidateObjects(rows) {
      return inTx(() => {
        const stmt = db.prepare(
          `INSERT INTO candidate_object (candidate_id, object_type, primary_key, document_id, chunk_id, char_start, char_end, props, extracted_by, extracted_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(candidate_id) DO UPDATE SET
             object_type = excluded.object_type, primary_key = excluded.primary_key,
             document_id = excluded.document_id, chunk_id = excluded.chunk_id,
             char_start = excluded.char_start, char_end = excluded.char_end, props = excluded.props,
             extracted_by = excluded.extracted_by, extracted_at = excluded.extracted_at`,
        )
        for (const r of rows)
          stmt.run(r.candidateId, r.objectType, r.primaryKey, r.documentId, r.chunkId,
            r.charStart, r.charEnd, JSON.stringify(r.props), r.extractedBy, r.extractedAt)
        return rows.length
      })
    },
    putCandidateLinks(rows) {
      return inTx(() => {
        const stmt = db.prepare(
          `INSERT INTO candidate_link (candidate_link_id, link_type, from_object_type, from_primary_key, to_object_type, to_primary_key, document_id, chunk_id, char_start, char_end, extracted_by, extracted_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(candidate_link_id) DO UPDATE SET
             link_type = excluded.link_type, from_object_type = excluded.from_object_type,
             from_primary_key = excluded.from_primary_key, to_object_type = excluded.to_object_type,
             to_primary_key = excluded.to_primary_key, document_id = excluded.document_id,
             chunk_id = excluded.chunk_id, char_start = excluded.char_start, char_end = excluded.char_end,
             extracted_by = excluded.extracted_by, extracted_at = excluded.extracted_at`,
        )
        for (const r of rows)
          stmt.run(r.candidateLinkId, r.linkType, r.fromObjectType, r.fromPrimaryKey, r.toObjectType, r.toPrimaryKey,
            r.documentId, r.chunkId, r.charStart, r.charEnd, r.extractedBy, r.extractedAt)
        return rows.length
      })
    },
    putUnmappedSpans(rows) {
      return inTx(() => {
        const stmt = db.prepare(
          `INSERT INTO unmapped_span (span_id, document_id, chunk_id, char_start, char_end, text, reason, object_type, extracted_by, extracted_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(span_id) DO UPDATE SET
             document_id = excluded.document_id, chunk_id = excluded.chunk_id,
             char_start = excluded.char_start, char_end = excluded.char_end, text = excluded.text,
             reason = excluded.reason, object_type = excluded.object_type,
             extracted_by = excluded.extracted_by, extracted_at = excluded.extracted_at`,
        )
        for (const r of rows)
          stmt.run(r.spanId, r.documentId, r.chunkId, r.charStart, r.charEnd, r.text, r.reason,
            r.objectType, r.extractedBy, r.extractedAt)
        return rows.length
      })
    },
    putObjectChunks(rows) {
      return inTx(() => {
        const stmt = db.prepare(
          `INSERT INTO object_chunk (object_type, object_id, property, document_id, chunk_id, char_start, char_end, matched_by, extracted_by, extracted_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(object_type, object_id, property, chunk_id) DO UPDATE SET
             document_id = excluded.document_id, char_start = excluded.char_start,
             char_end = excluded.char_end, matched_by = excluded.matched_by,
             extracted_by = excluded.extracted_by, extracted_at = excluded.extracted_at`,
        )
        for (const r of rows)
          stmt.run(r.objectType, r.objectId, r.property ?? '', r.documentId, r.chunkId,
            r.charStart, r.charEnd, r.matchedBy, r.extractedBy, r.extractedAt)
        return rows.length
      })
    },
    deleteDocument(documentId) {
      return inTx(() => {
        // Chunks first, so the FTS delete trigger keeps the index in step, then
        // the document. Explicit, not cascade-dependent: the cascade needs the
        // foreign-keys pragma, which is the mirror's, not ours to assume.
        const chunks = db.prepare('DELETE FROM chunk WHERE document_id = ?').run(documentId).changes
        const doc = db.prepare('DELETE FROM document WHERE document_id = ?').run(documentId).changes
        return { document: Number(doc), chunks: Number(chunks) }
      })
    },
    tx: (body) => inTx(body),
  }
  return store
}
