// gui/server/src/corpus/stage.ts — the staging surface of the corpus, as
// extract.ts is allowed to see it: six members of C1's CorpusStore plus its
// tx, and nothing that could reach an ontology table. This file is the only
// place in the extraction unit that opens the mirror; extract.ts reaches it
// through the CorpusStaging type alone.
import type { OntologyStore } from '../ontology/store.js'
import type { ChunkRow, CorpusStore, DocumentRow } from './schema.js'

export type StagedDocument = DocumentRow
export type StagedChunk = ChunkRow

/** The whole surface extract.ts has on storage. Two reads, four upserts, one transaction. */
export type CorpusStaging = Pick<CorpusStore,
  'document' | 'chunks' | 'putCandidateObjects' | 'putCandidateLinks' | 'putUnmappedSpans' | 'putObjectChunks' | 'tx'>

/** The adapter. Forwards the CorpusStore N2 already opened and C1 already migrated, one
 *  member at a time — never the whole corpus door, which would also hand over
 *  the ingest-side writers and the deleter. */
export function stagingFromStore(store: OntologyStore): CorpusStaging {
  const { corpus: c } = store
  return {
    document: (id) => c.document(id),
    chunks: (spec) => c.chunks(spec),
    putCandidateObjects: (rows) => c.putCandidateObjects(rows),
    putCandidateLinks: (rows) => c.putCandidateLinks(rows),
    putUnmappedSpans: (rows) => c.putUnmappedSpans(rows),
    putObjectChunks: (rows) => c.putObjectChunks(rows),
    tx: (body) => c.tx(body),
  }
}
