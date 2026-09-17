// gui/server/src/ontology/storeAdapter.ts — the ONE seam between the engine's narrow ActionStore
// port and N2's OntologyStore (N4 D-F). The edit_log table lives here, not in the registry: rows
// are append-only and never migrated destructively (fact sheet §7.5), so the store itself never
// touches the table — it only lends its handle (D-I), and there is never a second DatabaseSync.
import type { DatabaseSync } from 'node:sqlite'
import type { EditLogEntry, LinkEdit } from '@cfd/shared'
import type { OntologyStore } from './store.js'
import type { ActionStore } from './engine.js'

const EDIT_LOG_DDL = `CREATE TABLE IF NOT EXISTS edit_log (
  edit_id     TEXT PRIMARY KEY,
  proposal_id TEXT NOT NULL,
  action      TEXT NOT NULL,
  action_version   TEXT NOT NULL,
  ontology_version TEXT NOT NULL,
  applied_at  INTEGER NOT NULL,
  entry       TEXT NOT NULL              -- the whole EditLogEntry as JSON
)`
const EDIT_LOG_IX_DDL = 'CREATE INDEX IF NOT EXISTS edit_log_action ON edit_log(action, applied_at)'
const EDIT_LOG_INSERT =
  'INSERT INTO edit_log (edit_id, proposal_id, action, action_version, ontology_version, applied_at, entry) VALUES (?, ?, ?, ?, ?, ?, ?)'
const EDIT_LOG_SELECT = 'SELECT entry FROM edit_log ORDER BY applied_at, edit_id'

/** Runs the edit_log DDL once, at construction, through the lent handle; prepares its two
 *  statements; and maps the port onto N2's surface, nowhere else. */
export function createStoreActionStore(store: OntologyStore): ActionStore {
  const db: DatabaseSync = store.raw()
  db.exec(EDIT_LOG_DDL)
  db.exec(EDIT_LOG_IX_DDL)
  const insert = db.prepare(EDIT_LOG_INSERT)
  const selectAll = db.prepare(EDIT_LOG_SELECT)
  return {
    getObject: (objectType, id) => store.get(objectType, id)?.props ?? null,
    putObject: (objectType, id, props, sourcePath, importedAt) => store.put({ type: objectType, id, props, sourcePath, importedAt }),
    deleteObject: (objectType, id) => {
      store.deleteObject(objectType, id)
    },
    // fromType/toType ride the LinkEdit for the card; the store takes the endpoint types from the
    // link definition itself (store.ts C7), so they are dropped at this seam.
    putLink: (edit: LinkEdit, sourcePath, importedAt) =>
      store.putLink({ type: edit.linkType, fromId: edit.fromId, toId: edit.toId, props: edit.props, sourcePath, importedAt }),
    appendEditLog: (entry) => {
      insert.run(entry.editId, entry.proposalId, entry.action, entry.actionVersion, entry.ontologyVersion, entry.appliedAt, JSON.stringify(entry))
    },
    listEditLog: () => (selectAll.all() as Array<{ entry: string }>).map((r) => JSON.parse(r.entry) as EditLogEntry),
    // N2's tx is strictly synchronous; everything async happens before this is entered (C8 step 9)
    transaction<T>(fn: () => T): T {
      return store.tx(fn)
    },
  }
}
