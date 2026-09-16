// gui/server/src/ontology/handle.ts — one store+engine per mirror file, memoised for the life of
// the process (the geometryService shape), plus the per-session preview hook that fills N5's
// dormant TurnDeps.ontologyPreview. The memo is what makes a proposal survive between
// ontology_act and ontology_apply: both tools, both routes and the preview hold ONE engine, and
// the engine holds the proposals. A restart loses them; ontology_apply then answers NOT_FOUND
// and the model re-proposes — accepted behaviour, no proposal table in this unit.
import { ONTOLOGY, type ActionTypeDef } from '@cfd/shared'
import type { ServerConfig } from '../config.js'
import type { RunManager } from '../runs/types.js'
import type { Hub } from '../ws/types.js'
import { PREVIEW_CAP } from '../tools/index.js' // read lazily (below); only a constant
import { gitHead } from '../workspace/git.js'
import { createActionEngine, type OntologyActionEngine } from './engine.js'
import type { ProposalPreview } from '../agent/loop.js'
import { ONTOLOGY_ACT_TOOL, createProposalPreview } from './preview.js'
import { ontologyDbPath, openOntologyStore, type OntologyStore } from './store.js'
import { installProposeImport } from '../corpus/importBatch.js'

export interface OntologyDeps {
  config: ServerConfig
  runs: Pick<RunManager, 'list' | 'get' | 'start'>
  /** Optional: when present, an action's notify side effect reaches the session (N6 wires it). */
  hub?: Pick<Hub, 'sendToSession'>
}

export interface OntologyHandle {
  store: OntologyStore
  engine: OntologyActionEngine
}

const handles = new Map<string, Promise<OntologyHandle>>()

/** One store+engine per database file, memoised by the promise so two concurrent calls never open the file twice. */
export function ontologyHandle(deps: OntologyDeps): Promise<OntologyHandle> {
  const key = ontologyDbPath(deps.config)
  let p = handles.get(key)
  if (!p) {
    p = build(deps).catch((err: unknown) => {
      handles.delete(key)
      throw err
    })
    handles.set(key, p)
  }
  return p
}

async function build(deps: OntologyDeps): Promise<OntologyHandle> {
  const store = openOntologyStore({ path: ontologyDbPath(deps.config), ontology: ONTOLOGY })
  // One gitHead read per process, not one per proposal: a proposal's atCommit names the HEAD the
  // server booted at, and a 3 s execFile has no place in the approval card's critical path.
  const head = await gitHead(deps.config.workspaceRoot)
  const engine = createActionEngine({
    actions: ONTOLOGY.actionTypes as ActionTypeDef[],
    registry: ONTOLOGY,
    store: installProposeImport(store),
    server: {
      workspaceRoot: deps.config.workspaceRoot,
      now: () => Date.now(),
      gitHead: async () => head,
      runs: deps.runs,
    },
  })
  return { store, engine }
}

/** Only for tests and for a server shutting down; called by nothing in production. */
export async function closeOntologyHandles(): Promise<void> {
  const all = [...handles.values()]
  handles.clear()
  for (const p of all) {
    try {
      ;(await p).store.close()
    } catch {
      // a handle that already closed stays closed
    }
  }
}

/** The per-session hook for TurnDeps.ontologyPreview. Never throws: a thrown preview shows a
 *  blank card instead of an error. The preview text is capped at PREVIEW_CAP characters on the
 *  last newline before the cap, with the number of cut lines named. */
export function ontologyPreviewFor(deps: OntologyDeps, sessionId: string): ProposalPreview {
  return async (name, input, toolUseId) => {
    if (name !== ONTOLOGY_ACT_TOOL) return null
    try {
      const h = await ontologyHandle(deps)
      const text = await createProposalPreview(h.engine, sessionId)(name, input, toolUseId)
      return text === null ? null : capPreview(text)
    } catch {
      return null
    }
  }
}

function capPreview(text: string): string {
  if (text.length <= PREVIEW_CAP) return text
  const lines = text.split('\n')
  const kept: string[] = []
  let used = 0
  let i = 0
  for (; i < lines.length; i++) {
    const cost = lines[i].length + (i === 0 ? 0 : 1)
    if (used + cost > PREVIEW_CAP) break
    kept.push(lines[i])
    used += cost
  }
  const more = lines.length - kept.length
  return more > 0 ? `${kept.join('\n')}\n… and ${more} more` : text
}
