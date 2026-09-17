// gui/server/src/ontology/preview.ts — the approval card's preview (CONTRACT §8): plain text,
// one grammar, never JSON. previewProposal proposes under the caller's toolUseId (which writes
// nothing — propose stops before the store), binds the proposal to that id, and returns the
// edit-set summary, or the blocking lines when the proposal was refused. N5's ontology_act finds
// the operator's proposal by the same toolUseId, so what was approved is what applies.
import { ONTOLOGY, type OntologyRegistry, type Principal, type Proposal } from '@cfd/shared'
import type { OntologyActionEngine } from './engine.js'
import type { ProposalPreview } from '../agent/loop.js'

export const ONTOLOGY_ACT_TOOL = 'ontology_act'

/** CONTRACT §8, verbatim: the summary if there is one, else one `blocked: ` line per criterion. */
export function proposalPreviewText(p: Proposal, _registry: OntologyRegistry): string {
  return p.edits?.summary ?? p.blocking.map((c) => `blocked: ${c.message}`).join('\n')
}

// The engine exposes get(proposalId) but no way to iterate its proposals, so the toolUseId
// binding is indexed here — one WeakMap per engine, so two handles never share an index and a
// closed handle's index is collected with it.
const byToolUse = new WeakMap<OntologyActionEngine, Map<string, string>>()
function index(engine: OntologyActionEngine): Map<string, string> {
  let m = byToolUse.get(engine)
  if (!m) {
    m = new Map()
    byToolUse.set(engine, m)
  }
  return m
}

export function proposalByToolUseId(engine: OntologyActionEngine, toolUseId: string): Proposal | undefined {
  const id = index(engine).get(toolUseId)
  return id ? engine.get(id) : undefined
}

/** Propose under toolUseId and return the preview text. Catches everything: a thrown preview is
 *  a blank card on the operator's screen, which is worse than an honest null. */
export async function previewProposal(engine: OntologyActionEngine, name: string, input: unknown, principal: Principal, toolUseId: string): Promise<string | null> {
  try {
    if (name !== ONTOLOGY_ACT_TOOL) return null
    const i = (input ?? {}) as Record<string, unknown>
    const p = await engine.propose(String(i.action ?? ''), (i.parameters ?? {}) as Record<string, unknown>, principal, toolUseId)
    index(engine).set(toolUseId, p.proposalId)
    return proposalPreviewText(p, ONTOLOGY)
  } catch {
    return null
  }
}

export function createProposalPreview(engine: OntologyActionEngine, sessionId: string): ProposalPreview {
  return (name, input, toolUseId) =>
    previewProposal(engine, name, input, { kind: 'agent', id: 'assistant', sessionId }, toolUseId)
}
