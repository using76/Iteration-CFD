// gui/server/src/ontology/prepare.ts — an action's preparer computes the values its rules read
// through { from:'prepared', key } (N4 D-B). It is READ-ONLY: it may stat a file, hash a file or
// call N0's gitHead, and it writes nothing. At phase:'propose' nothing is spawned and before.spawn
// is null, so the card shows the literal placeholder r_? (N4 D-C); at phase:'apply' the BEFORE side
// effect has already spawned, so the Run's primary key is the id the run manager minted — the
// engine never mints a Run id.
import type { RunInfo } from '@cfd/shared'
import { createReadStream } from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'
import { createHash } from 'node:crypto'
import { resolveInWorkspace } from '../workspace/paths.js'
import { ATTACHMENT_CAP } from '../agent/service.js'   // the chat fold's cap, reused so it stays one number
import { mediaTypeFor, type CriterionContext } from './criteria.js'

export interface PrepareInput { spawn: RunInfo | null }
export type Preparer = (ctx: Omit<CriterionContext, 'prepared'>, phase: 'propose' | 'apply', before: PrepareInput) => Promise<Record<string, unknown>>

const str = (v: unknown): string => (v === null || v === undefined ? '' : String(v))
const list = (v: unknown): string[] => (Array.isArray(v) ? v : []).map(str)
const argList = (v: unknown): Array<{ flag: unknown; value: unknown }> => (Array.isArray(v) ? v : []) as Array<{ flag: unknown; value: unknown }>

/** The command line as the manager will build it (test-fakes.ts:219 is the same shape), so the
 *  card does not lie about what will run. */
function computedArgv(ctx: Omit<CriterionContext, 'prepared'>): string[] {
  const casePath = str(ctx.params.casePath)
  return [
    ...(casePath ? [casePath] : []),
    ...list(ctx.params.positionals),
    ...argList(ctx.params.args).flatMap((a) => [str(a.flag), str(a.value)]),
  ]
}

export const startRunPrepare: Preparer = async (ctx, phase, before) => {
  const head = await ctx.server.gitHead()
  return {
    spawnedRunId: phase === 'propose' ? 'r_?' : before.spawn!.id,
    argv:         phase === 'propose' ? computedArgv(ctx) : before.spawn!.argv,
    mode:         phase === 'propose' ? 'real'       : before.spawn!.mode,
    gitSha:       head.sha,                          // string | null, from N0's reader (N4 D-P)
    gitDirty:     head.dirty,                        // boolean | null
  }
}

// ---- attachFile (N4 Run 3, C12): the stream, the stat and the extension table -------------

/** The 16 KB text extract for text-ish media, read from the file head — the ONLY bytes this
 *  preparer ever holds; the hash streams and nothing buffers the whole file (N4 fact 4). */
async function headText(abs: string): Promise<string | null> {
  const fh = await fsp.open(abs, 'r')
  try {
    const buf = Buffer.alloc(ATTACHMENT_CAP)
    const { bytesRead } = await fh.read(buf, 0, ATTACHMENT_CAP, 0)
    return buf.subarray(0, bytesRead).toString('utf8')
  } finally {
    await fh.close()
  }
}

/** Runs IDENTICALLY at both phases (C12), so the edit set the card shows is the edit set that
 *  lands. No file bytes enter the returned map — a hash, a size, a name, a media type, a path
 *  and a capped extract; width/height stay null (no image decoder here, D-D). */
export const attachFilePrepare: Preparer = async (ctx) => {
  const rel = str(ctx.params.path)
  const name = str(ctx.params.filename) || path.basename(rel)
  let rp: ReturnType<typeof resolveInWorkspace> | null = null
  try {
    rp = resolveInWorkspace(ctx.server.workspaceRoot, rel, { mustExist: false })
  } catch { rp = null }
  if (rp === null || !rp.exists) {
    // A bad path is refused BY NAME by the criteria (the engine runs them after the preparer);
    // a throw here would be swallowed by previewProposal into a blank approval card.
    return { sha256: '', bytes: 0, mediaType: mediaTypeFor(name), filename: name, storedPath: rel, textExtract: null, sessionId: ctx.principal.sessionId }
  }
  const abs = rp.abs
  const st = await fsp.stat(abs)
  const hash = createHash('sha256')
  await new Promise<void>((res, rej) => {
    createReadStream(abs).on('data', (c) => hash.update(c)).on('end', () => res()).on('error', rej)
  })
  const mediaType = mediaTypeFor(name)
  return {
    sha256: hash.digest('hex'),
    bytes: st.size,
    mediaType,
    filename: name,
    storedPath: rel,
    textExtract: mediaType === 'text/plain' || mediaType === 'application/json' ? await headText(abs) : null,
    sessionId: ctx.principal.sessionId,
  }
}

import { DC_PREPARERS } from './actions.dc.js'
export const PREPARERS: Record<string, Preparer> = {
  startRun: startRunPrepare,
  attachFile: attachFilePrepare,
  ...DC_PREPARERS,
}
