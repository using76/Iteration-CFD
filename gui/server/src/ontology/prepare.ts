// gui/server/src/ontology/prepare.ts — an action's preparer computes the values its rules read
// through { from:'prepared', key } (N4 D-B). It is READ-ONLY: it may stat a file, hash a file or
// call N0's gitHead, and it writes nothing. At phase:'propose' nothing is spawned and before.spawn
// is null, so the card shows the literal placeholder r_? (N4 D-C); at phase:'apply' the BEFORE side
// effect has already spawned, so the Run's primary key is the id the run manager minted — the
// engine never mints a Run id.
import type { RunInfo } from '@cfd/shared'
import type { CriterionContext } from './criteria.js'

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

export const PREPARERS: Record<string, Preparer> = {
  startRun: startRunPrepare,
}
