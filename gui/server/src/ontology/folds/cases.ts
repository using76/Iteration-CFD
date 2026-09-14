// gui/server/src/ontology/folds/cases.ts — every *.jsonc under the workspace becomes one Case
// row, keyed on its workspace-relative path (C9). The family rule is content, not suffix (D8).
import path from 'node:path'
import { readCaseJsonc } from '../../formats/casejsonc.js'
import { emptyFoldReport, skip, upsert, walkFiles, type FoldContext, type FoldReport, type MirrorRow } from './base.js'

/** The family rule, in this order (C9, D8): the suffix rule reproduces 12 of the 14 measured
 *  rows, the content rule reproduces 14 of 14 - the measurement wins. The fourth declared
 *  enum value, 'foamDir', is for a directory-shaped case; this fold only ever opens *.jsonc
 *  files, so it never produces that word. */
function caseFormat(rel: string, parsed: unknown): string {
  if (rel.endsWith('.dc.jsonc')) return 'dc'
  if (rel.endsWith('.cht.jsonc')) return 'cht-1'
  if (parsed !== null && typeof parsed === 'object' && Array.isArray((parsed as Record<string, unknown>).regions)) return 'cht-1'
  return 'case-1'
}

export async function foldCases(ctx: FoldContext): Promise<FoldReport[]> {
  const rep = emptyFoldReport(ctx.type.case)
  const t0 = Date.now()
  if (!ctx.writer.hasObjectType(ctx.type.case)) {
    skip(rep, ctx.type.case, 'not declared in the ontology registry; the fold writes no row')
    rep.seconds = (Date.now() - t0) / 1000
    return [rep]
  }
  for await (const f of walkFiles(ctx.workspaceRoot, ctx.workspaceRoot, (name) => name.endsWith('.jsonc'))) {
    rep.filesRead++
    const info = await readCaseJsonc(f.abs, f.rel)
    const parsed = info.json
    const obj = parsed !== null && typeof parsed === 'object' ? (parsed as Record<string, unknown>) : null
    const stem = path.basename(f.rel).endsWith('.jsonc') ? path.basename(f.rel).slice(0, -'.jsonc'.length) : path.basename(f.rel)
    const topPatches = obj !== null && Array.isArray(obj.patches) ? obj.patches.length : 0
    const regionPatches = info.regions.reduce((n, r) => n + r.patches.length, 0)
    const schemaUrl = obj !== null && typeof obj.$schema === 'string' ? obj.$schema : null
    const row: MirrorRow = {
      objectType: ctx.type.case,
      primaryKey: f.rel,
      properties: {
        caseId: f.rel,
        name: info.name ?? stem,
        format: caseFormat(f.rel, parsed),
        turbulenceKind: info.turbulenceKind,
        turbulenceModel: info.model,
        mode: info.mode,
        regionsManifest: info.regionsManifest,
        outputDir: info.outputDir,
        nRegions: info.regions.length,
        nInterfaces: info.interfaces.length,
        nPatchRules: topPatches + regionPatches,
        runBlock: info.run,
        schemaUrl,
      },
      sourcePath: f.rel,
      importedAt: ctx.now(),
    }
    await upsert(ctx, rep, row)
    if (info.errors.length > 0)
      skip(rep, 'case parse errors', 'the case parsed with recoverable JSONC errors; the ontology declares no place to record them', info.errors.length)
    if (caseFormat(f.rel, parsed) === 'cht-1' || f.rel.endsWith('.cht.jsonc'))
      skip(rep, 'case regions[]/interfaces[]', 'region and interface rows come from regions.json; the case keeps the counts')
  }
  rep.notes.push('usesLayout: not written in this unit; Case.regionsManifest resolves against the case directory')
  rep.notes.push("the 'foamDir' format value is declared for a directory-shaped case; this fold only opens *.jsonc files")
  rep.notes.push("a case's family is decided by content, not by suffix: an Array-valued regions key makes a plain-named file a cht case (D8)")
  rep.seconds = (Date.now() - t0) / 1000
  return [rep]
}
