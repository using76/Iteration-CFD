// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// The mesh half of gui_control. open_mesh_dialog edits a per-session mesh
// draft held here (memory only - a server restart forgets it), and start_mesh
// runs that draft through the mesh_generate tool (or mesh_regions for the
// regions form): the same schema, the same refusals, the same approval. The
// window's Mesh dialog only shows the draft (UiState.meshDraft).
import { MESH_KINDS, MESH_PRESETS, type MeshDraft, type UiCommand, type UiState } from '@cfd/shared'
import { fail, okResult, type ToolContext, type ToolResult } from './context.js'
import { meshRegions } from './geomTool.js'
import { meshArgs, meshGenerate } from './mesh.js'

type OpenMeshDialogCmd = Extract<UiCommand, { type: 'open_mesh_dialog' }>

/** Session id -> draft. Memory only: a server restart forgets every draft. */
const drafts = new Map<string, MeshDraft>()

const keyOf = (sessionId: string | null | undefined): string => sessionId ?? ''

const copyDraft = (d: MeshDraft): MeshDraft => ({ ...d, cells: d.cells ? [d.cells[0], d.cells[1], d.cells[2]] : null })

/** The session's mesh draft (a copy), or null when open_mesh_dialog has not made one. */
export function getMeshDraft(sessionId: string | null | undefined): MeshDraft | null {
  const d = drafts.get(keyOf(sessionId))
  return d ? copyDraft(d) : null
}

/** Tests only: forget every session's draft. */
export function resetMeshDrafts(): void {
  drafts.clear()
}

/** A UiState with the session's mesh draft filled in; unchanged when the session has none (null stays null: no screen has reported). */
export function withMeshDraft(state: UiState | null | undefined, sessionId: string | null | undefined): UiState | null {
  if (!state) return null
  const meshDraft = getMeshDraft(sessionId)
  return meshDraft ? { ...state, meshDraft } : state
}

/** The mesh tool start_mesh stands in for (by the session draft's form); null for every other command. */
export function guiMeshDelegate(cmd: unknown, sessionId: string | null | undefined): 'mesh_generate' | 'mesh_regions' | null {
  if (typeof cmd !== 'object' || cmd === null || (cmd as { type?: unknown }).type !== 'start_mesh') return null
  return drafts.get(keyOf(sessionId))?.mode === 'regions' ? 'mesh_regions' : 'mesh_generate'
}

/** A preset meshed one cell deep: a single cell count would make it 3-D. */
const isTwoD = (kind: string): boolean => MESH_PRESETS.find((p) => p.kind === kind)?.defaultCells[2] === 1

/** open_mesh_dialog: edit the session's draft (a given field overwrites, an absent one keeps), then show it on screen, best effort. */
export async function applyMeshDialog(cmd: OpenMeshDialogCmd, ctx: ToolContext): Promise<ToolResult> {
  const key = keyOf(ctx.sessionId)
  // A failed call never changes the stored draft: edit a copy, store it at the end.
  const prev = drafts.get(key)
  const cur: MeshDraft = prev ? copyDraft(prev) : { mode: 'preset', preset: null, cells: null, outputDir: null, config: null, layoutDir: null, check: null, dryRun: false }
  if (cmd.preset != null) {
    if (!(MESH_KINDS as readonly string[]).includes(cmd.preset)) return fail('UNKNOWN_PRESET', `unknown mesh preset "${cmd.preset}"; the presets are ${MESH_KINDS.join(', ')}`)
    cur.preset = cmd.preset
  }
  if (cmd.outputDir != null) cur.outputDir = cmd.outputDir
  if (cmd.config != null) cur.config = cmd.config
  if (cmd.layoutDir != null) cur.layoutDir = cmd.layoutDir
  if (cmd.check != null) cur.check = cmd.check
  if (cmd.dryRun != null) cur.dryRun = cmd.dryRun
  cur.mode = cmd.mode ?? (cmd.layoutDir != null ? 'regions' : cmd.config != null ? 'automesher' : cmd.preset != null ? 'preset' : cur.mode)
  if (cmd.cells != null) {
    let cells: [number, number, number]
    if (typeof cmd.cells === 'number') {
      if (cur.preset !== null && isTwoD(cur.preset))
        return fail('CELLS_2D', `${cur.preset} is meshed one cell deep, so one cell count (${cmd.cells}) would fill all three axes; give cells as [nx, ny, nz], e.g. [${cmd.cells}, ${cmd.cells}, 1]`)
      cells = [cmd.cells, cmd.cells, cmd.cells]
    } else {
      cells = [cmd.cells[0], cmd.cells[1], cmd.cells[2]]
    }
    if (cells.some((n) => n < 1)) return fail('INVALID', `cells must be at least 1 in every direction (got [${cells.join(', ')}])`)
    cur.cells = cells
  }
  drafts.set(key, cur)
  const draft = copyDraft(cur)
  // Best effort: the window's Mesh dialog shows the whole draft; its failure does not fail the edit.
  const shown = await ctx.hub.requestUi({ type: 'open_mesh_dialog', ...draft }, { timeoutMs: 5_000, sessionId: ctx.sessionId })
  return okResult({ ok: true, command: 'open_mesh_dialog', draft, shown: shown.ok, state: withMeshDraft(shown.state ?? ctx.hub.getUiState(ctx.sessionId), ctx.sessionId) })
}

type GenerateInput = Parameters<typeof meshGenerate.run>[0]
type RegionsInput = Parameters<typeof meshRegions.run>[0]

export type MeshStartInput = { tool: 'mesh_generate'; input: GenerateInput } | { tool: 'mesh_regions'; input: RegionsInput }

const NO_DRAFT = (): ToolResult => fail('NO_MESH_DRAFT', 'no mesh draft for this session; send open_mesh_dialog {preset, cells, outputDir} (or {config}, or {mode: "regions", config, layoutDir}) first')

/** A delegate schema's refusal, worded as runTool words an invalid input. */
function invalidDraft(tool: string, issues: Array<{ path: PropertyKey[]; message: string }>): ToolResult {
  return fail('INVALID_INPUT', `the mesh draft is not a valid ${tool} input: ${issues.map((i) => `${i.path.map(String).join('.') || '(root)'}: ${i.message}`).join('; ')}`)
}

/** The delegate tool's input from the session draft, parsed by that tool's own schema; a refusal when it cannot be built. */
export function resolveMeshStart(sessionId: string | null | undefined): MeshStartInput | { refusal: ToolResult } {
  const d = drafts.get(keyOf(sessionId))
  if (!d) return { refusal: NO_DRAFT() }
  if (d.mode === 'regions') {
    const r = meshRegions.schema.safeParse({ config: d.config, step: null, outDir: d.layoutDir, tag: null, force: null, waitSeconds: null })
    return r.success ? { tool: 'mesh_regions', input: r.data } : { refusal: invalidDraft('mesh_regions', r.error.issues) }
  }
  const none = { stl: null, cutcell: null, wallModel: null, Ks: null, Cs: null, cyclic: null, permissive: null }
  // An unset preset or outputDir is left out, so mesh_generate's own refusal names it.
  const preset = { kind: d.preset ?? undefined, outputDir: d.outputDir ?? undefined, cells: d.cells, ...none }
  const automesher = { config: d.config, check: d.check !== null, checkDir: d.check, dryRun: d.dryRun, cells: null, ...none }
  const raw = d.mode === 'automesher' ? automesher : preset
  const r = meshGenerate.schema.safeParse(raw)
  return r.success ? { tool: 'mesh_generate', input: r.data } : { refusal: invalidDraft('mesh_generate', r.error.issues) }
}

/** start_mesh: the draft through mesh_generate (or mesh_regions), then a best-effort follow_run on screen. */
export async function guiStartMesh(ctx: ToolContext): Promise<ToolResult> {
  const r = resolveMeshStart(ctx.sessionId)
  if ('refusal' in r) return r.refusal
  const res = r.tool === 'mesh_regions' ? await meshRegions.run(r.input, ctx) : await meshGenerate.run(r.input, ctx)
  if (!res.ok) return res
  const runId = res.runId ?? (res.data as { runId?: string } | undefined)?.runId ?? null
  // Best effort: point the screen at the mesh run; its failure does not fail the start.
  const follow = runId ? await ctx.hub.requestUi({ type: 'follow_run', runId }, { timeoutMs: 5_000, sessionId: ctx.sessionId }) : null
  const state = withMeshDraft(follow?.state ?? ctx.hub.getUiState(ctx.sessionId), ctx.sessionId)
  return okResult({ ok: true, command: 'start_mesh', tool: r.tool, mesh: res.data, followed: follow?.ok ?? false, state }, runId ? { runId } : undefined)
}

/** The approval card for start_mesh: the exact line approvalPreview draws for mesh_generate; null for mesh_regions (it has none) and for no draft. */
export async function guiMeshPreview(cmd: UiCommand, ctx: ToolContext): Promise<string | null> {
  if (cmd.type !== 'start_mesh') return null
  const r = resolveMeshStart(ctx.sessionId)
  if ('refusal' in r || r.tool !== 'mesh_generate') return null
  const m = meshArgs(r.input)
  const args = m.args.map((a) => (a.value === true || a.value === null ? a.flag : `${a.flag} ${String(a.value)}`)).join(' ')
  return [m.binary, ...m.positionals, args].filter(Boolean).join(' ')
}
