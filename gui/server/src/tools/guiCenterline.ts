// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// The centreline half of gui_control (STU-09). set_centerline is resolved here
// on the server: the result (the one on screen, or an explicit path), the
// field (a quantity name maps to its field), the time (the one on screen, else
// the latest, falling back to the latest earlier time that has the field, as
// the viewer does for a frozen U) and the line (explicit p0..p1, or the
// longest axis of the result's bounds through the domain centre). The samples
// are line_sample's, and the window receives set_centerline with every key
// filled, so its chart fetches the same samples from GET /api/results/sample.
import type { UiCommand, ViewerDataset } from '@cfd/shared'
import { listTimeDirs, parseTimeName, resolveResultRoot } from '../formats/results.js'
import { errorMessage, fail, okResult, type ToolContext, type ToolResult } from './context.js'
import { resolveTool } from './paths.js'
import { LINE_SAMPLE_POINTS, SampleError, lineSample, type SampleComponent } from './sample.js'

type SetCenterlineCmd = Extract<UiCommand, { type: 'set_centerline' }>
type Vec3 = [number, number, number]

/** The screen's quantity names (show_field's) and the fields they stand for; any other quantity is a field name. */
const QUANTITY_FIELDS: Record<string, string> = { Temperature: 'T', Velocity: 'U', Pressure: 'p' }

export function centerlineField(quantity: string): string {
  return QUANTITY_FIELDS[quantity] ?? quantity
}

/** The default centreline: through the centre of the bounds along their longest axis (x before y before z on a tie). */
export function centerlineSegment(bounds: { min: readonly number[]; max: readonly number[] }): { axis: 0 | 1 | 2; p0: Vec3; p1: Vec3 } {
  const c: Vec3 = [0.5 * (bounds.min[0]! + bounds.max[0]!), 0.5 * (bounds.min[1]! + bounds.max[1]!), 0.5 * (bounds.min[2]! + bounds.max[2]!)]
  let axis: 0 | 1 | 2 = 0
  for (const a of [1, 2] as const) if (bounds.max[a]! - bounds.min[a]! > bounds.max[axis]! - bounds.min[axis]!) axis = a
  const p0: Vec3 = [c[0], c[1], c[2]]
  const p1: Vec3 = [c[0], c[1], c[2]]
  p0[axis] = bounds.min[axis]!
  p1[axis] = bounds.max[axis]!
  return { axis, p0, p1 }
}

/** The time label to sample: the target (or the latest) when it has the field, else the latest earlier time that has it. */
async function pickTime(rootAbs: string, target: string | null, field: string): Promise<string | null> {
  const times = await listTimeDirs((await resolveResultRoot(rootAbs)).rootAbs)
  if (times.length === 0) return target
  let i = times.length - 1
  if (target !== null) {
    const v = parseTimeName(target)
    const j = times.findIndex((t) => t.name === target || (v !== null && t.value === v))
    // An unknown label goes to line_sample as given, which names the available times.
    if (j < 0) return target
    i = j
  }
  for (let k = i; k >= 0; k--) if (times[k]!.fields.includes(field)) return times[k]!.name
  return times[i]!.name
}

const OPEN_TIMEOUT_MS = 60_000

/** Open a result as the viewer does and wait for its manifest (its progress stream says ready or error); a string is the failure. */
async function openReady(relPath: string, ctx: ToolContext): Promise<ViewerDataset | string> {
  let settle: (s: 'ready' | 'error') => void = () => {}
  const done = new Promise<'ready' | 'error'>((resolve) => (settle = resolve))
  // Subscribed before open, so a load that finishes at once is not missed.
  let id: string | null = null
  const off = ctx.datasets.onProgress((p) => {
    if (p.datasetId === id && (p.stage === 'ready' || p.stage === 'error')) settle(p.stage)
  })
  const timer = setTimeout(() => settle('error'), OPEN_TIMEOUT_MS)
  try {
    const r = await ctx.datasets.open(relPath)
    id = r.datasetId
    if (r.status === 'error') return r.error ?? 'the result could not be opened'
    if (r.manifest) return r.manifest
    // A load that ended between open and the id being set has its manifest (or not) already.
    const now = ctx.datasets.get(id)
    if (now) return now
    const stage = await done
    return ctx.datasets.get(id) ?? (stage === 'error' ? `the result failed to load, or did not load within ${OPEN_TIMEOUT_MS / 1000} s` : 'the result loaded without a manifest')
  } finally {
    clearTimeout(timer)
    off()
  }
}

/** The result to sample: the explicit path (opened as the viewer opens it), else the dataset on screen. */
async function centerlineDataset(cmd: SetCenterlineCmd, ctx: ToolContext): Promise<{ manifest: ViewerDataset; onScreen: boolean } | ToolResult> {
  if (cmd.path != null && cmd.path !== '') {
    try {
      const manifest = await openReady(cmd.path, ctx)
      return typeof manifest === 'string' ? fail('NO_RESULT', `${cmd.path}: ${manifest}`) : { manifest, onScreen: false }
    } catch (err) {
      return fail('NO_RESULT', `${cmd.path}: ${errorMessage(err)}`)
    }
  }
  const id = ctx.hub.getUiState(ctx.sessionId)?.viewer?.datasetId ?? null
  const manifest = id === null ? undefined : ctx.datasets.get(id)
  if (!manifest) return fail('NO_RESULT', 'no result is on screen; open_result first, or give set_centerline a path')
  return { manifest, onScreen: true }
}

export async function guiSetCenterline(cmd: SetCenterlineCmd, ctx: ToolContext): Promise<ToolResult> {
  if ((cmd.p0 == null) !== (cmd.p1 == null)) return fail('INVALID', 'give both p0 and p1 for an explicit line, or neither for the longest axis through the domain centre')
  const ds = await centerlineDataset(cmd, ctx)
  if ('ok' in ds) return ds
  const { manifest, onScreen } = ds
  const field = centerlineField(cmd.quantity)
  const component: SampleComponent = cmd.component ?? 'magnitude'
  const line = cmd.p0 != null && cmd.p1 != null ? { axis: null, p0: [...cmd.p0] as Vec3, p1: [...cmd.p1] as Vec3 } : centerlineSegment(manifest.bounds)
  const r = resolveTool(ctx.workspaceRoot, manifest.path, { mustExist: true })
  if (!r.ok) return r.result
  const screenTime = onScreen ? (ctx.hub.getUiState(ctx.sessionId)?.viewer?.time ?? null) : null
  const target = cmd.time != null && cmd.time !== '' ? cmd.time : screenTime === null ? null : (manifest.times.find((t) => t.value === screenTime)?.label ?? null)
  let sampled: Awaited<ReturnType<typeof lineSample>>
  try {
    const time = await pickTime(r.path.abs, target, field)
    sampled = await lineSample({ rootAbs: r.path.abs, rootRel: r.path.rel || '.', time, field, component, p0: line.p0, p1: line.p1, n: LINE_SAMPLE_POINTS })
  } catch (err) {
    if (err instanceof SampleError) return fail(err.code, err.message)
    return fail('SAMPLE_FAILED', errorMessage(err))
  }
  // Best effort: the window's chart draws the same line; its failure does not fail the samples.
  const forward: SetCenterlineCmd = { type: 'set_centerline', quantity: field, component, p0: line.p0, p1: line.p1, path: manifest.path, time: sampled.time }
  const shown = await ctx.hub.requestUi(forward, { timeoutMs: 5_000, sessionId: ctx.sessionId })
  return okResult({
    ok: true,
    command: 'set_centerline',
    path: manifest.path,
    field,
    component,
    time: sampled.time,
    axis: line.axis === null ? null : (['x', 'y', 'z'] as const)[line.axis],
    p0: line.p0,
    p1: line.p1,
    cells: sampled.cells,
    s: sampled.s,
    x: sampled.x,
    values: sampled.values,
    shown: shown.ok,
    state: shown.state ?? ctx.hub.getUiState(ctx.sessionId),
  })
}
