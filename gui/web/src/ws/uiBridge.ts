// Glue between the WS protocol and the studio screen for the ui.command /
// ui.result / ui.state trio: the agent's gui_control tool drives this client
// through the hub, and every command MUST be answered — an unanswered one
// parks the model for the hub's full 5 s timeout and it retries into the
// same void. Commands this web app can genuinely apply are applied here;
// everything else is refused immediately with UNSUPPORTED so the model can
// tell the operator the truth instead of waiting.
import type { ClientMsg, MeshDraft, MetricRecord, OpenMeshDialog, UiCommand, UiSelection, UiState, ViewId, ViewerCommand } from '@cfd/shared'
import { ViewerCommandSchema } from '@cfd/shared'
import { getViewerApi } from '../viewer'
import type { ProbeOutcome, ViewerTool, ViewportRect } from '../viewer'
import { linkCameras } from '../viewer/api/cameraLink'
import { actions, activeCasePath } from './actions'
import { uiGeometryState, useGeometryStore } from '../state/geometryStore'
import type { useSessionStore } from '../state/sessionStore'
// Type-only: the value import would pull monaco into the node tests (see editor below).
import type { SaveCaseOutcome } from '../state/editorStore'
// tabStateLabel / tabScreenLabel are value imports; uiStore has no DOM imports,
// so this stays node-test safe. useUiStore itself must NOT be imported here
// (the bridge takes its store through deps so tests can inject fakes).
import { tabScreenLabel, tabStateLabel, type LogStream, type useUiStore } from '../state/uiStore'
// Pure series helpers (metric matching, surface keys); series.ts only type-imports
// the store, so this value import stays node-test safe too.
import { findMetric, metricNames, surfaceSeriesKeys } from '../chart/series'

type SessionStore = ReturnType<typeof useSessionStore.getState>
type UiStoreState = ReturnType<typeof useUiStore.getState>

export interface UiBridgeDeps {
  send(msg: ClientMsg): boolean
  ui: { getState(): UiStoreState }
  session: { getState(): SessionStore }
  /** Client.ts owns run subscriptions; a followed run is one of them. */
  subscribeRun(runId: string): void
  /** Where a post_screenshot PNG goes; the default downloads it like Viewer3D's Snapshot button. Tests inject a recorder. */
  saveImage?(name: string, base64: string): void
  /** How long set_camera / fit_view may wait for the canvas to mount and the camera move to finish before answering; default CAMERA_ANSWER_MS. Tests inject a short one. */
  cameraBudgetMs?: number
  /** close_tab's bookkeeping on the editor store. Injected by client.ts; kept
   *  optional and behind this interface because importing editorStore here
   *  would pull monaco into the node tests. */
  editor?: {
    isDirty(path: string): boolean
    closeBuffer(path: string): void
    removeDiff(id: string): void
    /** save_case: validate the buffer on the server, refuse on errors unless force, then write it (editorStore.saveCase). */
    saveCase?(path: string, force: boolean): Promise<SaveCaseOutcome>
  }
}

/** set_camera / fit_view always answer within this, below the hub's 5 s UI_TIMEOUT_MS. */
export const CAMERA_ANSWER_MS = 4_000

/** show_field / post_field name the three display fields; the solver names them U/p/T. */
const FIELD_MAP: Record<string, { name: string; component: 'magnitude' | null }> = {
  Temperature: { name: 'T', component: null },
  Velocity: { name: 'U', component: 'magnitude' },
  Pressure: { name: 'p', component: null },
}

/** The four geometry edit ui.commands this screen applies through the
 *  geometry studio store (E1). */
type GeometryUiCommand = Extract<UiCommand, { type: 'geometry_part' | 'geometry_transform' | 'geometry_boolean' | 'geometry_save' }>

// ---- open_mesh_dialog / open_mesh_view (STU-06b) ---------------------------

/** An empty draft, as the server's guiMesh.ts starts one. */
const EMPTY_MESH_DRAFT: MeshDraft = { mode: 'preset', preset: null, cells: null, outputDir: null, config: null, layoutDir: null, check: null, dryRun: false }

/** open_mesh_dialog -> the whole draft the Mesh dialog shows. The server sends
 *  its whole draft; a field a command leaves out keeps the shown draft's value,
 *  one cell count fills three axes, and an absent mode follows the given form
 *  as guiMesh.ts infers it. Pure. */
export function meshDraftOf(cmd: OpenMeshDialog, prev: MeshDraft | null): MeshDraft {
  const d: MeshDraft = { ...(prev ?? EMPTY_MESH_DRAFT) }
  if (cmd.preset != null) d.preset = cmd.preset
  if (cmd.outputDir != null) d.outputDir = cmd.outputDir
  if (cmd.config != null) d.config = cmd.config
  if (cmd.layoutDir != null) d.layoutDir = cmd.layoutDir
  if (cmd.check != null) d.check = cmd.check
  if (cmd.dryRun != null) d.dryRun = cmd.dryRun
  if (cmd.cells != null) d.cells = typeof cmd.cells === 'number' ? [cmd.cells, cmd.cells, cmd.cells] : [cmd.cells[0], cmd.cells[1], cmd.cells[2]]
  else if (d.cells) d.cells = [d.cells[0], d.cells[1], d.cells[2]]
  d.mode = cmd.mode ?? (cmd.layoutDir != null ? 'regions' : cmd.config != null ? 'automesher' : cmd.preset != null ? 'preset' : d.mode)
  return d
}

/** The directory a draft's mesh lands in, for open_mesh_view: the preset form's
 *  outputDir, the automesher's -check case, the regions form's layout root; null
 *  when the draft names none. */
export function meshOutputOf(d: MeshDraft | null): string | null {
  if (!d) return null
  const p = d.mode === 'regions' ? d.layoutDir : d.mode === 'automesher' ? d.check : d.outputDir
  return p && p.trim() ? p.trim() : null
}

// ---- add_layer / remove_layer (STU-04) -------------------------------------
// A weaker model names layers loosely ("iso-surface", "ARROWS") and packs its
// options into JSON strings; these helpers normalise all of that into the
// viewer's own add* commands before the schema sees them.

/** The five layer kinds add_layer maps onto the viewer's add* commands. */
export const LAYER_KINDS = ['slice', 'plane', 'isoSurface', 'streamlines', 'glyphs'] as const
export type LayerKind = (typeof LAYER_KINDS)[number]

/** The args keys each kind takes; anything else is refused naming the key. */
const LAYER_ARGS: Record<LayerKind, readonly string[]> = {
  slice: ['id', 'axis', 'position', 'fraction'],
  plane: ['id', 'origin', 'point', 'normal'],
  isoSurface: ['id', 'field', 'values', 'value'],
  streamlines: ['id', 'field', 'seed', 'style', 'maxLength', 'direction'],
  glyphs: ['id', 'field', 'stride', 'scale', 'onSlice'],
}

/** "Iso-Surface" -> isoSurface: lower-case, drop everything outside a-z, then match the synonyms. */
export function layerKind(kind: string): LayerKind | null {
  const k = kind.toLowerCase().replace(/[^a-z]/g, '')
  if (k === 'slice' || k === 'slices') return 'slice'
  if (k === 'plane' || k === 'planes' || k === 'cutplane' || k === 'cuttingplane') return 'plane'
  if (k === 'iso' || k === 'isosurface' || k === 'isosurfaces' || k === 'isocontour' || k === 'contour' || k === 'contours') return 'isoSurface'
  if (k === 'streamline' || k === 'streamlines' || k === 'stream' || k === 'streamtrace' || k === 'streamtraces') return 'streamlines'
  if (k === 'glyph' || k === 'glyphs' || k === 'arrow' || k === 'arrows' || k === 'vector' || k === 'vectors' || k === 'quiver') return 'glyphs'
  return null
}

/** A string holding one JSON value ("{...}" / "[...]") becomes that value; anything else passes through. */
const jsonUnwrap = (v: unknown): unknown => {
  if (typeof v !== 'string') return v
  const t = v.trim()
  if (!t.startsWith('{') && !t.startsWith('[')) return v
  try {
    return JSON.parse(t)
  } catch {
    return v
  }
}

/** x / y / z (any case) -> the unit vector a plane normal wants. */
const AXIS_UNIT: Record<string, [number, number, number]> = { x: [1, 0, 0], y: [0, 1, 0], z: [0, 0, 1] }

/** Loose add_layer args -> one schema-valid viewer command, or the refusal
 *  that names what was wrong. Pure: no store, no viewer access. */
export function layerCommand(
  kind: LayerKind,
  args: Record<string, unknown>,
  ctx: { field: string | null; bounds: { min: [number, number, number]; max: [number, number, number] } | null },
): { ok: true; cmd: ViewerCommand } | { ok: false; error: string } {
  for (const key of Object.keys(args))
    if (!LAYER_ARGS[kind].includes(key))
      return { ok: false, error: `INVALID (add_layer): args.${key} is not an option of ${kind}; options: ${LAYER_ARGS[kind].join(', ')}` }
  // One level of JSON unwrapping on every value, and one more inside a plain-object seed.
  const a: Record<string, unknown> = {}
  for (const [key, v] of Object.entries(args)) a[key] = jsonUnwrap(v)
  if (a.seed != null && typeof a.seed === 'object' && !Array.isArray(a.seed)) {
    const seed: Record<string, unknown> = {}
    for (const [key, v] of Object.entries(a.seed as Record<string, unknown>)) seed[key] = jsonUnwrap(v)
    a.seed = seed
  }
  const id = typeof a.id === 'string' && a.id !== '' ? a.id : typeof a.id === 'number' && Number.isFinite(a.id) ? String(a.id) : null
  const lower = (v: unknown): string | null => (typeof v === 'string' ? v.toLowerCase() : null)
  let cmd: Record<string, unknown>
  if (kind === 'slice') {
    cmd = {
      type: 'addSlice',
      id,
      axis: lower(a.axis),
      position: a.position ?? (a.fraction !== undefined ? { fraction: a.fraction } : { fraction: 0.5 }),
    }
  } else if (kind === 'plane') {
    const b = ctx.bounds
    cmd = {
      type: 'addPlane',
      id,
      origin: a.origin ?? a.point ?? (b ? ([0, 1, 2].map((i) => (b.min[i] + b.max[i]) / 2) as [number, number, number]) : undefined),
      normal: typeof a.normal === 'string' ? (AXIS_UNIT[a.normal.trim().toLowerCase()] ?? a.normal) : a.normal,
    }
  } else if (kind === 'isoSurface') {
    let values = a.values ?? a.value
    if (typeof values === 'number' || typeof values === 'string') values = [values]
    cmd = { type: 'addIsoSurface', id, field: a.field ?? ctx.field ?? undefined, values }
  } else if (kind === 'streamlines') {
    cmd = { type: 'addStreamlines', id, field: a.field ?? null, seed: a.seed, style: lower(a.style), maxLength: a.maxLength ?? null, direction: lower(a.direction) }
  } else {
    cmd = { type: 'addGlyphs', id, field: a.field ?? null, stride: a.stride ?? null, scale: a.scale ?? null, onSlice: a.onSlice ?? null }
  }
  const parsed = ViewerCommandSchema.safeParse(cmd)
  if (!parsed.success) {
    const issues = parsed.error.issues
      .slice(0, 4)
      .map((i) => `args.${i.path.join('.')}: ${i.message}`)
    return { ok: false, error: `INVALID (add_layer): ${issues.join('; ')}` }
  }
  return { ok: true, cmd: parsed.data }
}

const asVec3 = (v: unknown): v is [number, number, number] => Array.isArray(v) && v.length === 3 && v.every((x) => typeof x === 'number')

/** A number s becomes [s, s, s]; a Vec3 must be all non-zero. */
const asScale = (v: unknown): [number, number, number] | null => {
  if (typeof v === 'number') return v === 0 ? null : [v, v, v]
  if (asVec3(v) && v.every((x) => x !== 0)) return v
  return null
}

const partResult = (err: string | null): { ok: true } | { ok: false; error: string } => (err == null ? { ok: true } : { ok: false, error: err })

function unsupported(type: string, why: string): { ok: false; error: string } {
  return { ok: false, error: `UNSUPPORTED (${type}): ${why}` }
}

/** The one-way camera link between the split halves; null while they are free. */
let unlinkCameras: (() => void) | null = null
function dropLink(): void {
  unlinkCameras?.()
  unlinkCameras = null
}
/** The focused half leads; a re-link replaces the old subscription entirely. */
function relink(leader: ViewId): void {
  dropLink()
  unlinkCameras = linkCameras(getViewerApi(leader), getViewerApi(leader === 'A' ? 'B' : 'A'))
}

type UiToolName = 'select' | 'move' | 'pan' | 'box' | 'probe'

/** ui.command tool -> viewer tool: probe IS the select tool (Toolbar.tsx "Select / probe"); box IS the section clip box; move orbits. */
const TOOL_TO_VIEWER: Record<UiToolName, ViewerTool> = { select: 'select', probe: 'select', pan: 'pan', move: 'rotate', box: 'section' }
/** viewer tool -> the ui name reported in ui.state.tool; zoom has no ui name and is reported as itself. */
const TOOL_TO_UI: Record<ViewerTool, string> = { select: 'select', pan: 'pan', rotate: 'move', section: 'box', zoom: 'zoom' }

/** The refusal for a viewer command with no canvas on screen; the case opens the viewer tab first so a retry works. */
function noViewer(type: string): { ok: false; error: string } {
  return { ok: false, error: `NO_VIEWER (${type}): no viewer canvas is mounted; the viewer tab has been opened - retry once it shows` }
}

/** Where a post_screenshot PNG goes without an injected sink: Viewer3D.tsx's Snapshot download, guarded for headless use. */
function defaultSaveImage(name: string, base64: string): void {
  if (typeof document === 'undefined') return
  const a = document.createElement('a')
  a.href = `data:image/png;base64,${base64}`
  a.download = name
  a.click()
}

export function createUiBridge(deps: UiBridgeDeps) {
  const { send, ui, session } = deps

  /** The last probe's answer, reported as ui.state.selection until the next probe. */
  let lastSelection: UiSelection | null = null

  /** The geometry tab a command targets and gui_state reports: the active tab
   *  when it is a geometry tab, else the first geometry tab. */
  function targetGeometryPath(): string | null {
    const u = ui.getState()
    const active = u.tabs.find((t) => t.id === u.activeTabId)
    if (active?.kind === 'geometry') return active.path
    const first = u.tabs.find((t) => t.kind === 'geometry')
    return first?.kind === 'geometry' ? first.path : null
  }

  /** What the model's gui_state reads. Every required field is filled; the
   *  nullable ones this app cannot know stay null rather than being guessed. */
  function snapshot(): UiState {
    const u = ui.getState()
    const s = session.getState()
    const casePath = activeCasePath()
    const v = getViewerApi()
    const vs = v.getState()
    return {
      activeTab: u.activeTabId,
      activeStep: null,
      rightTab: u.assistantVisible ? 'AI Assistant' : null,
      tool: TOOL_TO_UI[v.getTool()],
      frame: null,
      projection: vs.camera.projection === 'orthographic' ? 'Orthographic' : 'Perspective',
      showAxes: typeof u.showAxes === 'boolean' ? u.showAxes : null,
      showColorBars: typeof u.showColorBars === 'boolean' ? u.showColorBars : null,
      overlays: typeof u.showAxes === 'boolean' && typeof u.showColorBars === 'boolean' ? { axes: u.showAxes, colorBars: u.showColorBars } : null,
      bottomTab: u.bottomVisible ? (u.bottomTab ?? null) : null,
      logs: u.logFilter ? { text: u.logFilter.text, streams: [...u.logFilter.streams], follow: u.logFilter.follow } : null,
      metricPick: Array.isArray(u.metricSlots)
        ? { left: u.metricSlots[0] ?? null, right: u.metricSlots[1] ?? null, mode: u.metricMode ?? 'metrics' }
        : null,
      selection: lastSelection,
      runId: u.activeRunId,
      sim: null,
      case: casePath ? { path: casePath, name: null, dirty: null } : null,
      tabs: u.tabs.map((t) => ({ id: t.id, kind: t.kind, label: tabStateLabel(t) })),
      run: null,
      viewer: {
        datasetId: vs.datasetId,
        field: vs.field,
        time: vs.time?.value ?? null,
        colormap: vs.colormap,
        range: vs.range,
        representation: vs.representation,
        layers: Array.isArray(vs.layers) ? vs.layers.map((l) => ({ id: l.id, type: l.type, summary: l.summary })) : null,
        viewport: (() => {
          const r = v.viewport()
          return r ? { width: r.width, height: r.height } : null
        })(),
      },
      // The target geometry tab's live studio state; a tab whose store entry
      // has not landed yet still reports its path so gui_state is never blind.
      geometry: (() => {
        const geoPath = targetGeometryPath()
        if (geoPath == null) return null
        const entry = useGeometryStore.getState().byPath[geoPath]
        return entry
          ? uiGeometryState(entry)
          : { id: null, path: geoPath, triangleCount: null, closed: null, openEdges: null, solids: [], selected: null, edits: 0, dirty: false }
      })(),
      problems: Object.values(s.problems).reduce((n, list) => n + list.length, 0),
      connection: s.connection === 'online' ? 'connected' : s.connection,
      locale: u.locale,
      compareRunId: (() => { const r = u.tabs.find((t) => t.kind === 'residuals'); return r && r.kind === 'residuals' ? (r.compareRunId ?? null) : null })(),
      views: u.viewSplit
        ? {
            split: true,
            focused: u.focusedView,
            camerasLinked: u.camerasLinked,
            datasetA: getViewerApi('A').getState().datasetId,
            datasetB: getViewerApi('B').getState().datasetId,
          }
        : null,
    }
  }

  function reportState(): void {
    send({ t: 'ui.state', state: snapshot() })
  }

  /** The four geometry edit commands, applied through the store keyed by the
   *  target tab's path. Thrown errors — the store's refusals, the save
   *  route's 409/400 text — propagate to handleCommand's catch, which answers
   *  ok: false with err.message unchanged. */
  async function geometryCommand(cmd: GeometryUiCommand): Promise<{ ok: true } | { ok: false; error: string }> {
    const path = targetGeometryPath()
    if (path == null) return { ok: false, error: 'no geometry tab is open; geometry_open or geometry_import_step first' }
    const g = useGeometryStore.getState()
    const entry = g.byPath[path]
    if (!entry || entry.status === 'loading') return { ok: false, error: `geometry "${path}" is still loading; try again` }
    if (entry.status === 'error') return { ok: false, error: `geometry "${path}" failed to open: ${entry.error}` }
    switch (cmd.type) {
      case 'geometry_part':
        return partResult(g.setPart(path, cmd.name, cmd.action, cmd.newName ?? null))
      case 'geometry_transform': {
        const op = cmd.op
        if (op === 'undo') return g.undo(path) ? { ok: true } : { ok: false, error: 'nothing to undo' }
        if (op === 'redo') return g.redo(path) ? { ok: true } : { ok: false, error: 'nothing to redo' }
        if (op === 'reset') {
          g.reset(path)
          return { ok: true }
        }
        if (op === 'translate') {
          if (!asVec3(cmd.value)) return { ok: false, error: 'translate needs value [x,y,z] in metres' }
          g.pushEdit(path, { kind: 'translate', t: cmd.value })
        } else if (op === 'rotate') {
          if (!asVec3(cmd.value)) return { ok: false, error: 'rotate needs value [ax,ay,az] in degrees' }
          g.pushEdit(path, { kind: 'rotate', deg: cmd.value, pivot: cmd.pivot ?? 'centre' })
        } else if (op === 'scale') {
          const s = asScale(cmd.value)
          if (!s) return { ok: false, error: 'scale needs one non-zero factor or [sx,sy,sz]' }
          g.pushEdit(path, { kind: 'scale', s, pivot: cmd.pivot ?? 'centre' })
        } else {
          if (!asVec3(cmd.value) || (cmd.value[0] === 0 && cmd.value[1] === 0 && cmd.value[2] === 0)) {
            return { ok: false, error: 'mirror needs a non-zero plane normal [nx,ny,nz]' }
          }
          g.pushEdit(path, { kind: 'mirror', normal: cmd.value, pivot: cmd.pivot ?? 'centre' })
        }
        return { ok: true }
      }
      case 'geometry_boolean': {
        const cur = g.byPath[path]
        if (!cur || cur.info == null || cur.info.source?.kind !== 'step') {
          return unsupported('geometry_boolean', 'booleans run only on a STEP geometry; this tab opened ' + path + ' - geometry_import_step a .step first')
        }
        const { out } = await g.boolean(path, cmd.op, cmd.a, cmd.b)
        ui.getState().openGeometryTab(out)
        return { ok: true }
      }
      case 'geometry_save': {
        const reply = await g.save(path, {
          path: cmd.path,
          binary: cmd.binary ?? null,
          keepVisibleOnly: cmd.keepVisibleOnly === true,
          overwrite: cmd.overwrite === true,
        })
        ui.getState().openGeometryTab(reply.info.path)
        if (reply.asciiForced) session.getState().addNote('info', 'saved as ASCII STL so the part names survive')
        return { ok: true }
      }
    }
  }

  /** The first open tab addressed by `x`: an exact id wins, then any tab whose
   *  ui.state label or on-screen label (either locale, case-insensitive)
   *  equals it. select_tab and close_tab both resolve this way. */
  function findTab(u: UiStoreState, x: string) {
    const n = x.toLowerCase()
    return (
      u.tabs.find((t) => t.id === x) ??
      u.tabs.find((t) => tabStateLabel(t).toLowerCase() === n || tabScreenLabel(t, 'en').toLowerCase() === n || tabScreenLabel(t, 'ko').toLowerCase() === n) ??
      null
    )
  }

  /** The run the chart commands act on: the residuals tab's run when that tab
   *  holds one, else the followed run. */
  function chartRunId(): string | null {
    const u = ui.getState()
    const tab = u.tabs.find((t) => t.kind === 'residuals')
    return tab?.kind === 'residuals' && tab.runId ? tab.runId : u.activeRunId
  }

  /** The metric records a run reported; the injected session may not carry runData at all. */
  function runMetrics(runId: string): MetricRecord[] {
    return session.getState().runData?.[runId]?.metrics ?? []
  }

  /** The tab label a tree step or chart name matches against, best effort. */
  function openNamedTab(name: string): { ok: true } | { ok: false; error: string } {
    const n = name.toLowerCase()
    const u = ui.getState()
    if (/view|3d|뷰어/.test(n)) {
      u.openViewerTab()
      return { ok: true }
    }
    if (/residual|잔차|chart|그래프|metric/.test(n)) {
      u.openResidualsTab(u.activeRunId)
      return { ok: true }
    }
    if (/log|터미널|로그|terminal/.test(n)) {
      u.setBottomVisible(true)
      u.setBottomTab('logs')
      return { ok: true }
    }
    if (/problem|문제/.test(n)) {
      u.setBottomVisible(true)
      u.setBottomTab('problems')
      return { ok: true }
    }
    const ids = u.tabs.map((t) => t.id).join(', ') || 'none'
    return { ok: false, error: `NO_TAB (select_tab): no tab "${name}"; open tabs: ${ids}; kinds that open on select: viewer, residuals, logs, problems` }
  }

  /** Window pixel for a probe command: at:"center" and fx,fy need the canvas rect; x,y are window pixels already. */
  function probeTarget(cmd: Extract<UiCommand, { type: 'probe' }>, rect: ViewportRect): { x: number; y: number } | { error: string } {
    if (cmd.at === 'center') return { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 }
    if (cmd.fx != null && cmd.fy != null) {
      if (cmd.fx < 0 || cmd.fx > 1 || cmd.fy < 0 || cmd.fy > 1) return { error: `INVALID (probe): fx,fy must lie in 0..1 (got ${cmd.fx}, ${cmd.fy})` }
      return { x: rect.left + cmd.fx * rect.width, y: rect.top + cmd.fy * rect.height }
    }
    if (cmd.x != null && cmd.y != null) return { x: cmd.x, y: cmd.y }
    return { error: 'INVALID (probe): give x,y (window px) or fx,fy (0..1 across the canvas) or point [x,y,z] or at:"center"' }
  }

  /** A probe outcome becomes the answer and the selection ui.state reports. */
  function probeAnswer(outcome: ProbeOutcome): { ok: true } | { ok: false; error: string } {
    if (outcome.ok) {
      const h = outcome.hit
      lastSelection = { kind: 'cell', id: h.cell, center: h.center, value: h.value, field: h.field, patch: null }
      return { ok: true }
    }
    if (outcome.code === 'MISS') {
      lastSelection = { kind: 'none' }
      return { ok: false, error: `MISS (probe): ${outcome.message}` }
    }
    if (outcome.code === 'NO_STRUCTURED_GRID') return { ok: false, error: `UNSUPPORTED (probe): ${outcome.message}` }
    if (outcome.code === 'NO_DATASET') return { ok: false, error: `NO_DATASET (probe): ${outcome.message}` }
    return { ok: false, error: `NO_VIEWER (probe): ${outcome.message}` }
  }

  async function apply(cmd: UiCommand): Promise<{ ok: true } | { ok: false; error: string }> {
    const u = ui.getState()
    switch (cmd.type) {
      case 'notify':
        session.getState().addNote(cmd.level, cmd.text)
        return { ok: true }
      case 'open_tab': {
        // First test that matches wins; every tab on this screen is reachable.
        const k = cmd.kind.trim().toLowerCase()
        if (k.includes('view') || k.includes('3d')) u.openViewerTab()
        else if (k.includes('metric')) u.openResidualsTab(u.activeRunId, 'metrics')
        else if (k.includes('residual')) u.openResidualsTab(u.activeRunId, 'residuals')
        else if (k.includes('chart')) u.openResidualsTab(u.activeRunId)
        else if (k.includes('log') || k.includes('terminal')) {
          u.setBottomVisible(true)
          u.setBottomTab('logs')
        } else if (k.includes('problem')) {
          u.setBottomVisible(true)
          u.setBottomTab('problems')
        } else if (k.includes('geometry')) {
          const p = cmd.path ?? cmd.label
          const open = u.tabs.find((t) => t.kind === 'geometry')
          if (p) u.openGeometryTab(p)
          else if (open) u.activateTab(open.id)
          else return { ok: false, error: 'INVALID (open_tab): kind "geometry" needs path (a workspace .stl, .obj or .step file)' }
        } else if (k.includes('campaign')) {
          const p = cmd.path ?? cmd.label
          const open = u.tabs.find((t) => t.kind === 'campaign')
          if (p) u.openCampaignTab(p)
          else if (open) u.activateTab(open.id)
          else return { ok: false, error: 'INVALID (open_tab): kind "campaign" needs path (a campaign directory)' }
        } else
          return {
            ok: false,
            error: `INVALID (open_tab): tab kind "${cmd.kind}" does not exist on this screen; kinds: viewer, residuals, metrics, logs, problems, geometry, campaign`,
          }
        return { ok: true }
      }
      case 'select_tab': {
        const found = findTab(u, cmd.tab.trim())
        if (found) {
          u.activateTab(found.id)
          return { ok: true }
        }
        // Not open: the heuristics still open the named tab they always opened.
        return openNamedTab(cmd.tab.trim())
      }
      case 'close_tab': {
        const found = findTab(u, cmd.id.trim())
        if (!found) {
          const ids = u.tabs.map((t) => t.id).join(', ') || 'none'
          return { ok: false, error: `NO_TAB (close_tab): no tab "${cmd.id}"; open tabs: ${ids}` }
        }
        if (found.kind === 'file' && deps.editor?.isDirty(found.path))
          return { ok: false, error: `DIRTY (close_tab): "${found.path}" has unsaved edits; save it first or close it on screen` }
        u.closeTab(found.id)
        if (found.kind === 'file') deps.editor?.closeBuffer(found.path)
        else if (found.kind === 'diff') deps.editor?.removeDiff(found.id)
        return { ok: true }
      }
      case 'open_case':
        u.openFile(cmd.path)
        return { ok: true }
      case 'save_case': {
        // The open editor buffer of the screen's case, validated before it is written (STU-05).
        const path = activeCasePath()
        if (path === null) return { ok: false, error: 'NO_CASE (save_case): no case is open in the editor; open_case first' }
        if (!deps.editor?.saveCase) return { ok: false, error: 'NO_EDITOR (save_case): this window has no case editor to save from' }
        const showProblems = () => {
          u.setBottomVisible(true)
          u.setBottomTab('problems')
        }
        const r = await deps.editor.saveCase(path, cmd.force === true)
        if (r.ok) {
          if (r.errors > 0) showProblems()
          return { ok: true }
        }
        if (r.code === 'CASE_INVALID') {
          showProblems()
          return {
            ok: false,
            error: `CASE_INVALID (save_case): ${r.message}; nothing was written. The findings are in the Problems list: fix them, or send save_case {force: true} to save anyway`,
          }
        }
        return { ok: false, error: `${r.code} (save_case): ${r.message}` }
      }
      case 'new_case':
      case 'validate_case':
        return unsupported(cmd.type, 'gui_control applies this on the server through case_create / case_validate; this window should not receive it')
      case 'start_mesh':
        return unsupported(cmd.type, 'gui_control applies this on the server through mesh_generate / mesh_regions; this window should not receive it')
      case 'set_patch':
        return unsupported(cmd.type, 'gui_control applies this on the server through case_edit; this window should not receive it')
      case 'open_mesh_dialog':
        // The server keeps the draft (guiMesh.ts); the dialog shows what it sent.
        u.showMeshDialog(meshDraftOf(cmd, u.meshDialog?.draft ?? null))
        return { ok: true }
      case 'open_mesh_view': {
        // The mesh the dialog's draft made, else the open case's own mesh.
        const path = meshOutputOf(u.meshDialog?.draft ?? null) ?? activeCasePath()
        if (path === null)
          return { ok: false, error: 'NO_MESH (open_mesh_view): no mesh to open: the Mesh dialog shows no draft with an output directory and no case is open; open_mesh_dialog or open_case first' }
        u.openViewerTab()
        const v = getViewerApi()
        const loaded = await v.execute({ type: 'load', path, timeIndex: 'last', field: 'none', region: null })
        if (!loaded.ok) return { ok: false, error: loaded.error?.message ?? `the mesh at ${path} could not be opened` }
        const r = await v.execute({ type: 'setRepresentation', mode: cmd.representation ?? 'surfaceEdges', opacity: null, patches: cmd.patches ?? null })
        return r.ok ? { ok: true } : { ok: false, error: r.error?.message ?? 'the viewer refused the representation' }
      }
      case 'set_locale':
        u.setLocale(cmd.locale)
        actions.setSettings({ locale: cmd.locale })
        return { ok: true }
      case 'set_setting': {
        const patch: Record<string, unknown> = {}
        if (cmd.autoApprove != null) patch.autoApprove = cmd.autoApprove
        if (cmd.effort != null) patch.effort = cmd.effort
        if (cmd.notifyOnRunEnd != null) patch.notifyOnRunEnd = cmd.notifyOnRunEnd
        if (cmd.locale != null) patch.locale = cmd.locale
        if (!Object.keys(patch).length) return unsupported('set_setting', 'no setting in the command')
        if (patch.locale != null) ui.getState().setLocale(patch.locale as 'ko' | 'en')
        if (!actions.setSettings(patch)) return unsupported('set_setting', 'no session is open')
        return { ok: true }
      }
      case 'open_session':
        if (!actions.openSession(cmd.sessionId)) return unsupported('open_session', 'the socket is not connected')
        return { ok: true }
      case 'follow_run':
        ui.getState().setActiveRun(cmd.runId)
        ui.getState().openResidualsTab(cmd.runId)
        deps.subscribeRun(cmd.runId)
        return { ok: true }
      case 'set_post': {
        const v = getViewerApi()
        const vs = v.getState()
        const fieldish = cmd.colormap != null || cmd.range != null || cmd.component != null || cmd.log != null
        const repish = cmd.representation != null || cmd.opacity != null || cmd.patches != null
        if (!fieldish && !repish)
          return { ok: false, error: 'INVALID (set_post): nothing to change; give colormap, range, component, representation, opacity, patches or log' }
        if (fieldish) {
          if (vs.field == null) return { ok: false, error: 'NO_FIELD (set_post): no field is coloured; open_result or post_field first' }
          const r = await v.execute({ type: 'setField', field: vs.field, component: cmd.component ?? null, range: cmd.range ?? null, colormap: cmd.colormap ?? null, log: cmd.log ?? null })
          if (!r.ok) return { ok: false, error: r.error?.message ?? 'the viewer refused set_post' }
        }
        if (repish) {
          const r = await v.execute({ type: 'setRepresentation', mode: cmd.representation ?? vs.representation, opacity: cmd.opacity ?? null, patches: cmd.patches ?? null })
          if (!r.ok) return { ok: false, error: r.error?.message ?? 'the viewer refused set_post' }
        }
        ui.getState().openViewerTab()
        return { ok: true }
      }
      case 'show_metric': {
        const slot = cmd.slot ?? 1
        const metric = cmd.metric ?? null
        const trimmed = metric?.trim() ?? ''
        if (!trimmed && cmd.mode == null) return { ok: false, error: 'INVALID (show_metric): needs a metric or a mode' }
        if (slot !== 1 && slot !== 2) return { ok: false, error: `INVALID (show_metric): slot must be 1 (left) or 2 (right), got ${slot}` }
        if (trimmed && cmd.mode === 'sweeps')
          return { ok: false, error: 'INVALID (show_metric): a metric pick applies to mode "metrics"; drop the metric or send mode "metrics"' }
        const run = chartRunId()
        let found: string | null = null
        if (trimmed) {
          if (run == null) return { ok: false, error: 'NO_RUN (show_metric): no run is followed; follow_run first' }
          const names = metricNames(runMetrics(run))
          found = findMetric(names, metric!)
          if (found == null)
            return { ok: false, error: `NO_METRIC (show_metric): run "${run}" reports no metric "${metric}"; reported: ${names.join(', ') || 'none'}` }
        }
        if (cmd.mode != null) u.setMetricMode(cmd.mode)
        if (found != null) {
          if (cmd.mode == null) u.setMetricMode('metrics')
          u.setMetricSlot(slot as 1 | 2, found)
        }
        u.openResidualsTab(run, 'metrics')
        return { ok: true }
      }
      case 'set_log_filter': {
        if (cmd.text === undefined && cmd.streams == null && cmd.follow == null)
          return { ok: false, error: 'INVALID (set_log_filter): nothing to change; give text, streams or follow' }
        if (cmd.streams != null && cmd.streams.length === 0)
          return { ok: false, error: 'INVALID (set_log_filter): streams must name at least one of stdout, stderr, system' }
        const patch: { text?: string; streams?: LogStream[]; follow?: boolean } = {}
        if (cmd.text !== undefined) patch.text = cmd.text ?? ''
        if (cmd.streams != null) patch.streams = cmd.streams
        if (cmd.follow != null) patch.follow = cmd.follow
        u.setLogFilter(patch)
        u.setBottomVisible(true)
        u.setBottomTab('logs')
        return { ok: true }
      }
      case 'show_chart':
        if (cmd.chart === 'residuals' || cmd.chart === 'metrics') {
          if (cmd.runId) deps.subscribeRun(cmd.runId)
          u.openResidualsTab(cmd.runId ?? u.activeRunId, cmd.chart)
          return { ok: true }
        }
        // surface: the run's per-patch surface-integral series (<patch>.<quantity> metrics).
        {
          const run = cmd.runId ?? chartRunId()
          if (run == null) return { ok: false, error: 'NO_SURFACE_METRICS (show_chart): no run is followed; follow_run or pass runId' }
          if (cmd.runId) deps.subscribeRun(cmd.runId)
          const recs = runMetrics(run)
          if (!surfaceSeriesKeys(recs).length) {
            const names = metricNames(recs)
            return {
              ok: false,
              error: `NO_SURFACE_METRICS (show_chart): run "${run}" reports no per-patch surface-integral series (<patch>.<quantity>, e.g. fan.Q); its metrics: ${names.join(', ') || 'none'}`,
            }
          }
          u.openResidualsTab(run, 'surface')
          return { ok: true }
        }
      case 'add_layer': {
        // Loose kind + args -> one schema-valid viewer command, then the same
        // race set_camera runs: the answer must beat the hub's 5 s.
        const k = layerKind(cmd.kind)
        if (k == null)
          return { ok: false, error: `INVALID (add_layer): unknown kind "${cmd.kind}"; kinds: ${LAYER_KINDS.join(', ')}` }
        const vs = getViewerApi().getState()
        const built = layerCommand(k, cmd.args ?? {}, { field: vs.field ?? null, bounds: vs.bounds ?? null })
        if (!built.ok) return built
        ui.getState().openViewerTab()
        const v = getViewerApi()
        const budget = deps.cameraBudgetMs ?? CAMERA_ANSWER_MS
        let raceTimer: ReturnType<typeof setTimeout> | null = null
        const done = await Promise.race([
          v.execute(built.cmd),
          new Promise<null>((resolve) => {
            raceTimer = setTimeout(() => resolve(null), budget)
          }),
        ])
        if (raceTimer) clearTimeout(raceTimer)
        if (done === null)
          return { ok: false, error: `BUSY (add_layer): the viewer did not answer within ${budget} ms; the layer is queued and applies when the viewer is ready` }
        return done.ok ? { ok: true } : { ok: false, error: done.error ? `${done.error.code} (add_layer): ${done.error.message}` : 'the viewer refused add_layer' }
      }
      case 'remove_layer': {
        const v = getViewerApi()
        const ids = (v.getState().layers ?? []).map((l) => l.id)
        if (!ids.includes(cmd.id))
          return { ok: false, error: `NO_SUCH_LAYER (remove_layer): no layer "${cmd.id}"; layers: ${ids.join(', ') || 'none'}` }
        const budget = deps.cameraBudgetMs ?? CAMERA_ANSWER_MS
        let raceTimer: ReturnType<typeof setTimeout> | null = null
        const done = await Promise.race([
          v.execute({ type: 'remove', id: cmd.id }),
          new Promise<null>((resolve) => {
            raceTimer = setTimeout(() => resolve(null), budget)
          }),
        ])
        if (raceTimer) clearTimeout(raceTimer)
        if (done === null)
          return { ok: false, error: `BUSY (remove_layer): the viewer did not answer within ${budget} ms; the removal is queued` }
        return done.ok ? { ok: true } : { ok: false, error: done.error ? `${done.error.code} (remove_layer): ${done.error.message}` : 'the viewer refused remove_layer' }
      }
      case 'set_camera':
      case 'fit_view': {
        // A camera move must be answered inside the hub's 5 s: open the tab
        // first so the canvas can mount, wait out the mount, then race the
        // move against what is left of the budget.
        const budget = deps.cameraBudgetMs ?? CAMERA_ANSWER_MS
        const preset = cmd.type === 'fit_view' ? 'fit' : cmd.preset
        ui.getState().openViewerTab()
        const v = getViewerApi()
        const deadline = Date.now() + budget
        while (!v.isMounted() && Date.now() < deadline) {
          await new Promise((r) => setTimeout(r, Math.min(50, Math.max(1, deadline - Date.now()))))
        }
        if (!v.isMounted()) return noViewer(cmd.type)
        let raceTimer: ReturnType<typeof setTimeout> | null = null
        const moved = await Promise.race([
          v.execute({ type: 'setCamera', preset }),
          new Promise<null>((resolve) => {
            raceTimer = setTimeout(() => resolve(null), Math.max(0, deadline - Date.now()))
          }),
        ])
        if (raceTimer) clearTimeout(raceTimer)
        if (moved === null)
          return { ok: false, error: `BUSY (${cmd.type}): the viewer is still working on an earlier command; the camera move is queued and applies when it finishes` }
        return moved.ok ? { ok: true } : { ok: false, error: moved.error?.message ?? 'the viewer refused the camera move' }
      }
      case 'show_field': {
        const mapped = FIELD_MAP[cmd.field]
        if (!mapped) return unsupported(cmd.type, `field "${cmd.field}" is not shown here`)
        const r = await getViewerApi().execute({ type: 'setField', field: mapped.name, component: mapped.component, range: null })
        ui.getState().openViewerTab()
        return r.ok ? { ok: true } : { ok: false, error: r.error?.message ?? 'no loaded result carries that field' }
      }
      case 'post_field': {
        const r = await getViewerApi().execute({ type: 'setField', field: cmd.field, component: cmd.component ?? null, range: cmd.range ?? null })
        ui.getState().openViewerTab()
        return r.ok ? { ok: true } : { ok: false, error: r.error?.message ?? 'no loaded result carries that field' }
      }
      case 'post_time': {
        const r = await getViewerApi().execute({ type: 'setTime', index: cmd.index })
        return r.ok ? { ok: true } : { ok: false, error: r.error?.message ?? 'no loaded result has that time' }
      }
      case 'post_representation': {
        const r = await getViewerApi().execute({ type: 'setRepresentation', mode: cmd.mode, opacity: cmd.opacity ?? null, patches: cmd.patches ?? null })
        return r.ok ? { ok: true } : { ok: false, error: r.error?.message ?? 'the viewer refused the representation' }
      }
      case 'post_warp': {
        const r = await getViewerApi().execute({ type: 'setWarp', field: cmd.field ?? null, scale: cmd.scale ?? (cmd.field == null ? 0 : 1) })
        return r.ok ? { ok: true } : { ok: false, error: r.error?.message ?? 'the viewer refused the warp' }
      }
      case 'open_result': {
        ui.getState().openViewerTab()
        // The viewer command carries the path itself; a failing load is the honest answer.
        const r = await getViewerApi().execute({ type: 'load', path: cmd.path, timeIndex: cmd.timeIndex ?? 'last', field: null, region: cmd.region ?? null })
        return r.ok ? { ok: true } : { ok: false, error: r.error?.message ?? 'the result could not be opened' }
      }
      case 'geometry_open':
      case 'geometry_import_step':
        // One tab serves both: it routes by extension (STEP goes through the
        // server's import-step converter, STL/OBJ open directly) and reports
        // its own loading/error state on screen.
        ui.getState().openGeometryTab(cmd.path)
        return { ok: true }
      case 'geometry_part':
      case 'geometry_transform':
      case 'geometry_boolean':
      case 'geometry_save':
        // The studio's own edits (E1): real edits through the geometry store.
        return geometryCommand(cmd)
      case 'open_panel':
        if (cmd.panel === 'AI Assistant') {
          if (!ui.getState().assistantVisible) ui.getState().toggleAssistant()
          return { ok: true }
        }
        return unsupported('open_panel', `panel "${cmd.panel}" does not exist on this screen (AI Assistant only)`)
      case 'run':
      case 'start_run':
      case 'stop_run':
      case 'set_run_setting':
        return unsupported(cmd.type, 'gui_control applies this on the server through run_start / run_stop; this window should not receive it')
      case 'set_tool':
        // The controller replays a tool set before the canvas mounts (attachView), so no mount
        // gate here; it does NOT create the section clip box - Viewer3D's toolbar onTool does
        // that from the screen only.
        getViewerApi().setTool(TOOL_TO_VIEWER[cmd.tool])
        ui.getState().openViewerTab()
        return { ok: true }
      case 'set_projection': {
        const v = getViewerApi()
        if (!v.isMounted()) {
          ui.getState().openViewerTab()
          return noViewer(cmd.type)
        }
        const r = await v.execute({
          type: 'setCamera',
          preset: null,
          position: null,
          target: null,
          projection: cmd.projection === 'Orthographic' ? 'orthographic' : 'perspective',
        })
        ui.getState().openViewerTab()
        return r.ok ? { ok: true } : { ok: false, error: r.error?.message ?? 'the viewer refused the projection' }
      }
      case 'probe': {
        const v = getViewerApi()
        const rect = v.viewport()
        if (!v.isMounted() || !rect || rect.width <= 0 || rect.height <= 0) {
          ui.getState().openViewerTab()
          return noViewer(cmd.type)
        }
        if (cmd.point != null) return probeAnswer(v.probePoint(cmd.point))
        const t = probeTarget(cmd, rect)
        if ('error' in t) return { ok: false, error: t.error }
        return probeAnswer(v.probePixel(t.x, t.y))
      }
      case 'post_screenshot': {
        const v = getViewerApi()
        if (!v.isMounted()) {
          ui.getState().openViewerTab()
          return noViewer(cmd.type)
        }
        const r = await v.execute({ type: 'screenshot', width: null, height: null, includeLegend: u.showColorBars !== false })
        ui.getState().openViewerTab()
        if (!r.ok) return { ok: false, error: r.error?.message ?? 'the viewer refused the screenshot' }
        if (!r.image) return { ok: false, error: 'NO_IMAGE (post_screenshot): the viewer returned no image' }
        ;(deps.saveImage ?? defaultSaveImage)(`${v.getState().datasetName ?? 'viewer'}.png`, r.image.base64)
        return { ok: true }
      }
      case 'compare_run': {
        const known = session.getState().runs
        if (cmd.runId !== null && !known[cmd.runId])
          return unsupported('compare_run', `no run "${cmd.runId}" on this screen; known: ${Object.keys(known).slice(0, 8).join(', ') || 'none'}`)
        const resTab = u.tabs.find((t) => t.kind === 'residuals')
        const primary = resTab?.kind === 'residuals' ? resTab.runId : u.activeRunId
        if (cmd.runId !== null && cmd.runId === primary)
          return unsupported('compare_run', `run "${cmd.runId}" is the run the chart already follows; pick another or follow_run first`)
        u.setCompareRun(cmd.runId)
        if (cmd.runId) deps.subscribeRun(cmd.runId)
        return { ok: true }
      }
      case 'split_view':
        if (cmd.on) {
          u.openViewerTab()
          u.setViewSplit(true)
        } else {
          dropLink()
          u.setViewSplit(false)
        }
        return { ok: true }
      case 'focus_view': {
        if (!u.viewSplit) return unsupported('focus_view', 'the viewport is not split; send split_view true first')
        u.setFocusedView(cmd.view)
        if (u.camerasLinked) relink(cmd.view)
        return { ok: true }
      }
      case 'open_result_in_view': {
        if (cmd.view === 'B' && !u.viewSplit) u.setViewSplit(true)
        u.openViewerTab()
        const r = await getViewerApi(cmd.view).execute({ type: 'load', path: cmd.path, timeIndex: cmd.timeIndex ?? 'last', field: null })
        return r.ok ? { ok: true } : { ok: false, error: r.error?.message ?? 'the result could not be opened' }
      }
      case 'link_cameras':
        if (!u.viewSplit) return unsupported('link_cameras', 'the viewport is not split; send split_view true first')
        if (cmd.on) {
          relink(u.focusedView)
          u.setCamerasLinked(true)
        } else {
          dropLink()
          u.setCamerasLinked(false)
        }
        return { ok: true }
      case 'show_overlay':
        u.setOverlay(cmd.what, cmd.on)
        return { ok: true }
      case 'set_centerline':
      case 'select_step':
        return unsupported(cmd.type, 'this screen does not implement it')
      default:
        return unsupported(cmd.type, 'this screen does not implement it')
    }
  }

  return {
    async handleCommand(requestId: string, cmd: UiCommand): Promise<void> {
      let ok = true
      let error: string | null = null
      try {
        const r = await apply(cmd)
        ok = r.ok
        if (!r.ok) error = r.error
      } catch (err) {
        ok = false
        error = err instanceof Error ? err.message : String(err)
      }
      reportState()
      send({ t: 'ui.result', requestId, ok, error, state: snapshot() })
    },
    reportState,
  }
}
