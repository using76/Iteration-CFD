// Glue between the WS protocol and the studio screen for the ui.command /
// ui.result / ui.state trio: the agent's gui_control tool drives this client
// through the hub, and every command MUST be answered — an unanswered one
// parks the model for the hub's full 5 s timeout and it retries into the
// same void. Commands this web app can genuinely apply are applied here;
// everything else is refused immediately with UNSUPPORTED so the model can
// tell the operator the truth instead of waiting.
import type { ClientMsg, UiCommand, UiState } from '@cfd/shared'
import { getViewerApi } from '../viewer'
import { actions, activeCasePath } from './actions'
import { uiGeometryState, useGeometryStore } from '../state/geometryStore'
import type { useSessionStore } from '../state/sessionStore'
import type { useUiStore } from '../state/uiStore'

type SessionStore = ReturnType<typeof useSessionStore.getState>
type UiStoreState = ReturnType<typeof useUiStore.getState>

export interface UiBridgeDeps {
  send(msg: ClientMsg): boolean
  ui: { getState(): UiStoreState }
  session: { getState(): SessionStore }
  /** Client.ts owns run subscriptions; a followed run is one of them. */
  subscribeRun(runId: string): void
}

/** show_field / post_field name the three display fields; the solver names them U/p/T. */
const FIELD_MAP: Record<string, { name: string; component: 'magnitude' | null }> = {
  Temperature: { name: 'T', component: null },
  Velocity: { name: 'U', component: 'magnitude' },
  Pressure: { name: 'p', component: null },
}

/** The four geometry edit ui.commands this screen applies through the
 *  geometry studio store (E1). */
type GeometryUiCommand = Extract<UiCommand, { type: 'geometry_part' | 'geometry_transform' | 'geometry_boolean' | 'geometry_save' }>

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

export function createUiBridge(deps: UiBridgeDeps) {
  const { send, ui, session } = deps

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
    return {
      activeTab: u.activeTabId,
      activeStep: null,
      rightTab: u.assistantVisible ? 'AI Assistant' : null,
      tool: null,
      frame: null,
      projection: null,
      showAxes: null,
      showColorBars: null,
      selection: null,
      runId: u.activeRunId,
      sim: null,
      case: casePath ? { path: casePath, name: null, dirty: null } : null,
      tabs: u.tabs.map((t) => ({ id: t.id, kind: t.kind, label: t.kind === 'file' ? t.path : t.kind })),
      run: null,
      viewer: null,
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
    const known = ['viewer/뷰어', 'residuals/잔차', 'logs/로그', 'problems/문제']
    return unsupported('select_tab', `no tab matching "${name}"; this screen has: ${known.join(', ')}`)
  }

  async function apply(cmd: UiCommand): Promise<{ ok: true } | { ok: false; error: string }> {
    const u = ui.getState()
    switch (cmd.type) {
      case 'notify':
        session.getState().addNote(cmd.level, cmd.text)
        return { ok: true }
      case 'open_tab': {
        const k = cmd.kind.toLowerCase()
        if (k.includes('view')) u.openViewerTab()
        else if (k.includes('residual') || k.includes('chart')) u.openResidualsTab(u.activeRunId)
        else if (k.includes('log') || k.includes('terminal')) {
          u.setBottomVisible(true)
          u.setBottomTab('logs')
        } else return unsupported('open_tab', `tab kind "${cmd.kind}" does not exist on this screen (viewer, residuals, logs)`)
        return { ok: true }
      }
      case 'select_tab':
        return openNamedTab(cmd.tab)
      case 'open_case':
        u.openFile(cmd.path)
        return { ok: true }
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
      case 'show_chart':
        if (cmd.chart === 'residuals') {
          ui.getState().openResidualsTab(ui.getState().activeRunId)
          return { ok: true }
        }
        return unsupported('show_chart', `chart "${cmd.chart}" has no tab here (residuals only)`)
      case 'set_camera':
      case 'fit_view': {
        const preset = cmd.type === 'fit_view' ? 'fit' : cmd.preset
        const r = await getViewerApi().execute({ type: 'setCamera', preset })
        ui.getState().openViewerTab()
        return r.ok ? { ok: true } : { ok: false, error: r.error?.message ?? 'the viewer refused the camera move' }
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
        return unsupported(cmd.type, 'runs are driven by the run_start/run_stop tools on this screen, not by gui_control')
      case 'probe':
      case 'post_screenshot':
      case 'set_tool':
      case 'set_projection':
      case 'show_overlay':
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
