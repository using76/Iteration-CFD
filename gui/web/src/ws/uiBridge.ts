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

const GEOMETRY_EDIT_TYPES = new Set(['geometry_part', 'geometry_transform', 'geometry_boolean', 'geometry_save'])
const NO_GEOMETRY_EDITS = 'editing geometry on screen (part visibility, transforms, booleans, save) is not implemented in the web studio yet; geometry_open / geometry_import_step do work'

function unsupported(type: string, why: string): { ok: false; error: string } {
  return { ok: false, error: `UNSUPPORTED (${type}): ${why}` }
}

export function createUiBridge(deps: UiBridgeDeps) {
  const { send, ui, session } = deps

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
      // The open geometry tab, if any: the model reads this to know car.step
      // (or whatever) is on screen without driving the screen blind.
      geometry: (() => {
        const geoTab = u.tabs.find((t) => t.kind === 'geometry')
        return geoTab && geoTab.kind === 'geometry'
          ? { id: null, path: geoTab.path, triangleCount: null, closed: null, openEdges: null, solids: [], selected: null, edits: 0, dirty: false }
          : null
      })(),
      problems: Object.values(s.problems).reduce((n, list) => n + list.length, 0),
      connection: s.connection === 'online' ? 'connected' : s.connection,
      locale: u.locale,
    }
  }

  function reportState(): void {
    send({ t: 'ui.state', state: snapshot() })
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
        if (GEOMETRY_EDIT_TYPES.has(cmd.type)) return unsupported(cmd.type, NO_GEOMETRY_EDITS)
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
