// Layout and preference state, persisted to localStorage (versioned).
import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { Locale, MeshDraft, UiCommand, UiSelection, ViewId } from '@cfd/shared'
import type { ProbeOutcome } from '../viewer/controller/ViewerController'
import { tx } from '../i18n/extra'

export type Theme = 'light' | 'dark'
export type ActivityView = 'explorer' | 'search' | 'scm' | 'run' | 'extensions' | 'cfd'
export type BottomTab = 'terminal' | 'logs' | 'problems' | 'output'
/** What the one residuals tab draws (show_chart's chart; STU-03 adds surface). */
export type ChartKind = 'residuals' | 'metrics' | 'surface'
export type LogStream = 'stdout' | 'stderr' | 'system'
/** The Log tab's filter (set_log_filter's target), kept in the store so the pane and ui.state agree. */
export interface LogFilter {
  text: string
  streams: LogStream[]
  follow: boolean
}
/** The metrics chart's mode: the metric picker or the linear-solver sweep counts (show_metric). */
export type MetricMode = 'metrics' | 'sweeps'

export type Tab =
  | { id: string; kind: 'file'; path: string }
  | { id: 'viewer'; kind: 'viewer' }
  | { id: 'residuals'; kind: 'residuals'; runId: string | null; compareRunId?: string | null; chart?: ChartKind }
  | { id: string; kind: 'diff'; path: string; toolUseId: string | null }
  | { id: string; kind: 'geometry'; path: string }
  | { id: string; kind: 'campaign'; path: string }

export type PanelLayout = Record<string, number>

/** The answer the server gave the Mesh dialog's last mesh.draft / mesh.start. */
export interface MeshStatus {
  action: 'draft' | 'start'
  ok: boolean
  code: string | null
  error: string | null
}

/** The Mesh dialog on screen (STU-06b): the session's draft as the server last sent it, the mesh run Start began, and the last answer. */
export interface MeshDialogState {
  draft: MeshDraft
  runId: string | null
  status: MeshStatus | null
}

/** A right-column panel (STU-08a): the panels open_panel names, in the column's tab order. */
export type RightPanelId = Extract<UiCommand, { type: 'open_panel' }>['panel']
export const RIGHT_PANELS: readonly RightPanelId[] = ['AI Assistant', 'Post', 'Inspector', 'Properties']

/** What the Properties panel shows (STU-08a): the selected tree node and its keys. STU-08b's project tree sets it. */
export interface PropertiesNode {
  id: string
  label: string
  keys: Record<string, unknown>
}

/** The selection a probe outcome leaves (STU-08a): a hit selects its cell, a miss selects nothing, and
 *  an outcome that read nothing (no viewer, no dataset, no structured grid) leaves the selection as it was (null). */
export function selectionOfProbe(o: ProbeOutcome): UiSelection | null {
  if (o.ok) return { kind: 'cell', id: o.hit.cell, center: o.hit.center, value: o.hit.value, field: o.hit.field, patch: null }
  return o.code === 'MISS' ? { kind: 'none' } : null
}

/** The Boundary editor on screen (STU-07b). */
export interface BoundaryEditorState {
  path: string
  rev: number
}

export interface UiState {
  theme: Theme
  locale: Locale
  sideVisible: boolean
  bottomVisible: boolean
  assistantVisible: boolean
  mainLayout: PanelLayout | null
  centerLayout: PanelLayout | null
  bottomLayout: PanelLayout | null
  activity: ActivityView
  tabs: Tab[]
  activeTabId: string | null
  bottomTab: BottomTab
  explorerExpanded: string[]
  activeRunId: string | null
  terminalRunId: string | null
  composerDraft: string
  // transient (not persisted)
  paletteOpen: boolean
  settingsOpen: boolean
  cursor: { line: number; col: number } | null
  composerPrefill: { text: string; nonce: number } | null
  viewSplit: boolean
  focusedView: ViewId
  camerasLinked: boolean
  showAxes: boolean
  showColorBars: boolean
  /** The Log tab's filter (text, stream chips, follow-tail); set_log_filter writes it. */
  logFilter: LogFilter
  /** What the metrics chart draws: slot 1 (left axis) and slot 2 (right axis) metric picks. */
  metricSlots: [string | null, string | null]
  /** metrics or the sweeps view of the same chart. */
  metricMode: MetricMode
  /** The Mesh dialog; null when it is closed. Transient: the draft lives on the server. */
  meshDialog: MeshDialogState | null
  /** The Boundary editor (STU-07b): the case it shows, and a count that makes an open editor read its table again; null when it is closed. Transient. */
  boundaryEditor: BoundaryEditorState | null
  /** The right column's active panel (STU-08a); the column itself shows while assistantVisible. Transient. */
  rightTab: RightPanelId
  /** The last probe or screen pick (STU-08a), what Inspector shows; null before the first. Transient. */
  selection: UiSelection | null
  /** The node Properties shows (STU-08a); null when nothing is selected. Transient. */
  propertiesNode: PropertiesNode | null
}

export interface UiActions {
  setTheme(theme: Theme): void
  toggleTheme(): void
  setLocale(locale: Locale): void
  toggleLocale(): void
  toggleSide(): void
  toggleBottom(): void
  toggleAssistant(): void
  setBottomVisible(v: boolean): void
  /** Toggles the viewer's axes triad or colour bars (show_overlay's target). */
  setOverlay(what: 'axes' | 'colorbars', on: boolean): void
  setLayout(which: 'mainLayout' | 'centerLayout' | 'bottomLayout', layout: PanelLayout): void
  setActivity(view: ActivityView): void
  openTab(tab: Tab): void
  openFile(path: string): void
  openViewerTab(): void
  openGeometryTab(path: string): void
  openCampaignTab(path: string): void
  openResidualsTab(runId: string | null, chart?: ChartKind): void
  /** Merges a set_log_filter patch; streams land de-duplicated in canonical order. */
  setLogFilter(patch: Partial<LogFilter>): void
  /** Puts a metric on slot 1 (left axis) or slot 2 (right axis); null clears the slot. */
  setMetricSlot(slot: 1 | 2, metric: string | null): void
  setMetricMode(mode: MetricMode): void
  setCompareRun(runId: string | null): void
  setViewSplit(on: boolean): void
  setFocusedView(view: ViewId): void
  setCamerasLinked(on: boolean): void
  openDiffTab(id: string, path: string, toolUseId: string | null): void
  closeTab(id: string): void
  closeOtherTabs(id: string): void
  activateTab(id: string): void
  activateTabIndex(index: number): void
  moveTab(from: number, to: number): void
  setBottomTab(tab: BottomTab): void
  setExpanded(path: string, expanded: boolean): void
  collapseAll(): void
  setActiveRun(runId: string | null): void
  setTerminalRun(runId: string | null): void
  setComposerDraft(text: string): void
  setPaletteOpen(open: boolean): void
  setSettingsOpen(open: boolean): void
  setCursor(cursor: { line: number; col: number } | null): void
  prefillComposer(text: string): void
  /** Opens the Mesh dialog on a draft, or shows a new draft in the open one (its run and answer stay). */
  showMeshDialog(draft: MeshDraft): void
  closeMeshDialog(): void
  /** Records the server's mesh.status; a start that began a run points the dialog's log at it. */
  setMeshStatus(status: MeshStatus, runId: string | null): void
  /** Opens the Boundary editor on a case, or has the open one read its table again. */
  showBoundaryEditor(path: string): void
  closeBoundaryEditor(): void
  /** Shows the right column on one of its panels (open_panel's target). */
  openPanel(panel: RightPanelId): void
  setSelection(selection: UiSelection | null): void
  /** Puts a node in the Properties panel (null clears it). */
  selectNode(node: PropertiesNode | null): void
}

export type UiStore = UiState & UiActions

export const UI_STORE_VERSION = 1
const STORAGE_KEY = 'cfd-studio.ui'

const DEFAULT_LOCALE: Locale = 'ko'

const INITIAL: UiState = {
  theme: 'light',
  locale: DEFAULT_LOCALE,
  sideVisible: true,
  bottomVisible: true,
  assistantVisible: true,
  mainLayout: null,
  centerLayout: null,
  bottomLayout: null,
  activity: 'explorer',
  tabs: [],
  activeTabId: null,
  bottomTab: 'terminal',
  explorerExpanded: [],
  activeRunId: null,
  terminalRunId: null,
  composerDraft: '',
  paletteOpen: false,
  settingsOpen: false,
  cursor: null,
  composerPrefill: null,
  viewSplit: false,
  focusedView: 'A',
  camerasLinked: false,
  showAxes: true,
  showColorBars: true,
  logFilter: { text: '', streams: ['stdout', 'stderr', 'system'], follow: true },
  metricSlots: [null, null],
  metricMode: 'metrics',
  meshDialog: null,
  boundaryEditor: null,
  rightTab: 'AI Assistant',
  selection: null,
  propertiesNode: null,
}

export function fileTabId(path: string): string {
  return `file:${path}`
}

/** Same as app/hooks.ts's basename, local so this module stays free of React hooks. */
function basename(path: string): string {
  const i = path.lastIndexOf('/')
  return i >= 0 ? path.slice(i + 1) : path
}

/** The text TabBar shows for a tab (moved verbatim from TabBar's label). */
export function tabScreenLabel(tab: Tab, locale: Locale): string {
  return tab.kind === 'file'
    ? basename(tab.path)
    : tab.kind === 'viewer'
      ? tx(locale, 'tab.viewer')
      : tab.kind === 'residuals'
        ? tab.chart === 'metrics'
          ? tx(locale, 'residuals.metricsTitle')
          : tab.chart === 'surface'
            ? tx(locale, 'residuals.surfaceTitle')
            : tx(locale, 'tab.residuals')
        : tab.kind === 'geometry'
          ? `${tx(locale, 'tab.geometry')}: ${basename(tab.path)}`
          : tab.kind === 'campaign'
            ? `${tx(locale, 'autonomy.tab')}: ${basename(tab.path) || '.'}`
            : `${tx(locale, 'tab.diff')}: ${basename(tab.path)}`
}

/** The label ui.state reports for a tab: its path, its chart, or 'viewer'. */
export function tabStateLabel(tab: Tab): string {
  switch (tab.kind) {
    case 'file':
    case 'geometry':
    case 'campaign':
    case 'diff':
      return tab.path
    case 'residuals':
      return tab.chart ?? 'residuals'
    case 'viewer':
      return 'viewer'
  }
}

function upsertTab(tabs: Tab[], tab: Tab): Tab[] {
  const i = tabs.findIndex((t) => t.id === tab.id)
  if (i < 0) return [...tabs, tab]
  const out = tabs.slice()
  out[i] = tab
  return out
}

type PersistedUi = Pick<UiState, 'theme' | 'locale' | 'sideVisible' | 'bottomVisible' | 'assistantVisible' | 'mainLayout' | 'centerLayout' | 'bottomLayout' | 'activity' | 'tabs' | 'activeTabId' | 'bottomTab' | 'explorerExpanded' | 'activeRunId' | 'terminalRunId' | 'composerDraft' | 'showAxes' | 'showColorBars'>

function partialize(s: UiState): PersistedUi {
  return {
    theme: s.theme,
    locale: s.locale,
    sideVisible: s.sideVisible,
    bottomVisible: s.bottomVisible,
    assistantVisible: s.assistantVisible,
    mainLayout: s.mainLayout,
    centerLayout: s.centerLayout,
    bottomLayout: s.bottomLayout,
    activity: s.activity,
    // diff tabs hold in-memory payloads and cannot survive a reload
    tabs: s.tabs.filter((t) => t.kind !== 'diff'),
    activeTabId: s.activeTabId,
    bottomTab: s.bottomTab,
    explorerExpanded: s.explorerExpanded,
    activeRunId: s.activeRunId,
    terminalRunId: s.terminalRunId,
    composerDraft: s.composerDraft,
    showAxes: s.showAxes,
    showColorBars: s.showColorBars,
  }
}

export const useUiStore = create<UiStore>()(
  persist<UiStore, [], [], PersistedUi>(
    (set, get) => ({
      ...INITIAL,
      setTheme(theme) {
        set({ theme })
      },
      toggleTheme() {
        set((s) => ({ theme: s.theme === 'light' ? 'dark' : 'light' }))
      },
      setLocale(locale) {
        set({ locale })
      },
      toggleLocale() {
        set((s) => ({ locale: s.locale === 'ko' ? 'en' : 'ko' }))
      },
      toggleSide() {
        set((s) => ({ sideVisible: !s.sideVisible }))
      },
      toggleBottom() {
        set((s) => ({ bottomVisible: !s.bottomVisible }))
      },
      toggleAssistant() {
        set((s) => ({ assistantVisible: !s.assistantVisible }))
      },
      setBottomVisible(bottomVisible) {
        set({ bottomVisible })
      },
      setOverlay(what, on) {
        if (what === 'axes') set({ showAxes: on })
        else set({ showColorBars: on })
      },
      setLayout(which, layout) {
        set({ [which]: layout } as Partial<UiState>)
      },
      setActivity(view) {
        set((s) => (s.activity === view && s.sideVisible ? { sideVisible: false } : { activity: view, sideVisible: true }))
      },
      openTab(tab) {
        set((s) => ({ tabs: upsertTab(s.tabs, tab), activeTabId: tab.id }))
      },
      openFile(path) {
        // A campaign directory opens as one campaign tab, not as the raw
        // campaign.json text: the directory ('' at the workspace root) is the tab.
        if (path.split('/').pop() === 'campaign.json') {
          get().openCampaignTab(path.slice(0, path.length - 'campaign.json'.length).replace(/\/+$/, ''))
          return
        }
        // Surface files are geometry, not text: a binary STL in the text editor
        // is mojibake and a STEP is 60 KB of B-rep noise — show the 3D view.
        if (/\.(step|stp|stl|obj)$/i.test(path)) get().openGeometryTab(path)
        else get().openTab({ id: fileTabId(path), kind: 'file', path })
      },
      openViewerTab() {
        get().openTab({ id: 'viewer', kind: 'viewer' })
      },
      openGeometryTab(path) {
        get().openTab({ id: `geometry:${path}`, kind: 'geometry', path })
      },
      openCampaignTab(path) {
        const p = path.replace(/\/+$/, '')
        get().openTab({ id: `campaign:${p}`, kind: 'campaign', path: p })
      },
      openResidualsTab(runId, chart) {
        set((s) => {
          const existing = s.tabs.find((t) => t.kind === 'residuals')
          // No chart given keeps the tab's current chart, else the residuals chart.
          const nextChart = chart ?? (existing?.kind === 'residuals' ? (existing.chart ?? 'residuals') : 'residuals')
          const tab: Tab = {
            id: 'residuals',
            kind: 'residuals',
            runId: runId ?? (existing?.kind === 'residuals' ? existing.runId : null),
            compareRunId: existing?.kind === 'residuals' ? (existing.compareRunId ?? null) : null,
            chart: nextChart,
          }
          return { tabs: upsertTab(s.tabs, tab), activeTabId: 'residuals', activeRunId: runId ?? s.activeRunId }
        })
      },
      setCompareRun(runId) {
        set((s) => {
          const existing = s.tabs.find((t) => t.kind === 'residuals')
          const primary = existing?.kind === 'residuals' ? existing.runId : s.activeRunId
          // The compare overlay never switches the tab's chart.
          const tab: Tab = { id: 'residuals', kind: 'residuals', runId: primary, compareRunId: runId, chart: existing?.kind === 'residuals' ? (existing.chart ?? 'residuals') : 'residuals' }
          return { tabs: upsertTab(s.tabs, tab), activeTabId: 'residuals' }
        })
      },
      setLogFilter(patch) {
        set((s) => {
          const next = { ...s.logFilter, ...patch }
          if (patch.streams) {
            // Canonical order stdout, stderr, system, duplicates dropped.
            const canon: LogStream[] = ['stdout', 'stderr', 'system']
            next.streams = canon.filter((c) => patch.streams!.includes(c))
          }
          return { logFilter: next }
        })
      },
      setMetricSlot(slot, metric) {
        set((s) => {
          const metricSlots: [string | null, string | null] = [s.metricSlots[0], s.metricSlots[1]]
          metricSlots[slot - 1] = metric
          return { metricSlots }
        })
      },
      setMetricMode(metricMode) {
        set({ metricMode })
      },
      setViewSplit(on) {
        set(on ? { viewSplit: true } : { viewSplit: false, focusedView: 'A', camerasLinked: false })
      },
      setFocusedView(view) {
        set({ focusedView: view })
      },
      setCamerasLinked(on) {
        set({ camerasLinked: on })
      },
      openDiffTab(id, path, toolUseId) {
        get().openTab({ id, kind: 'diff', path, toolUseId })
      },
      closeTab(id) {
        set((s) => {
          const i = s.tabs.findIndex((t) => t.id === id)
          if (i < 0) return {}
          const tabs = s.tabs.filter((t) => t.id !== id)
          let activeTabId = s.activeTabId
          if (activeTabId === id) activeTabId = tabs[Math.min(i, tabs.length - 1)]?.id ?? null
          return { tabs, activeTabId }
        })
      },
      closeOtherTabs(id) {
        set((s) => ({ tabs: s.tabs.filter((t) => t.id === id), activeTabId: id }))
      },
      activateTab(id) {
        set((s) => (s.tabs.some((t) => t.id === id) ? { activeTabId: id } : {}))
      },
      activateTabIndex(index) {
        set((s) => (s.tabs[index] ? { activeTabId: s.tabs[index].id } : {}))
      },
      moveTab(from, to) {
        set((s) => {
          if (from === to || !s.tabs[from] || to < 0 || to >= s.tabs.length) return {}
          const tabs = s.tabs.slice()
          const [t] = tabs.splice(from, 1)
          tabs.splice(to, 0, t)
          return { tabs }
        })
      },
      setBottomTab(bottomTab) {
        set({ bottomTab, bottomVisible: true })
      },
      setExpanded(path, expanded) {
        set((s) => {
          const has = s.explorerExpanded.includes(path)
          if (expanded === has) return {}
          return { explorerExpanded: expanded ? [...s.explorerExpanded, path] : s.explorerExpanded.filter((p) => p !== path) }
        })
      },
      collapseAll() {
        set({ explorerExpanded: [] })
      },
      setActiveRun(activeRunId) {
        set({ activeRunId })
      },
      setTerminalRun(terminalRunId) {
        set({ terminalRunId })
      },
      setComposerDraft(composerDraft) {
        set({ composerDraft })
      },
      setPaletteOpen(paletteOpen) {
        set({ paletteOpen })
      },
      setSettingsOpen(settingsOpen) {
        set({ settingsOpen })
      },
      setCursor(cursor) {
        set({ cursor })
      },
      prefillComposer(text) {
        set((s) => ({ composerPrefill: { text, nonce: (s.composerPrefill?.nonce ?? 0) + 1 } }))
      },
      showMeshDialog(draft) {
        set((s) => ({ meshDialog: { draft, runId: s.meshDialog?.runId ?? null, status: s.meshDialog?.status ?? null } }))
      },
      closeMeshDialog() {
        set({ meshDialog: null })
      },
      setMeshStatus(status, runId) {
        set((s) => (s.meshDialog ? { meshDialog: { ...s.meshDialog, status, runId: runId ?? s.meshDialog.runId } } : {}))
      },
      showBoundaryEditor(path) {
        set((s) => ({ boundaryEditor: { path, rev: (s.boundaryEditor?.rev ?? 0) + 1 } }))
      },
      closeBoundaryEditor() {
        set({ boundaryEditor: null })
      },
      openPanel(panel) {
        set({ rightTab: panel, assistantVisible: true })
      },
      setSelection(selection) {
        set({ selection })
      },
      selectNode(node) {
        set({ propertiesNode: node })
      },
    }),
    {
      name: STORAGE_KEY,
      version: UI_STORE_VERSION,
      partialize,
      merge: (persisted, current) => {
        const p = (persisted ?? {}) as Partial<PersistedUi>
        const tabs = Array.isArray(p.tabs) ? p.tabs.filter((t) => t && t.kind !== 'diff') : current.tabs
        const activeTabId = p.activeTabId && tabs.some((t) => t.id === p.activeTabId) ? p.activeTabId : (tabs[0]?.id ?? null)
        return { ...current, ...p, tabs, activeTabId }
      },
      migrate: (persisted, version) => (version === UI_STORE_VERSION ? (persisted as PersistedUi) : partialize(INITIAL)),
    },
  ),
)

/** The workspace-relative path of the active file tab, or null. */
export function selectActiveFile(s: UiStore): string | null {
  const tab = s.tabs.find((t) => t.id === s.activeTabId)
  return tab && tab.kind === 'file' ? tab.path : null
}

export function selectActiveTab(s: UiStore): Tab | null {
  return s.tabs.find((t) => t.id === s.activeTabId) ?? null
}
