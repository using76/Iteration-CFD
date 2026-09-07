// Layout and preference state, persisted to localStorage (versioned).
import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { Locale } from '@cfd/shared'

export type Theme = 'light' | 'dark'
export type ActivityView = 'explorer' | 'search' | 'scm' | 'run' | 'extensions' | 'cfd'
export type BottomTab = 'terminal' | 'logs' | 'problems' | 'output'

export type Tab =
  | { id: string; kind: 'file'; path: string }
  | { id: 'viewer'; kind: 'viewer' }
  | { id: 'residuals'; kind: 'residuals'; runId: string | null }
  | { id: string; kind: 'diff'; path: string; toolUseId: string | null }

export type PanelLayout = Record<string, number>

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
  setLayout(which: 'mainLayout' | 'centerLayout' | 'bottomLayout', layout: PanelLayout): void
  setActivity(view: ActivityView): void
  openTab(tab: Tab): void
  openFile(path: string): void
  openViewerTab(): void
  openResidualsTab(runId: string | null): void
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
}

export function fileTabId(path: string): string {
  return `file:${path}`
}

function upsertTab(tabs: Tab[], tab: Tab): Tab[] {
  const i = tabs.findIndex((t) => t.id === tab.id)
  if (i < 0) return [...tabs, tab]
  const out = tabs.slice()
  out[i] = tab
  return out
}

type PersistedUi = Pick<UiState, 'theme' | 'locale' | 'sideVisible' | 'bottomVisible' | 'assistantVisible' | 'mainLayout' | 'centerLayout' | 'bottomLayout' | 'activity' | 'tabs' | 'activeTabId' | 'bottomTab' | 'explorerExpanded' | 'activeRunId' | 'terminalRunId' | 'composerDraft'>

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
        get().openTab({ id: fileTabId(path), kind: 'file', path })
      },
      openViewerTab() {
        get().openTab({ id: 'viewer', kind: 'viewer' })
      },
      openResidualsTab(runId) {
        set((s) => {
          const existing = s.tabs.find((t) => t.kind === 'residuals')
          const tab: Tab = { id: 'residuals', kind: 'residuals', runId: runId ?? (existing?.kind === 'residuals' ? existing.runId : null) }
          return { tabs: upsertTab(s.tabs, tab), activeTabId: 'residuals', activeRunId: runId ?? s.activeRunId }
        })
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
