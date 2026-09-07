// Slow-changing server facts fetched over REST: the registry, git status and
// the JSON schema. Cached for the life of the page, refreshable.
import { create } from 'zustand'
import { api, type GitStatusResponse, type RegistryResponse } from '../api/rest'

export interface MetaStore {
  registry: RegistryResponse | null
  registryError: string | null
  git: GitStatusResponse | null
  loadRegistry(): Promise<void>
  refreshGit(): Promise<void>
}

let registryPromise: Promise<void> | null = null
let gitTimer: ReturnType<typeof setTimeout> | null = null

export const useMetaStore = create<MetaStore>()((set) => ({
  registry: null,
  registryError: null,
  git: null,
  loadRegistry() {
    if (!registryPromise) {
      registryPromise = api
        .registry()
        .then((registry) => set({ registry, registryError: null }))
        .catch((err: unknown) => {
          registryPromise = null
          set({ registryError: err instanceof Error ? err.message : String(err) })
        })
    }
    return registryPromise
  },
  async refreshGit() {
    if (gitTimer) return
    gitTimer = setTimeout(() => {
      gitTimer = null
    }, 1500)
    try {
      set({ git: await api.gitStatus() })
    } catch {
      set({ git: { available: false, branch: null, upstream: null, ahead: 0, behind: 0, changes: [], error: 'unreachable' } })
    }
  },
}))
