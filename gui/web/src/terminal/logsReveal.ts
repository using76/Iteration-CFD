// Cross-pane request: "show log line `seq` of run `runId`" (Problems -> Logs).
import { create } from 'zustand'

interface LogsRevealStore {
  request: { runId: string; seq: number; nonce: number } | null
  reveal(runId: string, seq: number): void
}

export const useLogsReveal = create<LogsRevealStore>()((set) => ({
  request: null,
  reveal(runId, seq) {
    set((s) => ({ request: { runId, seq, nonce: (s.request?.nonce ?? 0) + 1 } }))
  },
}))
