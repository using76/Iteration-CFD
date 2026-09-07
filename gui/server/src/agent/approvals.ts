// Pending approvals: one request per tool round (all `ask` calls batched),
// resolved per tool_use id by tool.approve / tool.deny, or expired.
import type { PendingApproval } from '@cfd/shared'

export const APPROVAL_TTL_MS = 10 * 60 * 1000

export type ApprovalDecision = 'approved' | 'denied' | 'expired'

export interface ApprovalOutcome {
  decision: ApprovalDecision
  reason: string | null
}

interface Waiter {
  approval: PendingApproval
  resolvers: Map<string, (o: ApprovalOutcome) => void>
  timer: NodeJS.Timeout
}

export interface ApprovalRequest {
  approval: PendingApproval
  /** One promise per tool_use id, in the order of `calls`. */
  outcomes: Map<string, Promise<ApprovalOutcome>>
}

export interface ApprovalManager {
  /** `signal` settles every waiter as denied when the turn is cancelled, instead of leaving them to the TTL. */
  request(turnId: string, calls: PendingApproval['calls'], ttlMs?: number, signal?: AbortSignal): ApprovalRequest
  resolve(toolUseIds: string[], decision: 'approved' | 'denied', reason?: string | null): string[]
  /** Deny everything still pending (turn cancelled / session deleted). */
  cancelAll(reason: string): void
  pending(): PendingApproval[]
  /** Decisions already applied, so a late UI click can be answered. */
  has(toolUseId: string): boolean
}

export function createApprovalManager(onResolved?: (toolUseIds: string[], decision: ApprovalDecision) => void): ApprovalManager {
  const waiters = new Map<string, Waiter>()
  const byToolUse = new Map<string, Waiter>()

  function settle(w: Waiter, ids: string[], outcome: ApprovalOutcome): string[] {
    const done: string[] = []
    for (const id of ids) {
      const r = w.resolvers.get(id)
      if (!r) continue
      w.resolvers.delete(id)
      byToolUse.delete(id)
      r(outcome)
      done.push(id)
    }
    if (!w.resolvers.size) {
      clearTimeout(w.timer)
      waiters.delete(w.approval.turnId + ':' + w.approval.requestedAt)
    }
    if (done.length) onResolved?.(done, outcome.decision)
    return done
  }

  return {
    request(turnId, calls, ttlMs = APPROVAL_TTL_MS, signal?: AbortSignal) {
      const now = Date.now()
      const approval: PendingApproval = { turnId, toolUseIds: calls.map((c) => c.toolUseId), calls, requestedAt: now, expiresAt: now + ttlMs }
      const w: Waiter = { approval, resolvers: new Map(), timer: setTimeout(() => settle(w, [...w.resolvers.keys()], { decision: 'expired', reason: 'approval timed out' }), ttlMs) }
      w.timer.unref?.()
      const outcomes = new Map<string, Promise<ApprovalOutcome>>()
      for (const c of calls) {
        outcomes.set(
          c.toolUseId,
          new Promise<ApprovalOutcome>((resolve) => {
            w.resolvers.set(c.toolUseId, resolve)
          }),
        )
        byToolUse.set(c.toolUseId, w)
      }
      waiters.set(turnId + ':' + now, w)
      // A cancel that lands after the waiters exist is handled by cancelAll; one
      // that landed before they existed would otherwise hold the turn until the
      // 10-minute TTL, because there was nothing to cancel at the time.
      if (signal) {
        const onAbort = () => settle(w, [...w.resolvers.keys()], { decision: 'denied', reason: 'cancelled by user' })
        if (signal.aborted) onAbort()
        else signal.addEventListener('abort', onAbort, { once: true })
      }
      return { approval, outcomes }
    },
    resolve(toolUseIds, decision, reason = null) {
      const done: string[] = []
      const groups = new Map<Waiter, string[]>()
      for (const id of toolUseIds) {
        const w = byToolUse.get(id)
        if (w) groups.set(w, [...(groups.get(w) ?? []), id])
      }
      for (const [w, ids] of groups) done.push(...settle(w, ids, { decision, reason }))
      return done
    },
    cancelAll(reason) {
      for (const w of [...waiters.values()]) settle(w, [...w.resolvers.keys()], { decision: 'denied', reason })
    },
    pending: () => [...waiters.values()].map((w) => ({ ...w.approval, toolUseIds: [...w.resolvers.keys()], calls: w.approval.calls.filter((c) => w.resolvers.has(c.toolUseId)) })).filter((a) => a.toolUseIds.length),
    has: (id) => byToolUse.has(id),
  }
}
