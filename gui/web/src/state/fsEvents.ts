// Tiny broadcast for `fs.changed` so the explorer and search panes can
// refresh without the WS client knowing about them.
type Listener = (paths: string[]) => void
const listeners = new Set<Listener>()

export const fsEvents = {
  subscribe(l: Listener): () => void {
    listeners.add(l)
    return () => listeners.delete(l)
  },
  emit(paths: string[]): void {
    for (const l of listeners) l(paths)
  },
}
