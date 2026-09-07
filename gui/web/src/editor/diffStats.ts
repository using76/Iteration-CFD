// Cheap +/- line counts for diff cards (multiset difference of lines).
export function diffStats(before: string, after: string): { added: number; removed: number } {
  const count = (text: string) => {
    const m = new Map<string, number>()
    if (text === '') return m
    for (const line of text.split('\n')) m.set(line, (m.get(line) ?? 0) + 1)
    return m
  }
  const a = count(before)
  const b = count(after)
  let added = 0
  let removed = 0
  for (const [line, n] of b) added += Math.max(0, n - (a.get(line) ?? 0))
  for (const [line, n] of a) removed += Math.max(0, n - (b.get(line) ?? 0))
  return { added, removed }
}
