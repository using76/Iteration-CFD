// argv parsing for the mock CLI, driven by the shared registry so the mock
// accepts exactly what the real driver accepts and refuses the same things.
import { getBinary, type BinarySpec } from '@cfd/shared'

export class UsageError extends Error {
  constructor(
    message: string,
    readonly spec: BinarySpec | null,
  ) {
    super(message)
    this.name = 'UsageError'
  }
}

export interface MockArgs {
  spec: BinarySpec
  positionals: string[]
  flags: Map<string, string[]>
  has(flag: string): boolean
  str(flag: string): string | undefined
  num(flag: string, fallback: number): number
}

export function usageLine(spec: BinarySpec): string {
  const pos = spec.positionals.map((p) => (p.optional ? `[${p.name}]` : `<${p.name}>`)).join(' ')
  const flags = spec.flags
    .map((f) => {
      if (f.type === 'flag') return `[${f.name}]`
      const v = f.values ? f.values.join('|') : f.type === 'int' ? 'N' : f.type === 'path' ? 'PATH' : f.type === 'list' ? 'LIST' : 'X'
      return `[${f.name} ${v}]`
    })
    .join(' ')
  return `usage: ${spec.name} ${[pos, flags].filter(Boolean).join(' ')}`.trimEnd()
}

export function parseMockArgs(binary: string, argv: string[]): MockArgs {
  const spec = getBinary(binary)
  if (!spec) throw new UsageError(`unknown binary ${binary}`, null)
  const positionals: string[] = []
  const flags = new Map<string, string[]>()
  for (let i = 0; i < argv.length; i++) {
    const tok = argv[i]
    if (tok.startsWith('-') && tok.length > 1 && !/^-\d/.test(tok)) {
      const f = spec.flags.find((s) => s.name === tok)
      if (!f) throw new UsageError(`unknown option ${tok}`, spec)
      if (f.type === 'flag') {
        flags.set(tok, [...(flags.get(tok) ?? []), 'true'])
        continue
      }
      const v = argv[i + 1]
      if (v === undefined) throw new UsageError(`${tok} needs a value`, spec)
      i++
      flags.set(tok, [...(flags.get(tok) ?? []), v])
      continue
    }
    positionals.push(tok)
  }
  spec.positionals.forEach((p, i) => {
    if (positionals[i] !== undefined || p.optional) return
    const isCase = p.name === 'case' || p.name === 'caseDir'
    throw new UsageError(isCase ? 'no case directory given' : `missing <${p.name}>`, spec)
  })
  if (positionals.length > spec.positionals.length) throw new UsageError(`unknown option ${positionals[spec.positionals.length]}`, spec)
  return {
    spec,
    positionals,
    flags,
    has: (f) => flags.has(f),
    str: (f) => {
      const v = flags.get(f)
      return v ? v[v.length - 1] : undefined
    },
    num: (f, fallback) => {
      const v = flags.get(f)
      if (!v) return fallback
      const n = Number(v[v.length - 1])
      return Number.isFinite(n) ? n : fallback
    },
  }
}
