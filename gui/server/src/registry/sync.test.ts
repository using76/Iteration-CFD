// Keeps shared/registry.ts in step with the Rust sources: Cargo [[bin]]
// names/paths, the flag set of every `fn usage()` binary, the `const USAGE`
// text of the others, and the model names.
import fs from 'node:fs'
import path from 'node:path'
import { describe, expect, it } from 'vitest'
import { BINARIES, MODELS } from '@cfd/shared'
import { REPO_ROOT } from '../runs/test-helpers.js'

const RUST = path.join(REPO_ROOT, 'rust')
const read = (p: string) => fs.readFileSync(p, 'utf8')

function cargoBins(): Array<{ name: string; path: string }> {
  const text = read(path.join(RUST, 'Cargo.toml'))
  const out: Array<{ name: string; path: string }> = []
  for (const m of text.matchAll(/\[\[bin\]\]\s*\nname = "([^"]+)"\s*\npath = "([^"]+)"/g)) out.push({ name: m[1], path: m[2] })
  return out
}

/** The concatenated string literal of `fn usage()` with Rust's `\`-newline continuations and `{}` expanded. */
function usageText(source: string): string {
  const m = source.match(/fn usage\(\)\s*\{\s*eprintln!\(\s*"((?:[^"\\]|\\[\s\S])*)"/)
  if (!m) throw new Error('no fn usage() literal')
  return m[1]
    .replace(/\\\r?\n\s*/g, '')
    .replace(/\\n/g, '\n')
    .replace(/\{\}/g, '\n  -permissive     downgrade unsupported-setting errors to warnings')
}

function constUsage(source: string): string {
  const m = source.match(/const USAGE: &str = "((?:[^"\\]|\\[\s\S])*)"/)
  if (!m) throw new Error('no const USAGE')
  return m[1].replace(/\\\r?\n\s*/g, '')
}

const flagTokens = (text: string) => new Set([...text.matchAll(/(?<=[\s[])(-[A-Za-z][A-Za-z0-9]*)(?=[\s\]])/g)].map((m) => m[1]))

describe('registry <-> rust sources', () => {
  it('lists exactly the Cargo [[bin]] targets with their source paths', () => {
    const cargo = cargoBins().sort((a, b) => a.name.localeCompare(b.name))
    const reg = BINARIES.map((b) => ({ name: b.name, path: b.source })).sort((a, b) => a.name.localeCompare(b.name))
    expect(reg).toEqual(cargo)
    for (const b of BINARIES) expect(fs.existsSync(path.join(RUST, b.source)), b.source).toBe(true)
  })

  it('has the same flag set as every fn usage()', () => {
    for (const b of BINARIES.filter((x) => x.usageKind === 'usageFn')) {
      const usage = usageText(read(path.join(RUST, b.source)))
      const inUsage = [...flagTokens(usage)].sort()
      const inSpec = b.flags.map((f) => f.name).sort()
      expect(inSpec, `${b.name}: registry flags vs usage()`).toEqual(inUsage)
    }
  })

  it('mentions every registry flag in the const USAGE binaries', () => {
    for (const b of BINARIES.filter((x) => x.usageKind === 'constUsage')) {
      const usage = constUsage(read(path.join(RUST, b.source)))
      for (const f of b.flags) expect(usage, `${b.name} ${f.name}`).toContain(f.name)
      const tokens = [...flagTokens(usage)].filter((t) => t.length > 2)
      for (const t of tokens) expect(b.flags.map((f) => f.name), `${b.name}: ${t} documented but not in the registry`).toContain(t)
    }
  })

  it('names every model somewhere in rust/src', () => {
    const files: string[] = []
    const walk = (dir: string) => {
      for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
        const p = path.join(dir, e.name)
        if (e.isDirectory()) walk(p)
        else if (e.name.endsWith('.rs')) files.push(p)
      }
    }
    walk(path.join(RUST, 'src'))
    const corpus = files.map(read).join('\n')
    for (const m of MODELS) expect(corpus.includes(m.name), `model ${m.name}`).toBe(true)
    for (const m of MODELS) for (const d of m.drivers) expect(BINARIES.map((b) => b.name), `${m.name} driver ${d}`).toContain(d)
  })

  it('agrees with driver_for on which driver builds the k-epsilon family', () => {
    const common = read(path.join(RUST, 'src', 'bin', 'common', 'mod.rs'))
    expect(common).toMatch(/RasModel::KEpsilon \| RasModel::RealizableKE \| RasModel::RNGkEpsilon => "ofgpu-k-epsilon"/)
    const ke = BINARIES.find((b) => b.name === 'ofgpu-k-epsilon')!
    expect(ke.builds).toEqual(expect.arrayContaining(['kEpsilon', 'realizableKE', 'RNGkEpsilon']))
  })
})
