// Path helpers shared by the tools: workspace confinement, a minimal glob
// matcher and the explorer hide-list.
import { isHiddenDir, resolveInWorkspace, WorkspaceError, type ResolvedPath } from '../workspace/paths.js'
import { fail, type ToolResult } from './context.js'

export type Resolved = { ok: true; path: ResolvedPath } | { ok: false; result: ToolResult }

export function resolveTool(root: string, input: string, opts: { mustExist?: boolean } = {}): Resolved {
  try {
    return { ok: true, path: resolveInWorkspace(root, input, opts) }
  } catch (err) {
    if (err instanceof WorkspaceError) return { ok: false, result: fail(err.code, err.message) }
    return { ok: false, result: fail('INVALID', (err as Error).message) }
  }
}

/** `**`, `*`, `?` and `{a,b}`; matched against the relative path, or the basename when the glob has no slash. */
export function globToRegExp(glob: string): RegExp {
  let re = ''
  for (let i = 0; i < glob.length; i++) {
    const ch = glob[i]
    if (ch === '*') {
      if (glob[i + 1] === '*') {
        i++
        if (glob[i + 1] === '/') {
          i++
          re += '(?:.*/)?'
        } else re += '.*'
      } else re += '[^/]*'
    } else if (ch === '?') re += '[^/]'
    else if (ch === '{') {
      const end = glob.indexOf('}', i)
      if (end < 0) re += '\\{'
      else {
        const alts = glob
          .slice(i + 1, end)
          .split(',')
          .map((s) => s.replace(/[.+^$()|[\]\\]/g, '\\$&').replace(/\*/g, '[^/]*'))
        re += `(?:${alts.join('|')})`
        i = end
      }
    } else re += ch.replace(/[.+^$()|[\]\\]/g, '\\$&')
  }
  return new RegExp(`^${re}$`)
}

export function globMatcher(glob: string | null): (rel: string) => boolean {
  if (!glob) return () => true
  const re = globToRegExp(glob)
  const hasSlash = glob.includes('/')
  return (rel) => re.test(rel) || (!hasSlash && re.test(rel.slice(rel.lastIndexOf('/') + 1)))
}

export { isHiddenDir }
