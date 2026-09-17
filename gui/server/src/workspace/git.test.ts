import { describe, expect, it } from 'vitest'
import { gitHead, gitStatus, parsePorcelain } from './git.js'
import { REPO_ROOT } from '../runs/test-helpers.js'

describe('git status', () => {
  it('parses porcelain v1 with a branch header', () => {
    const s = parsePorcelain('## feature/x...origin/feature/x [ahead 2, behind 1]\n M gui/PLAN.md\n?? gui/server/src/http/\nR  a.txt -> b.txt\n')
    expect(s.branch).toBe('feature/x')
    expect(s.upstream).toBe('origin/feature/x')
    expect(s.ahead).toBe(2)
    expect(s.behind).toBe(1)
    expect(s.changes).toEqual([
      { status: ' M', path: 'gui/PLAN.md' },
      { status: '??', path: 'gui/server/src/http/' },
      { status: 'R ', path: 'b.txt' },
    ])
    expect(parsePorcelain('## main\n').branch).toBe('main')
    expect(parsePorcelain('## No commits yet on main\n').branch).toBe('main')
  })
  it('reads the repository status and degrades outside a repository', async () => {
    const s = await gitStatus(REPO_ROOT)
    expect(s.available).toBe(true)
    expect(typeof s.branch).toBe('string')
    const none = await gitStatus('/')
    expect(none.available).toBe(false)
    expect(none.changes).toEqual([])
  })
})

describe('git head', () => {
  it('reads the sha and the dirty flag of a real repository', async () => {
    const h = await gitHead(REPO_ROOT)
    expect(h.sha).toMatch(/^[0-9a-f]{40}$/)
    expect(typeof h.dirty).toBe('boolean')
  })
  it('degrades to nulls outside a repository', async () => {
    await expect(gitHead('/')).resolves.toEqual({ sha: null, dirty: null })
  })
})
