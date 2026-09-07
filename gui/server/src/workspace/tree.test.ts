// L1: the explorer tree must not walk out of the workspace through a symlink.
import fsp from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { fsTree } from './fs.js'

let base = ''
afterEach(async () => {
  if (base) await fsp.rm(base, { recursive: true, force: true })
  base = ''
})

/**
 * A real symlink needs developer mode or admin rights on Windows; a directory
 * junction needs neither and lstat reports it as a symbolic link just the same,
 * so try that second rather than skip the test on a normal Windows account.
 */
async function linkDir(target: string, link: string): Promise<boolean> {
  for (const type of ['dir', 'junction'] as const) {
    try {
      await fsp.symlink(target, link, type)
      return true
    } catch {
      // try the next kind
    }
  }
  return false
}

describe('fsTree', () => {
  it('lists neither the link nor what it points at', async (ctx) => {
    base = await fsp.mkdtemp(path.join(os.tmpdir(), 'cfd-tree-'))
    const root = path.join(base, 'workspace')
    const outside = path.join(base, 'outside')
    await fsp.mkdir(path.join(root, 'cases'), { recursive: true })
    await fsp.mkdir(outside, { recursive: true })
    await fsp.writeFile(path.join(outside, 'secret.txt'), 'not for the explorer')
    await fsp.writeFile(path.join(root, 'real.txt'), 'in the workspace')
    if (!(await linkDir(outside, path.join(root, 'escape')))) return ctx.skip()

    const tree = await fsTree(root, '', 3)
    const names = (tree.children ?? []).map((c) => c.name)
    expect(names).toContain('cases')
    expect(names).toContain('real.txt')
    expect(names).not.toContain('escape')
    expect(JSON.stringify(tree)).not.toContain('secret.txt')
  })
})
