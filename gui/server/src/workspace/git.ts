// Read-only `git status` for the Source Control panel and the status bar.
import { execFile } from 'node:child_process'
import { scrubbedEnv } from '../env.js'

export interface GitChange {
  /** Two-letter porcelain status ("M ", "??", "A ", ...). */
  status: string
  path: string
}

export interface GitStatus {
  available: boolean
  branch: string | null
  upstream: string | null
  ahead: number
  behind: number
  changes: GitChange[]
  error: string | null
}

export function parsePorcelain(text: string): Omit<GitStatus, 'available' | 'error'> {
  let branch: string | null = null
  let upstream: string | null = null
  let ahead = 0
  let behind = 0
  const changes: GitChange[] = []
  for (const line of text.split(/\r?\n/)) {
    if (!line) continue
    if (line.startsWith('## ')) {
      const head = line.slice(3)
      const m = head.match(/^(?:No commits yet on )?([^ .]+(?:\.\.\.(\S+))?)(?: \[(.*)\])?/)
      if (m) {
        branch = m[1].split('...')[0]
        upstream = m[2] ?? null
        const a = m[3]?.match(/ahead (\d+)/)
        const b = m[3]?.match(/behind (\d+)/)
        ahead = a ? Number(a[1]) : 0
        behind = b ? Number(b[1]) : 0
      }
      continue
    }
    const status = line.slice(0, 2)
    let p = line.slice(3)
    const arrow = p.indexOf(' -> ')
    if (arrow >= 0) p = p.slice(arrow + 4)
    changes.push({ status, path: p })
  }
  return { branch, upstream, ahead, behind, changes }
}

export function gitStatus(cwd: string): Promise<GitStatus> {
  return new Promise((resolve) => {
    execFile('git', ['status', '--porcelain=v1', '-b', '--untracked-files=normal'], { cwd, timeout: 10_000, windowsHide: true, maxBuffer: 8 * 1024 * 1024, env: scrubbedEnv() }, (err, stdout, stderr) => {
      if (err) {
        resolve({ available: false, branch: null, upstream: null, ahead: 0, behind: 0, changes: [], error: (String(stderr) || err.message).trim() })
        return
      }
      resolve({ available: true, error: null, ...parsePorcelain(String(stdout)) })
    })
  })
}
