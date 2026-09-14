// gui/server/src/ontology/folds/git.ts — a pure parser plus a thin spawn: one git log of HEAD,
// six US-separated fields with a leading RS so a shortstat tail stays inside its own record (C12).
// CommitFile rows, --all and the commit topology are declared refusals, not omissions (D11).
import { execFile } from 'node:child_process'
import { scrubbedEnv } from '../../env.js'
import { emptyFoldReport, skip, upsert, type FoldContext, type FoldReport, type MirrorRow } from './base.js'

const GIT_LOG_ARGV = [
  'log', '--no-color', '--max-count=5000', '--date=iso-strict', '--shortstat',
  '--format=%x1e%H%x1f%h%x1f%an%x1f%ad%x1f%cd%x1f%s%x1f',
]
const GIT_OPTS = { timeout: 30_000, windowsHide: true, maxBuffer: 16 * 1024 * 1024, env: scrubbedEnv() }

interface GitLogResult { ok: boolean; stdout: string; error: string | null }

async function spawnGit(argv: string[], cwd: string): Promise<GitLogResult> {
  return new Promise((resolve) => {
    execFile('git', argv, { cwd, ...GIT_OPTS }, (err, stdout, stderr) => {
      if (err) { resolve({ ok: false, stdout: '', error: (String(stderr) || err.message).trim() }); return }
      resolve({ ok: true, stdout: String(stdout), error: null })
    })
  })
}

export async function readGitLog(workspaceRoot: string): Promise<GitLogResult> {
  return spawnGit(['-C', workspaceRoot, ...GIT_LOG_ARGV], workspaceRoot)
}

export async function readGitBranch(workspaceRoot: string): Promise<string | null> {
  const res = await spawnGit(['-C', workspaceRoot, 'rev-parse', '--abbrev-ref', 'HEAD'], workspaceRoot)
  if (!res.ok) return null
  const branch = res.stdout.trim()
  return branch === 'HEAD' ? null : branch
}

/** Split on RS, drop the first (empty) chunk, split each chunk on US, and refuse a chunk that
 *  does not yield seven parts (six fields plus the shortstat tail). The RS leads and a US trails
 *  so `--shortstat`'s line lands in the same chunk as its own commit, in a seventh part the six
 *  fields never see. A subject containing a pipe survives whole. */
export function parseGitLog(stdout: string): Array<Record<string, unknown>> {
  const rows: Array<Record<string, unknown>> = []
  for (const chunk of stdout.split('\x1e').slice(1)) {
    const parts = chunk.split('\x1f')
    if (parts.length !== 7) continue
    const [sha, shortSha, author, authoredAt, committedAt, subject, tail] = parts
    rows.push({
      sha, shortSha, author, authoredAt, committedAt, subject,
      nFiles: Number(/(\d+) files? changed/.exec(tail)?.[1] ?? 0),
    })
  }
  return rows
}

export async function foldCommits(ctx: FoldContext): Promise<FoldReport[]> {
  const rep = emptyFoldReport(ctx.type.commit)
  const t0 = Date.now()
  const reports = [rep]
  if (!ctx.writer.hasObjectType(ctx.type.commit)) {
    skip(rep, ctx.type.commit, 'not declared in the ontology registry; the fold writes no row')
    return reports
  }
  const log = await readGitLog(ctx.workspaceRoot)
  if (!log.ok) {
    const err = new Error('git log failed (' + log.error + '); the tree may not be a repository')
    ;(err as Error & { source?: string; reports?: FoldReport[] }).source = 'git:HEAD'
    ;(err as Error & { source?: string; reports?: FoldReport[] }).reports = reports
    throw err
  }
  const branch = await readGitBranch(ctx.workspaceRoot)
  const rows = parseGitLog(log.stdout)
  for (const r of rows) {
    const sha = String(r.sha)
    const row: MirrorRow = {
      objectType: ctx.type.commit,
      primaryKey: sha,
      properties: {
        sha,
        shortSha: String(r.shortSha),
        subject: String(r.subject),
        author: String(r.author),
        authoredAt: String(r.authoredAt),
        committedAt: String(r.committedAt),
        nFiles: Number(r.nFiles),
        branch,
      },
      sourcePath: 'git:HEAD',
      importedAt: ctx.now(),
    }
    await upsert(ctx, rep, row)
  }
  // Three declared refusals (D11): the topology, the file list, and other branches.
  skip(rep, 'commit parents', 'the ontology declares no parent property and no commit-to-commit link; the merge topology is not in the stage-0 mirror', rows.length)
  skip(rep, 'CommitFile', '--numstat would add one row per commit and file that no declared link reaches; --shortstat buys nFiles without the rows', rows.length)
  skip(rep, '--all', 'the log reads HEAD only: the mirror records the branch the product is actually on', 1)
  rep.notes.push('every imported Commit row names git:HEAD as its source: a commit has no file, and that is the one reserved form (C14)')
  rep.seconds = (Date.now() - t0) / 1000
  return reports
}
