// A throwaway workspace and a matching ServerConfig, for the tests that need a
// real directory on disk rather than a fake.
//
// Reconstructed along with the rest of this directory - see types.ts. The
// shape is pinned by its six consumers: http/routes.test.ts (makeTempWorkspace,
// testConfig, REPO_ROOT), workspace/{fs,git,search}.test.ts (makeTempWorkspace,
// TempWorkspace) and registry/{schema,sync}.test.ts (REPO_ROOT).
import fsp from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import type { ServerConfig } from '../config.js'

const here = path.dirname(fileURLToPath(import.meta.url))

/** gui/ */
export const GUI_DIR = path.resolve(here, '..', '..', '..')
/** The repository root — where rust/, cases/ and docs/ live. */
export const REPO_ROOT = path.resolve(GUI_DIR, '..')

export interface TempWorkspace {
  /** The temp directory holding everything, including sessions/ and runs/. */
  tmp: string
  /** The workspace root the server is confined to. */
  root: string
  config: ServerConfig
  cleanup(): Promise<void>
}

export interface TempWorkspaceOptions {
  /** Copy cases/plume.jsonc in (default true) — most tests want a real case. */
  plume?: boolean
}

/**
 * A workspace under the OS temp directory with a `cases/` tree, and a config
 * whose sessions, runs, cache and config directories are all inside it, so a
 * test can delete one directory and leave nothing behind.
 */
export async function makeTempWorkspace(opts: TempWorkspaceOptions = {}): Promise<TempWorkspace> {
  const tmp = await fsp.mkdtemp(path.join(os.tmpdir(), 'cfd-runs-'))
  const root = path.join(tmp, 'ws')
  await fsp.mkdir(path.join(root, 'cases'), { recursive: true })
  if (opts.plume !== false) {
    await fsp.copyFile(path.join(REPO_ROOT, 'cases', 'plume.jsonc'), path.join(root, 'cases', 'plume.jsonc'))
  }
  const ws: TempWorkspace = {
    tmp,
    root,
    config: baseConfig(tmp, root),
    cleanup: () => fsp.rm(tmp, { recursive: true, force: true }),
  }
  return ws
}

function baseConfig(tmp: string, root: string): ServerConfig {
  return {
    version: '0.0.0-test',
    host: '127.0.0.1',
    port: 0,
    workspaceRoot: root,
    guiDir: GUI_DIR,
    sessionsDir: path.join(tmp, 'sessions'),
    runsDir: path.join(tmp, 'runs'),
    cacheDir: path.join(tmp, 'cache'),
    configDir: path.join(tmp, 'config'),
    demo: true,
    llm: 'mock',
    model: 'claude-opus-5',
    allowNoApiKey: true,
    binDir: null,
    // The mock solver paces itself against this; tests do not wait for realism.
    mockSpeed: 100,
    allowRemote: false,
    authToken: null,
    maxConcurrentRuns: 4,
    logLevel: 'error',
  }
}

/** The workspace's config, with any field a test needs to differ. */
export function testConfig(ws: TempWorkspace, over: Partial<ServerConfig> = {}): ServerConfig {
  return { ...baseConfig(ws.tmp, ws.root), ...over }
}
