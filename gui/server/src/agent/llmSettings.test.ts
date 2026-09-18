// The runtime LLM settings: resolution precedence (env CFD_LLM > stored file >
// boot auto-detect, stored key > env key) and the apply() round-trip through
// config/llm.json. Env keys are read from process.env directly, so every case
// saves and restores the names it touches.
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import type { ServerConfig } from '../config.js'
import { DEFAULT_ZAI_BASE_URL } from '../config.js'
import { createLlmSettingsHandle, loadLlmSettings, resolveEffectiveLlm } from './llmSettings.js'

const SAVED = ['CFD_LLM', 'ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'ZAI_API_KEY', 'CFD_DEMO']
let savedEnv: Record<string, string | undefined> = {}

function takeEnv(): void {
  savedEnv = Object.fromEntries(SAVED.map((k) => [k, process.env[k]]))
  for (const k of SAVED) delete process.env[k]
}

afterEach(() => {
  for (const [k, v] of Object.entries(savedEnv)) {
    if (v === undefined) delete process.env[k]
    else process.env[k] = v
  }
})

function config(overrides: Partial<ServerConfig> = {}): ServerConfig {
  return {
    version: 'test',
    host: '127.0.0.1',
    port: 0,
    workspaceRoot: '/w',
    guiDir: '/gui',
    sessionsDir: '/gui/sessions',
    runsDir: '/gui/runs',
    cacheDir: '/gui/.cache',
    configDir: '/gui/config',
    ontologyDir: '/gui/ontology',
    demo: false,
    llm: 'mock',
    model: 'mock-assistant',
    allowNoApiKey: true,
    binDir: null,
    mockSpeed: 1,
    allowRemote: false,
    authToken: null,
    maxConcurrentRuns: 1,
    longToolTimeoutMs: 60_000,
    logLevel: 'error',
    zai: { baseUrl: DEFAULT_ZAI_BASE_URL, key: null, keyFile: '/nope/zai-key', thinkingTokens: 4096, maxTokens: 64000 },
    ...overrides,
  }
}

describe('resolveEffectiveLlm', () => {
  it('a stored key beats an env key, and says where each came from', () => {
    takeEnv()
    process.env.ANTHROPIC_API_KEY = 'env-anthropic'
    process.env.ZAI_API_KEY = 'env-zai'
    const eff = resolveEffectiveLlm(config(), { zaiKey: 'ui-zai' })
    expect(eff.zaiKey).toBe('ui-zai')
    expect(eff.zaiSource).toBe('ui')
    expect(eff.anthropicKey).toBe('env-anthropic')
    expect(eff.anthropicSource).toBe('env')
  })

  it('env CFD_LLM names the provider outright; a stored provider is next; demo defaults to the mock', () => {
    takeEnv()
    process.env.CFD_LLM = 'anthropic'
    expect(resolveEffectiveLlm(config({ demo: true }), { provider: 'zai' }).provider).toBe('anthropic')
    delete process.env.CFD_LLM
    expect(resolveEffectiveLlm(config({ demo: true }), { provider: 'zai' }).provider).toBe('zai')
    expect(resolveEffectiveLlm(config({ demo: true }), {}).provider).toBe('mock')
    expect(resolveEffectiveLlm(config({ demo: false, llm: 'mock', model: 'mock-assistant' }), {}).provider).toBe('anthropic')
  })

  it('the model follows its own provider: stored per provider, else CFD_MODEL for the boot provider, else the family default', () => {
    takeEnv()
    const cfg = config({ llm: 'zai', model: 'glm-tuned' })
    expect(resolveEffectiveLlm(cfg, { provider: 'zai' }).model).toBe('glm-tuned')
    expect(resolveEffectiveLlm(cfg, { provider: 'anthropic' }).model).toBe('claude-opus-5')
    expect(resolveEffectiveLlm(cfg, { provider: 'anthropic', anthropicModel: 'claude-tuned' }).model).toBe('claude-tuned')
    expect(resolveEffectiveLlm(cfg, { provider: 'mock' }).model).toBe('mock-assistant')
  })
})

describe('stored settings file', () => {
  it('is {} when absent, and round-trips through save + load', async () => {
    takeEnv()
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'cfd-llm-'))
    expect(loadLlmSettings(dir)).toEqual({})
    const handle = createLlmSettingsHandle(config({ configDir: dir }))
    const view = await handle.apply({ provider: 'zai', zaiKey: ' sk-zai-123 ', zaiModel: 'glm-5.3-flash' })
    expect(view.provider).toBe('zai')
    expect(view.model).toBe('glm-5.3-flash')
    expect(view.keys.zai).toEqual({ set: true, source: 'ui' })
    expect(loadLlmSettings(dir)).toEqual({ provider: 'zai', zaiKey: 'sk-zai-123', zaiModel: 'glm-5.3-flash' })
    // The view never carries the raw key.
    expect(JSON.stringify(view)).not.toContain('sk-zai-123')
    fs.rmSync(dir, { recursive: true, force: true })
  })

  it('apply(null) clears a stored field and the next resolution falls back to the environment', async () => {
    takeEnv()
    process.env.ZAI_API_KEY = 'env-zai'
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'cfd-llm-'))
    const handle = createLlmSettingsHandle(config({ configDir: dir, demo: true }))
    await handle.apply({ provider: 'zai', zaiKey: 'ui-zai' })
    expect(handle.view().keys.zai).toEqual({ set: true, source: 'ui' })
    await handle.apply({ zaiKey: null })
    expect(handle.view().keys.zai).toEqual({ set: true, source: 'env' })
    fs.rmSync(dir, { recursive: true, force: true })
  })

  it('a client for a provider with no key falls back to the mock, and the view says so', async () => {
    takeEnv()
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'cfd-llm-'))
    const handle = createLlmSettingsHandle(config({ configDir: dir }))
    await handle.apply({ provider: 'zai', zaiKey: 'temp' })
    expect(handle.view().provider).toBe('zai')
    await handle.apply({ zaiKey: null })
    // zai selected, key gone: client() degrades to the scripted mock and the
    // redacted view names the mock, so the badge never claims a live GLM.
    expect(handle.client().kind).toBe('mock')
    expect(handle.view().provider).toBe('mock')
    fs.rmSync(dir, { recursive: true, force: true })
  })
})
