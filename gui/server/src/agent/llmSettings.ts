// Runtime LLM settings: the API keys and provider the settings UI writes,
// kept in config/llm.json beside the policy and custom-tool files. The file
// overrides the environment (it is the more recent, more deliberate choice),
// the environment still wins where it names the provider outright (CFD_LLM),
// and with neither the boot auto-detect applies. The raw keys never leave
// this module: the UI only ever sees the redacted LlmStateView.
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'
import Anthropic from '@anthropic-ai/sdk'
import type { LlmSettingsPatch, LlmStateView } from '@cfd/shared'
import { DEFAULT_ZAI_BASE_URL, type ServerConfig } from '../config.js'
import { createAnthropicClient } from './anthropic.js'
import type { LlmClient } from './llm.js'
import { createMockLlm } from './mockLlm.js'
import { createZaiClient } from './zai.js'

/** What config/llm.json holds. Absent fields mean "not chosen here". */
export interface StoredLlmSettings {
  provider?: 'anthropic' | 'zai' | 'mock'
  anthropicKey?: string
  zaiKey?: string
  anthropicModel?: string
  zaiModel?: string
}

export function llmSettingsFile(configDir: string): string {
  return path.join(configDir, 'llm.json')
}

/** The stored settings, or {} when the file is absent or unreadable - a broken file must never keep the server from booting. */
export function loadLlmSettings(configDir: string): StoredLlmSettings {
  try {
    const raw = JSON.parse(fs.readFileSync(llmSettingsFile(configDir), 'utf8')) as Record<string, unknown>
    const out: StoredLlmSettings = {}
    if (raw.provider === 'anthropic' || raw.provider === 'zai' || raw.provider === 'mock') out.provider = raw.provider
    for (const k of ['anthropicKey', 'zaiKey', 'anthropicModel', 'zaiModel'] as const) {
      const v = raw[k]
      if (typeof v === 'string' && v.trim()) out[k] = v.trim()
    }
    return out
  } catch {
    return {}
  }
}

export async function saveLlmSettings(configDir: string, settings: StoredLlmSettings): Promise<void> {
  await fsp.mkdir(configDir, { recursive: true })
  await fsp.writeFile(llmSettingsFile(configDir), JSON.stringify(settings, null, 2) + '\n', { mode: 0o600 })
}

export interface EffectiveLlm {
  provider: 'anthropic' | 'zai' | 'mock'
  model: string
  anthropicKey: string | null
  zaiKey: string | null
  /** Where each key came from, for the redacted view; null when the key is absent. */
  anthropicSource: 'ui' | 'env' | null
  zaiSource: 'ui' | 'env' | 'file' | null
}

/** The env key for the plain Anthropic client, or null. */
function envAnthropicKey(): string | null {
  return process.env.ANTHROPIC_API_KEY?.trim() || process.env.ANTHROPIC_AUTH_TOKEN?.trim() || null
}

/** ZAI_API_KEY, else the key file's contents - the same chain config.ts resolves at boot, re-read so a new key file counts without a restart. */
function envOrFileZaiKey(config: ServerConfig): { key: string | null; source: 'env' | 'file' | null } {
  const direct = process.env.ZAI_API_KEY?.trim()
  if (direct) return { key: direct, source: 'env' }
  const file = config.zai?.keyFile
  if (file) {
    try {
      const key = fs.readFileSync(file, 'utf8').trim()
      if (key) return { key, source: 'file' }
    } catch {
      // no key file: the ordinary case
    }
  }
  return { key: null, source: null }
}

/** Env CFD_LLM when it names a known provider, else null. */
function envProvider(): 'anthropic' | 'zai' | 'mock' | null {
  const raw = process.env.CFD_LLM?.trim()
  return raw === 'anthropic' || raw === 'zai' || raw === 'mock' ? raw : null
}

/** Merge the stored settings over the environment into the provider/model/keys the next turn uses. */
export function resolveEffectiveLlm(config: ServerConfig, stored: StoredLlmSettings): EffectiveLlm {
  const anthropicEnv = envAnthropicKey()
  const anthropicKey = stored.anthropicKey ?? anthropicEnv
  const zaiEnv = envOrFileZaiKey(config)
  const zaiKey = stored.zaiKey ?? zaiEnv.key
  // The same auto-detect config.ts uses at boot, but with the stored keys counted in.
  const provider = envProvider() ?? stored.provider ?? (config.demo ? 'mock' : anthropicKey ? 'anthropic' : zaiKey ? 'zai' : 'anthropic')
  // A stored model only makes sense for its own provider; CFD_MODEL keeps governing the provider it was set for.
  const model =
    provider === 'mock'
      ? 'mock-assistant'
      : provider === 'zai'
        ? stored.zaiModel ?? (config.llm === 'zai' ? config.model : 'glm-5.3-flash')
        : stored.anthropicModel ?? (config.llm === 'anthropic' ? config.model : 'claude-opus-5')
  return {
    provider,
    model,
    anthropicKey,
    zaiKey,
    anthropicSource: stored.anthropicKey ? 'ui' : anthropicEnv ? 'env' : null,
    zaiSource: stored.zaiKey ? 'ui' : zaiEnv.source,
  }
}

export function buildLlmClient(config: ServerConfig, eff: EffectiveLlm): LlmClient {
  if (eff.provider === 'mock') return createMockLlm({ model: eff.model })
  if (eff.provider === 'zai') {
    if (!eff.zaiKey) throw new Error('no z.ai API key: enter one in the settings popover, set ZAI_API_KEY, or put it in the CFD_ZAI_KEY_FILE')
    return createZaiClient({
      model: eff.model,
      zai: {
        baseUrl: config.zai?.baseUrl ?? DEFAULT_ZAI_BASE_URL,
        key: eff.zaiKey,
        keyFile: config.zai?.keyFile ?? '',
        thinkingTokens: config.zai?.thinkingTokens ?? 4096,
        maxTokens: config.zai?.maxTokens ?? 64000,
      },
    })
  }
  // An explicit key only when the stored one should win; otherwise the SDK reads
  // the environment itself, exactly as the boot-built client always did.
  const sdk = eff.anthropicSource === 'ui' && eff.anthropicKey ? new Anthropic({ apiKey: eff.anthropicKey }) : new Anthropic()
  return createAnthropicClient({ model: eff.model }, sdk)
}

/** The redacted view, derived once so view() and apply() cannot drift apart. The
 *  provider names the client that will actually run: a configured provider whose
 *  key is missing degrades to the mock, and the UI must say so. */
function viewOf(config: ServerConfig, stored: StoredLlmSettings, eff: EffectiveLlm, kind: LlmClient['kind']): LlmStateView {
  return {
    provider: kind,
    model: eff.model,
    keys: {
      anthropic: { set: eff.anthropicKey !== null, source: eff.anthropicSource },
      zai: { set: eff.zaiKey !== null, source: eff.zaiSource },
    },
    models: {
      anthropic: stored.anthropicModel ?? (config.llm === 'anthropic' ? config.model : 'claude-opus-5'),
      zai: stored.zaiModel ?? (config.llm === 'zai' ? config.model : 'glm-5.3-flash'),
    },
  }
}

/** Minimal logger shape so tests can pass a stub. */
export interface LlmSettingsLog {
  info(message: string): void
  warn(message: string): void
}

export interface LlmSettingsHandle {
  /** The client the next turn streams from. */
  client(): LlmClient
  /** The redacted state the settings UI renders. */
  view(): LlmStateView
  /** Merge a patch into the stored file, rebuild the client, and answer with the new view. */
  apply(patch: LlmSettingsPatch): Promise<LlmStateView>
}

export function createLlmSettingsHandle(config: ServerConfig, log?: LlmSettingsLog): LlmSettingsHandle {
  let stored = loadLlmSettings(config.configDir)
  let eff = resolveEffectiveLlm(config, stored)
  let client: LlmClient | null = null

  const current = (): LlmClient => {
    if (!client) {
      try {
        client = buildLlmClient(config, eff)
      } catch (err) {
        // An unusable real client must not kill the service: fall back to the
        // mock until a key exists, and retry the real build on the next turn.
        log?.warn(`${(err as Error).message}; using the scripted mock until then`)
        client = createMockLlm({ model: eff.model })
      }
    }
    return client
  }

  return {
    client: current,
    view: () => viewOf(config, stored, eff, current().kind),
    async apply(patch: LlmSettingsPatch): Promise<LlmStateView> {
      // A present field replaces the stored value; null clears it; an empty
      // string is the UI's "unchanged" and lands as absent.
      const next: StoredLlmSettings = { ...stored }
      const put = (k: 'anthropicKey' | 'zaiKey' | 'anthropicModel' | 'zaiModel', v: string | null | undefined) => {
        if (v === undefined) return
        if (v === null || !v.trim()) delete next[k]
        else next[k] = v.trim()
      }
      put('anthropicKey', patch.anthropicKey)
      put('zaiKey', patch.zaiKey)
      put('anthropicModel', patch.anthropicModel)
      put('zaiModel', patch.zaiModel)
      if (patch.provider !== undefined) {
        if (patch.provider === null) delete next.provider
        else next.provider = patch.provider
      }
      await saveLlmSettings(config.configDir, next)
      stored = next
      eff = resolveEffectiveLlm(config, stored)
      client = null
      log?.info(`llm provider=${eff.provider} model=${eff.model}`)
      return viewOf(config, stored, eff, current().kind)
    },
  }
}
