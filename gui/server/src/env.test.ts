import { describe, expect, it } from 'vitest'
import { isSecretEnvName, scrubbedEnv } from './env.js'

describe('scrubbedEnv', () => {
  it('drops the names a credential hides behind', () => {
    for (const n of ['ANTHROPIC_API_KEY', 'anthropic_auth_token', 'CFD_AUTH_TOKEN', 'AWS_SECRET_ACCESS_KEY', 'GH_TOKEN', 'DB_PASSWORD', 'GOOGLE_APPLICATION_CREDENTIALS'])
      expect(isSecretEnvName(n)).toBe(true)
  })

  it('keeps what a solver actually needs', () => {
    for (const n of ['PATH', 'HOME', 'TEMP', 'OFGPU_BIN_DIR', 'CUDA_VISIBLE_DEVICES', 'CFD_DEMO', 'KEYBOARD_LAYOUT', 'TOKENIZERS_PARALLELISM'])
      expect(isSecretEnvName(n)).toBe(false)
  })

  it('copies the rest and drops undefined', () => {
    const env = { PATH: '/usr/bin', ANTHROPIC_API_KEY: 'sk-ant-secret', OFGPU_BIN_DIR: 'C:/bin', EMPTY: undefined }
    expect(scrubbedEnv(env)).toEqual({ PATH: '/usr/bin', OFGPU_BIN_DIR: 'C:/bin' })
  })
})
