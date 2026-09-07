import os from 'node:os'
import { describe, expect, it } from 'vitest'
import { spawnCapture } from './shell.js'

const PRINT_ENV = 'process.stdout.write(JSON.stringify({key: process.env.ANTHROPIC_API_KEY ?? null, token: process.env.CFD_AUTH_TOKEN ?? null, path: Boolean(process.env.PATH)}))'

describe('spawnCapture', () => {
  it('does not hand the server credentials to the child', async () => {
    const before = { key: process.env.ANTHROPIC_API_KEY, token: process.env.CFD_AUTH_TOKEN }
    process.env.ANTHROPIC_API_KEY = 'sk-ant-must-not-appear'
    process.env.CFD_AUTH_TOKEN = 'token-must-not-appear'
    try {
      const res = await spawnCapture([process.execPath, '-e', PRINT_ENV], { cwd: os.tmpdir(), timeoutMs: 20_000 })
      expect(res.exitCode).toBe(0)
      expect(JSON.parse(res.stdout)).toEqual({ key: null, token: null, path: true })
      expect(res.stdout).not.toContain('must-not-appear')
    } finally {
      if (before.key === undefined) delete process.env.ANTHROPIC_API_KEY
      else process.env.ANTHROPIC_API_KEY = before.key
      if (before.token === undefined) delete process.env.CFD_AUTH_TOKEN
      else process.env.CFD_AUTH_TOKEN = before.token
    }
  })

  it('scrubs an env the caller supplied too', async () => {
    const res = await spawnCapture([process.execPath, '-e', PRINT_ENV], {
      cwd: os.tmpdir(),
      timeoutMs: 20_000,
      env: { ...process.env, ANTHROPIC_API_KEY: 'sk-ant-caller', CFD_AUTH_TOKEN: 'caller-token' },
    })
    expect(JSON.parse(res.stdout)).toEqual({ key: null, token: null, path: true })
  })
})
