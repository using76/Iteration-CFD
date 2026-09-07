// The environment a child process is allowed to see. PLAN §10: API keys are
// held only by the server, and a child's output goes to the model, then into
// the session file on disk - `env` in an error message or a `set`-alike
// command is enough to put a key in both.
const SECRET_NAMES = new Set(['ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_ADMIN_KEY', 'CFD_AUTH_TOKEN'])
const SECRET_SUFFIX = /(?:_KEY|_TOKEN|_SECRET|_SECRET_KEY|_PASSWORD|_PASSWD|_CREDENTIALS)$/i

export function isSecretEnvName(name: string): boolean {
  return SECRET_NAMES.has(name.toUpperCase()) || SECRET_SUFFIX.test(name)
}

/** A copy of `env` with every credential-shaped name removed. */
export function scrubbedEnv(env: NodeJS.ProcessEnv = process.env): NodeJS.ProcessEnv {
  const out: NodeJS.ProcessEnv = {}
  for (const [name, value] of Object.entries(env)) {
    if (value === undefined || isSecretEnvName(name)) continue
    out[name] = value
  }
  return out
}
