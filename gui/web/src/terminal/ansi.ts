// Light colouring for solver output written into xterm.
const RESET = '\x1b[0m'
const DIM = '\x1b[2m'
const GREEN = '\x1b[32m'
const RED = '\x1b[31m'
const YELLOW = '\x1b[33m'
const BOLD_BLUE = '\x1b[1;34m'

const NUMBER = /(?<![\w.])([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)(?![\w.])/g

export function colourLogLine(text: string, stream: 'stdout' | 'stderr' | 'system'): string {
  const clean = text.replace(/\r$/, '')
  if (stream === 'system') return `${BOLD_BLUE}${clean}${RESET}`
  if (stream === 'stderr' || /^\s*(error|ofgpu-[\w-]+|benchmark aborted):/i.test(clean) || /\*\*\* NaN\/Inf \*\*\*/.test(clean)) return `${RED}${clean}${RESET}`
  if (/^\s*converged\b/i.test(clean) || /\bconverged\b/i.test(clean)) return `${GREEN}${clean}${RESET}`
  if (/^\s*WARNING\b/i.test(clean)) return `${YELLOW}${clean}${RESET}`
  return clean.replace(NUMBER, `${DIM}$1${RESET}`)
}

export function commandEcho(argv: string[]): string {
  return `${BOLD_BLUE}$ ${argv.join(' ')}${RESET}`
}
