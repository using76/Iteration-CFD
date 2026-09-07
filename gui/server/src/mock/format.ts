// Ports of rust/src/bin/common/mod.rs `g` and `sci`, so the mock prints
// numbers in exactly the shape the real drivers do.

const trimmed = (s: string) => (s.includes('.') ? s.replace(/0+$/, '').replace(/\.$/, '') : s)

/** `%g`-like with `prec` significant digits ("0", "1.5e-05", "293.15"). */
export function g(x: number, prec = 6): string {
  if (x === 0) return '0'
  if (Number.isNaN(x)) return 'nan'
  if (!Number.isFinite(x)) return x > 0 ? 'inf' : '-inf'
  let exp = Math.floor(Math.log10(Math.abs(x)))
  if ((Math.abs(x) / 10 ** exp).toFixed(prec - 1).startsWith('10')) exp += 1
  if (exp < -4 || exp >= prec) {
    const mantissa = trimmed((x / 10 ** exp).toFixed(prec - 1))
    const sign = exp < 0 ? '-' : '+'
    return `${mantissa}e${sign}${String(Math.abs(exp)).padStart(2, '0')}`
  }
  return trimmed(x.toFixed(Math.max(prec - 1 - exp, 0)))
}

/** `%.*e`: mantissa with `prec` decimals, exponent with a sign and at least two digits ("1.234e-03"). */
export function sci(x: number, prec = 3): string {
  if (!Number.isFinite(x)) return g(x)
  const [m, e] = x.toExponential(prec).split('e')
  const sign = e.startsWith('-') ? '-' : '+'
  return `${m}e${sign}${e.replace(/^[-+]/, '').padStart(2, '0')}`
}

/** Deterministic noise in [-1, 1] (a small LCG) so mock runs are reproducible. */
export function noiseSource(seed = 12345): () => number {
  let s = seed >>> 0
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0
    return (s / 0xffffffff) * 2 - 1
  }
}

export const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms))
