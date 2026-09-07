// Regular expressions that arrive from outside the server: the model's
// `file_search regex=true` and `run_log grep`, and the REST search routes a
// browser can call. Node has no way to abort a running match, so one pattern
// with catastrophic backtracking - `(a+)+$` against a line of a's - pins the
// event loop and the whole server stops answering. The defence has to happen
// before the pattern is ever compiled.

/** Longest pattern accepted; a legitimate search does not need more. */
export const MAX_PATTERN_LENGTH = 1000
/** Longest line handed to a user pattern. Backtracking cost grows with input. */
export const MAX_TESTED_LINE = 4000

export class UnsafeRegexError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'UnsafeRegexError'
  }
}

/** Index just past a quantifier at `i`, or -1 when there is none. */
function quantifierEnd(source: string, i: number): number {
  const c = source[i]
  if (c === undefined) return -1
  if (c === '*' || c === '+' || c === '?') return source[i + 1] === '?' ? i + 2 : i + 1
  if (c === '{') {
    const m = /^\{\d+(?:,\d*)?\}/.exec(source.slice(i))
    if (m) return source[i + m[0].length] === '?' ? i + m[0].length + 1 : i + m[0].length
  }
  return -1
}

/**
 * Star height: how deeply quantifiers nest. `a+` and `(foo|bar)+` are height 1
 * and match in linear time; `(a+)+` and `(\d+)*` are height 2, which is the
 * shape whose backtracking is exponential in the length of the input. This is
 * a syntactic measure, not a decision procedure - it does not catch an
 * overlapping alternation like `(a|ab)+` - but it catches the shape a search
 * box actually produces by accident, and it never rejects a linear pattern.
 */
export function starHeight(source: string): number {
  let max = 0
  const stack: Array<{ inner: number }> = [{ inner: 0 }]
  const bump = (h: number): void => {
    if (h > max) max = h
    const top = stack[stack.length - 1]
    if (h > top.inner) top.inner = h
  }
  let i = 0
  while (i < source.length) {
    const c = source[i]
    if (c === '(') {
      stack.push({ inner: 0 })
      i++
      continue
    }
    if (c === ')') {
      const done = stack.length > 1 ? stack.pop()! : { inner: 0 }
      i++
      const q = quantifierEnd(source, i)
      if (q >= 0) {
        bump(done.inner + 1)
        i = q
      } else {
        const top = stack[stack.length - 1]
        if (done.inner > top.inner) top.inner = done.inner
      }
      continue
    }
    if (c === '|' || c === '^' || c === '$') {
      i++
      continue
    }
    if (c === '\\') i += 2
    else if (c === '[') {
      i++
      while (i < source.length && source[i] !== ']') i += source[i] === '\\' ? 2 : 1
      i++
    } else i++
    const q = quantifierEnd(source, i)
    if (q >= 0) {
      bump(1)
      i = q
    }
  }
  return max
}

/**
 * Compile a pattern that came from the model or the browser. Throws
 * UnsafeRegexError for one this server will not run and SyntaxError for one
 * that is not a regular expression at all.
 */
export function compileUserRegex(source: string, flags: string): RegExp {
  if (source.length > MAX_PATTERN_LENGTH) throw new UnsafeRegexError(`pattern is ${source.length} characters; the limit is ${MAX_PATTERN_LENGTH}`)
  const height = starHeight(source)
  if (height > 1) throw new UnsafeRegexError('pattern nests one quantifier inside another (star height ' + height + '), which can take exponential time to fail; rewrite it without the nesting')
  return new RegExp(source, flags)
}

/** A line short enough to hand to a user pattern. */
export function clampLine(line: string): string {
  return line.length > MAX_TESTED_LINE ? line.slice(0, MAX_TESTED_LINE) : line
}
