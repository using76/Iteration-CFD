import { describe, expect, it } from 'vitest'
import { clampLine, compileUserRegex, MAX_TESTED_LINE, starHeight, UnsafeRegexError } from './regex.js'

describe('starHeight', () => {
  it('is 1 for the patterns a search box produces', () => {
    for (const p of ['hello', 'fn \\w+\\(', '[a-z]+', 'a*b+c?', '(foo|bar)+', 'res\\s+[0-9.e-]+', '^\\s*written to', 'k\\s+res\\s+(\\S+)', '\\d{2,4}', 'a{1,3}?'])
      expect(starHeight(p)).toBeLessThanOrEqual(1)
  })

  it('is 2 or more for a quantifier nested in a quantified group', () => {
    expect(starHeight('(a+)+')).toBe(2)
    expect(starHeight('(a*)*')).toBe(2)
    expect(starHeight('(\\d+)*')).toBe(2)
    expect(starHeight('([a-z]+)+$')).toBe(2)
    expect(starHeight('(x(y+)*)+')).toBe(3)
    expect(starHeight('(?:a+)+')).toBe(2)
    expect(starHeight('(a+){2,}')).toBe(2)
  })

  it('does not count a quantifier inside a group nobody quantifies', () => {
    expect(starHeight('(a+)b')).toBe(1)
    expect(starHeight('(a+)(b+)')).toBe(1)
  })
})

describe('compileUserRegex', () => {
  it('compiles ordinary patterns', () => {
    expect(compileUserRegex('fn \\w+', 'i').test('fn main')).toBe(true)
    expect(compileUserRegex('(foo|bar)+', 'gi')).toBeInstanceOf(RegExp)
  })

  it('refuses the pattern that would pin the event loop', () => {
    // Left unguarded this takes longer than the process will live:
    // /(a+)+$/.test('a'.repeat(40) + '!')
    expect(() => compileUserRegex('(a+)+$', 'i')).toThrow(UnsafeRegexError)
    expect(() => compileUserRegex('(a+)+$', 'i')).toThrow(/star height 2/)
    expect(() => compileUserRegex('a'.repeat(1001), 'i')).toThrow(UnsafeRegexError)
  })

  it('still reports a pattern that is not a regex at all', () => {
    expect(() => compileUserRegex('(', 'i')).toThrow(SyntaxError)
  })
})

describe('clampLine', () => {
  it('bounds what a pattern is asked to match', () => {
    expect(clampLine('short')).toBe('short')
    expect(clampLine('x'.repeat(MAX_TESTED_LINE + 500))).toHaveLength(MAX_TESTED_LINE)
  })
})
