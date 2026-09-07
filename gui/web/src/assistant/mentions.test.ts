import { describe, expect, it } from 'vitest'
import { activeMention, completeMention, matchPaths, parseMentions } from './mentions'

describe('parseMentions', () => {
  it('extracts @paths, strips trailing punctuation and dedupes', () => {
    expect(parseMentions('Check @cases/plume.jsonc and @rust/src/bin/k_epsilon.rs.')).toEqual(['cases/plume.jsonc', 'rust/src/bin/k_epsilon.rs'])
    expect(parseMentions('(@cases/plume.jsonc), again @cases/plume.jsonc!')).toEqual(['cases/plume.jsonc'])
    expect(parseMentions('@cases/plume_jsonc/400 has @docs/01-intro.md; also @a')).toEqual(['cases/plume_jsonc/400', 'docs/01-intro.md', 'a'])
  })
  it('ignores emails and bare @', () => {
    expect(parseMentions('mail me@example.com or @ nothing')).toEqual([])
    expect(parseMentions('no mentions here')).toEqual([])
  })
})

describe('activeMention', () => {
  it('finds the mention under the caret', () => {
    expect(activeMention('open @cas', 9)).toEqual({ start: 5, query: 'cas' })
    expect(activeMention('open @', 6)).toEqual({ start: 5, query: '' })
    expect(activeMention('open @cases/plume.jsonc now', 23)).toEqual({ start: 5, query: 'cases/plume.jsonc' })
  })
  it('returns null after whitespace, inside words and for emails', () => {
    expect(activeMention('open @cases now', 15)).toBeNull()
    expect(activeMention('me@ex', 5)).toBeNull()
    expect(activeMention('plain text', 5)).toBeNull()
  })
})

describe('completeMention', () => {
  it('replaces the partial mention and places the caret after it', () => {
    const r = completeMention('open @cas please', 9, 5, 'cases/plume.jsonc')
    expect(r.text).toBe('open @cases/plume.jsonc  please')
    expect(r.caret).toBe(5 + '@cases/plume.jsonc '.length)
  })
})

describe('matchPaths', () => {
  const paths = ['README.md', 'cases/plume.jsonc', 'cases/channelPeriodicWF.jsonc', 'rust/src/bin/plume.rs', 'docs/01-intro.md']
  it('ranks basename prefix matches first, then path prefix, then substrings', () => {
    expect(matchPaths(paths, 'plu')).toEqual(['cases/plume.jsonc', 'rust/src/bin/plume.rs'])
    expect(matchPaths(paths, 'cases/ch')).toEqual(['cases/channelPeriodicWF.jsonc'])
    expect(matchPaths(paths, 'intro')).toEqual(['docs/01-intro.md'])
    expect(matchPaths(paths, 'zzz')).toEqual([])
  })
  it('returns the first entries for an empty query and honours the limit', () => {
    expect(matchPaths(paths, '', 2)).toEqual(['README.md', 'cases/plume.jsonc'])
  })
})
