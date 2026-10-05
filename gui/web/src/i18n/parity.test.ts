// Both string tables carry the same keys and placeholders in ko and en, and the campaign strings are translated.
import { describe, expect, it } from 'vitest'
import { STRING_TABLES } from '@cfd/shared'
import { EXTRA_TABLES, tx, type UiKey } from './extra'

const placeholders = (s: string): string[] => [...s.matchAll(/\{([^{}]+)\}/g)].map((m) => m[1]).sort()
const HANGUL = /[가-힣]/

describe('the web string tables', () => {
  it('same keys and placeholders in ko and en, shared and extra', () => {
    for (const table of [STRING_TABLES, EXTRA_TABLES]) {
      const ko = table.ko as Record<string, string>
      const en = table.en as Record<string, string>
      expect(Object.keys(ko).sort()).toEqual(Object.keys(en).sort())
      for (const key of Object.keys(ko)) {
        expect(ko[key].trim().length).toBeGreaterThan(0)
        expect(en[key].trim().length).toBeGreaterThan(0)
        expect(placeholders(ko[key])).toEqual(placeholders(en[key]))
      }
    }
  })

  it('the autonomy strings are Korean in ko and not in en', () => {
    const ko = EXTRA_TABLES.ko as Record<string, string>
    const en = EXTRA_TABLES.en as Record<string, string>
    const keys = Object.keys(en).filter((k) => k.startsWith('autonomy.'))
    expect(keys.length).toBe(62)
    for (const k of keys) {
      if (k !== 'autonomy.col.blc8') expect(HANGUL.test(ko[k])).toBe(true)
      expect(HANGUL.test(en[k])).toBe(false)
      expect(tx('ko', k as UiKey)).not.toBe(k)
      expect(tx('en', k as UiKey)).not.toBe(k)
    }
  })
})
