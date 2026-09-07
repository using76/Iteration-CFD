import { describe, expect, it } from 'vitest'
import { scanTopLevelKeys } from './outline'

const CASE = `// meteor-cfd case — comment with "quotes" and { braces }
{
  "$schema": "https://meteor-cfd.msimul.com/schema/case-1.json",
  "name": "plumeB",

  "mesh": {
    "kind": "cartesian",
    "bounds": { "min": [-8.32, -2.62, 0.0], "max": [6.32, 3.62, 3.0] },
    /* block comment: "nested": 1 } */
    "cells":  [98, 42, 20],
  },
  "physics": { "gravity": [0, 0, -9.81] },
  "turbulence": { "kind": "RAS", "model": "kEpsilon" }, // trailing comment
  "patches": [ { "match": "inlet", "kind": "inlet" } ],
  "run": { "endTime": 400 },
  "iters": 400,
  "adjust": true,
  "note": null,
  "esc\\"aped": "x",
}
`

describe('scanTopLevelKeys', () => {
  it('lists the root keys of a commented JSONC case with positions and value kinds', () => {
    const keys = scanTopLevelKeys(CASE)
    expect(keys.map((k) => k.key)).toEqual(['$schema', 'name', 'mesh', 'physics', 'turbulence', 'patches', 'run', 'iters', 'adjust', 'note', 'esc"aped'])
    expect(keys[0]).toMatchObject({ line: 3, col: 3, valueKind: 'string' })
    expect(keys[2]).toMatchObject({ key: 'mesh', line: 6, valueKind: 'object' })
    expect(keys[5]).toMatchObject({ key: 'patches', valueKind: 'array' })
    expect(keys[7]).toMatchObject({ key: 'iters', valueKind: 'number' })
    expect(keys[8]).toMatchObject({ key: 'adjust', valueKind: 'boolean' })
    expect(keys[9]).toMatchObject({ key: 'note', valueKind: 'null' })
  })

  it('ignores nested keys, strings that contain colons and keys inside comments', () => {
    const keys = scanTopLevelKeys('{ "a": { "b": 1 }, "c": "x: y", // "d": 2\n "e": [ { "f": 3 } ] }')
    expect(keys.map((k) => k.key)).toEqual(['a', 'c', 'e'])
  })

  it('returns nothing for arrays, empty input and unterminated text', () => {
    expect(scanTopLevelKeys('[1, 2, 3]')).toEqual([])
    expect(scanTopLevelKeys('')).toEqual([])
    expect(scanTopLevelKeys('{ "a": ')).toEqual([{ key: 'a', line: 1, col: 3, valueKind: 'unknown' }])
    expect(scanTopLevelKeys('{ "a": "unterminated')).toHaveLength(1)
  })

  it('stops at the end of the root object', () => {
    expect(scanTopLevelKeys('{ "a": 1 } { "b": 2 }').map((k) => k.key)).toEqual(['a'])
  })
})
