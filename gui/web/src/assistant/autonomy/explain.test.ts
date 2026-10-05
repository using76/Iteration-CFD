// The narration port proven against explain.py's own output: the golden texts and JSON, and every det_a geometry.
import { describe, expect, it } from 'vitest'
import rowsRaw from './fixtures/explain/rows.jsonl?raw'
import recordsRaw from './fixtures/explain/records.json?raw'
import detAExplainRaw from './fixtures/det_a_explain.json?raw'
import goldenBoxJson from './fixtures/explain/golden/box_sphere.json?raw'
import goldenBoxTxt from './fixtures/explain/golden/box_sphere.txt?raw'
import goldenWingJson from './fixtures/explain/golden/wing_a_L3.json?raw'
import goldenWingTxt from './fixtures/explain/golden/wing_a_L3.txt?raw'
import goldenDJson from './fixtures/explain/golden/D-1-002.json?raw'
import goldenDTxt from './fixtures/explain/golden/D-1-002.txt?raw'
import detAAttemptsRaw from '../../../../server/src/tools/fixtures/autonomy/det_a/attempts.jsonl?raw'
import detAGeometriesRaw from '../../../../server/src/tools/fixtures/autonomy/det_a/geometries.jsonl?raw'
import { editsText, explainGeometry, fmtAt, fmtPy, ID_RE, isPyFloat, parsePyJson, parsePyJsonl, TEMPLATES, triggerText } from './explain'
import type { Json, TaggedRecord } from './explain'

const rows = parsePyJsonl(rowsRaw).rows
const detARows = parsePyJsonl(detAAttemptsRaw).rows
const detAGeoms = parsePyJsonl(detAGeometriesRaw).rows
const goldenRecords = (parsePyJson(recordsRaw) as Json).records as unknown as Record<string, TaggedRecord[]>
const detAExplain = parsePyJson(detAExplainRaw) as Json

const rowsOf = (gid: string, source: Json[] = rows): Json[] => source.filter((r) => r.geometry_id === gid)
const norm = (s: string): string => s.replace(/\r\n/g, '\n')
const endRecordsOf = (gid: string): TaggedRecord[] => {
  const end = detAGeoms.find((g) => g.geometry_id === gid)
  return end !== undefined && Array.isArray(end.records) ? (end.records as unknown as TaggedRecord[]) : []
}

describe('explain', () => {
  it('fmtPy is Python fmt', () => {
    expect(fmtPy(0.1334231805929919)).toBe('0.133423')
    expect(fmtPy(0)).toBe('0')
    expect(fmtPy(1e-5)).toBe('1e-05')
    expect(fmtPy(0.0001)).toBe('0.0001')
    expect(fmtPy(585900)).toBe('585900')
    expect(fmtPy(126.41912)).toBe('126.419')
    expect(fmtPy(0.000123456789)).toBe('0.000123457')
    expect(fmtPy(123456.7)).toBe('123457')
    expect(fmtPy(1234567.8)).toBe('1.23457e+06')
    expect(fmtPy(999999.5)).toBe('1e+06')
    expect(fmtPy(0.00999999)).toBe('0.00999999')
    expect(fmtPy(5.719686900032684)).toBe('5.71969')
    expect(fmtPy(-2.5e-7)).toBe('-2.5e-07')
    expect(fmtPy(Infinity)).toBe('inf')
    expect(fmtPy(-Infinity)).toBe('-inf')
    expect(fmtPy(NaN)).toBe('nan')
    expect(fmtPy(1e22, true)).toBe('1e+22')
    expect(fmtPy(1400000, true)).toBe('1.4e+06')
    expect(fmtPy(1400000)).toBe('1400000')
    expect(fmtPy(-0, true)).toBe('-0')
    expect(fmtPy(null)).toBe('absent')
    expect(fmtPy(undefined)).toBe('absent')
    expect(fmtPy(true)).toBe('true')
    expect(fmtPy('min_thickness')).toBe('"min_thickness"')
    expect(fmtPy('a§b\x7f"')).toBe('"a\\u00a7b\\u007f\\""')
    expect(fmtPy([0.98, 1.02])).toBe('[0.98, 1.02]')
    expect(fmtPy([])).toBe('[]')
  })

  it('tells an integral float from an int by the JSON source text', () => {
    const v = parsePyJson('{"t":{"observable":"predicted_cells","op":"<=","value":1480716,"threshold":1400000.0,"source":"s"},"l":[1.0,2],"z":-0.0}') as any
    expect(isPyFloat(v.t, 'threshold')).toBe(true)
    expect(isPyFloat(v.t, 'value')).toBe(false)
    expect(isPyFloat(v.l, 0)).toBe(true)
    expect(isPyFloat(v.l, 1)).toBe(false)
    expect(fmtAt(v.t, 'threshold')).toBe('1.4e+06')
    expect(fmtAt(v, 'z')).toBe('-0')
    expect(fmtPy(v.l)).toBe('[1, 2]')
    expect(triggerText(v.t)).toBe('predicted_cells = 1480716 <= 1.4e+06 (s)')
    expect(editsText([])).toBe('none')
    const parsed = parsePyJsonl('{"a":1}\n\nnot json\n[1]\n{"b":2')
    expect(parsed.rows).toEqual([{ a: 1 }])
    expect(parsed.badLines).toEqual([3, 4, 5])
  })

  it('reproduces the three golden geometries byte for byte', () => {
    const cases: Array<[string, string, string]> = [
      ['box_sphere', goldenBoxJson, goldenBoxTxt],
      ['wing_a_L3', goldenWingJson, goldenWingTxt],
      ['D-1-002', goldenDJson, goldenDTxt],
    ]
    for (const [gid, goldenJson, goldenTxt] of cases) {
      const res = explainGeometry(rowsOf(gid), goldenRecords[gid], TEMPLATES)
      expect(res.ok).toBe(true)
      if (res.ok) {
        expect(res.value).toEqual(parsePyJson(goldenJson))
        expect(res.value.text).toBe(norm(goldenTxt))
      }
    }
    const wing = explainGeometry(rowsOf('wing_a_L3'), goldenRecords['wing_a_L3'], TEMPLATES)
    expect(wing.ok).toBe(true)
    if (wing.ok) {
      expect(wing.value.terminal).toBe('EXHAUSTED')
      expect(wing.value.attempts[1]?.moved_text).toBe('outcome.pinned_frac 0.133423 -> 0 (attempt 1 -> 2)')
    }
  })

  it('reproduces the 17 det_a geometries', () => {
    const gids = Object.keys(detAExplain.geometries as Json)
    expect(gids.length).toBe(17)
    for (const gid of gids) {
      const res = explainGeometry(rowsOf(gid, detARows), endRecordsOf(gid), TEMPLATES)
      expect(res.ok).toBe(true)
      if (res.ok) expect(res.value).toEqual((detAExplain.geometries as Json)[gid])
    }
    const b = explainGeometry(rowsOf('B-1-011', detARows), endRecordsOf('B-1-011'), TEMPLATES)
    expect(b.ok).toBe(true)
    if (b.ok) expect(b.value.attempts[0]?.why_text.startsWith('predicted_cells = 1480716 <= 1.4e+06')).toBe(true)
  })

  it('refuses by name, never a throw', () => {
    expect(explainGeometry([], null)).toEqual({ ok: false, error: 'no rows to explain' })
    const wingNo2 = rowsOf('wing_a_L3').filter((r) => r.attempt !== 2)
    expect(explainGeometry(wingNo2, null)).toEqual({ ok: false, error: 'the attempts of wing_a_L3 are [1, 3, 4], not 1..3' })
    const box = rowsOf('box_sphere')
    const mixed = [...box, { ...box[0], campaign_id: 'other', attempt: 2 }]
    const m = explainGeometry(mixed, null)
    expect(m.ok).toBe(false)
    if (!m.ok) expect(m.error.startsWith('mixed geometry_id or campaign_id')).toBe(true)
    const rec9 = [{ attempt: 9, record: goldenRecords['box_sphere'][0].record }]
    expect(explainGeometry(box, rec9)).toEqual({ ok: false, error: 'a record is tagged attempt 9, which box_sphere has no row for' })
    const recNope = [{ attempt: 1, record: { ...goldenRecords['box_sphere'][0].record, rule_id: 'R-NOPE' } }]
    expect(explainGeometry(box, recNope)).toEqual({ ok: false, error: "no template for rule id 'R-NOPE' (TEMPLATES has 41 ids)" })
    const recLayer = [{ attempt: 1, record: { ...goldenRecords['box_sphere'][0].record, layer: 'rule' } }]
    const lay = explainGeometry(box, recLayer)
    expect(lay.ok).toBe(false)
    if (!lay.ok) expect(lay.error.startsWith("rule id RM-CAPABILITY-LIMITED is a remedy template")).toBe(true)
    expect(explainGeometry([{ attempt: 1 }], null).ok).toBe(false)
  })

  it('the templates table', () => {
    const ids = Object.keys(TEMPLATES.templates)
    expect(ids.length).toBe(41)
    for (const id of ids) {
      expect(ID_RE.test(id)).toBe(true)
      expect(['preflight', 'rule', 'remedy']).toContain(TEMPLATES.templates[id].layer)
      expect(/\d/.test(TEMPLATES.templates[id].title)).toBe(false)
      expect(/\d/.test(TEMPLATES.templates[id].because)).toBe(false)
    }
    expect(TEMPLATES.flag_order).toEqual(['F1', 'F2', 'F3a', 'F3b', 'F3c', 'F3d', 'F4', 'F5'])
    expect(TEMPLATES.terminal_of).toEqual({
      'RM-PASS': 'PASS',
      'RM-CAPABILITY-LIMITED': 'CAPABILITY-LIMITED',
      'RM-EXHAUSTED': 'EXHAUSTED',
      'RM-NO-REMEDY': 'NO-REMEDY',
    })
  })
})
