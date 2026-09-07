import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, test } from 'vitest'
import { fmtG, formatTimeName, parseFoamFieldText, readFoamField, readFoamFieldHeader, writeFoamField } from './foam.js'
import { FoamScanner, parseFoamNumber, type FoamToken } from './foamtok.js'

let dir: string
beforeAll(async () => {
  dir = await fs.mkdtemp(path.join(os.tmpdir(), 'cfd-foam-'))
})
afterAll(async () => {
  await fs.rm(dir, { recursive: true, force: true })
})

// render_plume.py's own regexes, ported verbatim (re.S -> [\s\S]).
const PLUME_NONUNIFORM = /internalField\s+nonuniform\s+List<scalar>\s*\n?(\d+)\s*\n\(([\s\S]*?)\n\)/
const PLUME_UNIFORM = /internalField\s+uniform\s+([-\d.eE+]+)\s*;/

describe('fmtG / formatTimeName', () => {
  test('fmtG matches fields.rs fmt_g', () => {
    expect(fmtG(0)).toBe('0')
    expect(fmtG(-0)).toBe('-0')
    expect(fmtG(1e-5)).toBe('1e-05')
    expect(fmtG(293.15)).toBe('293.15')
    expect(fmtG(0.1)).toBe('0.1')
    expect(fmtG(1 / 3, 6)).toBe('0.333333')
    expect(fmtG(123456789012345, 12)).toBe('1.23456789012e+14')
    expect(fmtG(1e21)).toBe('1e+21')
    expect(fmtG(-2.5e-13, 3)).toBe('-2.5e-13')
    expect(fmtG(NaN)).toBe('nan')
    expect(fmtG(-Infinity)).toBe('-inf')
    expect(fmtG(0.1, 17)).toBe('0.10000000000000001')
    expect(fmtG(0.5, 17)).toBe('0.5')
    expect(fmtG(1000, 6)).toBe('1000')
    expect(fmtG(9.9999995e2, 6)).toBe('1000')
  })

  test('formatTimeName reproduces case.rs format_time_name', () => {
    expect(formatTimeName(1000)).toBe('1000')
    expect(formatTimeName(0.5)).toBe('0.5')
    expect(formatTimeName(1e-5)).toBe('1e-05')
    expect(formatTimeName(1)).toBe('1')
    expect(formatTimeName(0.001)).toBe('0.001')
    expect(formatTimeName(1e7)).toBe('1e+07')
    expect(formatTimeName(0)).toBe('0')
    expect(formatTimeName(4000)).toBe('4000')
    expect(formatTimeName(0.123456789)).toBe('0.123457')
    expect(formatTimeName(999999.7)).toBe('1e+06')
    expect(formatTimeName(-0.25)).toBe('-0.25')
  })
})

describe('number parsing', () => {
  const num = (s: string): number => parseFoamNumber(Buffer.from(s, 'latin1'), 0, s.length)
  test('spellings', () => {
    expect(num('1e-05')).toBe(1e-5)
    expect(Object.is(num('-0'), -0)).toBe(true)
    expect(num('nan')).toBeNaN()
    expect(num('-nan')).toBeNaN()
    expect(num('inf')).toBe(Infinity)
    expect(num('-inf')).toBe(-Infinity)
    expect(num('293.15')).toBe(293.15)
    expect(num('.5')).toBe(0.5)
    expect(num('5.')).toBe(5)
    expect(num('+2')).toBe(2)
    expect(num('1E3')).toBe(1000)
    expect(num('0.10000000000000001')).toBe(0.1)
    expect(num('12345678901234567890')).toBe(12345678901234567890)
    expect(num('1e400')).toBe(Infinity)
    expect(num('4.9e-324')).toBe(5e-324)
    expect(num('0.00234375')).toBe(0.00234375)
    expect(() => num('abc')).toThrow()
  })
  test('random round trips are exact', () => {
    let seed = 12345
    const rnd = (): number => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff
      return seed / 0x7fffffff
    }
    for (let i = 0; i < 2000; i++) {
      const v = (rnd() - 0.5) * 10 ** Math.floor(rnd() * 20 - 10)
      for (const s of [String(v), v.toExponential(11), fmtG(v, 12), fmtG(v, 17)]) {
        expect(num(s)).toBe(Number(s))
      }
    }
  })
})

describe('scanner', () => {
  test('byte-by-byte feeding gives the same tokens and numbers as one chunk', () => {
    const text = '/* banner */ FoamFile { a "x/y"; } // c\nlist 3(1e-05 -0 /* mid */ nan) ;\n"quoted name" { type wall; }'
    const run = (step: number): string[] => {
      const out: string[] = []
      const sc = new FoamScanner((t: FoamToken) => {
        out.push(`${t.kind}:${t.text}`)
        if (t.kind === 'punct' && t.text === '(') sc.beginList({ push: (v, d) => out.push(`num:${v}@${d}`) })
      })
      const buf = Buffer.from(text, 'latin1')
      for (let i = 0; i < buf.length; i += step) sc.feed(buf.subarray(i, Math.min(buf.length, i + step)))
      sc.finish()
      return out
    }
    const whole = run(10_000)
    expect(whole).toEqual(run(1))
    expect(whole).toEqual(run(7))
    expect(whole).toContain('num:0.00001@1')
    expect(whole).toContain('num:NaN@1')
    expect(whole).toContain('string:x/y')
    expect(whole).toContain('string:quoted name')
    expect(whole).not.toContain('word:banner')
  })
})

const SAMPLE = `/*--------------------------------*- C++ -*----------------------------------*\\
| banner // with a fake comment and a "quote                                  |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       volScalarField;
    location    "0/x";
    object      p;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -2 0 0 0 0];

internalField   nonuniform List<scalar>
5
(
1e-05
-0
nan
3.5 /* inline */ 4e2
)
;

boundaryField
{
    "wall.*"
    {
        type            zeroGradient;
    }
    inlet
    {
        type            fixedValue;
        value           nonuniform List<scalar> 2(1 2);
    }
    outlet
    {
        type            fixedValue;
        value           uniform 0;
        inGroups        1(outlets);
    }
}
`

describe('readFoamField', () => {
  test('parses comments, exotic numbers and quoted patch names', () => {
    const f = parseFoamFieldText(SAMPLE, 'p')
    expect(f.header.class).toBe('volScalarField')
    expect(f.header.location).toBe('0/x')
    expect(f.header.object).toBe('p')
    expect(f.header.dimensions).toBe('[0 2 -2 0 0 0 0]')
    expect(f.components).toBe(1)
    expect(f.count).toBe(5)
    expect(f.uniform).toBe(false)
    expect(f.data[0]).toBeCloseTo(1e-5, 10)
    expect(Object.is(f.data[1], -0)).toBe(true)
    expect(f.data[2]).toBeNaN()
    expect(f.data[3]).toBe(3.5)
    expect(f.data[4]).toBe(400)
    expect(f.patches).toEqual([
      { name: 'wall.*', type: 'zeroGradient' },
      { name: 'inlet', type: 'fixedValue' },
      { name: 'outlet', type: 'fixedValue' },
    ])
  })

  test('uniform scalar / vector expands to nCells', () => {
    const s = parseFoamFieldText('FoamFile { class volScalarField; object k; }\ninternalField uniform 0.375;\nboundaryField { }', 'k', { nCells: 4 })
    expect(s.uniform).toBe(true)
    expect(s.uniformValue).toEqual([0.375])
    expect(Array.from(s.data)).toEqual([0.375, 0.375, 0.375, 0.375])
    const v = parseFoamFieldText('FoamFile { class volVectorField; object U; }\ninternalField uniform (1 2 3);\nboundaryField {}', 'U', { nCells: 2 })
    expect(v.components).toBe(3)
    expect(v.count).toBe(2)
    expect(Array.from(v.data)).toEqual([1, 2, 3, 1, 2, 3])
    const one = parseFoamFieldText('internalField uniform 7;', 'x')
    expect(one.count).toBe(1)
    expect(one.header.class).toBe('')
  })

  test('nonuniform 0(), missing List<scalar>, compact N{v}, bare lists and vectors', () => {
    expect(parseFoamFieldText('internalField nonuniform 0();', 'e').count).toBe(0)
    const noType = parseFoamFieldText('internalField nonuniform 2(1 2);', 'a')
    expect(Array.from(noType.data)).toEqual([1, 2])
    const compact = parseFoamFieldText('internalField 3{2.5};', 'a')
    expect(Array.from(compact.data)).toEqual([2.5, 2.5, 2.5])
    const bare = parseFoamFieldText('internalField (1 2 3 4);', 'a')
    expect(Array.from(bare.data)).toEqual([1, 2, 3, 4])
    const sized = parseFoamFieldText('internalField 2 (5 6);', 'a')
    expect(Array.from(sized.data)).toEqual([5, 6])
    const vec = parseFoamFieldText('internalField nonuniform List<vector> \n2\n(\n(1 2 3)\n(4 5 6)\n)\n;', 'U')
    expect(vec.components).toBe(3)
    expect(vec.count).toBe(2)
    expect(Array.from(vec.data)).toEqual([1, 2, 3, 4, 5, 6])
    const vecNoType = parseFoamFieldText('internalField nonuniform 2((1 2 3) (4 5 6));', 'U')
    expect(vecNoType.components).toBe(3)
    expect(vecNoType.count).toBe(2)
  })

  test('declared count mismatch is an error', () => {
    expect(() => parseFoamFieldText('internalField nonuniform List<scalar> 3(1 2);', 'a')).toThrow(/declares 3/)
  })

  test('header-only read stops before the list and reports the class', async () => {
    const p = path.join(dir, 'phi')
    await fs.writeFile(p, 'FoamFile\n{\n    class       surfaceScalarField;\n    location    "100";\n    object      phi;\n}\ndimensions [0 3 -1 0 0 0 0];\ninternalField nonuniform List<scalar> 3(1 2 3);\n')
    const h = await readFoamFieldHeader(p)
    expect(h.class).toBe('surfaceScalarField')
    expect(h.location).toBe('100')
    expect(h.object).toBe('phi')
    expect(h.dimensions).toBe('[0 3 -1 0 0 0 0]')
    const q = path.join(dir, 'notes.txt')
    await fs.writeFile(q, 'just some text\nwith lines\n')
    expect((await readFoamFieldHeader(q)).class).toBe('')
  })
})

describe('writeFoamField round trips', () => {
  test('scalar nonuniform, 12 significant digits, render_plume regex', async () => {
    const n = 1000
    const data = new Float64Array(n)
    for (let i = 0; i < n; i++) data[i] = Math.sin(i) * 10 ** ((i % 9) - 4)
    const p = path.join(dir, '100', 'T')
    await writeFoamField(p, { name: 'T', class: 'volScalarField', dimensions: '[0 0 0 1 0 0 0]', time: '100', data, components: 1, patches: [{ name: 'wall', type: 'zeroGradient' }, { name: 'inlet', type: 'fixedValue', entries: { value: 'uniform 300' } }] })
    const text = await fs.readFile(p, 'utf8')
    expect(text.startsWith('FoamFile\n{\n    format      ascii;\n    class       volScalarField;\n    location    "100";\n    object      T;\n}\n')).toBe(true)
    expect(text).toContain('dimensions      [0 0 0 1 0 0 0];\n\ninternalField   nonuniform List<scalar> \n1000\n(\n')
    expect(text).toContain('\n)\n;\n\nboundaryField\n{\n    wall\n    {\n        type            zeroGradient;\n    }\n    inlet\n    {\n        type            fixedValue;\n        value           uniform 300;\n    }\n}\n')
    expect(text.endsWith('// ************************************************************************* //\n')).toBe(true)
    const m = PLUME_NONUNIFORM.exec(text)
    expect(m).not.toBeNull()
    expect(Number(m![1])).toBe(n)
    const values = m![2].trim().split(/\s+/).map(Number)
    expect(values.length).toBe(n)
    for (let i = 0; i < n; i++) expect(Math.abs(values[i] - data[i])).toBeLessThanOrEqual(Math.abs(data[i]) * 1e-11)
    const back = await readFoamField(p)
    expect(back.count).toBe(n)
    expect(back.header.object).toBe('T')
    expect(back.patches.map((x) => x.name)).toEqual(['wall', 'inlet'])
    for (let i = 0; i < n; i++) expect(back.data[i]).toBe(Math.fround(values[i]))
  })

  test('vector, uniform collapse, empty, quoted patch names', async () => {
    const U = Float64Array.from([1, 2, 3, -1e-5, 0, 2.5])
    const pU = path.join(dir, '0', 'U')
    await writeFoamField(pU, { name: 'U', class: 'volVectorField', dimensions: '[0 1 -1 0 0 0 0]', time: '0', data: U, components: 3, patches: [{ name: 'wall.*', type: 'noSlip' }] })
    const textU = await fs.readFile(pU, 'utf8')
    expect(textU).toContain('internalField   nonuniform List<vector> \n2\n(\n(1 2 3)\n(-1e-05 0 2.5)\n)\n;\n')
    expect(textU).toContain('    "wall.*"\n    {\n        type            noSlip;\n    }\n')
    const backU = await readFoamField(pU)
    expect(backU.components).toBe(3)
    expect(Array.from(backU.data)).toEqual(Array.from(U, Math.fround))
    expect(backU.patches[0].name).toBe('wall.*')

    const pk = path.join(dir, '0', 'k')
    await writeFoamField(pk, { name: 'k', class: 'volScalarField', dimensions: '[0 2 -2 0 0 0 0]', time: '0', data: Float32Array.from([0.375, 0.375, 0.375]), components: 1, patches: [] })
    const textK = await fs.readFile(pk, 'utf8')
    expect(textK).toContain('internalField   uniform 0.375;\n')
    expect(PLUME_UNIFORM.exec(textK)![1]).toBe('0.375')
    const backK = await readFoamField(pk, { nCells: 3 })
    expect(backK.uniform).toBe(true)
    expect(Array.from(backK.data)).toEqual([0.375, 0.375, 0.375])
    const keep = path.join(dir, '0', 'k2')
    await writeFoamField(keep, { name: 'k2', class: 'volScalarField', dimensions: '', time: '0', data: Float32Array.from([1, 1]), components: 1, patches: [], collapseUniform: false })
    expect(await fs.readFile(keep, 'utf8')).toContain('dimensions      [0 0 0 0 0 0 0];\n\ninternalField   nonuniform List<scalar> \n2\n(\n1\n1\n)\n;')

    const pe = path.join(dir, '0', 'empty')
    await writeFoamField(pe, { name: 'empty', class: 'volScalarField', dimensions: '[0 0 0 0 0 0 0]', time: '0', data: new Float32Array(0), components: 1, patches: [] })
    expect(await fs.readFile(pe, 'utf8')).toContain('internalField   nonuniform 0();\n')
    expect((await readFoamField(pe)).count).toBe(0)
  })

  test('a field larger than one read chunk survives the chunk boundaries', async () => {
    const n = 120_000
    const data = new Float32Array(n)
    for (let i = 0; i < n; i++) data[i] = Math.fround((i - 60_000) * 1.2345e-3)
    const p = path.join(dir, 'big', 'p')
    await writeFoamField(p, { name: 'p', class: 'volScalarField', dimensions: '[0 2 -2 0 0 0 0]', time: 'big', data, components: 1, patches: [] })
    expect((await fs.stat(p)).size).toBeGreaterThan(1 << 20)
    const back = await readFoamField(p)
    expect(back.count).toBe(n)
    for (let i = 0; i < n; i += 997) expect(back.data[i]).toBe(data[i])
    expect(back.data[n - 1]).toBe(data[n - 1])
  })
})
