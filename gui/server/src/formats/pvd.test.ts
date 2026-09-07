import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { afterAll, beforeAll, expect, test } from 'vitest'
import { formatPvd, parsePvd, readPvd, writePvd } from './pvd.js'

let dir: string
beforeAll(async () => {
  dir = await fs.mkdtemp(path.join(os.tmpdir(), 'cfd-pvd-'))
})
afterAll(async () => {
  await fs.rm(dir, { recursive: true, force: true })
})

test('pvd round trip', async () => {
  const file = path.join(dir, 'case.pvd')
  const series = [
    { time: 0, file: 'case_000000.vtu' },
    { time: 0.5, file: 'case_000010.vtu' },
    { time: 1e-5, file: 'sub\\dir\\case_000011.vtu' },
  ]
  await writePvd(file, series)
  const text = await fs.readFile(file, 'utf8')
  expect(text).toBe(
    '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n  <Collection>\n    <DataSet timestep="0" group="" part="0" file="case_000000.vtu"/>\n    <DataSet timestep="0.5" group="" part="0" file="case_000010.vtu"/>\n    <DataSet timestep="0.00001" group="" part="0" file="sub/dir/case_000011.vtu"/>\n  </Collection>\n</VTKFile>\n',
  )
  expect(await readPvd(file)).toEqual([
    { time: 0, file: 'case_000000.vtu' },
    { time: 0.5, file: 'case_000010.vtu' },
    { time: 1e-5, file: 'sub/dir/case_000011.vtu' },
  ])
  expect(formatPvd([])).toContain('<Collection>\n  </Collection>')
})

test('reads the Rust writer layout (no group attribute)', () => {
  const rust = '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n  <Collection>\n    <DataSet timestep="0" part="0" file="case_0.vtu"/>\n    <DataSet timestep="1" part="0" file="case_1.vtu"/>\n  </Collection>\n</VTKFile>\n'
  expect(parsePvd(rust)).toEqual([
    { time: 0, file: 'case_0.vtu' },
    { time: 1, file: 'case_1.vtu' },
  ])
})
