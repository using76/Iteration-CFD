// H3: a re-run that overwrites the same time directory must not be served from
// the cache of the previous run.
import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { afterEach, describe, expect, test } from 'vitest'
import { writeFoamField } from '../../formats/foam.js'
import { listTimeDirs, resolveResultRoot } from '../../formats/results.js'
import { datasetFingerprint, scanSeries } from '../manifest.js'

let base = ''
afterEach(async () => {
  if (base) await fs.rm(base, { recursive: true, force: true })
  base = ''
})

async function writeP(dir: string, values: number[]): Promise<void> {
  await writeFoamField(path.join(dir, 'p'), {
    name: 'p',
    class: 'volScalarField',
    dimensions: '[0 0 0 0 0 0 0]',
    time: path.basename(dir),
    data: Float32Array.from(values),
    components: 1,
    patches: [],
    collapseUniform: false,
  })
}

/** fs mtime resolution is coarse; the size change alone must be enough. */
async function fingerprintOf(root: string): Promise<string> {
  const resolved = await resolveResultRoot(root, root)
  return datasetFingerprint('case', resolved, await scanSeries(resolved))
}

describe('datasetFingerprint', () => {
  test('moves when a field file is rewritten in the same time directory', async () => {
    base = await fs.mkdtemp(path.join(os.tmpdir(), 'cfd-fp-'))
    const root = path.join(base, 'case')
    await writeP(path.join(root, '10'), [1, 2, 3, 4])
    const before = await fingerprintOf(root)

    await writeP(path.join(root, '10'), [9, 9, 9, 9, 9, 9])
    const after = await fingerprintOf(root)
    expect(after).not.toBe(before)

    // The directory's own mtime is the thing the old fingerprint watched, and
    // it is unchanged: no entry was added or removed.
    const dirs = await listTimeDirs(root)
    expect(dirs[0].fieldStamps.p).toBeDefined()
    expect(dirs[0].fieldStamps.p).not.toBe('missing')
  })

  test('moves when the dictionaries change under an unchanged result tree', async () => {
    base = await fs.mkdtemp(path.join(os.tmpdir(), 'cfd-fp-'))
    const root = path.join(base, 'case')
    await writeP(path.join(root, '10'), [1, 2, 3, 4])
    await fs.mkdir(path.join(root, 'system'), { recursive: true })
    await fs.writeFile(path.join(root, 'system', 'controlDict'), 'endTime 10;\n')
    const before = await fingerprintOf(root)

    await fs.writeFile(path.join(root, 'system', 'controlDict'), 'endTime 2000;\n')
    expect(await fingerprintOf(root)).not.toBe(before)
  })

  test('holds still when nothing on disk moved', async () => {
    base = await fs.mkdtemp(path.join(os.tmpdir(), 'cfd-fp-'))
    const root = path.join(base, 'case')
    await writeP(path.join(root, '10'), [1, 2, 3, 4])
    expect(await fingerprintOf(root)).toBe(await fingerprintOf(root))
  })
})
