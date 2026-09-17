import fsp from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { describe, expect, it } from 'vitest'
import { createRunManager } from './manager.js'
import { makeTempWorkspace, type TempWorkspace } from './test-helpers.js'

describe('run manager provenance', () => {
  it('start() stamps the case, the mesh and the machine into the record it returns', async () => {
    const ws: TempWorkspace = await makeTempWorkspace()
    try {
      // The mesh the plume case reads, with its summary: what mints a Mesh key.
      const poly = path.join(ws.root, 'cases', 'plume_jsonc', 'constant', 'polyMesh')
      await fsp.mkdir(poly, { recursive: true })
      await fsp.writeFile(path.join(poly, 'points'), '')
      await fsp.writeFile(path.join(poly, '.meshSummary.json'), JSON.stringify({ caseDir: 'cases/plume_jsonc' }))
      const runs = await createRunManager({ config: ws.config })
      try {
        const run = await runs.start({ binary: 'ofgpu-k-epsilon', casePath: 'cases/plume.jsonc', positionals: [], args: [], label: null })
        expect(run.caseId).toBe('cases/plume.jsonc')
        expect(run.meshId).toBe('cases/plume_jsonc/constant/polyMesh/.meshSummary.json#plume_jsonc')
        expect(run.machine?.hostname).toBe(os.hostname())
        // The monitor may or may not have polled; assert the type, never the value.
        expect(typeof run.machine?.gpu).toBe('string')
        expect('gitSha' in run && 'gitDirty' in run).toBe(true)
      } finally {
        await runs.shutdown()
      }
    } finally {
      await ws.cleanup()
    }
  })
})
