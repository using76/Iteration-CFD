import { z } from 'zod'
import { okResult, type ToolDef } from './context.js'

export const gpuInfo: ToolDef<z.ZodObject<Record<string, never>>> = {
  name: 'gpu_info',
  description: 'Report the GPU the solvers will use (nvidia-smi, or the demo stand-in), its memory use, and which ofgpu binaries the server can find.',
  schema: z.object({}),
  async run(_input, ctx) {
    const gpu = ctx.runs.gpu()
    return okResult({ ...gpu, mode: ctx.config.demo ? 'demo' : 'real', availableBinaries: ctx.runs.availableBinaries() })
  },
}
