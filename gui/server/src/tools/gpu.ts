import { z } from 'zod'
import { okResult, type ToolDef } from './context.js'

export const gpuInfo: ToolDef<z.ZodObject<Record<string, never>>> = {
  name: 'gpu_info',
  description: 'Report the GPU the solvers will use (nvidia-smi, or the demo stand-in), its memory use, every process nvidia-smi lists on it (processes, processCount; ownRunPids are this server\'s running solvers, the rest belong to other sessions or desktop apps), and which ofgpu binaries the server can find.',
  schema: z.object({}),
  async run(_input, ctx) {
    const gpu = ctx.runs.gpu()
    const ownRunPids = ctx.runs.list().filter((r) => r.status === 'running' && r.pid !== null).map((r) => r.pid as number)
    return okResult({ ...gpu, mode: ctx.config.demo ? 'demo' : 'real', availableBinaries: ctx.runs.availableBinaries(), ownRunPids })
  },
}
