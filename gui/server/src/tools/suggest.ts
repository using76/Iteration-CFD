import { z } from 'zod'
import { okResult, type ToolDef } from './context.js'

const SuggestSchema = z.object({
  items: z.array(z.string().min(1).max(120)).max(3).describe('Up to 3 short next-step suggestions in the user\'s language, e.g. "Run ofgpu-k-epsilon on this case".'),
})

export const suggestFollowups: ToolDef<typeof SuggestSchema> = {
  name: 'suggest_followups',
  description: 'Offer up to three short follow-up actions as clickable chips under your final answer. Call it once, at the end of a turn that completed work; it never fails and returns nothing useful.',
  schema: SuggestSchema,
  async run(input) {
    const items = input.items.map((s) => s.trim()).filter(Boolean).slice(0, 3)
    return okResult({ ok: true, count: items.length }, { suggestions: items })
  },
}
