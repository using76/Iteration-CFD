// Server entry point. Placeholder until the server core lands: boots config
// and prints it, so `npm run typecheck` and `tsx src/main.ts` both work.
import { loadConfig } from './config.js'

const config = loadConfig()
console.log(`[cfd-server] placeholder — workspace ${config.workspaceRoot}, demo=${config.demo}, llm=${config.llm}`)
