// STUB — replaced by the agent module owner.
import type { ServerConfig } from '../config.js'
import type { DatasetService } from '../datasets/types.js'
import type { RunManager } from '../runs/types.js'
import type { Hub } from '../ws/types.js'
import type { AgentService } from './types.js'

export interface AgentServiceDeps {
  config: ServerConfig
  hub: Hub
  runs: RunManager
  datasets: DatasetService
}

export function createAgentService(_deps: AgentServiceDeps): AgentService {
  throw new Error('not implemented: createAgentService')
}
