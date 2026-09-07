// STUB — replaced by the formats/datasets module owner.
import type { ServerConfig } from '../config.js'
import type { DatasetService } from './types.js'

export interface DatasetServiceDeps {
  config: ServerConfig
}

export function createDatasetService(_deps: DatasetServiceDeps): DatasetService {
  throw new Error('not implemented: createDatasetService')
}
