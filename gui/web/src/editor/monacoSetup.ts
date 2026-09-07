// Bundled Monaco: workers via Vite `?worker` imports, JSONC diagnostics with
// the case schema, and two themes that follow the app tokens.
import * as monaco from 'monaco-editor'
import { loader } from '@monaco-editor/react'
import editorWorker from 'monaco-editor/editor/editor.worker?worker'
import jsonWorker from 'monaco-editor/language/json/json.worker?worker'
import { REST } from '@cfd/shared'
import { api } from '../api/rest'

let configured = false
let schemaPromise: Promise<void> | null = null

export const CASE_SCHEMA_FILE_MATCH = ['**/cases/**/*.jsonc', '**/*.cfd.jsonc']

export function fileUri(path: string): string {
  return `file:///${path.replace(/^\/+/, '')}`
}

export function setupMonaco(): typeof monaco {
  if (configured) return monaco
  configured = true
  globalThis.MonacoEnvironment = {
    getWorker(_workerId: string, label: string) {
      return label === 'json' ? new jsonWorker() : new editorWorker()
    },
  }
  loader.config({ monaco })
  monaco.editor.defineTheme('cfd-light', {
    base: 'vs',
    inherit: true,
    rules: [],
    colors: { 'editor.background': '#ffffff', 'editorLineNumber.foreground': '#8a94a3', 'editor.lineHighlightBackground': '#f4f6f9', 'editorGutter.background': '#ffffff' },
  })
  monaco.editor.defineTheme('cfd-dark', {
    base: 'vs-dark',
    inherit: true,
    rules: [],
    colors: { 'editor.background': '#1b1f26', 'editorLineNumber.foreground': '#6f7886', 'editor.lineHighlightBackground': '#20252d', 'editorGutter.background': '#1b1f26' },
  })
  monaco.json.jsonDefaults.setDiagnosticsOptions({ validate: true, allowComments: true, trailingCommas: 'ignore', comments: 'ignore', enableSchemaRequest: false, schemas: [] })
  return monaco
}

/** Fetch docs/schema/case-1.json once and attach it to matching JSONC models. */
export function ensureCaseSchema(): Promise<void> {
  if (!schemaPromise) {
    schemaPromise = api
      .caseSchema()
      .then((schema) => {
        monaco.json.jsonDefaults.setDiagnosticsOptions({
          validate: true,
          allowComments: true,
          trailingCommas: 'ignore',
          comments: 'ignore',
          enableSchemaRequest: false,
          schemas: [{ uri: REST.caseSchema, fileMatch: CASE_SCHEMA_FILE_MATCH, schema }],
        })
      })
      .catch((err: unknown) => {
        schemaPromise = null
        console.warn('case schema unavailable', err)
      })
  }
  return schemaPromise
}

export { monaco }
