// Tool names, default approval policies and the one-line summaries the tool
// cards show. The server owns the schemas and implementations; this file is
// what the UI needs to render a call without knowing the tool internals.
import type { ToolPolicy } from './protocol'

export const TOOL_NAMES = [
  'case_read',
  'case_validate',
  'case_create',
  'case_edit',
  'mesh_generate',
  'run_start',
  'run_wait',
  'run_status',
  'run_log',
  'run_stop',
  'results_discover',
  'field_stats',
  'residuals_get',
  'viewer_command',
  'plot_residuals',
  'file_read',
  'file_list',
  'file_search',
  'file_write',
  'spec_lookup',
  'gpu_info',
  'custom_tool_create',
  'custom_tool_run',
  'shell_exec',
  'suggest_followups',
] as const
export type ToolName = (typeof TOOL_NAMES)[number]

export type ToolKind = 'read' | 'mutate' | 'long' | 'ui'

export interface ToolMeta {
  name: ToolName
  kind: ToolKind
  policy: ToolPolicy
  /** Korean / English labels for the card header. */
  label: { ko: string; en: string }
}

export const TOOL_META: Record<ToolName, ToolMeta> = {
  case_read: { name: 'case_read', kind: 'read', policy: 'auto', label: { ko: '케이스 설정 읽기', en: 'Read case configuration' } },
  case_validate: { name: 'case_validate', kind: 'read', policy: 'auto', label: { ko: '케이스 검증', en: 'Validate case' } },
  case_create: { name: 'case_create', kind: 'mutate', policy: 'ask', label: { ko: '케이스 생성', en: 'Create case' } },
  case_edit: { name: 'case_edit', kind: 'mutate', policy: 'ask', label: { ko: '케이스 편집', en: 'Edit case' } },
  mesh_generate: { name: 'mesh_generate', kind: 'long', policy: 'ask', label: { ko: '메쉬 생성', en: 'Generate mesh' } },
  run_start: { name: 'run_start', kind: 'long', policy: 'ask', label: { ko: '솔버 실행', en: 'Run solver' } },
  run_wait: { name: 'run_wait', kind: 'read', policy: 'auto', label: { ko: '실행 대기', en: 'Wait for run' } },
  run_status: { name: 'run_status', kind: 'read', policy: 'auto', label: { ko: '실행 상태', en: 'Run status' } },
  run_log: { name: 'run_log', kind: 'read', policy: 'auto', label: { ko: '로그 읽기', en: 'Read log' } },
  run_stop: { name: 'run_stop', kind: 'mutate', policy: 'ask', label: { ko: '실행 중단', en: 'Stop run' } },
  results_discover: { name: 'results_discover', kind: 'read', policy: 'auto', label: { ko: '결과 탐색', en: 'Discover results' } },
  field_stats: { name: 'field_stats', kind: 'read', policy: 'auto', label: { ko: '필드 통계', en: 'Field statistics' } },
  residuals_get: { name: 'residuals_get', kind: 'read', policy: 'auto', label: { ko: '잔차 조회', en: 'Get residuals' } },
  viewer_command: { name: 'viewer_command', kind: 'ui', policy: 'auto', label: { ko: '3D 뷰어', en: '3D viewer' } },
  plot_residuals: { name: 'plot_residuals', kind: 'ui', policy: 'auto', label: { ko: '잔차 플롯', en: 'Plot residuals' } },
  file_read: { name: 'file_read', kind: 'read', policy: 'auto', label: { ko: '파일 읽기', en: 'Read file' } },
  file_list: { name: 'file_list', kind: 'read', policy: 'auto', label: { ko: '디렉터리 목록', en: 'List directory' } },
  file_search: { name: 'file_search', kind: 'read', policy: 'auto', label: { ko: '코드 검색', en: 'Search files' } },
  file_write: { name: 'file_write', kind: 'mutate', policy: 'ask', label: { ko: '파일 쓰기', en: 'Write file' } },
  spec_lookup: { name: 'spec_lookup', kind: 'read', policy: 'auto', label: { ko: 'SPEC-LIT 조회', en: 'Look up SPEC-LIT' } },
  gpu_info: { name: 'gpu_info', kind: 'read', policy: 'auto', label: { ko: 'GPU 정보', en: 'GPU info' } },
  custom_tool_create: { name: 'custom_tool_create', kind: 'mutate', policy: 'ask', label: { ko: '사용자 도구 등록', en: 'Create custom tool' } },
  custom_tool_run: { name: 'custom_tool_run', kind: 'mutate', policy: 'ask', label: { ko: '사용자 도구 실행', en: 'Run custom tool' } },
  shell_exec: { name: 'shell_exec', kind: 'mutate', policy: 'never', label: { ko: '셸 명령', en: 'Shell command' } },
  suggest_followups: { name: 'suggest_followups', kind: 'ui', policy: 'auto', label: { ko: '후속 제안', en: 'Suggestions' } },
}

export function toolPolicy(name: string): ToolPolicy {
  return (TOOL_META as Record<string, ToolMeta | undefined>)[name]?.policy ?? 'ask'
}

export function toolLabel(name: string, locale: 'ko' | 'en'): string {
  const m = (TOOL_META as Record<string, ToolMeta | undefined>)[name]
  return m ? m.label[locale] : name
}

const fmtInt = (n: number) => n.toLocaleString('en-US')

/**
 * One-line, past-tense summary for a finished call. `result` is the parsed
 * tool result (or null); `input` the validated tool input.
 */
export function summarizeToolCall(name: string, input: unknown, result: unknown, ok: boolean, locale: 'ko' | 'en' = 'ko'): string {
  const i = (input ?? {}) as Record<string, unknown>
  const r = (result ?? {}) as Record<string, unknown>
  const ko = locale === 'ko'
  if (!ok) {
    const err = typeof r.error === 'object' && r.error ? (r.error as Record<string, unknown>).message : r.error
    return ko ? `${toolLabel(name, 'ko')} 실패${err ? `: ${String(err)}` : ''}` : `${toolLabel(name, 'en')} failed${err ? `: ${String(err)}` : ''}`
  }
  switch (name) {
    case 'case_read':
      return ko ? `케이스 설정 읽음 (${String(i.path ?? '')})` : `Read case configuration (${String(i.path ?? '')})`
    case 'case_validate':
      return r.ok ? (ko ? '케이스 검증 통과' : 'Case validation passed') : ko ? `케이스 검증 실패 (${(r.errors as unknown[] | undefined)?.length ?? 0}개 오류)` : `Case validation failed (${(r.errors as unknown[] | undefined)?.length ?? 0} errors)`
    case 'case_edit':
      return i.dryRun ? (ko ? `편집 미리보기 (${String(i.path ?? '')})` : `Previewed edit (${String(i.path ?? '')})`) : ko ? `케이스 편집됨 (${String(i.path ?? '')})` : `Edited case (${String(i.path ?? '')})`
    case 'case_create':
      return ko ? `케이스 생성됨 (${String(r.path ?? i.path ?? '')})` : `Created case (${String(r.path ?? i.path ?? '')})`
    case 'mesh_generate':
      return typeof r.cells === 'number' ? (ko ? `구조 격자 생성 (${fmtInt(r.cells)} 셀)` : `Generated structured mesh (${fmtInt(r.cells)} cells)`) : ko ? `메쉬 생성 시작 (${String(i.kind ?? '')})` : `Started mesh generation (${String(i.kind ?? '')})`
    case 'run_start':
      return ko ? `${String(i.binary ?? '솔버')} 실행 시작` : `Started ${String(i.binary ?? 'solver')}`
    case 'run_wait':
      return ko ? `실행 상태: ${String(r.status ?? '')}${typeof r.iter === 'number' ? ` (${fmtInt(r.iter)} 반복)` : ''}` : `Run ${String(r.status ?? '')}${typeof r.iter === 'number' ? ` (${fmtInt(r.iter)} iterations)` : ''}`
    case 'run_status':
      return ko ? '실행 상태 조회' : 'Checked run status'
    case 'run_log':
      return ko ? `로그 ${(r.lines as unknown[] | undefined)?.length ?? 0}줄 읽음` : `Read ${(r.lines as unknown[] | undefined)?.length ?? 0} log lines`
    case 'run_stop':
      return ko ? '실행 중단됨' : 'Stopped run'
    case 'results_discover':
      return ko ? `결과 ${(r.times as unknown[] | undefined)?.length ?? 0}개 시간 스텝 발견` : `Found ${(r.times as unknown[] | undefined)?.length ?? 0} result time steps`
    case 'field_stats':
      return ko ? `${String(i.field ?? '')} 통계 계산` : `Computed ${String(i.field ?? '')} statistics`
    case 'residuals_get':
      return ko ? '잔차 시계열 조회' : 'Fetched residual series'
    case 'viewer_command': {
      const t = String(i.type ?? '')
      const map: Record<string, [string, string]> = {
        load: ['3D 뷰어에 결과 로드', 'Loaded results in the 3D viewer'],
        setField: [`필드 ${String(i.field ?? '')} 표시`, `Showing field ${String(i.field ?? '')}`],
        addSlice: ['절단면 추가', 'Added slice'],
        addPlane: ['절단 평면 추가', 'Added cut plane'],
        addIsoSurface: ['등가면 추가', 'Added iso-surface'],
        addStreamlines: ['유선 추가', 'Added streamlines'],
        addGlyphs: ['벡터 글리프 추가', 'Added vector glyphs'],
        setCamera: ['카메라 이동', 'Moved camera'],
        screenshot: ['스크린샷 촬영', 'Took screenshot'],
        setRepresentation: ['표현 방식 변경', 'Changed representation'],
        setTime: ['시간 스텝 변경', 'Changed time step'],
        clear: ['뷰어 초기화', 'Cleared viewer'],
      }
      const pair = map[t] ?? [`뷰어 명령 ${t}`, `Viewer command ${t}`]
      return ko ? pair[0] : pair[1]
    }
    case 'plot_residuals':
      return ko ? '잔차 차트 열기' : 'Opened residual chart'
    case 'file_read':
      return ko ? `파일 읽음 (${String(i.path ?? '')})` : `Read ${String(i.path ?? '')}`
    case 'file_list':
      return ko ? `디렉터리 목록 (${String(i.dir ?? i.path ?? '')})` : `Listed ${String(i.dir ?? i.path ?? '')}`
    case 'file_search':
      return ko ? `검색 "${String(i.pattern ?? '')}" (${(r.hits as unknown[] | undefined)?.length ?? 0}건)` : `Searched "${String(i.pattern ?? '')}" (${(r.hits as unknown[] | undefined)?.length ?? 0} hits)`
    case 'file_write':
      return ko ? `파일 저장 (${String(i.path ?? '')})` : `Wrote ${String(i.path ?? '')}`
    case 'spec_lookup':
      return ko ? `SPEC-LIT ${String(i.section ?? i.query ?? '')} 조회` : `Looked up SPEC-LIT ${String(i.section ?? i.query ?? '')}`
    case 'gpu_info':
      return ko ? `GPU: ${String(r.name ?? 'unknown')}` : `GPU: ${String(r.name ?? 'unknown')}`
    case 'custom_tool_create':
      return ko ? `사용자 도구 등록 (${String(i.name ?? '')})` : `Registered custom tool ${String(i.name ?? '')}`
    case 'custom_tool_run':
      return ko ? `사용자 도구 실행 (${String(i.name ?? '')})` : `Ran custom tool ${String(i.name ?? '')}`
    case 'shell_exec':
      return ko ? '셸 명령 실행' : 'Ran shell command'
    case 'suggest_followups':
      return ko ? '후속 제안' : 'Suggested follow-ups'
    default:
      return name
  }
}
