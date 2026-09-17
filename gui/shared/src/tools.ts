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
  'line_sample',
  'geometry_open',
  'geometry_info',
  'geometry_save',
  'geometry_import_step',
  'geometry_edit',
  'mesh_regions',
  'regions_check',
  'residuals_get',
  'viewer_command',
  'plot_residuals',
  'gui_control',
  'gui_state',
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
  'ontology_query',
  'ontology_act',
  'ontology_apply',
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
  line_sample: { name: 'line_sample', kind: 'read', policy: 'auto', label: { ko: '선 샘플링', en: 'Sample along a line' } },
  geometry_open: { name: 'geometry_open', kind: 'read', policy: 'auto', label: { ko: '지오메트리 열기', en: 'Open geometry' } },
  geometry_info: { name: 'geometry_info', kind: 'read', policy: 'auto', label: { ko: '지오메트리 정보', en: 'Geometry info' } },
  geometry_save: { name: 'geometry_save', kind: 'mutate', policy: 'ask', label: { ko: '지오메트리 저장', en: 'Save geometry' } },
  geometry_import_step: { name: 'geometry_import_step', kind: 'long', policy: 'ask', label: { ko: 'STEP 가져오기', en: 'Import STEP' } },
  geometry_edit: { name: 'geometry_edit', kind: 'mutate', policy: 'ask', label: { ko: '지오메트리 편집', en: 'Edit geometry' } },
  mesh_regions: { name: 'mesh_regions', kind: 'long', policy: 'ask', label: { ko: '영역 격자 생성', en: 'Mesh regions' } },
  regions_check: { name: 'regions_check', kind: 'read', policy: 'auto', label: { ko: '영역 레이아웃 검사', en: 'Check regions' } },
  residuals_get: { name: 'residuals_get', kind: 'read', policy: 'auto', label: { ko: '잔차 조회', en: 'Get residuals' } },
  viewer_command: { name: 'viewer_command', kind: 'ui', policy: 'auto', label: { ko: '3D 뷰어', en: '3D viewer' } },
  plot_residuals: { name: 'plot_residuals', kind: 'ui', policy: 'auto', label: { ko: '잔차 플롯', en: 'Plot residuals' } },
  gui_control: { name: 'gui_control', kind: 'ui', policy: 'auto', label: { ko: '화면 제어', en: 'Drive the UI' } },
  gui_state: { name: 'gui_state', kind: 'read', policy: 'auto', label: { ko: '화면 상태', en: 'UI state' } },
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
  ontology_query: { name: 'ontology_query', kind: 'read', policy: 'auto', label: { ko: '온톨로지 조회', en: 'Query the ontology' } },
  ontology_act: { name: 'ontology_act', kind: 'mutate', policy: 'ask', label: { ko: '변경 제안', en: 'Propose a change' } },
  ontology_apply: { name: 'ontology_apply', kind: 'mutate', policy: 'auto', label: { ko: '제안 적용', en: 'Apply a proposal' } },
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
    case 'line_sample': {
      const pts = (r.values as unknown[] | undefined)?.length ?? 0
      return ko ? `${String(i.field ?? '')} 선 샘플링 (${fmtInt(pts)}개 지점)` : `Sampled ${String(i.field ?? '')} along ${fmtInt(pts)} points`
    }
    case 'geometry_open': {
      const base = String(r.path ?? i.path ?? '').split(/[\\/]/).pop() ?? ''
      const n = fmtInt(Number(r.triangleCount ?? 0))
      if (r.closed) return ko ? `${base} 열기: 삼각형 ${n}개, 닫힌 면` : `Opened ${base}: ${n} triangles, closed`
      const k = fmtInt(Number(r.openEdges ?? 0))
      return ko ? `${base} 열기: 삼각형 ${n}개, 열린 모서리 ${k}개` : `Opened ${base}: ${n} triangles, ${k} open edges`
    }
    case 'geometry_info': {
      const base = String(r.path ?? i.path ?? '').split(/[\\/]/).pop() || String(i.id ?? '')
      const n = fmtInt(Number(r.triangleCount ?? 0))
      const s = fmtInt((r.solids as unknown[] | undefined)?.length ?? 0)
      return ko ? `지오메트리 조회 (${base}: 삼각형 ${n}개, 솔리드 ${s}개)` : `Geometry info (${base}: ${n} triangles, ${s} solids)`
    }
    case 'geometry_save': {
      const base = String(r.path ?? i.path ?? '').split(/[\\/]/).pop() ?? ''
      const n = fmtInt(Number(r.triangleCount ?? 0))
      return ko ? `지오메트리 저장 (${base}, 삼각형 ${n}개)` : `Saved geometry ${base} (${n} triangles)`
    }
    case 'geometry_import_step': {
      const base = String(i.path ?? '').split(/[\\/]/).pop() ?? ''
      const s = fmtInt((r.solids as unknown[] | undefined)?.length ?? 0)
      const n = fmtInt(Number(r.triangleCount ?? 0))
      return ko ? `${base} 가져옴: 솔리드 ${s}개, 삼각형 ${n}개` : `Imported ${base}: ${s} solids, ${n} triangles`
    }
    case 'geometry_edit': {
      const base = String(i.path ?? '').split(/[\\/]/).pop() ?? ''
      const outBase = String(i.out ?? '').split(/[\\/]/).pop() ?? ''
      const k = fmtInt((r.applied as unknown[] | undefined)?.length ?? 0)
      const s = fmtInt((r.solids as unknown[] | undefined)?.length ?? 0)
      return ko ? `${base} 편집 -> ${outBase} (${k}개 연산, 솔리드 ${s}개)` : `Edited ${base} -> ${outBase} (${k} ops, ${s} solids)`
    }
    case 'mesh_regions': {
      if (r.manifest) {
        const regions = fmtInt((r.regions as unknown[] | undefined)?.length ?? 0)
        const ifs = fmtInt((r.interfaces as unknown[] | undefined)?.length ?? 0)
        return ko ? `영역 레이아웃 ${String(r.layout ?? '')}: 영역 ${regions}개, 인터페이스 ${ifs}개` : `Region layout ${String(r.layout ?? '')}: ${regions} regions, ${ifs} interfaces`
      }
      const ids = (r.runIds as string[] | undefined) ?? []
      const last = ids[ids.length - 1] ?? ''
      return ko ? `${String(r.stage ?? '')} 시작 (${last})` : `Started ${String(r.stage ?? '')} (${last})`
    }
    case 'regions_check': {
      const base = String(i.manifest ?? '').split(/[\\/]/).pop() ?? ''
      if (r.ok) return ko ? `레이아웃 통과 (${base})` : `Layout OK (${base})`
      const v = fmtInt((r.violations as unknown[] | undefined)?.length ?? 0)
      return ko ? `레이아웃 위반 ${v}건` : `Layout violates ${v} rule(s)`
    }
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
        setWarp: ['변형 형상 표시', 'Warped by displacement'],
        clear: ['뷰어 초기화', 'Cleared viewer'],
      }
      const pair = map[t] ?? [`뷰어 명령 ${t}`, `Viewer command ${t}`]
      return ko ? pair[0] : pair[1]
    }
    case 'plot_residuals':
      return ko ? '잔차 차트 열기' : 'Opened residual chart'
    case 'gui_control': {
      const t = String(i.type ?? '')
      const arg = String(i.field ?? i.tab ?? i.step ?? i.panel ?? i.tool ?? i.projection ?? i.quantity ?? i.action ?? i.path ?? i.kind ?? i.id ?? i.chart ?? i.locale ?? i.preset ?? i.op ?? i.name ?? i.a ?? '')
      const overlay = i.what !== undefined ? `${String(i.what)} ${i.on ? 'on' : 'off'}` : ''
      const map: Record<string, [string, string]> = {
        select_tab: [`탭 ${arg} 열기`, `Opened tab ${arg}`],
        show_field: [`필드 ${arg} 표시`, `Showing field ${arg}`],
        select_step: [`스텝 ${arg} 선택`, `Selected step ${arg}`],
        open_panel: [`패널 ${arg} 열기`, `Opened panel ${arg}`],
        set_tool: [`도구 ${arg} 선택`, `Set tool ${arg}`],
        set_projection: [`투영 ${arg} 전환`, `Set projection ${arg}`],
        fit_view: ['화면 맞춤', 'Fit the view'],
        show_overlay: [`오버레이 ${overlay}`, `Turned overlay ${overlay}`],
        set_centerline: [`중심선 수량 ${arg} 설정`, `Set centerline quantity ${arg}`],
        run: [arg === 'stop' ? '실행 중단 요청' : '실행 시작 요청', arg === 'stop' ? 'Asked the UI to stop the run' : 'Asked the UI to start the run'],
        notify: [`화면 알림: ${String(i.text ?? '').slice(0, 60)}`, `Notified the operator: ${String(i.text ?? '').slice(0, 60)}`],
        open_case: [`케이스 ${arg} 열기`, `Opened case ${arg}`],
        save_case: [i.force ? '검증 오류에도 케이스 저장 요청' : '케이스 저장 요청', i.force ? 'Asked the UI to save the case anyway' : 'Asked the UI to save the case'],
        validate_case: ['케이스 검증 요청', 'Asked the UI to validate the case'],
        new_case: [`새 케이스 ${String(i.name ?? '')} 만들기`, `Created case ${String(i.name ?? '')}`],
        set_run_setting: [`실행 설정 ${String(i.flag ?? '')} 변경`, `Changed run setting ${String(i.flag ?? '')}`],
        start_run: ['실행 시작 요청', 'Asked the UI to start the run'],
        stop_run: ['실행 중단 요청', 'Asked the UI to stop the run'],
        follow_run: [`실행 ${String(i.runId ?? '')} 따라가기`, `Followed run ${String(i.runId ?? '')}`],
        open_mesh_dialog: ['메쉬 대화상자 열기', 'Opened the mesh dialog'],
        start_mesh: ['메쉬 생성 시작 요청', 'Asked the UI to start the mesh'],
        open_mesh_view: ['메쉬 탭 열기', 'Opened the mesh tab'],
        show_chart: [`${arg === '' ? '차트' : arg} 차트 열기`, `Opened the ${arg === '' ? 'chart' : arg} chart`],
        show_metric: [`지표 ${String(i.metric ?? i.mode ?? '')} 표시`, `Showed metric ${String(i.metric ?? i.mode ?? '')}`],
        set_log_filter: ['로그 필터 변경', 'Changed the log filter'],
        open_result: [`결과 ${arg} 열기`, `Opened result ${arg}`],
        set_post: ['표시 설정 변경', 'Changed the post-processing settings'],
        post_field: [`표시 필드 ${String(i.field ?? '')} 설정`, `Coloured by ${String(i.field ?? '')}`],
        post_representation: [`표현 방식 ${String(i.mode ?? '')} 설정`, `Set the representation to ${String(i.mode ?? '')}`],
        post_time: [`시간 단계 ${String(i.index ?? '')} 이동`, `Moved to time step ${String(i.index ?? '')}`],
        post_screenshot: ['스크린샷 저장 요청', 'Saved a screenshot of the view'],
        post_warp: [`변형 표시 ${String(i.field ?? '없음')} x${String(i.scale ?? 1)}`, `Warped by ${String(i.field ?? 'none')} x${String(i.scale ?? 1)}`],
        add_layer: [`${arg} 레이어 추가`, `Added a ${arg} layer`],
        remove_layer: [`레이어 ${arg} 제거`, `Removed layer ${arg}`],
        set_camera: [`카메라 ${arg} 전환`, `Set camera ${arg}`],
        probe: ['화면 위치 값 조회', 'Probed the value under a screen point'],
        open_tab: [`탭 ${arg} 열기`, `Opened tab ${arg}`],
        close_tab: [`탭 ${arg} 닫기`, `Closed tab ${arg}`],
        set_locale: [`UI 언어를 ${arg}(으)로 전환`, `Switched the UI language to ${arg}`],
        set_patch: [
          i.reset ? `패치 ${String(i.patch ?? '')} 규칙 초기화` : `패치 ${String(i.patch ?? '')} ${String(i.kind ?? i.field ?? '')} 설정`,
          i.reset ? `Reset patch ${String(i.patch ?? '')}` : `Set patch ${String(i.patch ?? '')} ${String(i.kind ?? i.field ?? '')}`,
        ],
        open_boundary_editor: ['경계조건 편집기 열기', 'Opened the boundary editor'],
        open_session: [`대화 ${String(i.sessionId ?? '')} 열기`, `Opened session ${String(i.sessionId ?? '')}`],
        set_setting: [`설정 변경 (${Object.keys(i).filter((k) => k !== 'type' && i[k] != null).join(', ')})`, `Changed settings (${Object.keys(i).filter((k) => k !== 'type' && i[k] != null).join(', ')})`],
        run_custom_tool: [`사용자 도구 ${String(i.name ?? '')} 실행`, `Ran custom tool ${String(i.name ?? '')}`],
        split_view: [i.on ? '뷰포트 분할' : '뷰포트 분할 해제', i.on ? 'Split the viewport' : 'Closed the split view'],
        focus_view: [`뷰 ${String(i.view ?? '')} 선택`, `Focused view ${String(i.view ?? '')}`],
        open_result_in_view: [`뷰 ${String(i.view ?? '')}에 결과 ${arg} 열기`, `Opened result ${arg} in view ${String(i.view ?? '')}`],
        link_cameras: [i.on ? '카메라 연동' : '카메라 연동 해제', i.on ? 'Linked the cameras' : 'Unlinked the cameras'],
        compare_run: [i.runId ? `실행 ${String(i.runId)} 잔차 비교` : '잔차 비교 해제', i.runId ? `Compared run ${String(i.runId)} on the residual chart` : 'Removed the residual comparison'],
        geometry_open: [`지오메트리 ${arg} 열기`, `Opened geometry ${arg}`],
        geometry_import_step: [`STEP ${arg} 가져오기`, `Imported STEP ${arg}`],
        geometry_part: [`부품 ${String(i.name ?? '')} ${String(i.action ?? '')}`, `Part ${String(i.name ?? '')}: ${String(i.action ?? '')}`],
        geometry_transform: [`지오메트리 변환 ${String(i.op ?? '')}`, `Geometry transform ${String(i.op ?? '')}`],
        geometry_boolean: [`불리언 ${String(i.op ?? '')} (${String(i.a ?? '')}, ${String(i.b ?? '')})`, `Boolean ${String(i.op ?? '')} of ${String(i.a ?? '')} and ${String(i.b ?? '')}`],
        geometry_save: [`지오메트리 ${arg} 저장 요청`, `Asked the UI to save the geometry to ${arg}`],
      }
      const pair = map[t] ?? [`화면 명령 ${t}`, `UI command ${t}`]
      return ko ? `GUI: ${pair[0]}` : `GUI: ${pair[1]}`
    }
    case 'gui_state':
      return ko ? (r.state ? '화면 상태 조회' : '연결된 화면 없음') : r.state ? 'Read the UI state' : 'No UI connected'
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
    case 'ontology_query': {
      if ((r as Record<string, unknown>).kind === 'ontologyContext') {
        const p = fmtInt((r.chunks as unknown[] | undefined)?.length ?? 0)
        const o = fmtInt((r.objects as unknown[] | undefined)?.length ?? 0)
        return ko ? `온톨로지 검색: 구절 ${p}개, 객체 ${o}건` : `Searched the ontology (${p} passages, ${o} objects)`
      }
      return ko ? `온톨로지 조회: ${String(i.objectType ?? '')} ${fmtInt((r.objects as unknown[] | undefined)?.length ?? 0)}건` : `Queried ${String(i.objectType ?? '')} (${fmtInt((r.objects as unknown[] | undefined)?.length ?? 0)} objects)`
    }
    case 'ontology_act':
      return r.state === 'rejected'
        ? (ko ? `제안 거부됨: ${String(i.action ?? '')} (차단 ${(r.blocking as unknown[] | undefined)?.length ?? 0}건)` : `Proposal rejected: ${String(i.action ?? '')} (${(r.blocking as unknown[] | undefined)?.length ?? 0} blocking)`)
        : (ko ? `제안 생성: ${String(i.action ?? '')} (객체 ${r.objects ?? 0}, 링크 ${r.links ?? 0})` : `Proposed ${String(i.action ?? '')} (${r.objects ?? 0} objects, ${r.links ?? 0} links)`)
    case 'ontology_apply':
      return ko ? `적용됨: ${String(r.action ?? '')} (${String(r.editId ?? '')})` : `Applied ${String(r.action ?? '')} (edit ${String(r.editId ?? '')})`
    default:
      return name
  }
}
