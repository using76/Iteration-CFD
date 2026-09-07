// Quick actions (the buttons under the chat box) become ordinary user turns
// with a synthetic message; the loop treats them like typed text.
import { getBinary, isJsonCase, jsonCaseOutputDir, SOLVER_PROBLEM_PATTERNS, type QuickAction, type RunInfo } from '@cfd/shared'
import type { RunManager } from '../runs/types.js'

export interface QuickInput {
  action: QuickAction
  casePath: string | null
  runId: string | null
  activeFile: string | null
  locale: 'ko' | 'en'
  runs: Pick<RunManager, 'get' | 'list' | 'log'>
}

export const EXPLAIN_LOG_LINES = 200

function latestRun(runs: Pick<RunManager, 'list'>, casePath: string | null): RunInfo | null {
  const all = runs.list()
  const candidates = casePath ? all.filter((r) => r.casePath === casePath) : all
  const pool = candidates.length ? candidates : all
  return pool.length ? pool[pool.length - 1] : null
}

function latestResultDir(runs: Pick<RunManager, 'list'>, casePath: string | null): string | null {
  const all = runs.list()
  for (let i = all.length - 1; i >= 0; i--) {
    const r = all[i]
    if (casePath && r.casePath !== casePath) continue
    if (r.written.length) return r.written[r.written.length - 1]
  }
  if (casePath) return isJsonCase(casePath) ? jsonCaseOutputDir(casePath) : casePath
  return null
}

function logTail(runs: Pick<RunManager, 'get' | 'log'>, run: RunInfo): { lines: string[]; problems: string[] } {
  const from = Math.max(1, run.logLines - EXPLAIN_LOG_LINES + 1)
  const lines = runs.log(run.id, from, EXPLAIN_LOG_LINES).lines.map((l) => (l.stream === 'stderr' ? `! ${l.text}` : l.text))
  const problems = lines.filter((l) => SOLVER_PROBLEM_PATTERNS.some((p) => p.re.test(l)))
  if (run.error && !problems.includes(run.error)) problems.unshift(run.error)
  return { lines, problems }
}

export function buildQuickMessage(q: QuickInput): { text: string; synthetic: true } {
  const ko = q.locale === 'ko'
  const caseRef = q.casePath ?? (q.activeFile && (isJsonCase(q.activeFile) || /\/(constant|system|0)\//.test(q.activeFile)) ? q.activeFile : null)
  switch (q.action) {
    case 'mesh': {
      const target = caseRef ?? 'cases/channel'
      return { text: ko ? `${target}에 대한 메쉬를 생성해 주세요. 케이스에 맞는 프리셋을 고르고 셀 수를 알려주세요.` : `Generate a mesh for ${target}: pick the matching preset, and report the cell count.`, synthetic: true }
    }
    case 'run': {
      const target = caseRef ?? 'cases/plume.jsonc'
      return { text: ko ? `${target}에 적합한 솔버를 실행하고 끝날 때까지 모니터링한 뒤 결과를 요약해 주세요.` : `Run the appropriate solver for ${target} and monitor it until it finishes, then summarise the result.`, synthetic: true }
    }
    case 'explain': {
      const run = (q.runId && q.runs.get(q.runId)) || latestRun(q.runs, caseRef)
      if (!run) return { text: ko ? '최근 실행이 없습니다. 어떤 오류를 설명할까요?' : 'There is no recent run. Which error should I explain?', synthetic: true }
      const { lines, problems } = logTail(q.runs, run)
      const head = ko ? `실행 ${run.id} (${run.binary}${run.casePath ? `, ${run.casePath}` : ''}, 상태 ${run.status}, ${run.iter}회 반복)의 오류를 설명하고 해결 방법을 제안해 주세요.` : `Explain the error in run ${run.id} (${run.binary}${run.casePath ? `, ${run.casePath}` : ''}, status ${run.status}, ${run.iter} iterations) and propose a fix.`
      const probs = problems.length ? `\n\n${ko ? '문제' : 'Problems'}:\n${problems.map((p) => `- ${p}`).join('\n')}` : ''
      const tail = lines.length ? `\n\n${ko ? `로그 마지막 ${lines.length}줄` : `Last ${lines.length} log lines`}:\n\`\`\`\n${lines.join('\n')}\n\`\`\`` : ''
      return { text: `${head}${probs}${tail}`, synthetic: true }
    }
    case 'create_tool':
      return {
        text: ko
          ? '사용자 정의 도구를 만들고 싶습니다. custom_tool_create로 등록해 주세요: 먼저 도구 이름(snake_case), 하는 일, 입력 필드, 그리고 명령줄(argv, {{input.x}} 치환)인지 JavaScript(input, cfd API)인지 저에게 확인한 뒤 등록하고 custom_tool_run으로 한 번 시험해 주세요.'
          : 'I want to create a custom tool. Register it with custom_tool_create: first confirm with me the tool name (snake_case), what it does, its input fields, and whether it is a command line (argv with {{input.x}} substitution) or JavaScript (input + cfd API); then register it and try it once with custom_tool_run.',
        synthetic: true,
      }
    case 'postprocess': {
      const dir = latestResultDir(q.runs, caseRef) ?? 'cases/plume_jsonc'
      return { text: ko ? `${dir}의 최신 결과를 3D 뷰어에 로드하고(속도 U), 중앙 절단면과 유선을 추가한 뒤 무엇이 보이는지 설명해 주세요.` : `Load the latest results in ${dir} into the 3D viewer (field U), add a mid-plane slice and streamlines, and describe what is visible.`, synthetic: true }
    }
    case 'validate':
      return { text: ko ? 'ofgpu-validate를 실행하고 통과/실패 게이트를 요약해 주세요.' : 'Run ofgpu-validate and summarise the passed and failed gates.', synthetic: true }
    case 'export': {
      const dir = latestResultDir(q.runs, caseRef) ?? 'cases/plume_jsonc'
      const run = latestRun(q.runs, caseRef)
      const solver = run ? getBinary(run.binary)?.name : null
      return { text: ko ? `${dir}의 마지막 시간 스텝에서 U와 p의 통계(field_stats)를 계산하고, 뷰어 스크린샷을 찍어 결과 요약을 만들어 주세요.${solver ? ` (솔버: ${solver})` : ''}` : `Compute field statistics (field_stats) for U and p at the last time step of ${dir} and take a viewer screenshot for a result summary.${solver ? ` (solver: ${solver})` : ''}`, synthetic: true }
    }
  }
}
