#!/usr/bin/env bash
# 하위 코딩 에이전트 = z.ai GLM-5.3-Flash.
#
# 왜 별도 프로세스인가: ANTHROPIC_BASE_URL 은 Claude Code 프로세스 전역이라
# 한 세션 안에서 "상위 Opus(Anthropic) + 하위 GLM(z.ai)" 으로 못 가른다.
# 서브에이전트는 메인 세션과 다른 엔드포인트를 쓸 수 없다. 그래서 헤드리스
# `claude -p` 를 자식 프로세스로 띄우고 그 프로세스만 z.ai 에 물린다.
# 상위 세션은 Opus 그대로 유지된다.
#
# 티어 매핑: --model sonnet -> ANTHROPIC_DEFAULT_SONNET_MODEL -> glm-5.3-flash
#
# 사용법:
#   tools/glm-code.sh "작업 지시"
#   tools/glm-code.sh -f task.md          # 파일에서 지시를 읽는다
#   GLM_TIMEOUT=1800 tools/glm-code.sh "..."
#
# API 키는 이 파일에 두지 않는다. 저장소는 이 스크립트를 추적하므로 키가
# 여기 있으면 그대로 커밋된다. 아래 순서로 찾는다:
#   1) $ZAI_API_KEY
#   2) ~/.claude/zai-key   (권장, 저장소 밖)

set -euo pipefail

BASE_URL="${ZAI_BASE_URL:-https://api.z.ai/api/anthropic}"
MODEL_ID="${GLM_MODEL:-glm-5.3-flash}"
TIMEOUT="${GLM_TIMEOUT:-3600}"

key="${ZAI_API_KEY:-}"
if [ -z "$key" ] && [ -f "$HOME/.claude/zai-key" ]; then
  key="$(tr -d '\r\n' < "$HOME/.claude/zai-key")"
fi
if [ -z "$key" ]; then
  echo "glm-code: no API key. Set ZAI_API_KEY or write ~/.claude/zai-key" >&2
  exit 2
fi

# 지시문: -f 로 파일에서 읽거나, 인자 전체를 이어 붙이거나, stdin.
if [ "${1:-}" = "-f" ]; then
  [ -n "${2:-}" ] || { echo "glm-code: -f needs a file" >&2; exit 2; }
  prompt="$(cat "$2")"
elif [ "$#" -gt 0 ]; then
  prompt="$*"
else
  prompt="$(cat)"
fi

[ -n "${prompt//[[:space:]]/}" ] || { echo "glm-code: empty prompt" >&2; exit 2; }

# 자식 프로세스만 z.ai 로 향한다. 부모 셸의 환경은 건드리지 않는다.
ANTHROPIC_BASE_URL="$BASE_URL" \
ANTHROPIC_AUTH_TOKEN="$key" \
ANTHROPIC_API_KEY="$key" \
ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL_ID" \
ANTHROPIC_MODEL="$MODEL_ID" \
API_TIMEOUT_MS="$((TIMEOUT * 1000))" \
CLAUDE_CODE_MAX_CONTEXT_TOKENS="${GLM_MAX_CONTEXT_TOKENS:-256000}" \
CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
  claude -p "$prompt" \
    --model sonnet \
    ${GLM_PERMISSION_MODE:+--permission-mode "$GLM_PERMISSION_MODE"}
