#!/usr/bin/env node
// Talk to the MCP server the way a client does, and say what came back.
//
// This is the check the README tells people to run when a client says
// "ofgpu: failed to connect" and gives no reason. It exercises the protocol
// (initialize, tools/list, a read-only call, an unknown method, a refused
// path) rather than the solvers, so it passes on a machine with no GPU -- if
// it fails here, the problem is the server or the config, not CUDA.
import { spawn } from 'node:child_process'
import path from 'node:path'
import process from 'node:process'
import readline from 'node:readline'

const here = path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1'))
const server = path.join(here, 'server.mjs')
const workspace = process.env.OFGPU_WORKSPACE ?? path.resolve(here, '..')

const child = spawn(process.execPath, [server], {
  cwd: workspace,
  env: { ...process.env, OFGPU_WORKSPACE: workspace },
  stdio: ['pipe', 'pipe', 'inherit'],
})

const pending = new Map()
let nextId = 1
readline.createInterface({ input: child.stdout }).on('line', (line) => {
  if (!line.trim()) return
  let msg
  try {
    msg = JSON.parse(line)
  } catch {
    console.log('  ! not JSON:', line.slice(0, 120))
    return
  }
  const resolve = pending.get(msg.id)
  if (resolve) {
    pending.delete(msg.id)
    resolve(msg)
  }
})

function call(method, params) {
  const id = nextId++
  return new Promise((resolve, reject) => {
    pending.set(id, resolve)
    child.stdin.write(JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n')
    setTimeout(() => {
      if (pending.delete(id)) reject(new Error(`${method} timed out`))
    }, 60000)
  })
}

let failures = 0
function check(label, ok, detail = '') {
  console.log(`${ok ? 'ok  ' : 'FAIL'}  ${label}${detail ? ' — ' + detail : ''}`)
  if (!ok) failures++
}

const init = await call('initialize', { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 'smoke', version: '0' } })
check('initialize', init.result?.serverInfo?.name === 'ofgpu', init.result?.serverInfo?.name ?? JSON.stringify(init.error))
child.stdin.write(JSON.stringify({ jsonrpc: '2.0', method: 'notifications/initialized' }) + '\n')

const list = await call('tools/list', {})
const names = (list.result?.tools ?? []).map((t) => t.name)
check('tools/list', names.length === 6, names.join(', '))
check('every tool has a schema', (list.result?.tools ?? []).every((t) => t.inputSchema?.type === 'object'))

const cases = await call('tools/call', { name: 'ofgpu_list_cases', arguments: {} })
check('ofgpu_list_cases', cases.result?.isError === false, (cases.result?.content?.[0]?.text ?? '').split('\n')[0])

// A path outside the workspace must come back as a refusal, not a read.
const escape = await call('tools/call', { name: 'ofgpu_read_case_file', arguments: { path: '../../../etc/hosts' } })
check('refuses a path outside the workspace', escape.result?.isError === true, escape.result?.content?.[0]?.text?.slice(0, 80))

const missing = await call('tools/call', { name: 'nope', arguments: {} })
check('unknown tool is a JSON-RPC error', missing.error?.code === -32602)

const bad = await call('does/not/exist', {})
check('unknown method is method-not-found', bad.error?.code === -32601)

const pong = await call('ping', {})
check('ping', pong.result !== undefined)

child.stdin.end()
console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed')
process.exit(failures ? 1 : 0)
