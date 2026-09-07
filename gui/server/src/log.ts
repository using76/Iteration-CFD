// Minimal levelled logger. One prefix, one sink (stderr), no dependencies.
export type LogLevel = 'debug' | 'info' | 'warn' | 'error'

export interface Logger {
  level: LogLevel
  debug(msg: string, ...rest: unknown[]): void
  info(msg: string, ...rest: unknown[]): void
  warn(msg: string, ...rest: unknown[]): void
  error(msg: string, ...rest: unknown[]): void
  child(prefix: string): Logger
}

const ORDER: Record<LogLevel, number> = { debug: 10, info: 20, warn: 30, error: 40 }

export function createLogger(level: LogLevel = 'info', prefix = 'cfd-server'): Logger {
  const emit = (lvl: LogLevel, msg: string, rest: unknown[]) => {
    if (ORDER[lvl] < ORDER[logger.level]) return
    const line = `[${prefix}] ${lvl === 'info' ? '' : `${lvl}: `}${msg}`
    if (lvl === 'error' || lvl === 'warn') console.error(line, ...rest)
    else console.log(line, ...rest)
  }
  const logger: Logger = {
    level,
    debug: (m, ...r) => emit('debug', m, r),
    info: (m, ...r) => emit('info', m, r),
    warn: (m, ...r) => emit('warn', m, r),
    error: (m, ...r) => emit('error', m, r),
    child: (p) => {
      const c = createLogger(logger.level, `${prefix}:${p}`)
      return c
    },
  }
  return logger
}

export const silentLogger: Logger = {
  level: 'error',
  debug: () => {},
  info: () => {},
  warn: () => {},
  error: () => {},
  child: () => silentLogger,
}
