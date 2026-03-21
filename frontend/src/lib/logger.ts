type LogLevel = 'debug' | 'info' | 'warn' | 'error' | 'silent'

interface Logger {
  debug: (message: string, context?: Record<string, unknown>) => void
  info: (message: string, context?: Record<string, unknown>) => void
  warn: (message: string, context?: Record<string, unknown>) => void
  error: (message: string, context?: Record<string, unknown>) => void
}

const LOG_LEVELS: Record<LogLevel, number> = {
  debug: 0,
  info: 1,
  warn: 2,
  error: 3,
  silent: 4,
}

function getLogLevel(): LogLevel {
  if (import.meta.env.PROD) return 'warn'
  return (import.meta.env.VITE_LOG_LEVEL as LogLevel) || 'debug'
}

function sanitizeSecrets(value: unknown): unknown {
  if (typeof value === 'string') {
    return value
      .replace(/sk-[a-zA-Z0-9]{20,}/g, '[REDACTED]')
      .replace(/key[=:]\s*["']?[a-zA-Z0-9-_]{20,}["']?/gi, 'key=[REDACTED]')
      .replace(/token[=:]\s*["']?[a-zA-Z0-9-_.]{20,}["']?/gi, 'token=[REDACTED]')
  }
  if (typeof value === 'object' && value !== null) {
    const sanitized: Record<string, unknown> = {}
    for (const [k, v] of Object.entries(value)) {
      if (/key|token|secret|password|credential/i.test(k)) {
        sanitized[k] = '[REDACTED]'
      } else {
        sanitized[k] = sanitizeSecrets(v)
      }
    }
    return sanitized
  }
  return value
}

export function createLogger(namespace: string): Logger {
  const currentLevel = LOG_LEVELS[getLogLevel()]

  const log = (level: LogLevel, message: string, context?: Record<string, unknown>) => {
    if (LOG_LEVELS[level] < currentLevel) return

    const prefix = `[${namespace}]`
    const sanitizedContext = context ? sanitizeSecrets(context) : undefined

    switch (level) {
      case 'debug':
        console.debug(prefix, message, sanitizedContext ?? '')
        break
      case 'info':
        console.info(prefix, message, sanitizedContext ?? '')
        break
      case 'warn':
        console.warn(prefix, message, sanitizedContext ?? '')
        break
      case 'error':
        console.error(prefix, message, sanitizedContext ?? '')
        break
    }
  }

  return {
    debug: (message, context) => log('debug', message, context),
    info: (message, context) => log('info', message, context),
    warn: (message, context) => log('warn', message, context),
    error: (message, context) => log('error', message, context),
  }
}

export const log = {
  workflow: createLogger('Workflow'),
  websocket: createLogger('WebSocket'),
  nodes: createLogger('Nodes'),
  executor: createLogger('Executor'),
}
