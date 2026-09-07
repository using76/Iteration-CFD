// Some sandboxed Chromium builds report navigator.language as "en-US@posix",
// which makes Intl constructors throw. uPlot calls Intl.NumberFormat with it
// at module scope, so this must run before anything else is imported.
function valid(tag: string): boolean {
  try {
    new Intl.NumberFormat(tag)
    return true
  } catch {
    return false
  }
}

if (typeof navigator !== 'undefined' && !valid(navigator.language)) {
  const fallback = navigator.languages?.find(valid) ?? 'en-US'
  try {
    Object.defineProperty(Navigator.prototype, 'language', { get: () => fallback, configurable: true })
    Object.defineProperty(Navigator.prototype, 'languages', { get: () => [fallback], configurable: true })
  } catch {
    // read-only in this browser: uPlot will fail on its own
  }
}

export {}
