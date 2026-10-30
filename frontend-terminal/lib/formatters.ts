export const UNAVAILABLE = "—"

export const finite = (value: unknown): value is number | string =>
  value !== null && value !== undefined && value !== "" && Number.isFinite(Number(value))

export const clamp = (value: number, low: number, high: number) => Math.max(low, Math.min(high, value))

export const pct = (value: unknown, digits = 1) =>
  finite(value) ? (Number(value) * 100).toFixed(digits) + "%" : UNAVAILABLE

export const signedPct = (value: unknown, digits = 1) =>
  finite(value) ? (Number(value) > 0 ? "+" : "") + pct(value, digits) : UNAVAILABLE

export const decimal = (value: unknown, digits = 2) => (finite(value) ? Number(value).toFixed(digits) : UNAVAILABLE)

export const integer = (value: unknown) => (finite(value) ? String(Math.round(Number(value))) : UNAVAILABLE)

export function parseUtc(value?: string | null) {
  if (!value) return new Date(Number.NaN)
  let text = String(value)
  if (/^\d{4}-\d{2}-\d{2}$/.test(text)) text += "T00:00:00Z"
  else if (/^\d{4}-\d{2}-\d{2}T/.test(text) && !/(Z|[+-]\d{2}:?\d{2})$/i.test(text)) text += "Z"
  return new Date(text)
}

const longFormat = new Intl.DateTimeFormat("en-US", { year: "numeric", month: "short", day: "2-digit", timeZone: "UTC" })
const shortFormat = new Intl.DateTimeFormat("en-US", { month: "short", day: "2-digit", timeZone: "UTC" })
const monthFormat = new Intl.DateTimeFormat("en-US", { year: "2-digit", month: "short", timeZone: "UTC" })

function formatWith(format: Intl.DateTimeFormat, value?: string | null) {
  if (!value) return UNAVAILABLE
  const date = parseUtc(value)
  if (Number.isNaN(date.getTime())) return String(value).slice(0, 10)
  return format.format(date)
}

export const dateLabel = (value?: string | null) => formatWith(longFormat, value)
export const shortDate = (value?: string | null) => formatWith(shortFormat, value)
export const monthLabel = (value?: string | null) => formatWith(monthFormat, value)

export const modeLabel = (mode?: string) => (mode || "development").replace(/_/g, " ")
